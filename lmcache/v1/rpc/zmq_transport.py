# SPDX-License-Identifier: Apache-2.0
"""ZMQ-based transport implementations for RPC communication.

TODO: Implement ZmqPushTransport and ZmqPullTransport for
LMCacheAsyncLookupClient/Server.
"""

# Standard
from collections import namedtuple
from functools import partial
from typing import Any, Callable, Optional, cast
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.integration.base_service_factory import ServiceResultPublisher
from lmcache.logging import init_logger
from lmcache.v1.construction import (
    CloseMethodCleanupOwner,
    drain_lookup_construction_cleanup,
    retain_lookup_construction_cleanup,
)
from lmcache.v1.rpc.transport import (
    RpcClientTransport,
    RpcServerTransport,
)
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_socket,
)

logger = init_logger(__name__)

SocketParams = namedtuple("SocketParams", ["socket_path", "rank"])


def _pyzmq_resource_closed(resource: Any) -> bool:
    """Read PyZMQ's monotonic terminal-state bit without trusting test mocks."""
    return getattr(resource, "closed", False) is True


class _SocketCollectionCleanupOwner:
    """Retain a constructor-local socket ledger with retryable progress."""

    def __init__(self, sockets: list[zmq.Socket]) -> None:
        self.sockets = sockets
        self._socket_owners = [
            CloseMethodCleanupOwner(
                socket,
                method_kwargs={"linger": 0},
                thread_name="lmcache-zmq-socket-construction-cleanup",
                # PyZMQ socket close is idempotent. The state probe handles an
                # exception raised after the underlying socket was consumed.
                retry_safe_failure=True,
                completion_probe=partial(_pyzmq_resource_closed, socket),
            )
            for socket in sockets
        ]
        self._next_index = len(self._socket_owners) - 1
        self._lock = threading.Lock()

    def cleanup(self) -> None:
        """Close sockets in reverse construction order exactly to completion."""
        with self._lock:
            while self._next_index >= 0:
                owner = self._socket_owners[self._next_index]
                owner.cleanup()
                self._next_index -= 1


class _RouterConstructionCleanupOwner:
    """Retain one private ROUTER socket/context pair through rollback."""

    def __init__(self, ctx: zmq.Context, socket: zmq.Socket | None) -> None:
        self.ctx = ctx
        self.socket = socket
        # If socket creation itself failed, its earlier receipt owns that raw
        # socket. This owner is appended after it and may terminate the context
        # only after the earlier owner drains successfully.
        self._socket_owner = (
            None
            if socket is None
            else CloseMethodCleanupOwner(
                socket,
                method_kwargs={"linger": 0},
                thread_name="lmcache-zmq-router-socket-construction-cleanup",
                # PyZMQ Socket.close() is safe to repeat while not closed.
                retry_safe_failure=True,
                completion_probe=partial(_pyzmq_resource_closed, socket),
            )
        )
        self._context_owner = CloseMethodCleanupOwner(
            ctx,
            method_name="term",
            thread_name="lmcache-zmq-router-context-construction-cleanup",
            # PyZMQ Context.term() is terminal/idempotent; ``closed`` proves
            # completion if an exception followed the libzmq transition.
            retry_safe_failure=True,
            completion_probe=partial(_pyzmq_resource_closed, ctx),
        )
        self._lock = threading.Lock()

    def cleanup(self) -> None:
        """Close the socket before its private context, retaining progress."""
        with self._lock:
            if self._socket_owner is not None:
                self._socket_owner.cleanup()
            self._context_owner.cleanup()


def _close_unpublished_sockets(
    sockets: list[zmq.Socket],
    construction_error: BaseException,
) -> None:
    """Close every socket acquired by a constructor before it can escape.

    Cleanup is deliberately exhaustive and catches ``BaseException`` because
    the construction failure itself can be an asynchronous interruption.  The
    primary construction error remains the exception that callers observe; a
    cleanup failure is retained as its explicit cause.
    """
    cleanup_owner: _SocketCollectionCleanupOwner | None = None

    def attach_and_drain() -> None:
        nonlocal cleanup_owner
        if cleanup_owner is None:
            cleanup_owner = _SocketCollectionCleanupOwner(sockets)
        retain_lookup_construction_cleanup(
            construction_error,
            cleanup_owner,
        )
        drain_lookup_construction_cleanup(construction_error)

    try:
        attach_and_drain()
    except BaseException as cleanup_error:
        if cleanup_error is construction_error:
            raise
        attach_and_drain()


def _append_unpublished_socket(
    sockets: list[zmq.Socket],
    socket: zmq.Socket,
) -> None:
    """Transfer one configured socket into its constructor-owned ledger."""
    if any(retained is socket for retained in sockets):
        return
    try:
        sockets.append(socket)
    except BaseException:
        # The append may have committed before an injected interruption.
        if not any(retained is socket for retained in sockets):
            sockets.append(socket)


def _allocate_private_zmq_context_shell() -> tuple[Optional[type[Any]], Any]:
    """Return a resource-free real Context shell and its initializer type.

    The shell's ``underlying`` handle is zero until ``Context.__init__``. Its
    return-to-caller gap is therefore safe. The caller initializes it only
    after storing the exact object locally inside its rollback region.
    """
    context_class = zmq.Context
    if isinstance(context_class, type) and issubclass(context_class, zmq.Context):
        return context_class, cast(Any, context_class).__new__(context_class)
    # Compatibility for injected test factories. Production target contexts
    # always take the preallocated path above.
    return None, context_class()


class ZmqReqRepClientTransport(RpcClientTransport):
    """ZMQ REQ socket transport for synchronous RPC clients.

    Manages multiple REQ sockets (one per server rank) and
    provides send/recv with timeout + automatic socket
    recreation on failure.
    """

    # ``close`` records each successful raw socket and PyZMQ socket close is
    # idempotent after an ambiguous exception, so a containing owner may retry.
    construction_cleanup_close_retry_safe = True

    def __init__(
        self,
        socket_params: list[SocketParams],
        timeout_ms: int,
        result_publisher: Optional[ServiceResultPublisher] = None,
    ) -> None:
        self.ctx = get_zmq_context(use_asyncio=False)
        self.socket_params = socket_params
        self.timeout_ms = timeout_ms
        self._world_size = len(socket_params)
        self.encoder = msgspec.msgpack.Encoder()
        # Socket closes can fail independently. Preserve successful progress
        # so a later lifecycle retry only touches sockets still owned by this
        # transport; the process-shared context is never its close owner.
        self._lifecycle_lock = threading.Lock()
        self._close_started = False
        self._closed_socket_indices: set[int] = set()

        # ``get_zmq_context`` returns the process-wide singleton.  A failed
        # transport construction must reclaim only the sockets it created; it
        # must not terminate a context that other transports may still use.
        sockets: list[zmq.Socket] = []
        try:
            for params in self.socket_params:
                logger.info(
                    "Transport connecting to rank %s with socket path %s",
                    params.rank,
                    params.socket_path,
                )
                socket = self._create_socket(
                    params,
                    result_publisher=partial(
                        _append_unpublished_socket,
                        sockets,
                    ),
                )
                if not any(retained is socket for retained in sockets):
                    # Compatibility for injected/legacy socket helpers that
                    # do not accept the new callee-side publisher. The real
                    # target helper has already published before returning.
                    try:
                        _append_unpublished_socket(sockets, socket)
                    except BaseException as publication_error:
                        if not any(retained is socket for retained in sockets):
                            _close_unpublished_sockets(
                                [socket],
                                publication_error,
                            )
                        raise
            # Keep final attribute publication in the same rollback region as
            # socket acquisition. A failed STORE_ATTR must not strand the
            # constructor-local ledger.
            self.sockets = sockets
            if result_publisher is not None:
                result_publisher.publish_intermediate(
                    self,
                    CloseMethodCleanupOwner(
                        self,
                        retry_safe_failure=True,
                        thread_name=(
                            "lmcache-zmq-client-transport-construction-cleanup"
                        ),
                    ),
                )
        except BaseException as construction_error:
            if result_publisher is not None and result_publisher.owns_intermediate(
                self
            ):
                # Publication transferred the complete socket ledger into a
                # transport cleanup owner before this interruption. Local
                # rollback would create a second cleanup authority beneath
                # that retained transport.
                raise
            try:
                _close_unpublished_sockets(sockets, construction_error)
            except BaseException as cleanup_error:
                if cleanup_error is construction_error:
                    raise
                _close_unpublished_sockets(sockets, construction_error)
            raise

    def _create_socket(
        self,
        params: SocketParams,
        *,
        result_publisher: Optional[Callable[[zmq.Socket], None]] = None,
    ) -> zmq.Socket:
        """Create and configure a REQ socket."""
        socket = get_zmq_socket(
            self.ctx,
            params.socket_path,
            "ipc",
            zmq.REQ,
            "connect",
            result_publisher=result_publisher,
        )
        try:
            socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        except BaseException as construction_error:
            _close_unpublished_sockets([socket], construction_error)
            raise
        return socket

    def _recreate_all_sockets_locked(self) -> None:
        """Recreate all sockets while the transport is open and locked."""
        assert not self._close_started
        for rank_idx in range(self._world_size):
            old_socket = self.sockets[rank_idx]
            if old_socket is not None:
                try:
                    old_socket.close(linger=0)
                except zmq.ZMQError as e:
                    logger.warning(
                        "ZMQ error closing old socket for rank %s: %s",
                        rank_idx,
                        e,
                    )
                except AttributeError:
                    pass

            params = self.socket_params[rank_idx]
            logger.info(
                "Recreating socket for rank %s with path %s",
                params.rank,
                params.socket_path,
            )
            self.sockets[rank_idx] = self._create_socket(params)

    def send_and_recv_all(
        self,
        msg: list[Any],
    ) -> list[bytes]:
        """Send msg to all ranks and collect responses.

        Each element of msg is serialized via msgpack before
        sending. On timeout or ZMQ error, recreates all
        sockets and returns an empty list.

        Args:
            msg: Message frames to encode and send to every rank.

        Returns:
            One response per rank, or an empty list after transport recovery.

        Raises:
            RuntimeError: Transport closure has begun, so request I/O and
                socket recreation are permanently fenced.
        """
        encoded = [self.encoder.encode(m) for m in msg]
        with self._lifecycle_lock:
            if self._close_started:
                raise RuntimeError("ZMQ client transport close has begun")

            results: list[bytes] = []
            failed_rank = -1
            try:
                for i in range(self._world_size):
                    failed_rank = i
                    self.sockets[i].send_multipart(encoded, copy=False)

                for i in range(self._world_size):
                    failed_rank = i
                    resp = self.sockets[i].recv()
                    results.append(resp)
            except zmq.Again as e:
                logger.error(
                    "Timeout occurred for rank %s, recreating all sockets. Error: %s",
                    failed_rank,
                    e,
                )
                self._recreate_all_sockets_locked()
                return []
            except zmq.ZMQError as e:
                logger.error(
                    "ZMQ error for rank %s: %s, recreating all sockets",
                    failed_rank,
                    e,
                )
                self._recreate_all_sockets_locked()
                return []

            return results

    @property
    def world_size(self) -> int:
        return self._world_size

    def close(self) -> None:
        """Close every socket owned by this transport.

        The context came from ``Context.instance()`` and is process-shared, so
        it is deliberately never terminated here. Successful per-socket
        closure is retained across calls. If a socket cannot close, the
        exception is surfaced and a later call retries only that socket.

        Raises:
            RuntimeError: One or more sockets could not be closed. The
                transport remains retryable.

        Returns:
            None.
        """
        with self._lifecycle_lock:
            # This transition is permanent even if resource closure fails.
            # Once any socket may have closed, later calls are close retries;
            # request I/O and socket replacement can no longer be made safe.
            self._close_started = True
            errors: list[tuple[str, Exception]] = []
            for index, socket in enumerate(self.sockets):
                if index in self._closed_socket_indices:
                    continue
                try:
                    socket.close(linger=0)
                except Exception as error:
                    logger.error(
                        "Error closing ZMQ client socket %d: %s",
                        index,
                        error,
                    )
                    errors.append((f"socket[{index}]", error))
                else:
                    self._closed_socket_indices.add(index)

            if errors:
                resources = ", ".join(name for name, _ in errors)
                raise RuntimeError(
                    f"failed to close ZMQ client transport resources: {resources}"
                ) from errors[0][1]


class ZmqRouterServerTransport(RpcServerTransport):
    """ZMQ ROUTER socket transport for synchronous RPC servers.

    Listens for incoming requests and sends responses back
    using ROUTER socket identity-based routing.
    """

    # ``close`` publishes socket/context progress separately and both raw
    # PyZMQ terminal operations are idempotent after an ambiguous exception.
    construction_cleanup_close_retry_safe = True

    def __init__(
        self,
        socket_path: str,
        recv_timeout_ms: int = 1000,
        result_publisher: Optional[ServiceResultPublisher] = None,
    ):
        self.decoder = msgspec.msgpack.Decoder()
        self._lifecycle_lock = threading.Lock()
        self._socket_closed = False
        self._context_closed = False
        context_class, ctx = _allocate_private_zmq_context_shell()
        socket: zmq.Socket | None = None

        def publish_socket(exact_socket: zmq.Socket) -> None:
            nonlocal socket
            try:
                socket = exact_socket
            except BaseException:
                socket = exact_socket

        try:
            if context_class is not None:
                cast(Any, context_class).__init__(ctx)
            socket = get_zmq_socket(
                ctx,
                socket_path,
                "ipc",
                zmq.ROUTER,  # type: ignore[attr-defined]
                "bind",
                result_publisher=publish_socket,
            )
            socket.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
            # These are ownership publications, not cosmetic assignments.
            # Keep them inside the rollback region so a failed attribute store
            # still closes the local socket before terminating its context.
            self.ctx = ctx
            self.socket = socket
            self.socket_path = socket_path
            if result_publisher is not None:
                result_publisher.publish_intermediate(
                    self,
                    CloseMethodCleanupOwner(
                        self,
                        retry_safe_failure=True,
                        thread_name=(
                            "lmcache-zmq-server-transport-construction-cleanup"
                        ),
                    ),
                )
        except BaseException as construction_error:
            if result_publisher is not None and result_publisher.owns_intermediate(
                self
            ):
                # The result owner now owns this exact socket/context pair.
                # Do not also drain the constructor-local pair after an
                # interruption just beyond the publication call.
                raise
            primary_error = construction_error
            cleanup_owner: _RouterConstructionCleanupOwner | None = None

            def attach_and_drain() -> None:
                nonlocal cleanup_owner
                if cleanup_owner is None:
                    cleanup_owner = _RouterConstructionCleanupOwner(ctx, socket)
                retain_lookup_construction_cleanup(
                    primary_error,
                    cleanup_owner,
                )
                drain_lookup_construction_cleanup(primary_error)

            try:
                attach_and_drain()
            except BaseException as cleanup_error:
                if cleanup_error is primary_error:
                    raise
                attach_and_drain()
            raise

    def recv_request(
        self,
    ) -> tuple[bytes, list[Any]] | None:
        """Receive a request.

        Returns (identity, data_frames) or None on timeout.
        Each data frame is deserialized via msgpack.
        ROUTER socket frames:
          [0] = identity, [1] = empty delimiter, [2:] = data
        """
        try:
            frames = self.socket.recv_multipart(copy=False)
        except zmq.Again:
            return None

        identity = frames[0].bytes
        # frames[1] is the empty delimiter from REQ socket
        raw_frames = frames[2:]
        if len(raw_frames) < 3:
            logger.warning("Malformed request received: not enough frames.")
            return None
        data_frames = [self.decoder.decode(f) for f in raw_frames]
        return (identity, data_frames)

    def send_response(
        self,
        identity: bytes,
        response: bytes,
    ) -> None:
        """Send response back via ROUTER socket."""
        self.socket.send_multipart([identity, b"", response])

    def close(self) -> None:
        """Close the bound socket and then its private context, retryably."""
        with self._lifecycle_lock:
            if not self._socket_closed:
                self.socket.close(linger=0)
                self._socket_closed = True
            if not self._context_closed:
                self.ctx.term()
                self._context_closed = True
