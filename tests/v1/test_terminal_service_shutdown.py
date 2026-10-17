# SPDX-License-Identifier: Apache-2.0
"""Terminal shutdown barriers for threads that can still use the engine."""

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import asyncio
import threading

# Third Party
import msgspec
import pytest
import zmq

# First Party
from lmcache.integration.base_service_factory import ServiceResultPublisher
from lmcache.v1 import construction as construction_module
from lmcache.v1 import rpc_utils as rpc_utils_module
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.construction import (
    LookupConstructionCleanupReceipt,
    drain_lookup_construction_cleanup,
    get_lookup_construction_cleanup_receipt,
    retain_lookup_construction_cleanup,
)
from lmcache.v1.health_monitor.base import HealthMonitor
from lmcache.v1.internal_api_server import api_server as api_server_module
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.lookup_client import factory as lookup_factory_module
from lmcache.v1.lookup_client import lmcache_async_lookup_client as async_lookup_module
from lmcache.v1.lookup_client import lmcache_lookup_client as sync_lookup_module
from lmcache.v1.lookup_client.async_lookup_message import LookupRequestMsg
from lmcache.v1.lookup_client.factory import LookupClientFactory
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
    LMCacheAsyncLookupServer,
)
from lmcache.v1.lookup_client.lmcache_lookup_client import (
    LMCacheLookupClient,
    LMCacheLookupServer,
)
from lmcache.v1.manager import _RetainedFactoryCallOwner
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.offload_server import zmq_server as offload_server_module
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    ThreadLevel,
    ThreadRunSummary,
)
from lmcache.v1.rpc import zmq_transport as zmq_transport_module
from tests.v1.instruction_monitoring import (
    InstructionInterruption,
    instruction_offset,
    method_call_offset,
    requires_instruction_monitoring,
)
import lmcache.v1.token_database as token_database_module


class _ConstructionSocket:
    """Socket probe for exception-atomic transport construction."""

    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        option_error: BaseException | None = None,
        close_errors: list[BaseException] | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.option_error = option_error
        self.close_errors = list(close_errors or ())
        self.close_calls = 0

    def setsockopt(self, _option: object, _value: object) -> None:
        self.events.append(f"set:{self.name}")
        if self.option_error is not None:
            raise self.option_error

    def close(self, linger: int = 0) -> None:
        assert linger == 0
        self.close_calls += 1
        self.events.append(f"close:{self.name}")
        if self.close_errors:
            raise self.close_errors.pop(0)


class _FactoryService:
    """Exact successor whose cleanup transitively owns its transport."""

    def __init__(self, transport: MagicMock) -> None:
        self.transport = transport
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.transport.close()


class _FactoryCleanup:
    def __init__(self, resource: Any) -> None:
        self.resource = resource

    def cleanup(self) -> None:
        self.resource.close()


class _CleanupProbe:
    def __init__(self) -> None:
        self.calls = 0

    def cleanup(self) -> None:
        self.calls += 1


def _recording_socket_class(created: list[zmq.Socket]) -> type[zmq.Socket]:
    class _RecordingSocket(zmq.Socket):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    return _RecordingSocket


@requires_instruction_monitoring
@pytest.mark.parametrize(
    ("target", "offset"),
    [
        (
            construction_module._commit_lookup_construction_cleanup,
            instruction_offset(
                construction_module._commit_lookup_construction_cleanup,
                "CALL",
                occurrence=2,
            ),
        ),
        (
            construction_module._commit_lookup_construction_cleanup,
            instruction_offset(
                construction_module._commit_lookup_construction_cleanup,
                "CALL",
                occurrence=3,
            ),
        ),
        (
            LookupConstructionCleanupReceipt.retain,
            instruction_offset(
                LookupConstructionCleanupReceipt.retain,
                "CALL",
                occurrence=3,
                after=True,
            ),
        ),
    ],
    ids=["exception-attribute", "receipt-retain", "owner-list-append"],
)
def test_cleanup_owner_publication_replays_exact_instruction_interrupt(
    target: Any,
    offset: int,
) -> None:
    """Receipt attachment retains one exact owner across every commit edge."""
    construction_error = RuntimeError("cleanup owner publication probe")
    cleanup_owner = _CleanupProbe()
    injected = KeyboardInterrupt("cleanup owner publication interruption")
    interrupt = InstructionInterruption(target, offset, injected)

    with interrupt:
        receipt = retain_lookup_construction_cleanup(
            construction_error,
            cleanup_owner,
        )
    interrupt.assert_fired_once()
    assert receipt is get_lookup_construction_cleanup_receipt(construction_error)
    assert receipt.owner_count == 1
    assert drain_lookup_construction_cleanup(construction_error)
    assert cleanup_owner.calls == 1


@requires_instruction_monitoring
def test_raw_socket_cleanup_entry_interrupt_replays_attach_and_drain() -> None:
    """A raw socket remains local while its whole rollback unit replays."""
    events: list[str] = []
    socket = _ConstructionSocket("raw-cleanup", events)
    construction_error = ValueError("ordinary socket construction failure")
    injected = KeyboardInterrupt("before raw cleanup attach entered")
    target = rpc_utils_module._close_socket_after_construction_failure
    interrupt = InstructionInterruption(
        target,
        instruction_offset(target, "CALL", occurrence=0),
        injected,
    )

    with interrupt:
        target(socket, construction_error)
    interrupt.assert_fired_once()
    assert socket.close_calls == 1
    receipt = get_lookup_construction_cleanup_receipt(construction_error)
    assert receipt is not None
    assert receipt.complete


@requires_instruction_monitoring
def test_socket_ledger_cleanup_entry_interrupt_replays_exact_owner() -> None:
    """The transport socket ledger survives its attach-helper entry gap."""
    events: list[str] = []
    socket = _ConstructionSocket("ledger-cleanup", events)
    construction_error = ValueError("ordinary transport construction failure")
    injected = KeyboardInterrupt("before ledger cleanup attach entered")
    target = zmq_transport_module._close_unpublished_sockets
    interrupt = InstructionInterruption(
        target,
        instruction_offset(target, "CALL", occurrence=0),
        injected,
    )

    with interrupt:
        target(cast(Any, [socket]), construction_error)
    interrupt.assert_fired_once()
    assert socket.close_calls == 1
    receipt = get_lookup_construction_cleanup_receipt(construction_error)
    assert receipt is not None
    assert receipt.complete


@requires_instruction_monitoring
def test_raw_zmq_socket_store_interrupt_closes_prepublished_shell(
    tmp_path,
) -> None:
    """The Context.socket return gap retains and closes the exact C socket."""
    created: list[zmq.Socket] = []
    context = zmq.Context()
    context._socket_class = _recording_socket_class(created)
    injected = KeyboardInterrupt("after Context.socket returned")
    interrupt = InstructionInterruption(
        rpc_utils_module._new_zmq_socket,
        instruction_offset(
            rpc_utils_module._new_zmq_socket,
            "STORE_FAST",
            argval="socket",
            occurrence=1,
        ),
        injected,
    )

    try:
        with interrupt, pytest.raises(KeyboardInterrupt) as failure:
            rpc_utils_module.get_zmq_socket(
                context,
                str(tmp_path / "raw-client.sock"),
                "ipc",
                zmq.REQ,
                "connect",
            )
        interrupt.assert_fired_once()
        assert failure.value is injected
        assert len(created) == 1
        assert created[0].closed
    finally:
        for socket in created:
            if not socket.closed:
                socket.close(linger=0)
        context.term()


@requires_instruction_monitoring
def test_new_zmq_socket_return_interrupt_closes_outer_published_socket(
    tmp_path,
) -> None:
    """The raw callee publishes before its configured caller stores return."""
    created: list[zmq.Socket] = []
    context = zmq.Context()
    context._socket_class = _recording_socket_class(created)
    injected = KeyboardInterrupt("after _new_zmq_socket returned")
    interrupt = InstructionInterruption(
        rpc_utils_module.get_zmq_socket,
        instruction_offset(
            rpc_utils_module.get_zmq_socket,
            "STORE_FAST",
            argval="created_socket",
        ),
        injected,
    )

    try:
        with interrupt, pytest.raises(KeyboardInterrupt) as failure:
            rpc_utils_module.get_zmq_socket(
                context,
                str(tmp_path / "outer-raw-client.sock"),
                "ipc",
                zmq.REQ,
                "connect",
            )
        interrupt.assert_fired_once()
        assert failure.value is injected
        assert len(created) == 1
        assert created[0].closed
    finally:
        for socket in created:
            if not socket.closed:
                socket.close(linger=0)
        context.term()


@requires_instruction_monitoring
def test_reqrep_socket_return_interrupt_drains_outer_ledger(
    monkeypatch,
    tmp_path,
) -> None:
    """A configured socket is ledger-owned before _create_socket returns."""
    created: list[zmq.Socket] = []
    context = zmq.Context()
    context._socket_class = _recording_socket_class(created)
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_context",
        lambda **_kwargs: context,
    )
    injected = KeyboardInterrupt("after configured socket returned")
    interrupt = InstructionInterruption(
        zmq_transport_module.ZmqReqRepClientTransport.__init__,
        instruction_offset(
            zmq_transport_module.ZmqReqRepClientTransport.__init__,
            "STORE_DEREF",
            argval="socket",
        ),
        injected,
    )

    try:
        with interrupt, pytest.raises(KeyboardInterrupt) as failure:
            zmq_transport_module.ZmqReqRepClientTransport(
                [
                    zmq_transport_module.SocketParams(
                        str(tmp_path / "ledger-client.sock"),
                        0,
                    )
                ],
                timeout_ms=10,
            )
        interrupt.assert_fired_once()
        assert failure.value is injected
        assert len(created) == 1
        assert created[0].closed
    finally:
        context.term()


@requires_instruction_monitoring
def test_reqrep_transport_return_interrupt_retains_exact_transport(
    tmp_path,
) -> None:
    """The real REQ transport publishes itself before __init__ returns."""
    thread_name = "test-reqrep-transport-return"

    def build_transport(
        publisher: ServiceResultPublisher,
    ) -> zmq_transport_module.ZmqReqRepClientTransport:
        transport = zmq_transport_module.ZmqReqRepClientTransport(
            [
                zmq_transport_module.SocketParams(
                    str(tmp_path / "transport-return-client.sock"),
                    0,
                )
            ],
            timeout_ms=10,
            result_publisher=publisher,
        )
        publisher.publish_final(transport)
        return transport

    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        build_transport,
        thread_name,
    )
    injected = KeyboardInterrupt("after REQ transport __init__ returned")
    interrupt = InstructionInterruption(
        zmq_transport_module.ZmqReqRepClientTransport.__init__,
        instruction_offset(
            zmq_transport_module.ZmqReqRepClientTransport.__init__,
            "RETURN_CONST",
        ),
        injected,
        thread_name=thread_name,
    )

    with interrupt, pytest.raises(KeyboardInterrupt) as failure:
        owner.run()
    interrupt.assert_fired_once()
    assert failure.value is injected
    assert owner.last_error is injected
    cleanup_owner = owner.intermediate_cleanup_owner
    transport = cleanup_owner.resource
    assert owner.owns_intermediate(transport)
    assert not transport.sockets[0].closed

    cleanup_owner.cleanup()
    assert transport.sockets[0].closed


@requires_instruction_monitoring
def test_reqrep_transport_publication_interrupt_has_one_cleanup_owner(
    monkeypatch,
) -> None:
    """A published transport prevents constructor-local socket cleanup."""
    events: list[str] = []
    socket = _ConstructionSocket("published-client", events)
    thread_name = "test-reqrep-transport-publication"

    def create_socket(
        _self: object,
        _params: object,
        *,
        result_publisher: Any,
    ) -> _ConstructionSocket:
        result_publisher(socket)
        return socket

    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_context",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        zmq_transport_module.ZmqReqRepClientTransport,
        "_create_socket",
        create_socket,
    )

    def build_transport(
        publisher: ServiceResultPublisher,
    ) -> zmq_transport_module.ZmqReqRepClientTransport:
        transport = zmq_transport_module.ZmqReqRepClientTransport(
            [zmq_transport_module.SocketParams("published-client", 0)],
            timeout_ms=10,
            result_publisher=publisher,
        )
        publisher.publish_final(transport)
        return transport

    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        build_transport,
        thread_name,
    )
    injected = KeyboardInterrupt("after REQ transport publication")
    interrupt = InstructionInterruption(
        zmq_transport_module.ZmqReqRepClientTransport.__init__,
        method_call_offset(
            zmq_transport_module.ZmqReqRepClientTransport.__init__,
            "publish_intermediate",
            after=True,
        ),
        injected,
        thread_name=thread_name,
    )

    with interrupt, pytest.raises(KeyboardInterrupt) as failure:
        owner.run()
    interrupt.assert_fired_once()
    assert failure.value is injected
    assert owner.last_error is injected
    assert owner.intermediate_result_published
    assert socket.close_calls == 0

    cleanup_owner = owner.intermediate_cleanup_owner
    cleanup_owner.cleanup()
    cleanup_owner.cleanup()
    assert socket.close_calls == 1


@requires_instruction_monitoring
def test_router_context_init_interrupt_terminates_exact_shell(
    monkeypatch,
    tmp_path,
) -> None:
    """The private Context is local before its C handle is initialized."""
    context = zmq.Context.__new__(zmq.Context)
    monkeypatch.setattr(
        zmq_transport_module,
        "_allocate_private_zmq_context_shell",
        lambda: (zmq.Context, context),
    )
    injected = KeyboardInterrupt("after private Context initialization")
    interrupt = InstructionInterruption(
        zmq_transport_module.ZmqRouterServerTransport.__init__,
        instruction_offset(
            zmq_transport_module.ZmqRouterServerTransport.__init__,
            "CALL",
            occurrence=4,
            after=True,
        ),
        injected,
    )

    with interrupt, pytest.raises(KeyboardInterrupt) as failure:
        zmq_transport_module.ZmqRouterServerTransport(
            str(tmp_path / "router-context.sock")
        )
    interrupt.assert_fired_once()
    assert failure.value is injected
    assert context.closed


@requires_instruction_monitoring
def test_router_transport_return_interrupt_retains_exact_transport(
    tmp_path,
) -> None:
    """The real ROUTER transport publishes itself before __init__ returns."""
    thread_name = "test-router-transport-return"

    def build_transport(
        publisher: ServiceResultPublisher,
    ) -> zmq_transport_module.ZmqRouterServerTransport:
        transport = zmq_transport_module.ZmqRouterServerTransport(
            str(tmp_path / "transport-return-server.sock"),
            result_publisher=publisher,
        )
        publisher.publish_final(transport)
        return transport

    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        build_transport,
        thread_name,
    )
    injected = KeyboardInterrupt("after ROUTER transport __init__ returned")
    interrupt = InstructionInterruption(
        zmq_transport_module.ZmqRouterServerTransport.__init__,
        instruction_offset(
            zmq_transport_module.ZmqRouterServerTransport.__init__,
            "RETURN_CONST",
        ),
        injected,
        thread_name=thread_name,
    )

    with interrupt, pytest.raises(KeyboardInterrupt) as failure:
        owner.run()
    interrupt.assert_fired_once()
    assert failure.value is injected
    cleanup_owner = owner.intermediate_cleanup_owner
    transport = cleanup_owner.resource
    assert owner.owns_intermediate(transport)
    assert not transport.socket.closed
    assert not transport.ctx.closed

    cleanup_owner.cleanup()
    assert transport.socket.closed
    assert transport.ctx.closed


@requires_instruction_monitoring
def test_router_transport_publication_interrupt_has_one_cleanup_owner(
    monkeypatch,
) -> None:
    """A published transport prevents constructor-local pair cleanup."""
    events: list[str] = []
    socket = _ConstructionSocket("published-router", events)
    context = MagicMock()
    thread_name = "test-router-transport-publication"

    def create_socket(*_args: object, **kwargs: object) -> _ConstructionSocket:
        cast(Any, kwargs["result_publisher"])(socket)
        return socket

    monkeypatch.setattr(
        zmq_transport_module,
        "_allocate_private_zmq_context_shell",
        lambda: (None, context),
    )
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_socket",
        create_socket,
    )

    def build_transport(
        publisher: ServiceResultPublisher,
    ) -> zmq_transport_module.ZmqRouterServerTransport:
        transport = zmq_transport_module.ZmqRouterServerTransport(
            "published-router",
            result_publisher=publisher,
        )
        publisher.publish_final(transport)
        return transport

    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        build_transport,
        thread_name,
    )
    injected = KeyboardInterrupt("after ROUTER transport publication")
    interrupt = InstructionInterruption(
        zmq_transport_module.ZmqRouterServerTransport.__init__,
        method_call_offset(
            zmq_transport_module.ZmqRouterServerTransport.__init__,
            "publish_intermediate",
            after=True,
        ),
        injected,
        thread_name=thread_name,
    )

    with interrupt, pytest.raises(KeyboardInterrupt) as failure:
        owner.run()
    interrupt.assert_fired_once()
    assert failure.value is injected
    assert owner.last_error is injected
    assert owner.intermediate_result_published
    assert socket.close_calls == 0
    context.term.assert_not_called()

    cleanup_owner = owner.intermediate_cleanup_owner
    cleanup_owner.cleanup()
    cleanup_owner.cleanup()
    assert socket.close_calls == 1
    context.term.assert_called_once_with()


@requires_instruction_monitoring
def test_router_socket_return_interrupt_closes_socket_then_context(
    monkeypatch,
    tmp_path,
) -> None:
    """Callee publication owns the ROUTER socket before caller STORE_DEREF."""
    events: list[str] = []
    socket = _ConstructionSocket("router", events)
    context = zmq.Context.__new__(zmq.Context)
    monkeypatch.setattr(
        zmq_transport_module,
        "_allocate_private_zmq_context_shell",
        lambda: (zmq.Context, context),
    )

    def create_socket(*_args: object, **kwargs: object) -> _ConstructionSocket:
        cast(Any, kwargs["result_publisher"])(socket)
        return socket

    monkeypatch.setattr(zmq_transport_module, "get_zmq_socket", create_socket)
    injected = KeyboardInterrupt("after ROUTER socket returned")
    interrupt = InstructionInterruption(
        zmq_transport_module.ZmqRouterServerTransport.__init__,
        instruction_offset(
            zmq_transport_module.ZmqRouterServerTransport.__init__,
            "STORE_DEREF",
            argval="socket",
            occurrence=1,
        ),
        injected,
    )

    with interrupt, pytest.raises(KeyboardInterrupt) as failure:
        zmq_transport_module.ZmqRouterServerTransport(
            str(tmp_path / "router-socket.sock")
        )
    interrupt.assert_fired_once()
    assert failure.value is injected
    assert socket.close_calls == 1
    assert context.closed


def test_sync_reqrep_constructor_unwinds_all_rank_sockets(monkeypatch) -> None:
    """A later-rank interruption closes current and earlier sockets exactly."""
    events: list[str] = []
    option_error = KeyboardInterrupt("injected second-rank option interruption")
    first_socket = _ConstructionSocket("rank-0", events)
    second_socket = _ConstructionSocket(
        "rank-1",
        events,
        option_error=option_error,
    )
    context = MagicMock()
    socket_factory = MagicMock(side_effect=[first_socket, second_socket])
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_context",
        MagicMock(return_value=context),
    )
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_socket",
        socket_factory,
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        zmq_transport_module.ZmqReqRepClientTransport(
            [
                zmq_transport_module.SocketParams("rank-0-path", 0),
                zmq_transport_module.SocketParams("rank-1-path", 1),
            ],
            timeout_ms=10,
        )

    assert failure.value is option_error
    assert events[-2:] == ["close:rank-1", "close:rank-0"]
    assert first_socket.close_calls == 1
    assert second_socket.close_calls == 1
    # The client borrowed Context.instance(); failed construction must not
    # terminate a process-wide context that other transports may still use.
    context.term.assert_not_called()


def test_sync_reqrep_append_failure_unwinds_current_and_prior_sockets(
    monkeypatch,
) -> None:
    """A failed rank-ledger transfer cannot strand the configured socket."""
    events: list[str] = []
    append_error = MemoryError("injected rank-ledger allocation failure")
    first_socket = _ConstructionSocket("rank-0", events)
    second_socket = _ConstructionSocket("rank-1", events)
    context = MagicMock()

    def append_first_then_fail(
        sockets: list[object],
        socket: object,
    ) -> None:
        if not sockets:
            sockets.append(socket)
            return
        raise append_error

    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_context",
        MagicMock(return_value=context),
    )
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_socket",
        MagicMock(side_effect=[first_socket, second_socket]),
    )
    monkeypatch.setattr(
        zmq_transport_module,
        "_append_unpublished_socket",
        append_first_then_fail,
    )

    with pytest.raises(MemoryError) as failure:
        zmq_transport_module.ZmqReqRepClientTransport(
            [
                zmq_transport_module.SocketParams("rank-0-path", 0),
                zmq_transport_module.SocketParams("rank-1-path", 1),
            ],
            timeout_ms=10,
        )

    assert failure.value is append_error
    assert events[-2:] == ["close:rank-1", "close:rank-0"]
    assert first_socket.close_calls == 1
    assert second_socket.close_calls == 1
    context.term.assert_not_called()


def test_sync_reqrep_final_publication_failure_unwinds_socket(monkeypatch) -> None:
    """The completed local socket ledger remains rollback-owned until stored."""
    events: list[str] = []
    publication_error = MemoryError("injected client attribute-store failure")
    socket = _ConstructionSocket("rank-0", events)
    context = MagicMock()

    def interrupt_sockets_store(self: object, name: str, value: object) -> None:
        if name == "sockets":
            raise publication_error
        object.__setattr__(self, name, value)

    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_context",
        MagicMock(return_value=context),
    )
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_socket",
        MagicMock(return_value=socket),
    )
    monkeypatch.setattr(
        zmq_transport_module.ZmqReqRepClientTransport,
        "__setattr__",
        interrupt_sockets_store,
    )

    with pytest.raises(MemoryError) as failure:
        zmq_transport_module.ZmqReqRepClientTransport(
            [zmq_transport_module.SocketParams("rank-0-path", 0)],
            timeout_ms=10,
        )

    assert failure.value is publication_error
    assert socket.close_calls == 1
    context.term.assert_not_called()


def test_sync_reqrep_failed_socket_close_retains_exact_retry_owner(
    monkeypatch,
) -> None:
    """A raw socket close failure remains attached to the primary error."""
    events: list[str] = []
    option_error = KeyboardInterrupt("injected second-rank option interruption")
    close_error = SystemExit("injected raw socket close interruption")
    first_socket = _ConstructionSocket("rank-0", events)
    second_socket = _ConstructionSocket(
        "rank-1",
        events,
        option_error=option_error,
        # The inner and outer constructor cleanup attempts both fail. Manager
        # recovery will later perform the successful third attempt.
        close_errors=[close_error, close_error],
    )
    context = MagicMock()
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_context",
        MagicMock(return_value=context),
    )
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_socket",
        MagicMock(side_effect=[first_socket, second_socket]),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        zmq_transport_module.ZmqReqRepClientTransport(
            [
                zmq_transport_module.SocketParams("rank-0-path", 0),
                zmq_transport_module.SocketParams("rank-1-path", 1),
            ],
            timeout_ms=10,
        )

    assert failure.value is option_error
    assert failure.value.__cause__ is close_error
    receipt = get_lookup_construction_cleanup_receipt(option_error)
    assert receipt is not None
    assert not receipt.complete
    assert first_socket.close_calls == 0
    assert second_socket.close_calls == 2

    assert drain_lookup_construction_cleanup(option_error)
    assert receipt.complete
    assert first_socket.close_calls == 1
    assert second_socket.close_calls == 3
    context.term.assert_not_called()


def test_sync_router_constructor_drains_private_context(monkeypatch) -> None:
    """A post-bind interruption closes the socket and its owned context."""
    events: list[str] = []
    option_error = KeyboardInterrupt("injected router option interruption")
    socket = _ConstructionSocket(
        "router",
        events,
        option_error=option_error,
    )
    context = MagicMock()
    context.term.side_effect = lambda: events.append("term:context")
    monkeypatch.setattr(zmq_transport_module.zmq, "Context", lambda: context)
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_socket",
        MagicMock(return_value=socket),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        zmq_transport_module.ZmqRouterServerTransport("router-path")

    assert failure.value is option_error
    assert socket.close_calls == 1
    assert events[-2:] == ["close:router", "term:context"]
    context.term.assert_called_once_with()


def test_sync_router_failed_close_retains_socket_before_private_context(
    monkeypatch,
) -> None:
    """Router rollback cannot lose or terminate past an unclosed socket."""
    events: list[str] = []
    option_error = KeyboardInterrupt("injected router option interruption")
    close_error = SystemExit("injected router socket close interruption")
    socket = _ConstructionSocket(
        "router",
        events,
        option_error=option_error,
        close_errors=[close_error],
    )
    context = MagicMock()
    context.term.side_effect = lambda: events.append("term:context")
    monkeypatch.setattr(zmq_transport_module.zmq, "Context", lambda: context)
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_socket",
        MagicMock(return_value=socket),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        zmq_transport_module.ZmqRouterServerTransport("router-path")

    assert failure.value is option_error
    assert failure.value.__cause__ is close_error
    receipt = get_lookup_construction_cleanup_receipt(option_error)
    assert receipt is not None
    assert not receipt.complete
    assert socket.close_calls == 1
    context.term.assert_not_called()

    assert drain_lookup_construction_cleanup(option_error)
    assert receipt.complete
    assert socket.close_calls == 2
    context.term.assert_called_once_with()
    assert events[-2:] == ["close:router", "term:context"]


def test_sync_router_failed_context_term_retries_only_context(monkeypatch) -> None:
    """A failed private-context term never repeats a completed socket close."""
    events: list[str] = []
    option_error = KeyboardInterrupt("injected router option interruption")
    term_error = SystemExit("injected router context term interruption")
    socket = _ConstructionSocket(
        "router",
        events,
        option_error=option_error,
    )
    context = MagicMock()
    context.term.side_effect = [term_error, None]
    monkeypatch.setattr(zmq_transport_module.zmq, "Context", lambda: context)
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_socket",
        MagicMock(return_value=socket),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        zmq_transport_module.ZmqRouterServerTransport("router-path")

    assert failure.value is option_error
    assert failure.value.__cause__ is term_error
    receipt = get_lookup_construction_cleanup_receipt(option_error)
    assert receipt is not None
    assert not receipt.complete
    assert socket.close_calls == 1
    context.term.assert_called_once_with()

    assert drain_lookup_construction_cleanup(option_error)
    assert receipt.complete
    # The retained socket-close receipt had already published success; only
    # the context operation is attempted again.
    assert socket.close_calls == 1
    assert context.term.call_count == 2


def test_sync_router_final_publication_failure_drains_private_context(
    monkeypatch,
) -> None:
    """Router locals remain owned until socket publication completes."""
    events: list[str] = []
    publication_error = MemoryError("injected router attribute-store failure")
    socket = _ConstructionSocket("router", events)
    context = MagicMock()
    context.term.side_effect = lambda: events.append("term:context")

    def interrupt_socket_store(self: object, name: str, value: object) -> None:
        if name == "socket":
            raise publication_error
        object.__setattr__(self, name, value)

    monkeypatch.setattr(zmq_transport_module.zmq, "Context", lambda: context)
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_socket",
        MagicMock(return_value=socket),
    )
    monkeypatch.setattr(
        zmq_transport_module.ZmqRouterServerTransport,
        "__setattr__",
        interrupt_socket_store,
    )

    with pytest.raises(MemoryError) as failure:
        zmq_transport_module.ZmqRouterServerTransport("router-path")

    assert failure.value is publication_error
    assert socket.close_calls == 1
    assert events[-2:] == ["close:router", "term:context"]
    context.term.assert_called_once_with()


def test_sync_router_close_reclaims_private_context_once(monkeypatch) -> None:
    """Normal router ownership includes its dedicated ZMQ context."""
    events: list[str] = []
    socket = _ConstructionSocket("router", events)
    context = MagicMock()
    context.term.side_effect = lambda: events.append("term:context")
    monkeypatch.setattr(zmq_transport_module.zmq, "Context", lambda: context)
    monkeypatch.setattr(
        zmq_transport_module,
        "get_zmq_socket",
        MagicMock(return_value=socket),
    )
    transport = zmq_transport_module.ZmqRouterServerTransport("router-path")

    transport.close()
    transport.close()

    assert socket.close_calls == 1
    assert events[-2:] == ["close:router", "term:context"]
    context.term.assert_called_once_with()


def test_sync_lookup_server_start_then_interrupt_reclaims_transport(
    monkeypatch,
) -> None:
    """A started request thread is joined before constructor rollback closes."""
    recv_entered = threading.Event()
    transport = MagicMock()

    def idle_recv() -> None:
        recv_entered.set()
        return None

    transport.recv_request.side_effect = idle_recv
    engine = MagicMock()
    engine.config.enable_blending = False
    start_error = KeyboardInterrupt("injected post-start interruption")
    real_start = threading.Thread.start
    started_threads: list[threading.Thread] = []

    def close_after_quiesce() -> None:
        assert started_threads
        assert not started_threads[0].is_alive()

    transport.close.side_effect = close_after_quiesce

    def start_then_interrupt(thread: threading.Thread) -> None:
        if thread.name != "lookup-server-thread":
            real_start(thread)
            return
        started_threads.append(thread)
        real_start(thread)
        assert recv_entered.wait(timeout=2.0)
        raise start_error

    monkeypatch.setattr(
        sync_lookup_module.threading.Thread,
        "start",
        start_then_interrupt,
    )
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)

    try:
        with pytest.raises(KeyboardInterrupt) as failure:
            LMCacheLookupServer.__init__(server, engine, MagicMock(), transport)

        assert failure.value is start_error
        assert server.running is False
        assert len(started_threads) == 1
        assert not started_threads[0].is_alive()
        transport.close.assert_called_once_with()
    finally:
        server.running = False
        for thread in started_threads:
            thread.join(timeout=2.0)


def test_sync_client_factory_closes_transport_on_token_database_failure(
    monkeypatch,
) -> None:
    """The retained factory may resolve absent only after transport unwind."""
    config = MagicMock()
    config.external_lookup_client = None
    config.enable_scheduler_bypass_lookup = False
    config.enable_async_loading = False
    config.enable_blending = True
    config.hit_miss_ratio = None
    config.enable_chunk_statistics = False
    transport = MagicMock()
    construction_error = KeyboardInterrupt(
        "injected token-database construction interruption"
    )
    monkeypatch.setattr(
        LookupClientFactory,
        "_create_zmq_client_transport",
        MagicMock(return_value=transport),
    )
    monkeypatch.setattr(
        token_database_module,
        "SegmentTokenDatabase",
        MagicMock(side_effect=construction_error),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        LookupClientFactory.create_lookup_client(config, MagicMock())

    assert failure.value is construction_error
    transport.close.assert_called_once_with()


def test_sync_client_factory_retains_transport_when_cleanup_fails(
    monkeypatch,
) -> None:
    """A client-construction error keeps the exact transport retry owner."""
    config = MagicMock()
    config.external_lookup_client = None
    config.enable_scheduler_bypass_lookup = False
    config.enable_async_loading = False
    config.enable_blending = True
    config.hit_miss_ratio = None
    config.enable_chunk_statistics = False
    transport = MagicMock()
    construction_error = KeyboardInterrupt(
        "injected token-database construction interruption"
    )
    cleanup_error = SystemExit("injected transport close interruption")
    transport.close.side_effect = [cleanup_error, None]
    transport.construction_cleanup_close_retry_safe = True
    monkeypatch.setattr(
        LookupClientFactory,
        "_create_zmq_client_transport",
        MagicMock(return_value=transport),
    )
    monkeypatch.setattr(
        token_database_module,
        "SegmentTokenDatabase",
        MagicMock(side_effect=construction_error),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        LookupClientFactory.create_lookup_client(config, MagicMock())

    assert failure.value is construction_error
    assert failure.value.__cause__ is cleanup_error
    receipt = get_lookup_construction_cleanup_receipt(construction_error)
    assert receipt is not None
    assert not receipt.complete
    transport.close.assert_called_once_with()

    assert drain_lookup_construction_cleanup(construction_error)
    assert receipt.complete
    assert transport.close.call_count == 2


@requires_instruction_monitoring
def test_client_callee_return_interrupt_preserves_successor_owner(
    monkeypatch,
) -> None:
    """The factory catch cannot close a transport owned by its client."""
    config = MagicMock()
    config.external_lookup_client = None
    config.enable_scheduler_bypass_lookup = False
    config.enable_async_loading = False
    config.hit_miss_ratio = None
    config.enable_chunk_statistics = False
    metadata = MagicMock()
    transport = MagicMock()
    built: list[_FactoryService] = []

    def create_transport(
        *_args: object,
        result_publisher: ServiceResultPublisher,
        **_kwargs: object,
    ) -> MagicMock:
        result_publisher.publish_intermediate(
            transport,
            _FactoryCleanup(transport),
        )
        return transport

    def create_client(
        _config: object,
        _metadata: object,
        exact_transport: MagicMock,
        result_publisher: ServiceResultPublisher,
    ) -> _FactoryService:
        client = _FactoryService(exact_transport)
        built.append(client)
        result_publisher.replace_intermediate(
            exact_transport,
            client,
            _FactoryCleanup(client),
        )
        return client

    monkeypatch.setattr(
        LookupClientFactory,
        "_create_zmq_client_transport",
        create_transport,
    )
    monkeypatch.setattr(
        sync_lookup_module,
        "LMCacheLookupClient",
        create_client,
    )
    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        lambda publisher: LookupClientFactory.create_lookup_client(
            config,
            metadata,
            result_publisher=publisher,
        ),
        "test-client-callee-return",
    )
    injected = KeyboardInterrupt("after client callee returned")
    interrupt = InstructionInterruption(
        create_client,
        instruction_offset(create_client, "RETURN_VALUE"),
        injected,
        thread_name="test-client-callee-return",
    )

    with interrupt, pytest.raises(KeyboardInterrupt) as failure:
        owner.run()
    interrupt.assert_fired_once()
    assert failure.value is injected
    assert owner.last_error is injected
    assert len(built) == 1
    client = built[0]
    assert owner.owns_intermediate(client)
    transport.close.assert_not_called()

    owner.intermediate_cleanup_owner.cleanup()
    assert client.close_calls == 1
    transport.close.assert_called_once_with()


@requires_instruction_monitoring
def test_server_callee_return_interrupt_preserves_successor_owner(
    monkeypatch,
) -> None:
    """The factory catch cannot close beneath a published server owner."""
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig

    config = LMCacheEngineConfig.from_defaults()
    engine = MagicMock()
    engine.config = config
    metadata = MagicMock()
    metadata.use_mla = False
    metadata.world_size = 1
    metadata.worker_id = 0
    transport = MagicMock()
    built: list[_FactoryService] = []

    def create_transport(
        *_args: object,
        result_publisher: ServiceResultPublisher,
        **_kwargs: object,
    ) -> MagicMock:
        result_publisher.publish_intermediate(
            transport,
            _FactoryCleanup(transport),
        )
        return transport

    def create_server(
        _engine: object,
        _metadata: object,
        exact_transport: MagicMock,
        result_publisher: ServiceResultPublisher,
    ) -> _FactoryService:
        server = _FactoryService(exact_transport)
        built.append(server)
        result_publisher.replace_intermediate(
            exact_transport,
            server,
            _FactoryCleanup(server),
        )
        return server

    monkeypatch.setattr(
        LookupClientFactory,
        "_create_zmq_server_transport",
        create_transport,
    )
    monkeypatch.setattr(
        sync_lookup_module,
        "LMCacheLookupServer",
        create_server,
    )
    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        lambda publisher: LookupClientFactory.create_lookup_server(
            engine,
            metadata,
            result_publisher=publisher,
        ),
        "test-server-callee-return",
    )
    injected = KeyboardInterrupt("after server callee returned")
    interrupt = InstructionInterruption(
        create_server,
        instruction_offset(create_server, "RETURN_VALUE"),
        injected,
        thread_name="test-server-callee-return",
    )

    with interrupt, pytest.raises(KeyboardInterrupt) as failure:
        owner.run()
    interrupt.assert_fired_once()
    assert failure.value is injected
    assert owner.last_error is injected
    assert len(built) == 1
    server = built[0]
    assert owner.owns_intermediate(server)
    transport.close.assert_not_called()

    owner.intermediate_cleanup_owner.cleanup()
    assert server.close_calls == 1
    transport.close.assert_called_once_with()


@requires_instruction_monitoring
def test_real_lookup_client_return_interrupt_retains_exact_client() -> None:
    """The real client transfers transport ownership before __init__ returns."""
    config = LMCacheEngineConfig.from_defaults()
    metadata = MagicMock(spec=LMCacheMetadata)
    transport = MagicMock()
    thread_name = "test-real-lookup-client-return"

    def build_client(
        publisher: ServiceResultPublisher,
    ) -> LMCacheLookupClient:
        publisher.publish_intermediate(
            transport,
            _FactoryCleanup(transport),
        )
        client = LMCacheLookupClient(
            config,
            metadata,
            transport,
            result_publisher=publisher,
        )
        publisher.publish_final(client)
        return client

    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        build_client,
        thread_name,
    )
    injected = KeyboardInterrupt("after lookup client __init__ returned")
    interrupt = InstructionInterruption(
        LMCacheLookupClient.__init__,
        instruction_offset(LMCacheLookupClient.__init__, "RETURN_CONST"),
        injected,
        thread_name=thread_name,
    )

    with interrupt, pytest.raises(KeyboardInterrupt) as failure:
        owner.run()
    interrupt.assert_fired_once()
    assert failure.value is injected
    cleanup_owner = owner.intermediate_cleanup_owner
    client = cleanup_owner.resource
    assert isinstance(client, LMCacheLookupClient)
    assert owner.owns_intermediate(client)
    transport.close.assert_not_called()

    cleanup_owner.cleanup()
    transport.close.assert_called_once_with()


@requires_instruction_monitoring
def test_real_lookup_server_return_interrupt_retains_exact_server() -> None:
    """The real server is thread-aware before its __init__ can return."""
    config = LMCacheEngineConfig.from_defaults()
    engine = MagicMock()
    engine.config = config
    metadata = MagicMock(spec=LMCacheMetadata)
    transport = MagicMock()
    transport.recv_request.side_effect = lambda: None
    thread_name = "test-real-lookup-server-return"

    def build_server(
        publisher: ServiceResultPublisher,
    ) -> LMCacheLookupServer:
        publisher.publish_intermediate(
            transport,
            _FactoryCleanup(transport),
        )
        server = LMCacheLookupServer(
            engine,
            metadata,
            transport,
            result_publisher=publisher,
        )
        publisher.publish_final(server)
        return server

    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        build_server,
        thread_name,
    )
    injected = KeyboardInterrupt("after lookup server __init__ returned")
    interrupt = InstructionInterruption(
        LMCacheLookupServer.__init__,
        instruction_offset(LMCacheLookupServer.__init__, "RETURN_CONST"),
        injected,
        thread_name=thread_name,
    )

    with interrupt, pytest.raises(KeyboardInterrupt) as failure:
        owner.run()
    interrupt.assert_fired_once()
    assert failure.value is injected
    cleanup_owner = owner.intermediate_cleanup_owner
    server = cleanup_owner.server
    assert isinstance(server, LMCacheLookupServer)
    assert owner.owns_intermediate(server)
    assert server.thread.is_alive()
    transport.close.assert_not_called()

    cleanup_owner.cleanup()
    assert not server.thread.is_alive()
    transport.close.assert_called_once_with()


def test_client_factory_first_wrapper_failure_closes_base_owner(
    monkeypatch,
) -> None:
    """A failed first wrapper returns ownership to the base client."""
    config = MagicMock()
    config.external_lookup_client = "probe://base"
    config.enable_async_loading = False
    config.hit_miss_ratio = 0.5
    config.enable_chunk_statistics = False
    metadata = MagicMock()
    base_client = MagicMock()
    construction_error = KeyboardInterrupt("injected hit-limit wrapper interruption")
    hit_limit_constructor = MagicMock(side_effect=construction_error)
    monkeypatch.setattr(
        LookupClientFactory,
        "_create_external_lookup_client",
        MagicMock(return_value=base_client),
    )
    monkeypatch.setattr(
        lookup_factory_module,
        "HitLimitLookupClient",
        hit_limit_constructor,
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        LookupClientFactory.create_lookup_client(config, metadata)

    assert failure.value is construction_error
    hit_limit_constructor.assert_called_once_with(base_client, config)
    base_client.close.assert_called_once_with()


def test_client_factory_second_wrapper_failure_closes_first_wrapper(
    monkeypatch,
) -> None:
    """The first wrapper is the exact owner when the second wrapper fails."""
    config = MagicMock()
    config.external_lookup_client = "probe://base"
    config.enable_async_loading = False
    config.hit_miss_ratio = 0.5
    config.enable_chunk_statistics = True
    metadata = MagicMock()
    base_client = MagicMock()
    hit_limit_client = MagicMock()
    construction_error = KeyboardInterrupt(
        "injected chunk-statistics wrapper interruption"
    )
    hit_limit_constructor = MagicMock(return_value=hit_limit_client)
    chunk_statistics_constructor = MagicMock(side_effect=construction_error)
    monkeypatch.setattr(
        LookupClientFactory,
        "_create_external_lookup_client",
        MagicMock(return_value=base_client),
    )
    monkeypatch.setattr(
        lookup_factory_module,
        "HitLimitLookupClient",
        hit_limit_constructor,
    )
    monkeypatch.setattr(
        lookup_factory_module,
        "ChunkStatisticsLookupClient",
        chunk_statistics_constructor,
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        LookupClientFactory.create_lookup_client(config, metadata)

    assert failure.value is construction_error
    hit_limit_constructor.assert_called_once_with(base_client, config)
    chunk_statistics_constructor.assert_called_once_with(
        hit_limit_client,
        config,
        metadata,
    )
    hit_limit_client.close.assert_called_once_with()
    base_client.close.assert_not_called()


def test_client_factory_wrapper_failure_chains_close_error(monkeypatch) -> None:
    """Wrapper construction stays primary when current-owner close fails."""
    config = MagicMock()
    config.external_lookup_client = "probe://base"
    config.enable_async_loading = False
    config.hit_miss_ratio = 0.5
    config.enable_chunk_statistics = False
    base_client = MagicMock()
    construction_error = KeyboardInterrupt("injected wrapper construction interruption")
    cleanup_error = SystemExit("injected current-owner close interruption")
    base_client.close.side_effect = cleanup_error
    monkeypatch.setattr(
        LookupClientFactory,
        "_create_external_lookup_client",
        MagicMock(return_value=base_client),
    )
    monkeypatch.setattr(
        lookup_factory_module,
        "HitLimitLookupClient",
        MagicMock(side_effect=construction_error),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        LookupClientFactory.create_lookup_client(config, MagicMock())

    assert failure.value is construction_error
    assert failure.value.__cause__ is cleanup_error
    base_client.close.assert_called_once_with()
    receipt = get_lookup_construction_cleanup_receipt(construction_error)
    assert receipt is not None
    assert not receipt.complete


def test_sync_server_factory_closes_transport_on_early_server_failure(
    monkeypatch,
) -> None:
    """An error before thread startup still returns transport ownership."""
    # Local import avoids adding a heavyweight production dependency to the
    # transport-only probes above.
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig

    config = LMCacheEngineConfig.from_defaults()
    engine = MagicMock()
    engine.config = config
    metadata = SimpleNamespace(use_mla=False, world_size=1, worker_id=0)
    transport = MagicMock()
    construction_error = KeyboardInterrupt(
        "injected pre-thread lookup-server interruption"
    )
    monkeypatch.setattr(
        LookupClientFactory,
        "_create_zmq_server_transport",
        MagicMock(return_value=transport),
    )
    monkeypatch.setattr(
        sync_lookup_module,
        "OrderedDict",
        MagicMock(side_effect=construction_error),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        LookupClientFactory.create_lookup_server(
            engine,
            cast(LMCacheMetadata, metadata),
        )

    assert failure.value is construction_error
    transport.close.assert_called_once_with()


def test_sync_server_final_thread_publication_retains_nonquiesced_owner(
    monkeypatch,
) -> None:
    """Factory cannot close beneath a live unpublished request thread."""
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig

    config = LMCacheEngineConfig.from_defaults()
    engine = MagicMock()
    engine.config = config
    metadata = SimpleNamespace(use_mla=False, world_size=1, worker_id=0)
    transport = MagicMock()
    recv_entered = threading.Event()
    recv_release = threading.Event()
    publication_error = MemoryError("injected server thread publication failure")
    captured_threads: list[threading.Thread] = []

    def blocked_recv() -> None:
        recv_entered.set()
        assert recv_release.wait(timeout=2.0)
        return None

    def interrupt_thread_store(self: object, name: str, value: object) -> None:
        if name == "thread":
            assert isinstance(value, threading.Thread)
            captured_threads.append(value)
            raise publication_error
        object.__setattr__(self, name, value)

    transport.recv_request.side_effect = blocked_recv
    monkeypatch.setattr(
        LookupClientFactory,
        "_create_zmq_server_transport",
        MagicMock(return_value=transport),
    )
    monkeypatch.setattr(
        sync_lookup_module,
        "_LOOKUP_SERVER_CLOSE_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        sync_lookup_module.LMCacheLookupServer,
        "__setattr__",
        interrupt_thread_store,
    )

    try:
        with pytest.raises(MemoryError) as failure:
            LookupClientFactory.create_lookup_server(
                engine,
                cast(LMCacheMetadata, metadata),
            )

        assert failure.value is publication_error
        assert isinstance(failure.value.__cause__, RuntimeError)
        assert recv_entered.is_set()
        assert len(captured_threads) == 1
        assert captured_threads[0].is_alive()
        # The partial server owns this transport until its request thread is
        # proven quiescent; the factory must not race a close underneath it.
        transport.close.assert_not_called()
        receipt = get_lookup_construction_cleanup_receipt(publication_error)
        assert receipt is not None
        assert not receipt.complete

        recv_release.set()
        assert drain_lookup_construction_cleanup(publication_error)
        assert receipt.complete
        captured_threads[0].join(timeout=2.0)
        assert not captured_threads[0].is_alive()
        transport.close.assert_called_once_with()
    finally:
        recv_release.set()
        for thread in captured_threads:
            thread.join(timeout=2.0)


def test_offload_close_rejects_live_inflight_engine_store(monkeypatch) -> None:
    store_entered = threading.Event()
    store_release = threading.Event()
    engine = MagicMock()

    def blocked_store(**_kwargs) -> None:
        store_entered.set()
        assert store_release.wait(timeout=2.0)

    engine.store.side_effect = blocked_store
    server = ZMQOffloadServer.__new__(ZMQOffloadServer)
    server.lmcache_engine = engine
    server.running = True
    server.socket = MagicMock()
    server.thread = threading.Thread(
        target=lambda: server.offload([1], [0], [1]),
        daemon=True,
    )
    server.thread.start()
    assert store_entered.wait(timeout=2.0)
    monkeypatch.setattr(
        offload_server_module,
        "_OFFLOAD_SERVER_CLOSE_TIMEOUT_SECONDS",
        0.05,
    )

    with pytest.raises(RuntimeError, match="offload server thread"):
        server.close()

    assert server.thread.is_alive()
    store_release.set()
    server.thread.join(timeout=2.0)
    assert not server.thread.is_alive()
    server.close()


def test_async_lookup_server_close_wakes_idle_receiver(monkeypatch) -> None:
    poll_entered = threading.Event()
    server = LMCacheAsyncLookupServer.__new__(LMCacheAsyncLookupServer)
    server.lmcache_engine = MagicMock()
    server.running = True
    server._closed = False
    server._close_lock = threading.Lock()
    server._callback_condition = threading.Condition()
    server._active_response_callbacks = 0
    server._closing = False
    server.push_socket = MagicMock()
    server.pull_socket = MagicMock()
    server.ctx = MagicMock()

    def idle_poll(timeout: int) -> bool:
        poll_entered.set()
        return False

    server.pull_socket.poll.side_effect = idle_poll
    server.thread = threading.Thread(
        target=server.process_requests_from_scheduler,
        daemon=True,
    )
    monkeypatch.setattr(async_lookup_module, "_ASYNC_LOOKUP_POLL_TIMEOUT_MS", 1)
    monkeypatch.setattr(
        async_lookup_module,
        "_ASYNC_LOOKUP_CLOSE_TIMEOUT_SECONDS",
        0.2,
    )

    server.thread.start()
    assert poll_entered.wait(timeout=2.0)
    server.close()
    server.close()

    assert not server.thread.is_alive()
    server.pull_socket.close.assert_called_once_with(linger=0)
    server.push_socket.close.assert_called_once_with(linger=0)
    server.ctx.term.assert_called_once_with()


def test_async_lookup_server_first_response_and_close_from_constructor(
    monkeypatch,
) -> None:
    ctx = MagicMock()
    push_socket = MagicMock()
    pull_socket = MagicMock()
    pull_socket.poll.return_value = False
    metadata = SimpleNamespace(
        kv_connector_extra_config={},
        engine_id="constructor-lifecycle",
        worker_id=0,
    )
    monkeypatch.setattr(async_lookup_module.zmq, "Context", lambda: ctx)
    monkeypatch.setattr(
        async_lookup_module,
        "get_zmq_socket",
        MagicMock(side_effect=[push_socket, pull_socket]),
    )
    monkeypatch.setattr(async_lookup_module, "_ASYNC_LOOKUP_POLL_TIMEOUT_MS", 1)
    monkeypatch.setattr(
        async_lookup_module,
        "_ASYNC_LOOKUP_CLOSE_TIMEOUT_SECONDS",
        0.2,
    )

    server = LMCacheAsyncLookupServer(
        MagicMock(),
        cast(LMCacheMetadata, metadata),
    )
    server.send_response_to_scheduler("first", 1)
    server.close()
    server.close()

    push_socket.send.assert_called_once()
    pull_socket.close.assert_called_once_with(linger=0)
    push_socket.close.assert_called_once_with(linger=0)
    ctx.term.assert_called_once_with()


def test_async_lookup_client_close_wakes_idle_receiver(monkeypatch) -> None:
    poll_entered = threading.Event()
    client = LMCacheAsyncLookupClient.__new__(LMCacheAsyncLookupClient)
    client.running = True
    client._closed = False
    client._close_lock = threading.Lock()
    client.push_sockets = [MagicMock(), MagicMock()]
    client.pull_socket = MagicMock()
    client.ctx = MagicMock()

    def idle_poll(timeout: int) -> bool:
        poll_entered.set()
        return False

    client.pull_socket.poll.side_effect = idle_poll
    client.thread = threading.Thread(
        target=client.process_responses_from_workers,
        daemon=True,
    )
    monkeypatch.setattr(async_lookup_module, "_ASYNC_LOOKUP_POLL_TIMEOUT_MS", 1)
    monkeypatch.setattr(
        async_lookup_module,
        "_ASYNC_LOOKUP_CLOSE_TIMEOUT_SECONDS",
        0.2,
    )

    client.thread.start()
    assert poll_entered.wait(timeout=2.0)
    client.close()

    assert not client.thread.is_alive()
    for push_socket in client.push_sockets:
        push_socket.close.assert_called_once_with(linger=0)
    client.pull_socket.close.assert_called_once_with(linger=0)
    client.ctx.term.assert_called_once_with()


def test_async_lookup_server_close_retains_transport_during_engine_call(
    monkeypatch,
) -> None:
    lookup_entered = threading.Event()
    lookup_release = threading.Event()
    engine = MagicMock()

    def blocked_lookup(**_kwargs) -> None:
        lookup_entered.set()
        assert lookup_release.wait(timeout=2.0)

    engine.async_lookup_and_prefetch.side_effect = blocked_lookup
    server = LMCacheAsyncLookupServer.__new__(LMCacheAsyncLookupServer)
    server.lmcache_engine = engine
    server.running = True
    server._closed = False
    server._close_lock = threading.Lock()
    server._callback_condition = threading.Condition()
    server._active_response_callbacks = 0
    server._closing = False
    server.push_socket = MagicMock()
    server.pull_socket = MagicMock()
    server.ctx = MagicMock()
    server.pull_socket.poll.return_value = True
    server.pull_socket.recv.return_value = msgspec.msgpack.encode(
        LookupRequestMsg(
            lookup_id="blocked",
            hashes=[1],
            offsets=[1],
            request_configs=None,
        )
    )
    server.thread = threading.Thread(
        target=server.process_requests_from_scheduler,
        daemon=True,
    )
    monkeypatch.setattr(
        async_lookup_module,
        "_ASYNC_LOOKUP_CLOSE_TIMEOUT_SECONDS",
        0.05,
    )

    server.thread.start()
    assert lookup_entered.wait(timeout=2.0)
    with pytest.raises(RuntimeError, match="async lookup server thread"):
        server.close()

    assert server.thread.is_alive()
    server.pull_socket.close.assert_not_called()
    server.push_socket.close.assert_not_called()
    server.ctx.term.assert_not_called()

    lookup_release.set()
    server.thread.join(timeout=2.0)
    assert not server.thread.is_alive()
    server.close()
    server.pull_socket.close.assert_called_once_with(linger=0)
    server.push_socket.close.assert_called_once_with(linger=0)
    server.ctx.term.assert_called_once_with()


def test_async_lookup_server_close_retains_transport_during_response_callback(
    monkeypatch,
) -> None:
    callback_entered = threading.Event()
    callback_release = threading.Event()
    server = LMCacheAsyncLookupServer.__new__(LMCacheAsyncLookupServer)
    server.lmcache_engine = MagicMock()
    server.running = True
    server._closed = False
    server._close_lock = threading.Lock()
    server._callback_condition = threading.Condition()
    server._active_response_callbacks = 0
    server._closing = False
    server.push_socket = MagicMock()
    server.pull_socket = MagicMock()
    server.ctx = MagicMock()

    def blocked_send(*_args, **_kwargs) -> None:
        callback_entered.set()
        assert callback_release.wait(timeout=2.0)

    server.push_socket.send.side_effect = blocked_send
    server.thread = threading.Thread(target=lambda: None, daemon=True)
    server.thread.start()
    server.thread.join(timeout=2.0)
    callback_thread = threading.Thread(
        target=lambda: server.send_response_to_scheduler("blocked", 1),
        daemon=True,
    )
    callback_thread.start()
    assert callback_entered.wait(timeout=2.0)
    monkeypatch.setattr(
        async_lookup_module,
        "_ASYNC_LOOKUP_CLOSE_TIMEOUT_SECONDS",
        0.05,
    )

    with pytest.raises(RuntimeError, match="response callback"):
        server.close()

    assert callback_thread.is_alive()
    server.pull_socket.close.assert_not_called()
    server.push_socket.close.assert_not_called()
    server.ctx.term.assert_not_called()

    callback_release.set()
    callback_thread.join(timeout=2.0)
    assert not callback_thread.is_alive()
    server.close()
    server.pull_socket.close.assert_called_once_with(linger=0)
    server.push_socket.close.assert_called_once_with(linger=0)
    server.ctx.term.assert_called_once_with()


def test_internal_api_stop_rejects_live_inflight_request(monkeypatch) -> None:
    request_entered = threading.Event()
    request_release = threading.Event()
    server = InternalAPIServer.__new__(InternalAPIServer)
    server.enable = True
    server.server_log_info = "test server"
    server.server = SimpleNamespace(should_exit=False)
    server.socket_path = None
    server.thread = None

    async def blocked_run() -> None:
        request_entered.set()
        while not request_release.is_set():
            await asyncio.sleep(0.005)

    server.run = blocked_run  # type: ignore[method-assign]
    monkeypatch.setattr(
        api_server_module,
        "_INTERNAL_API_SERVER_CLOSE_TIMEOUT_SECONDS",
        0.05,
    )

    server.start()
    assert server.thread is not None
    assert request_entered.wait(timeout=2.0)

    with pytest.raises(RuntimeError, match="internal API server thread"):
        server.stop()

    assert server.server.should_exit is True
    assert server.thread is not None and server.thread.is_alive()
    request_release.set()
    server.thread.join(timeout=2.0)
    assert not server.thread.is_alive()
    server.stop()
    assert server.thread is None


def test_health_monitor_stop_rejects_live_engine_check() -> None:
    check_entered = threading.Event()
    check_release = threading.Event()
    monitor = HealthMonitor.__new__(HealthMonitor)
    PeriodicThread.__init__(
        monitor,
        name="test-health-monitor",
        interval=60.0,
        level=ThreadLevel.CRITICAL,
    )

    def blocked_check() -> ThreadRunSummary:
        check_entered.set()
        assert check_release.wait(timeout=2.0)
        return ThreadRunSummary(success=True)

    monitor._execute = blocked_check  # type: ignore[method-assign]
    thread = PeriodicThread.start(monitor)
    assert thread is not None
    assert check_entered.wait(timeout=2.0)

    with pytest.raises(RuntimeError, match="health monitor thread"):
        monitor.stop(timeout=0.05)

    assert thread.is_alive()
    check_release.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    monitor.stop(timeout=0.05)
