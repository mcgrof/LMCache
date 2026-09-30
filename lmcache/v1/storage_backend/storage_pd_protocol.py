# SPDX-License-Identifier: Apache-2.0
"""Wire messages for request-scoped shared-storage P/D handoff."""

# Future
from __future__ import annotations

# Standard
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Literal, NamedTuple
import threading
import time

# Third Party
import msgspec
import zmq

# First Party
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.raw_block.core import RawBlockPublicationReceipt

StoragePDState = Literal["READY", "FAILED", "CANCELLED"]


class StoragePDStatus(msgspec.Struct, tag=True):
    """Terminal producer status for one request and tensor-parallel rank."""

    req_id: str
    producer_instance_id: str
    tp_rank: int
    state: StoragePDState
    writer_epoch: str = ""
    namespace_identity: str = ""
    checkpoint_seq: int = 0
    key_count: int = 0
    manifest_digest: str = ""
    total_logical_bytes: int = 0
    total_padded_bytes: int = 0
    error_stage: str = ""
    error_text: str = ""

    @classmethod
    def ready(
        cls,
        req_id: str,
        tp_rank: int,
        receipt: RawBlockPublicationReceipt,
    ) -> "StoragePDStatus":
        """Build a READY status from a durable publication receipt."""
        return cls(
            req_id=req_id,
            producer_instance_id=receipt.writer_epoch,
            tp_rank=tp_rank,
            state="READY",
            writer_epoch=receipt.writer_epoch,
            namespace_identity=receipt.namespace_identity,
            checkpoint_seq=receipt.checkpoint_seq,
            key_count=receipt.key_count,
            manifest_digest=receipt.manifest_digest,
            total_logical_bytes=receipt.total_logical_bytes,
            total_padded_bytes=receipt.total_padded_bytes,
        )

    def publication_receipt(self) -> RawBlockPublicationReceipt:
        """Convert a READY wire status back to a core receipt."""
        if self.state != "READY":
            raise ValueError(f"storage P/D status is not READY: {self.state}")
        return RawBlockPublicationReceipt(
            writer_epoch=self.writer_epoch,
            checkpoint_seq=self.checkpoint_seq,
            key_count=self.key_count,
            manifest_digest=self.manifest_digest,
            namespace_identity=self.namespace_identity,
            total_logical_bytes=self.total_logical_bytes,
            total_padded_bytes=self.total_padded_bytes,
        )


class StoragePDReadAck(msgspec.Struct, tag=True):
    """Consumer acknowledgement after the advertised KV reached GPU memory."""

    req_id: str
    producer_instance_id: str
    consumer_instance_id: str
    tp_rank: int
    writer_epoch: str
    checkpoint_seq: int
    manifest_digest: str


StoragePDMsg = StoragePDStatus | StoragePDReadAck


def order_storage_pd_ready_statuses(
    statuses: Mapping[int, StoragePDStatus],
    num_tp_ranks: int,
) -> list[StoragePDStatus]:
    """Validate and order one READY status for every expected TP rank.

    Args:
        statuses: Statuses keyed by their claimed tensor-parallel rank.
        num_tp_ranks: Expected tensor-parallel world size.

    Returns:
        Statuses ordered by rank from zero through ``num_tp_ranks - 1``.

    Raises:
        ValueError: If the rank set or a status identity is invalid.
    """
    if num_tp_ranks <= 0:
        raise ValueError("storage P/D requires at least one TP rank")
    expected_ranks = set(range(num_tp_ranks))
    actual_ranks = set(statuses)
    if actual_ranks != expected_ranks:
        raise ValueError(
            "storage P/D READY rank set mismatch: "
            f"expected={sorted(expected_ranks)}, actual={sorted(actual_ranks)}"
        )
    ordered = [statuses[rank] for rank in range(num_tp_ranks)]
    req_ids = {status.req_id for status in ordered}
    if len(req_ids) != 1:
        raise ValueError("storage P/D READY statuses identify different requests")
    for rank, status in enumerate(ordered):
        if status.state != "READY" or status.tp_rank != rank:
            raise ValueError(
                f"storage P/D rank {rank} has invalid {status.state} status "
                f"claiming rank {status.tp_rank}"
            )
    return ordered


class StoragePDStatusSender:
    """Own a PUSH socket on one worker thread and send terminal statuses."""

    def __init__(
        self,
        proxy_host: str,
        proxy_port: int,
        *,
        timeout_s: float = 5.0,
    ) -> None:
        if not proxy_host:
            raise ValueError("storage P/D requires pd_proxy_host")
        if proxy_port <= 0:
            raise ValueError("storage P/D requires a positive pd_proxy_port")
        if timeout_s <= 0:
            raise ValueError("storage P/D status timeout must be positive")
        self._proxy_url = f"{proxy_host}:{proxy_port}"
        self._timeout_s = timeout_s
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="storage-pd-status",
        )
        self._socket: zmq.Socket | None = None
        self._closed = False
        self._lock = threading.Lock()

    def send(self, status: StoragePDMsg) -> None:
        """Synchronously enqueue a status on the owned ZeroMQ socket."""
        with self._lock:
            if self._closed:
                raise RuntimeError("storage P/D status sender is closed")
            future = self._executor.submit(self._send, status)
        try:
            future.result(timeout=self._timeout_s + 1.0)
        except FutureTimeoutError as exc:
            raise LMCacheTimeoutError(
                f"storage P/D status send timed out for {self._proxy_url}"
            ) from exc

    def close(self) -> None:
        """Close the socket on its owner thread and stop the worker."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close_future = self._executor.submit(self._close_socket)
        close_future.result()
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _send(self, status: StoragePDMsg) -> None:
        if self._socket is None:
            context = get_zmq_context(use_asyncio=False)
            self._socket = get_zmq_socket(
                context,
                self._proxy_url,
                "tcp",
                zmq.PUSH,
                "connect",
            )
            self._socket.setsockopt(zmq.SNDTIMEO, int(self._timeout_s * 1000))
        self._socket.send(msgspec.msgpack.encode(status))

    def _close_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=1000)
            self._socket = None


class StoragePDDelivery(NamedTuple):
    """The outcome of trying to hand one message to the proxy."""

    key: str
    state: Literal["DELIVERED", "ABANDONED"]
    detail: str = ""


class _PendingSend(NamedTuple):
    """One accepted message and the moment it stops being worth retrying."""

    key: str
    message: StoragePDMsg
    deadline: float


class StoragePDNotificationQueue:
    """Deliver terminal statuses without making the caller wait for a peer.

    Whether a request is READY or FAILED is settled before a message reaches
    this queue and is never revised here. What this tracks is the separate
    question of whether that decision has been *delivered*, which has its own
    capacity, its own retries and its own deadline. A caller enqueues and
    polls; it never blocks on the consumer.

    Delivery is ordered and one worker owns it, so an unreachable proxy paces
    every message queued behind it. That is the case the capacity bound and
    the deadline exist for: the queue fills, further enqueues are refused
    while the caller still holds the resources they would have announced, and
    each accepted message eventually settles one way or the other rather than
    being retried forever.
    """

    def __init__(
        self,
        sender: StoragePDStatusSender,
        *,
        capacity: int = 1024,
        retry_interval_s: float = 0.5,
        deadline_s: float = 60.0,
        on_unreported: Callable[[StoragePDDelivery], None] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("storage P/D notification capacity must be positive")
        if retry_interval_s < 0:
            raise ValueError("storage P/D retry interval must not be negative")
        if deadline_s < 0:
            raise ValueError("storage P/D delivery deadline must not be negative")
        self._sender = sender
        self._capacity = capacity
        self._retry_interval_s = retry_interval_s
        self._deadline_s = deadline_s
        self._on_unreported = on_unreported
        self._cond = threading.Condition()
        self._pending: deque[_PendingSend] = deque()
        self._in_flight: set[str] = set()
        self._unreported: set[str] = set()
        self._settled: list[StoragePDDelivery] = []
        self._stopping = False
        self._worker = threading.Thread(
            target=self._run,
            name="storage-pd-notify",
            daemon=True,
        )
        self._worker.start()

    def enqueue(
        self,
        key: str,
        message: StoragePDMsg,
        *,
        report: bool = True,
    ) -> bool:
        """Accept one message for delivery under ``key``.

        Returns False when the queue is closed, is full, or already holds an
        undelivered message for ``key``. A refusal takes nothing over: the
        caller still owns whatever the message would have announced, and can
        offer it again once :meth:`poll` has freed room.

        With ``report`` false the outcome is not retained for :meth:`poll`
        and is handed to ``on_unreported`` instead. That is for a caller with
        no polling point of its own, which would otherwise leave outcomes
        accumulating here for the life of the process.
        """
        with self._cond:
            if self._stopping:
                return False
            if key in self._in_flight:
                return False
            if len(self._pending) >= self._capacity:
                return False
            self._in_flight.add(key)
            if not report:
                self._unreported.add(key)
            self._pending.append(
                _PendingSend(
                    key=key,
                    message=message,
                    deadline=time.monotonic() + self._deadline_s,
                )
            )
            self._cond.notify()
        return True

    def poll(self) -> list[StoragePDDelivery]:
        """Take every delivery settled since the last call. Never blocks."""
        with self._cond:
            settled = self._settled
            self._settled = []
        return settled

    def pending_count(self) -> int:
        """Count messages accepted and not yet settled."""
        with self._cond:
            return len(self._in_flight)

    def close(self, timeout_s: float = 5.0) -> None:
        """Stop accepting, let the attempt in progress finish, settle the rest.

        Everything still accepted settles as abandoned, so a caller polling
        after shutdown gets a definite answer for every message it handed
        over instead of having some simply disappear.
        """
        with self._cond:
            if self._stopping:
                return
            self._stopping = True
            self._cond.notify_all()
        self._worker.join(timeout=timeout_s)
        unreported: list[StoragePDDelivery] = []
        with self._cond:
            self._pending.clear()
            for key in sorted(self._in_flight):
                dropped = self._settle_locked(
                    StoragePDDelivery(
                        key=key,
                        state="ABANDONED",
                        detail="storage P/D notification queue shut down",
                    )
                )
                if dropped is not None:
                    unreported.append(dropped)
            self._in_flight.clear()
        for delivery in unreported:
            if self._on_unreported is not None:
                self._on_unreported(delivery)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    # Leave the item queued: close() settles whatever is
                    # still accepted, and popping it here would lose it.
                    return
                item = self._pending.popleft()
            failure = ""
            try:
                self._sender.send(item.message)
            except BaseException as exc:  # noqa: BLE001 - reported, not raised
                failure = f"{type(exc).__name__}: {exc}"
            unreported: StoragePDDelivery | None = None
            with self._cond:
                if not failure:
                    unreported = self._settle_locked(
                        StoragePDDelivery(key=item.key, state="DELIVERED")
                    )
                elif time.monotonic() >= item.deadline:
                    unreported = self._settle_locked(
                        StoragePDDelivery(
                            key=item.key,
                            state="ABANDONED",
                            detail=failure,
                        )
                    )
                else:
                    self._pending.append(item)
                    if not self._stopping and self._retry_interval_s:
                        self._cond.wait(self._retry_interval_s)
            if unreported is not None and self._on_unreported is not None:
                self._on_unreported(unreported)

    def _settle_locked(self, delivery: StoragePDDelivery) -> StoragePDDelivery | None:
        """Retire one key, returning the delivery only if nobody polls it."""
        self._in_flight.discard(delivery.key)
        if delivery.key in self._unreported:
            self._unreported.discard(delivery.key)
            return delivery
        self._settled.append(delivery)
        return None
