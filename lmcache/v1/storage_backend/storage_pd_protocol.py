# SPDX-License-Identifier: Apache-2.0
"""Wire messages for request-scoped shared-storage P/D handoff."""

# Future
from __future__ import annotations

# Standard
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Literal, NamedTuple
import os
import threading
import time
import uuid

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.raw_block.core import RawBlockPublicationReceipt

logger = init_logger(__name__)

StoragePDState = Literal["READY", "FAILED", "CANCELLED"]

# This process's identity as a *consumer* of a handoff. A PID alone is not
# one: the operating system reuses it, so an acknowledgement from a previous
# occupant of this PID would name a lease the current occupant holds. The
# random half makes each start distinguishable from every other.
#
# A producer is not named by this. Its identity is the writer epoch of the
# engine that published, which is what the extents are keyed by and what the
# writer compares an arriving acknowledgement against. A process identity
# would be coarser than the thing being released -- two engines in one
# process share it -- and keeping a second producer name in play is how the
# two ends came to disagree about what a producer is.
STORAGE_PD_INCARNATION = f"pid:{os.getpid()}:{uuid.uuid4().hex[:12]}"


class StoragePDStatus(msgspec.Struct, tag=True):
    """Terminal producer status for one request and tensor-parallel rank."""

    req_id: str
    tp_rank: int
    state: StoragePDState
    # The producer incarnation that holds the extents this status names, and
    # the publication generation they belong to: one field, because they are
    # one thing. A writer mints this when it is constructed and never adopts
    # another engine's, so an acknowledgement naming it names this writer and
    # no other. Empty on a status that published nothing.
    writer_epoch: str = ""
    namespace_identity: str = ""
    checkpoint_seq: int = 0
    key_count: int = 0
    manifest_digest: str = ""
    total_logical_bytes: int = 0
    total_padded_bytes: int = 0
    error_stage: str = ""
    error_text: str = ""
    # Where this producer listens for read acknowledgements. Empty means it
    # is not listening, in which case nothing can release its leases and it
    # says so rather than letting a consumer assume otherwise.
    ack_endpoint: str = ""

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
            tp_rank=tp_rank,
            state="READY",
            writer_epoch=receipt.writer_epoch,
            namespace_identity=receipt.namespace_identity,
            checkpoint_seq=receipt.checkpoint_seq,
            key_count=receipt.key_count,
            manifest_digest=receipt.manifest_digest,
            total_logical_bytes=receipt.total_logical_bytes,
            total_padded_bytes=receipt.total_padded_bytes,
            ack_endpoint=receipt.ack_endpoint,
        )

    def publication_receipt(self) -> RawBlockPublicationReceipt:
        """Convert a READY wire status back to a core receipt."""
        if self.state != "READY":
            raise ValueError(f"storage P/D status is not READY: {self.state}")
        return RawBlockPublicationReceipt(
            ack_endpoint=self.ack_endpoint,
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
    consumer_instance_id: str
    tp_rank: int
    # Which producer this is addressed to. Taken from the adopted status, so
    # a consumer never invents it, and compared by the writer against its own
    # epoch before anything is looked up.
    writer_epoch: str
    checkpoint_seq: int
    manifest_digest: str

    @classmethod
    def for_status(
        cls,
        status: "StoragePDStatus",
        *,
        consumer_instance_id: str,
    ) -> "StoragePDReadAck":
        """Build the acknowledgement an adopted READY status is owed.

        Every correlating field comes from the status the producer sent, so
        a consumer never names a producer, a generation or a manifest of its
        own invention -- and the two ends cannot drift into disagreeing about
        what identifies a producer, because only one of them chooses.
        """
        if status.state != "READY":
            raise ValueError(f"storage P/D status is not READY: {status.state}")
        if not status.writer_epoch:
            raise ValueError(
                "storage P/D READY status names no producer; nothing can "
                "acknowledge a publication whose writer is unidentified"
            )
        if not consumer_instance_id:
            raise ValueError("storage P/D acknowledgements need a consumer identity")
        return cls(
            req_id=status.req_id,
            consumer_instance_id=consumer_instance_id,
            tp_rank=status.tp_rank,
            writer_epoch=status.writer_epoch,
            checkpoint_seq=status.checkpoint_seq,
            manifest_digest=status.manifest_digest,
        )


# What travels to the proxy. The acknowledgement does not: it goes straight
# from the consumer to the writer that holds the extents, because a relay's
# word about one is not the reader's word and the writer is the only party
# that can act on it.
StoragePDMsg = StoragePDStatus


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

    # Slack over an attempt's own timeout, covering the handoff to the owner
    # thread so a send that is merely slow to start is not read as stuck.
    _HANDOFF_GRACE_S = 0.5

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
        self._socket_timeout_s = -1.0
        self._closed = False
        self._close_future: Future[None] | None = None
        self._lock = threading.Lock()

    def send(self, status: StoragePDMsg, *, timeout_s: float | None = None) -> None:
        """Hand one status to the owned socket, bounded by ``timeout_s``.

        ``timeout_s`` caps this one attempt, so a caller working against a
        deadline can spend only what it has left rather than the configured
        timeout. Returning means the local socket accepted the message; it
        says nothing about the proxy having received it.
        """
        attempt_s = self._timeout_s
        if timeout_s is not None:
            attempt_s = max(0.0, min(attempt_s, timeout_s))
        with self._lock:
            if self._closed:
                raise RuntimeError("storage P/D status sender is closed")
            future = self._executor.submit(self._send, status, attempt_s)
        try:
            future.result(timeout=attempt_s + self._HANDOFF_GRACE_S)
        except FutureTimeoutError as exc:
            raise LMCacheTimeoutError(
                f"storage P/D status send timed out for {self._proxy_url}"
            ) from exc

    def close(self, timeout_s: float | None = None) -> bool:
        """Close the socket on its owner thread and stop the worker.

        ``timeout_s`` bounds the wait. Without it this blocks for as long as
        a stuck socket takes, which is why the notification queue owns this
        call and passes what is left of its own shutdown budget.

        Returns whether the socket's owner thread confirmed it had finished.
        Returning inside the budget is not that confirmation: a stuck socket
        makes this abandon the worker, which is then still alive and still
        holding the socket. A caller that treats the return as proof of
        termination is making a claim about a thread nobody observed
        stopping -- and on this path that thread is what reaches the
        transport.
        """
        with self._lock:
            if self._close_future is None:
                self._closed = True
                self._close_future = self._executor.submit(self._close_socket)
            close_future = self._close_future
        try:
            close_future.result(timeout=timeout_s)
        except FutureTimeoutError:
            logger.warning(
                "storage P/D status socket for %s did not close within its "
                "shutdown budget; abandoning its worker",
                self._proxy_url,
            )
            # Cleanup remains queued behind an unfinished send. Cancelling it
            # would leave the socket open even after its worker can progress.
            self._executor.shutdown(wait=False, cancel_futures=False)
            return False
        self._executor.shutdown(wait=True, cancel_futures=False)
        return True

    def _send(self, status: StoragePDMsg, attempt_s: float) -> None:
        if self._socket is None:
            context = get_zmq_context(use_asyncio=False)
            self._socket = get_zmq_socket(
                context,
                self._proxy_url,
                "tcp",
                zmq.PUSH,
                "connect",
            )
            self._socket_timeout_s = -1.0
        if attempt_s != self._socket_timeout_s:
            self._socket.setsockopt(zmq.SNDTIMEO, int(attempt_s * 1000))
            self._socket_timeout_s = attempt_s
        self._socket.send(msgspec.msgpack.encode(status))

    def _close_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=1000)
            self._socket = None


class StoragePDDelivery(NamedTuple):
    """The outcome of trying to hand one message to the proxy.

    ``LOCALLY_SENT`` means this process's socket accepted the message. It is
    deliberately not called "delivered": it does not establish that the proxy
    received READY or that a writer applied an acknowledgement. Those are
    remote events, observed separately -- the proxy's READY barrier is what
    authorizes a decode, and a writer releases a lease only on the
    acknowledgement it validates itself.
    """

    key: str
    state: Literal["LOCALLY_SENT", "ABANDONED"]
    detail: str = ""

    # Which of the two a caller sees for an attempt that finishes after its
    # obligation expired is decided by whichever settlement ran first, and
    # both are true statements about different things: the socket did accept
    # the message, and the obligation was not announced within its
    # deadline. An attempt that wins the race reports LOCALLY_SENT even
    # though the deadline has passed, because that is what was observed.
    # Nothing here reorders them into a single story, and neither state is
    # evidence that a peer acted.


class StoragePDObligation:
    """One status owed to a consumer, and the deadline it was owed under.

    The deadline is stamped when the obligation is created -- when the status
    is first owed -- and is never refreshed. Time spent waiting for queue
    capacity is time spent against it, so a status that cannot be admitted
    does not earn a fresh lifetime by being offered again.

    A caller therefore builds one of these once, keeps it, and re-offers the
    same object. Rebuilding it per attempt is the defect this type exists to
    make impossible.

    ``settled`` and ``last_failure`` are owned by the queue and are read and
    written only under the queue's lock. ``settled`` is what fences a late
    transport result: an attempt that finishes after shutdown or after expiry
    finds its obligation already accounted for and reports nothing further.
    """

    __slots__ = ("key", "message", "report", "deadline", "settled", "last_failure")

    def __init__(
        self,
        key: str,
        message: StoragePDMsg,
        *,
        report: bool,
        deadline: float,
    ) -> None:
        self.key = key
        self.message = message
        self.report = report
        self.deadline = deadline
        self.settled = False
        self.last_failure = ""

    def expired(self, now: float) -> bool:
        return now >= self.deadline


class StoragePDNotificationQueue:
    """Announce terminal statuses without making the caller wait for a peer.

    Whether a request is READY or FAILED is settled before a message reaches
    this queue and is never revised here. What this owns is the separate
    question of whether that decision has been *announced*, which has its own
    capacity, its own retries and its own deadline. A caller offers and polls;
    it never blocks on the consumer.

    Three rules hold, and the types enforce them rather than the callers
    remembering them:

    * **One obligation.** A status owed to a consumer is one
      :class:`StoragePDObligation`, created once by the caller.
    * **One deadline.** It is stamped when the status is first owed, counts
      the time spent awaiting capacity, and is never refreshed.
    * **One terminal settlement.** Each obligation settles exactly once. A
      transport attempt that finishes late -- after expiry, or after
      shutdown already gave up on it -- changes nothing.

    Announcement is ordered and one worker owns it, so an unreachable proxy
    paces every obligation queued behind it. The bound is therefore
    ``capacity`` obligations waiting plus the one being attempted; past that,
    offers are refused while the caller still holds what it would have
    announced.

    The queue owns the sender it is given, including closing it, so that
    shutdown has one budget and one owner rather than two.
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
        self._pending: deque[StoragePDObligation] = deque()
        self._attempting: StoragePDObligation | None = None
        self._in_flight: set[str] = set()
        self._settled: list[StoragePDDelivery] = []
        self._stopping = False
        self._worker = threading.Thread(
            target=self._run,
            name="storage-pd-notify",
            daemon=True,
        )
        self._worker.start()

    def obligation(
        self,
        key: str,
        message: StoragePDMsg,
        *,
        report: bool = True,
    ) -> StoragePDObligation:
        """Stamp the deadline for a status that is owed from now.

        Call this when the status becomes owed, not when there is room for
        it. With ``report`` false the outcome is handed to ``on_unreported``
        instead of being retained for :meth:`poll`, which is for a caller
        with no polling point of its own -- one that polled nothing would
        otherwise accumulate outcomes for the life of the process.
        """
        return StoragePDObligation(
            key,
            message,
            report=report,
            deadline=time.monotonic() + self._deadline_s,
        )

    def offer(self, obligation: StoragePDObligation) -> bool:
        """Try to admit an obligation. Returns False if it was not taken.

        A refusal takes nothing over: the caller still owns the obligation,
        keeps whatever it would have announced, and may offer the *same*
        object again later. It must not build a new one, which would restart
        a deadline that has already been running.

        An obligation refused until its deadline settles here rather than
        being refused forever. The deadline belongs to the obligation, not
        to an attempt, so it has to be reachable by one that was never
        admitted at all -- otherwise a permanently saturated queue leaves
        the caller re-offering something with no end, and no record that it
        was never announced.
        """
        unreported: StoragePDDelivery | None = None
        with self._cond:
            if self._stopping or obligation.settled:
                return False
            if obligation.expired(time.monotonic()):
                unreported = self._settle_locked(
                    obligation,
                    "ABANDONED",
                    "never admitted before its deadline",
                )
            elif obligation.key in self._in_flight:
                return False
            elif len(self._pending) >= self._capacity:
                return False
            else:
                self._in_flight.add(obligation.key)
                self._pending.append(obligation)
                self._cond.notify()
                return True
        if unreported is not None:
            self._report_unreported(unreported)
        return False

    def poll(self) -> list[StoragePDDelivery]:
        """Take every outcome settled since the last call. Never blocks."""
        with self._cond:
            settled = self._settled
            self._settled = []
        return settled

    def pending_count(self) -> int:
        """Count obligations admitted and not yet settled."""
        with self._cond:
            return len(self._in_flight)

    def close(self, timeout_s: float = 5.0) -> bool:
        """Stop, within one budget shared with the sender it owns.

        The attempt in progress gets what is left of the budget to finish.
        Whatever is still owed then settles as abandoned, and is marked
        settled so that a send returning afterwards reports nothing.

        Returns whether both this queue's worker and the sender's socket
        owner confirmed they had finished. Returning at all does not: a
        stuck socket is abandoned rather than waited on, and its thread is
        then still alive and still holding the transport. A caller that
        needs to destroy something those threads touch has to read this.
        """
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        budget_ends = time.monotonic() + max(0.0, timeout_s)
        self._worker.join(timeout=max(0.0, budget_ends - time.monotonic()))
        worker_stopped = not self._worker.is_alive()
        if not worker_stopped:
            logger.error(
                "storage P/D notification worker did not stop within its "
                "shutdown budget; it may still be attempting a send"
            )
        unreported: list[StoragePDDelivery] = []
        with self._cond:
            owed = list(self._pending)
            if self._attempting is not None:
                owed.append(self._attempting)
            self._pending.clear()
            for obligation in owed:
                dropped = self._settle_locked(
                    obligation,
                    "ABANDONED",
                    "storage P/D notification queue shut down",
                )
                if dropped is not None:
                    unreported.append(dropped)
            self._in_flight.clear()
        for delivery in unreported:
            self._report_unreported(delivery)
        sender_stopped = self._sender.close(
            timeout_s=max(0.0, budget_ends - time.monotonic())
        )
        return worker_stopped and sender_stopped

    def _report_unreported(self, delivery: StoragePDDelivery) -> None:
        if self._on_unreported is not None:
            self._on_unreported(delivery)

    def _settle_locked(
        self,
        obligation: StoragePDObligation,
        state: Literal["LOCALLY_SENT", "ABANDONED"],
        detail: str = "",
    ) -> StoragePDDelivery | None:
        """Settle one obligation once, returning it if nobody will poll it.

        A second call for the same obligation does nothing at all: whichever
        outcome arrived first is the one that stands, so a transport attempt
        completing after shutdown cannot contradict the abandonment that was
        already reported for it.
        """
        if obligation.settled:
            return None
        obligation.settled = True
        self._in_flight.discard(obligation.key)
        delivery = StoragePDDelivery(key=obligation.key, state=state, detail=detail)
        if obligation.report:
            self._settled.append(delivery)
            return None
        return delivery

    def _run(self) -> None:
        while True:
            unreported: StoragePDDelivery | None = None
            attempt: StoragePDObligation | None = None
            budget = 0.0
            with self._cond:
                while not self._pending and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    # Leave what is owed queued: close() settles it, and
                    # popping it here would lose it.
                    return
                obligation = self._pending.popleft()
                budget = obligation.deadline - time.monotonic()
                if budget <= 0:
                    # Expired while it waited for its turn. Do not attempt it
                    # at all: a send that succeeded now would report an
                    # announcement for an obligation whose lifetime is over.
                    # The clock is what ends it; an earlier attempt's error,
                    # if there was one, is what an operator needs to read.
                    unreported = self._settle_locked(
                        obligation,
                        "ABANDONED",
                        obligation.last_failure
                        or "deadline passed while awaiting an attempt",
                    )
                else:
                    self._attempting = obligation
                    attempt = obligation
            if attempt is None:
                if unreported is not None:
                    self._report_unreported(unreported)
                continue
            failure = ""
            try:
                self._sender.send(attempt.message, timeout_s=budget)
            except BaseException as exc:  # noqa: BLE001 - reported, not raised
                failure = f"{type(exc).__name__}: {exc}"
            with self._cond:
                self._attempting = None
                if attempt.settled:
                    # Shutdown, or expiry, already accounted for this one.
                    continue
                if not failure:
                    unreported = self._settle_locked(attempt, "LOCALLY_SENT")
                elif attempt.expired(time.monotonic()):
                    unreported = self._settle_locked(attempt, "ABANDONED", failure)
                else:
                    attempt.last_failure = failure
                    self._pending.append(attempt)
                    if not self._stopping and self._retry_interval_s:
                        self._cond.wait(self._retry_interval_s)
            if unreported is not None:
                self._report_unreported(unreported)
