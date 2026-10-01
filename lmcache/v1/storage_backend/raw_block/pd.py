# SPDX-License-Identifier: Apache-2.0
"""Request-scoped completion tracking for raw-block storage handoff."""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional, Sequence
import enum
import threading
import time

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.storage_backend.raw_block.core import (
        RawBlockCore,
        RawBlockPublicationReceipt,
    )

logger = init_logger(__name__)


@dataclass
class _RequestState:
    expected_chunks: int
    keys: list[str] = field(default_factory=list)
    seen_keys: set[str] = field(default_factory=set)
    completed_keys: set[str] = field(default_factory=set)
    saw_last_batch: bool = False
    publication_started: bool = False
    terminal: Future[RawBlockPublicationReceipt] = field(default_factory=Future)


# How many finished requests to remember. A finished request keeps only
# its terminal future, so a late caller gets the answer it already had
# instead of starting the request again. The oldest are forgotten, which
# means an identifier reused after this many requests is treated as new.
_FINISHED_HISTORY = 4096

# How many holds this writer will keep at once. A hold is released only by
# an acknowledgement, so a consumer that stops acknowledging grows this
# without bound -- and deduplicated requests grow it without consuming any
# additional device slot, so neither a finite device nor a bounded request
# history bounds it. Admission stops here instead. It is sized to the
# tombstone history so that a retry of anything still admissible can still
# be distinguished from a message about a hold this writer never had.
_MAX_LIVE_LEASES = _FINISHED_HISTORY


class ReadAckOutcome(enum.Enum):
    """What a writer did about one acknowledgement.

    ``APPLIED`` is reported only after the release operation returned. The
    two negative answers are kept apart because they mean different things
    to the consumer that is still responsible for the acknowledgement:
    ``REJECTED`` says this writer will never apply this message and the
    consumer should stop offering it, while ``UNRESOLVED`` says the writer
    cannot say, so the hold stands and the obligation has not been
    discharged.
    """

    APPLIED = "APPLIED"
    ALREADY_APPLIED = "ALREADY_APPLIED"
    REJECTED = "REJECTED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class ReadAckIdentity:
    """Everything an acknowledgement must match to release a hold.

    A request identifier is not an identity: a restarted consumer reuses
    one, and a stale message naming a previous incarnation's receipt is
    exactly the case that must not reclaim a live extent. Comparing the
    whole tuple is what makes a duplicate distinguishable from a forgery.
    """

    req_id: str
    consumer_instance_id: str
    tp_rank: int
    writer_epoch: str
    checkpoint_seq: int
    manifest_digest: str


@dataclass(frozen=True)
class ReadClaimOutcome:
    """A writer's answer to a consumer asking to read one publication.

    ``final`` says the answer cannot change while the session lasts, so a
    consumer told that stops asking instead of asking once per request.
    """

    granted: bool
    reason: str = ""
    final: bool = False


@dataclass(frozen=True)
class UnreadReleaseOutcome:
    """A writer's answer to being told one publication has no reader.

    ``released`` is returned only after the release returned, so a caller
    may treat it as proof the hold is gone and may treat nothing else that
    way.
    """

    released: bool
    reason: str = ""


@dataclass(frozen=True)
class _UnreadReleaseIdentity:
    """The exact no-reader assertion a writer has already applied."""

    req_id: str
    session_id: str
    receipt: "RawBlockPublicationReceipt"


@dataclass
class _Lease:
    """Extents held for one published request until it is acknowledged.

    The receipt is kept so that an acknowledgement can be matched against
    what was actually published, rather than trusted for naming a request
    identifier that a restarted consumer could reuse.

    ``unlock_ran`` records that a release was started for these extents and
    did not report back. Running it again could decrement a reference this
    writer does not own, so the lease stays and further acknowledgements
    for it are unresolved rather than applied.

    ``claimed_by`` names the consumer incarnation that asked to read this
    publication before reading it, and is empty while nobody has. Only that
    incarnation can acknowledge it, and only a publication nobody claimed
    can be released for want of a reader.
    """

    encoded_keys: list[str]
    receipt: "RawBlockPublicationReceipt"
    unlock_ran: bool = False
    claimed_by: str = ""
    claimed_session: str = ""


class RawBlockPDRequestTracker:
    """Turn raw-block batch completions into one publication receipt."""

    def __init__(
        self,
        core: RawBlockCore,
        *,
        max_live_leases: int = _MAX_LIVE_LEASES,
    ) -> None:
        if max_live_leases <= 0:
            raise ValueError("a raw-block P/D tracker needs a live-lease bound")
        self._core = core
        self._max_live_leases = max_live_leases
        self._lock = threading.Lock()
        self._requests: dict[str, _RequestState] = {}
        self._finished: OrderedDict[str, Future[RawBlockPublicationReceipt]] = (
            OrderedDict()
        )
        # Live leases, keyed by the identity the consumer was told. Each
        # holds the keys whose extents it protects and the receipt an
        # acknowledgement has to match to release them. This is separate
        # from `_finished`, which is a bounded record that a request
        # happened and says nothing about what is still held.
        self._leases: dict[str, _Lease] = {}
        # Tombstones, keyed by request and holding the whole identity that
        # released it. A request identifier alone would answer a mismatched
        # message with "already applied", which is a false success as soon
        # as a consumer acts on the answer.
        self._released: OrderedDict[str, ReadAckIdentity] = OrderedDict()
        self._released_unread: OrderedDict[str, _UnreadReleaseIdentity] = OrderedDict()
        # Set once a tombstone is evicted. After that this writer cannot
        # distinguish a retry of something it released from a message about
        # a hold it never had, and it says so rather than guessing.
        self._forgot_released = False
        self._forgot_released_unread = False
        # Which consumer incarnation may release each session's holds.
        self._bound_consumer: dict[str, str] = {}
        # Told to consumers so they know where to reply. Set by whoever
        # built the listener; empty until then, and empty forever if there
        # is none, which a consumer can see and act on.
        self.ack_endpoint = ""
        self._publisher = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="raw-block-pd-publish",
        )
        # Requests whose publication has been handed to the publisher and
        # has not finished, whether it ran, failed or was cancelled. This is
        # what shutdown waits on: a request still in here is one whose work
        # may be inside the core right now, and closing the core under it is
        # what the wait exists to prevent. An empty put set does not say
        # this, because publication is not a put.
        self._publishing: set[str] = set()
        self._closed = False

    def register_batch(
        self,
        req_id: str,
        encoded_keys: Sequence[str],
        *,
        expected_chunks: int,
        is_last_batch: bool,
        completed_keys: Sequence[str] = (),
    ) -> Future[RawBlockPublicationReceipt]:
        """Register one request batch and return its terminal future."""
        if not req_id:
            raise ValueError("storage P/D requires a non-empty request id")
        if expected_chunks <= 0:
            raise ValueError("storage P/D requires total_chunks > 0")
        batch_keys = list(encoded_keys)
        batch_key_set = set(batch_keys)
        if not batch_keys:
            raise ValueError("storage P/D batch must contain at least one key")
        if len(batch_key_set) != len(batch_keys):
            raise ValueError("storage P/D batch contains duplicate keys")
        if not set(completed_keys).issubset(batch_key_set):
            raise ValueError("completed keys must belong to the registered batch")
        publish: tuple[str, list[str], Future[RawBlockPublicationReceipt]] | None
        with self._lock:
            if self._closed:
                raise RuntimeError("raw-block P/D request tracker is closed")
            if (
                req_id not in self._requests
                and len(self._leases) + len(self._requests) >= self._max_live_leases
            ):
                # Refusing is the point. Publishing anyway would add a hold
                # nobody is going to release, and evicting an older one to
                # make room would hand a live extent to a later request --
                # so the writer stops admitting and says why.
                raise RuntimeError(
                    "raw-block P/D already holds "
                    f"{len(self._leases)} unacknowledged lease(s) and "
                    f"{len(self._requests)} in-flight request(s), at its "
                    f"bound of {self._max_live_leases}; refusing to publish "
                    f"{req_id} rather than abandon one of them"
                )
            if req_id in self._finished:
                raise RuntimeError(
                    f"request {req_id} has already finished and cannot be "
                    "registered again"
                )
            state = self._requests.get(req_id)
            if state is None:
                state = _RequestState(expected_chunks=expected_chunks)
                self._requests[req_id] = state
            elif state.expected_chunks != expected_chunks:
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(
                        f"request {req_id} changed total_chunks from "
                        f"{state.expected_chunks} to {expected_chunks}"
                    ),
                )
                return state.terminal
            elif state.terminal.done():
                raise RuntimeError(f"request {req_id} is already terminal")
            elif state.publication_started:
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(f"request {req_id} changed after publication began"),
                )
                return state.terminal
            elif state.saw_last_batch:
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(
                        f"request {req_id} added a batch after its last batch"
                    ),
                )
                return state.terminal

            duplicate_keys = batch_key_set & state.seen_keys
            if duplicate_keys:
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(
                        f"request {req_id} repeated keys across batches: "
                        + ", ".join(sorted(duplicate_keys))
                    ),
                )
                return state.terminal

            state.seen_keys.update(batch_keys)
            state.keys.extend(batch_keys)
            state.completed_keys.update(completed_keys)
            state.saw_last_batch = state.saw_last_batch or is_last_batch
            if len(state.seen_keys) > state.expected_chunks:
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(
                        f"request {req_id} supplied {len(state.seen_keys)} keys "
                        f"for total_chunks={state.expected_chunks}"
                    ),
                )
                return state.terminal
            if state.saw_last_batch and len(state.seen_keys) != state.expected_chunks:
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(
                        f"request {req_id} marked its last batch with "
                        f"{len(state.seen_keys)} of {state.expected_chunks} keys"
                    ),
                )
                return state.terminal
            publish = self._maybe_start_publication_locked(req_id, state)
            terminal = state.terminal
        if publish is not None:
            self._submit_publication(*publish)
        return terminal

    def has_request(self, req_id: str) -> bool:
        """Return whether an unfinished request has registered any batch."""
        with self._lock:
            state = self._requests.get(req_id)
            return state is not None and not state.terminal.done()

    def finalize_request(
        self,
        req_id: str,
        *,
        expected_chunks: int,
    ) -> Future[RawBlockPublicationReceipt]:
        """Mark an existing request complete when its final batch has no new keys."""
        publish: tuple[str, list[str], Future[RawBlockPublicationReceipt]] | None
        with self._lock:
            remembered = self._finished.get(req_id)
            if remembered is not None:
                return remembered
            state = self._requests.get(req_id)
            if state is None:
                raise RuntimeError(f"request {req_id} has no registered batches")
            if state.expected_chunks != expected_chunks:
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(
                        f"request {req_id} changed total_chunks from "
                        f"{state.expected_chunks} to {expected_chunks}"
                    ),
                )
                return state.terminal
            if state.terminal.done() or state.publication_started:
                return state.terminal
            state.saw_last_batch = True
            if len(state.seen_keys) != state.expected_chunks:
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(
                        f"request {req_id} finalized with {len(state.seen_keys)} "
                        f"of {state.expected_chunks} keys"
                    ),
                )
                return state.terminal
            publish = self._maybe_start_publication_locked(req_id, state)
            terminal = state.terminal
        if publish is not None:
            self._submit_publication(*publish)
        return terminal

    def complete_batch(
        self,
        req_id: str,
        encoded_keys: Sequence[str],
    ) -> None:
        """Record successful writes for one batch."""
        publish: tuple[str, list[str], Future[RawBlockPublicationReceipt]] | None
        with self._lock:
            state = self._requests.get(req_id)
            if state is None or state.terminal.done():
                return
            unknown = set(encoded_keys) - state.seen_keys
            if unknown:
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(
                        f"request {req_id} completed unregistered keys: "
                        + ", ".join(sorted(unknown))
                    ),
                )
                return
            state.completed_keys.update(encoded_keys)
            publish = self._maybe_start_publication_locked(req_id, state)
        if publish is not None:
            self._submit_publication(*publish)

    def fail_request(self, req_id: str, error: BaseException) -> None:
        """Fail a request once and prevent publication."""
        with self._lock:
            state = self._requests.get(req_id)
            if state is not None:
                self._fail_locked(req_id, state, error)

    def finish_request(self, req_id: str) -> None:
        """Fail a request that ended before its final batch was registered."""
        with self._lock:
            state = self._requests.get(req_id)
            if (
                state is not None
                and not state.terminal.done()
                and not state.saw_last_batch
            ):
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(
                        f"request {req_id} finished before its last storage batch"
                    ),
                )

    def close(self, timeout_s: float = 5.0) -> bool:
        """Fail unfinished requests, stop publication, and keep every lease.

        Returns whether publication was confirmed finished. A publication
        reaches the core, so a caller about to close that core needs to know
        the difference between "nothing is running" and "the wait gave up".

        This writer going away is not news about a reader. A lease is what
        keeps an extent from being handed to the next writer, and the only
        thing that establishes a consumer is finished with one is that
        consumer's own acknowledgement -- or, for a publication nobody was
        ever given, the terminal release in :meth:`release_unread`. A local
        shutdown is neither, so it releases nothing and says what it kept.

        Releasing a whole group's holds after the whole group has stopped is
        a separate decision with a separate authority, and lives in
        :meth:`release_quiesced_leases`.
        """
        with self._lock:
            if self._closed:
                return not self._publishing
            self._closed = True
            # Failing a request retires it, which removes it from this
            # table, so take the entries before walking them.
            for req_id, state in list(self._requests.items()):
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(f"request {req_id} aborted during shutdown"),
                )
        # Cancel what has not started: a queued publication would pin keys
        # and write an index through a core its caller has already finished
        # accounting for. Then wait, on a budget, for whatever is already
        # inside the core to leave it -- without holding the lock that work
        # needs, which is why this runs after the block above and not in it.
        self._publisher.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                running = len(self._publishing)
            if running == 0 or time.monotonic() >= deadline:
                break
            time.sleep(0.005)
        if running:
            logger.error(
                "RawBlockPDTracker did not confirm %d publication(s) had "
                "finished within %.1fs; they may still be inside the core",
                running,
                timeout_s,
            )
        with self._lock:
            held = len(self._leases)
            unresolved = sum(1 for lease in self._leases.values() if lease.unlock_ran)
        if held:
            logger.warning(
                "RawBlockPDTracker retaining %d lease(s) at shutdown, %d of "
                "them with a release that did not report back: nothing here "
                "establishes that a consumer has stopped reading those "
                "extents.",
                held,
                unresolved,
            )
        return running == 0

    def release_quiesced_leases(self) -> int:
        """Release the holds of a group that has stopped, and count them.

        This is the only authority besides a consumer's acknowledgement and
        the unread-publication release that frees a hold, and it is an
        operator's assertion rather than something observed here: every
        engine that could be reading this namespace has stopped. Nothing in
        this process can see that, which is why it is not what
        :meth:`close` does.

        A lease whose release was started and did not report back is kept
        even so. Running it again could decrement a reference this writer
        does not own, and a quiesced group does not make an unknown outcome
        known.
        """
        with self._lock:
            releasable = [
                (req_id, list(lease.encoded_keys))
                for req_id, lease in self._leases.items()
                if not lease.unlock_ran
            ]
            for req_id, _ in releasable:
                self._leases[req_id].unlock_ran = True
        released = 0
        for req_id, encoded_keys in releasable:
            try:
                self._core.unlock_many(encoded_keys)
            except Exception:
                logger.exception(
                    "Raw-block P/D could not release the quiesced extents for %s",
                    req_id,
                )
                continue
            with self._lock:
                self._leases.pop(req_id, None)
            released += 1
        with self._lock:
            kept = len(self._leases)
        if kept:
            logger.error(
                "RawBlockPDTracker kept %d lease(s) through a quiesced "
                "teardown: their releases did not report back, so running "
                "them again could drop a reference this writer does not own.",
                kept,
            )
        return released

    def claim_read(
        self,
        identity: ReadAckIdentity,
        *,
        expected_writer_epoch: str,
        expected_tp_rank: int,
        session_id: str = "",
    ) -> "ReadClaimOutcome":
        """Record a consumer as the reader of one publication, or refuse.

        This is asked before any bytes move, which is the whole point. The
        acknowledgement that follows is the only thing that releases an
        extent, so the writer has to settle *before* the read which
        incarnation is going to owe it one. Settling that on whichever
        acknowledgement arrives first is a race between a consumer and the
        process that replaced it, and the loser's reads are then held for
        the writer's lifetime.

        The first claim binds the session's reader for as long as the
        session lasts. A later, different incarnation under the same session
        is a consumer that restarted while these holds were live, and it is
        refused: the extents it would read and release belong to reads the
        new process never made. An operator changes the session identifier
        when the whole group restarts, which is what makes that legitimate.
        This is not authentication; it is a fence against reuse by a process
        that cannot have done the reading.

        Returns whether this consumer may read, why not when it may not,
        and whether asking again could ever change the answer.
        """
        with self._lock:
            if self._closed:
                return ReadClaimOutcome(False, "this writer is shutting down")
            refusal = self._identity_refusal_locked(
                identity,
                expected_writer_epoch=expected_writer_epoch,
                expected_tp_rank=expected_tp_rank,
            )
            if refusal is not None:
                return ReadClaimOutcome(False, refusal)
            lease = self._leases.get(identity.req_id)
            if lease is None:
                # Nothing is held under this name, so there is nothing to
                # grant the reading of. Granting anyway would let a consumer
                # read extents this writer has already released or never
                # published, and acknowledge them afterwards.
                return ReadClaimOutcome(False, "this writer holds no such publication")
            if not self._receipt_matches(lease.receipt, identity):
                return ReadClaimOutcome(
                    False, "this writer published a different manifest"
                )
            if lease.unlock_ran:
                # Release runs outside this lock and can fail after dropping
                # some holds. Neither an in-progress nor an unanswered
                # release leaves a publication safe for a new reader.
                return ReadClaimOutcome(False, "this publication is being released")

            bound = self._bound_consumer.get(session_id)
            if bound is None:
                self._bound_consumer[session_id] = identity.consumer_instance_id
                logger.info(
                    "Raw-block P/D bound session %s to consumer %s",
                    session_id,
                    identity.consumer_instance_id,
                )
            elif bound != identity.consumer_instance_id:
                logger.error(
                    "Raw-block P/D refused the read of %s by consumer %s: "
                    "session %s is bound to %s, and a consumer that "
                    "restarted cannot take over reads it never made",
                    identity.req_id,
                    identity.consumer_instance_id,
                    session_id,
                    bound,
                )
                return ReadClaimOutcome(
                    False,
                    "another consumer incarnation is this session's reader",
                    final=True,
                )
            if lease.claimed_by and lease.claimed_by != identity.consumer_instance_id:
                return ReadClaimOutcome(
                    False, "another consumer is already reading this publication"
                )
            lease.claimed_by = identity.consumer_instance_id
            lease.claimed_session = session_id
            return ReadClaimOutcome(True, "")

    def bound_consumer(self, session_id: str) -> Optional[str]:
        """Which consumer incarnation this session is bound to, if any."""
        with self._lock:
            return self._bound_consumer.get(session_id)

    def _identity_refusal_locked(
        self,
        identity: ReadAckIdentity,
        *,
        expected_writer_epoch: str,
        expected_tp_rank: int,
    ) -> Optional[str]:
        """Say why this message is not addressed to this writer, or nothing.

        The expected fields are the caller's own identity rather than the
        message's: comparing an arriving message against values it supplied
        itself would check nothing.
        """
        if not expected_writer_epoch:
            # An engine with no epoch published nothing, so it holds nothing
            # a consumer could read or release. Matching an empty name
            # against an empty name would accept every message that arrived
            # with the field unset.
            logger.error(
                "Raw-block P/D control message for %s reached an engine "
                "with no writer epoch",
                identity.req_id,
            )
            return "this engine published nothing"
        if identity.writer_epoch != expected_writer_epoch:
            logger.error(
                "Raw-block P/D control message for %s names producer %s, "
                "but this writer is %s",
                identity.req_id,
                identity.writer_epoch,
                expected_writer_epoch,
            )
            return "this message names another producer"
        if identity.tp_rank != expected_tp_rank:
            logger.error(
                "Raw-block P/D control message for %s claims rank %d, but "
                "this writer holds rank %d's extents",
                identity.req_id,
                identity.tp_rank,
                expected_tp_rank,
            )
            return "this message names another tensor-parallel rank"
        if not identity.consumer_instance_id:
            logger.error(
                "Raw-block P/D control message for %s names no consumer; a "
                "message nobody can be held to decides nothing",
                identity.req_id,
            )
            return "this message names no consumer"
        return None

    @staticmethod
    def _receipt_matches(
        receipt: "RawBlockPublicationReceipt",
        identity: ReadAckIdentity,
    ) -> bool:
        """Whether a message names the publication this lease protects."""
        return (
            receipt.writer_epoch == identity.writer_epoch
            and receipt.checkpoint_seq == identity.checkpoint_seq
            and receipt.manifest_digest == identity.manifest_digest
        )

    def apply_read_ack(
        self,
        identity: ReadAckIdentity,
        *,
        expected_writer_epoch: str,
        expected_tp_rank: int,
        session_id: str = "",
    ) -> ReadAckOutcome:
        """Release a hold on an acknowledgement this writer can vouch for.

        The writer validates every field against what it published itself.
        Nothing else holds the lease, so nothing else can decide that an
        extent is reclaimable -- a relay may carry an acknowledgement and
        may check it, and neither makes it true.

        ``expected_writer_epoch`` and ``expected_tp_rank`` are the caller's
        own identity. Comparing an acknowledgement against fields it
        supplied itself would check nothing, so they arrive separately from
        the message. The writer epoch is the producer incarnation: an engine
        mints one when it is built and never takes another's, so a message
        naming a different epoch is addressed to a different producer even
        when it reaches this socket.

        Only the consumer that claimed the read may acknowledge it. The
        claim happened before the bytes moved, so by the time one of these
        arrives the reader is already settled and this compares rather than
        decides.

        ``APPLIED`` is returned only after the release returned. A caller
        may treat it as proof that this hold is gone; it may treat nothing
        else that way.
        """
        with self._lock:
            if (
                self._identity_refusal_locked(
                    identity,
                    expected_writer_epoch=expected_writer_epoch,
                    expected_tp_rank=expected_tp_rank,
                )
                is not None
            ):
                return ReadAckOutcome.REJECTED

            settled = self._released.get(identity.req_id)
            lease = self._leases.get(identity.req_id)
            if lease is None:
                if settled is not None:
                    # A lost reply is safe to retry, and only for the same
                    # message. A different one naming a request this writer
                    # has already released is not a duplicate.
                    if settled == identity:
                        return ReadAckOutcome.ALREADY_APPLIED
                    logger.error(
                        "Raw-block P/D read ack for %s does not match the "
                        "acknowledgement that released it; releasing nothing",
                        identity.req_id,
                    )
                    return ReadAckOutcome.REJECTED
                if self._forgot_released:
                    # The record that would answer this was evicted. Saying
                    # "already applied" here would invent a success, and
                    # saying "rejected" would tell a consumer to stop
                    # retrying something that may still be owed.
                    logger.warning(
                        "Raw-block P/D read ack for %s names no live hold "
                        "and no remembered one; this writer cannot say",
                        identity.req_id,
                    )
                    return ReadAckOutcome.UNRESOLVED
                logger.warning(
                    "Raw-block P/D read ack for %s names no hold this "
                    "writer ever had; releasing nothing",
                    identity.req_id,
                )
                return ReadAckOutcome.REJECTED

            receipt = lease.receipt
            if not self._receipt_matches(receipt, identity):
                logger.error(
                    "Raw-block P/D read ack for %s from %s does not match "
                    "the receipt this writer published (epoch %s/%s, seq "
                    "%d/%d, digest %s/%s); releasing nothing",
                    identity.req_id,
                    identity.consumer_instance_id,
                    identity.writer_epoch,
                    receipt.writer_epoch,
                    identity.checkpoint_seq,
                    receipt.checkpoint_seq,
                    identity.manifest_digest,
                    receipt.manifest_digest,
                )
                return ReadAckOutcome.REJECTED
            if lease.unlock_ran:
                logger.error(
                    "Raw-block P/D already ran a release for %s that did "
                    "not report back; running it again could drop a "
                    "reference this writer does not own",
                    identity.req_id,
                )
                return ReadAckOutcome.UNRESOLVED

            if not lease.claimed_by:
                # Nobody asked to read this before reading it, so nobody
                # established that they had. Binding the session here
                # instead would settle who the reader is on whichever
                # message arrives first, which is the race claim_read()
                # exists to replace.
                logger.error(
                    "Raw-block P/D read ack for %s comes from consumer %s, "
                    "which never claimed the read; releasing nothing",
                    identity.req_id,
                    identity.consumer_instance_id,
                )
                return ReadAckOutcome.REJECTED
            if lease.claimed_session != session_id:
                logger.error(
                    "Raw-block P/D read ack for %s arrives under session %s, "
                    "but the read was claimed under %s; releasing nothing",
                    identity.req_id,
                    session_id,
                    lease.claimed_session,
                )
                return ReadAckOutcome.REJECTED
            if lease.claimed_by != identity.consumer_instance_id:
                logger.error(
                    "Raw-block P/D read ack for %s comes from consumer %s, "
                    "but %s claimed that read; a consumer that restarted "
                    "cannot release holds for reads it never made",
                    identity.req_id,
                    identity.consumer_instance_id,
                    lease.claimed_by,
                )
                return ReadAckOutcome.REJECTED

            # Marked before the lock is dropped, so a concurrent duplicate
            # is unresolved rather than a second release of the same keys.
            lease.unlock_ran = True
            encoded_keys = list(lease.encoded_keys)

        try:
            self._core.unlock_many(encoded_keys)
        except Exception:
            # The lease stays, still marked, so nothing releases these keys
            # again on a retry. Holding an extent nobody will reclaim is a
            # leak; releasing one twice hands a live extent to a later
            # request.
            logger.exception(
                "Raw-block P/D could not release the extents for %s",
                identity.req_id,
            )
            return ReadAckOutcome.UNRESOLVED

        with self._lock:
            self._leases.pop(identity.req_id, None)
            self._released[identity.req_id] = identity
            while len(self._released) > _FINISHED_HISTORY:
                self._released.popitem(last=False)
                self._forgot_released = True
        logger.info(
            "Raw-block P/D released %d extent(s) for request %s on an "
            "acknowledgement from %s",
            len(encoded_keys),
            identity.req_id,
            identity.consumer_instance_id,
        )
        return ReadAckOutcome.APPLIED

    def release_unread(
        self,
        req_id: str,
        receipt: "RawBlockPublicationReceipt",
        *,
        expected_writer_epoch: str,
        session_id: str = "",
        reason: str = "",
    ) -> UnreadReleaseOutcome:
        """Release a publication that was never handed to a reader.

        Some requests are answered without a decoder: the caller asked for
        one token, or the producer stopped on its own. The publication is
        real and durable, and no consumer is ever going to read it, so
        nothing would acknowledge it and the hold would stand for the life
        of this writer -- until the admission bound stopped it publishing
        at all.

        This is the terminal release for exactly that case, and it is the
        writer's: the party that knows no reader was assigned asks, and this
        writer checks the one thing that makes the request safe to honour.
        A publication some consumer claimed is refused, because a claim is a
        consumer saying it is about to read those extents. Nothing here
        fabricates an acknowledgement for a read that did not happen.
        """
        identity = _UnreadReleaseIdentity(req_id, session_id, receipt)
        with self._lock:
            if not expected_writer_epoch:
                return UnreadReleaseOutcome(False, "this engine published nothing")
            if receipt.writer_epoch != expected_writer_epoch:
                logger.error(
                    "Raw-block P/D was told %s has no reader, but it names "
                    "producer %s and this writer is %s; releasing nothing",
                    req_id,
                    receipt.writer_epoch,
                    expected_writer_epoch,
                )
                return UnreadReleaseOutcome(False, "this names another producer")
            lease = self._leases.get(req_id)
            if lease is None:
                if req_id in self._released:
                    return UnreadReleaseOutcome(False, "a reader already released this")
                released = self._released_unread.get(req_id)
                if released is not None:
                    if released == identity:
                        return UnreadReleaseOutcome(True, "already released")
                    return UnreadReleaseOutcome(
                        False, "a different no-reader assertion released this request"
                    )
                if self._forgot_released_unread:
                    return UnreadReleaseOutcome(
                        False, "the writer no longer remembers this release"
                    )
                return UnreadReleaseOutcome(
                    False, "this writer holds no such publication"
                )
            held = lease.receipt
            if held != receipt:
                return UnreadReleaseOutcome(
                    False, "this writer published a different manifest"
                )
            if lease.claimed_by:
                # A consumer said it was about to read these extents. Only
                # that consumer's acknowledgement releases them, whatever
                # anyone else believes about who was assigned the read.
                logger.error(
                    "Raw-block P/D was told %s has no reader, but %s claimed "
                    "the read; releasing nothing",
                    req_id,
                    lease.claimed_by,
                )
                return UnreadReleaseOutcome(False, "a consumer claimed this read")
            if lease.unlock_ran:
                return UnreadReleaseOutcome(
                    False, "a release for this did not report back"
                )
            lease.unlock_ran = True
            encoded_keys = list(lease.encoded_keys)

        try:
            self._core.unlock_many(encoded_keys)
        except Exception:
            logger.exception(
                "Raw-block P/D could not release the unread extents for %s",
                req_id,
            )
            return UnreadReleaseOutcome(False, "the release did not return")

        with self._lock:
            self._leases.pop(req_id, None)
            self._released_unread[req_id] = identity
            while len(self._released_unread) > _FINISHED_HISTORY:
                self._released_unread.popitem(last=False)
                self._forgot_released_unread = True
        logger.info(
            "Raw-block P/D released %d unread extent(s) for request %s%s",
            len(encoded_keys),
            req_id,
            f": {reason}" if reason else "",
        )
        return UnreadReleaseOutcome(True, "")

    def live_lease_count(self) -> int:
        """Count leases still protecting extents from reuse."""
        with self._lock:
            return len(self._leases)

    def report_status(self) -> dict[str, int]:
        """Summarize logical leases and the physical extents they share."""
        with self._lock:
            extent_references = [
                encoded_key
                for lease in self._leases.values()
                for encoded_key in lease.encoded_keys
            ]
            return {
                "live_lease_count": len(self._leases),
                "live_extent_reference_count": len(extent_references),
                "live_unique_extent_count": len(set(extent_references)),
                "claimed_lease_count": sum(
                    bool(lease.claimed_by) for lease in self._leases.values()
                ),
                "releasing_lease_count": sum(
                    lease.unlock_ran for lease in self._leases.values()
                ),
                "inflight_request_count": len(self._requests),
                "publication_task_count": len(self._publishing),
            }

    def _maybe_start_publication_locked(
        self,
        req_id: str,
        state: _RequestState,
    ) -> tuple[str, list[str], Future[RawBlockPublicationReceipt]] | None:
        if state.publication_started or state.terminal.done():
            return None
        if not state.saw_last_batch:
            return None
        if len(state.seen_keys) != state.expected_chunks:
            return None
        if state.completed_keys != state.seen_keys:
            return None
        state.publication_started = True
        # Account for the handoff before dropping the admission lock. Close
        # must also see a publication whose submitter has not run yet.
        self._publishing.add(req_id)
        return req_id, list(state.keys), state.terminal

    def _submit_publication(
        self,
        req_id: str,
        encoded_keys: list[str],
        terminal: Future[RawBlockPublicationReceipt],
    ) -> None:
        def pin_then_publish() -> RawBlockPublicationReceipt:
            """Hold the extents, then describe them.

            A receipt identifies a checkpoint by a digest over where these
            keys live. Taking the hold afterwards leaves an interval in
            which a key can be deleted and written again somewhere else:
            the hold would then pin the replacement while the receipt still
            describes what was there before, and a reader matching that
            receipt would be pointed at bytes nobody promised it.
            """
            leased = self._core.get_metadata_prefix(encoded_keys, lock=True)
            release_protection = getattr(
                self._core, "release_publication_protection", None
            )
            if release_protection is not None:
                release_protection(req_id)
            if len(leased) != len(encoded_keys):
                self._core.unlock_many(encoded_keys[: len(leased)])
                raise RuntimeError(
                    f"request {req_id} could not lease every published key"
                )
            try:
                return self._core.publish_request(encoded_keys)
            except BaseException:
                self._core.unlock_many(encoded_keys)
                raise

        try:
            with self._lock:
                if self._closed:
                    self._publishing.discard(req_id)
                    return
                publication = self._publisher.submit(pin_then_publish)
        except BaseException as exc:
            try:
                self.fail_request(req_id, exc)
            finally:
                with self._lock:
                    self._publishing.discard(req_id)
            return

        def settle(done: Future[RawBlockPublicationReceipt]) -> None:
            try:
                receipt = done.result()
            except BaseException as exc:
                self.fail_request(req_id, exc)
                return
            # Stamped here rather than in the core: the core publishes the
            # manifest, and where to reply about it is this tracker's
            # business because it is the thing holding the lease.
            receipt = replace(receipt, ack_endpoint=self.ack_endpoint)
            link_publication = getattr(self._core, "link_io_publication", None)
            if link_publication is not None:
                link_publication(req_id, receipt)
            with self._lock:
                state = self._requests.get(req_id)
                owned = (
                    state is not None
                    and state.terminal is terminal
                    and not terminal.done()
                )
                if owned:
                    self._leases[req_id] = _Lease(
                        encoded_keys=list(encoded_keys),
                        receipt=receipt,
                    )
                    terminal.set_result(receipt)
                    self._retire_locked(req_id, terminal)
            if not owned:
                # The request ended while this was in flight, so nothing
                # will ever release these extents by name. Let them go
                # rather than hold them for the life of the writer.
                self._core.unlock_many(encoded_keys)

        def finish(done: Future[RawBlockPublicationReceipt]) -> None:
            try:
                settle(done)
            finally:
                # Settlement may still release holds through the core.
                # Do not authorize its teardown until that work returns.
                with self._lock:
                    self._publishing.discard(req_id)

        publication.add_done_callback(finish)

    def _fail_locked(
        self,
        req_id: str,
        state: _RequestState,
        error: BaseException,
    ) -> None:
        release_protection = getattr(self._core, "release_publication_protection", None)
        if release_protection is not None:
            release_protection(req_id)
        if not state.terminal.done():
            state.terminal.set_exception(error)
        self._retire_locked(req_id, state.terminal)

    def _retire_locked(
        self,
        req_id: str,
        terminal: Future[RawBlockPublicationReceipt],
    ) -> None:
        """Drop a finished request's working state, keeping its answer.

        The batch and key sets a request accumulates are only useful while
        it is in flight, but a caller can still arrive after it ends, and
        forgetting the request entirely would let that caller start it
        again. Keep the terminal future, which is small, and bound how many
        of those are kept.
        """
        self._requests.pop(req_id, None)
        self._finished.pop(req_id, None)
        self._finished[req_id] = terminal
        while len(self._finished) > _FINISHED_HISTORY:
            self._finished.popitem(last=False)
