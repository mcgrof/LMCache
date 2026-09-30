# SPDX-License-Identifier: Apache-2.0
"""Request-scoped completion tracking for raw-block storage handoff."""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Sequence
import threading

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


@dataclass
class _Lease:
    """Extents held for one published request until it is acknowledged.

    The receipt is kept so that an acknowledgement can be matched against
    what was actually published, rather than trusted for naming a request
    identifier that a restarted consumer could reuse.
    """

    encoded_keys: list[str]
    receipt: "RawBlockPublicationReceipt"


class RawBlockPDRequestTracker:
    """Turn raw-block batch completions into one publication receipt."""

    def __init__(self, core: RawBlockCore) -> None:
        self._core = core
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
        self._released: OrderedDict[str, None] = OrderedDict()
        # Told to consumers so they know where to reply. Set by whoever
        # built the listener; empty until then, and empty forever if there
        # is none, which a consumer can see and act on.
        self.ack_endpoint = ""
        self._publisher = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="raw-block-pd-publish",
        )
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

    def close(self, release_leases: bool = True) -> None:
        """Fail unfinished requests, stop publication, and release leases.

        ``release_leases`` false keeps every lease held. Unlocking a key lets
        its entry be deleted and its extent returned to the free list, so a
        caller that cannot establish what the device is still doing passes
        false: the leases are what keep those extents from being handed to
        the next writer, and shutdown is not proof that a reader has stopped
        reading them either.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Failing a request retires it, which removes it from this
            # table, so take the entries before walking them.
            for req_id, state in list(self._requests.items()):
                self._fail_locked(
                    req_id,
                    state,
                    RuntimeError(f"request {req_id} aborted during shutdown"),
                )
        self._publisher.shutdown(wait=True, cancel_futures=False)
        if not release_leases:
            with self._lock:
                held = len(self._leases)
            if held:
                logger.error(
                    "RawBlockPDTracker retaining %d lease(s) at shutdown: "
                    "their extents must not be reused while what the device "
                    "is doing with them cannot be established.",
                    held,
                )
            return
        with self._lock:
            leased_keys = [lease.encoded_keys for lease in self._leases.values()]
            self._leases.clear()
        for encoded_keys in leased_keys:
            self._core.unlock_many(encoded_keys)

    def apply_read_ack(
        self,
        req_id: str,
        consumer_instance_id: str,
        tp_rank: int,
        writer_epoch: str,
        checkpoint_seq: int,
        manifest_digest: str,
        expected_tp_rank: int,
    ) -> str:
        """Release a lease on an acknowledgement this writer can vouch for.

        The writer validates every field itself. A proxy may relay an
        acknowledgement and may check it, but it does not hold the lease and
        cannot decide that an extent is reclaimable.

        ``expected_tp_rank`` is the caller's own rank: the identity belongs
        to whoever owns this tracker, and comparing an acknowledgement
        against a rank it supplied itself would check nothing.

        Returns one of ``"released"``, ``"already_released"`` or
        ``"rejected"``. ``"already_released"`` exists because a lost
        confirmation is safe to retry: a consumer that never heard back
        sends the same acknowledgement again, and it must free nothing a
        second time. Any mismatch is ``"rejected"`` and releases nothing --
        a stale acknowledgement naming a previous incarnation's receipt is
        exactly the case that must not reclaim a live extent.
        """
        with self._lock:
            lease = self._leases.get(req_id)
            if lease is None:
                if req_id in self._released:
                    return "already_released"
                logger.warning(
                    "Raw-block P/D read ack for %s from %s names no live "
                    "lease; releasing nothing",
                    req_id,
                    consumer_instance_id,
                )
                return "rejected"
            receipt = lease.receipt
            mismatch = (
                receipt.writer_epoch != writer_epoch
                or receipt.checkpoint_seq != checkpoint_seq
                or receipt.manifest_digest != manifest_digest
            )
            if mismatch:
                logger.error(
                    "Raw-block P/D read ack for %s from %s does not match "
                    "the receipt this writer published (epoch %s/%s, seq "
                    "%d/%d, digest %s/%s); releasing nothing",
                    req_id,
                    consumer_instance_id,
                    writer_epoch,
                    receipt.writer_epoch,
                    checkpoint_seq,
                    receipt.checkpoint_seq,
                    manifest_digest,
                    receipt.manifest_digest,
                )
                return "rejected"
            if tp_rank != expected_tp_rank:
                logger.error(
                    "Raw-block P/D read ack for %s claims rank %d, but this "
                    "writer holds rank %d's extents; releasing nothing",
                    req_id,
                    tp_rank,
                    expected_tp_rank,
                )
                return "rejected"
            self._leases.pop(req_id, None)
            self._released[req_id] = None
            while len(self._released) > _FINISHED_HISTORY:
                self._released.popitem(last=False)
            encoded_keys = lease.encoded_keys

        # Outside the lock: unlocking reaches the core, and a lease is
        # removed first so a duplicate arriving now is already_released
        # rather than a second unlock.
        self._core.unlock_many(encoded_keys)
        logger.info(
            "Raw-block P/D released %d extent(s) for request %s on an "
            "acknowledgement from %s",
            len(encoded_keys),
            req_id,
            consumer_instance_id,
        )
        return "released"

    def live_lease_count(self) -> int:
        """Count leases still protecting extents from reuse."""
        with self._lock:
            return len(self._leases)

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

        publication = self._publisher.submit(pin_then_publish)

        def finish(done: Future[RawBlockPublicationReceipt]) -> None:
            try:
                receipt = done.result()
            except BaseException as exc:
                self.fail_request(req_id, exc)
                return
            # Stamped here rather than in the core: the core publishes the
            # manifest, and where to reply about it is this tracker's
            # business because it is the thing holding the lease.
            receipt = replace(receipt, ack_endpoint=self.ack_endpoint)
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

        publication.add_done_callback(finish)

    def _fail_locked(
        self,
        req_id: str,
        state: _RequestState,
        error: BaseException,
    ) -> None:
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
