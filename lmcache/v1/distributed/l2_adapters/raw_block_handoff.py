# SPDX-License-Identifier: Apache-2.0
"""Shared-storage request control for independent MP raw-block owners."""

# Standard
from concurrent.futures import Future
import math
import threading
import time

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.storage_backend.raw_block import (
    RawBlockCore,
    RawBlockDerivationDescriptor,
    RawBlockPDRequestTracker,
    RawBlockPublicationReceipt,
    ReadAckIdentity,
    ReadAckOutcome,
    ReadClaimOutcome,
    UnreadReleaseOutcome,
    encode_object_key,
)


class RawBlockPublicationTimeout(TimeoutError):
    """Prepared objects did not complete before the publication deadline."""


class RawBlockMPHandoff:
    """Prepare, publish, adopt and acknowledge bounded one-writer requests.

    The writer must prepare before submitting model work. A publication waits
    for committed objects and takes extent leases before issuing a receipt.
    Only the claimed consumer may acknowledge that receipt. An unsuccessful
    reader retains its lease; elapsed time alone never proves a read stopped.

    Args:
        core: Borrowed core, closed by the owning adapter after this controller.
        derivation: Stable key derivation configured on the borrowed core.
        max_requests: Maximum simultaneous prepared/unacknowledged requests.

    Notes:
        This is cooperative control, not authentication. There is one writer,
        and writer restart/reclaim with outstanding readers is not supported.
    """

    def __init__(
        self,
        core: RawBlockCore,
        derivation: RawBlockDerivationDescriptor,
        max_requests: int = 256,
    ) -> None:
        self.derivation = derivation
        self._core = core
        self._tracker = RawBlockPDRequestTracker(core, max_live_leases=max_requests)
        self._max_requests = max_requests
        self._requests: dict[
            str, tuple[list[str], Future[RawBlockPublicationReceipt]]
        ] = {}
        self._lock = threading.Lock()
        self._closed = False

    def prepare(self, request_id: str, keys: list[ObjectKey]) -> None:
        """Protect a request before stores start, with idempotent exact retry.

        Args:
            request_id: Unique request identifier for this writer incarnation.
            keys: Ordered manifest of all complete chunks, without duplicates.

        Raises:
            ValueError: Identifier or keys are empty/duplicated.
            RuntimeError: Reader role, admission bound, changed retry or shutdown.
        """
        if self._core.role != "writer":
            raise RuntimeError("only a writer prepares publications")
        encoded = [encode_object_key(key).encoded for key in keys]
        with self._lock:
            if self._closed:
                raise RuntimeError("handoff controller is closed")
            existing = self._requests.get(request_id)
            if existing is not None:
                if existing[0] != encoded:
                    raise RuntimeError("publication retry changed its manifest")
                return
            if len(self._requests) >= self._max_requests:
                raise RuntimeError("handoff request admission bound reached")
            future = self._tracker.register_batch(
                request_id, encoded, expected_chunks=len(encoded), is_last_batch=True
            )
            try:
                self._core.protect_publication(request_id, encoded)
            except (RuntimeError, ValueError) as exc:
                self._tracker.fail_request(request_id, exc)
                raise
            self._requests[request_id] = (encoded, future)

    def publish(self, request_id: str, timeout_s: float) -> RawBlockPublicationReceipt:
        """Wait for all prepared objects and a completed checkpoint barrier.

        Args:
            request_id: Identifier successfully passed to ``prepare``.
            timeout_s: Finite positive deadline for object and checkpoint completion.

        Returns:
            Receipt identifying protected, committed extents; exact retries
            return the same receipt until acknowledgement retires the request.

        Raises:
            KeyError: Unknown/retired request.
            CancelledError: Preparation was cancelled while a caller waited.
            ValueError: Nonfinite or nonpositive deadline.
            TimeoutError: Incomplete objects or checkpoint by the deadline. No
                READY is issued. A checkpoint already running retains its pins.
            RuntimeError: Failed checkpoint or controller shutdown.
        """
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("publication timeout must be finite and positive")
        deadline = time.monotonic() + timeout_s
        with self._lock:
            if self._closed:
                raise RuntimeError("handoff controller is closed")
            encoded, future = self._requests[request_id]
        while not future.done():
            if all(meta is not None for meta in self._core.get_metadata_many(encoded)):
                self._tracker.complete_batch(request_id, encoded)
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RawBlockPublicationTimeout(
                    "publication has incomplete storage objects"
                )
            time.sleep(min(0.01, remaining))
        return future.result(timeout=max(0.0, deadline - time.monotonic()))

    def adopt(
        self,
        receipt: RawBlockPublicationReceipt,
        keys: list[ObjectKey],
        timeout_ms: int,
        layout: MemoryLayoutDesc,
    ) -> bool:
        """Adopt a matching generation before lookup or model dispatch.

        Args:
            receipt: Writer's advertised publication identity.
            keys: Exact ordered request manifest.
            timeout_ms: Nonnegative maximum wait for the checkpoint.
            layout: Reader's registered canonical chunk shape and dtype.

        Returns:
            Whether namespace, epoch, generation, manifest and tensor layout match.

        Raises:
            RuntimeError: Writer role or closed controller.
            ValueError: Negative timeout or incompatible derivation.
        """
        if timeout_ms < 0:
            raise ValueError("adoption timeout must be nonnegative")
        with self._lock:
            if self._closed:
                raise RuntimeError("handoff controller is closed")
            encoded = [encode_object_key(key).encoded for key in keys]
            adopted = self._core.refresh_until_publication(
                receipt,
                encoded,
                timeout_ms=timeout_ms,
                refresh_interval_ms=10,
            )
            if not adopted or len(layout.shapes) != 1:
                return False
            expected_bytes = layout.shapes[0].numel() * layout.dtypes[0].itemsize
            return all(
                meta is not None
                and [meta.shape] == layout.shapes
                and [meta.dtype] == layout.dtypes
                and meta.size == expected_bytes
                for meta in self._core.get_metadata_many(encoded)
            )

    def claim(self, identity: ReadAckIdentity, session_id: str) -> ReadClaimOutcome:
        """Bind a live writer lease to one consumer before that consumer reads.

        Args:
            identity: Complete request/receipt/consumer identity.
            session_id: Consumer group incarnation, changed on group restart.

        Returns:
            Existing tracker's grant or refusal, without releasing extents.
        """
        with self._lock:
            if self._closed:
                return ReadClaimOutcome(
                    False, "handoff controller is closed", final=True
                )
        return self._tracker.claim_read(
            identity,
            expected_writer_epoch=self._core.writer_epoch,
            expected_tp_rank=0,
            session_id=session_id,
        )

    def cancel_prepared(self, request_id: str) -> bool:
        """Retire preparation without releasing a published or claimed lease.

        Args:
            request_id: Prepared request whose model work was cancelled.

        Returns:
            True if its terminal future cannot issue READY. False for an
            unknown request or a completed successful publication; the latter
            requires an exact unread release or the claimed consumer's ACK.
            Publication already in flight retains its pins until it settles.
        """
        with self._lock:
            if self._closed:
                return False
            state = self._requests.get(request_id)
            if state is None:
                return False
            _, future = state
            cancelled = future.cancel()
            if not cancelled and (not future.done() or future.exception() is None):
                return False
            self._requests.pop(request_id)
            return True

    def release_unread(
        self,
        request_id: str,
        receipt: RawBlockPublicationReceipt,
        session_id: str,
    ) -> UnreadReleaseOutcome:
        """Release an exact publication only while no reader has claimed it.

        Args:
            request_id: Published request no longer assigned to a reader.
            receipt: Exact receipt returned by this writer.
            session_id: Dispatcher's group incarnation.

        Returns:
            Existing tracker's release or refusal. Claimed publications and
            changed identities cannot be reclaimed; exact retries are safe.
        """
        with self._lock:
            if self._closed:
                return UnreadReleaseOutcome(False, "controller is closed", final=False)
        result = self._tracker.release_unread(
            request_id,
            receipt,
            expected_writer_epoch=self._core.writer_epoch,
            session_id=session_id,
            tp_rank=0,
            reason="dispatcher cancelled before read claim",
        )
        if result.released:
            with self._lock:
                self._requests.pop(request_id, None)
        return result

    def acknowledge(self, identity: ReadAckIdentity, session_id: str) -> ReadAckOutcome:
        """Release exact claimed extents after the consumer has finished reading.

        Args:
            identity: Same full identity used by the successful claim.
            session_id: Same session incarnation used by the claim.

        Returns:
            Applied, exact duplicate, rejected or unresolved. Only applied
            outcomes retire admission state. Failed reads must not ACK.
        """
        with self._lock:
            if self._closed:
                return ReadAckOutcome.UNRESOLVED
        result = self._tracker.apply_read_ack(
            identity,
            expected_writer_epoch=self._core.writer_epoch,
            expected_tp_rank=0,
            session_id=session_id,
        )
        if result in (ReadAckOutcome.APPLIED, ReadAckOutcome.ALREADY_APPLIED):
            with self._lock:
                self._requests.pop(identity.req_id, None)
        return result

    def report_status(self) -> dict[str, int]:
        """Return tracker leases and bounded request admission counts."""
        with self._lock:
            return {
                **self._tracker.report_status(),
                "request_count": len(self._requests),
            }

    def close(self) -> None:
        """Stop publication admission and drain the publisher before core close.

        Raises:
            RuntimeError: Publisher could not quiesce; the owner must retain core.
        """
        with self._lock:
            self._closed = True
        if not self._tracker.close():
            raise RuntimeError("handoff publisher did not quiesce")
