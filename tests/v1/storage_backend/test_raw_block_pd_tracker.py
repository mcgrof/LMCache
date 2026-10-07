# SPDX-License-Identifier: Apache-2.0
"""Request-level completion tests for raw-block storage P/D."""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import (
    RawBlockPDRequestTracker,
    RawBlockPublicationReceipt,
    ReadAckIdentity,
    ReadAckOutcome,
    ReadClaimOutcome,
)


class _FakeCore:
    def __init__(self) -> None:
        self.publish_started = Event()
        self.allow_publish = Event()
        self.published: list[list[str]] = []
        self.leased: list[str] = []
        self.unlock_calls = 0
        self.unlock_raises = False
        self.linked_publications: list[tuple[str, RawBlockPublicationReceipt]] = []

    def publish_request(self, encoded_keys: list[str]) -> RawBlockPublicationReceipt:
        self.publish_started.set()
        if not self.allow_publish.wait(timeout=5):
            raise TimeoutError("test publication gate timed out")
        self.published.append(list(encoded_keys))
        return RawBlockPublicationReceipt(
            writer_epoch="writer-1",
            checkpoint_seq=7,
            key_count=len(encoded_keys),
            manifest_digest="digest",
        )

    def get_metadata_prefix(
        self,
        encoded_keys: list[str],
        *,
        lock: bool = False,
    ) -> list[object]:
        assert lock
        self.leased.extend(encoded_keys)
        return [object() for _ in encoded_keys]

    def unlock_many(self, encoded_keys: list[str]) -> None:
        if self.unlock_raises:
            raise OSError("the device refused to release these keys")
        self.unlock_calls += 1
        for encoded_key in encoded_keys:
            self.leased.remove(encoded_key)

    def link_io_publication(
        self,
        req_id: str,
        receipt: RawBlockPublicationReceipt,
    ) -> None:
        self.linked_publications.append((req_id, receipt))


@pytest.mark.parametrize("outcome", ["success", "failure", "invalid_batch", "close"])
def test_terminal_callbacks_can_query_retired_tracker_state(outcome: str) -> None:
    """Synchronous Future callbacks must not run inside the admission lock."""
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    terminal = tracker.register_batch(
        "callback-request", ["key"], expected_chunks=1, is_last_batch=True
    )
    callback_finished = Event()
    statuses: list[dict[str, int]] = []

    def callback(_future) -> None:
        statuses.append(tracker.report_status())
        callback_finished.set()

    def settle() -> None:
        if outcome == "success":
            core.allow_publish.set()
            tracker.complete_batch("callback-request", ["key"])
        elif outcome == "failure":
            tracker.fail_request("callback-request", OSError("write failed"))
        elif outcome == "invalid_batch":
            tracker.complete_batch("callback-request", ["unknown"])
        else:
            tracker.close()

    terminal.add_done_callback(callback)
    worker = Thread(target=settle, daemon=True)
    worker.start()
    try:
        assert callback_finished.wait(2), "terminal callback deadlocked on tracker"
        worker.join(2)
        assert not worker.is_alive()
        assert statuses[0]["inflight_request_count"] == 0
        assert statuses[0]["live_lease_count"] == (1 if outcome == "success" else 0)
    finally:
        core.allow_publish.set()
        if callback_finished.is_set():
            tracker.close()


def test_tracker_waits_for_every_write_before_publication() -> None:
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = tracker.register_batch(
            "request-1",
            ["key-1"],
            expected_chunks=2,
            is_last_batch=False,
        )
        same_terminal = tracker.register_batch(
            "request-1",
            ["key-2"],
            expected_chunks=2,
            is_last_batch=True,
        )
        assert same_terminal is terminal

        tracker.complete_batch("request-1", ["key-2"])
        assert not core.publish_started.wait(timeout=0.05)
        assert not terminal.done()

        tracker.complete_batch("request-1", ["key-1"])
        assert core.publish_started.wait(timeout=1)
        assert not terminal.done()

        core.allow_publish.set()
        receipt = terminal.result(timeout=1)
        assert receipt.checkpoint_seq == 7
        assert core.published == [["key-1", "key-2"]]
        assert core.leased == ["key-1", "key-2"]
        assert core.linked_publications == [("request-1", receipt)]
    finally:
        core.allow_publish.set()
        tracker.close()
    # Shutdown released nothing: the hold is what protects these extents
    # from the next writer, and nothing here says a reader is done with
    # them.
    assert core.leased == ["key-1", "key-2"]


def test_cancellation_at_receipt_delivery_releases_an_unclaimed_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the public future cannot consume admission capacity forever."""
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    terminal = tracker.register_batch(
        "cancel-at-delivery", ["key"], expected_chunks=1, is_last_batch=True
    )
    set_result = terminal.set_result
    delivery_attempted = Event()

    def cancel_before_delivery(receipt: RawBlockPublicationReceipt) -> None:
        assert terminal.cancel()
        delivery_attempted.set()
        set_result(receipt)

    monkeypatch.setattr(terminal, "set_result", cancel_before_delivery)
    try:
        core.allow_publish.set()
        tracker.complete_batch("cancel-at-delivery", ["key"])
        assert delivery_attempted.wait(2)
        assert tracker.close(timeout_s=2)
        assert terminal.cancelled()
        assert tracker.live_lease_count() == 0
        assert core.leased == []
    finally:
        tracker.close()


def test_tracker_failure_never_publishes() -> None:
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = tracker.register_batch(
            "request-failed",
            ["key-1"],
            expected_chunks=1,
            is_last_batch=True,
        )
        tracker.fail_request("request-failed", OSError("write failed"))
        tracker.complete_batch("request-failed", ["key-1"])

        with pytest.raises(OSError, match="write failed"):
            terminal.result(timeout=1)
        assert not core.publish_started.is_set()
        assert core.published == []
    finally:
        tracker.close()


def test_tracker_rejects_an_incomplete_last_batch() -> None:
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = tracker.register_batch(
            "request-short",
            ["key-1"],
            expected_chunks=2,
            is_last_batch=True,
        )
        with pytest.raises(RuntimeError, match="1 of 2 keys"):
            terminal.result(timeout=1)
        assert not core.publish_started.is_set()
    finally:
        tracker.close()


def test_tracker_rejects_keys_repeated_across_batches() -> None:
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = tracker.register_batch(
            "request-duplicate",
            ["key-1"],
            expected_chunks=2,
            is_last_batch=False,
        )
        same_terminal = tracker.register_batch(
            "request-duplicate",
            ["key-1"],
            expected_chunks=2,
            is_last_batch=False,
        )
        assert same_terminal is terminal
        with pytest.raises(RuntimeError, match="repeated keys"):
            terminal.result(timeout=1)
        assert not core.publish_started.is_set()
    finally:
        tracker.close()


def test_tracker_close_fails_unfinished_requests() -> None:
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    terminal = tracker.register_batch(
        "request-shutdown",
        ["key-1"],
        expected_chunks=1,
        is_last_batch=False,
    )
    tracker.close()
    with pytest.raises(RuntimeError, match="aborted during shutdown"):
        terminal.result(timeout=1)


def test_tracker_finalizes_when_last_iteration_has_no_new_keys() -> None:
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = tracker.register_batch(
            "request-finalize",
            ["key-1"],
            expected_chunks=1,
            is_last_batch=False,
            completed_keys=["key-1"],
        )
        assert tracker.has_request("request-finalize")
        assert (
            tracker.finalize_request(
                "request-finalize",
                expected_chunks=1,
            )
            is terminal
        )
        assert core.publish_started.wait(timeout=1)
        core.allow_publish.set()
        assert terminal.result(timeout=1).key_count == 1
        assert core.published == [["key-1"]]
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_cancelled_request_never_publishes() -> None:
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = tracker.register_batch(
            "request-cancelled",
            ["key-1"],
            expected_chunks=1,
            is_last_batch=False,
        )
        tracker.fail_request(
            "request-cancelled",
            RuntimeError("request was cancelled"),
        )
        tracker.complete_batch("request-cancelled", ["key-1"])
        with pytest.raises(RuntimeError, match="cancelled"):
            terminal.result(timeout=1)
        assert not core.publish_started.is_set()
    finally:
        tracker.close()


def test_tracker_finished_request_fails_without_a_last_batch() -> None:
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = tracker.register_batch(
            "request-incomplete",
            ["key-1"],
            expected_chunks=2,
            is_last_batch=False,
            completed_keys=["key-1"],
        )
        tracker.finish_request("request-incomplete")
        with pytest.raises(RuntimeError, match="before its last storage batch"):
            terminal.result(timeout=1)
        assert not core.publish_started.is_set()
    finally:
        tracker.close()


def test_tracker_finished_request_does_not_cancel_publication() -> None:
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = tracker.register_batch(
            "request-complete",
            ["key-1"],
            expected_chunks=1,
            is_last_batch=True,
            completed_keys=["key-1"],
        )
        assert core.publish_started.wait(timeout=1)
        tracker.finish_request("request-complete")
        assert not terminal.done()
        core.allow_publish.set()
        assert terminal.result(timeout=1).key_count == 1
    finally:
        core.allow_publish.set()
        tracker.close()


class _MappingCore:
    """A core whose keys can be moved, and which refuses to move a held one."""

    def __init__(self) -> None:
        self.mapping: dict[str, str] = {"key-1": "slot-a", "key-2": "slot-b"}
        self.held: dict[str, int] = {}
        self.publish_started = Event()
        self.allow_publish = Event()
        self.held_when_publishing: dict[str, int] = {}
        self.publish_error: BaseException | None = None

    def get_metadata_prefix(
        self,
        encoded_keys: list[str],
        *,
        lock: bool = False,
    ) -> list[object]:
        out: list[object] = []
        for encoded_key in encoded_keys:
            if encoded_key not in self.mapping:
                break
            if lock:
                self.held[encoded_key] = self.held.get(encoded_key, 0) + 1
            out.append(object())
        return out

    def unlock_many(self, encoded_keys: list[str]) -> None:
        for encoded_key in encoded_keys:
            remaining = self.held.get(encoded_key, 0) - 1
            if remaining > 0:
                self.held[encoded_key] = remaining
            else:
                self.held.pop(encoded_key, None)

    def publish_request(self, encoded_keys: list[str]) -> RawBlockPublicationReceipt:
        self.held_when_publishing = dict(self.held)
        self.publish_started.set()
        if not self.allow_publish.wait(timeout=5):
            raise TimeoutError("test publication gate timed out")
        if self.publish_error is not None:
            raise self.publish_error
        return RawBlockPublicationReceipt(
            writer_epoch="writer-1",
            checkpoint_seq=7,
            key_count=len(encoded_keys),
            manifest_digest="|".join(self.mapping[k] for k in encoded_keys),
        )

    def rebind(self, encoded_key: str, slot: str) -> None:
        """Delete a key and write it again elsewhere, if nothing holds it."""
        if encoded_key in self.held:
            raise RuntimeError(f"{encoded_key} is held")
        self.mapping[encoded_key] = slot


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll until a predicate holds, for state a background callback sets."""
    # Standard
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _publish_one(tracker, core, keys=("key-1", "key-2")):
    terminal = tracker.register_batch(
        "request-1",
        list(keys),
        expected_chunks=len(keys),
        is_last_batch=True,
    )
    tracker.complete_batch("request-1", list(keys))
    assert core.publish_started.wait(timeout=5)
    return terminal


def test_publication_holds_the_extents_before_it_describes_them() -> None:
    """A receipt describes where keys live, so they must already be held.

    Taking the hold afterwards leaves an interval in which a key can be
    deleted and written somewhere else, which would leave the hold on the
    replacement and the receipt on what was there before.
    """
    core = _MappingCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = _publish_one(tracker, core)

        assert core.held_when_publishing == {"key-1": 1, "key-2": 1}, (
            "publication began describing keys it had not taken a hold on"
        )
        with pytest.raises(RuntimeError, match="is held"):
            core.rebind("key-1", "slot-moved")

        core.allow_publish.set()
        receipt = terminal.result(timeout=5)
        assert receipt.manifest_digest == "slot-a|slot-b"
        assert core.held == {"key-1": 1, "key-2": 1}
    finally:
        core.allow_publish.set()
        tracker.close()


def test_publication_that_fails_releases_the_extents_it_held() -> None:
    """A hold taken for a receipt that never exists has no owner."""
    core = _MappingCore()
    core.publish_error = RuntimeError("checkpoint write failed")
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = _publish_one(tracker, core)
        core.allow_publish.set()
        with pytest.raises(RuntimeError, match="checkpoint write failed"):
            terminal.result(timeout=5)
        assert _wait_until(lambda: core.held == {}), core.held
        core.rebind("key-1", "slot-moved")
    finally:
        tracker.close()


def test_publication_releases_extents_when_the_request_ended_first() -> None:
    """Nothing will release a hold recorded against a request that is gone."""
    core = _MappingCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = _publish_one(tracker, core)
        tracker.fail_request("request-1", RuntimeError("request abandoned"))
        core.allow_publish.set()
        with pytest.raises(RuntimeError, match="request abandoned"):
            terminal.result(timeout=5)
        # The failure is visible as soon as it is recorded; releasing the
        # extents waits for the publication already in flight to return.
        assert _wait_until(lambda: core.held == {}), core.held
    finally:
        tracker.close()


def test_tracker_drops_a_finished_request_but_remembers_its_answer() -> None:
    """Working state is for requests in flight; answers outlive them.

    Keeping every request's key sets grows without bound for the life of
    the writer. Forgetting a request entirely is worse, because a caller
    that arrives late would start it over.
    """
    core = _MappingCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = _publish_one(tracker, core)
        core.allow_publish.set()
        receipt = terminal.result(timeout=5)
        assert _wait_until(lambda: not tracker._requests), tracker._requests

        # A late finalize gets the answer it already had, not a new request.
        again = tracker.finalize_request("request-1", expected_chunks=2)
        assert again.result(timeout=5) is receipt
        assert not tracker._requests

        # Registering it again is a protocol error, not a second publication.
        with pytest.raises(RuntimeError, match="already finished"):
            tracker.register_batch(
                "request-1",
                ["key-1"],
                expected_chunks=1,
                is_last_batch=True,
            )
        assert core.held == {"key-1": 1, "key-2": 1}
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_drops_a_failed_request_too() -> None:
    """A request that failed holds no more state than one that succeeded."""
    core = _MappingCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        terminal = tracker.register_batch(
            "request-1",
            ["key-1"],
            expected_chunks=1,
            is_last_batch=False,
        )
        tracker.fail_request("request-1", RuntimeError("gave up"))
        with pytest.raises(RuntimeError, match="gave up"):
            terminal.result(timeout=5)
        assert not tracker._requests
        assert tracker.has_request("request-1") is False
    finally:
        tracker.close()


def test_tracker_forgets_the_oldest_finished_requests() -> None:
    """The record of finished requests is bounded, not merely smaller."""
    # First Party
    from lmcache.v1.storage_backend.raw_block import pd as pd_module

    core = _MappingCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    original = pd_module._FINISHED_HISTORY
    pd_module._FINISHED_HISTORY = 4
    try:
        for index in range(10):
            req_id = f"request-{index}"
            terminal = tracker.register_batch(
                req_id,
                [f"key-{index}"],
                expected_chunks=1,
                is_last_batch=False,
            )
            tracker.fail_request(req_id, RuntimeError("gave up"))
            with pytest.raises(RuntimeError):
                terminal.result(timeout=5)
        assert len(tracker._finished) == 4
        assert list(tracker._finished) == [f"request-{i}" for i in range(6, 10)]
    finally:
        pd_module._FINISHED_HISTORY = original
        tracker.close()


class _RoleConfig:
    """The parts of an engine config that decide the raw-block role."""

    def __init__(self, *, shared: bool, pd_role: str | None) -> None:
        self.pd_uses_shared_storage = shared
        self.pd_role = pd_role


def test_raw_block_role_follows_the_pd_role_when_storage_is_the_data_path():
    """A deployment should name which side it is once, not twice.

    The P/D role already says it, so a handoff configured that way does not
    have to repeat it as a plugin setting where the two could disagree.
    """
    # First Party
    from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
        _resolve_role,
    )

    assert _resolve_role(_RoleConfig(shared=True, pd_role="sender"), {}) == "writer"
    assert _resolve_role(_RoleConfig(shared=True, pd_role="receiver"), {}) == "reader"
    # An explicit plugin setting still wins, for a pairing set up without the
    # P/D switch at all.
    assert (
        _resolve_role(
            _RoleConfig(shared=True, pd_role="receiver"),
            {"rust_raw_block.role": "writer"},
        )
        == "writer"
    )
    # Without the switch the plugin default stands.
    assert _resolve_role(_RoleConfig(shared=False, pd_role=None), {}) == "writer"


def _published_lease(core: _FakeCore, tracker: RawBlockPDRequestTracker) -> None:
    """Drive one request to a published lease."""
    terminal = tracker.register_batch(
        "request-1",
        ["key-1"],
        expected_chunks=1,
        is_last_batch=True,
    )
    tracker.complete_batch("request-1", ["key-1"])
    core.allow_publish.set()
    terminal.result(timeout=1)


def _published_lease_named(
    core: _FakeCore,
    tracker: RawBlockPDRequestTracker,
    req_id: str,
) -> None:
    """Drive one further request to a published lease."""
    terminal = tracker.register_batch(
        req_id,
        [f"key-for-{req_id}"],
        expected_chunks=1,
        is_last_batch=True,
    )
    tracker.complete_batch(req_id, [f"key-for-{req_id}"])
    terminal.result(timeout=1)


def _message(**overrides: object) -> tuple[ReadAckIdentity, str, int, str]:
    """Build one control message and the writer identity it is checked against."""
    fields: dict[str, object] = {
        "req_id": "request-1",
        "consumer_instance_id": "consumer-1",
        "tp_rank": 0,
        "writer_epoch": "writer-1",
        "checkpoint_seq": 7,
        "manifest_digest": "digest",
    }
    expected_rank = int(overrides.pop("expected_tp_rank", 0))  # type: ignore[call-overload]
    expected_epoch = str(overrides.pop("expected_writer_epoch", "writer-1"))
    session = str(overrides.pop("session_id", "session-1"))
    fields.update(overrides)
    return (
        ReadAckIdentity(**fields),  # type: ignore[arg-type]
        expected_epoch,
        expected_rank,
        session,
    )


def _claim(
    tracker: RawBlockPDRequestTracker,
    **overrides: object,
) -> ReadClaimOutcome:
    identity, epoch, rank, session = _message(**overrides)
    return tracker.claim_read(
        identity,
        expected_writer_epoch=epoch,
        expected_tp_rank=rank,
        session_id=session,
    )


def _ack(
    tracker: RawBlockPDRequestTracker,
    **overrides: object,
) -> ReadAckOutcome:
    identity, epoch, rank, session = _message(**overrides)
    return tracker.apply_read_ack(
        identity,
        expected_writer_epoch=epoch,
        expected_tp_rank=rank,
        session_id=session,
    )


def _claimed_lease(
    core: _FakeCore,
    tracker: RawBlockPDRequestTracker,
    **overrides: object,
) -> None:
    """Publish one request and let a consumer take the read, as serving does."""
    req_id = str(overrides.get("req_id", "request-1"))
    if req_id == "request-1":
        _published_lease(core, tracker)
    else:
        _published_lease_named(core, tracker, req_id)
    assert _claim(tracker, **overrides).granted


@pytest.mark.parametrize("release_raises", [False, True])
def test_tracker_refuses_a_read_after_an_unread_release_starts(
    release_raises: bool,
) -> None:
    """Keep a lease being released unavailable to new readers."""
    release_started = Event()
    finish_release = Event()

    class ReleasingCore(_FakeCore):
        def unlock_many(self, encoded_keys: list[str]) -> None:
            release_started.set()
            if not finish_release.wait(timeout=5):
                raise TimeoutError("test release gate timed out")
            super().unlock_many(encoded_keys)

    core = ReleasingCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        core.allow_publish.set()
        receipt = tracker.register_batch(
            "request-1",
            ["key-1"],
            expected_chunks=1,
            is_last_batch=True,
            completed_keys=["key-1"],
        ).result(timeout=1)
        core.unlock_raises = release_raises
        with ThreadPoolExecutor(max_workers=1) as executor:
            release = executor.submit(
                tracker.release_unread,
                "request-1",
                receipt,
                expected_writer_epoch="writer-1",
            )
            try:
                assert release_started.wait(timeout=1)
                claim = _claim(tracker)
                assert not claim.granted
                assert tracker.bound_consumer("session-1") is None
            finally:
                finish_release.set()
            outcome = release.result(timeout=1)
        assert outcome.released is not release_raises
        assert getattr(outcome, "final", True) is not release_raises
        repeated = tracker.release_unread(
            "request-1", receipt, expected_writer_epoch="writer-1"
        )
        assert repeated.released is not release_raises
        assert getattr(repeated, "final", True) is not release_raises
        assert core.unlock_calls == (0 if release_raises else 1)
        assert not _claim(tracker).granted
    finally:
        finish_release.set()
        tracker.close()


def test_unread_release_retry_matches_the_full_applied_identity() -> None:
    """A lost reply is idempotent only for the same session and receipt."""
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        core.allow_publish.set()
        receipt = tracker.register_batch(
            "request-1",
            ["key-1"],
            expected_chunks=1,
            is_last_batch=True,
            completed_keys=["key-1"],
        ).result(timeout=1)
        first = tracker.release_unread(
            "request-1",
            receipt,
            expected_writer_epoch="writer-1",
            session_id="session-1",
        )
        duplicate = tracker.release_unread(
            "request-1",
            receipt,
            expected_writer_epoch="writer-1",
            session_id="session-1",
        )
        mismatch = tracker.release_unread(
            "request-1",
            receipt,
            expected_writer_epoch="writer-1",
            session_id="session-2",
        )

        assert first.released
        assert duplicate.released
        assert "already" in duplicate.reason
        assert not mismatch.released
        assert core.unlock_calls == 1
    finally:
        tracker.close()


def test_tracker_releases_an_extent_on_a_matching_acknowledgement() -> None:
    """Nothing but an acknowledgement the writer can vouch for frees a lease."""
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        _claimed_lease(core, tracker)
        assert core.leased == ["key-1"]
        assert tracker.live_lease_count() == 1

        assert _ack(tracker) is ReadAckOutcome.APPLIED
        assert core.leased == []
        assert tracker.live_lease_count() == 0
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_releases_nothing_twice_for_a_duplicate_acknowledgement() -> None:
    """A lost confirmation is safe to retry, so a duplicate must be inert.

    The consumer that never heard back sends the same acknowledgement again.
    Releasing a second reference would free an extent a later request may
    already own.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        _claimed_lease(core, tracker)
        assert _ack(tracker) is ReadAckOutcome.APPLIED
        assert _ack(tracker) is ReadAckOutcome.ALREADY_APPLIED
        assert _ack(tracker) is ReadAckOutcome.ALREADY_APPLIED
        assert core.unlock_calls == 1
    finally:
        core.allow_publish.set()
        tracker.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"writer_epoch": "writer-0"},
        {"checkpoint_seq": 6},
        {"manifest_digest": "someone-elses-digest"},
        {"tp_rank": 1},
        {"req_id": "request-2"},
    ],
    ids=["stale-epoch", "stale-seq", "wrong-digest", "wrong-rank", "unknown-request"],
)
def test_tracker_rejects_an_acknowledgement_it_cannot_match(
    overrides: dict[str, object],
) -> None:
    """Every field is checked by the writer, because it holds the lease.

    A stale acknowledgement naming a previous incarnation's receipt is
    exactly the message that must not reclaim a live extent, and a relay
    that validated it for us would not change who is responsible.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        _claimed_lease(core, tracker)
        assert _ack(tracker, **overrides) is ReadAckOutcome.REJECTED
        assert core.leased == ["key-1"]
        assert tracker.live_lease_count() == 1
        assert core.unlock_calls == 0
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_keeps_every_lease_at_shutdown() -> None:
    """This writer going away is not news about a reader.

    Unlocking a key lets its entry be deleted and its extent handed to the
    next writer. A consumer elsewhere may still be reading those bytes, and
    nothing a local shutdown can see says otherwise.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    _claimed_lease(core, tracker)
    tracker.close()
    assert core.leased == ["key-1"]
    assert tracker.live_lease_count() == 1
    assert core.unlock_calls == 0


def test_a_quiesced_group_releases_its_holds_on_a_separate_decision() -> None:
    """Releasing a stopped group's holds is an assertion, not a shutdown.

    The operator says every engine that could read this namespace has
    stopped, which is a statement about other machines that this process
    cannot make for itself. So it is asked for separately, after close.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    _claimed_lease(core, tracker)
    tracker.close()
    assert core.leased == ["key-1"]

    assert tracker.release_quiesced_leases() == 1
    assert core.leased == []
    assert tracker.live_lease_count() == 0


def test_a_quiesced_teardown_keeps_a_release_that_never_reported_back() -> None:
    """A stopped group does not make an unknown outcome known.

    A release that was started and did not return may or may not have
    dropped its reference. Running it again could drop one this writer does
    not own, which hands a live extent to a later request -- and the whole
    group having stopped says nothing about that.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    _claimed_lease(core, tracker)
    core.unlock_raises = True
    assert _ack(tracker) is ReadAckOutcome.UNRESOLVED
    core.unlock_raises = False
    tracker.close()

    assert tracker.release_quiesced_leases() == 0
    assert tracker.live_lease_count() == 1
    assert core.unlock_calls == 0


def test_tracker_refuses_an_acknowledgement_naming_another_producer() -> None:
    """A writer only releases holds it published itself.

    The producer identity is on the wire, and comparing it against a field
    the same message supplied would check nothing, so the writer's own
    identity arrives separately. Without that comparison a message meant
    for a different writer -- or naming none -- releases this one's extents.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        _claimed_lease(core, tracker)
        assert _ack(tracker, writer_epoch="somebody-else") is ReadAckOutcome.REJECTED
        assert core.leased == ["key-1"]
        assert tracker.live_lease_count() == 1
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_refuses_an_acknowledgement_from_nobody() -> None:
    """An empty consumer identity is not a consumer.

    A message nobody can be held to still carried a matching receipt, and
    a writer that only logged the consumer released the extents for it.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        _claimed_lease(core, tracker)
        assert _ack(tracker, consumer_instance_id="") is ReadAckOutcome.REJECTED
        assert core.leased == ["key-1"]
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_binds_a_session_before_any_read_happens() -> None:
    """A consumer that restarted cannot take over reads it never made.

    The binding is settled by the claim, which happens before the bytes
    move. Settling it on whichever acknowledgement arrives first is a race
    between a consumer and the process that replaced it, and the loser's
    reads are then held for the writer's lifetime.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        _published_lease(core, tracker)
        assert _claim(tracker).granted
        assert tracker.bound_consumer("session-1") == "consumer-1"

        _published_lease_named(core, tracker, "request-2")
        restarted = _claim(
            tracker,
            req_id="request-2",
            consumer_instance_id="consumer-2-after-a-restart",
        )
        assert not restarted.granted
        # The answer cannot change while this session lasts, and the
        # consumer is told so rather than asking once per request.
        assert restarted.final

        # The original consumer has not acknowledged anything yet, and its
        # read is still the one this session is bound to.
        assert _ack(tracker) is ReadAckOutcome.APPLIED
        assert tracker.live_lease_count() == 1

        # A whole-group restart is a new session, and is allowed.
        assert _claim(
            tracker,
            req_id="request-2",
            consumer_instance_id="consumer-2-after-a-restart",
            session_id="session-2",
        ).granted
        assert (
            _ack(
                tracker,
                req_id="request-2",
                consumer_instance_id="consumer-2-after-a-restart",
                session_id="session-2",
            )
            is ReadAckOutcome.APPLIED
        )
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_refuses_an_acknowledgement_for_a_read_nobody_claimed() -> None:
    """Only the consumer that took the read may release it.

    Without this, the writer would settle who its reader is on whichever
    acknowledgement arrived first -- which is a message from a process that
    may have come up after the reads it is naming.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        _published_lease(core, tracker)
        assert _ack(tracker) is ReadAckOutcome.REJECTED
        assert core.leased == ["key-1"]
        assert tracker.live_lease_count() == 1
        assert core.unlock_calls == 0
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_refuses_a_claim_for_a_publication_it_does_not_hold() -> None:
    """A read cannot be granted over extents this writer is not holding.

    Granting one would let a consumer read a publication already released
    or never made, and then acknowledge it.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        _published_lease(core, tracker)
        assert not _claim(tracker, req_id="request-never-published").granted
        assert not _claim(tracker, manifest_digest="someone-elses-digest").granted
        assert not _claim(tracker, writer_epoch="somebody-else").granted
        assert not _claim(tracker, consumer_instance_id="").granted
        assert tracker.live_lease_count() == 1
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_does_not_answer_a_mismatched_duplicate_with_success() -> None:
    """A tombstone keeps the identity that released the hold, not the name.

    Remembering only the request identifier answered any later message
    naming it with "already applied". That frees nothing extra by itself,
    and becomes a false success the moment a consumer retires an
    obligation on the answer.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        _claimed_lease(core, tracker)
        assert _ack(tracker) is ReadAckOutcome.APPLIED
        assert _ack(tracker) is ReadAckOutcome.ALREADY_APPLIED
        assert (
            _ack(tracker, manifest_digest="someone-elses-digest")
            is ReadAckOutcome.REJECTED
        )
        assert core.unlock_calls == 1
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_does_not_report_success_before_the_release_returns() -> None:
    """A release that failed is not a release.

    The hold was dropped from the lease table and recorded as released
    before the unlock ran, so an unlock that raised left the extents
    locked forever with nothing able to try again -- and the writer's own
    accounting said they were free.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        _claimed_lease(core, tracker)
        core.unlock_raises = True

        assert _ack(tracker) is ReadAckOutcome.UNRESOLVED
        assert tracker.live_lease_count() == 1, "the hold must still be the writer's"

        # And a retry does not run the release a second time: the first one
        # may have partially applied, and nothing here can say.
        core.unlock_raises = False
        assert _ack(tracker) is ReadAckOutcome.UNRESOLVED
        assert core.unlock_calls == 0
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_says_it_cannot_tell_once_a_tombstone_is_gone() -> None:
    """An evicted record is not evidence of anything.

    Answering "rejected" tells a consumer to stop retrying something that
    may still be owed; answering "already applied" invents a success. The
    writer says it cannot tell, and the hold -- if there is one -- stands.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        _claimed_lease(core, tracker)
        assert _ack(tracker) is ReadAckOutcome.APPLIED

        # Evicting by hand: the real bound is thousands of requests, and
        # what matters is the answer after an eviction, not the number.
        tracker._released.clear()
        tracker._forgot_released = True

        assert _ack(tracker) is ReadAckOutcome.UNRESOLVED
    finally:
        core.allow_publish.set()
        tracker.close()


def test_tracker_stops_admitting_before_its_hold_bound_is_exceeded() -> None:
    """Deduplicated requests grow holds without consuming any device slot.

    A hold is released only by an acknowledgement, so a consumer that stops
    acknowledging grows this state indefinitely -- and because two requests
    naming the same extent each hold a reference, a finite device does not
    bound it and neither does a bounded request history. Publication stops
    instead.
    """
    core = _FakeCore()
    tracker = RawBlockPDRequestTracker(core, max_live_leases=2)  # type: ignore[arg-type]
    try:
        for index in range(2):
            req_id = f"request-{index}"
            terminal = tracker.register_batch(
                req_id,
                ["shared-key"],
                expected_chunks=1,
                is_last_batch=True,
            )
            tracker.complete_batch(req_id, ["shared-key"])
            core.allow_publish.set()
            terminal.result(timeout=1)
        assert tracker.live_lease_count() == 2
        status = tracker.report_status()
        assert status["live_lease_count"] == 2
        assert status["live_extent_reference_count"] == 2
        assert status["live_unique_extent_count"] == 1
        # Two requests, one key: two independent holds, so the extent is
        # protected until both are acknowledged.
        assert core.leased == ["shared-key", "shared-key"]

        with pytest.raises(RuntimeError, match="refusing to publish"):
            tracker.register_batch(
                "one-too-many",
                ["shared-key"],
                expected_chunks=1,
                is_last_batch=True,
            )

        # Acknowledging one frees exactly one hold, and admission resumes.
        assert _claim(
            tracker, req_id="request-0", consumer_instance_id="consumer-1"
        ).granted
        assert (
            _ack(tracker, req_id="request-0", consumer_instance_id="consumer-1")
            is ReadAckOutcome.APPLIED
        )
        assert core.leased == ["shared-key"]
        assert tracker.live_lease_count() == 1
        terminal = tracker.register_batch(
            "one-too-many",
            ["shared-key"],
            expected_chunks=1,
            is_last_batch=True,
        )
        tracker.complete_batch("one-too-many", ["shared-key"])
        terminal.result(timeout=1)
    finally:
        core.allow_publish.set()
        tracker.close()


def test_close_waits_for_publication_cleanup() -> None:
    """Keep the core alive until an abandoned publication releases its holds."""
    core = _MappingCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    unlock_started = Event()
    allow_unlock = Event()
    unlock = core.unlock_many

    def gated_unlock(encoded_keys: list[str]) -> None:
        unlock_started.set()
        assert allow_unlock.wait(5), "publication cleanup did not resume"
        unlock(encoded_keys)

    core.unlock_many = gated_unlock  # type: ignore[method-assign]
    try:
        terminal = _publish_one(tracker, core)
        tracker.fail_request("request-1", RuntimeError("request abandoned"))
        core.allow_publish.set()
        assert unlock_started.wait(5), "publication never reached cleanup"
        assert tracker.close(timeout_s=0.01) is False
        assert core.held == {"key-1": 1, "key-2": 1}
        with pytest.raises(RuntimeError, match="request abandoned"):
            terminal.result(timeout=5)
        allow_unlock.set()
        assert _wait_until(lambda: tracker.close(timeout_s=0.01))
        assert core.held == {}
    finally:
        core.allow_publish.set()
        allow_unlock.set()
        tracker.close()


def test_rejected_publication_submission_settles_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report executor refusal without leaving a phantom publication active."""
    core = _MappingCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("publisher unavailable")

    monkeypatch.setattr(tracker._publisher, "submit", refuse)
    try:
        terminal = tracker.register_batch(
            "request-1",
            ["key-1"],
            expected_chunks=1,
            is_last_batch=True,
            completed_keys=["key-1"],
        )
        with pytest.raises(RuntimeError, match="publisher unavailable"):
            terminal.result(timeout=5)
        assert tracker.close(timeout_s=0.01) is True
        assert core.held == {}
        assert not core.publish_started.is_set()
    finally:
        tracker.close()


def test_close_accounts_for_a_publication_waiting_to_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seal publication before executor handoff and refuse a late submitter."""
    # Standard
    from concurrent.futures import ThreadPoolExecutor

    core = _MappingCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    submit_started = Event()
    allow_submit = Event()
    submit = tracker._submit_publication

    def gated_submit(*args: Any, **kwargs: Any) -> None:
        submit_started.set()
        assert allow_submit.wait(5), "publication handoff did not resume"
        submit(*args, **kwargs)

    monkeypatch.setattr(tracker, "_submit_publication", gated_submit)
    with ThreadPoolExecutor(max_workers=1) as caller:
        registration = caller.submit(
            tracker.register_batch,
            "request-1",
            ["key-1"],
            expected_chunks=1,
            is_last_batch=True,
            completed_keys=["key-1"],
        )
        try:
            assert submit_started.wait(5)
            assert tracker.close(timeout_s=0.01) is False
            allow_submit.set()
            terminal = registration.result(timeout=5)
            with pytest.raises(RuntimeError, match="aborted during shutdown"):
                terminal.result(timeout=5)
            assert tracker.close(timeout_s=0.01) is True
            assert core.held == {}
            assert not core.publish_started.is_set()
        finally:
            allow_submit.set()
            tracker.close()
