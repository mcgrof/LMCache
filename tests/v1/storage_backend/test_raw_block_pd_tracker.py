# SPDX-License-Identifier: Apache-2.0
"""Request-level completion tests for raw-block storage P/D."""

# Future
from __future__ import annotations

# Standard
from threading import Event

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import (
    RawBlockPDRequestTracker,
    RawBlockPublicationReceipt,
)


class _FakeCore:
    def __init__(self) -> None:
        self.publish_started = Event()
        self.allow_publish = Event()
        self.published: list[list[str]] = []
        self.leased: list[str] = []

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
        for encoded_key in encoded_keys:
            self.leased.remove(encoded_key)


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
    finally:
        core.allow_publish.set()
        tracker.close()
    assert core.leased == []


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
