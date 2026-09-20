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
