# SPDX-License-Identifier: Apache-2.0
"""Publication ownership and persistence regressions through public APIs."""

# Standard
from dataclasses import replace
from pathlib import Path
from typing import Any
import importlib.util

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import (
    RawBlockCore,
    RawBlockPDRequestTracker,
    RawBlockPublicationReceipt,
    encode_object_key,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    make_test_derivation,
)


class _PublicationCore:
    def __init__(self) -> None:
        self.held: list[str] = []

    def get_metadata_prefix(self, keys: list[str], *, lock: bool) -> list[object]:
        assert lock
        self.held.extend(keys)
        return [object() for _ in keys]

    def unlock_many(self, keys: list[str]) -> None:
        for key in keys:
            self.held.remove(key)

    def publish_request(self, keys: list[str]) -> RawBlockPublicationReceipt:
        return RawBlockPublicationReceipt("writer", 1, len(keys), "digest")


class _RecordingPersistenceDevice:
    def __init__(self, raw: Any, fail_flush: int = 0) -> None:
        self.raw = raw
        self.fail_flush = fail_flush
        self.flushes = 0
        self.events: list[tuple[str, int]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)

    def pwrite_from_buffer(
        self, offset: int, buf: object, payload_len: int, total_len: int
    ) -> None:
        self.raw.pwrite_from_buffer(offset, buf, payload_len, total_len)
        self.events.append(("write", offset))

    def flush(self) -> None:
        self.flushes += 1
        self.events.append(("flush", self.flushes))
        if self.flushes == self.fail_flush:
            raise OSError("injected persistence failure")
        self.raw.flush()


def test_live_publication_id_cannot_be_reused_after_terminal_history_eviction() -> None:
    """Bounded terminal history must never authorize replacing a live lease."""
    core = _PublicationCore()
    tracker = RawBlockPDRequestTracker(core)  # type: ignore[arg-type]
    try:
        original = tracker.register_batch(
            "held",
            ["old-key"],
            expected_chunks=1,
            is_last_batch=True,
            completed_keys=["old-key"],
        ).result(timeout=5)
        # Failed requests consume terminal-history entries without consuming
        # the admission bound for live publication leases.
        for index in range(4096):
            request_id = f"failed-{index}"
            terminal = tracker.register_batch(
                request_id,
                ["unused"],
                expected_chunks=1,
                is_last_batch=False,
            )
            tracker.fail_request(request_id, RuntimeError("aborted"))
            assert isinstance(terminal.exception(timeout=5), RuntimeError)
        with pytest.raises(RuntimeError, match="no registered batches"):
            tracker.finalize_request("held", expected_chunks=1)
        with pytest.raises(RuntimeError, match="already finished"):
            tracker.register_batch(
                "held",
                ["replacement"],
                expected_chunks=1,
                is_last_batch=True,
                completed_keys=["replacement"],
            )
        assert core.held == ["old-key"]
        assert tracker.live_lease_count() == 1
        released = tracker.release_unread(
            "held", original, expected_writer_epoch="writer"
        )
        assert released.released
        assert core.held == []
    finally:
        tracker.close()


@pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="requires the native raw-block extension",
)
@pytest.mark.parametrize("fail_flush", [0, 1, 2])
def test_publication_orders_persistence_and_never_receipts_a_failed_flush(
    tmp_path: Path, fail_flush: int
) -> None:
    """KV/payload persistence precedes header persistence and receipt delivery."""
    path = make_raw_block_file(tmp_path)
    config = replace(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        close_writes_final_checkpoint=False,
    )
    core = RawBlockCore(config, key_namespace="object")
    raw = core.raw_device()
    recording = _RecordingPersistenceDevice(raw, fail_flush)
    core.set_raw_device_for_testing(recording)
    tracker = RawBlockPDRequestTracker(core)
    key = encode_object_key(make_object_key(71))
    try:
        assert core.put_many([key], [make_memory_obj(b"payload")]).results == [True]
        terminal = tracker.register_batch(
            "request",
            [key.encoded],
            expected_chunks=1,
            is_last_batch=True,
            completed_keys=[key.encoded],
        )
        if fail_flush:
            with pytest.raises(OSError, match="persistence failure"):
                terminal.result(timeout=5)
            assert core.report_status()["metadata_seq"] == 0
            assert tracker.live_lease_count() == 0
        else:
            receipt = terminal.result(timeout=5)
            assert receipt.checkpoint_seq == 1
            assert recording.flushes == 2
            assert tracker.release_unread(
                "request",
                receipt,
                expected_writer_epoch=core.writer_epoch,
            ).released
        metadata_offset = core.metadata_container_offsets()[0]
        checkpoint_events = recording.events[2:]
        expected = [("write", metadata_offset + config.block_align), ("flush", 1)]
        if fail_flush != 1:
            expected += [("write", metadata_offset), ("flush", 2)]
        assert checkpoint_events == expected
        assert core.lock_refcount(key.encoded) == 0
    finally:
        assert tracker.close()
        core.set_raw_device_for_testing(raw)
        core.close()
