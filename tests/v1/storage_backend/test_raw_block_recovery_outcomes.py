# SPDX-License-Identifier: Apache-2.0
"""Unreadable checkpoint entries must not become allocator-free extents."""

# Standard
from dataclasses import replace

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import RawBlockCore, encode_object_key
from tests.v1.storage_backend.raw_block_test_utils import (
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
)

pytest.importorskip("lmcache_rust_raw_block_io")


@pytest.fixture
def stored_core(tmp_path):
    path = make_raw_block_file(tmp_path)
    core = RawBlockCore(make_raw_block_core_config(path), key_namespace="object")
    keys = [encode_object_key(make_object_key(i)) for i in range(5)]
    assert core.put_many(keys, [make_memory_obj(b"kept")] * 5).results == [True] * 5
    raw = core.raw_device()
    try:
        yield core, keys, path
    finally:
        core.close()
        # Only test doubles issued the failed reads, so the actual native
        # device is idle even when the core deliberately retains its proxy.
        raw.close()


@pytest.mark.parametrize("engine", ["posix", "io_uring"])
def test_unreadable_recovery_keeps_entries_and_extents(
    stored_core, monkeypatch, engine
):
    core, keys, _ = stored_core
    offsets = [core.entry_offset(key.encoded) for key in keys]
    core.io_engine = engine
    monkeypatch.setattr(core, "_read_buffers", lambda *args: [False] * len(args[0]))
    core._validate_loaded_entries()
    assert [core.entry_offset(key.encoded) for key in keys] == offsets
    assert core.indexed_key_count() == 5
    assert core._free_slots == {}


@pytest.mark.parametrize("engine", ["posix", "io_uring"])
def test_known_invalid_headers_still_recycle(stored_core, monkeypatch, engine):
    core, keys, _ = stored_core
    old_offsets = {core.entry_offset(key.encoded) for key in keys}
    core.io_engine = engine

    def invalid(offsets, buffers, *args):
        for buffer in buffers:
            buffer[:] = bytes(len(buffer))
        return [True] * len(offsets)

    with monkeypatch.context() as failure:
        failure.setattr(core, "_read_buffers", invalid)
        core._validate_loaded_entries()
    assert core.indexed_key_count() == 0
    assert len(core._free_slots) == 5
    core.io_engine = "posix"
    replacement = encode_object_key(make_object_key(77))
    assert core.put_many([replacement], [make_memory_obj(b"new")]).results == [True]
    assert core.entry_offset(replacement.encoded) in old_offsets


def test_unknown_recovery_stops_before_another_batch(stored_core, monkeypatch):
    core, keys, _ = stored_core
    raw = core.raw_device()
    core.io_engine = "io_uring"
    core.iouring_queue_depth = 2
    calls = []

    class OutcomeProxy:
        unknown = False

        def __getattr__(self, name):
            return getattr(raw, name)

        def is_poisoned(self):
            return self.unknown

    proxy = OutcomeProxy()
    core.set_raw_device_for_testing(proxy)

    def unknown(offsets, *args):
        calls.append(list(offsets))
        proxy.unknown = True
        raise RuntimeError("completion is unknown")

    monkeypatch.setattr(core, "_read_buffers", unknown)
    core._validate_loaded_entries()
    assert len(calls) == 1 and len(calls[0]) == 2
    assert core.is_poisoned()
    assert core.indexed_key_count() == len(keys)
    assert core._free_slots == {}


@pytest.mark.parametrize("engine", ["posix", "io_uring"])
def test_fresh_reopen_does_not_recycle_an_unreadable_checkpoint(
    stored_core, monkeypatch, engine
):
    writer, keys, path = stored_core
    offsets = [writer.entry_offset(key.encoded) for key in keys]
    writer.close()
    read = RawBlockCore._read_buffers

    def unreadable_headers(self, offsets, *args):
        if all(offset >= self.meta_total_bytes for offset in offsets):
            return [False] * len(offsets)
        return read(self, offsets, *args)

    monkeypatch.setattr(RawBlockCore, "_read_buffers", unreadable_headers)
    recovered = RawBlockCore(
        replace(make_raw_block_core_config(path), io_engine=engine),
        key_namespace="object",
    )
    try:
        assert [recovered.entry_offset(key.encoded) for key in keys] == offsets
        assert recovered.indexed_key_count() == 5
        assert recovered._free_slots == {}
    finally:
        recovered.close()
