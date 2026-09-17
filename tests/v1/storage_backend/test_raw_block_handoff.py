# SPDX-License-Identifier: Apache-2.0
"""Prefill/decode handoff over one raw-block namespace, and device objects.

A writer core stores KV chunks and publishes its index as an on-device
checkpoint; a reader core on the same device (another process, or another
host over a shared namespace) adopts that index and loads the chunks.  The
reader never writes, and it verifies the slot header it reads so a slot the
writer has since reused is reported as a miss instead of wrong data.
"""

from __future__ import annotations

# Standard
from dataclasses import replace
import importlib.util
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import RawBlockCore, encode_object_key
from tests.v1.storage_backend.raw_block_test_utils import (
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)

requires_rust_raw_block_io = pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io extension is not installed",
)


def _reader_config(path):
    return replace(
        make_raw_block_core_config(path),
        role="reader",
        verify_slot_header_on_load=True,
    )


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_reader_adopts_published_index_and_loads(tmp_path):
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(make_raw_block_core_config(path), key_namespace="object")
    reader = RawBlockCore(_reader_config(path), key_namespace="object")
    try:
        keys = [make_object_key(i) for i in range(3)]
        specs = [encode_object_key(key) for key in keys]
        encoded = [spec.encoded for spec in specs]
        payloads = [bytes([7 + i]) * (1024 * (i + 1)) for i in range(3)]

        # The reader opened before anything was stored: it sees nothing.
        assert reader.exists_many(encoded) == [False, False, False]
        assert reader.refresh_index_from_device() is False

        assert writer.put_many(
            specs, [make_memory_obj(p) for p in payloads]
        ).results == [
            True,
            True,
            True,
        ]
        # Not published yet: the reader still misses.
        assert reader.refresh_index_from_device() is False
        assert reader.exists_many(encoded) == [False, False, False]

        assert writer.publish_index() is True
        assert reader.refresh_index_from_device() is True
        assert reader.exists_many(encoded) == [True, True, True]
        # Nothing new: a second refresh is a no-op.
        assert reader.refresh_index_from_device() is False

        loaded = [make_empty_memory_obj(len(p)) for p in payloads]
        assert reader.load_many_into(encoded, loaded) == [True, True, True]
        assert [memory_obj_bytes(obj) for obj in loaded] == payloads
    finally:
        reader.close()
        writer.close()


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_reader_never_writes(tmp_path):
    path = make_raw_block_file(tmp_path)
    reader = RawBlockCore(_reader_config(path), key_namespace="object")
    try:
        spec = encode_object_key(make_object_key(1))
        with pytest.raises(RuntimeError, match="reader core"):
            reader.put_many([spec], [make_memory_obj(b"x" * 512)])
        with pytest.raises(RuntimeError, match="reader core"):
            reader.delete_many([spec.encoded])
        assert reader.publish_index() is False
    finally:
        reader.close()
    # Closing a reader writes no checkpoint: a writer opening afterwards
    # finds no metadata and starts empty.
    writer = RawBlockCore(make_raw_block_core_config(path), key_namespace="object")
    try:
        assert writer.indexed_key_count() == 0
    finally:
        writer.close()


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_reader_rejects_slot_the_writer_reused(tmp_path):
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(make_raw_block_core_config(path), key_namespace="object")
    reader = RawBlockCore(_reader_config(path), key_namespace="object")
    try:
        first = encode_object_key(make_object_key(1))
        second = encode_object_key(make_object_key(2))
        assert writer.put_many([first], [make_memory_obj(b"a" * 1024)]).results == [
            True
        ]
        assert writer.publish_index()
        assert reader.refresh_index_from_device()
        assert reader.exists_many([first.encoded]) == [True]

        # The writer drops the first key and stores another one; the freed
        # slot is reused for it.  The reader still holds the old index.
        assert writer.delete_many([first.encoded]) == [True]
        assert writer.put_many([second], [make_memory_obj(b"b" * 1024)]).results == [
            True
        ]

        loaded = make_empty_memory_obj(1024)
        assert reader.load_many_into([first.encoded], [loaded]) == [False]

        # After the writer publishes again the reader follows the change.
        assert writer.publish_index()
        assert reader.refresh_index_from_device()
        assert reader.exists_many([first.encoded, second.encoded]) == [False, True]
        assert reader.load_many_into([second.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == b"b" * 1024
    finally:
        reader.close()
        writer.close()


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_reader_follows_a_writer_that_restarted_empty(tmp_path):
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(make_raw_block_core_config(path), key_namespace="object")
    first = encode_object_key(make_object_key(1))
    second = encode_object_key(make_object_key(2))
    # Publish several times so the device holds a high sequence number.
    for i in range(3):
        spec = encode_object_key(make_object_key(100 + i))
        writer.put_many([spec], [make_memory_obj(b"o" * 512)])
        assert writer.publish_index()
    assert writer.put_many([first], [make_memory_obj(b"a" * 1024)]).results == [True]
    assert writer.publish_index()
    writer.close()

    reader = RawBlockCore(_reader_config(path), key_namespace="object")
    fresh = None
    try:
        assert reader.exists_many([first.encoded]) == [True]
        # The writer comes back without loading its old index.
        fresh = RawBlockCore(
            replace(make_raw_block_core_config(path), load_checkpoint_on_init=False),
            key_namespace="object",
        )
        assert fresh.put_many([second], [make_memory_obj(b"b" * 1024)]).results == [
            True
        ]
        assert fresh.publish_index()
        assert reader.refresh_index_from_device() is True
        assert reader.exists_many([first.encoded, second.encoded]) == [False, True]
        loaded = make_empty_memory_obj(1024)
        assert reader.load_many_into([second.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == b"b" * 1024
    finally:
        reader.close()
        if fresh is not None:
            fresh.close()


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_publish_is_spaced_by_min_interval(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = replace(make_raw_block_core_config(path), publish_min_interval_ms=60_000)
    writer = RawBlockCore(config, key_namespace="object")
    try:
        spec = encode_object_key(make_object_key(1))
        assert writer.put_many([spec], [make_memory_obj(b"a" * 1024)]).results == [True]
        assert writer.publish_index() is True
        spec2 = encode_object_key(make_object_key(2))
        assert writer.put_many([spec2], [make_memory_obj(b"b" * 1024)]).results == [
            True
        ]
        # Dirty again, but inside the minimum spacing.
        assert writer.publish_index() is False
    finally:
        writer.close()
