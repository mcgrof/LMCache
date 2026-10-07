# SPDX-License-Identifier: Apache-2.0
"""A checkpoint's key namespace must not turn live extents into stale slots."""

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import (
    IncompatibleKeyDerivation,
    RawBlockCore,
    encode_object_key,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)


def test_recovery_refuses_another_namespace_without_overwriting_payload(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    key = encode_object_key(make_object_key(99))
    payload = b"the original cache must survive a refused open"
    writer = RawBlockCore(config, key_namespace="object")
    assert writer.put_many([key], [make_memory_obj(payload)]).results == [True]
    offset = writer.entry_offset(key.encoded)
    writer.close()

    with pytest.raises(IncompatibleKeyDerivation, match="namespace"):
        RawBlockCore(config, key_namespace="legacy")

    reopened = RawBlockCore(config, key_namespace="object")
    try:
        target = make_memory_obj(bytes(len(payload)))
        assert reopened.entry_offset(key.encoded) == offset
        assert reopened.load_many_into([key.encoded], [target]) == [True]
        assert memory_obj_bytes(target) == payload
    finally:
        reopened.close()


def test_namespace_refusal_leaves_running_index_and_old_metadata_compatible(tmp_path):
    path = make_raw_block_file(tmp_path)
    core = RawBlockCore(make_raw_block_core_config(path), key_namespace="object")
    key = encode_object_key(make_object_key(100))
    try:
        assert core.put_many([key], [make_memory_obj(b"original")]).results == [True]
        snapshot, _ = core._snapshot_state()
        assert snapshot["key_namespace"] == "object"
        offset = core.entry_offset(key.encoded)
        snapshot["key_namespace"] = "legacy"
        with pytest.raises(IncompatibleKeyDerivation, match="namespace"):
            core.apply_loaded_state(snapshot)
        assert core.entry_offset(key.encoded) == offset
        del snapshot["key_namespace"]
        assert core.apply_loaded_state(snapshot)
        assert core.entry_offset(key.encoded) == offset
    finally:
        core.close()
