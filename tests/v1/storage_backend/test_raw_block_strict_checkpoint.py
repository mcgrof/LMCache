# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import replace
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import RawBlockCore, encode_object_key
from tests.v1.storage_backend.raw_block_test_utils import (
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    make_test_derivation,
)


@pytest.mark.parametrize("field,value", [("version", 2), ("slot_bytes", 131072)])
def test_strict_checkpoint_rejection_preserves_the_live_index(
    tmp_path: Path, field: str, value: int
) -> None:
    """An incompatible checkpoint cannot turn a populated writer into empty space."""
    path = make_raw_block_file(tmp_path)
    core = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
    key = encode_object_key(make_object_key(501))
    try:
        assert core.put_many([key], [make_memory_obj(b"a" * 512)]).results == [True]
        # Construct the same persisted schema used by a live publication.
        state, _ = core._snapshot_state()
        state[field] = value
        with pytest.raises(RuntimeError, match=field + " mismatch"):
            core.apply_loaded_state(state)
        assert core.exists_many([key.encoded]) == [True]
    finally:
        core.close()


def test_strict_writer_refuses_matching_derivation_with_different_geometry(
    tmp_path: Path,
) -> None:
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path, derivation=make_test_derivation())
    writer = RawBlockCore(config, key_namespace="object")
    try:
        key = encode_object_key(make_object_key(502))
        assert writer.put_many([key], [make_memory_obj(b"b" * 512)]).results == [True]
        assert writer.publish_index()
    finally:
        writer.close()

    with pytest.raises(RuntimeError, match="slot_bytes mismatch"):
        RawBlockCore(
            replace(config, slot_bytes=config.slot_bytes * 2), key_namespace="object"
        )


@pytest.mark.parametrize("skip_index", [False, True])
@pytest.mark.parametrize("payload", [b"not-json", b"[]"])
def test_strict_writer_refuses_a_decoded_checkpoint_it_cannot_interpret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skip_index: bool, payload: bytes
) -> None:
    path = make_raw_block_file(tmp_path)
    config = replace(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        load_checkpoint_on_init=not skip_index,
    )
    monkeypatch.setattr(
        RawBlockCore, "_select_latest_checkpoint", lambda self: ({"seq": 7}, payload)
    )
    with pytest.raises(RuntimeError, match="strict raw-block checkpoint"):
        RawBlockCore(config, key_namespace="object")


def test_strict_checkpoint_cannot_hide_an_extent_above_next_slot(
    tmp_path: Path,
) -> None:
    path = make_raw_block_file(tmp_path)
    core = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
    key = encode_object_key(make_object_key(503))
    try:
        assert core.put_many([key], [make_memory_obj(b"c" * 512)]).results == [True]
        state, _ = core._snapshot_state()
        state["next_slot"] = 0
        with pytest.raises(RuntimeError, match="extent is invalid"):
            core.apply_loaded_state(state)
        assert core.exists_many([key.encoded]) == [True]
    finally:
        core.close()


@pytest.mark.parametrize("skip_index", [False, True])
def test_strict_open_refuses_nonblank_unrecognized_metadata(
    tmp_path: Path, skip_index: bool
) -> None:
    path = make_raw_block_file(tmp_path)
    with path.open("r+b") as device:
        device.write(b"another format")
    config = replace(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        load_checkpoint_on_init=not skip_index,
    )
    with pytest.raises(RuntimeError, match="cannot establish an empty namespace"):
        RawBlockCore(config, key_namespace="object")


def test_strict_open_does_not_treat_failed_metadata_reads_as_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path, derivation=make_test_derivation())
    monkeypatch.setattr(RawBlockCore, "_read_buffers", lambda *args, **kwargs: [False])
    with pytest.raises(RuntimeError, match="header read failed"):
        RawBlockCore(config, key_namespace="object")


def test_strict_open_keeps_a_valid_copy_when_the_other_is_torn(tmp_path: Path) -> None:
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path, derivation=make_test_derivation())
    writer = RawBlockCore(config, key_namespace="object")
    key = encode_object_key(make_object_key(504))
    assert writer.put_many([key], [make_memory_obj(b"d" * 512)]).results == [True]
    assert writer.publish_index()
    writer.close()
    # Generation one occupies the first half. Leave it intact and damage
    # the mirror, as a partially written next generation would do.
    with path.open("r+b") as device:
        device.seek(config.meta_total_bytes // 2)
        device.write(b"torn header")
    reader = RawBlockCore(replace(config, role="reader"), key_namespace="object")
    try:
        assert reader.exists_many([key.encoded]) == [True]
    finally:
        reader.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", ["bad"]),
        ("shape", [1 << 63]),
        ("dtype", "not-a-dtype"),
        ("cached_positions", ["bad"]),
        ("cached_positions", [1 << 63]),
        ("cached_positions", [-(1 << 63) - 1]),
        ("fmt", "not-a-format"),
    ],
)
def test_strict_tensor_metadata_is_validated_before_replacing_the_index(
    tmp_path: Path, field: str, value: object
) -> None:
    path = make_raw_block_file(tmp_path)
    core = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
    key = encode_object_key(make_object_key(505))
    try:
        assert core.put_many([key], [make_memory_obj(b"e" * 512)]).results == [True]
        state, _ = core._snapshot_state()
        state["entries"][key.encoded][field] = value
        with pytest.raises(RuntimeError, match="strict raw-block checkpoint"):
            core.apply_loaded_state(state)
        assert core.exists_many([key.encoded]) == [True]
    finally:
        core.close()
