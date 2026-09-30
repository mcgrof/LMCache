# SPDX-License-Identifier: Apache-2.0
"""Prefill/decode handoff over one raw-block namespace, and device objects.

A writer core stores KV chunks and publishes its index as an on-device
checkpoint; a reader core on the same device (another process, or another
host over a shared namespace) adopts that index and loads the chunks.  The
reader never writes, and it verifies the slot header it reads so a slot the
writer has since reused is reported as a miss instead of wrong data.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import replace
import importlib.util
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import (
    IncompatibleKeyDerivation,
    RawBlockCore,
    RawBlockDerivationDescriptor,
    encode_object_key,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    make_test_derivation,
    memory_obj_bytes,
)

requires_rust_raw_block_io = pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io extension is not installed",
)


def _reader_config(path):
    return replace(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        role="reader",
        verify_slot_header_on_load=True,
    )


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_reader_adopts_published_index_and_loads(tmp_path):
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
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
    writer = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
    try:
        assert writer.indexed_key_count() == 0
    finally:
        writer.close()


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_reader_rejects_slot_the_writer_reused(tmp_path):
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
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
    writer = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
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
            replace(
                make_raw_block_core_config(path, derivation=make_test_derivation()),
                load_checkpoint_on_init=False,
            ),
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
    config = replace(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        publish_min_interval_ms=60_000,
    )
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


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_request_receipt_adopts_exact_and_compatible_later_generation(tmp_path):
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
    reader = RawBlockCore(_reader_config(path), key_namespace="object")
    try:
        first = encode_object_key(make_object_key(1))
        second = encode_object_key(make_object_key(2))
        assert writer.put_many([first], [make_memory_obj(b"a" * 1024)]).results == [
            True
        ]

        first_receipt = writer.publish_request([first.encoded])
        assert reader.refresh_until_publication(
            first_receipt,
            [first.encoded],
            timeout_ms=1_000,
            refresh_interval_ms=1,
        )

        assert writer.put_many([second], [make_memory_obj(b"b" * 1024)]).results == [
            True
        ]
        second_receipt = writer.publish_request([second.encoded])
        assert second_receipt.checkpoint_seq > first_receipt.checkpoint_seq
        assert reader.refresh_until_publication(
            second_receipt,
            [second.encoded],
            timeout_ms=1_000,
            refresh_interval_ms=1,
        )
        assert reader.publication_matches(first_receipt, [first.encoded])

        loaded = make_empty_memory_obj(1024)
        assert reader.load_many_into([first.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == b"a" * 1024
    finally:
        reader.close()
        writer.close()


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_request_receipt_fences_a_restarted_writer(tmp_path):
    path = make_raw_block_file(tmp_path)
    first = encode_object_key(make_object_key(1))
    writer = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
    assert writer.put_many([first], [make_memory_obj(b"a" * 1024)]).results == [True]
    stale_receipt = writer.publish_request([first.encoded])
    writer.close()

    reader = RawBlockCore(_reader_config(path), key_namespace="object")
    restarted = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
    try:
        fresh_receipt = restarted.publish_request([first.encoded])
        assert fresh_receipt.writer_epoch != stale_receipt.writer_epoch
        assert reader.refresh_until_publication(
            fresh_receipt,
            [first.encoded],
            timeout_ms=1_000,
            refresh_interval_ms=1,
        )
        assert not reader.publication_matches(stale_receipt, [first.encoded])
    finally:
        restarted.close()
        reader.close()


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_request_publication_rejects_a_missing_key(tmp_path):
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
    try:
        missing = encode_object_key(make_object_key(1))
        with pytest.raises(RuntimeError, match="uncommitted key"):
            writer.publish_request([missing.encoded])
    finally:
        writer.close()


def test_namespace_identity_of_a_block_device_without_hardware_identity():
    """A namespace exposing no hardware identity still resolves locally.

    Requiring a persistent identity at construction turned every block
    target without the selected sysfs attributes into a startup failure,
    including ordinary local caches that never publish anything to
    another node.
    """
    # Standard
    from unittest import mock
    import os
    import stat as stat_module

    # First Party
    from lmcache.v1.storage_backend.raw_block import core as core_module

    # Device numbers no namespace claims, so no sysfs directory exists.
    major, minor = 4095, 4095
    fake = os.stat_result(
        (stat_module.S_IFBLK | 0o660, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        {"st_rdev": os.makedev(major, minor)},
    )
    with mock.patch.object(core_module.os, "stat", return_value=fake):
        identity = core_module._resolve_namespace_identity("/dev/does-not-exist")

    assert identity == f"block-local:{major}:{minor}"
    assert not core_module.namespace_identity_is_shareable(identity)


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_publication_refuses_an_identity_another_node_cannot_resolve(tmp_path):
    """Sharing is where a persistent identity actually matters.

    Device numbers are assigned by the local kernel, so a receipt built
    on them would name a different device on the node that reads it.
    """
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
    try:
        key = encode_object_key(make_object_key(1))
        assert writer.put_many([key], [make_memory_obj(b"a" * 1024)]).results == [True]

        writer.namespace_identity = "block-local:4095:4095"
        with pytest.raises(ValueError, match="persistent block namespace identity"):
            writer.publish_request([key.encoded])
    finally:
        writer.close()


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_publication_does_not_reuse_a_generation_that_moved_the_key(tmp_path):
    """A receipt must name where a key is now, not where it used to be.

    A key can be published at one extent, deleted, and written again
    somewhere else. The earlier checkpoint still advertises the extent it
    had then, and the key is still a member of that generation, so a
    publication that decides by membership alone hands back a receipt
    pointing at storage the writer has since given to another key. The
    holds the request takes protect the new extent, not the advertised one.
    """
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
    reader = RawBlockCore(_reader_config(path), key_namespace="object")
    try:
        first = encode_object_key(make_object_key(81))
        second = encode_object_key(make_object_key(82))

        assert writer.put_many([first], [make_memory_obj(b"A" * 1024)]).results == [
            True
        ]
        old_receipt = writer.publish_request([first.encoded])
        old_offset = writer._index[first.encoded].offset

        # Drop the key, let another one take its slot, then store it again so
        # it lands somewhere else.
        assert writer.delete_many([first.encoded]) == [True]
        assert writer.put_many([second], [make_memory_obj(b"B" * 1024)]).results == [
            True
        ]
        assert writer._index[second.encoded].offset == old_offset
        assert writer.put_many([first], [make_memory_obj(b"C" * 1024)]).results == [
            True
        ]
        new_offset = writer._index[first.encoded].offset
        assert new_offset != old_offset

        new_receipt = writer.publish_request([first.encoded])

        # A fresh generation, not the one that described the old extent.
        assert new_receipt.checkpoint_seq > old_receipt.checkpoint_seq

        # And a reader following the new receipt reads the key's current bytes,
        # not whatever now lives at the extent the old receipt advertised.
        assert reader.refresh_until_publication(
            new_receipt,
            [first.encoded],
            timeout_ms=1_000,
            refresh_interval_ms=1,
        )
        loaded = make_empty_memory_obj(1024)
        assert reader.load_many_into([first.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == b"C" * 1024

        # The fully deduplicated case still yields a current generation.
        repeat = writer.publish_request([first.encoded])
        assert repeat.checkpoint_seq > new_receipt.checkpoint_seq
    finally:
        reader.close()
        writer.close()


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_a_reader_refuses_a_namespace_derived_differently(tmp_path):
    """Two engines can agree on the layout and still read nothing of each
    other's.

    A different hash function, seed, chain root or key encoding changes the
    bytes of every key while leaving the geometry identical. The existing
    geometry checks answer a mismatch by ignoring the metadata and starting
    empty, which is the right answer for a layout this engine cannot read and
    the wrong one here: it means writing our own keys beside someone else's
    and reporting a cold cache instead of a misconfiguration.
    """
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(
        make_raw_block_core_config(path, derivation=make_test_derivation()),
        key_namespace="object",
    )
    try:
        keys = [make_object_key(i) for i in range(2)]
        specs = [encode_object_key(key) for key in keys]
        objs = [make_memory_obj(f"value-{i}".encode()) for i in range(2)]
        assert writer.put_many(specs, objs).results == [True, True]
        writer.publish_request([spec.encoded for spec in specs])
    finally:
        writer.close()

    # Refused while opening, before a single entry is adopted: the checkpoint
    # is read at construction, which is the first moment the two derivations
    # can be compared.
    theirs = replace(make_test_derivation(), chain_root="99")
    with pytest.raises(IncompatibleKeyDerivation, match="chain_root"):
        RawBlockCore(
            replace(
                make_raw_block_core_config(path, derivation=theirs),
                role="reader",
            ),
            key_namespace="object",
        )


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_a_reader_refuses_a_namespace_that_states_no_derivation(tmp_path):
    """Silence is as incompatible as disagreement.

    A writer that said nothing about its derivation named no contract, so
    nothing can be concluded about the keys already there.
    """
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(make_raw_block_core_config(path), key_namespace="object")
    try:
        keys = [make_object_key(i) for i in range(2)]
        specs = [encode_object_key(key) for key in keys]
        objs = [make_memory_obj(f"value-{i}".encode()) for i in range(2)]
        assert writer.put_many(specs, objs).results == [True, True]
        writer.checkpoint_now()
    finally:
        writer.close()

    with pytest.raises(IncompatibleKeyDerivation, match="no key derivation"):
        RawBlockCore(
            replace(
                make_raw_block_core_config(path, derivation=make_test_derivation()),
                role="reader",
            ),
            key_namespace="object",
        )


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_publication_refuses_without_a_stated_derivation(tmp_path):
    """Publishing is telling another engine to read these keys."""
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(make_raw_block_core_config(path), key_namespace="object")
    try:
        spec = encode_object_key(make_object_key(0))
        assert writer.put_many([spec], [make_memory_obj(b"value")]).results == [True]
        with pytest.raises(IncompatibleKeyDerivation, match="how they were derived"):
            writer.publish_request([spec.encoded])
    finally:
        writer.close()


def test_a_derivation_names_every_field_it_disagrees_on() -> None:
    """A refusal has to say which field differs, or it cannot be acted on."""
    ours = make_test_derivation()
    theirs = replace(ours, hash_algorithm="builtin", chain_root="7")
    mismatches = ours.describe_mismatch(theirs)
    assert len(mismatches) == 2
    assert any("hash_algorithm" in item for item in mismatches)
    assert any("chain_root" in item for item in mismatches)
    assert ours.describe_mismatch(ours) == []
    # And a descriptor survives the round trip it is stored through.
    assert RawBlockDerivationDescriptor.from_payload(ours.as_payload()) == ours
    assert RawBlockDerivationDescriptor.from_payload(None) is None
    assert RawBlockDerivationDescriptor.from_payload({"hash_algorithm": "x"}) is None


def test_the_hash_seed_is_compared_only_where_it_can_reach_a_key() -> None:
    """Comparing the seed unconditionally invents an availability failure.

    A cryptographic hash does not consult the process seed, and neither does
    the interpreter's for a key of integers: measured on this interpreter,
    ``hash((0, (1, 2, 3), ()))`` is identical under seeds 0, 12345 and 99.
    Refusing a namespace two nodes derive identically is a failure the check
    would have created rather than found.
    """
    cryptographic = make_test_derivation()
    other_seed = replace(cryptographic, hash_seed="12345")
    assert "hash_seed" not in cryptographic.compared_fields(other_seed)
    assert cryptographic.describe_mismatch(other_seed) == []

    # The interpreter's own hash is the one case where a seed can reach a
    # key, through a string, so there it is compared.
    interpreter = replace(
        cryptographic,
        hash_algorithm="builtin",
        hash_implementation="builtins.hash",
    )
    interpreter_other_seed = replace(interpreter, hash_seed="12345")
    assert "hash_seed" in interpreter.compared_fields(interpreter_other_seed)
    assert any(
        "hash_seed" in item
        for item in interpreter.describe_mismatch(interpreter_other_seed)
    )
    # And either side naming it is enough to bring the seed into the
    # comparison, since one of them is hashing that way.
    assert "hash_seed" in cryptographic.compared_fields(interpreter)


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_a_core_refuses_a_device_keyed_in_another_namespace(tmp_path):
    """A namespace mismatch recycled live data, and both answers were unsafe.

    A slot identity is derived from the encoded key and the namespace, so a
    core reading with a different namespace computes a different identity for
    every entry, reads each header as stale, drops the entry and returns its
    extent. The next allocation then lands on live data the other core's
    checkpoint still advertises -- measured, at exactly the same offset.

    Ignoring the metadata instead is no safer: the slot counter stays at zero
    and allocation starts from the bottom of the same region.
    """
    path = make_raw_block_file(tmp_path)
    writer = RawBlockCore(make_raw_block_core_config(path), key_namespace="object")
    try:
        spec = encode_object_key(make_object_key(0))
        assert writer.put_many([spec], [make_memory_obj(b"live")]).results == [True]
        writer.checkpoint_now()
        occupied = int(writer._index[spec.encoded].offset)
    finally:
        writer.close()

    with pytest.raises(IncompatibleKeyDerivation, match="keyed in namespace"):
        RawBlockCore(make_raw_block_core_config(path), key_namespace="legacy")

    # And the namespace it does belong to still opens and still holds it.
    reopened = RawBlockCore(make_raw_block_core_config(path), key_namespace="object")
    try:
        assert reopened.contains_key(spec.encoded)
        assert int(reopened._index[spec.encoded].offset) == occupied
    finally:
        reopened.close()
