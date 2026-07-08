# SPDX-License-Identifier: Apache-2.0
"""
CO3: codec-scheme domain separation in split-tier child keys.

A scale-aware (B / ``COMPUTED_PER_TENSOR``) V blob and a byte-through
(C / ``RAW_UNIT``) V blob for the SAME logical chunk must not collide on
one L2 key -- otherwise one silently overwrites/mis-serves the other, and
the codec-scheme mismatch is only caught at *decode*, i.e. after the
original object is already destroyed.

``derive_component_key`` gains a ``scheme``:

* ``COMPUTED_LEGACY`` (default) -> the historical role-only key, so every
  already-stored object and every existing caller is byte-identical.
* ``RAW_UNIT`` -> a magic-tagged key (``logical + MAGIC + scheme + role``)
  that can never reuse a legacy key.  The role byte is kept LAST so a
  reader predating this format fails closed (yields a bogus parent) rather
  than aliasing a real logical key.
"""

# Future
from __future__ import annotations

# Standard
import hashlib

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _filename_to_object_key,
    _object_key_to_filename,
)
from lmcache.v1.distributed.storage_placement import (
    _K_CHILD_MARKER,
    _SCHEME_MARKER_MAGIC,
    _V_CHILD_MARKER,
    ComponentKeyScheme,
    decode_component_key,
    derive_component_key,
    reverse_component_key,
)

_LEGACY = ComponentKeyScheme.COMPUTED_LEGACY
_RAW = ComponentKeyScheme.RAW_UNIT


def _logical(seed: bytes = b"seed", **overrides: object) -> ObjectKey:
    kwargs: dict[str, object] = dict(
        chunk_hash=hashlib.sha256(seed).digest(),  # 32 bytes
        model_name="test-model",
        kv_rank=0,
    )
    kwargs.update(overrides)
    return ObjectKey(**kwargs)  # type: ignore[arg-type]


def _legacy_reverse_strip(child: ObjectKey) -> ObjectKey:
    """Simulate a pre-CO3 reader: strip only the trailing role byte."""
    return ObjectKey(
        chunk_hash=child.chunk_hash[:-1],
        model_name=child.model_name,
        kv_rank=child.kv_rank,
        object_group_id=child.object_group_id,
        cache_salt=child.cache_salt,
    )


# =============================================================================
# B / C non-collision (the whole point)
# =============================================================================


def test_computed_and_raw_unit_v_keys_never_collide() -> None:
    logical = _logical()
    b_v = derive_component_key(logical, "v", _LEGACY)
    c_v = derive_component_key(logical, "v", _RAW)
    assert b_v != c_v
    assert b_v.chunk_hash != c_v.chunk_hash


def test_default_scheme_is_legacy_byte_identical() -> None:
    """No scheme arg == COMPUTED_LEGACY == the pre-CO3 key exactly."""
    logical = _logical()
    assert derive_component_key(logical, "v") == derive_component_key(
        logical, "v", _LEGACY
    )
    # Legacy key is exactly logical_hash + role marker, nothing more.
    assert (
        derive_component_key(logical, "v").chunk_hash
        == logical.chunk_hash + _V_CHILD_MARKER
    )
    assert (
        derive_component_key(logical, "k").chunk_hash
        == logical.chunk_hash + _K_CHILD_MARKER
    )


def test_raw_unit_key_has_magic_and_role_last() -> None:
    logical = _logical()
    c_v = derive_component_key(logical, "v", _RAW)
    expected = logical.chunk_hash + _SCHEME_MARKER_MAGIC + b"\x01" + _V_CHILD_MARKER
    assert c_v.chunk_hash == expected
    # role byte is LAST (old-reader fail-closed property depends on this).
    assert c_v.chunk_hash[-1:] == _V_CHILD_MARKER


def test_raw_unit_k_and_v_children_distinct() -> None:
    logical = _logical()
    assert derive_component_key(logical, "k", _RAW) != derive_component_key(
        logical, "v", _RAW
    )


def test_identity_fields_preserved_under_raw_unit() -> None:
    logical = _logical(model_name="m@bad".replace("@", "-"), kv_rank=7)
    logical = ObjectKey(
        chunk_hash=logical.chunk_hash,
        model_name="tenant-model",
        kv_rank=7,
        object_group_id=3,
        cache_salt="tenant-42",
    )
    c_v = derive_component_key(logical, "v", _RAW)
    assert c_v.model_name == "tenant-model"
    assert c_v.kv_rank == 7
    assert c_v.object_group_id == 3
    assert c_v.cache_salt == "tenant-42"


# =============================================================================
# decode_component_key round-trips (scheme recovered)
# =============================================================================


@pytest.mark.parametrize("role", ["k", "v"])
@pytest.mark.parametrize("scheme", [_LEGACY, _RAW])
def test_decode_round_trip(role: str, scheme: ComponentKeyScheme) -> None:
    logical = _logical(object_group_id=5, cache_salt="t")  # type: ignore[call-arg]
    child = derive_component_key(logical, role, scheme)
    decoded = decode_component_key(child)
    assert decoded is not None
    got_logical, got_role, got_scheme = decoded
    assert got_logical == logical
    assert got_role == role
    assert got_scheme == scheme


def test_reverse_component_key_two_tuple_handles_both_formats() -> None:
    logical = _logical()
    for scheme in (_LEGACY, _RAW):
        for role in ("k", "v"):
            child = derive_component_key(logical, role, scheme)
            result = reverse_component_key(child)
            assert result == (logical, role)


def test_decode_returns_none_for_logical_key() -> None:
    # A logical hash that does not end in a role marker byte.
    logical = ObjectKey(chunk_hash=b"\x00" * 32, model_name="m", kv_rank=0)
    assert decode_component_key(logical) is None
    assert reverse_component_key(logical) is None


# =============================================================================
# Old-reader fail-closed + malformed-magic fail-closed
# =============================================================================


def test_old_reader_of_raw_unit_key_yields_bogus_parent() -> None:
    """A pre-CO3 reader strips only the trailing role byte.  Applied to a
    RAW_UNIT key it must NOT recover the real logical key -- it gets
    logical+MAGIC+scheme, which matches no real object (fail closed)."""
    logical = _logical()
    c_v = derive_component_key(logical, "v", _RAW)
    bogus = _legacy_reverse_strip(c_v)
    assert bogus.chunk_hash != logical.chunk_hash
    assert bogus.chunk_hash == logical.chunk_hash + _SCHEME_MARKER_MAGIC + b"\x01"


def test_malformed_magic_scheme_byte_fails_closed() -> None:
    logical = _logical()
    bad = ObjectKey(
        chunk_hash=logical.chunk_hash
        + _SCHEME_MARKER_MAGIC
        + b"\x09"
        + _V_CHILD_MARKER,
        model_name=logical.model_name,
        kv_rank=logical.kv_rank,
    )
    assert decode_component_key(bad) is None
    assert reverse_component_key(bad) is None


def test_magic_claiming_legacy_scheme_fails_closed() -> None:
    """The magic form must never carry the legacy scheme byte (0x00);
    legacy keys use the role-only format."""
    logical = _logical()
    bad = ObjectKey(
        chunk_hash=logical.chunk_hash
        + _SCHEME_MARKER_MAGIC
        + b"\x00"
        + _V_CHILD_MARKER,
        model_name=logical.model_name,
        kv_rank=logical.kv_rank,
    )
    assert decode_component_key(bad) is None


def test_magic_with_bad_role_byte_fails_closed() -> None:
    logical = _logical()
    bad = ObjectKey(
        chunk_hash=logical.chunk_hash + _SCHEME_MARKER_MAGIC + b"\x01" + b"\x7f",
        model_name=logical.model_name,
        kv_rank=logical.kv_rank,
    )
    assert decode_component_key(bad) is None


def test_suffix_only_key_does_not_decode() -> None:
    """A key that is *only* a magic suffix (no real hash byte) must not
    decode as a child."""
    only_suffix = ObjectKey(
        chunk_hash=_SCHEME_MARKER_MAGIC + b"\x01" + _V_CHILD_MARKER,
        model_name="m",
        kv_rank=0,
    )
    assert decode_component_key(only_suffix) is None


def test_short_hash_does_not_decode() -> None:
    assert (
        decode_component_key(ObjectKey(chunk_hash=b"", model_name="m", kv_rank=0))
        is None
    )
    assert (
        decode_component_key(ObjectKey(chunk_hash=b"\x02", model_name="m", kv_rank=0))
        is None
    )


# =============================================================================
# Marker bytes survive the FS filename encoder (cmcp: prove the adapter path)
# =============================================================================


def test_raw_unit_key_survives_fs_filename_round_trip() -> None:
    logical = _logical(cache_salt="tenant-9")  # type: ignore[call-arg]
    c_v = derive_component_key(logical, "v", _RAW)
    fname = _object_key_to_filename(c_v)
    back = _filename_to_object_key(fname)
    assert back == c_v
    # And the recovered key still decodes to the same logical/role/scheme.
    assert decode_component_key(back) == (logical, "v", _RAW)
