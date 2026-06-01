# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the storage placement policy
(``lmcache/v1/distributed/storage_placement.py``).

Covers:

* :class:`StoragePlacementMode` resolution from configured serdes.
* :func:`derive_component_key` round-trip + invariants
  (preserves ``model_name`` / ``kv_rank`` / ``cache_salt``; child
  keys never collide with the logical key or with each other; cross
  role / logical-key collision impossible).
* :class:`SplitTierState` linear-forward enumeration.
* Mixed-placement rejection (V-only + asym Mode 1 on the same
  StorageManager).
"""

# Future
from __future__ import annotations

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_placement import (
    SplitTierState,
    StoragePlacementMode,
    derive_component_key,
    derive_storage_placement_mode,
)


# =============================================================================
# Helpers
# =============================================================================


class _FakeAdapterCfg:
    """Minimal stand-in for ``L2AdapterConfigBase`` exposing
    ``serde_config`` only -- which is all the placement resolver looks at."""

    def __init__(self, serde_config) -> None:
        self.serde_config = serde_config


def _make_key(chunk_hash: bytes = b"\x00" * 32, *, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name="test-model",
        kv_rank=0,
        cache_salt=salt,
    )


# =============================================================================
# derive_component_key
# =============================================================================


def test_derive_component_key_preserves_model_and_rank() -> None:
    logical = ObjectKey(
        chunk_hash=b"\x00" * 32,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        kv_rank=7,
        cache_salt="",
    )
    k = derive_component_key(logical, "k")
    v = derive_component_key(logical, "v")
    for child in (k, v):
        assert child.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert child.kv_rank == 7


def test_derive_component_key_preserves_cache_salt() -> None:
    """cache_salt drives per-tenant quota accounting -- the child
    keys must inherit it verbatim (codex: do NOT mutate)."""
    logical = _make_key(salt="tenant-42")
    assert derive_component_key(logical, "k").cache_salt == "tenant-42"
    assert derive_component_key(logical, "v").cache_salt == "tenant-42"


def test_derive_component_key_extends_chunk_hash() -> None:
    """Child chunk_hash is logical chunk_hash + single role marker."""
    logical = _make_key(chunk_hash=b"\xab" * 32)
    k = derive_component_key(logical, "k")
    v = derive_component_key(logical, "v")
    assert k.chunk_hash == b"\xab" * 32 + b"\x01"
    assert v.chunk_hash == b"\xab" * 32 + b"\x02"


def test_derive_component_key_k_v_never_collide() -> None:
    """K and V child keys have distinct role markers -> distinct hashes."""
    logical = _make_key()
    assert derive_component_key(logical, "k") != derive_component_key(logical, "v")


def test_derive_component_key_never_collides_with_logical_key() -> None:
    """Length differs (logical is 32 bytes; children are 33) so the
    keys can never collide regardless of the original hash content."""
    logical = _make_key(chunk_hash=b"\xff" * 32)
    k = derive_component_key(logical, "k")
    v = derive_component_key(logical, "v")
    assert len(k.chunk_hash) == 33
    assert len(v.chunk_hash) == 33
    assert k.chunk_hash != logical.chunk_hash
    assert v.chunk_hash != logical.chunk_hash


def test_derive_component_key_rejects_unknown_role() -> None:
    logical = _make_key()
    with pytest.raises(ValueError, match="role must be 'k' or 'v'"):
        derive_component_key(logical, "scale")
    with pytest.raises(ValueError):
        derive_component_key(logical, "")


def test_derive_component_key_deterministic() -> None:
    """Same logical key + role -> same child key, every time."""
    logical = _make_key(chunk_hash=b"\x11" * 32, salt="t")
    assert derive_component_key(logical, "k") == derive_component_key(logical, "k")
    assert derive_component_key(logical, "v") == derive_component_key(logical, "v")


# =============================================================================
# derive_storage_placement_mode
# =============================================================================


def test_derive_placement_mode_empty_returns_kv_together() -> None:
    """No adapters -> default kv_together."""
    assert (
        derive_storage_placement_mode([]) == StoragePlacementMode.KV_TOGETHER
    )


def test_derive_placement_mode_no_serde_returns_kv_together() -> None:
    cfgs = [_FakeAdapterCfg(serde_config=None)]
    assert (
        derive_storage_placement_mode(cfgs)
        == StoragePlacementMode.KV_TOGETHER
    )


def test_derive_placement_mode_fp8_returns_kv_together() -> None:
    """Single-tensor serdes (no slot mapping) place K+V together."""
    cfgs = [_FakeAdapterCfg(serde_config=SerdeConfig(type="fp8"))]
    assert (
        derive_storage_placement_mode(cfgs)
        == StoragePlacementMode.KV_TOGETHER
    )


def test_derive_placement_mode_asym_mode_1_returns_kv_together() -> None:
    """asym_k16_v8 storage-only: identity mapping (0, 1), no None
    slots -> both children in one blob (kv_together)."""
    cfgs = [_FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8"))]
    assert (
        derive_storage_placement_mode(cfgs)
        == StoragePlacementMode.KV_TOGETHER
    )


def test_derive_placement_mode_asym_v_only_returns_kv_split_tier() -> None:
    """asym_k16_v8_v_only: mapping (None, 1), so slot 0 (K) is
    absent from the L2 path -> split-tier placement."""
    cfgs = [_FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8_v_only"))]
    assert (
        derive_storage_placement_mode(cfgs)
        == StoragePlacementMode.KV_SPLIT_TIER
    )


def test_derive_placement_mode_mixed_rejected() -> None:
    """V-only + asym Mode 1 on the same StorageManager demand
    incompatible lifecycles -- reject at config time."""
    cfgs = [
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8")),
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8_v_only")),
    ]
    with pytest.raises(ValueError, match="Incompatible L2 adapter storage placement"):
        derive_storage_placement_mode(cfgs)


def test_derive_placement_mode_mixed_no_serde_and_v_only_rejected() -> None:
    """no-serde adapter (kv_together default) cannot coexist with a
    v-only adapter (kv_split_tier)."""
    cfgs = [
        _FakeAdapterCfg(serde_config=None),
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8_v_only")),
    ]
    with pytest.raises(ValueError, match="Incompatible L2 adapter storage placement"):
        derive_storage_placement_mode(cfgs)


def test_derive_placement_mode_two_v_only_adapters_returns_split_tier() -> None:
    """Two V-only adapters both demand split-tier -> compatible."""
    cfgs = [
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8_v_only")),
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8_v_only")),
    ]
    assert (
        derive_storage_placement_mode(cfgs)
        == StoragePlacementMode.KV_SPLIT_TIER
    )


# =============================================================================
# SplitTierState
# =============================================================================


def test_split_tier_state_values_distinct() -> None:
    values = [s.value for s in SplitTierState]
    assert len(values) == len(set(values))


def test_split_tier_state_names_are_documented() -> None:
    """All four documented states exist in the enum (the doc lists
    STORE_IN_FLIGHT, COMPLETE, INVALIDATED, DELETE_IN_FLIGHT)."""
    names = {s.name for s in SplitTierState}
    assert names == {
        "STORE_IN_FLIGHT",
        "COMPLETE",
        "INVALIDATED",
        "DELETE_IN_FLIGHT",
    }
