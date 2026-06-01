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
    SplitTierManifest,
    SplitTierState,
    StoragePlacementMode,
    derive_component_key,
    derive_storage_placement_mode,
    reverse_component_key,
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
# reverse_component_key
# =============================================================================


def test_reverse_component_key_round_trip_k() -> None:
    """A K child round-trips back to (logical, 'k')."""
    logical = _make_key(chunk_hash=b"\x77" * 32, salt="rev")
    k = derive_component_key(logical, "k")
    result = reverse_component_key(k)
    assert result is not None
    rev_logical, role = result
    assert role == "k"
    assert rev_logical == logical


def test_reverse_component_key_round_trip_v() -> None:
    """A V child round-trips back to (logical, 'v')."""
    logical = _make_key(chunk_hash=b"\x88" * 32)
    v = derive_component_key(logical, "v")
    result = reverse_component_key(v)
    assert result is not None
    rev_logical, role = result
    assert role == "v"
    assert rev_logical == logical


def test_reverse_component_key_returns_none_for_logical_key() -> None:
    """A non-child key (logical key with no role marker) is not a
    child -- reverse returns None and the eviction controller's
    paired logic skips it."""
    logical = _make_key(chunk_hash=b"\xaa" * 32)
    assert reverse_component_key(logical) is None


def test_reverse_component_key_returns_none_for_unknown_marker() -> None:
    """A 33-byte hash whose trailing byte isn't 0x01 or 0x02 (the K
    and V markers) is treated as a logical key, not a child."""
    weird = ObjectKey(
        chunk_hash=b"\xab" * 32 + b"\xff",
        model_name="t",
        kv_rank=0,
        cache_salt="",
    )
    assert reverse_component_key(weird) is None


def test_reverse_component_key_handles_empty_hash() -> None:
    """A pathologically empty / very-short chunk_hash returns None
    rather than crashing."""
    empty = ObjectKey(chunk_hash=b"", model_name="t", kv_rank=0, cache_salt="")
    one = ObjectKey(chunk_hash=b"\x01", model_name="t", kv_rank=0, cache_salt="")
    assert reverse_component_key(empty) is None
    assert reverse_component_key(one) is None


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


# =============================================================================
# SplitTierManifest
# =============================================================================


def test_manifest_lookup_unregistered_returns_none() -> None:
    m = SplitTierManifest()
    assert m.lookup(_make_key()) is None
    assert not m.is_complete(_make_key())


def test_manifest_register_then_complete() -> None:
    m = SplitTierManifest()
    k = _make_key()
    m.register_pending(k)
    assert m.lookup(k) == SplitTierState.STORE_IN_FLIGHT
    assert not m.is_complete(k)
    m.mark_complete(k)
    assert m.lookup(k) == SplitTierState.COMPLETE
    assert m.is_complete(k)


def test_manifest_register_rejects_duplicate() -> None:
    """A second register_pending under the same logical key is a
    wrapper bug -- surface it loudly rather than silently overwriting."""
    m = SplitTierManifest()
    k = _make_key()
    m.register_pending(k)
    with pytest.raises(ValueError, match="already tracked"):
        m.register_pending(k)


def test_manifest_mark_complete_rejects_wrong_state() -> None:
    m = SplitTierManifest()
    k = _make_key()
    # Untracked
    with pytest.raises(ValueError, match="UNTRACKED"):
        m.mark_complete(k)
    # In INVALIDATED
    m.register_pending(k)
    m.mark_invalidated(k)
    with pytest.raises(ValueError, match="INVALIDATED"):
        m.mark_complete(k)


def test_manifest_mark_invalidated_from_any_state_is_idempotent() -> None:
    """Invalidation must be tolerant of repeat calls and of any
    starting state (including STORE_IN_FLIGHT for a failed store)."""
    m = SplitTierManifest()
    k = _make_key()
    # Untracked: no-op.
    m.mark_invalidated(k)
    assert m.lookup(k) is None
    # From STORE_IN_FLIGHT: valid (failed-store cleanup path).
    m.register_pending(k)
    m.mark_invalidated(k)
    assert m.lookup(k) == SplitTierState.INVALIDATED
    # Idempotent repeat.
    m.mark_invalidated(k)
    assert m.lookup(k) == SplitTierState.INVALIDATED


def test_manifest_delete_in_flight_requires_invalidated() -> None:
    m = SplitTierManifest()
    k = _make_key()
    m.register_pending(k)
    m.mark_complete(k)
    # Not yet invalidated.
    with pytest.raises(ValueError, match="INVALIDATED"):
        m.mark_delete_in_flight(k)
    m.mark_invalidated(k)
    m.mark_delete_in_flight(k)
    assert m.lookup(k) == SplitTierState.DELETE_IN_FLIGHT


def test_manifest_drop_removes_entry() -> None:
    m = SplitTierManifest()
    k = _make_key()
    m.register_pending(k)
    m.mark_complete(k)
    m.mark_invalidated(k)
    m.mark_delete_in_flight(k)
    m.drop(k)
    assert m.lookup(k) is None
    # Idempotent.
    m.drop(k)


def test_manifest_distinct_keys_dont_interfere() -> None:
    """Two logical keys evolve independently through the state
    machine -- no cross-key state leakage."""
    m = SplitTierManifest()
    k1 = _make_key(chunk_hash=b"\x11" * 32)
    k2 = _make_key(chunk_hash=b"\x22" * 32)
    m.register_pending(k1)
    m.register_pending(k2)
    m.mark_complete(k1)
    assert m.lookup(k1) == SplitTierState.COMPLETE
    assert m.lookup(k2) == SplitTierState.STORE_IN_FLIGHT
    m.mark_invalidated(k1)
    assert m.lookup(k1) == SplitTierState.INVALIDATED
    assert m.lookup(k2) == SplitTierState.STORE_IN_FLIGHT


def test_manifest_len_tracks_outstanding_keys() -> None:
    m = SplitTierManifest()
    assert len(m) == 0
    k1 = _make_key(chunk_hash=b"\x11" * 32)
    k2 = _make_key(chunk_hash=b"\x22" * 32)
    m.register_pending(k1)
    assert len(m) == 1
    m.register_pending(k2)
    assert len(m) == 2
    m.drop(k1)
    assert len(m) == 1


def test_storage_manager_exposes_split_tier_manifest() -> None:
    """Sanity: every StorageManager has a manifest (empty by default).
    This is the integration point the wrapper depends on."""
    # Standard
    import shutil
    import tempfile

    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
    from lmcache.v1.distributed.serde import SerdeConfig
    from lmcache.v1.distributed.storage_manager import StorageManager

    disk_path = tempfile.mkdtemp(prefix="lmcache_manifest_test_")
    try:
        fs_cfg = FSL2AdapterConfig(
            base_path=disk_path,
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
        fs_cfg.serde_config = SerdeConfig(type="asym_k16_v8_v_only")
        sm_cfg = StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4 << 30,
                    use_lazy=True,
                    init_size_in_bytes=1 << 30,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
        )
        sm = StorageManager(sm_cfg)
        try:
            manifest = sm.split_tier_manifest
            assert isinstance(manifest, SplitTierManifest)
            assert len(manifest) == 0
        finally:
            sm.close()
    finally:
        shutil.rmtree(disk_path, ignore_errors=True)
