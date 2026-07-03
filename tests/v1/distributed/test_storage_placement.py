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

# Standard
from typing import cast

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.config import L2AdapterConfigBase
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


def _cfgs(*cfgs: _FakeAdapterCfg) -> list[L2AdapterConfigBase]:
    """Cast test stand-ins to the production adapter-config list type."""
    return cast("list[L2AdapterConfigBase]", list(cfgs))


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


def test_derive_component_key_preserves_object_group_id() -> None:
    """object_group_id is part of ObjectKey identity (hybrid /
    sliding-window models store one chunk per KV cache group under
    the same chunk_hash) -- the child keys must inherit it verbatim
    or same-hash chunks from different groups collide."""
    logical = ObjectKey(
        chunk_hash=b"\x42" * 32,
        model_name="hybrid-model",
        kv_rank=0,
        object_group_id=3,
        cache_salt="",
    )
    assert derive_component_key(logical, "k").object_group_id == 3
    assert derive_component_key(logical, "v").object_group_id == 3


def test_derive_component_key_no_collision_across_object_groups() -> None:
    """Two logical keys differing ONLY by object_group_id (the exact
    shape hybrid models produce: same content hash, one key per KV
    cache group) must derive non-colliding children for both roles."""
    group0 = ObjectKey(
        chunk_hash=b"\x42" * 32,
        model_name="hybrid-model",
        kv_rank=0,
        object_group_id=0,
        cache_salt="",
    )
    group1 = ObjectKey(
        chunk_hash=b"\x42" * 32,
        model_name="hybrid-model",
        kv_rank=0,
        object_group_id=1,
        cache_salt="",
    )
    assert derive_component_key(group0, "k") != derive_component_key(group1, "k")
    assert derive_component_key(group0, "v") != derive_component_key(group1, "v")


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


def test_reverse_component_key_round_trips_all_identity_fields() -> None:
    """Every ObjectKey identity field survives derive -> reverse,
    including object_group_id (paired eviction reverses a K-child to
    find its logical key; a dropped field would delete the wrong
    group's V child)."""
    logical = ObjectKey(
        chunk_hash=b"\x99" * 32,
        model_name="hybrid-model",
        kv_rank=5,
        object_group_id=2,
        cache_salt="tenant-7",
    )
    for role in ("k", "v"):
        child = derive_component_key(logical, role)
        result = reverse_component_key(child)
        assert result is not None
        rev_logical, rev_role = result
        assert rev_role == role
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
    assert derive_storage_placement_mode([]) == StoragePlacementMode.KV_TOGETHER


def test_derive_placement_mode_no_serde_returns_kv_together() -> None:
    cfgs = [_FakeAdapterCfg(serde_config=None)]
    assert (
        derive_storage_placement_mode(_cfgs(*cfgs)) == StoragePlacementMode.KV_TOGETHER
    )


def test_derive_placement_mode_fp8_returns_kv_together() -> None:
    """Single-tensor serdes (no slot mapping) place K+V together."""
    cfgs = [_FakeAdapterCfg(serde_config=SerdeConfig(type="fp8"))]
    assert (
        derive_storage_placement_mode(_cfgs(*cfgs)) == StoragePlacementMode.KV_TOGETHER
    )


def test_derive_placement_mode_asym_mode_1_returns_kv_together() -> None:
    """asym_k16_v8 storage-only: identity mapping (0, 1), no None
    slots -> both children in one blob (kv_together)."""
    cfgs = [_FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8"))]
    assert (
        derive_storage_placement_mode(_cfgs(*cfgs)) == StoragePlacementMode.KV_TOGETHER
    )


def test_derive_placement_mode_asym_v_only_returns_kv_split_tier() -> None:
    """asym_k16_v8_v_only: mapping (None, 1), so slot 0 (K) is
    absent from the L2 path -> split-tier placement."""
    cfgs = [_FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8_v_only"))]
    assert (
        derive_storage_placement_mode(_cfgs(*cfgs))
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
        derive_storage_placement_mode(_cfgs(*cfgs))


def test_derive_placement_mode_mixed_no_serde_and_v_only_rejected() -> None:
    """no-serde adapter (kv_together default) cannot coexist with a
    v-only adapter (kv_split_tier)."""
    cfgs = [
        _FakeAdapterCfg(serde_config=None),
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8_v_only")),
    ]
    with pytest.raises(ValueError, match="Incompatible L2 adapter storage placement"):
        derive_storage_placement_mode(_cfgs(*cfgs))


def test_derive_placement_mode_two_v_only_adapters_returns_split_tier() -> None:
    """Two V-only adapters both demand split-tier -> compatible."""
    cfgs = [
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8_v_only")),
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8_v_only")),
    ]
    assert (
        derive_storage_placement_mode(_cfgs(*cfgs))
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
    g = m.register_pending(k)
    assert m.lookup(k) == SplitTierState.STORE_IN_FLIGHT
    assert m.lookup_entry(k) == (SplitTierState.STORE_IN_FLIGHT, g)
    assert not m.is_complete(k)
    m.mark_complete(k, g)
    assert m.lookup(k) == SplitTierState.COMPLETE
    assert m.is_complete(k)


def test_manifest_register_returns_monotonic_generations() -> None:
    """Each register_pending hands out a strictly increasing,
    process-unique generation id -- even across drop (so a stale
    cleanup for an old generation can never collide with a fresh
    entry)."""
    m = SplitTierManifest()
    k1 = _make_key(chunk_hash=b"\x11" * 32)
    k2 = _make_key(chunk_hash=b"\x22" * 32)
    g1 = m.register_pending(k1)
    g2 = m.register_pending(k2)
    assert g2 > g1
    m.mark_complete(k1, g1)
    m.mark_invalidated(k1, g1)
    m.drop(k1, g1)
    g3 = m.register_pending(k1)  # same key, brand-new generation
    assert g3 > g2


def test_manifest_register_rejects_duplicate() -> None:
    """A second register_pending while a store is IN FLIGHT is a
    wrapper bug -- surface it loudly rather than silently overwriting."""
    m = SplitTierManifest()
    k = _make_key()
    m.register_pending(k)
    with pytest.raises(ValueError, match="already tracked"):
        m.register_pending(k)


def test_manifest_register_rejects_delete_in_flight() -> None:
    """Paired cleanup is actively deleting the previous generation's
    children -- a new store would race the deletes, so it is refused
    (the caller fails the task; a retry succeeds after drop())."""
    m = SplitTierManifest()
    k = _make_key()
    g = m.register_pending(k)
    m.mark_invalidated(k, g)
    m.mark_delete_in_flight(k, g)
    with pytest.raises(ValueError, match="DELETE_IN_FLIGHT"):
        m.register_pending(k)


def test_manifest_register_reclaims_stale_complete() -> None:
    """A COMPLETE entry whose K child was cleared out from under the
    manifest (POST /cache/clear tier=l1) must not block a re-store of
    the key forever -- register_pending starts a new generation."""
    m = SplitTierManifest()
    k = _make_key()
    g1 = m.register_pending(k)
    m.mark_complete(k, g1)
    g2 = m.register_pending(k)  # new generation, no raise
    assert g2 != g1
    assert m.lookup(k) == SplitTierState.STORE_IN_FLIGHT
    m.mark_complete(k, g2)
    assert m.is_complete(k)


def test_manifest_register_reclaims_invalidated() -> None:
    """A failed store's cleanup leaves INVALIDATED; the key must be
    storable again (one failure must never become a permanent per-key
    denial of caching)."""
    m = SplitTierManifest()
    k = _make_key()
    g1 = m.register_pending(k)
    m.mark_invalidated(k, g1)
    g2 = m.register_pending(k)  # new generation, no raise
    assert g2 != g1
    assert m.lookup(k) == SplitTierState.STORE_IN_FLIGHT


def test_manifest_stale_cleanup_never_touches_new_generation() -> None:
    """The core generation guard: a cleanup started for generation G
    (invalidate / delete-in-flight / drop) must be a no-op once the key
    has been reclaimed under G+1.  This is the invariant that keeps an
    eviction teardown or failed-store cleanup from destroying the
    composite a concurrent re-store just built."""
    m = SplitTierManifest()
    k = _make_key()
    g1 = m.register_pending(k)
    m.mark_complete(k, g1)
    # A re-store reclaims the key under a fresh generation.
    m.drop(k, g1)  # (in practice the K child was evicted first)
    g2 = m.register_pending(k)
    m.mark_complete(k, g2)
    assert m.is_complete(k)
    # The OLD generation's late cleanup must not touch g2.
    assert m.mark_invalidated(k, g1) is False
    assert m.is_complete(k)  # still COMPLETE under g2
    m.drop(k, g1)  # stale drop: no-op
    assert m.lookup_entry(k) == (SplitTierState.COMPLETE, g2)
    with pytest.raises(ValueError, match="expected generation"):
        m.mark_delete_in_flight(k, g1)
    # The current generation's own cleanup still works.
    assert m.mark_invalidated(k, g2) is True
    assert m.lookup(k) == SplitTierState.INVALIDATED


def test_manifest_mark_complete_wrong_generation_raises() -> None:
    """A stale completion (this task's ack arriving after the key was
    reclaimed) must not resurrect a superseded composite."""
    m = SplitTierManifest()
    k = _make_key()
    g1 = m.register_pending(k)
    m.mark_invalidated(k, g1)
    g2 = m.register_pending(k)
    with pytest.raises(ValueError, match="expected generation"):
        m.mark_complete(k, g1)
    # g2 can still complete.
    m.mark_complete(k, g2)
    assert m.is_complete(k)


def test_manifest_tracked_keys_snapshot() -> None:
    """tracked_keys returns every tracked logical key regardless of
    state (StorageManager.clear sweeps this snapshot)."""
    m = SplitTierManifest()
    k1 = _make_key(chunk_hash=b"\x11" * 32)
    k2 = _make_key(chunk_hash=b"\x22" * 32)
    assert m.tracked_keys() == []
    g1 = m.register_pending(k1)
    m.register_pending(k2)
    m.mark_complete(k1, g1)
    assert sorted(m.tracked_keys(), key=lambda k: k.chunk_hash) == [k1, k2]
    m.drop(k1, g1)
    assert m.tracked_keys() == [k2]


def test_manifest_mark_complete_rejects_wrong_state() -> None:
    m = SplitTierManifest()
    k = _make_key()
    # Untracked
    with pytest.raises(ValueError, match="untracked or owned"):
        m.mark_complete(k, 1)
    # In INVALIDATED
    g = m.register_pending(k)
    m.mark_invalidated(k, g)
    with pytest.raises(ValueError, match="INVALIDATED"):
        m.mark_complete(k, g)


def test_manifest_mark_invalidated_from_any_state_is_idempotent() -> None:
    """Invalidation must be tolerant of repeat calls and of any
    starting state (including STORE_IN_FLIGHT for a failed store)."""
    m = SplitTierManifest()
    k = _make_key()
    # Untracked: no-op, returns False (nothing owned).
    assert m.mark_invalidated(k, 1) is False
    assert m.lookup(k) is None
    # From STORE_IN_FLIGHT: valid (failed-store cleanup path).
    g = m.register_pending(k)
    assert m.mark_invalidated(k, g) is True
    assert m.lookup(k) == SplitTierState.INVALIDATED
    # Idempotent repeat at the same generation.
    assert m.mark_invalidated(k, g) is True
    assert m.lookup(k) == SplitTierState.INVALIDATED


def test_manifest_delete_in_flight_requires_invalidated() -> None:
    m = SplitTierManifest()
    k = _make_key()
    g = m.register_pending(k)
    m.mark_complete(k, g)
    # Not yet invalidated.
    with pytest.raises(ValueError, match="INVALIDATED"):
        m.mark_delete_in_flight(k, g)
    m.mark_invalidated(k, g)
    m.mark_delete_in_flight(k, g)
    assert m.lookup(k) == SplitTierState.DELETE_IN_FLIGHT


def test_manifest_drop_removes_entry() -> None:
    m = SplitTierManifest()
    k = _make_key()
    g = m.register_pending(k)
    m.mark_complete(k, g)
    m.mark_invalidated(k, g)
    m.mark_delete_in_flight(k, g)
    m.drop(k, g)
    assert m.lookup(k) is None
    # Idempotent.
    m.drop(k, g)


def test_manifest_distinct_keys_dont_interfere() -> None:
    """Two logical keys evolve independently through the state
    machine -- no cross-key state leakage."""
    m = SplitTierManifest()
    k1 = _make_key(chunk_hash=b"\x11" * 32)
    k2 = _make_key(chunk_hash=b"\x22" * 32)
    g1 = m.register_pending(k1)
    m.register_pending(k2)
    m.mark_complete(k1, g1)
    assert m.lookup(k1) == SplitTierState.COMPLETE
    assert m.lookup(k2) == SplitTierState.STORE_IN_FLIGHT
    m.mark_invalidated(k1, g1)
    assert m.lookup(k1) == SplitTierState.INVALIDATED
    assert m.lookup(k2) == SplitTierState.STORE_IN_FLIGHT


def test_manifest_len_tracks_outstanding_keys() -> None:
    m = SplitTierManifest()
    assert len(m) == 0
    k1 = _make_key(chunk_hash=b"\x11" * 32)
    k2 = _make_key(chunk_hash=b"\x22" * 32)
    g1 = m.register_pending(k1)
    assert len(m) == 1
    m.register_pending(k2)
    assert len(m) == 2
    m.drop(k1, g1)
    assert len(m) == 1


def test_manifest_state_counts_empty_has_all_states_zero() -> None:
    """The gauge backing ``state_counts`` always reports every state so
    the observability series set is stable even when the manifest is
    empty."""
    m = SplitTierManifest()
    counts = m.state_counts()
    assert set(counts) == set(SplitTierState)
    assert all(v == 0 for v in counts.values())


def test_manifest_state_counts_reflects_mixed_states() -> None:
    """One entry parked in each of the four states is counted in its own
    bucket, and the buckets sum to the tracked total."""
    m = SplitTierManifest()
    k_store = _make_key(chunk_hash=b"\x11" * 32)
    k_complete = _make_key(chunk_hash=b"\x22" * 32)
    k_invalidated = _make_key(chunk_hash=b"\x33" * 32)
    k_delete = _make_key(chunk_hash=b"\x44" * 32)

    m.register_pending(k_store)  # STORE_IN_FLIGHT

    g_complete = m.register_pending(k_complete)
    m.mark_complete(k_complete, g_complete)  # COMPLETE

    g_inv = m.register_pending(k_invalidated)
    m.mark_invalidated(k_invalidated, g_inv)  # INVALIDATED

    g_del = m.register_pending(k_delete)
    m.mark_invalidated(k_delete, g_del)
    m.mark_delete_in_flight(k_delete, g_del)  # DELETE_IN_FLIGHT

    counts = m.state_counts()
    assert counts[SplitTierState.STORE_IN_FLIGHT] == 1
    assert counts[SplitTierState.COMPLETE] == 1
    assert counts[SplitTierState.INVALIDATED] == 1
    assert counts[SplitTierState.DELETE_IN_FLIGHT] == 1
    assert sum(counts.values()) == len(m)


def test_manifest_state_counts_drop_decrements() -> None:
    """Dropping an entry removes it from the state distribution."""
    m = SplitTierManifest()
    k = _make_key(chunk_hash=b"\x55" * 32)
    g = m.register_pending(k)
    assert m.state_counts()[SplitTierState.STORE_IN_FLIGHT] == 1
    m.mark_complete(k, g)
    assert m.state_counts()[SplitTierState.COMPLETE] == 1
    m.mark_invalidated(k, g)
    m.drop(k, g)
    counts = m.state_counts()
    assert all(v == 0 for v in counts.values())


def test_storage_manager_exposes_split_tier_manifest() -> None:
    """Sanity: every StorageManager has a manifest (empty by default).
    This is the integration point the wrapper depends on (PR-2'+)."""
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


def test_storage_manager_clear_sweeps_stale_manifest_entries() -> None:
    """POST /cache/clear (tier=l1) removes K children from L1; the
    manifest sweep must drop the now-uncomposable entries (phantom
    COMPLETE hits + permanently blocked re-stores otherwise) while
    keeping entries whose K child survived the clear."""
    # Standard
    import shutil
    import tempfile

    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
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

    disk_path = tempfile.mkdtemp(prefix="lmcache_clear_sweep_test_")
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
                    size_in_bytes=1 << 30,
                    use_lazy=True,
                    init_size_in_bytes=64 << 20,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
        )
        sm = StorageManager(sm_cfg)
        try:
            manifest = sm.split_tier_manifest

            # Entry A: COMPLETE but its K child is NOT in L1 (the
            # stale-after-clear shape).  It must be swept.
            stale = _make_key(chunk_hash=b"\x51" * 32)
            stale_gen = manifest.register_pending(stale)
            manifest.mark_complete(stale, stale_gen)

            # Entry B: its K child holds an L1 WRITE lock across the
            # clear (an in-flight store), so force=False keeps the L1
            # entry and the sweep must keep the manifest entry.
            live = _make_key(chunk_hash=b"\x52" * 32)
            manifest.register_pending(live)
            live_k_child = derive_component_key(live, "k")
            # Seed the K-child L1 entry via the LOW-LEVEL l1_manager, the
            # way the real split-tier store path reserves single-component
            # K children.  The high-level StorageManager.reserve_write is
            # the layout-policy choke point (it would try to split this
            # single-component [16] buffer as a packed [2, ...] group).
            reserved = sm._l1_manager.reserve_write(
                keys=[live_k_child],
                is_temporary=[False],
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([16])], dtypes=[torch.bfloat16]
                ),
                mode="new",
            )
            assert live_k_child in reserved

            sm.clear(force=False)

            assert manifest.lookup(stale) is None, (
                "stale COMPLETE entry survived the clear sweep"
            )
            assert manifest.lookup(live) is not None, (
                "entry with a surviving K child was wrongly swept"
            )
            # The swept key is storable again.
            manifest.register_pending(stale)

            # Release the write lock before close.
            sm.finish_write([live_k_child])
        finally:
            sm.close()
    finally:
        shutil.rmtree(disk_path, ignore_errors=True)
