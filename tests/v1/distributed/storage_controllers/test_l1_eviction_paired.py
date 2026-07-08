# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the paired-eviction wiring on
:class:`L1EvictionController`.

Covers:

* When the controller is wired with a split-tier manifest +
  L2 adapters, evicting a K-child key invalidates the manifest,
  enqueues the paired V child against every L2 adapter's
  ``delete()``, and finishes with the L1 ``delete`` on the K
  child.
* The legacy DISCARD-only path is preserved when neither manifest
  nor adapters are wired.
* Non-K-child eviction targets (logical keys, V children, weird
  hashes) are passed through without touching the manifest.
* Idempotency: evicting an already-INVALIDATED key is a safe no-op.
"""

# Future
from __future__ import annotations

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)
from lmcache.v1.distributed.storage_placement import (
    ComponentKeyScheme,
    SplitTierManifest,
    SplitTierState,
    derive_component_key,
)


def _make_key(chunk_hash: bytes) -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name="evict-test",
        kv_rank=0,
        cache_salt="",
    )


def _build_controller(
    manifest: SplitTierManifest | None = None,
    l2_adapters: list | None = None,
) -> L1EvictionController:
    """Build a controller with mocked L1Manager so we don't need
    the full memory_manager scaffolding for these focused tests."""
    l1_mock = MagicMock()
    l1_mock.register_listener = MagicMock()
    l1_mock.delete = MagicMock()
    return L1EvictionController(
        l1_manager=l1_mock,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        split_tier_manifest=manifest,
        l2_adapters=l2_adapters,
    )


def test_legacy_discard_path_preserved_without_split_tier_wiring() -> None:
    """When no manifest + adapters are configured, eviction is
    legacy DISCARD: just l1_manager.delete(keys), no paired work."""
    ctrl = _build_controller()
    keys = [_make_key(b"\x00" * 32), _make_key(b"\x01" * 32)]
    ctrl.execute_eviction_action(
        EvictionAction(keys=keys, destination=EvictionDestination.DISCARD)
    )
    ctrl._l1_manager.delete.assert_called_once_with(keys)


def test_paired_eviction_k_child_invalidates_manifest_and_enqueues_v_delete() -> None:
    """Evicting a K child invalidates the manifest, enqueues V
    delete against every L2 adapter, then deletes the K child
    from L1."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\xab" * 32)
    k_child = derive_component_key(logical, "k")
    v_child = derive_component_key(logical, "v")
    # Drive the manifest through the happy path so it's COMPLETE
    # when eviction picks the K child.
    gen = manifest.register_pending(logical)
    manifest.mark_complete(logical, gen)

    adapter_a = MagicMock()
    adapter_b = MagicMock()
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter_a, adapter_b])
    ctrl.execute_eviction_action(
        EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD)
    )
    # Manifest entry dropped (DELETE_IN_FLIGHT then drop in one step).
    assert manifest.lookup(logical) is None
    # V child delete enqueued against every adapter.
    adapter_a.delete.assert_called_once_with([v_child])
    adapter_b.delete.assert_called_once_with([v_child])
    # K child physically deleted from L1.
    ctrl._l1_manager.delete.assert_called_once_with([k_child])


def test_paired_eviction_raw_unit_deletes_raw_unit_v_child() -> None:
    """A byte-through (RAW_UNIT) K victim must reclaim the RAW_UNIT V
    child, never a legacy V child.  The scheme travels via
    set_split_tier_paired_eviction (the seam StorageManager uses); the
    paired V-delete key must be derived under it."""
    raw = ComponentKeyScheme.RAW_UNIT
    manifest = SplitTierManifest(component_key_scheme=raw)
    logical = _make_key(b"\xcd" * 32)
    raw_k = derive_component_key(logical, "k", scheme=raw)
    raw_v = derive_component_key(logical, "v", scheme=raw)
    legacy_v = derive_component_key(logical, "v")  # must NOT be deleted
    gen = manifest.register_pending(logical)
    manifest.mark_complete(logical, gen)

    adapter = MagicMock()
    ctrl = _build_controller()
    ctrl.set_split_tier_paired_eviction(manifest, [adapter], raw)
    ctrl.execute_eviction_action(
        EvictionAction(keys=[raw_k], destination=EvictionDestination.DISCARD)
    )

    assert manifest.lookup(logical) is None
    # The RAW_UNIT V child is deleted; the legacy V key is never touched.
    adapter.delete.assert_called_once_with([raw_v])
    assert raw_v != legacy_v
    for call in adapter.delete.call_args_list:
        assert legacy_v not in call.args[0]
    ctrl._l1_manager.delete.assert_called_once_with([raw_k])


def test_paired_eviction_skips_logical_keys() -> None:
    """A non-child logical key in the eviction batch does NOT
    touch the manifest -- it's just a normal L1 discard.  This
    matters for KV_TOGETHER traffic flowing through the same
    StorageManager."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\x33" * 32)  # 32-byte hash, no role marker.
    adapter = MagicMock()
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter])
    ctrl.execute_eviction_action(
        EvictionAction(keys=[logical], destination=EvictionDestination.DISCARD)
    )
    # No adapter delete was triggered (logical isn't a K child).
    adapter.delete.assert_not_called()
    ctrl._l1_manager.delete.assert_called_once_with([logical])


def test_paired_eviction_skips_v_child_keys() -> None:
    """Evicting a V child key (somehow ending up in the L1 list)
    is NOT a paired-eviction trigger.  V lives on L2; its cleanup
    is the L2 controller's job, not L1."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\x44" * 32)
    v_child = derive_component_key(logical, "v")
    gen = manifest.register_pending(logical)
    manifest.mark_complete(logical, gen)
    adapter = MagicMock()
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter])
    ctrl.execute_eviction_action(
        EvictionAction(keys=[v_child], destination=EvictionDestination.DISCARD)
    )
    # No paired delete; manifest unchanged.
    adapter.delete.assert_not_called()
    assert manifest.is_complete(logical)
    ctrl._l1_manager.delete.assert_called_once_with([v_child])


def test_paired_eviction_idempotent_on_already_invalidated() -> None:
    """If the same K child is selected twice for eviction (e.g.
    racy policy), the second call is a no-op on the manifest
    side and still drives L1 delete."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\x55" * 32)
    k_child = derive_component_key(logical, "k")
    gen = manifest.register_pending(logical)
    manifest.mark_invalidated(logical, gen)
    manifest.mark_delete_in_flight(logical, gen)
    manifest.drop(logical, gen)
    adapter = MagicMock()
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter])
    # Manifest already fully cleaned up -> paired branch is a no-op.
    ctrl.execute_eviction_action(
        EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD)
    )
    adapter.delete.assert_not_called()
    ctrl._l1_manager.delete.assert_called_once_with([k_child])


def test_paired_eviction_skips_store_in_flight_avoids_corruption() -> None:
    """If a K child is somehow selected for eviction while the
    store is still STORE_IN_FLIGHT (a stale eviction-policy
    snapshot -- ``is_key_evictable`` pins these by design), the
    controller must leave it COMPLETELY alone: deleting the K child
    would tear the K out from under the active V codec / L2 write,
    and invalidating the manifest would make the store's later
    mark_complete blow up on an untracked key.
    """
    manifest = SplitTierManifest()
    logical = _make_key(b"\x66" * 32)
    k_child = derive_component_key(logical, "k")
    manifest.register_pending(logical)
    # STORE_IN_FLIGHT, not COMPLETE.
    assert manifest.lookup(logical) == SplitTierState.STORE_IN_FLIGHT
    adapter = MagicMock()
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter])
    ctrl.execute_eviction_action(
        EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD)
    )
    # Manifest untouched: the in-flight store proceeds normally.
    assert manifest.lookup(logical) == SplitTierState.STORE_IN_FLIGHT
    # No V delete enqueued and the K child excluded from the L1
    # delete batch.
    adapter.delete.assert_not_called()
    ctrl._l1_manager.delete.assert_called_once_with([])


def test_paired_eviction_locked_k_child_preserves_composite() -> None:
    """Between the eviction policy's is_key_evictable snapshot and
    the physical delete, a reader can read-lock the K child.  The
    delete then returns KEY_IS_LOCKED -- the composite (manifest
    entry + V child) must survive intact for a later retry instead
    of being torn down under the reader."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\x88" * 32)
    k_child = derive_component_key(logical, "k")
    v_child = derive_component_key(logical, "v")
    gen = manifest.register_pending(logical)
    manifest.mark_complete(logical, gen)

    adapter = MagicMock()
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter])
    ctrl._l1_manager.delete.return_value = {k_child: L1Error.KEY_IS_LOCKED}
    ctrl.execute_eviction_action(
        EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD)
    )
    # Composite intact: manifest still COMPLETE, no V delete issued.
    assert manifest.lookup(logical) == SplitTierState.COMPLETE
    adapter.delete.assert_not_called()

    # The reader finishes; a later eviction cycle succeeds and the
    # paired teardown completes.
    ctrl._l1_manager.delete.return_value = {k_child: L1Error.SUCCESS}
    ctrl.execute_eviction_action(
        EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD)
    )
    assert manifest.lookup(logical) is None
    adapter.delete.assert_called_once_with([v_child])


def test_paired_eviction_tolerates_adapter_delete_raise() -> None:
    """An L2 adapter whose ``delete()`` raises must not break the
    eviction loop.  The K child still gets deleted from L1; the V
    orphan is tolerable cold-tier waste (codex: delete is optional;
    log loudly)."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\x77" * 32)
    k_child = derive_component_key(logical, "k")
    gen = manifest.register_pending(logical)
    manifest.mark_complete(logical, gen)
    flaky_adapter = MagicMock()
    flaky_adapter.delete.side_effect = RuntimeError("L2 down")
    ctrl = _build_controller(manifest=manifest, l2_adapters=[flaky_adapter])
    # Must NOT raise.
    ctrl.execute_eviction_action(
        EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD)
    )
    # Manifest still progressed; L1 still deleted the K child.
    assert manifest.lookup(logical) is None
    ctrl._l1_manager.delete.assert_called_once_with([k_child])


def test_paired_eviction_stale_generation_preserves_new_complete() -> None:
    """After the eviction deletes the old K child, a fresh
    store reclaims the logical key under a NEW generation and completes
    before the teardown loop runs.  The teardown captured the OLD
    generation, so its invalidate / V-delete / drop must be no-ops --
    the new composite and its live V child must survive."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\x99" * 32)
    k_child = derive_component_key(logical, "k")
    g1 = manifest.register_pending(logical)
    manifest.mark_complete(logical, g1)  # old generation COMPLETE

    adapter = MagicMock()
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter])

    # The delete of the old K child opens the key; a concurrent re-store
    # reclaims it under a new generation and reaches COMPLETE before the
    # teardown loop runs.
    def _delete_then_restore(keys):
        g2 = manifest.register_pending(logical)
        manifest.mark_complete(logical, g2)
        return {k: L1Error.SUCCESS for k in keys}

    ctrl._l1_manager.delete.side_effect = _delete_then_restore
    ctrl.execute_eviction_action(
        EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD)
    )

    # The new composite survived untouched.
    assert manifest.is_complete(logical)
    adapter.delete.assert_not_called()


class _ReclaimAfterInvalidateManifest(SplitTierManifest):
    """A manifest that simulates a concurrent re-store reclaiming the
    logical key in the window right AFTER ``mark_invalidated`` succeeds
    and BEFORE the paired physical V-child delete -- the exact
    interleaving under concurrent re-store.  The V child's L2 key is
    generation-agnostic, so an un-fenced delete issued in this window
    destroys the NEW generation's live V child."""

    def __init__(self) -> None:
        super().__init__()
        self._reclaim_target: ObjectKey | None = None
        self.reclaimed_generation = 0

    def arm_reclaim(self, logical_key: ObjectKey) -> None:
        self._reclaim_target = logical_key

    def mark_invalidated(self, logical_key: ObjectKey, generation: int) -> bool:
        result = super().mark_invalidated(logical_key, generation)
        if result and logical_key == self._reclaim_target:
            self._reclaim_target = None
            # Concurrent re-store: reclaim under a fresh generation and
            # write + complete a new composite (a new V child lands on L2
            # under the same generation-agnostic key).
            g2 = super().register_pending(logical_key)
            super().mark_complete(logical_key, g2)
            self.reclaimed_generation = g2
        return result


def test_paired_eviction_reclaim_in_invalidate_window_preserves_new_v_child() -> None:
    """The paired V-child delete must
    be fenced by INVALIDATED -> DELETE_IN_FLIGHT BEFORE it touches L2.  If
    a re-store reclaims the logical key in the window right after
    ``mark_invalidated`` (register_pending is permitted from INVALIDATED),
    the generation-agnostic V-child delete would otherwise clobber the new
    generation's live V child, leaving a phantom COMPLETE with no V on L2.
    With the fence, the DELETE_IN_FLIGHT compare-and-set fails on the
    reclaimed generation and the delete is skipped."""
    manifest = _ReclaimAfterInvalidateManifest()
    logical = _make_key(b"\xbc" * 32)
    k_child = derive_component_key(logical, "k")
    g1 = manifest.register_pending(logical)
    manifest.mark_complete(logical, g1)  # old generation COMPLETE
    manifest.arm_reclaim(logical)

    adapter = MagicMock()
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter])
    ctrl._l1_manager.delete.return_value = {k_child: L1Error.SUCCESS}
    ctrl.execute_eviction_action(
        EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD)
    )

    # The reclaim landed inside the invalidate->delete window; the new
    # composite (generation g2) must survive intact and its V child must
    # NOT have been deleted.
    assert manifest.is_complete(logical)
    entry = manifest.lookup_entry(logical)
    assert entry is not None and entry[1] == manifest.reclaimed_generation
    adapter.delete.assert_not_called()


def test_paired_eviction_stale_generation_preserves_new_store_in_flight() -> None:
    """Scenario: a new STORE_IN_FLIGHT
    generation registers between the old K child's L1 delete and the
    teardown.  The teardown must NOT mark_invalidated it -- that would
    destroy the composite the new store is mid-way through writing."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\x9a" * 32)
    k_child = derive_component_key(logical, "k")
    g1 = manifest.register_pending(logical)
    manifest.mark_complete(logical, g1)

    adapter = MagicMock()
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter])

    def _delete_then_restore(keys):
        manifest.register_pending(logical)  # new gen, STORE_IN_FLIGHT
        return {k: L1Error.SUCCESS for k in keys}

    ctrl._l1_manager.delete.side_effect = _delete_then_restore
    ctrl.execute_eviction_action(
        EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD)
    )

    # The in-flight new generation was left alone.
    assert manifest.lookup(logical) == SplitTierState.STORE_IN_FLIGHT
    adapter.delete.assert_not_called()
