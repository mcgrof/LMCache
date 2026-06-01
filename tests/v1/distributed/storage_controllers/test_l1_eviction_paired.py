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
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)
from lmcache.v1.distributed.storage_placement import (
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
    manifest.register_pending(logical)
    manifest.mark_complete(logical)

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
    manifest.register_pending(logical)
    manifest.mark_complete(logical)
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
    manifest.register_pending(logical)
    manifest.mark_invalidated(logical)
    manifest.mark_delete_in_flight(logical)
    manifest.drop(logical)
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
    store is still STORE_IN_FLIGHT, the wrapper's drain path will
    fail to mark_complete (race) and clean up via the failed-store
    path.  The eviction's job here is the same: invalidate the
    manifest, enqueue V cleanup (best-effort), delete K from L1.
    """
    manifest = SplitTierManifest()
    logical = _make_key(b"\x66" * 32)
    k_child = derive_component_key(logical, "k")
    v_child = derive_component_key(logical, "v")
    manifest.register_pending(logical)
    # STORE_IN_FLIGHT, not COMPLETE.
    assert manifest.lookup(logical) == SplitTierState.STORE_IN_FLIGHT
    adapter = MagicMock()
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter])
    ctrl.execute_eviction_action(
        EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD)
    )
    # Manifest still gets invalidated + dropped.
    assert manifest.lookup(logical) is None
    # V child delete enqueued (the V store might be racing; best-effort).
    adapter.delete.assert_called_once_with([v_child])
    ctrl._l1_manager.delete.assert_called_once_with([k_child])


def test_paired_eviction_tolerates_adapter_delete_raise() -> None:
    """An L2 adapter whose ``delete()`` raises must not break the
    eviction loop.  The K child still gets deleted from L1; the V
    orphan is tolerable cold-tier waste (codex: delete is optional;
    log loudly)."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\x77" * 32)
    k_child = derive_component_key(logical, "k")
    manifest.register_pending(logical)
    manifest.mark_complete(logical)
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
