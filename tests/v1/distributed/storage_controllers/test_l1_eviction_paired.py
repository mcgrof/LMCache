# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the paired-eviction wiring on
:class:`L1EvictionController`.

Covers:

* When the controller is wired with a split-tier manifest +
  L2 adapters, evicting a K-child key fences the generation,
  deletes the K child, and enqueues the paired V child against
  every L2 adapter's ``delete()``.
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
import threading

# Third Party
import pytest

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
    l1_mock.delete.side_effect = lambda keys: {key: L1Error.SUCCESS for key in keys}
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
    ctrl._l1_manager.delete.side_effect = lambda _keys: {k_child: L1Error.KEY_IS_LOCKED}
    ctrl.execute_eviction_action(
        EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD)
    )
    # Composite intact: manifest still COMPLETE, no V delete issued.
    assert manifest.lookup(logical) == SplitTierState.COMPLETE
    adapter.delete.assert_not_called()

    # The reader finishes; a later eviction cycle succeeds and the
    # paired teardown completes.
    ctrl._l1_manager.delete.side_effect = lambda _keys: {k_child: L1Error.SUCCESS}
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


def test_paired_eviction_fences_replacement_until_both_children_deleted() -> None:
    """A replacement cannot reuse generation-blind child keys mid-delete.

    The manifest must already be DELETE_IN_FLIGHT when physical K deletion
    begins.  Registration then fails closed until K and V cleanup finishes,
    after which a new generation can register normally.
    """
    manifest = SplitTierManifest()
    logical = _make_key(b"\x99" * 32)
    k_child = derive_component_key(logical, "k")
    v_child = derive_component_key(logical, "v")
    g1 = manifest.register_pending(logical)
    manifest.mark_complete(logical, g1)

    adapter = MagicMock()
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter])
    delete_entered = threading.Event()
    allow_delete = threading.Event()

    def _blocking_delete(keys):
        delete_entered.set()
        assert allow_delete.wait(timeout=5)
        return {k: L1Error.SUCCESS for k in keys}

    ctrl._l1_manager.delete.side_effect = _blocking_delete
    eviction_thread = threading.Thread(
        target=ctrl.execute_eviction_action,
        args=(EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD),),
    )
    eviction_thread.start()
    assert delete_entered.wait(timeout=5)
    assert manifest.lookup(logical) == SplitTierState.DELETE_IN_FLIGHT
    with pytest.raises(ValueError, match="DELETE_IN_FLIGHT"):
        manifest.register_pending(logical)

    allow_delete.set()
    eviction_thread.join(timeout=5)
    assert not eviction_thread.is_alive()
    assert manifest.lookup(logical) is None
    adapter.delete.assert_called_once_with([v_child])

    g2 = manifest.register_pending(logical)
    assert g2 > g1
    assert manifest.lookup(logical) == SplitTierState.STORE_IN_FLIGHT


def test_rewire_waits_for_snapshotted_adapter_delete() -> None:
    """Detaching an adapter waits for a paired pass that already owns it."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\xbc" * 32)
    k_child = derive_component_key(logical, "k")
    generation = manifest.register_pending(logical)
    manifest.mark_complete(logical, generation)

    delete_entered = threading.Event()
    allow_delete = threading.Event()
    adapter = MagicMock()

    def _blocking_adapter_delete(_keys):
        delete_entered.set()
        assert allow_delete.wait(timeout=5)

    adapter.delete.side_effect = _blocking_adapter_delete
    ctrl = _build_controller(manifest=manifest, l2_adapters=[adapter])
    eviction_thread = threading.Thread(
        target=ctrl.execute_eviction_action,
        args=(EvictionAction(keys=[k_child], destination=EvictionDestination.DISCARD),),
    )
    eviction_thread.start()
    assert delete_entered.wait(timeout=5)

    result: list[bool] = []
    rewire_thread = threading.Thread(
        target=lambda: result.append(
            ctrl.set_split_tier_paired_eviction(manifest, [], timeout=5)
        )
    )
    rewire_thread.start()
    rewire_thread.join(timeout=0.05)
    assert rewire_thread.is_alive(), "rewire returned while old adapter was in use"

    allow_delete.set()
    eviction_thread.join(timeout=5)
    rewire_thread.join(timeout=5)
    assert not eviction_thread.is_alive()
    assert not rewire_thread.is_alive()
    assert result == [True]
