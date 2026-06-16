# SPDX-License-Identifier: Apache-2.0
"""Tests for the early-release-of-logical contract on split-tier stores.

The split-tier store path through ``SerdeL2AdapterWrapper`` copies K into
a dedicated L1 K-child and V into a slab-borrowed scratch tensor *before*
returning from ``submit_store_task``.  After that copy the original
producer-side logical L1 entry is no longer needed, so the wrapper
implements :class:`EarlyReleaseStoreAdapter` and hands its logical keys
to the StoreController for immediate release.

These tests pin the contract directly (no inner adapter, no StoreController
needed): the wrapper must

1. report logical keys as early-release for KV_SPLIT_TIER,
2. report nothing for KV_TOGETHER,
3. claim the keys exactly once per task (subsequent calls return ``[]``),
4. populate the V-scratch slab with a copy of V bytes so the codec can
   read V *after* the logical entry would be evictable.

The slab itself is exercised separately for acquire / release / over-
subscription semantics.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.l2_adapters.base import EarlyReleaseStoreAdapter
from lmcache.v1.distributed.l2_adapters.serde_wrapper import (
    _VScratchSlab,
    _VScratchSlot,
)


# =============================================================================
# _VScratchSlab — pool semantics (no wrapper / no StoreController needed)
# =============================================================================


def test_slab_acquire_returns_tensor_with_configured_shape_dtype() -> None:
    shape = torch.Size([4, 8])
    slab = _VScratchSlab(shape=shape, dtype=torch.float16, max_slots=4)
    t = slab.acquire()
    assert isinstance(t, torch.Tensor)
    assert t.shape == shape
    assert t.dtype == torch.float16
    assert t.device.type == "cpu"


def test_slab_release_makes_tensor_reusable() -> None:
    slab = _VScratchSlab(
        shape=torch.Size([2]), dtype=torch.float16, max_slots=2
    )
    t1 = slab.acquire()
    slab.release(t1)
    t2 = slab.acquire()
    # popleft returns the same object back
    assert t1 is t2


def test_slab_over_subscription_returns_one_shot_tensor() -> None:
    slab = _VScratchSlab(
        shape=torch.Size([2]), dtype=torch.float16, max_slots=1
    )
    # First acquire fills the slab to cap; second over-subscribes.
    t1 = slab.acquire()
    t2 = slab.acquire()
    assert t1 is not t2
    assert t1.shape == t2.shape
    stats = slab.stats()
    assert stats["over_subscriptions"] == 1
    assert stats["allocated"] == 1


def test_slab_release_rejects_mismatched_shape() -> None:
    slab = _VScratchSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=4
    )
    foreign = torch.empty(2, dtype=torch.float16)
    slab.release(foreign)  # silently dropped, must not raise
    stats = slab.stats()
    assert stats["free"] == 0


def test_slab_release_caps_queue_at_max_slots() -> None:
    slab = _VScratchSlab(
        shape=torch.Size([2]), dtype=torch.float16, max_slots=2
    )
    # Release more than max_slots distinct tensors -- extras silently
    # drop instead of growing the queue beyond the configured bound.
    for _ in range(5):
        slab.release(torch.empty(2, dtype=torch.float16))
    assert slab.stats()["free"] == 2


# =============================================================================
# _VScratchSlot — minimal MemoryObj-like used by the V-only codec
# =============================================================================


def test_v_scratch_slot_exposes_tensor() -> None:
    t = torch.arange(8, dtype=torch.float16)
    slot = _VScratchSlot(t)
    assert slot.tensor is t


# =============================================================================
# SerdeL2AdapterWrapper — early-release contract
# =============================================================================


@dataclass
class _Capture:
    """Records calls to L1Manager + inner adapter for assertions."""
    finish_read_calls: list[list]
    delete_calls: list[list]
    inner_submit_calls: list[tuple]


class _FakeL1Manager:
    """Bare minimum L1Manager surface the wrapper touches.

    Real L1Manager is heavy (NUMA, listeners, OTel gauges); we only need
    reserve_write / finish_write / reserve_read / finish_read / delete
    behaviors that the wrapper invokes during a split-tier store.
    """

    def __init__(self, capture: _Capture) -> None:
        self._capture = capture
        self._objects: dict = {}

    def register_listener(self, listener) -> None:
        pass

    def reserve_write(self, keys, is_temporary, layout_desc, mode):
        from lmcache.v1.distributed.error import L1Error
        # Allocate a tiny placeholder MemoryObj-like per key.
        out = {}
        for k in keys:
            tensor = torch.zeros(layout_desc.shapes[0], dtype=layout_desc.dtypes[0])
            obj = _GroupedMemoryObj(tensors=[tensor])
            self._objects[k] = obj
            out[k] = (L1Error.SUCCESS, obj)
        return out

    def finish_write(self, keys) -> None:
        pass

    def reserve_read(self, keys):
        from lmcache.v1.distributed.error import L1Error
        return {
            k: (L1Error.SUCCESS, self._objects[k])
            for k in keys
            if k in self._objects
        }

    def finish_read(self, keys) -> None:
        self._capture.finish_read_calls.append(list(keys))

    def delete(self, keys):
        self._capture.delete_calls.append(list(keys))
        return {}

    def finish_write_and_reserve_read(self, keys) -> None:
        pass


class _GroupedMemoryObj:
    """Two-group (K, V) MemoryObj-like exposing get_tensor + get_shapes."""

    def __init__(self, tensors: list) -> None:
        self._tensors = tensors

    def get_tensor(self, idx: int):
        return self._tensors[idx]

    def get_shapes(self):
        return [t.shape for t in self._tensors]

    def get_dtypes(self):
        return [t.dtype for t in self._tensors]


def _make_wrapper(placement_mode):
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import (
        SerdeL2AdapterWrapper,
    )
    from lmcache.v1.distributed.storage_placement import SplitTierManifest

    capture = _Capture(
        finish_read_calls=[],
        delete_calls=[],
        inner_submit_calls=[],
    )
    inner = _FakeInnerAdapter(capture)
    serde = _FakeSerdeProcessor()
    l1 = _FakeL1Manager(capture)
    manifest = SplitTierManifest()
    wrapper = SerdeL2AdapterWrapper(
        inner=inner,
        serde=serde,
        l1_manager=l1,
        placement_mode=placement_mode,
        split_tier_manifest=manifest,
    )
    return wrapper, capture, l1, manifest


class _FakeInnerAdapter:
    """Bare L2AdapterInterface surface the wrapper drives.

    Records ``submit_store_task`` calls.  Does not produce completions
    (the early-release test doesn't drive the drain loop).
    """

    def __init__(self, capture: _Capture) -> None:
        self._capture = capture
        from lmcache.v1.platform import create_event_notifier
        self._efd = create_event_notifier()

    def get_store_event_fd(self) -> int:
        return self._efd.fileno()

    def get_load_event_fd(self) -> int:
        # Distinct fd to satisfy the wrapper's poll-loop invariant.
        from lmcache.v1.platform import create_event_notifier
        if not hasattr(self, "_load_efd"):
            self._load_efd = create_event_notifier()
        return self._load_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        from lmcache.v1.platform import create_event_notifier
        if not hasattr(self, "_lookup_efd"):
            self._lookup_efd = create_event_notifier()
        return self._lookup_efd.fileno()

    def submit_store_task(self, keys, objects):
        self._capture.inner_submit_calls.append((list(keys), list(objects)))
        return 0

    def pop_completed_store_tasks(self):
        return {}

    def pop_completed_store_task_bytes(self):
        return {}

    def pop_completed_load_tasks(self):
        return {}

    def query_load_result(self, task_id):
        return None

    def add_listener(self, listener) -> None:
        pass

    def close(self) -> None:
        try:
            self._efd.close()
        except Exception:
            pass


class _FakeSerdeProcessor:
    """Minimal SerdeProcessor stand-in.  The wrapper's submit_serialize
    is exercised but we don't need it to produce a completion here.
    """

    def __init__(self) -> None:
        from lmcache.v1.platform import create_event_notifier
        self._efd = create_event_notifier()
        self._next_id = 0

    def get_serialize_event_fd(self) -> int:
        return self._efd.fileno()

    def get_deserialize_event_fd(self) -> int:
        from lmcache.v1.platform import create_event_notifier
        if not hasattr(self, "_d_efd"):
            self._d_efd = create_event_notifier()
        return self._d_efd.fileno()

    def submit_serialize(self, src, dst) -> int:
        sid = self._next_id
        self._next_id += 1
        return sid

    def submit_deserialize(self, src, dst) -> int:
        sid = self._next_id
        self._next_id += 1
        return sid

    def query_serialize_result(self, sid):
        return None

    def query_deserialize_result(self, sid):
        return None

    def input_slot_mapping(self):
        return (None, 1)

    def output_slot_mapping(self):
        return (0, 1)

    def serialized_layout_desc(self, layout_desc):
        from lmcache.v1.distributed.api import MemoryLayoutDesc
        return MemoryLayoutDesc(
            shapes=[torch.Size([1024])],
            dtypes=[torch.uint8],
        )

    def close(self) -> None:
        try:
            self._efd.close()
        except Exception:
            pass


def test_wrapper_implements_early_release_protocol() -> None:
    from lmcache.v1.distributed.storage_placement import StoragePlacementMode
    wrapper, *_ = _make_wrapper(StoragePlacementMode.KV_SPLIT_TIER)
    try:
        assert isinstance(wrapper, EarlyReleaseStoreAdapter)
    finally:
        wrapper.close()  # type: ignore[attr-defined]


def test_kv_together_claims_nothing() -> None:
    from lmcache.v1.distributed.storage_placement import StoragePlacementMode
    wrapper, *_ = _make_wrapper(StoragePlacementMode.KV_TOGETHER)
    try:
        # No task has been submitted -- claim returns [] regardless.
        assert wrapper.claim_early_release_keys(0) == []
        assert wrapper.claim_early_release_keys(99999) == []
    finally:
        wrapper.close()


# =============================================================================
# _KChildSlab — pool semantics (no L1Manager, no wrapper)
# =============================================================================


def test_kchild_slab_allocate_returns_tensor_memory_obj() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab
    from lmcache.v1.memory_management import TensorMemoryObj

    slab = _KChildSlab(
        shape=torch.Size([4, 8]), dtype=torch.float16, max_slots=4
    )
    obj = slab.allocate(None, None)
    assert isinstance(obj, TensorMemoryObj)
    assert obj.meta.shape == torch.Size([4, 8])
    assert obj.meta.dtype == torch.float16
    # The MemoryObj's parent_allocator must be the slab so free comes back.
    assert obj.parent_allocator is slab


def test_kchild_slab_batched_allocate_returns_n_objects() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab

    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=4
    )
    objs = slab.batched_allocate(None, None, batch_size=3)
    assert len(objs) == 3
    # Each backing tensor is distinct memory.
    addrs = {o.meta.address for o in objs}
    assert len(addrs) == 3


def test_kchild_slab_free_pools_for_reuse() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab

    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=2
    )
    o1 = slab.allocate(None, None)
    assert o1 is not None
    orig_data = o1.raw_data.data_ptr()
    slab.free(o1)
    o2 = slab.allocate(None, None)
    assert o2 is not None
    # The slab handed back the same MemoryObj (same backing buffer).
    assert o2.raw_data.data_ptr() == orig_data
    # And reset it for reuse — valid + ref_count restored.
    assert o2.is_valid()
    assert o2.meta.ref_count == 1
    assert o2._used_size_override is None


def test_kchild_slab_over_subscription_not_pooled() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab

    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=1
    )
    o1 = slab.allocate(None, None)
    o2 = slab.allocate(None, None)
    # Both valid, distinct objects; over-subscription counted.
    assert o1 is not o2
    assert slab.stats()["over_subscriptions"] == 1


def test_kchild_slab_free_rejects_mismatched_shape() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab
    from lmcache.v1.memory_management import TensorMemoryObj, MemoryObjMetadata, MemoryFormat

    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=2
    )
    # Build a TensorMemoryObj with the WRONG shape and try to free it
    # into the slab.  The slab silently drops it (doesn't add to free
    # deque) so a foreign object never gets handed back later.
    foreign_tensor = torch.empty(8, dtype=torch.float16)
    foreign_meta = MemoryObjMetadata(
        shape=torch.Size([8]),
        dtype=torch.float16,
        address=foreign_tensor.data_ptr(),
        phy_size=16,
        ref_count=1,
        fmt=MemoryFormat.KV_2LTD,
        shapes=[torch.Size([8])],
        dtypes=[torch.float16],
    )
    foreign = TensorMemoryObj(
        raw_data=foreign_tensor.view(torch.uint8).flatten(),
        metadata=foreign_meta,
        parent_allocator=slab,  # for __del__ correctness
    )
    slab.free(foreign)
    assert slab.stats()["free"] == 0


# =============================================================================
# L1Manager.reserve_external_writes — registers slab-backed MemoryObjs
# =============================================================================


def test_reserve_external_writes_registers_keys() -> None:
    from lmcache.v1.distributed.config import (
        L1ManagerConfig,
        L1MemoryManagerConfig,
    )
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab

    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            init_size_in_bytes=1 << 20,
        )
    )
    l1 = L1Manager(cfg)
    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=4
    )
    keys = ["k0", "k1"]
    objs = slab.batched_allocate(None, None, batch_size=2)
    results = l1.reserve_external_writes(keys, objs)
    for k in keys:
        err, obj = results[k]
        assert err == L1Error.SUCCESS
        assert obj is not None
    l1.finish_write(keys)
    # The keys are now in L1 and addressable via reserve_read.
    rr = l1.reserve_read(keys)
    for k in keys:
        err, obj = rr[k]
        assert err == L1Error.SUCCESS
    l1.finish_read(keys)


def test_reserve_external_writes_rejects_existing_key() -> None:
    from lmcache.v1.distributed.config import (
        L1ManagerConfig,
        L1MemoryManagerConfig,
    )
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab

    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            init_size_in_bytes=1 << 20,
        )
    )
    l1 = L1Manager(cfg)
    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=4
    )
    keys = ["k0"]
    o1 = slab.batched_allocate(None, None, batch_size=1)
    l1.reserve_external_writes(keys, o1)
    l1.finish_write(keys)
    # Second register attempt on the same key returns KEY_NOT_WRITABLE.
    o2 = slab.batched_allocate(None, None, batch_size=1)
    results = l1.reserve_external_writes(keys, o2)
    assert results["k0"][0] == L1Error.KEY_NOT_WRITABLE


# =============================================================================
# L1MemoryUsageProvider — slab bytes visible to L1MemoryManager
# =============================================================================


def test_slab_implements_l1_memory_usage_provider_protocol() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab
    from lmcache.v1.distributed.memory_manager import L1MemoryUsageProvider

    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=4
    )
    assert isinstance(slab, L1MemoryUsageProvider)


def test_kchild_slab_in_flight_tracks_one_shots() -> None:
    """The slab's L1MemoryUsageProvider reading must count ALL live
    K-children, including over-subscription one-shots that bypass the
    pool.  Without this the LRU policy is blind to most of the K-child
    footprint under heavy load (only the pooled subset is counted --
    e.g. 256 of 8000 K-children, the other 7700 invisible).
    """
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab

    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=2
    )
    # Empty.
    assert slab.get_used_capacity_bytes() == (0, 16)
    o1 = slab.allocate(None, None)
    o2 = slab.allocate(None, None)  # fills the pool
    assert o1 is not None and o2 is not None
    assert slab.get_used_capacity_bytes() == (16, 16)
    # Third acquire over-subscribes; used MUST cross capacity.
    o3 = slab.allocate(None, None)
    assert o3 is not None
    used, cap = slab.get_used_capacity_bytes()
    assert used == 24, f"expected 24, got {used}"
    assert cap == 16
    assert slab.stats()["over_subscriptions"] == 1
    assert slab.stats()["in_flight"] == 3
    # Free returns in_flight to zero regardless of pool/one-shot.
    slab.free(o3)
    assert slab.get_used_capacity_bytes() == (16, 16)
    slab.free(o2)
    slab.free(o1)
    assert slab.get_used_capacity_bytes() == (0, 16)


def test_v_scratch_slab_implements_provider_protocol() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _VScratchSlab
    from lmcache.v1.distributed.memory_manager import L1MemoryUsageProvider

    slab = _VScratchSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=2
    )
    assert isinstance(slab, L1MemoryUsageProvider)


def test_v_scratch_slab_in_flight_tracks_one_shots() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _VScratchSlab

    slab = _VScratchSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=2
    )
    assert slab.get_used_capacity_bytes() == (0, 16)
    t1 = slab.acquire()
    t2 = slab.acquire()
    t3 = slab.acquire()  # over-subscription
    assert slab.get_used_capacity_bytes() == (24, 16)
    assert slab.stats()["over_subscriptions"] == 1
    slab.release(t3)
    slab.release(t2)
    slab.release(t1)
    assert slab.get_used_capacity_bytes() == (0, 16)


def test_slab_get_used_capacity_bytes_tracks_live_objects() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab

    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=4
    )
    # Empty pool: nothing live, full capacity.
    used, capacity = slab.get_used_capacity_bytes()
    assert used == 0
    assert capacity == 4 * 4 * 2  # 4 slots * 4 elems * fp16 (2 bytes)

    # One live object: 8 bytes used, capacity unchanged.
    o1 = slab.allocate(None, None)
    assert o1 is not None
    used, capacity = slab.get_used_capacity_bytes()
    assert used == 8
    assert capacity == 32

    # Two live: 16 bytes used.
    o2 = slab.allocate(None, None)
    assert o2 is not None
    used, _ = slab.get_used_capacity_bytes()
    assert used == 16

    # Returning to pool drops bytes back.
    slab.free(o1)
    used, _ = slab.get_used_capacity_bytes()
    assert used == 8

    slab.free(o2)
    used, _ = slab.get_used_capacity_bytes()
    assert used == 0


def test_l1_memory_manager_aggregates_external_provider() -> None:
    from lmcache.v1.distributed.config import L1MemoryManagerConfig
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab
    from lmcache.v1.distributed.memory_manager import L1MemoryManager

    cfg = L1MemoryManagerConfig(
        size_in_bytes=1 << 20,  # 1 MiB
        use_lazy=False,
        init_size_in_bytes=1 << 20,
    )
    mm = L1MemoryManager(cfg)
    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=4
    )

    used_before, cap_before = mm.get_memory_usage()
    mm.register_external_memory_provider(slab)
    used_after_register, cap_after_register = mm.get_memory_usage()

    # Empty slab adds zero used but adds its capacity to the total.
    assert used_after_register == used_before
    assert cap_after_register == cap_before + 32

    # Allocate from slab: used grows, capacity unchanged.
    o = slab.allocate(None, None)
    assert o is not None
    used_after_alloc, cap_after_alloc = mm.get_memory_usage()
    assert used_after_alloc == used_before + 8
    assert cap_after_alloc == cap_after_register

    # Unregister: usage view goes back to baseline.
    slab.free(o)
    mm.unregister_external_memory_provider(slab)
    used_final, cap_final = mm.get_memory_usage()
    assert used_final == used_before
    assert cap_final == cap_before


def test_register_external_memory_provider_idempotent() -> None:
    from lmcache.v1.distributed.config import L1MemoryManagerConfig
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab
    from lmcache.v1.distributed.memory_manager import L1MemoryManager

    cfg = L1MemoryManagerConfig(
        size_in_bytes=1 << 20,
        use_lazy=False,
        init_size_in_bytes=1 << 20,
    )
    mm = L1MemoryManager(cfg)
    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=4
    )
    mm.register_external_memory_provider(slab)
    mm.register_external_memory_provider(slab)  # second add is no-op
    _, cap_once = mm.get_memory_usage()

    # Allocate one object; ensure exactly one provider contributed (not
    # double-counted).
    o = slab.allocate(None, None)
    assert o is not None
    used, _ = mm.get_memory_usage()
    assert used == 8  # single contribution, not 16
    slab.free(o)


# =============================================================================
# manifest-aware is_key_evictable — STORE_IN_FLIGHT K-children pinned
# =============================================================================


def _build_l1_with_manifest():
    """Helper: fresh L1Manager + SplitTierManifest wired together,
    plus a slab + a registered K-child key.

    Returns ``(l1, manifest, k_child_key, logical_key)``.
    """
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.config import (
        L1ManagerConfig,
        L1MemoryManagerConfig,
    )
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab
    from lmcache.v1.distributed.storage_placement import (
        SplitTierManifest,
        derive_component_key,
    )

    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            init_size_in_bytes=1 << 20,
        )
    )
    l1 = L1Manager(cfg)
    manifest = SplitTierManifest()
    l1.set_split_tier_manifest(manifest)

    logical = ObjectKey(
        chunk_hash=b"\x42" * 32,
        model_name="m",
        kv_rank=0,
        cache_salt="c",
    )
    k_child = derive_component_key(logical, "k")
    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=2
    )
    objs = slab.batched_allocate(None, None, batch_size=1)
    l1.reserve_external_writes([k_child], objs)
    l1.finish_write([k_child])
    return l1, manifest, k_child, logical


def test_kchild_pinned_during_store_in_flight() -> None:
    """K-child in STORE_IN_FLIGHT must NOT be evictable, otherwise
    the LRU policy can pull the K-child out from under an active V
    codec / L2 write (a same-pod sweep reproducer)."""
    l1, manifest, k_child, logical = _build_l1_with_manifest()
    manifest.register_pending(logical)
    assert l1.is_key_evictable(k_child) is False


def test_kchild_evictable_when_complete() -> None:
    """K-child in COMPLETE state is fair game for LRU eviction --
    paired eviction kicks in afterwards."""
    l1, manifest, k_child, logical = _build_l1_with_manifest()
    manifest.register_pending(logical)
    manifest.mark_complete(logical)
    assert l1.is_key_evictable(k_child) is True


def test_kchild_evictable_when_manifest_absent_orphan() -> None:
    """No manifest entry at all means the K-child is an orphan
    (manifest was dropped post-INVALIDATED).  Safe to evict."""
    l1, _manifest, k_child, _logical = _build_l1_with_manifest()
    # No register_pending: manifest has no entry for this logical.
    assert l1.is_key_evictable(k_child) is True


def test_non_kchild_keys_unaffected_by_manifest() -> None:
    """A plain logical key (not a derived child) goes through the
    legacy lock-only path even when a manifest is wired."""
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.config import (
        L1ManagerConfig,
        L1MemoryManagerConfig,
    )
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.storage_placement import SplitTierManifest

    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            init_size_in_bytes=1 << 20,
        )
    )
    l1 = L1Manager(cfg)
    l1.set_split_tier_manifest(SplitTierManifest())

    logical = ObjectKey(
        chunk_hash=b"\xab" * 32, model_name="m", kv_rank=0, cache_salt="c"
    )
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([4])], dtypes=[torch.float16]
    )
    l1.reserve_write([logical], [False], layout, mode="new")
    l1.finish_write([logical])
    # Logical key was never written to manifest; eviction gate falls
    # through to the lock-only check.
    assert l1.is_key_evictable(logical) is True


def test_no_manifest_wired_legacy_behavior_preserved() -> None:
    """Without ``set_split_tier_manifest``, ``is_key_evictable``
    behaves exactly as before -- K-child keys aren't gated."""
    from lmcache.v1.distributed.config import (
        L1ManagerConfig,
        L1MemoryManagerConfig,
    )
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab
    from lmcache.v1.distributed.storage_placement import derive_component_key
    from lmcache.v1.distributed.api import ObjectKey

    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            init_size_in_bytes=1 << 20,
        )
    )
    l1 = L1Manager(cfg)
    # Note: no set_split_tier_manifest call.
    logical = ObjectKey(
        chunk_hash=b"\xcd" * 32, model_name="m", kv_rank=0, cache_salt="c"
    )
    k_child = derive_component_key(logical, "k")
    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=2
    )
    objs = slab.batched_allocate(None, None, batch_size=1)
    l1.reserve_external_writes([k_child], objs)
    l1.finish_write([k_child])
    # Without a manifest wired, even a K-child is evictable -- the
    # caller (KV_TOGETHER deployments) doesn't have manifest state.
    assert l1.is_key_evictable(k_child) is True


def test_register_rejects_non_protocol_object() -> None:
    from lmcache.v1.distributed.config import L1MemoryManagerConfig
    from lmcache.v1.distributed.memory_manager import L1MemoryManager

    cfg = L1MemoryManagerConfig(
        size_in_bytes=1 << 20,
        use_lazy=False,
        init_size_in_bytes=1 << 20,
    )
    mm = L1MemoryManager(cfg)

    class _NotAProvider:
        pass

    with pytest.raises(TypeError):
        mm.register_external_memory_provider(_NotAProvider())  # type: ignore[arg-type]


def test_reserve_external_writes_validates_lengths() -> None:
    from lmcache.v1.distributed.config import (
        L1ManagerConfig,
        L1MemoryManagerConfig,
    )
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab

    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            init_size_in_bytes=1 << 20,
        )
    )
    l1 = L1Manager(cfg)
    slab = _KChildSlab(
        shape=torch.Size([4]), dtype=torch.float16, max_slots=4
    )
    objs = slab.batched_allocate(None, None, batch_size=2)
    with pytest.raises(ValueError):
        # 1 key, 2 objs
        l1.reserve_external_writes(["k0"], objs)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        l1.reserve_external_writes(
            ["k0", "k1"],  # type: ignore[list-item]
            objs,
            is_temporary=[False],
        )


def test_claim_is_single_shot() -> None:
    """Future-proofing: even if a controller polls more than once for the
    same task id, the wrapper hands the key list back exactly once.

    This is a unit test against the underlying state, not a full submit
    flow (that would require a working inner adapter + serde to drive
    submit_store_task end-to-end).  We probe the state directly.
    """
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _StoreTaskState
    from lmcache.v1.distributed.storage_placement import StoragePlacementMode

    wrapper, *_ = _make_wrapper(StoragePlacementMode.KV_SPLIT_TIER)
    try:
        # Synthesize a finished _StoreTaskState with early-release keys
        # pre-populated.  Exercises the latch in claim_early_release_keys.
        state = _StoreTaskState(
            wrapped_id=1,
            keys=["k0", "k1"],  # type: ignore[list-item]
            temp_keys=[],
            temp_objs=[],
            phase=None,  # type: ignore[arg-type]
            is_split_tier=True,
            early_release_keys=["k0", "k1"],  # type: ignore[list-item]
        )
        with wrapper._lock:
            wrapper._store_tasks[1] = state
        first = wrapper.claim_early_release_keys(1)
        second = wrapper.claim_early_release_keys(1)
        assert first == ["k0", "k1"]
        assert second == []
        assert state.early_release_claimed
    finally:
        wrapper.close()
