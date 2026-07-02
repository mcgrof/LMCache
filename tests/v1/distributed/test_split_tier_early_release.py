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
    slab = _VScratchSlab(shape=torch.Size([2]), dtype=torch.float16, max_slots=2)
    t1 = slab.acquire()
    slab.release(t1)
    t2 = slab.acquire()
    # popleft returns the same object back
    assert t1 is t2


def test_slab_over_subscription_returns_one_shot_tensor() -> None:
    slab = _VScratchSlab(shape=torch.Size([2]), dtype=torch.float16, max_slots=1)
    # First acquire fills the slab to cap; second over-subscribes.
    t1 = slab.acquire()
    t2 = slab.acquire()
    assert t1 is not t2
    assert t1.shape == t2.shape
    stats = slab.stats()
    assert stats["over_subscriptions"] == 1
    assert stats["allocated"] == 1


def test_slab_release_rejects_mismatched_shape() -> None:
    slab = _VScratchSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=4)
    foreign = torch.empty(2, dtype=torch.float16)
    slab.release(foreign)  # silently dropped, must not raise
    stats = slab.stats()
    assert stats["free"] == 0


def test_slab_release_caps_queue_at_max_slots() -> None:
    slab = _VScratchSlab(shape=torch.Size([2]), dtype=torch.float16, max_slots=2)
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
    finish_write_calls: list[list]
    fwrr_calls: list[list]
    drained_keys: list[list]
    delete_calls: list[list]
    inner_submit_calls: list[tuple]
    inner_delete_calls: list[list]
    inner_lookup_calls: list[list]
    inner_unlock_calls: list[list]


class _FakeL1Manager:
    """Bare minimum L1Manager surface the wrapper touches.

    Real L1Manager is heavy (NUMA, listeners, OTel gauges); we only need
    reserve_write / finish_write / reserve_read / finish_read / delete
    behaviors that the wrapper invokes during a split-tier store.
    """

    def __init__(self, capture: _Capture, fail_reserve_write: bool = False) -> None:
        self._capture = capture
        self._objects: dict = {}
        # When True, every reserve_write fails (simulates L1 memory
        # pressure at temp-buffer allocation time -- K-children go
        # through reserve_external_writes and are unaffected).
        self.fail_reserve_write = fail_reserve_write
        # Drain-listener subscribers (the real StoreController listener).
        # Populated via register_listener so a test can observe which
        # keys reach the L1->L2 drain when finish_write fires.
        self._listeners: list = []

    def register_listener(self, listener) -> None:
        self._listeners.append(listener)

    def reserve_write(self, keys, is_temporary, layout_desc, mode):
        from lmcache.v1.distributed.error import L1Error

        if self.fail_reserve_write:
            return {k: (L1Error.OUT_OF_MEMORY, None) for k in keys}
        # Allocate a tiny placeholder MemoryObj-like per key.
        out = {}
        for k in keys:
            tensor = torch.zeros(layout_desc.shapes[0], dtype=layout_desc.dtypes[0])
            obj = _GroupedMemoryObj(tensors=[tensor])
            self._objects[k] = obj
            out[k] = (L1Error.SUCCESS, obj)
        return out

    def reserve_external_writes(self, keys, memory_objs, is_temporary):
        from lmcache.v1.distributed.error import L1Error

        out = {}
        for k, obj in zip(keys, memory_objs, strict=True):
            self._objects[k] = obj
            out[k] = (L1Error.SUCCESS, obj)
        return out

    def finish_write(self, keys):
        from lmcache.v1.distributed.error import L1Error

        self._capture.finish_write_calls.append(list(keys))
        # Mirror the real L1Manager: finish_write releases the write
        # lock AND fires the L1->L2 drain listener.  A K child that
        # reaches this listener without the is_k_child_key filter is
        # exactly the Finding-A leak we are guarding against.
        for listener in self._listeners:
            listener.on_l1_keys_write_finished(list(keys))
            self._capture.drained_keys.append(list(keys))
        return {k: L1Error.SUCCESS for k in keys}

    def reserve_read(self, keys):
        from lmcache.v1.distributed.error import L1Error

        return {
            k: (L1Error.SUCCESS, self._objects[k]) for k in keys if k in self._objects
        }

    def finish_read(self, keys) -> None:
        self._capture.finish_read_calls.append(list(keys))

    def delete(self, keys):
        self._capture.delete_calls.append(list(keys))
        for k in keys:
            self._objects.pop(k, None)
        return {}

    def finish_write_and_reserve_read(self, keys):
        from lmcache.v1.distributed.error import L1Error

        # Mirror the real L1Manager: releases the write lock, takes a
        # read lock, and fires ONLY the (no-op in StoreController)
        # reserve-read listener -- never the L1->L2 drain listener.
        self._capture.fwrr_calls.append(list(keys))
        return {
            k: (L1Error.SUCCESS, self._objects[k]) for k in keys if k in self._objects
        }


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


def _make_wrapper(
    placement_mode,
    serialize_result: bool | None = None,
    fail_reserve_write: bool = False,
):
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import (
        SerdeL2AdapterWrapper,
    )
    from lmcache.v1.distributed.storage_placement import SplitTierManifest

    capture = _Capture(
        finish_read_calls=[],
        finish_write_calls=[],
        fwrr_calls=[],
        drained_keys=[],
        delete_calls=[],
        inner_submit_calls=[],
        inner_delete_calls=[],
        inner_lookup_calls=[],
        inner_unlock_calls=[],
    )
    inner = _FakeInnerAdapter(capture)
    serde = _FakeSerdeProcessor(serialize_result=serialize_result)
    l1 = _FakeL1Manager(capture, fail_reserve_write=fail_reserve_write)
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
        from lmcache.v1.distributed.api import KeyListPage
        from lmcache.v1.platform import create_event_notifier

        self._efd = create_event_notifier()
        # Page returned by list_l2_keys; a test overrides it to exercise
        # the wrapper's split-tier child-key re-presentation.
        self._list_page = KeyListPage(entries=(), next_page_token=None)

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

    def delete(self, keys):
        self._capture.inner_delete_calls.append(list(keys))

    def list_l2_keys(self, model_name=None, page_size=500, cursor=None):
        return self._list_page

    def submit_lookup_and_lock_task(self, keys, layout_desc):
        self._capture.inner_lookup_calls.append(list(keys))
        if not hasattr(self, "_lookup_tasks"):
            self._lookup_tasks = {}
            self._next_lookup_id = 100
        tid = self._next_lookup_id
        self._next_lookup_id += 1
        self._lookup_tasks[tid] = len(keys)
        return tid

    def query_lookup_and_lock_result(self, task_id):
        # First Party
        from lmcache.native_storage_ops import Bitmap

        n = getattr(self, "_lookup_tasks", {}).pop(task_id, None)
        if n is None:
            return None
        # Report every submitted key as a hit; the wrapper's manifest
        # gate decides what the caller actually sees.
        bitmap = Bitmap(n)
        for i in range(n):
            bitmap.set(i)
        return bitmap

    def submit_unlock(self, keys):
        self._capture.inner_unlock_calls.append(list(keys))

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
    is exercised but by default we don't produce a completion.

    Pass ``serialize_result=False`` to make every submitted serialize
    task complete as a FAILURE: the fake signals the serialize eventfd
    shortly after submit (from a timer thread, so the wrapper has
    registered its reverse-map entry by the time the drain runs) and
    ``query_serialize_result`` reports the failure.  This drives the
    wrapper's real drain-loop store-failure path end to end.
    """

    def __init__(self, serialize_result: bool | None = None) -> None:
        from lmcache.v1.platform import create_event_notifier

        self._efd = create_event_notifier()
        self._next_id = 0
        self._serialize_result = serialize_result
        self._submitted: set[int] = set()

    def get_serialize_event_fd(self) -> int:
        return self._efd.fileno()

    def get_deserialize_event_fd(self) -> int:
        from lmcache.v1.platform import create_event_notifier

        if not hasattr(self, "_d_efd"):
            self._d_efd = create_event_notifier()
        return self._d_efd.fileno()

    def submit_serialize(self, src, dst, keys=None) -> int:
        sid = self._next_id
        self._next_id += 1
        if self._serialize_result is not None:
            # Standard
            import threading

            self._submitted.add(sid)
            # Delay the completion signal so the wrapper's
            # submit_store_task has released its lock and registered
            # the serde-to-store reverse mapping before the drain
            # loop polls for results.
            threading.Timer(0.05, self._efd.notify).start()
        return sid

    def submit_deserialize(self, src, dst, keys=None) -> int:
        sid = self._next_id
        self._next_id += 1
        return sid

    def estimate_serialized_size(self, layout_desc) -> int:
        return 4096

    def query_serialize_result(self, sid):
        if self._serialize_result is not None and sid in self._submitted:
            return self._serialize_result
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

    slab = _KChildSlab(shape=torch.Size([4, 8]), dtype=torch.float16, max_slots=4)
    obj = slab.allocate(None, None)
    assert isinstance(obj, TensorMemoryObj)
    assert obj.meta.shape == torch.Size([4, 8])
    assert obj.meta.dtype == torch.float16
    # The MemoryObj's parent_allocator must be the slab so free comes back.
    assert obj.parent_allocator is slab


def test_kchild_slab_batched_allocate_returns_n_objects() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab

    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=4)
    objs = slab.batched_allocate(None, None, batch_size=3)
    assert len(objs) == 3
    # Each backing tensor is distinct memory.
    addrs = {o.meta.address for o in objs}
    assert len(addrs) == 3


def test_kchild_slab_free_pools_for_reuse() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab

    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=2)
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

    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=1)
    o1 = slab.allocate(None, None)
    o2 = slab.allocate(None, None)
    # Both valid, distinct objects; over-subscription counted.
    assert o1 is not o2
    assert slab.stats()["over_subscriptions"] == 1


def test_kchild_slab_free_rejects_mismatched_shape() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab
    from lmcache.v1.memory_management import (
        TensorMemoryObj,
        MemoryObjMetadata,
        MemoryFormat,
    )

    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=2)
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
    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=4)
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
    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=4)
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

    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=4)
    assert isinstance(slab, L1MemoryUsageProvider)


def test_kchild_slab_in_flight_tracks_one_shots() -> None:
    """The slab's L1MemoryUsageProvider reading must count ALL live
    K-children, including over-subscription one-shots that bypass the
    pool.  Without this the LRU policy is blind to most of the K-child
    footprint under heavy load (only the pooled subset is counted --
    e.g. 256 of 8000 K-children, the other 7700 invisible).
    """
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab

    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=2)
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

    slab = _VScratchSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=2)
    assert isinstance(slab, L1MemoryUsageProvider)


def test_v_scratch_slab_in_flight_tracks_one_shots() -> None:
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _VScratchSlab

    slab = _VScratchSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=2)
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

    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=4)
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
    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=4)

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
    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=4)
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
    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=2)
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
    gen = manifest.register_pending(logical)
    manifest.mark_complete(logical, gen)
    assert l1.is_key_evictable(k_child) is True


def test_kchild_evictable_when_manifest_absent_orphan() -> None:
    """No manifest entry at all means the K-child is an orphan
    (manifest was dropped post-INVALIDATED).  Safe to evict."""
    l1, _manifest, k_child, _logical = _build_l1_with_manifest()
    # No register_pending: manifest has no entry for this logical.
    assert l1.is_key_evictable(k_child) is True


def test_clear_force_false_preserves_in_flight_kchild() -> None:
    """clear(force=False) must NOT delete an UNLOCKED
    STORE_IN_FLIGHT K child.  finish_write released its write lock, so
    the legacy lock-only clear predicate would remove it -- destroying
    the composite's L1-canonical half while the store still reports
    success.  The manifest-aware predicate pins it exactly as
    is_key_evictable pins it against LRU eviction."""
    l1, manifest, k_child, logical = _build_l1_with_manifest()
    manifest.register_pending(logical)  # STORE_IN_FLIGHT
    assert l1.is_key_evictable(k_child) is False
    l1.clear(force=False)
    assert l1.get_object_state(k_child) is not None, (
        "in-flight STORE_IN_FLIGHT K child was wrongly cleared"
    )


def test_clear_force_false_removes_complete_kchild() -> None:
    """A COMPLETE K child is evictable, so clear(force=False) removes it
    like any other unlocked object -- the pin is specific to the
    in-flight window."""
    l1, manifest, k_child, logical = _build_l1_with_manifest()
    gen = manifest.register_pending(logical)
    manifest.mark_complete(logical, gen)
    assert l1.is_key_evictable(k_child) is True
    l1.clear(force=False)
    assert l1.get_object_state(k_child) is None


def test_clear_force_true_removes_in_flight_kchild() -> None:
    """force=True is a documented hard reset: it removes everything,
    including a manifest-pinned in-flight K child (the documented restart contract
    semantics -- the operator explicitly wiped the cache)."""
    l1, manifest, k_child, logical = _build_l1_with_manifest()
    manifest.register_pending(logical)
    l1.clear(force=True)
    assert l1.get_object_state(k_child) is None


def test_split_tier_list_l2_keys_maps_v_children_to_logical() -> None:
    """Operator listings must not leak internal child keys.  The
    wrapper re-presents each V child under its logical key, drops any
    anomalous K child, and passes non-child keys through -- pagination
    and sizes preserved."""
    # First Party
    from lmcache.v1.distributed.api import KeyEntry, KeyListPage, ObjectKey
    from lmcache.v1.distributed.storage_placement import (
        StoragePlacementMode,
        derive_component_key,
    )

    wrapper, _capture, _l1, _manifest = _make_wrapper(
        StoragePlacementMode.KV_SPLIT_TIER
    )
    try:
        logical = ObjectKey(chunk_hash=b"\x61" * 32, model_name="m", kv_rank=0)
        v_child = derive_component_key(logical, "v")
        k_child = derive_component_key(logical, "k")
        other = ObjectKey(chunk_hash=b"\x62" * 32, model_name="m", kv_rank=0)
        wrapper._inner._list_page = KeyListPage(
            entries=(
                KeyEntry(key=v_child.to_encoded_object_key(), size_bytes=128),
                KeyEntry(key=k_child.to_encoded_object_key(), size_bytes=64),
                KeyEntry(key=other.to_encoded_object_key(), size_bytes=99),
            ),
            next_page_token="tok",
        )
        page = wrapper.list_l2_keys(model_name="m")
        keys = [e.key.to_object_key() for e in page.entries]
        assert logical in keys  # V child re-presented under logical
        assert v_child not in keys
        assert k_child not in keys  # anomalous K on L2 dropped
        assert other in keys  # non-child passed through
        assert page.next_page_token == "tok"  # pagination preserved
        v_entry = next(e for e in page.entries if e.key.to_object_key() == logical)
        assert v_entry.size_bytes == 128  # size preserved through the remap
    finally:
        wrapper.close()


def test_kv_together_list_l2_keys_passthrough() -> None:
    """KV_TOGETHER returns the inner page verbatim (no child keys exist
    to re-present)."""
    # First Party
    from lmcache.v1.distributed.api import KeyEntry, KeyListPage, ObjectKey
    from lmcache.v1.distributed.storage_placement import StoragePlacementMode

    wrapper, _capture, _l1, _manifest = _make_wrapper(StoragePlacementMode.KV_TOGETHER)
    try:
        logical = ObjectKey(chunk_hash=b"\x63" * 32, model_name="m", kv_rank=0)
        page_in = KeyListPage(
            entries=(KeyEntry(key=logical.to_encoded_object_key(), size_bytes=42),),
            next_page_token=None,
        )
        wrapper._inner._list_page = page_in
        assert wrapper.list_l2_keys() is page_in
    finally:
        wrapper.close()


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
    layout = MemoryLayoutDesc(shapes=[torch.Size([4])], dtypes=[torch.float16])
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
    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=2)
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
    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=4)
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
    """Even if a controller polls more than once for the same task id,
    the wrapper hands the key list back exactly once.

    Driven through the public submit path: the fake serde never
    completes, so the task stays in flight with its early-release keys
    populated -- exactly the state a polling controller observes.
    """
    from lmcache.v1.distributed.storage_placement import StoragePlacementMode

    wrapper, *_ = _make_wrapper(StoragePlacementMode.KV_SPLIT_TIER)
    try:
        keys, task_id = _submit_split_tier_store(wrapper)
        first = wrapper.claim_early_release_keys(task_id)
        second = wrapper.claim_early_release_keys(task_id)
        assert first == keys
        assert second == []
    finally:
        wrapper.close()


# =============================================================================
# Split-tier store-failure lifecycle -- the manifest must never leak
# =============================================================================


def _submit_split_tier_store(wrapper, n_keys: int = 2):
    """Submit a split-tier store of ``n_keys`` grouped (K, V) objects
    through the wrapper's public surface.

    Returns:
        ``(logical_keys, task_id)``.
    """
    # First Party
    from lmcache.v1.distributed.api import ObjectKey

    keys = [
        ObjectKey(chunk_hash=bytes([0x30 + i]) * 32, model_name="m", kv_rank=0)
        for i in range(n_keys)
    ]
    objects = [
        _GroupedMemoryObj(
            tensors=[
                torch.zeros(torch.Size([4]), dtype=torch.bfloat16),
                torch.zeros(torch.Size([8]), dtype=torch.bfloat16),
            ]
        )
        for _ in range(n_keys)
    ]
    task_id = wrapper.submit_store_task(keys, objects)
    return keys, task_id


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    # Standard
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_split_tier_serialize_failure_drops_manifest_and_cleans_children() -> None:
    """A serialize failure surfaces through the wrapper's drain loop:
    the manifest entries must be DROPPED (not left INVALIDATED -- one
    failed store must never permanently block a key) and the K children
    deleted from L1.  The inner store was never submitted, so NO V-child
    delete is issued on the inner adapter (no bytes can have landed, and
    the delete would needlessly block the poll thread)."""
    # First Party
    from lmcache.v1.distributed.storage_placement import (
        StoragePlacementMode,
        derive_component_key,
    )

    wrapper, capture, _l1, manifest = _make_wrapper(
        StoragePlacementMode.KV_SPLIT_TIER, serialize_result=False
    )
    try:
        keys, task_id = _submit_split_tier_store(wrapper)
        # Registration happened synchronously on submit.
        assert len(manifest) == len(keys)

        popped: dict = {}

        def _task_failed() -> bool:
            popped.update(wrapper.pop_completed_store_tasks())
            return task_id in popped

        assert _wait_until(_task_failed), "store task never completed"
        assert not popped[task_id].is_successful()
        assert _wait_until(lambda: len(manifest) == 0), (
            "manifest entries leaked after a serialize failure"
        )

        # K children removed from L1.
        k_children = [derive_component_key(k, "k") for k in keys]
        deleted = [k for call in capture.delete_calls for k in call]
        assert all(k in deleted for k in k_children)

        # No inner V-child delete: the inner store was never submitted
        # (serialize failed first), so no V bytes could have landed.
        assert capture.inner_delete_calls == []

        # The keys are storable again: a new generation registers
        # without raising.
        for k in keys:
            manifest.register_pending(k)
    finally:
        wrapper.close()


def test_split_tier_inner_store_failure_deletes_v_children() -> None:
    """When the inner L2 store was actually SUBMITTED and then failed,
    a partial V blob may have landed, so the failed-store teardown must
    best-effort delete the V child names on the inner adapter.  This is
    the counterpart to the serialize-failure path, which skips the
    delete because nothing was submitted."""
    # First Party
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import (
        _StorePhase,
        _StoreTaskState,
    )
    from lmcache.v1.distributed.storage_placement import (
        StoragePlacementMode,
        derive_component_key,
    )

    wrapper, capture, _l1, manifest = _make_wrapper(StoragePlacementMode.KV_SPLIT_TIER)
    try:
        keys, _task_id = _submit_split_tier_store(wrapper)
        gens = {k: manifest.lookup_entry(k)[1] for k in keys}
        k_children = [derive_component_key(k, "k") for k in keys]
        v_children = [derive_component_key(k, "v") for k in keys]

        # Reconstruct the drain-loop state as it looks AFTER a successful
        # inner submit (phase INNER_STORE), then fail it.
        state = _StoreTaskState(
            wrapped_id=999,
            keys=list(keys),
            temp_keys=[],
            temp_objs=[],
            phase=_StorePhase.INNER_STORE,
            is_split_tier=True,
            k_child_keys=k_children,
            v_child_keys=v_children,
            split_tier_generations=gens,
        )
        capture.inner_delete_calls.clear()
        wrapper._invalidate_split_tier_pending(state)

        # V children WERE best-effort deleted on the inner adapter...
        inner_deleted = [k for call in capture.inner_delete_calls for k in call]
        assert all(v in inner_deleted for v in v_children)
        # ...and the manifest entries were dropped (keys storable again).
        assert all(manifest.lookup(k) is None for k in keys)
    finally:
        wrapper.close()


def test_split_tier_temp_alloc_failure_drops_manifest() -> None:
    """Temp-buffer allocation fails under L1 memory pressure -- exactly
    when retries are most likely.  The pending manifest entries must be
    dropped on the synchronous submit path so the keys stay storable."""
    # First Party
    from lmcache.v1.distributed.storage_placement import (
        StoragePlacementMode,
        derive_component_key,
    )

    wrapper, capture, _l1, manifest = _make_wrapper(
        StoragePlacementMode.KV_SPLIT_TIER, fail_reserve_write=True
    )
    try:
        keys, task_id = _submit_split_tier_store(wrapper)
        # The failure is synchronous: no drain loop involved.
        assert len(manifest) == 0, "manifest entries leaked after a temp-alloc failure"
        popped = wrapper.pop_completed_store_tasks()
        assert task_id in popped
        assert not popped[task_id].is_successful()

        k_children = [derive_component_key(k, "k") for k in keys]
        deleted = [k for call in capture.delete_calls for k in call]
        assert all(k in deleted for k in k_children)

        for k in keys:
            manifest.register_pending(k)
    finally:
        wrapper.close()


def test_split_tier_rejects_multi_kv_pair_object() -> None:
    """A multi-kernel-group object (two+ K/V pairs) must
    be rejected, not silently truncated to the first pair.  Admitting a
    4-group [g0K, g0V, g1K, g1V] object would mirror only g0->K and read
    g1->V, dropping the second KV group's bytes entirely."""
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.storage_placement import StoragePlacementMode

    wrapper, capture, _l1, manifest = _make_wrapper(StoragePlacementMode.KV_SPLIT_TIER)
    try:
        keys = [ObjectKey(chunk_hash=bytes([0x40]) * 32, model_name="m", kv_rank=0)]
        objects = [
            _GroupedMemoryObj(
                tensors=[
                    torch.zeros(torch.Size([4]), dtype=torch.bfloat16),
                    torch.zeros(torch.Size([8]), dtype=torch.bfloat16),
                    torch.zeros(torch.Size([4]), dtype=torch.bfloat16),
                    torch.zeros(torch.Size([8]), dtype=torch.bfloat16),
                ]
            )
        ]
        task_id = wrapper.submit_store_task(keys, objects)
        popped = wrapper.pop_completed_store_tasks()
        assert task_id in popped
        assert not popped[task_id].is_successful()
        # Fail closed: nothing registered, nothing submitted to the inner.
        assert len(manifest) == 0
        assert capture.inner_submit_calls == []
    finally:
        wrapper.close()


def test_split_tier_rejects_non_grouped_object() -> None:
    """A single-group (packed) object under split-tier is a placement /
    layout mismatch and must fail closed, not be treated as K-only."""
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.storage_placement import StoragePlacementMode

    wrapper, _capture, _l1, manifest = _make_wrapper(StoragePlacementMode.KV_SPLIT_TIER)
    try:
        keys = [ObjectKey(chunk_hash=bytes([0x41]) * 32, model_name="m", kv_rank=0)]
        objects = [
            _GroupedMemoryObj(
                tensors=[torch.zeros(torch.Size([4]), dtype=torch.bfloat16)]
            )
        ]
        task_id = wrapper.submit_store_task(keys, objects)
        popped = wrapper.pop_completed_store_tasks()
        assert not popped[task_id].is_successful()
        assert len(manifest) == 0
    finally:
        wrapper.close()


def test_split_tier_registration_collision_rolls_back_batch() -> None:
    """If register_pending collides mid-batch (a concurrent store of
    one key is in flight), the keys registered so far are rolled back,
    the K children released, and the COLLIDING key's pre-existing
    generation is left untouched."""
    # First Party
    from lmcache.v1.distributed.storage_placement import (
        SplitTierState,
        StoragePlacementMode,
        derive_component_key,
    )

    wrapper, capture, _l1, manifest = _make_wrapper(StoragePlacementMode.KV_SPLIT_TIER)
    try:
        # Pre-register the SECOND key (index 1) as an in-flight
        # generation owned by "another" store task.
        # First Party
        from lmcache.v1.distributed.api import ObjectKey

        colliding = ObjectKey(chunk_hash=bytes([0x31]) * 32, model_name="m", kv_rank=0)
        manifest.register_pending(colliding)

        keys, task_id = _submit_split_tier_store(wrapper)
        assert keys[1] == colliding

        # The batch failed; only the pre-existing entry remains.
        assert manifest.lookup(keys[0]) is None
        assert manifest.lookup(colliding) == SplitTierState.STORE_IN_FLIGHT
        popped = wrapper.pop_completed_store_tasks()
        assert task_id in popped
        assert not popped[task_id].is_successful()

        # Both freshly-allocated K children were released.
        k_children = [derive_component_key(k, "k") for k in keys]
        deleted = [k for call in capture.delete_calls for k in call]
        assert all(k in deleted for k in k_children)
    finally:
        wrapper.close()


def test_split_tier_rollback_release_never_drains_k_children() -> None:
    """The failure-path K-child release must NOT fire the
    L1->L2 drain listener (``on_l1_keys_write_finished``).

    On the registration-collision rollback the manifest entries (and
    their is_k_child_key side-set membership) are gone, so a plain
    ``finish_write`` release would drain the write-locked K children to
    the StoreController unfiltered -- routing a single-shape K-only
    buffer into ``submit_store_task`` and leaking a read-locked K child
    if the store loop wins the race.  The release must instead go
    through ``finish_write_and_reserve_read`` (drain-silent) + delete.
    """
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.storage_placement import (
        StoragePlacementMode,
        derive_component_key,
    )

    wrapper, capture, l1, manifest = _make_wrapper(StoragePlacementMode.KV_SPLIT_TIER)

    # A recording drain subscriber standing in for the StoreController
    # listener.  ANY K child reaching it on the rollback path is the bug.
    drained: list[ObjectKey] = []

    class _Recorder:
        def on_l1_keys_write_finished(self, keys) -> None:
            drained.extend(keys)

    l1.register_listener(_Recorder())

    try:
        # Force the mid-batch collision so the store rolls back after
        # allocating + copying both K children.
        colliding = ObjectKey(chunk_hash=bytes([0x31]) * 32, model_name="m", kv_rank=0)
        manifest.register_pending(colliding)

        keys, task_id = _submit_split_tier_store(wrapper)
        assert keys[1] == colliding
        popped = wrapper.pop_completed_store_tasks()
        assert not popped[task_id].is_successful()

        k_children = [derive_component_key(k, "k") for k in keys]

        # The drain listener never saw ANY key (rollback never commits
        # a finish_write), and specifically no K child leaked to it.
        assert drained == []
        assert capture.finish_write_calls == []
        assert not any(k in call for call in capture.drained_keys for k in k_children)

        # The K children were released via the drain-silent path...
        released_via_fwrr = [k for call in capture.fwrr_calls for k in call]
        assert all(k in released_via_fwrr for k in k_children)
        # ...and then deleted from L1.
        deleted = [k for call in capture.delete_calls for k in call]
        assert all(k in deleted for k in k_children)
    finally:
        wrapper.close()


# =============================================================================
# Parent-aware L1 frees -- slab-backed K children return to their pool
# =============================================================================


def _build_l1_and_slab():
    """Fresh real L1Manager + K-child slab + one registered external
    K-child entry (write-committed).

    Returns:
        ``(l1, slab, k_child_key)``.
    """
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.config import (
        L1ManagerConfig,
        L1MemoryManagerConfig,
    )
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import _KChildSlab
    from lmcache.v1.distributed.storage_placement import derive_component_key

    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            init_size_in_bytes=1 << 20,
        )
    )
    l1 = L1Manager(cfg)
    logical = ObjectKey(
        chunk_hash=b"\x66" * 32,
        model_name="m",
        kv_rank=0,
        cache_salt="",
    )
    k_child = derive_component_key(logical, "k")
    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=2)
    objs = slab.batched_allocate(None, None, batch_size=1)
    l1.reserve_external_writes([k_child], objs)
    l1.finish_write([k_child])
    return l1, slab, k_child


def test_external_kchild_delete_returns_buffer_to_slab() -> None:
    """Deleting a slab-backed K child must return the buffer to the
    SLAB, not hand its raw pointer to the L1 memory manager (which
    poisons the pinned pool's free list on the CPU tier and raises on
    Device-DAX, killing the eviction loop)."""
    # First Party
    from lmcache.v1.distributed.error import L1Error

    l1, slab, k_child = _build_l1_and_slab()
    before = slab.stats()
    assert before["in_flight"] == 1
    assert before["free"] == 0

    res = l1.delete([k_child])
    assert res[k_child] == L1Error.SUCCESS

    after = slab.stats()
    assert after["in_flight"] == 0, "slab accounting never refilled"
    assert after["free"] == 1, "buffer not returned to the slab pool"

    # And the pooled buffer is reusable: the next allocate is served
    # from the free deque instead of permanently falling back to the
    # slow path (the post-first-eviction-wave failure shape).
    again = slab.batched_allocate(None, None, batch_size=1)
    assert again is not None and len(again) == 1


def test_external_kchild_clear_routes_to_slab() -> None:
    """clear(force=True) frees every entry -- external ones must
    still route to their parent pool."""
    l1, slab, _k_child = _build_l1_and_slab()
    l1.clear(force=True)
    stats = slab.stats()
    assert stats["in_flight"] == 0
    assert stats["free"] == 1


def test_external_kchild_close_routes_to_slab() -> None:
    """close() frees every entry -- same routing requirement."""
    l1, slab, _k_child = _build_l1_and_slab()
    l1.close()
    stats = slab.stats()
    assert stats["in_flight"] == 0
    assert stats["free"] == 1


def test_own_objects_still_freed_via_memory_manager() -> None:
    """Catalog-owned objects (plain reserve_write) keep flowing
    through the memory manager's free -- the partitioned path must
    not change their lifecycle."""
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.config import (
        L1ManagerConfig,
        L1MemoryManagerConfig,
    )
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.l1_manager import L1Manager

    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            init_size_in_bytes=1 << 20,
        )
    )
    l1 = L1Manager(cfg)
    key = ObjectKey(chunk_hash=b"\x67" * 32, model_name="m", kv_rank=0)
    res = l1.reserve_write(
        keys=[key],
        is_temporary=[False],
        layout_desc=MemoryLayoutDesc(shapes=[torch.Size([16])], dtypes=[torch.float16]),
        mode="new",
    )
    assert res[key][0] == L1Error.SUCCESS
    l1.finish_write([key])
    used_before, _ = l1.get_memory_usage()
    assert used_before > 0
    assert l1.delete([key])[key] == L1Error.SUCCESS
    used_after, _ = l1.get_memory_usage()
    assert used_after < used_before
    l1.close()


# =============================================================================
# Manifest-gated lookup -- lock symmetry for masked V hits
# =============================================================================


def test_split_tier_lookup_masks_incomplete_and_unlocks() -> None:
    """query_lookup_and_lock_result must AND the inner V-hit bitmap
    with the manifest COMPLETE mask (its documented contract) and
    immediately unlock the masked-off V children: callers derive
    their unlocks from the bitmap they receive, so a hidden hit would
    hold its L2 lock forever."""
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.storage_placement import (
        StoragePlacementMode,
        derive_component_key,
    )

    wrapper, capture, _l1, manifest = _make_wrapper(StoragePlacementMode.KV_SPLIT_TIER)
    try:
        complete = ObjectKey(chunk_hash=b"\x71" * 32, model_name="m", kv_rank=0)
        in_flight = ObjectKey(chunk_hash=b"\x72" * 32, model_name="m", kv_rank=0)
        untracked = ObjectKey(chunk_hash=b"\x73" * 32, model_name="m", kv_rank=0)
        complete_gen = manifest.register_pending(complete)
        manifest.mark_complete(complete, complete_gen)
        manifest.register_pending(in_flight)

        layout = MemoryLayoutDesc(shapes=[], dtypes=[])
        task_id = wrapper.submit_lookup_and_lock_task(
            [complete, in_flight, untracked], layout
        )

        # The inner adapter saw V-CHILD keys, not logical keys.
        assert capture.inner_lookup_calls == [
            [
                derive_component_key(complete, "v"),
                derive_component_key(in_flight, "v"),
                derive_component_key(untracked, "v"),
            ]
        ]

        bitmap = wrapper.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        # Only the COMPLETE key survives the gate (the inner reported
        # all three V children as hits).
        assert bitmap.test(0)
        assert not bitmap.test(1)
        assert not bitmap.test(2)

        # The two masked V hits were unlocked on the inner adapter.
        unlocked = [k for call in capture.inner_unlock_calls for k in call]
        assert derive_component_key(in_flight, "v") in unlocked
        assert derive_component_key(untracked, "v") in unlocked
        assert derive_component_key(complete, "v") not in unlocked
    finally:
        wrapper.close()


def test_kv_together_lookup_passes_through_unmodified() -> None:
    """KV_TOGETHER keeps the legacy behavior: keys untranslated, the
    inner bitmap final, no unlocks issued by the wrapper."""
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.storage_placement import StoragePlacementMode

    wrapper, capture, _l1, _manifest = _make_wrapper(StoragePlacementMode.KV_TOGETHER)
    try:
        keys = [
            ObjectKey(chunk_hash=b"\x74" * 32, model_name="m", kv_rank=0),
            ObjectKey(chunk_hash=b"\x75" * 32, model_name="m", kv_rank=0),
        ]
        layout = MemoryLayoutDesc(shapes=[], dtypes=[])
        task_id = wrapper.submit_lookup_and_lock_task(keys, layout)
        assert capture.inner_lookup_calls == [keys]

        bitmap = wrapper.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) and bitmap.test(1)
        assert capture.inner_unlock_calls == []
    finally:
        wrapper.close()
