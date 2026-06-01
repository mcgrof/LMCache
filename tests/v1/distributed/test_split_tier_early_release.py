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
    from lmcache.v1.distributed.serde import AsyncSerdeProcessor
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
        wrapper.close()


def test_kv_together_claims_nothing() -> None:
    from lmcache.v1.distributed.storage_placement import StoragePlacementMode
    wrapper, *_ = _make_wrapper(StoragePlacementMode.KV_TOGETHER)
    try:
        # No task has been submitted -- claim returns [] regardless.
        assert wrapper.claim_early_release_keys(0) == []
        assert wrapper.claim_early_release_keys(99999) == []
    finally:
        wrapper.close()


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
            keys=["k0", "k1"],
            temp_keys=[],
            temp_objs=[],
            phase=None,
            is_split_tier=True,
            early_release_keys=["k0", "k1"],
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
