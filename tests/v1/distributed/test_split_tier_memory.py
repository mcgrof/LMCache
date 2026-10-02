# SPDX-License-Identifier: Apache-2.0
"""Verify split-tier pool reuse, overflow and allocation failure contracts."""

# Standard
from concurrent.futures import ThreadPoolExecutor
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.l2_adapters.split_tier_memory import (
    _VScratchSlab,
    _VScratchSlot,
)

pytestmark = pytest.mark.no_shared_allocator


def test_k_child_allocation_failure_does_not_create_memory_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed pooled and overflow allocations must not count as live bytes."""
    # First Party
    from lmcache.v1.distributed.l2_adapters.split_tier_memory import _KChildSlab

    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=1)

    def fail_empty(*_args: object, **_kwargs: object) -> None:
        raise MemoryError("injected tensor allocation failure")

    with monkeypatch.context() as failure:
        failure.setattr(torch, "empty", fail_empty)
        with pytest.raises(MemoryError):
            slab.allocate(None, None)
    assert slab.get_used_capacity_bytes() == (0, 8)

    obj = slab.allocate(None, None)
    assert obj is not None
    try:
        with monkeypatch.context() as failure:
            failure.setattr(torch, "empty", fail_empty)
            with pytest.raises(MemoryError):
                slab.allocate(None, None)
        assert slab.get_used_capacity_bytes() == (8, 8)
    finally:
        slab.free(obj)
        slab.close()
    assert slab.get_used_capacity_bytes()[0] == 0


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
    tensors = [slab.acquire() for _ in range(5)]
    for tensor in tensors:
        slab.release(tensor)
    assert slab.stats()["free"] == 2
    assert slab.stats()["in_flight"] == 0


def test_slab_concurrent_acquire_release_keeps_exact_accounting() -> None:
    """A burst cannot duplicate a buffer or lose in-flight updates."""
    workers = 16
    slab = _VScratchSlab(shape=torch.Size([64]), dtype=torch.float16, max_slots=4)
    barrier = threading.Barrier(workers)
    active: set[int] = set()
    active_lock = threading.Lock()
    duplicates: list[int] = []

    def borrow_once() -> None:
        tensor = slab.acquire()
        address = tensor.data_ptr()
        with active_lock:
            if address in active:
                duplicates.append(address)
            active.add(address)
        barrier.wait()
        with active_lock:
            active.remove(address)
        slab.release(tensor)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda _index: borrow_once(), range(workers)))

    assert duplicates == []
    assert active == set()
    assert slab.stats()["in_flight"] == 0
    assert slab.stats()["free"] == 4
    assert slab.get_used_capacity_bytes()[0] == 0


# =============================================================================
# _VScratchSlot — minimal MemoryObj-like used by the V-only codec
# =============================================================================


def test_v_scratch_slot_exposes_tensor() -> None:
    t = torch.arange(8, dtype=torch.float16)
    slot = _VScratchSlot(t)
    assert slot.tensor is t


def test_kchild_slab_allocate_returns_tensor_memory_obj() -> None:
    # First Party
    from lmcache.v1.distributed.l2_adapters.split_tier_memory import _KChildSlab
    from lmcache.v1.memory_management import TensorMemoryObj

    slab = _KChildSlab(shape=torch.Size([4, 8]), dtype=torch.float16, max_slots=4)
    obj = slab.allocate(None, None)
    assert isinstance(obj, TensorMemoryObj)
    assert obj.meta.shape == torch.Size([4, 8])
    assert obj.meta.dtype == torch.float16
    # The MemoryObj's parent_allocator must be the slab so free comes back.
    assert obj.parent_allocator is slab


def test_kchild_slab_batched_allocate_returns_n_objects() -> None:
    # First Party
    from lmcache.v1.distributed.l2_adapters.split_tier_memory import _KChildSlab

    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=4)
    objs = slab.batched_allocate(None, None, batch_size=3)
    assert len(objs) == 3
    # Each backing tensor is distinct memory.
    addrs = {o.meta.address for o in objs}
    assert len(addrs) == 3


def test_kchild_slab_free_pools_for_reuse() -> None:
    # First Party
    from lmcache.v1.distributed.l2_adapters.split_tier_memory import _KChildSlab

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
    # First Party
    from lmcache.v1.distributed.l2_adapters.split_tier_memory import _KChildSlab

    slab = _KChildSlab(shape=torch.Size([4]), dtype=torch.float16, max_slots=1)
    o1 = slab.allocate(None, None)
    o2 = slab.allocate(None, None)
    # Both valid, distinct objects; over-subscription counted.
    assert o1 is not o2
    assert slab.stats()["over_subscriptions"] == 1


def test_kchild_slab_free_rejects_mismatched_shape() -> None:
    # First Party
    from lmcache.v1.distributed.l2_adapters.split_tier_memory import _KChildSlab
    from lmcache.v1.memory_management import (
        MemoryFormat,
        MemoryObjMetadata,
        TensorMemoryObj,
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


def test_kchild_slab_concurrent_allocate_free_keeps_exact_accounting() -> None:
    """Concurrent K borrowers never share storage and return all capacity."""
    # First Party
    from lmcache.v1.distributed.l2_adapters.split_tier_memory import _KChildSlab

    workers = 16
    slab = _KChildSlab(shape=torch.Size([64]), dtype=torch.float16, max_slots=4)
    barrier = threading.Barrier(workers)
    active: set[int] = set()
    active_lock = threading.Lock()
    duplicates: list[int] = []

    def borrow_once() -> None:
        obj = slab.allocate(None, None)
        assert obj is not None
        address = obj.raw_data.data_ptr()
        with active_lock:
            if address in active:
                duplicates.append(address)
            active.add(address)
        barrier.wait()
        with active_lock:
            active.remove(address)
        slab.free(obj)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda _index: borrow_once(), range(workers)))

    assert duplicates == []
    assert active == set()
    assert slab.stats()["in_flight"] == 0
    assert slab.stats()["free"] == 4
    assert slab.get_used_capacity_bytes()[0] == 0
