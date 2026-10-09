# SPDX-License-Identifier: Apache-2.0
"""Exercise arena ownership through the public L1 interface."""

# Standard
import gc
import time
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error, L1ReconfigureError, strerror
from lmcache.v1.distributed.l1_manager import L1Manager

pytestmark = pytest.mark.no_shared_allocator


def test_retained_arena_survives_clear_close_and_owner_collection() -> None:
    manager = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(2 << 20, False, shm_name=""),
        )
    )
    key = ObjectKey(ObjectKey.IntHash2Bytes(1), "model", 0)
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    error, obj = manager.reserve_write([key], [False], layout, tag="io")[key]
    assert error is L1Error.SUCCESS and obj is not None and obj.tensor is not None
    obj.tensor.fill_(0xA5)
    manager.retain_backing_resources()
    manager.retain_backing_resources()
    manager.finish_write_and_delete([key], tag="io")
    manager.clear(force=True)
    manager.close()
    manager.close()
    assert obj.tensor is not None and bool(torch.all(obj.tensor == 0xA5))
    assert manager.report_status()["backing_resources_retained"]
    assert not manager.report_status()["is_healthy"]
    owner = weakref.ref(manager)
    del manager
    gc.collect()
    assert owner() is not None
    assert obj.tensor is not None and bool(torch.all(obj.tensor == 0xA5))


def test_retained_arena_refuses_reads_and_allocations() -> None:
    manager = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(2 << 20, False, shm_name=""),
        )
    )
    key = ObjectKey(ObjectKey.IntHash2Bytes(2), "model", 0)
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    assert manager.reserve_write([key], [False], layout)[key][0] is L1Error.SUCCESS
    manager.finish_write([key])
    manager.reserve_read([key])
    manager.retain_backing_resources()
    assert manager.reserve_read([key])[key] == (
        L1Error.BACKING_RESOURCES_RETAINED,
        None,
    )
    assert manager.unsafe_read([key])[key] == (L1Error.BACKING_RESOURCES_RETAINED, None)
    assert manager.reserve_write([key], [False], layout, tag="new")[key] == (
        L1Error.BACKING_RESOURCES_RETAINED,
        None,
    )
    with pytest.raises(L1ReconfigureError, match="retained"):
        manager.remove_devdax_device("/dev/dax-test")
    with pytest.raises(L1ReconfigureError, match="retained"):
        manager.add_devdax_device("/dev/dax-test", 4096)
    assert "retained" in strerror(L1Error.BACKING_RESOURCES_RETAINED)
    manager.close()


def test_released_arena_cannot_be_retained() -> None:
    manager = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(2 << 20, False, shm_name=""),
        )
    )
    manager.close()
    manager.close()
    with pytest.raises(RuntimeError, match="already closed"):
        manager.retain_backing_resources()


@pytest.mark.parametrize("buffers_reusable", [True, False])
def test_io_lease_survives_expiry_and_settles_after_bookkeeping(
    buffers_reusable: bool,
) -> None:
    manager = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(2 << 20, False, shm_name=""),
            write_ttl_seconds=1,
        )
    )
    key = ObjectKey(ObjectKey.IntHash2Bytes(3), "model", 0)
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    _, obj = manager.reserve_write([key], [False], layout, tag="load")[key]
    assert obj is not None and obj.tensor is not None
    obj.tensor.fill_(42)
    lease = manager.acquire_io_lease([key], [obj], tag="load")
    time.sleep(1.1)
    assert not manager.is_key_evictable(key)
    assert (
        manager.reserve_write([key], [False], layout, tag="load")[key][0]
        is L1Error.KEY_NOT_WRITABLE
    )
    assert manager.delete([key], force=True)[key] is L1Error.KEY_IS_LOCKED
    with pytest.raises(RuntimeError, match="external I/O"):
        manager.clear(force=True)
    assert manager.finish_write_and_delete([key], tag="load")[key] is L1Error.SUCCESS
    assert obj.is_valid() and obj.tensor is not None
    assert manager.report_status()["external_io_lease_count"] == 1
    manager.release_io_lease(lease, buffers_reusable=buffers_reusable)
    assert manager.report_status()["external_io_lease_count"] == 0
    assert manager.report_status()["backing_resources_retained"] is not buffers_reusable
    assert obj.is_valid() is not buffers_reusable
    if not buffers_reusable:
        assert obj.tensor is not None and bool(torch.all(obj.tensor == 42))
    with pytest.raises(KeyError):
        manager.release_io_lease(lease, buffers_reusable=True)
    manager.close()


def test_close_with_a_pending_lease_retains_the_arena() -> None:
    manager = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(2 << 20, False, shm_name=""),
        )
    )
    key = ObjectKey(ObjectKey.IntHash2Bytes(4), "model", 0)
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    _, obj = manager.reserve_write([key], [False], layout)[key]
    assert obj is not None
    lease = manager.acquire_io_lease([key], [obj], tag="")
    manager.close()
    assert manager.report_status()["backing_resources_retained"]
    assert obj.is_valid()
    manager.release_io_lease(lease, buffers_reusable=True)
    manager.close()
    assert obj.is_valid()


def test_closed_arena_refuses_new_buffer_access() -> None:
    manager = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(2 << 20, False, shm_name=""),
        )
    )
    key = ObjectKey(ObjectKey.IntHash2Bytes(5), "model", 0)
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager.reserve_read([key])
    with pytest.raises(RuntimeError, match="closed"):
        manager.unsafe_read([key])
    with pytest.raises(RuntimeError, match="closed"):
        manager.reserve_write([key], [False], layout)
