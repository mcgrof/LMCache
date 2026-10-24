# SPDX-License-Identifier: Apache-2.0
"""Check actual CPU L1 admission outcomes before logical span publication."""

# Standard
from collections.abc import Callable, Iterator

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import (
    DRAML1ManagerConfig,
    EvictionConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.storage_manager import L1WriteCompletion, StorageManager
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext


@pytest.fixture
def storage() -> Iterator[StorageManager]:
    context = MPCacheServerContext(
        StorageManagerConfig(
            DRAML1ManagerConfig(L1MemoryManagerConfig(8 << 20, False, shm_name="")),
            EvictionConfig("LRU"),
        ),
        chunk_size=8,
    )
    try:
        yield context.storage_manager
    finally:
        context.close()


def inputs() -> tuple[list[ObjectKey], MemoryLayoutDesc]:
    """Describe two synthetic 512-byte BF16 chunks with immutable distinct keys."""
    return (
        [ObjectKey(bytes([n]) * 32, "synthetic-admission", 0) for n in (1, 2)],
        MemoryLayoutDesc([torch.Size((2, 1, 8, 16))], [torch.bfloat16]),
    )


def test_write_status_reports_admission_and_actual_readback(
    storage: StorageManager,
) -> None:
    keys, layout = inputs()
    objects = storage.reserve_write(keys, layout)
    assert set(objects) == set(keys)
    for index, key in enumerate(keys):
        tensor = objects[key].get_tensor(0)
        assert tensor is not None and tensor.device.type == "cpu"
        tensor.fill_(7 + index)
    completion = storage.prepare_write_completion(objects)
    assert storage.finish_write_with_status(completion) == dict.fromkeys(
        keys, L1Error.SUCCESS
    )
    handle = storage.submit_prefetch_task(
        PrefetchTaskSpec([GroupedObjectKeys(keys, 0, layout)]), "admission-readback"
    )
    assert storage.wait_prefetch_status(handle, timeout=10)
    assert storage.query_prefetch_status(handle) is not None
    with storage.read_prefetched_results(keys) as read:
        assert read is not None and len(read) == 2
        for index, obj in enumerate(read):
            tensor = obj.get_tensor(0)
            assert tensor is not None and torch.all(tensor == 7 + index).item()
    storage.finish_read_prefetched(keys)


def test_write_status_exposes_partial_admission(storage: StorageManager) -> None:
    keys, layout = inputs()
    objects = storage.reserve_write(keys, layout)
    completion = storage.prepare_write_completion(objects)
    first = storage.prepare_write_completion({keys[0]: objects[keys[0]]})
    storage.finish_write_by_owner(first)
    assert storage.finish_write_with_status(completion) == {
        keys[0]: L1Error.KEY_NOT_EXIST,
        keys[1]: L1Error.SUCCESS,
    }


def test_write_status_validates_all_owners_before_admission(
    storage: StorageManager,
) -> None:
    keys, layout = inputs()
    objects = storage.reserve_write(keys, layout)
    completion = storage.prepare_write_completion(objects)
    with pytest.raises(ValueError, match="registered L1 owner"):
        storage.finish_write_with_status(completion + [(-1, keys)])
    assert storage.finish_write_with_status(completion) == dict.fromkeys(
        keys, L1Error.SUCCESS
    )


def test_legacy_owner_completion_keeps_void_result(storage: StorageManager) -> None:
    keys, layout = inputs()
    completion = storage.prepare_write_completion(storage.reserve_write(keys, layout))
    legacy_finish: Callable[[L1WriteCompletion], object] = storage.finish_write_by_owner
    assert legacy_finish(completion) is None
    assert storage.finish_write_with_status(completion) == dict.fromkeys(
        keys, L1Error.KEY_NOT_EXIST
    )


def large_inputs() -> tuple[ObjectKey, ObjectKey, MemoryLayoutDesc]:
    """Only one 6 MiB object fits in this fixture's 8 MiB CPU arena."""
    return (
        ObjectKey(b"a" * 32, "synthetic-lease", 0),
        ObjectKey(b"b" * 32, "synthetic-lease", 0),
        MemoryLayoutDesc([torch.Size((6 << 20,))], [torch.uint8]),
    )


def test_host_write_lease_prevents_reuse_after_reservation_discard(
    storage: StorageManager,
) -> None:
    first, second, layout = large_inputs()
    objects = storage.reserve_write([first], layout)
    assert set(objects) == {first}
    assert storage.acquire_write_transfer_leases(objects) == []
    leases = storage.acquire_write_transfer_leases(objects, all_buffers=True)
    assert len(leases) == 1
    try:
        storage.discard_write_by_owner(storage.prepare_write_completion(objects))
        assert storage.reserve_write([second], layout) == {}
    finally:
        storage.release_transfer_leases(leases)
    replacement = storage.reserve_write([second], layout)
    assert set(replacement) == {second}
    storage.discard_write_by_owner(storage.prepare_write_completion(replacement))


def test_host_read_lease_prevents_reuse_after_reader_release_and_eviction(
    storage: StorageManager,
) -> None:
    first, second, layout = large_inputs()
    objects = storage.reserve_write([first], layout)
    storage.finish_write_by_owner(storage.prepare_write_completion(objects))
    handle = storage.submit_prefetch_task(
        PrefetchTaskSpec([GroupedObjectKeys([first], 0, layout)]),
        "lease-readback",
        skip_l2=True,
    )
    assert storage.wait_prefetch_status(handle, timeout=10)
    result = storage.query_prefetch_status(handle)
    assert result is not None and result.hit_cells[0].test(0)
    keys, read = storage.unsafe_read([first], result.l1_owners)
    assert keys == [first] and len(read) == 1
    retained = dict(zip(keys, read, strict=True))
    assert storage.acquire_read_transfer_leases(retained) == []
    leases = storage.acquire_read_transfer_leases(retained, all_buffers=True)
    assert len(leases) == 1
    try:
        storage.finish_read_prefetched([first], l1_owners=result.l1_owners)
        assert storage.delete_l1_keys([first]) == (0, 1)
        assert storage.reserve_write([second], layout) == {}
    finally:
        storage.release_transfer_leases(leases)
    assert storage.delete_l1_keys([first]) == (1, 0)
    replacement = storage.reserve_write([second], layout)
    assert set(replacement) == {second}
    storage.discard_write_by_owner(storage.prepare_write_completion(replacement))
