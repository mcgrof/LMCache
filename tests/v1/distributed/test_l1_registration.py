# SPDX-License-Identifier: Apache-2.0
"""Stable host registration regions and export ownership for multiprocess L1.

Real exporter cases require LMCACHE_RUN_HOST_DMABUF_IO=1. Opt-in requires a
working /dev/udmabuf and device runtime; dependency failures remain failures.
"""

# Standard
from contextlib import closing
from multiprocessing import shared_memory
from typing import Any, Literal
import fcntl
import os
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L1MemoryRegion
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import get_dmabuf_region

requires_host_exports = pytest.mark.skipif(
    os.environ.get("LMCACHE_RUN_HOST_DMABUF_IO") != "1",
    reason="real host exporter tests require explicit opt-in",
)


@pytest.mark.parametrize("mode", ["fixed", "dmabuf"])
@pytest.mark.no_shared_allocator
@requires_host_exports
def test_l1_registration_descriptor_owns_exports(
    mode: Literal["fixed", "dmabuf"],
) -> None:
    manager = L1MemoryManager(
        L1MemoryManagerConfig(
            4 << 20,
            False,
            shm_name="",
            buffer_registration=mode,
        )
    )
    desc = manager.get_l1_memory_desc()
    assert desc == manager.get_l1_memory_desc()
    assert desc.buffer_registration == mode
    assert len(desc.regions) == 1
    region = desc.regions[0]
    assert (region.ptr, region.size) == (desc.ptr, desc.size)
    assert desc.ptr % desc.align_bytes == 0
    if mode == "dmabuf":
        assert region.dmabuf_fd is not None
        assert get_dmabuf_region(region.ptr) == (region.dmabuf_fd, region.ptr)
        os.fstat(region.dmabuf_fd)
    else:
        assert region.dmabuf_fd is None
    manager.close()
    if mode == "dmabuf":
        assert region.dmabuf_fd is not None
        assert get_dmabuf_region(region.ptr) is None
        with pytest.raises(OSError):
            os.fstat(region.dmabuf_fd)


@pytest.mark.no_shared_allocator
@requires_host_exports
def test_chunked_host_exports_close_as_one_arena() -> None:
    """Each chunk exports its own origin; one allocator owns all descriptors."""
    allocator = MixedMemoryAllocator(
        3 * 65536,
        dmabuf="udmabuf",
        allocation_boundary_bytes=65536,
    )
    ptr = allocator.buffer.data_ptr()
    exports = [get_dmabuf_region(ptr + offset) for offset in range(0, 3 * 65536, 65536)]
    assert all(export is not None for export in exports)
    assert len({export[0] for export in exports if export is not None}) == 3
    allocator.buffer.copy_(torch.arange(3 * 65536, dtype=torch.int64).to(torch.uint8))
    for offset, export in zip(range(0, 3 * 65536, 65536), exports, strict=True):
        assert export is not None
        fd, base = export
        assert base == ptr + offset
        assert os.lseek(fd, 0, os.SEEK_END) == 65536
    allocator.close()
    for export in exports:
        assert export is not None
        fd, base = export
        assert get_dmabuf_region(base) is None
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.no_shared_allocator
@requires_host_exports
def test_partial_export_failure_closes_created_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsuccessful arena export publishes no regions and closes early FDs."""
    ioctl = fcntl.ioctl
    created: list[int] = []

    def fail_second_export(fd: int, request: int, arg: bytearray) -> int:
        if created:
            raise OSError("export refused")
        result = ioctl(fd, request, arg)
        assert isinstance(result, int)
        created.append(result)
        return result

    monkeypatch.setattr(fcntl, "ioctl", fail_second_export)
    with pytest.raises(OSError, match="export refused"):
        MixedMemoryAllocator(131072, dmabuf="udmabuf", allocation_boundary_bytes=65536)
    assert len(created) == 1
    with pytest.raises(OSError):
        os.fstat(created[0])


@pytest.mark.no_shared_allocator
@requires_host_exports
def test_fixed_shm_region_matches_client_mapping() -> None:
    """Alignment must preserve shared-memory offsets used by MP clients."""
    name = "lmcache_l1_pool_registration_" + uuid.uuid4().hex
    manager = L1MemoryManager(
        L1MemoryManagerConfig(
            4 << 20,
            False,
            shm_name=name,
            buffer_registration="fixed",
        )
    )
    _error, objects = manager.allocate(
        MemoryLayoutDesc([torch.Size([4097])], [torch.uint8]),
        1,
    )
    tensor = objects[0].tensor
    assert tensor is not None
    tensor.fill_(73)
    with closing(shared_memory.SharedMemory(name=name)) as client:
        offset = objects[0].metadata.address
        assert bytes(client.buf[offset : offset + 4097]) == bytes([73]) * 4097
    manager.free(objects)
    manager.close()


@pytest.mark.no_shared_allocator
@requires_host_exports
def test_single_allocation_owns_its_padding_only() -> None:
    """Logical views stay short while fixed I/O can use the owned padding."""
    allocator = MixedMemoryAllocator(8192, allocation_boundary_bytes=8192)
    first = allocator.allocate(torch.Size([33]), torch.uint8)
    second = allocator.allocate(torch.Size([41]), torch.uint8)
    assert first is not None and second is not None
    first_tensor, second_tensor = first.tensor, second.tensor
    physical = first.physical_tensor
    assert first_tensor is not None and second_tensor is not None
    assert physical is not None and physical.nbytes == 4096
    assert first_tensor.nbytes == 33 and len(first.byte_array) == 33
    physical.fill_(99)
    second_tensor.fill_(17)
    first.zero_padding()
    assert torch.all(first_tensor == 99)
    assert torch.count_nonzero(physical[33:]) == 0
    assert torch.all(second_tensor == 17)
    allocator.free(first)
    allocator.free(second)
    allocator.close()


@pytest.mark.parametrize(
    "regions",
    [
        (),
        (L1MemoryRegion(8192, 4096),),
        (L1MemoryRegion(4096, 8192),),
        (L1MemoryRegion(4096, 0),),
    ],
)
def test_descriptor_rejects_incomplete_arena_coverage(
    regions: tuple[L1MemoryRegion, ...],
) -> None:
    with pytest.raises(ValueError, match="regions"):
        L1MemoryDesc(4096, 4096, 4096, "fixed", regions)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_lazy": True},
        {"devdax_path": "/dev/dax0.0"},
        {"shm_name": "pool", "buffer_registration": "dmabuf"},
        {"align_bytes": 3},
        {"align_bytes": 2 << 30},
        {"size_in_bytes": 1},
        {"size_in_bytes": (1 << 44) + 1},
        {"buffer_registration": "typo"},
        {"buffer_registration": "dmabuf", "align_bytes": 8192},
        {"shm_name": "pool", "align_bytes": 8192},
    ],
)
def test_registration_config_rejects_unstable_or_invalid_arenas(
    kwargs: dict[str, Any],
) -> None:
    values: dict[str, Any] = dict(
        size_in_bytes=4 << 20, use_lazy=False, shm_name="", buffer_registration="fixed"
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        L1MemoryManagerConfig(**values)
