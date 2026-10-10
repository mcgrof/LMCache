# SPDX-License-Identifier: Apache-2.0
"""Check GPU allocator accounting and registration admission contracts."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator


def test_unpaged_registration_boundary_and_accounting() -> None:
    """Keep physical padding within each configured registration window."""
    allocator = GPUMemoryAllocator(
        16384, device="cpu", align_bytes=4096, allocation_boundary_bytes=8192
    )
    first = allocator.allocate(torch.Size([5000]), torch.uint8)
    second = allocator.allocate(torch.Size([4097]), torch.uint8)
    assert first is not None and second is not None
    try:
        assert first.physical_tensor is not None and second.physical_tensor is not None
        assert first.physical_tensor.nbytes == second.physical_tensor.nbytes == 8192
        assert (
            second.physical_tensor.data_ptr() - first.physical_tensor.data_ptr() == 8192
        )
        assert allocator.get_memory_usage() == (16384, 16384)
        allocator.free(first)
        allocator.free(second)
        assert allocator.get_memory_usage() == (0, 16384)
    finally:
        allocator.close()
    with pytest.raises(RuntimeError, match="closed"):
        allocator.get_registration_regions()


def test_paged_allocations_cannot_use_unpaged_registration_boundaries() -> None:
    """Reject an incompatible request before allocating the GPU pool."""
    with pytest.raises(ValueError, match="unpaged"):
        GPUMemoryAllocator(
            16384, device="cpu", use_paging=True, allocation_boundary_bytes=8192
        )
