# SPDX-License-Identifier: Apache-2.0
"""Validate eager allocator objects before registered native span transfers."""

# Third Party
import torch

# First Party
from lmcache.v1.distributed.internal_api import L1_REGISTRATION_WINDOW_BYTES
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryObj


def require_registered_span_tensor(
    obj: MemoryObj, device: torch.device, size: int
) -> torch.Tensor:
    """Resolve one complete BF16 object in an eager pinned or device arena.

    Args:
        obj: Read-locked or write-reserved object with a physical transfer lease.
        device: Exact registered CUDA device, including its ordinal.
        size: Complete logical object size in bytes, including padded tokens.

    Returns:
        The contiguous tensor without copying or changing its owner.

    Raises:
        ValueError: The object, dtype, device, size or allocator bounds differ.

    The eager host wrapper retains object ownership; the GPU wrapper delegates
    it to its inner TensorMemoryAllocator. Validate the actual arena and offset
    rather than trusting a caller-supplied pointer. This check does not acquire
    a lease or prove I/O or device completion.
    """
    parent = obj.parent()
    allocator = (
        parent.pin_allocator if isinstance(parent, MixedMemoryAllocator) else parent
    )
    tensor = obj.get_tensor(0)
    if (
        device.type != "cuda"
        or device.index is None
        or not isinstance(allocator, TensorMemoryAllocator)
        or tensor is None
        or not obj.is_valid()
        or tensor.device not in (torch.device("cpu"), device)
        or allocator.buffer.device != tensor.device
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
        or tensor.nbytes != size
        or obj.get_size() != size
        or obj.meta.address < 0
        or obj.meta.phy_size < size
        or obj.meta.address + obj.meta.phy_size > allocator.buffer.nbytes
        or tensor.data_ptr() != allocator.buffer.data_ptr() + obj.meta.address
        or obj.data_ptr != tensor.data_ptr()
    ):
        raise ValueError("span requires a complete allocator-owned source object")
    if tensor.device.type == "cpu":
        start = obj.meta.address
        boundary = L1_REGISTRATION_WINDOW_BYTES
        if (
            not tensor.is_pinned()
            or start // boundary != (start + size - 1) // boundary
        ):
            raise ValueError("span requires one pinned host registration window")
    return tensor
