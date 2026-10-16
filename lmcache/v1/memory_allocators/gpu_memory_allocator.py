# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import nullcontext
from typing import List, Optional, Sequence, Union
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AllocationReceipt,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)


class GPUMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated GPU memory."""

    def __init__(
        self,
        size: int,
        device=torch_device_type,
        align_bytes: Optional[int] = None,
        use_paging: bool = False,
        **kwargs,
    ) -> None:
        """
        :param int size: The size of the GPU memory in bytes.
        :param Optional[int] align_bytes: The byte alignment for allocations.
        """
        if not torch_dev.is_available():
            device = "cpu"

        self.tensor = torch.empty(size, dtype=torch.uint8, device=device)

        self.allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.tensor,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            kwargs = {}
            if align_bytes is not None:
                kwargs["align_bytes"] = align_bytes
            self.allocator = TensorMemoryAllocator(self.tensor, **kwargs)

        self.device_mem_lock = threading.Lock() if not use_paging else nullcontext()

    def supports_allocation_receipts(self) -> bool:
        """Return whether this GPU allocator uses exact nonpaged reservations."""
        return isinstance(self.allocator, TensorMemoryAllocator)

    def can_allocate_batch(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
    ) -> bool:
        """Preflight one exact nonpaged equal-shape allocation batch."""
        if not isinstance(self.allocator, TensorMemoryAllocator):
            raise NotImplementedError(
                "allocation preflight requires the exact nonpaged GPU allocator"
            )
        with self.device_mem_lock:
            return self.allocator.can_allocate_batch(shapes, dtypes, batch_size)

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
        allocation_receipt: Optional[AllocationReceipt] = None,
    ) -> Optional[MemoryObj]:
        """Allocate one GPU-backed memory object.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            fmt: Memory format stored in the returned metadata.
            allocator_type: Optional allocator type string.

        Returns:
            A memory object, or ``None`` if the inner allocator is full.
        """
        if allocation_receipt is not None and not self.supports_allocation_receipts():
            raise NotImplementedError(
                "paged GPU allocations are outside the exact receipt path"
            )
        with self.device_mem_lock:
            if allocation_receipt is None:
                return self.allocator.allocate(shapes, dtypes, fmt, str(self))
            return self.allocator.allocate(
                shapes,
                dtypes,
                fmt,
                str(self),
                allocation_receipt=allocation_receipt,
            )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
        allocation_receipts: Optional[Sequence[AllocationReceipt]] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate multiple GPU-backed memory objects.

        Args:
            shapes: Logical tensor shape or shapes for each allocation.
            dtypes: Logical tensor dtype or dtypes for each allocation.
            batch_size: Number of memory objects to allocate.
            fmt: Memory format stored in each returned object's metadata.
            allocator_type: Optional allocator type string.

        Returns:
            Memory objects, or ``None`` if the inner allocator is full.
        """
        if allocation_receipts is not None and not self.supports_allocation_receipts():
            raise NotImplementedError(
                "paged GPU allocations are outside the exact receipt path"
            )
        with self.device_mem_lock:
            if allocation_receipts is None:
                return self.allocator.batched_allocate(
                    shapes, dtypes, batch_size, fmt, str(self)
                )
            return self.allocator.batched_allocate(
                shapes,
                dtypes,
                batch_size,
                fmt,
                str(self),
                allocation_receipts=allocation_receipts,
            )

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        """Free one GPU-backed memory object.

        Args:
            memory_obj: Memory object to release.
            allocator_type: Optional allocator type string.
        """
        with self.device_mem_lock:
            self.allocator.free(memory_obj)

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Free multiple GPU-backed memory objects.

        Args:
            memory_objs: Memory objects to release.
            allocator_type: Optional allocator type string.
            update_stats: Whether to update allocator statistics.
        """
        with self.device_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self) -> bool:
        """Return whether allocator state is consistent."""
        with self.device_mem_lock:
            return self.allocator.memcheck()

    def __str__(self) -> str:
        return "GPUMemoryAllocator"
