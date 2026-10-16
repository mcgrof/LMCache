# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List, Optional, Sequence, Union

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import (
    AddressManager,
    AllocationReceipt,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
    logger,
)


class TensorMemoryAllocator(MemoryAllocatorInterface):
    """
    Implements a "explicit list" memory allocator.
    Uses AddressManager for address space management.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        align_bytes: int = AddressManager.ALIGN_BYTES,
        init_address_space: int | None = None,
    ) -> None:
        """
        Args:
            tensor: The pre-allocated flat tensor to use as the memory pool.
            align_bytes: The alignment requirement for allocations.
            init_address_space: Initial size of the address space. If None,
                use the size of the provided tensor.

        Note:
            The `init_address_space` is used for lazy memory allocation.
            We probably want to have a better way to make sure that the
            LazyMemoryAllocator can be decoupled from TensorMemoryAllocator.
        """
        self.buffer = tensor.view(torch.uint8).flatten()

        # Use AddressManager for address space management
        self.address_manager = AddressManager(
            self.buffer.numel() if init_address_space is None else init_address_space,
            align_bytes,
        )

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    @property
    def num_active_allocations(self) -> int:
        """Authoritative live object count from AddressManager state."""
        return self.address_manager.active_allocation_count

    @property
    def total_allocated_size(self) -> int:
        """Return the total currently allocated bytes."""
        return self.address_manager.total_allocated_size

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
        allocation_receipt: Optional[AllocationReceipt] = None,
    ) -> Optional[TensorMemoryObj]:
        """Allocate one tensor-backed memory object.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            fmt: Memory format stored in the returned metadata.
            allocator_type: Optional parent allocator identifier.

        Returns:
            A tensor-backed memory object, or ``None`` if no block is available.
        """
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)

        # Calculate the size of the tensor
        raw_size = get_size_bytes(shapes, dtypes)

        receipt = allocation_receipt or AllocationReceipt()

        # Allocate from address manager. The receipt is bound before the
        # immutable address-state publication, so an outer owner can recover
        # even if this call never constructs or returns a MemoryObj.
        try:
            block_start, aligned_size = self.address_manager.allocate_once(
                raw_size, receipt
            )
        except RuntimeError:
            # No block found
            return None

        # Allocate the block
        raw_data = self._get_buffer_slice(block_start, raw_size)
        memory_obj = TensorMemoryObj(
            raw_data=raw_data,
            metadata=MemoryObjMetadata(
                shapes[0],
                dtypes[0],
                block_start,
                aligned_size,
                1,
                0,
                fmt,
                shapes=shapes,
                dtypes=dtypes,
            ),
            parent_allocator=self,
            allocation_receipt=receipt,
        )
        # This one immutable receipt-state assignment publishes the object and
        # its initial-reference release token before the object can escape.
        receipt.adopt(memory_obj)
        self._sync_debug_stats()
        return memory_obj

    def _get_buffer_slice(self, start: int, size: int) -> torch.Tensor:
        """Hook: Get buffer slice. Override for custom buffer access."""
        return self.buffer[start : start + size]

    def _sync_debug_stats(self) -> None:
        """Refresh observational metrics from authoritative address state."""
        self.stats_monitor.update_local_cache_usage(
            self.address_manager.total_allocated_size
        )
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
        allocation_receipts: Optional[Sequence[AllocationReceipt]] = None,
    ) -> Optional[List[TensorMemoryObj]]:
        """Allocate multiple equal-sized tensor-backed memory objects.

        Args:
            shapes: Logical tensor shape or shapes for each allocation.
            dtypes: Logical tensor dtype or dtypes for each allocation.
            batch_size: Number of memory objects to allocate.
            fmt: Memory format stored in each returned object's metadata.
            allocator_type: Optional parent allocator identifier.

        Returns:
            A list of tensor-backed memory objects, or ``None`` if there is
            not enough contiguous free space for the full batch.
        """
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)

        # Calculate the size of the tensor
        unit_raw_size = get_size_bytes(shapes, dtypes)
        unit_aligned_size = self.address_manager.compute_aligned_size(unit_raw_size)

        if allocation_receipts is None:
            receipts = [AllocationReceipt() for _ in range(batch_size)]
        else:
            if len(allocation_receipts) != batch_size:
                raise ValueError("allocation receipt count does not match batch size")
            receipts = list(allocation_receipts)

        try:
            alloc_results = self.address_manager.batched_allocate_once(
                unit_aligned_size, receipts
            )
        except RuntimeError:
            return None
        addresses = [addr for addr, _ in alloc_results]
        raw_datas = [
            self._get_buffer_slice(addr, unit_aligned_size) for addr in addresses
        ]

        tensor_mem_objs = []
        for raw_data, address, receipt in zip(
            raw_datas, addresses, receipts, strict=True
        ):
            memory_obj = TensorMemoryObj(
                raw_data=raw_data,
                metadata=MemoryObjMetadata(
                    shapes[0],
                    dtypes[0],
                    address,
                    unit_aligned_size,
                    1,
                    0,
                    fmt,
                    shapes=shapes,
                    dtypes=dtypes,
                ),
                parent_allocator=self,
                allocation_receipt=receipt,
            )
            receipt.adopt(memory_obj)
            tensor_mem_objs.append(memory_obj)

        self._sync_debug_stats()
        return tensor_mem_objs

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        """Free one tensor-backed memory object.

        Args:
            memory_obj: Memory object to release back to the allocator.
            allocator_type: Optional allocator type string.
        """
        receipt = (
            memory_obj.allocation_receipt
            if isinstance(memory_obj, TensorMemoryObj)
            else None
        )
        if receipt is not None:
            # Run this even for an invalidated object: interruption may have
            # landed after the address-state commit but before the receipt's
            # completion publication. Replaying the token cannot touch reuse.
            self.address_manager.free_once(receipt)
            memory_obj.invalidate()
            self._sync_debug_stats()
            return
        if not memory_obj.is_valid():
            return

        self.address_manager.free(memory_obj.meta.address, memory_obj.meta.phy_size)
        memory_obj.invalidate()
        self._sync_debug_stats()

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Free multiple tensor-backed memory objects.

        Unlike `batched_allocate`, this function does not
        assume that the memory objs are equal-sized.

        Args:
            memory_objs: Memory objects to release back to the allocator.
            allocator_type: Optional allocator type string.
            update_stats: Whether to update allocator statistics.
        """
        if not memory_objs:
            return

        exact_memory_objs: list[MemoryObj] = []
        legacy_memory_objs: list[MemoryObj] = []
        for memory_obj in memory_objs:
            if (
                isinstance(memory_obj, TensorMemoryObj)
                and memory_obj.allocation_receipt is not None
            ):
                exact_memory_objs.append(memory_obj)
            else:
                legacy_memory_objs.append(memory_obj)

        # Exact generations cannot be coalesced by address: their individual
        # tokens are what prevent a delayed free from touching address reuse.
        for memory_obj in exact_memory_objs:
            assert isinstance(memory_obj, TensorMemoryObj)
            assert memory_obj.allocation_receipt is not None
            self.address_manager.free_once(memory_obj.allocation_receipt)
            memory_obj.invalidate()

        # Coalesce adjacent legacy objects before freeing to reduce
        # the number of free operations
        coalesced_blocks: list[tuple[int, int, int]] = []  # (address, size, count)
        curr_start = None
        curr_size = 0
        curr_count = 0

        legacy_memory_objs.sort(key=lambda x: x.meta.address)
        for memory_obj in legacy_memory_objs:
            if not memory_obj.is_valid():
                logger.warning("Trying to free an invalidated MemoryObj")
                continue
            memory_obj.invalidate()

            if curr_start is None:
                curr_start = memory_obj.meta.address
                curr_size = memory_obj.meta.phy_size
                curr_count = 1
            elif curr_start + curr_size == memory_obj.meta.address:
                # Adjacent block, extend current
                curr_size += memory_obj.meta.phy_size
                curr_count += 1
            else:
                # Non-adjacent, save current and start new
                coalesced_blocks.append((curr_start, curr_size, curr_count))
                curr_start = memory_obj.meta.address
                curr_size = memory_obj.meta.phy_size
                curr_count = 1

        if curr_start is not None:
            coalesced_blocks.append((curr_start, curr_size, curr_count))

        # Free all coalesced blocks
        for address, size, _count in coalesced_blocks:
            self.address_manager.free(address, size)

        if update_stats:
            self._sync_debug_stats()

    def memcheck(self) -> bool:
        """Check allocator consistency for debugging.

        Returns:
            True if allocator accounting is internally consistent, otherwise
            False.
        """
        clear = True
        logger.info("Checking memory allocator consistency")
        logger.info(" - Total active allocations: %d", self.num_active_allocations)
        logger.info(
            " - Total allocated size: %f MB",
            self.address_manager.total_allocated_size / 1048576,
        )

        # Check the real total free size
        total_free_size = self.address_manager.get_free_size()
        logger.info(" - Total free size: %f MB", total_free_size / 1048576)

        # Check if the numbers are consistent
        if (
            total_free_size + self.address_manager.total_allocated_size
            != self.address_manager.get_heap_size()
        ):
            logger.error("Memory allocator size is inconsistent")
            logger.error("This implies a bug in the memory allocator")
            clear = False

        # Check if the blocks are coalesced
        if not self.address_manager.check_consistency():
            logger.error("Memory allocator has non-coalesced blocks")
            logger.error("This implies a bug in the memory allocator")
            clear = False

        return clear

    def __str__(self) -> str:
        """Return the allocator name."""
        return "TensorMemoryAllocator"
