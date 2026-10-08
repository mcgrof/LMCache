# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import nullcontext
from typing import List, Optional, Union
import importlib
import math
import os
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate, get_size_bytes
from lmcache.v1 import memory_management
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)

logger = init_logger(__name__)
_RETAINED_PROVIDER_ALLOCATORS: list["GPUMemoryAllocator"] = []


class GPUMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated GPU memory."""

    def __init__(
        self,
        size: int,
        device=torch_device_type,
        align_bytes: Optional[int] = None,
        use_paging: bool = False,
        buffer_provider: str = "native",
        **kwargs,
    ) -> None:
        """Create a pool aligned for whole host-page DMA-BUF exports.

        Args:
            size: Minimum pool capacity in bytes. Capacity is rounded up to
                whole host pages and, for paging, whole logical chunk slots.
            device: Device on which the backing tensor is allocated.
            align_bytes: Alignment for allocations when paging is disabled.
            use_paging: Whether fixed-size chunk slots are used.
            buffer_provider: ``native`` (default) or explicit experimental
                ``vulkan_rm`` allocation/export. Vulkan/RM requires CUDA,
                its optional native extension and the supported NVIDIA ABI.
            **kwargs: Paged pools require ``shapes``, ``dtypes``, and ``fmt``.

        Raises:
            ValueError: Pool geometry or provider selection is invalid.
            RuntimeError: The selected provider cannot initialize. Provider
                failure never switches an allocation already handed to callers.
        """
        self._provider_fd = -1
        self._closed = False
        self.buffer_provider = buffer_provider
        if buffer_provider not in ("native", "vulkan_rm"):
            raise ValueError("GPU buffer provider must be native or vulkan_rm")
        if buffer_provider == "vulkan_rm" and (
            torch.version.hip is not None or not torch_dev.is_available()
        ):
            raise RuntimeError(
                "Vulkan/RM staging requires an available NVIDIA CUDA GPU"
            )
        if not torch_dev.is_available():
            device = "cpu"

        # The buffer starts and ends on a host page boundary so it can be
        # exported as a dma-buf (the driver refuses a range that is not page
        # aligned).  Over-allocate by one page and slice to the first page
        # boundary; the extra page is the cost of the guarantee.
        page = os.sysconf("SC_PAGE_SIZE")
        allocation_alignment = page
        if use_paging:
            if not all(name in kwargs for name in ("shapes", "dtypes", "fmt")):
                raise ValueError("paged allocation requires shapes, dtypes, and fmt")
            allocation_alignment = math.lcm(
                page, get_size_bytes(kwargs["shapes"], kwargs["dtypes"])
            )
        aligned_size = (
            (size + allocation_alignment - 1)
            // allocation_alignment
            * allocation_alignment
        )
        if buffer_provider == "native":
            self._backing = torch.empty(
                aligned_size + page, dtype=torch.uint8, device=device
            )
            start = (-self._backing.data_ptr()) % page
            self.tensor = self._backing[start : start + aligned_size]
        else:
            if aligned_size <= 0 or aligned_size > 1 << 30:
                raise ValueError(
                    "Vulkan/RM staging pool must be between one page and 1 GiB"
                )
            selected = torch.device(device)
            if selected.type != "cuda":
                raise ValueError("Vulkan/RM staging requires a CUDA device")
            try:
                helper = importlib.import_module("lmcache._vulkan_rm")
            except ImportError as exc:
                raise RuntimeError(
                    "Vulkan/RM staging needs the optional "
                    "BUILD_WITH_VULKAN_RM=1 extension"
                ) from exc
            ordinal = selected.index
            if ordinal is None:
                ordinal = torch_dev.current_device()
            # The native storage deleter retains all allocation/export owners
            # through every tensor alias; the FD returned here is borrowed.
            tensor, borrowed_fd, identity = helper.allocate(aligned_size, ordinal)
            if (
                tensor.device != torch.device("cuda", ordinal)
                or tensor.dtype != torch.uint8
                or tensor.ndim != 1
                or tensor.numel() != aligned_size
                or not tensor.is_contiguous()
                or tensor.data_ptr() % page
            ):
                raise RuntimeError(
                    "Vulkan/RM helper returned an invalid staging tensor"
                )
            self._backing = tensor
            self.tensor = tensor
            self._provider_fd = os.dup(borrowed_fd)
            logger.info("GPU staging provider=vulkan_rm %s", identity)

        self.allocator: MemoryAllocatorInterface
        if use_paging:
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
        # Native DMA-BUF regions, exported on first request; they let
        # a raw_block backend register the paged buffers with its NVMe device so
        # loads and stores DMA straight to device memory.
        self._dmabuf_regions: Optional[list[tuple[int, int, int]]] = None

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
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
        with self.device_mem_lock:
            if self.buffer_provider == "vulkan_rm" and self._closed:
                return None
            return self.allocator.allocate(shapes, dtypes, fmt, str(self))

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
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
        with self.device_mem_lock:
            if self.buffer_provider == "vulkan_rm" and self._closed:
                return None
            return self.allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt, str(self)
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

    def get_paged_buffers(self) -> Optional[tuple[torch.Tensor, ...]]:
        """Paged buffers for fixed-buffer registration, when paged."""
        if isinstance(self.allocator, PagedTensorMemoryAllocator):
            return self.allocator.get_paged_buffers()
        return None

    def get_paged_dmabuf_regions(self) -> Optional[list[tuple[int, int]]]:
        """Return each paged buffer's (DMA-BUF FD, mapped base).

        Native export occurs on first use. Vulkan/RM exports during pool
        initialization and resolves views against that owned logical extent.

        Returns:
            Regions, or ``None`` when not paged or native export is unavailable.

        Raises:
            RuntimeError: A Vulkan/RM pool is closed or a view exceeds its extent.
        """
        buffers = self.get_paged_buffers()
        if not buffers:
            return None
        if self.buffer_provider == "vulkan_rm":
            if self._closed:
                raise RuntimeError("Vulkan/RM staging pool is closed")
            base = self.tensor.data_ptr()
            end = base + self.tensor.numel()
            if any(
                buf.data_ptr() < base
                or buf.data_ptr() + buf.numel() * buf.element_size() > end
                for buf in buffers
            ):
                raise RuntimeError("Paged buffer exceeds its Vulkan/RM logical extent")
            return [(self._provider_fd, base) for _ in buffers]
        if self._dmabuf_regions is None:
            try:
                self._dmabuf_regions = memory_management.export_device_dmabufs(
                    self.tensor
                )
            except Exception as exc:
                logger.warning(
                    "GPUMemoryAllocator: dma-buf export unavailable (%s); "
                    "device-direct I/O disabled",
                    exc,
                )
                self._dmabuf_regions = []
        if not self._dmabuf_regions:
            return None
        regions = [
            memory_management.get_dmabuf_region(buf.data_ptr()) for buf in buffers
        ]
        if any(r is None for r in regions):
            return None
        return regions  # type: ignore[return-value]

    def close(self) -> None:
        """Release registration FDs after the caller proves storage quiescence.

        Vulkan/RM tensor aliases continue to own the native allocation graph.
        Its final storage deleter waits for CUDA quiescence, then destroys the
        mapping, RM and Vulkan objects. Failed cleanup retains the graph and
        seals provider admission. Closing an allocator does not prove I/O drain.
        """
        if self._closed:
            return
        if self.buffer_provider == "vulkan_rm":
            self._closed = True
            fd, self._provider_fd = self._provider_fd, -1
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    _RETAINED_PROVIDER_ALLOCATORS.append(self)
                    raise
            self.allocator.close()
            return
        try:
            if self._dmabuf_regions:
                memory_management.release_device_dmabufs(self.tensor)
        finally:
            self._dmabuf_regions = []
            self.allocator.close()
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
