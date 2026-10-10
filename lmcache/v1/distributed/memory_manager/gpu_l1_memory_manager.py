# SPDX-License-Identifier: Apache-2.0
"""Own a process-local GPU L1 arena and its borrowed DMA-BUF exports."""

# First Party
from lmcache import torch_dev
from lmcache.v1.distributed.api import L1BackendType
from lmcache.v1.distributed.config import GPUL1ManagerConfig
from lmcache.v1.distributed.internal_api import (
    L1_REGISTRATION_WINDOW_BYTES,
    L1MemoryDesc,
    L1MemoryRegion,
)
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_management import MemoryObj


class GPUL1MemoryManager(L1MemoryManager):
    """Use GPU memory as the server's L1 staging medium.

    Engine KV tensors reach this arena through the existing CUDA IPC and
    gather/scatter path. The arena itself stays in its owning server process.
    DMA-BUF export failure aborts setup without host-memory fallback.
    """

    def __init__(self, config: GPUL1ManagerConfig) -> None:
        if not torch_dev.is_available():
            raise RuntimeError("GPU L1 requires an available device runtime")
        memory = config.memory_config
        self._gpu_allocator = GPUMemoryAllocator(
            memory.size_in_bytes,
            device=config.device,
            align_bytes=memory.align_bytes,
            buffer_provider=config.buffer_provider,
            allocation_boundary_bytes=L1_REGISTRATION_WINDOW_BYTES,
        )
        self._allocator = self._gpu_allocator
        self._size_in_bytes = memory.size_in_bytes
        self._align_bytes = memory.align_bytes
        self._buffer_registration = "dmabuf"
        try:
            self._regions = self._gpu_allocator.get_registration_regions()
        except (RuntimeError, OSError, ValueError):
            self._allocator.close()
            raise

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """Return borrowed GPU windows and export FDs owned by this arena."""
        return L1MemoryDesc(
            ptr=self._gpu_allocator.tensor.data_ptr(),
            size=self._gpu_allocator.tensor.nbytes,
            align_bytes=self._align_bytes,
            buffer_registration="dmabuf",
            regions=tuple(
                L1MemoryRegion(window.data_ptr(), window.nbytes, fd, base)
                for window, fd, base in self._regions
            ),
            device=str(self._gpu_allocator.tensor.device),
        )

    def get_backend_type(self, memory_obj: MemoryObj) -> L1BackendType:
        """Report GPU memory as the backing medium."""
        return L1BackendType.GPU
