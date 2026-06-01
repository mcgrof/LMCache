# SPDX-License-Identifier: Apache-2.0
"""CPU pinned-DRAM L1 memory manager."""

# Standard
from multiprocessing import shared_memory

# Standard
from typing import Protocol, runtime_checkable
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryObj,
    MixedMemoryAllocator,
)


@runtime_checkable
class L1MemoryUsageProvider(Protocol):
    """Auxiliary memory providers (e.g. SerdeL2AdapterWrapper's K-child
    slab) that hold L1-resident bytes outside ``L1MemoryManager``'s own
    allocator implement this protocol so :meth:`L1MemoryManager.get_memory_usage`
    can aggregate the true L1 footprint.

    Without this, slab K-children -- which bypass
    ``L1MemoryManager.allocate`` -- are invisible to the LRU eviction
    policy, and under sustained L1 pressure eviction never fires on
    those entries even though they occupy real memory.
    """

    def get_used_capacity_bytes(self) -> tuple[int, int]:
        """Return ``(used_bytes, capacity_bytes)`` for this provider.

        Returns:
            ``used_bytes``: bytes currently held by live MemoryObjs the
                provider owns (i.e. *not* sitting on its free list).
            ``capacity_bytes``: the provider's configured upper bound.
                Used by the eviction policy as the denominator of its
                pressure ratio.  May be larger than the bytes actually
                allocated so far if the provider grows lazily.
        """
        ...

logger = init_logger(__name__)


# HELPER FUNCTIONS
def _unlink_stale_shm(shm_name: str) -> None:
    """Remove a stale LMCache shm segment if it exists."""
    normalized = shm_name.lstrip("/")
    if "/" in normalized or "\\" in normalized:
        logger.warning("Refusing to unlink invalid shm name %s", shm_name)
        return
    if not normalized.startswith("lmcache_l1_pool_"):
        return
    try:
        shm = shared_memory.SharedMemory(name=normalized, create=False)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        return
    except OSError:
        logger.warning(
            "Failed to remove stale shm segment %s", normalized, exc_info=True
        )


def create_memory_allocator(config: L1MemoryManagerConfig) -> MemoryAllocatorInterface:
    """
    Create a memory allocator based on the provided configuration.

    Args:
        config (L1MemoryManagerConfig): Configuration for the memory manager.

    Returns:
        MemoryAllocatorInterface: An instance of a memory allocator.
    """
    if config.use_lazy:
        logger.debug(
            "use lazy memory allocator, init size is %d bytes, "
            "final size is %d bytes, align bytes is %d bytes",
            config.init_size_in_bytes,
            config.size_in_bytes,
            config.align_bytes,
        )
        return LazyMemoryAllocator(
            config.init_size_in_bytes, config.size_in_bytes, config.align_bytes
        )
    else:
        logger.debug(
            "use mixed memory allocator, total size is %d bytes, "
            "align bytes is %d bytes",
            config.size_in_bytes,
            config.align_bytes,
        )
        shm_name = config.shm_name
        if shm_name:
            # Keep the lmcache_l1_pool_ prefix in normalized SHM names so
            # stale-segment cleanup can recognize and unlink user-provided names.
            bare = shm_name.lstrip("/")
            if not bare.startswith("lmcache_l1_pool_"):
                shm_name = f"lmcache_l1_pool_{bare}"
            _unlink_stale_shm(shm_name)
            return MixedMemoryAllocator(
                config.size_in_bytes,
                align_bytes=config.align_bytes,
                shm_name=shm_name,
            )
        return MixedMemoryAllocator(
            config.size_in_bytes,
            align_bytes=config.align_bytes,
        )


# MAIN CLASS
class L1MemoryManager:
    """
    L1MemoryManager manages the allocation and deallocation of L1 memory.

    Observability metrics to emit:
    1. Memory usage
    2. Active allocations
    """

    def __init__(self, config: L1MemoryManagerConfig):
        self._allocator = create_memory_allocator(config)
        self._size_in_bytes = config.size_in_bytes
        self._align_bytes = config.align_bytes
        # External usage providers (e.g. SerdeL2AdapterWrapper's K-child
        # slab) registered via :meth:`register_external_memory_provider`.
        # Their bytes get summed into the reading returned by
        # :meth:`get_memory_usage` so the eviction policy sees true L1
        # pressure rather than the address-manager subset.
        self._external_providers: list[L1MemoryUsageProvider] = []
        self._external_lock = threading.Lock()

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """
        Allocate memory objects based on the provided layout description and count.
        This function should be thread-safe

        Args:
            layout_desc (MemoryLayoutDesc): Description of the memory layout.
            count (int): Number of memory objects to allocate.

        Returns:
            tuple[L1Error, list[MemoryObj]]: Error code and list of
            allocated memory objects.
            Error code will be `L1Error.OUT_OF_MEMORY` if allocation
            fails; otherwise, it will be `L1Error.SUCCESS`.

        Note:
            If the allocation fails, the memory object list will be empty.
        """
        objects = self._allocator.batched_allocate(
            layout_desc.shapes, layout_desc.dtypes, count
        )
        if objects is None:
            return L1Error.OUT_OF_MEMORY, []
        return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """
        Free the provided memory objects.
        This function should be thread-safe.

        Args:
            mem_objs (list[MemoryObj]): List of memory objects to free.

        Returns:
            L1Error: Error code indicating the result of the operation.
            It will be `L1Error.SUCCESS` if the operation succeeds.
        """
        self._allocator.batched_free(mem_objs)
        return L1Error.SUCCESS

    def get_memory_usage(self) -> tuple[int, int]:
        """
        Get the current memory usage. This function will mainly be used to support
        eviction decision.

        Returns:
            tuple[int, int]: A tuple containing used memory in bytes and total memory
            in bytes.

        Note:
            In the future, we may want to make a "callback" based mechanism to
            trigger eviction when the memory usage reaches a watermark.
        """

        if hasattr(self._allocator, "get_memory_usage"):
            return self._allocator.get_memory_usage()

        def get_address_manager(allocator: MemoryAllocatorInterface):
            if isinstance(allocator, MixedMemoryAllocator) and hasattr(
                allocator.pin_allocator, "address_manager"
            ):
                return allocator.pin_allocator.address_manager
            if isinstance(allocator, LazyMemoryAllocator):
                return allocator.get_address_manager()
            raise NotImplementedError(
                "get_memory_usage is not implemented for this allocator type."
            )

        address_manager = get_address_manager(self._allocator)
        free_size = address_manager.get_free_size()
        total_size = address_manager.get_heap_size()
        used_size = total_size - free_size

        # Aggregate external providers (slabs etc.).  Held briefly so a
        # late register/unregister can't race a concurrent read; each
        # provider's own accounting is responsible for thread safety on
        # the actual counters.
        with self._external_lock:
            providers = list(self._external_providers)
        for provider in providers:
            try:
                p_used, p_total = provider.get_used_capacity_bytes()
            except Exception:
                # A misbehaving provider must not crash the eviction
                # decision loop; treat as zero contribution and move on.
                logger.exception(
                    "L1MemoryManager: external provider %r raised in "
                    "get_used_capacity_bytes; skipping",
                    type(provider).__name__,
                )
                continue
            used_size += p_used
            total_size += p_total
        return used_size, total_size

    def register_external_memory_provider(
        self, provider: L1MemoryUsageProvider
    ) -> None:
        """Register an auxiliary memory provider whose bytes count
        toward :meth:`get_memory_usage`.

        Use case: ``SerdeL2AdapterWrapper`` carries slab pools
        (`_KChildSlab`) that hold L1-resident bytes outside
        ``L1MemoryManager``'s own allocator.  Without this hook those
        bytes are invisible to the LRU eviction policy.

        Idempotent: re-registering the same provider is a no-op.

        Args:
            provider: Anything implementing :class:`L1MemoryUsageProvider`
                — duck-typed via ``runtime_checkable``.
        """
        if not isinstance(provider, L1MemoryUsageProvider):
            raise TypeError(
                f"register_external_memory_provider: {provider!r} does "
                "not implement L1MemoryUsageProvider "
                "(get_used_capacity_bytes())"
            )
        with self._external_lock:
            if provider not in self._external_providers:
                self._external_providers.append(provider)

    def unregister_external_memory_provider(
        self, provider: L1MemoryUsageProvider
    ) -> None:
        """Remove a provider previously registered via
        :meth:`register_external_memory_provider`.  Idempotent: no-op
        when the provider was never registered.
        """
        with self._external_lock:
            try:
                self._external_providers.remove(provider)
            except ValueError:
                pass

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """
        Return an L1MemoryDesc describing the underlying memory buffer.

        Returns:
            L1MemoryDesc: Pointer, size, and alignment of the L1 buffer.

        Raises:
            NotImplementedError: If the allocator type does not support this operation.
        """
        if isinstance(self._allocator, MixedMemoryAllocator):
            buffer = self._allocator.buffer
        elif isinstance(self._allocator, LazyMemoryAllocator):
            # TODO(ApostaC): need to test if the RDMA registration works
            # before the lazy expansion is finished
            buffer = self._allocator.get_underlying_buffer()
        else:
            raise NotImplementedError(
                "get_l1_memory_desc is not implemented for this allocator type."
            )
        return L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=self._size_in_bytes,
            align_bytes=self._align_bytes,
        )

    def close(self) -> None:
        """
        Close the memory manager and release all resources.
        """
        self._allocator.close()

    # Debugging APIs
    def memcheck(self):
        return self._allocator.memcheck()
