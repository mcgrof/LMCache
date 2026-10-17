# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union, cast
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lifecycle import RetainedCall, _run_under_interrupt_safe_rlock
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_allocators.paged_cpu_gpu_memory_allocator import (
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.memory_management import (
    AllocationReceipt,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    RefCountDownReceipt,
    RefCountUpReceipt,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.batched_message_sender import BatchedMessageSender
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.system_detection import NUMADetector, SystemMemoryDetector

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


_MISSING_CACHE_ENTRY = object()


class _LocalCPUCacheAdmissionReceipt:
    """Retain one cache acquisition until its map publication completes."""

    def __init__(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        self.key = key
        self.memory_obj = memory_obj
        self.ref_count_up_receipt: RefCountUpReceipt = (
            memory_obj.make_ref_count_up_receipt()
        )

    def run(self, backend: "LocalCPUBackend") -> None:
        """Acquire the cache reference once and publish its exact object."""
        _run_under_interrupt_safe_rlock(
            backend.cpu_lock,
            lambda: self._run_locked(backend),
        )

    def _run_locked(self, backend: "LocalCPUBackend") -> None:
        cached = backend.hot_cache.get(self.key, _MISSING_CACHE_ENTRY)
        if cached is not _MISSING_CACHE_ENTRY and cached is not self.memory_obj:
            raise RuntimeError("LocalCPU cache key changed during admission recovery")
        self.memory_obj.ref_count_up_once(self.ref_count_up_receipt)
        backend.hot_cache[self.key] = self.memory_obj

    def complete(self, backend: "LocalCPUBackend") -> bool:
        """Prove both the exact acquisition and map publication."""
        return _run_under_interrupt_safe_rlock(
            backend.cpu_lock,
            lambda: self._complete_locked(backend),
        )

    def _complete_locked(self, backend: "LocalCPUBackend") -> bool:
        cached = backend.hot_cache.get(self.key, _MISSING_CACHE_ENTRY)
        if cached is not _MISSING_CACHE_ENTRY and cached is not self.memory_obj:
            raise RuntimeError("LocalCPU cache key changed during admission recovery")
        return cached is self.memory_obj and (
            self.memory_obj.ref_count_up_receipt_complete(self.ref_count_up_receipt)
        )

    def applied(self) -> bool:
        """Return whether the cache acquisition owns a live reference."""
        return self.memory_obj.ref_count_up_receipt_applied(self.ref_count_up_receipt)


class _LocalCPUCacheReleaseReceipt:
    """Retain one cache reference across removal interruptions."""

    def __init__(
        self, key: CacheEngineKey, memory_obj: MemoryObj, *, force: bool = False
    ) -> None:
        self.force = force
        self.key = key
        self.memory_obj = memory_obj
        self.ref_count_down_receipt: RefCountDownReceipt = (
            memory_obj.make_ref_count_down_receipt()
        )

    def run(self, backend: "LocalCPUBackend") -> None:
        """Remove the entry and consume its exact cache reference once."""
        _run_under_interrupt_safe_rlock(
            backend.cpu_lock,
            lambda: self._run_locked(backend),
        )

    def _run_locked(self, backend: "LocalCPUBackend") -> None:
        cached = backend.hot_cache.get(self.key, _MISSING_CACHE_ENTRY)
        if cached is self.memory_obj:
            if self.force:
                while self.memory_obj.is_pinned:
                    self.memory_obj.unpin()
            backend.hot_cache.pop(self.key)
        elif cached is not _MISSING_CACHE_ENTRY:
            raise RuntimeError("LocalCPU cache key changed during terminal cleanup")
        self.memory_obj.ref_count_down_once(self.ref_count_down_receipt)

    def complete(self, backend: "LocalCPUBackend") -> bool:
        """Prove both map removal and the exact reference decrement."""
        return _run_under_interrupt_safe_rlock(
            backend.cpu_lock,
            lambda: self._complete_locked(backend),
        )

    def _complete_locked(self, backend: "LocalCPUBackend") -> bool:
        cached = backend.hot_cache.get(self.key, _MISSING_CACHE_ENTRY)
        if cached is self.memory_obj:
            return False
        if cached is not _MISSING_CACHE_ENTRY:
            raise RuntimeError("LocalCPU cache key changed during terminal cleanup")
        return self.memory_obj.ref_count_down_receipt_complete(
            self.ref_count_down_receipt
        )


class _LocalCPUBackendCloseStage:
    """One exact LocalCPU terminal stage with audited retry semantics."""

    def __init__(
        self,
        call: Callable[[], None],
        *,
        completion_probe: Callable[[], bool],
        retry_safe_failure: Callable[[BaseException], bool],
        thread_name: str,
    ) -> None:
        self._completion_probe = completion_probe
        self._completed_by_probe = False
        self._call = RetainedCall(
            call,
            retry_safe_failure=retry_safe_failure,
            thread_name=thread_name,
        )

    @property
    def complete(self) -> bool:
        """Return whether the call or its exact state probe proves completion."""
        return self._completed_by_probe or self._call.complete

    def run(self) -> None:
        """Resume this stage without hiding a process-control interruption."""
        if self.complete:
            return
        try:
            self._call.run()
        except BaseException:
            # Preserve the interrupt for Manager's exhaustive cleanup/error
            # publication, but remember a callee-side terminal state that
            # proves the resource method returned before the interrupt.
            if self._completion_probe():
                self._completed_by_probe = True
            raise


class _LocalCPUBackendCloseOwner:
    """Retain exact progress across LocalCPUBackend terminal stages.

    The StorageManager wraps a backend's complete ``close()`` in one retained
    call.  LocalCPU cleanup itself has three independently fallible stages,
    though: stop the optional controller sender, evict cache references, and
    release the pinned allocator.  A process-control interruption between
    those calls must resume at the next stage without repeating an earlier
    non-idempotent close.

    Each stage therefore owns its own sticky retained receipt.  The outer
    backend call may be retried safely: completed stages return immediately,
    while a real stage failure re-raises from that exact receipt rather than
    invoking the stage again.
    """

    def __init__(self, backend: "LocalCPUBackend") -> None:
        self._sender_call = _LocalCPUBackendCloseStage(
            backend._close_batched_message_sender,
            completion_probe=backend._batched_message_sender_close_complete,
            retry_safe_failure=backend._batched_message_sender_close_retry_safe,
            thread_name="lmcache-local-cpu-close-sender",
        )
        self._cache_call = _LocalCPUBackendCloseStage(
            backend._clear_for_close,
            completion_probe=backend._cache_clear_complete,
            retry_safe_failure=lambda _error: True,
            thread_name="lmcache-local-cpu-close-cache",
        )
        self._allocator_call = _LocalCPUBackendCloseStage(
            backend._close_memory_allocator,
            completion_probe=backend._memory_allocator_close_complete,
            retry_safe_failure=backend._memory_allocator_close_retry_safe,
            thread_name="lmcache-local-cpu-close-allocator",
        )

    @property
    def complete(self) -> bool:
        """Return whether every exact terminal stage completed."""
        return (
            self._sender_call.complete
            and self._cache_call.complete
            and self._allocator_call.complete
        )

    def run(self) -> None:
        """Resume the first incomplete stage in dependency order."""
        self._sender_call.run()
        self._cache_call.run()
        self._allocator_call.run()


class LocalCPUBackend(AllocatorBackendInterface):
    """
    Even if local_cpu is False (the hot_cache is not used), contains(),
    insert_key(), remove(), get_blocking(), get_keys(), and clear()
    are still callable by the storage manager.
    """

    _close_owner_lock: Any
    _close_owner: Optional[_LocalCPUBackendCloseOwner]
    _cache_admission_receipts: list[_LocalCPUCacheAdmissionReceipt]
    _cache_release_receipts: list[_LocalCPUCacheReleaseReceipt]

    @staticmethod
    def _prepare_partial_state(backend: "LocalCPUBackend") -> None:
        """Make a retained raw backend safe for terminal cleanup."""
        backend.batched_msg_sender = None
        backend.memory_allocator = cast(MemoryAllocatorInterface, None)
        backend.hot_cache = {}
        backend.use_hot = False
        backend.cpu_lock = threading.RLock()
        backend.keys_in_request = []
        backend.config = None
        backend.metadata = None
        backend.lmcache_worker = None
        backend._close_owner_lock = threading.RLock()
        backend._close_owner = None
        backend._cache_admission_receipts = []
        backend._cache_release_receipts = []

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheMetadata] = None,
        dst_device: str = torch_device_type,
        lmcache_worker: Optional["LMCacheWorker"] = None,
        memory_allocator: Optional[MemoryAllocatorInterface] = None,
    ):
        LocalCPUBackend._prepare_partial_state(self)
        if torch_dev.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.hot_cache = self.cache_policy.init_mutable_mapping()

        self.use_hot = config.local_cpu
        # NOTE: we keep the memory allocator argument for temporary
        # test compatibility
        # TODO: fix the tests to get rid the memory allocator
        assert metadata is not None or memory_allocator is not None
        self.memory_allocator = (
            self.initialize_allocator(config, metadata)  # type: ignore
            if memory_allocator is None
            else memory_allocator
        )
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        # Eviction already holds this lock while calling ``remove`` through
        # the generic batched API. A reentrant lock lets every public remove
        # take the same atomic membership/pin-state lock without deadlocking
        # that internal eviction path.
        self.cpu_lock = threading.RLock()

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        self.layerwise = config.use_layerwise
        self.enable_blending = config.enable_blending

        # Store config and metadata for chunk budget calculation
        self.config = config
        self.metadata = metadata

        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []

        # Batched message sender for controller communication
        self.batched_msg_sender: Optional[BatchedMessageSender] = None

        # Initialize batched message sender
        if lmcache_worker and metadata is not None:
            self.batched_msg_sender = BatchedMessageSender(
                metadata=metadata,
                config=config,
                location=str(self),  # Backend location
                lmcache_worker=lmcache_worker,
            )
        else:
            logger.warning("Controller message sender is not initialized")

        self._setup_metrics()

    def _setup_metrics(self) -> None:
        if self.metadata is None:
            return

        prometheus_logger = PrometheusLogger.GetOrCreate(
            self.metadata,
            config=self.config,
        )
        prometheus_logger.local_cpu_hot_cache_count.set_function(
            lambda: len(self.hot_cache)
        )
        prometheus_logger.local_cpu_keys_in_request_count.set_function(
            lambda: len(self.keys_in_request)
        )

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            if pin:
                self.hot_cache[key].pin()
                # vllm lookup sets pin to True
                self.keys_in_request.append(key)
            return True

    def supports_lookup_pin_progress(self) -> bool:
        return True

    def pin_for_lookup(self, key: CacheEngineKey) -> bool:
        """Pin one local object without a post-publication exception."""
        with self.cpu_lock:
            memory_obj = self.hot_cache.get(key)
            if memory_obj is None:
                return False
            memory_obj.pin()
            try:
                self.keys_in_request.append(key)
            except BaseException:
                # Tensor/bytes pin transitions are exception-atomic, and their
                # unpin bookkeeping is best-effort after the count changes.
                memory_obj.unpin()
                raise
            return True

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.cpu_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.hot_cache)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        contains() and exists_in_put_tasks() should be checked together
        """
        return False

    def _drain_cache_admission_receipts_locked(self) -> None:
        """Finish every prepublished cache-reference acquisition."""
        for receipt in tuple(self._cache_admission_receipts):
            if not receipt.applied():
                self._cache_admission_receipts.remove(receipt)
            else:
                receipt.run(self)
                if receipt.complete(self):
                    self._cache_admission_receipts.remove(receipt)

    def _drain_cache_release_receipts_locked(self) -> None:
        """Finish every prepublished cache-reference release."""
        for receipt in tuple(self._cache_release_receipts):
            receipt.run(self)
            if receipt.complete(self):
                self._cache_release_receipts.remove(receipt)

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[Future]:
        """
        Synchronously put the MemoryObj into the local cpu backend.

        :param on_complete_callback: Optional callback invoked after the
            synchronous put completes. Callback exceptions are caught and logged.
        """

        def put_locked() -> bool:
            self._drain_cache_release_receipts_locked()
            self._drain_cache_admission_receipts_locked()
            if key in self.hot_cache:
                return False

            admission = _LocalCPUCacheAdmissionReceipt(key, memory_obj)
            # Retain the exact acquisition before it can increment the
            # object. Recovery can then complete both sides of the former
            # ref_count_up-to-hot_cache publication window.
            self._cache_admission_receipts.append(admission)
            admission.run(self)
            if admission.complete(self):
                self._cache_admission_receipts.remove(admission)

            self.cache_policy.update_on_put(key)

            # Push kv admit msg with batching
            if self.batched_msg_sender is not None:
                self.batched_msg_sender.add_kv_op(
                    op_type=OpType.ADMIT,
                    key=key.chunk_hash,
                )
            return True

        stored = _run_under_interrupt_safe_rlock(self.cpu_lock, put_locked)

        # Call callback after put completes (outside lock)
        if stored and on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning("on_complete_callback failed for key %s: %s", key, e)

        return None

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Synchronously put the MemoryObjs into the local cpu backend.

        :param on_complete_callback: Optional callback invoked once per key
            after that key's put completes (not once per batch).
        """
        if not self.use_hot:
            return

        # TODO(Jiayi): optimize this with batching
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(
                key, memory_obj, on_complete_callback=on_complete_callback
            )

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return None
            memory_obj = self.hot_cache[key]
            # ref count up for caller to avoid situation where the memory_obj
            # is evicted from the local cpu backend before the caller calls
            # ref count up themselves
            memory_obj.ref_count_up()
            return memory_obj

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        mem_objs = []
        with self.cpu_lock:
            for key in keys:
                mem_obj = self.hot_cache[key]
                mem_obj.ref_count_up()
                mem_objs.append(mem_obj)
        return mem_objs

    def batched_get_non_blocking_returns_pinned(self) -> bool:
        # Local objects remain owned by the cache. A lookup may have pinned
        # them independently, but retrieval itself acquires references only.
        return False

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        # NOTE(Jiayi): Only prefix chunks are counted.
        num_hit_chunks = 0
        with self.cpu_lock:
            for key in keys:
                if key not in self.hot_cache:
                    return num_hit_chunks
                if pin:
                    self.hot_cache[key].pin()
                    # vllm lookup sets pin to True
                    self.keys_in_request.append(key)
                num_hit_chunks += 1
        return num_hit_chunks

    def pin(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.unpin()
            return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        def remove_locked() -> bool:
            self._drain_cache_release_receipts_locked()
            self._drain_cache_admission_receipts_locked()
            memory_obj = self.hot_cache.get(key)
            if memory_obj is None or (not force and not memory_obj.can_evict):
                return False

            release = _LocalCPUCacheReleaseReceipt(key, memory_obj, force=force)
            # Publish the exact release before removing the map entry.
            self._cache_release_receipts.append(release)
            release.run(self)
            if release.complete(self):
                self._cache_release_receipts.remove(release)

            if force:
                self.cache_policy.update_on_force_evict(key)
            return True

        removed = _run_under_interrupt_safe_rlock(self.cpu_lock, remove_locked)

        if not removed:
            return False

        if self.batched_msg_sender is not None:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.EVICT,
                key=key.chunk_hash,
            )
        # NOTE (Jiayi): This `return True` might not accurately reflect
        # whether the key is removed from the actual memory because
        # other backends might still (temporarily) hold the memory object.
        return True

    def _calculate_effective_cpu_size(
        self,
        configured_cpu_size: float,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheMetadata] = None,
    ) -> float:
        """
        Calculate the effective CPU memory size based on system available memory
        and reserve memory configuration.

        Args:
            configured_cpu_size: The configured CPU memory size in GB
            config: The LMCache engine configuration
            metadata: Optional metadata for first rank handling

        Returns:
            The effective CPU memory size in GB
        """

        save_only_first_rank = (
            metadata is not None
            and config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if not save_only_first_rank:
            # Do not adjust cpu_size if save_only_first_rank is False for now
            return configured_cpu_size

        # Get the system available memory and calculate effective cpu_size
        system_available_memory_gb = SystemMemoryDetector.get_available_memory_gb()
        # Get reserve memory size from config
        reserve_cpu_size = config.reserve_local_cpu_size

        # TODO(baoloongmao): For disable save_only_first_rank case,
        #  we need to avoid multi-rank race condition in future.
        #  But for enable save_only_first_rank case,
        #  we can handle reserve memory simply since non-first ranks
        #  do not allocate memory.
        # Effective memory: min(configured_size, available_memory - reserve_size)
        if system_available_memory_gb > 0:
            max_usable_memory = max(0, system_available_memory_gb - reserve_cpu_size)
            effective_cpu_size = min(configured_cpu_size, max_usable_memory)
            logger.info(
                "Adjusted CPU memory size from %.2f GB "
                "to %.2f GB "
                "(system available: %.2f GB, "
                "reserve: %.2f GB)",
                configured_cpu_size,
                effective_cpu_size,
                system_available_memory_gb,
                reserve_cpu_size,
            )
            assert effective_cpu_size > 0
            return effective_cpu_size
        else:
            logger.warning(
                "Could not determine system available memory, using configured cpu_size"
            )
            return configured_cpu_size

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheMetadata] = None,
    ) -> MemoryAllocatorInterface:
        cpu_size = config.max_local_cpu_size
        use_hugepages = config.local_cpu_use_hugepages

        if metadata is not None:
            # save_only_first_rank only works when use mla
            save_only_first_rank = (
                config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
                and metadata.use_mla
            )

            if save_only_first_rank and metadata.is_first_rank():
                # Only the first rank will save the cache,
                # so we need to set it larger than other ranks
                cpu_size = config.get_extra_config_value(
                    "first_rank_max_local_cpu_size", cpu_size
                )

        # Detect the numa mapping
        numa_mapping = NUMADetector.get_numa_mapping(config)
        logger.info("NUMA mapping %s", numa_mapping)

        # Calculate effective CPU memory size
        cpu_size = self._calculate_effective_cpu_size(cpu_size, config, metadata)
        cpu_size_bytes = int(cpu_size * 1024**3)

        allocator_align_bytes = self._resolve_local_cpu_allocator_alignment(config)
        if allocator_align_bytes is not None:
            logger.info(
                "LocalCPUBackend: using pinned allocation alignment=%d bytes",
                allocator_align_bytes,
            )

        if config.enable_p2p:
            if use_hugepages:
                raise ValueError("Hugepages are not supported with P2P mode")

            # TODO(baoloongmao): Add lazy memory allocator support for P2P mode
            # For now, keep the original P2P implementation
            assert metadata is not None
            shapes = metadata.get_shapes()
            dtypes = metadata.get_dtypes()

            paged_mem_allocator = PagedCpuGpuMemoryAllocator()
            chunk_size_bytes = get_size_bytes(shapes, dtypes)
            origin_cpu_size_bytes = cpu_size_bytes
            align_cpu_size_bytes = (
                origin_cpu_size_bytes // chunk_size_bytes * chunk_size_bytes
            )
            logger.info(
                "Auto align cpu size bytes, origin: %s, aligned: %s, chunk size: %s",
                origin_cpu_size_bytes,
                align_cpu_size_bytes,
                chunk_size_bytes,
            )
            paged_mem_allocator.init_cpu_memory_allocator(
                align_cpu_size_bytes,
                shapes=shapes,
                dtypes=dtypes,
                fmt=MemoryFormat.KV_2LTD,  # TODO: remove this hardcode
                numa_mapping=numa_mapping,
            )
            return paged_mem_allocator
        else:
            extra_config = config.extra_config
            nixl_cpu_enabled = (
                extra_config is not None
                and extra_config.get("enable_nixl_storage")
                and config.nixl_buffer_device == "cpu"
            )

            if nixl_cpu_enabled:
                if metadata is None:
                    raise ValueError("metadata required for NIXL CPU mode")
                chunk_bytes = get_size_bytes(
                    metadata.get_shapes(), metadata.get_dtypes()
                )
                aligned_size = (cpu_size_bytes // chunk_bytes) * chunk_bytes
                if aligned_size == 0:
                    raise ValueError(
                        f"max_local_cpu_size ({cpu_size_bytes} bytes) is smaller than "
                        f"one chunk ({chunk_bytes} bytes); cannot initialize NIXL CPU "
                        f"shared pool"
                    )
                if aligned_size != cpu_size_bytes:
                    logger.warning(
                        "max_local_cpu_size not a multiple of chunk size; "
                        "rounding down from %d to %d bytes for NIXL shared pool",
                        cpu_size_bytes,
                        aligned_size,
                    )
                return self._construct_mixed_allocator(
                    aligned_size,
                    use_paging=True,
                    use_hugepages=use_hugepages,
                    numa_mapping=numa_mapping,
                    shapes=metadata.get_shapes(),
                    dtypes=metadata.get_dtypes(),
                    fmt=MemoryFormat.KV_2LTD,
                )

            # Check if io_uring is enabled for fixed buffer support
            io_engine = str(
                config.get_extra_config_value("rust_raw_block.io_engine", "") or ""
            ).lower()
            use_uring = (
                io_engine == "io_uring"
                or bool(
                    config.get_extra_config_value("rust_raw_block.use_iouring", False)
                )
                or bool(
                    config.get_extra_config_value("rust_raw_block.use_uring", False)
                )
            )

            # Check if lazy memory allocator should be enabled
            use_lazy = (
                config.enable_lazy_memory_allocator
                and cpu_size > config.lazy_memory_safe_size
            )
            if use_lazy:
                logger.warning(
                    "LazyMixedMemoryAllocator is temporarily unavailable; "
                    "falling back to MixedMemoryAllocator with full allocation. "
                    "Disable enable_lazy_memory_allocator or reduce "
                    "max_local_cpu_size to avoid large pinned allocations."
                )
            elif config.enable_lazy_memory_allocator:
                logger.info(
                    "LazyMixedMemoryAllocator is disabled because "
                    "cpu_size (%.2f GB) does not exceed "
                    "lazy_memory_safe_size "
                    "(%.2f GB). "
                    "Using MixedMemoryAllocator instead.",
                    cpu_size,
                    config.lazy_memory_safe_size,
                )

            # For io_uring, use paged memory allocator so that fixed buffer support
            # can be enabled
            if use_uring and metadata is not None:
                shapes = metadata.get_shapes()
                dtypes = metadata.get_dtypes()
                # Determine memory format based on layerwise and blending settings
                if config.use_layerwise:
                    if config.enable_blending:
                        fmt = MemoryFormat.KV_2TD
                    else:
                        fmt = MemoryFormat.KV_T2D
                else:
                    fmt = MemoryFormat.KV_2LTD

                # Calculate chunk size for alignment
                chunk_size_bytes = get_size_bytes(shapes, dtypes)
                origin_cpu_size_bytes = cpu_size_bytes
                # Align cpu_size_bytes to be a multiple of chunk_size_bytes
                align_cpu_size_bytes = (
                    origin_cpu_size_bytes // chunk_size_bytes * chunk_size_bytes
                )
                logger.info(
                    "LocalCPUBackend: using MixedMemoryAllocator with use_paging=True "
                    "for io_uring fixed buffer support. "
                    "Auto align cpu size bytes, origin: %s, "
                    "aligned: %s, chunk size: %s",
                    origin_cpu_size_bytes,
                    align_cpu_size_bytes,
                    chunk_size_bytes,
                )

                kwargs = {
                    "numa_mapping": numa_mapping,
                    "shapes": shapes,
                    "dtypes": dtypes,
                    "fmt": fmt,
                    **(
                        {"align_bytes": allocator_align_bytes}
                        if allocator_align_bytes is not None
                        else {}
                    ),
                }
                # A dma-buf backed buffer lets the raw_block backend register
                # the paged buffers with its device once (map-once, MDTS-sized
                # commands); hugetlb folios keep that path at full size on a
                # host without IOMMU translation.
                if config.local_cpu_dmabuf:
                    kwargs["dmabuf"] = config.local_cpu_dmabuf
                return self._construct_mixed_allocator(
                    align_cpu_size_bytes,
                    use_paging=True,
                    use_hugepages=bool(config.local_cpu_dmabuf) and use_hugepages,
                    allocation_boundary_bytes=None,
                    **kwargs,
                )

            # Default: use non-paged allocator
            if allocator_align_bytes is not None:
                return self._construct_mixed_allocator(
                    cpu_size_bytes,
                    numa_mapping=numa_mapping,
                    align_bytes=allocator_align_bytes,
                    use_hugepages=use_hugepages,
                )
            return self._construct_mixed_allocator(
                cpu_size_bytes,
                numa_mapping=numa_mapping,
                config=config,
                use_hugepages=use_hugepages,
            )

    def _construct_mixed_allocator(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> MixedMemoryAllocator:
        """Publish the target allocator before its arena construction."""
        allocator = MixedMemoryAllocator.__new__(MixedMemoryAllocator)
        prepare_partial_state = getattr(
            MixedMemoryAllocator,
            "_prepare_partial_state",
            None,
        )
        if prepare_partial_state is not None:
            prepare_partial_state(allocator)
        self.memory_allocator = allocator
        MixedMemoryAllocator.__init__(allocator, *args, **kwargs)
        return allocator

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    def supports_allocation_receipts(self) -> bool:
        """Return whether this backend's allocator has exact reservations."""
        return bool(
            isinstance(self.memory_allocator, MixedMemoryAllocator)
            and self.memory_allocator.supports_allocation_receipts()
        )

    def _resolve_local_cpu_allocator_alignment(
        self, config: LMCacheEngineConfig
    ) -> Optional[int]:
        """
        Determine pinned-memory alignment for LocalCPUBackend allocator.

        Precedence:
        1) explicit override: extra_config["local_cpu.pinned_align_bytes"]
        2) rust raw block auto mode:
           - rust_raw_block.device_path is set
           - rust_raw_block.use_odirect is true or rust_raw_block.use_uring is true
           - rust_raw_block.align_local_cpu_allocator is true (default)
           -> use rust_raw_block.block_align
        3) None (use allocator default)
        """
        extra = config.extra_config or {}

        explicit_align = extra.get("local_cpu.pinned_align_bytes")
        if explicit_align is not None:
            align = int(explicit_align)
            if not self._is_power_of_two(align):
                raise ValueError(
                    "extra_config['local_cpu.pinned_align_bytes'] must be "
                    "a positive power of two"
                )
            return align

        rust_device_path = extra.get("rust_raw_block.device_path")
        rust_use_odirect = bool(extra.get("rust_raw_block.use_odirect", False))
        rust_io_engine = str(extra.get("rust_raw_block.io_engine", "") or "").lower()
        rust_use_uring = (
            rust_io_engine == "io_uring"
            or bool(extra.get("rust_raw_block.use_iouring", False))
            or bool(extra.get("rust_raw_block.use_uring", False))
        )
        rust_auto_align = bool(
            extra.get("rust_raw_block.align_local_cpu_allocator", True)
        )

        if not rust_device_path:
            return None

        # Alignment is needed if either O_DIRECT is set or io_uring is enabled
        if not rust_use_odirect and not rust_use_uring:
            return None

        # For non io_uring_case, respect the auto_align flag
        if not rust_use_uring and not rust_auto_align:
            return None

        rust_block_align = int(extra.get("rust_raw_block.block_align", 4096))
        if not self._is_power_of_two(rust_block_align):
            raise ValueError(
                "extra_config['rust_raw_block.block_align'] must be a power "
                "of two when O_DIRECT or io_uring alignment is enabled"
            )
        return rust_block_align

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
        busy_loop: bool = True,
        allocation_receipt: Optional[AllocationReceipt] = None,
    ) -> Optional[MemoryObj]:
        """
        Allocate a memory object of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False

        busy_loop should only be used for retrieve
        the reasoning is that:

        1. synchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - one retrieve at a time (okay to busy loop because stores will clear)

        2. asynchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - many retrieves happen concurrently
        (we use the async serializer to handle this)
        """
        logger.debug(
            "Allocating memory in local cpu backend with busy loop: %s", busy_loop
        )
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        if allocation_receipt is not None and not self.supports_allocation_receipts():
            raise NotImplementedError(
                "configured LocalCPU allocator does not support exact receipts"
            )
        if allocation_receipt is None:
            memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
        else:
            memory_obj = self.memory_allocator.allocate(
                shapes,
                dtypes,
                fmt,
                allocation_receipt=allocation_receipt,
            )
        if memory_obj is not None or not eviction:
            return memory_obj

        evict_keys_count = 0
        num_attempts = 0
        while True:
            # whether or not this request needs to wait or other requests
            wait_other_requests = True
            if self.use_hot:
                # TODO(Jiayi): optimize `num_candidates` with estimation.
                # Accurate estimation is hard due to fragmentation
                num_candidates = 1
                evict_keys = None
                with self.cpu_lock:
                    evict_keys = self.cache_policy.get_evict_candidates(
                        self.hot_cache, num_candidates=num_candidates
                    )
                    if evict_keys:
                        # we can continue trying to evict from the hot_cache
                        # and don't need to wait for other requests yet
                        wait_other_requests = False
                        logger.debug(
                            "Evicting %d chunks from cpu memory", len(evict_keys)
                        )
                        # remove
                        self.batched_remove(evict_keys, force=False)
                        evict_keys_count += len(evict_keys)
                    else:
                        self.stats_monitor.update_local_cpu_evict_failed_count(
                            num_candidates
                        )

            if wait_other_requests:
                if not busy_loop:
                    logger.debug(
                        "Not busy looping because we are not immediately able to evict"
                    )
                    break

                # TODO: make time_to_wait a config
                time_to_wait = 0.1
                logger.warning(
                    "No eviction candidates found in local cpu backend. "
                    "Local cpu memory is under pressure. "
                    "Waiting for %s seconds before retrying.",
                    time_to_wait,
                )
                # self.memory_allocator.memcheck()
                # do not hold the lock during sleep
                time.sleep(time_to_wait)

            if allocation_receipt is None:
                memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
            else:
                memory_obj = self.memory_allocator.allocate(
                    shapes,
                    dtypes,
                    fmt,
                    allocation_receipt=allocation_receipt,
                )
            if memory_obj is not None:
                break

            num_attempts += 1
            logger.debug(
                "Unable to allocate memory object after %d"
                " attempts of local cpu backend allocate()",
                num_attempts,
            )

        self.stats_monitor.update_local_cpu_evict_metrics(evict_keys_count)
        return memory_obj

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
        busy_loop: bool = True,
        allocation_receipts: Optional[Sequence[AllocationReceipt]] = None,
    ) -> Optional[List[MemoryObj]]:
        """
        Batched allocate `batch_size` memory objects of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False

        busy_loop should only be used for retrieve
        the reasoning is that:

        1. synchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - one retrieve at a time (okay to busy loop because stores will clear)

        2. asynchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - many retrieves happen concurrently
        (we use the async serializer to handle this)
        """
        logger.debug(
            "Batched allocating memory in local cpu backend with busy loop: %s",
            busy_loop,
        )
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        if allocation_receipts is not None and not self.supports_allocation_receipts():
            raise NotImplementedError(
                "configured LocalCPU allocator does not support exact receipts"
            )
        if allocation_receipts is None:
            memory_objs = self.memory_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
        else:
            memory_objs = self.memory_allocator.batched_allocate(
                shapes,
                dtypes,
                batch_size,
                fmt,
                allocation_receipts=allocation_receipts,
            )

        if memory_objs is not None or not eviction:
            return memory_objs

        assert isinstance(self.memory_allocator, MixedMemoryAllocator)

        evict_keys_count = 0
        num_attempts = 0
        while True:
            wait_other_requests = True
            if self.use_hot:
                # TODO(Jiayi): optimize `num_candidates` with estimation.
                # Accurate estimation is hard due to fragmentation
                num_candidates = min(64, max(1, len(self.hot_cache)))
                evict_keys = None
                with self.cpu_lock:
                    evict_keys = self.cache_policy.get_evict_candidates(
                        self.hot_cache, num_candidates=num_candidates
                    )

                    # HACK: We assume batch_size=num_layers here.
                    # FIXME: We also assume if the one layer's ref_count > 1 or pinned,
                    # then the other layers are also ref_count > 1 or
                    # pinned in the cpu memory. This might not be true.
                    if evict_keys:
                        for evict_key in evict_keys:
                            evict_key_all_layer = evict_key.split_layers(batch_size)

                            # TODO(Jiayi): batched allocate is not supported through
                            # `batched_remove`. Therefore, features like usage tracking
                            # is not supported.
                            old_mem_objs: list[MemoryObj] = []
                            for key in evict_key_all_layer:
                                old_mem_obj = self.hot_cache.get(key)
                                if old_mem_obj is None or not old_mem_obj.can_evict:
                                    old_mem_objs = []
                                    break
                                old_mem_objs.append(old_mem_obj)

                            if not old_mem_objs:
                                continue

                            wait_other_requests = False
                            evict_keys_count += len(old_mem_objs)
                            for key in evict_key_all_layer:
                                self.cache_policy.update_on_force_evict(key)
                                self.hot_cache.pop(key, None)

                            self.memory_allocator.batched_free(old_mem_objs)

                            logger.debug(
                                "Evicting %d chunks from cpu memory", len(old_mem_objs)
                            )
                            break
                    if wait_other_requests:
                        self.stats_monitor.update_local_cpu_evict_failed_count(
                            num_candidates
                        )

            if wait_other_requests:
                if not busy_loop:
                    logger.debug(
                        "Not busy looping because we are not immediately able to evict"
                    )
                    break

                # TODO: make time_to_wait a config
                time_to_wait = 0.1
                logger.warning(
                    "No eviction candidates found in local cpu backend. "
                    "Local cpu memory is under pressure. "
                    "Waiting for %s seconds before retrying.",
                    time_to_wait,
                )
                # self.memory_allocator.memcheck()
                # do not hold the lock during sleep
                time.sleep(time_to_wait)

            if allocation_receipts is None:
                memory_objs = self.memory_allocator.batched_allocate(
                    shapes, dtypes, batch_size, fmt
                )
            else:
                memory_objs = self.memory_allocator.batched_allocate(
                    shapes,
                    dtypes,
                    batch_size,
                    fmt,
                    allocation_receipts=allocation_receipts,
                )
            if memory_objs is not None:
                break

            num_attempts += 1
            logger.debug(
                "Unable to allocate memory object after %d"
                " attempts of local cpu backend batched_allocate()",
                num_attempts,
            )
        self.stats_monitor.update_local_cpu_evict_metrics(evict_keys_count)
        return memory_objs

    def get_full_chunk_size_bytes(self) -> int:
        logger.info("Calculating the size of a single LMCache chunk")
        assert self.metadata is not None, (
            "metadata required for chunk budget calculation"
        )

        chunk_tokens = self.config.chunk_size
        # already accounted for parallelism
        kv_shape = (
            self.metadata.kv_shape
        )  # [num_layers, kv_size, chunk_size, num_heads, head_size]
        num_layers = kv_shape[0]
        kv_size = kv_shape[1]  # 1 for MLA, 2 for regular
        # per gpu
        num_heads = kv_shape[3]
        head_size = kv_shape[4]
        hidden_dim = num_heads * head_size
        dtype_size = self.metadata.kv_dtype.itemsize

        if self.layerwise:
            # layerwise: [chunk_tokens, kv_size, hidden_dim]
            chunk_bytes = chunk_tokens * kv_size * hidden_dim * dtype_size
        else:
            # full: [kv_size, num_layers, chunk_tokens, hidden_dim]
            chunk_bytes = kv_size * num_layers * chunk_tokens * hidden_dim * dtype_size
        logger.debug(
            "Stats received: num_layers=%s, kv_size=%s, "
            "chunk_tokens=%s, head_dim=%s, "
            "dtype_size=%s, "
            "hidden_dim=%s",
            num_layers,
            kv_size,
            chunk_tokens,
            head_size,
            dtype_size,
            hidden_dim,
        )
        logger.debug("Calculated bytes per chunk per rank: %s", chunk_bytes)
        return chunk_bytes

    def calculate_chunk_budget(self) -> int:
        """
        Calculate the maximum number of chunks that can be allocated concurrently
        without causing memory deadlocks in the async loading system.

        Returns:
            int: The estimated chunk budget for concurrent allocations
        """
        total_memory = int(self.config.max_local_cpu_size * 1024**3)
        chunk_bytes = self.get_full_chunk_size_bytes()
        # add alignment overhead
        # (MixedMemoryAllocator uses TensorMemoryAllocator with 4KB alignment)
        assert hasattr(self.memory_allocator, "align_bytes")
        alignment = self.memory_allocator.align_bytes
        aligned_chunk_bytes = ((chunk_bytes + alignment - 1) // alignment) * alignment

        # calculate budget with safety margin
        max_chunks = total_memory // aligned_chunk_bytes

        return max_chunks

    def get_keys(self) -> List[CacheEngineKey]:
        """
        array ordering of keys from LRU to MRU
        """
        with self.cpu_lock:
            return list(self.hot_cache.keys())

    def clear(self, force: bool = False) -> int:
        """
        counts the number of memory objects removed
        """
        if not self.use_hot:
            return 0
        clear_keys = []
        tokens_by_key: dict[CacheEngineKey, int] = {}
        with self.cpu_lock:
            for key in self.hot_cache:
                memory_obj = self.hot_cache[key]
                if not force and not memory_obj.can_evict:
                    continue
                clear_keys.append(key)
                tokens_by_key[key] = memory_obj.get_num_tokens()

        # A lookup may pin a scanned candidate before removal. ``remove``
        # rechecks can_evict under cpu_lock, so count only keys whose ownership
        # transition actually completed.
        num_cleared_tokens = 0
        for key in clear_keys:
            if self.remove(key, force=force):
                num_cleared_tokens += tokens_by_key[key]

        return num_cleared_tokens

    def get_allocator_backend(self):
        return self

    def get_memory_allocator(self):
        return self.memory_allocator

    def retain_backing_resources(self) -> None:
        """Keep the arena when a consumer cannot prove native I/O has stopped."""
        self._backing_resources_retained = True

    def _close_batched_message_sender(self) -> None:
        """Close the exact optional controller sender once."""
        batched_msg_sender = getattr(self, "batched_msg_sender", None)
        if batched_msg_sender is not None:
            batched_msg_sender.close()

    def _batched_message_sender_close_complete(self) -> bool:
        """Prove terminal state only for the in-tree sender implementation."""
        sender = getattr(self, "batched_msg_sender", None)
        if sender is None:
            return True
        if not isinstance(sender, BatchedMessageSender):
            return False
        thread = getattr(sender, "thread", None)
        return not bool(getattr(sender, "running", True)) and (
            thread is None or not thread.is_alive()
        )

    def _batched_message_sender_close_retry_safe(
        self,
        error: BaseException,
    ) -> bool:
        """Retry only the audited monotonic in-tree sender shutdown."""
        del error
        sender = getattr(self, "batched_msg_sender", None)
        return sender is None or isinstance(sender, BatchedMessageSender)

    def _clear_for_close(self) -> None:
        """Release terminal cache ownership through per-entry receipts."""
        while True:

            def prepare_locked() -> Optional[_LocalCPUCacheReleaseReceipt]:
                self._drain_cache_release_receipts_locked()
                self._drain_cache_admission_receipts_locked()
                if not self.hot_cache:
                    return None
                key = next(iter(self.hot_cache))
                pending = _LocalCPUCacheReleaseReceipt(
                    key,
                    self.hot_cache[key],
                )
                # Publish the exact object/release receipt before removing the
                # map entry. A retry can therefore finish either side of the
                # former pop-to-ref_count_down interruption window.
                self._cache_release_receipts.append(pending)
                return pending

            pending = _run_under_interrupt_safe_rlock(
                self.cpu_lock,
                prepare_locked,
            )
            if pending is None:
                return
            pending.run(self)

    def _cache_clear_complete(self) -> bool:
        """Prove no cache entry or receipted cache reference remains."""

        def complete_locked() -> bool:
            complete = not self._cache_admission_receipts and not self.hot_cache
            if complete:
                complete = all(
                    receipt.complete(self) for receipt in self._cache_release_receipts
                )
            if complete:
                # Every exact release is already authoritative. Dropping these
                # diagnostic owners is safe even if the caller is interrupted
                # immediately after this probe returns.
                self._cache_release_receipts.clear()
            return complete

        return _run_under_interrupt_safe_rlock(self.cpu_lock, complete_locked)

    def _close_memory_allocator(self) -> None:
        """Close the exact allocator retained by this backend."""
        memory_allocator = getattr(self, "memory_allocator", None)
        if memory_allocator is not None:
            memory_allocator.close()

    def _memory_allocator_close_complete(self) -> bool:
        """Prove release only for the target MixedMemoryAllocator receipt."""
        allocator = getattr(self, "memory_allocator", None)
        if allocator is None:
            return True
        if not isinstance(allocator, MixedMemoryAllocator):
            return False
        if bool(getattr(allocator, "_unregistered", False)):
            return True
        receipt = getattr(allocator, "_cpu_memory_allocation_receipt", None)
        if receipt is not None and receipt.released:
            # An instruction-level interruption can land after
            # ``receipt.release()`` returns but before MixedMemoryAllocator
            # publishes its diagnostic mirror bit.  The receipt is the exact
            # ownership authority, so make the mirror truthful while proving
            # that this stage is complete.
            allocator._unregistered = True
            return True
        return False

    def _memory_allocator_close_retry_safe(
        self,
        error: BaseException,
    ) -> bool:
        """Retry only the receipt-backed target allocator implementation."""
        del error
        allocator = getattr(self, "memory_allocator", None)
        return allocator is None or isinstance(allocator, MixedMemoryAllocator)

    def close(self) -> None:
        """Close each terminal child through its retained stage receipt."""
        if getattr(self, "_backing_resources_retained", False):
            raise RuntimeError(
                "Retaining CPU allocator: a storage consumer is not quiescent"
            )
        with self._close_owner_lock:
            owner = self._close_owner
            if owner is None:
                owner = _LocalCPUBackendCloseOwner(self)
                self._close_owner = owner
        owner.run()
        if not owner.complete:
            raise RuntimeError("LocalCPUBackend terminal cleanup is incomplete")

    def close_failure_is_retry_safe(self, error: BaseException) -> bool:
        """Allow only the staged outer dispatcher to be resumed.

        Individual sender/cache/allocator failures remain sticky in their
        inner retained calls.  Retrying this outer method can therefore only
        resume a stage that was never called or cross an interruption between
        two already-published stage receipts.
        """
        del error
        return True
