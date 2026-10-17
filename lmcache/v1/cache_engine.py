# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from concurrent.futures import Future
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.health_monitor.base import HealthMonitor

# Standard
import asyncio
import gc
import multiprocessing
import threading
import time
import weakref

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_telemetry import InitializeUsageContext, UsageContext
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    _lmcache_nvtx_annotate,
    compress_slot_mapping,
    convert_tokens_to_list,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.hidden_state_store import HiddenStateStore
from lmcache.v1.lifecycle import RetainedCall, RetryPolicy
from lmcache.v1.memory_allocators.cu_file_memory_allocator import CuFileMemoryAllocator
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import (
    AllocationReceipt,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    RefCountDownReceipt,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.platform import current_device_spec
from lmcache.v1.storage_backend.raw_block import (
    RawBlockPublicationReceipt,
    RawBlockReadContext,
)
from lmcache.v1.storage_backend.storage_manager import (
    PutOwnershipReceipt,
    StorageManager,
)
from lmcache.v1.system_detection import NUMADetector, NUMAMapping
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    ProcessTokensResult,
    SegmentTokenDatabase,
    TokenDatabase,
)
from lmcache.v1.token_span import ExplicitTokenSpans

logger = init_logger(__name__)

_LOOKUP_CANCELLED_GENERATION_LIMIT = 4096
_LOOKUP_CLEANUP_RETRY_INTERVAL_S = 0.2

# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]


class _StoreOperationLease:
    """Weakly observed proof that a store generator frame is still live."""


class _StoreCleanupOwner:
    """Retry state for one layerwise store outside its generator frame."""

    def __init__(
        self,
        gpu_connector: GPUConnectorInterface,
        storage_manager: StorageManager,
        operation_lease: Optional[_StoreOperationLease] = None,
    ):
        self._gpu_connector = gpu_connector
        self._storage_manager = storage_manager
        self._gpu_generator: Any = None
        self._allocation_receipts: list[AllocationReceipt] = []
        self._allocated: list[MemoryObj] = []
        self._put_receipts: list[tuple[PutOwnershipReceipt, object]] = []
        self._release_receipts: dict[int, Optional[RefCountDownReceipt]] = {}
        self._legacy_released: set[int] = set()
        self._operation_lease_ref = (
            weakref.ref(operation_lease) if operation_lease is not None else None
        )
        self._pending = False
        self._complete = False
        self._lock = threading.RLock()

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending

    @property
    def complete(self) -> bool:
        with self._lock:
            return self._complete

    @property
    def operation_alive(self) -> bool:
        """Return whether the store frame still owns its strong lease."""
        lease_ref = self._operation_lease_ref
        return lease_ref is None or lease_ref() is not None

    def add_allocated(self, memory_objs: list[MemoryObj]) -> None:
        with self._lock:
            if self._pending:
                raise RuntimeError("cannot allocate into pending store cleanup")
            self._refresh_allocated_from_receipts()
            for memory_obj in memory_objs:
                if not any(owned is memory_obj for owned in self._allocated):
                    self._allocated.append(memory_obj)

    def reserve_allocation_receipts(self, count: int) -> tuple[AllocationReceipt, ...]:
        """Publish exact allocation ownership before entering the backend."""
        with self._lock:
            if self._pending:
                raise RuntimeError("cannot allocate into pending store cleanup")
            receipts = tuple(AllocationReceipt() for _ in range(count))
            self._allocation_receipts.extend(receipts)
            return receipts

    def _refresh_allocated_from_receipts(self) -> None:
        """Adopt objects published inside allocators, including after unwind."""
        for allocation_receipt in self._allocation_receipts:
            memory_obj = allocation_receipt.memory_obj
            if memory_obj is None:
                continue
            for index, allocated in enumerate(self._allocated):
                if allocated is memory_obj:
                    break
            else:
                index = len(self._allocated)
                self._allocated.append(memory_obj)
            ref_receipt = allocation_receipt.ref_count_down_receipt
            if ref_receipt is not None:
                self._release_receipts.setdefault(index, ref_receipt)

    def _receipt_for(
        self,
        index: int,
        memory_obj: MemoryObj,
    ) -> Optional[RefCountDownReceipt]:
        if index in self._release_receipts:
            return self._release_receipts[index]
        for allocation_receipt in self._allocation_receipts:
            if allocation_receipt.memory_obj is memory_obj:
                receipt = allocation_receipt.ref_count_down_receipt
                if receipt is None:
                    raise RuntimeError("adopted allocation lacks a release receipt")
                self._release_receipts[index] = receipt
                return receipt
        try:
            receipt = memory_obj.make_ref_count_down_receipt()
        except (AttributeError, NotImplementedError):
            if isinstance(memory_obj, MemoryObj):
                raise RuntimeError(
                    f"{type(memory_obj).__name__} lacks an exact release receipt"
                ) from None
            receipt = None
        self._release_receipts[index] = receipt
        return receipt

    def ref_count_down_receipts_for(
        self,
        memory_objs: list[MemoryObj],
    ) -> tuple[Optional[RefCountDownReceipt], ...]:
        """Return the same release receipts that transfer into batched_put."""
        with self._lock:
            self._refresh_allocated_from_receipts()
            used: set[int] = set()
            receipts: list[Optional[RefCountDownReceipt]] = []
            for memory_obj in memory_objs:
                for index, allocated in enumerate(self._allocated):
                    if index in used or allocated is not memory_obj:
                        continue
                    used.add(index)
                    receipts.append(self._receipt_for(index, allocated))
                    break
                else:
                    raise ValueError("store cleanup does not own batched-put object")
            return tuple(receipts)

    def _release_complete(self, index: int, memory_obj: MemoryObj) -> bool:
        receipt = self._release_receipts.get(index)
        if receipt is None:
            return index in self._legacy_released
        return memory_obj.ref_count_down_receipt_complete(receipt)

    def attach_gpu_generator(self, generator: Any) -> None:
        with self._lock:
            if self._gpu_generator is not None:
                raise RuntimeError("store cleanup already owns a GPU generator")
            self._gpu_generator = generator

    def attach_put_receipt(self, receipt: PutOwnershipReceipt) -> None:
        """Publish shared put-transfer state before entering batched_put."""
        with self._lock:
            if self._pending:
                raise RuntimeError("cannot attach receipt to pending store cleanup")
            attachment_token = receipt._attach()
            self._put_receipts.append((receipt, attachment_token))

    def _storage_manager_owns(self, memory_obj: MemoryObj) -> bool:
        return any(
            receipt.owns_for(attachment_token, memory_obj)
            for receipt, attachment_token in self._put_receipts
        )

    def mark_pending(self) -> None:
        """Publish terminal cleanup with one non-fallible pointer assignment."""
        self._pending = True

    def drain(self) -> Optional[BaseException]:
        """Drain GPU work before exhaustively releasing CPU references."""
        with self._lock:
            if not self._pending:
                return None
            if self._complete:
                return None

            if self._gpu_generator is not None:
                try:
                    self._gpu_generator.close()
                except BaseException as error:
                    # A failed close can permanently exhaust a Python
                    # generator. Keep it until the next pass, where close is
                    # harmless and its connector ledger performs the retry.
                    return error
                self._gpu_generator = None

            try:
                self._gpu_connector.drain_pending_gpu_cleanups()
            except BaseException as error:
                return error

            first_error: Optional[BaseException] = None
            try:
                self._storage_manager.drain_pending_put_cleanups()
            except BaseException as error:
                first_error = error

            # Slots that never reached object adoption still name an exact live
            # address generation (or a no-allocation failure). Release those
            # reservations before considering this owner complete.
            for allocation_receipt in self._allocation_receipts:
                if allocation_receipt.adopted or allocation_receipt.release_complete():
                    continue
                try:
                    allocation_receipt.release_once()
                except BaseException as error:
                    if first_error is None:
                        first_error = error

            self._refresh_allocated_from_receipts()

            for index, memory_obj in enumerate(self._allocated):
                if self._storage_manager_owns(memory_obj):
                    continue
                receipt = self._receipt_for(index, memory_obj)
                if receipt is None and index in self._legacy_released:
                    continue
                try:
                    if receipt is None:
                        memory_obj.ref_count_down()
                    else:
                        memory_obj.ref_count_down_once(receipt)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                else:
                    if receipt is None:
                        self._legacy_released.add(index)

            cpu_complete = all(
                self._storage_manager_owns(memory_obj)
                or self._release_complete(index, memory_obj)
                for index, memory_obj in enumerate(self._allocated)
            )
            reservations_complete = all(
                allocation_receipt.adopted or allocation_receipt.release_complete()
                for allocation_receipt in self._allocation_receipts
            )
            self._complete = (
                self._gpu_generator is None
                and self._gpu_connector.pending_gpu_cleanup_count() == 0
                and self._storage_manager.pending_put_cleanup_count() == 0
                and cpu_complete
                and reservations_complete
            )
            return first_error


class CacheEngineEndSignal:
    pass


class _StorageManagerInitOwner:
    """Retain one post-init StorageManager construction attempt.

    The raw manager is published into the engine before this owner starts its
    constructor on a retained worker.  A caller-side interrupt can therefore
    interrupt only the wait: the same construction attempt remains reachable,
    and terminal engine cleanup waits for it before touching the raw manager.
    """

    def __init__(
        self,
        manager: StorageManager,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        event_manager: EventManager,
        lmcache_worker: Any,
        async_lookup_server: Any,
    ) -> None:
        self.manager = manager
        self._call = RetainedCall(
            lambda: StorageManager.__init__(
                manager,
                config,
                metadata,
                event_manager=event_manager,
                lmcache_worker=lmcache_worker,
                async_lookup_server=async_lookup_server,
            ),
            thread_name="lmcache-storage-manager-init",
        )

    def run(self) -> None:
        """Start or await the exact retained construction attempt."""
        self._call.run()

    def finish_before_close(self) -> None:
        """Wait for construction, treating its own failure as cleanup context."""
        try:
            self._call.run()
        except BaseException as error:
            # A construction failure is not a terminal-cleanup failure.  The
            # raw partial manager must still be closed.  A new interruption of
            # this waiter is different: it is not the retained worker's error
            # and must keep this close stage retryable.
            if self._call.last_error is not error:
                raise
            logger.info(
                "StorageManager construction failed before terminal cleanup: %s",
                error,
            )

    def running_in_current_thread(self) -> bool:
        """Return whether construction is reentering engine close itself."""
        return self._call.running_in_current_thread()


class _EngineCloseOwner:
    """Retain exact progress across every terminal engine child.

    The cleanup-ledger stages explicitly opt into retries because their public
    drains retain per-resource progress. Arbitrary child ``close`` methods do
    not: a child that publishes its terminal side effect and then raises must
    remain a sticky failure rather than being invoked a second time.
    """

    def __init__(self, engine: "LMCacheEngine") -> None:
        storage_manager_init_owner = getattr(
            engine,
            "_storage_manager_init_owner",
            None,
        )
        hidden_state_store = getattr(engine, "hidden_state_store", None)
        lmcache_worker = getattr(engine, "lmcache_worker", None)
        usage_context = getattr(engine, "_usage_context", None)
        storage_manager = (
            storage_manager_init_owner.manager
            if storage_manager_init_owner is not None
            else getattr(engine, "storage_manager", None)
        )

        def finish_storage_manager_construction() -> None:
            if storage_manager_init_owner is not None:
                storage_manager_init_owner.finish_before_close()

        def close_usage_context() -> None:
            if usage_context is not None:
                usage_context.close()

        def quiesce_store_owners() -> None:
            LMCacheEngine._seal_store_cleanup_admission(engine)
            active_store_count = LMCacheEngine._active_store_cleanup_count(engine)
            if active_store_count != 0:
                raise RuntimeError(
                    "cache engine retained "
                    f"{active_store_count} active layerwise store owner(s)"
                )
            LMCacheEngine.drain_pending_store_cleanups(engine)
            store_owner_count = LMCacheEngine._store_cleanup_owner_count(engine)
            if store_owner_count != 0:
                raise RuntimeError(
                    "cache engine retained "
                    f"{store_owner_count} layerwise store cleanup owner(s)"
                )

        def drain_gpu_owners() -> None:
            gpu_connector = getattr(engine, "gpu_connector", None)
            if gpu_connector is None:
                return
            gpu_connector.drain_pending_gpu_cleanups()
            if gpu_connector.pending_gpu_cleanup_count() != 0:
                raise RuntimeError("cache engine retained pending GPU cleanup")
            gpu_owner_count = getattr(
                gpu_connector,
                "gpu_cleanup_owner_count",
                lambda: 0,
            )()
            if gpu_owner_count != 0:
                raise RuntimeError("cache engine retained a GPU cleanup owner")

        def drain_put_owners() -> None:
            if storage_manager is None:
                return
            storage_manager.drain_pending_put_cleanups()
            if storage_manager.pending_put_cleanup_count() != 0:
                raise RuntimeError("cache engine retained pending put cleanup")

        def close_hidden_state_store() -> None:
            if hidden_state_store is not None:
                logger.info("Closing hidden_state_store...")
                hidden_state_store.close()

        def close_lmcache_worker() -> None:
            if lmcache_worker is not None:
                logger.info("Closing lmcache_worker...")
                lmcache_worker.close()
                logger.info("lmcache_worker closed successfully")

        def close_storage_manager() -> None:
            if storage_manager is not None:
                logger.info("Closing storage_manager...")
                storage_manager.close()
                logger.info("storage_manager closed successfully")

        # The first four operations own durable progress ledgers and are safe
        # to re-enter after their exact failure. StorageManager.close has the
        # same contract for backend owners and its event-loop shutdown.
        retryable_stages = (
            ((close_usage_context,) if usage_context is not None else ())
            + (
                (finish_storage_manager_construction,)
                if storage_manager_init_owner
                else ()
            )
            + (
                quiesce_store_owners,
                drain_gpu_owners,
                drain_put_owners,
                engine._close_lookup_cleanups,
            )
        )
        calls = [
            RetainedCall(
                call,
                retry_safe_failure=True,
                thread_name=f"lmcache-engine-close-{index}",
            )
            for index, call in enumerate(retryable_stages)
        ]
        calls.extend(
            (
                RetainedCall(
                    close_hidden_state_store,
                    thread_name="lmcache-engine-close-hidden-state",
                ),
                RetainedCall(
                    close_lmcache_worker,
                    thread_name="lmcache-engine-close-worker",
                ),
                RetainedCall(
                    close_storage_manager,
                    retry_safe_failure=True,
                    thread_name="lmcache-engine-close-storage-manager",
                ),
            )
        )
        self._calls = tuple(calls)

    @property
    def complete(self) -> bool:
        return all(call.complete for call in self._calls)

    def run(self) -> None:
        for call in self._calls:
            call.run()


class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    _close_owner_init_lock = threading.RLock()

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        logger.info("Creating LMCacheEngine with config: %s", config)
        self.config = config
        self.metadata = metadata
        self.token_database = token_database
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        self._store_cleanup_lock = threading.RLock()
        self._store_cleanup_owners: dict[int, _StoreCleanupOwner] = {}
        self._store_cleanup_closed = False
        # save_only_first_rank only works when use mla
        self.save_only_first_rank = (
            self.config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )

        if self.save_only_first_rank and self.gpu_connector is not None:
            self.broadcast_stream = (
                self.gpu_connector.load_stream
                if hasattr(self.gpu_connector, "load_stream")
                else torch_dev.Stream()
            )

        # Holds GPU-resident copies of the broadcast send buffers on the
        # leader rank so the subsequent batched_to_gpu can read from HBM
        # rather than re-reading the same L1 bytes over PCIe. Always empty
        # on non-leader ranks and outside the broadcast critical section.
        # Typed as List[MemoryObj] (the supertype) so the list can be
        # passed directly to batched_to_gpu without an invariance cast.
        self._leader_gpu_substitute_objs: List[MemoryObj] = []

        self.enable_controller = config.enable_controller

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        lmcache_worker_ids = config.get_lmcache_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        # lmcache_worker_ids is empty means start on all workers
        if (
            self.enable_controller
            and self.metadata.role != "scheduler"
            and (not lmcache_worker_ids or metadata.worker_id in lmcache_worker_ids)
        ):
            # Publish the raw child before its initializer opens sockets or
            # starts its event-loop thread.  A constructor-side BaseException
            # then leaves the partial worker reachable from this engine's
            # retained construction owner instead of orphaning it in the RHS
            # of an assignment.
            lmcache_worker = LMCacheWorker.__new__(LMCacheWorker)
            self.lmcache_worker = lmcache_worker
            LMCacheWorker.__init__(lmcache_worker, config, metadata, self)
        else:
            self.lmcache_worker = None
            logger.info(
                "LMCacheWorker is not initialized (related configs: "
                "enable_controller: %s, role: %s, worker_id: %d, worker_ids: %s).",
                self.enable_controller,
                self.metadata.role,
                self.metadata.worker_id,
                lmcache_worker_ids,
            )

        self.async_loading = config.enable_async_loading
        self.event_manager = EventManager()

        self.use_layerwise = config.use_layerwise

        # TODO: support save_only_first_rank when use layerwise
        # if use_layerwise is True, all ranks will initialize the storage_manager
        # if save_only_first_rank is False, all ranks will initialize
        # the storage_manager
        # if save_only_first_rank is True, only the first rank and
        # lookup server workers will initialize the storage_manager
        self.storage_manager: Optional[StorageManager] = None
        self._storage_manager_init_owner: Optional[_StorageManagerInitOwner] = None

        # KV events
        self.kv_events_enabled = False
        self.kv_events_enabled = config.enable_kv_events
        if self.kv_events_enabled:
            self.kv_events: List[CacheStoreEvent] = []
            logger.info("KV events are enabled.")
        else:
            logger.info("KV events are disabled.")

        # HACK: remove this in the future
        # NOTE (Jiayi): This is currently used to support
        # dropping the kv cache from the buffer in PD backend
        # at decoder.
        # A transfer-channel handoff hands the decoder its own copy, which
        # the decoder then owns. A handoff through shared storage does not:
        # the writer still owns those extents, so removing on retrieve would
        # drop a mapping the writer is holding for its own reasons.
        self.remove_after_retrieve = (
            config.enable_pd
            and not config.pd_uses_shared_storage
            and config.pd_role == "receiver"
        )

        # asymmetric store/retrieve location can be specified
        # this is typically used (but not limited) in PD system
        self.store_location = config.store_location
        self.retrieve_locations = config.retrieve_locations

        self.num_layers = metadata.kv_shape[0]
        self.fmt = None
        if self.use_layerwise:
            if metadata.use_mla:
                self.fmt = MemoryFormat.KV_MLA_FMT
            elif config.enable_blending:
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D
        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT

        # NOTE(ApostaC): we haven't support lookup-cache yet
        self.lookup_cache: dict[CacheEngineKey, Any] = {}

        # lookup_id -> {location -> [pinned keys]}
        self.lookup_pins: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Explicit sync-RPC generations must outlive the lookup server object.
        # A server can be recreated while this engine still owns request pins.
        self.lookup_pin_generations: dict[str, bytes] = {}
        self.lookup_cancelled_generations: OrderedDict[tuple[str, bytes], None] = (
            OrderedDict()
        )
        # Capacity exhaustion permanently fences explicit lookup publication
        # for this engine lifetime instead of evicting replay-safety state.
        self.lookup_explicit_replay_fenced = False
        self._lookup_pins_lock = threading.RLock()
        self.lookup_cleanup_registered: set[str] = set()
        self.lookup_cleanup_pending: set[str] = set()
        self._lookup_cleanup_retry_timer: Optional[threading.Timer] = None
        self._lookup_cleanup_closed = False

        self._usage_context: Optional[UsageContext] = None
        InitializeUsageContext(
            config,
            metadata,
            retain_context=self._retain_usage_context,
        )
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        # Initialize PinMonitor singleton with config
        PinMonitor.GetOrCreate(config, metadata)

        self.post_inited = False

        # Flag to control KVCache Check logging (can be toggled via API)
        self.kvcache_check_log_enabled = False

        gc.collect()
        if not config.py_enable_gc:
            gc.disable()

        # Health monitor reference (injected by LMCacheManager)
        self._health_monitor: Optional["HealthMonitor"] = None

        # Flag to indicate if initialization failed (irrecoverable error)
        self._init_failed = False

        # Hidden-state cache (logically separate from KV; lives on its own
        # pinned pool). Bound to storage_manager in post_init for coupled
        # eviction. None when disabled in config.
        self.hidden_state_store: Optional[HiddenStateStore] = None
        if config.enable_hidden_state_cache:
            # HiddenStateStore allocates a pinned pool in __init__; retain its
            # raw identity before entering that fallible allocation path.
            hidden_state_store = HiddenStateStore.__new__(HiddenStateStore)
            self.hidden_state_store = hidden_state_store
            HiddenStateStore.__init__(hidden_state_store, config, token_database)

    def set_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """
        Set the health monitor reference.

        This is called by LMCacheManager after creating the HealthMonitor
        to inject the reference into the engine.

        Args:
            health_monitor: The HealthMonitor instance from LMCacheManager
        """
        self._health_monitor = health_monitor

    def clear_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """Detach a health monitor if it is still the installed instance.

        Identity matching makes failed-construction rollback safe if a newer
        monitor has already been installed.

        Args:
            health_monitor: The exact monitor instance to detach.

        Returns:
            None.
        """
        if self._health_monitor is health_monitor:
            self._health_monitor = None

    def is_healthy(self) -> bool:
        """
        Check if the LMCache system is healthy.

        This method returns False if:
        - Initialization failed (irrecoverable error)
        - HealthMonitor reports unhealthy

        If no health monitor is set and initialization succeeded,
        it returns True (assume healthy).

        Returns:
            bool: True if healthy, False otherwise
        """
        if self._init_failed:
            return False
        if self._health_monitor is not None:
            return self._health_monitor.is_healthy()
        return True

    def supports_explicit_spans(self) -> bool:
        """Return whether this engine supports the explicit-span data path.

        The initial implementation is intentionally restricted to the
        synchronous layerwise CacheBlend path backed by
        :class:`SegmentTokenDatabase`.
        """
        return (
            self.use_layerwise
            and isinstance(self.token_database, SegmentTokenDatabase)
            and self.gpu_connector is not None
            and self.gpu_connector.supports_explicit_spans()
        )

    def _get_req_id(self, kwargs: dict) -> str:
        """Extracts request ID from kwargs for logging."""
        return kwargs.get("req_id", "unspecified")

    def _lookup_pins_guard(self) -> threading.RLock:
        """Return the lock serializing request pin ownership transitions."""
        lock = getattr(self, "_lookup_pins_lock", None)
        if lock is None:
            # Compatibility for lightweight test engines built via ``__new__``.
            lock = self._lookup_pins_lock = threading.RLock()
        return lock

    def _lookup_generation_state(
        self,
    ) -> tuple[dict[str, bytes], OrderedDict[tuple[str, bytes], None]]:
        """Return engine-lifetime explicit-generation ownership state."""
        generations = getattr(self, "lookup_pin_generations", None)
        if generations is None:
            generations = self.lookup_pin_generations = {}
        cancelled = getattr(self, "lookup_cancelled_generations", None)
        if cancelled is None:
            cancelled = self.lookup_cancelled_generations = OrderedDict()
        return generations, cancelled

    def _check_lookup_tombstone_capacity(
        self,
        lookup_id: str,
        generation: bytes,
    ) -> bool:
        """Fence explicit lookups instead of evicting replay-safety state."""
        with self._lookup_pins_guard():
            _, cancelled = self._lookup_generation_state()
            key = (lookup_id, generation)
            if getattr(self, "lookup_explicit_replay_fenced", False):
                return False
            if (
                key not in cancelled
                and len(cancelled) >= _LOOKUP_CANCELLED_GENERATION_LIMIT
            ):
                self.lookup_explicit_replay_fenced = True
                logger.error(
                    "Explicit lookup tombstone capacity is exhausted; "
                    "fencing explicit lookups for this engine lifetime"
                )
                return False
            return True

    def _remember_lookup_cancelled_generation(
        self,
        lookup_id: str,
        generation: bytes,
    ) -> None:
        """Remember an exact completed generation for delayed-RPC suppression.

        The history is memory-bounded but never evicts a generation without a
        transport-quiescence proof. Capacity exhaustion therefore fails closed.
        """
        with self._lookup_pins_guard():
            if not self._check_lookup_tombstone_capacity(lookup_id, generation):
                return
            _, cancelled = self._lookup_generation_state()
            key = (lookup_id, generation)
            cancelled[key] = None
            cancelled.move_to_end(key)

    def _lookup_cleanup_state(self) -> tuple[set[str], set[str]]:
        registered = getattr(self, "lookup_cleanup_registered", None)
        if registered is None:
            registered = self.lookup_cleanup_registered = set()
        pending = getattr(self, "lookup_cleanup_pending", None)
        if pending is None:
            pending = self.lookup_cleanup_pending = set()
        if not hasattr(self, "_lookup_cleanup_retry_timer"):
            self._lookup_cleanup_retry_timer = None
        if not hasattr(self, "_lookup_cleanup_closed"):
            self._lookup_cleanup_closed = False
        return registered, pending

    def register_lookup_cleanup(self, lookup_id: str) -> None:
        """Register worker ownership before any fallible request work."""
        with self._lookup_pins_guard():
            registered, _ = self._lookup_cleanup_state()
            if self._lookup_cleanup_closed:
                raise RuntimeError("lookup cleanup owner is closed")
            registered.add(lookup_id)

    def _arm_lookup_cleanup_retry_locked(self) -> None:
        _, pending = self._lookup_cleanup_state()
        if (
            self._lookup_cleanup_closed
            or not pending
            or self._lookup_cleanup_retry_timer is not None
        ):
            return
        timer = threading.Timer(
            _LOOKUP_CLEANUP_RETRY_INTERVAL_S,
            self._retry_lookup_cleanups,
        )
        timer.daemon = True
        self._lookup_cleanup_retry_timer = timer
        timer.start()

    def _retry_lookup_cleanups(self) -> None:
        current_timer = threading.current_thread()
        with self._lookup_pins_guard():
            registered, pending = self._lookup_cleanup_state()
            if self._lookup_cleanup_retry_timer is not current_timer:
                return
            self._lookup_cleanup_retry_timer = None
            if self._lookup_cleanup_closed:
                return
            lookup_ids = tuple(pending)
            # Keep the engine ownership lock through every backend call. Close
            # takes the same lock before fencing callbacks and closing storage,
            # so an already-running timer cannot resume I/O after close.
            for lookup_id in lookup_ids:
                try:
                    self._lookup_unpin_once(lookup_id)
                except Exception as error:
                    logger.warning(
                        "Worker lookup cleanup for %s remains pending: %s",
                        lookup_id,
                        error,
                    )
                else:
                    registered.discard(lookup_id)
                    pending.discard(lookup_id)
            self._arm_lookup_cleanup_retry_locked()

    def schedule_lookup_unpin(self, lookup_id: str) -> None:
        """Make a worker-owned cleanup retryable without blocking its peers."""
        with self._lookup_pins_guard():
            registered, pending = self._lookup_cleanup_state()
            if self._lookup_cleanup_closed:
                return
            registered.add(lookup_id)
            pending.add(lookup_id)
            try:
                self._lookup_unpin_once(lookup_id)
            except Exception as error:
                logger.warning(
                    "Worker lookup cleanup for %s will retry: %s",
                    lookup_id,
                    error,
                )
                self._arm_lookup_cleanup_retry_locked()
            else:
                registered.discard(lookup_id)
                pending.discard(lookup_id)

    def _close_lookup_cleanups(self) -> None:
        """Fence callbacks and retain every lookup whose cleanup failed."""
        with self._lookup_pins_guard():
            registered, pending = self._lookup_cleanup_state()
            self._lookup_cleanup_closed = True
            timer = self._lookup_cleanup_retry_timer
            self._lookup_cleanup_retry_timer = None
            if timer is not None:
                timer.cancel()
            pending.update(registered)
            lookup_ids = tuple(pending)
            first_error: Optional[BaseException] = None
            for lookup_id in lookup_ids:
                try:
                    self._lookup_unpin_once(lookup_id)
                except BaseException as error:
                    logger.error(
                        "Terminal lookup cleanup for %s failed: %s",
                        lookup_id,
                        error,
                    )
                    if first_error is None:
                        first_error = error
                else:
                    registered.discard(lookup_id)
                    pending.discard(lookup_id)
            if first_error is not None:
                raise first_error

    def mark_init_failed(self, reason: str = "") -> None:
        """
        Mark the engine as having failed initialization.

        This is called by LMCacheManager when an irrecoverable error occurs
        during initialization or post_init. Once marked, is_healthy() will
        always return False, causing the system to fall back to recomputation.

        Args:
            reason: Optional reason string for logging
        """
        self._init_failed = True
        if reason:
            logger.error("LMCacheEngine marked as init failed: %s", reason)
        else:
            logger.error("LMCacheEngine marked as init failed")

    def post_init(self, **kwargs) -> None:
        if not self.post_inited:
            logger.info("Post initializing LMCacheEngine")
            lookup_server_worker_ids = self.config.get_lookup_server_worker_ids(
                self.metadata.use_mla, self.metadata.world_size
            )
            if (
                self.lmcache_worker is not None
                or self.use_layerwise
                or not self.save_only_first_rank
                or self.metadata.is_first_rank()
                or len(lookup_server_worker_ids) == 0
                or self.metadata.worker_id in lookup_server_worker_ids
            ):
                logger.info(
                    "Initialize storage manager on rank %d, "
                    "use layerwise: %s,"
                    "save only first rank: %s",
                    self.metadata.worker_id,
                    self.use_layerwise,
                    self.save_only_first_rank,
                )
                async_lookup_server = kwargs.get("async_lookup_server", None)
                init_owner = self._storage_manager_init_owner
                if init_owner is None:
                    storage_manager = self.storage_manager
                    if storage_manager is None:
                        # Publish a close-safe raw manager before entering any
                        # constructor that can start a thread or allocate a
                        # backend arena.
                        storage_manager = StorageManager.__new__(StorageManager)
                        StorageManager._prepare_partial_state(
                            storage_manager,
                            self.config,
                            self.metadata,
                            event_manager=self.event_manager,
                            lmcache_worker=self.lmcache_worker,
                            async_lookup_server=async_lookup_server,
                        )
                        self.storage_manager = storage_manager
                    init_owner = _StorageManagerInitOwner(
                        storage_manager,
                        self.config,
                        self.metadata,
                        self.event_manager,
                        self.lmcache_worker,
                        async_lookup_server,
                    )
                    self._storage_manager_init_owner = init_owner
                elif self.storage_manager is not init_owner.manager:
                    raise RuntimeError(
                        "StorageManager construction owner changed identity"
                    )

                init_owner.run()
                if self.hidden_state_store is not None:
                    self.hidden_state_store.bind_storage_manager(
                        cast(StorageManager, self.storage_manager)
                    )
            self.post_inited = True

    def freeze(self, enabled: bool) -> None:
        """
        Set the freeze mode for the cache engine.

        When freeze mode is enabled:
        - All store operations will be skipped (no new data stored)
        - Only local_cpu backend will be used for retrieval
        - No admit/evict messages will be generated
        This protects the local_cpu hot cache from changes.

        Args:
            enabled (bool): Whether to enable freeze mode
        """
        if self.storage_manager is not None:
            self.storage_manager.set_freeze(enabled)

    def is_frozen(self) -> bool:
        """
        Get the current freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_frozen()
        return False

    def set_hot_cache(self, enabled: bool) -> None:
        """
        Dynamically enable or disable the LocalCPUBackend hot cache.

        When disabled, the existing hot cache entries will be cleared
        and no new data will be written to the hot cache.

        Args:
            enabled (bool): Whether to enable hot cache
        """
        if self.storage_manager is not None:
            self.storage_manager.set_hot_cache(enabled)

    def is_hot_cache_enabled(self) -> bool:
        """
        Get the current hot cache status of LocalCPUBackend.

        Returns:
            bool: True if hot cache is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_hot_cache_enabled()
        return False

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Future | None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: The storage manager's asynchronous completion when a batch is
            submitted, or None when no store is performed. In storage P/D mode
            the completion carries the durable publication receipt; finishing
            the GPU gather alone does not imply durable storage.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if kwargs.get("explicit_spans") is not None:
            raise NotImplementedError(
                "explicit spans are supported only by store_layer"
            )

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store operation")
            return None

        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )

        if self._is_passive():
            logger.debug("rank=%d ignore store", self.metadata.worker_id)
            return None

        assert self.storage_manager is not None

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        # Initialize num_to_store_tokens to avoid reference before assignment
        num_to_store_tokens = 0

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device=torch_device_type
            )

        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=False,
        )

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store operation for %d tokens",
                num_to_store_tokens,
            )
            return None

        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts: List[int] = []
        ends: List[int] = []
        keys: List[CacheEngineKey] = []
        memory_objs: List[MemoryObj] = []

        tot_kv_size = 0
        tot_token_num = 0

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        with store_stats.profile_process_tokens():
            prev_key = 0
            for start, end, key in self.token_database.process_tokens(
                tokens,
                hashes,
                offsets,
                mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)
                # A degenerate range carries no tokens (for example a separator
                # at the very end of a blend input). Skip it: a zero-byte
                # allocation is invalid, and the allocation stack reacts to a
                # rejected request by evicting cached objects or spinning in a
                # busy loop.
                if end <= start:
                    logger.debug(
                        "Skipping empty token range [%d, %d) during store", start, end
                    )
                    continue

                # Allocate the memory object
                num_tokens = end - start
                kv_shapes = self.metadata.get_shapes(num_tokens)
                kv_dtypes = self.metadata.get_dtypes()

                # TODO (Jiayi): should be batched in the future
                memory_obj = self.storage_manager.allocate(
                    kv_shapes,
                    kv_dtypes,
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                    fmt=self.fmt,
                )
                if memory_obj is None:
                    logger.warning(
                        "Local cpu memory under pressure so"
                        " choosing to store only "
                        " %d total chunks of KV cache.",
                        len(memory_objs),
                    )
                    break

                starts.append(start)
                ends.append(end)
                keys.append(key)
                memory_objs.append(memory_obj)
                tot_kv_size += memory_obj.get_size()
                tot_token_num += num_tokens

                # Create KV event
                if self.kv_events_enabled:
                    stored_event = CacheStoreEvent(
                        block_hashes=[key.chunk_hash],
                        parent_block_hash=None if start == 0 else prev_key,
                        token_ids=[],
                        block_size=num_tokens,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    if tokens is not None:
                        stored_event.token_ids = convert_tokens_to_list(
                            tokens,
                            start,
                            end,
                        )
                        if isinstance(tokens, torch.Tensor):
                            stored_event.medium = tokens.device
                    elif hashes is not None:
                        stored_event.token_ids = hashes[start : end + 1]
                    logger.debug(
                        (
                            "Added kv cache event '%s' to kv cache events queue"
                            % stored_event
                        )
                    )
                    self.kv_events.append(stored_event)
                    prev_key = key.chunk_hash

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return None

        try:
            with store_stats.profile_from_gpu():
                self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
        except BaseException:
            # Connector synchronization failures retain independent references
            # before raising. Release the engine's allocation references on
            # every failure, including reuse of a pooled object after __del__.
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()
            raise

        with store_stats.profile_put():
            transfer_spec = kwargs.get("transfer_spec", None)
            # TODO: we implicitly rely on batched_put to call ref_count_down
            # this management should be done in a cleaner way
            completion = self.storage_manager.batched_put(
                keys,
                memory_objs,
                transfer_spec=transfer_spec,
                location=self.store_location,
            )

        self.stats_monitor.on_store_finished(
            store_stats,
            tot_token_num,
        )
        tot_time = store_stats.time_to_store()

        logger.info(
            "[req_id=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms",
            req_id,
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
            store_stats.put_time * 1000,
        )
        return completion

    def _ensure_store_cleanup_state(self) -> None:
        """Initialize retry state for lightweight test and recovery engines."""
        if not hasattr(self, "_store_cleanup_lock"):
            self._store_cleanup_lock = threading.RLock()
        if not hasattr(self, "_store_cleanup_owners"):
            self._store_cleanup_owners = {}
        if not hasattr(self, "_store_cleanup_closed"):
            self._store_cleanup_closed = False

    def _register_store_cleanup(self, owner: _StoreCleanupOwner) -> None:
        LMCacheEngine._ensure_store_cleanup_state(self)
        with self._store_cleanup_lock:
            if self._store_cleanup_closed:
                raise RuntimeError(
                    "cache engine close has begun; refusing layerwise store"
                )
            self._store_cleanup_owners[id(owner)] = owner

    def _require_store_cleanup_admission_open(self) -> None:
        """Reject post-close store calls before they touch collaborators."""
        LMCacheEngine._ensure_store_cleanup_state(self)
        with self._store_cleanup_lock:
            if self._store_cleanup_closed:
                raise RuntimeError(
                    "cache engine close has begun; refusing layerwise store"
                )

    def _seal_store_cleanup_admission(self) -> None:
        """Permanently reject new layerwise stores before terminal cleanup."""
        LMCacheEngine._ensure_store_cleanup_state(self)
        with self._store_cleanup_lock:
            self._store_cleanup_closed = True

    def _active_store_cleanup_count(self) -> int:
        """Return registered stores that have not entered cleanup yet."""
        LMCacheEngine._ensure_store_cleanup_state(self)
        with self._store_cleanup_lock:
            owners = tuple(self._store_cleanup_owners.values())
        return sum(owner.operation_alive and not owner.pending for owner in owners)

    def _store_cleanup_owner_count(self) -> int:
        """Return every active or pending layerwise-store owner."""
        LMCacheEngine._ensure_store_cleanup_state(self)
        with self._store_cleanup_lock:
            return len(self._store_cleanup_owners)

    def _has_store_cleanup_owner(self, owner: _StoreCleanupOwner) -> bool:
        """Return whether ``owner`` was published in the engine ledger."""
        LMCacheEngine._ensure_store_cleanup_state(self)
        with self._store_cleanup_lock:
            return self._store_cleanup_owners.get(id(owner)) is owner

    def finalize_store_cleanup(self, owner: _StoreCleanupOwner) -> bool:
        """Atomically move an exactly published store owner into cleanup.

        The ledger identity check and the owner's single pending publication
        occur while the same ledger lock is held. Public drains can therefore
        never observe a registered, unwound store between those two states.
        """
        LMCacheEngine._ensure_store_cleanup_state(self)
        with self._store_cleanup_lock:
            # Publish first. Marking an unregistered local owner is harmless,
            # while interruption at every later helper boundary leaves a
            # registered owner visible to public drains as pending.
            owner.mark_pending()
            return self._store_cleanup_owners.get(id(owner)) is owner

    def _drain_store_cleanup_owner(
        self,
        owner: _StoreCleanupOwner,
    ) -> Optional[BaseException]:
        error = owner.drain()
        if owner.complete:
            with self._store_cleanup_lock:
                self._store_cleanup_owners.pop(id(owner), None)
        return error

    def drain_pending_store_cleanups(self) -> None:
        """Retry all pending layerwise-store cleanup owners.

        Raises:
            BaseException: The first cleanup failure after exhausting owners.
        """
        LMCacheEngine._ensure_store_cleanup_state(self)
        first_error: Optional[BaseException] = None
        with self._store_cleanup_lock:
            owners = tuple(self._store_cleanup_owners.values())
        for owner in owners:
            if not owner.pending:
                if owner.operation_alive:
                    continue
                if not self.finalize_store_cleanup(owner):
                    continue
            error = LMCacheEngine._drain_store_cleanup_owner(self, owner)
            if first_error is None and error is not None:
                first_error = error
        if first_error is not None:
            raise first_error

    def pending_store_cleanup_count(self) -> int:
        """Return the number of retry-owned layerwise-store cleanup states."""
        LMCacheEngine._ensure_store_cleanup_state(self)
        with self._store_cleanup_lock:
            return sum(
                owner.pending or not owner.operation_alive
                for owner in self._store_cleanup_owners.values()
            )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        self._require_explicit_spans_supported(explicit_spans)
        if explicit_spans is not None:
            explicit_spans.validate_token_count(len(tokens))
        self._require_store_cleanup_admission_open()

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store_layer operation")
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store_layer operation"
        )
        self.drain_pending_store_cleanups()

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)

        operation_lease: Optional[_StoreOperationLease] = _StoreOperationLease()
        cleanup_owner = _StoreCleanupOwner(
            self.gpu_connector,
            self.storage_manager,
            operation_lease,
        )
        operation_error: Optional[BaseException] = None
        try:
            self._register_store_cleanup(cleanup_owner)
            # KVCache Check logging
            self._log_kvcache_for_check(
                operation="Layerwise store",
                kwargs=kwargs,
                token_count=num_to_store_tokens,
                require_req_id=True,
            )

            monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

            # Check if freeze mode is enabled. Even this yield-only path owns
            # an admission record so terminal close cannot race a live store.
            if self.is_frozen():
                logger.debug(
                    "Freeze mode enabled, skipping store_layer for %d tokens",
                    num_to_store_tokens,
                )
                for _layer_id in range(self.num_layers):
                    yield
                return

            starts: list[int] = []
            ends: list[int] = []
            keys: list[list[CacheEngineKey]] = []
            memory_objs: list[list[MemoryObj]] = []
            tot_token_num = 0
            kv_dtype = self.metadata.kv_dtype
            prev_key = 0
            chunk_info_iterator = self._process_layerwise_tokens(
                tokens,
                mask,
                request_configs,
                explicit_spans,
            )
            for start, end, key in chunk_info_iterator:
                assert isinstance(key, CacheEngineKey)

                keys_multi_layer: list[CacheEngineKey] = [
                    layer_key for layer_key in key.split_layers(self.num_layers)
                ]
                if explicit_spans is not None:
                    hit_layers, block_mapping = self.storage_manager.batched_contains(
                        keys_multi_layer,
                        self.retrieve_locations,
                        False,
                    )
                    already_stored = (
                        hit_layers == self.num_layers and len(block_mapping) == 1
                    )
                else:
                    # Preserve the legacy first-layer shortcut.
                    already_stored = bool(
                        self.storage_manager.contains(
                            keys_multi_layer[0], self.retrieve_locations
                        )
                    )
                if already_stored:
                    continue

                num_tokens = end - start
                kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)
                supports_receipts = getattr(
                    self.storage_manager,
                    "supports_allocation_receipts",
                    lambda: False,
                )()
                if supports_receipts:
                    allocation_receipts = cleanup_owner.reserve_allocation_receipts(
                        self.num_layers
                    )
                    memory_objs_multi_layer = self.storage_manager.batched_allocate(
                        kv_shape_single_layer,
                        kv_dtype,
                        batch_size=self.num_layers,
                        fmt=self.fmt,
                        busy_loop=self.config.get_extra_config_value(
                            "force_store_wait",
                            False,
                        ),
                        allocation_receipts=allocation_receipts,
                    )
                else:
                    memory_objs_multi_layer = self.storage_manager.batched_allocate(
                        kv_shape_single_layer,
                        kv_dtype,
                        batch_size=self.num_layers,
                        fmt=self.fmt,
                        busy_loop=self.config.get_extra_config_value(
                            "force_store_wait",
                            False,
                        ),
                    )
                if memory_objs_multi_layer is None:
                    logger.warning(
                        "Local cpu memory under pressure so"
                        " choosing to not store the KV cache."
                    )
                    break

                cleanup_owner.add_allocated(memory_objs_multi_layer)
                starts.append(start)
                ends.append(end)
                keys.append(keys_multi_layer)
                memory_objs.append(memory_objs_multi_layer)
                tot_token_num += num_tokens

                if self.kv_events_enabled and tokens is not None:
                    stored_event = CacheStoreEvent(
                        block_hashes=[key.chunk_hash],
                        parent_block_hash=None if start == 0 else prev_key,
                        token_ids=[],
                        block_size=num_tokens,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    stored_event.token_ids = convert_tokens_to_list(
                        tokens,
                        start,
                        end,
                    )
                    if isinstance(tokens, torch.Tensor):
                        stored_event.medium = tokens.device
                    logger.debug(
                        "Added kv cache event '%s' to kv cache events queue",
                        stored_event,
                    )
                    self.kv_events.append(stored_event)
                    prev_key = key.chunk_hash

            if keys:
                layer_memory_objs = [
                    list(row) for row in zip(*memory_objs, strict=False)
                ]
                layer_keys = [list(row) for row in zip(*keys, strict=False)]
                tot_kv_size = sum(
                    memory_obj.get_size()
                    for layer_objs in layer_memory_objs
                    for memory_obj in layer_objs
                )

                assert_layerwise_gpu_connector(self.gpu_connector)
                t_start = time.perf_counter()
                mem_obj_generator = self.gpu_connector.batched_from_gpu(
                    layer_memory_objs,
                    starts,
                    ends,
                    **kwargs,
                )
                cleanup_owner.attach_gpu_generator(mem_obj_generator)
                next(mem_obj_generator)

                for layer_id in range(self.num_layers):
                    yield
                    next(mem_obj_generator)
                    # The receipt is visible to both cleanup owners before the
                    # fallible call. Its atomic adoption decides exactly which
                    # owner releases this layer, including interruptions at
                    # StorageManager admission and post-seal rejection.
                    owned_layer = layer_memory_objs[layer_id]
                    ownership_receipt = PutOwnershipReceipt(
                        owned_layer,
                        cleanup_owner.ref_count_down_receipts_for(owned_layer),
                    )
                    cleanup_owner.attach_put_receipt(ownership_receipt)
                    self.storage_manager.batched_put(
                        layer_keys[layer_id],
                        layer_memory_objs[layer_id],
                        location=self.store_location,
                        ownership_receipt=ownership_receipt,
                    )

                tot_time = time.perf_counter() - t_start
                logger.info(
                    "[req_id=%s] Stored %d out of total %d tokens. "
                    "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s",
                    req_id,
                    tot_token_num,
                    len(tokens),
                    tot_kv_size / 1024**3,
                    tot_time * 1000,
                    tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
                )
            else:
                for _layer_id in range(self.num_layers):
                    yield

            self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
            yield
        except GeneratorExit:
            raise
        except BaseException as error:
            operation_error = error
            raise
        finally:
            # End the operation lease before entering the fallible finalization
            # helper. If interruption lands on its first line, public drains
            # can promote this now-dead nonpending owner.
            operation_lease = None
            # Registration can be interrupted after ledger publication. Query
            # exact identity so an unregistered sealed rejection is a no-op,
            # while a published owner always transitions into retry cleanup.
            if self.finalize_store_cleanup(cleanup_owner):
                cleanup_error = self._drain_store_cleanup_owner(cleanup_owner)
                if cleanup_error is not None:
                    if operation_error is None:
                        raise cleanup_error
                    logger.error(
                        "Layerwise store cleanup also failed after %s: %s",
                        operation_error,
                        cleanup_error,
                    )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        *,
        storage_pd_read_context: Optional[RawBlockReadContext] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param Optional[RawBlockReadContext] storage_pd_read_context:
            Identity of the adopted publication for a synchronous storage P/D
            restore. This follows storage reads, not GPU connector operations.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.

        Active synchronous retrieval does not release lookup pins. After
        retrieval, the caller must release its pins with ``lookup_unpin(lookup_id)``
        before issuing further blocking retrievals that may need cache space.
        """
        if kwargs.get("explicit_spans") is not None:
            raise NotImplementedError(
                "explicit spans are supported only by retrieve_layer"
            )

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve operation")
            return torch.zeros(len(tokens), dtype=torch.bool)

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        tot_kv_size = 0

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="retrieve",
            kwargs=kwargs,
            token_count=num_required_tokens,
            require_req_id=True,
        )

        retrieve_stats = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[ProcessedChunk] = []
        if not self._is_passive():
            with retrieve_stats.profile_process_tokens():
                if self.async_loading:
                    reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )
                else:
                    reordered_chunks, tot_kv_size = self._process_tokens_internal(
                        tokens,
                        mask,
                        ret_mask,
                        storage_pd_read_context=storage_pd_read_context,
                        **kwargs,
                    )

        retrieve_completed = False
        try:
            if self.save_only_first_rank:
                with retrieve_stats.profile_broadcast():
                    with torch_dev.stream(self.broadcast_stream):
                        self._broadcast_or_receive_memory_objs(
                            reordered_chunks,
                            ret_mask,
                        )

                    # A shared load stream orders broadcast before scatter.
                    # Other connectors need the broadcast to finish explicitly.
                    if not hasattr(self.gpu_connector, "load_stream"):
                        self.broadcast_stream.synchronize()

            # NOTE(Jiayi): memory_obj doesn't have to be a pinned
            # cpu tensor for the sake of performance.
            # For example, disk->gpu is faster than disk->cpu->gpu.
            # RDMA is another example.
            if len(reordered_chunks) > 0:
                with retrieve_stats.profile_to_gpu():
                    _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
                    # When save_only_first_rank is enabled, the leader rank's
                    # memory_objs from L1 are CPU-resident. The broadcast above
                    # already created GPU-resident copies on the leader to use as
                    # the NCCL send buffer. Substitute those here so this kernel
                    # reads from HBM rather than re-reading the same L1 bytes
                    # over PCIe via zero-copy mapped pinned memory. Without this
                    # swap, batched_to_gpu on the leader takes ~9 ms (PCIe-bound)
                    # while passive ranks take ~0.5 ms (HBM-bound) — a structural
                    # asymmetry on the critical path of every retrieve.
                    if self.save_only_first_rank and self.metadata.is_first_rank():
                        if len(self._leader_gpu_substitute_objs) == len(memory_objs):
                            memory_objs_for_togpu = self._leader_gpu_substitute_objs
                        else:
                            # Substitute list should always match memory_objs after
                            # _broadcast_or_receive_memory_objs has run on the
                            # leader.  A mismatch indicates a bug or stale state;
                            # fall back to the CPU L1 source so retrieval is still
                            # correct, but warn so the issue is visible.
                            logger.warning(
                                "Leader rank: GPU substitute count (%d) does not "
                                "match memory_objs count (%d); falling back to "
                                "CPU L1 source for batched_to_gpu (PCIe-bound, "
                                "~9 ms slower).",
                                len(self._leader_gpu_substitute_objs),
                                len(memory_objs),
                            )
                            memory_objs_for_togpu = list(memory_objs)
                    else:
                        memory_objs_for_togpu = list(memory_objs)
                    try:
                        self.gpu_connector.batched_to_gpu(
                            memory_objs_for_togpu, list(starts), list(ends), **kwargs
                        )
                    finally:
                        # Release GPU substitute references so the temporary
                        # buffers can be freed; original memory_objs in
                        # reordered_chunks are still tracked for the cleanup
                        # loop below. Done in `finally` so a raise from
                        # batched_to_gpu (e.g. CUDA OOM) does not leave the
                        # references dangling on this long-lived engine.
                        if self.save_only_first_rank and self.metadata.is_first_rank():
                            self._leader_gpu_substitute_objs = []
            retrieve_completed = True
        finally:
            # TODO(Jiayi): Remove the following for loop with batched operations
            # TODO(Jiayi): Need to refactor the `remove_after_retrieve` logic.
            for key, memory_obj, _, _ in reordered_chunks:
                if (
                    retrieve_completed
                    and self.remove_after_retrieve
                    and not self._is_passive()
                ):
                    assert self.storage_manager is not None
                    self.storage_manager.remove(key, self.retrieve_locations)
                    # Sync PDBackend.remove() does NOT call ref_count_down() internally
                    # (unlike async PD and other backends), so we must call it manually.
                    # See pd_backend.py line 605 TODO comment.
                    if self._is_sync_pd_backend():
                        memory_obj.ref_count_down()
                else:
                    # Blocking gets acquire references, not ownership of lookup pins.
                    # The lookup owner releases those pins through lookup_unpin().
                    if (
                        self.async_loading or self._is_passive()
                    ) and memory_obj.is_pinned:
                        memory_obj.unpin()
                    memory_obj.ref_count_down()

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(
            retrieve_stats,
            retrieved_tokens,
        )
        onload_time = retrieve_stats.time_to_retrieve()
        # The retrieved may be larger than the need_to_load
        # Example (page_size=16, chunk_size=256):
        #
        # chunks:  [0..255]                [256..511]
        # pages:   [0..15]...[240..255]    [256..271][272..287] ...
        #
        # num_computed_tokens = 288 => vLLM already has [0..287] (18 pages)
        # LMCache hit_prefix_tokens = 512 => cache covers [0..511] (2 chunks)
        #
        # Skip chunk 1, retrieve chunk 2, overwrite [256..287] (32-token overlap)
        # need_to_load: 512 - 288 = 224 tokens
        # retrieved: 256 tokens
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d required tokens "
                "(from %d total tokens). size: %.4f gb, "
                "cost %.4f ms, throughput: %.4f GB/s;",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
                tot_kv_size / 1024**3,
                onload_time * 1000,
                tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
            )
        return ret_mask

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """
        self._require_explicit_spans_supported(explicit_spans)
        if explicit_spans is not None:
            explicit_spans.validate_token_count(len(tokens))

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve_layer operation")
            miss_mask = torch.zeros(len(tokens), dtype=torch.bool)
            # Preserve the prime + N layers + finish choreography so legacy
            # consumers remain bounded even when the health check fails.
            yield miss_mask
            for _ in range(self.num_layers):
                yield None
            yield miss_mask
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        location = None
        chunk_info_iterator = self._process_layerwise_tokens(
            tokens,
            mask,
            request_configs,
            explicit_spans,
        )
        for start, end, key in chunk_info_iterator:
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer: list[CacheEngineKey] = [
                layer_key for layer_key in key.split_layers(self.num_layers)
            ]

            if explicit_spans is not None:
                hit_layers, block_mapping = self.storage_manager.batched_contains(
                    keys_multi_layer,
                    self.retrieve_locations,
                    False,
                )
                if hit_layers == self.num_layers and len(block_mapping) > 1:
                    raise NotImplementedError(
                        "explicit layerwise spans require all layers in one "
                        "storage location"
                    )
                current_location = (
                    next(iter(block_mapping))
                    if hit_layers == self.num_layers and len(block_mapping) == 1
                    else None
                )
            else:
                # Preserve the legacy first-layer shortcut.
                current_location = self.storage_manager.contains(
                    keys_multi_layer[0], self.retrieve_locations
                )

            if current_location:
                if location is None:
                    location = current_location
                else:
                    # TODO(Jiayi): Support multi-location retrieval in the future
                    if explicit_spans is not None and location != current_location:
                        raise NotImplementedError(
                            "explicit layerwise spans require all hits in one "
                            "storage location"
                        )
                    assert location == current_location, (
                        "All retrieved keys should be from the same location "
                        "when use layerwise retrieval."
                        "Please support multi-location retrieval in the future."
                    )
            else:
                if explicit_spans is not None:
                    continue
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)

            ret_mask[start:end] = True

        mem_obj_consumer = None
        get_generator = None
        to_count_down: list[MemoryObj] = []
        released_refs: set[int] = set()
        retrieval_owns_pins = False
        released_pins: set[int] = set()
        # The storage task is submitted before each scheduler-visible yield.
        # Keep its ownership explicit so Generator.close() at that boundary
        # drains the uncancellable backend work and harvests returned objects.
        pending_task: Any = None
        pending_layer_id: Optional[int] = None
        try:
            if keys:
                # Transpose the keys into layer major format
                keys_layer_major = [list(row) for row in zip(*keys, strict=False)]

                retrieval_owns_pins = (
                    self.storage_manager.layerwise_batched_get_returns_pinned(
                        location=location
                    )
                )
                get_generator = self.storage_manager.layerwise_batched_get(
                    keys_layer_major,
                    location=location,
                )

                assert_layerwise_gpu_connector(self.gpu_connector)

                connector_kwargs = dict(kwargs)
                if explicit_spans is not None:
                    connector_kwargs["buffer_span"] = explicit_spans.cached_span
                mem_obj_consumer = self.gpu_connector.batched_to_gpu(
                    starts, ends, **connector_kwargs
                )
                next(mem_obj_consumer)

                for layer_id in range(self.num_layers):
                    pending_task = next(get_generator)
                    pending_layer_id = layer_id

                    assert pending_task is not None

                    if layer_id == 0:
                        # NOTE(Yuwei): For sglang integration we need to provide
                        # retrieved tokens number in the first layer loading since
                        # there is no lookup.
                        yield torch.sum(ret_mask)
                    else:
                        yield None

                    mem_objs_result = pending_task.result()
                    returned_count = (
                        len(mem_objs_result) if mem_objs_result is not None else 0
                    )
                    mem_objs_layer = (
                        [mem_obj for mem_obj in mem_objs_result if mem_obj is not None]
                        if mem_objs_result is not None
                        else []
                    )
                    # Track every returned object for cleanup before validating
                    # the result. In particular, a backend can return a short
                    # list under staging-memory pressure, or an overlong/partial
                    # list due to a backend bug. None of those results may reach
                    # the GPU connector or contribute to a successful hit mask.
                    to_count_down.extend(mem_objs_layer)
                    pending_task = None
                    pending_layer_id = None
                    expected_count = len(keys_layer_major[layer_id])
                    if (
                        returned_count != expected_count
                        or len(mem_objs_layer) != expected_count
                    ):
                        raise RuntimeError(
                            "Invalid layerwise retrieval result for layer "
                            f"{layer_id}: expected {expected_count} non-None "
                            f"objects, got {len(mem_objs_layer)} non-None out "
                            f"of {returned_count}"
                        )
                    mem_obj_consumer.send(mem_objs_layer)

                for index, mem_obj in enumerate(to_count_down):
                    released_refs.add(index)
                    mem_obj.ref_count_down()
            else:
                # If no cache is found, preserve the layerwise yield count.  No
                # GPU consumer exists in this path, so finalization must skip it.
                # The first boundary explicitly reports the zero-hit outcome.
                for layer_id in range(self.num_layers):
                    if layer_id == 0:
                        yield torch.sum(ret_mask)
                    else:
                        yield None

            yield None

            # Synchronize the last layer when a load was actually started.
            if mem_obj_consumer is not None:
                next(mem_obj_consumer)

            # Unpin any disk-loaded staging objects now that the device-side sync
            # has been enqueued (mem_obj_consumer advanced past its sync point).
            # Without this, pin_count stays at 1 forever and the CPU staging pool
            # fills up, causing the next retrieve to deadlock inside allocate().
            if retrieval_owns_pins:
                for index, mem_obj in enumerate(to_count_down):
                    released_pins.add(index)
                    mem_obj.unpin()

            retrieved_tokens = torch.sum(ret_mask)
            self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
            if not self._is_passive():
                logger.info(
                    "[req_id=%s] Retrieved %d out of %d out of total %d tokens",
                    req_id,
                    retrieved_tokens,
                    num_required_tokens,
                    len(tokens),
                )

            yield ret_mask
        finally:
            # ``next(get_generator)`` submits backend work before this
            # generator yields. Closing at that first-yield boundary must not
            # abandon the future: disk reads run in threads and cannot safely
            # be cancelled while they still write into staging memory. Drain
            # the task, harvest every returned object, then release it below.
            if pending_task is not None:
                try:
                    pending_result = pending_task.result()
                    if pending_result is not None:
                        to_count_down.extend(
                            mem_obj for mem_obj in pending_result if mem_obj is not None
                        )
                except Exception:
                    logger.exception(
                        "Failed to drain layer %s storage retrieval during close",
                        pending_layer_id,
                    )
                finally:
                    pending_task = None
                    pending_layer_id = None

            # Closing the GPU generator releases its request-owned temporary
            # buffers and clears any published layer mapping.
            if mem_obj_consumer is not None:
                try:
                    mem_obj_consumer.close()
                except Exception:
                    logger.exception("Failed to close layerwise GPU consumer")

            close_get_generator = getattr(get_generator, "close", None)
            if close_get_generator is not None:
                try:
                    close_get_generator()
                except Exception:
                    logger.exception("Failed to close layerwise storage retrieval")

            # A task may fail after some layers returned memory objects.  Release
            # and release every retrieval-owned resource that did arrive, without
            # masking the request's original exception if cleanup also fails.
            for index, mem_obj in enumerate(to_count_down):
                if index not in released_refs:
                    try:
                        mem_obj.ref_count_down()
                    except Exception:
                        logger.exception(
                            "Failed to release layerwise retrieved memory object"
                        )
                if retrieval_owns_pins and index not in released_pins:
                    try:
                        released_pins.add(index)
                        mem_obj.unpin()
                    except Exception:
                        logger.exception(
                            "Failed to unpin layerwise retrieved memory object"
                        )

    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: the input tokens,
        with shape [seq_len]

        :param Optional[List[int]] hashes: the input hashes, with length [num_chunks]
        :param Optional[List[int]] offsets: the offsets of each chunk,
        with length [num_chunks]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param Optional[str] lookup_id: The lookup ID to
            associate with the lookup. When pin is true, this argument is
            required to be not None.

        :param bool pin: If True, pin the KV cache in the storage.

        :param Optional[dict] request_configs: the configs of the request.

        :return: An int indicating how many prefix tokens exist inside LMCache.
        """
        self._require_explicit_spans_supported(explicit_spans)
        if explicit_spans is not None and (tokens is None or hashes is not None):
            raise ValueError("explicit spans require tokens and do not accept hashes")
        if explicit_spans is not None:
            assert tokens is not None
            explicit_spans.validate_token_count(len(tokens))

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping lookup operation")
            return 0

        assert self.storage_manager is not None

        if tokens is not None:
            lookup_stats = self.stats_monitor.on_lookup_request(len(tokens))
        else:
            assert offsets is not None
            assert hashes is not None
            lookup_stats = self.stats_monitor.on_lookup_request(sum(offsets))

        if search_range is None:
            search_range = self.retrieve_locations

        res = 0
        try:
            if explicit_spans is not None:
                assert tokens is not None
                assert isinstance(self.token_database, SegmentTokenDatabase)
                chunk_info_iterator = self.token_database.process_explicit_spans(
                    tokens=tokens,
                    explicit_spans=explicit_spans,
                    request_configs=request_configs,
                )
                res = self._lookup_explicit_spans(
                    chunk_info_iterator,
                    search_range,
                    pin,
                    lookup_id,
                )
                return res

            chunk_info_iterator = self.token_database.process_tokens(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
            )

            # TODO: support batched_contains when layerwise is enabled
            if self.use_layerwise:
                for start, end, key in chunk_info_iterator:
                    assert isinstance(key, CacheEngineKey)

                    # TODO(Jiayi): Optimize by checking only the existence of the key
                    # of one layer
                    key_all_layers = key.split_layers(self.num_layers)

                    hit_chunks, block_mapping = self.storage_manager.batched_contains(
                        key_all_layers,  # type: ignore
                        search_range,
                        pin,
                    )
                    # Only all layers are hit and hit in one location,
                    # we consider this key as a hit
                    if hit_chunks == self.num_layers and len(block_mapping) == 1:
                        if pin:
                            assert lookup_id is not None, (
                                "lookup_id is required when pin is True"
                            )
                            location = next(iter(block_mapping.keys()))
                            with self._lookup_pins_guard():
                                self.lookup_pins[lookup_id][location].extend(
                                    key_all_layers
                                )
                        res = end
                        continue
                    return res
            else:
                chunk_info_list = []
                keys = []
                for chunk_info in chunk_info_iterator:
                    assert isinstance(chunk_info[2], CacheEngineKey)
                    start, end, _ = chunk_info
                    chunk_info_list.append(chunk_info)
                    # chunk_info contains (start, end, key)
                    # chunk_info[2] is the key
                    keys.append(chunk_info[2])
                # hit chunks by prefix matching
                hit_chunks, block_mapping = self.storage_manager.batched_contains(
                    keys, search_range, pin
                )
                if pin and block_mapping:
                    assert lookup_id is not None, (
                        "lookup_id is required when pin is True"
                    )
                    with self._lookup_pins_guard():
                        self.lookup_pins[lookup_id] = block_mapping
                for idx, (start, end, key) in enumerate(chunk_info_list):
                    if idx < hit_chunks:
                        res = end
                        continue
                    return res

            # all tokens where found, return the maximal end
            return res
        finally:
            self.stats_monitor.on_lookup_finished(lookup_stats, res)
            # vllm lookup sets pin to True
            if pin:
                # touch_cache is tightly coupled with batched_contains
                self.storage_manager.touch_cache()

    def release_unread_storage_publication(
        self,
        req_id: str,
        receipt: RawBlockPublicationReceipt,
        *,
        reason: str = "",
    ) -> bool:
        """Resolve one published request that will never be read.

        For the configuration that runs without a consumer: nothing was
        ever told about these publications, so nothing will acknowledge
        them, and the extents would stay held until the writer stopped
        publishing altogether. The backend still refuses to release a
        publication some consumer claimed.

        Args:
            req_id: The request name the publication was announced under.
            receipt: Durable publication identity to match against.
            reason: Why no reader was assigned, for the writer's log.

        Returns:
            Whether the hold was released.
        """
        if self.storage_manager is None:
            raise RuntimeError("storage P/D requires a storage manager")
        backends = [
            backend
            for backend in self.storage_manager.storage_backends.values()
            if callable(getattr(backend, "release_unread_publication", None))
        ]
        if len(backends) != 1:
            raise RuntimeError(
                "storage P/D requires exactly one publication-aware backend"
            )
        return cast(Any, backends[0]).release_unread_publication(
            req_id, receipt, reason=reason
        )

    def adopt_storage_publication(
        self,
        tokens: Union[torch.Tensor, List[int]],
        receipt: RawBlockPublicationReceipt,
        *,
        request_configs: Optional[dict] = None,
        timeout_ms: int = 5000,
    ) -> Optional[int]:
        """Fence a shared-storage reader to an advertised request manifest.

        The decoder request contains the prefiller's first generated token,
        while the published KV manifest describes only the original prompt.
        Try the exact token list first, then the list without that continuation
        token.  Each candidate must still match the receipt's ordered digest.

        Args:
            tokens: Decoder tokens that may include one continuation token.
            receipt: Durable publication identity advertised by the writer.
            request_configs: Per-request cache key configuration.
            timeout_ms: Maximum time to wait for the compatible generation.

        Returns:
            The number of tokens covered by the adopted manifest, or None when
            no candidate matches.
        """
        if self.storage_manager is None:
            raise RuntimeError("storage P/D requires a storage manager")

        adoption_backends = [
            backend
            for backend in self.storage_manager.storage_backends.values()
            if callable(getattr(backend, "adopt_publication", None))
        ]
        if len(adoption_backends) != 1:
            raise RuntimeError(
                "storage P/D requires exactly one publication-aware backend"
            )
        backend = cast(Any, adoption_backends[0])

        token_candidates = [tokens]
        if len(tokens) > 0:
            token_candidates.append(tokens[:-1])
        for candidate_index, candidate_tokens in enumerate(token_candidates):
            chunk_infos = list(
                self.token_database.process_tokens(
                    tokens=candidate_tokens,
                    request_configs=request_configs,
                )
            )
            keys = [chunk_info[2] for chunk_info in chunk_infos]
            if len(keys) != receipt.key_count:
                continue
            candidate_timeout = timeout_ms if candidate_index else 0
            if backend.adopt_publication(
                receipt,
                keys,
                timeout_ms=candidate_timeout,
            ):
                return len(candidate_tokens)
        return None

    def publish_existing_storage_request(
        self,
        tokens: Union[torch.Tensor, List[int]],
        transfer_spec: Any,
        *,
        request_configs: Optional[dict] = None,
    ) -> Future:
        """Publish an all-deduplicated storage-P/D request.

        Args:
            tokens: Tokens whose cache keys belong to the request manifest.
            transfer_spec: P/D transfer metadata carrying request identity.
            request_configs: Per-request cache key configuration.

        Returns:
            Future completed with the durable publication receipt.
        """
        if self.storage_manager is None:
            raise RuntimeError("storage P/D requires a storage manager")
        backends = [
            backend
            for backend in self.storage_manager.storage_backends.values()
            if callable(getattr(backend, "publish_existing_request", None))
        ]
        if len(backends) != 1:
            raise RuntimeError(
                "storage P/D requires exactly one publication-aware backend"
            )
        keys = [
            chunk_info[2]
            for chunk_info in self.token_database.process_tokens(
                tokens=tokens,
                request_configs=request_configs,
            )
        ]
        return cast(Any, backends[0]).publish_existing_request(keys, transfer_spec)

    @_lmcache_nvtx_annotate
    def move(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
    ) -> int:
        """
        Perform cross-node move of the KV cache.
        """
        assert self.storage_manager is not None
        if not do_copy and self.supports_explicit_spans() is True:
            raise NotImplementedError(
                "destructive move is unavailable while explicit-span lookup "
                "ownership is enabled"
            )

        num_tokens = self.lookup(
            tokens,
            search_range=[old_position],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[old_position]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=old_position,
        )
        assert None not in memory_objs, "Failed to get memory objects to move"
        logger.debug(
            "Trying to send %d memory objects to %s",
            len(memory_objs),
            new_position,
        )

        # TODO: reduce loops
        token_dim = memory_objs[0].meta.fmt.token_dim()  # type: ignore
        offsets = [m.meta.shape[token_dim] for m in memory_objs]  # type: ignore

        transfer_spec = {
            "target_peer_init_url": new_position[0],
            "offsets": offsets,
        }

        logger.info(self.storage_manager.storage_backends)
        p2p_backend = self.storage_manager.storage_backends["P2PBackend"]

        future = asyncio.run_coroutine_threadsafe(
            p2p_backend.async_batched_submit_put_task(
                keys,
                memory_objs,  # type: ignore
                transfer_spec=transfer_spec,
            ),
            self.storage_manager.loop,
        )

        future.result()

        if not do_copy:
            self.storage_manager.batched_remove(keys, locations=[old_position])

        logger.debug(
            "Moving %d token from %s to %s", num_tokens, old_position, new_position
        )
        return num_tokens

    # TODO(Jiayi): Add layerwise support.
    @_lmcache_nvtx_annotate
    def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> None:
        """
        An async version of lookup + prefetch.

        There are three categories of backends:
        (1) sync lookup + sync retrieval (e.g., cpu)
        (2) sync lookup + async retrieval (e.g., disk)
        (3) async lookup + async retrieval (e.g., p2p)
        """
        if explicit_spans is not None:
            raise NotImplementedError(
                "async lookup/prefetch does not support explicit spans"
            )

        assert self.storage_manager is not None

        keys: list[CacheEngineKey] = []
        cum_chunk_lengths = [0]

        if search_range is None:
            search_range = self.retrieve_locations

        # When layerwise is enabled, store_layer writes per-layer keys
        # (LayerCacheEngineKey, chunk_hash + layer_id). Async lookup must
        # split each chunk key into num_layers per-layer keys so the
        # storage backend hot_cache lookups match the same key type.
        keys_per_chunk = self.num_layers if self.use_layerwise else 1

        # TODO(Jiayi): make token database able to return list.
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            if self.use_layerwise:
                keys.extend(key.split_layers(self.num_layers))
            else:
                keys.append(key)
            cum_chunk_lengths.append(end)

        asyncio.run_coroutine_threadsafe(
            self.storage_manager.async_lookup_and_prefetch(
                lookup_id,
                keys,
                cum_chunk_lengths,
                search_range,
                pin,
                keys_per_chunk=keys_per_chunk,
            ),
            self.storage_manager.loop,
        )

    def cleanup_memory_objs(self, lookup_id: str) -> None:
        """
        Cleanup memory objects allocated during prefetch for an aborted lookup.

        Called by the scheduler when it determines that an aborted lookup
        has finished its prefetch tasks.
        """
        try:
            # Get the completed future from event_manager
            if (
                self.event_manager.get_event_status(EventType.LOADING, lookup_id)
                != EventStatus.DONE
            ):
                logger.debug(
                    "No completed event found for lookup_id=%s to clean up.", lookup_id
                )
                return
            future = self.event_manager.pop_event(EventType.LOADING, lookup_id)

            # Get memory objects from the future result
            memory_objs = future.result()
            # Flatten nested lists (each backend returns a list of chunks)
            memory_objs_flat = [mm for m in memory_objs for mm in m]

            # Release each memory object
            for key, memory_obj in memory_objs_flat:
                try:
                    logger.debug("Releasing memory object for lookup_id=%s", lookup_id)
                    if memory_obj.is_pinned:
                        memory_obj.unpin()
                    memory_obj.ref_count_down()
                except Exception as e:
                    logger.error("Error releasing memory object: %s", e)
        except Exception as e:
            logger.error(
                "Error during cleanup_memory_objs for lookup_id=%s: %s",
                lookup_id,
                e,
            )

    # TODO(Jiayi): Need to handle the case where `tokens=None`.
    # In this case, we compress all tokens.
    # TODO(Jiayi): support other compression methods.
    @_lmcache_nvtx_annotate
    def compress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning("Unsupported compression method: %s.", method)
            return 0
        if self.supports_explicit_spans() is True:
            raise NotImplementedError(
                "compression is unavailable while explicit-span lookup "
                "ownership is enabled"
            )

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        serializer, _ = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )
        assert None not in memory_objs, (
            "LMCacheEngine.compress: Failed to get memory objects to compress"
        )

        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            if memory_obj.is_pinned:
                memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def decompress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning("Unsupported decompression method: %s.", method)
            return 0
        if self.supports_explicit_spans() is True:
            raise NotImplementedError(
                "decompression is unavailable while explicit-span lookup "
                "ownership is enabled"
            )

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        _, deserializer = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("there are no tokens to decompress.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )

        assert None not in compressed_memory_objs, (
            "LMCacheEngine.compress: Failed to get compressed "
            "memory objects to decompress"
        )

        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            if compressed_memory_obj.is_pinned:
                compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def lookup_unpin(self, lookup_id: str) -> None:
        with self._lookup_pins_guard():
            registered, pending = self._lookup_cleanup_state()
            if self._lookup_cleanup_closed:
                raise RuntimeError("lookup cleanup owner is closed")
            try:
                self._lookup_unpin_once(lookup_id)
            except Exception:
                # Exact control RPCs still see the failure and cannot ACK,
                # while the engine becomes a durable retry owner for a
                # transferred UUID the scheduler will never cancel again.
                registered.add(lookup_id)
                pending.add(lookup_id)
                self._arm_lookup_cleanup_retry_locked()
                raise
            else:
                registered.discard(lookup_id)
                pending.discard(lookup_id)

    def _lookup_unpin_once(self, lookup_id: str) -> None:
        with self._lookup_pins_guard():
            generations, _ = self._lookup_generation_state()
            generation = generations.get(lookup_id)
            if lookup_id in self.lookup_pins:
                assert self.storage_manager is not None
                pins_by_location = self.lookup_pins[lookup_id]
                if generation is not None:
                    # Establish exact replay suppression, or trip the global
                    # explicit-lookup fence, before releasing physical pins.
                    self._check_lookup_tombstone_capacity(lookup_id, generation)
                first_error: Optional[Exception] = None
                for location, keys in list(pins_by_location.items()):
                    failed_keys = []
                    # ``StorageManager.batched_unpin`` can apply a prefix before
                    # a later key fails. One-key calls make every successful
                    # ownership transition durable and retry only failures.
                    for key in list(keys):
                        try:
                            self.storage_manager.batched_unpin([key], [location])
                        except Exception as error:
                            failed_keys.append(key)
                            if first_error is None:
                                first_error = error
                    if failed_keys:
                        pins_by_location[location] = failed_keys
                    else:
                        pins_by_location.pop(location, None)

                if not pins_by_location:
                    if generation is not None:
                        self._remember_lookup_cancelled_generation(
                            lookup_id,
                            generation,
                        )
                    self.lookup_pins.pop(lookup_id, None)
                    generations.pop(lookup_id, None)
                if first_error is not None:
                    raise first_error

            elif generation is not None:
                # Recover a durable owner whose pin map was already consumed
                # (for example by worker-finally before server replacement).
                self._remember_lookup_cancelled_generation(lookup_id, generation)
                generations.pop(lookup_id, None)

            elif (
                self.async_loading is not None
                and self.event_manager.get_event_status(EventType.LOADING, lookup_id)
                != EventStatus.NOT_FOUND
            ):
                self.cleanup_memory_objs(lookup_id)

    @_lmcache_nvtx_annotate
    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        # TODO: need to clear by request_configs
        if self.save_only_first_rank:
            if self.metadata.is_first_rank():
                num_removed = self._clear(tokens, locations, request_configs)
                return num_removed
            else:
                return 0
        return self._clear(tokens, locations, request_configs)

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.kv_events_enabled and (events := self.kv_events):
            self.kv_events = []
            return events
        return []

    def _clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        assert self.storage_manager is not None
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations, force=False)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations, force=False)
            num_removed += removed
        return num_removed

    @_lmcache_nvtx_annotate
    def health(
        self,
    ) -> int:
        """
        Check the health of the cache engine.
        return: 0 if healthy, otherwise the error code
        """
        assert self.storage_manager is not None
        return 0 if self.storage_manager.memcheck() else -1

    def close(self) -> None:
        """Close every child exactly once and retain incomplete progress."""
        logger.info("Closing LMCacheEngine...")
        storage_manager_init_owner = getattr(
            self,
            "_storage_manager_init_owner",
            None,
        )
        if (
            storage_manager_init_owner is not None
            and storage_manager_init_owner.running_in_current_thread()
        ):
            raise RuntimeError(
                "StorageManager construction cannot reenter engine close"
            )
        with LMCacheEngine._close_owner_init_lock:
            owner = getattr(self, "_close_owner", None)
            if owner is None:
                owner = _EngineCloseOwner(self)
                self._close_owner = owner

        owner.run()
        if not owner.complete:
            raise RuntimeError("cache engine terminal cleanup returned incomplete")
        logger.info("LMCacheEngine closed.")

    def _retain_usage_context(self, context: UsageContext) -> None:
        """Publish telemetry ownership before its reporting thread starts."""
        self._usage_context = context

    def close_failure_is_retry_safe(self, error: BaseException) -> bool:
        """Return true because ``close`` retains every child-stage receipt."""
        del error
        return True

    def _async_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """
        This function is used to get the memory objects from the event manager.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert "req_id" in kwargs, "req_id is required for async loading"
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        tot_kv_size = 0
        chunks: List[ProcessedChunk] = []
        future = self.event_manager.get_event_future(
            EventType.LOADING, kwargs["req_id"]
        )
        # As mentioned in async_lookup_and_prefetch(), the future.result()
        # is key data pair for each chunk in each tier. So extract the key
        # and memory object pairs to memory_obj_map
        try:
            keyed_memory_objs = future.result()
            memory_obj_map: dict[CacheEngineKey, MemoryObj] = {}
        except Exception as e:
            logger.error("Error popping event for request %s: %s", kwargs["req_id"], e)
            return [], 0

        for backend_results in keyed_memory_objs:
            for key, memory_obj in backend_results:
                memory_obj_map[key] = memory_obj

        # TODO(Jiayi): hashing inside `process_tokens` can be skipped.
        used_keys: set[CacheEngineKey] = set()
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            memory_obj = memory_obj_map.get(key)
            if memory_obj is None:
                # returned chunks are expected to be contiguous.
                # break at the first missing chunk.
                break
            chunks.append((key, memory_obj, start, end))
            tot_kv_size += memory_obj.get_size()
            ret_mask[start:end] = True
            used_keys.add(key)

        # NOTE: free the memory objects that are not hit.
        for key, mem_obj in memory_obj_map.items():
            if key not in used_keys:
                mem_obj.ref_count_down()

        return chunks, tot_kv_size

    def _process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """Process tokens and populate the reordered lists.

        This function is used to process tokens and populate the reordered lists.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert self.storage_manager is not None

        tot_kv_size = 0
        reordered_chunks: List[ProcessedChunk] = []
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        chunk_infos = []
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            chunk_infos.append((key, start, end))

        # block_mapping: location -> [(CacheEngineKey, start, end)]
        if (
            "req_id" in kwargs
            and kwargs["req_id"] in self.lookup_pins
            and len(self.lookup_pins[kwargs["req_id"]]) == 1
        ):
            location = next(iter(self.lookup_pins[kwargs["req_id"]].keys()))
            block_mapping = {location: chunk_infos}
        else:
            block_mapping = self.storage_manager.get_block_mapping(chunk_infos)

        get_options: dict[str, Any] = {}
        if kwargs.get("storage_pd_read_context") is not None:
            get_options["storage_pd_read_context"] = kwargs["storage_pd_read_context"]

        last_failed_block_start = None
        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=location,
                **get_options,
            )

            used_keys: set[CacheEngineKey] = set()
            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "The cache block is in the storage, but it can't be retrieved"
                    )
                    if (
                        last_failed_block_start is None
                        # The minimum value should be taken here to ensure that
                        # the prefix keys are all consecutive successful.
                        or last_failed_block_start > start
                    ):
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()
                ret_mask[start:end] = True
                used_keys.add(key)

            for (key, _, _), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is not None and key not in used_keys:
                    logger.debug(
                        "ref_count_down for %s of %s as the previous key failed",
                        key,
                        location,
                    )
                    memory_obj.ref_count_down()

        if last_failed_block_start is not None:
            ret_mask[last_failed_block_start:] = False

            kept_chunks: List[ProcessedChunk] = []
            for key, memory_obj, start, end in reordered_chunks:
                if end <= last_failed_block_start:
                    kept_chunks.append((key, memory_obj, start, end))
                else:
                    tot_kv_size -= memory_obj.get_size()
                    # This chunk will not be used. If the engine is configured
                    # to remove-after-retrieve, the caller would normally call
                    # remove (which frees the block), but since we are dropping
                    # these chunks here, we must free them ourselves to avoid
                    # leaking PD buffer pool memory.
                    if self.remove_after_retrieve:
                        assert self.storage_manager is not None
                        self.storage_manager.remove(key, self.retrieve_locations)
                        # Sync PDBackend.remove() does NOT call ref_count_down()
                        # internally (unlike async PD and other backends), so we
                        # must call it manually. See pd_backend.py line 605.
                        if self._is_sync_pd_backend():
                            memory_obj.ref_count_down()
                    else:
                        memory_obj.ref_count_down()
            reordered_chunks = kept_chunks
        return reordered_chunks, tot_kv_size

    def _broadcast_or_receive_memory_objs(
        self,
        reordered_chunks,
        ret_mask,
    ):
        """
        Handles broadcasting or receiving memory objects in a distributed environment.

        This function implements the communication logic where:
        - The first rank (coordinator) broadcasts memory objects and metadata to others
        - Other ranks receive and reconstruct the memory objects

        Parameters:
        reordered_chunks: List of tuples containing [key, memory object, start, end]
        ret_mask: Boolean mask indicating which positions have been processed

        Side Effects:
        - On first rank:
          * Broadcasts chunk count and each chunk's combined metadata
          * Broadcasts tensor data
        - On other ranks:
          * Receives chunk data and populates reordered_chunks
          * Updates ret_mask to mark received positions as True
        """
        if self.metadata.is_first_rank():
            # Broadcast total chunk count
            chunk_count = len(reordered_chunks)
            self.broadcast_object_fn(chunk_count, self.metadata.first_rank)

            # Reset the GPU-resident copy list. We populate it during this
            # broadcast loop so the caller's subsequent batched_to_gpu can
            # read from HBM instead of re-reading the same CPU L1 buffer
            # over PCIe. (Declared in __init__; reassigning to a fresh list
            # rather than .clear() to drop any references the caller's
            # finally block missed if a previous retrieve raised.)
            self._leader_gpu_substitute_objs = []

            # Broadcast each chunk's data
            for key, memory_obj, start, end in reordered_chunks:
                # Combine (start, end) and metadata into single broadcast
                metadata_dict = memory_obj.metadata.to_dict()
                combined_metadata = (start, end, metadata_dict)
                self.broadcast_object_fn(combined_metadata, self.metadata.first_rank)

                # Broadcast tensor data
                raw_tensor = memory_obj.raw_tensor
                assert raw_tensor is not None
                tensor_to_broadcast = raw_tensor.to(
                    f"{torch_device_type}:{self.metadata.worker_id}"
                )
                self.broadcast_fn(tensor_to_broadcast, self.metadata.first_rank)

                # Keep this GPU-resident copy alive so the subsequent
                # batched_to_gpu can read from HBM rather than re-reading
                # the L1 buffer over PCIe.
                gpu_mo = TensorMemoryObj(
                    raw_data=tensor_to_broadcast,
                    metadata=memory_obj.metadata,
                    parent_allocator=None,
                )
                self._leader_gpu_substitute_objs.append(gpu_mo)
        else:
            # Receive total chunk count
            chunk_count = self.broadcast_object_fn(None, self.metadata.first_rank)
            if chunk_count is None:
                logger.warning(
                    "rank=%d received None chunk_count", self.metadata.worker_id
                )
                return

            # Fill reordered_chunks with received data
            for _ in range(chunk_count):
                # Receive combined metadata (start, end, metadata_dict)
                combined_metadata = self.broadcast_object_fn(
                    None, self.metadata.first_rank
                )
                if combined_metadata is None:
                    logger.warning(
                        "rank=%d received None combined_metadata",
                        self.metadata.worker_id,
                    )
                    break
                start, end, metadata_dict = combined_metadata
                ret_mask[start:end] = True

                # Create tensor and receive data
                metadata = MemoryObjMetadata.from_dict(metadata_dict)
                local_rank = self.metadata.worker_id % torch_dev.device_count()
                raw_tensor = torch.empty(
                    torch.Size([metadata.get_size()]),
                    dtype=torch.uint8,
                    device=f"{torch_device_type}:{local_rank}",
                )
                self.broadcast_fn(raw_tensor, self.metadata.first_rank)

                # Create temporary memory object (key not needed for other ranks)
                memory_obj = TensorMemoryObj(
                    raw_data=raw_tensor, metadata=metadata, parent_allocator=None
                )
                reordered_chunks.append((None, memory_obj, start, end))

    def _require_explicit_spans_supported(
        self, explicit_spans: Optional[ExplicitTokenSpans]
    ) -> None:
        if explicit_spans is None:
            return
        if not isinstance(explicit_spans, ExplicitTokenSpans):
            raise TypeError("explicit_spans must be ExplicitTokenSpans")
        if not self.supports_explicit_spans():
            raise NotImplementedError(
                "explicit spans require synchronous layerwise CacheBlend"
            )

    def _process_layerwise_tokens(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor],
        request_configs: Optional[dict],
        explicit_spans: Optional[ExplicitTokenSpans],
    ) -> Iterable[ProcessTokensResult]:
        if explicit_spans is None:
            return self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
            )
        assert isinstance(self.token_database, SegmentTokenDatabase)
        return self.token_database.process_explicit_spans(
            tokens=tokens,
            explicit_spans=explicit_spans,
            mask=mask,
            request_configs=request_configs,
        )

    def _lookup_explicit_spans(
        self,
        chunk_info_iterator: Iterable[ProcessTokensResult],
        search_range: Optional[List[str]],
        pin: bool,
        lookup_id: Optional[str],
    ) -> int:
        """Look up independent spans while retaining pins for complete hits."""
        assert self.storage_manager is not None
        if pin and lookup_id is None:
            raise ValueError("lookup_id is required when pin is True")

        if pin:
            # Provisional acquisition, cleanup, and durable ledger publication
            # are one ownership transaction. The lookup server already holds
            # this engine RLock, while direct callers acquire it here.
            with self._lookup_pins_guard():
                return self._lookup_explicit_spans_transaction(
                    chunk_info_iterator,
                    search_range,
                    True,
                    lookup_id,
                )
        return self._lookup_explicit_spans_transaction(
            chunk_info_iterator,
            search_range,
            False,
            lookup_id,
        )

    def _forget_explicit_lookup_pin(
        self,
        lookup_id: str,
        location: str,
        key: CacheEngineKey,
    ) -> None:
        pins_by_location = self.lookup_pins.get(lookup_id)
        if pins_by_location is None or location not in pins_by_location:
            raise RuntimeError("explicit lookup pin ledger is inconsistent")
        try:
            pins_by_location[location].remove(key)
        except ValueError as error:
            raise RuntimeError("explicit lookup pin ledger is inconsistent") from error
        if not pins_by_location[location]:
            pins_by_location.pop(location, None)
        if not pins_by_location:
            self.lookup_pins.pop(lookup_id, None)

    def _release_explicit_lookup_pins(
        self,
        lookup_id: str,
        pins_to_release: dict[str, list[CacheEngineKey]],
    ) -> Optional[BaseException]:
        """Release one key at a time and retain every unconfirmed owner."""
        assert self.storage_manager is not None
        first_error: Optional[BaseException] = None
        for location, pinned_keys in pins_to_release.items():
            for key in list(pinned_keys):
                try:
                    self.storage_manager.batched_unpin([key], [location])
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                else:
                    self._forget_explicit_lookup_pin(
                        lookup_id,
                        location,
                        key,
                    )
        return first_error

    def _acquire_explicit_lookup_pins(
        self,
        keys: list[CacheEngineKey],
        search_range: Optional[List[str]],
        lookup_id: str,
    ) -> tuple[int, dict[str, list[CacheEngineKey]]]:
        """Acquire pins with durable progress between every backend key."""
        assert self.storage_manager is not None
        progress_iterator = getattr(self.storage_manager, "iter_lookup_pins", None)
        if progress_iterator is None:
            raise NotImplementedError(
                "explicit-span pinned lookup requires progress-safe storage"
            )

        block_mapping: dict[str, list[CacheEngineKey]] = defaultdict(list)
        try:
            for location, key in progress_iterator(keys, search_range):
                # Record local cleanup progress first, then publish the same
                # physical owner in the engine ledger before asking storage to
                # acquire another key.
                block_mapping[location].append(key)
                self.lookup_pins[lookup_id][location].append(key)
        except BaseException:
            cleanup_error = self._release_explicit_lookup_pins(
                lookup_id,
                block_mapping,
            )
            if cleanup_error is not None:
                logger.error(
                    "Explicit lookup acquisition rollback retained failed pins: %s",
                    cleanup_error,
                )
            raise
        return sum(len(pinned_keys) for pinned_keys in block_mapping.values()), dict(
            block_mapping
        )

    def _lookup_explicit_spans_transaction(
        self,
        chunk_info_iterator: Iterable[ProcessTokensResult],
        search_range: Optional[List[str]],
        pin: bool,
        lookup_id: Optional[str],
    ) -> int:
        """Execute explicit lookup with all acquired pins in the ledger."""
        assert self.storage_manager is not None

        result = 0
        hit_location: Optional[str] = None
        pending_pins: dict[str, list[CacheEngineKey]] = defaultdict(list)
        try:
            for _, end, key in chunk_info_iterator:
                assert isinstance(key, CacheEngineKey)
                keys: list[CacheEngineKey] = (
                    [layer_key for layer_key in key.split_layers(self.num_layers)]
                    if self.use_layerwise
                    else [key]
                )
                if pin:
                    assert lookup_id is not None
                    hit_chunks, block_mapping = self._acquire_explicit_lookup_pins(
                        keys,
                        search_range,
                        lookup_id,
                    )
                else:
                    hit_chunks, block_mapping = self.storage_manager.batched_contains(
                        keys,
                        search_range,
                        False,
                    )
                if hit_chunks == len(keys) and len(block_mapping) > 1:
                    if pin:
                        assert lookup_id is not None
                        cleanup_error = self._release_explicit_lookup_pins(
                            lookup_id,
                            block_mapping,
                        )
                        if cleanup_error is not None:
                            raise cleanup_error
                    raise NotImplementedError(
                        "explicit layerwise spans require all layers in one "
                        "storage location"
                    )
                complete_hit = hit_chunks == len(keys)
                if self.use_layerwise:
                    complete_hit = complete_hit and len(block_mapping) == 1

                if not complete_hit:
                    if pin:
                        assert lookup_id is not None
                        cleanup_error = self._release_explicit_lookup_pins(
                            lookup_id,
                            block_mapping,
                        )
                        if cleanup_error is not None:
                            raise cleanup_error
                    continue

                current_location = next(iter(block_mapping))
                if hit_location is None:
                    hit_location = current_location
                elif self.use_layerwise and hit_location != current_location:
                    if pin:
                        assert lookup_id is not None
                        cleanup_error = self._release_explicit_lookup_pins(
                            lookup_id,
                            block_mapping,
                        )
                        if cleanup_error is not None:
                            raise cleanup_error
                    raise NotImplementedError(
                        "explicit layerwise spans require all hits in one "
                        "storage location"
                    )

                result = max(result, end)
                if pin:
                    for location, pinned_keys in block_mapping.items():
                        pending_pins[location].extend(pinned_keys)

            return result
        except BaseException:
            if pin and pending_pins:
                assert lookup_id is not None
                cleanup_error = self._release_explicit_lookup_pins(
                    lookup_id,
                    pending_pins,
                )
                if cleanup_error is not None:
                    logger.error(
                        "Explicit lookup rollback retained failed pins: %s",
                        cleanup_error,
                    )
            raise

    def _is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()

    def _is_sync_pd_backend(self) -> bool:
        """Check if the PD backend is the sync variant.

        :return: True when PD is enabled and ``pd_backend_mode`` is ``"sync"``.
        :rtype: bool
        """
        return (
            self.config.enable_pd
            and not self.config.pd_uses_shared_storage
            and self.config.pd_backend_mode == "sync"
        )

    def _get_slot_mapping_list(
        self,
        slot_mapping: Optional[Union[torch.Tensor, List[int]]],
    ) -> Optional[List[int]]:
        """
        Convert slot_mapping to list if it's a tensor, otherwise return as is.

        :param slot_mapping: The slot_mapping to convert,
            can be a torch.Tensor or List[int], or None
        :type slot_mapping: Optional[Union[torch.Tensor, List[int]]]
        :return: The slot_mapping as a List[int], or None if input is None
        :rtype: Optional[List[int]]
        """
        if slot_mapping is None:
            return None
        if isinstance(slot_mapping, torch.Tensor):
            return slot_mapping.tolist()
        # At this point, slot_mapping must be List[int]
        return slot_mapping

    def _log_kvcache_for_check(
        self,
        operation: str,
        kwargs: dict,
        token_count: int,
        require_req_id: bool = False,
    ) -> None:
        """
        Helper method to log KVCache Check information.

        This method centralizes the KVCache Check logging logic that was
        duplicated in multiple methods.

        Args:
            operation: The operation being performed (e.g., "Store", "retrieve")
            kwargs: The keyword arguments containing slot_mapping and req_id
            token_count: The number of tokens involved in the operation
            require_req_id: Whether req_id must be present (default: False)
        """
        if not self.kvcache_check_log_enabled:
            return

        slot_mapping = kwargs.get("slot_mapping")
        if slot_mapping is None:
            return

        if require_req_id:
            req_id = kwargs.get("req_id")
            if req_id is None:
                return
        else:
            req_id = kwargs.get("req_id", "unspecified")

        # Convert slot_mapping to list if it's a tensor
        slot_mapping_list = self._get_slot_mapping_list(slot_mapping)
        # slot_mapping_list should not be None when slot_mapping is not None
        assert slot_mapping_list is not None

        logger.info(
            "[KVCache Check] %s request %s, tokens=%d, slot_mapping: %s",
            operation,
            req_id,
            token_count,
            compress_slot_mapping(slot_mapping_list),
        )


class EngineDestroyResult(Enum):
    """Truthful result of one Builder destruction request."""

    DESTROYED = auto()
    ALREADY_ABSENT = auto()


class _BuilderGeneration:
    """One atomically published, internally consistent Builder generation."""

    __slots__ = ("config", "engine", "metadata", "stat_logger")

    def __init__(
        self,
        engine: LMCacheEngine,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        stat_logger: LMCacheStatsLogger,
    ) -> None:
        self.engine = engine
        self.config = config
        self.metadata = metadata
        self.stat_logger = stat_logger


class _BuilderCreationCancelled(RuntimeError):
    """Terminal Builder intent cancelled READY publication."""


class _BuilderRecoveryExpectationUnset:
    """Sentinel type for an omitted Builder recovery expectation."""


_BUILDER_RECOVERY_UNSET = _BuilderRecoveryExpectationUnset()


class _BuilderCreateOwner:
    """Retain every child from construction through one READY publication.

    ``LMCacheStatsLogger`` starts a thread in ``__init__`` and
    ``LMCacheEngine`` can acquire terminal children in its own constructor.
    Allocate and publish each raw object into this owner before invoking its
    initializer.  A constructor-side ``BaseException`` therefore still leaves
    a reachable object that the retained cleanup calls can inspect and drain.
    """

    def __init__(
        self,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> None:
        self.instance_id = instance_id
        self.config = config
        self.metadata = metadata
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        self.stat_logger: Optional[LMCacheStatsLogger] = None
        self.engine: Optional[LMCacheEngine] = None
        self.generation: Optional[_BuilderGeneration] = None
        self._created_pin_monitor: Optional[PinMonitor] = None
        self._state_lock = threading.RLock()
        self._terminal_cancelled = False
        self._publish_started = False

        self._construct_call = RetainedCall(
            self._construct,
            thread_name=f"lmcache-create-{instance_id}",
        )
        self._publish_call = self._new_publish_call()
        self._cleanup_calls = (
            RetainedCall(
                self._cleanup_engine,
                retry_safe_failure=True,
                thread_name=f"lmcache-create-clean-engine-{instance_id}",
            ),
            RetainedCall(
                self._cleanup_stat_logger,
                retry_safe_failure=True,
                thread_name=f"lmcache-create-clean-stats-{instance_id}",
            ),
            RetainedCall(
                self._cleanup_created_pin_monitor,
                retry_safe_failure=True,
                thread_name=f"lmcache-create-clean-pin-monitor-{instance_id}",
            ),
            RetainedCall(
                self._drop_blender,
                retry_safe_failure=True,
                thread_name=f"lmcache-create-clean-blender-{instance_id}",
            ),
            RetainedCall(
                self._unregister_model,
                retry_safe_failure=True,
                thread_name=f"lmcache-create-clean-model-{instance_id}",
            ),
            RetainedCall(
                self._rollback_publication,
                retry_safe_failure=True,
                thread_name=f"lmcache-create-rollback-{instance_id}",
            ),
        )

    def _new_publish_call(self) -> RetainedCall:
        return RetainedCall(
            self._publish_ready,
            thread_name=f"lmcache-create-publish-{self.instance_id}",
        )

    @property
    def construct_complete(self) -> bool:
        return self._construct_call.complete

    @property
    def construction_error(self) -> Optional[BaseException]:
        if self._construct_call.complete:
            return None
        return self._construct_call.last_error

    @property
    def publication_error(self) -> Optional[BaseException]:
        with self._state_lock:
            publish_call = self._publish_call
        if publish_call.complete:
            return None
        return publish_call.last_error

    @property
    def terminal_cancelled(self) -> bool:
        with self._state_lock:
            return self._terminal_cancelled

    @property
    def cleanup_complete(self) -> bool:
        return all(call.complete for call in self._cleanup_calls)

    def matches(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> bool:
        return self.config == config and self.metadata == metadata

    def cancel_publication(self) -> None:
        """Permanently convert this owner from create to cleanup-only."""
        with self._state_lock:
            self._terminal_cancelled = True

    @staticmethod
    def _prepare_partial_engine(engine: LMCacheEngine) -> None:
        """Install the minimum state needed by partial-construction close."""
        engine.gpu_connector = None
        engine.hidden_state_store = None
        engine.lmcache_worker = None
        engine.storage_manager = None
        engine._storage_manager_init_owner = None
        engine._usage_context = None
        engine.lookup_cleanup_registered = set()
        engine.lookup_cleanup_pending = set()
        engine._lookup_cleanup_retry_timer = None
        engine._lookup_cleanup_closed = False
        engine._lookup_pins_lock = threading.RLock()
        LMCacheEngine._ensure_store_cleanup_state(engine)

    def _construct(self) -> None:
        numa_mapping = NUMADetector.get_numa_mapping(self.config)
        logger.info(
            "NUMA mapping for instance %s: %s",
            self.instance_id,
            numa_mapping,
        )
        token_database = LMCacheEngineBuilder._Create_token_database(
            self.config,
            self.metadata,
        )

        # Publish the raw object before __init__: that initializer starts its
        # worker thread, so publishing only its return value is too late.
        stat_logger = LMCacheStatsLogger.__new__(LMCacheStatsLogger)
        self.stat_logger = stat_logger
        LMCacheStatsLogger.__init__(
            stat_logger,
            self.metadata,
            log_interval=10,
            config=self.config,
        )

        # The same rule applies to an engine constructor that may acquire a
        # controller worker or hidden-state store before returning.
        engine = LMCacheEngine.__new__(LMCacheEngine)
        self.engine = engine
        self._prepare_partial_engine(engine)
        # PinMonitor.GetOrCreate starts a registered periodic thread before it
        # publishes the singleton pointer. Capture both authorities so even a
        # constructor-side interruption in that narrow gap remains owned.
        # First Party
        from lmcache.v1.periodic_thread import PeriodicThreadRegistry

        pin_monitor_before = PinMonitor._instance
        pin_registry = PeriodicThreadRegistry.get_instance()
        registered_pin_before = pin_registry.get("PinMonitor-thread")
        try:
            LMCacheEngine.__init__(
                engine,
                self.config,
                self.metadata,
                token_database,
                self.gpu_connector,
                self.broadcast_fn,
                self.broadcast_object_fn,
            )
        finally:
            if pin_monitor_before is None:
                current_pin = PinMonitor._instance
                registered_pin = pin_registry.get("PinMonitor-thread")
                if isinstance(current_pin, PinMonitor):
                    self._created_pin_monitor = current_pin
                elif registered_pin_before is None and isinstance(
                    registered_pin, PinMonitor
                ):
                    self._created_pin_monitor = registered_pin
        self.generation = _BuilderGeneration(
            engine,
            self.config,
            self.metadata,
            stat_logger,
        )

    def _publish_ready(self) -> None:
        generation = self.generation
        if generation is None:
            raise RuntimeError("Builder construction has no complete generation")

        # These dictionaries remain compatibility/introspection mirrors.  The
        # final single assignment is the sole operational READY publication;
        # getters ignore every mirror while this creation owner is installed.
        with LMCacheEngineBuilder._destroy_lock:
            if LMCacheEngineBuilder._creation_owners.get(self.instance_id) is not self:
                raise RuntimeError("Builder creation owner changed before publication")
            LMCacheEngineBuilder._instances[self.instance_id] = generation.engine
            LMCacheEngineBuilder._cfgs[self.instance_id] = generation.config
            LMCacheEngineBuilder._metadatas[self.instance_id] = generation.metadata
            LMCacheEngineBuilder._stat_loggers[self.instance_id] = (
                generation.stat_logger
            )
            LMCacheEngineBuilder._ready_generations[self.instance_id] = generation

    def _cleanup_engine(self) -> None:
        engine = self.engine
        if engine is not None:
            LMCacheEngine.close(engine)

    def _cleanup_stat_logger(self) -> None:
        stat_logger = self.stat_logger
        if stat_logger is None:
            return
        # Before ``thread`` exists, the initializer has not started any live
        # terminal child.  Once it exists, make even a partially initialized
        # logger compatible with its exact-thread shutdown path.
        if not hasattr(stat_logger, "thread"):
            return
        if not hasattr(stat_logger, "shutdown_event"):
            stat_logger.shutdown_event = threading.Event()
        if not hasattr(stat_logger, "is_running"):
            stat_logger.is_running = True
        LMCacheStatsLogger.shutdown(stat_logger)

    def _cleanup_created_pin_monitor(self) -> None:
        pin_monitor = self._created_pin_monitor
        if pin_monitor is None:
            return
        # First Party
        from lmcache.v1.periodic_thread import PeriodicThreadRegistry

        # Stop and prove the exact thread dead before removing either public
        # owner. Each step is idempotent, so RetainedCall may safely retry an
        # injected interruption after any completed boundary.
        pin_monitor.stop()
        if pin_monitor.thread_alive:
            raise RuntimeError(
                "failed Builder construction retained a live PinMonitor thread"
            )
        PeriodicThreadRegistry.get_instance().unregister_instance(pin_monitor)
        with PinMonitor._instance_lock:
            if PinMonitor._instance is pin_monitor:
                PinMonitor._instance = None

    def _drop_blender(self) -> None:
        # First Party
        from lmcache.v1.compute.blend.utils import LMCBlenderBuilder

        LMCBlenderBuilder.drop(self.instance_id)

    def _unregister_model(self) -> None:
        # First Party
        from lmcache.v1.compute.models.utils import VLLMModelTracker

        VLLMModelTracker.unregister_model(self.instance_id)

    def _rollback_publication(self) -> None:
        generation = self.generation
        if generation is None:
            # A constructor can fail before a complete generation exists, but
            # no publication routine could have run in that state.
            return
        with LMCacheEngineBuilder._destroy_lock:
            if (
                LMCacheEngineBuilder._ready_generations.get(self.instance_id)
                is generation
            ):
                LMCacheEngineBuilder._ready_generations.pop(self.instance_id, None)
            mirrors = (
                (LMCacheEngineBuilder._instances, generation.engine),
                (LMCacheEngineBuilder._cfgs, generation.config),
                (LMCacheEngineBuilder._metadatas, generation.metadata),
                (LMCacheEngineBuilder._stat_loggers, generation.stat_logger),
            )
            for registry, expected in mirrors:
                if registry.get(self.instance_id) is expected:
                    registry.pop(self.instance_id, None)

    def run(self, *, allow_publication_retry: bool = True) -> None:
        self._construct_call.run()
        with self._state_lock:
            publish_call = self._publish_call
            if self._terminal_cancelled:
                if not self._publish_started:
                    raise _BuilderCreationCancelled(
                        f"Instance {self.instance_id} construction was cancelled"
                    )
                # If an attempt crossed the terminal-intent boundary, wait for
                # that exact sticky attempt. Never replace it after cancel.
            elif publish_call.last_error is not None:
                if not allow_publication_retry:
                    raise publish_call.last_error
                publish_call = self._new_publish_call()
                self._publish_call = publish_call
                self._publish_started = True
            else:
                self._publish_started = True
        publish_call.run()

    def cleanup(self) -> None:
        for call in self._cleanup_calls:
            call.run()


class _BuilderDestroyOwner:
    """Strong staged retry owner for one Builder registry generation."""

    def __init__(
        self,
        instance_id: str,
        engine: Optional[LMCacheEngine],
        calls: tuple[tuple[Callable[[], None], RetryPolicy], ...],
    ) -> None:
        self.instance_id = instance_id
        self.engine = engine
        self._calls = tuple(
            RetainedCall(
                call,
                retry_safe_failure=retry_safe_failure,
                thread_name=f"lmcache-destroy-{instance_id}-{index}",
            )
            for index, (call, retry_safe_failure) in enumerate(calls)
        )

    @property
    def complete(self) -> bool:
        return all(call.complete for call in self._calls)

    def run(self) -> None:
        for call in self._calls:
            call.run()


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}
    _ready_generations: Dict[str, _BuilderGeneration] = {}
    _creation_owners: Dict[str, _BuilderCreateOwner] = {}
    _destroy_intents: set[str] = set()
    _destroy_owners: Dict[str, _BuilderDestroyOwner] = {}
    _destroyed_engines: weakref.WeakKeyDictionary[LMCacheEngine, bool] = (
        weakref.WeakKeyDictionary()
    )
    _destroy_lock = threading.RLock()

    # TODO(Jiayi): Please remove this helper function in the future.
    # Currently, it's only used for testing.
    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> MemoryAllocatorInterface:
        # NOTE: should remove this function after fixing the unit tests:
        # raise RuntimeError("_Create_memory_allocator is deprecated!")
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )

        if enable_nixl_storage:
            # TODO(Jiayi): weird to import from transfer utils.
            # First Party
            from lmcache.v1.transfer_channel.transfer_utils import (
                get_correct_device,
            )

            corrected_device = get_correct_device(
                config.nixl_buffer_device,
                metadata.worker_id,
            )

            buffer = torch.empty(
                config.nixl_buffer_size,
                dtype=torch.uint8,
                device=corrected_device,
            )

            if corrected_device == "cpu":
                if not current_device_spec.pin_memory(
                    buffer.data_ptr(), config.nixl_buffer_size
                ):
                    raise RuntimeError("Failed to pin NIXL CPU buffer for DMA access")
            else:
                logger.info("Setting device to %s", corrected_device)
                torch_dev.set_device(corrected_device)

            return PagedTensorMemoryAllocator(
                buffer,
                [torch.Size(metadata.kv_shape)],
                [metadata.kv_dtype],
                MemoryFormat.KV_2LTD,
            )

        if config.gds_path is not None:
            assert config.gds_buffer_size is not None
            return CuFileMemoryAllocator(config.gds_buffer_size * 1024**2)

        max_local_cpu_size = config.max_local_cpu_size
        # save_only_first_rank only works when use mla
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank and metadata.is_first_rank():
            # Only the first rank will save the cache,
            # so we need to set it lager than other ranks
            first_rank_max_local_cpu_size = (
                config.extra_config.get(
                    "first_rank_max_local_cpu_size", max_local_cpu_size
                )
                if config.extra_config
                else max_local_cpu_size
            )
            return MixedMemoryAllocator(
                int(first_rank_max_local_cpu_size * 1024**3),
                numa_mapping=numa_mapping,
            )
        return MixedMemoryAllocator(
            int(max_local_cpu_size * 1024**3),
            numa_mapping=numa_mapping,
        )

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> TokenDatabase:
        if config.enable_blending:
            return SegmentTokenDatabase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def _ready_generation_locked(
        cls,
        instance_id: str,
    ) -> Optional[_BuilderGeneration]:
        """Return the sole READY authority, importing complete legacy state.

        The four historical dictionaries remain public test/introspection
        surfaces.  A complete legacy generation is imported once; a partial
        one is rejected.  A stale READY record whose compatibility mirrors
        were deliberately removed by old cleanup code is forgotten instead of
        retaining a closed engine forever.
        """
        generation = cls._ready_generations.get(instance_id)
        mirror_values = (
            cls._instances.get(instance_id),
            cls._cfgs.get(instance_id),
            cls._metadatas.get(instance_id),
            cls._stat_loggers.get(instance_id),
        )
        if generation is not None:
            expected = (
                generation.engine,
                generation.config,
                generation.metadata,
                generation.stat_logger,
            )
            if all(
                value is wanted
                for value, wanted in zip(mirror_values, expected, strict=True)
            ):
                return generation
            if all(value is None for value in mirror_values):
                cls._ready_generations.pop(instance_id, None)
                return None
            raise RuntimeError(
                f"Instance {instance_id} READY record diverged from its mirrors"
            )

        if all(value is None for value in mirror_values):
            return None
        names = ("engine", "config", "metadata", "stats logger")
        if any(value is None for value in mirror_values):
            missing = ", ".join(
                name
                for name, value in zip(names, mirror_values, strict=True)
                if value is None
            )
            raise RuntimeError(
                f"Instance {instance_id} is missing Builder registry: {missing}"
            )
        engine, config, metadata, stat_logger = mirror_values
        generation = _BuilderGeneration(
            cast(LMCacheEngine, engine),
            cast(LMCacheEngineConfig, config),
            cast(LMCacheMetadata, metadata),
            cast(LMCacheStatsLogger, stat_logger),
        )
        cls._ready_generations[instance_id] = generation
        return generation

    @classmethod
    def _clean_failed_creation(
        cls,
        instance_id: str,
        owner: _BuilderCreateOwner,
        construction_error: BaseException,
    ) -> None:
        """Drain a failed constructor without losing its exact owner."""
        try:
            owner.cleanup()
        except BaseException as cleanup_error:
            raise cleanup_error from construction_error
        if not owner.cleanup_complete:
            raise RuntimeError("Builder construction cleanup returned incomplete")
        with cls._destroy_lock:
            if cls._creation_owners.get(instance_id) is owner:
                cls._creation_owners.pop(instance_id)

    @classmethod
    def _finish_creation(
        cls,
        instance_id: str,
        owner: _BuilderCreateOwner,
        *,
        clean_constructor_failure: bool = True,
        terminal_intent: bool = False,
    ) -> _BuilderGeneration:
        """Resume one creation owner through its atomic READY boundary."""
        try:
            owner.run()
        except BaseException:
            construction_error = owner.construction_error
            if construction_error is not None and clean_constructor_failure:
                cls._clean_failed_creation(
                    instance_id,
                    owner,
                    construction_error,
                )
            raise

        generation = owner.generation
        if generation is None:
            raise RuntimeError("Builder construction completed without a generation")
        with cls._destroy_lock:
            if cls._ready_generations.get(instance_id) is not generation:
                raise RuntimeError(
                    "Builder construction returned before READY publication"
                )
            if not terminal_intent and (
                instance_id in cls._destroy_intents
                or instance_id in cls._destroy_owners
            ):
                raise RuntimeError(
                    f"Instance {instance_id} has terminal destruction intent"
                )
            if cls._creation_owners.get(instance_id) is owner:
                cls._creation_owners.pop(instance_id)
        return generation

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info("Creating LMCacheEngine instance %s", instance_id)
        with cls._destroy_lock:
            if (
                instance_id in cls._destroy_intents
                or instance_id in cls._destroy_owners
            ):
                raise RuntimeError(
                    f"Instance {instance_id} still has terminal cleanup ownership"
                )
            owner = cls._creation_owners.get(instance_id)
            if owner is not None:
                if not owner.matches(config, metadata):
                    raise ValueError(
                        f"Instance {instance_id} is being created with a different "
                        "configuration or metadata."
                    )
                creation_owner = owner
            else:
                generation = cls._ready_generation_locked(instance_id)
                if generation is not None:
                    if generation.config != config or generation.metadata != metadata:
                        raise ValueError(
                            f"Instance {instance_id} already exists with a different "
                            f"configuration or metadata."
                        )
                    return generation.engine

                creation_owner = _BuilderCreateOwner(
                    instance_id,
                    config,
                    metadata,
                    gpu_connector,
                    broadcast_fn,
                    broadcast_object_fn,
                )
                # This is the sole fallible-state publication before either
                # child constructor can start a thread or acquire a resource.
                cls._creation_owners[instance_id] = creation_owner

        # Constructors, cleanup callbacks, and their retained worker threads
        # may call back into Builder. Never wait for them while owning the
        # thread-affine RLock.
        return cls._finish_creation(instance_id, creation_owner).engine

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        with cls._destroy_lock:
            if (
                instance_id in cls._creation_owners
                or instance_id in cls._destroy_intents
            ):
                return None
            destroy_owner = cls._destroy_owners.get(instance_id)
            if destroy_owner is not None:
                return None
            generation = cls._ready_generation_locked(instance_id)
            return generation.engine if generation is not None else None

    @classmethod
    def recover(
        cls,
        instance_id: str,
        *,
        expected_config: Union[
            LMCacheEngineConfig,
            _BuilderRecoveryExpectationUnset,
        ] = _BUILDER_RECOVERY_UNSET,
        expected_metadata: Union[
            LMCacheMetadata,
            None,
            _BuilderRecoveryExpectationUnset,
        ] = _BUILDER_RECOVERY_UNSET,
    ) -> Optional[LMCacheEngine]:
        """Recover an interrupted factory call to this exact Builder ID.

        Unlike the intentionally nonblocking :meth:`get`, this method waits
        for a retained CREATING owner and resumes its exact construction and
        READY publication receipts.  Manager uses it only after knowing its
        factory call returned or was interrupted; terminal destruction always
        fails closed instead of being mistaken for a scheduler/no-engine role.
        """
        with cls._destroy_lock:
            if (
                instance_id in cls._destroy_intents
                or instance_id in cls._destroy_owners
            ):
                raise RuntimeError(
                    f"Instance {instance_id} has terminal cleanup ownership"
                )
            owner = cls._creation_owners.get(instance_id)
            if owner is None:
                generation = cls._ready_generation_locked(instance_id)
                if generation is not None:
                    if (
                        not isinstance(
                            expected_config,
                            _BuilderRecoveryExpectationUnset,
                        )
                        and generation.config != expected_config
                    ) or (
                        not isinstance(
                            expected_metadata,
                            _BuilderRecoveryExpectationUnset,
                        )
                        and generation.metadata != expected_metadata
                    ):
                        raise ValueError(
                            f"Instance {instance_id} recovery found a different "
                            "configuration or metadata."
                        )
                return generation.engine if generation is not None else None

            if (
                not isinstance(
                    expected_config,
                    _BuilderRecoveryExpectationUnset,
                )
                and owner.config != expected_config
            ) or (
                not isinstance(
                    expected_metadata,
                    _BuilderRecoveryExpectationUnset,
                )
                and owner.metadata != expected_metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} recovery found a different "
                    "configuration or metadata."
                )

        return cls._finish_creation(instance_id, owner).engine

    @classmethod
    def _prepare_destroy_owner_locked(
        cls,
        instance_id: str,
        expected_engine: Optional[LMCacheEngine],
    ) -> Union[_BuilderDestroyOwner, EngineDestroyResult]:
        """Resolve or publish destruction while holding the admission lock."""
        owner = cls._destroy_owners.get(instance_id)
        expected_is_destroyed = expected_engine is not None and bool(
            cls._destroyed_engines.get(expected_engine, False)
        )
        if owner is not None:
            if expected_engine is not None and owner.engine is not expected_engine:
                # An old manager retry is already satisfied even if a
                # replacement generation currently owns destruction for the
                # same string ID. Never resume the replacement on the old
                # manager's behalf.
                if expected_is_destroyed:
                    return EngineDestroyResult.DESTROYED
                raise RuntimeError(
                    "Builder destroy owner has another engine generation"
                )
            # A matching owner wins over its tombstone. commit_registries
            # publishes the tombstone before its final registry pop, so an
            # interrupted matching owner must resume and be removed rather
            # than returning early and leaking this strong retry owner.
            cls._destroy_intents.discard(instance_id)
            return owner
        if expected_is_destroyed:
            return EngineDestroyResult.DESTROYED

        generation = cls._ready_generation_locked(instance_id)
        engine = generation.engine if generation is not None else None
        if expected_engine is not None and engine is not expected_engine:
            raise RuntimeError("Builder engine generation does not match owner")

        def drop_blender() -> None:
            # First Party
            from lmcache.v1.compute.blend.utils import LMCBlenderBuilder

            LMCBlenderBuilder.drop(instance_id)

        def unregister_model() -> None:
            # First Party
            from lmcache.v1.compute.models.utils import VLLMModelTracker

            VLLMModelTracker.unregister_model(instance_id)

        def commit_registries() -> None:
            # Every prerequisite has its own retained completion receipt. Keep
            # _instances as the public retry owner until every auxiliary
            # registry is gone and install an engine-identity tombstone before
            # the final pop.
            with cls._destroy_lock:
                assert generation is not None
                assert engine is not None
                cls._destroyed_engines[engine] = True
                mirrors = (
                    (cls._cfgs, generation.config),
                    (cls._metadatas, generation.metadata),
                    (cls._stat_loggers, generation.stat_logger),
                    (cls._instances, generation.engine),
                )
                for registry, expected in mirrors:
                    current = registry.get(instance_id)
                    if current is expected:
                        registry.pop(instance_id, None)
                    elif current is not None:
                        raise RuntimeError("Builder registry changed engine generation")
                if cls._ready_generations.get(instance_id) is generation:
                    cls._ready_generations.pop(instance_id, None)

        calls: tuple[tuple[Callable[[], None], RetryPolicy], ...]
        if generation is None:
            # Accepted Builder behavior: a partial setup may publish a
            # model/blender without ever publishing an engine. Fence
            # recreation with this retained owner while both exact idempotent
            # ancillary registries are cleared.
            calls = (
                (drop_blender, True),
                (unregister_model, True),
            )
        else:
            assert engine is not None
            stat_logger = generation.stat_logger
            calls = (
                # LMCacheEngine.close retains a receipt for every terminal
                # child. LMCacheStatsLogger.shutdown only repeats its stop
                # signal and join of the same thread.
                (engine.close, engine.close_failure_is_retry_safe),
                (
                    stat_logger.shutdown,
                    stat_logger.shutdown_failure_is_retry_safe,
                ),
                # These four operations are exact idempotent clears or
                # identity-preserving dictionary pops.
                (LMCStatsMonitor.DestroyInstance, True),
                (drop_blender, True),
                (unregister_model, True),
                (commit_registries, True),
            )
        owner = _BuilderDestroyOwner(instance_id, engine, calls)
        cls._destroy_owners[instance_id] = owner
        cls._destroy_intents.discard(instance_id)
        return owner

    @classmethod
    def destroy(
        cls,
        instance_id: str,
        *,
        expected_engine: Optional[LMCacheEngine] = None,
    ) -> EngineDestroyResult:
        """Close one exact engine generation through a retained staged owner."""
        logger.info("Destroying LMCacheEngine instance: %s", instance_id)
        with cls._destroy_lock:
            existing_owner = cls._destroy_owners.get(instance_id)
            expected_is_destroyed = expected_engine is not None and bool(
                cls._destroyed_engines.get(expected_engine, False)
            )
            creation_owner = cls._creation_owners.get(instance_id)
            if existing_owner is not None or expected_is_destroyed:
                prepared = cls._prepare_destroy_owner_locked(
                    instance_id,
                    expected_engine,
                )
                creation_owner = None
            elif creation_owner is None:
                # Publish the absent/READY destroy owner in the same critical
                # section as admission lookup. A new creation can therefore
                # never slip in before the ancillary cleanup fence exists.
                prepared = cls._prepare_destroy_owner_locked(
                    instance_id,
                    expected_engine,
                )
            else:
                if (
                    expected_engine is not None
                    and creation_owner.engine is not expected_engine
                ):
                    raise RuntimeError(
                        "Builder creation owner has another engine generation"
                    )
                cls._destroy_intents.add(instance_id)
                prepared = None

        if isinstance(prepared, EngineDestroyResult):
            return prepared
        owner = prepared

        if creation_owner is not None:
            creation_owner.cancel_publication()
            try:
                cls._finish_creation(
                    instance_id,
                    creation_owner,
                    clean_constructor_failure=False,
                    terminal_intent=True,
                )
            except BaseException as creation_failure:
                construction_error = creation_owner.construction_error
                publication_error = creation_owner.publication_error
                if (
                    construction_error is None
                    and publication_error is None
                    and not isinstance(
                        creation_failure,
                        _BuilderCreationCancelled,
                    )
                ):
                    # The caller itself was interrupted while a retained
                    # attempt remained in flight. Preserve it for retry.
                    raise
                cleanup_cause = (
                    construction_error or publication_error or creation_failure
                )
                cls._clean_failed_creation(
                    instance_id,
                    creation_owner,
                    cleanup_cause,
                )
                with cls._destroy_lock:
                    cleanup_committed = (
                        cls._creation_owners.get(instance_id) is not creation_owner
                    )
                if not cleanup_committed:
                    raise
                logger.warning(
                    "Instance %s failed construction/publication and was cleaned",
                    instance_id,
                )
                with cls._destroy_lock:
                    cls._destroy_intents.discard(instance_id)
                return EngineDestroyResult.ALREADY_ABSENT

        if creation_owner is not None:
            with cls._destroy_lock:
                prepared = cls._prepare_destroy_owner_locked(
                    instance_id,
                    expected_engine,
                )
            if isinstance(prepared, EngineDestroyResult):
                return prepared
            owner = prepared

        assert owner is not None
        owner.run()
        if not owner.complete:
            raise RuntimeError("Builder terminal cleanup returned incomplete")
        with cls._destroy_lock:
            if cls._destroy_owners.get(instance_id) is owner:
                cls._destroy_intents.discard(instance_id)
                cls._destroy_owners.pop(instance_id)
        logger.info("LMCacheEngine instance %s destroyed", instance_id)
        return (
            EngineDestroyResult.DESTROYED
            if owner.engine is not None
            else EngineDestroyResult.ALREADY_ABSENT
        )
