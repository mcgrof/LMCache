# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
import asyncio
import functools
import threading
import weakref

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.observability import PrometheusLogger
from lmcache.utils import (
    CacheEngineKey,
    _lmcache_nvtx_annotate,
    start_loop_in_thread_with_exceptions,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.lifecycle import RetainedCall
from lmcache.v1.memory_management import (
    AllocationReceipt,
    MemoryFormat,
    MemoryObj,
    RefCountDownReceipt,
    retain_memory_owners,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends, is_cuda_worker
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.raw_block import RawBlockReadContext

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )

logger = init_logger(__name__)

# A failed GPU synchronization does not establish that a copy has stopped.
# Keep the stream alive so its identity cannot be reused, and reject new work
# on it rather than accumulating more allocations after the first failure.
_FAILED_COPY_STREAMS: dict[int, object] = {}
_FAILED_COPY_STREAMS_LOCK = threading.Lock()


def _aggregate_futures(futures: Sequence[Future]) -> Future | None:
    """Return one future that succeeds after every input future succeeds."""
    if not futures:
        return None
    aggregate: Future = Future()
    results: list[Any] = [None] * len(futures)
    remaining = len(futures)
    first_error: BaseException | None = None
    result_lock = threading.Lock()

    def completed(index: int, done: Future) -> None:
        nonlocal first_error, remaining
        error: BaseException | None
        try:
            value = done.result()
        except BaseException as exc:
            value = None
            error = exc
        else:
            error = None
        with result_lock:
            results[index] = value
            if error is not None and first_error is None:
                first_error = error
            remaining -= 1
            if remaining == 0 and not aggregate.done():
                if first_error is not None:
                    aggregate.set_exception(first_error)
                else:
                    aggregate.set_result(results)

    for index, future in enumerate(futures):
        future.add_done_callback(functools.partial(completed, index))
    return aggregate


# Helper function to get the class name of the backend
def get_backend_cname(backend: StorageBackendInterface) -> str:
    return backend.__class__.__name__


# Helper function to allocate and copy memory objects between D and H
def allocate_and_copy_objects(
    allocator_backend: AllocatorBackendInterface,
    keys: Sequence[CacheEngineKey],
    src_memory_objs: list[MemoryObj],
    stream: Any,
    on_allocate: Optional[Callable[[MemoryObj], None]] = None,
    on_copy_start: Optional[Callable[[Any], None]] = None,
    on_copy_synchronized: Optional[Callable[[Any], None]] = None,
) -> tuple[Sequence[CacheEngineKey], list[MemoryObj]]:
    """Allocate and synchronously finish copies into another allocator.

    Args:
        allocator_backend: Allocator receiving the copied objects.
        keys: Cache keys corresponding one-to-one with ``src_memory_objs``.
        src_memory_objs: Source owners that the caller holds through this call.
        stream: GPU copy stream, or ``None`` for blocking copies.
        on_allocate: Callback that adopts each object immediately after
            allocation and before any tensor access, copy, or synchronize.
        on_copy_start: Callback that arms retry ownership immediately before
            the first asynchronous copy is submitted to ``stream``.
        on_copy_synchronized: Callback after all submitted copies have been
            proven complete by a successful stream synchronization.

    Returns:
        Keys and owned destination objects for missing keys that fit in the
        allocator. Allocation exhaustion returns the successfully copied prefix.
        The caller must release the returned destination references.

    Raises:
        ValueError: Key and source lengths differ.
        RuntimeError: Copying or stream synchronization fails, or the stream
            previously failed synchronization.
        OSError: The allocator refuses an allocation.

    A failed enqueue drains earlier copies before releasing their destinations.
    Failed synchronization retains source and destination owners for the process
    lifetime, because neither allocation is safe to recycle. The manager must
    also preserve backend resources when closing a failed copy stream.
    """
    if len(keys) != len(src_memory_objs):
        raise ValueError("keys and src_memory_objs must have equal lengths")
    with _FAILED_COPY_STREAMS_LOCK:
        if id(stream) in _FAILED_COPY_STREAMS:
            raise RuntimeError("Storage copy stream is unusable after failed sync")

    allocated_keys: list[CacheEngineKey] = []
    allocated_objects: list[MemoryObj] = []
    copied_sources: list[MemoryObj] = []
    retained_copy_owner = all(
        callback is not None
        for callback in (on_allocate, on_copy_start, on_copy_synchronized)
    )
    copy_fence_armed = False
    copies_complete = False
    quiescent = stream is None
    try:
        for key, src_memory_obj in zip(keys, src_memory_objs, strict=True):
            if allocator_backend.contains(key):
                continue
            memory_obj = allocator_backend.allocate(
                src_memory_obj.get_shape(),
                src_memory_obj.get_dtype(),
                fmt=src_memory_obj.meta.fmt,
                eviction=True,
                busy_loop=False,
            )
            if memory_obj is None:
                break
            # Own the destination before inspecting or submitting its tensor:
            # both operations can fail after an allocation has succeeded.
            if on_allocate is not None:
                on_allocate(memory_obj)
            allocated_objects.append(memory_obj)
            allocated_keys.append(key)
            copied_sources.append(src_memory_obj)
            destination = memory_obj.tensor
            source = src_memory_obj.tensor
            if destination is None or source is None:
                raise RuntimeError("Storage copy requires tensor memory objects")
            if stream is not None and not copy_fence_armed:
                if on_copy_start is not None:
                    on_copy_start(stream)
                copy_fence_armed = True
            with torch_dev.stream(stream):
                destination.copy_(source, non_blocking=stream is not None)
        copies_complete = True
    finally:
        try:
            if stream is not None:
                stream.synchronize()
                quiescent = True
                if copy_fence_armed and on_copy_synchronized is not None:
                    on_copy_synchronized(stream)
        finally:
            if not quiescent and not retained_copy_owner:
                retain_memory_owners(
                    copied_sources + allocated_objects, allocator_backend, stream
                )
                with _FAILED_COPY_STREAMS_LOCK:
                    _FAILED_COPY_STREAMS[id(stream)] = stream
                # The quarantine owns one reference to each destination;
                # sources still belong to the caller until it unwinds.
                if on_allocate is None:
                    for memory_obj in allocated_objects:
                        memory_obj.ref_count_down()
            elif not copies_complete:
                if on_allocate is None:
                    for memory_obj in allocated_objects:
                        memory_obj.ref_count_down()
    return allocated_keys, allocated_objects


class WeightedSemaphore:
    def __init__(self, chunk_budget: int):
        # it is physically impossible to have more fragmentation than 50%
        # when all of the chunks are of the same size (save_unfull_chunk=False)
        # so we can safely allocate half of the chunk budget for concurrent requests
        self._concurrent_budget_cap = chunk_budget // 2
        self._chunk_budget_cap = chunk_budget
        self._current_chunks = self._concurrent_budget_cap
        self._cond = asyncio.Condition()

    async def acquire(self, n: int = 1) -> None:
        if n > self._chunk_budget_cap:
            raise ValueError(
                f"Trying to acquire {n} chunks, "
                f"Cannot acquire more than {self._chunk_budget_cap} chunks"
                "Please set the max local cpu size to a larger value"
            )

        async with self._cond:
            logger.debug("WeightedSemaphore: Attempting to acquire %d chunks", n)
            if n <= self._concurrent_budget_cap:
                await self._cond.wait_for(lambda: self._current_chunks >= n)
                self._current_chunks -= n
            else:
                # Oversized case: require exclusive access
                await self._cond.wait_for(
                    lambda: self._current_chunks == self._concurrent_budget_cap
                )
                # Reserve everything
                self._current_chunks = 0
            logger.debug(
                "WeightedSemaphore: Acquired %d chunks, remaining chunks: %d",
                n,
                self._current_chunks,
            )

    async def release(self, n: int = 1) -> None:
        async with self._cond:
            if n <= self._concurrent_budget_cap:
                self._current_chunks += n
            else:
                self._current_chunks = self._concurrent_budget_cap
            self._cond.notify_all()


class AsyncMultiSerializer:
    """
    Prevent race conditions where multiple batched_get's cause the local CPU
    backend to allocate memory objects in parallel and get deadlocked.
    Make the assumption that the save_unfull_chunk is False so that we
    can assume that we can always use 50% of the given memory
    """

    def __init__(
        self,
        allocator_backend: AllocatorBackendInterface,
        loop: asyncio.AbstractEventLoop,
    ):
        self.chunk_budget = allocator_backend.calculate_chunk_budget()
        self._sem = WeightedSemaphore(self.chunk_budget)
        self.loop = loop

    async def run(
        self,
        coro_fn: Coroutine[Any, Any, Any],
        num_chunks: int,
    ) -> Any:
        await self._sem.acquire(num_chunks)
        try:
            return await coro_fn
        finally:
            await self._sem.release(num_chunks)


class AsyncSingleSerializer:
    """
    Prevent race conditions in a naive way by forcing each request that
    is passed through to be serialized
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        # lazy init in run
        self.lock: Optional[asyncio.Lock] = None

    async def run(self, coro_fn: Coroutine[Any, Any, Any], *args, **kwargs) -> Any:
        # we need to lazily initialize the lock to
        # place it on the calling event loop
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:  # type: ignore
            return await coro_fn


AsyncSerializer = Union[AsyncSingleSerializer, AsyncMultiSerializer]


class StorageManagerClosedError(RuntimeError):
    """Put admission was refused without consuming caller-owned references."""


class PutOwnershipReceipt:
    """Shared proof of an all-or-nothing batched-put ownership transfer."""

    _NOT_ADOPTED = object()

    def __init__(
        self,
        memory_objs: Sequence[MemoryObj],
        ref_count_down_receipts: Optional[
            Sequence[Optional[RefCountDownReceipt]]
        ] = None,
    ):
        self._memory_obj_ids = tuple(id(memory_obj) for memory_obj in memory_objs)
        self._ref_count_down_receipts: tuple[Optional[RefCountDownReceipt], ...]
        if ref_count_down_receipts is None:
            self._ref_count_down_receipts = tuple(None for _ in memory_objs)
        else:
            if len(ref_count_down_receipts) != len(memory_objs):
                raise ValueError("batched-put release receipt count does not match")
            self._ref_count_down_receipts = tuple(ref_count_down_receipts)
        self._attachment_token: Optional[object] = None
        self._adopted_attachment_token: object = self._NOT_ADOPTED
        self._lock = threading.RLock()

    @property
    def adopted(self) -> bool:
        with self._lock:
            return self._adopted_attachment_token is not self._NOT_ADOPTED

    def owns(self, memory_obj: MemoryObj) -> bool:
        """Return whether StorageManager adopted this exact batch member."""
        with self._lock:
            return (
                self._adopted_attachment_token is not self._NOT_ADOPTED
                and id(memory_obj) in self._memory_obj_ids
            )

    def owns_for(self, attachment_token: object, memory_obj: MemoryObj) -> bool:
        """Return whether one exact outer owner transferred this member."""
        with self._lock:
            return (
                self._adopted_attachment_token is attachment_token
                and id(memory_obj) in self._memory_obj_ids
            )

    def matches(self, memory_objs: Sequence[MemoryObj]) -> bool:
        """Return whether this receipt describes exactly ``memory_objs``."""
        with self._lock:
            return self._memory_obj_ids == tuple(
                id(memory_obj) for memory_obj in memory_objs
            )

    def _adopt(self) -> None:
        """Commit transfer while both caller and manager share this receipt."""
        with self._lock:
            self._adopted_attachment_token = self._attachment_token

    def _attach(self) -> object:
        """Bind this receipt to one outer cleanup owner exactly once."""
        with self._lock:
            if self._attachment_token is not None:
                raise RuntimeError("batched-put ownership receipt was reattached")
            if self._adopted_attachment_token is not self._NOT_ADOPTED:
                raise RuntimeError("adopted batched-put receipt cannot be attached")
            attachment_token = object()
            self._attachment_token = attachment_token
            return attachment_token

    def _release_receipts(self) -> tuple[Optional[RefCountDownReceipt], ...]:
        """Return the exact per-reference receipts transferred on adoption."""
        with self._lock:
            return self._ref_count_down_receipts


class _PutOperationLease:
    """Weakly observed proof that a batched-put call is still live."""


class _PutRefCleanupOwner:
    """Retry owner for references consumed by one batched put."""

    def __init__(
        self,
        memory_objs: Sequence[MemoryObj],
        ownership_receipt: Optional[PutOwnershipReceipt] = None,
        operation_lease: Optional[_PutOperationLease] = None,
    ):
        self._memory_objs = list(memory_objs)
        self._ownership_receipt = ownership_receipt
        self._release_receipts: dict[int, Optional[RefCountDownReceipt]] = {}
        if ownership_receipt is not None:
            self._release_receipts.update(
                enumerate(ownership_receipt._release_receipts())
            )
        # Duck-typed test objects do not implement the production receipt API.
        # Their legacy markers remain test-only; every concrete runtime
        # MemoryObj used by serial_pic_v1 supplies an object-owned receipt.
        self._legacy_released: set[int] = set()
        self._copy_sync_stream: Any = None
        self._copy_sync_pending = False
        self._pending = False
        self._operation_lease_ref = (
            weakref.ref(operation_lease) if operation_lease is not None else None
        )
        self._lock = threading.RLock()

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending

    @property
    def complete(self) -> bool:
        with self._lock:
            if (
                self._ownership_receipt is not None
                and not self._ownership_receipt.adopted
            ):
                return True
            return not self._copy_sync_pending and all(
                self._release_complete(index, memory_obj)
                for index, memory_obj in enumerate(self._memory_objs)
            )

    @property
    def operation_alive(self) -> bool:
        """Return whether the batched-put frame still owns its strong lease."""
        lease_ref = self._operation_lease_ref
        return lease_ref is None or lease_ref() is not None

    def add(self, memory_objs: Sequence[MemoryObj]) -> None:
        """Add newly allocated references before backend submission."""
        with self._lock:
            if self._pending:
                raise RuntimeError("cannot add references to pending put cleanup")
            self._memory_objs.extend(memory_objs)

    def _receipt_for(
        self,
        index: int,
        memory_obj: MemoryObj,
    ) -> Optional[RefCountDownReceipt]:
        if index in self._release_receipts:
            return self._release_receipts[index]
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

    def _release_complete(self, index: int, memory_obj: MemoryObj) -> bool:
        receipt = self._release_receipts.get(index)
        if receipt is None:
            return index in self._legacy_released
        return memory_obj.ref_count_down_receipt_complete(receipt)

    def mark_pending(self) -> None:
        """Publish terminal cleanup with one non-fallible pointer assignment."""
        self._pending = True

    def arm_copy_sync(self, stream: Any) -> None:
        """Fence references before asynchronous copy submission."""
        with self._lock:
            if self._pending:
                raise RuntimeError("cannot arm copy sync on pending put cleanup")
            if self._copy_sync_pending and self._copy_sync_stream is not stream:
                raise RuntimeError("put cleanup already owns another copy stream")
            self._copy_sync_stream = stream
            self._copy_sync_pending = True

    def mark_copy_synchronized(self, stream: Any) -> None:
        """Record proof that every armed copy on ``stream`` completed."""
        with self._lock:
            if not self._copy_sync_pending or self._copy_sync_stream is not stream:
                raise RuntimeError("put cleanup has no matching copy sync fence")
            self._copy_sync_pending = False
            self._copy_sync_stream = None

    def drain(self) -> Optional[BaseException]:
        """Release every outstanding reference, retaining failed entries."""
        first_error: Optional[BaseException] = None
        with self._lock:
            if not self._pending:
                return None
            if (
                self._ownership_receipt is not None
                and not self._ownership_receipt.adopted
            ):
                return None
            if self._copy_sync_pending:
                try:
                    self._copy_sync_stream.synchronize()
                except BaseException as error:
                    return error
                self._copy_sync_pending = False
                self._copy_sync_stream = None
            for index, memory_obj in enumerate(self._memory_objs):
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
        return first_error


class _BackendCloseOwner:
    """Retained completion receipt for one storage-backend close."""

    def __init__(self, manager: "StorageManager", name: str, backend: Any):
        self.backend = backend
        retry_capability = getattr(backend, "close_failure_is_retry_safe", None)

        def retry_safe(error: BaseException) -> bool:
            if retry_capability is None:
                return False
            return bool(retry_capability(error))

        self._call = RetainedCall(
            backend.close,
            retry_safe_failure=retry_safe,
            thread_name=f"lmcache-close-{name}",
            on_worker_start=manager._register_backend_close_worker,
            on_worker_finish=manager._unregister_backend_close_worker,
        )

    @property
    def complete(self) -> bool:
        return self._call.complete

    def close(self) -> None:
        self._call.run()


# TODO: extend this class to implement caching policies and eviction policies
class StorageManager:
    """
    The StorageManager is responsible for managing the storage backends.
    """

    config: LMCacheEngineConfig
    metadata: LMCacheMetadata
    event_manager: EventManager
    lmcache_worker: Optional["LMCacheWorker"]
    async_lookup_server: Optional["LMCacheAsyncLookupServer"]
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    storage_backends: OrderedDict[str, StorageBackendInterface]
    manager_lock: Any
    _put_cleanup_lock: Any
    _put_cleanup_condition: threading.Condition
    _put_cleanup_owners: dict[int, _PutRefCleanupOwner]
    _put_cleanup_closed: bool
    _put_cleanup_teardown_active: bool
    _put_cleanup_teardown_owner: Optional[int]
    _put_cleanup_teardown_token: Optional[object]
    _closed_storage_backend_ids: set[int]
    _storage_backend_close_owners: dict[int, _BackendCloseOwner]
    _backend_close_worker_ids: set[int]
    non_allocator_backends: list[str]
    enable_pd: bool
    allocator_backend: Optional[AllocatorBackendInterface]
    local_cpu_backend: Optional[StorageBackendInterface]
    instance_id: Any
    worker_id: int
    async_serializer: Optional[AsyncSerializer]
    internal_copy_stream: Any
    _freeze: bool
    _freeze_lock: Any
    _bypassed_backends: set[str]
    _bypass_lock: Any

    @staticmethod
    def _prepare_partial_state(
        manager: "StorageManager",
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        event_manager: EventManager,
        lmcache_worker: Optional["LMCacheWorker"] = None,
        async_lookup_server: Optional["LMCacheAsyncLookupServer"] = None,
    ) -> None:
        """Install a close-safe owner before construction starts resources.

        ``LMCacheEngine.post_init`` calls this on a raw, already retained
        manager.  ``__init__`` repeats it before opening the event loop so an
        internal constructor failure at every later stage can be drained by
        :meth:`close` without guessing which attributes exist.
        """
        manager.config = config
        manager.metadata = metadata
        manager.event_manager = event_manager
        manager.lmcache_worker = lmcache_worker
        manager.async_lookup_server = async_lookup_server

        # The public surface requires constructed loop/thread values. Close is
        # the sole partial-state consumer and explicitly handles these sentinels.
        manager.loop = cast(asyncio.AbstractEventLoop, None)
        manager.thread = cast(threading.Thread, None)
        manager.storage_backends = OrderedDict()
        manager.manager_lock = threading.Lock()
        manager._put_cleanup_lock = threading.RLock()
        manager._put_cleanup_condition = threading.Condition(manager._put_cleanup_lock)
        manager._put_cleanup_owners = {}
        manager._put_cleanup_closed = False
        manager._put_cleanup_teardown_active = False
        manager._put_cleanup_teardown_owner = None
        manager._put_cleanup_teardown_token = None
        manager._closed_storage_backend_ids = set()
        manager._storage_backend_close_owners = {}
        manager._backend_close_worker_ids = set()

        manager.non_allocator_backends = []
        manager.enable_pd = False
        manager.allocator_backend = None
        manager.local_cpu_backend = None
        manager.instance_id = getattr(config, "lmcache_instance_id", "")
        manager.worker_id = getattr(metadata, "worker_id", 0)
        manager.async_serializer = None
        manager.internal_copy_stream = None
        manager._freeze = False
        manager._freeze_lock = threading.RLock()
        manager._bypassed_backends = set()
        manager._bypass_lock = threading.RLock()

    @staticmethod
    def _create_and_publish_event_loop(manager: "StorageManager") -> None:
        """Publish a new event loop or close the exact unpublished result.

        ``asyncio.new_event_loop()`` returns a live resource. Keep it local
        while publishing so an interruption before ``STORE_ATTR`` closes that
        exact loop. If the assignment already committed, partial-construction
        cleanup through :meth:`close` remains its sole terminal owner.
        """
        loop = asyncio.new_event_loop()
        try:
            manager.loop = loop
        except BaseException as publication_error:
            if manager.loop is loop:
                raise
            try:
                loop.close()
            except BaseException as cleanup_error:
                # A failed close must not erase the exact loop. Replay the
                # local publication so the retained manager can retry it.
                try:
                    manager.loop = loop
                except BaseException:
                    manager.loop = loop
                raise publication_error from cleanup_error
            raise

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        event_manager: EventManager,
        lmcache_worker: Optional["LMCacheWorker"] = None,
        async_lookup_server: Optional["LMCacheAsyncLookupServer"] = None,
    ) -> None:
        StorageManager._prepare_partial_state(
            self,
            config,
            metadata,
            event_manager,
            lmcache_worker,
            async_lookup_server,
        )
        StorageManager._create_and_publish_event_loop(self)

        self.thread = threading.Thread(
            target=start_loop_in_thread_with_exceptions,
            args=(self.loop,),
            name="storage-manager-event-loop",
        )
        self.internal_copy_stream = None
        self._copy_owners_retained = False

        self.storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()
        self.manager_lock = threading.Lock()
        self.lmcache_worker = lmcache_worker

        initialized = False
        try:
            self.thread.start()
            # Use the unified create path so that init and
            # dynamic creation share the same logic.
            self.create_backends()

            # the backend used for actual storage
            self.non_allocator_backends = self.get_non_allocator_backends()

            self.enable_pd = config.enable_pd

            self.allocator_backend = None
            if metadata.role != "scheduler":
                self.allocator_backend = self._get_allocator_backend(config)

            self.local_cpu_backend = self.storage_backends.get("LocalCPUBackend", None)

            self.instance_id = config.lmcache_instance_id
            self.worker_id = metadata.worker_id

            self.event_manager = event_manager

            self.async_lookup_server: Optional["LMCacheAsyncLookupServer"] = (
                async_lookup_server
            )
            self.async_serializer: Optional[AsyncSerializer] = None

            # The GPU stream for internal copies during put
            if is_cuda_worker(metadata):
                self.internal_copy_stream = torch_dev.Stream()
            else:
                self.internal_copy_stream = None

            # freeze mode: only use local_cpu backend for retrieval
            self._freeze = False
            self._freeze_lock = threading.RLock()

            # Backend bypass mode: skip specific backends during health check failures
            self._bypassed_backends: set[str] = set()
            self._bypass_lock = threading.RLock()

            if not self.enable_pd and self.config.enable_async_loading:
                assert self.allocator_backend is not None
                self.async_serializer = AsyncSingleSerializer(self.loop)

            self._setup_metrics()
            initialized = True
        finally:
            if not initialized:
                self.close()

    def _setup_metrics(self) -> None:
        prometheus_logger = PrometheusLogger.GetOrCreate(
            self.metadata,
            config=self.config,
        )

        metric_map = {
            "storage_events_ongoing_count": EventStatus.ONGOING,
            "storage_events_done_count": EventStatus.DONE,
            "storage_events_not_found_count": EventStatus.NOT_FOUND,
        }

        for metric_name, status in metric_map.items():
            metric = getattr(prometheus_logger, metric_name)
            metric.set_function(
                lambda s=status: self.event_manager.get_events_count_by_status(
                    EventType.LOADING, s
                )
            )

    def _ensure_put_cleanup_state(self) -> None:
        """Initialize retry state for lightweight test and recovery objects."""
        if not hasattr(self, "_put_cleanup_lock"):
            self._put_cleanup_lock = threading.RLock()
        if not hasattr(self, "_put_cleanup_condition"):
            self._put_cleanup_condition = threading.Condition(self._put_cleanup_lock)
        if not hasattr(self, "_put_cleanup_owners"):
            self._put_cleanup_owners = {}
        if not hasattr(self, "_put_cleanup_closed"):
            self._put_cleanup_closed = False
        if not hasattr(self, "_put_cleanup_teardown_active"):
            self._put_cleanup_teardown_active = False
        if not hasattr(self, "_put_cleanup_teardown_owner"):
            self._put_cleanup_teardown_owner = None
        if not hasattr(self, "_put_cleanup_teardown_token"):
            self._put_cleanup_teardown_token = None
        if not hasattr(self, "_closed_storage_backend_ids"):
            self._closed_storage_backend_ids = set()
        if not hasattr(self, "_storage_backend_close_owners"):
            self._storage_backend_close_owners = {}
        if not hasattr(self, "_backend_close_worker_ids"):
            self._backend_close_worker_ids = set()

    def _register_backend_close_worker(self) -> None:
        self._ensure_put_cleanup_state()
        with self._put_cleanup_lock:
            self._backend_close_worker_ids.add(threading.get_ident())

    def _unregister_backend_close_worker(self) -> None:
        self._ensure_put_cleanup_state()
        with self._put_cleanup_lock:
            self._backend_close_worker_ids.discard(threading.get_ident())

    def _register_put_cleanup(
        self,
        owner: _PutRefCleanupOwner,
        ownership_receipt: Optional[PutOwnershipReceipt],
    ) -> bool:
        """Atomically admit a put and publish its reference ownership."""
        self._ensure_put_cleanup_state()
        with self._put_cleanup_lock:
            if self._put_cleanup_closed:
                return False

            if ownership_receipt is None:
                self._put_cleanup_owners[id(owner)] = owner
                return True

            # Serialize receipt adoption with both this manager's admission
            # seal and accidental cross-manager receipt reuse. If an
            # interruption lands after owner publication but before adoption,
            # batched_put's finally block removes the unadopted owner without
            # consuming caller references.
            with ownership_receipt._lock:
                if ownership_receipt.adopted:
                    raise RuntimeError("batched-put ownership receipt was reused")
                self._put_cleanup_owners[id(owner)] = owner
                ownership_receipt._adopt()
                return True

    def _has_put_cleanup_owner(self, owner: _PutRefCleanupOwner) -> bool:
        """Return whether ``owner`` was published in the manager ledger."""
        self._ensure_put_cleanup_state()
        with self._put_cleanup_lock:
            return self._put_cleanup_owners.get(id(owner)) is owner

    def finalize_put_cleanup(self, owner: _PutRefCleanupOwner) -> bool:
        """Atomically publish put cleanup and verify its exact ledger owner.

        Pending is the helper's first state transition under the ledger lock.
        If interruption lands at any later boundary, public drains can recover
        the registered owner. An unregistered rejection is marked only on its
        private local object and remains a no-op.
        """
        self._ensure_put_cleanup_state()
        with self._put_cleanup_lock:
            owner.mark_pending()
            return self._put_cleanup_owners.get(id(owner)) is owner

    def _seal_put_cleanup_admission(self) -> None:
        """Permanently reject new backend I/O before terminal cleanup."""
        self._ensure_put_cleanup_state()
        with self._put_cleanup_lock:
            self._put_cleanup_closed = True

    def _begin_put_cleanup_teardown(self, teardown_token: object) -> None:
        """Seal admission and serialize retryable terminal teardown."""
        self._ensure_put_cleanup_state()
        with self._put_cleanup_condition:
            caller = threading.get_ident()
            while self._put_cleanup_teardown_token is not None:
                if (
                    self._put_cleanup_teardown_owner == caller
                    or caller in self._backend_close_worker_ids
                ):
                    raise RuntimeError("storage manager close is reentrant")
                self._put_cleanup_condition.wait()
            self._put_cleanup_closed = True
            owners = tuple(self._put_cleanup_owners.values())
            active_put_count = sum(
                owner.operation_alive and not owner.pending for owner in owners
            )
            if active_put_count != 0:
                raise RuntimeError(
                    "storage manager retained "
                    f"{active_put_count} active batched-put owner(s)"
                )
            # Admission remains sealed permanently. Rejected puts return
            # immediately and retain their caller-owned references, including
            # puts re-entered synchronously from a backend close callback.
            # The caller supplies the token before entering this method, so
            # its finally block can release teardown even if interruption
            # lands after publication but before this method returns.
            self._put_cleanup_teardown_token = teardown_token
            self._put_cleanup_teardown_owner = caller
            self._put_cleanup_teardown_active = True

    def _finish_put_cleanup_teardown(self, teardown_token: object) -> None:
        """Release the teardown serializer and wake concurrent closers."""
        self._ensure_put_cleanup_state()
        with self._put_cleanup_condition:
            if self._put_cleanup_teardown_token is not teardown_token:
                return
            # Clear the authoritative token first. The remaining fields are
            # diagnostics; waiters use the token and cannot be stranded by an
            # interruption between these assignments.
            try:
                self._put_cleanup_teardown_token = None
                self._put_cleanup_teardown_active = False
                self._put_cleanup_teardown_owner = None
            finally:
                self._put_cleanup_condition.notify_all()

    def _active_put_cleanup_count(self) -> int:
        """Return admitted puts that have not entered cleanup yet."""
        self._ensure_put_cleanup_state()
        with self._put_cleanup_lock:
            owners = tuple(self._put_cleanup_owners.values())
        return sum(owner.operation_alive and not owner.pending for owner in owners)

    def _put_cleanup_owner_count(self) -> int:
        """Return every active or pending batched-put owner."""
        self._ensure_put_cleanup_state()
        with self._put_cleanup_lock:
            return len(self._put_cleanup_owners)

    def _get_allocator_backend(
        self, config: LMCacheEngineConfig
    ) -> AllocatorBackendInterface:
        gpu_endpoints = [
            backend
            for backend in self.storage_backends.values()
            if getattr(backend, "is_gpu_endpoint", False)
        ]
        if self.enable_pd and not self.config.pd_uses_shared_storage:
            allocator_backend = self.storage_backends["PDBackend"]
        elif gpu_endpoints:
            # A backend that stages KV chunks in device memory (the raw-block
            # backend with a GPU pool) allocates for the engine, so the GPU
            # connector copies device-to-device and the backend moves the
            # chunk to storage straight from VRAM.
            allocator_backend = gpu_endpoints[0]
        elif "MaruBackend" in self.storage_backends:
            if "LocalCPUBackend" in self.storage_backends:
                allocator_backend = self.storage_backends["LocalCPUBackend"]
            else:
                allocator_backend = self.storage_backends["MaruBackend"]
        else:
            allocator_backend = self.storage_backends["LocalCPUBackend"]
        assert isinstance(allocator_backend, AllocatorBackendInterface)
        return allocator_backend

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
        allocation_receipt: Optional[AllocationReceipt] = None,
    ) -> Optional[MemoryObj]:
        """
        Allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        assert self.allocator_backend is not None
        if allocation_receipt is not None:
            return self.allocator_backend.allocate(
                shapes,
                dtypes,
                fmt,
                eviction=eviction,
                busy_loop=busy_loop,
                allocation_receipt=allocation_receipt,
            )
        return self.allocator_backend.allocate(
            shapes, dtypes, fmt, eviction=eviction, busy_loop=busy_loop
        )

    def supports_allocation_receipts(self) -> bool:
        """Return whether the active allocation backend is exact-reservation safe."""
        backend = getattr(self, "allocator_backend", None)
        return bool(
            backend is not None
            and getattr(backend, "supports_allocation_receipts", lambda: False)()
        )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
        allocation_receipts: Optional[Sequence[AllocationReceipt]] = None,
    ) -> Optional[list[MemoryObj]]:
        """
        Batched allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        if self.allocator_backend is None:
            raise RuntimeError("Allocator backend not available for scheduler role")
        if allocation_receipts is not None:
            return self.allocator_backend.batched_allocate(
                shapes,
                dtypes,
                batch_size,
                fmt,
                eviction=eviction,
                busy_loop=busy_loop,
                allocation_receipts=allocation_receipts,
            )
        return self.allocator_backend.batched_allocate(
            shapes, dtypes, batch_size, fmt, eviction=eviction, busy_loop=busy_loop
        )

    def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        location: Optional[str] = None,
    ) -> None:
        """
        Non-blocking function to put the memory object into the storages.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """
        raise RuntimeError(
            "StorageManager.put is deprecated and should not be called anymore"
        )

    def batched_put(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
        location: Optional[str] = None,
        ownership_receipt: Optional[PutOwnershipReceipt] = None,
    ) -> Future | None:
        """
        Non-blocking function to batched put the memory objects into the
        storage backends.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).

        An admitted call consumes one caller-owned reference per input object,
        including when later backend work fails. A call rejected after close
        raises :class:`StorageManagerClosedError` without consuming any input
        reference. Callers that have an outer cleanup owner should pass a
        matching :class:`PutOwnershipReceipt` and consult it after any
        interruption to determine which owner committed the transfer.
        Copied staging groups are released only after their copies finish;
        failed synchronization retains owners for the process lifetime.
        """
        # The dictionary from backend cname to objects and keys
        obj_dict: dict[
            str,
            tuple[Sequence[CacheEngineKey], list[MemoryObj]],
        ] = {}
        self._ensure_put_cleanup_state()
        if ownership_receipt is not None and not ownership_receipt.matches(memory_objs):
            raise ValueError(
                "batched-put ownership receipt does not match memory objects"
            )
        operation_lease: Optional[_PutOperationLease] = _PutOperationLease()
        cleanup_owner = _PutRefCleanupOwner(
            memory_objs,
            ownership_receipt,
            operation_lease,
        )

        futures: list[Future] = []
        # Once admission adopts the optional receipt, this call consumes one
        # caller-owned reference for every input object even when later work
        # raises. Before admission, the caller remains the sole owner.
        operation_error: Optional[BaseException] = None
        try:
            io_admitted = self._register_put_cleanup(
                cleanup_owner,
                ownership_receipt,
            )
            if not io_admitted:
                raise StorageManagerClosedError(
                    "storage manager close has begun; caller retains "
                    "batched-put references"
                )
            # Do not admit another put over failed release ownership. This
            # call's owner is still active and is therefore not drained.
            self.drain_pending_put_cleanups()
            if self.allocator_backend is None:
                raise RuntimeError("Batched put not available for scheduler role")
            obj_dict[get_backend_cname(self.allocator_backend)] = (
                keys,
                memory_objs,
            )
            for backend_name, backend in self.storage_backends.items():
                if location and backend_name != location:
                    continue
                # Skip bypassed backends
                with self._bypass_lock:
                    if backend_name in self._bypassed_backends:
                        continue

                allocator_backend = backend.get_allocator_backend()
                cname = get_backend_cname(allocator_backend)
                if cname not in obj_dict:
                    new_keys, new_objs = allocate_and_copy_objects(
                        allocator_backend,
                        keys,
                        memory_objs,
                        self.internal_copy_stream,
                        on_allocate=lambda memory_obj: cleanup_owner.add([memory_obj]),
                        on_copy_start=cleanup_owner.arm_copy_sync,
                        on_copy_synchronized=cleanup_owner.mark_copy_synchronized,
                    )
                    obj_dict[cname] = (new_keys, new_objs)

                # NOTE: the handling of exists_in_put_tasks
                # is done in the backend
                ks, objs = obj_dict[cname]
                submitted = backend.batched_submit_put_task(
                    ks,
                    objs,
                    transfer_spec=transfer_spec,
                )
                if submitted:
                    futures.extend(submitted)
        except BaseException as error:
            self._retain_failed_copy_resources()
            operation_error = error
            raise
        finally:
            # A dead lease lets public drains recover even if interruption
            # lands on the finalization helper's first executable line.
            operation_lease = None
            # Registration can itself be interrupted after owner publication.
            # Consult the ledger instead of a local flag so that exact owner is
            # always finalized. An unadopted receipt makes drain non-consuming.
            if self.finalize_put_cleanup(cleanup_owner):
                cleanup_error = self._drain_put_cleanup_owner(cleanup_owner)
                if cleanup_error is not None:
                    if operation_error is None:
                        raise cleanup_error
                    logger.error(
                        "Batched put reference cleanup also failed after %s: %s",
                        operation_error,
                        cleanup_error,
                    )

        return _aggregate_futures(futures)

    def _drain_put_cleanup_owner(
        self,
        owner: _PutRefCleanupOwner,
    ) -> Optional[BaseException]:
        """Drain one pending owner and forget it only after exact release."""
        error = owner.drain()
        if owner.complete:
            with self._put_cleanup_lock:
                self._put_cleanup_owners.pop(id(owner), None)
        return error

    def drain_pending_put_cleanups(self) -> None:
        """Retry every failed batched-put reference release exhaustively."""
        self._ensure_put_cleanup_state()
        first_error: Optional[BaseException] = None
        with self._put_cleanup_lock:
            owners = tuple(self._put_cleanup_owners.values())
        for owner in owners:
            if not owner.pending:
                if owner.operation_alive:
                    continue
                if not self.finalize_put_cleanup(owner):
                    continue
            error = self._drain_put_cleanup_owner(owner)
            if first_error is None and error is not None:
                first_error = error
        if first_error is not None:
            raise first_error

    def pending_put_cleanup_count(self) -> int:
        """Return the number of retry-owned batched-put cleanup states."""
        self._ensure_put_cleanup_state()
        with self._put_cleanup_lock:
            return sum(
                owner.pending or not owner.operation_alive
                for owner in self._put_cleanup_owners.values()
            )

    def get(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        Blocking function to get the memory object from the storages.

        Returns one caller-owned reference, or None for a miss. If CPU cache
        writeback fails, the loaded reference is released before propagating
        the failure (or retained if GPU copy quiescence is unknown).
        """

        # Search all backends for blocking get
        for backend_name, backend in self.get_active_storage_backends(location):
            # TODO(Jiayi): need to make sure all memory_objs returned
            # are allocated by the allocator backend.
            memory_obj = backend.get_blocking(key)
            if memory_obj:
                delivered = False
                try:
                    if (
                        backend_name
                        not in ["LocalCPUBackend", "PDBackend", "MaruBackend"]
                        and "LocalCPUBackend" in self.storage_backends
                    ):
                        local_cpu_backend = self.storage_backends["LocalCPUBackend"]
                        assert isinstance(local_cpu_backend, LocalCPUBackend)
                        self._write_back_to_local_cpu(
                            local_cpu_backend, [key], [memory_obj]
                        )
                    delivered = True
                    return memory_obj
                finally:
                    if not delivered:
                        memory_obj.ref_count_down()

        return None

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        """
        Non-blocking function to get the memory object from the storages.
        """
        # TODO (Jiayi): incorporate prefetching here

        # Search all backends for non-blocking get
        for backend_name, backend in self.get_active_storage_backends(location):
            # NOTE(Jiayi): bypass the allocator for now
            task = backend.get_non_blocking(key)
            if task:
                # TODO (Jiayi): add write-back logic here
                return task
        return None

    def batched_get(
        self,
        keys: List[CacheEngineKey],
        location: Optional[str] = None,
        *,
        storage_pd_read_context: Optional[RawBlockReadContext] = None,
    ) -> List[Optional[MemoryObj]]:
        """Load objects synchronously from the selected storage backends.

        Args:
            keys: Ordered cache keys to load.
            location: Backend name, or None to search all active backends.
            storage_pd_read_context: Adopted request identity to carry with
                this restore. Publication-aware backends receive it with each
                read; ordinary cache hits keep their existing behavior.

        Returns:
            Caller-owned objects aligned with keys, using None for misses.
            Failed CPU-cache writeback releases the loaded references before
            propagating the error; copies with unknown quiescence retain them.
        """
        # TODO (ApostaC): remove the nested optional here
        for backend_name, storage_backend in self.get_active_storage_backends(location):
            read_publication = getattr(
                storage_backend, "batched_get_for_publication", None
            )
            if storage_pd_read_context is not None and callable(read_publication):
                memory_objs = read_publication(keys, storage_pd_read_context)
            else:
                memory_objs = storage_backend.batched_get_blocking(keys)
            if memory_objs:
                delivered = False
                try:
                    # Align with single-key `get()` logic:
                    # auto-write remote data to local CPU cache
                    if (
                        backend_name
                        not in ["LocalCPUBackend", "PDBackend", "MaruBackend"]
                        and "LocalCPUBackend" in self.storage_backends
                        and None not in memory_objs
                    ):
                        logger.debug(
                            "Storing %s objects from %s to LocalCPUBackend",
                            len(keys),
                            backend_name,
                        )
                        local_cpu_backend = self.storage_backends["LocalCPUBackend"]
                        assert isinstance(local_cpu_backend, LocalCPUBackend)
                        # Type cast: Safe (we verified no Nones above)
                        # `batched_submit_put_task` expects list[MemoryObj]
                        # TODO (lisiG9): Refactor this write-back logic into caching
                        #  policy module
                        memory_objs_no_none = cast(List[MemoryObj], memory_objs)
                        self._write_back_to_local_cpu(
                            local_cpu_backend, keys, memory_objs_no_none
                        )
                    delivered = True
                    return memory_objs
                finally:
                    if not delivered:
                        for memory_obj in memory_objs:
                            if memory_obj is not None:
                                memory_obj.ref_count_down()
        return [None] * len(keys)

    def _write_back_to_local_cpu(
        self,
        local_cpu_backend: LocalCPUBackend,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        """Admit objects loaded from another backend into the local CPU cache.

        Objects that live in device memory (loaded by a GPU-endpoint backend)
        are copied into local CPU objects first; admitting them as they are
        would pin the device staging slots for as long as the CPU cache kept
        them.

        That copy is only worth paying for when the cache will keep the
        result.  With the hot cache off the backend drops every admission, so
        a load served out of device memory would otherwise copy each chunk to
        the host on the way back and then throw it away.
        """
        if not local_cpu_backend.use_hot:
            return
        if all(
            memory_obj.tensor is None or memory_obj.tensor.device.type == "cpu"
            for memory_obj in memory_objs
        ):
            local_cpu_backend.batched_submit_put_task(keys, memory_objs)
            return
        try:
            host_keys, host_objs = allocate_and_copy_objects(
                local_cpu_backend, keys, memory_objs, self.internal_copy_stream
            )
        except RuntimeError:
            self._retain_failed_copy_resources()
            raise
        try:
            if host_objs:
                local_cpu_backend.batched_submit_put_task(host_keys, host_objs)
        finally:
            for memory_obj in host_objs:
                memory_obj.ref_count_down()

    def layerwise_batched_get(
        self,
        keys: List[List[CacheEngineKey]],
        location: Optional[str] = None,
    ) -> Generator[Future, None, None]:
        """
        Non-blocking function to get the memory objects into the storages
        in a layerwise manner.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).

        :param List[List[CacheEngineKey]] keys: The keys to get. The first
            dimension corresponds to the number of layers, and the second
            dimension corresponds to the number of chunks.

        :return: A generator that yields a future for each layer.
        """
        if location is None:
            location = "LocalCPUBackend"
        for keys_multi_chunk in keys:
            # Retrieve all chunks for one layer
            backend = self.storage_backends[location]
            # TODO(Jiayi): need to make async loading and layerwise compatible
            coro = backend.batched_get_non_blocking("fake_lookup_id", keys_multi_chunk)
            task = asyncio.run_coroutine_threadsafe(coro, self.loop)
            yield task

    def layerwise_batched_get_returns_pinned(
        self,
        location: Optional[str] = None,
    ) -> bool:
        """Whether layerwise get returns one caller-owned pin per object."""
        if location is None:
            location = "LocalCPUBackend"
        backend = self.storage_backends[location]
        return backend.batched_get_non_blocking_returns_pinned()

    def prefetch_single_done_callback(
        self,
        future: asyncio.Future,
        keys: list[CacheEngineKey],
        backend_name: str,
    ) -> None:
        """
        Callback function when a single prefetch task
        (i.e., prefetching from a single backend) is done.
        """
        # TODO(Jiayi): support write-back policy here
        pass

    def prefetch_all_done_callback(
        self,
        task: asyncio.Future,
        lookup_id: str,
        cum_chunk_lengths_total: list[int],
        tier_expected_chunks: list[int],
        keys_per_chunk: int = 1,
    ) -> None:
        """
        Callback function when all prefetch tasks
        (i.e., prefetching from all backends for the entire request) are done.

        :param int keys_per_chunk: Number of storage keys per logical chunk
            (``num_layers`` for layerwise mode, ``1`` otherwise). Used to
            convert each tier's per-key result count back to chunk units.
        """
        assert self.async_lookup_server is not None
        self.event_manager.update_event_status(
            EventType.LOADING, lookup_id, status=EventStatus.DONE
        )
        res = task.result()

        # Calculate total retrieved chunks across all tiers based on actual results
        # from batched_get_non_blocking, not the batched_async_contains results.
        # This handles the case where chunks may be evicted between contains check
        # and actual retrieval.
        #
        # Example: chunk_size=256, 7 chunks total (1792 tokens) across 3 tiers
        #   cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280, 1536, 1792]
        #   tier_expected_chunks = [3, 2, 2]  # Tier 0: 3, Tier 1: 2, Tier 2: 2
        #
        #   Chunks:
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- stored in Tier0, tier_expected_chunks[0]==3
        #         |---|      <--- stored in Tier1, tier_expected_chunks[1]==2
        #             |---|  <--- stored in Tier2, tier_expected_chunks[2]==2
        #
        # Case 1: All chunks retrieved successfully
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- Tier0: retrieved 3 chunks (obj0, obj1, obj2)
        #         |---|      <--- Tier1: retrieved 2 chunks (obj3, obj4)
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6)
        #   res = [[obj0, obj1, obj2], [obj3, obj4], [obj5, obj6]]
        #   total_retrieved_chunks = 7
        #   retrieved_length = cum_chunk_lengths_total[7] = 1792
        #
        # Case 2: Tier 1 only got 1 chunk (eviction), Tier 2 got all 2 chunks
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- Tier0: retrieved 3 chunks (obj0, obj1, obj2)
        #         |-|X|      <--- Tier1: retrieved 1 chunk (obj3), missing obj4
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6) - IGNORED
        #   res = [[obj0, obj1, obj2], [obj3], [obj5, obj6]]
        #   total_retrieved_chunks = 4 (stop at tier 1, tier 2 chunks ignored)
        #   retrieved_length = cum_chunk_lengths_total[4] = 1024
        #   Note: Even though tier 2 successfully retrieved 2 chunks, they are
        #   not counted because tier 1 has a gap, breaking prefix continuity.
        #
        # Case 3: Tier 0 only got 2 chunks (eviction), other tiers got all
        #   [0 1 2 3 4 5 6]
        #   |---|X|          <--- Tier0: retrieved 2 chunks (obj0, obj1), missing obj2
        #         |---|      <--- Tier1: retrieved 2 chunks (obj3, obj4) - IGNORED
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6) - IGNORED
        #   res = [[obj0, obj1], [obj3, obj4], [obj5, obj6]]
        #   total_retrieved_chunks = 2 (stop at tier 0, all subsequent ignored)
        #   retrieved_length = cum_chunk_lengths_total[2] = 512
        total_retrieved_chunks = 0
        for tier_idx, tier_result in enumerate(res):
            # `tier_result` is a list of (key, mem_obj) pairs, one per
            # storage key. With layerwise on, each logical chunk maps to
            # keys_per_chunk per-layer keys, so divide to get chunk count.
            # We round down so a partially-retrieved chunk (e.g., one
            # layer evicted) is treated as a miss.
            actual_chunks = len(tier_result) // keys_per_chunk
            expected_chunks = tier_expected_chunks[tier_idx]
            total_retrieved_chunks += actual_chunks

            # Release the tail rounded off by actual_chunks; else staging buffer leaks.
            tail_start = actual_chunks * keys_per_chunk
            for _, mem_obj in tier_result[tail_start:]:
                mem_obj.ref_count_down()

            # If a tier retrieved fewer chunks than expected, we stop counting
            # because subsequent chunks are not contiguous
            if actual_chunks < expected_chunks:
                # Release all chunks in subsequent tiers since they won't be used
                for subsequent_tier in res[tier_idx + 1 :]:
                    for _, mem_obj in subsequent_tier:
                        mem_obj.ref_count_down()
                break

        retrieved_length = cum_chunk_lengths_total[total_retrieved_chunks]
        logger.info(
            "Responding to scheduler for lookup id %s with retrieved length %s",
            lookup_id,
            retrieved_length,
        )
        self.async_lookup_server.send_response_to_scheduler(lookup_id, retrieved_length)

    async def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        cum_chunk_lengths: list[int],
        search_range: Optional[list[str]] = None,
        pin: bool = False,
        keys_per_chunk: int = 1,
    ) -> None:
        """
        Perform asynchronous lookup and prefetching across all storage backends.

        :param str lookup_id: The unique id (e.g., request id) for the request.
        :param list[CacheEngineKey] keys: The keys to lookup and prefetch.
        :param list[int] cum_chunk_lengths: The cumulative token lengths of the chunks.
            This is a list where cum_chunk_lengths[i] represents the total number of
            tokens from chunk 0 to chunk i-1 (inclusive).
            Example: If chunk_size=256 and we have 3 chunks:
                - chunk 0: 256 tokens (tokens 0-255)
                - chunk 1: 256 tokens (tokens 256-511)
                - chunk 2: 128 tokens (tokens 512-639)
            Then cum_chunk_lengths = [0, 256, 512, 640]
            Note: len(cum_chunk_lengths) = (len(keys) // keys_per_chunk) + 1
        :param Optional[list[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now. If None, search in all backends.
        :param bool pin: Whether to pin the keys.
        :param int keys_per_chunk: Number of storage keys per logical chunk.
            For non-layerwise mode this is 1 (one key per chunk). For
            layerwise mode the caller passes num_layers, since each chunk
            is stored as num_layers per-layer keys (LayerCacheEngineKey).
            All chunk-level accounting below is derived from
            num_hit_keys // keys_per_chunk so that hit counts and
            cum_chunk_lengths indexing stay in chunk units even though the
            backend operates on per-layer keys.
        """

        # NOTE(Jiayi): Currently, the retrieval pattern is always
        # prefix-based. That is, we retrieve 0-t1 tokens from backend 1
        # and retrieve t1-t2 tokens from backend 2, etc. The assumption
        # here is that the suffix chunks are more likely to be evicted
        # than the prefix chunks.
        # TODO(Jiayi): We need to change/optimize this for non-prefix
        # based retrieval patterns or cases where middle chunks are missing.

        # NOTE(Jiayi): We can tolerate the last tier to have fewer loaded
        # chunks than its lookup result indicated. This is especially helpful
        # for P2PBackend.

        # All chunk-level accounting (num_total_chunks, num_total_hit_chunks,
        # tier_expected_chunks, cum_chunk_lengths indexing) is in *chunk*
        # units, while raw key indexing into `keys` is in *key* units.
        # When keys_per_chunk == 1 these are identical; when layerwise is
        # on, each chunk corresponds to num_layers keys.
        if keys_per_chunk < 1:
            raise ValueError(f"keys_per_chunk must be >= 1, got {keys_per_chunk}")
        if len(keys) % keys_per_chunk != 0:
            raise ValueError(
                f"len(keys)={len(keys)} is not a multiple of "
                f"keys_per_chunk={keys_per_chunk}"
            )
        num_total_chunks = len(keys) // keys_per_chunk
        num_total_hit_chunks = 0
        # cum_chunk_lengths_total: A copy of the original cumulative chunk lengths
        # for all chunks. This is preserved to calculate the final token count
        # based on the actual retrieved chunks.
        # Example: If chunk_size=256 and we have 3 chunks with total 640 tokens:
        #     cum_chunk_lengths_total = [0, 256, 512, 640]
        # If we retrieve 2 chunks, the retrieved token count is:
        #     cum_chunk_lengths_total[2] = 512 tokens
        cum_chunk_lengths_total = cum_chunk_lengths[:]
        loading_tasks = []
        tier_expected_chunks = []
        # we also keep track of the keys for each tier and each chunk
        loading_task_keys: list[list[CacheEngineKey]] = []
        for backend_name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            num_hit_keys_raw = await backend.batched_async_contains(
                lookup_id, keys, pin
            )
            # Round down to a whole-chunk boundary. If a backend has only
            # some of a chunk's per-layer keys (e.g., partial eviction),
            # we treat the chunk as a miss to keep the chunk-level
            # invariant required by the prefix-match retrieval pattern.
            num_hit_chunks = num_hit_keys_raw // keys_per_chunk
            num_hit_keys = num_hit_chunks * keys_per_chunk

            # `pin=True` pins every matching key inside batched_async_contains.
            # The rounded-off tail (keys[num_hit_keys:num_hit_keys_raw]) is
            # dropped from backend_keys and never retrieved, so release those
            # pins here to avoid leaking refcount budget.
            if pin and num_hit_keys < num_hit_keys_raw:
                for k in keys[num_hit_keys:num_hit_keys_raw]:
                    backend.unpin(k)

            if num_hit_chunks == 0:
                continue

            num_total_hit_chunks += num_hit_chunks
            tier_expected_chunks.append(num_hit_chunks)

            backend_keys = keys[:num_hit_keys]
            loading_task_keys.append(backend_keys)

            assert self.async_serializer is not None, (
                "Async serializer must be initialized via post_init before using "
                "async_lookup_and_prefetch."
            )
            # num_hit_chunks is only used for the multi serializer
            get_coro = self.async_serializer.run(
                backend.batched_get_non_blocking(
                    lookup_id,
                    backend_keys,
                    {"cum_chunk_lengths": cum_chunk_lengths[: num_hit_chunks + 1]},
                ),
                num_hit_chunks,
            )
            loading_task = asyncio.create_task(get_coro)
            loading_task.add_done_callback(
                functools.partial(
                    self.prefetch_single_done_callback,
                    keys=keys,
                    backend_name=backend_name,
                )
            )

            loading_tasks.append(loading_task)

            cum_chunk_lengths = cum_chunk_lengths[num_hit_chunks:]

            if num_total_hit_chunks == num_total_chunks:
                break
            keys = keys[num_hit_keys:]

        # If no chunks were hit across all backends, respond immediately and return.
        if num_total_hit_chunks == 0:
            if self.async_lookup_server is not None:
                self.async_lookup_server.send_response_to_scheduler(lookup_id, 0)
            return

        # gather_with_keys() here make a pair of (key, memory_obj) for each chunk
        # in each tier. The all_done result's layout is like following and
        # will be processed in _async_process_tokens_internal()
        # Tier 0:
        #  Tuple(loading_task_keys[0][0] : MemoryObj0)
        #  Tuple(loading_task_keys[0][1] : MemoryObj1)
        # Tier 1:
        #  Tuple(loading_task_keys[1][0] : MemoryObj2)
        #  Tuple(loading_task_keys[1][1] : MemoryObj3)
        async def gather_with_keys() -> list[list[tuple[CacheEngineKey, MemoryObj]]]:
            loading_results = await asyncio.gather(*loading_tasks)
            return [
                list(zip(keys, results, strict=False))
                for keys, results in zip(
                    loading_task_keys, loading_results, strict=False
                )
            ]

        all_done = asyncio.create_task(gather_with_keys())
        # Register the event before adding the callback to avoid race conditions
        self.event_manager.add_event(
            EventType.LOADING,
            lookup_id,
            all_done,
        )

        all_done.add_done_callback(
            lambda future: self.prefetch_all_done_callback(
                future,
                lookup_id,
                cum_chunk_lengths_total,
                tier_expected_chunks,
                keys_per_chunk=keys_per_chunk,
            )
        )

    def set_hot_cache(self, enabled: bool) -> None:
        """
        Dynamically enable or disable the hot cache on LocalCPUBackend.

        When disabled, the existing hot cache entries will be cleared
        and no new data will be written to the hot cache.

        Args:
            enabled: True to enable hot cache, False to disable
        """
        backend = self.local_cpu_backend
        if not isinstance(backend, LocalCPUBackend):
            logger.warning("Cannot set hot_cache: LocalCPUBackend not available")
            return

        if not enabled:
            backend.clear()
        backend.use_hot = enabled
        logger.info("LocalCPUBackend hot_cache set to %s", enabled)

    def is_hot_cache_enabled(self) -> bool:
        """
        Get the current hot cache status of LocalCPUBackend.

        Returns:
            bool: True if hot cache is enabled, False otherwise
        """
        backend = self.local_cpu_backend
        if not isinstance(backend, LocalCPUBackend):
            return False
        return backend.use_hot

    def set_freeze(self, enabled: bool) -> None:
        """
        Set freeze mode.

        When enabled, only local_cpu backend will be used for retrieval.
        """
        with self._freeze_lock:
            self._freeze = enabled
        logger.info("StorageManager freeze mode set to %s", enabled)

    def is_frozen(self) -> bool:
        """
        Get freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        with self._freeze_lock:
            return self._freeze

    def set_backend_bypass(self, backend_name: str, bypassed: bool) -> None:
        """
        Set bypass mode for a specific backend.

        When a backend is bypassed:
        - It will be skipped during contains/put/get operations
        - This is typically used when a health check fails with LOCAL_CPU fallback

        Args:
            backend_name: The name of the backend to bypass (e.g., "RemoteBackend")
            bypassed: True to bypass, False to restore normal operation
        """
        with self._bypass_lock:
            if bypassed:
                self._bypassed_backends.add(backend_name)
                logger.info("StorageManager: Backend %s is now bypassed", backend_name)
            else:
                self._bypassed_backends.discard(backend_name)
                logger.info(
                    "StorageManager: Backend %s bypass removed, "
                    "restored to normal operation",
                    backend_name,
                )

    def is_backend_bypassed(self, backend_name: str) -> bool:
        """
        Check if a backend is currently bypassed.

        Args:
            backend_name: The name of the backend to check

        Returns:
            bool: True if the backend is bypassed, False otherwise
        """
        with self._bypass_lock:
            return backend_name in self._bypassed_backends

    def get_bypassed_backends(self) -> List[str]:
        """
        Get the list of currently bypassed backend names.

        Returns:
            List[str]: List of bypassed backend names
        """
        with self._bypass_lock:
            return list(self._bypassed_backends)

    def get_all_backend_names(self) -> List[str]:
        """
        Get the list of all registered backend names.

        Returns:
            List[str]: List of all backend names
        """
        return list(self.storage_backends.keys())

    def contains(
        self,
        key: CacheEngineKey,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> Optional[str]:
        """
        Check whether the key exists in the storage backend.

        :param CacheEngineKey key: The key to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: True if the key exists in the specified storage backends.
        """

        for backend_name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            # NOTE(Jiayi): We do not pin for PDBackend
            pin_in_backend = pin if backend_name != "PDBackend" else False

            if backend.contains(key, pin_in_backend):
                return backend_name

        return None

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> tuple[int, dict]:
        """
        Check whether the key exists in the storage backend.

        :param List[CacheEngineKey] keys: The keys to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: Return hit chunks and block mapping by prefix match.
        """
        total_keys = len(keys)
        total_hit_chunks = 0
        block_mapping = {}
        for backend_name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            # NOTE(Jiayi): We do not pin for PDBackend
            pin_in_backend = pin if backend_name != "PDBackend" else False

            hit_chunks = backend.batched_contains(keys, pin_in_backend)
            if hit_chunks == 0:
                continue
            block_mapping[backend_name] = keys[:hit_chunks]
            total_hit_chunks += hit_chunks
            if total_hit_chunks == total_keys:
                break
            keys = keys[hit_chunks:]

        return total_hit_chunks, block_mapping

    def iter_lookup_pins(
        self,
        keys: List[CacheEngineKey],
        search_range: Optional[List[str]] = None,
    ) -> Generator[Tuple[str, CacheEngineKey], None, None]:
        """Yield progress after each exception-atomic lookup pin.

        The full backend set is validated before the first ownership change.
        Explicit-span pinning therefore fails closed when any selected backend
        cannot guarantee that an exception leaves the current key unpinned.
        """
        active_backends = list(
            self.get_active_storage_backends(search_range=search_range)
        )
        unsupported = [
            backend_name
            for backend_name, backend in active_backends
            if not backend.supports_lookup_pin_progress()
        ]
        if unsupported:
            raise NotImplementedError(
                "explicit-span pinned lookup requires progress-safe backends; "
                f"unsupported active backends: {unsupported}"
            )

        remaining = list(keys)
        for backend_name, backend in active_backends:
            hit_count = 0
            for key in remaining:
                if not backend.pin_for_lookup(key):
                    break
                hit_count += 1
                # Control returns to the engine after every physical pin, so
                # it can ledger this key before acquisition proceeds.
                yield backend_name, key
            if hit_count == len(remaining):
                return
            if hit_count:
                remaining = remaining[hit_count:]

    def get_block_mapping(
        self, chunk_infos: List[Tuple[CacheEngineKey, int, int]]
    ) -> Dict[str, List[Tuple[CacheEngineKey, int, int]]]:
        """
        Get block mapping for the given chunk infos, works by prefix match.

        :param List[Tuple[CacheEngineKey, int, int]] chunk_infos:
        List of chunk infos, each tuple contains (key, begin, end)

        :return: Dict[str, List[Tuple[CacheEngineKey, int, int]]]:
        Block mapping for the given chunk infos, each key is the backend name,
        each value is a list of chunk infos in the backend.
        """
        keys = [chunk_info[0] for chunk_info in chunk_infos]
        total_keys = len(keys)
        block_mapping = {}
        total_hit_chunks = 0
        for backend_name, backend in self.get_active_storage_backends():
            hit_chunks = backend.batched_contains(keys)
            if hit_chunks == 0:
                continue
            block_mapping[backend_name] = chunk_infos[
                total_hit_chunks : total_hit_chunks + hit_chunks
            ]
            total_hit_chunks += hit_chunks
            if total_hit_chunks == total_keys:
                break
            keys = keys[hit_chunks:]
        return block_mapping

    def touch_cache(self) -> None:
        for backend_name, backend in self.storage_backends.items():
            if backend_name == "LocalCPUBackend" or backend_name == "LocalDiskBackend":
                backend.touch_cache()

    def remove(
        self,
        key: CacheEngineKey,
        locations: Optional[List[str]] = None,
        force: bool = True,
    ) -> int:
        """
        Remove the key and the corresponding cache in the specified
        locations.

        :param CacheEngineKey key: The key to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """

        num_removed = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or backend_name in locations:
                num_removed += backend.remove(key, force=force)

        return num_removed

    def batched_remove(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
        force: bool = True,
    ) -> int:
        """
        Batched remove the keys and the corresponding cache in the specified
        locations.

        :param List[CacheEngineKey] keys: The keys to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """
        num_removed = 0
        for backend_name, backend in self.storage_backends.items():
            if locations is None or backend_name in locations:
                num_removed += backend.batched_remove(keys, force=force)

        return num_removed

    def batched_unpin(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
    ) -> None:
        """
        Unpin the keys in the specified locations.

        :param List[CacheEngineKey] keys: The keys to unpin.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `unpin` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `unpin` in all backends.
        """
        for backend_name, backend in self.storage_backends.items():
            if locations is None or backend_name in locations:
                for key in keys:
                    if not backend.unpin(key):
                        raise RuntimeError(
                            "lookup pin owner is missing from storage backend "
                            f"{backend_name}: {key}"
                        )

    def clear(
        self,
        locations: Optional[List[str]] = None,
        force: bool = False,
    ) -> int:
        """
        Clear all caches in the specified locations.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `clear` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `clear` in all backends.

        return: Total number of cleared tokens in the specified
        storage backends.
        """

        num_cleared_tokens = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or backend_name in locations:
                if hasattr(backend, "clear"):
                    num_cleared_tokens += backend.clear(force=force)
                else:
                    logger.warning(
                        "Storage backend %s does not support "
                        "clear operation. Skipping.",
                        backend_name,
                    )

        return num_cleared_tokens

    def memcheck(self) -> bool:
        """
        Check the integrity of the underlying storage backend's
        memory allocators

        Returns:
            True if everything is good otherwise False
        """
        for backend in self.storage_backends.values():
            if not isinstance(backend, AllocatorBackendInterface):
                continue
            if not backend.get_memory_allocator().memcheck():
                return False
        return True

    def get_active_storage_backends(
        self,
        location: Optional[str] = None,
        search_range: Optional[List[str]] = None,
    ) -> Generator[Tuple[str, StorageBackendInterface], None, None]:
        """
        Get the active storage backends based on freeze mode, bypass mode, and filters.

        :param Optional[str] location: If specified, only yield backends
            matching this exact name.
        :param Optional[List[str]] search_range: If specified, only yield
            backends whose names are in this list.

        :return: Generator of (backend_name, backend) tuples.
        """
        for backend_name, backend in self.storage_backends.items():
            # In freeze mode, only use local_cpu backend
            with self._freeze_lock:
                if self._freeze and backend_name != "LocalCPUBackend":
                    continue
            # Skip bypassed backends
            with self._bypass_lock:
                if backend_name in self._bypassed_backends:
                    continue
            if location and backend_name != location:
                continue
            if search_range and backend_name not in search_range:
                continue
            yield backend_name, backend

    def get_non_allocator_backends(self) -> List[str]:
        """
        Get the names of the actual storage backends. Some backends,
        such as LocalCPUBackend and PDBackend, in some cases, only
        serve as a backend for allocation.
        """
        storage_names = []
        for backend_name, backend in self.storage_backends.items():
            if "LocalCPUBackend" == backend_name and not self.config.local_cpu:
                # if local_cpu is False, means LocalCPUBackend is only a allocator
                continue
            if "PDBackend" == backend_name and backend.pd_config.role == "sender":  # type: ignore
                # if pd_config.role is sender, means PDBackend is only a allocator
                continue
            storage_names.append(backend_name)
        return storage_names

    def list_backends(self) -> Dict[str, str]:
        """
        List all active storage backends.

        Returns:
            Dict mapping backend name to its class name.
        """
        with self.manager_lock:
            return {
                name: type(backend).__name__
                for name, backend in self.storage_backends.items()
            }

    def close_backend(self, backend_name: str) -> bool:
        """
        Close and remove a specific storage backend by name.

        The backend will be closed and removed from the internal
        dict so that no stale references remain.

        Args:
            backend_name: The name of the backend to close.

        Returns:
            True if the backend was found and closed, False
            otherwise, including when failed copy synchronization prevents
            safe backend teardown.
        """
        if self._retain_failed_copy_resources():
            return False
        with self.manager_lock:
            backend = self.storage_backends.get(backend_name)
            if backend is None:
                logger.warning(
                    "Backend %s not found, cannot close",
                    backend_name,
                )
                return False

            try:
                logger.info("Closing backend: %s", backend_name)
                backend.close()
            except Exception:
                logger.exception("Error closing backend %s", backend_name)

            del self.storage_backends[backend_name]

            # Update derived references
            self.non_allocator_backends = self.get_non_allocator_backends()
            if backend_name == "LocalCPUBackend":
                self.local_cpu_backend = None
            logger.info("Backend %s closed and removed", backend_name)
            return True

    def create_backends(self) -> Dict[str, str]:
        """
        Create new storage backends based on current config.

        Backends that are already present will be skipped
        **before** instantiation so that no unnecessary
        resources are allocated.  This allows callers to close
        a subset of backends, update config via ``/conf``,
        and then call this method to bring up only the missing
        backends.

        Returns:
            Dict mapping newly created backend name to its
            class name.
        """
        with self.manager_lock:
            existing_names = set(self.storage_backends)
            new_backends = CreateStorageBackends(
                self.config,
                self.metadata,
                self.loop,
                dst_device=(
                    torch_device_type if is_cuda_worker(self.metadata) else "cpu"
                ),
                lmcache_worker=self.lmcache_worker,
                skip_backends=existing_names,
                existing_backends=self.storage_backends,
                retain_backend=self._retain_constructing_backend,
            )

            created: Dict[str, str] = {}
            for name, backend in new_backends.items():
                self.storage_backends[name] = backend
                created[name] = type(backend).__name__
                logger.info(
                    "Created backend: %s (%s)",
                    name,
                    created[name],
                )

            # Refresh derived references
            self.non_allocator_backends = self.get_non_allocator_backends()
            cpu = self.storage_backends.get("LocalCPUBackend")
            if cpu is not None:
                self.local_cpu_backend = cpu

            return created

    def _retain_constructing_backend(
        self,
        name: str,
        backend: StorageBackendInterface,
    ) -> None:
        """Publish one raw backend before its constructor starts resources.

        ``create_backends`` owns ``manager_lock`` while this callback runs.
        Keeping the write lock-free here avoids re-entering that non-reentrant
        lock and makes the identity publication the only state transition.
        """
        existing = self.storage_backends.get(name)
        if existing is not None and existing is not backend:
            raise RuntimeError(f"storage backend {name} changed identity")
        self.storage_backends[name] = backend

    def recreate_backend(self, backend_name: str) -> Dict[str, str]:
        """
        Close a backend and recreate it from current config.

        This is an atomic close-then-create operation that
        combines :meth:`close_backend` and :meth:`create_backends`
        into a single step.

        Args:
            backend_name: Name of the backend to recreate
                (e.g. ``RemoteBackend``).

        Returns:
            Dict mapping newly created backend name to its
            class name.

        Raises:
            RuntimeError: Failed copy synchronization prevents safe teardown.
            KeyError: If *backend_name* does not exist.
        """
        if self._retain_failed_copy_resources():
            raise RuntimeError("Cannot recreate backends after failed copy sync")
        with self.manager_lock:
            backend = self.storage_backends.get(backend_name)
            if backend is None:
                raise KeyError("Backend %s not found" % backend_name)

            # --- close ---
            try:
                logger.info("Closing backend: %s", backend_name)
                backend.close()
            except Exception:
                logger.exception("Error closing backend %s", backend_name)
            del self.storage_backends[backend_name]

            # --- create ---
            existing_names = set(self.storage_backends)
            new_backends = CreateStorageBackends(
                self.config,
                self.metadata,
                self.loop,
                dst_device=(
                    torch_device_type if is_cuda_worker(self.metadata) else "cpu"
                ),
                lmcache_worker=self.lmcache_worker,
                skip_backends=existing_names,
                existing_backends=self.storage_backends,
            )

            created: Dict[str, str] = {}
            for name, be in new_backends.items():
                self.storage_backends[name] = be
                created[name] = type(be).__name__
                logger.info(
                    "Recreated backend: %s (%s)",
                    name,
                    created[name],
                )

            # Refresh derived references
            self.non_allocator_backends = self.get_non_allocator_backends()
            cpu = self.storage_backends.get("LocalCPUBackend")
            if cpu is not None:
                self.local_cpu_backend = cpu
            elif backend_name == "LocalCPUBackend":
                self.local_cpu_backend = None

            return created

    def cancel_request(self, req_id: str) -> None:
        """
        Cancel an in-flight or pending request.

        Delegates to all storage backends. Backends that track per-request
        state will cancel the request; others will no-op.

        :param str req_id: The request identifier to cancel.
        :return: None
        """
        for backend in self.storage_backends.values():
            backend.cancel_request(req_id)

    def finish_request(self, req_id: str) -> None:
        """Notify storage backends that no more request batches will arrive.

        Args:
            req_id: Request identifier that finished on the model worker.
        """
        for backend in self.storage_backends.values():
            backend.finish_request(req_id)

    def close(self) -> None:
        """Close backends and stop the worker, including incomplete construction.

        A failed copy synchronization leaves DMA ownership unknown. In that
        case the backend graph and its arenas remain alive until process exit,
        while the manager's event-loop thread is still stopped.
        """
        logger.info("Closing StorageManager...")
        teardown_token = object()
        try:
            self._begin_put_cleanup_teardown(teardown_token)
            self.drain_pending_put_cleanups()
            put_owner_count = self._put_cleanup_owner_count()
            if put_owner_count != 0:
                raise RuntimeError(
                    "storage manager retained "
                    f"{put_owner_count} batched-put cleanup owner(s)"
                )

            first_error: Optional[BaseException] = None

            # A backend that closed successfully is committed and skipped on
            # retry.  Failed backends remain reachable and are retried while
            # every other terminal resource is still attempted exhaustively.
            backends = sorted(
                self.storage_backends.items(),
                key=lambda item: (
                    isinstance(item[1], LocalCPUBackend),
                    isinstance(item[1], AllocatorBackendInterface),
                ),
            )
            if self._retain_failed_copy_resources():
                backends = []
            for name, backend in backends:
                backend_id = id(backend)
                owner = self._storage_backend_close_owners.get(backend_id)
                if owner is None:
                    owner = _BackendCloseOwner(self, name, backend)
                    self._storage_backend_close_owners[backend_id] = owner
                elif owner.backend is not backend:
                    raise RuntimeError("storage backend identity was recycled")
                if owner.complete:
                    self._closed_storage_backend_ids.add(backend_id)
                    continue
                try:
                    logger.info("Closing storage backend: %s", name)
                    owner.close()
                except BaseException as error:
                    logger.error("Error closing backend %s: %s", name, error)
                    if first_error is None:
                        first_error = error
                if owner.complete:
                    self._closed_storage_backend_ids.add(backend_id)
                    logger.info("Storage backend %s closed successfully", name)

            # A failed backend may require the manager event loop to remain
            # alive for its next close attempt. Preserve that dependency and
            # propagate the exact backend failure before touching the loop.
            if first_error is not None:
                raise first_error

            loop = getattr(self, "loop", None)
            thread = getattr(self, "thread", None)

            try:
                # Queue stop even in the small interval after Thread.start()
                # but before run_forever() publishes is_running(). The queued
                # callback will stop the loop as soon as it begins running.
                if (
                    loop is not None
                    and not loop.is_closed()
                    and (
                        loop.is_running() or (thread is not None and thread.is_alive())
                    )
                ):
                    logger.info("Stopping event loop...")
                    loop.call_soon_threadsafe(loop.stop)
                    logger.info("Event loop stop signaled")
            except BaseException as error:
                logger.error("Error stopping storage event loop: %s", error)
                if first_error is None:
                    first_error = error

            try:
                if thread is not None and thread.is_alive():
                    logger.info("Waiting for storage manager thread to finish...")
                    thread.join(timeout=10.0)
                    if thread.is_alive():
                        raise RuntimeError(
                            "storage manager thread did not terminate within 10s"
                        )
                    logger.info("Storage manager thread terminated successfully")
                else:
                    logger.info("Storage manager thread already stopped")
            except BaseException as error:
                logger.error("Error joining storage manager thread: %s", error)
                if first_error is None:
                    first_error = error

            try:
                if (
                    loop is not None
                    and not loop.is_closed()
                    and (thread is None or not thread.is_alive())
                ):
                    loop.close()
            except BaseException as error:
                logger.error("Error closing storage event loop: %s", error)
                if first_error is None:
                    first_error = error

            if first_error is not None:
                raise first_error
            logger.info("Storage manager closed.")
        finally:
            self._finish_put_cleanup_teardown(teardown_token)

    def _retain_failed_copy_resources(self) -> bool:
        """Preserve the backend graph if the copy stream has unknown work."""
        with _FAILED_COPY_STREAMS_LOCK:
            if id(self.internal_copy_stream) not in _FAILED_COPY_STREAMS:
                return False
        with self.manager_lock:
            if not self._copy_owners_retained:
                for backend in self.storage_backends.values():
                    if isinstance(backend, LocalCPUBackend):
                        backend.retain_backing_resources()
                retain_memory_owners([], self)
                self._copy_owners_retained = True
                logger.error(
                    "Retaining storage backends after failed copy synchronization; "
                    "GPU access to staging allocations may still be in progress"
                )
        return True
