# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StoragePluginInterface,
)
from lmcache.v1.storage_backend.raw_block import (
    DEFAULT_IOURING_QUEUE_DEPTH,
    RawBlockCore,
    RawBlockCoreConfig,
    RawBlockKeySpec,
    RawBlockPutManyResult,
    decode_legacy_key,
    encode_legacy_key,
    normalize_raw_block_io_engine,
    round_up,
    validate_raw_block_io_options,
)

if TYPE_CHECKING:
    # Standard
    from concurrent.futures import Future

    # First Party
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

# Resources kept alive because a device stopped being able to say what it was
# doing with them. Nothing reads this list; holding a reference is the whole
# point, so that no allocator, finalizer or exit handler can release memory a
# command may still be reaching.
_RETAINED_AFTER_UNKNOWN_OUTCOME: list[Any] = []


def _probe_says_unknown(probe: Optional[Callable[[], Any]], what: str) -> bool:
    """Read a health probe, insisting that it answer the question asked.

    Only a genuine ``True`` means the outcome is unknown. The probe is a
    predicate returning a bool, so anything else is a caller that substituted
    something not answering this question -- and reading that as poison would
    quarantine on nothing. A probe that *raises* is different: it was asked
    and could not say, which is exactly the condition to fail closed on.
    """
    if probe is None:
        return False
    try:
        answer = probe()
    except Exception:  # pragma: no cover - a probe must not mask the path
        logger.exception("RustRawBlockBackend: could not read %s health", what)
        return True
    if answer is True:
        return True
    if answer is not False:
        logger.warning(
            "RustRawBlockBackend: %s health probe answered %r, not a bool; "
            "reading it as healthy",
            what,
            answer,
        )
    return False


_DEFAULT_META_MAGIC = b"LMCIDX01"
_DEFAULT_META_VERSION = 1

TPRankKey = int | str
PerTPDevicePaths = Mapping[TPRankKey, str]


def _validate_per_tp_device_paths(per_tp_devices: PerTPDevicePaths) -> None:
    """Validate that each TP rank uses a distinct raw-block device path.

    Args:
        per_tp_devices: Mapping from TP rank to raw-block device path.

    Raises:
        ValueError: If the same device path is assigned to multiple ranks.
    """
    values = list(per_tp_devices.values())
    if len(values) != len(set(values)):
        raise ValueError(
            "Duplicate device path configured in rust_raw_block.per_tp_device_paths"
        )


def _get_per_tp_device_path(
    per_tp_devices: PerTPDevicePaths, tp_rank: int
) -> Optional[str]:
    """Return the device path configured for a TP rank.

    Args:
        per_tp_devices: Mapping with string or integer rank keys.
        tp_rank: Tensor-parallel rank to look up.

    Returns:
        The configured path, or None when the rank is absent.
    """
    return per_tp_devices.get(str(tp_rank), per_tp_devices.get(tp_rank))


class RustRawBlockBackend(StoragePluginInterface, AllocatorBackendInterface):
    """
    Legacy raw-block storage plugin wrapper.

    The durable raw-device/index/checkpoint logic now lives in RawBlockCore.
    This wrapper preserves the existing non-MP interface and prefix semantics:
    - TP>1 still uses explicit per-TP device partitions
    - batched_async_contains reports only the leading hit prefix
    - batched_get_{blocking,non_blocking} load only the leading hit prefix
    """

    def __init__(
        self,
        config=None,
        metadata=None,
        local_cpu_backend=None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        dst_device: str = "cpu",
    ):
        super().__init__(
            dst_device=dst_device,
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
        )
        if self.loop is None:
            raise ValueError("RustRawBlockBackend requires an asyncio event loop")
        if self.config is None:
            raise ValueError("RustRawBlockBackend requires config")

        extra = self.config.extra_config or {}

        # Every chunk this backend reads or writes has to live somewhere the
        # device can reach: the local CPU pool, or the GPU staging pool when
        # one is configured.  With a GPU pool the CPU tier is not in the data
        # path at all, so do not require it; without either there is nowhere
        # to put a loaded chunk.
        if self.local_cpu_backend is None and not int(
            extra.get("rust_raw_block.gpu_buffer_bytes", 0) or 0
        ):
            raise ValueError(
                "RustRawBlockBackend needs a staging pool: either a "
                "local CPU backend (max_local_cpu_size > 0) or "
                "extra_config['rust_raw_block.gpu_buffer_bytes']"
            )

        self.device_path: str
        if self.metadata is not None and self.metadata.world_size > 1:
            tp_rank = self.metadata.worker_id
            per_tp_devices = extra.get("rust_raw_block.per_tp_device_paths", {})
            if not isinstance(per_tp_devices, Mapping):
                raise ValueError(
                    "rust_raw_block.per_tp_device_paths must be a mapping from "
                    "TP rank to device path"
                )
            if not per_tp_devices:
                raise ValueError(
                    "For TP > 1, rust_raw_block.per_tp_device_paths is required. "
                    "Each TP worker must have an explicit device path configured."
                )
            _validate_per_tp_device_paths(per_tp_devices)
            device_path = _get_per_tp_device_path(per_tp_devices, tp_rank)
            if not device_path:
                raise ValueError(
                    f"No device path configured for TP rank {tp_rank}. "
                    f"Available ranks: {list(per_tp_devices.keys())}"
                )
            self.device_path = device_path
        else:
            self.device_path = str(extra.get("rust_raw_block.device_path", "") or "")
            if not self.device_path:
                raise ValueError(
                    "extra_config['rust_raw_block.device_path'] is required"
                )

        self._core = RawBlockCore(
            self._build_core_config(extra),
            key_namespace="legacy",
        )
        # A GPU staging pool makes the device the endpoint of every raw-block
        # read and write: the engine registers the pool's slots with io_uring
        # as dma-bufs exported from device memory, stores go out of the
        # object's VRAM slot and loads land in a VRAM slot, with no host
        # bounce.  Without it the local CPU allocator's pinned pages are the
        # endpoints and the GPU connector copies through the host.
        self._gpu_allocator: Optional[Any] = None
        gpu_buffer_bytes = int(extra.get("rust_raw_block.gpu_buffer_bytes", 0) or 0)
        if gpu_buffer_bytes > 0:
            if self._core.io_engine != "io_uring" or self._core.use_uring_cmd:
                self._core.close()
                raise ValueError("GPU staging requires ordinary io_uring DMA-BUF I/O")
            try:
                self._gpu_allocator = self._build_gpu_allocator(
                    gpu_buffer_bytes,
                    extra.get("rust_raw_block.gpu_buffer_device"),
                )
            except BaseException:
                try:
                    outcome = self._core.close()
                except Exception:
                    outcome = None
                if outcome is None or not outcome.may_release_backing_resources:
                    _RETAINED_AFTER_UNKNOWN_OUTCOME.append((self._core,))
                raise
        if self._core.io_engine == "io_uring":
            try:
                self._core.register_fixed_buffers_from_allocator(
                    self.get_memory_allocator()
                )
            except Exception as e:
                if self._gpu_allocator is not None:
                    try:
                        outcome = self._core.close()
                    except Exception:
                        outcome = None
                    if outcome is not None and outcome.may_release_backing_resources:
                        self._gpu_allocator.close()
                    else:
                        _RETAINED_AFTER_UNKNOWN_OUTCOME.append(
                            (self._core, self._gpu_allocator, self.local_cpu_backend)
                        )
                        if self.local_cpu_backend is not None:
                            self.local_cpu_backend.retain_backing_resources()
                    raise
                logger.warning(
                    "RustRawBlockBackend: failed to register io_uring fixed "
                    "buffers: %s. Falling back to non-fixed buffer mode.",
                    e,
                )
        self._warn_if_loaded_metadata_looks_cross_rank()

        self._put_lock = threading.Lock()
        self._put_tasks: set[CacheEngineKey] = set()
        self._quarantined_objs: list[MemoryObj] = []
        self._pending_put_owners: list[list[MemoryObj]] = []
        self._active_operations = 0
        self._sealed = False
        self._closed_once = False
        self._pin_lock = threading.Lock()
        self._pinned_keys: set[str] = set()

    def __str__(self) -> str:
        return "RustRawBlockBackend"

    @property
    def capacity_bytes(self) -> int:
        """Return the effective raw-block capacity in bytes."""
        return int(self._core.capacity_bytes)

    @property
    def block_align(self) -> int:
        """Return the configured raw-device block alignment."""
        return int(self._core.block_align)

    @property
    def header_bytes(self) -> int:
        """Return the per-slot header reservation in bytes."""
        return int(self._core.header_bytes)

    @property
    def slot_bytes(self) -> int:
        """Return the configured raw-block slot size in bytes."""
        return int(self._core.slot_bytes)

    @property
    def meta_total_bytes(self) -> int:
        """Return the reserved metadata checkpoint region size."""
        return int(self._core.meta_total_bytes)

    @property
    def meta_magic_text(self) -> str:
        """Return the ASCII metadata checkpoint magic."""
        return str(self._core.meta_magic_text)

    @property
    def meta_version(self) -> int:
        """Return the metadata checkpoint format version."""
        return int(self._core.meta_version)

    @property
    def data_base_offset(self) -> int:
        """Return the byte offset where data slots begin."""
        return self._core.data_base_offset()

    @property
    def _raw(self) -> Any:
        """Return the raw device handle for legacy test compatibility."""
        return self._core.raw_device()

    @_raw.setter
    def _raw(self, raw_device: Any) -> None:
        """Replace the raw device handle for legacy test compatibility."""
        self._core.set_raw_device_for_testing(raw_device)

    def lock_refcount(self, encoded_key: str) -> int:
        """Return the L2 lock refcount for a legacy encoded key."""
        return self._core.lock_refcount(encoded_key)

    def inflight_io_count(self) -> int:
        """Return the number of active raw-device I/O operations."""
        return self._core.inflight_io_count()

    def indexed_key_count(self) -> int:
        """Return the number of keys currently indexed by raw-block."""
        return self._core.indexed_key_count()

    def entry_offset(self, key: CacheEngineKey) -> int | None:
        """Return the raw-block slot offset for a legacy key."""
        return self._core.entry_offset(encode_legacy_key(key).encoded)

    def metadata_container_offsets(self) -> list[int]:
        """Return checkpoint metadata container offsets in bytes."""
        return self._core.metadata_container_offsets()

    def apply_loaded_state(self, data: dict[str, Any]) -> bool:
        """Validate and apply a raw-block metadata checkpoint payload."""
        return self._core.apply_loaded_state(data)

    def _build_gpu_allocator(self, size_bytes: int, device: Optional[str]) -> Any:
        """Create the paged GPU staging pool used as the raw-block endpoint.

        The pool is paged with the engine's KV shapes so every slot is one
        chunk, exactly like the local CPU allocator, and the slot layout is
        what the engine registers with io_uring.  The metadata supplies the
        shapes and dtypes; without it there is no chunk geometry to page by.
        """
        # First Party
        from lmcache.v1.memory_allocators.gpu_memory_allocator import (
            GPUMemoryAllocator,
        )
        from lmcache.v1.memory_management import MemoryFormat

        if self.metadata is None:
            raise ValueError(
                "rust_raw_block.gpu_buffer_bytes needs the engine metadata for "
                "the KV chunk shapes"
            )
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        allocator = GPUMemoryAllocator(
            size_bytes,
            use_paging=True,
            shapes=self.metadata.get_shapes(),
            dtypes=self.metadata.get_dtypes(),
            fmt=MemoryFormat.KV_2LTD,
            **kwargs,
        )
        if allocator.tensor.device.type != "cuda":
            allocator.close()
            raise RuntimeError("GPU staging needs a CUDA or ROCm device allocation")
        logger.info(
            "RustRawBlockBackend: GPU staging pool of %d MiB on %s is the "
            "raw-block endpoint",
            size_bytes >> 20,
            allocator.tensor.device,
        )
        return allocator

    def _allocate_load_target(self, shape: Any, dtype: Any, fmt: Any) -> Any:
        """Allocate the object a load is read into: a GPU slot when the GPU
        staging pool exists, otherwise a local CPU object."""
        if self._gpu_allocator is not None:
            return self._gpu_allocator.allocate(shape, dtype, fmt)
        if self.local_cpu_backend is None:
            raise RuntimeError("RustRawBlockBackend has no staging pool to load into")
        return self.local_cpu_backend.allocate(shape, dtype, fmt)

    def _full_chunk_size_bytes(self) -> int:
        """Bytes one full KV chunk occupies, which sizes a device slot.

        The local CPU backend computes this from the engine metadata and the
        chunk size; ask it when it exists, and derive the same number from
        that metadata directly when the CPU tier is not configured.
        """
        for name in ("get_full_chunk_size_bytes", "get_full_chunk_size"):
            fn = getattr(self.local_cpu_backend, name, None)
            if callable(fn):
                return int(fn())
        if self.metadata is None:
            raise ValueError(
                "RustRawBlockBackend needs engine metadata to size a slot "
                "when there is no local CPU backend to ask"
            )
        assert self.config is not None
        # kv_shape is [num_layers, kv_size, chunk_size, num_heads, head_size],
        # already divided by the tensor-parallel world size.
        num_layers, kv_size, _, num_heads, head_size = self.metadata.kv_shape
        chunk_tokens = self.config.chunk_size
        hidden_dim = num_heads * head_size
        dtype_size = self.metadata.kv_dtype.itemsize
        if self.config.use_layerwise:
            # One key per layer: [chunk_tokens, kv_size, hidden_dim].
            return chunk_tokens * kv_size * hidden_dim * dtype_size
        # One key per chunk: [kv_size, num_layers, chunk_tokens, hidden_dim].
        return kv_size * num_layers * chunk_tokens * hidden_dim * dtype_size

    def _build_core_config(self, extra: Mapping[str, Any]) -> RawBlockCoreConfig:
        block_align = int(extra.get("rust_raw_block.block_align", 4096))
        header_bytes = int(extra.get("rust_raw_block.header_bytes", 4096))
        use_odirect = bool(extra.get("rust_raw_block.use_odirect", False))
        enable_zero_copy = bool(extra.get("rust_raw_block.enable_zero_copy", True))
        capacity_bytes = int(extra.get("rust_raw_block.capacity_bytes", 0))
        io_engine = normalize_raw_block_io_engine(
            extra.get("rust_raw_block.io_engine"),
            use_iouring=extra.get("rust_raw_block.use_iouring"),
            use_uring=extra.get("rust_raw_block.use_uring"),
        )
        iouring_queue_depth = int(
            extra.get("rust_raw_block.iouring_queue_depth", DEFAULT_IOURING_QUEUE_DEPTH)
        )
        use_uring_cmd = bool(extra.get("rust_raw_block.use_uring_cmd", False))
        max_data_transfer_size = int(
            extra.get("rust_raw_block.max_data_transfer_size", 0)
        )
        validate_raw_block_io_options(
            iouring_queue_depth=iouring_queue_depth,
        )
        meta_total_bytes = int(
            extra.get("rust_raw_block.meta_total_bytes", 128 * 1024 * 1024)
        )
        meta_magic_raw = extra.get("rust_raw_block.meta_magic", "LMCIDX01")
        if isinstance(meta_magic_raw, str):
            meta_magic = meta_magic_raw.encode("ascii")
        elif isinstance(meta_magic_raw, bytes):
            meta_magic = meta_magic_raw
        else:
            raise ValueError("rust_raw_block.meta_magic must be str or bytes")

        full_chunk_bytes = self._full_chunk_size_bytes()
        default_slot_bytes = round_up(header_bytes + full_chunk_bytes, block_align)
        slot_bytes = int(extra.get("rust_raw_block.slot_bytes", default_slot_bytes))

        return RawBlockCoreConfig(
            device_path=self.device_path,
            capacity_bytes=capacity_bytes,
            block_align=block_align,
            header_bytes=header_bytes,
            slot_bytes=slot_bytes,
            use_odirect=use_odirect,
            enable_zero_copy=enable_zero_copy,
            meta_total_bytes=meta_total_bytes,
            meta_magic=meta_magic,
            meta_version=int(
                extra.get("rust_raw_block.meta_version", _DEFAULT_META_VERSION)
            ),
            meta_checkpoint_interval_sec=int(
                extra.get("rust_raw_block.meta_checkpoint_interval_sec", 60)
            ),
            meta_idle_quiet_ms=int(extra.get("rust_raw_block.meta_idle_quiet_ms", 100)),
            meta_enable_periodic=bool(
                extra.get("rust_raw_block.meta_enable_periodic", True)
            ),
            load_checkpoint_on_init=bool(
                extra.get("rust_raw_block.load_checkpoint_on_init", True)
            ),
            meta_verify_on_load=bool(
                extra.get("rust_raw_block.meta_verify_on_load", True)
            ),
            max_data_transfer_size=max_data_transfer_size,
            io_engine=io_engine,
            iouring_queue_depth=iouring_queue_depth,
            use_uring_cmd=use_uring_cmd,
        )

    def _warn_if_loaded_metadata_looks_cross_rank(self) -> None:
        if self.metadata is None:
            return
        first_encoded_key = self._core.first_encoded_key()
        if first_encoded_key is None:
            return
        try:
            first_loaded_key = decode_legacy_key(first_encoded_key)
        except Exception:
            return
        expected_worker_id = int(self.metadata.worker_id)
        loaded_worker_id = int(first_loaded_key.worker_id)
        if loaded_worker_id == expected_worker_id:
            return
        logger.warning(
            "RustRawBlockBackend: loaded metadata may belong to another "
            "worker (device=%s, current_worker_id=%d, "
            "first_entry_worker_id=%d, first_entry_key=%s)",
            self.device_path,
            expected_worker_id,
            loaded_worker_id,
            first_loaded_key.to_string(),
        )

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        spec = encode_legacy_key(key)
        return (
            self._pin_if_needed(spec.encoded)
            if pin
            else self._core.contains_key(
                spec.encoded,
                lock=False,
            )
        )

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._put_lock:
            return key in self._put_tasks

    def pin(self, key: CacheEngineKey) -> bool:
        spec = encode_legacy_key(key)
        return self._pin_if_needed(spec.encoded)

    def unpin(self, key: CacheEngineKey) -> bool:
        spec = encode_legacy_key(key)
        return self._unpin_if_needed(spec.encoded)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        spec = encode_legacy_key(key)
        with self._pin_lock:
            removed = self._core.delete_many([spec.encoded], force=force)[0]
            if removed:
                self._pinned_keys.discard(spec.encoded)
        return removed

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        """Remove multiple keys in a single locked batch.

        Acquires ``_pin_lock`` once and issues one ``_core.delete_many`` call
        for the whole batch, instead of locking per key.

        Args:
            keys: Cache keys to remove.
            force: Passed through to ``RawBlockCore.delete_many``. When false,
                locked entries are preserved.

        Returns:
            Number of keys that were actually removed.
        """
        if not keys:
            return 0
        encoded_keys = [encode_legacy_key(key).encoded for key in keys]
        with self._pin_lock:
            results = self._core.delete_many(encoded_keys, force=force)
            for encoded_key, removed in zip(encoded_keys, results, strict=True):
                if removed:
                    self._pinned_keys.discard(encoded_key)
        return sum(results)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,  # noqa: ARG002
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> list[Future] | None:
        """Schedule writes unless the native io_uring worker has failed.

        Args:
            keys: Cache keys corresponding to ``objs``.
            objs: Memory objects retained until their writes finish.
            transfer_spec: Unused transfer metadata.
            on_complete_callback: Callback for each successfully stored key.

        Returns:
            Scheduled futures, or None when no writes are needed or the native
            worker has failed. Worker failure skips writes without retaining
            objects or raising to the caller.

        Raises:
            RuntimeError: If no event loop exists.
        """
        del transfer_spec
        loop = self.loop
        if loop is None:
            raise RuntimeError("RustRawBlockBackend requires an asyncio event loop")
        try:
            self._core.raise_if_failed()
        except RuntimeError:
            logger.exception("Skipping raw-block store after native worker failure")
            return None

        pending: list[tuple[CacheEngineKey, RawBlockKeySpec, MemoryObj]] = []
        for key, obj in zip(keys, objs, strict=False):
            with self._put_lock:
                if self._sealed:
                    break
                if key in self._put_tasks:
                    continue
                self._put_tasks.add(key)

            spec = encode_legacy_key(key)
            exists = self._core.contains_key(
                spec.encoded,
                lock=False,
            ) or self._core.exists_inflight(spec.encoded)
            if exists:
                with self._put_lock:
                    self._put_tasks.discard(key)
                continue

            obj.ref_count_up()
            pending.append((key, spec, obj))

        if not pending:
            return None

        # On scheduling failure, roll back unscheduled items; scheduled ones
        # release their ref / put-task entry in their coroutine's finally.
        scheduled_count = 0
        try:
            if self._core.io_engine == "io_uring" and len(pending) > 1:
                coro = self._submit_put_many(pending, on_complete_callback)
                try:
                    fut = asyncio.run_coroutine_threadsafe(coro, loop)
                except Exception:
                    coro.close()
                    raise
                return [fut]

            futures: list[Future] = []
            for key, spec, obj in pending:
                coro = self._submit_put_one(key, spec, obj, on_complete_callback)
                try:
                    fut = asyncio.run_coroutine_threadsafe(coro, loop)
                except Exception:
                    coro.close()
                    raise
                futures.append(fut)
                scheduled_count += 1
            return futures
        except Exception:
            for _, _, obj in pending[scheduled_count:]:
                obj.ref_count_down()
            with self._put_lock:
                for key, _, _ in pending[scheduled_count:]:
                    self._put_tasks.discard(key)
            raise

    def _outcome_is_unknown(self) -> bool:
        """Whether the worker or the core has stopped being able to say.

        The core is asked first. It keeps the answer once it has adopted it,
        and asking it costs nothing, whereas reaching the device can reopen
        one -- or be refused outright for exactly the reason being asked
        about.
        """
        if _probe_says_unknown(
            getattr(self._core, "is_poisoned", None),
            "raw-block core",
        ):
            return True
        return self._native_outcome_is_unknown()

    def _native_outcome_is_unknown(self) -> bool:
        """Ask the existing device without opening a new one during teardown."""
        raw = getattr(self._core, "_raw", None)
        return _probe_says_unknown(getattr(raw, "is_poisoned", None), "native engine")

    def _start_put_many(
        self,
        specs: Sequence[RawBlockKeySpec],
        memory_objs: Sequence[MemoryObj],
    ) -> tuple["asyncio.Future[Any]", threading.Event]:
        """Start a batched write on a worker thread and track it honestly.

        ``asyncio.to_thread`` hands work to an executor. Cancelling the task
        that awaits it stops the waiting and never stops the thread, so the
        task alone cannot say whether the device is still being handed these
        buffers. The returned event is set by the thread itself and can.
        """
        io_finished = threading.Event()

        def _run_put() -> RawBlockPutManyResult:
            try:
                return self._core.put_many(list(specs), list(memory_objs))
            finally:
                io_finished.set()

        return asyncio.ensure_future(asyncio.to_thread(_run_put)), io_finished

    def _settle_put_owners(
        self,
        pending: Sequence[tuple[CacheEngineKey, RawBlockKeySpec, MemoryObj]],
        io_task: "asyncio.Future[Any]",
        io_finished: threading.Event,
    ) -> None:
        """Release a batch's buffers, or keep them until that is provable.

        Dropping the reference can return the object's pool slice to the
        allocator, which is only safe once the device is known to have
        finished with it. Two things make that unknowable: the engine
        poisoned itself because its worker could not say, or -- the case a
        failed request does not cover -- this coroutine was cancelled while
        its I/O thread is still running, so nothing has happened yet that
        could report an outcome at all.
        """
        if not io_finished.is_set():
            self._retain_put_owners_until_done(pending, io_task, io_finished)
            return
        self._release_put_owners(pending)

    def _release_put_owners(
        self,
        pending: Sequence[tuple[CacheEngineKey, RawBlockKeySpec, MemoryObj]],
    ) -> None:
        unknown = self._outcome_is_unknown()
        for _key, _spec, memory_obj in pending:
            if unknown:
                self._quarantined_objs.append(memory_obj)
            else:
                memory_obj.ref_count_down()

    def _retain_put_owners_until_done(
        self,
        pending: Sequence[tuple[CacheEngineKey, RawBlockKeySpec, MemoryObj]],
        io_task: "asyncio.Future[Any]",
        io_finished: threading.Event,
    ) -> None:
        """Hold a batch's buffers until its I/O thread has actually finished.

        The batch is recorded where shutdown can see it, so an engine closing
        with a write still in a thread does not destroy the memory under it.
        If the callback never runs -- a loop torn down first, say -- the batch
        stays recorded, which is the safe direction.
        """
        owners = [item[2] for item in pending]
        with self._put_lock:
            self._pending_put_owners.append(owners)
        logger.warning(
            "Raw-block write for %d key(s) was abandoned while its I/O "
            "thread is still running; withholding its buffers until it ends",
            len(owners),
        )

        def _settle(_task: "asyncio.Future[Any]") -> None:
            with self._put_lock:
                if not any(batch is owners for batch in self._pending_put_owners):
                    return
            if not io_finished.is_set():
                # The task ended without its thread ending. Nothing here can
                # say what the device is doing with these buffers.
                self._quarantined_objs.extend(owners)
            else:
                self._release_put_owners(pending)
            # Keep the batch visible until releasing or quarantining every
            # owner is finished; shutdown must not miss that transition.
            with self._put_lock:
                for index, batch in enumerate(self._pending_put_owners):
                    if batch is owners:
                        del self._pending_put_owners[index]
                        break

        io_task.add_done_callback(_settle)

    def _withhold_unproven_reads(
        self,
        unproven: Sequence[RawBlockKeySpec],
        targets: Sequence[MemoryObj],
        locked_specs: list[RawBlockKeySpec],
    ) -> list[RawBlockKeySpec]:
        """Keep a read nobody can vouch for out of everyone's reach.

        Such a read may still be landing. Its destination buffers cannot go
        back to the allocator, where the next request would be handed memory
        the device is writing into; and its source extents cannot be
        unlocked, because an unlocked entry can be evicted and its slot given
        to a write while the read is still reading it. Neither is released
        again for the life of this engine.

        The keys are added to the pinned set, which is what already tells a
        later prefix lookup not to take a lock reference this backend is
        holding, so nothing re-locks or re-serves them either.

        Returns the list the caller's ``finally`` may unlock. It is a return
        value rather than a mutation so that there is no way to unlock what
        this withheld by forgetting to look.
        """
        withheld = {spec.encoded for spec in unproven}
        self._quarantined_objs.extend(targets)
        with self._pin_lock:
            self._pinned_keys |= withheld
        logger.error(
            "Raw-block read outcome is unknown for %d key(s); withholding "
            "%d destination buffer(s) and keeping those keys locked. This "
            "engine will not serve or reuse them again.",
            len(withheld),
            len(targets),
        )
        return [spec for spec in locked_specs if spec.encoded not in withheld]

    async def _submit_put_one(
        self,
        key: CacheEngineKey,
        spec: RawBlockKeySpec,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> None:
        pending = [(key, spec, memory_obj)]
        io_task, io_finished = self._start_put_many([spec], [memory_obj])
        try:
            put_result = await asyncio.shield(io_task)
            if not put_result.results or not put_result.results[0]:
                raise RuntimeError(f"Failed to persist raw-block key {spec.encoded}")
            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning("on_complete_callback failed for key %s: %s", key, e)
        finally:
            self._settle_put_owners(pending, io_task, io_finished)
            with self._put_lock:
                self._put_tasks.discard(key)

    async def _submit_put_many(
        self,
        pending: Sequence[tuple[CacheEngineKey, RawBlockKeySpec, MemoryObj]],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> None:
        """Persist multiple legacy raw-block keys in one background batch.

        Args:
            pending: Ordered ``(key, spec, memory_obj)`` tuples to persist.
            on_complete_callback: Optional per-key completion callback.

        Raises:
            RuntimeError: If any key fails to persist.
            Exception: Propagates raw-device write failures from the core.
        """
        keys = [item[0] for item in pending]
        specs = [item[1] for item in pending]
        memory_objs = [item[2] for item in pending]
        io_task, io_finished = self._start_put_many(specs, memory_objs)
        try:
            put_result = await asyncio.shield(io_task)
            if len(put_result.results) != len(pending) or not all(put_result.results):
                failed = []

                for key, spec, ok in zip(keys, specs, put_result.results, strict=False):
                    if ok:
                        if on_complete_callback is not None:
                            try:
                                on_complete_callback(key)
                            except Exception as e:
                                logger.warning(
                                    "on_complete_callback failed for key %s: %s", key, e
                                )
                    else:
                        failed.append(spec.encoded)

                if failed:
                    raise RuntimeError(
                        "Failed to persist raw-block keys: " + ", ".join(failed)
                    )
            if on_complete_callback is not None:
                for key in keys:
                    try:
                        on_complete_callback(key)
                    except Exception as e:
                        logger.warning(
                            "on_complete_callback failed for key %s: %s", key, e
                        )
        finally:
            self._settle_put_owners(pending, io_task, io_finished)
            for key, _spec, _memory_obj in pending:
                with self._put_lock:
                    self._put_tasks.discard(key)

    def _batched_get_prefix(
        self,
        keys: Sequence[CacheEngineKey],
    ) -> list[MemoryObj]:
        # Include preparation and cleanup, not just the native read: the
        # allocator is already in use while a destination is being built.
        with self._put_lock:
            if self._sealed:
                return []
            self._active_operations += 1
        try:
            return self._load_prefix(keys)
        finally:
            with self._put_lock:
                self._active_operations -= 1

    def _load_prefix(
        self,
        keys: Sequence[CacheEngineKey],
    ) -> list[MemoryObj]:
        if not keys:
            return []

        specs = [encode_legacy_key(key) for key in keys]
        encoded_keys = [spec.encoded for spec in specs]
        allocated: list[MemoryObj] = []
        locked_specs: list[RawBlockKeySpec] = []
        with self._pin_lock:
            prefix_metas = self._core.get_metadata_prefix(
                encoded_keys,
                lock=True,
                skip_locked=self._pinned_keys,
            )
            prefix_specs = specs[: len(prefix_metas)]
            locked_specs = [
                spec for spec in prefix_specs if spec.encoded not in self._pinned_keys
            ]

        if not prefix_specs:
            return []

        try:
            for spec, meta in zip(prefix_specs, prefix_metas, strict=False):
                if meta.shape is None or meta.dtype is None:
                    logger.warning(
                        "Raw-block metadata missing shape/dtype for key %s; "
                        "aborting prefix load",
                        spec.encoded,
                    )
                    break
                memory_obj = self._allocate_load_target(
                    meta.shape,
                    meta.dtype,
                    meta.fmt,
                )
                if memory_obj is None:
                    logger.error("Failed to allocate memory for key %s", spec.encoded)
                    break
                allocated.append(memory_obj)

            if not allocated:
                return []

            load_specs = prefix_specs[: len(allocated)]
            load_results = self._core.load_many_into(
                [spec.encoded for spec in load_specs],
                allocated,
            )
            loaded_count = 0
            for ok in load_results:
                if not ok:
                    break
                loaded_count += 1
            if loaded_count == len(allocated):
                return allocated

            if self._outcome_is_unknown():
                # The bitmap's leading True entries are real completions, so
                # that prefix is proven and is still served. What follows it
                # is not, and is withheld rather than freed.
                locked_specs = self._withhold_unproven_reads(
                    load_specs[loaded_count:],
                    allocated[loaded_count:],
                    locked_specs,
                )
                return allocated[:loaded_count]

            for obj in allocated[loaded_count:]:
                obj.ref_count_down()
            return allocated[:loaded_count]
        except Exception:
            if self._outcome_is_unknown():
                # Nothing was proven here, so none of it is released.
                locked_specs = self._withhold_unproven_reads(
                    prefix_specs, allocated, locked_specs
                )
                raise
            for obj in allocated:
                obj.ref_count_down()
            raise
        finally:
            self._core.unlock_many([spec.encoded for spec in locked_specs])

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        loaded = self._batched_get_prefix([key])
        return loaded[0] if loaded else None

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Synchronously load the leading raw-block hit prefix.

        Args:
            keys: Ordered legacy cache keys to load.

        Returns:
            A list aligned with ``keys`` containing loaded memory objects for
            the contiguous hit prefix and ``None`` for the remaining suffix.

        Raises:
            RuntimeError: If the local CPU allocator backend is unavailable.
        """
        if not keys:
            return []
        loaded = self._batched_get_prefix(keys)
        return [*loaded, *([None] * (len(keys) - len(loaded)))]

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        del lookup_id
        specs = [encode_legacy_key(key) for key in keys]
        encoded_keys = [spec.encoded for spec in specs]
        results = self._core.exists_many(encoded_keys, lock=False)
        prefix_hits = 0
        for ok in results:
            if not ok:
                break
            prefix_hits += 1
        if pin and prefix_hits > 0:
            pinned_hits = 0
            for encoded_key in encoded_keys[:prefix_hits]:
                if not self._pin_if_needed(encoded_key):
                    break
                pinned_hits += 1
            prefix_hits = pinned_hits
        return prefix_hits

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Asynchronously load the leading raw-block hit prefix.

        Args:
            lookup_id: Lookup identifier supplied by the storage manager.
            keys: Ordered legacy cache keys to load.
            transfer_spec: Optional transfer metadata; unused by raw-block.

        Returns:
            Loaded memory objects for the contiguous hit prefix only.

        Raises:
            RuntimeError: If the local CPU allocator backend is unavailable.
        """
        del lookup_id, transfer_spec
        return await asyncio.to_thread(self._batched_get_prefix, keys)

    @property
    def is_gpu_endpoint(self) -> bool:
        """True when this backend's GPU staging pool is the engine's allocator,
        so KV chunks are staged in VRAM and move to and from NVMe directly."""
        return self._gpu_allocator is not None

    def get_allocator_backend(self) -> AllocatorBackendInterface:
        if self._gpu_allocator is not None:
            return self
        if self.local_cpu_backend is None:
            raise RuntimeError("RustRawBlockBackend requires local_cpu_backend")
        return self.local_cpu_backend

    def initialize_allocator(self, config: Any, metadata: Any) -> Any:
        return self.get_memory_allocator()

    def get_memory_allocator(self) -> Any:
        if self._gpu_allocator is not None:
            return self._gpu_allocator
        if self.local_cpu_backend is None:
            raise RuntimeError("RustRawBlockBackend requires local_cpu_backend")
        return self.local_cpu_backend.get_memory_allocator()

    def allocate(
        self,
        shapes: Any,
        dtypes: Any,
        fmt: Any = None,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        # First Party
        from lmcache.v1.memory_management import MemoryFormat

        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        if self._gpu_allocator is not None:
            # The GPU pool has no evictor: a full pool returns None and the
            # engine skips the store.
            return self._gpu_allocator.allocate(shapes, dtypes, fmt)
        if self.local_cpu_backend is None:
            raise RuntimeError("RustRawBlockBackend requires local_cpu_backend")
        return self.local_cpu_backend.allocate(
            shapes, dtypes, fmt, eviction=eviction, busy_loop=busy_loop
        )

    def batched_allocate(
        self,
        shapes: Any,
        dtypes: Any,
        batch_size: int,
        fmt: Any = None,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        # First Party
        from lmcache.v1.memory_management import MemoryFormat

        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        if self._gpu_allocator is not None:
            return self._gpu_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)
        if self.local_cpu_backend is None:
            raise RuntimeError("RustRawBlockBackend requires local_cpu_backend")
        return self.local_cpu_backend.batched_allocate(
            shapes, dtypes, batch_size, fmt, eviction=eviction, busy_loop=busy_loop
        )

    def close(self) -> None:
        with self._put_lock:
            if self._closed_once:
                return
            self._closed_once = True
            self._sealed = True

        deadline = time.monotonic() + 10.0
        while True:
            with self._put_lock:
                pending = len(self._put_tasks) + self._active_operations
                retained_batches = len(self._pending_put_owners)
            if pending + retained_batches == 0 or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        if pending or retained_batches:
            # A deadline is not a fence for a caller still preparing I/O.
            self._retain_whole_graph(retained_batches)
            return

        unknown = bool(self._quarantined_objs) or self._outcome_is_unknown()
        try:
            outcome = self._core.close()
        except Exception:
            logger.exception("Raw-block core close could not prove quiescence")
            unknown = True
        else:
            if outcome is None or not outcome.may_release_backing_resources:
                unknown = True
        if unknown:
            self._retain_whole_graph(retained_batches)
        elif self._gpu_allocator is not None:
            # Native close fenced all registered transfers before exported
            # handles or the staging arena can be released.
            self._gpu_allocator.close()

    def _retain_whole_graph(self, retained_batches: int) -> None:
        """Keep owners alive and prevent an explicit allocator close as well."""
        if self.local_cpu_backend is not None:
            self.local_cpu_backend.retain_backing_resources()
        _RETAINED_AFTER_UNKNOWN_OUTCOME.append(
            (
                self._core,
                self.local_cpu_backend,
                self._gpu_allocator,
                self._quarantined_objs,
                self._pending_put_owners,
            )
        )
        logger.error(
            "Raw-block backend retaining backing memory, %d buffer owner(s) "
            "and %d abandoned batch(es): native quiescence was not proven",
            len(self._quarantined_objs),
            retained_batches,
        )

    def _pin_if_needed(self, encoded_key: str) -> bool:
        with self._pin_lock:
            if encoded_key in self._pinned_keys:
                return True
            if not self._core.exists_many([encoded_key], lock=True)[0]:
                return False
            self._pinned_keys.add(encoded_key)
            return True

    def _unpin_if_needed(self, encoded_key: str) -> bool:
        with self._pin_lock:
            if encoded_key in self._pinned_keys:
                self._core.unlock_many([encoded_key])
                self._pinned_keys.discard(encoded_key)
                return True
            return self._core.contains_key(encoded_key, lock=False)
