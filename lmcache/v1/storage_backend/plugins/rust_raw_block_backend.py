# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from concurrent.futures import CancelledError
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import os
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
    RawBlockDerivationDescriptor,
    RawBlockKeySpec,
    RawBlockPDRequestTracker,
    RawBlockPublicationReceipt,
    RawBlockPutManyResult,
    ReadAckIdentity,
    ReadAckOutcome,
    decode_legacy_key,
    encode_legacy_key,
    normalize_raw_block_io_engine,
    round_up,
    validate_raw_block_io_options,
)
from lmcache.v1.storage_backend.storage_pd_ack import (
    ACK_REJECTED,
    ACK_UNRESOLVED,
    StoragePDAckRequest,
    StoragePDAckServer,
    StoragePDClaimAnswer,
    StoragePDClaimRequest,
    StoragePDUnreadAnswer,
    StoragePDUnreadRequest,
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


def _resolve_role(config, extra: dict) -> str:
    """Return which side of a handoff this node runs, writer or reader.

    A handoff configured through ``pd_data_path`` already says which side
    this node is, in ``pd_role``. Reading it from there means a deployment
    states it once instead of twice, where the two could disagree. An
    explicit plugin setting still wins, for a raw-block pairing set up
    without the P/D switch at all.
    """
    explicit = str(extra.get("rust_raw_block.role", "") or "")
    if explicit:
        return explicit
    if getattr(config, "pd_uses_shared_storage", False):
        return "reader" if config.pd_role == "receiver" else "writer"
    return "writer"


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

        # Prefill/decode handoff over a shared NVMe namespace: the prefill
        # side runs the "writer" role and publishes its index after every put
        # batch; the decode side runs the "reader" role on the same namespace,
        # never writes, and re-reads the published index when a lookup misses.
        self._role = _resolve_role(self.config, extra)
        if self._role not in ("writer", "reader"):
            raise ValueError(
                f"rust_raw_block.role must be 'writer' or 'reader', got {self._role!r}"
            )
        self._publish_after_put = (
            bool(extra.get("rust_raw_block.publish_after_put", False))
            and self._role == "writer"
        )
        self._index_refresh_min_ms = int(
            extra.get("rust_raw_block.index_refresh_min_ms", 50)
        )
        self._index_refresh_wait_ms = int(
            extra.get("rust_raw_block.index_refresh_wait_ms", 0)
        )
        self._last_refresh_ts = 0.0
        self._refresh_lock = threading.Lock()
        self._warned_reader_put = False
        # Memory objects whose references are deliberately not dropped, because
        # the device's access to them was never shown to have ended. Held for
        # the life of this backend so their pool slices cannot be reallocated.
        self._quarantined_objs: list[Any] = []
        # A handoff configured through pd_data_path says the same thing as
        # the plugin-level switch, so either turns the mode on.
        self._storage_pd_mode = bool(
            extra.get("rust_raw_block.storage_pd_mode", False)
        ) or bool(getattr(self.config, "pd_uses_shared_storage", False))
        if self._storage_pd_mode:
            if bool(getattr(self.config, "use_layerwise", False)):
                raise ValueError(
                    "rust_raw_block.storage_pd_mode requires use_layerwise=false"
                )
            if bool(getattr(self.config, "enable_async_loading", False)):
                raise ValueError(
                    "rust_raw_block.storage_pd_mode requires enable_async_loading=false"
                )
            if not bool(getattr(self.config, "save_unfull_chunk", False)):
                raise ValueError(
                    "rust_raw_block.storage_pd_mode requires save_unfull_chunk=true"
                )
            if self._publish_after_put:
                raise ValueError(
                    "rust_raw_block.storage_pd_mode requires "
                    "publish_after_put=false; the request tracker owns publication"
                )
            allow_unsafe_test_io = bool(
                extra.get("rust_raw_block.allow_unsafe_pd_io_for_testing", False)
            )
            if not allow_unsafe_test_io and not bool(
                extra.get("rust_raw_block.require_dmabuf_registration", False)
            ):
                raise ValueError(
                    "rust_raw_block.storage_pd_mode requires strict dma-buf "
                    "registration; set require_dmabuf_registration=true"
                )
            if (
                not allow_unsafe_test_io
                and int(extra.get("rust_raw_block.gpu_buffer_bytes", 0) or 0) <= 0
            ):
                raise ValueError(
                    "rust_raw_block.storage_pd_mode requires a GPU staging "
                    "arena; set rust_raw_block.gpu_buffer_bytes"
                )

        core_config = self._build_core_config(extra)
        if self._storage_pd_mode:
            # Derived rather than configured: it is a fact about how this
            # deployment hashes, not a knob, and a knob would let the two
            # sides of a handoff disagree with the engines they describe.
            core_config = replace(
                core_config,
                derivation=self._observed_key_derivation("legacy"),
            )
        self._core = RawBlockCore(core_config, key_namespace="legacy")
        # A GPU staging pool makes the device the endpoint of every raw-block
        # read and write: the engine registers the pool's slots with io_uring
        # as dma-bufs exported from device memory, stores go out of the
        # object's VRAM slot and loads land in a VRAM slot, with no host
        # bounce.  Without it the local CPU allocator's pinned pages are the
        # endpoints and the GPU connector copies through the host.
        self._gpu_allocator: Optional[Any] = None
        gpu_buffer_bytes = int(extra.get("rust_raw_block.gpu_buffer_bytes", 0) or 0)
        if gpu_buffer_bytes > 0:
            self._gpu_allocator = self._build_gpu_allocator(
                gpu_buffer_bytes,
                extra.get("rust_raw_block.gpu_buffer_device"),
            )
        if self._core.io_engine == "io_uring":
            try:
                self._core.register_fixed_buffers_from_allocator(
                    self.get_memory_allocator()
                )
            except Exception as e:
                if self._core.require_dmabuf_registration:
                    self._core.close()
                    if self._gpu_allocator is not None:
                        close_gpu_allocator = getattr(
                            self._gpu_allocator, "close", None
                        )
                        if callable(close_gpu_allocator):
                            close_gpu_allocator()
                    raise RuntimeError(
                        "RustRawBlockBackend requires dma-buf fixed-buffer "
                        "registration, but registration failed"
                    ) from e
                logger.warning(
                    "RustRawBlockBackend: failed to register io_uring fixed "
                    "buffers: %s. Falling back to non-fixed buffer mode.",
                    e,
                )
        self._warn_if_loaded_metadata_looks_cross_rank()
        self._pd_tracker = (
            RawBlockPDRequestTracker(self._core)
            if self._storage_pd_mode and self._role == "writer"
            else None
        )
        # A writer that cannot hear an acknowledgement holds every extent it
        # ever publishes for its own lifetime, so the listener is built
        # beside the tracker that owns those leases.
        self._ack_receiver: Optional[StoragePDAckServer] = None
        self._ack_tp_rank = int(getattr(metadata, "worker_id", 0) or 0)
        # Names one producer/consumer group's run. Both ends carry it, so a
        # restart of the whole group is a new session and a restart of the
        # consumer alone is not -- which is what lets the writer refuse a
        # consumer that cannot have done the reading it claims.
        self._pd_session_id = str(
            extra.get("rust_raw_block.pd_session_id", "")
            or os.environ.get("LMCACHE_STORAGE_PD_SESSION", "")
        )
        # An operator's assertion that this engine's shutdown is part of a
        # whole-group stop: every engine that could read this namespace is
        # going away too, so holds kept for readers have nobody left to
        # protect. Nothing in this process can observe that, which is why
        # it is configured rather than inferred, and why it is off by
        # default -- a local shutdown says nothing about another machine.
        self._pd_group_quiesced_teardown = bool(
            extra.get("rust_raw_block.pd_group_quiesced_teardown", False)
        )
        if self._pd_tracker is not None:
            self._ack_receiver = self._build_ack_receiver(extra)
            self._pd_tracker.ack_endpoint = self.ack_endpoint()

        self._put_lock = threading.Lock()
        self._put_tasks: set[CacheEngineKey] = set()
        # Batches whose I/O thread is still running after the task awaiting
        # it went away. Their buffers are nobody's to release until it ends.
        self._pending_put_owners: list[list[MemoryObj]] = []
        self._closed_once = False
        # Set at the top of close, before anything waits. A wait that runs
        # while new work is still being admitted has no end, and every
        # admission path below checks this rather than discovering a
        # half-torn-down engine on its own.
        self._sealed = False
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
        """Whether the native engine has stopped being able to say.

        Once it cannot establish what the device is doing, nothing this
        backend releases can be shown to be safe to release, so it keeps what
        it holds rather than handing it back.
        """
        try:
            probe = getattr(self._raw, "is_poisoned", None)
        except Exception:
            # Obtaining the device is itself part of what can fail, and a
            # core that has stopped being able to say refuses to hand one
            # out at all. That refusal is not an answer about health: the
            # core's own flag is, and _outcome_is_unknown reads it.
            logger.warning(
                "RustRawBlockBackend: could not reach the native engine to "
                "ask about its health"
            )
            return False
        return _probe_says_unknown(probe, "native engine")

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
        config = self.config
        if config is None:
            raise ValueError("RustRawBlockBackend requires config to size a slot")
        # kv_shape is [num_layers, kv_size, chunk_size, num_heads, head_size],
        # already divided by the tensor-parallel world size.
        num_layers, kv_size, _, num_heads, head_size = self.metadata.kv_shape
        chunk_tokens = config.chunk_size
        hidden_dim = num_heads * head_size
        dtype_size = self.metadata.kv_dtype.itemsize
        if config.use_layerwise:
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

        role = _resolve_role(self.config, extra)
        return RawBlockCoreConfig(
            role=role,
            verify_slot_header_on_load=self._storage_pd_mode
            or bool(
                extra.get("rust_raw_block.verify_slot_header_on_load", role == "reader")
            ),
            publish_min_interval_ms=0
            if self._storage_pd_mode
            else int(extra.get("rust_raw_block.publish_min_interval_ms", 0)),
            require_dmabuf_registration=bool(
                extra.get("rust_raw_block.require_dmabuf_registration", False)
            ),
            writer_epoch=str(extra.get("rust_raw_block.writer_epoch", "") or ""),
            namespace_identity=str(
                extra.get("rust_raw_block.namespace_identity", "") or ""
            ),
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

    def _contains_here(self, key: CacheEngineKey, pin: bool) -> bool:
        spec = encode_legacy_key(key)
        return (
            self._pin_if_needed(spec.encoded)
            if pin
            else self._core.contains_key(
                spec.encoded,
                lock=False,
            )
        )

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        if self._contains_here(key, pin):
            return True
        if self._role == "reader" and self._refresh_index():
            return self._contains_here(key, pin)
        return False

    def batched_contains(self, keys: List[CacheEngineKey], pin: bool = False) -> int:
        """Return the prefix hit count, letting a reader wait for the writer.

        A reader that misses re-reads the writer's published index and, when
        ``rust_raw_block.index_refresh_wait_ms`` is set, keeps re-reading until
        the keys show up or the wait runs out.  That wait is the handoff: the
        decode side asks before the prefill side has finished publishing.
        """
        hit = 0
        while hit < len(keys) and self._contains_here(keys[hit], pin):
            hit += 1
        if hit == len(keys) or self._role != "reader":
            return hit
        deadline = time.monotonic() + self._index_refresh_wait_ms / 1000.0
        while True:
            if self._refresh_index():
                while hit < len(keys) and self._contains_here(keys[hit], pin):
                    hit += 1
                if hit == len(keys):
                    break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, max(self._index_refresh_min_ms, 1) / 1000.0))
        return hit

    def _refresh_index(self) -> bool:
        """Re-read the writer's index, no more often than the configured
        minimum spacing.  Returns True when the index changed."""
        with self._refresh_lock:
            now = time.monotonic()
            if (now - self._last_refresh_ts) * 1000.0 < self._index_refresh_min_ms:
                return False
            self._last_refresh_ts = now
            try:
                return bool(self._core.refresh_index_from_device())
            except Exception as e:
                logger.warning("RustRawBlockBackend: index refresh failed: %s", e)
                return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._put_lock:
            return key in self._put_tasks

    def cancel_request(self, req_id: str) -> None:
        """Prevent an aborted storage-P/D request from being published."""
        if self._pd_tracker is not None:
            self._pd_tracker.fail_request(
                req_id,
                CancelledError(f"storage P/D request {req_id} was cancelled"),
            )

    def finish_request(self, req_id: str) -> None:
        """Fail an incomplete request after its model execution has ended."""
        if self._pd_tracker is not None:
            self._pd_tracker.finish_request(req_id)

    def adopt_publication(
        self,
        receipt: RawBlockPublicationReceipt,
        keys: Sequence[CacheEngineKey],
        *,
        timeout_ms: int,
    ) -> bool:
        """Adopt the request generation advertised by a storage-P/D writer."""
        if self._role != "reader":
            raise RuntimeError("only a raw-block reader can adopt a publication")
        if self._sealed:
            raise RuntimeError(
                "raw-block storage P/D is shutting down and adopts no "
                "further publications"
            )
        encoded_keys = [encode_legacy_key(key).encoded for key in keys]
        return self._core.refresh_until_publication(
            receipt,
            encoded_keys,
            timeout_ms=timeout_ms,
            refresh_interval_ms=max(self._index_refresh_min_ms, 1),
        )

    def publish_existing_request(
        self,
        keys: Sequence[CacheEngineKey],
        transfer_spec: Any,
    ) -> Future:
        """Publish a final P/D request whose last iteration wrote no new KV."""
        if self._pd_tracker is None or self._role != "writer":
            raise RuntimeError("raw-block storage P/D writer is unavailable")
        if self._sealed:
            raise RuntimeError(
                "raw-block storage P/D is shutting down and admits no "
                "further publications"
            )
        req_id = str(getattr(transfer_spec, "req_id", "") or "")
        expected_chunks = int(getattr(transfer_spec, "total_chunks", 0) or 0)
        if self._pd_tracker.has_request(req_id):
            return self._pd_tracker.finalize_request(
                req_id,
                expected_chunks=expected_chunks,
            )
        encoded_keys = [encode_legacy_key(key).encoded for key in keys]
        return self._pd_tracker.register_batch(
            req_id,
            encoded_keys,
            expected_chunks=expected_chunks,
            is_last_batch=True,
            completed_keys=encoded_keys,
        )

    def pin(self, key: CacheEngineKey) -> bool:
        spec = encode_legacy_key(key)
        return self._pin_if_needed(spec.encoded)

    def unpin(self, key: CacheEngineKey) -> bool:
        spec = encode_legacy_key(key)
        return self._unpin_if_needed(spec.encoded)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        if self._role == "reader":
            return False
        spec = encode_legacy_key(key)
        with self._pin_lock:
            removed = self._core.delete_many(
                [spec.encoded],
                force=force and not self._storage_pd_mode,
            )[0]
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
        if not keys or self._role == "reader":
            return 0
        encoded_keys = [encode_legacy_key(key).encoded for key in keys]
        with self._pin_lock:
            results = self._core.delete_many(
                encoded_keys,
                force=force and not self._storage_pd_mode,
            )
            for encoded_key, removed in zip(encoded_keys, results, strict=True):
                if removed:
                    self._pinned_keys.discard(encoded_key)
        return sum(results)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> list[Future] | None:
        if self._sealed:
            logger.warning(
                "RustRawBlockBackend on %s is shutting down and admits no "
                "further stores",
                self.device_path,
            )
            return None
        if self._storage_pd_mode:
            return self._batched_submit_pd_request(
                keys,
                objs,
                transfer_spec,
                on_complete_callback,
            )
        del transfer_spec
        if self._role == "reader":
            if not self._warned_reader_put:
                self._warned_reader_put = True
                logger.warning(
                    "RustRawBlockBackend: reader role on %s stores nothing; "
                    "puts are dropped",
                    self.device_path,
                )
            return None
        loop = self.loop
        if loop is None:
            raise RuntimeError("RustRawBlockBackend requires an asyncio event loop")

        pending: list[tuple[CacheEngineKey, RawBlockKeySpec, MemoryObj]] = []
        for key, obj in zip(keys, objs, strict=False):
            with self._put_lock:
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

    def _batched_submit_pd_request(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> list[Future] | None:
        """Submit one storage-P/D batch and return request-level completion."""
        if self._role == "reader":
            if not self._warned_reader_put:
                self._warned_reader_put = True
                logger.warning(
                    "RustRawBlockBackend: reader role on %s stores nothing; "
                    "puts are dropped",
                    self.device_path,
                )
            return None
        if self._pd_tracker is None:
            raise RuntimeError("raw-block P/D request tracker is unavailable")
        if transfer_spec is None:
            raise ValueError("storage P/D puts require transfer_spec")
        if len(keys) != len(objs):
            raise ValueError("storage P/D keys and objects must have equal length")

        req_id = str(getattr(transfer_spec, "req_id", "") or "")
        expected_chunks = int(getattr(transfer_spec, "total_chunks", 0) or 0)
        is_last_batch = bool(getattr(transfer_spec, "is_last_prefill", False))
        specs = [encode_legacy_key(key) for key in keys]
        encoded_keys = [spec.encoded for spec in specs]
        if len(set(encoded_keys)) != len(encoded_keys):
            terminal = self._pd_tracker.register_batch(
                req_id,
                list(dict.fromkeys(encoded_keys)),
                expected_chunks=expected_chunks,
                is_last_batch=is_last_batch,
            )
            self._pd_tracker.fail_request(
                req_id,
                RuntimeError("storage P/D batch contains duplicate keys"),
            )
            return [terminal]

        pending: list[tuple[CacheEngineKey, RawBlockKeySpec, MemoryObj]] = []
        completed_keys: list[str] = []
        conflict: str | None = None
        for key, spec, obj in zip(keys, specs, objs, strict=True):
            with self._put_lock:
                already_scheduled = key in self._put_tasks
                if not already_scheduled:
                    self._put_tasks.add(key)
            if already_scheduled or self._core.exists_inflight(spec.encoded):
                conflict = spec.encoded
                if not already_scheduled:
                    with self._put_lock:
                        self._put_tasks.discard(key)
                break
            if self._core.contains_key(spec.encoded, lock=False):
                completed_keys.append(spec.encoded)
                with self._put_lock:
                    self._put_tasks.discard(key)
                continue
            obj.ref_count_up()
            pending.append((key, spec, obj))

        terminal = self._pd_tracker.register_batch(
            req_id,
            encoded_keys,
            expected_chunks=expected_chunks,
            is_last_batch=is_last_batch,
            completed_keys=completed_keys,
        )

        if conflict is not None or terminal.done():
            for key, _spec, obj in pending:
                obj.ref_count_down()
                with self._put_lock:
                    self._put_tasks.discard(key)
            if conflict is not None:
                self._pd_tracker.fail_request(
                    req_id,
                    RuntimeError(
                        "storage P/D refuses an in-flight dedup dependency for key "
                        f"{conflict}"
                    ),
                )
            return [terminal]

        if on_complete_callback is not None:
            callback_keys = list(keys)

            def complete_callbacks(done: Future) -> None:
                try:
                    done.result()
                except BaseException:
                    return
                for key in callback_keys:
                    try:
                        on_complete_callback(key)
                    except Exception as exc:
                        logger.warning(
                            "on_complete_callback failed for key %s: %s",
                            key,
                            exc,
                        )

            terminal.add_done_callback(complete_callbacks)

        if not pending:
            return [terminal]

        loop = self.loop
        if loop is None:
            self._pd_tracker.fail_request(
                req_id,
                RuntimeError("RustRawBlockBackend requires an asyncio event loop"),
            )
            return [terminal]
        coro = self._submit_pd_put_many(req_id, pending)
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as exc:
            coro.close()
            for key, _spec, obj in pending:
                obj.ref_count_down()
                with self._put_lock:
                    self._put_tasks.discard(key)
            self._pd_tracker.fail_request(req_id, exc)
        return [terminal]

    async def _submit_pd_put_many(
        self,
        req_id: str,
        pending: Sequence[tuple[CacheEngineKey, RawBlockKeySpec, MemoryObj]],
    ) -> None:
        """Persist a P/D batch and report only whole-batch success."""
        specs = [item[1] for item in pending]
        memory_objs = [item[2] for item in pending]
        io_task, io_finished = self._start_put_many(specs, memory_objs)
        try:
            put_result = await asyncio.shield(io_task)
            if len(put_result.results) != len(pending) or not all(put_result.results):
                failed = [
                    spec.encoded
                    for spec, ok in zip(specs, put_result.results, strict=False)
                    if not ok
                ]
                raise RuntimeError(
                    "storage P/D failed to persist request keys: "
                    + ", ".join(failed or ["unknown completion mismatch"])
                )
            assert self._pd_tracker is not None
            self._pd_tracker.complete_batch(
                req_id,
                [spec.encoded for spec in specs],
            )
        except BaseException as exc:
            assert self._pd_tracker is not None
            self._pd_tracker.fail_request(req_id, exc)
        finally:
            with self._put_lock:
                for key, _spec, _obj in pending:
                    self._put_tasks.discard(key)
            self._settle_put_owners(pending, io_task, io_finished)

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
            if self._publish_after_put:
                await asyncio.to_thread(self._core.publish_index)
            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning("on_complete_callback failed for key %s: %s", key, e)
        finally:
            with self._put_lock:
                self._put_tasks.discard(key)
            # The same question as the batched paths: a cancelled await has
            # not stopped the thread, so the buffer is not this coroutine's
            # to release until it has.
            self._settle_put_owners(pending, io_task, io_finished)

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
            if self._publish_after_put:
                await asyncio.to_thread(self._core.publish_index)
            if on_complete_callback is not None:
                for key in keys:
                    try:
                        on_complete_callback(key)
                    except Exception as e:
                        logger.warning(
                            "on_complete_callback failed for key %s: %s", key, e
                        )
        finally:
            with self._put_lock:
                for key, _spec, _obj in pending:
                    self._put_tasks.discard(key)
            self._settle_put_owners(pending, io_task, io_finished)

    def _observed_key_derivation(
        self, key_namespace: str
    ) -> RawBlockDerivationDescriptor:
        """Read this process's actual key derivation, not its intent.

        The configured algorithm is what was asked for; the token database's
        resolved function and chain root are what the keys are actually built
        from, and the two differ when a named algorithm silently falls back.
        Recording the request rather than the result would put a descriptor
        on the device that does not describe it.

        The chain root is a module global that building a token database
        sets, so it is read by building one -- twice. Measured: with
        ``PYTHONHASHSEED`` unset, every construction produces a *different*
        root, because the value is derived through the interpreter's
        randomized hash of fresh input. A root like that describes nothing:
        the engine that recorded it will not reproduce it, and two engines
        in one process refuse each other. That is refused here, with the
        variable named, rather than written to a device for a reader to
        discover as a mismatch.
        """
        # First Party
        from lmcache.v1 import token_database as token_database_module

        requested = str(getattr(self.config, "pre_caching_hash_algorithm", "") or "")
        implementation = "unresolved"
        first_root = str(token_database_module.NONE_HASH)
        second_root = first_root
        try:
            database = token_database_module.ChunkedTokenDatabase(
                self.config, self.metadata
            )
            resolved = getattr(database, "hash_func", None)
            implementation = (
                f"{getattr(resolved, '__module__', '?')}."
                f"{getattr(resolved, '__name__', repr(resolved))}"
            )
            first_root = str(token_database_module.NONE_HASH)
            token_database_module.ChunkedTokenDatabase(self.config, self.metadata)
            second_root = str(token_database_module.NONE_HASH)
        except Exception:
            logger.warning(
                "Raw-block storage P/D could not resolve the effective hash "
                "function; recording it as unresolved, which will not match "
                "an engine that did resolve one"
            )
        if first_root != second_root:
            raise ValueError(
                "raw-block storage P/D cannot describe this process's key "
                f"derivation: the chain root moved from {first_root} to "
                f"{second_root} between two token databases built from the "
                "same configuration, so it is not a property of the "
                "configuration and no two engines can agree on it. Set "
                "PYTHONHASHSEED to the same value on every node, or "
                "configure a deterministic pre-caching hash algorithm."
            )
        return RawBlockDerivationDescriptor(
            hash_algorithm=requested,
            hash_implementation=implementation,
            hash_seed=str(os.environ.get("PYTHONHASHSEED", "")),
            chain_root=first_root,
            key_namespace=key_namespace,
        )

    def _build_ack_receiver(
        self, extra: Mapping[str, Any]
    ) -> Optional[StoragePDAckServer]:
        """Answer read acknowledgements, or say what not answering costs.

        A port of zero means the operator has not configured one. That is
        allowed for a diagnostic run, and it is stated rather than assumed,
        because the consequence is that no extent this writer publishes is
        ever reclaimed. Where reuse is required it is an initialization
        error instead: a writer that cannot be acknowledged cannot recycle,
        and discovering that from a slow leak is worse than not starting.

        The bind address and the advertised address are separate. A
        wildcard says where to listen and is not an address a consumer can
        reply to.
        """
        host = str(extra.get("rust_raw_block.ack_listen_host", "0.0.0.0") or "")
        advertise = str(extra.get("rust_raw_block.ack_advertise_host", "") or "") or (
            "127.0.0.1" if host in ("0.0.0.0", "::", "*") else host
        )
        base_port = int(extra.get("rust_raw_block.ack_listen_port", 0) or 0)
        reuse_required = bool(extra.get("rust_raw_block.require_extent_reuse", False))
        if base_port <= 0:
            if reuse_required:
                raise ValueError(
                    "raw-block storage P/D requires extent reuse, so it "
                    "needs an acknowledgement port: set "
                    "rust_raw_block.ack_listen_port"
                )
            logger.warning(
                "Raw-block storage P/D has no acknowledgement listener "
                "(rust_raw_block.ack_listen_port is unset). Published "
                "extents stay leased for the life of this writer."
            )
            return None
        # One port per rank, so tensor-parallel writers on one host do not
        # contend for a single listener.
        port = base_port + self._ack_tp_rank
        try:
            server = StoragePDAckServer(
                self._answer_read_ack,
                claim_handler=self._answer_read_claim,
                unread_handler=self._answer_unread_publication,
                bind_host=host,
                port=port,
                advertise_host=advertise,
            )
        except Exception:
            if reuse_required:
                raise
            logger.exception(
                "Raw-block storage P/D could not listen for acknowledgements "
                "on %s:%d; published extents will stay leased",
                host,
                port,
            )
            return None
        logger.info(
            "Raw-block storage P/D answers read acknowledgements on %s (bound on %s)",
            server.endpoint,
            server.bind_endpoint,
        )
        return server

    def _answer_read_claim(
        self, request: StoragePDClaimRequest
    ) -> StoragePDClaimAnswer:
        """Say whether this consumer may read the publication it names.

        Asked before any bytes move, so this is what settles which
        incarnation will be allowed to acknowledge. Everything in the
        request came off a network; the tracker checks it against what this
        writer is actually holding and grants nothing it cannot match.
        """
        if self._pd_tracker is None:
            return StoragePDClaimAnswer(False, "this engine holds no publications")
        read = request.read
        if self._pd_session_id and request.session_id != self._pd_session_id:
            return StoragePDClaimAnswer(
                False, "the claim names another producer/consumer session"
            )
        outcome = self._pd_tracker.claim_read(
            ReadAckIdentity(
                req_id=read.req_id,
                consumer_instance_id=read.consumer_instance_id,
                tp_rank=read.tp_rank,
                writer_epoch=read.writer_epoch,
                checkpoint_seq=read.checkpoint_seq,
                manifest_digest=read.manifest_digest,
            ),
            expected_writer_epoch=self._core.writer_epoch,
            expected_tp_rank=self._ack_tp_rank,
            session_id=request.session_id,
        )
        logger.debug(
            "Raw-block storage P/D read claim for %s by %s: granted=%s %s",
            read.req_id,
            read.consumer_instance_id,
            outcome.granted,
            outcome.reason,
        )
        return StoragePDClaimAnswer(outcome.granted, outcome.reason, outcome.final)

    def _answer_unread_publication(
        self, request: StoragePDUnreadRequest
    ) -> StoragePDUnreadAnswer:
        """Resolve a publication the caller says was never given a reader.

        Everything in the request came off a network. What makes it safe to
        honour is checked here rather than trusted: the tracker releases
        nothing a consumer claimed, so a caller that is wrong about the
        reader assignment cannot reclaim an extent somebody is reading.
        """
        if self._pd_tracker is None:
            return StoragePDUnreadAnswer(False, "this engine holds no publications")
        status = request.status
        if self._pd_session_id and request.session_id != self._pd_session_id:
            return StoragePDUnreadAnswer(
                False, "this names another producer/consumer session"
            )
        if status.tp_rank != self._ack_tp_rank:
            return StoragePDUnreadAnswer(
                False, "this names another tensor-parallel rank"
            )
        try:
            receipt = status.publication_receipt()
        except ValueError as exc:
            return StoragePDUnreadAnswer(False, str(exc))
        outcome = self._pd_tracker.release_unread(
            status.req_id,
            receipt,
            expected_writer_epoch=self._core.writer_epoch,
            reason=request.reason,
        )
        return StoragePDUnreadAnswer(outcome.released, outcome.reason)

    def release_unread_publication(
        self,
        req_id: str,
        receipt: RawBlockPublicationReceipt,
        *,
        reason: str = "",
    ) -> bool:
        """Resolve one of this engine's own publications that has no reader.

        For the configuration that runs without a consumer at all: nobody
        was ever told about these publications, so nothing will acknowledge
        them, and this engine can say so about itself.
        """
        if self._pd_tracker is None:
            return False
        return self._pd_tracker.release_unread(
            req_id,
            receipt,
            expected_writer_epoch=self._core.writer_epoch,
            reason=reason,
        ).released

    def _answer_read_ack(self, request: StoragePDAckRequest) -> tuple[str, str]:
        """Apply one acknowledgement and say what happened.

        Everything in the request came off a network. The tracker re-checks
        it against what this writer actually published and releases nothing
        it cannot match, so this routes and reports. The answer is what the
        consumer retires its obligation on, which is why it is produced
        after the release rather than alongside it.
        """
        if self._pd_tracker is None:
            return ACK_REJECTED, "this engine holds no leases"
        ack = request.ack
        if self._pd_session_id and request.session_id != self._pd_session_id:
            return (
                ACK_REJECTED,
                "the acknowledgement names another producer/consumer session",
            )
        outcome = self._pd_tracker.apply_read_ack(
            ReadAckIdentity(
                req_id=ack.req_id,
                consumer_instance_id=ack.consumer_instance_id,
                tp_rank=ack.tp_rank,
                writer_epoch=ack.writer_epoch,
                checkpoint_seq=ack.checkpoint_seq,
                manifest_digest=ack.manifest_digest,
            ),
            expected_writer_epoch=self._core.writer_epoch,
            expected_tp_rank=self._ack_tp_rank,
            session_id=request.session_id,
        )
        logger.debug(
            "Raw-block storage P/D read ack for %s: %s", ack.req_id, outcome.value
        )
        if outcome is ReadAckOutcome.UNRESOLVED:
            return ACK_UNRESOLVED, "this writer could not establish the outcome"
        return outcome.value, ""

    def ack_endpoint(self) -> str:
        """Where a consumer should send this writer's acknowledgements."""
        return self._ack_receiver.endpoint if self._ack_receiver else ""

    def live_lease_count(self) -> int:
        """Count extents held pending acknowledgement."""
        return self._pd_tracker.live_lease_count() if self._pd_tracker else 0

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
                try:
                    self._pending_put_owners.remove(owners)
                except ValueError:  # pragma: no cover - settled already
                    return
            if not io_finished.is_set():
                # The task ended without its thread ending. Nothing here can
                # say what the device is doing with these buffers.
                self._quarantined_objs.extend(owners)
                return
            self._release_put_owners(pending)

        io_task.add_done_callback(_settle)

    def _batched_get_prefix(
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
            # engine skips the store, exactly like the P/D buffer does.
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
        # Seal admission before waiting for anything. A wait that runs while
        # new work is still being taken on is a wait with no end, and the
        # only reason to wait is to reach a state nothing can leave again.
        self._sealed = True

        deadline = time.monotonic() + 10.0
        while True:
            with self._put_lock:
                pending = len(self._put_tasks)
            if pending == 0 or time.monotonic() >= deadline:
                break
            time.sleep(0.01)

        if self._closed_once:
            # Running the rest again is not a repeat of a no-op. The core has
            # no device handle by then, and asking it about the device's
            # health goes through an accessor that opens a new writable one
            # on the same path -- which then gets closed along with a second
            # release of the allocator.
            logger.warning("Raw-block backend close was already run")
            return
        self._closed_once = True

        # The control handler reaches the core -- releasing a hold unlocks
        # keys through it -- so it stops first. A timed join that returns is
        # not proof a handler stopped touching the core, and the server says
        # which happened; an unconfirmed stop means the core is destroyed
        # under something that may still be using it, so nothing after this
        # can be released on evidence.
        control_quiesced = True
        if self._ack_receiver is not None:
            control_quiesced = self._ack_receiver.close()

        # Publication runs on the tracker's own thread and reaches the core,
        # so it stops here -- while the native worker is still alive, and
        # before the core is closed underneath it. A publication queued
        # before this point and started after would otherwise open a fresh
        # device and write an index into it. The core refuses a publication
        # once it is shutting down, which is the backstop; stopping the
        # thread is what makes the refusal unnecessary.
        publication_quiesced = True
        if self._pd_tracker is not None:
            try:
                publication_quiesced = self._pd_tracker.close()
            except Exception as e:
                publication_quiesced = False
                logger.error("Raw-block P/D tracker close raised: %s", e)

        # Everything below either releases a resource the device may still be
        # using or asks a question that could reopen the device, so the state
        # is sampled once, first. A deadline is not a fence: leaving that loop
        # with work still counted says the wait gave up, not that the device
        # did.
        with self._put_lock:
            retained_batches = len(self._pending_put_owners)
        unknown = (
            pending > 0
            or retained_batches > 0
            or bool(self._quarantined_objs)
            or not control_quiesced
            or not publication_quiesced
            or self._outcome_is_unknown()
        )

        if not control_quiesced or not publication_quiesced:
            # Something that reaches the core could not be confirmed
            # stopped. Closing the core now destroys it underneath a live
            # handler, and retaining the memory afterwards does not undo a
            # handler that already touched a device freed beneath it. So
            # this teardown does not complete: the whole graph is kept, the
            # core included, and nothing is released.
            self._retain_whole_graph(retained_batches)
            return

        # The local close comes next, because its result is what says
        # whether the memory behind this device is free. Asking afterwards is
        # asking a device that no longer exists.
        try:
            outcome = self._core.close()
        except Exception as e:
            unknown = True
            outcome = None
            logger.error("Raw-block core close raised: %s", e)
        if outcome is not None and not outcome.may_release_backing_resources:
            unknown = True

        if self._pd_tracker is not None:
            # Two different decisions, and only the first is ours to make
            # here: local quiescence says the memory behind this device is
            # free, and says nothing about a reader elsewhere still holding
            # a lease. So the tracker's own close above kept every hold, and
            # releasing them needs the operator's assertion that the whole
            # group has stopped -- which is a statement about other machines
            # that nothing in this process can observe.
            if self._pd_group_quiesced_teardown and not unknown:
                released = self._pd_tracker.release_quiesced_leases()
                logger.warning(
                    "Raw-block storage P/D released %d lease(s) on an "
                    "operator-declared quiesced teardown; this is correct "
                    "only if every engine that could read this namespace "
                    "has stopped.",
                    released,
                )

        if unknown:
            self._retain_whole_graph(retained_batches)
            return
        if self._gpu_allocator is None:
            return
        close_gpu_allocator = getattr(self._gpu_allocator, "close", None)
        if callable(close_gpu_allocator):
            close_gpu_allocator()

    def _retain_whole_graph(self, retained_batches: int) -> None:
        """Keep everything this teardown could not account for, and say so.

        Closing the allocator calls os.close() on every exported dma-buf and
        lets the arena behind it be reused. A command may still be landing in
        it, so the allocator, its exports and the withheld buffers are kept
        for the life of the process -- and kept referenced here, so no
        finalizer reaches close() either.

        One owner for the whole graph, the core included: the native engine
        is retaining owners of its own, and dropping the core runs the
        destructor that frees them. Two retentions that cannot see each other
        are one retention.
        """
        _RETAINED_AFTER_UNKNOWN_OUTCOME.append(
            (
                self._core,
                self._gpu_allocator,
                self._quarantined_objs,
                self._pending_put_owners,
                self._pd_tracker,
            )
        )
        logger.error(
            "Raw-block backend retaining its native device, its GPU "
            "allocator, %d exported buffer owner(s) and %d abandoned "
            "batch(es) at shutdown: what the device is doing with them "
            "could not be established.",
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
