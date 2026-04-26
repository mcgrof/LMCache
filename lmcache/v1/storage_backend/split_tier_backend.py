# SPDX-License-Identifier: Apache-2.0
"""SplitTierStorageBackend — `SplitTierStore` as a `StorageBackendInterface`.

Wraps the codec-layer `SplitTierStore` (K in CPU pinned memory,
V FP8 on NVMe) so the LMCache storage manager can register it
alongside `LocalDiskBackend`, `GdsBackend`, etc.

Design constraints, tracked from the original LMCache asymmetric
plan:

- Storage compression, native HBM layout, and tier placement are
  separate config axes.  This adapter owns *tier placement*; the
  caller chooses the codec (asym_k16_v8_e4m3 in our case) at the
  serde layer above.
- The `MemoryObj` arriving at `batched_submit_put_task` must be
  one of:
    a) A `BytesBufferMemoryObj` carrying an already-encoded
       asymmetric blob (storage-only path).  The adapter parses
       it and re-issues a structured `SplitTierStore.put`.
    b) A `TensorMemoryObj` with a `[2, ...]` FP16/BF16 KV tensor
       (the storage manager's normal output).  The adapter
       splits K and V, encodes through the codec, then calls
       `SplitTierStore.put`.
    c) Anything else — error loudly; do not silently fall back.

This adapter does NOT subclass `AllocatorBackendInterface` — it
delegates allocator queries to a constructor-provided
`LocalCPUBackend` like `LocalDiskBackend` does.

Caveat: this is the Phase 5 wiring committed alongside the codec
work.  Live integration with the storage manager (registry add,
serde-to-backend handshake, eviction policy) needs an end-to-end
test against a running LMCache engine on a GPU pod.  The pure
contract-shape tests live in `tests/v1/kv_codec/test_split_tier_backend.py`.
"""

# Standard
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    CodecHashes,
    PlacementPolicy,
    SplitTierByteCounts,
    SplitTierStore,
)
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObj,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface


logger = init_logger(__name__)


class SplitTierStorageBackend(StorageBackendInterface):
    """Storage backend that uses the K-CPU/V-NVMe split-tier
    placement under the hood.

    Args:
        root: Filesystem root for chunk directories.
        local_cpu_backend: The engine's CPU memory backend; used
            for `get_allocator_backend()` queries (this adapter
            doesn't manage allocator state itself).
        codec: Optional codec instance; defaults to
            `AsymK16V8Codec()` with per-tensor scales.
        policy: One of the `PlacementPolicy` values; default is
            `SPLIT_K_CPU_V_NVME` (the headline mode).
        cpu_pinned_budget_bytes: Max CPU memory the K-store may
            use.  Exceeding this triggers `on_cpu_full` behavior.
        on_cpu_full: 'raise' or 'demote_k_to_nvme'.
        expected_hashes: `CodecHashes` for cross-config gating;
            optional but recommended in production.
        dst_device: Where retrieved tensors should land.
            Default 'cuda'; falls back to 'cpu' if no GPU.
    """

    def __init__(
        self,
        root: Union[str, Path],
        local_cpu_backend: Any,
        *,
        codec: Optional[AsymK16V8Codec] = None,
        policy: PlacementPolicy = PlacementPolicy.SPLIT_K_CPU_V_NVME,
        cpu_pinned_budget_bytes: int = 16 * 1024 * 1024 * 1024,
        on_cpu_full: str = "raise",
        expected_hashes: Optional[CodecHashes] = None,
        dst_device: str = "cuda",
    ):
        super().__init__(
            dst_device if torch.cuda.is_available() else "cpu"
        )
        self.local_cpu_backend = local_cpu_backend
        self.policy = policy
        self.codec = codec or AsymK16V8Codec()
        self.store = SplitTierStore(
            root=Path(root),
            codec=self.codec,
            policy=policy,
            cpu_pinned_budget_bytes=cpu_pinned_budget_bytes,
            on_cpu_full=on_cpu_full,
            expected_hashes=expected_hashes,
        )
        # Track in-flight put tasks for `exists_in_put_tasks`.
        self._inflight: set = set()
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="split-tier-put"
        )
        self._pinned: set = set()
        # Optional: aggregated byte counts across all puts/gets,
        # exposed via `get_byte_counts()` for benchmarks.
        self._cumulative_writes = SplitTierByteCounts()
        self._cumulative_reads = SplitTierByteCounts()

    # -------- contract: contains / put-tasks --------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # SplitTierStore stores chunks under
        # root/cache_key/layer_<L>/chunk_<C>/.  A `contains` query
        # at the LMCache level identifies a chunk by the engine
        # CacheEngineKey; we use the key string as `cache_key`.
        # The presence test is the meta.bin file for layer_id=0,
        # chunk_id=0 (a single-chunk-per-key model is the v1
        # simplification — multi-chunk keys are Phase 6 work).
        cache_key = key.to_string()
        layout = self.store.root / cache_key / "layer_000" / "chunk_000000"
        present = (layout / "meta.bin").exists()
        if present and pin:
            self._pinned.add(cache_key)
        return present

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return key.to_string() in self._inflight

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[List[Future]]:
        if len(keys) != len(objs):
            raise ValueError(
                f"SplitTierStorageBackend: keys/objs length mismatch "
                f"({len(keys)} vs {len(objs)})"
            )

        futures: List[Future] = []
        for key, obj in zip(keys, objs):
            cache_str = key.to_string()
            self._inflight.add(cache_str)
            f = self._executor.submit(
                self._do_put, key, obj, on_complete_callback,
            )
            futures.append(f)
        return futures

    def _do_put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> None:
        try:
            k, v = self._extract_kv(memory_obj)
            counts = self.store.put(
                cache_key=key.to_string(),
                layer_id=0,
                chunk_id=0,
                k=k,
                v=v,
            )
            # Aggregate.
            self._cumulative_writes.nvme_bytes += counts.nvme_bytes
            self._cumulative_writes.cpu_bytes += counts.cpu_bytes
            self._cumulative_writes.meta_bytes += counts.meta_bytes
            memory_obj.ref_count_down()
        except Exception:
            logger.exception(
                "SplitTierStorageBackend put failed for key %s",
                key.to_string(),
            )
            raise
        finally:
            self._inflight.discard(key.to_string())
            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception:
                    logger.exception(
                        "on_complete_callback raised for key %s",
                        key.to_string(),
                    )

    @staticmethod
    def _extract_kv(memory_obj: MemoryObj):
        """Pull (K, V) out of a MemoryObj produced by the storage
        manager.

        Two shapes are supported:
        a) TensorMemoryObj with a [2, ...] tensor: tensor[0] is K,
           tensor[1] is V.
        b) MemoryObj with `.asym_view`: native_asym path (Phase 4).

        Anything else raises ValueError.
        """
        asym_view = getattr(memory_obj, "asym_view", None)
        if asym_view is not None:
            # native_asym — V is already FP8.  We don't currently
            # plumb precomputed_v_quant through SplitTierStore.put;
            # falling back to FP16 V (which the codec then
            # quantizes again) here would re-quantize and ruin the
            # bit-exact native_asym story.  Reject loudly so we
            # don't silently regress to that path.
            raise NotImplementedError(
                "SplitTierStorageBackend does not yet plumb "
                "native_asym AsymKVView into the SplitTierStore.put "
                "path; that integration is part of the connector "
                "+ store-API extension still outstanding.  For now, "
                "use the storage_only mode (TensorMemoryObj input)."
            )
        tensor = getattr(memory_obj, "tensor", None)
        if tensor is None:
            raise ValueError(
                "SplitTierStorageBackend: input MemoryObj must "
                "expose a [2, ...] tensor (storage_only path) or "
                "an asym_view (native_asym path); got "
                f"{type(memory_obj).__name__}"
            )
        if tensor.ndim < 2 or tensor.shape[0] != 2:
            raise ValueError(
                f"SplitTierStorageBackend: expected leading dim 2 "
                f"(K/V split), got tensor of shape "
                f"{tuple(tensor.shape)}"
            )
        return tensor[0].contiguous(), tensor[1].contiguous()

    # -------- contract: get_blocking --------

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """Reassemble the stored chunk into a `BytesBufferMemoryObj`.

        Returning a BytesBufferMemoryObj keeps the contract simple:
        the caller (typically the asym serde's deserializer) handles
        decoding back to a tensor.  This avoids re-implementing the
        codec materialization here.
        """
        cache_key = key.to_string()
        try:
            encoded, counts = self.store.get(
                cache_key=cache_key, layer_id=0, chunk_id=0
            )
        except FileNotFoundError:
            return None

        # Aggregate read counts.
        self._cumulative_reads.nvme_bytes += counts.nvme_bytes
        self._cumulative_reads.cpu_bytes += counts.cpu_bytes
        self._cumulative_reads.meta_bytes += counts.meta_bytes

        blob = self.codec.to_bytes(encoded)
        # First Party
        from lmcache.v1.memory_management import MemoryObjMetadata
        bytes_meta = MemoryObjMetadata(
            shape=torch.Size([len(blob), 0, 0, 0]),
            dtype=None,
            address=0,
            phy_size=0,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.BINARY_BUFFER,
            shapes=[torch.Size([2, *encoded.scale_shape] if encoded.scale_shape else [2])],
            dtypes=[encoded.k_dtype, encoded.v_dtype],
        )
        return BytesBufferMemoryObj(raw_bytes=blob, metadata=bytes_meta)

    # -------- contract: pin / unpin / remove --------

    def pin(self, key: CacheEngineKey) -> bool:
        if not self.contains(key):
            return False
        self._pinned.add(key.to_string())
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        cache_key = key.to_string()
        if cache_key in self._pinned:
            self._pinned.remove(cache_key)
            return True
        return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        cache_key = key.to_string()
        if cache_key in self._pinned and not force:
            return False
        layout = self.store.root / cache_key
        if not layout.exists():
            return False
        # Standard
        import shutil

        try:
            shutil.rmtree(layout)
        except Exception:
            logger.exception(
                "SplitTierStorageBackend remove failed for key %s",
                cache_key,
            )
            return False
        # Also clear K from CPU store.
        for layer_id in range(64):  # a generous upper bound
            self.store._k_store.evict(cache_key, layer_id, 0)
        self._pinned.discard(cache_key)
        return True

    def get_allocator_backend(self) -> Any:
        return self.local_cpu_backend

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    # -------- adapter-specific accessors (for benchmarks) --------

    def get_byte_counts(self) -> tuple[SplitTierByteCounts, SplitTierByteCounts]:
        """Returns (cumulative_writes, cumulative_reads).  Useful
        for benchmarks that want to report NVMe-vs-CPU byte
        attribution at engine level."""
        return self._cumulative_writes, self._cumulative_reads
