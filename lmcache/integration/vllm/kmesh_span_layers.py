# SPDX-License-Identifier: Apache-2.0
"""Consume completed registered placements through request-owned model layers."""

# Standard
from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal
import hashlib
import json

# Third Party
import blake3
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.compute.blend.layer_source import UnsettledLayerSourceError
from lmcache.v1.gpu_connector.load_state import LayerwiseLoadState
from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.multiprocess.context_ownership import ContextIdentity
from lmcache.v1.multiprocess.kmesh_domains import WorkerFence
from lmcache.v1.multiprocess.kmesh_types import SpanOperationStatus
from lmcache.v1.platform.stream import synchronize_device
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def _blocks_digest(blocks: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps([blocks], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tokens_digest(tokens: tuple[int, ...]) -> str:
    return blake3.blake3(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class CompletedSpanPlacement:
    """Pair a server receipt with the original immutable restore arguments.

    Args:
        request_id: Exact operation request ID retained before submission.
        generation: Its positive request generation.
        destination_start: First destination position sent in that request.
        receipt: Actual completed status returned by the owning cache service.

    The worker must retain its actual engine block reservation through the model
    consumer. This value does not acquire engine blocks or establish model repair.
    """

    request_id: str
    generation: int
    destination_start: int
    receipt: SpanOperationStatus


@dataclass(frozen=True)
class RestoredSpanBatch:
    """Bind completed placement to exact worker, token and block metadata.

    Args:
        fence: This worker's actual bound receiving owner and namespace.
        context: Exact registered paged generation retained by that worker.
        token_ids: Actual complete model input, including explicit gaps.
        block_ids: Full private block table used by every restore in this batch.
        explicit_spans: Exact recovery window and restored valid ranges.
        placements: Ordered receipts for the declared spans actually restored.
            Other declared spans remain mandatory recomputation gaps.

    These checks validate cooperative receipt consistency. They cannot attest
    a hostile caller, model weights or ownership of a scheduler reservation.
    """

    fence: WorkerFence
    context: ContextIdentity
    token_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    explicit_spans: ExplicitTokenSpans
    placements: tuple[CompletedSpanPlacement, ...]

    def __post_init__(self) -> None:
        if (
            type(self.token_ids) is not tuple
            or type(self.block_ids) is not tuple
            or type(self.placements) is not tuple
        ):
            raise ValueError("restored batch metadata must be immutable tuples")
        self.explicit_spans.validate_token_count(len(self.token_ids))
        if (
            not self.token_ids
            or any(type(t) is not int or not 0 <= t < 2**63 for t in self.token_ids)
            or not self.block_ids
            or any(type(b) is not int or b < 0 for b in self.block_ids)
            or len(set(self.block_ids)) != len(self.block_ids)
            or self.context.owner_session != self.fence.session_id
            or self.fence.lease.world_size != 1
            or self.fence.rank != 0
            or self.fence.lease.domain.semantics.key_schema != "kmesh-span-v1"
            or not self.placements
            or len(self.placements) > len(self.explicit_spans.spans)
        ):
            raise ValueError("invalid restored batch geometry or owner")
        operations: set[str] = set()
        extents = {span.start: span for span in self.explicit_spans.spans}
        previous_end = -1
        for placement in self.placements:
            extent = extents.get(placement.destination_start)
            if extent is None or extent.start < previous_end:
                raise ValueError("restore does not name an ordered declared span")
            receipt, source = placement.receipt, placement.receipt.span
            if (
                type(placement.generation) is not int
                or placement.generation <= 0
                or type(receipt.generation) is not int
                or type(placement.destination_start) is not int
                or not placement.request_id
                or receipt.fence != self.fence
                or receipt.context != self.context
                or receipt.request_id != placement.request_id
                or receipt.generation != placement.generation
                or receipt.kind != "restore"
                or receipt.blocks_digest != _blocks_digest(self.block_ids)
                or receipt.state not in ("loaded", "loaded-needs-repair")
                or receipt.error
                or not receipt.operation_id
                or receipt.operation_id in operations
                or source is None
                or type(source.token_count) is not int
                or type(source.source_start) is not int
                or source.source_start < 0
                or source.namespace != self.fence.lease.domain.namespace
                or source.layout_digest
                != self.fence.lease.domain.semantics.layout_digest
                or source.rope_digest != self.fence.lease.domain.semantics.rope_digest
                or placement.destination_start != extent.start
                or source.token_count != extent.end - extent.start
                or source.token_digest
                != _tokens_digest(self.token_ids[extent.start : extent.end])
            ):
                raise ValueError("restore receipt differs from the bound model input")
            operations.add(receipt.operation_id)
            previous_end = extent.end

    @property
    def valid_spans(self) -> tuple[TokenSpan, ...]:
        """Return only extents with completed placement, preserving declared gaps."""
        starts = {placement.destination_start for placement in self.placements}
        return tuple(span for span in self.explicit_spans.spans if span.start in starts)


# Plain TensorMemoryObj views have no suballocator that could reuse a failed
# load's storage. Keep their underlying tensors and cache views until real drain.
_RETAINED_LAYER_SOURCES: dict[int, "RestoredPagedLayerSource"] = {}


class RestoredPagedLayerSource:
    """Read completed paged placements and write repaired layers back in order.

    Args:
        batch: Validated receipts and immutable model input/reservation metadata.
        kv_caches: This registered worker's actual cache tensor views, retained
            with its engine block reservation for the complete consuming step.
        layout: Physical two-plane NHD or HND layout registered with the service.

    This serialized CUDA BF16 source does not apply RoPE again. Native service
    restoration already rotates cached keys. It supplies real layer inputs to
    CacheBlend, whose model still performs selection and deeper-layer repair.
    """

    def __init__(
        self,
        batch: RestoredSpanBatch,
        kv_caches: tuple[torch.Tensor, ...],
        layout: Literal["NHD", "HND"],
    ) -> None:
        fmt, normalized = normalize_kv_and_discover_format(
            list(kv_caches), EngineType.VLLM, layout_hints={"kv_layout": layout}
        )
        expected = {
            "NHD": "NL_X_TWO_NB_BS_NH_HS",
            "HND": "NL_X_TWO_NB_NH_BS_HS",
        }
        if (
            layout not in expected
            or fmt.name != expected[layout]
            or not isinstance(normalized, list)
            or not normalized
            or any(not isinstance(t, torch.Tensor) for t in normalized)
        ):
            raise ValueError("registered layer source requires two-plane NHD/HND")
        first = normalized[0]
        if any(
            t.device.type != "cuda"
            or t.device != first.device
            or t.dtype != torch.bfloat16
            or t.shape != first.shape
            or not t.is_contiguous()
            for t in normalized
        ):
            raise ValueError("registered layer views must be contiguous CUDA BF16")
        self._batch, self._caches, self._format = batch, tuple(normalized), fmt
        self._device = first.device
        self._block_size = int(first.shape[2 if layout == "NHD" else 3])
        self._blocks = int(first.shape[1])
        self._head_size = int(first.shape[-1])
        self._hidden = int(first.shape[3 if layout == "NHD" else 2]) * self._head_size
        if (
            max(batch.block_ids) >= self._blocks
            or len(batch.token_ids) > len(batch.block_ids) * self._block_size
        ):
            raise ValueError("registered model input exceeds its reserved blocks")
        self._used = False
        self._closed = False
        self._pending = False
        self._error: BaseException | None = None
        self._state: LayerwiseLoadState | None = None
        self._tensors: list[torch.Tensor] = []

    @property
    def explicit_spans(self) -> ExplicitTokenSpans:
        """Return the exact model repair window validated against receipts."""
        return self._batch.explicit_spans

    @property
    def pending(self) -> bool:
        """Report physical work still requiring an actual worker-device drain."""
        return self._pending

    @property
    def is_closed(self) -> bool:
        """Report generator retirement; pending physical work can still remain."""
        return self._closed

    def close(self) -> None:
        """Retire unused input or retry a closed generator's physical cleanup.

        Raises:
            RuntimeError: A live generator still owns its model step, or actual
                worker-device drain remains unresolved.
        """
        if not self._used:
            self._used = True
            self._closed = True
            self._caches = ()
            return
        if not self._closed:
            raise RuntimeError("a live model source still owns its worker blocks")
        self._cleanup()

    def drain_failed(self) -> None:
        """Retry cleanup of a closed failed source without another transfer.

        Raises:
            RuntimeError: The generator is live or actual device drain fails.
        """
        if not self._closed:
            raise RuntimeError("cannot drain a live model layer source")
        self._cleanup()

    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        load_state: LayerwiseLoadState,
    ) -> Generator[torch.Tensor | None, None, None]:
        """Publish one real restored input per layer, then checked writeback.

        Args:
            tokens: Actual model input, required to match the bound token IDs.
            mask: None or an all-true request mask for this bounded source.
            load_state: This model session's private layer mapping.

        Returns:
            Hit count, one model boundary per layer and final exact hit mask.

        Raises:
            ValueError: Input metadata or deferred rotation is unsupported.
            RuntimeError: Source reuse, native work or checked cleanup failed.
        """
        if self._used:
            raise RuntimeError("registered model layer source is single-use")
        self._used = True
        self._state = load_state
        _RETAINED_LAYER_SOURCES[id(self)] = self
        try:
            self._pending = True
            if (
                tokens.ndim != 1
                or tokens.dtype != torch.int64
                or tuple(tokens.cpu().tolist()) != self._batch.token_ids
                or load_state.defer_position_recovery
                or (
                    mask is not None
                    and (
                        mask.dtype != torch.bool
                        or mask.ndim != 1
                        or len(mask) != len(tokens)
                        or not bool(mask.all().item())
                    )
                )
            ):
                raise ValueError("model input differs from the restored span batch")
            load_state.begin()
            window = self.explicit_spans.cached_span.end
            positions = [
                p
                for span in self._batch.valid_spans
                for p in range(span.start, span.end)
            ]
            slots = [
                self._batch.block_ids[p // self._block_size] * self._block_size
                + p % self._block_size
                for p in range(window)
            ]
            full_slots = torch.tensor(slots, dtype=torch.int64, device=self._device)
            hit_indices = torch.tensor(
                positions, dtype=torch.int64, device=self._device
            )
            hit_slots = full_slots[hit_indices]
            pointers = torch.tensor(
                [t.data_ptr() for t in self._caches],
                dtype=torch.int64,
                device=self._device,
            )
            buffer = torch.empty(
                (2, window, self._hidden), dtype=torch.bfloat16, device=self._device
            )
            packed = torch.empty(
                (2, 1, len(positions), self._hidden),
                dtype=torch.bfloat16,
                device=self._device,
            )
            self._tensors = [
                full_slots,
                hit_indices,
                hit_slots,
                pointers,
                buffer,
                packed,
            ]
            obj = TensorMemoryObj(
                buffer.view(torch.uint8).view(-1),
                MemoryObjMetadata(
                    buffer.shape,
                    buffer.dtype,
                    buffer.data_ptr(),
                    buffer.nbytes,
                    1,
                    fmt=MemoryFormat.KV_2TD,
                ),
                None,
            )
            load_state.own_buffer(obj)
            hit_mask = torch.zeros(len(tokens), dtype=torch.bool)
            hit_mask[positions] = True
            load_state.gap_positions = torch.where(~hit_mask[:window])[0].to(
                self._device
            )
            self._drain()
            yield torch.tensor(len(positions))
            for layer in range(len(self._caches)):
                self._pending = True
                device_ops.multi_layer_kv_transfer(
                    packed,
                    pointers[layer : layer + 1],
                    hit_slots,
                    self._device,
                    self._blocks * self._block_size,
                    lmcache_native.TransferDirection.D2H,
                    self._format,
                    block_size=self._block_size,
                    head_size=self._head_size,
                )
                buffer.zero_()
                buffer.index_copy_(1, hit_indices, packed[:, 0])
                self._drain()
                load_state.buffer_mapping[layer] = obj
                # The model may queue reads or repair writes while suspended.
                self._pending = True
                yield None
                device_ops.multi_layer_kv_transfer(
                    buffer.unsqueeze(1),
                    pointers[layer : layer + 1],
                    full_slots,
                    self._device,
                    self._blocks * self._block_size,
                    lmcache_native.TransferDirection.H2D,
                    self._format,
                    block_size=self._block_size,
                    head_size=self._head_size,
                )
                self._drain()
                del load_state.buffer_mapping[layer]
            yield hit_mask
        except BaseException as error:
            self._error = error
            raise
        finally:
            self._closed = True
            self._cleanup()

    def _drain(self) -> None:
        if self._pending:
            synchronize_device(self._device)
            self._pending = False

    def _cleanup(self) -> None:
        try:
            self._drain()
        except RuntimeError as error:
            logger.exception("registered model layer cleanup remains unresolved")
            raise UnsettledLayerSourceError(
                f"registered layer cleanup remains unresolved: {error}",
                self.drain_failed,
            ) from error
        if self._state is not None:
            self._state.release()
        self._tensors.clear()
        self._caches = ()
        self._error = None
        _RETAINED_LAYER_SOURCES.pop(id(self), None)
