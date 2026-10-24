# SPDX-License-Identifier: Apache-2.0
"""Own one cooperative worker's registered span operations and raw exports."""

# Standard
from dataclasses import dataclass, field
from typing import Literal, TypeVar
import hashlib
import json
import re
import uuid

# Third Party
import blake3
import torch

# First Party
from lmcache import torch_dev
from lmcache.integration.vllm.kmesh_span_layers import (
    CompletedSpanPlacement,
    RestoredPagedLayerSource,
    RestoredSpanBatch,
)
from lmcache.integration.vllm.kmesh_span_layout import resolve_registered_paged_shape
from lmcache.logging import init_logger
from lmcache.v1.compute.blend.layer_source import UnsettledLayerSourceError
from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper
from lmcache.v1.multiprocess.kmesh_domains import (
    SemanticIdentity,
    ServerDescriptor,
    WorkerFence,
)
from lmcache.v1.multiprocess.kmesh_types import (
    ManagedRawRegistration,
    ManagedRawRelease,
    RegisteredRuntime,
    SpanHandle,
    SpanOperationStatus,
)
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.devices.cuda.ipc_wrapper import RawCudaIPCWrapper
from lmcache.v1.platform.stream import synchronize_device
from lmcache.v1.token_span import ExplicitTokenSpans

logger = init_logger(__name__)
T = TypeVar("T")


def _value(reply: tuple[T | None, str]) -> T:
    """Require the owning service's successful typed data/refusal reply."""
    value, error = reply
    if value is None or error:
        raise RuntimeError(error or "missing registered span reply")
    return value


def _sha256(value: object) -> str:
    """Hash canonical JSON using the service's block-table wire grammar."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tokens_digest(tokens: tuple[int, ...]) -> str:
    """Hash exact token content using the admitted source's wire grammar."""
    return blake3.blake3(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def _coordinates(position: int, context: str, blocks: tuple[int, ...]) -> None:
    """Reject mutable or malformed coordinates before any device work."""
    if (
        type(position) is not int
        or position < 0
        or type(context) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", context)
        or type(blocks) is not tuple
        or not blocks
        or any(type(block) is not int or block < 0 for block in blocks)
        or len(set(blocks)) != len(blocks)
    ):
        raise ValueError("invalid immutable registered span coordinates")


@dataclass(frozen=True)
class SpanWorkerArtifacts:
    """Identify verified model artifacts and separately frozen execution facts.

    Args:
        weights_digest: SHA-256 identity of actual immutable model weight files.
        config_digest: SHA-256 of resolved model configuration, including defaults.
        tokenizer_digest: SHA-256 identity of the actual tokenizer artifacts.
        execution_digest: SHA-256 of the validated bounded execution profile.

    The caller must derive these from its actual validated model and settings.
    The storage service does not load or attest the worker's weights.
    """

    weights_digest: str
    config_digest: str
    tokenizer_digest: str
    execution_digest: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (
                self.weights_digest,
                self.config_digest,
                self.tokenizer_digest,
                self.execution_digest,
            )
        ):
            raise ValueError("worker artifact identities must be lowercase SHA-256")


@dataclass
class _Exports:
    client: RequestClient
    server: ServerDescriptor
    nonce: str
    device: torch.device
    layout: Literal["NHD", "HND"]
    caches: tuple[torch.Tensor, ...]
    rotary: tuple[torch.Tensor, ...]
    wrappers: list[RawCudaIPCWrapper] = field(default_factory=list)
    events: list[object] = field(default_factory=list)
    allocation_ids: tuple[str, ...] = ()
    error: BaseException | None = None


@dataclass
class _Operation:
    request_id: str
    kind: Literal["store", "restore"]
    block_ids: tuple[int, ...]
    tokens: tuple[int, ...]
    position: int
    context_digest: str
    span: SpanHandle | None
    issued: bool = False


# A failed bootstrap may already have imported arbitrary interior raw views.
# Keep exporter references and traceback frames until matching retirement or
# process exit; an ordinary Python exception is not importer-exit evidence.
_RETAINED_EXPORTS: dict[str, _Exports] = {}


class RegisteredSpanWorker:
    """Retain actual CUDA exports and exact operation arguments for one worker.

    Use ``connect`` with the actual normalized engine views. The owning worker
    must also retain real scheduler block reservations through consuming model
    repair. This client neither allocates engine blocks nor marks tokens computed.
    The caller owns the transport and importing service process.
    """

    def __init__(
        self,
        exports: _Exports,
        registration: ManagedRawRegistration,
        runtime: RegisteredRuntime,
        fence: WorkerFence,
    ) -> None:
        self._exports = exports
        self.registration, self.runtime, self.fence = registration, runtime, fence
        self._pending: _Operation | None = None
        self._sources: list[RestoredPagedLayerSource] = []
        self._closing = False
        self._release: ManagedRawRelease | None = None

    @classmethod
    def connect(
        cls,
        client: RequestClient,
        kv_caches: tuple[torch.Tensor, ...],
        rotary_cache: torch.Tensor,
        *,
        layout: Literal["NHD", "HND"],
        model_name: str,
        is_neox_style: bool,
        artifacts: SpanWorkerArtifacts,
    ) -> "RegisteredSpanWorker":
        """Register actual raw views, ordered rotary state and one bound worker.

        Args:
            client: Actual same-host transport, retained by the caller.
            kv_caches: Contiguous physical engine cache views, with no data copy.
            rotary_cache: Actual contiguous model rotary table, retained unchanged.
            layout: Engine-provided NHD/HND ordering for separate or fused K/V.
            model_name: Actual serving model identifier.
            is_neox_style: Actual model rotary style.
            artifacts: Separately validated model and execution identities.

        Returns:
            One ready cooperative worker owning its raw exports and events.

        Raises:
            ValueError: Unsupported actual tensor geometry.
            RuntimeError: Registration failed or its outcome is uncertain. Any
                submitted exports remain retained; importer/process exit may
                be required before releasing them.
        """
        if type(kv_caches) is not tuple or not kv_caches:
            raise ValueError("registered worker requires actual engine cache views")
        first = kv_caches[0]
        geometry = resolve_registered_paged_shape(tuple(first.shape), layout)
        if (
            type(is_neox_style) is not bool
            or layout not in ("NHD", "HND")
            or any(
                t.device.type != "cuda"
                or t.device != first.device
                or t.dtype != torch.bfloat16
                or t.shape != first.shape
                or not t.is_contiguous()
                for t in kv_caches
            )
            or rotary_cache.device != first.device
            or not rotary_cache.is_contiguous()
        ):
            raise ValueError("registered worker requires contiguous CUDA BF16 views")
        server = _value(client.kmesh_get_server().result(timeout=30))
        exports = _Exports(
            client,
            server,
            uuid.uuid4().hex,
            first.device,
            layout,
            kv_caches,
            (rotary_cache,),
        )
        _RETAINED_EXPORTS[exports.nonce] = exports
        try:
            exports.wrappers = [RawCudaIPCWrapper.wrap(t) for t in kv_caches]
            cache_ipc: list[DeviceIPCWrapper] = list(exports.wrappers)
            registration = _value(
                client.kmesh_register_raw_context(
                    server.boot_id, exports.nonce, 1, cache_ipc, model_name, layout
                ).result(timeout=30)
            )
            if (
                registration.boot_id != server.boot_id
                or registration.nonce != exports.nonce
                or registration.instance_id != 1
                or registration.state != "registered"
                or set(registration.allocation_ids)
                != {w.allocation_identity for w in exports.wrappers}
            ):
                raise RuntimeError("raw registration differs from the actual exporter")
            rotary = RawCudaIPCWrapper.wrap(rotary_cache)
            exports.wrappers.append(rotary)
            exports.allocation_ids = tuple(
                sorted({w.allocation_identity for w in exports.wrappers})
            )
            backend = get_event_ipc_backend(first.device)
            event = backend.create_event(first.device)
            exports.events.append(event)
            backend.record_event(event, torch_dev.current_stream(first.device))
            head_size = geometry.head_size
            runtime = _value(
                client.kmesh_register_rotary(
                    1,
                    registration.context,
                    [rotary],
                    head_size,
                    is_neox_style,
                    [0],
                    [],
                    [head_size],
                    "",
                    backend.export_event(event, first.device),
                ).result(timeout=30)
            )
            if (
                runtime.context != registration.context
                or runtime.world_size != 1
                or runtime.model_name != model_name
                or runtime.kv_dtype != "bfloat16"
                or runtime.hash_algorithm != "blake3"
            ):
                raise RuntimeError("registered runtime differs from the actual worker")
            identity = SemanticIdentity(
                artifacts.weights_digest,
                artifacts.config_digest,
                artifacts.tokenizer_digest,
                runtime.rope_digest,
                runtime.layout_digest,
                runtime.kv_dtype,
                runtime.chunk_tokens,
                runtime.hash_algorithm,
                key_schema="kmesh-span-v1",
            )
            lease = _value(
                client.kmesh_open_registration(
                    identity, "model-" + uuid.uuid4().hex, 1, artifacts.execution_digest
                ).result(timeout=30)
            )
            if (
                lease.server != server
                or lease.domain.semantics != identity
                or lease.world_size != 1
                or lease.execution_digest != artifacts.execution_digest
            ):
                raise RuntimeError("registration lease differs from this worker")
            fence = _value(
                client.kmesh_bind_worker(
                    lease, 0, 1, runtime, identity, artifacts.execution_digest
                ).result(timeout=30)
            )
            if fence.lease != lease or fence.rank != 0 or fence.instance_id != 1:
                raise RuntimeError("bound fence differs from the actual worker")
            return cls(exports, registration, runtime, fence)
        except BaseException as error:
            exports.error = error
            logger.exception("registered worker bootstrap remains retained")
            raise

    @property
    def pending(self) -> bool:
        """Report an operation whose exact physical completion remains unknown."""
        return self._pending is not None or any(
            source.pending for source in self._sources
        )

    def block_table(self, slot_mapping: tuple[int, ...]) -> tuple[int, ...]:
        """Derive and check a prompt-origin table from actual private worker slots.

        Args:
            slot_mapping: Immutable complete model-input mapping from the engine.

        Returns:
            Nonaliased engine blocks covering those exact logical positions.

        Raises:
            ValueError: Mapping is empty, aliased, out of bounds or misaligned.
            RuntimeError: The worker has started retirement.
        """
        if self._closing:
            raise RuntimeError("registered worker is retiring")
        cache = self._exports.caches[0]
        geometry = resolve_registered_paged_shape(
            tuple(cache.shape), self._exports.layout
        )
        block_size = geometry.block_size
        if (
            type(slot_mapping) is not tuple
            or not slot_mapping
            or any(type(s) is not int or s < 0 for s in slot_mapping)
        ):
            raise ValueError("invalid immutable engine slot mapping")
        blocks = tuple(
            slot_mapping[p] // block_size
            for p in range(0, len(slot_mapping), block_size)
        )
        if (
            len(set(blocks)) != len(blocks)
            or max(blocks) >= geometry.blocks
            or any(
                slot != blocks[p // block_size] * block_size + p % block_size
                for p, slot in enumerate(slot_mapping)
            )
        ):
            raise ValueError(
                "engine slots do not describe private origin-aligned blocks"
            )
        return blocks

    def layer_source(
        self,
        token_ids: tuple[int, ...],
        block_ids: tuple[int, ...],
        explicit_spans: ExplicitTokenSpans,
        placements: tuple[CompletedSpanPlacement, ...],
    ) -> RestoredPagedLayerSource:
        """Bind completed restores to this owner's exact registered cache views.

        Args:
            token_ids: Actual model input, including recomputed private gaps.
            block_ids: Actual reserved table used by these restore operations.
            explicit_spans: Model recovery window and all declared cache spans.
            placements: Exact completed operations for the valid restored subset.

        Returns:
            One serial model-input source retained through checked model cleanup.

        Raises:
            RuntimeError: The worker is retiring or still owns earlier work.
            ValueError: Receipts or geometry differ from this model input.
        """
        if self._closing or self.pending or any(not s.is_closed for s in self._sources):
            raise RuntimeError("earlier registered model work still owns its blocks")
        batch = RestoredSpanBatch(
            self.fence,
            self.runtime.context,
            token_ids,
            block_ids,
            explicit_spans,
            placements,
        )
        source = RestoredPagedLayerSource(
            batch, self._exports.caches, self._exports.layout
        )
        self._sources.append(source)
        return source

    def store(
        self,
        token_ids: tuple[int, ...],
        source_start: int,
        context_digest: str,
        block_ids: tuple[int, ...],
    ) -> SpanHandle:
        """Store actual model slots while the caller owns their source blocks.

        Args:
            token_ids: Exact source segment content, including any partial tail.
            source_start: Its original position in the model prompt.
            context_digest: Identity of the actual preceding source context.
            block_ids: Original full private block table, retained by the worker.

        Returns:
            An immutable handle from checked native storage admission.

        Raises:
            RuntimeError: Refusal, failed admission or unresolved physical work.
        """
        _coordinates(source_start, context_digest, block_ids)
        if (
            type(token_ids) is not tuple
            or not token_ids
            or any(type(t) is not int or not 0 <= t < 2**63 for t in token_ids)
        ):
            raise ValueError("span store requires immutable valid model tokens")
        op = _Operation(
            uuid.uuid4().hex,
            "store",
            block_ids,
            token_ids,
            source_start,
            context_digest,
            None,
        )
        result = self._execute(op)
        if result.span is None:
            raise RuntimeError("completed span store returned no source")
        return result.span

    def restore(
        self,
        span: SpanHandle,
        destination_start: int,
        context_digest: str,
        block_ids: tuple[int, ...],
    ) -> CompletedSpanPlacement:
        """Restore into retained destination blocks, without claiming model repair.

        Args:
            span: Exact previously admitted immutable source handle.
            destination_start: Destination position sent to the native service.
            context_digest: Actual destination preceding-context identity.
            block_ids: Full private destination table owned through model repair.

        Returns:
            The exact completed receipt and original restore arguments.

        Raises:
            RuntimeError: Failed restoration or unresolved physical work.
        """
        _coordinates(destination_start, context_digest, block_ids)
        semantics = self.fence.lease.domain.semantics
        if (
            span.namespace != self.fence.lease.domain.namespace
            or span.layout_digest != semantics.layout_digest
            or span.rope_digest != semantics.rope_digest
        ):
            raise ValueError("source handle belongs to another registered namespace")
        op = _Operation(
            uuid.uuid4().hex,
            "restore",
            block_ids,
            (),
            destination_start,
            context_digest,
            span,
        )
        result = self._execute(op)
        return CompletedSpanPlacement(op.request_id, 1, destination_start, result)

    def drain_pending(self) -> None:
        """Resolve and physically settle exact pending work without another copy.

        Raises:
            RuntimeError: Status is absent, mismatched or still cannot drain.
        """
        op = self._pending
        if op is None:
            for source in self._sources:
                if source.pending:
                    source.drain_failed()
            return
        if not op.issued:
            synchronize_device(self._exports.device)
            self._pending = None
            self._exports.error = None
            return
        result = _value(
            self._exports.client.kmesh_span_status(self.fence, op.request_id, 1).result(
                timeout=30
            )
        )
        self._validate_status(op, result)
        if result.state in ("executing", "failed-undrained"):
            result = _value(
                self._exports.client.kmesh_drain_span(
                    self.fence, op.request_id, 1
                ).result(timeout=30)
            )
            self._validate_status(op, result)
        if result.state not in (
            "stored",
            "loaded",
            "loaded-needs-repair",
            "failed-drained",
        ):
            raise RuntimeError("registered operation still owes physical completion")
        self._pending = None
        self._exports.error = None

    def close(self) -> ManagedRawRelease:
        """Retire exact raw imports before dropping exporter tensors or events.

        Returns:
            Matching checked retirement; repeated close returns the same receipt.

        Raises:
            RuntimeError: Physical work, imports or retirement remain unresolved.
        """
        if self._release is not None:
            return self._release
        self._closing = True
        for source in self._sources:
            source.close()
        self.drain_pending()
        receipt = _value(
            self._exports.client.kmesh_retire_raw_context(
                self._exports.server.boot_id, self._exports.nonce, self.runtime.context
            ).result(timeout=30)
        )
        if (
            receipt.boot_id != self._exports.server.boot_id
            or receipt.nonce != self._exports.nonce
            or receipt.instance_id != 1
            or receipt.context != self.runtime.context
            or receipt.rotary != self.runtime.rotary
            or tuple(sorted(receipt.allocation_ids)) != self._exports.allocation_ids
        ):
            raise RuntimeError("raw retirement does not cover this exact worker")
        self._release = receipt
        self._exports.caches = ()
        self._exports.rotary = ()
        self._exports.wrappers.clear()
        self._exports.events.clear()
        self._exports.error = None
        self._sources.clear()
        _RETAINED_EXPORTS.pop(self._exports.nonce)
        return receipt

    def _execute(self, op: _Operation) -> SpanOperationStatus:
        """Retain submitted arguments until the exact operation proves drain."""
        if self._closing or self.pending or any(not s.is_closed for s in self._sources):
            raise RuntimeError(
                "registered worker is closing or still owns pending work"
            )
        if (
            _value(self._exports.client.kmesh_heartbeat(self.fence).result(timeout=30))
            is not True
        ):
            raise RuntimeError("worker heartbeat did not confirm its bound owner")
        self._pending = op
        try:
            backend = get_event_ipc_backend(self._exports.device)
            event = backend.create_event(self._exports.device)
            self._exports.events.append(event)
            backend.record_event(event, torch_dev.current_stream(self._exports.device))
            encoded = backend.export_event(event, self._exports.device)
            op.issued = True
            if op.kind == "store":
                future = self._exports.client.kmesh_store_span(
                    self.fence,
                    op.request_id,
                    1,
                    list(op.tokens),
                    op.position,
                    op.context_digest,
                    [list(op.block_ids)],
                    encoded,
                )
            else:
                if op.span is None:
                    raise RuntimeError("restore operation lost its source handle")
                future = self._exports.client.kmesh_restore_span(
                    self.fence,
                    op.request_id,
                    1,
                    op.span,
                    op.position,
                    op.context_digest,
                    [list(op.block_ids)],
                    encoded,
                )
            result = _value(future.result(timeout=30))
            self._validate_status(op, result)
            if result.state == "failed-drained":
                self._pending = None
                raise RuntimeError(result.error or "registered span operation failed")
            expected = (
                ("stored",) if op.kind == "store" else ("loaded", "loaded-needs-repair")
            )
            if result.state not in expected or result.error or result.span is None:
                raise RuntimeError("registered span operation has no completed result")
            source = result.span
            semantics = self.fence.lease.domain.semantics
            if (
                source.namespace != self.fence.lease.domain.namespace
                or source.layout_digest != semantics.layout_digest
                or source.rope_digest != semantics.rope_digest
            ):
                raise RuntimeError("source handle differs from this semantic namespace")
            if op.kind == "store" and (
                source.source_start != op.position
                or source.token_count != len(op.tokens)
                or source.token_digest != _tokens_digest(op.tokens)
                or source.context_digest != op.context_digest
                or source.span_id != result.operation_id
            ):
                raise RuntimeError(
                    "admitted source differs from the actual model input"
                )
            self._pending = None
            self._exports.error = None
            return result
        except BaseException as error:
            self._exports.error = error
            logger.exception("registered span operation failed")
            if self._pending is not None and isinstance(error, Exception):
                raise UnsettledLayerSourceError(
                    str(error), self.drain_pending
                ) from error
            raise

    def _validate_status(self, op: _Operation, status: SpanOperationStatus) -> None:
        """Refuse status for another operation, owner, context or reservation."""
        if (
            status.fence != self.fence
            or type(status.generation) is not int
            or status.request_id != op.request_id
            or status.generation != 1
            or status.context != self.runtime.context
            or status.kind != op.kind
            or status.blocks_digest != _sha256([op.block_ids])
            or not status.operation_id
            or (
                op.kind == "restore"
                and status.span is not None
                and status.span != op.span
            )
        ):
            raise RuntimeError(
                "operation receipt differs from the actual worker request"
            )
