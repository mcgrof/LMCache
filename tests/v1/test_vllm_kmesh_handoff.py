# SPDX-License-Identifier: Apache-2.0

# Standard
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, cast
import hashlib
import inspect
import json
import pickle
import sys
import threading
import time

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRequestSidecar,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.kmesh_connector import (
    KMeshConnectorMetadata,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kmesh import (
    KMESH_SCHEMA_VERSION,
    KMeshCompatibilityKey,
    KMeshGroupKind,
    KMeshPlanner,
    KMeshRequestContext,
    KMeshRequestContextParseResult,
)
from vllm.v1.kmesh import (
    KMeshRequestLookupDescriptor as VLLMKMeshRequestLookupDescriptor,
)
from vllm.v1.kmesh import (
    KMeshRequestObservation,
    KMeshSegment,
    KMeshSegmentKind,
    KMeshStateKind,
)
from vllm.v1.request import Request
from vllm.v1.utils import ConstantList

# First Party
from lmcache.integration.vllm import lmcache_connector_v1 as connector_wrapper_module
from lmcache.integration.vllm import (
    lmcache_connector_v1_085 as connector_wrapper_085_module,
)
from lmcache.integration.vllm import vllm_v1_adapter as adapter_module
from lmcache.integration.vllm.connector_construction import (
    ConnectorConstructionPhase,
    RetainedConnectorConstructionOwner,
    cleanup_failed_connector_construction,
)
from lmcache.integration.vllm.kmesh import (
    KMESH_REQUEST_LOOKUP_WIRE_TAG,
    KMESH_REQUEST_SIDECAR_PROVIDER_ID,
    KMeshRequestExecutionEvidence,
    KMeshRuntimeEvidence,
)
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic
from lmcache.integration.vllm.target_profile import (
    COLLECTIVE_BATCHED_POSITION_RECOVERY_V1,
    COLLECTIVE_BATCHED_QKV_V1,
    COLLECTIVE_LOCKSTEP_V1,
    SERIAL_PIC_V1,
)
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    SaveSpec,
)
from lmcache.v1.compute.blend import (
    LMCBlendGroupMemberResult,
    LMCBlendGroupOutcome,
    LMCBlendGroupRequest,
    LMCBlendGroupResult,
    LMCBlendOperationCounts,
)
from lmcache.v1.lifecycle import RetainedCall
from tests.v1.instruction_monitoring import (
    InstructionInterruption,
    instruction_offset,
    requires_instruction_monitoring,
)

TOKENS = (10, 11, 12, 20, 21, 30, 31, 32)


@dataclass
class _Request:
    request_id: str
    prompt_token_ids: list[int]
    num_prompt_tokens: int
    cache_salt: object
    sampling_params: SimpleNamespace
    all_token_ids: ConstantList[int]
    num_computed_tokens: int = 0


class _Poison:
    """Fail if a closed-gate path reaches a legacy cache collaborator."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"legacy cache collaborator was accessed: {name}")


class _LookupClient:
    """Deterministic synchronous explicit-span lookup owner for adapter tests."""

    def __init__(self, result: object = 0) -> None:
        self.result = result
        self.cached: dict[str, object] = {}
        self.validate_calls: list[tuple[object, ...]] = []
        self.lookup_calls: list[tuple[object, ...]] = []
        self.transfers: list[tuple[str, ...]] = []
        self.cancels: list[str] = []

    def supports_explicit_spans(self) -> bool:
        return True

    def validate_lookup_request(
        self,
        token_ids: list[int],
        lookup_id: str,
        request_configs: dict[str, object] | None = None,
        explicit_spans: object = None,
    ) -> None:
        self.validate_calls.append(
            (
                tuple(token_ids),
                lookup_id,
                request_configs,
                explicit_spans,
            )
        )

    def lookup_cache(self, lookup_id: str) -> object:
        return self.cached.get(lookup_id, -1)

    def lookup(
        self,
        token_ids: list[int],
        lookup_id: str,
        request_configs: dict[str, object] | None = None,
        explicit_spans: object = None,
    ) -> object:
        self.lookup_calls.append(
            (
                tuple(token_ids),
                lookup_id,
                request_configs,
                explicit_spans,
            )
        )
        if isinstance(self.result, BaseException):
            raise self.result
        self.cached[lookup_id] = self.result
        return self.result

    def transfer_lookups(self, lookup_ids: list[str]) -> None:
        self.transfers.append(tuple(lookup_ids))
        for lookup_id in lookup_ids:
            self.cached.pop(lookup_id, None)

    def cancel_lookup(self, lookup_id: str) -> None:
        self.cancels.append(lookup_id)
        self.cached.pop(lookup_id, None)

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.cached.pop(lookup_id, None)


class _Stats:
    """Minimal CPU-only stats recorder used by the worker contract tests."""

    def __init__(self) -> None:
        self.vllm_hits: list[int] = []
        self.prompt_tokens: list[int] = []

    def update_interval_vllm_hit_tokens(self, value: int) -> None:
        self.vllm_hits.append(value)

    def update_interval_prompt_tokens(self, value: int) -> None:
        self.prompt_tokens.append(value)


class _Blender:
    """Return request-local retrieval masks while recording exact inputs."""

    def __init__(self) -> None:
        self.masks: dict[str, torch.Tensor | BaseException] = {}
        self.calls: list[dict[str, object]] = []
        self.group_calls: list[tuple[LMCBlendGroupRequest, ...]] = []
        self.group_collective_flags: list[bool] = []
        self.group_qkv_flags: list[bool] = []
        self.group_failures: dict[str, Exception] = {}
        self.group_result_override: LMCBlendGroupResult | None = None

    def blend(
        self,
        token_ids: list[int],
        token_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        call = {"token_ids": token_ids, "token_mask": token_mask, **kwargs}
        self.calls.append(call)
        request_id = cast(str, kwargs["request_id"])
        result = self.masks.get(
            request_id,
            torch.zeros(len(token_ids), dtype=torch.bool),
        )
        if isinstance(result, BaseException):
            raise result
        return result.clone()

    def blend_group(
        self,
        requests: tuple[LMCBlendGroupRequest, ...],
        *,
        collective_position_recovery: bool = False,
        collective_qkv: bool = False,
    ) -> LMCBlendGroupResult:
        self.group_calls.append(requests)
        self.group_collective_flags.append(collective_position_recovery)
        self.group_qkv_flags.append(collective_qkv)
        if self.group_result_override is not None:
            return self.group_result_override
        members = []
        for request in requests:
            error = self.group_failures.get(request.request_id)
            if error is not None:
                members.append(
                    LMCBlendGroupMemberResult(
                        request_id=request.request_id,
                        generation=request.generation,
                        outcome=LMCBlendGroupOutcome.FAILED,
                        retrieval_mask=None,
                        operation_counts=LMCBlendOperationCounts(),
                        error=error,
                    )
                )
                continue
            value = self.masks.get(
                request.request_id,
                torch.zeros(len(request.tokens), dtype=torch.bool),
            )
            if isinstance(value, BaseException):
                raise value
            recovered = bool(torch.any(value).item())
            members.append(
                LMCBlendGroupMemberResult(
                    request_id=request.request_id,
                    generation=request.generation,
                    outcome=(
                        LMCBlendGroupOutcome.RECOVERED
                        if recovered
                        else LMCBlendGroupOutcome.MISS
                    ),
                    retrieval_mask=value.clone(),
                    operation_counts=(
                        LMCBlendOperationCounts(
                            serial_rope_operations=2,
                            serial_diff_operations=1,
                        )
                        if recovered
                        else LMCBlendOperationCounts()
                    ),
                )
            )
        return LMCBlendGroupResult(tuple(members))


class _WorkerGPUConnector:
    """Record exact collective capacity admission in CPU-only tests."""

    def __init__(self) -> None:
        self.capacity = True
        self.preflight_calls: list[dict[str, object]] = []

    def preflight_blend_group_capacity(
        self,
        kvcaches: list[torch.Tensor],
        *,
        num_tokens: int,
        request_count: int,
    ) -> bool:
        self.preflight_calls.append(
            {
                "kvcaches": tuple(kvcaches),
                "num_tokens": num_tokens,
                "request_count": request_count,
            }
        )
        return self.capacity


class _WorkerStoreSession:
    """Retryable close-aware iterator used by the worker engine double."""

    def __init__(self, engine: "_WorkerEngine", request_id: str) -> None:
        self.engine = engine
        self.request_id = request_id
        self.next_index = 0
        self.closed = False

    def __iter__(self) -> "_WorkerStoreSession":
        return self

    def __next__(self) -> None:
        if self.closed:
            raise StopIteration
        fail_after = self.engine.store_failures.get(self.request_id)
        if fail_after == self.next_index:
            raise RuntimeError(f"synthetic store failure: {self.request_id}")
        if self.next_index < len(self.engine.layer_names):
            layer_name = self.engine.layer_names[self.next_index]
            self.engine.store_events.append(("layer", self.request_id, layer_name))
        elif self.next_index == len(self.engine.layer_names):
            self.engine.store_events.append(("flush", self.request_id))
        else:
            raise StopIteration
        self.next_index += 1

    def close(self) -> None:
        remaining = self.engine.store_close_failures.get(self.request_id, 0)
        if remaining > 0:
            self.engine.store_close_failures[self.request_id] = remaining - 1
            self.engine.store_events.append(("close-failed", self.request_id))
            raise RuntimeError(f"synthetic store close failure: {self.request_id}")
        if not self.closed:
            self.engine.store_events.append(("close", self.request_id))
            self.closed = True


class _WorkerEngine:
    """Own worker lookup cleanup and request-local layerwise store sessions."""

    def __init__(self, layer_names: tuple[str, ...]) -> None:
        self.layer_names = layer_names
        self.registered: list[str] = []
        self.cleanup_attempts: list[str] = []
        self.scheduled: list[str] = []
        self.cleanup_failures: dict[str, int] = {}
        self.store_calls: list[dict[str, object]] = []
        self.store_events: list[tuple[object, ...]] = []
        self.store_failures: dict[str, int] = {}
        self.store_close_failures: dict[str, int] = {}
        self.store_cleanup_calls = 0
        self.store_cleanup_failures = 0
        self.store_cleanup_pending = 0
        self.gpu_connector = _WorkerGPUConnector()

    def register_lookup_cleanup(self, lookup_id: str) -> None:
        self.registered.append(lookup_id)

    def schedule_lookup_unpin(self, lookup_id: str) -> None:
        self.cleanup_attempts.append(lookup_id)
        remaining = self.cleanup_failures.get(lookup_id, 0)
        if remaining > 0:
            self.cleanup_failures[lookup_id] = remaining - 1
            raise RuntimeError(f"synthetic lookup cleanup failure: {lookup_id}")
        self.scheduled.append(lookup_id)

    def store_layer(self, token_ids: list[int], **kwargs: object) -> object:
        request_id = cast(str, kwargs["req_id"])
        call = {"token_ids": token_ids, **kwargs}
        self.store_calls.append(call)
        return _WorkerStoreSession(self, request_id)

    def drain_pending_store_cleanups(self) -> None:
        self.store_cleanup_calls += 1
        if self.store_cleanup_failures:
            self.store_cleanup_failures -= 1
            self.store_cleanup_pending = 1
            raise RuntimeError("synthetic engine store cleanup failure")
        self.store_cleanup_pending = 0

    def pending_store_cleanup_count(self) -> int:
        return self.store_cleanup_pending


class _WorkerManager:
    """Worker manager double with observable shutdown behavior."""

    def __init__(self, engine: _WorkerEngine) -> None:
        self.lookup_client = None
        self.lmcache_engine = engine
        self.stop_calls = 0
        self.retain_cleanup_calls = 0
        self.services_stopped = False
        self.terminal_cleanup_errors: tuple[BaseException, ...] = ()

    def stop_services(self) -> None:
        self.stop_calls += 1
        self.services_stopped = True

    def retain_construction_cleanup(self) -> None:
        self.retain_cleanup_calls += 1


class _WorkerParent:
    """Expose the exact metadata object vLLM bound for the current step."""

    def __init__(self) -> None:
        self._connector_metadata: LMCacheConnectorMetadata | None = None

    def _get_connector_metadata(self) -> LMCacheConnectorMetadata | None:
        return self._connector_metadata


class _InterruptingSamplingParams:
    """Interrupt metadata construction through an otherwise public input."""

    @property
    def extra_args(self) -> None:
        raise KeyboardInterrupt("synthetic metadata-build interruption")


def _digest(token_ids: tuple[int, ...]) -> str:
    payload = json.dumps(
        list(token_ids), separators=(",", ":"), ensure_ascii=True
    ).encode()
    digest = hashlib.sha256(b"kmesh-token-slice-v1\0" + payload).hexdigest()
    return f"sha256-v1:{digest}"


def _span(
    *,
    segment_id: str = "shared-a",
    token_start: int = 0,
    token_end: int = 3,
    logical_order: int = 0,
    tokens: tuple[int, ...] = TOKENS,
) -> tuple[object, ...]:
    return (
        segment_id,
        "task",
        token_end - token_start,
        "source-a",
        "lineage-a",
        True,
        _digest(tokens[token_start:token_end]),
        token_start,
        token_end,
        logical_order,
    )


def _payload(
    *,
    request_id: str = "request-a",
    admission_generation: int = 7,
    tokens: tuple[int, ...] = TOKENS,
    cache_domain: str | None = "tenant-a",
    fallback: str | None = None,
    span_payloads: tuple[tuple[object, ...], ...] | None = None,
) -> tuple[object, ...]:
    spans: tuple[tuple[object, ...], ...]
    if fallback is None and span_payloads is not None:
        spans = span_payloads
    elif fallback is None:
        spans = (
            _span(tokens=tokens),
            _span(
                segment_id="shared-b",
                token_start=5,
                token_end=8,
                logical_order=2,
                tokens=tokens,
            ),
        )
    else:
        spans = ()
    return (
        KMESH_REQUEST_LOOKUP_WIRE_TAG,
        "serial_pic",
        request_id,
        admission_generation,
        len(tokens),
        cache_domain,
        fallback,
        spans,
    )


def _sidecar(**changes: Any) -> KVConnectorRequestSidecar:
    return KVConnectorRequestSidecar(
        provider_id=KMESH_REQUEST_SIDECAR_PROVIDER_ID,
        payload=cast(Any, _payload(**changes)),
    )


def _runtime(**changes: object) -> KMeshRuntimeEvidence:
    values: dict[str, object] = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "7ae55760",
        "tokenizer_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "tokenizer_revision": "7ae55760",
        "model_dtype": "bfloat16",
        "effective_kv_dtype": "bfloat16",
        "kv_layout": "NHD",
        "attention_kind": "full",
        "rope_configuration": '{"rope_theta":1000000.0,"rope_type":"default"}',
        "attention_backend": "ROCM_ATTN",
        "attention_backend_configuration": "rocm-attn-v1:no-mla:no-sliding",
        "runtime_configuration": "serial-pic-v1:bf16:eager:max-len-2048",
        "device_identity": "rocm-7.2.3:gfx1100",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    values.update(changes)
    return KMeshRuntimeEvidence(**cast(Any, values))


def _request(
    *,
    request_id: str = "request-a",
    tokens: tuple[int, ...] = TOKENS,
    request_config: str = "alpha",
) -> _Request:
    token_list = list(tokens)
    return _Request(
        request_id=request_id,
        prompt_token_ids=token_list,
        num_prompt_tokens=len(token_list),
        cache_salt=None,
        sampling_params=SimpleNamespace(
            extra_args={
                "kv_transfer_params": {"lmcache.user": request_config},
            }
        ),
        all_token_ids=ConstantList(token_list.copy()),
    )


def _adapter(
    role: KVConnectorRole,
    runtime: KMeshRuntimeEvidence,
    *,
    lookup_result: object = 0,
) -> LMCacheConnectorV1Impl:
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    state = cast(Any, connector)
    state._target_profile = SERIAL_PIC_V1
    state._role = role
    state._kmesh_runtime_evidence = runtime
    state._vllm_config = object()
    state.config = SimpleNamespace(blend_check_layers=[1])
    state._kmesh_prepared_lookups = {}
    state._kmesh_prepared_request_objects = {}
    state._kmesh_allocated_requests = set()
    state._kmesh_active_worker_step = None
    state._kmesh_pending_worker_step = None
    state._kmesh_last_worker_step_generation = 0
    state._kmesh_metadata_transaction_active = False
    state._kmesh_metadata_transaction_committed = False
    state._kmesh_metadata_transaction_finalized = False
    state._kmesh_metadata_transaction_checkpoint = None
    state._kmesh_store_sessions = {}
    state._kmesh_failed_worker_requests = set()
    state._kmesh_outstanding_lookup_ids = set()
    state._kmesh_registered_lookup_cleanups = set()
    state._kmesh_completed_save_generation = 0
    state._kmesh_completed_cleanup_generation = 0
    state._kmesh_next_wait_layer = 0
    state._kmesh_next_save_layer = 0
    state._kmesh_store_cleanup_errors = []
    state._request_trackers = {}
    state._unfinished_requests = {}
    state.load_specs = {}
    state._requests_priority = {}
    state._block_size = 4
    state._lmcache_chunk_size = 4
    state._test_lookup_client = _LookupClient(lookup_result)
    state._prealloc_lookup_ids = {}
    state._blocked_prealloc_requests = set()
    state._allocated_lookup_ids = {}
    state._worker_owned_lookup_requests = set()
    layer_names = ("layer-0", "layer-1")
    state._test_worker_engine = _WorkerEngine(layer_names)
    state._test_worker_manager = _WorkerManager(state._test_worker_engine)
    state._test_blender = _Blender()
    state._test_parent = _WorkerParent()
    state._manager = (
        state._test_worker_manager
        if role == KVConnectorRole.WORKER
        else SimpleNamespace(
            lookup_client=state._test_lookup_client,
            lmcache_engine=_Poison(),
        )
    )
    state._parent = state._test_parent if role == KVConnectorRole.WORKER else _Poison()
    state.use_layerwise = False
    state.enable_blending = role == KVConnectorRole.WORKER
    state.blender = state._test_blender if role == KVConnectorRole.WORKER else None
    state.async_loading = False
    state._invalid_block_ids = set()
    state.device = torch.device("cpu")
    state.kv_caches = {
        layer_name: torch.zeros(1, dtype=torch.float32) for layer_name in layer_names
    }
    state.num_layers = len(layer_names)
    state._stats_monitor = _Stats()
    return connector


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> KMeshRuntimeEvidence:
    evidence = _runtime()
    monkeypatch.setattr(
        adapter_module,
        "build_kmesh_runtime_evidence",
        lambda *_args, **_kwargs: evidence,
    )
    return evidence


def _prepared_execution(
    connector: LMCacheConnectorV1Impl,
    request_id: str = "request-a",
) -> KMeshRequestExecutionEvidence:
    prepared = cast(Any, connector)._kmesh_prepared_lookups[request_id]
    execution = prepared.execution
    assert type(execution) is KMeshRequestExecutionEvidence
    return execution


def _new_request_data(
    request: _Request,
    *,
    block_ids: tuple[int, ...] = (3, 5),
    num_computed_tokens: int = 0,
) -> NewRequestData:
    return NewRequestData(
        req_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids.copy(),
        mm_features=[],
        sampling_params=cast(Any, request.sampling_params),
        pooling_params=None,
        block_ids=(list(block_ids),),
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
        prefill_token_ids=request.prompt_token_ids.copy(),
    )


def _scheduler_output(
    *,
    new_requests: list[NewRequestData] | None = None,
    cached_requests: CachedRequestData | None = None,
    num_scheduled_tokens: dict[str, int] | None = None,
    finished_req_ids: set[str] | None = None,
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = new_requests or []
    output.scheduled_cached_reqs = cached_requests or CachedRequestData.make_empty()
    output.num_scheduled_tokens = num_scheduled_tokens or {}
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    output.finished_req_ids = finished_req_ids or set()
    return output


def _build_new_request_metadata(
    runtime: KMeshRuntimeEvidence,
    *,
    request_id: str = "request-a",
    tokens: tuple[int, ...] = TOKENS,
    block_ids: tuple[int, ...] = (3, 5),
    lookup_result: int = 0,
) -> tuple[
    LMCacheConnectorV1Impl,
    _Request,
    LMCacheConnectorMetadata,
]:
    connector = _adapter(
        KVConnectorRole.SCHEDULER,
        runtime,
        lookup_result=lookup_result,
    )
    request = _request(request_id=request_id, tokens=tokens)
    connector.prepare_request_lookup(
        request,
        (_sidecar(request_id=request_id, tokens=tokens),),
    )
    advertised_tokens = lookup_result - int(lookup_result == len(tokens))
    assert connector.get_num_new_matched_tokens(request, 0) == advertised_tokens
    connector.update_state_after_alloc(request, advertised_tokens)
    request.num_computed_tokens = advertised_tokens
    metadata = connector.build_connector_meta(
        _scheduler_output(
            new_requests=[
                _new_request_data(
                    request,
                    block_ids=block_ids,
                    num_computed_tokens=advertised_tokens,
                )
            ],
            num_scheduled_tokens={request_id: len(tokens) - advertised_tokens},
        )
    )
    assert type(metadata) is LMCacheConnectorMetadata
    return connector, request, metadata


def _prepare_and_allocate(
    connector: LMCacheConnectorV1Impl,
    request: _Request,
    sidecar: KVConnectorRequestSidecar,
) -> None:
    connector.prepare_request_lookup(request, (sidecar,))
    advertised_tokens = connector.get_num_new_matched_tokens(request, 0)
    assert type(advertised_tokens) is int
    connector.update_state_after_alloc(request, advertised_tokens)


def _metadata_signature(
    metadata: LMCacheConnectorMetadata,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            request.req_id,
            tuple(request.token_ids),
            tuple(request.slot_mapping.tolist()),
            request.is_last_prefill,
            request.save_spec,
            request.load_spec,
            request.disagg_spec,
            request.request_configs,
            request.kmesh_execution,
        )
        for request in metadata.requests
    )


def _vllm_descriptor(**changes: Any) -> VLLMKMeshRequestLookupDescriptor:
    return VLLMKMeshRequestLookupDescriptor.from_sidecar_payload(_payload(**changes))


def _plan_batch(*request_ids: str) -> object:
    observations = [
        KMeshRequestObservation(
            request_id,
            KMeshRequestContextParseResult.no_metadata(),
            None,
        )
        for request_id in request_ids
    ]
    return KMeshPlanner(block_size=4, clock_ns=lambda: 0).observe(observations)


def _collective_context(request_id: str) -> KMeshRequestContext:
    return KMeshRequestContext(
        schema_version=KMESH_SCHEMA_VERSION,
        enabled=True,
        execution_id="execution-a",
        group_kind=KMeshGroupKind.AGENT_ROUND,
        group_id="group-a",
        round_id=1,
        branch_id=request_id,
        parent_branch_id=None,
        state_kind=KMeshStateKind.KV_CACHE,
        cache_domain="tenant-a",
        segments=(
            KMeshSegment(
                segment_id="shared-a",
                kind=KMeshSegmentKind.TASK,
                token_start=0,
                token_end=3,
                logical_order=0,
                source_id="source-a",
                shareable=True,
                lineage_id="lineage-a",
            ),
            KMeshSegment(
                segment_id=f"private-{request_id}",
                kind=KMeshSegmentKind.PRIVATE_HISTORY,
                token_start=3,
                token_end=5,
                logical_order=1,
                source_id=f"private-source-{request_id}",
                shareable=False,
                lineage_id=f"private-lineage-{request_id}",
            ),
            KMeshSegment(
                segment_id="shared-b",
                kind=KMeshSegmentKind.TASK,
                token_start=5,
                token_end=8,
                logical_order=2,
                source_id="source-a",
                shareable=True,
                lineage_id="lineage-a",
            ),
        ),
    )


def _permuted_collective_context(request_id: str) -> KMeshRequestContext:
    return KMeshRequestContext(
        schema_version=KMESH_SCHEMA_VERSION,
        enabled=True,
        execution_id="execution-a",
        group_kind=KMeshGroupKind.AGENT_ROUND,
        group_id="group-a",
        round_id=1,
        branch_id=request_id,
        parent_branch_id=None,
        state_kind=KMeshStateKind.KV_CACHE,
        cache_domain="tenant-a",
        segments=(
            KMeshSegment(
                segment_id="shared-b",
                kind=KMeshSegmentKind.TASK,
                token_start=0,
                token_end=3,
                logical_order=0,
                source_id="source-a",
                shareable=True,
                lineage_id="lineage-a",
            ),
            KMeshSegment(
                segment_id=f"private-{request_id}",
                kind=KMeshSegmentKind.PRIVATE_HISTORY,
                token_start=3,
                token_end=5,
                logical_order=1,
                source_id=f"private-source-{request_id}",
                shareable=False,
                lineage_id=f"private-lineage-{request_id}",
            ),
            KMeshSegment(
                segment_id="shared-a",
                kind=KMeshSegmentKind.TASK,
                token_start=5,
                token_end=8,
                logical_order=2,
                source_id="source-a",
                shareable=True,
                lineage_id="lineage-a",
            ),
        ),
    )


def _collective_key(context: KMeshRequestContext) -> KMeshCompatibilityKey:
    return KMeshCompatibilityKey(
        execution_id=context.execution_id,
        group_id=context.group_id,
        round_id=context.round_id,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_revision="7ae55760",
        tokenizer_id_or_revision="7ae55760",
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        kv_layout="NHD",
        attention_kind="full",
        rope_configuration='{"rope_theta":1000000.0}',
        attention_backend="ROCM_ATTN",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        lora_adapter_identity=None,
        cache_salt=None,
        cache_domain=context.cache_domain,
        active_prompt_length=len(TOKENS),
        cached_span=(0, 3),
        prefill_chunk_span=(3, len(TOKENS)),
    )


def _collective_plan_batch(*request_ids: str) -> object:
    observations = []
    for request_id in request_ids:
        context = _collective_context(request_id)
        observations.append(
            KMeshRequestObservation.from_context(
                request_id,
                context,
                _collective_key(context),
                TOKENS,
            )
        )
    return KMeshPlanner(block_size=4, clock_ns=lambda: 0).observe(observations)


def _partially_collective_plan_batch(
    grouped_request_ids: tuple[str, ...],
    fallback_request_ids: tuple[str, ...],
) -> object:
    observations = []
    for request_id in grouped_request_ids:
        context = _collective_context(request_id)
        observations.append(
            KMeshRequestObservation.from_context(
                request_id,
                context,
                _collective_key(context),
                TOKENS,
            )
        )
    observations.extend(
        KMeshRequestObservation(
            request_id,
            KMeshRequestContextParseResult.no_metadata(),
            None,
        )
        for request_id in fallback_request_ids
    )
    return KMeshPlanner(block_size=4, clock_ns=lambda: 0).observe(observations)


def _permuted_collective_plan_batch(
    permuted_tokens: tuple[int, ...],
) -> object:
    context_a = _collective_context("request-a")
    context_b = _permuted_collective_context("request-b")
    observations = (
        KMeshRequestObservation.from_context(
            "request-a",
            context_a,
            _collective_key(context_a),
            TOKENS,
        ),
        KMeshRequestObservation.from_context(
            "request-b",
            context_b,
            _collective_key(context_b),
            permuted_tokens,
        ),
    )
    return KMeshPlanner(block_size=4, clock_ns=lambda: 0).observe(observations)


def _kmesh_metadata(
    descriptors: tuple[VLLMKMeshRequestLookupDescriptor, ...],
    *,
    generation: int = 1,
    plan_batch: object | None = None,
) -> KMeshConnectorMetadata:
    request_ids = tuple(descriptor.request_id for descriptor in descriptors)
    return KMeshConnectorMetadata(
        cast(
            Any,
            _plan_batch(*request_ids) if plan_batch is None else plan_batch,
        ),
        tuple(sorted(descriptors, key=lambda item: item.request_id)),
        generation,
    )


def _bind_worker(
    runtime: KMeshRuntimeEvidence,
    lmcache_metadata: LMCacheConnectorMetadata,
    *,
    descriptors: tuple[VLLMKMeshRequestLookupDescriptor, ...] | None = None,
    generation: int = 1,
) -> LMCacheConnectorV1Impl:
    worker = _adapter(KVConnectorRole.WORKER, runtime)
    cast(Any, worker)._test_parent._connector_metadata = lmcache_metadata
    worker.bind_peer_connector_metadata(
        (
            _kmesh_metadata(
                descriptors or (_vllm_descriptor(),),
                generation=generation,
            ),
            lmcache_metadata,
        )
    )
    return worker


def _collective_worker(
    runtime: KMeshRuntimeEvidence,
    *,
    lookup_result: int = 3,
    request_ids: tuple[str, ...] = ("request-a", "request-b"),
    plan_batch: object | None = None,
    tamper_metadata: Callable[[LMCacheConnectorMetadata], None] | None = None,
) -> tuple[LMCacheConnectorV1Impl, LMCacheConnectorMetadata]:
    scheduler = _adapter(
        KVConnectorRole.SCHEDULER,
        runtime,
        lookup_result=lookup_result,
    )
    requests = tuple(_request(request_id=request_id) for request_id in request_ids)
    for request in requests:
        _prepare_and_allocate(
            scheduler,
            request,
            _sidecar(request_id=request.request_id),
        )
    metadata = scheduler.build_connector_meta(
        _scheduler_output(
            new_requests=[
                _new_request_data(
                    request,
                    block_ids=(3 + 4 * index, 5 + 4 * index),
                )
                for index, request in enumerate(requests)
            ],
            num_scheduled_tokens={request.request_id: 8 for request in requests},
        )
    )
    if tamper_metadata is not None:
        tamper_metadata(metadata)
    descriptors = tuple(
        _vllm_descriptor(request_id=request_id) for request_id in request_ids
    )
    worker = _adapter(KVConnectorRole.WORKER, runtime)
    state = cast(Any, worker)
    state._target_profile = COLLECTIVE_LOCKSTEP_V1
    state._test_parent._connector_metadata = metadata
    worker.bind_peer_connector_metadata(
        (
            _kmesh_metadata(
                descriptors,
                plan_batch=(
                    _collective_plan_batch(*request_ids)
                    if plan_batch is None
                    else plan_batch
                ),
            ),
            metadata,
        )
    )
    return worker, metadata


def _collective_member(
    default_request_id: str,
    **changes: object,
) -> LMCBlendGroupMemberResult:
    values: dict[str, object] = {
        "request_id": default_request_id,
        "generation": 1,
        "outcome": LMCBlendGroupOutcome.RECOVERED,
        "retrieval_mask": torch.tensor(
            [True, True, True, False, False, False, False, False]
        ),
        "operation_counts": LMCBlendOperationCounts(
            serial_rope_operations=2,
            serial_diff_operations=1,
        ),
        "error": None,
    }
    values.update(changes)
    return LMCBlendGroupMemberResult(**cast(Any, values))


def _assert_malformed_collective_result_fails_closed(
    runtime: KMeshRuntimeEvidence,
    result: LMCBlendGroupResult,
) -> None:
    worker, _metadata = _collective_worker(runtime)
    state = cast(Any, worker)
    state._test_blender.group_result_override = result

    worker.start_load_kv(cast(Any, _forward_context()))

    assert len(state._test_blender.group_calls) == 1
    assert state._kmesh_collective_results == []
    assert state._kmesh_failed_worker_requests == {"request-a", "request-b"}
    assert state._invalid_block_ids == {3, 5, 7, 9}


def _forward_context() -> SimpleNamespace:
    return SimpleNamespace(attn_metadata=object())


def test_prepare_binds_exact_sidecar_and_repeat_is_idempotent(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    sidecar = _sidecar()

    connector.prepare_request_lookup(request, (sidecar,))
    first = _prepared_execution(connector)
    connector.prepare_request_lookup(request, (sidecar,))

    second = _prepared_execution(connector)
    assert second is first
    assert second.request_id == request.request_id
    assert second.prompt_token_ids == TOKENS
    assert pickle.loads(pickle.dumps(second)) == second


def test_prepare_rejects_request_replacement_and_descriptor_drift(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    connector.prepare_request_lookup(request, (_sidecar(),))

    with pytest.raises(ValueError, match="request object changed"):
        connector.prepare_request_lookup(_request(), (_sidecar(),))
    with pytest.raises(ValueError, match="descriptor changed"):
        connector.prepare_request_lookup(
            request,
            (_sidecar(admission_generation=8),),
        )


def test_repeat_prepare_rejects_full_prompt_and_request_config_drift(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    sidecar = _sidecar()
    connector.prepare_request_lookup(request, (sidecar,))

    request.prompt_token_ids[1] = 999
    with pytest.raises(ValueError, match="token slice"):
        connector.prepare_request_lookup(request, (sidecar,))
    request.prompt_token_ids[1] = TOKENS[1]

    request.sampling_params.extra_args["kv_transfer_params"]["lmcache.user"] = "beta"
    with pytest.raises(ValueError, match="execution evidence|request configs"):
        connector.prepare_request_lookup(request, (sidecar,))


def test_repeat_prepare_rejects_runtime_drift(
    runtime: KMeshRuntimeEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    sidecar = _sidecar()
    connector.prepare_request_lookup(request, (sidecar,))
    monkeypatch.setattr(
        adapter_module,
        "build_kmesh_runtime_evidence",
        lambda *_args, **_kwargs: _runtime(model_revision="drifted"),
    )

    with pytest.raises(ValueError, match="runtime configuration drifted"):
        connector.prepare_request_lookup(request, (sidecar,))


@pytest.mark.parametrize(
    "sidecars",
    [
        pytest.param([], id="mutable-list"),
        pytest.param(
            (KVConnectorRequestSidecar("unknown", None),),
            id="unknown-provider",
        ),
        pytest.param(
            (
                KVConnectorRequestSidecar(
                    KMESH_REQUEST_SIDECAR_PROVIDER_ID,
                    cast(Any, ("malformed",)),
                ),
            ),
            id="malformed-payload",
        ),
        pytest.param((_sidecar(), _sidecar()), id="duplicate-provider"),
    ],
)
def test_prepare_rejects_non_exact_sidecar_collections(
    runtime: KMeshRuntimeEvidence,
    sidecars: object,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)

    with pytest.raises((TypeError, ValueError)):
        connector.prepare_request_lookup(_request(), cast(Any, sidecars))


def test_empty_sidecars_clear_prior_probe_and_fail_closed_at_lookup(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    connector.prepare_request_lookup(request, (_sidecar(),))

    connector.prepare_request_lookup(request, ())

    state = cast(Any, connector)
    assert state._kmesh_prepared_lookups == {}
    assert state._kmesh_prepared_request_objects == {}
    with pytest.raises(RuntimeError, match="lacks prepared request evidence"):
        connector.get_num_new_matched_tokens(request, 0)


def test_zero_lookup_binds_exact_uuid_and_nonloading_state(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    connector.prepare_request_lookup(request, (_sidecar(),))

    assert connector.get_num_new_matched_tokens(request, 0) == 0
    state = cast(Any, connector)
    lookup_id = state._prealloc_lookup_ids[request.request_id]
    assert state.load_specs == {request.request_id: LoadSpec(0, 0, False)}
    assert state._test_lookup_client.validate_calls[0][:2] == (TOKENS, lookup_id)
    assert state._test_lookup_client.lookup_calls[0][:2] == (TOKENS, lookup_id)
    execution = _prepared_execution(connector)
    assert state._test_lookup_client.lookup_calls[0][2] == (
        execution.as_request_configs()
    )
    assert state._test_lookup_client.lookup_calls[0][3] == execution.explicit_spans


@pytest.mark.parametrize("value", [True, 1, -1])
def test_allocation_rejects_non_exact_advertised_hit(
    runtime: KMeshRuntimeEvidence,
    value: object,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    connector.prepare_request_lookup(request, (_sidecar(),))
    assert connector.get_num_new_matched_tokens(request, 0) == 0

    with pytest.raises(ValueError, match="non-negative exact integer|advertised"):
        connector.update_state_after_alloc(request, cast(Any, value))


def test_actionable_allocation_requires_a_completed_lookup(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    connector.prepare_request_lookup(request, (_sidecar(),))

    with pytest.raises(RuntimeError, match="lost lookup state"):
        connector.update_state_after_alloc(request, 0)


def test_zero_allocation_retains_uuid_and_exact_nonloading_state(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    connector.prepare_request_lookup(request, (_sidecar(),))
    assert connector.get_num_new_matched_tokens(request, 0) == 0
    connector.update_state_after_alloc(request, 0)

    state = cast(Any, connector)
    assert state._unfinished_requests == {request.request_id: request}
    assert state._kmesh_allocated_requests == {request.request_id}
    assert state.load_specs == {request.request_id: LoadSpec(0, 0, False)}
    assert request.request_id not in state._prealloc_lookup_ids
    assert request.request_id in state._allocated_lookup_ids
    assert _prepared_execution(connector).prompt_token_ids == TOKENS


@pytest.mark.parametrize(
    ("lookup_result", "advertised", "expected_load"),
    [
        pytest.param(3, 3, LoadSpec(0, 3, True), id="first-span"),
        pytest.param(8, 7, LoadSpec(0, 8, True), id="full-hit-recompute-last"),
    ],
)
def test_positive_explicit_hit_is_advertised_and_transferred_exactly(
    runtime: KMeshRuntimeEvidence,
    lookup_result: int,
    advertised: int,
    expected_load: LoadSpec,
) -> None:
    connector, request, metadata = _build_new_request_metadata(
        runtime,
        lookup_result=lookup_result,
    )

    assert request.num_computed_tokens == advertised
    assert len(metadata.requests) == 1
    request_meta = metadata.requests[0]
    assert request_meta.load_spec == expected_load
    assert request_meta.save_spec == SaveSpec(0, True)
    assert type(request_meta.lookup_id) is str
    state = cast(Any, connector)
    assert state._test_lookup_client.transfers == [(request_meta.lookup_id,)]
    assert state._test_lookup_client.cancels == []


@pytest.mark.parametrize(
    "lookup_result",
    [
        pytest.param(4, id="undeclared-span-end"),
        pytest.param(RuntimeError("synthetic lookup failure"), id="exception"),
    ],
)
def test_invalid_lookup_revokes_attempt_and_retry_uses_fresh_uuid(
    runtime: KMeshRuntimeEvidence,
    lookup_result: object,
) -> None:
    connector = _adapter(
        KVConnectorRole.SCHEDULER,
        runtime,
        lookup_result=lookup_result,
    )
    request = _request()
    connector.prepare_request_lookup(request, (_sidecar(),))

    with pytest.raises((RuntimeError, ValueError)):
        connector.get_num_new_matched_tokens(request, 0)

    state = cast(Any, connector)
    assert len(state._test_lookup_client.cancels) == 1
    failed_lookup_id = state._test_lookup_client.cancels[0]
    assert state._prealloc_lookup_ids == {}
    assert state.load_specs == {}

    state._test_lookup_client.result = 3
    assert connector.get_num_new_matched_tokens(request, 0) == 3
    retry_lookup_id = state._prealloc_lookup_ids[request.request_id]
    assert retry_lookup_id != failed_lookup_id


def test_new_full_prefill_metadata_is_exact_pickle_safe_and_online(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector, request, metadata = _build_new_request_metadata(runtime)

    assert len(metadata.requests) == 1
    request_meta = metadata.requests[0]
    assert type(request_meta) is ReqMeta
    assert request_meta.req_id == request.request_id
    assert request_meta.token_ids == list(TOKENS)
    assert request_meta.slot_mapping.tolist() == [
        12,
        13,
        14,
        15,
        20,
        21,
        22,
        23,
    ]
    assert request_meta.slot_mapping.dtype == torch.long
    assert type(request_meta.lookup_id) is str
    assert request_meta.load_spec == LoadSpec(0, 0, False)
    assert request_meta.disagg_spec is None
    assert request_meta.is_last_prefill is True
    assert type(request_meta.save_spec) is SaveSpec
    assert request_meta.save_spec == SaveSpec(0, True)
    assert type(request_meta.kmesh_execution) is KMeshRequestExecutionEvidence
    assert request_meta.kmesh_execution.prompt_token_ids == TOKENS
    assert request_meta.request_configs == (
        request_meta.kmesh_execution.as_request_configs()
    )

    restored = pickle.loads(pickle.dumps(metadata))
    assert type(restored) is LMCacheConnectorMetadata
    assert len(restored.requests) == 1
    assert restored.requests[0].req_id == request_meta.req_id
    assert restored.requests[0].token_ids == request_meta.token_ids
    assert torch.equal(
        restored.requests[0].slot_mapping,
        request_meta.slot_mapping,
    )
    assert restored.requests[0].kmesh_execution == request_meta.kmesh_execution
    state = cast(Any, connector)
    assert state._kmesh_prepared_lookups == {}
    assert state._kmesh_prepared_request_objects == {}
    assert state._kmesh_allocated_requests == set()
    assert state.load_specs == {}
    assert state._test_lookup_client.transfers == [(request_meta.lookup_id,)]
    assert state._worker_owned_lookup_requests == {request.request_id}


def test_fallback_descriptor_transports_no_lmcache_request_metadata(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    fallback = _sidecar(cache_domain=None, fallback="no_metadata")
    connector.prepare_request_lookup(request, (fallback,))
    assert connector.get_num_new_matched_tokens(request, 0) == 0
    connector.update_state_after_alloc(request, 0)

    metadata = connector.build_connector_meta(
        _scheduler_output(
            new_requests=[_new_request_data(request)],
            num_scheduled_tokens={request.request_id: len(TOKENS)},
        )
    )

    assert type(metadata) is LMCacheConnectorMetadata
    assert metadata.requests == []
    assert cast(Any, connector)._request_trackers[request.request_id].skip_save is True


def test_real_vllm_request_constant_token_view_builds_metadata(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = Request(
        request_id="request-a",
        prompt_token_ids=list(TOKENS),
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "kv_transfer_params": {"lmcache.user": "alpha"},
            },
        ),
        pooling_params=None,
    )
    assert type(request.all_token_ids) is ConstantList
    _prepare_and_allocate(connector, cast(Any, request), _sidecar())

    metadata = connector.build_connector_meta(
        _scheduler_output(
            new_requests=[
                NewRequestData.from_request(
                    request,
                    ([3, 5],),
                    request.prompt_token_ids,
                )
            ],
            num_scheduled_tokens={request.request_id: len(TOKENS)},
        )
    )

    assert len(metadata.requests) == 1
    assert metadata.requests[0].token_ids == list(TOKENS)


@pytest.mark.parametrize(
    ("invalid_all_token_ids", "error"),
    [
        pytest.param(list(TOKENS), "exactly vLLM ConstantList", id="plain-list"),
        pytest.param(
            ConstantList([*TOKENS[:-1], True]),
            "plain non-negative integer list",
            id="boolean-token",
        ),
    ],
)
def test_live_all_token_ids_require_exact_valid_constant_view(
    runtime: KMeshRuntimeEvidence,
    invalid_all_token_ids: object,
    error: str,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    _prepare_and_allocate(connector, request, _sidecar())
    request.all_token_ids = cast(Any, invalid_all_token_ids)

    with pytest.raises(TypeError, match=error):
        connector.build_connector_meta(
            _scheduler_output(
                new_requests=[_new_request_data(request)],
                num_scheduled_tokens={request.request_id: len(TOKENS)},
            )
        )

    request.all_token_ids = ConstantList(list(TOKENS))
    _prepare_and_allocate(connector, request, _sidecar())
    assert (
        len(
            connector.build_connector_meta(
                _scheduler_output(
                    new_requests=[_new_request_data(request)],
                    num_scheduled_tokens={request.request_id: len(TOKENS)},
                )
            ).requests
        )
        == 1
    )


def test_decode_step_does_not_replay_prior_execution_evidence(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector, request, first_metadata = _build_new_request_metadata(runtime)
    assert len(first_metadata.requests) == 1
    request.num_computed_tokens = len(TOKENS)
    request.all_token_ids = ConstantList([*request.all_token_ids, 99])
    cached = CachedRequestData(
        req_ids=[request.request_id],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[len(TOKENS)],
        num_output_tokens=[1],
    )

    second_metadata = connector.build_connector_meta(
        _scheduler_output(
            cached_requests=cached,
            num_scheduled_tokens={request.request_id: 1},
        )
    )

    assert type(second_metadata) is LMCacheConnectorMetadata
    assert second_metadata.requests == []
    tracker = cast(Any, connector)._request_trackers[request.request_id]
    assert tracker.kmesh_execution is None
    assert tracker.skip_save is True


def test_transferred_load_failure_recomputes_one_full_cached_prefill_locally(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector, request, first_metadata = _build_new_request_metadata(
        runtime,
        lookup_result=len(TOKENS),
    )
    assert len(first_metadata.requests) == 1
    state = cast(Any, connector)
    assert state._worker_owned_lookup_requests == {request.request_id}

    request.num_computed_tokens = 0
    cached = CachedRequestData(
        req_ids=[request.request_id],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )
    metadata = connector.build_connector_meta(
        _scheduler_output(
            cached_requests=cached,
            num_scheduled_tokens={request.request_id: len(TOKENS)},
        )
    )

    assert metadata.requests == []
    tracker = state._request_trackers[request.request_id]
    assert tracker.token_ids == list(TOKENS)
    assert tracker.prompt_len == len(TOKENS)
    assert tracker.allocated_block_ids == [3, 5]
    assert tracker.kmesh_execution is None
    assert tracker.skip_save is True
    assert state._worker_owned_lookup_requests == {request.request_id}
    assert state._kmesh_prepared_lookups == {}
    assert state._kmesh_prepared_request_objects == {}
    assert state._kmesh_allocated_requests == set()
    assert state._prealloc_lookup_ids == {}
    assert state._allocated_lookup_ids == {}
    assert state.load_specs == {}

    worker = _adapter(KVConnectorRole.WORKER, runtime)
    worker.bind_peer_connector_metadata((_kmesh_metadata((), generation=2), metadata))
    step = cast(Any, worker)._kmesh_active_worker_step
    assert step.generation == 2
    assert step.requests == ()
    worker.clear_peer_connector_metadata()


@pytest.mark.parametrize(
    "case",
    [
        "missing-worker-owner",
        "resumed",
        "partial",
        "output-present",
        "new-block-ids",
        "prepared-not-allocated",
        "stale-load-spec",
        "stale-allocated-lookup",
        "nonexact-tracker-tokens",
        "insufficient-tracker-blocks",
    ],
)
def test_cached_prefill_without_exact_load_failure_witness_fails_closed(
    runtime: KMeshRuntimeEvidence,
    case: str,
) -> None:
    connector, request, _ = _build_new_request_metadata(
        runtime,
        lookup_result=len(TOKENS),
    )
    state = cast(Any, connector)
    request.num_computed_tokens = 0
    cached = CachedRequestData(
        req_ids=[request.request_id],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )
    num_scheduled = len(TOKENS)
    error = "lacks exact preparation and allocation"

    if case == "missing-worker-owner":
        state._worker_owned_lookup_requests.clear()
    elif case == "resumed":
        cached.resumed_req_ids.add(request.request_id)
    elif case == "partial":
        num_scheduled -= 1
    elif case == "output-present":
        cached.num_output_tokens[0] = 1
    elif case == "new-block-ids":
        cached.new_block_ids[0] = ([7, 8],)
    elif case == "prepared-not-allocated":
        connector.prepare_request_lookup(request, (_sidecar(),))
        error = "prepared but not allocated"
    elif case == "stale-load-spec":
        state.load_specs[request.request_id] = LoadSpec(0, 0, False)
    elif case == "stale-allocated-lookup":
        state._allocated_lookup_ids[request.request_id] = "stale-lookup"
    elif case == "nonexact-tracker-tokens":
        state._request_trackers[request.request_id].token_ids.append(99)
    else:
        state._request_trackers[request.request_id].allocated_block_ids = [3]

    with pytest.raises(RuntimeError, match=error):
        connector.build_connector_meta(
            _scheduler_output(
                cached_requests=cached,
                num_scheduled_tokens={request.request_id: num_scheduled},
            )
        )


def test_preempted_full_prefill_rebuilds_exact_current_step_metadata(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector, request, _ = _build_new_request_metadata(runtime)
    connector.prepare_request_lookup(request, (_sidecar(),))
    assert connector.get_num_new_matched_tokens(request, 0) == 0
    connector.update_state_after_alloc(request, 0)
    request.num_computed_tokens = 0
    cached = CachedRequestData(
        req_ids=[request.request_id],
        resumed_req_ids={request.request_id},
        new_token_ids=[[]],
        all_token_ids={request.request_id: list(TOKENS)},
        new_block_ids=[([7, 8],)],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )

    metadata = connector.build_connector_meta(
        _scheduler_output(
            cached_requests=cached,
            num_scheduled_tokens={request.request_id: len(TOKENS)},
        )
    )

    assert len(metadata.requests) == 1
    request_meta = metadata.requests[0]
    assert request_meta.token_ids == list(TOKENS)
    assert request_meta.slot_mapping.tolist() == list(range(28, 36))
    assert type(request_meta.kmesh_execution) is KMeshRequestExecutionEvidence
    assert request_meta.save_spec == SaveSpec(0, True)
    assert request_meta.load_spec == LoadSpec(0, 0, False)
    assert type(request_meta.lookup_id) is str


def test_unscheduled_probe_is_forgotten_at_round_boundary(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    connector.prepare_request_lookup(request, (_sidecar(),))

    metadata = connector.build_connector_meta(_scheduler_output())

    assert type(metadata) is LMCacheConnectorMetadata
    assert metadata.requests == []
    state = cast(Any, connector)
    assert state._kmesh_prepared_lookups == {}
    assert state._kmesh_prepared_request_objects == {}
    assert state._kmesh_allocated_requests == set()


def test_finished_request_cleans_prepared_allocated_and_tracker_state(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector, request, _ = _build_new_request_metadata(runtime)
    connector.prepare_request_lookup(request, (_sidecar(),))
    assert connector.get_num_new_matched_tokens(request, 0) == 0
    connector.update_state_after_alloc(request, 0)

    metadata = connector.build_connector_meta(
        _scheduler_output(finished_req_ids={request.request_id})
    )

    assert metadata.requests == []
    state = cast(Any, connector)
    assert state._kmesh_prepared_lookups == {}
    assert state._kmesh_prepared_request_objects == {}
    assert state._kmesh_allocated_requests == set()
    assert state._request_trackers == {}
    assert state._unfinished_requests == {}


def test_finished_and_scheduled_requests_retire_and_transfer_in_same_batch(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    finished = _request(request_id="request-finished")
    scheduled = _request(request_id="request-scheduled")
    _prepare_and_allocate(
        connector,
        finished,
        _sidecar(request_id=finished.request_id),
    )
    _prepare_and_allocate(
        connector,
        scheduled,
        _sidecar(request_id=scheduled.request_id),
    )
    state = cast(Any, connector)
    finished_lookup_id = state._allocated_lookup_ids[finished.request_id]
    scheduled_lookup_id = state._allocated_lookup_ids[scheduled.request_id]

    metadata = connector.build_connector_meta(
        _scheduler_output(
            new_requests=[_new_request_data(scheduled)],
            num_scheduled_tokens={scheduled.request_id: len(TOKENS)},
            finished_req_ids={finished.request_id},
        )
    )

    assert [request.req_id for request in metadata.requests] == [scheduled.request_id]
    assert state._test_lookup_client.cancels == [finished_lookup_id]
    assert state._test_lookup_client.transfers == [(scheduled_lookup_id,)]
    assert finished.request_id not in state._unfinished_requests
    assert state._worker_owned_lookup_requests == {scheduled.request_id}


def test_metadata_error_revokes_attempt_and_corrected_retry_reprobes(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    _prepare_and_allocate(connector, request, _sidecar())
    state = cast(Any, connector)
    failed_lookup_id = state._allocated_lookup_ids[request.request_id]

    with pytest.raises(RuntimeError, match="omitted from scheduler metadata"):
        connector.build_connector_meta(_scheduler_output())

    assert state._test_lookup_client.cancels == [failed_lookup_id]
    assert state._allocated_lookup_ids == {}
    assert state._kmesh_allocated_requests == set()
    _prepare_and_allocate(connector, request, _sidecar())
    retry_lookup_id = state._allocated_lookup_ids[request.request_id]
    assert retry_lookup_id != failed_lookup_id
    metadata = connector.build_connector_meta(
        _scheduler_output(
            new_requests=[_new_request_data(request)],
            num_scheduled_tokens={request.request_id: len(TOKENS)},
        )
    )

    assert len(metadata.requests) == 1
    assert metadata.requests[0].req_id == request.request_id
    assert metadata.requests[0].token_ids == list(TOKENS)


def test_partial_new_request_build_revokes_batch_and_reprobe_is_equivalent(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request_a = _request(request_id="request-a")
    request_b = _request(request_id="request-b")
    _prepare_and_allocate(connector, request_a, _sidecar(request_id="request-a"))
    _prepare_and_allocate(connector, request_b, _sidecar(request_id="request-b"))
    new_a = _new_request_data(request_a, block_ids=(3, 5))
    invalid_new_b = _new_request_data(
        request_b,
        block_ids=(7, 9),
        num_computed_tokens=1,
    )

    with pytest.raises(ValueError, match="num_computed_tokens"):
        connector.build_connector_meta(
            _scheduler_output(
                new_requests=[new_a, invalid_new_b],
                num_scheduled_tokens={"request-a": 8, "request-b": 8},
            )
        )

    assert len(cast(Any, connector)._test_lookup_client.cancels) == 2
    _prepare_and_allocate(connector, request_a, _sidecar(request_id="request-a"))
    _prepare_and_allocate(connector, request_b, _sidecar(request_id="request-b"))
    corrected = connector.build_connector_meta(
        _scheduler_output(
            new_requests=[new_a, _new_request_data(request_b, block_ids=(7, 9))],
            num_scheduled_tokens={"request-a": 8, "request-b": 8},
        )
    )

    fresh = _adapter(KVConnectorRole.SCHEDULER, runtime)
    fresh_a = _request(request_id="request-a")
    fresh_b = _request(request_id="request-b")
    _prepare_and_allocate(fresh, fresh_a, _sidecar(request_id="request-a"))
    _prepare_and_allocate(fresh, fresh_b, _sidecar(request_id="request-b"))
    expected = fresh.build_connector_meta(
        _scheduler_output(
            new_requests=[
                _new_request_data(fresh_a, block_ids=(3, 5)),
                _new_request_data(fresh_b, block_ids=(7, 9)),
            ],
            num_scheduled_tokens={"request-a": 8, "request-b": 8},
        )
    )

    assert _metadata_signature(corrected) == _metadata_signature(expected)


def test_partial_new_request_build_revokes_on_base_exception(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request_a = _request(request_id="request-a")
    request_b = _request(request_id="request-b")
    _prepare_and_allocate(connector, request_a, _sidecar(request_id="request-a"))
    _prepare_and_allocate(connector, request_b, _sidecar(request_id="request-b"))
    new_a = _new_request_data(request_a, block_ids=(3, 5))
    interrupted_b = _new_request_data(request_b, block_ids=(7, 9))
    interrupted_b.sampling_params = cast(Any, _InterruptingSamplingParams())

    with pytest.raises(KeyboardInterrupt, match="metadata-build interruption"):
        connector.build_connector_meta(
            _scheduler_output(
                new_requests=[new_a, interrupted_b],
                num_scheduled_tokens={"request-a": 8, "request-b": 8},
            )
        )

    assert len(cast(Any, connector)._test_lookup_client.cancels) == 2
    _prepare_and_allocate(connector, request_a, _sidecar(request_id="request-a"))
    _prepare_and_allocate(connector, request_b, _sidecar(request_id="request-b"))
    metadata = connector.build_connector_meta(
        _scheduler_output(
            new_requests=[
                new_a,
                _new_request_data(request_b, block_ids=(7, 9)),
            ],
            num_scheduled_tokens={"request-a": 8, "request-b": 8},
        )
    )

    assert tuple(request.req_id for request in metadata.requests) == (
        "request-a",
        "request-b",
    )
    assert tuple(request.slot_mapping.tolist() for request in metadata.requests) == (
        [12, 13, 14, 15, 20, 21, 22, 23],
        [28, 29, 30, 31, 36, 37, 38, 39],
    )


def test_partial_cached_build_requires_fresh_lookup_after_error(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request_a = _request(request_id="request-a")
    request_b = _request(request_id="request-b")
    _prepare_and_allocate(connector, request_a, _sidecar(request_id="request-a"))
    _prepare_and_allocate(connector, request_b, _sidecar(request_id="request-b"))
    connector.build_connector_meta(
        _scheduler_output(
            new_requests=[
                _new_request_data(request_a, block_ids=(3, 5)),
                _new_request_data(request_b, block_ids=(7, 9)),
            ],
            num_scheduled_tokens={"request-a": 8, "request-b": 8},
        )
    )

    _prepare_and_allocate(connector, request_a, _sidecar(request_id="request-a"))
    _prepare_and_allocate(connector, request_b, _sidecar(request_id="request-b"))
    invalid_cached = CachedRequestData(
        req_ids=["request-a", "request-b"],
        resumed_req_ids={"request-a", "request-b"},
        new_token_ids=[[], []],
        all_token_ids={
            "request-a": list(TOKENS),
            "request-b": list(TOKENS),
        },
        new_block_ids=[([11, 12],), ([13, 14],)],
        num_computed_tokens=[0, 1],
        num_output_tokens=[0, 0],
    )
    with pytest.raises(ValueError, match="num_computed_tokens"):
        connector.build_connector_meta(
            _scheduler_output(
                cached_requests=invalid_cached,
                num_scheduled_tokens={"request-a": 8, "request-b": 8},
            )
        )

    assert len(cast(Any, connector)._test_lookup_client.cancels) == 2
    _prepare_and_allocate(connector, request_a, _sidecar(request_id="request-a"))
    _prepare_and_allocate(connector, request_b, _sidecar(request_id="request-b"))
    corrected_cached = deepcopy(invalid_cached)
    corrected_cached.num_computed_tokens[1] = 0
    metadata = connector.build_connector_meta(
        _scheduler_output(
            cached_requests=corrected_cached,
            num_scheduled_tokens={"request-a": 8, "request-b": 8},
        )
    )

    assert tuple(request.req_id for request in metadata.requests) == (
        "request-a",
        "request-b",
    )
    assert tuple(request.slot_mapping.tolist() for request in metadata.requests) == (
        list(range(44, 52)),
        list(range(52, 60)),
    )


def test_missing_allocation_is_distinct_from_allocated_fallback(
    runtime: KMeshRuntimeEvidence,
) -> None:
    request = _request()
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    connector.prepare_request_lookup(request, (_sidecar(),))
    output = _scheduler_output(
        new_requests=[_new_request_data(request)],
        num_scheduled_tokens={request.request_id: len(TOKENS)},
    )

    with pytest.raises(RuntimeError, match="lacks exact allocation"):
        connector.build_connector_meta(output)

    assert connector.get_num_new_matched_tokens(request, 0) == 0
    connector.update_state_after_alloc(request, 0)
    assert len(connector.build_connector_meta(output).requests) == 1

    fallback_request = _request(request_id="request-fallback")
    fallback = _adapter(KVConnectorRole.SCHEDULER, runtime)
    _prepare_and_allocate(
        fallback,
        fallback_request,
        _sidecar(
            request_id="request-fallback",
            cache_domain=None,
            fallback="no_metadata",
        ),
    )
    fallback_metadata = fallback.build_connector_meta(
        _scheduler_output(
            new_requests=[_new_request_data(fallback_request)],
            num_scheduled_tokens={"request-fallback": len(TOKENS)},
        )
    )

    assert fallback_metadata.requests == []


def test_new_tracker_bypasses_legacy_constructor_and_rejects_mm_before_access(
    runtime: KMeshRuntimeEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    _prepare_and_allocate(connector, request, _sidecar())
    new_request = _new_request_data(request)
    new_request.mm_features = [cast(Any, _Poison())]
    monkeypatch.setattr(
        RequestTracker,
        "from_new_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy RequestTracker constructor was called")
        ),
    )

    with pytest.raises(ValueError, match="no multimodal data"):
        connector.build_connector_meta(
            _scheduler_output(
                new_requests=[new_request],
                num_scheduled_tokens={request.request_id: len(TOKENS)},
            )
        )

    _prepare_and_allocate(connector, request, _sidecar())
    new_request.mm_features = []
    metadata = connector.build_connector_meta(
        _scheduler_output(
            new_requests=[new_request],
            num_scheduled_tokens={request.request_id: len(TOKENS)},
        )
    )
    assert len(metadata.requests) == 1


@pytest.mark.parametrize(
    "corruption, error",
    [
        pytest.param("request-id", "bounded plain string", id="request-id"),
        pytest.param("token-ids", "IDs differ", id="token-ids"),
        pytest.param("total", "exact sum", id="total"),
    ],
)
def test_scheduler_id_and_cardinality_rejection_requires_reprobe(
    runtime: KMeshRuntimeEvidence,
    corruption: str,
    error: str,
) -> None:
    connector = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request = _request()
    _prepare_and_allocate(connector, request, _sidecar())
    new_request = _new_request_data(request)
    output = _scheduler_output(
        new_requests=[new_request],
        num_scheduled_tokens={request.request_id: len(TOKENS)},
    )
    if corruption == "request-id":
        new_request.req_id = " request-a"
    elif corruption == "token-ids":
        output.num_scheduled_tokens = {"request-other": len(TOKENS)}
    else:
        output.total_num_scheduled_tokens += 1

    with pytest.raises(ValueError, match=error):
        connector.build_connector_meta(output)

    _prepare_and_allocate(connector, request, _sidecar())
    corrected = _scheduler_output(
        new_requests=[_new_request_data(request)],
        num_scheduled_tokens={request.request_id: len(TOKENS)},
    )
    assert len(connector.build_connector_meta(corrected).requests) == 1


def test_cached_num_computed_and_cardinality_are_exact_and_retry_safe(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector, request, _ = _build_new_request_metadata(runtime)
    request.num_computed_tokens = len(TOKENS)
    request.all_token_ids = ConstantList([*request.all_token_ids, 99])
    cached = CachedRequestData(
        req_ids=[request.request_id],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[],
        num_output_tokens=[1],
    )

    with pytest.raises(ValueError, match="cardinality"):
        connector.build_connector_meta(
            _scheduler_output(
                cached_requests=cached,
                num_scheduled_tokens={request.request_id: 1},
            )
        )

    cached.num_computed_tokens = [len(TOKENS) - 1]
    with pytest.raises(ValueError, match="num_computed_tokens"):
        connector.build_connector_meta(
            _scheduler_output(
                cached_requests=cached,
                num_scheduled_tokens={request.request_id: 1},
            )
        )

    cached.num_computed_tokens = [len(TOKENS)]
    metadata = connector.build_connector_meta(
        _scheduler_output(
            cached_requests=cached,
            num_scheduled_tokens={request.request_id: 1},
        )
    )
    assert metadata.requests == []


def test_request_finished_is_idempotent_and_touches_no_cache_collaborator(
    runtime: KMeshRuntimeEvidence,
) -> None:
    connector, request, _ = _build_new_request_metadata(runtime)
    connector.prepare_request_lookup(request, (_sidecar(),))
    assert connector.get_num_new_matched_tokens(request, 0) == 0
    connector.update_state_after_alloc(request, 0)

    assert connector.request_finished(request, [3, 5]) == (False, None)
    assert connector.request_finished(request, [3, 5]) == (False, None)
    state = cast(Any, connector)
    assert state._kmesh_prepared_lookups == {}
    assert state._kmesh_prepared_request_objects == {}
    assert state._kmesh_allocated_requests == set()
    assert state._request_trackers == {}
    assert state._unfinished_requests == {}


def test_worker_accepts_exact_actionable_and_fallback_descriptor_sets(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    actionable = _vllm_descriptor()
    fallback = _vllm_descriptor(
        request_id="request-fallback",
        cache_domain=None,
        fallback="no_metadata",
    )
    kmesh_metadata = _kmesh_metadata((actionable, fallback))
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    worker.bind_peer_connector_metadata((kmesh_metadata, lmcache_metadata))

    step = cast(Any, worker)._kmesh_active_worker_step
    assert step.generation == 1
    assert tuple(contract.request_id for contract in step.requests) == ("request-a",)
    assert cast(Any, worker)._kmesh_last_worker_step_generation == 1


@pytest.mark.parametrize(
    "make_peer_metadata",
    [
        pytest.param(
            lambda kmesh, lmcache: [kmesh, lmcache],
            id="mutable-list",
        ),
        pytest.param(
            lambda kmesh, lmcache: (kmesh,),
            id="missing-lmcache",
        ),
        pytest.param(
            lambda kmesh, lmcache: (lmcache, kmesh),
            id="wrong-order",
        ),
        pytest.param(
            lambda kmesh, lmcache: (kmesh, SimpleNamespace(requests=[])),
            id="wrong-lmcache-type",
        ),
    ],
)
def test_worker_rejects_non_exact_two_item_peer_metadata(
    runtime: KMeshRuntimeEvidence,
    make_peer_metadata: Any,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    kmesh_metadata = _kmesh_metadata((_vllm_descriptor(),))
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    with pytest.raises((TypeError, ValueError)):
        worker.bind_peer_connector_metadata(
            cast(Any, make_peer_metadata(kmesh_metadata, lmcache_metadata))
        )


def test_worker_rejects_actionable_set_without_lmcache_execution(
    runtime: KMeshRuntimeEvidence,
) -> None:
    kmesh_metadata = _kmesh_metadata((_vllm_descriptor(),))
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    with pytest.raises(ValueError, match="actionable request sets differ"):
        worker.bind_peer_connector_metadata(
            (kmesh_metadata, LMCacheConnectorMetadata())
        )


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(
            lambda metadata: metadata.requests[0].token_ids.__setitem__(1, 999),
            id="full-token-drift",
        ),
        pytest.param(
            lambda metadata: metadata.requests[0].slot_mapping.__setitem__(1, 12),
            id="duplicate-slot",
        ),
        pytest.param(
            lambda metadata: metadata.requests[0].slot_mapping.__setitem__(1, -1),
            id="negative-slot",
        ),
        pytest.param(
            lambda metadata: metadata.requests[0].request_configs.__setitem__(
                "lmcache.user", "tampered"
            ),
            id="request-config-drift",
        ),
        pytest.param(
            lambda metadata: setattr(metadata.requests[0], "lookup_id", "legacy"),
            id="legacy-lookup-id",
        ),
        pytest.param(
            lambda metadata: setattr(
                metadata.requests[0], "save_spec", SaveSpec(0, False)
            ),
            id="save-disabled",
        ),
    ],
)
def test_worker_rejects_tampered_lmcache_execution_metadata(
    runtime: KMeshRuntimeEvidence,
    tamper: Any,
) -> None:
    _, _, original = _build_new_request_metadata(runtime)
    lmcache_metadata = deepcopy(original)
    tamper(lmcache_metadata)
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    with pytest.raises((TypeError, ValueError)):
        worker.bind_peer_connector_metadata(
            (_kmesh_metadata((_vllm_descriptor(),)), lmcache_metadata)
        )


def test_worker_rejects_descriptor_tamper_against_scheduler_execution(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    drifted = _vllm_descriptor(admission_generation=8)
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    with pytest.raises(ValueError, match="descriptor differs"):
        worker.bind_peer_connector_metadata(
            (_kmesh_metadata((drifted,)), lmcache_metadata)
        )


def test_worker_revalidates_mutated_kmesh_container_invariants(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    descriptor = _vllm_descriptor()
    kmesh_metadata = _kmesh_metadata((descriptor,))
    object.__setattr__(
        kmesh_metadata,
        "scheduled_descriptors",
        (descriptor, descriptor),
    )
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    with pytest.raises(ValueError, match="unique and sorted"):
        worker.bind_peer_connector_metadata((kmesh_metadata, lmcache_metadata))


def test_worker_rejects_runtime_drift_before_binding(
    runtime: KMeshRuntimeEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    worker = _adapter(KVConnectorRole.WORKER, runtime)
    monkeypatch.setattr(
        adapter_module,
        "build_kmesh_runtime_evidence",
        lambda *_args, **_kwargs: _runtime(model_revision="worker-drift"),
    )

    with pytest.raises(ValueError, match="runtime configuration drifted"):
        worker.bind_peer_connector_metadata(
            (_kmesh_metadata((_vllm_descriptor(),)), lmcache_metadata)
        )


def test_worker_bind_replay_double_bind_clear_and_higher_generation(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    descriptor = _vllm_descriptor()
    first = _kmesh_metadata((descriptor,), generation=1)
    second = _kmesh_metadata((descriptor,), generation=2)
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    worker.bind_peer_connector_metadata((first, lmcache_metadata))
    with pytest.raises(RuntimeError, match="already bound"):
        worker.bind_peer_connector_metadata((second, lmcache_metadata))

    worker.clear_peer_connector_metadata()
    worker.clear_peer_connector_metadata()
    with pytest.raises(ValueError, match="stale|already consumed"):
        worker.bind_peer_connector_metadata((first, lmcache_metadata))

    worker.bind_peer_connector_metadata((second, lmcache_metadata))
    worker.clear_peer_connector_metadata()
    with pytest.raises(ValueError, match="stale|already consumed"):
        worker.bind_peer_connector_metadata((second, lmcache_metadata))


def test_worker_validation_is_mutation_free_before_bind(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    peer_metadata = (
        _kmesh_metadata((_vllm_descriptor(),), generation=1),
        lmcache_metadata,
    )
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    worker.validate_peer_connector_metadata(peer_metadata)
    worker.bind_peer_connector_metadata(peer_metadata)
    worker.clear_peer_connector_metadata()
    with pytest.raises(ValueError, match="stale|already consumed"):
        worker.bind_peer_connector_metadata(peer_metadata)


def test_worker_transaction_abort_allows_corrected_same_generation_retry(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    kmesh_metadata = _kmesh_metadata((_vllm_descriptor(),), generation=1)
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    worker.begin_connector_metadata_transaction()
    with pytest.raises(ValueError, match="actionable request sets differ"):
        worker.bind_peer_connector_metadata(
            (kmesh_metadata, LMCacheConnectorMetadata())
        )
    worker.abort_connector_metadata_transaction()

    worker.begin_connector_metadata_transaction()
    worker.bind_peer_connector_metadata((kmesh_metadata, lmcache_metadata))
    worker.commit_connector_metadata_transaction()
    worker.finalize_connector_metadata_transaction()
    worker.clear_peer_connector_metadata()
    with pytest.raises(ValueError, match="stale|already consumed"):
        worker.bind_peer_connector_metadata((kmesh_metadata, lmcache_metadata))


def test_worker_transaction_abort_retains_store_ownership_until_retry(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    peer_metadata = (
        _kmesh_metadata((_vllm_descriptor(),), generation=1),
        lmcache_metadata,
    )
    worker = _adapter(KVConnectorRole.WORKER, runtime)
    state = cast(Any, worker)
    state._test_parent._connector_metadata = lmcache_metadata

    worker.begin_connector_metadata_transaction()
    worker.bind_peer_connector_metadata(peer_metadata)
    worker.commit_connector_metadata_transaction()
    worker.wait_for_layer_load("layer-0")
    worker.save_kv_layer(
        "layer-0",
        state.kv_caches["layer-0"],
        cast(Any, object()),
    )
    engine = state._test_worker_engine
    engine.store_cleanup_pending = 1
    # Session.close() drains once and the exhaustive close-all pass drains
    # again. Keep both attempts failed during the first abort.
    engine.store_cleanup_failures = 2

    with pytest.raises(RuntimeError, match="engine store cleanup failure"):
        worker.abort_connector_metadata_transaction()

    assert state._kmesh_metadata_transaction_active is True
    assert state._kmesh_metadata_transaction_committed is True
    assert state._kmesh_metadata_transaction_checkpoint is not None
    assert state._kmesh_active_worker_step.generation == 1
    assert set(state._kmesh_store_sessions) == {"request-a"}
    assert state._kmesh_next_wait_layer == 1
    assert state._kmesh_next_save_layer == 1
    assert engine.store_cleanup_calls == 2
    assert engine.pending_store_cleanup_count() == 1
    assert state._kmesh_outstanding_lookup_ids == set()

    worker.abort_connector_metadata_transaction()

    assert state._kmesh_metadata_transaction_active is False
    assert state._kmesh_metadata_transaction_committed is False
    assert state._kmesh_metadata_transaction_checkpoint is None
    assert state._kmesh_active_worker_step is None
    assert state._kmesh_pending_worker_step is None
    assert state._kmesh_last_worker_step_generation == 0
    assert state._kmesh_next_wait_layer == 0
    assert state._kmesh_next_save_layer == 0
    assert state._kmesh_store_sessions == {}
    assert engine.store_cleanup_calls == 4
    assert engine.pending_store_cleanup_count() == 0


def test_worker_transaction_abort_makes_cleaned_generation_terminal(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    peer_metadata = (
        _kmesh_metadata((_vllm_descriptor(),), generation=1),
        lmcache_metadata,
    )
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    worker.begin_connector_metadata_transaction()
    worker.bind_peer_connector_metadata(peer_metadata)
    worker.abort_connector_metadata_transaction()

    worker.begin_connector_metadata_transaction()
    with pytest.raises(
        ValueError,
        match="cleanup was already scheduled|stale|already consumed",
    ):
        worker.bind_peer_connector_metadata(peer_metadata)
    worker.abort_connector_metadata_transaction()

    _, _, retry_metadata = _build_new_request_metadata(runtime)
    cast(Any, worker)._test_parent._connector_metadata = retry_metadata
    retry_peer = (
        _kmesh_metadata((_vllm_descriptor(),), generation=2),
        retry_metadata,
    )
    worker.begin_connector_metadata_transaction()
    worker.bind_peer_connector_metadata(retry_peer)
    worker.commit_connector_metadata_transaction()
    worker.finalize_connector_metadata_transaction()
    worker.clear_peer_connector_metadata()
    with pytest.raises(
        ValueError,
        match="cleanup was already scheduled|stale|already consumed",
    ):
        worker.bind_peer_connector_metadata(retry_peer)


def test_worker_abort_after_finalize_cannot_resurrect_cleaned_generation(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    peer_metadata = (
        _kmesh_metadata((_vllm_descriptor(),), generation=1),
        lmcache_metadata,
    )
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    worker.begin_connector_metadata_transaction()
    worker.bind_peer_connector_metadata(peer_metadata)
    worker.commit_connector_metadata_transaction()
    worker.finalize_connector_metadata_transaction()
    worker.clear_peer_connector_metadata()
    worker.abort_connector_metadata_transaction()

    worker.begin_connector_metadata_transaction()
    with pytest.raises(
        ValueError,
        match="cleanup was already scheduled|stale|already consumed",
    ):
        worker.bind_peer_connector_metadata(peer_metadata)
    worker.abort_connector_metadata_transaction()

    _, _, retry_metadata = _build_new_request_metadata(runtime)
    cast(Any, worker)._test_parent._connector_metadata = retry_metadata
    retry_peer = (
        _kmesh_metadata((_vllm_descriptor(),), generation=2),
        retry_metadata,
    )
    worker.begin_connector_metadata_transaction()
    worker.bind_peer_connector_metadata(retry_peer)
    worker.commit_connector_metadata_transaction()
    worker.finalize_connector_metadata_transaction()
    worker.clear_peer_connector_metadata()

    with pytest.raises(
        ValueError,
        match="cleanup was already scheduled|stale|already consumed",
    ):
        worker.bind_peer_connector_metadata(retry_peer)


def test_worker_next_begin_replaces_only_finalized_rollback_checkpoint(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, lmcache_metadata = _build_new_request_metadata(runtime)
    descriptor = _vllm_descriptor()
    first = (_kmesh_metadata((descriptor,), generation=1), lmcache_metadata)
    second = (_kmesh_metadata((descriptor,), generation=2), lmcache_metadata)
    worker = _adapter(KVConnectorRole.WORKER, runtime)

    worker.begin_connector_metadata_transaction()
    worker.bind_peer_connector_metadata(first)
    worker.commit_connector_metadata_transaction()
    worker.finalize_connector_metadata_transaction()
    worker.clear_peer_connector_metadata()

    worker.begin_connector_metadata_transaction()
    worker.bind_peer_connector_metadata(second)
    worker.commit_connector_metadata_transaction()
    worker.finalize_connector_metadata_transaction()
    worker.clear_peer_connector_metadata()
    worker.abort_connector_metadata_transaction()

    with pytest.raises(
        ValueError,
        match="cleanup was already scheduled|stale|already consumed",
    ):
        worker.bind_peer_connector_metadata(first)
    with pytest.raises(
        ValueError,
        match="cleanup was already scheduled|stale|already consumed",
    ):
        worker.bind_peer_connector_metadata(second)

    _, _, third_metadata = _build_new_request_metadata(runtime)
    cast(Any, worker)._test_parent._connector_metadata = third_metadata
    third = (_kmesh_metadata((descriptor,), generation=3), third_metadata)
    worker.begin_connector_metadata_transaction()
    worker.bind_peer_connector_metadata(third)
    worker.commit_connector_metadata_transaction()
    worker.finalize_connector_metadata_transaction()


def test_worker_cleanup_uses_immutable_bound_uuid_after_metadata_mutation(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, metadata = _build_new_request_metadata(runtime, lookup_result=3)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)
    original_lookup_id = metadata.requests[0].lookup_id
    metadata.requests[0].lookup_id = "11111111-1111-4111-8111-111111111111"

    with pytest.raises(ValueError, match="lookup owner changed"):
        worker.start_load_kv(cast(Any, _forward_context()))

    assert state._test_worker_engine.registered == [original_lookup_id]
    assert state._test_worker_engine.scheduled == [original_lookup_id]
    assert metadata.requests[0].lookup_id not in (
        state._test_worker_engine.registered + state._test_worker_engine.scheduled
    )


def test_worker_positive_hit_loads_only_whole_spans_up_to_hit_end(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, metadata = _build_new_request_metadata(runtime, lookup_result=3)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)
    state._test_blender.masks["request-a"] = torch.tensor(
        [True, True, True, False, False, False, False, False]
    )

    worker.start_load_kv(cast(Any, _forward_context()))

    assert state._invalid_block_ids == set()
    assert state._kmesh_failed_worker_requests == set()
    assert len(state._test_blender.calls) == 1
    call = state._test_blender.calls[0]
    assert call["token_ids"] == list(TOKENS)
    assert torch.equal(call["token_mask"], torch.ones(8, dtype=torch.bool))
    assert cast(torch.Tensor, call["slot_mapping"]).tolist() == [
        12,
        13,
        14,
        15,
        20,
        21,
        22,
        23,
    ]
    execution = metadata.requests[0].kmesh_execution
    assert execution is not None
    assert call["explicit_spans"] == execution.explicit_spans
    assert call["request_configs"] == metadata.requests[0].request_configs
    assert call["req_id"] == metadata.requests[0].lookup_id
    assert call["request_id"] == "request-a"
    assert call["generation"] == 1


def test_collective_worker_executes_transport_plan_in_one_lockstep_group(
    runtime: KMeshRuntimeEvidence,
) -> None:
    worker, _metadata = _collective_worker(runtime)
    state = cast(Any, worker)
    hit_mask = torch.tensor([True, True, True, False, False, False, False, False])
    state._test_blender.masks.update({"request-a": hit_mask, "request-b": hit_mask})

    step = state._kmesh_active_worker_step
    assert len(step.plans) == 1
    plan = step.plans[0]
    assert tuple(member.request_id for member in plan.members) == (
        "request-a",
        "request-b",
    )
    assert [
        (span.start, span.end) for span in plan.members[0].explicit_spans.spans
    ] == [(0, 3), (5, 8)]

    worker.start_load_kv(cast(Any, _forward_context()))

    assert state._invalid_block_ids == set()
    assert state._kmesh_failed_worker_requests == set()
    assert state._test_blender.calls == []
    assert len(state._test_blender.group_calls) == 1
    group = state._test_blender.group_calls[0]
    assert tuple(member.request_id for member in group) == (
        "request-a",
        "request-b",
    )
    assert all(
        member.retrieve_kwargs["explicit_spans"] == plan.members[index].explicit_spans
        for index, member in enumerate(group)
    )
    gpu_connector = state._test_worker_engine.gpu_connector
    assert len(gpu_connector.preflight_calls) == 1
    assert gpu_connector.preflight_calls[0]["num_tokens"] == 8
    assert gpu_connector.preflight_calls[0]["request_count"] == 2
    assert len(state._kmesh_collective_results) == 1
    result = state._kmesh_collective_results[0][1]
    assert [
        member.operation_counts.serial_rope_operations for member in result.members
    ] == [2, 2]
    assert [
        member.operation_counts.serial_diff_operations for member in result.members
    ] == [1, 1]
    assert all(
        member.operation_counts.collective_rope_operations == 0
        and member.operation_counts.collective_diff_operations == 0
        for member in result.members
    )


def test_collective_worker_rejects_non_tuple_result_members(
    runtime: KMeshRuntimeEvidence,
) -> None:
    members = [
        _collective_member("request-a"),
        _collective_member("request-b"),
    ]
    result = LMCBlendGroupResult(members=cast(Any, members))

    _assert_malformed_collective_result_fails_closed(runtime, result)


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"request_id": "request-b"}, id="request-identity"),
        pytest.param({"generation": 2}, id="generation"),
    ],
)
def test_collective_worker_rejects_result_identity_drift(
    runtime: KMeshRuntimeEvidence,
    changes: dict[str, object],
) -> None:
    result = LMCBlendGroupResult(
        members=(
            _collective_member("request-a", **changes),
            _collective_member("request-b"),
        )
    )

    _assert_malformed_collective_result_fails_closed(runtime, result)


@pytest.mark.parametrize(
    "operation_counts",
    [
        pytest.param(SimpleNamespace(), id="wrong-container-type"),
        pytest.param(
            LMCBlendOperationCounts(serial_rope_operations=True),
            id="bool-scalar",
        ),
        pytest.param(
            LMCBlendOperationCounts(serial_rope_operations=-1),
            id="negative-scalar",
        ),
        pytest.param(
            LMCBlendOperationCounts(collective_rope_operations=1),
            id="false-collective-claim",
        ),
        pytest.param(
            LMCBlendOperationCounts(
                serial_rope_operations=1,
                serial_diff_operations=1,
            ),
            id="incomplete-recovered-work",
        ),
    ],
)
def test_collective_worker_rejects_invalid_operation_counts(
    runtime: KMeshRuntimeEvidence,
    operation_counts: object,
) -> None:
    result = LMCBlendGroupResult(
        members=(
            _collective_member(
                "request-a",
                operation_counts=operation_counts,
            ),
            _collective_member("request-b"),
        )
    )

    _assert_malformed_collective_result_fails_closed(runtime, result)


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param(
            {"outcome": "recovered"},
            id="outcome-not-exact-enum",
        ),
        pytest.param(
            {"error": RuntimeError("recovered member carried an error")},
            id="recovered-with-error",
        ),
        pytest.param(
            {"retrieval_mask": None},
            id="recovered-without-mask",
        ),
        pytest.param(
            {"retrieval_mask": torch.zeros(len(TOKENS), dtype=torch.bool)},
            id="recovered-with-miss-mask",
        ),
        pytest.param(
            {
                "outcome": LMCBlendGroupOutcome.MISS,
                "retrieval_mask": torch.zeros(len(TOKENS), dtype=torch.bool),
                "operation_counts": LMCBlendOperationCounts(),
                "error": RuntimeError("miss member carried an error"),
            },
            id="miss-with-error",
        ),
        pytest.param(
            {
                "outcome": LMCBlendGroupOutcome.MISS,
                "retrieval_mask": torch.tensor(
                    [True, True, True, False, False, False, False, False]
                ),
                "operation_counts": LMCBlendOperationCounts(),
            },
            id="miss-with-recovered-mask",
        ),
        pytest.param(
            {
                "outcome": LMCBlendGroupOutcome.FAILED,
                "retrieval_mask": None,
                "operation_counts": LMCBlendOperationCounts(),
            },
            id="failed-without-error",
        ),
        pytest.param(
            {
                "outcome": LMCBlendGroupOutcome.FAILED,
                "retrieval_mask": torch.zeros(len(TOKENS), dtype=torch.bool),
                "operation_counts": LMCBlendOperationCounts(),
                "error": RuntimeError("synthetic failure"),
            },
            id="failed-with-mask",
        ),
        pytest.param(
            {
                "outcome": LMCBlendGroupOutcome.FAILED,
                "retrieval_mask": None,
                "operation_counts": LMCBlendOperationCounts(),
                "error": "not an exception",
            },
            id="failed-with-non-exception-error",
        ),
    ],
)
def test_collective_worker_rejects_inconsistent_member_result(
    runtime: KMeshRuntimeEvidence,
    changes: dict[str, object],
) -> None:
    result = LMCBlendGroupResult(
        members=(
            _collective_member("request-a", **changes),
            _collective_member("request-b"),
        )
    )

    _assert_malformed_collective_result_fails_closed(runtime, result)


def test_collective_worker_orders_spans_by_each_request_position(
    runtime: KMeshRuntimeEvidence,
) -> None:
    permuted_tokens = (30, 31, 32, 20, 21, 10, 11, 12)
    permuted_spans = (
        _span(
            segment_id="shared-b",
            token_start=0,
            token_end=3,
            logical_order=0,
            tokens=permuted_tokens,
        ),
        _span(
            segment_id="shared-a",
            token_start=5,
            token_end=8,
            logical_order=2,
            tokens=permuted_tokens,
        ),
    )
    scheduler = _adapter(KVConnectorRole.SCHEDULER, runtime, lookup_result=3)
    request_a = _request(request_id="request-a")
    request_b = _request(request_id="request-b", tokens=permuted_tokens)
    _prepare_and_allocate(
        scheduler,
        request_a,
        _sidecar(request_id="request-a"),
    )
    _prepare_and_allocate(
        scheduler,
        request_b,
        _sidecar(
            request_id="request-b",
            tokens=permuted_tokens,
            span_payloads=permuted_spans,
        ),
    )
    lmcache_metadata = scheduler.build_connector_meta(
        _scheduler_output(
            new_requests=[
                _new_request_data(request_a, block_ids=(3, 5)),
                _new_request_data(request_b, block_ids=(7, 9)),
            ],
            num_scheduled_tokens={"request-a": 8, "request-b": 8},
        )
    )
    descriptors = (
        _vllm_descriptor(request_id="request-a"),
        _vllm_descriptor(
            request_id="request-b",
            tokens=permuted_tokens,
            span_payloads=permuted_spans,
        ),
    )
    plan_batch = _permuted_collective_plan_batch(permuted_tokens)
    public_b_mapping = cast(Any, plan_batch).plans[0].request_segment_maps[1]
    assert [
        (entry.token_start, entry.token_end) for entry in public_b_mapping.entries
    ] == [(5, 8), (0, 3)]

    worker = _adapter(KVConnectorRole.WORKER, runtime)
    state = cast(Any, worker)
    state._target_profile = COLLECTIVE_LOCKSTEP_V1
    state._test_parent._connector_metadata = lmcache_metadata
    worker.bind_peer_connector_metadata(
        (
            _kmesh_metadata(descriptors, plan_batch=plan_batch),
            lmcache_metadata,
        )
    )

    frozen_b_spans = state._kmesh_active_worker_step.plans[0].members[1].explicit_spans
    assert [(span.start, span.end) for span in frozen_b_spans.spans] == [
        (0, 3),
        (5, 8),
    ]


def test_collective_worker_contains_one_member_failure_and_finishes_sibling(
    runtime: KMeshRuntimeEvidence,
) -> None:
    worker, _metadata = _collective_worker(runtime)
    state = cast(Any, worker)
    hit_mask = torch.tensor([True, True, True, False, False, False, False, False])
    state._test_blender.masks.update({"request-a": hit_mask, "request-b": hit_mask})
    state._test_blender.group_failures["request-a"] = RuntimeError(
        "synthetic member failure"
    )

    worker.start_load_kv(cast(Any, _forward_context()))

    assert len(state._test_blender.group_calls) == 1
    assert state._kmesh_failed_worker_requests == {"request-a"}
    assert state._invalid_block_ids == {3, 5}
    result = state._kmesh_collective_results[0][1]
    assert [member.outcome for member in result.members] == [
        LMCBlendGroupOutcome.FAILED,
        LMCBlendGroupOutcome.RECOVERED,
    ]


def test_collective_worker_rejects_capacity_before_group_mutation(
    runtime: KMeshRuntimeEvidence,
) -> None:
    worker, _metadata = _collective_worker(runtime)
    state = cast(Any, worker)
    state._test_worker_engine.gpu_connector.capacity = False

    worker.start_load_kv(cast(Any, _forward_context()))

    assert state._test_blender.group_calls == []
    assert state._kmesh_collective_results == []
    assert state._kmesh_failed_worker_requests == {"request-a", "request-b"}
    assert state._invalid_block_ids == {3, 5, 7, 9}


def test_collective_worker_runs_unplanned_actionable_request_serially(
    runtime: KMeshRuntimeEvidence,
) -> None:
    request_ids = ("request-a", "request-b", "request-c")
    plan_batch = _partially_collective_plan_batch(
        ("request-a", "request-b"),
        ("request-c",),
    )
    worker, _metadata = _collective_worker(
        runtime,
        request_ids=request_ids,
        plan_batch=plan_batch,
    )
    state = cast(Any, worker)
    hit_mask = torch.tensor([True, True, True, False, False, False, False, False])
    state._test_blender.masks.update(
        {request_id: hit_mask for request_id in request_ids}
    )

    worker.start_load_kv(cast(Any, _forward_context()))

    assert len(state._test_blender.group_calls) == 1
    assert tuple(
        member.request_id for member in state._test_blender.group_calls[0]
    ) == ("request-a", "request-b")
    assert [call["request_id"] for call in state._test_blender.calls] == ["request-c"]
    assert state._kmesh_failed_worker_requests == set()
    assert state._invalid_block_ids == set()


def test_collective_worker_rejects_cross_request_destination_collision(
    runtime: KMeshRuntimeEvidence,
) -> None:
    def overlap_first_slot(metadata: LMCacheConnectorMetadata) -> None:
        metadata.requests[1].slot_mapping[0] = metadata.requests[0].slot_mapping[0]

    with pytest.raises(ValueError, match="share a destination slot"):
        _collective_worker(runtime, tamper_metadata=overlap_first_slot)


@pytest.mark.parametrize(
    ("mask", "expected_error"),
    [
        pytest.param(
            [True, False, True, False, False, False, False, False],
            "partial declared span",
            id="partial-span",
        ),
        pytest.param(
            [True, True, True, False, False, True, True, True],
            "span beyond the lookup hit",
            id="beyond-hit-end",
        ),
    ],
)
def test_worker_invalid_retrieval_mask_fails_request_and_invalidates_prompt(
    runtime: KMeshRuntimeEvidence,
    mask: list[bool],
    expected_error: str,
) -> None:
    _, _, metadata = _build_new_request_metadata(runtime, lookup_result=3)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)
    result_mask = torch.tensor(mask)
    state._test_blender.masks["request-a"] = result_mask
    contract = state._kmesh_active_worker_step.requests[0]
    with pytest.raises(ValueError, match=expected_error):
        worker._validate_kmesh_retrieval_mask(contract, result_mask)

    worker.start_load_kv(cast(Any, _forward_context()))

    assert state._kmesh_failed_worker_requests == {"request-a"}
    assert state._invalid_block_ids == {3, 5}


def test_worker_all_false_race_invalidates_hit_and_suppresses_store(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, metadata = _build_new_request_metadata(runtime, lookup_result=3)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)
    state._test_blender.masks["request-a"] = torch.zeros(8, dtype=torch.bool)

    worker.start_load_kv(cast(Any, _forward_context()))
    for layer_name, kv_layer in state.kv_caches.items():
        worker.wait_for_layer_load(layer_name)
        worker.save_kv_layer(layer_name, kv_layer, cast(Any, object()))
    worker.wait_for_save()

    assert state._invalid_block_ids == {3}
    assert state._test_worker_engine.store_calls == []
    assert state._test_worker_engine.scheduled == [metadata.requests[0].lookup_id]


def test_worker_gap_recompute_failure_rolls_back_prompt_and_suppresses_store(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, metadata = _build_new_request_metadata(runtime, lookup_result=3)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)
    state._test_blender.masks["request-a"] = RuntimeError(
        "mandatory explicit gap recompute failed"
    )

    worker.start_load_kv(cast(Any, _forward_context()))
    assert state._kmesh_failed_worker_requests == {"request-a"}
    assert state._invalid_block_ids == {3, 5}
    for layer_name, kv_layer in state.kv_caches.items():
        worker.wait_for_layer_load(layer_name)
        worker.save_kv_layer(layer_name, kv_layer, cast(Any, object()))
    worker.wait_for_save()

    assert state._invalid_block_ids == {3, 5}
    assert state._test_worker_engine.store_calls == []
    assert state._test_worker_engine.scheduled == [metadata.requests[0].lookup_id]


def test_worker_store_observes_exact_layer_order_and_flushes_once(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, metadata = _build_new_request_metadata(runtime)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)

    worker.start_load_kv(cast(Any, _forward_context()))
    for layer_name, kv_layer in state.kv_caches.items():
        worker.wait_for_layer_load(layer_name)
        worker.save_kv_layer(layer_name, kv_layer, cast(Any, object()))
    worker.wait_for_save()

    assert state._test_worker_engine.store_events == [
        ("layer", "request-a", "layer-0"),
        ("layer", "request-a", "layer-1"),
        ("flush", "request-a"),
        ("close", "request-a"),
    ]
    assert len(state._test_worker_engine.store_calls) == 1
    call = state._test_worker_engine.store_calls[0]
    assert call["token_ids"] == list(TOKENS)
    assert cast(torch.Tensor, call["mask"]).tolist() == [True] * 8
    assert cast(torch.Tensor, call["slot_mapping"]).tolist() == [
        12,
        13,
        14,
        15,
        20,
        21,
        22,
        23,
    ]
    execution = metadata.requests[0].kmesh_execution
    assert execution is not None
    assert call["explicit_spans"] == execution.explicit_spans
    assert call["request_configs"] == metadata.requests[0].request_configs
    assert call["offset"] == 0
    assert call["sync"] is True
    assert call["req_id"] == "request-a"


def test_worker_rejects_out_of_order_layer_hooks(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, metadata = _build_new_request_metadata(runtime)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)

    with pytest.raises(ValueError, match="layer wait order"):
        worker.wait_for_layer_load("layer-1")

    worker.wait_for_layer_load("layer-0")
    with pytest.raises(RuntimeError, match="before saving the prior layer"):
        worker.wait_for_layer_load("layer-1")
    with pytest.raises(ValueError, match="different KV-cache tensor"):
        worker.save_kv_layer(
            "layer-0",
            torch.zeros(1),
            cast(Any, object()),
        )
    assert state._test_worker_engine.scheduled == [metadata.requests[0].lookup_id]


def test_worker_missing_store_layer_closes_without_advancing_future_layer(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, metadata = _build_new_request_metadata(runtime)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)

    worker.wait_for_layer_load("layer-0")
    worker.save_kv_layer(
        "layer-0",
        state.kv_caches["layer-0"],
        cast(Any, object()),
    )
    with pytest.raises(RuntimeError, match="every ordered layer hook"):
        worker.wait_for_save()

    assert state._test_worker_engine.store_events == [
        ("layer", "request-a", "layer-0"),
        ("close", "request-a"),
    ]
    assert state._kmesh_store_sessions == {}


def test_worker_wait_retries_cleanup_and_close_before_surfacing_store_error(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, metadata = _build_new_request_metadata(runtime)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)
    lookup_id = metadata.requests[0].lookup_id

    for layer_name, kv_layer in state.kv_caches.items():
        worker.wait_for_layer_load(layer_name)
        worker.save_kv_layer(layer_name, kv_layer, cast(Any, object()))
    state._test_worker_engine.cleanup_failures[lookup_id] = 1
    # The first close is from session.finish(); the second is the bounded
    # close-all retry in the same wait_for_save call.
    state._test_worker_engine.store_close_failures["request-a"] = 2

    with pytest.raises(RuntimeError, match="lookup cleanup failure"):
        worker.wait_for_save()

    assert state._kmesh_outstanding_lookup_ids == {lookup_id}
    assert set(state._kmesh_store_sessions) == {"request-a"}
    assert len(state._kmesh_store_cleanup_errors) == 1
    assert "store close failure" in str(state._kmesh_store_cleanup_errors[0])
    session = state._kmesh_store_sessions["request-a"]
    assert session.flushed is True
    assert session.closed is False
    assert state._test_worker_engine.store_events[-3:] == [
        ("flush", "request-a"),
        ("close-failed", "request-a"),
        ("close-failed", "request-a"),
    ]

    with pytest.raises(RuntimeError, match="store close failure"):
        worker.wait_for_save()

    assert state._kmesh_outstanding_lookup_ids == set()
    assert state._kmesh_store_sessions == {}
    assert state._kmesh_store_cleanup_errors == []
    assert state._test_worker_engine.scheduled == [lookup_id]
    assert state._test_worker_engine.store_events[-1] == ("close", "request-a")


def test_worker_clear_recovery_drops_stale_store_failure_before_next_generation(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, first_metadata = _build_new_request_metadata(runtime)
    worker = _bind_worker(runtime, first_metadata)
    state = cast(Any, worker)
    first_lookup_id = first_metadata.requests[0].lookup_id
    engine = state._test_worker_engine
    engine.store_failures["request-a"] = 0
    engine.store_close_failures["request-a"] = 1
    engine.cleanup_failures[first_lookup_id] = 1

    worker.wait_for_layer_load("layer-0")
    worker.save_kv_layer(
        "layer-0",
        state.kv_caches["layer-0"],
        cast(Any, object()),
    )

    assert state._kmesh_failed_worker_requests == {"request-a"}
    assert set(state._kmesh_store_sessions) == {"request-a"}
    assert len(state._kmesh_store_cleanup_errors) == 1
    assert "store close failure" in str(state._kmesh_store_cleanup_errors[0])

    with pytest.raises(RuntimeError, match="lookup cleanup failure"):
        worker.clear_peer_connector_metadata()

    assert state._kmesh_outstanding_lookup_ids == {first_lookup_id}
    assert state._kmesh_store_sessions == {}
    assert state._kmesh_failed_worker_requests == {"request-a"}
    assert len(state._kmesh_store_cleanup_errors) == 1

    worker.clear_peer_connector_metadata()

    assert state._kmesh_outstanding_lookup_ids == set()
    assert state._kmesh_failed_worker_requests == set()
    assert state._kmesh_store_cleanup_errors == []

    engine.store_failures.pop("request-a")
    _, _, second_metadata = _build_new_request_metadata(runtime)
    second_lookup_id = second_metadata.requests[0].lookup_id
    assert second_lookup_id != first_lookup_id
    state._test_parent._connector_metadata = second_metadata
    worker.bind_peer_connector_metadata(
        (
            _kmesh_metadata((_vllm_descriptor(),), generation=2),
            second_metadata,
        )
    )

    for layer_name, kv_layer in state.kv_caches.items():
        worker.wait_for_layer_load(layer_name)
        worker.save_kv_layer(layer_name, kv_layer, cast(Any, object()))
    worker.wait_for_save()

    assert state._kmesh_store_sessions == {}
    assert state._kmesh_store_cleanup_errors == []
    assert state._kmesh_failed_worker_requests == set()
    assert engine.scheduled == [first_lookup_id, second_lookup_id]
    assert engine.store_events[-4:] == [
        ("layer", "request-a", "layer-0"),
        ("layer", "request-a", "layer-1"),
        ("flush", "request-a"),
        ("close", "request-a"),
    ]


def test_worker_shutdown_retries_cleanup_before_stopping_services(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, metadata = _build_new_request_metadata(runtime)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)
    lookup_id = metadata.requests[0].lookup_id
    state._test_worker_engine.cleanup_failures[lookup_id] = 1
    state._test_worker_engine.store_cleanup_pending = 1
    state._test_worker_engine.store_cleanup_failures = 1

    with pytest.raises(RuntimeError, match="lookup cleanup failure"):
        worker.shutdown()

    assert state._kmesh_outstanding_lookup_ids == {lookup_id}
    assert state._test_worker_engine.pending_store_cleanup_count() == 1
    assert state._test_worker_engine.store_cleanup_calls == 1
    assert state._test_worker_manager.stop_calls == 0
    assert state._test_worker_manager.retain_cleanup_calls == 1
    assert state.blender is state._test_blender

    worker.shutdown()

    assert state._kmesh_outstanding_lookup_ids == set()
    assert state._test_worker_engine.pending_store_cleanup_count() == 0
    assert state._test_worker_engine.store_cleanup_calls == 2
    assert state._test_worker_engine.scheduled == [lookup_id]
    assert state._test_worker_manager.stop_calls == 1
    assert state.blender is None


def test_worker_shutdown_waits_for_racing_store_session_insertion(
    runtime: KMeshRuntimeEvidence,
) -> None:
    """A save admitted first is published and closed before shutdown returns."""
    _, _, metadata = _build_new_request_metadata(runtime)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)
    worker.start_load_kv(cast(Any, _forward_context()))
    worker.wait_for_layer_load("layer-0")

    store_entered = threading.Event()
    store_release = threading.Event()
    original_store_layer = state._test_worker_engine.store_layer

    def blocking_store_layer(
        token_ids: list[int],
        **kwargs: object,
    ) -> object:
        store_entered.set()
        if not store_release.wait(timeout=2.0):
            raise RuntimeError("timed out waiting to release store insertion")
        return original_store_layer(token_ids, **kwargs)

    state._test_worker_engine.store_layer = blocking_store_layer
    save_errors: list[BaseException] = []
    shutdown_errors: list[BaseException] = []

    def run_save() -> None:
        try:
            worker.save_kv_layer(
                "layer-0",
                state.kv_caches["layer-0"],
                cast(Any, object()),
            )
        except BaseException as error:
            save_errors.append(error)

    def run_shutdown() -> None:
        try:
            worker.shutdown()
        except BaseException as error:
            shutdown_errors.append(error)

    save_thread = threading.Thread(target=run_save)
    save_thread.start()
    assert store_entered.wait(timeout=2.0)

    shutdown_thread = threading.Thread(target=run_shutdown)
    shutdown_thread.start()
    time.sleep(0.05)
    assert shutdown_thread.is_alive()
    assert state._test_worker_manager.stop_calls == 0

    store_release.set()
    save_thread.join(timeout=2.0)
    shutdown_thread.join(timeout=2.0)
    assert not save_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert save_errors == []
    assert shutdown_errors == []
    assert state._kmesh_store_sessions == {}
    assert state._kmesh_store_sessions_closed
    assert state._test_worker_engine.store_events == [
        ("layer", "request-a", "layer-0"),
        ("close", "request-a"),
    ]
    assert state._test_worker_manager.stop_calls == 1
    assert state.blender is None

    with pytest.raises(RuntimeError, match="shutdown has begun"):
        worker.save_kv_layer(
            "layer-0",
            state.kv_caches["layer-0"],
            cast(Any, object()),
        )
    assert len(state._test_worker_engine.store_calls) == 1


def test_worker_clear_wins_race_and_late_save_revalidates_step(
    runtime: KMeshRuntimeEvidence,
) -> None:
    """Clear holds the session lock through close and exact state reset."""
    _, _, metadata = _build_new_request_metadata(runtime)
    worker = _bind_worker(runtime, metadata)
    state = cast(Any, worker)
    worker.start_load_kv(cast(Any, _forward_context()))
    worker.wait_for_layer_load("layer-0")

    close_entered = threading.Event()
    close_release = threading.Event()

    class _BlockingSession:
        def close(self) -> None:
            close_entered.set()
            if not close_release.wait(timeout=2.0):
                raise RuntimeError("timed out waiting to release session close")

    state._kmesh_store_sessions["request-a"] = _BlockingSession()
    clear_errors: list[BaseException] = []
    save_errors: list[BaseException] = []

    def run_clear() -> None:
        try:
            worker.clear_peer_connector_metadata()
        except BaseException as error:
            clear_errors.append(error)

    def run_save() -> None:
        try:
            worker.save_kv_layer(
                "layer-0",
                state.kv_caches["layer-0"],
                cast(Any, object()),
            )
        except BaseException as error:
            save_errors.append(error)

    clear_thread = threading.Thread(target=run_clear)
    clear_thread.start()
    assert close_entered.wait(timeout=2.0)

    save_thread = threading.Thread(target=run_save)
    save_thread.start()
    time.sleep(0.05)
    assert save_thread.is_alive()

    close_release.set()
    clear_thread.join(timeout=2.0)
    save_thread.join(timeout=2.0)
    assert not clear_thread.is_alive()
    assert not save_thread.is_alive()
    assert clear_errors == []
    assert len(save_errors) == 1
    assert "metadata was not bound" in str(save_errors[0])
    assert state._kmesh_store_sessions == {}
    assert state._kmesh_active_worker_step is None
    assert state._test_worker_engine.store_calls == []


def test_worker_request_local_store_failure_does_not_block_sibling_finalization(
    runtime: KMeshRuntimeEvidence,
) -> None:
    scheduler = _adapter(KVConnectorRole.SCHEDULER, runtime)
    request_a = _request(request_id="request-a")
    request_b = _request(request_id="request-b")
    _prepare_and_allocate(scheduler, request_a, _sidecar(request_id="request-a"))
    _prepare_and_allocate(scheduler, request_b, _sidecar(request_id="request-b"))
    metadata = scheduler.build_connector_meta(
        _scheduler_output(
            new_requests=[
                _new_request_data(request_a, block_ids=(3, 5)),
                _new_request_data(request_b, block_ids=(7, 9)),
            ],
            num_scheduled_tokens={"request-a": 8, "request-b": 8},
        )
    )
    worker = _bind_worker(
        runtime,
        metadata,
        descriptors=(
            _vllm_descriptor(request_id="request-a"),
            _vllm_descriptor(request_id="request-b"),
        ),
    )
    state = cast(Any, worker)
    state._test_worker_engine.store_failures["request-a"] = 1

    for layer_name, kv_layer in state.kv_caches.items():
        worker.wait_for_layer_load(layer_name)
        worker.save_kv_layer(layer_name, kv_layer, cast(Any, object()))
    worker.wait_for_save()

    assert state._test_worker_engine.store_events == [
        ("layer", "request-a", "layer-0"),
        ("layer", "request-b", "layer-0"),
        ("close", "request-a"),
        ("layer", "request-b", "layer-1"),
        ("flush", "request-b"),
        ("close", "request-b"),
    ]
    assert len(state._test_worker_engine.scheduled) == 2
    assert set(state._test_worker_engine.scheduled) == {
        metadata.requests[0].lookup_id,
        metadata.requests[1].lookup_id,
    }


def test_worker_rejects_non_list_or_non_exact_lmcache_request_items(
    runtime: KMeshRuntimeEvidence,
) -> None:
    _, _, original = _build_new_request_metadata(runtime)
    kmesh_metadata = _kmesh_metadata((_vllm_descriptor(),))

    non_list = deepcopy(original)
    non_list.requests = cast(Any, tuple(non_list.requests))
    worker = _adapter(KVConnectorRole.WORKER, runtime)
    with pytest.raises(TypeError, match="plain list"):
        worker.bind_peer_connector_metadata((kmesh_metadata, non_list))

    wrong_item = LMCacheConnectorMetadata(
        requests=[cast(Any, SimpleNamespace(req_id="request-a"))]
    )
    worker = _adapter(KVConnectorRole.WORKER, runtime)
    with pytest.raises(TypeError, match="exactly ReqMeta"):
        worker.bind_peer_connector_metadata((kmesh_metadata, wrong_item))


def test_scheduler_and_worker_methods_reject_wrong_role(
    runtime: KMeshRuntimeEvidence,
) -> None:
    scheduler = _adapter(KVConnectorRole.SCHEDULER, runtime)
    worker = _adapter(KVConnectorRole.WORKER, runtime)
    request = _request()

    with pytest.raises(RuntimeError, match="only bind on the scheduler"):
        worker.prepare_request_lookup(request, (_sidecar(),))
    with pytest.raises(RuntimeError, match="only run on the scheduler"):
        worker.get_num_new_matched_tokens(request, 0)
    with pytest.raises(RuntimeError, match="only bind on the worker"):
        scheduler.bind_peer_connector_metadata(cast(Any, ()))
    with pytest.raises(RuntimeError, match="only clear on the worker"):
        scheduler.clear_peer_connector_metadata()


class _ForwardRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def prepare_request_lookup(
        self,
        request: object,
        peer_sidecars: object,
    ) -> None:
        self.calls.append(("prepare", request, peer_sidecars))

    def bind_peer_connector_metadata(self, peer_metadata: object) -> None:
        self.calls.append(("bind", peer_metadata))

    def validate_peer_connector_metadata(self, peer_metadata: object) -> None:
        self.calls.append(("validate", peer_metadata))

    def begin_connector_metadata_transaction(self) -> None:
        self.calls.append(("begin",))

    def commit_connector_metadata_transaction(self) -> None:
        self.calls.append(("commit",))

    def finalize_connector_metadata_transaction(self) -> None:
        self.calls.append(("finalize",))

    def abort_connector_metadata_transaction(self) -> None:
        self.calls.append(("abort",))

    def clear_peer_connector_metadata(self) -> None:
        self.calls.append(("clear",))

    def shutdown(self) -> None:
        self.calls.append(("shutdown",))


class _ProtocolV2DriftAdapter:
    """Public protocol double for drift between begin and bind."""

    KMESH_HANDOFF_PROTOCOL_VERSION = 2

    def __init__(self) -> None:
        self._drifted = False
        self._checkpoint: int | None = None
        self._pending_generation: int | None = None
        self._consumed_generation = 0

    @property
    def consumed_generation(self) -> int:
        """Return the last finalized generation."""
        return self._consumed_generation

    def induce_runtime_drift(self) -> None:
        """Make target binding unavailable after transaction begin."""
        self._drifted = True

    def restore_runtime(self) -> None:
        """Restore the public target behavior for a corrected retry."""
        self._drifted = False

    def begin_connector_metadata_transaction(self) -> None:
        """Checkpoint the generation before a public wrapper transaction."""
        self._checkpoint = self._consumed_generation
        self._pending_generation = None

    def bind_peer_connector_metadata(self, peer_metadata: object) -> None:
        """Bind a generation unless the injected runtime drifted."""
        if self._drifted:
            return
        generation = getattr(cast(Any, peer_metadata)[0], "step_generation", None)
        if type(generation) is not int or generation <= self._consumed_generation:
            raise ValueError("metadata generation is stale or already consumed")
        self._pending_generation = generation

    def commit_connector_metadata_transaction(self) -> None:
        """Fail if drift prevented the wrapper from binding metadata."""
        if self._checkpoint is None:
            raise RuntimeError("metadata transaction has no checkpoint")
        if self._pending_generation is None:
            raise RuntimeError("metadata transaction has no pending handoff")
        self._consumed_generation = self._pending_generation

    def finalize_connector_metadata_transaction(self) -> None:
        """Finish the successful public wrapper transaction."""
        if self._pending_generation is None:
            raise RuntimeError("metadata transaction has no pending handoff")

    def abort_connector_metadata_transaction(self) -> None:
        """Restore the checkpoint even while the runtime remains drifted."""
        if self._checkpoint is not None:
            self._consumed_generation = self._checkpoint
        self._checkpoint = None
        self._pending_generation = None


class _WrapperConstructionManager:
    """Small process-owner double for wrapper publication tests."""

    retained: set["_WrapperConstructionManager"] = set()
    provisional: dict[object, "_WrapperConstructionManager"] = {}

    def __init__(
        self,
        events: list[str],
        *,
        retain_errors: list[BaseException] | None = None,
        release_errors: list[BaseException] | None = None,
    ) -> None:
        self.events = events
        self.services_stopped = False
        self.retain_errors = list(retain_errors or ())
        self.release_errors = list(release_errors or ())

    def retain_construction_cleanup(self) -> None:
        self.events.append("retain-failed-manager")
        type(self).retained.add(self)

    def new_construction_handoff_receipt(self) -> object:
        return object()

    def retain_construction_handoff(self, receipt: object) -> None:
        self.events.append("retain-manager")
        if self.retain_errors:
            raise self.retain_errors.pop(0)
        type(self).provisional[receipt] = self

    def release_construction_handoff(self, receipt: object) -> None:
        self.events.append("release-manager")
        if self.release_errors:
            raise self.release_errors.pop(0)

    def promote_construction_handoff_failure(self, receipt: object) -> None:
        self.events.append("promote-manager")
        type(self).retained.add(self)

    @classmethod
    def _discard_incomplete_construction_owner(
        cls,
        manager: "_WrapperConstructionManager",
    ) -> None:
        cls.retained.discard(manager)

    def stop_services(self) -> None:
        self.events.append("stop-manager")
        self.services_stopped = True
        type(self)._discard_incomplete_construction_owner(self)
        adopted_receipts = tuple(
            receipt
            for receipt, manager in type(self).provisional.items()
            if manager is self
        )
        for receipt in adopted_receipts:
            type(self).provisional.pop(receipt, None)

    @classmethod
    def retry_retained(cls) -> int:
        for manager in tuple(cls.retained):
            manager.stop_services()
        return len(cls.retained)


class _WrapperConstructionImpl:
    """Returned adapter double with Manager-backed shutdown ownership."""

    def __init__(
        self,
        manager: _WrapperConstructionManager,
        shutdown_errors: list[BaseException] | None = None,
    ) -> None:
        self._manager = manager
        self.shutdown_errors = list(shutdown_errors or ())
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._manager.events.append("shutdown-impl")
        if self.shutdown_errors:
            raise self.shutdown_errors.pop(0)
        self._manager.stop_services()


def _publish_wrapper_impl(
    impl: _WrapperConstructionImpl,
    args: tuple[Any, ...],
) -> _WrapperConstructionImpl:
    """Model the adapter's callee-side result receipt in wrapper tests."""
    publisher = cast(Callable[[_WrapperConstructionImpl], None], args[-1])
    publisher(impl)
    return impl


def _wrapper_line_interrupt(
    function: Any,
    source: str,
    message: str,
    events: list[str],
) -> Any:
    lines, start = inspect.getsourcelines(function)
    line = next(
        start + offset for offset, text in enumerate(lines) if text.strip() == source
    )
    triggered = False

    def trace(frame: Any, event: str, arg: Any) -> Any:
        del arg
        nonlocal triggered
        if (
            not triggered
            and frame.f_code is function.__code__
            and event == "line"
            and frame.f_lineno == line
        ):
            triggered = True
            events.append("interrupt-wrapper")
            sys.settrace(None)
            raise KeyboardInterrupt(message)
        return trace

    return trace


def _retained_worker_line_interrupt(
    function: Any,
    source: str,
    thread_name: str,
    error: BaseException,
) -> Any:
    """Interrupt one retained worker at an exact post-publication line."""
    lines, start = inspect.getsourcelines(function)
    line = next(
        start + offset for offset, text in enumerate(lines) if text.strip() == source
    )
    triggered = False

    def trace(frame: Any, event: str, arg: Any) -> Any:
        del arg
        nonlocal triggered
        if (
            not triggered
            and frame.f_code is function.__code__
            and event == "line"
            and frame.f_lineno == line
            and threading.current_thread().name == thread_name
        ):
            triggered = True
            sys.settrace(None)
            raise error
        return trace

    return trace


_WRAPPER_MODULES = (
    connector_wrapper_module,
    connector_wrapper_085_module,
)


@pytest.mark.parametrize(
    "post_publication_line",
    [
        "self._manager = manager",
        "receipt = manager.new_construction_handoff_receipt()",
    ],
    ids=["connector", "manager-mirror"],
)
def test_external_owner_recovers_worker_interrupted_published_connector(
    post_publication_line: str,
) -> None:
    """Connector identity, not a later flag, proves factory ownership."""
    events: list[str] = []
    manager = _WrapperConstructionManager(events)
    impl = _WrapperConstructionImpl(manager)
    factory_calls = 0
    thread_name = "test-retained-wrapper-publication"
    worker_error = KeyboardInterrupt("after retained connector publication")

    def build_impl(
        publisher: Callable[[_WrapperConstructionImpl], None],
    ) -> _WrapperConstructionImpl:
        nonlocal factory_calls
        factory_calls += 1
        publisher(impl)
        return impl

    owner = RetainedConnectorConstructionOwner(
        build_impl,
        thread_name=thread_name,
    )
    trace = _retained_worker_line_interrupt(
        RetainedConnectorConstructionOwner._invoke,
        post_publication_line,
        thread_name,
        worker_error,
    )

    try:
        threading.settrace(trace)
        try:
            with pytest.raises(KeyboardInterrupt) as failure:
                owner.invoke()
        finally:
            threading.settrace(cast(Any, None))

        assert failure.value is worker_error
        assert owner.factory_returned
        assert owner.last_error is worker_error
        cleanup_failed_connector_construction(owner, worker_error)

        assert factory_calls == 1
        assert impl.shutdown_calls == 1
        assert manager.services_stopped
        assert events == [
            "retain-failed-manager",
            "shutdown-impl",
            "stop-manager",
        ]
        assert manager not in _WrapperConstructionManager.retained
    finally:
        _WrapperConstructionManager.retained.discard(manager)


def test_external_owner_commits_callback_completion_interrupt() -> None:
    """A returned connector commits success before wrapper publication."""
    events: list[str] = []
    manager = _WrapperConstructionManager(events)
    impl = _WrapperConstructionImpl(manager)
    factory_calls = 0
    thread_name = "test-retained-wrapper-completion-boundary"
    worker_error = KeyboardInterrupt("before retained completion receipt")

    def build_impl(
        publisher: Callable[[_WrapperConstructionImpl], None],
    ) -> _WrapperConstructionImpl:
        nonlocal factory_calls
        factory_calls += 1
        publisher(impl)
        return impl

    owner = RetainedConnectorConstructionOwner(
        build_impl,
        thread_name=thread_name,
    )
    trace = _retained_worker_line_interrupt(
        RetainedCall._publish_outcome,
        "attempt.outcome = outcome",
        thread_name,
        worker_error,
    )

    try:
        threading.settrace(trace)
        try:
            owner.invoke()
        finally:
            threading.settrace(cast(Any, None))

        assert owner.factory_returned
        assert owner.last_error is None
        assert owner.connector is impl
        owner.release_provisional_manager()

        assert factory_calls == 1
        assert impl.shutdown_calls == 0
        assert not manager.services_stopped
        assert events == [
            "retain-manager",
            "release-manager",
        ]
        assert manager not in _WrapperConstructionManager.retained
        impl.shutdown()
        assert manager not in _WrapperConstructionManager.provisional.values()
    finally:
        _WrapperConstructionManager.retained.discard(manager)


@pytest.mark.parametrize(
    "wrapper_module",
    _WRAPPER_MODULES,
    ids=["current", "v085"],
)
def test_external_wrapper_success_transfers_provisional_manager_owner(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_module: Any,
) -> None:
    """Normal STORE_ATTR releases the provisional process-level owner."""
    events: list[str] = []
    manager = _WrapperConstructionManager(events)
    impl = _WrapperConstructionImpl(manager)
    wrappers: list[Any] = []

    monkeypatch.setattr(
        wrapper_module.KVConnectorBase_V1,
        "__init__",
        lambda wrapper, *args, **kwargs: wrappers.append(wrapper),
    )
    monkeypatch.setattr(
        wrapper_module,
        "LMCacheConnectorV1Impl",
        lambda *args: _publish_wrapper_impl(impl, args),
    )

    wrapper = wrapper_module.LMCacheConnectorV1Dynamic(
        SimpleNamespace(),
        KVConnectorRole.SCHEDULER,
    )

    assert wrappers == [wrapper]
    assert wrapper._lmcache_engine is impl
    assert wrapper._lmcache_engine_construction_owner is None
    assert events == ["retain-manager", "release-manager"]
    assert manager not in _WrapperConstructionManager.retained
    assert impl.shutdown_calls == 0
    assert manager in _WrapperConstructionManager.provisional.values()

    impl.shutdown()
    assert manager not in _WrapperConstructionManager.provisional.values()


@pytest.mark.parametrize(
    "wrapper_module",
    _WRAPPER_MODULES,
    ids=["current", "v085"],
)
def test_external_wrapper_post_release_interrupt_promotes_adopted_owner(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_module: Any,
) -> None:
    """Both wrappers retain the exact slot through final constructor return."""
    events: list[str] = []
    manager = _WrapperConstructionManager(events)
    cleanup_interrupt = KeyboardInterrupt("during adopted adapter shutdown")
    impl = _WrapperConstructionImpl(manager, [cleanup_interrupt])
    wrappers: list[Any] = []

    monkeypatch.setattr(
        wrapper_module.KVConnectorBase_V1,
        "__init__",
        lambda wrapper, *args, **kwargs: wrappers.append(wrapper),
    )
    monkeypatch.setattr(
        wrapper_module,
        "LMCacheConnectorV1Impl",
        lambda *args: _publish_wrapper_impl(impl, args),
    )
    interrupt = _wrapper_line_interrupt(
        wrapper_module.LMCacheConnectorV1Dynamic.__init__,
        "self._lmcache_engine_construction_owner = None",
        "after successful Manager handoff release",
        events,
    )

    try:
        sys.settrace(interrupt)
        with pytest.raises(KeyboardInterrupt) as failure:
            wrapper_module.LMCacheConnectorV1Dynamic(
                SimpleNamespace(),
                KVConnectorRole.SCHEDULER,
            )
        assert "successful Manager handoff release" in str(failure.value)
        assert failure.value.__cause__ is cleanup_interrupt
        assert len(wrappers) == 1
        assert manager in _WrapperConstructionManager.retained
        assert manager in _WrapperConstructionManager.provisional.values()
        assert events == [
            "retain-manager",
            "release-manager",
            "interrupt-wrapper",
            "promote-manager",
            "shutdown-impl",
        ]

        assert _WrapperConstructionManager.retry_retained() == 0
        assert manager.services_stopped
        assert manager not in _WrapperConstructionManager.provisional.values()
    finally:
        sys.settrace(None)
        _WrapperConstructionManager.retained.discard(manager)


@pytest.mark.parametrize(
    "wrapper_module",
    _WRAPPER_MODULES,
    ids=["current", "v085"],
)
def test_external_wrapper_recovers_impl_lost_before_store_attr(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_module: Any,
) -> None:
    """A returned adapter is retained and shut down before wrapper unwind."""
    events: list[str] = []
    manager = _WrapperConstructionManager(events)
    impl = _WrapperConstructionImpl(manager)
    wrappers: list[Any] = []
    factory_calls = 0

    def base_init(wrapper: Any, *args: Any, **kwargs: Any) -> None:
        wrappers.append(wrapper)

    def build_impl(*args: Any) -> _WrapperConstructionImpl:
        nonlocal factory_calls
        factory_calls += 1
        cast(Callable[[_WrapperConstructionImpl], None], args[-1])(impl)
        return impl

    monkeypatch.setattr(wrapper_module.KVConnectorBase_V1, "__init__", base_init)
    monkeypatch.setattr(wrapper_module, "LMCacheConnectorV1Impl", build_impl)
    interrupt = _wrapper_line_interrupt(
        wrapper_module.LMCacheConnectorV1Dynamic.__init__,
        "self._lmcache_engine = connector",
        "after external adapter return",
        events,
    )

    sys.settrace(interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="external adapter return"):
            wrapper_module.LMCacheConnectorV1Dynamic(
                SimpleNamespace(),
                KVConnectorRole.SCHEDULER,
            )
    finally:
        sys.settrace(None)

    assert len(wrappers) == 1
    wrapper = wrappers[0]
    owner = wrapper._lmcache_engine_construction_owner
    assert owner.phase is ConnectorConstructionPhase.INVOCATION_MAY_HAVE_BEGUN
    assert owner.connector is impl
    assert not hasattr(wrapper, "_lmcache_engine")
    assert factory_calls == 1
    assert impl.shutdown_calls == 1
    assert events == [
        "retain-manager",
        "interrupt-wrapper",
        "promote-manager",
        "shutdown-impl",
        "stop-manager",
    ]
    assert manager not in _WrapperConstructionManager.retained


@requires_instruction_monitoring
@pytest.mark.parametrize(
    "wrapper_module",
    _WRAPPER_MODULES,
    ids=["current", "v085"],
)
def test_external_wrapper_exact_store_interrupt_recovers_impl(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_module: Any,
) -> None:
    """Both real wrapper bodies recover before `_lmcache_engine` STORE_ATTR."""
    events: list[str] = []
    manager = _WrapperConstructionManager(events)
    impl = _WrapperConstructionImpl(manager)
    wrappers: list[Any] = []

    monkeypatch.setattr(
        wrapper_module.KVConnectorBase_V1,
        "__init__",
        lambda wrapper, *args, **kwargs: wrappers.append(wrapper),
    )
    monkeypatch.setattr(
        wrapper_module,
        "LMCacheConnectorV1Impl",
        lambda *args: _publish_wrapper_impl(impl, args),
    )
    injected = KeyboardInterrupt("before wrapper connector STORE_ATTR")
    interrupt = InstructionInterruption(
        wrapper_module.LMCacheConnectorV1Dynamic.__init__,
        instruction_offset(
            wrapper_module.LMCacheConnectorV1Dynamic.__init__,
            "STORE_ATTR",
            argval="_lmcache_engine",
        ),
        injected,
    )

    with interrupt, pytest.raises(KeyboardInterrupt) as failure:
        wrapper_module.LMCacheConnectorV1Dynamic(
            SimpleNamespace(),
            KVConnectorRole.SCHEDULER,
        )
    interrupt.assert_fired_once()

    assert failure.value is injected
    assert len(wrappers) == 1
    wrapper = wrappers[0]
    assert not hasattr(wrapper, "_lmcache_engine")
    assert impl.shutdown_calls == 1
    assert manager.services_stopped
    assert events == [
        "retain-manager",
        "promote-manager",
        "shutdown-impl",
        "stop-manager",
    ]
    assert manager not in _WrapperConstructionManager.retained


@pytest.mark.parametrize(
    "wrapper_module",
    _WRAPPER_MODULES,
    ids=["current", "v085"],
)
def test_external_wrapper_precall_interrupt_constructs_nothing(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_module: Any,
) -> None:
    """PRECALL wrapper failure neither invokes nor recovers an adapter."""
    events: list[str] = []
    wrappers: list[Any] = []
    factory_calls = 0

    def base_init(wrapper: Any, *args: Any, **kwargs: Any) -> None:
        wrappers.append(wrapper)

    def build_impl(*args: Any) -> _WrapperConstructionImpl:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("PRECALL invoked adapter constructor")

    monkeypatch.setattr(wrapper_module.KVConnectorBase_V1, "__init__", base_init)
    monkeypatch.setattr(wrapper_module, "LMCacheConnectorV1Impl", build_impl)
    interrupt = _wrapper_line_interrupt(
        wrapper_module.LMCacheConnectorV1Dynamic.__init__,
        "owner.invoke()",
        "before external adapter admission",
        events,
    )

    sys.settrace(interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="adapter admission"):
            wrapper_module.LMCacheConnectorV1Dynamic(
                SimpleNamespace(),
                KVConnectorRole.SCHEDULER,
            )
    finally:
        sys.settrace(None)

    assert len(wrappers) == 1
    owner = wrappers[0]._lmcache_engine_construction_owner
    assert owner.phase is ConnectorConstructionPhase.PRECALL
    assert factory_calls == 0
    assert events == ["interrupt-wrapper"]


def test_external_wrapper_cleanup_interrupt_retains_public_manager_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted exact adapter shutdown leaves its Manager retryable."""
    wrapper_module = connector_wrapper_module
    events: list[str] = []
    manager = _WrapperConstructionManager(events)
    cleanup_interrupt = KeyboardInterrupt("during exact adapter shutdown")
    impl = _WrapperConstructionImpl(manager, [cleanup_interrupt])

    monkeypatch.setattr(
        wrapper_module.KVConnectorBase_V1,
        "__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        wrapper_module,
        "LMCacheConnectorV1Impl",
        lambda *args: _publish_wrapper_impl(impl, args),
    )
    interrupt = _wrapper_line_interrupt(
        wrapper_module.LMCacheConnectorV1Dynamic.__init__,
        "self._lmcache_engine = connector",
        "after external adapter return",
        events,
    )

    try:
        sys.settrace(interrupt)
        with pytest.raises(KeyboardInterrupt) as failure:
            wrapper_module.LMCacheConnectorV1Dynamic(
                SimpleNamespace(),
                KVConnectorRole.SCHEDULER,
            )
        assert "external adapter return" in str(failure.value)
        assert failure.value.__cause__ is cleanup_interrupt
        assert manager in _WrapperConstructionManager.retained
        assert events == [
            "retain-manager",
            "interrupt-wrapper",
            "promote-manager",
            "shutdown-impl",
        ]

        assert _WrapperConstructionManager.retry_retained() == 0
        assert manager.services_stopped
        assert events[-1:] == ["stop-manager"]
    finally:
        sys.settrace(None)
        _WrapperConstructionManager.retained.discard(manager)


def test_external_wrapper_reclaims_impl_after_provisional_retain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-return Manager-retain error is not a constructor failure."""
    wrapper_module = connector_wrapper_module
    events: list[str] = []
    retain_error = KeyboardInterrupt("during provisional Manager retention")
    manager = _WrapperConstructionManager(
        events,
        retain_errors=[retain_error],
    )
    impl = _WrapperConstructionImpl(manager)

    monkeypatch.setattr(
        wrapper_module.KVConnectorBase_V1,
        "__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        wrapper_module,
        "LMCacheConnectorV1Impl",
        lambda *args: _publish_wrapper_impl(impl, args),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        wrapper_module.LMCacheConnectorV1Dynamic(
            SimpleNamespace(),
            KVConnectorRole.SCHEDULER,
        )

    assert failure.value is retain_error
    assert impl.shutdown_calls == 1
    assert manager.services_stopped
    assert manager not in _WrapperConstructionManager.retained
    assert events == [
        "retain-manager",
        "promote-manager",
        "shutdown-impl",
        "stop-manager",
    ]


def test_external_wrapper_reclaims_impl_after_partial_owner_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupt after provisional discard still shuts down the adapter."""
    wrapper_module = connector_wrapper_module
    events: list[str] = []
    release_error = KeyboardInterrupt("after provisional Manager discard")
    manager = _WrapperConstructionManager(
        events,
        release_errors=[release_error],
    )
    impl = _WrapperConstructionImpl(manager)

    monkeypatch.setattr(
        wrapper_module.KVConnectorBase_V1,
        "__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        wrapper_module,
        "LMCacheConnectorV1Impl",
        lambda *args: _publish_wrapper_impl(impl, args),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        wrapper_module.LMCacheConnectorV1Dynamic(
            SimpleNamespace(),
            KVConnectorRole.SCHEDULER,
        )

    assert failure.value is release_error
    assert impl.shutdown_calls == 1
    assert manager.services_stopped
    assert manager not in _WrapperConstructionManager.retained
    assert events == [
        "retain-manager",
        "release-manager",
        "promote-manager",
        "shutdown-impl",
        "stop-manager",
    ]


@pytest.mark.parametrize(
    "wrapper_module",
    _WRAPPER_MODULES,
    ids=["current", "v085"],
)
def test_public_retry_ignores_healthy_wrapper_handoff(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_module: Any,
) -> None:
    """Both wrappers keep a healthy provisional Manager out of retry."""
    events: list[str] = []
    manager = _WrapperConstructionManager(events)
    impl = _WrapperConstructionImpl(manager)
    retain_entered = threading.Event()
    allow_retain = threading.Event()
    original_retain = manager.retain_construction_handoff
    wrappers: list[Any] = []
    failures: list[BaseException] = []

    def block_after_retain(receipt: object) -> None:
        original_retain(receipt)
        retain_entered.set()
        if not allow_retain.wait(timeout=5.0):
            raise RuntimeError("timed out holding healthy wrapper handoff")

    def construct_wrapper() -> None:
        try:
            wrappers.append(
                wrapper_module.LMCacheConnectorV1Dynamic(
                    SimpleNamespace(),
                    KVConnectorRole.SCHEDULER,
                )
            )
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(
        wrapper_module.KVConnectorBase_V1,
        "__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        wrapper_module,
        "LMCacheConnectorV1Impl",
        lambda *args: _publish_wrapper_impl(impl, args),
    )
    monkeypatch.setattr(manager, "retain_construction_handoff", block_after_retain)

    constructor = threading.Thread(target=construct_wrapper)
    constructor.start()
    try:
        assert retain_entered.wait(timeout=5.0)
        assert manager in _WrapperConstructionManager.provisional.values()
        assert manager not in _WrapperConstructionManager.retained
        assert _WrapperConstructionManager.retry_retained() == 0
        assert not manager.services_stopped
        assert impl.shutdown_calls == 0
    finally:
        allow_retain.set()
        constructor.join(timeout=5.0)

    assert not constructor.is_alive()
    assert failures == []
    assert len(wrappers) == 1
    assert wrappers[0]._lmcache_engine is impl
    assert events == ["retain-manager", "release-manager"]

    impl.shutdown()
    assert manager not in _WrapperConstructionManager.provisional.values()


def test_dynamic_wrapper_retries_same_generation_after_post_begin_drift() -> None:
    adapter = _ProtocolV2DriftAdapter()
    wrapper = LMCacheConnectorV1Dynamic.__new__(LMCacheConnectorV1Dynamic)
    cast(Any, wrapper)._lmcache_engine = adapter
    peer_metadata = cast(Any, (SimpleNamespace(step_generation=1),))

    wrapper.begin_connector_metadata_transaction()
    adapter.induce_runtime_drift()
    wrapper.bind_peer_connector_metadata(peer_metadata)
    with pytest.raises(RuntimeError, match="no pending handoff"):
        wrapper.commit_connector_metadata_transaction()
    wrapper.abort_connector_metadata_transaction()

    adapter.restore_runtime()
    wrapper.begin_connector_metadata_transaction()
    wrapper.bind_peer_connector_metadata(peer_metadata)
    wrapper.commit_connector_metadata_transaction()
    wrapper.finalize_connector_metadata_transaction()

    assert adapter.consumed_generation == 1


def test_dynamic_wrapper_advertises_exact_protocol_and_forwards_handoff() -> None:
    recorder = _ForwardRecorder()
    wrapper = LMCacheConnectorV1Dynamic.__new__(LMCacheConnectorV1Dynamic)
    cast(Any, wrapper)._lmcache_engine = recorder
    request = _request()
    sidecars = (_sidecar(),)
    peer_metadata = cast(Any, (SimpleNamespace(), SimpleNamespace()))

    assert type(LMCacheConnectorV1Dynamic.KMESH_HANDOFF_PROTOCOL_VERSION) is int
    assert LMCacheConnectorV1Dynamic.KMESH_HANDOFF_PROTOCOL_VERSION == 2
    assert type(wrapper.KMESH_HANDOFF_PROTOCOL_VERSION) is int
    assert wrapper.KMESH_HANDOFF_PROTOCOL_VERSION == 2
    assert type(LMCacheConnectorV1Impl.KMESH_HANDOFF_PROTOCOL_VERSION) is int
    assert LMCacheConnectorV1Impl.KMESH_HANDOFF_PROTOCOL_VERSION == 2

    wrapper.prepare_request_lookup(request, sidecars)
    wrapper.validate_peer_connector_metadata(peer_metadata)
    wrapper.begin_connector_metadata_transaction()
    wrapper.bind_peer_connector_metadata(peer_metadata)
    wrapper.commit_connector_metadata_transaction()
    wrapper.finalize_connector_metadata_transaction()
    wrapper.abort_connector_metadata_transaction()
    wrapper.clear_peer_connector_metadata()
    wrapper.shutdown()

    assert recorder.calls == [
        ("prepare", request, sidecars),
        ("validate", peer_metadata),
        ("begin",),
        ("bind", peer_metadata),
        ("commit",),
        ("finalize",),
        ("abort",),
        ("clear",),
        ("shutdown",),
    ]


def test_lockstep_profile_dispatches_per_member_group_execution(
    runtime: KMeshRuntimeEvidence,
) -> None:
    worker, _metadata = _collective_worker(runtime)
    state = cast(Any, worker)

    worker.start_load_kv(cast(Any, _forward_context()))

    assert state._test_blender.group_collective_flags == [False]


def test_batched_profile_dispatches_collective_position_recovery(
    runtime: KMeshRuntimeEvidence,
) -> None:
    worker, _metadata = _collective_worker(runtime)
    state = cast(Any, worker)
    state._target_profile = COLLECTIVE_BATCHED_POSITION_RECOVERY_V1
    counts = LMCBlendOperationCounts(
        serial_rope_operations=2,
        serial_diff_operations=1,
        position_recovery_collective_operations=2,
    )
    state._test_blender.group_result_override = LMCBlendGroupResult(
        members=(
            _collective_member("request-a", operation_counts=counts),
            _collective_member("request-b", operation_counts=counts),
        )
    )

    worker.start_load_kv(cast(Any, _forward_context()))

    assert state._test_blender.group_collective_flags == [True]
    assert state._kmesh_failed_worker_requests == set()
    assert len(state._kmesh_collective_results) == 1
    _plan_id, stored = state._kmesh_collective_results[0]
    assert all(
        member.operation_counts.position_recovery_collective_operations == 2
        and member.operation_counts.position_recovery_serial_operations == 0
        for member in stored.members
    )


def test_batched_profile_rejects_member_without_layerwise_recovery(
    runtime: KMeshRuntimeEvidence,
) -> None:
    worker, _metadata = _collective_worker(runtime)
    state = cast(Any, worker)
    state._target_profile = COLLECTIVE_BATCHED_POSITION_RECOVERY_V1
    state._test_blender.group_result_override = LMCBlendGroupResult(
        members=(
            _collective_member("request-a"),
            _collective_member("request-b"),
        )
    )

    worker.start_load_kv(cast(Any, _forward_context()))

    assert state._kmesh_collective_results == []
    assert state._kmesh_failed_worker_requests == {"request-a", "request-b"}
    assert state._invalid_block_ids == {3, 5, 7, 9}


def test_lockstep_profile_rejects_batched_recovery_claims(
    runtime: KMeshRuntimeEvidence,
) -> None:
    counts = LMCBlendOperationCounts(
        serial_rope_operations=2,
        serial_diff_operations=1,
        position_recovery_collective_operations=2,
    )
    _assert_malformed_collective_result_fails_closed(
        runtime,
        LMCBlendGroupResult(
            members=(
                _collective_member("request-a", operation_counts=counts),
                _collective_member("request-b", operation_counts=counts),
            )
        ),
    )


def test_qkv_profile_dispatches_both_collective_flags(
    runtime: KMeshRuntimeEvidence,
) -> None:
    worker, _metadata = _collective_worker(runtime)
    state = cast(Any, worker)
    state._target_profile = COLLECTIVE_BATCHED_QKV_V1
    counts = LMCBlendOperationCounts(
        collective_rope_operations=2,
        collective_diff_operations=1,
        position_recovery_collective_operations=2,
    )
    state._test_blender.group_result_override = LMCBlendGroupResult(
        members=(
            _collective_member("request-a", operation_counts=counts),
            _collective_member("request-b", operation_counts=counts),
        )
    )

    worker.start_load_kv(cast(Any, _forward_context()))

    assert state._test_blender.group_collective_flags == [True]
    assert state._test_blender.group_qkv_flags == [True]
    assert state._kmesh_failed_worker_requests == set()
    assert len(state._kmesh_collective_results) == 1


def test_qkv_profile_rejects_serial_member_counts(
    runtime: KMeshRuntimeEvidence,
) -> None:
    worker, _metadata = _collective_worker(runtime)
    state = cast(Any, worker)
    state._target_profile = COLLECTIVE_BATCHED_QKV_V1
    state._test_blender.group_result_override = LMCBlendGroupResult(
        members=(
            _collective_member("request-a"),
            _collective_member("request-b"),
        )
    )

    worker.start_load_kv(cast(Any, _forward_context()))

    assert state._kmesh_collective_results == []
    assert state._kmesh_failed_worker_requests == {"request-a", "request-b"}


def test_position_recovery_profile_rejects_model_side_collective_claims(
    runtime: KMeshRuntimeEvidence,
) -> None:
    worker, _metadata = _collective_worker(runtime)
    state = cast(Any, worker)
    state._target_profile = COLLECTIVE_BATCHED_POSITION_RECOVERY_V1
    counts = LMCBlendOperationCounts(
        serial_rope_operations=2,
        serial_diff_operations=1,
        collective_rope_operations=2,
        position_recovery_collective_operations=2,
    )
    state._test_blender.group_result_override = LMCBlendGroupResult(
        members=(
            _collective_member("request-a", operation_counts=counts),
            _collective_member("request-b", operation_counts=counts),
        )
    )

    worker.start_load_kv(cast(Any, _forward_context()))

    assert state._kmesh_collective_results == []
    assert state._kmesh_failed_worker_requests == {"request-a", "request-b"}
