# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Union, cast
from uuid import UUID, uuid4
import math
import os
import sys
import threading

# Third Party
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRequestSidecar,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import RequestStatus
from vllm.v1.utils import ConstantList
from vllm.version import __version__ as VLLM_VERSION
import torch

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache import utils
from lmcache.banner import print_banner_once
from lmcache.integration.vllm.kmesh import (
    KMESH_REQUEST_SIDECAR_PROVIDER_ID,
    KMeshRequestExecutionEvidence,
    KMeshRequestLookupDescriptor,
    KMeshRuntimeEvidence,
    bind_kmesh_request_lookup,
    parse_kmesh_request_sidecar,
    validate_kmesh_request_lookup_evidence,
)
from lmcache.integration.vllm.kmesh_runtime import build_kmesh_runtime_evidence
from lmcache.integration.vllm.target_profile import (
    COLLECTIVE_BATCHED_POSITION_RECOVERY_V1,
    COLLECTIVE_BATCHED_QKV_V1,
    KMESH_COLLECTIVE_PROFILES,
    KMESH_PIC_V1_PROFILES,
    requested_target_profile,
    validate_target_profile,
)
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    lmcache_get_or_create_config,
)
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheStoreEvent, _lmcache_nvtx_annotate, cdiv
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.compute.blend import (
    LMCBlenderBuilder,
    LMCBlendGroupMemberResult,
    LMCBlendGroupOutcome,
    LMCBlendGroupRequest,
    LMCBlendGroupResult,
    LMCBlendOperationCounts,
)
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import validate_and_set_config_value
from lmcache.v1.lifecycle import RetainedCall
from lmcache.v1.manager import LMCacheManager
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    # First Party
    from lmcache.v1.compute.blend.blender import LMCBlender
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)
_MANAGER_UNPUBLISHED = object()


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0
    total_chunks: int = 0
    receiver_query_port: Optional[list[int]] = None


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


@dataclass(frozen=True, slots=True)
class _KMeshPreparedLookup:
    """Scheduler-only live request ownership before metadata transport."""

    descriptor: KMeshRequestLookupDescriptor
    request_object: object = field(repr=False, compare=False, hash=False)
    execution: Optional[KMeshRequestExecutionEvidence]

    def __post_init__(self) -> None:
        if type(self.descriptor) is not KMeshRequestLookupDescriptor:
            raise TypeError("descriptor must be exactly KMeshRequestLookupDescriptor")
        if self.descriptor.permits_serial_pic:
            if type(self.execution) is not KMeshRequestExecutionEvidence:
                raise TypeError(
                    "actionable prepared lookup requires exact execution evidence"
                )
            self.execution.validate_peer_descriptor(self.descriptor)
        elif self.execution is not None:
            raise ValueError("fallback prepared lookup cannot carry execution evidence")
        validate_kmesh_request_lookup_evidence(
            self.descriptor,
            self.request_object,
        )

    def validate_repeated_prepare(
        self,
        request: object,
        descriptor: KMeshRequestLookupDescriptor,
    ) -> None:
        """Reject request replacement or descriptor drift between probes."""
        if request is not self.request_object:
            raise ValueError("KMesh request object changed between lookup probes")
        validate_kmesh_request_lookup_evidence(self.descriptor, request)
        if type(descriptor) is not KMeshRequestLookupDescriptor:
            raise TypeError("descriptor must be exactly KMeshRequestLookupDescriptor")
        if descriptor.to_wire_payload() != self.descriptor.to_wire_payload():
            raise ValueError("KMesh descriptor changed between lookup probes")


@dataclass(frozen=True, slots=True)
class _KMeshConsumedSchedulerExecution:
    """One scheduler handoff state consumed by a staged metadata build."""

    allocation_observed: bool
    descriptor: Optional[KMeshRequestLookupDescriptor]
    execution: Optional[KMeshRequestExecutionEvidence]

    def __post_init__(self) -> None:
        if type(self.allocation_observed) is not bool:
            raise TypeError("allocation_observed must be a boolean")
        if not self.allocation_observed:
            if self.descriptor is not None or self.execution is not None:
                raise ValueError("absent allocation cannot carry KMesh evidence")
            return
        if type(self.descriptor) is not KMeshRequestLookupDescriptor:
            raise TypeError("allocated handoff requires an exact descriptor")
        if self.descriptor.permits_serial_pic:
            if type(self.execution) is not KMeshRequestExecutionEvidence:
                raise TypeError("actionable handoff requires exact execution evidence")
        elif self.execution is not None:
            raise ValueError("fallback handoff cannot carry execution evidence")


@dataclass(slots=True)
class _KMeshLayerwiseStoreSession:
    """One bounded explicit-span store owned by a worker request."""

    generator: Generator[None, None, None] = field(repr=False)
    cleanup: Callable[[], None] = field(repr=False)
    generation: int
    expected_layer_names: tuple[str, ...]
    completed_layers: int = 0
    flushed: bool = False
    closed: bool = False

    def step_layer(self, layer_name: str) -> None:
        """Advance exactly one layer hook and record its completion."""
        if self.closed:
            raise RuntimeError("KMesh store session is closed")
        if self.completed_layers >= len(self.expected_layer_names):
            raise RuntimeError("KMesh store session received too many layers")
        expected = self.expected_layer_names[self.completed_layers]
        if layer_name != expected:
            raise ValueError(
                "KMesh store layer order differs from registered KV caches: "
                f"expected {expected!r}, received {layer_name!r}"
            )
        next(self.generator)
        self.completed_layers += 1

    def finish(self) -> bool:
        """Flush the final boundary only after every layer hook ran."""
        if self.closed:
            return False
        if self.completed_layers != len(self.expected_layer_names):
            self.close()
            return False
        if not self.flushed:
            next(self.generator)
            self.flushed = True
        self.close()
        return True

    def close(self) -> None:
        """Close without advancing into a layer whose hook never ran."""
        if self.closed:
            return
        self.generator.close()
        self.cleanup()
        self.closed = True


@dataclass(frozen=True, slots=True)
class _KMeshWorkerRequestContract:
    """Immutable worker ownership and data-plane contract frozen at bind."""

    execution: KMeshRequestExecutionEvidence
    lookup_id: str
    token_ids: tuple[int, ...]
    slot_mapping: tuple[int, ...]
    load_spec: tuple[int, int, bool]
    save_spec: tuple[int, bool]
    is_last_prefill: bool

    @property
    def request_id(self) -> str:
        """Return the request ID carried by the immutable execution evidence."""
        return self.execution.request_id

    @property
    def advertised_tokens(self) -> int:
        """Return the exact external-token count advertised to vLLM."""
        _vllm_cached, lmcache_cached, _can_load = self.load_spec
        return lmcache_cached - int(
            lmcache_cached == len(self.token_ids) and lmcache_cached > 0
        )


@dataclass(frozen=True, slots=True)
class _KMeshWorkerPlanMember:
    """One plan-selected request and its exact local recovery spans."""

    request_id: str
    explicit_spans: ExplicitTokenSpans


@dataclass(frozen=True, slots=True)
class _KMeshWorkerReusePlan:
    """Immutable worker-local projection of one transported reuse plan."""

    plan_id: str
    reuse_group_id: str
    members: tuple[_KMeshWorkerPlanMember, ...]


@dataclass(frozen=True, slots=True)
class _KMeshWorkerStepContract:
    """Immutable composite handoff plus registered KV-layer identities."""

    generation: int
    requests: tuple[_KMeshWorkerRequestContract, ...]
    plans: tuple[_KMeshWorkerReusePlan, ...]
    layer_names: tuple[str, ...]
    kv_cache_ids: tuple[int, ...]


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params and sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    allocated_block_ids: list[int]

    # Unique LMCache lookup attempt transferred to this worker batch.
    lookup_id: Optional[str] = None

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Exact scheduler evidence for the online serial-PIC target path.
    kmesh_execution: Optional[KMeshRequestExecutionEvidence] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    # The number of tokens that are cached in LMCache for this request
    num_lmcache_cached_tokens: int = 0

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
        lookup_id: Optional[str] = None,
        kmesh_execution: Optional[KMeshRequestExecutionEvidence] = None,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # tuple[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            # According to the vLLM code
            # (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/
            # sched/scheduler.py#L943),
            # only one KVCacheGroup is supported in connector for now.

            # TODO: Please support multiple KVCacheGroup in connector.
            # NOTE: Also, `update` method in RequestTracker should be
            # updated accordingly.
            unfolded_block_ids = new_request.block_ids[0].copy()

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            lookup_id=lookup_id,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
            kmesh_execution=kmesh_execution,
            num_lmcache_cached_tokens=lmcache_cached_tokens,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
        preempted: bool = False,
        lmcache_cached_tokens: int = 0,
        vllm_cached_tokens: int = 0,
        all_token_ids: Optional[list[int]] = None,
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again

        vllm_cached_tokens: the number of tokens that are cached in vLLM
        is only used for preempted requests
        all_token_ids: the full token list from the vLLM request, used to
        restore token_ids for preempted requests to ensure chunk keys match
        """

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            # If input is a list, flatten it to handle potential nesting.
            # This also correctly processes already-flat lists.
            new_block_ids = [
                i
                for elem in new_block_ids
                for i in (elem if isinstance(elem, list) else [elem])
            ]
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")

        if preempted:
            assert all_token_ids is not None, (
                f"Preempted request {self.req_id} has no all_token_ids"
            )
            # the block ids will change after preemption
            self.allocated_block_ids = new_block_ids
            # reset the number of saved tokens
            self.num_saved_tokens = lmcache_cached_tokens
            num_computed_tokens = max(lmcache_cached_tokens, vllm_cached_tokens)

            # FIX: For preempted requests, restore token_ids from the full
            # token list to ensure chunk keys match what was used during
            # lookup. The lookup uses request.all_token_ids, so we need the
            # same tokens for retrieve.
            num_tokens_needed = max(
                num_computed_tokens + len(new_token_ids),
                lmcache_cached_tokens,
            )
            self.token_ids = all_token_ids[:num_tokens_needed]
        else:
            self.allocated_block_ids.extend(new_block_ids)
            self.token_ids.extend(new_token_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor

    # Unique LMCache lookup attempt whose pins this worker batch owns.
    lookup_id: Optional[str] = None

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None

    # Pickle-safe evidence compared against the worker's KMesh peer metadata.
    kmesh_execution: Optional[KMeshRequestExecutionEvidence] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len >= tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None and tracker.kmesh_execution is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        token_ids = input_token_ids[:num_tokens_to_save]

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            # TODO: Optimize this
            token_tensor = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_tensor, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_tensor.tolist()

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks"
                " for request %s. "
                "Something might be wrong in scheduling logic!",
                tracker.req_id,
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: log if the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens (%d cached in vLLM) for request %s",
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
                tracker.req_id,
            )

        # For disagg requests, compute total_chunks for sender admission control.
        if tracker.disagg_spec is not None and tracker.disagg_spec.total_chunks == 0:
            # Only compute once (on first batch)
            total_chunks_for_req = math.ceil(tracker.prompt_len / lmcache_chunk_size)
            tracker.disagg_spec.total_chunks = total_chunks_for_req

        # Note: We keep load_spec even when can_load=False to pass metrics to worker
        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            lookup_id=tracker.lookup_id,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
            kmesh_execution=tracker.kmesh_execution,
        )


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class _RetainedManagerConstructionOwner:
    """Retain one target Manager call and its exact successful result.

    ``KeyboardInterrupt`` is delivered to the adapter-construction thread. By
    running the target Manager constructor on a retained worker, an interrupt
    of the waiter cannot erase a Manager that has returned but has not yet
    reached ``LMCacheConnectorV1Impl._manager``. The adapter's failure path can
    resume the same call and stop that exact result.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        service_factory: Any,
        connector: "LMCacheConnectorV1Impl",
        role: KVConnectorRole,
    ) -> None:
        self._config = config
        self._service_factory = service_factory
        self._connector = connector
        self._manager: Union[object, LMCacheManager] = _MANAGER_UNPUBLISHED
        self._handoff_receipt: Optional[Any] = None
        promotion_thread_name = (
            "lmcache-vllm-manager-failure-promotion-" + role.name.lower()
        )
        # Prepublish the promotion attempt before Manager construction. Once
        # cleanup admits it, an interruption of the adapter thread cannot
        # strand the still-healthy receipt between failure and terminal stop.
        self._failure_promotion_call = RetainedCall(
            self._promote_failed_handoff_once,
            # Exact receipt promotion is monotonic and idempotent. A worker
            # error before/after phase publication may safely replay the same
            # token; it can never target another construction generation.
            retry_safe_failure=True,
            thread_name=promotion_thread_name,
        )
        self._call = RetainedCall(
            self._invoke,
            thread_name=("lmcache-vllm-manager-construction-" + role.name.lower()),
        )

    def _invoke(self) -> None:
        manager = LMCacheManager(
            self._config,
            self._service_factory,
            connector=self._connector,
            construction_result_publisher=self._publish_manager,
        )
        if self.manager is not manager:
            raise RuntimeError("target Manager returned a different published result")
        # Publish the exact successful constructor result to this retained
        # owner before the provisional process-registry operation. If that
        # operation itself raises, adapter unwind can still recover and stop
        # this Manager without re-running its constructor.
        receipt = manager.new_construction_handoff_receipt()
        # Publish the exact token in this owner before registry insertion. If
        # registration itself is interrupted after its side effect, recovery
        # can promote this same receipt rather than guessing by Manager ID.
        self._handoff_receipt = receipt
        manager.retain_construction_handoff(receipt)

    def _publish_manager(self, manager: LMCacheManager) -> None:
        """Accept the exact Manager before its constructor can return."""
        try:
            self._manager = manager
        except BaseException:
            self._manager = manager

    @property
    def manager(self) -> LMCacheManager:
        """Return the exact Manager result once its constructor returned."""
        manager = self._manager
        if manager is _MANAGER_UNPUBLISHED:
            raise RuntimeError("target Manager construction has no retained result")
        return cast(LMCacheManager, manager)

    @property
    def manager_returned(self) -> bool:
        """Whether the exact Manager exists even if provisional retain failed."""
        return self._manager is not _MANAGER_UNPUBLISHED

    @property
    def last_error(self) -> Optional[BaseException]:
        """Return the exact sticky Manager-construction failure, if any."""
        return self._call.last_error

    def run(self) -> None:
        """Start or resume the exactly-once Manager construction."""
        self._call.run()

    def release_handoff(self) -> None:
        """Release this exact provisional receipt after full adapter success."""
        manager = self.manager
        receipt = self._handoff_receipt
        if receipt is None:
            raise RuntimeError("target Manager handoff has no exact receipt")
        manager.release_construction_handoff(receipt)

    def _promote_failed_handoff_once(self) -> None:
        """Worker body publishing the exact failure token once."""
        manager = self.manager
        receipt = self._handoff_receipt
        if receipt is None:
            # Receipt preparation itself failed after Manager publication.
            # The exact Manager is still retained by this owner, so publish a
            # failure-only cleanup owner before attempting terminal cleanup.
            manager.retain_construction_cleanup()
            return
        manager.promote_construction_handoff_failure(receipt)

    def promote_failed_handoff(self) -> None:
        """Make this handoff retryable through a retained promotion call."""
        try:
            self._failure_promotion_call.run()
        except BaseException as promotion_error:
            # RetainedCall publishes its exact attempt before Thread.start and
            # keeps a started worker reachable. Resume it once before honoring
            # the caller interruption so failed ownership cannot remain hidden
            # merely because the adapter waiter was interrupted at admission
            # or after phase publication.
            try:
                self._failure_promotion_call.run()
            except BaseException as recovery_error:
                if recovery_error is not promotion_error:
                    raise promotion_error from recovery_error
            raise


class LMCacheConnectorV1Impl:
    KMESH_HANDOFF_PROTOCOL_VERSION = 2

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
        construction_result_publisher: Optional[
            Callable[["LMCacheConnectorV1Impl"], None]
        ] = None,
    ):
        # Banner from the scheduler role only, so tensor-parallel
        # deployments print it once rather than once per worker.
        if role == KVConnectorRole.SCHEDULER:
            print_banner_once(sys.stderr)

        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size

        # Load and configure LMCache config
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        self._apply_extra_config(config, vllm_config)

        # vLLM connector overrides are applied after the config file/env/remote
        # sources have already been validated.  Re-run the generic invariants
        # before any service can be constructed, then enforce an explicitly
        # requested narrow target envelope over that fully resolved config.
        config.validate()
        target_profile = requested_target_profile(vllm_config)
        validate_target_profile(
            config,
            target_profile,
            role=role,
            vllm_config=vllm_config,
        )
        self.config = config
        self._target_profile = target_profile
        self._kmesh_runtime_evidence = (
            build_kmesh_runtime_evidence(
                vllm_config,
                config,
                role=role,
                target_profile=target_profile,
            )
            if target_profile in KMESH_PIC_V1_PROFILES
            else None
        )

        service_factory = VllmServiceFactory(
            config,
            vllm_config,
            role.name.lower(),
            target_profile=target_profile,
        )

        try:
            if target_profile in KMESH_PIC_V1_PROFILES:
                self._acquire_target_manager(config, service_factory, role)
            else:
                # Preserve the established construction thread and lifecycle
                # for integrations outside the bounded serial-PIC profile.
                self._manager = LMCacheManager(
                    config,
                    service_factory,
                    connector=self,
                )

            # Start services managed by LMCacheManager, then verify the exact
            # role-local graph before any target request can be admitted.
            self._require_target_manager_ready(post_init=False)
            self._manager.start_services()
            self._require_target_manager_ready(post_init=False)

            # Initialize connector-specific state
            self._init_connector_state(role, vllm_config, config)

            # Setup metrics for monitoring data structures
            self._setup_metrics()

            logger.info(
                "LMCache initialized for role %s with version %s, "
                "vllm version %s, lmcache cache_engine metadata: %s",
                role,
                utils.get_version(),
                VLLM_VERSION,
                getattr(self.lmcache_engine, "metadata", None),
            )
            # Keep the exact Manager handoff strong and hidden from public
            # retry through every readiness, service, state, and metrics step.
            # This is the final successful adapter-construction operation.
            self._release_target_manager_handoff()
            if construction_result_publisher is not None:
                construction_result_publisher(self)
        except BaseException as init_error:
            self._cleanup_target_manager_after_failure(
                "adapter initialization",
                init_error,
            )
            raise

    def _acquire_target_manager(
        self,
        config: LMCacheEngineConfig,
        service_factory: Any,
        role: KVConnectorRole,
    ) -> None:
        """Construct and publish the target Manager through a retained owner."""
        owner = _RetainedManagerConstructionOwner(
            config,
            service_factory,
            self,
            role,
        )
        # Publish the recovery owner before the Manager call can begin.
        self._manager_construction_owner: Optional[
            _RetainedManagerConstructionOwner
        ] = owner
        owner.run()
        manager = owner.manager
        self._manager = manager

    def _release_target_manager_handoff(self) -> None:
        """Release the exact Manager receipt after full adapter construction."""
        if self._target_profile not in KMESH_PIC_V1_PROFILES:
            return
        owner = getattr(self, "_manager_construction_owner", None)
        if owner is None:
            return
        owner.release_handoff()
        self._manager_construction_owner = None

    def _apply_extra_config(
        self, config: LMCacheEngineConfig, vllm_config: "VllmConfig"
    ) -> None:
        """Apply extra config from vLLM to LMCache config."""
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            "Updated config %s from vLLM extra config",
                            config_key,
                        )

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        """Initialize connector-specific state variables."""
        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            Generator[Optional[torch.Tensor], None, None]
        ] = []
        self._layerwise_save_storers: dict[
            str, Generator[Optional[torch.Tensor], None, None]
        ] = {}
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self._kmesh_prepared_lookups: dict[str, _KMeshPreparedLookup] = {}
        self._kmesh_prepared_request_objects: dict[str, object] = {}
        self._kmesh_allocated_requests: set[str] = set()
        self._kmesh_active_worker_step: Optional[_KMeshWorkerStepContract] = None
        self._kmesh_pending_worker_step: Optional[_KMeshWorkerStepContract] = None
        self._kmesh_last_worker_step_generation = 0
        self._kmesh_metadata_transaction_active = False
        self._kmesh_metadata_transaction_committed = False
        self._kmesh_metadata_transaction_finalized = False
        self._kmesh_metadata_transaction_checkpoint: Optional[
            tuple[
                Optional[_KMeshWorkerStepContract],
                Optional[_KMeshWorkerStepContract],
                int,
                int,
                int,
            ]
        ] = None
        self._kmesh_store_sessions: dict[str, _KMeshLayerwiseStoreSession] = {}
        self._kmesh_store_sessions_lock = threading.RLock()
        self._kmesh_store_sessions_closed = False
        self._kmesh_failed_worker_requests: set[str] = set()
        self._kmesh_outstanding_lookup_ids: set[str] = set()
        self._kmesh_registered_lookup_cleanups: set[str] = set()
        self._kmesh_completed_save_generation = 0
        self._kmesh_completed_cleanup_generation = 0
        self._kmesh_next_wait_layer = 0
        self._kmesh_next_save_layer = 0
        self._kmesh_store_cleanup_errors: list[BaseException] = []

        # Role-specific initialization
        if role == KVConnectorRole.SCHEDULER:
            self._unfinished_requests: dict[str, "Request"] = {}
            # A vLLM request can be probed repeatedly or preempted. Never use
            # its public request ID as worker pin ownership: each scheduling
            # attempt gets a fresh UUID until allocation transfers it.
            self._prealloc_lookup_ids: dict[str, str] = {}
            self._blocked_prealloc_requests: set[str] = set()
            self._allocated_lookup_ids: dict[str, str] = {}
            self._worker_owned_lookup_requests: set[str] = set()
        else:
            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending
            self.blender: Optional["LMCBlender"] = None

            if self.enable_blending:
                assert self.lmcache_engine is not None
                assert self.lmcache_engine.gpu_connector is not None, (
                    "GPU connector must be available for blending"
                )
                logger.info(
                    "CacheBlend is waiting for the loaded vLLM model to be registered"
                )

        # Legacy compatibility check
        self._check_legacy_register_kv_caches()

        self.kv_caches: dict[str, torch.Tensor] = {}
        self._block_size = vllm_config.cache_config.block_size
        self.load_specs: dict[str, LoadSpec] = {}

        self.kv_cache_manager: Optional["KVCacheManager"] = None
        self._request_trackers: dict[str, RequestTracker] = {}

        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
        )

        self._lmcache_chunk_size = config.chunk_size

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

        self.force_skip_save = (
            False
            if self._target_profile in KMESH_PIC_V1_PROFILES
            else bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))
        )
        self._requests_priority: dict[str, int] = {}

        # Chunked KV loading: cap the number of external tokens
        # reported per scheduling step to prevent GPU block pool
        # exhaustion at high concurrency with long contexts.
        # Configurable via kv_connector_extra_config:
        #   "lmcache.max_tokens_per_load": <int>
        # Default 0 (no cap — original behavior, all matched tokens
        # reported at once). Set to a positive value (e.g. 131072)
        # to enable chunked loading.
        self._max_tokens_per_load: int = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.max_tokens_per_load", 0
            )
        )
        self._invalid_block_ids: set[int] = set()

    def _ensure_kmesh_store_session_lifecycle(self) -> None:
        """Initialize lifecycle state for lightweight recovery/test objects."""
        if not hasattr(self, "_kmesh_store_sessions_lock"):
            self._kmesh_store_sessions_lock = threading.RLock()
        if not hasattr(self, "_kmesh_store_sessions_closed"):
            self._kmesh_store_sessions_closed = False

    def _require_kmesh_store_session_admission_open(self) -> None:
        """Reject save admission after terminal connector shutdown begins."""
        self._ensure_kmesh_store_session_lifecycle()
        with self._kmesh_store_sessions_lock:
            if self._kmesh_store_sessions_closed:
                raise RuntimeError(
                    "KMesh connector shutdown has begun; refusing store session"
                )

    def _check_legacy_register_kv_caches(self) -> None:
        """Check for legacy connector without register_kv_caches implementation."""
        if self.lmcache_engine is None:
            return

        child_class = self._parent.__class__
        parent_class = KVConnectorBase_V1
        child_method = getattr(child_class, "register_kv_caches", None)
        parent_method = getattr(parent_class, "register_kv_caches", None)

        if child_method is None or parent_method is None:
            implements = False
        else:
            implements = child_method is not parent_method

        if not implements:
            logger.warning(
                "Please use the latest lmcache connector, otherwise some "
                "features may not work, such as DSA"
            )
            self._post_init_manager()

    def _require_target_manager_ready(self, *, post_init: bool) -> None:
        """Reject degraded or incomplete target-profile service graphs."""
        if self._target_profile not in KMESH_PIC_V1_PROFILES:
            return
        profile = self._target_profile

        self._manager.raise_if_init_failed()
        if not self._manager.is_healthy():
            raise RuntimeError(f"LMCache {profile} manager is unhealthy")
        if self.lmcache_engine_metadata is None:
            raise RuntimeError(f"LMCache {profile} metadata is unavailable")
        if (
            self._manager.offload_server is not None
            or self._manager.api_server is not None
            or self._manager.runtime_plugin_launcher is not None
        ):
            raise RuntimeError(f"LMCache {profile} constructed an excluded service")

        if self._role == KVConnectorRole.SCHEDULER:
            if self.lookup_client is None:
                raise RuntimeError(
                    f"LMCache {profile} scheduler lookup client is unavailable"
                )
            if self.lmcache_engine is not None or self.lookup_server is not None:
                raise RuntimeError(
                    f"LMCache {profile} scheduler service graph is not exact"
                )
        elif self._role == KVConnectorRole.WORKER:
            if self.lmcache_engine is None or self.lookup_server is None:
                raise RuntimeError(
                    f"LMCache {profile} worker engine or lookup server is unavailable"
                )
            if self.lookup_client is not None:
                raise RuntimeError(
                    f"LMCache {profile} worker service graph is not exact"
                )
        else:
            raise RuntimeError(
                f"LMCache {profile} received unexpected role {self._role!r}"
            )

        if post_init and self._manager.health_monitor is None:
            raise RuntimeError(
                f"LMCache {profile} health monitor is unavailable after post-init"
            )

    def _cleanup_target_manager_after_failure(
        self,
        phase: str,
        init_error: BaseException,
    ) -> None:
        """Quiesce partial target services while preserving both failures."""
        if self._target_profile not in KMESH_PIC_V1_PROFILES:
            return

        manager = getattr(self, "_manager", None)
        if manager is None:
            owner = getattr(self, "_manager_construction_owner", None)
            if owner is None:
                return
            try:
                owner.run()
                manager = owner.manager
            except BaseException as recovery_error:
                if owner.manager_returned:
                    # Manager construction succeeded. A later provisional
                    # retain failed, but the exact result was prepublished to
                    # the owner and can be retained/stopped below.
                    manager = owner.manager
                elif owner.last_error is init_error:
                    # The Manager constructor itself raised the error already
                    # being propagated. It owns rollback of its partial state.
                    return
                else:
                    raise init_error from recovery_error

        try:
            owner = getattr(self, "_manager_construction_owner", None)
            if owner is not None:
                # Publish failed ownership before terminal cleanup can itself
                # be interrupted. Public retry never sees healthy provisional
                # handoffs, but it sees this exact promoted receipt.
                owner.promote_failed_handoff()
            else:
                # Failures after a completed construction handoff (for example
                # post-init or later public registration) use the established
                # failure-only registry.
                manager.retain_construction_cleanup()
            manager.stop_services()
            if not manager.services_stopped:
                cleanup_errors = manager.terminal_cleanup_errors
                if cleanup_errors:
                    raise cleanup_errors[0]
                raise RuntimeError("managed services did not fully quiesce")
        except BaseException as cleanup_error:
            if cleanup_error is init_error:
                raise
            raise init_error from cleanup_error

    def _post_init_manager(self) -> None:
        try:
            self._manager.post_init()
            self._require_target_manager_ready(post_init=True)
        except BaseException as init_error:
            self._cleanup_target_manager_after_failure("post-init", init_error)
            raise

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance from manager."""
        return self._manager.lmcache_engine

    @property
    def lmcache_engine_metadata(self):
        """Get the LMCache engine metadata from manager."""
        return self._manager.lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional["LookupClientInterface"]:
        """Get the lookup client from manager."""
        return self._manager.lookup_client

    @property
    def lookup_server(self):
        """Get the lookup server from manager."""
        return self._manager.lookup_server

    def _setup_metrics(self) -> None:
        """Setup metrics for monitoring data structures in the connector."""
        metadata = self._manager.lmcache_engine_metadata
        if metadata is None:
            logger.warning(
                "LMCache metadata is not initialized, "
                "connector metrics will not be collected"
            )
            return
        prometheus_logger = PrometheusLogger.GetOrCreate(
            metadata,
            config=self.config,
        )

        # Set up metrics for scheduler-specific and general data structures
        metrics_map = {
            "_unfinished_requests": "scheduler_unfinished_requests_count",
            "load_specs": "connector_load_specs_count",
            "_request_trackers": "connector_request_trackers_count",
            "kv_caches": "connector_kv_caches_count",
            "layerwise_retrievers": "connector_layerwise_retrievers_count",
            "_invalid_block_ids": "connector_invalid_block_ids_count",
            "_requests_priority": "connector_requests_priority_count",
        }

        for attr_name, metric_name in metrics_map.items():
            if hasattr(self, attr_name):
                metric = getattr(prometheus_logger, metric_name)
                # Use a default argument in the lambda to capture
                # the current value of `attr_name`
                # to avoid issues with late binding in closures.
                metric.set_function(lambda name=attr_name: len(getattr(self, name)))

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    # TODO(chunxiaozheng): in the latest lmcache_connector, we use `register_kv_caches`
    #  to init self.kv_caches, we keep it in order to be compatible with old versions
    #  and will be removed in the future.
    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

    ####################
    # Worker side APIs
    ####################
    @staticmethod
    def _worker_lookup_id(request: ReqMeta) -> str:
        return getattr(request, "lookup_id", None) or request.req_id

    def _register_worker_lookup_cleanups(
        self,
        metadata: LMCacheConnectorMetadata,
    ) -> None:
        engine = self.lmcache_engine
        if engine is None:
            return
        register = getattr(engine, "register_lookup_cleanup", None)
        if register is None:
            return
        for request in metadata.requests:
            register(self._worker_lookup_id(request))

    def _schedule_worker_lookup_cleanup(
        self,
        request: ReqMeta,
    ) -> Optional[BaseException]:
        engine = self.lmcache_engine
        if engine is None:
            return None
        lookup_id = self._worker_lookup_id(request)
        try:
            schedule = getattr(engine, "schedule_lookup_unpin", None)
            if schedule is not None:
                schedule(lookup_id)
            else:
                engine.lookup_unpin(lookup_id)
        except BaseException as error:
            logger.error(
                "Worker lookup cleanup for %s failed: %s",
                lookup_id,
                error,
            )
            return error
        return None

    def _schedule_worker_lookup_cleanups(
        self,
        metadata: LMCacheConnectorMetadata,
    ) -> Optional[BaseException]:
        first_error: Optional[BaseException] = None
        for request in metadata.requests:
            error = self._schedule_worker_lookup_cleanup(request)
            if first_error is None and error is not None:
                first_error = error
        return first_error

    def _register_kmesh_lookup_cleanups(
        self,
        contracts: tuple[_KMeshWorkerRequestContract, ...],
    ) -> None:
        """Register immutable worker UUID ownership before fallible data work."""
        if not contracts:
            return
        active = self._kmesh_active_worker_step or self._kmesh_pending_worker_step
        if (
            active is not None
            and active.generation == self._kmesh_completed_cleanup_generation
        ):
            raise RuntimeError("KMesh worker step already scheduled terminal cleanup")
        lookup_ids = tuple(contract.lookup_id for contract in contracts)
        # Record ownership before the first fallible engine call. Clear/abort/
        # shutdown can then drain the UUID even if registration itself fails.
        self._kmesh_outstanding_lookup_ids.update(lookup_ids)
        engine = self.lmcache_engine
        if engine is None:
            raise RuntimeError("KMesh target requires a live LMCache engine")
        register = getattr(engine, "register_lookup_cleanup", None)
        if not callable(register):
            raise RuntimeError("KMesh target engine lacks lookup cleanup ownership")
        for lookup_id in lookup_ids:
            if lookup_id in self._kmesh_registered_lookup_cleanups:
                continue
            register(lookup_id)
            self._kmesh_registered_lookup_cleanups.add(lookup_id)

    def _schedule_kmesh_lookup_cleanups(
        self,
    ) -> Optional[BaseException]:
        """Schedule exact immutable worker UUID cleanup at most once."""
        if not self._kmesh_outstanding_lookup_ids:
            return None
        engine = self.lmcache_engine
        if engine is None:
            return RuntimeError("KMesh target lost its LMCache cleanup engine")
        first_error: Optional[BaseException] = None
        for lookup_id in tuple(self._kmesh_outstanding_lookup_ids):
            try:
                schedule = getattr(engine, "schedule_lookup_unpin", None)
                if callable(schedule):
                    schedule(lookup_id)
                else:
                    engine.lookup_unpin(lookup_id)
            except BaseException as error:
                logger.error(
                    "KMesh worker lookup cleanup for %s failed: %s",
                    lookup_id,
                    error,
                )
                if first_error is None:
                    first_error = error
            else:
                self._kmesh_outstanding_lookup_ids.discard(lookup_id)
                self._kmesh_registered_lookup_cleanups.discard(lookup_id)
        if not self._kmesh_outstanding_lookup_ids:
            active = self._kmesh_active_worker_step or self._kmesh_pending_worker_step
            if active is not None:
                self._kmesh_completed_cleanup_generation = max(
                    self._kmesh_completed_cleanup_generation,
                    active.generation,
                )
        return first_error

    def _is_kmesh_pic_target(self) -> bool:
        """Return whether this adapter uses either exact online PIC profile."""
        return getattr(self, "_target_profile", None) in KMESH_PIC_V1_PROFILES

    def _is_kmesh_collective_lockstep_target(self) -> bool:
        """Return whether transported plans select worker group execution."""
        return getattr(self, "_target_profile", None) in KMESH_COLLECTIVE_PROFILES

    def _is_kmesh_batched_position_recovery_target(self) -> bool:
        """Return whether group execution batches cached-K rotation."""
        return getattr(self, "_target_profile", None) in (
            COLLECTIVE_BATCHED_POSITION_RECOVERY_V1,
            COLLECTIVE_BATCHED_QKV_V1,
        )

    def _is_kmesh_batched_qkv_target(self) -> bool:
        """Return whether group execution also batches model-side QKV work."""
        return getattr(self, "_target_profile", None) == COLLECTIVE_BATCHED_QKV_V1

    @staticmethod
    def _validate_kmesh_request_id_value(
        request_id: object,
        field_name: str,
    ) -> str:
        """Validate one bounded plain request ID without equality aliases."""
        if (
            type(request_id) is not str
            or not request_id
            or request_id != request_id.strip()
            or len(request_id) > 256
        ):
            raise ValueError(f"{field_name} must be a bounded plain string")
        return request_id

    @classmethod
    def _kmesh_request_id(cls, request: object) -> str:
        """Read one bounded plain request ID without accepting equality aliases."""
        return cls._validate_kmesh_request_id_value(
            getattr(request, "request_id", None),
            "KMesh request_id",
        )

    @classmethod
    def _validate_kmesh_lookup_id(
        cls,
        lookup_id: object,
        field_name: str,
    ) -> str:
        """Validate one canonical UUIDv4 lookup-attempt owner."""
        value = cls._validate_kmesh_request_id_value(lookup_id, field_name)
        try:
            parsed = UUID(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be a canonical UUIDv4") from error
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError(f"{field_name} must be a canonical UUIDv4")
        return value

    def _require_kmesh_worker_step(
        self,
    ) -> _KMeshWorkerStepContract:
        """Return the validated active worker handoff or fail closed."""
        if self._role != KVConnectorRole.WORKER:
            raise RuntimeError("KMesh worker metadata is only valid on the worker role")
        active = self._kmesh_active_worker_step
        if active is None:
            raise RuntimeError("KMesh peer metadata was not bound for this worker step")
        return active

    def _validate_kmesh_worker_layers(
        self,
        step: _KMeshWorkerStepContract,
    ) -> tuple[torch.Tensor, ...]:
        """Return the frozen layer tensors after revalidating their identity."""
        layer_names = tuple(self.kv_caches)
        kv_caches = tuple(self.kv_caches.values())
        if (
            len(layer_names) != self.num_layers
            or layer_names != step.layer_names
            or tuple(id(item) for item in kv_caches) != step.kv_cache_ids
        ):
            raise RuntimeError("KMesh registered KV-cache identity changed after bind")
        return kv_caches

    def _kmesh_worker_requests(
        self,
        metadata: LMCacheConnectorMetadata,
    ) -> tuple[
        int,
        tuple[tuple[ReqMeta, _KMeshWorkerRequestContract], ...],
    ]:
        """Revalidate current request metadata against the bound worker step."""
        step = self._require_kmesh_worker_step()
        generation = step.generation
        active_contracts = step.requests
        if type(metadata.requests) is not list:
            raise TypeError("LMCache request metadata must be a plain list")
        active = {contract.request_id: contract for contract in active_contracts}
        if len(active) != len(active_contracts):
            raise RuntimeError("KMesh worker step contains duplicate request IDs")

        pairs: list[tuple[ReqMeta, _KMeshWorkerRequestContract]] = []
        seen: set[str] = set()
        for index, request in enumerate(metadata.requests):
            if type(request) is not ReqMeta:
                raise TypeError(f"LMCache requests[{index}] must be exactly ReqMeta")
            execution = request.kmesh_execution
            if type(execution) is not KMeshRequestExecutionEvidence:
                raise TypeError("KMesh worker request lost exact execution evidence")
            request_id = self._validate_kmesh_request_id_value(
                request.req_id,
                f"KMesh worker requests[{index}].req_id",
            )
            contract = active.get(request_id)
            if contract is None:
                raise ValueError("KMesh worker request is absent from the bound step")
            if request_id != contract.request_id:
                raise ValueError("KMesh worker request ID changed after bind")
            self._validate_same_kmesh_execution(contract.execution, execution)
            if request_id in seen:
                raise ValueError("KMesh worker requests contain a duplicate ID")
            seen.add(request_id)
            if (
                type(request.token_ids) is not list
                or any(type(token_id) is not int for token_id in request.token_ids)
                or tuple(request.token_ids) != contract.token_ids
            ):
                raise ValueError("KMesh worker request lost its full prompt")
            if (
                type(request.slot_mapping) is not torch.Tensor
                or request.slot_mapping.dtype != torch.long
                or request.slot_mapping.ndim != 1
                or len(request.slot_mapping) != len(request.token_ids)
                or bool(torch.any(request.slot_mapping < 0))
                or len(torch.unique(request.slot_mapping)) != len(request.slot_mapping)
            ):
                raise ValueError("KMesh worker request lost its full slot mapping")
            if tuple(int(slot) for slot in request.slot_mapping.tolist()) != (
                contract.slot_mapping
            ):
                raise ValueError("KMesh worker request slot mapping changed after bind")
            if type(request.lookup_id) is not str or request.lookup_id != (
                contract.lookup_id
            ):
                raise ValueError("KMesh worker request lookup owner changed after bind")
            load_spec = request.load_spec
            if type(load_spec) is not LoadSpec:
                raise TypeError("KMesh worker request lost its load state")
            if (
                type(load_spec.vllm_cached_tokens) is not int
                or type(load_spec.lmcache_cached_tokens) is not int
                or type(load_spec.can_load) is not bool
                or (
                    load_spec.vllm_cached_tokens,
                    load_spec.lmcache_cached_tokens,
                    load_spec.can_load,
                )
                != contract.load_spec
            ):
                raise ValueError("KMesh worker request load state changed after bind")
            if (
                type(request.save_spec) is not SaveSpec
                or type(request.save_spec.skip_leading_tokens) is not int
                or type(request.save_spec.can_save) is not bool
                or (
                    request.save_spec.skip_leading_tokens,
                    request.save_spec.can_save,
                )
                != contract.save_spec
            ):
                raise ValueError("KMesh worker request lost its save state")
            if request.disagg_spec is not None:
                raise ValueError("KMesh worker request gained disaggregation state")
            if request.is_last_prefill is not contract.is_last_prefill:
                raise ValueError(
                    "KMesh worker request prefill state changed after bind"
                )
            self._validate_kmesh_request_config_match(
                contract.execution.as_request_configs(),
                request.request_configs,
                "KMesh worker request configs",
            )
            pairs.append((request, contract))
        if seen != set(active):
            raise ValueError("KMesh bound worker request set changed after binding")
        return generation, tuple(pairs)

    def _record_kmesh_load_failure(
        self,
        contract: _KMeshWorkerRequestContract,
        reason: str,
        *,
        invalidate_full_prompt: bool = False,
    ) -> None:
        """Invalidate every block covered by one advertised external hit."""
        if not contract.load_spec[2]:
            raise RuntimeError("KMesh load failure has no advertised external hit")
        invalid_token_count = (
            len(contract.slot_mapping)
            if invalidate_full_prompt
            else contract.advertised_tokens
        )
        if invalid_token_count <= 0 or invalid_token_count > len(contract.slot_mapping):
            raise RuntimeError("KMesh load failure has an invalid hit boundary")
        slots = torch.tensor(
            contract.slot_mapping[:invalid_token_count],
            device="cpu",
            dtype=torch.long,
        )
        invalid = {int(item) for item in torch.unique(slots // self._block_size)}
        if not invalid:
            raise RuntimeError("KMesh load failure covers no destination block")
        self._invalid_block_ids.update(invalid)
        self._kmesh_failed_worker_requests.add(contract.request_id)
        logger.warning(
            "KMesh request %s invalidated %d externally advertised blocks: %s",
            contract.request_id,
            len(invalid),
            reason,
        )

    @staticmethod
    def _validate_kmesh_retrieval_mask(
        contract: _KMeshWorkerRequestContract,
        ret_mask: object,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> torch.Tensor:
        """Require a union of whole declared spans and no undeclared tokens."""
        if (
            type(ret_mask) is not torch.Tensor
            or ret_mask.dtype != torch.bool
            or ret_mask.ndim != 1
            or len(ret_mask) != len(contract.token_ids)
        ):
            raise ValueError("KMesh CacheBlend returned an invalid token mask")
        mask = ret_mask.to(device="cpu", dtype=torch.bool)
        spans = (
            contract.execution.explicit_spans
            if explicit_spans is None
            else explicit_spans
        )
        if type(spans) is not ExplicitTokenSpans:
            raise TypeError("KMesh recovery spans have an invalid exact type")
        spans.validate_token_count(len(contract.token_ids))
        declared = torch.zeros(len(contract.token_ids), dtype=torch.bool)
        hit_end = contract.load_spec[1]
        for span in spans.spans:
            span_mask = mask[span.start : span.end]
            if span.end > hit_end:
                if bool(torch.any(span_mask).item()):
                    raise ValueError(
                        "KMesh CacheBlend returned a span beyond the lookup hit"
                    )
                continue
            if bool(torch.any(span_mask).item()) and not bool(
                torch.all(span_mask).item()
            ):
                raise ValueError("KMesh CacheBlend returned a partial declared span")
            declared[span.start : span.end] = True
        if bool(torch.any(mask & ~declared).item()):
            raise ValueError("KMesh CacheBlend returned tokens outside declared spans")
        return mask

    def _fresh_kmesh_runtime_evidence(self) -> KMeshRuntimeEvidence:
        """Rebuild runtime evidence and reject post-construction config drift."""
        initial = self._kmesh_runtime_evidence
        if initial is None:
            raise RuntimeError("KMesh runtime evidence is unavailable")
        current = build_kmesh_runtime_evidence(
            self._vllm_config,
            self.config,
            role=self._role,
            target_profile=self._target_profile,
        )
        if current.to_canonical_payload() != initial.to_canonical_payload():
            raise ValueError("KMesh runtime configuration drifted after construction")
        return current

    @staticmethod
    def _validate_same_kmesh_execution(
        expected: KMeshRequestExecutionEvidence,
        actual: KMeshRequestExecutionEvidence,
    ) -> None:
        """Compare executions without accepting bool/int equality aliases."""
        if actual != expected:
            raise ValueError("KMesh execution evidence changed between lookup probes")
        if len(actual.request_configs) != len(expected.request_configs):
            raise ValueError("KMesh request configs changed between lookup probes")
        for actual_item, expected_item in zip(
            actual.request_configs,
            expected.request_configs,
            strict=True,
        ):
            actual_key, actual_value = actual_item
            expected_key, expected_value = expected_item
            if (
                actual_key != expected_key
                or type(actual_value) is not type(expected_value)
                or actual_value != expected_value
            ):
                raise ValueError("KMesh request configs changed between lookup probes")

    def _rebuild_kmesh_execution(
        self,
        prepared: _KMeshPreparedLookup,
        request: object,
    ) -> Optional[KMeshRequestExecutionEvidence]:
        """Rebuild current scheduler evidence and compare it with admission."""
        prepared.validate_repeated_prepare(request, prepared.descriptor)
        if not prepared.descriptor.permits_serial_pic:
            if prepared.execution is not None:
                raise ValueError("fallback KMesh lookup retained execution evidence")
            return None
        if type(prepared.execution) is not KMeshRequestExecutionEvidence:
            raise TypeError("actionable KMesh lookup lost execution evidence")
        runtime_evidence = self._fresh_kmesh_runtime_evidence()
        sampling_params = getattr(request, "sampling_params", None)
        binding = bind_kmesh_request_lookup(
            prepared.descriptor,
            request,
            admission_generation=prepared.descriptor.admission_generation,
            cache_domain=prepared.descriptor.cache_domain,
            runtime_evidence=runtime_evidence,
            request_configs=extract_request_configs(sampling_params),
        )
        current = KMeshRequestExecutionEvidence.from_binding(binding)
        self._validate_same_kmesh_execution(prepared.execution, current)
        return current

    def _validate_kmesh_worker_destination_ownership(
        self,
        contracts: tuple[_KMeshWorkerRequestContract, ...],
    ) -> None:
        """Require request-owned slots and blocks across the whole worker step."""
        if type(self._block_size) is not int or self._block_size <= 0:
            raise RuntimeError("KMesh worker block size is invalid")
        slot_owner: dict[int, str] = {}
        block_owner: dict[int, str] = {}
        for contract in contracts:
            for slot in contract.slot_mapping:
                prior = slot_owner.setdefault(slot, contract.request_id)
                if prior != contract.request_id:
                    raise ValueError("KMesh worker requests share a destination slot")
            for block_id in {
                slot // self._block_size for slot in contract.slot_mapping
            }:
                prior = block_owner.setdefault(block_id, contract.request_id)
                if prior != contract.request_id:
                    raise ValueError("KMesh worker requests share a destination block")

    def _freeze_kmesh_worker_plans(
        self,
        plan_batch: object,
        contracts: tuple[_KMeshWorkerRequestContract, ...],
    ) -> tuple[_KMeshWorkerReusePlan, ...]:
        """Project transported public plans into strict LMCache-local state."""
        # Third Party
        from vllm.v1.kmesh import (
            KMeshCompatibilityKey,
            KMeshPlanBatch,
            KMeshRequestSegmentMap,
            KMeshReuseGroup,
            KMeshReusePlan,
            KMeshSegmentMapEntry,
            KMeshSharedSegmentIdentity,
        )

        if type(plan_batch) is not KMeshPlanBatch:
            raise TypeError("KMesh collective target requires exact plan-batch data")
        if type(plan_batch.plans) is not tuple:
            raise TypeError("KMesh plan batch must contain an immutable plan tuple")
        by_request = {contract.request_id: contract for contract in contracts}
        frozen: list[_KMeshWorkerReusePlan] = []
        planned_requests: set[str] = set()
        plan_ids: list[str] = []
        reuse_group_ids: set[str] = set()
        for plan_index, plan in enumerate(plan_batch.plans):
            if type(plan) is not KMeshReusePlan:
                raise TypeError(f"KMesh plans[{plan_index}] has an invalid exact type")
            if type(plan.group) is not KMeshReuseGroup:
                raise TypeError(f"KMesh plans[{plan_index}].group has an invalid type")
            if type(plan.plan_id) is not str or not plan.plan_id:
                raise TypeError("KMesh plan IDs must be nonempty exact strings")
            if plan.plan_id in plan_ids or (plan_ids and plan.plan_id < plan_ids[-1]):
                raise ValueError("KMesh worker plan IDs must be unique and sorted")
            plan_ids.append(plan.plan_id)
            if type(plan.request_segment_maps) is not tuple:
                raise TypeError("KMesh request segment maps must be immutable")
            if type(plan.block_size) is not int or plan.block_size != self._block_size:
                raise ValueError("KMesh plan block size differs from the worker")
            group = plan.group
            if type(group.compatibility_key) is not KMeshCompatibilityKey:
                raise TypeError("KMesh reuse-group compatibility has an invalid type")
            KMeshCompatibilityKey.__post_init__(group.compatibility_key)
            if type(group.shared_segments) is not tuple:
                raise TypeError("KMesh shared segments must be immutable")
            for shared_segment in group.shared_segments:
                if type(shared_segment) is not KMeshSharedSegmentIdentity:
                    raise TypeError("KMesh shared segment has an invalid exact type")
                KMeshSharedSegmentIdentity.__post_init__(shared_segment)
            if type(group.reuse_group_id) is not str or not group.reuse_group_id:
                raise TypeError("KMesh reuse-group IDs must be nonempty exact strings")
            if group.reuse_group_id in reuse_group_ids:
                raise ValueError("KMesh worker plans repeat a reuse-group ID")
            reuse_group_ids.add(group.reuse_group_id)
            if type(group.request_ids) is not tuple:
                raise TypeError("KMesh reuse-group membership must be immutable")
            if group.request_ids != tuple(sorted(set(group.request_ids))):
                raise ValueError(
                    "KMesh reuse-group membership must be unique and sorted"
                )
            if len(group.request_ids) < 2:
                raise ValueError("KMesh collective plans require at least two members")
            if type(group.shared_segments) is not tuple or not group.shared_segments:
                raise ValueError("KMesh reuse groups require immutable shared segments")
            if len(plan.request_segment_maps) != len(group.request_ids):
                raise ValueError("KMesh collective plan membership is incomplete")
            KMeshReuseGroup.__post_init__(group)

            members: list[_KMeshWorkerPlanMember] = []
            for member_index, (request_id, mapping) in enumerate(
                zip(
                    group.request_ids,
                    plan.request_segment_maps,
                    strict=True,
                )
            ):
                if type(request_id) is not str:
                    raise TypeError("KMesh plan request IDs must be plain strings")
                if request_id in planned_requests:
                    raise ValueError("KMesh request entered more than one worker plan")
                if type(mapping) is not KMeshRequestSegmentMap:
                    raise TypeError(
                        f"KMesh plan member {member_index} has an invalid map type"
                    )
                if mapping.request_id != request_id:
                    raise ValueError("KMesh plan member order changed in transport")
                if type(mapping.entries) is not tuple or not mapping.entries:
                    raise ValueError(
                        "KMesh plan member has no immutable selected spans"
                    )
                if len(mapping.entries) != len(group.shared_segments):
                    raise ValueError("KMesh plan member selected the wrong segment set")
                contract = by_request.get(request_id)
                if contract is None:
                    raise ValueError("KMesh plan member has no LMCache worker contract")

                selected: list[TokenSpan] = []
                declared_ranges = {
                    (span.start, span.end)
                    for span in contract.execution.explicit_spans.spans
                }
                for entry_index, entry in enumerate(mapping.entries):
                    if type(entry) is not KMeshSegmentMapEntry:
                        raise TypeError(
                            f"KMesh plan entry {entry_index} has an invalid exact type"
                        )
                    KMeshSegmentMapEntry.__post_init__(entry)
                    if entry.shared_segment != group.shared_segments[entry_index]:
                        raise ValueError(
                            "KMesh plan member segment order drifted in transport"
                        )
                    token_range = (entry.token_start, entry.token_end)
                    if token_range not in declared_ranges:
                        raise ValueError(
                            "KMesh plan selected a span absent from worker evidence"
                        )
                    selected.append(TokenSpan(*token_range))

                KMeshRequestSegmentMap.__post_init__(mapping)

                # Plans are identity-major, while a request may place the
                # same shared identities in a different logical order. LMCache
                # explicit spans are request-position-major by contract.
                selected.sort(key=lambda span: (span.start, span.end))

                members.append(
                    _KMeshWorkerPlanMember(
                        request_id=request_id,
                        explicit_spans=ExplicitTokenSpans(
                            token_count=len(contract.token_ids),
                            cached_span=TokenSpan(0, len(contract.token_ids)),
                            spans=tuple(selected),
                        ),
                    )
                )
                planned_requests.add(request_id)
            KMeshReusePlan.__post_init__(plan)
            frozen.append(
                _KMeshWorkerReusePlan(
                    plan_id=plan.plan_id,
                    reuse_group_id=group.reuse_group_id,
                    members=tuple(members),
                )
            )
        return tuple(frozen)

    def _validate_kmesh_peer_connector_metadata(
        self,
        peer_connector_metadata: tuple[KVConnectorMetadata, ...],
    ) -> _KMeshWorkerStepContract:
        """Return one validated KMesh/LMCache handoff without changing state."""
        if self._role != KVConnectorRole.WORKER:
            raise RuntimeError("KMesh peer metadata may only bind on the worker role")
        if type(peer_connector_metadata) is not tuple:
            raise TypeError("KMesh peer metadata must be an immutable tuple")
        if len(peer_connector_metadata) != 2:
            raise ValueError(
                "KMesh target requires exactly two connector metadata items"
            )
        if self._kmesh_active_worker_step is not None:
            raise RuntimeError("KMesh worker metadata is already bound")
        if getattr(self, "_kmesh_pending_worker_step", None) is not None:
            raise RuntimeError("KMesh worker metadata is already pending")

        # Third Party
        from vllm.distributed.kv_transfer.kv_connector.v1.kmesh_connector import (
            KMeshConnectorMetadata,
        )
        from vllm.v1.kmesh import (
            KMeshRequestLookupDescriptor as VLLMKMeshRequestLookupDescriptor,
        )

        kmesh_metadata, lmcache_metadata = peer_connector_metadata
        if type(kmesh_metadata) is not KMeshConnectorMetadata:
            raise TypeError("KMesh metadata must be the first exact tuple item")
        if type(lmcache_metadata) is not LMCacheConnectorMetadata:
            raise TypeError("LMCache metadata must be the second exact tuple item")

        # Reconstruct the public DTO to rerun its invariants after unpickling or
        # adversarial post-construction mutation. Do not read connector-private
        # metadata from either peer.
        KMeshConnectorMetadata(
            kmesh_metadata.plan_batch,
            kmesh_metadata.scheduled_descriptors,
            kmesh_metadata.step_generation,
        )
        generation = kmesh_metadata.step_generation
        if type(generation) is not int or generation <= 0:
            raise ValueError("KMesh worker step_generation must be positive")
        if generation <= self._kmesh_last_worker_step_generation:
            raise ValueError("KMesh worker metadata is stale or was already consumed")

        parsed_descriptors: dict[str, KMeshRequestLookupDescriptor] = {}
        descriptor_ids: set[str] = set()
        for index, peer_descriptor in enumerate(kmesh_metadata.scheduled_descriptors):
            if type(peer_descriptor) is not VLLMKMeshRequestLookupDescriptor:
                raise TypeError(
                    f"KMesh scheduled_descriptors[{index}] has an invalid exact type"
                )
            to_payload = getattr(peer_descriptor, "to_sidecar_payload", None)
            if not callable(to_payload):
                raise TypeError(
                    f"KMesh scheduled_descriptors[{index}] has no public payload"
                )
            descriptor = parse_kmesh_request_sidecar(
                KMESH_REQUEST_SIDECAR_PROVIDER_ID,
                to_payload(),
            )
            if descriptor.request_id in descriptor_ids:
                raise ValueError("KMesh worker descriptors contain a duplicate ID")
            descriptor_ids.add(descriptor.request_id)
            if descriptor.permits_serial_pic:
                parsed_descriptors[descriptor.request_id] = descriptor

        if type(lmcache_metadata.requests) is not list:
            raise TypeError("LMCache request metadata must be a plain list")
        contracts: dict[str, _KMeshWorkerRequestContract] = {}
        lookup_ids: set[str] = set()
        runtime_evidence = self._fresh_kmesh_runtime_evidence()
        for index, request in enumerate(lmcache_metadata.requests):
            if type(request) is not ReqMeta:
                raise TypeError(f"LMCache requests[{index}] must be exactly ReqMeta")
            execution = request.kmesh_execution
            if type(execution) is not KMeshRequestExecutionEvidence:
                raise TypeError(
                    "KMesh target metadata requires exact execution evidence"
                )
            if (
                type(request.req_id) is not str
                or request.req_id != execution.request_id
            ):
                raise ValueError("LMCache request ID differs from KMesh evidence")
            if request.req_id in contracts:
                raise ValueError(
                    "LMCache KMesh evidence contains a duplicate request ID"
                )
            lookup_id = self._validate_kmesh_lookup_id(
                request.lookup_id,
                f"LMCache requests[{index}].lookup_id",
            )
            if lookup_id in lookup_ids:
                raise ValueError("LMCache KMesh metadata reuses a lookup ID")
            lookup_ids.add(lookup_id)
            load_spec = request.load_spec
            if type(load_spec) is not LoadSpec:
                raise TypeError("KMesh target metadata requires exact load state")
            if (
                type(load_spec.vllm_cached_tokens) is not int
                or load_spec.vllm_cached_tokens != 0
                or type(load_spec.lmcache_cached_tokens) is not int
                or type(load_spec.can_load) is not bool
            ):
                raise ValueError("KMesh target load state has invalid exact fields")
            valid_hit_ends = {
                0,
                *(span.end for span in execution.explicit_spans.spans),
            }
            if load_spec.lmcache_cached_tokens not in valid_hit_ends:
                raise ValueError("KMesh target load state has an invalid span end")
            advertised_tokens = load_spec.lmcache_cached_tokens - int(
                load_spec.lmcache_cached_tokens == len(execution.prompt_token_ids)
                and load_spec.lmcache_cached_tokens > 0
            )
            if load_spec.can_load is not (advertised_tokens > 0):
                raise ValueError("KMesh target load allocation state is inconsistent")
            if (
                type(request.save_spec) is not SaveSpec
                or request.save_spec.can_save is not True
                or type(request.save_spec.skip_leading_tokens) is not int
                or request.save_spec.skip_leading_tokens != 0
            ):
                raise ValueError("KMesh target metadata requires exact save state")
            if request.disagg_spec is not None:
                raise ValueError("non-actionable KMesh metadata cannot disaggregate")
            if request.is_last_prefill is not True:
                raise ValueError(
                    "KMesh evidence requires one full-prefill metadata step"
                )
            if type(request.token_ids) is not list or any(
                type(token_id) is not int for token_id in request.token_ids
            ):
                raise TypeError("KMesh request tokens must be a plain integer list")
            if tuple(request.token_ids) != execution.prompt_token_ids:
                raise ValueError("LMCache request tokens differ from KMesh evidence")
            if (
                type(request.slot_mapping) is not torch.Tensor
                or request.slot_mapping.dtype != torch.long
                or request.slot_mapping.ndim != 1
                or len(request.slot_mapping) != len(request.token_ids)
                or bool(torch.any(request.slot_mapping < 0))
                or len(torch.unique(request.slot_mapping)) != len(request.slot_mapping)
            ):
                raise ValueError(
                    "KMesh request slot mapping is not exact full-prompt evidence"
                )

            expected_configs = execution.as_request_configs()
            actual_configs = request.request_configs
            if type(actual_configs) is not dict or set(actual_configs) != set(
                expected_configs
            ):
                raise ValueError("LMCache request configs differ from KMesh evidence")
            for key, expected_value in expected_configs.items():
                actual_value = actual_configs[key]
                if (
                    type(actual_value) is not type(expected_value)
                    or actual_value != expected_value
                ):
                    raise ValueError(
                        "LMCache request configs differ from KMesh evidence"
                    )

            scheduled_descriptor = parsed_descriptors.get(request.req_id)
            if scheduled_descriptor is None:
                raise ValueError("LMCache evidence has no scheduled KMesh descriptor")
            execution.validate_peer_descriptor(scheduled_descriptor)
            execution.validate_runtime_evidence(runtime_evidence)
            frozen_descriptor = KMeshRequestLookupDescriptor.from_wire_payload(
                scheduled_descriptor.to_wire_payload()
            )
            frozen_runtime = KMeshRuntimeEvidence(
                model_id=runtime_evidence.model_id,
                model_revision=runtime_evidence.model_revision,
                tokenizer_id=runtime_evidence.tokenizer_id,
                tokenizer_revision=runtime_evidence.tokenizer_revision,
                model_dtype=runtime_evidence.model_dtype,
                effective_kv_dtype=runtime_evidence.effective_kv_dtype,
                kv_layout=runtime_evidence.kv_layout,
                attention_kind=runtime_evidence.attention_kind,
                rope_configuration=runtime_evidence.rope_configuration,
                attention_backend=runtime_evidence.attention_backend,
                attention_backend_configuration=(
                    runtime_evidence.attention_backend_configuration
                ),
                runtime_configuration=runtime_evidence.runtime_configuration,
                device_identity=runtime_evidence.device_identity,
                tensor_parallel_size=runtime_evidence.tensor_parallel_size,
                pipeline_parallel_size=runtime_evidence.pipeline_parallel_size,
            )
            frozen_execution = KMeshRequestExecutionEvidence(
                descriptor=frozen_descriptor,
                prompt_token_ids=tuple(execution.prompt_token_ids),
                descriptor_fingerprint=frozen_descriptor.fingerprint,
                explicit_spans=frozen_descriptor.explicit_token_spans(),
                runtime_evidence=frozen_runtime,
                cache_tags=tuple((key, value) for key, value in execution.cache_tags),
                request_configs=tuple(
                    (key, value) for key, value in execution.request_configs
                ),
            )
            contracts[request.req_id] = _KMeshWorkerRequestContract(
                execution=frozen_execution,
                lookup_id=lookup_id,
                token_ids=tuple(request.token_ids),
                slot_mapping=tuple(int(slot) for slot in request.slot_mapping.tolist()),
                load_spec=(
                    load_spec.vllm_cached_tokens,
                    load_spec.lmcache_cached_tokens,
                    load_spec.can_load,
                ),
                save_spec=(
                    request.save_spec.skip_leading_tokens,
                    request.save_spec.can_save,
                ),
                is_last_prefill=request.is_last_prefill,
            )

        if set(contracts) != set(parsed_descriptors):
            raise ValueError("KMesh and LMCache actionable request sets differ")

        ordered = tuple(contracts[key] for key in sorted(contracts))
        self._validate_kmesh_worker_destination_ownership(ordered)
        plans = (
            self._freeze_kmesh_worker_plans(kmesh_metadata.plan_batch, ordered)
            if self._is_kmesh_collective_lockstep_target()
            else ()
        )
        if len(self.kv_caches) != self.num_layers or not self.kv_caches:
            raise RuntimeError(
                "KMesh worker must register every KV-cache layer before binding"
            )
        layer_names = tuple(self.kv_caches)
        if any(
            type(layer_name) is not str or not layer_name for layer_name in layer_names
        ):
            raise TypeError("KMesh registered layer names must be plain strings")
        kv_cache_ids = tuple(id(kv_cache) for kv_cache in self.kv_caches.values())
        return _KMeshWorkerStepContract(
            generation=generation,
            requests=ordered,
            plans=plans,
            layer_names=layer_names,
            kv_cache_ids=kv_cache_ids,
        )

    def validate_peer_connector_metadata(
        self,
        peer_connector_metadata: tuple[KVConnectorMetadata, ...],
    ) -> None:
        """Validate the exact composite handoff without changing adapter state."""
        if not self._is_kmesh_pic_target():
            return
        self._validate_kmesh_peer_connector_metadata(peer_connector_metadata)

    def begin_connector_metadata_transaction(self) -> None:
        """Checkpoint worker state for one composite metadata transaction."""
        if not self._is_kmesh_pic_target():
            return
        if self._role != KVConnectorRole.WORKER:
            raise RuntimeError("KMesh metadata transactions belong to the worker")
        if getattr(self, "_kmesh_metadata_transaction_active", False):
            raise RuntimeError("KMesh metadata transaction is already active")
        if self._kmesh_outstanding_lookup_ids:
            raise RuntimeError("KMesh worker retained outstanding lookup ownership")
        if self._kmesh_active_worker_step is not None:
            raise RuntimeError("KMesh worker metadata is already bound")
        if getattr(self, "_kmesh_pending_worker_step", None) is not None:
            raise RuntimeError("KMesh worker metadata is already pending")
        prior_checkpoint = getattr(
            self,
            "_kmesh_metadata_transaction_checkpoint",
            None,
        )
        if prior_checkpoint is not None and not getattr(
            self,
            "_kmesh_metadata_transaction_finalized",
            False,
        ):
            raise RuntimeError("KMesh metadata transaction retained a checkpoint")
        self._kmesh_metadata_transaction_checkpoint = (
            self._kmesh_active_worker_step,
            self._kmesh_pending_worker_step,
            self._kmesh_last_worker_step_generation,
            self._kmesh_next_wait_layer,
            self._kmesh_next_save_layer,
        )
        self._kmesh_metadata_transaction_committed = False
        self._kmesh_metadata_transaction_finalized = False
        self._kmesh_metadata_transaction_active = True

    def bind_peer_connector_metadata(
        self,
        peer_connector_metadata: tuple[KVConnectorMetadata, ...],
    ) -> None:
        """Bind one validated handoff without executing LMCache data I/O."""
        if not self._is_kmesh_pic_target():
            return
        worker_step = self._validate_kmesh_peer_connector_metadata(
            peer_connector_metadata
        )
        if worker_step.generation <= self._kmesh_completed_cleanup_generation:
            raise ValueError("KMesh worker metadata cleanup was already scheduled")
        lookup_ids = {contract.lookup_id for contract in worker_step.requests}
        reused = lookup_ids & self._kmesh_outstanding_lookup_ids
        if reused:
            raise ValueError("KMesh worker metadata reuses a consumed lookup UUID")
        if getattr(self, "_kmesh_metadata_transaction_active", False):
            if getattr(self, "_kmesh_metadata_transaction_committed", False):
                raise RuntimeError("KMesh metadata transaction already committed")
            self._kmesh_pending_worker_step = worker_step
            self._register_kmesh_lookup_cleanups(worker_step.requests)
            return
        self._kmesh_active_worker_step = worker_step
        self._kmesh_last_worker_step_generation = worker_step.generation
        self._kmesh_next_wait_layer = 0
        self._kmesh_next_save_layer = 0
        self._register_kmesh_lookup_cleanups(worker_step.requests)

    def commit_connector_metadata_transaction(self) -> None:
        """Tentatively publish a step while retaining exact rollback state."""
        if not getattr(self, "_kmesh_metadata_transaction_active", False):
            return
        if getattr(self, "_kmesh_metadata_transaction_committed", False):
            return
        if getattr(self, "_kmesh_metadata_transaction_checkpoint", None) is None:
            raise RuntimeError("KMesh metadata transaction has no checkpoint")
        pending = getattr(self, "_kmesh_pending_worker_step", None)
        if pending is None:
            raise RuntimeError("KMesh metadata transaction has no pending handoff")
        self._kmesh_active_worker_step = pending
        self._kmesh_last_worker_step_generation = pending.generation
        self._kmesh_pending_worker_step = None
        self._kmesh_next_wait_layer = 0
        self._kmesh_next_save_layer = 0
        self._kmesh_metadata_transaction_committed = True

    def finalize_connector_metadata_transaction(self) -> None:
        """Finalize a step while retaining rollback until the next begin."""
        if not getattr(self, "_kmesh_metadata_transaction_active", False):
            return
        if not getattr(self, "_kmesh_metadata_transaction_committed", False):
            raise RuntimeError("KMesh metadata transaction is not committed")
        if getattr(self, "_kmesh_metadata_transaction_checkpoint", None) is None:
            raise RuntimeError("KMesh metadata transaction has no checkpoint")
        self._kmesh_metadata_transaction_active = False
        self._kmesh_metadata_transaction_committed = False
        self._kmesh_metadata_transaction_finalized = True

    def abort_connector_metadata_transaction(self) -> None:
        """Restore exact pre-begin state, including after tentative commit."""
        self._ensure_kmesh_store_session_lifecycle()
        with self._kmesh_store_sessions_lock:
            self._abort_connector_metadata_transaction_locked()

    def _abort_connector_metadata_transaction_locked(self) -> None:
        """Abort while serializing store-session cleanup and state reset."""
        checkpoint = getattr(
            self,
            "_kmesh_metadata_transaction_checkpoint",
            None,
        )
        if checkpoint is None and not getattr(
            self, "_kmesh_metadata_transaction_active", False
        ):
            cleanup_error = self._schedule_kmesh_lookup_cleanups()
            close_error = self._close_kmesh_store_sessions()
            if cleanup_error is None:
                cleanup_error = close_error
            if cleanup_error is not None:
                raise cleanup_error
            self._kmesh_store_cleanup_errors.clear()
            self._kmesh_failed_worker_requests.clear()
            return
        if checkpoint is None:
            raise RuntimeError("KMesh metadata transaction has no checkpoint")
        cleanup_error = self._schedule_kmesh_lookup_cleanups()
        close_error = self._close_kmesh_store_sessions()
        if cleanup_error is None:
            cleanup_error = close_error
        if cleanup_error is not None:
            raise cleanup_error
        self._kmesh_store_cleanup_errors.clear()
        self._kmesh_failed_worker_requests.clear()
        (
            self._kmesh_active_worker_step,
            self._kmesh_pending_worker_step,
            self._kmesh_last_worker_step_generation,
            self._kmesh_next_wait_layer,
            self._kmesh_next_save_layer,
        ) = checkpoint
        self._kmesh_metadata_transaction_active = False
        self._kmesh_metadata_transaction_committed = False
        self._kmesh_metadata_transaction_finalized = False
        self._kmesh_metadata_transaction_checkpoint = None

    def clear_peer_connector_metadata(self) -> None:
        """Clear one validated worker step while retaining replay protection."""
        if not self._is_kmesh_pic_target():
            return
        if self._role != KVConnectorRole.WORKER:
            raise RuntimeError("KMesh peer metadata may only clear on the worker role")
        self._ensure_kmesh_store_session_lifecycle()
        with self._kmesh_store_sessions_lock:
            self._clear_peer_connector_metadata_locked()

    def _clear_peer_connector_metadata_locked(self) -> None:
        """Clear one worker step under the store-session lifecycle lock."""
        first_error = self._schedule_kmesh_lookup_cleanups()
        close_error = self._close_kmesh_store_sessions()
        if first_error is None:
            first_error = close_error
        if first_error is not None:
            raise first_error
        self._kmesh_store_cleanup_errors.clear()
        self._kmesh_failed_worker_requests.clear()
        self._kmesh_active_worker_step = None
        self._kmesh_pending_worker_step = None
        self._kmesh_next_wait_layer = 0
        self._kmesh_next_save_layer = 0

    def register_model(self, model: torch.nn.Module) -> None:
        """Register the already-loaded vLLM model for CacheBlend.

        vLLM constructs KV connectors after loading the model, but it does not
        currently expose the model through connector configuration.  A narrow
        worker hook can call this method immediately after connector
        construction.  Non-blending and scheduler connectors ignore the hook.

        Args:
            model: Loaded vLLM model owned by this worker.

        Raises:
            RuntimeError: If the LMCache engine or blending GPU connector did
                not initialize.
        """
        if self._role != KVConnectorRole.WORKER or not self.enable_blending:
            return
        if self.lmcache_engine is None or self.lmcache_engine.gpu_connector is None:
            raise RuntimeError(
                "Cannot register a CacheBlend model without an LMCache engine "
                "and blending GPU connector"
            )

        VLLMModelTracker.register_model(ENGINE_NAME, model)
        self.blender = LMCBlenderBuilder.get_or_create(
            ENGINE_NAME,
            self.lmcache_engine,
            self.lmcache_engine.gpu_connector,
            self.config,
        )

    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        previous_kv_caches = self.kv_caches
        self.kv_caches = kv_caches
        try:
            self._post_init_manager()
        except BaseException:
            self.kv_caches = previous_kv_caches
            raise

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        if self._is_kmesh_pic_target():
            self._ensure_kmesh_store_session_lifecycle()
            with self._kmesh_store_sessions_lock:
                self._require_kmesh_store_session_admission_open()
                step = self._require_kmesh_worker_step()
                contracts = step.requests
                self._register_kmesh_lookup_cleanups(contracts)
                try:
                    metadata = self._parent._get_connector_metadata()
                    if type(metadata) is not LMCacheConnectorMetadata:
                        raise TypeError("KMesh worker requires exact LMCache metadata")
                    if forward_context.attn_metadata is None:
                        cleanup_error = self._schedule_kmesh_lookup_cleanups()
                        if cleanup_error is not None:
                            raise cleanup_error
                        return
                    self._start_kmesh_load_kv_locked(forward_context, metadata)
                except BaseException:
                    self._schedule_kmesh_lookup_cleanups()
                    raise
            return
        if self.lmcache_engine is None:
            return

        raw_metadata = getattr(self._parent, "_connector_metadata", None)
        cleanup_metadata = (
            raw_metadata if isinstance(raw_metadata, LMCacheConnectorMetadata) else None
        )
        try:
            if cleanup_metadata is not None:
                self._register_worker_lookup_cleanups(cleanup_metadata)
            metadata = self._parent._get_connector_metadata()
            assert isinstance(metadata, LMCacheConnectorMetadata)
            if metadata is not cleanup_metadata:
                self._register_worker_lookup_cleanups(metadata)
                cleanup_metadata = metadata
            # vLLM can build and transfer request metadata for a no-forward
            # batch. It will not call wait_for_save, so this hook is the
            # terminal worker owner for those UUIDs.
            if forward_context.attn_metadata is None:
                cleanup_error = self._schedule_worker_lookup_cleanups(metadata)
                if cleanup_error is not None:
                    raise cleanup_error
                return
            self._start_load_kv_impl(forward_context, **kwargs)
        except BaseException:
            if cleanup_metadata is not None:
                self._schedule_worker_lookup_cleanups(cleanup_metadata)
            raise

    def _start_kmesh_load_kv(
        self,
        forward_context: "ForwardContext",
        metadata: LMCacheConnectorMetadata,
    ) -> None:
        """Run synchronous full-prompt explicit CacheBlend for one worker step."""
        self._ensure_kmesh_store_session_lifecycle()
        with self._kmesh_store_sessions_lock:
            self._require_kmesh_store_session_admission_open()
            self._start_kmesh_load_kv_locked(forward_context, metadata)

    def _run_kmesh_serial_load(
        self,
        contract: _KMeshWorkerRequestContract,
        *,
        generation: int,
        kvcaches: list[torch.Tensor],
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> None:
        """Run one request-local recovery and contain ordinary failures."""
        spans = (
            contract.execution.explicit_spans
            if explicit_spans is None
            else explicit_spans
        )
        try:
            token_mask = torch.ones(len(contract.token_ids), dtype=torch.bool)
            slot_mapping = torch.tensor(
                contract.slot_mapping,
                dtype=torch.long,
                device=self.device,
            )
            assert self.blender is not None
            ret_mask = self.blender.blend(
                list(contract.token_ids),
                token_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                vllm_cached_tokens=0,
                request_configs=contract.execution.as_request_configs(),
                explicit_spans=spans,
                req_id=contract.lookup_id,
                request_id=contract.request_id,
                generation=generation,
            )
            ret_mask = self._validate_kmesh_retrieval_mask(
                contract,
                ret_mask,
                spans,
            )
            if not bool(torch.any(ret_mask).item()):
                self._record_kmesh_load_failure(
                    contract,
                    "explicit lookup/retrieve race produced an all-false mask",
                )
        except Exception as error:
            if contract.request_id not in self._kmesh_failed_worker_requests:
                self._record_kmesh_load_failure(
                    contract,
                    f"request-local CacheBlend failure: {error}",
                    invalidate_full_prompt=True,
                )
            logger.exception(
                "KMesh request %s CacheBlend failed; siblings continue",
                contract.request_id,
            )

    def _preflight_kmesh_collective_capacity(
        self,
        plan: _KMeshWorkerReusePlan,
        contracts: dict[str, _KMeshWorkerRequestContract],
        kvcaches: list[torch.Tensor],
    ) -> None:
        """Prove every group buffer fits before any member session advances."""
        token_counts = {
            len(contracts[member.request_id].token_ids) for member in plan.members
        }
        if len(token_counts) != 1:
            raise ValueError("KMesh collective plan members differ in token count")
        engine = self.lmcache_engine
        if engine is None:
            raise RuntimeError("KMesh collective target lost its LMCache engine")
        gpu_connector = engine.gpu_connector
        preflight = getattr(gpu_connector, "preflight_blend_group_capacity", None)
        if not callable(preflight):
            raise RuntimeError(
                "KMesh collective target lacks exact GPU capacity preflight"
            )
        result = preflight(
            kvcaches,
            num_tokens=next(iter(token_counts)),
            request_count=len(plan.members),
        )
        if type(result) is not bool:
            raise TypeError("KMesh collective capacity preflight returned non-bool")
        if not result:
            raise MemoryError("KMesh collective group GPU capacity is insufficient")

    def _validate_kmesh_collective_operation_counts(
        self,
        outcome: LMCBlendGroupOutcome,
        counts: object,
    ) -> None:
        """Validate exact lockstep-reference operation evidence."""
        if type(counts) is not LMCBlendOperationCounts:
            raise TypeError("KMesh collective result has invalid operation counts")
        values = (
            counts.serial_rope_operations,
            counts.serial_diff_operations,
            counts.collective_rope_operations,
            counts.collective_diff_operations,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("KMesh collective operation counts must be exact")
        qkv = self._is_kmesh_batched_qkv_target()
        if not qkv and (
            counts.collective_rope_operations or counts.collective_diff_operations
        ):
            raise ValueError(
                "only the batched-QKV profile batches model-side tensor operations"
            )
        recovery = (
            counts.position_recovery_serial_operations,
            counts.position_recovery_collective_operations,
        )
        if any(type(value) is not int or value < 0 for value in recovery):
            raise ValueError("KMesh position-recovery counts must be exact nonnegative")
        batched = self._is_kmesh_batched_position_recovery_target()
        if outcome is LMCBlendGroupOutcome.RECOVERED:
            if batched:
                if (
                    counts.position_recovery_collective_operations != self.num_layers
                    or counts.position_recovery_serial_operations != 0
                ):
                    raise ValueError(
                        "batched profile member lacks exactly one collective "
                        "position recovery per layer"
                    )
            elif counts.position_recovery_collective_operations:
                raise ValueError(
                    "collective_lockstep_v1 cannot claim batched position recovery"
                )
        if outcome is LMCBlendGroupOutcome.RECOVERED:
            expected_diff = len(set(self.config.blend_check_layers or ()))
            if qkv:
                if (
                    counts.serial_rope_operations != 0
                    or counts.serial_diff_operations != 0
                    or counts.collective_rope_operations != self.num_layers
                    or counts.collective_diff_operations != expected_diff
                ):
                    raise ValueError(
                        "batched-QKV member lacks exactly one collective "
                        "rotary per layer and one difference pass per "
                        "check layer with zero serial work"
                    )
            elif (
                counts.serial_rope_operations != self.num_layers
                or counts.serial_diff_operations != expected_diff
            ):
                raise ValueError(
                    "KMesh recovered member has incomplete serial operation counts"
                )
        elif outcome is LMCBlendGroupOutcome.MISS and any(values):
            raise ValueError("KMesh missed member reported tensor operations")
        elif outcome is LMCBlendGroupOutcome.FAILED:
            expected_diff = len(set(self.config.blend_check_layers or ()))
            if (
                counts.serial_rope_operations > self.num_layers
                or counts.serial_diff_operations > expected_diff
            ):
                raise ValueError(
                    "KMesh failed member reported impossible serial operations"
                )

    def _run_kmesh_collective_plan(
        self,
        plan: _KMeshWorkerReusePlan,
        *,
        generation: int,
        contracts: dict[str, _KMeshWorkerRequestContract],
        kvcaches: list[torch.Tensor],
    ) -> None:
        """Execute one preflighted transported plan in stable lockstep."""
        requests: list[LMCBlendGroupRequest] = []
        for member in plan.members:
            contract = contracts[member.request_id]
            requests.append(
                LMCBlendGroupRequest(
                    request_id=contract.request_id,
                    tokens=list(contract.token_ids),
                    mask=torch.ones(len(contract.token_ids), dtype=torch.bool),
                    generation=generation,
                    retrieve_kwargs={
                        "kvcaches": kvcaches,
                        "slot_mapping": torch.tensor(
                            contract.slot_mapping,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        "vllm_cached_tokens": 0,
                        "request_configs": (contract.execution.as_request_configs()),
                        "explicit_spans": member.explicit_spans,
                        "req_id": contract.lookup_id,
                    },
                )
            )

        assert self.blender is not None
        result = self.blender.blend_group(
            tuple(requests),
            collective_position_recovery=(
                self._is_kmesh_batched_position_recovery_target()
            ),
            collective_qkv=self._is_kmesh_batched_qkv_target(),
        )
        if type(result) is not LMCBlendGroupResult:
            raise TypeError("KMesh collective runner returned an invalid result")
        if type(result.members) is not tuple:
            raise TypeError("KMesh collective result members must be immutable")
        if len(result.members) != len(plan.members):
            raise ValueError("KMesh collective result membership is incomplete")

        validated_members: list[
            tuple[_KMeshWorkerRequestContract, LMCBlendGroupMemberResult]
        ] = []
        for planned, member_result in zip(
            plan.members,
            result.members,
            strict=True,
        ):
            contract = contracts[planned.request_id]
            if type(member_result) is not LMCBlendGroupMemberResult:
                raise TypeError("KMesh collective member result has invalid type")
            if (
                member_result.request_id != planned.request_id
                or member_result.generation != generation
                or type(member_result.outcome) is not LMCBlendGroupOutcome
            ):
                raise ValueError("KMesh collective result identity drifted")
            self._validate_kmesh_collective_operation_counts(
                member_result.outcome,
                member_result.operation_counts,
            )
            counts = member_result.operation_counts
            if member_result.outcome is LMCBlendGroupOutcome.FAILED:
                if not isinstance(member_result.error, Exception):
                    raise ValueError(
                        "KMesh failed collective member requires an exception"
                    )
                if member_result.retrieval_mask is not None:
                    raise ValueError(
                        "KMesh failed collective member cannot publish a mask"
                    )
                frozen_mask = None
            else:
                if member_result.error is not None:
                    raise ValueError(
                        "KMesh terminal collective member cannot publish an error"
                    )
                frozen_mask = self._validate_kmesh_retrieval_mask(
                    contract,
                    member_result.retrieval_mask,
                    planned.explicit_spans,
                )
                recovered = bool(torch.any(frozen_mask).item())
                if member_result.outcome is LMCBlendGroupOutcome.RECOVERED:
                    if not recovered:
                        raise ValueError(
                            "KMesh recovered collective member has an empty mask"
                        )
                elif recovered:
                    raise ValueError(
                        "KMesh missed collective member has a recovered mask"
                    )
                frozen_mask = frozen_mask.clone()

            validated_members.append(
                (
                    contract,
                    LMCBlendGroupMemberResult(
                        request_id=member_result.request_id,
                        generation=member_result.generation,
                        outcome=member_result.outcome,
                        retrieval_mask=frozen_mask,
                        operation_counts=LMCBlendOperationCounts(
                            serial_rope_operations=counts.serial_rope_operations,
                            serial_diff_operations=counts.serial_diff_operations,
                            collective_rope_operations=(
                                counts.collective_rope_operations
                            ),
                            collective_diff_operations=(
                                counts.collective_diff_operations
                            ),
                            position_recovery_serial_operations=(
                                counts.position_recovery_serial_operations
                            ),
                            position_recovery_collective_operations=(
                                counts.position_recovery_collective_operations
                            ),
                        ),
                        error=member_result.error,
                    ),
                )
            )

        frozen_result = LMCBlendGroupResult(
            tuple(member for _contract, member in validated_members)
        )
        self._kmesh_collective_results.append((plan.plan_id, frozen_result))

        for contract, member_result in validated_members:
            if member_result.outcome is LMCBlendGroupOutcome.FAILED:
                assert member_result.error is not None
                self._record_kmesh_load_failure(
                    contract,
                    "request-local collective CacheBlend failure: "
                    f"{member_result.error}",
                    invalidate_full_prompt=True,
                )
                continue
            if member_result.outcome is LMCBlendGroupOutcome.MISS:
                self._record_kmesh_load_failure(
                    contract,
                    "collective explicit lookup/retrieve produced a miss",
                )

    def _start_kmesh_load_kv_locked(
        self,
        forward_context: "ForwardContext",
        metadata: LMCacheConnectorMetadata,
    ) -> None:
        """Run KMesh load while store-session lifecycle state is stable."""
        if self.lmcache_engine is None:
            raise RuntimeError("KMesh target requires a live LMCache engine")
        if self._kmesh_store_sessions:
            raise RuntimeError("KMesh worker retained a store session across steps")
        self._kmesh_failed_worker_requests.clear()

        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)
        if not self.kv_caches:
            raise RuntimeError("KMesh worker has no registered KV caches")
        step = self._require_kmesh_worker_step()
        kvcaches = list(self._validate_kmesh_worker_layers(step))
        generation, requests = self._kmesh_worker_requests(metadata)
        if any(contract.load_spec[2] for _request, contract in requests) and (
            self.blender is None
        ):
            raise RuntimeError(
                "CacheBlend model is not registered. The vLLM worker must "
                "call connector.register_model(model) after construction."
            )

        self._kmesh_collective_results: list[tuple[str, LMCBlendGroupResult]] = []
        for _request, contract in requests:
            self._stats_monitor.update_interval_vllm_hit_tokens(0)
            self._stats_monitor.update_interval_prompt_tokens(len(contract.token_ids))

        if self._is_kmesh_collective_lockstep_target():
            by_request = {
                contract.request_id: contract for _request, contract in requests
            }
            planned: set[str] = set()
            for plan in step.plans:
                plan_contracts = [
                    by_request[member.request_id] for member in plan.members
                ]
                planned.update(member.request_id for member in plan.members)
                if not all(contract.load_spec[2] for contract in plan_contracts):
                    for member, contract in zip(
                        plan.members,
                        plan_contracts,
                        strict=True,
                    ):
                        if contract.load_spec[2]:
                            self._run_kmesh_serial_load(
                                contract,
                                generation=generation,
                                kvcaches=kvcaches,
                                explicit_spans=member.explicit_spans,
                            )
                    continue
                try:
                    self._preflight_kmesh_collective_capacity(
                        plan,
                        by_request,
                        kvcaches,
                    )
                    self._run_kmesh_collective_plan(
                        plan,
                        generation=generation,
                        contracts=by_request,
                        kvcaches=kvcaches,
                    )
                except Exception as error:
                    for contract in plan_contracts:
                        if (
                            contract.request_id
                            not in self._kmesh_failed_worker_requests
                        ):
                            self._record_kmesh_load_failure(
                                contract,
                                f"collective CacheBlend group failure: {error}",
                                invalidate_full_prompt=True,
                            )
                    logger.exception(
                        "KMesh collective plan %s failed; other plans continue",
                        plan.plan_id,
                    )

            for _request, contract in requests:
                if contract.request_id in planned or not contract.load_spec[2]:
                    continue
                self._run_kmesh_serial_load(
                    contract,
                    generation=generation,
                    kvcaches=kvcaches,
                )
            return

        for _request, contract in requests:
            if not contract.load_spec[2]:
                continue
            self._run_kmesh_serial_load(
                contract,
                generation=generation,
                kvcaches=kvcaches,
            )

    def _start_load_kv_impl(
        self,
        forward_context: "ForwardContext",
        **kwargs,
    ) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation
        """
        self.current_layer = 0

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        # LMCache failed to initialize and is running in degraded mode; skip the
        # KV load so vLLM falls back to recompute instead of crashing EngineCore.
        if self.lmcache_engine is None:
            return

        self.layerwise_retrievers = []

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            # Update metrics for all requests that have a load_spec
            if request.load_spec is not None:
                self._stats_monitor.update_interval_vllm_hit_tokens(
                    request.load_spec.vllm_cached_tokens
                )
                self._stats_monitor.update_interval_prompt_tokens(
                    len(request.token_ids)
                )

            if request.load_spec is None or not request.load_spec.can_load:
                continue

            lookup_id = getattr(request, "lookup_id", None) or request.req_id
            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.to(self.device)
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    if self.blender is None:
                        raise RuntimeError(
                            "CacheBlend model is not registered. The vLLM worker "
                            "must call connector.register_model(model) after "
                            "constructing the connector."
                        )
                    ret_token_mask = self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        req_id=lookup_id,
                    )
                    self._record_load_failure_if_needed(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping[:lmcache_cached_tokens],
                        lmcache_cached_tokens - request.load_spec.vllm_cached_tokens,
                    )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                        req_id=lookup_id,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    request_configs=request.request_configs,
                    req_id=lookup_id,
                )

                self._record_load_failure_if_needed(
                    request.req_id,
                    token_mask[:lmcache_cached_tokens],
                    ret_token_mask,
                    slot_mapping[:lmcache_cached_tokens],
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens,
                )

    def _record_load_failure_if_needed(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_expected_tokens: int,
    ) -> None:
        """Invalidate vLLM blocks when an LMCache load is incomplete."""
        num_retrieved_tokens = int(ret_mask.sum().item())
        if num_retrieved_tokens >= num_expected_tokens:
            return

        logger.error(
            "Request %s: The number of retrieved tokens is less than the "
            "expected number of tokens! This should not happen!",
            request_id,
        )
        logger.error(
            "Num retrieved tokens: %d, num expected tokens: %d",
            num_retrieved_tokens,
            num_expected_tokens,
        )
        missing_blocks = self.record_failed_blocks(
            request_id,
            expected_mask,
            ret_mask,
            slot_mapping,
        )
        self._invalid_block_ids.update(missing_blocks)

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> set[int]:
        """Record block IDs associated with failed load attempts.

        Args:
            request_id: request id from vLLM.
            expected_mask: Boolean tensor indicating which tokens were expected to
                be loaded from LMCache. True means the token should be loaded,
                False means the token is already cached in vLLM and does not need
                to be loaded from LMCache.
            ret_mask: Boolean tensor indicating which tokens were actually
                successfully retrieved from LMCache. True means the token was
                successfully loaded. For example, if 256 tokens are expected to be
                loaded, but only 192 tokens are successfully loaded, then the
                ret_mask will be a tensor of 256 items like [T, T, ..., F, F, ...]
                where the first 192 elements are True and the last 64 elements
                are False.
            slot_mapping: Tensor indicating slot IDs for each token. The block
                ID is computed by dividing the slot ID by the block size.

        Example:
            expected_mask = [F, T, T, T] meaning the 1st is in vLLM cache
            ret_mask = [F, T, F, F] meaning failure from loading the 3rd
            missing_mask = expected_mask & ~ret_mask = [F, F, T, T]
            missing_indices = [2, 3]
            then missing_blocks is calculated from slot_mapping and missing_indices

        Returns:
            set[int]: Set of block IDs that failed to load.
        """

        if expected_mask.numel() == 0:
            return set()

        expected_mask_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
        ret_mask_cpu = ret_mask.to(device="cpu", dtype=torch.bool)

        if ret_mask_cpu.shape[0] != expected_mask_cpu.shape[0]:
            logger.debug("expected_mask_cpu.shape[0] != ret_mask_cpu.shape[0]")
            return set()

        missing_mask = expected_mask_cpu & ~ret_mask_cpu
        if not torch.any(missing_mask):
            return set()

        missing_indices = torch.nonzero(missing_mask, as_tuple=False).view(-1)
        if missing_indices.numel() == 0:
            return set()

        slot_mapping_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
        if slot_mapping_cpu.shape[0] > missing_mask.shape[0]:
            slot_mapping_cpu = slot_mapping_cpu[: missing_mask.shape[0]]

        missing_blocks_tensor = torch.unique(
            slot_mapping_cpu[missing_indices] // self._block_size
        )
        missing_blocks = {int(block.item()) for block in missing_blocks_tensor}

        if not missing_blocks:
            return set()

        logger.warning(
            "Request %s failed to load %d tokens across %d blocks",
            request_id,
            missing_indices.numel(),
            len(missing_blocks),
        )
        return missing_blocks

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._is_kmesh_pic_target():
            step = self._require_kmesh_worker_step()
            self._validate_kmesh_worker_layers(step)
            index = self._kmesh_next_wait_layer
            if index >= len(step.layer_names):
                raise RuntimeError("KMesh worker received too many layer waits")
            if self._kmesh_next_save_layer != index:
                raise RuntimeError(
                    "KMesh worker waited for a new layer before saving the prior layer"
                )
            if layer_name != step.layer_names[index]:
                raise ValueError(
                    "KMesh worker layer wait order differs from registered KV caches"
                )
            self._kmesh_next_wait_layer += 1
            return
        raw_metadata = getattr(self._parent, "_connector_metadata", None)
        metadata = (
            raw_metadata if isinstance(raw_metadata, LMCacheConnectorMetadata) else None
        )
        if metadata is not None:
            self._register_worker_lookup_cleanups(metadata)
        try:
            self._wait_for_layer_load_impl(layer_name)
        except BaseException:
            if metadata is not None:
                self._schedule_worker_lookup_cleanups(metadata)
            raise

    def _wait_for_layer_load_impl(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.layerwise_retrievers:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        # Wait for the layer to be loaded
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)

            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")

        if self.layerwise_retrievers:
            self.current_layer += 1

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        if self._is_kmesh_pic_target():
            self._ensure_kmesh_store_session_lifecycle()
            with self._kmesh_store_sessions_lock:
                self._require_kmesh_store_session_admission_open()
                step = self._require_kmesh_worker_step()
                contracts = step.requests
                self._register_kmesh_lookup_cleanups(contracts)
                try:
                    metadata = self._parent._get_connector_metadata()
                    if type(metadata) is not LMCacheConnectorMetadata:
                        raise TypeError("KMesh worker requires exact LMCache metadata")
                    self._save_kmesh_kv_layer_locked(
                        metadata,
                        layer_name,
                        kv_layer,
                    )
                except BaseException:
                    close_error = self._close_kmesh_store_sessions()
                    if close_error is not None:
                        self._kmesh_store_cleanup_errors.append(close_error)
                    self._schedule_kmesh_lookup_cleanups()
                    raise
            return
        if self.lmcache_engine is None:
            return
        raw_metadata = getattr(self._parent, "_connector_metadata", None)
        if raw_metadata is None:
            self._save_kv_layer_impl(
                layer_name,
                kv_layer,
                attn_metadata,
                **kwargs,
            )
            return

        cleanup_metadata = (
            raw_metadata if isinstance(raw_metadata, LMCacheConnectorMetadata) else None
        )
        try:
            if cleanup_metadata is not None:
                self._register_worker_lookup_cleanups(cleanup_metadata)
            metadata = self._parent._get_connector_metadata()
            assert isinstance(metadata, LMCacheConnectorMetadata)
            if metadata is not cleanup_metadata:
                self._register_worker_lookup_cleanups(metadata)
                cleanup_metadata = metadata
            self._save_kv_layer_impl(
                layer_name,
                kv_layer,
                attn_metadata,
                **kwargs,
            )
        except BaseException:
            if cleanup_metadata is not None:
                self._schedule_worker_lookup_cleanups(cleanup_metadata)
            raise

    def _save_kmesh_kv_layer(
        self,
        metadata: LMCacheConnectorMetadata,
        layer_name: str,
        kv_layer: torch.Tensor,
    ) -> None:
        """Advance each live request's explicit store by exactly one layer."""
        self._ensure_kmesh_store_session_lifecycle()
        with self._kmesh_store_sessions_lock:
            self._require_kmesh_store_session_admission_open()
            self._save_kmesh_kv_layer_locked(metadata, layer_name, kv_layer)

    def _save_kmesh_kv_layer_locked(
        self,
        metadata: LMCacheConnectorMetadata,
        layer_name: str,
        kv_layer: torch.Tensor,
    ) -> None:
        """Advance KMesh stores under the session lifecycle lock."""
        if self.lmcache_engine is None:
            raise RuntimeError("KMesh target requires a live LMCache engine")
        if not self.kv_caches:
            raise RuntimeError("KMesh worker has no registered KV caches")
        step = self._require_kmesh_worker_step()
        kvcaches = list(self._validate_kmesh_worker_layers(step))
        layer_index = self._kmesh_next_save_layer
        if layer_index >= len(step.layer_names):
            raise RuntimeError("KMesh worker received too many layer saves")
        if self._kmesh_next_wait_layer != layer_index + 1:
            raise RuntimeError("KMesh worker saved a layer before its matching wait")
        if layer_name != step.layer_names[layer_index]:
            raise ValueError(
                "KMesh worker layer save order differs from registered KV caches"
            )
        if id(kv_layer) != step.kv_cache_ids[layer_index]:
            raise ValueError("KMesh store hook carried a different KV-cache tensor")

        generation, requests = self._kmesh_worker_requests(metadata)
        first_new_session = not self._kmesh_store_sessions

        for _request, contract in requests:
            request_id = contract.request_id
            if request_id in self._kmesh_failed_worker_requests:
                continue
            created_session = False
            try:
                session = self._kmesh_store_sessions.get(request_id)
                if session is None:
                    slot_mapping = torch.tensor(
                        contract.slot_mapping,
                        dtype=torch.long,
                        device=self.device,
                    )
                    store_mask = torch.ones(
                        len(contract.token_ids),
                        dtype=torch.bool,
                    )
                    generator = self.lmcache_engine.store_layer(
                        list(contract.token_ids),
                        mask=store_mask,
                        explicit_spans=contract.execution.explicit_spans,
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping,
                        offset=0,
                        sync=first_new_session,
                        request_configs=contract.execution.as_request_configs(),
                        req_id=request_id,
                    )
                    session = _KMeshLayerwiseStoreSession(
                        generator=generator,
                        cleanup=self.lmcache_engine.drain_pending_store_cleanups,
                        generation=generation,
                        expected_layer_names=step.layer_names,
                    )
                    self._kmesh_store_sessions[request_id] = session
                    created_session = True
                elif session.generation != generation:
                    raise RuntimeError("KMesh store session crossed worker generations")
                session.step_layer(layer_name)
                if created_session:
                    first_new_session = False
            except Exception:
                close_error: Optional[BaseException] = None
                try:
                    if session is not None:
                        session.close()
                except BaseException as caught:
                    close_error = caught
                finally:
                    self._kmesh_failed_worker_requests.add(request_id)
                    if close_error is None:
                        self._kmesh_store_sessions.pop(request_id, None)
                logger.exception(
                    "KMesh request %s explicit store failed; siblings continue",
                    request_id,
                )
                if close_error is not None:
                    self._kmesh_store_cleanup_errors.append(close_error)
                    logger.error(
                        "KMesh request %s store cleanup also failed: %s",
                        request_id,
                        close_error,
                    )
        self._kmesh_next_save_layer += 1

    def _close_kmesh_store_sessions(self) -> Optional[BaseException]:
        """Close every KMesh storer without advancing any missing layer."""
        self._ensure_kmesh_store_session_lifecycle()
        with self._kmesh_store_sessions_lock:
            return self._close_kmesh_store_sessions_locked()

    def _close_kmesh_store_sessions_locked(self) -> Optional[BaseException]:
        """Close every KMesh storer while the session map is stable."""
        first_error: Optional[BaseException] = None
        for request_id, session in tuple(self._kmesh_store_sessions.items()):
            try:
                session.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            else:
                self._kmesh_store_sessions.pop(request_id, None)
        if self.lmcache_engine is not None:
            try:
                self.lmcache_engine.drain_pending_store_cleanups()
                if self.lmcache_engine.pending_store_cleanup_count() != 0:
                    raise RuntimeError(
                        "KMesh worker retained pending store cleanup ownership"
                    )
            except BaseException as error:
                if first_error is None:
                    first_error = error
        return first_error

    def _save_kv_layer_impl(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        # Degraded mode (LMCache init failed): nothing to save, fall back silently.
        if self.lmcache_engine is None:
            return

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0

        kvcaches = list(self.kv_caches.values())
        is_first = True

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            layerwise_storer = self._layerwise_save_storers.get(request.req_id)
            if layerwise_storer is None:
                token_ids = request.token_ids
                assert isinstance(token_ids, list)

                slot_mapping = request.slot_mapping
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.to(self.device)

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    assert save_spec is not None
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.debug(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                    req_id=request.req_id,
                )
                self._layerwise_save_storers[request.req_id] = layerwise_storer
                if is_first:
                    is_first = False

            next(layerwise_storer)

    def _wait_for_kmesh_save_locked(self) -> None:
        """Finish KMesh stores while cleanup and insertion are serialized."""
        self._require_kmesh_store_session_admission_open()
        step = self._require_kmesh_worker_step()
        generation = step.generation
        contracts = step.requests
        if self._kmesh_completed_save_generation == generation:
            cleanup_error = self._schedule_kmesh_lookup_cleanups()
            if cleanup_error is not None:
                raise cleanup_error
            return
        self._register_kmesh_lookup_cleanups(contracts)
        cleanup_error = self._schedule_kmesh_lookup_cleanups()
        data_error = (
            self._kmesh_store_cleanup_errors[0]
            if self._kmesh_store_cleanup_errors
            else None
        )
        try:
            metadata = self._parent._get_connector_metadata()
            if type(metadata) is not LMCacheConnectorMetadata:
                raise TypeError("KMesh worker requires exact LMCache metadata")
            self._kmesh_worker_requests(metadata)
        except BaseException as error:
            if data_error is None:
                data_error = error

        layers_complete = self._kmesh_next_wait_layer == len(
            step.layer_names
        ) and self._kmesh_next_save_layer == len(step.layer_names)
        if not layers_complete and data_error is None:
            data_error = RuntimeError(
                "KMesh worker did not receive every ordered layer hook"
            )

        request_ids = {contract.request_id for contract in contracts}
        unexpected = set(self._kmesh_store_sessions) - request_ids
        if unexpected and data_error is None:
            data_error = RuntimeError(
                "KMesh worker retained a storer for an unknown request"
            )

        for contract in contracts:
            request_id = contract.request_id
            session = self._kmesh_store_sessions.get(request_id)
            try:
                if request_id in self._kmesh_failed_worker_requests:
                    if session is not None:
                        session.close()
                        self._kmesh_store_sessions.pop(request_id, None)
                    continue
                if session is None:
                    raise RuntimeError(
                        f"KMesh request {request_id} received no store hooks"
                    )
                if not layers_complete or session.completed_layers != len(
                    session.expected_layer_names
                ):
                    completed = session.completed_layers
                    expected = len(session.expected_layer_names)
                    session.close()
                    self._kmesh_store_sessions.pop(request_id, None)
                    raise RuntimeError(
                        f"KMesh request {request_id} store stopped after "
                        f"{completed} of {expected} layers"
                    )
                session.finish()
                self._kmesh_store_sessions.pop(request_id, None)
            except BaseException as error:
                if data_error is None:
                    data_error = error

        close_error = self._close_kmesh_store_sessions()
        if data_error is None:
            data_error = close_error
        if data_error is not None and all(
            existing is not data_error for existing in self._kmesh_store_cleanup_errors
        ):
            self._kmesh_store_cleanup_errors.append(data_error)
        if data_error is None:
            self._kmesh_completed_save_generation = generation
        if cleanup_error is not None:
            raise cleanup_error
        if data_error is not None:
            if not self._kmesh_store_sessions:
                self._kmesh_store_cleanup_errors.clear()
                self._kmesh_failed_worker_requests.clear()
            raise data_error
        self._kmesh_store_cleanup_errors.clear()
        self._kmesh_failed_worker_requests.clear()

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        if self._is_kmesh_pic_target():
            self._ensure_kmesh_store_session_lifecycle()
            with self._kmesh_store_sessions_lock:
                self._wait_for_kmesh_save_locked()
            return

        # Degraded mode (LMCache init failed): no engine to save to / unpin from.
        if self.lmcache_engine is None:
            return

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)
        self._register_worker_lookup_cleanups(connector_metadata)
        # At this point the forward/load phase is complete. Arm every cleanup
        # before advancing a storer or entering any request-specific save work;
        # one failure can no longer suppress ownership cleanup for later IDs.
        cleanup_error = self._schedule_worker_lookup_cleanups(connector_metadata)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            # But still need to unpin the kv caches according to req_id
            # to balance the pin count from contains()
            if cleanup_error is not None:
                raise cleanup_error
            return

        if self.use_layerwise:
            first_error = cleanup_error
            for request in connector_metadata.requests:
                try:
                    layerwise_storer = self._layerwise_save_storers.pop(
                        request.req_id, None
                    )
                    if layerwise_storer is not None:
                        next(layerwise_storer)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        # Probe decoder cache before store if bidirectional mode is enabled
        bidir_enabled = getattr(self.config, "pd_bidirectional", False)

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            if len(slot_mapping) != len(token_ids):
                logger.warning(
                    "Skipping KV save for request %s: slot_mapping/token_ids "
                    "length mismatch (slot_mapping=%d, token_ids=%d). Likely "
                    "an upstream allocation/preemption desync; the engine "
                    "stays alive and only this request's save is dropped.",
                    request.req_id,
                    len(slot_mapping),
                    len(token_ids),
                )
                continue

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.to(self.device)

            skip_leading_tokens = save_spec.skip_leading_tokens
            # shared storage disaggregation will not have a disagg_spec passed in
            if self.kv_role == "kv_producer" and request.disagg_spec:
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.debug(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                if not self.enable_blending:
                    token_len = len(token_ids)
                    aligned_token_len = (
                        token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

            # Probe decoder cache before store
            if bidir_enabled and request.disagg_spec is not None:
                try:
                    self._probe_decoder_cache(request, token_ids)
                except Exception as e:
                    logger.warning(
                        "Bidirectional NIXL cache probe failed for %s: %s",
                        request.req_id,
                        e,
                    )

            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
                req_id=request.req_id,
            )

            # Probe decoder cache after store
            if (
                bidir_enabled
                and request.disagg_spec is not None
                and request.disagg_spec.receiver_query_port is not None
            ):
                try:
                    self._probe_decoder_cache(request, token_ids)
                except Exception as e:
                    logger.warning(
                        "Bidirectional NIXL cache probe failed for %s: %s",
                        request.req_id,
                        e,
                    )

            # Update skip_leading_tokens only on last rank to ensure
            # each PP stage stores its own KV cache
            if get_pp_group().is_last_rank:
                # NOTE(Jiayi): We assume all tokens are saved
                save_spec.skip_leading_tokens = len(token_ids)
                if request.disagg_spec:
                    request.disagg_spec.num_transferred_tokens = len(token_ids)

        if cleanup_error is not None:
            raise cleanup_error

    def _probe_decoder_cache(self, request: ReqMeta, token_ids: list[int]) -> None:
        """Query the decoder's cache to check which blocks are already cached.

        This is the bidirectional NIXL cache probe: the prefiller queries the
        decoder via ZMQ to find out which KV blocks are already in the
        decoder's GPU memory. This validates the cache query channel works
        E2E through the real inference path.

        In the future, this information can be used to skip prefill
        computation for cached blocks.
        """
        sm = self.lmcache_engine.storage_manager  # type: ignore[union-attr]
        if sm is None or sm.allocator_backend is None:
            return
        pd_backend = sm.allocator_backend
        if not hasattr(pd_backend, "query_remote_cache"):
            return
        if not hasattr(pd_backend, "cache_query_sockets"):
            return

        # Get query port from LMCache config (pd_peer_query_port)
        query_ports = self.config.pd_peer_query_port
        if query_ports is None:
            return

        # Build cache keys using the token database's process_tokens
        td = self.lmcache_engine.token_database  # type: ignore[union-attr]
        if td is None:
            return

        chunk_keys = []
        for _start, _end, key in td.process_tokens(
            tokens=token_ids, mask=None, make_key=True
        ):
            chunk_keys.append(key)

        if not chunk_keys:
            return

        # Build receiver_id from disagg_spec
        disagg = request.disagg_spec
        init_port = disagg.receiver_init_port  # type: ignore[union-attr]
        if isinstance(init_port, list):
            init_port = init_port[pd_backend.tp_rank]  # type: ignore[union-attr]
        receiver_id = disagg.receiver_host + str(init_port)  # type: ignore[union-attr]

        # Ensure peer and cache query connections
        alloc_port = disagg.receiver_alloc_port  # type: ignore[union-attr]
        if isinstance(alloc_port, list):
            alloc_port = alloc_port[pd_backend.tp_rank]  # type: ignore[union-attr]
        query_port = query_ports[pd_backend.tp_rank]  # type: ignore[union-attr]

        pd_backend._ensure_peer_connection(  # type: ignore[union-attr]
            receiver_id=receiver_id,
            receiver_host=disagg.receiver_host,  # type: ignore[union-attr]
            receiver_init_port=init_port,
            receiver_alloc_port=alloc_port,
        )
        pd_backend._ensure_cache_query_connection(  # type: ignore[union-attr]
            receiver_id=receiver_id,
            receiver_host=disagg.receiver_host,  # type: ignore[union-attr]
            receiver_query_port=query_port,
        )

        # Query decoder cache
        cache_resp = pd_backend.query_remote_cache(receiver_id, chunk_keys)

        logger.info(
            "Bidirectional NIXL cache probe: req=%s, "
            "queried %d chunks, decoder has %d cached "
            "(%.0f%% hit rate)",
            request.req_id,
            len(chunk_keys),
            len(cache_resp.cached_keys),
            100.0 * len(cache_resp.cached_keys) / len(chunk_keys) if chunk_keys else 0,
        )

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
        return invalid_blocks

    @_lmcache_nvtx_annotate
    def shutdown(self) -> None:
        """Shutdown the connector by delegating to LMCacheManager."""
        logger.info("Starting LMCacheConnector shutdown...")
        try:
            if self._is_kmesh_pic_target() and self._role == KVConnectorRole.WORKER:
                self._shutdown_kmesh_store_session_lifecycle()

            self._manager.stop_services()
            if not self._manager.services_stopped:
                cleanup_errors = self._manager.terminal_cleanup_errors
                if cleanup_errors:
                    raise cleanup_errors[0]
                raise RuntimeError("LMCacheManager terminal cleanup did not complete")
        except BaseException:
            if not self._manager.services_stopped:
                # The connector may be dropped after this exact failure
                # escapes. Retain a process-level strong owner so public
                # cleanup retry can still reach the Manager and its last
                # engine-owner registration.
                self._manager.retain_construction_cleanup()
            raise
        if self._role == KVConnectorRole.WORKER:
            self.blender = None

    def _shutdown_kmesh_store_session_lifecycle(self) -> None:
        """Seal worker save admission and drain every published session."""
        self._ensure_kmesh_store_session_lifecycle()
        with self._kmesh_store_sessions_lock:
            self._kmesh_store_sessions_closed = True
            cleanup_error = self._schedule_kmesh_lookup_cleanups()
            close_error = self._close_kmesh_store_sessions()
            if cleanup_error is not None:
                raise cleanup_error
            if self._kmesh_outstanding_lookup_ids:
                raise RuntimeError("KMesh shutdown retained lookup cleanup ownership")
            if close_error is not None:
                raise close_error
            if self._kmesh_store_sessions:
                raise RuntimeError("KMesh shutdown retained a store session")

    ###################
    # Scheduler side APIs
    ####################

    def prepare_request_lookup(
        self,
        request: "Request",
        peer_sidecars: tuple[KVConnectorRequestSidecar, ...],
    ) -> None:
        """Bind one target request to the exact KMesh scheduler sidecar."""
        if not self._is_kmesh_pic_target():
            return
        if self._role != KVConnectorRole.SCHEDULER:
            raise RuntimeError("KMesh request evidence may only bind on the scheduler")
        if type(peer_sidecars) is not tuple:
            raise TypeError("KMesh request sidecars must be an immutable tuple")

        request_id = self._kmesh_request_id(request)
        prior_request = self._kmesh_prepared_request_objects.get(request_id)
        if prior_request is not None and prior_request is not request:
            raise ValueError("KMesh request object changed between lookup probes")

        kmesh_sidecar: Optional[KVConnectorRequestSidecar] = None
        for index, sidecar in enumerate(peer_sidecars):
            if type(sidecar) is not KVConnectorRequestSidecar:
                raise TypeError(f"peer_sidecars[{index}] has an invalid exact type")
            if type(sidecar.provider_id) is not str:
                raise TypeError(f"peer_sidecars[{index}].provider_id must be a string")
            if sidecar.provider_id != KMESH_REQUEST_SIDECAR_PROVIDER_ID:
                raise ValueError("KMesh target received an unknown sidecar provider")
            if kmesh_sidecar is not None:
                raise ValueError("KMesh target received duplicate request sidecars")
            kmesh_sidecar = sidecar

        prior_lookup = self._kmesh_prepared_lookups.get(request_id)
        if kmesh_sidecar is None:
            if prior_lookup is not None:
                self._rebuild_kmesh_execution(prior_lookup, request)
            self._kmesh_prepared_lookups.pop(request_id, None)
            self._kmesh_prepared_request_objects.pop(request_id, None)
            return

        descriptor = parse_kmesh_request_sidecar(
            kmesh_sidecar.provider_id,
            kmesh_sidecar.payload,
        )
        validate_kmesh_request_lookup_evidence(descriptor, request)
        if descriptor.request_id != request_id:
            raise ValueError("KMesh descriptor request ID differs from the request")

        if prior_lookup is not None:
            prior_lookup.validate_repeated_prepare(request, descriptor)
            self._rebuild_kmesh_execution(prior_lookup, request)
            self._kmesh_prepared_request_objects[request_id] = request
            return

        execution = None
        if descriptor.permits_serial_pic:
            runtime_evidence = self._fresh_kmesh_runtime_evidence()
            binding = bind_kmesh_request_lookup(
                descriptor,
                request,
                admission_generation=descriptor.admission_generation,
                cache_domain=descriptor.cache_domain,
                runtime_evidence=runtime_evidence,
                request_configs=extract_request_configs(request.sampling_params),
            )
            execution = KMeshRequestExecutionEvidence.from_binding(binding)

        self._kmesh_prepared_lookups[request_id] = _KMeshPreparedLookup(
            descriptor=descriptor,
            request_object=request,
            execution=execution,
        )
        self._kmesh_prepared_request_objects[request_id] = request

    def _ensure_lookup_attempt_state(self) -> None:
        """Initialize scheduler lookup-attempt state for normal/test objects."""
        if not hasattr(self, "_prealloc_lookup_ids"):
            self._prealloc_lookup_ids = {}
        if not hasattr(self, "_blocked_prealloc_requests"):
            self._blocked_prealloc_requests = set()
        if not hasattr(self, "_allocated_lookup_ids"):
            self._allocated_lookup_ids = {}
        if not hasattr(self, "_worker_owned_lookup_requests"):
            self._worker_owned_lookup_requests = set()

    def _get_or_create_prealloc_lookup_id(self, request_id: str) -> str:
        self._ensure_lookup_attempt_state()
        lookup_id = self._prealloc_lookup_ids.get(request_id)
        if lookup_id is None:
            lookup_id = str(uuid4())
            self._prealloc_lookup_ids[request_id] = lookup_id
        return lookup_id

    def _cancel_kmesh_prealloc_attempt(
        self,
        request_id: str,
        lookup_id: str,
    ) -> Optional[BaseException]:
        """Forget and exact-cancel one failed scheduler lookup attempt."""
        cleanup_error: Optional[BaseException] = None
        try:
            if self.lookup_client is None:
                raise RuntimeError("KMesh scheduler lost its lookup client")
            self.lookup_client.cancel_lookup(lookup_id)
        except BaseException as error:
            cleanup_error = error
        finally:
            if self._prealloc_lookup_ids.get(request_id) == lookup_id:
                self._prealloc_lookup_ids.pop(request_id, None)
            self._blocked_prealloc_requests.discard(request_id)
            self.load_specs.pop(request_id, None)
            self._requests_priority.pop(request_id, None)
        return cleanup_error

    def _mark_lookup_allocated(self, request_id: str) -> None:
        self._ensure_lookup_attempt_state()
        if request_id in self._blocked_prealloc_requests:
            raise RuntimeError(
                "cannot allocate a lookup before cancellation is acknowledged"
            )
        lookup_id = self._prealloc_lookup_ids.pop(request_id, None)
        if lookup_id is None:
            self.lookup_client.clear_lookup_status(request_id)  # type: ignore[union-attr]
            return
        # Keep scheduler cancellation identity alive until build_connector_meta
        # has completed every fallible operation and is ready to hand this UUID
        # to the worker. Allocation alone is not worker ownership.
        self._allocated_lookup_ids[request_id] = lookup_id

    def _finish_lookup_allocation(self, request_id: str) -> None:
        if getattr(self, "async_loading", False):
            self.lookup_client.clear_lookup_status(request_id)  # type: ignore[union-attr]
        else:
            self._mark_lookup_allocated(request_id)

    def _commit_metadata_lookup_transfers(
        self,
        metadata: "LMCacheConnectorMetadata",
    ) -> None:
        """Atomically retire scheduler state for UUIDs encoded in metadata."""
        self._ensure_lookup_attempt_state()
        encoded: dict[str, str] = {}
        for request in metadata.requests:
            lookup_id = request.lookup_id
            if lookup_id is None:
                continue
            prior = encoded.setdefault(request.req_id, lookup_id)
            if prior != lookup_id:
                raise RuntimeError("request metadata contains multiple lookup attempts")
        if encoded != self._allocated_lookup_ids:
            raise RuntimeError(
                "allocated lookup attempts were omitted or changed in metadata"
            )
        if not encoded:
            return

        # Stateful clients preflight the whole batch under one lock and then
        # perform only non-failing local retirement. Do not mutate phase maps
        # until that atomic client handoff succeeds.
        lookup_ids = list(encoded.values())
        self.lookup_client.transfer_lookups(lookup_ids)  # type: ignore[union-attr]
        for request_id in encoded:
            self._allocated_lookup_ids.pop(request_id, None)
            self._worker_owned_lookup_requests.add(request_id)

    def _cancel_allocated_lookup_ids(self) -> None:
        """Rollback every allocation that did not reach returned metadata."""
        self._ensure_lookup_attempt_state()
        for request_id, lookup_id in tuple(self._allocated_lookup_ids.items()):
            self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[union-attr]
            self._allocated_lookup_ids.pop(request_id, None)

    def _revoke_kmesh_allocated_attempts(self) -> Optional[BaseException]:
        """Cancel failed metadata attempts and require a fresh scheduler probe."""
        self._ensure_lookup_attempt_state()
        first_error: Optional[BaseException] = None
        for request_id, lookup_id in tuple(self._allocated_lookup_ids.items()):
            try:
                if self.lookup_client is None:
                    raise RuntimeError("KMesh scheduler lost its lookup client")
                self.lookup_client.cancel_lookup(lookup_id)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            finally:
                self._allocated_lookup_ids.pop(request_id, None)
                self._discard_kmesh_scheduler_request(request_id)
                self._unfinished_requests.pop(request_id, None)
                self._requests_priority.pop(request_id, None)
                self._blocked_prealloc_requests.discard(request_id)
        return first_error

    def _retire_finished_kmesh_scheduler_requests(
        self,
        finished_ids: frozenset[str],
    ) -> None:
        """Cancel scheduler-owned finished attempts before batch transfer."""
        first_error: Optional[BaseException] = None
        for request_id in finished_ids:
            lookup_ids = tuple(
                lookup_id
                for lookup_id in (
                    self._prealloc_lookup_ids.pop(request_id, None),
                    self._allocated_lookup_ids.pop(request_id, None),
                )
                if lookup_id is not None
            )
            for lookup_id in lookup_ids:
                try:
                    if self.lookup_client is None:
                        raise RuntimeError("KMesh scheduler lost its lookup client")
                    self.lookup_client.cancel_lookup(lookup_id)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            self._worker_owned_lookup_requests.discard(request_id)
            self._blocked_prealloc_requests.discard(request_id)
            self._discard_kmesh_scheduler_request(request_id)
            self._request_trackers.pop(request_id, None)
            self._unfinished_requests.pop(request_id, None)
            self._requests_priority.pop(request_id, None)
        if first_error is not None:
            raise first_error

    def _cancel_unscheduled_prealloc_lookups(self) -> None:
        """Cancel probes that did not receive allocation in this round."""
        self._ensure_lookup_attempt_state()
        attempts = tuple(self._prealloc_lookup_ids.items())
        if self._is_kmesh_pic_target():
            first_error: Optional[BaseException] = None
            for request_id, lookup_id in attempts:
                error = self._cancel_kmesh_prealloc_attempt(request_id, lookup_id)
                if first_error is None and error is not None:
                    first_error = error
            if first_error is not None:
                raise first_error
            return
        for request_id, lookup_id in attempts:
            getattr(self, "load_specs", {}).pop(request_id, None)
            getattr(self, "_requests_priority", {}).pop(request_id, None)
            if request_id in self._blocked_prealloc_requests:
                if self.lookup_client.lookup_cache(lookup_id) is None:  # type: ignore[union-attr]
                    continue
            else:
                self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[union-attr]
                if self.lookup_client.lookup_cache(lookup_id) is None:  # type: ignore[union-attr]
                    self._blocked_prealloc_requests.add(request_id)
                    continue
            self._blocked_prealloc_requests.discard(request_id)
            self._prealloc_lookup_ids.pop(request_id, None)

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self._is_kmesh_pic_target():
            if self._role != KVConnectorRole.SCHEDULER:
                raise RuntimeError("KMesh target lookup may only run on the scheduler")
            request_id = self._kmesh_request_id(request)
            if self._kmesh_prepared_request_objects.get(request_id) is not request:
                raise RuntimeError(
                    "KMesh target lookup lacks prepared request evidence"
                )
            prepared = self._kmesh_prepared_lookups.get(request_id)
            if prepared is None:
                raise RuntimeError("KMesh target lookup lost prepared evidence")
            execution = self._rebuild_kmesh_execution(prepared, request)
            if execution is None:
                if request_id in self.load_specs:
                    raise RuntimeError("fallback KMesh lookup retained load state")
                return 0
            if type(num_computed_tokens) is not int or num_computed_tokens != 0:
                raise ValueError(
                    f"{self._target_profile} requires exactly zero local "
                    "computed tokens"
                )
            if self.lookup_client is None:
                raise RuntimeError("KMesh target requires a scheduler lookup client")
            if not self.lookup_client.supports_explicit_spans():
                raise RuntimeError(
                    "KMesh target lookup client does not support explicit spans"
                )

            lookup_id = self._get_or_create_prealloc_lookup_id(request_id)
            try:
                if request_id in self._blocked_prealloc_requests:
                    cached = self.lookup_client.lookup_cache(lookup_id)
                    if cached == -1:
                        self._blocked_prealloc_requests.discard(request_id)
                        self._prealloc_lookup_ids.pop(request_id, None)
                        lookup_id = self._get_or_create_prealloc_lookup_id(request_id)

                token_ids = list(execution.prompt_token_ids)
                kmesh_request_configs = execution.as_request_configs()
                explicit_spans = execution.explicit_spans
                self.lookup_client.validate_lookup_request(
                    token_ids,
                    lookup_id=lookup_id,
                    request_configs=kmesh_request_configs,
                    explicit_spans=explicit_spans,
                )
                hit_end = self.lookup_client.lookup_cache(lookup_id)
                if hit_end == -1:
                    hit_end = self.lookup_client.lookup(
                        token_ids,
                        lookup_id=lookup_id,
                        request_configs=kmesh_request_configs,
                        explicit_spans=explicit_spans,
                    )
                if hit_end is None:
                    self._blocked_prealloc_requests.add(request_id)
                    return None
                valid_hit_ends = {0, *(span.end for span in explicit_spans.spans)}
                if type(hit_end) is not int or hit_end not in valid_hit_ends:
                    raise ValueError(
                        "KMesh explicit lookup returned an invalid greatest span end"
                    )

                recalculate_last = int(hit_end == len(token_ids) and hit_end > 0)
                advertised_tokens = hit_end - recalculate_last
                self.load_specs[request_id] = LoadSpec(
                    vllm_cached_tokens=0,
                    lmcache_cached_tokens=hit_end,
                    can_load=False,
                )
                return advertised_tokens
            except BaseException:
                cleanup_error = self._cancel_kmesh_prealloc_attempt(
                    request_id,
                    lookup_id,
                )
                if cleanup_error is not None:
                    logger.error(
                        "KMesh lookup cleanup for %s also failed: %s",
                        lookup_id,
                        cleanup_error,
                    )
                raise

        # Ignore DP attention mock requests
        if request.request_id.startswith("mock_req"):
            return 0
        # to handle preempted requests, we want `get_num_new_matched_tokens` to be
        # idempotent under the condition that `update_state_after_alloc` is NOT called
        # then the two side-effects that must be idempotent are:
        # 1. lookup_client caches a result
        #     uncached in `update_state_after_alloc` if this request can be scheduled
        # 2. cache engine will pin the KV caches for the request
        #     unpinned in `wait_for_save` if this request can be scheduled
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        req_id = request.request_id

        # Degraded mode (LMCache init failed): no lookup client is available, so
        # report no external hits and let vLLM recompute instead of asserting and
        # crashing EngineCore during scheduling.
        if self.lookup_client is None:
            return 0

        # Async lookup workers retain their established request-ID protocol.
        # Every synchronous scheduling attempt instead uses a fresh UUID so a
        # preempted/retried public request ID cannot unpin another attempt.
        async_loading = getattr(self, "async_loading", False)
        lookup_id = (
            req_id if async_loading else self._get_or_create_prealloc_lookup_id(req_id)
        )
        if not async_loading and req_id in self._blocked_prealloc_requests:
            if self.lookup_client.lookup_cache(lookup_id) == -1:
                # The prior attempt's typed cancellation has completed. Only
                # now may this request rotate to a fresh UUID and re-probe.
                self._blocked_prealloc_requests.discard(req_id)
                self._prealloc_lookup_ids.pop(req_id, None)
                lookup_id = self._get_or_create_prealloc_lookup_id(req_id)

        # Reconstruct the exact request content on every scheduler probe. Sync
        # and bypass clients bind cached status to this content; async/external
        # clients keep the validation hook as a no-op and preserve their
        # existing non-blocking lookup lifecycle.
        token_ids = request.all_token_ids

        # If the request has multimodal hashes, apply them to the token ids.
        mm_hashes, mm_positions = extract_mm_features(request)
        if mm_hashes and mm_positions:
            # TODO(Jiayi): Optimize this
            token_ids = torch.tensor(request.prompt_token_ids)
            apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
            token_ids = token_ids.tolist()

        request_configs = extract_request_configs(request.sampling_params)
        if self.skip_last_n_tokens > 0:
            token_ids = token_ids[: -self.skip_last_n_tokens]

        self.lookup_client.validate_lookup_request(
            token_ids,
            lookup_id=lookup_id,
            request_configs=request_configs,
        )
        if (
            num_external_hit_tokens := self.lookup_client.lookup_cache(
                lookup_id=lookup_id
            )
        ) != -1:
            # -1 means no result cached
            # None or int means ongoing (async) or cached result
            logger.debug(
                f"Found {num_external_hit_tokens} hit tokens for request"
                f" {req_id} in the lookup cache."
            )
        else:
            logger.debug(f"Looking up cache for the first time for request {req_id}!")
            self._requests_priority[req_id] = getattr(request, "priority", 0)

            num_external_hit_tokens = self.lookup_client.lookup(
                token_ids,
                lookup_id=lookup_id,
                request_configs=request_configs,
            )

        if num_external_hit_tokens is None:
            if not async_loading:
                # Sync ``None`` means delivery was uncertain and its UUID is
                # being generation-cancelled. Keep the request bound to that
                # UUID until all workers return the typed cancel ACK.
                self._blocked_prealloc_requests.add(req_id)
            logger.debug(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: None.",
                req_id,
                request.num_tokens,
                num_computed_tokens,
            )
            return None

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            need_to_allocate -= 1

        # Check if hit tokens meet the minimum for retrieve
        # If below minimum, skip retrieve but still record hit tokens
        # for skip_leading_tokens to avoid re-storing existing chunks
        min_retrieve = self.config.min_retrieve_tokens
        below_min_retrieve = min_retrieve > 0 and need_to_allocate < min_retrieve

        if below_min_retrieve:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, but need to load: %d < min_retrieve %d, "
                "skip retrieve but record for save skip",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
                min_retrieve,
            )
        else:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, need to load: %d",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
            )

        # Chunked KV loading: cap the number of tokens reported
        # to the scheduler to avoid exhausting the GPU block pool
        # when many concurrent requests each need large allocations.
        # Remaining tokens beyond the cap will be computed locally
        # via chunked prefill rather than loaded from external cache.
        capped_lmcache_tokens = num_external_hit_tokens
        if (
            self._max_tokens_per_load > 0
            and need_to_allocate > self._max_tokens_per_load
        ):
            # Align cap to LMCache chunk boundaries so that the
            # retrieve path receives chunk-aligned token ranges.
            cap = (
                self._max_tokens_per_load
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            need_to_allocate = cap
            # Align capped_lmcache_tokens to chunk boundary so that
            # session.get_hashes() receives a chunk-aligned end value.
            capped_lmcache_tokens = (
                (num_computed_tokens + cap)
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            logger.debug(
                "Reqid: %s, Chunked KV loading: capped from "
                "%d to %d external tokens "
                "(max_tokens_per_load=%d)",
                req_id,
                num_external_hit_tokens - num_computed_tokens,
                need_to_allocate,
                self._max_tokens_per_load,
            )

        self.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=capped_lmcache_tokens,
            can_load=False,
        )

        if below_min_retrieve or need_to_allocate <= 0:
            return 0

        # TODO: Align to vLLM block size. Should test whether it can be removed
        # need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size

        return need_to_allocate

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """

        if self._is_kmesh_pic_target():
            if self._role != KVConnectorRole.SCHEDULER:
                raise RuntimeError("KMesh allocation state belongs to the scheduler")
            if type(num_external_tokens) is not int or num_external_tokens < 0:
                raise ValueError(
                    "KMesh allocation requires a non-negative exact integer hit"
                )
            request_id = self._kmesh_request_id(request)
            if self._kmesh_prepared_request_objects.get(request_id) is not request:
                raise RuntimeError("KMesh allocation lacks prepared request evidence")
            prepared = self._kmesh_prepared_lookups.get(request_id)
            if prepared is None:
                raise RuntimeError("KMesh allocation lost prepared request evidence")
            execution = self._rebuild_kmesh_execution(prepared, request)
            load_spec = self.load_specs.get(request_id)
            if execution is None:
                if num_external_tokens != 0 or load_spec is not None:
                    raise ValueError("fallback KMesh allocation cannot load cache data")
            else:
                if load_spec is None:
                    raise RuntimeError("actionable KMesh allocation lost lookup state")
                recalculate_last = int(
                    load_spec.lmcache_cached_tokens == len(execution.prompt_token_ids)
                    and load_spec.lmcache_cached_tokens > 0
                )
                expected = load_spec.lmcache_cached_tokens - recalculate_last
                if num_external_tokens != expected:
                    raise ValueError(
                        "KMesh allocation differs from the advertised explicit hit"
                    )
                load_spec.can_load = num_external_tokens > 0
                self._finish_lookup_allocation(request_id)
            self._unfinished_requests[request_id] = request
            self._kmesh_allocated_requests.add(request_id)
            return

        # Degraded mode (LMCache init failed): there is no lookup client to clear
        # and no load spec was recorded, so nothing to do.
        if self.lookup_client is None:
            return

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
                receiver_query_port=req_disagg_spec.get("receiver_query_port"),
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec
        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            self._finish_lookup_allocation(request.request_id)
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            self._finish_lookup_allocation(request.request_id)
            return

        recalc_last = (
            1
            if (
                self.load_specs[request.request_id].lmcache_cached_tokens
                == request.num_tokens
            )
            else 0
        )
        assert (
            num_external_tokens
            == self.load_specs[request.request_id].lmcache_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
            - recalc_last
        ), (
            f"Mismatch in tokens to load: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} "
            "(tokens in lmcache) - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens} "
            "(tokens in vllm) - "
            f"{recalc_last} "
            "(full lmcache hits subtracts last token to recalculate logits)"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True
        self._finish_lookup_allocation(request.request_id)

    def _discard_kmesh_scheduler_request(self, request_id: str) -> None:
        """Idempotently discard all scheduler-only handoff state for a request."""
        self._kmesh_prepared_lookups.pop(request_id, None)
        self._kmesh_prepared_request_objects.pop(request_id, None)
        self._kmesh_allocated_requests.discard(request_id)
        self.load_specs.pop(request_id, None)

    def _consume_kmesh_scheduler_execution(
        self,
        request_id: str,
    ) -> _KMeshConsumedSchedulerExecution:
        """Consume an exact allocated handoff without conflating its fallback."""
        if request_id not in self._kmesh_allocated_requests:
            if (
                request_id in self._kmesh_prepared_lookups
                or request_id in self._kmesh_prepared_request_objects
            ):
                raise RuntimeError(
                    "scheduled KMesh request was prepared but not allocated"
                )
            return _KMeshConsumedSchedulerExecution(False, None, None)
        request = self._unfinished_requests.get(request_id)
        prepared_request = self._kmesh_prepared_request_objects.get(request_id)
        if request is None or prepared_request is not request:
            raise RuntimeError("allocated KMesh request lost its live scheduler owner")
        prepared = self._kmesh_prepared_lookups.get(request_id)
        if prepared is None:
            raise RuntimeError("allocated KMesh request lost its sidecar evidence")
        execution = self._rebuild_kmesh_execution(prepared, request)
        self._kmesh_allocated_requests.remove(request_id)
        self._kmesh_prepared_lookups.pop(request_id, None)
        self._kmesh_prepared_request_objects.pop(request_id, None)
        return _KMeshConsumedSchedulerExecution(
            True,
            prepared.descriptor,
            execution,
        )

    @staticmethod
    def _validate_kmesh_token_list(value: object, field_name: str) -> list[int]:
        """Validate a plain non-negative token-ID list."""
        if type(value) is not list or any(
            type(token_id) is not int or token_id < 0 for token_id in value
        ):
            raise TypeError(f"{field_name} must be a plain non-negative integer list")
        return value

    @classmethod
    def _validate_kmesh_live_token_list(
        cls,
        value: object,
        field_name: str,
    ) -> list[int]:
        """Copy and validate vLLM's exact read-only live token view."""
        if type(value) is not ConstantList:
            raise TypeError(f"{field_name} must be exactly vLLM ConstantList")
        token_ids = cast(ConstantList[int], value).copy()
        return cls._validate_kmesh_token_list(token_ids, field_name)

    @staticmethod
    def _validate_kmesh_block_ids(
        value: object,
        field_name: str,
        *,
        allow_none: bool,
    ) -> list[int]:
        """Validate the target's exact one-cache-group block-ID shape."""
        if value is None and allow_none:
            return []
        if type(value) is not tuple or len(value) != 1 or type(value[0]) is not list:
            raise TypeError(f"{field_name} must contain exactly one plain block list")
        block_ids = value[0]
        if any(type(block_id) is not int or block_id < 0 for block_id in block_ids):
            raise TypeError(f"{field_name} must contain non-negative integers")
        if len(set(block_ids)) != len(block_ids):
            raise ValueError(f"{field_name} must not contain duplicate block IDs")
        return block_ids.copy()

    @staticmethod
    def _validate_kmesh_request_config_match(
        expected: Optional[dict],
        actual: Optional[dict],
        field_name: str,
    ) -> None:
        """Compare optional request configs without bool/int equality aliases."""
        if expected is None or actual is None:
            if expected is not actual:
                raise ValueError(f"{field_name} differs from the live request")
            return
        if type(expected) is not dict or type(actual) is not dict:
            raise TypeError(f"{field_name} must be a plain dictionary or null")
        if any(type(key) is not str for key in expected) or any(
            type(key) is not str for key in actual
        ):
            raise TypeError(f"{field_name} keys must be plain strings")
        if set(expected) != set(actual):
            raise ValueError(f"{field_name} differs from the live request")
        for key in expected:
            expected_value = expected[key]
            actual_value = actual[key]
            if (
                type(expected_value) is not type(actual_value)
                or expected_value != actual_value
            ):
                raise ValueError(f"{field_name} differs from the live request")

    def _validate_kmesh_scheduler_output(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[tuple[str, ...], tuple[str, ...], frozenset[str]]:
        """Validate exact scheduler cardinality, IDs, and allocation ownership."""
        if type(scheduler_output) is not SchedulerOutput:
            raise TypeError("KMesh target requires exact SchedulerOutput metadata")
        if type(scheduler_output.scheduled_new_reqs) is not list:
            raise TypeError("scheduled_new_reqs must be a plain list")

        new_ids: list[str] = []
        for index, scheduled_request in enumerate(scheduler_output.scheduled_new_reqs):
            if type(scheduled_request) is not NewRequestData:
                raise TypeError(
                    f"scheduled_new_reqs[{index}] must be exactly NewRequestData"
                )
            new_ids.append(
                self._validate_kmesh_request_id_value(
                    scheduled_request.req_id,
                    f"scheduled_new_reqs[{index}].req_id",
                )
            )

        cached = scheduler_output.scheduled_cached_reqs
        if type(cached) is not CachedRequestData:
            raise TypeError("KMesh target requires exact CachedRequestData metadata")
        if type(cached.req_ids) is not list:
            raise TypeError("cached req_ids must be a plain list")
        cached_ids = tuple(
            self._validate_kmesh_request_id_value(
                request_id,
                f"scheduled_cached_reqs.req_ids[{index}]",
            )
            for index, request_id in enumerate(cached.req_ids)
        )

        new_id_tuple = tuple(new_ids)
        all_ids = (*new_id_tuple, *cached_ids)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("scheduled KMesh request IDs must be unique")

        cached_count = len(cached_ids)
        exact_cached_lists = {
            "new_block_ids": cached.new_block_ids,
            "num_computed_tokens": cached.num_computed_tokens,
            "num_output_tokens": cached.num_output_tokens,
        }
        for field_name, values in exact_cached_lists.items():
            if type(values) is not list or len(values) != cached_count:
                raise ValueError(
                    f"cached {field_name} cardinality differs from req_ids"
                )
        for index, token_count in enumerate(cached.num_computed_tokens):
            if type(token_count) is not int or token_count < 0:
                raise TypeError(
                    "cached num_computed_tokens items must be non-negative ints; "
                    f"item {index} is invalid"
                )
        for index, token_count in enumerate(cached.num_output_tokens):
            if type(token_count) is not int or token_count < 0:
                raise TypeError(
                    "cached num_output_tokens items must be non-negative ints; "
                    f"item {index} is invalid"
                )
        if type(cached.new_token_ids) is not list or len(cached.new_token_ids) not in (
            0,
            cached_count,
        ):
            raise ValueError("cached new_token_ids cardinality differs from req_ids")
        for index, token_ids in enumerate(cached.new_token_ids):
            self._validate_kmesh_token_list(
                token_ids,
                f"scheduled_cached_reqs.new_token_ids[{index}]",
            )
        if type(cached.all_token_ids) is not dict:
            raise TypeError("cached all_token_ids must be a plain dictionary")
        unexpected_all_token_ids = set(cached.all_token_ids) - set(cached_ids)
        if unexpected_all_token_ids:
            raise ValueError("cached all_token_ids contains an unscheduled request")
        for cached_request_id, token_ids in cached.all_token_ids.items():
            self._validate_kmesh_request_id_value(
                cached_request_id,
                "scheduled_cached_reqs.all_token_ids key",
            )
            self._validate_kmesh_token_list(
                token_ids,
                f"scheduled_cached_reqs.all_token_ids[{cached_request_id!r}]",
            )
        if type(cached.resumed_req_ids) is not set:
            raise TypeError("cached resumed_req_ids must be a plain set")
        for resumed_request_id in cached.resumed_req_ids:
            self._validate_kmesh_request_id_value(
                resumed_request_id,
                "scheduled_cached_reqs.resumed_req_ids item",
            )
        if not cached.resumed_req_ids <= set(cached_ids):
            raise ValueError("cached resumed_req_ids contains an unscheduled request")

        scheduled_tokens = scheduler_output.num_scheduled_tokens
        if type(scheduled_tokens) is not dict:
            raise TypeError("num_scheduled_tokens must be a plain dictionary")
        for scheduled_request_id, token_count in scheduled_tokens.items():
            self._validate_kmesh_request_id_value(
                scheduled_request_id,
                "num_scheduled_tokens key",
            )
            if type(token_count) is not int or token_count <= 0:
                raise ValueError("scheduled KMesh token counts must be positive ints")
        if set(scheduled_tokens) != set(all_ids):
            raise ValueError("num_scheduled_tokens IDs differ from scheduled requests")
        if type(
            scheduler_output.total_num_scheduled_tokens
        ) is not int or scheduler_output.total_num_scheduled_tokens != sum(
            scheduled_tokens.values()
        ):
            raise ValueError("total_num_scheduled_tokens differs from its exact sum")

        if type(scheduler_output.finished_req_ids) is not set:
            raise TypeError("finished_req_ids must be a plain set")
        finished_ids = frozenset(
            self._validate_kmesh_request_id_value(
                request_id,
                "finished_req_ids item",
            )
            for request_id in scheduler_output.finished_req_ids
        )
        if finished_ids & set(all_ids):
            raise ValueError("finished and scheduled KMesh request IDs overlap")

        prepared_ids = set(self._kmesh_prepared_lookups)
        prepared_owner_ids = set(self._kmesh_prepared_request_objects)
        if prepared_ids != prepared_owner_ids:
            raise RuntimeError("KMesh prepared evidence and owner maps diverged")
        allocated_ids = set(self._kmesh_allocated_requests)
        if not allocated_ids <= prepared_ids:
            raise RuntimeError("allocated KMesh request lacks prepared evidence")
        if not set(new_id_tuple) <= allocated_ids:
            raise RuntimeError("scheduled new KMesh request lacks exact allocation")
        unexpected_allocated = allocated_ids - set(all_ids) - set(finished_ids)
        if unexpected_allocated:
            missing = ", ".join(sorted(unexpected_allocated))
            raise RuntimeError(
                "allocated KMesh requests were omitted from scheduler metadata: "
                + missing
            )
        for allocated_request_id in allocated_ids:
            live_request = self._unfinished_requests.get(allocated_request_id)
            if live_request is None:
                raise RuntimeError("allocated KMesh request lost its live owner")
            if (
                self._kmesh_prepared_request_objects[allocated_request_id]
                is not live_request
            ):
                raise RuntimeError("allocated KMesh request owner identity drifted")

        return new_id_tuple, cached_ids, finished_ids

    def _build_kmesh_new_request_tracker(
        self,
        new_request: NewRequestData,
        live_request: object,
        scheduled_tokens: int,
        handoff: _KMeshConsumedSchedulerExecution,
    ) -> RequestTracker:
        """Build a target tracker without legacy global or multimodal side effects."""
        if not handoff.allocation_observed or handoff.descriptor is None:
            raise RuntimeError("scheduled new KMesh request lacks exact handoff state")
        request_id = self._kmesh_request_id(live_request)
        if new_request.req_id != request_id:
            raise ValueError("scheduled new request ID differs from its live owner")

        live_tokens = self._validate_kmesh_token_list(
            getattr(live_request, "prompt_token_ids", None),
            "live KMesh prompt_token_ids",
        )
        new_tokens = self._validate_kmesh_token_list(
            new_request.prompt_token_ids,
            "scheduled new prompt_token_ids",
        )
        if new_tokens != live_tokens:
            raise ValueError("scheduled new prompt tokens differ from the live request")
        prefill_tokens = self._validate_kmesh_token_list(
            new_request.prefill_token_ids,
            "scheduled new prefill_token_ids",
        )
        if prefill_tokens != live_tokens:
            raise ValueError(
                "scheduled new prefill tokens differ from the live request"
            )
        if handoff.descriptor.prompt_token_count != len(live_tokens):
            raise ValueError("scheduled new prompt length differs from KMesh evidence")
        if handoff.execution is not None and (
            handoff.execution.prompt_token_ids != tuple(live_tokens)
        ):
            raise ValueError("scheduled new prompt differs from KMesh execution")
        live_all_tokens = self._validate_kmesh_live_token_list(
            getattr(live_request, "all_token_ids", None),
            "live KMesh all_token_ids",
        )
        if live_all_tokens != live_tokens:
            raise ValueError("scheduled new request contains non-prompt live tokens")

        live_num_computed = getattr(live_request, "num_computed_tokens", None)
        if type(live_num_computed) is not int or live_num_computed < 0:
            raise TypeError("live num_computed_tokens must be a non-negative int")
        if (
            type(new_request.num_computed_tokens) is not int
            or new_request.num_computed_tokens != live_num_computed
        ):
            raise ValueError(
                "scheduled new num_computed_tokens differs from the live request"
            )
        if live_num_computed + scheduled_tokens != len(live_tokens):
            raise ValueError("scheduled new KMesh request is not one full prefill")

        if type(new_request.mm_features) is not list or new_request.mm_features:
            raise ValueError("KMesh non-actionable handoff requires no multimodal data")
        if new_request.prompt_embeds is not None:
            raise ValueError("KMesh non-actionable handoff rejects prompt embeddings")
        if new_request.lora_request is not None:
            raise ValueError("KMesh non-actionable handoff rejects live LoRA state")
        if new_request.pooling_params is not None:
            raise ValueError("KMesh non-actionable handoff rejects pooling requests")
        prompt_is_token_ids = new_request.prompt_is_token_ids
        if prompt_is_token_ids is not None and (
            type(prompt_is_token_ids) is not list
            or len(prompt_is_token_ids) != len(live_tokens)
            or any(
                type(item) is not bool or item is not True
                for item in prompt_is_token_ids
            )
        ):
            raise ValueError("scheduled new prompt_is_token_ids evidence is invalid")

        live_sampling_params = getattr(live_request, "sampling_params", None)
        live_request_configs = extract_request_configs(live_sampling_params)
        scheduled_request_configs = extract_request_configs(new_request.sampling_params)
        self._validate_kmesh_request_config_match(
            live_request_configs,
            scheduled_request_configs,
            "scheduled new request configs",
        )

        block_ids = self._validate_kmesh_block_ids(
            new_request.block_ids,
            "scheduled new block_ids",
            allow_none=False,
        )
        if len(block_ids) * self._block_size < len(live_tokens):
            raise ValueError("scheduled new prompt does not fit its block slots")
        request_configs = (
            handoff.execution.as_request_configs()
            if handoff.execution is not None
            else live_request_configs
        )
        return RequestTracker(
            req_id=request_id,
            prompt_len=len(live_tokens),
            token_ids=live_tokens.copy(),
            allocated_block_ids=block_ids,
            lookup_id=None,
            num_saved_tokens=0,
            disagg_spec=None,
            mm_hashes=None,
            mm_positions=None,
            request_configs=(
                request_configs.copy()
                if type(request_configs) is dict
                else request_configs
            ),
            kmesh_execution=None,
            skip_save=True,
            num_lmcache_cached_tokens=0,
        )

    @staticmethod
    def _clone_kmesh_request_tracker(tracker: RequestTracker) -> RequestTracker:
        """Clone one closed-gate tracker before a fallible staged update."""
        if type(tracker) is not RequestTracker:
            raise TypeError("KMesh scheduler tracker must be exactly RequestTracker")
        LMCacheConnectorV1Impl._validate_kmesh_request_id_value(
            tracker.req_id,
            "KMesh scheduler tracker request ID",
        )
        if (
            type(tracker.prompt_len) is not int
            or tracker.prompt_len <= 0
            or tracker.prompt_len > len(tracker.token_ids)
        ):
            raise ValueError("KMesh scheduler tracker prompt length is invalid")
        if tracker.lookup_id is not None or tracker.disagg_spec is not None:
            raise ValueError("KMesh scheduler tracker owns legacy transfer state")
        if tracker.mm_hashes or tracker.mm_positions:
            raise ValueError("KMesh scheduler tracker owns multimodal state")
        if tracker.kmesh_execution is not None:
            raise ValueError(
                "KMesh scheduler tracker retained stale execution evidence"
            )
        if tracker.skip_save is not True:
            raise ValueError("KMesh scheduler tracker must keep cache saving disabled")
        if tracker.num_saved_tokens != 0 or tracker.num_lmcache_cached_tokens != 0:
            raise ValueError("KMesh scheduler tracker retained cache accounting state")
        if type(tracker.token_ids) is not list or any(
            type(token_id) is not int or token_id < 0 for token_id in tracker.token_ids
        ):
            raise TypeError("KMesh scheduler tracker token IDs are invalid")
        if type(tracker.allocated_block_ids) is not list or any(
            type(block_id) is not int or block_id < 0
            for block_id in tracker.allocated_block_ids
        ):
            raise TypeError("KMesh scheduler tracker block IDs are invalid")
        if len(set(tracker.allocated_block_ids)) != len(tracker.allocated_block_ids):
            raise ValueError("KMesh scheduler tracker contains duplicate block IDs")
        clone = RequestTracker(
            req_id=tracker.req_id,
            prompt_len=tracker.prompt_len,
            token_ids=tracker.token_ids.copy(),
            allocated_block_ids=tracker.allocated_block_ids.copy(),
            lookup_id=None,
            num_saved_tokens=tracker.num_saved_tokens,
            disagg_spec=None,
            mm_hashes=None,
            mm_positions=None,
            request_configs=(
                tracker.request_configs.copy()
                if type(tracker.request_configs) is dict
                else tracker.request_configs
            ),
            kmesh_execution=None,
            skip_save=True,
            num_lmcache_cached_tokens=tracker.num_lmcache_cached_tokens,
        )
        if type(tracker.is_decode_phase) is not bool:
            raise TypeError("KMesh tracker decode phase must be a boolean")
        clone.is_decode_phase = tracker.is_decode_phase
        return clone

    def _build_kmesh_non_actionable_req_meta(
        self,
        tracker: RequestTracker,
        execution: KMeshRequestExecutionEvidence,
        lookup_id: str,
        load_spec: LoadSpec,
    ) -> ReqMeta:
        """Build exact full-prompt metadata for one online explicit request."""
        if type(execution) is not KMeshRequestExecutionEvidence:
            raise TypeError("execution must be exactly KMeshRequestExecutionEvidence")
        if tracker.req_id != execution.request_id:
            raise ValueError("request tracker ID differs from KMesh execution")
        if tracker.prompt_len != len(execution.prompt_token_ids):
            raise ValueError(
                "request tracker prompt length differs from KMesh evidence"
            )
        if tuple(tracker.token_ids[: tracker.prompt_len]) != execution.prompt_token_ids:
            raise ValueError("request tracker prompt tokens differ from KMesh evidence")
        if tracker.lookup_id is not None or tracker.disagg_spec is not None:
            raise ValueError("KMesh non-actionable metadata cannot own transfer state")
        self._validate_kmesh_lookup_id(lookup_id, "KMesh lookup_id")
        if type(load_spec) is not LoadSpec:
            raise TypeError("KMesh load_spec must be exactly LoadSpec")
        if (
            type(load_spec.vllm_cached_tokens) is not int
            or load_spec.vllm_cached_tokens != 0
            or type(load_spec.lmcache_cached_tokens) is not int
            or type(load_spec.can_load) is not bool
        ):
            raise ValueError("KMesh load_spec has invalid exact fields")
        valid_hit_ends = {
            0,
            *(span.end for span in execution.explicit_spans.spans),
        }
        if load_spec.lmcache_cached_tokens not in valid_hit_ends:
            raise ValueError("KMesh load_spec has an invalid greatest span end")
        advertised_tokens = load_spec.lmcache_cached_tokens - int(
            load_spec.lmcache_cached_tokens == len(execution.prompt_token_ids)
            and load_spec.lmcache_cached_tokens > 0
        )
        if load_spec.can_load is not (advertised_tokens > 0):
            raise ValueError("KMesh load_spec allocation state is inconsistent")

        token_ids = list(execution.prompt_token_ids)
        num_blocks = len(tracker.allocated_block_ids)
        if num_blocks * self._block_size < len(token_ids):
            raise ValueError("KMesh prompt does not fit in its allocated block slots")
        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, self._block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, self._block_size))
            + block_ids.reshape((num_blocks, 1)) * self._block_size
        ).flatten()[: len(token_ids)]
        if len(slot_mapping) != len(token_ids):
            raise ValueError("KMesh full-prompt slot mapping is incomplete")

        request_configs = execution.as_request_configs()
        tracker.request_configs = request_configs.copy()
        tracker.kmesh_execution = execution
        try:
            return ReqMeta(
                req_id=tracker.req_id,
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                lookup_id=lookup_id,
                is_last_prefill=True,
                save_spec=SaveSpec(0, True),
                load_spec=load_spec,
                disagg_spec=None,
                request_configs=request_configs.copy(),
                kmesh_execution=execution,
            )
        finally:
            # Evidence is valid only for the current context-step metadata.
            # Cached decode must not replay it after KMesh omits the descriptor.
            tracker.kmesh_execution = None

    def _build_kmesh_non_actionable_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> LMCacheConnectorMetadata:
        """Stage exact evidence and commit scheduler state only after success."""
        if self._role != KVConnectorRole.SCHEDULER:
            raise RuntimeError("KMesh connector metadata may only build on scheduler")

        new_ids, cached_ids, finished_ids = self._validate_kmesh_scheduler_output(
            scheduler_output
        )
        original_prepared = self._kmesh_prepared_lookups
        original_prepared_owners = self._kmesh_prepared_request_objects
        original_allocated = self._kmesh_allocated_requests
        original_trackers = self._request_trackers
        original_unfinished = self._unfinished_requests
        original_load_specs = self.load_specs

        staged_prepared = original_prepared.copy()
        staged_prepared_owners = original_prepared_owners.copy()
        staged_allocated = original_allocated.copy()
        staged_trackers = original_trackers.copy()
        staged_unfinished = original_unfinished.copy()
        staged_load_specs = original_load_specs.copy()
        try:
            self._kmesh_prepared_lookups = staged_prepared
            self._kmesh_prepared_request_objects = staged_prepared_owners
            self._kmesh_allocated_requests = staged_allocated
            self._request_trackers = staged_trackers
            self._unfinished_requests = staged_unfinished
            self.load_specs = staged_load_specs
            return self._build_staged_kmesh_non_actionable_meta(
                scheduler_output,
                new_ids,
                cached_ids,
                finished_ids,
            )
        except BaseException:
            self._kmesh_prepared_lookups = original_prepared
            self._kmesh_prepared_request_objects = original_prepared_owners
            self._kmesh_allocated_requests = original_allocated
            self._request_trackers = original_trackers
            self._unfinished_requests = original_unfinished
            self.load_specs = original_load_specs
            raise

    def _build_staged_kmesh_non_actionable_meta(
        self,
        scheduler_output: SchedulerOutput,
        new_ids: tuple[str, ...],
        cached_ids: tuple[str, ...],
        finished_ids: frozenset[str],
    ) -> LMCacheConnectorMetadata:
        """Build metadata against private staged maps owned by one transaction."""

        meta = LMCacheConnectorMetadata()
        for request_id in finished_ids:
            self._discard_kmesh_scheduler_request(request_id)
            self._request_trackers.pop(request_id, None)
            self._unfinished_requests.pop(request_id, None)

        for request_id, scheduled_request in zip(
            new_ids,
            scheduler_output.scheduled_new_reqs,
            strict=True,
        ):
            handoff = self._consume_kmesh_scheduler_execution(request_id)
            if not handoff.allocation_observed:
                raise RuntimeError("scheduled new KMesh request lacks exact allocation")
            live_request = self._unfinished_requests.get(request_id)
            if live_request is None:
                raise RuntimeError("scheduled new KMesh request lost its live owner")
            tracker = self._build_kmesh_new_request_tracker(
                scheduled_request,
                live_request,
                scheduler_output.num_scheduled_tokens[request_id],
                handoff,
            )
            self._request_trackers[request_id] = tracker
            if handoff.execution is not None:
                load_spec = self.load_specs.pop(request_id, None)
                lookup_id = self._allocated_lookup_ids.get(request_id)
                if load_spec is None or lookup_id is None:
                    raise RuntimeError(
                        "actionable KMesh metadata lost lookup ownership"
                    )
                meta.add_request(
                    self._build_kmesh_non_actionable_req_meta(
                        tracker,
                        handoff.execution,
                        lookup_id,
                        load_spec,
                    )
                )
            elif (
                request_id in self.load_specs
                or request_id in self._allocated_lookup_ids
            ):
                raise RuntimeError("fallback KMesh metadata retained lookup state")

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(cached_ids):
            existing_tracker = self._request_trackers.get(request_id)
            live_request = self._unfinished_requests.get(request_id)
            if existing_tracker is None or live_request is None:
                raise ValueError("cached KMesh request has no scheduler-owned tracker")
            tracker = self._clone_kmesh_request_tracker(existing_tracker)
            self._request_trackers[request_id] = tracker

            cached_num_computed = cached_reqs.num_computed_tokens[index]
            live_num_computed = getattr(live_request, "num_computed_tokens", None)
            if (
                type(cached_num_computed) is not int
                or cached_num_computed < 0
                or type(live_num_computed) is not int
                or cached_num_computed != live_num_computed
            ):
                raise ValueError(
                    "cached num_computed_tokens differs from the live request"
                )
            handoff = self._consume_kmesh_scheduler_execution(request_id)
            num_new_tokens = scheduler_output.num_scheduled_tokens[request_id]
            num_current_tokens = cached_num_computed
            new_block_ids = self._validate_kmesh_block_ids(
                cached_reqs.new_block_ids[index],
                f"scheduled cached new_block_ids[{index}]",
                allow_none=True,
            )
            # A synchronous worker load failure leaves the scheduler's prior
            # transferred-attempt witness but no new lookup handoff. Accept
            # only the exact, residue-free full-prompt local recompute shape.
            local_full_recompute = (
                not handoff.allocation_observed
                and request_id in self._worker_owned_lookup_requests
                and request_id not in cached_reqs.resumed_req_ids
                and cached_reqs.num_output_tokens[index] == 0
                and not new_block_ids
                and num_current_tokens < tracker.prompt_len
                and num_current_tokens + num_new_tokens == tracker.prompt_len
                and len(existing_tracker.token_ids) == existing_tracker.prompt_len
                and len(existing_tracker.allocated_block_ids) * self._block_size
                >= existing_tracker.prompt_len
                and request_id not in self._kmesh_prepared_lookups
                and request_id not in self._kmesh_prepared_request_objects
                and request_id not in self._kmesh_allocated_requests
                and request_id not in self._prealloc_lookup_ids
                and request_id not in self._blocked_prealloc_requests
                and request_id not in self._allocated_lookup_ids
                and request_id not in self.load_specs
            )
            if handoff.allocation_observed:
                if handoff.descriptor is None:
                    raise RuntimeError("allocated cached KMesh handoff lost evidence")
                if handoff.descriptor.prompt_token_count != tracker.prompt_len:
                    raise ValueError(
                        "cached KMesh handoff prompt differs from its tracker"
                    )
                if num_current_tokens + num_new_tokens != tracker.prompt_len:
                    raise ValueError(
                        "allocated cached KMesh handoff is not one full prefill"
                    )
            elif num_current_tokens < tracker.prompt_len and not local_full_recompute:
                raise RuntimeError(
                    "cached KMesh prefill lacks exact preparation and allocation"
                )

            all_token_ids = self._validate_kmesh_live_token_list(
                getattr(live_request, "all_token_ids", None),
                "live cached all_token_ids",
            )
            carried_all_token_ids = cached_reqs.all_token_ids.get(request_id)
            if (
                carried_all_token_ids is not None
                and carried_all_token_ids != all_token_ids
            ):
                raise ValueError("cached all_token_ids differs from the live request")
            tracker_len = len(tracker.token_ids)
            slice_base = min(num_current_tokens, tracker_len)
            new_token_ids = all_token_ids[slice_base : slice_base + num_new_tokens]
            if len(new_token_ids) != num_new_tokens:
                raise ValueError("cached live tokens do not cover num_scheduled_tokens")
            if any(cached_reqs.new_token_ids) and (
                cached_reqs.new_token_ids[index] != new_token_ids
            ):
                raise ValueError("cached new_token_ids differs from the live request")
            preempted = request_id in cached_reqs.resumed_req_ids

            if num_current_tokens < tracker_len:
                num_token_slots = len(tracker.allocated_block_ids) * self._block_size
                tokens_to_keep = min(num_current_tokens, num_token_slots)
                tracker.token_ids = all_token_ids[:tokens_to_keep]
                tracker.num_saved_tokens = 0

            tracker.update(
                new_token_ids,
                (new_block_ids,),
                preempted=preempted,
                lmcache_cached_tokens=0,
                vllm_cached_tokens=0,
                all_token_ids=(all_token_ids.copy() if preempted else None),
            )
            if len(set(tracker.allocated_block_ids)) != len(
                tracker.allocated_block_ids
            ):
                raise ValueError("cached KMesh tracker contains duplicate block IDs")
            tracker.skip_save = True
            tracker.kmesh_execution = None
            if handoff.execution is not None:
                load_spec = self.load_specs.pop(request_id, None)
                lookup_id = self._allocated_lookup_ids.get(request_id)
                if load_spec is None or lookup_id is None:
                    raise RuntimeError(
                        "actionable cached KMesh metadata lost lookup ownership"
                    )
                tracker.token_ids = list(handoff.execution.prompt_token_ids)
                meta.add_request(
                    self._build_kmesh_non_actionable_req_meta(
                        tracker,
                        handoff.execution,
                        lookup_id,
                        load_spec,
                    )
                )
            elif (
                request_id in self.load_specs
                or request_id in self._allocated_lookup_ids
            ):
                raise RuntimeError(
                    "fallback cached KMesh metadata retained lookup state"
                )

        if self._kmesh_allocated_requests:
            missing = ", ".join(sorted(self._kmesh_allocated_requests))
            raise RuntimeError(
                "allocated KMesh requests were omitted from scheduler metadata: "
                + missing
            )

        # Prepared-but-unscheduled probes own no cache resources and are safe to
        # forget at the scheduling-round boundary.
        self._kmesh_prepared_lookups.clear()
        self._kmesh_prepared_request_objects.clear()
        return meta

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Build metadata and make its lookup ownership handoff transactional."""
        if self._is_kmesh_pic_target():
            try:
                self._cancel_unscheduled_prealloc_lookups()
                _new_ids, _cached_ids, finished_ids = (
                    self._validate_kmesh_scheduler_output(scheduler_output)
                )
                self._retire_finished_kmesh_scheduler_requests(finished_ids)
            except BaseException:
                cleanup_error = self._revoke_kmesh_allocated_attempts()
                if cleanup_error is not None:
                    logger.error(
                        "KMesh scheduler rollback cleanup also failed: %s",
                        cleanup_error,
                    )
                raise

            scheduler_state = (
                self._kmesh_prepared_lookups,
                self._kmesh_prepared_request_objects,
                self._kmesh_allocated_requests,
                self._request_trackers,
                self._unfinished_requests,
                self.load_specs,
            )
            try:
                metadata = self._build_kmesh_non_actionable_meta(scheduler_output)
                self._commit_metadata_lookup_transfers(metadata)
                return metadata
            except BaseException:
                (
                    self._kmesh_prepared_lookups,
                    self._kmesh_prepared_request_objects,
                    self._kmesh_allocated_requests,
                    self._request_trackers,
                    self._unfinished_requests,
                    self.load_specs,
                ) = scheduler_state
                cleanup_error = self._revoke_kmesh_allocated_attempts()
                if cleanup_error is not None:
                    logger.error(
                        "KMesh scheduler rollback cleanup also failed: %s",
                        cleanup_error,
                    )
                raise
        try:
            metadata = self._build_connector_meta(scheduler_output)
            self._commit_metadata_lookup_transfers(metadata)
            return metadata
        except BaseException:
            # EngineCore does not guarantee a later request_finished callback
            # when metadata construction itself raises. Exact-cancel every
            # ALLOCATED_PENDING_METADATA UUID before propagating the failure.
            self._cancel_allocated_lookup_ids()
            raise

    def _build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> LMCacheConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        # Degraded mode (LMCache init failed): no lookup client means no load specs
        # were ever recorded, so return empty metadata for the worker-side hooks.
        if self.lookup_client is None:
            return LMCacheConnectorMetadata()

        # ``get_num_new_matched_tokens`` can probe more requests than vLLM
        # allocates in this scheduling round. No-forward batches skip worker
        # ``wait_for_save``, so the scheduler must cancel every UUID that did
        # not move to the transferred map in ``update_state_after_alloc``.
        self._cancel_unscheduled_prealloc_lookups()

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        # We should load KV for:
        # 1. new requests
        # 2. preempted requests (once per recovery)
        # can_load will only be True if `update_state_after_alloc` has been called
        # which only happens when vLLM's KV manager has space to receive KV from LMCache
        for scheduled_request in scheduler_output.scheduled_new_reqs:
            # Ignore DP attention mock requests
            if scheduled_request.req_id.startswith("mock_req"):
                continue
            load_spec = self.load_specs.pop(scheduled_request.req_id, None)
            num_tokens_to_compute = (
                scheduled_request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[scheduled_request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(scheduled_request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            lookup_id = self._allocated_lookup_ids.get(scheduled_request.req_id)
            request_tracker = RequestTracker.from_new_request(
                self.config,
                scheduled_request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
                lookup_id=lookup_id,
            )
            self._request_trackers[scheduled_request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)
            request_tracker.lookup_id = None

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                load_spec = self.load_specs.pop(req.req_id, None)
                lmcache_cached_tokens = 0
                vllm_cached_tokens = 0
                if load_spec is not None:
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                request_tracker = self._request_trackers[req.req_id]
                request_tracker.lookup_id = self._allocated_lookup_ids.get(req.req_id)

                # Pass all_token_ids for preempted requests to restore
                # token_ids correctly for chunk key computation
                all_token_ids = None
                if req.resumed_from_preemption:
                    vllm_request = self._unfinished_requests.get(req.req_id)
                    assert vllm_request is not None, (
                        f"Preempted request {req.req_id} not found "
                        "in _unfinished_requests"
                    )
                    all_token_ids = list(vllm_request.all_token_ids)

                request_tracker.update(
                    req.new_token_ids,
                    req.new_block_ids,
                    req.resumed_from_preemption,
                    lmcache_cached_tokens=lmcache_cached_tokens,
                    vllm_cached_tokens=vllm_cached_tokens,
                    all_token_ids=all_token_ids,
                )

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self.config.save_decode_cache,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
                request_tracker.lookup_id = None
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            request_tracker.lookup_id = self._allocated_lookup_ids.get(req_id)
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: this is a dangerous reference to the request object inside vllm
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = request.num_computed_tokens
                # tracker_len < num_computed_tokens during decode
                #   (important for save_decode_cache).
                # num_computed_tokens < tracker_len after preemption.
                tracker_len = len(request_tracker.token_ids)
                slice_base = min(num_current_tokens, tracker_len)
                new_token_ids = request.all_token_ids[
                    slice_base : slice_base + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            load_spec = self.load_specs.pop(req_id, None)
            lmcache_cached_tokens = 0
            vllm_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                vllm_cached_tokens = load_spec.vllm_cached_tokens

            # Handle both old and new versions of CachedRequestData
            if hasattr(cached_reqs, "resumed_req_ids"):
                # New version with resumed_req_ids
                preempted = req_id in cached_reqs.resumed_req_ids
            elif hasattr(cached_reqs, "resumed_from_preemption"):
                # Old version with resumed_from_preemption
                preempted = cached_reqs.resumed_from_preemption[i]
            else:
                # This case should not be reached with supported vLLM versions.
                # Raising an error is safer than assuming not preempted.
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}. "
                    f"This might be due to an unsupported vLLM version."
                )
            if preempted:
                assert load_spec is not None, (
                    f"Request {req_id} is preempted but was not given a load spec"
                )
                # num_computed_tokens should be reset to 0 during preemption
                # and then set to the number of already cached tokens (maxxing
                # prefix caching and lmcache)
                # this assumption is crucial for the update() call of RequestTracker
                # On full cache hit, get_num_new_matched_tokens subtracts 1
                # to force last-token recomputation. This only affects
                # num_computed_tokens when lmcache has all tokens AND
                # provides more than vLLM's local cache.
                expected = max(lmcache_cached_tokens, load_spec.vllm_cached_tokens)
                full_hit_adj = (
                    lmcache_cached_tokens == len(request.all_token_ids)
                    and lmcache_cached_tokens > load_spec.vllm_cached_tokens
                )
                if full_hit_adj:
                    expected -= 1
                assert request.num_computed_tokens == expected, (
                    f"Preempted request {req_id} has "
                    f"num_computed_tokens {request.num_computed_tokens} "
                    f"but expected {expected} "
                    f"(full_hit_adj={full_hit_adj})"
                )

            # When retrieve fail, vllm will call _handle_invalid_blocks to
            # reset request.num_computed_tokens, this will lead to
            # request_tracker.token_ids being not matched with vllm
            if num_current_tokens < len(request_tracker.token_ids):
                logger.warning(
                    "Request %s rolled back from %d to %d tokens; "
                    "truncating tracker state.",
                    req_id,
                    len(request_tracker.token_ids),
                    num_current_tokens,
                )
                num_token_slots = (
                    len(request_tracker.allocated_block_ids) * self._block_size
                )
                tokens_to_keep = num_current_tokens
                if num_token_slots < num_current_tokens:
                    logger.warning(
                        "Request %s tracker has %d token slots but %d tokens; "
                        "capping token_ids to slot capacity.",
                        req_id,
                        num_token_slots,
                        num_current_tokens,
                    )
                    tokens_to_keep = num_token_slots

                request_tracker.token_ids = list(request.all_token_ids[:tokens_to_keep])
                request_tracker.num_saved_tokens = min(
                    request_tracker.num_saved_tokens, tokens_to_keep
                )

            # Pass all_token_ids for preempted requests to restore
            # token_ids correctly for chunk key computation
            all_token_ids = list(request.all_token_ids) if preempted else None

            request_tracker.update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)
            request_tracker.lookup_id = None

        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Layerwise save uses request-scoped generators. If request finishes
        # without entering wait_for_save (abort/error/evict path), make sure
        # we release the generator entry to avoid leaking state.
        if getattr(self, "use_layerwise", False) and hasattr(
            self, "_layerwise_save_storers"
        ):
            self._layerwise_save_storers.pop(request.request_id, None)

        if self._is_kmesh_pic_target():
            if self._role != KVConnectorRole.SCHEDULER:
                raise RuntimeError("KMesh request completion belongs to scheduler")
            request_id = self._kmesh_request_id(request)
            self._ensure_lookup_attempt_state()
            prealloc_lookup_id = self._prealloc_lookup_ids.pop(request_id, None)
            allocated_lookup_id = self._allocated_lookup_ids.pop(request_id, None)
            scheduler_lookup_id = prealloc_lookup_id or allocated_lookup_id
            had_worker_owner = request_id in self._worker_owned_lookup_requests
            self._worker_owned_lookup_requests.discard(request_id)
            self._blocked_prealloc_requests.discard(request_id)
            if scheduler_lookup_id is not None:
                if self.lookup_client is None:
                    raise RuntimeError(
                        "KMesh request completion lost its lookup client"
                    )
                self.lookup_client.cancel_lookup(scheduler_lookup_id)
            elif had_worker_owner:
                # Metadata transfer made the worker the sole pin owner. Its
                # wait/no-forward/clear path performs the exact unpin.
                pass
            self._discard_kmesh_scheduler_request(request_id)
            self._request_trackers.pop(request_id, None)
            self._unfinished_requests.pop(request_id, None)
            return False, None

        self._ensure_lookup_attempt_state()
        request_id = request.request_id
        prealloc_lookup_id = self._prealloc_lookup_ids.pop(request_id, None)
        allocated_lookup_id = self._allocated_lookup_ids.pop(request_id, None)
        had_worker_owned_lookup = request_id in self._worker_owned_lookup_requests
        scheduler_owned_lookup_id = prealloc_lookup_id or allocated_lookup_id
        # Current PREALLOC or ALLOCATED_PENDING_METADATA ownership takes
        # precedence over an older worker-owned attempt after preemption.
        lookup_transferred = (
            scheduler_owned_lookup_id is None and had_worker_owned_lookup
        )
        self._blocked_prealloc_requests.discard(request_id)
        if had_worker_owned_lookup:
            self._worker_owned_lookup_requests.discard(request_id)

        # Cleanup if request was aborted. The per-branch logic below already
        # degrades gracefully when ``lmcache_engine`` and/or ``lookup_client``
        # are missing (it warns and skips only the unavailable backend), so we
        # must not short-circuit the whole method here -- doing so would drop
        # the storage/lookup cancels that still need to run when just one of
        # the two is present. See LMCache#3337.
        if request.status == RequestStatus.FINISHED_ABORTED:
            # ``request_finished`` is a Scheduler-side connector API.
            # The Scheduler typically does not initialize the storage
            # engine (unless ``enable_scheduler_bypass_lookup`` is set);
            # only the Worker role builds it by default. The Scheduler
            # *does* own the lookup_client though, so the async lookup
            # cancel below must run independently of the engine check
            # to avoid leaking in-flight async lookups on Scheduler-side
            # aborts. See LMCache#3337.
            if self.lmcache_engine is None:
                logger.warning(
                    "Skipping abort-time backend cleanup for request %s: "
                    "lmcache_engine is not initialized (Scheduler role "
                    "without enable_scheduler_bypass_lookup).",
                    request.request_id,
                )
            else:
                # Notify storage backends of aborted requests
                sm = self.lmcache_engine.storage_manager
                if sm is not None:
                    sm.cancel_request(request.request_id)

            # Cancel worker-side lookup resources independently of
            # ``lmcache_engine``. In the ordinary split-process topology the
            # Scheduler owns only the lookup client, while synchronous lookup
            # pins live on the Worker engine. An abort before allocation never
            # reaches Worker ``wait_for_save()``, so this RPC is the only owner
            # capable of balancing those pins.
            if self.lookup_client is None:
                logger.warning(
                    "Skipping abort-time lookup cancel for request %s: "
                    "lookup_client is not initialized. Engine stays alive; "
                    "this request's lookup is dropped.",
                    request.request_id,
                )
            elif getattr(self, "async_loading", False):
                self.lookup_client.cancel_lookup(request_id)
            elif not lookup_transferred:
                # PREALLOC is scheduler-owned. A fresh attempt UUID is used
                # when present; ALLOCATED_PENDING_METADATA is still rollback-
                # capable here. The request-ID fallback preserves degraded and
                # legacy test paths that never entered scheduler lookup.
                self.lookup_client.cancel_lookup(
                    scheduler_owned_lookup_id or request_id
                )
        elif self.lookup_client is not None:
            if getattr(self, "async_loading", False):
                self.lookup_client.finish_lookup(request_id)
            elif not lookup_transferred:
                if scheduler_owned_lookup_id is not None:
                    self.lookup_client.cancel_lookup(scheduler_owned_lookup_id)
                else:
                    self.lookup_client.finish_lookup(request_id)

        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        if self.config.get_extra_config_value(
            "enable_cache_usage_details_in_response", False
        ):
            request_tracker = self._request_trackers.get(request.request_id)
            if request_tracker:
                return_params = return_params or {}
                return_params["num_lmcache_cached_tokens"] = (
                    request_tracker.num_lmcache_cached_tokens
                )

        return False, return_params

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.lmcache_engine is not None:
            return self.lmcache_engine.get_kv_events()
        return []
