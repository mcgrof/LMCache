# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional, Union
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
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm.version import __version__ as VLLM_VERSION
import msgspec
import torch

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache import utils
from lmcache.banner import print_banner_once
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    extract_request_configs_from_sampling_params,
    lmcache_get_or_create_config,
)
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheStoreEvent, _lmcache_nvtx_annotate, cdiv
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import validate_and_set_config_value
from lmcache.v1.manager import LMCacheManager
from lmcache.v1.storage_backend.raw_block import RawBlockPublicationReceipt
from lmcache.v1.storage_backend.storage_pd_protocol import (
    STORAGE_PD_INCARNATION,
    StoragePDDelivery,
    StoragePDNotificationQueue,
    StoragePDObligation,
    StoragePDReadAck,
    StoragePDState,
    StoragePDStatus,
    StoragePDStatusSender,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    # First Party
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


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


# How many finished storage P/D requests to remember. A finished request
# keeps only its identifier, so a status that arrives after it ended is
# recognised instead of starting it again. The oldest are forgotten.
STORAGE_PD_REQUEST_HISTORY = 4096


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    return extract_request_configs_from_sampling_params(sampling_params)


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

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    # The number of tokens that are cached in LMCache for this request
    num_lmcache_cached_tokens: int = 0

    # Per-rank durable shared-storage publications supplied by the proxy.
    storage_pd_statuses: Optional[list[dict[str, Any]]] = None
    storage_pd_request_id: Optional[str] = None

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
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

        kv_transfer_params = getattr(new_request, "kv_transfer_params", None)
        storage_pd_statuses = None
        storage_pd_request_id = None
        if isinstance(kv_transfer_params, dict):
            raw_statuses = kv_transfer_params.get("lmcache.storage_pd_statuses")
            if raw_statuses is not None:
                if not isinstance(raw_statuses, list) or not all(
                    isinstance(status, dict) for status in raw_statuses
                ):
                    raise ValueError(
                        "lmcache.storage_pd_statuses must be a list of mappings"
                    )
                storage_pd_statuses = raw_statuses
                raw_request_id = kv_transfer_params.get("lmcache.storage_pd_request_id")
                if not isinstance(raw_request_id, str) or not raw_request_id:
                    raise ValueError(
                        "lmcache.storage_pd_request_id must be a non-empty string"
                    )
                storage_pd_request_id = raw_request_id

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
            num_lmcache_cached_tokens=lmcache_cached_tokens,
            storage_pd_statuses=storage_pd_statuses,
            storage_pd_request_id=storage_pd_request_id,
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
    storage_pd_statuses: Optional[list[dict[str, Any]]] = None
    storage_pd_request_id: Optional[str] = None

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

        if skip_save and load_spec is None:
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
            token_ids = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids.tolist()

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
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
            storage_pd_statuses=tracker.storage_pd_statuses,
            storage_pd_request_id=tracker.storage_pd_request_id,
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


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
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
        self.config = config

        service_factory = VllmServiceFactory(config, vllm_config, role.name.lower())
        self._manager = LMCacheManager(config, service_factory, connector=self)

        # Start services managed by LMCacheManager
        self._manager.start_services()

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
        extra_config = config.extra_config or {}
        # A handoff configured through pd_data_path says the same thing as
        # the plugin-level switch, and it already names which side this node
        # is, so neither has to be repeated as a plugin setting.
        config_storage_pd = bool(getattr(config, "pd_uses_shared_storage", False))
        self._storage_pd_mode = (
            bool(extra_config.get("rust_raw_block.storage_pd_mode", False))
            or config_storage_pd
        )
        raw_role = str(extra_config.get("rust_raw_block.role", "") or "")
        if not raw_role and config_storage_pd:
            raw_role = "reader" if config.pd_role == "receiver" else "writer"
        self._storage_pd_raw_role = raw_role or "writer"
        self._storage_pd_store_futures: dict[str, Future] = {}
        self._storage_pd_wire_req_ids: dict[str, str] = {}
        self._storage_pd_engine_finished: set[str] = set()
        self._storage_pd_returned: OrderedDict[str, None] = OrderedDict()
        self._storage_pd_aborted: set[str] = set()
        self._storage_pd_failures: dict[str, str] = {}
        self._storage_pd_terminal_states: dict[str, StoragePDState] = {}
        self._storage_pd_receipts: dict[str, RawBlockPublicationReceipt] = {}
        # A status owed to a consumer lives here from the moment it is
        # owed until its announcement settles, so its deadline is stamped
        # once and survives a queue that has no room for it yet.
        self._storage_pd_obligations: dict[str, StoragePDObligation] = {}
        self._storage_pd_acks_sent: OrderedDict[str, None] = OrderedDict()
        # Acknowledgements this consumer owes a producer, kept until the
        # queue takes them. Independent of the decoder request's own
        # lifetime: the request is long gone by the time a saturated queue
        # has room.
        self._storage_pd_ack_outbox: OrderedDict[str, StoragePDObligation] = (
            OrderedDict()
        )
        self._storage_pd_lock = threading.Lock()
        self._storage_pd_tp_rank = 0
        self._storage_pd_status_sender: Optional[StoragePDStatusSender] = None
        self._storage_pd_notify_queue: Optional[StoragePDNotificationQueue] = None
        # Whether a consumer is waiting to be told about this producer's
        # publications. It does not depend on which role this instance
        # plays, so an instance that completes requests without a sender
        # can tell that it is misconfigured rather than assume nobody
        # needed to hear from it.
        self._storage_pd_notify_required = bool(
            self._storage_pd_mode
            and self._storage_pd_raw_role in ("writer", "reader")
            and not bool(config.pd_skip_proxy_notification)
        )
        if self._storage_pd_mode and role != KVConnectorRole.SCHEDULER:
            self._init_storage_pd_notification(config, extra_config)

        # Role-specific initialization
        if role == KVConnectorRole.SCHEDULER:
            self._unfinished_requests: dict[str, "Request"] = {}
        else:
            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending

            if self.enable_blending:
                assert self.lmcache_engine is not None
                assert self.lmcache_engine.gpu_connector is not None, (
                    "GPU connector must be available for blending"
                )
                self.blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
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

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))
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
            self._manager.post_init()

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
    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        self._manager.post_init()

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
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
            adopted_publication: tuple[StoragePDStatus, int] | None = None
            if request.storage_pd_statuses is not None:
                adopted_publication = self._adopt_storage_pd_publication(request)
            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
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
                    req_id=request.req_id,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s"
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!",
                        request.req_id,
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    """
                    Report failed block IDs in case of partial failure.
                    """
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping[:lmcache_cached_tokens],
                    )
                    self._invalid_block_ids.update(missing_blocks)
                elif adopted_publication is not None:
                    status, published_tokens = adopted_publication
                    resident_tokens = min(
                        request.load_spec.vllm_cached_tokens,
                        published_tokens,
                    )
                    restored_tokens = int(
                        ret_token_mask[:published_tokens].sum().item()
                    )
                    if lmcache_cached_tokens < published_tokens or (
                        resident_tokens + restored_tokens < published_tokens
                    ):
                        raise RuntimeError(
                            "storage P/D restore did not cover the advertised "
                            f"manifest for {request.req_id}: published_tokens="
                            f"{published_tokens}, lmcache_cached_tokens="
                            f"{lmcache_cached_tokens}, resident_tokens="
                            f"{resident_tokens}, restored_tokens={restored_tokens}"
                        )
                    self._ack_storage_pd_restore(request.req_id, status)

    def _adopt_storage_pd_publication(
        self, request: ReqMeta
    ) -> tuple[StoragePDStatus, int]:
        """Adopt this worker rank's advertised checkpoint before KV restore."""
        assert request.storage_pd_statuses is not None
        statuses = [
            msgspec.convert(raw_status, type=StoragePDStatus)
            for raw_status in request.storage_pd_statuses
        ]
        matching = [
            status for status in statuses if status.tp_rank == self._storage_pd_tp_rank
        ]
        if len(matching) != 1:
            raise RuntimeError(
                "storage P/D requires exactly one READY status for TP rank "
                f"{self._storage_pd_tp_rank}, got {len(matching)}"
            )
        status = matching[0]
        if (
            request.storage_pd_request_id is None
            or status.req_id != request.storage_pd_request_id
            or status.state != "READY"
        ):
            raise RuntimeError(
                "storage P/D status does not identify this READY request"
            )
        assert self.lmcache_engine is not None
        timeout_ms = int(
            (self.config.extra_config or {}).get(
                "rust_raw_block.publication_adopt_timeout_ms",
                5000,
            )
        )
        published_tokens = self.lmcache_engine.adopt_storage_publication(
            request.token_ids,
            status.publication_receipt(),
            request_configs=request.request_configs,
            timeout_ms=timeout_ms,
        )
        if published_tokens is None:
            raise RuntimeError(
                "storage P/D reader could not adopt the advertised request "
                f"manifest for {request.req_id} rank {self._storage_pd_tp_rank}"
            )
        return status, published_tokens

    def _ack_storage_pd_restore(
        self,
        req_id: str,
        status: StoragePDStatus,
    ) -> None:
        """Report that the advertised bytes reached this rank's GPU cache.

        ``req_id`` names the request inside this engine and is only used to
        avoid acknowledging it twice. What goes on the wire is the identity
        the producer published, which the adopted status carries.

        The acknowledgement is queued, not sent: this runs on the restore
        path, where waiting for a producer that may have exited would stall
        the load. Saturation keeps the obligation rather than dropping it,
        so the same acknowledgement is offered again without needing
        another restore.
        """
        if req_id in self._storage_pd_acks_sent:
            return
        if self._storage_pd_notify_queue is not None:
            ack: StoragePDReadAck = StoragePDReadAck(
                # The producer knows this request by the identity it
                # published and checked READY against, not by the name
                # this engine happens to give it. The two differ, and an
                # acknowledgement carrying the local name matches nothing
                # on the side that has to act on it.
                req_id=status.req_id,
                producer_instance_id=status.producer_instance_id,
                consumer_instance_id=STORAGE_PD_INCARNATION,
                tp_rank=self._storage_pd_tp_rank,
                writer_epoch=status.writer_epoch,
                checkpoint_seq=status.checkpoint_seq,
                manifest_digest=status.manifest_digest,
            )
            # Keyed apart from a producer status so a reader and a writer
            # sharing one connector cannot displace each other's message.
            # The obligation is created and kept here, so saturation costs a
            # later attempt rather than the acknowledgement itself, and the
            # retry does not need another restore to happen.
            obligation = self._storage_pd_notify_queue.obligation(
                f"ack:{req_id}", ack, report=False
            )
            self._storage_pd_ack_outbox[req_id] = obligation
        self._storage_pd_acks_sent.pop(req_id, None)
        self._storage_pd_acks_sent[req_id] = None
        while len(self._storage_pd_acks_sent) > STORAGE_PD_REQUEST_HISTORY:
            self._storage_pd_acks_sent.popitem(last=False)
        self._drain_storage_pd_ack_outbox()

    def _drain_storage_pd_ack_outbox(self) -> None:
        """Offer every acknowledgement still owed, oldest first.

        Called from the restore path, which is the only regular event a
        reader has. An obligation the queue refuses stays here with the
        deadline it was created under, so the next restore retries it; the
        outbox is what makes a lost acknowledgement this engine's problem
        rather than something that silently evaporated.
        """
        queue = self._storage_pd_notify_queue
        if queue is None:
            return
        owed = list(self._storage_pd_ack_outbox.items())
        for req_id, obligation in owed:
            if obligation.settled:
                self._storage_pd_ack_outbox.pop(req_id, None)
                continue
            queue.offer(obligation)
        if len(self._storage_pd_ack_outbox) > STORAGE_PD_REQUEST_HISTORY:
            # A reader that cannot reach its producer at all must not grow
            # without bound either. The oldest are dropped, loudly: their
            # extents stay protected on the writer, which is the safe
            # direction, and the writer's own accounting is what surfaces
            # them.
            while len(self._storage_pd_ack_outbox) > STORAGE_PD_REQUEST_HISTORY:
                dropped, _ = self._storage_pd_ack_outbox.popitem(last=False)
                logger.error(
                    "Raw-block storage P/D gave up acknowledging request "
                    "%s: %d acknowledgements are already owed and unsent",
                    dropped,
                    STORAGE_PD_REQUEST_HISTORY,
                )

    @staticmethod
    def _log_storage_pd_delivery(delivery: "StoragePDDelivery") -> None:
        """Report the fate of a message whose sender has no polling point."""
        if delivery.state == "ABANDONED":
            logger.error(
                "Raw-block storage P/D gave up sending %s: %s",
                delivery.key,
                delivery.detail,
            )

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
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.layerwise_retrievers:
            logger.debug("Waiting for layer %s to be loaded", self.current_layer)

        # Wait for the layer to be loaded
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)

            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info("Retrieved %s tokens", num_retrieved_tokens)

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

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        # Degraded mode (LMCache init failed): no engine to save to / unpin from.
        if self.lmcache_engine is None:
            return

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            # But still need to unpin the kv caches according to req_id
            # to balance the pin count from contains()
            assert self.lmcache_engine is not None, (
                "LMCacheEngine must be initialized to unpin requests."
            )
            for request in connector_metadata.requests:
                self.lmcache_engine.lookup_unpin(request.req_id)

            return

        if self.use_layerwise:
            for request in connector_metadata.requests:
                layerwise_storer = self._layerwise_save_storers.pop(
                    request.req_id, None
                )
                if layerwise_storer is not None:
                    next(layerwise_storer)
                # unpin the kv caches according to req_id
                self.lmcache_engine.lookup_unpin(request.req_id)
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        # Probe decoder cache before store if bidirectional mode is enabled
        bidir_enabled = getattr(self.config, "pd_bidirectional", False)

        for request in connector_metadata.requests:
            # unpin the kv caches according to req_id
            self.lmcache_engine.lookup_unpin(request.req_id)

            if self._storage_pd_mode and request.disagg_spec is not None:
                self._storage_pd_wire_req_ids[request.req_id] = (
                    request.disagg_spec.req_id
                )

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
                if self._storage_pd_mode:
                    self._record_storage_pd_failure(
                        request.req_id,
                        "slot_mapping/token_ids length mismatch",
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
                if self._storage_pd_mode:
                    if request.disagg_spec is None:
                        self._record_storage_pd_failure(
                            request.req_id,
                            "storage P/D request is missing its transfer spec",
                        )
                    else:
                        try:
                            completion = (
                                self.lmcache_engine.publish_existing_storage_request(
                                    token_ids,
                                    request.disagg_spec,
                                    request_configs=request.request_configs,
                                )
                            )
                        except Exception as exc:
                            self._record_storage_pd_failure(
                                request.req_id,
                                str(exc),
                            )
                        else:
                            with self._storage_pd_lock:
                                self._storage_pd_store_futures.setdefault(
                                    request.req_id,
                                    completion,
                                )
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

            try:
                completion = self.lmcache_engine.store(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    transfer_spec=request.disagg_spec,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )
            except Exception as exc:
                if not self._storage_pd_mode:
                    raise
                self._record_storage_pd_failure(request.req_id, str(exc))
                continue
            if self._storage_pd_mode:
                with self._storage_pd_lock:
                    if completion is None:
                        self._storage_pd_failures[request.req_id] = (
                            "store returned no request completion"
                        )
                        self._storage_pd_terminal_states[request.req_id] = "FAILED"
                        self._storage_pd_aborted.add(request.req_id)
                    else:
                        self._storage_pd_store_futures.setdefault(
                            request.req_id,
                            completion,
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
    def _init_storage_pd_notification(self, config, extra_config: dict) -> None:
        """Build this worker's status sender, or refuse the configuration.

        Only a worker reaches this. The scheduler builds no sender by design
        and completes no handoff, so requiring one there would reject a
        configuration that is correct.
        """
        engine_metadata = getattr(self.lmcache_engine, "metadata", None)
        self._storage_pd_tp_rank = int(
            getattr(engine_metadata, "worker_id", os.environ.get("LOCAL_RANK", 0))
        )
        skip_notification = bool(config.pd_skip_proxy_notification)
        if self._storage_pd_raw_role in ("writer", "reader") and not skip_notification:
            if config.pd_proxy_host is None or config.pd_proxy_port is None:
                raise ValueError(
                    "raw-block storage P/D requires pd_proxy_host and "
                    "pd_proxy_port, or pd_skip_proxy_notification=true"
                )
            self._storage_pd_status_sender = StoragePDStatusSender(
                config.pd_proxy_host,
                config.pd_proxy_port,
                timeout_s=float(
                    extra_config.get(
                        "rust_raw_block.status_send_timeout_s",
                        5.0,
                    )
                ),
            )
        if self._storage_pd_status_sender is not None:
            # A status is only worth what the consumer eventually hears, but
            # the engine step is the wrong place to wait for it. The queue
            # owns delivery from here: the completion path hands it a decided
            # status and asks later what became of it.
            self._storage_pd_notify_queue = StoragePDNotificationQueue(
                self._storage_pd_status_sender,
                capacity=int(
                    extra_config.get("rust_raw_block.status_queue_capacity", 1024)
                ),
                retry_interval_s=float(
                    extra_config.get("rust_raw_block.status_retry_interval_s", 0.5)
                ),
                deadline_s=float(
                    extra_config.get("rust_raw_block.status_deadline_s", 60.0)
                ),
                on_unreported=self._log_storage_pd_delivery,
            )
        if self._storage_pd_notify_required and self._storage_pd_status_sender is None:
            # A worker that owes a consumer a status and has no way to send
            # one cannot serve this configuration. Say so now, rather than on
            # the first request that finishes.
            raise ValueError(
                "raw-block storage P/D requires a status sender on a worker: "
                "set pd_proxy_host and pd_proxy_port, or "
                "pd_skip_proxy_notification=true to run without a consumer"
            )

    def _storage_pd_retire_locked(self, req_id: str) -> None:
        """Drop a finished request's state, keeping only that it finished.

        Every container here holds something a request needs while it is in
        flight and nothing anyone reads afterwards, so keeping them costs
        one entry per request served for the life of the engine. Dropping
        the identifier as well would let a late status be taken for a new
        request, so that one identifier is kept, and the record of them is
        bounded.
        """
        self._storage_pd_store_futures.pop(req_id, None)
        self._storage_pd_wire_req_ids.pop(req_id, None)
        self._storage_pd_engine_finished.discard(req_id)
        self._storage_pd_aborted.discard(req_id)
        self._storage_pd_failures.pop(req_id, None)
        self._storage_pd_terminal_states.pop(req_id, None)
        self._storage_pd_receipts.pop(req_id, None)
        self._storage_pd_obligations.pop(req_id, None)
        self._storage_pd_returned.pop(req_id, None)
        self._storage_pd_returned[req_id] = None
        while len(self._storage_pd_returned) > STORAGE_PD_REQUEST_HISTORY:
            self._storage_pd_returned.popitem(last=False)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if (
            not self._storage_pd_mode
            or self._storage_pd_raw_role != "writer"
            or self._role == KVConnectorRole.SCHEDULER
        ):
            # The scheduler builds no status sender by design, so it must not
            # run the writer's completion path either: doing so would either
            # demand a sender it should not have or record deliveries that
            # never happened.
            return None, None

        releasable: set[str] = set()
        pending_sends: list[tuple[str, StoragePDStatus]] = []
        with self._storage_pd_lock:
            if self.lmcache_engine is not None:
                storage_manager = self.lmcache_engine.storage_manager
                if storage_manager is not None:
                    for req_id in finished_req_ids:
                        storage_manager.finish_request(
                            self._storage_pd_wire_req_ids.get(req_id, req_id)
                        )
            self._storage_pd_engine_finished.update(finished_req_ids)
            for req_id in list(self._storage_pd_engine_finished):
                if req_id in self._storage_pd_returned:
                    self._storage_pd_engine_finished.discard(req_id)
                    continue
                completion = self._storage_pd_store_futures.get(req_id)
                # A recorded receipt is proof the publication happened. The
                # future is dropped once its result is taken, so a request
                # whose status still has to go out reaches here with neither
                # a future nor a failure, and only the receipt distinguishes
                # it from one that never published at all.
                if (
                    completion is None
                    and req_id not in self._storage_pd_receipts
                    and req_id not in self._storage_pd_aborted
                ):
                    self._storage_pd_failures[req_id] = (
                        "request finished without a raw-block publication"
                    )
                    self._storage_pd_terminal_states[req_id] = "FAILED"
                    self._storage_pd_aborted.add(req_id)
                if completion is not None and not completion.done():
                    continue
                if completion is not None:
                    try:
                        results = completion.result()
                        if isinstance(results, RawBlockPublicationReceipt):
                            results = [results]
                        receipts = [
                            result
                            for result in results
                            if isinstance(result, RawBlockPublicationReceipt)
                        ]
                        if len(receipts) != 1:
                            raise RuntimeError(
                                "storage P/D expected exactly one raw-block receipt"
                            )
                        self._storage_pd_receipts[req_id] = receipts[0]
                    except BaseException as exc:
                        self._storage_pd_failures[req_id] = str(exc)
                        self._storage_pd_terminal_states[req_id] = (
                            "CANCELLED" if isinstance(exc, CancelledError) else "FAILED"
                        )
                        self._storage_pd_aborted.add(req_id)
                        logger.error(
                            "Raw-block storage P/D store failed for request %s: %s",
                            req_id,
                            exc,
                        )
                    self._storage_pd_store_futures.pop(req_id, None)

                if req_id not in self._storage_pd_obligations:
                    wire_req_id = self._storage_pd_wire_req_ids.get(req_id, req_id)
                    if req_id in self._storage_pd_aborted:
                        status = StoragePDStatus(
                            req_id=wire_req_id,
                            producer_instance_id=STORAGE_PD_INCARNATION,
                            tp_rank=self._storage_pd_tp_rank,
                            state=self._storage_pd_terminal_states.get(
                                req_id, "FAILED"
                            ),
                            error_stage="WRITE_OR_PUBLISH",
                            error_text=self._storage_pd_failures.get(
                                req_id, "storage P/D request failed"
                            ),
                        )
                    else:
                        receipt = self._storage_pd_receipts.get(req_id)
                        if receipt is None:
                            continue
                        status = StoragePDStatus.ready(
                            wire_req_id,
                            self._storage_pd_tp_rank,
                            receipt,
                        )
                    # Hand the status to the queue below rather than reach
                    # the peer from here: a consumer that has stopped reading
                    # can block this socket for as long as it likes, and every
                    # other user of this state would wait behind it.
                    pending_sends.append((req_id, status))
                # A request already handed to the queue is not yet announced.
                # Releasing it here would tell the engine the handoff is done
                # while the consumer has heard nothing; only a settled
                # delivery below may do that.

        if pending_sends and self._storage_pd_notify_queue is None:
            # Releasing these would tell the engine the handoff is done
            # while the consumer is still waiting to hear that anything
            # was published. Nothing here can reach it, so say so instead
            # of recording a delivery that never happened.
            if self._storage_pd_notify_required:
                raise RuntimeError(
                    "raw-block storage P/D has statuses to send and no "
                    "status sender; set pd_proxy_host and pd_proxy_port, "
                    "or pd_skip_proxy_notification=true to run without a "
                    "consumer"
                )
            # Configured to run without a consumer: there is nobody to tell,
            # so the request is done as soon as its bytes are durable.
            with self._storage_pd_lock:
                for req_id, _ in pending_sends:
                    releasable.add(req_id)
                    self._storage_pd_retire_locked(req_id)
            return releasable, None

        queue = self._storage_pd_notify_queue
        if queue is not None:
            with self._storage_pd_lock:
                for req_id, status in pending_sends:
                    # The obligation, and with it the deadline, is created
                    # here: the moment the status became owed. Waiting for
                    # capacity below spends that deadline rather than
                    # postponing it.
                    self._storage_pd_obligations[req_id] = queue.obligation(
                        req_id, status
                    )
                outstanding = list(self._storage_pd_obligations.values())
            for obligation in outstanding:
                # A refusal needs no handling: the obligation is still ours,
                # and offering the same object on a later step is exactly
                # what its unrefreshed deadline is for.
                queue.offer(obligation)
            for delivery in queue.poll():
                if delivery.state == "ABANDONED":
                    # The bytes are durable and the publication stands; what
                    # failed is telling anyone. Holding the engine's request
                    # open would not change that, so it is released and the
                    # loss is recorded loudly. Releasing the request is not a
                    # handoff and reclaims nothing: the extent lease is the
                    # writer's, it stays held, and only an acknowledgement
                    # the writer validates itself can release it.
                    logger.error(
                        "Raw-block storage P/D gave up announcing request %s: %s",
                        delivery.key,
                        delivery.detail,
                    )
                with self._storage_pd_lock:
                    releasable.add(delivery.key)
                    self._storage_pd_retire_locked(delivery.key)
        return releasable, None

    def _record_storage_pd_failure(self, req_id: str, error: str) -> None:
        """Record a terminal producer failure for proxy notification."""
        with self._storage_pd_lock:
            self._storage_pd_failures[req_id] = error
            self._storage_pd_terminal_states[req_id] = "FAILED"
            self._storage_pd_aborted.add(req_id)

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
        return invalid_blocks

    @_lmcache_nvtx_annotate
    def shutdown(self):
        """Shutdown the connector by delegating to LMCacheManager."""
        logger.info("Starting LMCacheConnector shutdown...")
        try:
            self._manager.stop_services()
        finally:
            # The queue owns the sender, including closing it, so shutdown
            # has one budget and one owner. Closing the sender from here as
            # well would race its worker and could block past that budget.
            if self._storage_pd_notify_queue is not None:
                self._storage_pd_notify_queue.close()
            elif self._storage_pd_status_sender is not None:
                self._storage_pd_status_sender.close()

    ###################
    # Scheduler side APIs
    ####################

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

        if (
            num_external_hit_tokens := self.lookup_client.lookup_cache(lookup_id=req_id)
        ) != -1:
            # -1 means no result cached
            # None or int means ongoing (async) or cached result
            logger.debug(
                "Found %s hit tokens for request %s in the lookup cache.",
                num_external_hit_tokens,
                req_id,
            )
        else:
            logger.debug("Looking up cache for the first time for request %s!", req_id)
            self._requests_priority[req_id] = getattr(request, "priority", 0)

            # token_ids = request.prompt_token_ids
            # all token ids covers the preemption case
            token_ids = request.all_token_ids

            # If the request has multimodal hashes, apply them to the token ids
            mm_hashes, mm_positions = extract_mm_features(request)
            if mm_hashes and mm_positions:
                # TODO(Jiayi): Optimize this
                token_ids = torch.tensor(request.prompt_token_ids)
                apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
                token_ids = token_ids.tolist()

            request_configs = extract_request_configs(request.sampling_params)
            if self.skip_last_n_tokens > 0:
                token_ids = token_ids[: -self.skip_last_n_tokens]

            num_external_hit_tokens = self.lookup_client.lookup(
                token_ids,
                lookup_id=req_id,
                request_configs=request_configs,
            )

        if num_external_hit_tokens is None:
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

        # Degraded mode (LMCache init failed): there is no lookup client to clear
        # and no load spec was recorded, so nothing to do.
        if self.lookup_client is None:
            return

        # Clear local status in lookup client when a new request is
        # successfully scheduled.
        self.lookup_client.clear_lookup_status(request.request_id)

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
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
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

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
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
        for request in scheduler_output.scheduled_new_reqs:
            # Ignore DP attention mock requests
            if request.req_id.startswith("mock_req"):
                continue
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
            )
            self._request_trackers[request.req_id] = request_tracker

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
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
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

            if self.async_loading:
                # Cancel any ongoing async lookup and prefetch tasks on
                # workers. Independent of ``lmcache_engine`` because the
                # Scheduler owns ``lookup_client`` even when it does not
                # build an engine.
                lookup_id = request.request_id
                if self.lookup_client is None:
                    logger.warning(
                        "Skipping abort-time async lookup cancel for "
                        "request %s: lookup_client is not initialized "
                        "while async_loading is enabled. Engine stays "
                        "alive; this request's lookup is dropped.",
                        request.request_id,
                    )
                else:
                    self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[attr-defined]

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

        defer_free = (
            self._storage_pd_mode
            and self._storage_pd_raw_role == "writer"
            and request.status != RequestStatus.FINISHED_ABORTED
        )
        return defer_free, return_params

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.lmcache_engine is not None:
            return self.lmcache_engine.get_kv_events()
        return []
