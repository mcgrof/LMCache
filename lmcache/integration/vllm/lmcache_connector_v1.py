# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRequestSidecar,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
import torch

# First Party
from lmcache.integration.vllm.connector_construction import (
    RetainedConnectorConstructionOwner,
    cleanup_failed_connector_construction,
)
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheConnectorV1Dynamic(KVConnectorBase_V1):
    KMESH_HANDOFF_PROTOCOL_VERSION = 2

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional[Any] = None,
    ):
        if kv_cache_config is not None:
            super().__init__(
                vllm_config=vllm_config,
                role=role,
                kv_cache_config=kv_cache_config,
            )
        else:
            super().__init__(vllm_config=vllm_config, role=role)
        owner = RetainedConnectorConstructionOwner(
            lambda publish: LMCacheConnectorV1Impl(
                vllm_config,
                role,
                self,
                publish,
            ),
            thread_name=("lmcache-vllm-wrapper-construction-" + role.name.lower()),
        )
        self._lmcache_engine_construction_owner: Optional[
            RetainedConnectorConstructionOwner[LMCacheConnectorV1Impl]
        ] = owner
        try:
            owner.invoke()
            connector = owner.connector
            self._lmcache_engine = connector
            owner.release_provisional_manager()
            self._lmcache_engine_construction_owner = None
        except BaseException as init_error:
            cleanup_failed_connector_construction(owner, init_error)
            raise

    # ==============================
    # Worker-side methods
    # ==============================
    def register_model(self, model: torch.nn.Module) -> None:
        """Register the worker's loaded model for CacheBlend sessions.

        Args:
            model: Loaded vLLM model owned by this worker.
        """
        self._lmcache_engine.register_model(model)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        self._lmcache_engine.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation
        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self._lmcache_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self._lmcache_engine.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self._lmcache_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return self._lmcache_engine.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return block IDs that failed to load during the last interval."""
        return self._lmcache_engine.get_block_ids_with_load_errors()

    def shutdown(self) -> None:
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        self._lmcache_engine.shutdown()

    # ==============================
    # Scheduler-side methods
    # ==============================
    def prepare_request_lookup(
        self,
        request: "Request",
        peer_sidecars: tuple[KVConnectorRequestSidecar, ...],
    ) -> None:
        """Forward composite request evidence to the LMCache adapter."""
        self._lmcache_engine.prepare_request_lookup(request, peer_sidecars)

    def bind_peer_connector_metadata(
        self,
        peer_connector_metadata: tuple[KVConnectorMetadata, ...],
    ) -> None:
        """Forward the complete composite worker-metadata tuple."""
        self._lmcache_engine.bind_peer_connector_metadata(peer_connector_metadata)

    def validate_peer_connector_metadata(
        self,
        peer_connector_metadata: tuple[KVConnectorMetadata, ...],
    ) -> None:
        """Validate composite metadata without changing adapter state."""
        self._lmcache_engine.validate_peer_connector_metadata(peer_connector_metadata)

    def begin_connector_metadata_transaction(self) -> None:
        """Begin a retry-safe composite metadata transaction."""
        self._lmcache_engine.begin_connector_metadata_transaction()

    def commit_connector_metadata_transaction(self) -> None:
        """Tentatively commit the pending composite metadata generation."""
        self._lmcache_engine.commit_connector_metadata_transaction()

    def finalize_connector_metadata_transaction(self) -> None:
        """Finalize metadata after every composite child tentatively commits."""
        self._lmcache_engine.finalize_connector_metadata_transaction()

    def abort_connector_metadata_transaction(self) -> None:
        """Restore pre-begin metadata state after any transaction failure."""
        self._lmcache_engine.abort_connector_metadata_transaction()

    def clear_peer_connector_metadata(self) -> None:
        """Clear request-independent peer metadata after one worker step."""
        self._lmcache_engine.clear_peer_connector_metadata()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        return self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        ), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self._lmcache_engine.update_state_after_alloc(request, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self._lmcache_engine.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return self._lmcache_engine.request_finished(request, block_ids)
