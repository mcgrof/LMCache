# SPDX-License-Identifier: Apache-2.0
"""Restore precisely the advertised KV, including already-resident prefixes."""

# Standard
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any
import threading

# Third Party
import msgspec
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm import SamplingParams

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
)
from lmcache.v1.storage_backend.raw_block import RawBlockPublicationReceipt
from lmcache.v1.storage_backend.storage_pd_protocol import StoragePDStatus


@pytest.mark.parametrize("resident_tokens", [0, 3, 7, 8])
def test_restore_omits_unpublished_continuation_and_acknowledges_coverage(
    resident_tokens: int,
) -> None:
    """The first output token must not alter the published partial-tail key."""
    status = StoragePDStatus.ready(
        "producer-request",
        0,
        RawBlockPublicationReceipt(
            "writer", 1, 1, "digest", ack_endpoint="127.0.0.1:5999"
        ),
    )
    retrieved: list[list[int]] = []
    acknowledged: list[str] = []
    unpinned: list[str] = []

    class Engine:
        def adopt_storage_publication(
            self, tokens: list[int], *args: Any, **kwargs: Any
        ) -> int:
            assert tokens == list(range(9))
            return 8

        def retrieve(
            self, tokens: list[int], mask: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            retrieved.append(tokens)
            assert len(kwargs["slot_mapping"]) == 8
            return mask.clone()

        def lookup_unpin(self, request_id: str) -> None:
            unpinned.append(request_id)

    class Client:
        def reserve(self, **kwargs: Any) -> Any:
            return SimpleNamespace(release=lambda: None)

        def claim(self, *args: Any, **kwargs: Any) -> bool:
            return True

        def owe(self, ack: Any, **kwargs: Any) -> object:
            acknowledged.append(ack.req_id)
            return object()

    request = ReqMeta(
        req_id="decoder-request",
        token_ids=list(range(9)),
        slot_mapping=torch.arange(9),
        load_spec=LoadSpec(resident_tokens, 9, True),
        storage_pd_request_id=status.req_id,
        storage_pd_statuses=[msgspec.to_builtins(status)],
    )
    metadata = LMCacheConnectorMetadata(requests=[request])
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._manager = SimpleNamespace(lmcache_engine=Engine())  # type: ignore[assignment]
    connector._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    connector._stats_monitor = SimpleNamespace(  # type: ignore[assignment]
        update_interval_vllm_hit_tokens=lambda _count: None,
        update_interval_prompt_tokens=lambda _count: None,
    )
    connector.kv_caches = {"layer": torch.zeros(1)}
    connector.device = "cpu"
    connector.use_layerwise = False
    connector.async_loading = False
    connector._lmcache_chunk_size = 4
    connector._storage_pd_tp_rank = 0
    connector._storage_pd_lock = threading.Lock()
    connector._storage_pd_claims = OrderedDict()
    connector._storage_pd_acks_sent = OrderedDict()
    connector._storage_pd_ack_client = Client()  # type: ignore[assignment]
    connector._storage_pd_session_id = "session"
    connector._storage_pd_ack_deadline_s = 1.0
    connector.config = SimpleNamespace(extra_config={})

    connector.start_load_kv(SimpleNamespace(attn_metadata=object()))

    assert retrieved == [list(range(8))]
    assert acknowledged == [status.req_id]
    assert unpinned == [request.req_id]


@pytest.mark.parametrize("has_publication", [False, True])
def test_local_prefix_hit_still_schedules_publication_settlement(
    has_publication: bool,
) -> None:
    """Zero external blocks does not discharge a READY publication's hold."""
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    cleared: list[str] = []
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lookup_client=SimpleNamespace(clear_lookup_status=cleared.append)
    )
    connector.load_specs = {"decoder-request": LoadSpec(8, 8, False)}
    connector._unfinished_requests = {}
    params = (
        {
            "lmcache.storage_pd_request_id": "producer-request",
            "lmcache.storage_pd_statuses": [{"state": "READY"}],
        }
        if has_publication
        else None
    )
    request = SimpleNamespace(
        request_id="decoder-request",
        num_tokens=9,
        kv_transfer_params=params,
        sampling_params=SamplingParams(),
    )

    connector.update_state_after_alloc(request, num_external_tokens=0)

    assert cleared == [request.request_id]
    assert connector.load_specs[request.request_id].can_load is has_publication
