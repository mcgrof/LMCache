# SPDX-License-Identifier: Apache-2.0
"""Return acknowledgement capacity when a claimed restore fails."""

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
    _extract_storage_pd_request,
)
from lmcache.v1.storage_backend.raw_block import RawBlockPublicationReceipt
from lmcache.v1.storage_backend.storage_pd_ack import StoragePDAckClient
from lmcache.v1.storage_backend.storage_pd_protocol import StoragePDStatus


def test_storage_pd_receipt_survives_vllm_extra_args_only_shape() -> None:
    """Older NewRequestData omits the direct kv_transfer_params attribute."""
    statuses = [{"state": "READY", "req_id": "published-request"}]
    request_configs = {
        "lmcache.storage_pd_request_id": "published-request",
        "lmcache.storage_pd_statuses": statuses,
    }

    extracted_statuses, request_id = _extract_storage_pd_request(
        None,
        request_configs,
    )

    assert extracted_statuses is statuses
    assert request_id == "published-request"


def test_direct_storage_pd_params_override_extra_args_copy() -> None:
    direct_statuses = [{"state": "READY", "req_id": "direct"}]
    extracted_statuses, request_id = _extract_storage_pd_request(
        {
            "lmcache.storage_pd_request_id": "direct",
            "lmcache.storage_pd_statuses": direct_statuses,
        },
        {
            "lmcache.storage_pd_request_id": "copy",
            "lmcache.storage_pd_statuses": [{"state": "READY", "req_id": "copy"}],
        },
    )

    assert extracted_statuses is direct_statuses
    assert request_id == "direct"


@pytest.mark.parametrize(
    ("min_retrieve_tokens", "max_tokens_per_load"),
    [(0, 0), (16, 0), (0, 4)],
)
def test_storage_pd_miss_is_scheduled_for_receipt_adoption(
    min_retrieve_tokens: int, max_tokens_per_load: int
) -> None:
    """A READY miss must fail adoption, not recompute and orphan its hold."""
    status = StoragePDStatus.ready(
        "published-request",
        0,
        RawBlockPublicationReceipt(
            "writer-epoch", 1, 1, "digest", ack_endpoint="127.0.0.1:5999"
        ),
    )

    class Lookup:
        cached = -1

        @staticmethod
        def lookup_cache(*, lookup_id: str) -> int:
            assert lookup_id == "local-request"
            return Lookup.cached

        @staticmethod
        def lookup(tokens, *, lookup_id: str, request_configs: Any) -> int:
            assert tokens == list(range(7))
            assert lookup_id == "local-request"
            Lookup.cached = 0
            return 0

    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.kv_role = "kv_consumer"
    connector._manager = SimpleNamespace(lookup_client=Lookup())  # type: ignore[assignment]
    connector._requests_priority = {}
    connector.skip_last_n_tokens = 1
    connector.config = SimpleNamespace(min_retrieve_tokens=min_retrieve_tokens)
    connector._max_tokens_per_load = max_tokens_per_load
    connector._lmcache_chunk_size = 256
    connector.load_specs = {}
    request = SimpleNamespace(
        request_id="local-request",
        num_tokens=8,
        all_token_ids=list(range(8)),
        prompt_token_ids=list(range(8)),
        sampling_params=SamplingParams(),
        priority=0,
        mm_features=[],
        kv_transfer_params={
            "lmcache.storage_pd_request_id": status.req_id,
            "lmcache.storage_pd_statuses": [msgspec.to_builtins(status)],
        },
    )

    if max_tokens_per_load:
        with pytest.raises(RuntimeError, match="complete publication"):
            connector.get_num_new_matched_tokens(request, 0)
        assert not connector.load_specs
        return

    matched = connector.get_num_new_matched_tokens(request, 0)
    matched_from_cached_miss = connector.get_num_new_matched_tokens(request, 0)

    assert matched == 7
    assert matched_from_cached_miss == 7
    spec = connector.load_specs[request.request_id]
    assert spec.lmcache_cached_tokens == 7
    assert not spec.can_load


@pytest.mark.parametrize(
    "failure",
    [
        "adoption_raises",
        "adoption_missing",
        "retrieve_raises",
        "partial",
        "coverage",
        "overlapping_prefix",
        "ordinary_partial",
    ],
)
def test_failed_restore_returns_its_acknowledgement_capacity(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep failed reads from permanently filling the consumer's outbox."""
    client = StoragePDAckClient(max_live_obligations=1)
    claims: list[str] = []

    def claim(ack: Any, **kwargs: Any) -> bool:
        claims.append(ack.req_id)
        return True

    monkeypatch.setattr(client, "claim", claim)
    status = StoragePDStatus.ready(
        "published-request",
        0,
        RawBlockPublicationReceipt(
            "writer-epoch", 1, 1, "digest", ack_endpoint="127.0.0.1:5999"
        ),
    )
    request = SimpleNamespace(
        req_id="local-request",
        token_ids=list(range(8)),
        slot_mapping=torch.arange(8),
        load_spec=LoadSpec(3 if failure == "overlapping_prefix" else 0, 8, True),
        request_configs=None,
        storage_pd_request_id=status.req_id,
        storage_pd_statuses=(
            None if failure == "ordinary_partial" else [msgspec.to_builtins(status)]
        ),
    )

    class Engine:
        def adopt_storage_publication(self, *args: Any, **kwargs: Any) -> int | None:
            if failure == "adoption_raises":
                raise OSError("publication read failed")
            if failure == "adoption_missing":
                return None
            return 16 if failure == "coverage" else 8

        def retrieve(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            read_context = kwargs["storage_pd_read_context"]
            if failure == "ordinary_partial":
                assert read_context is None
            else:
                assert read_context.request_id == status.req_id
                assert read_context.receipt == status.publication_receipt()
            if failure == "retrieve_raises":
                raise OSError("payload read failed")
            mask = torch.ones(8, dtype=torch.bool)
            if failure in ("partial", "ordinary_partial"):
                mask[-1] = False
            if failure == "overlapping_prefix":
                # Five restored tokens plus three resident tokens is eight,
                # but positions 0..2 overlap and positions 5..7 are missing.
                mask[5:] = False
            return mask

    metadata = LMCacheConnectorMetadata(requests=[request])  # type: ignore[list-item]
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
    connector.async_loading = True
    connector._lmcache_chunk_size = 8
    connector._invalid_block_ids = set()
    connector._storage_pd_tp_rank = 0
    connector._storage_pd_lock = threading.Lock()
    connector._storage_pd_claims = OrderedDict()
    connector._storage_pd_acks_sent = OrderedDict()
    connector._storage_pd_ack_client = client
    connector._storage_pd_session_id = "session"
    connector._storage_pd_ack_deadline_s = 1.0
    connector.config = SimpleNamespace(extra_config={})
    fallback_requests: list[str] = []

    def record_failed_blocks(req_id: str, *args: Any) -> set[int]:
        fallback_requests.append(req_id)
        return {1}

    monkeypatch.setattr(connector, "record_failed_blocks", record_failed_blocks)
    try:
        context = SimpleNamespace(attn_metadata=object())
        if failure == "ordinary_partial":
            connector.start_load_kv(context)
            assert fallback_requests == [request.req_id]
            assert connector._invalid_block_ids == {1}
        else:
            with pytest.raises((OSError, RuntimeError)):
                connector.start_load_kv(context)
            assert not fallback_requests

        assert claims == ([] if failure == "ordinary_partial" else [status.req_id])
        assert client.live_count() == 0, "an unfinished restore must not acknowledge"
        assert client.reserved_count() == 0
        assert not connector._storage_pd_claims
        next_read = client.reserve(endpoint=status.ack_endpoint)
        assert next_read is not None
        next_read.release()
    finally:
        client.close()
