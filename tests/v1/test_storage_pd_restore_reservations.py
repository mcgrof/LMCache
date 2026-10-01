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

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
)
from lmcache.v1.storage_backend.raw_block import RawBlockPublicationReceipt
from lmcache.v1.storage_backend.storage_pd_ack import StoragePDAckClient
from lmcache.v1.storage_backend.storage_pd_protocol import StoragePDStatus


@pytest.mark.parametrize(
    "failure",
    ["adoption_raises", "adoption_missing", "retrieve_raises", "partial", "coverage"],
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
        load_spec=LoadSpec(0, 8, True),
        request_configs=None,
        storage_pd_request_id=status.req_id,
        storage_pd_statuses=[msgspec.to_builtins(status)],
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
            assert read_context.request_id == status.req_id
            assert read_context.receipt == status.publication_receipt()
            if failure == "retrieve_raises":
                raise OSError("payload read failed")
            mask = torch.ones(8, dtype=torch.bool)
            if failure == "partial":
                mask[-1] = False
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
    connector._lmcache_chunk_size = 8
    connector._invalid_block_ids = set()
    connector._storage_pd_tp_rank = 0
    connector._storage_pd_lock = threading.Lock()
    connector._storage_pd_claims = OrderedDict()
    connector._storage_pd_acks_sent = OrderedDict()
    connector._storage_pd_ack_client = client
    connector._storage_pd_session_id = "session"
    connector.config = SimpleNamespace(extra_config={})
    monkeypatch.setattr(connector, "record_failed_blocks", lambda *args: set())
    try:
        context = SimpleNamespace(attn_metadata=object())
        if failure == "partial":
            connector.start_load_kv(context)
        else:
            with pytest.raises((OSError, RuntimeError)):
                connector.start_load_kv(context)

        assert claims == [status.req_id]
        assert client.live_count() == 0, "an unfinished restore must not acknowledge"
        assert client.reserved_count() == 0
        assert not connector._storage_pd_claims
        next_read = client.reserve(endpoint=status.ack_endpoint)
        assert next_read is not None
        next_read.release()
    finally:
        client.close()
