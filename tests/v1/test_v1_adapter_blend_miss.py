# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
)


class _FakeParent:
    def __init__(self, metadata: LMCacheConnectorMetadata) -> None:
        self.metadata = metadata

    def _get_connector_metadata(self) -> LMCacheConnectorMetadata:
        return self.metadata


class _FakeStatsMonitor:
    def update_interval_vllm_hit_tokens(self, count: int) -> None:
        del count

    def update_interval_prompt_tokens(self, count: int) -> None:
        del count


class _FakeBlender:
    def __init__(self, result: torch.Tensor) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def blend(
        self,
        tokens: list[int],
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.calls.append({"tokens": tokens, "mask": mask, **kwargs})
        return self.result


@pytest.mark.parametrize(
    ("result", "expected_invalid_blocks"),
    [
        pytest.param(torch.zeros(4, dtype=torch.bool), {0, 1}, id="miss"),
        pytest.param(torch.ones(4, dtype=torch.bool), set(), id="complete-hit"),
    ],
)
def test_layerwise_blend_load_records_only_failed_vllm_blocks(
    result: torch.Tensor,
    expected_invalid_blocks: set[int],
) -> None:
    """Blend misses invalidate exact blocks; complete hits invalidate none."""
    request = ReqMeta(
        req_id="blend-miss",
        token_ids=[10, 11, 12, 13],
        slot_mapping=torch.tensor([0, 1, 2, 3]),
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=4,
            can_load=True,
        ),
    )
    metadata = LMCacheConnectorMetadata(requests=[request])
    blender = _FakeBlender(result)
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)  # type: ignore[assignment]
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=object()
    )
    connector._stats_monitor = _FakeStatsMonitor()  # type: ignore[assignment]
    connector.device = "cpu"
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector.use_layerwise = True
    connector.enable_blending = True
    connector.blender = blender  # type: ignore[assignment]
    connector._lmcache_chunk_size = 2
    connector._block_size = 2
    connector._invalid_block_ids = set()

    connector.start_load_kv(SimpleNamespace(attn_metadata=object()))

    assert len(blender.calls) == 1
    assert blender.calls[0]["req_id"] == "blend-miss"
    assert connector._invalid_block_ids == expected_invalid_blocks
