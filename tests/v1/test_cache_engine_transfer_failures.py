# SPDX-License-Identifier: Apache-2.0
"""Known transfer failures must release engine refs on reused real pages."""

# Standard
from contextlib import nullcontext
from typing import Any
from unittest.mock import MagicMock
import gc

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat, retain_memory_owners
import lmcache.v1.memory_management as memory


def _pool() -> PagedTensorMemoryAllocator:
    return PagedTensorMemoryAllocator(
        torch.zeros(4096, dtype=torch.uint8),
        shapes=[torch.Size([4096])],
        dtypes=[torch.uint8],
        fmt=MemoryFormat.KV_2LTD,
    )


def _engine(pool: PagedTensorMemoryAllocator) -> Any:
    engine: Any = LMCacheEngine.__new__(LMCacheEngine)
    engine.is_healthy = lambda: True
    engine.is_frozen = lambda: False
    engine._is_passive = lambda: False
    engine._get_req_id = lambda kwargs: "request"
    engine._log_kvcache_for_check = lambda **kwargs: None
    engine.config = MagicMock()
    engine.config.get_extra_config_value.return_value = False
    engine.metadata = MagicMock()
    engine.metadata.get_shapes.return_value = pool.shapes
    engine.metadata.get_dtypes.return_value = pool.dtypes
    key = CacheEngineKey("model", 1, 0, 1, torch.uint8)
    engine.token_database = MagicMock()
    engine.token_database.process_tokens.return_value = [(0, 4, key)]
    engine.fmt = MemoryFormat.KV_2LTD
    engine.kv_events_enabled = False
    engine.store_location = None
    engine.async_loading = False
    engine.save_only_first_rank = False
    engine.remove_after_retrieve = False
    stats = MagicMock()
    for name in [
        "profile_process_tokens",
        "profile_from_gpu",
        "profile_put",
        "profile_to_gpu",
    ]:
        getattr(stats, name).side_effect = nullcontext
    stats.time_to_store.return_value = stats.time_to_retrieve.return_value = 1.0
    stats.process_tokens_time = stats.from_gpu_time = stats.put_time = 0.0
    engine.stats_monitor = MagicMock()
    engine.stats_monitor.on_store_request.return_value = stats
    engine.stats_monitor.on_retrieve_request.return_value = stats
    engine.storage_manager = MagicMock()
    engine.storage_manager.allocate.side_effect = (
        lambda shapes, dtypes, **kwargs: pool.allocate(shapes, dtypes, engine.fmt)
    )
    engine.gpu_connector = MagicMock()
    return engine


@pytest.mark.no_shared_allocator
def test_repeated_known_gather_failure_releases_the_same_page() -> None:
    pool = _pool()
    engine = _engine(pool)
    seen: list[int] = []

    def fail(objects, *args, **kwargs):
        seen.append(id(objects[0]))
        raise RuntimeError("known drained gather failure")

    engine.gpu_connector.batched_from_gpu.side_effect = fail
    for _ in range(3):
        with pytest.raises(RuntimeError, match="known drained"):
            engine.store(tokens=[1, 2, 3, 4])
        gc.collect()
        assert len(pool.free_blocks) == 1
        assert pool.num_active_allocations == 0
    assert len(set(seen)) == 1
    engine.storage_manager.batched_put.assert_not_called()


@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("unknown", [False, True])
def test_scatter_failure_releases_get_ref_and_preserves_lookup_pin(
    monkeypatch, unknown
) -> None:
    pool = _pool()
    engine = _engine(pool)
    obj = pool.allocate(pool.shapes, pool.dtypes)
    assert obj is not None
    # Represent the independent lookup owner's pin without starting its monitor.
    obj.metadata.pin_count = 1
    monkeypatch.setattr(memory, "_UNCERTAIN_MEMORY_OWNERS", [])

    def get(tokens, mask, ret_mask, **kwargs):
        obj.ref_count_up()
        ret_mask[:] = True
        return [("key", obj, 0, 4)], obj.get_size()

    def fail(objects, *args, **kwargs):
        if unknown:
            retain_memory_owners(objects, engine.gpu_connector)
        raise RuntimeError("scatter failure")

    engine._process_tokens_internal = get
    engine.gpu_connector.batched_to_gpu.side_effect = fail
    try:
        for attempt in range(2):
            with pytest.raises(RuntimeError, match="scatter failure"):
                engine.retrieve([1, 2, 3, 4])
            assert obj.get_ref_count() == 1 + (attempt + 1 if unknown else 0)
            assert obj.metadata.pin_count == 1
            assert len(pool.free_blocks) == 0
    finally:
        # These transfers are CPU seams; teardown balances only this test's holds.
        for objects, _ in memory._UNCERTAIN_MEMORY_OWNERS:
            for held in objects:
                held.ref_count_down()
        memory._UNCERTAIN_MEMORY_OWNERS.clear()
        obj.metadata.pin_count = 0
        obj.ref_count_down()
    assert len(pool.free_blocks) == 1
