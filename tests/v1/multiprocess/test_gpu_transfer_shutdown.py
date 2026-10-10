# SPDX-License-Identifier: Apache-2.0
"""Check imported GPU context retention through the public transfer module API."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)


def test_failed_context_fence_retains_imports_and_reports_unhealthy() -> None:
    """Withhold import release and layout removal after losing stream drain."""
    engine = MagicMock()
    cache = MagicMock()
    cache.kv_layer_groups_manager.num_object_groups = 1
    cache.close.side_effect = RuntimeError("injected context fence failure")
    layout = MemoryLayoutDesc([torch.Size([16])], [torch.bfloat16])
    with (
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.create_cache_context",
            return_value=cache,
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.get_event_ipc_backend"
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.get_layout_desc",
            return_value=layout,
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.torch_dev.empty_cache"
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.torch_dev.ipc_collect"
        ),
    ):
        module = LMCacheDrivenTransferModule(engine)
        before = module.report_status()["retained_gpu_context_count"]
        try:
            module.register_kv_cache(31, [], "test-model", 1, EngineType.VLLM, {}, [])
            module.unregister_kv_cache(31)
            status = module.report_status()
            assert status["registered_gpu_ids"] == []
            assert status["retained_gpu_context_count"] == before + 1
            assert not status["is_healthy"]
            engine.layout_desc_registry.unregister.assert_not_called()
            module.close()
            assert module.report_status()["retained_gpu_context_count"] == before + 1
        finally:
            module.close()
