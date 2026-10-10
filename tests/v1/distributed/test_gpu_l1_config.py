# SPDX-License-Identifier: Apache-2.0
"""Validate GPU L1 selection before allocating GPU memory or opening storage."""

# Standard
from typing import Literal
from unittest.mock import patch
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import L1BackendType
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GPUL1ManagerConfig,
    L1MemoryManagerConfig,
    get_arg_parser,
    get_configured_capacity_bytes,
    l1_exposes_single_memory_region,
    parse_args_to_config,
)
from lmcache.v1.distributed.memory_manager.gpu_l1_memory_manager import (
    GPUL1MemoryManager,
)


def test_tagged_gpu_l1_configuration() -> None:
    config = parse_args_to_config(
        get_arg_parser().parse_args(
            [
                "--l1-manager",
                json.dumps(
                    {
                        "type": "GPU",
                        "tag": "gpu",
                        "size_gb": 0.0625,
                        "device": "cuda:0",
                        "buffer_provider": "vulkan_rm",
                        "eviction": {"eviction_policy": "noop"},
                    }
                ),
            ]
        )
    )
    l1 = config.l1_manager_config
    assert isinstance(l1, GPUL1ManagerConfig)
    assert l1.memory_config.buffer_registration == "dmabuf"
    assert get_configured_capacity_bytes(l1) == {L1BackendType.GPU: 64 << 20}
    assert not l1_exposes_single_memory_region(config)


@pytest.mark.parametrize("device,provider", [("cpu", "native"), ("cuda:0", "auto")])
def test_gpu_configuration_rejects_host_fallback(device: str, provider: str) -> None:
    with pytest.raises(ValueError):
        GPUL1ManagerConfig(
            L1MemoryManagerConfig(4 << 20, False, shm_name=""),
            tag="gpu",
            device=device,
            buffer_provider=provider,
        )


def test_gpu_manager_does_not_allocate_on_host_without_runtime() -> None:
    config = GPUL1ManagerConfig(
        L1MemoryManagerConfig(4 << 20, False, shm_name=""), tag="gpu"
    )
    with (
        patch(
            "lmcache.v1.distributed.memory_manager.gpu_l1_memory_manager.torch_dev.is_available",
            return_value=False,
        ),
        patch(
            "lmcache.v1.distributed.memory_manager.gpu_l1_memory_manager.GPUMemoryAllocator"
        ) as allocator,
    ):
        with pytest.raises(RuntimeError, match="device runtime"):
            GPUL1MemoryManager(config)
        allocator.assert_not_called()


@pytest.mark.parametrize(
    "mode,p2p", [("auto", False), ("engine_driven", False), ("lmcache_driven", True)]
)
def test_server_rejects_unsupported_gpu_transfers_before_allocation(
    mode: Literal["auto", "engine_driven", "lmcache_driven"], p2p: bool
) -> None:
    # First Party
    from lmcache.v1.distributed.config import StorageManagerConfig
    from lmcache.v1.mp_observability.config import ObservabilityConfig
    from lmcache.v1.multiprocess.config import MPServerConfig, P2PConfig
    from lmcache.v1.multiprocess.server import run_cache_server

    config = StorageManagerConfig(
        GPUL1ManagerConfig(
            L1MemoryManagerConfig(4 << 20, False, shm_name=""), tag="gpu"
        ),
        EvictionConfig("noop"),
    )
    with patch("lmcache.v1.multiprocess.server.StorageManager") as manager:
        with pytest.raises(ValueError, match="GPU L1"):
            run_cache_server(
                MPServerConfig(
                    supported_transfer_mode=mode,
                    p2p_config=P2PConfig(advertise_url="127.0.0.1:9999" if p2p else ""),
                ),
                config,
                ObservabilityConfig(),
                start_prometheus_http_server=False,
            )
        manager.assert_not_called()


def test_gpu_l1_rejects_cpu_byte_view_serde() -> None:
    # First Party
    from lmcache.v1.distributed.config import (
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
        RawBlockL2AdapterConfig,
    )
    from lmcache.v1.distributed.serde import SerdeConfig

    raw = RawBlockL2AdapterConfig(
        device_path="/unused",
        slot_bytes=16384,
        meta_total_bytes=1 << 20,
        use_odirect=True,
        io_engine="io_uring",
    )
    raw.affinity_tag = "gpu"
    raw.serde_config = SerdeConfig("aesgcm")
    with pytest.raises(ValueError, match="fp8 serde only"):
        StorageManagerConfig(
            GPUL1ManagerConfig(
                L1MemoryManagerConfig(4 << 20, False, shm_name=""), tag="gpu"
            ),
            EvictionConfig("noop"),
            l2_adapter_config=L2AdaptersConfig([raw]),
        )
