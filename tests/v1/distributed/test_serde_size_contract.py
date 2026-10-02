# SPDX-License-Identifier: Apache-2.0
"""Backend compatibility tests for serde serialized-size contracts."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    L2AdaptersConfig,
)
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import S3L2AdapterConfig
from lmcache.v1.distributed.l2_adapters.valkey_l2_adapter import (
    ValkeyL2AdapterConfig,
)
from lmcache.v1.distributed.serde import (
    SerdeConfig,
    create_serde_processor,
    register_serde_factory,
)
from lmcache.v1.distributed.storage_manager import StorageManager

# The conservative declaration exercises admission independently of payload format.
register_serde_factory(
    "test-upper-bound-admission",
    lambda _: create_serde_processor(SerdeConfig(type="fp8")),
)


def _config(adapters: list[L2AdapterConfigBase]) -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=64 << 20,
                use_lazy=False,
                init_size_in_bytes=64 << 20,
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=adapters),
    )


@pytest.mark.parametrize(
    "adapter",
    [
        S3L2AdapterConfig(
            s3_endpoint="s3://test-bucket",
            s3_region="us-east-1",
        ),
        ValkeyL2AdapterConfig(startup_nodes=[("localhost", 6379)]),
    ],
    ids=["s3", "valkey"],
)
def test_upper_bound_serde_rejects_backend_without_used_length_before_resources(
    adapter: S3L2AdapterConfig | ValkeyL2AdapterConfig,
) -> None:
    """A registered upper-bound contract requires actual-used-length loads."""
    adapter.serde_config = SerdeConfig(type="test-upper-bound-admission")
    with pytest.raises(ValueError, match="Upper-bound serde.*actual-used-length"):
        StorageManager(_config([adapter]))


def test_upper_bound_serde_accepts_filesystem_used_length_contract(tmp_path) -> None:
    adapter = FSL2AdapterConfig(
        base_path=str(tmp_path),
        relative_tmp_dir=None,
        read_ahead_size=None,
        use_odirect=False,
    )
    adapter.serde_config = SerdeConfig(type="test-upper-bound-admission")
    manager = StorageManager(_config([adapter]))
    manager.close()


@pytest.mark.parametrize(
    "adapter",
    [
        S3L2AdapterConfig(
            s3_endpoint="s3://test-bucket",
            s3_region="us-east-1",
        ),
        ValkeyL2AdapterConfig(startup_nodes=[("localhost", 6379)]),
    ],
    ids=["s3", "valkey"],
)
def test_computed_asym_rejects_backend_without_used_length_before_resources(
    adapter: S3L2AdapterConfig | ValkeyL2AdapterConfig,
) -> None:
    """COMPUTED KV_TOGETHER emits less than its upper-bound estimate."""
    adapter.serde_config = SerdeConfig(type="asym_k16_v8")
    with pytest.raises(ValueError, match="Upper-bound serde.*actual-used-length"):
        StorageManager(_config([adapter]))


def test_computed_asym_accepts_filesystem_used_length_contract(tmp_path) -> None:
    adapter = FSL2AdapterConfig(
        base_path=str(tmp_path),
        relative_tmp_dir=None,
        read_ahead_size=None,
        use_odirect=False,
    )
    adapter.serde_config = SerdeConfig(type="asym_k16_v8")
    manager = StorageManager(_config([adapter]))
    manager.close()
