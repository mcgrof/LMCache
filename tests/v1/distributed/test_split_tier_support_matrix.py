# SPDX-License-Identifier: Apache-2.0
"""Config-time support-matrix enforcement for KV_SPLIT_TIER.

The V-only split-tier placement composes an L1-resident exact-K child
with an L2-resident compressed-V child under one logical key.  It is
only correct for a narrow, documented configuration.  These tests pin
that every unsupported combination is rejected at config-validation time
(before any resource is constructed) with a clear ValueError, and that a
config inside the matrix passes.
"""

# Future
from __future__ import annotations

# Standard
import tempfile

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_manager import (
    StorageManager,
    _reject_unsupported_split_tier_config,
)


def _v_only_fs_adapter(base_path: str) -> FSL2AdapterConfig:
    """An FS L2 adapter carrying the V-only serde (derives KV_SPLIT_TIER)."""
    cfg = FSL2AdapterConfig(
        base_path=base_path,
        relative_tmp_dir=None,
        read_ahead_size=None,
        use_odirect=False,
    )
    cfg.serde_config = SerdeConfig(type="asym_k16_v8_v_only")
    return cfg


def _pinned_l1() -> L1ManagerConfig:
    return L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 30,
            use_lazy=True,
            init_size_in_bytes=64 << 20,
        ),
    )


def _cfg(
    l1: L1ManagerConfig | None = None,
    eviction: EvictionConfig | None = None,
    adapters: list[FSL2AdapterConfig] | None = None,
) -> StorageManagerConfig:
    if adapters is None:
        adapters = [_v_only_fs_adapter(tempfile.mkdtemp(prefix="lmcache_sm_test_"))]
    return StorageManagerConfig(
        l1_manager_config=l1 if l1 is not None else _pinned_l1(),
        eviction_config=eviction
        if eviction is not None
        else EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=adapters),  # type: ignore[arg-type]
    )


def test_matrix_accepts_supported_split_tier_config() -> None:
    """The supported shape -- pinned-DRAM L1, one V-only FS adapter, LRU,
    no per-adapter eviction -- passes the matrix check."""
    # Should not raise.
    _reject_unsupported_split_tier_config(_cfg())


def test_matrix_rejects_gds_l1() -> None:
    l1 = _pinned_l1()
    l1.gds_l1_config = GdsL1Config(file_location="/tmp/gds", size_in_bytes=1 << 30)
    with pytest.raises(ValueError, match="GDS L1"):
        _reject_unsupported_split_tier_config(_cfg(l1=l1))


def test_matrix_rejects_device_dax_l1() -> None:
    l1 = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 30,
            use_lazy=False,
            init_size_in_bytes=64 << 20,
            shm_name="",
            devdax_path="/dev/dax0.0",
        ),
    )
    with pytest.raises(ValueError, match="Device-DAX L1"):
        _reject_unsupported_split_tier_config(_cfg(l1=l1))


def test_matrix_rejects_multiple_l2_adapters() -> None:
    adapters = [
        _v_only_fs_adapter(tempfile.mkdtemp(prefix="lmcache_sm_a_")),
        _v_only_fs_adapter(tempfile.mkdtemp(prefix="lmcache_sm_b_")),
    ]
    with pytest.raises(ValueError, match="multiple L2 adapters"):
        _reject_unsupported_split_tier_config(_cfg(adapters=adapters))


def test_matrix_rejects_per_adapter_l2_eviction() -> None:
    adapter = _v_only_fs_adapter(tempfile.mkdtemp(prefix="lmcache_sm_ev_"))
    adapter.eviction_config = EvictionConfig(eviction_policy="LRU")
    with pytest.raises(ValueError, match="per-adapter L2 eviction"):
        _reject_unsupported_split_tier_config(_cfg(adapters=[adapter]))


def test_matrix_rejects_isolated_lru_quota_on_l1() -> None:
    with pytest.raises(ValueError, match="IsolatedLRU"):
        _reject_unsupported_split_tier_config(
            _cfg(eviction=EvictionConfig(eviction_policy="IsolatedLRU"))
        )


def test_matrix_reports_all_violations_at_once() -> None:
    """An operator sees every matrix violation in one error, not just the
    first."""
    l1 = _pinned_l1()
    l1.gds_l1_config = GdsL1Config(file_location="/tmp/gds", size_in_bytes=1 << 30)
    adapters = [
        _v_only_fs_adapter(tempfile.mkdtemp(prefix="lmcache_sm_m1_")),
        _v_only_fs_adapter(tempfile.mkdtemp(prefix="lmcache_sm_m2_")),
    ]
    with pytest.raises(ValueError) as ei:
        _reject_unsupported_split_tier_config(_cfg(l1=l1, adapters=adapters))
    msg = str(ei.value)
    assert "GDS L1" in msg
    assert "multiple L2 adapters" in msg


def test_storage_manager_init_rejects_before_building_resources() -> None:
    """Wiring: StorageManager.__init__ runs the matrix check (gated on the
    derived placement mode) before constructing the L1 manager or adapters,
    so a rejected config raises with nothing built."""
    adapters = [
        _v_only_fs_adapter(tempfile.mkdtemp(prefix="lmcache_sm_w1_")),
        _v_only_fs_adapter(tempfile.mkdtemp(prefix="lmcache_sm_w2_")),
    ]
    with pytest.raises(ValueError, match="multiple L2 adapters"):
        StorageManager(_cfg(adapters=adapters))
