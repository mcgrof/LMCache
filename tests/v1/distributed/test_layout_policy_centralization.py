# SPDX-License-Identifier: Apache-2.0
"""Centralized L1 storage-layout-policy application.

The L1 storage-layout policy (``StorageManager.apply_layout_policy`` ->
``apply_kv_component_split``) is applied at exactly ONE choke point per
direction -- ``reserve_write`` on the store path and
``submit_prefetch_task`` on the load path -- instead of at scattered call
sites.

This module pins the choke-point contract for the default PACKED config
(no serde / KV_TOGETHER): the apply is a pure identity no-op, so the
reserved object keeps the transfer-side shape and KV_TOGETHER traffic is
not regressed.  The transforming (multi-output serde) direction of the
same choke point -- component split applied exactly once, and the
non-idempotency guard that is why the old external pre-applies were
removed -- is exercised by the multi-output serde work, which introduces
the first non-identity layout.
"""

# Future
from __future__ import annotations

# Standard
import shutil
import tempfile

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_layout import StorageLayoutMode
from lmcache.v1.distributed.storage_manager import StorageManager


def _make_key(chunk_hash: bytes) -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name="pr0-layout",
        kv_rank=0,
        object_group_id=0,
    )


def _build_sm(base_path: str, serde_type: str | None) -> StorageManager:
    """StorageManager over one FS adapter, optionally carrying a serde."""
    fs_cfg = FSL2AdapterConfig(
        base_path=base_path,
        relative_tmp_dir=None,
        read_ahead_size=None,
        use_odirect=False,
    )
    if serde_type is not None:
        fs_cfg.serde_config = SerdeConfig(type=serde_type)
    sm_cfg = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1 << 30,
                use_lazy=True,
                init_size_in_bytes=64 << 20,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
    )
    return StorageManager(sm_cfg)


def test_reserve_write_packed_config_is_identity_noop() -> None:
    """A no-serde (PACKED) StorageManager reserves the transfer-side shape
    unchanged -- the centralized apply is a pure pass-through, so
    KV_TOGETHER traffic is not regressed."""
    disk = tempfile.mkdtemp(prefix="lmcache_pr0_packed_")
    try:
        sm = _build_sm(disk, serde_type=None)
        try:
            assert sm.storage_layout_mode == StorageLayoutMode.PACKED
            key = _make_key(b"\x01" * 32)
            packed = MemoryLayoutDesc(
                shapes=[torch.Size([2, 4, 8])], dtypes=[torch.bfloat16]
            )
            reserved = sm.reserve_write([key], packed, mode="new")
            assert list(reserved) == [key]
            # PACKED: single group, shape unchanged (not split).
            assert reserved[key].get_shapes() == [torch.Size([2, 4, 8])]
        finally:
            sm.close()
    finally:
        shutil.rmtree(disk, ignore_errors=True)
