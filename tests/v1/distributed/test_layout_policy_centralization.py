# SPDX-License-Identifier: Apache-2.0
"""Centralized L1 storage-layout-policy application.

The L1 storage-layout policy (``StorageManager.apply_layout_policy`` ->
``apply_kv_component_split``) is applied at exactly ONE choke point per
direction -- ``reserve_write`` on the store path and
``submit_prefetch_task`` on the load path -- instead of at scattered call
sites.  These tests pin the choke-point contract:

* For a PACKED config (no serde / KV_TOGETHER) the apply is a pure
  identity no-op: the reserved object keeps the transfer-side shape.  This
  is what guarantees the centralization does not regress KV_TOGETHER
  traffic.
* For a multi-output serde (KV_COMPONENT_GROUPS) the apply splits each
  ``[2, ...]`` group into K and V component groups -- applied EXACTLY once,
  from the packed layout the caller passes (callers must not pre-apply).
* Because ``apply_kv_component_split`` is not idempotent, a caller that
  DOES pre-apply and then hands the already-split layout to the choke
  point fails loudly -- which is why the old external pre-applies were
  removed.

The test uses ``asym_k16_v8`` (Mode 1: KV_COMPONENT_GROUPS layout with
KV_TOGETHER placement) so the layout centralization is exercised WITHOUT
any split-tier placement semantics.
"""

# Future
from __future__ import annotations

# Standard
import shutil
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.api import PrefetchRequestSpec
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
from lmcache.v1.distributed.storage_placement import StoragePlacementMode


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
            assert sm.storage_placement_mode == StoragePlacementMode.KV_TOGETHER
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


def test_reserve_write_applies_component_split_exactly_once() -> None:
    """A multi-output-serde (KV_COMPONENT_GROUPS) StorageManager splits the
    packed ``[2, ...]`` layout into two component groups inside
    reserve_write -- from the packed layout the caller passes, with no
    caller-side pre-apply, and split exactly once (two groups, not four)."""
    disk = tempfile.mkdtemp(prefix="lmcache_pr0_split_")
    try:
        sm = _build_sm(disk, serde_type="asym_k16_v8")
        try:
            assert sm.storage_layout_mode == StorageLayoutMode.KV_COMPONENT_GROUPS
            # Mode 1: component layout but NOT split-tier placement.
            assert sm.storage_placement_mode == StoragePlacementMode.KV_TOGETHER
            key = _make_key(b"\x02" * 32)
            component_shape = torch.Size([4, 256, 128])
            packed = MemoryLayoutDesc(
                shapes=[torch.Size([2, 4, 256, 128])], dtypes=[torch.bfloat16]
            )
            reserved = sm.reserve_write([key], packed, mode="new")
            assert list(reserved) == [key]
            # Split once: exactly two component groups, each the [2,...]
            # leading dim stripped -- not four (that would be a double-apply).
            assert reserved[key].get_shapes() == [component_shape, component_shape]
        finally:
            sm.close()
    finally:
        shutil.rmtree(disk, ignore_errors=True)


def test_reserve_write_rejects_pre_applied_layout() -> None:
    """The transform is not idempotent: a caller that pre-applies the
    policy and then hands the already-split layout to reserve_write fails
    loudly (the split's leading dim is no longer 2).  This is exactly why
    the old external pre-applies were removed when the choke point was
    added."""
    disk = tempfile.mkdtemp(prefix="lmcache_pr0_double_")
    try:
        sm = _build_sm(disk, serde_type="asym_k16_v8")
        try:
            key = _make_key(b"\x03" * 32)
            packed = MemoryLayoutDesc(
                shapes=[torch.Size([2, 4, 256, 128])], dtypes=[torch.bfloat16]
            )
            pre_applied = sm.apply_layout_policy(packed)
            assert len(pre_applied.shapes) == 2
            with pytest.raises(ValueError, match="leading dim 2"):
                sm.reserve_write([key], pre_applied, mode="new")
        finally:
            sm.close()
    finally:
        shutil.rmtree(disk, ignore_errors=True)


def test_submit_prefetch_task_rejects_pre_applied_layout() -> None:
    """The prefetch choke point applies the policy too: handing it an
    already-split layout double-applies and raises, mirroring the
    reserve_write guard."""
    disk = tempfile.mkdtemp(prefix="lmcache_pr0_pfdouble_")
    try:
        sm = _build_sm(disk, serde_type="asym_k16_v8")
        try:
            key = _make_key(b"\x04" * 32)
            packed = MemoryLayoutDesc(
                shapes=[torch.Size([2, 4, 256, 128])], dtypes=[torch.bfloat16]
            )
            pre_applied = sm.apply_layout_policy(packed)
            with pytest.raises(ValueError, match="leading dim 2"):
                sm.submit_prefetch_task(PrefetchRequestSpec([key], {0: pre_applied}))
        finally:
            sm.close()
    finally:
        shutil.rmtree(disk, ignore_errors=True)
