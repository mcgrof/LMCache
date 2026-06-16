# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the storage layout policy
(``lmcache/v1/distributed/storage_layout.py``).

Covers:

* :class:`StorageLayoutMode` selection from configured serdes.
* The ``[2, ...]`` -> ``[...] + [...]`` packed-to-component split.
* Mixed-serde rejection (single-tensor + multi-output on the same
  StorageManager is invalid).
"""

# Future
from __future__ import annotations

# Standard
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.l2_adapters.config import L2AdapterConfigBase
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_layout import (
    StorageLayoutMode,
    apply_kv_component_split,
    apply_layout_policy,
    derive_storage_layout_mode,
)


# =============================================================================
# Helpers: minimal fake L2 adapter config carrying a serde_config.
# =============================================================================


class _FakeAdapterCfg:
    """Bare-minimum stand-in for ``L2AdapterConfigBase`` exposing only
    ``serde_config`` -- which is all ``derive_storage_layout_mode``
    looks at."""

    def __init__(self, serde_config) -> None:
        self.serde_config = serde_config


def _cfgs(*cfgs: _FakeAdapterCfg) -> list[L2AdapterConfigBase]:
    """Cast test stand-ins to the production adapter-config list type."""
    return cast("list[L2AdapterConfigBase]", list(cfgs))


# =============================================================================
# derive_storage_layout_mode
# =============================================================================


def test_derive_storage_layout_mode_empty_returns_packed() -> None:
    """No adapters configured -> default packed."""
    assert derive_storage_layout_mode([]) == StorageLayoutMode.PACKED


def test_derive_storage_layout_mode_no_serde_returns_packed() -> None:
    """An adapter without a serde_config -> packed."""
    cfgs = [_FakeAdapterCfg(serde_config=None)]
    assert derive_storage_layout_mode(_cfgs(*cfgs)) == StorageLayoutMode.PACKED


def test_derive_storage_layout_mode_fp8_returns_packed() -> None:
    """The built-in single-tensor fp8 serde -> packed."""
    cfgs = [_FakeAdapterCfg(serde_config=SerdeConfig(type="fp8"))]
    assert derive_storage_layout_mode(_cfgs(*cfgs)) == StorageLayoutMode.PACKED


def test_derive_storage_layout_mode_asym_returns_kv_component_groups() -> None:
    """A multi-output asym_k16_v8 serde -> KV_COMPONENT_GROUPS."""
    cfgs = [_FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8"))]
    assert (
        derive_storage_layout_mode(_cfgs(*cfgs))
        == StorageLayoutMode.KV_COMPONENT_GROUPS
    )


def test_derive_storage_layout_mode_asym_v_only_returns_kv_component_groups() -> None:
    """The V-only variant is also multi-output -> KV_COMPONENT_GROUPS."""
    cfgs = [_FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8_v_only"))]
    assert (
        derive_storage_layout_mode(_cfgs(*cfgs))
        == StorageLayoutMode.KV_COMPONENT_GROUPS
    )


def test_derive_storage_layout_mode_all_packed_returns_packed() -> None:
    """Multiple adapters all asking for packed (mix of no-serde and
    fp8) -> packed."""
    cfgs = [
        _FakeAdapterCfg(serde_config=None),
        _FakeAdapterCfg(serde_config=SerdeConfig(type="fp8")),
        _FakeAdapterCfg(serde_config=None),
    ]
    assert derive_storage_layout_mode(_cfgs(*cfgs)) == StorageLayoutMode.PACKED


def test_derive_storage_layout_mode_all_kv_components_returns_kv_components() -> None:
    """Multiple multi-output adapters all asking for KV-component
    groups -> KV_COMPONENT_GROUPS."""
    cfgs = [
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8")),
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8_v_only")),
    ]
    assert (
        derive_storage_layout_mode(_cfgs(*cfgs))
        == StorageLayoutMode.KV_COMPONENT_GROUPS
    )


def test_derive_storage_layout_mode_mixed_rejected() -> None:
    """A single-tensor serde alongside a multi-output serde is rejected
    -- they demand incompatible L1 ``MemoryObj`` shapes."""
    cfgs = [
        _FakeAdapterCfg(serde_config=SerdeConfig(type="fp8")),
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8")),
    ]
    with pytest.raises(ValueError, match="Incompatible L2 adapter storage layout"):
        derive_storage_layout_mode(_cfgs(*cfgs))


def test_derive_storage_layout_mode_mixed_no_serde_and_multi_rejected() -> None:
    """An adapter with no serde (defaults to packed) cannot coexist
    with one that demands component groups."""
    cfgs = [
        _FakeAdapterCfg(serde_config=None),
        _FakeAdapterCfg(serde_config=SerdeConfig(type="asym_k16_v8")),
    ]
    with pytest.raises(ValueError, match="Incompatible L2 adapter storage layout"):
        derive_storage_layout_mode(_cfgs(*cfgs))


# =============================================================================
# apply_kv_component_split
# =============================================================================


def test_apply_kv_component_split_single_group() -> None:
    """A single ``[2, ...]`` BF16 group -> two ``[...]`` BF16 groups."""
    packed = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 256, 128])],
        dtypes=[torch.bfloat16],
    )
    split = apply_kv_component_split(packed)
    assert len(split.shapes) == 2
    assert split.shapes[0] == torch.Size([4, 256, 128])
    assert split.shapes[1] == torch.Size([4, 256, 128])
    assert split.dtypes == [torch.bfloat16, torch.bfloat16]


def test_apply_kv_component_split_preserves_total_bytes() -> None:
    """Splitting must not change total bytes (same buffer, retyped)."""
    packed = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 256, 128])],
        dtypes=[torch.bfloat16],
    )
    split = apply_kv_component_split(packed)
    packed_bytes = sum(
        s.numel() * d.itemsize
        for s, d in zip(packed.shapes, packed.dtypes, strict=True)
    )
    split_bytes = sum(
        s.numel() * d.itemsize for s, d in zip(split.shapes, split.dtypes, strict=True)
    )
    assert packed_bytes == split_bytes


def test_apply_kv_component_split_multi_group() -> None:
    """Two input groups (e.g. multi-layer-group MLA) each split -> four
    output groups in order [g0_K, g0_V, g1_K, g1_V]."""
    packed = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 128]), torch.Size([2, 8, 64])],
        dtypes=[torch.bfloat16, torch.float16],
    )
    split = apply_kv_component_split(packed)
    assert len(split.shapes) == 4
    assert split.shapes[0] == torch.Size([4, 128])
    assert split.shapes[1] == torch.Size([4, 128])
    assert split.shapes[2] == torch.Size([8, 64])
    assert split.shapes[3] == torch.Size([8, 64])
    assert split.dtypes == [
        torch.bfloat16,
        torch.bfloat16,
        torch.float16,
        torch.float16,
    ]


def test_apply_kv_component_split_rejects_non_2_leading_dim() -> None:
    """An input group whose leading dim is not 2 (not K|V packed) is
    rejected -- otherwise we'd silently produce a wrong split."""
    bad = MemoryLayoutDesc(
        shapes=[torch.Size([3, 4, 128])],
        dtypes=[torch.bfloat16],
    )
    with pytest.raises(ValueError, match="leading dim 2"):
        apply_kv_component_split(bad)


def test_apply_kv_component_split_rejects_empty_shape() -> None:
    """A 0-d input shape has no leading dim -> rejected."""
    bad = MemoryLayoutDesc(
        shapes=[torch.Size([])],
        dtypes=[torch.bfloat16],
    )
    with pytest.raises(ValueError, match="leading dim 2"):
        apply_kv_component_split(bad)


# =============================================================================
# apply_layout_policy (dispatch on mode)
# =============================================================================


def test_apply_layout_policy_packed_is_passthrough() -> None:
    """PACKED mode returns the input layout unchanged."""
    packed = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 128])],
        dtypes=[torch.bfloat16],
    )
    out = apply_layout_policy(packed, StorageLayoutMode.PACKED)
    assert out.shapes == packed.shapes
    assert out.dtypes == packed.dtypes


def test_apply_layout_policy_kv_components_splits() -> None:
    """KV_COMPONENT_GROUPS mode delegates to apply_kv_component_split."""
    packed = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 128])],
        dtypes=[torch.bfloat16],
    )
    out = apply_layout_policy(packed, StorageLayoutMode.KV_COMPONENT_GROUPS)
    assert len(out.shapes) == 2
    assert out.shapes[0] == torch.Size([4, 128])
    assert out.shapes[1] == torch.Size([4, 128])
