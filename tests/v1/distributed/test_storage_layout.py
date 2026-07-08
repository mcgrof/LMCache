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
    StorageInputForm,
    StorageLayoutMode,
    apply_kv_component_split,
    apply_layout_policy,
    classify_input_form,
    derive_storage_layout_mode,
    pass_through_presplit_component_groups,
)
from lmcache.v1.distributed.storage_placement import ComponentKeyScheme


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


# --- CO4: heterogeneous K/V dtype children (asymmetric K16/V8) ----------------


def test_component_split_default_is_byte_identical_regression() -> None:
    """No dtype override reproduces the pre-change same-dtype behaviour."""
    packed = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 8, 128])],
        dtypes=[torch.bfloat16],
    )
    out = apply_kv_component_split(packed)
    assert out.shapes == [torch.Size([4, 8, 128]), torch.Size([4, 8, 128])]
    assert out.dtypes == [torch.bfloat16, torch.bfloat16]


def test_component_split_v_dtype_override_is_heterogeneous() -> None:
    """v_dtype makes the V child fp8 while K stays bf16 (K16/V8)."""
    packed = MemoryLayoutDesc(
        shapes=[torch.Size([2, 32, 64, 8, 128])],
        dtypes=[torch.bfloat16],
    )
    out = apply_kv_component_split(packed, v_dtype=torch.float8_e4m3fn)
    assert len(out.shapes) == 2
    assert out.shapes[0] == torch.Size([32, 64, 8, 128])
    assert out.shapes[1] == torch.Size([32, 64, 8, 128])
    assert out.dtypes[0] is torch.bfloat16  # K child unchanged
    assert out.dtypes[1] is torch.float8_e4m3fn  # V child overridden


def test_component_split_k_dtype_override() -> None:
    """k_dtype overrides the K child independently of V."""
    packed = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 128])],
        dtypes=[torch.float16],
    )
    out = apply_kv_component_split(
        packed, k_dtype=torch.bfloat16, v_dtype=torch.float8_e4m3fn
    )
    assert out.dtypes == [torch.bfloat16, torch.float8_e4m3fn]


def test_component_split_multi_group_applies_override_per_group() -> None:
    """Every input group's V child takes the override, in order."""
    packed = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 128]), torch.Size([2, 4, 128])],
        dtypes=[torch.bfloat16, torch.bfloat16],
    )
    out = apply_kv_component_split(packed, v_dtype=torch.float8_e4m3fn)
    assert out.dtypes == [
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float8_e4m3fn,
    ]


def test_component_split_bad_leading_dim_still_raises() -> None:
    """Leading dim != 2 still raises regardless of dtype overrides."""
    bad = MemoryLayoutDesc(shapes=[torch.Size([3, 4, 128])], dtypes=[torch.bfloat16])
    with pytest.raises(ValueError, match="leading dim 2"):
        apply_kv_component_split(bad, v_dtype=torch.float8_e4m3fn)


def test_apply_layout_policy_forwards_v_dtype() -> None:
    """apply_layout_policy threads v_dtype into the component split."""
    packed = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 128])],
        dtypes=[torch.bfloat16],
    )
    out = apply_layout_policy(
        packed, StorageLayoutMode.KV_COMPONENT_GROUPS, v_dtype=torch.float8_e4m3fn
    )
    assert out.dtypes == [torch.bfloat16, torch.float8_e4m3fn]


def test_apply_layout_policy_packed_rejects_dtype_override() -> None:
    """PACKED cannot express heterogeneous K/V dtypes -> loud error."""
    packed = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 128])],
        dtypes=[torch.bfloat16],
    )
    with pytest.raises(ValueError, match="PACKED"):
        apply_layout_policy(
            packed, StorageLayoutMode.PACKED, v_dtype=torch.float8_e4m3fn
        )


# =============================================================================
# Gate 0 -- classify_input_form (structural: packed vs pre-split)
# =============================================================================


def test_classify_input_form_packed_leading_dim_2() -> None:
    """One leading-dim-2 group -> PACKED_KV_UNIFORM."""
    ld = MemoryLayoutDesc(shapes=[torch.Size([2, 4, 128])], dtypes=[torch.bfloat16])
    assert classify_input_form(ld) == StorageInputForm.PACKED_KV_UNIFORM


def test_classify_input_form_presplit_leading_dim_1() -> None:
    """Two leading-dim-1 component groups (K, V) -> PRESPLIT_COMPONENTS."""
    ld = MemoryLayoutDesc(
        shapes=[torch.Size([1, 28, 256]), torch.Size([1, 28, 256])],
        dtypes=[torch.bfloat16, torch.float8_e4m3fn],
    )
    assert classify_input_form(ld) == StorageInputForm.PRESPLIT_COMPONENTS


def test_classify_input_form_mixed_rejected() -> None:
    """A layout mixing leading-dim-2 and leading-dim-1 is rejected."""
    ld = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 128]), torch.Size([1, 4, 128])],
        dtypes=[torch.bfloat16, torch.float8_e4m3fn],
    )
    with pytest.raises(ValueError, match="mixes or has unexpected leading dims"):
        classify_input_form(ld)


def test_classify_input_form_unexpected_leading_dim_rejected() -> None:
    """A leading dim other than 1 or 2 is rejected."""
    ld = MemoryLayoutDesc(shapes=[torch.Size([3, 4, 128])], dtypes=[torch.bfloat16])
    with pytest.raises(ValueError, match="mixes or has unexpected leading dims"):
        classify_input_form(ld)


def test_classify_input_form_empty_rejected() -> None:
    """An empty layout has no form."""
    ld = MemoryLayoutDesc(shapes=[], dtypes=[])
    with pytest.raises(ValueError, match="empty layout_desc"):
        classify_input_form(ld)


# =============================================================================
# Gate 1 -- pass_through_presplit_component_groups (validate, no split)
# =============================================================================


def _presplit_kv(k_dtype: torch.dtype, v_dtype: torch.dtype) -> MemoryLayoutDesc:
    """A canonical [K, V] pre-split layout (each leading dim 1)."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([1, 28, 256]), torch.Size([1, 28, 256])],
        dtypes=[k_dtype, v_dtype],
    )


def test_passthrough_valid_kv_is_unchanged() -> None:
    """A canonical bf16-K / fp8-V pair passes through byte-identically."""
    ld = _presplit_kv(torch.bfloat16, torch.float8_e4m3fn)
    out = pass_through_presplit_component_groups(ld, v_dtype=torch.float8_e4m3fn)
    assert out.shapes == ld.shapes
    assert out.dtypes == ld.dtypes


def test_passthrough_v_not_fp8_rejected() -> None:
    """A V component that is not the required fp8 dtype fails closed."""
    ld = _presplit_kv(torch.bfloat16, torch.bfloat16)
    with pytest.raises(ValueError, match="V component at index 1"):
        pass_through_presplit_component_groups(ld, v_dtype=torch.float8_e4m3fn)


def test_passthrough_swapped_roles_same_order_rejected() -> None:
    """K and V swapped (fp8 at the K position) is caught by dtype -- this
    is the silent-corruption guard: order is the contract, dtype validates it."""
    ld = _presplit_kv(torch.float8_e4m3fn, torch.bfloat16)
    with pytest.raises(ValueError, match="K component at index 0"):
        pass_through_presplit_component_groups(ld, v_dtype=torch.float8_e4m3fn)


def test_passthrough_odd_group_count_rejected() -> None:
    """An odd number of component groups is not [K, V] pairs."""
    ld = MemoryLayoutDesc(shapes=[torch.Size([1, 28, 256])], dtypes=[torch.bfloat16])
    with pytest.raises(ValueError, match="even, non-zero number"):
        pass_through_presplit_component_groups(ld, v_dtype=torch.float8_e4m3fn)


def test_passthrough_non_leading_dim_1_rejected() -> None:
    """A component group whose leading dim is not 1 is rejected."""
    ld = MemoryLayoutDesc(
        shapes=[torch.Size([2, 28, 256]), torch.Size([2, 28, 256])],
        dtypes=[torch.bfloat16, torch.float8_e4m3fn],
    )
    with pytest.raises(ValueError, match="must have leading dim 1"):
        pass_through_presplit_component_groups(ld, v_dtype=torch.float8_e4m3fn)


def test_passthrough_multi_pair_ok() -> None:
    """Multiple [K, V] pairs (even count) all validate and pass through."""
    ld = MemoryLayoutDesc(
        shapes=[torch.Size([1, 8, 128])] * 4,
        dtypes=[
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.bfloat16,
            torch.float8_e4m3fn,
        ],
    )
    out = pass_through_presplit_component_groups(ld, v_dtype=torch.float8_e4m3fn)
    assert out.dtypes == ld.dtypes


# =============================================================================
# Gate 2 -- apply_layout_policy dispatch (packed split vs pre-split passthrough)
# =============================================================================


def test_apply_layout_policy_presplit_raw_unit_passes_through() -> None:
    """KV_COMPONENT_GROUPS + pre-split input + RAW_UNIT scheme -> pass-through,
    no split, no override."""
    ld = _presplit_kv(torch.bfloat16, torch.float8_e4m3fn)
    out = apply_layout_policy(
        ld,
        StorageLayoutMode.KV_COMPONENT_GROUPS,
        v_dtype=torch.float8_e4m3fn,
        scheme=ComponentKeyScheme.RAW_UNIT,
    )
    assert out.shapes == ld.shapes
    assert out.dtypes == ld.dtypes


def test_apply_layout_policy_presplit_requires_raw_unit_scheme() -> None:
    """Pre-split input under COMPUTED_LEGACY is a contract violation."""
    ld = _presplit_kv(torch.bfloat16, torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="requires the RAW_UNIT"):
        apply_layout_policy(
            ld,
            StorageLayoutMode.KV_COMPONENT_GROUPS,
            v_dtype=torch.float8_e4m3fn,
            scheme=ComponentKeyScheme.COMPUTED_LEGACY,
        )


def test_apply_layout_policy_presplit_requires_v_dtype() -> None:
    """RAW_UNIT pre-split pass-through cannot validate without v_dtype."""
    ld = _presplit_kv(torch.bfloat16, torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="requires .*v_dtype"):
        apply_layout_policy(
            ld,
            StorageLayoutMode.KV_COMPONENT_GROUPS,
            scheme=ComponentKeyScheme.RAW_UNIT,
        )


def test_apply_layout_policy_packed_still_splits_under_raw_unit() -> None:
    """A packed byte-through input (M2 synthetic) still splits with the V
    override -- pre-split awareness must not regress the packed path."""
    packed = MemoryLayoutDesc(shapes=[torch.Size([2, 4, 128])], dtypes=[torch.bfloat16])
    out = apply_layout_policy(
        packed,
        StorageLayoutMode.KV_COMPONENT_GROUPS,
        v_dtype=torch.float8_e4m3fn,
        scheme=ComponentKeyScheme.RAW_UNIT,
    )
    assert out.dtypes == [torch.bfloat16, torch.float8_e4m3fn]


def test_apply_layout_policy_packed_split_needs_no_scheme() -> None:
    """The packed split path is scheme-agnostic (regression: existing
    callers pass no scheme)."""
    packed = MemoryLayoutDesc(shapes=[torch.Size([2, 4, 128])], dtypes=[torch.bfloat16])
    out = apply_layout_policy(packed, StorageLayoutMode.KV_COMPONENT_GROUPS)
    assert len(out.shapes) == 2
