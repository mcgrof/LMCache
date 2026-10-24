# SPDX-License-Identifier: Apache-2.0
"""Resolve physical geometry without treating fused content as rotary width."""

# Standard
from typing import Literal

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.kmesh_span_layout import resolve_registered_paged_shape
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format


@pytest.mark.parametrize(
    "shape,layout,fmt",
    [
        ((2, 16, 32, 2, 64), "NHD", "NL_X_TWO_NB_BS_NH_HS"),
        ((2, 16, 2, 32, 64), "HND", "NL_X_TWO_NB_NH_BS_HS"),
        ((16, 32, 2, 128), "NHD", "NL_X_NB_BS_NH_CS"),
        ((16, 2, 32, 128), "HND", "NL_X_NB_NH_BS_CS"),
    ],
)
def test_physical_geometry_preserves_real_head_size(
    shape: tuple[int, ...], layout: Literal["NHD", "HND"], fmt: str
) -> None:
    result = resolve_registered_paged_shape(shape, layout)
    assert (result.blocks, result.block_size, result.kv_heads, result.head_size) == (
        16,
        32,
        2,
        64,
    )
    assert result.format_name == fmt


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("packed", [False, True])
def test_geometry_agrees_with_actual_native_format_discovery(
    layout: Literal["NHD", "HND"], packed: bool
) -> None:
    shape: tuple[int, ...] = (
        (2, 16, 32, 2, 64) if layout == "NHD" else (2, 16, 2, 32, 64)
    )
    if packed:
        shape = (16, 32, 2, 128) if layout == "NHD" else (16, 2, 32, 128)
    original = [torch.empty(shape, dtype=torch.bfloat16)]
    fmt, views = normalize_kv_and_discover_format(
        original, EngineType.VLLM, layout_hints={"kv_layout": layout}
    )
    assert fmt.name == resolve_registered_paged_shape(shape, layout).format_name
    assert isinstance(views, list) and isinstance(views[0], torch.Tensor)
    assert views[0].data_ptr() == original[0].data_ptr()
    assert views[0].shape == original[0].shape


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1, 2, 3),
        (1, 16, 32, 2, 64),
        (16, 32, 2, 127),
        (16, 32, 2, 126),
        (16, 0, 2, 128),
        (True, 32, 2, 128),
    ],
)
def test_invalid_geometry_refuses_before_registration(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        resolve_registered_paged_shape(shape, "NHD")
