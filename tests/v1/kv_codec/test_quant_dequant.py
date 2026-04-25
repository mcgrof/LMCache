# SPDX-License-Identifier: Apache-2.0
"""Quantize/dequantize roundtrip noise bounds.

FP8 e4m3fn has ~6.3% worst-case relative rounding error around
typical activation magnitudes.  These tests bound the observed
roundtrip error against that.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    ScaleScope,
    compute_v_scales,
    dequantize_v_fp8,
    quantize_v_fp8,
)


# Empirical FP8 e4m3fn relative-error bound around mean activation
# values; tested against random gaussian tensors.  We use a conservative
# 7.5% bound.  If a future change blows past this on random gaussian
# data, the codec is rounding the wrong way.
FP8_E4M3_REL_ERR_BOUND = 0.075


@pytest.mark.parametrize(
    "shape", [(1, 8, 2, 16), (2, 32, 4, 32), (4, 128, 8, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_per_tensor_roundtrip_bound(shape, dtype):
    v = torch.randn(*shape, dtype=dtype)
    s = compute_v_scales(v, ScaleScope.PER_TENSOR)
    q = quantize_v_fp8(v, s, ScaleScope.PER_TENSOR)
    dq = dequantize_v_fp8(q, s, ScaleScope.PER_TENSOR, out_dtype=torch.float32)
    v32 = v.to(torch.float32)
    abs_err = (dq - v32).abs()
    rel_err = abs_err / (v32.abs() + 1e-6)
    # median relative error should be small; max can be larger near
    # zero crossings where rel_err is dominated by tiny denominators.
    assert rel_err.median().item() < FP8_E4M3_REL_ERR_BOUND, rel_err.median()


@pytest.mark.parametrize(
    "shape", [(1, 8, 2, 16), (2, 32, 4, 32), (4, 128, 8, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_per_layer_head_roundtrip_bound(shape, dtype):
    v = torch.randn(*shape, dtype=dtype)
    s = compute_v_scales(v, ScaleScope.PER_LAYER_HEAD)
    q = quantize_v_fp8(v, s, ScaleScope.PER_LAYER_HEAD)
    dq = dequantize_v_fp8(q, s, ScaleScope.PER_LAYER_HEAD, out_dtype=torch.float32)
    rel = (dq - v.to(torch.float32)).abs() / (v.abs().to(torch.float32) + 1e-6)
    assert rel.median().item() < FP8_E4M3_REL_ERR_BOUND


def test_per_page_head_roundtrip_bound(paged_kv):
    _, v = paged_kv
    s = compute_v_scales(
        v, ScaleScope.PER_PAGE_HEAD, page_axis=0, head_axis=2
    )
    q = quantize_v_fp8(
        v, s, ScaleScope.PER_PAGE_HEAD, page_axis=0, head_axis=2
    )
    dq = dequantize_v_fp8(
        q, s, ScaleScope.PER_PAGE_HEAD,
        page_axis=0, head_axis=2,
        out_dtype=torch.float32,
    )
    rel = (dq - v.to(torch.float32)).abs() / (v.abs().to(torch.float32) + 1e-6)
    assert rel.median().item() < FP8_E4M3_REL_ERR_BOUND


def test_dequant_dtype_is_requested():
    """Catches: dequant returns its internal dtype instead of out_dtype."""
    v = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    s = compute_v_scales(v, ScaleScope.PER_TENSOR)
    q = quantize_v_fp8(v, s, ScaleScope.PER_TENSOR)
    for out in (torch.float16, torch.bfloat16, torch.float32):
        dq = dequantize_v_fp8(q, s, ScaleScope.PER_TENSOR, out_dtype=out)
        assert dq.dtype == out


def test_quant_dtype_is_fp8():
    """Catches: quant returns int8 or fp16 by mistake."""
    v = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    s = compute_v_scales(v, ScaleScope.PER_TENSOR)
    q = quantize_v_fp8(v, s, ScaleScope.PER_TENSOR)
    assert q.dtype == torch.float8_e4m3fn


def test_zero_tensor_roundtrip_is_zero():
    v = torch.zeros(1, 8, 2, 16, dtype=torch.float16)
    s = compute_v_scales(v, ScaleScope.PER_TENSOR)
    q = quantize_v_fp8(v, s, ScaleScope.PER_TENSOR)
    dq = dequantize_v_fp8(q, s, ScaleScope.PER_TENSOR, out_dtype=torch.float16)
    assert torch.equal(dq, v)


def test_saturation_not_overflow():
    """Values > FP8 max saturate at ±finfo.max, never inf or nan."""
    v = torch.tensor([1e6, -1e6, 0.5], dtype=torch.float32)
    s = torch.tensor(1.0)
    q = quantize_v_fp8(v, s, ScaleScope.PER_TENSOR)
    f = q.to(torch.float32)
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    assert torch.isfinite(f).all(), f
    assert f.abs().max().item() <= fp8_max + 1e-3
