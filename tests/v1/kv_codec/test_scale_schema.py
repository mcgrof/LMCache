# SPDX-License-Identifier: Apache-2.0
"""Scale-computation correctness tests.

These catch:
- per-tensor / per-layer-head / per-page-head scales computed against
  the wrong axis
- 0/0 NaN when a head/page is all-zero
- single-outlier scaling that destroys precision on the rest of the
  tensor (the Qwen-fragility failure mode in physical form)
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    ScaleScope,
    UnsupportedConfigError,
    compute_v_scales,
    dequantize_v_fp8,
    quantize_v_fp8,
)


def test_per_tensor_scale_value(small_kv, fp8_max):
    """Catches: scale formula bug (should be amax / qmax, not amax,
    not amax * qmax)."""
    _, v = small_kv
    expected = v.abs().amax().to(torch.float32) / fp8_max
    got = compute_v_scales(v, ScaleScope.PER_TENSOR)
    torch.testing.assert_close(got.reshape(()), expected.reshape(()))


def test_per_layer_head_scale_shape(small_kv, fp8_max):
    """Catches: wrong axis selection for per-head reduction.
    Expected scale shape is (n_heads,)."""
    _, v = small_kv
    n_heads = v.shape[-2]
    scales = compute_v_scales(v, ScaleScope.PER_LAYER_HEAD)
    assert scales.shape == (n_heads,)


def test_per_layer_head_scale_per_head_amax(fp8_max):
    """Each head's scale equals that head's amax / qmax independently
    of other heads.  Catches: leaked reduction across head axis."""
    n_heads = 4
    head_dim = 8
    v = torch.zeros(1, 1, n_heads, head_dim, dtype=torch.float16)
    # Make head 0's max=10, head 1's max=100, others 1.0
    v[..., 0, 0] = 10.0
    v[..., 1, 0] = 100.0
    v[..., 2, 0] = 1.0
    v[..., 3, 0] = 0.5
    scales = compute_v_scales(v, ScaleScope.PER_LAYER_HEAD)
    for h, expected_amax in enumerate([10.0, 100.0, 1.0, 0.5]):
        torch.testing.assert_close(
            scales[h].item(),
            expected_amax / fp8_max,
            atol=1e-3,
            rtol=0,
        )


def test_all_zero_head_does_not_nan(fp8_max):
    """Catches: 0/0 NaN when amax==0.  We use a 1.0 sentinel scale."""
    v = torch.zeros(1, 1, 2, 8, dtype=torch.float16)
    s = compute_v_scales(v, ScaleScope.PER_LAYER_HEAD)
    assert torch.isfinite(s).all(), s
    assert (s > 0).all(), s


def test_zero_tensor_per_tensor_scope(fp8_max):
    """Per-tensor amax=0 also gets the sentinel."""
    v = torch.zeros(1, 8, 2, 16, dtype=torch.float16)
    s = compute_v_scales(v, ScaleScope.PER_TENSOR)
    assert torch.isfinite(s).all()
    assert s.item() > 0


def test_per_page_head_requires_page_axis(small_kv):
    """Catches: forgot to pass page_axis."""
    _, v = small_kv
    with pytest.raises(UnsupportedConfigError, match="page_axis"):
        compute_v_scales(v, ScaleScope.PER_PAGE_HEAD)


def test_per_page_head_shape(paged_kv):
    """Catches: scale shape mismatch for paged KV."""
    _, v = paged_kv  # (n_pages=4, page_size=8, n_heads=2, head_dim=16)
    scales = compute_v_scales(
        v, ScaleScope.PER_PAGE_HEAD, page_axis=0, head_axis=2
    )
    assert scales.shape == (4, 2), scales.shape


def test_external_scope_raises():
    v = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    with pytest.raises(UnsupportedConfigError, match="EXTERNAL"):
        compute_v_scales(v, ScaleScope.EXTERNAL)


def test_unknown_dtype_raises():
    v = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    with pytest.raises(UnsupportedConfigError):
        compute_v_scales(v, ScaleScope.PER_TENSOR, fp8_dtype=torch.float32)


def test_per_tensor_quant_dequant_zero_passthrough():
    """Quantizing all-zeros and dequantizing returns all-zeros, no NaN.

    This is the key safety property when models have rare all-zero
    layers/heads after RMSNorm or particular pruning configurations.
    """
    v = torch.zeros(1, 8, 2, 16, dtype=torch.float16)
    s = compute_v_scales(v, ScaleScope.PER_TENSOR)
    q = quantize_v_fp8(v, s, ScaleScope.PER_TENSOR)
    dq = dequantize_v_fp8(q, s, ScaleScope.PER_TENSOR, out_dtype=torch.float16)
    torch.testing.assert_close(dq, v)


def test_outlier_does_not_break_other_heads(fp8_max):
    """Per-layer-head scaling: a single outlier in one head must not
    affect quantization of OTHER heads.  This is the physical
    counterpart to the Qwen-fragility argument: per-tensor scaling
    fails because outliers destroy precision globally; per-head
    scaling localizes the damage."""
    n_heads = 4
    head_dim = 8
    v = torch.randn(1, 32, n_heads, head_dim, dtype=torch.float16) * 0.1
    # Inject one huge outlier into head 0
    v[0, 0, 0, 0] = 1000.0
    s = compute_v_scales(v, ScaleScope.PER_LAYER_HEAD)
    q = quantize_v_fp8(v, s, ScaleScope.PER_LAYER_HEAD)
    dq = dequantize_v_fp8(q, s, ScaleScope.PER_LAYER_HEAD, out_dtype=torch.float32)
    # Heads 1, 2, 3 should reconstruct cleanly (within FP8 noise).
    for h in (1, 2, 3):
        clean_v = v[0, :, h, :].to(torch.float32)
        clean_dq = dq[0, :, h, :]
        # FP8 e4m3 has ~6.3% rounding worst-case bound; allow a
        # bit more for tiny values.
        max_err = (clean_dq - clean_v).abs().max() / max(
            float(clean_v.abs().max()), 1e-6
        )
        assert max_err < 0.10, (h, max_err.item())


def test_per_tensor_outlier_destroys_bulk(fp8_max):
    """Per-tensor scaling does worse on bulk precision when a single
    outlier exists than per-head scaling does on the same data.

    Use a deterministic structured pattern so the assertion is not
    flaky: bulk values at exactly +/-0.05 in heads 1-3, outlier of
    1000.0 in head 0.  Per-tensor scale becomes ~1000/qmax, so the
    per-tensor quant grid step around 0.05 is much coarser than the
    per-head grid step (which sees max ~0.05 in heads 1-3).
    """
    n_heads = 4
    head_dim = 8
    v = torch.full((1, 8, n_heads, head_dim), 0.05, dtype=torch.float16)
    # Alternate sign on bulk values to exercise both halves of FP8.
    v[..., 1::2] = -0.05
    # Outlier in head 0 only.
    v[0, 0, 0, 0] = 1000.0

    s_pt = compute_v_scales(v, ScaleScope.PER_TENSOR)
    s_ph = compute_v_scales(v, ScaleScope.PER_LAYER_HEAD, head_axis=2)
    q_pt = quantize_v_fp8(v, s_pt, ScaleScope.PER_TENSOR)
    q_ph = quantize_v_fp8(v, s_ph, ScaleScope.PER_LAYER_HEAD, head_axis=2)
    dq_pt = dequantize_v_fp8(q_pt, s_pt, ScaleScope.PER_TENSOR, out_dtype=torch.float32)
    dq_ph = dequantize_v_fp8(q_ph, s_ph, ScaleScope.PER_LAYER_HEAD, head_axis=2, out_dtype=torch.float32)
    # Bulk: heads 1, 2, 3.
    bulk_v = v[0, :, 1:, :].to(torch.float32)
    bulk_dq_pt = dq_pt[0, :, 1:, :]
    bulk_dq_ph = dq_ph[0, :, 1:, :]
    err_pt = (bulk_dq_pt - bulk_v).abs().max().item()
    err_ph = (bulk_dq_ph - bulk_v).abs().max().item()
    # Per-head must be strictly better on bulk.
    assert err_ph < err_pt, (err_ph, err_pt)


def test_quant_saturates_at_fp8_max(fp8_max):
    """Catches: overflow to ±inf or wrap on values > fp8 max."""
    v = torch.tensor([[float(fp8_max) * 4]], dtype=torch.float16)
    # scale=1.0 forces saturation rather than rescaling.
    s = torch.tensor(1.0)
    q = quantize_v_fp8(
        v.unsqueeze(0).unsqueeze(0), s, ScaleScope.PER_TENSOR
    )
    # FP8 should saturate to ±finfo.max, never inf or nan.
    assert torch.isfinite(q.to(torch.float32)).all()
    assert q.to(torch.float32).abs().max().item() <= fp8_max + 1e-3


def test_quant_idempotent_after_first_pass():
    """Catches: re-quantizing an already-FP8 tensor changes its bits."""
    v = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    s = compute_v_scales(v, ScaleScope.PER_TENSOR)
    q1 = quantize_v_fp8(v, s, ScaleScope.PER_TENSOR)
    # Round-trip dequant -> quant with the same scale.
    dq = dequantize_v_fp8(q1, s, ScaleScope.PER_TENSOR, out_dtype=torch.float32)
    q2 = quantize_v_fp8(dq, s, ScaleScope.PER_TENSOR)
    # Bit-equality on the FP8 dtype.
    assert torch.equal(
        q1.view(torch.uint8), q2.view(torch.uint8)
    ), "second-pass quantization changed bits"
