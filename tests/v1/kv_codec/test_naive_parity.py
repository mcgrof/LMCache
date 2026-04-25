# SPDX-License-Identifier: Apache-2.0
"""Degenerate-case parity tests.

When the codec is configured to "do nothing" (e.g., V already at
FP8 with unit scale, or all-zero V), the roundtrip should be
arithmetically pristine.  These catch accidental drift in the
codec where it adds noise that shouldn't be there.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    ScaleScope,
    compute_v_scales,
    dequantize_v_fp8,
    quantize_v_fp8,
)


def test_quant_unit_scale_is_truncation():
    """When scale=1.0, quant is just clamp+cast; no scaling math.

    Catches: scale path applied even when scale=1.0 (silent
    rescaling) or saturation point set wrong (overflow to inf/NaN).
    """
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    # Use values that are well-representable in FP8 e4m3 (powers of
    # two near 1.0, plus a saturation case).
    v = torch.tensor(
        [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, fp8_max * 4],
        dtype=torch.float32,
    )
    s = torch.tensor(1.0)
    q = quantize_v_fp8(v.unsqueeze(0).unsqueeze(0).unsqueeze(0), s, ScaleScope.PER_TENSOR)
    f = q.to(torch.float32).flatten()
    # Powers of two within ±fp8_max are exactly representable in
    # FP8 e4m3 — the first 7 values must round-trip exactly.
    for x_in, x_out in zip(v[:-1].tolist(), f[:-1].tolist()):
        assert x_out == pytest.approx(x_in, abs=1e-6), (x_in, x_out)
    # The saturation case clamps to fp8_max with sign preserved.
    assert f[-1].item() == pytest.approx(fp8_max, abs=1.0)


def test_zero_kv_roundtrip_bit_exact():
    """All-zero V roundtrips to all-zero, bit-exact (no NaN, no
    drift from sentinel-scale division)."""
    codec = AsymK16V8Codec()
    k = torch.zeros(1, 8, 2, 16, dtype=torch.float16)
    v = torch.zeros(1, 8, 2, 16, dtype=torch.float16)
    enc = codec.encode(k, v)
    blob = codec.to_bytes(enc)
    parsed = codec.from_bytes(blob)
    k_back, v_back, _ = codec.decode(parsed, out_v_dtype=torch.float16)
    assert torch.equal(k_back.view(k.shape), k)
    assert torch.equal(v_back.view(v.shape), v)


def test_codec_total_bytes_close_to_3_quarters_of_fp16():
    """Sanity check on the storage win: encoded size should be
    K_bytes(2/elem) + V_bytes(1/elem) + scale + header = ~3/4 of
    a hypothetical FP16 K+V serialization, plus a tiny header
    overhead.  Catches accidental dtype drift that would balloon
    the encoded size."""
    codec = AsymK16V8Codec()
    k = torch.randn(1, 1024, 8, 64, dtype=torch.float16)
    v = torch.randn(1, 1024, 8, 64, dtype=torch.float16)
    n_elems = k.numel()  # K and V have same numel
    fp16_total = 2 * n_elems * 2  # K(fp16) + V(fp16) = 4 bytes/elem
    enc = codec.encode(k, v)
    blob = codec.to_bytes(enc)
    encoded = len(blob)
    # Asymmetric: K(fp16=2) + V(fp8=1) = 3 bytes/elem = 0.75 * fp16_total
    expected_payload = 3 * n_elems
    # Header is small constant (~< 1 KB).  Allow up to 1 KB overhead.
    assert encoded < expected_payload + 1024, (encoded, expected_payload)
    assert encoded >= expected_payload, (encoded, expected_payload)
    # Storage ratio < 80% (we claim ~75% in the paper)
    ratio = encoded / fp16_total
    assert ratio < 0.80, ratio
