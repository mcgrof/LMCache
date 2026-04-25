# SPDX-License-Identifier: Apache-2.0
"""End-to-end codec encode -> bytes -> decode tests.

These ensure the full pipeline (compute scales, quantize V, pack
header, write payload, read back, dequantize) preserves K bit-exact
and V within FP8 noise.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    EncodedKV,
    ScaleScope,
    UnsupportedConfigError,
)


@pytest.fixture
def codec():
    return AsymK16V8Codec(scale_scope=ScaleScope.PER_TENSOR)


def test_k_returned_bit_exact(codec, small_kv):
    k, v = small_kv
    enc = codec.encode(k, v)
    blob = codec.to_bytes(enc)
    parsed = codec.from_bytes(blob)
    k_back, _, _ = codec.decode(parsed)
    assert k_back.dtype == k.dtype
    assert torch.equal(k_back, k.view(k_back.shape))


def test_v_returns_fp8_by_default(codec, small_kv):
    """Default decode (no out_v_dtype) returns V at native FP8 dtype.
    This is the native_asym path: caller chooses to dequantize or not.
    """
    k, v = small_kv
    enc = codec.encode(k, v)
    blob = codec.to_bytes(enc)
    parsed = codec.from_bytes(blob)
    _, v_back, _ = codec.decode(parsed)
    assert v_back.dtype == torch.float8_e4m3fn


def test_v_dequantizes_when_requested(codec, small_kv):
    k, v = small_kv
    enc = codec.encode(k, v)
    blob = codec.to_bytes(enc)
    parsed = codec.from_bytes(blob)
    _, v_back, _ = codec.decode(parsed, out_v_dtype=torch.float16)
    assert v_back.dtype == torch.float16
    # Reshape v_back to v shape for comparison
    v_back = v_back.view(v.shape)
    rel = (v_back.to(torch.float32) - v.to(torch.float32)).abs() / (
        v.abs().to(torch.float32) + 1e-6
    )
    assert rel.median().item() < 0.075


def test_scales_returned(codec, small_kv):
    k, v = small_kv
    enc = codec.encode(k, v)
    blob = codec.to_bytes(enc)
    parsed = codec.from_bytes(blob)
    _, _, scales_back = codec.decode(parsed)
    assert scales_back.dtype == torch.float32
    # Per-tensor scope -> 0-dim scale
    assert scales_back.shape == ()


def test_per_layer_head_scope_roundtrip():
    codec = AsymK16V8Codec(scale_scope=ScaleScope.PER_LAYER_HEAD)
    k = torch.randn(1, 8, 4, 16, dtype=torch.float16)
    v = torch.randn(1, 8, 4, 16, dtype=torch.float16)
    enc = codec.encode(k, v, head_axis=2)
    blob = codec.to_bytes(enc)
    parsed = codec.from_bytes(blob)
    assert parsed.scale_shape == (4,)


def test_e5m2_reserved_not_implemented():
    """e5m2 is reserved as a future format flag, not implemented in v1."""
    with pytest.raises(UnsupportedConfigError, match="e5m2"):
        AsymK16V8Codec(fp8_dtype=torch.float8_e5m2)


def test_invalid_fp8_dtype_rejected():
    with pytest.raises(UnsupportedConfigError):
        AsymK16V8Codec(fp8_dtype=torch.float32)


def test_kv_shape_mismatch_rejected(codec):
    k = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    v = torch.randn(1, 8, 4, 16, dtype=torch.float16)  # different head count
    with pytest.raises(UnsupportedConfigError, match="shapes must match"):
        codec.encode(k, v)


def test_payload_layout_is_K_V_scales(codec, small_kv):
    """Catches: payload layout reordered, breaking on-disk compat."""
    k, v = small_kv
    enc = codec.encode(k, v)
    # Check declared lengths sum to actual payload
    assert (
        enc.k_payload_len + enc.v_payload_len + enc.scale_payload_len
        == len(enc.payload)
    )
    # K bytes are first; same byte count as numel * itemsize
    expected_k = k.numel() * k.element_size()
    assert enc.k_payload_len == expected_k
    # V bytes are 1 byte/elem (FP8 e4m3fn)
    assert enc.v_payload_len == v.numel()


def test_total_bytes_property(codec, small_kv):
    k, v = small_kv
    enc = codec.encode(k, v)
    blob = codec.to_bytes(enc)
    assert enc.total_bytes == len(blob)


def test_native_asym_passthrough_uses_precomputed_quant(codec, small_kv):
    """The native_asym path: caller already has V as FP8.  Codec
    must not re-quantize."""
    k, v = small_kv
    # Manually quantize V then pass it back in
    # First Party
    from lmcache.v1.kv_codec import compute_v_scales, quantize_v_fp8
    scales = compute_v_scales(v, ScaleScope.PER_TENSOR)
    v_q = quantize_v_fp8(v, scales, ScaleScope.PER_TENSOR)
    # Encode with precomputed; codec must accept and use as-is
    enc = codec.encode(
        k, v,
        precomputed_v_quant=v_q,
        precomputed_v_scales=scales,
    )
    blob = codec.to_bytes(enc)
    parsed = codec.from_bytes(blob)
    _, v_back, _ = codec.decode(parsed)
    # v_back is FP8 read back; should bit-equal v_q
    assert torch.equal(
        v_back.view(v_q.shape).view(torch.uint8),
        v_q.view(torch.uint8),
    )


def test_precomputed_quant_without_scales_rejected(codec, small_kv):
    k, v = small_kv
    # First Party
    from lmcache.v1.kv_codec import compute_v_scales, quantize_v_fp8
    scales = compute_v_scales(v, ScaleScope.PER_TENSOR)
    v_q = quantize_v_fp8(v, scales, ScaleScope.PER_TENSOR)
    with pytest.raises(UnsupportedConfigError, match="precomputed_v_scales"):
        codec.encode(k, v, precomputed_v_quant=v_q)


def test_precomputed_quant_wrong_dtype_rejected(codec, small_kv):
    """Catches: caller passed an FP16 tensor as precomputed_v_quant."""
    k, v = small_kv
    fake_q = v.to(torch.float16)  # not FP8
    fake_s = torch.tensor(0.1)
    with pytest.raises(UnsupportedConfigError, match="dtype"):
        codec.encode(
            k, v,
            precomputed_v_quant=fake_q,
            precomputed_v_scales=fake_s,
        )
