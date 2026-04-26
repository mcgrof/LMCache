# SPDX-License-Identifier: Apache-2.0
"""vLLM connector glue: build_asym_kv_view tests.

These tests use mocked vLLM attention layers to exercise the
`build_asym_kv_view` helper without needing a running vLLM
process or GPU.  They validate the contract:

- Returns None on symmetric vLLM (caller falls back to FP16 path).
- Returns AsymKVView on asymmetric vLLM with both kv_layer-tuple
  and layer.kv_cache-attribute extraction modes.
- Surfaces `_v_scale_float` (per-tensor scalar) and `_v_scale`
  (per-head tensor) variants correctly.
- Refuses to build a view on dtype mismatches and logs a warning
  rather than crashing the forward pass (when strict=False).
"""

# Standard
from dataclasses import dataclass
from typing import Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.asym_kv_view_builder import (
    build_asym_kv_view,
)
from lmcache.v1.storage_backend.naive_serde.asym_serde import AsymKVView


# Helper: a vLLM Attention stand-in.
class _SymmetricAttention:
    kv_cache_dtype = "fp8_e4m3"  # string, not tuple


class _AsymmetricAttention:
    kv_cache_dtype = ("auto", "fp8_e4m3")
    _v_scale_float: float = 0.234

    def __init__(self, kv_cache=None, v_scale_tensor=None):
        if kv_cache is not None:
            self.kv_cache = kv_cache
        if v_scale_tensor is not None:
            self._v_scale = v_scale_tensor


def _kv_buffers(shape=(4, 16, 8, 64)):
    """Build (K, V_fp8) buffers as the asymmetric vLLM branch would."""
    g = torch.Generator()
    g.manual_seed(7)
    k = torch.randn(*shape, dtype=torch.float16, generator=g)
    v_fp8 = torch.randn(*shape, dtype=torch.float16, generator=g).to(
        torch.float8_e4m3fn
    )
    return k, v_fp8


def test_symmetric_vllm_returns_None():
    """The capability detector says no; helper must NOT pretend
    it can build an asym view."""
    attn = _SymmetricAttention()
    k, v = _kv_buffers()
    out = build_asym_kv_view(attn, kv_layer=(k, v))
    assert out is None


def test_asymmetric_with_tuple_kv_layer():
    """Tuple kv_layer is the most common asymmetric form."""
    attn = _AsymmetricAttention()
    k, v_fp8 = _kv_buffers()
    out = build_asym_kv_view(attn, kv_layer=(k, v_fp8))
    assert isinstance(out, AsymKVView)
    assert out.k.dtype == torch.float16
    assert out.v_fp8.dtype == torch.float8_e4m3fn
    # Per-tensor scalar from _v_scale_float reshaped to ()
    assert out.v_scales.dtype == torch.float32
    assert out.v_scales.shape == ()
    assert float(out.v_scales) == pytest.approx(0.234)


def test_asymmetric_with_layer_kv_cache_attribute():
    """When kv_layer isn't a tuple but the layer exposes
    `attention_layer.kv_cache`, helper falls back to that."""
    k, v_fp8 = _kv_buffers()
    attn = _AsymmetricAttention(kv_cache=(k, v_fp8))
    # kv_layer is a single tensor (which we'd otherwise reject); the
    # layer.kv_cache attribute is the actual source.
    fake_kv_layer = torch.zeros(1)  # placeholder
    out = build_asym_kv_view(attn, kv_layer=fake_kv_layer)
    assert isinstance(out, AsymKVView)
    assert torch.equal(out.k, k)


def test_v_scale_tensor_takes_precedence_over_float():
    """Per-head `_v_scale` tensor is preferred over scalar
    `_v_scale_float`."""
    k, v_fp8 = _kv_buffers(shape=(4, 16, 8, 64))
    n_heads = 8
    per_head_scales = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    attn = _AsymmetricAttention(
        v_scale_tensor=per_head_scales.clone()
    )
    out = build_asym_kv_view(attn, kv_layer=(k, v_fp8))
    assert isinstance(out, AsymKVView)
    assert out.v_scales.dtype == torch.float32
    assert out.v_scales.shape == (n_heads,)
    torch.testing.assert_close(out.v_scales, per_head_scales)


def test_K_dtype_mismatch_returns_None():
    """Asymmetric branch should produce K at FP16/BF16; if it
    ever produces K at a different dtype we must not silently
    pretend it works.  strict=False -> warning + None."""
    attn = _AsymmetricAttention()
    g = torch.Generator()
    g.manual_seed(0)
    bad_k = torch.randn(4, 16, 8, 64, dtype=torch.float32, generator=g)
    v_fp8 = torch.zeros(4, 16, 8, 64, dtype=torch.float8_e4m3fn)
    out = build_asym_kv_view(attn, kv_layer=(bad_k, v_fp8))
    assert out is None


def test_K_dtype_mismatch_strict_raises():
    """In strict mode, build raises ValueError instead of
    returning None — useful in tests where the failure should
    not be silent."""
    attn = _AsymmetricAttention()
    bad_k = torch.zeros(4, 16, 8, 64, dtype=torch.float32)
    v_fp8 = torch.zeros(4, 16, 8, 64, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="K dtype"):
        build_asym_kv_view(attn, kv_layer=(bad_k, v_fp8), strict=True)


def test_V_dtype_mismatch_returns_None():
    """Asymmetric V must be FP8 e4m3."""
    attn = _AsymmetricAttention()
    k = torch.zeros(4, 16, 8, 64, dtype=torch.float16)
    bad_v = torch.zeros(4, 16, 8, 64, dtype=torch.float16)  # not FP8
    out = build_asym_kv_view(attn, kv_layer=(k, bad_v))
    assert out is None


def test_missing_v_scale_returns_None():
    """A capability-detected layer that lacks scales should fail
    cleanly; capability detection requires `_v_scale_float`, so
    this case shouldn't normally happen — defensive check."""

    class _BogusAsym:
        kv_cache_dtype = ("auto", "fp8_e4m3")
        # Note: capability detection requires _v_scale_float but the
        # extractor checks for either _v_scale or _v_scale_float.
        # Bogus class has neither.

    attn = _BogusAsym()
    # Force capability check to pass via direct access — in real
    # use, the capability detector would reject this layer up front.
    # First Party
    from lmcache.v1.storage_backend.naive_serde.asym_serde import (
        detect_native_asym_capability,
    )
    assert not detect_native_asym_capability(attn)
    out = build_asym_kv_view(attn, kv_layer=_kv_buffers())
    assert out is None  # capability rejected upstream


def test_no_kv_layer_no_layer_attribute_returns_None():
    """Helper handles malformed input by returning None (not
    raising) when strict=False."""
    attn = _AsymmetricAttention()
    out = build_asym_kv_view(attn, kv_layer=torch.zeros(1))
    assert out is None


def test_attn_metadata_is_optional():
    """attn_metadata is reserved for future use; current code
    must not require it."""
    attn = _AsymmetricAttention()
    k, v_fp8 = _kv_buffers()
    # No attn_metadata kwarg
    out = build_asym_kv_view(attn, kv_layer=(k, v_fp8))
    assert isinstance(out, AsymKVView)
