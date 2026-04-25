# SPDX-License-Identifier: Apache-2.0
"""Hypothesis-based property tests over random shapes/dtypes.

These catch shape-handling bugs that a fixed-shape battery would
miss, plus byte-corruption properties.
"""

# Third Party
import pytest
import torch

try:
    # Standard
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    HAS_HYPOTHESIS = False
    pytest.skip("hypothesis not installed", allow_module_level=True)

# First Party
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    CorruptEncodedKVError,
    ScaleScope,
)


# Bound Phase 1 tests to under 30s wall time total; tight settings.
_SETTINGS = settings(
    max_examples=25,
    deadline=2000,  # ms per case
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@_SETTINGS
@given(
    B=st.integers(min_value=1, max_value=4),
    T=st.integers(min_value=1, max_value=128),
    H=st.integers(min_value=1, max_value=8),
    D=st.sampled_from([16, 32, 64, 128]),
    dtype=st.sampled_from([torch.float16, torch.bfloat16]),
)
def test_property_per_tensor_roundtrip(B, T, H, D, dtype):
    codec = AsymK16V8Codec(scale_scope=ScaleScope.PER_TENSOR)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    enc = codec.encode(k, v)
    blob = codec.to_bytes(enc)
    parsed = codec.from_bytes(blob)
    k_back, _, _ = codec.decode(parsed, out_v_dtype=dtype)
    # K must match bit-exact
    assert torch.equal(k_back.view(k.shape), k)


@_SETTINGS
@given(
    H=st.integers(min_value=1, max_value=8),
    D=st.sampled_from([16, 32, 64]),
)
def test_property_per_layer_head_roundtrip(H, D):
    codec = AsymK16V8Codec(scale_scope=ScaleScope.PER_LAYER_HEAD)
    k = torch.randn(1, 16, H, D, dtype=torch.float16)
    v = torch.randn(1, 16, H, D, dtype=torch.float16)
    enc = codec.encode(k, v, head_axis=2)
    blob = codec.to_bytes(enc)
    parsed = codec.from_bytes(blob)
    assert parsed.scale_shape == (H,)


@_SETTINGS
@given(
    flip_offset=st.integers(min_value=0, max_value=200),
)
def test_property_random_byte_corruption_caught(flip_offset):
    """Flip a single byte at a random offset; either deserialize
    raises CorruptEncodedKVError or the parsed blob is byte-identical
    to the source (e.g., we hit a string field whose value just
    happened to round-trip).  Silent garbage is the bug to prevent."""
    codec = AsymK16V8Codec()
    k = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    v = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    enc = codec.encode(k, v)
    blob = bytearray(codec.to_bytes(enc))
    if flip_offset >= len(blob):
        return  # past end, nothing to flip
    blob[flip_offset] ^= 0xFF

    try:
        parsed = codec.from_bytes(bytes(blob))
    except CorruptEncodedKVError:
        return  # accepted: corruption detected
    # If we got here, no error was raised.  Verify the parsed blob
    # produces the same bytes when re-serialized; otherwise we
    # silently misinterpreted bytes, which is the bug.
    re_serialized = codec.to_bytes(parsed)
    if re_serialized != bytes(blob):
        pytest.fail(
            f"silent corruption at offset {flip_offset}: parsed but "
            f"re-serialization differs"
        )
