# SPDX-License-Identifier: Apache-2.0
"""Phase 4 (CPU portion): native_asym passthrough tests.

The GPU-side memory-snapshot test (asserting no full-size FP16 V
allocation on the hit path) lives in test_native_asym_gpu.py and is
gated on CUDA + the asymmetric vLLM branch.  These CPU tests cover
the contract: scales preserved exactly through the codec, mismatch
detection, multi-layer plumbing, and the AsymKVMemoryObj
non-tensor-ness that prevents accidental dequant.
"""

# Standard
from dataclasses import dataclass

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    CodecMismatchError,
    ScaleScope,
    compute_v_scales,
    quantize_v_fp8,
)
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.naive_serde.asym_serde import (
    AsymK16V8Deserializer,
    AsymK16V8Serializer,
    AsymKVMemoryObj,
    AsymKVView,
    detect_native_asym_capability,
)


@dataclass
class _FakeMetadata:
    model_name: str = "qwen2.5-7b"


@dataclass
class _FakeConfig:
    chunk_size: int = 256


# A MemoryObj-shaped stand-in that carries a `.asym_view`.  This is
# what the GPU connector will hand the serializer in real use.
class _AsymInputMemoryObj:
    def __init__(self, view: AsymKVView, original_shape):
        self.asym_view = view
        self.tensor = None
        self.metadata = MemoryObjMetadata(
            shape=torch.Size([0, 0, 0, 0]),
            dtype=None,
            address=0,
            phy_size=0,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.UNDEFINED,
            shapes=[torch.Size(original_shape)],
            dtypes=[view.k.dtype],
        )


def _make_asym_view(shape=(4, 16, 32), dtype=torch.float16, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    k = torch.randn(*shape, dtype=dtype, generator=g)
    v_fp16 = torch.randn(*shape, dtype=dtype, generator=g)
    scales = compute_v_scales(v_fp16, ScaleScope.PER_TENSOR).to(torch.float32)
    v_fp8 = quantize_v_fp8(v_fp16, scales, ScaleScope.PER_TENSOR)
    return AsymKVView(k=k, v_fp8=v_fp8, v_scales=scales)


@pytest.fixture
def native_serde_pair():
    cfg = _FakeConfig()
    md = _FakeMetadata()
    return (
        AsymK16V8Serializer(cfg, md, runtime_layout="native_asym"),
        AsymK16V8Deserializer(cfg, md, runtime_layout="native_asym"),
    )


def test_native_asym_serialize_accepts_asym_view(native_serde_pair):
    """Catches: native_asym serializer fails to read the asym_view
    attribute and falls back to the storage_only path."""
    ser, _ = native_serde_pair
    view = _make_asym_view()
    inp = _AsymInputMemoryObj(view, original_shape=view.k.shape)
    out = ser.serialize(inp)
    assert isinstance(out, BytesBufferMemoryObj)
    assert out.metadata.fmt == MemoryFormat.BINARY_BUFFER


def test_native_asym_serialize_without_view_rejected(native_serde_pair):
    """Catches: serializer silently quantizes V from an FP16 tensor
    when caller meant to use native_asym."""
    ser, _ = native_serde_pair
    # Pass a "regular" memory object with a tensor but no asym_view
    t = torch.randn(2, 4, 16, 32, dtype=torch.float16)
    meta = MemoryObjMetadata(
        shape=torch.Size(t.shape),
        dtype=t.dtype,
        address=0,
        phy_size=t.numel() * t.element_size(),
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    obj = TensorMemoryObj(raw_data=t, metadata=meta, parent_allocator=None)
    with pytest.raises(ValueError, match="asym_view"):
        ser.serialize(obj)


def test_native_asym_roundtrip_returns_AsymKVMemoryObj(native_serde_pair):
    """Catches: deserializer returns a regular MemoryObj which would
    silently dequant V."""
    ser, des = native_serde_pair
    view = _make_asym_view()
    inp = _AsymInputMemoryObj(view, original_shape=view.k.shape)
    encoded = ser.serialize(inp)
    decoded = des.deserialize(encoded)
    assert isinstance(decoded, AsymKVMemoryObj)
    assert decoded.tensor is None, (
        "AsymKVMemoryObj.tensor must be None to prevent silent dequant"
    )
    assert decoded.k.dtype == torch.float16
    assert decoded.v_fp8.dtype == torch.float8_e4m3fn


def test_native_asym_K_bit_exact(native_serde_pair):
    """K must round-trip bit-exact in native_asym, just like
    storage_only."""
    ser, des = native_serde_pair
    view = _make_asym_view()
    inp = _AsymInputMemoryObj(view, original_shape=view.k.shape)
    encoded = ser.serialize(inp)
    decoded = des.deserialize(encoded)
    assert torch.equal(decoded.k, view.k)


def test_native_asym_V_bit_exact_no_requantization(native_serde_pair):
    """The whole point of native_asym: V comes back at the SAME FP8
    bytes that went in.  No re-quantization, no rounding drift."""
    ser, des = native_serde_pair
    view = _make_asym_view()
    inp = _AsymInputMemoryObj(view, original_shape=view.k.shape)
    encoded = ser.serialize(inp)
    decoded = des.deserialize(encoded)
    # Bit-equality on FP8 dtype.
    assert torch.equal(
        decoded.v_fp8.view(torch.uint8),
        view.v_fp8.view(torch.uint8),
    ), "V FP8 bytes changed on round-trip — re-quantization happened!"


def test_native_asym_scales_preserved_exactly(native_serde_pair):
    """Catches: rescaling drift on the read path.  Scales must come
    back bit-equal so the runtime computes attention against the
    same scales the writer used."""
    ser, des = native_serde_pair
    view = _make_asym_view()
    inp = _AsymInputMemoryObj(view, original_shape=view.k.shape)
    encoded = ser.serialize(inp)
    decoded = des.deserialize(encoded)
    torch.testing.assert_close(decoded.v_scales, view.v_scales, atol=0, rtol=0)


def test_native_asym_get_size_excludes_fp16_V_bytes(native_serde_pair):
    """The decoded MemoryObj's get_size() must reflect FP8 V bytes
    (1 byte/elem), not the dequantized FP16 size (2 bytes/elem).
    Catches: capacity accounting that double-counts V."""
    ser, des = native_serde_pair
    view = _make_asym_view(shape=(8, 32, 64))
    inp = _AsymInputMemoryObj(view, original_shape=view.k.shape)
    encoded = ser.serialize(inp)
    decoded = des.deserialize(encoded)
    expected_min = (
        view.k.numel() * 2          # K at 2 B/elem (fp16)
        + view.v_fp8.numel() * 1    # V at 1 B/elem (fp8)
    )
    # Add a small allowance for the scales tensor (4 bytes here).
    actual = decoded.get_size()
    assert actual >= expected_min
    # MUST NOT report FP16-V byte count.
    fp16_v_total = view.k.numel() * 2 + view.v_fp8.numel() * 2
    assert actual < fp16_v_total, (actual, fp16_v_total)


def test_native_asym_multilayer_plumbing(native_serde_pair):
    """16 layers, each with different per-tensor scales.  All round
    trip independently with the right scale-to-layer mapping.
    Catches: scales accidentally shared/cached across layers."""
    ser, des = native_serde_pair
    n_layers = 16
    layers = [_make_asym_view(seed=i) for i in range(n_layers)]
    encoded = []
    for v in layers:
        inp = _AsymInputMemoryObj(v, original_shape=v.k.shape)
        encoded.append(ser.serialize(inp))
    # Read in shuffled order.
    order = [9, 2, 14, 0, 11, 5, 1, 7, 12, 3, 6, 10, 13, 8, 4, 15]
    for i in order:
        out = des.deserialize(encoded[i])
        torch.testing.assert_close(out.v_scales, layers[i].v_scales, atol=0, rtol=0)
        assert torch.equal(
            out.v_fp8.view(torch.uint8),
            layers[i].v_fp8.view(torch.uint8),
        )


def test_capability_detection_symmetric_vllm():
    """Catches: capability check returns True for a symmetric vLLM,
    which would silently fall back to storage_only behavior under a
    native_asym config."""
    # Symmetric vLLM mock: kv_cache_dtype is a string, no _v_scale_float
    class _SymAttention:
        kv_cache_dtype = "fp8_e4m3"

    assert detect_native_asym_capability(_SymAttention()) is False


def test_capability_detection_asymmetric_vllm():
    """Asymmetric vLLM mock: kv_cache_dtype is a tuple,
    _v_scale_float exists."""

    class _AsymAttention:
        kv_cache_dtype = ("auto", "fp8_e4m3")
        _v_scale_float = 0.123

    assert detect_native_asym_capability(_AsymAttention()) is True


def test_capability_detection_partial_asymmetric_rejected():
    """A vLLM that has kv_cache_dtype tuple but missing _v_scale_float
    is broken: don't claim native_asym capability."""

    class _Partial:
        kv_cache_dtype = ("auto", "fp8_e4m3")
        # no _v_scale_float

    assert detect_native_asym_capability(_Partial()) is False


def test_capability_detection_no_kv_cache_dtype():
    """A bare object with no kv_cache_dtype attribute returns False."""

    class _Bare:
        pass

    assert detect_native_asym_capability(_Bare()) is False


def test_native_asym_cross_codec_mismatch_rejected():
    """Native asym is also subject to cross-model gating."""
    cfg = _FakeConfig()
    ser_a = AsymK16V8Serializer(
        cfg,
        _FakeMetadata(model_name="qwen2.5-7b"),
        runtime_layout="native_asym",
    )
    des_b = AsymK16V8Deserializer(
        cfg,
        _FakeMetadata(model_name="llama-3.1-8b"),
        runtime_layout="native_asym",
    )
    view = _make_asym_view()
    inp = _AsymInputMemoryObj(view, original_shape=view.k.shape)
    encoded = ser_a.serialize(inp)
    with pytest.raises(CodecMismatchError, match="model_id"):
        des_b.deserialize(encoded)


def test_native_asym_size_matches_storage_only_size(native_serde_pair):
    """Encoded bytes for the same K/V should be identical between
    storage_only and native_asym (ignoring header noise) — they
    write the same payload, just from different sources."""
    ser_native, _ = native_serde_pair
    cfg = _FakeConfig()
    md = _FakeMetadata()
    ser_storage = AsymK16V8Serializer(cfg, md, runtime_layout="storage_only_dequant")

    view = _make_asym_view()
    # Storage-only path: feed FP16 K and V in a [2, ...] tensor,
    # serializer quantizes V internally.
    full_kv = torch.stack(
        [view.k, view.v_fp8.to(torch.float32) * view.v_scales],
        dim=0,
    ).to(torch.float16)
    meta = MemoryObjMetadata(
        shape=torch.Size(full_kv.shape),
        dtype=full_kv.dtype,
        address=0,
        phy_size=full_kv.numel() * full_kv.element_size(),
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    storage_input = TensorMemoryObj(
        raw_data=full_kv, metadata=meta, parent_allocator=None
    )
    storage_blob = ser_storage.serialize(storage_input).raw_data
    native_input = _AsymInputMemoryObj(view, original_shape=view.k.shape)
    native_blob = ser_native.serialize(native_input).raw_data
    # Sizes should match within a tiny constant (header CRC of the
    # different V payloads differs; lengths are the same).
    assert len(storage_blob) == len(native_blob), (
        len(storage_blob), len(native_blob)
    )
