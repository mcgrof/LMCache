# SPDX-License-Identifier: Apache-2.0
"""Storage-only-dequant serde end-to-end tests.

These exercise the path:

    [2, ...] FP16 tensor
        -> AsymK16V8Serializer.serialize  -> BytesBufferMemoryObj
        -> bytes (would hit disk here)
        -> AsymK16V8Deserializer.deserialize -> [2, ...] FP16 tensor

without bringing up a real storage backend.  Backend wiring tests
that DO bring up local disk live in tests that import the storage
manager; those need the heavyweight conftest at the repo root.
The tests here are meant to run from the kv_codec confcutdir so
they don't pull in the full LMCache import graph.
"""

# Standard
from dataclasses import dataclass

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import CodecMismatchError
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.naive_serde.asym_serde import (
    AsymK16V8Deserializer,
    AsymK16V8Serializer,
)


# Tiny stand-ins so the tests don't need a real LMCacheEngine.
@dataclass
class _FakeMetadata:
    model_name: str = "qwen2.5-7b"


@dataclass
class _FakeConfig:
    chunk_size: int = 256


def _make_kv_tensor(shape=(2, 4, 16, 32), dtype=torch.float16, seed=0):
    """Make a [2, ...] KV-shaped tensor.  Default shape:
    [K/V split=2, num_layers=4, num_tokens=16, hidden=32]."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, generator=g)


def _to_memory_obj(tensor):
    meta = MemoryObjMetadata(
        shape=torch.Size(tensor.shape),
        dtype=tensor.dtype,
        address=0,
        phy_size=tensor.numel() * tensor.element_size(),
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(
        raw_data=tensor, metadata=meta, parent_allocator=None
    )


@pytest.fixture
def serde_pair():
    cfg = _FakeConfig()
    md = _FakeMetadata()
    return (AsymK16V8Serializer(cfg, md), AsymK16V8Deserializer(cfg, md))


def test_serialize_returns_bytes_buffer_memory_obj(serde_pair):
    """Catches: serializer returns the raw input or a TensorMemoryObj
    by mistake.  Disk path needs a BytesBufferMemoryObj."""
    ser, _ = serde_pair
    t = _make_kv_tensor()
    out = ser.serialize(_to_memory_obj(t))
    assert isinstance(out, BytesBufferMemoryObj)
    assert isinstance(out.raw_data, (bytes, bytearray))
    assert out.metadata.fmt == MemoryFormat.BINARY_BUFFER


def test_serialize_size_is_about_three_quarters_of_fp16(serde_pair):
    """Catches: V being written at FP16 instead of FP8 (would balloon
    encoded size to ~FP16 size + header).  Encoded payload should be
    ~75% of the FP16 K+V byte total + a small constant header."""
    ser, _ = serde_pair
    t = _make_kv_tensor(shape=(2, 8, 32, 64), dtype=torch.float16)
    n_elems_per_half = t[0].numel()
    fp16_total_payload = 2 * 2 * n_elems_per_half  # K(fp16) + V(fp16)
    asym_total_payload = 2 * n_elems_per_half + n_elems_per_half  # K(2B) + V(1B)

    out = ser.serialize(_to_memory_obj(t))
    encoded = len(out.raw_data)
    # Encoded size = header + K + V + scales.  Header is small const
    # (< 1 KB); scales are tiny for per_tensor scope.
    # Verify within a tight band of the K+V theoretical.
    assert encoded < fp16_total_payload, (encoded, fp16_total_payload)
    # 75% of fp16 + < 1024 byte header
    assert encoded - asym_total_payload < 1024, (encoded, asym_total_payload)


def test_off_by_25_percent_capacity_bug(serde_pair):
    """Test specifically for the off-by-25% capacity-accounting bug.

    If a backend computes asymmetric capacity using the legacy
    single-dtype size (treating V as FP16), it would double-count
    half the V bytes and report the encoded size as fp16-equivalent.
    This test fails loudly if the encoded MemoryObj reports its
    size as the original FP16 size.
    """
    ser, _ = serde_pair
    t = _make_kv_tensor(shape=(2, 8, 32, 64), dtype=torch.float16)
    fp16_size = t.numel() * 2
    out = ser.serialize(_to_memory_obj(t))
    # The encoded MemoryObj's get_size() reports actual bytes.  If
    # this still reports fp16_size, the storage manager would refuse
    # to count the disk-savings.
    encoded_size = out.get_size()
    # Asymmetric encoding writes ~75% of FP16 payload + small header.
    # Allow small slack (< 5% headroom for the header).
    assert encoded_size < int(fp16_size * 0.80), (encoded_size, fp16_size)


def test_roundtrip_returns_tensor_memory_obj(serde_pair):
    """Catches: deserializer returns the bytes object instead of
    materializing a tensor."""
    ser, des = serde_pair
    t = _make_kv_tensor()
    encoded = ser.serialize(_to_memory_obj(t))
    decoded = des.deserialize(encoded)
    assert isinstance(decoded, TensorMemoryObj)
    assert decoded.tensor.shape == t.shape
    assert decoded.tensor.dtype == t.dtype


def test_roundtrip_K_bit_exact(serde_pair):
    """Catches: K being inadvertently quantized.  Asymmetric is a
    K16/V8 codec; K must round-trip bit-exact."""
    ser, des = serde_pair
    t = _make_kv_tensor()
    encoded = ser.serialize(_to_memory_obj(t))
    decoded = des.deserialize(encoded)
    assert torch.equal(decoded.tensor[0], t[0]), "K is not bit-exact"


def test_roundtrip_V_within_fp8_noise(serde_pair):
    """Catches: V scale arithmetic broken — observed error far above
    FP8 e4m3 rounding bound."""
    ser, des = serde_pair
    t = _make_kv_tensor(shape=(2, 4, 64, 32))
    encoded = ser.serialize(_to_memory_obj(t))
    decoded = des.deserialize(encoded)
    v_in = t[1].to(torch.float32)
    v_out = decoded.tensor[1].to(torch.float32)
    rel = (v_out - v_in).abs() / (v_in.abs() + 1e-6)
    assert rel.median().item() < 0.075, rel.median()


def test_zero_kv_roundtrip_bit_exact(serde_pair):
    """All-zero KV must round-trip bit-exact (no NaN from zero scales)."""
    ser, des = serde_pair
    t = torch.zeros(2, 4, 16, 32, dtype=torch.float16)
    encoded = ser.serialize(_to_memory_obj(t))
    decoded = des.deserialize(encoded)
    assert torch.equal(decoded.tensor, t)


def test_serializer_rejects_non_kv_split_tensor(serde_pair):
    """Catches: serde quietly accepting a non-[2,...] tensor."""
    ser, _ = serde_pair
    bad = torch.randn(4, 16, 32, dtype=torch.float16)  # leading dim 4
    with pytest.raises(ValueError, match="leading dim 2"):
        ser.serialize(_to_memory_obj(bad))


def test_serializer_rejects_quantized_input(serde_pair):
    """Catches: serde fed an already-FP8 tensor in storage-only mode.
    Storage-only is the dequantize-on-read path; the input must be
    FP16/BF16."""
    ser, _ = serde_pair
    fp8 = torch.zeros(2, 4, 16, 32, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="not FP16 or BF16"):
        ser.serialize(_to_memory_obj(fp8))


def test_deserializer_rejects_non_bytes_input(serde_pair):
    """Catches: deserializer fed a TensorMemoryObj by mistake."""
    _, des = serde_pair
    t = _make_kv_tensor()
    with pytest.raises(ValueError, match="BytesBufferMemoryObj"):
        des.deserialize(_to_memory_obj(t))


def test_cross_model_read_rejected():
    """Write under model A, read under model B — must raise
    CodecMismatchError, not silently return wrong-distribution data."""
    cfg = _FakeConfig()
    ser_a = AsymK16V8Serializer(cfg, _FakeMetadata(model_name="qwen2.5-7b"))
    des_b = AsymK16V8Deserializer(cfg, _FakeMetadata(model_name="llama-3.1-8b"))
    t = _make_kv_tensor()
    encoded = ser_a.serialize(_to_memory_obj(t))
    with pytest.raises(CodecMismatchError, match="model_id"):
        des_b.deserialize(encoded)


def test_native_asym_runtime_layout_not_yet_implemented():
    """Phase 2 ships only storage_only_dequant; native_asym must
    fail loud, not silently fall back to FP16."""
    with pytest.raises(NotImplementedError, match="Phase 4"):
        AsymK16V8Serializer(
            _FakeConfig(),
            _FakeMetadata(),
            runtime_layout="native_asym",
        )


def test_unknown_runtime_layout_rejected():
    with pytest.raises(ValueError, match="unknown runtime_layout"):
        AsymK16V8Serializer(
            _FakeConfig(),
            _FakeMetadata(),
            runtime_layout="bogus",
        )


def test_bfloat16_roundtrip(serde_pair):
    """BF16 K must also be bit-exact (different dtype than fp16 but
    same byte size)."""
    ser, des = serde_pair
    t = _make_kv_tensor(dtype=torch.bfloat16)
    encoded = ser.serialize(_to_memory_obj(t))
    decoded = des.deserialize(encoded)
    assert decoded.tensor.dtype == torch.bfloat16
    assert torch.equal(decoded.tensor[0], t[0])
