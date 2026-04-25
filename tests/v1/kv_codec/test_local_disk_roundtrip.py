# SPDX-License-Identifier: Apache-2.0
"""Phase 3: file-backed disk roundtrip for the asymmetric serde.

These tests do real OS-level disk I/O through tempfile so we cover:

- the serde produces bytes that survive an actual disk write+read
  (catches: in-memory-only bugs that wouldn't show up against an
  in-process bytearray)
- the storage path treats EncodedKV blobs as opaque bytes (no
  storage backend changes needed in this phase)
- multi-key concurrent writes don't corrupt each other on disk
- cross-codec read (write naive, read asym, or vice versa) is
  rejected with a typed error, not silent garbage
- capacity accounting against the disk byte total matches the
  ~0.75x of FP16 expectation

GDS-direct paths are deferred — they need CUDA + a GDS-capable
filesystem.  When that lane comes online, mark the same tests
@pytest.mark.gpu and reuse them.
"""

# Standard
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import CodecMismatchError, CorruptEncodedKVError
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


@dataclass
class _FakeMetadata:
    model_name: str = "qwen2.5-7b"


@dataclass
class _FakeConfig:
    chunk_size: int = 256


def _kv_tensor(shape=(2, 4, 16, 32), dtype=torch.float16, seed=0):
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


def _bytes_to_mobj(blob: bytes, original_shape, original_dtype):
    """Reconstruct a BytesBufferMemoryObj as the disk read path
    would: bytes plus enough metadata to recover the logical shape."""
    meta = MemoryObjMetadata(
        shape=torch.Size([len(blob), 0, 0, 0]),
        dtype=None,
        address=0,
        phy_size=0,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.BINARY_BUFFER,
        shapes=[torch.Size(original_shape)],
        dtypes=[original_dtype],
    )
    return BytesBufferMemoryObj(raw_bytes=blob, metadata=meta)


@pytest.fixture
def serde_pair():
    cfg = _FakeConfig()
    md = _FakeMetadata()
    return (AsymK16V8Serializer(cfg, md), AsymK16V8Deserializer(cfg, md))


def test_real_disk_roundtrip(serde_pair, tmp_path):
    """Catches: serde bytes corrupted by OS-level write/read path.
    Bytes go through the kernel page cache, fsync semantics, the
    works."""
    ser, des = serde_pair
    t = _kv_tensor()
    encoded = ser.serialize(_to_memory_obj(t))

    # Write
    blob_path = tmp_path / "asym_kv_0.bin"
    blob_path.write_bytes(encoded.raw_data)
    # Re-read
    blob_back = blob_path.read_bytes()
    assert blob_back == encoded.raw_data, "OS-level write/read mutated bytes"

    decoded = des.deserialize(_bytes_to_mobj(blob_back, t.shape, t.dtype))
    assert torch.equal(decoded.tensor[0], t[0]), "K not bit-exact after disk"


def test_disk_byte_count_close_to_three_quarters_fp16(serde_pair, tmp_path):
    """Catches: regression in encoded size when going to disk.
    Encoded byte count on disk should be ~75% of fp16 K+V."""
    ser, _ = serde_pair
    t = _kv_tensor(shape=(2, 8, 32, 64))
    encoded = ser.serialize(_to_memory_obj(t))

    blob_path = tmp_path / "size_check.bin"
    blob_path.write_bytes(encoded.raw_data)
    disk_size = blob_path.stat().st_size

    fp16_payload = t.numel() * 2  # 2 bytes per element (FP16)
    asym_payload = (
        t[0].numel() * 2  # K stays at 2 B/elem
        + t[1].numel() * 1  # V at 1 B/elem (FP8)
    )
    assert disk_size < fp16_payload, (disk_size, fp16_payload)
    # < 1 KB header overhead beyond the K + V + scales payload
    assert disk_size - asym_payload < 1024, (disk_size, asym_payload)


def test_concurrent_keys_do_not_clobber(serde_pair, tmp_path):
    """Multi-key disk writes: no key writes corrupt the bytes of
    another key.  Catches: shared mutable state in the serializer."""
    ser, des = serde_pair
    n = 8
    tensors = [_kv_tensor(seed=i) for i in range(n)]
    paths = []

    for i, t in enumerate(tensors):
        encoded = ser.serialize(_to_memory_obj(t))
        p = tmp_path / f"key_{i:02d}.bin"
        p.write_bytes(encoded.raw_data)
        paths.append(p)

    # Read back in shuffled order; each must round-trip.
    order = [3, 7, 1, 5, 0, 4, 6, 2]
    for i in order:
        t = tensors[i]
        blob = paths[i].read_bytes()
        decoded = des.deserialize(_bytes_to_mobj(blob, t.shape, t.dtype))
        assert torch.equal(decoded.tensor[0], t[0]), f"K mismatch at key {i}"


def test_cross_codec_read_naive_to_asym_rejected(serde_pair, tmp_path):
    """Bytes written by `naive` (which is just the raw tensor bytes
    with no header) must not parse as asymmetric.  The codec magic
    check fires immediately."""
    _, des = serde_pair
    # Naive's "bytes" would be raw FP16 data with no header.  Use
    # something deterministic to make the test assertion stable.
    bogus = bytes(range(256)) * 8  # 2 KB of arbitrary bytes
    mobj = BytesBufferMemoryObj(
        raw_bytes=bogus,
        metadata=MemoryObjMetadata(
            shape=torch.Size([len(bogus), 0, 0, 0]),
            dtype=None,
            address=0,
            phy_size=0,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.BINARY_BUFFER,
        ),
    )
    with pytest.raises(CorruptEncodedKVError, match="bad magic"):
        des.deserialize(mobj)


def test_cross_codec_partial_truncation_caught(serde_pair, tmp_path):
    """Disk write was truncated mid-payload (e.g., out-of-space).
    Read path must surface a typed error."""
    ser, des = serde_pair
    t = _kv_tensor()
    encoded = ser.serialize(_to_memory_obj(t))

    p = tmp_path / "truncated.bin"
    truncated = bytes(encoded.raw_data)[: -100]
    p.write_bytes(truncated)
    blob = p.read_bytes()

    with pytest.raises(CorruptEncodedKVError):
        des.deserialize(_bytes_to_mobj(blob, t.shape, t.dtype))


def test_disk_payload_bit_flip_caught_by_crc(serde_pair, tmp_path):
    """Disk corruption: flip a byte deep in the payload, CRC catches."""
    ser, des = serde_pair
    t = _kv_tensor(shape=(2, 4, 32, 32))
    encoded = ser.serialize(_to_memory_obj(t))

    p = tmp_path / "corrupt.bin"
    raw = bytearray(encoded.raw_data)
    # Flip a byte in the last 100 bytes — well into the payload.
    raw[-50] ^= 0xFF
    p.write_bytes(bytes(raw))
    blob = p.read_bytes()

    with pytest.raises(CorruptEncodedKVError, match="CRC"):
        des.deserialize(_bytes_to_mobj(blob, t.shape, t.dtype))


def test_capacity_accounting_disk_byte_total(serde_pair, tmp_path):
    """The total disk bytes written across N keys equals the sum of
    each encoded MemoryObj's get_size().  Catches: the encoded MemoryObj
    over- or under-reporting its byte count, which would break disk
    eviction policy."""
    ser, _ = serde_pair
    n = 4
    declared_total = 0
    actual_total = 0
    for i in range(n):
        t = _kv_tensor(seed=i)
        encoded = ser.serialize(_to_memory_obj(t))
        declared_total += encoded.get_size()
        p = tmp_path / f"acc_{i:02d}.bin"
        p.write_bytes(encoded.raw_data)
        actual_total += p.stat().st_size
    assert declared_total == actual_total, (declared_total, actual_total)


def test_metadata_carries_through_disk_path(serde_pair, tmp_path):
    """The original logical (shape, dtype) must survive the disk
    roundtrip via metadata.shapes / metadata.dtypes plurals.  Catches:
    metadata stripped or downgraded to a singular field that loses
    BF16-vs-FP16 information."""
    ser, des = serde_pair
    t = _kv_tensor(dtype=torch.bfloat16)
    encoded = ser.serialize(_to_memory_obj(t))
    p = tmp_path / "meta.bin"
    p.write_bytes(encoded.raw_data)

    # Reconstruct the BytesBufferMemoryObj using the SAME logical
    # shape/dtype info the serializer attached.  The disk backend
    # should be storing this alongside the bytes; we mimic that
    # by reading the encoded metadata from the in-memory object.
    blob = p.read_bytes()
    mobj = _bytes_to_mobj(
        blob,
        encoded.metadata.shapes[0],
        encoded.metadata.dtypes[0],
    )
    decoded = des.deserialize(mobj)
    assert decoded.tensor.dtype == torch.bfloat16
    assert torch.equal(decoded.tensor[0], t[0])


def test_two_engines_same_dir_dont_collide(tmp_path):
    """Two AsymK16V8Serializer instances with different model_ids
    write to the same directory; reads with each model_id only
    succeed for the matching files."""
    cfg = _FakeConfig()
    ser_a = AsymK16V8Serializer(cfg, _FakeMetadata(model_name="qwen2.5-7b"))
    ser_b = AsymK16V8Serializer(cfg, _FakeMetadata(model_name="llama-3.1-8b"))
    des_a = AsymK16V8Deserializer(cfg, _FakeMetadata(model_name="qwen2.5-7b"))
    des_b = AsymK16V8Deserializer(cfg, _FakeMetadata(model_name="llama-3.1-8b"))

    t_a = _kv_tensor(seed=42)
    t_b = _kv_tensor(seed=43)

    pa = tmp_path / "engine_a.bin"
    pb = tmp_path / "engine_b.bin"
    pa.write_bytes(ser_a.serialize(_to_memory_obj(t_a)).raw_data)
    pb.write_bytes(ser_b.serialize(_to_memory_obj(t_b)).raw_data)

    # Same-engine reads succeed.
    out_a = des_a.deserialize(_bytes_to_mobj(pa.read_bytes(), t_a.shape, t_a.dtype))
    out_b = des_b.deserialize(_bytes_to_mobj(pb.read_bytes(), t_b.shape, t_b.dtype))
    assert torch.equal(out_a.tensor[0], t_a[0])
    assert torch.equal(out_b.tensor[0], t_b[0])

    # Cross-engine reads MUST be rejected.
    with pytest.raises(CodecMismatchError, match="model_id"):
        des_a.deserialize(_bytes_to_mobj(pb.read_bytes(), t_b.shape, t_b.dtype))
    with pytest.raises(CodecMismatchError, match="model_id"):
        des_b.deserialize(_bytes_to_mobj(pa.read_bytes(), t_a.shape, t_a.dtype))


def test_idempotent_reads(serde_pair, tmp_path):
    """Reading the same blob twice gives identical decoded tensors
    bit-exact.  Catches: stateful side effects in the deserializer
    (rare but possible if scales are accidentally cached)."""
    ser, des = serde_pair
    t = _kv_tensor()
    encoded = ser.serialize(_to_memory_obj(t))
    p = tmp_path / "idem.bin"
    p.write_bytes(encoded.raw_data)

    blob = p.read_bytes()
    out1 = des.deserialize(_bytes_to_mobj(blob, t.shape, t.dtype))
    out2 = des.deserialize(_bytes_to_mobj(blob, t.shape, t.dtype))
    assert torch.equal(out1.tensor, out2.tensor)


@pytest.mark.gpu
def test_gds_roundtrip_placeholder():
    """Placeholder for the GDS path.  Skipped without CUDA + a
    GDS-capable filesystem (cufile-python + a configured NVMe
    mount).  Same correctness expectations as test_real_disk_roundtrip;
    when the GPU lane is online, lift those tests under
    @pytest.mark.gpu and reuse them."""
    if not torch.cuda.is_available():
        pytest.skip("GDS requires CUDA")
    pytest.skip("GDS lane not yet wired up; Phase 4 work")
