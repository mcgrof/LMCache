# SPDX-License-Identifier: Apache-2.0
"""
CO2: byte-through (Mode C) V-only asym K16/V8 serde.

Unit tests for ``AsymBytethroughK16V8VOnly{Serializer,Deserializer}``
in ``lmcache/v1/distributed/serde/asym_k16_v8.py``.

The byte-through path is for the vLLM live-asymmetric K16/V8 layout,
where V is *already* FP8 e4m3 in HBM.  There is no full-precision V to
compute scales from, so:

* ``serialize`` copies the raw e4m3 code bytes straight through, stores
  no scales, and emits a :class:`ScaleScheme.RAW_UNIT` /
  :class:`CodecVersion.V2` blob.
* ``deserialize`` restores the raw codes bit-exact into an
  ``float8_e4m3fn`` destination -- no dequantization.

The fail-closed guarantees under test:

* byte-through and scale-aware blobs are mutually unreadable (each
  deserializer rejects the other's scheme),
* the store side refuses an fp8 V into the *scale-aware* serdes (which
  would double-quantize) and refuses a non-e4m3 V into the byte-through
  serde,
* the restore side refuses widening a RAW_UNIT blob into fp16/bf16
  (there is no scale to widen with).
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.asym_k16_v8 import (
    AsymBytethroughK16V8VOnlyMultiDeserializer,
    AsymBytethroughK16V8VOnlyMultiSerializer,
    AsymK16V8MultiDeserializer,
    AsymK16V8MultiSerializer,
    AsymK16V8VOnlyMultiDeserializer,
    AsymK16V8VOnlyMultiSerializer,
)
from lmcache.v1.distributed.serde.multi import MemoryObjGroup
from lmcache.v1.kv_codec import (
    CodecVersion,
    ScaleScheme,
    ScaleScope,
    deserialize_header,
)
from lmcache.v1.memory_management import MemoryObj

_TEST_KEY = ObjectKey(chunk_hash=b"\x00" * 32, model_name="test", kv_rank=0)


@dataclass
class _FakeMemoryObj:
    tensor: torch.Tensor


def _bf16_tensor(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=torch.bfloat16, generator=g).contiguous()


def _fp8_tensor(*shape: int, seed: int) -> torch.Tensor:
    """A real e4m3 tensor: quantize a bf16 draw into the fp8 grid."""
    return _bf16_tensor(*shape, seed=seed).to(torch.float8_e4m3fn).contiguous()


def _byte_buffer(num_bytes: int) -> MemoryObj:
    return cast(
        MemoryObj, _FakeMemoryObj(tensor=torch.zeros(num_bytes, dtype=torch.uint8))
    )


def _grp(*objs: object) -> MemoryObjGroup:
    return cast(MemoryObjGroup, objs)


def _u8(t: torch.Tensor) -> torch.Tensor:
    """NaN-safe bit view for exact comparison of fp8 tensors."""
    return t.contiguous().view(-1).view(torch.uint8)


_LLAMA_KV_SHAPE = (32, 64, 8, 128)


def _serialize_bytethrough(v: torch.Tensor) -> tuple[MemoryObj, int]:
    s = AsymBytethroughK16V8VOnlyMultiSerializer()
    layout = (None, MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]))
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    n = s.serialize(_grp(None, _FakeMemoryObj(tensor=v)), buf, _TEST_KEY)
    return buf, n


def _blob_bytes(buf: MemoryObj, n: int) -> bytes:
    return buf.tensor[:n].numpy().tobytes()


# =============================================================================
# Contract surface
# =============================================================================


def test_group_size_and_slot_mappings() -> None:
    s = AsymBytethroughK16V8VOnlyMultiSerializer()
    d = AsymBytethroughK16V8VOnlyMultiDeserializer()
    assert s.group_size == 2
    assert d.group_size == 2
    assert s.input_slot_mapping() == (None, 1)
    assert d.output_slot_mapping() == (None, 1)


def test_constructor_rejects_non_e4m3fn() -> None:
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        AsymBytethroughK16V8VOnlyMultiSerializer(fp8_dtype=torch.float8_e5m2)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        AsymBytethroughK16V8VOnlyMultiDeserializer(fp8_dtype=torch.float8_e5m2)


def test_serialize_rejects_k_present() -> None:
    s = AsymBytethroughK16V8VOnlyMultiSerializer()
    k = _FakeMemoryObj(tensor=_fp8_tensor(2, 4, seed=0))
    v = _FakeMemoryObj(tensor=_fp8_tensor(2, 4, seed=1))
    with pytest.raises(ValueError, match="K slot must be None"):
        s.serialize(_grp(k, v), _byte_buffer(64), _TEST_KEY)


def test_serialize_requires_v() -> None:
    s = AsymBytethroughK16V8VOnlyMultiSerializer()
    buf = _byte_buffer(64)
    with pytest.raises(ValueError, match="V slot is required"):
        s.serialize(_grp(None, None), buf, _TEST_KEY)
    v_no_tensor = _FakeMemoryObj(tensor=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="V slot is required"):
        s.serialize(_grp(None, v_no_tensor), buf, _TEST_KEY)


def test_serialize_rejects_native_dtype_v() -> None:
    """A bf16 V is a mistake here -- it must go through the scale-aware
    serde that quantizes.  Byte-through only accepts already-fp8 V."""
    s = AsymBytethroughK16V8VOnlyMultiSerializer()
    v = _FakeMemoryObj(tensor=_bf16_tensor(2, 4, seed=2))
    with pytest.raises(ValueError, match="must already be float8_e4m3fn"):
        s.serialize(_grp(None, v), _byte_buffer(256), _TEST_KEY)


def test_serialize_rejects_wrong_fp8_variant() -> None:
    """e5m2 V is fp8 but not the e4m3 the live-asym layout stores."""
    s = AsymBytethroughK16V8VOnlyMultiSerializer()
    v = _FakeMemoryObj(
        tensor=_bf16_tensor(2, 4, seed=3).to(torch.float8_e5m2).contiguous()
    )
    with pytest.raises(ValueError, match="must already be float8_e4m3fn"):
        s.serialize(_grp(None, v), _byte_buffer(256), _TEST_KEY)


def test_estimate_size_rejects_k_layout_and_requires_v() -> None:
    s = AsymBytethroughK16V8VOnlyMultiSerializer()
    k_layout = MemoryLayoutDesc(shapes=[torch.Size([2, 4])], dtypes=[torch.bfloat16])
    v_layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4])], dtypes=[torch.float8_e4m3fn]
    )
    with pytest.raises(ValueError, match="K layout must be None"):
        s.estimate_serialized_size((k_layout, v_layout))
    with pytest.raises(ValueError, match="V layout is required"):
        s.estimate_serialized_size((None, None))


# =============================================================================
# Round-trip: bit-exact (no quantization anywhere on the path)
# =============================================================================


def test_round_trip_is_bit_exact() -> None:
    v = _fp8_tensor(*_LLAMA_KV_SHAPE, seed=10)
    buf, n = _serialize_bytethrough(v)
    assert 0 < n

    d = AsymBytethroughK16V8VOnlyMultiDeserializer()
    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v))
    d.deserialize(buf, _grp(None, v_out), _TEST_KEY)

    # Exact byte equality: the stored codes are the restored codes.
    assert torch.equal(_u8(v_out.tensor), _u8(v))


def test_blob_header_is_raw_unit_v2_no_scales() -> None:
    v = _fp8_tensor(4, 8, seed=11)
    buf, n = _serialize_bytethrough(v)
    enc = deserialize_header(_blob_bytes(buf, n))

    assert enc.scale_scheme is ScaleScheme.RAW_UNIT
    # RAW_UNIT must advertise NONE scope, not PER_TENSOR: there is no
    # scalar to key off, and PER_TENSOR would invite one.
    assert enc.scale_scope is ScaleScope.NONE
    assert enc.header_bytes is not None
    # V2 is the on-disk version for RAW_UNIT (V1-only readers reject it).
    version = enc.header_bytes[8] | (enc.header_bytes[9] << 8)
    assert version == int(CodecVersion.V2)
    assert enc.k_payload_len == 0
    assert enc.scale_payload_len == 0
    assert enc.v_payload_len == v.numel()  # 1 byte per fp8 element
    assert enc.v_dtype == torch.float8_e4m3fn


def test_deserialize_ignores_k_slot() -> None:
    v = _fp8_tensor(2, 4, 8, 64, seed=12)
    buf, _ = _serialize_bytethrough(v)

    d = AsymBytethroughK16V8VOnlyMultiDeserializer()
    sentinel = torch.full((2, 4), fill_value=7, dtype=torch.uint8)
    k_unused = _FakeMemoryObj(tensor=sentinel.clone())
    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v))
    d.deserialize(buf, _grp(k_unused, v_out), _TEST_KEY)

    assert torch.equal(k_unused.tensor, sentinel)
    assert torch.equal(_u8(v_out.tensor), _u8(v))


def test_deserialize_rejects_none_v_target() -> None:
    v = _fp8_tensor(2, 4, seed=13)
    buf, _ = _serialize_bytethrough(v)
    d = AsymBytethroughK16V8VOnlyMultiDeserializer()
    # V is the ONLY payload; a None target must fail closed, not no-op --
    # otherwise a broken load path drops V and still reports success.
    with pytest.raises(ValueError, match="V dst slot is required"):
        d.deserialize(buf, _grp(None, None), _TEST_KEY)


def test_round_trip_strided_source_and_dest_is_logical_order() -> None:
    """The stored invariant is logical element order, not physical HBM
    layout.  A non-contiguous source view must round-trip byte-for-byte
    over its LOGICAL elements into a non-contiguous destination view.

    This mirrors the vLLM region-split KV cache, where the live-asym V
    half is a strided zero-copy view."""
    # Build a contiguous fp8 block, then take a strided view (drop every
    # other row) so the logical tensor is non-contiguous.
    base = _fp8_tensor(8, 16, seed=30)
    v_view = base[::2]  # shape (4, 16), non-contiguous
    assert not v_view.is_contiguous()

    buf, _ = _serialize_bytethrough(v_view)

    # Restore into a non-contiguous destination view with a different
    # underlying stride than the source.
    dst_base = torch.zeros(4, 32, dtype=torch.float8_e4m3fn)
    dst_view = dst_base[:, ::2]  # shape (4, 16), non-contiguous
    assert not dst_view.is_contiguous()
    v_out = _FakeMemoryObj(tensor=dst_view)

    d = AsymBytethroughK16V8VOnlyMultiDeserializer()
    d.deserialize(buf, _grp(None, v_out), _TEST_KEY)

    # Compare logical element order (contiguous copies), NaN-safe.
    assert torch.equal(_u8(v_out.tensor.contiguous()), _u8(v_view.contiguous()))


# =============================================================================
# Restore-target strictness
# =============================================================================


def test_deserialize_rejects_non_fp8_target() -> None:
    v = _fp8_tensor(4, 8, seed=14)
    buf, _ = _serialize_bytethrough(v)
    d = AsymBytethroughK16V8VOnlyMultiDeserializer()
    v_out = _FakeMemoryObj(tensor=torch.zeros(4, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="must be float8_e4m3fn"):
        d.deserialize(buf, _grp(None, v_out), _TEST_KEY)


def test_deserialize_rejects_shape_mismatch() -> None:
    v = _fp8_tensor(4, 8, seed=15)
    buf, _ = _serialize_bytethrough(v)
    d = AsymBytethroughK16V8VOnlyMultiDeserializer()
    v_out = _FakeMemoryObj(tensor=torch.zeros(4, 9, dtype=torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="elements"):
        d.deserialize(buf, _grp(None, v_out), _TEST_KEY)


# =============================================================================
# Cross-mode mutual rejection (RAW_UNIT vs COMPUTED_PER_TENSOR)
# =============================================================================


def test_bytethrough_deserializer_rejects_scale_aware_blob() -> None:
    """A scale-aware (COMPUTED_PER_TENSOR) blob must not be read as
    raw codes -- the byte-through deserializer rejects it."""
    scale_aware_s = AsymK16V8VOnlyMultiSerializer()
    v = _bf16_tensor(4, 8, 8, 64, seed=16)
    layout = (None, MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]))
    buf = _byte_buffer(scale_aware_s.estimate_serialized_size(layout))
    scale_aware_s.serialize(_grp(None, _FakeMemoryObj(tensor=v)), buf, _TEST_KEY)

    d = AsymBytethroughK16V8VOnlyMultiDeserializer()
    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v).to(torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="not RAW_UNIT"):
        d.deserialize(buf, _grp(None, v_out), _TEST_KEY)


def test_scale_aware_v_only_deserializer_rejects_bytethrough_blob() -> None:
    """A RAW_UNIT blob has no scales; the scale-aware V-only
    deserializer must reject it rather than try to dequantize with
    absent scales."""
    v = _fp8_tensor(4, 8, 8, 64, seed=17)
    buf, _ = _serialize_bytethrough(v)

    d = AsymK16V8VOnlyMultiDeserializer()
    v_out = _FakeMemoryObj(tensor=torch.zeros(*v.shape, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="not COMPUTED_PER_TENSOR"):
        d.deserialize(buf, _grp(None, v_out), _TEST_KEY)


def test_storage_only_deserializer_rejects_bytethrough_blob() -> None:
    """Every scale-aware reader must reject RAW_UNIT as a first-class
    mismatch -- including the full (K, V) storage-only deserializer."""
    v = _fp8_tensor(4, 8, 8, 64, seed=21)
    buf, _ = _serialize_bytethrough(v)

    d = AsymK16V8MultiDeserializer()
    k_out = _FakeMemoryObj(tensor=torch.zeros(*v.shape, dtype=torch.bfloat16))
    v_out = _FakeMemoryObj(tensor=torch.zeros(*v.shape, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="not COMPUTED_PER_TENSOR"):
        d.deserialize(buf, _grp(k_out, v_out), _TEST_KEY)


# =============================================================================
# Double-quantize guards on the scale-aware store paths
# =============================================================================


def test_scale_aware_v_only_serializer_rejects_fp8_v() -> None:
    """Feeding an already-fp8 V into the quantizing serde would
    double-quantize; it must be refused."""
    s = AsymK16V8VOnlyMultiSerializer()
    v = _FakeMemoryObj(tensor=_fp8_tensor(2, 4, seed=18))
    with pytest.raises(ValueError, match="already FP8"):
        s.serialize(_grp(None, v), _byte_buffer(256), _TEST_KEY)


def test_storage_only_serializer_rejects_fp8_v() -> None:
    s = AsymK16V8MultiSerializer()
    k = _FakeMemoryObj(tensor=_bf16_tensor(2, 4, seed=19))
    v = _FakeMemoryObj(tensor=_fp8_tensor(2, 4, seed=20))
    with pytest.raises(ValueError, match="already FP8"):
        s.serialize(_grp(k, v), _byte_buffer(256), _TEST_KEY)
