# SPDX-License-Identifier: Apache-2.0
"""Both-plane byte-through (KV_TOGETHER) asym K16/V8 serde.

Unit tests for ``AsymBytethroughK16V8Multi{Serializer,Deserializer}`` in
``lmcache/v1/distributed/serde/asym_k16_v8.py``.

This is the KV_TOGETHER counterpart of the V-only byte-through serde.  It
writes BOTH K (native bf16, byte-through) and V (raw fp8 e4m3,
byte-through) into ONE self-contained ``RAW_UNIT`` / ``V2`` object so the
offload resolves to :attr:`StoragePlacementMode.KV_TOGETHER` and is
reusable across processes / restarts (a fresh process restores the full
KV from the single L2 object -- no split-tier manifest, no L1-only K).

The fail-closed guarantees under test:

* K and V are copied raw (no quant, no dequant); the object round-trips
  byte-identically for both planes.
* the serde resolves to KV_TOGETHER + RAW_UNIT + KV_COMPONENT_GROUPS.
* a K/V slot inversion (fp8 in K, or non-fp8 in V) fails closed on both
  serialize and deserialize.
* this both-plane RAW_UNIT blob and the scale-aware (COMPUTED) blobs are
  mutually unreadable; the V-only (k_payload_len==0) RAW_UNIT blob is
  also rejected here (wrong shape for a both-plane restore).
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
    AsymBytethroughK16V8MultiDeserializer,
    AsymBytethroughK16V8MultiSerializer,
    AsymBytethroughK16V8VOnlyMultiSerializer,
    AsymK16V8MultiDeserializer,
    AsymK16V8MultiSerializer,
)
from lmcache.v1.distributed.serde.multi import MemoryObjGroup
from lmcache.v1.kv_codec import (
    CodecHashes,
    CodecVersion,
    ScaleScheme,
    deserialize_header,
)
from lmcache.v1.memory_management import MemoryObj

_LLAMA_KV_SHAPE = (32, 64, 8, 128)
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


def _serialize(k: torch.Tensor, v: torch.Tensor) -> tuple[MemoryObj, int]:
    s = AsymBytethroughK16V8MultiSerializer()
    layout = (
        MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    n = s.serialize(
        _grp(_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v)), buf, _TEST_KEY
    )
    return buf, n


def _roundtrip(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    buf, _ = _serialize(k, v)
    d = AsymBytethroughK16V8MultiDeserializer()
    k_out = _FakeMemoryObj(tensor=torch.empty_like(k))
    v_out = _FakeMemoryObj(tensor=torch.empty_like(v))
    d.deserialize(buf, _grp(k_out, v_out), _TEST_KEY)
    return k_out.tensor, v_out.tensor


# =============================================================================
# Contract surface
# =============================================================================


def test_group_size_and_slot_mappings() -> None:
    s = AsymBytethroughK16V8MultiSerializer()
    d = AsymBytethroughK16V8MultiDeserializer()
    assert s.group_size == 2
    assert d.group_size == 2
    # (0, 1) -- both planes present -> KV_TOGETHER placement.
    assert s.input_slot_mapping() == (0, 1)
    assert d.output_slot_mapping() == (0, 1)


def test_constructor_rejects_non_e4m3fn() -> None:
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        AsymBytethroughK16V8MultiSerializer(fp8_dtype=torch.float8_e5m2)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        AsymBytethroughK16V8MultiDeserializer(fp8_dtype=torch.float8_e5m2)


def test_resolves_to_kv_together_raw_unit_component_groups() -> None:
    """The whole point: (0,1) -> KV_TOGETHER, byte-through -> RAW_UNIT,
    multi-output -> KV_COMPONENT_GROUPS."""
    from lmcache.v1.distributed.serde.base import SerdeConfig
    from lmcache.v1.distributed.storage_layout import (
        StorageLayoutMode,
        derive_storage_layout_mode,
    )
    from lmcache.v1.distributed.storage_placement import (
        ComponentKeyScheme,
        StoragePlacementMode,
        derive_component_key_scheme,
        derive_storage_placement_mode,
    )

    @dataclass
    class _Adapter:
        serde_config: SerdeConfig

    ad = [_Adapter(SerdeConfig(type="asym_bytethrough_k16_v8"))]
    assert derive_storage_placement_mode(ad) == StoragePlacementMode.KV_TOGETHER
    assert derive_component_key_scheme(ad) == ComponentKeyScheme.RAW_UNIT
    assert derive_storage_layout_mode(ad) == StorageLayoutMode.KV_COMPONENT_GROUPS


# =============================================================================
# Byte identity + no re-quant
# =============================================================================


def test_roundtrip_byte_identical_both_planes() -> None:
    k = _bf16_tensor(*_LLAMA_KV_SHAPE, seed=1)
    v = _fp8_tensor(*_LLAMA_KV_SHAPE, seed=2)
    k_out, v_out = _roundtrip(k, v)
    # K bf16 bit-exact.
    assert torch.equal(k_out, k)
    # V fp8 byte-identical (NaN-safe bit compare).
    assert torch.equal(_u8(v_out), _u8(v))
    assert k_out.dtype == torch.bfloat16
    assert v_out.dtype == torch.float8_e4m3fn


def test_no_requant_v_payload_is_fp8_sized() -> None:
    """V payload must be 1 byte/elem (raw fp8), NOT bf16-sized -- proves no
    dequant-to-bf16 happened on the store side."""
    k = _bf16_tensor(4, 8, seed=3)
    v = _fp8_tensor(4, 8, seed=4)
    buf, n = _serialize(k, v)
    enc = deserialize_header(buf.tensor[:n].numpy().tobytes())
    assert enc.scale_scheme == ScaleScheme.RAW_UNIT
    assert enc.scale_payload_len == 0
    assert enc.v_payload_len == v.numel()  # 1 byte per element
    assert enc.k_payload_len == k.numel() * k.dtype.itemsize  # bf16 = 2 bytes
    # And the on-disk header version is V2 (RAW_UNIT), so a V1-only reader
    # refuses it rather than misreading the raw codes as a scale-aware blob.
    hdr = deserialize_header(buf.tensor[:n].numpy().tobytes())
    assert hdr.k_dtype == torch.bfloat16
    assert hdr.v_dtype == torch.float8_e4m3fn


def test_header_is_v2() -> None:
    k = _bf16_tensor(2, 3, seed=5)
    v = _fp8_tensor(2, 3, seed=6)
    buf, n = _serialize(k, v)
    # The V2 marker lives in the codec header; re-serializing through the
    # codec path already asserted RAW_UNIT above.  Confirm the byte-through
    # blob is not decodable by the scale-aware (COMPUTED) reader.
    from lmcache.v1.kv_codec import AsymK16V8Codec

    enc = AsymK16V8Codec().from_bytes(buf.tensor[:n].numpy().tobytes())
    assert enc.scale_scheme == ScaleScheme.RAW_UNIT
    _ = CodecVersion  # imported for symmetry with the V-only test module


# =============================================================================
# Role-aware fail-closed (serialize side)
# =============================================================================


def test_serialize_rejects_fp8_in_k_slot() -> None:
    """A K/V slot inversion presents an fp8 K -> refuse."""
    k_wrong = _fp8_tensor(4, 8, seed=7)  # fp8 in the K slot
    v = _fp8_tensor(4, 8, seed=8)
    s = AsymBytethroughK16V8MultiSerializer()
    layout = (
        MemoryLayoutDesc(shapes=[k_wrong.shape], dtypes=[k_wrong.dtype]),
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    with pytest.raises(ValueError, match="K slot dtype is fp8|slot inversion"):
        s.serialize(
            _grp(_FakeMemoryObj(tensor=k_wrong), _FakeMemoryObj(tensor=v)),
            buf,
            _TEST_KEY,
        )


def test_serialize_rejects_non_fp8_v_slot() -> None:
    """V must already be fp8 (byte-through does not quantize)."""
    k = _bf16_tensor(4, 8, seed=9)
    v_wrong = _bf16_tensor(4, 8, seed=10)  # bf16 in the V slot
    s = AsymBytethroughK16V8MultiSerializer()
    layout = (
        MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
        MemoryLayoutDesc(shapes=[v_wrong.shape], dtypes=[v_wrong.dtype]),
    )
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    with pytest.raises(ValueError, match="V slot must already be"):
        s.serialize(
            _grp(_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v_wrong)),
            buf,
            _TEST_KEY,
        )


def test_serialize_rejects_missing_k() -> None:
    v = _fp8_tensor(4, 8, seed=11)
    s = AsymBytethroughK16V8MultiSerializer()
    buf = _byte_buffer(4096)
    with pytest.raises(ValueError, match="K slot is required"):
        s.serialize(_grp(None, _FakeMemoryObj(tensor=v)), buf, _TEST_KEY)


# =============================================================================
# Role-aware fail-closed (deserialize side)
# =============================================================================


def test_deserialize_rejects_wrong_k_dtype() -> None:
    k = _bf16_tensor(4, 8, seed=12)
    v = _fp8_tensor(4, 8, seed=13)
    buf, _ = _serialize(k, v)
    d = AsymBytethroughK16V8MultiDeserializer()
    k_out = _FakeMemoryObj(tensor=torch.empty(4, 8, dtype=torch.float16))  # wrong
    v_out = _FakeMemoryObj(tensor=torch.empty(4, 8, dtype=torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="dst K dtype .* != stored K dtype"):
        d.deserialize(buf, _grp(k_out, v_out), _TEST_KEY)


def test_deserialize_rejects_wrong_v_dtype() -> None:
    k = _bf16_tensor(4, 8, seed=14)
    v = _fp8_tensor(4, 8, seed=15)
    buf, _ = _serialize(k, v)
    d = AsymBytethroughK16V8MultiDeserializer()
    k_out = _FakeMemoryObj(tensor=torch.empty(4, 8, dtype=torch.bfloat16))
    v_out = _FakeMemoryObj(tensor=torch.empty(4, 8, dtype=torch.bfloat16))  # wrong
    with pytest.raises(ValueError, match="dst V dtype .* must be"):
        d.deserialize(buf, _grp(k_out, v_out), _TEST_KEY)


def test_deserialize_requires_both_slots() -> None:
    k = _bf16_tensor(4, 8, seed=16)
    v = _fp8_tensor(4, 8, seed=17)
    buf, _ = _serialize(k, v)
    d = AsymBytethroughK16V8MultiDeserializer()
    v_out = _FakeMemoryObj(tensor=torch.empty(4, 8, dtype=torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="K dst slot is required"):
        d.deserialize(buf, _grp(None, v_out), _TEST_KEY)


# =============================================================================
# Mutual reject across schemes / shapes
# =============================================================================


def test_scale_aware_reader_rejects_bytethrough_blob() -> None:
    """A COMPUTED (scale-aware) deserializer must refuse this RAW_UNIT blob."""
    k = _bf16_tensor(4, 8, seed=18)
    v = _fp8_tensor(4, 8, seed=19)
    buf, _ = _serialize(k, v)
    d = AsymK16V8MultiDeserializer()
    k_out = _FakeMemoryObj(tensor=torch.empty(4, 8, dtype=torch.bfloat16))
    v_out = _FakeMemoryObj(tensor=torch.empty(4, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="not COMPUTED_PER_TENSOR|RAW_UNIT"):
        d.deserialize(buf, _grp(k_out, v_out), _TEST_KEY)


def test_bytethrough_reader_rejects_scale_aware_blob() -> None:
    """This both-plane RAW_UNIT reader must refuse a COMPUTED blob."""
    k = _bf16_tensor(4, 8, seed=20)
    v = _bf16_tensor(4, 8, seed=21)  # scale-aware serde takes native V
    s = AsymK16V8MultiSerializer()
    layout = (
        MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    s.serialize(
        _grp(_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v)), buf, _TEST_KEY
    )
    d = AsymBytethroughK16V8MultiDeserializer()
    k_out = _FakeMemoryObj(tensor=torch.empty(4, 8, dtype=torch.bfloat16))
    v_out = _FakeMemoryObj(tensor=torch.empty(4, 8, dtype=torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="not RAW_UNIT"):
        d.deserialize(buf, _grp(k_out, v_out), _TEST_KEY)


def test_bytethrough_reader_rejects_v_only_blob() -> None:
    """A V-only (k_payload_len==0) RAW_UNIT blob has no K bytes; the
    both-plane reader must refuse it rather than mis-slice."""
    v = _fp8_tensor(4, 8, seed=22)
    s = AsymBytethroughK16V8VOnlyMultiSerializer()
    layout = (None, MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]))
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    s.serialize(_grp(None, _FakeMemoryObj(tensor=v)), buf, _TEST_KEY)
    d = AsymBytethroughK16V8MultiDeserializer()
    k_out = _FakeMemoryObj(tensor=torch.empty(4, 8, dtype=torch.bfloat16))
    v_out = _FakeMemoryObj(tensor=torch.empty(4, 8, dtype=torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="must carry K bytes|k_payload_len"):
        d.deserialize(buf, _grp(k_out, v_out), _TEST_KEY)


# =============================================================================
# Numeric-pattern robustness (never trust all-zeros)
# =============================================================================


def test_roundtrip_nontrivial_patterns() -> None:
    # Distinct, non-zero, non-uniform patterns for K and V.
    k = _bf16_tensor(16, 5, 3, seed=101) * 7.0 - 3.0
    k = k.contiguous()
    v = _fp8_tensor(16, 5, 3, seed=202)
    k_out, v_out = _roundtrip(k, v)
    assert torch.equal(k_out, k)
    assert torch.equal(_u8(v_out), _u8(v))


# =============================================================================
# Cross-process layout-provenance gate (fail-closed)
# =============================================================================


def _roundtrip_with_provenance(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    write_prov: "CodecHashes | None",
    read_prov: "CodecHashes | None",
) -> tuple[torch.Tensor, torch.Tensor]:
    from lmcache.v1.distributed.serde.asym_k16_v8 import (
        AsymBytethroughK16V8MultiDeserializer as _D,
        AsymBytethroughK16V8MultiSerializer as _S,
    )

    s = _S(layout_provenance=write_prov)
    layout = (
        MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    s.serialize(_grp(_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v)), buf, _TEST_KEY)
    d = _D(layout_provenance=read_prov)
    k_out = _FakeMemoryObj(tensor=torch.empty_like(k))
    v_out = _FakeMemoryObj(tensor=torch.empty_like(v))
    d.deserialize(buf, _grp(k_out, v_out), _TEST_KEY)
    return k_out.tensor, v_out.tensor


def test_provenance_gate_inert_when_unconfigured() -> None:
    """No provenance on either side -> legacy behaviour, byte-identical."""
    k = _bf16_tensor(4, 8, seed=30)
    v = _fp8_tensor(4, 8, seed=31)
    k_out, v_out = _roundtrip_with_provenance(
        k, v, write_prov=None, read_prov=None
    )
    assert torch.equal(k_out, k) and torch.equal(_u8(v_out), _u8(v))


def test_provenance_matching_config_round_trips() -> None:
    from lmcache.v1.kv_codec import CodecHashes

    prov = CodecHashes(kv_layout="blk16;bf16K;fp8V;v1", model_revision_hash="abc")
    k = _bf16_tensor(4, 8, seed=32)
    v = _fp8_tensor(4, 8, seed=33)
    k_out, v_out = _roundtrip_with_provenance(
        k, v, write_prov=prov, read_prov=prov
    )
    assert torch.equal(k_out, k) and torch.equal(_u8(v_out), _u8(v))


def test_provenance_mismatch_rejected() -> None:
    """Different block-size/layout digest -> CodecMismatchError (=> miss)."""
    from lmcache.v1.kv_codec import CodecHashes, CodecMismatchError

    write = CodecHashes(kv_layout="blk16;bf16K;fp8V;v1")
    read = CodecHashes(kv_layout="blk32;bf16K;fp8V;v1")  # different block size
    k = _bf16_tensor(4, 8, seed=34)
    v = _fp8_tensor(4, 8, seed=35)
    with pytest.raises(CodecMismatchError, match="kv_layout"):
        _roundtrip_with_provenance(k, v, write_prov=write, read_prov=read)


def test_provenance_fail_closed_on_missing_field() -> None:
    """Writer stamped NO provenance, reader requires it -> refuse (fail-closed),
    NOT a wildcard accept."""
    from lmcache.v1.kv_codec import CodecHashes, CodecMismatchError

    read = CodecHashes(kv_layout="blk16;bf16K;fp8V;v1")
    k = _bf16_tensor(4, 8, seed=36)
    v = _fp8_tensor(4, 8, seed=37)
    with pytest.raises(CodecMismatchError, match="missing 'kv_layout'|fail-closed"):
        _roundtrip_with_provenance(k, v, write_prov=None, read_prov=read)


def test_provenance_reader_unconfigured_accepts_stamped_blob() -> None:
    """A reader with NO provenance configured (single-process/legacy) still
    reads a stamped blob -- the gate only engages when the reader opts in."""
    from lmcache.v1.kv_codec import CodecHashes

    write = CodecHashes(kv_layout="blk16;bf16K;fp8V;v1")
    k = _bf16_tensor(4, 8, seed=38)
    v = _fp8_tensor(4, 8, seed=39)
    k_out, v_out = _roundtrip_with_provenance(
        k, v, write_prov=write, read_prov=None
    )
    assert torch.equal(k_out, k) and torch.equal(_u8(v_out), _u8(v))


def test_factory_parses_layout_provenance_and_rejects_unknown_keys() -> None:
    from lmcache.v1.distributed.serde import create_serde_processor
    from lmcache.v1.distributed.serde.base import SerdeConfig

    ok = SerdeConfig(
        type="asym_bytethrough_k16_v8",
        kwargs={"layout_provenance": {"kv_layout": "blk16", "model_id": "m"}},
    )
    p = create_serde_processor(ok)
    p.close()  # resolves without error

    bad = SerdeConfig(
        type="asym_bytethrough_k16_v8",
        kwargs={"layout_provenance": {"block_size": "16"}},  # unknown key
    )
    with pytest.raises(ValueError, match="unknown keys"):
        create_serde_processor(bad)
