# SPDX-License-Identifier: Apache-2.0
"""KMesh family serde: bitwise round-trips through the multi contract."""

# Standard
from dataclasses import dataclass
from typing import Optional, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde.kmesh_family import (
    KMeshFamilyCodecError,
    KMeshFamilyMasterMirrorDeserializer,
    KMeshFamilyMasterMirrorSerializer,
    decode_family,
    encode_family,
)
from lmcache.v1.distributed.serde.multi import LayoutDescGroup, MemoryObjGroup
from lmcache.v1.memory_management import MemoryObj

HIDDEN = 16
KEY = cast(ObjectKey, "kmesh-family-test-key")


@dataclass
class _FakeMemoryObj:
    tensor: Optional[torch.Tensor]


def _grp(*objs: object) -> MemoryObjGroup:
    return cast(MemoryObjGroup, objs)


def _family(
    n: int = 4,
    tokens: int = 64,
    shared: int = 40,
    identical_shared_v: bool = True,
    v_changed: int = 0,
) -> tuple[list[torch.Tensor], tuple[tuple[int, int], ...]]:
    g = torch.Generator().manual_seed(7)
    shared_k = torch.randn(shared, HIDDEN, generator=g).bfloat16()
    shared_v = torch.randn(shared, HIDDEN, generator=g).bfloat16()
    tensors, spans = [], []
    for i in range(n):
        start = 4 + i
        end = start + shared
        t = torch.randn(2, tokens, HIDDEN, generator=g).bfloat16()
        t[0, start:end, :] = (shared_k.view(torch.int16) ^ (i + 1)).view(torch.bfloat16)
        if identical_shared_v:
            t[1, start:end, :] = shared_v
        if i > 0 and v_changed:
            t[1, start : start + v_changed, :] = torch.randn(
                v_changed, HIDDEN, generator=g
            ).bfloat16()
        tensors.append(t)
        spans.append((start, end))
    return tensors, tuple(spans)


def _roundtrip(tensors, spans):
    serializer = KMeshFamilyMasterMirrorSerializer(spans)
    buffer = _FakeMemoryObj(tensor=torch.zeros(8 * 1024 * 1024, dtype=torch.uint8))
    n = serializer.serialize(
        _grp(*[_FakeMemoryObj(tensor=t) for t in tensors]),
        cast(MemoryObj, buffer),
        KEY,
    )
    deserializer = KMeshFamilyMasterMirrorDeserializer(len(tensors))
    assert serializer.group_size == len(tensors)
    assert deserializer.group_size == len(tensors)
    outs = [_FakeMemoryObj(tensor=torch.zeros_like(t)) for t in tensors]
    src = _FakeMemoryObj(tensor=buffer.tensor[:n].clone())
    deserializer.deserialize(cast(MemoryObj, src), _grp(*outs), KEY)
    return n, outs


def test_roundtrip_bitwise_and_v_sharing_compresses() -> None:
    tensors, spans = _family(identical_shared_v=True)
    n_shared, outs = _roundtrip(tensors, spans)
    for t, out in zip(tensors, outs, strict=True):
        assert bool(out.tensor.view(torch.int16).eq(t.view(torch.int16)).all())
    dissimilar, spans_d = _family(identical_shared_v=False)
    n_dense, outs_d = _roundtrip(dissimilar, spans_d)
    for t, out in zip(dissimilar, outs_d, strict=True):
        assert bool(out.tensor.view(torch.int16).eq(t.view(torch.int16)).all())
    assert n_shared < n_dense


def test_recomputed_v_tokens_roundtrip() -> None:
    tensors, spans = _family(v_changed=5)
    _n, outs = _roundtrip(tensors, spans)
    for t, out in zip(tensors, outs, strict=True):
        assert bool(out.tensor.view(torch.int16).eq(t.view(torch.int16)).all())


def test_payload_corruption_is_rejected() -> None:
    tensors, spans = _family()
    blob = bytearray(encode_family(list(tensors), spans, 0))
    blob[-3] ^= 0xFF
    with pytest.raises(KMeshFamilyCodecError, match="corrupt"):
        decode_family(bytes(blob))


def test_span_and_capacity_failures_close() -> None:
    tensors, spans = _family()
    bad = list(spans)
    bad[1] = (bad[1][0], bad[1][1] - 1)
    with pytest.raises(KMeshFamilyCodecError, match="length differs"):
        encode_family(list(tensors), tuple(bad), 0)

    serializer = KMeshFamilyMasterMirrorSerializer(spans)
    small = _FakeMemoryObj(tensor=torch.zeros(64, dtype=torch.uint8))
    with pytest.raises(KMeshFamilyCodecError, match="capacity"):
        serializer.serialize(
            _grp(*[_FakeMemoryObj(tensor=t) for t in tensors]),
            cast(MemoryObj, small),
            KEY,
        )
    with pytest.raises(KMeshFamilyCodecError, match="supply a tensor"):
        serializer.serialize(
            _grp(None, *[_FakeMemoryObj(tensor=t) for t in tensors[1:]]),
            cast(MemoryObj, small),
            KEY,
        )


def test_group_size_and_layout_mismatches_close() -> None:
    tensors, spans = _family()
    deserializer = KMeshFamilyMasterMirrorDeserializer(3)
    blob = encode_family(list(tensors), spans, 0)
    src = _FakeMemoryObj(tensor=torch.frombuffer(bytearray(blob), dtype=torch.uint8))
    outs = [_FakeMemoryObj(tensor=torch.zeros_like(t)) for t in tensors[:3]]
    with pytest.raises(KMeshFamilyCodecError, match="expected 3"):
        deserializer.deserialize(cast(MemoryObj, src), _grp(*outs), KEY)

    full = KMeshFamilyMasterMirrorDeserializer(4)
    wrong = [_FakeMemoryObj(tensor=torch.zeros_like(t)) for t in tensors]
    wrong[2] = _FakeMemoryObj(tensor=torch.zeros(2, 8, HIDDEN, dtype=torch.bfloat16))
    with pytest.raises(KMeshFamilyCodecError, match="layout mismatch"):
        full.deserialize(cast(MemoryObj, src), _grp(*wrong), KEY)


def test_estimate_bounds_actual_size() -> None:
    tensors, spans = _family(identical_shared_v=False)
    serializer = KMeshFamilyMasterMirrorSerializer(spans)

    @dataclass
    class _Layout:
        shapes: tuple
        dtypes: tuple

    layouts = tuple(
        _Layout(shapes=(tuple(t.shape),), dtypes=(t.dtype,)) for t in tensors
    )
    estimate = serializer.estimate_serialized_size(cast(LayoutDescGroup, layouts))
    actual = len(encode_family(list(tensors), spans, 0))
    assert estimate >= actual
