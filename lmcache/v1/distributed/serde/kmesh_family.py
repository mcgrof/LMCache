# SPDX-License-Identifier: Apache-2.0
"""KMesh Master--Mirror family serde on the multi-output contract.

A recovered collective family stores one dense master plus sparse
per-mirror differences.  Position-aligned diffing measures zero sharing
whenever siblings carry the shared span at different offsets, so this
codec keys the diff by segment content: every member declares its shared
span, the master stores both planes dense, and each mirror stores its
private tokens dense plus per-token diffs over the content-aligned
shared span, computed from raw ``int16`` storage views so decoding is
bitwise-exact by construction on any family.  On a family with no
cross-sibling reuse the encoding degrades to dense plus index instead of
silently losing exactness.

Group contract: ``group_size`` members, each a ``[2, tokens, hidden]``
bfloat16 tensor (plane 0 = K, plane 1 = V).  The serializer is
constructed per family with every member's shared span, taken from the
group's reuse-plan metadata; the blob header is self-describing, so the
deserializer needs only the expected group size.  This serde is opt-in
and is not factory-registered.
"""

# Future
from __future__ import annotations

# Standard
import hashlib
import io
import json
import math

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde.multi import (
    LayoutDescGroup,
    MemoryObjGroup,
    MultiDeserializer,
    MultiSerializer,
    validate_group_size,
)
from lmcache.v1.memory_management import MemoryObj

MAGIC = b"KMMF1\n"
_PLANES = 2


class KMeshFamilyCodecError(ValueError):
    """Raised for malformed inputs, spans, indexes, or corrupt payloads."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.int16).numpy().tobytes()


def _validate_member(tensor: torch.Tensor, span: tuple[int, int], index: int) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 3
        or tensor.shape[0] != _PLANES
        or tensor.dtype != torch.bfloat16
    ):
        raise KMeshFamilyCodecError(
            f"family member {index} must be a [2, tokens, hidden] bf16 tensor"
        )
    start, end = span
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= tensor.shape[1]
    ):
        raise KMeshFamilyCodecError(
            f"family member {index} shared span {span} is out of range"
        )


def encode_family(
    tensors: list[torch.Tensor],
    spans: tuple[tuple[int, int], ...],
    master_index: int,
) -> bytes:
    """Encode one family into a self-describing blob."""
    if len(tensors) != len(spans) or len(tensors) < 2:
        raise KMeshFamilyCodecError(
            "a family needs at least two members with one span each"
        )
    if not 0 <= master_index < len(tensors):
        raise KMeshFamilyCodecError("master index outside the family")
    hidden = tensors[0].shape[2]
    for index, (tensor, span) in enumerate(zip(tensors, spans, strict=True)):
        _validate_member(tensor, span, index)
        if tensor.shape[2] != hidden:
            raise KMeshFamilyCodecError("family members disagree on hidden size")
    master = tensors[master_index]
    ms, me = spans[master_index]
    shared_len = me - ms
    for index, (start, end) in enumerate(spans):
        if end - start != shared_len:
            raise KMeshFamilyCodecError(
                f"family member {index} shared span length differs from master"
            )

    sections: list[tuple[str, bytes]] = []
    header: dict = {
        "schema": "kmesh.family_master_mirror_serde.v1",
        "hidden": hidden,
        "master_index": master_index,
        "members": [],
        "sections": [],
    }
    sections.append(("master", _tensor_bytes(master)))
    for index, (tensor, (start, end)) in enumerate(zip(tensors, spans, strict=True)):
        entry: dict = {
            "tokens": tensor.shape[1],
            "shared_span": [start, end],
        }
        if index != master_index:
            private = list(range(0, start)) + list(range(end, tensor.shape[1]))
            entry["private_token_indices"] = private
            if private:
                sections.append(
                    (f"{index}/private", _tensor_bytes(tensor[:, private, :]))
                )
            for plane, name in ((0, "k"), (1, "v")):
                mine = tensor[plane, start:end, :]
                ref = master[plane, ms:me, :]
                changed = [
                    int(i)
                    for i in (
                        mine.view(torch.int16)
                        .ne(ref.view(torch.int16))
                        .any(dim=1)
                        .nonzero()
                        .flatten()
                    )
                ]
                entry[f"{name}_changed_aligned_indices"] = changed
                if changed:
                    sections.append(
                        (
                            f"{index}/{name}/shared_diff",
                            _tensor_bytes(mine[changed, :]),
                        )
                    )
        header["members"].append(entry)

    payload = io.BytesIO()
    offset = 0
    for label, data in sections:
        header["sections"].append(
            {
                "label": label,
                "offset": offset,
                "length": len(data),
                "sha256": _sha256(data),
            }
        )
        payload.write(data)
        offset += len(data)
    header_json = json.dumps(header, sort_keys=True).encode()
    blob = io.BytesIO()
    blob.write(MAGIC)
    blob.write(len(header_json).to_bytes(8, "little"))
    blob.write(header_json)
    blob.write(payload.getvalue())
    return blob.getvalue()


def decode_family(blob: bytes) -> list[torch.Tensor]:
    """Decode a family blob into bitwise-exact member tensors."""
    if blob[: len(MAGIC)] != MAGIC:
        raise KMeshFamilyCodecError("bad family blob magic")
    pos = len(MAGIC)
    header_len = int.from_bytes(blob[pos : pos + 8], "little")
    pos += 8
    if pos + header_len > len(blob):
        raise KMeshFamilyCodecError("truncated family header")
    try:
        header = json.loads(blob[pos : pos + header_len].decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KMeshFamilyCodecError(f"malformed family header: {error}") from error
    payload = blob[pos + header_len :]
    if header.get("schema") != "kmesh.family_master_mirror_serde.v1":
        raise KMeshFamilyCodecError("unknown family schema")
    hidden = header["hidden"]

    sections: dict[str, bytes] = {}
    for section in header["sections"]:
        data = payload[section["offset"] : section["offset"] + section["length"]]
        if len(data) != section["length"]:
            raise KMeshFamilyCodecError(f"truncated family section {section['label']}")
        if _sha256(data) != section["sha256"]:
            raise KMeshFamilyCodecError(f"corrupt family section {section['label']}")
        sections[section["label"]] = data

    def as_tensor(data: bytes, planes: int, tokens: int) -> torch.Tensor:
        expected = planes * tokens * hidden * 2
        if len(data) != expected:
            raise KMeshFamilyCodecError("family section length mismatch")
        return (
            torch.frombuffer(bytearray(data), dtype=torch.int16)
            .reshape(planes, tokens, hidden)
            .view(torch.bfloat16)
        )

    master_index = header["master_index"]
    master_entry = header["members"][master_index]
    master = as_tensor(sections["master"], _PLANES, master_entry["tokens"])
    ms, me = master_entry["shared_span"]

    members: list[torch.Tensor] = []
    for index, entry in enumerate(header["members"]):
        tokens = entry["tokens"]
        start, end = entry["shared_span"]
        if not 0 <= start < end <= tokens or end - start != me - ms:
            raise KMeshFamilyCodecError(f"family member {index} span is inconsistent")
        if index == master_index:
            members.append(master.clone())
            continue
        private = entry["private_token_indices"]
        if private != list(range(0, start)) + list(range(end, tokens)):
            raise KMeshFamilyCodecError(
                f"family member {index} private index set is inconsistent"
            )
        tensor = torch.empty(_PLANES, tokens, hidden, dtype=torch.bfloat16)
        tensor[:, start:end, :] = master[:, ms:me, :]
        if private:
            tensor[:, private, :] = as_tensor(
                sections[f"{index}/private"], _PLANES, len(private)
            )
        for plane, name in ((0, "k"), (1, "v")):
            changed = entry[f"{name}_changed_aligned_indices"]
            if changed != sorted(set(changed)) or any(
                not 0 <= i < end - start for i in changed
            ):
                raise KMeshFamilyCodecError(
                    f"family member {index}/{name} changed indexes are invalid"
                )
            if changed:
                diff = as_tensor(
                    sections[f"{index}/{name}/shared_diff"], 1, len(changed)
                )
                tensor[plane, [start + i for i in changed], :] = diff[0]
        members.append(tensor)
    return members


class KMeshFamilyMasterMirrorSerializer(MultiSerializer):
    """Serialize one recovered family as master plus sparse mirrors."""

    def __init__(
        self,
        member_spans: tuple[tuple[int, int], ...],
        master_index: int = 0,
    ) -> None:
        if len(member_spans) < 2:
            raise KMeshFamilyCodecError("a family needs at least two members")
        if not 0 <= master_index < len(member_spans):
            raise KMeshFamilyCodecError("master index outside the family")
        self._spans = tuple((int(s), int(e)) for s, e in member_spans)
        self._master_index = master_index

    @property
    def group_size(self) -> int:
        return len(self._spans)

    def serialize(self, src: MemoryObjGroup, dst: MemoryObj, key: ObjectKey) -> int:
        # ``key`` unused: spans come from the reuse plan at construction.
        validate_group_size(src, len(self._spans), role="src")
        tensors: list[torch.Tensor] = []
        for index, obj in enumerate(src):
            if obj is None or obj.tensor is None:
                raise KMeshFamilyCodecError(
                    f"family member {index} must supply a tensor"
                )
            tensors.append(obj.tensor)
        if dst.tensor is None:
            raise KMeshFamilyCodecError("family serialize dst.tensor is None")
        blob = encode_family(tensors, self._spans, self._master_index)
        n = len(blob)
        if dst.tensor.numel() * dst.tensor.element_size() < n:
            raise KMeshFamilyCodecError(
                f"family dst capacity {dst.tensor.numel()} bytes below required {n}"
            )
        dst_view = dst.tensor.view(torch.uint8)
        dst_view[:n].copy_(torch.frombuffer(bytearray(blob), dtype=torch.uint8))
        return n

    def estimate_serialized_size(self, layout_descs: LayoutDescGroup) -> int:
        validate_group_size(layout_descs, len(self._spans), role="layout")
        total = 0
        for index, layout in enumerate(layout_descs):
            if layout is None:
                raise KMeshFamilyCodecError(f"family member {index} layout is required")
            member_bytes = sum(
                math.prod(shape) * dtype.itemsize
                for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
            )
            # Worst case a mirror degrades to dense plus its index list;
            # allow four bytes of JSON index per token plus a header
            # allowance per member.
            total += member_bytes + 4096
        tokens = max(end for _, end in self._spans)
        return total + 8 * tokens * len(self._spans) + len(MAGIC) + 8


class KMeshFamilyMasterMirrorDeserializer(MultiDeserializer):
    """Restore every family member bitwise from one serialized blob."""

    def __init__(self, group_size: int) -> None:
        if type(group_size) is not int or group_size < 2:
            raise KMeshFamilyCodecError("a family needs at least two members")
        self._group_size = group_size

    @property
    def group_size(self) -> int:
        return self._group_size

    def deserialize(self, src: MemoryObj, dst: MemoryObjGroup, key: ObjectKey) -> None:
        # ``key`` unused: the blob header is self-describing.
        validate_group_size(dst, self._group_size, role="dst")
        if src.tensor is None:
            raise KMeshFamilyCodecError("family deserialize src.tensor is None")
        blob = bytes(src.tensor.view(torch.uint8).numpy().tobytes())
        members = decode_family(blob)
        if len(members) != self._group_size:
            raise KMeshFamilyCodecError(
                f"family blob holds {len(members)} members, expected {self._group_size}"
            )
        for index, (member, obj) in enumerate(zip(members, dst, strict=True)):
            if obj is None or obj.tensor is None:
                raise KMeshFamilyCodecError(
                    f"family member {index} needs a destination tensor"
                )
            if obj.tensor.shape != member.shape or obj.tensor.dtype != member.dtype:
                raise KMeshFamilyCodecError(
                    f"family member {index} destination layout mismatch"
                )
            obj.tensor.copy_(member)
