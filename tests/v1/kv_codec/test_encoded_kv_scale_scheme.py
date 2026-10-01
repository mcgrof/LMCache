# SPDX-License-Identifier: Apache-2.0
"""CO1: ScaleScheme header tag in the KV codec (byte-through vs scale-aware).

The formerly-reserved 6th uint16 header slot is repurposed as a
:class:`ScaleScheme` enum with fail-closed decode. Legacy blobs (which wrote 0
into that slot) must keep decoding as ``COMPUTED_PER_TENSOR``; an unknown value
must raise rather than be silently ignored.
"""

# Future
from __future__ import annotations

# Standard
import struct
import zlib

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    CorruptEncodedKVError,
    EncodedKV,
    ScaleScheme,
    deserialize_header,
    serialize_header,
)
from lmcache.v1.kv_codec.encoded_kv import _FIXED_HEADER_LEN

# Byte offsets in the fixed header. magic "8s" occupies bytes 0-7; then six
# uint16: version (8), scale_scope (10), k_dtype (12), v_dtype (14),
# scale_dtype (16), scale_scheme (18).
_VERSION_OFFSET = 8
_SCHEME_OFFSET = 18


def _make_enc(scheme: ScaleScheme, payload: bytes = b"\x01\x02\x03\x04") -> EncodedKV:
    return EncodedKV(
        k_dtype=torch.bfloat16,
        v_dtype=torch.float8_e4m3fn,
        scale_scheme=scheme,
        k_payload_len=0,
        v_payload_len=len(payload),
        scale_payload_len=0,
        payload=payload,
    )


def _roundtrip(enc: EncodedKV) -> EncodedKV:
    header = serialize_header(enc)
    return deserialize_header(header + bytes(enc.payload))


def test_scale_scheme_roundtrips_raw_unit() -> None:
    back = _roundtrip(_make_enc(ScaleScheme.RAW_UNIT))
    assert back.scale_scheme is ScaleScheme.RAW_UNIT


def test_scale_scheme_defaults_to_computed_per_tensor() -> None:
    # A fresh EncodedKV (no scheme given) is the legacy scale-aware scheme.
    enc = EncodedKV(k_dtype=torch.bfloat16, v_dtype=torch.bfloat16)
    assert enc.scale_scheme is ScaleScheme.COMPUTED_PER_TENSOR


def test_legacy_zero_slot_decodes_as_computed_per_tensor() -> None:
    # A header whose scheme slot is literally 0 (what legacy writers emit)
    # must decode as COMPUTED_PER_TENSOR, not raise.
    enc = _make_enc(ScaleScheme.COMPUTED_PER_TENSOR)
    header = bytearray(serialize_header(enc))
    assert struct.unpack_from("<H", header, _SCHEME_OFFSET)[0] == 0
    back = deserialize_header(bytes(header) + bytes(enc.payload))
    assert back.scale_scheme is ScaleScheme.COMPUTED_PER_TENSOR


def _reseal_v2_crc(header: bytearray, payload: bytes) -> None:
    """Recompute the V2 header CRC (header-up-to-CRC ‖ payload) in place.

    Since #36 the V2 CRC covers the header, so a forged header byte trips the
    CRC before any field-level guard.  Re-sealing makes the blob self-consistent
    again so a test can exercise a specific decode guard in isolation.
    """
    crc = zlib.crc32(bytes(header[:-4]))
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    struct.pack_into("<I", header, len(header) - 4, crc)


def test_unknown_scale_scheme_raises_corrupt() -> None:
    # Forge an unknown scheme, then re-seal the V2 CRC so decode reaches the
    # scheme guard (rather than tripping the header CRC first) and fails closed.
    enc = _make_enc(ScaleScheme.RAW_UNIT)
    header = bytearray(serialize_header(enc))
    struct.pack_into("<H", header, _SCHEME_OFFSET, 99)  # not a valid scheme
    _reseal_v2_crc(header, bytes(enc.payload))
    with pytest.raises(CorruptEncodedKVError, match="scale_scheme"):
        deserialize_header(bytes(header) + bytes(enc.payload))


def test_fixed_header_length_unchanged() -> None:
    # Repurposing the reserved slot must not change the 76-byte fixed region.
    # A COMPUTED blob (V1) and a RAW_UNIT blob (V2) differ in the version byte
    # and the scheme byte within the fixed header; no length drift.
    assert _FIXED_HEADER_LEN == 76
    a = serialize_header(_make_enc(ScaleScheme.COMPUTED_PER_TENSOR))
    b = serialize_header(_make_enc(ScaleScheme.RAW_UNIT))
    assert len(a) == len(b)
    differing = {i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y}
    # Inside the fixed header, only version + scheme differ.
    fixed_diffs = {i for i in differing if i < _FIXED_HEADER_LEN}
    assert fixed_diffs == {_VERSION_OFFSET, _SCHEME_OFFSET}
    # The trailing 4-byte CRC also differs: since #36 the V2 CRC folds the
    # header in, so it no longer equals V1's payload-only CRC.
    crc_offsets = set(range(len(a) - 4, len(a)))
    assert differing - {_VERSION_OFFSET, _SCHEME_OFFSET} == crc_offsets


def test_raw_unit_is_v2_and_computed_is_v1() -> None:
    # RAW_UNIT bumps the codec version so a V1-only reader rejects it;
    # COMPUTED_PER_TENSOR stays V1 (byte-identical to the legacy format).
    raw = serialize_header(_make_enc(ScaleScheme.RAW_UNIT))
    computed = serialize_header(_make_enc(ScaleScheme.COMPUTED_PER_TENSOR))
    assert struct.unpack_from("<H", raw, _VERSION_OFFSET)[0] == 2
    assert struct.unpack_from("<H", computed, _VERSION_OFFSET)[0] == 1


def test_unknown_version_is_rejected() -> None:
    # The mechanism that makes the bumped version safe: a reader rejects a
    # version it does not know. This is how a V1-only reader fails closed on a
    # V2 RAW_UNIT blob instead of misreading the raw fp8 bytes as scale-aware.
    enc = _make_enc(ScaleScheme.RAW_UNIT)
    header = bytearray(serialize_header(enc))
    struct.pack_into("<H", header, _VERSION_OFFSET, 99)
    with pytest.raises(CorruptEncodedKVError, match="codec version"):
        deserialize_header(bytes(header) + bytes(enc.payload))


def test_v1_nonzero_scheme_slot_is_rejected() -> None:
    # A V1 blob whose legacy (reserved) scheme slot is non-zero is corrupt/forged.
    enc = _make_enc(ScaleScheme.COMPUTED_PER_TENSOR)  # written as V1
    header = bytearray(serialize_header(enc))
    assert struct.unpack_from("<H", header, _VERSION_OFFSET)[0] == 1
    struct.pack_into("<H", header, _SCHEME_OFFSET, 1)  # forge RAW_UNIT in a V1 blob
    with pytest.raises(CorruptEncodedKVError, match="reserved/scale_scheme"):
        deserialize_header(bytes(header) + bytes(enc.payload))
