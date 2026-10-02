# SPDX-License-Identifier: Apache-2.0
"""Header integrity via the codec CRC (V2 includes the header).

The payload CRC alone leaves the fixed header (dtype, scale_scheme, page_size,
payload-length fields, ...) unprotected: an intra-header bit-flip that still
parses as a structurally-valid field would be honoured silently.  For the
byte-through / RAW_UNIT path (``CodecVersion.V2``) — the one that is durably
shared across engines, where a mis-restored V is a silent corruption — the CRC
covers the whole header-up-to-the-CRC-field followed by the payload.

V1 (legacy scale-aware ``COMPUTED_PER_TENSOR``) keeps payload-only CRC coverage
so its blobs stay byte-identical to the pre-existing format; these tests pin
that asymmetry deliberately.
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
    UnsupportedConfigError,
    deserialize_header,
    serialize_header,
)

# Fixed-header field byte offsets (see _FIXED_HEADER_FMT = "<8sHHHHHH" + 7q).
# 8s magic (0); H version (8); H scale_scope (10); H k_dtype (12);
# H v_dtype (14); H scale_dtype (16); H scale_scheme (18); then int64s:
# layer_id (20), chunk_id (28), chunk_size (36), page_size (44),
# kv_head_count (52), head_dim (60), scale_shape_n (68).
_PAGE_SIZE_OFFSET = 44
_HEAD_DIM_OFFSET = 60
_PAYLOAD_LENGTHS_OFFSET = 76


def _make_enc(scheme: ScaleScheme, payload: bytes = b"\x11\x22\x33\x44") -> EncodedKV:
    """A minimal V-only blob (no K, no scales) with concrete layout fields."""
    return EncodedKV(
        k_dtype=torch.bfloat16,
        v_dtype=torch.float8_e4m3fn,
        scale_scheme=scheme,
        page_size=16,
        kv_head_count=8,
        head_dim=128,
        k_payload_len=0,
        v_payload_len=len(payload),
        scale_payload_len=0,
        payload=payload,
    )


def _blob(enc: EncodedKV) -> bytes:
    return serialize_header(enc) + bytes(enc.payload)


def _flip_byte(buf: bytes, offset: int) -> bytes:
    b = bytearray(buf)
    b[offset] ^= 0x01
    return bytes(b)


# --- V2 (RAW_UNIT byte-through): header IS covered ------------------------


def test_v2_clean_roundtrip_preserves_layout_fields() -> None:
    enc = _make_enc(ScaleScheme.RAW_UNIT)
    back = deserialize_header(_blob(enc))
    assert back.scale_scheme is ScaleScheme.RAW_UNIT
    assert back.page_size == 16
    assert back.head_dim == 128
    assert bytes(back.payload) == bytes(enc.payload)


def test_v2_crc_actually_incorporates_the_header() -> None:
    # The stored CRC must differ from a payload-only CRC, proving the header
    # is folded in (otherwise the V2 guarantee would be vacuous).
    enc = _make_enc(ScaleScheme.RAW_UNIT)
    header = serialize_header(enc)
    (stored_crc,) = struct.unpack_from("<I", header, len(header) - 4)
    payload_only = zlib.crc32(bytes(enc.payload)) & 0xFFFFFFFF
    assert stored_crc != payload_only


def test_v2_header_bitflip_page_size_is_caught() -> None:
    enc = _make_enc(ScaleScheme.RAW_UNIT)
    corrupt = _flip_byte(_blob(enc), _PAGE_SIZE_OFFSET)
    with pytest.raises(CorruptEncodedKVError, match="CRC mismatch"):
        deserialize_header(corrupt)


def test_v2_header_bitflip_head_dim_is_caught() -> None:
    enc = _make_enc(ScaleScheme.RAW_UNIT)
    corrupt = _flip_byte(_blob(enc), _HEAD_DIM_OFFSET)
    with pytest.raises(CorruptEncodedKVError, match="CRC mismatch"):
        deserialize_header(corrupt)


def test_v2_payload_bitflip_still_caught() -> None:
    # Folding the header in must not weaken payload coverage.
    enc = _make_enc(ScaleScheme.RAW_UNIT)
    blob = _blob(enc)
    corrupt = _flip_byte(blob, len(blob) - 1)  # last payload byte
    with pytest.raises(CorruptEncodedKVError, match="CRC mismatch"):
        deserialize_header(corrupt)


def test_v2_negative_payload_length_is_rejected_before_slicing() -> None:
    blob = bytearray(_blob(_make_enc(ScaleScheme.RAW_UNIT)))
    struct.pack_into("<q", blob, _PAYLOAD_LENGTHS_OFFSET, -1)
    with pytest.raises(CorruptEncodedKVError, match="negative payload length"):
        deserialize_header(bytes(blob))


def test_v2_trailing_bytes_are_rejected() -> None:
    blob = _blob(_make_enc(ScaleScheme.RAW_UNIT)) + b"trailing"
    with pytest.raises(CorruptEncodedKVError, match="trailing bytes"):
        deserialize_header(blob)


# --- V1 (legacy COMPUTED_PER_TENSOR): payload-only, back-compat -----------


def test_v1_header_bitflip_is_not_a_crc_error() -> None:
    # Deliberate back-compat asymmetry: V1 CRC covers payload only, so a
    # header bit-flip in a validation-free int64 field decodes without a CRC
    # error (proving V1 semantics are byte-identical to legacy).  head_dim has
    # no range check, so the flipped value round-trips through decode.
    enc = _make_enc(ScaleScheme.COMPUTED_PER_TENSOR)
    corrupt = _flip_byte(_blob(enc), _HEAD_DIM_OFFSET)
    back = deserialize_header(corrupt)  # must NOT raise
    assert back.head_dim != enc.head_dim  # the corruption was honoured silently


def test_v1_payload_bitflip_still_caught() -> None:
    enc = _make_enc(ScaleScheme.COMPUTED_PER_TENSOR)
    blob = _blob(enc)
    corrupt = _flip_byte(blob, len(blob) - 1)
    with pytest.raises(CorruptEncodedKVError, match="CRC mismatch"):
        deserialize_header(corrupt)


def test_v1_crc_is_payload_only() -> None:
    enc = _make_enc(ScaleScheme.COMPUTED_PER_TENSOR)
    header = serialize_header(enc)
    (stored_crc,) = struct.unpack_from("<I", header, len(header) - 4)
    payload_only = zlib.crc32(bytes(enc.payload)) & 0xFFFFFFFF
    assert stored_crc == payload_only


def test_v1_trailing_bytes_remain_backward_compatible() -> None:
    """Legacy writers could persist an over-estimated padded buffer."""
    enc = _make_enc(ScaleScheme.COMPUTED_PER_TENSOR)
    back = deserialize_header(_blob(enc) + b"legacy-padding")
    assert bytes(back.payload) == bytes(enc.payload)


def test_serialize_rejects_negative_payload_lengths() -> None:
    enc = _make_enc(ScaleScheme.RAW_UNIT)
    enc.k_payload_len = -1
    enc.v_payload_len += 1  # keep the old sum equal to len(payload)
    with pytest.raises(
        UnsupportedConfigError, match="payload lengths must be non-negative"
    ):
        serialize_header(enc)
