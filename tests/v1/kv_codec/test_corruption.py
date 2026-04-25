# SPDX-License-Identifier: Apache-2.0
"""Header / payload corruption detection.

Each test deliberately damages a serialized blob and verifies that
deserialization raises CorruptEncodedKVError, not silent garbage.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    CodecHashes,
    CorruptEncodedKVError,
    EncodedKV,
    ScaleScope,
    deserialize_header,
    serialize_header,
)


def _blob():
    payload = b"\xaa" * 8 + b"\xbb" * 8 + b"\xcc" * 4
    enc = EncodedKV(
        k_dtype=torch.float16,
        v_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float32,
        scale_scope=ScaleScope.PER_TENSOR,
        hashes=CodecHashes(),
        scale_shape=(),
        k_payload_len=8,
        v_payload_len=8,
        scale_payload_len=4,
        payload=payload,
    )
    return bytearray(serialize_header(enc) + enc.payload)


def test_truncated_buffer_raises():
    blob = _blob()
    # Trim to a clearly-truncated size.
    with pytest.raises(CorruptEncodedKVError):
        deserialize_header(bytes(blob[:32]))


def test_bad_magic_raises():
    blob = _blob()
    blob[0] = blob[0] ^ 0xFF
    with pytest.raises(CorruptEncodedKVError, match="bad magic"):
        deserialize_header(bytes(blob))


def test_bad_version_raises():
    blob = _blob()
    # Version field starts at offset 8 (after magic), 2 bytes LE.
    blob[8] = 0xFF
    blob[9] = 0xFF
    with pytest.raises(CorruptEncodedKVError, match="version"):
        deserialize_header(bytes(blob))


def test_bad_dtype_id_raises():
    """An unknown dtype index must error, not silently map to None."""
    blob = _blob()
    # k_dtype_id is at offset 12 (8 magic + 2 version + 2 scope = 12).
    blob[12] = 0xFE
    blob[13] = 0xFF
    with pytest.raises(CorruptEncodedKVError, match="unknown dtype"):
        deserialize_header(bytes(blob))


def test_payload_corruption_caught_by_crc():
    blob = _blob()
    # Flip a byte deep in the payload.  Header is fixed-size +
    # variable scale_shape + payload_lens(24) + hashes (~84) + crc(4)
    # so the payload starts well past byte 100 in this minimal blob.
    payload_start_approx = len(blob) - 20  # near end of blob
    blob[payload_start_approx] ^= 0xFF
    with pytest.raises(CorruptEncodedKVError, match="CRC"):
        deserialize_header(bytes(blob))


def test_payload_truncated_mid_v():
    blob = _blob()
    # Drop last byte of the payload.
    with pytest.raises(CorruptEncodedKVError, match="truncated"):
        deserialize_header(bytes(blob[:-1]))


def test_implausible_scale_shape_rejected():
    blob = _blob()
    # scale_shape_n is the LAST int64 in the fixed header.
    # Fixed header layout: 8 magic + 2*6 shorts + 8*7 int64 = 76 bytes,
    # scale_shape_n occupies bytes [68, 76).  Set it implausibly large.
    blob[68:76] = (9999).to_bytes(8, "little", signed=True)
    with pytest.raises(CorruptEncodedKVError, match="scale_shape_n"):
        deserialize_header(bytes(blob))


def test_string_length_mismatch_rejected():
    """A hash key declares length N but only M < N bytes follow."""
    blob = _blob()
    # Find the hash section (after CRC field at offset N).  We
    # exploit that for the empty-hashes case, all hash strings are
    # length 0.  Instead, write a corrupted blob that declares a
    # 256-byte hash key while the actual key is much shorter.
    #
    # Easier: manually build a malformed header.
    # Standard
    import struct
    fixed = (
        b"LMCKV\x01\x00\x01"           # magic
        + struct.pack("<HHHHHH", 1, 0, 2, 7, 4, 0)  # ver, scope, k=fp16, v=fp8, scale=fp32, reserved
        + struct.pack("<qqqqqqq", -1, -1, -1, -1, -1, -1, 0)  # ids, no scale dims
        + struct.pack("<qqq", 0, 0, 0)  # zero payload lengths
        + struct.pack("<H", 1)  # one hash pair
        + struct.pack("<H", 0xFFFF)  # claims 65535-byte key but...
        + b"X"  # ...only one byte
    )
    with pytest.raises(CorruptEncodedKVError):
        deserialize_header(fixed)
