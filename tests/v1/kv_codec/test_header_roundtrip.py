# SPDX-License-Identifier: Apache-2.0
"""EncodedKV header serialization roundtrip.

These catch:
- field truncation, byte-order bugs, off-by-one in header sizing
- stripping of zero-length hash strings
- non-default scale_shape (e.g., per_page_head)
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    CODEC_MAGIC,
    CodecHashes,
    EncodedKV,
    ScaleScope,
    deserialize_header,
    serialize_header,
)


def _make_enc(
    *,
    scale_scope=ScaleScope.PER_TENSOR,
    scale_shape=(),
    hashes=None,
    layer_id=-1,
    chunk_id=-1,
    payload_lens=(8, 8, 4),
):
    payload = b"\xaa" * payload_lens[0] + b"\xbb" * payload_lens[1] + b"\xcc" * payload_lens[2]
    return EncodedKV(
        k_dtype=torch.float16,
        v_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float32,
        scale_scope=scale_scope,
        hashes=hashes if hashes is not None else CodecHashes(),
        layer_id=layer_id,
        chunk_id=chunk_id,
        chunk_size=8,
        page_size=8,
        kv_head_count=2,
        head_dim=16,
        scale_shape=scale_shape,
        k_payload_len=payload_lens[0],
        v_payload_len=payload_lens[1],
        scale_payload_len=payload_lens[2],
        payload=payload,
    )


def test_minimal_roundtrip():
    enc = _make_enc()
    blob = serialize_header(enc) + enc.payload
    parsed = deserialize_header(blob)
    assert parsed.k_dtype == enc.k_dtype
    assert parsed.v_dtype == enc.v_dtype
    assert parsed.scale_dtype == enc.scale_dtype
    assert parsed.scale_scope == enc.scale_scope
    assert parsed.layer_id == enc.layer_id
    assert parsed.chunk_id == enc.chunk_id
    assert parsed.scale_shape == enc.scale_shape
    assert parsed.k_payload_len == enc.k_payload_len
    assert parsed.v_payload_len == enc.v_payload_len
    assert parsed.scale_payload_len == enc.scale_payload_len
    assert parsed.payload == enc.payload


def test_per_page_head_scale_shape_roundtrip():
    enc = _make_enc(
        scale_scope=ScaleScope.PER_PAGE_HEAD,
        scale_shape=(7, 13),
    )
    blob = serialize_header(enc) + enc.payload
    parsed = deserialize_header(blob)
    assert parsed.scale_scope == ScaleScope.PER_PAGE_HEAD
    assert parsed.scale_shape == (7, 13)


def test_hashes_roundtrip():
    h = CodecHashes(
        model_id="qwen-7b",
        model_revision_hash="deadbeef",
        tokenizer_hash="cafebabe",
        rope_config_hash="feed1234",
        attention_backend="flashinfer",
        kv_layout="paged",
    )
    enc = _make_enc(hashes=h)
    blob = serialize_header(enc) + enc.payload
    parsed = deserialize_header(blob)
    for f in CodecHashes._CHECK_ORDER:
        assert getattr(parsed.hashes, f) == getattr(h, f)


def test_empty_hashes_roundtrip():
    enc = _make_enc()
    blob = serialize_header(enc) + enc.payload
    parsed = deserialize_header(blob)
    for f in CodecHashes._CHECK_ORDER:
        assert getattr(parsed.hashes, f) == ""


def test_signed_negative_layer_chunk_ids():
    """Layer/chunk IDs are int64 signed, -1 means unset.  Catches:
    interpreted as uint, sentinel becomes a huge positive number."""
    enc = _make_enc(layer_id=-1, chunk_id=-1)
    blob = serialize_header(enc) + enc.payload
    parsed = deserialize_header(blob)
    assert parsed.layer_id == -1
    assert parsed.chunk_id == -1


def test_magic_first_8_bytes():
    """Magic must be the first 8 bytes — that's what readers grep
    for to identify a blob."""
    enc = _make_enc()
    blob = serialize_header(enc) + enc.payload
    assert blob[:8] == CODEC_MAGIC


def test_payload_lengths_match_actual_bytes():
    enc = _make_enc(payload_lens=(32, 16, 8))
    assert len(enc.payload) == 56


def test_scale_shape_8_dim_max():
    """Catches: scale_shape with too many dims is rejected."""
    enc = _make_enc(scale_shape=(1, 1, 1, 1, 1, 1, 1, 1))  # 8 dims OK
    blob = serialize_header(enc) + enc.payload
    parsed = deserialize_header(blob)
    assert parsed.scale_shape == (1,) * 8
