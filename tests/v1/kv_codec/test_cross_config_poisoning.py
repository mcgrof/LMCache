# SPDX-License-Identifier: Apache-2.0
"""Cross-config cache poisoning gates.

If a cache entry was written under model A and a later run requests
it under model B, the codec must refuse rather than producing a
plausibly-shaped tensor with the wrong semantics.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    CodecHashes,
    CodecMismatchError,
)


@pytest.fixture
def codec():
    return AsymK16V8Codec()


@pytest.fixture
def kv_pair():
    k = torch.randn(2, 8, 4, 16, dtype=torch.float16)
    v = torch.randn(2, 8, 4, 16, dtype=torch.float16)
    return k, v


def _encode_under(codec, kv_pair, **hash_kwargs):
    k, v = kv_pair
    return codec.to_bytes(
        codec.encode(k, v, hashes=CodecHashes(**hash_kwargs))
    )


def test_model_id_mismatch_rejected(codec, kv_pair):
    blob = _encode_under(codec, kv_pair, model_id="qwen2.5-7b")
    with pytest.raises(CodecMismatchError, match="model_id"):
        codec.from_bytes(
            blob, expected_hashes=CodecHashes(model_id="qwen3-7b")
        )


def test_tokenizer_hash_mismatch_rejected(codec, kv_pair):
    blob = _encode_under(codec, kv_pair, tokenizer_hash="abc123")
    with pytest.raises(CodecMismatchError, match="tokenizer_hash"):
        codec.from_bytes(
            blob, expected_hashes=CodecHashes(tokenizer_hash="zzz999")
        )


def test_rope_config_hash_mismatch_rejected(codec, kv_pair):
    blob = _encode_under(codec, kv_pair, rope_config_hash="rope-1")
    with pytest.raises(CodecMismatchError, match="rope_config_hash"):
        codec.from_bytes(
            blob, expected_hashes=CodecHashes(rope_config_hash="rope-2")
        )


def test_attention_backend_mismatch_rejected(codec, kv_pair):
    blob = _encode_under(codec, kv_pair, attention_backend="flashinfer")
    with pytest.raises(CodecMismatchError, match="attention_backend"):
        codec.from_bytes(
            blob, expected_hashes=CodecHashes(attention_backend="flash_attn")
        )


def test_kv_layout_mismatch_rejected(codec, kv_pair):
    blob = _encode_under(codec, kv_pair, kv_layout="paged")
    with pytest.raises(CodecMismatchError, match="kv_layout"):
        codec.from_bytes(
            blob, expected_hashes=CodecHashes(kv_layout="contiguous")
        )


def test_no_expected_hashes_no_check(codec, kv_pair):
    """If the caller passes no expected_hashes, we don't gate."""
    blob = _encode_under(codec, kv_pair, model_id="qwen2.5-7b")
    enc = codec.from_bytes(blob)  # no expected_hashes
    assert enc.hashes.model_id == "qwen2.5-7b"


def test_empty_expected_field_is_wildcard(codec, kv_pair):
    """expected_hashes with empty model_id should not gate model_id."""
    blob = _encode_under(codec, kv_pair, model_id="qwen2.5-7b")
    enc = codec.from_bytes(
        blob, expected_hashes=CodecHashes(model_id="")
    )
    assert enc.hashes.model_id == "qwen2.5-7b"


def test_match_succeeds(codec, kv_pair):
    blob = _encode_under(
        codec,
        kv_pair,
        model_id="qwen2.5-7b",
        tokenizer_hash="abc",
    )
    enc = codec.from_bytes(
        blob,
        expected_hashes=CodecHashes(
            model_id="qwen2.5-7b", tokenizer_hash="abc"
        ),
    )
    assert enc.hashes.model_id == "qwen2.5-7b"
