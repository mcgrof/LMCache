# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the asymmetric K16/V8 *split-tier* (V-only) multi-output
serde in ``lmcache/v1/distributed/serde/asym_k16_v8.py``.

Validates the Mode 2 / V-only-write path:

* ``serialize`` accepts ``src = (None, V)``: K is *not* written to the
  byte buffer (the K slot must be ``None``); only V_fp8 + scales hit
  the durable storage.
* ``deserialize`` accepts ``src = blob``, ``dst = (None|K_skip, V_out)``:
  V is restored from the blob's FP8 + scales; the K slot is a no-op
  regardless of whether the caller provides one.

The byte-ratio claim under test is paper Eq.~4: the V-only blob is
``V_8 / (K_16 + V_16) = 1/4`` of FP16 KV — equivalently ``1/3`` of the
all-NVMe Mode 1 (K16/V8) blob.

The ``Mode 1 vs Mode 2 mixing`` cases are also covered: a V-only
deserializer must refuse a Mode 1 blob (K is present), and a Mode 1
deserializer would mis-decode a V-only blob (only Mode 2 deserializer
recognizes ``k_payload_len = 0``).
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.asym_k16_v8 import (
    AsymK16V8MultiSerializer,
    AsymK16V8VOnlyMultiDeserializer,
    AsymK16V8VOnlyMultiSerializer,
)


# Mirror the _FakeMemoryObj used elsewhere; lets the test stay GPU-free
# and L1Manager-free.


@dataclass
class _FakeMemoryObj:
    tensor: Optional[torch.Tensor]


def _bf16_tensor(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=torch.bfloat16, generator=g).contiguous()


def _byte_buffer(num_bytes: int) -> _FakeMemoryObj:
    return _FakeMemoryObj(tensor=torch.zeros(num_bytes, dtype=torch.uint8))


# Llama-3.1-8B-Instruct-shaped chunk; same dimensions as the
# storage-only-dequant tests for direct ratio comparability.
_LLAMA_KV_SHAPE = (32, 64, 8, 128)


# =============================================================================
# group_size and contract surface
# =============================================================================


def test_group_size_is_two_on_both_endpoints() -> None:
    s = AsymK16V8VOnlyMultiSerializer()
    d = AsymK16V8VOnlyMultiDeserializer()
    assert s.group_size == 2
    assert d.group_size == 2


def test_serialize_rejects_k_present() -> None:
    """K slot must be None — providing K is a contract error."""
    s = AsymK16V8VOnlyMultiSerializer()
    k = _FakeMemoryObj(tensor=_bf16_tensor(2, 4, seed=0))
    v = _FakeMemoryObj(tensor=_bf16_tensor(2, 4, seed=1))
    buf = _byte_buffer(64)
    with pytest.raises(ValueError, match="K slot must be None"):
        s.serialize((k, v), buf)


def test_serialize_requires_v() -> None:
    s = AsymK16V8VOnlyMultiSerializer()
    buf = _byte_buffer(64)
    with pytest.raises(ValueError, match="V slot is required"):
        s.serialize((None, None), buf)
    with pytest.raises(ValueError, match="V slot is required"):
        s.serialize((None, _FakeMemoryObj(tensor=None)), buf)


def test_estimate_serialized_size_rejects_k_layout_present() -> None:
    s = AsymK16V8VOnlyMultiSerializer()
    v_layout = MemoryLayoutDesc(shapes=[torch.Size([2, 4])], dtypes=[torch.bfloat16])
    k_layout = MemoryLayoutDesc(shapes=[torch.Size([2, 4])], dtypes=[torch.bfloat16])
    with pytest.raises(ValueError, match="K layout must be None"):
        s.estimate_serialized_size((k_layout, v_layout))
    with pytest.raises(ValueError, match="V layout is required"):
        s.estimate_serialized_size((None, None))


# =============================================================================
# Round-trip: V-only write, V-only read
# =============================================================================


def test_v_only_round_trip_v_within_fp8_noise() -> None:
    s = AsymK16V8VOnlyMultiSerializer()
    d = AsymK16V8VOnlyMultiDeserializer()

    v = _bf16_tensor(*_LLAMA_KV_SHAPE, seed=2)

    layout = (
        None,
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    capacity = s.estimate_serialized_size(layout)
    buf = _byte_buffer(capacity)
    n = s.serialize((None, _FakeMemoryObj(tensor=v)), buf)
    assert 0 < n <= capacity

    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v))
    d.deserialize(buf, (None, v_out))

    v_diff = (v_out.tensor.float() - v.float()).abs()
    rel = v_diff / (v.float().abs() + 1e-6)
    assert rel.median().item() < 0.075, (
        f"V-only relative error median {rel.median().item():.4f} "
        f"exceeds FP8 noise threshold 0.075"
    )


def test_v_only_deserialize_ignores_k_slot() -> None:
    """A non-None K dst slot is a no-op for the V-only deserializer."""
    s = AsymK16V8VOnlyMultiSerializer()
    d = AsymK16V8VOnlyMultiDeserializer()

    v = _bf16_tensor(2, 4, 8, 64, seed=3)
    layout = (None, MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]))
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    s.serialize((None, _FakeMemoryObj(tensor=v)), buf)

    sentinel = torch.full((2, 4, 8, 64), fill_value=42.0, dtype=torch.bfloat16)
    k_unused = _FakeMemoryObj(tensor=sentinel.clone())
    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v))
    d.deserialize(buf, (k_unused, v_out))

    # K must be untouched.
    assert torch.equal(k_unused.tensor, sentinel)


# =============================================================================
# Byte ratio: Mode 2 == 1/3 of Mode 1 (paper Eq. 4)
# =============================================================================


def test_v_only_blob_is_one_third_of_mode1_blob() -> None:
    """V-only blob bytes = V_8 + scales + small header.

    Mode 1 blob bytes = K_16 + V_8 + scales + small header.  The
    layout-invariant ratio Mode 2 / Mode 1 = V_8 / (K_16 + V_8) = 1/3.
    Test on a chunk size where the headers are negligible relative to
    the payload so the ratio resolves cleanly.
    """
    mode1_s = AsymK16V8MultiSerializer()
    v_only_s = AsymK16V8VOnlyMultiSerializer()

    # Big-enough chunk that header overhead is < 0.5%.
    k = _bf16_tensor(*_LLAMA_KV_SHAPE, seed=4)
    v = _bf16_tensor(*_LLAMA_KV_SHAPE, seed=5)

    m1_layout = (
        MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    m2_layout = (None, MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]))

    m1_buf = _byte_buffer(mode1_s.estimate_serialized_size(m1_layout))
    m2_buf = _byte_buffer(v_only_s.estimate_serialized_size(m2_layout))
    m1_n = mode1_s.serialize(
        (_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v)), m1_buf
    )
    m2_n = v_only_s.serialize((None, _FakeMemoryObj(tensor=v)), m2_buf)

    ratio = m2_n / m1_n
    # Paper Eq. 4: 1/3 = 0.333... ; tolerance for header overhead.
    assert abs(ratio - 1.0 / 3.0) < 0.005, (
        f"Mode2/Mode1 byte ratio {ratio:.4f} should be ~0.3333 (Eq.~4); "
        f"m1_bytes={m1_n}, m2_bytes={m2_n}"
    )


# =============================================================================
# Cross-mode: Mode 1 deserializer should refuse a Mode 2 blob and v.v.
# =============================================================================


def test_v_only_deserializer_refuses_mode1_blob() -> None:
    """A Mode 1 blob has k_payload_len > 0; the V-only deserializer
    must reject it rather than silently mis-decode."""
    mode1_s = AsymK16V8MultiSerializer()
    v_only_d = AsymK16V8VOnlyMultiDeserializer()

    k = _bf16_tensor(2, 4, 8, 64, seed=6)
    v = _bf16_tensor(2, 4, 8, 64, seed=7)
    layout = (
        MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    buf = _byte_buffer(mode1_s.estimate_serialized_size(layout))
    mode1_s.serialize((_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v)), buf)

    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v))
    with pytest.raises(ValueError, match="Mode 1 / storage-only-dequant"):
        v_only_d.deserialize(buf, (None, v_out))
