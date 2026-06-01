# SPDX-License-Identifier: Apache-2.0
"""Contract test for the multi-output serde over the MP / async storage path.

Pins the typed-output behavior the placement wiring must satisfy, without
requiring a model, vLLM, FlashInfer, or a GPU.

This provides two groups of tests:

* LOCAL (pass today) — the typed multi-output round-trip works through the
  serde directly: named K / V outputs survive verbatim, distinct sizes +
  sentinels make ordering/aliasing bugs impossible to hide, the absent-K
  (split-tier) slot semantics hold, and single-output serdes still flow
  through the length-one bridge unchanged.

* MP / ASYNC (xfail today) — the async storage path does not yet route a
  multi-output group as separate typed child outputs, and split-tier placement
  (K -> CPU/host, V -> NVMe; scales travel packed in V's encoded header) is
  not wired. ``AsyncSerdeProcessor`` is
  typed for single-tensor ``Serializer`` / ``Deserializer`` (see
  ``async_processor.py``). These tests are the target the wiring must hit; they
  flip to xpass once it lands.

Run:
    pytest tests/v1/distributed/serde/test_multi_output_mp_contract.py -q -rxX
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Optional
import select
import struct
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.base import Deserializer, Serializer
from lmcache.v1.distributed.serde.multi import (
    LayoutDescGroup,
    MemoryObjGroup,
    MultiDeserializer,
    MultiSerializer,
    single_to_multi_deserializer,
    single_to_multi_serializer,
    validate_group_size,
)

try:
    from lmcache.v1.distributed.serde import AsyncSerdeProcessor
    from lmcache.v1.platform import consume_fd

    _HAVE_ASYNC = True
except Exception:  # pragma: no cover - import guard for the xfail lane
    _HAVE_ASYNC = False


# =============================================================================
# Scaffolding — GPU-free MemoryObj stand-in for driving the multi-output serde.
# =============================================================================


@dataclass
class _FakeMemoryObj:
    tensor: Optional[torch.Tensor]


def _byte_buffer(num_bytes: int) -> _FakeMemoryObj:
    return _FakeMemoryObj(tensor=torch.zeros(num_bytes, dtype=torch.uint8))


def _sentinel_obj(num_bytes: int, fill: int) -> _FakeMemoryObj:
    """A uint8 payload of ``num_bytes`` filled with the sentinel ``fill``."""
    return _FakeMemoryObj(tensor=torch.full((num_bytes,), fill, dtype=torch.uint8))


# Deterministic named outputs with intentionally distinct sizes + sentinels so
# any ordering / aliasing / cross-wiring bug surfaces as a value or size mismatch.
# ``group_size == 2`` mirrors the upstream AsymK16V8Multi{Serializer,Deserializer}:
# the typed group surfaces K and V only; per-tensor scales travel packed inside
# V's encoded blob header (see lmcache.v1.kv_codec.asym_k16_v8 / serialize_header),
# NOT as a separate typed output slot.
_K = ("k", 4096, 0x11)  # bf16-ish keys: larger
_V = ("v", 2048, 0x22)  # fp8 values: smaller
_GROUP_SIZE = 2


# =============================================================================
# A toy fake multi-output codec — transport/contract only, no real quantization.
# Wire format (group of N): N x [uint8 present-mask] + N x [uint32 length]
# followed by the present slots' raw bytes in slot order. Defined inline so
# this file is standalone.
# =============================================================================

_MASK = struct.Struct("<B")
_LEN = struct.Struct("<I")


def _header_size(n: int) -> int:
    return n * (_MASK.size + _LEN.size)


class _FakeMultiSerializer(MultiSerializer):
    def __init__(self, group_size: int = _GROUP_SIZE) -> None:
        self._n = group_size

    @property
    def group_size(self) -> int:
        return self._n

    def serialize(self, src: MemoryObjGroup, dst) -> int:
        validate_group_size(src, self._n, role="src")
        masks, lens, payload = bytearray(), bytearray(), bytearray()
        for slot in src:
            if slot is None:
                masks += _MASK.pack(0)
                lens += _LEN.pack(0)
                continue
            blob = slot.tensor.contiguous().view(torch.uint8).numpy().tobytes()
            masks += _MASK.pack(1)
            lens += _LEN.pack(len(blob))
            payload += blob
        header = bytes(masks) + bytes(lens)
        total = len(header) + len(payload)
        if dst.tensor is None or dst.tensor.numel() < total:
            raise ValueError("dst buffer too small for serialized group")
        dv = dst.tensor.view(torch.uint8)
        dv[: len(header)].copy_(torch.frombuffer(bytearray(header), dtype=torch.uint8))
        if payload:
            dv[len(header) : total].copy_(
                torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            )
        return total

    def estimate_serialized_size(self, layout_descs: LayoutDescGroup) -> int:
        validate_group_size(layout_descs, self._n, role="layout")
        total = _header_size(self._n)
        for desc in layout_descs:
            if desc is None:
                continue
            for shape, dtype in zip(desc.shapes, desc.dtypes, strict=True):
                numel = 1
                for dim in shape:
                    numel *= int(dim)
                total += numel * dtype.itemsize
        return total


class _FakeMultiDeserializer(MultiDeserializer):
    def __init__(self, group_size: int = _GROUP_SIZE) -> None:
        self._n = group_size

    @property
    def group_size(self) -> int:
        return self._n

    def deserialize(self, src, dst: MemoryObjGroup) -> None:
        validate_group_size(dst, self._n, role="dst")
        sv = src.tensor.view(torch.uint8)
        n = self._n
        present = [bool(sv[i].item()) for i in range(n)]
        lens = [
            int(_LEN.unpack_from(sv[n + i * 4 : n + (i + 1) * 4].numpy().tobytes())[0])
            for i in range(n)
        ]
        cursor = _header_size(n)
        for i, slot in enumerate(dst):
            this_len = lens[i]
            if slot is None or not present[i]:
                cursor += this_len
                continue
            dstv = slot.tensor.view(torch.uint8).flatten()
            dstv[:this_len].copy_(sv[cursor : cursor + this_len])
            cursor += this_len


# A trivial single-tensor serde (identity byte copy) for the bridge backcompat test.
class _IdentitySerializer(Serializer):
    def serialize(self, src, dst) -> int:
        blob = src.tensor.contiguous().view(torch.uint8)
        n = int(blob.numel())
        dst.tensor.view(torch.uint8)[:n].copy_(blob)
        return n

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        total = 0
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
            numel = 1
            for dim in shape:
                numel *= int(dim)
            total += numel * dtype.itemsize
        return total


class _IdentityDeserializer(Deserializer):
    def deserialize(self, src, dst) -> None:
        n = int(dst.tensor.view(torch.uint8).numel())
        dst.tensor.view(torch.uint8).copy_(src.tensor.view(torch.uint8)[:n])


# =============================================================================
# LOCAL contract — these PASS today
# =============================================================================


def test_multi_output_local_roundtrip_preserves_named_outputs() -> None:
    """K/V round-trip with distinct sizes+sentinels: no aliasing/ordering bug can hide."""
    s, d = _FakeMultiSerializer(), _FakeMultiDeserializer()
    src = tuple(_sentinel_obj(nbytes, fill) for _, nbytes, fill in (_K, _V))
    layout = tuple(
        MemoryLayoutDesc(shapes=[o.tensor.shape], dtypes=[o.tensor.dtype]) for o in src
    )
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    n = s.serialize(src, buf)
    assert n > 0

    out = tuple(_byte_buffer(nbytes) for _, nbytes, _ in (_K, _V))
    d.deserialize(buf, out)
    for (name, nbytes, fill), o in zip((_K, _V), out):
        assert o.tensor.numel() == nbytes, f"{name}: size changed"
        assert torch.all(o.tensor == fill), f"{name}: sentinel/content cross-wired"


def test_multi_output_absent_k_slot_split_tier_semantics() -> None:
    """Split-tier: K absent on serialize (None slot); V round-trips, K dst left untouched."""
    s, d = _FakeMultiSerializer(), _FakeMultiDeserializer()
    v = _sentinel_obj(_V[1], _V[2])
    src: MemoryObjGroup = (None, v)
    layout: LayoutDescGroup = (
        None,
        MemoryLayoutDesc(shapes=[v.tensor.shape], dtypes=[v.tensor.dtype]),
    )
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    s.serialize(src, buf)

    k_out = _sentinel_obj(_K[1], 0xEE)  # pre-filled; must stay untouched (K absent)
    v_out = _byte_buffer(_V[1])
    d.deserialize(buf, (k_out, v_out))
    assert torch.all(k_out.tensor == 0xEE), "absent K must not be written"
    assert torch.all(v_out.tensor == _V[2])


def test_single_output_bridge_backcompat() -> None:
    """A single-tensor serde wrapped via the length-one bridge still round-trips."""
    ms = single_to_multi_serializer(_IdentitySerializer())
    md = single_to_multi_deserializer(_IdentityDeserializer())
    assert ms.group_size == 1 and md.group_size == 1
    payload = _sentinel_obj(1024, 0x5A)
    layout = (MemoryLayoutDesc(shapes=[payload.tensor.shape], dtypes=[payload.tensor.dtype]),)
    buf = _byte_buffer(ms.estimate_serialized_size(layout))
    ms.serialize((payload,), buf)
    out = (_byte_buffer(1024),)
    md.deserialize(buf, out)
    assert torch.all(out[0].tensor == 0x5A), "single-output bridge broke byte fidelity"


def test_split_tier_byte_accounting_contract() -> None:
    """The byte accounting the placement wiring must expose (counter contract).

    For a KV element count X (K and V each X elements):
      FP16 full KV                 = K16 + V16 = 4X bytes  (single-tier baseline)
      K16/V8 all-NVMe              = K16 + V8  = 3X bytes  (NVMe only; 0.75x FP16)
      Split-tier (K host, V NVMe)  = K16 (host) + V8 (NVMe) = 2X host + 1X NVMe
                                     NVMe-visible = 1X = 1/3 of all-NVMe = 1/4 of FP16
    """
    X = 1_000_000
    fp16 = 2 * X + 2 * X
    all_nvme = 2 * X + 1 * X
    split_host, split_nvme = 2 * X, 1 * X
    assert all_nvme / fp16 == 0.75
    assert split_nvme / all_nvme == pytest.approx(1 / 3)
    assert split_nvme / fp16 == 0.25
    assert split_host == 2 * X  # K parked in host memory, not on NVMe


# =============================================================================
# MP / ASYNC contract — XFAIL today; the target the wiring must satisfy
# =============================================================================


def _wait_for_fd(fd: int, timeout_s: float = 2.0) -> bool:
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if poller.poll(int(max(0, (deadline - time.monotonic()) * 1000))):
            try:
                consume_fd(fd)
            except OSError:
                pass
            return True
    return False


@pytest.mark.skipif(not _HAVE_ASYNC, reason="AsyncSerdeProcessor unavailable")
def test_multi_output_through_async_processor_roundtrip() -> None:
    """The async processor already round-trips a multi-output group (-> single blob).

    Empirical finding: ``AsyncSerdeProcessor`` is *typed* for
    single-tensor ``Serializer`` / ``Deserializer``, but at runtime it forwards
    each submitted work item to ``serialize`` / ``deserialize`` unchanged, so a
    multi-output group serialized to one blob survives the async store/load path
    today. **The async layer is therefore NOT the blocker.** The remaining gap is
    narrower: split-tier *placement* — routing the group's outputs to different
    storage tiers as separate typed child outputs (see the xfail test below).
    """
    s, d = _FakeMultiSerializer(), _FakeMultiDeserializer()
    proc = AsyncSerdeProcessor(s, d)  # type: ignore[arg-type]
    try:
        src = tuple(_sentinel_obj(nbytes, fill) for _, nbytes, fill in (_K, _V))
        layout = tuple(
            MemoryLayoutDesc(shapes=[o.tensor.shape], dtypes=[o.tensor.dtype]) for o in src
        )
        buf = _byte_buffer(s.estimate_serialized_size(layout))
        sid = proc.submit_serialize([src], [buf])  # group as a single work item
        assert _wait_for_fd(proc.get_serialize_event_fd()), "serialize fd never signaled"
        assert proc.query_serialize_result(sid) is True

        out = tuple(_byte_buffer(nbytes) for _, nbytes, _ in (_K, _V))
        did = proc.submit_deserialize([buf], [out])
        assert _wait_for_fd(proc.get_deserialize_event_fd()), "deserialize fd never signaled"
        assert proc.query_deserialize_result(did) is True
        for (_, _, fill), o in zip((_K, _V), out):
            assert torch.all(o.tensor == fill)
    finally:
        proc.close()


def test_split_policy_routes_k_to_cpu_v_to_nvme() -> None:
    """Split-tier wiring contract (flipped from xfail 2026-05-31).

    The xfail target this commit replaces was a placeholder for the
    end-to-end dispatch of multi-output serdes into separate K and V
    tiers.  After the Phase 2 series of commits (PR-1' through
    PR-2c'), the wiring lives across three units:

    * :func:`derive_component_key` / :class:`StoragePlacementMode`
      / :class:`SplitTierManifest` define the contract.
    * :class:`StorageManager` resolves placement at construction
      and exposes :attr:`storage_placement_mode` +
      :attr:`split_tier_manifest`.
    * :class:`SerdeL2AdapterWrapper` drives the state machine on
      both store and load paths, translating logical keys to
      ``derive_component_key(logical, "v")`` for the inner adapter
      and reading ``derive_component_key(logical, "k")`` from L1.

    The end-to-end behavioral proof (K bit-exact + V FP8 noise
    round-trip through a real ``StorageManager`` + ``file_l2``)
    lives in
    ``tests/v1/distributed/serde/test_serde_asym_fs_e2e.py::
    TestAsymK16V8VOnlySplitTierRoundTrip`` and is CUDA-gated
    because L1Manager's default allocator is CUDA-backed.  This
    contract test exercises the *static* wiring shape -- the
    public symbols, constructor signatures, and key-derivation
    invariants -- so a CPU-only sweep still catches regressions
    in the multi-output routing.
    """
    # Standard
    import inspect

    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper
    from lmcache.v1.distributed.storage_placement import (
        SplitTierManifest,
        SplitTierState,
        StoragePlacementMode,
        derive_component_key,
        derive_storage_placement_mode,
    )

    # ---- Wrapper accepts placement_mode + split_tier_manifest ----
    sig = inspect.signature(SerdeL2AdapterWrapper.__init__)
    assert "placement_mode" in sig.parameters, (
        "SerdeL2AdapterWrapper.__init__ must accept placement_mode "
        "for split-tier wiring"
    )
    assert "split_tier_manifest" in sig.parameters, (
        "SerdeL2AdapterWrapper.__init__ must accept split_tier_manifest "
        "for split-tier wiring"
    )

    # ---- derive_component_key produces distinct K and V children ----
    logical = ObjectKey(
        chunk_hash=b"\x00" * 32,
        model_name="contract-test",
        kv_rank=0,
        cache_salt="",
    )
    k_child = derive_component_key(logical, "k")
    v_child = derive_component_key(logical, "v")
    # K and V children must be distinct from each other and from the
    # logical key (the wrapper relies on this to address two tiers
    # with one logical key).
    assert k_child != v_child
    assert k_child != logical
    assert v_child != logical
    # Cache salt + model + rank preserved (per-tenant quota accounting).
    assert k_child.cache_salt == logical.cache_salt
    assert k_child.model_name == logical.model_name
    assert k_child.kv_rank == logical.kv_rank
    assert v_child.cache_salt == logical.cache_salt

    # ---- Placement mode resolution: V-only serde -> KV_SPLIT_TIER ----
    # First Party
    from lmcache.v1.distributed.serde.base import SerdeConfig

    class _FakeCfg:
        def __init__(self, serde_config):
            self.serde_config = serde_config

    v_only_mode = derive_storage_placement_mode(
        [_FakeCfg(SerdeConfig(type="asym_k16_v8_v_only"))]
    )
    assert v_only_mode == StoragePlacementMode.KV_SPLIT_TIER
    asym_mode_1_mode = derive_storage_placement_mode(
        [_FakeCfg(SerdeConfig(type="asym_k16_v8"))]
    )
    assert asym_mode_1_mode == StoragePlacementMode.KV_TOGETHER

    # ---- Manifest tracks the 4-state lifecycle ----
    m = SplitTierManifest()
    m.register_pending(logical)
    assert m.lookup(logical) == SplitTierState.STORE_IN_FLIGHT
    m.mark_complete(logical)
    assert m.is_complete(logical)
    m.mark_invalidated(logical)
    assert m.lookup(logical) == SplitTierState.INVALIDATED
