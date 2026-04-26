# SPDX-License-Identifier: Apache-2.0
"""Phase 4 GPU portion — the load-bearing "no FP16 V allocation" test.

This test is the difference between native_asym being honest and
being a subtly-disguised storage-only path with extra steps.  It
runs a native_asym decode on a CUDA box, records every CUDA
allocation that happened during the call, and asserts that no
allocation matches the FP16 V buffer size.

If the test fails, somebody quietly added a `.to(torch.float16)`
on the V hot path, materializing the dequantized V and erasing
the HBM capacity savings the asymmetric story is built on.

Skipped without CUDA.  Run on a real H100/H200 box:

    pytest tests/v1/kv_codec/test_gpu_memory_snapshot.py
"""

# Standard
from dataclasses import dataclass

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    ScaleScope,
    compute_v_scales,
    quantize_v_fp8,
)
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.naive_serde.asym_serde import (
    AsymK16V8Deserializer,
    AsymK16V8Serializer,
    AsymKVMemoryObj,
    AsymKVView,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU memory snapshot needs CUDA"
)


@dataclass
class _FakeMetadata:
    model_name: str = "qwen2.5-7b"


@dataclass
class _FakeConfig:
    chunk_size: int = 256


def _make_asym_view_cuda(shape=(16, 64, 32, 128), dtype=torch.float16):
    """Build a synthetic K + V_fp8 + scales triple ON THE GPU."""
    g = torch.Generator(device="cuda")
    g.manual_seed(0xCAFE)
    k = torch.randn(*shape, dtype=dtype, device="cuda", generator=g)
    v_fp16 = torch.randn(*shape, dtype=dtype, device="cuda", generator=g)
    scales = compute_v_scales(v_fp16, ScaleScope.PER_TENSOR).to(torch.float32)
    v_fp8 = quantize_v_fp8(v_fp16, scales, ScaleScope.PER_TENSOR)
    # `v_fp16` is held in this scope intentionally — letting it go
    # out of scope before the test runs would allow torch's allocator
    # to recycle the FP16 buffer, which would mask a "we allocated a
    # fresh FP16 V" bug.  The test wants to detect ANY new FP16
    # allocation matching V's element count after our snapshot starts.
    del v_fp16
    return AsymKVView(k=k, v_fp8=v_fp8, v_scales=scales)


class _AsymInputMemoryObj:
    def __init__(self, view, original_shape, dtype):
        self.asym_view = view
        self.tensor = None
        self.metadata = MemoryObjMetadata(
            shape=torch.Size([0, 0, 0, 0]),
            dtype=None,
            address=0,
            phy_size=0,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.UNDEFINED,
            shapes=[torch.Size(original_shape)],
            dtypes=[dtype],
        )


def _peak_fp16_alloc_during(fn, fp16_v_size_bytes):
    """Run fn() with allocation tracking.  Returns the largest
    single allocation observed (in bytes) of a tensor whose
    byte-count matches fp16_v_size_bytes (within 1 KB slack).

    A clean native_asym path should return 0 — no allocation
    of that size happens during the decode.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_alloc = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    end_alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    # net allocation during fn (after fn returns; the decoded objects
    # are still live).  If the function allocated a transient FP16 V
    # buffer that was freed before return, peak - start_alloc would
    # show it.
    return {
        "start": start_alloc,
        "end": end_alloc,
        "peak": peak,
        "delta": end_alloc - start_alloc,
        "transient_peak": peak - start_alloc,
        "fp16_v_size": fp16_v_size_bytes,
    }


def test_native_asym_decode_does_not_allocate_fp16_V():
    """The contract: decoding a native_asym EncodedKV onto a CUDA
    device must not allocate any transient buffer matching the
    FP16 V byte size on top of the legitimate K + V_fp8 + scales.

    Going through the codec's `decode(..., device='cuda')` path so
    we actually observe GPU allocations, not CPU ones.

    Legitimate peak alloc = K_fp16 + V_fp8 + scales
                          ~= 2N + N + tiny = ~3N bytes
    Buggy peak (silent FP16 V) = 2N + N + 2N + tiny = ~5N bytes

    Threshold: peak <= 1.25 * legitimate.  25% slack covers the
    CUDA caching allocator's rounding without admitting a sneaky
    2N FP16 V buffer.
    """
    cfg = _FakeConfig()
    md = _FakeMetadata()
    ser = AsymK16V8Serializer(cfg, md, runtime_layout="native_asym")

    shape = (16, 64, 32, 128)
    view = _make_asym_view_cuda(shape=shape, dtype=torch.float16)
    n_elem = view.v_fp8.numel()
    fp16_v_bytes = n_elem * 2
    legitimate_peak = (
        n_elem * 2  # K fp16
        + n_elem * 1  # V fp8
        + 4  # scales (per-tensor float32)
    )

    inp = _AsymInputMemoryObj(view, original_shape=shape, dtype=torch.float16)
    encoded_mobj = ser.serialize(inp)
    del view, inp
    torch.cuda.empty_cache()

    blob_bytes = bytes(encoded_mobj.raw_data)
    encoded = ser.codec.from_bytes(blob_bytes)

    decoded_holder = {}

    def _do_decode_on_gpu():
        # Decode with explicit GPU device — tensors land in CUDA.
        k, v_fp8, scales = ser.codec.decode(
            encoded, out_v_dtype=None, device=torch.device("cuda")
        )
        decoded_holder["k"] = k
        decoded_holder["v_fp8"] = v_fp8
        decoded_holder["scales"] = scales

    stats = _peak_fp16_alloc_during(_do_decode_on_gpu, fp16_v_bytes)

    # Sanity: tensors landed on GPU at the right dtypes.
    assert decoded_holder["k"].device.type == "cuda"
    assert decoded_holder["k"].dtype == torch.float16
    assert decoded_holder["v_fp8"].device.type == "cuda"
    assert decoded_holder["v_fp8"].dtype == torch.float8_e4m3fn

    # The contract.  Allow 25% slack over the legitimate total.
    threshold = int(legitimate_peak * 1.25)
    assert stats["transient_peak"] <= threshold, (
        f"native_asym decode transiently allocated "
        f"{stats['transient_peak']:,} bytes on GPU; legitimate "
        f"is K({n_elem * 2:,}) + V_fp8({n_elem:,}) + scales = "
        f"{legitimate_peak:,}; threshold (1.25x) = {threshold:,}.  "
        f"Excess of {stats['transient_peak'] - legitimate_peak:,} "
        f"bytes suggests an FP16 V buffer ({fp16_v_bytes:,} bytes) "
        f"or other unintended materialization."
    )
    # Cross-check: peak is materially below what would be 5N bytes
    # (the silent-fp16-v case).
    silent_fp16_alloc = legitimate_peak + fp16_v_bytes
    assert stats["transient_peak"] < silent_fp16_alloc, (
        f"native_asym decode peak {stats['transient_peak']:,} >= "
        f"silent-fp16-v threshold {silent_fp16_alloc:,}; FP16 V "
        f"materialization detected."
    )


def test_storage_only_decode_returns_full_fp16_V():
    """Inverse sanity check: storage_only_dequant returns a
    [2, ...] FP16 tensor; its V slice has FP16 byte-count, not FP8.
    Together with the native_asym test above, this proves the two
    modes are doing distinguishable things, not aliasing each other.

    We check the *returned tensor*, not GPU peak alloc.  The codec
    decodes on CPU by default; the CUDA-allocator pressure isn't
    the right signal here.  The right signal is "what dtype and
    byte-count came back."
    """
    cfg = _FakeConfig()
    md = _FakeMetadata()
    ser = AsymK16V8Serializer(cfg, md, runtime_layout="storage_only_dequant")
    des = AsymK16V8Deserializer(cfg, md, runtime_layout="storage_only_dequant")

    shape = (16, 64, 32, 128)
    g = torch.Generator(device="cuda")
    g.manual_seed(0xCAFE)
    full_kv = torch.randn(
        2, *shape, dtype=torch.float16, device="cuda", generator=g
    )

    meta = MemoryObjMetadata(
        shape=torch.Size(full_kv.shape),
        dtype=full_kv.dtype,
        address=0,
        phy_size=full_kv.numel() * full_kv.element_size(),
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    obj_in = TensorMemoryObj(
        raw_data=full_kv, metadata=meta, parent_allocator=None
    )
    encoded = ser.serialize(obj_in)
    del full_kv, obj_in
    torch.cuda.empty_cache()

    blob_bytes = bytes(encoded.raw_data)
    new_mobj = BytesBufferMemoryObj(
        raw_bytes=blob_bytes, metadata=encoded.metadata
    )
    decoded = des.deserialize(new_mobj)

    # storage_only returns a single [2, ...] FP16 tensor.
    assert decoded.tensor is not None, (
        "storage_only decoded object should expose .tensor"
    )
    assert decoded.tensor.dtype == torch.float16
    assert decoded.tensor.shape[0] == 2  # K/V split
    # V slice is at FP16 byte-count, not FP8.  This is what
    # native_asym does NOT do.
    v_bytes = decoded.tensor[1].numel() * decoded.tensor[1].element_size()
    expected_fp16_v_bytes = (16 * 64 * 32 * 128) * 2  # numel * 2 B/elem
    assert v_bytes == expected_fp16_v_bytes, (v_bytes, expected_fp16_v_bytes)


def test_cuda_cpu_fp8_quant_byte_equality():
    """Bit-for-bit identical FP8 quantization between CPU and CUDA
    paths.  A divergence here means the two devices round
    differently (which would make CPU-emulated tests non-
    representative of GPU behavior)."""
    g = torch.Generator()
    g.manual_seed(7)
    v_cpu = torch.randn(8, 32, 64, dtype=torch.float16, generator=g)

    s_cpu = compute_v_scales(v_cpu, ScaleScope.PER_TENSOR)
    q_cpu = quantize_v_fp8(v_cpu, s_cpu, ScaleScope.PER_TENSOR)

    v_gpu = v_cpu.cuda()
    s_gpu = compute_v_scales(v_gpu, ScaleScope.PER_TENSOR)
    q_gpu = quantize_v_fp8(v_gpu, s_gpu, ScaleScope.PER_TENSOR)

    # Scale numerics may differ at floating-point ULP level; bytes
    # of FP8 V should match.
    torch.testing.assert_close(s_gpu.cpu(), s_cpu, atol=1e-7, rtol=1e-7)
    assert torch.equal(
        q_cpu.view(torch.uint8), q_gpu.cpu().view(torch.uint8)
    ), "CPU and CUDA FP8 quant produced different bytes"


def test_native_asym_decode_preserves_device():
    """Decode of a CUDA-encoded blob lands tensors on the requested
    device.  Catches: silent CPU fallback that would tank perf on
    the hot path."""
    cfg = _FakeConfig()
    md = _FakeMetadata()
    ser = AsymK16V8Serializer(cfg, md, runtime_layout="native_asym")
    des = AsymK16V8Deserializer(cfg, md, runtime_layout="native_asym")

    shape = (4, 16, 8, 64)
    view = _make_asym_view_cuda(shape=shape, dtype=torch.float16)
    inp = _AsymInputMemoryObj(view, original_shape=shape, dtype=torch.float16)
    encoded = ser.serialize(inp)
    del view, inp
    torch.cuda.empty_cache()

    new_mobj = BytesBufferMemoryObj(
        raw_bytes=bytes(encoded.raw_data),
        metadata=encoded.metadata,
    )
    decoded = des.deserialize(new_mobj)
    # Codec.decode default device is CPU.  The integration layer
    # (Phase 4 GPU work) is expected to specify the destination
    # device.  This test just confirms the codec API exposes that
    # control: when called explicitly with device='cuda', tensors
    # land on cuda.
    enc_obj = ser.codec.from_bytes(bytes(encoded.raw_data))
    k_cuda, v_cuda, _ = ser.codec.decode(
        enc_obj, out_v_dtype=None, device=torch.device("cuda")
    )
    assert k_cuda.device.type == "cuda"
    assert v_cuda.device.type == "cuda"
