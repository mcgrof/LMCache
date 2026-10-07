# SPDX-License-Identifier: Apache-2.0
"""Fused K/V engine layout into a KV_2LTD offload buffer.

vLLM's blocks-first attention backends pack K and V into the trailing axis of
one per-layer tensor (``CS == 2 * HS``), while a non-MLA offload buffer keeps
them apart as ``[2, L, T, NH*HS]``.  The buffer's per-token width is then half
the engine's, so the transfer must walk the engine at its own width and split
each head's content: K from the first half, V from the second.

The existing round-trip test uses a single fused plane, which cannot catch a
wrong per-token stride (the same wrong stride on the way out and back in
still round-trips).  Here the reference is a pure-torch gather, so a wrong
stride, a dropped half, or a K/V mix-up all fail.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format
from lmcache.v1.gpu_connector.utils import get_kernel_head_size
import lmcache.lmcache_native as lmcache_native

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NB, NL, NH, BS, HS = 8, 3, 2, 4, 8
CS = 2 * HS  # the engine packs K and V per head
SPT = NH * HS  # per-token width of one offload plane


def make_pool(order: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Blocks-first pool: every block holds all layers, so the per-block step
    is larger than one layer's tight step."""
    inner = (NH, BS, CS) if order == "BLHNC" else (BS, NH, CS)
    buf = torch.arange(
        NB * NL * NH * BS * CS, dtype=torch.float32, device="cuda"
    ).reshape(NB, NL, *inner)
    return buf, [buf[:, layer] for layer in range(NL)]


def torch_reference(buf: torch.Tensor, order: str, slots: torch.Tensor) -> torch.Tensor:
    """[2, NL, T, NH*HS]: K plane then V plane, heads flattened."""
    blocks, offsets = slots // BS, slots % BS
    rows = []
    for b, o in zip(blocks.tolist(), offsets.tolist(), strict=True):
        if order == "BLHNC":  # buf [NB, NL, NH, BS, CS]
            rows.append(buf[b, :, :, o])  # [NL, NH, CS]
        else:  # buf [NB, NL, BS, NH, CS]
            rows.append(buf[b, :, o])  # [NL, NH, CS]
    raw = torch.stack(rows, dim=1)  # [NL, T, NH, CS]
    k = raw[..., :HS].reshape(NL, len(slots), SPT)
    v = raw[..., HS:].reshape(NL, len(slots), SPT)
    return torch.stack([k, v], dim=0)


@cuda_only
@pytest.mark.parametrize("order", ["BLHNC", "BLNHC"])
def test_fused_engine_into_split_buffer(order):
    buf, views = make_pool(order)
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": order})
    expected = (
        lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS
        if order == "BLHNC"
        else lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS
    )
    assert fmt == expected
    head_size = get_kernel_head_size(kv, fmt)
    assert head_size == HS, "the kernel wants the real per-head width"

    ptrs = torch.tensor([v.data_ptr() for v in kv], dtype=torch.int64, device="cuda")
    block_stride = kv[0].stride(0)
    slots = torch.tensor([1, 2, 3, 9, 10, 20, 21, 31], device="cuda")
    staging = torch.zeros(2, NL, len(slots), SPT, dtype=torch.float32, device="cuda")

    device_ops.multi_layer_kv_transfer(
        staging,
        ptrs,
        slots,
        torch.device("cuda:0"),
        NB * BS,
        int(lmcache_native.TransferDirection.D2H),
        int(fmt),
        BS,
        head_size,
        0,
        block_stride,
    )
    torch.cuda.synchronize()

    ref = torch_reference(buf, order, slots.cpu())
    assert torch.equal(staging.cpu(), ref.cpu()), "captured KV is not the engine's KV"
    assert (staging[1] != 0).any(), "the V plane was never written"

    # Write it back into a zeroed pool and expect the original bytes.
    original = buf.clone()
    for b in {int(s) // BS for s in slots.cpu()}:
        buf[b].zero_()
    device_ops.multi_layer_kv_transfer(
        staging,
        ptrs,
        slots,
        torch.device("cuda:0"),
        NB * BS,
        int(lmcache_native.TransferDirection.H2D),
        int(fmt),
        BS,
        head_size,
        0,
        block_stride,
    )
    torch.cuda.synchronize()
    for s in slots.cpu().tolist():
        b, o = s // BS, s % BS
        if order == "BLHNC":
            assert torch.equal(buf[b, :, :, o], original[b, :, :, o])
        else:
            assert torch.equal(buf[b, :, o], original[b, :, o])
