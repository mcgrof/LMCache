# SPDX-License-Identifier: Apache-2.0
"""Exercise native packed-KV layer copies against independent token indexing."""

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires native CUDA layer transfers",
)


@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_packed_layer_round_trip(hnd, token_major, padded, dtype):
    # First Party
    from lmcache import cuda_ops

    blocks, block_size, heads, head_size, tokens = 2, 6, 2, 8, 4
    shape = (
        (blocks, heads, block_size, 2 * head_size)
        if hnd
        else (blocks, block_size, heads, 2 * head_size)
    )
    inner = heads * block_size * 2 * head_size
    block_stride = inner + (64 if padded else 0)
    backing = torch.full((blocks * block_stride,), -1000, device="cuda", dtype=dtype)
    strides = (block_stride, shape[2] * shape[3], shape[3], 1)
    cache = backing.as_strided(shape, strides)
    expected_backing = backing.clone()
    expected_cache = expected_backing.as_strided(shape, strides)
    source = torch.arange(tokens * 2 * heads * head_size, device="cuda").to(dtype)
    source = source.reshape(tokens, 2, heads * head_size)
    lmc = source.clone() if token_major else source.permute(1, 0, 2).contiguous()
    lmc_before = lmc.clone()
    slots = [0, 5, -1, 11]
    slot_mapping = torch.tensor(slots, device="cuda", dtype=torch.int64)
    for index, slot in enumerate(slots):
        if slot < 0:
            continue
        block, offset = divmod(slot, block_size)
        packed = torch.cat(
            (
                source[index, 0].reshape(heads, head_size),
                source[index, 1].reshape(heads, head_size),
            ),
            dim=-1,
        )
        if hnd:
            expected_cache[block, :, offset, :] = packed
        else:
            expected_cache[block, offset, :, :] = packed
    fmt = EngineKVFormat.NL_X_NB_NH_BS_CS if hnd else EngineKVFormat.NL_X_NB_BS_NH_CS

    cuda_ops.single_layer_kv_transfer(
        lmc, cache, slot_mapping, TransferDirection.H2D, fmt, token_major
    )
    torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)
    torch.testing.assert_close(lmc, lmc_before, rtol=0, atol=0)

    restored = torch.full_like(lmc, -2000)
    expected = torch.full_like(source, -2000)
    for index, slot in enumerate(slots):
        if slot >= 0:
            expected[index] = source[index]
    if not token_major:
        expected = expected.permute(1, 0, 2).contiguous()
    cuda_ops.single_layer_kv_transfer(
        restored, cache, slot_mapping, TransferDirection.D2H, fmt, token_major
    )
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)


def test_packed_layer_rejects_invalid_geometry_before_copy():
    # First Party
    from lmcache import cuda_ops

    slots = torch.tensor([0], device="cuda", dtype=torch.int64)
    lmc = torch.zeros((2, 1, 16), device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros((1, 6, 2, 15), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="positive and even"):
        cuda_ops.single_layer_kv_transfer(
            lmc,
            cache,
            slots,
            TransferDirection.D2H,
            EngineKVFormat.NL_X_NB_BS_NH_CS,
            False,
        )


@pytest.mark.parametrize("case", ["unaligned", "empty_heads", "large_elements"])
def test_packed_layer_rejects_unsafe_vector_copy(case):
    # First Party
    from lmcache import cuda_ops

    slots = torch.tensor([0], device="cuda", dtype=torch.int64)
    lmc = torch.zeros((2, 1, 16), device="cuda", dtype=torch.bfloat16)
    if case == "unaligned":
        backing = torch.zeros(1 + 6 * 2 * 16, device="cuda", dtype=torch.bfloat16)
        cache = backing[1:].reshape(1, 6, 2, 16)
        message = "eight-byte aligned"
    elif case == "empty_heads":
        cache = torch.zeros((1, 6, 0, 16), device="cuda", dtype=torch.bfloat16)
        message = "packed KV"
    else:
        cache = torch.zeros((1, 6, 2, 16), device="cuda", dtype=torch.complex128)
        message = "elements must fit"
    with pytest.raises(RuntimeError, match=message):
        cuda_ops.single_layer_kv_transfer(
            lmc,
            cache,
            slots,
            TransferDirection.D2H,
            EngineKVFormat.NL_X_NB_BS_NH_CS,
            False,
        )
