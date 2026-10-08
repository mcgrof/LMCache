# SPDX-License-Identifier: Apache-2.0
"""Native skip-prefix boundaries, independent of storage or staging allocators."""

# Standard
from types import ModuleType
import os

# Third Party
import pytest
import torch

# First Party
import lmcache.lmcache_native as native


@pytest.fixture
def transfer_case() -> tuple[
    ModuleType, torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor
]:
    """Prepare independently initialized native packed and paged CUDA tensors."""
    if os.environ.get("LMCACHE_TEST_CUDA_STAGING_ORDERING") != "1":
        pytest.skip("set LMCACHE_TEST_CUDA_STAGING_ORDERING=1")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("requires a CUDA native build")
    cuda = pytest.importorskip("lmcache.cuda_ops")
    packed = torch.full((2, 2, 8, 64), 97, dtype=torch.float16, device="cuda:0")
    paged = [
        torch.full((2, 8, 16, 1, 64), 23, dtype=torch.float16, device="cuda:0")
        for _ in range(2)
    ]
    pointers = torch.tensor(
        [layer.data_ptr() for layer in paged], dtype=torch.int64, device="cuda:0"
    )
    slots = torch.arange(37, 45, dtype=torch.int64, device="cuda:0")
    return cuda, packed, paged, pointers, slots


@pytest.mark.parametrize("resident", [0, 3, 8])
@pytest.mark.parametrize(
    "direction", [native.TransferDirection.H2D, native.TransferDirection.D2H]
)
def test_native_skip_prefix(
    transfer_case: tuple[
        ModuleType, torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor
    ],
    resident: int,
    direction: native.TransferDirection,
) -> None:
    """Keep the skipped prefix and unselected slots exact in both directions."""
    cuda, packed, paged, pointers, slots = transfer_case
    expected_packed = packed.clone()
    expected_paged = [layer.clone() for layer in paged]
    if direction == native.TransferDirection.H2D:
        for layer in expected_paged:
            for slot in range(37 + resident, 45):
                layer[:, slot // 16, slot % 16].fill_(97)
    else:
        expected_packed[:, :, resident:].fill_(23)
    cuda.multi_layer_kv_transfer(
        packed,
        pointers,
        slots,
        torch.device("cuda:0"),
        128,
        direction,
        native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        block_size=16,
        head_size=64,
        skip_prefix_n_tokens=resident,
        block_stride_elems=0,
    )
    torch.cuda.synchronize()
    assert torch.equal(packed, expected_packed)
    assert all(
        torch.equal(observed, expected)
        for observed, expected in zip(paged, expected_paged, strict=True)
    )


@pytest.mark.parametrize("resident", [-1, 9])
def test_native_rejects_out_of_range_prefix(
    transfer_case: tuple[
        ModuleType, torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor
    ],
    resident: int,
) -> None:
    """Reject invalid prefixes before constructing a CUDA grid."""
    cuda, packed, paged, pointers, slots = transfer_case
    with pytest.raises(RuntimeError, match="skip_prefix_n_tokens must be between"):
        cuda.multi_layer_kv_transfer(
            packed,
            pointers,
            slots,
            torch.device("cuda:0"),
            128,
            native.TransferDirection.H2D,
            native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            block_size=16,
            head_size=64,
            skip_prefix_n_tokens=resident,
            block_stride_elems=0,
        )
    torch.cuda.synchronize()
    assert all(bool(torch.all(layer == 23)) for layer in paged)
