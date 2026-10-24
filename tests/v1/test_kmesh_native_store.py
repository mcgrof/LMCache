# SPDX-License-Identifier: Apache-2.0
"""Check canonical host padding after a completed packed-tail transfer."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.kmesh_native_store import RegisteredSpanGather


@pytest.mark.parametrize("valid", [1, 3, 7])
def test_drained_packed_tail_preserves_rows_and_zeroes_padding(valid: int) -> None:
    canonical = torch.full((2, 24, 8, 32), -9, dtype=torch.bfloat16)
    packed = canonical.view(-1)[: 2 * 24 * valid * 32].view(2, 24, valid, 32)
    expected = (torch.arange(packed.numel()).reshape(packed.shape) % 53).to(
        torch.bfloat16
    )
    packed.copy_(expected)
    gathered = RegisteredSpanGather([], ((packed, canonical),))
    gathered.canonicalize()
    assert torch.equal(
        canonical[:, :, :valid].view(torch.uint16), expected.view(torch.uint16)
    )
    assert torch.count_nonzero(canonical[:, :, valid:]).item() == 0
    with pytest.raises(RuntimeError, match="already"):
        gathered.canonicalize()
    assert torch.equal(
        canonical[:, :, :valid].view(torch.uint16), expected.view(torch.uint16)
    )
