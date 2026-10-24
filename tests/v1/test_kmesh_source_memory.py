# SPDX-License-Identifier: Apache-2.0
"""Refuse unpinned, foreign and incomplete canonical source reservations."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.multiprocess.kmesh_source_memory import require_registered_span_tensor


@pytest.mark.parametrize(
    "case",
    [
        "pageable",
        "cpu-context",
        "unspecified-device",
        "size",
        "dtype",
        "offset",
        "physical-size",
        "foreign-pointer",
        "unowned",
    ],
)
def test_source_requires_actual_pinned_owner_and_exact_extent(case: str) -> None:
    arena = torch.zeros(8192, dtype=torch.uint8)
    allocator = TensorMemoryAllocator(arena)
    obj = allocator.allocate(
        torch.Size((2, 2, 8, 16)), torch.float32 if case == "dtype" else torch.bfloat16
    )
    assert obj is not None
    address, physical, raw, parent = (
        obj.meta.address,
        obj.meta.phy_size,
        obj.raw_data,
        obj.parent_allocator,
    )
    device = torch.device("cuda", 0)
    if case == "cpu-context":
        device = torch.device("cpu")
    elif case == "unspecified-device":
        device = torch.device("cuda")
    elif case == "offset":
        obj.meta.address = arena.nbytes
    elif case == "physical-size":
        obj.meta.phy_size = 1
    elif case == "foreign-pointer":
        obj.raw_data = torch.zeros_like(obj.raw_data)
    elif case == "unowned":
        obj.parent_allocator = None
    try:
        with pytest.raises(ValueError, match="allocator-owned|pinned host"):
            require_registered_span_tensor(
                obj, device, obj.get_size() - (2 if case == "size" else 0)
            )
        assert torch.count_nonzero(arena).item() == 0
    finally:
        obj.meta.address, obj.meta.phy_size = address, physical
        obj.raw_data, obj.parent_allocator = raw, parent
        obj.ref_count_down()
    assert allocator.num_active_allocations == 0
