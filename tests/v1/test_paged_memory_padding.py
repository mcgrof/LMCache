# SPDX-License-Identifier: Apache-2.0
"""Public allocation contracts for logical payloads and physical I/O padding."""

# Standard
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat


@pytest.mark.parametrize("batched", [False, True])
def test_partial_page_zeros_only_owned_tail_and_rebinds_on_reuse(batched: bool) -> None:
    """A 1,280-byte payload cannot expose the preceding owner's 2,816-byte tail."""
    full_shape = torch.Size([2, 1, 16, 64])
    partial_shape = torch.Size([2, 1, 5, 64])
    backing = torch.full((8192,), 0xD3, dtype=torch.uint8)
    allocator = PagedTensorMemoryAllocator(
        backing, [full_shape], [torch.float16], MemoryFormat.KV_2LTD
    )
    guard = allocator.allocate(full_shape, torch.float16)
    assert guard is not None
    if batched:
        allocated = allocator.batched_allocate(partial_shape, torch.float16, 1)
        assert allocated is not None
        partial = allocated[0]
    else:
        partial = allocator.allocate(partial_shape, torch.float16)
    assert partial is not None
    assert partial.get_size() == 1280
    assert partial.get_physical_size() == 4096
    logical = partial.raw_tensor
    physical = partial.physical_tensor
    assert logical is not None and physical is not None
    assert logical.nbytes == 1280 and physical.nbytes == 4096
    logical.fill_(0x61)
    partial.zero_padding()
    assert bytes(physical.tolist()) == bytes([0x61]) * 1280 + bytes(2816)
    assert bytes(backing[:4096].tolist()) == bytes([0xD3]) * 4096

    # A different payload length in the next allocation must not retain the
    # previous group's offsets or a serialization-specific used-size override.
    partial.set_used_size(17)
    partial.ref_count_down()
    reused = allocator.allocate(full_shape, torch.float16)
    assert reused is not None
    assert reused.get_size() == 4096
    tensor = reused.tensor
    assert tensor is not None and tensor.shape == full_shape
    reused.ref_count_down()
    guard.ref_count_down()
    assert allocator.memcheck()


def test_partial_group_offsets_follow_the_rebound_layout() -> None:
    """Each group is exposed at its new logical offset after a partial allocation."""
    backing = torch.full((4096,), 0xD3, dtype=torch.uint8)
    allocator = PagedTensorMemoryAllocator(
        backing,
        [torch.Size([512]), torch.Size([1024])],
        [torch.float32, torch.float16],
    )
    obj = allocator.allocate(
        [torch.Size([3]), torch.Size([5])], [torch.float32, torch.float16]
    )
    assert obj is not None and obj.get_size() == 22
    first, second = obj.get_tensor(0), obj.get_tensor(1)
    assert first is not None and second is not None
    first.fill_(3.0)
    second.fill_(5.0)
    assert first.data_ptr() == obj.data_ptr
    assert second.data_ptr() == obj.data_ptr + 12
    obj.zero_padding()
    assert torch.equal(backing[22:], torch.zeros(4074, dtype=torch.uint8))
    assert first.tolist() == [3.0] * 3 and second.tolist() == [5.0] * 5
    obj.ref_count_down()


@pytest.mark.parametrize("batched", [False, True])
def test_oversized_layout_does_not_consume_a_pool_page(batched: bool) -> None:
    """An invalid request leaves the sole free page available for a valid request."""
    allocator = PagedTensorMemoryAllocator(
        torch.empty(4096, dtype=torch.uint8), [torch.Size([4096])], [torch.uint8]
    )
    with pytest.raises(ValueError, match="exceeds"):
        if batched:
            allocator.batched_allocate(torch.Size([4097]), torch.uint8, 1)
        else:
            allocator.allocate(torch.Size([4097]), torch.uint8)
    valid = allocator.allocate(torch.Size([4096]), torch.uint8)
    assert valid is not None
    valid.ref_count_down()


@pytest.mark.parametrize("page_bytes", [4096, 65536])
@pytest.mark.parametrize("slot_bytes", [4096, 12288])
def test_gpu_pool_alignment_keeps_host_pages_distinct_from_slots(
    monkeypatch: pytest.MonkeyPatch, page_bytes: int, slot_bytes: int
) -> None:
    """The pool can contain complete I/O slots on either host-page geometry."""
    sysconf = os.sysconf
    monkeypatch.setattr(
        os,
        "sysconf",
        lambda name: page_bytes if name == "SC_PAGE_SIZE" else sysconf(name),
    )
    allocator = GPUMemoryAllocator(
        65537,
        device="cpu",
        use_paging=True,
        shapes=[torch.Size([slot_bytes])],
        dtypes=[torch.uint8],
        fmt=MemoryFormat.BINARY_BUFFER,
    )
    try:
        pages = allocator.get_paged_buffers()
        assert pages is not None
        assert pages[0].data_ptr() % page_bytes == 0
        assert sum(page.nbytes for page in pages) % page_bytes == 0
        assert all(page.nbytes == slot_bytes for page in pages)
        assert sum(page.nbytes for page in pages) >= 65537
    finally:
        allocator.close()
