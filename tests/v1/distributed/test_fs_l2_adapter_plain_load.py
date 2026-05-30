# SPDX-License-Identifier: Apache-2.0
"""
Regression test: the plain (no-serde) FS L2 adapter load path.

The FS adapter reports the number of bytes it read into the destination
via ``MemoryObj.set_used_size``. For serde temp buffers (flat uint8,
sized from an upper-bound estimate) that call narrows the logical view
to the bytes actually on disk. For plain no-serde loads the destination
is a fixed-layout KV object (bf16, possibly multi-group) read at full
size -- the report must pass through as a no-op instead of tripping the
narrowing-only validation, which failed every vanilla FS load with
``ValueError`` and surfaced as a spurious cache miss.
"""

# Standard
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.platform import consume_fd

_SHAPE = torch.Size([2, 4, 8])


def _make_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="plain-fs-test",
        kv_rank=0,
    )


def _make_kv_obj(fill_value: float, groups: int) -> MemoryObj:
    """A bf16 KV-layout object -- the plain-path destination shape."""
    allocator = AdHocMemoryAllocator(device="cpu")
    obj = allocator.allocate(
        [_SHAPE] * groups, [torch.bfloat16] * groups, fmt=MemoryFormat.KV_2LTD
    )
    assert obj is not None
    for g in range(groups):
        tensor = obj.get_tensor(g)
        assert tensor is not None
        tensor.fill_(fill_value)
    return obj


def _wait_fd(event_fd: int, timeout: float = 10.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if not events:
        return False
    consume_fd(event_fd)
    return True


@pytest.mark.parametrize("groups", [1, 2])
def test_plain_fs_store_load_round_trip(tmp_path, groups: int) -> None:
    """Store a bf16 KV object through the RAW FS adapter (no serde
    wrapper) and load it back into a fresh object of the same layout.

    The load must report a hit (bitmap bit set) and return the stored
    bytes verbatim; a regression in the used-size report turns this
    into a miss for every plain FS deployment."""
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        key = _make_key(1)
        src = _make_kv_obj(fill_value=1.5, groups=groups)

        store_task = adapter.submit_store_task([key], [src])
        assert _wait_fd(adapter.get_store_event_fd())
        completed = adapter.pop_completed_store_tasks()
        assert completed[store_task].is_successful()

        dst = _make_kv_obj(fill_value=0.0, groups=groups)
        load_task = adapter.submit_load_task([key], [dst])
        assert _wait_fd(adapter.get_load_event_fd())
        bitmap = adapter.query_load_result(load_task)
        assert bitmap is not None
        assert bitmap.test(0), (
            "plain (no-serde) FS load reported a miss for a key that "
            "was just stored -- the full-size used-size report must "
            "not be rejected"
        )

        for g in range(groups):
            src_t = src.get_tensor(g)
            dst_t = dst.get_tensor(g)
            assert src_t is not None and dst_t is not None
            assert torch.equal(dst_t, src_t)
        # The fixed layout is untouched: full-size report is a no-op.
        assert dst.get_size() == src.get_size()
    finally:
        adapter.close()
