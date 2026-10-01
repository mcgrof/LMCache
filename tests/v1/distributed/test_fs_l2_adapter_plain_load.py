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
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_SHAPE = torch.Size([2, 4, 8])


def _make_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="plain-fs-test",
        kv_rank=0,
    )


def _make_kv_obj(
    fill_value: float,
    groups: int,
    dtype: torch.dtype = torch.bfloat16,
    fmt: MemoryFormat = MemoryFormat.KV_2LTD,
) -> MemoryObj:
    """A fixed-layout object for the plain-path destination shape."""
    allocator = AdHocMemoryAllocator(device="cpu")
    obj = allocator.allocate([_SHAPE] * groups, [dtype] * groups, fmt=fmt)
    assert obj is not None
    for g in range(groups):
        tensor = obj.get_tensor(g)
        assert tensor is not None
        tensor.fill_(fill_value)
    return obj


def _make_byte_buffer_obj(fill_value: int) -> TensorMemoryObj:
    """Build a variable-length-capable buffer with real capacity metadata."""
    raw = torch.full((_SHAPE.numel(),), fill_value, dtype=torch.uint8)
    shape = raw.shape
    return TensorMemoryObj(
        raw_data=raw,
        metadata=MemoryObjMetadata(
            shape=shape,
            dtype=torch.uint8,
            address=raw.data_ptr(),
            phy_size=raw.numel(),
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.BINARY_BUFFER,
            shapes=[shape],
            dtypes=[torch.uint8],
        ),
        parent_allocator=None,
    )


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


@pytest.mark.parametrize("read_ahead_size", [None, 17])
def test_truncated_fp8_fixed_layout_is_a_miss(tmp_path, read_ahead_size) -> None:
    """A short FP8 file must not be accepted merely because FP8 is 1 byte.

    Both the ordinary and read-ahead paths fail closed before modifying the
    destination layout or reporting a hit.
    """
    adapter = FSL2Adapter(
        FSL2AdapterConfig(
            base_path=str(tmp_path),
            read_ahead_size=read_ahead_size,
        )
    )
    try:
        key = _make_key(3)
        src = _make_kv_obj(1.0, groups=1, dtype=torch.float8_e4m3fn)
        store_task = adapter.submit_store_task([key], [src])
        assert _wait_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[store_task].is_successful()

        stored = list(tmp_path.rglob("*.data"))
        assert len(stored) == 1
        blob = stored[0].read_bytes()
        stored[0].write_bytes(blob[:-1])

        dst = _make_kv_obj(0.0, groups=1, dtype=torch.float8_e4m3fn)
        original_shape = dst.get_shape()
        original_size = dst.get_size()
        load_task = adapter.submit_load_task([key], [dst])
        assert _wait_fd(adapter.get_load_event_fd())
        bitmap = adapter.query_load_result(load_task)
        assert bitmap is not None and not bitmap.test(0)
        assert dst.get_shape() == original_shape
        assert dst.get_size() == original_size
        assert dst.get_dtype() == torch.float8_e4m3fn
    finally:
        adapter.close()


def test_short_binary_buffer_load_narrows_and_hits(tmp_path) -> None:
    """Explicit serde byte buffers retain the variable-length contract."""
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        key = _make_key(4)
        src = _make_byte_buffer_obj(1)
        store_task = adapter.submit_store_task([key], [src])
        assert _wait_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[store_task].is_successful()

        stored = list(tmp_path.rglob("*.data"))
        assert len(stored) == 1
        used = src.get_size() - 7
        stored[0].write_bytes(stored[0].read_bytes()[:used])

        dst = _make_byte_buffer_obj(0)
        load_task = adapter.submit_load_task([key], [dst])
        assert _wait_fd(adapter.get_load_event_fd())
        bitmap = adapter.query_load_result(load_task)
        assert bitmap is not None and bitmap.test(0)
        assert dst.get_size() == used
        assert len(dst.byte_array) == used
    finally:
        adapter.close()


def test_unaligned_fs_round_trip_falls_back_from_odirect(tmp_path) -> None:
    """An unaligned object buffered on store must also use buffered load.

    Serialized buffers commonly have a short versioned header and are not
    block-size aligned.  Attempting O_DIRECT on such a file during load raises
    ``EINVAL`` and turns a durable object into a false cache miss.
    """
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path), use_odirect=True))
    try:
        key = _make_key(2)
        src = _make_kv_obj(fill_value=2.5, groups=1)
        assert src.get_size() == 128  # smaller than an O_DIRECT sector

        store_task = adapter.submit_store_task([key], [src])
        assert _wait_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[store_task].is_successful()

        dst = _make_kv_obj(fill_value=0.0, groups=1)
        load_task = adapter.submit_load_task([key], [dst])
        assert _wait_fd(adapter.get_load_event_fd())
        bitmap = adapter.query_load_result(load_task)
        assert bitmap is not None and bitmap.test(0)
        assert torch.equal(dst.get_tensor(0), src.get_tensor(0))
    finally:
        adapter.close()
