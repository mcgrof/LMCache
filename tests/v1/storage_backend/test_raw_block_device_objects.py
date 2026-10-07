# SPDX-License-Identifier: Apache-2.0
"""Raw-block I/O straight out of device-resident memory objects.

A memory object whose storage is not on the CPU has no host byte view; the
core must hand the engine the object's flat physical tensor and the
logical and aligned lengths without ever calling byte_array.  Meta tensors
stand in for VRAM slots here: they have a device, a data pointer and a
size, and no host bytes at all.
"""

# Future
from __future__ import annotations

# Standard
import importlib.util
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.raw_block import RawBlockCore
from lmcache.v1.storage_backend.raw_block.core import (
    _device_payload_tensor,
    _logical_payload_len,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    make_memory_obj,
    make_raw_block_core_config,
    make_raw_block_file,
)

requires_rust_raw_block_io = pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io extension is not installed",
)


def _meta_device_obj(nbytes: int, slot_bytes: int) -> TensorMemoryObj:
    """A memory object whose physical slot is a meta-device tensor: it has
    no host bytes at all, like a VRAM slot."""
    raw = torch.empty(slot_bytes, dtype=torch.uint8, device="meta")
    meta = MemoryObjMetadata(
        shape=torch.Size([nbytes]),
        dtype=torch.uint8,
        address=0,
        phy_size=slot_bytes,
        ref_count=1,
        fmt=MemoryFormat.BINARY,
    )
    return TensorMemoryObj(raw_data=raw, metadata=meta, parent_allocator=None)


def test_device_payload_tensor_only_for_device_objects():
    host = make_memory_obj(b"z" * 512)
    assert _device_payload_tensor(host) is None
    assert _logical_payload_len(host) == 512

    dev = _meta_device_obj(1000, 4096)
    flat = _device_payload_tensor(dev)
    assert flat is not None
    assert flat.device.type == "meta"
    assert flat.dtype == torch.uint8
    assert flat.nbytes == 4096
    assert _logical_payload_len(dev) == 1000


@pytest.mark.parametrize("batched", [False, True])
def test_paged_partial_device_object_keeps_its_full_physical_slot(batched: bool):
    """Logical narrowing must not truncate the registered O_DIRECT endpoint."""
    full_shape = torch.Size([2, 1, 16, 64])
    partial_shape = torch.Size([2, 1, 5, 64])
    allocator = PagedTensorMemoryAllocator(
        tensor=torch.empty(8192, dtype=torch.uint8, device="meta"),
        shapes=[full_shape],
        dtypes=[torch.float16],
        fmt=MemoryFormat.KV_2LTD,
    )
    if batched:
        objects = allocator.batched_allocate([partial_shape], [torch.float16], 2)
        assert objects is not None
    else:
        obj = allocator.allocate([partial_shape], [torch.float16])
        assert obj is not None
        objects = [obj]
    core = object.__new__(RawBlockCore)
    core.use_odirect = True
    core.use_uring_cmd = False
    core.block_align = 4096
    core.header_bytes = 4096
    core.slot_bytes = 8192
    try:
        for obj in objects:
            assert obj.raw_tensor.nbytes == 1280
            physical, logical, total = core._prepare_write_payload(obj)
            assert physical.nbytes == total == 4096
            assert logical == 1280
            assert physical.data_ptr() == obj.raw_tensor.data_ptr()
    finally:
        for obj in objects:
            obj.ref_count_down()
    assert allocator.memcheck()


@requires_rust_raw_block_io
@pytest.mark.skipif(sys.platform != "linux", reason="raw-block is Linux only")
def test_device_write_payload_uses_the_slot_and_aligned_length(tmp_path):
    path = make_raw_block_file(tmp_path)
    core = RawBlockCore(make_raw_block_core_config(path), key_namespace="object")
    try:
        # Flip alignment on after opening: the backing file may sit on tmpfs,
        # which refuses O_DIRECT opens, and only the payload preparation is
        # under test here.
        core.use_odirect = True
        obj = _meta_device_obj(1000, core.block_align * 2)
        buf, payload_len, total_len = core._prepare_write_payload(obj)
        assert buf.device.type == "meta"
        assert payload_len == 1000
        assert total_len == core.block_align

        # A slot too small for the O_DIRECT tail is refused rather than
        # bounced through the host.
        cramped = _meta_device_obj(1000, 1000)
        with pytest.raises(RuntimeError, match="GPU slot"):
            core._prepare_write_payload(cramped)
    finally:
        core.close()


@pytest.mark.parametrize("registration", ["no-export", "refused", "uring-cmd"])
def test_device_registration_never_falls_back_to_host_mapping(
    monkeypatch, registration
):
    # Standard
    from types import SimpleNamespace
    from unittest.mock import Mock

    core = object.__new__(RawBlockCore)
    core.io_engine = "io_uring"
    core.use_uring_cmd = registration == "uring-cmd"
    raw = Mock()
    if registration == "refused":
        raw.register_fixed_dmabufs.side_effect = OSError("unsupported kernel")
    monkeypatch.setattr(core, "_rawdev", lambda: raw)
    allocator = SimpleNamespace(
        get_paged_buffers=lambda: (torch.empty(4096, device="meta"),),
        get_paged_dmabuf_regions=lambda: (
            None if registration == "no-export" else [(7, 0)]
        ),
    )
    with pytest.raises(RuntimeError, match="GPU staging requires"):
        core.register_fixed_buffers_from_allocator(allocator)
    raw.register_fixed_buffers.assert_not_called()


@pytest.mark.parametrize("direction", ["read", "write"])
@pytest.mark.parametrize("cramped", [False, True])
def test_bounded_device_io_slices_the_device_slot_without_host_bounce(
    monkeypatch, direction, cramped
):
    # Standard
    from unittest.mock import Mock

    core = object.__new__(RawBlockCore)
    core.use_odirect = True
    core.use_uring_cmd = False
    core.block_align = 4096
    core.max_data_transfer_size = 4096
    raw = Mock()
    monkeypatch.setattr(core, "_rawdev", lambda: raw)
    monkeypatch.setattr(
        core,
        "_wait_iouring_results",
        lambda _raw, _batch, count, _label: [True] * count,
    )
    bounce = Mock(side_effect=AssertionError("must not allocate a host bounce"))
    monkeypatch.setattr(core, "_allocate_aligned_buffer", bounce)
    slot = torch.empty(6000 if cramped else 8192, dtype=torch.uint8, device="meta")
    if direction == "write":
        if cramped:
            with pytest.raises(ValueError, match="GPU slot"):
                core._write_bounded_io_uring_buffers([4096], [slot], [6000], [8192])
            raw.batched_write.assert_not_called()
        else:
            core._write_bounded_io_uring_buffers([4096], [slot], [6000], [8192])
    else:
        assert core._read_bounded_io_uring_buffers([4096], [slot], [6000], [8192]) == [
            not cramped
        ]
        if cramped:
            raw.batched_read.assert_not_called()
    if not cramped:
        method = raw.batched_read if direction == "read" else raw.batched_write
        offsets, buffers, lengths = method.call_args.args[:3]
        assert offsets == [4096, 8192]
        assert lengths == [4096, 4096]
        assert all(buf.device.type == "meta" and buf.nbytes == 4096 for buf in buffers)
    bounce.assert_not_called()
