# SPDX-License-Identifier: Apache-2.0
"""Raw-block I/O straight out of device-resident memory objects.

A memory object whose storage is not on the CPU has no host byte view; the
core must hand the engine the object's flat physical tensor and the
logical and aligned lengths without ever calling byte_array.  Meta tensors
stand in for VRAM slots here: they have a device, a data pointer and a
size, and no host bytes at all.
"""

from __future__ import annotations

# Standard
import importlib.util
import sys

# Third Party
import pytest
import torch

# First Party
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
        with pytest.raises(RuntimeError, match="device slot"):
            core._prepare_write_payload(cramped)
    finally:
        core.close()
