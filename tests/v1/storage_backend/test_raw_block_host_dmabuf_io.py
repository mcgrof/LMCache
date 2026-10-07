# SPDX-License-Identifier: Apache-2.0
"""Opt-in real host DMA-BUF padding and bounded restore regressions.

Set LMCACHE_RUN_HOST_DMABUF_IO=1 and LMCACHE_HOST_DMABUF_TEST_DIR to a
qualified filesystem directory. Tests create only owned regular files.
Explicit opt-in requires a working udmabuf exporter, ordinary native extension
and the patched io_uring kernel; dependency failures are test failures.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import replace
from pathlib import Path
from typing import Iterator
import ctypes
import os
import tempfile

# Third Party
import pytest
import torch

if os.environ.get("LMCACHE_RUN_HOST_DMABUF_IO") != "1":
    pytest.skip(
        "real host DMA-BUF I/O requires explicit opt-in", allow_module_level=True
    )

# isort: split
# Third Party
from lmcache_rust_raw_block_io import RawBlockDevice  # noqa: E402

# First Party
from lmcache.v1.memory_allocators.mixed_memory_allocator import (  # noqa: E402
    MixedMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat  # noqa: E402
from lmcache.v1.storage_backend.raw_block import RawBlockCore  # noqa: E402
from lmcache.v1.storage_backend.raw_block.key_codec import (  # noqa: E402
    encode_object_key,
)
from tests.v1.storage_backend.raw_block_test_utils import (  # noqa: E402
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
)

PHYSICAL_BYTES = 4096
PAYLOAD_BYTES = 1233


@pytest.fixture
def directory() -> Iterator[Path]:
    """Create an exclusive test directory under the assigned filesystem."""
    setting = os.environ.get("LMCACHE_HOST_DMABUF_TEST_DIR", "")
    if not setting:
        raise ValueError("set LMCACHE_HOST_DMABUF_TEST_DIR to a qualified directory")
    root = Path(setting).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("host DMA-BUF tests require a directory, never a device")
    if RawBlockDevice.has_fault_injection():
        pytest.fail("real host DMA-BUF tests require the ordinary native build")
    with tempfile.TemporaryDirectory(prefix="lmcache-host-dmabuf-", dir=root) as path:
        yield Path(path)


def _allocator(pages: int) -> MixedMemoryAllocator:
    """Allocate actual udmabuf pages through the public mixed allocator."""
    return MixedMemoryAllocator(
        pages * PHYSICAL_BYTES,
        use_paging=True,
        dmabuf="udmabuf",
        shapes=[torch.Size([PHYSICAL_BYTES])],
        dtypes=[torch.uint8],
        fmt=MemoryFormat.KV_2LTD,
    )


def _view(ptr: int, size: int) -> memoryview:
    """Expose only the caller-specified CPU capacity."""
    return memoryview((ctypes.c_ubyte * size).from_address(ptr)).cast("B")


@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("registration", ["none", "host_fixed", "dmabuf"])
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("capacity", [PAYLOAD_BYTES, PHYSICAL_BYTES])
def test_imported_host_writes_preserve_source_and_zero_media_padding(
    directory: Path, registration: str, synchronous: bool, capacity: int
) -> None:
    """Host import must preserve the ordinary zero-padding contract."""
    allocator = _allocator(1)
    buffers = allocator.get_paged_buffers()
    regions = allocator.get_paged_dmabuf_regions()
    assert buffers is not None and regions is not None
    ptr = buffers[0].data_ptr()
    ctypes.memset(ptr, ord("P"), PAYLOAD_BYTES)
    ctypes.memset(ptr + PAYLOAD_BYTES, ord("S"), PHYSICAL_BYTES - PAYLOAD_BYTES)
    path = directory / "media.bin"
    with path.open("xb") as handle:
        handle.truncate(65536)
    device = RawBlockDevice(
        str(path),
        writable=True,
        use_iouring=True,
        use_odirect=True,
        alignment=PHYSICAL_BYTES,
        iouring_queue_depth=8,
    )
    try:
        if registration == "dmabuf":
            device.register_fixed_dmabufs(
                [ptr], [PHYSICAL_BYTES], [regions[0][0]], [regions[0][1]]
            )
        elif registration == "host_fixed":
            device.register_fixed_buffers([ptr], [PHYSICAL_BYTES])
        view = _view(ptr, capacity)
        if synchronous:
            device.write_uring(PHYSICAL_BYTES, view, PAYLOAD_BYTES, PHYSICAL_BYTES)
        else:
            batch = device.batched_write(
                [PHYSICAL_BYTES], [view], [PHYSICAL_BYTES], [None], [PAYLOAD_BYTES]
            )
            assert device.wait_iouring(batch) == ([True], [])
        del view
        with path.open("rb") as handle:
            handle.seek(PHYSICAL_BYTES)
            assert handle.read(PHYSICAL_BYTES) == (
                b"P" * PAYLOAD_BYTES + bytes(PHYSICAL_BYTES - PAYLOAD_BYTES)
            )
        assert ctypes.string_at(ptr, PHYSICAL_BYTES) == (
            b"P" * PAYLOAD_BYTES + b"S" * (PHYSICAL_BYTES - PAYLOAD_BYTES)
        )
        journal, dropped = device.take_io_journal()
        assert dropped == 0 and journal
        assert all(row["path"] == "bounce" for row in journal)
    finally:
        device.close()
        allocator.close()


@pytest.mark.no_shared_allocator
def test_core_imported_partial_host_store_and_restore(directory: Path) -> None:
    """Actual Core construction and public I/O accept logical-short host views."""
    allocator = _allocator(2)
    source = allocator.allocate(torch.Size([PAYLOAD_BYTES]), torch.uint8)
    destination = allocator.allocate(torch.Size([PAYLOAD_BYTES]), torch.uint8)
    assert source is not None and destination is not None
    source_physical = source.physical_tensor
    destination_physical = destination.physical_tensor
    assert source_physical is not None and destination_physical is not None
    source_physical.fill_(ord("S"))
    source.raw_tensor.fill_(ord("P"))
    destination_physical.fill_(ord("T"))
    path = make_raw_block_file(directory)
    config = replace(
        make_raw_block_core_config(path),
        io_engine="io_uring",
        use_odirect=True,
        enable_zero_copy=True,
    )
    core = RawBlockCore(config, key_namespace="object")
    try:
        core.register_fixed_buffers_from_allocator(allocator)
        spec = encode_object_key(make_object_key(123))
        assert core.put_many([spec], [source]).results == [True]
        assert core.load_many_into([spec.encoded], [destination]) == [True]
        assert destination.raw_tensor.tolist() == [ord("P")] * PAYLOAD_BYTES
        assert destination_physical[PAYLOAD_BYTES:].tolist() == (
            [ord("T")] * (PHYSICAL_BYTES - PAYLOAD_BYTES)
        )
        assert source_physical[PAYLOAD_BYTES:].tolist() == (
            [ord("S")] * (PHYSICAL_BYTES - PAYLOAD_BYTES)
        )
        assert not core.is_poisoned()
    finally:
        outcome = core.close()
        assert outcome.may_release_backing_resources
        source.ref_count_down()
        destination.ref_count_down()
        allocator.close()
