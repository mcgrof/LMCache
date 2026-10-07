# SPDX-License-Identifier: Apache-2.0
"""Check host-export lifetime with real mappings and mocked exporter ioctls."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock
import errno
import fcntl
import mmap
import os
import struct

# Third Party
import pytest

# First Party
from lmcache.v1 import memory_management as memory
from lmcache.v1.storage_backend.raw_block import RawBlockCore


@pytest.fixture
def exporter(monkeypatch):
    state = SimpleNamespace(owners=[], controls=[], mappings=[], failure="")
    real_open = os.open
    real_memfd = os.memfd_create
    real_mmap = mmap.mmap
    real_tensor = memory.torch.frombuffer
    spec = SimpleNamespace(
        is_pin_supported=True,
        pin_memory=Mock(return_value=True),
        unpin_memory=Mock(return_value=True),
    )
    monkeypatch.setattr(memory, "current_device_spec", spec)

    def make_memfd(name, flags=0):
        fd = real_memfd(name, flags)
        if name == "lmcache-kv":
            state.owners.append(fd)
        return fd

    def open_control(path, flags, *args, **kwargs):
        if path == "/dev/udmabuf" or str(path).startswith("/dev/dma_heap/"):
            fd = os.memfd_create("export-control", os.MFD_CLOEXEC)
            state.controls.append((path, fd))
            return fd
        return real_open(path, flags, *args, **kwargs)

    def ioctl(fd, command, request):
        if state.failure == "ioctl":
            raise OSError("export failed")
        if command == memory._UDMABUF_CREATE:
            backing, flags, offset, size = struct.unpack("IIQQ", request)
            assert flags == memory._UDMABUF_FLAGS_CLOEXEC and offset == 0
            assert os.fstat(backing).st_size == size
            exported = os.dup(backing)
            state.owners.append(exported)
            return exported
        assert command == memory._DMA_HEAP_IOCTL_ALLOC
        size, _, flags, heap_flags = struct.unpack("QIIQ", request)
        assert flags & os.O_CLOEXEC and heap_flags == 0
        exported = os.memfd_create("heap-export", os.MFD_CLOEXEC)
        os.ftruncate(exported, size)
        state.owners.append(exported)
        request[:] = struct.pack("QIIQ", size, exported, flags, heap_flags)
        return 0

    def map_arena(*args, **kwargs):
        if state.failure == "mmap":
            raise OSError("mapping failed")
        mapping = real_mmap(*args, **kwargs)
        state.mappings.append(mapping)
        return mapping

    def make_tensor(*args, **kwargs):
        if state.failure == "tensor":
            raise RuntimeError("tensor setup failed")
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(os, "open", open_control)
    monkeypatch.setattr(os, "memfd_create", make_memfd)
    monkeypatch.setattr(fcntl, "ioctl", ioctl)
    monkeypatch.setattr(mmap, "mmap", map_arena)
    monkeypatch.setattr(memory.torch, "frombuffer", make_tensor)
    yield state, spec


def assert_closed(fd):
    with pytest.raises(OSError) as error:
        os.fstat(fd)
    assert error.value.errno == errno.EBADF


@pytest.mark.parametrize(
    "kind,pinned",
    [
        ("udmabuf", True),
        ("udmabuf", False),
        ("system_heap", False),
        ("cma_heap", False),
        ("/dev/dma_heap/pernuma0", False),
    ],
)
def test_host_export_keeps_descriptors_until_release(exporter, kind, pinned):
    state, spec = exporter
    spec.pin_memory.return_value = pinned
    buffer = memory._allocate_cpu_memory(5000, dmabuf=kind)
    ptr = buffer.data_ptr()
    region = memory.get_dmabuf_region(ptr)
    assert region is not None
    fd, base = region
    assert base == ptr and buffer.numel() == 8192
    assert memory.get_dmabuf_region(ptr + 8191) == (fd, ptr)
    assert memory.get_dmabuf_region(ptr + 8192) is None
    for owner in state.owners:
        os.fstat(owner)
    if kind == "udmabuf":
        spec.pin_memory.assert_called_once_with(ptr, 8192)
    else:
        spec.pin_memory.assert_not_called()
    memory._free_cpu_memory(buffer, 5000)
    if pinned:
        spec.unpin_memory.assert_called_once_with(ptr)
    else:
        spec.unpin_memory.assert_not_called()
    assert memory.get_dmabuf_region(ptr) is None
    assert all(mapping.closed for mapping in state.mappings)
    for owner in state.owners:
        assert_closed(owner)


@pytest.mark.parametrize("kind", ["udmabuf", "system_heap"])
@pytest.mark.parametrize("failure", ["ioctl", "mmap", "tensor"])
def test_failed_host_setup_releases_every_export(exporter, kind, failure):
    state, spec = exporter
    state.failure = failure
    before = dict(memory._DMABUF_REGIONS)
    with pytest.raises((OSError, RuntimeError), match="failed"):
        memory._allocate_cpu_memory(5000, dmabuf=kind)
    assert memory._DMABUF_REGIONS == before
    assert all(mapping.closed for mapping in state.mappings)
    for owner in state.owners:
        assert_closed(owner)
    for _, control in state.controls:
        assert_closed(control)
    spec.unpin_memory.assert_not_called()


def test_unpin_failure_keeps_export_and_mapping_owned(exporter):
    state, spec = exporter
    buffer = memory._allocate_cpu_memory(4096, dmabuf="udmabuf")
    ptr = buffer.data_ptr()
    region = memory.get_dmabuf_region(ptr)
    spec.unpin_memory.side_effect = RuntimeError("unpin failed")
    try:
        with pytest.raises(RuntimeError, match="unpin failed"):
            memory._free_cpu_memory(buffer, 4096)
        assert memory.get_dmabuf_region(ptr) == region
        assert all(not mapping.closed for mapping in state.mappings)
        for owner in state.owners:
            os.fstat(owner)
    finally:
        spec.unpin_memory.side_effect = None
        memory._free_cpu_memory(buffer, 4096)


def test_rejected_unpin_keeps_export_and_mapping_owned(exporter):
    """The platform reports unregister failures as False, not an exception."""
    state, spec = exporter
    buffer = memory._allocate_cpu_memory(4096, dmabuf="udmabuf")
    ptr = buffer.data_ptr()
    region = memory.get_dmabuf_region(ptr)
    spec.unpin_memory.return_value = False
    try:
        with pytest.raises(RuntimeError, match="unpin|unregister"):
            memory._free_cpu_memory(buffer, 4096)
        assert memory.get_dmabuf_region(ptr) == region
        assert all(not mapping.closed for mapping in state.mappings)
        for owner in state.owners:
            os.fstat(owner)
    finally:
        spec.unpin_memory.return_value = True
        if memory.get_dmabuf_region(ptr) is not None:
            memory._free_cpu_memory(buffer, 4096)


@pytest.mark.parametrize("path", ["dmabuf", "refused", "passthrough"])
def test_core_prefers_exported_host_regions_only_for_fixed_rw(monkeypatch, path):
    core = object.__new__(RawBlockCore)
    core.io_engine = "io_uring"
    core.use_uring_cmd = path == "passthrough"
    raw = SimpleNamespace(register_fixed_dmabufs=Mock(), register_fixed_buffers=Mock())
    if path == "refused":
        raw.register_fixed_dmabufs.side_effect = OSError("unsupported registration")
    monkeypatch.setattr(core, "_rawdev", lambda: raw)
    tensor = SimpleNamespace(
        data_ptr=lambda: 0x1000, numel=lambda: 4096, element_size=lambda: 1
    )
    allocator = SimpleNamespace(
        get_paged_buffers=lambda: [tensor],
        get_paged_dmabuf_regions=lambda: [(11, 0x1000)],
    )
    core.register_fixed_buffers_from_allocator(allocator)
    if path == "passthrough":
        raw.register_fixed_dmabufs.assert_not_called()
    else:
        raw.register_fixed_dmabufs.assert_called_once_with(
            [0x1000], [4096], [11], [0x1000]
        )
    if path == "dmabuf":
        raw.register_fixed_buffers.assert_not_called()
    else:
        raw.register_fixed_buffers.assert_called_once_with([0x1000], [4096])
