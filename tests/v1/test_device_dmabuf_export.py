# SPDX-License-Identifier: Apache-2.0
"""Check device-export ownership with driver calls replaced and real file handles."""

# Standard
from types import SimpleNamespace
from typing import Any, cast
import ctypes
import errno
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1 import memory_management


@pytest.fixture
def device_tensor(monkeypatch: pytest.MonkeyPatch) -> torch.Tensor:
    """Describe two aligned chunks without allocating or accessing GPU memory."""
    sysconf = os.sysconf
    monkeypatch.setattr(
        os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else sysconf(name)
    )
    return cast(
        torch.Tensor,
        SimpleNamespace(
            device=torch.device("cuda"),
            data_ptr=lambda: 0x100000000,
            numel=lambda: 8192,
            element_size=lambda: 1,
        ),
    )


def assert_fd_closed(fd: int) -> None:
    """Check that the exported handle has been returned to the operating system."""
    with pytest.raises(OSError) as error:
        os.fstat(fd)
    assert error.value.errno == errno.EBADF


def close_if_open(fd: int) -> None:
    """Keep a failed assertion from leaking a test handle."""
    try:
        os.close(fd)
    except OSError as error:
        if error.errno != errno.EBADF:
            raise


@pytest.mark.parametrize("hip", [False, True], ids=["cuda", "hip"])
def test_failed_device_export_releases_prior_chunks(
    monkeypatch: pytest.MonkeyPatch, device_tensor: torch.Tensor, hip: bool
) -> None:
    """Publish no usable regions when exporting a later chunk fails."""
    exported: list[int] = []

    def export_chunk(ptr: int, size: int) -> int | tuple[int, int]:
        assert size == 4096
        if exported:
            raise RuntimeError("second chunk export failed")
        fd = os.memfd_create("device-export-test", os.MFD_CLOEXEC)
        exported.append(fd)
        return (fd, 0) if hip else fd

    monkeypatch.setattr(memory_management, "_DMABUF_CHUNK_BYTES", 4096)
    monkeypatch.setattr(torch.version, "hip", "test-driver" if hip else None)
    exporter = "_export_hip_dmabuf" if hip else "_export_cuda_dmabuf"
    monkeypatch.setattr(memory_management, exporter, export_chunk)
    base = device_tensor.data_ptr()
    try:
        with pytest.raises(RuntimeError, match="second chunk export failed"):
            memory_management.export_device_dmabufs(device_tensor)
        assert len(exported) == 1
        assert_fd_closed(exported[0])
        assert memory_management.get_dmabuf_region(base) is None
        assert memory_management.get_dmabuf_region(base + 4096) is None
    finally:
        memory_management.release_device_dmabufs(device_tensor)
        for fd in exported:
            close_if_open(fd)


@pytest.mark.parametrize("page_bytes", [4096, 65536])
@pytest.mark.parametrize("base_delta,size", [(0, 65536), (4096, 65536), (0, 4096)])
def test_device_export_obeys_host_page_geometry(
    monkeypatch: pytest.MonkeyPatch, page_bytes: int, base_delta: int, size: int
) -> None:
    """Export validates the host page size independently of 4-KiB I/O slots."""
    base = 0x200000000 + base_delta
    tensor = cast(
        torch.Tensor,
        SimpleNamespace(
            device=torch.device("cuda"),
            data_ptr=lambda: base,
            numel=lambda: size,
            element_size=lambda: 1,
        ),
    )
    sysconf = os.sysconf
    monkeypatch.setattr(
        os,
        "sysconf",
        lambda name: page_bytes if name == "SC_PAGE_SIZE" else sysconf(name),
    )
    exported: list[int] = []

    def export_chunk(ptr: int, length: int) -> int:
        assert ptr % page_bytes == length % page_bytes == 0
        fd = os.memfd_create("device-page-geometry-test", os.MFD_CLOEXEC)
        exported.append(fd)
        return fd

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(memory_management, "_export_cuda_dmabuf", export_chunk)
    try:
        if base % page_bytes or size % page_bytes:
            with pytest.raises(ValueError, match="not page aligned"):
                memory_management.export_device_dmabufs(tensor)
            assert not exported
        else:
            regions = memory_management.export_device_dmabufs(tensor)
            assert regions == [(exported[0], base, size)]
    finally:
        memory_management.release_device_dmabufs(tensor)
        for fd in exported:
            close_if_open(fd)


def test_hip_export_releases_a_handle_with_an_invalid_offset(
    monkeypatch: pytest.MonkeyPatch, device_tensor: torch.Tensor
) -> None:
    """Close an export whose zero-offset origin cannot be represented."""
    fd = os.memfd_create("hip-offset-test", os.MFD_CLOEXEC)

    def export_with_offset(ptr: int, size: int, fd_out: Any, offset_out: Any) -> int:
        ctypes.cast(fd_out, ctypes.POINTER(ctypes.c_int))[0] = fd
        ctypes.cast(offset_out, ctypes.POINTER(ctypes.c_uint64))[0] = ptr + 4096
        return 0

    monkeypatch.setattr(torch.version, "hip", "test-driver")
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda name: SimpleNamespace(hsa_amd_portable_export_dmabuf=export_with_offset),
    )
    try:
        with pytest.raises(RuntimeError, match="offset"):
            memory_management.export_device_dmabufs(device_tensor)
        assert_fd_closed(fd)
        assert memory_management.get_dmabuf_region(device_tensor.data_ptr()) is None
    finally:
        memory_management.release_device_dmabufs(device_tensor)
        close_if_open(fd)


def test_hip_export_preserves_offsets_without_exposing_neighboring_memory(
    monkeypatch: pytest.MonkeyPatch, device_tensor: torch.Tensor
) -> None:
    """Two exported slices may share an origin without sharing handle ownership."""
    exported: list[int] = []
    origin = device_tensor.data_ptr() - 4096

    def export_with_offset(ptr: int, size: int, fd_out: Any, offset_out: Any) -> int:
        fd = os.memfd_create("hip-offset-test", os.MFD_CLOEXEC)
        exported.append(fd)
        ctypes.cast(fd_out, ctypes.POINTER(ctypes.c_int))[0] = fd
        ctypes.cast(offset_out, ctypes.POINTER(ctypes.c_uint64))[0] = ptr - origin
        return 0

    monkeypatch.setattr(memory_management, "_DMABUF_CHUNK_BYTES", 4096)
    monkeypatch.setattr(torch.version, "hip", "test-driver")
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda name: SimpleNamespace(hsa_amd_portable_export_dmabuf=export_with_offset),
    )
    base = device_tensor.data_ptr()
    try:
        regions = memory_management.export_device_dmabufs(device_tensor)
        assert regions == [(exported[0], base, 4096), (exported[1], base + 4096, 4096)]
        assert memory_management.get_dmabuf_region(base - 1) is None
        assert memory_management.get_dmabuf_region(base + 8192) is None
        for index, fd in enumerate(exported):
            address = base + index * 4096
            assert memory_management.get_dmabuf_region(address) == (fd, origin)
            assert memory_management.get_dmabuf_region(address + 4095) == (fd, origin)
        memory_management.release_device_dmabufs(device_tensor)
        for fd in exported:
            assert_fd_closed(fd)
        assert memory_management.get_dmabuf_region(base) is None
        assert memory_management.get_dmabuf_region(base + 4096) is None
    finally:
        memory_management.release_device_dmabufs(device_tensor)
        for fd in exported:
            close_if_open(fd)


def test_successful_device_export_keeps_handles_until_release(
    monkeypatch: pytest.MonkeyPatch, device_tensor: torch.Tensor
) -> None:
    """Resolve each complete export while its owner holds the handles."""
    exported: list[int] = []

    def export_chunk(ptr: int, size: int) -> int:
        fd = os.memfd_create("device-export-test", os.MFD_CLOEXEC)
        exported.append(fd)
        return fd

    monkeypatch.setattr(memory_management, "_DMABUF_CHUNK_BYTES", 4096)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(memory_management, "_export_cuda_dmabuf", export_chunk)
    base = device_tensor.data_ptr()
    try:
        regions = memory_management.export_device_dmabufs(device_tensor)
        assert regions == [(exported[0], base, 4096), (exported[1], base + 4096, 4096)]
        for fd, address, size in regions:
            os.fstat(fd)
            assert memory_management.get_dmabuf_region(address + size - 1) == (
                fd,
                address,
            )
        memory_management.release_device_dmabufs(device_tensor)
        for fd, address, _size in regions:
            assert_fd_closed(fd)
            assert memory_management.get_dmabuf_region(address) is None
    finally:
        memory_management.release_device_dmabufs(device_tensor)
        for fd in exported:
            close_if_open(fd)
