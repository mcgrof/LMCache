# SPDX-License-Identifier: Apache-2.0
"""Strict target admission on explicitly assigned XFS/ext4 scratch space."""

# Standard
from collections.abc import Iterator
from pathlib import Path
from typing import Any
import fcntl
import os
import tempfile

# Third Party
import pytest

native = pytest.importorskip("lmcache_rust_raw_block_io")
CAPACITY = 64 * 1024


@pytest.fixture
def target() -> Iterator[Path]:
    """Create a new initialized file only in the opted-in scratch directory."""
    directory = os.environ.get("LMCACHE_TEST_STRICT_FILESYSTEM_DIR")
    if not directory:
        pytest.skip("set LMCACHE_TEST_STRICT_FILESYSTEM_DIR to assigned XFS/ext4 space")
    with tempfile.TemporaryDirectory(prefix="strict-file-", dir=directory) as task:
        path = Path(task) / "cache.bin"
        path.write_bytes(bytes(CAPACITY))
        yield path


def open_target(path: Path, direct: bool = True) -> Any:
    """Open the real native file descriptor without io_uring for preflight tests."""
    return native.RawBlockDevice(
        str(path), writable=True, use_odirect=direct, alignment=4096, io_engine="posix"
    )


def test_initialized_file_and_open_descriptor_identity(target: Path) -> None:
    device = open_target(target)
    try:
        renamed = target.with_suffix(".original")
        target.rename(renamed)
        target.touch()
        assert device.validate_dmabuf_target(CAPACITY) == "file"
    finally:
        device.close()


@pytest.mark.parametrize("capacity", [0, CAPACITY + 4096])
def test_invalid_capacity_is_rejected(target: Path, capacity: int) -> None:
    device = open_target(target)
    try:
        with pytest.raises(ValueError, match="capacity must be positive and fit"):
            device.validate_dmabuf_target(capacity)
    finally:
        device.close()


def test_sparse_file_is_rejected(target: Path) -> None:
    with target.open("wb") as handle:
        handle.write(bytes(4096))
        handle.seek(CAPACITY - 4096)
        handle.write(bytes(4096))
    device = open_target(target)
    try:
        with pytest.raises(ValueError, match="hole"):
            device.validate_dmabuf_target(CAPACITY)
    finally:
        device.close()


def test_preallocated_unwritten_file_is_rejected(target: Path) -> None:
    with target.open("wb") as handle:
        os.posix_fallocate(handle.fileno(), 0, CAPACITY)
    device = open_target(target)
    try:
        with pytest.raises(ValueError, match="initialized private extents"):
            device.validate_dmabuf_target(CAPACITY)
    finally:
        device.close()


def test_truncation_after_open_is_rejected(target: Path) -> None:
    device = open_target(target)
    try:
        with target.open("r+b") as handle:
            handle.truncate(4096)
        with pytest.raises(ValueError, match="capacity must be positive and fit"):
            device.validate_dmabuf_target(CAPACITY)
    finally:
        device.close()


def test_buffered_target_is_rejected(target: Path) -> None:
    device = open_target(target, direct=False)
    try:
        with pytest.raises(ValueError, match="ordinary O_DIRECT"):
            device.validate_dmabuf_target(CAPACITY)
    finally:
        device.close()


def test_reflink_extent_is_rejected(target: Path) -> None:
    clone = target.with_suffix(".clone")
    with target.open("rb") as source, clone.open("xb") as destination:
        try:
            fcntl.ioctl(destination, 0x40049409, source.fileno())  # FICLONE
        except OSError as exc:
            pytest.skip(f"assigned filesystem does not support reflinks: {exc}")
    device = open_target(clone)
    try:
        with pytest.raises(ValueError, match="initialized private extents"):
            device.validate_dmabuf_target(CAPACITY)
    finally:
        device.close()
