# SPDX-License-Identifier: Apache-2.0
"""Exercise ordinary io_uring batches at and beyond the submission-ring depth.

Use a deliberately small four-entry ring and synthetic per-offset payloads.
Reads and writes span one and several admission passes, with repeated batches
checking that completions remain attributed to the correct operation. Payload
comparisons detect dropped or misattributed work, not duplicate identical I/O.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import os

# Third Party
import pytest

# First Party
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    make_raw_block_file,
)

lmcache_rust_raw_block_io = pytest.importorskip("lmcache_rust_raw_block_io")
RawBlockDevice = lmcache_rust_raw_block_io.RawBlockDevice

RING_DEPTH = 4
# One pass under the ring, exactly the ring, one entry past it, and a count
# that forces the worker through more than two passes.
OPERATION_COUNTS = [RING_DEPTH - 1, RING_DEPTH, RING_DEPTH + 1, 2 * RING_DEPTH + 1]


def _payload(index: int, length: int) -> bytearray:
    """Build a payload whose every byte identifies the operation it belongs to."""
    return bytearray(bytes([(index * 7 + 1) & 0xFF]) * length)


def _open_device(path: Path):
    kwargs = dict(
        writable=True,
        use_odirect=False,
        use_iouring=True,
        alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
        io_engine="io_uring",
        iouring_queue_depth=RING_DEPTH,
    )
    return RawBlockDevice(str(path), **kwargs)


@pytest.mark.parametrize("count", OPERATION_COUNTS)
def test_batched_write_preserves_every_payload(tmp_path: Path, count: int) -> None:
    path = make_raw_block_file(tmp_path)
    length = RAW_BLOCK_CI_BLOCK_ALIGN
    offsets = [RAW_BLOCK_CI_BLOCK_ALIGN * (i + 1) for i in range(count)]
    payloads = [_payload(i, length) for i in range(count)]

    dev = _open_device(path)
    try:
        batch_id = dev.batched_write(offsets, payloads, [length] * count, None)
        succeeded, errors = dev.wait_iouring(batch_id)
        assert list(errors) == []
        assert list(succeeded) == [True] * count
    finally:
        dev.close()

    with open(path, "rb") as f:
        for offset, expected in zip(offsets, payloads, strict=True):
            f.seek(offset)
            assert f.read(length) == bytes(expected)


@pytest.mark.parametrize("count", OPERATION_COUNTS)
def test_batched_read_returns_each_operations_own_bytes(
    tmp_path: Path, count: int
) -> None:
    path = make_raw_block_file(tmp_path)
    length = RAW_BLOCK_CI_BLOCK_ALIGN
    offsets = [RAW_BLOCK_CI_BLOCK_ALIGN * (i + 1) for i in range(count)]
    payloads = [_payload(i, length) for i in range(count)]

    with open(path, "r+b") as f:
        for offset, payload in zip(offsets, payloads, strict=True):
            f.seek(offset)
            f.write(bytes(payload))
        f.flush()
        os.fsync(f.fileno())

    dev = _open_device(path)
    try:
        buffers = [bytearray(length) for _ in range(count)]
        batch_id = dev.batched_read(offsets, buffers, [length] * count)
        succeeded, errors = dev.wait_iouring(batch_id)
        assert list(errors) == []
        assert list(succeeded) == [True] * count
    finally:
        dev.close()

    for got, expected in zip(buffers, payloads, strict=True):
        assert got == expected


def test_ring_pressure_survives_repeated_batches(tmp_path: Path) -> None:
    """Keep the ring under pressure across batches on one long-lived device.

    Resident entries belong to the batch that pushed them, so a later batch
    must neither adopt them nor issue them again.
    """
    path = make_raw_block_file(tmp_path)
    length = RAW_BLOCK_CI_BLOCK_ALIGN
    count = 2 * RING_DEPTH + 1
    dev = _open_device(path)
    try:
        for round_index in range(4):
            base = RAW_BLOCK_CI_BLOCK_ALIGN * (1 + round_index * count)
            offsets = [base + RAW_BLOCK_CI_BLOCK_ALIGN * i for i in range(count)]
            payloads = [_payload(round_index * count + i, length) for i in range(count)]
            batch_id = dev.batched_write(offsets, payloads, [length] * count, None)
            succeeded, errors = dev.wait_iouring(batch_id)
            assert list(errors) == []
            assert list(succeeded) == [True] * count

            read_back = [bytearray(length) for _ in range(count)]
            batch_id = dev.batched_read(offsets, read_back, [length] * count)
            succeeded, errors = dev.wait_iouring(batch_id)
            assert list(errors) == []
            assert list(succeeded) == [True] * count
            for got, expected in zip(read_back, payloads, strict=True):
                assert got == expected
    finally:
        dev.close()
