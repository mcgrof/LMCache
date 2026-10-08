# SPDX-License-Identifier: Apache-2.0
"""Opt-in native connector/ordinary io_uring DMA-BUF ordering regression.

Run on a qualified GPU/exporter/topology and the patched filesystem kernel::

    LMCACHE_RUN_DMABUF_ORDERING=1 LMCACHE_DMABUF_TEST_DIR=/chosen/xfs/directory \
        pytest -xvs tests/v1/gpu_connector/test_vllm_staging_ordering_dmabuf.py

Only a newly created exclusive regular file is written. No raw-device path is
accepted. The file is allocated and initialized before registration. Registration
stays warm across generation changes, and both directions use independent CPU
oracles rather than a round trip that could reproduce the same wrong bytes.

This tests the real connector, allocator exporter and RawBlockDevice, NOT the
cache engine or P/D READY/ACK protocol. It exercises the production external-DMA
visibility contract without adding a guessed flush or a device-wide synchronize
between storage-read completion and scatter. Unsupported export, registration,
filesystem or visibility fails the explicitly enabled test; it never falls back.
CUDA or ROCm and storage hardware are required; collection is not qualification.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import json
import mmap
import os
import stat
import tempfile

# Third Party
import pytest

if os.environ.get("LMCACHE_RUN_DMABUF_ORDERING") != "1":
    pytest.skip(
        "real DMA-BUF ordering test requires explicit opt-in", allow_module_level=True
    )

# Third Party
# Opt-in dependency failures must remain failures, rather than importorskip.
import torch  # noqa: E402

# isort: split
# Third Party
from lmcache_rust_raw_block_io import RawBlockDevice  # noqa: E402

# First Party
from lmcache import device_ops  # noqa: E402
from lmcache.v1.gpu_connector.gpu_connectors import (  # noqa: E402
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemGPUConnectorV3,
)
from lmcache.v1.memory_allocators.gpu_memory_allocator import (  # noqa: E402
    GPUMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat  # noqa: E402
from lmcache.v1.metadata import LMCacheMetadata  # noqa: E402
import lmcache.cuda_ops as cuda_ops  # noqa: E402

SLOT_BYTES = 4096
FILE_BYTES = 64 * 1024
WRITE_OFFSET = SLOT_BYTES
READ_OFFSET = 3 * SLOT_BYTES
STAGING_SHAPE = torch.Size([2, 1, 16, 64])
KV_SHAPE = (2, 8, 16, 1, 64)
# An unknown CUDA/native outcome keeps all owners alive for the process lifetime.
# Do not resume hardware work in this process after this list becomes nonempty.
QUARANTINED_OWNERS: list[tuple[object, ...]] = []


def _host_read(fd: int, offset: int, size: int) -> bytes:
    """Read aligned bytes through the independent host O_DIRECT descriptor."""
    with mmap.mmap(-1, size) as aligned:
        result = os.preadv(fd, [aligned], offset)
        assert result == size, f"short host O_DIRECT read: {result}/{size}"
        return aligned[:]


def _host_write(fd: int, offset: int, payload: bytes) -> None:
    """Write aligned oracle bytes using a page-aligned host buffer."""
    with mmap.mmap(-1, len(payload)) as aligned:
        aligned[:] = payload
        result = os.pwritev(fd, [aligned], offset)
        assert result == len(payload), f"short host O_DIRECT write: {result}"


def _delay(stream: torch.cuda.Stream) -> torch.cuda.Event:
    """Enqueue finite native GPU work and return its completion event.

    PyTorch's private test-only sleep primitive deliberately produces a native
    device delay without replacing the transfer kernel or using CPU sleep.
    """
    event = torch.cuda.Event()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)  # noqa: SLF001
        event.record(stream)
    assert not event.query(), "CUDA delay expired: this trial cannot expose the race"
    return event


def _cpu_fixture(generation: int, tokens: int) -> tuple[torch.Tensor, bytes]:
    """Create KV bytes and an independent oracle for the selected token count."""
    source = (
        (
            (
                torch.arange(2 * 8 * 16 * 64, dtype=torch.int32, device="cpu") * 7
                + generation * 19
            )
            % 251
        )
        .to(torch.float16)
        .reshape(KV_SHAPE)
    )
    # This CPU gather is independent of the native transfer kernel.
    packed = source.reshape(2, 128, 64)[:, 37 : 37 + tokens].unsqueeze(1).contiguous()
    return source, packed.view(torch.uint8).numpy().tobytes()


@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("tokens", [16, 5])
def test_native_staging_dmabuf_ordering(version: int, tokens: int) -> None:
    """Check delayed native gathers/writes and independent reads/scatters.

    Args:
        version: V2 or V3 connector implementation to exercise.
        tokens: Full or partial chunk; a five-token write must zero 2,816 tail bytes.

    Raises:
        AssertionError: Byte comparisons or completion/readiness contracts fail.
        RuntimeError: GPU, export, registration or storage operations fail.
        ValueError: The explicit test-directory configuration is absent/invalid.
    """
    if QUARANTINED_OWNERS:
        pytest.fail("a prior test has unknown DMA ownership; restart the process")
    if not torch.cuda.is_available():
        pytest.fail("explicit DMA-BUF test requires a CUDA or ROCm runtime")
    for option in (
        "CUDA_LAUNCH_BLOCKING",
        "HIP_LAUNCH_BLOCKING",
        "AMD_SERIALIZE_KERNEL",
        "AMD_SERIALIZE_COPY",
    ):
        if os.environ.get(option, "0") not in ("", "0"):
            pytest.fail(f"unset {option}: it hides the race under test")
    if not hasattr(torch.cuda, "_sleep"):
        pytest.fail("native finite GPU delay is unavailable")
    if device_ops.multi_layer_kv_transfer is not cuda_ops.multi_layer_kv_transfer:
        pytest.fail("test requires the native GPU transfer binding, not a fallback")
    directory_setting = os.environ.get("LMCACHE_DMABUF_TEST_DIR", "")
    if not directory_setting:
        raise ValueError(
            "set LMCACHE_DMABUF_TEST_DIR to the patched XFS/ext4 directory"
        )
    directory = Path(directory_setting).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("LMCACHE_DMABUF_TEST_DIR must be a directory, never a device")

    metadata = LMCacheMetadata(
        model_name="staging-ordering-regression",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(1, 2, 16, 1, 64),
        chunk_size=16,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    connector_class = (
        VLLMPagedMemGPUConnectorV2 if version == 2 else VLLMPagedMemGPUConnectorV3
    )
    connector = connector_class.from_metadata(
        metadata, device=device, use_gpu=False, layout_hints={"kv_layout": "NHD"}
    )
    allocator = GPUMemoryAllocator(
        2 * 1024 * 1024,
        device=device,
        use_paging=True,
        buffer_provider=os.environ.get("LMCACHE_DMABUF_TEST_PROVIDER", "native"),
        shapes=[STAGING_SHAPE],
        dtypes=[torch.float16],
        fmt=MemoryFormat.KV_2LTD,
    )
    guard_obj = allocator.allocate(STAGING_SHAPE, torch.float16, MemoryFormat.KV_2LTD)
    staging_shape = torch.Size([2, 1, tokens, 64])
    staging = allocator.allocate(staging_shape, torch.float16, MemoryFormat.KV_2LTD)
    assert guard_obj is not None and staging is not None
    objects = [guard_obj, staging]
    assert staging.get_size() == tokens * 256
    raw = staging.physical_tensor
    assert raw is not None and raw.nbytes == SLOT_BYTES and raw.is_cuda
    guard = guard_obj.raw_tensor
    assert guard is not None
    guard.fill_(0xA7)
    kv = [torch.zeros(KV_SHAPE, dtype=torch.float16, device=device)]
    slots = torch.arange(37, 37 + tokens, dtype=torch.int64, device=device)
    producer = torch.cuda.Stream(device=device)
    # Warm pointer setup and both native kernels before timed/adversarial work.
    connector.batched_from_gpu(
        [staging], [0], [tokens], kvcaches=kv, slot_mapping=slots
    )
    connector.batched_to_gpu([staging], [0], [tokens], kvcaches=kv, slot_mapping=slots)
    torch.cuda.current_stream(device).synchronize()

    fd, filename = tempfile.mkstemp(prefix="lmcache-dmabuf-ordering-", dir=directory)
    path = Path(filename)
    host_fd = -1
    raw_device: RawBlockDevice | None = None
    owners: tuple[object, ...] = (allocator, objects, connector, kv, slots, producer)
    try:
        assert stat.S_ISREG(os.fstat(fd).st_mode)
        os.posix_fallocate(fd, 0, FILE_BYTES)
        assert os.pwrite(fd, bytes([0xA5]) * FILE_BYTES, 0) == FILE_BYTES
        os.fdatasync(fd)
        host_fd = os.open(path, os.O_RDWR | os.O_DIRECT)
        raw_device = RawBlockDevice(
            str(path),
            writable=True,
            use_odirect=True,
            alignment=SLOT_BYTES,
            io_engine="io_uring",
            iouring_queue_depth=8,
        )
        owners += (raw_device,)
        buffers = allocator.get_paged_buffers()
        regions = allocator.get_paged_dmabuf_regions()
        assert buffers is not None and regions is not None, "GPU DMA-BUF export failed"
        assert raw.data_ptr() == buffers[1].data_ptr(), "test must use the second slot"
        export_offset = raw.data_ptr() - regions[1][1]
        assert export_offset > 0, "test must exercise a nonzero DMA-BUF offset"
        raw_device.register_fixed_dmabufs(
            [buffer.data_ptr() for buffer in buffers[:2]],
            [buffer.nbytes for buffer in buffers[:2]],
            [region[0] for region in regions[:2]],
            [region[1] for region in regions[:2]],
        )
        print(
            f"file={path} GPU_slot={raw.data_ptr():#x} slot_bytes={SLOT_BYTES} "
            f"dmabuf_offset={export_offset}"
        )
        for generation in (3, 17, 53, 97):
            source, expected_write = _cpu_fixture(generation, tokens)
            prepared_source = source.to(device)
            owners += (prepared_source,)
            kv[0].fill_(-1)
            raw.fill_(0xD3)
            torch.cuda.current_stream(device).synchronize()
            gather_delay = _delay(connector.store_stream)
            producer_delay = _delay(producer)
            with torch.cuda.stream(producer):
                kv[0].copy_(prepared_source)
                # The caller stream is the explicit producer dependency.
                connector.batched_from_gpu(
                    [staging], [0], [tokens], kvcaches=kv, slot_mapping=slots
                )
            assert producer_delay.query(), "gather returned before its producer"
            assert gather_delay.query() and connector.store_stream.query(), (
                "gather returned with work still queued on its stream"
            )
            # Do not insert a tensor comparison or any CUDA wait before submit:
            # doing so would hide the gather-ready race in the original code.
            batch = raw_device.batched_write(
                [WRITE_OFFSET],
                [raw],
                [SLOT_BYTES],
                request_tag=f"ordering/v{version}/g{generation}/write",
            )
            assert raw_device.wait_iouring(batch) == ([True], [])
            assert _host_read(host_fd, WRITE_OFFSET, SLOT_BYTES) == (
                expected_write + bytes(SLOT_BYTES - len(expected_write))
            ), "stored physical padding contains bytes from a prior owner"

            _, expected_read = _cpu_fixture(generation + 101, tokens)
            _host_write(
                host_fd,
                READ_OFFSET,
                expected_read + bytes(SLOT_BYTES - len(expected_read)),
            )
            os.fdatasync(host_fd)
            raw.fill_(0x3B)
            kv[0].fill_(-2)
            torch.cuda.current_stream(device).synchronize()
            # Storage completion must precede scatter. This load-stream delay
            # also checks that the public batch method drains its consumer.
            consumer_delay = _delay(connector.load_stream)
            batch = raw_device.batched_read(
                [READ_OFFSET],
                [raw],
                [SLOT_BYTES],
                request_tag=f"ordering/v{version}/g{generation}/read",
            )
            assert raw_device.wait_iouring(batch) == ([True], [])
            connector.batched_to_gpu(
                [staging], [0], [tokens], kvcaches=kv, slot_mapping=slots
            )
            assert consumer_delay.query(), "scatter returned while still queued"
            assert connector.load_stream.query(), "scatter still running at handoff"
            # Reuse staging immediately on another stream after scatter returns.
            with torch.cuda.stream(producer):
                raw.fill_(0xD9)
            restored = kv[0].reshape(2, 128, 64)[:, 37 : 37 + tokens].contiguous()
            assert restored.cpu().view(torch.uint8).numpy().tobytes() == expected_read
            producer.synchronize()
            assert torch.all(guard == 0xA7).item(), "neighboring GPU slot changed"
            media = _host_read(host_fd, 0, FILE_BYTES)
            for begin, end in (
                (0, WRITE_OFFSET),
                (2 * SLOT_BYTES, READ_OFFSET),
                (4 * SLOT_BYTES, FILE_BYTES),
            ):
                assert media[begin:end] == bytes([0xA5]) * (end - begin)
            print(
                f"V{version} tokens={tokens} generation={generation}: "
                "write/read/padding/guards PASS"
            )
        journal, dropped = raw_device.take_io_journal()
        assert dropped == 0 and journal, "native route evidence is incomplete"
        assert all(row["path"] == "dmabuf_fixed" for row in journal)
        print(
            json.dumps(
                {
                    "version": version,
                    "tokens": tokens,
                    "journal": journal,
                    "dropped": dropped,
                }
            )
        )
    finally:
        try:
            if raw_device is not None:
                if raw_device.is_poisoned():
                    raise RuntimeError(
                        "native completion unknown; retaining all owners"
                    )
                raw_device.close()  # Drains and unregisters before export release.
            producer.synchronize()
            connector.store_stream.synchronize()
            connector.load_stream.synchronize()
        except BaseException:
            QUARANTINED_OWNERS.append(owners)
            # Preserve the file and fd identities for diagnosis after unknown I/O.
            raise
        else:
            for obj in objects:
                obj.ref_count_down()
            allocator.close()
            if host_fd >= 0:
                os.close(host_fd)
            os.close(fd)
            path.unlink()
