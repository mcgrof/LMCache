# SPDX-License-Identifier: Apache-2.0
"""Selection regressions and explicit hardware owner/lifetime qualification."""

# Standard
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
import asyncio
import errno
import fcntl
import gc
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)
from lmcache.v1.storage_backend.raw_block import RawBlockCore


def test_native_default_does_not_load_vulkan(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ordinary CPU/native pool must not import the optional extension."""

    def forbidden(name: str) -> None:
        pytest.fail(f"unexpected optional import: {name}")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    pool = GPUMemoryAllocator(8192, device="cpu")
    assert pool.buffer_provider == "native"
    obj = pool.allocate(torch.Size([16]), torch.uint8)
    assert obj is not None
    obj.ref_count_down()
    pool.close()


@pytest.mark.parametrize("provider", ["auto", "vulkan", "", "cpu"])
def test_unknown_provider_rejected(provider: str) -> None:
    """Only implemented policies may be selected; no implicit fallback."""
    with pytest.raises(ValueError, match="native or vulkan_rm"):
        GPUMemoryAllocator(4096, device="cpu", buffer_provider=provider)


@pytest.mark.parametrize("provider,size", [("auto", 4096), ("vulkan_rm", 0)])
def test_backend_rejects_policy_before_opening_storage(
    provider: str, size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid selection must fail before native storage initialization."""

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid provider reached storage initialization")

    monkeypatch.setattr(RawBlockCore, "__init__", forbidden)
    config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "rust_raw_block.gpu_buffer_provider": provider,
            "rust_raw_block.gpu_buffer_bytes": size,
        }
    )
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ValueError, match="gpu_buffer_provider"):
            RustRawBlockBackend(config=config, loop=loop)
    finally:
        loop.close()


def test_unavailable_cuda_is_not_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit Vulkan/RM pool cannot become a CPU allocation."""
    monkeypatch.setattr(torch_dev, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="available NVIDIA CUDA GPU"):
        GPUMemoryAllocator(4096, buffer_provider="vulkan_rm")


@pytest.mark.parametrize("size", [0, (1 << 30) + 1])
def test_provider_limit_before_helper_load(
    size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject excessive aligned capacity before importing or allocating."""
    monkeypatch.setattr(torch_dev, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(ValueError, match="1 GiB"):
        GPUMemoryAllocator(size, device="cuda:0", buffer_provider="vulkan_rm")


def test_missing_helper_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing optional build is an error, not native/CPU substitution."""
    monkeypatch.setattr(torch_dev, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setitem(sys.modules, "lmcache._vulkan_rm", None)
    with pytest.raises(RuntimeError, match="BUILD_WITH_VULKAN_RM=1"):
        GPUMemoryAllocator(4096, device="cuda:0", buffer_provider="vulkan_rm")


def test_provider_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported ABI/identity or allocation errors never trigger retry."""
    monkeypatch.setattr(torch_dev, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    calls: list[tuple[int, int]] = []

    def allocate(size: int, ordinal: int) -> None:
        calls.append((size, ordinal))
        raise RuntimeError("unsupported NVIDIA RM ABI")

    monkeypatch.setitem(
        sys.modules, "lmcache._vulkan_rm", SimpleNamespace(allocate=allocate)
    )
    with pytest.raises(RuntimeError, match="unsupported NVIDIA RM ABI"):
        GPUMemoryAllocator(4096, device="cuda:3", buffer_provider="vulkan_rm")
    assert calls == [(4096, 3)]


@pytest.fixture
def provider_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Run a bounded background event loop for asynchronous backend writes."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    loop.call_soon(ready.set)
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        assert ready.wait(5), "backend test loop did not start"
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive(), "backend test loop did not stop"
        loop.close()


@pytest.mark.skipif(
    os.environ.get("LMCACHE_RUN_VULKAN_RM") != "1",
    reason="explicit GPU opt-in required",
)
@pytest.mark.no_shared_allocator
def test_native_owner_survives_delayed_alias() -> None:
    """CUDA storage aliases retain the whole owner, including its export FD."""
    helper = importlib.import_module("lmcache._vulkan_rm")
    for generation in range(8):
        tensor, borrowed_fd, identity = helper.allocate(2 << 20, 0)
        assert "exporter=nv_dmabuf" in identity
        alias = tensor[4096:8192]
        alias.fill_(generation + 17)
        del tensor
        gc.collect()
        fcntl.fcntl(borrowed_fd, fcntl.F_GETFD)
        assert torch.equal(
            alias.cpu(), torch.full((4096,), generation + 17, dtype=torch.uint8)
        )
        del alias
        gc.collect()
        with pytest.raises(OSError) as exc:
            fcntl.fcntl(borrowed_fd, fcntl.F_GETFD)
        assert exc.value.errno == errno.EBADF


@pytest.mark.skipif(
    os.environ.get("LMCACHE_RUN_VULKAN_RM") != "1",
    reason="explicit GPU opt-in required",
)
def test_provider_initializes_fresh_cuda_context() -> None:
    """The first allocation must not require a prior ordinary Torch tensor."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; from lmcache.v1.memory_allocators.gpu_memory_allocator "
            "import GPUMemoryAllocator; "
            "pool = GPUMemoryAllocator(4096, device='cuda:0', "
            "buffer_provider='vulkan_rm'); pool.tensor.fill_(37); "
            "assert torch.all(pool.tensor.cpu() == 37).item(); pool.close()",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    os.environ.get("LMCACHE_RUN_VULKAN_RM") != "1",
    reason="explicit GPU opt-in required",
)
@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("size", [64 << 20, 256 << 20])
def test_large_provider_pool_extent(size: int) -> None:
    """Larger exports preserve exact extents and nonzero-offset CUDA views."""
    helper = importlib.import_module("lmcache._vulkan_rm")
    tensor, borrowed_fd, identity = helper.allocate(size, 0)
    assert f" bytes={size} " in identity
    with open(f"/proc/self/fdinfo/{borrowed_fd}") as fdinfo:
        fields = dict(line.strip().split(":", 1) for line in fdinfo if ":" in line)
    assert int(fields["size"]) == size
    assert fields["exp_name"].strip() == "nv_dmabuf"
    tensor.fill_(37)
    alias = tensor[-4096:]
    alias.fill_(91)
    assert torch.all(tensor[:-4096] == 37).item()
    del tensor
    gc.collect()
    assert torch.equal(alias.cpu(), torch.full((4096,), 91, dtype=torch.uint8))
    fcntl.fcntl(borrowed_fd, fcntl.F_GETFD)
    del alias
    gc.collect()
    with pytest.raises(OSError) as exc:
        fcntl.fcntl(borrowed_fd, fcntl.F_GETFD)
    assert exc.value.errno == errno.EBADF


@pytest.mark.skipif(
    os.environ.get("LMCACHE_RUN_VULKAN_RM") != "1",
    reason="explicit GPU opt-in required",
)
@pytest.mark.no_shared_allocator
def test_paged_provider_close_keeps_views() -> None:
    """Close stops admission without invalidating already-owned tensor views."""
    shape = torch.Size([2, 1, 16, 64])
    pool = GPUMemoryAllocator(
        8192,
        device="cuda:0",
        buffer_provider="vulkan_rm",
        use_paging=True,
        shapes=[shape],
        dtypes=[torch.float16],
        fmt=MemoryFormat.KV_2LTD,
    )
    first = pool.allocate(shape, torch.float16)
    second = pool.allocate(shape, torch.float16)
    assert first is not None and second is not None
    regions = pool.get_paged_dmabuf_regions()
    buffers = pool.get_paged_buffers()
    assert regions and buffers
    assert regions[0] == regions[1]
    assert buffers[1].data_ptr() - regions[1][1] == 4096
    alias = second.tensor
    assert alias is not None
    alias.fill_(23)
    fd = regions[0][0]
    first.ref_count_down()
    second.ref_count_down()
    pool.close()
    pool.close()
    assert pool.allocate(shape, torch.float16) is None
    with pytest.raises(RuntimeError, match="closed"):
        pool.get_paged_dmabuf_regions()
    with pytest.raises(OSError):
        fcntl.fcntl(fd, fcntl.F_GETFD)
    del first, second, buffers, pool
    gc.collect()
    assert torch.equal(alias.cpu(), torch.full(shape, 23, dtype=torch.float16))


@pytest.mark.skipif(
    os.environ.get("LMCACHE_RUN_VULKAN_RM") != "1",
    reason="explicit GPU opt-in required",
)
@pytest.mark.no_shared_allocator
def test_backend_provider_store_retrieve(
    provider_loop: asyncio.AbstractEventLoop,
) -> None:
    """Select through public configuration and store/retrieve without CPU tier.

    Only an exclusive new file under the explicitly supplied scratch directory
    is written. A failed trial preserves the file for ownership investigation.
    This is an ordinary backend test, not model serving or P/D qualification.
    """
    directory_setting = os.environ.get("LMCACHE_DMABUF_TEST_DIR", "")
    if not directory_setting:
        raise ValueError("set LMCACHE_DMABUF_TEST_DIR to a qualified scratch directory")
    directory = Path(directory_setting).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("LMCACHE_DMABUF_TEST_DIR must be a directory")
    fd, filename = tempfile.mkstemp(prefix="lmcache-vulkan-rm-backend-", dir=directory)
    backend = None
    completed = False
    try:
        file_bytes = 8 << 20
        os.posix_fallocate(fd, 0, file_bytes)
        assert os.pwrite(fd, bytes(file_bytes), 0) == file_bytes
        os.fdatasync(fd)
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0,
            extra_config={
                "rust_raw_block.device_path": filename,
                "rust_raw_block.block_align": 4096,
                "rust_raw_block.header_bytes": 4096,
                "rust_raw_block.meta_total_bytes": 4 << 20,
                "rust_raw_block.meta_enable_periodic": False,
                "rust_raw_block.io_engine": "io_uring",
                "rust_raw_block.use_odirect": True,
                # The strict option requires a block device. GPU endpoint
                # setup independently refuses failed DMA-BUF registration.
                "rust_raw_block.gpu_buffer_bytes": 64 << 10,
                "rust_raw_block.gpu_buffer_device": "cuda:0",
                "rust_raw_block.gpu_buffer_provider": "vulkan_rm",
            },
        )
        metadata = LMCacheMetadata(
            model_name="vulkan-rm-backend-regression",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float16,
            kv_shape=(1, 2, 16, 1, 64),
            chunk_size=16,
        )
        backend = RustRawBlockBackend(
            config=config, metadata=metadata, loop=provider_loop, dst_device="cuda:0"
        )
        assert backend.is_gpu_endpoint
        assert backend.get_memory_allocator().buffer_provider == "vulkan_rm"
        shape = torch.Size([2, 1, 16, 64])
        for generation in (3, 17, 53, 97):
            key = CacheEngineKey(metadata.model_name, 1, 0, generation, torch.float16)
            obj = backend.allocate(shape, torch.float16)
            assert obj is not None and obj.tensor is not None
            obj.tensor.fill_(generation)
            torch_dev.current_stream().synchronize()
            futures = backend.batched_submit_put_task([key], [obj])
            assert futures is not None
            for future in futures:
                future.result(timeout=10)
            obj.ref_count_down()
            out = backend.get_blocking(key)
            assert out is not None and out.tensor is not None and out.tensor.is_cuda
            assert torch.equal(
                out.tensor.cpu(), torch.full(shape, generation, dtype=torch.float16)
            )
            out.ref_count_down()
        status = backend.report_status()
        core = status["core"]
        assert core["buffer_registration_mode"] == "dmabuf"
        assert core["inflight_io_count"] == core["quarantined_slot_count"] == 0
        assert not core["poisoned"]
        for direction in ("payload_writes", "payload_reads"):
            assert core[direction]["completed_operations"] == 4
            assert core[direction]["completed_padded_bytes"] == 4 * 4096
        print("backend_provider_receipt=" + json.dumps(status, sort_keys=True))
        backend.close()
        assert not backend.report_status()["core"]["poisoned"]
        backend = None
        completed = True
    finally:
        if backend is not None:
            backend.close()
        os.close(fd)
        if completed:
            os.unlink(filename)
