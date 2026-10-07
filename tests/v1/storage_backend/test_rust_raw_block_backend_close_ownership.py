# SPDX-License-Identifier: Apache-2.0
"""Retain registered backing until native teardown proves it is safe."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import asyncio
import gc
import threading
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.plugins import rust_raw_block_backend as plugin
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)
from lmcache.v1.storage_backend.raw_block import NativeQuiescence, RawBlockCloseOutcome

_METADATA = LMCacheMetadata(
    model_name="test_model",
    world_size=1,
    local_world_size=1,
    worker_id=0,
    local_worker_id=0,
    kv_dtype=torch.bfloat16,
    kv_shape=(4, 2, 256, 8, 128),
)


def _config(dev_path: str) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=False,
        max_local_cpu_size=0,
        lmcache_instance_id="test_raw_block_close_ownership",
    )
    config.storage_plugins = []
    config.extra_config = {
        "rust_raw_block.device_path": dev_path,
        "rust_raw_block.block_align": 4096,
        "rust_raw_block.header_bytes": 4096,
        "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
        "rust_raw_block.meta_enable_periodic": False,
        "rust_raw_block.io_engine": "io_uring",
    }
    return config


@pytest.fixture
def loop_in_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="test-loop", daemon=True)
    t.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        loop.close()


@pytest.fixture(autouse=True)
def _keep_the_retention_list_to_this_test():
    """The retention list outlives a process on purpose, not a test."""
    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    yield
    del plugin._RETAINED_AFTER_UNKNOWN_OUTCOME[before:]


@pytest.mark.no_shared_allocator
def test_optional_cpu_registration_failure_retains_poisoned_backing(
    monkeypatch: pytest.MonkeyPatch,
    loop_in_thread: asyncio.AbstractEventLoop,
    tmp_path: Path,
) -> None:
    """Optional DMA-BUF registration cannot fall back after failed cleanup."""

    class CPUOwner:
        def __init__(self) -> None:
            self.retained = False

        def get_full_chunk_size_bytes(self) -> int:
            return 4096

        def get_memory_allocator(self) -> CPUOwner:
            return self

        def retain_backing_resources(self) -> None:
            self.retained = True

    class RegistrationCore:
        io_engine = "io_uring"
        use_uring_cmd = False
        require_dmabuf_registration = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def register_fixed_buffers_from_allocator(self, allocator: Any) -> None:
            raise RuntimeError("partial registration cleanup failed")

        def is_poisoned(self) -> bool:
            return True

        def close(self) -> RawBlockCloseOutcome:
            return RawBlockCloseOutcome(
                quiescence=NativeQuiescence.RETAINED,
                poisoned=True,
                final_checkpoint_written=False,
                reason="registered exports could not be released",
            )

    monkeypatch.setattr(plugin, "RawBlockCore", RegistrationCore)
    path = tmp_path / "registration.bin"
    with path.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024)
    config = _config(str(path))
    config.extra_config["rust_raw_block.gpu_buffer_bytes"] = 0
    cpu = CPUOwner()
    owner = weakref.ref(cpu)
    with pytest.raises(RuntimeError, match="partial registration cleanup failed"):
        RustRawBlockBackend(
            config=config,
            metadata=_METADATA,
            local_cpu_backend=cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
    assert cpu.retained
    del cpu
    gc.collect()
    assert owner() is not None
