# SPDX-License-Identifier: Apache-2.0
"""Qualify registered MP storage and serde on owned NVMe filesystem files.

Set LMCACHE_RUN_MP_REGISTERED_IO=1 and put pytest's temporary directory on
qualified NVMe-backed XFS or ext4. GPU cases require a working exporter;
LMCACHE_MP_GPU_PROVIDER selects native or explicit experimental vulkan_rm.
"""

# Standard
from pathlib import Path
from typing import Literal
import os
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GPUL1ManagerConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2AdapterConfig,
)
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.utils import single_row_spec
from tests.v1.storage_backend.raw_block_test_utils import make_object_key

pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        os.environ.get("LMCACHE_RUN_MP_REGISTERED_IO") != "1",
        reason="real MP registered I/O requires explicit opt-in",
    ),
]


@pytest.mark.parametrize(
    "medium,serde",
    [
        (medium, codec)
        for medium in ("fixed", "dmabuf")
        for codec in ("none", "fp8", "aesgcm")
    ]
    + [("gpu", "none"), ("gpu", "fp8")],
)
@pytest.mark.parametrize("elements", [4096, 4097])
def test_registered_mp_storage_roundtrip(
    tmp_path: Path,
    medium: Literal["fixed", "dmabuf", "gpu"],
    serde: str,
    elements: int,
) -> None:
    """Prove payload bytes, compressed lengths and completed native fixed I/O."""
    # Third Party
    from lmcache_rust_raw_block_io import RawBlockDevice

    assert not RawBlockDevice.has_fault_injection()
    media = tmp_path / "mp-registered.bin"
    media.write_bytes(bytes(8 << 20))
    memory = L1MemoryManagerConfig(
        4 << 20,
        False,
        shm_name="",
        buffer_registration="dmabuf" if medium == "gpu" else medium,
    )
    l1 = (
        GPUL1ManagerConfig(
            memory,
            tag="gpu",
            buffer_provider=os.environ.get("LMCACHE_MP_GPU_PROVIDER", "native"),
        )
        if medium == "gpu"
        else L1ManagerConfig(memory)
    )
    adapter = RawBlockL2AdapterConfig(
        device_path=str(media),
        slot_bytes=65536,
        meta_total_bytes=1 << 20,
        use_odirect=True,
        io_engine="io_uring",
        meta_enable_periodic=False,
    )
    adapter.affinity_tag = l1.tag
    if serde == "aesgcm":
        master_key = tmp_path / "synthetic-master-key"
        master_key.write_bytes(bytes(range(32)))
        adapter.serde_config = SerdeConfig(
            "aesgcm", {"master_key_path": str(master_key)}
        )
    elif serde == "fp8":
        adapter.serde_config = SerdeConfig("fp8")
    manager = StorageManager(
        StorageManagerConfig(l1, EvictionConfig("noop"), L2AdaptersConfig([adapter]))
    )
    dtype = torch.bfloat16 if serde == "fp8" else torch.uint8
    layout = MemoryLayoutDesc([torch.Size([elements])], [dtype])
    key = make_object_key(17)
    expected = torch.arange(elements, dtype=torch.int64).remainder(16).to(dtype)
    try:
        source = manager.reserve_write([key], layout)[key]
        assert source.tensor is not None
        assert source.tensor.device.type == ("cuda" if medium == "gpu" else "cpu")
        source.tensor.copy_(expected)
        source.zero_padding()
        if medium == "gpu":
            torch_dev.current_stream().synchronize()
        manager.finish_write([key])
        deadline = time.monotonic() + 10
        while manager.l2_adapters()[0][1].get_usage().total_bytes_used == 0:
            assert time.monotonic() < deadline, manager.report_status()
            time.sleep(0.01)
        while manager.report_status()["l1_manager"]["external_io_lease_count"]:
            assert time.monotonic() < deadline, manager.report_status()
            time.sleep(0.01)
        manager.clear()
        assert manager.get_l1_usage()[0] == 0
        handle = manager.submit_prefetch_task(single_row_spec([key], layout))
        assert manager.wait_prefetch_status(handle, 10)
        result = manager.query_prefetch_status(handle)
        assert result is not None and result.hit_cells[0].test(0)
        with manager.read_prefetched_results([key]) as objects:
            assert objects is not None and objects[0].tensor is not None
            assert torch.equal(objects[0].tensor.cpu(), expected)
        io = manager.l2_adapters()[0][1].report_status()["core"]["io_by_request"]
        operations = [
            op
            for op in io["operations"]
            if op["context"]["request_id"].startswith(("store-", "load-"))
        ]
        expected_path = "host_fixed" if medium == "fixed" else "dmabuf_fixed"
        payloads = [op for op in operations if op["path"] == expected_path]
        assert len(payloads) == 2
        assert {op["direction"] for op in payloads} == {"read", "write"}
        logical = elements if serde == "fp8" else elements * dtype.itemsize
        if serde == "aesgcm":
            # AES-GCM framing: version byte, 12-byte IV and 16-byte tag.
            logical += 1 + 12 + 16
        assert all(
            op["complete"] and op["completed_bytes"] == (logical + 4095) // 4096 * 4096
            for op in payloads
        )
        assert io["collection_failures"] == io["dropped_rows"] == 0
        manager.close()
        status = manager.report_status()["l1_manager"]
        assert not status["backing_resources_retained"]
        assert status["external_io_lease_count"] == 0
    finally:
        manager.close()


@pytest.mark.parametrize("operation", ["serialize", "deserialize"])
def test_gpu_codec_fence_failure_retains_arena(tmp_path: Path, operation: str) -> None:
    """Inject loss of a codec fence while retaining real GPU mapping/export owners."""
    # Standard
    from unittest.mock import patch

    # First Party
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
        MockL2Adapter,
        MockL2AdapterConfig,
    )
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper
    from lmcache.v1.distributed.serde import (
        AsyncSerdeProcessor,
        Fp8QuantizationDeserializer,
        Fp8QuantizationSerializer,
    )

    l1 = L1Manager(
        GPUL1ManagerConfig(
            L1MemoryManagerConfig(4 << 20, False, shm_name=""),
            tag="gpu",
            buffer_provider=os.environ.get("LMCACHE_MP_GPU_PROVIDER", "native"),
        )
    )
    key = make_object_key(19)
    _, source = l1.reserve_write(
        [key], [False], MemoryLayoutDesc([torch.Size([4097])], [torch.bfloat16])
    )[key]
    assert source is not None and source.tensor is not None
    source.tensor.fill_(2)
    torch_dev.current_stream().synchronize()
    l1.finish_write([key])
    descriptor = l1.get_l1_memory_desc()
    assert descriptor is not None
    fds = [region.dmabuf_fd for region in descriptor.regions]
    processor = AsyncSerdeProcessor(
        Fp8QuantizationSerializer(), Fp8QuantizationDeserializer()
    )
    wrapper = SerdeL2AdapterWrapper(
        MockL2Adapter(MockL2AdapterConfig(0.01, 10)), processor, l1
    )
    try:
        if operation == "deserialize":
            # Serialize safely so the stored object really contains FP8 bytes.
            task = wrapper.submit_store_task([key], [source])
            deadline = time.monotonic() + 5
            while task not in wrapper.pop_completed_store_outcomes():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            l1.delete([key], force=True)
            _, source = l1.reserve_write(
                [key], [False], MemoryLayoutDesc([torch.Size([4097])], [torch.bfloat16])
            )[key]
            assert source is not None
        with patch(
            "lmcache.v1.distributed.serde.async_processor.torch_dev.current_stream"
        ) as stream:
            stream.return_value.synchronize.side_effect = RuntimeError(
                "injected GPU fence failure"
            )
            task = (
                wrapper.submit_store_task([key], [source])
                if operation == "serialize"
                else wrapper.submit_load_task([key], [source])
            )
            deadline = time.monotonic() + 5
            outcome = None
            while outcome is None:
                assert time.monotonic() < deadline
                outcome = (
                    wrapper.pop_completed_store_outcomes().get(task)
                    if operation == "serialize"
                    else wrapper.query_load_outcome(task)
                )
                time.sleep(0.01)
            assert not outcome.buffers_reusable
        assert l1.report_status()["backing_resources_retained"]
        assert l1.report_status()["external_io_lease_count"] == 0
        assert not wrapper.close_with_outcome().buffers_reusable
        l1.close()
        assert source.is_valid()
        for fd in fds:
            assert fd is not None
            os.fstat(fd)
    finally:
        wrapper.close()
        l1.close()


def test_gpu_storage_fence_failure_retains_arena(tmp_path: Path) -> None:
    """Withhold arena ownership even if native storage closes after a lost GPU fence."""
    # Standard
    from unittest.mock import patch

    # First Party
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
        RawBlockL2Adapter,
    )
    from tests.v1.distributed.test_mp_registration import await_event, media_config

    l1 = L1Manager(
        GPUL1ManagerConfig(
            L1MemoryManagerConfig(4 << 20, False, shm_name=""),
            tag="gpu",
            buffer_provider=os.environ.get("LMCACHE_MP_GPU_PROVIDER", "native"),
        )
    )
    descriptor = l1.get_l1_memory_desc()
    assert descriptor is not None
    adapter = RawBlockL2Adapter(media_config(tmp_path), descriptor)
    key = make_object_key(22)
    _, source = l1.reserve_write(
        [key], [False], MemoryLayoutDesc([torch.Size([4097])], [torch.uint8])
    )[key]
    assert source is not None and source.tensor is not None
    source.tensor.fill_(11)
    torch_dev.current_stream().synchronize()
    l1.finish_write([key])
    lease = l1.acquire_io_lease([key], [source])
    try:
        with patch(
            "lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter.torch_dev.current_stream"
        ) as stream:
            stream.return_value.synchronize.side_effect = RuntimeError(
                "injected GPU fence failure"
            )
            task = adapter.submit_store_task([key], [source])
            await_event(adapter.get_store_event_fd())
            outcome = adapter.pop_completed_store_outcomes()[task]
            assert not outcome.buffers_reusable and not outcome.result.is_successful()
        l1.finish_read([key])
        l1.release_io_lease(lease, buffers_reusable=outcome.buffers_reusable)
        assert not adapter.close_with_outcome().buffers_reusable
        l1.close()
        assert source.is_valid()
        for region in descriptor.regions:
            assert region.dmabuf_fd is not None
            os.fstat(region.dmabuf_fd)
    finally:
        adapter.close()
        l1.close()


@pytest.mark.parametrize("operation", ["write", "read"])
@pytest.mark.parametrize("reusable", [True, False])
def test_gpu_transfer_lease_prevents_recycling(
    tmp_path: Path, operation: str, reusable: bool
) -> None:
    """Keep the exact GPU allocation alive through forced deletion and shutdown."""
    manager = StorageManager(
        StorageManagerConfig(
            GPUL1ManagerConfig(
                L1MemoryManagerConfig(4 << 20, False, shm_name=""),
                tag="gpu",
                buffer_provider=os.environ.get("LMCACHE_MP_GPU_PROVIDER", "native"),
            ),
            EvictionConfig("noop"),
        )
    )
    key = make_object_key(24)
    try:
        source = manager.reserve_write(
            [key], MemoryLayoutDesc([torch.Size([4097])], [torch.uint8])
        )[key]
        assert source.tensor is not None
        source.tensor.fill_(11)
        torch_dev.current_stream().synchronize()
        if operation == "read":
            manager.finish_write([key])
            leases = manager.acquire_read_transfer_leases({key: source})
        else:
            leases = manager.acquire_write_transfer_leases({key: source})
        assert leases
        assert manager.delete_l1_keys([key], force=True) == (0, 1)
        assert source.is_valid()
        assert manager.get_l1_usage()[0] > 0
        if reusable:
            manager.release_transfer_leases(leases)
            assert manager.delete_l1_keys([key], force=True) == (1, 0)
        else:
            manager.retain_transfer_leases(leases)
        assert source.is_valid() is not reusable
        assert manager.report_status()["l1_manager"]["external_io_lease_count"] == 0
        assert (
            manager.report_status()["l1_manager"]["backing_resources_retained"]
            is not reusable
        )
        manager.close()
        assert source.is_valid() is not reusable
    finally:
        manager.close()
