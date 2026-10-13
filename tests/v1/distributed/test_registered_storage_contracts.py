# SPDX-License-Identifier: Apache-2.0
"""Exercise registered storage shutdown and real short-read ownership.

Set LMCACHE_RUN_HOST_DMABUF_IO=1 and place pytest's temporary directory on
an owned XFS or ext4 filesystem with the matching DMA-BUF io_uring ABI.
"""

# Standard
from pathlib import Path
from typing import Literal

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import RawBlockL2Adapter
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.platform import consume_fd
from tests.v1.distributed.test_mp_registration import (
    await_event,
    media_config,
    requires_registered_io,
)
from tests.v1.storage_backend.raw_block_test_utils import make_object_key

pytestmark = [pytest.mark.no_shared_allocator, requires_registered_io]


@pytest.mark.parametrize("mode", ["fixed", "dmabuf"])
@pytest.mark.parametrize("queue_depth", [2, 8])
def test_registered_shutdown_drains_queued_stores(
    tmp_path: Path, mode: Literal["fixed", "dmabuf"], queue_depth: int
) -> None:
    """Close with more submitted payloads than the ring can hold at once."""
    l1 = L1MemoryManager(
        L1MemoryManagerConfig(4 << 20, False, shm_name="", buffer_registration=mode)
    )
    workers = 4
    config = media_config(
        tmp_path, iouring_queue_depth=queue_depth, num_store_workers=workers
    )
    adapter = RawBlockL2Adapter(config, l1.get_l1_memory_desc())
    layout = MemoryLayoutDesc([torch.Size([config.block_align + 1])], [torch.uint8])
    # Keep more work pending than the ring can accept for each worker.
    error, objects = l1.allocate(layout, 2 * workers * queue_depth)
    assert error == L1Error.SUCCESS
    keys = [make_object_key(index) for index in range(len(objects))]
    try:
        for index, obj in enumerate(objects):
            assert obj.tensor is not None
            obj.tensor.fill_(index + 1)
        tasks = [
            adapter.submit_store_task([key], [obj])
            for key, obj in zip(keys, objects, strict=True)
        ]
        assert adapter.report_status()["store_inflight_task_count"] > 0
        closed = adapter.close_with_outcome()
        assert closed.buffers_reusable
        outcomes = adapter.pop_completed_store_outcomes()
        assert set(outcomes) == set(tasks)
        assert all(
            outcome.buffers_reusable and outcome.result.is_successful()
            for outcome in outcomes.values()
        )
        journal = adapter.report_status()["core"]["io_by_request"]
        payloads = [
            operation
            for operation in journal["operations"]
            if operation["context"]["request_id"].startswith("store-")
            and operation["path"] in ("host_fixed", "dmabuf_fixed")
        ]
        assert len(payloads) == len(objects)
        assert all(
            operation["complete"]
            and operation["path"]
            == ("host_fixed" if mode == "fixed" else "dmabuf_fixed")
            and operation["completed_bytes"] == 2 * config.block_align
            for operation in payloads
        )
        assert journal["dropped_rows"] == journal["collection_failures"] == 0
        reopened = RawBlockL2Adapter(config, l1.get_l1_memory_desc())
        try:
            for index, (key, obj) in enumerate(zip(keys, objects, strict=True)):
                assert obj.tensor is not None
                obj.tensor.fill_(173)
                task = reopened.submit_load_task([key], [obj])
                await_event(reopened.get_load_event_fd())
                consume_fd(reopened.get_load_event_fd())
                outcome = reopened.query_load_outcome(task)
                assert outcome is not None
                assert outcome.buffers_reusable and outcome.bitmap.test(0)
                assert torch.all(obj.tensor == index + 1)
        finally:
            assert reopened.close_with_outcome().buffers_reusable
    finally:
        assert adapter.close_with_outcome().buffers_reusable
        l1.free(objects)
        l1.close()


@pytest.mark.parametrize("mode", ["fixed", "dmabuf"])
def test_registered_short_read_returns_ownership_without_a_hit(
    tmp_path: Path, mode: Literal["fixed", "dmabuf"]
) -> None:
    """Truncate completed data and require a known failure from the real ring."""
    l1 = L1MemoryManager(
        L1MemoryManagerConfig(4 << 20, False, shm_name="", buffer_registration=mode)
    )
    config = media_config(tmp_path)
    adapter = RawBlockL2Adapter(config, l1.get_l1_memory_desc())
    error, objects = l1.allocate(
        MemoryLayoutDesc([torch.Size([config.block_align + 1])], [torch.uint8]), 1
    )
    assert error == L1Error.SUCCESS
    obj = objects[0]
    assert obj.tensor is not None
    key = make_object_key(77)
    try:
        obj.tensor.fill_(42)
        task = adapter.submit_store_task([key], [obj])
        await_event(adapter.get_store_event_fd())
        stored = adapter.pop_completed_store_outcomes()[task]
        assert stored.buffers_reusable and stored.result.is_successful()
        with Path(config.device_path).open("r+b") as media:
            media.truncate(0)
        obj.tensor.fill_(173)
        task = adapter.submit_load_task([key], [obj])
        await_event(adapter.get_load_event_fd())
        loaded = adapter.query_load_outcome(task)
        assert loaded is not None
        assert loaded.buffers_reusable and not loaded.bitmap.test(0)
        assert adapter.report_status()["core"]["is_healthy"]
    finally:
        assert adapter.close_with_outcome().buffers_reusable
        l1.free(objects)
        l1.close()
