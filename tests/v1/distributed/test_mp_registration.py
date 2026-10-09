# SPDX-License-Identifier: Apache-2.0
"""Registered MP payload I/O and affinity-arena ownership on setup failure.

Set LMCACHE_RUN_HOST_DMABUF_IO=1 and place pytest's temporary directory on
qualified NVMe-backed XFS or ext4 for real I/O. Opt-in requires the ordinary
extension, /dev/udmabuf and the matching DMA-BUF io_uring kernel ABI.
"""

# Standard
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch
import os
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    DRAML1ManagerConfig,
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    get_arg_parser,
    parse_args_to_config,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInitializationError
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import TensorMemoryObj
from lmcache.v1.storage_backend.raw_block.core import (
    NativeQuiescence,
    RawBlockCloseOutcome,
    RawBlockCore,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    make_memory_obj,
    make_object_key,
)

pytestmark = pytest.mark.no_shared_allocator

requires_registered_io = pytest.mark.skipif(
    os.environ.get("LMCACHE_RUN_HOST_DMABUF_IO") != "1",
    reason="real registered MP I/O requires explicit opt-in",
)


def media_config(tmp_path: Path, **overrides: Any) -> RawBlockL2AdapterConfig:
    """Create an initialized, task-owned filesystem target."""
    path = tmp_path / "registered.bin"
    path.write_bytes(bytes(8 << 20))
    values: dict[str, Any] = dict(
        device_path=str(path),
        slot_bytes=65536,
        meta_total_bytes=1 << 20,
        io_engine="io_uring",
        use_odirect=True,
        meta_enable_periodic=False,
    )
    values.update(overrides)
    return RawBlockL2AdapterConfig(**values)


def await_event(fd: int) -> None:
    """Wait for a public adapter completion notification."""
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    assert poll.poll(5000), "adapter completion timed out"


@pytest.mark.parametrize("mode", ["fixed", "dmabuf"])
@pytest.mark.parametrize("length", [4096, 4097])
@requires_registered_io
def test_registered_payload_roundtrip(
    tmp_path: Path,
    mode: Literal["fixed", "dmabuf"],
    length: int,
) -> None:
    """Prove native fixed reads and writes, including padding."""
    # Third Party
    from lmcache_rust_raw_block_io import RawBlockDevice

    assert not RawBlockDevice.has_fault_injection()
    l1 = L1MemoryManager(
        L1MemoryManagerConfig(
            4 << 20,
            False,
            shm_name="",
            buffer_registration=mode,
        )
    )
    adapter = RawBlockL2Adapter(media_config(tmp_path), l1.get_l1_memory_desc())
    error, objects = l1.allocate(
        MemoryLayoutDesc([torch.Size([length])], [torch.uint8]), 3
    )
    assert error == L1Error.SUCCESS
    source, target, neighbor = objects
    source_tensor, target_tensor, neighbor_tensor = (
        source.tensor,
        target.tensor,
        neighbor.tensor,
    )
    assert source_tensor is not None
    assert target_tensor is not None
    assert neighbor_tensor is not None
    expected = torch.arange(length, dtype=torch.int64).remainder(251).to(torch.uint8)
    source_tensor.copy_(expected)
    target_tensor.fill_(0)
    neighbor_tensor.fill_(137)
    key = make_object_key(77)
    try:
        outsider = make_memory_obj(bytes(length))
        raw = source.raw_tensor
        assert raw is not None
        displaced = TensorMemoryObj(
            raw[1:],
            source.metadata,
            parent_allocator=None,
            physical_data=source.physical_tensor,
        )
        for submit in (adapter.submit_store_task, adapter.submit_load_task):
            with pytest.raises(ValueError, match="registered L1 region"):
                submit([key], [outsider])
            with pytest.raises(ValueError, match="registered L1 region"):
                submit([key], [displaced])
        task = adapter.submit_store_task([key], [source])
        await_event(adapter.get_store_event_fd())
        store = adapter.pop_completed_store_outcomes()[task]
        assert store.buffers_reusable and store.result.is_successful()
        task = adapter.submit_lookup_and_lock_task([key], {0: MemoryLayoutDesc([], [])})
        await_event(adapter.get_lookup_and_lock_event_fd())
        hits = adapter.query_lookup_and_lock_result(task)
        assert hits is not None and hits.test(0)
        task = adapter.submit_load_task([key], [target])
        await_event(adapter.get_load_event_fd())
        load = adapter.query_load_outcome(task)
        assert load is not None and load.buffers_reusable and load.bitmap.test(0)
        assert torch.equal(target_tensor, expected)
        assert torch.all(neighbor_tensor == 137)
        status = adapter.report_status()["core"]
        assert status["buffer_registration_mode"] == (
            "classic" if mode == "fixed" else "dmabuf"
        )
        path = "host_fixed" if mode == "fixed" else "dmabuf_fixed"
        operations = [
            op
            for op in status["io_by_request"]["operations"]
            if op["context"]["request_id"].startswith(("store-", "load-"))
        ]
        assert all(op["complete"] for op in operations)
        for direction in ("read", "write"):
            fixed = [
                op
                for op in operations
                if op["direction"] == direction and op["path"] == path
            ]
            assert len(fixed) == 1
            assert fixed[0]["completed_bytes"] == (length + 4095) // 4096 * 4096
        assert len(operations) == 3  # slot header, payload write, payload read
        adapter.submit_unlock([key])
    finally:
        assert adapter.close_with_outcome().buffers_reusable
        assert adapter.close_with_outcome().buffers_reusable
        l1.free(objects)
        l1.close()


@pytest.mark.parametrize("reusable", [False, True])
@requires_registered_io
def test_factory_failure_retains_only_an_unresolved_l1(
    tmp_path: Path,
    reusable: bool,
) -> None:
    """Inject setup/cleanup verdicts and check the factory's actual L1 owner."""
    untouched = L1ManagerConfig(
        L1MemoryManagerConfig(
            4 << 20,
            False,
            shm_name="",
            buffer_registration="fixed",
        )
    )
    affinity = L1ManagerConfig(
        L1MemoryManagerConfig(
            4 << 20,
            False,
            shm_name="",
            buffer_registration="dmabuf",
        ),
        tag="io",
    )
    adapter = media_config(tmp_path)
    adapter.affinity_tag = "io"
    config = StorageManagerConfig(
        untouched,
        EvictionConfig("noop"),
        L2AdaptersConfig([adapter]),
        l1_manager_configs=[untouched, affinity],
    )
    owners: list[L1Manager] = []

    def owner_factory(config: L1ManagerConfig) -> L1Manager:
        owner = L1Manager(config)
        owners.append(owner)
        return owner

    verdict = RawBlockCloseOutcome(
        NativeQuiescence.PROVEN if reusable else NativeQuiescence.RETAINED,
        not reusable,
    )
    with (
        patch(
            "lmcache.v1.distributed.storage_manager.L1Manager",
            side_effect=owner_factory,
        ),
        patch.object(
            RawBlockCore,
            "register_fixed_buffer_regions",
            side_effect=RuntimeError("registration refused"),
        ),
        patch.object(RawBlockCore, "close", return_value=verdict),
        pytest.raises(RuntimeError if reusable else L2AdapterInitializationError),
    ):
        StorageManager(config)
    assert not owners[0].report_status()["backing_resources_retained"]
    assert owners[1].report_status()["backing_resources_retained"] is not reusable


@pytest.mark.parametrize("mode", ["fixed", "dmabuf"])
def test_dram_json_selects_registration(mode: str) -> None:
    config = DRAML1ManagerConfig.from_dict(
        dict(
            type="DRAM",
            tag="_default",
            size_gb=0.004,
            use_lazy=False,
            buffer_registration=mode,
        ),
        EvictionConfig("noop"),
    )
    assert config.memory_config.buffer_registration == mode


def test_cli_selects_fixed_registration() -> None:
    args = get_arg_parser().parse_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "noop",
            "--no-l1-use-lazy",
            "--l1-buffer-registration",
            "fixed",
        ]
    )
    assert (
        parse_args_to_config(args).l1_manager_config.memory_config.buffer_registration
        == "fixed"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"io_engine": "posix"},
        {"use_uring_cmd": True},
        {"use_odirect": False},
        {"enable_zero_copy": False},
    ],
)
def test_config_rejects_incompatible_registration_before_allocation(
    tmp_path: Path,
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="ordinary O_DIRECT"):
        StorageManagerConfig(
            L1ManagerConfig(
                L1MemoryManagerConfig(
                    4 << 20,
                    False,
                    shm_name="",
                    buffer_registration="fixed",
                )
            ),
            EvictionConfig("noop"),
            L2AdaptersConfig([media_config(tmp_path, **overrides)]),
        )
