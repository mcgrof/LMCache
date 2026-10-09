# SPDX-License-Identifier: Apache-2.0
"""Verify MP raw-block ownership results and ordinary native io_uring I/O."""

# Standard
from pathlib import Path
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.storage_backend.raw_block import (
    NativeQuiescence,
    RawBlockCloseOutcome,
    RawBlockCore,
    RawBlockPutManyResult,
)
from tests.v1.distributed.test_raw_block_l2_adapter import (
    _create_memory_obj,
    _create_object_key,
    _wait_event_fd,
)


@pytest.mark.parametrize("operation", ["store", "load"])
@pytest.mark.parametrize("poisoned", [False, True])
def test_completed_failure_does_not_imply_buffer_ownership(
    operation: str,
    poisoned: bool,
) -> None:
    """A simulated core verdict must survive task and legacy API boundaries."""
    config = RawBlockL2AdapterConfig(
        device_path="/unused", slot_bytes=65536, io_engine="io_uring"
    )
    with patch(
        "lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter.RawBlockCore",
        autospec=RawBlockCore,
    ) as core_class:
        core = core_class.return_value
        core.report_status.return_value = {}
        core.snapshot_indexed_keys.return_value = []
        core.is_poisoned.return_value = poisoned
        core.put_many.return_value = RawBlockPutManyResult([False], [])
        core.load_many_into.return_value = [False]
        core.close.return_value = RawBlockCloseOutcome(
            NativeQuiescence.RETAINED if poisoned else NativeQuiescence.PROVEN,
            poisoned,
        )
        adapter = RawBlockL2Adapter(config)
        key, obj = _create_object_key(1), _create_memory_obj()
        try:
            if operation == "store":
                task = adapter.submit_store_task([key], [obj])
                assert _wait_event_fd(adapter.get_store_event_fd())
                if poisoned:
                    with pytest.raises(RuntimeError, match="ownership outcomes"):
                        adapter.pop_completed_store_tasks()
                outcomes = adapter.pop_completed_store_outcomes()
                assert outcomes[task].buffers_reusable is not poisoned
                assert not outcomes[task].result.is_successful()
                assert adapter.pop_completed_store_outcomes() == {}
            else:
                task = adapter.submit_load_task([key], [obj])
                assert _wait_event_fd(adapter.get_load_event_fd())
                if poisoned:
                    with pytest.raises(RuntimeError, match="ownership outcomes"):
                        adapter.query_load_result(task)
                outcome = adapter.query_load_outcome(task)
                assert outcome is not None and outcome.buffers_reusable is not poisoned
                assert not outcome.bitmap.test(0)
                assert adapter.query_load_outcome(task) is None
            closed = adapter.close_with_outcome()
            assert closed.buffers_reusable is not poisoned
            assert adapter.close_with_outcome() is closed
            core.close.assert_called_once()
            with pytest.raises(RuntimeError, match="closed"):
                adapter.submit_load_task([key], [obj])
        finally:
            adapter.close()


@pytest.mark.parametrize("io_engine", ["posix", "io_uring"])
@pytest.mark.parametrize("use_odirect", [False, True])
def test_native_store_load_and_shutdown_return_ownership(
    tmp_path: Path,
    io_engine: str,
    use_odirect: bool,
) -> None:
    """Use the ordinary extension on an initialized, owned regular file."""
    native = pytest.importorskip("lmcache_rust_raw_block_io")
    assert not native.RawBlockDevice.has_fault_injection()
    path = tmp_path / "media.bin"
    path.write_bytes(bytes(8 << 20))
    config = RawBlockL2AdapterConfig(
        device_path=str(path),
        slot_bytes=65536,
        use_odirect=use_odirect,
        io_engine=io_engine,
        meta_total_bytes=1 << 20,
        meta_enable_periodic=False,
    )
    adapter = RawBlockL2Adapter(config)
    key = _create_object_key(2)
    source = _create_memory_obj(fill_value=42)
    target = _create_memory_obj(fill_value=0)
    try:
        store = adapter.submit_store_task([key], [source])
        assert _wait_event_fd(adapter.get_store_event_fd())
        outcome = adapter.pop_completed_store_outcomes()[store]
        assert outcome.buffers_reusable and outcome.result.is_successful()
        lookup = adapter.submit_lookup_and_lock_task(
            [key], {0: MemoryLayoutDesc([], [])}
        )
        assert _wait_event_fd(adapter.get_lookup_and_lock_event_fd())
        bitmap = adapter.query_lookup_and_lock_result(lookup)
        assert bitmap is not None and bitmap.test(0)
        load = adapter.submit_load_task([key], [target])
        assert _wait_event_fd(adapter.get_load_event_fd())
        loaded = adapter.query_load_outcome(load)
        assert loaded is not None and loaded.buffers_reusable and loaded.bitmap.test(0)
        assert source.tensor is not None and target.tensor is not None
        assert source.tensor.equal(target.tensor)
        adapter.submit_unlock([key])
        closed = adapter.close_with_outcome()
        assert closed.buffers_reusable
        assert adapter.close_with_outcome() is closed
    finally:
        adapter.close()
