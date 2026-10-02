# SPDX-License-Identifier: Apache-2.0
"""Keep split-tier clear's adapter alive until physical cleanup finishes."""

# Standard
from pathlib import Path
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_manager import StorageManager


@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("operation", ["remove", "close"])
def test_clear_drains_before_adapter_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Removal and service shutdown wait for an already-started clear."""
    fs_config = FSL2AdapterConfig(base_path=str(tmp_path))
    fs_config.serde_config = SerdeConfig(type="asym_k16_v8_v_only")
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4 << 20,
                    use_lazy=True,
                    init_size_in_bytes=1 << 20,
                )
            ),
            eviction_config=EvictionConfig(eviction_policy="noop"),
            l2_adapter_config=L2AdaptersConfig(adapters=[fs_config]),
        )
    )
    descriptor, adapter = manager.l2_adapters()[0]
    logical = ObjectKey(chunk_hash=b"lifecycle", model_name="test", kv_rank=0)
    generation = manager.split_tier_manifest.register_pending(logical)
    manager.split_tier_manifest.mark_complete(logical, generation)
    entered_delete = threading.Event()
    release_delete = threading.Event()
    closed = threading.Event()
    errors: list[BaseException] = []
    original_close = adapter.close

    def blocked_delete(_keys: list[ObjectKey]) -> None:
        entered_delete.set()
        assert release_delete.wait(5), "test never released physical cleanup"
        assert not closed.is_set(), "adapter closed during its physical cleanup"

    def record_close() -> None:
        closed.set()
        original_close()

    def clear() -> None:
        try:
            manager.clear()
        except BaseException as error:
            errors.append(error)

    def shutdown() -> None:
        try:
            if operation == "remove":
                manager.delete_l2_adapter(descriptor.index)
            else:
                manager.close()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(adapter, "delete", blocked_delete)
    monkeypatch.setattr(adapter, "close", record_close)
    clear_thread = threading.Thread(target=clear)
    shutdown_thread = threading.Thread(target=shutdown)
    clear_thread.start()
    try:
        assert entered_delete.wait(5)
        shutdown_thread.start()
        assert not closed.wait(0.1), "shutdown passed an active clear operation"
    finally:
        release_delete.set()
        clear_thread.join(5)
        if shutdown_thread.ident is not None:
            shutdown_thread.join(5)
        if operation == "remove":
            manager.close()
    assert not clear_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert errors == []
    assert closed.is_set()
    assert manager.split_tier_manifest.lookup(logical) is None
