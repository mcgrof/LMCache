# SPDX-License-Identifier: Apache-2.0
"""Exercise controller ownership transitions with real L1 allocations."""

# Standard
from threading import Event

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import (
    L2CloseOutcome,
    L2LoadOutcome,
    L2StoreOutcome,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreController,
    StoreListener,
)
from lmcache.v1.distributed.storage_controllers.store_policy import DefaultStorePolicy
from lmcache.v1.memory_management import MemoryObj
from tests.v1.distributed.test_prefetch_controller import (
    make_controller,
    make_descriptor,
    make_layout,
    make_object_key,
    single_row_spec,
    store_keys_in_l2,
    wait_for_condition,
)

pytestmark = pytest.mark.no_shared_allocator


class OwnershipAdapter(MockL2Adapter):
    """Gate public completion proofs independently of task success."""

    def __init__(self, reusable: bool, pending: bool, success: bool = True) -> None:
        super().__init__(MockL2AdapterConfig(0.01, 10))
        self.reusable = reusable
        self.pending = pending
        self.success = success
        self.submitted = Event()
        self.objects: list[MemoryObj] = []

    def submit_store_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        task = super().submit_store_task(keys, objects)
        self.objects = list(objects)
        self.submitted.set()
        return task

    def submit_load_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        task = super().submit_load_task(keys, objects)
        self.objects = list(objects)
        self.submitted.set()
        return task

    def pop_completed_store_outcomes(self) -> dict[L2TaskId, L2StoreOutcome]:
        if self.pending:
            return {}
        return {
            task: L2StoreOutcome(
                result if self.success else L2StoreResult(False, 0), self.reusable
            )
            for task, result in self.pop_completed_store_tasks().items()
        }

    def query_load_outcome(self, task_id: L2TaskId) -> L2LoadOutcome | None:
        if self.pending:
            return None
        bitmap = self.query_load_result(task_id)
        return None if bitmap is None else L2LoadOutcome(bitmap, self.reusable)

    def close_with_outcome(self) -> L2CloseOutcome:
        self.close()
        return L2CloseOutcome(self.reusable)


def new_manager() -> L1Manager:
    return L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(4 << 20, False, shm_name=""),
        )
    )


def write(manager: L1Manager, key: ObjectKey, layout: MemoryLayoutDesc) -> None:
    _, obj = manager.reserve_write([key], [False], layout)[key]
    assert obj is not None
    manager.finish_write([key])


@pytest.mark.parametrize(
    "reusable,success", [(True, True), (True, False), (False, False)]
)
def test_store_completion_retains_only_unresolved_sources(
    reusable: bool, success: bool
) -> None:
    manager = new_manager()
    adapter = OwnershipAdapter(reusable, False, success)
    controller = StoreController(
        manager, [adapter], [make_descriptor(0)], DefaultStorePolicy()
    )
    controller.start()
    try:
        key = make_object_key(1)
        write(manager, key, make_layout())
        assert adapter.submitted.wait(5)
        assert wait_for_condition(
            lambda: controller.report_status()["in_flight_task_count"] == 0
        )
        assert manager.report_status()["external_io_lease_count"] == 0
        assert manager.report_status()["backing_resources_retained"] is not reusable
        controller.stop_submission()
        controller.finish_shutdown({0: adapter.close_with_outcome()})
        manager.clear(force=True)
        manager.close()
        assert adapter.objects[0].is_valid() is not reusable
    finally:
        if controller.report_status()["thread_alive"]:
            controller.stop()
        manager.close()


@pytest.mark.parametrize("reusable", [True, False])
def test_load_completion_withholds_unknown_data_and_source_unlock(
    reusable: bool,
) -> None:
    manager = new_manager()
    adapter = OwnershipAdapter(reusable, False)
    key = make_object_key(2)
    store_keys_in_l2(adapter, [key], make_layout())
    adapter.submitted.clear()
    controller = make_controller(manager, [adapter])
    controller.start()
    try:
        request = controller.submit_prefetch_request(single_row_spec([key]))
        assert controller.wait_prefetch_result(request, 5)
        result = controller.query_prefetch_result(request)
        assert result is not None
        assert result.hit_cells[0].get_indices_list() == ([0] if reusable else [])
        assert manager.report_status()["backing_resources_retained"] is not reusable
        assert manager.report_status()["external_io_lease_count"] == 0
        assert wait_for_condition(
            lambda: adapter.debug_get_locked_key_count() == (0 if reusable else 1)
        )
        controller.stop_submission()
        controller.finish_shutdown({0: adapter.close_with_outcome()})
        manager.clear(force=True)
        manager.close()
        assert adapter.objects[0].is_valid() is not reusable
    finally:
        if controller.report_status()["thread_alive"]:
            controller.stop()
        manager.close()


@pytest.mark.parametrize("operation", ["store", "load"])
@pytest.mark.parametrize("reusable", [True, False])
def test_pending_shutdown_requires_adapter_quiescence(
    operation: str, reusable: bool
) -> None:
    manager = new_manager()
    adapter = OwnershipAdapter(reusable, True)
    key = make_object_key(3)
    controller: StoreController | PrefetchController
    if operation == "store":
        controller = StoreController(
            manager, [adapter], [make_descriptor(0)], DefaultStorePolicy()
        )
        controller.start()
        write(manager, key, make_layout())
    else:
        store_keys_in_l2(adapter, [key], make_layout())
        adapter.submitted.clear()
        controller = make_controller(manager, [adapter])
        controller.start()
        controller.submit_prefetch_request(single_row_spec([key]))
    assert adapter.submitted.wait(5)
    assert manager.report_status()["external_io_lease_count"] == 1
    controller.stop_submission()
    assert manager.report_status()["external_io_lease_count"] == 1
    controller.finish_shutdown({0: adapter.close_with_outcome()})
    assert manager.report_status()["external_io_lease_count"] == 0
    assert manager.report_status()["backing_resources_retained"] is not reusable
    manager.clear(force=True)
    manager.close()
    assert adapter.objects[0].is_valid() is not reusable


@pytest.mark.parametrize("close_raises", [False, True])
def test_storage_shutdown_retains_affinity_arena_without_pending_tasks(
    close_raises: bool,
) -> None:
    # Standard
    from unittest.mock import patch

    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L2AdaptersConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.storage_manager import StorageManager

    adapter = OwnershipAdapter(False, True)
    config = StorageManagerConfig(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(4 << 20, False, shm_name="")
        ),
        EvictionConfig(eviction_policy="noop"),
        L2AdaptersConfig([MockL2AdapterConfig(0.01, 10)]),
    )
    with patch(
        "lmcache.v1.distributed.storage_manager.create_l2_adapter", return_value=adapter
    ):
        manager = StorageManager(config)
    key = make_object_key(4)
    obj = manager.reserve_write([key], make_layout())[key]
    with patch.object(
        adapter, "close_with_outcome", wraps=adapter.close_with_outcome
    ) as close:
        if close_raises:

            def failed_close() -> L2CloseOutcome:
                adapter.close()
                raise RuntimeError("shutdown proof unavailable")

            close.side_effect = failed_close
        manager.close()
        manager.close()
        close.assert_called_once()
    assert manager.report_status()["l1_manager"]["backing_resources_retained"]
    assert obj.is_valid()
    with pytest.raises(RuntimeError, match="closed"):
        manager.reserve_write([key], make_layout())


def test_partial_initialization_retains_an_unresolved_adapter_owner() -> None:
    # Standard
    from unittest.mock import create_autospec, patch

    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L2AdaptersConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
    from lmcache.v1.distributed.serde import SerdeConfig
    from lmcache.v1.distributed.storage_manager import StorageManager

    inner = create_autospec(L2AdapterInterface, instance=True)
    inner.may_report_unresolved_io = True
    inner.close_with_outcome.return_value = L2CloseOutcome(False)
    adapter_config = MockL2AdapterConfig(0.01, 10)
    adapter_config.serde_config = SerdeConfig("fp8")
    config = StorageManagerConfig(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(4 << 20, False, shm_name="")
        ),
        EvictionConfig(eviction_policy="noop"),
        L2AdaptersConfig([adapter_config]),
    )
    owners: list[L1Manager] = []

    def new_owner(owner_config: L1ManagerConfig) -> L1Manager:
        owner = L1Manager(owner_config)
        owners.append(owner)
        return owner

    with (
        patch(
            "lmcache.v1.distributed.storage_manager.L1Manager", side_effect=new_owner
        ),
        patch(
            "lmcache.v1.distributed.storage_manager.create_l2_adapter",
            return_value=inner,
        ),
        pytest.raises(ValueError, match="intermediate buffers"),
    ):
        StorageManager(config)
    inner.close_with_outcome.assert_called_once()
    assert owners[0].report_status()["backing_resources_retained"]


def test_closed_store_listener_ignores_late_l1_write_completion() -> None:
    listener = StoreListener()
    listener.close()
    listener.on_l1_keys_write_finished([make_object_key(5)])
    listener.notify()
    listener.close()
    assert listener.pending_count() == 0


@pytest.mark.parametrize("operation", ["store", "load"])
def test_failed_shutdown_query_retains_pending_buffer_owner(operation: str) -> None:
    # Standard
    from unittest.mock import patch

    manager = new_manager()
    adapter = OwnershipAdapter(True, True)
    key = make_object_key(6)
    controller: StoreController | PrefetchController
    if operation == "store":
        controller = StoreController(
            manager, [adapter], [make_descriptor(0)], DefaultStorePolicy()
        )
        controller.start()
        write(manager, key, make_layout())
        query = "pop_completed_store_outcomes"
    else:
        store_keys_in_l2(adapter, [key], make_layout())
        adapter.submitted.clear()
        controller = make_controller(manager, [adapter])
        controller.start()
        controller.submit_prefetch_request(single_row_spec([key]))
        query = "query_load_outcome"
    assert adapter.submitted.wait(5)
    controller.stop_submission()
    with patch.object(
        adapter, query, side_effect=RuntimeError("completion query failed")
    ):
        with pytest.raises(RuntimeError, match="query failed"):
            controller.finish_shutdown({0: None})
    assert manager.report_status()["backing_resources_retained"]
    manager.close()
    assert adapter.objects[0].is_valid()
    adapter.close()
