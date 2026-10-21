# SPDX-License-Identifier: Apache-2.0
"""Exercise joined fills through the real controller, L1 manager and mock L2."""

# Standard
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Event
from unittest.mock import MagicMock, patch
import os

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import PrefetchLockMode
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2LoadOutcome
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2Adapter
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    RetainPrefetchPolicy,
)
from tests.v1.distributed.test_prefetch_controller import (
    make_adapter,
    make_controller,
    make_l1_config,
    make_layout,
    make_object_key,
    row_bits,
    single_row_spec,
    store_keys_in_l2,
    wait_for_condition,
    wait_for_result,
)


@pytest.fixture
def l1_manager() -> Iterator[L1Manager]:
    """Create a bounded real L1 arena for overlap tests."""
    manager = L1Manager(make_l1_config())
    yield manager
    manager.close()


@contextmanager
def joined_controller(
    manager: L1Manager, max_pending: int = 1024
) -> Iterator[tuple[PrefetchController, MockL2Adapter, MagicMock, Callable[[], None]]]:
    """Hold real load completion until a test releases the public eventfd."""
    adapter = make_adapter()
    ctrl = make_controller(
        manager,
        [adapter],
        policy=RetainPrefetchPolicy(),
        join_overlaps=True,
        max_pending=max_pending,
    )
    allow_completion = Event()
    original_query = adapter.query_load_outcome

    def gated_query(task_id: int) -> L2LoadOutcome | None:
        return original_query(task_id) if allow_completion.is_set() else None

    def release() -> None:
        allow_completion.set()
        os.eventfd_write(adapter.get_load_event_fd(), 1)

    with (
        patch.object(adapter, "query_load_outcome", side_effect=gated_query),
        patch.object(
            adapter, "submit_load_task", wraps=adapter.submit_load_task
        ) as loads,
    ):
        ctrl.start()
        try:
            yield ctrl, adapter, loads, release
        finally:
            release()
            ctrl.stop_submission()
            outcome = adapter.close()
            ctrl.finish_shutdown({0: outcome})


@pytest.mark.parametrize("warm_first", [False, True])
def test_distinct_tails_join_same_fill(l1_manager: L1Manager, warm_first: bool) -> None:
    """Different query lengths share objects, while every reader owns its refs."""
    keys = [make_object_key(i) for i in range(4)]
    with joined_controller(l1_manager) as (ctrl, adapter, loads, release):
        store_keys_in_l2(adapter, keys, make_layout())
        leader = single_row_spec(
            keys if warm_first else keys + [make_object_key(100)],
            lock_mode=PrefetchLockMode.NO_LOCK if warm_first else PrefetchLockMode.LOCK,
            fetching_policy="full" if warm_first else "prefix",
            num_kv_readers=2,
        )
        first = ctrl.submit_prefetch_request(leader)
        assert wait_for_condition(lambda: loads.call_count == 1)
        children = [
            ctrl.submit_prefetch_request(
                single_row_spec(keys + [make_object_key(101 + i)], num_kv_readers=2)
            )
            for i in range(3)
        ]
        duplicate_hint = ctrl.submit_prefetch_request(
            single_row_spec(
                keys, fetching_policy="full", lock_mode=PrefetchLockMode.NO_LOCK
            )
        )
        assert wait_for_condition(
            lambda: ctrl.report_status()["dependency_waiter_count"] == 4
        )
        assert loads.call_count == 1
        assert l1_manager.report_status()["external_io_lease_count"] == 1
        release()
        assert row_bits(wait_for_result(ctrl, first)) == list(range(4))
        objects = l1_manager.unsafe_read(keys)
        for req_id in children + [duplicate_hint]:
            assert row_bits(wait_for_result(ctrl, req_id)) == list(range(4))
            assert all(
                l1_manager.unsafe_read(keys)[key][1] is objects[key][1] for key in keys
            )
        assert loads.call_count == 1
        for _ in range(3 + int(not warm_first)):
            l1_manager.finish_read(keys, read_locks=2)
        assert l1_manager.report_status()["read_locked_count"] == 0
        assert l1_manager.report_status()["temporary_count"] == 0
        assert ctrl.report_status()["accepted_handle_count"] == 0
        late = ctrl.submit_prefetch_request(single_row_spec(keys))
        assert row_bits(wait_for_result(ctrl, late)) == list(range(4))
        assert all(
            l1_manager.unsafe_read(keys)[key][1] is objects[key][1] for key in keys
        )
        l1_manager.finish_read(keys)
        assert loads.call_count == 1


def test_disjoint_fill_progresses_while_overlap_waits(l1_manager: L1Manager) -> None:
    keys = [make_object_key(i) for i in range(4)]
    other = [make_object_key(20 + i) for i in range(4)]
    with joined_controller(l1_manager) as (ctrl, adapter, loads, release):
        store_keys_in_l2(adapter, keys + other, make_layout())
        first = ctrl.submit_prefetch_request(single_row_spec(keys))
        assert wait_for_condition(lambda: loads.call_count == 1)
        joined = ctrl.submit_prefetch_request(
            single_row_spec(keys + [make_object_key(99)])
        )
        disjoint = ctrl.submit_prefetch_request(single_row_spec(other))
        assert wait_for_condition(lambda: loads.call_count == 2)
        assert ctrl.report_status()["dependency_waiter_count"] == 1
        release()
        for req_id in [first, joined, disjoint]:
            assert row_bits(wait_for_result(ctrl, req_id)) == list(range(4))
        assert loads.call_count == 2
        l1_manager.finish_read(keys, read_locks=2)
        l1_manager.finish_read(other)
        assert l1_manager.report_status()["read_locked_count"] == 0


@pytest.mark.parametrize("cancel_leader", [False, True])
def test_cancel_keeps_accepted_fill_until_completion(
    l1_manager: L1Manager, cancel_leader: bool
) -> None:
    keys = [make_object_key(i) for i in range(4)]
    with joined_controller(l1_manager) as (ctrl, adapter, loads, release):
        store_keys_in_l2(adapter, keys, make_layout())
        first = ctrl.submit_prefetch_request(single_row_spec(keys))
        assert wait_for_condition(lambda: loads.call_count == 1)
        joined = ctrl.submit_prefetch_request(
            single_row_spec(keys + [make_object_key(99)])
        )
        assert wait_for_condition(
            lambda: ctrl.report_status()["dependency_waiter_count"] == 1
        )
        cancelled, survivor = (first, joined) if cancel_leader else (joined, first)
        assert ctrl.cancel_prefetch_request(cancelled)
        assert ctrl.query_prefetch_result(cancelled) is None
        assert l1_manager.report_status()["external_io_lease_count"] == 1
        release()
        assert row_bits(wait_for_result(ctrl, survivor)) == list(range(4))
        assert wait_for_condition(
            lambda: ctrl.report_status()["accepted_handle_count"] == 0
        )
        assert ctrl.query_prefetch_result(cancelled) is None
        l1_manager.finish_read(keys)
        assert l1_manager.report_status()["read_locked_count"] == 0
        assert l1_manager.report_status()["external_io_lease_count"] == 0
        assert loads.call_count == 1


def test_unconsumed_result_counts_toward_capacity_and_can_be_cancelled(
    l1_manager: L1Manager,
) -> None:
    keys = [make_object_key(0)]
    with joined_controller(l1_manager, max_pending=1) as (
        ctrl,
        adapter,
        loads,
        release,
    ):
        store_keys_in_l2(adapter, keys, make_layout())
        first = ctrl.submit_prefetch_request(single_row_spec(keys))
        assert wait_for_condition(lambda: loads.call_count == 1)
        with pytest.raises(RuntimeError, match="admission capacity"):
            ctrl.submit_prefetch_request(single_row_spec(keys))
        release()
        assert ctrl.wait_prefetch_result(first, 5)
        with pytest.raises(RuntimeError, match="admission capacity"):
            ctrl.submit_prefetch_request(single_row_spec(keys))
        assert ctrl.cancel_prefetch_request(first)
        assert l1_manager.report_status()["read_locked_count"] == 0
        second = ctrl.submit_prefetch_request(single_row_spec(keys))
        assert row_bits(wait_for_result(ctrl, second)) == [0]
        l1_manager.finish_read(keys)
        assert not ctrl.cancel_prefetch_request(second)


def test_unused_hint_drains_without_reader_refs(l1_manager: L1Manager) -> None:
    keys = [make_object_key(0)]
    with joined_controller(l1_manager) as (ctrl, adapter, loads, release):
        store_keys_in_l2(adapter, keys, make_layout())
        hint = ctrl.submit_prefetch_request(
            single_row_spec(
                keys, fetching_policy="full", lock_mode=PrefetchLockMode.NO_LOCK
            )
        )
        assert wait_for_condition(lambda: loads.call_count == 1)
        release()
        assert row_bits(wait_for_result(ctrl, hint)) == [0]
        assert l1_manager.report_status()["read_locked_count"] == 0
        assert l1_manager.report_status()["temporary_count"] == 0
        assert ctrl.report_status()["accepted_handle_count"] == 0


def test_failed_fill_releases_dependency_and_owned_buffers(
    l1_manager: L1Manager,
) -> None:
    keys = [make_object_key(0)]
    with joined_controller(l1_manager) as (ctrl, adapter, loads, release):
        store_keys_in_l2(adapter, keys, make_layout())
        first = ctrl.submit_prefetch_request(single_row_spec(keys))
        assert wait_for_condition(lambda: loads.call_count == 1)
        joined = ctrl.submit_prefetch_request(single_row_spec(keys))
        assert wait_for_condition(
            lambda: ctrl.report_status()["dependency_waiter_count"] == 1
        )
        original_query = adapter.query_load_outcome

        def fail_first(task_id: int) -> L2LoadOutcome | None:
            result = original_query(task_id)
            if result is None:
                return None
            if loads.call_count == 1:
                return L2LoadOutcome(Bitmap(len(result.bitmap)), True)
            return result

        with patch.object(adapter, "query_load_outcome", side_effect=fail_first):
            release()
            assert row_bits(wait_for_result(ctrl, first)) == []
            assert row_bits(wait_for_result(ctrl, joined)) == [0]
        l1_manager.finish_read(keys)
        assert l1_manager.report_status()["read_locked_count"] == 0
        assert l1_manager.report_status()["external_io_lease_count"] == 0
        assert loads.call_count == 2


def test_shutdown_publishes_every_accepted_handle(l1_manager: L1Manager) -> None:
    keys = [make_object_key(0)]
    adapter = make_adapter()
    ctrl = make_controller(
        l1_manager, [adapter], policy=RetainPrefetchPolicy(), join_overlaps=True
    )
    ctrl.start()
    first = ctrl.submit_prefetch_request(single_row_spec(keys))
    joined = ctrl.submit_prefetch_request(single_row_spec(keys))
    ctrl.stop_submission()
    ctrl.finish_shutdown({0: adapter.close()})
    assert row_bits(ctrl.query_prefetch_result(first)) == []
    assert row_bits(ctrl.query_prefetch_result(joined)) == []
    assert ctrl.report_status()["accepted_handle_count"] == 0
