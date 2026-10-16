# SPDX-License-Identifier: Apache-2.0
"""Deterministic interruption regressions for KMesh cleanup receipts."""

# Standard
from typing import Any, Callable, cast
from unittest.mock import MagicMock, patch
import inspect
import sys
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.cache_engine import (
    LMCacheEngine,
    LMCacheEngineBuilder,
    _StoreCleanupOwner,
    _StoreOperationLease,
)
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMBufferLayerwiseGPUConnector,
    _GPUStoreCleanupOwner,
    _GPUStoreOperationLease,
)
from lmcache.v1.lifecycle import RetainedCall
from lmcache.v1.manager import LMCacheManager
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_allocators.tensor_memory_allocator import (
    TensorMemoryAllocator,
)
from lmcache.v1.memory_management import (
    AddressManager,
    AllocationReceipt,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.storage_manager import (
    StorageManager,
    _BackendCloseOwner,
    _PutOperationLease,
    _PutRefCleanupOwner,
)


def _nth_source_line(function: Callable[..., Any], needle: str, nth: int) -> int:
    """Return the line number of the ``nth`` exact source-text match."""
    lines, start_line = inspect.getsourcelines(function)
    matches = [
        start_line + offset
        for offset, line in enumerate(lines)
        if line.strip() == needle
    ]
    assert len(matches) > nth, (function, needle, matches)
    return matches[nth]


def _interrupt_on_line(
    function: Callable[..., Any],
    line_number: int,
    message: str,
) -> tuple[Callable[..., Any], Callable[[], bool]]:
    """Build a one-shot current-thread trace interrupt for one function line."""
    triggered = False

    def trace(frame: Any, event: str, arg: Any) -> Any:
        del arg
        nonlocal triggered
        if (
            not triggered
            and frame.f_code is function.__code__
            and event == "line"
            and frame.f_lineno == line_number
        ):
            triggered = True
            sys.settrace(None)
            raise KeyboardInterrupt(message)
        return trace

    return trace, lambda: triggered


def _tensor_memory_obj(ref_count: int = 2) -> TensorMemoryObj:
    metadata = MemoryObjMetadata(
        shape=torch.Size([1]),
        dtype=torch.uint8,
        address=0,
        phy_size=1,
        ref_count=ref_count,
        fmt=MemoryFormat.BINARY,
    )
    return TensorMemoryObj(
        raw_data=torch.zeros(1, dtype=torch.uint8),
        metadata=metadata,
        parent_allocator=None,
    )


def test_ref_count_receipt_replay_after_atomic_commit_resyncs_mirror() -> None:
    memory_obj = _tensor_memory_obj()
    receipt = memory_obj.make_ref_count_down_receipt()

    # This is the second mirror assignment in the method: the first is the
    # already-completed fast path, while this one immediately follows the
    # atomic authoritative-state pointer assignment.
    interrupt_line = _nth_source_line(
        TensorMemoryObj._ref_count_down_once_locked,
        "self.meta.ref_count = state.count",
        1,
    )
    trace, was_triggered = _interrupt_on_line(
        TensorMemoryObj._ref_count_down_once_locked,
        interrupt_line,
        "injected after atomic ref-state commit",
    )

    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="atomic ref-state commit"):
            memory_obj.ref_count_down_once(receipt)
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert memory_obj.get_ref_count() == 1
    assert memory_obj.meta.ref_count == 2
    assert not memory_obj.ref_count_down_receipt_complete(receipt)

    metadata = memory_obj.metadata

    assert metadata.ref_count == 1
    assert memory_obj.get_ref_count() == 1
    assert not memory_obj.ref_count_down_receipt_complete(receipt)

    memory_obj.ref_count_down_once(receipt)
    memory_obj.ref_count_down_once(receipt)

    assert memory_obj.get_ref_count() == 1
    assert memory_obj.meta.ref_count == 1
    assert memory_obj.ref_count_down_receipt_complete(receipt)


def test_ref_count_up_receipt_replays_atomic_increment_once() -> None:
    memory_obj = _tensor_memory_obj()
    receipt = memory_obj.make_ref_count_up_receipt()
    interrupt_line = _nth_source_line(
        TensorMemoryObj._ref_count_up_once_locked,
        "self.meta.ref_count = state.count",
        1,
    )
    trace, was_triggered = _interrupt_on_line(
        TensorMemoryObj._ref_count_up_once_locked,
        interrupt_line,
        "injected after atomic ref-count acquisition",
    )

    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="ref-count acquisition"):
            memory_obj.ref_count_up_once(receipt)
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert memory_obj.get_ref_count() == 3
    assert memory_obj.meta.ref_count == 2
    assert memory_obj.ref_count_up_receipt_applied(receipt)
    assert not memory_obj.ref_count_up_receipt_complete(receipt)

    memory_obj.ref_count_up_once(receipt)
    memory_obj.ref_count_up_once(receipt)

    assert memory_obj.get_ref_count() == 3
    assert memory_obj.meta.ref_count == 3
    assert memory_obj.ref_count_up_receipt_complete(receipt)


def test_concurrent_replay_of_same_ref_count_receipt_decrements_once() -> None:
    memory_obj = _tensor_memory_obj()
    receipt = memory_obj.make_ref_count_down_receipt()
    worker_count = 16
    start = threading.Barrier(worker_count + 1)
    errors: list[BaseException] = []

    def release() -> None:
        try:
            start.wait(timeout=5.0)
            memory_obj.ref_count_down_once(receipt)
        except BaseException as error:
            errors.append(error)

    workers = [threading.Thread(target=release) for _ in range(worker_count)]
    for worker in workers:
        worker.start()
    start.wait(timeout=5.0)
    for worker in workers:
        worker.join(timeout=5.0)

    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    assert memory_obj.get_ref_count() == 1
    assert memory_obj.meta.ref_count == 1
    assert memory_obj.ref_count_down_receipt_complete(receipt)


def test_address_allocation_receipt_recovers_commit_before_return_interrupt() -> None:
    manager = AddressManager(size=1, align_bytes=1)
    receipt = AllocationReceipt()
    allocate_once = inspect.unwrap(AddressManager.allocate_once)
    interrupt_line = _nth_source_line(
        allocate_once,
        "return start, allocated_size",
        0,
    )
    trace, was_triggered = _interrupt_on_line(
        allocate_once,
        interrupt_line,
        "injected after address allocation commit",
    )

    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="address allocation commit"):
            manager.allocate_once(1, receipt)
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert manager.active_allocation_count == 1
    assert manager.total_allocated_size == 1
    assert manager.get_free_size() == 0
    assert not receipt.adopted
    assert not receipt.release_complete()

    receipt.release_once()
    receipt.release_once()

    assert receipt.release_complete()
    assert manager.active_allocation_count == 0
    assert manager.total_allocated_size == 0
    assert manager.get_free_size() == 1
    assert manager.check_consistency()


def test_address_batched_receipts_recover_atomic_commit_interrupt() -> None:
    manager = AddressManager(size=4, align_bytes=1)
    receipts = [AllocationReceipt() for _ in range(4)]
    batched_allocate_once = inspect.unwrap(AddressManager.batched_allocate_once)
    interrupt_line = _nth_source_line(
        batched_allocate_once,
        "return results",
        0,
    )
    trace, was_triggered = _interrupt_on_line(
        batched_allocate_once,
        interrupt_line,
        "injected after batched address commit",
    )

    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="batched address commit"):
            manager.batched_allocate_once(1, receipts)
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert manager.active_allocation_count == 4
    assert manager.total_allocated_size == 4
    assert manager.get_free_size() == 0

    # Release the kth reservation first, replay it, then drain the rest. Each
    # receipt names its own generation even though the batch published once.
    receipts[2].release_once()
    receipts[2].release_once()
    for index in (0, 3, 1):
        receipts[index].release_once()

    assert all(receipt.release_complete() for receipt in receipts)
    assert manager.active_allocation_count == 0
    assert manager.total_allocated_size == 0
    assert manager.get_free_size() == 4
    assert manager.check_consistency()


def test_single_allocation_receipt_allocate_release_race_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release between the final receipt check and commit cannot leak."""
    allocator = TensorMemoryAllocator(
        torch.zeros(1, dtype=torch.uint8),
        align_bytes=1,
    )
    manager = allocator.address_manager
    receipt = AllocationReceipt()
    commit_checked = threading.Event()
    permit_commit = threading.Event()
    release_published = threading.Event()
    allocation_results: list[TensorMemoryObj | None] = []
    allocation_errors: list[BaseException] = []
    release_errors: list[BaseException] = []

    original_may_commit = receipt._allocation_may_commit

    def pause_after_commit_check(candidate: Any) -> bool:
        may_commit = original_may_commit(candidate)
        commit_checked.set()
        assert permit_commit.wait(timeout=5.0)
        return may_commit

    original_begin_release = receipt._begin_address_release

    def publish_release(candidate: Any) -> bool:
        started = original_begin_release(candidate)
        release_published.set()
        return started

    monkeypatch.setattr(receipt, "_allocation_may_commit", pause_after_commit_check)
    monkeypatch.setattr(receipt, "_begin_address_release", publish_release)

    def allocate() -> None:
        try:
            allocation_results.append(
                allocator.allocate(
                    torch.Size([1]),
                    torch.uint8,
                    MemoryFormat.BINARY,
                    allocation_receipt=receipt,
                )
            )
        except BaseException as error:
            allocation_errors.append(error)

    def release() -> None:
        try:
            receipt.release_once()
        except BaseException as error:
            release_errors.append(error)

    allocator_thread = threading.Thread(target=allocate)
    allocator_thread.start()
    assert commit_checked.wait(timeout=5.0)

    release_thread = threading.Thread(target=release)
    release_thread.start()
    assert release_published.wait(timeout=5.0)

    # The allocator still holds the AddressManager lock. Release intent is
    # durable, while neither the immutable allocation nor its free has landed.
    assert manager.active_allocation_count == 0
    permit_commit.set()
    allocator_thread.join(timeout=5.0)
    release_thread.join(timeout=5.0)

    assert not allocator_thread.is_alive()
    assert not release_thread.is_alive()
    assert allocation_results == []
    assert release_errors == []
    assert len(allocation_errors) == 1
    assert isinstance(allocation_errors[0], RuntimeError)
    assert "cannot adopt a released allocation receipt" in str(allocation_errors[0])
    assert receipt.release_complete()
    assert manager.active_allocation_count == 0
    assert manager.total_allocated_size == 0
    assert manager.get_free_size() == 1
    assert manager.check_consistency()


def test_batched_allocation_receipt_allocate_release_race_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kth-slot cancellation leaves every other batch slot recoverable."""
    allocator = TensorMemoryAllocator(
        torch.zeros(4, dtype=torch.uint8),
        align_bytes=1,
    )
    manager = allocator.address_manager
    receipts = [AllocationReceipt() for _ in range(4)]
    cancelled = receipts[2]
    commit_checked = threading.Event()
    permit_commit = threading.Event()
    release_published = threading.Event()
    allocation_results: list[list[TensorMemoryObj] | None] = []
    allocation_errors: list[BaseException] = []
    release_errors: list[BaseException] = []

    original_may_commit = cancelled._allocation_may_commit

    def pause_after_kth_commit_check(candidate: Any) -> bool:
        may_commit = original_may_commit(candidate)
        commit_checked.set()
        assert permit_commit.wait(timeout=5.0)
        return may_commit

    original_begin_release = cancelled._begin_address_release

    def publish_kth_release(candidate: Any) -> bool:
        started = original_begin_release(candidate)
        release_published.set()
        return started

    monkeypatch.setattr(
        cancelled,
        "_allocation_may_commit",
        pause_after_kth_commit_check,
    )
    monkeypatch.setattr(cancelled, "_begin_address_release", publish_kth_release)

    def allocate_batch() -> None:
        try:
            allocation_results.append(
                allocator.batched_allocate(
                    torch.Size([1]),
                    torch.uint8,
                    4,
                    MemoryFormat.BINARY,
                    allocation_receipts=receipts,
                )
            )
        except BaseException as error:
            allocation_errors.append(error)

    def release_kth() -> None:
        try:
            cancelled.release_once()
        except BaseException as error:
            release_errors.append(error)

    allocator_thread = threading.Thread(target=allocate_batch)
    allocator_thread.start()
    assert commit_checked.wait(timeout=5.0)

    release_thread = threading.Thread(target=release_kth)
    release_thread.start()
    assert release_published.wait(timeout=5.0)
    assert manager.active_allocation_count == 0
    permit_commit.set()
    allocator_thread.join(timeout=5.0)
    release_thread.join(timeout=5.0)

    assert not allocator_thread.is_alive()
    assert not release_thread.is_alive()
    assert allocation_results == []
    assert release_errors == []
    assert len(allocation_errors) == 1
    assert isinstance(allocation_errors[0], RuntimeError)
    assert "cannot adopt a released allocation receipt" in str(allocation_errors[0])
    assert [receipt.adopted for receipt in receipts] == [True, True, False, False]
    assert cancelled.release_complete()
    assert not receipts[3].release_complete()
    assert manager.active_allocation_count == 3

    # Slots before the interrupted adoption own objects; the tail still owns a
    # raw reservation. The same receipt API drains both without range guesses.
    for index in (3, 0, 1):
        receipts[index].release_once()

    assert all(receipt.release_complete() for receipt in receipts)
    assert manager.active_allocation_count == 0
    assert manager.total_allocated_size == 0
    assert manager.get_free_size() == 4
    assert manager.check_consistency()


def test_store_owner_drains_partial_batch_adoption_interrupt() -> None:
    """Store cleanup owns both adopted objects and raw batch reservations."""

    class _NoopGPU:
        def drain_pending_gpu_cleanups(self) -> None:
            return

        def pending_gpu_cleanup_count(self) -> int:
            return 0

    class _NoopStorage:
        def drain_pending_put_cleanups(self) -> None:
            return

        def pending_put_cleanup_count(self) -> int:
            return 0

    allocator = TensorMemoryAllocator(
        torch.zeros(4, dtype=torch.uint8),
        align_bytes=1,
    )
    lease = _StoreOperationLease()
    owner = _StoreCleanupOwner(
        cast(Any, _NoopGPU()),
        cast(Any, _NoopStorage()),
        lease,
    )
    receipts = owner.reserve_allocation_receipts(4)
    batched_allocate = inspect.unwrap(TensorMemoryAllocator.batched_allocate)
    adoption_line = _nth_source_line(
        batched_allocate,
        "receipt.adopt(memory_obj)",
        0,
    )
    adoption_visits = 0

    def interrupt_before_third_adoption(frame: Any, event: str, arg: Any) -> Any:
        del arg
        nonlocal adoption_visits
        if (
            frame.f_code is batched_allocate.__code__
            and event == "line"
            and frame.f_lineno == adoption_line
        ):
            adoption_visits += 1
            if adoption_visits == 3:
                sys.settrace(None)
                raise KeyboardInterrupt("injected before third object adoption")
        return interrupt_before_third_adoption

    sys.settrace(interrupt_before_third_adoption)
    try:
        with pytest.raises(KeyboardInterrupt, match="third object adoption"):
            allocator.batched_allocate(
                torch.Size([1]),
                torch.uint8,
                4,
                MemoryFormat.BINARY,
                allocation_receipts=receipts,
            )
    finally:
        sys.settrace(None)

    assert adoption_visits == 3
    assert [receipt.adopted for receipt in receipts] == [True, True, False, False]
    # The third TensorMemoryObj was constructed before the injected line. Its
    # synchronous destructor frees that exact receipt during unwind; the fourth
    # slot was never constructed and remains a raw owner reservation.
    assert receipts[2].release_complete()
    assert not receipts[3].release_complete()
    assert allocator.num_active_allocations == 3
    assert allocator.total_allocated_size == 3

    owner.mark_pending()
    assert owner.drain() is None

    assert owner.complete
    assert all(receipt.release_complete() for receipt in receipts)
    assert allocator.num_active_allocations == 0
    assert allocator.total_allocated_size == 0
    assert allocator.address_manager.get_free_size() == 4
    assert allocator.address_manager.check_consistency()


def test_gpu_owner_drains_commit_before_object_adoption_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The connector's pre-published receipt recovers the raw GPU slot."""
    connector = VLLMBufferLayerwiseGPUConnector.__new__(VLLMBufferLayerwiseGPUConnector)
    state = cast(Any, connector)
    state.num_layers = 1
    state.hidden_dim_size = 1
    state.dtype = torch.uint8
    state.cache_positions = False
    state.kvcaches = [torch.zeros((2, 1, 1), dtype=torch.uint8)]
    state.initialize_kvcaches_ptr = lambda **kwargs: None
    state._lazy_initialize_buffer = lambda kvcaches: None
    state.get_shape = lambda num_tokens: torch.Size([1])
    state.store_stream = object()
    allocator = GPUMemoryAllocator(size=1, device="cpu", align_bytes=1)
    state.gpu_buffer_allocator = allocator
    inner_allocator = cast(TensorMemoryAllocator, allocator.allocator)
    captured_owners: list[_GPUStoreCleanupOwner] = []
    original_register = connector._register_gpu_store_cleanup

    def capture_owner(owner: _GPUStoreCleanupOwner) -> None:
        captured_owners.append(owner)
        original_register(owner)

    def interrupt_after_address_commit(start: int, size: int) -> torch.Tensor:
        del start, size
        assert len(captured_owners) == 1
        owner = captured_owners[0]
        assert not owner.allocation_receipt.adopted
        assert inner_allocator.num_active_allocations == 1
        raise KeyboardInterrupt("injected after GPU address commit")

    monkeypatch.setattr(connector, "_register_gpu_store_cleanup", capture_owner)
    monkeypatch.setattr(
        inner_allocator,
        "_get_buffer_slice",
        interrupt_after_address_commit,
    )
    storer = connector.batched_from_gpu(
        [[]],
        [0],
        [1],
        slot_mapping=torch.tensor([0]),
    )

    with pytest.raises(KeyboardInterrupt, match="GPU address commit"):
        next(storer)

    assert len(captured_owners) == 1
    assert captured_owners[0].allocation_receipt.release_complete()
    assert inner_allocator.num_active_allocations == 0
    assert inner_allocator.total_allocated_size == 0
    assert inner_allocator.address_manager.get_free_size() == 1
    assert inner_allocator.address_manager.check_consistency()
    assert connector.pending_gpu_cleanup_count() == 0
    assert connector.gpu_cleanup_owner_count() == 0


def test_interrupted_old_address_free_cannot_free_reused_generation() -> None:
    manager = AddressManager(size=1, align_bytes=1)
    old_receipt = AllocationReceipt()
    old_address = manager.allocate_once(1, old_receipt)
    free_once = inspect.unwrap(AddressManager.free_once)
    interrupt_line = _nth_source_line(
        free_once,
        "receipt._finish_address_release(self)",
        0,
    )
    trace, was_triggered = _interrupt_on_line(
        free_once,
        interrupt_line,
        "injected after exact address free commit",
    )

    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="exact address free commit"):
            manager.free_once(old_receipt)
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert not old_receipt.release_complete()
    assert manager.active_allocation_count == 0
    assert manager.get_free_size() == 1

    new_receipt = AllocationReceipt()
    new_address = manager.allocate_once(1, new_receipt)
    assert new_address == old_address

    old_receipt.release_once()

    assert old_receipt.release_complete()
    assert manager.active_allocation_count == 1
    assert manager.total_allocated_size == 1
    assert manager.get_free_size() == 0

    blocked_receipt = AllocationReceipt()
    with pytest.raises(RuntimeError, match="no memory is available"):
        manager.allocate_once(1, blocked_receipt)

    new_receipt.release_once()
    replacement_address = manager.allocate_once(1, blocked_receipt)

    assert replacement_address == new_address
    assert manager.active_allocation_count == 1
    assert manager.check_consistency()

    blocked_receipt.release_once()
    assert manager.active_allocation_count == 0
    assert manager.get_free_size() == 1


def test_ref_receipt_rejects_different_object_at_reused_address_and_later_epoch() -> (
    None
):
    allocator = TensorMemoryAllocator(
        torch.zeros(1, dtype=torch.uint8),
        align_bytes=1,
    )
    old_obj = allocator.allocate(
        torch.Size([1]),
        torch.uint8,
        MemoryFormat.BINARY,
    )
    assert old_obj is not None
    old_address = old_obj.meta.address
    old_receipt = old_obj.make_ref_count_down_receipt()

    # Release A through a different exact owner so its ref receipt remains
    # stale and available for the adversarial cross-object replay.
    old_obj.ref_count_down()
    assert allocator.num_active_allocations == 0

    new_obj = allocator.allocate(
        torch.Size([1]),
        torch.uint8,
        MemoryFormat.BINARY,
    )
    assert new_obj is not None
    assert new_obj is not old_obj
    assert new_obj.meta.address == old_address

    # Move B beyond A's epoch. Object identity must be checked before the
    # stale-epoch fast path; the old receipt can never name this new object.
    new_obj.reset_ref_count_for_reuse()
    with pytest.raises(ValueError, match="another memory object"):
        new_obj.ref_count_down_once(old_receipt)
    with pytest.raises(ValueError, match="another memory object"):
        new_obj.ref_count_down_receipt_complete(old_receipt)

    assert new_obj.get_ref_count() == 1
    assert new_obj.metadata.ref_count == 1
    assert allocator.num_active_allocations == 1
    assert allocator.total_allocated_size == 1

    new_obj.ref_count_down()
    assert allocator.num_active_allocations == 0
    assert allocator.total_allocated_size == 0


def test_paged_reuse_discards_replayed_old_epoch_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = PagedTensorMemoryAllocator(
        torch.zeros(1, dtype=torch.uint8),
        [torch.Size([1])],
        [torch.uint8],
        MemoryFormat.BINARY,
    )
    memory_obj = allocator.allocate(
        torch.Size([1]),
        torch.uint8,
        MemoryFormat.BINARY,
    )
    assert memory_obj is not None
    receipt = memory_obj.make_ref_count_down_receipt()

    original_free = allocator.free

    def free_then_interrupt(obj: TensorMemoryObj, allocator_type: Any = None) -> None:
        original_free(obj, allocator_type)
        raise KeyboardInterrupt("injected after free publication")

    monkeypatch.setattr(allocator, "free", free_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="free publication"):
        memory_obj.ref_count_down_once(receipt)

    assert memory_obj.get_ref_count() == 0
    assert len(allocator.free_blocks) == 1
    assert not memory_obj.ref_count_down_receipt_complete(receipt)

    monkeypatch.setattr(allocator, "free", original_free)
    original_popleft = allocator.free_blocks.popleft
    page_popped = threading.Event()
    permit_reuse_reset = threading.Event()

    def popleft_then_pause() -> TensorMemoryObj:
        obj = original_popleft()
        page_popped.set()
        assert permit_reuse_reset.wait(timeout=5.0)
        return obj

    monkeypatch.setattr(allocator.free_blocks, "popleft", popleft_then_pause)
    allocation_result: list[TensorMemoryObj | None] = []
    allocation_errors: list[BaseException] = []

    def reuse_page() -> None:
        try:
            allocation_result.append(
                allocator.allocate(
                    torch.Size([1]),
                    torch.uint8,
                    MemoryFormat.BINARY,
                )
            )
        except BaseException as error:
            allocation_errors.append(error)

    allocator_thread = threading.Thread(target=reuse_page)
    allocator_thread.start()
    assert page_popped.wait(timeout=5.0)

    # Replay the interrupted old-epoch release after popleft. It republishes
    # the page, then the allocator must discard that stale publication while
    # advancing the allocation epoch under the object's lock.
    memory_obj.ref_count_down_once(receipt)
    assert len(allocator.free_blocks) == 1
    assert memory_obj.ref_count_down_receipt_complete(receipt)

    permit_reuse_reset.set()
    allocator_thread.join(timeout=5.0)

    assert not allocator_thread.is_alive()
    assert allocation_errors == []
    assert allocation_result == [memory_obj]
    assert len(allocator.free_blocks) == 0
    assert memory_obj.get_ref_count() == 1
    assert memory_obj.meta.ref_count == 1

    # A delayed cleanup holding the old receipt treats the newer allocation as
    # already complete and cannot release or re-enqueue the reused page.
    memory_obj.ref_count_down_once(receipt)
    assert memory_obj.ref_count_down_receipt_complete(receipt)
    assert memory_obj.get_ref_count() == 1
    assert len(allocator.free_blocks) == 0

    memory_obj.ref_count_down()
    assert len(allocator.free_blocks) == 1


def test_backend_close_retains_success_across_waiter_interrupt() -> None:
    class _Manager:
        def __init__(self) -> None:
            self.worker_starts = 0
            self.worker_finishes = 0

        def _register_backend_close_worker(self) -> None:
            self.worker_starts += 1

        def _unregister_backend_close_worker(self) -> None:
            self.worker_finishes += 1

    class _Backend:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    manager = _Manager()
    backend = _Backend()
    owner = _BackendCloseOwner(cast(Any, manager), "receipt-test", backend)

    interrupt_line = _nth_source_line(RetainedCall.run, "with self._lock:", 1)
    trace, was_triggered = _interrupt_on_line(
        RetainedCall.run,
        interrupt_line,
        "injected after retained close waiter returned",
    )
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="retained close waiter"):
            owner.close()
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert owner.complete
    assert backend.close_calls == 1

    owner.close()

    assert owner.complete
    assert backend.close_calls == 1
    assert manager.worker_starts == 1
    assert manager.worker_finishes == 1


def test_retained_call_restarts_published_worker_after_pre_start_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0
    start_attempts: list[threading.Thread] = []
    thread_name = "kmesh-retained-pre-start-interrupt"
    original_start = threading.Thread.start

    def terminal_call() -> None:
        nonlocal call_count
        call_count += 1

    def interrupt_first_start(worker: threading.Thread) -> None:
        if worker.name == thread_name:
            start_attempts.append(worker)
            if len(start_attempts) == 1:
                raise KeyboardInterrupt("injected before retained worker start")
        original_start(worker)

    monkeypatch.setattr(threading.Thread, "start", interrupt_first_start)
    retained = RetainedCall(terminal_call, thread_name=thread_name)

    with pytest.raises(KeyboardInterrupt, match="before retained worker start"):
        retained.run()

    assert not retained.complete
    assert call_count == 0

    retained.run()
    retained.run()

    assert retained.complete
    assert call_count == 1
    assert len(start_attempts) == 2
    assert start_attempts[0] is start_attempts[1]


def test_manager_retries_builder_after_final_instance_pop_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance_id = "kmesh-builder-post-instance-pop-interrupt"
    engine = MagicMock(spec=LMCacheEngine)
    config = MagicMock()
    metadata = MagicMock()
    stat_logger = MagicMock()

    class _InterruptAfterPop(dict[str, LMCacheEngine]):
        interrupted = False

        def pop(  # type: ignore[override]
            self,
            key: str,
            default: LMCacheEngine | None = None,
        ) -> LMCacheEngine | None:
            value = super().pop(key, default)
            if key == instance_id and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt("injected after final instance pop")
            return value

    assert instance_id not in LMCacheEngineBuilder._instances
    interrupted_instances = _InterruptAfterPop(LMCacheEngineBuilder._instances)
    interrupted_instances[instance_id] = engine
    monkeypatch.setattr(LMCacheEngineBuilder, "_instances", interrupted_instances)
    LMCacheEngineBuilder._cfgs[instance_id] = config
    LMCacheEngineBuilder._metadatas[instance_id] = metadata
    LMCacheEngineBuilder._stat_loggers[instance_id] = stat_logger

    factory = MagicMock()
    factory.get_or_create_metadata.return_value = metadata
    factory.get_engine_instance_id.return_value = instance_id
    factory.get_or_create_lmcache_engine.return_value = engine
    factory.maybe_create_lookup_client.return_value = None
    factory.maybe_create_lookup_server.return_value = None
    factory.maybe_create_offload_server.return_value = None
    factory.maybe_create_runtime_plugin_launcher.return_value = None
    factory.maybe_create_internal_api_server.return_value = None
    manager = LMCacheManager(config, factory)

    try:
        with (
            patch(
                "lmcache.v1.cache_engine.LMCStatsMonitor.DestroyInstance"
            ) as destroy_monitor,
            patch(
                "lmcache.v1.compute.blend.utils.LMCBlenderBuilder.drop"
            ) as drop_blender,
            patch(
                "lmcache.v1.compute.models.utils.VLLMModelTracker.unregister_model"
            ) as unregister_model,
        ):
            with pytest.raises(KeyboardInterrupt, match="final instance pop"):
                manager._release_engine_owner()

            assert interrupted_instances.interrupted
            assert instance_id not in LMCacheEngineBuilder._instances
            assert instance_id not in LMCacheEngineBuilder._cfgs
            assert instance_id not in LMCacheEngineBuilder._metadatas
            assert instance_id not in LMCacheEngineBuilder._stat_loggers
            assert LMCacheEngineBuilder._destroyed_engines.get(engine, False)
            assert instance_id in LMCacheEngineBuilder._destroy_owners
            assert manager._engine_owner_registered
            assert LMCacheManager._engine_owner_counts[engine] == 1

            manager._release_engine_owner()

            assert instance_id not in LMCacheEngineBuilder._destroy_owners
            assert not manager._engine_owner_registered
            assert engine not in LMCacheManager._engine_owner_counts
            assert engine.close.call_count == 1
            stat_logger.shutdown.assert_called_once_with()
            destroy_monitor.assert_called_once_with()
            drop_blender.assert_called_once_with(instance_id)
            unregister_model.assert_called_once_with(instance_id)
    finally:
        LMCacheEngineBuilder._instances.pop(instance_id, None)
        LMCacheEngineBuilder._cfgs.pop(instance_id, None)
        LMCacheEngineBuilder._metadatas.pop(instance_id, None)
        LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)
        LMCacheEngineBuilder._destroy_owners.pop(instance_id, None)
        LMCacheEngineBuilder._destroyed_engines.pop(engine, None)
        with LMCacheManager._engine_ownership_lock:
            LMCacheManager._engine_ownership_receipts.pop(engine, None)


def test_gpu_registration_interrupt_drains_published_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = VLLMBufferLayerwiseGPUConnector.__new__(VLLMBufferLayerwiseGPUConnector)
    state = cast(Any, connector)
    state.num_layers = 1
    state.hidden_dim_size = 1
    state.dtype = torch.float32
    state.cache_positions = False
    state.kvcaches = [torch.zeros((2, 1, 1))]
    state.initialize_kvcaches_ptr = lambda **kwargs: None
    state._lazy_initialize_buffer = lambda kvcaches: None
    state.get_shape = lambda num_tokens: torch.Size([2, num_tokens, 1])
    state.store_stream = object()
    state.gpu_buffer_allocator = object()
    original_register = connector._register_gpu_store_cleanup

    def register_then_interrupt(owner: Any) -> None:
        original_register(owner)
        raise KeyboardInterrupt("injected after GPU cleanup publication")

    monkeypatch.setattr(
        connector,
        "_register_gpu_store_cleanup",
        register_then_interrupt,
    )
    storer = connector.batched_from_gpu(
        [[]],
        [0],
        [1],
        slot_mapping=torch.tensor([0]),
    )

    with pytest.raises(KeyboardInterrupt, match="GPU cleanup publication"):
        next(storer)

    assert connector.gpu_cleanup_owner_count() == 0
    assert connector.pending_gpu_cleanup_count() == 0


def test_store_dead_lease_recovers_first_finalize_line_interrupt() -> None:
    class _NoopGPU:
        def drain_pending_gpu_cleanups(self) -> None:
            return

        def pending_gpu_cleanup_count(self) -> int:
            return 0

    class _NoopStorage:
        def drain_pending_put_cleanups(self) -> None:
            return

        def pending_put_cleanup_count(self) -> int:
            return 0

    engine = LMCacheEngine.__new__(LMCacheEngine)
    lease = _StoreOperationLease()
    owner = _StoreCleanupOwner(cast(Any, _NoopGPU()), cast(Any, _NoopStorage()), lease)
    engine._register_store_cleanup(owner)

    # A live yielded/in-flight operation is never promoted by a public drain.
    engine.drain_pending_store_cleanups()
    assert engine._active_store_cleanup_count() == 1
    assert engine.pending_store_cleanup_count() == 0
    assert engine._store_cleanup_owner_count() == 1

    del lease
    helper = LMCacheEngine.finalize_store_cleanup
    interrupt_line = _nth_source_line(helper, "owner.mark_pending()", 0)
    trace, was_triggered = _interrupt_on_line(
        helper,
        interrupt_line,
        "injected on first store finalize line",
    )
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="first store finalize line"):
            engine.finalize_store_cleanup(owner)
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert not owner.pending
    assert not owner.operation_alive
    assert engine._active_store_cleanup_count() == 0
    assert engine.pending_store_cleanup_count() == 1

    engine.drain_pending_store_cleanups()
    assert engine.pending_store_cleanup_count() == 0
    assert engine._store_cleanup_owner_count() == 0

    # Exercise the real generator frame: it kills its strong lease before the
    # helper call, so interruption on the helper's first line is recoverable.
    state = cast(Any, engine)
    state.storage_manager = _NoopStorage()
    state.gpu_connector = _NoopGPU()
    state.stats_monitor = MagicMock()
    state.stats_monitor.on_store_request.return_value = 1
    state._require_explicit_spans_supported = lambda spans: None
    state.is_healthy = lambda: True
    state._get_req_id = lambda kwargs: None
    state._log_kvcache_for_check = lambda **kwargs: None
    state.num_layers = 1

    def fail_after_registration() -> bool:
        raise RuntimeError("injected after store registration")

    state.is_frozen = fail_after_registration
    trace, was_triggered = _interrupt_on_line(
        helper,
        interrupt_line,
        "injected on first live-store finalize line",
    )
    storer = engine.store_layer([1])
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="live-store finalize line"):
            next(storer)
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert engine._active_store_cleanup_count() == 0
    assert engine.pending_store_cleanup_count() == 1
    engine.drain_pending_store_cleanups()
    assert engine._store_cleanup_owner_count() == 0


def test_put_dead_lease_recovers_first_finalize_line_interrupt() -> None:
    manager = StorageManager.__new__(StorageManager)
    lease = _PutOperationLease()
    owner = _PutRefCleanupOwner([], operation_lease=lease)
    assert manager._register_put_cleanup(owner, None)

    manager.drain_pending_put_cleanups()
    assert manager._active_put_cleanup_count() == 1
    assert manager.pending_put_cleanup_count() == 0
    assert manager._put_cleanup_owner_count() == 1

    del lease
    helper = StorageManager.finalize_put_cleanup
    interrupt_line = _nth_source_line(helper, "owner.mark_pending()", 0)
    trace, was_triggered = _interrupt_on_line(
        helper,
        interrupt_line,
        "injected on first put finalize line",
    )
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="first put finalize line"):
            manager.finalize_put_cleanup(owner)
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert not owner.pending
    assert not owner.operation_alive
    assert manager._active_put_cleanup_count() == 0
    assert manager.pending_put_cleanup_count() == 1

    manager.drain_pending_put_cleanups()
    assert manager.pending_put_cleanup_count() == 0
    assert manager._put_cleanup_owner_count() == 0

    # The real batched_put frame clears its lease before entering the helper.
    cast(Any, manager).allocator_backend = None
    trace, was_triggered = _interrupt_on_line(
        helper,
        interrupt_line,
        "injected on first live-put finalize line",
    )
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="live-put finalize line"):
            manager.batched_put([], [])
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert manager._active_put_cleanup_count() == 0
    assert manager.pending_put_cleanup_count() == 1
    manager.drain_pending_put_cleanups()
    assert manager._put_cleanup_owner_count() == 0


def test_gpu_dead_lease_recovers_first_finalize_line_interrupt() -> None:
    connector = VLLMBufferLayerwiseGPUConnector.__new__(VLLMBufferLayerwiseGPUConnector)
    lease = _GPUStoreOperationLease()
    owner = _GPUStoreCleanupOwner(MagicMock(), lease)
    connector._register_gpu_store_cleanup(owner)

    connector.drain_pending_gpu_cleanups()
    assert owner.operation_alive
    assert connector.pending_gpu_cleanup_count() == 0
    assert connector.gpu_cleanup_owner_count() == 1

    del lease
    helper = VLLMBufferLayerwiseGPUConnector.finalize_gpu_store_cleanup
    interrupt_line = _nth_source_line(helper, "owner.mark_pending()", 0)
    trace, was_triggered = _interrupt_on_line(
        helper,
        interrupt_line,
        "injected on first GPU finalize line",
    )
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="first GPU finalize line"):
            connector.finalize_gpu_store_cleanup(owner)
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert not owner.pending
    assert not owner.operation_alive
    assert connector.pending_gpu_cleanup_count() == 1

    connector.drain_pending_gpu_cleanups()
    assert connector.pending_gpu_cleanup_count() == 0
    assert connector.gpu_cleanup_owner_count() == 0

    # Exercise the real GPU generator frame with a post-registration allocator
    # failure; its dead lease remains recoverable when helper entry is cut.
    state = cast(Any, connector)
    state.num_layers = 1
    state.hidden_dim_size = 1
    state.dtype = torch.float32
    state.cache_positions = False
    state.kvcaches = [torch.zeros((2, 1, 1))]
    state.initialize_kvcaches_ptr = lambda **kwargs: None
    state._lazy_initialize_buffer = lambda kvcaches: None
    state.get_shape = lambda num_tokens: torch.Size([2, num_tokens, 1])
    state.store_stream = object()
    state.gpu_buffer_allocator = object()
    trace, was_triggered = _interrupt_on_line(
        helper,
        interrupt_line,
        "injected on first live-GPU finalize line",
    )
    storer = connector.batched_from_gpu(
        [[]],
        [0],
        [1],
        slot_mapping=torch.tensor([0]),
    )
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="live-GPU finalize line"):
            next(storer)
    finally:
        sys.settrace(None)

    assert was_triggered()
    assert connector.pending_gpu_cleanup_count() == 1
    connector.drain_pending_gpu_cleanups()
    assert connector.gpu_cleanup_owner_count() == 0
