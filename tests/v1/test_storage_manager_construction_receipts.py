# SPDX-License-Identifier: Apache-2.0
"""Crash-safe ownership tests for target StorageManager construction."""

# Standard
from types import SimpleNamespace
from typing import Any, Callable, cast
from unittest.mock import MagicMock
import asyncio
import ctypes
import inspect
import sys
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, start_loop_in_thread_with_exceptions
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.lifecycle import RetainedCall, _run_under_interrupt_safe_rlock
from lmcache.v1.memory_allocators.mixed_memory_allocator import (
    MixedMemoryAllocator,
)
from lmcache.v1.memory_allocators.tensor_memory_allocator import (
    TensorMemoryAllocator,
)
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.batched_message_sender import BatchedMessageSender
from lmcache.v1.storage_backend.local_cpu_backend import (
    LocalCPUBackend,
    _LocalCPUBackendCloseOwner,
    _LocalCPUCacheAdmissionReceipt,
    _LocalCPUCacheReleaseReceipt,
)
from lmcache.v1.storage_backend.storage_manager import StorageManager
from tests.v1.instruction_monitoring import (
    InstructionInterruption,
    instruction_offset,
    requires_instruction_monitoring,
)
import lmcache.v1.memory_management as memory_management


class _ConstructorAbort(BaseException):
    pass


class _TrackedEventLoop:
    def __init__(self) -> None:
        self.close_calls = 0
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    def is_running(self) -> bool:
        return False

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _ReceiptMemoryObj:
    """Minimal exact-once cache reference used by close interruption tests."""

    def __init__(self) -> None:
        self.refdowns = 0
        self._applied: set[object] = set()
        self._completed: set[object] = set()

    def make_ref_count_down_receipt(
        self,
    ) -> memory_management.RefCountDownReceipt:
        return memory_management.RefCountDownReceipt(0, self, object())

    def ref_count_down_once(
        self,
        receipt: memory_management.RefCountDownReceipt,
    ) -> None:
        assert receipt.allocation_epoch == 0
        assert receipt.allocation_token is self
        if receipt.token not in self._applied:
            self.refdowns += 1
            self._applied.add(receipt.token)
        self._completed.add(receipt.token)

    def ref_count_down_receipt_complete(
        self,
        receipt: memory_management.RefCountDownReceipt,
    ) -> bool:
        assert receipt.allocation_epoch == 0
        assert receipt.allocation_token is self
        return receipt.token in self._completed


def _tensor_memory_obj(
    address: int = 0,
    ref_count: int = 1,
) -> memory_management.TensorMemoryObj:
    metadata = memory_management.MemoryObjMetadata(
        shape=torch.Size([1]),
        dtype=torch.uint8,
        address=address,
        phy_size=1,
        ref_count=ref_count,
        fmt=memory_management.MemoryFormat.BINARY,
    )
    return memory_management.TensorMemoryObj(
        raw_data=torch.zeros(1, dtype=torch.uint8),
        metadata=metadata,
        parent_allocator=None,
    )


def _source_line(function: Callable[..., Any], source: str) -> int:
    lines, start = inspect.getsourcelines(function)
    return next(
        start + offset for offset, text in enumerate(lines) if text.strip() == source
    )


def _post_init_engine_shell() -> LMCacheEngine:
    engine = LMCacheEngine.__new__(LMCacheEngine)
    state = cast(Any, engine)
    state.post_inited = False
    state.config = MagicMock()
    state.config.get_lookup_server_worker_ids.return_value = []
    state.metadata = MagicMock()
    state.metadata.use_mla = False
    state.metadata.world_size = 1
    state.metadata.worker_id = 0
    state.metadata.is_first_rank.return_value = True
    state.lmcache_worker = None
    state.use_layerwise = True
    state.save_only_first_rank = False
    state.event_manager = MagicMock()
    state.storage_manager = None
    state._storage_manager_init_owner = None
    state.hidden_state_store = None
    state.gpu_connector = None
    return engine


@requires_instruction_monitoring
@pytest.mark.parametrize("stage_index", range(3))
@pytest.mark.parametrize("after_call", [False, True])
def test_local_cpu_close_resumes_each_exact_terminal_stage(
    after_call: bool,
    stage_index: int,
) -> None:
    """An outer interruption cannot skip or repeat a LocalCPU close stage."""

    class _CountingClose:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    backend = LocalCPUBackend.__new__(LocalCPUBackend)
    LocalCPUBackend._prepare_partial_state(backend)
    sender = _CountingClose()
    allocator = _CountingClose()
    backend.batched_msg_sender = cast(Any, sender)
    backend.memory_allocator = cast(Any, allocator)
    cache_close_calls = 0
    clear_for_close = backend._clear_for_close

    def close_cache() -> None:
        nonlocal cache_close_calls
        cache_close_calls += 1
        clear_for_close()

    cast(Any, backend)._clear_for_close = close_cache

    manager = StorageManager.__new__(StorageManager)
    StorageManager._prepare_partial_state(
        manager,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    manager.storage_backends["LocalCPUBackend"] = backend

    target = _LocalCPUBackendCloseOwner.run
    injected = _ConstructorAbort(
        f"LocalCPU close stage {stage_index} publication interrupted"
    )
    interrupt = InstructionInterruption(
        target,
        instruction_offset(
            target,
            "CALL",
            occurrence=stage_index,
            after=after_call,
        ),
        injected,
        thread_name="lmcache-close-LocalCPUBackend",
    )

    with pytest.raises(_ConstructorAbort) as caught:
        with interrupt:
            manager.close()

    interrupt.assert_fired_once()
    assert caught.value is injected

    # StorageManager retries the outer backend dispatcher. The backend's
    # retained per-stage receipts resume the first missing stage and prevent
    # every completed close-like operation from running twice.
    manager.close()
    manager.close()

    assert sender.close_calls == 1
    assert cache_close_calls == 1
    assert allocator.close_calls == 1
    close_owner = manager._storage_backend_close_owners[id(backend)]
    assert close_owner.complete
    assert backend._close_owner is not None
    assert backend._close_owner.complete


@requires_instruction_monitoring
@pytest.mark.parametrize("after_call", [False, True])
@pytest.mark.parametrize("call_occurrence", [1, 3])
def test_local_cpu_close_receipts_cover_pop_to_refdown_window(
    after_call: bool,
    call_occurrence: int,
) -> None:
    """An empty cache cannot hide an unconsumed cache reference."""
    backend = LocalCPUBackend.__new__(LocalCPUBackend)
    LocalCPUBackend._prepare_partial_state(backend)
    backend.use_hot = True
    memory_obj = _ReceiptMemoryObj()
    backend.hot_cache[cast(Any, "terminal-entry")] = cast(Any, memory_obj)

    manager = StorageManager.__new__(StorageManager)
    StorageManager._prepare_partial_state(
        manager,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    manager.storage_backends["LocalCPUBackend"] = backend

    target = _LocalCPUCacheReleaseReceipt._run_locked
    injected = _ConstructorAbort("LocalCPU cache release interrupted")
    interrupt = InstructionInterruption(
        target,
        instruction_offset(
            target,
            "CALL",
            occurrence=call_occurrence,
            after=after_call,
        ),
        injected,
        thread_name="lmcache-local-cpu-close-cache",
    )

    with pytest.raises(_ConstructorAbort) as caught:
        with interrupt:
            manager.close()

    interrupt.assert_fired_once()
    assert caught.value is injected

    manager.close()
    manager.close()

    assert memory_obj.refdowns == 1
    assert not backend.hot_cache
    assert not backend._cache_release_receipts
    close_owner = manager._storage_backend_close_owners[id(backend)]
    assert close_owner.complete
    assert backend._close_owner is not None
    assert backend._close_owner.complete


def test_local_cpu_close_receipts_release_real_tensor_entries_once() -> None:
    """The target TensorMemoryObj receipt composes across multiple entries."""
    backend = LocalCPUBackend.__new__(LocalCPUBackend)
    LocalCPUBackend._prepare_partial_state(backend)
    backend.use_hot = True
    memory_objs = [_tensor_memory_obj(0), _tensor_memory_obj(1)]
    backend.hot_cache[cast(Any, "entry-0")] = memory_objs[0]
    backend.hot_cache[cast(Any, "entry-1")] = memory_objs[1]

    manager = StorageManager.__new__(StorageManager)
    StorageManager._prepare_partial_state(
        manager,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    manager.storage_backends["LocalCPUBackend"] = backend

    manager.close()
    manager.close()

    assert [memory_obj.get_ref_count() for memory_obj in memory_objs] == [0, 0]
    assert not backend.hot_cache
    assert not backend._cache_release_receipts
    assert backend._close_owner is not None
    assert backend._close_owner.complete


@requires_instruction_monitoring
@pytest.mark.parametrize("after_call", [False, True])
def test_local_cpu_put_admission_receipt_survives_increment_publication_cut(
    after_call: bool,
) -> None:
    """A failed put cannot orphan its cache-side reference increment."""
    backend = LocalCPUBackend.__new__(LocalCPUBackend)
    LocalCPUBackend._prepare_partial_state(backend)
    backend.use_hot = True
    backend.cache_policy = MagicMock()

    manager = StorageManager.__new__(StorageManager)
    StorageManager._prepare_partial_state(
        manager,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    manager.allocator_backend = backend
    manager.storage_backends["LocalCPUBackend"] = backend

    key = CacheEngineKey("receipt-test", 1, 0, 1, torch.bfloat16)
    memory_obj = _tensor_memory_obj()
    target = _LocalCPUCacheAdmissionReceipt._run_locked
    injected = _ConstructorAbort("LocalCPU cache admission interrupted")
    interrupt = InstructionInterruption(
        target,
        instruction_offset(target, "CALL", occurrence=2, after=after_call),
        injected,
    )

    with pytest.raises(_ConstructorAbort) as caught:
        with interrupt:
            manager.batched_put([key], [memory_obj])

    interrupt.assert_fired_once()
    assert caught.value is injected
    assert manager.pending_put_cleanup_count() == 0

    manager.close()
    manager.close()

    assert memory_obj.get_ref_count() == 0
    assert not backend.hot_cache
    assert not backend._cache_admission_receipts
    assert not backend._cache_release_receipts
    close_owner = manager._storage_backend_close_owners[id(backend)]
    assert close_owner.complete
    assert backend._close_owner is not None
    assert backend._close_owner.complete


@requires_instruction_monitoring
def test_local_cpu_remove_receipt_survives_pop_to_refdown_cut() -> None:
    """A failed ordinary eviction remains visible to terminal cleanup."""
    backend = LocalCPUBackend.__new__(LocalCPUBackend)
    LocalCPUBackend._prepare_partial_state(backend)
    backend.use_hot = True
    backend.cache_policy = MagicMock()
    key = CacheEngineKey("receipt-test", 1, 0, 2, torch.bfloat16)
    memory_obj = _tensor_memory_obj()
    backend.hot_cache[key] = memory_obj

    target = _LocalCPUCacheReleaseReceipt._run_locked
    injected = _ConstructorAbort("LocalCPU ordinary remove interrupted")
    interrupt = InstructionInterruption(
        target,
        instruction_offset(target, "CALL", occurrence=1, after=True),
        injected,
    )

    with pytest.raises(_ConstructorAbort) as caught:
        with interrupt:
            backend.remove(key)

    interrupt.assert_fired_once()
    assert caught.value is injected
    assert not backend.hot_cache
    assert len(backend._cache_release_receipts) == 1

    backend.close()
    backend.close()

    assert memory_obj.get_ref_count() == 0
    assert not backend._cache_release_receipts
    assert backend._close_owner is not None
    assert backend._close_owner.complete


@requires_instruction_monitoring
def test_local_cpu_release_completion_interrupt_does_not_leak_cpu_lock() -> None:
    """Completion result publication occurs before leaving the lock context."""
    backend = LocalCPUBackend.__new__(LocalCPUBackend)
    LocalCPUBackend._prepare_partial_state(backend)
    key = CacheEngineKey("receipt-test", 1, 0, 3, torch.bfloat16)
    memory_obj = _tensor_memory_obj()
    backend.hot_cache[key] = memory_obj
    release = _LocalCPUCacheReleaseReceipt(key, memory_obj)
    backend._cache_release_receipts.append(release)
    release.run(backend)

    target = _LocalCPUCacheReleaseReceipt._complete_locked
    injected = _ConstructorAbort("LocalCPU release completion interrupted")
    interrupt = InstructionInterruption(
        target,
        instruction_offset(target, "CALL", occurrence=2, after=True),
        injected,
    )

    with pytest.raises(_ConstructorAbort) as caught:
        with interrupt:
            release.complete(backend)

    interrupt.assert_fired_once()
    assert caught.value is injected

    lock_acquired = threading.Event()

    def acquire_from_peer() -> None:
        if backend.cpu_lock.acquire(timeout=1):
            lock_acquired.set()
            backend.cpu_lock.release()

    peer = threading.Thread(target=acquire_from_peer)
    peer.start()
    peer.join(timeout=2)

    assert not peer.is_alive()
    assert lock_acquired.is_set()
    assert backend._cache_clear_complete()
    assert not backend._cache_release_receipts


@requires_instruction_monitoring
@pytest.mark.parametrize("nested", [False, True])
def test_interrupt_safe_rlock_closes_normal_exit_gap(nested: bool) -> None:
    """Normal cleanup restores only the recursion level acquired here."""
    lock = threading.RLock()
    if nested:
        lock.acquire()

    target = _run_under_interrupt_safe_rlock
    injected = _ConstructorAbort("RLock normal-exit cleanup interrupted")
    interrupt = InstructionInterruption(
        target,
        instruction_offset(
            target,
            "LOAD_FAST",
            argval="lock",
            occurrence=2,
        ),
        injected,
    )

    with pytest.raises(_ConstructorAbort) as caught:
        with interrupt:
            target(lock, lambda: None)

    interrupt.assert_fired_once()
    assert caught.value is injected

    if nested:
        lock.release()

    lock_acquired = threading.Event()

    def acquire_from_peer() -> None:
        if lock.acquire(timeout=1):
            lock_acquired.set()
            lock.release()

    peer = threading.Thread(target=acquire_from_peer)
    peer.start()
    peer.join(timeout=2)

    assert not peer.is_alive()
    assert lock_acquired.is_set()


@requires_instruction_monitoring
@pytest.mark.parametrize("stage_name", ["sender", "cache", "allocator"])
@pytest.mark.parametrize("after_call", [False, True])
def test_local_cpu_close_resumes_inside_each_resource_wrapper(
    after_call: bool,
    stage_name: str,
) -> None:
    """Callee-side state closes each nested resource-return window."""

    class _StoppedThread:
        def is_alive(self) -> bool:
            return False

    backend = LocalCPUBackend.__new__(LocalCPUBackend)
    LocalCPUBackend._prepare_partial_state(backend)

    sender = BatchedMessageSender.__new__(BatchedMessageSender)
    sender.running = True
    sender.thread = cast(Any, _StoppedThread())
    sender_close_calls = 0

    def close_sender() -> None:
        nonlocal sender_close_calls
        sender_close_calls += 1
        sender.running = False

    sender.close = close_sender  # type: ignore[method-assign]
    backend.batched_msg_sender = sender

    memory_obj = _ReceiptMemoryObj()
    backend.hot_cache["terminal-entry"] = cast(Any, memory_obj)

    allocator = MixedMemoryAllocator.__new__(MixedMemoryAllocator)
    MixedMemoryAllocator._prepare_partial_state(allocator)
    allocator._unregistered = False
    arena = torch.zeros(8, dtype=torch.uint8)
    native_release_calls = 0

    def release_arena() -> None:
        nonlocal native_release_calls
        native_release_calls += 1

    allocator._cpu_memory_allocation_receipt.retain_buffer(
        arena,
        release_arena,
    )
    allocator.buffer = arena
    allocator_close_calls = 0

    def close_allocator() -> None:
        nonlocal allocator_close_calls
        allocator_close_calls += 1
        MixedMemoryAllocator.close(allocator)

    allocator.close = close_allocator  # type: ignore[method-assign]
    backend.memory_allocator = allocator

    manager = StorageManager.__new__(StorageManager)
    StorageManager._prepare_partial_state(
        manager,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    manager.storage_backends["LocalCPUBackend"] = backend

    targets = {
        "sender": (
            LocalCPUBackend._close_batched_message_sender,
            1,
            "lmcache-local-cpu-close-sender",
        ),
        "cache": (
            LocalCPUBackend._clear_for_close,
            0,
            "lmcache-local-cpu-close-cache",
        ),
        "allocator": (
            LocalCPUBackend._close_memory_allocator,
            1,
            "lmcache-local-cpu-close-allocator",
        ),
    }
    target, occurrence, thread_name = targets[stage_name]
    injected = _ConstructorAbort(f"LocalCPU {stage_name} resource return interrupted")
    interrupt = InstructionInterruption(
        target,
        instruction_offset(
            target,
            "CALL",
            occurrence=occurrence,
            after=after_call,
        ),
        injected,
        thread_name=thread_name,
    )

    with pytest.raises(_ConstructorAbort) as caught:
        with interrupt:
            manager.close()

    interrupt.assert_fired_once()
    assert caught.value is injected

    manager.close()
    manager.close()

    assert sender_close_calls == 1
    assert memory_obj.refdowns == 1
    assert allocator_close_calls == 1
    assert native_release_calls == 1
    assert not sender.running
    assert not backend.hot_cache
    assert allocator._unregistered
    close_owner = manager._storage_backend_close_owners[id(backend)]
    assert close_owner.complete
    assert backend._close_owner is not None
    assert backend._close_owner.complete


@requires_instruction_monitoring
@pytest.mark.parametrize("after_store", [False, True])
def test_event_loop_store_interrupt_has_exact_terminal_owner(
    monkeypatch: pytest.MonkeyPatch,
    after_store: bool,
) -> None:
    """The live loop is either locally closed or retained by the manager."""
    manager = StorageManager.__new__(StorageManager)
    StorageManager._prepare_partial_state(
        manager,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    loop = _TrackedEventLoop()
    monkeypatch.setattr(asyncio, "new_event_loop", lambda: cast(Any, loop))
    target = StorageManager._create_and_publish_event_loop
    injected = _ConstructorAbort("event-loop publication interrupted")
    interrupt = InstructionInterruption(
        target,
        instruction_offset(
            target,
            "STORE_ATTR",
            argval="loop",
            after=after_store,
        ),
        injected,
    )

    with pytest.raises(_ConstructorAbort) as caught:
        with interrupt:
            target(manager)

    interrupt.assert_fired_once()
    assert caught.value is injected
    if after_store:
        assert manager.loop is loop
        assert loop.close_calls == 0
        manager.close()
    else:
        assert manager.loop is None
    assert loop.close_calls == 1
    assert loop.closed


def test_post_init_interrupt_retains_manager_and_real_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller KI cannot orphan the constructor worker or its loop thread."""
    engine = _post_init_engine_shell()
    loop_thread_started = threading.Event()
    allow_constructor_return = threading.Event()
    captured: dict[str, Any] = {}

    def controlled_init(
        manager: StorageManager,
        config: Any,
        metadata: Any,
        event_manager: Any,
        lmcache_worker: Any = None,
        async_lookup_server: Any = None,
    ) -> None:
        StorageManager._prepare_partial_state(
            manager,
            config,
            metadata,
            event_manager,
            lmcache_worker,
            async_lookup_server,
        )
        manager.loop = asyncio.new_event_loop()
        manager.thread = threading.Thread(
            target=start_loop_in_thread_with_exceptions,
            args=(manager.loop,),
            name="receipt-storage-manager-loop",
        )
        captured["manager"] = manager
        captured["thread"] = manager.thread
        manager.thread.start()
        loop_thread_started.set()
        assert allow_constructor_return.wait(timeout=5.0)

    monkeypatch.setattr(StorageManager, "__init__", controlled_init)
    wait_line = _source_line(RetainedCall.run, "attempt.done.wait()")
    interrupted = False

    def interrupt_wait(frame: Any, event: str, arg: Any) -> Any:
        del arg
        nonlocal interrupted
        if (
            not interrupted
            and frame.f_code is RetainedCall.run.__code__
            and event == "line"
            and frame.f_lineno == wait_line
        ):
            assert loop_thread_started.wait(timeout=5.0)
            interrupted = True
            sys.settrace(None)
            raise KeyboardInterrupt("caller left StorageManager construction")
        return interrupt_wait

    sys.settrace(interrupt_wait)
    try:
        with pytest.raises(KeyboardInterrupt, match="caller left"):
            engine.post_init()
    finally:
        sys.settrace(None)

    manager = captured["manager"]
    loop_thread = captured["thread"]
    assert interrupted
    assert engine.storage_manager is manager
    assert engine._storage_manager_init_owner is not None
    assert loop_thread.is_alive()
    assert not loop_thread.daemon

    init_attempt = engine._storage_manager_init_owner._call._attempt
    assert init_attempt is not None
    assert init_attempt.worker is not None
    assert init_attempt.worker.is_alive()

    allow_constructor_return.set()
    engine.close()

    init_attempt.worker.join(timeout=5.0)
    loop_thread.join(timeout=5.0)
    assert not init_attempt.worker.is_alive()
    assert not loop_thread.is_alive()
    assert manager.loop is not None
    assert manager.loop.is_closed()


def test_constructor_abort_retains_local_mixed_tensor_chain_for_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every target owner remains reachable after nested constructor failure."""
    config = SimpleNamespace(
        extra_config={},
        enable_pd=False,
        local_cpu=True,
        max_local_cpu_size=1,
        enable_p2p=False,
        local_disk=False,
        max_local_disk_size=0,
        gds_path=None,
        maru_path=None,
        remote_storage_plugins=[],
        remote_url=None,
        storage_plugins=[],
        lmcache_instance_id="construction-receipt",
    )
    metadata = SimpleNamespace(role="worker", worker_id=0)
    manager = StorageManager.__new__(StorageManager)
    StorageManager._prepare_partial_state(
        manager,
        cast(Any, config),
        cast(Any, metadata),
        MagicMock(),
    )
    arena = torch.zeros(64, dtype=torch.uint8)
    freed: list[torch.Tensor] = []

    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda *args, **kwargs: arena,
    )
    monkeypatch.setattr(
        memory_management,
        "_free_cpu_memory",
        lambda buffer, *args, **kwargs: freed.append(buffer),
    )

    def abort_tensor_init(
        tensor_allocator: TensorMemoryAllocator,
        tensor: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        del kwargs
        tensor_allocator.buffer = tensor
        raise _ConstructorAbort("after Tensor owner publication")

    monkeypatch.setattr(TensorMemoryAllocator, "__init__", abort_tensor_init)

    def construct_target_backend(
        backend: LocalCPUBackend,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        LocalCPUBackend._prepare_partial_state(backend)
        backend._construct_mixed_allocator(64)

    monkeypatch.setattr(LocalCPUBackend, "__init__", construct_target_backend)

    with pytest.raises(_ConstructorAbort, match="Tensor owner publication"):
        CreateStorageBackends(
            cast(Any, config),
            cast(Any, metadata),
            cast(Any, None),
            dst_device="cpu",
            retain_backend=manager._retain_constructing_backend,
        )

    backend = manager.storage_backends["LocalCPUBackend"]
    assert isinstance(backend, LocalCPUBackend)
    allocator = backend.memory_allocator
    assert isinstance(allocator, MixedMemoryAllocator)
    assert allocator.buffer is arena
    assert isinstance(allocator.pin_allocator, TensorMemoryAllocator)
    assert allocator.pin_allocator.buffer is arena
    assert not allocator._unregistered

    manager.close()
    manager.close()

    assert len(freed) == 1
    assert freed[0] is arena
    assert allocator._unregistered
    close_owner = manager._storage_backend_close_owners[id(backend)]
    assert close_owner.complete


def test_native_arena_freed_when_tensor_wrapper_construction_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The raw pinned pointer remains locally owned until Tensor publication."""
    allocation = ctypes.create_string_buffer(32)
    pointer = ctypes.addressof(allocation)
    freed: list[int] = []

    class _ResolvedAllocator:
        def alloc(self) -> int:
            return pointer

        def free(self, owned_pointer: int) -> None:
            freed.append(owned_pointer)

    monkeypatch.setattr(
        memory_management,
        "_resolve_pinned_alloc_free",
        lambda *args, **kwargs: _ResolvedAllocator(),
    )

    def abort_frombuffer(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise _ConstructorAbort("torch wrapper failed")

    monkeypatch.setattr(memory_management.torch, "frombuffer", abort_frombuffer)

    with pytest.raises(_ConstructorAbort, match="torch wrapper failed"):
        memory_management._allocate_cpu_memory(32)

    assert freed == [pointer]
