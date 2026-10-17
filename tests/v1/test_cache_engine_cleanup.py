# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for LMCacheEngine.cleanup_memory_objs.

Some storage backends (LocalCPU, LocalDisk, P2P, Maru) pin the MemoryObj
they return via async prefetch, while others (Remote, Nixl, plugin tiers)
do not. cleanup_memory_objs must skip unpin for the latter to avoid
driving pin_count below zero.
"""

# Standard
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch
import logging
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.usage_telemetry import UsageContext
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventStatus
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.pin_monitor import PinMonitor
import lmcache.v1.cache_engine as cache_engine_module

# Local
from .utils import create_test_memory_obj


class _StoreRaceMemory:
    """Small caller-owned object for layerwise-store lifecycle tests."""

    def __init__(self) -> None:
        self.release_count = 0

    def get_size(self) -> int:
        return 1

    def ref_count_down(self) -> None:
        self.release_count += 1


class _StoreRaceStorage:
    """Storage collaborator that records terminal teardown ordering."""

    def __init__(self) -> None:
        self.allocations: list[_StoreRaceMemory] = []
        self.put_calls = 0
        self.drain_calls = 0
        self.drain_after_close_calls = 0
        self.close_calls = 0
        self.close_error: BaseException | None = None

    def contains(self, key: CacheEngineKey, search_range: Any) -> None:
        del key, search_range
        return None

    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        **kwargs: Any,
    ) -> list[MemoryObj]:
        del shape, dtype, kwargs
        allocated = [_StoreRaceMemory() for _ in range(batch_size)]
        self.allocations.extend(allocated)
        return cast(list[MemoryObj], allocated)

    def batched_put(
        self,
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        **kwargs: Any,
    ) -> None:
        del keys
        ownership_receipt = kwargs.pop("ownership_receipt", None)
        if ownership_receipt is not None:
            ownership_receipt._adopt()
        del kwargs
        self.put_calls += 1
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()

    def drain_pending_put_cleanups(self) -> None:
        self.drain_calls += 1
        if self.close_calls:
            self.drain_after_close_calls += 1

    def pending_put_cleanup_count(self) -> int:
        return 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            error = self.close_error
            self.close_error = None
            raise error


class _StoreRaceGPU:
    """Layerwise GPU collaborator with an optional terminal-drain barrier."""

    def __init__(
        self,
        drain_entered: threading.Event | None = None,
        release_drain: threading.Event | None = None,
    ) -> None:
        self.drain_entered = drain_entered
        self.release_drain = release_drain
        self.drain_calls = 0
        self.generator_started = False
        self.generator_closed = False
        self.layer_advances = 0

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size((num_tokens,))

    def batched_from_gpu(
        self,
        memory_objs: list[list[MemoryObj]],
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> Generator[None, None, None]:
        del starts, ends, kwargs
        self.generator_started = True
        try:
            yield
            for _layer_memory_objs in memory_objs:
                self.layer_advances += 1
                yield
        finally:
            self.generator_closed = True

    def drain_pending_gpu_cleanups(self) -> None:
        self.drain_calls += 1
        if self.drain_entered is not None and self.drain_calls == 1:
            self.drain_entered.set()
            assert self.release_drain is not None
            assert self.release_drain.wait(timeout=5.0)

    def pending_gpu_cleanup_count(self) -> int:
        return 0


class _StoreRaceStats:
    """Minimal store statistics collaborator."""

    def __init__(self) -> None:
        self.request_calls = 0
        self.finished_calls = 0

    def on_store_request(self, count: int) -> int:
        self.request_calls += 1
        return count

    def on_store_finished(self, monitor_id: int, count: int) -> None:
        del monitor_id, count
        self.finished_calls += 1


def _store_race_engine(
    *,
    drain_entered: threading.Event | None = None,
    release_drain: threading.Event | None = None,
) -> tuple[LMCacheEngine, _StoreRaceStorage, _StoreRaceGPU, _StoreRaceStats]:
    """Build a CPU-only engine shell around the real store/close methods."""
    engine = LMCacheEngine.__new__(LMCacheEngine)
    state = cast(Any, engine)
    storage = _StoreRaceStorage()
    gpu = _StoreRaceGPU(drain_entered, release_drain)
    stats = _StoreRaceStats()
    key = CacheEngineKey("store-race", 1, 0, 1, torch.bfloat16)

    state.config = SimpleNamespace(
        get_extra_config_value=lambda key_name, default: default
    )
    state.metadata = SimpleNamespace(kv_dtype=torch.bfloat16)
    state.storage_manager = storage
    state.gpu_connector = gpu
    state.stats_monitor = stats
    state.num_layers = 1
    state.fmt = MemoryFormat.KV_2TD
    state.retrieve_locations = None
    state.store_location = "LocalCPUBackend"
    state.kv_events_enabled = False
    state.kvcache_check_log_enabled = False
    state.hidden_state_store = None
    state.lmcache_worker = None
    state._usage_context = None
    state.is_healthy = lambda: True
    state.is_frozen = lambda: False
    state._process_layerwise_tokens = lambda *args: ((0, 1, key),)
    state._close_lookup_cleanups = MagicMock()
    LMCacheEngine._ensure_store_cleanup_state(engine)
    return engine, storage, gpu, stats


def test_engine_close_owns_and_retries_usage_thread_join() -> None:
    """Telemetry must terminate before engine collaborators are torn down."""
    report_entered = threading.Event()
    release_report = threading.Event()

    class BlockingUsageContext(UsageContext):
        def report_once(self) -> None:
            report_entered.set()
            assert release_report.wait(timeout=5.0)

    context = BlockingUsageContext(MagicMock(), MagicMock())
    context.REPORT_THREAD_JOIN_TIMEOUT_SECONDS = 0.01
    context.start_reporting()
    assert report_entered.wait(timeout=5.0)

    engine, storage, _, _ = _store_race_engine()
    cast(Any, engine)._usage_context = context

    with pytest.raises(TimeoutError, match="did not terminate"):
        engine.close()
    assert context.thread_alive
    assert not context.closed
    assert storage.close_calls == 0

    release_report.set()
    engine.close()
    assert context.closed
    assert not context.thread_alive
    assert storage.close_calls == 1


def test_builder_destroy_retains_registries_until_close_retry() -> None:
    """A failed engine close must keep the builder's exact retry owner."""
    instance_id = "test_destroy_close_retry"
    engine = MagicMock(spec=LMCacheEngine)
    config = MagicMock(spec=LMCacheEngineConfig)
    metadata = MagicMock()
    stat_logger = MagicMock()
    close_error = RuntimeError("injected engine close failure")
    engine.close.side_effect = [close_error, None]

    assert instance_id not in LMCacheEngineBuilder._instances
    LMCacheEngineBuilder._instances[instance_id] = engine
    LMCacheEngineBuilder._cfgs[instance_id] = config
    LMCacheEngineBuilder._metadatas[instance_id] = metadata
    LMCacheEngineBuilder._stat_loggers[instance_id] = stat_logger

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
            with pytest.raises(RuntimeError) as failure:
                LMCacheEngineBuilder.destroy(instance_id)

            assert failure.value is close_error
            assert LMCacheEngineBuilder._instances[instance_id] is engine
            assert LMCacheEngineBuilder._cfgs[instance_id] is config
            assert LMCacheEngineBuilder._metadatas[instance_id] is metadata
            assert LMCacheEngineBuilder._stat_loggers[instance_id] is stat_logger
            stat_logger.shutdown.assert_not_called()
            destroy_monitor.assert_not_called()
            drop_blender.assert_not_called()
            unregister_model.assert_not_called()

            LMCacheEngineBuilder.destroy(instance_id)

            assert instance_id not in LMCacheEngineBuilder._instances
            assert instance_id not in LMCacheEngineBuilder._cfgs
            assert instance_id not in LMCacheEngineBuilder._metadatas
            assert instance_id not in LMCacheEngineBuilder._stat_loggers
            assert engine.close.call_count == 2
            stat_logger.shutdown.assert_called_once_with()
            destroy_monitor.assert_called_once_with()
            drop_blender.assert_called_once_with(instance_id)
            unregister_model.assert_called_once_with(instance_id)
    finally:
        LMCacheEngineBuilder._instances.pop(instance_id, None)
        LMCacheEngineBuilder._cfgs.pop(instance_id, None)
        LMCacheEngineBuilder._metadatas.pop(instance_id, None)
        LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)


def test_builder_retains_engine_when_storage_close_rejects_active_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A StorageManager lifecycle refusal propagates through both owners."""
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    instance_id = "test_storage_close_retry"
    engine, storage, _, _ = _store_race_engine()
    config = MagicMock(spec=LMCacheEngineConfig)
    metadata = MagicMock()
    stat_logger = MagicMock()
    close_error = RuntimeError("storage manager retained 1 active batched-put owner(s)")
    storage.close_error = close_error

    assert instance_id not in LMCacheEngineBuilder._instances
    LMCacheEngineBuilder._instances[instance_id] = engine
    LMCacheEngineBuilder._cfgs[instance_id] = config
    LMCacheEngineBuilder._metadatas[instance_id] = metadata
    LMCacheEngineBuilder._stat_loggers[instance_id] = stat_logger

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
            with pytest.raises(RuntimeError) as failure:
                LMCacheEngineBuilder.destroy(instance_id)

            assert failure.value is close_error
            assert storage.close_calls == 1
            assert LMCacheEngineBuilder._instances[instance_id] is engine
            assert LMCacheEngineBuilder._cfgs[instance_id] is config
            assert LMCacheEngineBuilder._metadatas[instance_id] is metadata
            assert LMCacheEngineBuilder._stat_loggers[instance_id] is stat_logger
            stat_logger.shutdown.assert_not_called()
            destroy_monitor.assert_not_called()
            drop_blender.assert_not_called()
            unregister_model.assert_not_called()

            LMCacheEngineBuilder.destroy(instance_id)

            assert storage.close_calls == 2
            assert instance_id not in LMCacheEngineBuilder._instances
            assert instance_id not in LMCacheEngineBuilder._cfgs
            assert instance_id not in LMCacheEngineBuilder._metadatas
            assert instance_id not in LMCacheEngineBuilder._stat_loggers
            stat_logger.shutdown.assert_called_once_with()
            destroy_monitor.assert_called_once_with()
            drop_blender.assert_called_once_with(instance_id)
            unregister_model.assert_called_once_with(instance_id)
    finally:
        LMCacheEngineBuilder._instances.pop(instance_id, None)
        LMCacheEngineBuilder._cfgs.pop(instance_id, None)
        LMCacheEngineBuilder._metadatas.pop(instance_id, None)
        LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)


def test_close_vs_store_registration_win_is_retry_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admitted live store makes close fail before collaborator teardown."""
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, gpu, stats = _store_race_engine()
    registered = threading.Event()
    release_registration = threading.Event()
    store_errors: list[BaseException] = []
    original_register = engine._register_store_cleanup

    def register_then_pause(owner: Any) -> None:
        original_register(owner)
        registered.set()
        assert release_registration.wait(timeout=5.0)

    cast(Any, engine)._register_store_cleanup = register_then_pause
    storer = engine.store_layer([1], req_id="admitted-store")

    def advance_store() -> None:
        try:
            next(storer)
        except BaseException as error:
            store_errors.append(error)

    store_thread = threading.Thread(target=advance_store)
    store_thread.start()
    assert registered.wait(timeout=5.0)

    try:
        with pytest.raises(RuntimeError, match="active layerwise store"):
            engine.close()

        assert gpu.drain_calls == 0
        assert storage.drain_calls == 0
        assert storage.close_calls == 0
        cast(MagicMock, cast(Any, engine)._close_lookup_cleanups).assert_not_called()

        rejected_storer = engine.store_layer([2], req_id="post-close-store")
        with pytest.raises(RuntimeError, match="close has begun"):
            next(rejected_storer)
        assert storage.allocations == []
        assert stats.request_calls == 0
    finally:
        release_registration.set()
        store_thread.join(timeout=5.0)

    assert not store_thread.is_alive()
    assert store_errors == []
    assert len(storage.allocations) == 1
    assert gpu.generator_started

    # The failed close did not tear down collaborators, so the already
    # admitted generator can safely finish its current layer before cleanup.
    assert next(storer) is None
    assert storage.put_calls == 1
    storer.close()
    assert gpu.generator_closed
    assert engine._store_cleanup_owner_count() == 0

    engine.close()
    assert storage.close_calls == 1
    cast(MagicMock, cast(Any, engine)._close_lookup_cleanups).assert_called_once()


def test_close_vs_store_close_win_rejects_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sealed close prevents concurrent stores from publishing an owner."""
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    drain_entered = threading.Event()
    release_drain = threading.Event()
    engine, storage, _, stats = _store_race_engine(
        drain_entered=drain_entered,
        release_drain=release_drain,
    )
    close_errors: list[BaseException] = []

    def close_engine() -> None:
        try:
            engine.close()
        except BaseException as error:
            close_errors.append(error)

    close_thread = threading.Thread(target=close_engine)
    close_thread.start()
    assert drain_entered.wait(timeout=5.0)

    try:
        storer = engine.store_layer([1], req_id="close-lost-store")
        with pytest.raises(RuntimeError, match="close has begun"):
            next(storer)
        assert storage.allocations == []
        assert stats.request_calls == 0
        assert storage.close_calls == 0
    finally:
        release_drain.set()
        close_thread.join(timeout=5.0)

    assert not close_thread.is_alive()
    assert close_errors == []
    assert storage.close_calls == 1


def test_close_and_store_finalizer_share_one_completed_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late finalizer cannot touch storage after close drained its owner."""
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, gpu, _ = _store_race_engine()
    storer = engine.store_layer([1], req_id="pending-store")
    next(storer)

    with cast(Any, engine)._store_cleanup_lock:
        owner = next(iter(cast(Any, engine)._store_cleanup_owners.values()))
    original_drain = owner.drain
    first_drain_entered = threading.Event()
    release_first_drain = threading.Event()
    drain_call_lock = threading.Lock()
    drain_calls = 0

    def pause_first_drain() -> Any:
        nonlocal drain_calls
        with drain_call_lock:
            call_index = drain_calls
            drain_calls += 1
        if call_index == 0:
            first_drain_entered.set()
            assert release_first_drain.wait(timeout=5.0)
        return original_drain()

    owner.drain = pause_first_drain
    store_close_errors: list[BaseException] = []

    def close_store() -> None:
        try:
            storer.close()
        except BaseException as error:
            store_close_errors.append(error)

    store_close_thread = threading.Thread(target=close_store)
    store_close_thread.start()
    assert first_drain_entered.wait(timeout=5.0)

    try:
        # The engine drains the now-pending owner and can close. The delayed
        # generator finalizer must observe the completed latch and no-op.
        engine.close()
        assert storage.close_calls == 1
        assert store_close_thread.is_alive()
    finally:
        release_first_drain.set()
        store_close_thread.join(timeout=5.0)

    assert not store_close_thread.is_alive()
    assert store_close_errors == []
    assert storage.drain_after_close_calls == 0
    assert storage.drain_calls == 2
    assert gpu.drain_calls == 2
    assert gpu.generator_closed
    assert storage.allocations[0].release_count == 1


def test_store_prelude_failures_never_orphan_active_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed input and pre-body BaseException leave no live store owner."""
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, _, stats = _store_race_engine()

    malformed = engine.store_layer(
        [1],
        req_id="malformed-config",
        request_configs=["not-a-dict"],
    )
    with pytest.raises(AssertionError):
        next(malformed)

    assert engine._store_cleanup_owner_count() == 0
    assert storage.allocations == []
    assert stats.request_calls == 0

    class _StorePreludeInterrupt(BaseException):
        pass

    prelude_error = _StorePreludeInterrupt("injected stats prelude interrupt")
    cast(Any, stats).on_store_request = MagicMock(side_effect=prelude_error)
    interrupted = engine.store_layer([1], req_id="prelude-interrupt")
    with pytest.raises(_StorePreludeInterrupt) as failure:
        next(interrupted)

    assert failure.value is prelude_error
    assert engine._store_cleanup_owner_count() == 0
    assert storage.allocations == []
    assert storage.drain_calls == 1


def test_store_registration_interrupt_never_orphans_active_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupt after ledger publication still enters final cleanup."""
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, _, stats = _store_race_engine()
    original_register = engine._register_store_cleanup

    class _RegistrationInterrupt(BaseException):
        pass

    register_error = _RegistrationInterrupt("injected store-registration interrupt")

    def register_then_interrupt(owner: Any) -> None:
        original_register(owner)
        raise register_error

    monkeypatch.setattr(
        engine,
        "_register_store_cleanup",
        register_then_interrupt,
    )
    interrupted = engine.store_layer([1], req_id="registration-interrupt")
    with pytest.raises(_RegistrationInterrupt) as failure:
        next(interrupted)

    assert failure.value is register_error
    assert engine._store_cleanup_owner_count() == 0
    assert engine.pending_store_cleanup_count() == 0
    assert storage.allocations == []
    assert storage.drain_calls == 1
    assert stats.request_calls == 0


@pytest.fixture
def pin_monitor() -> Generator[None, None, None]:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256, lmcache_instance_id="test_cleanup"
    )
    PinMonitor.GetOrCreate(config)
    yield
    PinMonitor.DestroyInstance()


def test_cleanup_memory_objs_handles_mixed_pin_state(
    caplog: pytest.LogCaptureFixture, pin_monitor: Any
) -> None:
    """
    Cleanup must unpin chunks that were pinned during prefetch and
    skip unpin for chunks that were not, leaving every chunk with a
    non-negative pin_count and no "Double unpin" warning.
    """
    pinned_obj = create_test_memory_obj()
    pinned_obj.pin()
    pinned_obj.ref_count_up()
    assert pinned_obj.metadata.pin_count == 1

    nonpinned_obj = create_test_memory_obj()
    assert nonpinned_obj.metadata.pin_count == 0

    future = MagicMock()
    future.result.return_value = [
        [(None, pinned_obj)],
        [(None, nonpinned_obj)],
    ]

    engine = SimpleNamespace(event_manager=MagicMock())
    engine.event_manager.get_event_status.return_value = EventStatus.DONE
    engine.event_manager.pop_event.return_value = future

    caplog.set_level(logging.WARNING, logger="lmcache")

    LMCacheEngine.cleanup_memory_objs(engine, "test_lookup")  # type: ignore[arg-type]

    assert pinned_obj.metadata.pin_count == 0
    assert nonpinned_obj.metadata.pin_count == 0
    assert "Double unpin" not in caplog.text
    assert "is negative" not in caplog.text

    pinned_obj.ref_count_down()
