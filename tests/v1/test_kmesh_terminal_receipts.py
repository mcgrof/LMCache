# SPDX-License-Identifier: Apache-2.0
"""Deterministic terminal-lifecycle regressions for the KMesh profile."""

# Standard
from typing import Any, cast
from unittest.mock import MagicMock, patch
import threading

# Third Party
import pytest

# First Party
from lmcache.observability import LMCacheStatsLogger
from lmcache.v1.cache_controller.worker import LMCacheWorker
from lmcache.v1.cache_engine import (
    EngineDestroyResult,
    LMCacheEngine,
    LMCacheEngineBuilder,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.hidden_state_store import HiddenStateStore
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
import lmcache.v1.cache_controller.worker as worker_module
import lmcache.v1.cache_engine as cache_engine_module
import lmcache.v1.rpc_utils as rpc_utils_module


class _PublishThenRaise:
    """Non-idempotent terminal child used to detect a duplicate close."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise self.error


class _StorageChild:
    """Minimal successful storage child for a lightweight engine shell."""

    def __init__(self) -> None:
        self.close_calls = 0

    def drain_pending_put_cleanups(self) -> None:
        return None

    def pending_put_cleanup_count(self) -> int:
        return 0

    def close(self) -> None:
        self.close_calls += 1


class _ConstructorSocket:
    """Exact fake socket for constructor-unwind and worker-close probes."""

    def __init__(
        self,
        *,
        bind_error: BaseException | None = None,
        option_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.bind_error = bind_error
        self.option_error = option_error
        self.close_error = close_error
        self.close_calls = 0
        self.closed = False

    def bind(self, address: str) -> None:
        del address
        if self.bind_error is not None:
            raise self.bind_error

    def connect(self, address: str) -> None:
        self.bind(address)

    def setsockopt(self, option: Any, value: Any) -> None:
        del option, value
        if self.option_error is not None:
            raise self.option_error

    def close(self, linger: int | None = None) -> None:
        assert linger in (None, 0)
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


def test_rpc_socket_bind_failure_closes_unpublished_socket_once() -> None:
    """A socket that cannot bind never escapes its constructor helper."""
    bind_error = KeyboardInterrupt("injected bind interruption")
    socket = _ConstructorSocket(bind_error=bind_error)
    context = MagicMock()
    context.socket.return_value = socket

    with pytest.raises(KeyboardInterrupt) as failure:
        rpc_utils_module.get_zmq_socket(
            context,
            "worker-endpoint",
            "tcp",
            object(),
            "bind",
        )

    assert failure.value is bind_error
    assert failure.value.__cause__ is None
    assert socket.close_calls == 1
    assert socket.closed


def test_rpc_socket_option_failure_chains_cleanup_failure() -> None:
    """Timeout setup preserves both its failure and a failed socket close."""
    option_error = KeyboardInterrupt("injected timeout-option interruption")
    cleanup_error = RuntimeError("injected socket cleanup failure")
    socket = _ConstructorSocket(
        option_error=option_error,
        close_error=cleanup_error,
    )
    context = MagicMock()
    context.socket.return_value = socket

    with pytest.raises(KeyboardInterrupt) as failure:
        rpc_utils_module.get_zmq_socket_with_timeout(
            context,
            "worker-endpoint",
            "tcp",
            object(),
            "connect",
            100,
            100,
        )

    assert failure.value is option_error
    assert failure.value.__cause__ is cleanup_error
    assert socket.close_calls == 1
    assert not socket.closed


def _engine_shell(
    *,
    hidden_state_store: Any = None,
    lmcache_worker: Any = None,
    storage_manager: Any = None,
) -> LMCacheEngine:
    engine = LMCacheEngine.__new__(LMCacheEngine)
    state = cast(Any, engine)
    state.gpu_connector = None
    state.hidden_state_store = hidden_state_store
    state.lmcache_worker = lmcache_worker
    state.storage_manager = storage_manager
    state._close_lookup_cleanups = MagicMock()
    LMCacheEngine._ensure_store_cleanup_state(engine)
    return engine


def _install_builder_generation(
    instance_id: str,
    engine: LMCacheEngine,
    *,
    config: Any = None,
    metadata: Any = None,
    stat_logger: Any = None,
) -> tuple[Any, Any, Any]:
    config = config if config is not None else MagicMock(spec=LMCacheEngineConfig)
    metadata = metadata if metadata is not None else MagicMock()
    stat_logger = stat_logger if stat_logger is not None else MagicMock()
    LMCacheEngineBuilder._instances[instance_id] = engine
    LMCacheEngineBuilder._cfgs[instance_id] = config
    LMCacheEngineBuilder._metadatas[instance_id] = metadata
    LMCacheEngineBuilder._stat_loggers[instance_id] = stat_logger
    return config, metadata, stat_logger


def _forget_builder_generation(instance_id: str, *engines: LMCacheEngine) -> None:
    LMCacheEngineBuilder._instances.pop(instance_id, None)
    LMCacheEngineBuilder._cfgs.pop(instance_id, None)
    LMCacheEngineBuilder._metadatas.pop(instance_id, None)
    LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)
    LMCacheEngineBuilder._ready_generations.pop(instance_id, None)
    LMCacheEngineBuilder._creation_owners.pop(instance_id, None)
    LMCacheEngineBuilder._destroy_intents.discard(instance_id)
    LMCacheEngineBuilder._destroy_owners.pop(instance_id, None)
    for engine in engines:
        LMCacheEngineBuilder._destroyed_engines.pop(engine, None)


def test_get_or_create_lookup_is_serialized_with_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Destroy cannot install an owner between lookup and engine return."""
    instance_id = "kmesh-terminal-create-destroy-race"
    contains_entered = threading.Event()
    release_contains = threading.Event()
    destroy_started = threading.Event()
    close_entered = threading.Event()
    release_close = threading.Event()

    engine = MagicMock(spec=LMCacheEngine)

    def blocking_close() -> None:
        close_entered.set()
        assert release_close.wait(timeout=5.0)

    engine.close.side_effect = blocking_close
    config = MagicMock(spec=LMCacheEngineConfig)
    metadata = MagicMock()
    stat_logger = MagicMock()

    class _BlockingReady(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            if key == instance_id and threading.current_thread().name == "kmesh-getter":
                contains_entered.set()
                assert release_contains.wait(timeout=5.0)
            return super().get(key, default)

    blocking_ready = _BlockingReady(LMCacheEngineBuilder._ready_generations)
    monkeypatch.setattr(LMCacheEngineBuilder, "_ready_generations", blocking_ready)
    _install_builder_generation(
        instance_id,
        cast(LMCacheEngine, engine),
        config=config,
        metadata=metadata,
        stat_logger=stat_logger,
    )
    results: list[LMCacheEngine] = []
    errors: list[BaseException] = []

    def get_existing() -> None:
        try:
            results.append(
                LMCacheEngineBuilder.get_or_create(
                    instance_id,
                    config,
                    metadata,
                    None,
                    MagicMock(),
                    MagicMock(),
                )
            )
        except BaseException as error:
            errors.append(error)

    def destroy_existing() -> None:
        destroy_started.set()
        try:
            LMCacheEngineBuilder.destroy(instance_id)
        except BaseException as error:
            errors.append(error)

    getter = threading.Thread(target=get_existing, name="kmesh-getter")
    destroyer = threading.Thread(target=destroy_existing, name="kmesh-destroyer")
    try:
        with (
            patch("lmcache.v1.cache_engine.LMCStatsMonitor.DestroyInstance"),
            patch("lmcache.v1.compute.blend.utils.LMCBlenderBuilder.drop"),
            patch("lmcache.v1.compute.models.utils.VLLMModelTracker.unregister_model"),
        ):
            getter.start()
            assert contains_entered.wait(timeout=5.0)
            destroyer.start()
            assert destroy_started.wait(timeout=5.0)

            # The getter is paused after its owner check. Destroy must still be
            # excluded by the same lock and therefore cannot enter close.
            assert not close_entered.wait(timeout=0.1)
            release_contains.set()
            getter.join(timeout=5.0)
            assert not getter.is_alive()
            assert results == [engine]
            assert close_entered.wait(timeout=5.0)
            assert LMCacheEngineBuilder.get(instance_id) is None
            release_close.set()
            destroyer.join(timeout=5.0)
            assert not destroyer.is_alive()
            assert errors == []
    finally:
        release_contains.set()
        release_close.set()
        getter.join(timeout=5.0)
        if destroyer.ident is not None:
            destroyer.join(timeout=5.0)
        _forget_builder_generation(instance_id, cast(LMCacheEngine, engine))


def test_absent_destroy_atomically_fences_new_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No creator can enter between an absent lookup and destroy ownership."""
    instance_id = "kmesh-terminal-absent-destroy-admission"
    allow_creator = threading.Event()
    creator_done = threading.Event()
    creator_errors: list[BaseException] = []
    destroy_results: list[EngineDestroyResult] = []
    destroy_errors: list[BaseException] = []

    class _YieldAfterDestroyAdmission:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._yielded = False

        def __enter__(self) -> "_YieldAfterDestroyAdmission":
            self._lock.acquire()
            return self

        def __exit__(self, *args: Any) -> None:
            self._lock.release()
            if (
                threading.current_thread().name == "kmesh-absent-destroyer"
                and not self._yielded
            ):
                self._yielded = True
                allow_creator.set()
                assert creator_done.wait(timeout=5.0)

    class _StopAfterCreationPublication(BaseException):
        pass

    def stop_after_creation_publication(
        cls: type[LMCacheEngineBuilder],
        owner_instance_id: str,
        owner: Any,
        **kwargs: Any,
    ) -> Any:
        del cls, owner, kwargs
        assert owner_instance_id == instance_id
        raise _StopAfterCreationPublication("creation owner was admitted")

    monkeypatch.setattr(
        LMCacheEngineBuilder,
        "_destroy_lock",
        _YieldAfterDestroyAdmission(),
    )
    monkeypatch.setattr(
        LMCacheEngineBuilder,
        "_finish_creation",
        classmethod(stop_after_creation_publication),
    )

    def create_after_destroy_lookup() -> None:
        assert allow_creator.wait(timeout=5.0)
        try:
            LMCacheEngineBuilder.get_or_create(
                instance_id,
                LMCacheEngineConfig.from_defaults(),
                MagicMock(),
                None,
                MagicMock(),
                MagicMock(),
            )
        except BaseException as error:
            creator_errors.append(error)
        finally:
            creator_done.set()

    def destroy_absent() -> None:
        try:
            destroy_results.append(LMCacheEngineBuilder.destroy(instance_id))
        except BaseException as error:
            destroy_errors.append(error)

    creator = threading.Thread(
        target=create_after_destroy_lookup,
        name="kmesh-absent-creator",
    )
    destroyer = threading.Thread(
        target=destroy_absent,
        name="kmesh-absent-destroyer",
    )
    try:
        with (
            patch("lmcache.v1.compute.blend.utils.LMCBlenderBuilder.drop"),
            patch("lmcache.v1.compute.models.utils.VLLMModelTracker.unregister_model"),
        ):
            creator.start()
            destroyer.start()
            creator.join(timeout=5.0)
            destroyer.join(timeout=5.0)

        assert not creator.is_alive()
        assert not destroyer.is_alive()
        assert destroy_errors == []
        assert destroy_results == [EngineDestroyResult.ALREADY_ABSENT]
        assert len(creator_errors) == 1
        assert isinstance(creator_errors[0], RuntimeError)
        assert "terminal cleanup ownership" in str(creator_errors[0])
        assert instance_id not in LMCacheEngineBuilder._creation_owners
        assert instance_id not in LMCacheEngineBuilder._destroy_owners
    finally:
        allow_creator.set()
        creator_done.set()
        creator.join(timeout=5.0)
        destroyer.join(timeout=5.0)
        _forget_builder_generation(instance_id)


def test_get_hides_generation_after_returned_engine_close_failure() -> None:
    """A failed terminal owner retains the engine without exposing it."""
    instance_id = "kmesh-terminal-returned-close-failure"
    close_error = RuntimeError("injected returned engine close failure")
    engine = MagicMock(spec=LMCacheEngine)
    engine.close.side_effect = close_error
    engine.close_failure_is_retry_safe.return_value = True
    stat_logger = MagicMock()
    stat_logger.shutdown_failure_is_retry_safe.return_value = True
    _install_builder_generation(
        instance_id,
        cast(LMCacheEngine, engine),
        stat_logger=stat_logger,
    )

    try:
        with (
            patch("lmcache.v1.cache_engine.LMCStatsMonitor.DestroyInstance"),
            patch("lmcache.v1.compute.blend.utils.LMCBlenderBuilder.drop"),
            patch("lmcache.v1.compute.models.utils.VLLMModelTracker.unregister_model"),
        ):
            with pytest.raises(RuntimeError) as failure:
                LMCacheEngineBuilder.destroy(instance_id)

            assert failure.value is close_error
            assert LMCacheEngineBuilder.get(instance_id) is None
            assert LMCacheEngineBuilder._instances[instance_id] is engine
            assert LMCacheEngineBuilder._destroy_owners[instance_id].engine is engine

            engine.close.side_effect = None
            assert (
                LMCacheEngineBuilder.destroy(instance_id)
                is EngineDestroyResult.DESTROYED
            )
            assert LMCacheEngineBuilder.get(instance_id) is None
    finally:
        engine.close.side_effect = None
        _forget_builder_generation(instance_id, cast(LMCacheEngine, engine))


def test_old_expected_tombstone_wins_over_replacement_generation() -> None:
    """An old manager retry must not inspect or close a replacement engine."""
    instance_id = "kmesh-terminal-old-tombstone"
    old_engine = LMCacheEngine.__new__(LMCacheEngine)
    replacement = MagicMock(spec=LMCacheEngine)
    config, metadata, stat_logger = _install_builder_generation(
        instance_id,
        cast(LMCacheEngine, replacement),
    )
    LMCacheEngineBuilder._destroyed_engines[old_engine] = True

    try:
        result = LMCacheEngineBuilder.destroy(
            instance_id,
            expected_engine=old_engine,
        )

        assert result is EngineDestroyResult.DESTROYED
        assert LMCacheEngineBuilder._instances[instance_id] is replacement
        assert LMCacheEngineBuilder._cfgs[instance_id] is config
        assert LMCacheEngineBuilder._metadatas[instance_id] is metadata
        assert LMCacheEngineBuilder._stat_loggers[instance_id] is stat_logger
        replacement.close.assert_not_called()
    finally:
        _forget_builder_generation(
            instance_id,
            old_engine,
            cast(LMCacheEngine, replacement),
        )


def test_old_expected_tombstone_does_not_cancel_replacement_creation() -> None:
    """An old manager retry cannot publish intent against a new creator."""
    instance_id = "kmesh-terminal-old-tombstone-creating"
    old_engine = LMCacheEngine.__new__(LMCacheEngine)
    replacement_owner = MagicMock()
    replacement_owner.engine = None
    LMCacheEngineBuilder._creation_owners[instance_id] = replacement_owner
    LMCacheEngineBuilder._destroyed_engines[old_engine] = True

    try:
        with patch.object(LMCacheEngineBuilder, "_finish_creation") as finish:
            result = LMCacheEngineBuilder.destroy(
                instance_id,
                expected_engine=old_engine,
            )

        assert result is EngineDestroyResult.DESTROYED
        assert LMCacheEngineBuilder._creation_owners[instance_id] is replacement_owner
        assert instance_id not in LMCacheEngineBuilder._destroy_intents
        replacement_owner.cancel_publication.assert_not_called()
        finish.assert_not_called()
    finally:
        _forget_builder_generation(instance_id, old_engine)


def test_destroy_owner_publication_recovers_interrupted_intent_discard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A published destroy owner clears a stale intent on public retry."""
    instance_id = "kmesh-terminal-destroy-owner-intent-discard"
    engine = MagicMock(spec=LMCacheEngine)
    engine.close_failure_is_retry_safe.return_value = True
    stat_logger = MagicMock(spec=LMCacheStatsLogger)
    stat_logger.shutdown_failure_is_retry_safe.return_value = True
    _install_builder_generation(
        instance_id,
        cast(LMCacheEngine, engine),
        stat_logger=stat_logger,
    )

    class _InterruptBeforeFirstDiscard(set[str]):
        def __init__(self, values: set[str]) -> None:
            super().__init__(values)
            self.interrupted = False

        def discard(self, value: str) -> None:
            if value == instance_id and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt("before destroy intent discard")
            super().discard(value)

    intents = _InterruptBeforeFirstDiscard({instance_id})
    monkeypatch.setattr(LMCacheEngineBuilder, "_destroy_intents", intents)

    try:
        with (
            patch("lmcache.v1.cache_engine.LMCStatsMonitor.DestroyInstance"),
            patch("lmcache.v1.compute.blend.utils.LMCBlenderBuilder.drop"),
            patch("lmcache.v1.compute.models.utils.VLLMModelTracker.unregister_model"),
        ):
            with pytest.raises(KeyboardInterrupt, match="intent discard"):
                LMCacheEngineBuilder.destroy(
                    instance_id,
                    expected_engine=cast(LMCacheEngine, engine),
                )

            assert instance_id in LMCacheEngineBuilder._destroy_owners
            assert instance_id in LMCacheEngineBuilder._destroy_intents
            assert LMCacheEngineBuilder.get(instance_id) is None

            result = LMCacheEngineBuilder.destroy(
                instance_id,
                expected_engine=cast(LMCacheEngine, engine),
            )

        assert result is EngineDestroyResult.DESTROYED
        assert instance_id not in LMCacheEngineBuilder._destroy_owners
        assert instance_id not in LMCacheEngineBuilder._destroy_intents
        engine.close.assert_called_once_with()
        stat_logger.shutdown.assert_called_once_with()
    finally:
        _forget_builder_generation(instance_id, cast(LMCacheEngine, engine))


def test_destroy_owner_retained_until_final_intent_discard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final intent cleanup cannot strand an ownerless terminal fence."""
    instance_id = "kmesh-terminal-final-intent-discard"
    engine = MagicMock(spec=LMCacheEngine)
    engine.close_failure_is_retry_safe.return_value = True
    stat_logger = MagicMock(spec=LMCacheStatsLogger)
    stat_logger.shutdown_failure_is_retry_safe.return_value = True
    _install_builder_generation(
        instance_id,
        cast(LMCacheEngine, engine),
        stat_logger=stat_logger,
    )

    class _InterruptBeforeSecondDiscard(set[str]):
        def __init__(self) -> None:
            super().__init__({instance_id})
            self.discard_calls = 0

        def discard(self, value: str) -> None:
            self.discard_calls += 1
            if value == instance_id and self.discard_calls == 2:
                # Model a second clear that was interrupted with the terminal
                # fence still present. The retry owner must remain reachable.
                super().add(value)
                raise KeyboardInterrupt("before final intent discard")
            super().discard(value)

    intents = _InterruptBeforeSecondDiscard()
    monkeypatch.setattr(LMCacheEngineBuilder, "_destroy_intents", intents)

    try:
        with (
            patch("lmcache.v1.cache_engine.LMCStatsMonitor.DestroyInstance"),
            patch("lmcache.v1.compute.blend.utils.LMCBlenderBuilder.drop"),
            patch("lmcache.v1.compute.models.utils.VLLMModelTracker.unregister_model"),
        ):
            with pytest.raises(KeyboardInterrupt, match="final intent discard"):
                LMCacheEngineBuilder.destroy(
                    instance_id,
                    expected_engine=cast(LMCacheEngine, engine),
                )

            assert instance_id in LMCacheEngineBuilder._destroy_owners
            assert instance_id in LMCacheEngineBuilder._destroy_intents
            assert LMCacheEngineBuilder.get(instance_id) is None

            result = LMCacheEngineBuilder.destroy(
                instance_id,
                expected_engine=cast(LMCacheEngine, engine),
            )

        assert result is EngineDestroyResult.DESTROYED
        assert instance_id not in LMCacheEngineBuilder._destroy_owners
        assert instance_id not in LMCacheEngineBuilder._destroy_intents
        engine.close.assert_called_once_with()
        stat_logger.shutdown.assert_called_once_with()
    finally:
        _forget_builder_generation(instance_id, cast(LMCacheEngine, engine))


def test_child_publish_then_raise_is_sticky_and_blocks_registry_commit() -> None:
    """A non-idempotent child failure is propagated but never repeated."""
    instance_id = "kmesh-terminal-publish-then-raise"
    close_error = RuntimeError("hidden-state close published then failed")
    hidden_state_store = _PublishThenRaise(close_error)
    engine = _engine_shell(hidden_state_store=hidden_state_store)
    config, metadata, stat_logger = _install_builder_generation(instance_id, engine)

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
            for _ in range(2):
                with pytest.raises(RuntimeError) as failure:
                    LMCacheEngineBuilder.destroy(instance_id)
                assert failure.value is close_error

            assert hidden_state_store.close_calls == 1
            assert LMCacheEngineBuilder._instances[instance_id] is engine
            assert LMCacheEngineBuilder._cfgs[instance_id] is config
            assert LMCacheEngineBuilder._metadatas[instance_id] is metadata
            assert LMCacheEngineBuilder._stat_loggers[instance_id] is stat_logger
            stat_logger.shutdown.assert_not_called()
            destroy_monitor.assert_not_called()
            drop_blender.assert_not_called()
            unregister_model.assert_not_called()
    finally:
        _forget_builder_generation(instance_id, engine)


def test_worker_close_failure_is_not_swallowed_or_repeated() -> None:
    """An ordinary worker exception blocks every later terminal child."""
    close_error = RuntimeError("worker close failed after deregistration")
    worker = _PublishThenRaise(close_error)
    storage_manager = _StorageChild()
    engine = _engine_shell(
        lmcache_worker=worker,
        storage_manager=storage_manager,
    )

    for _ in range(2):
        with pytest.raises(RuntimeError) as failure:
            engine.close()
        assert failure.value is close_error

    assert worker.close_calls == 1
    assert storage_manager.close_calls == 0


def test_lookup_cleanup_failure_stays_owned_for_close_retry() -> None:
    """Terminal lookup cleanup retains only IDs that did not confirm release."""
    engine = LMCacheEngine.__new__(LMCacheEngine)
    state = cast(Any, engine)
    state.lookup_cleanup_registered = {"lookup-1"}
    state.lookup_cleanup_pending = set()
    state._lookup_cleanup_closed = False
    state._lookup_cleanup_retry_timer = None
    cleanup_error = RuntimeError("lookup unpin did not confirm")
    cleanup_calls = 0

    def cleanup_once(lookup_id: str) -> None:
        nonlocal cleanup_calls
        assert lookup_id == "lookup-1"
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise cleanup_error

    state._lookup_unpin_once = cleanup_once

    with pytest.raises(RuntimeError) as failure:
        engine._close_lookup_cleanups()
    assert failure.value is cleanup_error
    assert state.lookup_cleanup_registered == {"lookup-1"}
    assert state.lookup_cleanup_pending == {"lookup-1"}

    engine._close_lookup_cleanups()
    assert cleanup_calls == 2
    assert state.lookup_cleanup_registered == set()
    assert state.lookup_cleanup_pending == set()


def test_stats_logger_timeout_retains_builder_until_same_thread_stops() -> None:
    """A live logger thread prevents monitor and registry destruction."""
    instance_id = "kmesh-terminal-stats-timeout"

    class _ControlledThread:
        def __init__(self) -> None:
            self.alive = True
            self.join_calls = 0

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            assert timeout == 5.0
            self.join_calls += 1

    controlled_thread = _ControlledThread()
    stat_logger = LMCacheStatsLogger.__new__(LMCacheStatsLogger)
    stat_state = cast(Any, stat_logger)
    stat_state.is_running = True
    stat_state.shutdown_event = threading.Event()
    stat_state.thread = controlled_thread

    engine = MagicMock(spec=LMCacheEngine)
    config, metadata, _ = _install_builder_generation(
        instance_id,
        cast(LMCacheEngine, engine),
        stat_logger=stat_logger,
    )

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
            with pytest.raises(RuntimeError, match="did not terminate"):
                LMCacheEngineBuilder.destroy(instance_id)

            assert controlled_thread.join_calls == 1
            assert stat_state.shutdown_event.is_set()
            assert LMCacheEngineBuilder._instances[instance_id] is engine
            assert LMCacheEngineBuilder._cfgs[instance_id] is config
            assert LMCacheEngineBuilder._metadatas[instance_id] is metadata
            destroy_monitor.assert_not_called()
            drop_blender.assert_not_called()
            unregister_model.assert_not_called()

            controlled_thread.alive = False
            assert (
                LMCacheEngineBuilder.destroy(instance_id)
                is EngineDestroyResult.DESTROYED
            )

            assert engine.close.call_count == 1
            assert instance_id not in LMCacheEngineBuilder._instances
            destroy_monitor.assert_called_once_with()
            drop_blender.assert_called_once_with(instance_id)
            unregister_model.assert_called_once_with(instance_id)
    finally:
        controlled_thread.alive = False
        _forget_builder_generation(
            instance_id,
            cast(LMCacheEngine, engine),
        )


class _ConstructionThread:
    """Controllable exact stats worker for constructor interruption tests."""

    def __init__(self) -> None:
        self.alive = True
        self.join_calls = 0

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float) -> None:
        assert timeout == 5.0
        self.join_calls += 1
        self.alive = False


def _patch_lightweight_construction(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stat_init: Any,
    engine_init: Any,
) -> None:
    monkeypatch.setattr(
        cache_engine_module.NUMADetector,
        "get_numa_mapping",
        staticmethod(lambda config: None),
    )
    monkeypatch.setattr(
        LMCacheEngineBuilder,
        "_Create_token_database",
        staticmethod(lambda config, metadata: object()),
    )
    monkeypatch.setattr(LMCacheStatsLogger, "__init__", stat_init)
    monkeypatch.setattr(LMCacheEngine, "__init__", engine_init)


def _fake_stats_init(
    stat_logger: LMCacheStatsLogger,
    metadata: Any,
    log_interval: int,
    config: Any,
) -> None:
    del metadata, log_interval, config
    state = cast(Any, stat_logger)
    state.is_running = True
    state.shutdown_event = threading.Event()
    state.thread = _ConstructionThread()


def _fake_engine_init(
    engine: LMCacheEngine,
    config: Any,
    metadata: Any,
    token_database: Any,
    gpu_connector: Any,
    broadcast_fn: Any,
    broadcast_object_fn: Any,
) -> None:
    del (
        config,
        metadata,
        token_database,
        gpu_connector,
        broadcast_fn,
        broadcast_object_fn,
    )
    # _BuilderCreateOwner already installed every partial-close prerequisite.
    assert engine.storage_manager is None


def test_builder_retained_worker_restores_caller_device_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NUMA and engine init run only after restoring the caller's GPU."""
    instance_id = "kmesh-create-device-affinity"
    caller_ident = threading.get_ident()
    events: list[tuple[str, Any, str, int]] = []

    def current_device() -> int:
        events.append(
            (
                "capture",
                3,
                threading.current_thread().name,
                threading.get_ident(),
            )
        )
        return 3

    def set_device(device: Any) -> None:
        events.append(
            (
                "set",
                device,
                threading.current_thread().name,
                threading.get_ident(),
            )
        )

    def detect_numa(config: Any) -> None:
        del config
        events.append(
            (
                "numa",
                None,
                threading.current_thread().name,
                threading.get_ident(),
            )
        )

    def engine_init(*args: Any, **kwargs: Any) -> None:
        events.append(
            (
                "engine",
                None,
                threading.current_thread().name,
                threading.get_ident(),
            )
        )
        _fake_engine_init(*args, **kwargs)

    monkeypatch.setattr(cache_engine_module.torch_dev, "is_available", lambda: True)
    monkeypatch.setattr(cache_engine_module.torch_dev, "current_device", current_device)
    monkeypatch.setattr(cache_engine_module.torch_dev, "set_device", set_device)
    _patch_lightweight_construction(
        monkeypatch,
        stat_init=_fake_stats_init,
        engine_init=engine_init,
    )
    monkeypatch.setattr(
        cache_engine_module.NUMADetector,
        "get_numa_mapping",
        staticmethod(detect_numa),
    )
    metadata = MagicMock()
    metadata.role = "worker"
    engine: LMCacheEngine | None = None

    try:
        engine = LMCacheEngineBuilder.get_or_create(
            instance_id,
            MagicMock(spec=LMCacheEngineConfig),
            metadata,
            None,
            MagicMock(),
            MagicMock(),
        )

        captures = [event for event in events if event[0] == "capture"]
        assert captures
        assert all(event[3] == caller_ident for event in captures)
        construct_thread = f"lmcache-create-{instance_id}"
        construct_events = [
            (name, value)
            for name, value, thread_name, _ in events
            if thread_name == construct_thread
        ]
        assert construct_events == [
            ("set", 3),
            ("numa", None),
            ("engine", None),
        ]
    finally:
        with (
            patch("lmcache.v1.cache_engine.LMCStatsMonitor.DestroyInstance"),
            patch("lmcache.v1.compute.blend.utils.LMCBlenderBuilder.drop"),
            patch("lmcache.v1.compute.models.utils.VLLMModelTracker.unregister_model"),
        ):
            if engine is not None:
                LMCacheEngineBuilder.destroy(instance_id)
        _forget_builder_generation(
            instance_id, *([engine] if engine is not None else [])
        )


def test_hidden_state_constructor_interrupt_retains_and_frees_exact_arena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hidden-state arena is owned before its allocator init can abort."""
    allocators: list[MixedMemoryAllocator] = []
    physical_frees = 0

    def allocate_then_interrupt(
        allocator: MixedMemoryAllocator,
        size: int,
        use_paging: bool = False,
        use_hugepages: bool = False,
        **kwargs: Any,
    ) -> None:
        del size, use_paging, use_hugepages, kwargs
        MixedMemoryAllocator._prepare_partial_state(allocator)
        allocator._unregistered = False
        allocator.buffer = object()
        allocators.append(allocator)
        raise KeyboardInterrupt("injected after hidden-state arena publication")

    def exact_free(allocator: MixedMemoryAllocator) -> None:
        nonlocal physical_frees
        if allocator._unregistered:
            return
        physical_frees += 1
        allocator._unregistered = True

    monkeypatch.setattr(MixedMemoryAllocator, "__init__", allocate_then_interrupt)
    monkeypatch.setattr(MixedMemoryAllocator, "close", exact_free)
    config = MagicMock(spec=LMCacheEngineConfig)
    config.max_hidden_state_cpu_size = 1.0
    store = HiddenStateStore.__new__(HiddenStateStore)

    with pytest.raises(KeyboardInterrupt, match="arena publication"):
        HiddenStateStore.__init__(store, config, MagicMock())

    assert len(allocators) == 1
    assert store._allocator is allocators[0]
    store.close()
    store.close()
    assert physical_frees == 1
    assert allocators[0]._unregistered


def test_worker_constructor_interrupt_closes_startup_race_children_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-start interrupt drains the exact loop, thread, and sockets."""

    class _StartupLoop:
        def __init__(self) -> None:
            self.closed = False
            self.stop_calls = 0

        def run_forever(self) -> None:
            return None

        def is_running(self) -> bool:
            # Exercise the interval after Thread.start() but before the loop
            # publishes its running state.
            return False

        def is_closed(self) -> bool:
            return self.closed

        def call_soon_threadsafe(self, callback: Any) -> None:
            self.stop_calls += 1
            callback()

        def stop(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class _StartupThread:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.alive = False
            self.join_calls = 0

        def start(self) -> None:
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            assert timeout == 5.0
            self.join_calls += 1
            self.alive = False

    loop = _StartupLoop()
    thread = _StartupThread()
    push_socket = _ConstructorSocket()
    reply_socket = _ConstructorSocket()
    req_socket = _ConstructorSocket()
    ordinary_sockets = iter((push_socket, reply_socket))

    monkeypatch.setattr(worker_module.asyncio, "new_event_loop", lambda: loop)
    monkeypatch.setattr(
        worker_module.threading,
        "Thread",
        lambda **kwargs: thread,
    )
    monkeypatch.setattr(
        worker_module,
        "get_zmq_socket",
        lambda *args, **kwargs: next(ordinary_sockets),
    )
    monkeypatch.setattr(
        worker_module,
        "get_zmq_socket_with_timeout",
        lambda *args, **kwargs: req_socket,
    )
    monkeypatch.setattr(worker_module, "get_ip", lambda: "127.0.0.1")

    def interrupt_after_start(coroutine: Any, event_loop: Any) -> None:
        assert event_loop is loop
        coroutine.close()
        raise KeyboardInterrupt("injected after worker thread start")

    monkeypatch.setattr(
        worker_module.asyncio,
        "run_coroutine_threadsafe",
        interrupt_after_start,
    )
    config = MagicMock(spec=LMCacheEngineConfig)
    config.lmcache_instance_id = "partial-worker"
    config.controller_pull_url = "controller-pull"
    config.controller_reply_url = "controller-reply"
    config.enable_p2p = False
    config.lmcache_worker_ports = [1234]
    config.get_extra_config_value.side_effect = lambda name, default: default
    config.get_lmcache_worker_ids.return_value = []
    metadata = MagicMock()
    metadata.worker_id = 0
    metadata.world_size = 1
    metadata.use_mla = False
    worker = LMCacheWorker.__new__(LMCacheWorker)

    with pytest.raises(KeyboardInterrupt, match="worker thread start"):
        LMCacheWorker.__init__(worker, config, metadata, MagicMock())

    assert not worker._construction_complete
    assert thread.is_alive()
    deregister = MagicMock()
    cast(Any, worker).deregister = deregister
    worker.close()
    worker.close()

    deregister.assert_not_called()
    assert loop.stop_calls == 1
    assert loop.closed
    assert not thread.is_alive()
    assert thread.join_calls == 1
    for socket in (push_socket, reply_socket, req_socket):
        assert socket.close_calls == 1
        assert socket.closed


def test_stats_init_interrupt_after_thread_start_is_owned_and_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A logger __init__ interrupt cannot orphan its already-started thread."""
    instance_id = "kmesh-create-stats-init-interrupt"
    workers: list[_ConstructionThread] = []

    def stat_init_then_interrupt(
        stat_logger: LMCacheStatsLogger,
        metadata: Any,
        log_interval: int,
        config: Any,
    ) -> None:
        _fake_stats_init(stat_logger, metadata, log_interval, config)
        worker = cast(Any, stat_logger).thread
        workers.append(worker)
        raise KeyboardInterrupt("injected after stats thread start")

    _patch_lightweight_construction(
        monkeypatch,
        stat_init=stat_init_then_interrupt,
        engine_init=_fake_engine_init,
    )
    config = MagicMock(spec=LMCacheEngineConfig)
    metadata = MagicMock()

    try:
        with pytest.raises(KeyboardInterrupt, match="stats thread start"):
            LMCacheEngineBuilder.get_or_create(
                instance_id,
                config,
                metadata,
                None,
                MagicMock(),
                MagicMock(),
            )

        assert len(workers) == 1
        assert not workers[0].alive
        assert workers[0].join_calls == 1
        assert LMCacheEngineBuilder.get(instance_id) is None
        assert instance_id not in LMCacheEngineBuilder._creation_owners
        assert instance_id not in LMCacheEngineBuilder._ready_generations
    finally:
        _forget_builder_generation(instance_id)


def test_engine_init_interrupt_retains_child_and_allows_builder_reentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial engine child remains close-owned without Builder deadlock."""
    instance_id = "kmesh-create-engine-init-interrupt"
    child = _StorageChild()
    reentrant_gets: list[LMCacheEngine | None] = []
    workers: list[_ConstructionThread] = []

    def tracked_stats_init(
        stat_logger: LMCacheStatsLogger,
        metadata: Any,
        log_interval: int,
        config: Any,
    ) -> None:
        _fake_stats_init(stat_logger, metadata, log_interval, config)
        workers.append(cast(Any, stat_logger).thread)

    def engine_init_then_interrupt(
        engine: LMCacheEngine,
        config: Any,
        metadata: Any,
        token_database: Any,
        gpu_connector: Any,
        broadcast_fn: Any,
        broadcast_object_fn: Any,
    ) -> None:
        del (
            config,
            metadata,
            token_database,
            gpu_connector,
            broadcast_fn,
            broadcast_object_fn,
        )
        engine.hidden_state_store = cast(Any, child)
        reentrant_gets.append(LMCacheEngineBuilder.get(instance_id))
        raise KeyboardInterrupt("injected after engine child construction")

    _patch_lightweight_construction(
        monkeypatch,
        stat_init=tracked_stats_init,
        engine_init=engine_init_then_interrupt,
    )

    try:
        with pytest.raises(KeyboardInterrupt, match="engine child construction"):
            LMCacheEngineBuilder.get_or_create(
                instance_id,
                MagicMock(spec=LMCacheEngineConfig),
                MagicMock(),
                None,
                MagicMock(),
                MagicMock(),
            )

        assert reentrant_gets == [None]
        assert child.close_calls == 1
        assert len(workers) == 1
        assert not workers[0].alive
        assert instance_id not in LMCacheEngineBuilder._creation_owners
        assert LMCacheEngineBuilder.get(instance_id) is None
    finally:
        _forget_builder_generation(instance_id)


@pytest.mark.parametrize(
    "registry_name",
    [
        "_instances",
        "_cfgs",
        "_metadatas",
        "_stat_loggers",
        "_ready_generations",
    ],
)
def test_ready_publication_boundary_retries_same_generation(
    monkeypatch: pytest.MonkeyPatch,
    registry_name: str,
) -> None:
    """Every compatibility/READY publication boundary is exact-retry safe."""
    instance_id = f"kmesh-create-publish-{registry_name}"

    class _InterruptAfterSet(dict[str, Any]):
        interrupted = False

        def __setitem__(self, key: str, value: Any) -> None:
            super().__setitem__(key, value)
            if key == instance_id and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt(f"injected after {registry_name} publication")

    registry = _InterruptAfterSet(getattr(LMCacheEngineBuilder, registry_name))
    monkeypatch.setattr(LMCacheEngineBuilder, registry_name, registry)
    stats_inits = 0
    engine_inits = 0

    def count_stats_init(*args: Any, **kwargs: Any) -> None:
        nonlocal stats_inits
        stats_inits += 1
        _fake_stats_init(*args, **kwargs)

    def count_engine_init(*args: Any, **kwargs: Any) -> None:
        nonlocal engine_inits
        engine_inits += 1
        _fake_engine_init(*args, **kwargs)

    _patch_lightweight_construction(
        monkeypatch,
        stat_init=count_stats_init,
        engine_init=count_engine_init,
    )
    config = MagicMock(spec=LMCacheEngineConfig)
    metadata = MagicMock()

    try:
        with pytest.raises(KeyboardInterrupt, match="publication"):
            LMCacheEngineBuilder.get_or_create(
                instance_id,
                config,
                metadata,
                None,
                MagicMock(),
                MagicMock(),
            )
        owner = LMCacheEngineBuilder._creation_owners[instance_id]
        expected_engine = owner.engine
        assert expected_engine is not None
        assert LMCacheEngineBuilder.get(instance_id) is None

        recovered = LMCacheEngineBuilder.recover(instance_id)

        assert recovered is expected_engine
        assert stats_inits == 1
        assert engine_inits == 1
        generation = LMCacheEngineBuilder._ready_generations[instance_id]
        assert generation.engine is expected_engine
        assert LMCacheEngineBuilder._instances[instance_id] is expected_engine
        assert LMCacheEngineBuilder._cfgs[instance_id] is config
        assert LMCacheEngineBuilder._metadatas[instance_id] is metadata
        assert LMCacheEngineBuilder._stat_loggers[instance_id] is owner.stat_logger
    finally:
        with (
            patch("lmcache.v1.cache_engine.LMCStatsMonitor.DestroyInstance"),
            patch("lmcache.v1.compute.blend.utils.LMCBlenderBuilder.drop"),
            patch("lmcache.v1.compute.models.utils.VLLMModelTracker.unregister_model"),
        ):
            if LMCacheEngineBuilder.get(instance_id) is not None:
                LMCacheEngineBuilder.destroy(instance_id)
        _forget_builder_generation(instance_id)


def test_destroy_recovers_persistent_ready_publication_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal intent drains children when READY can never be published."""
    instance_id = "kmesh-create-persistent-publish-failure"
    publish_calls = 0
    workers: list[_ConstructionThread] = []

    def tracked_stats_init(
        stat_logger: LMCacheStatsLogger,
        metadata: Any,
        log_interval: int,
        config: Any,
    ) -> None:
        _fake_stats_init(stat_logger, metadata, log_interval, config)
        workers.append(cast(Any, stat_logger).thread)

    def persistent_publish_failure(owner: Any) -> None:
        nonlocal publish_calls
        publish_calls += 1
        raise RuntimeError("persistent READY publication failure")

    _patch_lightweight_construction(
        monkeypatch,
        stat_init=tracked_stats_init,
        engine_init=_fake_engine_init,
    )
    monkeypatch.setattr(
        cache_engine_module._BuilderCreateOwner,
        "_publish_ready",
        persistent_publish_failure,
    )

    try:
        with pytest.raises(RuntimeError, match="persistent READY"):
            LMCacheEngineBuilder.get_or_create(
                instance_id,
                MagicMock(spec=LMCacheEngineConfig),
                MagicMock(),
                None,
                MagicMock(),
                MagicMock(),
            )

        assert (
            LMCacheEngineBuilder.destroy(instance_id)
            is EngineDestroyResult.ALREADY_ABSENT
        )
        # Terminal intent is sticky: destroy never retries READY publication.
        assert publish_calls == 1
        assert len(workers) == 1
        assert not workers[0].alive
        assert instance_id not in LMCacheEngineBuilder._creation_owners
        assert instance_id not in LMCacheEngineBuilder._ready_generations
        assert instance_id not in LMCacheEngineBuilder._instances
        assert instance_id not in LMCacheEngineBuilder._destroy_intents
    finally:
        _forget_builder_generation(instance_id)


def test_terminal_cancel_never_republishes_after_cleanup_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup retry cannot return to READY after engine close committed."""
    instance_id = "kmesh-create-cancelled-cleanup-interrupt"
    child = _StorageChild()
    publish_calls = 0
    cleanup_calls = 0
    original_publish = cache_engine_module._BuilderCreateOwner._publish_ready
    original_cleanup_engine = cache_engine_module._BuilderCreateOwner._cleanup_engine

    def engine_init_with_child(*args: Any, **kwargs: Any) -> None:
        engine = cast(LMCacheEngine, args[0])
        _fake_engine_init(*args, **kwargs)
        engine.hidden_state_store = cast(Any, child)

    def transient_publish_failure(owner: Any) -> None:
        nonlocal publish_calls
        publish_calls += 1
        if publish_calls == 1:
            raise RuntimeError("transient READY publication failure")
        original_publish(owner)

    def interrupt_after_engine_cleanup(owner: Any) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        original_cleanup_engine(owner)
        if cleanup_calls == 1:
            raise KeyboardInterrupt("injected after engine cleanup commit")

    _patch_lightweight_construction(
        monkeypatch,
        stat_init=_fake_stats_init,
        engine_init=engine_init_with_child,
    )
    monkeypatch.setattr(
        cache_engine_module._BuilderCreateOwner,
        "_publish_ready",
        transient_publish_failure,
    )
    monkeypatch.setattr(
        cache_engine_module._BuilderCreateOwner,
        "_cleanup_engine",
        interrupt_after_engine_cleanup,
    )

    try:
        with pytest.raises(RuntimeError, match="transient READY"):
            LMCacheEngineBuilder.get_or_create(
                instance_id,
                MagicMock(spec=LMCacheEngineConfig),
                MagicMock(),
                None,
                MagicMock(),
                MagicMock(),
            )

        with pytest.raises(KeyboardInterrupt, match="engine cleanup commit"):
            LMCacheEngineBuilder.destroy(instance_id)

        assert publish_calls == 1
        assert cleanup_calls == 1
        assert child.close_calls == 1
        assert instance_id in LMCacheEngineBuilder._creation_owners
        assert instance_id in LMCacheEngineBuilder._destroy_intents
        assert instance_id not in LMCacheEngineBuilder._ready_generations

        assert (
            LMCacheEngineBuilder.destroy(instance_id)
            is EngineDestroyResult.ALREADY_ABSENT
        )
        assert publish_calls == 1
        assert cleanup_calls == 2
        assert child.close_calls == 1
        assert instance_id not in LMCacheEngineBuilder._creation_owners
        assert instance_id not in LMCacheEngineBuilder._destroy_intents
        assert instance_id not in LMCacheEngineBuilder._ready_generations
        assert instance_id not in LMCacheEngineBuilder._instances
    finally:
        _forget_builder_generation(instance_id)


def test_destroy_intent_wins_over_inflight_creation_without_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal request fences get/recover before CREATING reaches READY."""
    instance_id = "kmesh-destroy-vs-create"
    stats_entered = threading.Event()
    release_stats = threading.Event()
    destroy_intent_published = threading.Event()
    engines: list[LMCacheEngine] = []

    class _ObservedDestroyIntents(set[str]):
        def add(self, value: str) -> None:
            super().add(value)
            if value == instance_id:
                destroy_intent_published.set()

    monkeypatch.setattr(
        LMCacheEngineBuilder,
        "_destroy_intents",
        _ObservedDestroyIntents(LMCacheEngineBuilder._destroy_intents),
    )

    def blocking_stats_init(
        stat_logger: LMCacheStatsLogger,
        metadata: Any,
        log_interval: int,
        config: Any,
    ) -> None:
        _fake_stats_init(stat_logger, metadata, log_interval, config)
        stats_entered.set()
        assert release_stats.wait(timeout=5.0)

    def tracked_engine_init(*args: Any, **kwargs: Any) -> None:
        engine = cast(LMCacheEngine, args[0])
        engines.append(engine)
        _fake_engine_init(*args, **kwargs)

    _patch_lightweight_construction(
        monkeypatch,
        stat_init=blocking_stats_init,
        engine_init=tracked_engine_init,
    )
    config = MagicMock(spec=LMCacheEngineConfig)
    metadata = MagicMock()
    creator_results: list[LMCacheEngine] = []
    creator_errors: list[BaseException] = []
    destroy_results: list[EngineDestroyResult] = []
    destroy_errors: list[BaseException] = []

    def create() -> None:
        try:
            creator_results.append(
                LMCacheEngineBuilder.get_or_create(
                    instance_id,
                    config,
                    metadata,
                    None,
                    MagicMock(),
                    MagicMock(),
                )
            )
        except BaseException as error:
            creator_errors.append(error)

    def destroy() -> None:
        try:
            destroy_results.append(LMCacheEngineBuilder.destroy(instance_id))
        except BaseException as error:
            destroy_errors.append(error)

    creator = threading.Thread(target=create, name="kmesh-create-racer")
    destroyer = threading.Thread(target=destroy, name="kmesh-destroy-racer")
    try:
        creator.start()
        assert stats_entered.wait(timeout=5.0)
        assert LMCacheEngineBuilder.get(instance_id) is None

        destroyer.start()
        assert destroy_intent_published.wait(timeout=5.0)
        with pytest.raises(RuntimeError, match="terminal cleanup ownership"):
            LMCacheEngineBuilder.recover(instance_id)

        release_stats.set()
        creator.join(timeout=5.0)
        destroyer.join(timeout=5.0)

        assert not creator.is_alive()
        assert not destroyer.is_alive()
        assert creator_results == []
        assert len(creator_errors) == 1
        assert "construction was cancelled" in str(creator_errors[0])
        assert destroy_errors == []
        assert destroy_results == [EngineDestroyResult.ALREADY_ABSENT]
        assert len(engines) == 1
        assert LMCacheEngineBuilder.get(instance_id) is None
        assert instance_id not in LMCacheEngineBuilder._creation_owners
        assert instance_id not in LMCacheEngineBuilder._destroy_owners
        assert instance_id not in LMCacheEngineBuilder._destroy_intents
        assert instance_id not in LMCacheEngineBuilder._ready_generations
    finally:
        release_stats.set()
        creator.join(timeout=5.0)
        if destroyer.ident is not None:
            destroyer.join(timeout=5.0)
        _forget_builder_generation(instance_id, *engines)
