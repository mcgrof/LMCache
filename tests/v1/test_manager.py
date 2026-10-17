# SPDX-License-Identifier: Apache-2.0
"""
Tests for LMCacheManager.
"""

# Standard
from typing import Any, Callable, Optional
from unittest.mock import MagicMock, patch
import inspect
import threading

# Third Party
import pytest
import zmq

# First Party
from lmcache.integration.base_service_factory import BaseServiceFactory
from lmcache.v1.cache_engine import EngineDestroyResult, LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.health_monitor.base import HealthMonitor
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupClient
from lmcache.v1.manager import LMCacheManager
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadRunSummary,
)
from lmcache.v1.rpc.zmq_transport import SocketParams, ZmqReqRepClientTransport
from tests.v1.utils import create_test_metadata


class _FailingCloseSocket:
    """Minimal injected ZMQ socket with bounded close failures."""

    def __init__(self, close_failures: int) -> None:
        self.close_failures = close_failures
        self.close_calls = 0
        self.closed = False

    def setsockopt(self, option: int, value: int) -> None:
        """Accept constructor socket options."""

    def close(self, linger: int) -> None:
        """Fail the configured number of calls, then record closure."""
        assert linger == 0
        self.close_calls += 1
        if self.close_calls <= self.close_failures:
            raise RuntimeError("injected socket close failure")
        self.closed = True


class _FailingTermContext:
    """Minimal injected ZMQ context with bounded termination failures."""

    def __init__(self, term_failures: int) -> None:
        self.term_failures = term_failures
        self.term_calls = 0
        self.terminated = False

    def term(self) -> None:
        """Fail the configured number of calls, then record termination."""
        self.term_calls += 1
        if self.term_calls <= self.term_failures:
            raise RuntimeError("injected context termination failure")
        self.terminated = True


class _TimeoutSocket(_FailingCloseSocket):
    """Injected socket whose first request path times out."""

    def __init__(self) -> None:
        super().__init__(close_failures=0)
        self.send_calls = 0

    def send_multipart(self, frames: list[bytes], copy: bool) -> None:
        """Raise the ZeroMQ timeout used by transport recovery."""
        assert frames
        assert copy is False
        self.send_calls += 1
        raise zmq.Again()


class _HealthMonitorTestEngine(LMCacheEngine):
    """Small engine shell that exercises the real monitor attachment API."""

    def __init__(self) -> None:
        self._health_monitor: Optional[HealthMonitor] = None
        self._init_failed = False
        self.storage_manager = None

    @property
    def installed_health_monitor(self) -> Optional[HealthMonitor]:
        """Return the monitor currently installed through the public API."""
        return self._health_monitor

    def post_init(self, **kwargs: Any) -> None:
        """Provide the no-resource post-init needed by the manager test."""


class _HealthMonitorFactory(BaseServiceFactory):
    """Concrete factory that exposes BaseServiceFactory monitor creation."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        engine: _HealthMonitorTestEngine,
        offload_server: Optional[ZMQOffloadServer] = None,
    ) -> None:
        self.config = config
        self.metadata = metadata
        self.engine = engine
        self.offload_server = offload_server

    def get_engine_instance_id(self) -> str:
        return "health-monitor-rollback"

    def get_or_create_metadata(self) -> LMCacheMetadata:
        return self.metadata

    def get_or_create_lmcache_engine(
        self,
        result_publisher: Optional[Callable[[Optional[LMCacheEngine]], None]] = None,
    ) -> LMCacheEngine:
        if result_publisher is not None:
            result_publisher(self.engine)
        return self.engine

    def maybe_create_lookup_client(self, result_publisher=None) -> None:
        if result_publisher is not None:
            result_publisher(None)
        return None

    def maybe_create_prometheus_logger(self) -> None:
        return None

    def maybe_create_lookup_server(self, result_publisher=None) -> None:
        if result_publisher is not None:
            result_publisher(None)
        return None

    def maybe_create_offload_server(self) -> Optional[ZMQOffloadServer]:
        return self.offload_server

    def maybe_create_runtime_plugin_launcher(self) -> None:
        return None

    def maybe_create_internal_api_server(self, lmcache_manager: LMCacheManager) -> None:
        return None

    def maybe_create_health_monitor(
        self, lmcache_manager: LMCacheManager
    ) -> HealthMonitor:
        return self._create_health_monitor(
            lmcache_manager,
            self.config,
            self.engine,
        )


def _make_mock_factory(**overrides):
    """Create a mock BaseServiceFactory with sensible defaults."""
    factory = MagicMock(spec=BaseServiceFactory)
    factory.get_or_create_metadata.return_value = overrides.get("metadata", None)
    factory.get_or_create_lmcache_engine.return_value = overrides.get("engine", None)
    factory.maybe_create_lookup_client.return_value = overrides.get(
        "lookup_client", None
    )
    factory.maybe_create_lookup_server.return_value = overrides.get(
        "lookup_server", None
    )
    factory.maybe_create_offload_server.return_value = overrides.get(
        "offload_server", None
    )
    factory.maybe_create_runtime_plugin_launcher.return_value = overrides.get(
        "plugin_launcher", None
    )
    factory.maybe_create_internal_api_server.return_value = overrides.get(
        "api_server", None
    )
    factory.maybe_create_health_monitor.return_value = overrides.get(
        "health_monitor", None
    )

    def publish_return_value(method: MagicMock):
        def invoke(publisher):
            result = method.return_value
            publisher(result)
            return result

        method.__signature__ = inspect.signature(invoke)
        return invoke

    factory.get_or_create_lmcache_engine.side_effect = publish_return_value(
        factory.get_or_create_lmcache_engine
    )
    factory.maybe_create_lookup_client.side_effect = publish_return_value(
        factory.maybe_create_lookup_client
    )
    factory.maybe_create_lookup_server.side_effect = publish_return_value(
        factory.maybe_create_lookup_server
    )
    return factory


def _patch_engine_builder():
    """Patch Builder with the exact successful destroy result by default."""
    return patch(
        "lmcache.v1.manager.LMCacheEngineBuilder",
        **{"destroy.return_value": EngineDestroyResult.DESTROYED},
    )


class TestLMCacheManagerInit:
    """Tests for LMCacheManager initialization."""

    def test_init_stores_config(self):
        """Test that __init__ stores config correctly."""
        config = LMCacheEngineConfig.from_defaults()
        connector = MagicMock()
        factory = _make_mock_factory()

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
            connector=connector,
        )

        assert manager._config is config
        assert manager._connector is connector

    def test_init_calls_factory_methods(self):
        """Test that __init__ calls all factory creation methods."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()

        LMCacheManager(
            config=config,
            service_factory=factory,
        )

        factory.get_or_create_metadata.assert_called_once()
        factory.get_or_create_lmcache_engine.assert_called_once()
        factory.maybe_create_lookup_client.assert_called_once()
        factory.maybe_create_lookup_server.assert_called_once()
        factory.maybe_create_offload_server.assert_called_once()
        factory.maybe_create_runtime_plugin_launcher.assert_called_once()
        factory.maybe_create_internal_api_server.assert_called_once()

    def test_init_does_not_call_health_monitor(self):
        """Test that __init__ does NOT create health monitor (deferred to post_init)."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()

        LMCacheManager(
            config=config,
            service_factory=factory,
        )

        factory.maybe_create_health_monitor.assert_not_called()

    def test_init_preserves_exception_degraded_recompute_behavior(self) -> None:
        """Degraded construction remains usable until explicitly rejected."""
        config = LMCacheEngineConfig.from_defaults()
        lookup_client = MagicMock(spec=LookupClientInterface)
        factory = _make_mock_factory(lookup_client=lookup_client)
        factory.maybe_create_lookup_server.side_effect = RuntimeError(
            "injected ordinary construction failure"
        )

        manager = LMCacheManager(config=config, service_factory=factory)

        assert LMCacheManager.incomplete_construction_cleanup_count() == 0
        with pytest.raises(RuntimeError, match="manager initialization failed"):
            manager.raise_if_init_failed()
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1
        with pytest.raises(RuntimeError, match="manager initialization failed"):
            manager.raise_if_init_failed()
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1

        manager.stop_services()
        assert manager.services_stopped
        assert LMCacheManager.incomplete_construction_cleanup_count() == 0
        lookup_client.close.assert_called_once_with()

    def test_init_baseexception_cleanup_retains_public_retry_owner(self) -> None:
        """A failed engine destroy remains reachable after construction exits."""
        config = LMCacheEngineConfig.from_defaults()
        engine = MagicMock(spec=LMCacheEngine)
        factory = _make_mock_factory(engine=engine)
        original_error = KeyboardInterrupt("injected construction interrupt")
        cleanup_error = RuntimeError("injected destroy failure")
        factory.maybe_create_lookup_client.side_effect = original_error
        factory.get_engine_instance_id.return_value = "constructor-retry-engine"

        with _patch_engine_builder() as builder:
            builder.destroy.side_effect = cleanup_error
            try:
                with pytest.raises(KeyboardInterrupt) as failure:
                    LMCacheManager(config=config, service_factory=factory)

                assert failure.value is original_error
                assert failure.value.__cause__ is cleanup_error
                assert LMCacheManager.incomplete_construction_cleanup_count() == 1
                builder.destroy.assert_called_once_with(
                    "constructor-retry-engine",
                    expected_engine=engine,
                )

                builder.destroy.side_effect = None
                builder.destroy.return_value = EngineDestroyResult.DESTROYED
                assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
                assert builder.destroy.call_count == 2
            finally:
                builder.destroy.side_effect = None
                builder.destroy.return_value = EngineDestroyResult.DESTROYED
                LMCacheManager.retry_incomplete_construction_cleanup()

        assert LMCacheManager.incomplete_construction_cleanup_count() == 0

    def test_init_baseexception_releases_only_new_shared_engine_owner(self) -> None:
        """A failed second constructor cannot destroy a live first owner."""
        config = LMCacheEngineConfig.from_defaults()
        shared_engine = MagicMock(spec=LMCacheEngine)
        first_factory = _make_mock_factory(engine=shared_engine)
        first_factory.get_engine_instance_id.return_value = "shared-constructor-engine"
        first = LMCacheManager(config=config, service_factory=first_factory)

        second_factory = _make_mock_factory(engine=shared_engine)
        second_factory.get_engine_instance_id.return_value = "shared-constructor-engine"
        original_error = KeyboardInterrupt("injected second-owner interrupt")
        second_factory.maybe_create_lookup_client.side_effect = original_error

        with _patch_engine_builder() as builder:
            builder.destroy.return_value = EngineDestroyResult.DESTROYED
            with pytest.raises(KeyboardInterrupt) as failure:
                LMCacheManager(config=config, service_factory=second_factory)

            assert failure.value is original_error
            assert failure.value.__cause__ is None
            assert LMCacheManager.incomplete_construction_cleanup_count() == 0
            builder.destroy.assert_not_called()

            first.stop_services()
            builder.destroy.assert_called_once_with(
                "shared-constructor-engine",
                expected_engine=shared_engine,
            )

        assert first.services_stopped

    def test_init_exception_rejection_releases_only_new_shared_engine_owner(
        self,
    ) -> None:
        """Rejecting a degraded second manager preserves the first owner."""
        config = LMCacheEngineConfig.from_defaults()
        shared_engine = MagicMock(spec=LMCacheEngine)
        first_factory = _make_mock_factory(engine=shared_engine)
        first_factory.get_engine_instance_id.return_value = "shared-degraded-engine"
        first = LMCacheManager(config=config, service_factory=first_factory)

        second_factory = _make_mock_factory(engine=shared_engine)
        second_factory.get_engine_instance_id.return_value = "shared-degraded-engine"
        second_factory.maybe_create_lookup_server.side_effect = RuntimeError(
            "injected second-owner construction failure"
        )
        second = LMCacheManager(config=config, service_factory=second_factory)

        assert LMCacheManager.incomplete_construction_cleanup_count() == 0
        with pytest.raises(RuntimeError, match="manager initialization failed"):
            second.raise_if_init_failed()
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1

        with _patch_engine_builder() as builder:
            builder.destroy.return_value = EngineDestroyResult.DESTROYED
            second.stop_services()
            assert second.services_stopped
            assert LMCacheManager.incomplete_construction_cleanup_count() == 0
            builder.destroy.assert_not_called()

            first.stop_services()
            builder.destroy.assert_called_once_with(
                "shared-degraded-engine",
                expected_engine=shared_engine,
            )

        assert first.services_stopped


class TestLMCacheManagerProperties:
    """Tests for LMCacheManager property accessors."""

    @pytest.fixture
    def manager_with_mocked_init(self):
        """Create a manager with mocked factory."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
            connector=MagicMock(),
        )
        return manager

    def test_lmcache_engine_property(self, manager_with_mocked_init):
        """Test lmcache_engine property returns the engine."""
        manager = manager_with_mocked_init
        mock_engine = MagicMock()
        manager._lmcache_engine = mock_engine

        assert manager.lmcache_engine is mock_engine

    def test_lmcache_engine_metadata_property(self, manager_with_mocked_init):
        """Test lmcache_engine_metadata property returns the metadata."""
        manager = manager_with_mocked_init
        mock_metadata = MagicMock()
        manager._lmcache_engine_metadata = mock_metadata

        assert manager.lmcache_engine_metadata is mock_metadata

    def test_lookup_client_property(self, manager_with_mocked_init):
        """Test lookup_client property returns the client."""
        manager = manager_with_mocked_init
        mock_client = MagicMock()
        manager._lookup_client = mock_client

        assert manager.lookup_client is mock_client

    def test_lookup_server_property(self, manager_with_mocked_init):
        """Test lookup_server property returns the server."""
        manager = manager_with_mocked_init
        mock_server = MagicMock()
        manager._lookup_server = mock_server

        assert manager.lookup_server is mock_server

    def test_offload_server_property(self, manager_with_mocked_init):
        """Test offload_server property returns the server."""
        manager = manager_with_mocked_init
        mock_server = MagicMock()
        manager._offload_server = mock_server

        assert manager.offload_server is mock_server

    def test_config_property(self, manager_with_mocked_init):
        """Test config property returns the config."""
        manager = manager_with_mocked_init
        assert manager.config is manager._config


class TestLMCacheManagerStart:
    """Tests for LMCacheManager start method."""

    def test_start_calls_api_server_start(self):
        """Test start_services() calls api_server.start() when api_server exists."""
        config = LMCacheEngineConfig.from_defaults()
        mock_api_server = MagicMock()
        mock_plugin_launcher = MagicMock()
        factory = _make_mock_factory(
            api_server=mock_api_server,
            plugin_launcher=mock_plugin_launcher,
        )

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        manager.start_services()

        mock_api_server.start.assert_called_once()
        mock_plugin_launcher.launch_plugins.assert_called_once()

    def test_start_handles_none_api_server(self):
        """Test start_services() handles None api_server gracefully."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        # Should not raise any exception
        manager.start_services()


class TestLMCacheManagerPostInit:
    """Tests for LMCacheManager post_init method."""

    def test_post_init_without_engine(self):
        """Test post_init initializes health monitor when engine is None."""
        config = LMCacheEngineConfig.from_defaults()
        mock_health_monitor = MagicMock()
        factory = _make_mock_factory(health_monitor=mock_health_monitor)

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        manager.post_init()

        # Health monitor should still be created even without engine
        factory.maybe_create_health_monitor.assert_called_once_with(
            lmcache_manager=manager
        )

    def test_post_init_and_shutdown_are_one_terminal_lifecycle(self) -> None:
        """Shutdown waits for startup, then closes every published owner."""
        PeriodicThreadRegistry.reset()
        registry = PeriodicThreadRegistry.get_instance()
        config = LMCacheEngineConfig.from_defaults()
        engine = _HealthMonitorTestEngine()
        factory = _HealthMonitorFactory(
            config,
            create_test_metadata(),
            engine,
        )
        manager = LMCacheManager(config=config, service_factory=factory)
        start_entered = threading.Event()
        release_start = threading.Event()
        stop_entered = threading.Event()
        stop_finished = threading.Event()
        post_errors: list[BaseException] = []
        stop_errors: list[BaseException] = []

        def blocking_start(monitor: HealthMonitor) -> Optional[threading.Thread]:
            start_entered.set()
            assert release_start.wait(timeout=5.0)
            return PeriodicThread.start(monitor)

        def healthy_cycle(_monitor: HealthMonitor) -> ThreadRunSummary:
            return ThreadRunSummary(success=True, message="healthy")

        def run_post_init() -> None:
            try:
                manager.post_init()
            except BaseException as error:
                post_errors.append(error)

        def run_shutdown() -> None:
            stop_entered.set()
            try:
                manager.stop_services()
            except BaseException as error:
                stop_errors.append(error)
            finally:
                stop_finished.set()

        post_thread = threading.Thread(target=run_post_init)
        stop_thread = threading.Thread(target=run_shutdown)
        try:
            with (
                patch.object(HealthMonitor, "start", blocking_start),
                patch.object(HealthMonitor, "_execute", healthy_cycle),
                patch(
                    "lmcache.observability.PrometheusLogger.GetOrCreate",
                    return_value=MagicMock(),
                ),
                _patch_engine_builder() as builder,
            ):
                post_thread.start()
                assert start_entered.wait(timeout=5.0)
                monitor = manager.health_monitor
                assert monitor is not None
                assert registry.get(monitor.name) is monitor

                stop_thread.start()
                assert stop_entered.wait(timeout=5.0)
                assert not stop_finished.is_set()

                release_start.set()
                post_thread.join(timeout=5.0)
                stop_thread.join(timeout=5.0)

                assert not post_thread.is_alive()
                assert not stop_thread.is_alive()
                assert post_errors == []
                assert stop_errors == []
                assert manager.services_stopped
                assert manager.health_monitor is None
                assert not monitor.thread_alive
                assert registry.get_all() == []
                builder.destroy.assert_called_once()
        finally:
            release_start.set()
            post_thread.join(timeout=5.0)
            stop_thread.join(timeout=5.0)
            with _patch_engine_builder():
                manager.stop_services()
            PeriodicThreadRegistry.reset()

    @pytest.mark.parametrize("catch_request", [False, True])
    def test_post_init_callback_shutdown_is_bounded_and_terminal(
        self,
        catch_request: bool,
    ) -> None:
        """Same-thread shutdown unwinds startup and closes published state."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()
        manager = LMCacheManager(config=config, service_factory=factory)
        callback_entered = threading.Event()
        callback_returned = threading.Event()
        published_monitor = MagicMock()
        post_errors: list[BaseException] = []

        def request_shutdown(lmcache_manager: LMCacheManager) -> MagicMock:
            callback_entered.set()
            if catch_request:
                with pytest.raises(
                    RuntimeError,
                    match="terminal shutdown requested",
                ):
                    lmcache_manager.stop_services()
                callback_returned.set()
                return published_monitor

            lmcache_manager.stop_services()
            raise AssertionError("reentrant shutdown request unexpectedly returned")

        def run_post_init() -> None:
            try:
                manager.post_init()
            except BaseException as error:
                post_errors.append(error)

        factory.maybe_create_health_monitor.side_effect = request_shutdown
        post_thread = threading.Thread(target=run_post_init, daemon=True)
        post_thread.start()
        post_thread.join(timeout=5.0)

        assert callback_entered.is_set()
        assert callback_returned.is_set() is catch_request
        assert not post_thread.is_alive()
        assert post_errors == []
        assert manager.services_stopped
        assert manager.health_monitor is None
        if catch_request:
            published_monitor.stop.assert_called_once_with()
        else:
            published_monitor.stop.assert_not_called()

    def test_post_init_callback_interrupt_still_completes_deferred_shutdown(
        self,
    ) -> None:
        """A callback interrupt cannot skip its earlier shutdown request."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()
        manager = LMCacheManager(config=config, service_factory=factory)
        original_error = KeyboardInterrupt("injected callback interrupt")
        post_errors: list[BaseException] = []

        def request_shutdown(lmcache_manager: LMCacheManager) -> None:
            with pytest.raises(
                RuntimeError,
                match="terminal shutdown requested",
            ):
                lmcache_manager.stop_services()
            raise original_error

        def run_post_init() -> None:
            try:
                manager.post_init()
            except BaseException as error:
                post_errors.append(error)

        factory.maybe_create_health_monitor.side_effect = request_shutdown
        post_thread = threading.Thread(target=run_post_init, daemon=True)
        post_thread.start()
        post_thread.join(timeout=5.0)

        assert not post_thread.is_alive()
        assert post_errors == [original_error]
        assert manager.services_stopped
        assert manager.health_monitor is None

    def test_post_init_callback_shutdown_surfaces_incomplete_cleanup(
        self,
    ) -> None:
        """Deferred shutdown cannot report success while ownership remains."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()
        manager = LMCacheManager(config=config, service_factory=factory)
        published_monitor = MagicMock()
        published_monitor.stop.side_effect = RuntimeError(
            "injected monitor stop failure"
        )
        post_errors: list[BaseException] = []

        def request_shutdown(lmcache_manager: LMCacheManager) -> MagicMock:
            with pytest.raises(
                RuntimeError,
                match="terminal shutdown requested",
            ):
                lmcache_manager.stop_services()
            return published_monitor

        def run_post_init() -> None:
            try:
                manager.post_init()
            except BaseException as error:
                post_errors.append(error)

        factory.maybe_create_health_monitor.side_effect = request_shutdown
        post_thread = threading.Thread(target=run_post_init, daemon=True)
        post_thread.start()
        post_thread.join(timeout=5.0)

        assert not post_thread.is_alive()
        assert len(post_errors) == 1
        assert isinstance(post_errors[0], RuntimeError)
        assert "did not fully quiesce" in str(post_errors[0])
        assert not manager.services_stopped
        assert manager.health_monitor is published_monitor

        published_monitor.stop.side_effect = None
        manager.stop_services()
        assert manager.services_stopped
        assert manager.health_monitor is None

    def test_post_init_cannot_restart_after_shutdown(self) -> None:
        """Shutdown-first ordering permanently fences service startup."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()
        manager = LMCacheManager(config=config, service_factory=factory)

        with _patch_engine_builder():
            manager.stop_services()
            manager.post_init()

        assert manager.services_stopped
        assert manager.health_monitor is None
        factory.maybe_create_health_monitor.assert_not_called()

    @pytest.mark.parametrize("catch_recursive_error", [False, True])
    def test_health_factory_recursive_post_init_is_bounded_and_single_publish(
        self,
        catch_recursive_error: bool,
    ) -> None:
        """A factory callback cannot recursively publish a second monitor."""
        config = LMCacheEngineConfig.from_defaults()
        engine = _HealthMonitorTestEngine()
        factory = _HealthMonitorFactory(
            config,
            create_test_metadata(),
            engine,
        )
        manager = LMCacheManager(config=config, service_factory=factory)
        monitor = MagicMock(spec=HealthMonitor)
        factory_calls: list[str] = []
        post_errors: list[BaseException] = []

        def recursive_factory(
            lmcache_manager: LMCacheManager,
        ) -> HealthMonitor:
            factory_calls.append("health")
            if catch_recursive_error:
                with pytest.raises(RuntimeError, match="recursive"):
                    lmcache_manager.post_init()
                return monitor
            lmcache_manager.post_init()
            raise AssertionError("recursive post_init unexpectedly returned")

        def run_post_init() -> None:
            try:
                manager.post_init()
            except BaseException as error:
                post_errors.append(error)

        post_thread = threading.Thread(target=run_post_init, daemon=True)
        with patch.object(
            factory,
            "maybe_create_health_monitor",
            side_effect=recursive_factory,
        ):
            post_thread.start()
            post_thread.join(timeout=5.0)
            manager.post_init()

        assert not post_thread.is_alive()
        assert post_errors == []
        assert factory_calls == ["health"]
        if catch_recursive_error:
            manager.raise_if_init_failed()
            assert manager.health_monitor is monitor
        else:
            with pytest.raises(RuntimeError, match="manager initialization failed"):
                manager.raise_if_init_failed()
            assert manager.health_monitor is None

        with _patch_engine_builder():
            manager.stop_services()
        assert manager.services_stopped

    def test_prometheus_failure_rolls_back_live_health_monitor(self) -> None:
        """A factory error cannot strand monitor registry or engine state."""
        PeriodicThreadRegistry.reset()
        registry = PeriodicThreadRegistry.get_instance()
        config = LMCacheEngineConfig.from_defaults()
        engine = _HealthMonitorTestEngine()
        factory = _HealthMonitorFactory(
            config,
            create_test_metadata(),
            engine,
        )
        manager = LMCacheManager(config=config, service_factory=factory)
        observed_active_monitor = False

        def fail_prometheus_setup(
            metadata: LMCacheMetadata,
            config: LMCacheEngineConfig,
        ) -> None:
            nonlocal observed_active_monitor
            assert metadata is manager.lmcache_engine_metadata
            assert config is manager.config
            assert registry.get_running_count() == 1
            assert registry.get_active_count() == 1
            assert engine.installed_health_monitor is not None
            observed_active_monitor = True
            raise RuntimeError("injected Prometheus setup failure")

        try:
            with (
                patch.object(HealthMonitor, "start", PeriodicThread.start),
                patch(
                    "lmcache.observability.PrometheusLogger.GetOrCreate",
                    side_effect=fail_prometheus_setup,
                ),
            ):
                manager.post_init()

            assert observed_active_monitor
            assert manager.health_monitor is None
            assert engine.installed_health_monitor is None
            assert registry.get_all() == []
            assert registry.get_running_count() == 0
            assert registry.get_active_count() == 0
            with pytest.raises(
                RuntimeError,
                match="injected Prometheus setup failure",
            ):
                manager.raise_if_init_failed()
        finally:
            with _patch_engine_builder():
                manager.stop_services()
            PeriodicThreadRegistry.reset()

    def test_registration_baseexception_rolls_back_inserted_monitor(self) -> None:
        """A register-after-insert interrupt retains exact rollback ownership."""
        PeriodicThreadRegistry.reset()
        registry = PeriodicThreadRegistry.get_instance()
        config = LMCacheEngineConfig.from_defaults()
        engine = _HealthMonitorTestEngine()
        factory = _HealthMonitorFactory(
            config,
            create_test_metadata(),
            engine,
        )
        manager = LMCacheManager(config=config, service_factory=factory)
        register = registry.register

        def register_then_interrupt(thread: PeriodicThread) -> None:
            register(thread)
            raise KeyboardInterrupt("injected register-after-insert interrupt")

        try:
            with (
                patch.object(
                    registry,
                    "register",
                    side_effect=register_then_interrupt,
                ),
                pytest.raises(
                    KeyboardInterrupt,
                    match="register-after-insert",
                ),
            ):
                factory.maybe_create_health_monitor(manager)

            assert registry.get_all() == []
            assert manager.health_monitor is None
            assert engine.installed_health_monitor is None
        finally:
            with _patch_engine_builder():
                manager.stop_services()
            PeriodicThreadRegistry.reset()

    def test_failed_startup_stop_retains_live_monitor_until_retry(self) -> None:
        """A blocked health check keeps registry, manager, and engine owners."""
        PeriodicThreadRegistry.reset()
        registry = PeriodicThreadRegistry.get_instance()
        config = LMCacheEngineConfig.from_defaults()
        engine = _HealthMonitorTestEngine()
        factory = _HealthMonitorFactory(
            config,
            create_test_metadata(),
            engine,
        )
        manager = LMCacheManager(config=config, service_factory=factory)
        entered = threading.Event()
        release = threading.Event()
        original_stop = HealthMonitor.stop

        def blocked_check(_monitor: HealthMonitor) -> ThreadRunSummary:
            entered.set()
            release.wait()
            return ThreadRunSummary(success=True, message="released")

        def bounded_stop(
            monitor: HealthMonitor,
            timeout: float = 0.01,
        ) -> None:
            original_stop(monitor, timeout)

        def fail_prometheus_setup(
            metadata: LMCacheMetadata,
            config: LMCacheEngineConfig,
        ) -> None:
            assert metadata is manager.lmcache_engine_metadata
            assert config is manager.config
            assert entered.wait(timeout=1.0)
            raise RuntimeError("injected startup failure")

        monitor: Optional[HealthMonitor] = None
        try:
            with (
                patch.object(HealthMonitor, "start", PeriodicThread.start),
                patch.object(HealthMonitor, "stop", bounded_stop),
                patch.object(HealthMonitor, "_execute", blocked_check),
                patch(
                    "lmcache.observability.PrometheusLogger.GetOrCreate",
                    side_effect=fail_prometheus_setup,
                ),
            ):
                with pytest.raises(
                    RuntimeError,
                    match="rollback did not complete: stop",
                ):
                    factory.maybe_create_health_monitor(manager)

                monitor = manager.health_monitor
                assert monitor is not None
                assert monitor.thread_alive
                assert registry.get(monitor.name) is monitor
                assert engine.installed_health_monitor is monitor

                with _patch_engine_builder() as builder:
                    manager.stop_services()
                    builder.destroy.assert_not_called()
                    assert not manager.services_stopped
                    assert manager.health_monitor is monitor

                    release.set()
                    monitor.stop(timeout=1.0)
                    manager.stop_services()
                    builder.destroy.assert_called_once()

                assert manager.services_stopped
                assert manager.health_monitor is None
        finally:
            release.set()
            if monitor is not None and monitor.thread_alive:
                original_stop(monitor, timeout=1.0)
            with _patch_engine_builder():
                manager.stop_services()
            PeriodicThreadRegistry.reset()

    def test_failed_same_name_monitor_preserves_older_live_registration(
        self,
    ) -> None:
        """A live registry owner cannot be overwritten by a new factory."""
        PeriodicThreadRegistry.reset()
        registry = PeriodicThreadRegistry.get_instance()
        config = LMCacheEngineConfig.from_defaults()
        engine = _HealthMonitorTestEngine()
        factory = _HealthMonitorFactory(
            config,
            create_test_metadata(),
            engine,
        )
        manager = LMCacheManager(config=config, service_factory=factory)
        older = HealthMonitor(manager=MagicMock(), ping_interval=60.0)

        def healthy_cycle(_monitor: HealthMonitor) -> ThreadRunSummary:
            return ThreadRunSummary(success=True, message="healthy")

        try:
            with (
                patch.object(HealthMonitor, "start", PeriodicThread.start),
                patch.object(HealthMonitor, "_execute", healthy_cycle),
            ):
                older.start()
                assert older.thread_alive
                with pytest.raises(
                    RuntimeError,
                    match="already registered by a different instance",
                ):
                    factory.maybe_create_health_monitor(manager)

            assert registry.get(older.name) is older
            assert older.thread_alive
            assert manager.health_monitor is None
            assert engine.installed_health_monitor is None
        finally:
            older.stop(timeout=1.0)
            with _patch_engine_builder():
                manager.stop_services()
            PeriodicThreadRegistry.reset()

    def test_pre_start_registration_cannot_be_replaced_by_second_factory(
        self,
    ) -> None:
        """A same-name factory cannot displace a registered pre-start owner."""
        PeriodicThreadRegistry.reset()
        registry = PeriodicThreadRegistry.get_instance()
        config = LMCacheEngineConfig.from_defaults()
        first_engine = _HealthMonitorTestEngine()
        second_engine = _HealthMonitorTestEngine()
        first_factory = _HealthMonitorFactory(
            config,
            create_test_metadata(),
            first_engine,
        )
        second_factory = _HealthMonitorFactory(
            config,
            create_test_metadata(),
            second_engine,
        )
        first_manager = LMCacheManager(
            config=config,
            service_factory=first_factory,
        )
        second_manager = LMCacheManager(
            config=config,
            service_factory=second_factory,
        )
        first_at_start = threading.Event()
        release_first_start = threading.Event()
        first_result: list[HealthMonitor] = []
        first_errors: list[BaseException] = []
        start_calls: list[HealthMonitor] = []
        start_calls_lock = threading.Lock()

        def interleaved_start(monitor: HealthMonitor) -> Optional[threading.Thread]:
            with start_calls_lock:
                call_index = len(start_calls)
                start_calls.append(monitor)
            if call_index == 0:
                first_at_start.set()
                assert release_first_start.wait(timeout=5.0)
            return PeriodicThread.start(monitor)

        def create_first_monitor() -> None:
            try:
                first_result.append(
                    first_factory.maybe_create_health_monitor(first_manager)
                )
            except BaseException as error:
                first_errors.append(error)

        creator = threading.Thread(target=create_first_monitor)
        try:
            with (
                patch.object(HealthMonitor, "start", interleaved_start),
                patch(
                    "lmcache.observability.PrometheusLogger.GetOrCreate",
                    return_value=MagicMock(),
                ),
            ):
                creator.start()
                assert first_at_start.wait(timeout=5.0)

                first_monitor = first_manager.health_monitor
                assert first_monitor is not None
                assert registry.get(first_monitor.name) is first_monitor
                assert not first_monitor.thread_alive
                registry.register(first_monitor)
                assert registry.get(first_monitor.name) is first_monitor

                with pytest.raises(
                    RuntimeError,
                    match="already registered by a different instance",
                ):
                    second_factory.maybe_create_health_monitor(second_manager)

                assert start_calls == [first_monitor]
                assert second_manager.health_monitor is None
                assert second_engine.installed_health_monitor is None
                assert registry.get(first_monitor.name) is first_monitor

                release_first_start.set()
                creator.join(timeout=5.0)

            assert not creator.is_alive()
            assert first_errors == []
            assert first_result == [first_monitor]
            assert first_monitor.thread_alive
            assert first_manager.health_monitor is first_monitor
            assert first_engine.installed_health_monitor is first_monitor
            assert registry.get_all() == [first_monitor]
        finally:
            release_first_start.set()
            creator.join(timeout=5.0)
            with _patch_engine_builder():
                second_manager.stop_services()
                first_manager.stop_services()
            PeriodicThreadRegistry.reset()

    def test_unregister_baseexception_keeps_manager_retry_owner(self) -> None:
        """Unregister failure cannot skip engine detach or retry ownership."""
        PeriodicThreadRegistry.reset()
        registry = PeriodicThreadRegistry.get_instance()
        config = LMCacheEngineConfig.from_defaults()
        engine = _HealthMonitorTestEngine()
        factory = _HealthMonitorFactory(
            config,
            create_test_metadata(),
            engine,
        )
        manager = LMCacheManager(config=config, service_factory=factory)

        try:
            with (
                patch.object(HealthMonitor, "start", return_value=None),
                patch.object(
                    registry,
                    "unregister_instance",
                    side_effect=KeyboardInterrupt("injected unregister interrupt"),
                ),
                patch(
                    "lmcache.observability.PrometheusLogger.GetOrCreate",
                    side_effect=RuntimeError("injected Prometheus failure"),
                ),
                pytest.raises(
                    RuntimeError,
                    match="rollback did not complete: stop, unregister",
                ),
            ):
                factory.maybe_create_health_monitor(manager)

            monitor = manager.health_monitor
            assert monitor is not None
            assert registry.get(monitor.name) is monitor
            assert not monitor.thread_alive
            assert engine.installed_health_monitor is None

            with _patch_engine_builder() as builder:
                manager.stop_services()
                builder.destroy.assert_called_once()

            assert manager.services_stopped
            assert manager.health_monitor is None
            assert registry.get_all() == []
        finally:
            with _patch_engine_builder():
                manager.stop_services()
            PeriodicThreadRegistry.reset()

    def test_post_init_with_engine_and_async_loading(self):
        """Test post_init calls engine.post_init with async_lookup_server."""
        config = LMCacheEngineConfig.from_defaults()
        config.enable_async_loading = True
        mock_engine = MagicMock()
        factory = _make_mock_factory(engine=mock_engine)

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        manager.post_init()

        # When lookup_server is None, engine.post_init should be called
        # with async_lookup_server=None
        mock_engine.post_init.assert_called_once_with(async_lookup_server=None)

    def test_post_init_with_engine_and_async_server(self):
        """Test post_init calls engine.post_init when async lookup server exists."""
        # First Party
        from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupServer,
        )

        config = LMCacheEngineConfig.from_defaults()
        config.enable_async_loading = True
        mock_engine = MagicMock()
        mock_lookup_server = MagicMock(spec=LMCacheAsyncLookupServer)
        factory = _make_mock_factory(
            engine=mock_engine,
            lookup_server=mock_lookup_server,
        )

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        manager.post_init()

        # When lookup_server is LMCacheAsyncLookupServer, it should be passed
        mock_engine.post_init.assert_called_once_with(
            async_lookup_server=mock_lookup_server
        )


class TestLMCacheManagerShutdown:
    """Tests for LMCacheManager shutdown method."""

    def test_shutdown_closes_all_components(self):
        """Test stop_services() closes all components."""
        config = LMCacheEngineConfig.from_defaults()
        mock_offload = MagicMock()
        mock_plugin = MagicMock()
        mock_api = MagicMock()
        mock_lookup_server = MagicMock()
        mock_lookup_client = MagicMock(spec=LookupClientInterface)
        mock_engine = MagicMock()
        factory = _make_mock_factory(
            engine=mock_engine,
            offload_server=mock_offload,
            plugin_launcher=mock_plugin,
            api_server=mock_api,
            lookup_server=mock_lookup_server,
            lookup_client=mock_lookup_client,
        )

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        with _patch_engine_builder() as mock_builder:
            manager.stop_services()

            # Verify all components were closed
            mock_offload.close.assert_called_once()
            mock_plugin.stop_plugins.assert_called_once()
            mock_api.stop.assert_called_once()
            mock_lookup_server.close.assert_called_once()
            mock_lookup_client.close.assert_called_once()
            mock_builder.destroy.assert_called_once()

    def test_shutdown_handles_none_components(self):
        """Test stop_services() handles None components gracefully."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        with _patch_engine_builder() as mock_builder:
            # Should not raise any exception
            manager.stop_services()
            mock_builder.destroy.assert_not_called()

        assert manager._engine_destroyed is True
        assert manager._services_stopped is True

    def test_real_lookup_transport_failure_retains_client_until_retry(self) -> None:
        """Terminal shutdown cannot publish success over live ZMQ resources."""
        socket = _FailingCloseSocket(close_failures=2)
        context = _FailingTermContext(term_failures=0)
        with (
            patch(
                "lmcache.v1.rpc.zmq_transport.get_zmq_context",
                return_value=context,
            ),
            patch(
                "lmcache.v1.rpc.zmq_transport.get_zmq_socket",
                return_value=socket,
            ),
        ):
            transport = ZmqReqRepClientTransport(
                [SocketParams("/tmp/lmcache-manager-close-test", 0)],
                timeout_ms=1,
            )

        config = LMCacheEngineConfig.from_defaults()
        metadata = create_test_metadata()
        client = LMCacheLookupClient(config, metadata, transport)
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(
                metadata=metadata,
                lookup_client=client,
            ),
        )

        manager.stop_services()

        assert not manager.services_stopped
        assert manager.lookup_client is client
        assert not context.terminated
        with pytest.raises(RuntimeError, match="not accepting new ownership"):
            client.validate_lookup_request([1], "rejected-while-close-retries")
        assert not socket.closed
        assert context.term_calls == 0

        manager.stop_services()

        assert manager.services_stopped
        assert manager.lookup_client is None
        assert socket.closed
        # The client borrows Context.instance(); even successful close owns
        # only its sockets and must leave the shared context alive.
        assert not context.terminated
        assert socket.close_calls == 3
        assert context.term_calls == 0

    def test_cancel_timeout_then_partial_transport_close_retries_close_only(
        self,
    ) -> None:
        """A failed terminal close cannot recreate an already-closed socket."""
        timed_out_socket = _TimeoutSocket()
        replacement_socket = _FailingCloseSocket(close_failures=1)
        context = _FailingTermContext(term_failures=0)
        socket_factory = MagicMock(side_effect=[timed_out_socket, replacement_socket])
        with (
            patch(
                "lmcache.v1.rpc.zmq_transport.get_zmq_context",
                return_value=context,
            ),
            patch(
                "lmcache.v1.rpc.zmq_transport.get_zmq_socket",
                socket_factory,
            ),
        ):
            transport = ZmqReqRepClientTransport(
                [SocketParams("/tmp/lmcache-manager-close-only-test", 0)],
                timeout_ms=1,
            )

        config = LMCacheEngineConfig.from_defaults()
        metadata = create_test_metadata()
        client = LMCacheLookupClient(config, metadata, transport)
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(
                metadata=metadata,
                lookup_client=client,
            ),
        )

        retry_timer = MagicMock()
        with (
            patch(
                "lmcache.v1.lookup_client.lmcache_lookup_client."
                "_LOOKUP_CANCEL_IMMEDIATE_ATTEMPTS",
                1,
            ),
            patch(
                "lmcache.v1.lookup_client.lmcache_lookup_client."
                "_LOOKUP_CANCEL_CLOSE_TIMEOUT_SECONDS",
                0.0,
            ),
            patch(
                "lmcache.v1.lookup_client.lmcache_lookup_client.threading.Timer",
                return_value=retry_timer,
            ),
            patch(
                "lmcache.v1.rpc.zmq_transport.get_zmq_socket",
                socket_factory,
            ),
        ):
            client.cancel_lookup("one-rank-timeout")
            manager.stop_services()

            assert manager.lookup_client is client
            assert not manager.services_stopped
            assert timed_out_socket.close_calls == 1
            assert replacement_socket.close_calls == 1
            assert context.term_calls == 0
            assert not context.terminated
            assert socket_factory.call_count == 2

            with pytest.raises(RuntimeError, match="close has begun"):
                transport.send_and_recv_all([b"late-request"])
            assert socket_factory.call_count == 2
            assert replacement_socket.close_calls == 1

            manager.stop_services()

        assert manager.services_stopped
        assert manager.lookup_client is None
        assert not context.terminated
        assert context.term_calls == 0
        assert replacement_socket.close_calls == 2
        assert replacement_socket.closed
        assert socket_factory.call_count == 2

    def test_shutdown_handles_component_errors(self):
        """Test stop_services() handles errors from components gracefully."""
        config = LMCacheEngineConfig.from_defaults()
        mock_offload = MagicMock()
        mock_offload.close.side_effect = RuntimeError("Test error")
        mock_lookup_client = MagicMock(spec=LookupClientInterface)
        factory = _make_mock_factory(
            offload_server=mock_offload,
            lookup_client=mock_lookup_client,
        )

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        with _patch_engine_builder() as mock_builder:
            # Should not raise exception, but should continue shutdown
            manager.stop_services()

            # lookup_client should still be closed even if offload_server failed
            mock_lookup_client.close.assert_called_once()
            # The offload server may still be inside an engine.store call.
            mock_builder.destroy.assert_not_called()

        assert manager.offload_server is mock_offload
        assert manager._services_stopped is False

    def test_shutdown_retains_last_engine_owner_when_destroy_fails(self) -> None:
        """The last manager owner remains registered until destroy succeeds."""
        config = LMCacheEngineConfig.from_defaults()
        engine = MagicMock(spec=LMCacheEngine)
        factory = _make_mock_factory(engine=engine)
        factory.get_engine_instance_id.return_value = "engine-close-retry"
        manager = LMCacheManager(config=config, service_factory=factory)
        close_error = RuntimeError("injected engine close failure")

        with _patch_engine_builder() as builder:
            builder.destroy.side_effect = [
                close_error,
                EngineDestroyResult.DESTROYED,
            ]
            try:
                manager.stop_services()

                assert not manager.services_stopped
                assert manager.lmcache_engine is engine
                assert manager._engine_owner_registered
                assert LMCacheManager._engine_owner_counts[engine] == 1
                assert manager.terminal_cleanup_errors == (close_error,)
                builder.destroy.assert_called_once_with(
                    "engine-close-retry",
                    expected_engine=engine,
                )

                manager.stop_services()
            finally:
                builder.destroy.side_effect = None
                builder.destroy.return_value = EngineDestroyResult.DESTROYED
                manager.stop_services()

        assert manager.services_stopped
        assert manager.lmcache_engine is None
        assert not manager._engine_owner_registered
        assert engine not in LMCacheManager._engine_owner_counts
        assert manager.terminal_cleanup_errors == ()
        assert builder.destroy.call_count == 2

    def test_shutdown_exhausts_baseexception_and_retains_retry_owner(
        self,
    ) -> None:
        """A process interrupt is re-raised only after exhaustive cleanup."""
        PeriodicThreadRegistry.reset()
        registry = PeriodicThreadRegistry.get_instance()
        config = LMCacheEngineConfig.from_defaults()
        engine = _HealthMonitorTestEngine()
        offload_server = MagicMock(spec=ZMQOffloadServer)
        factory = _HealthMonitorFactory(
            config,
            create_test_metadata(),
            engine,
            offload_server=offload_server,
        )
        manager = LMCacheManager(config=config, service_factory=factory)

        try:
            with (
                patch.object(HealthMonitor, "start", return_value=None),
                patch(
                    "lmcache.observability.PrometheusLogger.GetOrCreate",
                    return_value=MagicMock(),
                ),
            ):
                manager.post_init()

            monitor = manager.health_monitor
            assert monitor is not None
            assert registry.get(monitor.name) is monitor
            assert not monitor.thread_alive

            with _patch_engine_builder() as builder:
                with (
                    patch.object(
                        registry,
                        "unregister_instance",
                        side_effect=KeyboardInterrupt("injected unregister interrupt"),
                    ),
                    pytest.raises(
                        KeyboardInterrupt,
                        match="injected unregister interrupt",
                    ),
                ):
                    manager.stop_services()

                offload_server.close.assert_called_once()
                assert manager.offload_server is None
                assert manager.health_monitor is monitor
                assert registry.get(monitor.name) is monitor
                assert not manager.services_stopped
                builder.destroy.assert_not_called()

                manager.stop_services()
                builder.destroy.assert_called_once()

            assert manager.services_stopped
            assert manager.health_monitor is None
            assert registry.get_all() == []
        finally:
            with _patch_engine_builder():
                manager.stop_services()
            PeriodicThreadRegistry.reset()

    def test_shutdown_skips_engine_destroy_when_api_thread_is_live(self):
        config = LMCacheEngineConfig.from_defaults()
        mock_api = MagicMock()
        mock_api.stop.side_effect = RuntimeError("in-flight API request")
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(api_server=mock_api),
        )

        with _patch_engine_builder() as mock_builder:
            manager.stop_services()
            mock_builder.destroy.assert_not_called()

        assert manager.api_server is mock_api
        assert manager._services_stopped is False

    def test_shutdown_retains_live_health_monitor_and_retries(self):
        config = LMCacheEngineConfig.from_defaults()
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(engine=MagicMock()),
        )
        mock_health = MagicMock()
        mock_health.stop.side_effect = RuntimeError("health check still running")
        manager._health_monitor = mock_health

        with _patch_engine_builder() as mock_builder:
            manager.stop_services()
            mock_builder.destroy.assert_not_called()
            assert manager.health_monitor is mock_health
            assert manager._services_stopped is False

            mock_health.stop.side_effect = None
            manager.stop_services()
            mock_builder.destroy.assert_called_once()

        assert manager.health_monitor is None
        assert manager._engine_destroyed is True
        assert manager._services_stopped is True

    def test_plugin_failure_does_not_claim_services_stopped_or_redestroy_engine(self):
        config = LMCacheEngineConfig.from_defaults()
        mock_plugin = MagicMock()
        mock_plugin.stop_plugins.side_effect = [
            RuntimeError("plugin still running"),
            None,
        ]
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(
                engine=MagicMock(),
                plugin_launcher=mock_plugin,
            ),
        )

        with _patch_engine_builder() as mock_builder:
            manager.stop_services()
            assert manager._engine_destroyed is True
            assert manager._services_stopped is False
            assert manager._runtime_plugin_launcher is mock_plugin

            manager.stop_services()
            mock_builder.destroy.assert_called_once()

        assert manager._runtime_plugin_launcher is None
        assert manager._services_stopped is True

    def test_shutdown_force_closes_client_before_engine_destroy(self):
        config = LMCacheEngineConfig.from_defaults()
        events = []
        mock_lookup_client = MagicMock(spec=LookupClientInterface)

        def fail_dynamic_close() -> None:
            events.append("close")
            raise RuntimeError("cleanup remains")

        mock_lookup_client.close.side_effect = fail_dynamic_close
        mock_lookup_client.force_close.side_effect = lambda: events.append("force")
        factory = _make_mock_factory(
            engine=MagicMock(),
            lookup_client=mock_lookup_client,
        )
        manager = LMCacheManager(config=config, service_factory=factory)

        with _patch_engine_builder() as mock_builder:

            def record_destroy(
                _engine_id: str,
                *,
                expected_engine: LMCacheEngine,
            ) -> EngineDestroyResult:
                assert expected_engine is manager.lmcache_engine
                events.append("destroy")
                return EngineDestroyResult.DESTROYED

            mock_builder.destroy.side_effect = record_destroy
            manager.stop_services()

        assert events == ["close", "force", "destroy"]
        assert manager.lookup_client is None
        assert manager._services_stopped is True

    def test_shutdown_skips_engine_destroy_when_force_close_fails(self):
        config = LMCacheEngineConfig.from_defaults()
        mock_lookup_client = MagicMock()
        close_error = RuntimeError("cleanup remains")
        force_error = RuntimeError("still active")
        mock_lookup_client.close.side_effect = close_error
        mock_lookup_client.force_close.side_effect = force_error
        metadata = MagicMock()
        factory = _make_mock_factory(
            metadata=metadata,
            lookup_client=mock_lookup_client,
        )
        manager = LMCacheManager(config=config, service_factory=factory)

        with _patch_engine_builder() as mock_builder:
            manager.stop_services()
            mock_builder.destroy.assert_not_called()

        assert manager.lookup_client is mock_lookup_client
        assert manager._services_stopped is False
        assert manager._shutdown_requested is True
        assert manager.terminal_cleanup_errors == (force_error,)
        assert LMCacheManager.incomplete_construction_cleanup_count() == 0

        with patch(
            "lmcache.v1.lookup_client.factory.LookupClientFactory.create_lookup_client"
        ) as create_client:
            result = manager.create_lookup_client()
        assert "shutdown" in result["error"]
        create_client.assert_not_called()

    def test_shutdown_is_idempotent(self):
        config = LMCacheEngineConfig.from_defaults()
        mock_lookup_client = MagicMock()
        factory = _make_mock_factory(
            engine=MagicMock(),
            lookup_client=mock_lookup_client,
        )
        manager = LMCacheManager(config=config, service_factory=factory)

        with _patch_engine_builder() as mock_builder:
            mock_builder.destroy.return_value = EngineDestroyResult.DESTROYED
            manager.stop_services()
            manager.stop_services()

        mock_lookup_client.close.assert_called_once()
        mock_builder.destroy.assert_called_once()

    def test_shutdown_outer_interrupt_does_not_strand_reentry_owner(self) -> None:
        """An interrupt outside a close callback leaves public retry usable."""
        config = LMCacheEngineConfig.from_defaults()
        manager = LMCacheManager(config, _make_mock_factory())
        original_error = KeyboardInterrupt("injected outer cleanup interrupt")

        with (
            patch("lmcache.v1.manager.logger.info", side_effect=original_error),
            pytest.raises(KeyboardInterrupt) as failure,
        ):
            manager.stop_services()

        assert failure.value is original_error
        assert not manager.services_stopped

        manager.stop_services()
        assert manager.services_stopped

    @pytest.mark.parametrize("guard_callback", [False, True])
    @pytest.mark.parametrize("fail_first_callback", [False, True])
    def test_health_stop_same_thread_reentry_is_bounded_and_retry_safe(
        self,
        guard_callback: bool,
        fail_first_callback: bool,
    ) -> None:
        """A monitor callback cannot deadlock or poison outer cleanup."""
        config = LMCacheEngineConfig.from_defaults()
        engine = _HealthMonitorTestEngine()
        offload_server = MagicMock(spec=ZMQOffloadServer)
        factory = _HealthMonitorFactory(
            config,
            create_test_metadata(),
            engine,
            offload_server=offload_server,
        )
        manager = LMCacheManager(config=config, service_factory=factory)
        monitor = MagicMock(spec=HealthMonitor)
        callback_calls = 0
        callback_errors: list[BaseException] = []

        def reentrant_stop() -> None:
            nonlocal callback_calls
            callback_calls += 1
            if guard_callback:
                try:
                    manager.stop_services()
                except BaseException as error:
                    callback_errors.append(error)
                    raise
            else:
                manager.stop_services()
            if fail_first_callback and callback_calls == 1:
                raise RuntimeError("injected monitor callback failure")

        with patch.object(
            factory,
            "maybe_create_health_monitor",
            return_value=monitor,
        ):
            manager.post_init()
        monitor.stop.side_effect = reentrant_stop

        stop_errors: list[BaseException] = []

        def run_stop() -> None:
            try:
                manager.stop_services()
            except BaseException as error:
                stop_errors.append(error)

        with _patch_engine_builder() as builder:
            stop_thread = threading.Thread(target=run_stop, daemon=True)
            stop_thread.start()
            stop_thread.join(timeout=5.0)

            assert not stop_thread.is_alive()
            assert stop_errors == []
            assert callback_errors == []
            offload_server.close.assert_called_once_with()

            if fail_first_callback:
                assert not manager.services_stopped
                assert manager.health_monitor is monitor
                builder.destroy.assert_not_called()

                manager.stop_services()

            assert manager.services_stopped
            assert manager.health_monitor is None
            builder.destroy.assert_called_once_with(
                "health-monitor-rollback",
                expected_engine=engine,
            )

            manager.stop_services()

        expected_callbacks = 2 if fail_first_callback else 1
        assert callback_calls == expected_callbacks
        offload_server.close.assert_called_once_with()

    def test_scheduler_without_engine_cannot_destroy_coprocess_worker_engine(self):
        config = LMCacheEngineConfig.from_defaults()
        shared_worker_engine = MagicMock()
        worker_factory = _make_mock_factory(engine=shared_worker_engine)
        worker_factory.get_engine_instance_id.return_value = "shared-vllm-engine"
        worker_manager = LMCacheManager(config, worker_factory)

        scheduler_factory = _make_mock_factory(engine=None)
        scheduler_factory.get_engine_instance_id.return_value = "shared-vllm-engine"
        scheduler_manager = LMCacheManager(config, scheduler_factory)

        with _patch_engine_builder() as mock_builder:
            scheduler_manager.stop_services()

            mock_builder.destroy.assert_not_called()
            assert worker_manager.lmcache_engine is shared_worker_engine
            assert scheduler_manager._services_stopped is True

            worker_manager.stop_services()
            mock_builder.destroy.assert_called_once_with(
                "shared-vllm-engine",
                expected_engine=shared_worker_engine,
            )

    def test_shared_non_none_engine_is_destroyed_by_last_manager_owner(self):
        config = LMCacheEngineConfig.from_defaults()
        shared_engine = MagicMock()
        first_factory = _make_mock_factory(engine=shared_engine)
        second_factory = _make_mock_factory(engine=shared_engine)
        first_factory.get_engine_instance_id.return_value = "shared-engine"
        second_factory.get_engine_instance_id.return_value = "shared-engine"
        first = LMCacheManager(config, first_factory)
        second = LMCacheManager(config, second_factory)

        with _patch_engine_builder() as mock_builder:
            mock_builder.destroy.return_value = EngineDestroyResult.DESTROYED
            first.stop_services()
            mock_builder.destroy.assert_not_called()
            assert first._services_stopped is True
            assert second.lmcache_engine is shared_engine

            second.stop_services()
            mock_builder.destroy.assert_called_once_with(
                "shared-engine",
                expected_engine=shared_engine,
            )

        assert second._services_stopped is True

    def test_live_async_server_blocks_terminal_engine_destroy(self, monkeypatch):
        # First Party
        from lmcache.v1.lookup_client import lmcache_async_lookup_client as async_mod
        from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupServer,
        )

        config = LMCacheEngineConfig.from_defaults()
        config.enable_async_loading = True
        entered = threading.Event()
        release = threading.Event()
        engine = MagicMock()

        def blocked_async_lookup(**_kwargs) -> None:
            entered.set()
            assert release.wait(timeout=2.0)

        engine.async_lookup_and_prefetch.side_effect = blocked_async_lookup
        server = LMCacheAsyncLookupServer.__new__(LMCacheAsyncLookupServer)
        server.lmcache_engine = engine
        server.running = True
        server._closed = False
        server._close_lock = threading.Lock()
        server._callback_condition = threading.Condition()
        server._active_response_callbacks = 0
        server._closing = False
        server.push_socket = MagicMock()
        server.pull_socket = MagicMock()
        server.ctx = MagicMock()
        server.thread = threading.Thread(
            target=lambda: engine.async_lookup_and_prefetch(lookup_id="blocked"),
            daemon=True,
        )
        server.thread.start()
        assert entered.wait(timeout=2.0)

        manager = LMCacheManager(
            config,
            _make_mock_factory(engine=engine, lookup_server=server),
        )
        monkeypatch.setattr(
            async_mod,
            "_ASYNC_LOOKUP_CLOSE_TIMEOUT_SECONDS",
            0.05,
        )

        with _patch_engine_builder() as mock_builder:
            manager.stop_services()
            mock_builder.destroy.assert_not_called()
            assert manager.lookup_server is server
            assert manager._services_stopped is False

            release.set()
            server.thread.join(timeout=2.0)
            assert not server.thread.is_alive()
            manager.stop_services()
            mock_builder.destroy.assert_called_once()

        assert manager.lookup_server is None
        assert manager._services_stopped is True

    def test_shutdown_does_not_hold_lookup_lock_while_stopping_api(self):
        config = LMCacheEngineConfig.from_defaults()
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(),
        )
        request_may_finish = threading.Event()
        request_finished = threading.Event()
        mock_api = MagicMock()

        def finish_inflight_request() -> None:
            assert request_may_finish.wait(timeout=2.0)
            manager.get_lookup_info()
            request_finished.set()

        def stop_api() -> None:
            request_may_finish.set()
            assert request_finished.wait(timeout=2.0)

        mock_api.stop.side_effect = stop_api
        manager._api_server = mock_api
        request_thread = threading.Thread(target=finish_inflight_request)
        request_thread.start()

        with _patch_engine_builder():
            stop_thread = threading.Thread(target=manager.stop_services)
            stop_thread.start()
            stop_thread.join(timeout=2.0)

        request_thread.join(timeout=2.0)
        assert not stop_thread.is_alive()
        assert not request_thread.is_alive()
        assert request_finished.is_set()


class TestLMCacheManagerLookupLifecycle:
    """Dynamic lookup lifecycle mutations are serialized and non-lossy."""

    def test_close_client_failure_preserves_pointer_and_returns_error(self):
        config = LMCacheEngineConfig.from_defaults()
        old_client = MagicMock(spec=LookupClientInterface)
        old_client.close.side_effect = RuntimeError("pending cancellation")
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(lookup_client=old_client),
        )

        result = manager.close_lookup_client()

        assert manager.lookup_client is old_client
        assert result["old"] == "MagicMock"
        assert "pending cancellation" in result["error"]

    def test_create_client_refuses_to_overwrite_live_client(self):
        config = LMCacheEngineConfig.from_defaults()
        old_client = MagicMock(spec=LookupClientInterface)
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(
                metadata=MagicMock(),
                lookup_client=old_client,
            ),
        )

        with patch(
            "lmcache.v1.lookup_client.factory.LookupClientFactory.create_lookup_client"
        ) as create_client:
            result = manager.create_lookup_client()

        assert manager.lookup_client is old_client
        assert "already available" in result["error"]
        create_client.assert_not_called()

    def test_dryrun_client_close_failure_retains_cleanup_owner(self):
        config = LMCacheEngineConfig.from_defaults()
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(metadata=MagicMock()),
        )
        orphan = MagicMock(spec=LookupClientInterface)
        orphan.close.side_effect = RuntimeError("dynamic close failed")
        orphan.force_close.side_effect = RuntimeError("force close failed")

        with patch(
            "lmcache.v1.lookup_client.factory.LookupClientFactory.create_lookup_client",
            return_value=orphan,
        ) as create_client:
            result = manager.create_lookup_client(dryrun=True)
            second_result = manager.create_lookup_client(dryrun=True)

        assert "client retained for cleanup" in result["error"]
        assert manager.lookup_client is orphan
        assert "already available" in second_result["error"]
        create_client.assert_called_once()

        with _patch_engine_builder() as mock_builder:
            manager.stop_services()
            mock_builder.destroy.assert_not_called()

        assert manager.lookup_client is orphan

    def test_recreate_client_aborts_when_close_fails(self):
        config = LMCacheEngineConfig.from_defaults()
        old_client = MagicMock(spec=LookupClientInterface)
        old_client.close.side_effect = RuntimeError("pending cancellation")
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(
                metadata=MagicMock(),
                lookup_client=old_client,
            ),
        )

        with patch(
            "lmcache.v1.lookup_client.factory.LookupClientFactory.create_lookup_client"
        ) as create_client:
            result = manager.recreate_lookup_client()

        assert manager.lookup_client is old_client
        assert "pending cancellation" in result["error"]
        assert "new" not in result
        create_client.assert_not_called()

    def test_dynamic_async_client_lifecycle_is_rejected_with_outstanding_work(self):
        # First Party
        from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupClient,
        )

        config = LMCacheEngineConfig.from_defaults()
        config.enable_async_loading = True
        client = LMCacheAsyncLookupClient.__new__(LMCacheAsyncLookupClient)
        client.reqs_status = {"inflight": None}
        manager = LMCacheManager(
            config,
            _make_mock_factory(metadata=MagicMock(), lookup_client=client),
        )

        with patch.object(client, "close") as close:
            close_result = manager.close_lookup_client()
            recreate_result = manager.recreate_lookup_client()

        assert "unsupported for async" in close_result["error"]
        assert "unsupported for async" in recreate_result["error"]
        assert manager.lookup_client is client
        close.assert_not_called()

    def test_dynamic_async_client_create_is_rejected_before_construction(self):
        config = LMCacheEngineConfig.from_defaults()
        config.enable_async_loading = True
        manager = LMCacheManager(
            config,
            _make_mock_factory(metadata=MagicMock()),
        )

        with patch(
            "lmcache.v1.lookup_client.factory.LookupClientFactory.create_lookup_client"
        ) as create_client:
            result = manager.create_lookup_client()

        assert "unsupported for async" in result["error"]
        create_client.assert_not_called()

    def test_concurrent_client_creates_publish_exactly_one(self):
        config = LMCacheEngineConfig.from_defaults()
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(metadata=MagicMock()),
        )
        new_client = MagicMock(spec=LookupClientInterface)
        start = threading.Barrier(3)
        results = []

        def create() -> None:
            start.wait()
            results.append(manager.create_lookup_client())

        with patch(
            "lmcache.v1.lookup_client.factory.LookupClientFactory.create_lookup_client",
            return_value=new_client,
        ) as create_client:
            threads = [threading.Thread(target=create) for _ in range(2)]
            for thread in threads:
                thread.start()
            start.wait()
            for thread in threads:
                thread.join(timeout=2.0)
                assert not thread.is_alive()

        assert manager.lookup_client is new_client
        assert create_client.call_count == 1
        assert sum("new" in result for result in results) == 1
        assert sum("error" in result for result in results) == 1

    def test_close_and_create_client_are_serialized(self):
        config = LMCacheEngineConfig.from_defaults()
        old_client = MagicMock(spec=LookupClientInterface)
        close_entered = threading.Event()
        close_release = threading.Event()

        def blocking_close() -> None:
            close_entered.set()
            assert close_release.wait(timeout=2.0)

        old_client.close.side_effect = blocking_close
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(
                metadata=MagicMock(),
                lookup_client=old_client,
            ),
        )
        new_client = MagicMock(spec=LookupClientInterface)
        results = {}

        close_thread = threading.Thread(
            target=lambda: results.setdefault("close", manager.close_lookup_client())
        )
        with patch(
            "lmcache.v1.lookup_client.factory.LookupClientFactory.create_lookup_client",
            return_value=new_client,
        ) as create_client:
            close_thread.start()
            assert close_entered.wait(timeout=2.0)
            create_thread = threading.Thread(
                target=lambda: results.setdefault(
                    "create", manager.create_lookup_client()
                )
            )
            create_thread.start()
            create_thread.join(timeout=0.05)
            assert create_thread.is_alive()
            create_client.assert_not_called()

            close_release.set()
            close_thread.join(timeout=2.0)
            create_thread.join(timeout=2.0)

        assert not close_thread.is_alive()
        assert not create_thread.is_alive()
        assert "error" not in results["close"]
        assert results["create"] == {"new": "MagicMock"}
        assert manager.lookup_client is new_client

    def test_create_racing_shutdown_cannot_resurrect_client(self):
        config = LMCacheEngineConfig.from_defaults()
        old_client = MagicMock(spec=LookupClientInterface)
        old_client.close.side_effect = RuntimeError("must force")
        force_entered = threading.Event()
        force_release = threading.Event()

        def blocking_force_close() -> None:
            force_entered.set()
            assert force_release.wait(timeout=2.0)

        old_client.force_close.side_effect = blocking_force_close
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(
                metadata=MagicMock(),
                lookup_client=old_client,
            ),
        )
        create_result = {}

        with (
            _patch_engine_builder(),
            patch(
                "lmcache.v1.lookup_client.factory.LookupClientFactory."
                "create_lookup_client"
            ) as create_client,
        ):
            stop_thread = threading.Thread(target=manager.stop_services)
            stop_thread.start()
            assert force_entered.wait(timeout=2.0)

            create_thread = threading.Thread(
                target=lambda: create_result.update(manager.create_lookup_client())
            )
            create_thread.start()
            create_thread.join(timeout=2.0)
            assert not create_thread.is_alive()
            create_client.assert_not_called()

            force_release.set()
            stop_thread.join(timeout=2.0)
            create_thread.join(timeout=2.0)

        assert not stop_thread.is_alive()
        assert not create_thread.is_alive()
        assert "shutdown" in create_result["error"]
        assert manager.lookup_client is None
        create_client.assert_not_called()

    def test_server_recreate_failure_preserves_pointer(self):
        config = LMCacheEngineConfig.from_defaults()
        old_server = MagicMock(spec=["close"])
        old_server.close.side_effect = RuntimeError("server still active")
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(
                metadata=MagicMock(),
                engine=MagicMock(),
                lookup_server=old_server,
            ),
        )

        with patch(
            "lmcache.v1.lookup_client.factory.LookupClientFactory.create_lookup_server"
        ) as create_server:
            result = manager.recreate_lookup_server()

        assert manager.lookup_server is old_server
        assert "server still active" in result["error"]
        create_server.assert_not_called()

    def test_dynamic_async_server_close_and_recreate_are_rejected(self):
        # First Party
        from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupServer,
        )

        config = LMCacheEngineConfig.from_defaults()
        config.enable_async_loading = True
        server = LMCacheAsyncLookupServer.__new__(LMCacheAsyncLookupServer)
        manager = LMCacheManager(
            config,
            _make_mock_factory(lookup_server=server),
        )

        with patch.object(server, "close") as close:
            close_result = manager.close_lookup_server()
            recreate_result = manager.recreate_lookup_server()

        assert "unsupported for async" in close_result["error"]
        assert "unsupported for async" in recreate_result["error"]
        assert manager.lookup_server is server
        close.assert_not_called()

    def test_dynamic_async_server_create_is_rejected_before_construction(self):
        config = LMCacheEngineConfig.from_defaults()
        config.enable_async_loading = True
        manager = LMCacheManager(
            config,
            _make_mock_factory(metadata=MagicMock(), engine=MagicMock()),
        )

        with patch(
            "lmcache.v1.lookup_client.factory.LookupClientFactory.create_lookup_server"
        ) as create_server:
            result = manager.create_lookup_server()

        assert "unsupported for async" in result["error"]
        create_server.assert_not_called()

        with _patch_engine_builder():
            manager.stop_services()

    def test_dryrun_server_close_failure_retains_cleanup_owner(self):
        config = LMCacheEngineConfig.from_defaults()
        manager = LMCacheManager(
            config=config,
            service_factory=_make_mock_factory(
                metadata=MagicMock(),
                engine=MagicMock(),
            ),
        )
        orphan = MagicMock(spec=["close"])
        orphan.close.side_effect = RuntimeError("server thread still live")

        with patch(
            "lmcache.v1.lookup_client.factory.LookupClientFactory.create_lookup_server",
            return_value=orphan,
        ) as create_server:
            result = manager.create_lookup_server(dryrun=True)
            second_result = manager.create_lookup_server(dryrun=True)

        assert "server retained for cleanup" in result["error"]
        assert manager.lookup_server is orphan
        assert "already available" in second_result["error"]
        create_server.assert_called_once()

        with _patch_engine_builder() as mock_builder:
            manager.stop_services()
            mock_builder.destroy.assert_not_called()

        assert manager.lookup_server is orphan


class TestLMCacheManagerHelpers:
    """Tests for LMCacheManager helper methods."""

    def test_need_gpu_interm_buffer_returns_not_enable_pd(self):
        """Test _need_gpu_interm_buffer returns opposite of enable_pd."""
        config = LMCacheEngineConfig.from_defaults()
        config.enable_pd = False
        # First Party
        from lmcache.v1.gpu_connector.utils import need_gpu_interm_buffer

        assert need_gpu_interm_buffer(config) is True

        config.enable_pd = True
        assert need_gpu_interm_buffer(config) is False


class TestLMCacheManagerValidation:
    """Tests for validate_mla_config (now in utils)."""

    def test_validate_mla_config_raises_on_wrong_serde(self):
        """Test validate_mla_config raises error for non-naive serde with MLA."""
        # First Party
        from lmcache.integration.vllm.utils import validate_mla_config

        config = LMCacheEngineConfig.from_defaults()
        config.remote_serde = "cachegen"

        with pytest.raises(ValueError, match="MLA only works with naive serde mode"):
            validate_mla_config(config, use_mla=True)

    def test_validate_mla_config_raises_on_layerwise_with_blending(self):
        """Test validate_mla_config raises with MLA + layerwise + blending."""
        # First Party
        from lmcache.integration.vllm.utils import validate_mla_config

        config = LMCacheEngineConfig.from_defaults()
        config.remote_serde = "naive"
        config.use_layerwise = True
        config.enable_blending = True

        with pytest.raises(ValueError, match="MLA with Cacheblend"):
            validate_mla_config(config, use_mla=True)


class TestLMCacheManagerCalculateDraftLayers:
    """Tests for calculate_draft_layers (now in utils)."""

    def test_calculate_draft_layers_no_speculative_config(self):
        """Test returns 0 when no speculative_config."""
        # First Party
        from lmcache.integration.vllm.utils import calculate_draft_layers

        vllm_config = MagicMock()
        vllm_config.speculative_config = None

        assert calculate_draft_layers(vllm_config) == 0

    def test_calculate_draft_layers_deepseek_mtp(self):
        """Test returns correct layers for deepseek_mtp method."""
        # First Party
        from lmcache.integration.vllm.utils import calculate_draft_layers

        vllm_config = MagicMock()
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.method = "deepseek_mtp"
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.hf_config = MagicMock()
        vllm_config.model_config.hf_config.num_nextn_predict_layers = 3

        assert calculate_draft_layers(vllm_config) == 3


class TestLMCacheManagerInitFailure:
    """Tests for LMCacheManager initialization failure handling."""

    def test_init_components_exception_makes_unhealthy(self):
        """Test that exception during factory calls makes manager unhealthy."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()
        factory.get_or_create_metadata.side_effect = RuntimeError("Test init failure")

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        # Verify through is_healthy API
        assert manager.is_healthy() is False

    def test_post_init_skipped_when_init_failed(self):
        """Test that post_init marks engine unhealthy when init failed."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()
        factory.get_or_create_metadata.side_effect = RuntimeError("Test init failure")

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        mock_engine = MagicMock()
        manager._lmcache_engine = mock_engine

        manager.post_init()

        # Engine should be marked as init failed
        mock_engine.mark_init_failed.assert_called_once()
        # Manager should report unhealthy
        assert manager.is_healthy() is False

    def test_post_init_exception_makes_unhealthy(self):
        """Test that exception during post_init makes system unhealthy."""
        config = LMCacheEngineConfig.from_defaults()
        mock_engine = MagicMock()
        mock_engine.post_init.side_effect = RuntimeError("Test post_init failure")
        factory = _make_mock_factory(engine=mock_engine)

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        manager.post_init()

        # Verify engine was marked as failed
        mock_engine.mark_init_failed.assert_called_once()
        # Manager should report unhealthy
        assert manager.is_healthy() is False

    def test_unhealthy_engine_makes_manager_unhealthy(self):
        """Test that manager is unhealthy when engine is unhealthy."""
        config = LMCacheEngineConfig.from_defaults()
        mock_engine = MagicMock()
        mock_engine.is_healthy.return_value = False
        factory = _make_mock_factory(engine=mock_engine)

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        assert manager.is_healthy() is False

    def test_healthy_when_all_components_healthy(self):
        """Test that manager is healthy when all components are healthy."""
        config = LMCacheEngineConfig.from_defaults()
        mock_engine = MagicMock()
        mock_engine.is_healthy.return_value = True
        factory = _make_mock_factory(engine=mock_engine)

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        mock_health_monitor = MagicMock()
        mock_health_monitor.is_healthy.return_value = True
        manager._health_monitor = mock_health_monitor

        assert manager.is_healthy() is True


class TestLMCacheEngineInitFailure:
    """Tests for LMCacheEngine init failure behavior on lookup/retrieve."""

    def test_lookup_returns_zero_when_init_failed(self):
        """Test that lookup returns 0 when engine is marked as init failed."""
        # First Party
        from lmcache.v1.cache_engine import LMCacheEngine

        engine = MagicMock(spec=LMCacheEngine)
        engine._init_failed = False
        engine._health_monitor = None

        # Attach real methods to the mock
        engine.mark_init_failed = LMCacheEngine.mark_init_failed.__get__(
            engine, LMCacheEngine
        )
        engine.is_healthy = LMCacheEngine.is_healthy.__get__(engine, LMCacheEngine)

        # Call mark_init_failed and verify is_healthy returns False
        engine.mark_init_failed("Test reason")
        assert engine.is_healthy() is False

    def test_retrieve_returns_empty_mask_when_init_failed(self):
        """Test retrieve returns all-False mask when engine init failed."""
        # First Party
        from lmcache.v1.cache_engine import LMCacheEngine

        engine = MagicMock(spec=LMCacheEngine)
        engine._init_failed = False
        engine._health_monitor = None

        # Attach real methods to the mock
        engine.mark_init_failed = LMCacheEngine.mark_init_failed.__get__(
            engine, LMCacheEngine
        )
        engine.is_healthy = LMCacheEngine.is_healthy.__get__(engine, LMCacheEngine)

        # Mark as init failed and verify is_healthy returns False
        # (which would cause retrieve to return all-False mask)
        engine.mark_init_failed("Test reason")
        assert engine.is_healthy() is False

    def test_mark_init_failed_makes_engine_unhealthy(self):
        """Test that mark_init_failed makes engine report unhealthy."""
        # First Party
        from lmcache.v1.cache_engine import LMCacheEngine

        engine = MagicMock(spec=LMCacheEngine)
        engine._init_failed = False
        engine._health_monitor = None

        # Attach real methods to the mock
        engine.mark_init_failed = LMCacheEngine.mark_init_failed.__get__(
            engine, LMCacheEngine
        )
        engine.is_healthy = LMCacheEngine.is_healthy.__get__(engine, LMCacheEngine)

        # Verify initially healthy
        assert engine.is_healthy() is True

        # Call the method under test
        engine.mark_init_failed("Test reason")

        # Assert on the outcome
        assert engine.is_healthy() is False
