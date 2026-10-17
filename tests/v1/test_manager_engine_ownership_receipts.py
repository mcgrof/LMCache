# SPDX-License-Identifier: Apache-2.0
"""Interruption tests for exact per-manager engine ownership receipts."""

# Standard
from types import SimpleNamespace
from typing import Any, Callable, Optional, cast
from unittest.mock import MagicMock, call, patch
import inspect
import sys
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.base_service_factory import (
    BaseServiceFactory,
    ServiceResultPublisher,
)
from lmcache.integration.vllm.connector_construction import (
    RetainedConnectorConstructionOwner,
)
from lmcache.observability import LMCacheStatsLogger
from lmcache.v1.cache_engine import (
    EngineDestroyResult,
    LMCacheEngine,
    LMCacheEngineBuilder,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.construction import retain_lookup_construction_cleanup
from lmcache.v1.manager import (
    LMCacheManager,
    _EngineAcquisitionPhase,
    _RetainedFactoryCallOwner,
)
from tests.v1.instruction_monitoring import (
    InstructionInterruption,
    instruction_offset,
    requires_instruction_monitoring,
)
import lmcache.v1.cache_engine as cache_engine_module


def _factory(engine: Optional[LMCacheEngine], instance_id: str) -> MagicMock:
    factory = MagicMock(spec=BaseServiceFactory)
    factory.get_engine_instance_id.return_value = instance_id
    factory.get_or_create_metadata.return_value = None
    factory.get_or_create_lmcache_engine.return_value = engine
    factory.maybe_create_lookup_client.return_value = None
    factory.maybe_create_lookup_server.return_value = None
    factory.maybe_create_offload_server.return_value = None
    factory.maybe_create_runtime_plugin_launcher.return_value = None
    factory.maybe_create_internal_api_server.return_value = None

    def publish_return_value(
        method: MagicMock,
    ) -> Callable[[Callable[[Any], None]], Any]:
        def invoke(publisher: Callable[[Any], None]) -> Any:
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


def _manager_without_services(instance_id: str) -> LMCacheManager:
    """Build a real Manager whose lifecycle has no external resources."""
    return LMCacheManager(
        LMCacheEngineConfig.from_defaults(),
        _factory(None, instance_id),
    )


def test_public_retry_filters_healthy_handoff_from_failed_exact_token() -> None:
    """A failed generation cannot drain the same Manager's healthy handoff."""
    manager = _manager_without_services("handoff-mixed-generations")
    healthy_receipt = manager.new_construction_handoff_receipt()
    failed_receipt = manager.new_construction_handoff_receipt()
    late_receipt = manager.new_construction_handoff_receipt()

    try:
        manager.retain_construction_handoff(healthy_receipt)
        manager.retain_construction_handoff(failed_receipt)
        manager.promote_construction_handoff_failure(failed_receipt)

        # Promotion publishes the shutdown fence in the same lifecycle->class
        # lock order as healthy retain. No post-snapshot generation can enter.
        with pytest.raises(RuntimeError, match="terminal LMCache Manager"):
            manager.retain_construction_handoff(late_receipt)

        assert LMCacheManager.provisional_construction_handoff_count() == 1
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1

        # A stale successful-path release cannot erase failed ownership.
        manager.release_construction_handoff(failed_receipt)
        with patch.object(
            manager,
            "stop_services",
            wraps=manager.stop_services,
        ) as manager_stop:
            assert LMCacheManager.retry_incomplete_construction_cleanup() == 1
        manager_stop.assert_not_called()
        assert not manager.services_stopped
        assert LMCacheManager.provisional_construction_handoff_count() == 1

        manager.release_construction_handoff(healthy_receipt)
        assert LMCacheManager.provisional_construction_handoff_count() == 0
        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        assert manager.services_stopped
    finally:
        manager.release_construction_handoff(healthy_receipt)
        manager.stop_services()


def test_stale_exact_release_preserves_new_handoff_and_failure_owner() -> None:
    """An old receipt cannot erase an independent failure or newer token."""
    manager = _manager_without_services("handoff-stale-release")
    old_receipt = manager.new_construction_handoff_receipt()
    new_receipt = manager.new_construction_handoff_receipt()

    try:
        manager.retain_construction_handoff(old_receipt)
        manager.retain_construction_handoff(new_receipt)
        manager.retain_construction_cleanup()

        manager.release_construction_handoff(old_receipt)
        # Idempotent stale replay is ABA-safe: it cannot match the newer
        # identity token or the independent failure-only Manager owner.
        manager.release_construction_handoff(old_receipt)
        assert LMCacheManager.provisional_construction_handoff_count() == 1
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1

        manager.release_construction_handoff(new_receipt)
        assert LMCacheManager.provisional_construction_handoff_count() == 0
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1
        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        assert manager.services_stopped
    finally:
        if old_receipt.phase != "PREPARED":
            manager.release_construction_handoff(old_receipt)
        if new_receipt.phase != "PREPARED":
            manager.release_construction_handoff(new_receipt)
        manager.stop_services()


def test_adopted_handoff_promotes_in_place_after_successful_release() -> None:
    """The final constructor-return gap keeps its preallocated exact slot."""
    manager = _manager_without_services("handoff-adopted-promotion")
    receipt = manager.new_construction_handoff_receipt()

    try:
        manager.retain_construction_handoff(receipt)
        manager.release_construction_handoff(receipt)
        assert receipt.phase == "RELEASED"
        assert LMCacheManager.provisional_construction_handoff_count() == 0
        with LMCacheManager._incomplete_construction_lock:
            assert LMCacheManager._construction_handoff_owners == {receipt: manager}

        manager.promote_construction_handoff_failure(receipt)
        assert receipt.phase == "FAILED_CLEANUP"
        with LMCacheManager._incomplete_construction_lock:
            assert LMCacheManager._construction_handoff_owners == {receipt: manager}
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1

        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        assert manager.services_stopped
        with LMCacheManager._incomplete_construction_lock:
            assert receipt not in LMCacheManager._construction_handoff_owners
    finally:
        manager.stop_services()


@pytest.mark.parametrize("interrupt_after_start", [False, True])
def test_wrapper_promotion_admission_interrupt_keeps_exact_failed_token(
    monkeypatch: pytest.MonkeyPatch,
    interrupt_after_start: bool,
) -> None:
    """Promotion survives interruption before start or after phase publish."""
    manager = _manager_without_services(
        "handoff-promotion-after-start"
        if interrupt_after_start
        else "handoff-promotion-before-start"
    )
    connector = SimpleNamespace(_manager=manager)
    thread_name = "test-wrapper-promotion-admission"

    def build_connector(publish: Callable[[Any], None]) -> Any:
        publish(connector)
        return connector

    owner = RetainedConnectorConstructionOwner(
        build_connector,
        thread_name=thread_name,
    )
    owner.invoke()
    receipt = owner._handoff_receipt
    assert receipt is not None
    assert receipt.phase == "PROVISIONAL"

    original_thread_start = threading.Thread.start
    phase_published = threading.Event()
    promotion_calls = 0
    target_start_calls = 0
    injected = KeyboardInterrupt("promotion admission interrupted")
    original_promote = manager.promote_construction_handoff_failure

    def record_promotion(exact_receipt: object) -> None:
        nonlocal promotion_calls
        promotion_calls += 1
        original_promote(cast(Any, exact_receipt))
        phase_published.set()

    def interrupt_target_start(thread: threading.Thread) -> None:
        nonlocal target_start_calls
        if thread.name != thread_name + "-failure-promotion":
            original_thread_start(thread)
            return
        target_start_calls += 1
        if target_start_calls == 1:
            if interrupt_after_start:
                original_thread_start(thread)
                assert phase_published.wait(timeout=5.0)
            raise injected
        original_thread_start(thread)

    monkeypatch.setattr(
        manager,
        "promote_construction_handoff_failure",
        record_promotion,
    )
    monkeypatch.setattr(threading.Thread, "start", interrupt_target_start)

    try:
        with pytest.raises(KeyboardInterrupt) as failure:
            owner.promote_failed_manager_handoff()

        assert failure.value is injected
        assert phase_published.is_set()
        assert promotion_calls == 1
        assert target_start_calls == (1 if interrupt_after_start else 2)
        assert receipt.phase == "FAILED_CLEANUP"
        with LMCacheManager._incomplete_construction_lock:
            assert LMCacheManager._construction_handoff_owners == {receipt: manager}
        assert LMCacheManager.provisional_construction_handoff_count() == 0
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1

        # A transferred/stale wrapper release is harmless after promotion.
        owner.release_provisional_manager()
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1
        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        assert manager.services_stopped
    finally:
        monkeypatch.setattr(threading.Thread, "start", original_thread_start)
        manager.stop_services()


def _source_line(
    function: Callable[..., Any],
    source: str,
    occurrence: int = 0,
) -> int:
    lines, start = inspect.getsourcelines(function)
    matches = [
        start + offset for offset, text in enumerate(lines) if text.strip() == source
    ]
    return matches[occurrence]


def _interrupt_on_line(
    function: Callable[..., Any],
    line: int,
    message: str,
) -> Callable[[Any, str, Any], Any]:
    triggered = False

    def trace(frame: Any, event: str, arg: Any) -> Any:
        del arg
        nonlocal triggered
        if (
            not triggered
            and frame.f_code is function.__code__
            and event == "line"
            and frame.f_lineno == line
        ):
            triggered = True
            sys.settrace(None)
            raise KeyboardInterrupt(message)
        return trace

    return trace


def _interrupt_on_factory_resource(
    function: Callable[..., Any],
    line: int,
    resource_name: str,
    message: str,
) -> Callable[[Any, str, Any], Any]:
    """Interrupt one generic factory helper invocation by resource name."""
    triggered = False

    def trace(frame: Any, event: str, arg: Any) -> Any:
        del arg
        nonlocal triggered
        if (
            not triggered
            and frame.f_code is function.__code__
            and event == "line"
            and frame.f_lineno == line
            and frame.f_locals.get("resource_name") == resource_name
        ):
            triggered = True
            sys.settrace(None)
            raise KeyboardInterrupt(message)
        return trace

    return trace


class _InterruptAfterRemove(set[Any]):
    """Publish one exact receipt removal, then interrupt its caller."""

    def __init__(self, values: set[Any], target: Any, message: str) -> None:
        super().__init__(values)
        self._target = target
        self._message = message
        self._triggered = False

    def remove(self, value: Any) -> None:
        super().remove(value)
        if value is self._target and not self._triggered:
            self._triggered = True
            raise KeyboardInterrupt(self._message)


class _FactoryCleanupProbe:
    """Minimal exact intermediate cleanup owner for receipt-state tests."""

    def __init__(self) -> None:
        self.calls = 0

    def cleanup(self) -> None:
        self.calls += 1


class _OpaqueLegacyFactory:
    """Zero-argument third-party callable whose signature cannot be read."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    @property
    def __signature__(self) -> Any:
        raise ValueError("opaque legacy signature")

    def __call__(self) -> object:
        self.calls += 1
        return self.result


def test_factory_result_owner_preserves_opaque_zero_arg_compatibility() -> None:
    result = object()
    legacy_factory = _OpaqueLegacyFactory(result)
    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        cast(Any, legacy_factory),
        "test-opaque-legacy-factory",
    )

    owner.run()

    assert legacy_factory.calls == 1
    assert owner.complete
    assert owner.final_result_published
    assert owner.result is result


def test_factory_result_owner_preserves_variadic_mock_compatibility() -> None:
    result = object()
    legacy_factory = MagicMock(return_value=result)
    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        cast(Any, legacy_factory),
        "test-variadic-legacy-factory",
    )

    owner.run()

    legacy_factory.assert_called_once_with()
    assert owner.complete
    assert owner.final_result_published
    assert owner.result is result


def test_factory_result_receipt_distinguishes_final_none() -> None:
    def publish_none(publisher: ServiceResultPublisher) -> None:
        publisher.publish_final(None)
        return None

    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        publish_none,
        "test-final-none-receipt",
    )

    assert not owner.result_published
    owner.run()
    assert owner.complete
    assert owner.final_result_published
    assert owner.result_published
    assert owner.result is None


def test_factory_result_receipt_enforces_exact_replacement_identity() -> None:
    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        lambda _publisher: None,
        "test-result-replacement-receipt",
    )
    transport = object()
    client = object()
    wrong_transport = object()
    transport_cleanup = _FactoryCleanupProbe()
    client_cleanup = _FactoryCleanupProbe()

    owner.publish_intermediate(transport, transport_cleanup)
    assert owner.owns_intermediate(transport)
    with pytest.raises(RuntimeError, match="identity changed"):
        owner.replace_intermediate(
            wrong_transport,
            client,
            client_cleanup,
        )
    assert owner.owns_intermediate(transport)

    owner.replace_intermediate(transport, client, client_cleanup)
    assert owner.owns_intermediate(client)
    assert owner.intermediate_cleanup_owner is client_cleanup
    with pytest.raises(RuntimeError, match="identity changed"):
        owner.replace_intermediate(transport, object(), _FactoryCleanupProbe())

    owner.publish_final(client)
    owner.publish_final(client)
    assert owner.final_result_published
    assert owner.result is client
    with pytest.raises(RuntimeError, match="identity changed"):
        owner.publish_final(object())


@requires_instruction_monitoring
def test_factory_result_publication_store_interrupt_replays_exact_state() -> None:
    owner: _RetainedFactoryCallOwner[object] = _RetainedFactoryCallOwner(
        lambda _publisher: None,
        "test-result-store-replay",
    )
    result = object()
    injected = KeyboardInterrupt("before exact factory-result state store")
    interrupt = InstructionInterruption(
        _RetainedFactoryCallOwner._commit_publication,
        instruction_offset(
            _RetainedFactoryCallOwner._commit_publication,
            "STORE_ATTR",
            argval="_publication",
        ),
        injected,
    )

    with interrupt:
        owner.publish_final(result)
    interrupt.assert_fired_once()
    assert owner.final_result_published
    assert owner.result is result


def test_shared_owner_removal_interrupt_cannot_destroy_peer_engine() -> None:
    engine = MagicMock(spec=LMCacheEngine)
    instance_id = "receipt-shared-engine"
    first = LMCacheManager(
        LMCacheEngineConfig.from_defaults(),
        _factory(engine, instance_id),
    )
    second = LMCacheManager(
        LMCacheEngineConfig.from_defaults(),
        _factory(engine, instance_id),
    )
    first_receipt = first._engine_ownership_receipt
    assert first_receipt is not None

    with LMCacheManager._engine_ownership_lock:
        owners = LMCacheManager._engine_ownership_receipts[engine]
        LMCacheManager._engine_ownership_receipts[engine] = _InterruptAfterRemove(
            owners,
            first_receipt,
            "after shared owner removal",
        )

    with patch.object(
        LMCacheEngineBuilder,
        "destroy",
        return_value=EngineDestroyResult.DESTROYED,
    ) as destroy:
        with pytest.raises(KeyboardInterrupt, match="shared owner removal"):
            first._release_engine_owner()

        assert not first._engine_owner_registered
        assert second._engine_owner_registered
        assert LMCacheManager._engine_owner_counts[engine] == 1

        first._release_engine_owner()
        destroy.assert_not_called()

        second._release_engine_owner()

    destroy.assert_called_once_with(instance_id, expected_engine=engine)
    assert engine not in LMCacheManager._engine_owner_counts


def test_last_owner_removal_interrupt_is_idempotent() -> None:
    engine = MagicMock(spec=LMCacheEngine)
    instance_id = "receipt-last-engine"
    manager = LMCacheManager(
        LMCacheEngineConfig.from_defaults(),
        _factory(engine, instance_id),
    )
    receipt = manager._engine_ownership_receipt
    assert receipt is not None

    with LMCacheManager._engine_ownership_lock:
        owners = LMCacheManager._engine_ownership_receipts[engine]
        LMCacheManager._engine_ownership_receipts[engine] = _InterruptAfterRemove(
            owners,
            receipt,
            "after last owner removal",
        )

    with patch.object(
        LMCacheEngineBuilder,
        "destroy",
        return_value=EngineDestroyResult.DESTROYED,
    ) as destroy:
        with pytest.raises(KeyboardInterrupt, match="last owner removal"):
            manager._release_engine_owner()

        assert not manager._engine_owner_registered
        manager._release_engine_owner()

    destroy.assert_called_once_with(instance_id, expected_engine=engine)
    assert engine not in LMCacheManager._engine_owner_counts


def test_interrupt_after_release_return_retries_without_redestroy() -> None:
    engine = MagicMock(spec=LMCacheEngine)
    instance_id = "receipt-post-return-engine"
    manager = LMCacheManager(
        LMCacheEngineConfig.from_defaults(),
        _factory(engine, instance_id),
    )
    interrupt = _interrupt_on_line(
        LMCacheManager._stop_services_impl,
        _source_line(
            LMCacheManager._stop_services_impl,
            "engine_destroyed = True",
            occurrence=1,
        ),
        "after engine-owner release returned",
    )

    with patch.object(
        LMCacheEngineBuilder,
        "destroy",
        return_value=EngineDestroyResult.DESTROYED,
    ) as destroy:
        sys.settrace(interrupt)
        try:
            with pytest.raises(KeyboardInterrupt, match="release returned"):
                manager.stop_services()
        finally:
            sys.settrace(None)

        assert not manager.services_stopped
        assert manager.lmcache_engine is engine
        assert not manager._engine_owner_registered

        manager.stop_services()

    assert manager.services_stopped
    assert manager.lmcache_engine is None
    destroy.assert_called_once_with(instance_id, expected_engine=engine)


def test_constructor_recovers_engine_lost_after_factory_return() -> None:
    engine = MagicMock(spec=LMCacheEngine)
    instance_id = "receipt-constructor-publication"
    config = LMCacheEngineConfig.from_defaults()
    factory = _factory(engine, instance_id)
    interrupt = _interrupt_on_line(
        LMCacheManager.__init__,
        _source_line(
            LMCacheManager.__init__,
            "self._publish_engine_acquisition_locked(engine)",
        ),
        "after engine factory returned",
    )

    with (
        patch.object(
            LMCacheEngineBuilder,
            "recover",
            return_value=engine,
        ) as recover_engine,
        patch.object(
            LMCacheEngineBuilder,
            "destroy",
            return_value=EngineDestroyResult.DESTROYED,
        ) as destroy,
    ):
        sys.settrace(interrupt)
        try:
            with pytest.raises(KeyboardInterrupt, match="engine factory returned"):
                LMCacheManager(config, factory)
        finally:
            sys.settrace(None)

    # The exact retained call already owns the returned engine; Builder lookup
    # is reserved for a retained attempt that failed before it could return.
    recover_engine.assert_not_called()
    destroy.assert_called_once_with(instance_id, expected_engine=engine)
    assert engine not in LMCacheManager._engine_owner_counts
    assert LMCacheManager.incomplete_construction_cleanup_count() == 0


def test_stop_retries_interrupted_constructor_engine_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructor cleanup must recover before treating engine=None as final."""
    engine = MagicMock(spec=LMCacheEngine)
    instance_id = "receipt-constructor-recovery-interrupt"
    config = LMCacheEngineConfig.from_defaults()
    factory = _factory(engine, instance_id)
    interrupt_after_factory = _interrupt_on_line(
        LMCacheManager.__init__,
        _source_line(
            LMCacheManager.__init__,
            "self._publish_engine_acquisition_locked(engine)",
        ),
        "after engine factory returned",
    )
    original_recover = LMCacheManager._recover_engine_acquisition_locked
    recovery_calls = 0

    def interrupt_first_recovery(manager: LMCacheManager) -> None:
        nonlocal recovery_calls
        recovery_calls += 1
        if recovery_calls == 1:
            raise KeyboardInterrupt("during constructor engine recovery")
        original_recover(manager)

    monkeypatch.setattr(
        LMCacheManager,
        "_recover_engine_acquisition_locked",
        interrupt_first_recovery,
    )

    with (
        patch.object(
            LMCacheEngineBuilder,
            "recover",
            return_value=engine,
        ) as recover_engine,
        patch.object(
            LMCacheEngineBuilder,
            "destroy",
            return_value=EngineDestroyResult.DESTROYED,
        ) as destroy,
    ):
        sys.settrace(interrupt_after_factory)
        try:
            with pytest.raises(KeyboardInterrupt, match="engine factory returned"):
                LMCacheManager(config, factory)
        finally:
            sys.settrace(None)

    assert recovery_calls == 2
    recover_engine.assert_not_called()
    destroy.assert_called_once_with(instance_id, expected_engine=engine)
    assert engine not in LMCacheManager._engine_owner_counts
    assert LMCacheManager.incomplete_construction_cleanup_count() == 0


@pytest.mark.parametrize(
    "failing_method",
    ["get_or_create_metadata", "get_engine_instance_id"],
    ids=["metadata", "instance-id"],
)
def test_precall_lookup_interrupt_resolves_as_no_engine(
    monkeypatch: pytest.MonkeyPatch,
    failing_method: str,
) -> None:
    """Metadata/ID interruption proves that no factory invocation occurred."""
    engine = MagicMock(spec=LMCacheEngine)
    instance_id = f"receipt-precall-{failing_method}"
    config = LMCacheEngineConfig.from_defaults()
    metadata = MagicMock()
    factory = _factory(engine, instance_id)
    factory.get_or_create_metadata.return_value = metadata
    getattr(factory, failing_method).side_effect = KeyboardInterrupt(
        f"injected {failing_method} interruption"
    )

    cleaned_managers: list[LMCacheManager] = []
    original_stop = LMCacheManager.stop_services

    def tracked_stop(manager: LMCacheManager) -> None:
        cleaned_managers.append(manager)
        original_stop(manager)

    monkeypatch.setattr(LMCacheManager, "stop_services", tracked_stop)

    with (
        patch.object(LMCacheEngineBuilder, "recover") as recover,
        patch.object(LMCacheEngineBuilder, "destroy") as destroy,
    ):
        with pytest.raises(KeyboardInterrupt, match=failing_method):
            LMCacheManager(config, factory)

    assert len(cleaned_managers) == 1
    manager = cleaned_managers[0]
    acquisition = manager._engine_acquisition_receipt
    assert acquisition.phase is _EngineAcquisitionPhase.PRECALL
    assert acquisition.call_owner is None
    assert acquisition.resolved
    assert manager.lmcache_engine is None
    assert manager.services_stopped
    factory.get_or_create_lmcache_engine.assert_not_called()
    recover.assert_not_called()
    destroy.assert_not_called()
    if failing_method == "get_or_create_metadata":
        factory.get_engine_instance_id.assert_not_called()
    else:
        factory.get_or_create_metadata.assert_called_once_with()
        factory.get_engine_instance_id.assert_called_once_with()
    assert LMCacheManager.incomplete_construction_cleanup_count() == 0


def test_traced_precall_boundary_runs_exact_retained_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A published invocation owner runs once without adopting Builder state."""
    instance_id = "receipt-traced-precall-scheduler-none"
    config = LMCacheEngineConfig.from_defaults()
    metadata = MagicMock()
    factory = _factory(MagicMock(spec=LMCacheEngine), instance_id)
    factory.get_or_create_metadata.return_value = metadata
    factory.get_or_create_lmcache_engine.return_value = None
    interrupt = _interrupt_on_line(
        LMCacheManager._invoke_engine_factory_locked,
        _source_line(
            LMCacheManager._invoke_engine_factory_locked,
            "owner.run()",
        ),
        "before retained engine factory invocation",
    )

    cleaned_managers: list[LMCacheManager] = []
    original_stop = LMCacheManager.stop_services

    def tracked_stop(manager: LMCacheManager) -> None:
        cleaned_managers.append(manager)
        original_stop(manager)

    monkeypatch.setattr(LMCacheManager, "stop_services", tracked_stop)

    with (
        patch.object(LMCacheEngineBuilder, "recover") as recover,
        patch.object(LMCacheEngineBuilder, "destroy") as destroy,
    ):
        sys.settrace(interrupt)
        try:
            with pytest.raises(KeyboardInterrupt, match="before retained"):
                LMCacheManager(config, factory)
        finally:
            sys.settrace(None)

    assert len(cleaned_managers) == 1
    manager = cleaned_managers[0]
    acquisition = manager._engine_acquisition_receipt
    assert acquisition.phase is _EngineAcquisitionPhase.INVOCATION_MAY_HAVE_BEGUN
    assert acquisition.call_owner is not None
    assert acquisition.call_owner.complete
    assert acquisition.resolved
    assert manager.lmcache_engine is None
    assert manager.services_stopped
    factory.get_or_create_lmcache_engine.assert_called_once()
    recover.assert_not_called()
    destroy.assert_not_called()
    assert LMCacheManager.incomplete_construction_cleanup_count() == 0


@pytest.mark.parametrize(
    ("resource_name", "factory_method", "receipt_attribute"),
    [
        (
            "lookup-client",
            "maybe_create_lookup_client",
            "_lookup_client_acquisition_receipt",
        ),
        (
            "lookup-server",
            "maybe_create_lookup_server",
            "_lookup_server_acquisition_receipt",
        ),
    ],
    ids=["client", "server"],
)
def test_lookup_factory_return_is_retained_before_manager_publication(
    monkeypatch: pytest.MonkeyPatch,
    resource_name: str,
    factory_method: str,
    receipt_attribute: str,
) -> None:
    """Cleanup adopts and closes a successful result lost before assignment."""
    config = LMCacheEngineConfig.from_defaults()
    factory = _factory(MagicMock(spec=LMCacheEngine), f"receipt-{resource_name}")
    factory.get_or_create_lmcache_engine.return_value = None
    service = MagicMock()
    getattr(factory, factory_method).return_value = service
    interrupt = _interrupt_on_factory_resource(
        LMCacheManager._acquire_retained_factory_result,
        _source_line(
            LMCacheManager._acquire_retained_factory_result,
            "publish_result(result)",
        ),
        resource_name,
        f"after {resource_name} factory returned",
    )

    cleanup_entries: list[tuple[LMCacheManager, Any, Any]] = []
    original_stop = LMCacheManager.stop_services
    original_recover = LMCacheManager._recover_retained_factory_result
    target_recovery_calls = 0

    def interrupt_first_recovery(
        manager: LMCacheManager,
        acquisition: Any,
        publish_result: Any,
        recovering_resource_name: str,
    ) -> None:
        nonlocal target_recovery_calls
        if recovering_resource_name == resource_name:
            target_recovery_calls += 1
            if target_recovery_calls == 1:
                raise KeyboardInterrupt(f"during {resource_name} acquisition recovery")
        original_recover(
            manager,
            acquisition,
            publish_result,
            recovering_resource_name,
        )

    def tracked_stop(manager: LMCacheManager) -> None:
        cleanup_entries.append((manager, manager.lookup_client, manager.lookup_server))
        original_stop(manager)

    monkeypatch.setattr(
        LMCacheManager,
        "_recover_retained_factory_result",
        interrupt_first_recovery,
    )
    monkeypatch.setattr(LMCacheManager, "stop_services", tracked_stop)

    with patch.object(LMCacheEngineBuilder, "destroy") as destroy:
        sys.settrace(interrupt)
        try:
            with pytest.raises(KeyboardInterrupt, match="factory returned"):
                LMCacheManager(config, factory)
        finally:
            sys.settrace(None)

    assert len(cleanup_entries) == 1
    manager, client_at_cleanup, server_at_cleanup = cleanup_entries[0]
    assert client_at_cleanup is None
    assert server_at_cleanup is None
    assert target_recovery_calls == 2
    if resource_name == "lookup-client":
        factory.maybe_create_lookup_server.assert_not_called()
        service.force_close.assert_not_called()
    else:
        factory.maybe_create_lookup_client.assert_called_once()

    acquisition = getattr(manager, receipt_attribute)
    assert acquisition.phase is _EngineAcquisitionPhase.INVOCATION_MAY_HAVE_BEGUN
    assert acquisition.call_owner is not None
    assert acquisition.call_owner.complete
    assert acquisition.result is service
    assert acquisition.resolved
    getattr(factory, factory_method).assert_called_once()
    service.close.assert_called_once_with()
    assert manager.lookup_client is None
    assert manager.lookup_server is None
    assert manager.services_stopped
    destroy.assert_not_called()
    assert LMCacheManager.incomplete_construction_cleanup_count() == 0


@pytest.mark.parametrize(
    ("resource_name", "factory_method", "receipt_attribute"),
    [
        (
            "lookup-client",
            "maybe_create_lookup_client",
            "_lookup_client_acquisition_receipt",
        ),
        (
            "lookup-server",
            "maybe_create_lookup_server",
            "_lookup_server_acquisition_receipt",
        ),
    ],
    ids=["client", "server"],
)
@requires_instruction_monitoring
def test_lookup_result_survives_worker_interrupt_after_owner_publication(
    monkeypatch: pytest.MonkeyPatch,
    resource_name: str,
    factory_method: str,
    receipt_attribute: str,
) -> None:
    """A callee-return interrupt cannot hide its already-final result."""
    config = LMCacheEngineConfig.from_defaults()
    factory = _factory(None, f"worker-result-{resource_name}")
    service = MagicMock()
    getattr(factory, factory_method).return_value = service
    worker_error = KeyboardInterrupt(
        f"after retained {resource_name} result publication"
    )
    factory_call = getattr(factory, factory_method).side_effect
    assert callable(factory_call)
    interrupt = InstructionInterruption(
        factory_call,
        instruction_offset(factory_call, "RETURN_VALUE"),
        worker_error,
        thread_name=f"lmcache-{resource_name}-factory",
    )
    cleaned_managers: list[LMCacheManager] = []
    original_stop = LMCacheManager.stop_services

    def tracked_stop(manager: LMCacheManager) -> None:
        cleaned_managers.append(manager)
        original_stop(manager)

    monkeypatch.setattr(LMCacheManager, "stop_services", tracked_stop)

    with interrupt:
        with pytest.raises(KeyboardInterrupt) as failure:
            LMCacheManager(config, factory)
    interrupt.assert_fired_once()

    assert failure.value is worker_error
    assert len(cleaned_managers) == 1
    manager = cleaned_managers[0]
    acquisition = getattr(manager, receipt_attribute)
    owner = acquisition.call_owner
    assert owner is not None
    assert owner.result_published
    assert not owner.complete
    assert owner.last_error is worker_error
    assert acquisition.result is service
    assert acquisition.resolved
    getattr(factory, factory_method).assert_called_once()
    service.close.assert_called_once_with()
    assert manager.lookup_client is None
    assert manager.lookup_server is None
    assert manager.services_stopped
    assert LMCacheManager.incomplete_construction_cleanup_count() == 0


@requires_instruction_monitoring
def test_engine_result_survives_worker_interrupt_after_owner_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final engine survives interruption at its factory return."""
    instance_id = "worker-result-engine"
    config = LMCacheEngineConfig.from_defaults()
    engine = MagicMock(spec=LMCacheEngine)
    factory = _factory(engine, instance_id)
    worker_error = KeyboardInterrupt("after retained engine result publication")
    factory_call = factory.get_or_create_lmcache_engine.side_effect
    assert callable(factory_call)
    interrupt = InstructionInterruption(
        factory_call,
        instruction_offset(factory_call, "RETURN_VALUE"),
        worker_error,
        thread_name=f"lmcache-engine-factory-{instance_id}",
    )
    cleaned_managers: list[LMCacheManager] = []
    original_stop = LMCacheManager.stop_services

    def tracked_stop(manager: LMCacheManager) -> None:
        cleaned_managers.append(manager)
        original_stop(manager)

    monkeypatch.setattr(LMCacheManager, "stop_services", tracked_stop)

    with (
        patch.object(LMCacheEngineBuilder, "recover") as recover,
        patch.object(
            LMCacheEngineBuilder,
            "destroy",
            return_value=EngineDestroyResult.DESTROYED,
        ) as destroy,
    ):
        with interrupt:
            with pytest.raises(KeyboardInterrupt) as failure:
                LMCacheManager(config, factory)
        interrupt.assert_fired_once()

    assert failure.value is worker_error
    assert len(cleaned_managers) == 1
    manager = cleaned_managers[0]
    owner = manager._engine_acquisition_receipt.call_owner
    assert owner is not None
    assert owner.result_published
    assert not owner.complete
    assert owner.last_error is worker_error
    assert manager._engine_acquisition_receipt.resolved
    recover.assert_not_called()
    destroy.assert_called_once_with(instance_id, expected_engine=engine)
    assert manager.lmcache_engine is None
    assert manager.services_stopped
    assert LMCacheManager.incomplete_construction_cleanup_count() == 0


@pytest.mark.parametrize(
    ("resource_name", "factory_method", "receipt_attribute"),
    [
        (
            "lookup-client",
            "maybe_create_lookup_client",
            "_lookup_client_acquisition_receipt",
        ),
        (
            "lookup-server",
            "maybe_create_lookup_server",
            "_lookup_server_acquisition_receipt",
        ),
    ],
    ids=["client", "server"],
)
def test_lookup_factory_precall_interrupt_resolves_absent(
    monkeypatch: pytest.MonkeyPatch,
    resource_name: str,
    factory_method: str,
    receipt_attribute: str,
) -> None:
    """An interruption before admission never invokes the lookup factory."""
    config = LMCacheEngineConfig.from_defaults()
    factory = _factory(MagicMock(spec=LMCacheEngine), f"precall-{resource_name}")
    factory.get_or_create_lmcache_engine.return_value = None
    interrupt = _interrupt_on_factory_resource(
        LMCacheManager._invoke_retained_factory_result,
        _source_line(
            LMCacheManager._invoke_retained_factory_result,
            ("acquisition.phase = _EngineAcquisitionPhase.INVOCATION_MAY_HAVE_BEGUN"),
        ),
        resource_name,
        f"before {resource_name} factory admission",
    )

    cleaned_managers: list[LMCacheManager] = []
    original_stop = LMCacheManager.stop_services

    def tracked_stop(manager: LMCacheManager) -> None:
        cleaned_managers.append(manager)
        original_stop(manager)

    monkeypatch.setattr(LMCacheManager, "stop_services", tracked_stop)

    with patch.object(LMCacheEngineBuilder, "destroy") as destroy:
        sys.settrace(interrupt)
        try:
            with pytest.raises(KeyboardInterrupt, match="factory admission"):
                LMCacheManager(config, factory)
        finally:
            sys.settrace(None)

    assert len(cleaned_managers) == 1
    manager = cleaned_managers[0]
    acquisition = getattr(manager, receipt_attribute)
    assert acquisition.phase is _EngineAcquisitionPhase.PRECALL
    assert acquisition.call_owner is not None
    assert not acquisition.call_owner.complete
    assert acquisition.result is None
    assert acquisition.resolved
    getattr(factory, factory_method).assert_not_called()
    if resource_name == "lookup-server":
        factory.maybe_create_lookup_client.assert_called_once()
    else:
        factory.maybe_create_lookup_server.assert_not_called()
    assert manager.lookup_client is None
    assert manager.lookup_server is None
    assert manager.services_stopped
    destroy.assert_not_called()
    assert LMCacheManager.incomplete_construction_cleanup_count() == 0


@pytest.mark.parametrize(
    ("resource_name", "factory_method", "receipt_attribute"),
    [
        (
            "lookup-client",
            "maybe_create_lookup_client",
            "_lookup_client_acquisition_receipt",
        ),
        (
            "lookup-server",
            "maybe_create_lookup_server",
            "_lookup_server_acquisition_receipt",
        ),
    ],
    ids=["client", "server"],
)
def test_lookup_factory_cleanup_must_drain_before_absent_resolution(
    monkeypatch: pytest.MonkeyPatch,
    resource_name: str,
    factory_method: str,
    receipt_attribute: str,
) -> None:
    """Persistent construction cleanup fences Manager engine destruction."""
    instance_id = f"retained-construction-cleanup-{resource_name}"
    config = LMCacheEngineConfig.from_defaults()
    engine = MagicMock(spec=LMCacheEngine)
    factory = _factory(engine, instance_id)
    factory_error = KeyboardInterrupt(f"{resource_name} construction failed")
    cleanup_error = RuntimeError(f"{resource_name} cleanup still retained")

    class _SequencedCleanupOwner:
        def __init__(self) -> None:
            self.calls = 0

        def cleanup(self) -> None:
            self.calls += 1
            if self.calls <= 2:
                raise cleanup_error

    cleanup_owner = _SequencedCleanupOwner()
    retain_lookup_construction_cleanup(factory_error, cleanup_owner)
    getattr(factory, factory_method).side_effect = factory_error

    cleaned_managers: list[LMCacheManager] = []
    original_stop = LMCacheManager.stop_services

    def tracked_stop(manager: LMCacheManager) -> None:
        cleaned_managers.append(manager)
        original_stop(manager)

    monkeypatch.setattr(LMCacheManager, "stop_services", tracked_stop)

    with patch.object(
        LMCacheEngineBuilder,
        "destroy",
        return_value=EngineDestroyResult.DESTROYED,
    ) as destroy:
        with pytest.raises(KeyboardInterrupt) as failure:
            LMCacheManager(config, factory)

        assert failure.value is factory_error
        assert failure.value.__cause__ is cleanup_error
        assert len(cleaned_managers) == 1
        manager = cleaned_managers[0]
        acquisition = getattr(manager, receipt_attribute)
        assert not acquisition.resolved
        assert acquisition.result is None
        assert cleanup_owner.calls == 2
        assert not manager.services_stopped
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1
        destroy.assert_not_called()

        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        assert cleanup_owner.calls == 3
        assert acquisition.resolved
        assert acquisition.result is None
        assert manager.services_stopped
        destroy.assert_called_once_with(instance_id, expected_engine=engine)

    assert getattr(factory, factory_method).call_count == 1
    if resource_name == "lookup-client":
        factory.maybe_create_lookup_server.assert_not_called()
    else:
        factory.maybe_create_lookup_client.assert_called_once()


@pytest.mark.parametrize(
    (
        "resource_name",
        "factory_method",
        "receipt_attribute",
        "cleanup_interrupt_type",
    ),
    [
        (
            "lookup-client",
            "maybe_create_lookup_client",
            "_lookup_client_acquisition_receipt",
            KeyboardInterrupt,
        ),
        (
            "lookup-server",
            "maybe_create_lookup_server",
            "_lookup_server_acquisition_receipt",
            SystemExit,
        ),
    ],
    ids=["client-keyboard-interrupt", "server-system-exit"],
)
def test_lookup_cleanup_interrupt_outranks_ordinary_factory_failure(
    monkeypatch: pytest.MonkeyPatch,
    resource_name: str,
    factory_method: str,
    receipt_attribute: str,
    cleanup_interrupt_type: type[BaseException],
) -> None:
    """Rollback process control cannot enter Manager's degraded path."""
    instance_id = f"cleanup-interrupt-precedence-{resource_name}"
    config = LMCacheEngineConfig.from_defaults()
    engine = MagicMock(spec=LMCacheEngine)
    factory = _factory(engine, instance_id)
    factory_error = ValueError(f"ordinary {resource_name} factory failure")
    cleanup_interrupt = cleanup_interrupt_type(
        f"{resource_name} construction cleanup interrupted"
    )

    class _SequencedCleanupOwner:
        def __init__(self) -> None:
            self.calls = 0

        def cleanup(self) -> None:
            self.calls += 1
            if self.calls <= 2:
                raise cleanup_interrupt

    cleanup_owner = _SequencedCleanupOwner()
    retain_lookup_construction_cleanup(factory_error, cleanup_owner)
    getattr(factory, factory_method).side_effect = factory_error

    cleaned_managers: list[LMCacheManager] = []
    original_stop = LMCacheManager.stop_services

    def tracked_stop(manager: LMCacheManager) -> None:
        cleaned_managers.append(manager)
        original_stop(manager)

    monkeypatch.setattr(LMCacheManager, "stop_services", tracked_stop)

    with patch.object(
        LMCacheEngineBuilder,
        "destroy",
        return_value=EngineDestroyResult.DESTROYED,
    ) as destroy:
        with pytest.raises(cleanup_interrupt_type) as failure:
            LMCacheManager(config, factory)

        assert failure.value is cleanup_interrupt
        assert cleanup_interrupt.__cause__ is factory_error
        assert factory_error.__cause__ is None
        assert cleanup_interrupt.__cause__ is not cleanup_interrupt
        assert factory_error.__cause__ is not cleanup_interrupt
        assert len(cleaned_managers) == 1
        manager = cleaned_managers[0]
        acquisition = getattr(manager, receipt_attribute)
        assert not acquisition.resolved
        assert cleanup_owner.calls == 2
        assert not manager.services_stopped
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1
        destroy.assert_not_called()

        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        assert cleanup_owner.calls == 3
        assert acquisition.resolved
        assert acquisition.result is None
        assert manager.services_stopped
        destroy.assert_called_once_with(instance_id, expected_engine=engine)

    assert getattr(factory, factory_method).call_count == 1


def test_factory_failure_cannot_adopt_incompatible_builder_generation() -> None:
    """Recovery must not turn a config-mismatch failure into ownership."""
    instance_id = "receipt-incompatible-builder-recovery"
    existing_config = LMCacheEngineConfig.from_defaults()
    requested_config = LMCacheEngineConfig.from_defaults()
    requested_config.max_local_cpu_size = existing_config.max_local_cpu_size + 1
    metadata = MagicMock()
    engine = MagicMock(spec=LMCacheEngine)
    stat_logger = MagicMock(spec=LMCacheStatsLogger)

    with LMCacheEngineBuilder._destroy_lock:
        LMCacheEngineBuilder._instances[instance_id] = engine
        LMCacheEngineBuilder._cfgs[instance_id] = existing_config
        LMCacheEngineBuilder._metadatas[instance_id] = metadata
        LMCacheEngineBuilder._stat_loggers[instance_id] = stat_logger

    factory = _factory(engine, instance_id)
    factory.get_or_create_metadata.return_value = metadata
    factory.get_or_create_lmcache_engine.side_effect = ValueError(
        "factory rejected incompatible configuration"
    )

    try:
        with patch.object(LMCacheEngineBuilder, "destroy") as destroy:
            manager = LMCacheManager(requested_config, factory)
            assert manager.lmcache_engine is None
            assert manager._engine_ownership_receipt is None
            assert engine not in LMCacheManager._engine_owner_counts

            manager.stop_services()
            destroy.assert_not_called()

        assert LMCacheEngineBuilder.get(instance_id) is engine
    finally:
        with LMCacheEngineBuilder._destroy_lock:
            LMCacheEngineBuilder._instances.pop(instance_id, None)
            LMCacheEngineBuilder._cfgs.pop(instance_id, None)
            LMCacheEngineBuilder._metadatas.pop(instance_id, None)
            LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)
            LMCacheEngineBuilder._ready_generations.pop(instance_id, None)


def test_factory_failure_cannot_adopt_generation_when_metadata_is_none() -> None:
    """An explicit None metadata expectation is not a recovery wildcard."""
    instance_id = "receipt-none-metadata-builder-recovery"
    config = LMCacheEngineConfig.from_defaults()
    foreign_metadata = MagicMock()
    engine = MagicMock(spec=LMCacheEngine)
    stat_logger = MagicMock(spec=LMCacheStatsLogger)

    with LMCacheEngineBuilder._destroy_lock:
        LMCacheEngineBuilder._instances[instance_id] = engine
        LMCacheEngineBuilder._cfgs[instance_id] = config
        LMCacheEngineBuilder._metadatas[instance_id] = foreign_metadata
        LMCacheEngineBuilder._stat_loggers[instance_id] = stat_logger

    factory = _factory(engine, instance_id)
    factory.get_or_create_metadata.return_value = None
    factory.get_or_create_lmcache_engine.side_effect = ValueError(
        "factory intended no engine"
    )

    try:
        manager = LMCacheManager(config, factory)

        assert manager.lmcache_engine is None
        assert manager._engine_ownership_receipt is None
        assert not manager._engine_acquisition_receipt.resolved
        assert engine not in LMCacheManager._engine_owner_counts
        assert LMCacheEngineBuilder.get(instance_id) is engine
    finally:
        with LMCacheEngineBuilder._destroy_lock:
            LMCacheEngineBuilder._instances.pop(instance_id, None)
            LMCacheEngineBuilder._cfgs.pop(instance_id, None)
            LMCacheEngineBuilder._metadatas.pop(instance_id, None)
            LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)
            LMCacheEngineBuilder._ready_generations.pop(instance_id, None)


def test_recovery_interrupt_outranks_ordinary_factory_failure() -> None:
    """A recovery KeyboardInterrupt cannot enter degraded Exception flow."""
    instance_id = "receipt-recovery-interrupt-precedence"
    config = LMCacheEngineConfig.from_defaults()
    metadata = MagicMock()
    engine = MagicMock(spec=LMCacheEngine)
    factory = _factory(engine, instance_id)
    factory.get_or_create_metadata.return_value = metadata
    factory_error = RuntimeError("ordinary factory failure")
    recovery_interrupt = KeyboardInterrupt("recovery was interrupted")
    factory.get_or_create_lmcache_engine.side_effect = factory_error

    with (
        patch.object(
            LMCacheEngineBuilder,
            "recover",
            side_effect=[recovery_interrupt, None],
        ) as recover_engine,
        patch.object(LMCacheEngineBuilder, "destroy") as destroy,
    ):
        with pytest.raises(KeyboardInterrupt) as failure:
            LMCacheManager(config, factory)

    assert failure.value is recovery_interrupt
    assert failure.value.__cause__ is factory_error
    recover_engine.assert_has_calls(
        [
            call(
                instance_id,
                expected_config=config,
                expected_metadata=metadata,
            ),
            call(
                instance_id,
                expected_config=config,
                expected_metadata=metadata,
            ),
        ]
    )
    assert recover_engine.call_count == 2
    destroy.assert_not_called()
    assert engine not in LMCacheManager._engine_owner_counts
    assert LMCacheManager.incomplete_construction_cleanup_count() == 0


def test_constructor_interrupt_after_builder_ready_recovers_exact_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover a retained Builder READY publication before factory adoption."""
    instance_id = "receipt-real-builder-ready-interrupt"
    config = LMCacheEngineConfig.from_defaults()
    metadata = MagicMock()
    constructed_engines: list[LMCacheEngine] = []
    recovered_engines: list[LMCacheEngine | None] = []
    engine_close_calls: list[LMCacheEngine] = []
    ready_published = threading.Event()
    release_ready_publisher = threading.Event()
    delegate_results: list[LMCacheEngine] = []
    delegate_errors: list[BaseException] = []
    delegate_threads: list[threading.Thread] = []

    class _StatsWorker:
        def __init__(self) -> None:
            self.alive = True
            self.join_calls = 0

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            assert timeout == 5.0
            self.join_calls += 1
            self.alive = False

    stats_workers: list[_StatsWorker] = []

    def stats_init(
        stat_logger: LMCacheStatsLogger,
        metadata: Any,
        log_interval: int,
        config: Any,
    ) -> None:
        del metadata, log_interval, config
        worker = _StatsWorker()
        stats_workers.append(worker)
        state = cast(Any, stat_logger)
        state.is_running = True
        state.shutdown_event = threading.Event()
        state.thread = worker

    def engine_init(
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
        constructed_engines.append(engine)
        assert engine.storage_manager is None

    original_engine_close = LMCacheEngine.close

    def counted_engine_close(engine: LMCacheEngine) -> None:
        engine_close_calls.append(engine)
        original_engine_close(engine)

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
    monkeypatch.setattr(LMCacheStatsLogger, "__init__", stats_init)
    monkeypatch.setattr(LMCacheEngine, "__init__", engine_init)
    monkeypatch.setattr(LMCacheEngine, "close", counted_engine_close)

    original_publish_ready = cache_engine_module._BuilderCreateOwner._publish_ready

    def publish_ready_then_block(owner: Any) -> None:
        original_publish_ready(owner)
        ready_published.set()
        assert release_ready_publisher.wait(timeout=5.0)

    monkeypatch.setattr(
        cache_engine_module._BuilderCreateOwner,
        "_publish_ready",
        publish_ready_then_block,
    )

    def create_on_builder() -> None:
        try:
            delegate_results.append(
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
            delegate_errors.append(error)

    def delegate_builder_creation(_publisher: object) -> LMCacheEngine:
        delegate = threading.Thread(
            target=create_on_builder,
            name="manager-builder-delegate",
        )
        delegate_threads.append(delegate)
        delegate.start()
        assert ready_published.wait(timeout=5.0)
        raise KeyboardInterrupt("after real Builder READY publication")

    factory = _factory(MagicMock(spec=LMCacheEngine), instance_id)
    factory.get_or_create_metadata.return_value = metadata
    factory.get_or_create_lmcache_engine.side_effect = delegate_builder_creation
    factory.get_or_create_lmcache_engine.__signature__ = inspect.signature(
        delegate_builder_creation
    )
    original_recover = LMCacheEngineBuilder.recover

    def recover(
        instance_id: str,
        *,
        expected_config: LMCacheEngineConfig,
        expected_metadata: Any,
    ) -> LMCacheEngine | None:
        assert expected_config is config
        assert expected_metadata is metadata
        with LMCacheEngineBuilder._destroy_lock:
            creation_owner = LMCacheEngineBuilder._creation_owners.get(instance_id)
            ready_generation = LMCacheEngineBuilder._ready_generations.get(instance_id)
            assert creation_owner is not None
            assert ready_generation is not None
            assert creation_owner.engine is ready_generation.engine
        assert LMCacheEngineBuilder.get(instance_id) is None

        release_ready_publisher.set()
        recovered = original_recover(
            instance_id,
            expected_config=expected_config,
            expected_metadata=expected_metadata,
        )
        assert len(delegate_threads) == 1
        delegate_threads[0].join(timeout=5.0)
        assert not delegate_threads[0].is_alive()
        assert delegate_errors == []
        assert delegate_results == [recovered]
        recovered_engines.append(recovered)
        return recovered

    try:
        with (
            patch.object(
                LMCacheEngineBuilder,
                "recover",
                side_effect=recover,
            ) as recover_engine,
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
            with pytest.raises(
                KeyboardInterrupt,
                match="after real Builder READY publication",
            ):
                LMCacheManager(config, factory)

        assert len(constructed_engines) == 1
        engine = constructed_engines[0]
        assert delegate_results == [engine]
        assert delegate_errors == []
        assert recovered_engines == [engine]
        recover_engine.assert_called_once_with(
            instance_id,
            expected_config=config,
            expected_metadata=metadata,
        )
        factory.get_or_create_lmcache_engine.assert_called_once()
        assert engine_close_calls == [engine]
        assert len(stats_workers) == 1
        assert not stats_workers[0].alive
        assert stats_workers[0].join_calls == 1
        destroy_monitor.assert_called_once_with()
        drop_blender.assert_called_once_with(instance_id)
        unregister_model.assert_called_once_with(instance_id)

        assert engine not in LMCacheManager._engine_owner_counts
        assert LMCacheManager.incomplete_construction_cleanup_count() == 0
        assert instance_id not in LMCacheEngineBuilder._creation_owners
        assert instance_id not in LMCacheEngineBuilder._destroy_owners
        assert instance_id not in LMCacheEngineBuilder._destroy_intents
        assert instance_id not in LMCacheEngineBuilder._ready_generations
        assert instance_id not in LMCacheEngineBuilder._instances
        assert instance_id not in LMCacheEngineBuilder._cfgs
        assert instance_id not in LMCacheEngineBuilder._metadatas
        assert instance_id not in LMCacheEngineBuilder._stat_loggers
        assert LMCacheEngineBuilder._destroyed_engines.get(engine) is True
    finally:
        sys.settrace(None)
        release_ready_publisher.set()
        for delegate in delegate_threads:
            delegate.join(timeout=5.0)
        with LMCacheEngineBuilder._destroy_lock:
            LMCacheEngineBuilder._creation_owners.pop(instance_id, None)
            LMCacheEngineBuilder._destroy_owners.pop(instance_id, None)
            LMCacheEngineBuilder._destroy_intents.discard(instance_id)
            LMCacheEngineBuilder._ready_generations.pop(instance_id, None)
            LMCacheEngineBuilder._instances.pop(instance_id, None)
            LMCacheEngineBuilder._cfgs.pop(instance_id, None)
            LMCacheEngineBuilder._metadatas.pop(instance_id, None)
            LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)
            for engine in constructed_engines:
                LMCacheEngineBuilder._destroyed_engines.pop(engine, None)
        for worker in stats_workers:
            worker.alive = False
