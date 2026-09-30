# SPDX-License-Identifier: Apache-2.0
"""What each connector role builds for a shared-storage P/D handoff.

Every other test of this path builds the connector with ``__new__`` and
sets the attributes it needs, which cannot show that the constructor
reaches the same state -- and the constructor is where the roles diverge:
a scheduler must build no status sender and a worker must refuse to run
without one. These run the real ``__init__``.

LMCache's service layer is stubbed, so this covers the constructor's role
gating and notification wiring and not engine startup. Nothing here opens
a socket or a device.
"""

# Standard
from types import SimpleNamespace
from typing import Any, Iterator, Optional
import threading

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)

# First Party
from lmcache.integration.vllm import vllm_v1_adapter
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from lmcache.v1.config import LMCacheEngineConfig


class _FakeKVTransferConfig:
    """The slice of vLLM's KV transfer config the constructor reads."""

    def __init__(self, extra: dict[str, Any]) -> None:
        self.kv_role = "kv_producer"
        self.kv_connector_extra_config = extra

    def get_from_extra_config(self, key: str, default: Any) -> Any:
        return self.kv_connector_extra_config.get(key, default)


class _FakeManager:
    """Stand in for LMCacheManager without starting any service."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.started = False
        self.post_init_calls = 0
        self.lmcache_engine: Optional[Any] = None
        self.lmcache_engine_metadata = None

    def start_services(self) -> None:
        self.started = True

    def post_init(self) -> None:
        self.post_init_calls += 1


class _RecordingSender:
    """Record how the constructor configured the sender, open nothing."""

    instances: list["_RecordingSender"] = []

    def __init__(self, host: str, port: int, *, timeout_s: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sent: list[Any] = []
        _RecordingSender.instances.append(self)

    def send(self, message: Any) -> None:
        self.sent.append(message)

    def close(self) -> None:
        return None


def _vllm_config(extra: dict[str, Any]) -> Any:
    return SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        kv_transfer_config=_FakeKVTransferConfig(extra),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(get_num_layers=lambda _parallel: 2),
    )


class _FakeParent:
    """The owning connector, which nothing under test calls into.

    Deliberately not a ``KVConnectorBase_V1`` subclass: the constructor
    only inspects the parent to warn about a connector predating
    ``register_kv_caches``, and that check returns early because the
    stubbed manager has no engine.
    """

    def __init__(self) -> None:
        self._connector_metadata = None


@pytest.fixture(autouse=True)
def _stub_service_layer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Let __init__ run for real without services, sockets or Prometheus."""
    _RecordingSender.instances = []
    monkeypatch.setattr(vllm_v1_adapter, "LMCacheManager", _FakeManager)
    monkeypatch.setattr(
        vllm_v1_adapter,
        "VllmServiceFactory",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(vllm_v1_adapter, "StoragePDStatusSender", _RecordingSender)
    monkeypatch.setattr(vllm_v1_adapter, "print_banner_once", lambda _stream: None)
    yield
    _RecordingSender.instances = []


def _build(
    role: KVConnectorRole,
    *,
    storage_pd: bool = True,
    proxy: bool = True,
    skip_notification: bool = False,
    extra_config: Optional[dict[str, Any]] = None,
    monkeypatch: pytest.MonkeyPatch,
) -> LMCacheConnectorV1Impl:
    """Construct a connector through its real ``__init__``."""
    extra: dict[str, Any] = {"rust_raw_block.storage_pd_mode": storage_pd}
    extra.update(extra_config or {})
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        lmcache_instance_id="test_storage_pd_construction",
    )
    config.extra_config = extra
    config.pd_skip_proxy_notification = skip_notification
    config.pd_proxy_host = "127.0.0.1" if proxy else None
    config.pd_proxy_port = 30081 if proxy else None
    monkeypatch.setattr(vllm_v1_adapter, "lmcache_get_or_create_config", lambda: config)
    return LMCacheConnectorV1Impl(
        _vllm_config(extra),  # type: ignore[arg-type]
        role,
        _FakeParent(),
    )


def _close(connector: LMCacheConnectorV1Impl) -> None:
    queue = connector._storage_pd_notify_queue
    if queue is not None:
        queue.close()


def test_a_worker_builds_a_status_sender_and_a_delivery_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _build(
        KVConnectorRole.WORKER,
        extra_config={
            "rust_raw_block.status_queue_capacity": 4,
            "rust_raw_block.status_retry_interval_s": 0.25,
            "rust_raw_block.status_deadline_s": 3.0,
            "rust_raw_block.status_send_timeout_s": 2.0,
        },
        monkeypatch=monkeypatch,
    )
    try:
        assert connector._storage_pd_mode
        assert connector._storage_pd_raw_role == "writer"
        sender = connector._storage_pd_status_sender
        assert isinstance(sender, _RecordingSender)
        assert (sender.host, sender.port, sender.timeout_s) == (
            "127.0.0.1",
            30081,
            2.0,
        )
        queue = connector._storage_pd_notify_queue
        assert queue is not None
        # The policy comes from the configuration, not from the defaults.
        assert queue._capacity == 4
        assert queue._retry_interval_s == 0.25
        assert queue._deadline_s == 3.0
    finally:
        _close(connector)


def test_a_scheduler_builds_no_sender_and_completes_no_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler owes a consumer nothing, and must not claim otherwise.

    It has no publication to announce, so demanding a sender there would
    reject a configuration that is correct, and letting it run the
    writer's completion path would record deliveries that never happened.
    """
    connector = _build(KVConnectorRole.SCHEDULER, monkeypatch=monkeypatch)
    try:
        assert connector._storage_pd_mode
        assert connector._storage_pd_status_sender is None
        assert connector._storage_pd_notify_queue is None
        assert _RecordingSender.instances == []
        assert connector.get_finished({"request-1"}) == (None, None)
    finally:
        _close(connector)


def test_a_worker_with_no_way_to_announce_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail at construction, not on the first request that finishes."""
    with pytest.raises(ValueError, match="pd_proxy_host"):
        _build(KVConnectorRole.WORKER, proxy=False, monkeypatch=monkeypatch)
    assert _RecordingSender.instances == []


def test_a_worker_told_to_run_without_a_consumer_builds_neither(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running with no consumer stays supported when it is asked for."""
    connector = _build(
        KVConnectorRole.WORKER,
        proxy=False,
        skip_notification=True,
        monkeypatch=monkeypatch,
    )
    try:
        assert connector._storage_pd_status_sender is None
        assert connector._storage_pd_notify_queue is None
        assert not connector._storage_pd_notify_required
    finally:
        _close(connector)


def test_a_worker_outside_storage_pd_mode_builds_neither(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary deployment must not acquire a P/D notification path."""
    connector = _build(
        KVConnectorRole.WORKER, storage_pd=False, monkeypatch=monkeypatch
    )
    try:
        assert not connector._storage_pd_mode
        assert connector._storage_pd_status_sender is None
        assert connector._storage_pd_notify_queue is None
        assert connector.get_finished({"request-1"}) == (None, None)
    finally:
        _close(connector)


def test_the_delivery_queue_a_worker_builds_actually_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructed queue is a working one, not just an object.

    A delivery worker that was never started would leave every status
    pending forever, and each test above would still pass.
    """
    connector = _build(KVConnectorRole.WORKER, monkeypatch=monkeypatch)
    queue = connector._storage_pd_notify_queue
    assert queue is not None
    try:
        message = SimpleNamespace(req_id="request-1")
        assert queue.enqueue("request-1", message)  # type: ignore[arg-type]
        settled: list[Any] = []
        for _ in range(5000):
            settled = queue.poll()
            if settled:
                break
            threading.Event().wait(0.002)
        assert [(item.key, item.state) for item in settled] == [
            ("request-1", "DELIVERED")
        ]
        sender = connector._storage_pd_status_sender
        assert isinstance(sender, _RecordingSender)
        assert sender.sent == [message]
    finally:
        _close(connector)
