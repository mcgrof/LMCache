# SPDX-License-Identifier: Apache-2.0
"""Regression test for LMCache#3318.

The vLLM v1 adapter previously asserted
``len(slot_mapping) == len(token_ids)`` inside ``wait_for_save``. When the
state desynced (e.g. upstream allocation failure or preemption-induced
mismatch) the assertion fired as an unhandled ``AssertionError`` and
killed the entire EngineCore process for every connected user.

The fix replaces the assert with a logged ``continue`` so the engine
stays alive and only the affected request's save is dropped. This test
locks in that behavior by feeding ``wait_for_save`` a request whose
``slot_mapping`` and ``token_ids`` lengths disagree and asserting:

1. ``wait_for_save`` does not raise.
2. A warning is emitted naming the request id and both lengths.
3. ``lmcache_engine.store`` is not called for the desynced request
   (the save is dropped, not silently corrupted).
4. ``lookup_unpin`` is still called so the pin count stays balanced.
"""

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
import logging
import threading
import time

# Third Party
from vllm.v1.request import RequestStatus
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    SaveSpec,
)
from lmcache.v1.storage_backend.raw_block import RawBlockPublicationReceipt
from lmcache.v1.storage_backend.storage_pd_protocol import (
    StoragePDNotificationQueue,
    StoragePDReadAck,
    StoragePDStatus,
)


class _FakeParent:
    def __init__(self, metadata: LMCacheConnectorMetadata) -> None:
        self._connector_metadata = metadata

    def _get_connector_metadata(self) -> LMCacheConnectorMetadata:
        return self._connector_metadata


class _FakeEngine:
    """Records calls to ``lookup_unpin`` and ``store`` so the test can
    assert which paths fired."""

    def __init__(self) -> None:
        self.unpinned: list[str] = []
        self.store_calls: list[str] = []

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store(self, *args, **kwargs) -> None:
        self.store_calls.append(kwargs.get("req_id", "<unknown>"))


def _make_desync_request(
    req_id: str, token_ids_len: int, slot_mapping_len: int
) -> SimpleNamespace:
    """Build a request whose ``token_ids`` and ``slot_mapping`` lengths
    disagree, simulating a state desync."""
    return SimpleNamespace(
        req_id=req_id,
        token_ids=list(range(token_ids_len)),
        slot_mapping=torch.arange(slot_mapping_len, dtype=torch.long),
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=True),
        disagg_spec=None,
        is_last_prefill=True,
        request_configs=None,
    )


def _make_connector(
    requests: list[SimpleNamespace],
) -> tuple[LMCacheConnectorV1Impl, _FakeEngine]:
    metadata = LMCacheConnectorMetadata(requests=requests)  # type: ignore[arg-type]
    engine = _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)
    # ``lmcache_engine`` is a read-only property backed by ``self._manager``;
    # inject the fake engine through the manager so the property resolves to it.
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=engine
    )
    connector.kv_role = "kv_producer"
    connector.use_layerwise = False
    connector.enable_blending = False
    connector.device = "cpu"
    connector._lmcache_chunk_size = 8
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector.config = SimpleNamespace(pd_bidirectional=False)
    # This fixture builds the connector without running __init__, so every
    # attribute the code under test reads has to be set here. wait_for_save
    # consults the storage handoff mode to decide whether to record a wire
    # request id; this connector is not in that mode.
    connector._storage_pd_mode = False
    return connector, engine


def _make_storage_pd_connector() -> LMCacheConnectorV1Impl:
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._storage_pd_mode = True
    connector._storage_pd_raw_role = "writer"
    connector._storage_pd_store_futures = {}
    connector._storage_pd_wire_req_ids = {}
    connector._storage_pd_engine_finished = set()
    connector._storage_pd_returned = OrderedDict()
    connector._storage_pd_aborted = set()
    connector._storage_pd_failures = {}
    connector._storage_pd_terminal_states = {}
    connector._storage_pd_receipts = {}
    connector._storage_pd_obligations = {}
    connector._storage_pd_ack_outbox = OrderedDict()
    connector._storage_pd_acks_sent = OrderedDict()
    connector._storage_pd_status_sender = None
    connector._storage_pd_notify_queue = None
    connector._storage_pd_notify_required = False
    # get_finished consults the role: a scheduler never completes a storage
    # handoff. This fixture stands in for a worker.
    connector._role = KVConnectorRole.WORKER
    connector._storage_pd_tp_rank = 0
    connector._storage_pd_lock = threading.Lock()
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=None
    )
    connector.use_layerwise = False
    connector.async_loading = False
    connector._request_trackers = {}
    connector.config = SimpleNamespace(
        get_extra_config_value=lambda _key, default: default
    )
    return connector


_OPEN_QUEUES: list[StoragePDNotificationQueue] = []

SETTLE_TIMEOUT_S = 10.0


@pytest.fixture(autouse=True)
def _close_notification_queues() -> Iterator[None]:
    """Stop any delivery worker a test started, however the test ended."""
    yield
    while _OPEN_QUEUES:
        _OPEN_QUEUES.pop().close()


class _SenderAdapter:
    """Present a test sender the way the queue now calls one.

    The queue bounds each attempt by what is left of an obligation's deadline
    and closes the sender it owns, so it passes a timeout to both. Absorbing
    that here keeps every test's sender about sending, and keeps a dropped
    keyword from looking like an unreachable peer -- which is how it first
    presented.
    """

    def __init__(self, sender: Any) -> None:
        self._sender = sender
        self.closed_with: list[Any] = []

    def __getattr__(self, item: str) -> Any:
        return getattr(self._sender, item)

    def send(self, message: Any, *, timeout_s: Any = None) -> None:
        self._sender.send(message)

    def close(self, timeout_s: Any = None) -> None:
        self.closed_with.append(timeout_s)
        inner = getattr(self._sender, "close", None)
        if callable(inner):
            inner()


def _attach_notification_queue(
    connector: LMCacheConnectorV1Impl,
    sender: Any,
    **kwargs: Any,
) -> StoragePDNotificationQueue:
    """Give the connector the real delivery queue over a test sender."""
    kwargs.setdefault("retry_interval_s", 0.001)
    kwargs.setdefault("deadline_s", SETTLE_TIMEOUT_S)
    sender = _SenderAdapter(sender)
    queue = StoragePDNotificationQueue(
        sender,
        on_unreported=connector._log_storage_pd_delivery,
        **kwargs,
    )
    _OPEN_QUEUES.append(queue)
    connector._storage_pd_status_sender = sender
    connector._storage_pd_notify_queue = queue
    return queue


@contextmanager
def _captured_adapter_logs(level: int) -> Iterator[list[logging.LogRecord]]:
    """Collect the adapter's own records, whatever pytest can see.

    ``init_logger`` sets ``propagate = False``, so these never reach
    ``caplog``. Attaching a handler to the named logger is what the
    desynced-save regression below does too, and for the same reason.
    """
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=level)
    adapter_logger = logging.getLogger("lmcache.integration.vllm.vllm_v1_adapter")
    original_level = adapter_logger.level
    adapter_logger.setLevel(level)
    adapter_logger.addHandler(handler)
    try:
        yield records
    finally:
        adapter_logger.removeHandler(handler)
        adapter_logger.setLevel(original_level)


def _finish_until_released(
    connector: LMCacheConnectorV1Impl,
    finished_req_ids: set[str] | None = None,
) -> set[str]:
    """Step ``get_finished`` until it releases something, as vLLM does.

    Delivery no longer happens inside the call that decides a status, so a
    request is released on a later step rather than the one that queued it.
    """
    released, _ = connector.get_finished(finished_req_ids or set())
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while not released and time.monotonic() < deadline:
        time.sleep(0.002)
        released, _ = connector.get_finished(set())
    return released


def test_storage_pd_reader_does_not_run_writer_completion() -> None:
    connector = _make_storage_pd_connector()
    connector._storage_pd_raw_role = "reader"

    assert connector.get_finished({"request-1"}) == (None, None)


def test_storage_pd_writer_defers_source_block_free_until_publication() -> None:
    connector = _make_storage_pd_connector()
    request = SimpleNamespace(
        request_id="request-1",
        status=RequestStatus.FINISHED_STOPPED,
        kv_transfer_params=None,
    )

    assert connector.request_finished(request, [1, 2]) == (True, None)


def test_storage_pd_reader_does_not_defer_source_block_free() -> None:
    connector = _make_storage_pd_connector()
    connector._storage_pd_raw_role = "reader"
    request = SimpleNamespace(
        request_id="request-1",
        status=RequestStatus.FINISHED_STOPPED,
        kv_transfer_params=None,
    )

    assert connector.request_finished(request, [1, 2]) == (False, None)


def test_storage_pd_aborted_writer_does_not_defer_source_block_free() -> None:
    connector = _make_storage_pd_connector()
    request = SimpleNamespace(
        request_id="request-1",
        status=RequestStatus.FINISHED_ABORTED,
        kv_transfer_params=None,
    )

    assert connector.request_finished(request, [1, 2]) == (False, None)


def test_storage_pd_get_finished_waits_for_store_publication() -> None:
    class Sender:
        def __init__(self) -> None:
            self.statuses: list[StoragePDStatus] = []

        def send(self, status: StoragePDStatus) -> None:
            self.statuses.append(status)

    connector = _make_storage_pd_connector()
    sender = Sender()
    _attach_notification_queue(connector, sender)
    completion: Future[list[RawBlockPublicationReceipt]] = Future()
    connector._storage_pd_store_futures["request-1"] = completion

    assert connector.get_finished({"request-1"}) == (set(), None)
    assert sender.statuses == []
    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    completion.set_result([receipt])
    assert _finish_until_released(connector) == {"request-1"}
    assert [status.manifest_digest for status in sender.statuses] == ["digest"]
    assert connector.get_finished({"request-1"}) == (set(), None)


def test_storage_pd_get_finished_keeps_wire_and_vllm_request_ids_distinct() -> None:
    class StorageManager:
        def __init__(self) -> None:
            self.finished: list[str] = []

        def finish_request(self, req_id: str) -> None:
            self.finished.append(req_id)

    class Sender:
        def __init__(self) -> None:
            self.statuses: list[StoragePDStatus] = []

        def send(self, status: StoragePDStatus) -> None:
            self.statuses.append(status)

    connector = _make_storage_pd_connector()
    storage_manager = StorageManager()
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=SimpleNamespace(storage_manager=storage_manager)
    )
    sender = Sender()
    _attach_notification_queue(connector, sender)
    connector._storage_pd_wire_req_ids["cmpl-internal-0"] = "proxy-uuid"
    completion: Future[list[RawBlockPublicationReceipt]] = Future()
    completion.set_result([RawBlockPublicationReceipt("writer", 1, 1, "digest")])
    connector._storage_pd_store_futures["cmpl-internal-0"] = completion

    assert _finish_until_released(connector, {"cmpl-internal-0"}) == {"cmpl-internal-0"}
    assert storage_manager.finished == ["proxy-uuid"]
    assert [status.req_id for status in sender.statuses] == ["proxy-uuid"]


def test_storage_pd_a_failed_send_does_not_become_a_failed_publication() -> None:
    """A send that fails must not turn a durable publication into a failure.

    The publication already happened and its receipt is recorded, so the
    only thing left is telling the peer. Reporting FAILED instead would
    send the consumer looking elsewhere for an object that is on the
    device and correct. Retrying is the delivery queue's job now, and the
    request is held until it either lands or is given up on.
    """

    class FlakySender:
        def __init__(self) -> None:
            self.statuses: list[StoragePDStatus] = []
            self.attempts = 0

        def send(self, status: StoragePDStatus) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("peer not reachable")
            self.statuses.append(status)

    connector = _make_storage_pd_connector()
    sender = FlakySender()
    _attach_notification_queue(connector, sender)
    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    completion: Future[list[RawBlockPublicationReceipt]] = Future()
    completion.set_result([receipt])
    connector._storage_pd_store_futures["request-1"] = completion

    assert _finish_until_released(connector, {"request-1"}) == {"request-1"}
    assert sender.attempts >= 2
    assert [status.state for status in sender.statuses] == ["READY"]
    assert connector._storage_pd_failures == {}
    assert "request-1" not in connector._storage_pd_aborted


def test_storage_pd_get_finished_does_not_wait_for_the_peer() -> None:
    """A peer that stops reading must not be able to freeze the connector.

    The status socket can block for as long as the consumer likes. The
    engine step therefore hands the status over and returns; it neither
    waits for the peer nor holds the connector's state lock while the
    send is in flight, or every other user of this connector would queue
    behind an unrelated consumer.
    """
    release_peer = threading.Event()
    sending = threading.Event()
    lock_free_during_send: list[bool] = []

    class BlockedSender:
        def __init__(self, connector: LMCacheConnectorV1Impl) -> None:
            self._connector = connector

        def send(self, status: StoragePDStatus) -> None:
            acquired = self._connector._storage_pd_lock.acquire(blocking=False)
            lock_free_during_send.append(acquired)
            if acquired:
                self._connector._storage_pd_lock.release()
            sending.set()
            assert release_peer.wait(SETTLE_TIMEOUT_S)

    connector = _make_storage_pd_connector()
    _attach_notification_queue(connector, BlockedSender(connector))
    completion: Future[list[RawBlockPublicationReceipt]] = Future()
    completion.set_result([RawBlockPublicationReceipt("writer", 1, 1, "digest")])
    connector._storage_pd_store_futures["request-1"] = completion

    assert connector.get_finished({"request-1"}) == (set(), None)
    assert sending.wait(SETTLE_TIMEOUT_S)
    # The peer is still inside the send and the step already returned.
    assert connector.get_finished(set()) == (set(), None)
    assert lock_free_during_send == [True], (
        "the state lock was held across the status send"
    )

    release_peer.set()
    assert _finish_until_released(connector) == {"request-1"}


def test_storage_pd_get_finished_refuses_to_release_without_a_consumer() -> None:
    """A producer that owes a consumer a status must not pretend it sent one.

    Releasing the request would tell the engine the handoff is done while
    the consumer is still waiting to hear that anything was published.
    Nothing in this instance can reach it, so the misconfiguration has to
    surface rather than pass for a delivery.
    """
    connector = _make_storage_pd_connector()
    connector._storage_pd_notify_required = True
    completion: Future[list[RawBlockPublicationReceipt]] = Future()
    completion.set_result([RawBlockPublicationReceipt("writer", 1, 1, "digest")])
    connector._storage_pd_store_futures["request-1"] = completion

    with pytest.raises(RuntimeError, match="no status sender"):
        connector.get_finished({"request-1"})
    assert not connector._storage_pd_returned


def test_storage_pd_get_finished_releases_when_notification_is_disabled() -> None:
    """Running without a consumer stays supported when it is asked for."""
    connector = _make_storage_pd_connector()
    connector._storage_pd_notify_required = False
    completion: Future[list[RawBlockPublicationReceipt]] = Future()
    completion.set_result([RawBlockPublicationReceipt("writer", 1, 1, "digest")])
    connector._storage_pd_store_futures["request-1"] = completion

    assert connector.get_finished({"request-1"}) == ({"request-1"}, None)


def test_storage_pd_read_ack_carries_the_published_request_identity() -> None:
    """The acknowledgement must name the request the producer published.

    A proxy assigns the identity that both sides agreed on, and the
    decoder's engine gives the same request a different local name.
    READY is checked against the published identity, so the
    acknowledgement has to use it too, or the producer cannot match it to
    anything and the extents it is holding stay held.
    """

    class Sender:
        def __init__(self) -> None:
            self.acks: list[StoragePDReadAck] = []

        def send(self, message) -> None:
            self.acks.append(message)

    connector = _make_storage_pd_connector()
    sender = Sender()
    _attach_notification_queue(connector, sender)
    receipt = RawBlockPublicationReceipt("writer-epoch", 7, 1, "digest")
    status = StoragePDStatus.ready("proxy-uuid", 0, receipt)

    connector._ack_storage_pd_restore("cmpl-internal-0", status)
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while not sender.acks and time.monotonic() < deadline:
        time.sleep(0.002)

    assert [ack.req_id for ack in sender.acks] == ["proxy-uuid"]
    ack = sender.acks[0]
    assert ack.writer_epoch == "writer-epoch"
    assert ack.checkpoint_seq == 7
    assert ack.manifest_digest == "digest"
    assert ack.tp_rank == connector._storage_pd_tp_rank

    # The local name still governs sending it only once.
    connector._ack_storage_pd_restore("cmpl-internal-0", status)
    assert len(sender.acks) == 1


def test_storage_pd_releasing_a_request_clears_its_state() -> None:
    """A finished request must not leave one entry per container behind.

    Every map here holds something the request needs while it is in
    flight. Keeping them costs the engine one entry per request served,
    for as long as it runs.
    """
    connector = _make_storage_pd_connector()

    class Sender:
        def send(self, status) -> None:
            return None

    _attach_notification_queue(connector, Sender())
    connector._storage_pd_wire_req_ids["request-1"] = "proxy-uuid"
    completion: Future[list[RawBlockPublicationReceipt]] = Future()
    completion.set_result([RawBlockPublicationReceipt("writer", 1, 1, "digest")])
    connector._storage_pd_store_futures["request-1"] = completion

    assert _finish_until_released(connector, {"request-1"}) == {"request-1"}

    assert connector._storage_pd_store_futures == {}
    assert connector._storage_pd_wire_req_ids == {}
    assert connector._storage_pd_receipts == {}
    assert connector._storage_pd_obligations == {}
    assert connector._storage_pd_engine_finished == set()
    assert connector._storage_pd_failures == {}
    assert connector._storage_pd_terminal_states == {}
    # The identifier survives, so a late status is not taken for new work.
    assert list(connector._storage_pd_returned) == ["request-1"]
    assert connector.get_finished({"request-1"}) == (set(), None)


def test_storage_pd_forgets_the_oldest_finished_requests() -> None:
    """The record of finished requests is bounded, not merely smaller."""
    # First Party
    from lmcache.integration.vllm import vllm_v1_adapter

    connector = _make_storage_pd_connector()

    class Sender:
        def send(self, status) -> None:
            return None

    _attach_notification_queue(connector, Sender())
    original = vllm_v1_adapter.STORAGE_PD_REQUEST_HISTORY
    vllm_v1_adapter.STORAGE_PD_REQUEST_HISTORY = 3
    try:
        for index in range(8):
            req_id = f"request-{index}"
            completion: Future[list[RawBlockPublicationReceipt]] = Future()
            completion.set_result(
                [RawBlockPublicationReceipt("writer", 1, 1, "digest")]
            )
            connector._storage_pd_store_futures[req_id] = completion
            assert _finish_until_released(connector, {req_id}) == {req_id}
        assert list(connector._storage_pd_returned) == [
            f"request-{i}" for i in range(5, 8)
        ]
    finally:
        vllm_v1_adapter.STORAGE_PD_REQUEST_HISTORY = original


def test_storage_pd_get_finished_reports_failure_but_releases_source_blocks() -> None:
    class Sender:
        def __init__(self) -> None:
            self.statuses: list[StoragePDStatus] = []

        def send(self, status: StoragePDStatus) -> None:
            self.statuses.append(status)

    connector = _make_storage_pd_connector()
    sender = Sender()
    _attach_notification_queue(connector, sender)
    completion: Future[list[RawBlockPublicationReceipt]] = Future()
    connector._storage_pd_store_futures["request-failed"] = completion
    completion.set_exception(OSError("write failed"))

    assert _finish_until_released(connector, {"request-failed"}) == {"request-failed"}
    assert [(s.state, s.error_text) for s in sender.statuses] == [
        ("FAILED", "write failed")
    ]
    assert connector.get_finished({"request-failed"}) == (set(), None)


def test_storage_pd_releases_a_request_it_could_not_announce() -> None:
    """Giving up on delivery still has to be a decision, and a loud one.

    The bytes are durable and the publication stands; what failed is
    telling anyone. Holding the engine's request open would not change
    that, so it is released -- and the loss is recorded, because it
    authorizes no extent reuse: the writer's lease is released on an
    acknowledgement this request will now never receive.
    """

    class DeadSender:
        def send(self, status: StoragePDStatus) -> None:
            raise ConnectionRefusedError("proxy is gone")

    connector = _make_storage_pd_connector()
    _attach_notification_queue(connector, DeadSender(), deadline_s=0.0)
    completion: Future[list[RawBlockPublicationReceipt]] = Future()
    completion.set_result([RawBlockPublicationReceipt("writer", 1, 1, "digest")])
    connector._storage_pd_store_futures["request-1"] = completion

    with _captured_adapter_logs(logging.ERROR) as records:
        assert _finish_until_released(connector, {"request-1"}) == {"request-1"}

    assert any(
        "gave up announcing request" in record.getMessage()
        and "request-1" in record.getMessage()
        for record in records
    ), "giving up on a durable publication was not reported"


def test_storage_pd_defers_an_announcement_the_queue_cannot_take() -> None:
    """A full queue postpones a request; it does not lose or release it."""
    release_peer = threading.Event()
    sending = threading.Event()

    class BlockedSender:
        def send(self, status: StoragePDStatus) -> None:
            sending.set()
            assert release_peer.wait(SETTLE_TIMEOUT_S)

    connector = _make_storage_pd_connector()
    _attach_notification_queue(connector, BlockedSender(), capacity=1)
    for req_id in ("request-1", "request-2", "request-3"):
        completion: Future[list[RawBlockPublicationReceipt]] = Future()
        completion.set_result([RawBlockPublicationReceipt("writer", 1, 1, "digest")])
        connector._storage_pd_store_futures[req_id] = completion

    assert connector.get_finished({"request-1", "request-2", "request-3"}) == (
        set(),
        None,
    )
    assert sending.wait(SETTLE_TIMEOUT_S)
    # One is in the sender and one is queued behind it; the third was
    # refused, so it is not recorded as queued and will be offered again.
    assert len(connector._storage_pd_obligations) == 3
    assert len(connector._storage_pd_engine_finished) == 3

    release_peer.set()
    released: set[str] = set()
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while len(released) < 3 and time.monotonic() < deadline:
        step, _ = connector.get_finished(set())
        released |= step
        time.sleep(0.002)
    assert released == {"request-1", "request-2", "request-3"}


def test_wait_for_save_skips_desynced_request_and_keeps_engine_alive() -> None:
    """Length mismatch must drop only the affected request's save, log a
    warning, and let ``wait_for_save`` return normally.

    Regression for https://github.com/LMCache/LMCache/issues/3318.
    """
    # lmcache's ``init_logger`` sets ``propagate = False`` on the adapter
    # logger so its records do not reach pytest's ``caplog`` (which
    # attaches to the root logger). Toggling ``propagate`` is fragile --
    # any lazy import that re-runs ``init_logger`` resets it. Attach a
    # local handler directly to the named logger instead so we capture
    # the warning regardless of how lmcache configures propagation.
    captured_records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    adapter_logger = logging.getLogger("lmcache.integration.vllm.vllm_v1_adapter")
    # ``init_logger`` sets the logger level from ``LMCACHE_LOG_LEVEL`` (default
    # INFO). If a prior import set it above WARNING, ``logger.warning`` would be
    # filtered before reaching our handler. Force WARNING for the duration of
    # the test and restore the original level in ``finally``.
    original_level = adapter_logger.level
    adapter_logger.setLevel(logging.WARNING)
    adapter_logger.addHandler(handler)
    try:
        desync_req = _make_desync_request(
            "req-desync", token_ids_len=4, slot_mapping_len=3
        )
        connector, engine = _make_connector([desync_req])

        connector.wait_for_save()

        # 1. lookup_unpin still ran (pin balance preserved)
        assert engine.unpinned == ["req-desync"]

        # 2. store was NOT called for the desynced request (save dropped)
        assert engine.store_calls == []

        # 3. A warning was emitted naming the request and both lengths
        warnings = [r for r in captured_records if r.levelno == logging.WARNING]
        assert any(
            "req-desync" in r.getMessage()
            and "slot_mapping=3" in r.getMessage()
            and "token_ids=4" in r.getMessage()
            for r in warnings
        ), (
            "Expected desync warning naming req-desync; "
            f"got {[r.getMessage() for r in warnings]}"
        )
    finally:
        adapter_logger.removeHandler(handler)
        adapter_logger.setLevel(original_level)


def test_storage_pd_worker_without_notification_config_fails_at_init() -> None:
    """A worker that owes statuses and cannot send them must not start.

    Refusing on the first completed request instead lets the engine come up,
    answer its health check and serve, which looks exactly like a healthy
    start. This drives the real setup the constructor runs, not a
    reimplementation of its decision.
    """
    connector = _make_storage_pd_connector()
    connector._storage_pd_notify_required = True
    connector._storage_pd_raw_role = "writer"
    connector._storage_pd_status_sender = None

    # A writer that must notify, with no proxy to notify.
    config = SimpleNamespace(
        pd_skip_proxy_notification=False,
        pd_proxy_host=None,
        pd_proxy_port=None,
    )
    with pytest.raises(ValueError, match="pd_proxy_host"):
        connector._init_storage_pd_notification(config, {})

    # Notification deliberately turned off: no sender is needed, and the
    # requirement is not in force either.
    connector._storage_pd_notify_required = False
    connector._init_storage_pd_notification(
        SimpleNamespace(
            pd_skip_proxy_notification=True,
            pd_proxy_host=None,
            pd_proxy_port=None,
        ),
        {},
    )
    assert connector._storage_pd_status_sender is None


def test_storage_pd_scheduler_does_not_complete_a_handoff() -> None:
    """The scheduler builds no sender by design, so it must not complete.

    Running the writer's completion path there would either demand a sender
    it should not have or record deliveries that never happened.
    """
    connector = _make_storage_pd_connector()
    connector._role = KVConnectorRole.SCHEDULER
    completion: Future[list[RawBlockPublicationReceipt]] = Future()
    completion.set_result([RawBlockPublicationReceipt("writer", 1, 1, "digest")])
    connector._storage_pd_store_futures["request-1"] = completion

    assert connector.get_finished({"request-1"}) == (None, None)
    # Nothing was released and nothing was retired.
    assert "request-1" in connector._storage_pd_store_futures
