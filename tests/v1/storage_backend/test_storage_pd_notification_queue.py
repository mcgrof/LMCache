# SPDX-License-Identifier: Apache-2.0
"""Delivery progress is tracked apart from the publication it announces."""

# Standard
from typing import Any, cast
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.storage_pd_protocol import (
    StoragePDDelivery,
    StoragePDNotificationQueue,
    StoragePDStatus,
    StoragePDStatusSender,
)

SETTLE_TIMEOUT_S = 10.0


class _RecordingSender:
    """A sender whose reachability and pace the test decides."""

    def __init__(
        self,
        *,
        fail_first: int = 0,
        fail_always: bool = False,
        gate: threading.Event | None = None,
    ) -> None:
        self.attempts = 0
        self.sent: list[StoragePDStatus] = []
        self._fail_first = fail_first
        self._fail_always = fail_always
        self._gate = gate
        self._lock = threading.Lock()

    def send(self, message: StoragePDStatus) -> None:
        with self._lock:
            self.attempts += 1
            attempt = self.attempts
        if self._gate is not None:
            assert self._gate.wait(SETTLE_TIMEOUT_S)
        if self._fail_always or attempt <= self._fail_first:
            raise ConnectionRefusedError("proxy is not accepting statuses")
        with self._lock:
            self.sent.append(message)


def _status(req_id: str) -> StoragePDStatus:
    return StoragePDStatus(
        req_id=req_id,
        producer_instance_id="producer",
        tp_rank=0,
        state="READY",
    )


def _queue(sender: _RecordingSender, **kwargs: Any) -> StoragePDNotificationQueue:
    kwargs.setdefault("retry_interval_s", 0.001)
    kwargs.setdefault("deadline_s", SETTLE_TIMEOUT_S)
    return StoragePDNotificationQueue(
        cast(StoragePDStatusSender, sender),
        **kwargs,
    )


def _settle(queue: StoragePDNotificationQueue) -> list[StoragePDDelivery]:
    """Wait for at least one delivery to settle, or fail the test."""
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while time.monotonic() < deadline:
        settled = queue.poll()
        if settled:
            return settled
        time.sleep(0.002)
    pytest.fail("no delivery settled")


def test_enqueue_returns_before_the_peer_is_reached() -> None:
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender)
    try:
        assert queue.enqueue("request", _status("request"))
        # Nothing has been delivered: the sender is still inside its first
        # attempt. The caller got control back anyway, which is the point.
        assert queue.poll() == []
        assert queue.pending_count() == 1
        gate.set()
        assert _settle(queue) == [
            StoragePDDelivery(key="request", state="DELIVERED", detail="")
        ]
        assert queue.pending_count() == 0
    finally:
        gate.set()
        queue.close()


def test_a_failed_first_send_is_retried_until_it_lands() -> None:
    sender = _RecordingSender(fail_first=2)
    queue = _queue(sender)
    try:
        assert queue.enqueue("request", _status("request"))
        assert _settle(queue) == [
            StoragePDDelivery(key="request", state="DELIVERED", detail="")
        ]
        assert sender.attempts >= 3
        assert [status.req_id for status in sender.sent] == ["request"]
    finally:
        queue.close()


def test_an_unavailable_proxy_is_abandoned_at_the_deadline() -> None:
    sender = _RecordingSender(fail_always=True)
    queue = _queue(sender, deadline_s=0.0)
    try:
        assert queue.enqueue("request", _status("request"))
        settled = _settle(queue)
        assert [(item.key, item.state) for item in settled] == [
            ("request", "ABANDONED")
        ]
        assert "ConnectionRefusedError" in settled[0].detail
        assert queue.pending_count() == 0
    finally:
        queue.close()


def test_a_full_queue_refuses_rather_than_growing() -> None:
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender, capacity=2)
    try:
        # The worker takes the first message out of the queue and blocks in
        # the sender, so capacity applies to the two behind it.
        assert queue.enqueue("held", _status("held"))
        while sender.attempts == 0:
            time.sleep(0.002)
        assert queue.enqueue("first", _status("first"))
        assert queue.enqueue("second", _status("second"))

        assert not queue.enqueue("third", _status("third"))
        # A refusal takes nothing over, so the caller may offer it again.
        assert queue.pending_count() == 3
        gate.set()
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while queue.pending_count() and time.monotonic() < deadline:
            time.sleep(0.002)
        assert queue.enqueue("third", _status("third"))
    finally:
        gate.set()
        queue.close()


def test_a_duplicate_key_is_not_announced_twice() -> None:
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender)
    try:
        assert queue.enqueue("request", _status("request"))
        assert not queue.enqueue("request", _status("request"))
        assert queue.pending_count() == 1
        gate.set()
        assert len(_settle(queue)) == 1
        assert len(sender.sent) == 1
    finally:
        gate.set()
        queue.close()


def test_shutdown_settles_everything_it_accepted() -> None:
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender, capacity=8)
    assert queue.enqueue("held", _status("held"))
    while sender.attempts == 0:
        time.sleep(0.002)
    assert queue.enqueue("queued", _status("queued"))
    gate.set()

    queue.close()

    # Both keys have a definite outcome: none may simply disappear, or a
    # caller waiting to release a request would wait for a message that is
    # no longer anybody's to send.
    assert {item.key for item in queue.poll()} == {"held", "queued"}
    assert queue.pending_count() == 0
    # A closed queue accepts nothing further.
    assert not queue.enqueue("late", _status("late"))


def test_an_unreported_message_is_handed_to_the_callback_instead() -> None:
    seen: list[StoragePDDelivery] = []
    sender = _RecordingSender(fail_always=True)
    queue = _queue(sender, deadline_s=0.0, on_unreported=seen.append)
    try:
        assert queue.enqueue("ack:request", _status("request"), report=False)
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while not seen and time.monotonic() < deadline:
            time.sleep(0.002)
        assert [(item.key, item.state) for item in seen] == [
            ("ack:request", "ABANDONED")
        ]
        # It is not also retained, which is what keeps a caller that never
        # polls from accumulating outcomes for the life of the process.
        assert queue.poll() == []
        assert queue.pending_count() == 0
    finally:
        queue.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity": 0},
        {"retry_interval_s": -1.0},
        {"deadline_s": -1.0},
    ],
)
def test_an_unusable_policy_is_refused_at_construction(kwargs: Any) -> None:
    with pytest.raises(ValueError):
        _queue(_RecordingSender(), **kwargs)
