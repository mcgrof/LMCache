# SPDX-License-Identifier: Apache-2.0
"""One obligation, one deadline, one terminal settlement.

Announcement progress is tracked apart from the publication it announces:
whether a request is READY or FAILED is decided before a message reaches the
queue and is never revised here. These check the three rules the queue exists
to enforce, and the two ways an earlier version broke them -- a message that
expired while queued being sent anyway, and one settling twice with
contradictory outcomes.
"""

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
    StoragePDObligation,
    StoragePDStatus,
    StoragePDStatusSender,
)

SETTLE_TIMEOUT_S = 10.0


class _RecordingSender:
    """A sender whose reachability, pace and bookkeeping the test decides."""

    def __init__(
        self,
        *,
        fail_first: int = 0,
        fail_always: bool = False,
        gate: threading.Event | None = None,
    ) -> None:
        self.attempts = 0
        self.sent: list[StoragePDStatus] = []
        self.timeouts: list[float | None] = []
        self.closed_with: list[float | None] = []
        self._fail_first = fail_first
        self._fail_always = fail_always
        self._gate = gate
        self.stops_cleanly = True
        self._lock = threading.Lock()

    def send(self, message: StoragePDStatus, *, timeout_s: float | None = None) -> None:
        with self._lock:
            self.attempts += 1
            attempt = self.attempts
            self.timeouts.append(timeout_s)
        if self._gate is not None:
            assert self._gate.wait(SETTLE_TIMEOUT_S)
        if self._fail_always or attempt <= self._fail_first:
            raise ConnectionRefusedError("proxy is not accepting statuses")
        with self._lock:
            self.sent.append(message)

    def close(self, timeout_s: float | None = None) -> bool:
        self.closed_with.append(timeout_s)
        return self.stops_cleanly


class _Clock:
    """A monotonic clock the test advances, for deadlines without waiting."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


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
    """Wait for at least one outcome to settle, or fail the test."""
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while time.monotonic() < deadline:
        settled = queue.poll()
        if settled:
            return settled
        time.sleep(0.002)
    pytest.fail("no outcome settled")


def test_offer_returns_before_the_peer_is_reached() -> None:
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender)
    try:
        assert queue.offer(queue.obligation("request", _status("request")))
        # Nothing has settled: the sender is still inside its first attempt.
        # The caller got control back anyway, which is the point.
        assert queue.poll() == []
        assert queue.pending_count() == 1
        gate.set()
        assert _settle(queue) == [
            StoragePDDelivery(key="request", state="LOCALLY_SENT", detail="")
        ]
        assert queue.pending_count() == 0
    finally:
        gate.set()
        queue.close()


def test_a_failed_attempt_is_retried_until_it_lands() -> None:
    sender = _RecordingSender(fail_first=2)
    queue = _queue(sender)
    try:
        assert queue.offer(queue.obligation("request", _status("request")))
        assert _settle(queue) == [
            StoragePDDelivery(key="request", state="LOCALLY_SENT", detail="")
        ]
        assert sender.attempts >= 3
        assert [status.req_id for status in sender.sent] == ["request"]
    finally:
        queue.close()


def test_an_unavailable_proxy_is_abandoned_at_the_deadline() -> None:
    sender = _RecordingSender(fail_always=True)
    queue = _queue(sender, deadline_s=0.05)
    try:
        assert queue.offer(queue.obligation("request", _status("request")))
        settled = _settle(queue)
        assert [(item.key, item.state) for item in settled] == [
            ("request", "ABANDONED")
        ]
        # It was attempted, so the reason names the transport rather than
        # the lifetime.
        assert "ConnectionRefusedError" in settled[0].detail
        assert sender.attempts >= 1
        assert queue.pending_count() == 0
    finally:
        queue.close()


def test_an_obligation_with_no_lifetime_is_abandoned_unattempted() -> None:
    """A deadline of zero is a refusal, not a single free attempt.

    It is also not admitted: offering reports that nothing was taken on,
    and the obligation settles on the spot. The deadline belongs to the
    obligation rather than to an attempt, so whether it can be reached must
    not depend on there having been room.
    """
    sender = _RecordingSender()
    queue = _queue(sender, deadline_s=0.0)
    try:
        assert queue.offer(queue.obligation("request", _status("request"))) is False
        assert queue.pending_count() == 0
        settled = _settle(queue)
        assert [(item.key, item.state) for item in settled] == [
            ("request", "ABANDONED")
        ]
        assert "deadline" in settled[0].detail
        assert sender.attempts == 0
    finally:
        queue.close()


def test_an_obligation_refused_until_its_deadline_still_ends() -> None:
    """A full queue must not leave a caller re-offering something forever.

    The obligation was never admitted, so no worker will ever settle it.
    Reaching its own deadline while being refused is the only end it has,
    and it has to produce the same record as any other abandonment.
    """
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender, capacity=1, retry_interval_s=0.0)
    far = time.monotonic() + 3600.0
    try:
        # One obligation occupies the worker and one fills the queue, both
        # with deadlines far enough away that neither of them is what ends.
        held = StoragePDObligation("held", _status("held"), report=True, deadline=far)
        assert queue.offer(held)
        while sender.attempts == 0:
            time.sleep(0.002)
        filler = StoragePDObligation(
            "filler", _status("filler"), report=True, deadline=far
        )
        assert queue.offer(filler)

        refused = StoragePDObligation(
            "refused",
            _status("refused"),
            report=True,
            deadline=time.monotonic() + 0.2,
        )
        assert queue.offer(refused) is False, "the queue has no room"

        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while not refused.settled and time.monotonic() < deadline:
            assert queue.offer(refused) is False
            time.sleep(0.01)
        assert refused.settled, "it was refused until its deadline and never ended"

        settled = {item.key: item for itemin_ in () for item in ()}
        settled = {item.key: item for item in queue.poll()}
        assert settled["refused"].state == "ABANDONED"
        assert "never admitted" in settled["refused"].detail
    finally:
        gate.set()
        queue.close()


def test_an_obligation_that_expired_while_queued_is_never_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deadline governs the attempt, not just the retry after a failure.

    An obligation whose lifetime ran out while it waited for its turn must
    not be attempted: a send that succeeded now would report an announcement
    for something nobody is waiting on any more.
    """
    clock = _Clock()
    monkeypatch.setattr("lmcache.v1.storage_backend.storage_pd_protocol.time", clock)
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender, capacity=4, deadline_s=1.0, retry_interval_s=0.0)
    try:
        held = queue.obligation("held", _status("held"))
        assert queue.offer(held)
        while sender.attempts == 0:
            time.sleep(0.002)
        # Queued behind a sender that is not moving, and its lifetime runs
        # out there rather than in an attempt.
        queued = queue.obligation("queued", _status("queued"))
        assert queue.offer(queued)
        clock.now = 2.0
        gate.set()

        outcomes: dict[str, str] = {}
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while len(outcomes) < 2 and time.monotonic() < deadline:
            for item in queue.poll():
                outcomes[item.key] = item.state
            time.sleep(0.002)
        assert outcomes == {"held": "LOCALLY_SENT", "queued": "ABANDONED"}
        # The expired one was never handed to the sender at all.
        assert [status.req_id for status in sender.sent] == ["held"]
        assert sender.attempts == 1
    finally:
        gate.set()
        queue.close()


def test_an_attempt_is_bounded_by_what_is_left_of_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A send may spend only the time the obligation has left."""
    clock = _Clock()
    monkeypatch.setattr("lmcache.v1.storage_backend.storage_pd_protocol.time", clock)
    sender = _RecordingSender()
    queue = _queue(sender, deadline_s=30.0)
    try:
        obligation = queue.obligation("request", _status("request"))
        clock.now = 20.0
        assert queue.offer(obligation)
        assert _settle(queue)[0].state == "LOCALLY_SENT"
        # Ten seconds of a thirty-second lifetime remain, not thirty.
        assert sender.timeouts == [10.0]
    finally:
        queue.close()


def test_shutdown_and_a_late_attempt_do_not_both_settle_one_obligation() -> None:
    """Whichever outcome arrives first is the one that stands.

    Shutdown gives up on an obligation whose attempt has not returned. When
    that attempt then succeeds, reporting it would contradict an abandonment
    the caller has already acted on.
    """
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender, capacity=1)
    obligation = queue.obligation("request", _status("request"))
    assert queue.offer(obligation)
    while sender.attempts == 0:
        time.sleep(0.002)

    # Shut down without waiting for the attempt in progress.
    queue.close(timeout_s=0.0)
    assert [(item.key, item.state) for item in queue.poll()] == [
        ("request", "ABANDONED")
    ]

    # Now let the send finish. It succeeded, and it changes nothing.
    gate.set()
    queue._worker.join(timeout=SETTLE_TIMEOUT_S)
    assert queue.poll() == []
    assert queue.pending_count() == 0


def test_shutdown_settles_everything_it_accepted() -> None:
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender, capacity=8)
    assert queue.offer(queue.obligation("held", _status("held")))
    while sender.attempts == 0:
        time.sleep(0.002)
    assert queue.offer(queue.obligation("queued", _status("queued")))
    gate.set()

    queue.close()

    # Both keys have a definite outcome: none may simply disappear, or a
    # caller waiting to release a request would wait for a message that is
    # no longer anybody's to send.
    assert {item.key for item in queue.poll()} == {"held", "queued"}
    assert queue.pending_count() == 0
    assert not queue.offer(queue.obligation("late", _status("late")))


def test_shutdown_closes_the_sender_within_one_budget() -> None:
    """The queue owns the sender, so teardown has one budget and one owner."""
    sender = _RecordingSender()
    queue = _queue(sender)
    queue.close(timeout_s=5.0)
    assert len(sender.closed_with) == 1
    budget = sender.closed_with[0]
    assert budget is not None and 0.0 <= budget <= 5.0


def test_a_full_queue_refuses_rather_than_growing() -> None:
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender, capacity=2)
    try:
        # The worker takes the first obligation out of the queue and blocks
        # in the sender, so capacity applies to the two behind it.
        assert queue.offer(queue.obligation("held", _status("held")))
        while sender.attempts == 0:
            time.sleep(0.002)
        assert queue.offer(queue.obligation("first", _status("first")))
        assert queue.offer(queue.obligation("second", _status("second")))

        third = queue.obligation("third", _status("third"))
        assert not queue.offer(third)
        assert queue.pending_count() == 3
        gate.set()
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while queue.pending_count() and time.monotonic() < deadline:
            time.sleep(0.002)
        # The same obligation is accepted once there is room; the caller
        # never had to build another one.
        assert queue.offer(third)
    finally:
        gate.set()
        queue.close()


def test_a_refused_obligation_keeps_the_deadline_it_was_created_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting for capacity spends the lifetime; it does not postpone it."""
    clock = _Clock()
    monkeypatch.setattr("lmcache.v1.storage_backend.storage_pd_protocol.time", clock)
    sender = _RecordingSender()
    queue = _queue(sender, deadline_s=5.0)
    try:
        obligation = queue.obligation("request", _status("request"))
        assert obligation.deadline == 5.0
        clock.now = 4.0
        # Re-offering does not touch it, however long it waited.
        queue.offer(obligation)
        assert obligation.deadline == 5.0
    finally:
        queue.close()


def test_a_duplicate_key_is_not_announced_twice() -> None:
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender)
    try:
        assert queue.offer(queue.obligation("request", _status("request")))
        assert not queue.offer(queue.obligation("request", _status("request")))
        assert queue.pending_count() == 1
        gate.set()
        assert len(_settle(queue)) == 1
        assert len(sender.sent) == 1
    finally:
        gate.set()
        queue.close()


def test_an_unreported_obligation_goes_to_the_callback_instead() -> None:
    seen: list[StoragePDDelivery] = []
    sender = _RecordingSender(fail_always=True)
    queue = _queue(sender, deadline_s=0.0, on_unreported=seen.append)
    try:
        assert (
            queue.offer(
                queue.obligation("ack:request", _status("request"), report=False)
            )
            is False
        )
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


def test_an_unreported_obligation_abandoned_at_shutdown_reaches_the_callback() -> None:
    seen: list[StoragePDDelivery] = []
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender, capacity=4, on_unreported=seen.append)
    assert queue.offer(queue.obligation("held", _status("held")))
    while sender.attempts == 0:
        time.sleep(0.002)
    assert queue.offer(queue.obligation("ack:queued", _status("queued"), report=False))

    queue.close(timeout_s=0.0)
    gate.set()
    queue._worker.join(timeout=SETTLE_TIMEOUT_S)

    assert [(item.key, item.state) for item in seen] == [("ack:queued", "ABANDONED")]


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


def test_close_does_not_call_an_abandoned_worker_stopped() -> None:
    """A budget running out is not a thread stopping.

    A send stuck in the transport outlives the shutdown budget, and close
    abandons it rather than waiting. That thread is still alive and still
    holding the socket, so reporting success here would let a caller
    destroy what it is using.
    """
    gate = threading.Event()
    sender = _RecordingSender(gate=gate)
    queue = _queue(sender, capacity=2)
    try:
        assert queue.offer(queue.obligation("held", _status("held")))
        while sender.attempts == 0:
            time.sleep(0.002)

        assert queue.close(timeout_s=0.1) is False
    finally:
        gate.set()


def test_close_reports_a_clean_stop_when_there_is_one() -> None:
    """The refusal above has to be an observation, not a stuck answer."""
    sender = _RecordingSender()
    queue = _queue(sender, capacity=2)
    assert queue.offer(queue.obligation("request", _status("request")))
    _settle(queue)
    assert queue.close(timeout_s=5.0) is True


def test_close_carries_the_senders_own_answer() -> None:
    """The sender owns the socket, so its answer is part of this one.

    A queue worker that stopped says nothing about the thread holding the
    transport, and that thread is the one still touching a socket a caller
    may be about to destroy.
    """
    sender = _RecordingSender()
    sender.stops_cleanly = False
    queue = _queue(sender, capacity=2)
    assert queue.offer(queue.obligation("request", _status("request")))
    _settle(queue)
    assert queue.close(timeout_s=5.0) is False
