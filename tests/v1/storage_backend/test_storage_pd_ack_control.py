# SPDX-License-Identifier: Apache-2.0
"""Drive the acknowledgement exchange over a real loopback socket.

These are protocol tests, not serving tests: everything here runs on
127.0.0.1 with no device, no model and no GPU, because what is under test
is who is allowed to decide that an extent is reclaimable and what counts
as having decided it.

The properties they hold the exchange to are the ones a lost message would
otherwise break quietly. A consumer must not retire an obligation on a
successful send; a writer must not answer before it has acted; a duplicate
must free nothing twice; a message that does not correlate with the
question being asked must be discarded rather than accepted as its answer.
"""

# Future
from __future__ import annotations

# Standard
from typing import Optional
import socket as socketlib
import threading
import time

# Third Party
import msgspec
import pytest
import zmq

# First Party
from lmcache.v1.storage_backend.storage_pd_ack import (
    ACK_ALREADY_APPLIED,
    ACK_APPLIED,
    ACK_REJECTED,
    ACK_UNRESOLVED,
    StoragePDAckClient,
    StoragePDAckReply,
    StoragePDAckRequest,
    StoragePDAckServer,
)
from lmcache.v1.storage_backend.storage_pd_protocol import StoragePDReadAck

LOOPBACK = "127.0.0.1"


def _free_port() -> int:
    with socketlib.socket(socketlib.AF_INET, socketlib.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return int(probe.getsockname()[1])


def _ack(req_id: str = "request-1", **overrides) -> StoragePDReadAck:
    fields = {
        "req_id": req_id,
        "producer_instance_id": "producer-1",
        "consumer_instance_id": "consumer-1",
        "tp_rank": 0,
        "writer_epoch": "writer-1",
        "checkpoint_seq": 7,
        "manifest_digest": "digest",
    }
    fields.update(overrides)
    return StoragePDReadAck(**fields)  # type: ignore[arg-type]


class _Writer:
    """A writer that releases a hold exactly once per request."""

    def __init__(self) -> None:
        self.held: dict[str, int] = {}
        self.applied: list[str] = []
        self.seen: list[StoragePDAckRequest] = []
        self.lock = threading.Lock()
        self.reply_after_apply = True
        self.raise_in_handler = False

    def hold(self, req_id: str, count: int = 1) -> None:
        with self.lock:
            self.held[req_id] = self.held.get(req_id, 0) + count

    def handle(self, request: StoragePDAckRequest) -> tuple[str, str]:
        with self.lock:
            self.seen.append(request)
            if self.raise_in_handler:
                raise OSError("the writer could not reach its device")
            ack = request.ack
            if ack.producer_instance_id != "producer-1":
                return ACK_REJECTED, "not this writer"
            if ack.consumer_instance_id != "consumer-1":
                return ACK_REJECTED, "not the bound consumer"
            if ack.manifest_digest != "digest":
                return ACK_REJECTED, "receipt mismatch"
            if ack.req_id in self.applied:
                return ACK_ALREADY_APPLIED, ""
            if self.held.get(ack.req_id, 0) <= 0:
                return ACK_REJECTED, "no such hold"
            self.held[ack.req_id] -= 1
            self.applied.append(ack.req_id)
        if not self.reply_after_apply:
            # Applied, then the answer is lost. The consumer must retry and
            # must not be told twice that it released something.
            raise _DroppedReply()
        return ACK_APPLIED, ""


class _DroppedReply(Exception):
    """Raised after a successful release to model a lost reply."""


@pytest.fixture
def writer():
    return _Writer()


@pytest.fixture
def server(writer):
    port = _free_port()
    instance = StoragePDAckServer(
        writer.handle,
        bind_host=LOOPBACK,
        port=port,
        advertise_host=LOOPBACK,
        recv_timeout_ms=50,
    )
    yield instance
    instance.close(timeout_s=5.0)


@pytest.fixture
def client():
    instance = StoragePDAckClient(
        attempt_timeout_ms=300,
        retry_interval_s=0.05,
        max_live_obligations=4,
        poll_interval_s=0.01,
    )
    yield instance
    instance.close(timeout_s=5.0)


def _settled(obligation, timeout: float = 10.0) -> Optional[str]:
    return obligation.wait(timeout=timeout)


def test_a_writer_applies_and_the_consumer_hears_it(server, client, writer):
    """The ordinary case: one request, one release, one validated answer."""
    writer.hold("request-1")
    obligation = client.owe(
        _ack(),
        endpoint=server.endpoint,
        session_id="session-1",
        deadline_s=10.0,
    )
    assert obligation is not None
    assert _settled(obligation) == ACK_APPLIED
    assert writer.held["request-1"] == 0
    assert writer.applied == ["request-1"]


def test_a_dropped_reply_after_a_release_frees_nothing_twice(server, client, writer):
    """The writer released the hold and its answer was lost.

    The consumer has no way to know the difference between that and a
    request that never arrived, so it asks again. The second answer must
    say the hold was already applied, and the writer must not release a
    second reference.
    """
    writer.hold("request-1")
    writer.reply_after_apply = False
    obligation = client.owe(
        _ack(),
        endpoint=server.endpoint,
        session_id="session-1",
        deadline_s=10.0,
    )
    assert obligation is not None

    # Wait until the release has happened, then let the writer answer.
    deadline = time.monotonic() + 10.0
    while not writer.applied and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.applied == ["request-1"]
    assert not obligation.settled, "a lost reply is not an answer"
    writer.reply_after_apply = True

    assert _settled(obligation) == ACK_ALREADY_APPLIED
    assert writer.held["request-1"] == 0
    assert writer.applied == ["request-1"], "the release must have run once"
    assert obligation.attempts >= 2


def test_an_unreachable_writer_leaves_the_obligation_owed(client, writer):
    """A send that cannot even be made is not an acknowledgement.

    Nothing is listening, so every attempt fails. The obligation stays owed
    until its own deadline and then reports that it was never answered --
    it never reports that the hold is gone.
    """
    obligation = client.owe(
        _ack(),
        endpoint=f"{LOOPBACK}:{_free_port()}",
        session_id="session-1",
        deadline_s=0.5,
    )
    assert obligation is not None
    assert _settled(obligation) == ACK_UNRESOLVED
    assert obligation.attempts >= 1
    assert writer.applied == []


def test_a_writer_that_arrives_late_is_still_reached(client, writer):
    """The consumer may finish reading before the writer is listening.

    The obligation is owned by a background worker rather than by the
    restore that created it, so it keeps trying with no further serving
    request of any kind.
    """
    port = _free_port()
    writer.hold("request-1")
    obligation = client.owe(
        _ack(),
        endpoint=f"{LOOPBACK}:{port}",
        session_id="session-1",
        deadline_s=15.0,
    )
    assert obligation is not None

    deadline = time.monotonic() + 5.0
    while obligation.attempts < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert obligation.attempts >= 1
    assert not obligation.settled

    server = StoragePDAckServer(
        writer.handle,
        bind_host=LOOPBACK,
        port=port,
        advertise_host=LOOPBACK,
        recv_timeout_ms=50,
    )
    try:
        assert _settled(obligation) == ACK_APPLIED
    finally:
        server.close(timeout_s=5.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"producer_instance_id": "someone-else"},
        {"consumer_instance_id": "a-restarted-consumer"},
        {"manifest_digest": "someone-elses-digest"},
        {"req_id": "a-request-that-was-never-published"},
    ],
    ids=["wrong-producer", "wrong-consumer", "wrong-receipt", "unknown-request"],
)
def test_a_wrong_identity_releases_nothing(server, client, writer, overrides):
    """The writer checks every field, and a rejection is final.

    A consumer told "rejected" stops offering the message: this writer will
    never apply it, and retrying forever would keep a control worker busy
    over something that can only be answered one way.
    """
    writer.hold("request-1")
    obligation = client.owe(
        _ack(**overrides),
        endpoint=server.endpoint,
        session_id="session-1",
        deadline_s=10.0,
    )
    assert obligation is not None
    assert _settled(obligation) == ACK_REJECTED
    assert writer.applied == []
    assert writer.held["request-1"] == 1


def test_two_requests_sharing_a_key_have_two_holds(server, client, writer):
    """One acknowledgement releases one hold, not the key.

    Two requests whose manifests name the same extent each hold a
    reference. Treating the key as the unit would let the first
    acknowledgement free the extent while the second reader is still
    reading it.
    """
    writer.hold("request-1")
    writer.hold("request-2")
    first = client.owe(
        _ack("request-1"),
        endpoint=server.endpoint,
        session_id="session-1",
        deadline_s=10.0,
    )
    assert first is not None
    assert _settled(first) == ACK_APPLIED

    assert writer.held == {"request-1": 0, "request-2": 1}

    second = client.owe(
        _ack("request-2"),
        endpoint=server.endpoint,
        session_id="session-1",
        deadline_s=10.0,
    )
    assert second is not None
    assert _settled(second) == ACK_APPLIED
    assert writer.held == {"request-1": 0, "request-2": 0}


def test_saturation_refuses_new_work_and_recovers_without_a_next_request(
    client, writer
):
    """A full outbox stops admitting rather than abandoning a hold.

    Evicting the oldest obligation to make room is how a writer ends up
    holding extents nobody is responsible for. The bound refuses instead,
    and the refusal is the caller's signal to stop publishing.

    Recovery then has to happen with no further request arriving, which is
    what a background owner is for.
    """
    port = _free_port()
    endpoint = f"{LOOPBACK}:{port}"
    owed = []
    for index in range(4):
        writer.hold(f"request-{index}")
        obligation = client.owe(
            _ack(f"request-{index}"),
            endpoint=endpoint,
            session_id="session-1",
            deadline_s=20.0,
        )
        assert obligation is not None
        owed.append(obligation)

    writer.hold("one-too-many")
    assert (
        client.owe(
            _ack("one-too-many"),
            endpoint=endpoint,
            session_id="session-1",
            deadline_s=20.0,
        )
        is None
    )
    assert client.live_count() == 4

    server = StoragePDAckServer(
        writer.handle,
        bind_host=LOOPBACK,
        port=port,
        advertise_host=LOOPBACK,
        recv_timeout_ms=50,
    )
    try:
        for obligation in owed:
            assert _settled(obligation, timeout=20.0) == ACK_APPLIED
    finally:
        server.close(timeout_s=5.0)

    assert sorted(writer.applied) == [f"request-{index}" for index in range(4)]
    # And the room is back, without anything having asked.
    deadline = time.monotonic() + 5.0
    while client.live_count() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.live_count() == 0
    again = client.owe(
        _ack("one-too-many"),
        endpoint=endpoint,
        session_id="session-1",
        deadline_s=5.0,
    )
    assert again is not None


def test_a_writer_that_cannot_say_is_not_a_writer_that_applied(server, client, writer):
    """A handler that raised has established nothing.

    The reply has to carry that, because the alternative is a consumer
    retiring an obligation for a release that never happened.
    """
    writer.hold("request-1")
    writer.raise_in_handler = True
    obligation = client.owe(
        _ack(),
        endpoint=server.endpoint,
        session_id="session-1",
        deadline_s=1.0,
    )
    assert obligation is not None
    assert _settled(obligation) == ACK_UNRESOLVED
    assert writer.applied == []
    assert writer.held["request-1"] == 1
    assert len(writer.seen) >= 2, "an unresolved answer is not a final one"


def test_an_advertised_wildcard_is_refused(writer):
    """A bind wildcard is not an address a consumer can reach.

    Publishing "0.0.0.0" as the place to reply tells every consumer to
    connect to itself. The bind address and the advertised address are
    separate, and the advertised one has to be routable.
    """
    with pytest.raises(ValueError, match="not an address"):
        StoragePDAckServer(
            writer.handle,
            bind_host="0.0.0.0",
            port=_free_port(),
        )


def test_a_reply_about_something_else_is_not_an_answer(client):
    """A well-formed reply that answers a different question is discarded.

    A reply carrying another attempt's correlation is what a socket out of
    step with its own alternation delivers. Accepting it would settle this
    obligation on a writer's statement about some other hold -- and if that
    statement is APPLIED, the consumer stops asking about an extent still
    held.
    """
    port = _free_port()
    replies: list[str] = []

    def _answer_the_wrong_question() -> None:
        context: zmq.Context = zmq.Context.instance()
        rep = context.socket(zmq.REP)
        rep.setsockopt(zmq.LINGER, 0)
        rep.setsockopt(zmq.RCVTIMEO, 200)
        rep.bind(f"tcp://{LOOPBACK}:{port}")
        try:
            while len(replies) < 2:
                try:
                    raw = rep.recv()
                except zmq.Again:
                    continue
                request = msgspec.msgpack.decode(raw, type=StoragePDAckRequest)
                if not replies:
                    # Correct in every field but the correlation.
                    replies.append("foreign")
                    rep.send(
                        msgspec.msgpack.encode(
                            StoragePDAckReply(
                                outcome=ACK_APPLIED,
                                req_id=request.ack.req_id,
                                producer_instance_id=(request.ack.producer_instance_id),
                                consumer_instance_id=(request.ack.consumer_instance_id),
                                nonce="a-nonce-from-another-attempt",
                            )
                        )
                    )
                    continue
                replies.append("correlated")
                rep.send(
                    msgspec.msgpack.encode(
                        StoragePDAckReply(
                            outcome=ACK_APPLIED,
                            req_id=request.ack.req_id,
                            producer_instance_id=request.ack.producer_instance_id,
                            consumer_instance_id=request.ack.consumer_instance_id,
                            nonce=request.nonce,
                        )
                    )
                )
        finally:
            rep.close(linger=0)

    responder = threading.Thread(target=_answer_the_wrong_question, daemon=True)
    responder.start()
    try:
        obligation = client.owe(
            _ack(),
            endpoint=f"{LOOPBACK}:{port}",
            session_id="session-1",
            deadline_s=15.0,
        )
        assert obligation is not None
        assert _settled(obligation) == ACK_APPLIED
        assert replies == ["foreign", "correlated"]
        assert obligation.attempts >= 2, "the foreign reply must not have settled it"
    finally:
        responder.join(timeout=5)


def test_an_unintelligible_request_still_gets_an_answer(server, writer):
    """A reply socket owes an answer for every message it accepted.

    Skipping the reply leaves the socket out of step with its own
    alternation: the next receive raises, the server thread exits, and
    every later acknowledgement times out against a writer that is up and
    listening. One malformed message would take the whole channel down.
    """
    writer.hold("request-1")
    context: zmq.Context = zmq.Context.instance()
    req = context.socket(zmq.REQ)
    req.setsockopt(zmq.LINGER, 0)
    req.setsockopt(zmq.RCVTIMEO, 3000)
    req.setsockopt(zmq.SNDTIMEO, 3000)
    req.connect(f"tcp://{server.endpoint}")
    try:
        req.send(b"this is not a msgpack acknowledgement")
        first = msgspec.msgpack.decode(req.recv(), type=StoragePDAckReply)
        assert first.outcome == ACK_REJECTED
        assert "undecodable" in first.reason
    finally:
        req.close(linger=0)

    # And the channel still works for a real one.
    client = StoragePDAckClient(
        attempt_timeout_ms=1000,
        retry_interval_s=0.05,
        max_live_obligations=2,
        poll_interval_s=0.01,
    )
    try:
        obligation = client.owe(
            _ack(),
            endpoint=server.endpoint,
            session_id="session-1",
            deadline_s=10.0,
        )
        assert obligation is not None
        assert _settled(obligation) == ACK_APPLIED
    finally:
        client.close(timeout_s=5.0)
