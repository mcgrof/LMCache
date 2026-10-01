# SPDX-License-Identifier: Apache-2.0
"""Drive the READY-to-acknowledgement chain with the real builders.

The parts of this handoff were each covered on their own and still did not
work together: a producer named itself one way when it announced a
publication and expected another name back, so an ordinary acknowledgement
was refused by the very writer that had asked for it. A test that writes
both identities by hand cannot see that, because the hand is what makes
them agree.

So nothing here spells an identity out. A writer mints its epoch, the
receipt carries it, :meth:`StoragePDStatus.ready` puts it on the wire,
:meth:`StoragePDReadAck.for_status` reads it back off, and the backend's
own handler compares what arrived against what the writer is. Every string
that has to match travels that whole path, and a test fails if any step
substitutes a different notion of who the producer is.

Storage is inert: there is no device, model or GPU here, because what is
under test is correlation.
"""

# Future
from __future__ import annotations

# Standard
import uuid

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)
from lmcache.v1.storage_backend.raw_block import (
    RawBlockPDRequestTracker,
    RawBlockPublicationReceipt,
)
from lmcache.v1.storage_backend.raw_block.core import RawBlockCore
from lmcache.v1.storage_backend.storage_pd_ack import (
    ACK_ALREADY_APPLIED,
    ACK_APPLIED,
    ACK_REJECTED,
    StoragePDAckRequest,
    StoragePDClaimAnswer,
    StoragePDClaimRequest,
    StoragePDUnreadAnswer,
    StoragePDUnreadRequest,
)
from lmcache.v1.storage_backend.storage_pd_protocol import (
    STORAGE_PD_INCARNATION,
    StoragePDReadAck,
    StoragePDStatus,
)

SESSION = "pd-session-under-test"


class _InertWriterCore:
    """A writer's identity and lock bookkeeping, with no device behind it.

    The epoch accessor is the core's own, so this cannot agree with the
    production path by restating it.
    """

    writer_epoch = RawBlockCore.writer_epoch
    role = "writer"

    def __init__(self) -> None:
        self._writer_epoch = str(uuid.uuid4())
        self._seq = 0
        self.locked: list[str] = []

    def publish_request(self, encoded_keys: list[str]) -> RawBlockPublicationReceipt:
        self._seq += 1
        return RawBlockPublicationReceipt(
            writer_epoch=self.writer_epoch,
            checkpoint_seq=self._seq,
            key_count=len(encoded_keys),
            manifest_digest=f"digest-{self._seq}",
            namespace_identity="namespace-under-test",
        )

    def get_metadata_prefix(
        self,
        encoded_keys: list[str],
        *,
        lock: bool = False,
    ) -> list[object]:
        assert lock
        self.locked.extend(encoded_keys)
        return [object() for _ in encoded_keys]

    def unlock_many(self, encoded_keys: list[str]) -> None:
        for encoded_key in encoded_keys:
            self.locked.remove(encoded_key)


class _Producer:
    """One writer rank: its core, its tracker and its backend handler."""

    def __init__(self) -> None:
        self.core = _InertWriterCore()
        self.tracker = RawBlockPDRequestTracker(self.core)  # type: ignore[arg-type]
        self.tracker.ack_endpoint = "127.0.0.1:0"
        self.backend = RustRawBlockBackend.__new__(RustRawBlockBackend)
        self.backend._core = self.core  # type: ignore[assignment]
        self.backend._pd_tracker = self.tracker
        self.backend._pd_session_id = SESSION
        self.backend._ack_tp_rank = 0

    def publish(self, req_id: str, keys: list[str]) -> StoragePDStatus:
        """Store one request the way the backend does and announce it."""
        terminal = self.tracker.register_batch(
            req_id,
            keys,
            expected_chunks=len(keys),
            is_last_batch=True,
            completed_keys=keys,
        )
        receipt = terminal.result(timeout=5)
        return StoragePDStatus.ready(req_id, 0, receipt)

    def answer(self, ack: StoragePDReadAck) -> tuple[str, str]:
        """Answer one acknowledgement through the backend's own handler."""
        return self.backend._answer_read_ack(
            StoragePDAckRequest(
                ack=ack,
                session_id=SESSION,
                attempt=1,
                nonce=uuid.uuid4().hex,
            )
        )

    def answer_claim(
        self,
        ack: StoragePDReadAck,
        *,
        session_id: str = SESSION,
    ) -> StoragePDClaimAnswer:
        """Answer one read claim through the backend's own handler."""
        return self.backend._answer_read_claim(
            StoragePDClaimRequest(
                read=ack,
                session_id=session_id,
                nonce=uuid.uuid4().hex,
            )
        )

    def report_unread(
        self,
        status: StoragePDStatus,
        *,
        session_id: str = SESSION,
    ) -> StoragePDUnreadAnswer:
        """Tell the writer this publication was never given a reader."""
        return self.backend._answer_unread_publication(
            StoragePDUnreadRequest(
                status=status,
                session_id=session_id,
                nonce=uuid.uuid4().hex,
                reason="no decoder was assigned this publication",
            )
        )

    def read(self, status: StoragePDStatus, consumer: str) -> StoragePDReadAck:
        """Claim and then acknowledge, the order a restore does it in."""
        ack = _consumer_ack(status, consumer)
        assert self.answer_claim(ack).granted
        return ack

    def close(self) -> None:
        self.tracker.close(release_leases=False)


@pytest.fixture
def producer():
    instance = _Producer()
    yield instance
    instance.close()


def _consumer_ack(
    status: StoragePDStatus,
    consumer_instance_id: str = STORAGE_PD_INCARNATION,
) -> StoragePDReadAck:
    """Build the acknowledgement a consumer owes, over the real wire codec.

    The round trip through msgspec is the point: a field the producer
    stopped sending, or one the consumer invents locally, does not survive
    it.
    """
    encoded = msgspec.msgpack.encode(status)
    adopted = msgspec.msgpack.decode(encoded, type=StoragePDStatus)
    return StoragePDReadAck.for_status(
        adopted,
        consumer_instance_id=consumer_instance_id,
    )


def test_an_ordinary_ready_is_acknowledged_by_its_own_writer(producer) -> None:
    status = producer.publish("request-1", ["key-1", "key-2"])
    assert producer.tracker.live_lease_count() == 1
    ack = producer.read(status, STORAGE_PD_INCARNATION)

    outcome, reason = producer.answer(ack)

    assert (outcome, reason) == (ACK_APPLIED, "")
    assert producer.tracker.live_lease_count() == 0
    assert producer.core.locked == []


def test_a_lost_reply_is_safe_to_retry(producer) -> None:
    status = producer.publish("request-1", ["key-1"])
    ack = producer.read(status, STORAGE_PD_INCARNATION)

    assert producer.answer(ack)[0] == ACK_APPLIED
    # The consumer heard nothing and asks again with the same message.
    assert producer.answer(ack)[0] == ACK_ALREADY_APPLIED
    assert producer.core.locked == []


def test_another_producers_acknowledgement_releases_nothing(producer) -> None:
    status = producer.publish("request-1", ["key-1"])
    other = _Producer()
    try:
        elsewhere = other.publish("request-1", ["key-1"])
        assert elsewhere.writer_epoch != status.writer_epoch

        outcome, _ = producer.answer(_consumer_ack(elsewhere))
    finally:
        other.close()

    assert outcome == ACK_REJECTED
    assert producer.tracker.live_lease_count() == 1
    assert producer.core.locked == ["key-1"]


def test_an_acknowledgement_for_an_earlier_generation_releases_nothing(
    producer,
) -> None:
    first = producer.publish("request-1", ["key-1"])
    assert producer.answer(producer.read(first, STORAGE_PD_INCARNATION))[0] == (
        ACK_APPLIED
    )
    second = producer.publish("request-2", ["key-2"])
    producer.read(second, STORAGE_PD_INCARNATION)
    assert second.checkpoint_seq != first.checkpoint_seq

    # Same writer, same rank, same session, but the manifest and generation
    # belong to the request that already finished.
    stale = msgspec.structs.replace(
        second,
        checkpoint_seq=first.checkpoint_seq,
        manifest_digest=first.manifest_digest,
    )
    outcome, _ = producer.answer(_consumer_ack(stale))

    assert outcome == ACK_REJECTED
    assert producer.tracker.live_lease_count() == 1
    assert producer.core.locked == ["key-2"]


def test_a_reader_engine_answers_for_no_publication() -> None:
    producer = _Producer()
    try:
        status = producer.publish("request-1", ["key-1"])
        producer.read(status, STORAGE_PD_INCARNATION)
        # A reader core adopted this namespace's epoch from a checkpoint.
        # The epoch it holds names the writer that published, not itself, so
        # it is not an identity an acknowledgement can be addressed by.
        producer.core.role = "reader"

        outcome, _ = producer.answer(_consumer_ack(status))

        assert outcome == ACK_REJECTED
        assert producer.tracker.live_lease_count() == 1
    finally:
        producer.core.role = "writer"
        producer.close()


def test_a_failed_status_names_no_producer_to_acknowledge() -> None:
    failed = StoragePDStatus(
        req_id="request-1",
        tp_rank=0,
        state="FAILED",
        error_stage="WRITE_OR_PUBLISH",
        error_text="the device refused the write",
    )
    with pytest.raises(ValueError, match="not READY"):
        StoragePDReadAck.for_status(failed, consumer_instance_id="consumer")


def test_a_consumer_that_restarted_before_acknowledging_takes_nothing_over(
    producer,
) -> None:
    """The reader is settled before the read, not by whoever asks first.

    A consumer reads one publication and has not acknowledged it yet when
    the process is replaced. The replacement is handed a second publication
    and gets there first. If the binding were settled by the first
    acknowledgement, the replacement would take the session and the
    original's completed read would be refused forever.
    """
    original = producer.publish("request-1", ["key-1"])
    replacement = producer.publish("request-2", ["key-2"])

    # The original consumer claims and reads, and is slow to acknowledge.
    original_ack = producer.read(original, "consumer-before-the-restart")

    # Its replacement comes up and asks for the other publication first.
    refused = producer.answer_claim(
        _consumer_ack(replacement, "consumer-after-the-restart")
    )
    assert not refused.granted
    assert refused.final

    # The original's acknowledgement still applies, and the publication the
    # replacement was refused is still held for nobody -- which is correct:
    # a process that came up afterwards did not do that reading.
    assert producer.answer(original_ack)[0] == ACK_APPLIED
    assert producer.tracker.live_lease_count() == 1
    assert producer.core.locked == ["key-2"]


def test_two_requests_sharing_a_key_hold_it_twice(producer) -> None:
    """Deduplication does not merge two reads into one release.

    Two requests naming the same extent each take a hold, so the extent
    stays protected until both consumers are done with it.
    """
    first = producer.publish("request-1", ["shared-key"])
    second = producer.publish("request-2", ["shared-key"])
    assert producer.core.locked == ["shared-key", "shared-key"]

    assert producer.answer(producer.read(first, STORAGE_PD_INCARNATION))[0] == (
        ACK_APPLIED
    )
    assert producer.core.locked == ["shared-key"]

    assert producer.answer(producer.read(second, STORAGE_PD_INCARNATION))[0] == (
        ACK_APPLIED
    )
    assert producer.core.locked == []


def test_a_claim_under_another_session_is_not_this_writers_business(
    producer,
) -> None:
    """A producer answers for the group it was configured into, and no other."""
    status = producer.publish("request-1", ["key-1"])

    answer = producer.answer_claim(
        _consumer_ack(status),
        session_id="a-different-deployment",
    )

    assert not answer.granted
    assert producer.tracker.live_lease_count() == 1


def test_a_publication_nobody_was_given_can_be_resolved(producer) -> None:
    """A request answered without a decoder still has to resolve its hold.

    The caller asked for one token, or the producer stopped on its own.
    Nothing is ever going to acknowledge the publication, so a writer that
    only released on acknowledgements would hold those extents until its
    own admission bound stopped it publishing at all.
    """
    status = producer.publish("request-1", ["key-1"])
    assert producer.tracker.live_lease_count() == 1

    answer = producer.report_unread(status)

    assert answer.released
    assert producer.tracker.live_lease_count() == 0
    assert producer.core.locked == []


def test_a_claimed_publication_is_not_resolved_as_unread(producer) -> None:
    """A consumer that took the read is the only one who can release it.

    Only that consumer's acknowledgement frees these extents, whatever
    anybody else believes about who was assigned the read -- and a caller
    that is wrong about the assignment must not reclaim an extent somebody
    is reading.
    """
    status = producer.publish("request-1", ["key-1"])
    ack = producer.read(status, STORAGE_PD_INCARNATION)

    answer = producer.report_unread(status)

    assert not answer.released
    assert producer.tracker.live_lease_count() == 1
    assert producer.core.locked == ["key-1"]
    # And the real acknowledgement still works afterwards.
    assert producer.answer(ack)[0] == ACK_APPLIED


def test_an_unread_report_naming_another_producer_releases_nothing(
    producer,
) -> None:
    """A report is matched against what this writer published, like an ack."""
    status = producer.publish("request-1", ["key-1"])
    other = _Producer()
    try:
        elsewhere = other.publish("request-1", ["key-1"])
        answer = producer.report_unread(elsewhere)
    finally:
        other.close()

    assert not answer.released
    assert producer.tracker.live_lease_count() == 1
    assert status.writer_epoch != elsewhere.writer_epoch


def test_an_unread_report_for_a_different_manifest_releases_nothing(
    producer,
) -> None:
    """Same writer, same rank, different publication."""
    status = producer.publish("request-1", ["key-1"])
    stale = msgspec.structs.replace(status, manifest_digest="not-what-was-published")

    answer = producer.report_unread(stale)

    assert not answer.released
    assert producer.tracker.live_lease_count() == 1
