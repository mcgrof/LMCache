# SPDX-License-Identifier: Apache-2.0
"""The whole acknowledgement protocol, over a socket, with nothing stubbed.

The integration test beside the backend calls its handlers directly, which
is enough to show that the identities line up but not that the exchange
works: a reply that never correlates, a claim that never reaches the
writer, or a serializer that drops a field are all invisible to an
in-process call.

So this runs the real thing end to end. The real adapter builds the claim
and the acknowledgement from an adopted READY status, the real client puts
them on a loopback socket, the real server answers them, and the real
backend handlers reach a real tracker. Nothing here is hand-matched:
every correlating field is minted by the writer and travels the whole
path.

Storage is inert -- no device, no model, no GPU -- because what is under
test is who is allowed to decide that an extent is reclaimable.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from types import SimpleNamespace
import asyncio
import contextlib
import socket as socketlib
import threading
import uuid

# Third Party
import pytest

pytest.importorskip("vllm")

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
    StoragePDAckClient,
    StoragePDAckServer,
)
from lmcache.v1.storage_backend.storage_pd_protocol import (
    StoragePDReadAck,
    StoragePDStatus,
)

# First Party
from lmcache.integration.vllm import vllm_v1_adapter as adapter_module  # isort: skip
from lmcache.integration.vllm.vllm_v1_adapter import (  # isort: skip
    LMCacheConnectorV1Impl,
)

LOOPBACK = "127.0.0.1"
SESSION = "loopback-session"


def _free_port() -> int:
    with socketlib.socket(socketlib.AF_INET, socketlib.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return int(probe.getsockname()[1])


class _InertWriterCore:
    """A writer's identity and lock bookkeeping, with no device behind it."""

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
        self, encoded_keys: list[str], *, lock: bool = False
    ) -> list[object]:
        assert lock
        self.locked.extend(encoded_keys)
        return [object() for _ in encoded_keys]

    def unlock_many(self, encoded_keys: list[str]) -> None:
        for encoded_key in encoded_keys:
            self.locked.remove(encoded_key)


class _Writer:
    """One writer rank: its core, its tracker, its backend and its socket."""

    def __init__(self) -> None:
        self.core = _InertWriterCore()
        self.tracker = RawBlockPDRequestTracker(self.core)  # type: ignore[arg-type]
        self.backend = RustRawBlockBackend.__new__(RustRawBlockBackend)
        self.backend._core = self.core  # type: ignore[assignment]
        self.backend._pd_tracker = self.tracker
        self.backend._pd_session_id = SESSION
        self.backend._ack_tp_rank = 0
        self.server = StoragePDAckServer(
            self.backend._answer_read_ack,
            claim_handler=self.backend._answer_read_claim,
            unread_handler=self.backend._answer_unread_publication,
            bind_host=LOOPBACK,
            port=_free_port(),
            advertise_host=LOOPBACK,
            recv_timeout_ms=50,
        )
        self.tracker.ack_endpoint = self.server.endpoint

    def publish(self, req_id: str, keys: list[str]) -> StoragePDStatus:
        terminal = self.tracker.register_batch(
            req_id,
            keys,
            expected_chunks=len(keys),
            is_last_batch=True,
            completed_keys=keys,
        )
        return StoragePDStatus.ready(req_id, 0, terminal.result(timeout=5))

    def close(self) -> None:
        self.server.close(timeout_s=5.0)
        self.tracker.close()


class _Consumer:
    """One reader: the real adapter methods over a real client.

    ``incarnation`` stands in for a restart. The adapter names itself with
    a module-level identity minted once per process, which is exactly the
    point of it -- so two consumers in one process are the same
    incarnation unless that name is replaced for one of them, and a test
    that did not replace it would be driving one consumer twice.
    """

    def __init__(self, *, max_owed: int = 8, incarnation: str = "") -> None:
        self.client = StoragePDAckClient(
            attempt_timeout_ms=500,
            retry_interval_s=0.05,
            max_live_obligations=max_owed,
            poll_interval_s=0.01,
            claim_attempts=2,
        )
        self.adapter = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
        self.adapter._storage_pd_ack_client = self.client
        self.adapter._storage_pd_lock = threading.Lock()
        self.adapter._storage_pd_claims = OrderedDict()
        self.adapter._storage_pd_acks_sent = OrderedDict()
        self.adapter._storage_pd_session_id = SESSION
        self.adapter._storage_pd_ack_deadline_s = 20.0
        self.incarnation = incarnation

    @contextlib.contextmanager
    def _named(self):
        """Run the adapter's own code under this consumer's identity."""
        if not self.incarnation:
            yield
            return
        previous = adapter_module.STORAGE_PD_INCARNATION
        adapter_module.STORAGE_PD_INCARNATION = self.incarnation
        try:
            yield
        finally:
            adapter_module.STORAGE_PD_INCARNATION = previous

    def claim(self, status: StoragePDStatus) -> None:
        """Ask to read, the way the restore path asks."""
        with self._named():
            self.adapter._claim_storage_pd_read(status)

    def acknowledge(self, status: StoragePDStatus):
        """Owe the acknowledgement, the way a finished restore owes it."""
        with self._named():
            self.adapter._ack_storage_pd_restore(status.req_id, status)
        with self.adapter._storage_pd_lock:
            owed = list(self.client._owed)
        return owed[-1] if owed else None

    def retry_the_same_acknowledgement(self, status: StoragePDStatus):
        """Ask again with the message already sent, as a lost reply does.

        A retry is the same obligation attempted again, not a new claim: the
        hold may already be gone, and asking to read it again would be a
        different request entirely.
        """
        ack = StoragePDReadAck.for_status(
            status,
            consumer_instance_id=self.incarnation
            or adapter_module.STORAGE_PD_INCARNATION,
        )
        reservation = self.client.reserve(endpoint=status.ack_endpoint)
        assert reservation is not None
        return self.client.owe(
            ack,
            endpoint=status.ack_endpoint,
            session_id=SESSION,
            deadline_s=20.0,
            reservation=reservation,
        )

    def close(self) -> None:
        self.client.close(timeout_s=5.0)


@pytest.fixture
def writer():
    instance = _Writer()
    yield instance
    instance.close()


@pytest.fixture
def consumer():
    instance = _Consumer()
    yield instance
    instance.close()


def test_a_whole_handoff_settles_over_a_socket(writer, consumer) -> None:
    """Claim, read, acknowledge: the ordinary case, nothing stubbed."""
    status = writer.publish("request-1", ["key-1"])
    assert writer.tracker.live_lease_count() == 1

    consumer.claim(status)
    obligation = consumer.acknowledge(status)

    assert obligation is not None
    assert obligation.wait(timeout=20) == ACK_APPLIED
    assert writer.tracker.live_lease_count() == 0
    assert writer.core.locked == []


def test_a_lost_reply_retries_as_already_applied(writer, consumer) -> None:
    """The writer released the hold and its answer was lost.

    The consumer cannot tell that from a request that never arrived, so it
    asks again. The writer must say the hold was already applied and must
    not release a second reference.
    """
    status = writer.publish("request-1", ["key-1"])
    consumer.claim(status)
    first = consumer.acknowledge(status)
    assert first is not None
    assert first.wait(timeout=20) == ACK_APPLIED

    # The same message again, from the same consumer, over the same socket.
    retry = consumer.retry_the_same_acknowledgement(status)
    assert retry is not None
    assert retry.wait(timeout=20) == ACK_ALREADY_APPLIED
    assert writer.core.locked == []


def test_another_writers_publication_is_refused_on_the_wire(writer, consumer) -> None:
    """A producer answers only for what it published itself."""
    writer.publish("request-1", ["key-1"])
    elsewhere = _Writer()
    try:
        other = elsewhere.publish("request-1", ["key-1"])
        # Addressed to this writer's socket, naming the other's publication.
        misdirected = StoragePDStatus(
            req_id=other.req_id,
            tp_rank=other.tp_rank,
            state="READY",
            writer_epoch=other.writer_epoch,
            namespace_identity=other.namespace_identity,
            checkpoint_seq=other.checkpoint_seq,
            key_count=other.key_count,
            manifest_digest=other.manifest_digest,
            ack_endpoint=writer.server.endpoint,
        )
        with pytest.raises(RuntimeError, match="did not grant"):
            consumer.claim(misdirected)
    finally:
        elsewhere.close()

    assert writer.tracker.live_lease_count() == 1
    assert writer.core.locked == ["key-1"]


def test_a_restarted_consumer_cannot_take_the_session(writer) -> None:
    """Settled before the read, so arriving first wins nothing.

    The original consumer claims and has not acknowledged. Its replacement
    comes up and asks for a different publication. If the binding were
    settled by the first acknowledgement the replacement would take the
    session and the original's completed read would be refused forever.
    """
    first = writer.publish("request-1", ["key-1"])
    second = writer.publish("request-2", ["key-2"])
    original = _Consumer(incarnation="pid:1000:before-the-restart")
    replacement = _Consumer(incarnation="pid:1001:after-the-restart")
    try:
        original.claim(first)

        with pytest.raises(RuntimeError, match="did not grant"):
            replacement.claim(second)

        obligation = original.acknowledge(first)
        assert obligation is not None
        assert obligation.wait(timeout=20) == ACK_APPLIED
    finally:
        original.close()
        replacement.close()

    assert writer.tracker.live_lease_count() == 1
    assert writer.core.locked == ["key-2"]


def test_two_requests_sharing_a_key_need_two_acknowledgements(writer, consumer) -> None:
    """Deduplication does not merge two reads into one release."""
    first = writer.publish("request-1", ["shared-key"])
    second = writer.publish("request-2", ["shared-key"])
    assert writer.core.locked == ["shared-key", "shared-key"]

    for status in (first, second):
        consumer.claim(status)
        obligation = consumer.acknowledge(status)
        assert obligation is not None
        assert obligation.wait(timeout=20) == ACK_APPLIED

    assert writer.core.locked == []


def test_a_consumer_at_its_bound_does_not_read(writer) -> None:
    """Room is taken before the read, and refused before it too.

    Reading and then finding there is no room to acknowledge leaves the
    writer holding extents for a read that did happen, and no later
    restore is guaranteed to come along and retry.
    """
    statuses = [writer.publish(f"request-{i}", [f"key-{i}"]) for i in range(3)]
    consumer = _Consumer(max_owed=2)
    try:
        consumer.claim(statuses[0])
        consumer.claim(statuses[1])

        with pytest.raises(RuntimeError, match="no room to acknowledge"):
            consumer.claim(statuses[2])

        # And the reads that were admitted still settle.
        for status in statuses[:2]:
            obligation = consumer.acknowledge(status)
            assert obligation is not None
            assert obligation.wait(timeout=20) == ACK_APPLIED
    finally:
        consumer.close()

    # The refused one was never read and is still held, for nobody.
    assert writer.tracker.live_lease_count() == 1
    assert writer.core.locked == ["key-2"]


def test_a_producer_only_publication_is_resolved_over_the_wire(writer) -> None:
    """The single-token answer, resolved by the party that knows.

    No decoder was assigned, so nothing will ever acknowledge this
    publication. The proxy's own notifier tells the writer, which checks
    that nobody claimed the read before releasing anything.
    """
    # First Party
    from examples.disagg_prefill import disagg_proxy_server as proxy

    status = writer.publish("request-1", ["key-1"])
    assert writer.tracker.live_lease_count() == 1

    previous = getattr(proxy, "global_args", None)
    proxy.global_args = SimpleNamespace(storage_pd_session=SESSION)
    try:
        asyncio.run(
            proxy.tell_producers_nobody_will_read(
                [status], reason="the producer's single token is the whole answer"
            )
        )
    finally:
        if previous is None:
            del proxy.global_args
        else:
            proxy.global_args = previous

    assert writer.tracker.live_lease_count() == 0
    assert writer.core.locked == []


def test_a_claimed_publication_is_not_resolved_as_unread(writer, consumer) -> None:
    """A consumer that took the read is the only one who can release it."""
    # First Party
    from examples.disagg_prefill import disagg_proxy_server as proxy

    status = writer.publish("request-1", ["key-1"])
    consumer.claim(status)

    previous = getattr(proxy, "global_args", None)
    proxy.global_args = SimpleNamespace(storage_pd_session=SESSION)
    try:
        asyncio.run(
            proxy.tell_producers_nobody_will_read([status], reason="no decoder")
        )
    finally:
        if previous is None:
            del proxy.global_args
        else:
            proxy.global_args = previous

    assert writer.tracker.live_lease_count() == 1
    obligation = consumer.acknowledge(status)
    assert obligation is not None
    assert obligation.wait(timeout=20) == ACK_APPLIED
    assert writer.core.locked == []


def test_an_unreachable_writer_is_not_read_from(consumer) -> None:
    """Nothing is granted by a writer that never answered."""
    receipt = RawBlockPublicationReceipt(
        writer_epoch="a-writer-that-is-not-listening",
        checkpoint_seq=1,
        key_count=1,
        manifest_digest="digest",
        namespace_identity="namespace-under-test",
        ack_endpoint=f"{LOOPBACK}:{_free_port()}",
    )

    with pytest.raises(RuntimeError, match="did not grant"):
        consumer.claim(StoragePDStatus.ready("request-1", 0, receipt))


def test_the_rejected_identity_never_came_from_the_message(writer, consumer) -> None:
    """Every field that must match was minted by the writer.

    The guard this protects is that the expected identity is the writer's
    own, not one taken from the arriving message -- a check against fields
    the message supplied would check nothing.
    """
    status = writer.publish("request-1", ["key-1"])
    consumer.claim(status)

    # The writer's epoch is what the status carries, and what the backend
    # compares against is the core that minted it. Neither was written here.
    assert status.writer_epoch == writer.core.writer_epoch
    assert status.writer_epoch not in (None, "")

    obligation = consumer.acknowledge(status)
    assert obligation is not None
    assert obligation.wait(timeout=20) == ACK_APPLIED
    assert ACK_REJECTED not in (obligation.outcome,)
