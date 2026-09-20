# SPDX-License-Identifier: Apache-2.0

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.storage_backend.pd_backend import PDMsg
from lmcache.v1.storage_backend.raw_block import RawBlockPublicationReceipt
from lmcache.v1.storage_backend.storage_pd_protocol import (
    StoragePDStatus,
    StoragePDStatusSender,
    order_storage_pd_ready_statuses,
)


def test_storage_pd_ready_status_round_trip_through_pd_union() -> None:
    receipt = RawBlockPublicationReceipt(
        writer_epoch="writer-epoch",
        checkpoint_seq=7,
        key_count=3,
        manifest_digest="manifest",
        namespace_identity="block:uuid:namespace",
        total_logical_bytes=123,
        total_padded_bytes=4096,
    )

    encoded = msgspec.msgpack.encode(StoragePDStatus.ready("request", 2, receipt))
    decoded = msgspec.msgpack.decode(encoded, type=PDMsg)

    assert isinstance(decoded, StoragePDStatus)
    assert decoded.req_id == "request"
    assert decoded.tp_rank == 2
    assert decoded.state == "READY"
    assert decoded.publication_receipt() == receipt


def test_storage_pd_failure_cannot_be_converted_to_a_receipt() -> None:
    status = StoragePDStatus(
        req_id="request",
        producer_instance_id="producer",
        tp_rank=0,
        state="FAILED",
        error_stage="WRITE",
        error_text="short completion",
    )

    try:
        status.publication_receipt()
    except ValueError as exc:
        assert "not READY" in str(exc)
    else:
        raise AssertionError("FAILED status produced a publication receipt")
def test_storage_pd_ready_barrier_requires_the_exact_rank_set() -> None:
    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    statuses = {
        0: StoragePDStatus.ready("request", 0, receipt),
        1: StoragePDStatus.ready("request", 1, receipt),
    }

    assert order_storage_pd_ready_statuses(statuses, 2) == [
        statuses[0],
        statuses[1],
    ]
    with pytest.raises(ValueError, match="rank set mismatch"):
        order_storage_pd_ready_statuses({1: statuses[1], 2: statuses[1]}, 2)


def test_storage_pd_ready_barrier_rejects_mixed_request_ids() -> None:
    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    statuses = {
        0: StoragePDStatus.ready("request-a", 0, receipt),
        1: StoragePDStatus.ready("request-b", 1, receipt),
    }

    with pytest.raises(ValueError, match="different requests"):
        order_storage_pd_ready_statuses(statuses, 2)


def test_storage_pd_status_sender_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        StoragePDStatusSender("localhost", 1, timeout_s=0)
