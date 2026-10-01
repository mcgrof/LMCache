# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, cast

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
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


def test_decoder_adoption_retries_without_the_continuation_token() -> None:
    class TokenDatabase:
        def process_tokens(self, *, tokens, request_configs):
            del request_configs
            yield (
                0,
                len(tokens),
                CacheEngineKey(
                    "model",
                    1,
                    0,
                    hash(tuple(tokens)),
                    torch.bfloat16,
                ),
            )

    class Backend:
        def __init__(self) -> None:
            self.keys: list[int] = []

        def adopt_publication(self, receipt, keys, *, timeout_ms, request_id=""):
            del receipt, timeout_ms, request_id
            self.keys.append(keys[0].chunk_hash)
            return keys[0].chunk_hash == hash((1, 2, 3))

    backend = Backend()
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.token_database = cast(Any, TokenDatabase())
    engine.storage_manager = type(
        "StorageManager",
        (),
        {"storage_backends": {"raw": backend}},
    )()
    receipt = RawBlockPublicationReceipt(
        writer_epoch="writer",
        checkpoint_seq=1,
        key_count=1,
        manifest_digest="digest",
    )

    assert engine.adopt_storage_publication([1, 2, 3, 4], receipt) == 3
    assert backend.keys == [hash((1, 2, 3, 4)), hash((1, 2, 3))]


def test_sender_close_does_not_claim_a_socket_owner_it_abandoned() -> None:
    """Abandoning an executor is not the same as it having finished.

    A socket stuck in the transport outlasts the shutdown budget, and close
    gives up on it rather than blocking. The thread is then still alive and
    still holding the socket, so the caller has to be told -- otherwise it
    destroys what that thread is using next.
    """
    # Standard
    import threading

    sender = StoragePDStatusSender("127.0.0.1", 1, timeout_s=1.0)
    stuck = threading.Event()
    try:

        def _never_finishes() -> None:
            assert stuck.wait(30.0)

        sender._close_socket = _never_finishes  # type: ignore[method-assign]
        assert sender.close(timeout_s=0.1) is False
    finally:
        stuck.set()


def test_sender_close_reports_a_clean_stop_when_there_is_one() -> None:
    """The refusal above is an observation, not a stuck answer."""
    sender = StoragePDStatusSender("127.0.0.1", 1, timeout_s=1.0)
    assert sender.close(timeout_s=5.0) is True
    # A second close has nothing left to establish and says so.
    assert sender.close(timeout_s=5.0) is True
