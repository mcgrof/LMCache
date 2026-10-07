# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import replace
from typing import Any
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block.core import (
    RawBlockIoAttribution,
    RawBlockIoContext,
    RawBlockPublicationReceipt,
)


def journal_row(tag: str, event_outcome: str, **changes: Any) -> dict[str, Any]:
    """Build one native-format event for a registered payload operation."""
    return {
        "request_tag": tag,
        "device_instance_id": "1",
        "batch_id": "1",
        "operation_id": "1",
        "attempt": "0",
        "direction": "read",
        "path": "regular",
        "outcome": event_outcome,
        "bytes": "4096",
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"outcome": "lost"},
        {"operation_id": "not-an-integer"},
        {"operation_id": 1.5},
        {"operation_id": True},
        {"operation_id": "01"},
        {"device_instance_id": None},
        {"bytes": "not-an-integer"},
        {"attempt": "-1"},
        {"batch_id": ""},
        {"direction": "sideways"},
        {"path": "unknown"},
        {"bytes": "-1"},
    ],
)
def test_malformed_native_events_fail_evidence_without_raising(
    changes: dict[str, Any],
) -> None:
    record = RawBlockIoAttribution()
    tag = record.register_context(RawBlockIoContext(request_id="request"))
    record.record([journal_row(tag, "submitted", **changes)])
    payload = record.as_payload()
    assert payload["malformed_rows"] == 1
    assert "malformed_rows=1" in payload["evidence_failures"]
    assert payload["operations"] == []


def test_an_operation_without_its_context_cannot_pass_the_join() -> None:
    record = RawBlockIoAttribution()
    record.record(
        [journal_row("unknown", outcome) for outcome in ("submitted", "completed")]
    )
    operations, failures = record.operation_join()
    assert not operations[0]["complete"]
    assert any("operation_has_no_context" in failure for failure in failures)


@pytest.mark.parametrize("field", ["batch_id", "request_tag", "direction", "path"])
def test_an_operation_cannot_change_identity_between_submission_and_cqe(
    field: str,
) -> None:
    record = RawBlockIoAttribution()
    tag = record.register_context(RawBlockIoContext(request_id="request"))
    values = {
        "batch_id": "2",
        "request_tag": "another",
        "direction": "write",
        "path": "bounce",
    }
    record.record(
        [
            journal_row(tag, "submitted"),
            journal_row(tag, "completed", **{field: values[field]}),
        ]
    )
    operations, failures = record.operation_join()
    assert len(operations) == 1
    assert not operations[0]["complete"]
    assert any("operation_identity_changed" in failure for failure in failures)


def test_repeated_engine_metadata_is_bounded_and_reports_eviction() -> None:
    record = RawBlockIoAttribution(max_requests=2, max_events=4)
    context = RawBlockIoContext(request_id="<engine-metadata>")
    for number in range(10):
        tag = record.register_context(context)
        record.record(
            [
                journal_row(tag, outcome, operation_id=str(number))
                for outcome in ("submitted", "completed")
            ]
        )
        payload = record.as_payload()
        assert len(payload["operations"]) <= 2
    assert payload["evicted_requests"] > 0
    assert any("evicted_rows=" in failure for failure in payload["evidence_failures"])


@pytest.mark.parametrize(
    "first_outcome,first_bytes,remainder_bytes",
    [("short", "0", "4096"), ("short", "4096", "0"), ("completed", "4096", "0")],
)
def test_invalid_remainder_chains_cannot_pass_balanced_byte_totals(
    first_outcome: str,
    first_bytes: str,
    remainder_bytes: str,
) -> None:
    record = RawBlockIoAttribution()
    tag = record.register_context(RawBlockIoContext(request_id="request"))
    record.record(
        [
            journal_row(tag, "submitted"),
            journal_row(tag, first_outcome, bytes=first_bytes),
            journal_row(tag, "submitted", attempt="1", bytes=remainder_bytes),
            journal_row(tag, "completed", attempt="1", bytes=remainder_bytes),
        ]
    )
    operations, failures = record.operation_join()
    assert operations[0]["requested_bytes"] == operations[0]["completed_bytes"]
    assert not operations[0]["complete"]
    assert failures


def test_status_snapshot_blocks_recorders_until_all_views_are_captured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = RawBlockIoAttribution()
    tag = record.register_context(RawBlockIoContext(request_id="request"))
    submitted = journal_row(tag, "submitted")
    completed = journal_row(tag, "completed")
    record.record([submitted])
    joining = threading.Event()
    recording = threading.Event()
    old_join = record.operation_join

    def join() -> tuple[list[dict[str, Any]], list[str]]:
        result = old_join()
        joining.set()
        assert recording.wait(5)
        return result

    monkeypatch.setattr(record, "operation_join", join)

    def complete() -> None:
        assert joining.wait(5)
        recording.set()
        record.record([completed])

    thread = threading.Thread(target=complete)
    thread.start()
    try:
        payload = record.as_payload()
    finally:
        thread.join(5)
    assert not thread.is_alive()
    operation = payload["operations"][0]
    assert not operation["complete"]
    assert payload["unanswered"] == {f"{tag}/read/regular": 1}
    assert record.unanswered() == {}


def test_registering_the_same_context_preserves_its_linked_publication() -> None:
    record = RawBlockIoAttribution()
    context = RawBlockIoContext(request_id="request", incarnation="writer")
    tag = record.register_context(context)
    receipt = RawBlockPublicationReceipt(
        "writer", 7, 1, "manifest", namespace_identity="namespace"
    )
    record.link_publication("request", receipt)
    assert record.register_context(context) == tag
    linked = record.contexts()[tag]
    assert linked["advertised_checkpoint_seq"] == 7
    assert linked["manifest_digest"] == "manifest"
    assert record.as_payload()["context_conflicts"] == 0


def test_reused_tag_cannot_silently_replace_restore_identity() -> None:
    record = RawBlockIoAttribution()
    context = RawBlockIoContext(
        request_id="request", incarnation="writer", checkpoint_seq=3
    )
    tag = record.register_context(context)
    assert record.register_context(replace(context, checkpoint_seq=4)) == tag
    payload = record.as_payload()
    assert payload["contexts"][tag]["advertised_checkpoint_seq"] == 3
    assert payload["context_conflicts"] == 1
    assert "context_conflict_rows=1" in payload["evidence_failures"]


def test_reordered_concurrent_drains_join_when_submission_arrives() -> None:
    record = RawBlockIoAttribution()
    tag = record.register_context(RawBlockIoContext(request_id="request"))
    record.record([journal_row(tag, "completed")])
    assert record.operation_join()[1]
    record.record([journal_row(tag, "submitted")])
    operations, failures = record.operation_join()
    assert failures == []
    assert operations[0]["complete"]
    assert record.as_payload()["malformed_rows"] == 0


def test_failed_completion_never_credits_payload_bytes() -> None:
    record = RawBlockIoAttribution()
    tag = record.register_context(RawBlockIoContext(request_id="request"))
    record.record([journal_row(tag, "submitted"), journal_row(tag, "failed")])
    payload = record.as_payload()
    assert payload["rows"][f"{tag}/read/regular"]["bytes"] == 0
    operation = payload["operations"][0]
    assert operation["completed_bytes"] == 0
    assert operation["attempts"][0]["completed_bytes"] == 0
    assert not operation["complete"]
