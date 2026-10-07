# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Any
import hashlib
import json

# Third Party
import pytest

# First Party
from examples.disagg_prefill.storage_pd_timeline_check import (
    SCHEMA,
    TimelineError,
    check_timeline,
)


def _row(event: str, timestamp: int, **fields: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "event": event,
        "monotonic_ns": timestamp,
        "pid": 123,
        "process_sequence": timestamp,
        "request_id": "wire-request",
        "tp_rank": 0,
        "writer_epoch": "writer-epoch",
        "consumer_instance_id": "consumer-instance",
        **(
            {
                "device_instance_id": "1",
                "batch_id": "1",
                "operation_id": str(timestamp),
                "attempt": "0",
            }
            if event == "io_cqe"
            else {}
        ),
        **fields,
    }


def _timeline() -> list[dict[str, Any]]:
    receipt = {
        "advertised_checkpoint_seq": 7,
        "manifest_digest": "digest",
        "namespace_identity": "namespace",
    }
    attempt = {
        **receipt,
        "consumer_request_id": "consumer-request",
        "restore_attempt_id": "attempt",
    }
    return [
        _row(
            "io_cqe",
            5,
            direction="write",
            outcome="completed",
            path="bounce",
            completed_bytes=4096,
        ),
        _row(
            "io_cqe",
            10,
            direction="write",
            outcome="completed",
            path="dmabuf_fixed",
            completed_bytes=4096,
        ),
        _row("publication", 20, **receipt),
        _row("ready", 30, **receipt),
        _row("claim", 40, **attempt),
        _row("adoption", 50, **attempt),
        _row(
            "io_cqe",
            60,
            direction="read",
            outcome="completed",
            path="dmabuf_fixed",
            completed_bytes=4096,
            **attempt,
        ),
        _row("restore_complete", 70, **attempt),
        _row("ack_owed", 80, **attempt),
        _row("ack_applied", 90, **receipt),
        _row("decode", 100, **receipt),
    ]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _reuse_timeline() -> list[dict[str, Any]]:
    record = {
        "key": "key",
        "size": 4096,
        "shape": [2048],
        "dtype": "float16",
        "fmt": "KV_2LTD",
    }
    digest = hashlib.sha256(
        json.dumps([record], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    original = _timeline()
    for row in original:
        if "manifest_digest" in row:
            row["manifest_digest"] = digest
    original[0]["offset"] = 0
    original[1]["offset"] = 4096
    original[2].update(
        key_count=1,
        payload_provenance=[
            {
                "record": record,
                "offset": 4096,
                "padded_bytes": 4096,
                "source_request_id": "wire-request",
                "write_generation": "generation-1",
            }
        ],
    )
    original.insert(
        2,
        _row(
            "payload_commit",
            15,
            key="key",
            offset=4096,
            padded_bytes=4096,
            write_generation="generation-1",
        ),
    )
    repeated = []
    for row in original[3:]:
        if row["event"] == "io_cqe":
            continue
        repeated.append(
            {
                **row,
                "request_id": "deduplicated-request",
                "monotonic_ns": row["monotonic_ns"] + 100,
                "process_sequence": row["process_sequence"] + 100,
            }
        )
    for row in repeated:
        if row["event"] == "restore_complete":
            row.update(published_tokens=16, resident_tokens=16, restored_tokens=0)
    return original + repeated


def test_timeline_proves_deduplicated_publication_and_resident_restore(
    tmp_path: Path,
) -> None:
    path = tmp_path / "timeline.jsonl"
    _write(path, _reuse_timeline())
    report = check_timeline(path, required_path="dmabuf_fixed")
    assert report["publication_count"] == 2
    assert report["restore_count"] == 2
    assert report["attempts"][1]["fully_resident"] is True
    assert report["attempts"][1]["read_bytes"] == 0


@pytest.mark.parametrize(
    "mutation", ["generation", "source", "offset", "key", "short", "path", "overwrite"]
)
def test_timeline_rejects_unproved_payload_reuse(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "timeline.jsonl"
    rows = _reuse_timeline()
    if mutation in ("generation", "source", "offset", "key"):
        field, value = {
            "generation": ("write_generation", "stale"),
            "source": ("source_request_id", "other-request"),
            "offset": ("offset", 8192),
            "key": ("record", {"key": "wrong-key"}),
        }[mutation]
        rows[-8]["payload_provenance"] = [
            dict(rows[-8]["payload_provenance"][0], **{field: value})
        ]
    elif mutation == "short":
        rows[1]["completed_bytes"] = 2048
    elif mutation == "path":
        rows[1]["path"] = "bounce"
    else:
        rows.insert(
            12,
            _row(
                "payload_commit",
                115,
                key="another-key",
                offset=4096,
                padded_bytes=4096,
                write_generation="generation-2",
            ),
        )
    _write(path, rows)
    with pytest.raises(TimelineError):
        check_timeline(path, required_path="dmabuf_fixed")


@pytest.mark.parametrize(
    "field,value",
    [
        ("resident_tokens", 15),
        ("published_tokens", 0),
        ("resident_tokens", True),
        ("restored_tokens", 1),
    ],
)
def test_timeline_rejects_unproved_resident_restore(
    tmp_path: Path, field: str, value: Any
) -> None:
    path = tmp_path / "timeline.jsonl"
    rows = _reuse_timeline()
    next(
        row
        for row in rows
        if row["event"] == "restore_complete"
        and row["request_id"] == "deduplicated-request"
    )[field] = value
    _write(path, rows)
    with pytest.raises(TimelineError, match="no read CQE"):
        check_timeline(path, required_path="dmabuf_fixed")


def test_timeline_proves_dma_buf_restore(tmp_path: Path) -> None:
    path = tmp_path / "timeline.jsonl"
    _write(path, _timeline())

    report = check_timeline(path, required_path="dmabuf_fixed")

    assert report["result"] == "PASS"
    assert report["publication_count"] == 1
    assert report["restore_count"] == 1
    assert report["unread_count"] == 0
    assert report["attempts"][0]["restore_attempt_id"] == "attempt"


def test_timeline_proves_an_unread_publication_was_released(tmp_path: Path) -> None:
    path = tmp_path / "timeline.jsonl"
    rows = _timeline()[:4]
    receipt = {
        "advertised_checkpoint_seq": 7,
        "manifest_digest": "digest",
        "namespace_identity": "namespace",
    }
    rows.extend(
        [
            _row("unread_owed", 40, **receipt),
            _row("unread_applied", 50, **receipt),
        ]
    )
    _write(path, rows)

    report = check_timeline(path, required_path="dmabuf_fixed")

    assert report["result"] == "PASS"
    assert report["restore_count"] == 0
    assert report["unread_count"] == 1


def test_timeline_rejects_decode_before_restore(tmp_path: Path) -> None:
    path = tmp_path / "timeline.jsonl"
    rows = _timeline()
    rows[-1]["monotonic_ns"] = 65
    rows[-1]["pid"] = 456
    _write(path, rows)

    with pytest.raises(TimelineError, match="decoded before restore"):
        check_timeline(path, required_path="dmabuf_fixed")


@pytest.mark.parametrize("direction", ["write", "read"])
def test_timeline_rejects_completions_without_a_publication(
    tmp_path: Path, direction: str
) -> None:
    path = tmp_path / "timeline.jsonl"
    rows = _timeline()
    source = rows[1] if direction == "write" else rows[6]
    rows.append(
        {
            **source,
            "request_id": "missing-publication",
            "operation_id": "110",
            "process_sequence": 110,
            "monotonic_ns": 110,
        }
    )
    _write(path, rows)

    with pytest.raises(TimelineError, match="orphan CQE"):
        check_timeline(path, required_path="dmabuf_fixed")


def test_timeline_rejects_duplicate_native_completions(tmp_path: Path) -> None:
    path = tmp_path / "timeline.jsonl"
    rows = _timeline()
    rows.insert(7, {**rows[6], "process_sequence": 61, "monotonic_ns": 61})
    _write(path, rows)

    with pytest.raises(TimelineError, match="duplicate native completion"):
        check_timeline(path, required_path="dmabuf_fixed")


def test_timeline_rejects_another_consumers_ack(tmp_path: Path) -> None:
    path = tmp_path / "timeline.jsonl"
    rows = _timeline()
    rows[-2]["consumer_instance_id"] = "other-consumer-instance"
    _write(path, rows)

    with pytest.raises(TimelineError, match="one acknowledged consumer instance"):
        check_timeline(path, required_path="dmabuf_fixed")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", []),
        ("tp_rank", False),
        ("writer_epoch", 1),
        ("completed_bytes", True),
        ("device_instance_id", None),
        ("batch_id", "-1"),
        ("operation_id", "nan"),
        ("attempt", []),
        ("path", "unknown"),
        ("monotonic_ns", 0),
    ],
)
def test_timeline_rejects_malformed_completion_identity(
    tmp_path: Path, field: str, value: Any
) -> None:
    path = tmp_path / "timeline.jsonl"
    rows = _timeline()
    rows[0][field] = value
    _write(path, rows)

    with pytest.raises(TimelineError, match="invalid"):
        check_timeline(path, required_path="dmabuf_fixed")


@pytest.mark.parametrize("field", ["consumer_request_id", "restore_attempt_id"])
def test_timeline_rejects_unhashable_restore_identity(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "timeline.jsonl"
    rows = _timeline()
    rows[4][field] = []
    _write(path, rows)

    with pytest.raises(TimelineError, match=f"invalid {field}"):
        check_timeline(path, required_path="dmabuf_fixed")
