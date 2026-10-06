# SPDX-License-Identifier: Apache-2.0
"""Check a single-host storage P/D causal JSONL receipt.

I/O timestamps identify Python's observation of a native completion, not
the instant the completion entered the ring. Observing a completion before
publication or restore completion proves that ordering, but not I/O latency.
"""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from pathlib import Path
from typing import Any, cast
import argparse
import json

SCHEMA = "lmcache.storage_pd.timeline.v1"
RESTORE_STAGES = (
    "claim",
    "adoption",
    "restore_complete",
    "ack_owed",
    "ack_applied",
    "decode",
)
UNREAD_STAGES = ("unread_owed", "unread_applied")
STAGES = ("publication", "ready", *RESTORE_STAGES, *UNREAD_STAGES)
IO_PATHS = {
    "dmabuf_fixed",
    "host_fixed",
    "bounce",
    "regular",
    "uring_cmd",
    "uring_cmd_fixed",
}


class TimelineError(ValueError):
    """The trace cannot prove the storage P/D causal chain."""


def _validate_identity(row: dict[str, Any], line_number: int) -> None:
    text_fields = ["request_id", "writer_epoch"]
    integer_fields = ["tp_rank"]
    if row["event"] != "io_cqe" or row.get("direction") == "read":
        text_fields.extend(("manifest_digest", "namespace_identity"))
        integer_fields.append("advertised_checkpoint_seq")
    if row["event"] in ("claim", "adoption", "restore_complete", "ack_owed") or (
        row["event"] == "io_cqe" and row.get("direction") == "read"
    ):
        text_fields.extend(("restore_attempt_id", "consumer_request_id"))
    if row["event"] in ("claim", "ack_owed", "ack_applied"):
        text_fields.append("consumer_instance_id")
    for field in text_fields:
        if not isinstance(row.get(field), str) or not row[field]:
            raise TimelineError(f"line {line_number} has invalid {field}")
    for field in integer_fields:
        if type(row.get(field)) is not int or row[field] < 0:
            raise TimelineError(f"line {line_number} has invalid {field}")
    if row["event"] != "io_cqe":
        return
    if row.get("direction") not in ("write", "read"):
        raise TimelineError(f"line {line_number} has invalid I/O direction")
    if not isinstance(row.get("path"), str) or row["path"] not in IO_PATHS:
        raise TimelineError(f"line {line_number} has invalid I/O path")
    for field in ("device_instance_id", "batch_id", "operation_id", "attempt"):
        value = row.get(field)
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
            raise TimelineError(f"line {line_number} has invalid {field}")
    if row.get("outcome") != "completed":
        raise TimelineError(f"line {line_number} has non-completed CQE")
    if type(row.get("completed_bytes")) is not int or row["completed_bytes"] <= 0:
        raise TimelineError(f"line {line_number} has invalid completed byte count")


def _receipt_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("request_id"),
        row.get("tp_rank"),
        row.get("writer_epoch"),
        row.get("advertised_checkpoint_seq"),
        row.get("manifest_digest"),
        row.get("namespace_identity"),
    )


def _writer_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("request_id"),
        row.get("tp_rank"),
        row.get("writer_epoch"),
    )


def _one(
    by_event: dict[str, list[dict[str, Any]]],
    event: str,
    identity: tuple[Any, ...],
) -> dict[str, Any]:
    matches = [row for row in by_event[event] if _receipt_identity(row) == identity]
    if len(matches) != 1:
        raise TimelineError(
            f"receipt {identity!r} needs exactly one {event}, found {len(matches)}"
        )
    return matches[0]


def load_timeline(path: Path) -> list[dict[str, Any]]:
    """Load and structurally validate a trace without reordering it."""
    rows: list[dict[str, Any]] = []
    seen_process_sequences: set[tuple[int, int]] = set()
    last_by_process: dict[int, tuple[int, int]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TimelineError(f"line {line_number} is not JSON: {exc}") from exc
            if not isinstance(row, dict) or row.get("schema") != SCHEMA:
                raise TimelineError(f"line {line_number} has the wrong schema")
            event = row.get("event")
            timestamp = row.get("monotonic_ns")
            pid = row.get("pid")
            sequence = row.get("process_sequence")
            if event not in (*STAGES, "io_cqe"):
                raise TimelineError(f"line {line_number} has unknown event {event!r}")
            if not all(type(value) is int for value in (timestamp, pid, sequence)):
                raise TimelineError(f"line {line_number} has invalid clock identity")
            timestamp = cast(int, timestamp)
            pid = cast(int, pid)
            sequence = cast(int, sequence)
            if min(timestamp, pid, sequence) <= 0:
                raise TimelineError(f"line {line_number} has invalid clock identity")
            _validate_identity(row, line_number)
            process_key = (pid, sequence)
            if process_key in seen_process_sequences:
                raise TimelineError(f"line {line_number} duplicates {process_key!r}")
            seen_process_sequences.add(process_key)
            previous = last_by_process.get(pid)
            if previous is not None and (
                sequence <= previous[0] or timestamp < previous[1]
            ):
                raise TimelineError(
                    f"line {line_number} moves process {pid}'s clock backwards"
                )
            last_by_process[pid] = (sequence, timestamp)
            rows.append(row)
    if not rows:
        raise TimelineError("the timeline is empty")
    return rows


def check_timeline(
    path: Path,
    *,
    required_path: str | None = None,
) -> dict[str, Any]:
    """Prove every publication reaches restore/ACK or an unread release.

    Reject unassigned completions and duplicate native attempts. This checks
    the recorded trace; completeness of the workload and submitted I/O still
    requires the request ledger and native submission/completion journal.
    """
    rows = load_timeline(path)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[str(row["event"])].append(row)
    native_attempts: set[tuple[Any, ...]] = set()
    for cqe in by_event["io_cqe"]:
        attempt_identity = (
            cqe["pid"],
            int(cqe["device_instance_id"]),
            int(cqe["operation_id"]),
            int(cqe["attempt"]),
        )
        if attempt_identity in native_attempts:
            raise TimelineError(f"duplicate native completion: {attempt_identity!r}")
        native_attempts.add(attempt_identity)

    publications: dict[tuple[Any, ...], dict[str, Any]] = {}
    writer_publications: set[tuple[Any, ...]] = set()
    for publication in by_event["publication"]:
        identity = _receipt_identity(publication)
        if any(value in (None, "", -1) for value in identity):
            raise TimelineError(f"publication has an incomplete identity: {identity!r}")
        if identity in publications:
            raise TimelineError(f"duplicate publication: {identity!r}")
        writer_identity = _writer_identity(publication)
        if writer_identity in writer_publications:
            raise TimelineError(
                f"writer request/rank published more than once: {writer_identity!r}"
            )
        publications[identity] = publication
        writer_publications.add(writer_identity)
    if not publications:
        raise TimelineError("the timeline contains no publication")

    checked_attempts: list[dict[str, Any]] = []
    checked_unread: list[dict[str, Any]] = []
    used_stage_rows: set[tuple[int, int]] = set()
    used_cqe_rows: set[tuple[int, int]] = set()
    for identity, publication in publications.items():
        ready = _one(by_event, "ready", identity)
        used_stage_rows.add((publication["pid"], publication["process_sequence"]))
        used_stage_rows.add((ready["pid"], ready["process_sequence"]))

        writer_cqes = [
            row
            for row in by_event["io_cqe"]
            if row.get("direction") == "write"
            and _writer_identity(row) == _writer_identity(publication)
        ]
        reader_cqes = [
            row
            for row in by_event["io_cqe"]
            if row.get("direction") == "read" and _receipt_identity(row) == identity
        ]
        if not writer_cqes:
            raise TimelineError(f"receipt {identity!r} has no write CQE")
        used_cqe_rows.update(
            (row["pid"], row["process_sequence"])
            for row in (*writer_cqes, *reader_cqes)
        )
        if required_path is not None and not any(
            cqe.get("path") == required_path for cqe in writer_cqes
        ):
            raise TimelineError(
                f"receipt {identity!r} has no write CQE on {required_path!r}"
            )

        claim_rows = [
            row for row in by_event["claim"] if _receipt_identity(row) == identity
        ]
        unread_rows = [
            row for row in by_event["unread_owed"] if _receipt_identity(row) == identity
        ]
        if bool(claim_rows) == bool(unread_rows):
            raise TimelineError(
                f"receipt {identity!r} has neither or both restore and unread lanes"
            )

        first_write = min(row["monotonic_ns"] for row in writer_cqes)
        last_write = max(row["monotonic_ns"] for row in writer_cqes)
        common_ordered = (
            ("last_write_cqe", last_write),
            ("publication", publication["monotonic_ns"]),
            ("ready", ready["monotonic_ns"]),
        )

        if unread_rows:
            stages = {event: _one(by_event, event, identity) for event in UNREAD_STAGES}
            used_stage_rows.update(
                (row["pid"], row["process_sequence"]) for row in stages.values()
            )
            if reader_cqes:
                raise TimelineError(f"unread receipt {identity!r} has read CQEs")
            unread_ordered = (
                *common_ordered,
                ("unread_owed", stages["unread_owed"]["monotonic_ns"]),
            )
            for (left_name, left), (right_name, right) in zip(
                unread_ordered, unread_ordered[1:], strict=False
            ):
                if left >= right:
                    raise TimelineError(
                        f"receipt {identity!r} violates {left_name} < "
                        f"{right_name}: {left} >= {right}"
                    )
            if (
                stages["unread_owed"]["monotonic_ns"]
                > stages["unread_applied"]["monotonic_ns"]
            ):
                raise TimelineError(
                    f"receipt {identity!r} applied unread release before it was owed"
                )
            checked_unread.append(
                {
                    "request_id": identity[0],
                    "tp_rank": identity[1],
                    "writer_epoch": identity[2],
                    "checkpoint_seq": identity[3],
                    "manifest_digest": identity[4],
                    "namespace_identity": identity[5],
                    "write_cqes": len(writer_cqes),
                    "write_bytes": sum(row["completed_bytes"] for row in writer_cqes),
                    "required_path_write_bytes": sum(
                        row["completed_bytes"]
                        for row in writer_cqes
                        if row.get("path") == required_path
                    ),
                    "first_write_ns": first_write,
                    "unread_applied_ns": stages["unread_applied"]["monotonic_ns"],
                }
            )
            continue

        stages = {event: _one(by_event, event, identity) for event in RESTORE_STAGES}
        used_stage_rows.update(
            (row["pid"], row["process_sequence"]) for row in stages.values()
        )
        if not reader_cqes:
            raise TimelineError(f"restored receipt {identity!r} has no read CQE")
        if required_path is not None and not any(
            cqe.get("path") == required_path for cqe in reader_cqes
        ):
            raise TimelineError(
                f"receipt {identity!r} has no read CQE on {required_path!r}"
            )

        attempt_stages = (
            stages["claim"],
            stages["adoption"],
            stages["restore_complete"],
            stages["ack_owed"],
        )
        attempt_ids = {row.get("restore_attempt_id") for row in attempt_stages}
        if len(attempt_ids) != 1 or not all(
            isinstance(value, str) and value for value in attempt_ids
        ):
            raise TimelineError(
                f"receipt {identity!r} does not have one restore attempt: "
                f"{attempt_ids!r}"
            )
        attempt_id = next(iter(attempt_ids))
        consumer_ids = {row.get("consumer_request_id") for row in attempt_stages}
        if len(consumer_ids) != 1 or not all(
            isinstance(value, str) and value for value in consumer_ids
        ):
            raise TimelineError(
                f"receipt {identity!r} does not have one consumer request: "
                f"{consumer_ids!r}"
            )
        consumer_request_id = next(iter(consumer_ids))
        consumer_instances = {
            stages[event].get("consumer_instance_id")
            for event in ("claim", "ack_owed", "ack_applied")
        }
        if len(consumer_instances) != 1 or not all(
            isinstance(value, str) and value for value in consumer_instances
        ):
            raise TimelineError(
                f"receipt {identity!r} does not have one acknowledged consumer instance"
            )
        for cqe in reader_cqes:
            if (
                cqe.get("restore_attempt_id") != attempt_id
                or cqe.get("consumer_request_id") != consumer_request_id
            ):
                raise TimelineError(
                    f"receipt {identity!r} has a read outside its restore attempt"
                )

        first_read = min(row["monotonic_ns"] for row in reader_cqes)
        last_read = max(row["monotonic_ns"] for row in reader_cqes)
        restore_ordered = (
            *common_ordered,
            ("claim", stages["claim"]["monotonic_ns"]),
            ("adoption", stages["adoption"]["monotonic_ns"]),
            ("first_read_cqe", first_read),
        )
        for (left_name, left), (right_name, right) in zip(
            restore_ordered, restore_ordered[1:], strict=False
        ):
            if left >= right:
                raise TimelineError(
                    f"receipt {identity!r} violates {left_name} < {right_name}: "
                    f"{left} >= {right}"
                )
        restore_ns = stages["restore_complete"]["monotonic_ns"]
        if last_read >= restore_ns:
            raise TimelineError(f"receipt {identity!r} completed restore before reads")
        if restore_ns >= stages["decode"]["monotonic_ns"]:
            raise TimelineError(f"receipt {identity!r} decoded before restore")
        if restore_ns >= stages["ack_owed"]["monotonic_ns"]:
            raise TimelineError(f"receipt {identity!r} owed ACK before restore")
        if stages["ack_owed"]["monotonic_ns"] > stages["ack_applied"]["monotonic_ns"]:
            raise TimelineError(f"receipt {identity!r} applied ACK before it was owed")

        checked_attempts.append(
            {
                "request_id": identity[0],
                "tp_rank": identity[1],
                "writer_epoch": identity[2],
                "checkpoint_seq": identity[3],
                "manifest_digest": identity[4],
                "namespace_identity": identity[5],
                "consumer_request_id": consumer_request_id,
                "restore_attempt_id": attempt_id,
                "write_cqes": len(writer_cqes),
                "read_cqes": len(reader_cqes),
                "write_bytes": sum(row["completed_bytes"] for row in writer_cqes),
                "read_bytes": sum(row["completed_bytes"] for row in reader_cqes),
                "required_path_write_bytes": sum(
                    row["completed_bytes"]
                    for row in writer_cqes
                    if row.get("path") == required_path
                ),
                "required_path_read_bytes": sum(
                    row["completed_bytes"]
                    for row in reader_cqes
                    if row.get("path") == required_path
                ),
                "first_write_ns": first_write,
                "decode_ns": stages["decode"]["monotonic_ns"],
            }
        )

    orphan_stages = [
        row
        for event in STAGES
        for row in by_event[event]
        if (row["pid"], row["process_sequence"]) not in used_stage_rows
    ]
    if orphan_stages:
        raise TimelineError(f"timeline has {len(orphan_stages)} orphan stage event(s)")
    orphan_cqes = [
        row
        for row in by_event["io_cqe"]
        if (row["pid"], row["process_sequence"]) not in used_cqe_rows
    ]
    if orphan_cqes:
        raise TimelineError(f"timeline has {len(orphan_cqes)} orphan CQE event(s)")
    return {
        "schema": SCHEMA,
        "result": "PASS",
        "publication_count": len(publications),
        "restore_count": len(checked_attempts),
        "unread_count": len(checked_unread),
        "event_count": len(rows),
        "required_path": required_path,
        "attempts": checked_attempts,
        "unread": checked_unread,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--require-path")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = check_timeline(args.timeline, required_path=args.require_path)
    except (OSError, TimelineError) as exc:
        report = {"schema": SCHEMA, "result": "FAIL", "error": str(exc)}
        if args.json_out is not None:
            args.json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
