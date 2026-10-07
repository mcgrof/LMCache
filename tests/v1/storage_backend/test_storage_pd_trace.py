# SPDX-License-Identifier: Apache-2.0

# Standard
import json

# First Party
from lmcache.v1.storage_backend.storage_pd_trace import trace_storage_pd_event


def test_trace_is_disabled_without_path(monkeypatch, tmp_path):
    path = tmp_path / "disabled.jsonl"
    monkeypatch.delenv("LMCACHE_STORAGE_PD_TRACE_FILE", raising=False)

    trace_storage_pd_event("publication", request_id="request")

    assert not path.exists()


def test_trace_appends_structured_events(monkeypatch, tmp_path):
    path = tmp_path / "timeline.jsonl"
    monkeypatch.setenv("LMCACHE_STORAGE_PD_TRACE_FILE", str(path))

    trace_storage_pd_event("claim", request_id="wire-id", tp_rank=0)
    trace_storage_pd_event("adoption", request_id="wire-id", tp_rank=0)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["claim", "adoption"]
    assert all(row["schema"] == "lmcache.storage_pd.timeline.v1" for row in rows)
    assert all(row["request_id"] == "wire-id" for row in rows)
    assert rows[0]["pid"] == rows[1]["pid"]
    assert rows[0]["process_sequence"] < rows[1]["process_sequence"]
    assert rows[0]["monotonic_ns"] <= rows[1]["monotonic_ns"]
