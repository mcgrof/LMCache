# SPDX-License-Identifier: Apache-2.0
"""Exercise exact-prefix identity, cancellation, expiry and local-hit admission."""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.external_singleflight import (
    JOIN,
    LEADER,
    ExternalSingleflightRegistry,
    build_identity,
)

TOKENS = list(range(100, 612))


def _identity(
    model_name: str = "qwen",
    world_size: int = 1,
    cache_salt: str | None = "",
    adapter_id: str | None = None,
    token_ids: list[int] = TOKENS,
    covered_tokens: int = 512,
) -> str:
    return build_identity(
        model_name, world_size, cache_salt, adapter_id, token_ids, covered_tokens
    )


def test_identity_equal_for_equal_prefix_and_namespace() -> None:
    assert _identity() == _identity(token_ids=TOKENS + [7, 8, 9])
    # Same covered prefix, longer tail: identical identity.


@pytest.mark.parametrize(
    "other",
    [
        _identity(model_name="other-model"),
        _identity(world_size=2),
        _identity(cache_salt="tenant-b"),
        _identity(adapter_id="3:my-lora"),
        _identity(covered_tokens=256),
        _identity(token_ids=[t + 1 for t in TOKENS]),
    ],
)
def test_identity_differs_on_any_namespace_dimension(other: str) -> None:
    assert _identity() != other


def test_identity_rejects_short_token_stream() -> None:
    with pytest.raises(ValueError):
        _identity(token_ids=TOKENS[:100], covered_tokens=512)


def _registry(timeout: float = 30.0) -> ExternalSingleflightRegistry:
    return ExternalSingleflightRegistry(timeout_s=timeout)


def test_leader_then_joiners_then_saved_loads() -> None:
    reg = _registry()
    key = _identity()
    assert reg.acquire(key, "A", now=0.0) == LEADER
    assert reg.acquire(key, "B", now=0.1) == JOIN
    assert reg.acquire(key, "C", now=0.2) == JOIN
    # Re-polling the same follower does not double-count the join.
    assert reg.acquire(key, "B", now=0.3) == JOIN
    assert reg.snapshot()["leader_total"] == 1
    assert reg.snapshot()["join_total"] == 2
    reg.mark_published("A")
    # Followers admit on later local hits.
    reg.note_local_hit("B", now=1.1)
    reg.note_local_hit("C", now=1.2)
    assert reg.snapshot()["saved_load_total"] == 2
    assert reg.snapshot()["join_wait_seconds"] == (pytest.approx(2.0))
    assert reg.snapshot()["joined"] == 0


def test_leader_reacquire_is_idempotent() -> None:
    reg = _registry()
    key = _identity()
    assert reg.acquire(key, "A", now=0.0) == LEADER
    assert reg.acquire(key, "A", now=0.5) == LEADER
    assert reg.snapshot()["leader_total"] == 1


def test_published_leader_finish_is_not_a_failure() -> None:
    reg = _registry()
    key = _identity()
    reg.acquire(key, "A", now=0.0)
    reg.acquire(key, "B", now=0.1)
    reg.mark_published("A")
    reg.on_request_finished("A")
    assert reg.snapshot()["leader_failure_total"] == 0
    assert reg.snapshot()["inflight"] == 0


def test_unpublished_leader_finish_with_joiners_counts_failure() -> None:
    # Leader cancellation / load error before publication.
    reg = _registry()
    key = _identity()
    reg.acquire(key, "A", now=0.0)
    reg.acquire(key, "B", now=0.1)
    reg.on_request_finished("A")
    assert reg.snapshot()["leader_failure_total"] == 1
    # The stranded follower proceeds as the next leader — no hang, no
    # infinite loop, and its retry is counted.
    assert reg.acquire(key, "B", now=0.2) == LEADER
    assert reg.snapshot()["follower_retry_total"] == 1


def test_leader_lookup_miss_creates_no_entry() -> None:
    # Entries exist only after a positive external hit decision; a
    # miss path never calls acquire. Registry stays empty.
    reg = _registry()
    assert reg.snapshot()["inflight"] == 0
    reg.on_request_finished("never-seen")
    assert reg.snapshot()["inflight"] == 0


def test_follower_cancellation_leaves_no_residue() -> None:
    reg = _registry()
    key = _identity()
    reg.acquire(key, "A", now=0.0)
    reg.acquire(key, "B", now=0.1)
    reg.on_request_finished("B")
    assert reg.snapshot()["joined"] == 0
    assert reg.snapshot()["saved_load_total"] == 0
    # Leader unaffected.
    assert reg.acquire(key, "A", now=0.2) == LEADER


def test_timeout_expiry_allows_takeover() -> None:
    # Request-timeout gate: an expired unpublished leader never blocks
    # a follower; the taker leads a fresh entry.
    reg = _registry(timeout=5.0)
    key = _identity()
    reg.acquire(key, "A", now=0.0)
    assert reg.acquire(key, "B", now=1.0) == JOIN
    assert reg.acquire(key, "B", now=6.5) == LEADER
    assert reg.snapshot()["leader_failure_total"] == 1
    assert reg.snapshot()["follower_retry_total"] == 1
    assert reg.snapshot()["leader_total"] == 2


def test_generation_reset_clears_inflight_state() -> None:
    # Reset / cache-domain generation change gate.
    reg = _registry()
    key = _identity()
    reg.acquire(key, "A", now=0.0)
    reg.acquire(key, "B", now=0.1)
    reg.clear()
    assert reg.snapshot()["inflight"] == 0
    assert reg.acquire(key, "B", now=0.2) == LEADER
    # Counters survive the reset.
    assert reg.snapshot()["leader_total"] == 2


def test_eviction_between_publish_and_admission() -> None:
    # Leader publishes and finishes; the follower's local retry then
    # MISSES (eviction) and its external lookup hits again with no
    # inflight entry: it becomes the new leader instead of looping.
    reg = _registry()
    key = _identity()
    reg.acquire(key, "A", now=0.0)
    reg.acquire(key, "B", now=0.1)
    reg.mark_published("A")
    reg.on_request_finished("A")
    assert reg.snapshot()["inflight"] == 0
    assert reg.acquire(key, "B", now=0.5) == LEADER
    assert reg.snapshot()["follower_retry_total"] == 1


def test_distinct_identities_do_not_coalesce() -> None:
    reg = _registry()
    assert reg.acquire(_identity(), "A", now=0.0) == LEADER
    assert reg.acquire(_identity(cache_salt="tenant-b"), "B", now=0.1) == LEADER
    assert reg.snapshot()["join_total"] == 0


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timeout_rejected(timeout: float) -> None:
    with pytest.raises(ValueError):
        ExternalSingleflightRegistry(timeout_s=timeout)


def test_namespace_fields_cannot_escape_into_adjacent_fields() -> None:
    assert _identity(model_name="a\0b", cache_salt="c") != _identity(
        model_name="a", cache_salt="b\0c"
    )


def test_snapshot_is_detached() -> None:
    reg = _registry()
    snapshot = reg.snapshot()
    snapshot["leader_total"] = 99
    assert reg.snapshot()["leader_total"] == 0
