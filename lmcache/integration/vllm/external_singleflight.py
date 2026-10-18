# SPDX-License-Identifier: Apache-2.0
"""Coalesce external loads until vLLM publishes a locally reusable prefix.

The scheduler loop owns this registry. Followers retry the ordinary local
prefix lookup; a registry entry never substitutes for a local cache hit.
Timeout, cancellation and cache reset remove deferral without assuming that
an external load succeeded. Backend lookup/prefetch work is independent.
"""

# Standard
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import math
import time

LEADER = "leader"
JOIN = "join"


def build_identity(
    model_name: str,
    world_size: int,
    cache_salt: str | None,
    adapter_id: str | None,
    token_ids: Iterable[int],
    covered_tokens: int,
) -> str:
    """Hash a covered token prefix and its complete text-cache namespace.

    Args:
        model_name: Connector's resolved cache model name.
        world_size: Number of parallel engine workers; must be positive.
        cache_salt: Tenant cache namespace, or None for the default namespace.
        adapter_id: Adapter identity, or None for the base model.
        token_ids: Nonnegative token IDs, with at least covered_tokens entries.
        covered_tokens: Positive number of externally available prefix tokens.

    Returns:
        SHA-256 identity, independent of tokens after the covered prefix.

    Raises:
        ValueError: If coverage, world size or token stream is invalid.
    """
    if covered_tokens <= 0 or world_size <= 0:
        raise ValueError("coverage and world size must be positive")
    header = json.dumps(
        [model_name, world_size, cache_salt or "", adapter_id, covered_tokens],
        separators=(",", ":"),
    ).encode()
    hasher = hashlib.sha256(header)
    count = 0
    for token in token_ids:
        if count == covered_tokens:
            break
        hasher.update(token.to_bytes(4, "little", signed=False))
        count += 1
    if count != covered_tokens:
        raise ValueError("token stream is shorter than covered prefix")
    return hasher.hexdigest()


@dataclass
class _InflightEntry:
    leader_req_id: str
    deadline: float
    published: bool = False


class ExternalSingleflightRegistry:
    """Track exact-prefix leaders on the scheduler's single calling thread."""

    def __init__(self, timeout_s: float = 30.0) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("singleflight timeout must be finite and positive")
        self._timeout_s = timeout_s
        self._entries: dict[str, _InflightEntry] = {}
        self._leaders: dict[str, str] = {}
        # Follower accounting survives leader retirement until the follower
        # retries, finishes or the cache generation changes.
        self._joined: dict[str, tuple[str, float]] = {}
        self._counters: dict[str, int | float] = {
            "leader_total": 0,
            "join_total": 0,
            "saved_load_total": 0,
            "leader_failure_total": 0,
            "follower_retry_total": 0,
            "join_wait_seconds": 0.0,
        }

    def acquire(self, identity: str, req_id: str, now: float | None = None) -> str:
        """Choose LEADER to load or JOIN to defer without allocating.

        Args:
            identity: Exact prefix identity returned by build_identity.
            req_id: Unique live request identifier.
            now: Monotonic timestamp, or None to read the system clock.

        Returns:
            LEADER for a new, existing or replacement leader; JOIN otherwise.
        """
        now = time.monotonic() if now is None else now
        entry = self._entries.get(identity)
        if entry is not None and entry.leader_req_id == req_id:
            return LEADER
        if entry is not None and now < entry.deadline:
            if req_id not in self._joined:
                self._joined[req_id] = (identity, now)
                self._counters["join_total"] += 1
            return JOIN
        if entry is not None:
            if not entry.published:
                self._counters["leader_failure_total"] += 1
            self._remove_entry(identity)
        if req_id in self._joined:
            self._counters["follower_retry_total"] += 1
            self._joined.pop(req_id)
        self._entries[identity] = _InflightEntry(req_id, now + self._timeout_s)
        self._leaders[req_id] = identity
        self._counters["leader_total"] += 1
        return LEADER

    def mark_published(self, req_id: str) -> None:
        """Record successful completion for a leader without releasing followers.

        Args:
            req_id: Leader request that completed successfully.
        """
        identity = self._leaders.get(req_id)
        if identity is not None:
            self._entries[identity].published = True

    def note_local_hit(self, req_id: str, now: float | None = None) -> None:
        """Count a deferred follower admitted through a fresh local cache hit.

        Args:
            req_id: Request admitted without an external load.
            now: Monotonic timestamp, or None to read the system clock.
        """
        joined = self._joined.pop(req_id, None)
        if joined is not None:
            now = time.monotonic() if now is None else now
            self._counters["saved_load_total"] += 1
            self._counters["join_wait_seconds"] += max(0.0, now - joined[1])
            entry = self._entries.get(joined[0])
            if entry is not None:
                entry.published = True

    def on_request_finished(self, req_id: str) -> None:
        """Release leader or follower state after completion or cancellation.

        Args:
            req_id: Finished or cancelled request identifier.
        """
        identity = self._leaders.get(req_id)
        if identity is not None:
            if not self._entries[identity].published:
                self._counters["leader_failure_total"] += 1
            self._remove_entry(identity)
        self._joined.pop(req_id, None)

    def clear(self) -> None:
        """Invalidate all deferrals on a cache-generation change, keeping counts."""
        self._entries.clear()
        self._leaders.clear()
        self._joined.clear()

    def snapshot(self) -> dict[str, int | float]:
        """Return detached cumulative counts and current scheduler-state sizes.

        Returns:
            Counts, join wait seconds, and inflight/joined request counts.
        """
        return self._counters | {
            "inflight": len(self._entries),
            "joined": len(self._joined),
        }

    def _remove_entry(self, identity: str) -> None:
        entry = self._entries.pop(identity)
        self._leaders.pop(entry.leader_req_id)
