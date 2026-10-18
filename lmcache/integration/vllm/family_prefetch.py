# SPDX-License-Identifier: Apache-2.0
"""Coordinate declared preloads with child lookup and local-hit attribution.

The caller triggers preloading and later releases its declaration. Children
without local coverage wait before LOOKUP while that warm request is active.
Completion, cancellation and expiry release the wait; actual local admission
drives attribution. The identity is shared with external-load singleflight.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import math
import time

# First Party
from lmcache.integration.vllm.external_singleflight import build_identity

# Reuse the singleflight identity verbatim so a family declaration and
# a child lookup for the same object produce equal keys.
__all__ = ["build_identity", "FamilyPrefetchRegistry"]

_COUNTER_NAMES = (
    "family_prefetch_deferred_total",
    "family_ensure_total",  # declarations recorded
    "family_prefetch_hit_total",  # children that found a declared object
    "family_prefetch_wasted_total",  # declarations no child ever consumed
    "family_prefetch_bytes",  # bytes covered by consumed declarations
)


@dataclass
class _Declaration:
    expected_children: int
    covered_tokens: int
    bytes_per_token: int
    deadline: float
    prefetch_request_id: str | None = None
    prefetch_pending: bool = False
    deferred_ids: set[str] = field(default_factory=set)
    consumed: bool = False
    attached_ids: set[str] = field(default_factory=set)


class FamilyPrefetchRegistry:
    """Scheduler-side declaration map. Not thread-safe by design: the
    scheduler role drives every call from one loop, exactly as the
    singleflight registry does."""

    def __init__(self, timeout_s: float = 30.0, max_pending: int = 4096) -> None:
        if max_pending <= 0:
            raise ValueError("declaration capacity must be positive")
        self._max_pending = max_pending
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("family prefetch timeout must be positive")
        self._timeout_s = float(timeout_s)
        self._decls: dict[str, _Declaration] = {}
        self._counters: dict[str, int] = {name: 0 for name in _COUNTER_NAMES}

    def declare(
        self,
        identity: str,
        expected_children: int,
        covered_tokens: int,
        bytes_per_token: int = 0,
        now: float | None = None,
        prefetch_request_id: str | None = None,
    ) -> bool:
        """Record an exact ancestor, or decline when the table is full.

        Args:
            identity: Exact-prefix identity from build_identity.
            expected_children: Maximum distinct child hits to attribute.
            covered_tokens: Positive ancestor length in tokens.
            bytes_per_token: Nonnegative logical KV size; not backend bytes.
            now: Monotonic timestamp, or None for the system clock.
            prefetch_request_id: Internal engine warm request ID to exclude.

        Returns:
            True when recorded or refreshed, False on bounded-capacity refusal.

        Raises:
            ValueError: Invalid child count, coverage or logical byte size.
        """
        if expected_children <= 0 or covered_tokens <= 0 or bytes_per_token < 0:
            raise ValueError("invalid family declaration geometry")
        now = time.monotonic() if now is None else now
        self._expire(now)
        existing = self._decls.get(identity)
        if existing is not None:
            if (existing.covered_tokens, existing.bytes_per_token) != (
                covered_tokens,
                bytes_per_token,
            ):
                raise ValueError(
                    "declaration geometry changed for an existing identity"
                )
            existing.prefetch_request_id = prefetch_request_id
            existing.prefetch_pending = prefetch_request_id is not None
            existing.expected_children = expected_children
            existing.deadline = now + self._timeout_s
            return True
        if len(self._decls) >= self._max_pending:
            return False
        self._decls[identity] = _Declaration(
            expected_children=expected_children,
            covered_tokens=int(covered_tokens),
            bytes_per_token=int(bytes_per_token),
            deadline=now + self._timeout_s,
            prefetch_request_id=prefetch_request_id,
            prefetch_pending=prefetch_request_id is not None,
        )
        self._counters["family_ensure_total"] += 1
        return True

    def coverage_lengths(self, now: float | None = None) -> tuple[int, ...]:
        """Return live declared ancestor lengths for exact-identity matching.

        Args:
            now: Monotonic timestamp, or None for the system clock.

        Returns:
            Distinct covered prefix lengths, longest first.
        """
        self._expire(time.monotonic() if now is None else now)
        return tuple(
            sorted({d.covered_tokens for d in self._decls.values()}, reverse=True)
        )

    def should_defer(
        self,
        identity: str,
        req_id: str,
        local_tokens: int,
        now: float | None = None,
    ) -> bool:
        """Wait for an active warm request unless the ancestor is already local.

        Args:
            identity: Exact ancestor identity for the child's token namespace.
            req_id: Child's internal engine request identifier.
            local_tokens: Fresh local prefix coverage reported by vLLM.
            now: Monotonic timestamp, or None for the system clock.

        Returns:
            True to defer before LOOKUP, False for normal admission. Local
            coverage permits admission independently of external lookup.
        """
        now = time.monotonic() if now is None else now
        self._expire(now)
        decl = self._decls.get(identity)
        if decl is None or req_id == decl.prefetch_request_id:
            return False
        if local_tokens >= decl.covered_tokens:
            return False
        if not decl.prefetch_pending:
            return False
        if (
            req_id not in decl.deferred_ids
            and len(decl.deferred_ids) < decl.expected_children
        ):
            decl.deferred_ids.add(req_id)
            self._counters["family_prefetch_deferred_total"] += 1
        return True

    def on_request_finished(self, req_id: str) -> None:
        """Stop deferring children after warm completion or cancellation.

        Args:
            req_id: Finished or cancelled internal engine request identifier.
        """
        for decl in self._decls.values():
            if req_id == decl.prefetch_request_id:
                decl.prefetch_pending = False
            decl.deferred_ids.discard(req_id)

    def note_child_hit(
        self, identity: str, req_id: str, now: float | None = None
    ) -> bool:
        """Attribute one fresh local hit to a live ancestor declaration.

        Args:
            identity: Exact prefix found in the local engine cache.
            req_id: Unique child request identifier; repeated polls count once.
            now: Monotonic timestamp, or None for the system clock.

        Returns:
            True for an attributed child, False without a live declaration or
            when its declared child budget has been consumed.
        """
        now = time.monotonic() if now is None else now
        self._expire(now)
        decl = self._decls.get(identity)
        if decl is None or req_id == decl.prefetch_request_id:
            return False
        if req_id in decl.attached_ids:
            return True
        if len(decl.attached_ids) >= decl.expected_children:
            return False
        decl.attached_ids.add(req_id)
        decl.consumed = True
        self._counters["family_prefetch_hit_total"] += 1
        self._counters["family_prefetch_bytes"] += (
            decl.covered_tokens * decl.bytes_per_token
        )
        return True

    def release(self, identity: str) -> None:
        """Retire a declaration and charge it as wasted if no child hit it.

        Args:
            identity: Declaration to retire; an unknown identity is a no-op.
        """
        decl = self._decls.pop(identity, None)
        if decl is not None and not decl.consumed:
            self._counters["family_prefetch_wasted_total"] += 1

    def clear(self) -> None:
        """Reset gate: a cache-domain/generation change invalidates every
        pending declaration. Counters survive; state does not. Pending
        unconsumed declarations are charged wasted on the way out."""
        for decl in self._decls.values():
            if not decl.consumed:
                self._counters["family_prefetch_wasted_total"] += 1
        self._decls.clear()

    def snapshot(self) -> dict[str, int]:
        """Return detached attribution counters and the pending declaration count.

        Returns:
            Logical declared-hit/byte counts and current pending declarations.
        """
        return self._counters | {"pending": len(self._decls)}

    def _expire(self, now: float) -> None:
        expired = [ident for ident, d in self._decls.items() if now >= d.deadline]
        for ident in expired:
            decl = self._decls.pop(ident)
            if not decl.consumed:
                self._counters["family_prefetch_wasted_total"] += 1
