# SPDX-License-Identifier: Apache-2.0
"""Proactive family-prefetch accounting: declaration lifecycle,
consumed-vs-wasted attribution, identity reuse, and the fail-open
expiry gate. Registry-level and device-less by design; the fork-gap
sweep on hardware proves the timing prediction."""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.external_singleflight import build_identity
from lmcache.integration.vllm.family_prefetch import FamilyPrefetchRegistry

TOKENS = list(range(100, 612))


def _identity(token_ids: list[int] = TOKENS, cache_salt: str = "") -> str:
    return build_identity("qwen", 1, cache_salt, None, token_ids, 512)


def test_identity_is_the_singleflight_identity() -> None:
    # A declaration and a child lookup for the same object must key equal.
    assert _identity() == _identity(token_ids=TOKENS + [1, 2, 3])
    assert _identity() != _identity(cache_salt="tenant-b")


def test_declare_counts_ensure_once_and_is_idempotent() -> None:
    reg = FamilyPrefetchRegistry()
    ident = _identity()
    reg.declare(ident, expected_children=4, covered_tokens=512, now=0.0)
    reg.declare(ident, expected_children=8, covered_tokens=512, now=1.0)
    assert reg.snapshot()["family_ensure_total"] == 1
    assert reg.snapshot()["pending"] == 1


def test_child_hit_attributed_and_bytes_accumulated() -> None:
    reg = FamilyPrefetchRegistry()
    ident = _identity()
    reg.declare(
        ident, expected_children=4, covered_tokens=512, bytes_per_token=12288, now=0.0
    )
    assert reg.note_child_hit(ident, "r0", now=0.1) is True
    assert reg.note_child_hit(ident, "r1", now=0.1) is True
    # Same child twice: no double count.
    assert reg.note_child_hit(ident, "r0", now=0.1) is True
    assert reg.snapshot()["family_prefetch_hit_total"] == 2
    assert reg.snapshot()["family_prefetch_bytes"] == 2 * 512 * 12288


def test_child_without_declaration_is_ordinary_path() -> None:
    reg = FamilyPrefetchRegistry()
    assert reg.note_child_hit(_identity(), "r0", now=0.0) is False
    assert reg.snapshot()["family_prefetch_hit_total"] == 0


def test_wasted_declaration_charged_on_release() -> None:
    reg = FamilyPrefetchRegistry()
    ident = _identity()
    reg.declare(ident, expected_children=4, covered_tokens=512, now=0.0)
    reg.release(ident)  # no child ever hit
    assert reg.snapshot()["family_prefetch_wasted_total"] == 1


def test_consumed_declaration_not_wasted_on_release() -> None:
    reg = FamilyPrefetchRegistry()
    ident = _identity()
    reg.declare(ident, expected_children=4, covered_tokens=512, now=0.0)
    reg.note_child_hit(ident, "r0", now=0.1)
    reg.release(ident)
    assert reg.snapshot()["family_prefetch_wasted_total"] == 0


def test_expiry_is_fail_open_and_charges_wasted() -> None:
    reg = FamilyPrefetchRegistry(timeout_s=10.0)
    ident = _identity()
    reg.declare(ident, expected_children=4, covered_tokens=512, now=0.0)
    # A child arriving past the deadline gets the ordinary path.
    assert reg.note_child_hit(ident, "r0", now=20.0) is False
    # And a fresh declaration triggers expiry accounting for the stale one.
    reg.declare(
        _identity(cache_salt="x"),
        expected_children=2,
        covered_tokens=512,
        now=21.0,
    )
    assert reg.snapshot()["family_prefetch_wasted_total"] == 1


def test_clear_charges_pending_and_drops_state() -> None:
    reg = FamilyPrefetchRegistry()
    reg.declare(_identity(), expected_children=4, covered_tokens=512, now=0.0)
    reg.declare(
        _identity(cache_salt="b"),
        expected_children=2,
        covered_tokens=512,
        now=0.0,
    )
    reg.clear()
    assert reg.snapshot()["pending"] == 0
    assert reg.snapshot()["family_prefetch_wasted_total"] == 2


def test_zero_expected_children_rejected() -> None:
    reg = FamilyPrefetchRegistry()
    with pytest.raises(ValueError):
        reg.declare(_identity(), expected_children=0, covered_tokens=512)


def test_nonpositive_timeout_rejected() -> None:
    with pytest.raises(ValueError):
        FamilyPrefetchRegistry(timeout_s=0.0)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_timeout(timeout: float) -> None:
    with pytest.raises(ValueError):
        FamilyPrefetchRegistry(timeout_s=timeout)


def test_capacity_declines_without_blocking_or_attributing() -> None:
    reg = FamilyPrefetchRegistry(max_pending=1)
    assert reg.declare(_identity(), 2, 512, now=0.0)
    assert not reg.declare(_identity(cache_salt="other"), 2, 512, now=0.1)
    assert not reg.note_child_hit(_identity(cache_salt="other"), "r", now=0.2)
    assert reg.snapshot()["pending"] == 1
    reg.release(_identity())
    assert reg.declare(_identity(cache_salt="other"), 2, 512, now=0.3)


def test_declared_child_budget_bounds_attribution() -> None:
    reg = FamilyPrefetchRegistry()
    assert reg.declare(_identity(), 1, 512, now=0.0)
    assert reg.note_child_hit(_identity(), "a", now=0.1)
    assert not reg.note_child_hit(_identity(), "b", now=0.1)
    assert reg.snapshot()["family_prefetch_hit_total"] == 1


def test_warm_request_local_hit_cannot_consume_a_declaration() -> None:
    reg = FamilyPrefetchRegistry()
    assert reg.declare(_identity(), 2, 512, now=0.0, prefetch_request_id="warm")
    assert not reg.note_child_hit(_identity(), "warm", now=0.1)
    reg.release(_identity())
    assert reg.snapshot()["family_prefetch_wasted_total"] == 1
    assert reg.snapshot()["family_prefetch_hit_total"] == 0


def test_child_waits_before_lookup_until_warm_finishes() -> None:
    reg = FamilyPrefetchRegistry()
    identity = _identity()
    reg.declare(identity, 2, 512, now=0.0, prefetch_request_id="warm")
    assert not reg.should_defer(identity, "warm", 0, now=0.1)
    assert reg.should_defer(identity, "child", 0, now=0.1)
    assert reg.should_defer(identity, "child", 0, now=0.2)
    assert reg.snapshot()["family_prefetch_deferred_total"] == 1
    reg.on_request_finished("warm")
    assert not reg.should_defer(identity, "child", 512, now=0.3)
    assert reg.snapshot()["family_prefetch_hit_total"] == 0
    assert reg.note_child_hit(identity, "child", now=0.3)
    assert reg.snapshot()["family_prefetch_hit_total"] == 1


def test_cancel_or_timeout_releases_children_without_fabricating_hits() -> None:
    reg = FamilyPrefetchRegistry(timeout_s=1.0)
    identity = _identity()
    reg.declare(identity, 2, 512, now=0.0, prefetch_request_id="warm")
    assert reg.should_defer(identity, "child", 0, now=0.1)
    reg.on_request_finished("warm")
    assert not reg.should_defer(identity, "child", 0, now=0.2)
    assert reg.snapshot()["family_prefetch_hit_total"] == 0
    reg.declare(identity, 2, 512, now=0.3, prefetch_request_id="other-warm")
    assert not reg.should_defer(identity, "child", 0, now=2.0)
    assert reg.snapshot()["family_prefetch_wasted_total"] == 1
    assert reg.coverage_lengths(now=2.0) == ()
