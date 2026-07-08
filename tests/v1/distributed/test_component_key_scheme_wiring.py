# SPDX-License-Identifier: Apache-2.0
"""
M1 wiring proof (pure-Python): the per-engine ``ComponentKeyScheme`` is
derived once from the adapter configs and threaded consistently into the
``SplitTierManifest`` so its K-child side-set matches the key the wrapper
actually allocates.

Companion to ``test_component_key_scheme.py`` (the key-layer primitive)
and the wrapper-level proofs in ``test_split_tier_early_release.py``.

The byte-equality round-trip is proven directly on the serde in
``test_asym_bytethrough_k16_v8_v_only.py``; the full StorageManager
byte-returning round-trip is CUDA-gated and runs on a GPU host.
"""

# Future
from __future__ import annotations

# Standard
import hashlib
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_placement import (
    ComponentKeyScheme,
    SplitTierManifest,
    SplitTierState,
    derive_component_key,
    derive_component_key_scheme,
)

_LEGACY = ComponentKeyScheme.COMPUTED_LEGACY
_RAW = ComponentKeyScheme.RAW_UNIT


def _logical(seed: bytes = b"seed") -> ObjectKey:
    return ObjectKey(
        chunk_hash=hashlib.sha256(seed).digest(), model_name="m", kv_rank=0
    )


def _adapter(serde_type: object) -> SimpleNamespace:
    """A stub L2 adapter config exposing ``serde_config.type``."""
    sc = None if serde_type is None else SimpleNamespace(type=serde_type)
    return SimpleNamespace(serde_config=sc)


# =============================================================================
# derive_component_key_scheme: the once-per-engine config inspection
# =============================================================================


def test_bytethrough_serde_yields_raw_unit() -> None:
    cfgs = [_adapter("asym_bytethrough_k16_v8_v_only")]
    assert derive_component_key_scheme(cfgs) is _RAW


def test_scale_aware_serde_yields_computed_legacy() -> None:
    assert derive_component_key_scheme([_adapter("asym_k16_v8_v_only")]) is _LEGACY
    assert derive_component_key_scheme([_adapter("fp8")]) is _LEGACY


def test_no_serde_yields_computed_legacy() -> None:
    assert derive_component_key_scheme([_adapter(None)]) is _LEGACY


def test_empty_adapter_list_yields_computed_legacy() -> None:
    assert derive_component_key_scheme([]) is _LEGACY


def test_uniform_bytethrough_set_accepted() -> None:
    cfgs = [_adapter("asym_bytethrough_k16_v8_v_only")] * 3
    assert derive_component_key_scheme(cfgs) is _RAW


def test_mixed_scheme_set_rejected() -> None:
    cfgs = [
        _adapter("asym_bytethrough_k16_v8_v_only"),
        _adapter("asym_k16_v8_v_only"),
    ]
    with pytest.raises(ValueError, match="Incompatible.*component-key schemes"):
        derive_component_key_scheme(cfgs)


# =============================================================================
# Manifest K-child side-set stays consistent with the engine scheme
# (the cross-module coupling: is_k_child_key must match the key the
#  wrapper allocates, or the StoreController mis-routes the K child to L2)
# =============================================================================


def test_manifest_raw_unit_side_set_matches_raw_unit_k_child() -> None:
    manifest = SplitTierManifest(component_key_scheme=_RAW)
    logical = _logical()
    manifest.register_pending(logical)

    raw_k = derive_component_key(logical, "k", scheme=_RAW)
    legacy_k = derive_component_key(logical, "k", scheme=_LEGACY)

    # The side-set holds the RAW_UNIT K child, NOT the legacy one.
    assert manifest.is_k_child_key(raw_k) is True
    assert manifest.is_k_child_key(legacy_k) is False


def test_manifest_legacy_side_set_matches_legacy_k_child() -> None:
    manifest = SplitTierManifest(component_key_scheme=_LEGACY)
    logical = _logical()
    manifest.register_pending(logical)

    assert manifest.is_k_child_key(derive_component_key(logical, "k")) is True
    assert (
        manifest.is_k_child_key(derive_component_key(logical, "k", scheme=_RAW))
        is False
    )


def test_manifest_default_scheme_is_legacy() -> None:
    manifest = SplitTierManifest()
    logical = _logical()
    manifest.register_pending(logical)
    # Default (no scheme arg) is byte-identical to the pre-wiring behavior.
    assert manifest.is_k_child_key(derive_component_key(logical, "k")) is True


def test_manifest_drop_discards_raw_unit_side_set_entry() -> None:
    manifest = SplitTierManifest(component_key_scheme=_RAW)
    logical = _logical()
    gen = manifest.register_pending(logical)
    raw_k = derive_component_key(logical, "k", scheme=_RAW)
    assert manifest.is_k_child_key(raw_k) is True
    # drop must discard the SAME (RAW_UNIT) key it added, else the
    # side-set leaks membership forever.
    manifest.mark_complete(logical, gen)
    manifest.drop(logical, gen)
    assert manifest.is_k_child_key(raw_k) is False


# =============================================================================
# Generation race: a late stale mark_complete must not complete a
# superseded entry (part of the M1 closure; scheme-independent)
# =============================================================================


def test_late_stale_mark_complete_does_not_complete_superseded_entry() -> None:
    manifest = SplitTierManifest(component_key_scheme=_RAW)
    logical = _logical()

    g1 = manifest.register_pending(logical)
    manifest.mark_complete(logical, g1)  # g1 now stale-COMPLETE
    # A new store reclaims the ended generation.
    g2 = manifest.register_pending(logical)
    assert g2 != g1
    assert manifest.lookup(logical) is SplitTierState.STORE_IN_FLIGHT

    # The late completion for the OLD generation must be refused, not
    # silently applied to g2's in-flight entry.
    with pytest.raises(ValueError):
        manifest.mark_complete(logical, g1)

    # g2 is still in flight; no phantom COMPLETE leaked.
    assert manifest.lookup(logical) is SplitTierState.STORE_IN_FLIGHT
    counts = manifest.state_counts()
    assert counts[SplitTierState.COMPLETE] == 0
    assert counts[SplitTierState.STORE_IN_FLIGHT] == 1
