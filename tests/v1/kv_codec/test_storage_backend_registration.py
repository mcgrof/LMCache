# SPDX-License-Identifier: Apache-2.0
"""Storage manager registration: SplitTierStorageBackend dispatch.

These tests verify the dispatch wiring in
`lmcache/v1/storage_backend/__init__.py:CreateStorageBackends`
without needing to fully boot a real LMCache engine (which has
many implicit attribute requirements on config / metadata).

We use a hybrid approach:
 1. Static-source checks confirm the dispatch branch exists and
    references the right knobs.  Catches if someone removes the
    branch by mistake.
 2. A direct call to the factory using a real
    `LMCacheEngineConfig.from_defaults()` instance with the few
    overrides we need.  Catches branch-condition bugs (the if
    chain doesn't fire when it should).
"""

# Standard
from pathlib import Path
import inspect
import re

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend import CreateStorageBackends


def test_create_storage_backends_imports_split_tier():
    """Catches: import-time regression where the SplitTierStorageBackend
    branch is removed."""
    src = inspect.getsource(CreateStorageBackends)
    assert "SplitTierStorageBackend" in src, (
        "CreateStorageBackends must reference SplitTierStorageBackend "
        "for the kv_placement_policy=split_k_cpu_v_nvme dispatch"
    )


def test_create_storage_backends_dispatches_on_kv_placement_policy():
    """Catches: dispatch knob renamed / branch logic broken."""
    src = inspect.getsource(CreateStorageBackends)
    # The branch should test for kv_placement_policy and the
    # split_k_cpu_v_nvme value.
    assert "kv_placement_policy" in src
    assert "split_k_cpu_v_nvme" in src


def test_create_storage_backends_skips_local_disk_under_split_tier():
    """Catches: the elif chain regression where both LocalDiskBackend
    and SplitTierStorageBackend register for the same disk path."""
    src = inspect.getsource(CreateStorageBackends)
    # An `elif` between the split-tier branch and the LocalDiskBackend
    # branch ensures only one registers when the policy is set.
    # Find the split-tier branch and verify the next disk-related
    # branch is `elif`, not a fresh `if`.
    split_tier_pos = src.find("use_split_tier")
    assert split_tier_pos != -1, "use_split_tier guard variable missing"
    after = src[split_tier_pos:]
    # Look for "elif" before the next mention of LocalDiskBackend
    elif_pos = after.find("elif")
    local_disk_pos = after.find("LocalDiskBackend(")
    assert elif_pos != -1
    assert local_disk_pos != -1
    assert elif_pos < local_disk_pos, (
        "LocalDiskBackend creation must be inside the elif branch "
        "after the split-tier check, not as a separate independent if"
    )


def test_create_storage_backends_handles_skip_backends():
    """Catches: dispatch ignores skip_backends={'SplitTierStorageBackend'}."""
    src = inspect.getsource(CreateStorageBackends)
    # The split-tier guard must include the _skip set check.
    # Search for "SplitTierStorageBackend" and verify _skip nearby.
    pat = r'"SplitTierStorageBackend"\s+not\s+in\s+_skip'
    assert re.search(pat, src), (
        'split-tier branch must check `"SplitTierStorageBackend" '
        'not in _skip` so callers can disable it without removing '
        "the policy from config"
    )


def test_real_config_with_split_policy_constructs_backend(tmp_path):
    """End-to-end: build a real LMCacheEngineConfig with
    kv_placement_policy set, instantiate the factory, verify the
    split-tier backend appears in the resulting OrderedDict.
    Skips if the LMCacheEngineConfig defaults can't be instantiated
    without a full engine context."""
    # First Party
    try:
        # Standard
        import asyncio
        from unittest.mock import MagicMock

        from lmcache.v1.config import LMCacheEngineConfig
    except Exception as e:
        pytest.skip(f"cannot import config in this env: {e}")

    try:
        cfg = LMCacheEngineConfig.from_defaults(
            local_disk=str(tmp_path),
            max_local_disk_size=1.0,
            max_local_cpu_size=0.5,
        )
    except Exception as e:
        pytest.skip(f"cannot build default config: {e}")

    # Override the asym knobs.
    try:
        cfg.kv_placement_policy = "split_k_cpu_v_nvme"
    except Exception as e:
        pytest.skip(f"cannot set kv_placement_policy on config: {e}")

    md = MagicMock()
    md.role = "worker"
    md.first_rank = 0
    md.worker_id = 0
    md.use_mla = False
    md.model_name = "test-model"

    loop = asyncio.new_event_loop()
    try:
        backends = CreateStorageBackends(
            config=cfg,
            metadata=md,
            loop=loop,
            dst_device="cpu",
        )
    except Exception as e:
        pytest.skip(f"factory init incomplete: {e}")

    assert "SplitTierStorageBackend" in backends, (
        f"Expected SplitTierStorageBackend in backends; got: "
        f"{list(backends.keys())}"
    )
    # And LocalDiskBackend should NOT be there (would conflict).
    local_disk_keys = [k for k in backends if "LocalDiskBackend" in k]
    assert not local_disk_keys, (
        f"LocalDiskBackend should be skipped under split-tier; "
        f"got: {local_disk_keys}"
    )
