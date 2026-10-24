# SPDX-License-Identifier: Apache-2.0
"""CPU admission checks for the opt-in managed raw CUDA context."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector import gds_context
from lmcache.v1.platform.devices.cuda.cache_context import (
    GPUCacheContext,
    retained_managed_raw_contexts,
)


@pytest.mark.parametrize("active", [False, True])
def test_checked_raw_profile_requires_inactive_singleton(
    monkeypatch: pytest.MonkeyPatch, active: bool
) -> None:
    monkeypatch.setattr(
        gds_context, "get_gds_context", lambda: SimpleNamespace(initialized=active)
    )
    if active:
        with pytest.raises(RuntimeError, match="inactive GDS"):
            gds_context.require_inactive_gds()
    else:
        gds_context.require_inactive_gds()


@pytest.mark.parametrize("owners", [[], [object()]])
def test_checked_raw_context_refuses_missing_or_mixed_owners(owners: list) -> None:
    retained = retained_managed_raw_contexts()
    with pytest.raises(ValueError, match="raw CUDA wrappers"):
        GPUCacheContext(owners, managed_raw=True)
    assert retained_managed_raw_contexts() == retained
