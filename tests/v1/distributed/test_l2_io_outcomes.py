# SPDX-License-Identifier: Apache-2.0
"""Check ownership propagation through adapter compatibility and wrappers."""

# Standard
from unittest.mock import create_autospec

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2CloseOutcome,
    L2LoadOutcome,
    L2StoreOutcome,
)
from lmcache.v1.distributed.l2_adapters.fault_inject_l2_adapter import (
    FaultInjectL2Adapter,
)


@pytest.mark.parametrize("reusable", [False, True])
def test_fault_wrapper_preserves_all_ownership_outcomes(reusable: bool) -> None:
    inner = create_autospec(L2AdapterInterface, instance=True)
    store = L2StoreOutcome(L2StoreResult(False, 0), reusable)
    load = L2LoadOutcome(Bitmap(1), reusable)
    load.bitmap.set(0)
    close = L2CloseOutcome(reusable)
    inner.pop_completed_store_outcomes.return_value = {7: store}
    inner.submit_load_task.return_value = 8
    inner.query_load_outcome.side_effect = [None, load, None]
    inner.close_with_outcome.return_value = close
    inner.may_report_unresolved_io = True
    wrapper = FaultInjectL2Adapter(inner, rate=0, seed=0, gap_indices=(0,))
    key = ObjectKey(ObjectKey.IntHash2Bytes(1), "model", 0)
    assert wrapper.submit_load_task([key], []) == 8
    assert wrapper.query_load_outcome(8) is None
    assert wrapper.query_load_outcome(8) is load
    assert load.buffers_reusable is reusable and not load.bitmap.test(0)
    assert wrapper.query_load_outcome(8) is None
    assert wrapper.pop_completed_store_outcomes() == {7: store}
    assert wrapper.may_report_unresolved_io
    assert wrapper.close_with_outcome() is close
    inner.close_with_outcome.assert_called_once()
