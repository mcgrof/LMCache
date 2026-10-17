# SPDX-License-Identifier: Apache-2.0
"""Preserve the allocator interface when the optional CXL driver lacks receipts."""

# Standard
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import AllocationReceipt, MemoryFormat


@pytest.fixture
def backend(monkeypatch):
    # Only the allocation interface is exercised; no CXL driver is constructed.
    maru = ModuleType("maru")
    maru.MaruConfig = object
    maru.MaruHandler = object
    adapter = ModuleType("maru_lmcache")
    adapter.CxlMemoryAdapter = object
    monkeypatch.setitem(sys.modules, "maru", maru)
    monkeypatch.setitem(sys.modules, "maru_lmcache", adapter)
    source = Path(__file__).parents[3] / "lmcache/v1/storage_backend/maru_backend.py"
    spec = spec_from_file_location("maru_receipt_contract", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.MaruBackend.__new__(module.MaruBackend)
    instance.memory_allocator = MagicMock()
    yield instance


@pytest.mark.parametrize("batched", [False, True])
def test_receipts_are_rejected_before_allocation(backend, batched):
    shape = torch.Size([4])
    receipt = AllocationReceipt()
    with pytest.raises(NotImplementedError, match="allocation receipts"):
        if batched:
            backend.batched_allocate(
                shape, torch.float32, 1, allocation_receipts=[receipt]
            )
        else:
            backend.allocate(shape, torch.float32, allocation_receipt=receipt)
    backend.memory_allocator.allocate.assert_not_called()
    backend.memory_allocator.batched_allocate.assert_not_called()


def test_ordinary_allocation_keeps_legacy_dispatch(backend):
    shape = torch.Size([4])
    backend.memory_allocator.allocate.return_value = None
    assert backend.allocate(shape, torch.float32) is None
    backend.memory_allocator.allocate.assert_called_once_with(
        shape, torch.float32, MemoryFormat.KV_2LTD
    )
    result = backend.batched_allocate(shape, torch.float32, 2)
    assert result is backend.memory_allocator.batched_allocate.return_value
    backend.memory_allocator.batched_allocate.assert_called_once_with(
        shape, torch.float32, 2, MemoryFormat.KV_2LTD
    )
