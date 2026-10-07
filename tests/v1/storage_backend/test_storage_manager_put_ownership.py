# SPDX-License-Identifier: Apache-2.0
"""A refused backend must not strand the manager's staging references."""

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.storage_manager import StorageManager


class _OriginalAllocator:
    pass


class _CopiedAllocator:
    pass


class _LaterAllocator:
    pass


@pytest.mark.parametrize("failure", ["dispatch", "copy", "none"])
def test_batched_put_releases_every_acquired_group(monkeypatch, failure):
    original = SimpleNamespace(ref_count_down=Mock())
    copied = SimpleNamespace(ref_count_down=Mock())
    primary_allocator = _OriginalAllocator()
    copied_allocator = _CopiedAllocator()
    later_allocator = _LaterAllocator()
    dispatch_error = OSError("backend refused the batch")
    copied_backend = SimpleNamespace(
        get_allocator_backend=lambda: copied_allocator,
        batched_submit_put_task=Mock(
            side_effect=dispatch_error if failure == "dispatch" else None,
            return_value=None,
        ),
    )
    later_backend = SimpleNamespace(
        get_allocator_backend=lambda: later_allocator,
        batched_submit_put_task=Mock(return_value=None),
    )
    manager = object.__new__(StorageManager)
    manager.allocator_backend = cast(Any, primary_allocator)
    manager.storage_backends = cast(Any, {"copied": copied_backend})
    if failure == "copy":
        manager.storage_backends["later"] = cast(Any, later_backend)
    manager._bypass_lock = threading.Lock()
    manager._bypassed_backends = set()
    manager.internal_copy_stream = cast(Any, None)

    def allocate(allocator, keys, objects, stream):
        if allocator is later_allocator:
            raise OSError("later staging allocation failed")
        assert allocator is copied_allocator
        assert objects == [original]
        return keys, [copied]

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.allocate_and_copy_objects",
        allocate,
    )
    if failure == "none":
        manager.batched_put(cast(Any, ["key"]), cast(Any, [original]))
    else:
        with pytest.raises(OSError, match="refused|allocation failed"):
            manager.batched_put(cast(Any, ["key"]), cast(Any, [original]))

    original.ref_count_down.assert_called_once_with()
    copied.ref_count_down.assert_called_once_with()
    copied_backend.batched_submit_put_task.assert_called_once()
    later_backend.batched_submit_put_task.assert_not_called()
