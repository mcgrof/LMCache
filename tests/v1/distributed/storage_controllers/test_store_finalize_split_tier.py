# SPDX-License-Identifier: Apache-2.0
"""Split-tier store finalization is tied to the split-tier adapter.

On a successful L2 store completion, ``StoreController`` additionally
deletes the original logical L1 entries of split-tier keys (the full
K+V staging is redundant once K is mirrored to the K child and V is on
L2). That deletion is only sound when the completing adapter IS the
split-tier wrapper: with multiple adapters configured, another
adapter's success -- e.g. a peer adapter acknowledging instantly with
zero bytes -- proves nothing about the K/V mirroring and must not
trigger the deletion.

These tests drive the finalize transition directly with mocked
collaborators (the full submit loop needs a CUDA-pinned L1 pool; the
gate under test is pure control flow on the completion record).
"""

# Future
from __future__ import annotations

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_controllers.store_controller import (
    InFlightStoreTask,
    StoreController,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
)
from lmcache.v1.distributed.storage_placement import SplitTierManifest


def _make_key(chunk_hash: bytes) -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name="finalize-test",
        kv_rank=0,
        cache_salt="",
    )


def _build_controller(manifest: SplitTierManifest) -> StoreController:
    l1 = MagicMock()
    adapter = MagicMock()
    descriptor = MagicMock(spec=AdapterDescriptor)
    descriptor.index = 0
    descriptor.type_name = "mock"
    ctrl = StoreController(
        l1_manager=l1,
        l2_adapters=[adapter],
        adapter_descriptors=[descriptor],
        policy=DefaultStorePolicy(),
        split_tier_manifest=manifest,
    )
    return ctrl


def _finalize_one(
    ctrl: StoreController,
    task: InFlightStoreTask,
) -> None:
    """Record the task as in flight and drive the finalize transition."""
    task_key = (task.adapter_index, 7)
    ctrl._in_flight_tasks[task_key] = task
    ctrl._status_in_flight_count += 1
    task.l2_store_result = True
    ctrl._advance_request(task_key, task)


def test_split_tier_adapter_success_deletes_tracked_logicals() -> None:
    """The split-tier wrapper's own completion deletes the redundant
    logical staging entries of tracked keys."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\x11" * 32)
    manifest.register_pending(logical)

    ctrl = _build_controller(manifest)
    _finalize_one(
        ctrl,
        InFlightStoreTask(
            adapter_index=0,
            keys=[logical],
            read_locked_keys=[logical],
            adapter_is_split_tier=True,
        ),
    )
    deleted = [
        k for call in ctrl._l1_manager.delete.call_args_list for k in call.args[0]
    ]
    assert logical in deleted


def test_non_split_tier_adapter_success_leaves_logicals_alone() -> None:
    """Another adapter's success (instant peer-adapter ack) must NOT
    delete the logical entries -- the split-tier store for those keys
    may still be in flight and reading from the staging."""
    manifest = SplitTierManifest()
    logical = _make_key(b"\x22" * 32)
    manifest.register_pending(logical)

    ctrl = _build_controller(manifest)
    _finalize_one(
        ctrl,
        InFlightStoreTask(
            adapter_index=0,
            keys=[logical],
            read_locked_keys=[logical],
            adapter_is_split_tier=False,
        ),
    )
    deleted = [
        k for call in ctrl._l1_manager.delete.call_args_list for k in call.args[0]
    ]
    assert logical not in deleted
    # The manifest entry is untouched -- lifecycle stays owned by the
    # split-tier adapter's own task.
    assert manifest.lookup(logical) is not None
