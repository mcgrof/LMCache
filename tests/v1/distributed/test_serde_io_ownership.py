# SPDX-License-Identifier: Apache-2.0
"""Check serde buffer ownership with real L1 allocations and fp8 transforms."""

# Standard
from threading import Event
import os
import select
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import (
    L2CloseOutcome,
    L2LoadOutcome,
    L2StoreOutcome,
)
from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper
from lmcache.v1.distributed.serde import (
    AsyncSerdeProcessor,
    Fp8QuantizationDeserializer,
    Fp8QuantizationSerializer,
)
from lmcache.v1.memory_management import MemoryObj
from tests.v1.distributed.test_controller_io_ownership import OwnershipAdapter
from tests.v1.distributed.test_prefetch_controller import wait_for_condition

pytestmark = pytest.mark.no_shared_allocator


class ObservedDeserializer(Fp8QuantizationDeserializer):
    """Count real decode invocations to detect unsafe decoding of unknown loads."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        self.calls += 1
        super().deserialize(src, dst, key)


class QueryFailureAdapter(OwnershipAdapter):
    """Lose the load ownership proof after accepting a real buffer."""

    def query_load_outcome(self, task_id: int) -> L2LoadOutcome | None:
        raise RuntimeError("injected completion query failure")


class BlockingSerializer(Fp8QuantizationSerializer):
    """Keep a real codec operation pending until its input lifetime is checked."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        self.entered.set()
        assert self.release.wait(5)
        return super().serialize(src, dst, key)


def manager_and_source() -> tuple[L1Manager, ObjectKey, MemoryObj]:
    """Reserve a source containing exactly representable fp8 values."""
    manager = L1Manager(
        L1ManagerConfig(L1MemoryManagerConfig(4 << 20, False, shm_name=""))
    )
    key = ObjectKey(ObjectKey.IntHash2Bytes(71), "serde-ownership", 0)
    _, source = manager.reserve_write(
        [key], [False], MemoryLayoutDesc([torch.Size([4097])], [torch.bfloat16])
    )[key]
    assert source is not None and source.tensor is not None
    source.tensor.fill_(2)
    manager.finish_write([key])
    return manager, key, source


def store_result(wrapper: SerdeL2AdapterWrapper, task: int) -> L2StoreOutcome:
    """Consume one store result before a bounded deadline."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        results = wrapper.pop_completed_store_outcomes()
        if task in results:
            return results[task]
        time.sleep(0.01)
    raise AssertionError("store completion timed out")


@pytest.mark.parametrize("reusable", [True, False])
def test_store_clear_cannot_recycle_inflight_serde_temps(reusable: bool) -> None:
    manager, key, source = manager_and_source()
    adapter = OwnershipAdapter(reusable, True)
    wrapper = SerdeL2AdapterWrapper(
        adapter,
        AsyncSerdeProcessor(Fp8QuantizationSerializer(), ObservedDeserializer()),
        manager,
    )
    try:
        task = wrapper.submit_store_task([key], [source])
        assert adapter.submitted.wait(5)
        with pytest.raises(RuntimeError, match="external I/O"):
            manager.clear(force=True)
        adapter.pending = False
        os.eventfd_write(adapter.get_store_event_fd(), 1)
        assert select.select([wrapper.get_store_event_fd()], [], [], 5)[0]
        if not reusable:
            with pytest.raises(RuntimeError, match="outcomes"):
                wrapper.pop_completed_store_tasks()
        outcome = store_result(wrapper, task)
        assert outcome.buffers_reusable is reusable
        assert outcome.result.is_successful() is reusable
        assert manager.report_status()["external_io_lease_count"] == 0
        assert manager.report_status()["backing_resources_retained"] is not reusable
        assert adapter.objects[0].is_valid() is not reusable
        assert wrapper.close_with_outcome().buffers_reusable is reusable
    finally:
        wrapper.close()
        manager.close()


@pytest.mark.parametrize(
    "reusable,query_failure", [(True, False), (False, False), (False, True)]
)
def test_load_does_not_decode_unknown_io(reusable: bool, query_failure: bool) -> None:
    manager, key, source = manager_and_source()
    adapter = (
        QueryFailureAdapter(True, False)
        if query_failure
        else OwnershipAdapter(True, False)
    )
    decoder = ObservedDeserializer()
    wrapper = SerdeL2AdapterWrapper(
        adapter, AsyncSerdeProcessor(Fp8QuantizationSerializer(), decoder), manager
    )
    try:
        assert store_result(
            wrapper, wrapper.submit_store_task([key], [source])
        ).result.is_successful()
        manager.clear(force=True)
        _, dst = manager.reserve_write(
            [key], [False], MemoryLayoutDesc([torch.Size([4097])], [torch.bfloat16])
        )[key]
        assert dst is not None and dst.tensor is not None
        dst.tensor.fill_(7)
        adapter.reusable = reusable
        task = wrapper.submit_load_task([key], [dst])
        if not reusable:
            assert select.select([wrapper.get_load_event_fd()], [], [], 5)[0]
            with pytest.raises(RuntimeError, match="outcome"):
                wrapper.query_load_result(task)
        deadline = time.monotonic() + 5
        outcome = None
        while outcome is None and time.monotonic() < deadline:
            outcome = wrapper.query_load_outcome(task)
            time.sleep(0.01)
        assert outcome is not None
        assert outcome.buffers_reusable is reusable
        assert outcome.bitmap.test(0) is reusable
        assert decoder.calls == int(reusable)
        assert torch.all(dst.tensor == (2 if reusable else 7))
        assert manager.report_status()["backing_resources_retained"] is not reusable
        assert manager.report_status()["external_io_lease_count"] == 0
    finally:
        wrapper.close()
        manager.close()


def test_codec_intermediate_is_leased_before_submission() -> None:
    manager, key, source = manager_and_source()
    serializer = BlockingSerializer()
    wrapper = SerdeL2AdapterWrapper(
        OwnershipAdapter(True, False),
        AsyncSerdeProcessor(serializer, ObservedDeserializer()),
        manager,
    )
    try:
        task = wrapper.submit_store_task([key], [source])
        assert serializer.entered.wait(5)
        assert manager.report_status()["external_io_lease_count"] == 1
        serializer.release.set()
        assert store_result(wrapper, task).buffers_reusable
        assert manager.report_status()["external_io_lease_count"] == 0
    finally:
        serializer.release.set()
        wrapper.close()
        manager.close()


@pytest.mark.parametrize("reusable", [True, False])
def test_pending_store_close_uses_shutdown_proof(reusable: bool) -> None:
    manager, key, source = manager_and_source()
    adapter = OwnershipAdapter(reusable, True)
    wrapper = SerdeL2AdapterWrapper(
        adapter,
        AsyncSerdeProcessor(Fp8QuantizationSerializer(), ObservedDeserializer()),
        manager,
    )
    try:
        wrapper.submit_store_task([key], [source])
        assert adapter.submitted.wait(5)
        assert wait_for_condition(
            lambda: manager.report_status()["external_io_lease_count"] == 1
        )
        outcome = wrapper.close_with_outcome()
        assert outcome == L2CloseOutcome(reusable)
        assert wrapper.close_with_outcome() is outcome
        assert manager.report_status()["external_io_lease_count"] == 0
        assert manager.report_status()["backing_resources_retained"] is not reusable
        with pytest.raises(RuntimeError, match="closed"):
            wrapper.submit_store_task([key], [source])
    finally:
        wrapper.close()
        manager.close()


@pytest.mark.parametrize("failure", ["deserialize_submission", "serialize_query"])
def test_codec_failure_cannot_release_intermediates_without_proof(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, key, source = manager_and_source()
    processor = AsyncSerdeProcessor(Fp8QuantizationSerializer(), ObservedDeserializer())
    wrapper = SerdeL2AdapterWrapper(OwnershipAdapter(True, False), processor, manager)

    def lose_proof(*args: object) -> None:
        raise RuntimeError("injected codec ownership failure")

    try:
        if failure == "deserialize_submission":
            assert store_result(
                wrapper, wrapper.submit_store_task([key], [source])
            ).buffers_reusable
            manager.clear(force=True)
            _, source = manager.reserve_write(
                [key], [False], MemoryLayoutDesc([torch.Size([4097])], [torch.bfloat16])
            )[key]
            assert source is not None
            monkeypatch.setattr(processor, "submit_deserialize", lose_proof)
            task = wrapper.submit_load_task([key], [source])
            assert select.select([wrapper.get_load_event_fd()], [], [], 5)[0]
            outcome = wrapper.query_load_outcome(task)
            assert outcome is not None and not outcome.buffers_reusable
            assert not outcome.bitmap.test(0)
        else:
            monkeypatch.setattr(processor, "query_serialize_outcome", lose_proof)
            stored = store_result(wrapper, wrapper.submit_store_task([key], [source]))
            assert not stored.buffers_reusable
            assert not stored.result.is_successful()
        assert manager.report_status()["backing_resources_retained"]
        assert manager.report_status()["external_io_lease_count"] == 0
        assert not wrapper.close_with_outcome().buffers_reusable
    finally:
        wrapper.close()
        manager.close()
