# SPDX-License-Identifier: Apache-2.0
"""Failed waits and cancellation do not fence native I/O or release its arena."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock
import asyncio
import gc
import threading
import time
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.plugins import rust_raw_block_backend as plugin
from lmcache.v1.storage_backend.raw_block import NativeQuiescence, encode_legacy_key
from lmcache.v1.storage_backend.storage_manager import StorageManager
from tests.v1.storage_backend.test_rust_raw_block_backend import (
    _install_fake_raw_block_device,
    _make_byte_obj,
    _make_raw_block_backend,
)

pytestmark = pytest.mark.no_shared_allocator


@pytest.fixture
def loop_in_thread():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


@pytest.fixture
def backend(monkeypatch, loop_in_thread):
    _install_fake_raw_block_device(monkeypatch, size_bytes=64 * 1024 * 1024)
    instance = _make_raw_block_backend(
        "/tmp/ownership-device", AdHocMemoryAllocator(device="cpu"), loop_in_thread
    )
    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    try:
        yield instance
    finally:
        if not instance._closed_once:
            instance.close()
        del plugin._RETAINED_AFTER_UNKNOWN_OUTCOME[before:]


def test_unknown_write_withholds_slot_and_source(backend, monkeypatch):
    raw = backend._core.raw_device()
    monkeypatch.setattr(raw, "is_poisoned", lambda: False, raising=False)
    real_wait = raw.wait_iouring

    def unknown_wait(batch):
        results, errors = real_wait(batch)
        monkeypatch.setattr(raw, "is_poisoned", lambda: True)
        return [False] * len(results), errors

    monkeypatch.setattr(raw, "wait_iouring", unknown_wait)
    objects = [_make_byte_obj(32), _make_byte_obj(64)]
    keys = [CacheEngineKey("test_model", 1, 0, i, torch.bfloat16) for i in (1, 2)]
    futures = backend.batched_submit_put_task(keys, objects)
    assert futures is not None
    with pytest.raises(RuntimeError, match="persist"):
        futures[0].result(timeout=5)

    assert all(obj.get_ref_count() == 2 for obj in objects)
    assert all(
        any(held is obj for held in backend._quarantined_objs) for obj in objects
    )
    assert len(backend._core._quarantined_slots) == 2
    assert not backend._core._free_slots
    with pytest.raises(RuntimeError, match="outcome"):
        backend._core.put_many([encode_legacy_key(keys[0])], objects[:1])
    outcome = backend._core.close()
    assert outcome.quiescence is NativeQuiescence.RETAINED
    assert backend._core.close() is outcome
    backend.close()
    assert backend._core._raw is raw
    assert backend.local_cpu_backend._backing_resources_retained


@pytest.mark.parametrize("failed_read", ["bitmap", "exception"])
def test_unknown_read_retains_destination_and_source_extent(
    backend, monkeypatch, failed_read
):
    key = CacheEngineKey("test_model", 1, 0, 3, torch.bfloat16)
    obj = _make_byte_obj(32)
    spec = encode_legacy_key(key)
    assert backend._core.put_many([spec], [obj]).results == [True]
    target = _make_byte_obj(32)
    monkeypatch.setattr(backend.local_cpu_backend, "allocate", lambda *args: target)

    def unknown_read(*args):
        backend._core._poisoned = True
        if failed_read == "exception":
            raise OSError("completion lost")
        return [False]

    monkeypatch.setattr(backend._core, "load_many_into", unknown_read)
    if failed_read == "exception":
        with pytest.raises(OSError, match="completion lost"):
            backend.get_blocking(key)
    else:
        assert backend.get_blocking(key) is None
    assert target.get_ref_count() == 1
    assert backend._quarantined_objs == [target]
    assert spec.encoded in backend._pinned_keys
    assert backend._core._lock_refcnt[spec.encoded] == 1


@pytest.mark.parametrize("many", [False, True])
def test_cancelled_put_retains_owners_until_thread_finishes(
    backend, loop_in_thread, monkeypatch, many
):
    entered = threading.Event()
    finish = threading.Event()
    real_put = backend._core.put_many

    def blocked_put(*args):
        entered.set()
        assert finish.wait(5)
        return real_put(*args)

    monkeypatch.setattr(backend._core, "put_many", blocked_put)
    count = 2 if many else 1
    keys = [
        CacheEngineKey("test_model", 1, 0, i + 4, torch.bfloat16) for i in range(count)
    ]
    objects = [_make_byte_obj(32) for _ in keys]
    futures = backend.batched_submit_put_task(keys, objects)
    assert futures is not None and entered.wait(5)
    assert futures[0].cancel()
    try:
        deadline = time.monotonic() + 5
        while not backend._pending_put_owners and time.monotonic() < deadline:
            time.sleep(0.01)
        assert backend._pending_put_owners
        assert all(obj.get_ref_count() == 2 for obj in objects)
    finally:
        finish.set()
    deadline = time.monotonic() + 5
    while (
        any(obj.get_ref_count() != 1 for obj in objects) and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert all(obj.get_ref_count() == 1 for obj in objects)
    assert backend._pending_put_owners == []


@pytest.mark.parametrize("close_failure", ["exception", "busy", "active-caller"])
def test_unproven_close_preserves_entire_backing_graph(
    backend, monkeypatch, close_failure
):
    raw = backend._core.raw_device()
    if close_failure == "exception":
        monkeypatch.setattr(raw, "close", Mock(side_effect=OSError("still draining")))
    elif close_failure == "busy":
        monkeypatch.setattr(raw, "is_idle", lambda: False, raising=False)
    else:
        backend._core._inflight_io_count = 1
    core_ref = weakref.ref(backend._core)
    cpu_ref = weakref.ref(backend.local_cpu_backend)
    backend.close()
    assert backend._core._raw is raw
    assert backend.local_cpu_backend._backing_resources_retained
    assert plugin._RETAINED_AFTER_UNKNOWN_OUTCOME
    backend._core = None
    backend.local_cpu_backend = None
    gc.collect()
    assert core_ref() is not None and cpu_ref() is not None


def test_proven_close_cannot_reopen_device(backend):
    backend.close()
    assert not getattr(backend.local_cpu_backend, "_backing_resources_retained", False)
    assert backend._core._raw is None
    with pytest.raises(RuntimeError, match="closed"):
        backend._core.raw_device()
    assert backend.batched_get_blocking([]) == []


def test_cpu_close_honors_storage_retention():
    cpu = object.__new__(LocalCPUBackend)
    cpu.memory_allocator = Mock()
    cpu.batched_msg_sender = None
    cpu.clear = Mock()
    cpu.retain_backing_resources()
    cpu.close()
    cpu.memory_allocator.close.assert_not_called()
    cpu.clear.assert_not_called()


def test_manager_closes_consumers_before_cpu_arena():
    order = []
    cpu = object.__new__(LocalCPUBackend)
    cpu.close = lambda: order.append("allocator")
    manager = object.__new__(StorageManager)
    manager.storage_backends = {
        "LocalCPUBackend": cpu,
        "RawBlock": SimpleNamespace(close=lambda: order.append("consumer")),
    }
    manager.loop = asyncio.new_event_loop()
    manager.thread = SimpleNamespace(is_alive=lambda: False)
    manager.internal_copy_stream = None
    manager._copy_owners_retained = False
    manager.close()
    assert order == ["consumer", "allocator"]
