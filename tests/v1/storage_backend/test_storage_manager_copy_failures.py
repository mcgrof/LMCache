# SPDX-License-Identifier: Apache-2.0
"""Real staging-copy failures must preserve owners until copies have stopped."""

# Standard
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
import asyncio
import gc
import threading
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager


class CopyStream:
    """Complete deferred copies only when their owners are still allocated."""

    def __init__(self, sync_fails: bool = False) -> None:
        self.pending: list[tuple[CopyObject, CopyObject]] = []
        self.sync_fails = sync_fails
        self.drains = 0

    def synchronize(self) -> None:
        self.drains += 1
        if self.sync_fails:
            raise RuntimeError("copy synchronization failed")
        for destination, source in self.pending:
            assert destination.references > 0
            assert source.references > 0
            destination.value = source.value
        self.pending.clear()


class CopyObject:
    """Small owner whose tensor queues a transfer through the real helper."""

    def __init__(
        self,
        stream: CopyStream,
        value: int,
        *,
        enqueue_fails: bool = False,
        device: str = "cpu",
    ) -> None:
        self.references = 1
        self.value = value
        self.meta = SimpleNamespace(fmt=MemoryFormat.KV_T2D)
        self.tensor = SimpleNamespace(
            device=SimpleNamespace(type=device),
            owner=self,
            copy_=self.enqueue,
        )
        self.stream = stream
        self.enqueue_fails = enqueue_fails

    def enqueue(self, source: SimpleNamespace, non_blocking: bool) -> None:
        self.stream.pending.append((self, source.owner))
        if self.enqueue_fails:
            raise RuntimeError("second copy failed")

    def get_shape(self) -> torch.Size:
        return torch.Size([2])

    def get_dtype(self) -> torch.dtype:
        return torch.float32

    def ref_count_up(self) -> None:
        self.references += 1

    def ref_count_down(self) -> None:
        self.references -= 1
        assert self.references >= 0


def manager_inputs() -> tuple[LMCacheEngineConfig, LMCacheMetadata]:
    """Return a scheduler configuration that needs no native GPU runtime."""
    return (
        LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            local_cpu=True,
            lmcache_instance_id="copy-failure-test",
        ),
        LMCacheMetadata(
            model_name="copy-failure-test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float32,
            kv_shape=(2, 2, 16, 1, 2),
            role="scheduler",
        ),
    )


def make_manager(
    monkeypatch: pytest.MonkeyPatch,
    host: Mock,
    stream: CopyStream,
    source: Mock | None = None,
) -> StorageManager:
    """Construct the manager normally with isolated backend fault seams."""
    backends = OrderedDict([("LocalCPUBackend", host)])
    if source is not None:
        backends["Storage"] = source
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.CreateStorageBackends",
        lambda *args, **kwargs: backends,
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.torch_dev.stream",
        lambda _: nullcontext(),
    )
    config, metadata = manager_inputs()
    manager = StorageManager(config, metadata, EventManager())
    manager.allocator_backend = Mock(spec=AllocatorBackendInterface)
    manager.internal_copy_stream = cast(Any, stream)
    return manager


def cpu_backend(destinations: list[CopyObject | OSError]) -> Mock:
    """Return an allocator seam whose public methods record ownership use."""
    backend = Mock(spec=LocalCPUBackend)
    backend.use_hot = True
    backend.contains.return_value = False
    backend.allocate.side_effect = destinations
    backend.get_allocator_backend.return_value = backend
    backend.batched_submit_put_task.return_value = None
    return backend


def cache_keys(count: int) -> list[CacheEngineKey]:
    """Return one distinct cache key per source object."""
    return [CacheEngineKey("copy", 1, 0, i, torch.float32) for i in range(count)]


@pytest.mark.parametrize("failure", ["allocation", "copy", "dispatch", "none"])
def test_real_copy_helper_releases_partial_groups(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    stream = CopyStream()
    sources = [CopyObject(stream, i + 1) for i in range(2)]
    destinations = [
        CopyObject(stream, 0),
        CopyObject(stream, 0, enqueue_fails=failure == "copy"),
    ]
    allocation: list[CopyObject | OSError] = list(destinations)
    if failure == "allocation":
        allocation[1] = OSError("second allocation failed")
    backend = cpu_backend(allocation)
    if failure == "dispatch":
        backend.batched_submit_put_task.side_effect = OSError("dispatch refused")
    manager = make_manager(monkeypatch, backend, stream)
    try:
        if failure == "none":
            manager.batched_put(cache_keys(2), cast(Any, sources))
        else:
            with pytest.raises((RuntimeError, OSError), match="failed|refused"):
                manager.batched_put(cache_keys(2), cast(Any, sources))
        assert stream.drains == 1
        assert not stream.pending
        assert [obj.references for obj in sources] == [0, 0]
        assert destinations[0].references == 0
        assert destinations[1].references == (1 if failure == "allocation" else 0)
        if failure in ("allocation", "copy"):
            backend.batched_submit_put_task.assert_not_called()
        else:
            backend.batched_submit_put_task.assert_called_once()
        assert destinations[0].value == sources[0].value
    finally:
        manager.close()


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("failure", ["copy", "dispatch", "none"])
def test_writeback_failure_releases_loaded_and_host_owners(
    monkeypatch: pytest.MonkeyPatch, batched: bool, failure: str
) -> None:
    stream = CopyStream()
    count = 2 if batched else 1
    sources = [CopyObject(stream, i + 1, device="cuda") for i in range(count)]
    destinations = [
        CopyObject(stream, 0, enqueue_fails=failure == "copy" and i == count - 1)
        for i in range(count)
    ]
    host = cpu_backend(list(destinations))
    if failure == "dispatch":
        host.batched_submit_put_task.side_effect = OSError("dispatch refused")
    storage = Mock()
    storage.get_blocking.return_value = sources[0]
    storage.batched_get_blocking.return_value = sources
    manager = make_manager(monkeypatch, host, stream, storage)
    keys = cache_keys(count)
    try:

        def load() -> Any:
            if batched:
                return manager.batched_get(keys, location="Storage")
            return manager.get(keys[0], location="Storage")

        if failure == "none":
            loaded = load()
            assert loaded == (sources if batched else sources[0])
        else:
            with pytest.raises((RuntimeError, OSError), match="failed|refused"):
                load()
        assert not stream.pending
        assert all(obj.references == 0 for obj in destinations)
        assert all(obj.references == (1 if failure == "none" else 0) for obj in sources)
    finally:
        manager.close()


def test_failed_sync_retains_owners_rejects_reuse_and_preserves_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = CopyStream(sync_fails=True)
    source = CopyObject(stream, 1)
    destination = CopyObject(stream, 0)
    backend = cpu_backend([destination])
    manager = make_manager(monkeypatch, backend, stream)
    source_ref = weakref.ref(source)
    destination_ref = weakref.ref(destination)
    manager_ref = weakref.ref(manager)
    with pytest.raises(RuntimeError, match="synchronization failed"):
        manager.batched_put(cache_keys(1), cast(Any, [source]))
    assert source.references == 1
    assert destination.references == 1
    untouched = CopyObject(stream, 2)
    with pytest.raises(RuntimeError, match="unusable"):
        manager.batched_put(cache_keys(1), cast(Any, [untouched]))
    assert untouched.references == 0
    assert backend.allocate.call_count == 1
    assert manager.close_backend("LocalCPUBackend") is False
    with pytest.raises(RuntimeError, match="Cannot recreate"):
        manager.recreate_backend("LocalCPUBackend")
    backend.retain_backing_resources.assert_called_once()
    manager.close()
    manager.close()
    backend.close.assert_not_called()
    backend.retain_backing_resources.assert_called_once()
    assert not manager.thread.is_alive()
    assert manager.loop.is_closed()
    # Remove the fault seam's incidental references; quarantine must be the
    # owner even after callers discard the failed batch and its stream.
    stream.pending.clear()
    backend.reset_mock(return_value=True, side_effect=True)
    del source, destination, manager, backend, stream
    gc.collect()
    assert source_ref() is not None
    assert destination_ref() is not None
    assert manager_ref() is not None


@pytest.mark.parametrize("loop_started", [False, True])
def test_constructor_failure_stops_and_closes_loop(
    monkeypatch: pytest.MonkeyPatch, loop_started: bool
) -> None:
    loops: list[asyncio.AbstractEventLoop] = []
    threads: list[threading.Thread] = []
    release_start = threading.Event()
    actual_loop_factory = asyncio.new_event_loop
    actual_thread_class = threading.Thread

    def make_loop() -> asyncio.AbstractEventLoop:
        loop = actual_loop_factory()
        loops.append(loop)
        if not loop_started:
            queue_callback = loop.call_soon_threadsafe

            def queue_stop(callback: Any, *args: Any, **kwargs: Any) -> Any:
                handle = queue_callback(callback, *args, **kwargs)
                if callback == loop.stop:
                    release_start.set()
                return handle

            monkeypatch.setattr(loop, "call_soon_threadsafe", queue_stop)
        return loop

    def make_thread(*args: Any, **kwargs: Any) -> threading.Thread:
        target = kwargs["target"]
        target_args = kwargs["args"]

        def run() -> None:
            if not loop_started:
                assert release_start.wait(timeout=15)
            target(*target_args)

        thread = actual_thread_class(target=run, name=kwargs["name"])
        threads.append(thread)
        return thread

    def fail_factory(*args: Any, **kwargs: Any) -> None:
        if loop_started:
            ready = threading.Event()
            loops[0].call_soon_threadsafe(ready.set)
            assert ready.wait(timeout=5)
        raise OSError("backend construction failed")

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.asyncio.new_event_loop", make_loop
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.threading.Thread", make_thread
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.CreateStorageBackends", fail_factory
    )
    config, metadata = manager_inputs()
    try:
        with pytest.raises(OSError, match="backend construction failed"):
            StorageManager(config, metadata, EventManager())
        assert len(threads) == 1
        assert not threads[0].is_alive()
        assert loops[0].is_closed()
    finally:
        release_start.set()
        for loop, thread in zip(loops, threads, strict=True):
            if not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            if not loop.is_closed() and not thread.is_alive():
                loop.close()


def test_later_constructor_failure_closes_created_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Mock(spec=LocalCPUBackend)
    loops: list[asyncio.AbstractEventLoop] = []

    def create_backends(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        **kwargs: Any,
    ) -> OrderedDict[str, Any]:
        loops.append(loop)
        return OrderedDict([("LocalCPUBackend", backend)])

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.CreateStorageBackends",
        create_backends,
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.PrometheusLogger.GetOrCreate",
        Mock(side_effect=OSError("metrics setup failed")),
    )
    config, metadata = manager_inputs()
    with pytest.raises(OSError, match="metrics setup failed"):
        StorageManager(config, metadata, EventManager())
    backend.close.assert_called_once()
    assert loops[0].is_closed()
