# SPDX-License-Identifier: Apache-2.0
"""What a raw-block backend is allowed to release when it shuts down.

Closing this backend hands three things back: the arena behind the GPU
staging allocator (and an ``os.close`` on every dma-buf exported from it),
the device slots the buffers were written to, and any read lease a peer
decoder is holding. All three are authorized by the same fact -- that the
device is finished with those buffers -- and none of them is authorized by
close having been *asked for*.

These tests drive the real ``close()`` and assert on what the process can
still reach afterwards: whether the allocator was closed, whether a lease
was unlocked, and whether the objects the device may still be writing into
are alive. A counter saying "1 quarantined" is not that; the object being
alive is.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Optional
import asyncio
import gc
import os
import tempfile
import threading
import time
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.plugins import rust_raw_block_backend as plugin
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)
from lmcache.v1.storage_backend.raw_block import (
    NativeQuiescence,
    RawBlockCloseOutcome,
    encode_legacy_key,
)

_METADATA = LMCacheMetadata(
    model_name="test_model",
    world_size=1,
    local_world_size=1,
    worker_id=0,
    local_worker_id=0,
    kv_dtype=torch.bfloat16,
    kv_shape=(4, 2, 256, 8, 128),
)


class _CountingAllocator:
    """Stands in for the GPU staging allocator.

    Only ``close`` matters here: it is the call that releases the arena and
    the exported descriptors, so the number of times it ran is the whole
    question.
    """

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _RecordingTracker:
    """Stands in for the P/D obligation tracker."""

    def __init__(self) -> None:
        self.release_leases: Optional[bool] = None
        self.close_calls = 0

    def close(self, release_leases: bool = True) -> None:
        self.close_calls += 1
        self.release_leases = release_leases


class _BackingOwner:
    """A stand-in for something the device may still be writing into.

    Its identity is the assertion: if this object is collected, whatever the
    device was told to write into it has been handed back to the allocator.
    """


def _config(dev_path: str) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=False,
        max_local_cpu_size=0,
        lmcache_instance_id="test_raw_block_close_ownership",
    )
    config.storage_plugins = []
    config.extra_config = {
        "rust_raw_block.device_path": dev_path,
        "rust_raw_block.block_align": 4096,
        "rust_raw_block.header_bytes": 4096,
        "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
        "rust_raw_block.meta_enable_periodic": False,
        "rust_raw_block.gpu_buffer_bytes": 1 << 20,
    }
    return config


@pytest.fixture
def loop_in_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="test-loop", daemon=True)
    t.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        loop.close()


@pytest.fixture
def backend(loop_in_thread, monkeypatch):
    """A backend with a fake staging allocator and a real core."""
    monkeypatch.setattr(
        RustRawBlockBackend,
        "_build_gpu_allocator",
        lambda self, size_bytes, device: _CountingAllocator(),
    )
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)
        instance = RustRawBlockBackend(
            config=_config(dev_path),
            metadata=_METADATA,
            local_cpu_backend=None,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        yield instance
        if not instance._closed_once:
            instance.close()


@pytest.fixture(autouse=True)
def _keep_the_retention_list_to_this_test():
    """The retention list outlives a process on purpose, not a test."""
    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    yield
    del plugin._RETAINED_AFTER_UNKNOWN_OUTCOME[before:]


def _retained_now(before: int) -> list[Any]:
    return list(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME[before:])


def test_a_close_that_failed_releases_nothing_it_was_holding(backend):
    """A native close that raises has not established quiescence.

    The device may still be draining, so neither the arena nor a peer's read
    lease is this backend's to hand back -- and dropping the backend must not
    become a second, quieter authorization.

    Nothing is quarantined and no batch is outstanding, so the failed close
    is the only thing that can refuse: a backend that reads the raise as
    health reaches the release with every other input saying it may.
    """
    allocator = backend._gpu_allocator
    tracker = _RecordingTracker()
    backend._pd_tracker = tracker
    assert backend._quarantined_objs == []
    assert backend._pending_put_owners == []

    def _fails_while_draining():
        raise OSError("unregistering buffers failed")

    backend._core.close = _fails_while_draining  # type: ignore[method-assign]

    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    backend.close()

    assert allocator.close_calls == 0
    assert tracker.close_calls == 1
    assert tracker.release_leases is False
    assert len(_retained_now(before)) == 1

    # Dropping the backend object is the second way the same release can
    # happen: a finalizer, or the allocator simply becoming unreachable.
    ref = weakref.ref(allocator)
    backend._gpu_allocator = None
    del allocator
    gc.collect()
    assert ref() is not None
    assert ref().close_calls == 0


def test_a_close_that_could_not_prove_quiescence_releases_nothing(backend):
    """A core that returns RETAINED has answered, and the answer is no.

    This is the shape of an unpolled submission: the core completed its own
    close without raising, and still cannot say what the device is doing, so
    it reports that rather than a proof.
    """
    allocator = backend._gpu_allocator
    tracker = _RecordingTracker()
    backend._pd_tracker = tracker
    assert backend._quarantined_objs == []
    assert backend._pending_put_owners == []

    backend._core.close = lambda: RawBlockCloseOutcome(  # type: ignore[method-assign]
        quiescence=NativeQuiescence.RETAINED,
        poisoned=True,
        final_checkpoint_written=False,
        reason="a batch nobody polled is still outstanding",
    )

    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    backend.close()

    assert allocator.close_calls == 0
    assert tracker.release_leases is False
    assert len(_retained_now(before)) == 1

    ref = weakref.ref(allocator)
    backend._gpu_allocator = None
    del allocator
    gc.collect()
    assert ref() is not None
    assert ref().close_calls == 0


def test_a_batch_nobody_polled_keeps_its_owners_past_the_backend(backend):
    """The submission whose fate was never collected still owns its buffers.

    Nothing polled the ring, so no completion said the write landed. The
    objects the device was handed are therefore still the device's, and they
    have to outlive both the close and the backend -- under the allocation
    pressure that would otherwise reuse them.
    """
    allocator = backend._gpu_allocator
    owner = _BackingOwner()
    backend._pending_put_owners.append([owner])
    alive = weakref.ref(owner)
    del owner

    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    backend.close()

    assert allocator.close_calls == 0
    assert len(_retained_now(before)) == 1

    # Every route back to the object through the backend is cut, so what
    # keeps it alive is the retention itself.
    backend._pending_put_owners = []
    backend._gpu_allocator = None
    del allocator
    gc.collect()
    ballast = [bytearray(1 << 20) for _ in range(16)]
    gc.collect()
    assert alive() is not None
    assert alive() is _retained_now(before)[0][3][0][0]
    del ballast


def test_a_proven_close_does_release_what_it_holds(backend):
    """The refusals above have to be a decision, not a stuck fail-closed.

    A healthy close proves the device is finished, so the arena goes back --
    while a lease still does not, because local quiescence says nothing
    about a reader elsewhere.
    """
    allocator = backend._gpu_allocator
    tracker = _RecordingTracker()
    backend._pd_tracker = tracker

    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    backend.close()

    assert allocator.close_calls == 1
    assert tracker.release_leases is True
    assert _retained_now(before) == []


def test_closing_twice_does_not_release_twice(backend):
    """A second close must not reach the device accessor.

    That accessor builds a device when the core has none, so asking a
    closed core about its device's health opens a new writable handle on
    the same path -- and then releases the allocator a second time. Two
    constructions and two allocator closes, for a call that should do
    nothing.
    """
    allocator = backend._gpu_allocator
    backend.close()
    assert allocator.close_calls == 1

    backend.close()
    assert allocator.close_calls == 1
    assert backend._core._raw is None


def test_a_cancelled_put_keeps_its_buffer_until_the_thread_lets_go(
    backend, loop_in_thread
):
    """Cancelling the await does not stop the thread doing the write.

    ``asyncio.to_thread`` hands the work to an executor; cancelling the task
    that waits on it cancels the waiting. The thread still holds the buffer
    and is still handing it to the device, so releasing it here returns the
    allocator a slice that is being written into. The identity of the object
    is the assertion -- this exact buffer, still held, then released once the
    thread says it is done.
    """
    allocator = AdHocMemoryAllocator(device="cpu")
    memory_obj = allocator.allocate(
        [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
    )
    assert memory_obj is not None
    assert memory_obj.get_ref_count() == 1
    key = CacheEngineKey("test_model", 1, 0, 4004, torch.bfloat16)
    spec = encode_legacy_key(key)

    in_the_thread = threading.Event()
    may_finish = threading.Event()
    real_put_many = backend._core.put_many

    def _blocks_inside_the_thread(specs, objs):
        in_the_thread.set()
        assert may_finish.wait(10)
        return real_put_many(specs, objs)

    backend._core.put_many = _blocks_inside_the_thread  # type: ignore[method-assign]

    spawned: dict[str, Any] = {}

    def _spawn() -> None:
        spawned["task"] = asyncio.ensure_future(
            backend._submit_put_one(key, spec, memory_obj, None)
        )

    loop_in_thread.call_soon_threadsafe(_spawn)
    assert in_the_thread.wait(10), "the write never reached its thread"
    while "task" not in spawned:  # pragma: no cover - set by the callback above
        pass
    task = spawned["task"]
    loop_in_thread.call_soon_threadsafe(task.cancel)

    deadline = time.monotonic() + 10.0
    while not task.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert task.done()

    # The await is over and the thread is not. This exact object is held.
    with backend._put_lock:
        held = [owners for owners in backend._pending_put_owners]
    assert len(held) == 1
    assert held[0][0] is memory_obj
    assert memory_obj.get_ref_count() == 1

    may_finish.set()
    deadline = time.monotonic() + 10.0
    while memory_obj.get_ref_count() == 1 and time.monotonic() < deadline:
        time.sleep(0.01)

    # Once the thread has ended, the same object is handed back -- the
    # retention is a wait, not a leak.
    assert memory_obj.get_ref_count() == 0
    with backend._put_lock:
        assert backend._pending_put_owners == []


def test_a_control_handler_that_did_not_stop_blocks_every_release(backend):
    """Releasing a hold reaches the core, so the handler stops before it.

    A timed join that returns is not proof the handler stopped touching the
    core: the thread may have been descheduled inside it. The server says
    which happened, and an unconfirmed stop means nothing after it can be
    released on evidence -- the core is being destroyed under something
    that may still be using it.
    """
    allocator = backend._gpu_allocator
    tracker = _RecordingTracker()
    backend._pd_tracker = tracker

    class _NeverConfirms:
        def __init__(self) -> None:
            self.close_calls = 0
            self.endpoint = "127.0.0.1:1"

        def close(self, timeout_s: float = 5.0) -> bool:
            self.close_calls += 1
            return False

    server = _NeverConfirms()
    backend._ack_receiver = server

    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    backend.close()

    assert server.close_calls == 1
    assert allocator.close_calls == 0
    assert tracker.release_leases is False
    assert len(_retained_now(before)) == 1


def test_a_control_handler_that_stopped_does_not_block_release(backend):
    """The refusal above has to be the server's answer, not a stuck default."""
    allocator = backend._gpu_allocator
    tracker = _RecordingTracker()
    backend._pd_tracker = tracker

    class _Confirms:
        def __init__(self) -> None:
            self.close_calls = 0
            self.endpoint = "127.0.0.1:1"

        def close(self, timeout_s: float = 5.0) -> bool:
            self.close_calls += 1
            return True

    server = _Confirms()
    backend._ack_receiver = server

    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    backend.close()

    assert server.close_calls == 1
    assert allocator.close_calls == 1
    assert tracker.release_leases is True
    assert _retained_now(before) == []
