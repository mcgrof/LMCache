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
from typing import Any
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
    RawBlockCore,
    RawBlockPDRequestTracker,
    RawBlockPublicationReceipt,
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
        self.close_calls = 0
        self.quiesced_release_calls = 0
        self.publication_quiesced = True

    def close(self, timeout_s: float = 5.0) -> bool:
        self.close_calls += 1
        return self.publication_quiesced

    def release_quiesced_leases(self) -> int:
        self.quiesced_release_calls += 1
        return 0


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
        "rust_raw_block.io_engine": "io_uring",
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
def _stub_staging_registration(monkeypatch):
    """The counting arena tests teardown, not GPU registration or transfers."""
    monkeypatch.setattr(
        RawBlockCore, "register_fixed_buffers_from_allocator", lambda *args: None
    )


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
    assert tracker.close_calls == 1
    assert tracker.quiesced_release_calls == 0
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
    assert tracker.close_calls == 1
    assert tracker.quiesced_release_calls == 0
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
    assert alive() is _retained_now(before)[0][4][0][0]
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
    # A healthy local close is still not release authority over a reader's
    # hold, and no operator declared the group stopped.
    assert tracker.close_calls == 1
    assert tracker.quiesced_release_calls == 0
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

    def _blocks_inside_the_thread(specs, objs, **kwargs):
        in_the_thread.set()
        assert may_finish.wait(10)
        return real_put_many(specs, objs, **kwargs)

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
    assert tracker.close_calls == 1
    assert tracker.quiesced_release_calls == 0
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
    # A healthy local close is still not release authority over a reader's
    # hold, and no operator declared the group stopped.
    assert tracker.close_calls == 1
    assert tracker.quiesced_release_calls == 0
    assert _retained_now(before) == []


def test_an_operator_declared_quiesced_teardown_releases_the_holds(backend):
    """The one close that may release a reader's hold, and only on a say-so.

    Nothing in this process can see that every engine able to read this
    namespace has stopped. The operator asserts it in configuration, and
    only then does teardown release holds that a consumer would otherwise
    be the only one able to free.
    """
    allocator = backend._gpu_allocator
    tracker = _RecordingTracker()
    backend._pd_tracker = tracker
    backend._pd_group_quiesced_teardown = True

    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    backend.close()

    assert tracker.close_calls == 1
    assert tracker.quiesced_release_calls == 1
    assert allocator.close_calls == 1
    assert _retained_now(before) == []


def test_a_quiesced_declaration_does_not_survive_an_unknown_outcome(backend):
    """A group that stopped is not a device whose state is known.

    The holds a reader would free are one question; what the device is
    still doing with these extents is another, and the declaration answers
    only the first.
    """
    tracker = _RecordingTracker()
    backend._pd_tracker = tracker
    backend._pd_group_quiesced_teardown = True
    backend._pending_put_owners.append([_BackingOwner()])

    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    backend.close()

    assert tracker.close_calls == 1
    assert tracker.quiesced_release_calls == 0
    assert backend._gpu_allocator.close_calls == 0
    assert len(_retained_now(before)) == 1


def _publication_tracker(backend) -> RawBlockPDRequestTracker:
    """Give this backend a real publication tracker over its real core."""
    tracker = RawBlockPDRequestTracker(backend._core)
    tracker.ack_endpoint = "127.0.0.1:0"
    backend._pd_tracker = tracker
    return tracker


def test_a_queued_publication_never_reaches_the_closed_core(backend):
    """Publication is not a put, and an empty put set does not cover it.

    A publication accepted before shutdown and started after it would open a
    fresh device -- one that knows nothing about what the old one was doing
    -- pin keys in it and write an index nobody is watching. So shutdown
    cancels what has not started, and the core refuses a publication once it
    is shutting down.
    """
    tracker = _publication_tracker(backend)
    occupied = threading.Event()
    gate = threading.Event()

    def occupy() -> None:
        occupied.set()
        assert gate.wait(10), "the test gate was never released"

    tracker._publisher.submit(occupy)
    assert occupied.wait(5)

    key = CacheEngineKey("test_model", 1, 0, 9001, torch.bfloat16)
    encoded = encode_legacy_key(key).encoded
    terminal = tracker.register_batch(
        "request-1",
        [encoded],
        expected_chunks=1,
        is_last_batch=True,
        completed_keys=[encoded],
    )

    closed = threading.Thread(target=backend.close)
    closed.start()
    deadline = time.monotonic() + 10.0
    while not tracker._closed and time.monotonic() < deadline:
        time.sleep(0.005)
    assert tracker._closed, "close never reached publication shutdown"

    gate.set()
    closed.join(timeout=20.0)
    assert not closed.is_alive()

    # The publication did not happen, and nothing opened a device for it.
    assert terminal.done() and terminal.exception() is not None
    assert backend._core._raw is None
    assert backend._core._terminal
    with pytest.raises(RuntimeError, match="will not be reopened"):
        backend._core._rawdev()


def test_a_publication_inside_the_core_blocks_every_release(backend):
    """Work that could not be confirmed stopped is work the core still has.

    Closing the core here destroys it underneath a publication that is
    inside it, and retaining the memory afterwards does not undo a job that
    already touched a device freed beneath it. So the teardown does not
    complete: the whole graph is kept, the core included.
    """
    tracker = _publication_tracker(backend)
    allocator = backend._gpu_allocator
    inside = threading.Event()
    gate = threading.Event()
    # Gated at the first thing publication asks of the core, so the job is
    # provably inside it when close runs. What the publication goes on to
    # do with the answer is not the question here.
    real_prefix = backend._core.get_metadata_prefix

    def gated_prefix(encoded_keys, *, lock=False):
        inside.set()
        assert gate.wait(20), "the test gate was never released"
        return real_prefix(encoded_keys, lock=lock)

    backend._core.get_metadata_prefix = gated_prefix  # type: ignore[method-assign]

    key = CacheEngineKey("test_model", 1, 0, 9002, torch.bfloat16)
    encoded = encode_legacy_key(key).encoded
    tracker.register_batch(
        "request-1",
        [encoded],
        expected_chunks=1,
        is_last_batch=True,
        completed_keys=[encoded],
    )
    assert inside.wait(10), "publication never reached the core"

    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    # A short budget, because the point is what happens when the wait gives
    # up rather than how long it waits.
    tracker_close = tracker.close
    backend._pd_tracker.close = lambda timeout_s=0.2: tracker_close(timeout_s=0.2)  # type: ignore[method-assign]
    backend.close()

    assert allocator.close_calls == 0
    assert backend._core._raw is not None
    assert not backend._core._closed
    assert len(_retained_now(before)) == 1

    gate.set()
    # The publication finishes against the core that was deliberately kept.
    deadline = time.monotonic() + 10.0
    while tracker._publishing and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not tracker._publishing


def _staging_object():
    """One staging buffer, the way the engine hands them to the device."""
    allocator = AdHocMemoryAllocator(device="cpu")
    memory_obj = allocator.allocate(
        [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
    )
    assert memory_obj is not None
    return memory_obj


@pytest.fixture
def iouring_backend(loop_in_thread, monkeypatch):
    """A backend whose core really goes through the io_uring path.

    Registration is stubbed for the counting arena; actual payloads in this
    fixture are CPU buffers handled by the native io_uring worker.
    """
    monkeypatch.setattr(
        RustRawBlockBackend,
        "_build_gpu_allocator",
        lambda self, size_bytes, device: _CountingAllocator(),
    )
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)
        config = _config(dev_path)
        config.extra_config["rust_raw_block.io_engine"] = "io_uring"
        instance = RustRawBlockBackend(
            config=config,
            metadata=_METADATA,
            local_cpu_backend=None,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        yield instance
        if not instance._closed_once:
            instance.close()


def test_an_unknown_outcome_withholds_the_extent_and_the_buffer(iouring_backend):
    """One unknown outcome, two things that must not be handed back.

    A weakref proving a Python object is still alive says a staging buffer
    was not released. It says nothing about the device extent that buffer
    was being written into: an extent returned to the free list goes to the
    next request, which then gets storage an earlier write may still be
    landing in -- and the slot header would name one key over another key's
    bytes, validating clean because the header belongs to the later,
    successful writer.

    So the extent and the memory behind the device are asserted together,
    from the same failure: the allocator cannot reach the extent by any
    route, and the arena the staging buffers were exported from is not
    closed. Owner-level retention inside the native engine is a separate
    statement, made against a ring that can be asked what it holds.
    """
    core = iouring_backend._core
    raw = core.raw_device()
    settled = CacheEngineKey("test_model", 1, 0, 7001, torch.bfloat16)
    assert core.put_many(
        [encode_legacy_key(settled)],
        [_staging_object()],
    ).results == [True]
    free_before = set(core._free_slots)
    next_slot_before = core._next_slot

    class _Unknown:
        """Admit the write, then lose its outcome while waiting for it."""

        unknown = False

        def __getattr__(self, item):
            return getattr(raw, item)

        def is_poisoned(self):
            return self.unknown

        def quarantined_batch_count(self):
            return 1

        def wait_iouring(self, batch_id):
            results, errors = raw.wait_iouring(batch_id)
            self.unknown = True
            return [False] * len(results), errors

    core.set_raw_device_for_testing(_Unknown())
    try:
        unprovable = CacheEngineKey("test_model", 1, 0, 7002, torch.bfloat16)
        assert core.put_many(
            [encode_legacy_key(unprovable)], [_staging_object()]
        ).results == [False]

        # The extent did not go back where it could be handed out again,
        # and the allocator cannot reach it by any route.
        assert set(core._free_slots) == free_before
        assert set(core._quarantined_slots or {}) == {next_slot_before}
        with core._lock, pytest.raises(RuntimeError, match="unknown I/O"):
            core._allocate_slot_locked(None)
    finally:
        core.set_raw_device_for_testing(raw)

    # And the teardown that follows releases nothing behind the device.
    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    allocator = iouring_backend._gpu_allocator
    iouring_backend.close()

    assert allocator.close_calls == 0
    assert len(_retained_now(before)) == 1


@pytest.mark.parametrize("operation", ["read", "store", "adopt"])
def test_close_keeps_a_caller_preparing_work_alive(
    backend: RustRawBlockBackend, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Count preparation before it reaches either a put task or native I/O."""
    # Standard
    from concurrent.futures import Future, ThreadPoolExecutor

    entered = threading.Event()
    resume = threading.Event()
    result: list[Any] | bool | None = (
        [] if operation == "read" else False if operation == "adopt" else None
    )
    allocator = backend.get_memory_allocator()
    assert isinstance(allocator, _CountingAllocator)

    def delayed(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert resume.wait(5), "the admitted caller did not resume"
        return result

    if operation == "adopt":
        backend._role = "reader"
        monkeypatch.setattr(backend._core, "refresh_until_publication", delayed)
    elif operation == "read":
        monkeypatch.setattr(backend._core, "get_metadata_prefix", delayed)
    else:
        backend._storage_pd_mode = True
        monkeypatch.setattr(backend, "_batched_submit_pd_request", delayed)
    clock = time.monotonic()
    ticks = iter([clock, clock + 11])
    # Expire only this backend's drain budget; its core and the executor
    # continue to use the real clock.
    monkeypatch.setattr(
        plugin,
        "time",
        type("Clock", (), {"monotonic": lambda: next(ticks), "sleep": time.sleep}),
    )
    before = len(plugin._RETAINED_AFTER_UNKNOWN_OUTCOME)
    with ThreadPoolExecutor(max_workers=1) as caller:
        pending: Future[Any]
        if operation == "read":
            key = CacheEngineKey("test_model", 1, 0, 9011, torch.bfloat16)
            pending = caller.submit(backend.get_blocking, key)
        elif operation == "adopt":
            receipt = RawBlockPublicationReceipt("writer", 1, 0, "manifest")
            pending = caller.submit(
                backend.adopt_publication, receipt, [], timeout_ms=1
            )
        else:
            pending = caller.submit(backend.batched_submit_put_task, [], [])
        try:
            assert entered.wait(5), "the caller was not admitted"
            backend.close()
            assert allocator.close_calls == 0
            assert not backend._core._closed
            assert len(_retained_now(before)) == 1
            resume.set()
            assert pending.result(timeout=5) is (
                False if operation == "adopt" else None
            )
        finally:
            resume.set()


def test_close_refuses_a_reader_before_allocating(
    backend: RustRawBlockBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not allocate a new read destination after allocator shutdown."""
    backend.close()

    def unexpected_read(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a closed backend attempted to prepare a read")

    monkeypatch.setattr(backend._core, "get_metadata_prefix", unexpected_read)
    key = CacheEngineKey("test_model", 1, 0, 9010, torch.bfloat16)
    assert backend.get_blocking(key) is None
