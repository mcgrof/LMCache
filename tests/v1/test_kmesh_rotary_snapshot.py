# SPDX-License-Identifier: Apache-2.0
"""Qualify ordered CPU registration; these tests allocate no device memory."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess import kmesh_bindings as bindings_module
from lmcache.v1.multiprocess.context_ownership import ContextOwnershipError
from lmcache.v1.multiprocess.kmesh_bindings import RegisteredWorkerBindings
from lmcache.v1.multiprocess.kmesh_domains import DomainRegistry, KMeshFenceError
from lmcache.v1.multiprocess.kmesh_rotary_snapshot import normalize_rope_snapshot
from lmcache.v1.multiprocess.posix_shm import shm_unlink
from lmcache.v1.multiprocess.transport.request_context import request_session
from lmcache.v1.platform.devices.cpu.shm import migrate_to_shm_and_wrap
from tests.v1.test_registered_rope_ownership import rotary_server


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_snapshot_normalization_keeps_dtype_and_unit_rotation(
    dtype: torch.dtype, scale: float
) -> None:
    cache = torch.zeros((64, 8), dtype=dtype)
    cache[:, :4] = scale
    normalized = normalize_rope_snapshot(cache)
    assert normalized.dtype == dtype
    assert torch.equal(normalized[:, :4], torch.ones((64, 4), dtype=dtype))
    assert normalized[:, 4:].count_nonzero() == 0
    assert torch.equal(cache[:, :4], torch.full((64, 4), scale, dtype=dtype))


@pytest.mark.parametrize("bad", ["zero", "nan", "inf", "odd", "empty"])
def test_normalization_refuses_unusable_completed_snapshots(bad: str) -> None:
    cache = torch.ones((64, 7 if bad == "odd" else 8))
    if bad == "empty":
        cache = cache[:0]
    elif bad in ("zero", "nan", "inf"):
        cache.fill_({"zero": 0.0, "nan": float("nan"), "inf": float("inf")}[bad])
    with pytest.raises(KMeshFenceError):
        normalize_rope_snapshot(cache)


def test_registration_publishes_independent_normalized_exact_generation() -> None:
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        cache = torch.zeros((64, 8))
        cache[:, 4:] = 2
        wrapper = migrate_to_shm_and_wrap(cache)
        try:
            with request_session("zmq", b"worker"):
                context = bindings.context_identity(99)
                old = bindings.describe(99)
                runtime = bindings.register_rotary_snapshot(
                    99, context, [wrapper], 8, True, [0], [], [], "", b""
                )
                assert runtime.context == context
                assert runtime.rotary != old.rotary
                assert runtime.layout_digest == old.layout_digest
                assert runtime.rope_digest != old.rope_digest
                wrapper.to_tensor().fill_(17)
                assert bindings.describe(99) == runtime
                rope = blend.borrow_rope(99, runtime.rotary, context, lambda: None)
                try:
                    expected = torch.zeros((64, 8))
                    expected[:, 4:] = 1
                    assert torch.equal(rope.state.cos_sin_caches[0], expected)
                    assert rope.state.cos_sin_caches[0].data_ptr() != cache.data_ptr()
                finally:
                    rope.release()
                with pytest.raises(KMeshFenceError, match="no retained copy"):
                    bindings.retry_rotary_drain(context)
        finally:
            shm_unlink(wrapper.shm_name)
            bindings.close()


@pytest.mark.parametrize("bad", ["foreign", "stale", "width", "map", "nan"])
def test_failed_preparation_preserves_previous_runtime(bad: str) -> None:
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        cache = torch.ones((64, 8))
        wrapper = migrate_to_shm_and_wrap(cache)
        try:
            with request_session("zmq", b"worker"):
                context = bindings.context_identity(99)
                previous = bindings.describe(99)
            if bad == "stale":
                context = replace(context, generation="old")
            if bad == "nan":
                wrapper.to_tensor().fill_(float("nan"))
            with request_session("zmq", b"other" if bad == "foreign" else b"worker"):
                with pytest.raises((KMeshFenceError, ContextOwnershipError)):
                    bindings.register_rotary_snapshot(
                        99,
                        context,
                        [wrapper],
                        16 if bad == "width" else 8,
                        True,
                        [1] if bad == "map" else [0],
                        [],
                        [],
                        "",
                        b"",
                    )
            with request_session("zmq", b"worker"):
                assert bindings.describe(99) == previous
        finally:
            shm_unlink(wrapper.shm_name)
            bindings.close()


def test_preparation_blocks_clear_and_cannot_publish_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, resume = threading.Event(), threading.Event()
    normalize = normalize_rope_snapshot

    def paused(cache: torch.Tensor) -> torch.Tensor:
        entered.set()
        assert resume.wait(10)
        return normalize(cache)

    monkeypatch.setattr(bindings_module, "normalize_rope_snapshot", paused)
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        cache = torch.ones((64, 8))
        wrapper = migrate_to_shm_and_wrap(cache)
        with request_session("zmq", b"worker"):
            context = bindings.context_identity(99)

        def prepare() -> None:
            with request_session("zmq", b"worker"):
                bindings.register_rotary_snapshot(
                    99, context, [wrapper], 8, True, [0], [], [], "", b""
                )

        try:
            with ThreadPoolExecutor(1) as pool:
                task = pool.submit(prepare)
                try:
                    assert entered.wait(10)
                    with pytest.raises(KMeshFenceError):
                        bindings.clear_store(
                            lambda: pytest.fail("clear during prepare")
                        )
                    with pytest.raises(KMeshFenceError):
                        bindings.close()
                finally:
                    resume.set()
                with pytest.raises(KMeshFenceError, match="retired"):
                    task.result(timeout=10)
            bindings.close()
        finally:
            resume.set()
            shm_unlink(wrapper.shm_name)
