# SPDX-License-Identifier: Apache-2.0
"""Exercise real CPU KV/rotary registrations with simulated dispatch context.

The ContextVar supplies a logical receiving client; these tests do not transport
device events or prove GPU completion. Holds deliberately simulate that debt.
"""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import (
    DRAML1ManagerConfig,
    EvictionConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.multiprocess.context_ownership import (
    ContextOwnership,
    ContextOwnershipError,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.posix_shm import shm_unlink
from lmcache.v1.multiprocess.transport.request_context import request_session
from lmcache.v1.platform.devices.cpu.shm import migrate_to_shm_and_wrap


@contextmanager
def rotary_server(
    owner: bytes | None = b"worker",
    *,
    hash_algorithm: str = "sha256_cbor",
    world_size: int = 1,
    num_blocks: int = 16,
) -> Iterator[tuple[BlendModule, LMCacheDrivenTransferModule]]:
    # Eight MiB bounds a synthetic DRAM store; registration performs no storage IO.
    context = MPCacheServerContext(
        StorageManagerConfig(
            DRAML1ManagerConfig(L1MemoryManagerConfig(8 << 20, False, shm_name="")),
            EvictionConfig("LRU"),
        ),
        chunk_size=8,
        hash_algorithm=hash_algorithm,
    )
    transfer = LMCacheDrivenTransferModule(context)
    blend = BlendModule(context, transfer)
    kv = torch.zeros((2, num_blocks, 4, 2, 8), dtype=torch.bfloat16)
    rotary = torch.zeros((64, 8), dtype=torch.float32)
    rotary[:, :4] = 1
    wrappers = [migrate_to_shm_and_wrap(tensor) for tensor in (kv, rotary)]
    try:
        with request_session("zmq", owner) if owner is not None else nullcontext():
            transfer.register_kv_cache(
                99,
                [wrappers[0]],
                "synthetic-model",
                world_size,
                EngineType.VLLM,
                {"kv_layout": "NHD"},
                [],
            )
            blend.cb_register_rope(99, [wrappers[1]], 8, True, [0])
        yield blend, transfer
    finally:
        blend.close()
        transfer.close()
        context.close()
        for wrapper in wrappers:
            shm_unlink(wrapper.shm_name)


@pytest.mark.parametrize("mismatch", ["client", "rotary", "context"])
def test_borrow_requires_rotary_context_and_actual_owner(mismatch: str) -> None:
    with rotary_server() as (blend, transfer):
        expected = blend.rope_identity(99)
        context = transfer.context_identity(99)
        assert expected is not None and context is not None
        owner = b"foreign" if mismatch == "client" else b"worker"
        if mismatch == "rotary":
            expected = replace(
                expected, rotary=replace(expected.rotary, generation="stale")
            )
        if mismatch == "context":
            context = replace(context, generation="replacement")
        with request_session("zmq", owner):
            with pytest.raises(ContextOwnershipError):
                blend.borrow_rope(99, expected, context, lambda: None)


def test_ownerless_legacy_rope_cannot_grant_a_borrow() -> None:
    with rotary_server(owner=None) as (blend, transfer):
        expected = blend.rope_identity(99)
        context = transfer.context_identity(99)
        assert expected is not None and context is not None
        with request_session("zmq", b"worker"):
            with pytest.raises(ContextOwnershipError):
                blend.borrow_rope(99, expected, context, lambda: None)


@pytest.mark.parametrize("retirement", ["unregister", "reap", "close"])
def test_rotary_retirement_waits_for_exact_borrowed_tensors(retirement: str) -> None:
    with rotary_server() as (blend, transfer):
        expected = blend.rope_identity(99)
        context = transfer.context_identity(99)
        assert expected is not None and context is not None
        revoked = threading.Event()
        with request_session("zmq", b"worker"):
            paged = transfer.borrow_context(99, context, lambda: None)
            borrowed = blend.borrow_rope(99, expected, context, revoked.set)

        def retire() -> None:
            if retirement == "unregister":
                blend.cb_unregister_rope(99)
            elif retirement == "reap":
                blend.drop_instance_state(99)
            else:
                blend.close()

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(retire)
                try:
                    assert revoked.wait(timeout=5)
                    assert not future.done()
                    assert (
                        borrowed.state.cos_sin_caches[0][0].tolist()
                        == [1.0] * 4 + [0.0] * 4
                    )
                    with request_session("zmq", b"worker"):
                        with pytest.raises(ContextOwnershipError):
                            blend.borrow_rope(99, expected, context, lambda: None)
                finally:
                    borrowed.release()
                future.result(timeout=5)
        finally:
            borrowed.release()
            paged.release()
        assert blend.rope_identity(99) is None


def test_concurrent_rotary_replacement_cannot_overwrite_new_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with rotary_server() as (blend, transfer):
        original = blend.rope_identity(99)
        assert original is not None
        barrier = threading.Barrier(2, timeout=5)
        retire = ContextOwnership.retire

        def meet(ownership: ContextOwnership, timeout: float = 5.0) -> None:
            barrier.wait()
            retire(ownership, timeout)

        def register() -> None:
            with request_session("zmq", b"worker"):
                blend.cb_register_rope(99, [], 8, True, [-1])

        with monkeypatch.context() as patch:
            patch.setattr(ContextOwnership, "retire", meet)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(register) for _ in range(2)]
                results = []
                for future in futures:
                    try:
                        future.result(timeout=5)
                    except ContextOwnershipError:
                        results.append("stale")
                    else:
                        results.append("installed")
        assert sorted(results) == ["installed", "stale"]
        current = blend.rope_identity(99)
        context = transfer.context_identity(99)
        assert current is not None and current != original and context is not None
        with request_session("zmq", b"worker"):
            with pytest.raises(ContextOwnershipError):
                blend.borrow_rope(99, original, context, lambda: None)
            paged = transfer.borrow_context(99, context, lambda: None)
            borrowed = blend.borrow_rope(99, current, context, lambda: None)
            try:
                assert borrowed.state.cos_sin_caches == []
            finally:
                borrowed.release()
                paged.release()
