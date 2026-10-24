# SPDX-License-Identifier: Apache-2.0
"""Qualify retained raw imports with real CUDA copies in a separate process.

Receiving identity is supplied explicitly. These are copy/cleanup primitives,
not public raw registration RPCs, exporter receipts or model-serving tests.
"""

# Standard
from multiprocessing.connection import Connection
import multiprocessing as mp
import traceback

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import (
    DRAML1ManagerConfig,
    EvictionConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.multiprocess import kmesh_device_copy as copy_module
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.kmesh_device_copy import RetainedTensorCopies
from lmcache.v1.multiprocess.kmesh_domains import KMeshFenceError
from lmcache.v1.multiprocess.modules.blend.rope import BorrowedRope, provisional_rope
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    BorrowedContext,
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.transport.request_context import request_session
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.devices.cuda.ipc_wrapper import RawCudaIPCWrapper
from lmcache.v1.platform.devices.cuda.utils import _cuda
from tests.v1.test_kmesh_device_copy_gpu import receive

pytestmark = pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires raw CUDA IPC and actual stream copies",
)


def validate_raw_metadata(paged: BorrowedContext, rotary: BorrowedRope) -> None:
    """Check this fixture's metadata without reading device values."""
    assert paged.entry.model_name == "synthetic-raw-copy"
    assert all(
        cache.device == paged.entry.cache_context.device
        and tuple(cache.shape) == (64, 8)
        and cache.dtype == torch.float32
        and cache.is_contiguous()
        for cache in rotary.state.cos_sin_caches
    )


def raw_copy_server(channel: Connection, failure: str) -> None:
    try:
        run_raw_copy_server(channel, failure)
    except BaseException:
        channel.send(("error", traceback.format_exc()))
        raise
    else:
        channel.send(("closed",))
    finally:
        channel.close()


def run_raw_copy_server(channel: Connection, failure: str) -> None:
    context = MPCacheServerContext(
        StorageManagerConfig(
            DRAML1ManagerConfig(L1MemoryManagerConfig(8 << 20, False, shm_name="")),
            EvictionConfig("LRU"),
        ),
        chunk_size=8,
        hash_algorithm="blake3",
    )
    transfer = LMCacheDrivenTransferModule(context)
    manager = RetainedTensorCopies()
    borrowed: tuple[BorrowedContext, BorrowedRope] | None = None
    consumed = False
    try:
        kv_wrapper, owners = channel.recv()
        with request_session("zmq", b"raw-copy-worker"):
            transfer.register_kv_cache(
                99,
                [kv_wrapper],
                "synthetic-raw-copy",
                1,
                EngineType.VLLM,
                {"kv_layout": "NHD"},
                [],
            )
            identity = transfer.context_identity(99)
            assert identity is not None
            paged = transfer.borrow_context(99, identity, lambda: None)
            rotary = provisional_rope(
                identity, [], 8, True, [0], [], [], "", tuple(owners)
            )
            borrowed = paged, rotary
        backend = paged.entry.event_backend
        assert backend is not None
        channel.send(("registered",))
        event = channel.recv()
        imported = backend.import_event(event, paged.entry.cache_context.device)
        pending = not backend.query_event(imported)
        close_attempts: list[int] = []
        native_close = _cuda.runtime.cudaIpcCloseMemHandle
        reference_close = RawCudaIPCWrapper.close_checked
        raw_import = RawCudaIPCWrapper.to_tensor
        tensor_copy = torch.Tensor.copy_
        imports = 0

        def observe_close(pointer: int) -> object:
            close_attempts.append(pointer)
            return native_close(pointer)

        def refuse_reference_close(owner: RawCudaIPCWrapper) -> None:
            if failure in ("close-refusal", "partial-import-close-refusal"):
                raise RuntimeError("injected before checked raw reference release")
            reference_close(owner)

        def partial_import(owner: RawCudaIPCWrapper) -> torch.Tensor:
            nonlocal imports
            imports += 1
            if imports == 2:
                raise RuntimeError("injected second raw import failure")
            return raw_import(owner)

        def copy_then_fail(
            target: torch.Tensor, source: torch.Tensor, non_blocking: bool = False
        ) -> torch.Tensor:
            result = tensor_copy(target, source, non_blocking=non_blocking)
            if target.device.type == "cpu" and source.device.type == "cuda":
                raise RuntimeError("injected after actual raw snapshot copy")
            return result

        def refuse_drain(device: object, stream: object) -> None:
            raise RuntimeError("injected uncertain raw snapshot drain")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_cuda.runtime, "cudaIpcCloseMemHandle", observe_close)
            patch.setattr(RawCudaIPCWrapper, "close_checked", refuse_reference_close)
            if failure == "partial-import-close-refusal":
                patch.setattr(RawCudaIPCWrapper, "to_tensor", partial_import)
            if failure == "copy-undrained":
                patch.setattr(torch.Tensor, "copy_", copy_then_fail)
                patch.setattr(copy_module, "synchronize_stream", refuse_drain)
            if failure == "success":
                consumed = True
                snapshots = manager.copy_raw_to_cpu(
                    paged, rotary, event, validate_raw_metadata
                )
                assert all(owner.raw_mapping_released for owner in owners)
                channel.send(
                    (
                        "copied",
                        pending,
                        snapshots[0].view(torch.uint8).numpy().tobytes(),
                    )
                )
            else:
                with pytest.raises(KMeshFenceError):
                    consumed = True
                    manager.copy_raw_to_cpu(paged, rotary, event, validate_raw_metadata)
                assert not owners[0].raw_mapping_released
                debt = (
                    manager.pending,
                    paged.entry.ownership.pending,
                    rotary.state.ownership.pending,
                )
                assert not close_attempts, "native close ran before qualified cleanup"
                channel.send(("failed", pending, debt))
                assert channel.recv() == "drain"
        assert manager.drain_quarantined(identity) == 0
        assert all(owner.raw_mapping_released for owner in owners)
        assert paged.entry.ownership.pending == rotary.state.ownership.pending == 0
    finally:
        if borrowed is not None and not consumed:
            borrowed[1].release()
            borrowed[0].release()
        manager.close()
        transfer.close()
        context.close()


@pytest.mark.parametrize(
    "failure",
    [
        "success",
        "close-refusal",
        "copy-undrained",
        "partial-import-close-refusal",
    ],
)
def test_raw_input_copy_retains_imports_through_cleanup(failure: str) -> None:
    device = torch.device("cuda", 0)
    kv = torch.zeros((2, 16, 4, 2, 8), device=device, dtype=torch.bfloat16)
    # A dedicated allocation makes rotary close independent of the paged alias.
    pool = torch.empty(64 * (1 << 20) // 4, device=device, dtype=torch.float32)
    rope = pool[:512].reshape(64, 8)
    rope.fill_(float("nan"))
    torch_dev.synchronize(device)
    kv_wrapper = RawCudaIPCWrapper.wrap(kv)
    owners = [
        RawCudaIPCWrapper.wrap(rope)
        for _ in range(2 if failure == "partial-import-close-refusal" else 1)
    ]
    assert kv_wrapper.allocation_identity != owners[0].allocation_identity
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(
        target=raw_copy_server, args=(child, failure)
    )
    process.start()
    child.close()
    try:
        parent.send((kv_wrapper, owners))
        assert receive(parent) == ("registered",)
        phases = (
            torch.arange(64, dtype=torch.float32)[:, None] * torch.arange(1, 5) * 0.03
        )
        expected = torch.cat((phases.cos(), phases.sin()), dim=1).pin_memory()
        stream = torch_dev.Stream(device=device)
        backend = get_event_ipc_backend(device)
        ready = backend.create_event(device)
        matrix = torch.full((2048, 2048), 1 / 2048, device=device)
        workspace = torch.empty_like(matrix)
        torch_dev.synchronize(device)
        with torch_dev.stream(stream):
            for _ in range(512):
                torch.mm(matrix, matrix, out=workspace)
            rope.copy_(expected, non_blocking=True)
            backend.record_event(ready, stream)
        parent.send(backend.export_event(ready, device))
        report = receive(parent)
        assert report[1], "producer challenge already drained"
        if failure == "success":
            assert report == (
                "copied",
                True,
                expected.view(torch.uint8).numpy().tobytes(),
            )
        else:
            assert report == ("failed", True, (1, 1, 1))
            parent.send("drain")
        assert receive(parent) == ("closed",)
        process.join(timeout=30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=30)
        parent.close()
        # Keep allocations and the producer event alive until importer exit.
        torch_dev.synchronize(device)
