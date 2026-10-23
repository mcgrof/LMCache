# SPDX-License-Identifier: Apache-2.0
"""Copy real registered CUDA IPC tensors in a separate server process.

Registration starts from already-drained producer tensors. A later write uses
a non-default producer stream and an actual IPC event. These tests qualify the
copy primitive, not ordered registration, public control RPCs or engine serving.
"""

# Standard
from multiprocessing.connection import Connection
from typing import Any
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
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.transport.request_context import request_session
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper

pytestmark = pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires CUDA IPC and physical device copies",
)


def receive(channel: Connection) -> Any:
    """Receive a bounded child report, surfacing its original exception."""
    assert channel.poll(60), "separate copy process did not respond"
    result = channel.recv()
    assert result[0] != "error", result
    return result


def copy_server(channel: Connection, failure: str) -> None:
    """Report child setup failures and acknowledge only completed cleanup."""
    try:
        run_copy_server(channel, failure)
    except BaseException:
        channel.send(("error", traceback.format_exc()))
        raise
    else:
        channel.send(("closed",))
    finally:
        channel.close()


def run_copy_server(channel: Connection, failure: str) -> None:
    """Own the real server registrations until its copy manager drains."""
    context = MPCacheServerContext(
        StorageManagerConfig(
            DRAML1ManagerConfig(L1MemoryManagerConfig(8 << 20, False, shm_name="")),
            EvictionConfig("LRU"),
        ),
        chunk_size=8,
        hash_algorithm="blake3",
    )
    transfer = LMCacheDrivenTransferModule(context)
    blend = BlendModule(context, transfer)
    manager = RetainedTensorCopies()
    try:
        kv_wrapper, rope_wrapper = channel.recv()
        with request_session("zmq", b"copy-worker"):
            transfer.register_kv_cache(
                99,
                [kv_wrapper],
                "synthetic-copy",
                1,
                EngineType.VLLM,
                {"kv_layout": "NHD"},
                [],
            )
            blend.cb_register_rope(99, [rope_wrapper], 8, True, [0])
            context_id, rotary_id = (
                transfer.context_identity(99),
                blend.rope_identity(99),
            )
            assert context_id is not None and rotary_id is not None
            paged = transfer.borrow_context(99, context_id, lambda: None)
            rotary = blend.borrow_rope(99, rotary_id, context_id, lambda: None)
        backend = paged.entry.event_backend
        assert backend is not None
        channel.send(("registered",))
        producer = channel.recv()
        imported = backend.import_event(producer, paged.entry.cache_context.device)
        producer_pending = not backend.query_event(imported)
        with pytest.MonkeyPatch.context() as patch:
            if failure != "success":
                copy_tensor = torch.Tensor.copy_

                def copy_then_fail(
                    target: torch.Tensor,
                    source: torch.Tensor,
                    non_blocking: bool = False,
                ) -> torch.Tensor:
                    result = copy_tensor(target, source, non_blocking=non_blocking)
                    if target.device.type == "cpu" and source.device.type == "cuda":
                        raise RuntimeError("injected after real host copy submission")
                    return result

                patch.setattr(torch.Tensor, "copy_", copy_then_fail)
            if failure == "copy-undrained":

                def refuse_drain(device: object, stream: object) -> None:
                    raise RuntimeError("injected uncertain stream drain")

                patch.setattr(copy_module, "synchronize_stream", refuse_drain)
            if failure == "success":
                snapshots = manager.copy_to_cpu(paged, rotary, producer)
                channel.send(
                    (
                        "copied",
                        producer_pending,
                        snapshots[0].view(torch.uint8).numpy().tobytes(),
                    )
                )
            else:
                with pytest.raises(KMeshFenceError, match="copy failed"):
                    manager.copy_to_cpu(paged, rotary, producer)
                debt = (
                    manager.pending,
                    paged.entry.ownership.pending,
                    rotary.state.ownership.pending,
                )
                channel.send(("failed", debt))
                assert channel.recv() == "drain"
        assert manager.drain_quarantined() == 0
        assert paged.entry.ownership.pending == rotary.state.ownership.pending == 0
    finally:
        manager.close()
        blend.close()
        transfer.close()
        context.close()


@pytest.mark.parametrize("failure", ["success", "copy-drained", "copy-undrained"])
def test_registered_ipc_copy_and_failed_drain_retain_the_exporter(failure: str) -> None:
    device = torch.device("cuda", 0)
    kv = torch.zeros((2, 16, 4, 2, 8), dtype=torch.bfloat16, device=device)
    rope = torch.zeros((64, 8), dtype=torch.float32, device=device)
    rope[:, :4] = 1
    # Legacy registration reads its input: drain initialization before exporting.
    torch_dev.synchronize(device)
    wrappers = (CudaIPCWrapper.wrap(kv), CudaIPCWrapper.wrap(rope))
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=copy_server, args=(child, failure))
    process.start()
    child.close()
    try:
        parent.send(wrappers)
        assert receive(parent) == ("registered",)
        phases = (
            torch.arange(64, dtype=torch.float32)[:, None] * torch.arange(1, 5) * 0.03
        )
        expected = torch.cat((phases.cos(), phases.sin()), dim=1).pin_memory()
        producer_stream = torch_dev.Stream(device=device)
        backend = get_event_ipc_backend(device)
        ready = backend.create_event(device)
        # A bounded synthetic matrix sequence keeps the producer pending while
        # the already-initialized child receives the event; this is not a model.
        matrix = torch.full((2048, 2048), 1 / 2048, device=device)
        workspace = torch.empty_like(matrix)
        torch_dev.synchronize(device)
        with torch_dev.stream(producer_stream):
            for _ in range(512):
                torch.mm(matrix, matrix, out=workspace)
            rope.copy_(expected, non_blocking=True)
            backend.record_event(ready, producer_stream)
        parent.send(backend.export_event(ready, device))
        result = receive(parent)
        if failure == "success":
            assert result[:2] == ("copied", True), "producer challenge already drained"
            assert result[2] == expected.view(torch.uint8).numpy().tobytes()
        else:
            expected_debt = (1, 1, 1) if failure == "copy-undrained" else (0, 0, 0)
            assert result == ("failed", expected_debt)
            parent.send("drain")
        assert receive(parent) == ("closed",)
        process.join(timeout=30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=30)
        parent.close()
        # Keep all exporter tensors/events alive until the child has exited.
        torch_dev.synchronize(device)
