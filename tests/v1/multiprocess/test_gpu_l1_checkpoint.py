# SPDX-License-Identifier: Apache-2.0
"""Exercise separate MP GPU servers with CUDA IPC and an owned storage file.

Opt in with LMCACHE_RUN_MP_REGISTERED_IO=1 on qualified NVMe-backed XFS or
ext4. LMCACHE_MP_GPU_PROVIDER selects the GPU exporter. The test uses raw
CUDA IPC for engine tensors and ordinary io_uring for the GPU L1 arena.
"""

# Standard
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any
import multiprocessing as mp
import os
import random
import socket
import time
import traceback

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        os.environ.get("LMCACHE_RUN_MP_REGISTERED_IO") != "1",
        reason="real MP GPU registered I/O requires explicit opt-in",
    ),
]


def server_worker(
    path: str, port: int, serde: bool, provider: str, pipe: Connection
) -> None:
    """Run one server incarnation and report storage and close ownership."""
    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        GPUL1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
        RawBlockL2AdapterConfig,
    )
    from lmcache.v1.distributed.serde import SerdeConfig
    from lmcache.v1.mp_observability.config import ObservabilityConfig
    from lmcache.v1.multiprocess.config import MPServerConfig
    from lmcache.v1.multiprocess.server import run_cache_server

    try:
        l1 = GPUL1ManagerConfig(
            L1MemoryManagerConfig(16 << 20, False, shm_name=""),
            tag="gpu",
            buffer_provider=provider,
        )
        raw = RawBlockL2AdapterConfig(
            device_path=path,
            slot_bytes=262144,
            meta_total_bytes=1 << 20,
            use_odirect=True,
            io_engine="io_uring",
            meta_enable_periodic=False,
        )
        raw.affinity_tag = "gpu"
        if serde:
            raw.serde_config = SerdeConfig("fp8")
        config = StorageManagerConfig(
            l1, EvictionConfig("noop"), L2AdaptersConfig([raw])
        )
        result = run_cache_server(
            MPServerConfig(
                host="127.0.0.1",
                port=port,
                chunk_size=256,
                supported_transfer_mode="lmcache_driven",
            ),
            config,
            ObservabilityConfig(),
            return_engine=True,
            start_prometheus_http_server=False,
        )
        assert result is not None
        server, engine = result
        pipe.send({"ready": True, "pid": os.getpid()})
        try:
            while True:
                command = pipe.recv()
                if command == "status":
                    adapter = engine.storage_manager.l2_adapters()[0][1]
                    pipe.send(
                        {
                            "l1": engine.storage_manager.report_status()["l1_manager"],
                            "io": adapter.report_status()["core"]["io_by_request"],
                            "used_l2": adapter.get_usage().total_bytes_used,
                        }
                    )
                elif command == "close":
                    server.close()
                    engine.close()
                    pipe.send(
                        {
                            "closed": True,
                            "l1": engine.storage_manager.report_status()["l1_manager"],
                        }
                    )
                    return
                else:
                    raise ValueError(command)
        finally:
            server.close()
            engine.close()
    except Exception:
        pipe.send({"error": traceback.format_exc()})
        raise
    finally:
        pipe.close()


def receive(pipe: Connection) -> dict[str, Any]:
    """Receive one server reply within a bounded deadline."""
    assert pipe.poll(45), "server response deadline expired"
    response = pipe.recv()
    assert "error" not in response, response
    return response


def get_status(pipe: Connection) -> dict[str, Any]:
    """Request live ownership and native journal status."""
    pipe.send("status")
    return receive(pipe)


@pytest.mark.parametrize("serde", [False, True])
def test_gpu_checkpoint_between_server_processes(tmp_path: Path, serde: bool) -> None:
    """Gather, checkpoint, reopen, prefetch and scatter exact synthetic KV bytes."""
    # First Party
    from lmcache.v1.platform.devices.cuda.ipc_wrapper import RawCudaIPCWrapper

    provider = os.environ.get("LMCACHE_MP_GPU_PROVIDER", "native")
    media = tmp_path / "kv.bin"
    with media.open("xb") as handle:
        handle.write(bytes(16 << 20))
        handle.flush()
        os.fsync(handle.fileno())
    tensors = [
        torch.empty((2, 32, 16, 1, 32), dtype=torch.bfloat16, device="cuda:0")
        for _ in range(2)
    ]
    for layer, tensor in enumerate(tensors):
        # Layer-seeded values distinguish pages and K/V while remaining exact in FP8.
        rng = random.Random(layer)
        tensor.copy_(
            torch.tensor(
                [rng.randrange(16) for _ in range(tensor.numel())],
                dtype=tensor.dtype,
                device=tensor.device,
            ).reshape(tensor.shape)
        )
    expected = [tensor[:, :16].clone() for tensor in tensors]
    torch_dev.synchronize()
    event_backend = get_event_ipc_backend(0)
    event_owners = []
    context = mp.get_context("spawn")
    key = IPCCacheServerKey.from_token_ids(
        "testmodel", 1, 0, [77] * 256, start=0, end=256, request_id="gpu-checkpoint"
    )
    payload_receipts = []
    for phase in ("prefill", "decode"):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        parent, child = context.Pipe()
        process = context.Process(
            target=server_worker, args=(str(media), port, serde, provider, child)
        )
        process.start()
        child.close()
        client = None
        try:
            assert receive(parent)["ready"]
            client = RequestClientFactory.create(f"tcp://127.0.0.1:{port}")
            instance = os.getpid()
            assert (
                client.register_kv_cache(
                    instance,
                    [RawCudaIPCWrapper(t) for t in tensors],
                    "testmodel",
                    1,
                    EngineType.VLLM,
                    {},
                    [],
                ).result(timeout=20)
                is None
            )
            event = event_backend.create_event(0)
            event_owners.append(event)
            if phase == "prefill":
                event_backend.record_event(event, None)
                assert (
                    client.store(
                        key,
                        instance,
                        [list(range(16))],
                        event_backend.export_event(event, 0),
                    )
                    .to_device_future()
                    .result(timeout=20)
                )
                deadline = time.monotonic() + 20
                while True:
                    receipt = get_status(parent)
                    if (
                        receipt["used_l2"]
                        and receipt["l1"]["external_io_lease_count"] == 0
                    ):
                        break
                    assert time.monotonic() < deadline, receipt
                    time.sleep(0.01)
            else:
                for tensor in tensors:
                    tensor[:, 16:].fill_(23)
                torch_dev.synchronize()
                event_backend.record_event(event, None)
                client.lookup(key.no_worker_id_version(), 1).result(timeout=20)
                deadline = time.monotonic() + 20
                while (
                    hits := client.query_prefetch_status(key.request_id).result(
                        timeout=20
                    )
                ) is None:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                assert hits == 1
                assert (
                    client.retrieve(
                        key,
                        instance,
                        [list(range(16, 32))],
                        event_backend.export_event(event, 0),
                        0,
                    )
                    .to_device_future()
                    .result(timeout=20)
                )
                torch_dev.synchronize()
                assert all(
                    torch.equal(t[:, 16:], oracle)
                    for t, oracle in zip(tensors, expected, strict=True)
                )
                receipt = get_status(parent)
            payloads = [
                op
                for op in receipt["io"]["operations"]
                if op["context"]["request_id"].startswith(("store-", "load-"))
                and op["path"] == "dmabuf_fixed"
            ]
            assert len(payloads) == 1 and payloads[0]["complete"]
            assert payloads[0]["direction"] == (
                "write" if phase == "prefill" else "read"
            )
            # Two layers, K+V, 256 tokens, 32 values per token, bf16 or FP8.
            assert payloads[0]["completed_bytes"] == 2 * 2 * 256 * 32 * (
                1 if serde else 2
            )
            payload_receipts.append(payloads[0])
            assert (
                receipt["io"]["dropped_rows"]
                == receipt["io"]["collection_failures"]
                == 0
            )
            client.unregister_kv_cache(instance).result(timeout=20)
            client.close()
            client = None
            parent.send("close")
            closed = receive(parent)
            assert closed["l1"]["external_io_lease_count"] == 0
            assert not closed["l1"]["backing_resources_retained"]
            process.join(20)
            assert process.exitcode == 0
        finally:
            if client is not None:
                client.close()
            if process.is_alive():
                process.terminate()
                process.join(10)
            parent.close()
    assert (
        payload_receipts[0]["context"]["writer_epoch"]
        != payload_receipts[1]["context"]["writer_epoch"]
    )
