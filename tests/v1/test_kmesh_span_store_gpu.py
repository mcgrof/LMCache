# SPDX-License-Identifier: Apache-2.0
"""Exercise actual native span I/O through separate CUDA service processes."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import wraps
from multiprocessing.connection import Connection
from typing import Any, Literal
import math
import multiprocessing as mp
import traceback
import uuid

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.config import (
    DRAML1ManagerConfig,
    EvictionConfig,
    GPUL1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess import kmesh_span_store as span_module
from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper
from lmcache.v1.multiprocess.kmesh_bindings import RegisteredWorkerBindings
from lmcache.v1.multiprocess.kmesh_domains import RegistrationState, WorkerFence
from lmcache.v1.multiprocess.kmesh_native_store import RegisteredSpanGather
from lmcache.v1.multiprocess.kmesh_raw_server import create_managed_raw_server
from lmcache.v1.multiprocess.kmesh_types import RegisteredRuntime, SpanOperationStatus
from lmcache.v1.multiprocess.modules.blend.retrieve import RegisteredRetrievePlan
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.grpc_impl.server import GrpcMultiprocessServer
from lmcache.v1.multiprocess.transport.zmq_impl.mq import MessageQueueServer
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.devices.cuda.ipc_wrapper import RawCudaIPCWrapper
from tests.v1.test_kmesh_bindings import semantics
from tests.v1.test_kmesh_control import value
from tests.v1.test_kmesh_rotary_snapshot_gpu import read_snapshot

pytestmark = pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires actual native CUDA span transfers and raw IPC",
)


@dataclass(frozen=True)
class SpanConsumerCase:
    """Expose real completed placement to a separately selected consumer test."""

    fence: WorkerFence
    runtime: RegisteredRuntime
    receipt: SpanOperationStatus
    kv: torch.Tensor
    blocks: tuple[int, ...]
    destination_start: int
    token_count: int
    layout: Literal["NHD", "HND"]


def span_service(
    channel: Connection,
    transport: Literal["zmq", "grpc"],
    mode: str,
    source_memory: Literal["lazy-host", "eager-host", "vulkan_rm"] = "lazy-host",
) -> None:
    """Observe real calls; inject refusals without manufacturing device faults."""
    armed = {"enabled": False}
    observed = {"enqueues": 0, "gathers": 0, "drains": 0}
    gather = span_module.gather_registered_span
    enqueue = RegisteredRetrievePlan.enqueue
    drain = span_module.synchronize_device
    read = StorageManager.unsafe_read
    prefetch = StorageManager.submit_prefetch_task
    admit = StorageManager.finish_write_with_status
    publish = RegisteredWorkerBindings.publish

    @wraps(gather)
    def actual_gather(*args: Any, **kwargs: Any) -> RegisteredSpanGather:
        observed["gathers"] += 1
        result = gather(*args, **kwargs)
        if armed["enabled"] and mode == "source-undrained":
            raise RuntimeError("injected after actual native source gather")
        return result

    @wraps(enqueue)
    def actual_enqueue(plan: RegisteredRetrievePlan) -> None:
        observed["enqueues"] += 1
        enqueue(plan)
        if armed["enabled"] and mode in ("enqueue-failure", "drain-refusal"):
            raise RuntimeError("injected after actual native span enqueue")

    @wraps(drain)
    def actual_drain(device: object) -> None:
        if armed["enabled"] and mode in ("drain-refusal", "source-undrained"):
            raise RuntimeError("injected before span device drain")
        drain(device)
        observed["drains"] += 1

    @wraps(read)
    def actual_read(self: StorageManager, *args: Any, **kwargs: Any) -> Any:
        result = read(self, *args, **kwargs)
        if armed["enabled"] and mode == "corrupt" and result[1]:
            tensor = result[1][0].get_tensor(0)
            assert tensor is not None
            tensor.flatten()[0] += 1
        return result

    @wraps(prefetch)
    def actual_prefetch(
        self: StorageManager, spec: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if armed["enabled"] and mode == "evict":
            keys = spec.key_groups[0].keys
            assert self.delete_l1_keys(keys) == (len(keys), 0)
        return prefetch(self, spec, *args, **kwargs)

    @wraps(admit)
    def actual_admit(self: StorageManager, completion: Any) -> Any:
        result = admit(self, completion)
        if armed["enabled"] and mode == "admission-refusal":
            result = admit(self, completion)
            assert all(v == L1Error.KEY_NOT_EXIST for v in result.values())
        return result

    @wraps(publish)
    def actual_publish(
        self: RegisteredWorkerBindings, operation: Any, commit: Any
    ) -> None:
        if armed["enabled"] and mode == "stale-publication":
            self.revoke(operation.ticket.fence.lease)
        publish(self, operation, commit)

    server = engine = None
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(span_module, "gather_registered_span", actual_gather)
            patch.setattr(RegisteredRetrievePlan, "enqueue", actual_enqueue)
            patch.setattr(span_module, "synchronize_device", actual_drain)
            patch.setattr(StorageManager, "unsafe_read", actual_read)
            patch.setattr(StorageManager, "submit_prefetch_task", actual_prefetch)
            patch.setattr(StorageManager, "finish_write_with_status", actual_admit)
            patch.setattr(RegisteredWorkerBindings, "publish", actual_publish)
            storage = None
            if source_memory != "lazy-host":
                memory = L1MemoryManagerConfig(
                    64 << 20,
                    False,
                    shm_name="",
                    buffer_registration="fixed"
                    if source_memory == "eager-host"
                    else "dmabuf",
                )
                l1 = (
                    DRAML1ManagerConfig(memory)
                    if source_memory == "eager-host"
                    else GPUL1ManagerConfig(memory, buffer_provider="vulkan_rm")
                )
                storage = StorageManagerConfig(l1, EvictionConfig("LRU"))
            server, engine = create_managed_raw_server(
                port=0,
                transport=transport,
                chunk_tokens=8,
                dram_bytes=64 << 20,
                enable_span_execution=True,
                storage_config=storage,
            )
            server.start()
            if isinstance(server, GrpcMultiprocessServer):
                url = f"grpc://127.0.0.1:{server.bound_port}"
            else:
                assert isinstance(server, MessageQueueServer)
                url = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
            channel.send(url)
            while True:
                assert channel.poll(90), "span exporter stopped controlling service"
                command = channel.recv()
                if command == "stop":
                    break
                if command == "arm":
                    armed["enabled"] = True
                elif command == "cleanup":
                    armed["enabled"] = False
                else:
                    assert command == "observe"
                channel.send(dict(observed))
            server.close()
            server = None
            engine.close()
            engine = None
            channel.send("closed")
    except BaseException:
        channel.send(("error", traceback.format_exc()))
        raise
    finally:
        if server is not None:
            server.close()
        if engine is not None:
            try:
                engine.close()
            except RuntimeError:
                pass


def rotate_reference(source: torch.Tensor, delta: int) -> torch.Tensor:
    """Rotate post-RoPE keys independently and preserve values byte for byte."""
    result = source.clone()
    keys = source[:, 0].float()
    frequencies = torch.arange(1, 9, dtype=torch.float32) * 0.013
    cosine, sine = (delta * frequencies).cos(), (delta * frequencies).sin()
    first, second = keys[..., :8], keys[..., 8:]
    result[:, 0, ..., :8] = first * cosine - second * sine
    result[:, 0, ..., 8:] = first * sine + second * cosine
    return result


def place_source(
    kv: torch.Tensor,
    source: torch.Tensor,
    blocks: list[int],
    start: int,
    layout: str,
) -> None:
    """Write actual token slots on the caller's current CUDA stream."""
    for index in range(source.shape[2]):
        position = start + index
        block, offset = blocks[position // 4], position % 4
        if kv.ndim == 5:
            target = (
                kv[:, block, offset] if layout == "NHD" else kv[:, block, :, offset]
            )
            half = target.shape[-1] // 2
            target[..., :half].copy_(source[:, 0, index], non_blocking=True)
            target[..., half:].copy_(source[:, 1, index], non_blocking=True)
        else:
            target = (
                kv[:, :, block, offset]
                if layout == "NHD"
                else kv[:, :, block, :, offset]
            )
            target.copy_(source[:, :, index], non_blocking=True)


def read_window(
    kv: torch.Tensor,
    blocks: list[int],
    start: int,
    count: int,
    layout: str,
) -> torch.Tensor:
    """Read real paged token slots after the server's completed receipt."""
    rows: list[torch.Tensor] = []
    for position in range(start, start + count):
        block, offset = blocks[position // 4], position % 4
        if kv.ndim == 5:
            row = kv[:, block, offset] if layout == "NHD" else kv[:, block, :, offset]
            half = row.shape[-1] // 2
            rows.append(torch.stack((row[..., :half], row[..., half:]), dim=1))
        else:
            rows.append(
                kv[:, :, block, offset]
                if layout == "NHD"
                else kv[:, :, block, :, offset]
            )
    return torch.stack(rows, dim=2).cpu()


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
@pytest.mark.parametrize(
    "mode,source_start,destination_start,layout,layers,token_count",
    [
        (mode, 0, 3, "NHD", 1, 16)
        for mode in (
            "success",
            "corrupt",
            "evict",
            "enqueue-failure",
            "drain-refusal",
            "admission-refusal",
            "stale-publication",
        )
    ]
    + [
        ("success", 5, destination, layout, 24, 16)
        for layout in ("NHD", "HND")
        for destination in (3, 9)
    ]
    + [("source-undrained", 5, 3, layout, 24, 16) for layout in ("NHD", "HND")]
    + [
        ("success", 5, destination, layout, 24, count)
        for count in (1, 9, 15)
        for layout in ("NHD", "HND")
        for destination in (3, 9)
    ]
    + [("success", 0, 3, layout, 24, 13) for layout in ("NHD", "HND")]
    + [("source-undrained", 5, 3, layout, 24, 9) for layout in ("NHD", "HND")]
    + [("rope-capacity", 5, 23, layout, 24, 1) for layout in ("NHD", "HND")],
)
def test_managed_span_native_store_and_restore(
    transport: Literal["zmq", "grpc"],
    mode: str,
    source_start: int,
    destination_start: int,
    layout: Literal["NHD", "HND"],
    layers: int,
    token_count: int,
    consume: Callable[[SpanConsumerCase], None] | None = None,
    packed: bool = False,
    source_memory: Literal["lazy-host", "eager-host", "vulkan_rm"] = "lazy-host",
) -> None:
    device = torch.device("cuda", 0)
    # Synthetic two-plane geometry includes the model's 24-layer count.
    # Each layer has 16 blocks, 4 tokens/block, two heads, 16 elements/head.
    # Separate 64 MiB exporter allocations avoid sharing small CUDA segments.
    pool = torch.empty((64 << 20) // 2, dtype=torch.bfloat16, device=device)
    shape: tuple[int, ...] = (2, 16, 4, 2, 16) if layout == "NHD" else (2, 16, 2, 4, 16)
    if packed:
        shape = (16, 4, 2, 32) if layout == "NHD" else (16, 2, 4, 32)
    kv = pool[: layers * math.prod(shape)].reshape(layers, *shape)
    kv.fill_(-9)
    source = (
        (
            torch.arange(
                layers * 2 * token_count * 2 * 16, dtype=torch.float32
            ).reshape(layers, 2, token_count, 2, 16)
            % 53
            / 17
        )
        .to(torch.bfloat16)
        .pin_memory()
    )
    source_gpu = source.to(device)
    source_blocks = [2, 4, 6, 8, 10, 12]
    if mode == "slot-reuse":
        source_blocks = list(range(12))
    rotary_pool = torch.empty((64 << 20) // 4, dtype=torch.float32, device=device)
    rotary_tokens = 24 if mode == "rope-capacity" else 64
    rope = rotary_pool[: rotary_tokens * 16].reshape(rotary_tokens, 16)
    phases = (
        torch.arange(rotary_tokens, dtype=torch.float32)[:, None]
        * torch.arange(1, 9)
        * 0.013
    )
    rope.copy_(torch.cat((phases.cos(), phases.sin()), dim=1))
    backend = get_event_ipc_backend(device)
    ready = backend.create_event(device)
    backend.record_event(ready, torch_dev.current_stream(device))
    event = backend.export_event(ready, device)
    wrappers = [RawCudaIPCWrapper.wrap(t) for t in (*kv.unbind(0), rope)]
    cache_wrappers: list[DeviceIPCWrapper] = list(wrappers[:-1])
    kv_allocations = {w.allocation_identity for w in wrappers[:-1]}
    assert len(kv_allocations) == 1
    assert wrappers[-1].allocation_identity not in kv_allocations
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(
        target=span_service, args=(child, transport, mode, source_memory)
    )
    process.start()
    child.close()
    client = foreign = None
    try:
        url = read_snapshot(parent)
        client, foreign = (
            RequestClientFactory.create(url),
            RequestClientFactory.create(url),
        )
        server = value(client.kmesh_get_server().result(timeout=30))
        nonce = uuid.uuid4().hex
        registration = value(
            client.kmesh_register_raw_context(
                server.boot_id,
                nonce,
                99,
                cache_wrappers,
                "synthetic-span",
                layout,
            ).result(timeout=30)
        )
        runtime = value(
            client.kmesh_register_rotary(
                99,
                registration.context,
                [wrappers[-1]],
                16,
                True,
                [0],
                [],
                [],
                "",
                event,
            ).result(timeout=30)
        )
        identity = replace(semantics(runtime), key_schema="kmesh-span-v1")
        lease = value(
            client.kmesh_open_registration(
                identity, "synthetic-span-run", 1, "4" * 64
            ).result(timeout=30)
        )
        fence = value(
            client.kmesh_bind_worker(lease, 0, 99, runtime, identity, "4" * 64).result(
                timeout=30
            )
        )
        # Submit the source write behind real outstanding device work. A server
        # that ignores the recorded producer can gather the poisoned old slots.
        kv.fill_(-9)
        stream = torch_dev.Stream(device=device)
        matrix = torch.full((2048, 2048), 1 / 2048, device=device)
        workspace = torch.empty_like(matrix)
        # Load this exact strided copy before the pending-work challenge; first
        # CUDA module initialization can otherwise drain the queued producer.
        place_source(kv, source_gpu, source_blocks, source_start, layout)
        kv.fill_(-9)
        torch_dev.synchronize(device)
        with torch_dev.stream(stream):
            for _ in range(512):
                torch.mm(matrix, matrix, out=workspace)
            place_source(kv, source_gpu, source_blocks, source_start, layout)
            backend.record_event(ready, stream)
        assert not backend.query_event(ready)
        event = backend.export_event(ready, device)
        context_digest = "5" * 64
        store_args = (
            fence,
            "source",
            1,
            list(range(token_count)),
            source_start,
            context_digest,
            [source_blocks],
            event,
        )
        refused = foreign.kmesh_store_span(*store_args).result(timeout=30)
        assert refused[0] is None and refused[1]
        if mode in ("admission-refusal", "source-undrained"):
            parent.send("arm")
            read_snapshot(parent)
        stored = value(client.kmesh_store_span(*store_args).result(timeout=30))
        assert (
            value(client.kmesh_span_status(fence, "source", 1).result(timeout=30))
            == stored
        )
        assert value(client.kmesh_store_span(*store_args).result(timeout=30)) == stored
        if mode in ("admission-refusal", "source-undrained"):
            assert stored.span is None
            if mode == "admission-refusal":
                assert stored.state == "failed-drained" and "admission" in stored.error
            else:
                assert (
                    stored.state == "failed-undrained"
                    and "source gather" in stored.error
                )
                assert (
                    value(
                        client.kmesh_registration_state(lease).result(timeout=30)
                    ).pending
                    == 1
                )
                blocked = client.kmesh_retire_raw_context(
                    server.boot_id, nonce, registration.context
                ).result(timeout=30)
                assert blocked[0] is None and blocked[1]
                parent.send("cleanup")
                before = read_snapshot(parent)
                drained = value(
                    client.kmesh_drain_span(fence, "source", 1).result(timeout=30)
                )
                assert drained.state == "failed-drained" and drained.span is None
                parent.send("observe")
                after = read_snapshot(parent)
                assert after["gathers"] == before["gathers"] == 1
                assert after["drains"] == before["drains"] + 1
        else:
            assert stored.state == "stored" and stored.span is not None
            span = stored.span
            assert span.source_start == source_start and span.token_count == token_count
            changed = client.kmesh_store_span(
                fence,
                "source",
                1,
                [1] * token_count,
                source_start,
                context_digest,
                [source_blocks],
                event,
            ).result(timeout=30)
            assert changed[0] is None and "changed" in changed[1]
            kv.fill_(-9)
            backend.record_event(ready, torch_dev.current_stream(device))
            event = backend.export_event(ready, device)
            destination = [[13, 11, 9, 7, 5, 3, 1]]
            if mode == "slot-reuse":
                destination = [list(range(12))]
            altered = client.kmesh_restore_span(
                fence,
                "bad-handle",
                1,
                replace(span, token_digest="0" * 64),
                destination_start,
                context_digest,
                destination,
                event,
            ).result(timeout=30)
            assert altered[0] is None and altered[1]
            parent.send("observe")
            assert read_snapshot(parent)["enqueues"] == 0
            if mode != "success":
                parent.send("arm")
                read_snapshot(parent)
            restore_args = (
                fence,
                "destination",
                1,
                span,
                destination_start,
                "6" * 64,
                destination,
                event,
            )
            restored = value(
                client.kmesh_restore_span(*restore_args).result(timeout=30)
            )
            if mode == "success":
                assert restored.state == "loaded-needs-repair" and restored.span == span
                torch_dev.synchronize(device)
                actual = read_window(
                    kv, destination[0], destination_start, token_count, layout
                )
                expected = rotate_reference(source, destination_start - source_start)
                torch.testing.assert_close(
                    actual[:, 0], expected[:, 0], rtol=0.012, atol=0.025
                )
                assert torch.equal(
                    actual[:, 1].view(torch.uint16), source[:, 1].view(torch.uint16)
                )
                all_rows = read_window(kv, destination[0], 0, 28, layout)
                untouched = [
                    p
                    for p in range(28)
                    if not destination_start <= p < destination_start + token_count
                ]
                assert torch.all(all_rows[:, :, untouched] == -9).item()
                assert (
                    value(client.kmesh_restore_span(*restore_args).result(timeout=30))
                    == restored
                )
                parent.send("observe")
                assert read_snapshot(parent)["enqueues"] == 1
                if consume is not None:
                    consume(
                        SpanConsumerCase(
                            fence,
                            runtime,
                            restored,
                            kv,
                            tuple(destination[0]),
                            destination_start,
                            token_count,
                            layout,
                        )
                    )
                kv.fill_(-9)
                backend.record_event(ready, torch_dev.current_stream(device))
                event = backend.export_event(ready, device)
                same = value(
                    client.kmesh_restore_span(
                        fence,
                        "same-position",
                        1,
                        span,
                        source_start,
                        context_digest,
                        [source_blocks],
                        event,
                    ).result(timeout=30)
                )
                assert same.state == "loaded"
                torch_dev.synchronize(device)
                actual = read_window(
                    kv, source_blocks, source_start, token_count, layout
                )
                torch.testing.assert_close(
                    actual[:, 0], source[:, 0], rtol=0.012, atol=0.025
                )
                assert torch.equal(
                    actual[:, 1].view(torch.uint16), source[:, 1].view(torch.uint16)
                )
            else:
                assert restored.span is None
                assert restored.state == (
                    "failed-undrained" if mode == "drain-refusal" else "failed-drained"
                )
                if mode in ("corrupt", "evict", "rope-capacity", "slot-reuse"):
                    expected_error = {
                        "corrupt": "checksum",
                        "evict": "evicted",
                        "rope-capacity": "rotary capacity",
                        "slot-reuse": "working slot",
                    }[mode]
                    assert expected_error in restored.error
                    parent.send("observe")
                    assert read_snapshot(parent)["enqueues"] == 0
                status = value(
                    client.kmesh_registration_state(lease).result(timeout=30)
                )
                assert status.state == RegistrationState.REVOKED
                assert status.pending == (mode == "drain-refusal")
                if mode == "drain-refusal":
                    retirement = client.kmesh_retire_raw_context(
                        server.boot_id, nonce, registration.context
                    ).result(timeout=30)
                    assert retirement[0] is None and retirement[1]
                    parent.send("cleanup")
                    before = read_snapshot(parent)
                    drained = value(
                        client.kmesh_drain_span(fence, "destination", 1).result(
                            timeout=30
                        )
                    )
                    assert drained.state == "failed-drained"
                    assert (
                        value(
                            client.kmesh_drain_span(fence, "destination", 1).result(
                                timeout=30
                            )
                        )
                        == drained
                    )
                    parent.send("observe")
                    after = read_snapshot(parent)
                    assert after["enqueues"] == before["enqueues"] == 1
                    assert after["drains"] == before["drains"] + 1
        released = value(
            client.kmesh_retire_raw_context(
                server.boot_id, nonce, registration.context
            ).result(timeout=30)
        )
        assert set(released.allocation_ids) == {w.allocation_identity for w in wrappers}
        parent.send("stop")
        assert read_snapshot(parent) == "closed"
        process.join(30)
        assert process.exitcode == 0
    finally:
        for connection in (client, foreign):
            if connection is not None:
                connection.close()
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()
