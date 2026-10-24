# SPDX-License-Identifier: Apache-2.0
"""Exercise managed raw registration and release through both real transports."""

# Standard
from dataclasses import replace
from functools import wraps
from multiprocessing.connection import Connection
from typing import Any, Literal
import multiprocessing as mp
import traceback
import uuid

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.multiprocess import kmesh_device_copy as copy_module
from lmcache.v1.multiprocess.context_ownership import ContextIdentity
from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper
from lmcache.v1.multiprocess.kmesh_device_copy import RetainedTensorCopies
from lmcache.v1.multiprocess.kmesh_domains import DomainRegistry, KMeshFenceError
from lmcache.v1.multiprocess.kmesh_raw_server import create_managed_raw_server
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.modules.kmesh_managed_raw import KMeshManagedRawModule
from lmcache.v1.multiprocess.request_handler import iter_request_handlers
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.grpc_impl.server import GrpcMultiprocessServer
from lmcache.v1.multiprocess.transport.zmq_impl.mq import MessageQueueServer
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.devices.cuda.cache_context import retained_managed_raw_contexts
from lmcache.v1.platform.devices.cuda.ipc_wrapper import RawCudaIPCWrapper
from lmcache.v1.platform.devices.cuda.utils import _cuda
from tests.v1.test_kmesh_control import value
from tests.v1.test_kmesh_rotary_snapshot_gpu import read_snapshot

pytestmark = pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires real CUDA managed registration and physical release",
)


def managed_service(
    channel: Connection, transport: Literal["zmq", "grpc"], mode: str
) -> None:
    observed: dict[str, Any] = {"imports": 0, "native_closes": 0}
    fault = {"close": False, "copy": mode == "copy-undrained"}
    private_caches: list[torch.Tensor] = []
    raw_import = RawCudaIPCWrapper.to_tensor
    raw_close = RawCudaIPCWrapper.close_checked
    native_close = _cuda.runtime.cudaIpcCloseMemHandle
    raw_copy = RetainedTensorCopies.copy_raw_to_cpu
    publish = BlendModule.publish_rope_snapshot
    tensor_copy = torch.Tensor.copy_
    synchronize = copy_module.synchronize_stream

    def inspect_import(owner: RawCudaIPCWrapper) -> torch.Tensor:
        observed["imports"] += 1
        if mode == "partial-construction" and observed["imports"] == 2:
            raise RuntimeError("injected before second context import")
        return raw_import(owner)

    def inspect_native_close(pointer: int) -> object:
        observed["native_closes"] += 1
        if mode == "unknown-close" and fault["close"]:
            return (_cuda.runtime.cudaError_t.cudaErrorInvalidValue,)
        return native_close(pointer)

    def controlled_close(owner: RawCudaIPCWrapper) -> None:
        if mode == "close-refusal" and fault["close"]:
            raise RuntimeError("injected before managed checked close")
        raw_close(owner)

    @wraps(raw_copy)
    def inspect_copy(
        manager: RetainedTensorCopies,
        paged: Any,
        rotary: Any,
        producer_event: bytes,
        validate_metadata: Any,
    ) -> list[torch.Tensor]:
        backend = paged.entry.event_backend
        assert backend is not None
        event = backend.import_event(producer_event, paged.entry.cache_context.device)
        observed["producer_pending"] = not backend.query_event(event)
        result = raw_copy(manager, paged, rotary, producer_event, validate_metadata)
        observed["host_bytes"] = result[0].view(torch.uint8).numpy().tobytes()
        return result

    @wraps(publish)
    def inspect_publication(
        module: BlendModule, instance_id: int, context: Any, rotary: Any
    ) -> None:
        private_caches[:] = rotary.state.cos_sin_caches
        observed["private_bytes"] = (
            rotary.state.cos_sin_caches[0].cpu().view(torch.uint8).numpy().tobytes()
        )
        publish(module, instance_id, context, rotary)

    def fail_after_copy(
        target: torch.Tensor, source: torch.Tensor, non_blocking: bool = False
    ) -> torch.Tensor:
        result = tensor_copy(target, source, non_blocking=non_blocking)
        if (
            fault["copy"]
            and target.device.type == "cpu"
            and source.device.type == "cuda"
        ):
            raise RuntimeError("injected after managed raw snapshot submission")
        return result

    def controlled_drain(device: object, stream: object) -> None:
        if fault["copy"]:
            raise RuntimeError("injected managed snapshot drain refusal")
        synchronize(device, stream)

    server = engine = None
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(RawCudaIPCWrapper, "to_tensor", inspect_import)
            patch.setattr(RawCudaIPCWrapper, "close_checked", controlled_close)
            patch.setattr(_cuda.runtime, "cudaIpcCloseMemHandle", inspect_native_close)
            patch.setattr(RetainedTensorCopies, "copy_raw_to_cpu", inspect_copy)
            patch.setattr(BlendModule, "publish_rope_snapshot", inspect_publication)
            patch.setattr(torch.Tensor, "copy_", fail_after_copy)
            patch.setattr(copy_module, "synchronize_stream", controlled_drain)
            server, engine = create_managed_raw_server(
                port=0, transport=transport, chunk_tokens=8, dram_bytes=8 << 20
            )
            server.start()
            if isinstance(server, GrpcMultiprocessServer):
                url = f"grpc://127.0.0.1:{server.bound_port}"
            else:
                assert isinstance(server, MessageQueueServer)
                url = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
            channel.send(url)
            while True:
                assert channel.poll(90), "exporter stopped controlling the service"
                command = channel.recv()
                if command == "stop":
                    break
                if command == "arm-close":
                    fault["close"] = True
                elif command == "allow-cleanup":
                    fault["close"] = fault["copy"] = False
                else:
                    assert command == "observe"
                if private_caches:
                    observed["private_bytes"] = (
                        private_caches[0].cpu().view(torch.uint8).numpy().tobytes()
                    )
                channel.send(
                    dict(observed) | {"retained": retained_managed_raw_contexts()}
                )
            server.close()
            server = None
            try:
                engine.close()
            except KMeshFenceError:
                if mode not in ("partial-construction", "unknown-close"):
                    raise
                assert retained_managed_raw_contexts() == 1
                channel.send(("retained-until-importer-exit",))
            else:
                assert retained_managed_raw_contexts() == 0
                channel.send(("released",))
            engine = None
    except BaseException:
        channel.send(("error", traceback.format_exc()))
        raise
    finally:
        if server is not None:
            server.close()
        channel.close()


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "close-refusal",
        "copy-undrained",
        "partial-construction",
        "unknown-close",
    ],
)
def test_managed_raw_service_owns_registration_and_release(
    transport: Literal["zmq", "grpc"], mode: str
) -> None:
    device = torch.device("cuda", 0)
    pool = torch.empty(64 * (1 << 20) // 2, dtype=torch.bfloat16, device=device)
    kv = pool[:2048].reshape(2, 16, 4, 2, 8)
    other_view = pool[4096:6144].reshape(2, 16, 4, 2, 8)
    kv.zero_()
    other_view.fill_(9)
    rotary_pool = torch.empty(64 * (1 << 20) // 4, dtype=torch.float32, device=device)
    rope = rotary_pool[:512].reshape(64, 8)
    rope.fill_(float("nan"))
    torch_dev.synchronize(device)
    wrappers = [RawCudaIPCWrapper.wrap(x) for x in (kv, other_view, rope)]
    assert wrappers[0].view_identity != wrappers[1].view_identity
    assert wrappers[0].allocation_identity == wrappers[1].allocation_identity
    assert wrappers[0].allocation_identity != wrappers[2].allocation_identity
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(
        target=managed_service, args=(child, transport, mode)
    )
    process.start()
    child.close()
    client = foreign = None
    try:
        url = read_snapshot(parent)
        client = RequestClientFactory.create(url)
        foreign = RequestClientFactory.create(url)
        server = value(client.kmesh_get_server().result(timeout=30))
        inventory = [h.operation for h in iter_request_handlers(KMeshManagedRawModule)]
        assert all(name.startswith("kmesh_") for name in inventory)
        assert (
            server.capabilities_digest
            == DomainRegistry(inventory).server.capabilities_digest
        )
        nonce = uuid.uuid4().hex
        inputs: list[DeviceIPCWrapper] = list(
            wrappers[:2] if mode == "partial-construction" else wrappers[:1]
        )
        args: tuple[str, str, int, list[DeviceIPCWrapper], str, str] = (
            server.boot_id,
            nonce,
            99,
            inputs,
            "synthetic-managed",
            "NHD",
        )
        stale = client.kmesh_register_raw_context("old-boot", *args[1:]).result(
            timeout=30
        )
        assert stale[0] is None and stale[1]
        first = client.kmesh_register_raw_context(*args).result(timeout=30)
        registration = value(
            client.kmesh_raw_context_status(server.boot_id, nonce).result(timeout=30)
        )
        assert registration.boot_id == server.boot_id and registration.nonce == nonce
        context = registration.context
        if mode == "partial-construction":
            assert first[0] is None and first[1]
            assert registration.state == "poisoned"
            assert (
                value(client.kmesh_register_raw_context(*args).result(timeout=30))
                == registration
            )
            refused = client.kmesh_retire_raw_context(
                server.boot_id, nonce, context
            ).result(timeout=30)
            assert refused[0] is None and "importer teardown" in refused[1]
        else:
            assert value(first) == registration and registration.state == "registered"
            assert (
                value(client.kmesh_register_raw_context(*args).result(timeout=30))
                == registration
            )
            changed = client.kmesh_register_raw_context(
                server.boot_id, nonce, 99, [wrappers[1]], "synthetic-managed", "NHD"
            ).result(timeout=30)
            assert changed[0] is None and "metadata changed" in changed[1]
            wrong = foreign.kmesh_raw_context_status(server.boot_id, nonce).result(
                timeout=30
            )
            assert wrong[0] is None and "another client" in wrong[1]
            stale_retirement = client.kmesh_retire_raw_context(
                server.boot_id, nonce, replace(context, generation="stale")
            ).result(timeout=30)
            assert stale_retirement[0] is None and stale_retirement[1]
            parent.send("observe")
            assert read_snapshot(parent)["imports"] == 1
            phases = (
                torch.arange(64, dtype=torch.float32)[:, None]
                * torch.arange(1, 5)
                * 0.03
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
            producer = backend.export_event(ready, device)
            rotary_args: tuple[
                int,
                ContextIdentity,
                list[DeviceIPCWrapper],
                int,
                bool,
                list[int],
                list[list[int]],
                list[int],
                str,
                bytes,
            ] = (
                99,
                context,
                list(wrappers[2:]),
                8,
                True,
                [0],
                [],
                [],
                "",
                producer,
            )
            result = client.kmesh_register_rotary(*rotary_args).result(timeout=30)
            parent.send("observe")
            observed = read_snapshot(parent)
            assert observed["producer_pending"] and observed["imports"] == 2
            if mode == "copy-undrained":
                assert result[0] is None and result[1]
                blocked = client.kmesh_retire_raw_context(
                    server.boot_id, nonce, context
                ).result(timeout=30)
                assert blocked[0] is None and "drain" in blocked[1]
                assert observed["native_closes"] == 0
                parent.send("allow-cleanup")
                read_snapshot(parent)
                assert (
                    value(client.kmesh_retry_rotary_drain(context).result(timeout=30))
                    == 0
                )
                runtime = None
            else:
                runtime = value(result)
                assert (
                    observed["host_bytes"]
                    == expected.view(torch.uint8).numpy().tobytes()
                )
                assert (
                    observed["private_bytes"]
                    == expected.view(torch.uint8).numpy().tobytes()
                )
                assert (
                    value(client.kmesh_register_rotary(*rotary_args).result(timeout=30))
                    == runtime
                )
                rope.fill_(23)
                torch_dev.synchronize(device)
                assert value(client.kmesh_get_runtime(99).result(timeout=30)) == runtime
                parent.send("observe")
                after_mutation = read_snapshot(parent)
                assert (
                    after_mutation["private_bytes"]
                    == expected.view(torch.uint8).numpy().tobytes()
                )
            if mode in ("close-refusal", "unknown-close"):
                parent.send("arm-close")
                before = read_snapshot(parent)
                failed = client.kmesh_retire_raw_context(
                    server.boot_id, nonce, context
                ).result(timeout=30)
                assert failed[0] is None and failed[1]
                parent.send("observe")
                after = read_snapshot(parent)
                assert after["retained"] == 1
                assert after["native_closes"] == before["native_closes"] + (
                    mode == "unknown-close"
                )
                parent.send("allow-cleanup")
                read_snapshot(parent)
            reply = client.kmesh_retire_raw_context(
                server.boot_id, nonce, context
            ).result(timeout=30)
            if mode == "unknown-close":
                assert reply[0] is None and "importer teardown" in reply[1]
                parent.send("observe")
                assert read_snapshot(parent)["native_closes"] == after["native_closes"]
            else:
                release = value(reply)
                assert release.boot_id == server.boot_id and release.nonce == nonce
                assert release.context == context and release.instance_id == 99
                assert release.rotary == (runtime.rotary if runtime else None)
                assert release.allocation_ids == tuple(
                    sorted({w.allocation_identity for w in wrappers})
                )
                # Validate release before actually freeing exported allocations.
                # The importer remains alive for retries and independent readback.
                del kv, other_view, rope, pool, rotary_pool
                torch_dev.empty_cache()
                replacement_kv = torch.full(
                    (64 * (1 << 20) // 2,), 11, dtype=torch.bfloat16, device=device
                )
                replacement_rope = torch.full(
                    (64 * (1 << 20) // 4,), 13, dtype=torch.float32, device=device
                )
                assert torch.all(replacement_kv == 11).item()
                assert torch.all(replacement_rope == 13).item()
                assert (
                    value(
                        client.kmesh_retire_raw_context(
                            server.boot_id, nonce, context
                        ).result(timeout=30)
                    )
                    == release
                )
                assert (
                    value(
                        client.kmesh_raw_context_status(server.boot_id, nonce).result(
                            timeout=30
                        )
                    ).state
                    == "released"
                )
                forbidden = client.kmesh_register_raw_context(
                    server.boot_id,
                    uuid.uuid4().hex,
                    99,
                    inputs,
                    "synthetic-managed",
                    "NHD",
                ).result(timeout=30)
                assert forbidden[0] is None and forbidden[1]
                parent.send("observe")
                after_free = read_snapshot(parent)
                assert after_free["imports"] == 2
                assert after_free["native_closes"] == 2
                if runtime is not None:
                    assert (
                        after_free["private_bytes"]
                        == expected.view(torch.uint8).numpy().tobytes()
                    )
        parent.send("stop")
        outcome = read_snapshot(parent)
        assert outcome == (
            ("retained-until-importer-exit",)
            if mode in ("partial-construction", "unknown-close")
            else ("released",)
        )
        process.join(timeout=30)
        assert process.exitcode == 0
    finally:
        if client is not None:
            client.close()
        if foreign is not None:
            foreign.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=30)
        parent.close()
        # The exporter keeps allocations and producer events until importer exit.
        torch_dev.synchronize(device)
