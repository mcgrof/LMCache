# SPDX-License-Identifier: Apache-2.0
"""Exercise ordered device registration through separate public RPC servers.

Inspection hooks observe producer state and completed private GPU bytes. Errors
are injected around actual copies; no device fault or exporter death is claimed.
"""

# Standard
from functools import wraps
from multiprocessing.connection import Connection
from typing import Any, Literal
import multiprocessing as mp
import traceback

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.multiprocess import kmesh_device_copy as copy_module
from lmcache.v1.multiprocess.kmesh_bindings import RegisteredWorkerBindings
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper
from tests.v1.test_kmesh_control import serve, value

pytestmark = pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires CUDA IPC and physical device registration",
)


def read_snapshot(channel: Connection) -> Any:
    """Bound child replies and surface any preserved server exception."""
    assert channel.poll(60), "public snapshot server did not respond"
    result = channel.recv()
    assert not (isinstance(result, tuple) and result[0] == "error"), result
    return result


def snapshot_server(
    channel: Connection, transport: Literal["zmq", "grpc"], mode: str
) -> None:
    """Observe the actual public-server path without changing its readiness proof."""
    observed: dict[str, Any] = {}
    original = RegisteredWorkerBindings.register_rotary_snapshot
    publish = BlendModule.publish_rope_snapshot
    copy_tensor = torch.Tensor.copy_
    drain = copy_module.synchronize_stream
    fault = {"enabled": mode == "copy-undrained"}

    @wraps(original)
    def inspect_producer(
        self: RegisteredWorkerBindings, *args: Any, **kwargs: Any
    ) -> Any:
        backend = get_event_ipc_backend(torch.device("cuda", 0))
        imported = backend.import_event(args[-1], torch.device("cuda", 0))
        observed["pending"] = not backend.query_event(imported)
        return original(self, *args, **kwargs)

    @wraps(publish)
    def inspect_publication(self: BlendModule, *args: Any, **kwargs: Any) -> None:
        publish(self, *args, **kwargs)
        observed["bytes"] = (
            args[-1].state.cos_sin_caches[0].cpu().view(torch.uint8).numpy().tobytes()
        )

    def copy_then_fail(
        target: torch.Tensor, source: torch.Tensor, non_blocking: bool = False
    ) -> torch.Tensor:
        result = copy_tensor(target, source, non_blocking=non_blocking)
        if fault["enabled"] and target.device.type != source.device.type:
            raise RuntimeError("injected after real snapshot copy submission")
        return result

    def controlled_drain(device: torch.device, stream: Any) -> None:
        if fault["enabled"]:
            raise RuntimeError("injected snapshot stream drain refusal")
        drain(device, stream)

    try:
        # The public entry point retains its module assembly and typed handlers.
        # The side channel only observes state and controls this injected fault.
        with pytest.MonkeyPatch.context() as patch:
            if mode != "disabled":
                patch.setattr(
                    RegisteredWorkerBindings,
                    "register_rotary_snapshot",
                    inspect_producer,
                )
            patch.setattr(BlendModule, "publish_rope_snapshot", inspect_publication)
            patch.setattr(torch.Tensor, "copy_", copy_then_fail)
            patch.setattr(copy_module, "synchronize_stream", controlled_drain)
            serve(
                transport,
                channel,
                device_snapshots=mode != "disabled",
                observer=lambda command: observe(command, observed, fault),
            )
    except BaseException:
        if not channel.closed:
            channel.send(("error", traceback.format_exc()))
        raise
    finally:
        channel.close()


def observe(command: str, observed: dict[str, Any], fault: dict[str, bool]) -> object:
    """Test-side observation, never a production completion assertion."""
    if command == "allow-drain":
        fault["enabled"] = False
    return dict(observed)


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
@pytest.mark.parametrize("mode", ["success", "copy-undrained", "disabled"])
def test_ordered_device_snapshot_uses_real_public_server(
    transport: Literal["zmq", "grpc"], mode: str
) -> None:
    device = torch.device("cuda", 0)
    kv = torch.zeros((2, 16, 4, 2, 8), dtype=torch.bfloat16, device=device)
    rope = torch.full((64, 8), float("nan"), dtype=torch.float32, device=device)
    torch_dev.synchronize(device)
    kv_wrapper = CudaIPCWrapper.wrap(kv)
    # A refused request must not create an outgoing Torch IPC reference that
    # no receiver consumes. Reuse metadata already consumed by KV registration;
    # disabled admission must refuse before importing or inspecting its shape.
    rope_wrapper = kv_wrapper if mode == "disabled" else CudaIPCWrapper.wrap(rope)
    wrappers = kv_wrapper, rope_wrapper
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(
        target=snapshot_server, args=(child, transport, mode)
    )
    process.start()
    child.close()
    client = None
    try:
        url = read_snapshot(parent)
        assert isinstance(url, str), url
        client = RequestClientFactory.create(url)
        client.register_kv_cache(
            99,
            [wrappers[0]],
            "synthetic-snapshot",
            1,
            EngineType.VLLM,
            {"kv_layout": "NHD"},
            [],
        ).result(timeout=30)
        context = value(client.kmesh_get_context(99).result(timeout=30))
        expected = torch.zeros((64, 8), dtype=torch.float32).pin_memory()
        expected[:, :4] = 1
        scaled = (expected * 2).pin_memory()
        stream = torch_dev.Stream(device=device)
        backend = get_event_ipc_backend(device)
        ready = backend.create_event(device)
        matrix = torch.full((2048, 2048), 1 / 2048, device=device)
        workspace = torch.empty_like(matrix)
        torch_dev.synchronize(device)
        with torch_dev.stream(stream):
            for _ in range(512):
                torch.mm(matrix, matrix, out=workspace)
            rope.copy_(scaled, non_blocking=True)
            backend.record_event(ready, stream)
        event = backend.export_event(ready, device)
        runtime, error = client.kmesh_register_rotary(
            99, context, [wrappers[1]], 8, True, [0], [], [], "", event
        ).result(timeout=30)
        parent.send("observe")
        observed = read_snapshot(parent)
        if mode == "disabled":
            assert runtime is None and "explicit opt-in" in error
            assert "bytes" not in observed
        elif mode == "copy-undrained":
            assert runtime is None and "copy failed" in error
            assert observed["pending"] and "bytes" not in observed
            parent.send("allow-drain")
            read_snapshot(parent)
            assert (
                value(client.kmesh_retry_rotary_drain(context).result(timeout=30)) == 0
            )
            runtime = value(
                client.kmesh_register_rotary(
                    99,
                    context,
                    [CudaIPCWrapper.wrap(rope)],
                    8,
                    True,
                    [0],
                    [],
                    [],
                    "",
                    event,
                ).result(timeout=30)
            )
        else:
            assert runtime is not None and error == ""
            assert observed["pending"], "producer challenge already drained"
        if runtime is not None:
            parent.send("observe")
            observed = read_snapshot(parent)
            assert observed["bytes"] == expected.view(torch.uint8).numpy().tobytes()
            assert value(client.kmesh_get_runtime(99).result(timeout=30)) == runtime
            # Private server state must survive a later change to the exporter.
            rope.zero_()
            torch_dev.synchronize(device)
            assert value(client.kmesh_get_runtime(99).result(timeout=30)) == runtime
            client.cb_unregister_rope(99).result(timeout=30)
        client.unregister_kv_cache(99).result(timeout=30)
        parent.send("stop")
        process.join(timeout=30)
        assert process.exitcode == 0
    finally:
        if client is not None:
            client.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=30)
        parent.close()
        # Retain the exporter until the server process has exited, including failure.
        torch_dev.synchronize(device)
