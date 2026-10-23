# SPDX-License-Identifier: Apache-2.0
"""Qualify typed CPU controls with real servers and simulated completion debt."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from multiprocessing.connection import Connection
from typing import Literal, TypeVar
import multiprocessing
import threading

# Third Party
import grpc
import pytest
import torch
import zmq

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import (
    DRAML1ManagerConfig,
    EvictionConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import ObservabilityConfig
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.kmesh_domains import (
    Completion,
    KMeshFenceError,
    RegistrationLease,
    RegistrationState,
)
from lmcache.v1.multiprocess.modules.kmesh_control import KMeshControlModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.posix_shm import shm_unlink
from lmcache.v1.multiprocess.server import run_cache_server
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    kmesh_service_pb2,
    kmesh_service_pb2_grpc,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server import GrpcMultiprocessServer
from lmcache.v1.multiprocess.transport.request_context import request_session
from lmcache.v1.multiprocess.transport.zmq_impl.mq import MessageQueueServer
from lmcache.v1.platform.devices.cpu.shm import migrate_to_shm_and_wrap
from tests.v1.test_kmesh_bindings import semantics
from tests.v1.test_registered_rope_ownership import rotary_server

T = TypeVar("T")


def value(reply: tuple[T | None, str]) -> T:
    result, error = reply
    assert error == "" and result is not None, error
    return result


def serve(transport: Literal["zmq", "grpc"], connection: Connection) -> None:
    """Start the public server entry point in a CPU-only child process."""
    result = run_cache_server(
        MPServerConfig(
            transport=transport,
            host="127.0.0.1",
            port=0,
            chunk_size=8,
            engine_type="blend",
            enable_kmesh_control=True,
            worker_reap_timeout_seconds=0,
        ),
        StorageManagerConfig(
            DRAML1ManagerConfig(L1MemoryManagerConfig(8 << 20, False, shm_name="")),
            EvictionConfig("LRU"),
        ),
        ObservabilityConfig(enabled=False),
        return_engine=True,
        start_prometheus_http_server=False,
    )
    assert result is not None
    server, engine = result
    try:
        if transport == "grpc":
            assert isinstance(server, GrpcMultiprocessServer)
            url = f"grpc://127.0.0.1:{server.bound_port}"
        else:
            assert isinstance(server, MessageQueueServer)
            url = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        connection.send(url)
        if not connection.poll(60):
            raise RuntimeError("parent did not stop the control server")
        assert connection.recv() == "stop"
    finally:
        server.close()
        engine.close()
        connection.close()


@contextmanager
def remote_server(transport: Literal["zmq", "grpc"]) -> Iterator[str]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=serve, args=(transport, child))
    process.start()
    child.close()
    try:
        assert parent.poll(20), "server did not start"
        url = parent.recv()
        assert isinstance(url, str)
        yield url
        parent.send("stop")
        process.join(timeout=20)
        assert process.exitcode == 0
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)


@contextmanager
def worker_tensors(client: RequestClient, fill: float = 1.0) -> Iterator[None]:
    """Retain producer mappings through actual server unregistration."""
    kv = torch.full((2, 16, 4, 2, 8), fill, dtype=torch.bfloat16)
    rope = torch.zeros((64, 8), dtype=torch.float32)
    rope[:, :4] = 1
    wrappers = [migrate_to_shm_and_wrap(tensor) for tensor in (kv, rope)]
    try:
        client.register_kv_cache(
            99,
            [wrappers[0]],
            "synthetic-model",
            1,
            EngineType.VLLM,
            {"kv_layout": "NHD"},
            [],
        ).result(timeout=10)
        client.cb_register_rope(99, [wrappers[1]], 8, True, [0], [], []).result(
            timeout=10
        )
        yield
    finally:
        try:
            client.cb_unregister_rope(99).result(timeout=10)
            client.unregister_kv_cache(99).result(timeout=10)
        finally:
            for wrapper in wrappers:
                shm_unlink(wrapper.shm_name)


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_real_control_server_binds_only_current_client_registrations(
    transport: Literal["zmq", "grpc"],
) -> None:
    with remote_server(transport) as url:
        first, second = [RequestClientFactory.create(url) for _ in range(2)]
        try:
            server = value(first.kmesh_get_server().result(timeout=10))
            with worker_tensors(first):
                runtime = value(first.kmesh_get_runtime(99).result(timeout=10))
                refused_runtime, error = second.kmesh_get_runtime(99).result(timeout=10)
                assert refused_runtime is None and "borrowable" in error
                identity = semantics(runtime)
                lease = value(
                    second.kmesh_open_registration(
                        identity, "engine", 1, "5" * 64
                    ).result(timeout=10)
                )
                assert lease.server == server
                fence = value(
                    first.kmesh_bind_worker(
                        lease, 0, 99, runtime, identity, "5" * 64
                    ).result(timeout=10)
                )
                status = value(
                    second.kmesh_registration_state(lease).result(timeout=10)
                )
                assert status.state == RegistrationState.READY and status.pending == 0
                refused_heartbeat, error = second.kmesh_heartbeat(fence).result(
                    timeout=10
                )
                assert refused_heartbeat is None and "not bound" in error
                assert value(first.kmesh_heartbeat(fence).result(timeout=10)) is True
                assert (
                    value(second.kmesh_revoke_registration(lease).result(timeout=10))
                    == 0
                )
            with worker_tensors(second, fill=7.0):
                replacement = value(second.kmesh_get_runtime(99).result(timeout=10))
                assert replacement.context != runtime.context
                assert replacement.layout_digest == runtime.layout_digest
                assert replacement.rope_digest == runtime.rope_digest
                identity = semantics(replacement)
                next_lease = value(
                    second.kmesh_open_registration(
                        identity, "replacement-engine", 1, "5" * 64
                    ).result(timeout=10)
                )
                refused_worker, error = second.kmesh_bind_worker(
                    next_lease, 0, 99, runtime, identity, "5" * 64
                ).result(timeout=10)
                assert refused_worker is None and error
                next_fence = value(
                    second.kmesh_bind_worker(
                        next_lease, 0, 99, replacement, identity, "5" * 64
                    ).result(timeout=10)
                )
                assert next_fence.session_id != fence.session_id
                assert next_lease.domain.namespace == lease.domain.namespace
                assert (
                    value(second.kmesh_heartbeat(next_fence).result(timeout=10)) is True
                )
                refused_heartbeat, error = first.kmesh_heartbeat(fence).result(
                    timeout=10
                )
                assert refused_heartbeat is None and error
                assert (
                    value(
                        second.kmesh_revoke_registration(next_lease).result(timeout=10)
                    )
                    == 0
                )
        finally:
            first.close()
            second.close()


def test_missing_grpc_identity_returns_refusal_without_invented_descriptor() -> None:
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        control = KMeshControlModule(transfer.context, transfer, blend)
        server = GrpcMultiprocessServer(
            "grpc://127.0.0.1:0",
            max_cpu_workers=2,
            max_gpu_workers=1,
            grpc_server_workers=4,
        )
        server.add_modules([control])
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{server.bound_port}") as channel:
                stub = kmesh_service_pb2_grpc.KmeshServiceStub(channel)
                for metadata in (
                    (),
                    (("lmcache-client-id-bin", b""),),
                    (("lmcache-client-id-bin", b"client"),) * 2,
                ):
                    reply = stub.KmeshGetServer(
                        kmesh_service_pb2.KmeshGetServerRequest(),
                        timeout=10,
                        metadata=metadata,
                    )
                    assert not reply.HasField("value") and "bound" in reply.error
        finally:
            server.close()
            control.close()


def test_expiry_and_close_cannot_rearm_or_discard_accepted_work() -> None:
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        now = [0.0]
        control = KMeshControlModule(
            transfer.context, transfer, blend, lease_seconds=2.0, clock=lambda: now[0]
        )
        manager = ManagementModule(transfer.context, store_clear_guard=control)
        with request_session("zmq", b"worker"):
            runtime = value(control.kmesh_get_runtime(99))
            identity = semantics(runtime)
            lease = value(
                control.kmesh_open_registration(identity, "engine", 1, "5" * 64)
            )
            fence = value(
                control.kmesh_bind_worker(lease, 0, 99, runtime, identity, "5" * 64)
            )
            operation = control.bindings.begin(fence, "request", 1)
            now[0] = 1.0
            assert value(control.kmesh_heartbeat(fence)) is True
            now[0] = 2.5
            assert (
                value(control.kmesh_registration_state(lease)).state
                == RegistrationState.READY
            )
        try:
            now[0] = 4.0
            control.reap_stale_instances(0, 0)
            assert control.bindings.state(lease) == RegistrationState.REVOKED
            assert control.bindings.pending(lease) == 1
            with pytest.raises(KMeshFenceError, match="drain before clear"):
                manager.clear(force=True)
            with pytest.raises(KMeshFenceError, match="drain before shutdown"):
                control.close()
            with request_session("zmq", b"worker"):
                refused, error = control.kmesh_open_registration(
                    identity, "new-engine", 1, "5" * 64
                )
                assert refused is None and "closing" in error
        finally:
            control.bindings.finish(operation, Completion.CANCELLED_DRAINED)
            control.close()
            manager.close()


def test_registration_waits_until_physical_clear_callback_returns() -> None:
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        control = KMeshControlModule(transfer.context, transfer, blend)
        with request_session("zmq", b"worker"):
            runtime = value(control.kmesh_get_runtime(99))
        identity = semantics(runtime)
        lease = control.bindings.open_registration(identity, "engine", 1, "5" * 64)
        entered, release, opening = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def clear() -> None:
            entered.set()
            assert release.wait(timeout=5)

        def register() -> RegistrationLease:
            opening.set()
            return control.bindings.open_registration(
                identity, "new-engine", 1, "5" * 64
            )

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                clearing = pool.submit(control.clear_store, clear)
                try:
                    assert entered.wait(timeout=5)
                    registering = pool.submit(register)
                    assert opening.wait(timeout=5) and not registering.done()
                finally:
                    release.set()
                clearing.result(timeout=5)
                replacement = registering.result(timeout=5)
                assert replacement.domain.namespace != lease.domain.namespace
        finally:
            release.set()
            control.close()
        assert control.bindings.state(replacement) == RegistrationState.REVOKED
