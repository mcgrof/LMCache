# SPDX-License-Identifier: Apache-2.0
"""Loopback request-session propagation through real transport executors."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.connection import Connection
import multiprocessing
import threading

# Third Party
import grpc
import pytest
import zmq

# First Party
from lmcache.v1.multiprocess.request_handler import (
    HandlerType,
    iter_request_handlers,
    request_handler,
)
from lmcache.v1.multiprocess.transport.base import RequestServer
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    debug_service_pb2,
    debug_service_pb2_grpc,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server import GrpcMultiprocessServer
from lmcache.v1.multiprocess.transport.request_context import get_request_session
from lmcache.v1.multiprocess.transport.zmq_impl.mq import MessageQueueServer


class SessionProbe:
    """Observe executor context without creating cache or device resources."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.observed: list[tuple[str, str, int]] = []

    def record(self, operation: str) -> str:
        """Record transport-provided context, including absent legacy metadata."""
        try:
            session = get_request_session()
        except RuntimeError:
            session = "missing"
        with self.lock:
            self.observed.append((operation, session, threading.get_ident()))
        return session

    @request_handler()
    def noop(self) -> str:
        return self.record("sync")

    @request_handler(HandlerType.BLOCKING)
    def get_chunk_size(self) -> int:
        self.record("normal")
        return 1

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def cb_protocol_handshake(self, client_version: int) -> tuple[int, bool]:
        self.record("affinity")
        return client_version, True


@contextmanager
def loopback_server(transport: str, probe: SessionProbe) -> Iterator[str]:
    """Start the actual request transport with normal and affinity executors."""
    server: RequestServer
    if transport == "grpc":
        grpc_server = GrpcMultiprocessServer(
            "grpc://127.0.0.1:0",
            max_cpu_workers=2,
            max_gpu_workers=1,
            grpc_server_workers=4,
        )
        grpc_server.add_modules([probe])
        server = grpc_server
        url = f"grpc://127.0.0.1:{grpc_server.bound_port}"
    else:
        zmq_server = MessageQueueServer("tcp://127.0.0.1:*", zmq.Context.instance())
        for registered in iter_request_handlers(probe):
            zmq_server.add_handler(
                registered.operation,
                registered.options.handler_type,
                registered.handler,
            )
        zmq_server.add_normal_thread_pool(["get_chunk_size"], max_workers=2)
        zmq_server.add_affinity_thread_pool(["cb_protocol_handshake"], max_workers=1)
        url = zmq_server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        server = zmq_server
    server.start()
    try:
        yield url
    finally:
        server.close()


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_real_transports_preserve_client_across_all_handler_pools(
    transport: str,
) -> None:
    probe = SessionProbe()
    with loopback_server(transport, probe) as url:
        clients = [RequestClientFactory.create(url), RequestClientFactory.create(url)]
        try:
            sessions = [client.noop().result(timeout=10) for client in clients]
            assert len(set(sessions)) == 2
            assert all(len(session) == 64 for session in sessions)
            for _ in range(8):
                normal = [client.get_chunk_size() for client in clients]
                affinity = [client.cb_protocol_handshake(1) for client in clients]
                assert [future.result(timeout=10) for future in normal] == [1, 1]
                assert [future.result(timeout=10) for future in affinity] == [
                    (1, True)
                ] * 2
            assert [client.noop().result(timeout=10) for client in clients] == sessions
        finally:
            for client in clients:
                client.close()
    for session in sessions:
        for operation, count in (("sync", 2), ("normal", 8), ("affinity", 8)):
            threads = [
                thread
                for op, seen, thread in probe.observed
                if op == operation and seen == session
            ]
            assert len(threads) == count
            assert all(thread != threading.get_ident() for thread in threads)
    with pytest.raises(RuntimeError, match="no bound"):
        get_request_session()


def test_grpc_missing_or_duplicate_metadata_never_invents_an_identity() -> None:
    probe = SessionProbe()
    with loopback_server("grpc", probe) as url:
        with grpc.insecure_channel(url.removeprefix("grpc://")) as channel:
            stub = debug_service_pb2_grpc.DebugServiceStub(channel)
            request = debug_service_pb2.NoopRequest()
            assert stub.Noop(request, timeout=10).message == "missing"
            metadata = (("lmcache-client-id-bin", b"one-client"),)
            first = stub.Noop(request, timeout=10, metadata=metadata).message
            assert len(first) == 64
            assert stub.Noop(request, timeout=10, metadata=metadata).message == first
            assert (
                stub.Noop(request, timeout=10, metadata=metadata * 2).message
                == "missing"
            )
            assert (
                stub.Noop(
                    request, timeout=10, metadata=(("lmcache-client-id-bin", b""),)
                ).message
                == "missing"
            )
            assert stub.Noop(request, timeout=10).message == "missing"


def _serve_child(transport: str, connection: Connection) -> None:
    """Keep a separate real server process alive until its parent stops it."""
    try:
        with loopback_server(transport, SessionProbe()) as url:
            connection.send(url)
            if not connection.poll(20):
                raise RuntimeError("parent did not stop the loopback server")
            if connection.recv() != "stop":
                raise RuntimeError("unexpected loopback server command")
    finally:
        connection.close()


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_separate_server_distinguishes_replacement_clients(
    transport: str,
) -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    server = context.Process(target=_serve_child, args=(transport, child))
    server.start()
    child.close()
    try:
        assert parent.poll(10), "server failed to provide its loopback endpoint"
        url = parent.recv()
        sessions: list[str] = []
        for _ in range(2):
            client = RequestClientFactory.create(url)
            try:
                session = client.noop().result(timeout=10)
                assert client.noop().result(timeout=10) == session
                sessions.append(session)
            finally:
                client.close()
        assert len(set(sessions)) == 2
        parent.send("stop")
        server.join(timeout=10)
        assert server.exitcode == 0
    finally:
        parent.close()
        if server.is_alive():
            server.terminate()
            server.join(timeout=10)
