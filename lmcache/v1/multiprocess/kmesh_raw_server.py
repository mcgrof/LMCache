# SPDX-License-Identifier: Apache-2.0
"""Run a bounded managed raw service with optional native span transfers."""

# Standard
from typing import Literal
import argparse
import signal
import threading

# First Party
from lmcache import torch_device_type
from lmcache.v1.distributed.config import (
    DRAML1ManagerConfig,
    EvictionConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.gpu_connector.gds_context import require_inactive_gds
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.kmesh_managed_raw import KMeshManagedRawModule
from lmcache.v1.multiprocess.request_handler import iter_request_handlers
from lmcache.v1.multiprocess.server import MPCacheServer
from lmcache.v1.multiprocess.transport.base import RequestServer
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from lmcache.v1.platform.ipc_policy import is_isolated_ipc


def create_managed_raw_server(
    host: str = "127.0.0.1",
    port: int = 5555,
    transport: Literal["zmq", "grpc"] = "zmq",
    chunk_tokens: int = 256,
    dram_bytes: int = 64 << 20,
    enable_span_execution: bool = False,
) -> tuple[RequestServer, MPCacheServer]:
    """Construct the bounded raw service; the caller starts its transport.

    Args:
        host: Request bind address.
        port: Request port, or zero for a test-selected free port.
        transport: Same-host ZeroMQ or gRPC request transport.
        chunk_tokens: Positive attention chunk size used for registration.
        dram_bytes: Positive CPU L1 capacity; GPU-L1/GDS is not admitted.
        enable_span_execution: Opt into native CPU-L1 span transactions.

    Returns:
        Unstarted request transport and its resource-owning cache server.

    Raises:
        ValueError: Non-CUDA or isolated-timeline profile, invalid geometry.
    """
    if torch_device_type != "cuda" or is_isolated_ipc():
        raise ValueError("managed raw service requires CUDA and same-host event IPC")
    if chunk_tokens < 1 or dram_bytes < 1 or transport not in ("zmq", "grpc"):
        raise ValueError("invalid managed raw service configuration")
    require_inactive_gds()
    context = MPCacheServerContext(
        StorageManagerConfig(
            DRAML1ManagerConfig(
                L1MemoryManagerConfig(
                    dram_bytes,
                    enable_span_execution,
                    init_size_in_bytes=dram_bytes,
                    shm_name="",
                )
            ),
            EvictionConfig("LRU"),
        ),
        chunk_size=chunk_tokens,
        hash_algorithm="blake3",
    )
    module = KMeshManagedRawModule(context, enable_span_execution=enable_span_execution)
    module.set_handler_inventory([h.operation for h in iter_request_handlers(module)])
    config = MPServerConfig(
        host=host,
        port=port,
        transport=transport,
        engine_type="blend",
        enable_kmesh_control=True,
        enable_kmesh_device_snapshots=True,
        worker_reap_timeout_seconds=0,
    )
    engine = MPCacheServer(context, [module])
    try:
        return create_request_server([module], config), engine
    except BaseException:
        engine.close()
        raise


def main() -> None:
    """Serve the bounded raw service until a signal; preserve unreleased resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--transport", choices=("zmq", "grpc"), default="zmq")
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--dram-bytes", type=int, default=64 << 20)
    parser.add_argument("--enable-span-execution", action="store_true")
    args = parser.parse_args()
    server, engine = create_managed_raw_server(
        args.host,
        args.port,
        args.transport,
        args.chunk_tokens,
        args.dram_bytes,
        args.enable_span_execution,
    )
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    try:
        server.start()
        stop.wait()
    finally:
        server.close()
        engine.close()


if __name__ == "__main__":
    main()
