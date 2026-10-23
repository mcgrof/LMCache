# SPDX-License-Identifier: Apache-2.0
"""Actual CPU cache registrations and transport-bound context lifetimes."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
from lmcache.v1.gpu_connector.kv_format.types import LayoutHints
from lmcache.v1.multiprocess.context_ownership import (
    ContextIdentity,
    ContextOwnershipError,
)
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    BorrowedContext,
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.posix_shm import shm_unlink
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.platform.devices.cpu.shm import migrate_to_shm_and_wrap
from tests.v1.test_request_session_transport import SessionProbe, loopback_server


class RegistrationProbe(SessionProbe):
    """Dispatch real registration and borrowing without creating device I/O."""

    def __init__(self, transfer: LMCacheDrivenTransferModule) -> None:
        super().__init__()
        self.transfer = transfer
        self.expected: ContextIdentity | None = None
        self.holds: list[BorrowedContext] = []
        self.revoked = threading.Event()

    @request_handler()
    def register_kv_cache(
        self,
        instance_id: int,
        kv_cache: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        self.transfer.register_kv_cache(
            instance_id,
            kv_cache,
            model_name,
            world_size,
            engine_type,
            layout_hints,
            engine_group_infos,
        )

    @request_handler()
    def unregister_kv_cache(self, instance_id: int) -> None:
        self.transfer.unregister_kv_cache(instance_id)

    @request_handler(HandlerType.BLOCKING)
    def get_chunk_size(self) -> int:
        if self.expected is None:
            return 0
        try:
            hold = self.transfer.borrow_context(99, self.expected, self.revoked.set)
        except ContextOwnershipError:
            return 0
        self.holds.append(hold)
        return 1


@contextmanager
def registered_server(transport: str) -> Iterator[tuple[str, RegistrationProbe]]:
    """Use a real small DRAM store, shared CPU tensors and request transport."""
    # Eight MiB holds no model weights; registration does not perform storage IO.
    context = MPCacheServerContext(
        StorageManagerConfig(
            DRAML1ManagerConfig(L1MemoryManagerConfig(8 << 20, False, shm_name="")),
            EvictionConfig("LRU"),
        ),
        chunk_size=8,
        hash_algorithm="sha256_cbor",
    )
    transfer = LMCacheDrivenTransferModule(context)
    probe = RegistrationProbe(transfer)
    try:
        with loopback_server(transport, probe) as url:
            yield url, probe
    finally:
        for hold in probe.holds:
            hold.release()
        transfer.close()
        context.close()


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_actual_registration_preserves_owner_and_drains_before_replacement(
    transport: str,
) -> None:
    tensors = [
        torch.full((2, 16, 4, 2, 8), float(layer + 1), dtype=torch.bfloat16)
        for layer in range(2)
    ]
    wrappers = [migrate_to_shm_and_wrap(tensor) for tensor in tensors]
    replacement_tensors = [tensor + 10 for tensor in tensors]
    replacement_wrappers = [
        migrate_to_shm_and_wrap(tensor) for tensor in replacement_tensors
    ]
    try:
        with registered_server(transport) as (url, probe):
            clients = [
                RequestClientFactory.create(url),
                RequestClientFactory.create(url),
            ]
            try:
                first, second = clients
                first.register_kv_cache(
                    99,
                    list(wrappers),
                    "synthetic-model",
                    1,
                    EngineType.VLLM,
                    {"kv_layout": "NHD"},
                    [],
                ).result(timeout=10)
                identity = probe.transfer.context_identity(99)
                assert identity is not None
                assert identity.owner_session == first.noop().result(timeout=10)
                probe.expected = identity
                assert second.get_chunk_size().result(timeout=10) == 0
                # Refreshing an existing instance cannot transfer its ownership.
                second.register_kv_cache(
                    99,
                    list(replacement_wrappers),
                    "other-model",
                    1,
                    EngineType.VLLM,
                    {"kv_layout": "NHD"},
                    [],
                ).result(timeout=10)
                assert probe.transfer.context_identity(99) == identity
                assert first.get_chunk_size().result(timeout=10) == 1
                borrowed = probe.holds[-1]
                unregistered = first.unregister_kv_cache(99)
                try:
                    assert probe.revoked.wait(timeout=5)
                    assert not unregistered.query()
                    assert borrowed.entry.ownership.pending == 1
                    assert torch.equal(
                        borrowed.entry.cache_context.kv_tensors[0], tensors[0]
                    )
                finally:
                    borrowed.release()
                unregistered.result(timeout=10)
                assert probe.transfer.context_identity(99) is None
                second.register_kv_cache(
                    99,
                    list(replacement_wrappers),
                    "synthetic-model",
                    1,
                    EngineType.VLLM,
                    {"kv_layout": "NHD"},
                    [],
                ).result(timeout=10)
                replacement = probe.transfer.context_identity(99)
                assert replacement is not None and replacement != identity
                assert replacement.owner_session == second.noop().result(timeout=10)
                assert second.get_chunk_size().result(timeout=10) == 0
                probe.expected = replacement
                assert first.get_chunk_size().result(timeout=10) == 0
                assert second.get_chunk_size().result(timeout=10) == 1
                replacement_context = probe.holds[-1]
                try:
                    assert torch.equal(
                        replacement_context.entry.cache_context.kv_tensors[0],
                        replacement_tensors[0],
                    )
                    replacement_tensors[0].fill_(41)
                    assert torch.equal(
                        replacement_context.entry.cache_context.kv_tensors[0],
                        replacement_tensors[0],
                    )
                    assert torch.equal(
                        borrowed.entry.cache_context.kv_tensors[0], tensors[0]
                    )
                finally:
                    replacement_context.release()
            finally:
                for client in clients:
                    client.close()
    finally:
        for wrapper in wrappers + replacement_wrappers:
            shm_unlink(wrapper.shm_name)


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_reaper_waits_for_the_actual_borrowed_context(transport: str) -> None:
    tensor = torch.zeros((2, 16, 4, 2, 8), dtype=torch.bfloat16)
    wrapper = migrate_to_shm_and_wrap(tensor)
    try:
        with registered_server(transport) as (url, probe):
            client = RequestClientFactory.create(url)
            try:
                client.register_kv_cache(
                    99,
                    [wrapper],
                    "synthetic-model",
                    1,
                    EngineType.VLLM,
                    {"kv_layout": "NHD"},
                    [],
                ).result(timeout=10)
                probe.expected = probe.transfer.context_identity(99)
                assert client.get_chunk_size().result(timeout=10) == 1
                with ThreadPoolExecutor(max_workers=1) as pool:
                    reaping = pool.submit(probe.transfer.reap_stale_instances, 0, 0)
                    try:
                        assert probe.revoked.wait(timeout=5)
                        assert not reaping.done()
                        assert probe.transfer.context_identity(99) is None
                    finally:
                        probe.holds[-1].release()
                    assert reaping.result(timeout=10) == [99]
            finally:
                client.close()
    finally:
        shm_unlink(wrapper.shm_name)
