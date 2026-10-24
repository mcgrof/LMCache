# SPDX-License-Identifier: Apache-2.0
"""CPU ownership controls for the bounded managed raw service integration."""

# Standard
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import (
    DRAML1ManagerConfig,
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.multiprocess import kmesh_raw_server
from lmcache.v1.multiprocess.context_ownership import (
    ContextOwnership,
    ContextOwnershipError,
)
from lmcache.v1.multiprocess.kmesh_bindings import RegisteredWorkerBindings
from lmcache.v1.multiprocess.kmesh_domains import DomainRegistry, KMeshFenceError
from lmcache.v1.multiprocess.modules.kmesh_managed_raw import KMeshManagedRawModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import ContextEntry
from lmcache.v1.multiprocess.posix_shm import shm_unlink
from lmcache.v1.multiprocess.request_handler import iter_request_handlers
from lmcache.v1.multiprocess.transport.request_context import request_session
from lmcache.v1.platform.cache_context import create_cache_context
from lmcache.v1.platform.devices.cpu.shm import migrate_to_shm_and_wrap
from tests.v1.test_registered_rope_ownership import rotary_server


def test_managed_server_factory_refuses_cpu_before_resource_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kmesh_raw_server, "torch_device_type", "cpu")
    with pytest.raises(ValueError, match="requires CUDA"):
        kmesh_raw_server.create_managed_raw_server()


@pytest.mark.parametrize("case", ["disabled", "legacy-tier", "peer-tier"])
def test_managed_storage_refuses_unbounded_profiles_before_allocation(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    monkeypatch.setattr(kmesh_raw_server, "torch_device_type", "cuda")
    monkeypatch.setattr(kmesh_raw_server, "is_isolated_ipc", lambda: False)
    memory = L1MemoryManagerConfig(64 << 20, True, shm_name="")
    l1 = (
        L1ManagerConfig(memory)
        if case == "legacy-tier"
        else DRAML1ManagerConfig(memory)
    )
    peers = [l1, DRAML1ManagerConfig(memory, tag="peer")] if case == "peer-tier" else []
    storage = StorageManagerConfig(l1, EvictionConfig("LRU"), l1_manager_configs=peers)
    with pytest.raises(ValueError, match="one host or GPU L1"):
        kmesh_raw_server.create_managed_raw_server(
            enable_span_execution=case != "disabled", storage_config=storage
        )


def test_managed_control_inventory_excludes_legacy_context_consumers() -> None:
    operations = {h.operation for h in iter_request_handlers(KMeshManagedRawModule)}
    assert {
        "kmesh_register_raw_context",
        "kmesh_raw_context_status",
        "kmesh_retire_raw_context",
        "kmesh_register_rotary",
    } <= operations
    assert all(name.startswith("kmesh_") for name in operations)
    assert (
        not {
            "register_kv_cache",
            "unregister_kv_cache",
            "store",
            "retrieve",
            "lookup",
            "cb_register_rope",
            "cb_retrieve",
            "ping",
        }
        & operations
    )


def test_adopted_cpu_entry_preserves_exact_owner_and_physical_hold() -> None:
    tensor = torch.zeros((2, 16, 4, 2, 8), dtype=torch.bfloat16)
    wrapper = migrate_to_shm_and_wrap(tensor)
    cache = create_cache_context(
        [wrapper], 8, {"kv_layout": "NHD"}, engine_type=EngineType.VLLM
    )
    try:
        with rotary_server() as (_, transfer):
            with request_session("zmq", b"worker"):
                entry = ContextEntry(
                    cache, "synthetic-adoption", 1, ownership=ContextOwnership()
                )
            with request_session("zmq", b"foreign"):
                with pytest.raises(ContextOwnershipError, match="another client"):
                    transfer.adopt_context_entry(101, entry)
            with request_session("zmq", b"worker"):
                transfer.adopt_context_entry(101, entry)
                expected = entry.ownership.identity
                assert transfer.context_identity(101) == expected
                assert (
                    transfer.context.layout_desc_registry.find(entry.model_name, 1)
                    is not None
                )
                with pytest.raises(ContextOwnershipError, match="already registered"):
                    transfer.adopt_context_entry(101, entry)
                revoked = threading.Event()
                hold = transfer.borrow_context(101, expected, revoked.set)
                assert hold.entry is entry
                with pytest.raises(ContextOwnershipError, match="undrained"):
                    transfer.detach_context_entry(101, entry)
                assert revoked.is_set()
                assert transfer.context_identity(101) == expected
                hold.release()
                cache.close()
                transfer.detach_context_entry(101, entry)
                assert transfer.context_identity(101) is None
                assert (
                    transfer.context.layout_desc_registry.find(entry.model_name, 1)
                    is None
                )
    finally:
        cache.close()
        shm_unlink(wrapper.shm_name)


def test_raw_snapshot_mode_refuses_cpu_context_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.zeros((64, 8), dtype=torch.float32)
    wrapper = migrate_to_shm_and_wrap(tensor)

    def forbidden_import() -> torch.Tensor:
        raise AssertionError("raw snapshot attempted CPU import")

    monkeypatch.setattr(wrapper, "to_tensor", forbidden_import)
    try:
        with rotary_server(hash_algorithm="blake3") as (blend, transfer):
            binding = RegisteredWorkerBindings(
                DomainRegistry([]),
                transfer,
                blend,
                allow_device_snapshots=True,
                raw_snapshot_inputs=True,
            )
            with request_session("zmq", b"worker"):
                context = transfer.context_identity(99)
                assert context is not None
                with pytest.raises(KMeshFenceError, match="require CUDA"):
                    binding.register_rotary_snapshot(
                        99, context, [wrapper], 8, True, [0], [], [], "", b""
                    )
            binding.close()
    finally:
        shm_unlink(wrapper.shm_name)
