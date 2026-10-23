# SPDX-License-Identifier: Apache-2.0
"""Bind real CPU registrations; simulate physical server completion explicitly."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import MagicMock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.context_ownership import ContextOwnershipError
from lmcache.v1.multiprocess.kmesh_bindings import (
    RegisteredRuntime,
    RegisteredWorkerBindings,
)
from lmcache.v1.multiprocess.kmesh_domains import (
    Completion,
    DomainRegistry,
    KMeshFenceError,
    RegistrationState,
    SemanticIdentity,
)
from lmcache.v1.multiprocess.posix_shm import shm_unlink
from lmcache.v1.multiprocess.request_handler import iter_request_handlers
from lmcache.v1.multiprocess.transport.request_context import request_session
from lmcache.v1.platform.devices.cpu.shm import migrate_to_shm_and_wrap
from tests.v1.test_registered_rope_ownership import rotary_server


def semantics(runtime: RegisteredRuntime) -> SemanticIdentity:
    # Synthetic artifact digests; the server does not load model weights here.
    return SemanticIdentity(
        "1" * 64,
        "2" * 64,
        "3" * 64,
        runtime.rope_digest,
        runtime.layout_digest,
        runtime.kv_dtype,
        runtime.chunk_tokens,
        runtime.hash_algorithm,
    )


def test_runtime_derives_geometry_but_excludes_allocated_capacity() -> None:
    runtimes = []
    for blocks in (16, 32):
        with rotary_server(hash_algorithm="blake3", num_blocks=blocks) as (
            blend,
            transfer,
        ):
            inventory = [
                h.operation
                for module in (blend, transfer)
                for h in iter_request_handlers(module)
            ]
            bindings = RegisteredWorkerBindings(
                DomainRegistry(inventory), transfer, blend
            )
            with request_session("zmq", b"worker"):
                runtimes.append(bindings.describe(99))
    first, second = runtimes
    assert first.context != second.context
    assert first.layout_digest == second.layout_digest
    assert first.rope_digest == second.rope_digest
    assert first.chunk_tokens == 8 and first.hash_algorithm == "blake3"
    assert first.model_name == "synthetic-model" and first.world_size == 1


def test_runtime_refuses_an_unqualified_hash_implementation() -> None:
    with rotary_server() as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        with request_session("zmq", b"worker"):
            with pytest.raises(KMeshFenceError, match="stable MP BLAKE3"):
                bindings.describe(99)


def test_device_discovery_refuses_before_any_device_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock device metadata on real CPU tensors; never allocate a device tensor."""
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        copy = MagicMock(side_effect=AssertionError("unqualified device copy"))
        with monkeypatch.context() as patch:
            patch.setattr(
                torch.Tensor, "device", property(lambda self: torch.device("cuda"))
            )
            patch.setattr(torch.Tensor, "cpu", copy)
            with request_session("zmq", b"worker"):
                with pytest.raises(KMeshFenceError, match="qualified copy/drain"):
                    bindings.describe(99)
            copy.assert_not_called()


@pytest.mark.parametrize("mismatch", ["layout", "rotary", "chunk", "world"])
def test_binding_rejects_semantics_that_disagree_with_actual_registration(
    mismatch: str,
) -> None:
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        with request_session("zmq", b"worker"):
            runtime = bindings.describe(99)
            identity = semantics(runtime)
            if mismatch == "layout":
                identity = replace(identity, layout_digest="4" * 64)
            elif mismatch == "rotary":
                identity = replace(identity, rope_digest="4" * 64)
            elif mismatch == "chunk":
                identity = replace(identity, chunk_tokens=16)
            lease = bindings.open_registration(
                identity, "engine", 2 if mismatch == "world" else 1, "5" * 64
            )
            with pytest.raises(KMeshFenceError, match="registered runtime"):
                bindings.bind(lease, 0, 99, runtime, identity, "5" * 64)
        assert bindings.state(lease) == RegistrationState.REVOKED
        assert bindings.pending(lease) == 0


def test_rotary_contents_change_runtime_evidence_and_refuse_old_snapshot() -> None:
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        tensor = torch.full((64, 8), 2.0**-0.5, dtype=torch.float32)
        wrapper = migrate_to_shm_and_wrap(tensor)
        try:
            with request_session("zmq", b"worker"):
                before = bindings.describe(99)
                identity = semantics(before)
                lease = bindings.open_registration(identity, "engine", 1, "5" * 64)
                blend.cb_register_rope(99, [wrapper], 8, True, [0])
                after = bindings.describe(99)
                assert before.layout_digest == after.layout_digest
                assert before.rope_digest != after.rope_digest
                with pytest.raises(ContextOwnershipError):
                    bindings.bind(lease, 0, 99, before, identity, "5" * 64)
                with pytest.raises(KMeshFenceError, match="registered runtime"):
                    bindings.bind(lease, 0, 99, after, identity, "5" * 64)
        finally:
            shm_unlink(wrapper.shm_name)
            bindings.revoke(lease)


@pytest.mark.parametrize("retire", ["paged", "rotary"])
def test_registration_retirement_keeps_accepted_work_until_physical_completion(
    retire: str,
) -> None:
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        with request_session("zmq", b"worker"):
            runtime = bindings.describe(99)
            identity = semantics(runtime)
            lease = bindings.open_registration(identity, "engine", 1, "5" * 64)
            fence = bindings.bind(lease, 0, 99, runtime, identity, "5" * 64)
            operation = bindings.begin(fence, "request", 5)
        starting = threading.Event()

        def unregister() -> None:
            starting.set()
            if retire == "paged":
                transfer.unregister_kv_cache(99)
            else:
                blend.cb_unregister_rope(99)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(unregister)
                try:
                    assert starting.wait(timeout=5)
                    # A retirement callback releases only persistent holds.
                    # Wait on the public registry state rather than assume scheduling.
                    for _ in range(500):
                        if bindings.state(lease) == RegistrationState.REVOKED:
                            break
                        threading.Event().wait(0.01)
                    assert bindings.state(lease) == RegistrationState.REVOKED
                    assert not future.done()
                    assert bindings.pending(lease) == 1
                    assert operation.paged.entry.cache_context.kv_tensors[0].sum() == 0
                    with pytest.raises(KMeshFenceError):
                        bindings.publish(operation, lambda namespace: None)
                    bindings.revoke(lease)
                    assert bindings.pending(lease) == 1
                finally:
                    bindings.finish(operation, Completion.CANCELLED_DRAINED)
                future.result(timeout=5)
        finally:
            bindings.revoke(lease)
        assert bindings.pending(lease) == 0


def test_undrained_failure_and_generation_survive_revocation() -> None:
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        with request_session("zmq", b"worker"):
            runtime = bindings.describe(99)
            identity = semantics(runtime)
            lease = bindings.open_registration(identity, "engine", 1, "5" * 64)
            fence = bindings.bind(lease, 0, 99, runtime, identity, "5" * 64)
            operation = bindings.begin(fence, "request", 5)
        try:
            bindings.finish(operation, Completion.FAILED_UNDRAINED)
            bindings.revoke(lease)
            assert bindings.state(lease) == RegistrationState.REVOKED
            assert bindings.pending(lease) == 1
            with pytest.raises(KMeshFenceError, match="owes a drain"):
                bindings.open_registration(identity, "engine", 1, "5" * 64)
        finally:
            bindings.finish(operation, Completion.CANCELLED_DRAINED)
        replacement = bindings.open_registration(identity, "engine", 1, "5" * 64)
        with request_session("zmq", b"worker"):
            new_fence = bindings.bind(replacement, 0, 99, runtime, identity, "5" * 64)
            current = bindings.begin(new_fence, "request", 5)
            bindings.finish(current, Completion.CANCELLED_DRAINED)
            for old_generation in (2, 5):
                with pytest.raises(KMeshFenceError, match="rolled back"):
                    bindings.begin(new_fence, "request", old_generation)
        bindings.revoke(replacement)


def test_incomplete_barrier_cannot_accept_work_and_foreign_client_cannot_bind() -> None:
    with rotary_server(hash_algorithm="blake3", world_size=2) as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        with request_session("zmq", b"worker"):
            runtime = bindings.describe(99)
            identity = semantics(runtime)
            lease = bindings.open_registration(identity, "engine", 2, "5" * 64)
        with request_session("zmq", b"foreign"):
            with pytest.raises(ContextOwnershipError):
                bindings.bind(lease, 0, 99, runtime, identity, "5" * 64)
        try:
            with request_session("zmq", b"worker"):
                fence = bindings.bind(lease, 0, 99, runtime, identity, "5" * 64)
                assert bindings.state(lease) == RegistrationState.REGISTERING
                with pytest.raises(KMeshFenceError, match="barrier is incomplete"):
                    bindings.begin(fence, "request", 1)
                assert bindings.pending(lease) == 0
        finally:
            bindings.revoke(lease)


@pytest.mark.parametrize("invalidation", ["store", "capabilities"])
def test_server_publication_and_late_completion_keep_separate_authority(
    invalidation: str,
) -> None:
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        with request_session("zmq", b"worker"):
            runtime = bindings.describe(99)
            identity = semantics(runtime)
            lease = bindings.open_registration(identity, "engine", 1, "5" * 64)
            fence = bindings.bind(lease, 0, 99, runtime, identity, "5" * 64)
            operation = bindings.begin(fence, "request", 1)
        namespaces: list[str] = []
        try:
            # Completion callbacks can run outside the receiving RPC context.
            bindings.publish(operation, namespaces.append)
            assert namespaces == [lease.domain.namespace]
            if invalidation == "store":
                bindings.invalidate_store()
            else:
                bindings.update_capabilities(["new_handler"])
            assert bindings.state(lease) == RegistrationState.REVOKED
            assert bindings.pending(lease) == 1
            with pytest.raises(KMeshFenceError):
                bindings.publish(operation, namespaces.append)
            assert len(namespaces) == 1
        finally:
            bindings.finish(operation, Completion.SUCCEEDED)
        assert bindings.pending(lease) == 0
        assert bindings.state(lease) == RegistrationState.REVOKED
        replacement = bindings.open_registration(identity, "engine", 1, "5" * 64)
        if invalidation == "store":
            assert replacement.domain.namespace != lease.domain.namespace
        else:
            assert (
                replacement.server.capabilities_digest
                != lease.server.capabilities_digest
            )
        bindings.revoke(replacement)
