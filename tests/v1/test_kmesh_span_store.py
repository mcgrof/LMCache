# SPDX-License-Identifier: Apache-2.0
"""Refuse unsupported span execution before native work using real CPU bindings."""

# Standard
from dataclasses import replace

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.kmesh_bindings import RegisteredWorkerBindings
from lmcache.v1.multiprocess.kmesh_domains import DomainRegistry, KMeshFenceError
from lmcache.v1.multiprocess.kmesh_span_store import KMeshSpanStore
from lmcache.v1.multiprocess.transport.request_context import request_session
from tests.v1.test_kmesh_bindings import semantics
from tests.v1.test_registered_rope_ownership import rotary_server


@pytest.mark.parametrize(
    "case", ["key-schema", "empty-event", "foreign", "cpu-runtime"]
)
def test_span_refusal_preserves_exact_completion_boundary(case: str) -> None:
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        executor = KMeshSpanStore(bindings, transfer.context.storage_manager, blend)
        try:
            with request_session("zmq", b"worker"):
                runtime = bindings.describe(99)
                identity = replace(
                    semantics(runtime),
                    key_schema=(
                        "kmesh-mp-v1" if case == "key-schema" else "kmesh-span-v1"
                    ),
                )
                lease = bindings.open_registration(
                    identity, "synthetic-cpu", 1, "4" * 64
                )
                fence = bindings.bind(lease, 0, 99, runtime, identity, "4" * 64)
            with request_session("zmq", b"foreign" if case == "foreign" else b"worker"):
                with pytest.raises(KMeshFenceError, match="unknown|another client"):
                    executor.status(fence, "request", 1)
                args = (
                    fence,
                    "request",
                    1,
                    list(range(8)),
                    0,
                    "5" * 64,
                    [[1, 2]],
                    b"" if case == "empty-event" else b"not-a-device-event",
                )
                if case == "cpu-runtime":
                    result = executor.store(*args)
                    assert result.state == "failed-drained" and result.span is None
                    assert "runtime geometry" in result.error
                    assert executor.status(fence, "request", 1) == result
                    assert executor.store(*args) == result
                    assert executor.drain(fence, "request", 1) == result
                else:
                    with pytest.raises(KMeshFenceError, match="metadata|owner"):
                        executor.store(*args)
                assert bindings.pending(lease) == 0
        finally:
            bindings.close()
