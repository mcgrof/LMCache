# SPDX-License-Identifier: Apache-2.0
"""Ordinary cache reads cannot masquerade as publication restores."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)
from lmcache.v1.storage_backend.raw_block import RawBlockIoContext


@pytest.fixture
def backend() -> RustRawBlockBackend:
    obj = RustRawBlockBackend.__new__(RustRawBlockBackend)
    obj._put_lock = threading.Lock()
    obj._sealed = False
    obj._active_operations = 0
    obj._run_id = "run"
    obj._ack_tp_rank = 0
    obj._core = SimpleNamespace(writer_epoch="writer")  # type: ignore[assignment]
    obj._load_prefix = Mock(return_value=[])  # type: ignore[method-assign]
    return obj


def test_blocking_loads_have_distinct_local_non_publication_contexts(backend):
    backend._batched_get_prefix([])
    backend._batched_get_prefix([])
    contexts = [
        call.kwargs["io_context"] for call in backend._load_prefix.call_args_list
    ]
    assert contexts[0].tag() != contexts[1].tag()
    for context in contexts:
        assert context.request_id.startswith("<ordinary-cache-load:")
        assert context.work_kind == "ordinary_cache"
        assert context.run_id == "run" and context.incarnation == "writer"
        assert not context.manifest_digest and not context.restore_attempt_id
    assert backend._active_operations == 0


def test_publication_context_is_not_replaced(backend):
    context = RawBlockIoContext(
        request_id="producer-request", restore_attempt_id="attempt"
    )
    backend._batched_get_prefix([], io_context=context)
    assert backend._load_prefix.call_args.kwargs["io_context"] is context


@pytest.mark.asyncio
async def test_async_load_retains_lookup_identity_without_claiming_a_publication(
    backend,
):
    await backend.batched_get_non_blocking("lookup-request", [])
    context = backend._load_prefix.call_args.kwargs["io_context"]
    assert context.request_id == "lookup-request"
    assert context.work_kind == "ordinary_cache"
    assert not context.manifest_digest and not context.restore_attempt_id
    assert backend._active_operations == 0


def test_load_exception_releases_admission_with_a_context(backend):
    backend._load_prefix.side_effect = RuntimeError("load failed")
    with pytest.raises(RuntimeError, match="load failed"):
        backend._batched_get_prefix([])
    assert backend._active_operations == 0
    assert backend._load_prefix.call_args.kwargs["io_context"].tag()
