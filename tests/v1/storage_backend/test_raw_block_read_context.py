# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)
from lmcache.v1.storage_backend.raw_block import (
    RawBlockIoContext,
    RawBlockPublicationReceipt,
    RawBlockReadContext,
)
from lmcache.v1.storage_backend.storage_manager import StorageManager
from tests.v1.storage_backend.raw_block_test_utils import make_empty_memory_obj


def _manager(backend: Any) -> StorageManager:
    manager = StorageManager.__new__(StorageManager)
    manager.storage_backends = OrderedDict([("reader", backend)])
    manager._freeze_lock = threading.RLock()
    manager._freeze = False
    manager._bypass_lock = threading.RLock()
    manager._bypassed_backends = set()
    return manager


def _context(request_id: str) -> RawBlockReadContext:
    return RawBlockReadContext(
        request_id,
        RawBlockPublicationReceipt(
            writer_epoch="published-writer",
            checkpoint_seq=7,
            key_count=2,
            manifest_digest=request_id,
            namespace_identity="namespace-under-test",
        ),
        consumer_request_id=f"consumer-{request_id}",
        restore_attempt_id=f"attempt-{request_id}",
        adopted_checkpoint_seq=9,
    )


@pytest.mark.parametrize("fail_second_request", [False, True])
def test_overlapping_restores_keep_the_request_that_issued_each_read(
    fail_second_request: bool,
) -> None:
    """Carry identities across the real engine, manager and backend calls.

    The storage I/O boundary blocks both requests after submission, and one
    may fail its suffix. Their shared prefix key cannot identify either
    request; the context supplied with each restore must survive unchanged.
    """
    entered = threading.Barrier(2)
    seen: list[RawBlockIoContext] = []
    seen_lock = threading.Lock()

    def load(
        keys: list[str],
        objects: list[MemoryObj],
        *,
        io_context: RawBlockIoContext,
    ) -> list[bool]:
        assert io_context is not None, "restore identity lost before payload read"
        with seen_lock:
            seen.append(io_context)
        entered.wait(timeout=5)
        return [True, not (fail_second_request and io_context.request_id == "wire-b")]

    # Only the payload boundary is simulated. Constructors are covered by
    # backend integration tests; these fixtures exercise public restore calls.
    backend: Any = RustRawBlockBackend.__new__(RustRawBlockBackend)
    backend._role = "reader"
    backend._run_id = "run-under-test"
    backend._ack_tp_rank = 3
    backend._pin_lock = threading.Lock()
    backend._put_lock = threading.Lock()
    backend._sealed = False
    backend._active_operations = 0
    backend._pinned_keys = set()
    backend._gpu_allocator = None
    backend._quarantined_objs = []
    backend._core = SimpleNamespace(
        namespace_identity="namespace-under-test",
        # A later adoption must not change the earlier restore's epoch.
        writer_epoch="later-writer",
        get_metadata_prefix=lambda keys, **kwargs: [
            SimpleNamespace(
                shape=torch.Size([512]), dtype=torch.uint8, fmt=MemoryFormat.BINARY
            )
            for _ in keys
        ],
        load_many_into=load,
        unlock_many=lambda keys: None,
        is_poisoned=lambda: False,
        raw_device=lambda: SimpleNamespace(is_poisoned=lambda: False),
    )
    backend._allocate_load_target = lambda *args: make_empty_memory_obj(512)
    manager = _manager(backend)
    engine: Any = LMCacheEngine.__new__(LMCacheEngine)
    engine._init_failed = False
    engine._health_monitor = None
    engine.kvcache_check_log_enabled = False
    engine.async_loading = False
    engine.save_only_first_rank = False
    engine.stats_monitor = MagicMock()
    stats = engine.stats_monitor.on_retrieve_request.return_value
    stats.profile_process_tokens.side_effect = nullcontext
    stats.profile_to_gpu.side_effect = nullcontext
    stats.time_to_retrieve.return_value = 1.0
    gpu_calls: list[dict[str, Any]] = []

    def to_gpu(
        memory_objs: list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> None:
        gpu_calls.append(kwargs)

    engine.gpu_connector = SimpleNamespace(batched_to_gpu=to_gpu)
    engine.storage_manager = manager
    engine.lookup_pins = {"local-a": {"reader": []}, "local-b": {"reader": []}}
    engine.remove_after_retrieve = False
    engine.token_database = SimpleNamespace(
        process_tokens=lambda *, tokens, **kwargs: [
            (index, index + 1, CacheEngineKey("model", 1, 3, token, torch.uint8))
            for index, token in enumerate(tokens)
        ]
    )

    def restore(name: str, tokens: list[int]) -> list[bool]:
        return engine.retrieve(
            tokens,
            req_id="local-" + name,
            storage_pd_read_context=_context("wire-" + name),
        ).tolist()

    with ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(restore, "a", [10, 20])
        b = executor.submit(restore, "b", [10, 30])
        assert a.result(timeout=10) == [True, True]
        assert b.result(timeout=10) == [True, not fail_second_request]
    assert len(gpu_calls) == 2
    assert all("storage_pd_read_context" not in kwargs for kwargs in gpu_calls)
    assert sorted(context.tag() for context in seen) == [
        "run-under-test/wire-a/r3/published-writer/aattempt-wire-a",
        "run-under-test/wire-b/r3/published-writer/aattempt-wire-b",
    ]
    assert {context.consumer_request_id for context in seen} == {
        "consumer-wire-a",
        "consumer-wire-b",
    }
    assert {context.checkpoint_seq for context in seen} == {7}
    assert {context.manifest_digest for context in seen} == {"wire-a", "wire-b"}
    assert {context.adopted_checkpoint_seq for context in seen} == {9}


@pytest.mark.parametrize("context", [None, _context("wire-a")])
def test_ordinary_cache_hits_keep_the_original_backend_call_signature(
    context: RawBlockReadContext | None,
) -> None:
    calls: list[list[CacheEngineKey]] = []
    key = CacheEngineKey("model", 1, 3, 10, torch.uint8)
    obj = make_empty_memory_obj(512)

    def get(keys: list[CacheEngineKey]) -> list[MemoryObj]:
        calls.append(keys)
        return [obj]

    manager = _manager(SimpleNamespace(batched_get_blocking=get))
    try:
        assert manager.batched_get(
            [key], location="reader", storage_pd_read_context=context
        ) == [obj]
        assert calls == [[key]]
    finally:
        obj.ref_count_down()


@pytest.mark.parametrize(
    "role,namespace",
    [("writer", "namespace-under-test"), ("reader", "another-namespace")],
)
def test_publication_read_refuses_the_wrong_backend_before_payload_io(
    role: str,
    namespace: str,
) -> None:
    backend: Any = RustRawBlockBackend.__new__(RustRawBlockBackend)
    backend._role = role
    backend._core = SimpleNamespace(namespace_identity=namespace)
    with pytest.raises(RuntimeError, match="does not name this reader's namespace"):
        backend.batched_get_for_publication([], _context("wire-a"))
