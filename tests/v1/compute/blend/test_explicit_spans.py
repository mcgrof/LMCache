# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Generator, Optional, cast
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMBufferLayerwiseGPUConnector,
)
from lmcache.v1.gpu_connector.load_state import LayerwiseLoadState
from lmcache.v1.gpu_connector.xpu_connectors import (
    VLLMBufferLayerwiseXPUConnector,
)
from lmcache.v1.lookup_client.lmcache_lookup_client import (
    LMCacheLookupClient,
    LMCacheLookupServer,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.storage_manager import (
    PutOwnershipReceipt,
    StorageManager,
    StorageManagerClosedError,
)
from lmcache.v1.token_database import SegmentTokenDatabase
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan
from tests.v1.utils import dumb_metadata
import lmcache.v1.cache_engine as cache_engine_module
import lmcache.v1.gpu_connector.gpu_connectors as gpu_connector_module
import lmcache.v1.storage_backend.storage_manager as storage_manager_module


class InjectedExplicitLoadFailure(RuntimeError):
    """Failure injected after a layer's storage objects are acquired."""


class InjectedCleanupFailure(BaseException):
    """BaseException used to verify retryable terminal ownership."""


@dataclass
class _Memory:
    payload: Optional[tuple[int, int, int]] = None
    ref_count: int = 1
    ref_down_count: int = 0
    pin_count: int = 0
    unpin_count: int = 0

    def get_size(self) -> int:
        return 1

    def ref_count_up(self) -> None:
        self.ref_count += 1

    def ref_count_down(self) -> None:
        self.ref_down_count += 1
        self.ref_count -= 1
        if self.ref_count < 0:
            raise AssertionError("negative fake reference count")

    def pin(self) -> None:
        self.pin_count += 1

    @property
    def is_pinned(self) -> bool:
        return self.pin_count > 0

    def unpin(self) -> None:
        if self.pin_count <= 0:
            raise AssertionError("unpin without a matching pin")
        self.pin_count -= 1
        self.unpin_count += 1


class _RetryCleanupMemory(_Memory):
    """Memory object whose release can fail before consuming its reference."""

    def __init__(self, failures: int = 0) -> None:
        super().__init__()
        self.release_attempts = 0
        self.release_failures = failures

    def ref_count_down(self) -> None:
        self.release_attempts += 1
        if self.release_failures:
            self.release_failures -= 1
            raise InjectedCleanupFailure("injected memory release failure")
        super().ref_count_down()


class _Task:
    def __init__(self, memory_objs: list[_Memory]) -> None:
        self.memory_objs = memory_objs

    def result(self) -> list[MemoryObj]:
        return cast(list[MemoryObj], self.memory_objs)


class _Storage:
    location = "LocalCPUBackend"

    def __init__(self, retrieval_owns_pins: bool = False) -> None:
        self.objects: dict[CacheEngineKey, _Memory] = {}
        self.allocations: list[_Memory] = []
        self.contains_calls: list[CacheEngineKey] = []
        self.touch_count = 0
        self.retrieval_owns_pins = retrieval_owns_pins
        self.cancelled_requests: list[str] = []

    def contains(
        self,
        key: CacheEngineKey,
        search_range: Optional[list[str]] = None,
    ) -> Optional[str]:
        del search_range
        self.contains_calls.append(key)
        return self.location if key in self.objects else None

    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        **kwargs: Any,
    ) -> list[MemoryObj]:
        del shape, dtype, kwargs
        allocated = [_Memory() for _ in range(batch_size)]
        self.allocations.extend(allocated)
        return cast(list[MemoryObj], allocated)

    def batched_put(
        self,
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        **kwargs: Any,
    ) -> None:
        ownership_receipt = kwargs.pop("ownership_receipt", None)
        if ownership_receipt is not None:
            ownership_receipt._adopt()
        del kwargs
        for key, memory_obj in zip(keys, memory_objs, strict=True):
            prior = self.objects.get(key)
            if prior is not None and prior is not memory_obj:
                prior.ref_count_down()
            self.objects[key] = cast(_Memory, memory_obj)

    def batched_contains(
        self,
        keys: list[CacheEngineKey],
        search_range: Optional[list[str]],
        pin: bool,
    ) -> tuple[int, dict[str, list[CacheEngineKey]]]:
        del search_range
        hits: list[CacheEngineKey] = []
        for key in keys:
            if key not in self.objects:
                break
            hits.append(key)
            if pin:
                self.objects[key].pin()
        mapping = {self.location: hits} if hits else {}
        return len(hits), mapping

    def iter_lookup_pins(
        self,
        keys: list[CacheEngineKey],
        search_range: Optional[list[str]],
    ) -> Generator[tuple[str, CacheEngineKey], None, None]:
        del search_range
        for key in keys:
            if key not in self.objects:
                return
            self.objects[key].pin()
            yield self.location, key

    def batched_unpin(self, keys: list[CacheEngineKey], locations: list[str]) -> None:
        assert locations == [self.location]
        for key in keys:
            self.objects[key].unpin()

    def touch_cache(self) -> None:
        self.touch_count += 1

    def cancel_request(self, req_id: str) -> None:
        self.cancelled_requests.append(req_id)

    def drain_pending_put_cleanups(self) -> None:
        return

    def pending_put_cleanup_count(self) -> int:
        return 0

    def layerwise_batched_get(
        self,
        keys_layer_major: list[list[CacheEngineKey]],
        location: str,
    ) -> Generator[_Task, None, None]:
        assert location == self.location
        for keys in keys_layer_major:
            memory_objs = [self.objects[key] for key in keys]
            for memory_obj in memory_objs:
                memory_obj.ref_count_up()
                if self.retrieval_owns_pins:
                    memory_obj.pin()
            yield _Task(memory_objs)

    def layerwise_batched_get_returns_pinned(self, location: str) -> bool:
        assert location == self.location
        return self.retrieval_owns_pins


class _LayerwiseConnector:
    def __init__(self, num_layers: int, fail_on_layer: Optional[int] = None) -> None:
        self.num_layers = num_layers
        self.fail_on_layer = fail_on_layer
        self.stored_spans: list[tuple[int, int]] = []
        self.loaded_spans: list[tuple[int, int]] = []
        self.loaded_payloads: list[list[Optional[tuple[int, int, int]]]] = []
        self.buffer_span: Optional[TokenSpan] = None
        self.closed = False
        self.store_closed = False

    def supports_explicit_spans(self) -> bool:
        return True

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size((2, num_tokens, 1))

    def drain_pending_gpu_cleanups(self) -> None:
        return

    def pending_gpu_cleanup_count(self) -> int:
        return 0

    def batched_from_gpu(
        self,
        memory_objs: list[list[MemoryObj]],
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> Generator[None, None, None]:
        del kwargs
        self.stored_spans = list(zip(starts, ends, strict=True))
        try:
            yield
            for layer_id, layer_objs in enumerate(memory_objs):
                for start, end, memory_obj in zip(
                    starts,
                    ends,
                    layer_objs,
                    strict=True,
                ):
                    cast(_Memory, memory_obj).payload = (layer_id, start, end)
                yield
        finally:
            self.store_closed = True

    def batched_to_gpu(
        self,
        starts: list[int],
        ends: list[int],
        *,
        buffer_span: Optional[TokenSpan] = None,
        **kwargs: Any,
    ) -> Generator[None, list[MemoryObj], None]:
        del kwargs
        self.loaded_spans = list(zip(starts, ends, strict=True))
        self.buffer_span = buffer_span
        try:
            for layer_id in range(self.num_layers):
                memory_objs = yield
                self.loaded_payloads.append(
                    [cast(_Memory, memory_obj).payload for memory_obj in memory_objs]
                )
                if self.fail_on_layer == layer_id:
                    raise InjectedExplicitLoadFailure(
                        f"injected explicit load failure at layer {layer_id}"
                    )
            yield
            yield
        finally:
            self.closed = True


class _StatsMonitor:
    def on_store_request(self, count: int) -> int:
        return count

    def on_store_finished(self, monitor_id: int, count: int) -> None:
        del monitor_id, count

    def on_retrieve_request(self, count: int) -> int:
        return count

    def on_retrieve_finished(self, monitor_id: int, count: torch.Tensor) -> None:
        del monitor_id, count

    def on_lookup_request(self, count: int) -> int:
        return count

    def on_lookup_finished(self, monitor_id: int, count: int) -> None:
        del monitor_id, count


class _TensorMemory:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.metadata = SimpleNamespace(fmt=MemoryFormat.KV_2TD)
        self.release_count = 0

    def ref_count_down(self) -> None:
        self.release_count += 1


class _TensorAllocator:
    def __init__(self) -> None:
        self.allocations: list[_TensorMemory] = []

    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
    ) -> MemoryObj:
        assert fmt == MemoryFormat.KV_2TD
        memory_obj = _TensorMemory(torch.full(shape, -1.0, dtype=dtype))
        self.allocations.append(memory_obj)
        return cast(MemoryObj, memory_obj)


def _token_database() -> SegmentTokenDatabase:
    metadata = dumb_metadata()
    database = SegmentTokenDatabase.__new__(SegmentTokenDatabase)
    database.metadata = metadata
    database.save_only_first_rank = False
    database.hash_func = hash
    database.sep_tokens = torch.tensor([99])
    database.sep_len = 1
    return database


def _engine(
    *,
    num_layers: int = 2,
    fail_on_layer: Optional[int] = None,
    retrieval_owns_pins: bool = False,
) -> tuple[LMCacheEngine, _Storage, _LayerwiseConnector]:
    database = _token_database()
    storage = _Storage(retrieval_owns_pins)
    connector = _LayerwiseConnector(num_layers, fail_on_layer)
    engine = LMCacheEngine.__new__(LMCacheEngine)
    state = cast(Any, engine)
    state.config = SimpleNamespace(get_extra_config_value=lambda key, default: default)
    state.metadata = database.metadata
    state.token_database = database
    state.gpu_connector = connector
    state.storage_manager = storage
    state.use_layerwise = True
    state.num_layers = num_layers
    state.fmt = MemoryFormat.KV_2TD
    state.retrieve_locations = None
    state.store_location = storage.location
    state.stats_monitor = _StatsMonitor()
    state.lookup_pins = defaultdict(lambda: defaultdict(list))
    state.lookup_pin_generations = {}
    state.lookup_cancelled_generations = OrderedDict()
    state._lookup_pins_lock = threading.RLock()
    state.kv_events_enabled = False
    state.save_only_first_rank = False
    state.remove_after_retrieve = False
    state.async_loading = None
    state.kvcache_check_log_enabled = False
    state.is_healthy = lambda: True
    state.is_frozen = lambda: False
    return engine, storage, connector


def _keys_for_span(
    engine: LMCacheEngine,
    tokens: list[int],
    declaration: ExplicitTokenSpans,
    span_index: int,
) -> list[CacheEngineKey]:
    database = cast(SegmentTokenDatabase, engine.token_database)
    key = list(database.process_explicit_spans(tokens, declaration))[span_index][2]
    assert isinstance(key, CacheEngineKey)
    return [layer_key for layer_key in key.split_layers(engine.num_layers)]


def _populate_span(
    engine: LMCacheEngine,
    storage: _Storage,
    tokens: list[int],
    declaration: ExplicitTokenSpans,
    span_index: int,
) -> list[_Memory]:
    memory_objs = [_Memory() for _ in range(engine.num_layers)]
    for key, memory_obj in zip(
        _keys_for_span(engine, tokens, declaration, span_index),
        memory_objs,
        strict=True,
    ):
        storage.objects[key] = memory_obj
    return memory_objs


class _LoopbackLookupTransport:
    world_size = 1

    def __init__(self, server: LMCacheLookupServer) -> None:
        self.server = server
        self.call_count = 0
        self.messages: list[list[Any]] = []

    def send_and_recv_all(self, message: list[Any]) -> list[bytes]:
        self.call_count += 1
        self.messages.append(message)
        return [self.server._process_request_frames(message)]

    def close(self) -> None:
        return


def _sync_lookup_client(
    engine: LMCacheEngine,
) -> tuple[LMCacheLookupClient, _LoopbackLookupTransport]:
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine
    server.enable_blending = True
    transport = _LoopbackLookupTransport(server)
    client = LMCacheLookupClient.__new__(LMCacheLookupClient)
    client.config = SimpleNamespace(
        use_layerwise=True,
        enable_async_loading=False,
    )
    client.enable_blending = True
    client.transport = transport  # type: ignore[assignment]
    client.reqs_status = {}
    client.reqs_fingerprints = {}
    client.reqs_generations = {}
    client.pending_cancel_ids = set()
    return client, transport


def _wait_for_lookup_cancel(client: LMCacheLookupClient, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while client.pending_cancel_ids:
        if time.monotonic() >= deadline:
            pytest.fail("timed out waiting for lookup cancel retry")
        time.sleep(0.01)


def test_lookup_pin_progress_preflight_rejects_every_backend_before_pinning() -> None:
    class _SafeBackend:
        def __init__(self) -> None:
            self.pin_calls = 0

        def supports_lookup_pin_progress(self) -> bool:
            return True

        def pin_for_lookup(self, key: CacheEngineKey) -> bool:
            del key
            self.pin_calls += 1
            return True

    class _UnsafeBackend:
        def supports_lookup_pin_progress(self) -> bool:
            return False

    engine, _, _ = _engine()
    declaration = ExplicitTokenSpans(
        2,
        TokenSpan(0, 2),
        (TokenSpan(0, 2),),
    )
    key = _keys_for_span(engine, [0, 1], declaration, 0)[0]
    safe_backend = _SafeBackend()
    manager = StorageManager.__new__(StorageManager)
    cast(Any, manager).get_active_storage_backends = lambda **kwargs: iter(
        [
            ("LocalCPUBackend", safe_backend),
            ("UnsafeBackend", _UnsafeBackend()),
        ]
    )

    with pytest.raises(NotImplementedError, match="UnsafeBackend"):
        list(manager.iter_lookup_pins([key]))
    assert safe_backend.pin_calls == 0


def test_false_backend_unpin_preserves_exact_lookup_ledger() -> None:
    class _MissingUnpinBackend:
        def __init__(self) -> None:
            self.unpin_calls = 0

        def unpin(self, key: CacheEngineKey) -> bool:
            del key
            self.unpin_calls += 1
            return False

    engine, _, _ = _engine()
    declaration = ExplicitTokenSpans(
        2,
        TokenSpan(0, 2),
        (TokenSpan(0, 2),),
    )
    key = _keys_for_span(engine, [0, 1], declaration, 0)[0]
    lookup_id = "missing-backend-owner"
    generation = b"g" * 16
    backend = _MissingUnpinBackend()
    manager = StorageManager.__new__(StorageManager)
    cast(Any, manager).storage_backends = {"LocalCPUBackend": backend}
    cast(Any, engine).storage_manager = manager
    engine.lookup_pins[lookup_id] = {"LocalCPUBackend": [key]}
    engine.lookup_pin_generations[lookup_id] = generation

    with pytest.raises(RuntimeError, match="owner is missing"):
        engine._lookup_unpin_once(lookup_id)

    assert backend.unpin_calls == 1
    assert engine.lookup_pins[lookup_id] == {"LocalCPUBackend": [key]}
    assert engine.lookup_pin_generations[lookup_id] == generation
    assert (lookup_id, generation) not in engine.lookup_cancelled_generations


def test_destructive_admin_paths_fail_closed_with_explicit_span_ownership() -> None:
    engine, storage, _ = _engine()

    with pytest.raises(NotImplementedError, match="destructive move"):
        engine.move(
            [0, 1],
            "LocalCPUBackend",
            ("peer", "LocalCPUBackend"),
            "move-event",
            do_copy=False,
        )
    with pytest.raises(NotImplementedError, match="compression is unavailable"):
        engine.compress(
            [0, 1],
            "cachegen",
            "LocalCPUBackend",
            "compress-event",
        )
    with pytest.raises(NotImplementedError, match="decompression is unavailable"):
        engine.decompress(
            [0, 1],
            "cachegen",
            "LocalCPUBackend",
            "decompress-event",
        )

    assert storage.objects == {}
    assert engine.lookup_pins == {}


def test_prefix_pin_acquisition_exception_is_progress_ledgered_and_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, storage, _ = _engine()
    tokens = list(range(2))
    declaration = ExplicitTokenSpans(
        2,
        TokenSpan(0, 2),
        (TokenSpan(0, 2),),
    )
    memory_objs = _populate_span(engine, storage, tokens, declaration, 0)
    ordinary_iter = storage.iter_lookup_pins

    def fail_after_first_pin(keys, search_range):
        for location, key in ordinary_iter(keys, search_range):
            yield location, key
            raise RuntimeError("injected acquisition failure after prefix pin")

    monkeypatch.setattr(storage, "iter_lookup_pins", fail_after_first_pin)
    client, _ = _sync_lookup_client(engine)
    lookup_id = "prefix-pin-acquisition-failure"

    assert client.lookup(tokens, lookup_id, explicit_spans=declaration) is None
    assert client.pending_cancel_ids == set()
    assert lookup_id not in engine.lookup_pins
    assert lookup_id not in engine.lookup_pin_generations
    assert memory_objs[0].pin_count == 0
    assert memory_objs[0].unpin_count == 1
    assert memory_objs[1].pin_count == 0
    assert memory_objs[1].unpin_count == 0
    assert any(
        cancelled_id == lookup_id
        for cancelled_id, _ in engine.lookup_cancelled_generations
    )
    client.close()


def test_incomplete_hit_cleanup_failure_retries_before_cancel_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, storage, _ = _engine()
    tokens = list(range(2))
    declaration = ExplicitTokenSpans(
        2,
        TokenSpan(0, 2),
        (TokenSpan(0, 2),),
    )
    memory_objs = _populate_span(engine, storage, tokens, declaration, 0)
    del storage.objects[_keys_for_span(engine, tokens, declaration, 0)[1]]
    ordinary_unpin = storage.batched_unpin
    unpin_calls = 0

    def fail_cleanup_and_first_cancel(keys, locations):
        nonlocal unpin_calls
        unpin_calls += 1
        if unpin_calls <= 2:
            raise RuntimeError("injected incomplete-hit unpin failure")
        ordinary_unpin(keys, locations)

    monkeypatch.setattr(storage, "batched_unpin", fail_cleanup_and_first_cancel)
    client, _ = _sync_lookup_client(engine)
    lookup_id = "incomplete-hit-cleanup-failure"

    assert client.lookup(tokens, lookup_id, explicit_spans=declaration) is None
    assert client.pending_cancel_ids == {lookup_id}
    assert list(engine.lookup_pins[lookup_id].values()) == [
        [_keys_for_span(engine, tokens, declaration, 0)[0]]
    ]
    assert lookup_id in engine.lookup_pin_generations
    assert memory_objs[0].pin_count == 1

    _wait_for_lookup_cancel(client)
    assert unpin_calls == 3
    assert lookup_id not in engine.lookup_pins
    assert lookup_id not in engine.lookup_pin_generations
    assert memory_objs[0].pin_count == 0
    assert memory_objs[0].unpin_count == 1
    assert any(
        cancelled_id == lookup_id
        for cancelled_id, _ in engine.lookup_cancelled_generations
    )
    client.close()


def test_later_span_failure_preserves_only_failed_rollback_key_until_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, storage, _ = _engine()
    tokens = list(range(4))
    declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(2, 4)),
    )
    first_span_objects = _populate_span(engine, storage, tokens, declaration, 0)
    ordinary_iter = storage.iter_lookup_pins
    span_calls = 0

    def fail_on_later_span(keys, search_range):
        nonlocal span_calls
        span_calls += 1
        if span_calls == 2:
            raise RuntimeError("injected later-span acquisition failure")
        yield from ordinary_iter(keys, search_range)

    monkeypatch.setattr(storage, "iter_lookup_pins", fail_on_later_span)
    ordinary_unpin = storage.batched_unpin
    unpin_calls = 0

    def partial_rollback_then_retry(keys, locations):
        nonlocal unpin_calls
        unpin_calls += 1
        if unpin_calls in (2, 3):
            raise RuntimeError("injected rollback/cancel failure")
        ordinary_unpin(keys, locations)

    monkeypatch.setattr(storage, "batched_unpin", partial_rollback_then_retry)
    client, _ = _sync_lookup_client(engine)
    lookup_id = "later-span-partial-rollback"

    assert client.lookup(tokens, lookup_id, explicit_spans=declaration) is None
    assert client.pending_cancel_ids == {lookup_id}
    remaining = [
        key
        for pinned_keys in engine.lookup_pins[lookup_id].values()
        for key in pinned_keys
    ]
    assert remaining == [_keys_for_span(engine, tokens, declaration, 0)[1]]
    assert first_span_objects[0].pin_count == 0
    assert first_span_objects[0].unpin_count == 1
    assert first_span_objects[1].pin_count == 1

    _wait_for_lookup_cancel(client)
    assert unpin_calls == 4
    assert lookup_id not in engine.lookup_pins
    assert lookup_id not in engine.lookup_pin_generations
    assert all(memory_obj.pin_count == 0 for memory_obj in first_span_objects)
    assert all(memory_obj.unpin_count == 1 for memory_obj in first_span_objects)
    client.close()


def test_sync_rpc_pins_retrieves_and_unpins_on_the_worker_engine(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, _ = _engine()
    cast(Any, engine).config = SimpleNamespace(
        enable_blending=True,
        get_extra_config_value=lambda key, default: default,
    )
    tokens = list(range(6))
    declaration = ExplicitTokenSpans(
        6,
        TokenSpan(0, 6),
        (TokenSpan(0, 2), TokenSpan(4, 6)),
    )
    _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)
    memory_objs = list(storage.objects.values())

    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine
    server.enable_blending = True
    transport = _LoopbackLookupTransport(server)
    client = LMCacheLookupClient.__new__(LMCacheLookupClient)
    client.config = SimpleNamespace(
        use_layerwise=True,
        enable_async_loading=False,
    )
    client.enable_blending = True
    client.transport = transport  # type: ignore[assignment]
    client.reqs_status = {}
    client.reqs_fingerprints = {}
    lookup_id = "worker-owned-explicit-lifecycle"

    assert (
        client.lookup(
            tokens,
            lookup_id,
            explicit_spans=declaration,
        )
        == 6
    )
    assert transport.call_count == 1
    assert lookup_id in engine.lookup_pins
    assert all(memory_obj.pin_count == 1 for memory_obj in memory_objs)
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)

    list(engine.retrieve_layer(tokens, explicit_spans=declaration))

    assert all(memory_obj.pin_count == 1 for memory_obj in memory_objs)
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)

    engine.lookup_unpin(lookup_id)

    assert lookup_id not in engine.lookup_pins
    assert all(memory_obj.pin_count == 0 for memory_obj in memory_objs)
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)


def test_sync_rpc_scheduler_abort_unpins_worker_before_allocation(
    monkeypatch,
) -> None:
    pytest.importorskip("vllm")

    # Third Party
    from vllm.v1.request import RequestStatus

    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, _ = _engine()
    cast(Any, engine).config = SimpleNamespace(
        enable_blending=True,
        get_extra_config_value=lambda key, default: default,
    )
    tokens = list(range(6))
    declaration = ExplicitTokenSpans(
        6,
        TokenSpan(0, 6),
        (TokenSpan(0, 2), TokenSpan(4, 6)),
    )
    _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)
    memory_objs = list(storage.objects.values())

    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine
    server.enable_blending = True
    transport = _LoopbackLookupTransport(server)
    client = LMCacheLookupClient.__new__(LMCacheLookupClient)
    client.config = SimpleNamespace(
        use_layerwise=True,
        enable_async_loading=False,
    )
    client.enable_blending = True
    client.transport = transport  # type: ignore[assignment]
    client.reqs_status = {}
    client.reqs_fingerprints = {}
    client.reqs_generations = {}
    client.pending_cancel_ids = set()
    lookup_id = "worker-owned-preallocation-abort"

    assert client.lookup(tokens, lookup_id, explicit_spans=declaration) == 6
    assert all(memory_obj.pin_count == 1 for memory_obj in memory_objs)
    assert lookup_id in engine.lookup_pins
    generation = client.reqs_generations[lookup_id]

    # Scheduling clears the cached probe result but must preserve the cleanup
    # identity through a later abort.
    client.clear_lookup_status(lookup_id)
    assert client.lookup_cache(lookup_id) == -1
    assert client.reqs_generations[lookup_id] == generation

    scheduler_connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    scheduler_connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=None,
        lookup_client=client,
    )
    scheduler_connector.async_loading = False
    scheduler_connector.use_layerwise = False
    scheduler_connector.config = SimpleNamespace(
        get_extra_config_value=lambda key, default: default
    )
    scheduler_connector._request_trackers = {}
    aborted_request = SimpleNamespace(
        request_id=lookup_id,
        status=RequestStatus.FINISHED_ABORTED,
    )

    scheduler_connector.request_finished(aborted_request, [])

    assert transport.call_count == 2
    assert lookup_id not in engine.lookup_pins
    assert all(memory_obj.pin_count == 0 for memory_obj in memory_objs)
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)
    assert storage.cancelled_requests == []
    assert client.lookup_cache(lookup_id) == -1
    assert lookup_id not in client.reqs_fingerprints
    assert lookup_id not in client.reqs_generations


def test_sync_rpc_cancel_retries_failed_worker_unpin_transactionally(
    monkeypatch,
) -> None:
    engine, storage, _ = _engine()
    cast(Any, engine).config = SimpleNamespace(
        enable_blending=True,
        get_extra_config_value=lambda key, default: default,
    )
    tokens = list(range(4))
    declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(2, 4)),
    )
    _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)
    memory_objs = list(storage.objects.values())

    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine
    server.enable_blending = True
    transport = _LoopbackLookupTransport(server)
    client = LMCacheLookupClient.__new__(LMCacheLookupClient)
    client.config = SimpleNamespace(
        use_layerwise=True,
        enable_async_loading=False,
    )
    client.enable_blending = True
    client.transport = transport  # type: ignore[assignment]
    client.reqs_status = {}
    client.reqs_fingerprints = {}
    client.reqs_generations = {}
    client.pending_cancel_ids = set()
    lookup_id = "retry-worker-unpin"

    assert client.lookup(tokens, lookup_id, explicit_spans=declaration) == len(tokens)
    assert all(memory_obj.pin_count == 1 for memory_obj in memory_objs)

    ordinary_batched_unpin = storage.batched_unpin
    unpin_calls = 0

    def fail_first_batched_unpin(keys, locations):
        nonlocal unpin_calls
        unpin_calls += 1
        if unpin_calls == 1:
            raise RuntimeError("injected worker unpin failure")
        ordinary_batched_unpin(keys, locations)

    monkeypatch.setattr(storage, "batched_unpin", fail_first_batched_unpin)
    client.cancel_lookup(lookup_id)

    # The failed transaction retains both server generation state and engine
    # pin ownership until the autonomous retry succeeds.
    assert client.pending_cancel_ids == {lookup_id}
    assert lookup_id in engine.lookup_pins
    assert sum(memory_obj.pin_count for memory_obj in memory_objs) == 1
    assert sum(len(keys) for keys in engine.lookup_pins[lookup_id].values()) == 1

    _wait_for_lookup_cancel(client)
    assert unpin_calls == 1 + len(memory_objs)
    assert lookup_id not in engine.lookup_pins
    assert all(memory_obj.pin_count == 0 for memory_obj in memory_objs)
    assert lookup_id not in client.reqs_generations


def test_lookup_unpin_retries_only_failed_keys_after_partial_progress(
    monkeypatch,
) -> None:
    engine, storage, _ = _engine()
    tokens = list(range(4))
    declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(2, 4)),
    )
    _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)
    memory_objs = list(storage.objects.values())
    lookup_id = "partial-unpin-progress"
    assert engine.lookup(
        tokens=tokens,
        lookup_id=lookup_id,
        pin=True,
        explicit_spans=declaration,
    ) == len(tokens)

    ordinary_batched_unpin = storage.batched_unpin
    call_count = 0

    def fail_after_first_key(keys, locations):
        nonlocal call_count
        call_count += 1
        assert len(keys) == 1
        if call_count == 2:
            raise RuntimeError("injected failure after prefix progress")
        ordinary_batched_unpin(keys, locations)

    monkeypatch.setattr(storage, "batched_unpin", fail_after_first_key)
    with pytest.raises(RuntimeError, match="prefix progress"):
        engine.lookup_unpin(lookup_id)

    remaining = [key for keys in engine.lookup_pins[lookup_id].values() for key in keys]
    assert len(remaining) == 1
    assert sum(memory_obj.pin_count for memory_obj in memory_objs) == 1

    engine.lookup_unpin(lookup_id)
    assert lookup_id not in engine.lookup_pins
    assert all(memory_obj.pin_count == 0 for memory_obj in memory_objs)
    assert all(memory_obj.unpin_count == 1 for memory_obj in memory_objs)


def test_lookup_unpin_serializes_concurrent_normal_and_abort_cleanup(
    monkeypatch,
) -> None:
    engine, storage, _ = _engine()
    tokens = list(range(4))
    declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(2, 4)),
    )
    _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)
    memory_objs = list(storage.objects.values())
    lookup_id = "concurrent-unpin"
    assert engine.lookup(
        tokens=tokens,
        lookup_id=lookup_id,
        pin=True,
        explicit_spans=declaration,
    ) == len(tokens)

    ordinary_batched_unpin = storage.batched_unpin
    first_unpin_entered = threading.Event()
    release_first_unpin = threading.Event()
    backend_calls = 0

    def blocked_first_unpin(keys, locations):
        nonlocal backend_calls
        backend_calls += 1
        if backend_calls == 1:
            first_unpin_entered.set()
            assert release_first_unpin.wait(timeout=5)
        ordinary_batched_unpin(keys, locations)

    monkeypatch.setattr(storage, "batched_unpin", blocked_first_unpin)
    errors: list[BaseException] = []

    def unpin() -> None:
        try:
            engine.lookup_unpin(lookup_id)
        except BaseException as error:
            errors.append(error)

    normal_thread = threading.Thread(target=unpin)
    abort_thread = threading.Thread(target=unpin)
    normal_thread.start()
    assert first_unpin_entered.wait(timeout=2)
    abort_thread.start()
    assert abort_thread.is_alive()
    release_first_unpin.set()
    normal_thread.join(timeout=2)
    abort_thread.join(timeout=2)

    assert not normal_thread.is_alive()
    assert not abort_thread.is_alive()
    assert errors == []
    assert backend_calls == len(memory_objs)
    assert lookup_id not in engine.lookup_pins
    assert all(memory_obj.pin_count == 0 for memory_obj in memory_objs)
    assert all(memory_obj.unpin_count == 1 for memory_obj in memory_objs)


def test_same_generation_replay_revalidates_worker_pin_under_lock(
    monkeypatch,
) -> None:
    engine, storage, _ = _engine()
    cast(Any, engine).config = SimpleNamespace(
        enable_blending=True,
        get_extra_config_value=lambda key, default: default,
    )
    tokens = list(range(4))
    declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(2, 4)),
    )
    _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)

    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine
    server.enable_blending = True
    transport = _LoopbackLookupTransport(server)
    client = LMCacheLookupClient.__new__(LMCacheLookupClient)
    client.config = SimpleNamespace(
        use_layerwise=True,
        enable_async_loading=False,
    )
    client.enable_blending = True
    client.transport = transport  # type: ignore[assignment]
    client.reqs_status = {}
    client.reqs_fingerprints = {}
    client.reqs_generations = {}
    client.pending_cancel_ids = set()
    lookup_id = "replay-vs-unpin"
    assert client.lookup(tokens, lookup_id, explicit_spans=declaration) == 4
    lookup_message = transport.messages[0]
    ordinary_lookup = engine.lookup
    replay_engine_lookups = 0

    def count_replay_engine_lookup(*args, **kwargs):
        nonlocal replay_engine_lookups
        replay_engine_lookups += 1
        return ordinary_lookup(*args, **kwargs)

    monkeypatch.setattr(engine, "lookup", count_replay_engine_lookup)

    ordinary_batched_unpin = storage.batched_unpin
    unpin_entered = threading.Event()
    release_unpin = threading.Event()

    def blocked_first_unpin(keys, locations):
        if not unpin_entered.is_set():
            unpin_entered.set()
            assert release_unpin.wait(timeout=5)
        ordinary_batched_unpin(keys, locations)

    monkeypatch.setattr(storage, "batched_unpin", blocked_first_unpin)
    unpin_thread = threading.Thread(target=engine.lookup_unpin, args=(lookup_id,))
    replay_results: list[bytes] = []
    replay_thread = threading.Thread(
        target=lambda: replay_results.append(
            server._process_request_frames(lookup_message)
        )
    )
    unpin_thread.start()
    assert unpin_entered.wait(timeout=2)
    replay_thread.start()
    assert replay_thread.is_alive()

    release_unpin.set()
    unpin_thread.join(timeout=2)
    replay_thread.join(timeout=2)
    assert not unpin_thread.is_alive()
    assert not replay_thread.is_alive()
    assert replay_results[0].endswith((0).to_bytes(4, "big"))
    assert replay_engine_lookups == 0
    assert lookup_id not in engine.lookup_pins
    assert all(memory_obj.pin_count == 0 for memory_obj in storage.objects.values())


def test_explicit_store_retrieve_fake_round_trip(monkeypatch) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine()
    tokens = list(range(10))
    declaration = ExplicitTokenSpans(
        10,
        TokenSpan(0, 10),
        (TokenSpan(2, 4), TokenSpan(6, 8)),
    )

    store_yields = list(engine.store_layer(tokens, explicit_spans=declaration))
    retrieve_yields = list(engine.retrieve_layer(tokens, explicit_spans=declaration))

    assert len(store_yields) == engine.num_layers + 1
    assert len(retrieve_yields) == engine.num_layers + 2
    assert connector.stored_spans == [(2, 4), (6, 8)]
    assert connector.loaded_spans == connector.stored_spans
    assert connector.buffer_span == TokenSpan(0, 10)
    assert connector.loaded_payloads == [
        [(0, 2, 4), (0, 6, 8)],
        [(1, 2, 4), (1, 6, 8)],
    ]
    assert torch.equal(retrieve_yields[0], torch.tensor(4))
    assert torch.equal(
        retrieve_yields[-1],
        torch.tensor(
            [False, False, True, True, False, False, True, True, False, False]
        ),
    )
    assert len(storage.objects) == len(declaration.spans) * engine.num_layers
    assert all(memory_obj.ref_count == 1 for memory_obj in storage.objects.values())
    assert all(memory_obj.pin_count == 0 for memory_obj in storage.objects.values())


def test_explicit_store_close_at_prime_releases_every_allocation(monkeypatch) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine()
    tokens = list(range(10))
    declaration = ExplicitTokenSpans(
        10,
        TokenSpan(0, 10),
        (TokenSpan(2, 4), TokenSpan(6, 8)),
    )

    storer = engine.store_layer(tokens, explicit_spans=declaration)
    next(storer)
    assert len(storage.allocations) == engine.num_layers * len(declaration.spans)

    storer.close()

    assert connector.store_closed
    assert storage.objects == {}
    assert all(memory_obj.ref_count == 0 for memory_obj in storage.allocations)
    assert all(memory_obj.ref_down_count == 1 for memory_obj in storage.allocations)


def test_explicit_store_close_failures_remain_retry_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, _ = _engine()
    tokens = list(range(10))
    declaration = ExplicitTokenSpans(
        10,
        TokenSpan(0, 10),
        (TokenSpan(2, 4), TokenSpan(6, 8)),
    )
    events: list[str] = []

    class _RetrySyncStream:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def wait_stream(self, stream: object) -> None:
            del stream

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            if self.synchronize_calls == 1:
                events.append("sync-failed")
                raise InjectedCleanupFailure("injected stream sync failure")
            events.append("sync-ok")

    class _TrackedCPUMemory:
        def __init__(
            self,
            tensor: torch.Tensor,
            label: str,
            failures: int = 0,
        ) -> None:
            self.tensor = tensor
            self.metadata = SimpleNamespace(fmt=MemoryFormat.KV_2TD)
            self.label = label
            self.ref_count = 1
            self.ref_down_count = 0
            self.release_attempts = 0
            self.release_failures = failures

        def get_size(self) -> int:
            return self.tensor.numel() * self.tensor.element_size()

        def ref_count_down(self) -> None:
            self.release_attempts += 1
            if self.release_failures:
                self.release_failures -= 1
                events.append(f"{self.label}-failed")
                raise InjectedCleanupFailure("injected memory release failure")
            events.append(f"{self.label}-ok")
            self.ref_down_count += 1
            self.ref_count -= 1

    class _TrackedGPUBuffer(_TensorMemory):
        def ref_count_down(self) -> None:
            events.append("tmp-refdown")
            super().ref_count_down()

    class _TrackedGPUAllocator:
        def __init__(self) -> None:
            self.allocations: list[_TrackedGPUBuffer] = []

        def allocate(
            self,
            shape: torch.Size,
            dtype: torch.dtype,
            fmt: MemoryFormat,
        ) -> MemoryObj:
            assert fmt == MemoryFormat.KV_2TD
            memory_obj = _TrackedGPUBuffer(torch.full(shape, -1.0, dtype=dtype))
            self.allocations.append(memory_obj)
            return cast(MemoryObj, memory_obj)

    def retryable_allocate(
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        **kwargs: Any,
    ) -> list[MemoryObj]:
        del kwargs
        allocated = [
            _TrackedCPUMemory(
                torch.full(shape, -1.0, dtype=dtype),
                f"cpu-{len(storage.allocations) + layer_id}",
                failures=int(not storage.allocations and layer_id == 0),
            )
            for layer_id in range(batch_size)
        ]
        storage.allocations.extend(cast(list[_Memory], allocated))
        return cast(list[MemoryObj], allocated)

    monkeypatch.setattr(storage, "batched_allocate", retryable_allocate)
    connector = VLLMBufferLayerwiseGPUConnector.__new__(VLLMBufferLayerwiseGPUConnector)
    connector_state = cast(Any, connector)
    connector_state.num_layers = engine.num_layers
    connector_state.hidden_dim_size = 1
    connector_state.dtype = torch.float32
    connector_state.engine_kv_format = "fake-kv-format"
    connector_state.cache_positions = True
    connector_state.kvcaches = [
        torch.zeros((2, len(tokens), 1)) for _ in range(engine.num_layers)
    ]
    connector_state.initialize_kvcaches_ptr = lambda **kwargs: None
    connector_state._lazy_initialize_buffer = lambda kvcaches: None
    store_stream = _RetrySyncStream()
    connector_state.store_stream = store_stream
    allocator = _TrackedGPUAllocator()
    connector_state.gpu_buffer_allocator = allocator
    cast(Any, engine).gpu_connector = connector

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        gpu_connector_module.lmc_ops,
        "single_layer_kv_transfer",
        lambda *args, **kwargs: events.append("transfer"),
    )

    storer = engine.store_layer(
        tokens,
        explicit_spans=declaration,
        slot_mapping=torch.arange(len(tokens)),
    )
    next(storer)
    with pytest.raises(InjectedCleanupFailure, match="stream sync failure"):
        storer.close()

    allocations = cast(list[_TrackedCPUMemory], storage.allocations)
    assert len(allocations) == engine.num_layers * len(declaration.spans)
    assert store_stream.synchronize_calls == 1
    assert len(allocator.allocations) == 1
    assert allocator.allocations[0].release_count == 0
    assert [obj.release_attempts for obj in allocations] == [0, 0, 0, 0]
    assert [obj.ref_count for obj in allocations] == [1, 1, 1, 1]
    assert events == ["transfer", "sync-failed"]

    with pytest.raises(InjectedCleanupFailure, match="memory release failure"):
        engine.drain_pending_store_cleanups()

    assert store_stream.synchronize_calls == 2
    assert allocator.allocations[0].release_count == 1
    assert [obj.release_attempts for obj in allocations] == [1, 1, 1, 1]
    assert [obj.ref_count for obj in allocations] == [1, 0, 0, 0]
    assert events == [
        "transfer",
        "sync-failed",
        "sync-ok",
        "tmp-refdown",
        "cpu-0-failed",
        "cpu-1-ok",
        "cpu-2-ok",
        "cpu-3-ok",
    ]

    engine.drain_pending_store_cleanups()
    assert [obj.release_attempts for obj in allocations] == [2, 1, 1, 1]
    assert [obj.ref_down_count for obj in allocations] == [1, 1, 1, 1]
    assert [obj.ref_count for obj in allocations] == [0, 0, 0, 0]
    assert events[-1] == "cpu-0-ok"

    engine.drain_pending_store_cleanups()
    assert store_stream.synchronize_calls == 2
    assert allocator.allocations[0].release_count == 1
    assert [obj.release_attempts for obj in allocations] == [2, 1, 1, 1]


def test_explicit_store_partial_close_never_drains_future_layer_and_retries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine()
    tokens = list(range(10))
    declaration = ExplicitTokenSpans(
        10,
        TokenSpan(0, 10),
        (TokenSpan(2, 4), TokenSpan(6, 8)),
    )

    storer = engine.store_layer(tokens, explicit_spans=declaration)
    next(storer)
    next(storer)
    first_attempt = storage.allocations.copy()
    storer.close()

    assert connector.store_closed
    assert len(storage.objects) == len(declaration.spans)
    assert [memory_obj.payload for memory_obj in first_attempt] == [
        (0, 2, 4),
        None,
        (0, 6, 8),
        None,
    ]
    assert [memory_obj.ref_count for memory_obj in first_attempt] == [1, 0, 1, 0]
    assert [memory_obj.ref_down_count for memory_obj in first_attempt] == [0, 1, 0, 1]
    assert engine.lookup(tokens=tokens, explicit_spans=declaration) == 0

    connector.store_closed = False
    list(engine.store_layer(tokens, explicit_spans=declaration))

    assert connector.store_closed
    assert engine.lookup(tokens=tokens, explicit_spans=declaration) == 8
    assert all(memory_obj.ref_count == 0 for memory_obj in first_attempt)
    assert all(memory_obj.ref_down_count == 1 for memory_obj in first_attempt)


def test_explicit_store_partial_allocation_failure_releases_owned_refs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine(num_layers=3)
    tokens = list(range(10))
    declaration = ExplicitTokenSpans(
        10,
        TokenSpan(0, 10),
        (TokenSpan(2, 4), TokenSpan(6, 8)),
    )
    ordinary_allocate = storage.batched_allocate
    allocation_calls = 0

    def fail_second_span_allocation(*args: Any, **kwargs: Any) -> list[MemoryObj]:
        nonlocal allocation_calls
        allocation_calls += 1
        if allocation_calls == 2:
            raise MemoryError("injected second-span allocation failure")
        return ordinary_allocate(*args, **kwargs)

    monkeypatch.setattr(storage, "batched_allocate", fail_second_span_allocation)

    with pytest.raises(MemoryError, match="second-span allocation failure"):
        list(engine.store_layer(tokens, explicit_spans=declaration))

    assert allocation_calls == 2
    assert len(storage.allocations) == engine.num_layers
    assert connector.stored_spans == []
    assert not connector.store_closed
    assert storage.objects == {}
    assert [memory_obj.ref_down_count for memory_obj in storage.allocations] == [
        1,
        1,
        1,
    ]
    assert all(memory_obj.ref_count == 0 for memory_obj in storage.allocations)


@pytest.mark.parametrize("failing_put_call", [1, 2])
def test_explicit_store_batched_put_failure_stops_at_owned_layer_and_releases_once(
    monkeypatch: pytest.MonkeyPatch,
    failing_put_call: int,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine(num_layers=3)
    tokens = list(range(10))
    declaration = ExplicitTokenSpans(
        10,
        TokenSpan(0, 10),
        (TokenSpan(2, 4), TokenSpan(6, 8)),
    )
    put_calls = 0

    def consume_then_fail(
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        **kwargs: Any,
    ) -> None:
        nonlocal put_calls
        del keys
        ownership_receipt = kwargs.pop("ownership_receipt")
        ownership_receipt._adopt()
        del kwargs
        put_calls += 1
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()
        if put_calls == failing_put_call:
            raise RuntimeError(f"injected batched_put failure {put_calls}")

    monkeypatch.setattr(storage, "batched_put", consume_then_fail)

    with pytest.raises(RuntimeError, match="injected batched_put failure"):
        list(engine.store_layer(tokens, explicit_spans=declaration))

    assert put_calls == failing_put_call
    assert connector.store_closed
    assert storage.objects == {}
    assert len(storage.allocations) == engine.num_layers * len(declaration.spans)
    for span_index, span in enumerate(declaration.spans):
        for layer_id in range(engine.num_layers):
            memory_obj = storage.allocations[span_index * engine.num_layers + layer_id]
            expected_payload = (
                (layer_id, span.start, span.end)
                if layer_id < failing_put_call
                else None
            )
            assert memory_obj.payload == expected_payload
            assert memory_obj.ref_down_count == 1
            assert memory_obj.ref_count == 0


def test_explicit_store_storage_manager_submission_failure_consumes_refs_once() -> None:
    class _FailingBackend:
        def __init__(self) -> None:
            self.submit_calls = 0

        def get_allocator_backend(self) -> "_FailingBackend":
            return self

        def batched_submit_put_task(
            self,
            keys: list[CacheEngineKey],
            memory_objs: list[MemoryObj],
            **kwargs: Any,
        ) -> None:
            del keys, memory_objs, kwargs
            self.submit_calls += 1
            raise RuntimeError("injected backend submission failure")

    engine, _, _ = _engine()
    tokens = [0, 1]
    declaration = ExplicitTokenSpans(
        2,
        TokenSpan(0, 2),
        (TokenSpan(0, 2),),
    )
    keys = _keys_for_span(engine, tokens, declaration, 0)
    memory_objs = [_Memory() for _ in keys]
    backend = _FailingBackend()
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state.allocator_backend = backend
    state.storage_backends = {"FailingBackend": backend}
    state._bypass_lock = threading.RLock()
    state._bypassed_backends = set()
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}

    with pytest.raises(RuntimeError, match="backend submission failure"):
        manager.batched_put(keys, cast(list[MemoryObj], memory_objs))

    assert backend.submit_calls == 1
    assert [memory_obj.ref_down_count for memory_obj in memory_objs] == [1, 1]
    assert [memory_obj.ref_count for memory_obj in memory_objs] == [0, 0]


def test_storage_manager_refdown_failure_is_exhaustive_and_retry_owned() -> None:
    class _AcceptingBackend:
        def __init__(self) -> None:
            self.submit_calls = 0

        def get_allocator_backend(self) -> "_AcceptingBackend":
            return self

        def batched_submit_put_task(
            self,
            keys: list[CacheEngineKey],
            memory_objs: list[MemoryObj],
            **kwargs: Any,
        ) -> None:
            del keys, memory_objs, kwargs
            self.submit_calls += 1

    engine, _, _ = _engine()
    tokens = [0, 1]
    declaration = ExplicitTokenSpans(
        2,
        TokenSpan(0, 2),
        (TokenSpan(0, 2),),
    )
    keys = _keys_for_span(engine, tokens, declaration, 0)
    memory_objs = [_RetryCleanupMemory(1), _RetryCleanupMemory()]
    backend = _AcceptingBackend()
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state.allocator_backend = backend
    state.storage_backends = {"AcceptingBackend": backend}
    state._bypass_lock = threading.RLock()
    state._bypassed_backends = set()
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}

    with pytest.raises(InjectedCleanupFailure, match="memory release failure"):
        manager.batched_put(keys, cast(list[MemoryObj], memory_objs))

    assert backend.submit_calls == 1
    assert [obj.release_attempts for obj in memory_objs] == [1, 1]
    assert [obj.ref_count for obj in memory_objs] == [1, 0]

    manager.drain_pending_put_cleanups()

    assert [obj.release_attempts for obj in memory_objs] == [2, 1]
    assert [obj.ref_down_count for obj in memory_objs] == [1, 1]
    assert [obj.ref_count for obj in memory_objs] == [0, 0]

    manager.drain_pending_put_cleanups()
    assert [obj.release_attempts for obj in memory_objs] == [2, 1]


@pytest.mark.parametrize("failure_point", ["partial-copy", "persistent-sync"])
def test_storage_manager_copy_fence_retains_refs_until_synchronized(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    events: list[str] = []

    class _CopyTensor:
        def __init__(self, label: str) -> None:
            self.label = label

        def copy_(self, source: object, non_blocking: bool) -> None:
            del source
            assert non_blocking is True
            events.append(f"{self.label}-copy")
            if failure_point == "partial-copy" and self.label == "target-1":
                raise InjectedCleanupFailure("injected copy stage failure")

    class _CopyOwnedMemory:
        def __init__(
            self,
            label: str,
            tensor: object,
        ) -> None:
            self.label = label
            self.tensor = tensor
            self.meta = SimpleNamespace(fmt=MemoryFormat.KV_2TD)
            self.ref_count = 1
            self.ref_down_count = 0
            self.release_attempts = 0

        def get_shape(self) -> torch.Size:
            return torch.Size([1])

        def get_dtype(self) -> torch.dtype:
            return torch.float32

        def ref_count_down(self) -> None:
            self.release_attempts += 1
            events.append(f"{self.label}-released")
            self.ref_down_count += 1
            self.ref_count -= 1

    class _SourceAllocator:
        pass

    class _TargetAllocator:
        def __init__(self) -> None:
            self.allocations: list[_CopyOwnedMemory] = []

        def contains(self, key: CacheEngineKey) -> bool:
            del key
            return False

        def allocate(self, *args: Any, **kwargs: Any) -> MemoryObj:
            del args, kwargs
            index = len(self.allocations)
            memory_obj = _CopyOwnedMemory(
                f"target-{index}",
                _CopyTensor(f"target-{index}"),
            )
            self.allocations.append(memory_obj)
            return cast(MemoryObj, memory_obj)

    class _CopyBackend:
        def __init__(self, allocator: _TargetAllocator) -> None:
            self.allocator = allocator
            self.submit_calls = 0

        def get_allocator_backend(self) -> _TargetAllocator:
            return self.allocator

        def batched_submit_put_task(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.submit_calls += 1

    class _CopyStream:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            events.append("stream-synchronize")
            if failure_point == "persistent-sync" and self.synchronize_calls <= 3:
                raise InjectedCleanupFailure("injected copy stream sync failure")

    engine, _, _ = _engine()
    tokens = [0, 1]
    declaration = ExplicitTokenSpans(
        2,
        TokenSpan(0, 2),
        (TokenSpan(0, 2),),
    )
    keys = _keys_for_span(engine, tokens, declaration, 0)
    assert len(keys) == 2
    source_objs = [
        _CopyOwnedMemory(f"source-{index}", torch.zeros(1))
        for index in range(len(keys))
    ]
    target_allocator = _TargetAllocator()
    backend = _CopyBackend(target_allocator)
    stream = _CopyStream()
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state.allocator_backend = _SourceAllocator()
    state.storage_backends = {"CopyBackend": backend}
    state._bypass_lock = threading.RLock()
    state._bypassed_backends = set()
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}
    state.internal_copy_stream = stream
    monkeypatch.setattr(
        storage_manager_module.torch_dev,
        "stream",
        lambda copy_stream: nullcontext(),
    )

    expected_error = "copy stage failure|copy stream sync failure"
    with pytest.raises(InjectedCleanupFailure, match=expected_error):
        manager.batched_put(keys, cast(list[MemoryObj], source_objs))

    assert len(target_allocator.allocations) == len(keys)
    assert backend.submit_calls == 0
    all_objs = [*source_objs, *target_allocator.allocations]

    if failure_point == "partial-copy":
        assert stream.synchronize_calls == 1
        assert [obj.release_attempts for obj in all_objs] == [1, 1, 1, 1]
        assert [obj.ref_down_count for obj in all_objs] == [1, 1, 1, 1]
        assert [obj.ref_count for obj in all_objs] == [0, 0, 0, 0]
        assert events == [
            "target-0-copy",
            "target-1-copy",
            "stream-synchronize",
            "source-0-released",
            "source-1-released",
            "target-0-released",
            "target-1-released",
        ]
        assert manager.pending_put_cleanup_count() == 0
    else:
        # The helper's synchronization and the finally-path retry both fail.
        # No source or destination reference may be released without proof
        # that the asynchronous copies completed.
        assert stream.synchronize_calls == 2
        assert [obj.release_attempts for obj in all_objs] == [0, 0, 0, 0]
        assert [obj.ref_count for obj in all_objs] == [1, 1, 1, 1]
        assert manager.pending_put_cleanup_count() == 1

        with pytest.raises(InjectedCleanupFailure, match="copy stream sync failure"):
            manager.drain_pending_put_cleanups()
        assert stream.synchronize_calls == 3
        assert [obj.release_attempts for obj in all_objs] == [0, 0, 0, 0]
        assert [obj.ref_count for obj in all_objs] == [1, 1, 1, 1]
        assert manager.pending_put_cleanup_count() == 1

        manager.drain_pending_put_cleanups()
        assert stream.synchronize_calls == 4
        assert [obj.release_attempts for obj in all_objs] == [1, 1, 1, 1]
        assert [obj.ref_down_count for obj in all_objs] == [1, 1, 1, 1]
        assert [obj.ref_count for obj in all_objs] == [0, 0, 0, 0]
        assert manager.pending_put_cleanup_count() == 0

    manager.drain_pending_put_cleanups()
    assert [obj.release_attempts for obj in all_objs] == [1, 1, 1, 1]


@pytest.mark.parametrize("race_order", ["put-wins", "close-wins"])
def test_storage_manager_close_and_put_are_atomically_fenced(
    race_order: str,
) -> None:
    class _RacingBackend:
        def __init__(self) -> None:
            self.submit_entered = threading.Event()
            self.submit_release = threading.Event()
            self.close_entered = threading.Event()
            self.close_release = threading.Event()
            self.block_submit = race_order == "put-wins"
            self.block_close = race_order == "close-wins"
            self.submit_calls = 0
            self.close_calls = 0

        def get_allocator_backend(self) -> "_RacingBackend":
            return self

        def batched_submit_put_task(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.submit_calls += 1
            self.submit_entered.set()
            if self.block_submit and not self.submit_release.wait(timeout=2.0):
                raise RuntimeError("timed out waiting to release put")

        def close(self) -> None:
            self.close_calls += 1
            self.close_entered.set()
            if self.block_close and not self.close_release.wait(timeout=2.0):
                raise RuntimeError("timed out waiting to release close")

    class _StoppedLoop:
        def is_running(self) -> bool:
            return False

        def is_closed(self) -> bool:
            return True

    class _StoppedThread:
        def is_alive(self) -> bool:
            return False

    engine, _, _ = _engine()
    tokens = [0, 1]
    declaration = ExplicitTokenSpans(
        2,
        TokenSpan(0, 2),
        (TokenSpan(0, 2),),
    )
    keys = _keys_for_span(engine, tokens, declaration, 0)
    backend = _RacingBackend()
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state.allocator_backend = backend
    state.storage_backends = OrderedDict([("_RacingBackend", backend)])
    state._bypass_lock = threading.RLock()
    state._bypassed_backends = set()
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}
    state._put_cleanup_closed = False
    state.loop = _StoppedLoop()
    state.thread = _StoppedThread()

    if race_order == "put-wins":
        memory_objs = [_RetryCleanupMemory(), _RetryCleanupMemory()]
        put_errors: list[BaseException] = []

        def run_put() -> None:
            try:
                manager.batched_put(keys, cast(list[MemoryObj], memory_objs))
            except BaseException as error:
                put_errors.append(error)

        put_thread = threading.Thread(target=run_put)
        put_thread.start()
        assert backend.submit_entered.wait(timeout=2.0)

        with pytest.raises(RuntimeError, match="active batched-put owner"):
            manager.close()
        assert backend.close_calls == 0

        rejected_objs = [_RetryCleanupMemory(), _RetryCleanupMemory()]
        with pytest.raises(StorageManagerClosedError, match="caller retains"):
            manager.batched_put(keys, cast(list[MemoryObj], rejected_objs))
        assert backend.submit_calls == 1
        assert [obj.ref_down_count for obj in rejected_objs] == [0, 0]
        assert manager.pending_put_cleanup_count() == 0
        assert manager._put_cleanup_owner_count() == 1
        for memory_obj in rejected_objs:
            memory_obj.ref_count_down()

        backend.submit_release.set()
        put_thread.join(timeout=2.0)
        assert not put_thread.is_alive()
        assert put_errors == []
        assert [obj.ref_down_count for obj in memory_objs] == [1, 1]
        assert manager.pending_put_cleanup_count() == 0

        manager.close()
        assert backend.close_calls == 1
    else:
        close_errors: list[BaseException] = []

        def run_close() -> None:
            try:
                manager.close()
            except BaseException as error:
                close_errors.append(error)

        close_thread = threading.Thread(target=run_close)
        close_thread.start()
        assert backend.close_entered.wait(timeout=2.0)

        rejected_objs = [_RetryCleanupMemory(), _RetryCleanupMemory()]
        rejected_put_errors: list[BaseException] = []
        put_entered = threading.Event()

        def run_rejected_put() -> None:
            put_entered.set()
            try:
                manager.batched_put(keys, cast(list[MemoryObj], rejected_objs))
            except BaseException as error:
                rejected_put_errors.append(error)

        put_thread = threading.Thread(target=run_rejected_put)
        put_thread.start()
        assert put_entered.wait(timeout=2.0)
        put_thread.join(timeout=0.5)
        assert not put_thread.is_alive()
        assert backend.submit_calls == 0
        assert [obj.ref_down_count for obj in rejected_objs] == [0, 0]
        assert manager.pending_put_cleanup_count() == 0
        assert manager._put_cleanup_owner_count() == 0
        assert len(rejected_put_errors) == 1
        assert isinstance(rejected_put_errors[0], StorageManagerClosedError)
        for memory_obj in rejected_objs:
            memory_obj.ref_count_down()

        backend.close_release.set()
        close_thread.join(timeout=2.0)
        assert not close_thread.is_alive()
        assert close_errors == []
        assert [obj.ref_down_count for obj in rejected_objs] == [1, 1]
        assert manager.pending_put_cleanup_count() == 0
        assert backend.close_calls == 1


def test_storage_manager_close_propagates_and_retries_backend_failure() -> None:
    class _RetryCloseBackend:
        def __init__(self, close_failures: int) -> None:
            self.close_failures = close_failures
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls <= self.close_failures:
                raise RuntimeError("injected storage backend close failure")

        def close_failure_is_retry_safe(self, error: BaseException) -> bool:
            return isinstance(error, RuntimeError)

    class _StoppedLoop:
        def is_running(self) -> bool:
            return False

        def is_closed(self) -> bool:
            return True

    class _StoppedThread:
        def is_alive(self) -> bool:
            return False

    failing = _RetryCloseBackend(close_failures=1)
    successful = _RetryCloseBackend(close_failures=0)
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state.storage_backends = OrderedDict(
        [("failing", failing), ("successful", successful)]
    )
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}
    state._put_cleanup_closed = False
    state.loop = _StoppedLoop()
    state.thread = _StoppedThread()

    with pytest.raises(RuntimeError, match="storage backend close failure"):
        manager.close()

    assert failing.close_calls == 1
    assert successful.close_calls == 1
    assert id(failing) not in state._closed_storage_backend_ids
    assert id(successful) in state._closed_storage_backend_ids

    manager.close()

    assert failing.close_calls == 2
    assert successful.close_calls == 1
    assert id(failing) in state._closed_storage_backend_ids


def test_storage_manager_concurrent_closes_are_serialized() -> None:
    class _BlockingBackend:
        def __init__(self) -> None:
            self.close_calls = 0
            self.close_entered = threading.Event()
            self.close_release = threading.Event()

        def close(self) -> None:
            self.close_calls += 1
            self.close_entered.set()
            if not self.close_release.wait(timeout=2.0):
                raise RuntimeError("timed out waiting to release close")

    class _StoppedLoop:
        def is_running(self) -> bool:
            return False

        def is_closed(self) -> bool:
            return True

    class _StoppedThread:
        def is_alive(self) -> bool:
            return False

    backend = _BlockingBackend()
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state.storage_backends = OrderedDict([("blocking", backend)])
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}
    state._put_cleanup_closed = False
    state.loop = _StoppedLoop()
    state.thread = _StoppedThread()

    close_errors: list[BaseException] = []

    def run_close() -> None:
        try:
            manager.close()
        except BaseException as error:
            close_errors.append(error)

    first = threading.Thread(target=run_close)
    first.start()
    assert backend.close_entered.wait(timeout=2.0)

    second = threading.Thread(target=run_close)
    second.start()
    assert second.is_alive()
    assert backend.close_calls == 1

    backend.close_release.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert close_errors == []
    assert backend.close_calls == 1


def test_storage_manager_interrupted_begin_releases_exact_teardown_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Backend:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class _StoppedLoop:
        def is_running(self) -> bool:
            return False

        def is_closed(self) -> bool:
            return True

    class _StoppedThread:
        def is_alive(self) -> bool:
            return False

    backend = _Backend()
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state.storage_backends = OrderedDict([("backend", backend)])
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}
    state._put_cleanup_closed = False
    state.loop = _StoppedLoop()
    state.thread = _StoppedThread()

    original_begin = manager._begin_put_cleanup_teardown

    def begin_then_interrupt(teardown_token: object) -> None:
        original_begin(teardown_token)
        raise KeyboardInterrupt("injected post-begin interrupt")

    monkeypatch.setattr(
        manager,
        "_begin_put_cleanup_teardown",
        begin_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt, match="post-begin interrupt"):
        manager.close()

    assert not state._put_cleanup_teardown_active
    assert state._put_cleanup_teardown_owner is None
    assert state._put_cleanup_teardown_token is None
    assert backend.close_calls == 0

    monkeypatch.setattr(
        manager,
        "_begin_put_cleanup_teardown",
        original_begin,
    )
    manager.close()
    assert backend.close_calls == 1


def test_storage_manager_interrupted_token_clear_still_notifies_waiters() -> None:
    class _InterruptTokenClearManager(StorageManager):
        def __setattr__(self, name: str, value: Any) -> None:
            object.__setattr__(self, name, value)
            if (
                name == "_put_cleanup_teardown_token"
                and value is None
                and getattr(self, "_interrupt_token_clear", False)
            ):
                object.__setattr__(self, "_interrupt_token_clear", False)
                raise KeyboardInterrupt("injected token-clear interrupt")

    class _RecordingCondition:
        def __init__(self, lock: Any) -> None:
            self.lock = lock
            self.notify_calls = 0

        def __enter__(self) -> "_RecordingCondition":
            self.lock.acquire()
            return self

        def __exit__(self, *_args: object) -> None:
            self.lock.release()

        def wait(self) -> None:
            raise AssertionError("finish unexpectedly waited")

        def notify_all(self) -> None:
            self.notify_calls += 1

    manager = _InterruptTokenClearManager.__new__(_InterruptTokenClearManager)
    state = cast(Any, manager)
    state._interrupt_token_clear = False
    state._put_cleanup_lock = threading.RLock()
    condition = _RecordingCondition(state._put_cleanup_lock)
    state._put_cleanup_condition = condition
    state._put_cleanup_owners = {}
    state._put_cleanup_closed = True
    state._put_cleanup_teardown_active = True
    state._put_cleanup_teardown_owner = threading.get_ident()
    teardown_token = object()
    state._put_cleanup_teardown_token = teardown_token
    state._interrupt_token_clear = True

    with pytest.raises(KeyboardInterrupt, match="token-clear interrupt"):
        manager._finish_put_cleanup_teardown(teardown_token)

    assert state._put_cleanup_teardown_token is None
    assert condition.notify_calls == 1

    retry_token = object()
    manager._begin_put_cleanup_teardown(retry_token)
    manager._finish_put_cleanup_teardown(retry_token)
    assert state._put_cleanup_teardown_token is None
    assert condition.notify_calls == 2


def test_storage_manager_reentrant_close_cannot_release_outer_token() -> None:
    class _StoppedLoop:
        def is_running(self) -> bool:
            return False

        def is_closed(self) -> bool:
            return True

    class _StoppedThread:
        def is_alive(self) -> bool:
            return False

    manager = StorageManager.__new__(StorageManager)
    observed_errors: list[BaseException] = []
    observed_outer_active: list[bool] = []

    class _ReentrantBackend:
        def close(self) -> None:
            try:
                manager.close()
            except BaseException as error:
                observed_errors.append(error)
            state = cast(Any, manager)
            observed_outer_active.append(
                state._put_cleanup_teardown_token is not None
                and state._put_cleanup_teardown_active
                and state._put_cleanup_teardown_owner != threading.get_ident()
                and threading.get_ident() in state._backend_close_worker_ids
            )

    backend = _ReentrantBackend()
    state = cast(Any, manager)
    state.storage_backends = OrderedDict([("reentrant", backend)])
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}
    state._put_cleanup_closed = False
    state.loop = _StoppedLoop()
    state.thread = _StoppedThread()

    manager.close()

    assert len(observed_errors) == 1
    assert isinstance(observed_errors[0], RuntimeError)
    assert "close is reentrant" in str(observed_errors[0])
    assert observed_outer_active == [True]
    assert state._put_cleanup_teardown_token is None
    assert not state._put_cleanup_teardown_active


@pytest.mark.parametrize("interrupt_after_adopt", [False, True])
def test_put_receipt_interrupt_boundary_selects_exact_cleanup_owner(
    interrupt_after_adopt: bool,
) -> None:
    class _InterruptReceipt(PutOwnershipReceipt):
        def _adopt(self) -> None:
            if interrupt_after_adopt:
                super()._adopt()
            raise KeyboardInterrupt("injected receipt-adoption interrupt")

    class _NoopGPU:
        def drain_pending_gpu_cleanups(self) -> None:
            return

        def pending_gpu_cleanup_count(self) -> int:
            return 0

    memory_obj = _RetryCleanupMemory()
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}
    state._put_cleanup_closed = False

    receipt = _InterruptReceipt(cast(list[MemoryObj], [memory_obj]))
    outer_owner = cache_engine_module._StoreCleanupOwner(
        cast(Any, _NoopGPU()),
        manager,
    )
    outer_owner.add_allocated(cast(list[MemoryObj], [memory_obj]))
    outer_owner.attach_put_receipt(receipt)

    with pytest.raises(KeyboardInterrupt, match="receipt-adoption interrupt"):
        manager.batched_put(
            [],
            cast(list[MemoryObj], [memory_obj]),
            ownership_receipt=receipt,
        )

    assert receipt.adopted is interrupt_after_adopt
    assert manager.pending_put_cleanup_count() == 0
    assert manager._put_cleanup_owner_count() == 0
    assert memory_obj.ref_down_count == int(interrupt_after_adopt)

    outer_owner.mark_pending()
    assert outer_owner.drain() is None
    assert outer_owner.complete
    assert memory_obj.release_attempts == 1
    assert memory_obj.ref_down_count == 1
    assert memory_obj.ref_count == 0


def test_pre_adopted_receipt_cannot_steal_fresh_outer_owner() -> None:
    class _NoopGPU:
        def drain_pending_gpu_cleanups(self) -> None:
            return

        def pending_gpu_cleanup_count(self) -> int:
            return 0

    memory_obj = _RetryCleanupMemory()
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}

    receipt = PutOwnershipReceipt(cast(list[MemoryObj], [memory_obj]))
    receipt._adopt()
    outer_owner = cache_engine_module._StoreCleanupOwner(
        cast(Any, _NoopGPU()),
        manager,
    )
    outer_owner.add_allocated(cast(list[MemoryObj], [memory_obj]))

    with pytest.raises(RuntimeError, match="cannot be attached"):
        outer_owner.attach_put_receipt(receipt)

    outer_owner.mark_pending()
    assert outer_owner.drain() is None
    assert outer_owner.complete
    assert memory_obj.release_attempts == 1
    assert memory_obj.ref_down_count == 1
    assert memory_obj.ref_count == 0


def test_storage_manager_backend_close_reentrant_put_is_nonblocking() -> None:
    class _StoppedLoop:
        def is_running(self) -> bool:
            return False

        def is_closed(self) -> bool:
            return True

    class _StoppedThread:
        def is_alive(self) -> bool:
            return False

    memory_obj = _RetryCleanupMemory()
    observed_errors: list[BaseException] = []
    manager = StorageManager.__new__(StorageManager)

    class _ReentrantBackend:
        def close(self) -> None:
            try:
                manager.batched_put([], cast(list[MemoryObj], [memory_obj]))
            except BaseException as error:
                observed_errors.append(error)
            assert memory_obj.ref_down_count == 0
            memory_obj.ref_count_down()

    backend = _ReentrantBackend()
    state = cast(Any, manager)
    state.storage_backends = OrderedDict([("reentrant", backend)])
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}
    state._put_cleanup_closed = False
    state.loop = _StoppedLoop()
    state.thread = _StoppedThread()

    manager.close()

    assert len(observed_errors) == 1
    assert isinstance(observed_errors[0], StorageManagerClosedError)
    assert memory_obj.ref_down_count == 1
    assert memory_obj.ref_count == 0
    assert manager._put_cleanup_owner_count() == 0


def test_storage_manager_backend_retry_keeps_event_loop_running() -> None:
    class _RunningLoop:
        def __init__(self) -> None:
            self.running = True
            self.stop_calls = 0

        def is_running(self) -> bool:
            return self.running

        def is_closed(self) -> bool:
            return not self.running

        def stop(self) -> None:
            self.running = False

        def call_soon_threadsafe(self, callback: Any) -> None:
            self.stop_calls += 1
            callback()

    class _StoppedThread:
        def is_alive(self) -> bool:
            return False

    loop = _RunningLoop()

    class _LoopDependentBackend:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected retryable backend failure")
            if not loop.is_running():
                raise RuntimeError("event loop was stopped before backend retry")

        def close_failure_is_retry_safe(self, error: BaseException) -> bool:
            return isinstance(error, RuntimeError)

    backend = _LoopDependentBackend()
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state.storage_backends = OrderedDict([("loop-dependent", backend)])
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}
    state._put_cleanup_closed = False
    state.loop = loop
    state.thread = _StoppedThread()

    with pytest.raises(RuntimeError, match="retryable backend failure"):
        manager.close()

    assert loop.is_running()
    assert loop.stop_calls == 0

    manager.close()

    assert backend.close_calls == 2
    assert loop.stop_calls == 1
    assert not loop.is_running()


def test_storage_manager_loop_stop_failure_is_truthful_and_retryable() -> None:
    stop_error = InjectedCleanupFailure("injected event-loop stop failure")

    class _RetryStopLoop:
        def __init__(self) -> None:
            self.running = True
            self.stop_calls = 0

        def is_running(self) -> bool:
            return self.running

        def is_closed(self) -> bool:
            return not self.running

        def stop(self) -> None:
            self.running = False

        def call_soon_threadsafe(self, callback: Any) -> None:
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise stop_error
            callback()

    class _StoppedThread:
        def is_alive(self) -> bool:
            return False

    class _Backend:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    loop = _RetryStopLoop()
    backend = _Backend()
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state.storage_backends = OrderedDict([("backend", backend)])
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}
    state._put_cleanup_closed = False
    state.loop = loop
    state.thread = _StoppedThread()

    with pytest.raises(InjectedCleanupFailure) as failure:
        manager.close()
    assert failure.value is stop_error
    assert backend.close_calls == 1
    assert loop.is_running()

    manager.close()

    assert backend.close_calls == 1
    assert loop.stop_calls == 2
    assert not loop.is_running()


def test_storage_manager_live_thread_timeout_is_truthful_and_retryable() -> None:
    class _StoppedLoop:
        def is_running(self) -> bool:
            return False

        def is_closed(self) -> bool:
            return True

    class _RetryThread:
        def __init__(self) -> None:
            self.alive = True
            self.join_calls = 0

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            assert timeout == 10.0
            self.join_calls += 1

    class _Backend:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    thread = _RetryThread()
    backend = _Backend()
    manager = StorageManager.__new__(StorageManager)
    state = cast(Any, manager)
    state.storage_backends = OrderedDict([("backend", backend)])
    state._put_cleanup_lock = threading.RLock()
    state._put_cleanup_owners = {}
    state._put_cleanup_closed = False
    state.loop = _StoppedLoop()
    state.thread = thread

    with pytest.raises(RuntimeError, match="did not terminate within 10s"):
        manager.close()

    assert backend.close_calls == 1
    assert thread.join_calls == 1

    thread.alive = False
    manager.close()

    assert backend.close_calls == 1
    assert thread.join_calls == 1


@pytest.mark.parametrize("started_layers", [1, 2])
def test_explicit_store_gpu_connector_close_does_not_start_future_layers(
    monkeypatch: pytest.MonkeyPatch,
    started_layers: int,
) -> None:
    num_layers = 3
    connector = VLLMBufferLayerwiseGPUConnector.__new__(VLLMBufferLayerwiseGPUConnector)
    state = cast(Any, connector)
    state.num_layers = num_layers
    state.hidden_dim_size = 1
    state.dtype = torch.float32
    state.engine_kv_format = "fake-kv-format"
    state.cache_positions = True
    state.kvcaches = [
        torch.full((2, 2, 1), float(layer_id)) for layer_id in range(num_layers)
    ]
    state.initialize_kvcaches_ptr = lambda **kwargs: None
    state._lazy_initialize_buffer = lambda kvcaches: None

    class _FakeStream:
        def __init__(self) -> None:
            self.wait_calls = 0
            self.synchronize_calls = 0

        def wait_stream(self, stream: object) -> None:
            del stream
            self.wait_calls += 1

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    store_stream = _FakeStream()
    state.store_stream = store_stream
    allocator = _TensorAllocator()
    state.gpu_buffer_allocator = allocator
    transferred_layers: list[int] = []

    def fake_single_layer_transfer(
        target: torch.Tensor,
        source: torch.Tensor,
        slot_mapping: torch.Tensor,
        direction: Any,
        kv_format: Any,
        **kwargs: Any,
    ) -> None:
        del slot_mapping, direction, kv_format, kwargs
        transferred_layers.append(int(source[0, 0, 0].item()))
        target.copy_(source)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        gpu_connector_module.lmc_ops,
        "single_layer_kv_transfer",
        fake_single_layer_transfer,
    )
    layer_memory_objs = [
        [
            cast(
                MemoryObj,
                _TensorMemory(torch.full((2, 2, 1), -1.0)),
            )
        ]
        for _ in range(num_layers)
    ]

    storer = connector.batched_from_gpu(
        layer_memory_objs,
        [0],
        [2],
        slot_mapping=torch.tensor([0, 1]),
    )
    for _ in range(started_layers):
        next(storer)
    storer.close()

    assert transferred_layers == list(range(started_layers))
    assert store_stream.wait_calls == started_layers
    assert store_stream.synchronize_calls == started_layers
    assert len(allocator.allocations) == 1
    assert allocator.allocations[0].release_count == 1
    for layer_id, layer_memory in enumerate(layer_memory_objs):
        tensor_memory = cast(_TensorMemory, layer_memory[0])
        expected = float(layer_id) if layer_id < started_layers else -1.0
        assert torch.all(tensor_memory.tensor == expected)


def test_explicit_store_gpu_connector_transfer_failure_synchronizes_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = VLLMBufferLayerwiseGPUConnector.__new__(VLLMBufferLayerwiseGPUConnector)
    state = cast(Any, connector)
    state.num_layers = 2
    state.hidden_dim_size = 1
    state.dtype = torch.float32
    state.engine_kv_format = "fake-kv-format"
    state.cache_positions = True
    state.kvcaches = [torch.zeros((2, 2, 1)) for _ in range(state.num_layers)]
    state.initialize_kvcaches_ptr = lambda **kwargs: None
    state._lazy_initialize_buffer = lambda kvcaches: None

    class _FakeStream:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def wait_stream(self, stream: object) -> None:
            del stream

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    store_stream = _FakeStream()
    state.store_stream = store_stream
    allocator = _TensorAllocator()
    state.gpu_buffer_allocator = allocator

    def fail_transfer(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("injected transfer submission failure")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        gpu_connector_module.lmc_ops,
        "single_layer_kv_transfer",
        fail_transfer,
    )
    layer_memory_objs = [
        [cast(MemoryObj, _TensorMemory(torch.full((2, 2, 1), -1.0)))]
        for _ in range(state.num_layers)
    ]

    storer = connector.batched_from_gpu(
        layer_memory_objs,
        [0],
        [2],
        slot_mapping=torch.tensor([0, 1]),
    )
    with pytest.raises(RuntimeError, match="transfer submission failure"):
        next(storer)

    assert store_stream.synchronize_calls == 1
    assert len(allocator.allocations) == 1
    assert allocator.allocations[0].release_count == 1
    assert all(
        torch.all(cast(_TensorMemory, layer[0]).tensor == -1.0)
        for layer in layer_memory_objs
    )


def test_gpu_store_sync_failure_retains_buffer_until_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = VLLMBufferLayerwiseGPUConnector.__new__(VLLMBufferLayerwiseGPUConnector)
    state = cast(Any, connector)
    state.num_layers = 2
    state.hidden_dim_size = 1
    state.dtype = torch.float32
    state.engine_kv_format = "fake-kv-format"
    state.cache_positions = True
    state.kvcaches = [torch.zeros((2, 2, 1)) for _ in range(state.num_layers)]
    state.initialize_kvcaches_ptr = lambda **kwargs: None
    state._lazy_initialize_buffer = lambda kvcaches: None

    class _RetrySyncStream:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def wait_stream(self, stream: object) -> None:
            del stream

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            if self.synchronize_calls == 1:
                raise InjectedCleanupFailure("injected stream sync failure")

    store_stream = _RetrySyncStream()
    state.store_stream = store_stream
    allocator = _TensorAllocator()
    state.gpu_buffer_allocator = allocator

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        gpu_connector_module.lmc_ops,
        "single_layer_kv_transfer",
        lambda *args, **kwargs: None,
    )
    layer_memory_objs = [
        [cast(MemoryObj, _TensorMemory(torch.full((2, 2, 1), -1.0)))]
        for _ in range(state.num_layers)
    ]

    storer = connector.batched_from_gpu(
        layer_memory_objs,
        [0],
        [2],
        slot_mapping=torch.tensor([0, 1]),
    )
    next(storer)
    with pytest.raises(InjectedCleanupFailure, match="stream sync failure"):
        storer.close()

    assert store_stream.synchronize_calls == 1
    assert len(allocator.allocations) == 1
    assert allocator.allocations[0].release_count == 0

    connector.drain_pending_gpu_cleanups()

    assert store_stream.synchronize_calls == 2
    assert allocator.allocations[0].release_count == 1

    connector.drain_pending_gpu_cleanups()
    assert store_stream.synchronize_calls == 2
    assert allocator.allocations[0].release_count == 1


def test_gpu_store_pending_sync_preflights_before_new_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = VLLMBufferLayerwiseGPUConnector.__new__(VLLMBufferLayerwiseGPUConnector)
    state = cast(Any, connector)
    state.num_layers = 1
    state.hidden_dim_size = 1
    state.dtype = torch.float32
    state.engine_kv_format = "fake-kv-format"
    state.cache_positions = True
    state.kvcaches = [torch.zeros((2, 2, 1))]
    state.initialize_kvcaches_ptr = lambda **kwargs: None
    state._lazy_initialize_buffer = lambda kvcaches: None

    class _PersistentFailureStream:
        def __init__(self) -> None:
            self.remaining_failures = 2
            self.synchronize_calls = 0

        def wait_stream(self, stream: object) -> None:
            del stream

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            if self.remaining_failures:
                self.remaining_failures -= 1
                raise InjectedCleanupFailure("persistent stream sync failure")

    store_stream = _PersistentFailureStream()
    state.store_stream = store_stream
    allocator = _TensorAllocator()
    state.gpu_buffer_allocator = allocator
    transfer_calls = 0

    def record_transfer(*args: Any, **kwargs: Any) -> None:
        nonlocal transfer_calls
        del args, kwargs
        transfer_calls += 1

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        gpu_connector_module.lmc_ops,
        "single_layer_kv_transfer",
        record_transfer,
    )

    def new_storer() -> Generator[None, None, None]:
        return connector.batched_from_gpu(
            [[cast(MemoryObj, _TensorMemory(torch.zeros((2, 2, 1))))]],
            [0],
            [2],
            slot_mapping=torch.tensor([0, 1]),
        )

    first = new_storer()
    next(first)
    with pytest.raises(InjectedCleanupFailure, match="persistent stream sync failure"):
        first.close()

    assert transfer_calls == 1
    assert len(allocator.allocations) == 1
    assert allocator.allocations[0].release_count == 0

    second = new_storer()
    with pytest.raises(InjectedCleanupFailure, match="persistent stream sync failure"):
        next(second)

    assert store_stream.synchronize_calls == 2
    assert transfer_calls == 1
    assert len(allocator.allocations) == 1
    assert allocator.allocations[0].release_count == 0

    connector.drain_pending_gpu_cleanups()
    assert store_stream.synchronize_calls == 3
    assert transfer_calls == 1
    assert len(allocator.allocations) == 1
    assert allocator.allocations[0].release_count == 1


def test_explicit_lookup_and_retrieve_continue_after_multiple_misses(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine()
    tokens = list(range(10))
    declaration = ExplicitTokenSpans(
        10,
        TokenSpan(0, 10),
        tuple(TokenSpan(start, start + 2) for start in range(0, 10, 2)),
    )
    _populate_span(engine, storage, tokens, declaration, 1)
    _populate_span(engine, storage, tokens, declaration, 3)

    greatest_hit_end = engine.lookup(tokens=tokens, explicit_spans=declaration)
    yields = list(engine.retrieve_layer(tokens, explicit_spans=declaration))

    assert greatest_hit_end == 8
    assert connector.loaded_spans == [(2, 4), (6, 8)]
    assert connector.buffer_span == TokenSpan(0, 10)
    assert torch.equal(
        yields[-1],
        torch.tensor(
            [False, False, True, True, False, False, True, True, False, False]
        ),
    )


def test_explicit_lookup_unpins_partial_layer_miss_and_retains_only_hits() -> None:
    engine, storage, _ = _engine()
    tokens = list(range(6))
    declaration = ExplicitTokenSpans(
        6,
        TokenSpan(0, 6),
        (TokenSpan(0, 2), TokenSpan(2, 4), TokenSpan(4, 6)),
    )
    partial_objects = _populate_span(engine, storage, tokens, declaration, 0)
    del storage.objects[_keys_for_span(engine, tokens, declaration, 0)[1]]
    hit_objects = _populate_span(engine, storage, tokens, declaration, 2)

    result = engine.lookup(
        tokens=tokens,
        explicit_spans=declaration,
        lookup_id="explicit-pins",
        pin=True,
    )

    assert result == 6
    assert partial_objects[0].pin_count == 0
    assert partial_objects[0].unpin_count == 1
    assert [memory_obj.pin_count for memory_obj in hit_objects] == [1, 1]
    assert sum(len(keys) for keys in engine.lookup_pins["explicit-pins"].values()) == 2

    engine.lookup_unpin("explicit-pins")
    assert [memory_obj.pin_count for memory_obj in hit_objects] == [0, 0]
    assert storage.touch_count == 1


@pytest.mark.parametrize("retrieval_owns_pins", [False, True])
def test_pinned_explicit_lookup_survives_retrieve_until_lookup_unpin(
    monkeypatch,
    retrieval_owns_pins: bool,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, _ = _engine(retrieval_owns_pins=retrieval_owns_pins)
    tokens = list(range(6))
    declaration = ExplicitTokenSpans(
        6,
        TokenSpan(0, 6),
        (TokenSpan(0, 2), TokenSpan(4, 6)),
    )
    _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)
    memory_objs = list(storage.objects.values())
    lookup_id = "pinned-explicit-lifecycle"

    assert (
        engine.lookup(
            tokens=tokens,
            explicit_spans=declaration,
            lookup_id=lookup_id,
            pin=True,
        )
        == 6
    )
    assert lookup_id in engine.lookup_pins
    assert all(memory_obj.pin_count == 1 for memory_obj in memory_objs)
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)

    list(engine.retrieve_layer(tokens, explicit_spans=declaration))

    assert all(memory_obj.pin_count == 1 for memory_obj in memory_objs)
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)
    assert all(
        memory_obj.unpin_count == int(retrieval_owns_pins) for memory_obj in memory_objs
    )

    engine.lookup_unpin(lookup_id)

    assert lookup_id not in engine.lookup_pins
    assert all(memory_obj.pin_count == 0 for memory_obj in memory_objs)
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)
    assert all(
        memory_obj.unpin_count == 1 + int(retrieval_owns_pins)
        for memory_obj in memory_objs
    )


@pytest.mark.parametrize("retrieval_owns_pins", [False, True])
@pytest.mark.parametrize("yield_count", [1, 2])
def test_explicit_retrieve_close_at_layer_boundary_drains_pending_task(
    monkeypatch,
    retrieval_owns_pins: bool,
    yield_count: int,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine(retrieval_owns_pins=retrieval_owns_pins)
    tokens = list(range(4))
    declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(2, 4)),
    )
    _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)
    memory_objs = list(storage.objects.values())

    retriever = engine.retrieve_layer(tokens, explicit_spans=declaration)
    for _ in range(yield_count):
        next(retriever)
    retriever.close()

    assert connector.closed
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)
    assert all(memory_obj.pin_count == 0 for memory_obj in memory_objs)
    assert sum(memory_obj.unpin_count for memory_obj in memory_objs) == (
        len(declaration.spans) * yield_count * int(retrieval_owns_pins)
    )


def test_explicit_retrieve_close_waits_for_pending_task_before_release(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine(retrieval_owns_pins=True)
    tokens = list(range(4))
    declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(2, 4)),
    )
    first_layer_objs = _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)
    result_waiting = threading.Event()
    release_result = threading.Event()

    class _BlockingTask:
        def result(self) -> list[MemoryObj]:
            result_waiting.set()
            assert release_result.wait(timeout=5)
            return cast(list[MemoryObj], first_layer_objs)

    def blocked_layerwise_batched_get(keys_layer_major, location):
        assert location == storage.location
        first_keys = keys_layer_major[0]
        assert [storage.objects[key] for key in first_keys] == first_layer_objs
        for memory_obj in first_layer_objs:
            memory_obj.ref_count_up()
            memory_obj.pin()
        yield _BlockingTask()

    cast(Any, storage).layerwise_batched_get = blocked_layerwise_batched_get
    retriever = engine.retrieve_layer(tokens, explicit_spans=declaration)
    next(retriever)

    close_errors: list[BaseException] = []

    def close_retriever() -> None:
        try:
            retriever.close()
        except BaseException as error:
            close_errors.append(error)

    close_thread = threading.Thread(target=close_retriever)
    close_thread.start()
    assert result_waiting.wait(timeout=2)

    # The backend may still write into these objects, so close must retain
    # both ownership references until the task actually completes.
    assert close_thread.is_alive()
    assert all(memory_obj.ref_count == 2 for memory_obj in first_layer_objs)
    assert all(memory_obj.pin_count == 1 for memory_obj in first_layer_objs)

    release_result.set()
    close_thread.join(timeout=2)
    assert not close_thread.is_alive()
    assert close_errors == []
    assert connector.closed
    assert all(memory_obj.ref_count == 1 for memory_obj in first_layer_objs)
    assert all(memory_obj.pin_count == 0 for memory_obj in first_layer_objs)


def test_explicit_retrieve_releases_backend_owned_staging_pins(monkeypatch) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, _ = _engine(retrieval_owns_pins=True)
    tokens = list(range(4))
    declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(2, 4)),
    )
    _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)

    list(engine.retrieve_layer(tokens, explicit_spans=declaration))

    assert all(memory_obj.ref_count == 1 for memory_obj in storage.objects.values())
    assert all(memory_obj.pin_count == 0 for memory_obj in storage.objects.values())
    assert all(memory_obj.unpin_count == 1 for memory_obj in storage.objects.values())


@pytest.mark.parametrize("invalid_result", ["short", "none", "extra"])
def test_explicit_retrieve_rejects_invalid_partial_layer_result(
    monkeypatch,
    invalid_result: str,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine(retrieval_owns_pins=True)
    tokens = list(range(4))
    declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(2, 4)),
    )
    _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)
    tracked_memory_objs = list(storage.objects.values())

    def invalid_layerwise_batched_get(keys_layer_major, location):
        assert location == storage.location
        for layer_id, keys in enumerate(keys_layer_major):
            memory_objs: list[Optional[_Memory]] = [
                storage.objects[key] for key in keys
            ]
            if layer_id == 1:
                if invalid_result == "short":
                    memory_objs = memory_objs[:1]
                elif invalid_result == "none":
                    memory_objs[1] = None
                else:
                    extra_memory_obj = _Memory()
                    tracked_memory_objs.append(extra_memory_obj)
                    memory_objs.append(extra_memory_obj)
            for memory_obj in memory_objs:
                if memory_obj is not None:
                    memory_obj.ref_count_up()
                    memory_obj.pin()
            yield _Task(cast(list[_Memory], memory_objs))

    cast(Any, storage).layerwise_batched_get = invalid_layerwise_batched_get

    retriever = engine.retrieve_layer(tokens, explicit_spans=declaration)
    assert torch.equal(next(retriever), torch.tensor(4))
    assert next(retriever) is None
    with pytest.raises(RuntimeError, match="Invalid layerwise retrieval result"):
        next(retriever)

    # Layer 0 was sent before layer 1 failed validation, but no final success
    # mask was produced and the malformed layer never reached the connector.
    assert len(connector.loaded_payloads) == 1
    assert connector.closed
    assert all(memory_obj.ref_count == 1 for memory_obj in tracked_memory_objs)
    assert all(memory_obj.pin_count == 0 for memory_obj in tracked_memory_objs)


def test_explicit_retrieve_skips_partial_layer_span_without_acquiring_it(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine()
    tokens = list(range(6))
    declaration = ExplicitTokenSpans(
        6,
        TokenSpan(0, 6),
        (TokenSpan(0, 2), TokenSpan(2, 4), TokenSpan(4, 6)),
    )
    partial_objects = _populate_span(engine, storage, tokens, declaration, 0)
    del storage.objects[_keys_for_span(engine, tokens, declaration, 0)[1]]
    hit_objects = _populate_span(engine, storage, tokens, declaration, 2)

    yields = list(engine.retrieve_layer(tokens, explicit_spans=declaration))

    assert connector.loaded_spans == [(4, 6)]
    assert torch.equal(
        yields[-1],
        torch.tensor([False, False, False, False, True, True]),
    )
    assert partial_objects[0].ref_count == 1
    assert partial_objects[0].pin_count == 0
    assert all(memory_obj.ref_count == 1 for memory_obj in hit_objects)
    assert all(memory_obj.pin_count == 0 for memory_obj in hit_objects)


def test_explicit_multi_location_layers_fail_closed_and_release_all_pins(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine()
    tokens = list(range(4))
    declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(2, 4)),
    )
    first_hit = _populate_span(engine, storage, tokens, declaration, 0)
    split_hit = _populate_span(engine, storage, tokens, declaration, 1)
    split_keys = _keys_for_span(engine, tokens, declaration, 1)
    ordinary_batched_contains = storage.batched_contains
    ordinary_iter_lookup_pins = storage.iter_lookup_pins

    def batched_contains(keys, search_range, pin):
        if keys != split_keys:
            return ordinary_batched_contains(keys, search_range, pin)
        if pin:
            for key in keys:
                storage.objects[key].pin()
        return len(keys), {
            storage.location: [keys[0]],
            "RemoteBackend": [keys[1]],
        }

    def iter_lookup_pins(keys, search_range):
        if keys != split_keys:
            yield from ordinary_iter_lookup_pins(keys, search_range)
            return
        for location, key in zip(
            (storage.location, "RemoteBackend"),
            keys,
            strict=True,
        ):
            storage.objects[key].pin()
            yield location, key

    def batched_unpin(keys, locations):
        del locations
        for key in keys:
            storage.objects[key].unpin()

    storage_state = cast(Any, storage)
    storage_state.batched_contains = batched_contains
    storage_state.iter_lookup_pins = iter_lookup_pins
    storage_state.batched_unpin = batched_unpin

    with pytest.raises(NotImplementedError, match="one storage location"):
        engine.lookup(
            tokens=tokens,
            explicit_spans=declaration,
            lookup_id="split-locations",
            pin=True,
        )

    assert all(memory_obj.pin_count == 0 for memory_obj in first_hit + split_hit)
    assert all(memory_obj.unpin_count == 1 for memory_obj in first_hit + split_hit)
    assert not engine.lookup_pins["split-locations"]

    with pytest.raises(NotImplementedError, match="one storage location"):
        list(engine.retrieve_layer(tokens, explicit_spans=declaration))

    assert connector.loaded_spans == []
    assert all(memory_obj.ref_count == 1 for memory_obj in first_hit + split_hit)
    assert all(memory_obj.pin_count == 0 for memory_obj in first_hit + split_hit)


def test_explicit_spans_in_different_locations_fail_closed_and_release_pins(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine()
    tokens = list(range(4))
    declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(2, 4)),
    )
    first_hit = _populate_span(engine, storage, tokens, declaration, 0)
    second_hit = _populate_span(engine, storage, tokens, declaration, 1)
    second_keys = _keys_for_span(engine, tokens, declaration, 1)
    ordinary_batched_contains = storage.batched_contains
    ordinary_iter_lookup_pins = storage.iter_lookup_pins

    def batched_contains(keys, search_range, pin):
        if keys != second_keys:
            return ordinary_batched_contains(keys, search_range, pin)
        if pin:
            for key in keys:
                storage.objects[key].pin()
        return len(keys), {"RemoteBackend": keys}

    def iter_lookup_pins(keys, search_range):
        if keys != second_keys:
            yield from ordinary_iter_lookup_pins(keys, search_range)
            return
        del search_range
        for key in keys:
            storage.objects[key].pin()
            yield "RemoteBackend", key

    def batched_unpin(keys, locations):
        del locations
        for key in keys:
            storage.objects[key].unpin()

    storage_state = cast(Any, storage)
    storage_state.batched_contains = batched_contains
    storage_state.iter_lookup_pins = iter_lookup_pins
    storage_state.batched_unpin = batched_unpin

    with pytest.raises(NotImplementedError, match="all hits in one"):
        engine.lookup(
            tokens=tokens,
            explicit_spans=declaration,
            lookup_id="different-locations",
            pin=True,
        )

    all_objects = first_hit + second_hit
    assert all(memory_obj.pin_count == 0 for memory_obj in all_objects)
    assert all(memory_obj.unpin_count == 1 for memory_obj in all_objects)
    assert not engine.lookup_pins["different-locations"]

    with pytest.raises(NotImplementedError, match="all hits in one"):
        list(engine.retrieve_layer(tokens, explicit_spans=declaration))

    assert connector.loaded_spans == []
    assert all(memory_obj.ref_count == 1 for memory_obj in all_objects)
    assert all(memory_obj.pin_count == 0 for memory_obj in all_objects)


def test_explicit_retrieve_failure_releases_every_acquired_ref_and_pin(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine(
        fail_on_layer=1,
        retrieval_owns_pins=True,
    )
    tokens = list(range(6))
    declaration = ExplicitTokenSpans(
        6,
        TokenSpan(0, 6),
        (TokenSpan(0, 2), TokenSpan(4, 6)),
    )
    _populate_span(engine, storage, tokens, declaration, 0)
    _populate_span(engine, storage, tokens, declaration, 1)

    retriever = engine.retrieve_layer(tokens, explicit_spans=declaration)
    assert torch.equal(next(retriever), torch.tensor(4))
    next(retriever)
    with pytest.raises(InjectedExplicitLoadFailure, match="layer 1"):
        next(retriever)

    assert connector.closed
    assert all(memory_obj.ref_count == 1 for memory_obj in storage.objects.values())
    assert all(memory_obj.pin_count == 0 for memory_obj in storage.objects.values())
    assert all(memory_obj.unpin_count == 1 for memory_obj in storage.objects.values())


@pytest.mark.parametrize(
    ("connector_type", "device_module"),
    [
        (VLLMBufferLayerwiseGPUConnector, torch.cuda),
        (VLLMBufferLayerwiseXPUConnector, torch.xpu),
    ],
)
def test_explicit_buffer_anchors_at_zero_and_zeros_every_gap(
    monkeypatch,
    connector_type,
    device_module,
) -> None:
    monkeypatch.setattr(device_module, "synchronize", lambda: None)
    monkeypatch.setattr(device_module, "stream", lambda stream: nullcontext())

    connector = connector_type.__new__(connector_type)
    connector_state = cast(Any, connector)
    allocator = _TensorAllocator()
    connector_state.cache_positions = False
    connector_state.fused_rotary_emb = None
    connector_state.dtype = torch.float32
    connector_state.kvcaches = [torch.empty(1)]
    connector_state.gpu_buffer_allocator = allocator
    connector_state.load_stream = object()
    connector_state.num_layers = 1
    connector_state.initialize_kvcaches_ptr = lambda **kwargs: None
    connector_state._lazy_initialize_buffer = lambda kvcaches: None
    connector_state.get_shape = lambda num_tokens: torch.Size((2, num_tokens, 1))
    state = LayerwiseLoadState()

    consumer = connector.batched_to_gpu(
        [2, 6],
        [4, 8],
        load_state=state,
        buffer_span=TokenSpan(0, 10),
        slot_mapping=torch.arange(10),
    )
    next(consumer)
    memory_objs = [
        _TensorMemory(torch.full((2, 2, 1), 2.0)),
        _TensorMemory(torch.full((2, 2, 1), 6.0)),
    ]
    consumer.send(cast(list[MemoryObj], memory_objs))

    assert [memory_obj.tensor.shape[1] for memory_obj in allocator.allocations] == [
        10,
        10,
    ]
    assert state.gap_positions is not None
    assert torch.equal(state.gap_positions, torch.tensor([0, 1, 4, 5, 8, 9]))
    loaded = state.buffer_mapping[0].tensor
    assert loaded is not None
    assert torch.equal(loaded[:, 0:2], torch.zeros((2, 2, 1)))
    assert torch.equal(loaded[:, 2:4], torch.full((2, 2, 1), 2.0))
    assert torch.equal(loaded[:, 4:6], torch.zeros((2, 2, 1)))
    assert torch.equal(loaded[:, 6:8], torch.full((2, 2, 1), 6.0))
    assert torch.equal(loaded[:, 8:10], torch.zeros((2, 2, 1)))

    consumer.close()
    assert state.is_released
    assert [memory_obj.release_count for memory_obj in allocator.allocations] == [
        1,
        1,
    ]


def test_legacy_buffer_still_anchors_at_first_hit(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    connector = VLLMBufferLayerwiseGPUConnector.__new__(VLLMBufferLayerwiseGPUConnector)
    connector_state = cast(Any, connector)
    allocator = _TensorAllocator()
    connector_state.cache_positions = False
    connector_state.fused_rotary_emb = None
    connector_state.dtype = torch.float32
    connector_state.kvcaches = [torch.empty(1)]
    connector_state.gpu_buffer_allocator = allocator
    connector_state.load_stream = object()
    connector_state.num_layers = 1
    connector_state.initialize_kvcaches_ptr = lambda **kwargs: None
    connector_state._lazy_initialize_buffer = lambda kvcaches: None
    connector_state.get_shape = lambda num_tokens: torch.Size((2, num_tokens, 1))
    state = LayerwiseLoadState()

    consumer = connector.batched_to_gpu(
        [2, 6],
        [4, 8],
        load_state=state,
        slot_mapping=torch.arange(10),
    )
    next(consumer)
    memory_objs = [
        _TensorMemory(torch.full((2, 2, 1), 2.0)),
        _TensorMemory(torch.full((2, 2, 1), 6.0)),
    ]
    consumer.send(cast(list[MemoryObj], memory_objs))

    assert [memory_obj.tensor.shape[1] for memory_obj in allocator.allocations] == [
        6,
        6,
    ]
    assert state.gap_positions is not None
    assert torch.equal(state.gap_positions, torch.tensor([2, 3]))

    consumer.close()


def test_legacy_layerwise_retrieve_keeps_first_miss_and_yield_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )
    engine, storage, connector = _engine()
    tokens = [10, 11, 99, 20, 21, 99, 30, 31]
    database = cast(SegmentTokenDatabase, engine.token_database)
    legacy_chunks = list(database.process_tokens(tokens=tokens))
    for chunk_index in (0, 2):
        key = legacy_chunks[chunk_index][2]
        assert isinstance(key, CacheEngineKey)
        for layer_key in key.split_layers(engine.num_layers):
            storage.objects[layer_key] = _Memory()

    yields = list(engine.retrieve_layer(tokens))

    assert len(yields) == engine.num_layers + 2
    assert connector.loaded_spans == [legacy_chunks[0][:2]]
    assert connector.buffer_span is None
    assert torch.equal(
        yields[-1],
        torch.tensor([True, True, False, False, False, False, False, False]),
    )
    checked_hashes = [key.chunk_hash for key in storage.contains_calls]
    assert checked_hashes == [
        cast(CacheEngineKey, legacy_chunks[0][2]).chunk_hash,
        cast(CacheEngineKey, legacy_chunks[1][2]).chunk_hash,
    ]
