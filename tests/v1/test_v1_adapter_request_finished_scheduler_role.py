# SPDX-License-Identifier: Apache-2.0
"""Regression test for LMCache#3337.

``request_finished`` is a Scheduler-side connector API. In vLLM v1's
split-process architecture:

- The Scheduler typically does **not** initialize ``self.lmcache_engine``;
  only the Worker role builds it by default (the Scheduler also builds
  it when ``enable_scheduler_bypass_lookup`` is set).
- The Scheduler **does** own ``self.lookup_client`` -- it is created
  only on the Scheduler role (see
  ``vllm_service_factory.maybe_create_lookup_client``).

Previously the ``FINISHED_ABORTED`` cleanup path asserted both
``self.lmcache_engine is not None`` and ``self.lookup_client is not None``.
Under concurrent abort cleanups the first assert fired as an unhandled
``AssertionError`` and killed the EngineCore process for every connected
user.

The fix replaces both asserts with ``if-warn-skip`` so the engine stays
alive and only the affected request's cleanup is dropped. It also moves
the async lookup cancel out of the engine-None branch -- since the
Scheduler owns ``lookup_client`` independently of the engine, the lookup
cancel must still run for Scheduler-side aborts to avoid leaking
in-flight async lookups.

This test locks in:

1. Scheduler default (engine=None, lookup_client=None, async_loading=False)
   + abort -> no crash, warning emitted, no engine attribute is read.
2. Scheduler default (engine=None) + lookup_client present + abort ->
   ``lookup_client.cancel_lookup`` still runs for synchronous and asynchronous
   lookup (resource-leak regression guard).
3. Engine initialized + abort -> ``storage_manager.cancel_request`` runs.
4. Engine initialized + ``async_loading=True`` + lookup_client present
   + abort -> both cancels run.
5. ``async_loading=True`` + lookup_client missing + abort -> warns and
   skips the lookup cancel instead of crashing.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast
import logging
import threading

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
# Third Party (after importorskip)
from vllm.v1.request import RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
)
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupClient

_ADAPTER_LOGGER_NAME = "lmcache.integration.vllm.vllm_v1_adapter"


@contextmanager
def _capture_adapter_warnings() -> Iterator[list[logging.LogRecord]]:
    """Capture WARNING records emitted by the adapter logger.

    lmcache's ``init_logger`` sets ``propagate = False`` on the adapter
    logger, so its records never reach pytest's ``caplog`` (which attaches a
    handler to the root logger). Attach a handler directly to the named
    logger instead -- the convention used across the v1 adapter tests -- and
    force WARNING level so a higher ``LMCACHE_LOG_LEVEL`` cannot filter the
    records out before the handler sees them.
    """
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    logger = logging.getLogger(_ADAPTER_LOGGER_NAME)
    original_level = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


class _FakeStorageManager:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    def cancel_request(self, req_id: str) -> None:
        self.cancelled.append(req_id)


class _FakeEngine:
    def __init__(self) -> None:
        self.storage_manager = _FakeStorageManager()


class _FakeLookupClient:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.transferred: list[str] = []
        self.finished: list[str] = []

    def lookup_cache(self, req_id: str) -> int | None:
        return -1

    def clear_lookup_status(self, req_id: str) -> None:
        return

    def transfer_lookup(self, req_id: str) -> None:
        self.transferred.append(req_id)

    def transfer_lookups(self, req_ids: list[str]) -> None:
        self.transferred.extend(req_ids)

    def cancel_lookup(self, req_id: str) -> None:
        self.cancelled.append(req_id)

    def finish_lookup(self, req_id: str) -> None:
        self.finished.append(req_id)


def _make_aborted_request(req_id: str) -> SimpleNamespace:
    """Build the minimum request shape ``request_finished`` reads."""
    return SimpleNamespace(
        request_id=req_id,
        status=RequestStatus.FINISHED_ABORTED,
    )


def _make_connector(
    *,
    engine: object,
    lookup_client: object,
    async_loading: bool,
) -> LMCacheConnectorV1Impl:
    """Bypass ``__init__`` and pin only the attributes ``request_finished``
    reads, so the test exercises the abort cleanup path in isolation.

    ``lmcache_engine`` and ``lookup_client`` are read-only properties on
    ``LMCacheConnectorV1Impl`` that delegate to ``self._manager``, so the
    fixture installs a fake manager rather than assigning to those
    properties directly.
    """
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=engine,
        lookup_client=lookup_client,
    )
    connector.async_loading = async_loading
    connector.use_layerwise = False
    connector.config = SimpleNamespace(
        priority_limit=None,
        get_extra_config_value=lambda key, default: default,
    )
    connector._request_trackers = {}
    connector._unfinished_requests = {}
    connector.load_specs = {}
    connector._requests_priority = {}
    connector._prealloc_lookup_ids = {}
    connector._blocked_prealloc_requests = set()
    connector._allocated_lookup_ids = {}
    connector._worker_owned_lookup_requests = set()
    connector.kv_role = "kv_both"
    connector.force_skip_save = False
    return connector


def test_request_finished_scheduler_default_engine_none_skips_cleanup() -> None:
    """Scheduler default config: ``lmcache_engine`` is None, abort must
    not crash the EngineCore.

    Regression for https://github.com/LMCache/LMCache/issues/3337.
    """
    connector = _make_connector(engine=None, lookup_client=None, async_loading=False)
    request = _make_aborted_request("req-scheduler-default-abort")

    with _capture_adapter_warnings() as records:
        delay_free, return_params = connector.request_finished(request, [0, 1])

    assert delay_free is False
    assert return_params is None

    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert any(
        "req-scheduler-default-abort" in r.getMessage()
        and "lmcache_engine" in r.getMessage()
        for r in warnings
    ), (
        "Expected a warning naming the request id and lmcache_engine; "
        f"got {[r.getMessage() for r in warnings]}"
    )


def test_request_finished_scheduler_default_async_loading_runs_lookup_cancel() -> None:
    """Scheduler default (engine=None) + ``async_loading=True`` +
    lookup_client present: ``lookup_client.cancel_lookup`` must still
    run. Resource-leak regression guard -- the Scheduler owns
    ``lookup_client`` even when it does not build an engine, so the
    lookup cancel cannot be nested inside the engine-not-None branch.
    """
    lookup_client = _FakeLookupClient()
    connector = _make_connector(
        engine=None, lookup_client=lookup_client, async_loading=True
    )
    request = _make_aborted_request("req-scheduler-async-abort")

    connector.request_finished(request, [0, 1])

    assert lookup_client.cancelled == ["req-scheduler-async-abort"], (
        "Scheduler-side abort must still cancel async lookups even when "
        "lmcache_engine is None; otherwise async lookups leak."
    )


def test_request_finished_scheduler_default_sync_runs_lookup_cancel() -> None:
    """A sync worker lookup pins before Scheduler allocation. If allocation
    never happens, ``wait_for_save`` cannot unpin and abort must use the lookup
    client's worker cleanup RPC even though the Scheduler has no engine.
    """
    lookup_client = _FakeLookupClient()
    connector = _make_connector(
        engine=None, lookup_client=lookup_client, async_loading=False
    )
    request = _make_aborted_request("req-scheduler-sync-abort")

    connector.request_finished(request, [0, 1])

    assert lookup_client.cancelled == ["req-scheduler-sync-abort"]


def test_request_finished_engine_initialized_runs_storage_manager_cancel() -> None:
    """Engine-initialized path (Worker, or Scheduler with bypass lookup):
    ``cancel_request`` must still fire on abort. Guards against the fix
    accidentally short-circuiting the happy path.
    """
    engine = _FakeEngine()
    connector = _make_connector(engine=engine, lookup_client=None, async_loading=False)
    request = _make_aborted_request("req-engine-abort")

    delay_free, return_params = connector.request_finished(request, [0, 1])

    assert delay_free is False
    assert return_params is None
    assert engine.storage_manager.cancelled == ["req-engine-abort"]


def test_request_finished_engine_with_async_loading_runs_both_cancels() -> None:
    """Engine-initialized + ``async_loading=True`` + ``lookup_client``
    present: both ``storage_manager.cancel_request`` and
    ``lookup_client.cancel_lookup`` must fire on abort.
    """
    engine = _FakeEngine()
    lookup_client = _FakeLookupClient()
    connector = _make_connector(
        engine=engine, lookup_client=lookup_client, async_loading=True
    )
    request = _make_aborted_request("req-async-abort")

    connector.request_finished(request, [0, 1])

    assert engine.storage_manager.cancelled == ["req-async-abort"]
    assert lookup_client.cancelled == ["req-async-abort"]


def test_request_finished_async_loading_without_lookup_client_skips_cancel() -> None:
    """``async_loading=True`` + ``lookup_client`` missing: the second
    assert in the original block would have crashed here too. The fix
    must warn and skip, matching the fail-soft contract.
    """
    engine = _FakeEngine()
    connector = _make_connector(engine=engine, lookup_client=None, async_loading=True)
    request = _make_aborted_request("req-no-lookup-abort")

    with _capture_adapter_warnings() as records:
        connector.request_finished(request, [0, 1])

    # storage_manager.cancel_request still fired (only the lookup path skipped)
    assert engine.storage_manager.cancelled == ["req-no-lookup-abort"]

    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert any(
        "req-no-lookup-abort" in r.getMessage() and "lookup_client" in r.getMessage()
        for r in warnings
    ), (
        "Expected a warning naming the request id and lookup_client; "
        f"got {[r.getMessage() for r in warnings]}"
    )


def test_build_meta_cancels_unscheduled_prealloc_attempt_once() -> None:
    lookup_client = _FakeLookupClient()
    connector = _make_connector(
        engine=None,
        lookup_client=lookup_client,
        async_loading=False,
    )
    request_id = "req-no-forward"
    lookup_id = "lookup-no-forward"
    connector._prealloc_lookup_ids[request_id] = lookup_id
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={},
    )

    metadata = connector.build_connector_meta(scheduler_output)

    assert metadata.requests == []
    assert lookup_client.cancelled == [lookup_id]
    assert connector._prealloc_lookup_ids == {}

    connector.build_connector_meta(scheduler_output)
    assert lookup_client.cancelled == [lookup_id]


def test_build_meta_rolls_back_allocated_lookup_when_internal_build_raises() -> None:
    lookup_client = _FakeLookupClient()
    connector = _make_connector(
        engine=None,
        lookup_client=lookup_client,
        async_loading=False,
    )
    request_id = "req-build-failure"
    lookup_id = "lookup-build-failure"
    connector._allocated_lookup_ids[request_id] = lookup_id
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(req_id=request_id, num_computed_tokens=0)],
        scheduled_cached_reqs=[],
        # Deliberately omit request_id so the real metadata builder fails
        # after allocation but before it can return worker metadata.
        num_scheduled_tokens={},
    )

    with pytest.raises(KeyError):
        connector.build_connector_meta(scheduler_output)

    assert lookup_client.cancelled == [lookup_id]
    assert lookup_client.transferred == []
    assert connector._allocated_lookup_ids == {}
    assert connector._worker_owned_lookup_requests == set()


def test_build_meta_rejects_and_cancels_allocated_lookup_omission() -> None:
    lookup_client = _FakeLookupClient()
    connector = _make_connector(
        engine=None,
        lookup_client=lookup_client,
        async_loading=False,
    )
    request_id = "req-metadata-omission"
    lookup_id = "lookup-metadata-omission"
    connector._allocated_lookup_ids[request_id] = lookup_id

    def build_omitted_metadata(scheduler_output) -> LMCacheConnectorMetadata:
        return LMCacheConnectorMetadata()

    connector._build_connector_meta = (  # type: ignore[method-assign, assignment]
        build_omitted_metadata
    )

    with pytest.raises(RuntimeError, match="omitted or changed"):
        connector.build_connector_meta(SimpleNamespace())

    assert lookup_client.cancelled == [lookup_id]
    assert lookup_client.transferred == []
    assert connector._allocated_lookup_ids == {}
    assert connector._worker_owned_lookup_requests == set()


def test_build_meta_batch_transfer_failure_rolls_back_every_lookup() -> None:
    class FailingBatchLookupClient(_FakeLookupClient):
        def __init__(self) -> None:
            super().__init__()
            self.attempted_batches: list[list[str]] = []

        def transfer_lookups(self, req_ids: list[str]) -> None:
            self.attempted_batches.append(list(req_ids))
            raise RuntimeError("injected batch preflight failure")

    lookup_client = FailingBatchLookupClient()
    connector = _make_connector(
        engine=None,
        lookup_client=lookup_client,
        async_loading=False,
    )
    allocated = {
        "req-batch-a": "lookup-batch-a",
        "req-batch-b": "lookup-batch-b",
    }
    connector._allocated_lookup_ids.update(allocated)

    def build_batch_metadata(scheduler_output) -> LMCacheConnectorMetadata:
        return LMCacheConnectorMetadata(
            requests=[
                cast(
                    ReqMeta,
                    SimpleNamespace(req_id=request_id, lookup_id=lookup_id),
                )
                for request_id, lookup_id in allocated.items()
            ]
        )

    connector._build_connector_meta = (  # type: ignore[method-assign, assignment]
        build_batch_metadata
    )

    with pytest.raises(RuntimeError, match="batch preflight failure"):
        connector.build_connector_meta(SimpleNamespace())

    assert lookup_client.attempted_batches == [list(allocated.values())]
    assert lookup_client.transferred == []
    assert lookup_client.cancelled == list(allocated.values())
    assert connector._allocated_lookup_ids == {}
    assert connector._worker_owned_lookup_requests == set()


def test_update_validation_failure_keeps_prealloc_cancel_owner() -> None:
    lookup_client = _FakeLookupClient()
    connector = _make_connector(
        engine=None,
        lookup_client=lookup_client,
        async_loading=False,
    )
    request_id = "req-update-failure"
    lookup_id = "lookup-update-failure"
    connector._prealloc_lookup_ids[request_id] = lookup_id
    connector.load_specs[request_id] = LoadSpec(
        vllm_cached_tokens=0,
        lmcache_cached_tokens=4,
        can_load=False,
    )
    request = SimpleNamespace(
        request_id=request_id,
        num_tokens=4,
        status=RequestStatus.FINISHED_ABORTED,
    )

    with pytest.raises(AssertionError, match="Mismatch in tokens to load"):
        connector.update_state_after_alloc(request, 1)

    assert connector._prealloc_lookup_ids == {request_id: lookup_id}
    assert connector._allocated_lookup_ids == {}
    assert lookup_client.transferred == []

    connector.request_finished(request, [0, 1])
    assert lookup_client.cancelled == [lookup_id]
    assert connector._prealloc_lookup_ids == {}


def test_postalloc_abort_does_not_race_worker_pin_consumption() -> None:
    request_id = "postalloc-race"
    lookup_id = "lookup-postalloc-race"
    entered = threading.Event()
    resume = threading.Event()

    class BarrierPins(dict):
        def __contains__(self, key) -> bool:
            if threading.current_thread().name == "retrieve-thread":
                entered.set()
                assert resume.wait(timeout=2.0)
            return super().__contains__(key)

    class TokenDatabase:
        def __init__(self, key: CacheEngineKey) -> None:
            self.key = key

        def process_tokens(self, **kwargs):
            yield 0, 2, self.key

    class Storage:
        def __init__(self) -> None:
            self.unpinned: list[CacheEngineKey] = []

        def batched_unpin(self, keys, locations) -> None:
            self.unpinned.extend(keys)

        def batched_get(self, keys, location):
            return [None for _ in keys]

        def get_block_mapping(self, chunks):
            return {}

    key = CacheEngineKey("model", 1, 0, 123, torch.float16, {})
    storage = Storage()
    worker_engine = LMCacheEngine.__new__(LMCacheEngine)
    worker_engine.storage_manager = storage  # type: ignore[assignment]
    worker_engine.token_database = TokenDatabase(key)  # type: ignore[assignment]
    worker_engine.lookup_pins = BarrierPins({lookup_id: {"LocalCPUBackend": [key]}})
    worker_engine.lookup_pin_generations = {}
    worker_engine._lookup_pins_lock = threading.RLock()
    worker_engine.async_loading = None

    class WorkerBackedLookupClient(_FakeLookupClient):
        def cancel_lookup(self, req_id: str) -> None:
            super().cancel_lookup(req_id)
            worker_engine.lookup_unpin(req_id)

    lookup_client = WorkerBackedLookupClient()
    connector = _make_connector(
        engine=None,
        lookup_client=lookup_client,
        async_loading=False,
    )
    connector._prealloc_lookup_ids[request_id] = lookup_id
    request = SimpleNamespace(
        request_id=request_id,
        status=RequestStatus.FINISHED_ABORTED,
    )
    connector.update_state_after_alloc(request, 0)
    assert lookup_client.transferred == []
    assert connector._allocated_lookup_ids == {request_id: lookup_id}
    connector._commit_metadata_lookup_transfers(
        LMCacheConnectorMetadata(
            requests=[
                cast(
                    ReqMeta,
                    SimpleNamespace(req_id=request_id, lookup_id=lookup_id),
                )
            ]
        )
    )
    assert lookup_client.transferred == [lookup_id]
    assert connector._allocated_lookup_ids == {}
    assert request_id in connector._worker_owned_lookup_requests

    errors: list[BaseException] = []

    def retrieve() -> None:
        try:
            worker_engine._process_tokens_internal(
                [1, 2],
                None,
                torch.zeros(2, dtype=torch.bool),
                req_id=lookup_id,
            )
        except BaseException as error:
            errors.append(error)

    retrieve_thread = threading.Thread(target=retrieve, name="retrieve-thread")
    retrieve_thread.start()
    assert entered.wait(timeout=2.0)

    connector.request_finished(request, [0, 1])
    assert lookup_client.cancelled == []

    resume.set()
    retrieve_thread.join(timeout=2.0)
    assert not retrieve_thread.is_alive()
    assert errors == []
    assert lookup_id in worker_engine.lookup_pins

    worker_engine.lookup_unpin(lookup_id)
    assert worker_engine.lookup_pins == {}
    assert storage.unpinned == [key]


def test_resumed_prealloc_abort_overrides_old_worker_owned_attempt() -> None:
    lookup_client = _FakeLookupClient()
    connector = _make_connector(
        engine=None,
        lookup_client=lookup_client,
        async_loading=False,
    )
    request_id = "req-resumed-abort"
    new_lookup_id = "lookup-attempt-b"
    connector._worker_owned_lookup_requests.add(request_id)
    connector._prealloc_lookup_ids[request_id] = new_lookup_id
    request = _make_aborted_request(request_id)

    connector.request_finished(request, [0, 1])

    assert lookup_client.cancelled == [new_lookup_id]
    assert request_id not in connector._worker_owned_lookup_requests
    assert connector._prealloc_lookup_ids == {}


def test_uncertain_probe_retains_uuid_until_cancel_ack_then_rotates() -> None:
    class DeferredLookupClient(_FakeLookupClient):
        def __init__(self) -> None:
            super().__init__()
            self.statuses: dict[str, int | None] = {}
            self.bindings: dict[str, tuple[tuple[int, ...], tuple]] = {}
            self.lookup_calls: list[str] = []

        def validate_lookup_request(
            self,
            token_ids,
            lookup_id: str,
            request_configs=None,
            explicit_spans=None,
        ) -> None:
            binding = (
                tuple(token_ids),
                tuple(sorted((request_configs or {}).items())),
            )
            prior = self.bindings.setdefault(lookup_id, binding)
            if prior != binding:
                raise ValueError(
                    "lookup_id was reused with different lookup request content"
                )

        def lookup_cache(self, lookup_id: str) -> int | None:
            return self.statuses.get(lookup_id, -1)

        def lookup(
            self,
            token_ids,
            lookup_id: str,
            request_configs=None,
            explicit_spans=None,
        ) -> None:
            self.lookup_calls.append(lookup_id)
            self.statuses[lookup_id] = None
            return None

    lookup_client = DeferredLookupClient()
    connector = _make_connector(
        engine=None,
        lookup_client=lookup_client,
        async_loading=False,
    )
    connector.skip_last_n_tokens = 0
    connector._max_tokens_per_load = 0
    connector._lmcache_chunk_size = 2
    connector.config.min_retrieve_tokens = 0
    request = SimpleNamespace(
        request_id="req-deferred-probe",
        all_token_ids=[1, 2, 3, 4],
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SimpleNamespace(extra_args={}),
        priority=0,
        num_tokens=4,
    )

    assert connector.get_num_new_matched_tokens(request, 0) is None
    first_lookup_id = connector._prealloc_lookup_ids[request.request_id]
    assert lookup_client.lookup_calls == [first_lookup_id]

    # Repeated scheduling ticks keep one blocked UUID and do not issue a new
    # lookup while cancellation remains uncertain.
    assert connector.get_num_new_matched_tokens(request, 0) is None
    assert connector._prealloc_lookup_ids[request.request_id] == first_lookup_id
    assert lookup_client.lookup_calls == [first_lookup_id]

    request.all_token_ids = [1, 2, 9, 4]
    with pytest.raises(ValueError, match="different lookup request content"):
        connector.get_num_new_matched_tokens(request, 0)

    # A typed cancel ACK is represented by local retirement (-1). The next
    # scheduling attempt may now rotate to a fresh UUID and probe again.
    request.all_token_ids = [1, 2, 3, 4]
    lookup_client.statuses[first_lookup_id] = -1
    assert connector.get_num_new_matched_tokens(request, 0) is None
    second_lookup_id = connector._prealloc_lookup_ids[request.request_id]
    assert second_lookup_id != first_lookup_id
    assert lookup_client.lookup_calls == [first_lookup_id, second_lookup_id]


def test_cached_scheduler_probe_validates_tokens_and_request_configs() -> None:
    lookup_client = LMCacheLookupClient.__new__(LMCacheLookupClient)
    lookup_client.config = SimpleNamespace(
        use_layerwise=True,
        enable_async_loading=False,
    )
    lookup_client.enable_blending = True
    lookup_client.reqs_status = {}
    lookup_client.reqs_fingerprints = {}
    lookup_client.reqs_generations = {}
    lookup_client.pending_cancel_ids = set()
    request_configs = {"lmcache.tag.tenant": "a"}
    lookup_id = "cached-probe-attempt"
    lookup_client.validate_lookup_request(
        [1, 2, 3, 4],
        lookup_id,
        request_configs,
    )
    lookup_client.reqs_status[lookup_id] = 2

    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=None,
        lookup_client=lookup_client,
    )
    connector.kv_role = "kv_both"
    connector.skip_last_n_tokens = 0
    connector._requests_priority = {}
    connector.config = SimpleNamespace(min_retrieve_tokens=0)
    connector._max_tokens_per_load = 0
    connector._lmcache_chunk_size = 2
    connector.load_specs = {}
    connector._prealloc_lookup_ids = {"cached-probe": lookup_id}
    connector._blocked_prealloc_requests = set()
    connector._allocated_lookup_ids = {}
    connector._worker_owned_lookup_requests = set()
    request = SimpleNamespace(
        request_id="cached-probe",
        all_token_ids=[1, 2, 3, 4],
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SimpleNamespace(
            extra_args={
                "kv_transfer_params": {"lmcache.tag.tenant": "a"},
            }
        ),
        priority=0,
        num_tokens=4,
    )

    assert connector.get_num_new_matched_tokens(request, 0) == 2

    request.all_token_ids = [1, 2, 9, 4]
    with pytest.raises(ValueError, match="different lookup request content"):
        connector.get_num_new_matched_tokens(request, 0)

    request.all_token_ids = [1, 2, 3, 4]
    request.sampling_params.extra_args["kv_transfer_params"]["lmcache.tag.tenant"] = "b"
    with pytest.raises(ValueError, match="different lookup request content"):
        connector.get_num_new_matched_tokens(request, 0)
