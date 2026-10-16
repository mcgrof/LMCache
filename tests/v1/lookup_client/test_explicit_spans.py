# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Callable, Optional, cast
from unittest.mock import MagicMock, patch
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.hit_limit_lookup_client import HitLimitLookupClient
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
)
from lmcache.v1.lookup_client.lmcache_lookup_client import (
    LMCacheLookupClient,
    LMCacheLookupServer,
)
from lmcache.v1.lookup_client.lmcache_lookup_client_bypass import (
    LMCacheBypassLookupClient,
)
from lmcache.v1.lookup_client.mooncake_lookup_client import MooncakeLookupClient
from lmcache.v1.manager import LMCacheManager
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan


def _declaration() -> ExplicitTokenSpans:
    return ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 2), TokenSpan(3, 4)),
    )


def _explicit_lookup_message(lookup_id: str, generation: bytes) -> list[Any]:
    return [
        {
            "schema": "lmcache.lookup.explicit-spans",
            "version": 1,
            "generation": generation,
            "tokens": [1, 2, 3, 4],
            "token_count": 4,
            "cached_span": [0, 4],
            "spans": [[0, 2], [3, 4]],
        },
        lookup_id,
        "",
    ]


def _explicit_cancel_message(lookup_id: str, generation: bytes) -> list[Any]:
    return [
        {
            "schema": "lmcache.lookup.control",
            "version": 1,
            "operation": "cancel",
            "generation": generation,
        },
        lookup_id,
        "",
    ]


def _call_and_signal(
    operation: Callable[..., Any],
    completed: threading.Event,
    *args: Any,
) -> None:
    operation(*args)
    completed.set()


@pytest.mark.parametrize(
    "client_type",
    [LMCacheAsyncLookupClient, MooncakeLookupClient],
)
def test_async_and_external_clients_fail_closed_on_explicit_spans(
    client_type,
) -> None:
    client = client_type.__new__(client_type)

    assert not client.supports_explicit_spans()
    with pytest.raises(NotImplementedError, match="explicit token spans"):
        client.lookup([1, 2, 3, 4], "lookup", explicit_spans=_declaration())


def test_hit_limit_wrapper_reports_explicit_spans_unsupported() -> None:
    client = HitLimitLookupClient.__new__(HitLimitLookupClient)

    assert not client.supports_explicit_spans()
    with pytest.raises(NotImplementedError, match="preserve explicit span"):
        client.lookup([1, 2, 3, 4], "lookup", explicit_spans=_declaration())


class _RecordingEngine:
    def __init__(self, supported: bool, lookup_result: int = 4) -> None:
        self.supported = supported
        self.lookup_result = lookup_result
        self.kwargs: Optional[dict[str, Any]] = None
        self.lookup_count = 0
        self.pinned_ids: set[str] = set()
        self.lookup_pins: dict[str, Any] = {}
        self.lookup_pin_generations: dict[str, bytes] = {}
        self.lookup_cancelled_generations: OrderedDict[tuple[str, bytes], None] = (
            OrderedDict()
        )
        self.unpinned_ids: list[str] = []
        self.storage_manager: Any = SimpleNamespace(
            cancelled=[], cancel_request=self._cancel
        )

    def supports_explicit_spans(self) -> bool:
        return self.supported

    def lookup(self, **kwargs: Any) -> int:
        self.lookup_count += 1
        self.kwargs = kwargs
        if kwargs.get("pin") and self.lookup_result > 0:
            self.pinned_ids.add(kwargs["lookup_id"])
            self.lookup_pins[kwargs["lookup_id"]] = []
        return self.lookup_result

    def lookup_unpin(self, lookup_id: str) -> None:
        self.pinned_ids.discard(lookup_id)
        self.lookup_pins.pop(lookup_id, None)
        self.unpinned_ids.append(lookup_id)

    def _cancel(self, lookup_id: str) -> None:
        self.storage_manager.cancelled.append(lookup_id)


class _FailOnceUnpinEngine(_RecordingEngine):
    def __init__(self) -> None:
        super().__init__(True)
        self.unpin_attempts = 0

    def lookup_unpin(self, lookup_id: str) -> None:
        self.unpin_attempts += 1
        if self.unpin_attempts == 1:
            raise RuntimeError("injected bypass unpin failure")
        super().lookup_unpin(lookup_id)


class _FailThenBlockUnpinEngine(_FailOnceUnpinEngine):
    def __init__(self) -> None:
        super().__init__()
        self.retry_entered = threading.Event()
        self.retry_release = threading.Event()

    def lookup_unpin(self, lookup_id: str) -> None:
        if self.unpin_attempts == 1:
            self.retry_entered.set()
            if not self.retry_release.wait(timeout=2.0):
                raise RuntimeError("timed out waiting to release bypass unpin")
        super().lookup_unpin(lookup_id)


class _ToggleUnpinEngine(_RecordingEngine):
    def __init__(self) -> None:
        super().__init__(True)
        self.cancel_enabled = threading.Event()
        self.unpin_attempts = 0

    def lookup_unpin(self, lookup_id: str) -> None:
        self.unpin_attempts += 1
        if not self.cancel_enabled.is_set():
            raise RuntimeError("injected unavailable bypass cleanup")
        super().lookup_unpin(lookup_id)


class _BlockingLookupEngine(_RecordingEngine):
    def __init__(self) -> None:
        super().__init__(True)
        self.lookup_entered = threading.Event()
        self.lookup_release = threading.Event()

    def lookup(self, **kwargs: Any) -> int:
        result = super().lookup(**kwargs)
        self.lookup_entered.set()
        if not self.lookup_release.wait(timeout=2.0):
            raise RuntimeError("timed out waiting to release engine lookup")
        return result


class _BlockingUnpinEngine(_RecordingEngine):
    def __init__(self) -> None:
        super().__init__(True)
        self.unpin_entered = threading.Event()
        self.unpin_release = threading.Event()

    def lookup_unpin(self, lookup_id: str) -> None:
        self.unpin_entered.set()
        if not self.unpin_release.wait(timeout=2.0):
            raise RuntimeError("timed out waiting to release engine unpin")
        super().lookup_unpin(lookup_id)


class _GenerationStorageManager:
    def __init__(self) -> None:
        self.unpinned: list[Any] = []
        self.fail_once: set[Any] = set()
        self.block_entered = threading.Event()
        self.block_release = threading.Event()
        self.block = False
        self.closed = False

    def batched_unpin(self, keys: list[Any], locations: list[str]) -> None:
        assert len(keys) == len(locations) == 1
        key = keys[0]
        if self.block:
            self.block_entered.set()
            if not self.block_release.wait(timeout=2.0):
                raise RuntimeError("timed out waiting to release unpin")
        if key in self.fail_once:
            self.fail_once.remove(key)
            raise RuntimeError("injected partial unpin failure")
        self.unpinned.append(key)

    def close(self) -> None:
        self.closed = True

    def drain_pending_put_cleanups(self) -> None:
        return None

    def pending_put_cleanup_count(self) -> int:
        return 0


class _EngineGenerationRecordingEngine(_RecordingEngine):
    """Recording engine using LMCacheEngine's real generation cleanup."""

    def __init__(self) -> None:
        super().__init__(True)
        self._lookup_pins_lock = threading.RLock()
        self.lookup_explicit_replay_fenced = False
        self.storage_manager = _GenerationStorageManager()
        self.gpu_connector = None
        self.async_loading = None
        self.lookup_cleanup_registered: set[str] = set()
        self.lookup_cleanup_pending: set[str] = set()
        self._lookup_cleanup_retry_timer: Optional[threading.Thread] = None
        self._lookup_cleanup_closed = False
        self.hidden_state_store: Any = None
        self.lmcache_worker: Any = None

    def lookup(self, **kwargs: Any) -> int:
        self.lookup_count += 1
        self.kwargs = kwargs
        if kwargs.get("pin") and self.lookup_result > 0:
            lookup_id = kwargs["lookup_id"]
            self.lookup_pins[lookup_id] = {"cpu": [f"pin-{lookup_id}"]}
        return self.lookup_result

    def _lookup_pins_guard(self):
        return LMCacheEngine._lookup_pins_guard(self)

    def _lookup_generation_state(self):
        return LMCacheEngine._lookup_generation_state(self)

    def _check_lookup_tombstone_capacity(
        self,
        lookup_id: str,
        generation: bytes,
    ) -> bool:
        return LMCacheEngine._check_lookup_tombstone_capacity(
            cast(LMCacheEngine, self),
            lookup_id,
            generation,
        )

    def _remember_lookup_cancelled_generation(
        self,
        lookup_id: str,
        generation: bytes,
    ) -> None:
        LMCacheEngine._remember_lookup_cancelled_generation(
            cast(LMCacheEngine, self),
            lookup_id,
            generation,
        )

    def _lookup_cleanup_state(self):
        return LMCacheEngine._lookup_cleanup_state(cast(LMCacheEngine, self))

    def _arm_lookup_cleanup_retry_locked(self) -> None:
        LMCacheEngine._arm_lookup_cleanup_retry_locked(cast(LMCacheEngine, self))

    def _retry_lookup_cleanups(self) -> None:
        LMCacheEngine._retry_lookup_cleanups(cast(LMCacheEngine, self))

    def register_lookup_cleanup(self, lookup_id: str) -> None:
        LMCacheEngine.register_lookup_cleanup(cast(LMCacheEngine, self), lookup_id)

    def schedule_lookup_unpin(self, lookup_id: str) -> None:
        LMCacheEngine.schedule_lookup_unpin(cast(LMCacheEngine, self), lookup_id)

    def _close_lookup_cleanups(self) -> None:
        LMCacheEngine._close_lookup_cleanups(cast(LMCacheEngine, self))

    def _lookup_unpin_once(self, lookup_id: str) -> None:
        LMCacheEngine._lookup_unpin_once(cast(LMCacheEngine, self), lookup_id)

    def lookup_unpin(self, lookup_id: str) -> None:
        LMCacheEngine.lookup_unpin(cast(LMCacheEngine, self), lookup_id)

    def close(self) -> None:
        LMCacheEngine.close(cast(LMCacheEngine, self))


class _BlockingGenerationLookupEngine(_EngineGenerationRecordingEngine):
    def __init__(self) -> None:
        super().__init__()
        self.lookup_entered = threading.Event()
        self.lookup_release = threading.Event()

    def lookup(self, **kwargs: Any) -> int:
        result = super().lookup(**kwargs)
        self.lookup_entered.set()
        if not self.lookup_release.wait(timeout=2.0):
            raise RuntimeError("timed out waiting to release generation lookup")
        return result


class _RaiseAfterGenerationPinEngine(_EngineGenerationRecordingEngine):
    def lookup(self, **kwargs: Any) -> int:
        super().lookup(**kwargs)
        raise RuntimeError("injected failure after pin publication")


class _RaiseBeforeGenerationPinEngine(_EngineGenerationRecordingEngine):
    def lookup(self, **kwargs: Any) -> int:
        self.lookup_count += 1
        self.kwargs = kwargs
        raise RuntimeError("injected failure before pin publication")


class _LoopbackTransport:
    world_size = 1

    def __init__(self, server: LMCacheLookupServer) -> None:
        self.server = server
        self.messages: list[list[Any]] = []

    def send_and_recv_all(self, message: list[Any]) -> list[bytes]:
        self.messages.append(message)
        return [self.server._process_request_frames(message)]

    def close(self) -> None:
        return


class _ScriptedCancelTransport(_LoopbackTransport):
    def __init__(
        self,
        server: LMCacheLookupServer,
        cancel_outcomes: list[Any],
        world_size: int = 1,
    ) -> None:
        super().__init__(server)
        self.cancel_outcomes = cancel_outcomes
        self.world_size = world_size
        self.outcomes_lock = threading.Lock()

    def append_cancel_outcome(self, outcome: Any) -> None:
        with self.outcomes_lock:
            self.cancel_outcomes.append(outcome)

    def send_and_recv_all(self, message: list[Any]) -> list[bytes]:
        self.messages.append(message)
        payload = message[0]
        if not (
            isinstance(payload, dict)
            and payload.get("schema") == "lmcache.lookup.control"
        ):
            response = self.server._process_request_frames(message)
            return [response for _ in range(self.world_size)]

        with self.outcomes_lock:
            outcome = self.cancel_outcomes.pop(0) if self.cancel_outcomes else []
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "server":
            response = self.server._process_request_frames(message)
            return [response for _ in range(self.world_size)]
        if outcome == "server_drop":
            self.server._process_request_frames(message)
            return []
        return outcome


class _ToggleCancelTransport(_LoopbackTransport):
    def __init__(self, server: LMCacheLookupServer) -> None:
        super().__init__(server)
        self.cancel_enabled = threading.Event()
        self.cancel_calls = 0
        self.closed = False

    def send_and_recv_all(self, message: list[Any]) -> list[bytes]:
        if self.closed:
            raise RuntimeError("transport is closed")
        self.messages.append(message)
        payload = message[0]
        if (
            isinstance(payload, dict)
            and payload.get("schema") == "lmcache.lookup.control"
        ):
            self.cancel_calls += 1
            if not self.cancel_enabled.is_set():
                return []
        return [self.server._process_request_frames(message)]

    def close(self) -> None:
        self.closed = True


class _BlockingCancelTransport(_ToggleCancelTransport):
    def __init__(
        self,
        server: LMCacheLookupServer,
        block_after: int = 1,
    ) -> None:
        super().__init__(server)
        self.block_after = block_after
        self.seen_cancels = 0
        self.cancel_entered = threading.Event()
        self.cancel_release = threading.Event()

    def send_and_recv_all(self, message: list[Any]) -> list[bytes]:
        payload = message[0]
        if (
            isinstance(payload, dict)
            and payload.get("schema") == "lmcache.lookup.control"
        ):
            self.seen_cancels += 1
            if self.seen_cancels >= self.block_after:
                self.cancel_entered.set()
                if not self.cancel_release.wait(timeout=2.0):
                    raise RuntimeError("timed out waiting to release cancel")
        return super().send_and_recv_all(message)


class _BlockingLookupTransport(_LoopbackTransport):
    def __init__(
        self,
        server: LMCacheLookupServer,
        raise_after_lookup: bool = False,
    ) -> None:
        super().__init__(server)
        self.raise_after_lookup = raise_after_lookup
        self.lookup_entered = threading.Event()
        self.lookup_release = threading.Event()
        self.closed = False

    def send_and_recv_all(self, message: list[Any]) -> list[bytes]:
        if self.closed:
            raise RuntimeError("transport is closed")
        self.messages.append(message)
        payload = message[0]
        if (
            isinstance(payload, dict)
            and payload.get("schema") == "lmcache.lookup.control"
        ):
            return [self.server._process_request_frames(message)]
        response = self.server._process_request_frames(message)
        self.lookup_entered.set()
        if not self.lookup_release.wait(timeout=2.0):
            raise RuntimeError("timed out waiting to release lookup response")
        if self.raise_after_lookup:
            raise RuntimeError("injected response loss after worker pin")
        return [response]

    def close(self) -> None:
        self.closed = True


class _HeldLookupTransport(_LoopbackTransport):
    """Hold one lookup while independently gating control acknowledgements."""

    def __init__(self, server: LMCacheLookupServer) -> None:
        super().__init__(server)
        self.cancel_enabled = threading.Event()
        self.held_lookup: Optional[list[Any]] = None
        self.lock = threading.Lock()

    def send_and_recv_all(self, message: list[Any]) -> list[bytes]:
        self.messages.append(message)
        payload = message[0]
        if (
            isinstance(payload, dict)
            and payload.get("schema") == "lmcache.lookup.control"
        ):
            if not self.cancel_enabled.is_set():
                return []
            return [self.server._process_request_frames(message)]
        with self.lock:
            assert self.held_lookup is None
            self.held_lookup = message
        return []

    def release_lookup(self) -> bytes:
        with self.lock:
            assert self.held_lookup is not None
            message = self.held_lookup
            self.held_lookup = None
        return self.server._process_request_frames(message)


class _LegacyOnlyTransport:
    """Model the previous server, which treats frame zero only as tokens."""

    world_size = 1

    def __init__(self) -> None:
        self.messages: list[list[Any]] = []
        self.lookup_count = 0

    def send_and_recv_all(self, message: list[Any]) -> list[bytes]:
        self.messages.append(message)
        self.lookup_count += 1
        try:
            if not isinstance(message[0], list):
                raise TypeError("legacy token request must be a list")
        except Exception:
            # The previous server catches request-processing failures and sends
            # no response; the client transport reports that as an empty list.
            return []
        return [(4).to_bytes(4, "big")]

    def close(self) -> None:
        return


class _MalformedExplicitResultTransport:
    world_size = 1

    def __init__(self, result_kind: str) -> None:
        self.result_kind = result_kind
        self.messages: list[list[Any]] = []

    def send_and_recv_all(self, message: list[Any]) -> list[bytes]:
        self.messages.append(message)
        generation = message[0]["generation"]
        raw_result = (4).to_bytes(4, "big")
        if self.result_kind == "raw":
            return [raw_result]
        version = 2 if self.result_kind == "version" else 1
        response_generation = (
            b"x" * 16 if self.result_kind == "generation" else generation
        )
        response = (
            b"LMCacheExplicitLookupResult"
            + bytes((version,))
            + response_generation
            + raw_result
        )
        if self.result_kind == "truncated":
            response = response[:-1]
        return [response]

    def close(self) -> None:
        return


class _SplitWorkerTransport:
    world_size = 2

    def __init__(
        self,
        first_server: LMCacheLookupServer,
        second_server: LMCacheLookupServer,
        lookup_failure: str,
    ) -> None:
        self.servers = (first_server, second_server)
        self.lookup_failure = lookup_failure
        self.messages: list[list[Any]] = []

    def send_and_recv_all(self, message: list[Any]) -> list[bytes]:
        self.messages.append(message)
        payload = message[0]
        if (
            isinstance(payload, dict)
            and payload.get("schema") == "lmcache.lookup.control"
        ):
            return [server._process_request_frames(message) for server in self.servers]

        first_response = self.servers[0]._process_request_frames(message)
        if self.lookup_failure == "empty":
            return []
        if self.lookup_failure == "partial":
            return [first_response]
        raise RuntimeError("injected lookup transport failure")

    def close(self) -> None:
        return


class _HashTokenDatabase:
    def process_tokens(self, token_ids, make_key=False):
        assert token_ids == [1, 2, 3, 4]
        assert not make_key
        yield 0, 2, 111
        yield 2, 4, 222


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("timed out waiting for asynchronous lookup cancellation")
        time.sleep(0.01)


def _sync_rpc_client(
    engine: _RecordingEngine,
) -> tuple[LMCacheLookupClient, _LoopbackTransport]:
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine  # type: ignore[assignment]
    server.enable_blending = True
    transport = _LoopbackTransport(server)

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


def _bypass_client(engine: _RecordingEngine) -> LMCacheBypassLookupClient:
    client = LMCacheBypassLookupClient.__new__(LMCacheBypassLookupClient)
    client.lmcache_engine = engine  # type: ignore[assignment]
    client.enable_blending = True
    client.reqs_status = {}
    client.reqs_fingerprints = {}
    client.pending_cancel_ids = set()
    return client


def test_sync_rpc_carries_explicit_spans_and_binds_cached_status() -> None:
    declaration = _declaration()
    request_configs = {"lmcache.tag.tenant": "a"}
    engine = _RecordingEngine(True)
    client, transport = _sync_rpc_client(engine)

    assert client.supports_explicit_spans()
    assert (
        client.lookup(
            [1, 2, 3, 4],
            "lookup",
            request_configs,
            explicit_spans=declaration,
        )
        == 4
    )
    generation = client.reqs_generations["lookup"]
    assert transport.messages == [
        [
            {
                "schema": "lmcache.lookup.explicit-spans",
                "version": 1,
                "generation": generation,
                "tokens": [1, 2, 3, 4],
                "token_count": 4,
                "cached_span": [0, 4],
                "spans": [[0, 2], [3, 4]],
            },
            "lookup",
            '{"lmcache.tag.tenant": "a"}',
        ]
    ]
    assert engine.kwargs == {
        "tokens": [1, 2, 3, 4],
        "lookup_id": "lookup",
        "pin": True,
        "request_configs": request_configs,
        "explicit_spans": declaration,
    }

    # An identical scheduler probe is idempotent and does not pin twice.
    assert (
        client.lookup(
            [1, 2, 3, 4],
            "lookup",
            request_configs,
            explicit_spans=declaration,
        )
        == 4
    )
    assert len(transport.messages) == 1
    assert engine.lookup_count == 1

    changed_declaration = ExplicitTokenSpans(
        4,
        TokenSpan(0, 4),
        (TokenSpan(0, 1), TokenSpan(3, 4)),
    )
    drifted_requests = [
        ([1, 2, 9, 4], request_configs, declaration),
        ([1, 2, 3, 4], {"lmcache.tag.tenant": "b"}, declaration),
        ([1, 2, 3, 4], request_configs, changed_declaration),
    ]
    for tokens, configs, spans in drifted_requests:
        with pytest.raises(ValueError, match="different lookup request content"):
            client.lookup(tokens, "lookup", configs, explicit_spans=spans)
    assert len(transport.messages) == 1
    assert engine.lookup_count == 1

    client.clear_lookup_status("lookup")
    assert client.lookup_cache("lookup") == -1
    assert "lookup" not in client.reqs_fingerprints
    assert client.reqs_generations["lookup"] == generation
    client.finish_lookup("lookup")
    assert "lookup" not in client.reqs_generations


def test_sync_and_bypass_reject_orphan_cached_status() -> None:
    engine = _RecordingEngine(True)
    sync_client, _ = _sync_rpc_client(engine)
    sync_client.reqs_status["orphan"] = 4

    with pytest.raises(ValueError, match="no request-content binding"):
        sync_client.validate_lookup_request([1, 2, 3, 4], "orphan")
    with pytest.raises(ValueError, match="no request-content binding"):
        sync_client.lookup([1, 2, 3, 4], "orphan")
    assert "orphan" not in sync_client.reqs_fingerprints

    bypass_client = LMCacheBypassLookupClient.__new__(LMCacheBypassLookupClient)
    bypass_client.lmcache_engine = engine  # type: ignore[assignment]
    bypass_client.reqs_status = {"orphan": 4}
    bypass_client.reqs_fingerprints = {}
    with pytest.raises(ValueError, match="no request-content binding"):
        bypass_client.validate_lookup_request([1, 2, 3, 4], "orphan")
    assert "orphan" not in bypass_client.reqs_fingerprints


def test_sync_rpc_preserves_legacy_wire_and_rejects_unsupported_envelope() -> None:
    engine = _RecordingEngine(True)
    client, transport = _sync_rpc_client(engine)

    assert client.lookup([1, 2, 3, 4], "legacy") == 4
    assert transport.messages == [[[1, 2, 3, 4], "legacy", ""]]
    assert engine.kwargs == {
        "tokens": [1, 2, 3, 4],
        "lookup_id": "legacy",
        "pin": True,
        "request_configs": None,
        "explicit_spans": None,
    }

    client.cancel_lookup("legacy")
    assert transport.messages[1] == [
        {
            "schema": "lmcache.lookup.control",
            "version": 1,
            "operation": "cancel",
            "generation": None,
        },
        "legacy",
        "",
    ]
    assert engine.pinned_ids == set()
    assert engine.unpinned_ids == ["legacy"]
    assert engine.storage_manager.cancelled == []
    assert client.lookup_cache("legacy") == -1

    # A repeated legacy cancel is harmless. Legacy request IDs do not carry a
    # generation, so the engine's idempotent unpin may be invoked again.
    client.cancel_lookup("legacy")
    assert engine.unpinned_ids == ["legacy", "legacy"]
    assert engine.storage_manager.cancelled == []

    unsupported_engine = _RecordingEngine(False)
    unsupported_server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    unsupported_server.lmcache_engine = unsupported_engine  # type: ignore[assignment]
    unsupported_server.enable_blending = True
    with pytest.raises(NotImplementedError, match="worker does not support"):
        unsupported_server._process_request_frames(
            [
                {
                    "schema": "lmcache.lookup.explicit-spans",
                    "version": 1,
                    "generation": b"g" * 16,
                    "tokens": [1, 2, 3, 4],
                    "token_count": 4,
                    "cached_span": [0, 4],
                    "spans": [[0, 2], [3, 4]],
                },
                "lookup",
                "",
            ]
        )
    assert unsupported_engine.lookup_count == 0


def test_sync_rpc_preserves_legacy_hash_wire() -> None:
    engine = _RecordingEngine(False)
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine  # type: ignore[assignment]
    server.enable_blending = False
    transport = _LoopbackTransport(server)
    client = LMCacheLookupClient.__new__(LMCacheLookupClient)
    client.config = SimpleNamespace(
        use_layerwise=False,
        enable_async_loading=False,
    )
    client.enable_blending = False
    client.token_database = _HashTokenDatabase()  # type: ignore[assignment]
    client.transport = transport  # type: ignore[assignment]
    client.reqs_status = {}
    client.reqs_fingerprints = {}
    client.reqs_generations = {}
    client.pending_cancel_ids = set()

    assert client.lookup([1, 2, 3, 4], "legacy-hash") == 4
    assert transport.messages == [[[111, 222], [2, 2], "legacy-hash", ""]]
    assert engine.kwargs == {
        "hashes": [111, 222],
        "offsets": [2, 2],
        "lookup_id": "legacy-hash",
        "pin": True,
        "request_configs": None,
    }

    client.cancel_lookup("legacy-hash")
    assert transport.messages[1] == [
        {
            "schema": "lmcache.lookup.control",
            "version": 1,
            "operation": "cancel",
            "generation": None,
        },
        [],
        "legacy-hash",
        "",
    ]
    assert engine.pinned_ids == set()
    assert engine.storage_manager.cancelled == []


def test_sync_cancel_retries_transient_failure_and_requires_typed_ack() -> None:
    engine = _RecordingEngine(True)
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine  # type: ignore[assignment]
    server.enable_blending = True
    transport = _ScriptedCancelTransport(server, [[], "server"])
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

    assert client.lookup([1, 2, 3, 4], "transient") == 4
    client.cancel_lookup("transient")

    assert client.pending_cancel_ids == {"transient"}
    _wait_until(lambda: client.lookup_cache("transient") == -1)
    assert len(transport.messages) == 3
    assert engine.unpinned_ids == ["transient"]
    assert engine.storage_manager.cancelled == []
    assert client.lookup_cache("transient") == -1
    assert client.pending_cancel_ids == set()


def test_sync_cancel_partial_ack_retries_all_workers() -> None:
    ack = b"LMCacheCancelAck\x01"
    engine = _RecordingEngine(True)
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine  # type: ignore[assignment]
    server.enable_blending = True
    transport = _ScriptedCancelTransport(
        server,
        [[ack], [ack, ack]],
        world_size=2,
    )
    client = LMCacheLookupClient.__new__(LMCacheLookupClient)
    client.enable_blending = True
    client.transport = transport  # type: ignore[assignment]
    client.reqs_status = {"partial": 4}
    client.reqs_fingerprints = {"partial": b"bound"}
    client.reqs_generations = {}
    client.pending_cancel_ids = set()

    client.cancel_lookup("partial")

    _wait_until(lambda: client.lookup_cache("partial") == -1)
    assert len(transport.messages) == 2
    assert client.lookup_cache("partial") == -1
    assert "partial" not in client.reqs_fingerprints
    assert client.pending_cancel_ids == set()


def test_sync_cancel_rejects_legacy_zero_ack_and_retains_tombstone() -> None:
    engine = _RecordingEngine(True)
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine  # type: ignore[assignment]
    server.enable_blending = True
    legacy_zero = (0).to_bytes(4, "big")
    transport = _ScriptedCancelTransport(server, [[legacy_zero]])
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

    assert client.lookup([1, 2, 3, 4], "old-server") == 4
    fingerprint = client.reqs_fingerprints["old-server"]
    client.cancel_lookup("old-server")

    assert client.lookup_cache("old-server") is None
    assert client.reqs_fingerprints["old-server"] == fingerprint
    assert client.pending_cancel_ids == {"old-server"}
    assert client.lookup([1, 2, 3, 4], "old-server") is None
    assert engine.pinned_ids == {"old-server"}

    # The retained tombstone owns a real future retry; no second scheduler
    # callback is needed when the server becomes available.
    transport.append_cancel_outcome("server")
    _wait_until(lambda: client.lookup_cache("old-server") == -1)
    assert client.lookup_cache("old-server") == -1
    assert "old-server" not in client.reqs_fingerprints
    assert client.pending_cancel_ids == set()
    assert engine.pinned_ids == set()


def test_sync_cancel_response_loss_is_exactly_once_for_explicit_generation() -> None:
    engine = _RecordingEngine(True)
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine  # type: ignore[assignment]
    server.enable_blending = True
    transport = _ScriptedCancelTransport(server, ["server_drop", "server"])
    client, _ = _sync_rpc_client(engine)
    client.transport = transport  # type: ignore[assignment]

    assert (
        client.lookup(
            [1, 2, 3, 4],
            "response-loss",
            explicit_spans=_declaration(),
        )
        == 4
    )
    client.cancel_lookup("response-loss")

    assert client.pending_cancel_ids == {"response-loss"}
    _wait_until(lambda: client.lookup_cache("response-loss") == -1)
    assert engine.unpinned_ids == ["response-loss"]
    assert engine.pinned_ids == set()


def test_sync_batch_transfer_preflights_all_ids_before_retiring_any() -> None:
    engine = _RecordingEngine(True)
    client, _ = _sync_rpc_client(engine)
    lookup_ids = ["sync-batch-a", "sync-batch-b"]
    for lookup_id in lookup_ids:
        assert (
            client.lookup(
                [1, 2, 3, 4],
                lookup_id,
                explicit_spans=_declaration(),
            )
            == 4
        )

    status_before = client.reqs_status.copy()
    fingerprints_before = client.reqs_fingerprints.copy()
    generations_before = client.reqs_generations.copy()
    client.pending_cancel_ids.add(lookup_ids[1])

    with pytest.raises(RuntimeError, match="cancellation is pending"):
        client.transfer_lookups(lookup_ids)

    assert client.reqs_status == status_before
    assert client.reqs_fingerprints == fingerprints_before
    assert client.reqs_generations == generations_before

    client.pending_cancel_ids.remove(lookup_ids[1])
    for lookup_id in lookup_ids:
        client.cancel_lookup(lookup_id)
    assert engine.pinned_ids == set()
    client.close()


def test_bypass_batch_transfer_preflights_all_ids_before_retiring_any() -> None:
    engine = _RecordingEngine(True)
    client = LMCacheBypassLookupClient.__new__(LMCacheBypassLookupClient)
    client.lmcache_engine = engine  # type: ignore[assignment]
    client.enable_blending = True
    client.reqs_status = {}
    client.reqs_fingerprints = {}
    client.pending_cancel_ids = set()
    lookup_ids = ["bypass-batch-a", "bypass-batch-b"]
    for lookup_id in lookup_ids:
        assert (
            client.lookup(
                [1, 2, 3, 4],
                lookup_id,
                explicit_spans=_declaration(),
            )
            == 4
        )

    status_before = client.reqs_status.copy()
    fingerprints_before = client.reqs_fingerprints.copy()
    client.pending_cancel_ids.add(lookup_ids[1])

    with pytest.raises(RuntimeError, match="cancellation is pending"):
        client.transfer_lookups(lookup_ids)

    assert client.reqs_status == status_before
    assert client.reqs_fingerprints == fingerprints_before

    client.pending_cancel_ids.remove(lookup_ids[1])
    for lookup_id in lookup_ids:
        client.cancel_lookup(lookup_id)
    assert engine.pinned_ids == set()
    client.close()


def test_uncertain_explicit_lookup_defers_until_cancel_ack() -> None:
    engine = _RecordingEngine(True)
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine  # type: ignore[assignment]
    server.enable_blending = True
    transport = _HeldLookupTransport(server)
    client, _ = _sync_rpc_client(engine)
    client.transport = transport  # type: ignore[assignment]
    lookup_id = "uncertain-held"

    assert (
        client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
        is None
    )
    generation = client.reqs_generations[lookup_id]
    assert client.pending_cancel_ids == {lookup_id}
    assert client.lookup_cache(lookup_id) is None
    assert len(transport.messages) == 2

    # Same-content scheduler probes remain deferred and do not send another
    # lookup. Content drift is still rejected while the exact retry owns ID.
    assert (
        client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
        is None
    )
    assert len(transport.messages) == 2
    with pytest.raises(ValueError, match="different lookup request content"):
        client.lookup(
            [1, 2, 9, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )

    # The uncertain lookup can win the first race and pin. The retained cancel
    # retry still owns its generation and converges once ACKs become available.
    response = transport.release_lookup()
    assert response.endswith((4).to_bytes(4, "big"))
    assert engine.pinned_ids == {lookup_id}
    transport.cancel_enabled.set()
    _wait_until(lambda: lookup_id not in client.pending_cancel_ids)

    assert engine.unpinned_ids == [lookup_id]
    assert engine.pinned_ids == set()
    assert lookup_id not in engine.lookup_pin_generations
    assert (lookup_id, generation) in engine.lookup_cancelled_generations
    assert client.lookup_cache(lookup_id) == -1
    assert lookup_id not in client.reqs_fingerprints
    assert lookup_id not in client.reqs_generations
    client.close()


def test_cancel_ack_suppresses_uncertain_delayed_explicit_lookup() -> None:
    engine = _RecordingEngine(True)
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine  # type: ignore[assignment]
    server.enable_blending = True
    transport = _HeldLookupTransport(server)
    transport.cancel_enabled.set()
    client, _ = _sync_rpc_client(engine)
    client.transport = transport  # type: ignore[assignment]
    lookup_id = "cancel-before-delayed"

    assert (
        client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
        is None
    )
    assert client.pending_cancel_ids == set()
    assert client.lookup_cache(lookup_id) == -1

    delayed_response = transport.release_lookup()
    assert delayed_response.endswith((0).to_bytes(4, "big"))
    assert engine.lookup_count == 0
    assert engine.pinned_ids == set()
    assert engine.lookup_pin_generations == {}
    client.close()


def test_explicit_cancel_survives_lookup_server_recreation() -> None:
    engine = _RecordingEngine(True)
    client, transport = _sync_rpc_client(engine)
    lookup_id = "server-recreated"

    assert (
        client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
        == 4
    )
    generation = client.reqs_generations[lookup_id]
    assert engine.lookup_pin_generations[lookup_id] == generation
    assert engine.pinned_ids == {lookup_id}

    replacement = LMCacheLookupServer.__new__(LMCacheLookupServer)
    replacement.lmcache_engine = engine  # type: ignore[assignment]
    replacement.enable_blending = True
    transport.server = replacement

    client.cancel_lookup(lookup_id)

    assert client.pending_cancel_ids == set()
    assert client.lookup_cache(lookup_id) == -1
    assert engine.unpinned_ids == [lookup_id]
    assert engine.pinned_ids == set()
    assert lookup_id not in engine.lookup_pin_generations
    assert (lookup_id, generation) in replacement.cancelled_explicit_generations


def test_cancel_tombstone_survives_lookup_server_recreation() -> None:
    engine = _RecordingEngine(True)
    generation = b"t" * 16
    lookup_id = "delayed-after-recreation"
    first_server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    first_server.lmcache_engine = engine  # type: ignore[assignment]
    first_server.enable_blending = True
    cancel_message: list[Any] = [
        {
            "schema": "lmcache.lookup.control",
            "version": 1,
            "operation": "cancel",
            "generation": generation,
        },
        lookup_id,
        "",
    ]

    assert first_server._process_request_frames(cancel_message) == (
        b"LMCacheCancelAck\x01"
    )
    assert (lookup_id, generation) in engine.lookup_cancelled_generations

    replacement = LMCacheLookupServer.__new__(LMCacheLookupServer)
    replacement.lmcache_engine = engine  # type: ignore[assignment]
    replacement.enable_blending = True
    delayed_lookup: list[Any] = [
        {
            "schema": "lmcache.lookup.explicit-spans",
            "version": 1,
            "generation": generation,
            "tokens": [1, 2, 3, 4],
            "token_count": 4,
            "cached_span": [0, 4],
            "spans": [[0, 2], [3, 4]],
        },
        lookup_id,
        "",
    ]

    response = replacement._process_request_frames(delayed_lookup)
    assert response.endswith((0).to_bytes(4, "big"))
    assert engine.lookup_count == 0
    assert engine.pinned_ids == set()
    assert (lookup_id, generation) in replacement.cancelled_explicit_generations


def test_worker_cleanup_tombstone_suppresses_replacement_server_replay() -> None:
    engine = _EngineGenerationRecordingEngine()
    first_server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    first_server.lmcache_engine = engine  # type: ignore[assignment]
    first_server.enable_blending = True
    lookup_id = "worker-cleanup-replay"
    generation = b"w" * 16
    message = _explicit_lookup_message(lookup_id, generation)

    assert first_server._process_request_frames(message).endswith(
        (4).to_bytes(4, "big")
    )
    assert engine.lookup_pin_generations[lookup_id] == generation
    assert lookup_id in engine.lookup_pins

    # This is the normal worker-finally path rather than a scheduler control
    # RPC. It must still leave durable replay suppression behind.
    engine.lookup_unpin(lookup_id)
    assert lookup_id not in engine.lookup_pins
    assert lookup_id not in engine.lookup_pin_generations
    assert (lookup_id, generation) in engine.lookup_cancelled_generations

    replacement = LMCacheLookupServer.__new__(LMCacheLookupServer)
    replacement.lmcache_engine = engine  # type: ignore[assignment]
    replacement.enable_blending = True
    assert replacement._process_request_frames(message).endswith((0).to_bytes(4, "big"))
    assert engine.lookup_count == 1
    assert engine.lookup_pins == {}


def test_post_pin_lookup_failure_retains_exact_owner_across_server_replacement() -> (
    None
):
    engine = _RaiseAfterGenerationPinEngine()
    first_server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    first_server.lmcache_engine = engine  # type: ignore[assignment]
    first_server.enable_blending = True
    lookup_id = "post-pin-generation-failure"
    generation = b"e" * 16
    lookup_message = _explicit_lookup_message(lookup_id, generation)

    with pytest.raises(RuntimeError, match="after pin publication"):
        first_server._process_request_frames(lookup_message)

    assert lookup_id in engine.lookup_pins
    assert engine.lookup_pin_generations[lookup_id] == generation
    assert first_server.explicit_lookup_generations[lookup_id] == generation
    assert (lookup_id, generation) not in engine.lookup_cancelled_generations

    replacement = LMCacheLookupServer.__new__(LMCacheLookupServer)
    replacement.lmcache_engine = engine  # type: ignore[assignment]
    replacement.enable_blending = True
    assert replacement._process_request_frames(lookup_message).endswith(
        (0).to_bytes(4, "big")
    )
    assert replacement.explicit_lookup_generations[lookup_id] == generation
    assert engine.lookup_count == 1

    assert replacement._process_request_frames(
        _explicit_cancel_message(lookup_id, generation)
    ) == (b"LMCacheCancelAck\x01")
    assert lookup_id not in engine.lookup_pins
    assert lookup_id not in engine.lookup_pin_generations
    assert lookup_id not in replacement.explicit_lookup_generations
    assert (lookup_id, generation) in engine.lookup_cancelled_generations

    first_server._prune_completed_explicit_generations()
    assert lookup_id not in first_server.explicit_lookup_generations


def test_pre_pin_lookup_failure_retires_and_tombstones_exact_generation() -> None:
    engine = _RaiseBeforeGenerationPinEngine()
    first_server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    first_server.lmcache_engine = engine  # type: ignore[assignment]
    first_server.enable_blending = True
    lookup_id = "pre-pin-generation-failure"
    generation = b"n" * 16
    lookup_message = _explicit_lookup_message(lookup_id, generation)

    with pytest.raises(RuntimeError, match="before pin publication"):
        first_server._process_request_frames(lookup_message)

    assert lookup_id not in engine.lookup_pins
    assert lookup_id not in engine.lookup_pin_generations
    assert lookup_id not in first_server.explicit_lookup_generations
    assert (lookup_id, generation) in engine.lookup_cancelled_generations

    replacement = LMCacheLookupServer.__new__(LMCacheLookupServer)
    replacement.lmcache_engine = engine  # type: ignore[assignment]
    replacement.enable_blending = True
    assert replacement._process_request_frames(lookup_message).endswith(
        (0).to_bytes(4, "big")
    )
    assert engine.lookup_count == 1


def test_client_uncertain_cancel_cleans_post_pin_engine_failure() -> None:
    engine = _RaiseAfterGenerationPinEngine()
    client, transport = _sync_rpc_client(engine)
    lookup_id = "client-post-pin-generation-failure"

    assert (
        client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
        is None
    )
    generation = next(
        generation
        for cancelled_id, generation in engine.lookup_cancelled_generations
        if cancelled_id == lookup_id
    )
    assert client.pending_cancel_ids == set()
    assert lookup_id not in engine.lookup_pins
    assert lookup_id not in engine.lookup_pin_generations
    assert lookup_id not in transport.server.explicit_lookup_generations
    assert (lookup_id, generation) in engine.lookup_cancelled_generations
    assert engine.storage_manager.unpinned == [f"pin-{lookup_id}"]
    client.close()


def test_partial_generation_unpin_retains_owner_until_retry_ack() -> None:
    engine = _EngineGenerationRecordingEngine()
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine  # type: ignore[assignment]
    server.enable_blending = True
    lookup_id = "partial-generation-unpin"
    generation = b"p" * 16

    assert server._process_request_frames(
        _explicit_lookup_message(lookup_id, generation)
    ).endswith((4).to_bytes(4, "big"))
    pin_key = engine.lookup_pins[lookup_id]["cpu"][0]
    engine.storage_manager.fail_once.add(pin_key)

    with pytest.raises(RuntimeError, match="partial unpin"):
        server._process_request_frames(_explicit_cancel_message(lookup_id, generation))

    assert engine.lookup_pins[lookup_id] == {"cpu": [pin_key]}
    assert engine.lookup_pin_generations[lookup_id] == generation
    assert (lookup_id, generation) not in engine.lookup_cancelled_generations

    assert (
        server._process_request_frames(_explicit_cancel_message(lookup_id, generation))
        == b"LMCacheCancelAck\x01"
    )
    assert lookup_id not in engine.lookup_pins
    assert lookup_id not in engine.lookup_pin_generations
    assert (lookup_id, generation) in engine.lookup_cancelled_generations


def test_worker_owned_partial_unpin_retries_to_exact_tombstone() -> None:
    engine = _EngineGenerationRecordingEngine()
    lookup_id = "worker-owned-partial-unpin"
    generation = b"r" * 16
    pin_key = "worker-retry-pin"
    engine.lookup_pins[lookup_id] = {"cpu": [pin_key]}
    engine.lookup_pin_generations[lookup_id] = generation
    engine.storage_manager.fail_once.add(pin_key)

    engine.register_lookup_cleanup(lookup_id)
    engine.schedule_lookup_unpin(lookup_id)

    assert engine.lookup_cleanup_registered == {lookup_id}
    assert engine.lookup_cleanup_pending == {lookup_id}
    assert engine.lookup_pins[lookup_id] == {"cpu": [pin_key]}
    _wait_until(lambda: lookup_id not in engine.lookup_cleanup_pending)

    assert engine.lookup_cleanup_registered == set()
    assert lookup_id not in engine.lookup_pins
    assert lookup_id not in engine.lookup_pin_generations
    assert engine.storage_manager.unpinned == [pin_key]
    assert (lookup_id, generation) in engine.lookup_cancelled_generations


def test_lookup_cleanup_close_waits_for_inflight_retry_backend_call() -> None:
    engine = _EngineGenerationRecordingEngine()
    lookup_id = "cleanup-close-barrier"
    generation = b"q" * 16
    pin_key = "blocked-retry-pin"
    engine.lookup_pins[lookup_id] = {"cpu": [pin_key]}
    engine.lookup_pin_generations[lookup_id] = generation
    engine.lookup_cleanup_registered = {lookup_id}
    engine.lookup_cleanup_pending = {lookup_id}
    engine._lookup_cleanup_closed = False
    engine.storage_manager.block = True
    engine.hidden_state_store = None
    engine.lmcache_worker = None

    retry_thread = threading.Thread(
        target=engine._retry_lookup_cleanups,
        name="deterministic-lookup-cleanup-retry",
    )
    engine._lookup_cleanup_retry_timer = retry_thread
    retry_thread.start()
    assert engine.storage_manager.block_entered.wait(timeout=2.0)

    close_returned = threading.Event()
    close_thread = threading.Thread(
        target=_call_and_signal,
        args=(engine.close, close_returned),
    )
    close_thread.start()
    assert not close_returned.wait(timeout=0.05)
    assert not engine.storage_manager.closed

    engine.storage_manager.block_release.set()
    retry_thread.join(timeout=2.0)
    close_thread.join(timeout=2.0)
    assert not retry_thread.is_alive()
    assert not close_thread.is_alive()
    assert close_returned.is_set()
    assert engine.storage_manager.closed
    assert lookup_id not in engine.lookup_pins
    assert (lookup_id, generation) in engine.lookup_cancelled_generations


def test_cancel_transaction_blocks_replay_until_tombstone_is_durable() -> None:
    engine = _EngineGenerationRecordingEngine()
    first_server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    first_server.lmcache_engine = engine  # type: ignore[assignment]
    first_server.enable_blending = True
    replacement = LMCacheLookupServer.__new__(LMCacheLookupServer)
    replacement.lmcache_engine = engine  # type: ignore[assignment]
    replacement.enable_blending = True
    lookup_id = "cancel-first-replay"
    generation = b"c" * 16
    lookup_message = _explicit_lookup_message(lookup_id, generation)
    cancel_message = _explicit_cancel_message(lookup_id, generation)
    assert first_server._process_request_frames(lookup_message).endswith(
        (4).to_bytes(4, "big")
    )

    engine.storage_manager.block = True
    cancel_result: list[bytes] = []
    cancel_thread = threading.Thread(
        target=lambda: cancel_result.append(
            replacement._process_request_frames(cancel_message)
        )
    )
    cancel_thread.start()
    assert engine.storage_manager.block_entered.wait(timeout=2.0)

    replay_result: list[bytes] = []
    replay_thread = threading.Thread(
        target=lambda: replay_result.append(
            first_server._process_request_frames(lookup_message)
        )
    )
    replay_thread.start()
    time.sleep(0.05)
    assert replay_thread.is_alive()

    engine.storage_manager.block_release.set()
    cancel_thread.join(timeout=2.0)
    replay_thread.join(timeout=2.0)
    assert not cancel_thread.is_alive()
    assert not replay_thread.is_alive()
    assert cancel_result == [b"LMCacheCancelAck\x01"]
    assert replay_result[0].endswith((0).to_bytes(4, "big"))
    assert engine.lookup_count == 1
    assert engine.lookup_pins == {}
    assert engine.lookup_pin_generations == {}


def test_lookup_transaction_cannot_publish_pins_after_cancel_ack() -> None:
    engine = _BlockingGenerationLookupEngine()
    old_server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    old_server.lmcache_engine = engine  # type: ignore[assignment]
    old_server.enable_blending = True
    replacement = LMCacheLookupServer.__new__(LMCacheLookupServer)
    replacement.lmcache_engine = engine  # type: ignore[assignment]
    replacement.enable_blending = True
    lookup_id = "lookup-first-cancel"
    generation = b"l" * 16
    lookup_result: list[bytes] = []
    lookup_thread = threading.Thread(
        target=lambda: lookup_result.append(
            old_server._process_request_frames(
                _explicit_lookup_message(lookup_id, generation)
            )
        )
    )
    lookup_thread.start()
    assert engine.lookup_entered.wait(timeout=2.0)

    cancel_result: list[bytes] = []
    cancel_thread = threading.Thread(
        target=lambda: cancel_result.append(
            replacement._process_request_frames(
                _explicit_cancel_message(lookup_id, generation)
            )
        )
    )
    cancel_thread.start()
    time.sleep(0.05)
    assert cancel_thread.is_alive()

    engine.lookup_release.set()
    lookup_thread.join(timeout=2.0)
    cancel_thread.join(timeout=2.0)
    assert not lookup_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert lookup_result[0].endswith((4).to_bytes(4, "big"))
    assert cancel_result == [b"LMCacheCancelAck\x01"]
    # The positive lookup linearizes before cancel. The ACK linearizes after
    # its publication and cleanup; no later pin can appear behind that ACK.
    assert engine.lookup_pins == {}
    assert engine.lookup_pin_generations == {}
    assert (lookup_id, generation) in engine.lookup_cancelled_generations


def test_tombstone_capacity_exhaustion_permanently_fences_explicit_lookup() -> None:
    engine = _EngineGenerationRecordingEngine()
    for index in range(4096):
        generation = index.to_bytes(16, "big")
        engine.lookup_cancelled_generations[(f"old-{index}", generation)] = None
    oldest_key = next(iter(engine.lookup_cancelled_generations))
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine  # type: ignore[assignment]
    server.enable_blending = True

    overflow_generation = (4096).to_bytes(16, "big")
    assert (
        server._process_request_frames(
            _explicit_cancel_message("overflow", overflow_generation)
        )
        == b"LMCacheCancelAck\x01"
    )
    assert engine.lookup_explicit_replay_fenced
    assert len(engine.lookup_cancelled_generations) == 4096
    assert oldest_key in engine.lookup_cancelled_generations

    oldest_lookup_id, oldest_generation = oldest_key
    oldest_response = server._process_request_frames(
        _explicit_lookup_message(oldest_lookup_id, oldest_generation)
    )
    new_response = server._process_request_frames(
        _explicit_lookup_message("brand-new", b"n" * 16)
    )
    assert oldest_response.endswith((0).to_bytes(4, "big"))
    assert new_response.endswith((0).to_bytes(4, "big"))
    assert engine.lookup_count == 0
    assert engine.lookup_pins == {}


def test_lookup_server_close_fails_until_request_thread_quiesces() -> None:
    class _ControllableThread:
        def __init__(self) -> None:
            self.alive = True
            self.join_calls = 0

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            assert timeout == 2.0
            self.join_calls += 1

    class _RecordingCloseTransport:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    thread = _ControllableThread()
    transport = _RecordingCloseTransport()
    server.running = True
    server.thread = thread  # type: ignore[assignment]
    server.transport = transport  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="did not quiesce"):
        server.close()
    assert not server.running
    assert thread.join_calls == 1
    assert transport.close_calls == 0

    thread.alive = False
    server.close()
    assert transport.close_calls == 1


def test_zero_hit_explicit_generation_cancel_is_safe_ack() -> None:
    engine = _RecordingEngine(True, lookup_result=0)
    client, transport = _sync_rpc_client(engine)

    assert (
        client.lookup(
            [1, 2, 3, 4],
            "zero-hit",
            explicit_spans=_declaration(),
        )
        == 0
    )
    generation = client.reqs_generations["zero-hit"]
    client.cancel_lookup("zero-hit")

    assert client.lookup_cache("zero-hit") == -1
    assert client.pending_cancel_ids == set()
    assert engine.unpinned_ids == []
    assert (
        "zero-hit",
        generation,
    ) in transport.server.cancelled_explicit_generations


def test_explicit_generation_survives_status_clear_and_prevents_aba_unpin() -> None:
    engine = _RecordingEngine(True)
    client, transport = _sync_rpc_client(engine)

    assert (
        client.lookup(
            [1, 2, 3, 4],
            "reused-id",
            explicit_spans=_declaration(),
        )
        == 4
    )
    old_generation = client.reqs_generations["reused-id"]

    # ``update_state_after_alloc`` clears only the scheduler probe status.
    # Abort must still address the exact worker-side pin generation.
    client.clear_lookup_status("reused-id")
    assert client.reqs_generations["reused-id"] == old_generation
    client.cancel_lookup("reused-id")
    assert engine.unpinned_ids == ["reused-id"]

    assert (
        client.lookup(
            [1, 2, 3, 4],
            "reused-id",
            explicit_spans=_declaration(),
        )
        == 4
    )
    new_generation = client.reqs_generations["reused-id"]
    assert new_generation != old_generation

    stale_cancel = [
        {
            "schema": "lmcache.lookup.control",
            "version": 1,
            "operation": "cancel",
            "generation": old_generation,
        },
        "reused-id",
        "",
    ]
    assert transport.server._process_request_frames(stale_cancel) == (
        b"LMCacheCancelAck\x01"
    )
    assert engine.pinned_ids == {"reused-id"}
    assert engine.unpinned_ids == ["reused-id"]

    unknown_cancel = [
        {
            "schema": "lmcache.lookup.control",
            "version": 1,
            "operation": "cancel",
            "generation": b"x" * 16,
        },
        "reused-id",
        "",
    ]
    assert transport.server._process_request_frames(unknown_cancel) == (
        b"LMCacheCancelAck\x01"
    )
    assert engine.pinned_ids == {"reused-id"}

    delayed_unknown_lookup = [
        {
            "schema": "lmcache.lookup.explicit-spans",
            "version": 1,
            "generation": b"x" * 16,
            "tokens": [1, 2, 3, 4],
            "token_count": 4,
            "cached_span": [0, 4],
            "spans": [[0, 2], [3, 4]],
        },
        "reused-id",
        "",
    ]
    assert transport.server._process_request_frames(delayed_unknown_lookup).endswith(
        (0).to_bytes(4, "big")
    )
    assert engine.lookup_count == 2
    assert engine.pinned_ids == {"reused-id"}

    client.cancel_lookup("reused-id")
    assert engine.pinned_ids == set()
    assert engine.unpinned_ids == ["reused-id", "reused-id"]


def test_server_rejects_conflicting_active_generation_before_engine_lookup() -> None:
    engine = _RecordingEngine(True)
    client, transport = _sync_rpc_client(engine)
    assert (
        client.lookup(
            [1, 2, 3, 4],
            "active-id",
            explicit_spans=_declaration(),
        )
        == 4
    )
    original_message = transport.messages[0]
    original_generation = client.reqs_generations["active-id"]

    # Transport-level replay fails closed rather than manufacturing a second
    # positive result whose sole pin owner could disappear before delivery.
    replay_response = transport.server._process_request_frames(original_message)
    assert replay_response.endswith((0).to_bytes(4, "big"))
    assert engine.lookup_count == 1
    assert engine.pinned_ids == {"active-id"}

    conflicting_payload = dict(cast(dict[str, Any], original_message[0]))
    conflicting_payload["generation"] = b"z" * 16
    conflicting_message: list[Any] = [conflicting_payload, "active-id", ""]
    assert conflicting_payload["generation"] != original_generation
    with pytest.raises(ValueError, match="different active generation"):
        transport.server._process_request_frames(conflicting_message)

    assert engine.lookup_count == 1
    assert engine.pinned_ids == {"active-id"}
    assert transport.server.explicit_lookup_generations["active-id"] == (
        original_generation
    )
    client.cancel_lookup("active-id")


def test_server_rejects_legacy_and_explicit_pin_owner_collision() -> None:
    engine = _RecordingEngine(True)
    client, transport = _sync_rpc_client(engine)
    assert client.lookup([1, 2, 3, 4], "mixed-id") == 4
    assert engine.lookup_count == 1
    assert engine.pinned_ids == {"mixed-id"}

    explicit_message: list[Any] = [
        {
            "schema": "lmcache.lookup.explicit-spans",
            "version": 1,
            "generation": b"e" * 16,
            "tokens": [1, 2, 3, 4],
            "token_count": 4,
            "cached_span": [0, 4],
            "spans": [[0, 2], [3, 4]],
        },
        "mixed-id",
        "",
    ]
    with pytest.raises(ValueError, match="legacy pin ownership"):
        transport.server._process_request_frames(explicit_message)
    assert engine.lookup_count == 1
    assert engine.pinned_ids == {"mixed-id"}

    client.cancel_lookup("mixed-id")
    assert engine.pinned_ids == set()

    assert (
        client.lookup(
            [1, 2, 3, 4],
            "mixed-id",
            explicit_spans=_declaration(),
        )
        == 4
    )
    legacy_message: list[Any] = [[1, 2, 3, 4], "mixed-id", ""]
    with pytest.raises(ValueError, match="active pin ownership"):
        transport.server._process_request_frames(legacy_message)
    assert engine.lookup_count == 2
    assert engine.pinned_ids == {"mixed-id"}
    client.cancel_lookup("mixed-id")


@pytest.mark.parametrize("lookup_failure", ["empty", "partial", "exception"])
def test_explicit_cancel_converges_when_worker_missed_lookup(
    lookup_failure: str,
) -> None:
    engines = [_RecordingEngine(True), _RecordingEngine(True)]
    servers: list[LMCacheLookupServer] = []
    for engine in engines:
        server = LMCacheLookupServer.__new__(LMCacheLookupServer)
        server.lmcache_engine = engine  # type: ignore[assignment]
        server.enable_blending = True
        servers.append(server)
    transport = _SplitWorkerTransport(servers[0], servers[1], lookup_failure)
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

    assert (
        client.lookup(
            [1, 2, 3, 4],
            f"split-{lookup_failure}",
            explicit_spans=_declaration(),
        )
        is None
    )
    lookup_message = transport.messages[0]
    generation = cast(dict[str, Any], lookup_message[0])["generation"]
    assert isinstance(generation, bytes)
    # Rank zero processed and pinned before the empty/partial/error response;
    # the uncertainty path immediately generation-cancelled it.
    assert engines[0].pinned_ids == set()
    assert engines[1].pinned_ids == set()

    assert client.pending_cancel_ids == set()
    assert engines[0].pinned_ids == set()
    assert engines[0].unpinned_ids == [f"split-{lookup_failure}"]
    assert engines[1].unpinned_ids == []
    assert (
        f"split-{lookup_failure}",
        generation,
    ) in servers[1].cancelled_explicit_generations

    # If the missing worker observes the original RPC after cancellation, the
    # tombstone suppresses the lookup and it cannot recreate a pin.
    delayed_response = servers[1]._process_request_frames(lookup_message)
    assert delayed_response.endswith((0).to_bytes(4, "big"))
    assert engines[1].lookup_count == 0
    assert engines[1].pinned_ids == set()
    client.close()


def test_new_explicit_client_fails_closed_against_legacy_server() -> None:
    transport = _LegacyOnlyTransport()
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

    assert (
        client.lookup(
            [1, 2, 3, 4],
            "new-to-old",
            explicit_spans=_declaration(),
        )
        is None
    )
    assert isinstance(transport.messages[0][0], dict)
    assert client.lookup_cache("new-to-old") is None

    # The uncertain compatibility probe is still idempotent and content-bound.
    assert (
        client.lookup(
            [1, 2, 3, 4],
            "new-to-old",
            explicit_spans=_declaration(),
        )
        is None
    )
    explicit_messages = [
        message
        for message in transport.messages
        if message[0].get("schema") == "lmcache.lookup.explicit-spans"
    ]
    assert len(explicit_messages) == 1
    with pytest.raises(ValueError, match="different lookup request content"):
        client.lookup(
            [1, 2, 9, 4],
            "new-to-old",
            explicit_spans=_declaration(),
        )
    client.force_close()


@pytest.mark.parametrize(
    "result_kind",
    ["raw", "version", "generation", "truncated"],
)
def test_explicit_client_rejects_untyped_or_mismatched_result(
    result_kind: str,
) -> None:
    transport = _MalformedExplicitResultTransport(result_kind)
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

    # A four-byte legacy value must never masquerade as a successful explicit
    # result. The version and generation are part of the result contract too.
    assert (
        client.lookup(
            [1, 2, 3, 4],
            f"malformed-{result_kind}",
            explicit_spans=_declaration(),
        )
        is None
    )
    explicit_messages = [
        message
        for message in transport.messages
        if message[0].get("schema") == "lmcache.lookup.explicit-spans"
    ]
    assert len(explicit_messages) == 1
    assert client.lookup_cache(f"malformed-{result_kind}") is None
    client.force_close()


@pytest.mark.parametrize(
    "frames",
    [
        [[1, 2, 3, 4], "lookup"],
        [[1, 2, 3, 4], "lookup", "", "extra"],
        [[1, 2, 3, 4], 7, ""],
        [[1, 2, 3, 4], "lookup", object()],
        [[1, 2, 3, 4], "lookup", "[]"],
        [
            {
                "schema": "lmcache.lookup.explicit-spans",
                "version": True,
                "generation": b"g" * 16,
                "tokens": [1, 2, 3, 4],
                "token_count": 4,
                "cached_span": [0, 4],
                "spans": [[0, 2], [3, 4]],
            },
            "lookup",
            "",
        ],
        [
            {
                "schema": "lmcache.lookup.explicit-spans",
                "version": 2,
                "generation": b"g" * 16,
                "tokens": [1, 2, 3, 4],
                "token_count": 4,
                "cached_span": [0, 4],
                "spans": [[0, 2], [3, 4]],
            },
            "lookup",
            "",
        ],
        [
            {
                "schema": "lmcache.lookup.explicit-spans",
                "version": 1,
                "generation": b"g" * 16,
                "tokens": [1, True, 3, 4],
                "token_count": 4,
                "cached_span": [0, 4],
                "spans": [[0, 2], [3, 4]],
            },
            "lookup",
            "",
        ],
        [
            {
                "schema": "lmcache.lookup.explicit-spans",
                "version": 1,
                "generation": b"g" * 16,
                "tokens": [1, "2", 3, 4],
                "token_count": 4,
                "cached_span": [0, 4],
                "spans": [[0, 2], [3, 4]],
            },
            "lookup",
            "",
        ],
        [
            {
                "schema": "lmcache.lookup.explicit-spans",
                "version": 1,
                "generation": b"g" * 16,
                "tokens": [1, [2], 3, 4],
                "token_count": 4,
                "cached_span": [0, 4],
                "spans": [[0, 2], [3, 4]],
            },
            "lookup",
            "",
        ],
        [
            {
                "schema": "lmcache.lookup.control",
                "version": True,
                "operation": "cancel",
            },
            "lookup",
            "",
        ],
        [
            {
                "schema": "lmcache.lookup.control",
                "version": 1,
                "operation": "release-something-else",
            },
            "lookup",
            "",
        ],
        [
            {
                "schema": "lmcache.lookup.control",
                "version": 1,
                "operation": "cancel",
                "generation": "not-primitive-bytes",
            },
            "lookup",
            "",
        ],
        [
            {
                "schema": "lmcache.lookup.control",
                "version": 1,
                "operation": "cancel",
            },
            "lookup",
            '{"not": "allowed"}',
        ],
    ],
)
def test_sync_server_rejects_malformed_frames_before_engine_lookup(
    frames: list[Any],
) -> None:
    engine = _RecordingEngine(True)
    server = LMCacheLookupServer.__new__(LMCacheLookupServer)
    server.lmcache_engine = engine  # type: ignore[assignment]
    server.enable_blending = True

    with pytest.raises((TypeError, ValueError)):
        server._process_request_frames(frames)
    assert engine.lookup_count == 0
    assert engine.unpinned_ids == []
    assert engine.storage_manager.cancelled == []


def test_bypass_client_forwards_explicit_spans_only_when_capable() -> None:
    declaration = _declaration()
    engine = _RecordingEngine(True)
    client = LMCacheBypassLookupClient.__new__(LMCacheBypassLookupClient)
    client.lmcache_engine = engine  # type: ignore[assignment]
    client.reqs_status = {}
    client.reqs_fingerprints = {}

    assert client.supports_explicit_spans()
    assert client.lookup_cache("lookup") == -1
    assert (
        client.lookup(
            [1, 2, 3, 4],
            "lookup",
            {"lmcache.tag.tenant": "a"},
            explicit_spans=declaration,
        )
        == 4
    )
    assert engine.kwargs == {
        "tokens": [1, 2, 3, 4],
        "lookup_id": "lookup",
        "pin": True,
        "request_configs": {"lmcache.tag.tenant": "a"},
        "explicit_spans": declaration,
    }
    assert client.lookup_cache("lookup") == 4

    # A repeated scheduler probe must reuse the status instead of pinning the
    # same worker-side objects a second time.
    assert (
        client.lookup(
            [1, 2, 3, 4],
            "lookup",
            {"lmcache.tag.tenant": "a"},
            explicit_spans=declaration,
        )
        == 4
    )
    assert engine.lookup_count == 1

    with pytest.raises(ValueError, match="different lookup request content"):
        client.lookup(
            [1, 2, 9, 4],
            "lookup",
            {"lmcache.tag.tenant": "a"},
            explicit_spans=declaration,
        )
    assert engine.lookup_count == 1

    client.clear_lookup_status("lookup")
    assert client.lookup_cache("lookup") == -1
    assert "lookup" not in client.reqs_fingerprints

    unsupported_engine = _RecordingEngine(False)
    client.lmcache_engine = unsupported_engine  # type: ignore[assignment]
    with pytest.raises(NotImplementedError, match="layerwise CacheBlend"):
        client.lookup(
            [1, 2, 3, 4],
            "lookup",
            explicit_spans=declaration,
        )
    assert unsupported_engine.kwargs is None


def test_bypass_cancel_retains_binding_until_retry_unpins() -> None:
    engine = _FailOnceUnpinEngine()
    client = LMCacheBypassLookupClient.__new__(LMCacheBypassLookupClient)
    client.lmcache_engine = engine  # type: ignore[assignment]
    client.reqs_status = {}
    client.reqs_fingerprints = {}
    client.pending_cancel_ids = set()
    lookup_id = "bypass-retry"

    assert (
        client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
        == 4
    )
    fingerprint = client.reqs_fingerprints[lookup_id]
    client.cancel_lookup(lookup_id)

    assert client.pending_cancel_ids == {lookup_id}
    assert client.lookup_cache(lookup_id) is None
    assert client.reqs_fingerprints[lookup_id] == fingerprint
    assert engine.pinned_ids == {lookup_id}
    assert (
        client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
        is None
    )

    _wait_until(lambda: client.lookup_cache(lookup_id) == -1)
    assert engine.unpin_attempts == 2
    assert engine.pinned_ids == set()
    assert client.pending_cancel_ids == set()
    assert lookup_id not in client.reqs_fingerprints
    client.close()


def test_bypass_close_waits_for_inflight_cancel_retry() -> None:
    engine = _FailThenBlockUnpinEngine()
    client = LMCacheBypassLookupClient.__new__(LMCacheBypassLookupClient)
    client.lmcache_engine = engine  # type: ignore[assignment]
    client.enable_blending = True
    client.reqs_status = {}
    client.reqs_fingerprints = {}
    client.pending_cancel_ids = set()
    lookup_id = "bypass-close-barrier"

    assert client.lookup([1, 2, 3, 4], lookup_id) == 4
    client.cancel_lookup(lookup_id)
    assert engine.retry_entered.wait(timeout=2.0)

    close_returned = threading.Event()

    def close_client() -> None:
        client.close()
        close_returned.set()

    close_thread = threading.Thread(target=close_client)
    close_thread.start()
    assert not close_returned.wait(timeout=0.05)

    engine.retry_release.set()
    close_thread.join(timeout=2.0)
    assert not close_thread.is_alive()
    assert close_returned.is_set()
    _wait_until(lambda: lookup_id not in client.pending_cancel_ids)
    assert engine.unpin_attempts == 2
    assert engine.pinned_ids == set()


def test_sync_close_actively_drains_queued_cancel() -> None:
    engine = _RecordingEngine(True)
    client, transport_base = _sync_rpc_client(engine)
    transport = _ToggleCancelTransport(transport_base.server)
    client.transport = transport  # type: ignore[assignment]
    lookup_id = "sync-close-drain"

    assert (
        client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
        == 4
    )
    client.cancel_lookup(lookup_id)
    assert client.pending_cancel_ids == {lookup_id}
    assert engine.pinned_ids == {lookup_id}

    transport.cancel_enabled.set()
    client.close()

    assert transport.closed
    assert client.pending_cancel_ids == set()
    assert client.reqs_fingerprints == {}
    assert client.reqs_generations == {}
    assert engine.pinned_ids == set()
    assert client._cancel_retry_timer is None


def test_bypass_close_actively_drains_queued_cancel() -> None:
    engine = _ToggleUnpinEngine()
    client = _bypass_client(engine)
    lookup_id = "bypass-close-drain"

    assert client.lookup([1, 2, 3, 4], lookup_id) == 4
    client.cancel_lookup(lookup_id)
    assert client.pending_cancel_ids == {lookup_id}
    assert engine.pinned_ids == {lookup_id}

    engine.cancel_enabled.set()
    client.close()

    assert client.pending_cancel_ids == set()
    assert client.reqs_fingerprints == {}
    assert engine.pinned_ids == set()
    assert client._cancel_retry_timer is None


def test_sync_close_timeout_rolls_back_to_live_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.lookup_client.lmcache_lookup_client."
        "_LOOKUP_CANCEL_CLOSE_TIMEOUT_SECONDS",
        0.03,
    )
    engine = _RecordingEngine(True)
    client, transport_base = _sync_rpc_client(engine)
    transport = _ToggleCancelTransport(transport_base.server)
    client.transport = transport  # type: ignore[assignment]
    lookup_id = "sync-close-timeout"
    tokens = [1, 2, 3, 4]

    assert client.lookup(tokens, lookup_id, explicit_spans=_declaration()) == 4
    generation = client.reqs_generations[lookup_id]
    fingerprint = client.reqs_fingerprints[lookup_id]
    client.cancel_lookup(lookup_id)

    with pytest.raises(RuntimeError, match="close timed out"):
        client.close()

    assert not transport.closed
    assert client.pending_cancel_ids == {lookup_id}
    assert client.reqs_generations[lookup_id] == generation
    assert client.reqs_fingerprints[lookup_id] == fingerprint
    assert client.lookup(tokens, lookup_id, explicit_spans=_declaration()) is None
    assert client._cancel_retry_timer is not None

    transport.cancel_enabled.set()
    _wait_until(lambda: lookup_id not in client.pending_cancel_ids)
    client.close()
    assert transport.closed
    assert engine.pinned_ids == set()


def test_bypass_close_timeout_rolls_back_to_live_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.lookup_client.lmcache_lookup_client_bypass."
        "_BYPASS_CANCEL_CLOSE_TIMEOUT_SECONDS",
        0.03,
    )
    engine = _ToggleUnpinEngine()
    client = _bypass_client(engine)
    lookup_id = "bypass-close-timeout"
    tokens = [1, 2, 3, 4]

    assert client.lookup(tokens, lookup_id) == 4
    fingerprint = client.reqs_fingerprints[lookup_id]
    client.cancel_lookup(lookup_id)

    with pytest.raises(RuntimeError, match="close timed out"):
        client.close()

    assert client.pending_cancel_ids == {lookup_id}
    assert client.reqs_fingerprints[lookup_id] == fingerprint
    assert client.lookup(tokens, lookup_id) is None
    assert client._cancel_retry_timer is not None
    assert engine.pinned_ids == {lookup_id}

    engine.cancel_enabled.set()
    _wait_until(lambda: lookup_id not in client.pending_cancel_ids)
    client.close()
    assert engine.pinned_ids == set()


def test_bypass_force_close_transfers_later_id_to_shared_engine_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first manager cannot strand a later pin on its still-live engine."""
    first_id = "a-first-deadline"
    later_id = "b-later-retry"
    engine = _EngineGenerationRecordingEngine()
    engine.lookup_pins = {
        first_id: {"cpu": [f"pin-{first_id}"]},
        later_id: {"cpu": [f"pin-{later_id}"]},
    }
    engine.storage_manager.fail_once.update({f"pin-{first_id}", f"pin-{later_id}"})

    client = _bypass_client(engine)
    client.reqs_status = {first_id: 4, later_id: 4}
    client.reqs_fingerprints = {first_id: b"first", later_id: b"later"}
    attempted_during_close: list[str] = []

    def consume_deadline(deadline: float) -> bool:
        attempted_during_close.append(first_id)
        assert not client._attempt_cancel(first_id)
        while time.monotonic() < deadline:
            time.sleep(0.001)
        return False

    client._drain_pending_cancels = consume_deadline  # type: ignore[method-assign]
    monkeypatch.setattr(
        "lmcache.v1.lookup_client.lmcache_lookup_client_bypass."
        "_BYPASS_CANCEL_CLOSE_TIMEOUT_SECONDS",
        0.01,
    )

    def manager_factory(lookup_client=None):
        factory = MagicMock()
        factory.get_or_create_metadata.return_value = None
        factory.get_or_create_lmcache_engine.return_value = engine
        factory.maybe_create_lookup_client.return_value = lookup_client
        factory.maybe_create_lookup_server.return_value = None
        factory.maybe_create_offload_server.return_value = None
        factory.maybe_create_runtime_plugin_launcher.return_value = None
        factory.maybe_create_internal_api_server.return_value = None
        factory.get_engine_instance_id.return_value = "shared-bypass-engine"
        return factory

    config = LMCacheEngineConfig.from_defaults()
    first_manager = LMCacheManager(config, manager_factory(client))
    second_manager = LMCacheManager(config, manager_factory())

    with patch("lmcache.v1.manager.LMCacheEngineBuilder") as builder:
        first_manager.stop_services()

        assert attempted_during_close == [first_id]
        assert builder.destroy.call_count == 0
        assert first_manager.lookup_client is None
        assert first_manager._services_stopped
        assert second_manager.lmcache_engine is engine

        # The dynamic close never reached this UUID. Force-close nevertheless
        # registered and scheduled it before releasing the first manager's
        # shared-engine reference. Its first engine-owned attempt failed, so
        # the durable timer must retain it while the second manager stays live.
        assert later_id in engine.lookup_cleanup_registered
        assert later_id in engine.lookup_cleanup_pending
        assert later_id in engine.lookup_pins

        _wait_until(lambda: later_id not in engine.lookup_pins)
        assert second_manager.lmcache_engine is engine
        assert engine.lookup_cleanup_registered == set()
        assert engine.lookup_cleanup_pending == set()
        assert engine.lookup_pins == {}

        second_manager.stop_services()
        builder.destroy.assert_called_once_with(
            "shared-bypass-engine", expected_engine=engine
        )


def test_sync_close_waits_for_inflight_cancel_retry() -> None:
    engine = _RecordingEngine(True)
    client, transport_base = _sync_rpc_client(engine)
    transport = _BlockingCancelTransport(
        transport_base.server,
        block_after=2,
    )
    client.transport = transport  # type: ignore[assignment]
    lookup_id = "sync-close-retry-barrier"

    assert client.lookup([1, 2, 3, 4], lookup_id) == 4
    client.cancel_lookup(lookup_id)
    assert transport.cancel_entered.wait(timeout=2.0)
    transport.cancel_enabled.set()

    close_returned = threading.Event()
    close_thread = threading.Thread(
        target=_call_and_signal,
        args=(client.close, close_returned),
    )
    close_thread.start()
    assert not close_returned.wait(timeout=0.05)

    transport.cancel_release.set()
    close_thread.join(timeout=2.0)
    assert not close_thread.is_alive()
    assert close_returned.is_set()
    assert transport.closed
    assert client.pending_cancel_ids == set()
    assert engine.pinned_ids == set()


def test_stale_sync_retry_cannot_send_after_close() -> None:
    engine = _RecordingEngine(True)
    client, transport_base = _sync_rpc_client(engine)
    transport = _ToggleCancelTransport(transport_base.server)
    transport.cancel_enabled.set()
    client.transport = transport  # type: ignore[assignment]
    lookup_id = "sync-stale-retry"

    assert client.lookup([1, 2, 3, 4], lookup_id) == 4
    client.pending_cancel_ids.add(lookup_id)
    retry_entered = threading.Event()
    retry_release = threading.Event()
    original_send = client._send_cancel_once

    def gated_send(lookup_id: str) -> bool:
        if threading.current_thread().name == "stale-sync-retry":
            retry_entered.set()
            assert retry_release.wait(timeout=2.0)
        return original_send(lookup_id)

    client._send_cancel_once = gated_send  # type: ignore[method-assign]
    retry_thread = threading.Thread(
        target=client._retry_pending_cancels,
        name="stale-sync-retry",
    )
    retry_thread.start()
    assert retry_entered.wait(timeout=2.0)

    client.close()
    assert transport.closed
    assert transport.cancel_calls == 1

    retry_release.set()
    retry_thread.join(timeout=2.0)
    assert not retry_thread.is_alive()
    assert transport.cancel_calls == 1
    assert client.pending_cancel_ids == set()


def test_sync_draining_rejects_unowned_cancel_after_empty_drain() -> None:
    engine = _RecordingEngine(True)
    client, transport_base = _sync_rpc_client(engine)
    transport = _ToggleCancelTransport(transport_base.server)
    client.transport = transport  # type: ignore[assignment]
    drain_empty = threading.Event()
    publish_closed = threading.Event()
    original_drain = client._drain_pending_cancels

    def paused_drain(deadline: float) -> bool:
        result = original_drain(deadline)
        drain_empty.set()
        assert publish_closed.wait(timeout=2.0)
        return result

    client._drain_pending_cancels = paused_drain  # type: ignore[method-assign]
    close_thread = threading.Thread(target=client.close)
    close_thread.start()
    assert drain_empty.wait(timeout=2.0)

    cancel_returned = threading.Event()
    cancel_thread = threading.Thread(
        target=_call_and_signal,
        args=(client.cancel_lookup, cancel_returned, "late-unowned"),
    )
    cancel_thread.start()
    assert not cancel_returned.wait(timeout=0.05)
    assert client.pending_cancel_ids == set()
    assert "late-unowned" not in client.reqs_fingerprints
    assert transport.cancel_calls == 0

    publish_closed.set()
    close_thread.join(timeout=2.0)
    cancel_thread.join(timeout=2.0)
    assert not close_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert cancel_returned.is_set()
    assert transport.closed


def test_bypass_draining_rejects_unowned_cancel_after_empty_drain() -> None:
    engine = _RecordingEngine(True)
    client = _bypass_client(engine)
    drain_empty = threading.Event()
    publish_closed = threading.Event()
    original_drain = client._drain_pending_cancels

    def paused_drain(deadline: float) -> bool:
        result = original_drain(deadline)
        drain_empty.set()
        assert publish_closed.wait(timeout=2.0)
        return result

    client._drain_pending_cancels = paused_drain  # type: ignore[method-assign]
    close_thread = threading.Thread(target=client.close)
    close_thread.start()
    assert drain_empty.wait(timeout=2.0)

    client.cancel_lookup("late-unowned")
    assert client.pending_cancel_ids == set()
    assert "late-unowned" not in client.reqs_fingerprints
    assert engine.unpinned_ids == []

    publish_closed.set()
    close_thread.join(timeout=2.0)
    assert not close_thread.is_alive()


@pytest.mark.parametrize("raise_after_lookup", [False, True])
def test_sync_close_quiesces_positive_or_uncertain_inflight_lookup(
    raise_after_lookup: bool,
) -> None:
    engine = _RecordingEngine(True)
    client, transport_base = _sync_rpc_client(engine)
    transport = _BlockingLookupTransport(
        transport_base.server,
        raise_after_lookup=raise_after_lookup,
    )
    client.transport = transport  # type: ignore[assignment]
    lookup_id = f"sync-close-lookup-{raise_after_lookup}"
    result: list[Optional[int]] = []
    lookup_thread = threading.Thread(
        target=lambda: result.append(
            client.lookup(
                [1, 2, 3, 4],
                lookup_id,
                explicit_spans=_declaration(),
            )
        )
    )
    lookup_thread.start()
    assert transport.lookup_entered.wait(timeout=2.0)
    assert engine.pinned_ids == {lookup_id}

    close_returned = threading.Event()
    close_thread = threading.Thread(
        target=_call_and_signal,
        args=(client.close, close_returned),
    )
    close_thread.start()
    _wait_until(lambda: client._cancel_lifecycle_state == "draining")
    with pytest.raises(RuntimeError, match="not accepting new ownership"):
        client.validate_lookup_request([9], "rejected-during-close")
    assert "rejected-during-close" not in client.reqs_fingerprints
    assert not close_returned.is_set()

    transport.lookup_release.set()
    lookup_thread.join(timeout=2.0)
    close_thread.join(timeout=2.0)
    assert not lookup_thread.is_alive()
    assert not close_thread.is_alive()
    expected_result: list[Optional[int]] = [None] if raise_after_lookup else [4]
    assert result == expected_result
    assert transport.closed
    assert engine.pinned_ids == set()
    assert client.pending_cancel_ids == set()
    assert client.reqs_fingerprints == {}


def test_bypass_close_quiesces_inflight_lookup() -> None:
    engine = _BlockingLookupEngine()
    client = _bypass_client(engine)
    lookup_id = "bypass-close-lookup"
    result: list[Optional[int]] = []
    lookup_thread = threading.Thread(
        target=lambda: result.append(client.lookup([1, 2, 3, 4], lookup_id))
    )
    lookup_thread.start()
    assert engine.lookup_entered.wait(timeout=2.0)
    assert engine.pinned_ids == {lookup_id}

    close_returned = threading.Event()
    close_thread = threading.Thread(
        target=_call_and_signal,
        args=(client.close, close_returned),
    )
    close_thread.start()
    _wait_until(lambda: client._cancel_lifecycle_state == "draining")
    with pytest.raises(RuntimeError, match="not accepting new ownership"):
        client.validate_lookup_request([9], "rejected-during-close")
    assert not close_returned.is_set()

    engine.lookup_release.set()
    lookup_thread.join(timeout=2.0)
    close_thread.join(timeout=2.0)
    assert not lookup_thread.is_alive()
    assert not close_thread.is_alive()
    assert result == [4]
    assert engine.pinned_ids == set()
    assert client.pending_cancel_ids == set()


def test_sync_transfer_waits_for_lookup_publication() -> None:
    engine = _RecordingEngine(True)
    client, transport_base = _sync_rpc_client(engine)
    transport = _BlockingLookupTransport(transport_base.server)
    client.transport = transport  # type: ignore[assignment]
    lookup_id = "sync-transfer-barrier"
    lookup_thread = threading.Thread(
        target=lambda: client.lookup([1, 2, 3, 4], lookup_id)
    )
    lookup_thread.start()
    assert transport.lookup_entered.wait(timeout=2.0)

    transfer_returned = threading.Event()
    transfer_thread = threading.Thread(
        target=_call_and_signal,
        args=(client.transfer_lookup, transfer_returned, lookup_id),
    )
    transfer_thread.start()
    assert not transfer_returned.wait(timeout=0.05)

    transport.lookup_release.set()
    lookup_thread.join(timeout=2.0)
    transfer_thread.join(timeout=2.0)
    assert transfer_returned.is_set()
    assert lookup_id not in client.reqs_fingerprints
    assert lookup_id not in client.reqs_status
    assert engine.pinned_ids == {lookup_id}
    client.close()
    assert engine.pinned_ids == {lookup_id}
    engine.lookup_unpin(lookup_id)


def test_bypass_transfer_waits_for_lookup_publication() -> None:
    engine = _BlockingLookupEngine()
    client = _bypass_client(engine)
    lookup_id = "bypass-transfer-barrier"
    lookup_thread = threading.Thread(
        target=lambda: client.lookup([1, 2, 3, 4], lookup_id)
    )
    lookup_thread.start()
    assert engine.lookup_entered.wait(timeout=2.0)

    transfer_returned = threading.Event()
    transfer_thread = threading.Thread(
        target=_call_and_signal,
        args=(client.transfer_lookup, transfer_returned, lookup_id),
    )
    transfer_thread.start()
    assert not transfer_returned.wait(timeout=0.05)

    engine.lookup_release.set()
    lookup_thread.join(timeout=2.0)
    transfer_thread.join(timeout=2.0)
    assert transfer_returned.is_set()
    assert lookup_id not in client.reqs_fingerprints
    assert lookup_id not in client.reqs_status
    assert engine.pinned_ids == {lookup_id}
    client.close()
    assert engine.pinned_ids == {lookup_id}
    engine.lookup_unpin(lookup_id)


def test_sync_transfer_first_rejects_late_cancel_of_worker_owner() -> None:
    engine = _RecordingEngine(True)
    client, transport = _sync_rpc_client(engine)
    lookup_id = "sync-transfer-before-cancel"

    assert client.lookup([1, 2, 3, 4], lookup_id) == 4
    client.transfer_lookup(lookup_id)
    message_count = len(transport.messages)
    client.cancel_lookup(lookup_id)

    assert len(transport.messages) == message_count
    assert client.pending_cancel_ids == set()
    assert lookup_id in client.transferred_lookup_ids
    assert engine.pinned_ids == {lookup_id}
    with pytest.raises(ValueError, match="transferred lookup_id cannot be reused"):
        client.lookup([1, 2, 3, 4], lookup_id)
    assert engine.lookup_count == 1
    client.close()
    assert engine.pinned_ids == {lookup_id}
    engine.lookup_unpin(lookup_id)


def test_sync_cancel_first_makes_concurrent_transfer_fail() -> None:
    engine = _RecordingEngine(True)
    client, transport_base = _sync_rpc_client(engine)
    transport = _BlockingCancelTransport(transport_base.server)
    transport.cancel_enabled.set()
    client.transport = transport  # type: ignore[assignment]
    lookup_id = "sync-cancel-before-transfer"
    assert client.lookup([1, 2, 3, 4], lookup_id) == 4

    cancel_thread = threading.Thread(target=lambda: client.cancel_lookup(lookup_id))
    cancel_thread.start()
    assert transport.cancel_entered.wait(timeout=2.0)
    transfer_errors: list[Exception] = []

    def transfer() -> None:
        try:
            client.transfer_lookup(lookup_id)
        except Exception as error:
            transfer_errors.append(error)

    transfer_thread = threading.Thread(target=transfer)
    transfer_thread.start()
    time.sleep(0.05)
    assert transfer_thread.is_alive()

    transport.cancel_release.set()
    cancel_thread.join(timeout=2.0)
    transfer_thread.join(timeout=2.0)
    assert not cancel_thread.is_alive()
    assert not transfer_thread.is_alive()
    assert len(transfer_errors) == 1
    assert "without a live request binding" in str(transfer_errors[0])
    assert engine.pinned_ids == set()
    assert client.pending_cancel_ids == set()
    client.close()


def test_bypass_transfer_first_rejects_late_cancel_of_worker_owner() -> None:
    engine = _RecordingEngine(True)
    client = _bypass_client(engine)
    lookup_id = "bypass-transfer-before-cancel"

    assert client.lookup([1, 2, 3, 4], lookup_id) == 4
    client.transfer_lookup(lookup_id)
    client.cancel_lookup(lookup_id)

    assert client.pending_cancel_ids == set()
    assert lookup_id in client.transferred_lookup_ids
    assert engine.unpinned_ids == []
    assert engine.pinned_ids == {lookup_id}
    with pytest.raises(ValueError, match="transferred lookup_id cannot be reused"):
        client.lookup([1, 2, 3, 4], lookup_id)
    assert engine.lookup_count == 1
    client.close()
    assert engine.pinned_ids == {lookup_id}
    engine.lookup_unpin(lookup_id)


def test_bypass_cancel_first_makes_concurrent_transfer_fail() -> None:
    engine = _BlockingUnpinEngine()
    client = _bypass_client(engine)
    lookup_id = "bypass-cancel-before-transfer"
    assert client.lookup([1, 2, 3, 4], lookup_id) == 4

    cancel_thread = threading.Thread(target=lambda: client.cancel_lookup(lookup_id))
    cancel_thread.start()
    assert engine.unpin_entered.wait(timeout=2.0)
    transfer_errors: list[Exception] = []

    def transfer() -> None:
        try:
            client.transfer_lookup(lookup_id)
        except Exception as error:
            transfer_errors.append(error)

    transfer_thread = threading.Thread(target=transfer)
    transfer_thread.start()
    time.sleep(0.05)
    assert transfer_thread.is_alive()

    engine.unpin_release.set()
    cancel_thread.join(timeout=2.0)
    transfer_thread.join(timeout=2.0)
    assert not cancel_thread.is_alive()
    assert not transfer_thread.is_alive()
    assert len(transfer_errors) == 1
    assert "without a live request binding" in str(transfer_errors[0])
    assert engine.pinned_ids == set()
    assert client.pending_cancel_ids == set()
    client.close()


def test_sync_finish_waits_for_lookup_then_exact_cancels_owner() -> None:
    engine = _RecordingEngine(True)
    client, transport_base = _sync_rpc_client(engine)
    transport = _BlockingLookupTransport(transport_base.server)
    client.transport = transport  # type: ignore[assignment]
    lookup_id = "sync-finish-barrier"
    lookup_thread = threading.Thread(
        target=lambda: client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
    )
    lookup_thread.start()
    assert transport.lookup_entered.wait(timeout=2.0)

    finish_returned = threading.Event()
    finish_thread = threading.Thread(
        target=_call_and_signal,
        args=(client.finish_lookup, finish_returned, lookup_id),
    )
    finish_thread.start()
    assert not finish_returned.wait(timeout=0.05)

    transport.lookup_release.set()
    lookup_thread.join(timeout=2.0)
    finish_thread.join(timeout=2.0)
    assert not lookup_thread.is_alive()
    assert not finish_thread.is_alive()
    assert finish_returned.is_set()
    assert engine.pinned_ids == set()
    assert client.pending_cancel_ids == set()
    assert client.reqs_fingerprints == {}
    client.close()


def test_bypass_finish_waits_for_lookup_then_exact_cancels_owner() -> None:
    engine = _BlockingLookupEngine()
    client = _bypass_client(engine)
    lookup_id = "bypass-finish-barrier"
    lookup_thread = threading.Thread(
        target=lambda: client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
    )
    lookup_thread.start()
    assert engine.lookup_entered.wait(timeout=2.0)

    finish_returned = threading.Event()
    finish_thread = threading.Thread(
        target=_call_and_signal,
        args=(client.finish_lookup, finish_returned, lookup_id),
    )
    finish_thread.start()
    assert not finish_returned.wait(timeout=0.05)

    engine.lookup_release.set()
    lookup_thread.join(timeout=2.0)
    finish_thread.join(timeout=2.0)
    assert not lookup_thread.is_alive()
    assert not finish_thread.is_alive()
    assert finish_returned.is_set()
    assert engine.pinned_ids == set()
    assert client.pending_cancel_ids == set()
    assert client.reqs_fingerprints == {}
    client.close()


def test_sync_transfer_tombstone_capacity_fences_future_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.lookup_client.lmcache_lookup_client._TRANSFERRED_LOOKUP_ID_LIMIT",
        1,
    )
    engine = _RecordingEngine(True)
    client, _ = _sync_rpc_client(engine)
    lookup_id = "sync-capacity-owner"

    assert (
        client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
        == 4
    )
    client.transfer_lookup(lookup_id)
    assert client.lookup_transfer_fenced
    assert client.transferred_lookup_ids == {lookup_id}

    with pytest.raises(RuntimeError, match="safety capacity is exhausted"):
        client.lookup(
            [1, 2, 3, 4],
            "sync-after-capacity",
            explicit_spans=_declaration(),
        )
    assert engine.lookup_count == 1
    assert "sync-after-capacity" not in client.reqs_fingerprints
    client.cancel_lookup(lookup_id)
    assert engine.pinned_ids == {lookup_id}
    client.close()
    engine.lookup_unpin(lookup_id)


def test_bypass_transfer_tombstone_capacity_fences_future_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.lookup_client.lmcache_lookup_client_bypass."
        "_TRANSFERRED_LOOKUP_ID_LIMIT",
        1,
    )
    engine = _RecordingEngine(True)
    client = _bypass_client(engine)
    lookup_id = "bypass-capacity-owner"

    assert (
        client.lookup(
            [1, 2, 3, 4],
            lookup_id,
            explicit_spans=_declaration(),
        )
        == 4
    )
    client.transfer_lookup(lookup_id)
    assert client.lookup_transfer_fenced
    assert client.transferred_lookup_ids == {lookup_id}

    with pytest.raises(RuntimeError, match="safety capacity is exhausted"):
        client.lookup(
            [1, 2, 3, 4],
            "bypass-after-capacity",
            explicit_spans=_declaration(),
        )
    assert engine.lookup_count == 1
    assert "bypass-after-capacity" not in client.reqs_fingerprints
    client.cancel_lookup(lookup_id)
    assert engine.pinned_ids == {lookup_id}
    client.close()
    engine.lookup_unpin(lookup_id)


def test_engine_async_and_nonlayerwise_paths_fail_closed() -> None:
    declaration = _declaration()
    engine = LMCacheEngine.__new__(LMCacheEngine)

    with pytest.raises(NotImplementedError, match="async lookup"):
        engine.async_lookup_and_prefetch(
            "lookup",
            tokens=[1, 2, 3, 4],
            explicit_spans=declaration,
        )

    engine.use_layerwise = False
    with pytest.raises(NotImplementedError, match="layerwise CacheBlend"):
        engine.lookup(tokens=[1, 2, 3, 4], explicit_spans=declaration)

    with pytest.raises(NotImplementedError, match="only by store_layer"):
        engine.store([1, 2, 3, 4], explicit_spans=declaration)
    with pytest.raises(NotImplementedError, match="only by retrieve_layer"):
        engine.retrieve([1, 2, 3, 4], explicit_spans=declaration)
