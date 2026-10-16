# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from contextlib import nullcontext
from secrets import token_bytes
from typing import TYPE_CHECKING, Any, Optional, Union, cast
import json
import threading
import time

# Third Party
import torch

# First Party
from lmcache.integration.base_service_factory import ServiceResultPublisher
from lmcache.logging import init_logger
from lmcache.v1.construction import (
    CloseMethodCleanupOwner,
    drain_lookup_construction_cleanup,
    retain_lookup_construction_cleanup,
)
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.request_fingerprint import (
    make_lookup_request_fingerprint,
    make_serializable_token_ids,
)
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.rpc.transport import (
        RpcClientTransport,
        RpcServerTransport,
    )

logger = init_logger(__name__)

_EXPLICIT_LOOKUP_SCHEMA = "lmcache.lookup.explicit-spans"
_EXPLICIT_LOOKUP_VERSION = 1
_EXPLICIT_LOOKUP_RESULT_MAGIC = b"LMCacheExplicitLookupResult"
_EXPLICIT_LOOKUP_RESULT_VERSION = 1
_LOOKUP_CONTROL_SCHEMA = "lmcache.lookup.control"
_LOOKUP_CONTROL_VERSION = 1
_LOOKUP_CANCEL_OPERATION = "cancel"
_LOOKUP_CANCEL_ACK = b"LMCacheCancelAck\x01"
_LOOKUP_CANCEL_IMMEDIATE_ATTEMPTS = 1
_LOOKUP_CANCEL_RETRY_INTERVAL_SECONDS = 0.25
_LOOKUP_CANCEL_CLOSE_TIMEOUT_SECONDS = 1.0
_LOOKUP_SERVER_CLOSE_TIMEOUT_SECONDS = 2.0
_LOOKUP_GENERATION_BYTES = 16
_CANCELLED_GENERATION_HISTORY = 4096
_TRANSFERRED_LOOKUP_ID_LIMIT = 4096
_LOOKUP_CLIENT_OPEN = "open"
_LOOKUP_CLIENT_DRAINING = "draining"
_LOOKUP_CLIENT_CLOSED = "closed"


def _encode_explicit_lookup_result(result: int, generation: bytes) -> bytes:
    """Bind an explicit result to its protocol version and generation."""
    return b"".join(
        (
            _EXPLICIT_LOOKUP_RESULT_MAGIC,
            bytes((_EXPLICIT_LOOKUP_RESULT_VERSION,)),
            generation,
            result.to_bytes(4, "big"),
        )
    )


def _decode_explicit_lookup_result(response: bytes, generation: bytes) -> int:
    expected_size = (
        len(_EXPLICIT_LOOKUP_RESULT_MAGIC) + 1 + _LOOKUP_GENERATION_BYTES + 4
    )
    if not isinstance(response, bytes) or len(response) != expected_size:
        raise ValueError("explicit lookup response has an invalid envelope size")
    magic_end = len(_EXPLICIT_LOOKUP_RESULT_MAGIC)
    if response[:magic_end] != _EXPLICIT_LOOKUP_RESULT_MAGIC:
        raise ValueError("explicit lookup response has an unknown type")
    version = response[magic_end]
    if version != _EXPLICIT_LOOKUP_RESULT_VERSION:
        raise ValueError("unsupported explicit lookup response version")
    generation_start = magic_end + 1
    generation_end = generation_start + _LOOKUP_GENERATION_BYTES
    if response[generation_start:generation_end] != generation:
        raise ValueError("explicit lookup response generation does not match request")
    return int.from_bytes(response[generation_end:], "big")


def _encode_explicit_lookup_request(
    token_ids: list[int],
    explicit_spans: ExplicitTokenSpans,
    generation: bytes,
) -> dict[str, Any]:
    """Encode the versioned explicit-span request envelope."""
    return {
        "schema": _EXPLICIT_LOOKUP_SCHEMA,
        "version": _EXPLICIT_LOOKUP_VERSION,
        "generation": generation,
        "tokens": token_ids,
        "token_count": explicit_spans.token_count,
        "cached_span": [
            explicit_spans.cached_span.start,
            explicit_spans.cached_span.end,
        ],
        "spans": [[span.start, span.end] for span in explicit_spans.spans],
    }


def _decode_wire_span(value: Any, field_name: str) -> TokenSpan:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-element list")
    return TokenSpan(value[0], value[1])


def _decode_explicit_lookup_request(
    envelope: dict[str, Any],
) -> tuple[list[int], ExplicitTokenSpans, bytes]:
    """Decode and validate one versioned explicit-span request envelope."""
    if envelope.get("schema") != _EXPLICIT_LOOKUP_SCHEMA:
        raise ValueError("unknown lookup request schema")
    version = envelope.get("version")
    if type(version) is not int or version != _EXPLICIT_LOOKUP_VERSION:
        raise ValueError("unsupported explicit lookup request version")
    generation = envelope.get("generation")
    if not isinstance(generation, bytes) or len(generation) != _LOOKUP_GENERATION_BYTES:
        raise TypeError("explicit lookup generation must be a 16-byte primitive value")

    token_ids = envelope.get("tokens")
    if not isinstance(token_ids, list):
        raise TypeError("explicit lookup tokens must be a list")
    if any(type(token_id) is not int for token_id in token_ids):
        raise TypeError("explicit lookup tokens must be plain Python integers")
    cached_span = _decode_wire_span(envelope.get("cached_span"), "cached_span")
    raw_spans = envelope.get("spans")
    if not isinstance(raw_spans, list):
        raise TypeError("explicit lookup spans must be a list")
    token_count_value = envelope.get("token_count")
    if type(token_count_value) is not int:
        raise TypeError("explicit lookup token_count must be a plain Python integer")
    explicit_spans = ExplicitTokenSpans(
        token_count=cast(int, token_count_value),
        cached_span=cached_span,
        spans=tuple(
            _decode_wire_span(span, f"spans[{index}]")
            for index, span in enumerate(raw_spans)
        ),
    )
    explicit_spans.validate_token_count(len(token_ids))
    return token_ids, explicit_spans, generation


def _encode_lookup_cancel_request(generation: Optional[bytes]) -> dict[str, Any]:
    return {
        "schema": _LOOKUP_CONTROL_SCHEMA,
        "version": _LOOKUP_CONTROL_VERSION,
        "operation": _LOOKUP_CANCEL_OPERATION,
        "generation": generation,
    }


def _decode_lookup_control_request(envelope: dict[str, Any]) -> Optional[bytes]:
    if envelope.get("schema") != _LOOKUP_CONTROL_SCHEMA:
        raise ValueError("unknown lookup request schema")
    version = envelope.get("version")
    if type(version) is not int or version != _LOOKUP_CONTROL_VERSION:
        raise ValueError("unsupported lookup control request version")
    operation = envelope.get("operation")
    if operation != _LOOKUP_CANCEL_OPERATION:
        raise ValueError("unsupported lookup control operation")
    generation = envelope.get("generation")
    if generation is not None and (
        not isinstance(generation, bytes) or len(generation) != _LOOKUP_GENERATION_BYTES
    ):
        raise TypeError("lookup control generation must be a 16-byte primitive value")
    return generation


class LMCacheLookupClient(LookupClientInterface):
    """
    Lookup client that communicates with a lookup server
    via an injected RpcClientTransport.

    The client is decoupled from the underlying communication
    mechanism. The transport layer handles connection management,
    retries, and error recovery.

    Related extra_config:
    - lookup_server_worker_ids:
        is a config to control create lookup server on some
        workers.
        if mla is not enabled, default is [];
        if mla is enabled, default is [0];
        - if lookup_server_worker_ids is [], start lookup
          server on all workers
        - if lookup_server_worker_ids is [0], start lookup
          server on worker0
    - if lookup_server_worker_ids is [0, 3, 6], start
      lookup server on worker0, worker3 and worker6
    """

    # ``close`` leaves DRAINING state and transport progress installed after
    # failure; a later call resumes transport-only cleanup without admission.
    construction_cleanup_close_retry_safe = True

    def __init__(
        self,
        config: "LMCacheEngineConfig",
        metadata: "LMCacheMetadata",
        transport: "RpcClientTransport",
        result_publisher: Optional[ServiceResultPublisher] = None,
    ):
        self.config = config
        self.transport = transport

        # NOTE: map from lookup_id (i.e., req_id) to
        # req's status.
        # int indicates number of hit tokens.
        # The assumption here is that once a request is
        # looked up, the following lookups of the same
        # request must have the same result.
        self.reqs_status: dict[str, Optional[int]] = {}
        self.reqs_fingerprints: dict[str, bytes] = {}
        self.reqs_generations: dict[str, bytes] = {}
        self.pending_cancel_ids: set[str] = set()
        # KMesh scheduler-attempt UUIDs are one-shot. Keep transferred IDs for
        # this client lifetime because no transport horizon proves that a late
        # scheduler cancel can be forgotten safely. This is an experimental
        # correctness tombstone, not a production session-fencing design. At
        # the fixed capacity, lookup admission is permanently fenced rather
        # than evicting an ID and reopening its late-cancel ABA.
        self.transferred_lookup_ids: set[str] = set()
        self.lookup_transfer_fenced = False
        # ``RpcClientTransport`` uses request/reply sockets, which must not be
        # driven concurrently by scheduler lookups and the abort retry loop.
        self._transport_lock = threading.Lock()
        self._cancel_state_lock = threading.Lock()
        self._lookup_operation_lock = threading.RLock()
        self._cancel_close_lock = threading.Lock()
        self._cancel_retry_timer: Optional[threading.Timer] = None
        self._cancel_retry_closing = False
        self._cancel_lifecycle_state = _LOOKUP_CLIENT_OPEN

        # First Party
        from lmcache.v1.token_database import (
            ChunkedTokenDatabase,
            SegmentTokenDatabase,
            TokenDatabase,
        )

        self.enable_blending = config.enable_blending
        self.token_database: TokenDatabase
        if self.enable_blending:
            self.token_database = SegmentTokenDatabase(config, metadata)
        else:
            self.token_database = ChunkedTokenDatabase(config, metadata)
        if result_publisher is not None:
            result_publisher.replace_intermediate(
                transport,
                self,
                CloseMethodCleanupOwner(
                    self,
                    retry_safe_failure=True,
                    thread_name="lmcache-lookup-client-construction-cleanup",
                ),
            )

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        "-1 means not found;
        None means ongoing; (not supported in sync client)
        int >= 0 means number of hit tokens
        """
        self._ensure_cancel_state()
        with self._cancel_state_lock:
            if lookup_id in self.pending_cancel_ids:
                return None
            return self.reqs_status.get(lookup_id, -1)

    def _validate_and_bind_lookup_request(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict],
        explicit_spans: Optional[ExplicitTokenSpans],
    ) -> list[int]:
        if explicit_spans is not None and not self.supports_explicit_spans():
            # Capability failures must not leave a fingerprint tombstone.
            raise NotImplementedError(
                "synchronous RPC explicit spans require layerwise CacheBlend"
            )

        serializable_ids = make_serializable_token_ids(token_ids)
        request_fingerprint = make_lookup_request_fingerprint(
            serializable_ids,
            request_configs,
            explicit_spans,
        )
        self._ensure_cancel_state()
        with self._cancel_state_lock:
            if self._cancel_lifecycle_state != _LOOKUP_CLIENT_OPEN:
                raise RuntimeError("lookup client is not accepting new ownership")
            if self.lookup_transfer_fenced:
                raise RuntimeError("lookup transfer safety capacity is exhausted")
            bound_fingerprint = self.reqs_fingerprints.get(lookup_id)
            if bound_fingerprint is None:
                if lookup_id in self.transferred_lookup_ids:
                    raise ValueError("transferred lookup_id cannot be reused")
                if lookup_id in self.pending_cancel_ids:
                    raise RuntimeError(
                        "pending lookup cancellation has no request-content binding"
                    )
                if lookup_id in self.reqs_status:
                    raise ValueError("cached lookup_id has no request-content binding")
                self.reqs_fingerprints[lookup_id] = request_fingerprint
            elif bound_fingerprint != request_fingerprint:
                raise ValueError(
                    "lookup_id was reused with different lookup request content"
                )
        return serializable_ids

    def validate_lookup_request(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> None:
        self._validate_and_bind_lookup_request(
            token_ids,
            lookup_id,
            request_configs,
            explicit_spans,
        )

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> Optional[int]:
        self._ensure_cancel_state()
        with self._lookup_operation_lock:
            with self._cancel_state_lock:
                if self._cancel_lifecycle_state != _LOOKUP_CLIENT_OPEN:
                    raise RuntimeError("lookup client is not accepting new ownership")
            return self._lookup(
                token_ids,
                lookup_id,
                request_configs,
                explicit_spans,
            )

    def _lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> Optional[int]:
        serializable_ids = self._validate_and_bind_lookup_request(
            token_ids,
            lookup_id,
            request_configs,
            explicit_spans,
        )
        cached_result = self.lookup_cache(lookup_id)
        if cached_result != -1:
            return cached_result

        request_configs_str = ""
        if request_configs is not None and len(request_configs) != 0:
            request_configs_str = json.dumps(request_configs)

        msg_buf: list[Any]
        explicit_generation: Optional[bytes] = None
        if explicit_spans is not None:
            generation = getattr(self, "reqs_generations", {}).get(lookup_id)
            if generation is None:
                generation = token_bytes(_LOOKUP_GENERATION_BYTES)
                if not hasattr(self, "reqs_generations"):
                    self.reqs_generations = {}
                self.reqs_generations[lookup_id] = generation
            explicit_generation = generation
            msg_buf = [
                _encode_explicit_lookup_request(
                    serializable_ids,
                    explicit_spans,
                    generation,
                ),
                lookup_id,
                request_configs_str,
            ]
        # NOTE(Jiayi): We cannot only send hashes when
        # blending enabled because the blender need the
        # input embedding.
        elif not self.enable_blending:
            hashes = []
            offsets = []

            for (
                start,
                end,
                key,
            ) in self.token_database.process_tokens(token_ids, make_key=False):
                hashes.append(key)
                offsets.append(end - start)

            # if the token database returns no hashes,
            # return 0
            if not hashes:
                self.reqs_status[lookup_id] = 0
                return 0

            msg_buf = [
                hashes,
                offsets,
                lookup_id,
                request_configs_str,
            ]
        else:
            # Convert token_ids to a plain list for msgpack serialization
            # (vLLM 0.18+ may pass ConstantList which msgspec can't encode)
            msg_buf = [
                serializable_ids,
                lookup_id,
                request_configs_str,
            ]

        try:
            self._ensure_cancel_state()
            with self._transport_lock:
                responses = self.transport.send_and_recv_all(msg_buf)
            if not responses:
                raise RuntimeError("lookup transport returned no responses")
            if len(responses) != self.transport.world_size:
                raise RuntimeError(
                    "lookup response count does not match worker world size"
                )
            if explicit_generation is not None:
                results = [
                    _decode_explicit_lookup_result(response, explicit_generation)
                    for response in responses
                ]
            else:
                if any(
                    not isinstance(response, bytes) or len(response) != 4
                    for response in responses
                ):
                    raise ValueError("legacy lookup responses must be four-byte values")
                results = [int.from_bytes(response, "big") for response in responses]
        except Exception as error:
            if explicit_generation is not None:
                # Delivery is uncertain: one or more workers may process this
                # generation after the request socket has already failed. Do
                # not turn that uncertainty into a transferable miss. Retain
                # ownership until a generation-bound cancel is ACKed, and ask
                # the scheduler to defer this allocation attempt.
                logger.error(
                    "Explicit lookup %s has uncertain delivery; cancelling "
                    "generation before retry: %s",
                    lookup_id,
                    error,
                )
                self.reqs_status[lookup_id] = None
                self.cancel_lookup(lookup_id)
                return None
            logger.error("Synchronous lookup %s failed closed: %s", lookup_id, error)
            self.reqs_status[lookup_id] = 0
            return 0

        if len(set(results)) > 1:
            logger.warning(
                "Lookup results (number of hit tokens) "
                "differ across (TP and PP) ranks: %s.",
                results,
            )
        # NOTE: it is possible that the number of hit
        # tokens is different across (TP and PP) ranks,
        # so we can use the minimum value.
        num_hit_toks = min(results)
        self.reqs_status[lookup_id] = num_hit_toks

        return num_hit_toks

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.reqs_status.pop(lookup_id, None)
        self.reqs_fingerprints.pop(lookup_id, None)
        # Keep cleanup identity until the scheduler either cancels this
        # pre-allocation attempt or explicitly transfers it to the worker.

    def transfer_lookup(self, lookup_id: str) -> None:
        """Retire scheduler ownership after successful worker allocation."""
        self.transfer_lookups([lookup_id])

    def transfer_lookups(self, lookup_ids: list[str]) -> None:
        """Retire a fully preflighted batch without worker cancellation."""
        self._ensure_cancel_state()
        with self._lookup_operation_lock:
            with self._cancel_state_lock:
                if self._cancel_lifecycle_state != _LOOKUP_CLIENT_OPEN:
                    raise RuntimeError("lookup client is not accepting transfers")
                unbound = [
                    lookup_id
                    for lookup_id in lookup_ids
                    if lookup_id not in self.reqs_fingerprints
                ]
                if unbound:
                    raise RuntimeError(
                        "lookup ownership cannot transfer without a live request "
                        f"binding: {unbound}"
                    )
                new_transfers = set(lookup_ids).difference(self.transferred_lookup_ids)
                if self.lookup_transfer_fenced or (
                    len(self.transferred_lookup_ids) + len(new_transfers)
                    > _TRANSFERRED_LOOKUP_ID_LIMIT
                ):
                    self.lookup_transfer_fenced = True
                    raise RuntimeError("lookup transfer safety capacity is exhausted")
                pending = self.pending_cancel_ids.intersection(lookup_ids)
                if pending:
                    raise RuntimeError(
                        "lookup ownership cannot transfer while cancellation is pending"
                    )
                for lookup_id in lookup_ids:
                    self._retire_lookup_state(lookup_id)
                    self.transferred_lookup_ids.add(lookup_id)
                if len(self.transferred_lookup_ids) >= _TRANSFERRED_LOOKUP_ID_LIMIT:
                    self.lookup_transfer_fenced = True

    def finish_lookup(self, lookup_id: str) -> None:
        """Retire a request and balance any scheduler-owned worker pin owner."""
        self._ensure_cancel_state()
        with self._lookup_operation_lock:
            with self._cancel_state_lock:
                scheduler_owned = lookup_id in self._owned_lookup_ids_locked()
            if scheduler_owned:
                # Reentrant operation locking lets the normal cancellation
                # path provide the single exact cleanup transition after any
                # admitted lookup has finished publishing its result.
                self.cancel_lookup(lookup_id)
                return
            with self._cancel_state_lock:
                self._retire_lookup_state(lookup_id)

    def _retire_lookup_state(self, lookup_id: str) -> None:
        self.clear_lookup_status(lookup_id)
        getattr(self, "reqs_generations", {}).pop(lookup_id, None)

    def _ensure_cancel_state(self) -> None:
        """Initialize cancel-loop state for regular and ``__new__`` clients."""
        if not hasattr(self, "pending_cancel_ids"):
            self.pending_cancel_ids = set()
        if not hasattr(self, "transferred_lookup_ids"):
            self.transferred_lookup_ids = set()
        if not hasattr(self, "lookup_transfer_fenced"):
            self.lookup_transfer_fenced = False
        if not hasattr(self, "_transport_lock"):
            self._transport_lock = threading.Lock()
        if not hasattr(self, "_cancel_state_lock"):
            self._cancel_state_lock = threading.Lock()
        if not hasattr(self, "_lookup_operation_lock"):
            self._lookup_operation_lock = threading.RLock()
        if not hasattr(self, "_cancel_close_lock"):
            self._cancel_close_lock = threading.Lock()
        if not hasattr(self, "_cancel_retry_timer"):
            self._cancel_retry_timer = None
        if not hasattr(self, "_cancel_retry_closing"):
            self._cancel_retry_closing = False
        if not hasattr(self, "_cancel_lifecycle_state"):
            self._cancel_lifecycle_state = (
                _LOOKUP_CLIENT_CLOSED
                if self._cancel_retry_closing
                else _LOOKUP_CLIENT_OPEN
            )

    def _owned_lookup_ids_locked(self) -> set[str]:
        """Return every lookup whose cleanup is still scheduler-owned.

        The caller holds ``_cancel_state_lock``. A lookup can be between
        binding, result publication, and retry registration, so no single map
        is a complete ownership ledger on its own.
        """
        return set(self.pending_cancel_ids).union(
            getattr(self, "reqs_fingerprints", {}),
            getattr(self, "reqs_status", {}),
            getattr(self, "reqs_generations", {}),
        )

    def _enqueue_owned_lookup_cancels(self) -> None:
        with self._cancel_state_lock:
            self.pending_cancel_ids.update(self._owned_lookup_ids_locked())

    def _drain_pending_cancels(self, deadline: float) -> bool:
        """Actively drain cancellation ownership until ``deadline``."""
        while True:
            with self._cancel_state_lock:
                lookup_ids = tuple(self.pending_cancel_ids)
            if not lookup_ids:
                return True

            for lookup_id in lookup_ids:
                if time.monotonic() >= deadline:
                    return False
                if self._send_cancel_once(lookup_id):
                    self._complete_cancel(lookup_id)

            with self._cancel_state_lock:
                if not self.pending_cancel_ids:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.01, remaining))

    def _cancel_message(self, lookup_id: str) -> list[Any]:
        generation = getattr(self, "reqs_generations", {}).get(lookup_id)
        control_envelope = _encode_lookup_cancel_request(generation)
        if self.enable_blending:
            return [control_envelope, lookup_id, ""]
        # Preserve the legacy non-CacheBlend four-frame shape. The empty
        # offsets placeholder is ignored by control-aware workers.
        return [control_envelope, [], lookup_id, ""]

    def _send_cancel_once(self, lookup_id: str) -> bool:
        try:
            with self._transport_lock:
                # A retry can pass its outer check and then lose this lock to
                # successful close. Recheck under the I/O lock so no stale
                # callback sends on a closed request socket.
                with self._cancel_state_lock:
                    if (
                        self._cancel_lifecycle_state == _LOOKUP_CLIENT_CLOSED
                        or self._cancel_retry_closing
                        or lookup_id not in self.pending_cancel_ids
                    ):
                        return False
                responses = self.transport.send_and_recv_all(
                    self._cancel_message(lookup_id)
                )
            return len(responses) == self.transport.world_size and all(
                response == _LOOKUP_CANCEL_ACK for response in responses
            )
        except Exception:
            # Abort cleanup must not take down the scheduler. Ownership stays
            # with the retry loop until every worker returns the typed ACK.
            logger.exception("Lookup cancel for %s failed", lookup_id)
            return False

    def _complete_cancel(self, lookup_id: str) -> None:
        with self._cancel_state_lock:
            if lookup_id not in self.pending_cancel_ids:
                return
            self._retire_lookup_state(lookup_id)
            # Publish ID reuse only after every stale status/fingerprint/
            # generation field has been retired under the same lock.
            self.pending_cancel_ids.remove(lookup_id)

    def _schedule_cancel_retry(self) -> None:
        self._ensure_cancel_state()
        with self._cancel_state_lock:
            if (
                self._cancel_lifecycle_state == _LOOKUP_CLIENT_CLOSED
                or self._cancel_retry_closing
                or not self.pending_cancel_ids
            ):
                return
            if self._cancel_retry_timer is not None:
                return
            timer = threading.Timer(
                _LOOKUP_CANCEL_RETRY_INTERVAL_SECONDS,
                self._retry_pending_cancels,
            )
            timer.daemon = True
            timer.name = "lookup-cancel-retry"
            self._cancel_retry_timer = timer
            timer.start()

    def _retry_pending_cancels(self) -> None:
        with self._cancel_state_lock:
            self._cancel_retry_timer = None
            if (
                self._cancel_lifecycle_state == _LOOKUP_CLIENT_CLOSED
                or self._cancel_retry_closing
            ):
                return
            lookup_ids = tuple(self.pending_cancel_ids)

        for lookup_id in lookup_ids:
            with self._cancel_state_lock:
                if (
                    self._cancel_lifecycle_state == _LOOKUP_CLIENT_CLOSED
                    or self._cancel_retry_closing
                    or lookup_id not in self.pending_cancel_ids
                ):
                    continue
            if self._send_cancel_once(lookup_id):
                self._complete_cancel(lookup_id)
            else:
                logger.warning(
                    "Lookup cancel for %s remains pending; retrying in %.1fs",
                    lookup_id,
                    _LOOKUP_CANCEL_RETRY_INTERVAL_SECONDS,
                )

        self._schedule_cancel_retry()

    def cancel_lookup(self, lookup_id: str) -> None:
        """Release request-owned pins on every synchronous lookup worker.

        The scheduler performs one bounded attempt. A failed or partial ACK is
        retained as a tombstone and retried asynchronously until every worker
        returns the versioned control ACK. Reusing the lookup ID is rejected
        while that cleanup ownership is outstanding.
        """
        self._ensure_cancel_state()
        with self._lookup_operation_lock:
            with self._cancel_state_lock:
                if self._cancel_lifecycle_state == _LOOKUP_CLIENT_CLOSED:
                    logger.warning(
                        "Ignoring lookup cancel for %s after client close began",
                        lookup_id,
                    )
                    return
                if self._cancel_retry_closing:
                    logger.warning(
                        "Ignoring lookup cancel for %s after transport close began",
                        lookup_id,
                    )
                    return
                if lookup_id in self.pending_cancel_ids:
                    # The existing owner (possibly the timer callback currently
                    # using the request socket) will complete this cancellation.
                    return
                if lookup_id in self.transferred_lookup_ids:
                    logger.warning(
                        "Ignoring lookup cancel for transferred worker owner %s",
                        lookup_id,
                    )
                    return
                if (
                    self._cancel_lifecycle_state == _LOOKUP_CLIENT_DRAINING
                    and lookup_id not in self._owned_lookup_ids_locked()
                ):
                    # An admitted lookup may discover uncertain delivery after
                    # close has entered DRAINING; its fingerprint/generation is
                    # already present and must still be cancelled. A brand-new ID
                    # has no such ownership proof and cannot be allowed to race
                    # drain-empty -> CLOSED publication.
                    logger.warning(
                        "Ignoring unowned lookup cancel for %s while client drains",
                        lookup_id,
                    )
                    return
                self.pending_cancel_ids.add(lookup_id)

            for _ in range(_LOOKUP_CANCEL_IMMEDIATE_ATTEMPTS):
                if self._send_cancel_once(lookup_id):
                    self._complete_cancel(lookup_id)
                    return

            logger.warning(
                "Lookup cancel for %s was not acknowledged; scheduling retry",
                lookup_id,
            )
            self._schedule_cancel_retry()

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports
        producer kvcache reuse"""
        return True

    def supports_explicit_spans(self) -> bool:
        """Whether this synchronous client can send explicit-span lookups."""
        return (
            self.enable_blending
            and self.config.use_layerwise
            and not self.config.enable_async_loading
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        """Drain scheduler ownership and close the request transport.

        Raises:
            RuntimeError: Exact cancellation did not converge within the
                close retry window. A single RPC attempt remains subject to
                the configured transport timeout, so wall time can exceed the
                nominal retry window.

        Returns:
            None.
        """
        self._ensure_cancel_state()
        with self._cancel_close_lock:
            with self._cancel_state_lock:
                if self._cancel_lifecycle_state == _LOOKUP_CLIENT_CLOSED:
                    return
                self._cancel_lifecycle_state = _LOOKUP_CLIENT_DRAINING
                close_retry_only = self._cancel_retry_closing
                timer = self._cancel_retry_timer
                self._cancel_retry_timer = None
            if timer is not None:
                timer.cancel()

            # An admitted lookup owns this lock through response publication
            # and through uncertain-delivery cancel registration. Once this
            # barrier is acquired, the ownership maps form a stable close-time
            # snapshot; new lookup and transfer admission is already closed.
            with self._lookup_operation_lock:
                if close_retry_only:
                    self._close_transport_and_publish_closed()
                    return
                self._enqueue_owned_lookup_cancels()
                deadline = time.monotonic() + _LOOKUP_CANCEL_CLOSE_TIMEOUT_SECONDS
                if not self._drain_pending_cancels(deadline):
                    with self._cancel_state_lock:
                        self._cancel_lifecycle_state = _LOOKUP_CLIENT_OPEN
                        self._cancel_retry_closing = False
                    self._schedule_cancel_retry()
                    raise RuntimeError(
                        "lookup client close timed out with cancellation "
                        "ownership still pending"
                    )

                self._close_transport_and_publish_closed()

    def force_close(self) -> None:
        """Close the transport even when exact cleanup cannot converge.

        This is reserved for terminal engine shutdown, where the manager will
        destroy the worker engines immediately afterward. Dynamic client
        replacement must use :meth:`close` so ownership is never abandoned.

        Raises:
            RuntimeError: The underlying transport could not finish closing.

        Returns:
            None.
        """
        self._ensure_cancel_state()
        with self._cancel_close_lock:
            with self._cancel_state_lock:
                if self._cancel_lifecycle_state == _LOOKUP_CLIENT_CLOSED:
                    return
                self._cancel_lifecycle_state = _LOOKUP_CLIENT_DRAINING
                timer = self._cancel_retry_timer
                self._cancel_retry_timer = None
            if timer is not None:
                timer.cancel()

            with self._lookup_operation_lock:
                self._close_transport_and_publish_closed()

    def _close_transport_and_publish_closed(self) -> None:
        """Close the transport before committing the terminal lifecycle.

        The caller holds both the close and lookup-operation barriers. Keeping
        the client in ``DRAINING`` until transport closure succeeds prevents
        new ownership while leaving both :meth:`close` and
        :meth:`force_close` independently retryable after a partial transport
        failure.
        """
        # Match the transport-before-state lock ordering in
        # ``_send_cancel_once``. A stale timer callback either finishes before
        # this barrier or observes CLOSED before it can use the transport.
        with self._transport_lock:
            with self._cancel_state_lock:
                # Publish the permanent close-only fence before the first
                # resource can close. A failed transport close remains
                # DRAINING and all subsequent lifecycle attempts retry only
                # transport closure.
                self._cancel_retry_closing = True
                timer = self._cancel_retry_timer
                self._cancel_retry_timer = None
            if timer is not None:
                timer.cancel()
            self.transport.close()
            with self._cancel_state_lock:
                self._cancel_lifecycle_state = _LOOKUP_CLIENT_CLOSED


class _LookupServerConstructionCleanupOwner:
    """Retain a partial synchronous lookup server until it is quiescent."""

    def __init__(
        self,
        server: Any,
        transport: Any,
        thread: threading.Thread | None,
    ) -> None:
        self.server = server
        self.transport = transport
        self.thread = thread
        self._running_fenced = False
        self._thread_quiesced = thread is None
        self._transport_owner = CloseMethodCleanupOwner(
            transport,
            thread_name="lmcache-lookup-server-transport-construction-cleanup",
            retry_safe_failure=lambda _error: (
                getattr(
                    transport,
                    "construction_cleanup_close_retry_safe",
                    False,
                )
                is True
            ),
        )
        self._lock = threading.Lock()

    def cleanup(self) -> None:
        """Fence/join the request thread before closing its transport."""
        with self._lock:
            if not self._running_fenced:
                self.server.running = False
                self._running_fenced = True

            thread = self.thread
            if not self._thread_quiesced and thread is not None:
                if thread.is_alive():
                    thread.join(timeout=_LOOKUP_SERVER_CLOSE_TIMEOUT_SECONDS)
                if thread.is_alive():
                    raise RuntimeError(
                        "lookup server thread did not quiesce during construction"
                    )
                self._thread_quiesced = True

            self._transport_owner.cleanup()


class LMCacheLookupServer:
    """Lookup server that handles lookup requests using
    LMCacheEngine, with an injected RpcServerTransport."""

    construction_cleanup_close_retry_safe = True

    def __init__(
        self,
        lmcache_engine: "LMCacheEngine",
        metadata: "LMCacheMetadata",
        transport: "RpcServerTransport",
        result_publisher: Optional[ServiceResultPublisher] = None,
    ):
        def process_request():
            while self.running:
                try:
                    result = self.transport.recv_request()
                    if result is None:
                        continue

                    identity, data_frames = result
                    response = self._process_request_frames(data_frames)
                    self.transport.send_response(identity, response)
                except json.JSONDecodeError as e:
                    logger.error("Error decoding JSON in lookup request: %s", e)
                except UnicodeDecodeError as e:
                    logger.error("Error decoding UTF-8 in lookup request: %s", e)
                except Exception as e:
                    logger.error("Error processing lookup request: %s", e)

        thread: threading.Thread | None = None
        try:
            self.transport = transport
            self.lmcache_engine = lmcache_engine
            self.running = True
            self.enable_blending = lmcache_engine.config.enable_blending
            self.explicit_lookup_generations: dict[str, bytes] = {}
            self.cancelled_explicit_generations: OrderedDict[
                tuple[str, bytes], None
            ] = OrderedDict()

            logger.info("lmcache lookup server started")
            thread = threading.Thread(
                target=process_request,
                daemon=True,
                name="lookup-server-thread",
            )
            # Publish a thread-aware partial server before the request thread
            # can become live. If Thread.start() returns and the next caller
            # instruction is interrupted, Manager recovery already owns the
            # exact fence/join/transport cleanup state.
            if result_publisher is not None:
                result_publisher.replace_intermediate(
                    transport,
                    self,
                    _LookupServerConstructionCleanupOwner(
                        self,
                        transport,
                        thread,
                    ),
                )
            thread.start()
            # Final thread publication belongs to the rollback region. A
            # failed STORE_ATTR must leave the local thread in the attached
            # cleanup receipt rather than an unowned live request loop.
            self.thread = thread
        except BaseException as construction_error:
            if result_publisher is not None and result_publisher.owns_intermediate(
                self
            ):
                # The retained factory owner already has the exact partial
                # server and its thread-aware cleanup owner. Do not create a
                # second cleanup authority or close beneath its live thread.
                raise
            self._cleanup_failed_thread_start(thread, construction_error)
            raise

    def _cleanup_failed_thread_start(
        self,
        thread: threading.Thread | None,
        construction_error: BaseException,
    ) -> None:
        """Reclaim transferred transport ownership after startup fails.

        ``Thread.start`` can start the request loop and then be interrupted
        before returning to this constructor.  Fence the loop, prove that it
        has stopped, and only then close the transport it could use.  The
        construction interruption remains primary if cleanup also fails.
        """
        transport = getattr(self, "transport", None)
        if transport is None:
            # The constructor failed before transferring the transport into
            # ``self``. The factory still owns it and will attach that exact
            # owner instead.
            return
        owner: _LookupServerConstructionCleanupOwner | None = None

        def attach_and_drain() -> None:
            nonlocal owner
            if owner is None:
                owner = _LookupServerConstructionCleanupOwner(
                    self,
                    transport,
                    thread,
                )
            retain_lookup_construction_cleanup(construction_error, owner)
            drain_lookup_construction_cleanup(construction_error)

        try:
            attach_and_drain()
        except BaseException as cleanup_error:
            if cleanup_error is construction_error:
                raise
            attach_and_drain()

    def supports_explicit_spans(self) -> bool:
        """Whether this worker can execute explicit-span lookups safely."""
        return self.enable_blending and self.lmcache_engine.supports_explicit_spans()

    def _generation_state(
        self,
    ) -> tuple[dict[str, bytes], OrderedDict[tuple[str, bytes], None]]:
        active = getattr(self, "explicit_lookup_generations", None)
        if active is None:
            active = self.explicit_lookup_generations = {}
        cancelled = getattr(
            self.lmcache_engine,
            "lookup_cancelled_generations",
            None,
        )
        if cancelled is None:
            cancelled = getattr(self, "cancelled_explicit_generations", None)
            if cancelled is None:
                cancelled = OrderedDict()
            self.lmcache_engine.lookup_cancelled_generations = cancelled
        # Preserve the server-facing name for diagnostics and compatibility,
        # but keep the bounded tombstones owned by the engine so delayed RPCs
        # remain suppressed after a lookup-server replacement.
        self.cancelled_explicit_generations = cancelled
        return active, cancelled

    def _engine_pin_guard(self):
        guard_factory = getattr(self.lmcache_engine, "_lookup_pins_guard", None)
        return guard_factory() if guard_factory is not None else nullcontext()

    def _engine_generation_state(self) -> dict[str, bytes]:
        generations = getattr(self.lmcache_engine, "lookup_pin_generations", None)
        if generations is None:
            generations = self.lmcache_engine.lookup_pin_generations = {}
        return generations

    def _prune_completed_explicit_generations(self) -> None:
        active, _ = self._generation_state()
        lookup_pins = getattr(self.lmcache_engine, "lookup_pins", None)
        if lookup_pins is None:
            return
        with self._engine_pin_guard():
            for lookup_id in list(active):
                if lookup_id not in lookup_pins:
                    generation = active.pop(lookup_id)
                    self._engine_generation_state().pop(lookup_id, None)
                    # Preserve the exact completed generation. This lets a late
                    # or response-loss retry receive an idempotent ACK without
                    # letting it unpin a newer generation that reused the ID.
                    self._remember_cancelled_generation(lookup_id, generation)

    def _remember_cancelled_generation(
        self,
        lookup_id: str,
        generation: bytes,
    ) -> None:
        engine_remember = getattr(
            self.lmcache_engine,
            "_remember_lookup_cancelled_generation",
            None,
        )
        if engine_remember is not None:
            engine_remember(lookup_id, generation)
            # Keep the server-facing alias synchronized for diagnostics and
            # lightweight callers that inspect this historical attribute.
            self._generation_state()
            return

        # Compatibility for lightweight non-LMCacheEngine test doubles.
        _, cancelled = self._generation_state()
        key = (lookup_id, generation)
        if key not in cancelled and len(cancelled) >= _CANCELLED_GENERATION_HISTORY:
            self.lmcache_engine.lookup_explicit_replay_fenced = True
            return
        cancelled[key] = None
        cancelled.move_to_end(key)

    def _cancel_lookup_generation(
        self,
        lookup_id: str,
        generation: Optional[bytes],
    ) -> None:
        active, cancelled = self._generation_state()
        with self._engine_pin_guard():
            durable = self._engine_generation_state()
            lookup_pins = getattr(self.lmcache_engine, "lookup_pins", None)
            has_pins = lookup_pins is not None and lookup_id in lookup_pins
            active_generation = active.get(lookup_id)
            durable_generation = durable.get(lookup_id)
            if (
                active_generation is not None
                and durable_generation is not None
                and active_generation != durable_generation
            ):
                raise RuntimeError("explicit generation ownership is inconsistent")
            owner_generation = durable_generation or active_generation

            if generation is None:
                if owner_generation is not None:
                    raise ValueError(
                        "explicit lookup cancellation requires its request generation"
                    )
                self.lmcache_engine.lookup_unpin(lookup_id)
                return

            cancelled_key = (lookup_id, generation)
            if cancelled_key in cancelled and not (
                has_pins and owner_generation == generation
            ):
                cancelled.move_to_end(cancelled_key)
                return

            if has_pins:
                if owner_generation is None:
                    # A server reset must never guess that legacy/untracked
                    # ownership belongs to this explicit cancel generation.
                    raise ValueError("pinned lookup has no durable generation owner")
                if owner_generation == generation:
                    self.lmcache_engine.lookup_unpin(lookup_id)
                    active.pop(lookup_id, None)
                    durable.pop(lookup_id, None)
            else:
                if active_generation == generation:
                    active.pop(lookup_id, None)
                if durable_generation == generation:
                    durable.pop(lookup_id, None)

            # A worker may never have observed the original lookup when the
            # scheduler received an empty/partial/error result from the TP
            # group. Tombstone and ACK that generation. Any delayed matching
            # lookup is suppressed, and different durable ownership is intact.
            self._remember_cancelled_generation(lookup_id, generation)

    def _process_request_frames(self, data_frames: list[Any]) -> bytes:
        """Validate and execute one decoded lookup request.

        Keeping frame processing separate from the transport loop makes the
        versioned protocol independently testable. Legacy requests retain
        their original three-frame (CacheBlend) and four-frame (hash) forms.
        """
        self._prune_completed_explicit_generations()
        expected_frames = 3 if self.enable_blending else 4
        if len(data_frames) != expected_frames:
            raise ValueError(
                "malformed lookup request: expected "
                f"{expected_frames} frames, got {len(data_frames)}"
            )

        lookup_id_value = data_frames[-2]
        request_configs_value = data_frames[-1]
        if not isinstance(lookup_id_value, (bytes, str)):
            raise TypeError("lookup_id must be bytes or str")
        if not isinstance(request_configs_value, (bytes, str)):
            raise TypeError("request_configs must be bytes or str")

        lookup_id = (
            lookup_id_value.decode("utf-8")
            if isinstance(lookup_id_value, bytes)
            else lookup_id_value
        )
        request_configs_str = (
            request_configs_value.decode("utf-8")
            if isinstance(request_configs_value, bytes)
            else request_configs_value
        )
        request_configs = (
            json.loads(request_configs_str) if request_configs_str else None
        )
        if request_configs is not None and not isinstance(request_configs, dict):
            raise TypeError("request_configs must decode to a dictionary")

        payload = data_frames[0]
        if (
            isinstance(payload, dict)
            and payload.get("schema") == _LOOKUP_CONTROL_SCHEMA
        ):
            generation = _decode_lookup_control_request(payload)
            if request_configs is not None:
                raise ValueError("lookup control requests cannot carry configs")
            if not self.enable_blending and data_frames[1] != []:
                raise ValueError(
                    "non-CacheBlend lookup control placeholder must be empty"
                )
            self._cancel_lookup_generation(lookup_id, generation)
            return _LOOKUP_CANCEL_ACK

        if not self.enable_blending:
            with self._engine_pin_guard():
                lookup_pins = getattr(self.lmcache_engine, "lookup_pins", None)
                if lookup_pins is not None and lookup_id in lookup_pins:
                    raise ValueError("lookup_id already has active pin ownership")
            lookup_result = self.lmcache_engine.lookup(
                hashes=data_frames[0],
                offsets=data_frames[1],
                lookup_id=lookup_id,
                pin=True,
                request_configs=request_configs,
            )
        else:
            if isinstance(payload, dict):
                if not self.supports_explicit_spans():
                    raise NotImplementedError(
                        "worker does not support explicit-span lookup requests"
                    )
                tokens, explicit_spans, explicit_generation = (
                    _decode_explicit_lookup_request(payload)
                )
                # One engine-lifetime transaction linearizes old/new lookup
                # server objects against exact generation cancellation. The
                # engine lookup reacquires this RLock when publishing pins.
                with self._engine_pin_guard():
                    active, cancelled = self._generation_state()
                    if getattr(
                        self.lmcache_engine,
                        "lookup_explicit_replay_fenced",
                        False,
                    ):
                        return _encode_explicit_lookup_result(
                            0,
                            explicit_generation,
                        )
                    cancelled_key = (lookup_id, explicit_generation)
                    if cancelled_key in cancelled:
                        # Cancel won before this delayed lookup acquired the
                        # transaction. It cannot publish pins afterward.
                        cancelled.move_to_end(cancelled_key)
                        return _encode_explicit_lookup_result(
                            0,
                            explicit_generation,
                        )

                    active_generation = active.get(lookup_id)
                    durable = self._engine_generation_state()
                    durable_generation = durable.get(lookup_id)
                    lookup_pins = getattr(self.lmcache_engine, "lookup_pins", None)
                    if (
                        active_generation is None
                        and lookup_pins is not None
                        and lookup_id in lookup_pins
                    ):
                        if durable_generation is None:
                            raise ValueError(
                                "lookup_id already has legacy pin ownership"
                            )
                        if durable_generation != explicit_generation:
                            raise ValueError(
                                "lookup_id already has a different active generation"
                            )
                        active[lookup_id] = durable_generation
                        # The lookup server was recreated, or this is a
                        # transport replay. Durable engine ownership proves the
                        # generation. Adopt it into this server's diagnostic
                        # alias, but duplicate positive hits remain unsafe.
                        return _encode_explicit_lookup_result(
                            0,
                            explicit_generation,
                        )
                    if (
                        active_generation is not None
                        and durable_generation is not None
                        and active_generation != durable_generation
                    ):
                        raise RuntimeError(
                            "explicit generation ownership is inconsistent"
                        )
                    if (
                        active_generation is not None
                        and lookup_pins is not None
                        and lookup_id not in lookup_pins
                    ):
                        active.pop(lookup_id, None)
                        durable.pop(lookup_id, None)
                        self._remember_cancelled_generation(
                            lookup_id,
                            active_generation,
                        )
                        self._remember_cancelled_generation(
                            lookup_id,
                            explicit_generation,
                        )
                        return _encode_explicit_lookup_result(
                            0,
                            explicit_generation,
                        )
                    if active_generation is None and durable_generation is not None:
                        # A durable owner without physical pins is a completed
                        # generation whose server-local alias was lost (for
                        # example across server replacement). Never overwrite
                        # it with a new generation: retire both identities and
                        # suppress this replay/request conservatively.
                        durable.pop(lookup_id, None)
                        self._remember_cancelled_generation(
                            lookup_id,
                            durable_generation,
                        )
                        self._remember_cancelled_generation(
                            lookup_id,
                            explicit_generation,
                        )
                        return _encode_explicit_lookup_result(
                            0,
                            explicit_generation,
                        )
                    if active_generation is not None:
                        if durable_generation is None:
                            # Heal pre-transaction server-local ownership so a
                            # replacement server can cancel it exactly.
                            durable[lookup_id] = active_generation
                        if active_generation != explicit_generation:
                            raise ValueError(
                                "lookup_id already has a different active generation"
                            )
                        # A duplicate RPC cannot safely reuse the positive
                        # result: worker-finally may release its sole pin owner
                        # immediately after any presence check. Fail closed to
                        # a generation-bound miss and leave original ownership
                        # for lifecycle cancellation.
                        return _encode_explicit_lookup_result(0, explicit_generation)

                    # Publish exact ownership before the engine can publish a
                    # physical pin. The engine lookup reacquires this RLock;
                    # therefore even an exception from its post-lookup hooks
                    # leaves every physical pin bound to a durable generation.
                    durable[lookup_id] = explicit_generation
                    active[lookup_id] = explicit_generation
                    try:
                        lookup_result = self.lmcache_engine.lookup(
                            tokens=tokens,
                            lookup_id=lookup_id,
                            pin=True,
                            request_configs=request_configs,
                            explicit_spans=explicit_spans,
                        )
                    except BaseException:
                        lookup_pins = getattr(
                            self.lmcache_engine,
                            "lookup_pins",
                            None,
                        )
                        if lookup_pins is None or lookup_id not in lookup_pins:
                            if active.get(lookup_id) == explicit_generation:
                                active.pop(lookup_id, None)
                            if durable.get(lookup_id) == explicit_generation:
                                durable.pop(lookup_id, None)
                            self._remember_cancelled_generation(
                                lookup_id,
                                explicit_generation,
                            )
                        raise
                    lookup_pins = getattr(self.lmcache_engine, "lookup_pins", None)
                    if lookup_pins is not None and lookup_id in lookup_pins:
                        if (
                            durable.get(lookup_id) != explicit_generation
                            or active.get(lookup_id) != explicit_generation
                        ):
                            raise RuntimeError(
                                "explicit generation ownership changed during lookup"
                            )
                    else:
                        # Zero-hit explicit lookups have no worker ownership. A
                        # positive result without a pin owner is unsafe too.
                        if active.get(lookup_id) == explicit_generation:
                            active.pop(lookup_id, None)
                        if durable.get(lookup_id) == explicit_generation:
                            durable.pop(lookup_id, None)
                        self._remember_cancelled_generation(
                            lookup_id,
                            explicit_generation,
                        )
                        lookup_result = 0
                    return _encode_explicit_lookup_result(
                        lookup_result,
                        explicit_generation,
                    )
            else:
                tokens = payload
                with self._engine_pin_guard():
                    lookup_pins = getattr(self.lmcache_engine, "lookup_pins", None)
                    if lookup_pins is not None and lookup_id in lookup_pins:
                        raise ValueError("lookup_id already has active pin ownership")
                lookup_result = self.lmcache_engine.lookup(
                    tokens=tokens,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                    explicit_spans=None,
                )
        return lookup_result.to_bytes(4, "big")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        # Stop the processing thread first
        self.running = False

        # Wait for thread to finish with timeout
        if self.thread.is_alive():
            self.thread.join(timeout=_LOOKUP_SERVER_CLOSE_TIMEOUT_SECONDS)
            if self.thread.is_alive():
                # A live request thread may still be inside ``engine.lookup``.
                # Returning success would let dynamic recreation or terminal
                # shutdown replace/destroy the engine underneath that call.
                # Keep the transport intact so a later close retry can finish
                # after the request quiesces.
                raise RuntimeError("lookup server thread did not quiesce")

        # Close transport after thread is stopped
        self.transport.close()
