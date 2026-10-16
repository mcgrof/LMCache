# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.request_fingerprint import (
    make_lookup_request_fingerprint,
    make_serializable_token_ids,
)
from lmcache.v1.token_span import ExplicitTokenSpans

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

_BYPASS_CANCEL_RETRY_INTERVAL_SECONDS = 0.25
_BYPASS_CANCEL_CLOSE_TIMEOUT_SECONDS = 1.0
_TRANSFERRED_LOOKUP_ID_LIMIT = 4096
_BYPASS_CLIENT_OPEN = "open"
_BYPASS_CLIENT_DRAINING = "draining"
_BYPASS_CLIENT_CLOSED = "closed"


class LMCacheBypassLookupClient(LookupClientInterface):
    """
    Bypass lookup client that directly calls LMCacheEngine.lookup()
    instead of using ZMQ communication. This is particularly useful
    for MLA scenarios where only rank 0 needs to perform lookups.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: "LMCacheMetadata",
        lmcache_engine: "LMCacheEngine",
    ):
        """
        Initialize the bypass lookup client.

        Args:
            config: The LMCacheEngine configuration
            metadata: The LMCacheEngine metadata
            lmcache_engine: The LMCacheEngine instance to use for lookups
        """
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration should be passed."
        )

        self.lmcache_engine = lmcache_engine
        self.config = config

        # Use the token database from the provided LMCacheEngine
        self.token_database = self.lmcache_engine.token_database
        self.enable_blending = self.config.enable_blending
        self.reqs_status: dict[str, Optional[int]] = {}
        self.reqs_fingerprints: dict[str, bytes] = {}
        self.pending_cancel_ids: set[str] = set()
        # KMesh scheduler-attempt UUIDs are one-shot. Retain transfer
        # tombstones for this experimental client lifetime; there is no safe
        # horizon for forgetting a delayed scheduler cancel yet. Capacity
        # exhaustion fences admission rather than evicting ABA protection.
        self.transferred_lookup_ids: set[str] = set()
        self.lookup_transfer_fenced = False
        self._cancel_state_lock = threading.Lock()
        self._engine_cancel_lock = threading.RLock()
        self._cancel_close_lock = threading.Lock()
        self._cancel_retry_timer: Optional[threading.Timer] = None
        self._cancel_retry_closing = False
        self._cancel_lifecycle_state = _BYPASS_CLIENT_OPEN

        logger.info("LMCacheBypassLookupClient initialized")

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        self._ensure_cancel_state()
        with self._cancel_state_lock:
            if lookup_id in self.pending_cancel_ids:
                return None
            return self.reqs_status.get(lookup_id, -1)

    def _validate_and_bind_lookup_request(
        self,
        token_ids: Union["torch.Tensor", list[int]],
        lookup_id: str,
        request_configs: Optional[dict],
        explicit_spans: Optional[ExplicitTokenSpans],
    ) -> list[int]:
        if explicit_spans is not None and not self.supports_explicit_spans():
            raise NotImplementedError(
                "bypass lookup explicit spans require layerwise CacheBlend"
            )
        serializable_ids = make_serializable_token_ids(token_ids)
        request_fingerprint = make_lookup_request_fingerprint(
            serializable_ids,
            request_configs,
            explicit_spans,
        )
        self._ensure_cancel_state()
        with self._cancel_state_lock:
            if self._cancel_lifecycle_state != _BYPASS_CLIENT_OPEN:
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
        token_ids: Union["torch.Tensor", list[int]],
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
        token_ids: Union["torch.Tensor", list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> Optional[int]:
        self._ensure_cancel_state()
        with self._engine_cancel_lock:
            with self._cancel_state_lock:
                if self._cancel_lifecycle_state != _BYPASS_CLIENT_OPEN:
                    raise RuntimeError("lookup client is not accepting new ownership")
            return self._lookup(
                token_ids,
                lookup_id,
                request_configs,
                explicit_spans,
            )

    def _lookup(
        self,
        token_ids: Union["torch.Tensor", list[int]],
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

        try:
            if explicit_spans is not None:
                result = self.lmcache_engine.lookup(
                    tokens=serializable_ids,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                    explicit_spans=explicit_spans,
                )
            elif not self.enable_blending:
                # Process tokens to get hashes and offsets
                hashes = []
                offsets = []
                for start, end, key in self.token_database.process_tokens(
                    token_ids, make_key=False
                ):
                    hashes.append(key)
                    offsets.append(end - start)
                if not hashes:
                    self.reqs_status[lookup_id] = 0
                    return 0

                # Call LMCacheEngine lookup with hashes and offsets
                result = self.lmcache_engine.lookup(
                    hashes=hashes,
                    offsets=offsets,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                )
            else:
                # For blending mode, pass tokens directly
                result = self.lmcache_engine.lookup(
                    tokens=token_ids,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                )

            self.reqs_status[lookup_id] = result
            return result

        except Exception as e:
            logger.error("Error in bypass lookup: %s", e)
            self.reqs_status[lookup_id] = 0
            return 0

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.reqs_status.pop(lookup_id, None)
        self.reqs_fingerprints.pop(lookup_id, None)

    def transfer_lookup(self, lookup_id: str) -> None:
        """Transfer colocated pin cleanup to the allocated worker path."""
        self.transfer_lookups([lookup_id])

    def transfer_lookups(self, lookup_ids: list[str]) -> None:
        """Retire a fully preflighted colocated batch without unpinning."""
        self._ensure_cancel_state()
        with self._engine_cancel_lock:
            with self._cancel_state_lock:
                if self._cancel_lifecycle_state != _BYPASS_CLIENT_OPEN:
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
                    self.clear_lookup_status(lookup_id)
                    self.transferred_lookup_ids.add(lookup_id)
                if len(self.transferred_lookup_ids) >= _TRANSFERRED_LOOKUP_ID_LIMIT:
                    self.lookup_transfer_fenced = True

    def finish_lookup(self, lookup_id: str) -> None:
        """Quiesce lookup publication, then release scheduler-owned pins."""
        self._ensure_cancel_state()
        with self._engine_cancel_lock:
            with self._cancel_state_lock:
                scheduler_owned = lookup_id in self._owned_lookup_ids_locked()
            if scheduler_owned:
                self.cancel_lookup(lookup_id)
                return
            with self._cancel_state_lock:
                self.clear_lookup_status(lookup_id)

    def _ensure_cancel_state(self) -> None:
        if not hasattr(self, "pending_cancel_ids"):
            self.pending_cancel_ids = set()
        if not hasattr(self, "transferred_lookup_ids"):
            self.transferred_lookup_ids = set()
        if not hasattr(self, "lookup_transfer_fenced"):
            self.lookup_transfer_fenced = False
        if not hasattr(self, "_cancel_state_lock"):
            self._cancel_state_lock = threading.Lock()
        if not hasattr(self, "_engine_cancel_lock"):
            self._engine_cancel_lock = threading.RLock()
        if not hasattr(self, "_cancel_close_lock"):
            self._cancel_close_lock = threading.Lock()
        if not hasattr(self, "_cancel_retry_timer"):
            self._cancel_retry_timer = None
        if not hasattr(self, "_cancel_retry_closing"):
            self._cancel_retry_closing = False
        if not hasattr(self, "_cancel_lifecycle_state"):
            self._cancel_lifecycle_state = (
                _BYPASS_CLIENT_CLOSED
                if self._cancel_retry_closing
                else _BYPASS_CLIENT_OPEN
            )

    def _owned_lookup_ids_locked(self) -> set[str]:
        return set(self.pending_cancel_ids).union(
            getattr(self, "reqs_fingerprints", {}),
            getattr(self, "reqs_status", {}),
        )

    def _enqueue_owned_lookup_cancels(self) -> None:
        with self._cancel_state_lock:
            self.pending_cancel_ids.update(self._owned_lookup_ids_locked())

    def _drain_pending_cancels(self, deadline: float) -> bool:
        while True:
            with self._cancel_state_lock:
                lookup_ids = tuple(self.pending_cancel_ids)
            if not lookup_ids:
                return True

            for lookup_id in lookup_ids:
                if time.monotonic() >= deadline:
                    return False
                if self._attempt_cancel(lookup_id):
                    self._complete_cancel(lookup_id)

            with self._cancel_state_lock:
                if not self.pending_cancel_ids:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.01, remaining))

    def _complete_cancel(self, lookup_id: str) -> None:
        with self._cancel_state_lock:
            if lookup_id not in self.pending_cancel_ids:
                return
            self.clear_lookup_status(lookup_id)
            self.pending_cancel_ids.remove(lookup_id)

    def _attempt_cancel(self, lookup_id: str) -> bool:
        with self._engine_cancel_lock:
            # Pair this check with the close-time engine barrier. An attempt
            # that has not entered the engine before close begins must not do
            # so after the manager starts tearing that engine down.
            with self._cancel_state_lock:
                if (
                    self._cancel_lifecycle_state == _BYPASS_CLIENT_CLOSED
                    or lookup_id not in self.pending_cancel_ids
                ):
                    return False
            try:
                self.lmcache_engine.lookup_unpin(lookup_id)
                return True
            except Exception:
                logger.exception("Failed to cancel bypass lookup %s", lookup_id)
                return False

    def _schedule_cancel_retry(self) -> None:
        with self._cancel_state_lock:
            if (
                self._cancel_lifecycle_state == _BYPASS_CLIENT_CLOSED
                or not self.pending_cancel_ids
            ):
                return
            if self._cancel_retry_timer is not None:
                return
            timer = threading.Timer(
                _BYPASS_CANCEL_RETRY_INTERVAL_SECONDS,
                self._retry_pending_cancels,
            )
            timer.daemon = True
            timer.name = "bypass-lookup-cancel-retry"
            self._cancel_retry_timer = timer
            timer.start()

    def _retry_pending_cancels(self) -> None:
        with self._cancel_state_lock:
            self._cancel_retry_timer = None
            if self._cancel_lifecycle_state == _BYPASS_CLIENT_CLOSED:
                return
            lookup_ids = tuple(self.pending_cancel_ids)
        for lookup_id in lookup_ids:
            with self._cancel_state_lock:
                if (
                    self._cancel_lifecycle_state == _BYPASS_CLIENT_CLOSED
                    or lookup_id not in self.pending_cancel_ids
                ):
                    continue
            if self._attempt_cancel(lookup_id):
                self._complete_cancel(lookup_id)
        self._schedule_cancel_retry()

    def cancel_lookup(self, lookup_id: str) -> None:
        """Cancel lookup resources on the colocated bypass engine."""
        self._ensure_cancel_state()
        with self._engine_cancel_lock:
            with self._cancel_state_lock:
                if (
                    self._cancel_lifecycle_state == _BYPASS_CLIENT_CLOSED
                    or lookup_id in self.pending_cancel_ids
                ):
                    return
                if lookup_id in self.transferred_lookup_ids:
                    logger.warning(
                        "Ignoring bypass cancel for transferred worker owner %s",
                        lookup_id,
                    )
                    return
                if (
                    self._cancel_lifecycle_state == _BYPASS_CLIENT_DRAINING
                    and lookup_id not in self._owned_lookup_ids_locked()
                ):
                    logger.warning(
                        "Ignoring unowned bypass lookup cancel for %s while client "
                        "drains",
                        lookup_id,
                    )
                    return
                self.pending_cancel_ids.add(lookup_id)
            if self._attempt_cancel(lookup_id):
                self._complete_cancel(lookup_id)
                return
            self._schedule_cancel_retry()

    def supports_producer_reuse(self) -> bool:
        return True

    def supports_explicit_spans(self) -> bool:
        return self.lmcache_engine.supports_explicit_spans()

    def close(self) -> None:
        """Drain scheduler ownership and fence calls into the bypass engine.

        Raises:
            RuntimeError: Exact unpin did not converge within the close retry
                window. An individual engine call is not preempted, so wall
                time can exceed the nominal retry window.
        """
        self._ensure_cancel_state()
        with self._cancel_close_lock:
            with self._cancel_state_lock:
                if self._cancel_lifecycle_state == _BYPASS_CLIENT_CLOSED:
                    return
                self._cancel_lifecycle_state = _BYPASS_CLIENT_DRAINING
                timer = self._cancel_retry_timer
                self._cancel_retry_timer = None
            if timer is not None:
                timer.cancel()

            # Quiesce an admitted lookup through engine return and local
            # result publication. New lookups and transfers are rejected by
            # the DRAINING state before they can acquire ownership.
            with self._engine_cancel_lock:
                pass
            self._enqueue_owned_lookup_cancels()
            deadline = time.monotonic() + _BYPASS_CANCEL_CLOSE_TIMEOUT_SECONDS
            if not self._drain_pending_cancels(deadline):
                with self._cancel_state_lock:
                    self._cancel_lifecycle_state = _BYPASS_CLIENT_OPEN
                    self._cancel_retry_closing = False
                self._schedule_cancel_retry()
                raise RuntimeError(
                    "bypass lookup client close timed out with cancellation "
                    "ownership still pending"
                )

            # Exclude a retry attempt that took its outer snapshot before the
            # drain completed. Publishing CLOSED under the engine-call lock
            # prevents that attempt from reaching a soon-to-be-destroyed
            # engine after successful close.
            with self._engine_cancel_lock:
                with self._cancel_state_lock:
                    self._cancel_lifecycle_state = _BYPASS_CLIENT_CLOSED
                    self._cancel_retry_closing = True
                    timer = self._cancel_retry_timer
                    self._cancel_retry_timer = None
            if timer is not None:
                timer.cancel()
            logger.info("LMCacheBypassLookupClient closed")

    def force_close(self) -> None:
        """Fence calls after transferring every scheduler cleanup to the engine.

        A manager may share this process-local engine with another manager. In
        that topology force-closing this client does not imply immediate engine
        destruction, so terminal client shutdown must leave every lookup UUID
        with the engine's durable retry owner before local ownership is dropped.
        """
        self._ensure_cancel_state()
        with self._cancel_close_lock:
            with self._cancel_state_lock:
                if self._cancel_lifecycle_state == _BYPASS_CLIENT_CLOSED:
                    return
                self._cancel_lifecycle_state = _BYPASS_CLIENT_DRAINING
                timer = self._cancel_retry_timer
                self._cancel_retry_timer = None
            if timer is not None:
                timer.cancel()

            with self._engine_cancel_lock:
                with self._cancel_state_lock:
                    lookup_ids = tuple(sorted(self._owned_lookup_ids_locked()))
                    self.pending_cancel_ids.update(lookup_ids)

                try:
                    # Register the whole batch before starting any fallible
                    # per-ID cleanup. A slow first unpin must not prevent a
                    # later UUID from acquiring a durable retry owner.
                    for lookup_id in lookup_ids:
                        self.lmcache_engine.register_lookup_cleanup(lookup_id)
                    for lookup_id in lookup_ids:
                        self.lmcache_engine.schedule_lookup_unpin(lookup_id)
                except Exception as error:
                    # Keep this client published and retryable if the engine
                    # could not accept the complete ownership transfer.
                    with self._cancel_state_lock:
                        self._cancel_lifecycle_state = _BYPASS_CLIENT_OPEN
                        self._cancel_retry_closing = False
                    self._schedule_cancel_retry()
                    raise RuntimeError(
                        "failed to transfer bypass lookup cleanup ownership "
                        "to the engine"
                    ) from error

                with self._cancel_state_lock:
                    for lookup_id in lookup_ids:
                        self.clear_lookup_status(lookup_id)
                        self.pending_cancel_ids.discard(lookup_id)
                    self._cancel_lifecycle_state = _BYPASS_CLIENT_CLOSED
                    self._cancel_retry_closing = True
                    timer = self._cancel_retry_timer
                    self._cancel_retry_timer = None
            if timer is not None:
                timer.cancel()
            logger.info("LMCacheBypassLookupClient force-closed")
