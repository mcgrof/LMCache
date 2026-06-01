# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from abc import abstractmethod
from collections import Counter
from typing import TYPE_CHECKING
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction import L1EvictionPolicy, L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy import CreateEvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.distributed.storage_placement import (
    SplitTierManifest,
    SplitTierState,
    derive_component_key,
    reverse_component_key,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.quota_manager import QuotaManager

logger = init_logger(__name__)


class EvictionController(StorageControllerInterface):
    """
    Abstract base class for eviction controllers.

    Provides the shared eviction loop structure: background thread and stop
    flag. Subclasses implement eviction_loop and execute_eviction_action
    for their specific tier (L1 or L2).
    """

    def __init__(self):
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self.eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    @abstractmethod
    def report_status(self) -> dict:
        """Return a status dict for this controller.

        The child class needs to override this function to report
        controller-specific health and configuration information.
        """
        pass

    @abstractmethod
    def eviction_loop(self):
        """Run the eviction loop.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass

    @abstractmethod
    def execute_eviction_action(self, action: EvictionAction):
        """Execute a single eviction action.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass


class L1EvictionController(EvictionController):
    """
    Eviction controller for L1 cache.

    Uses an L1EvictionPolicy bridge to keep the eviction policy up-to-date
    with L1 manager events, and periodically triggers eviction based on
    L1 memory usage.
    """

    def __init__(
        self,
        l1_manager: L1Manager,
        eviction_config: EvictionConfig,
        split_tier_manifest: "SplitTierManifest | None" = None,
        l2_adapters: "list[L2AdapterInterface] | None" = None,
    ):
        super().__init__()
        self._eviction_config = eviction_config
        self._eviction_policy = CreateEvictionPolicy(eviction_config)
        self._l1_manager = l1_manager
        self._listener = L1EvictionPolicy(self._eviction_policy)
        self._l1_manager.register_listener(self._listener)
        self._event_bus = get_event_bus()
        # Phase 2 (PR-3') paired-eviction wiring.  ``None`` for both
        # preserves the legacy DISCARD-only behavior; callers that wire
        # KV_SPLIT_TIER pass both so the controller can drive the
        # INVALIDATED -> DELETE_IN_FLIGHT lifecycle and enqueue V child
        # deletes against the L2 adapters.
        self._split_tier_manifest = split_tier_manifest
        self._l2_adapters = list(l2_adapters or [])

    def set_split_tier_paired_eviction(
        self,
        manifest: "SplitTierManifest",
        l2_adapters: "list[L2AdapterInterface]",
    ) -> None:
        """Wire paired-eviction state post-construction.

        The L1EvictionController is constructed before the L2 adapters
        in the StorageManager init order (the L1 manager wants its
        listener registered before any allocate event fires).  This
        setter lets the StorageManager attach the manifest + L2
        adapters once they exist.  Calling more than once replaces the
        previous wiring; passing empty adapters reverts to legacy
        DISCARD-only behavior.
        """
        self._split_tier_manifest = manifest
        self._l2_adapters = list(l2_adapters)

    def report_status(self) -> dict:
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "eviction_policy": self._eviction_config.eviction_policy,
            "trigger_watermark": self._eviction_config.trigger_watermark,
            "eviction_ratio": self._eviction_config.eviction_ratio,
        }

    def _publish_skipped(self, usage: float, watermark: float) -> None:
        """Publish a below-watermark loop tick (no eviction this cycle)."""
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={
                    "usage": usage,
                    "watermark": watermark,
                    "triggered": False,
                },
            )
        )

    def _publish_triggered(self, usage: float, watermark: float) -> None:
        """Publish an above-watermark loop tick (eviction policy ran)."""
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={
                    "usage": usage,
                    "watermark": watermark,
                    "triggered": True,
                },
            )
        )

    def eviction_loop(self):
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio

        while not self._stop_flag.is_set():
            time.sleep(1)
            used_bytes, total_bytes = self._l1_manager.get_memory_usage()
            usage = 0 if total_bytes == 0 else used_bytes / total_bytes
            if usage < watermark:
                logger.debug(
                    "L1 memory usage %.2f below watermark %.2f; skipping eviction.",
                    usage,
                    watermark,
                )
                self._publish_skipped(usage, watermark)
                continue

            logger.info(
                "L1 memory usage %.2f above watermark %.2f; triggering eviction.",
                usage,
                watermark,
            )
            actions = self._eviction_policy.get_eviction_actions(
                eviction_ratio,
                key_eligible_filter=self._l1_manager.is_key_evictable,
            )
            for action in actions:
                self.execute_eviction_action(action)
            self._publish_triggered(usage, watermark)

    def execute_eviction_action(self, action: EvictionAction):
        if action.destination == EvictionDestination.DISCARD:
            self._evict_with_split_tier_pairing(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            self._evict_with_split_tier_pairing(action.keys)

    def _evict_with_split_tier_pairing(
        self, keys: "list[ObjectKey]"
    ) -> None:
        """Discard L1 keys; for split-tier K children, also invalidate
        the manifest and enqueue the paired V child for L2 deletion.

        The order matters for correctness (codex correction: logical
        invalidation FIRST, physical cleanup second).  For each key
        that looks like a K child:

          1. Resolve the logical key via :func:`reverse_component_key`.
          2. Mark the manifest INVALIDATED (lookup returns miss
             immediately; the load path masks any in-flight reader's
             bitmap to miss for this key).
          3. Enqueue the V child delete against every configured L2
             adapter (best-effort; adapters whose ``delete()`` is a
             no-op leave the V orphan as tolerable cold-tier waste).
          4. Transition the manifest to DELETE_IN_FLIGHT.

        After the per-key paired step, the actual L1 ``delete()`` runs
        on the original ``keys`` list so the K child (and any
        non-split-tier keys in the same batch) get removed in one
        call.  Manifest entries are dropped only after the L2 delete
        future settles -- but since the L2 ``delete()`` API is
        fire-and-forget, we drop the manifest entry here once
        DELETE_IN_FLIGHT is set (the V child is no longer addressable
        through this StorageManager regardless of whether the L2
        adapter physically purged it).
        """
        if self._split_tier_manifest is None or not self._l2_adapters:
            # Legacy DISCARD-only path; no split-tier semantics wired.
            self._l1_manager.delete(keys)
            return

        for k in keys:
            decomposed = reverse_component_key(k)
            if decomposed is None:
                # Not a derived child key; nothing to pair.
                continue
            logical_key, role = decomposed
            if role != "k":
                # Only K-child evictions trigger the paired cleanup.
                # V-children live on L2; the L2 eviction controller
                # handles them via its own policy.
                continue
            state = self._split_tier_manifest.lookup(logical_key)
            if state is None:
                # Manifest already cleaned up; nothing to do.
                continue
            if state in (
                SplitTierState.INVALIDATED,
                SplitTierState.DELETE_IN_FLIGHT,
            ):
                # Idempotent on repeat eviction of the same K child.
                continue
            # Step 1+2: logical invalidation first.  Lookup returns
            # miss from here on; readers observe consistent state.
            self._split_tier_manifest.mark_invalidated(logical_key)
            # Step 3: enqueue V child delete against every L2
            # adapter.  Best-effort; default ``delete()`` is no-op
            # on adapters that don't support eviction, which leaves
            # the V orphan as tolerable cold-tier waste.
            v_child_key = derive_component_key(logical_key, "v")
            for adapter in self._l2_adapters:
                try:
                    adapter.delete([v_child_key])
                except Exception:
                    logger.exception(
                        "L1EvictionController paired-evict: "
                        "adapter %s delete(v_child) raised for "
                        "logical %s",
                        type(adapter).__name__,
                        logical_key,
                    )
            # Step 4: transition to DELETE_IN_FLIGHT then drop the
            # manifest entry (the V child is no longer addressable
            # through us regardless of L2 physical purge).
            try:
                self._split_tier_manifest.mark_delete_in_flight(logical_key)
            except ValueError:
                # Lost a race against a concurrent invalidation /
                # delete; the manifest is already INVALIDATED or
                # DELETE_IN_FLIGHT.  Idempotent.
                pass
            self._split_tier_manifest.drop(logical_key)

        # Physical cleanup of K child (and any other K_TOGETHER keys
        # in the batch).  Runs after the manifest invalidations so
        # readers never observe the K child gone while the manifest
        # still says COMPLETE.
        self._l1_manager.delete(keys)


class L2AdapterEvictionState:
    """Per-adapter eviction state: its own policy, listener, and config."""

    def __init__(
        self,
        adapter_id: int,
        adapter: L2AdapterInterface,
        eviction_config: EvictionConfig,
    ):
        self.adapter_id = adapter_id
        self.adapter = adapter
        self.eviction_config = eviction_config
        self.eviction_policy = CreateEvictionPolicy(eviction_config)
        self.listener = L2EvictionPolicy(self.eviction_policy)
        adapter.register_listener(self.listener)


class L2EvictionController(StorageControllerInterface):
    """
    Unified eviction controller for all L2 adapters.

    Each adapter gets its own eviction policy and listener bridge, but a
    single background thread loops over all of them.

    When the adapter's policy sets ``support_isolation == True``
    (e.g. :class:`IsolatedLRUEvictionPolicy`), the controller consults
    the injected :class:`QuotaManager` to decide which ``cache_salt``
    buckets are over budget and evicts from each one in isolation.
    Otherwise it uses the adapter's aggregate ``usage_fraction``
    against the configured watermark — unchanged from the pre-PR5
    behavior.
    """

    def __init__(
        self,
        l2_adapter_states: list[L2AdapterEvictionState],
        quota_manager: QuotaManager | None = None,
    ):
        self._adapter_states = l2_adapter_states
        self._quota_manager = quota_manager
        # Guards _adapter_states against concurrent runtime add/remove.
        self._states_lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    def add_adapter_state(self, state: L2AdapterEvictionState) -> None:
        """Register a new adapter's eviction state at runtime."""
        with self._states_lock:
            self._adapter_states.append(state)

    def remove_adapter_state(self, adapter_id: int) -> None:
        """Drop the eviction state for ``adapter_id``.

        Blocks until any in-progress eviction pass finishes (it holds the
        same lock), so the adapter is guaranteed idle here before the
        caller closes it. A no-op if the adapter has no eviction state.
        """
        with self._states_lock:
            self._adapter_states = [
                s for s in self._adapter_states if s.adapter_id != adapter_id
            ]

    def report_status(self) -> dict:
        # NOTE: ``usage.bytes_by_cache_salt`` is intentionally NOT
        # surfaced here. A deployment can have 10k+ salts, so embedding
        # the full bucket map in the status response would blow up the
        # payload. Per-salt inspection goes through the dedicated HTTP
        # quota endpoints (which pull from ``QuotaManager`` +
        # ``StorageManager.get_usage_bytes_by_cache_salt``).
        adapter_statuses = []
        with self._states_lock:
            states = list(self._adapter_states)
        for state in states:
            usage = state.adapter.get_usage()
            adapter_statuses.append(
                {
                    "eviction_policy": state.eviction_config.eviction_policy,
                    "trigger_watermark": state.eviction_config.trigger_watermark,
                    "eviction_ratio": state.eviction_config.eviction_ratio,
                    "current_usage": usage.usage_fraction,
                    "total_bytes_used": usage.total_bytes_used,
                    "total_capacity_bytes": usage.total_capacity_bytes,
                    "num_cache_salt_buckets": len(usage.bytes_by_cache_salt),
                }
            )
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "adapters": adapter_statuses,
        }

    def _eviction_loop(self):
        while not self._stop_flag.is_set():
            time.sleep(1)
            # Hold the lock across the whole pass so remove_adapter_state
            # cannot detach (and the caller close) an adapter while we are
            # calling into it.
            with self._states_lock:
                for state in self._adapter_states:
                    self._check_and_evict(state)

    def _check_and_evict(self, state: L2AdapterEvictionState):
        if state.eviction_policy.support_isolation and self._quota_manager is not None:
            self._check_and_evict_by_cache_salt(state)
        else:
            self._check_and_evict_global(state)

    def _check_and_evict_global(self, state: L2AdapterEvictionState):
        """Aggregate-usage eviction (``LRU`` / ``noop``)."""
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio

        # ``usage_fraction == -1`` means the adapter doesn't support
        # usage-based eviction (no max_capacity_bytes declared), so we
        # do not trigger eviction. Adapters with ``supports_global_eviction ==
        # False`` should already have been filtered out at construction
        # time in ``StorageManager``; this check is a defensive belt.
        current_usage = state.adapter.get_usage().usage_fraction
        if current_usage < 0 or current_usage < watermark:
            logger.debug(
                "L2 usage %.2f below watermark %.2f; skipping eviction.",
                current_usage,
                watermark,
            )
            return

        logger.info(
            "L2 usage %.2f above watermark %.2f; triggering eviction.",
            current_usage,
            watermark,
        )
        actions = state.eviction_policy.get_eviction_actions(eviction_ratio)
        for action in actions:
            self._execute_eviction_action(state.adapter, action)

    def _check_and_evict_by_cache_salt(self, state: L2AdapterEvictionState):
        """Per-``cache_salt`` eviction driven by :class:`QuotaManager`.

        For every salt with non-zero bytes, compare its usage against
        ``watermark * quota``. Salts over threshold get eviction scoped
        to their own LRU list. Salts with no quota registered have an
        effective limit of ``0`` and are therefore always over budget,
        so they get a full eviction (``effective_ratio=1.0``) — this
        enforces the allowlist rule: only registered salts retain data.

        Per-destination keys are batched across all over-budget salts
        before invoking the adapter — one ``adapter.delete(...)`` call
        per destination instead of one per (salt, destination) pair.
        Adapters with non-trivial per-call overhead (NIXL handle setup,
        FS sync, etc.) see this as a real win when many salts go over
        budget in the same cycle.
        """
        assert self._quota_manager is not None
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio
        usage = state.adapter.get_usage()

        # destination -> accumulated keys across all over-budget salts.
        pending: dict[EvictionDestination, list[ObjectKey]] = {}

        for cache_salt, user_bytes in usage.bytes_by_cache_salt.items():
            if user_bytes <= 0:
                continue
            limit = self._quota_manager.get_limit_bytes(cache_salt)
            # Trigger on ``>=`` to match the global branch's ``usage <
            # watermark`` short-circuit. Salts with no quota (limit=0)
            # always land here because ``user_bytes > 0 >= 0``.
            if user_bytes < watermark * limit:
                continue

            # Unregistered / zero-quota salts: wipe everything.
            # Registered salts: evict the configured ratio of their list.
            effective_ratio = 1.0 if limit == 0 else eviction_ratio
            logger.info(
                "cache_salt=%r over quota (bytes=%d, limit=%d, "
                "watermark=%.2f); evicting ratio=%.2f.",
                cache_salt,
                user_bytes,
                limit,
                watermark,
                effective_ratio,
            )
            actions = state.eviction_policy.get_eviction_actions(
                effective_ratio, cache_salt=cache_salt
            )
            for action in actions:
                pending.setdefault(action.destination, []).extend(action.keys)

        for destination, keys in pending.items():
            self._execute_eviction_action(
                state.adapter,
                EvictionAction(keys=keys, destination=destination),
            )

    def _execute_eviction_action(
        self, adapter: L2AdapterInterface, action: EvictionAction
    ):
        if action.destination == EvictionDestination.DISCARD:
            adapter.delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            adapter.delete(action.keys)

        if action.keys:
            get_event_bus().publish(
                Event(
                    event_type=EventType.L2_KEYS_EVICTED,
                    metadata={
                        "key_count": len(action.keys),
                        "key_count_per_salt": Counter(
                            k.cache_salt for k in action.keys
                        ),
                    },
                )
            )
