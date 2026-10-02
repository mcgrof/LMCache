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
from lmcache.v1.distributed.error import L1Error
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
    ComponentKeyScheme,
    SplitTierManifest,
    SplitTierState,
    derive_component_key,
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
        self._last_extra_log = time.monotonic()
        # Paired-eviction wiring.  ``None`` for both
        # preserves the legacy DISCARD-only behavior; callers that wire
        # KV_SPLIT_TIER pass both so the controller can drive the
        # INVALIDATED -> DELETE_IN_FLIGHT lifecycle and enqueue V child
        # deletes against the L2 adapters.
        self._split_tier_manifest = split_tier_manifest
        self._l2_adapters = list(l2_adapters or [])
        self._paired_eviction_condition = threading.Condition()
        self._active_paired_adapter_passes = 0
        # Per-engine child-key scheme; paired eviction derives the V-child
        # delete key under it so a RAW_UNIT K victim reclaims the RAW_UNIT
        # V child (not a legacy one).  Set via set_split_tier_paired_eviction.
        self._component_key_scheme = ComponentKeyScheme.COMPUTED_LEGACY

    def set_split_tier_paired_eviction(
        self,
        manifest: "SplitTierManifest",
        l2_adapters: "list[L2AdapterInterface]",
        component_key_scheme: ComponentKeyScheme = ComponentKeyScheme.COMPUTED_LEGACY,
        timeout: float | None = None,
    ) -> bool:
        """Wire paired-eviction state post-construction.

        The L1EvictionController is constructed before the L2 adapters
        in the StorageManager init order (the L1 manager wants its
        listener registered before any allocate event fires).  This
        setter lets the StorageManager attach the manifest + L2
        adapters once they exist.  Calling more than once replaces the
        previous wiring; passing empty adapters reverts to legacy
        DISCARD-only behavior.

        ``component_key_scheme`` is the per-engine child-key scheme; the
        paired V-child delete is derived under it so a RAW_UNIT (byte-
        through) K victim reclaims the RAW_UNIT V child rather than a
        legacy one.

        Args:
            manifest: Manifest shared with the split-tier wrapper.
            l2_adapters: Replacement adapter snapshot for future passes.
            component_key_scheme: Child-key namespace used by the engine.
            timeout: Maximum seconds to wait for already-started paired
                passes. ``None`` waits without a deadline.

        Returns:
            ``True`` after prior passes drain. ``False`` on timeout; the
            previous wiring is restored in that case.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._paired_eviction_condition:
            previous_manifest = self._split_tier_manifest
            previous_adapters = self._l2_adapters
            previous_scheme = self._component_key_scheme
            self._split_tier_manifest = manifest
            # Detach first: a new pass snapshots only the replacement list.
            self._l2_adapters = list(l2_adapters)
            self._component_key_scheme = component_key_scheme
            while self._active_paired_adapter_passes:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if remaining == 0.0:
                    self._split_tier_manifest = previous_manifest
                    self._l2_adapters = previous_adapters
                    self._component_key_scheme = previous_scheme
                    return False
                self._paired_eviction_condition.wait(timeout=remaining)
            return True

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

    def _maybe_log_memory_usage(self, used_bytes: int, total_bytes: int) -> None:
        """Emit the opt-in L1 memory usage INFO line, throttled to the interval."""
        now = time.monotonic()
        if now - self._last_extra_log < self._eviction_config.extra_logging_interval:
            return
        self._last_extra_log = now
        pct = 0.0 if total_bytes == 0 else used_bytes / total_bytes * 100.0
        logger.info(
            "L1 memory usage: %.2f/%.2f GiB (%.1f%%)",
            used_bytes / (1 << 30),
            total_bytes / (1 << 30),
            pct,
        )

    def eviction_loop(self):
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio

        while not self._stop_flag.is_set():
            time.sleep(1)
            used_bytes, total_bytes = self._l1_manager.get_memory_usage()
            if self._eviction_config.extra_logging_enabled:
                self._maybe_log_memory_usage(used_bytes, total_bytes)
            usage = self._l1_manager.get_memory_pressure()
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

    def _evict_with_split_tier_pairing(self, keys: "list[ObjectKey]") -> None:
        """Discard keys using one stable paired-adapter snapshot.

        Runtime adapter removal first detaches the adapter from future
        snapshots, then waits for every snapshot already in use.  Acquiring
        the pass reference before any manifest or L1 mutation guarantees an
        adapter cannot close between claiming cleanup and issuing its paired
        V-child delete.
        """
        with self._paired_eviction_condition:
            manifest = self._split_tier_manifest
            adapters = list(self._l2_adapters)
            component_key_scheme = self._component_key_scheme
            if manifest is not None and adapters:
                self._active_paired_adapter_passes += 1

        if manifest is None or not adapters:
            self._l1_manager.delete(keys)
            return

        try:
            self._evict_paired_snapshot(
                keys,
                manifest,
                adapters,
                component_key_scheme,
            )
        finally:
            with self._paired_eviction_condition:
                self._active_paired_adapter_passes -= 1
                self._paired_eviction_condition.notify_all()

    def _evict_paired_snapshot(
        self,
        keys: "list[ObjectKey]",
        manifest: "SplitTierManifest",
        adapters: "list[L2AdapterInterface]",
        component_key_scheme: ComponentKeyScheme,
    ) -> None:
        """Discard L1 keys and their paired V children.

        The manifest first enters ``DELETE_IN_FLIGHT``.  Child keys do not
        carry generations, so this fence prevents a replacement store from
        publishing under the same physical K/V names while either delete is
        delayed.  If the K child became read-locked after policy selection,
        the fence is cancelled and the intact composite is retried later.

        Per successfully-deleted K child:

          1. Resolve the logical key through manifest child membership.
          2. Atomically fence the current generation as DELETE_IN_FLIGHT.
          3. Delete the K child from L1.
          4. Delete the V child through every snapshotted L2 adapter.
             Split-tier admission restricts those adapters to filesystem
             storage, whose ``delete()`` return is a terminal unlink
             boundary.
          5. Drop the entry only after the physical deletes return, opening
             the generation-less child names to a replacement store.

        K children whose manifest state is ``STORE_IN_FLIGHT`` are
        excluded from the delete batch entirely: they are pinned by
        ``is_key_evictable`` by design, so seeing one here means the
        policy snapshot went stale -- deleting it would tear the K out
        from under an active V codec / L2 write.
        """
        # Capture the manifest generation of each K child at the moment
        # we decide to delete it.  Only one generation's K child can be
        # resident in L1 under a given key at a time (a re-store collides
        # on the K-child allocation while the old child is present), so
        # the generation observed here is the one whose K child the
        # delete below removes.  The teardown then acts only on THAT
        # generation: if a fresh store reclaims the logical key after the
        # delete opens it, the captured generation no longer matches and
        # the teardown leaves the new composite untouched.
        delete_batch: list[ObjectKey] = []
        cleanup_claims: dict[ObjectKey, tuple[ObjectKey, int, SplitTierState]] = {}
        for k in keys:
            logical_key = manifest.logical_for_k_child(k)
            if logical_key is not None:
                entry = manifest.lookup_entry(logical_key)
                if entry is not None:
                    entry_state, entry_generation = entry
                    if entry_state in (
                        SplitTierState.STORE_IN_FLIGHT,
                        SplitTierState.DELETE_IN_FLIGHT,
                    ):
                        logger.warning(
                            "L1EvictionController paired-evict: skipping "
                            "K child of logical %s -- lifecycle operation "
                            "still in flight (stale eviction snapshot)",
                            logical_key,
                        )
                        continue
                    previous = manifest.begin_physical_cleanup(
                        logical_key,
                        entry_generation,
                        (SplitTierState.COMPLETE, SplitTierState.INVALIDATED),
                    )
                    if previous is None:
                        continue
                    cleanup_claims[k] = (
                        logical_key,
                        entry_generation,
                        previous,
                    )
            delete_batch.append(k)

        results = self._l1_manager.delete(delete_batch)

        v_children_deleted = 0
        for k in delete_batch:
            outcome = results.get(k)
            claim = cleanup_claims.get(k)
            if claim is None:
                # No manifest entry was associated with this K-shaped key.
                continue
            logical_key, generation, previous_state = claim
            if outcome not in (L1Error.SUCCESS, L1Error.KEY_NOT_EXIST):
                # A reader won the race for the K child; the composite
                # stays intact and a later eviction cycle retries.
                manifest.cancel_physical_cleanup(
                    logical_key, generation, previous_state
                )
                logger.info(
                    "L1EvictionController paired-evict: K child of "
                    "logical %s was not deleted (%s); preserving the "
                    "composite for retry",
                    logical_key,
                    outcome,
                )
                continue
            # DELETE_IN_FLIGHT is held across the delete, so the
            # generation-agnostic V key still names THIS generation's V
            # child.  Delete it against every L2 adapter, then drop.
            v_child_key = derive_component_key(
                logical_key, "v", scheme=component_key_scheme
            )
            for adapter in adapters:
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
            v_children_deleted += 1
            manifest.drop(logical_key, generation)

        if v_children_deleted:
            self._event_bus.publish(
                Event(
                    event_type=EventType.SPLIT_TIER_V_CHILD_DELETED,
                    metadata={
                        "count": v_children_deleted,
                        "trigger": "eviction",
                    },
                )
            )


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
