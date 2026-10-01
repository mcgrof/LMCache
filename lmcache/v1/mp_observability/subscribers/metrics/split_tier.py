# SPDX-License-Identifier: Apache-2.0

"""Split-tier (KV_SPLIT_TIER / V-only) metrics subscriber.

Event-driven OTel counters for the split-tier store lifecycle.  These
complement the pull-based ``lmcache_mp.split_tier_manifest_entries`` gauge
(registered by the StorageManager over
:meth:`SplitTierManifest.state_counts`), which reports the live
state-machine distribution.

Counters:

- ``lmcache_mp.split_tier_store_completed`` -- logical keys whose V child
  reached L2 and whose composite is now a valid lookup hit.  Tagged by
  ``model_name``.
- ``lmcache_mp.split_tier_store_invalidated`` -- logical keys whose store
  failed and was rolled back (the key is storable again).  Tagged by
  ``reason`` (``pre_submit`` = serialize / temp-alloc failed before the V
  store was submitted; ``inner_store`` = the submitted V store failed) and
  ``model_name``.
- ``lmcache_mp.split_tier_v_child_deleted`` -- V children whose L2 delete
  was issued during paired cleanup.  Tagged by ``trigger`` (``eviction`` =
  a K-child eviction, ``clear`` = a cache clear sweep, ``store_failure`` =
  rollback of a failed store that had submitted its V child).  Best-effort:
  an adapter ``delete()`` that raises is logged but still counted (matching
  the L2-eviction counter), so this reflects deletes *issued*, not a
  media-durable purge.  (Removing the last split-tier adapter sweeps the
  manifest but issues no L2 delete -- the adapter is gone -- so it is not
  counted here.)
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class SplitTierMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for the split-tier store lifecycle."""

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache_mp.split_tier")
        self._completed_counter = meter.create_counter(
            "lmcache_mp.split_tier_store_completed",
            description=(
                "Count of split-tier logical keys whose V child reached L2 "
                "(composite is now a valid lookup hit). Tagged by "
                "``model_name``."
            ),
            unit="chunks",
        )
        self._invalidated_counter = meter.create_counter(
            "lmcache_mp.split_tier_store_invalidated",
            description=(
                "Count of split-tier logical keys whose store failed and "
                "was rolled back. Tagged by ``reason`` = pre_submit | "
                "inner_store and ``model_name``."
            ),
            unit="chunks",
        )
        self._v_deleted_counter = meter.create_counter(
            "lmcache_mp.split_tier_v_child_deleted",
            description=(
                "Count of split-tier V children whose L2 delete was issued "
                "during paired cleanup (best-effort). Tagged by ``trigger`` "
                "= eviction | clear | store_failure."
            ),
            unit="chunks",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.SPLIT_TIER_STORE_COMPLETED: self._on_store_completed,
            EventType.SPLIT_TIER_STORE_INVALIDATED: self._on_store_invalidated,
            EventType.SPLIT_TIER_V_CHILD_DELETED: self._on_v_child_deleted,
        }

    def _on_store_completed(self, event: Event) -> None:
        model_names: Counter = event.metadata["model_names"]
        for model_name, count in model_names.items():
            self._completed_counter.add(count, {"model_name": model_name})

    def _on_store_invalidated(self, event: Event) -> None:
        reason: str = event.metadata["reason"]
        model_names: Counter = event.metadata["model_names"]
        for model_name, count in model_names.items():
            self._invalidated_counter.add(
                count, {"reason": reason, "model_name": model_name}
            )

    def _on_v_child_deleted(self, event: Event) -> None:
        trigger: str = event.metadata["trigger"]
        count: int = event.metadata["count"]
        self._v_deleted_counter.add(count, {"trigger": trigger})
