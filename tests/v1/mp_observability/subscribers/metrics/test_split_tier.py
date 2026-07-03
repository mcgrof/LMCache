# SPDX-License-Identifier: Apache-2.0

"""Tests for SplitTierMetricsSubscriber.

Verifies that the split-tier lifecycle events
(``SPLIT_TIER_STORE_COMPLETED``, ``SPLIT_TIER_STORE_INVALIDATED``,
``SPLIT_TIER_V_CHILD_DELETED``) produce the expected OTel counters with
correct ``model_name`` / ``reason`` / ``trigger`` attributes. Uses the
shared ``InMemoryMetricReader`` to assert on counter deltas keyed by
(metric_name, attributes).
"""

# Standard
from collections import Counter
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.split_tier import (
    SplitTierMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.counter_helpers import (
    counter_delta,
    counter_value,
    read_tagged_counters,
)

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = SplitTierMetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    before = read_tagged_counters()

    def get_delta():
        return counter_delta(before, read_tagged_counters())

    return get_delta


class TestStoreCompleted:
    def test_single_model(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.SPLIT_TIER_STORE_COMPLETED,
                metadata={"count": 3, "model_names": Counter({"llama-7b": 3})},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.split_tier_store_completed",
                model_name="llama-7b",
            )
            == 3
        )

    def test_multi_model_buckets_separately(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.SPLIT_TIER_STORE_COMPLETED,
                metadata={
                    "count": 5,
                    "model_names": Counter({"llama-7b": 2, "mistral-7b": 3}),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.split_tier_store_completed",
                model_name="llama-7b",
            )
            == 2
        )
        assert (
            counter_value(
                delta,
                "lmcache_mp.split_tier_store_completed",
                model_name="mistral-7b",
            )
            == 3
        )

    def test_accumulates_across_events(self, bus, subscriber, snapshot):
        bus.start()
        for _ in range(4):
            bus.publish(
                Event(
                    event_type=EventType.SPLIT_TIER_STORE_COMPLETED,
                    metadata={"count": 1, "model_names": Counter({"llama-7b": 1})},
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.split_tier_store_completed",
                model_name="llama-7b",
            )
            == 4
        )


class TestStoreInvalidated:
    def test_pre_submit_reason(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.SPLIT_TIER_STORE_INVALIDATED,
                metadata={
                    "count": 2,
                    "reason": "pre_submit",
                    "model_names": Counter({"llama-7b": 2}),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.split_tier_store_invalidated",
                reason="pre_submit",
                model_name="llama-7b",
            )
            == 2
        )

    def test_inner_store_reason(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.SPLIT_TIER_STORE_INVALIDATED,
                metadata={
                    "count": 1,
                    "reason": "inner_store",
                    "model_names": Counter({"mistral-7b": 1}),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.split_tier_store_invalidated",
                reason="inner_store",
                model_name="mistral-7b",
            )
            == 1
        )

    def test_reasons_counted_separately(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.SPLIT_TIER_STORE_INVALIDATED,
                metadata={
                    "count": 1,
                    "reason": "pre_submit",
                    "model_names": Counter({"llama-7b": 1}),
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.SPLIT_TIER_STORE_INVALIDATED,
                metadata={
                    "count": 2,
                    "reason": "inner_store",
                    "model_names": Counter({"llama-7b": 2}),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.split_tier_store_invalidated",
                reason="pre_submit",
                model_name="llama-7b",
            )
            == 1
        )
        assert (
            counter_value(
                delta,
                "lmcache_mp.split_tier_store_invalidated",
                reason="inner_store",
                model_name="llama-7b",
            )
            == 2
        )


class TestVChildDeleted:
    @pytest.mark.parametrize("trigger", ["eviction", "clear", "store_failure"])
    def test_trigger_tagged(self, bus, subscriber, snapshot, trigger):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.SPLIT_TIER_V_CHILD_DELETED,
                metadata={"count": 7, "trigger": trigger},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.split_tier_v_child_deleted",
                trigger=trigger,
            )
            == 7
        )

    def test_triggers_counted_separately(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.SPLIT_TIER_V_CHILD_DELETED,
                metadata={"count": 3, "trigger": "eviction"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.SPLIT_TIER_V_CHILD_DELETED,
                metadata={"count": 5, "trigger": "clear"},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.split_tier_v_child_deleted",
                trigger="eviction",
            )
            == 3
        )
        assert (
            counter_value(
                delta,
                "lmcache_mp.split_tier_v_child_deleted",
                trigger="clear",
            )
            == 5
        )


class TestSplitTierSubscriptions:
    def test_subscriptions_cover_three_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.SPLIT_TIER_STORE_COMPLETED in subs
        assert EventType.SPLIT_TIER_STORE_INVALIDATED in subs
        assert EventType.SPLIT_TIER_V_CHILD_DELETED in subs
        assert len(subs) == 3

    def test_no_subscription_for_unrelated_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L2_STORE_COMPLETED not in subs
        assert EventType.L1_WRITE_FINISHED not in subs
