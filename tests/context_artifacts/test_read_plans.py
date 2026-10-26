# SPDX-License-Identifier: Apache-2.0
"""Public prepared-read, integrity, ownership and capacity-wait contracts."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.context_artifacts import ArtifactStore
from lmcache.v1.memory_allocators.host_memory_allocator import HostMemoryAllocator


class Catalog:
    def __init__(self):
        self.records = {}
        self.lookups = 0

    def register_artifact(self, key, record):
        if key in self.records and self.records[key] != record:
            raise ValueError("conflicting identity")
        self.records[key] = deepcopy(record)

    def lookup_artifact(self, key):
        self.lookups += 1
        return deepcopy(self.records.get(key))


@pytest.fixture
def banks(tmp_path):
    catalog = Catalog()
    allocator = HostMemoryAllocator(65536)
    store = ArtifactStore(catalog, allocator, max_read_plans=2)
    payloads = {}
    for index in range(4):
        key = f"bank-{index}"
        data = bytes([index + 1]) * 40000
        path = tmp_path / key
        path.write_bytes(b"prefix" + data + b"suffix")
        store.publish(
            key, {"bank": index}, path, hashlib.sha256(data).hexdigest(), len(data), 6
        )
        payloads[key] = data
    yield store, catalog, allocator, payloads
    store.close()
    assert allocator.memcheck()


def read(store, key, **options):
    with store.lease(key, **options) as obj:
        return bytes(obj.byte_array)


def test_cached_metadata_keeps_reload_integrity(banks):
    store, catalog, _, payloads = banks
    plan = store.prepare("bank-0")
    assert read(store, "bank-0", plan=plan) == payloads["bank-0"]
    lookups = catalog.lookups
    for _ in range(20):
        assert read(store, "bank-0", plan=plan) == payloads["bank-0"]
    assert catalog.lookups == lookups
    assert store.read_count == 1
    assert store.evict("bank-0")
    assert read(store, "bank-0", plan=plan) == payloads["bank-0"]
    assert catalog.lookups == lookups + 1
    assert store.evict("bank-0")
    with open(plan.path, "r+b") as stream:
        stream.seek(plan.offset)
        stream.write(b"!")
    with pytest.raises(ValueError, match="checksum"):
        read(store, "bank-0", plan=plan)
    assert store.read_count == 2


def test_changed_catalog_and_forged_or_foreign_plans_rejected(banks):
    store, catalog, allocator, _ = banks
    plan = store.prepare("bank-0")
    with pytest.raises(ValueError, match="not issued"):
        read(store, "bank-0", plan=replace(plan, offset=0))
    other = ArtifactStore(catalog, allocator)
    try:
        with pytest.raises(ValueError, match="not issued"):
            read(other, "bank-0", plan=plan)
    finally:
        other.close()
    catalog.records["bank-0"]["offset"] = 0
    with pytest.raises(ValueError, match="catalog record"):
        read(store, "bank-0", plan=plan)
    assert store.read_count == 0


def test_plan_cache_eviction_keeps_issued_handles_valid(banks):
    store, _, _, payloads = banks
    first = store.prepare("bank-0")
    store.prepare("bank-1")
    store.prepare("bank-2")
    assert store.prepare("bank-0") is not first
    assert read(store, "bank-0", plan=first) == payloads["bank-0"]


def test_capacity_wait_reuses_admitted_plan_and_releases_leases(banks):
    store, catalog, _, payloads = banks
    plan = store.prepare("bank-1")
    with ThreadPoolExecutor(max_workers=1) as executor:
        with store.lease("bank-0"):
            before = catalog.lookups
            future = executor.submit(read, store, "bank-1", plan=plan, timeout=2)
            deadline = time.monotonic() + 2
            while not store.capacity_waits:
                assert time.monotonic() < deadline
                threading.Event().wait(0.001)
            assert not future.done()
            assert catalog.lookups == before + 1
            assert not store.evict("bank-0")
            with pytest.raises(RuntimeError, match="active"):
                store.close()
        assert future.result(timeout=2) == payloads["bank-1"]
    assert catalog.lookups == before + 1
    assert store.capacity_wait_seconds > 0


def test_timeout_default_failure_and_recovery(banks):
    store, _, _, payloads = banks
    with store.lease("bank-0"):
        with pytest.raises(MemoryError):
            read(store, "bank-1")
        with pytest.raises(TimeoutError, match="capacity"):
            read(store, "bank-1", timeout=0.02)
    assert read(store, "bank-1", timeout=1) == payloads["bank-1"]


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_invalid_wait_deadlines(banks, timeout):
    with pytest.raises(ValueError, match="timeout"):
        read(banks[0], "bank-0", timeout=timeout)


def test_failed_inflight_read_wakes_capacity_waiter(banks, monkeypatch):
    # First Party
    import lmcache.v1.context_artifacts as implementation

    store, _, _, payloads = banks
    original = implementation.read_file_range_into
    entered, release = threading.Event(), threading.Event()

    def fail_first(path, *args):
        if path.endswith("bank-0"):
            entered.set()
            assert release.wait(2)
            raise OSError("injected read failure")
        return original(path, *args)

    monkeypatch.setattr(implementation, "read_file_range_into", fail_first)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(read, store, "bank-0")
        assert entered.wait(2)
        second = executor.submit(read, store, "bank-1", timeout=2)
        deadline = time.monotonic() + 2
        while not store.capacity_waits:
            assert time.monotonic() < deadline
            threading.Event().wait(0.001)
        release.set()
        with pytest.raises(OSError, match="injected"):
            first.result(timeout=2)
        assert second.result(timeout=2) == payloads["bank-1"]
    assert store.read_count == 1


def test_coalesced_wait_timeout_does_not_cancel_owner(banks, monkeypatch):
    # First Party
    import lmcache.v1.context_artifacts as implementation

    store, _, _, payloads = banks
    original = implementation.read_file_range_into
    entered, release = threading.Event(), threading.Event()

    def delayed(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)

    monkeypatch.setattr(implementation, "read_file_range_into", delayed)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(read, store, "bank-0")
        assert entered.wait(2)
        try:
            with pytest.raises(TimeoutError, match="coalescing"):
                read(store, "bank-0", timeout=0.02)
        finally:
            release.set()
        assert first.result(timeout=2) == payloads["bank-0"]
    assert read(store, "bank-0") == payloads["bank-0"]
    assert store.read_count == 1
