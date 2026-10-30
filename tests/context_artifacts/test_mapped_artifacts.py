# SPDX-License-Identifier: Apache-2.0
"""Published-page borrowing, integrity policy and lifetime contracts."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import hashlib
import os
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.context_artifacts import ArtifactReadPlan
from lmcache.v1.mapped_artifacts import ArtifactVerification, MappedArtifactStore

# Local
from .test_read_plans import Catalog


def published(
    tmp_path: Path,
    verification: ArtifactVerification = ArtifactVerification.ON_MAP,
    **kwargs: int,
) -> tuple[MappedArtifactStore, Path, bytes]:
    """Publish a range with unaligned boundaries using the public API."""
    store = MappedArtifactStore(Catalog(), verification=verification, **kwargs)
    data = b"01234567" * 1024
    path = tmp_path / "file"
    path.write_bytes(b"prefix!" + data + b"suffix")
    store.publish(
        "one", {"kind": "bytes"}, path, hashlib.sha256(data).hexdigest(), len(data), 7
    )
    return store, path, data


def test_existing_buffer_reused_and_tensor_lives_through_eviction_and_close(
    tmp_path: Path,
) -> None:
    store, _, data = published(tmp_path)
    with store.lease("one") as first:
        tensor = torch.frombuffer(first.byte_array, dtype=torch.uint8)
        with store.lease("one") as second:
            other = torch.frombuffer(second.byte_array, dtype=torch.uint8)
            assert tensor.data_ptr() == other.data_ptr()
        assert not store.evict("one")
        with pytest.raises(RuntimeError, match="active"):
            store.close()
    assert store.evict("one")
    del first, second, other
    store.close()
    assert bytes(tensor.numpy()) == data
    assert store.map_count == 1
    assert store.read_count == store.read_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        with store.lease("one"):
            pass


@pytest.mark.parametrize("policy", list(ArtifactVerification))
def test_explicit_verification_policy_and_short_file(
    tmp_path: Path, policy: ArtifactVerification
) -> None:
    store, path, data = published(tmp_path, policy)
    with store.lease("one") as obj:
        assert bytes(obj.byte_array) == data
    assert store.evict("one")
    # Between leases only: violate published immutability to distinguish the
    # strict checking mode from the explicitly trusted-publication mode.
    with path.open("r+b") as stream:
        stream.seek(7)
        stream.write(b"!")
    if policy is ArtifactVerification.ON_MAP:
        with pytest.raises(ValueError, match="checksum"):
            with store.lease("one"):
                pass
    else:
        with store.lease("one") as obj:
            assert bytes(obj.byte_array)[0:1] == b"!"
        assert store.checksum_bytes == 0
    store.evict("one")
    # Drop exported values before truncation; live mmap truncation is outside
    # the immutable-file contract and can cause SIGBUS.
    del obj
    path.write_bytes(b"short")
    with pytest.raises(EOFError, match="range"):
        with store.lease("one"):
            pass
    store.close()


def test_foreign_plans_catalog_change_and_failed_open_do_not_publish(
    tmp_path: Path,
) -> None:
    store, path, _ = published(tmp_path)
    plan = store.prepare("one")
    other = MappedArtifactStore(store.catalog)
    assert isinstance(store.catalog, Catalog)
    for candidate, supplied in [(store, replace(plan)), (other, plan)]:
        with pytest.raises(ValueError, match="not issued"):
            with candidate.lease("one", plan=supplied):
                pass
    store.catalog.records["one"]["offset"] = 0
    with pytest.raises(ValueError, match="catalog record"):
        with store.lease("one", plan=plan):
            pass
    store.catalog.records["one"]["offset"] = 7
    path.unlink()
    with pytest.raises(FileNotFoundError):
        with store.lease("one"):
            pass
    assert not store.resident and not store.active
    store.close()
    other.close()


@pytest.mark.parametrize("policy", list(ArtifactVerification))
def test_mapping_miss_catalog_recheck_follows_policy(
    tmp_path: Path, policy: ArtifactVerification
) -> None:
    """Strict maps recheck the catalog; published maps trust the current plan."""
    store, _, data = published(tmp_path, policy, max_read_plans=1)
    assert isinstance(store.catalog, Catalog)
    plan = store.prepare("one")
    for _ in range(3):
        before = store.catalog.lookups
        with store.lease("one", plan=plan) as obj:
            assert bytes(obj.byte_array) == data
        assert store.evict("one")
        expected = 1 if policy is ArtifactVerification.ON_MAP else 0
        assert store.catalog.lookups - before == expected
    store.catalog.records["one"]["offset"] = 0
    if policy is ArtifactVerification.ON_MAP:
        with pytest.raises(ValueError, match="catalog record"):
            with store.lease("one", plan=plan):
                pass
    else:
        # A current plan is trusted like published bytes.
        with store.lease("one", plan=plan) as obj:
            assert bytes(obj.byte_array) == data
        assert store.evict("one")
    store.catalog.records["one"]["offset"] = 7
    store.publish("two", {"kind": "other"}, plan.path, plan.sha256, plan.length, 7)
    store.prepare("two")  # The retained plan for "one" is no longer current.
    store.catalog.records["one"]["offset"] = 0
    with pytest.raises(ValueError, match="catalog record"):
        with store.lease("one", plan=plan):
            pass
    assert not store.resident.get("one")
    store.close()


def test_concurrent_borrows_coalesce_and_descriptor_count_is_stable(
    tmp_path: Path,
) -> None:
    store, _, data = published(tmp_path, max_mappings=1)
    before = len(os.listdir("/proc/self/fd"))

    def consume(_: int) -> bytes:
        with store.lease("one") as obj:
            return bytes(obj.byte_array)

    with ThreadPoolExecutor(8) as pool:
        assert all(row == data for row in pool.map(consume, range(1024)))
    assert store.map_count == 1
    assert len(os.listdir("/proc/self/fd")) <= before + 1
    assert not any(store.active.values())
    store.close()
    assert len(os.listdir("/proc/self/fd")) == before


def test_capacity_bound_and_wait_timeout(tmp_path: Path) -> None:
    store, path, data = published(tmp_path, max_mappings=1)
    store.publish(
        "two", {"kind": "other"}, path, hashlib.sha256(data).hexdigest(), len(data), 7
    )
    with store.lease("one"):
        with pytest.raises(MemoryError, match="capacity"):
            with store.lease("two"):
                pass
        with pytest.raises(TimeoutError, match="capacity"):
            with store.lease("two", timeout=0.001):
                pass
    with store.lease("two") as obj:
        assert bytes(obj.byte_array) == data
    assert len(store.resident) == 1
    store.close()


@pytest.mark.parametrize("policy", list(ArtifactVerification))
def test_waiting_readers_recheck_their_own_catalog_snapshot(
    tmp_path: Path, policy: ArtifactVerification
) -> None:
    """Concurrent waiters with different admitted bindings must not share bytes."""
    store, path, data = published(
        tmp_path, policy, max_mappings=1, max_read_plans=1
    )
    assert isinstance(store.catalog, Catalog)
    store.publish(
        "two", {"version": "old"}, path, hashlib.sha256(data).hexdigest(), len(data), 7
    )
    old = store.prepare("two")
    store.prepare("one")  # Evict metadata only; the old caller retains its plan.
    store.catalog.records["two"]["manifest"] = {"version": "new"}
    new = store.prepare("two")

    def consume(plan: ArtifactReadPlan) -> bytes:
        with store.lease("two", plan=plan, timeout=2) as obj:
            return bytes(obj.byte_array)

    def wait_for(count: int) -> None:
        limit = time.monotonic() + 2
        while store.capacity_waits < count:
            if time.monotonic() > limit:
                raise TimeoutError("test waiters did not enter capacity wait")
            time.sleep(0.001)

    with ThreadPoolExecutor(2) as pool:
        for _ in range(16):
            store.evict("two")
            before = store.capacity_waits
            with store.lease("one"):
                fresh = pool.submit(consume, new)
                wait_for(before + 1)
                stale = pool.submit(consume, old)
                wait_for(before + 2)
            assert fresh.result() == data
            with pytest.raises(ValueError, match="binding|catalog record"):
                stale.result()
    store.close()
