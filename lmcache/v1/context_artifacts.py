# SPDX-License-Identifier: Apache-2.0
"""Typed external artifacts backed by a catalog and LMCache memory objects."""

# Standard
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol
import hashlib
import json
import math
import os
import threading
import time
import weakref

# Third Party
import torch

# First Party
from lmcache.v1.buffered_file import read_file_range_into
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError


class ArtifactCatalog(Protocol):
    """Metadata operations implemented by the cartridge registry."""

    def register_artifact(self, artifact_id: str, record: dict) -> None:
        """Publish immutable metadata after storage completion."""
        ...

    def lookup_artifact(self, artifact_id: str) -> dict | None:
        """Return an exact artifact record, or None."""
        ...


@dataclass(frozen=True)
class ArtifactReadPlan:
    """Immutable metadata admitted by one ArtifactStore; never payload data.

    Plans are issued by ``prepare`` and cannot be transferred to another
    store. A new load checks the catalog; its backing selects payload checking.
    Keeping a plan does not reserve allocator capacity or a tensor lease.
    """

    artifact_id: str
    path: str
    offset: int
    length: int
    sha256: str
    manifest_json: str
    record_json: str


class ArtifactMetadataStore:
    """Share publication and immutable read-plan admission across backings."""

    def __init__(self, catalog: ArtifactCatalog, max_read_plans: int = 1024) -> None:
        """Create bounded metadata caches without allocating payload memory.

        Args:
            catalog: Caller-owned immutable location catalog.
            max_read_plans: Positive maximum cached metadata entries.

        Raises:
            ValueError: The entry limit is not a positive integer.
        """
        if type(max_read_plans) is not int or max_read_plans <= 0:
            raise ValueError("max_read_plans must be positive")
        self.catalog = catalog
        self.max_read_plans = max_read_plans
        self._plans: OrderedDict[str, ArtifactReadPlan] = OrderedDict()
        self._issued: weakref.WeakValueDictionary[int, ArtifactReadPlan] = (
            weakref.WeakValueDictionary()
        )
        self.records: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.closed = False

    def publish(
        self,
        artifact_id: str,
        manifest: dict,
        path: str | Path,
        digest: str,
        length: int,
        offset: int = 0,
    ) -> None:
        """Verify an existing immutable file range and register its location.

        Args:
            artifact_id: Exact external identity, including semantic metadata.
            manifest: Typed consumer metadata validated by its producer.
            path: Existing durable regular file; caller retains its lifetime.
            digest: SHA-256 of the complete range.
            length: Positive byte count, independent of the containing file.
            offset: Nonnegative beginning of the range.

        Raises:
            ValueError: Invalid range, digest, or conflicting publication.
            EOFError: The source is short.
        """
        if length <= 0 or offset < 0 or len(digest) != 64:
            raise ValueError("invalid artifact range or digest")
        resolved = Path(path).resolve(strict=True)
        checksum = hashlib.sha256()
        with resolved.open("rb", buffering=0) as stream:
            stream.seek(offset)
            left = length
            while left:
                chunk = stream.read(min(left, 1 << 20))
                if not chunk:
                    raise EOFError("short artifact publication")
                checksum.update(chunk)
                left -= len(chunk)
            os.fsync(stream.fileno())
        directory = os.open(resolved.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if checksum.hexdigest() != digest:
            raise ValueError("artifact publication checksum mismatch")
        self.catalog.register_artifact(
            artifact_id,
            {
                "manifest": manifest,
                "path": str(resolved),
                "offset": offset,
                "length": length,
                "sha256": digest,
            },
        )

    def describe(self, artifact_id: str) -> dict[str, Any]:
        """Resolve metadata without loading payload bytes.

        Raises:
            KeyError: The requested exact identity is absent.
        """
        with self.lock:
            if artifact_id in self.records:
                return deepcopy(self.records[artifact_id])
        record = self.catalog.lookup_artifact(artifact_id)
        if record is None:
            raise KeyError(artifact_id)
        return record

    def prepare(self, artifact_id: str) -> ArtifactReadPlan:
        """Admit an immutable catalog snapshot without loading its payload.

        The entry cache is bounded by ``max_read_plans``. A caller may retain
        an issued plan beyond cache eviction. Catalog changes are rejected
        on every subsequent load; resident hits use their admitted binding.
        The backing's verification policy controls full-payload checks.

        Args:
            artifact_id: Exact published identity to admit.

        Returns:
            Immutable plan issued by this store, with no payload allocation.

        Raises:
            KeyError: The identity is absent.
            ValueError: Its range or metadata is invalid.
            RuntimeError: The store is closed.
        """
        with self.lock:
            if self.closed:
                raise RuntimeError("artifact store is closed")
            plan = self._plans.get(artifact_id)
            if plan is not None:
                self._plans.move_to_end(artifact_id)
                return plan
            record = self.describe(artifact_id)
            if (
                not isinstance(record.get("manifest"), dict)
                or not isinstance(record.get("path"), str)
                or not Path(record["path"]).is_absolute()
                or type(record.get("length")) is not int
                or record["length"] <= 0
                or type(record.get("offset")) is not int
                or record["offset"] < 0
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in record["sha256"])
            ):
                raise ValueError("invalid artifact read plan")
            plan = ArtifactReadPlan(
                artifact_id,
                record["path"],
                record["offset"],
                record["length"],
                record["sha256"],
                json.dumps(record["manifest"], sort_keys=True, separators=(",", ":")),
                json.dumps(record, sort_keys=True, separators=(",", ":")),
            )
            self._plans[artifact_id] = plan
            self._issued[id(plan)] = plan
            if len(self._plans) > self.max_read_plans:
                self._plans.popitem(last=False)
            return plan


class ArtifactStore(ArtifactMetadataStore):
    """Read immutable file ranges into bounded LMCache allocator memory.

    The catalog owns identity and location. This store owns process-local
    residency. A lease keeps its MemoryObj alive until the caller finishes
    consuming every view, including any asynchronous device transfer.
    This is an external-artifact adapter, not a prompt-token KV cache.
    """

    def __init__(
        self,
        catalog: ArtifactCatalog,
        allocator: MemoryAllocatorInterface,
        max_read_plans: int = 1024,
    ) -> None:
        """Create a process-local store using caller-owned catalog/allocator.

        Args:
            catalog: Immutable identity/location catalog.
            allocator: Memory allocator shared under the caller's ownership.
            max_read_plans: Positive LRU entry budget, independent of payload
                capacity. Active leases and caller-retained plans stay alive
                outside this cache; the limit is not a total memory bound.
        """
        super().__init__(catalog, max_read_plans)
        self.allocator = allocator
        self._resident_plans: dict[str, ArtifactReadPlan] = {}
        self._loading: set[str] = set()
        self.resident: OrderedDict[str, MemoryObj] = OrderedDict()
        self.active: dict[str, int] = {}
        self.waiters: dict[str, int] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.capacity = threading.Condition(self.lock)
        self.read_count = 0
        self.read_bytes = 0
        self.cache_hits = 0
        self.capacity_waits = 0
        self.capacity_wait_seconds = 0.0

    @contextmanager
    def lease(
        self,
        artifact_id: str,
        *,
        plan: ArtifactReadPlan | None = None,
        timeout: float = 0.0,
    ) -> Iterator[MemoryObj]:
        """Acquire one loaded object and release the caller reference on exit.

        Args:
            artifact_id: Exact published identity.
            plan: Optional immutable plan issued by this store.
            timeout: Maximum coalescing/capacity wait in seconds. Zero keeps
                immediate capacity failure and ordinary miss coalescing.
                File/catalog I/O already running is not interrupted.

        Yields:
            LMCache MemoryObj containing the verified file range as bytes.

        Raises:
            MemoryError: The allocator is full and no idle object is evictable.
            ValueError: Bytes no longer match their immutable digest.
            RuntimeError: The store has closed.
            TimeoutError: A bounded coalescing or capacity wait expired.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout if timeout else None
        plan = self.prepare(artifact_id) if plan is None else plan
        with self.lock:
            if self.closed:
                raise RuntimeError("artifact store is closed")
            if (
                self._issued.get(id(plan)) is not plan
                or plan.artifact_id != artifact_id
            ):
                raise ValueError("read plan was not issued for this store and identity")
            key_lock = self.locks.setdefault(artifact_id, threading.Lock())
            self.waiters[artifact_id] = self.waiters.get(artifact_id, 0) + 1
        try:
            if deadline is None:
                key_lock.acquire()
            elif not key_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
                raise LMCacheTimeoutError("artifact coalescing wait expired")
            try:
                obj = self._acquire(artifact_id, plan, deadline)
            finally:
                key_lock.release()
        finally:
            with self.lock:
                self.waiters[artifact_id] -= 1
                if not self.waiters[artifact_id]:
                    del self.waiters[artifact_id]
                    del self.locks[artifact_id]
        try:
            yield obj
        finally:
            with self.lock:
                self.active[artifact_id] -= 1
                obj.ref_count_down()
                self.capacity.notify_all()

    def evict(self, artifact_id: str) -> bool:
        """Release an idle resident object; active leases prevent eviction."""
        with self.lock:
            if self.active.get(artifact_id, 0):
                return False
            obj = self.resident.pop(artifact_id, None)
            if obj is None:
                return False
            self.active.pop(artifact_id, None)
            self.records.pop(artifact_id, None)
            self._resident_plans.pop(artifact_id, None)
            obj.ref_count_down()
            self.capacity.notify_all()
            return True

    def close(self) -> None:
        """Release resident objects after all leases finish.

        Raises:
            RuntimeError: A caller still holds a lease.
        """
        with self.lock:
            if any(self.active.values()):
                raise RuntimeError("active artifact leases remain")
            # A pending read owns an allocator object outside this lock.
            if self.waiters:
                raise RuntimeError("artifact load in progress")
            self.closed = True
            for key in list(self.resident):
                self.evict(key)
            self._plans.clear()
            self._issued.clear()

    def _acquire(
        self,
        artifact_id: str,
        plan: ArtifactReadPlan,
        deadline: float | None,
    ) -> MemoryObj:
        with self.lock:
            if self.closed:
                raise RuntimeError("artifact store is closed")
            obj = self.resident.get(artifact_id)
            if obj is not None:
                previous = self._resident_plans[artifact_id]
                if previous is not plan and previous.record_json != plan.record_json:
                    raise ValueError("resident artifact binding differs from read plan")
                self.resident.move_to_end(artifact_id)
                self.cache_hits += 1
                self.active[artifact_id] += 1
                obj.ref_count_up()
                return obj
        # Preserve a real catalog read on misses. Metadata caches must not
        # make a changed durable location or identity invisible on reload.
        record = self.catalog.lookup_artifact(artifact_id)
        if record is None:
            raise KeyError(artifact_id)
        if (
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            != plan.record_json
        ):
            raise ValueError("catalog record differs from admitted read plan")
        with self.capacity:
            obj = self.allocator.allocate(
                torch.Size([plan.length]), torch.uint8, MemoryFormat.BINARY
            )
            while obj is None:
                victim = next(
                    (key for key in self.resident if not self.active.get(key, 0)), None
                )
                if victim is None:
                    if deadline is None or (
                        not any(self.active.values()) and not self._loading
                    ):
                        raise MemoryError("artifact pool has no evictable capacity")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LMCacheTimeoutError("artifact capacity wait expired")
                    self.capacity_waits += 1
                    tick = time.monotonic()
                    self.capacity.wait(remaining)
                    self.capacity_wait_seconds += time.monotonic() - tick
                else:
                    self.evict(victim)
                obj = self.allocator.allocate(
                    torch.Size([plan.length]), torch.uint8, MemoryFormat.BINARY
                )
            self._loading.add(artifact_id)
        try:
            buffer = memoryview(obj.byte_array).cast("B")
            count = read_file_range_into(plan.path, buffer, plan.offset)
            if hashlib.sha256(buffer).hexdigest() != plan.sha256:
                raise ValueError("loaded artifact checksum mismatch")
        except BaseException:
            with self.lock:
                obj.ref_count_down()
                self._loading.remove(artifact_id)
                self.capacity.notify_all()
            raise
        with self.lock:
            self.read_count += 1
            self.read_bytes += count
            self.resident[artifact_id] = obj
            self.records[artifact_id] = deepcopy(record)
            self._resident_plans[artifact_id] = plan
            self._loading.remove(artifact_id)
            self.active[artifact_id] = 1
            obj.ref_count_up()
            self.capacity.notify_all()
        return obj
