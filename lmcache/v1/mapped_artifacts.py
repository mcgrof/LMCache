# SPDX-License-Identifier: Apache-2.0
"""Borrow published file pages without an intermediate host allocation."""

# Standard
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator
import hashlib
import json
import math
import mmap
import os
import stat
import threading
import time

# First Party
from lmcache.v1.context_artifacts import (
    ArtifactCatalog,
    ArtifactMetadataStore,
    ArtifactReadPlan,
)
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError


class ArtifactVerification(str, Enum):
    """Select reload checking independently of publication verification."""

    ON_MAP = "on-map"
    PUBLISHED = "published"


@dataclass(eq=False)
class MappedArtifactView:
    """Own a mapping; exported buffers independently retain its lifetime.

    Consumers must not mutate borrowed bytes. A private copy-on-write mapping
    permits tensor libraries to expose buffers without a read-only warning;
    it does not copy payload pages unless a consumer writes them. The store
    never calls mmap.close on exported storage. Python releases the mapping
    only after the final owner or buffer/tensor reference disappears.
    """

    mapping: mmap.mmap
    start: int
    length: int
    plan: ArtifactReadPlan

    @property
    def byte_array(self) -> memoryview:
        """Return a borrowed byte view; retaining it retains the mapping."""
        return memoryview(self.mapping)[self.start : self.start + self.length]


class MappedArtifactStore(ArtifactMetadataStore):
    """Manage mapped artifact identity and leases, leaving pages to the OS.

    Published files must remain unchanged, including their inode contents.
    Entry limits bound retained mapping descriptors/metadata, not page-cache
    RAM, caller-retained views, or virtual bytes. No arena is allocated.
    """

    def __init__(
        self,
        catalog: ArtifactCatalog,
        max_mappings: int = 512,
        max_read_plans: int = 1024,
        verification: ArtifactVerification = ArtifactVerification.ON_MAP,
    ) -> None:
        """Create an empty mapped store without allocating payload buffers.

        Args:
            catalog: Caller-owned immutable publication catalog.
            max_mappings: Positive maximum lookup-resident mappings.
            max_read_plans: Positive metadata cache entry limit.
            verification: Full checksum and catalog recheck on each new map,
                or trust the bytes verified at publication and the store's
                current admitted catalog snapshot. Published mode does not
                detect later payload corruption or catalog edits; neither mode
                permits concurrent writers.

        Raises:
            ValueError: An entry limit or verification policy is invalid.
        """
        super().__init__(catalog, max_read_plans)
        if type(max_mappings) is not int or max_mappings < 1:
            raise ValueError("max_mappings must be positive")
        self.verification = ArtifactVerification(verification)
        self.max_mappings = max_mappings
        self.resident: OrderedDict[str, MappedArtifactView] = OrderedDict()
        self.active: dict[str, int] = {}
        self.capacity = threading.Condition(self.lock)
        self.map_count = 0
        self.map_bytes = 0
        self.checksum_bytes = 0
        self.cache_hits = 0
        # Compatibility counters: mapped access performs no buffered copy.
        self.read_count = 0
        self.read_bytes = 0
        self.capacity_waits = 0
        self.capacity_wait_seconds = 0.0

    @contextmanager
    def lease(
        self,
        artifact_id: str,
        *,
        plan: ArtifactReadPlan | None = None,
        timeout: float = 0.0,
    ) -> Iterator[MappedArtifactView]:
        """Borrow a mapped range until consumption, including transfers, ends.

        Args:
            artifact_id: Exact published identity.
            plan: Optional plan issued by this store's prepare method.
            timeout: Maximum wait for an idle mapping slot; zero fails
                immediately. An already running file/map/hash is not timed out.

        Yields:
            Backing object whose byte_array aliases file-backed pages.

        Raises:
            ValueError: A foreign plan, changed record or checksum mismatch.
            EOFError: The current regular file does not contain the range.
            MemoryError: Every mapping slot is actively borrowed.
            TimeoutError: Waiting for an idle slot exceeded timeout.
            RuntimeError: The store has closed.
            OSError: Opening or mapping the file failed.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout if timeout else None
        plan = self.prepare(artifact_id) if plan is None else plan
        with self.capacity:
            if self.closed:
                raise RuntimeError("artifact store is closed")
            if (
                self._issued.get(id(plan)) is not plan
                or plan.artifact_id != artifact_id
            ):
                raise ValueError("read plan was not issued for this store and identity")
            obj = self.resident.get(artifact_id)
            if obj is not None:
                self.cache_hits += 1
            else:
                while len(self.resident) >= self.max_mappings:
                    victim = next(
                        (k for k in self.resident if not self.active[k]), None
                    )
                    if victim is not None:
                        self.evict(victim)
                        break
                    if deadline is None:
                        raise MemoryError("mapped artifact store has no idle capacity")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LMCacheTimeoutError(
                            "mapped artifact capacity wait expired"
                        )
                    tick = time.monotonic()
                    self.capacity_waits += 1
                    self.capacity.wait(remaining)
                    self.capacity_wait_seconds += time.monotonic() - tick
                    if self.closed:
                        raise RuntimeError("artifact store is closed")
                    # Another waiter may have installed this same identity.
                    obj = self.resident.get(artifact_id)
                    if obj is not None:
                        break
                if obj is None:
                    obj = self._map(plan)
                    self.resident[artifact_id] = obj
                    self.active[artifact_id] = 0
            # A wait releases the lock. Recheck the binding even when another
            # waiter installed this identity while capacity was unavailable.
            if obj.plan.record_json != plan.record_json:
                raise ValueError("resident artifact binding differs from read plan")
            self.resident.move_to_end(artifact_id)
            self.active[artifact_id] += 1
        try:
            yield obj
        finally:
            with self.capacity:
                self.active[artifact_id] -= 1
                self.capacity.notify_all()

    def evict(self, artifact_id: str) -> bool:
        """Drop an idle lookup mapping; exported buffers retain their storage.

        Args:
            artifact_id: Identity to remove from future lookups.

        Returns:
            True if an idle resident entry was removed, otherwise False.
        """
        with self.capacity:
            if self.active.get(artifact_id, 0) or artifact_id not in self.resident:
                return False
            del self.resident[artifact_id]
            del self.active[artifact_id]
            self.capacity.notify_all()
            return True

    def close(self) -> None:
        """Drop lookup ownership after every borrow ends; idempotent.

        Raises:
            RuntimeError: A caller still holds a lease.
        """
        with self.capacity:
            if any(self.active.values()):
                raise RuntimeError("active artifact leases remain")
            self.closed = True
            self.resident.clear()
            self.active.clear()
            self._plans.clear()
            self._issued.clear()
            self.capacity.notify_all()

    def _map(self, plan: ArtifactReadPlan) -> MappedArtifactView:
        """Map and optionally verify one exact range under the store lock."""
        # PUBLISHED trusts the publication-time snapshot this store currently
        # admits, as it trusts the published bytes. A caller-retained plan
        # that is no longer current, and every ON_MAP miss, rechecks the
        # catalog before mapping.
        if (
            self.verification is ArtifactVerification.ON_MAP
            or self._plans.get(plan.artifact_id) is not plan
        ):
            record = self.catalog.lookup_artifact(plan.artifact_id)
            if (
                json.dumps(record, sort_keys=True, separators=(",", ":"))
                != plan.record_json
            ):
                raise ValueError("catalog record differs from admitted read plan")
        with open(plan.path, "rb", buffering=0) as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or plan.offset + plan.length > info.st_size
            ):
                raise EOFError("mapped artifact range exceeds regular file")
            base = (
                plan.offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
            )
            start = plan.offset - base
            # Python >=3.13 can release both the source fd and mmap's duplicate.
            # Older Python keeps one fd per mapping, bounded by max_mappings.
            try:
                mapping = mmap.mmap(  # type: ignore[call-arg]
                    stream.fileno(),
                    start + plan.length,
                    access=mmap.ACCESS_COPY,
                    offset=base,
                    trackfd=False,
                    # Python 3.12 typeshed lacks this 3.13 Unix parameter.
                    # The TypeError fallback below keeps older Python usable.
                )
            except TypeError:
                mapping = mmap.mmap(
                    stream.fileno(),
                    start + plan.length,
                    access=mmap.ACCESS_COPY,
                    offset=base,
                )
        obj = MappedArtifactView(mapping, start, plan.length, plan)
        if self.verification is ArtifactVerification.ON_MAP:
            self.checksum_bytes += plan.length
            if hashlib.sha256(obj.byte_array).hexdigest() != plan.sha256:
                raise ValueError("mapped artifact checksum mismatch")
        self.map_count += 1
        self.map_bytes += plan.length
        return obj
