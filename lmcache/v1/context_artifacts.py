# SPDX-License-Identifier: Apache-2.0
"""Typed external artifacts backed by a catalog and LMCache memory objects."""

# Standard
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, Protocol
import hashlib
import os
import threading

# Third Party
import torch

# First Party
from lmcache.v1.buffered_file import read_file_range_into
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)


class ArtifactCatalog(Protocol):
    """Metadata operations implemented by the cartridge registry."""

    def register_artifact(self, artifact_id: str, record: dict) -> None:
        """Publish immutable metadata after storage completion."""
        ...

    def lookup_artifact(self, artifact_id: str) -> dict | None:
        """Return an exact artifact record, or None."""
        ...


class ArtifactStore:
    """Read immutable file ranges into bounded LMCache allocator memory.

    The catalog owns identity and location. This store owns process-local
    residency. A lease keeps its MemoryObj alive until the caller finishes
    consuming every view, including any asynchronous device transfer.
    This is an external-artifact adapter, not a prompt-token KV cache.
    """

    def __init__(
        self, catalog: ArtifactCatalog, allocator: MemoryAllocatorInterface
    ) -> None:
        self.catalog = catalog
        self.allocator = allocator
        self.resident: OrderedDict[str, MemoryObj] = OrderedDict()
        self.active: dict[str, int] = {}
        self.records: dict[str, dict] = {}
        self.waiters: dict[str, int] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.lock = threading.RLock()
        self.closed = False
        self.read_count = 0
        self.read_bytes = 0
        self.cache_hits = 0

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

    @contextmanager
    def lease(self, artifact_id: str) -> Iterator[MemoryObj]:
        """Acquire one loaded object and release the caller reference on exit.

        Args:
            artifact_id: Exact published identity.

        Yields:
            LMCache MemoryObj containing the verified file range as bytes.

        Raises:
            MemoryError: The allocator is full and no idle object is evictable.
            ValueError: Bytes no longer match their immutable digest.
            RuntimeError: The store has closed.
        """
        record = self.describe(artifact_id)
        with self.lock:
            key_lock = self.locks.setdefault(artifact_id, threading.Lock())
            self.waiters[artifact_id] = self.waiters.get(artifact_id, 0) + 1
        try:
            with key_lock:
                obj = self._acquire(artifact_id, record)
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
            obj.ref_count_down()
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

    def _acquire(self, artifact_id: str, record: dict) -> MemoryObj:
        with self.lock:
            if self.closed:
                raise RuntimeError("artifact store is closed")
            obj = self.resident.get(artifact_id)
            if obj is not None:
                self.resident.move_to_end(artifact_id)
                self.cache_hits += 1
                self.active[artifact_id] += 1
                obj.ref_count_up()
                return obj
            obj = self.allocator.allocate(
                torch.Size([record["length"]]), torch.uint8, MemoryFormat.BINARY
            )
            while obj is None:
                victim = next(
                    (key for key in self.resident if not self.active.get(key, 0)), None
                )
                if victim is None:
                    raise MemoryError("artifact pool has no evictable capacity")
                self.evict(victim)
                obj = self.allocator.allocate(
                    torch.Size([record["length"]]), torch.uint8, MemoryFormat.BINARY
                )
        try:
            buffer = memoryview(obj.byte_array).cast("B")
            count = read_file_range_into(record["path"], buffer, record["offset"])
            if hashlib.sha256(buffer).hexdigest() != record["sha256"]:
                raise ValueError("loaded artifact checksum mismatch")
        except BaseException:
            with self.lock:
                obj.ref_count_down()
            raise
        with self.lock:
            self.read_count += 1
            self.read_bytes += count
            self.resident[artifact_id] = obj
            self.records[artifact_id] = deepcopy(record)
            self.active[artifact_id] = 1
            obj.ref_count_up()
        return obj
