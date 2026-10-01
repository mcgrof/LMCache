# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional
import ctypes
import enum
import hashlib
import json
import os
import re
import stat
import struct
import threading
import time
import uuid
import zlib

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    DiskCacheMetadata,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.raw_block.key_codec import (
    RawBlockKeyNamespace,
    RawBlockKeySpec,
    decode_legacy_key,
    slot_identity_from_encoded_key,
)

logger = init_logger(__name__)


_DEFAULT_META_MAGIC = b"LMCIDX01"
_DEFAULT_META_VERSION = 1
_META_HEADER_STRUCT = struct.Struct("<8sIQQI")
RAW_BLOCK_IO_ENGINES = frozenset({"posix", "io_uring"})
DEFAULT_IOURING_QUEUE_DEPTH = 256
_MAX_PUT_MANY_IO_URING_BATCH_KEYS = 64
_MAX_FDP_PLACEMENT_ID = 0xFFFF

# FDP placement ID semantics are shared by design across raw-block write paths.
# None omits the directive. Explicit identifiers must be positive because
# default writes already use the RUH mapping associated with Placement
# Identifier 0. RawBlockCore rejects explicit identifier 0 so KV data never sends
# an FDP directive for the default placement identifier. Non-zero identifiers are
# encoded as 16-bit NVMe directive-specific values.
# Metadata checkpoint placement is optional. ``None`` keeps the historical
# default NVMe write behavior; a positive identifier emits an FDP directive.
PlacementId = int | None


def round_up(x: int, align: int) -> int:
    """Round a value up to the nearest alignment boundary.

    Args:
        x: Value to align.
        align: Positive alignment in bytes.

    Returns:
        ``x`` rounded up to a multiple of ``align``.
    """
    return ((x + align - 1) // align) * align


def normalize_raw_block_io_engine(
    io_engine: Any = None,
    *,
    use_iouring: Any = None,
    use_uring: Any = None,
) -> str:
    """Normalize raw-block I/O engine config with legacy compatibility.

    Args:
        io_engine: Explicit engine string. Valid values are ``"posix"``,
            and ``"io_uring"``.
        use_iouring: Legacy boolean knob. Used only when ``io_engine`` is not
            set.
        use_uring: Legacy boolean alias. Used only when ``io_engine`` is not
            set.

    Returns:
        The normalized engine string.

    Raises:
        ValueError: If ``io_engine`` names an unsupported engine.
    """
    if io_engine is None or io_engine == "":
        if bool(use_iouring) or bool(use_uring):
            return "io_uring"
        return "posix"
    normalized = str(io_engine).lower()
    if normalized not in RAW_BLOCK_IO_ENGINES:
        allowed = ", ".join(sorted(RAW_BLOCK_IO_ENGINES))
        raise ValueError(f"io_engine must be one of: {allowed}")
    return normalized


def normalize_raw_block_placement_ids(
    placement_ids: Sequence[PlacementId] | None,
    expected_len: int,
    *,
    field_name: str = "placement_ids",
    allow_none: bool = True,
) -> list[PlacementId]:
    """Validate FDP placement identifiers and preserve omitted directives."""
    if placement_ids is None:
        return [None] * expected_len
    if len(placement_ids) != expected_len:
        raise ValueError(f"{field_name} must have length {expected_len}")

    normalized: list[PlacementId] = []
    for placement_id in placement_ids:
        if placement_id is None:
            if not allow_none:
                raise ValueError(f"{field_name} must contain integers")
            normalized.append(None)
            continue
        if not isinstance(placement_id, int) or isinstance(placement_id, bool):
            raise ValueError(f"{field_name} must contain integers or None")
        if placement_id == 0:
            raise ValueError(f"{field_name} must not contain placement identifier 0")
        if placement_id < 0:
            raise ValueError(f"{field_name} must contain positive integers or None")
        if placement_id > _MAX_FDP_PLACEMENT_ID:
            raise ValueError(
                f"{field_name} must contain placement identifiers in range "
                f"1..={_MAX_FDP_PLACEMENT_ID}"
                f"{' or None' if allow_none else ''}"
            )
        normalized.append(int(placement_id))
    return normalized


def validate_raw_block_io_options(
    *,
    iouring_queue_depth: int,
) -> None:
    """Validate numeric raw-block I/O engine options.

    Args:
        iouring_queue_depth: Queue depth for the Rust io_uring path.

    Raises:
        ValueError: If any numeric option is not positive.
    """
    if int(iouring_queue_depth) <= 0:
        raise ValueError("iouring_queue_depth must be > 0")


def _resolve_sysfs_queue_dir(device_path: str) -> Optional[str]:
    """Resolve sysfs queue directory for NVMe character device paths."""
    base_name = os.path.basename(device_path)
    match = re.fullmatch(r"ng(\d+)n(\d+)", base_name)
    if match:
        ctrl, nsid = match.groups()
        return f"/sys/block/nvme{ctrl}n{nsid}/queue"
    return None


# A block namespace that exposes no hardware identity gets one built from
# its device numbers. Those are assigned by the local kernel, so they say
# nothing about which device another node would reach by the same numbers.
_LOCAL_BLOCK_IDENTITY_PREFIX = "block-local:"


# Bumped when the bytes of a key, or the way one is derived, change in a way
# that makes an older namespace unreadable. It is part of the descriptor
# rather than implied by the metadata version because two writers can agree
# on every geometry field and still derive different keys.
KEY_CODEC_VERSION = 1


@dataclass(frozen=True)
class RawBlockDerivationDescriptor:
    """How the keys in a namespace were derived.

    Two writers can agree on every geometry field in a checkpoint and still
    produce different keys: a different hash function, a different seed, a
    different chain root or a different key encoding all change the bytes
    while leaving the layout identical. A reader that adopts such a namespace
    does not fail loudly -- it misses every key, which looks like a cold
    cache.

    So this travels with the namespace and is compared before adoption. It is
    deliberately not part of the geometry checks: those treat a mismatch as
    "ignore this metadata and start empty", and starting empty is exactly the
    wrong response to a namespace someone else is writing with a different
    derivation.
    """

    hash_algorithm: str
    hash_implementation: str
    hash_seed: str
    chain_root: str
    key_codec_version: int = KEY_CODEC_VERSION
    key_namespace: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "hash_algorithm": self.hash_algorithm,
            "hash_implementation": self.hash_implementation,
            "hash_seed": self.hash_seed,
            "chain_root": self.chain_root,
            "key_codec_version": int(self.key_codec_version),
            "key_namespace": self.key_namespace,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> Optional["RawBlockDerivationDescriptor"]:
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                hash_algorithm=str(payload["hash_algorithm"]),
                hash_implementation=str(payload["hash_implementation"]),
                hash_seed=str(payload["hash_seed"]),
                chain_root=str(payload["chain_root"]),
                key_codec_version=int(payload["key_codec_version"]),
                key_namespace=str(payload.get("key_namespace", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def uses_the_interpreter_hash(self) -> bool:
        """Whether keys are hashed by the interpreter's own ``hash``.

        That is the only case in which the process hash seed can reach a key,
        and then only through a string: measured on this interpreter,
        ``hash((0, (1, 2, 3), ()))`` is identical under seeds 0, 12345 and
        99, while the same tuple carrying a string differs under each.
        """
        return "builtin" in self.hash_implementation.lower()

    def compared_fields(self, other: "RawBlockDerivationDescriptor") -> tuple[str, ...]:
        """Which fields have to agree for two engines to read one namespace.

        The seed is recorded always and compared only where it can matter.
        Comparing it unconditionally refuses a namespace two nodes derive
        identically, which is an availability failure invented by the check
        rather than found by it: a cryptographic hash does not consult the
        seed, and neither does the interpreter's for a key of integers.
        """
        fields = [
            "hash_algorithm",
            "hash_implementation",
            "chain_root",
            "key_codec_version",
            "key_namespace",
        ]
        if self.uses_the_interpreter_hash() or other.uses_the_interpreter_hash():
            fields.append("hash_seed")
        return tuple(fields)

    def describe_mismatch(self, other: "RawBlockDerivationDescriptor") -> list[str]:
        """Name every field on which two derivations disagree."""
        return [
            f"{field}: ours={getattr(self, field)!r} theirs={getattr(other, field)!r}"
            for field in self.compared_fields(other)
            if getattr(self, field) != getattr(other, field)
        ]


# What a byte moving through this engine was for. A header commits a slot
# and a checkpoint commits an index; neither is cache payload, and adding
# them together produces a number that answers no question -- "did the
# direct path carry the KV" least of all.
IO_KIND_PAYLOAD = "payload"
IO_KIND_SLOT_HEADER = "slot_header"
IO_KIND_CHECKPOINT = "checkpoint"

# Which route actually carried it. Recorded where the route is chosen
# rather than where it was asked for, because what an operator needs to
# know is which path ran, not which one the configuration implies.
IO_PATH_SYNC = "sync"
IO_PATH_IOURING_BATCHED = "iouring_batched"
IO_PATH_IOURING_BOUNDED = "iouring_bounded"
IO_PATH_IOURING_PER_WRITE = "iouring_per_write"


@dataclass
class RawBlockIoTally:
    """One direction, one kind and one path, counted at two granularities.

    A logical request is what a caller asked this engine to move. A
    physical operation is one transfer handed to the device, and a route
    that splits a request issues several for one of them -- so the two are
    named apart rather than compared as though they were the same unit. The
    bytes belong to the logical side, because a split does not move more of
    them.

    ``submitted`` is what this engine handed over; it is deliberately not
    called "accepted", because whether the kernel took an entry is the
    kernel's answer and not this count. ``completed`` is what the device
    reported finishing successfully. The difference between those two is
    the whole question: a submission with no completion is an operation
    whose outcome nobody established.
    """

    logical_requests: int = 0
    logical_bytes: int = 0
    padded_bytes: int = 0
    submitted_operations: int = 0
    submitted_padded_bytes: int = 0
    completed_operations: int = 0
    completed_padded_bytes: int = 0

    def as_payload(self) -> dict[str, int]:
        return {
            "logical_requests": self.logical_requests,
            "logical_bytes": self.logical_bytes,
            "padded_bytes": self.padded_bytes,
            "submitted_operations": self.submitted_operations,
            "submitted_padded_bytes": self.submitted_padded_bytes,
            "completed_operations": self.completed_operations,
            "completed_padded_bytes": self.completed_padded_bytes,
        }


@dataclass(frozen=True)
class RawBlockIoContext:
    """Immutable identity for the work one batch of I/O belongs to.

    Carried with the job rather than looked up when a completion is reaped.
    A "current request" global, thread-local or otherwise, is read on
    whichever thread happens to be reaping -- the executor thread that
    submitted is already gone by then, and the I/O worker never had the
    context at all -- so an attribution built that way names whoever was
    most recently active instead of whoever the bytes are for.

    ``request_id`` is the name the *peer* knows this request by, because
    that is what a receipt, a READY status and an acknowledgement all carry;
    an engine-local name correlates with nothing on the other side.
    """

    run_id: str = ""
    request_id: str = ""
    tp_rank: int = -1
    incarnation: str = ""

    def tag(self) -> str:
        """One opaque string, stable for the life of this request."""
        if not self.request_id:
            return ""
        rank = "?" if self.tp_rank < 0 else str(self.tp_rank)
        return f"{self.run_id}/{self.request_id}/r{rank}/{self.incarnation}"


class RawBlockIoAttribution:
    """What each request's operations did, by the path they actually took.

    Separate from the ledger because it answers a different question. The
    ledger says how much a route moved; this says which request moved it,
    over which buffer path the device really used, and whether every
    operation was answered for.

    The "path" here is the native engine's own choice -- registered dma-buf,
    classic host-fixed, bounce or ordinary -- not the Python helper route
    that led there. A configured dma-buf pool whose buffers miss the
    registration map issues ordinary SQEs, and the configuration cannot tell
    you that.

    Bounded by the number of requests it remembers, because a long serving
    run has unboundedly many. Eviction is counted, so a reader can see that
    a sum is incomplete rather than trusting one that quietly lost rows.
    """

    _OUTCOMES = ("submitted", "completed", "short", "failed")

    def __init__(self, max_requests: int = 4096) -> None:
        if max_requests <= 0:
            raise ValueError("an attribution record needs a request bound")
        self._lock = threading.Lock()
        self._max_requests = max_requests
        self._rows: OrderedDict[tuple[str, str, str], dict[str, int]] = OrderedDict()
        self._tags: OrderedDict[str, None] = OrderedDict()
        self.dropped_rows = 0
        self.evicted_requests = 0
        self.untagged_operations = 0

    def record(
        self,
        rows: Sequence[Mapping[str, str]],
        dropped: int = 0,
    ) -> None:
        """Take a drained native journal, one row per operation event."""
        with self._lock:
            self.dropped_rows += int(dropped)
            for row in rows:
                outcome = str(row.get("outcome", ""))
                if outcome not in self._OUTCOMES:
                    continue
                tag = str(row.get("request_tag", ""))
                if not tag:
                    # An operation nobody named. Counted rather than
                    # attributed to a neighbour, because guessing which
                    # request it belonged to is what this record exists to
                    # avoid.
                    self.untagged_operations += 1
                    continue
                key = (tag, str(row.get("direction", "")), str(row.get("path", "")))
                counts = self._rows.get(key)
                if counts is None:
                    counts = dict.fromkeys(self._OUTCOMES, 0)
                    counts["bytes"] = 0
                    self._rows[key] = counts
                counts[outcome] += 1
                if outcome != "submitted":
                    counts["bytes"] += max(0, int(row.get("bytes", 0) or 0))
                self._tags.pop(tag, None)
                self._tags[tag] = None
            while len(self._tags) > self._max_requests:
                evicted, _ = self._tags.popitem(last=False)
                for key in [key for key in self._rows if key[0] == evicted]:
                    del self._rows[key]
                self.evicted_requests += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Every attributed row, keyed "request/direction/native-path"."""
        with self._lock:
            return {
                f"{tag}/{direction}/{path}": dict(counts)
                for (tag, direction, path), counts in sorted(self._rows.items())
            }

    def unanswered(self) -> dict[str, int]:
        """Rows where operations were submitted and never answered for.

        A nonzero entry here is a failed evidence gate: the bytes of that
        request cannot be accounted for, and no total that happens to add up
        changes that.
        """
        with self._lock:
            gaps: dict[str, int] = {}
            for (tag, direction, path), counts in sorted(self._rows.items()):
                answered = sum(
                    counts[outcome] for outcome in ("completed", "short", "failed")
                )
                outstanding = counts["submitted"] - answered
                if outstanding:
                    gaps[f"{tag}/{direction}/{path}"] = outstanding
            return gaps

    def as_payload(self) -> dict[str, Any]:
        """Everything a receipt needs, including what makes it incomplete."""
        return {
            "rows": self.snapshot(),
            "unanswered": self.unanswered(),
            "untagged_operations": self.untagged_operations,
            "dropped_rows": self.dropped_rows,
            "evicted_requests": self.evicted_requests,
        }


def _io_tag(io_context: Optional[RawBlockIoContext]) -> str:
    """The attribution one batch carries, or nothing when it has none."""
    return io_context.tag() if io_context is not None else ""


class RawBlockIoLedger:
    """Count what each route moved, exactly enough to subtract two runs.

    Every update takes the lock. These are small integer additions on a
    path that already performs a device I/O, and the alternative -- lost
    concurrent increments -- makes a delta approximate, which is exactly
    what cannot be used to certify that a fallback path carried nothing.
    A total that is only nearly right cannot answer "was it zero".
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tallies: dict[tuple[str, str, str], RawBlockIoTally] = {}

    def _tally_locked(self, key: tuple[str, str, str]) -> RawBlockIoTally:
        tally = self._tallies.get(key)
        if tally is None:
            tally = RawBlockIoTally()
            self._tallies[key] = tally
        return tally

    def logical_request(
        self,
        direction: str,
        path: str,
        kinds: Sequence[str],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> None:
        """Record what a caller asked this engine to move, once per request."""
        with self._lock:
            for kind, payload_len, total_len in zip(
                kinds, payload_lens, total_lens, strict=True
            ):
                tally = self._tally_locked((direction, kind, path))
                tally.logical_requests += 1
                tally.logical_bytes += int(payload_len)
                tally.padded_bytes += int(total_len)

    def submitted(
        self,
        direction: str,
        path: str,
        kinds: Sequence[str],
        total_lens: Sequence[int],
    ) -> None:
        """Record physical operations handed to the device, one per transfer.

        Counted at the granularity the device sees, which is what makes it
        comparable with the completions. It says these were handed over; it
        does not say the kernel took them, which is the kernel's answer.
        """
        with self._lock:
            for kind, total_len in zip(kinds, total_lens, strict=True):
                tally = self._tally_locked((direction, kind, path))
                tally.submitted_operations += 1
                tally.submitted_padded_bytes += int(total_len)

    def completed(
        self,
        direction: str,
        path: str,
        kinds: Sequence[str],
        total_lens: Sequence[int],
        succeeded: Sequence[bool],
    ) -> None:
        """Record operations the device reported finishing."""
        with self._lock:
            for kind, total_len, ok in zip(kinds, total_lens, succeeded, strict=False):
                if not ok:
                    continue
                tally = self._tally_locked((direction, kind, path))
                tally.completed_operations += 1
                tally.completed_padded_bytes += int(total_len)

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Take every tally as flat rows, keyed "direction/kind/path"."""
        with self._lock:
            return {
                f"{direction}/{kind}/{path}": tally.as_payload()
                for (direction, kind, path), tally in sorted(self._tallies.items())
            }

    def totals(self, *, direction: str = "", kind: str = "") -> RawBlockIoTally:
        """Add up the rows matching a direction and kind, or all of them."""
        summed = RawBlockIoTally()
        with self._lock:
            for (row_direction, row_kind, _path), tally in self._tallies.items():
                if direction and row_direction != direction:
                    continue
                if kind and row_kind != kind:
                    continue
                summed.logical_requests += tally.logical_requests
                summed.logical_bytes += tally.logical_bytes
                summed.padded_bytes += tally.padded_bytes
                summed.submitted_operations += tally.submitted_operations
                summed.submitted_padded_bytes += tally.submitted_padded_bytes
                summed.completed_operations += tally.completed_operations
                summed.completed_padded_bytes += tally.completed_padded_bytes
        return summed


def device_has_nothing_outstanding(device: Any) -> bool:
    """Whether a device has answered for everything it was handed.

    This is the question a writer's final index may be published on, and it
    is deliberately weaker than proof of quiescence. Proof requires stopping
    the worker, and stopping the worker takes away the thing the index write
    needs -- so a writer that drained first would hang, and one that
    published first would be deciding on a question asked too early.

    What it does establish is that nothing is in flight and nothing has been
    quarantined as of now. A submission the worker quarantines later, while
    shutting down, is a separate event that only the close reports; a
    checkpoint published before it is recorded as written but not vouched
    for.

    Fails closed for the same reason the health probe does: a probe that
    raises was asked and could not answer. A device that cannot be asked at
    all is an older native build, which can still answer the health
    question.
    """
    probe = getattr(device, "is_idle", None)
    if probe is None:
        return not device_says_outcome_is_unknown(device)
    try:
        answer = probe()
    except Exception:
        logger.exception("RawBlockCore could not ask the device for its state")
        return False
    return answer is True


def device_says_outcome_is_unknown(device: Any) -> bool:
    """Ask a device whether it has stopped being able to say, failing closed.

    A probe that raises was asked and could not answer, which is exactly the
    condition to treat as unknown. Reading the raise as health is worse than
    not asking: the caller then recycles on the strength of a question that
    was never answered. A probe that returns something other than a bool is
    an object that does not answer this question, and reading that as unknown
    would quarantine on nothing.

    A device with no such probe cannot report the condition at all, so the
    absence is not an unanswered question and reads as healthy. That is a
    real gap against an older native build, which is why the core keeps its
    own flag once it has adopted one.
    """
    probe = getattr(device, "is_poisoned", None)
    if probe is None:
        return False
    try:
        answer = probe()
    except Exception:
        logger.exception("RawBlockCore could not read device health")
        return True
    return answer is True


class NativeQuiescence(enum.Enum):
    """What a close established about the device it was closing.

    ``PROVEN`` is the only one that authorizes releasing anything the device
    was given: the worker stopped, its registration was withdrawn and the
    descriptor was closed. ``RETAINED`` covers every way that failed --
    refused, poisoned, or a wait that gave up -- because they differ in what
    to report, not in what may be released. ``NOT_ATTEMPTED`` exists so that
    a second caller cannot read a repeat close as a fresh proof.
    """

    PROVEN = "proven"
    RETAINED = "retained"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class RawBlockCloseOutcome:
    """The result of closing one core, and what it permits."""

    quiescence: NativeQuiescence
    poisoned: bool
    reason: str = ""
    quarantined_slots: int = 0
    final_checkpoint_written: bool = False

    @property
    def final_checkpoint_is_vouched_for(self) -> bool:
        """Whether the published index names only extents that were proven.

        A writer publishes its last index while it still has a worker to
        write with, and can only know at that point that everything it had
        handed over was answered for. If the close that follows then cannot
        prove quiescence -- because the worker quarantined something on its
        way out -- the index is already durable and may name an extent whose
        write was never observed. It is written and it is not vouched for,
        and those are different things a caller has to be able to tell
        apart.
        """
        return self.final_checkpoint_written and self.quiescence is (
            NativeQuiescence.PROVEN
        )

    @property
    def may_release_backing_resources(self) -> bool:
        """Whether the memory and descriptors behind this device are free.

        Local quiescence only. It says nothing about a reader elsewhere still
        holding a lease, which is a separate decision with a separate owner.
        """
        return self.quiescence is NativeQuiescence.PROVEN and not self.poisoned


class IncompatibleKeyDerivation(RuntimeError):
    """A namespace's keys were derived in a way this engine cannot reproduce."""


def namespace_identity_is_shareable(identity: str) -> bool:
    """Return whether an identity names the same namespace on another node."""
    return bool(identity) and not identity.startswith(_LOCAL_BLOCK_IDENTITY_PREFIX)


def _resolve_namespace_identity(device_path: str) -> str:
    """Return a stable-enough identity for checkpoint and handoff receipts."""
    try:
        device_stat = os.stat(device_path)
    except OSError:
        # Unit-test fakes and file-backed development targets may not exist
        # until the native engine opens them.  A canonical path is sufficient
        # to fence two such endpoints; production block devices still require
        # a persistent hardware identity below.
        return f"path:{os.path.realpath(device_path)}"
    if stat.S_ISREG(device_stat.st_mode):
        return f"file:{device_stat.st_dev}:{device_stat.st_ino}"
    if stat.S_ISBLK(device_stat.st_mode):
        major = os.major(device_stat.st_rdev)
        minor = os.minor(device_stat.st_rdev)
        sysfs_device = f"/sys/dev/block/{major}:{minor}"
        for field in ("wwid", "uuid", "nguid", "eui"):
            for candidate in (
                os.path.join(sysfs_device, field),
                os.path.join(sysfs_device, "device", field),
            ):
                try:
                    with open(candidate, "r", encoding="utf-8") as identity_file:
                        value = identity_file.read().strip()
                except OSError:
                    continue
                if value:
                    return f"block:{field}:{value}"
        # No hardware identity is exposed for this namespace. That only
        # matters for a target two nodes share, so name it in a way that
        # says as much and let publication be the thing that refuses it.
        # An ordinary local cache on such a device stays usable.
        return f"{_LOCAL_BLOCK_IDENTITY_PREFIX}{major}:{minor}"
    # A character device (the io_uring_cmd passthrough node) is identified by
    # the device it refers to, not by the filesystem its node lives on:
    # st_dev names devtmpfs and would differ between a container and its host
    # for the same namespace.
    return f"device:{os.major(device_stat.st_rdev)}:{os.minor(device_stat.st_rdev)}"


def _read_sysfs_int(path: str) -> Optional[int]:
    """Read an integer value from sysfs and return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def _device_payload_tensor(memory_obj: MemoryObj) -> Optional[torch.Tensor]:
    """Return the flat device tensor behind a memory object, or None on CPU.

    A memory object whose storage lives on a GPU has no buffer-protocol view,
    so ``byte_array`` cannot describe it.  The Rust engine accepts any object
    exposing ``data_ptr()`` and ``nbytes`` in place of a buffer, and a paged
    GPU allocator registers its slots with io_uring as dma-bufs, so a device
    object hands the engine the physical uint8 tensor the allocator carved
    it from and the read or write moves VRAM to NVMe with no host copy.
    """
    raw = getattr(memory_obj, "raw_data", None)
    if not isinstance(raw, torch.Tensor) or raw.device.type == "cpu":
        return None
    if not raw.is_contiguous():
        raise RuntimeError(
            "RawBlockCore: device memory object is not contiguous; the engine "
            "needs one flat region to hand to io_uring"
        )
    return raw.view(-1).view(torch.uint8)


class UnsupportedDevicePayload(Exception):
    """A device object this lane cannot describe a transfer for.

    Raised while planning, before any slot is reserved, so a caller turns it
    into a failed result for that key rather than abandoning storage it has
    already allocated.
    """


def _logical_payload_len(memory_obj: MemoryObj) -> int:
    """Return the logical payload length of a memory object in bytes.

    Device objects have no ``byte_array`` (it would need a host copy), so
    their logical size comes from the metadata instead.
    """
    if _device_payload_tensor(memory_obj) is not None:
        # TensorMemoryObj.get_size() returns a group prefix sum computed once
        # at construction from the pool's grouped shapes.  A paged allocator
        # rebinds a free block by updating the scalar meta.shape and slicing
        # raw_data, and never refreshes that sum, so a reused block reports
        # the pool's full chunk even when it now holds the final partial
        # chunk of a request.  The engine writes from raw_data, so take the
        # length from what this object actually describes.
        #
        # An explicit used-length override is already the logical length, and
        # a grouped object's length is the sum over its groups: neither may be
        # replaced by the scalar shape.  This lane supports only the
        # single-group device representation, so refuse anything else rather
        # than silently truncating it.
        # Decide whether the representation is supported before trusting any
        # length from it.  An override narrows a length; it does not make a
        # grouped object single-group, so it cannot be returned first.
        meta = memory_obj.metadata
        shapes = getattr(meta, "shapes", None)
        if shapes is not None and len(shapes) > 1:
            raise UnsupportedDevicePayload(
                f"a device object with {len(shapes)} representation groups "
                "is not supported by the raw-block lane; its logical length "
                "is the sum over the groups, not the scalar shape"
            )
        if getattr(memory_obj, "_used_size_override", None) is not None:
            return int(memory_obj.get_size())
        if meta.shape is not None and meta.dtype is not None:
            return int(meta.shape.numel()) * int(meta.dtype.itemsize)
        return int(memory_obj.get_size())
    return len(memory_obj.byte_array)


@dataclass(frozen=True)
class RawBlockCoreConfig:
    """Configuration for RawBlockCore device layout, I/O, and checkpoints."""

    device_path: str
    capacity_bytes: int
    block_align: int
    header_bytes: int
    slot_bytes: int
    use_odirect: bool
    enable_zero_copy: bool
    meta_total_bytes: int
    meta_magic: bytes
    meta_version: int
    meta_checkpoint_interval_sec: int
    meta_idle_quiet_ms: int
    meta_enable_periodic: bool
    meta_verify_on_load: bool
    max_data_transfer_size: int = 0
    load_checkpoint_on_init: bool = True
    io_engine: str = "posix"
    iouring_queue_depth: int = DEFAULT_IOURING_QUEUE_DEPTH
    use_uring_cmd: bool = False
    meta_checkpoint_placement_id: PlacementId = None
    fdp_slot_affinity_enabled: bool = False
    # "writer" owns the device: it allocates slots, stores objects and writes
    # the on-device index checkpoints.  "reader" shares the same device from
    # another process or host, never writes, and adopts the writer's index
    # through refresh_index_from_device().  This is how a prefill node hands
    # KV chunks to a decode node over a shared NVMe namespace.
    role: str = "writer"
    # Read each slot's header alongside its payload and reject the load when
    # the header names a different key: protects a reader from a slot the
    # writer has since reused.
    verify_slot_header_on_load: bool = False
    # Minimum spacing between two publish_index() checkpoints, so a writer
    # that publishes after every put batch does not rewrite the index more
    # often than this.
    publish_min_interval_ms: int = 0
    # Refuse startup unless every staging buffer is registered through the
    # dma-buf io_uring path. This is used by correctness runs that must not
    # silently fall back to classic fixed or per-command mapped buffers.
    require_dmabuf_registration: bool = False
    # Unique writer incarnation carried in every checkpoint. An empty value
    # creates a fresh UUID for a writer and is populated from checkpoints by a
    # reader.
    writer_epoch: str = ""
    # Persistent namespace identity carried in request receipts. When omitted,
    # regular files use device/inode identity and block devices use sysfs WWID,
    # UUID, NGUID, or EUI data. Setting it is an operator's assertion that
    # every node using this string reaches the same namespace; nothing here
    # verifies that, so two nodes given the same string for different
    # devices will read each other's receipts as their own.
    namespace_identity: str = ""
    # Whether close writes one last checkpoint. The strict shared-storage
    # lane sets this false: every successful READY already required its own
    # completed forced publication, so a close-time generation can only
    # make an unfinished request look complete -- and writing one needs the
    # device at exactly the moment teardown is trying to settle it.
    close_writes_final_checkpoint: bool = True
    # Refuse to run unless the native engine can be asked whether anything
    # is outstanding. An older build cannot, and the health question it can
    # answer is a weaker one; accepting that silently in the lane whose
    # whole point is knowing would qualify a build on a question it never
    # asked.
    require_native_idle_capability: bool = False
    # How this engine derives its keys. Required where a namespace is shared,
    # because two engines agreeing on every field above can still derive
    # different keys and read nothing of each other's. Absent elsewhere, and
    # nothing is compared then.
    derivation: Optional[RawBlockDerivationDescriptor] = None


@dataclass
class _Entry:
    offset: int
    size: int
    meta: DiskCacheMetadata


@dataclass
class _Inflight:
    offset: int
    meta: DiskCacheMetadata
    canceled: bool = False


@dataclass(frozen=True)
class RawBlockPutManyResult:
    """Result of a RawBlockCore batched write."""

    results: list[bool]
    stored_keys: list[str]


@dataclass(frozen=True)
class RawBlockPublicationReceipt:
    """Identify one published request manifest."""

    writer_epoch: str
    checkpoint_seq: int
    key_count: int
    manifest_digest: str
    namespace_identity: str = ""
    total_logical_bytes: int = 0
    total_padded_bytes: int = 0
    # Where the writer that produced this listens for the acknowledgement
    # that releases its extents. Empty means it is not listening, so a
    # consumer cannot release anything by replying.
    ack_endpoint: str = ""


class RawBlockCore:
    """
    Shared raw-block storage engine used by both legacy non-MP and MP L2 paths.

    This class owns the raw-device I/O path, slot allocation, checkpoint/recovery,
    and lock refcounts that protect slots from deletion while in use.
    """

    # Declared on the class so it exists however the core was built. Several
    # tests construct one through __new__ without running __init__, and a
    # guard that only the initializer installs is a guard that is sometimes
    # absent from the very paths it protects.
    _poisoned: bool = False
    # Whether close has decided this device's fate. Declared here for the
    # same reason: a reopen guard that only the initializer installs is
    # absent from a core built another way.
    _terminal: bool = False
    # Slots withheld because the device may still be writing them. A class
    # default keeps it readable on a core built without __init__; the first
    # quarantine replaces it with an instance dict.
    _quarantined_slots: Optional[dict[int, None]] = None
    # Byte counters, declared here for the same reason: they are metrics, and
    # a core built without __init__ must still be able to add to them.
    # Bytes this engine never had to write because the key was already on
    # the device. Not an I/O quantity, so it is not in the ledger.
    _bytes_deduplicated: int = 0
    # The attribution record, declared on the class for the same reason as
    # the ledger below: a core built without __init__ must still be able to
    # attribute an operation.
    _io_attribution_instance: Optional["RawBlockIoAttribution"] = None
    # This engine's own metadata traffic: index checkpoints and slot headers
    # it reads to validate. Named rather than left unattributed, because an
    # unnamed operation is a gap in a request's accounting and these are not
    # that -- they belong to no request, which is a different statement.
    _metadata_io_context = RawBlockIoContext(request_id="<engine-metadata>")
    # The I/O ledger, declared on the class so it exists however the core
    # was built. Several tests construct one through __new__ without
    # running __init__, and accounting that only exists when the
    # initializer ran is accounting missing from paths that use it. It
    # cannot be a shared class-level instance: two cores would add into
    # one ledger, so it is created per instance on first use.
    _io_ledger_instance: Optional["RawBlockIoLedger"] = None
    _io_ledger_creation_lock = threading.Lock()
    # What the first close established. Read by a repeat close, which must
    # not reach the device accessor and open one that knows nothing.
    _close_outcome: Optional["RawBlockCloseOutcome"] = None

    def __init__(
        self,
        config: RawBlockCoreConfig,
        *,
        key_namespace: RawBlockKeyNamespace,
    ):
        """Initialize the raw-block storage engine.

        Args:
            config: Raw-block device, layout, I/O, and checkpoint settings.
            key_namespace: Encoding namespace used by keys stored in this core.

        Raises:
            ValueError: If the supplied configuration is invalid.
            RuntimeError: If the raw device cannot be opened or the computed
                layout cannot fit metadata and at least one data slot.

        Notes:
            If initialization opens the device and a later recovery step fails,
            the partially opened resources are closed before the exception is
            re-raised.
        """
        self.device_path = config.device_path
        self.capacity_bytes = int(config.capacity_bytes)
        self.block_align = int(config.block_align)
        self.header_bytes = int(config.header_bytes)
        self.slot_bytes = int(config.slot_bytes)
        self.use_odirect = bool(config.use_odirect)
        self.enable_zero_copy = bool(config.enable_zero_copy)

        self.meta_total_bytes = int(config.meta_total_bytes)
        self.meta_magic = bytes(config.meta_magic)
        self.meta_version = int(config.meta_version)
        self.meta_checkpoint_interval_sec = int(config.meta_checkpoint_interval_sec)
        self.meta_idle_quiet_ms = int(config.meta_idle_quiet_ms)
        self.meta_enable_periodic = bool(config.meta_enable_periodic)
        self.load_checkpoint_on_init = bool(config.load_checkpoint_on_init)
        self.meta_verify_on_load = bool(config.meta_verify_on_load)
        self.close_writes_final_checkpoint = bool(config.close_writes_final_checkpoint)
        self.require_native_idle_capability = bool(
            config.require_native_idle_capability
        )
        self.role = str(getattr(config, "role", "writer") or "writer")
        if self.role not in ("writer", "reader"):
            raise ValueError(
                f"RawBlockCore role must be 'writer' or 'reader', got {self.role!r}"
            )
        self.verify_slot_header_on_load = bool(
            getattr(config, "verify_slot_header_on_load", False)
        )
        self.publish_min_interval_ms = int(
            getattr(config, "publish_min_interval_ms", 0) or 0
        )
        self.require_dmabuf_registration = bool(
            getattr(config, "require_dmabuf_registration", False)
        )
        self._last_publish_ts: float = 0.0
        self._buffer_registration_mode = "none"
        # Set when the native engine could not establish what the device was
        # doing. Separate from ``_closed``: marking the core closed would make
        # ``close()`` return without draining or unregistering, and the point
        # is to hold resources rather than release them.
        self._poisoned = False
        # Set once close has decided this device's fate. Separate from
        # ``_closed``, which is set on the way in so nothing new is
        # admitted: close's own final checkpoint still runs between the two,
        # and it needs the device it is checkpointing.
        self._terminal = False
        # Supplied by whoever knows the token database's effective settings.
        # Absent outside the strict shared-storage lane, where nothing else
        # is reading these keys.
        self._derivation: Optional[RawBlockDerivationDescriptor] = getattr(
            config, "derivation", None
        )
        configured_namespace = str(getattr(config, "namespace_identity", "") or "")
        self.namespace_identity = configured_namespace or _resolve_namespace_identity(
            self.device_path
        )
        configured_epoch = str(getattr(config, "writer_epoch", "") or "")
        self._writer_epoch = (
            configured_epoch
            if configured_epoch
            else str(uuid.uuid4())
            if self.role == "writer"
            else ""
        )
        self._published_keys: frozenset[str] = frozenset()
        self._published_manifest: dict[str, dict[str, Any]] = {}
        self._published_writer_epoch = ""
        self.io_engine = normalize_raw_block_io_engine(config.io_engine)
        self.iouring_queue_depth = int(config.iouring_queue_depth)
        self.use_uring_cmd = bool(config.use_uring_cmd)
        self.max_data_transfer_size = 0
        self.fdp_slot_affinity_enabled = bool(config.fdp_slot_affinity_enabled)
        self.meta_checkpoint_placement_id = normalize_raw_block_placement_ids(
            [config.meta_checkpoint_placement_id],
            1,
            field_name="meta_checkpoint_placement_id",
        )[0]
        if self.meta_checkpoint_placement_id is not None and (
            self.io_engine != "io_uring" or not self.use_uring_cmd
        ):
            raise ValueError(
                "meta_checkpoint_placement_id requires "
                "io_engine='io_uring' and use_uring_cmd=true"
            )
        if self.use_uring_cmd and self.use_odirect:
            logger.warning(
                "RawBlockCore: use_odirect is ignored for NVMe namespace "
                "character devices when use_uring_cmd=true"
            )
            self.use_odirect = False
        self.key_namespace = key_namespace

        if not self.device_path:
            raise ValueError("RawBlockCore requires a non-empty device_path")
        if self.block_align <= 0 or (self.block_align & (self.block_align - 1)) != 0:
            raise ValueError(
                f"block_align must be a power of 2, got {self.block_align}"
            )
        if self.header_bytes < 24:
            raise ValueError("header_bytes must be >= 24")
        if self.header_bytes % self.block_align != 0:
            raise ValueError("header_bytes must be a multiple of block_align")
        if self.slot_bytes < self.header_bytes + 1:
            raise ValueError("slot_bytes must be >= header_bytes + 1")
        if self.slot_bytes % self.block_align != 0:
            raise ValueError("slot_bytes must be a multiple of block_align")
        if self.meta_total_bytes <= self.block_align:
            raise ValueError("meta_total_bytes must provide room for metadata header")
        if self.meta_total_bytes % self.block_align != 0:
            raise ValueError("meta_total_bytes must be a multiple of block_align")
        if len(self.meta_magic) != 8:
            raise ValueError("meta_magic must be exactly 8 bytes")
        if self.meta_version <= 0:
            raise ValueError("meta_version must be > 0")
        validate_raw_block_io_options(
            iouring_queue_depth=self.iouring_queue_depth,
        )
        if self.use_uring_cmd and self.io_engine != "io_uring":
            raise ValueError("use_uring_cmd requires io_uring as io_engine")
        if self.require_dmabuf_registration:
            if self.io_engine != "io_uring":
                raise ValueError(
                    "require_dmabuf_registration requires io_engine='io_uring'"
                )
            if not self.use_odirect:
                raise ValueError(
                    "require_dmabuf_registration requires use_odirect=true"
                )
            if self.use_uring_cmd:
                raise ValueError(
                    "require_dmabuf_registration is incompatible with use_uring_cmd"
                )
            try:
                target_mode = os.stat(self.device_path).st_mode
            except OSError as exc:
                raise ValueError(
                    "require_dmabuf_registration requires an existing block device"
                ) from exc
            if not stat.S_ISBLK(target_mode):
                raise ValueError(
                    "require_dmabuf_registration requires a block-device target"
                )
        if self.use_uring_cmd:
            try:
                mode = os.stat(self.device_path).st_mode
            except OSError as e:
                raise ValueError(
                    "use_uring_cmd requires an existing NVMe namespace "
                    f"character device path, got {self.device_path!r}"
                ) from e
            if not stat.S_ISCHR(mode):
                raise ValueError(
                    "use_uring_cmd requires an NVMe namespace character device "
                    f"(for example /dev/ng0n1), got {self.device_path!r}"
                )
            # Validate NVMe generic namespace naming pattern (ng<ctrl>n<ns>)
            basename = os.path.basename(self.device_path)
            if not re.match(r"^ng\d+n\d+$", basename):
                raise ValueError(
                    "use_uring_cmd requires an NVMe generic namespace character device "
                    f"with naming pattern ng<ctrl>n<ns> (for example /dev/ng0n1), "
                    f"got {self.device_path!r}"
                )

        # io_uring_cmd resolves zero as an opt-in device-limit probe. Regular
        # io_uring keeps zero as its historic unsplit behavior, but honors an
        # explicitly configured transfer ceiling.
        if self.use_uring_cmd or config.max_data_transfer_size > 0:
            self.max_data_transfer_size = self._resolve_max_data_transfer_size(
                config.max_data_transfer_size
            )

        try:
            self.meta_magic_text = self.meta_magic.decode("ascii")
        except UnicodeDecodeError as e:
            raise ValueError("meta_magic must be ASCII bytes") from e

        self._meta_copy_count: int = 2
        self._meta_container_bytes: int = (
            (self.meta_total_bytes // self._meta_copy_count) // self.block_align
        ) * self.block_align
        if self._meta_container_bytes <= self.block_align:
            raise ValueError(
                "meta_total_bytes must provide room for at least two metadata copies"
            )

        self._lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()
        self._index: dict[str, _Entry] = {}
        self._lock_refcnt: dict[str, int] = {}
        self._inflight: dict[str, _Inflight] = {}

        self._next_slot: int = 0
        self._free_slots: dict[int, None] = {}
        self._quarantined_slots = {}
        # Four different questions, kept apart because one number cannot
        # answer them. Logical is what callers handed over; deduplicated is
        # what was already on the device and so was never written again;
        # submitted is the physical length given to the ring, which includes
        # block padding; completed is what the device reported finishing.
        # Reporting any one of these as "bytes written" overstates or
        # understates a different one.
        self._bytes_deduplicated = 0
        self._io_ledger_instance = RawBlockIoLedger()
        self._free_slots_by_placement_id: dict[int, dict[int, None]] = {}
        self._slot_placement_ids: dict[int, int] = {}
        self._fdp_slot_affinity_hit_count: int = 0
        self._fdp_slot_affinity_fallback_count: int = 0
        self._max_slots: int = 0
        self._effective_capacity_bytes: int = 0
        self._data_base_offset: int = 0

        self._raw = None
        self._closed = False

        self._meta_seq: int = 0
        self._meta_dirty_total: int = 0
        self._meta_persisted: int = 0
        self._inflight_io_count: int = 0
        self._last_io_ts: float = time.monotonic()
        self._meta_stop_evt = threading.Event()
        # A core that did not load the device's index must still number its
        # checkpoints after whatever is already on the device: readers adopt
        # only a higher sequence number, so a writer restarting at 0 would be
        # ignored until it caught up with its previous life.  Resolved at the
        # first checkpoint write so opening and storing do no extra reads.
        self._meta_seq_resume_pending = not bool(config.load_checkpoint_on_init)
        self._meta_thread: Optional[threading.Thread] = None

        try:
            self._ensure_capacity_and_layout()
            if self.load_checkpoint_on_init:
                self._load_checkpoint_from_device()
            else:
                logger.info("RawBlockCore: skipping on-device metadata checkpoint load")
                # Whether to *use* the index is a performance choice; whether
                # this engine may write into this namespace at all is not.
                # Skipping the load would otherwise open a namespace somebody
                # else is writing, under a derivation whose keys this engine
                # cannot read, as a fresh empty cache.
                self._require_namespace_authority()

            if self.meta_enable_periodic and self.role == "writer":
                self._meta_thread = threading.Thread(
                    target=self._checkpoint_loop,
                    daemon=True,
                    name="raw-block-core-checkpoint",
                )
                self._meta_thread.start()
        except Exception:
            self._cleanup_after_init_failure()
            raise

    @property
    def _requires_transfer_alignment(self) -> bool:
        """Return whether I/O transfers require block alignment.

        Returns:
            True when transfers must be aligned to ``self.block_align``.
            This is required for O_DIRECT I/O and for io_uring_cmd operations.
        """
        return self.use_odirect or self.use_uring_cmd

    def _resolve_max_data_transfer_size(self, configured_size: int) -> int:
        """Resolve transfer split size from config or NVMe sysfs queue limits.

        When auto-detecting, the size is bounded by both the device's
        ``max_hw_sectors_kb`` (total transfer size) and its ``max_segments``
        scatter-gather limit. The NVMe ``io_uring_cmd`` passthrough path builds
        one iovec segment per page, so a single transfer can consume up to
        ``ceil(len / page_size)`` segments. Exceeding ``max_segments`` makes the
        kernel reject the command with ``EINVAL``, so the resolved size is
        capped at ``max_segments * page_size``.

        Args:
            configured_size: Explicitly configured max data transfer size in bytes.
                If > 0, this value is used directly. If <= 0, the size is
                auto-detected from device queue limits.

        Returns:
            The resolved max data transfer size in bytes, guaranteed to be
            a multiple of ``self.block_align``.

        Raises:
            ValueError: If ``configured_size`` is > 0 but not a multiple of
                ``self.block_align``.
            RuntimeError: If sysfs queue limits cannot be resolved during
                auto-detection.
        """
        if configured_size > 0:
            if configured_size % self.block_align != 0:
                raise ValueError(
                    f"max_data_transfer_size ({configured_size}) must be a "
                    f"multiple of block_align ({self.block_align})"
                )
            return configured_size

        queue_dir = _resolve_sysfs_queue_dir(self.device_path)
        if queue_dir is None:
            raise RuntimeError(
                "RustRawBlockBackend: unable to derive NVMe sysfs queue path from "
                "NVMe character device path "
                f"{self.device_path} for auto max_data_transfer_size"
            )

        max_hw_sectors_kb = _read_sysfs_int(f"{queue_dir}/max_hw_sectors_kb")
        if max_hw_sectors_kb is None or max_hw_sectors_kb <= 0:
            raise RuntimeError(
                "RustRawBlockBackend: failed to read max_hw_sectors_kb from "
                f"{queue_dir} for auto max_data_transfer_size"
            )

        resolved_bytes = max_hw_sectors_kb * 1024

        # The io_uring_cmd passthrough path builds one iovec segment per page,
        # so cap the transfer at the device's scatter-gather segment limit.
        max_segments = _read_sysfs_int(f"{queue_dir}/max_segments")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if max_segments is not None and max_segments > 0:
            segment_limit_bytes = max_segments * page_size
            resolved_bytes = min(resolved_bytes, segment_limit_bytes)

        aligned_bytes = (resolved_bytes // self.block_align) * self.block_align
        if aligned_bytes <= 0:
            aligned_bytes = self.block_align

        logger.info(
            "RustRawBlockBackend: auto max_data_transfer_size=%d bytes "
            "(device=%s, max_hw_sectors_kb=%s, max_segments=%s, page_size=%d)",
            aligned_bytes,
            self.device_path,
            max_hw_sectors_kb,
            max_segments,
            page_size,
        )
        return aligned_bytes

    def _rawdev(self):
        """Return the lazily opened Rust raw-block device binding."""
        if self._terminal:
            # Close has made its decision about this device, and that
            # decision is what the caller's own teardown acted on -- freeing
            # the memory behind it, or deliberately keeping it. Handing out a
            # device here would either open a fresh one that knows nothing
            # about the old one, or hand back a handle that was retained
            # precisely so nobody would touch it again.
            raise RuntimeError(
                f"raw-block device {self.device_path} is closed; it will not "
                "be reopened for work that arrived afterwards"
            )
        if self._raw is None and self._poisoned:
            # Opening a fresh device here would answer every question about
            # quiescence with a confident "nothing outstanding", which is
            # exactly wrong after this core stopped being able to say. Being
            # merely closed is not the same thing and keeps its old
            # behaviour.
            raise RuntimeError(
                f"raw-block device {self.device_path} could not establish "
                "what it was doing; it will not be reopened"
            )
        if self._raw is None:
            try:
                # Third Party
                from lmcache_rust_raw_block_io import RawBlockDevice  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Rust raw-block extension is not installed. "
                    "Install / build `rust_raw_block_io` and retry."
                ) from e
            self._raw = RawBlockDevice(
                self.device_path,
                writable=self.role == "writer",
                use_odirect=self.use_odirect,
                alignment=self.block_align,
                io_engine=self.io_engine,
                iouring_queue_depth=self.iouring_queue_depth,
                use_uring_cmd=self.use_uring_cmd,
            )
            if self.require_native_idle_capability and not hasattr(
                self._raw, "is_idle"
            ):
                # This lane publishes an index for another engine to read,
                # and what makes that safe is being able to ask whether the
                # device has answered for everything it was handed. A build
                # that cannot be asked can only answer the weaker health
                # question, and accepting that silently would qualify this
                # engine on a question it never asked.
                raise RuntimeError(
                    f"raw-block device {self.device_path} is backed by a "
                    "native engine that cannot report whether anything is "
                    "outstanding; rebuild the rust_raw_block_io extension "
                    "from this tree"
                )
        return self._raw

    def raw_device(self) -> Any:
        """Return the lazily opened Rust raw-block device.

        Returns:
            The underlying Rust ``RawBlockDevice`` object.

        Raises:
            Exception: Propagates raw-device open errors from the Rust binding.
        """
        return self._rawdev()

    def fetch_fdp_status(self) -> list[tuple[int, int]]:
        """Fetch NVMe FDP placement/RUH status from the raw device.

        Returns:
            List of ``(placement_id, ruh_id)`` tuples.

        Raises:
            RuntimeError: If the raw device binding or target device cannot
                provide FDP status.
        """
        return [
            (int(pid), int(ruhid)) for pid, ruhid in self._rawdev().fetch_fdp_status()
        ]

    def set_raw_device_for_testing(self, raw_device: Any) -> None:
        """Replace the raw device handle used by this core.

        Args:
            raw_device: Object implementing the Rust raw-device methods.
        """
        self._raw = raw_device

    def register_fixed_buffers_from_allocator(self, memory_allocator: Any) -> None:
        """Register allocator pages with io_uring when the allocator exposes them.

        Args:
            memory_allocator: Local CPU allocator that may expose
                ``get_paged_buffers()``.

        Raises:
            Exception: Propagates Rust registration errors after logging.
        """
        if self.io_engine != "io_uring":
            return
        paged_buffers = getattr(memory_allocator, "get_paged_buffers", None)
        if not callable(paged_buffers):
            if self.require_dmabuf_registration:
                raise RuntimeError(
                    "strict dma-buf mode requires an allocator with paged buffers"
                )
            logger.warning(
                "RawBlockCore: allocator does not expose paged buffers; "
                "io_uring fixed-buffer zero-copy is disabled"
            )
            return
        buffers = paged_buffers()
        if not buffers:
            if self.require_dmabuf_registration:
                raise RuntimeError(
                    "strict dma-buf mode requires non-empty paged buffers"
                )
            logger.warning(
                "RawBlockCore: allocator returned no paged buffers; "
                "io_uring fixed-buffer zero-copy is disabled"
            )
            return
        buffer_ptrs = [buf.data_ptr() for buf in buffers]
        buffer_sizes = [buf.numel() * buf.element_size() for buf in buffers]
        # A dma-buf backed allocator exposes, per paged buffer, the dma-buf fd
        # exporting its memory and the address that dma-buf is mapped at.
        # Registering those maps the memory to the device once and lets each
        # fixed read or write be one command up to the device's dma-buf
        # ceiling; without it every command is DMA-mapped on its own and, on a
        # translating IOMMU, clamped at 128 KiB.  Fall back to the classic
        # registration when the kernel or the device refuses.
        regions = getattr(memory_allocator, "get_paged_dmabuf_regions", None)
        regions = regions() if callable(regions) else None
        if regions and not self.use_uring_cmd:
            try:
                self._rawdev().register_fixed_dmabufs(
                    buffer_ptrs,
                    buffer_sizes,
                    [fd for fd, _base in regions],
                    [base for _fd, base in regions],
                )
                logger.info(
                    "RawBlockCore: registered %d paged buffers as dma-buf "
                    "fixed buffers (%d dma-buf(s)) for io_uring map-once I/O",
                    len(buffers),
                    len({fd for fd, _base in regions}),
                )
                self._buffer_registration_mode = "dmabuf"
                return
            except Exception as exc:
                if self.require_dmabuf_registration:
                    raise RuntimeError(
                        "strict dma-buf fixed-buffer registration failed"
                    ) from exc
                logger.warning(
                    "RawBlockCore: dma-buf fixed-buffer registration refused "
                    "(%s); falling back to per-command mapping",
                    exc,
                )
        if self.require_dmabuf_registration:
            raise RuntimeError(
                "strict dma-buf mode requires dma-buf regions for every paged buffer"
            )
        self._rawdev().register_fixed_buffers(buffer_ptrs, buffer_sizes)
        self._buffer_registration_mode = "classic"
        logger.info(
            "RawBlockCore: registered %d paged buffers for io_uring fixed I/O",
            len(buffers),
        )

    def contains_key(self, encoded_key: str, *, lock: bool = False) -> bool:
        """Return whether one encoded key is present in the raw-block index.

        Args:
            encoded_key: Encoded raw-block key string.
            lock: If true, increment the key's L2 lock refcount on hit.

        Returns:
            True when the key is indexed and available for load.
        """
        return self.exists_many([encoded_key], lock=lock)[0]

    def exists_inflight(self, encoded_key: str) -> bool:
        """Return whether a key currently has an in-flight write.

        Args:
            encoded_key: Encoded raw-block key string.

        Returns:
            True when the key is being written but not committed yet.
        """
        with self._lock:
            return encoded_key in self._inflight

    def get_metadata_many(
        self, encoded_keys: Sequence[str]
    ) -> list[DiskCacheMetadata | None]:
        """Return metadata for encoded keys without loading payload bytes.

        Args:
            encoded_keys: Ordered encoded raw-block keys to inspect.

        Returns:
            A metadata-or-None list aligned with ``encoded_keys``.
        """
        with self._lock:
            metas: list[DiskCacheMetadata | None] = []
            for encoded_key in encoded_keys:
                entry = self._index.get(encoded_key)
                metas.append(entry.meta if entry is not None else None)
            return metas

    def get_metadata_prefix(
        self,
        encoded_keys: Sequence[str],
        *,
        lock: bool = False,
        skip_locked: set[str] | None = None,
    ) -> list[DiskCacheMetadata]:
        """Return leading-hit metadata and optionally lock those entries.

        Args:
            encoded_keys: Ordered encoded raw-block keys to inspect.
            lock: If true, increment L2 lock refcounts for every returned
                metadata entry while holding the index lock.
            skip_locked: Encoded keys that are already protected by the caller
                and should not receive an additional lock refcount.

        Returns:
            Metadata for the contiguous leading hit prefix. The returned list
            stops at the first missing key.
        """
        with self._lock:
            metas: list[DiskCacheMetadata] = []
            for encoded_key in encoded_keys:
                entry = self._index.get(encoded_key)
                if entry is None:
                    break
                metas.append(entry.meta)
                if lock and (skip_locked is None or encoded_key not in skip_locked):
                    self._lock_refcnt[encoded_key] = (
                        self._lock_refcnt.get(encoded_key, 0) + 1
                    )
            return metas

    def first_encoded_key(self) -> str | None:
        """Return one indexed encoded key for diagnostics.

        Returns:
            The first indexed key according to dictionary iteration order, or
            None if the recovered/indexed metadata is empty.
        """
        with self._lock:
            return next(iter(self._index), None)

    def lock_refcount(self, encoded_key: str) -> int:
        """Return the L2 lock refcount for an encoded key.

        Args:
            encoded_key: Encoded raw-block key string.

        Returns:
            Current lock refcount, or zero when the key is unlocked or absent.
        """
        with self._lock:
            return int(self._lock_refcnt.get(encoded_key, 0))

    def inflight_io_count(self) -> int:
        """Return the number of currently active raw-device I/O operations."""
        with self._lock:
            return int(self._inflight_io_count)

    def indexed_key_count(self) -> int:
        """Return the number of entries currently present in the key index."""
        with self._lock:
            return len(self._index)

    def snapshot_indexed_keys(self) -> list[str]:
        """Return a detached snapshot of encoded keys currently in the index."""
        with self._lock:
            return list(self._index.keys())

    def entry_offset(self, encoded_key: str) -> int | None:
        """Return the raw-device slot offset for an indexed key.

        Args:
            encoded_key: Encoded raw-block key string.

        Returns:
            Slot offset in bytes, or None when the key is not indexed.
        """
        with self._lock:
            entry = self._index.get(encoded_key)
            return None if entry is None else int(entry.offset)

    def metadata_container_offsets(self) -> list[int]:
        """Return checkpoint metadata container offsets in bytes."""
        return self._meta_container_offsets()

    def data_base_offset(self) -> int:
        """Return the byte offset where raw-block data slots begin."""
        return int(self._data_base_offset)

    def put_many(
        self,
        keys: Sequence[RawBlockKeySpec],
        objs: Sequence[MemoryObj],
        placement_ids: Sequence[PlacementId] | None = None,
        *,
        io_context: Optional[RawBlockIoContext] = None,
    ) -> RawBlockPutManyResult:
        """Persist a batch of memory objects into raw-block slots.

        With ``io_engine='io_uring'`` and more than one key, writes are
        submitted in batches and failure is no longer independent per key: a
        device write failure rolls back every key submitted in the same batch.
        Batches are bounded, so a request larger than one batch can leave
        earlier batches committed while a later one rolls back.

        Args:
            keys: Ordered raw-block key specs corresponding to ``objs``.
            objs: Memory objects whose byte buffers should be written.
            placement_ids: Optional per-key FDP placement identifiers for
                raw-block writes. ``None`` omits the directive; explicit identifier
                0 is rejected because default writes already use that mapping.

        Returns:
            Per-key success results and newly stored encoded keys. A key is
            reported as failed when no free raw-block slot is available or when
            its payload does not fit a slot. Slot reclamation is owned by the
            adapter/controller calling ``delete_many``.

        Raises:
            ValueError: If either sequence is empty, sequence lengths do not
                match, or a placement identifier is 0.
        """
        if self.role == "reader":
            raise RuntimeError(
                "RawBlockCore: refusing to store on a reader core; "
                "only the writer owns the device"
            )
        if not keys or not objs:
            raise ValueError("keys and objs must be non-empty")
        if len(keys) != len(objs):
            raise ValueError("keys and objs must have the same length")
        per_key_placement_ids = normalize_raw_block_placement_ids(
            placement_ids,
            len(keys),
            field_name="placement_ids",
        )

        if self.io_engine == "io_uring" and len(keys) > 1:
            return self._put_many_batch_io(
                keys, objs, per_key_placement_ids, io_context=io_context
            )

        results = [False] * len(keys)
        stored_keys: list[str] = []

        for i, (key, obj) in enumerate(zip(keys, objs, strict=False)):
            placement_id = per_key_placement_ids[i]
            if self._closed or self._poisoned:
                break

            with self._lock:
                if key.encoded in self._index:
                    # Already here: a hit, not a write. Counting these as
                    # bytes stored would report device traffic that never
                    # happened.
                    self._bytes_deduplicated += int(self._index[key.encoded].size)
                    results[i] = True
                    continue
                if key.encoded in self._inflight:
                    continue

                # Establish the length first: an unsupported representation
                # must not leave a reserved slot behind.
                try:
                    payload_len = _logical_payload_len(obj)
                except UnsupportedDevicePayload as exc:
                    logger.warning(
                        "RawBlockCore: refusing key %s: %s", key.encoded, exc
                    )
                    results[i] = False
                    continue

                try:
                    offset = self._allocate_slot_locked(placement_id)
                except RuntimeError:
                    logger.warning(
                        "RawBlockCore: no free slot available for key %s",
                        key.encoded,
                    )
                    continue

                meta = DiskCacheMetadata(
                    path=f"{self.device_path}@{offset}",
                    size=payload_len,
                    shape=obj.metadata.shape,
                    dtype=obj.metadata.dtype,
                    cached_positions=obj.metadata.cached_positions,
                    fmt=obj.metadata.fmt,
                    pin_count=0,
                )
                self._inflight[key.encoded] = _Inflight(offset=offset, meta=meta)

            success = self._write_one(
                key, obj, offset, placement_id=placement_id, io_context=io_context
            )

            with self._lock:
                inflight = self._inflight.pop(key.encoded, None)
                if inflight is None:
                    results[i] = False
                    continue
                if inflight.canceled or not success:
                    self._release_submitted_slot_locked(
                        self._offset_to_slot(int(inflight.offset))
                    )
                    self._meta_dirty_total += 1
                    results[i] = False
                    continue

                self._index[key.encoded] = _Entry(
                    offset=inflight.offset,
                    size=inflight.meta.size,
                    meta=inflight.meta,
                )
                self._meta_dirty_total += 1
                results[i] = True
                stored_keys.append(key.encoded)

        return RawBlockPutManyResult(
            results=results,
            stored_keys=stored_keys,
        )

    def exists_many(
        self,
        encoded_keys: Sequence[str],
        *,
        lock: bool = False,
    ) -> list[bool]:
        """Return a full hit bitmap as booleans for encoded keys.

        Args:
            encoded_keys: Ordered encoded raw-block keys to check.
            lock: If true, increment L2 lock refcounts for every hit.

        Returns:
            A list of booleans aligned with ``encoded_keys``.
        """
        results: list[bool] = []
        with self._lock:
            for encoded_key in encoded_keys:
                found = encoded_key in self._index
                results.append(found)
                if found and lock:
                    self._lock_refcnt[encoded_key] = (
                        self._lock_refcnt.get(encoded_key, 0) + 1
                    )
        return results

    def load_many_into(
        self,
        encoded_keys: Sequence[str],
        objs: Sequence[MemoryObj],
        *,
        io_context: Optional[RawBlockIoContext] = None,
    ) -> list[bool]:
        """Load raw-block payloads into caller-provided memory objects.

        Args:
            encoded_keys: Ordered encoded raw-block keys to load.
            objs: Destination memory objects. Buffers must remain valid until
                this method returns.

        Returns:
            A list of per-key load success booleans aligned with
            ``encoded_keys``.

        Raises:
            ValueError: If either sequence is empty or the sequence lengths do
                not match.
        """
        if not encoded_keys or not objs:
            raise ValueError("encoded_keys and objs must be non-empty")
        if len(encoded_keys) != len(objs):
            raise ValueError("encoded_keys and objs must have the same length")
        if self._poisoned:
            # Both write paths refuse here; a read has the same problem. It
            # would hand the device fresh destination buffers on an engine
            # that has already stopped being able to say when it is finished
            # with the ones it has.
            return [False] * len(encoded_keys)

        with self._lock:
            items = [
                (encoded_key, self._index.get(encoded_key))
                for encoded_key in encoded_keys
            ]
            self._inflight_io_count += 1

        results = [False] * len(encoded_keys)
        try:
            read_indices: list[int] = []
            read_offsets: list[int] = []
            read_buffers: list[Any] = []
            read_payload_lens: list[int] = []
            read_total_lens: list[int] = []
            read_kinds: list[str] = []

            for i, (encoded_key, entry) in enumerate(items):
                if entry is None:
                    continue
                try:
                    payload_len = int(entry.size)
                    total_len = (
                        round_up(payload_len, self.block_align)
                        if self._requires_transfer_alignment
                        else payload_len
                    )
                    dev_buf = _device_payload_tensor(objs[i])
                    if dev_buf is not None:
                        # A device object is read straight into its slot; the
                        # allocator's page-aligned slots leave room for the
                        # O_DIRECT tail, so the whole aligned length lands in
                        # place and nothing is bounced through the host.
                        buf = dev_buf
                        direct_view = dev_buf if dev_buf.nbytes >= total_len else None
                    else:
                        buf = memoryview(objs[i].byte_array)
                        try:
                            buf = buf.cast("B")
                        except Exception:
                            pass

                        direct_view = self._build_direct_odirect_view(
                            memory_obj=objs[i],
                            payload_len=payload_len,
                            total_len=total_len,
                            buffer_len=len(buf),
                            zero_tail=False,
                        )
                    if direct_view is not None:
                        read_buffer = direct_view
                        read_payload_len = (
                            total_len if len(direct_view) >= total_len else payload_len
                        )
                    else:
                        read_buffer = buf
                        read_payload_len = payload_len

                    read_indices.append(i)
                    read_offsets.append(entry.offset + self.header_bytes)
                    read_buffers.append(read_buffer)
                    read_payload_lens.append(read_payload_len)
                    read_total_lens.append(total_len)
                    read_kinds.append(IO_KIND_PAYLOAD)
                except Exception as e:
                    logger.error("RawBlockCore load failed for %s: %s", encoded_key, e)

            if read_indices:
                # With header verification each payload read is paired with a
                # read of its slot header in the same batch; the header must
                # still name this key with this size, or the writer reused
                # the slot after we adopted its index and the load is a miss.
                header_bufs: list[bytearray] = []
                if self.verify_slot_header_on_load:
                    for item_idx in read_indices:
                        entry = items[item_idx][1]
                        assert entry is not None
                        hdr = bytearray(self.header_bytes)
                        header_bufs.append(hdr)
                        read_offsets.append(entry.offset)
                        read_buffers.append(hdr)
                        read_payload_lens.append(self.header_bytes)
                        read_total_lens.append(self.header_bytes)
                        read_kinds.append(IO_KIND_SLOT_HEADER)
                try:
                    io_results = self._read_buffers(
                        read_offsets,
                        read_buffers,
                        read_payload_lens,
                        read_total_lens,
                        read_kinds,
                        io_context=io_context,
                    )
                except Exception as e:
                    logger.error("RawBlockCore batched load failed: %s", e)
                    io_results = [False] * len(read_offsets)
                if header_bufs:
                    n = len(read_indices)
                    payload_ok = list(io_results[:n])
                    header_ok = list(io_results[n:])
                    for pos, item_idx in enumerate(read_indices):
                        if not payload_ok[pos] or not header_ok[pos]:
                            payload_ok[pos] = False
                            continue
                        encoded_key, entry = items[item_idx]
                        assert entry is not None
                        decoded = self._decode_slot_header(bytes(header_bufs[pos]))
                        expected = slot_identity_from_encoded_key(
                            encoded_key, self.key_namespace
                        )
                        if decoded is None or decoded != (expected, int(entry.size)):
                            logger.warning(
                                "RawBlockCore: slot header for %s no longer matches "
                                "(slot reused by the writer); treating as a miss",
                                encoded_key,
                            )
                            payload_ok[pos] = False
                    io_results = payload_ok

                for item_idx, ok in zip(read_indices, io_results, strict=True):
                    if not ok:
                        continue
                    entry = items[item_idx][1]
                    if entry is None:
                        continue
                    objs[
                        item_idx
                    ].metadata.cached_positions = entry.meta.cached_positions
                    results[item_idx] = True
        finally:
            with self._lock:
                self._inflight_io_count -= 1
                self._last_io_ts = time.monotonic()
        return results

    def unlock_many(self, encoded_keys: Sequence[str]) -> None:
        """Release L2 lock references for encoded keys.

        Args:
            encoded_keys: Encoded raw-block keys whose lock refcounts should be
                decremented. Missing keys and underflow are treated as no-ops.
        """
        with self._lock:
            for encoded_key in encoded_keys:
                refcnt = self._lock_refcnt.get(encoded_key, 0)
                if refcnt <= 1:
                    self._lock_refcnt.pop(encoded_key, None)
                else:
                    self._lock_refcnt[encoded_key] = refcnt - 1

    def delete_many(
        self,
        encoded_keys: Sequence[str],
        *,
        force: bool = False,
    ) -> list[bool]:
        """Delete indexed keys and recycle their slots when allowed.

        Args:
            encoded_keys: Ordered encoded raw-block keys to delete.
            force: If true, delete locked keys as well. Normal MP eviction uses
                false so locked entries are preserved.

        Returns:
            A list of per-key deletion booleans aligned with ``encoded_keys``.
        """
        if self.role == "reader":
            raise RuntimeError(
                "RawBlockCore: refusing to delete on a reader core; "
                "only the writer owns the device"
            )
        deleted: list[bool] = []
        with self._lock:
            for encoded_key in encoded_keys:
                entry = self._index.get(encoded_key)
                locked = self._lock_refcnt.get(encoded_key, 0) > 0
                if entry is not None and locked and not force:
                    deleted.append(False)
                    continue

                removed_entry = self._index.pop(encoded_key, None)
                inflight = self._inflight.get(encoded_key)
                if inflight is not None:
                    inflight.canceled = True
                self._lock_refcnt.pop(encoded_key, None)
                if removed_entry is not None:
                    self._release_submitted_slot_locked(
                        self._offset_to_slot(int(removed_entry.offset))
                    )
                    self._meta_dirty_total += 1
                deleted.append(removed_entry is not None or inflight is not None)
        return deleted

    def usage(self) -> tuple[float, float]:
        """Return current raw-block slot usage fractions.

        Returns:
            ``(current_usage, projected_usage)``. Raw-block has no separate
            projected value, so both values are identical. ``(-1.0, -1.0)``
            indicates that usable capacity is unknown.
        """
        with self._lock:
            usable_capacity = self._max_slots * self.slot_bytes
            if usable_capacity <= 0:
                return (-1.0, -1.0)
            # A withheld extent is occupied, not free. Leaving it out of
            # both terms would report capacity this engine will never hand
            # out again as available.
            used_slots = (
                len(self._index)
                + len(self._inflight)
                + len(self._quarantined_slots or {})
            )
            usage = (used_slots * self.slot_bytes) / usable_capacity
            return (usage, usage)

    def checkpoint_now(self) -> None:
        """Synchronously write a metadata checkpoint."""
        self._checkpoint_once(force=True)

    def publish_index(self) -> bool:
        """Write the index checkpoint so a reader core can adopt it.

        A writer calls this after a put batch completes.  The checkpoint is
        the whole index serialized as JSON (about 200 bytes per entry) mirrored
        into the metadata area, so back-to-back calls are spaced by
        ``publish_min_interval_ms``.  Returns True when a checkpoint was
        written.
        """
        if self.role != "writer":
            return False
        with self._checkpoint_lock:
            now = time.monotonic()
            if (now - self._last_publish_ts) * 1000.0 < self.publish_min_interval_ms:
                return False
            written = self._checkpoint_once_locked(force=True)
            if written:
                self._last_publish_ts = now
            return written

    @property
    def writer_epoch(self) -> str:
        """This engine's producer incarnation.

        A writer mints one when it is constructed and keeps it for life; a
        reader adopts the epoch of the checkpoint it loaded, which names
        somebody else's publication and so is not an identity this engine
        can be addressed by. Empty where there is none.
        """
        return self._writer_epoch if self.role == "writer" else ""

    def publish_request(
        self, encoded_keys: Sequence[str]
    ) -> RawBlockPublicationReceipt:
        """Publish and identify a checkpoint containing all request keys.

        Args:
            encoded_keys: Ordered keys that form the request manifest.

        Returns:
            A receipt that a reader can match against the same ordered keys.

        Raises:
            RuntimeError: If called on a reader, a key is absent, or the
                checkpoint cannot be written.
        """
        if self.role != "writer":
            raise RuntimeError("only a writer core can publish a request")
        if self._closed:
            # A publication queued before shutdown and started after it
            # would reopen a device the caller's teardown has already
            # accounted for, and name extents in an index written by a core
            # nobody is watching any more.
            raise RuntimeError(
                "raw-block core cannot publish: it is shutting down and "
                "stopped admitting publications"
            )
        if self._poisoned:
            # A receipt tells a reader where to look. This core can no longer
            # say what the device is doing with the extents it would name.
            raise RuntimeError(
                "raw-block core cannot publish: the native engine could not "
                "establish what the device is still doing"
            )
        if not encoded_keys:
            raise ValueError("request publication requires at least one key")
        with self._checkpoint_lock:
            if (
                self._manifest_digest_from_records(
                    encoded_keys,
                    self._manifest_from_index(),
                )
                is None
            ):
                raise RuntimeError("request publication contains an uncommitted key")
            # Write a fresh generation for every request rather than reuse one
            # that already names these keys. Membership does not mean the
            # mapping is the same: a key can be published at one extent,
            # deleted, and written again somewhere else, and the earlier
            # checkpoint still advertises the extent it had then. A reader
            # adopting that receipt would be sent to storage the writer has
            # since given to something else, and the holds this request takes
            # protect where the key is now, not where the old checkpoint says
            # it was.
            #
            # The cost falls only on a request whose keys were all already
            # published, since any new key forced a checkpoint anyway. The
            # cheaper alternative is to record the extent each key occupied
            # when it was published and reuse the generation only while every
            # requested key still sits where it did, which is a physical
            # identity check and deliberately not part of the content digest.
            # That needs keeping the extra map correct at every site that
            # republishes, and a single missed site restores this bug
            # silently, so it is not what this milestone does.
            if not self._checkpoint_once_locked(force=True, rewrite_clean=True):
                raise RuntimeError("failed to publish the request checkpoint")
            if not set(encoded_keys).issubset(self._published_keys):
                raise RuntimeError("published checkpoint is missing request keys")
            digest = self._published_manifest_digest(encoded_keys)
            if digest is None:
                raise RuntimeError(
                    "published checkpoint is missing request manifest metadata"
                )
            if self._derivation is None:
                raise IncompatibleKeyDerivation(
                    "raw-block storage P/D publishes keys for another engine "
                    "to read, so it must state how they were derived; "
                    "configure the key derivation descriptor"
                )
            if not namespace_identity_is_shareable(self.namespace_identity):
                raise ValueError(
                    "raw-block storage P/D requires a persistent block "
                    "namespace identity; this device exposes none, so "
                    "configure rust_raw_block.namespace_identity with an "
                    "identity that names the same namespace on every node "
                    "that shares it"
                )
            self._last_publish_ts = time.monotonic()
            manifest_records = [
                self._published_manifest[encoded_key] for encoded_key in encoded_keys
            ]
            total_logical_bytes = sum(
                int(record["size"]) for record in manifest_records
            )
            return RawBlockPublicationReceipt(
                writer_epoch=self._writer_epoch,
                checkpoint_seq=self._meta_seq,
                key_count=len(encoded_keys),
                manifest_digest=digest,
                namespace_identity=self.namespace_identity,
                total_logical_bytes=total_logical_bytes,
                total_padded_bytes=sum(
                    round_up(int(record["size"]), self.block_align)
                    for record in manifest_records
                ),
            )

    def publication_matches(
        self,
        receipt: RawBlockPublicationReceipt,
        encoded_keys: Sequence[str],
    ) -> bool:
        """Return whether the adopted checkpoint contains a request receipt."""
        if len(encoded_keys) != receipt.key_count:
            return False
        if receipt.namespace_identity != self.namespace_identity:
            return False
        if self._writer_epoch != receipt.writer_epoch:
            return False
        if self._meta_seq < receipt.checkpoint_seq:
            return False
        digest = self._published_manifest_digest(encoded_keys)
        return digest == receipt.manifest_digest

    def refresh_until_publication(
        self,
        receipt: RawBlockPublicationReceipt,
        encoded_keys: Sequence[str],
        *,
        timeout_ms: int,
        refresh_interval_ms: int,
    ) -> bool:
        """Wait until a reader adopts the advertised or a compatible checkpoint."""
        if self.role != "reader":
            raise RuntimeError("only a reader core can adopt a publication receipt")
        deadline = time.monotonic() + max(timeout_ms, 0) / 1000.0
        while True:
            if self.publication_matches(receipt, encoded_keys):
                return True
            self.refresh_index_from_device()
            if self.publication_matches(receipt, encoded_keys):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(remaining, max(refresh_interval_ms, 1) / 1000.0))

    def _max_checkpoint_seq_on_device(self) -> int:
        """Highest checkpoint sequence number in any valid header, or 0."""
        best = 0
        for offset in self._meta_container_offsets():
            header = self._read_meta_header(offset)
            if header is not None:
                best = max(best, int(header["seq"]))
        return best

    def refresh_index_from_device(self) -> bool:
        """Adopt the newest on-device index checkpoint if it is newer than ours.

        Only the checkpoint headers are read until a newer sequence number
        shows up; then the payload is loaded and replaces the index wholesale,
        so entries the writer dropped disappear here too.  Per-slot header
        validation is skipped: with ``verify_slot_header_on_load`` each load
        checks the slot it reads instead.  Returns True when the index changed.
        """
        best: Optional[dict[str, int]] = None
        for offset in self._meta_container_offsets():
            header = self._read_meta_header(offset)
            if header is None:
                continue
            if best is None or int(header["seq"]) > int(best["seq"]):
                best = header
        if best is None or int(best["seq"]) <= self._meta_seq:
            return False
        payload = self._load_meta_payload(best)
        if payload is None:
            return False
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            logger.warning("RawBlockCore: failed to decode refreshed metadata payload")
            return False
        if not self._apply_loaded_state(data, verify=False):
            logger.warning("RawBlockCore: refreshed metadata payload rejected")
            return False
        self._meta_seq = int(best["seq"])
        self._published_keys = frozenset(self._index)
        self._published_manifest = self._manifest_from_index()
        self._published_writer_epoch = self._writer_epoch
        logger.debug(
            "RawBlockCore adopted checkpoint seq=%d entries=%d",
            self._meta_seq,
            len(self._index),
        )
        return True

    def apply_loaded_state(self, data: dict[str, Any]) -> bool:
        """Validate and apply a recovered metadata checkpoint payload.

        Args:
            data: Decoded checkpoint dictionary.

        Returns:
            True when the payload shape and layout match this core and all
            valid entries were applied. Invalid per-entry records are skipped.
        """
        return self._apply_loaded_state(data)

    def _require_namespace_authority(self) -> None:
        """Refuse a populated namespace this engine is not entitled to open.

        Only asked where sharing is declared, and asked independently of
        whether the index is loaded: a namespace that already holds another
        engine's publication is not an empty cache, whatever this engine
        intends to do with the index.

        An unreadable or absent checkpoint is not an answer either way, so
        it is left to the ordinary path: a device with nothing legible on it
        is the one case that can follow normal initialization.
        """
        if self._derivation is None:
            return
        try:
            header, payload = self._select_latest_checkpoint()
        except Exception:
            logger.warning(
                "RawBlockCore could not read %s to establish whether it is "
                "already in use",
                self.device_path,
            )
            return
        if header is None or payload is None:
            return
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            logger.warning(
                "RawBlockCore could not decode the checkpoint on %s to "
                "establish whether it is already in use",
                self.device_path,
            )
            return
        if not isinstance(data, dict):
            return
        self._require_matching_key_namespace(data.get("key_namespace"))
        self._require_compatible_derivation(data.get("derivation"))

    def is_poisoned(self) -> bool:
        """Whether this core has stopped being able to say what the device did.

        Sticky for the life of the engine. Callers use it to decide whether a
        resource may be handed back or has to be withheld; it is read without
        the lock on purpose, because a caller that sees a stale False will be
        refused by the guarded path it goes on to take.
        """
        return self._poisoned

    @property
    def _io_attribution(self) -> "RawBlockIoAttribution":
        """This core's request attribution, created on first use."""
        record = self._io_attribution_instance
        if record is None:
            with RawBlockCore._io_ledger_creation_lock:
                record = self._io_attribution_instance
                if record is None:
                    record = RawBlockIoAttribution()
                    self._io_attribution_instance = record
        return record

    def _collect_native_io_journal(self) -> None:
        """Take what the device recorded and attribute it.

        Drained rather than read, so no row is counted twice, and drained
        wherever this engine is already waiting on the device: the native
        record is bounded, and leaving it to fill means losing the oldest
        rows of the very request being measured.
        """
        # Asked of the device this core has, if it has one. A core built
        # without an initializer -- as several tests do -- has none, and the
        # accessor would open one just to be asked for a journal.
        take = getattr(getattr(self, "_raw", None), "take_io_journal", None)
        if take is None:
            return
        try:
            rows, dropped = take()
        except Exception:
            logger.warning("RawBlockCore could not read the native I/O journal")
            return
        if rows or dropped:
            self._io_attribution.record(rows, dropped)

    @property
    def _io_ledger(self) -> "RawBlockIoLedger":
        """This core's I/O ledger, created on first use."""
        ledger = self._io_ledger_instance
        if ledger is None:
            with RawBlockCore._io_ledger_creation_lock:
                ledger = self._io_ledger_instance
                if ledger is None:
                    ledger = RawBlockIoLedger()
                    self._io_ledger_instance = ledger
        return ledger

    def report_status(self) -> dict:
        """Return raw-block health, layout, metadata, and I/O accounting.

        The payload rows are summed outside this core's lock because the
        ledger has its own; taking both in one order here and the other way
        anywhere else is how a deadlock is built.
        """
        self._collect_native_io_journal()
        payload_writes = self._io_ledger.totals(direction="write", kind=IO_KIND_PAYLOAD)
        payload_reads = self._io_ledger.totals(direction="read", kind=IO_KIND_PAYLOAD)
        with self._lock:
            return {
                "is_healthy": not self._closed and not self._poisoned,
                "poisoned": self._poisoned,
                "type": "RawBlockCore",
                "key_namespace": self.key_namespace,
                "device_path": self.device_path,
                "block_align": self.block_align,
                "header_bytes": self.header_bytes,
                "slot_bytes": self.slot_bytes,
                "meta_total_bytes": self.meta_total_bytes,
                "usable_capacity_bytes": self._max_slots * self.slot_bytes,
                "indexed_key_count": len(self._index),
                "inflight_key_count": len(self._inflight),
                "locked_key_count": sum(
                    1 for refcnt in self._lock_refcnt.values() if refcnt > 0
                ),
                "free_slot_count": len(self._free_slots),
                "quarantined_slot_count": len(self._quarantined_slots or {}),
                "bytes_deduplicated": self._bytes_deduplicated,
                "payload_writes": payload_writes.as_payload(),
                "payload_reads": payload_reads.as_payload(),
                "io_by_kind_and_path": self._io_ledger.snapshot(),
                "io_by_request": self._io_attribution.as_payload(),
                "next_slot": self._next_slot,
                "max_slots": self._max_slots,
                "metadata_seq": self._meta_seq,
                "writer_epoch": self._writer_epoch,
                "metadata_dirty_total": self._meta_dirty_total,
                "metadata_persisted": self._meta_persisted,
                "inflight_io_count": self._inflight_io_count,
                "use_odirect": self.use_odirect,
                "enable_zero_copy": self.enable_zero_copy,
                "io_engine": self.io_engine,
                "iouring_queue_depth": self.iouring_queue_depth,
                "use_uring_cmd": self.use_uring_cmd,
                "buffer_registration_mode": self._buffer_registration_mode,
                "require_dmabuf_registration": self.require_dmabuf_registration,
                "fdp_slot_affinity_enabled": self.fdp_slot_affinity_enabled,
                "fdp_slot_affinity_hit_count": (self._fdp_slot_affinity_hit_count),
                "fdp_slot_affinity_fallback_count": (
                    self._fdp_slot_affinity_fallback_count
                ),
            }

    def close(self) -> RawBlockCloseOutcome:
        """Stop checkpointing, write a final checkpoint, and close the device.

        Returns what was established, because the caller's next decisions --
        whether to free the memory behind this device, whether to release
        what a reader may still be reading -- cannot be made from a return
        of None. A repeated close reports the first result rather than
        inventing a fresh one: it must not reach the device accessor, which
        opens a new device that knows nothing about the old one.
        """
        with self._lock:
            if self._closed:
                return self._close_outcome or RawBlockCloseOutcome(
                    quiescence=NativeQuiescence.NOT_ATTEMPTED,
                    poisoned=self._poisoned,
                    reason="a close was already in progress or complete",
                    quarantined_slots=len(self._quarantined_slots or {}),
                )
            self._closed = True

        self._meta_stop_evt.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=5)
            self._meta_thread = None

        # Asked of the device this core actually has: the accessor below
        # reopens a fresh one when the reference is dropped, and a fresh
        # device knows nothing about what the old one could not establish.
        #
        # And answered by the device, not by the absence of a complaint:
        # a writer about to publish its last index needs to know that
        # everything it handed over has been answered for.
        unknown = self._poisoned
        if not unknown and self._raw is not None:
            unknown = not device_has_nothing_outstanding(self._raw)
        self._poisoned = unknown

        checkpointed = False
        if self.role == "writer" and not self.close_writes_final_checkpoint:
            # Nothing to do and nothing lost: every published request wrote
            # its own forced checkpoint when it was published, so the last
            # generation on the device already names every request that
            # completed. A close-time checkpoint here could only add a
            # generation naming a request that did not.
            logger.info(
                "RawBlockCore %s: writing no close-time checkpoint; each "
                "published request already has its own generation.",
                self.device_path,
            )
        elif self.role == "writer" and not unknown:
            # A writer that cannot say what the device holds must not
            # publish an index naming it. publish_request already refuses
            # while running, and shutdown is not an exemption.
            try:
                # The helper's own answer: it returns false for state that
                # needed no checkpoint, and reporting that as "written"
                # describes a generation that does not exist.
                checkpointed = bool(self._checkpoint_once(force=True))
            except Exception as e:
                logger.warning("RawBlockCore final checkpoint failed: %s", e)
        elif self.role == "writer":
            logger.error(
                "RawBlockCore %s: skipping the final checkpoint because this "
                "writer cannot establish what the device holds. The last "
                "published index stands; these extents are not advertised.",
                self.device_path,
            )

        def settle(quiescence: NativeQuiescence, reason: str) -> RawBlockCloseOutcome:
            outcome = RawBlockCloseOutcome(
                quiescence=quiescence,
                poisoned=self._poisoned,
                reason=reason,
                quarantined_slots=len(self._quarantined_slots or {}),
                final_checkpoint_written=checkpointed,
            )
            self._close_outcome = outcome
            # From here nothing may open or be handed this device again.
            self._terminal = True
            if outcome.final_checkpoint_written and not (
                outcome.final_checkpoint_is_vouched_for
            ):
                logger.error(
                    "RawBlockCore %s published a final index and then could "
                    "not prove the device was finished (%s). That index is "
                    "durable and may name an extent whose write was never "
                    "observed; a later incarnation reading it will serve "
                    "those bytes as a hit.",
                    self.device_path,
                    reason,
                )
            return outcome

        if self._raw is None:
            return settle(
                NativeQuiescence.NOT_ATTEMPTED,
                "no native device was open",
            )
        if unknown:
            # Dropping this reference runs the native destructor, which frees
            # the very owners the engine retained. Keep it bound: the handle,
            # its registration and the buffers behind it stay alive for the
            # life of the process, which is the price of not knowing when the
            # device finished.
            logger.error(
                "RawBlockCore %s: retaining the native device, its buffer "
                "registration and %d withheld extent(s) because an outcome "
                "could not be established.",
                self.device_path,
                len(self._quarantined_slots or {}),
            )
            return settle(
                NativeQuiescence.RETAINED,
                "the engine could not establish what the device was doing",
            )
        try:
            self._raw.close()
        except Exception as e:
            # The native close refuses when it cannot prove quiescence. That
            # is a report, not noise: keep the reference and stay unhealthy.
            self._poisoned = True
            logger.error(
                "Failed to close raw block device %s: %s. Retaining the "
                "native device and everything it holds.",
                self.device_path,
                e,
            )
            return settle(
                NativeQuiescence.RETAINED,
                f"the native close was refused: {e}",
            )
        self._raw = None
        return settle(NativeQuiescence.PROVEN, "the native engine closed")

    def _cleanup_after_init_failure(self) -> None:
        """Close resources that may have been opened before init failed."""
        self._meta_stop_evt.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=5)
            self._meta_thread = None
        if self._raw is not None:
            try:
                self._raw.close()
            except Exception as e:
                logger.warning(
                    "Failed to close raw block device %s: %s", self.device_path, e
                )
            finally:
                self._raw = None
        self._closed = True
        self._terminal = True

    def _byte_view(self, buf: Any) -> memoryview:
        """Return a byte-addressable memoryview over a Python buffer.

        Args:
            buf: Object implementing the Python buffer protocol.

        Returns:
            A memoryview with one-byte elements.

        Raises:
            TypeError: If ``buf`` does not expose a compatible contiguous buffer.
        """
        view = buf if isinstance(buf, memoryview) else memoryview(buf)
        if view.itemsize == 1 and view.format in ("B", "b", "c"):
            return view
        return view.cast("B")

    def _is_buffer_aligned(self, buf: Any) -> bool:
        """Check if a buffer is aligned to the block alignment boundary.

        Args:
            buf: Object implementing the Python buffer protocol.

        Returns:
            True if the buffer is aligned, False otherwise.
        """
        if not self.use_odirect:
            return True
        view = self._byte_view(buf)
        # Check if the buffer pointer is aligned
        ptr = ctypes.addressof((ctypes.c_byte * 1).from_buffer(view))
        return ptr % self.block_align == 0

    def _allocate_aligned_buffer(self, length: int) -> memoryview:
        """Allocate a writable byte buffer aligned to ``self.block_align``.

        Args:
            length: Number of bytes to expose through the returned memoryview.

        Returns:
            A memoryview whose starting address is aligned to ``self.block_align``.

        Raises:
            ValueError: If ``length`` is negative.
        """
        if length < 0:
            raise ValueError("length must be >= 0")
        if length == 0:
            return memoryview(bytearray())

        backing = bytearray(length + self.block_align - 1)
        ptr = ctypes.addressof((ctypes.c_byte * 1).from_buffer(backing))
        offset = (-ptr) % self.block_align
        return memoryview(backing)[offset : offset + length]

    def _build_direct_odirect_view(
        self,
        memory_obj: MemoryObj,
        payload_len: int,
        total_len: int,
        buffer_len: int,
        *,
        zero_tail: bool,
    ) -> Optional[memoryview]:
        """Build an aligned memoryview for direct O_DIRECT I/O when possible.

        Args:
            memory_obj: Memory object whose backing allocation may be aligned.
            payload_len: Logical payload length in bytes.
            total_len: I/O length after any O_DIRECT padding.
            buffer_len: Available buffer length in bytes.
            zero_tail: Whether to zero any padded tail bytes before writing.

        Returns:
            A direct memoryview over the allocation, or None when the memory
            object is unsuitable for direct I/O.
        """
        if not self.use_odirect or not self.enable_zero_copy:
            return None

        ptr_val = getattr(memory_obj, "data_ptr", None)
        if callable(ptr_val):
            try:
                ptr_val = ptr_val()
            except Exception:
                ptr_val = None
        if ptr_val is None:
            return None
        if buffer_len <= 0:
            return None

        ptr = int(ptr_val)
        if ptr <= 0 or ptr % self.block_align != 0:
            return None
        if buffer_len < payload_len:
            return None

        view_len = min(buffer_len, total_len)
        if view_len < payload_len:
            return None

        try:
            raw = (ctypes.c_ubyte * view_len).from_address(ptr)
            view = memoryview(raw)
            if zero_tail and total_len > payload_len and view_len >= total_len:
                ctypes.memset(ptr + payload_len, 0, total_len - payload_len)
            return view
        except Exception:
            return None

    def _payload_fits_slot(self, payload_len: int) -> bool:
        """Return whether a logical payload can fit in one raw-block slot."""
        payload_capacity = self.slot_bytes - self.header_bytes
        if payload_len > payload_capacity:
            return False
        if self._requires_transfer_alignment:
            return round_up(payload_len, self.block_align) <= payload_capacity
        return True

    def _prepare_write_payload(self, memory_obj: MemoryObj) -> tuple[Any, int, int]:
        """Prepare the payload buffer and lengths for a raw-block write.

        Args:
            memory_obj: Source object to persist.

        Returns:
            A tuple of ``(buffer, payload_len, total_len)`` where ``total_len``
            includes any O_DIRECT padding.

        Raises:
            RuntimeError: If the aligned payload would exceed slot capacity.
        """
        dev_buf = _device_payload_tensor(memory_obj)
        if dev_buf is not None:
            return self._prepare_device_write_payload(memory_obj, dev_buf)
        buf = memory_obj.byte_array
        if hasattr(buf, "cast"):
            buf = buf.cast("B")
        payload_len = len(memory_obj.byte_array)
        payload_capacity = self.slot_bytes - self.header_bytes
        if payload_len > payload_capacity:
            raise RuntimeError(
                f"RawBlockCore payload {payload_len} exceeds slot capacity "
                f"{payload_capacity}"
            )
        total_len = payload_len
        if self._requires_transfer_alignment:
            total_len = round_up(payload_len, self.block_align)
            if total_len > payload_capacity:
                raise RuntimeError(
                    f"Aligned payload {total_len} exceeds slot capacity "
                    f"{payload_capacity}"
                )
            direct_view = self._build_direct_odirect_view(
                memory_obj=memory_obj,
                payload_len=payload_len,
                total_len=total_len,
                buffer_len=len(buf),
                zero_tail=True,
            )
            if direct_view is not None:
                buf = direct_view
        return buf, payload_len, total_len

    def _prepare_device_write_payload(
        self, memory_obj: MemoryObj, dev_buf: torch.Tensor
    ) -> tuple[Any, int, int]:
        """Prepare a raw-block write straight out of a device memory object.

        The engine writes from the object's physical slot, so the O_DIRECT
        tail past the logical payload must fit inside that slot.  The tail is
        not zeroed: it carries whatever the slot held before, and a load only
        ever consumes the logical payload, so nothing leaves the process that
        the process did not already own.

        Raises:
            RuntimeError: If the payload or its aligned length exceeds the
                slot, or the device slot is too small for the aligned length.
        """
        payload_len = _logical_payload_len(memory_obj)
        payload_capacity = self.slot_bytes - self.header_bytes
        if payload_len > payload_capacity:
            raise RuntimeError(
                f"RawBlockCore payload {payload_len} exceeds slot capacity "
                f"{payload_capacity}"
            )
        total_len = payload_len
        if self._requires_transfer_alignment:
            total_len = round_up(payload_len, self.block_align)
            if total_len > payload_capacity:
                raise RuntimeError(
                    f"Aligned payload {total_len} exceeds slot capacity "
                    f"{payload_capacity}"
                )
            if dev_buf.nbytes < total_len:
                raw = getattr(memory_obj, "raw_data", None)
                meta = getattr(memory_obj, "metadata", None)
                raise RuntimeError(
                    f"RawBlockCore: the object's GPU slot holds "
                    f"{dev_buf.nbytes} bytes but its payload needs "
                    f"{total_len} aligned ({payload_len} logical); the slot "
                    "the allocator carved is smaller than the size the object "
                    "reports, so the allocator's paging shape and the object's "
                    "shape disagree "
                    f"[object={type(memory_obj).__name__} "
                    f"raw={tuple(raw.shape) if raw is not None else None} "
                    f"meta_shape={getattr(meta, 'shape', None)} "
                    f"fmt={getattr(meta, 'fmt', None)}]"
                )
        return dev_buf, payload_len, total_len

    def _validate_io_uring_chunk(self, offset: int, total_len: int) -> None:
        """Validate one bounded io_uring transfer range.

        Args:
            offset: Device byte offset for the transfer.
            total_len: Transfer size in bytes.

        Raises:
            ValueError: If either value is not block aligned.
        """
        if not self._requires_transfer_alignment:
            return
        if offset % self.block_align != 0:
            raise ValueError("io_uring requires aligned offsets")
        if total_len % self.block_align != 0:
            raise ValueError("io_uring requires aligned transfer lengths")

    def _write_bounded_io_uring_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        placement_ids: Sequence[PlacementId] | None = None,
        kinds: Sequence[str] | None = None,
        *,
        io_context: Optional[RawBlockIoContext] = None,
    ) -> None:
        """Write buffers as chunks bounded by ``max_data_transfer_size``.

        Args:
            offsets: Device offsets for each logical write.
            buffers: Source buffers.
            payload_lens: Logical source byte counts.
            total_lens: Physical transfer sizes, including padding.
            placement_ids: Optional FDP placement identifiers for each logical
                write. ``None`` omits the directive; explicit identifier 0 is
                rejected.

        Raises:
            ValueError: If lengths are inconsistent or unaligned.
            Exception: Propagates Rust raw-device write errors.
        """
        raw_dev = self._rawdev()
        chunk_offsets: list[int] = []
        chunk_buffers: list[memoryview] = []
        chunk_lens: list[int] = []
        chunk_placement_ids: list[PlacementId] = []
        # One entry per chunk, so a completion bitmap that is per chunk can
        # still be attributed to the logical write's kind.
        chunk_kinds: list[str] = []
        keepalive: list[memoryview] = []
        per_write_placement_ids = normalize_raw_block_placement_ids(
            placement_ids,
            len(offsets),
            field_name="placement_ids",
        )
        write_kinds = self._normalize_io_kinds(kinds, len(offsets))
        self._io_ledger.logical_request(
            "write",
            IO_PATH_IOURING_BOUNDED,
            write_kinds,
            payload_lens,
            total_lens,
        )

        for offset, buf, payload_len, total_len, placement_id, kind in zip(
            offsets,
            buffers,
            payload_lens,
            total_lens,
            per_write_placement_ids,
            write_kinds,
            strict=True,
        ):
            offset = int(offset)
            payload_len = int(payload_len)
            total_len = int(total_len)
            self._validate_io_uring_chunk(offset, total_len)

            view = self._byte_view(buf)
            if len(view) < total_len:
                if len(view) < payload_len:
                    raise ValueError("input buffer shorter than payload_len")
                padded = self._allocate_aligned_buffer(total_len)
                padded[:payload_len] = view[:payload_len]
                view = padded
            else:
                view = view[:total_len]
            keepalive.append(view)

            cursor = 0
            while cursor < total_len:
                chunk_len = min(self.max_data_transfer_size, total_len - cursor)
                self._validate_io_uring_chunk(offset + cursor, chunk_len)
                chunk_offsets.append(offset + cursor)
                chunk_buffers.append(view[cursor : cursor + chunk_len])
                chunk_lens.append(chunk_len)
                chunk_placement_ids.append(placement_id)
                chunk_kinds.append(kind)
                cursor += chunk_len

        if not chunk_offsets:
            return
        self._io_ledger.submitted(
            "write", IO_PATH_IOURING_BOUNDED, chunk_kinds, chunk_lens
        )
        batch_id = raw_dev.batched_write(
            chunk_offsets,
            chunk_buffers,
            chunk_lens,
            chunk_placement_ids,
            request_tag=_io_tag(io_context),
        )
        completed = self._wait_iouring_results(
            raw_dev,
            batch_id,
            len(chunk_offsets),
            "bounded io_uring write",
        )
        self._io_ledger.completed(
            "write",
            IO_PATH_IOURING_BOUNDED,
            chunk_kinds,
            chunk_lens,
            completed,
        )
        if not all(completed):
            raise RuntimeError("raw-block bounded io_uring write failed")
        keepalive.clear()

    def _read_bounded_io_uring_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        kinds: Sequence[str] | None = None,
        *,
        io_context: Optional[RawBlockIoContext] = None,
    ) -> list[bool]:
        """Read buffers as bounded NVMe raw-command chunks.

        Args:
            offsets: Device offsets for each logical read.
            buffers: Destination buffers.
            payload_lens: Logical bytes to expose to callers.
            total_lens: Physical transfer sizes, including padding.

        Returns:
            A list of per-logical-read success booleans aligned with
            ``offsets``. If a submitted batch returns too few or too many
            completions, all submitted logical reads are reported as false.
        """
        raw_dev = self._rawdev()
        results = [False] * len(offsets)
        chunk_offsets: list[int] = []
        chunk_buffers: list[memoryview] = []
        chunk_lens: list[int] = []
        chunk_logical_indices: list[int] = []
        chunk_statuses: list[list[bool]] = [[] for _ in offsets]
        copy_back_targets: dict[int, tuple[memoryview, memoryview, int]] = {}
        keepalive: list[Any] = []
        read_kinds = self._normalize_io_kinds(kinds, len(offsets))
        self._io_ledger.logical_request(
            "read",
            IO_PATH_IOURING_BOUNDED,
            read_kinds,
            payload_lens,
            total_lens,
        )

        for logical_idx, (offset, buf, payload_len, total_len) in enumerate(
            zip(offsets, buffers, payload_lens, total_lens, strict=True)
        ):
            try:
                offset = int(offset)
                payload_len = int(payload_len)
                total_len = int(total_len)
                self._validate_io_uring_chunk(offset, total_len)

                dst = self._byte_view(buf)
                if len(dst) < total_len:
                    if len(dst) < payload_len:
                        raise ValueError("output buffer shorter than payload_len")
                    target = self._allocate_aligned_buffer(total_len)
                    copy_back = True
                else:
                    target = dst[:total_len]
                    copy_back = False
                keepalive.append(target)

                cursor = 0
                max_chunk_len = (
                    self.max_data_transfer_size
                    if self.max_data_transfer_size > 0
                    else total_len
                )
                while cursor < total_len:
                    chunk_len = min(max_chunk_len, total_len - cursor)
                    self._validate_io_uring_chunk(offset + cursor, chunk_len)
                    chunk_offsets.append(offset + cursor)
                    chunk_buffers.append(target[cursor : cursor + chunk_len])
                    chunk_lens.append(chunk_len)
                    chunk_logical_indices.append(logical_idx)
                    cursor += chunk_len

                if copy_back:
                    copy_back_targets[logical_idx] = (dst, target, payload_len)
            except Exception:
                continue

        if not chunk_offsets:
            return results

        chunk_kinds = [read_kinds[idx] for idx in chunk_logical_indices]
        self._io_ledger.submitted(
            "read", IO_PATH_IOURING_BOUNDED, chunk_kinds, chunk_lens
        )
        try:
            batch_id = raw_dev.batched_read(
                chunk_offsets,
                chunk_buffers,
                chunk_lens,
                request_tag=_io_tag(io_context),
            )
            chunk_results = self._wait_iouring_results(
                raw_dev,
                batch_id,
                len(chunk_offsets),
                "bounded io_uring read",
            )
        except Exception:
            return results

        self._io_ledger.completed(
            "read",
            IO_PATH_IOURING_BOUNDED,
            chunk_kinds,
            chunk_lens,
            chunk_results,
        )
        for chunk_idx, logical_idx in enumerate(chunk_logical_indices):
            ok = chunk_idx < len(chunk_results) and bool(chunk_results[chunk_idx])
            chunk_statuses[logical_idx].append(ok)

        for logical_idx, statuses in enumerate(chunk_statuses):
            if not statuses or not all(statuses):
                continue
            if logical_idx in copy_back_targets:
                dst, target, payload_len = copy_back_targets[logical_idx]
                dst[:payload_len] = target[:payload_len]
            results[logical_idx] = True

        keepalive.clear()
        return results

    def _write_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        placement_ids: Sequence[PlacementId] | None = None,
        kinds: Sequence[str] | None = None,
        *,
        io_context: Optional[RawBlockIoContext] = None,
    ) -> None:
        """Write one or more buffers through the configured Rust I/O path.

        Args:
            offsets: Device offsets for each write.
            buffers: Python buffers to write.
            payload_lens: Logical payload lengths for each buffer.
            total_lens: Physical I/O lengths for each buffer.
            placement_ids: Optional FDP placement identifiers for raw-block writes.
                ``None`` omits the directive; explicit identifier 0 is rejected.

        Raises:
            RuntimeError: If the requested io_uring mode is unavailable.
            Exception: Propagates Rust raw-device write errors.
        """
        raw_dev = self._rawdev()
        per_write_placement_ids = normalize_raw_block_placement_ids(
            placement_ids,
            len(offsets),
            field_name="placement_ids",
        )
        write_kinds = self._normalize_io_kinds(kinds, len(offsets))

        if self.io_engine != "io_uring":
            # Recorded against the route that is about to run, so the row
            # names what carried the bytes. A synchronous write returning
            # is its completion: there is no separate reported outcome.
            self._io_ledger.logical_request(
                "write", IO_PATH_SYNC, write_kinds, payload_lens, total_lens
            )
            self._io_ledger.submitted("write", IO_PATH_SYNC, write_kinds, total_lens)
            for offset, buf, payload_len, total_len, kind in zip(
                offsets, buffers, payload_lens, total_lens, write_kinds, strict=True
            ):
                raw_dev.pwrite_from_buffer(offset, buf, payload_len, total_len)
                self._io_ledger.completed(
                    "write", IO_PATH_SYNC, [kind], [total_len], [True]
                )
            return

        if self.max_data_transfer_size > 0:
            self._write_bounded_io_uring_buffers(
                offsets,
                buffers,
                payload_lens,
                total_lens,
                per_write_placement_ids,
                write_kinds,
                io_context=io_context,
            )
            return

        can_batch = all(
            int(payload_len) == int(total_len)
            for payload_len, total_len in zip(payload_lens, total_lens, strict=True)
        )
        # batched_write carries a single length per entry, so it cannot express
        # O_DIRECT padding where payload_len < total_len. Fall back to
        # write_uring, which takes both lengths and lets Rust build the aligned
        # padded transfer.
        if can_batch:
            self._io_ledger.logical_request(
                "write",
                IO_PATH_IOURING_BATCHED,
                write_kinds,
                payload_lens,
                total_lens,
            )
            self._io_ledger.submitted(
                "write", IO_PATH_IOURING_BATCHED, write_kinds, total_lens
            )
            batch_id = raw_dev.batched_write(
                [int(offset) for offset in offsets],
                list(buffers),
                [int(total_len) for total_len in total_lens],
                per_write_placement_ids,
                request_tag=_io_tag(io_context),
            )
            completed = self._wait_iouring_results(
                raw_dev,
                batch_id,
                len(offsets),
                "io_uring write",
            )
            self._io_ledger.completed(
                "write",
                IO_PATH_IOURING_BATCHED,
                write_kinds,
                total_lens,
                completed,
            )
            if not all(completed):
                raise RuntimeError("raw-block io_uring write failed")
            return

        # The per-write path returns its own completion error and never
        # reaches wait_iouring, so this is the only place it can adopt the
        # worker's verdict. Without it a padded O_DIRECT write that the
        # worker quarantined leaves the core healthy, and the extent it rolls
        # back is handed to the very next request.
        self._io_ledger.logical_request(
            "write",
            IO_PATH_IOURING_PER_WRITE,
            write_kinds,
            payload_lens,
            total_lens,
        )
        self._io_ledger.submitted(
            "write", IO_PATH_IOURING_PER_WRITE, write_kinds, total_lens
        )
        for offset, buf, payload_len, total_len, placement_id, kind in zip(
            offsets,
            buffers,
            payload_lens,
            total_lens,
            per_write_placement_ids,
            write_kinds,
            strict=True,
        ):
            try:
                raw_dev.write_uring(
                    int(offset),
                    buf,
                    int(payload_len),
                    int(total_len),
                    placement_id,
                    request_tag=_io_tag(io_context),
                )
            except BaseException:
                self._adopt_native_poison(raw_dev, "io_uring write")
                raise
            # This route waits inside the call and raises on failure, so
            # returning is the outcome. Leaving it uncounted reported zero
            # completed bytes for a write that had landed -- and this is
            # the route a padded O_DIRECT payload takes, which is most of
            # them.
            self._io_ledger.completed(
                "write", IO_PATH_IOURING_PER_WRITE, [kind], [total_len], [True]
            )

    def _read_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        kinds: Sequence[str] | None = None,
        *,
        io_context: Optional[RawBlockIoContext] = None,
    ) -> list[bool]:
        """Read one or more buffers through the configured Rust I/O path.

        Args:
            offsets: Device offsets for each read.
            buffers: Destination Python buffers.
            payload_lens: Logical payload lengths to expose to callers.
            total_lens: Physical I/O lengths for each read.

        Returns:
            A list of per-read success booleans aligned with ``offsets``. The
            returned list always has the same length as ``offsets``; completion
            count mismatches are reported as false entries.

        Raises:
            RuntimeError: If the requested io_uring mode is unavailable before
                reads are submitted.
        """
        raw_dev = self._rawdev()
        read_kinds = self._normalize_io_kinds(kinds, len(offsets))
        if self.io_engine != "io_uring":
            self._io_ledger.logical_request(
                "read", IO_PATH_SYNC, read_kinds, payload_lens, total_lens
            )
            self._io_ledger.submitted("read", IO_PATH_SYNC, read_kinds, total_lens)
            results: list[bool] = []
            for offset, buf, payload_len, total_len, kind in zip(
                offsets, buffers, payload_lens, total_lens, read_kinds, strict=True
            ):
                try:
                    raw_dev.pread_into(offset, buf, payload_len, total_len)
                    results.append(True)
                except Exception:
                    results.append(False)
                self._io_ledger.completed(
                    "read", IO_PATH_SYNC, [kind], [total_len], [results[-1]]
                )
            return results

        if self.max_data_transfer_size > 0:
            return self._read_bounded_io_uring_buffers(
                offsets,
                buffers,
                payload_lens,
                total_lens,
                read_kinds,
                io_context=io_context,
            )

        self._io_ledger.logical_request(
            "read",
            IO_PATH_IOURING_BATCHED,
            read_kinds,
            payload_lens,
            total_lens,
        )
        self._io_ledger.submitted(
            "read", IO_PATH_IOURING_BATCHED, read_kinds, total_lens
        )
        batch_id = raw_dev.batched_read(
            [int(offset) for offset in offsets],
            list(buffers),
            [int(total_len) for total_len in total_lens],
            request_tag=_io_tag(io_context),
        )
        results = self._wait_iouring_results(
            raw_dev,
            batch_id,
            len(offsets),
            "io_uring read",
        )
        self._io_ledger.completed(
            "read",
            IO_PATH_IOURING_BATCHED,
            read_kinds,
            total_lens,
            results,
        )
        return results

    @staticmethod
    def _normalize_io_kinds(
        kinds: Sequence[str] | None,
        count: int,
    ) -> list[str]:
        """Give every operation a kind, defaulting to payload.

        A caller that does not say is writing or reading cache payload --
        the header and checkpoint routes are the ones that have to be
        explicit, because they are the traffic that must not be added into
        a payload total.
        """
        if kinds is None:
            return [IO_KIND_PAYLOAD] * count
        classified = list(kinds)
        if len(classified) != count:
            raise ValueError("io kinds must align with the operations")
        return classified

    def _require_matching_key_namespace(self, theirs: Any) -> None:
        """Refuse a device whose entries were keyed in another namespace.

        A slot identity is derived from the encoded key and the namespace, so
        a core reading with a different namespace computes a different
        identity for every entry and header validation reads each one as
        stale. It then drops the entry and returns the extent -- and the next
        allocation lands on live data the other core's checkpoint still
        advertises. Measured on a temp-file device: the first allocation came
        back at exactly the offset the other core had just written.

        Ignoring the metadata instead is no safer: that leaves the slot
        counter at zero and allocates from the bottom of the same region.
        """
        if theirs is None:
            # Written before this field existed. Nothing can be concluded,
            # and the geometry checks above have already accepted it, so this
            # stays permissive rather than refusing every older device.
            return
        if str(theirs) == self.key_namespace:
            return
        raise IncompatibleKeyDerivation(
            f"raw-block device {self.device_path} holds entries keyed in "
            f"namespace {theirs!r} and this core keys in "
            f"{self.key_namespace!r}. Every header would read as stale and "
            "its extent would be recycled over live data; refusing instead."
        )

    def _require_compatible_derivation(self, payload: Any) -> None:
        """Refuse a namespace whose keys this engine cannot reproduce.

        Only the strict lane carries a descriptor. Where one is configured,
        an absent descriptor on the device is as incompatible as a differing
        one: it names a writer that made no statement about its derivation,
        so nothing can be concluded about the keys already there.
        """
        if self._derivation is None:
            return
        theirs = RawBlockDerivationDescriptor.from_payload(payload)
        if theirs is None:
            raise IncompatibleKeyDerivation(
                f"raw-block namespace {self.namespace_identity} carries no key "
                "derivation descriptor, and this engine requires one. Its "
                "existing keys cannot be shown to be readable here; refusing "
                "rather than adding ours beside them."
            )
        mismatches = self._derivation.describe_mismatch(theirs)
        if mismatches:
            raise IncompatibleKeyDerivation(
                f"raw-block namespace {self.namespace_identity} derives keys "
                "differently from this engine, so neither can read the "
                "other's: " + "; ".join(mismatches)
            )

    def _adopt_native_poison(
        self,
        raw_dev: Any,
        operation: str,
        batch_id: Optional[int] = None,
    ) -> bool:
        """Take on the worker's verdict that it cannot say what happened.

        A logical failure and an unknown outcome are different statements.
        The engine poisons itself when it cannot say what the device is
        doing, and it does so before signalling any completion, so asking
        after one is not a race. Adopting it here is what makes the core stop
        handing out storage and stop letting ordinary cleanup recycle an
        extent, because a failed result is not proof the device has finished.

        Every path that waits on the worker has to ask, not only the batched
        one: a caller that never asks leaves the core healthy while the
        worker has already given up, and an extent it rolls back becomes
        immediately re-allocatable.

        Returns whether the core is poisoned once this call is done.
        """
        if self._poisoned:
            return True
        if not device_says_outcome_is_unknown(raw_dev):
            return False
        self._poisoned = True
        logger.error(
            "RawBlockCore %s%s: the native engine could not establish what "
            "the device is still doing. Refusing further work on this device "
            "and withholding %d quarantined batch(es).",
            operation,
            f" batch {batch_id}" if batch_id is not None else "",
            int(getattr(raw_dev, "quarantined_batch_count", int)()),
        )
        return True

    def _wait_iouring_results(
        self,
        raw_dev: Any,
        batch_id: int,
        expected_count: int,
        operation: str,
    ) -> list[bool]:
        """Wait for an io_uring batch, log failures, and return its bitmap.

        ``expected_count`` is the number of individual I/O entries submitted
        in the Rust batch. For io_uring_cmd, this is the post-splitting chunk
        count, not the number of logical reads or writes.
        """
        results, completion_errors = raw_dev.wait_iouring(batch_id)
        results = list(results)
        # Here, because this is where the engine is already waiting on the
        # device. The native record is bounded, so leaving it to fill means
        # losing the oldest rows of the very request being measured.
        self._collect_native_io_journal()
        self._adopt_native_poison(raw_dev, operation, batch_id)
        for operation_index, error in completion_errors:
            logger.error(
                "RawBlockCore %s batch %d operation %d failed: %s",
                operation,
                batch_id,
                operation_index,
                error,
            )
        if len(results) != expected_count:
            logger.error(
                "RawBlockCore %s completion count mismatch: expected %d, got %d",
                operation,
                expected_count,
                len(results),
            )
            return [False] * expected_count
        return [bool(result) for result in results]

    def _write_one(
        self,
        key: RawBlockKeySpec,
        memory_obj: MemoryObj,
        offset: int,
        *,
        placement_id: PlacementId = None,
        io_context: Optional[RawBlockIoContext] = None,
    ) -> bool:
        """Write one object header and payload into a raw-block slot.

        Args:
            key: Raw-block key spec with the slot-header identity.
            memory_obj: Source object to write.
            offset: Slot byte offset on the raw device.
            placement_id: FDP placement identifier for this raw-block write.
                ``None`` omits the directive; explicit identifier 0 is rejected.

        Returns:
            True when both header and payload writes complete; false otherwise.
        """
        try:
            header = self._encode_header(
                key.slot_identity, _logical_payload_len(memory_obj)
            )
            buf, payload_len, total_len = self._prepare_write_payload(memory_obj)

            with self._lock:
                self._inflight_io_count += 1
            try:
                hdr_total = (
                    round_up(len(header), self.block_align)
                    if self._requires_transfer_alignment
                    else len(header)
                )
                header_buf: Any = header
                if self.io_engine != "io_uring" and len(header) < hdr_total:
                    padded_header = bytearray(header)
                    padded_header.extend(b"\x00" * (hdr_total - len(header)))
                    header_buf = padded_header
                # Keep each slot header on the same placement identifier as its
                # payload; future policy can split them if needed.
                self._write_buffers(
                    [offset, offset + self.header_bytes],
                    [header_buf, buf],
                    [
                        hdr_total if self.io_engine == "io_uring" else len(header),
                        payload_len,
                    ],
                    [hdr_total, total_len],
                    [placement_id, placement_id],
                    [IO_KIND_SLOT_HEADER, IO_KIND_PAYLOAD],
                    io_context=io_context,
                )
            finally:
                with self._lock:
                    self._inflight_io_count -= 1
                    self._last_io_ts = time.monotonic()
            return True
        except Exception as e:
            logger.error("RawBlockCore write failed for %s: %s", key.encoded, e)
            return False

    def _put_many_batch_io(
        self,
        keys: Sequence[RawBlockKeySpec],
        objs: Sequence[MemoryObj],
        placement_ids: Sequence[PlacementId],
        *,
        io_context: Optional[RawBlockIoContext] = None,
    ) -> RawBlockPutManyResult:
        """Persist objects using bounded io_uring batch submissions.

        Large ``put_many`` calls are split into chunks so one caller cannot
        monopolize the RawBlockCore lock while planning slots, and so the
        transient memory a single batch holds stays bounded.

        Each key contributes at least two write entries (header + payload).
        The io_uring_cmd path splits those further by
        ``max_data_transfer_size``, so one chunk can expand to many more
        entries there. Alignment and padding are handled by the existing write
        paths, which may allocate bounce buffers retained until I/O completes.

        Args:
            keys: Ordered raw-block key specs corresponding to ``objs``.
            objs: Memory objects whose byte buffers should be written.
            placement_ids: Normalized per-key FDP placement identifiers, one
                entry per key. ``None`` entries omit the directive.

        Returns:
            Per-key success results aligned with ``keys`` and the list of
            encoded keys that were newly committed to the index.

        Raises:
            ValueError: If ``keys``, ``objs``, and ``placement_ids`` do not all
                have the same length.
        """
        if len(keys) <= _MAX_PUT_MANY_IO_URING_BATCH_KEYS:
            return self._put_many_batch_io_chunk(
                keys, objs, placement_ids, io_context=io_context
            )

        results = [False] * len(keys)
        stored_keys: list[str] = []
        first_occurrences: dict[str, int] = {}
        unique_plan: list[tuple[int, RawBlockKeySpec, MemoryObj, PlacementId]] = []
        duplicate_indices: list[tuple[int, int]] = []

        # Deduplicate before chunking so duplicates that cross chunk boundaries
        # still inherit the first occurrence result without being rewritten.
        for i, (key, obj, placement_id) in enumerate(
            zip(keys, objs, placement_ids, strict=True)
        ):
            first_index = first_occurrences.get(key.encoded)
            if first_index is not None:
                duplicate_indices.append((i, first_index))
                continue
            first_occurrences[key.encoded] = i
            unique_plan.append((i, key, obj, placement_id))

        chunk_size = _MAX_PUT_MANY_IO_URING_BATCH_KEYS
        for start in range(0, len(unique_plan), chunk_size):
            chunk = unique_plan[start : start + chunk_size]
            chunk_result = self._put_many_batch_io_chunk(
                [key for _, key, _, _ in chunk],
                [obj for _, _, obj, _ in chunk],
                [placement_id for _, _, _, placement_id in chunk],
                io_context=io_context,
            )
            for local_i, (global_i, _key, _obj, _placement_id) in enumerate(chunk):
                results[global_i] = chunk_result.results[local_i]
            stored_keys.extend(chunk_result.stored_keys)

        for duplicate_i, first_i in duplicate_indices:
            results[duplicate_i] = results[first_i]

        return RawBlockPutManyResult(results=results, stored_keys=stored_keys)

    def _put_many_batch_io_chunk(
        self,
        keys: Sequence[RawBlockKeySpec],
        objs: Sequence[MemoryObj],
        placement_ids: Sequence[PlacementId],
        *,
        io_context: Optional[RawBlockIoContext] = None,
    ) -> RawBlockPutManyResult:
        """Persist one bounded chunk through a single ``_write_buffers`` call.

        Eligible new keys are submitted through one ``_write_buffers`` call so
        the io_uring path can batch those writes when their lengths allow it.

        Failures before submission are reported per key: already indexed keys
        report success without rewriting, duplicates of keys already reserved
        in this chunk share that key's final result, and keys with no free
        slot, payloads that cannot fit one slot, or buffer preparation errors
        fail individually.
        Once a combined device write is submitted, any write failure rolls back
        every submitted new key and commits none of them.

        Args:
            keys: Ordered raw-block key specs corresponding to ``objs``. Must be
                the same length as ``objs``.
            objs: Memory objects whose byte buffers should be written. Must be
                the same length as ``keys``.
            placement_ids: Normalized per-key FDP placement identifiers. Must be
                the same length as ``keys``. Each key's header and payload write
                inherit that key's identifier; ``None`` omits the directive.

        Returns:
            Per-key success results aligned with ``keys`` and the list of
            encoded keys that were newly committed to the index.

        Raises:
            ValueError: If ``keys``, ``objs``, and ``placement_ids`` do not all
                have the same length.
        """
        results = [False] * len(keys)
        stored_keys: list[str] = []
        write_plan: list[tuple[int, RawBlockKeySpec, MemoryObj, int, PlacementId]] = []
        planned_keys: set[str] = set()
        batch_duplicates: list[tuple[int, str]] = []

        # Reserve slots for eligible first-occurrence keys under the lock.
        # A reservation that is neither written nor freed costs its slot
        # for the life of the process, so give back everything this call
        # reserved if planning cannot finish.
        try:
            with self._lock:
                for i, (key, obj, placement_id) in enumerate(
                    zip(keys, objs, placement_ids, strict=True)
                ):
                    if self._closed or self._poisoned:
                        break
                    encoded_key = key.encoded
                    if encoded_key in self._index:
                        self._bytes_deduplicated += int(self._index[encoded_key].size)
                        results[i] = True
                        continue
                    if encoded_key in planned_keys:
                        batch_duplicates.append((i, encoded_key))
                        continue
                    if encoded_key in self._inflight:
                        continue
                    try:
                        payload_len = _logical_payload_len(obj)
                    except UnsupportedDevicePayload as exc:
                        logger.warning(
                            "RawBlockCore: refusing key %s: %s", encoded_key, exc
                        )
                        results[i] = False
                        continue
                    if not self._payload_fits_slot(payload_len):
                        logger.warning(
                            "RawBlockCore: payload for key %s does not fit slot",
                            encoded_key,
                        )
                        continue
                    try:
                        offset = self._allocate_slot_locked(placement_id)
                    except RuntimeError:
                        logger.warning(
                            "RawBlockCore: no free slot available for key %s",
                            key.encoded,
                        )
                        continue
                    meta = DiskCacheMetadata(
                        path=f"{self.device_path}@{offset}",
                        size=payload_len,
                        shape=obj.metadata.shape,
                        dtype=obj.metadata.dtype,
                        cached_positions=obj.metadata.cached_positions,
                        fmt=obj.metadata.fmt,
                        pin_count=0,
                    )
                    self._inflight[encoded_key] = _Inflight(offset=offset, meta=meta)
                    planned_keys.add(encoded_key)
                    write_plan.append((i, key, obj, offset, placement_id))
        except BaseException:
            with self._lock:
                for encoded_key in planned_keys:
                    inflight = self._inflight.pop(encoded_key, None)
                    if inflight is not None:
                        # The free list is keyed by slot index; an in-flight
                        # record carries the slot's byte offset. Passing the
                        # offset loses the slot for the life of the process:
                        # the range check drops it silently, and with other
                        # geometry it would name an unrelated slot.
                        self._append_free_slot_locked(
                            self._offset_to_slot(int(inflight.offset))
                        )
            raise

        if not write_plan:
            return RawBlockPutManyResult(results=results, stored_keys=stored_keys)

        # Build header/payload write entries outside the lock. Preparation
        # failures happen before device submission and are isolated per key.
        offsets: list[int] = []
        buffers: list[Any] = []
        payload_lens: list[int] = []
        total_lens: list[int] = []
        write_placement_ids: list[PlacementId] = []
        write_kinds: list[str] = []
        prepared_plan: list[tuple[int, RawBlockKeySpec, MemoryObj, int]] = []
        write_succeeded = True
        for i, key, obj, offset, placement_id in write_plan:
            try:
                header = self._encode_header(
                    key.slot_identity, _logical_payload_len(obj)
                )
                hdr_total = (
                    round_up(len(header), self.block_align)
                    if self._requires_transfer_alignment
                    else len(header)
                )
                buf, payload_len, total_len = self._prepare_write_payload(obj)
            except Exception as e:
                logger.error(
                    "RawBlockCore batch buffer preparation failed for %s: %s",
                    key.encoded,
                    e,
                )
                with self._lock:
                    inflight = self._inflight.pop(key.encoded, None)
                    if inflight is not None:
                        self._append_free_slot_locked(
                            self._offset_to_slot(int(inflight.offset))
                        )
                        self._meta_dirty_total += 1
                continue

            # Queue the key only once every fallible step has succeeded, so a
            # failed key never leaves a header behind for a slot that the
            # rollback above just returned to the free list.
            offsets.extend((offset, offset + self.header_bytes))
            buffers.extend((header, buf))
            payload_lens.extend((hdr_total, payload_len))
            total_lens.extend((hdr_total, total_len))
            write_placement_ids.extend((placement_id, placement_id))
            write_kinds.extend((IO_KIND_SLOT_HEADER, IO_KIND_PAYLOAD))
            prepared_plan.append((i, key, obj, offset))

        if prepared_plan:
            with self._lock:
                self._inflight_io_count += len(prepared_plan)
            try:
                self._write_buffers(
                    offsets,
                    buffers,
                    payload_lens,
                    total_lens,
                    write_placement_ids,
                    write_kinds,
                    io_context=io_context,
                )
            except Exception as e:
                write_succeeded = False
                logger.error("RawBlockCore batched write failed: %s", e)
            finally:
                with self._lock:
                    self._inflight_io_count -= len(prepared_plan)
                    self._last_io_ts = time.monotonic()

        # Commit successful writes, or roll back submitted keys if the device
        # write failed.
        with self._lock:
            for i, key, _obj, _offset in prepared_plan:
                inflight = self._inflight.pop(key.encoded, None)
                if inflight is None:
                    continue
                if not write_succeeded or inflight.canceled:
                    self._release_submitted_slot_locked(
                        self._offset_to_slot(int(inflight.offset))
                    )
                    self._meta_dirty_total += 1
                    continue
                self._index[key.encoded] = _Entry(
                    offset=inflight.offset,
                    size=inflight.meta.size,
                    meta=inflight.meta,
                )
                self._meta_dirty_total += 1
                results[i] = True
                stored_keys.append(key.encoded)
            for i, encoded_key in batch_duplicates:
                results[i] = encoded_key in self._index

        return RawBlockPutManyResult(results=results, stored_keys=stored_keys)

    def _encode_header(self, slot_identity: int, payload_len: int) -> bytes:
        """Encode a fixed-size raw-block slot header."""
        hdr = bytearray(self.header_bytes)
        hdr[0:8] = b"LMCBLK01"
        hdr[8:16] = int(slot_identity & ((1 << 64) - 1)).to_bytes(
            8,
            "little",
            signed=False,
        )
        hdr[16:24] = int(payload_len).to_bytes(8, "little", signed=False)
        return bytes(hdr)

    def _decode_slot_header(self, hdr: bytes) -> Optional[tuple[int, int]]:
        """Decode a raw-block slot header into identity and payload length."""
        if len(hdr) < 24 or hdr[0:8] != b"LMCBLK01":
            return None
        slot_identity = int.from_bytes(hdr[8:16], "little", signed=False)
        payload_len = int.from_bytes(hdr[16:24], "little", signed=False)
        return slot_identity, payload_len

    def _read_slot_header(self, offset: int) -> tuple[str, Optional[tuple[int, int]]]:
        """Read one slot header, saying which of three things happened.

        ``"decoded"`` with the identity and length; ``"invalid"`` when bytes
        arrived and are not one of ours; ``"unreadable"`` when the read did
        not complete.

        The third is not the same statement as the second, and collapsing
        them is what let an I/O outcome nobody could establish look like
        ordinary stale metadata -- which is recycled.
        """
        buf = bytearray(self.header_bytes)
        try:
            with self._lock:
                self._inflight_io_count += 1
            if not all(
                self._read_buffers(
                    [offset],
                    [buf],
                    [self.header_bytes],
                    [self.header_bytes],
                    [IO_KIND_SLOT_HEADER],
                    io_context=self._metadata_io_context,
                )
            ):
                return "unreadable", None
            decoded = self._decode_slot_header(buf)
            if decoded is None:
                return "invalid", None
            return "decoded", decoded
        except Exception:
            return "unreadable", None
        finally:
            with self._lock:
                self._inflight_io_count -= 1
                self._last_io_ts = time.monotonic()

    def _ensure_capacity_and_layout(self) -> None:
        """Open the device if needed and compute metadata/data layout."""
        if self._effective_capacity_bytes > 0 and self._max_slots > 0:
            return

        device_size = int(self._rawdev().size_bytes())
        requested = self.capacity_bytes if self.capacity_bytes > 0 else device_size
        self._effective_capacity_bytes = min(requested, device_size)
        self.capacity_bytes = self._effective_capacity_bytes

        if self.meta_total_bytes >= self._effective_capacity_bytes:
            raise RuntimeError("metadata region exceeds usable device capacity")

        self._data_base_offset = self.meta_total_bytes
        data_bytes = self._effective_capacity_bytes - self._data_base_offset
        self._max_slots = data_bytes // self.slot_bytes
        if self._max_slots <= 0:
            raise RuntimeError(
                "raw block capacity too small for slot size after metadata"
            )

    def _slot_to_offset(self, slot: int) -> int:
        """Convert a data-slot index to its byte offset."""
        return self._data_base_offset + slot * self.slot_bytes

    def _offset_to_slot(self, offset: int) -> int:
        """Convert a data-slot byte offset to its slot index."""
        return (offset - self._data_base_offset) // self.slot_bytes

    def _allocate_slot_locked(self, placement_id: PlacementId = None) -> int:
        """Allocate a slot offset while ``self._lock`` is held."""
        self._ensure_capacity_and_layout()

        if self.fdp_slot_affinity_enabled and placement_id is not None:
            affinity_slots = self._free_slots_by_placement_id.get(placement_id)
            if affinity_slots:
                slot, _ = affinity_slots.popitem()
                if not affinity_slots:
                    self._free_slots_by_placement_id.pop(placement_id, None)
                self._free_slots.pop(slot, None)
                self._fdp_slot_affinity_hit_count += 1
                self._set_slot_placement_id_locked(slot, placement_id)
                return self._slot_to_offset(slot)

        if self._free_slots:
            slot, _ = self._free_slots.popitem()
            self._remove_slot_from_affinity_pool_locked(slot)
            if self.fdp_slot_affinity_enabled and placement_id is not None:
                self._fdp_slot_affinity_fallback_count += 1
            self._set_slot_placement_id_locked(slot, placement_id)
            return self._slot_to_offset(slot)

        if self._next_slot < self._max_slots:
            slot = self._next_slot
            self._next_slot += 1
            self._set_slot_placement_id_locked(slot, placement_id)
            return self._slot_to_offset(slot)
        raise RuntimeError("No free slots available")

    def _append_free_slot_locked(self, slot: int) -> None:
        """Add a slot to the free list while ``self._lock`` is held."""
        if slot < 0 or slot >= self._max_slots:
            return
        if slot in self._free_slots:
            return
        self._free_slots[slot] = None
        if not self.fdp_slot_affinity_enabled:
            return
        placement_id = self._slot_placement_ids.get(slot)
        if placement_id is not None:
            self._free_slots_by_placement_id.setdefault(placement_id, {})[slot] = None

    def _release_submitted_slot_locked(self, slot: int) -> None:
        """Give back a slot the device was already told to write.

        Whether that is safe is exactly the question poison answers. A
        logical failure is not a DMA fence: if the worker could not
        establish what happened, a later request handed this slot would be
        given storage an earlier write may still be landing in.

        Only use this where the extent reached the submission ring. A slot
        reserved and then abandoned before submission was never the device's
        and goes straight back to the free list.
        """
        if self._poisoned:
            self._quarantine_slot_locked(slot)
            return
        self._append_free_slot_locked(slot)

    def _quarantine_slot_locked(self, slot: int) -> None:
        """Withhold a slot from allocation for this engine's lifetime.

        Nothing takes a slot out of quarantine. Reuse needs proof that the
        device is done with it, which this engine cannot obtain once its
        worker has reported an outcome it could not determine.
        """
        if self._quarantined_slots is None:
            self._quarantined_slots = {}
        if slot in self._quarantined_slots:
            return
        self._quarantined_slots[slot] = None
        self._free_slots.pop(slot, None)
        self._remove_slot_from_affinity_pool_locked(slot)

    def _remove_slot_from_affinity_pool_locked(self, slot: int) -> None:
        """Remove an allocated slot from its PID-specific free-slot pool."""
        placement_id = self._slot_placement_ids.get(slot)
        if placement_id is None:
            return
        affinity_slots = self._free_slots_by_placement_id.get(placement_id)
        if affinity_slots is None:
            return
        affinity_slots.pop(slot, None)
        if not affinity_slots:
            self._free_slots_by_placement_id.pop(placement_id, None)

    def _set_slot_placement_id_locked(
        self,
        slot: int,
        placement_id: PlacementId,
    ) -> None:
        """Record the latest runtime-only placement identifier for a slot."""
        if not self.fdp_slot_affinity_enabled or placement_id is None:
            self._slot_placement_ids.pop(slot, None)
            return
        self._slot_placement_ids[slot] = placement_id

    def _checkpoint_loop(self) -> None:
        """Periodically checkpoint dirty metadata until shutdown."""
        interval = max(1, self.meta_checkpoint_interval_sec)
        while not self._meta_stop_evt.wait(interval):
            try:
                self._checkpoint_once(force=False)
            except Exception as e:
                logger.warning("Periodic raw-block metadata checkpoint failed: %s", e)

    def _meta_payload_capacity(self) -> int:
        """Return usable bytes in one metadata checkpoint payload area."""
        return self._meta_container_bytes - self.block_align

    def _meta_container_offsets(self) -> list[int]:
        """Return byte offsets for mirrored metadata checkpoint containers."""
        return [
            idx * self._meta_container_bytes for idx in range(self._meta_copy_count)
        ]

    def _read_meta_header(self, container_offset: int) -> Optional[dict[str, int]]:
        """Read and validate a metadata checkpoint header."""
        buf = bytearray(self.block_align)
        try:
            if not all(
                self._read_buffers(
                    [container_offset],
                    [buf],
                    [self.block_align],
                    [self.block_align],
                    [IO_KIND_CHECKPOINT],
                    io_context=self._metadata_io_context,
                )
            ):
                return None
        except Exception:
            return None

        hdr = bytes(buf[: _META_HEADER_STRUCT.size])
        magic, version, seq, payload_len, crc = _META_HEADER_STRUCT.unpack(hdr)
        if magic != self.meta_magic or version != self.meta_version:
            return None

        payload_cap = self._meta_payload_capacity()
        if payload_len <= 0 or payload_len > payload_cap:
            return None
        return {
            "seq": int(seq),
            "payload_len": int(payload_len),
            "crc": int(crc),
            "container_offset": int(container_offset),
        }

    def _load_meta_payload(self, header: dict[str, int]) -> Optional[bytes]:
        """Load and CRC-validate a checkpoint payload for a metadata header."""
        payload_len = int(header["payload_len"])
        payload_off = int(header["container_offset"]) + self.block_align
        total_len = round_up(payload_len, self.block_align)
        buf = bytearray(total_len)
        try:
            if not all(
                self._read_buffers(
                    [payload_off],
                    [buf],
                    [payload_len],
                    [total_len],
                    [IO_KIND_CHECKPOINT],
                    io_context=self._metadata_io_context,
                )
            ):
                return None
        except Exception:
            return None

        payload = bytes(buf[:payload_len])
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        if crc != int(header["crc"]):
            return None
        return payload

    def _select_latest_checkpoint(
        self,
    ) -> tuple[Optional[dict[str, int]], Optional[bytes]]:
        """Return the newest valid checkpoint header and payload."""
        best_header: Optional[dict[str, int]] = None
        best_payload: Optional[bytes] = None
        for offset in self._meta_container_offsets():
            header = self._read_meta_header(offset)
            if header is None:
                continue
            payload = self._load_meta_payload(header)
            if payload is None:
                continue
            if best_header is None or int(header["seq"]) > int(best_header["seq"]):
                best_header = header
                best_payload = payload
        return best_header, best_payload

    def _snapshot_state(self) -> tuple[dict[str, Any], int]:
        """Build a JSON-serializable checkpoint state snapshot."""
        with self._lock:
            dirty_total = self._meta_dirty_total
            snapshot = {
                "version": 1,
                "writer_epoch": self._writer_epoch,
                "device_path": self.device_path,
                "capacity_bytes": self.capacity_bytes,
                "block_align": self.block_align,
                "header_bytes": self.header_bytes,
                "slot_bytes": self.slot_bytes,
                "meta_total_bytes": self.meta_total_bytes,
                "meta_magic": self.meta_magic_text,
                "meta_version": self.meta_version,
                # The namespace these keys were derived in. A core reading
                # this device with a different one computes a different slot
                # identity for every entry, so it reads each header as stale.
                "key_namespace": self.key_namespace,
                "derivation": (
                    self._derivation.as_payload() if self._derivation else None
                ),
                "data_base_offset": self._data_base_offset,
                "next_slot": self._next_slot,
                "entries": {
                    encoded_key: {
                        "offset": entry.offset,
                        "size": entry.meta.size,
                        "shape": list(entry.meta.shape)
                        if entry.meta.shape is not None
                        else None,
                        "dtype": self._checkpoint_dtype_name(entry.meta.dtype),
                        "fmt": (
                            entry.meta.fmt.name
                            if entry.meta.fmt is not None
                            and hasattr(entry.meta.fmt, "name")
                            else str(entry.meta.fmt)
                            if entry.meta.fmt is not None
                            else None
                        ),
                        "cached_positions": (
                            entry.meta.cached_positions.tolist()
                            if entry.meta.cached_positions is not None
                            and hasattr(entry.meta.cached_positions, "tolist")
                            else None
                        ),
                    }
                    for encoded_key, entry in self._index.items()
                },
            }
        return snapshot, dirty_total

    @staticmethod
    def _manifest_digest_from_records(
        encoded_keys: Sequence[str],
        records: Mapping[str, dict[str, Any]],
    ) -> str | None:
        """Hash ordered key identities and compatibility metadata."""
        manifest: list[dict[str, Any]] = []
        for encoded_key in encoded_keys:
            record = records.get(encoded_key)
            if record is None:
                return None
            manifest.append(record)
        payload = json.dumps(
            manifest,
            separators=(",", ":"),
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _published_manifest_digest(self, encoded_keys: Sequence[str]) -> str | None:
        """Hash a request against the exact serialized checkpoint generation."""
        return self._manifest_digest_from_records(
            encoded_keys,
            self._published_manifest,
        )

    def _manifest_from_index_locked(self) -> dict[str, dict[str, Any]]:
        """Build compatibility records while ``_lock`` is held."""
        return {
            encoded_key: {
                "key": encoded_key,
                "size": int(entry.meta.size),
                "shape": list(entry.meta.shape)
                if entry.meta.shape is not None
                else None,
                "dtype": self._checkpoint_dtype_name(entry.meta.dtype),
                "fmt": entry.meta.fmt.name
                if entry.meta.fmt is not None and hasattr(entry.meta.fmt, "name")
                else str(entry.meta.fmt)
                if entry.meta.fmt is not None
                else None,
            }
            for encoded_key, entry in self._index.items()
        }

    def _manifest_from_index(self) -> dict[str, dict[str, Any]]:
        """Return detached compatibility records for the current index."""
        with self._lock:
            return self._manifest_from_index_locked()

    def _checkpoint_dtype_name(self, dtype: torch.dtype | None) -> str | None:
        """Return a durable checkpoint string for a torch dtype.

        Args:
            dtype: Torch dtype from recovered or live memory metadata.

        Returns:
            Stable LMCache dtype name when known, ``str(dtype)`` for unknown
            torch dtypes, or None when no dtype is available.
        """
        if dtype is None:
            return None
        return TORCH_DTYPE_TO_STR_DTYPE.get(dtype, str(dtype))

    def _write_checkpoint(
        self,
        payload: bytes,
        dirty_total_snapshot: int,
        published_manifest: dict[str, dict[str, Any]],
        published_writer_epoch: str,
    ) -> bool:
        """Write one checkpoint copy and advance persisted metadata counters."""
        payload_cap = self._meta_payload_capacity()
        if len(payload) > payload_cap:
            logger.warning(
                "RawBlockCore metadata payload too large (%d > %d), "
                "skipping checkpoint",
                len(payload),
                payload_cap,
            )
            return False

        if self._meta_seq_resume_pending:
            self._meta_seq = max(self._meta_seq, self._max_checkpoint_seq_on_device())
            self._meta_seq_resume_pending = False
        next_seq = self._meta_seq + 1
        target_idx = int((next_seq - 1) % self._meta_copy_count)
        target = self._meta_container_offsets()[target_idx]

        payload_len = len(payload)
        payload_total_len = round_up(payload_len, self.block_align)
        payload_off = target + self.block_align
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        header_block = bytearray(self.block_align)
        header_block[: _META_HEADER_STRUCT.size] = _META_HEADER_STRUCT.pack(
            self.meta_magic,
            self.meta_version,
            int(next_seq),
            int(payload_len),
            int(crc),
        )

        placement_id = self.meta_checkpoint_placement_id
        # The header is the checkpoint's commit record.  Do not batch it with
        # the payload: independent io_uring requests may complete out of order,
        # exposing a valid new header before its payload is durable/readable.
        self._write_buffers(
            [payload_off],
            [payload],
            [payload_len],
            [payload_total_len],
            [placement_id],
            [IO_KIND_CHECKPOINT],
            io_context=self._metadata_io_context,
        )
        self._write_buffers(
            [target],
            [header_block],
            [self.block_align],
            [self.block_align],
            [placement_id],
            [IO_KIND_CHECKPOINT],
            io_context=self._metadata_io_context,
        )

        with self._lock:
            self._meta_seq = int(next_seq)
            self._meta_persisted = max(self._meta_persisted, int(dirty_total_snapshot))
            self._published_manifest = published_manifest
            self._published_keys = frozenset(published_manifest)
            self._published_writer_epoch = published_writer_epoch
        return True

    def _checkpoint_once(self, force: bool) -> bool:
        """Write a metadata checkpoint when dirty and sufficiently idle."""
        with self._checkpoint_lock:
            return self._checkpoint_once_locked(force)

    def _checkpoint_once_locked(
        self, force: bool, *, rewrite_clean: bool = False
    ) -> bool:
        """Write a checkpoint while ``_checkpoint_lock`` is held."""
        with self._lock:
            dirty = self._meta_dirty_total > self._meta_persisted
            idle_ok = self._inflight_io_count == 0 and (
                time.monotonic() - self._last_io_ts
            ) >= (self.meta_idle_quiet_ms / 1000.0)

        if not dirty and not rewrite_clean:
            return False
        if not force and not idle_ok:
            return False

        snapshot, dirty_total_snapshot = self._snapshot_state()
        published_manifest = {
            str(encoded_key): {
                "key": str(encoded_key),
                "size": int(entry["size"]),
                "shape": entry.get("shape"),
                "dtype": entry.get("dtype"),
                "fmt": entry.get("fmt"),
            }
            for encoded_key, entry in snapshot["entries"].items()
        }
        payload = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
        return self._write_checkpoint(
            payload,
            dirty_total_snapshot,
            published_manifest,
            str(snapshot.get("writer_epoch", "")),
        )

    def _is_valid_checkpoint_entry(self, offset: int, size: int) -> bool:
        """Return whether a checkpoint entry references a valid data slot."""
        if offset < self._data_base_offset:
            return False
        rel = offset - self._data_base_offset
        if rel % self.slot_bytes != 0:
            return False
        slot = rel // self.slot_bytes
        if slot >= self._max_slots:
            return False
        return 0 < size <= (self.slot_bytes - self.header_bytes)

    def _apply_loaded_state(
        self, data: dict[str, Any], *, verify: Optional[bool] = None
    ) -> bool:
        """Apply decoded checkpoint state after validating layout fields.

        ``verify`` overrides ``meta_verify_on_load`` for this call.
        """
        if not isinstance(data, dict):
            return False
        if int(data.get("version", 0)) != 1:
            return False

        # Before any geometry check, because every one of those treats a
        # mismatch as "ignore this metadata and start empty" -- which is
        # right for a layout this engine cannot read and wrong here. A
        # device someone else is writing with a different derivation, or
        # under a different key namespace, is not empty: starting empty
        # means allocating over their live extents while their checkpoint
        # still advertises them. These raise rather than return False.
        self._require_matching_key_namespace(data.get("key_namespace"))
        self._require_compatible_derivation(data.get("derivation"))

        writer_epoch = data.get("writer_epoch", "")
        if not isinstance(writer_epoch, str):
            logger.warning("Device metadata writer_epoch is invalid; ignoring metadata")
            return False
        checkpoint_device_path = data.get("device_path")
        if checkpoint_device_path and checkpoint_device_path != self.device_path:
            logger.warning("Device metadata device_path mismatch; ignoring metadata")
            return False
        if int(data.get("block_align", self.block_align)) != self.block_align:
            logger.warning("Device metadata block_align mismatch; ignoring metadata")
            return False
        if int(data.get("header_bytes", self.header_bytes)) != self.header_bytes:
            logger.warning("Device metadata header_bytes mismatch; ignoring metadata")
            return False
        if int(data.get("slot_bytes", self.slot_bytes)) != self.slot_bytes:
            logger.warning("Device metadata slot_bytes mismatch; ignoring metadata")
            return False
        if (
            int(data.get("meta_total_bytes", self.meta_total_bytes))
            != self.meta_total_bytes
        ):
            logger.warning(
                "Device metadata meta_total_bytes mismatch; ignoring metadata"
            )
            return False
        if str(data.get("meta_magic", self.meta_magic_text)) != self.meta_magic_text:
            logger.warning("Device metadata meta_magic mismatch; ignoring metadata")
            return False
        if int(data.get("meta_version", self.meta_version)) != self.meta_version:
            logger.warning("Device metadata meta_version mismatch; ignoring metadata")
            return False

        try:
            next_slot = int(data.get("next_slot", 0))
        except Exception:
            logger.warning("Device metadata next_slot is invalid; ignoring metadata")
            return False
        if next_slot < 0 or next_slot > self._max_slots:
            logger.warning(
                "Device metadata next_slot out of range (%d); ignoring metadata",
                next_slot,
            )
            return False

        with self._lock:
            self._next_slot = next_slot
            self._free_slots = {}
            self._free_slots_by_placement_id.clear()
            self._slot_placement_ids.clear()
            self._index.clear()
            self._lock_refcnt.clear()

            entries = data.get("entries", {})
            if isinstance(entries, dict):
                for encoded_key, entry in entries.items():
                    if not isinstance(entry, dict):
                        continue

                    offset = int(entry.get("offset", 0))
                    size = int(entry.get("size", 0))
                    shape_list = entry.get("shape")
                    fmt_name = entry.get("fmt")
                    cached_positions_list = entry.get("cached_positions")
                    dtype_name = entry.get("dtype")

                    if not self._is_valid_checkpoint_entry(offset, size):
                        continue

                    shape = (
                        torch.Size(list(shape_list)) if shape_list is not None else None
                    )
                    fmt = (
                        MemoryFormat[fmt_name]
                        if isinstance(fmt_name, str)
                        and fmt_name in MemoryFormat.__members__
                        else MemoryFormat.UNDEFINED
                    )
                    cached_positions = (
                        torch.tensor(cached_positions_list, dtype=torch.long)
                        if cached_positions_list is not None
                        else None
                    )
                    dtype = self._recover_checkpoint_dtype(
                        str(encoded_key),
                        dtype_name,
                    )

                    meta = DiskCacheMetadata(
                        path=f"{self.device_path}@{offset}",
                        size=size,
                        shape=shape,
                        dtype=dtype,
                        cached_positions=cached_positions,
                        fmt=fmt,
                        pin_count=0,
                    )
                    self._index[encoded_key] = _Entry(
                        offset=offset, size=size, meta=meta
                    )

            used_slots = {
                self._offset_to_slot(int(entry.offset))
                for entry in self._index.values()
            }
            # Rebuild from committed entries instead of trusting checkpoint
            # free_slots. A crash-time checkpoint can otherwise preserve a slot
            # reserved by an uncommitted in-flight write as neither used nor free.
            # Quarantined slots are in neither set and must stay withheld: the
            # index no longer names them, so a plain rebuild would hand back
            # exactly the extents whose outcome is unknown.
            quarantined = self._quarantined_slots or {}
            self._free_slots = {
                slot: None
                for slot in range(self._next_slot)
                if slot not in used_slots and slot not in quarantined
            }

            self._meta_dirty_total = 0
            self._meta_persisted = 0
            self._published_writer_epoch = writer_epoch
            self._published_manifest = self._manifest_from_index_locked()
            self._published_keys = frozenset(self._published_manifest)
            if self.role == "reader":
                self._writer_epoch = writer_epoch

        if self.meta_verify_on_load if verify is None else verify:
            self._validate_loaded_entries()
        return True

    def _recover_checkpoint_dtype(
        self,
        encoded_key: str,
        dtype_name: Any,
    ) -> torch.dtype | None:
        """Recover checkpoint dtype from entry metadata or legacy key strings.

        Args:
            encoded_key: Encoded raw-block key from the checkpoint entry.
            dtype_name: Raw dtype value stored in the checkpoint entry.

        Returns:
            A torch dtype when recovery succeeds, otherwise None.
        """
        if isinstance(dtype_name, str):
            dtype = STR_DTYPE_TO_TORCH_DTYPE.get(dtype_name)
            if dtype is not None:
                return dtype

            torch_prefix = "torch."
            if dtype_name.startswith(torch_prefix):
                dtype_attr = dtype_name.removeprefix(torch_prefix)
                dtype = STR_DTYPE_TO_TORCH_DTYPE.get(dtype_attr)
                if dtype is not None:
                    return dtype
                torch_dtype = getattr(torch, dtype_attr, None)
                if isinstance(torch_dtype, torch.dtype):
                    return torch_dtype

        if self.key_namespace != "legacy":
            return None

        try:
            return decode_legacy_key(encoded_key).dtype
        except Exception:
            logger.debug(
                "Unable to recover dtype from legacy raw-block key %s",
                encoded_key,
                exc_info=True,
            )
            return None

    def _validate_loaded_entries(self) -> None:
        """Drop recovered entries whose slot headers do not match metadata."""
        to_drop: list[str] = []
        unreadable: Optional[str] = None
        with self._lock:
            items = list(self._index.items())

        for encoded_key, entry in items:
            state, slot_hdr = self._read_slot_header(int(entry.offset))
            if state == "unreadable":
                # Not evidence this entry is stale -- evidence that the
                # device did not answer. Dropping it and recycling its slot
                # would be a decision made on no information, so validation
                # stops and every entry it has not judged is kept. The same
                # condition applies to all of them.
                unreadable = encoded_key
                break
            if slot_hdr is None:
                to_drop.append(encoded_key)
                continue
            try:
                expected_identity = slot_identity_from_encoded_key(
                    encoded_key,
                    self.key_namespace,
                )
            except Exception:
                to_drop.append(encoded_key)
                continue
            slot_identity, payload_len = slot_hdr
            if int(slot_identity) != int(expected_identity):
                to_drop.append(encoded_key)
                continue
            if int(payload_len) != int(entry.size):
                to_drop.append(encoded_key)

        if unreadable is not None:
            logger.error(
                "RawBlockCore could not read the slot header for key %s; "
                "stopping validation and keeping every entry it had not "
                "judged. An unreadable header says the device did not "
                "answer, not that the entry is stale.",
                unreadable,
            )

        if not to_drop:
            return

        with self._lock:
            for encoded_key in to_drop:
                removed_entry = self._index.pop(encoded_key, None)
                self._lock_refcnt.pop(encoded_key, None)
                if removed_entry is not None:
                    # A header that arrived and is not ours is a known
                    # outcome, so ordinary recycling applies -- unless this
                    # core has stopped being able to say, which this decides.
                    self._release_submitted_slot_locked(
                        self._offset_to_slot(int(removed_entry.offset))
                    )
            self._meta_dirty_total += 1

        logger.warning(
            "RawBlockCore dropped %d stale metadata entries after "
            "slot-header validation",
            len(to_drop),
        )

    def _load_checkpoint_from_device(self) -> None:
        """Load the newest valid checkpoint from the raw device if present."""
        header, payload = self._select_latest_checkpoint()
        if header is None:
            logger.info("RawBlockCore: no valid on-device metadata checkpoint found")
            return
        if payload is None:
            logger.warning("RawBlockCore: checkpoint header had no payload")
            return
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            logger.warning("RawBlockCore: failed to decode metadata payload")
            return
        if not self.apply_loaded_state(data):
            logger.warning("RawBlockCore: metadata payload rejected by checks")
            return
        self._meta_seq = int(header["seq"])
        self._published_keys = frozenset(self._index)
        self._published_manifest = self._manifest_from_index()
        self._published_writer_epoch = str(data.get("writer_epoch", ""))
        logger.info(
            "RawBlockCore loaded checkpoint (entries=%d next_slot=%d seq=%d device=%s)",
            len(self._index),
            self._next_slot,
            self._meta_seq,
            self.device_path,
        )
