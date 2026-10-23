# SPDX-License-Identifier: Apache-2.0
"""CPU ownership rules for an opt-in KMesh multiprocess data plane.

This registry does not register tensors, serve objects, or enable a connector.
Adapters must bind evidence from their actual worker and transport session and
must check publication before committing an accepted write. See the design doc
for the remaining data-plane obligations.
"""

# Standard
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
import threading
import time
import uuid


class KMeshFenceError(RuntimeError):
    """A stale, incomplete, or incompatible KMesh ownership claim."""


def _digest(value: object) -> str:
    """Hash a versioned, JSON-serializable contract with stable ordering."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _label(value: str) -> None:
    """Reject empty, oversized, or control-bearing local identifiers."""
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        raise ValueError("invalid KMesh identifier")


def _positive(value: int) -> None:
    """Require a positive integer, excluding booleans."""
    if type(value) is not int or value <= 0:
        raise ValueError("expected a positive integer")


@dataclass(frozen=True, slots=True)
class SemanticIdentity:
    """Version-one text-only KV semantics, separate from execution settings.

    Digests must describe resolved values, including defaults, not config file
    names. Layout covers layer/head partitions, shapes, strides, attention kind,
    scale representation and group ordering. RoPE covers all effective position
    parameters. Weights and tokenizer digests identify immutable artifacts.
    Producing these digests from the actual runtime is an adapter obligation.
    """

    weights_digest: str
    model_config_digest: str
    tokenizer_digest: str
    rope_digest: str
    layout_digest: str
    kv_dtype: str
    chunk_tokens: int
    hash_algorithm: str
    schema: int = 1
    key_schema: str = "kmesh-mp-v1"
    adapter_identity: str = ""

    def __post_init__(self) -> None:
        for value in (
            self.weights_digest,
            self.model_config_digest,
            self.tokenizer_digest,
            self.rope_digest,
            self.layout_digest,
        ):
            if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("semantic digests must be lowercase SHA-256")
        if type(self.schema) is not int or self.schema != 1:
            raise ValueError("unsupported semantic schema")
        if self.key_schema != "kmesh-mp-v1" or self.adapter_identity != "":
            raise ValueError("unsupported key schema or adapter identity")
        if self.kv_dtype != "bfloat16":
            raise ValueError("the initial KMesh MP contract requires BF16 KV")
        _positive(self.chunk_tokens)
        _label(self.hash_algorithm)

    @property
    def fingerprint(self) -> str:
        """Return the semantic namespace digest, excluding runtime placement."""
        return _digest(("kmesh-semantics-v1", asdict(self)))


@dataclass(frozen=True, slots=True)
class ServerDescriptor:
    """Identity of one server boot, store generation and handler inventory."""

    boot_id: str
    store_epoch: int
    capabilities_digest: str


@dataclass(frozen=True, slots=True)
class DomainDescriptor:
    """A semantic domain generation within one server store generation."""

    domain_id: str
    domain_epoch: int
    store_epoch: int
    boot_id: str
    semantics: SemanticIdentity

    @property
    def namespace(self) -> str:
        """Return an object namespace that changes on store/domain invalidation."""
        return _digest(
            (
                "kmesh-objects-v1",
                self.boot_id,
                self.domain_id,
                self.domain_epoch,
                self.store_epoch,
                self.semantics.fingerprint,
            )
        )


@dataclass(frozen=True, slots=True)
class RegistrationLease:
    """Expected rank barrier; a scheduler's copy alone cannot authorize I/O."""

    registration_id: str
    lease_epoch: int
    engine_run: str
    world_size: int
    execution_digest: str
    server: ServerDescriptor
    domain: DomainDescriptor


@dataclass(frozen=True, slots=True)
class WorkerFence:
    """Server-issued binding to a worker instance and actual client session."""

    lease: RegistrationLease
    rank: int
    instance_id: int
    session_id: str
    token: str


@dataclass(frozen=True, slots=True)
class OperationTicket:
    """Accepted work whose physical completion remains owed after revocation."""

    fence: WorkerFence
    request_id: str
    generation: int
    token: str


class Completion(str, Enum):
    """Physical terminal outcome; an undrained failure retains ownership debt."""

    SUCCEEDED = "succeeded"
    CANCELLED_DRAINED = "cancelled_drained"
    FAILED_UNDRAINED = "failed_undrained"


class RegistrationState(str, Enum):
    """Rank-barrier state exposed without leaking mutable registry state."""

    REGISTERING = "registering"
    READY = "ready"
    REVOKED = "revoked"


@dataclass
class _Registration:
    lease: RegistrationLease
    deadline: float
    state: RegistrationState = RegistrationState.REGISTERING
    workers: dict[int, WorkerFence] = field(default_factory=dict)
    heartbeats: dict[int, float] = field(default_factory=dict)
    generations: dict[tuple[int, str], int] = field(default_factory=dict)
    tickets: dict[str, OperationTicket] = field(default_factory=dict)
    published: set[str] = field(default_factory=set)


class DomainRegistry:
    """Serialize KMesh domain, registration and accepted-operation ownership.

    This is cooperative same-user correctness, not authentication. All methods
    are thread safe. The transport must supply the actual session identity; it
    must never accept a session name echoed from a scheduler payload. Physical
    drain is reported by completion callbacks, never inferred from a timeout.

    Args:
        operations: Names discovered from the server's live handler registry.
        lease_seconds: Maximum silence per bound worker or incomplete barrier.
        clock: Monotonic clock, injectable for deterministic expiration tests.
    """

    def __init__(
        self,
        operations: Iterable[str],
        lease_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be finite and positive")
        self._lock = threading.RLock()
        self._clock = clock
        self._lease_seconds = lease_seconds
        self._server = ServerDescriptor(
            uuid.uuid4().hex, 1, self._capabilities(operations)
        )
        self._domains: dict[str, DomainDescriptor] = {}
        self._registrations: dict[str, _Registration] = {}
        self._next_lease = 0

    @staticmethod
    def _capabilities(operations: Iterable[str]) -> str:
        names = tuple(sorted(set(operations)))
        for name in names:
            _label(name)
        return _digest(("kmesh-handler-inventory-v1", names))

    @property
    def server(self) -> ServerDescriptor:
        """Return the current server descriptor, never an authorization token."""
        with self._lock:
            return self._server

    def open_domain(self, semantics: SemanticIdentity) -> DomainDescriptor:
        """Atomically open or create a namespace for validated semantics.

        Incompatible semantics get disjoint domains, including when old objects
        survive deregistration. Reject unsupported schema/adapter identities in
        SemanticIdentity construction, before this method is called.
        """
        if type(semantics) is not SemanticIdentity:
            raise ValueError("expected a validated SemanticIdentity")
        with self._lock:
            key = semantics.fingerprint
            if key not in self._domains:
                self._domains[key] = DomainDescriptor(
                    key, 1, self._server.store_epoch, self._server.boot_id, semantics
                )
            return self._domains[key]

    def open_registration(
        self,
        domain: DomainDescriptor,
        engine_run: str,
        world_size: int,
        execution_digest: str,
    ) -> RegistrationLease:
        """Create an incomplete barrier; all actual workers must bind to arm.

        Args:
            domain: Current descriptor from open_domain.
            engine_run: Unique engine invocation, not a reusable server URL.
            world_size: Exact number of ranks required before any I/O.
            execution_digest: Frozen research-profile digest, distinct from KV
                compatibility. All ranks must independently agree on it.

        Raises:
            KMeshFenceError: Stale domain or an existing undrained engine run.
            ValueError: Malformed input.
        """
        _label(engine_run)
        _positive(world_size)
        if not re.fullmatch(r"[0-9a-f]{64}", execution_digest):
            raise ValueError("execution_digest must be lowercase SHA-256")
        with self._lock:
            self._expire()
            self._domain(domain)
            for entry in self._registrations.values():
                if entry.lease.engine_run == engine_run and (
                    entry.state != RegistrationState.REVOKED or entry.tickets
                ):
                    raise KMeshFenceError("engine run is active or owes a drain")
            self._next_lease += 1
            lease = RegistrationLease(
                uuid.uuid4().hex,
                self._next_lease,
                engine_run,
                world_size,
                execution_digest,
                self._server,
                domain,
            )
            self._registrations[lease.registration_id] = _Registration(
                lease, self._clock() + self._lease_seconds
            )
            return lease

    def bind_worker(
        self,
        lease: RegistrationLease,
        rank: int,
        instance_id: int,
        session_id: str,
        semantics: SemanticIdentity,
        execution_digest: str,
    ) -> WorkerFence:
        """Bind actual worker evidence and negotiated context to a rank.

        Duplicate ranks, reused live sessions/instances, mismatched semantics
        or profile evidence revoke this barrier. Replacement of an instance
        with pending accepted work refuses until that work physically drains.
        This method must run only after successful tensor registration.
        """
        _label(session_id)
        if type(instance_id) is not int or instance_id < 0:
            raise ValueError("invalid instance_id")
        with self._lock:
            entry = self._live(lease)
            if (
                type(rank) is not int
                or not 0 <= rank < lease.world_size
                or rank in entry.workers
                or semantics != lease.domain.semantics
                or execution_digest != lease.execution_digest
            ):
                entry.state = RegistrationState.REVOKED
                raise KMeshFenceError("inconsistent or duplicate worker evidence")
            for other in self._registrations.values():
                for worker in other.workers.values():
                    if (
                        worker.instance_id == instance_id
                        or worker.session_id == session_id
                    ) and (
                        other.state != RegistrationState.REVOKED
                        or any(t.fence == worker for t in other.tickets.values())
                    ):
                        entry.state = RegistrationState.REVOKED
                        raise KMeshFenceError("worker is bound or owes a drain")
            fence = WorkerFence(lease, rank, instance_id, session_id, uuid.uuid4().hex)
            entry.workers[rank] = fence
            entry.heartbeats[rank] = self._clock() + self._lease_seconds
            if len(entry.workers) == lease.world_size:
                entry.state = RegistrationState.READY
            return fence

    def state(self, lease: RegistrationLease) -> RegistrationState:
        """Return barrier state, expiring a missed heartbeat before reporting."""
        with self._lock:
            self._expire()
            return self._entry(lease).state

    def heartbeat(self, fence: WorkerFence, session_id: str) -> None:
        """Refresh one actual worker; expiration is irreversible for this lease."""
        with self._lock:
            entry = self._worker(fence, session_id, require_ready=False)
            entry.heartbeats[fence.rank] = self._clock() + self._lease_seconds

    def acquire(
        self, fence: WorkerFence, session_id: str, request_id: str, generation: int
    ) -> OperationTicket:
        """Accept one transaction with a strictly increasing request generation.

        The generation is consumed even if the transaction aborts. A ticket
        covers the whole lookup/read/write transaction, not each sub-operation.
        Only a READY, live registration can accept a new transaction.
        """
        _label(request_id)
        _positive(generation)
        with self._lock:
            entry = self._worker(fence, session_id)
            key = (fence.rank, request_id)
            if generation <= entry.generations.get(key, 0):
                raise KMeshFenceError(
                    "request generation cannot be reused or rolled back"
                )
            if any(
                t.fence == fence and t.request_id == request_id
                for t in entry.tickets.values()
            ):
                raise KMeshFenceError("request still owns accepted work")
            entry.generations[key] = generation
            ticket = OperationTicket(fence, request_id, generation, uuid.uuid4().hex)
            entry.tickets[ticket.token] = ticket
            return ticket

    def publish(
        self, ticket: OperationTicket, session_id: str, commit: Callable[[str], None]
    ) -> None:
        """Commit accepted output atomically with respect to registry revocation.

        The callback must synchronously publish already-completed data under the
        supplied namespace. It must not wait for asynchronous work or re-enter
        this registry. Failure invalidates the domain generation and leaves the
        ticket owed; caller must drain it. Partially published bytes therefore
        cannot be read under a subsequent registration for the same semantics.
        """
        with self._lock:
            entry = self._ticket(ticket)
            if ticket.token in entry.published:
                raise KMeshFenceError("operation is already published")
            self._worker(ticket.fence, session_id)
            namespace = ticket.fence.lease.domain.namespace
            try:
                commit(namespace)
            except Exception:
                self.invalidate_domain(ticket.fence.lease.domain)
                raise
            entry.published.add(ticket.token)

    def finish(self, ticket: OperationTicket, completion: Completion) -> None:
        """Report physical completion, including after revocation or store reset.

        FAILED_UNDRAINED revokes the whole registration and keeps the ticket.
        Later CANCELLED_DRAINED/SUCCEEDED releases the debt but cannot rearm it.
        Success means physical completion only; publication requires publish().
        """
        if type(completion) is not Completion:
            raise ValueError("expected Completion")
        with self._lock:
            entry = self._ticket(ticket)
            if completion == Completion.FAILED_UNDRAINED:
                entry.state = RegistrationState.REVOKED
            else:
                del entry.tickets[ticket.token]
                entry.published.discard(ticket.token)

    def pending(self, lease: RegistrationLease) -> int:
        """Return accepted transactions still requiring physical completion."""
        with self._lock:
            return len(self._entry(lease).tickets)

    def revoke(self, lease: RegistrationLease) -> None:
        """Disarm a registration without discarding accepted-operation debt."""
        with self._lock:
            self._entry(lease).state = RegistrationState.REVOKED

    def invalidate_domain(self, domain: DomainDescriptor) -> DomainDescriptor:
        """Advance an index/geometry generation even if object bytes survive."""
        with self._lock:
            self._domain(domain)
            replacement = DomainDescriptor(
                domain.domain_id,
                domain.domain_epoch + 1,
                domain.store_epoch,
                domain.boot_id,
                domain.semantics,
            )
            self._domains[domain.domain_id] = replacement
            for entry in self._registrations.values():
                if entry.lease.domain == domain:
                    entry.state = RegistrationState.REVOKED
            return replacement

    def invalidate_store(self) -> None:
        """Invalidate all logical objects, retaining physical completion debt."""
        with self._lock:
            self._server = ServerDescriptor(
                self._server.boot_id,
                self._server.store_epoch + 1,
                self._server.capabilities_digest,
            )
            self._domains.clear()
            for entry in self._registrations.values():
                entry.state = RegistrationState.REVOKED

    def update_capabilities(self, operations: Iterable[str]) -> None:
        """Disarm all registrations if the actual handler inventory changes."""
        digest = self._capabilities(operations)
        with self._lock:
            if digest != self._server.capabilities_digest:
                self._server = ServerDescriptor(
                    self._server.boot_id, self._server.store_epoch, digest
                )
                for entry in self._registrations.values():
                    entry.state = RegistrationState.REVOKED

    def _domain(self, domain: DomainDescriptor) -> None:
        if self._domains.get(domain.domain_id) != domain:
            raise KMeshFenceError("stale domain descriptor")

    def _entry(self, lease: RegistrationLease) -> _Registration:
        entry = self._registrations.get(lease.registration_id)
        if entry is None or entry.lease != lease:
            raise KMeshFenceError("unknown registration or server boot")
        return entry

    def _expire(self) -> None:
        now = self._clock()
        for entry in self._registrations.values():
            if (
                entry.state == RegistrationState.REGISTERING
                and now >= entry.deadline
                or any(now >= expiry for expiry in entry.heartbeats.values())
            ):
                entry.state = RegistrationState.REVOKED

    def _live(self, lease: RegistrationLease) -> _Registration:
        self._expire()
        entry = self._entry(lease)
        if entry.state == RegistrationState.REVOKED or lease.server != self._server:
            raise KMeshFenceError("registration is revoked or stale")
        self._domain(lease.domain)
        return entry

    def _worker(
        self, fence: WorkerFence, session_id: str, *, require_ready: bool = True
    ) -> _Registration:
        entry = self._live(fence.lease)
        if entry.workers.get(fence.rank) != fence or fence.session_id != session_id:
            raise KMeshFenceError("worker session is not bound")
        if require_ready and entry.state != RegistrationState.READY:
            raise KMeshFenceError("rank barrier is incomplete")
        return entry

    def _ticket(self, ticket: OperationTicket) -> _Registration:
        entry = self._entry(ticket.fence.lease)
        if entry.tickets.get(ticket.token) != ticket:
            raise KMeshFenceError("unknown or already completed operation")
        return entry
