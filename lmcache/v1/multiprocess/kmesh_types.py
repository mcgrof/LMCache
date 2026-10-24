# SPDX-License-Identifier: Apache-2.0
"""Typed KMesh control values shared by clients and server modules."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.multiprocess.context_ownership import ContextIdentity, RopeIdentity
from lmcache.v1.multiprocess.kmesh_domains import RegistrationState, WorkerFence


@dataclass(frozen=True, slots=True)
class RegisteredRuntime:
    """Describe exact registered tensors without authorizing any operation."""

    context: ContextIdentity
    rotary: RopeIdentity
    model_name: str
    world_size: int
    layout_digest: str
    rope_digest: str
    kv_dtype: str
    chunk_tokens: int
    hash_algorithm: str


@dataclass(frozen=True, slots=True)
class RegistrationStatus:
    """Report a rank barrier and accepted debt, not exporter teardown."""

    state: RegistrationState
    pending: int


@dataclass(frozen=True, slots=True)
class ManagedRawRegistration:
    """Report one actual receiving client's bounded raw context registration.

    Status lookup resolves a lost reply; only a separate release receipt permits
    exporter teardown. Poisoned construction requires importing-process exit.
    """

    boot_id: str
    nonce: str
    instance_id: int
    context: ContextIdentity
    state: str
    allocation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ManagedRawRelease:
    """Acknowledge server-observed retirement of one exact raw registration.

    The server has closed admission, drained physical work and retired all raw
    input mappings before caching this idempotent value. This covers only its
    bounded managed profile, not other IPC or service implementations.
    """

    boot_id: str
    nonce: str
    instance_id: int
    context: ContextIdentity
    rotary: RopeIdentity | None
    allocation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpanHandle:
    """Identify a published immutable source within one semantic namespace."""

    namespace: str
    span_id: str
    source_start: int
    token_count: int
    token_digest: str
    context_digest: str
    layout_digest: str
    rope_digest: str


@dataclass(frozen=True, slots=True)
class SpanOperationStatus:
    """Describe server-observed work for one exact request generation.

    Executing and failed-undrained work still owns its blocks and buffers.
    Stored/loaded/failed-drained states prove this operation's physical drain.
    Loaded-needs-repair proves I/O completion only: model state must be repaired
    before the worker can mark those tokens computed. None or unknown status
    cannot establish completion.
    """

    fence: WorkerFence
    request_id: str
    generation: int
    operation_id: str
    context: ContextIdentity
    kind: str
    blocks_digest: str
    state: str
    error: str
    span: SpanHandle | None
