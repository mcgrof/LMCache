# SPDX-License-Identifier: Apache-2.0
"""Typed KMesh control values shared by clients and server modules."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.multiprocess.context_ownership import ContextIdentity, RopeIdentity
from lmcache.v1.multiprocess.kmesh_domains import RegistrationState


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
