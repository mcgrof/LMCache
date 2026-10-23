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
