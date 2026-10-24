# SPDX-License-Identifier: Apache-2.0
"""Expose opt-in registered-state controls without enabling KMesh recovery."""

# Standard
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import TypeVar
import time

# First Party
from lmcache.v1.multiprocess.context_ownership import ContextIdentity
from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.kmesh_bindings import RegisteredWorkerBindings
from lmcache.v1.multiprocess.kmesh_domains import (
    DomainRegistry,
    RegistrationLease,
    SemanticIdentity,
    ServerDescriptor,
    WorkerFence,
)
from lmcache.v1.multiprocess.kmesh_types import RegisteredRuntime, RegistrationStatus
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.request_context import get_request_session

T = TypeVar("T")


def control_reply(action: Callable[[], T]) -> tuple[T | None, str]:
    """Return an explicit refusal on both transports without inventing evidence.

    ZeroMQ does not send a response when an ordinary handler raises. Expected
    ownership refusals therefore travel as typed absent-value/error replies.

    Args:
        action: Trusted server operation requiring an actual request session.

    Returns:
        The operation result with an empty error, or no value and a bounded
        expected-refusal message. This wrapper does not create completion proof.
    """
    try:
        get_request_session()
        return action(), ""
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        return None, str(error)[:512]


class KMeshControlModule(InstanceLivenessTarget):
    """Serve CPU registration controls; keep physical completion server-owned.

    Args:
        ctx: Actual shared storage-server context.
        transfer: Module owning paged tensor registrations.
        blend: Module owning rotary registrations.
        lease_seconds: Worker-silence budget, separate from ordinary PING.
        clock: Monotonic lease clock; injectable for deterministic CPU tests.
        allow_device_snapshots: Explicit experimental ordered-device opt-in.
        raw_snapshot_inputs: Select managed raw-only inputs for the dedicated server.
    """

    def __init__(
        self,
        ctx: MPCacheServerContext,
        transfer: LMCacheDrivenTransferModule,
        blend: BlendModule,
        lease_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        *,
        allow_device_snapshots: bool = False,
        raw_snapshot_inputs: bool = False,
    ) -> None:
        self._ctx = ctx
        self._bindings = RegisteredWorkerBindings(
            DomainRegistry([], lease_seconds, clock),
            transfer,
            blend,
            allow_device_snapshots=allow_device_snapshots,
            raw_snapshot_inputs=raw_snapshot_inputs,
        )

    @property
    def context(self) -> MPCacheServerContext:
        """Return the actual shared server context."""
        return self._ctx

    @property
    def bindings(self) -> RegisteredWorkerBindings:
        """Expose server-owned resources to executors, never through a client RPC."""
        return self._bindings

    def set_handler_inventory(self, operations: list[str]) -> None:
        """Bind capabilities to the actual assembled server handlers."""
        self._bindings.update_capabilities(operations)

    def report_status(self) -> dict:
        """Report control identity without claiming engine or device readiness."""
        return {"kmesh_control": asdict(self._bindings.server)}

    def clear_store(self, clear: Callable[[], None]) -> None:
        """Serialize store clearing with admission and refuse physical debt."""
        self._bindings.clear_store(clear)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Release expired lease holds without impersonating ordinary PING."""
        self._bindings.reap_expired()
        return []

    def close(self) -> None:
        """Revoke leases before tensor modules close; retain undrained work."""
        self._bindings.close()

    @request_handler()
    def kmesh_get_server(self) -> tuple[ServerDescriptor | None, str]:
        return control_reply(lambda: self._bindings.server)

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def kmesh_get_runtime(
        self, instance_id: int
    ) -> tuple[RegisteredRuntime | None, str]:
        return control_reply(lambda: self._bindings.describe(instance_id))

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def kmesh_get_context(self, instance_id: int) -> tuple[ContextIdentity | None, str]:
        return control_reply(lambda: self._bindings.context_identity(instance_id))

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def kmesh_register_rotary(
        self,
        instance_id: int,
        context: ContextIdentity,
        caches_ipc: list[DeviceIPCWrapper],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        group_rot: list[list[int]],
        group_head_size: list[int],
        kv_quant: str,
        producer_event: bytes,
    ) -> tuple[RegisteredRuntime | None, str]:
        return control_reply(
            lambda: self._bindings.register_rotary_snapshot(
                instance_id,
                context,
                caches_ipc,
                head_size,
                is_neox_style,
                group_to_cache,
                group_rot,
                group_head_size,
                kv_quant,
                producer_event,
            )
        )

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def kmesh_retry_rotary_drain(
        self, context: ContextIdentity
    ) -> tuple[int | None, str]:
        """Report actual copy drain; zero does not acknowledge exporter teardown."""
        return control_reply(lambda: self._bindings.retry_rotary_drain(context))

    @request_handler()
    def kmesh_open_registration(
        self,
        semantics: SemanticIdentity,
        engine_run: str,
        world_size: int,
        execution_digest: str,
    ) -> tuple[RegistrationLease | None, str]:
        return control_reply(
            lambda: self._bindings.open_registration(
                replace(semantics), engine_run, world_size, execution_digest
            )
        )

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def kmesh_bind_worker(
        self,
        lease: RegistrationLease,
        rank: int,
        instance_id: int,
        expected: RegisteredRuntime,
        semantics: SemanticIdentity,
        execution_digest: str,
    ) -> tuple[WorkerFence | None, str]:
        return control_reply(
            lambda: self._bindings.bind(
                lease, rank, instance_id, expected, replace(semantics), execution_digest
            )
        )

    @request_handler()
    def kmesh_heartbeat(self, fence: WorkerFence) -> tuple[bool | None, str]:
        def heartbeat() -> bool:
            self._bindings.heartbeat(fence)
            return True

        return control_reply(heartbeat)

    @request_handler()
    def kmesh_registration_state(
        self, lease: RegistrationLease
    ) -> tuple[RegistrationStatus | None, str]:
        return control_reply(
            lambda: RegistrationStatus(
                self._bindings.state(lease), self._bindings.pending(lease)
            )
        )

    @request_handler()
    def kmesh_revoke_registration(
        self, lease: RegistrationLease
    ) -> tuple[int | None, str]:
        """Return accepted debt only; this does not acknowledge exporter teardown."""

        def revoke() -> int:
            self._bindings.revoke(lease)
            return self._bindings.pending(lease)

        return control_reply(revoke)
