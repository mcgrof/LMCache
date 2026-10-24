# SPDX-License-Identifier: Apache-2.0
"""Own one raw context through registration, ordered preparation and retirement."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import hashlib
import json
import re
import threading

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.context_ownership import ContextIdentity, ContextOwnership
from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.kmesh_domains import KMeshFenceError, WorkerFence
from lmcache.v1.multiprocess.kmesh_span_store import KMeshSpanStore
from lmcache.v1.multiprocess.kmesh_types import (
    ManagedRawRegistration,
    ManagedRawRelease,
    RegisteredRuntime,
    SpanHandle,
    SpanOperationStatus,
)
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.modules.kmesh_control import (
    KMeshControlModule,
    control_reply,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry,
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.request_context import get_request_session
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.devices.cuda.cache_context import GPUCacheContext
from lmcache.v1.platform.devices.cuda.ipc_wrapper import RawCudaIPCWrapper


@dataclass
class _Registration:
    nonce: str
    instance_id: int
    model_name: str
    signature: tuple[str, ...]
    owners: tuple[RawCudaIPCWrapper, ...]
    ownership: ContextOwnership = field(default_factory=ContextOwnership)
    state: str = "constructing"
    cache: GPUCacheContext | None = None
    entry: ContextEntry | None = None
    rotary_owners: tuple[RawCudaIPCWrapper, ...] = ()
    rotary_signature: str | None = None
    runtime: RegisteredRuntime | None = None
    rotary_retired: bool = False
    release: ManagedRawRelease | None = None


class KMeshManagedRawModule(KMeshControlModule):
    """Expose a bounded raw-only control surface with server-owned release.

    The standalone compositor serves only this module: no integer registration,
    generic transfer, lookup, peer, HTTP or reaper can touch its context. One
    registration and one rotary preparation are admitted per server lifetime.
    Native span execution is an explicit opt-in; model admission remains
    disabled. A loaded span may still require deeper-layer model repair.
    """

    def __init__(
        self, ctx: MPCacheServerContext, *, enable_span_execution: bool = False
    ) -> None:
        self._transfer = LMCacheDrivenTransferModule(ctx)
        self._blend = BlendModule(ctx, self._transfer)
        super().__init__(
            ctx,
            self._transfer,
            self._blend,
            allow_device_snapshots=True,
            raw_snapshot_inputs=True,
        )
        self._managed_lock = threading.RLock()
        self._registration: _Registration | None = None
        self._spans = (
            KMeshSpanStore(self.bindings, ctx.storage_manager, self._blend)
            if enable_span_execution
            else None
        )

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def kmesh_store_span(
        self,
        fence: WorkerFence,
        request_id: str,
        generation: int,
        token_ids: list[int],
        source_start: int,
        context_digest: str,
        block_ids: list[list[int]],
        producer_event: bytes,
    ) -> tuple[SpanOperationStatus | None, str]:
        """Gather source chunks and return observed admission or failure state."""
        return control_reply(
            lambda: self._span_executor(fence).store(
                fence,
                request_id,
                generation,
                token_ids,
                source_start,
                context_digest,
                block_ids,
                producer_event,
            )
        )

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def kmesh_restore_span(
        self,
        fence: WorkerFence,
        request_id: str,
        generation: int,
        span: SpanHandle,
        destination_start: int,
        context_digest: str,
        block_ids: list[list[int]],
        producer_event: bytes,
    ) -> tuple[SpanOperationStatus | None, str]:
        """Restore into reserved blocks; positional/context changes need repair."""
        return control_reply(
            lambda: self._span_executor(fence).restore(
                fence,
                request_id,
                generation,
                span,
                destination_start,
                context_digest,
                block_ids,
                producer_event,
            )
        )

    @request_handler()
    def kmesh_span_status(
        self,
        fence: WorkerFence,
        request_id: str,
        generation: int,
    ) -> tuple[SpanOperationStatus | None, str]:
        """Inspect known work independently of execution; None is not drain."""

        def status() -> SpanOperationStatus:
            if self._spans is None:
                raise KMeshFenceError("managed span execution is not admitted")
            return self._spans.status(fence, request_id, generation)

        return control_reply(status)

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def kmesh_drain_span(
        self,
        fence: WorkerFence,
        request_id: str,
        generation: int,
    ) -> tuple[SpanOperationStatus | None, str]:
        """Retry physical cleanup without resubmitting a failed native transfer."""

        def drain() -> SpanOperationStatus:
            if self._spans is None:
                raise KMeshFenceError("managed span execution is not admitted")
            return self._spans.drain(fence, request_id, generation)

        return control_reply(drain)

    def _record(self, boot_id: str, nonce: str) -> _Registration:
        record = self._registration
        if boot_id != self.bindings.server.boot_id:
            raise KMeshFenceError("managed registration belongs to another server boot")
        if record is None or record.nonce != nonce:
            raise KMeshFenceError("managed registration nonce is unknown")
        if record.ownership.identity.owner_session != get_request_session():
            raise KMeshFenceError("managed registration belongs to another client")
        return record

    def _status(self, record: _Registration) -> ManagedRawRegistration:
        return ManagedRawRegistration(
            self.bindings.server.boot_id,
            record.nonce,
            record.instance_id,
            record.ownership.identity,
            record.state,
            tuple(
                sorted(
                    {
                        w.allocation_identity
                        for w in record.owners + record.rotary_owners
                    }
                )
            ),
        )

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def kmesh_register_raw_context(
        self,
        boot_id: str,
        nonce: str,
        instance_id: int,
        caches_ipc: list[DeviceIPCWrapper],
        model_name: str,
        layout: str,
    ) -> tuple[ManagedRawRegistration | None, str]:
        def register() -> ManagedRawRegistration:
            if boot_id != self.bindings.server.boot_id:
                raise KMeshFenceError(
                    "managed registration belongs to another server boot"
                )
            if (
                not re.fullmatch(r"[0-9a-f]{32}", nonce)
                or instance_id < 1
                or not 0 < len(model_name) <= 256
                or layout not in ("NHD", "HND")
                or not caches_ipc
                or any(type(w) is not RawCudaIPCWrapper for w in caches_ipc)
            ):
                raise KMeshFenceError("invalid managed raw registration metadata")
            owners = tuple(w for w in caches_ipc if isinstance(w, RawCudaIPCWrapper))
            signature = (
                model_name,
                layout,
                str(instance_id),
                *(w.view_identity for w in owners),
            )
            with self._managed_lock:
                if self._registration is not None:
                    record = self._record(boot_id, nonce)
                    if record.signature != signature:
                        raise KMeshFenceError("registration nonce metadata changed")
                    return self._status(record)
                record = _Registration(
                    nonce, instance_id, model_name, signature, owners
                )
                self._registration = record
                try:
                    cache = GPUCacheContext(
                        list(owners),
                        self.context.chunk_size,
                        {"kv_layout": "NHD" if layout == "NHD" else "HND"},
                        engine_type=EngineType.VLLM,
                        managed_raw=True,
                    )
                    record.cache = cache
                    if any(
                        t.dtype != torch.bfloat16 or not t.is_contiguous()
                        for t in cache.kv_tensors
                    ) or any(
                        group.recurrent_state or group.sw_size_tokens != -1
                        for group in cache.kv_layer_groups_manager.kernel_groups
                    ):
                        raise KMeshFenceError(
                            "managed context requires full BF16 attention"
                        )
                    backend = get_event_ipc_backend(cache.device)
                    backend.check_event_support(cache.device)
                    entry = ContextEntry(
                        cache,
                        model_name,
                        1,
                        event_backend=backend,
                        ownership=record.ownership,
                    )
                    record.entry = entry
                    self._transfer.adopt_context_entry(instance_id, entry)
                    record.state = "registered"
                    return self._status(record)
                except BaseException:
                    record.state = "poisoned"
                    raise

        return control_reply(register)

    @request_handler()
    def kmesh_raw_context_status(
        self, boot_id: str, nonce: str
    ) -> tuple[ManagedRawRegistration | None, str]:
        def status() -> ManagedRawRegistration:
            with self._managed_lock:
                return self._status(self._record(boot_id, nonce))

        return control_reply(status)

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
        def prepare() -> RegisteredRuntime:
            with self._managed_lock:
                record = self._registration
                if (
                    record is None
                    or record.ownership.identity != context
                    or context.owner_session != get_request_session()
                    or record.instance_id != instance_id
                    or record.state != "registered"
                    or not caches_ipc
                    or any(type(w) is not RawCudaIPCWrapper for w in caches_ipc)
                ):
                    raise KMeshFenceError(
                        "raw rotary preparation lost its registration"
                    )
                owners = tuple(
                    w for w in caches_ipc if isinstance(w, RawCudaIPCWrapper)
                )
                signature = hashlib.sha256(
                    json.dumps(
                        (
                            tuple(w.view_identity for w in owners),
                            head_size,
                            is_neox_style,
                            group_to_cache,
                            group_rot,
                            group_head_size,
                            kv_quant,
                            producer_event.hex(),
                        ),
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                if record.rotary_signature is not None:
                    if (
                        signature == record.rotary_signature
                        and record.runtime is not None
                    ):
                        return record.runtime
                    raise KMeshFenceError(
                        "raw rotary attempt cannot be replaced; retire context"
                    )
                # Own every declared input before bindings or a mapping attempt.
                record.rotary_owners = owners
                record.rotary_signature = signature
                runtime = self.bindings.register_rotary_snapshot(
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
                record.runtime = runtime
                return runtime

        return control_reply(prepare)

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def kmesh_retire_raw_context(
        self, boot_id: str, nonce: str, context: ContextIdentity
    ) -> tuple[ManagedRawRelease | None, str]:
        def retire() -> ManagedRawRelease:
            with self._managed_lock:
                record = self._record(boot_id, nonce)
                if record.ownership.identity != context:
                    raise KMeshFenceError(
                        "managed retirement context generation changed"
                    )
                if record.release is not None:
                    return record.release
                if record.state == "poisoned":
                    raise KMeshFenceError(
                        "managed construction requires importer teardown"
                    )
                record.state = "retiring"
                self.bindings.close()
                record.ownership.retire(0)
                expected_rotary = record.runtime.rotary if record.runtime else None
                if not record.rotary_retired:
                    if self._blend.rope_identity(record.instance_id) != expected_rotary:
                        record.state = "poisoned"
                        raise KMeshFenceError("managed rotary generation changed")
                    self._blend.drop_rope_state(record.instance_id)
                    record.rotary_retired = True
                cache, entry = record.cache, record.entry
                if cache is None or entry is None:
                    record.state = "poisoned"
                    raise KMeshFenceError(
                        "managed context is not completely constructed"
                    )
                released = cache.close_raw_checked()
                if set(released) != {w.allocation_identity for w in record.owners}:
                    record.state = "poisoned"
                    raise KMeshFenceError(
                        "managed release allocation identities changed"
                    )
                for owner in record.rotary_owners:
                    owner.retire_released_raw_imports()
                try:
                    self._transfer.detach_context_entry(record.instance_id, entry)
                    receipt = ManagedRawRelease(
                        boot_id,
                        nonce,
                        record.instance_id,
                        context,
                        expected_rotary,
                        self._status(record).allocation_ids,
                    )
                    record.release = receipt
                    record.state = "released"
                    record.cache = None
                    record.entry = None
                    return receipt
                except BaseException:
                    record.state = "poisoned"
                    raise

        return control_reply(retire)

    def close(self) -> None:
        """Refuse implicit retirement; only the owning client obtains its receipt.

        An unreleased or poisoned registration must remain retained until the
        importing process ends. Server shutdown cannot manufacture a reply.
        """
        with self._managed_lock:
            self.bindings.close()
            if self._registration is not None and self._registration.release is None:
                raise KMeshFenceError(
                    "managed registration requires client retirement or importer exit"
                )
            self._blend.close()
            self._transfer.close()

    def _span_executor(self, fence: WorkerFence) -> KMeshSpanStore:
        with self._managed_lock:
            record = self._registration
            if (
                self._spans is None
                or record is None
                or record.state != "registered"
                or record.instance_id != fence.instance_id
                or record.ownership.identity.owner_session != get_request_session()
            ):
                raise KMeshFenceError("managed span execution is not admitted")
            return self._spans
