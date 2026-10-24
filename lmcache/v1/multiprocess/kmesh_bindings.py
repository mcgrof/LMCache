# SPDX-License-Identifier: Apache-2.0
"""Bind KMesh leases and accepted work to actual registered server resources.

This server-side boundary does not expose RPCs, serve storage objects or arm an
engine. Device executors must report physical completion; client cancellation
and request END are not completion reports.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
import hashlib
import json
import threading

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.multiprocess.context_ownership import ContextIdentity
from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper
from lmcache.v1.multiprocess.kmesh_device_copy import RetainedTensorCopies
from lmcache.v1.multiprocess.kmesh_domains import (
    Completion,
    DomainRegistry,
    KMeshFenceError,
    OperationTicket,
    RegistrationLease,
    RegistrationState,
    SemanticIdentity,
    ServerDescriptor,
    WorkerFence,
)
from lmcache.v1.multiprocess.kmesh_rotary_snapshot import (
    canonical_rope_cache_facts,
    normalize_rope_snapshot,
)
from lmcache.v1.multiprocess.kmesh_types import RegisteredRuntime as RegisteredRuntime
from lmcache.v1.multiprocess.modules.blend.rope import (
    BorrowedRope,
    CanonicalRopeSnapshot,
    RopeIdentity,
    provisional_rope,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import BorrowedContext
from lmcache.v1.multiprocess.transport.request_context import get_request_session

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.modules.blend import BlendModule
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
        LMCacheDrivenTransferModule,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _registered_layout_digest(paged: BorrowedContext, rotary: BorrowedRope) -> str:
    cache = paged.entry.cache_context
    rope = rotary.state
    formats = {
        "NL_X_TWO_NB_NH_BS_HS",
        "NL_X_TWO_NB_BS_NH_HS",
        "NL_X_NB_NH_BS_TWO_HS",
        "NL_X_NB_NH_BS_CS",
        "NL_X_NB_BS_NH_CS",
    }
    if rope.kv_quant or rope.is_neox_style is not True:
        raise KMeshFenceError("binding requires unquantized NeoX rotary state")
    if not cache.kv_tensors or any(not t.is_contiguous() for t in cache.kv_tensors):
        raise KMeshFenceError("binding requires contiguous paged tensors")
    manager = cache.kv_layer_groups_manager
    groups = []
    for index, group in enumerate(manager.kernel_groups):
        shape = group.shape_desc
        fmt = cache.get_engine_kv_format(index).name
        head_size = rope.head_size_for_group(group.engine_group_idx)
        # Native descriptors use zero for the tight, format-specific default.
        tight_stride = shape.bs * shape.nh * shape.hs
        block_stride = (
            shape.block_stride_elems if shape.block_stride_elems > 0 else tight_stride
        )
        if (
            fmt not in formats
            or group.dtype != torch.bfloat16
            or group.recurrent_state
            or group.sw_size_tokens != -1
            or group.extra_object_group_tag
            or group.tokens_per_block != group.slots_per_block
            or block_stride != tight_stride
            or head_size <= 0
            or head_size % 2
            or rope.rot_for_group(group.engine_group_idx, group.dtype) != (0, head_size)
            or shape.hs != head_size * (2 if shape.kv_size == 1 else 1)
        ):
            raise KMeshFenceError("unsupported registered KV/rotary geometry")
        rotary_cache = rope.cache_for_group(group.engine_group_idx)
        if (
            rotary_cache is None
            or rotary_cache.ndim != 2
            or rotary_cache.shape[1] != head_size
            or rotary_cache.shape[0] < 1
        ):
            raise KMeshFenceError("registered rotary cache does not cover its group")
        groups.append(
            (
                tuple(group.layer_indices),
                group.engine_group_idx,
                fmt,
                str(group.dtype),
                shape.kv_size,
                shape.nl,
                shape.bs,
                shape.nh,
                shape.hs,
                shape.element_size,
                block_stride,
                group.tokens_per_block,
                group.slots_per_block,
            )
        )
    if not groups or any(
        g.aux or g.recurrent or g.sw_size_chunks != -1 for g in manager.object_groups
    ):
        raise KMeshFenceError("binding requires full-attention object groups")
    return _digest(
        (
            "kmesh-layout-v1",
            groups,
            [tuple(g.kernel_group_indices) for g in manager.object_groups],
        )
    )


def _runtime(
    paged: BorrowedContext,
    rotary: BorrowedRope,
    context: ContextIdentity,
    rope_identity: RopeIdentity,
    chunk_tokens: int,
    hash_algorithm: str,
) -> RegisteredRuntime:
    cache = paged.entry.cache_context
    rope = rotary.state
    snapshot = rope.canonical_snapshot
    device = any(t.device.type != "cpu" for t in cache.kv_tensors + rope.cos_sin_caches)
    if device and snapshot is None:
        raise KMeshFenceError(
            "device runtime discovery requires a qualified copy/drain backend"
        )
    if snapshot is not None and (
        type(snapshot) is not CanonicalRopeSnapshot
        or snapshot.context != context
        or snapshot.rotary != rope_identity.rotary
        or snapshot.rotary != rope.ownership.identity
        or len(snapshot.cache_facts) != len(rope.cos_sin_caches)
        or any(t.device != cache.device for t in cache.kv_tensors + rope.cos_sin_caches)
        or any(
            (str(t.dtype), tuple(t.shape)) != fact[:2]
            for t, fact in zip(rope.cos_sin_caches, snapshot.cache_facts, strict=True)
        )
    ):
        raise KMeshFenceError("canonical rotary generation or tensor metadata changed")
    if hash_algorithm != "blake3":
        raise KMeshFenceError(
            "binding requires the stable MP BLAKE3 hash implementation"
        )
    layout_digest = _registered_layout_digest(paged, rotary)
    caches = (
        snapshot.cache_facts
        if device and snapshot is not None
        else canonical_rope_cache_facts(rope.cos_sin_caches)
    )
    if snapshot is not None and caches != snapshot.cache_facts:
        raise KMeshFenceError("canonical CPU rotary contents changed")
    return RegisteredRuntime(
        context,
        rope_identity,
        paged.entry.model_name,
        paged.entry.world_size,
        layout_digest,
        _digest(
            (
                "kmesh-rope-v1",
                rope.head_size,
                rope.is_neox_style,
                rope.group_to_cache,
                rope.group_rot,
                rope.group_head_size,
                caches,
            )
        ),
        "bfloat16",
        chunk_tokens,
        hash_algorithm,
    )


@dataclass(frozen=True)
class BoundOperation:
    """Keep exact server resources alive until a physical completion report."""

    ticket: OperationTicket
    paged: BorrowedContext
    rotary: BorrowedRope


@dataclass
class _BoundWorker:
    fence: WorkerFence
    runtime: RegisteredRuntime
    paged: BorrowedContext
    rotary: BorrowedRope


class RegisteredWorkerBindings:
    """Own registration holds separately from accepted-operation obligations.

    The registry's handler inventory must come from the actual server, and
    workers must independently supply immutable model/tokenizer semantics.
    These bindings verify registered layout and rotary contents, not weights
    that the storage server does not load. Legacy device registrations cannot
    supply discovery evidence. Ordered device snapshots require explicit opt-in.
    Storage and engine admission remain separate until their namespace and
    completion paths are qualified.
    """

    def __init__(
        self,
        registry: DomainRegistry,
        transfer: "LMCacheDrivenTransferModule",
        blend: "BlendModule",
        *,
        allow_device_snapshots: bool = False,
    ) -> None:
        self._registry = registry
        self._transfer = transfer
        self._blend = blend
        self._lock = threading.RLock()
        self._workers: dict[str, _BoundWorker] = {}
        self._leases: dict[str, RegistrationLease] = {}
        self._operations: dict[str, BoundOperation] = {}
        self._copies = RetainedTensorCopies()
        self._preparations: set[ContextIdentity] = set()
        self._allow_device_snapshots = allow_device_snapshots
        self._closing = False

    @property
    def server(self) -> ServerDescriptor:
        """Read the registry's boot, store and capability generation."""
        return self._registry.server

    def _borrow(
        self,
        instance_id: int,
        runtime: RegisteredRuntime | None,
        revoke: Callable[[], None],
    ) -> tuple[BorrowedContext, BorrowedRope, ContextIdentity, RopeIdentity]:
        if self._closing:
            raise KMeshFenceError("server bindings are closing")
        context = (
            runtime.context if runtime else self._transfer.context_identity(instance_id)
        )
        rotary = runtime.rotary if runtime else self._blend.rope_identity(instance_id)
        if context is None or rotary is None:
            raise KMeshFenceError("worker has no complete registered context")
        paged = self._transfer.borrow_context(instance_id, context, revoke)
        try:
            rope = self._blend.borrow_rope(instance_id, rotary, context, revoke)
        except BaseException:
            paged.release()
            raise
        return paged, rope, context, rotary

    def describe(self, instance_id: int) -> RegisteredRuntime:
        """Derive facts from the actual client's currently registered resources.

        Args:
            instance_id: Worker whose paged and rotary registrations to inspect.

        Returns:
            Immutable geometry and content evidence, excluding placement/capacity.

        Raises:
            KMeshFenceError: Unsupported geometry or retirement during discovery.
            ContextOwnershipError: Missing, foreign or stale registrations.
        """
        revoked = threading.Event()
        paged, rotary, context, identity = self._borrow(instance_id, None, revoked.set)
        try:
            runtime = _runtime(
                paged,
                rotary,
                context,
                identity,
                self._transfer.context.chunk_size,
                self._transfer.context.token_hasher.hash_algorithm_name,
            )
            if revoked.is_set():
                raise KMeshFenceError("registration retired during runtime discovery")
            return runtime
        finally:
            rotary.release()
            paged.release()

    def context_identity(self, instance_id: int) -> ContextIdentity:
        """Read the actual client's paged generation without granting an operation."""
        context = self._transfer.context_identity(instance_id)
        if context is None:
            raise KMeshFenceError("worker has no registered paged context")
        paged = self._transfer.borrow_context(instance_id, context, lambda: None)
        try:
            with self._lock:
                if self._closing:
                    raise KMeshFenceError("server bindings are closing")
            return context
        finally:
            paged.release()

    def register_rotary_snapshot(
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
    ) -> RegisteredRuntime:
        """Snapshot inputs before normalization and publish completed private state.

        The worker must retain its inputs without writes until server physical
        completion, including after a failed request. The producer event orders
        earlier writes; it does not enforce that immutability contract.

        Args:
            instance_id: Actual client's registered worker instance.
            context: Expected exact paged generation from context discovery.
            caches_ipc: Raw worker rotary mappings; CUDA currently requires
                PyTorch IPC rather than raw-driver or virtual-memory wrappers.
            head_size: Model-wide rotary width, positive and even.
            is_neox_style: Must be true for contiguous cosine/sine halves.
            group_to_cache: Engine-group to rotary-cache indices.
            group_rot: Optional per-group offset/width windows.
            group_head_size: Optional per-group head widths.
            kv_quant: Must be empty for this unquantized profile.
            producer_event: Event recorded after the last device write.

        Returns:
            Server-derived facts for the completed exact rotary generation.

        Raises:
            KMeshFenceError: Unsupported geometry, stale context, concurrent
                preparation, closing admission or uncertain device completion.
            ContextOwnershipError: Missing, foreign or retired paged generation.
        """
        if (
            type(context) is not ContextIdentity
            or context.owner_session != get_request_session()
        ):
            raise KMeshFenceError("rotary preparation belongs to another client")
        with self._lock:
            if self._closing:
                raise KMeshFenceError("server bindings are closing")
            if context in self._preparations or self._copies.pending_for(context):
                raise KMeshFenceError("previous rotary preparation must drain")
            self._preparations.add(context)
        paged: BorrowedContext | None = None
        provisional: BorrowedRope | None = None
        try:
            revoked = threading.Event()
            paged = self._transfer.borrow_context(instance_id, context, revoked.set)
            cache = paged.entry.cache_context
            if cache.device.type != "cpu" and not self._allow_device_snapshots:
                raise KMeshFenceError(
                    "ordered device snapshots require explicit opt-in"
                )
            if cache.device.type != "cpu":
                # First Party
                from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper

                if (
                    cache.device.type != "cuda"
                    or torch_device_type != "cuda"
                    or any(
                        type(wrapper) is not CudaIPCWrapper for wrapper in caches_ipc
                    )
                ):
                    raise KMeshFenceError(
                        "ordered device snapshots currently require PyTorch CUDA IPC"
                    )
            if (
                type(head_size) is not int
                or head_size <= 0
                or head_size % 2
                or is_neox_style is not True
                or kv_quant != ""
                or not caches_ipc
            ):
                raise KMeshFenceError(
                    "ordered snapshots require full unquantized NeoX state"
                )
            raw = [wrapper.to_tensor() for wrapper in caches_ipc]
            if any(
                t.device != cache.device
                or t.ndim != 2
                or not t.shape[0]
                or t.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                or not t.is_contiguous()
                for t in raw
            ):
                raise KMeshFenceError("invalid rotary input tensor metadata")
            if group_to_cache and (
                min(group_to_cache) < -1 or max(group_to_cache) >= len(raw)
            ):
                raise KMeshFenceError("rotary cache mapping is out of range")
            rotations: list[tuple[int, int] | None] = []
            for window in group_rot:
                if len(window) == 0:
                    rotations.append(None)
                elif len(window) == 2 and window[0] >= 0 and window[1] > 0:
                    rotations.append((window[0], window[1]))
                else:
                    raise KMeshFenceError("invalid rotary window")
            provisional = provisional_rope(
                context,
                raw,
                head_size,
                is_neox_style,
                group_to_cache,
                rotations,
                group_head_size,
                kv_quant,
                tuple(caches_ipc),
            )
            _registered_layout_digest(paged, provisional)
            copy_context = self._transfer.borrow_context(
                instance_id, context, revoked.set
            )
            consumed, provisional = provisional, None
            host = self._copies.copy_to_cpu(copy_context, consumed, producer_event)
            normalized = [normalize_rope_snapshot(t) for t in host]
            with self._lock:
                if self._closing or revoked.is_set():
                    raise KMeshFenceError("rotary preparation was retired")
            provisional = provisional_rope(
                context,
                normalized,
                head_size,
                is_neox_style,
                group_to_cache,
                rotations,
                group_head_size,
                kv_quant,
            )
            upload_context = self._transfer.borrow_context(
                instance_id, context, revoked.set
            )
            consumed, provisional = provisional, None
            targets = self._copies.copy_to_device(upload_context, consumed)
            provisional = provisional_rope(
                context,
                targets,
                head_size,
                is_neox_style,
                group_to_cache,
                rotations,
                group_head_size,
                kv_quant,
            )
            identity = RopeIdentity(provisional.state.ownership.identity, context)
            provisional.state.canonical_snapshot = CanonicalRopeSnapshot(
                context, identity.rotary, canonical_rope_cache_facts(normalized)
            )
            runtime = _runtime(
                paged,
                provisional,
                context,
                identity,
                self._transfer.context.chunk_size,
                self._transfer.context.token_hasher.hash_algorithm_name,
            )
            with self._lock:
                if self._closing or revoked.is_set():
                    raise KMeshFenceError("rotary preparation was retired")
            self._blend.publish_rope_snapshot(instance_id, context, provisional)
            if revoked.is_set():
                raise KMeshFenceError(
                    "paged context retired during snapshot publication"
                )
            return runtime
        finally:
            if provisional is not None:
                provisional.release()
            if paged is not None:
                paged.release()
            with self._lock:
                self._preparations.remove(context)

    def retry_rotary_drain(self, context: ContextIdentity) -> int:
        """Request a physical copy drain for the actual client's exact context.

        Returns:
            Remaining copies for this generation, excluding exporter teardown.

        Raises:
            KMeshFenceError: Foreign context or no known copy obligation.
        """
        if context.owner_session != get_request_session():
            raise KMeshFenceError("rotary drain belongs to another client")
        if not self._copies.pending_for(context):
            raise KMeshFenceError("no retained copy for this exact context")
        return self._copies.drain_quarantined(context)

    def open_registration(
        self,
        semantics: SemanticIdentity,
        engine_run: str,
        world_size: int,
        execution_digest: str,
    ) -> RegistrationLease:
        """Create a rank barrier for independently supplied worker semantics."""
        with self._lock:
            if self._closing:
                raise KMeshFenceError("server bindings are closing")
            self._release_revoked()
            domain = self._registry.open_domain(semantics)
            lease = self._registry.open_registration(
                domain, engine_run, world_size, execution_digest
            )
            self._leases[lease.registration_id] = lease
            return lease

    def bind(
        self,
        lease: RegistrationLease,
        rank: int,
        instance_id: int,
        expected: RegisteredRuntime,
        semantics: SemanticIdentity,
        execution_digest: str,
    ) -> WorkerFence:
        """Bind one actual worker and retain its paged and rotary registrations.

        Args:
            lease: Current domain registration and required rank barrier.
            rank: Unique rank within that barrier.
            instance_id: Actual receiving client's registered worker.
            expected: Previously observed exact registration generations/facts.
            semantics: Independently derived worker model and KV semantics.
            execution_digest: Frozen worker execution settings.

        Returns:
            Server-issued fence bound to this actual receiving client.

        Raises:
            KMeshFenceError: Stale lease, mismatched facts or inconsistent rank.
            ContextOwnershipError: Foreign, missing or retiring registrations.
        """
        session = get_request_session()
        with self._lock:
            self._release_revoked()
            if self._registry.state(lease) == RegistrationState.REVOKED:
                raise KMeshFenceError("registration is revoked")
            paged, rotary, context, identity = self._borrow(
                instance_id, expected, lambda: self.revoke(lease)
            )
            try:
                actual = _runtime(
                    paged,
                    rotary,
                    context,
                    identity,
                    self._transfer.context.chunk_size,
                    self._transfer.context.token_hasher.hash_algorithm_name,
                )
                if (
                    actual != expected
                    or actual.world_size != lease.world_size
                    or actual.layout_digest != semantics.layout_digest
                    or actual.rope_digest != semantics.rope_digest
                    or actual.kv_dtype != semantics.kv_dtype
                    or actual.chunk_tokens != semantics.chunk_tokens
                    or actual.hash_algorithm != semantics.hash_algorithm
                ):
                    raise KMeshFenceError(
                        "worker semantics disagree with registered runtime"
                    )
                fence = self._registry.bind_worker(
                    lease, rank, instance_id, session, semantics, execution_digest
                )
                self._workers[fence.token] = _BoundWorker(fence, actual, paged, rotary)
                return fence
            except BaseException:
                rotary.release()
                paged.release()
                self.revoke(lease)
                raise

    def begin(
        self,
        fence: WorkerFence,
        request_id: str,
        generation: int,
    ) -> BoundOperation:
        """Accept a transaction while retaining its exact execution resources."""
        session = get_request_session()
        with self._lock:
            self._release_revoked()
            worker = self._workers.get(fence.token)
            if worker is None or worker.fence != fence:
                raise KMeshFenceError("worker resources are no longer bound")
            if (
                self._transfer.context.chunk_size != worker.runtime.chunk_tokens
                or self._transfer.context.token_hasher.hash_algorithm_name
                != worker.runtime.hash_algorithm
            ):
                self.revoke(fence.lease)
                raise KMeshFenceError("server chunk/hash schema changed")
            paged, rotary, _, _ = self._borrow(
                fence.instance_id, worker.runtime, lambda: self.revoke(fence.lease)
            )
            try:
                ticket = self._registry.acquire(fence, session, request_id, generation)
            except BaseException:
                rotary.release()
                paged.release()
                raise
            operation = BoundOperation(ticket, paged, rotary)
            self._operations[ticket.token] = operation
            return operation

    def publish(self, operation: BoundOperation, commit: Callable[[str], None]) -> None:
        """Publish already-completed data under the current fenced namespace.

        Use the retained server operation's accepting session, so device
        callbacks need no invented receiving ContextVar. Neither this operation
        object nor a physical-completion report is accepted from a client RPC.
        The callback must synchronously publish and must not re-enter bindings.
        """
        with self._lock:
            self._operation(operation)
            try:
                self._registry.publish(
                    operation.ticket, operation.ticket.fence.session_id, commit
                )
            finally:
                self._release_revoked()

    def finish(self, operation: BoundOperation, completion: Completion) -> None:
        """Report server-observed physical completion, never client cancellation.

        FAILED_UNDRAINED retains both resources and its ticket. A later drained
        report can release them after revocation, without authorizing publication
        or rearming the lease. Device executors own this report; there is no
        client-facing completion method.
        """
        with self._lock:
            self._operation(operation)
            self._registry.finish(operation.ticket, completion)
            if completion != Completion.FAILED_UNDRAINED:
                del self._operations[operation.ticket.token]
                operation.rotary.release()
                operation.paged.release()
            self._release_revoked()

    def revoke(self, lease: RegistrationLease) -> None:
        """Disarm and release registration holds, retaining every accepted ticket."""
        with self._lock:
            self._registry.revoke(lease)
            self._release_revoked()

    def state(self, lease: RegistrationLease) -> RegistrationState:
        """Read the rank barrier and release expired registration holds."""
        with self._lock:
            self._release_revoked()
            return self._registry.state(lease)

    def pending(self, lease: RegistrationLease) -> int:
        """Count accepted transactions still requiring physical completion."""
        return self._registry.pending(lease)

    def invalidate_store(self) -> None:
        """Disarm a logical store generation without releasing physical work."""
        with self._lock:
            self._registry.invalidate_store()
            self._release_revoked()

    def update_capabilities(self, operations: list[str]) -> None:
        """Disarm bindings if the actual server's handler inventory changes."""
        with self._lock:
            self._registry.update_capabilities(operations)
            self._release_revoked()

    def heartbeat(self, fence: WorkerFence) -> None:
        """Refresh a live bound worker using its actual receiving client."""
        with self._lock:
            self._release_revoked()
            worker = self._workers.get(fence.token)
            if worker is None or worker.fence != fence:
                raise KMeshFenceError("worker resources are no longer bound")
            self._registry.heartbeat(fence, get_request_session())

    def reap_expired(self) -> None:
        """Release expired registration holds while retaining physical debt."""
        with self._lock:
            self._release_revoked()

    def clear_store(self, clear: Callable[[], None]) -> None:
        """Serialize admission through clear, refusing any accepted work.

        Advance the logical generation before attempting physical clear. Keep
        admission serialized until the callback returns, including on failure,
        so a new registration cannot publish while storage is still clearing.
        """
        with self._lock:
            if self._closing:
                raise KMeshFenceError("server bindings are closing")
            self.invalidate_store()
            if self._operations or self._preparations or self._copies.pending:
                raise KMeshFenceError("accepted KMesh work must drain before clear")
            clear()

    def close(self) -> None:
        """Revoke registrations and refuse shutdown while physical debt remains."""
        with self._lock:
            self._closing = True
            for lease in self._leases.values():
                self._registry.revoke(lease)
            for worker in tuple(self._workers.values()):
                self._registry.revoke(worker.fence.lease)
            self._release_revoked()
            if self._operations or self._preparations or self._copies.pending:
                raise KMeshFenceError("accepted KMesh work must drain before shutdown")
        self._copies.close()

    def _operation(self, operation: BoundOperation) -> None:
        if self._operations.get(operation.ticket.token) is not operation:
            raise KMeshFenceError("operation is not retained by this server binding")

    def _release_revoked(self) -> None:
        for token, worker in tuple(self._workers.items()):
            if self._registry.state(worker.fence.lease) == RegistrationState.REVOKED:
                del self._workers[token]
                worker.rotary.release()
                worker.paged.release()
