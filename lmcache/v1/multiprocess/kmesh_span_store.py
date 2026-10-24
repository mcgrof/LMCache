# SPDX-License-Identifier: Apache-2.0
"""Execute bounded immutable-span transfers against accepted server resources."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field, replace
import hashlib
import json
import re
import threading

# Third Party
import blake3
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    ObjectKey,
    PrefetchHandle,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.storage_manager import (
    L1TransferLeases,
    L1WriteCompletion,
    StorageManager,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import CBMatchResult
from lmcache.v1.multiprocess.kmesh_bindings import (
    BoundOperation,
    RegisteredWorkerBindings,
)
from lmcache.v1.multiprocess.kmesh_domains import (
    Completion,
    KMeshFenceError,
    WorkerFence,
)
from lmcache.v1.multiprocess.kmesh_types import SpanHandle, SpanOperationStatus
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.modules.blend.retrieve import RegisteredRetrievePlan
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import get_layout_desc
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.transport.request_context import get_request_session
from lmcache.v1.platform.stream import synchronize_device
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _checksum(obj: MemoryObj) -> str:
    tensor = obj.get_tensor(0)
    if tensor is None or tensor.device.type != "cpu":
        raise KMeshFenceError("span requires actual CPU source bytes")
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class _Source:
    handle: SpanHandle
    keys: tuple[ObjectKey, ...]
    checksums: tuple[str, ...]


@dataclass
class _Work:
    status: SpanOperationStatus
    signature: str
    operation: BoundOperation
    objects: dict[ObjectKey, MemoryObj] = field(default_factory=dict)
    completion: L1WriteCompletion = field(default_factory=list)
    leases: L1TransferLeases = field(default_factory=list)
    prefetch: PrefetchHandle | None = None
    requested: list[ObjectKey] = field(default_factory=list)
    read_keys: list[ObjectKey] = field(default_factory=list)
    read_owners: dict[ObjectKey, int] = field(default_factory=dict)
    staged_blocks: list[torch.Tensor] = field(default_factory=list)
    producer: object | None = None
    plan: RegisteredRetrievePlan | None = None
    device_touched: bool = False
    error: BaseException | None = None
    settled: bool = False


# A partial launch must outlive transport/server teardown. Preserve the executor,
# allocator, planner, borrowed resources and exception frames until actual drain.
_RETAINED_SPAN_WORK: dict[str, tuple["KMeshSpanStore", _Work]] = {}
_RETAINED_LOCK = threading.Lock()


class KMeshSpanStore:
    """Serialize native CPU-L1 span transfers for one managed CUDA context.

    Args:
        bindings: Actual server's registration and accepted-resource owner.
        storage: CPU L1 manager using lazy pinned allocation and no L2 adapters.
        blend: Native positional planner retaining its private staging buffers.

    Operation results are immutable receipts scoped to the receiving session,
    request generation and block-table digest. Callers must reserve those actual
    engine blocks until a drained result arrives. Loaded-needs-repair does not
    authorize model admission. This service does not supply model repair or GDS.
    """

    def __init__(
        self,
        bindings: RegisteredWorkerBindings,
        storage: StorageManager,
        blend: BlendModule,
    ) -> None:
        self._bindings, self._storage, self._blend = bindings, storage, blend
        self._execution = threading.Lock()
        self._records_lock = threading.Lock()
        self._records: dict[tuple[str, str, int], _Work] = {}
        self._sources: dict[str, _Source] = {}

    def store(
        self,
        fence: WorkerFence,
        request_id: str,
        generation: int,
        token_ids: list[int],
        source_start: int,
        context_digest: str,
        block_ids: list[list[int]],
        producer_event: bytes,
    ) -> SpanOperationStatus:
        """Gather complete source chunks, admit each key, then publish a handle.

        Args:
            fence: Exact bound worker and receiving transport session.
            request_id: Logical request whose generation is consumed once.
            generation: Strictly increasing positive request generation.
            token_ids: Nonnegative source token IDs in complete chunk order.
            source_start: Chunk-aligned position in the original sequence.
            context_digest: SHA-256 identity of the source preceding context.
            block_ids: Full source sequence's private engine block table.
            producer_event: Recorded CUDA IPC event after its last source write.

        Returns:
            Stored or failed status; undrained work still owns its blocks.

        Raises:
            KMeshFenceError: Invalid metadata, stale ownership or changed retry.
        """
        self._metadata(fence, context_digest, block_ids, producer_event)
        chunk = fence.lease.domain.semantics.chunk_tokens
        if (
            type(source_start) is not int
            or source_start < 0
            or source_start % chunk
            or not token_ids
            or len(token_ids) % chunk
            or any(type(t) is not int or not 0 <= t < 2**63 for t in token_ids)
        ):
            raise KMeshFenceError("span store requires aligned complete token chunks")
        signature = _digest(
            (
                "store",
                token_ids,
                source_start,
                context_digest,
                block_ids,
                producer_event.hex(),
            )
        )

        def execute(work: _Work) -> SpanHandle:
            cache = work.operation.paged.entry.cache_context
            self._geometry(work, source_start, len(token_ids), block_ids)
            namespace = fence.lease.domain.namespace
            keys = [
                ObjectKey(
                    blake3.blake3(
                        f"kmesh-span-v1:{namespace}:{work.status.operation_id}:{i}".encode()
                    ).digest(),
                    f"kmesh-{namespace}",
                    0,
                )
                for i in range(len(token_ids) // chunk)
            ]
            layout = get_layout_desc(cache, chunk, 0)
            work.objects = self._storage.reserve_write(keys, layout)
            work.completion = self._storage.prepare_write_completion(work.objects)
            if set(work.objects) != set(keys):
                raise KMeshFenceError("span write reservation is incomplete")
            work.leases = self._storage.acquire_write_transfer_leases(
                work.objects, all_buffers=True
            )
            self._wait_producer(work, producer_event)
            block_size = cache.kv_layer_groups_manager.kernel_groups[0].tokens_per_block
            selected = [
                block_ids[0][
                    source_start // block_size : (source_start + len(token_ids))
                    // block_size
                ]
            ]
            with torch_dev.device(cache.device), torch_dev.stream(cache.stream):
                work.staged_blocks = downsample_and_stage_block_ids(cache, selected)
                transfer_kv_per_object_group(
                    cache,
                    work.staged_blocks,
                    [work.objects[k] for k in keys],
                    0,
                    1,
                    0,
                    lmcache_native.TransferDirection.D2H,
                    transfer_key=work.status.operation_id,
                    block_ids_host=selected,
                )
            self._drain_device(work)
            handle = SpanHandle(
                namespace,
                work.status.operation_id,
                source_start,
                len(token_ids),
                blake3.blake3(
                    json.dumps(token_ids, separators=(",", ":")).encode()
                ).hexdigest(),
                context_digest,
                fence.lease.domain.semantics.layout_digest,
                fence.lease.domain.semantics.rope_digest,
            )
            source = _Source(
                handle, tuple(keys), tuple(_checksum(work.objects[k]) for k in keys)
            )

            def publish(accepted: str) -> None:
                if accepted != namespace:
                    raise KMeshFenceError("span publication namespace changed")
                outcomes = self._storage.finish_write_with_status(work.completion)
                if set(outcomes) != set(keys) or any(
                    v != L1Error.SUCCESS for v in outcomes.values()
                ):
                    raise KMeshFenceError("span admission failed")
                self._sources[handle.span_id] = source

            self._bindings.publish(work.operation, publish)
            work.completion = []
            return handle

        return self._run(
            fence,
            request_id,
            generation,
            "store",
            block_ids,
            signature,
            execute,
            "stored",
        )

    def restore(
        self,
        fence: WorkerFence,
        request_id: str,
        generation: int,
        span: SpanHandle,
        destination_start: int,
        context_digest: str,
        block_ids: list[list[int]],
        producer_event: bytes,
    ) -> SpanOperationStatus:
        """Read verified canonical chunks and rotate/scatter into reserved blocks.

        Args:
            fence: Exact bound worker and receiving transport session.
            request_id: Logical destination request.
            generation: Positive monotonically increasing request generation.
            span: Exact server-published source handle, never client-invented.
            destination_start: First destination token position.
            context_digest: SHA-256 identity of its preceding context.
            block_ids: Full destination sequence's private engine block table.
            producer_event: Recorded IPC event after destination slots are ready.

        Returns:
            Drained loaded status, possibly requiring model repair, or failure.

        Raises:
            KMeshFenceError: Invalid metadata, ownership, source or changed retry.
        """
        self._metadata(fence, context_digest, block_ids, producer_event)
        if type(destination_start) is not int or destination_start < 0:
            raise KMeshFenceError("invalid span destination position")
        source = self._sources.get(span.span_id)
        semantics = fence.lease.domain.semantics
        if (
            source is None
            or source.handle != span
            or span.namespace != fence.lease.domain.namespace
            or span.layout_digest != semantics.layout_digest
            or span.rope_digest != semantics.rope_digest
        ):
            raise KMeshFenceError("span does not belong to this immutable namespace")
        signature = _digest(
            (
                "restore",
                span.span_id,
                destination_start,
                context_digest,
                block_ids,
                producer_event.hex(),
            )
        )

        def execute(work: _Work) -> SpanHandle:
            cache = work.operation.paged.entry.cache_context
            self._geometry(work, destination_start, span.token_count, block_ids)
            chunk = semantics.chunk_tokens
            work.requested = list(source.keys)
            work.prefetch = self._storage.submit_prefetch_task(
                PrefetchTaskSpec(
                    [
                        GroupedObjectKeys(
                            work.requested, 0, get_layout_desc(cache, chunk, 0)
                        )
                    ]
                ),
                work.status.operation_id,
                skip_l2=True,
            )
            self._collect_prefetch(work)
            if work.read_keys != work.requested:
                raise KMeshFenceError("span source was evicted or is incomplete")
            keys, objects = self._storage.unsafe_read(work.read_keys, work.read_owners)
            work.objects = dict(zip(keys, objects, strict=True))
            if keys != work.requested:
                raise KMeshFenceError("span source lost its retained objects")
            work.leases = self._storage.acquire_read_transfer_leases(
                work.objects, all_buffers=True
            )
            if tuple(_checksum(work.objects[k]) for k in keys) != source.checksums:
                raise KMeshFenceError("span source checksum mismatch")
            self._wait_producer(work, producer_event)
            matches = [
                CBMatchResult(
                    span.source_start + i * chunk,
                    span.source_start + (i + 1) * chunk,
                    destination_start + i * chunk,
                    destination_start + (i + 1) * chunk,
                    key.chunk_hash,
                )
                for i, key in enumerate(keys)
            ]
            work.plan = self._blend.prepare_registered_retrieve(
                work.operation.paged, work.operation.rotary, matches, objects, block_ids
            )
            work.plan.enqueue()
            self._drain_device(work)
            self._bindings.publish(work.operation, lambda _: None)
            return span

        terminal = (
            "loaded"
            if destination_start == span.source_start
            and context_digest == span.context_digest
            else "loaded-needs-repair"
        )
        return self._run(
            fence,
            request_id,
            generation,
            "restore",
            block_ids,
            signature,
            execute,
            terminal,
        )

    def status(
        self, fence: WorkerFence, request_id: str, generation: int
    ) -> SpanOperationStatus:
        """Return cached observed state for the exact receiving owner/generation.

        Args:
            fence: Exact original accepting fence, including after revocation.
            request_id: Accepted logical request identity.
            generation: Accepted request generation.

        Returns:
            Cached immutable status without waiting for the GPU execution lock.

        Raises:
            KMeshFenceError: Unknown or foreign work. Absence cannot prove drain.
        """
        with self._records_lock:
            return self._work(fence, request_id, generation).status

    def drain(
        self, fence: WorkerFence, request_id: str, generation: int
    ) -> SpanOperationStatus:
        """Retry actual physical cleanup of known failed work, never its transfer.

        Args:
            fence: Original accepting fence, including after revocation.
            request_id: Exact accepted request identity.
            generation: Exact accepted request generation.

        Returns:
            Cached terminal state or failed-undrained while cleanup remains owed.

        Raises:
            KMeshFenceError: Unknown or foreign work.
        """
        with self._execution:
            with self._records_lock:
                work = self._work(fence, request_id, generation)
            if not work.settled:
                self._fail(work)
            return work.status

    def _metadata(
        self,
        fence: WorkerFence,
        context_digest: str,
        block_ids: list[list[int]],
        producer_event: bytes,
    ) -> None:
        if (
            fence.session_id != get_request_session()
            or fence.lease.world_size != 1
            or fence.rank != 0
            or fence.lease.domain.semantics.key_schema != "kmesh-span-v1"
            or fence.lease.domain.semantics.hash_algorithm != "blake3"
            or type(context_digest) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", context_digest)
            or len(block_ids) != 1
            or not block_ids[0]
            or any(type(b) is not int or b < 0 for b in block_ids[0])
            or len(set(block_ids[0])) != len(block_ids[0])
            or type(producer_event) is not bytes
            or not producer_event
        ):
            raise KMeshFenceError("unsupported span metadata or receiving owner")

    def _geometry(
        self, work: _Work, start: int, count: int, blocks: list[list[int]]
    ) -> None:
        cache = work.operation.paged.entry.cache_context
        manager = cache.kv_layer_groups_manager
        chunk = cache.lmcache_tokens_per_chunk
        rope = work.operation.rotary.state
        if (
            cache.device.type != "cuda"
            or manager.num_object_groups != 1
            or len(manager.kernel_groups) != 1
            or manager.kernel_groups[0].dtype != torch.bfloat16
            or manager.kernel_groups[0].sw_size_tokens != -1
            or manager.kernel_groups[0].recurrent_state
            or rope.canonical_snapshot is None
        ):
            raise KMeshFenceError("unsupported span runtime geometry")
        block_size = manager.kernel_groups[0].tokens_per_block
        if (
            chunk % block_size
            or count % chunk
            or any(b >= cache.num_blocks for b in blocks[0])
            or start + count > len(blocks[0]) * block_size
            or any(start + count > t.shape[0] for t in rope.cos_sin_caches)
        ):
            raise KMeshFenceError("span coordinates exceed registered capacity")

    def _work(self, fence: WorkerFence, request_id: str, generation: int) -> _Work:
        if fence.session_id != get_request_session():
            raise KMeshFenceError("span operation belongs to another client")
        work = self._records.get((fence.token, request_id, generation))
        if work is None or work.status.fence != fence:
            raise KMeshFenceError("span operation is unknown")
        return work

    def _set_status(
        self, work: _Work, state: str, error: str = "", span: SpanHandle | None = None
    ) -> None:
        with self._records_lock:
            work.status = replace(
                work.status, state=state, error=error[:512], span=span
            )

    def _run(
        self,
        fence: WorkerFence,
        request_id: str,
        generation: int,
        kind: str,
        blocks: list[list[int]],
        signature: str,
        execute: Callable[[_Work], SpanHandle],
        terminal: str,
    ) -> SpanOperationStatus:
        with self._execution:
            key = (fence.token, request_id, generation)
            with self._records_lock:
                old = self._records.get(key)
                if old is not None:
                    if old.status.fence != fence or old.signature != signature:
                        raise KMeshFenceError("span retry metadata changed")
                    return old.status
                if any(not w.settled for w in self._records.values()):
                    raise KMeshFenceError(
                        "earlier span work still requires physical drain"
                    )
            operation = self._bindings.begin(fence, request_id, generation)
            work = _Work(
                SpanOperationStatus(
                    fence,
                    request_id,
                    generation,
                    operation.ticket.token,
                    operation.paged.entry.ownership.identity,
                    kind,
                    _digest(blocks),
                    "executing",
                    "",
                    None,
                ),
                signature,
                operation,
            )
            with self._records_lock:
                self._records[key] = work
            with _RETAINED_LOCK:
                _RETAINED_SPAN_WORK[work.status.operation_id] = (self, work)
            try:
                span = execute(work)
                self._cleanup(work)
                self._bindings.finish(operation, Completion.SUCCEEDED)
                self._release(work)
                self._set_status(work, terminal, span=span)
            except BaseException as error:
                # Retain traceback frames: preparation may have queued work before
                # returning its plan. Process interrupts owe the same cleanup.
                work.error = error
                logger.exception("span execution failed")
                self._fail(work)
                if not isinstance(error, Exception):
                    raise
            return work.status

    def _wait_producer(self, work: _Work, event: bytes) -> None:
        entry = work.operation.paged.entry
        cache, backend = entry.cache_context, entry.event_backend
        if backend is None:
            raise KMeshFenceError("span requires recorded device-event IPC")
        work.device_touched = True
        work.producer = backend.import_event(event, cache.device)
        backend.wait_event(work.producer, cache.stream)

    def _drain_device(self, work: _Work) -> None:
        if work.device_touched:
            synchronize_device(work.operation.paged.entry.cache_context.device)
            work.device_touched = False

    def _collect_prefetch(self, work: _Work) -> None:
        if work.prefetch is None:
            return
        if not self._storage.wait_prefetch_status(work.prefetch, timeout=10):
            raise KMeshFenceError("span prefetch has not completed")
        result = self._storage.query_prefetch_status(work.prefetch)
        if result is None:
            raise KMeshFenceError("span prefetch completion is unknown")
        work.read_keys = [
            k for i, k in enumerate(work.requested) if result.hit_cells[0].test(i)
        ]
        work.read_owners = result.l1_owners
        work.prefetch = None

    def _cleanup(self, work: _Work) -> None:
        self._drain_device(work)
        self._collect_prefetch(work)
        if work.completion:
            self._storage.discard_write_by_owner(work.completion)
        if work.read_keys:
            self._storage.finish_read_prefetched(
                work.read_keys, l1_owners=work.read_owners
            )
            work.read_keys = []
        if work.leases:
            self._storage.release_transfer_leases(work.leases)
            work.leases = []
        if work.completion:
            self._storage.delete_l1_keys(list(work.objects))
            work.completion = []
        work.objects.clear()
        work.plan = None
        work.staged_blocks.clear()
        work.producer = None

    def _fail(self, work: _Work) -> None:
        error = (
            str(work.error) if work.error is not None else "physical cleanup required"
        )
        self._bindings.revoke(work.status.fence.lease)
        try:
            self._cleanup(work)
        except Exception:
            logger.exception("span cleanup remains physically unresolved")
            self._bindings.finish(work.operation, Completion.FAILED_UNDRAINED)
            self._set_status(work, "failed-undrained", error)
            return
        self._bindings.finish(work.operation, Completion.CANCELLED_DRAINED)
        self._release(work)
        self._set_status(work, "failed-drained", error)

    def _release(self, work: _Work) -> None:
        work.settled = True
        work.error = None
        with _RETAINED_LOCK:
            del _RETAINED_SPAN_WORK[work.status.operation_id]
