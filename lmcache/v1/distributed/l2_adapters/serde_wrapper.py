# SPDX-License-Identifier: Apache-2.0
"""
SerdeL2AdapterWrapper: wraps an inner L2 adapter with a SerdeProcessor
so controllers see a plain ``L2AdapterInterface`` while data is
transparently serialized on store and deserialized on load.

Threading: the wrapper owns an internal poll thread that reacts to
inner-adapter and serde event notifiers and chains

    store : caller → serialize → inner.store            → signal store_efd
    load  : caller → inner.load → deserialize           → signal load_efd

Lookup / unlock / delete / eviction pass straight through to the inner
adapter (no serde transform involved).

Temp buffer lifecycle: temp byte buffers come from the injected
``L1Manager`` so the extra memory shows up in L1 accounting just like
the non-serde path's temporary KV buffers. For a store, the temp holds
serialized bytes; for a load, the temp catches the bytes L2 reads
before deserialize copies them into the caller-provided KV buffer.

Failure policy: all-or-nothing per submit. A partial temp-alloc
failure fails the whole task (``success=False`` for store, all-zeros
bitmap for load). This preserves the coarse-grained success semantic
of ``L2AdapterInterface`` and means the caller's lock / lifecycle
invariants don't need to change.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import enum
import select
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L2AdapterListener, L2StoreResult
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import (
    AdapterUsage,
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.serde import (
    SerdeProcessor,
    SerdeTaskId,
    make_temp_key,
    serialized_layout_desc,
)
from lmcache.v1.distributed.serde.multi import GroupSlotView, MemoryObjGroup
from lmcache.v1.distributed.storage_placement import (
    SplitTierManifest,
    StoragePlacementMode,
    derive_component_key,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import consume_fd, create_event_notifier

logger = init_logger(__name__)

_POLL_TIMEOUT_MS = 500


class _StorePhase(enum.Enum):
    SERIALIZE = enum.auto()
    INNER_STORE = enum.auto()


@dataclass
class _StoreTaskState:
    wrapped_id: L2TaskId
    keys: list[ObjectKey]
    temp_keys: list[ObjectKey]
    temp_objs: list[MemoryObj]
    phase: _StorePhase
    """SERIALIZE while temps are write-locked; INNER_STORE after the
    serialize→store transition. Only read on shutdown to pick the right
    lock-release path; assignment is done under ``self._lock``."""

    # ---- split-tier (KV_SPLIT_TIER placement) extras ----
    is_split_tier: bool = False
    """True iff this store is being routed through the V-only state
    machine.  When set, ``k_child_keys`` and ``v_child_keys`` are
    populated and used in place of the default logical-key flow."""

    k_child_keys: list[ObjectKey] = field(default_factory=list)
    """K child keys in L1 (one per logical key).  Allocated and
    written synchronously in ``submit_store_task``; survive after
    the inner L2 store completes (they are the canonical L1
    residents for V-only mode)."""

    v_child_keys: list[ObjectKey] = field(default_factory=list)
    """V child keys passed to the inner L2 adapter in place of the
    logical keys.  Derived deterministically from the logical keys
    via ``derive_component_key(logical, 'v')``."""


@dataclass
class _LoadTaskState:
    wrapped_id: L2TaskId
    keys: list[ObjectKey]
    dst_objs: list[MemoryObj]
    temp_keys: list[ObjectKey]
    temp_objs: list[MemoryObj]
    load_bitmap: Bitmap = field(default_factory=lambda: Bitmap(0))
    """Inner adapter's per-key load bitmap; populated in
    ``_drain_inner_load`` before the task transitions to the deserialize
    stage. ``Bitmap(0)`` means "not populated yet" — by the time
    ``_drain_deserialize`` reads it, this placeholder has been
    overwritten with the real bitmap."""

    # ---- split-tier (KV_SPLIT_TIER placement) extras ----
    is_split_tier: bool = False
    """True iff this load is going through the V-only state machine
    (inner adapter is loading V child blobs from L2; K child bytes
    are sourced from L1 and copied into the dst by the wrapper)."""

    k_child_keys: list[ObjectKey] = field(default_factory=list)
    """K child keys to look up in L1.  Derived deterministically
    from the caller's logical keys via ``derive_component_key`` so
    the load path can reassemble without requiring an explicit
    manifest pointer."""

    v_child_keys: list[ObjectKey] = field(default_factory=list)
    """V child keys passed to the inner L2 adapter in place of the
    logical keys."""


class SerdeL2AdapterWrapper(L2AdapterInterface):
    """L2 adapter that adds transparent serde on top of an inner adapter.

    Args:
        inner: The wrapped L2 adapter doing the actual storage.
        serde: The SerdeProcessor used to (de)serialize KV data.
        l1_manager: L1 manager used to allocate temp byte buffers.
    """

    def __init__(
        self,
        inner: L2AdapterInterface,
        serde: SerdeProcessor,
        l1_manager: L1Manager,
        placement_mode: StoragePlacementMode = StoragePlacementMode.KV_TOGETHER,
        split_tier_manifest: SplitTierManifest | None = None,
    ) -> None:
        super().__init__()
        self._inner = inner
        self._serde = serde
        self._l1_manager = l1_manager
        self._placement_mode = placement_mode
        # Always constructed; empty + unused when placement is
        # KV_TOGETHER.  When KV_SPLIT_TIER, the wrapper drives the
        # state-machine transitions on this manifest during store
        # and (PR-2c') load.
        # NB: use `is None` not `or` -- SplitTierManifest implements
        # __len__, so an empty manifest is falsy and `or` would
        # silently mint a fresh local manifest instead of using the
        # caller-provided one.  The bug bit Phase 2 pod testing.
        self._split_tier_manifest = (
            SplitTierManifest()
            if split_tier_manifest is None
            else split_tier_manifest
        )

        # Thread pool used to parallelize the per-key K-bytes copy in
        # _alloc_split_tier_children.  Sized to 4 by default; the
        # actual work is a torch tensor copy_ which releases the GIL,
        # so this scales with CPU count up to the producer batch
        # size.  Only allocated when split-tier placement is active
        # so KV_TOGETHER incurs zero overhead.
        self._split_tier_copy_pool: ThreadPoolExecutor | None = None
        if placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
            self._split_tier_copy_pool = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="serde-l2-st-kcopy",
            )

        # Our own notifiers for store/load completion. Lookup passes the
        # inner adapter's fd straight through (no chaining needed there).
        self._store_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Task-id space separate from inner's. Reverse maps let the
        # internal thread pair inner / serde completions back to our
        # wrapped task id.
        self._lock = threading.Lock()
        self._next_task_id: L2TaskId = 0
        self._store_tasks: dict[L2TaskId, _StoreTaskState] = {}
        self._load_tasks: dict[L2TaskId, _LoadTaskState] = {}
        self._serde_to_store: dict[SerdeTaskId, L2TaskId] = {}
        self._inner_to_store: dict[L2TaskId, L2TaskId] = {}
        self._inner_to_load: dict[L2TaskId, L2TaskId] = {}
        self._serde_to_load: dict[SerdeTaskId, L2TaskId] = {}

        # User-visible completion queues (drained by controller polls).
        self._completed_store: dict[L2TaskId, L2StoreResult] = {}
        self._completed_load: dict[L2TaskId, Bitmap] = {}

        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="serde-l2-wrapper",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Event fds
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        # Lookup doesn't touch serde; passing through the inner adapter's
        # fd avoids a useless thread-hop per lookup.
        return self._inner.get_lookup_and_lock_event_fd()

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a wrapped store (serialize → inner.store).

        All-or-nothing: if temp alloc fails for any key or serialize
        submission raises, the whole task is marked failed and the
        caller's next ``pop_completed_store_tasks`` call sees it.

        For :attr:`StoragePlacementMode.KV_SPLIT_TIER`, the path
        diverges: a K-only L1 child object is allocated and populated
        synchronously before the V serialize is submitted, the
        manifest enters STORE_IN_FLIGHT, and the inner L2 store is
        invoked with V child keys (not logical keys).  On completion,
        the manifest transitions to COMPLETE and the original logical
        L1 entry is deleted so the L1 footprint drops to K-only.
        """
        with self._lock:
            wrapped_id = self._next_task_id
            self._next_task_id += 1

        is_split_tier = (
            self._placement_mode == StoragePlacementMode.KV_SPLIT_TIER
        )
        k_child_keys: list[ObjectKey] = []
        v_child_keys: list[ObjectKey] = []
        if is_split_tier:
            k_child_keys, v_child_keys = self._alloc_split_tier_children(
                keys, objects
            )
            if not k_child_keys:
                # K-child alloc failed (out of L1 or non-grouped input).
                logger.warning(
                    "Serde wrapper: split-tier K-child alloc failed for "
                    "store task %d",
                    wrapped_id,
                )
                self._finalize_store(wrapped_id, success=False)
                return wrapped_id

        temp_keys, temp_objs = self._alloc_temp_buffers(keys, objects)
        if temp_objs is None:
            logger.warning(
                "Serde wrapper: temp alloc failed for store task %d",
                wrapped_id,
            )
            if k_child_keys:
                # Release the K-children we just allocated; they would
                # otherwise leak L1 with no V partner to back them.
                self._release_split_tier_k_children(k_child_keys)
            self._finalize_store(wrapped_id, success=False)
            return wrapped_id

        # NB: register_pending was already called inside
        # _alloc_split_tier_children (before K-child finish_write
        # fires the L1 listener), so we don't repeat it here.

        # Hold the wrapper lock across submit + reverse-map registration
        # so the internal drain thread cannot observe a half-state where
        # the serde already signaled completion but ``_serde_to_store``
        # has no entry — which would leave the wrapped task hanging.
        state = _StoreTaskState(
            wrapped_id=wrapped_id,
            keys=list(keys),
            temp_keys=temp_keys,
            temp_objs=temp_objs,
            phase=_StorePhase.SERIALIZE,
            is_split_tier=is_split_tier,
            k_child_keys=k_child_keys,
            v_child_keys=v_child_keys,
        )
        try:
            with self._lock:
                self._store_tasks[wrapped_id] = state
                serde_src = self._build_serde_src_inputs(objects)
                serde_task_id = self._serde.submit_serialize(
                    serde_src,  # type: ignore[arg-type]
                    temp_objs,
                )
                self._serde_to_store[serde_task_id] = wrapped_id
        except Exception:
            logger.exception(
                "Serde wrapper: submit_serialize raised for store task %d",
                wrapped_id,
            )
            with self._lock:
                self._store_tasks.pop(wrapped_id, None)
            self._release_write_temps(temp_keys)
            self._finalize_store(wrapped_id, success=False)
            return wrapped_id
        return wrapped_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            result = self._completed_store
            self._completed_store = {}
        return result

    # ------------------------------------------------------------------
    # Lookup / unlock
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        """Submit a lookup-and-lock for the given logical keys.

        For :attr:`StoragePlacementMode.KV_SPLIT_TIER`, translate
        logical keys → V child keys before delegating to the inner
        adapter (V lives on L2 under derived names).  The caller's
        view stays logical; the wrapper's
        :meth:`query_lookup_and_lock_result` combines inner's V-hit
        bitmap with the manifest's ``COMPLETE`` state so a logical
        key resolves as a composite hit only when both children are
        addressable.
        """
        if self._placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
            v_child_keys = [derive_component_key(k, "v") for k in keys]
            return self._inner.submit_lookup_and_lock_task(v_child_keys)
        return self._inner.submit_lookup_and_lock_task(keys)

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        # For KV_TOGETHER the inner result is the final answer.  For
        # KV_SPLIT_TIER the caller-visible bitmap is the intersection
        # of inner's V-on-L2 hit AND the manifest's COMPLETE state;
        # we cannot mask here without the original logical keys, so
        # the load path takes care of the manifest gate on read.  The
        # inner result is forwarded as-is (callers that need strict
        # manifest semantics should rely on submit_load_task's bitmap).
        return self._inner.query_lookup_and_lock_result(task_id)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        if self._placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
            self._inner.submit_unlock([derive_component_key(k, "v") for k in keys])
            return
        self._inner.submit_unlock(keys)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a wrapped load (inner.load → deserialize).

        All-or-nothing: if temp alloc or inner submission fails, the
        caller gets an all-zeros bitmap on next ``query_load_result``.

        For :attr:`StoragePlacementMode.KV_SPLIT_TIER`, the inner
        adapter is asked for V child blobs (derived V child keys),
        and ``_drain_inner_load`` composes the result by copying K
        bytes out of the L1 K children before submitting deserialize.
        Manifest entries that are not ``COMPLETE`` are masked off
        before the inner call so the load returns a miss for those
        keys.
        """
        with self._lock:
            wrapped_id = self._next_task_id
            self._next_task_id += 1

        is_split_tier = (
            self._placement_mode == StoragePlacementMode.KV_SPLIT_TIER
        )
        k_child_keys: list[ObjectKey] = []
        v_child_keys: list[ObjectKey] = []
        if is_split_tier:
            k_child_keys = [derive_component_key(k, "k") for k in keys]
            v_child_keys = [derive_component_key(k, "v") for k in keys]

        temp_keys, temp_objs = self._alloc_temp_buffers(keys, objects)
        if temp_objs is None:
            logger.warning(
                "Serde wrapper: temp alloc failed for load task %d",
                wrapped_id,
            )
            self._finalize_load(wrapped_id, Bitmap(len(keys)))
            return wrapped_id

        # Hold the wrapper lock across submit + reverse-map registration
        # so the internal drain thread cannot observe a half-state where
        # the inner already signaled completion but ``_inner_to_load``
        # has no entry.
        state = _LoadTaskState(
            wrapped_id=wrapped_id,
            keys=list(keys),
            dst_objs=list(objects),
            temp_keys=temp_keys,
            temp_objs=temp_objs,
            is_split_tier=is_split_tier,
            k_child_keys=k_child_keys,
            v_child_keys=v_child_keys,
        )
        # Inner sees V child keys for split-tier; logical keys otherwise.
        inner_load_keys = v_child_keys if is_split_tier else keys
        try:
            with self._lock:
                self._load_tasks[wrapped_id] = state
                inner_task_id = self._inner.submit_load_task(
                    inner_load_keys, temp_objs
                )
                self._inner_to_load[inner_task_id] = wrapped_id
        except Exception:
            logger.exception(
                "Serde wrapper: inner.submit_load_task raised for task %d",
                wrapped_id,
            )
            with self._lock:
                self._load_tasks.pop(wrapped_id, None)
            self._release_write_temps(temp_keys)
            self._finalize_load(wrapped_id, Bitmap(len(keys)))
            return wrapped_id
        return wrapped_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_load.pop(task_id, None)

    # ------------------------------------------------------------------
    # Eviction / metadata / listeners (delegate to inner)
    # ------------------------------------------------------------------

    @property
    def inner_adapter(self) -> L2AdapterInterface:
        """Return the wrapped L2 adapter."""
        return self._inner

    @property
    def supports_global_eviction(self) -> bool:
        return self._inner.supports_global_eviction

    def get_usage(self) -> AdapterUsage:
        return self._inner.get_usage()

    def delete(self, keys: list[ObjectKey]) -> None:
        self._inner.delete(keys)

    def register_listener(self, listener: L2AdapterListener) -> None:
        # Listeners track what's actually stored — which is inner's job.
        self._inner.register_listener(listener)

    def report_status(self) -> dict:
        inner_status = self._inner.report_status()
        return {**inner_status, "serde_wrapped": True}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._stop_flag.set()
        self._thread.join()
        if self._split_tier_copy_pool is not None:
            self._split_tier_copy_pool.shutdown(wait=True)
            self._split_tier_copy_pool = None

        # Shut down the inner adapter and serde processor BEFORE
        # releasing temp buffers. Both ``close()`` calls block until
        # their in-flight reads / writes against the temp MemoryObjs
        # finish; releasing the temps first would let L1 reclaim memory
        # that the inner adapter or serde thread pool is still touching
        # (use-after-free).
        self._inner.close()
        self._serde.close()

        # Now safe to release leftover temp buffers. Store tasks in
        # SERIALIZE phase hold write locks on their temps; tasks in
        # INNER_STORE phase hold read locks (transitioned after
        # serialize). Load tasks always hold write locks.
        with self._lock:
            write_locked: list[ObjectKey] = []
            read_locked: list[ObjectKey] = []
            for s in self._store_tasks.values():
                if s.phase is _StorePhase.SERIALIZE:
                    write_locked.extend(s.temp_keys)
                else:
                    read_locked.extend(s.temp_keys)
            for load in self._load_tasks.values():
                write_locked.extend(load.temp_keys)
            self._store_tasks.clear()
            self._load_tasks.clear()
            self._serde_to_store.clear()
            self._inner_to_store.clear()
            self._inner_to_load.clear()
            self._serde_to_load.clear()

        if write_locked:
            try:
                self._l1_manager.finish_write(write_locked)
                self._l1_manager.delete(write_locked)
            except Exception:
                logger.exception(
                    "Serde wrapper: error releasing write-locked leftover temps"
                )
        if read_locked:
            try:
                self._l1_manager.finish_read(read_locked)
            except Exception:
                logger.exception(
                    "Serde wrapper: error releasing read-locked leftover temps"
                )

        self._store_efd.close()
        self._load_efd.close()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        poller = select.poll()
        inner_store_efd = self._inner.get_store_event_fd()
        inner_load_efd = self._inner.get_load_event_fd()
        serialize_efd = self._serde.get_serialize_event_fd()
        deserialize_efd = self._serde.get_deserialize_event_fd()

        poller.register(inner_store_efd, select.POLLIN)
        poller.register(inner_load_efd, select.POLLIN)
        poller.register(serialize_efd, select.POLLIN)
        poller.register(deserialize_efd, select.POLLIN)

        while not self._stop_flag.is_set():
            ready = poller.poll(_POLL_TIMEOUT_MS)
            for fd, events in ready:
                if not (events & select.POLLIN):
                    continue
                try:
                    consume_fd(fd)
                except OSError:
                    pass
                try:
                    if fd == serialize_efd:
                        self._drain_serialize()
                    elif fd == inner_store_efd:
                        self._drain_inner_store()
                    elif fd == inner_load_efd:
                        self._drain_inner_load()
                    elif fd == deserialize_efd:
                        self._drain_deserialize()
                except Exception:
                    logger.exception("Serde wrapper: internal loop error on fd %d", fd)

    def _drain_serialize(self) -> None:
        """Poll pending serialize tasks; on success submit inner store."""
        with self._lock:
            pending = list(self._serde_to_store.keys())
        for serde_id in pending:
            result = self._serde.query_serialize_result(serde_id)
            if result is None:
                continue
            with self._lock:
                wrapped_id = self._serde_to_store.pop(serde_id, None)
                state = (
                    self._store_tasks.get(wrapped_id)
                    if wrapped_id is not None
                    else None
                )
            if wrapped_id is None or state is None:
                continue

            if not result:
                self._release_write_temps(state.temp_keys)
                self._finalize_store(wrapped_id, success=False)
                continue

            # Serialize succeeded — transition temps write → read so inner
            # can safely read them during the store.
            self._l1_manager.finish_write_and_reserve_read(state.temp_keys)
            # For KV_SPLIT_TIER the inner adapter sees V child keys, not
            # logical keys, so V lives under a deterministic derived
            # name on L2 that the load path will re-derive.  For
            # KV_TOGETHER it's the logical keys as today.
            inner_store_keys = (
                state.v_child_keys if state.is_split_tier else state.keys
            )
            try:
                inner_id = self._inner.submit_store_task(
                    inner_store_keys, state.temp_objs
                )
            except Exception:
                logger.exception(
                    "Serde wrapper: inner.submit_store_task raised for task %d",
                    wrapped_id,
                )
                # Temps are now read-locked and temporary — finish_read
                # is enough; the entries auto-delete.
                self._l1_manager.finish_read(state.temp_keys)
                if state.is_split_tier:
                    self._invalidate_split_tier_pending(state)
                self._finalize_store(wrapped_id, success=False)
                continue

            # Phase flip and reverse-map insert happen under the same
            # lock so ``close()``'s cleanup can't observe a half-way
            # transition.
            with self._lock:
                state.phase = _StorePhase.INNER_STORE
                self._inner_to_store[inner_id] = wrapped_id

    def _drain_inner_store(self) -> None:
        """Drain inner store completions; release temp read locks (auto-
        delete) and finalize the wrapped tasks.

        For :attr:`StoragePlacementMode.KV_SPLIT_TIER`, additionally
        mark the manifest entries COMPLETE on success (which makes
        future lookup return composite hits) and delete the original
        logical L1 entries (K is preserved in the K child; V is on
        L2; the full K+V staging is now redundant -- this is what
        delivers the L1 footprint win for V-only).  On failure,
        invalidate the manifest entries and release the K children
        we reserved up-front so they don't orphan in L1.
        """
        completed = self._inner.pop_completed_store_tasks()
        for inner_id, result in completed.items():
            with self._lock:
                wrapped_id = self._inner_to_store.pop(inner_id, None)
                state = (
                    self._store_tasks.get(wrapped_id)
                    if wrapped_id is not None
                    else None
                )
            if wrapped_id is None:
                logger.warning(
                    "Serde wrapper: inner store task %d has no wrapped id",
                    inner_id,
                )
                continue
            if state is not None:
                self._l1_manager.finish_read(state.temp_keys)
                if state.is_split_tier:
                    if result.is_successful():
                        self._finalize_split_tier_success(state)
                    else:
                        self._invalidate_split_tier_pending(state)
            self._finalize_store(
                wrapped_id, result.is_successful(), result.bytes_transferred()
            )

    def _drain_inner_load(self) -> None:
        """Drain inner load completions; on per-key success submit
        deserialize, otherwise fail the keys immediately.

        For :attr:`StoragePlacementMode.KV_SPLIT_TIER`, gate each
        per-key success on the manifest being ``COMPLETE`` AND the
        K child being readable in L1, then synchronously copy K
        bytes from the K child into the dst's group-0 view before
        submitting deserialize for V.  Keys whose K child has been
        evicted in the meantime get masked off in the bitmap so the
        caller sees them as miss (the inner-loaded V blob is
        discarded with the rest of the L1 temp on finalize).
        """
        with self._lock:
            pending = list(self._inner_to_load.keys())
        for inner_id in pending:
            bitmap = self._inner.query_load_result(inner_id)
            if bitmap is None:
                continue
            with self._lock:
                wrapped_id = self._inner_to_load.pop(inner_id, None)
                state = (
                    self._load_tasks.get(wrapped_id) if wrapped_id is not None else None
                )
            if wrapped_id is None or state is None:
                continue

            if state.is_split_tier:
                # Mask non-COMPLETE manifest entries to miss before
                # bothering with K-from-L1 lookups.
                for i, logical_key in enumerate(state.keys):
                    if bitmap.test(i) and not self._split_tier_manifest.is_complete(
                        logical_key
                    ):
                        bitmap.clear(i)
                # Compose K from L1: for each surviving idx, read the
                # K child + copy K bytes into the dst's group-0 view.
                self._compose_split_tier_k_from_l1(state, bitmap)

            src_objs: list[MemoryObj] = []
            dst_objs: list[MemoryObj] = []
            for i in range(len(state.keys)):
                if bitmap.test(i):
                    src_objs.append(state.temp_objs[i])
                    dst_objs.append(state.dst_objs[i])

            if not src_objs:
                # Inner loaded nothing — skip deserialize, finalize.
                self._release_write_temps(state.temp_keys)
                self._finalize_load(wrapped_id, bitmap)
                continue

            state.load_bitmap = bitmap
            try:
                serde_dst = self._build_serde_dst_outputs(dst_objs)
                serde_id = self._serde.submit_deserialize(
                    src_objs,
                    serde_dst,  # type: ignore[arg-type]
                )
            except Exception:
                logger.exception(
                    "Serde wrapper: submit_deserialize raised for task %d",
                    wrapped_id,
                )
                self._release_write_temps(state.temp_keys)
                self._finalize_load(wrapped_id, Bitmap(len(state.keys)))
                continue
            with self._lock:
                self._serde_to_load[serde_id] = wrapped_id

    def _drain_deserialize(self) -> None:
        """Drain deserialize completions; report inner's load bitmap on
        success, all-zeros on deserialize failure."""
        with self._lock:
            pending = list(self._serde_to_load.keys())
        for serde_id in pending:
            result = self._serde.query_deserialize_result(serde_id)
            if result is None:
                continue
            with self._lock:
                wrapped_id = self._serde_to_load.pop(serde_id, None)
                state = (
                    self._load_tasks.get(wrapped_id) if wrapped_id is not None else None
                )
            if wrapped_id is None or state is None:
                continue

            if result:
                # ``load_bitmap`` was populated by _drain_inner_load before
                # the task was registered in ``_serde_to_load``.
                final_bitmap = state.load_bitmap
            else:
                logger.warning(
                    "Serde wrapper: deserialize failed for task %d; "
                    "reporting all keys as failed",
                    wrapped_id,
                )
                final_bitmap = Bitmap(len(state.keys))

            self._release_write_temps(state.temp_keys)
            self._finalize_load(wrapped_id, final_bitmap)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _alloc_temp_buffers(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> tuple[list[ObjectKey], list[MemoryObj] | None]:
        """Reserve one temp byte buffer per input key. All-or-nothing:
        any single failure releases the partial successes and returns
        ``(temp_keys, None)``.

        Args:
            keys: Original logical keys; used only to derive temp keys.
            objects: Source (store) or destination (load) MemoryObjs.
                All entries must share a single ``(shape, dtype)`` — the
                caller (store/prefetch controller) is responsible for
                shape-grouping before submission.
        """
        shape_0 = objects[0].get_shapes()
        dtype_0 = objects[0].get_dtypes()
        temp_keys = [make_temp_key(k) for k in keys]
        layout = serialized_layout_desc(
            MemoryLayoutDesc(shapes=shape_0, dtypes=dtype_0), self._serde
        )
        results = self._l1_manager.reserve_write(
            keys=temp_keys,
            is_temporary=[True] * len(temp_keys),
            layout_desc=layout,
            mode="new",
        )
        # First pass: collect every key whose reserve_write succeeded.
        # We must scan the full list (not bail on the first failure)
        # so a mixed-success result still releases all reserved keys.
        successful_temp_keys: list[ObjectKey] = []
        for temp_key in temp_keys:
            r = results.get(temp_key)
            if r is not None and r[0] == L1Error.SUCCESS:
                successful_temp_keys.append(temp_key)
        if len(successful_temp_keys) != len(temp_keys):
            self._release_write_temps(successful_temp_keys)
            return temp_keys, None
        temp_objs = [results[tk][1] for tk in temp_keys]
        return temp_keys, temp_objs

    def _release_write_temps(self, temp_keys: list[ObjectKey]) -> None:
        """Release write-locked temps and delete them. No-op on empty."""
        if not temp_keys:
            return
        try:
            self._l1_manager.finish_write(temp_keys)
            self._l1_manager.delete(temp_keys)
        except Exception:
            logger.exception("Serde wrapper: failed releasing write-locked temps")

    # ------------------------------------------------------------------
    # Multi-output dispatch: build per-slot GroupSlotView tuples when
    # the underlying serde is multi-output (e.g. AsymK16V8Multi*), or
    # pass MemoryObjs through unchanged for single-tensor serdes.  The
    # mapping is queried from the SerdeProcessor (default ``None`` means
    # single-tensor; a non-None tuple defines the per-slot ↔ parent-group
    # routing -- e.g. identity ``(0, 1)`` for storage-only Mode 1,
    # ``(None, 1)`` for V-only Mode 2).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Split-tier (KV_SPLIT_TIER placement) helpers.  Only invoked when
    # ``self._placement_mode == KV_SPLIT_TIER``; the KV_TOGETHER path
    # never touches these.
    # ------------------------------------------------------------------

    def _alloc_split_tier_children(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> tuple[list[ObjectKey], list[ObjectKey]]:
        """Allocate K-only L1 child entries + derive V child keys.

        For each logical key:

        * Derive ``K_child_key = derive_component_key(logical, "k")``
          and ``V_child_key = derive_component_key(logical, "v")``.
        * Allocate a K-only L1 ``MemoryObj`` under the K child key
          (single-group: ``meta.shapes[0]`` / ``meta.dtypes[0]`` from
          the producer's grouped staging object).
        * Synchronously copy K bytes from the staging object's
          group-0 tensor into the K child's tensor.
        * Commit the K child write so it persists as the canonical
          L1 resident.

        Returns ``(k_child_keys, v_child_keys)`` on success.  Returns
        ``([], [])`` if any L1 allocation fails or the input
        ``objects`` aren't grouped (sanity check -- shouldn't happen
        if placement is correctly KV_SPLIT_TIER, but raises a clear
        error instead of corrupting L1).
        """
        if not objects:
            return [], []
        # Sanity: V-only placement requires grouped K/V layout in L1.
        # Use the public get_shapes/get_dtypes API (not meta.shapes
        # directly) so this works across all MemoryObj subclasses /
        # allocators.
        first = objects[0]
        try:
            shapes = first.get_shapes()
            dtypes = first.get_dtypes()
        except Exception:
            shapes = None
            dtypes = None
        if shapes is None or dtypes is None or len(shapes) < 2:
            logger.error(
                "Serde wrapper: split-tier requires grouped (K, V) "
                "input; got non-grouped MemoryObj (shapes=%r, "
                "dtypes=%r).  This is a placement / layout config "
                "mismatch.",
                shapes,
                dtypes,
            )
            return [], []

        k_child_keys = [derive_component_key(k, "k") for k in keys]
        v_child_keys = [derive_component_key(k, "v") for k in keys]

        # K-only layout descriptor (group 0 of the staging object).
        k_layout = MemoryLayoutDesc(
            shapes=[shapes[0]],
            dtypes=[dtypes[0]],
        )

        results = self._l1_manager.reserve_write(
            keys=k_child_keys,
            is_temporary=[False] * len(k_child_keys),
            layout_desc=k_layout,
            mode="new",
        )
        # All-or-nothing on K-child allocations.
        successful: list[ObjectKey] = []
        for k_child_key in k_child_keys:
            r = results.get(k_child_key)
            if r is not None and r[0] == L1Error.SUCCESS:
                successful.append(k_child_key)
        if len(successful) != len(k_child_keys):
            # Partial success -- release the partials and fail the
            # whole task (the caller has not yet registered the
            # manifest entries).
            self._release_split_tier_k_children(successful)
            return [], []

        # Copy K bytes from each staging object's group-0 tensor into
        # the K child.  torch's .copy_() between CPU tensors releases
        # the GIL, so we fan the per-key copies across a small thread
        # pool to overlap them on multi-CPU hosts (the StoreController
        # submits batched calls so N>1 is common).
        def _copy_one(idx: int) -> bool:
            k_child_obj = results[k_child_keys[idx]][1]
            if k_child_obj is None:
                return False
            src_k = objects[idx].get_tensor(0)
            dst_k = k_child_obj.get_tensor(0)
            if src_k is None or dst_k is None:
                return False
            dst_k.copy_(src_k)
            return True

        if self._split_tier_copy_pool is not None and len(keys) > 1:
            ok_list = list(
                self._split_tier_copy_pool.map(_copy_one, range(len(keys)))
            )
        else:
            # Single-key batch or no pool: avoid the dispatch overhead.
            ok_list = [_copy_one(i) for i in range(len(keys))]
        if not all(ok_list):
            self._release_split_tier_k_children(successful)
            return [], []

        # Register the manifest BEFORE finish_write so the
        # StoreController's listener filter (is_k_child_key) sees
        # the K-child key when the L1 manager's on_l1_keys_write_finished
        # fires.  Otherwise the K-child key races into the pending
        # queue and gets routed back into submit_store_task as a
        # bare 1-shape MemoryObj, failing the multi-group check
        # (Phase 2 pod regression discovered 2026-05-31).
        for logical_key in keys:
            self._split_tier_manifest.register_pending(logical_key)

        # Commit the K-child writes.  After this they are read-only
        # cache residents under their child keys.
        self._l1_manager.finish_write(k_child_keys)
        return k_child_keys, v_child_keys

    def _release_split_tier_k_children(self, k_child_keys: list[ObjectKey]) -> None:
        """Release K-child L1 entries.  Used on a partial / failed
        store path so K children don't orphan in L1 with no V partner."""
        if not k_child_keys:
            return
        try:
            self._l1_manager.finish_write(k_child_keys)
        except Exception:
            # If finish_write fails some children may not have been in
            # write phase -- the delete below is still authoritative.
            logger.debug(
                "Serde wrapper split-tier release: finish_write raised; "
                "continuing to delete %d K children",
                len(k_child_keys),
            )
        try:
            self._l1_manager.delete(k_child_keys)
        except Exception:
            logger.exception(
                "Serde wrapper split-tier release: delete raised for "
                "%d K children",
                len(k_child_keys),
            )

    def _finalize_split_tier_success(self, state: _StoreTaskState) -> None:
        """Inner L2 V-store succeeded for a split-tier task.

        Mark all logical keys COMPLETE in the manifest.  The
        physical deletion of the original logical-key L1 entries
        (full K+V staging) is driven by the StoreController in
        ``_finalize_store`` after it releases the read lock the
        controller acquired on those keys -- attempting to delete
        them here would race the lock and fail with KEY_IS_LOCKED.
        """
        for logical_key in state.keys:
            try:
                self._split_tier_manifest.mark_complete(logical_key)
            except ValueError:
                # Pre-empted by an invalidate from the eviction
                # controller (PR-3').  The K child + manifest entry
                # cleanup happens through the invalidation path; here
                # we just refuse to mark COMPLETE on an INVALIDATED key.
                logger.debug(
                    "Serde wrapper split-tier: logical key already "
                    "invalidated before COMPLETE; skipping"
                )

    def _invalidate_split_tier_pending(self, state: _StoreTaskState) -> None:
        """Inner L2 V-store failed (or serialize failed) for a
        split-tier task: invalidate the manifest and release the K
        children we reserved.

        Idempotent on the manifest side via :meth:`SplitTierManifest.mark_invalidated`.
        """
        for logical_key in state.keys:
            self._split_tier_manifest.mark_invalidated(logical_key)
        self._release_split_tier_k_children(state.k_child_keys)

    def _compose_split_tier_k_from_l1(
        self,
        state: _LoadTaskState,
        bitmap: "Bitmap",
    ) -> None:
        """Copy K bytes from the L1 K children into each surviving
        dst's group-0 view.

        Reads are reserved per surviving index, the copy happens
        synchronously (cheap CPU memcpy; runs in the wrapper's poll
        loop), then the read locks are released.  Any K child whose
        ``reserve_read`` fails (KEY_NOT_EXIST -- evicted between
        manifest check and load) is masked off in the bitmap so the
        caller sees that key as miss.  The wrapper does not eagerly
        invalidate the manifest entry here -- PR-3' (paired
        eviction) is responsible for that lifecycle.
        """
        # Surviving indices = those still set in the bitmap after the
        # manifest mask.
        surviving = [
            i for i in range(len(state.keys)) if bitmap.test(i)
        ]
        if not surviving:
            return
        k_keys_to_read = [state.k_child_keys[i] for i in surviving]
        results = self._l1_manager.reserve_read(k_keys_to_read)
        successful_k_keys: list[ObjectKey] = []
        for idx, k_key in zip(surviving, k_keys_to_read, strict=True):
            r = results.get(k_key)
            if r is None or r[0] != L1Error.SUCCESS or r[1] is None:
                # K child missing or unreadable -- mask the key off
                # so the deserialize path skips it and the caller
                # sees a miss.
                bitmap.clear(idx)
                continue
            k_child_obj = r[1]
            successful_k_keys.append(k_key)
            try:
                src_k = k_child_obj.get_tensor(0)
                dst_k = state.dst_objs[idx].get_tensor(0)
                if src_k is None or dst_k is None:
                    bitmap.clear(idx)
                    continue
                dst_k.copy_(src_k)
            except Exception:
                logger.exception(
                    "Serde wrapper split-tier load: K copy failed for "
                    "logical key %s",
                    state.keys[idx],
                )
                bitmap.clear(idx)
        # Release K read locks (idempotent on already-released keys).
        if successful_k_keys:
            try:
                self._l1_manager.finish_read(successful_k_keys)
            except Exception:
                logger.exception(
                    "Serde wrapper split-tier load: finish_read raised "
                    "for %d K children",
                    len(successful_k_keys),
                )

    def _build_serde_src_inputs(
        self, objects: list[MemoryObj]
    ) -> "list[MemoryObj] | list[MemoryObjGroup]":
        """Adapt source ``objects`` to the shape the serde expects.

        Returns the input list unchanged for single-tensor serdes.  For
        multi-output serdes, returns a parallel list of
        :data:`MemoryObjGroup` tuples whose slots are
        :class:`GroupSlotView` instances over the parent's groups (or
        ``None`` for slots the mapping marks absent).
        """
        mapping = self._serde.input_slot_mapping()
        if mapping is None:
            return objects
        return [
            tuple(
                GroupSlotView(obj, idx) if idx is not None else None
                for idx in mapping
            )
            for obj in objects
        ]

    def _build_serde_dst_outputs(
        self, dst_objs: list[MemoryObj]
    ) -> "list[MemoryObj] | list[MemoryObjGroup]":
        """Adapt destination ``dst_objs`` to the shape the deserializer
        expects.  Symmetric to :meth:`_build_serde_src_inputs` for the
        load path.
        """
        mapping = self._serde.output_slot_mapping()
        if mapping is None:
            return dst_objs
        return [
            tuple(
                GroupSlotView(obj, idx) if idx is not None else None
                for idx in mapping
            )
            for obj in dst_objs
        ]

    def _finalize_store(
        self,
        wrapped_id: L2TaskId,
        success: bool,
        bytes_transferred: int = 0,
    ) -> None:
        with self._lock:
            self._store_tasks.pop(wrapped_id, None)
            self._completed_store[wrapped_id] = L2StoreResult(
                success, bytes_transferred
            )
        try:
            self._store_efd.notify()
        except OSError:
            logger.exception("Serde wrapper: failed to signal store notifier")

    def _finalize_load(self, wrapped_id: L2TaskId, bitmap: Bitmap) -> None:
        with self._lock:
            self._load_tasks.pop(wrapped_id, None)
            self._completed_load[wrapped_id] = bitmap
        try:
            self._load_efd.notify()
        except OSError:
            logger.exception("Serde wrapper: failed to signal load notifier")
