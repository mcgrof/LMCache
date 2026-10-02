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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import enum
import select
import threading

# Third Party
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    KeyEntry,
    KeyListPage,
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L2AdapterListener, L2StoreResult
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import (
    AdapterUsage,
    EarlyReleaseStoreAdapter,
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.split_tier_memory import (
    _KChildSlab,
    _VScratchSlab,
    _VScratchSlot,
)
from lmcache.v1.distributed.memory_manager.l1_memory_manager import (
    L1MemoryUsageProvider,
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
    SplitTierState,
    StoragePlacementMode,
    derive_component_key,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.platform import consume_fd, create_event_notifier

logger = init_logger(__name__)

_POLL_TIMEOUT_MS = 500

# L1 write tag for the wrapper's temp buffers. Temp keys are unique per task,
# so no two reservations ever share a key; the tag documents the owner.
_L1_WRITE_TAG = "serde_wrapper"


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

    split_tier_generations: dict[ObjectKey, int] = field(default_factory=dict)
    """Manifest generation id per logical key, returned by
    ``register_pending`` in ``_alloc_split_tier_children``.  Threaded
    into every cleanup call (``mark_complete`` / ``mark_invalidated`` /
    ``drop``) so this task only ever transitions the generation it
    registered -- a later re-store of the same key gets a fresh
    generation and this task's cleanup no longer touches it."""

    early_release_keys: list[ObjectKey] = field(default_factory=list)
    """Logical keys whose StoreController read locks can be released
    immediately after ``submit_store_task`` returns.  Set only in
    split-tier placement, where the wrapper extracts both K and V
    into private buffers before the codec runs and so no longer
    needs the caller-side L1 entry to stay alive across the L2
    write.  Drained by ``claim_early_release_keys`` on the
    ``EarlyReleaseStoreAdapter`` protocol."""

    early_release_claimed: bool = False
    """Latched true the first time ``claim_early_release_keys``
    consumes ``early_release_keys``.  Single-claim semantics keep
    the StoreController from accidentally releasing the same locks
    twice on a retry / poll loop."""

    v_scratch_tensors: list[torch.Tensor] = field(default_factory=list)
    """Slab-borrowed V tensors holding a private copy of V bytes
    extracted from the producer-side logical entries.  Held alive
    here so the V-only codec can read from them after the logical
    L1 entries are released; returned to the slab in
    ``_drain_inner_store`` once the inner L2 store acks."""


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


class SerdeL2AdapterWrapper(L2AdapterInterface, EarlyReleaseStoreAdapter):
    """L2 adapter that adds transparent serde on top of an inner adapter.

    Implements :class:`EarlyReleaseStoreAdapter` so the StoreController
    can release the caller's read locks on split-tier logical keys as
    soon as the wrapper has extracted K and V into private buffers --
    well before the inner L2 store completes.  See
    :meth:`claim_early_release_keys`.

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
        v_scratch_max_slots: int = 64,
        k_child_max_slots: int = 256,
    ) -> None:
        super().__init__()
        self._inner = inner
        self._serde = serde
        self._l1_manager = l1_manager
        self._placement_mode = placement_mode
        # Always constructed; empty + unused when placement is
        # KV_TOGETHER.  When KV_SPLIT_TIER, the wrapper drives the
        # state-machine transitions on this manifest during store
        # and load.
        # NB: use `is None` not `or` -- SplitTierManifest implements
        # __len__, so an empty manifest is falsy and `or` would
        # silently mint a fresh local manifest instead of using the
        # caller-provided one.  The bug bit pod testing.
        self._split_tier_manifest = (
            SplitTierManifest() if split_tier_manifest is None else split_tier_manifest
        )

        # Bounded thread pool for per-key K-byte copies in
        # _alloc_split_tier_children.  torch copy_ releases the GIL, so a
        # small pool can overlap copies in a batched submission without
        # creating one worker per key.  Only split-tier placement pays for
        # the pool.
        self._split_tier_copy_pool: ThreadPoolExecutor | None = None
        if placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
            self._split_tier_copy_pool = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="serde-l2-st-kcopy",
            )

        # V-scratch pools are created lazily per exact (shape, dtype), since
        # one wrapper may serve multiple model/layout buckets.  The configured
        # bound applies independently to every materialized layout pool.
        self._v_scratch_slabs: dict[
            tuple[tuple[int, ...], torch.dtype], _VScratchSlab
        ] = {}
        self._v_scratch_max_slots = max(1, v_scratch_max_slots)
        self._v_scratch_init_lock = threading.Lock()

        # K-child slab (lazy-init and pinned to the first K layout; later
        # layouts safely fall back to the ordinary allocator).  Pre-allocated
        # K-shape ``TensorMemoryObj``s
        # registered into L1Manager via ``reserve_external_writes``;
        # skips ``L1MemoryManager.allocate``'s address-manager scan on
        # every K-child store.  ``None`` outside split-tier.
        self._k_child_slab: _KChildSlab | None = None
        self._k_child_max_slots = max(1, k_child_max_slots)
        self._k_child_init_lock = threading.Lock()

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
        # Split-tier lookup tasks: inner task id -> the caller's
        # LOGICAL keys, retained so query_lookup_and_lock_result can
        # apply the manifest COMPLETE gate (and unlock masked-off V
        # children on the inner adapter).  Entries are popped when the
        # (once-only) inner result is surfaced.
        self._lookup_logical_keys: dict[L2TaskId, list[ObjectKey]] = {}

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

        is_split_tier = self._placement_mode == StoragePlacementMode.KV_SPLIT_TIER
        if not is_split_tier and not self._serde_mapping_covers_objects(
            objects, self._serde.input_slot_mapping(), role="store input"
        ):
            self._finalize_store(wrapped_id, success=False)
            return wrapped_id
        k_child_keys: list[ObjectKey] = []
        v_child_keys: list[ObjectKey] = []
        split_tier_generations: dict[ObjectKey, int] = {}
        v_scratch_tensors: list[torch.Tensor] = []
        early_release_keys: list[ObjectKey] = []
        if is_split_tier:
            (
                k_child_keys,
                v_child_keys,
                split_tier_generations,
            ) = self._alloc_split_tier_children(keys, objects)
            if not k_child_keys:
                # K-child alloc failed (out of L1 or non-grouped input).
                logger.warning(
                    "Serde wrapper: split-tier K-child alloc failed for store task %d",
                    wrapped_id,
                )
                self._publish_split_tier_store_invalidated(keys, "pre_submit")
                self._finalize_store(wrapped_id, success=False)
                return wrapped_id

            # Extract V into a private slab-borrowed scratch buffer so
            # the logical L1 entries can be released the moment this
            # method returns (instead of staying read-locked across the
            # V codec + L2 write).  Once K + V both live in wrapper-
            # private buffers, the producer-side L1 entry is redundant;
            # releasing it immediately drops the steady-state L1
            # footprint per chunk from ~1.5× (K+V logical + K-child) to
            # ~0.5× (K-child only), which is the predicted V-only L1
            # capacity win that the L2-completion-time delete alone
            # could not realize while the disk write remained pending.
            v_scratch_tensors = self._alloc_v_scratch_and_copy(objects)
            if not v_scratch_tensors:
                # V-extract failed (no V tensor on input objects).
                logger.warning(
                    "Serde wrapper: split-tier V-scratch extract failed "
                    "for store task %d",
                    wrapped_id,
                )
                self._release_split_tier_k_children(k_child_keys)
                self._drop_split_tier_pending(keys, split_tier_generations)
                self._publish_split_tier_store_invalidated(keys, "pre_submit")
                self._finalize_store(wrapped_id, success=False)
                return wrapped_id
            # Logical keys can be released immediately by the
            # StoreController via claim_early_release_keys().
            early_release_keys = list(keys)

        temp_keys, temp_objs = self._alloc_temp_buffers(keys, objects)
        if temp_objs is None:
            logger.warning(
                "Serde wrapper: temp alloc failed for store task %d",
                wrapped_id,
            )
            if k_child_keys:
                # Release the K-children we just allocated; they would
                # otherwise leak L1 with no V partner to back them --
                # and drop the pending manifest entries so the keys
                # remain storable (this exit fires precisely under L1
                # memory pressure, when retries are most likely).
                self._release_split_tier_k_children(k_child_keys)
                self._drop_split_tier_pending(keys, split_tier_generations)
            if v_scratch_tensors:
                self._return_v_scratch(v_scratch_tensors)
            if is_split_tier:
                self._publish_split_tier_store_invalidated(keys, "pre_submit")
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
            split_tier_generations=split_tier_generations,
            early_release_keys=early_release_keys,
            v_scratch_tensors=v_scratch_tensors,
        )
        try:
            with self._lock:
                self._store_tasks[wrapped_id] = state
                # The split-tier branch builds a V-only group over
                # scratch tensors, the default branch builds the packed
                # serde src inputs.
                serde_src: (
                    list[MemoryObj]
                    | list[MemoryObjGroup]
                    | list[tuple[None, _VScratchSlot]]
                )
                if is_split_tier and v_scratch_tensors:
                    # V-only codec reads V from src[i][1].tensor; route it
                    # to the slab-borrowed scratch tensor instead of the
                    # logical L1 entry (which the StoreController is about
                    # to release).
                    serde_src = [(None, _VScratchSlot(t)) for t in v_scratch_tensors]
                else:
                    serde_src = self._build_serde_src_inputs(objects)
                serde_task_id = self._serde.submit_serialize(
                    serde_src,  # type: ignore[arg-type]
                    temp_objs,
                    state.keys,
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
            if v_scratch_tensors:
                self._return_v_scratch(v_scratch_tensors)
            if k_child_keys:
                self._release_split_tier_k_children(k_child_keys)
                self._drop_split_tier_pending(keys, split_tier_generations)
            if is_split_tier:
                self._publish_split_tier_store_invalidated(keys, "pre_submit")
            self._finalize_store(wrapped_id, success=False)
            return wrapped_id
        return wrapped_id

    def claim_early_release_keys(self, task_id: L2TaskId) -> list[ObjectKey]:
        """Hand the StoreController the set of logical keys whose read
        locks can be released right now (before the inner L2 store
        completes).

        In :attr:`StoragePlacementMode.KV_SPLIT_TIER`, the wrapper has
        already copied K into the K-child L1 slot and V into a slab-
        borrowed scratch buffer before this method is called, so the
        caller-side logical L1 entry is redundant for the rest of the
        task's lifetime.

        Single-claim: subsequent calls for the same ``task_id`` return
        ``[]`` so the StoreController does not double-release on retry
        / re-poll.

        Args:
            task_id: The task id returned by ``submit_store_task``.

        Returns:
            Logical keys to early-release.  Empty for
            ``KV_TOGETHER`` placement (the inner store reads K+V from
            the caller's objects, so the caller must keep them
            read-locked across the L2 write).
        """
        with self._lock:
            state = self._store_tasks.get(task_id)
            if state is None or state.early_release_claimed:
                return []
            state.early_release_claimed = True
            return list(state.early_release_keys)

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            result = self._completed_store
            self._completed_store = {}
        return result

    # ------------------------------------------------------------------
    # Lookup / unlock
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        """Submit a lookup-and-lock for the given logical keys.

        For :attr:`StoragePlacementMode.KV_SPLIT_TIER`, translate
        logical keys → V child keys before delegating to the inner
        adapter (V lives on L2 under derived names).  The caller's
        view stays logical; the wrapper retains the logical keys per
        task so :meth:`query_lookup_and_lock_result` can combine
        inner's V-hit bitmap with the manifest's ``COMPLETE`` state --
        a logical key resolves as a composite hit only when both
        children are addressable.
        """
        if self._placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
            v_child_keys = [derive_component_key(k, "v") for k in keys]
            task_id = self._inner.submit_lookup_and_lock_task(
                v_child_keys, group_layout_descs
            )
            with self._lock:
                self._lookup_logical_keys[task_id] = list(keys)
            return task_id
        return self._inner.submit_lookup_and_lock_task(keys, group_layout_descs)

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Query a lookup-and-lock result; applies the manifest gate.

        For ``KV_TOGETHER`` the inner result is the final answer.  For
        :attr:`StoragePlacementMode.KV_SPLIT_TIER` the caller-visible
        bitmap is the intersection of inner's V-on-L2 hit AND the
        manifest's ``COMPLETE`` state.  V hits masked off by the gate
        are immediately unlocked on the inner adapter: the caller
        derives its unlocks from the bitmap it receives, so a hidden
        hit would otherwise hold its L2 lock forever.  The load path
        re-applies the gate on read as defense in depth (the manifest
        can transition between this query and the load).
        """
        result = self._inner.query_lookup_and_lock_result(task_id)
        if result is None:
            return None
        with self._lock:
            logical_keys = self._lookup_logical_keys.pop(task_id, None)
        if logical_keys is None:
            # KV_TOGETHER (nothing recorded) -- inner result is final.
            return result
        masked_v_children: list[ObjectKey] = []
        for i, logical_key in enumerate(logical_keys):
            if not result.test(i):
                continue
            if self._split_tier_manifest.is_complete(logical_key):
                continue
            result.clear(i)
            masked_v_children.append(derive_component_key(logical_key, "v"))
        if masked_v_children:
            logger.info(
                "Serde wrapper split-tier lookup: masked %d V hit(s) "
                "whose manifest state is not COMPLETE; unlocking them "
                "on the inner adapter",
                len(masked_v_children),
            )
            # Guard the unlock: the inner lookup result is once-only and
            # already consumed above, so a raising submit_unlock would
            # lose the gated bitmap entirely and hang the caller's
            # prefetch.  Return the bitmap regardless; the masked V
            # children keep their inner lock until TTL / close (a bounded
            # leak, and a no-op on adapters whose submit_unlock is a
            # no-op, e.g. the FS adapter).
            try:
                self._inner.submit_unlock(masked_v_children)
            except Exception:
                logger.exception(
                    "Serde wrapper split-tier lookup: inner submit_unlock "
                    "raised for %d masked V child(ren); returning the "
                    "gated bitmap anyway",
                    len(masked_v_children),
                )
        return result

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

        is_split_tier = self._placement_mode == StoragePlacementMode.KV_SPLIT_TIER
        if not is_split_tier and not self._serde_mapping_covers_objects(
            objects, self._serde.output_slot_mapping(), role="load output"
        ):
            self._finalize_load(wrapped_id, Bitmap(len(keys)))
            return wrapped_id
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
                inner_task_id = self._inner.submit_load_task(inner_load_keys, temp_objs)
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

    def list_l2_keys(
        self,
        model_name: str | None = None,
        page_size: int = 500,
        cursor: str | None = None,
    ) -> KeyListPage:
        """List L2 keys, re-presenting split-tier V children under their
        logical key.

        For ``KV_TOGETHER`` the inner page is returned verbatim.  For
        :attr:`StoragePlacementMode.KV_SPLIT_TIER` the inner adapter
        stores V children under derived keys carrying a 1-byte role
        marker; surfacing those raw would leak internal child keys onto
        operator listing surfaces (GET /cache/objects), so each ``v``
        child is mapped back to its logical key.  Any ``k``-role entry
        (K children are L1-only; one on L2 is anomalous) is dropped.
        Pagination and ``model_name`` filtering are unaffected -- the
        cursor is the inner adapter's, and ``derive_component_key``
        preserves ``model_name`` so the inner filter still matches.
        """
        page = self._inner.list_l2_keys(
            model_name=model_name,
            page_size=page_size,
            cursor=cursor,
        )
        if self._placement_mode != StoragePlacementMode.KV_SPLIT_TIER:
            return page
        mapped: list[KeyEntry] = []
        for entry in page.entries:
            physical_key = entry.key.to_object_key()
            logical_key = self._split_tier_manifest.logical_for_v_child(physical_key)
            if logical_key is not None:
                mapped.append(replace(entry, key=logical_key.to_encoded_object_key()))
                continue
            if self._split_tier_manifest.logical_for_k_child(physical_key) is not None:
                # K children never live on L2; drop any that surface.
                continue
            # Not a tracked component key; present as-is (defensive).
            mapped.append(entry)
        return replace(page, entries=tuple(mapped))

    def register_listener(self, listener: L2AdapterListener) -> None:
        # Listeners track what's actually stored — which is inner's job.
        self._inner.register_listener(listener)

    def set_backend_identity(self, name: str, shared: bool = False) -> None:
        """Forward the event-tagging identity to the inner adapter (which
        owns the listener-notify funnel that tags cache events)."""
        self._inner.set_backend_identity(name, shared)

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

        # Unregister both slabs from L1's usage aggregation so the
        # L1Manager doesn't keep stale references after the wrapper
        # goes away.  Idempotent if the slab was never registered.
        slabs: list[L1MemoryUsageProvider | None] = [self._k_child_slab]
        slabs.extend(self._v_scratch_slabs.values())
        for slab in slabs:
            if slab is None:
                continue
            try:
                self._l1_manager.unregister_external_memory_provider(slab)
            except AttributeError:
                pass

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
            self._lookup_logical_keys.clear()

        if write_locked:
            try:
                self._l1_manager.finish_write_and_delete(
                    write_locked, tag=_L1_WRITE_TAG
                )
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
                if state.is_split_tier:
                    self._invalidate_split_tier_pending(state)
                if state.v_scratch_tensors:
                    self._return_v_scratch(state.v_scratch_tensors)
                    state.v_scratch_tensors = []
                self._finalize_store(wrapped_id, success=False)
                continue

            # Serialize succeeded — transition temps write → read so inner
            # can safely read them during the store.
            self._l1_manager.finish_write_and_reserve_read(
                state.temp_keys, tag=_L1_WRITE_TAG
            )
            # For KV_SPLIT_TIER the inner adapter sees V child keys, not
            # logical keys, so V lives under a deterministic derived
            # name on L2 that the load path will re-derive.  For
            # KV_TOGETHER it's the logical keys as today.
            inner_store_keys = state.v_child_keys if state.is_split_tier else state.keys
            # Temps became read-locked above, and an adapter submission may
            # have side effects before it raises.  Flip phase before the call
            # so both close-time temp cleanup and split-tier failure cleanup
            # conservatively treat this as an attempted inner store.
            with self._lock:
                state.phase = _StorePhase.INNER_STORE
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
                if state.v_scratch_tensors:
                    self._return_v_scratch(state.v_scratch_tensors)
                    state.v_scratch_tensors = []
                self._finalize_store(wrapped_id, success=False)
                continue

            # Reverse-map insertion is locked so close cannot observe a
            # partially published successful submission.
            with self._lock:
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
                # Return V-scratch tensors to the slab regardless of
                # success: the codec finished with them either way.
                if state.v_scratch_tensors:
                    self._return_v_scratch(state.v_scratch_tensors)
                    state.v_scratch_tensors = []
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
            sel_keys: list[ObjectKey] = []
            for i in range(len(state.keys)):
                if bitmap.test(i):
                    src_objs.append(state.temp_objs[i])
                    dst_objs.append(state.dst_objs[i])
                    sel_keys.append(state.keys[i])

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
                    sel_keys,
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
        temp_keys = [make_temp_key(k) for k in keys]
        try:
            layout = serialized_layout_desc(
                MemoryLayoutDesc(
                    shapes=objects[0].get_shapes(),
                    dtypes=objects[0].get_dtypes(),
                ),
                self._serde,
            )
            results = self._l1_manager.reserve_write(
                keys=temp_keys,
                is_temporary=[True] * len(temp_keys),
                layout_desc=layout,
                tag=_L1_WRITE_TAG,
            )
        except (MemoryError, RuntimeError, ValueError):
            # A serializer can reject a layout, or allocation can raise
            # before returning its per-key result. Release any reservations
            # before handing failure back to the caller's child rollback.
            logger.exception("Serde wrapper: temp-buffer allocation failed")
            self._release_write_temps(temp_keys)
            return temp_keys, None
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
        # ``set_used_size`` must distinguish estimate-sized serialized
        # buffers from fixed-layout one-byte KV tensors (FP8/int8).  Mark
        # only wrapper-owned temp objects as variable-length byte buffers;
        # the allocator restores its normal format when the block is reused.
        for obj in temp_objs:
            # ``metadata`` is part of MemoryObj's public interface.  Keep
            # duck-typed test/third-party objects compatible: objects that do
            # not expose metadata also cannot implement TensorMemoryObj's
            # narrowing semantics, so AsyncSerdeProcessor will leave them
            # alone through its existing ``set_used_size`` guard.
            if hasattr(obj, "metadata"):
                obj.metadata.fmt = MemoryFormat.BINARY_BUFFER
        return temp_keys, temp_objs

    def _release_write_temps(self, temp_keys: list[ObjectKey]) -> None:
        """Atomically release write-locked temps and delete them. No-op on empty."""
        if not temp_keys:
            return
        try:
            self._l1_manager.finish_write_and_delete(temp_keys, tag=_L1_WRITE_TAG)
        except Exception:
            logger.exception("Serde wrapper: failed releasing write-locked temps")

    # ------------------------------------------------------------------
    # Multi-output dispatch: build per-slot GroupSlotView tuples when
    # the underlying serde is multi-output (e.g. AsymK16V8Multi*), or
    # pass MemoryObjs through unchanged for single-tensor serdes.  The
    # mapping is queried from the SerdeProcessor (default ``None`` means
    # single-tensor; a non-None tuple defines the per-slot ↔ parent-group
    # routing -- e.g. identity ``(0, 1)`` for together placement,
    # ``(None, 1)`` for split-tier placement).
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
    ) -> tuple[list[ObjectKey], list[ObjectKey], dict[ObjectKey, int]]:
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

        Returns ``(k_child_keys, v_child_keys, generations)`` on
        success, where ``generations`` maps each logical key to the
        manifest generation :meth:`register_pending` assigned it (the
        caller threads it into every later cleanup call).  Returns
        ``([], [], {})`` if any L1 allocation fails or the input
        ``objects`` aren't grouped (sanity check -- shouldn't happen
        if placement is correctly KV_SPLIT_TIER, but raises a clear
        error instead of corrupting L1).
        """
        if not objects:
            return [], [], {}
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
        # Split-tier requires EXACTLY one post-policy (K, V) component
        # pair: shapes == [K_shape, V_shape].  len < 2 is a non-grouped
        # (packed) object -- a placement/layout mismatch.  len > 2 is a
        # multi-kernel-group object (e.g. [g0K, g0V, g1K, g1V] from a
        # model with more than one KV cache group); the store path only
        # mirrors group 0 -> K child and reads group 1 -> V, so admitting
        # it would SILENTLY DROP every kernel group beyond the first
        # pair.  Fail closed on both: the single-KV-pair topology
        # is static per StorageManager, so a mismatch is a hard config
        # error, not a transient failure.
        if shapes is None or dtypes is None or len(shapes) != 2:
            logger.error(
                "Serde wrapper: split-tier requires exactly one post-policy "
                "(K, V) component pair; got shapes=%r, dtypes=%r.  A "
                "non-grouped object is a placement/layout mismatch; more "
                "than one pair is a multi-KV-group model topology outside "
                "the split-tier support matrix.",
                shapes,
                dtypes,
            )
            return [], [], {}

        k_child_keys = [derive_component_key(k, "k") for k in keys]
        v_child_keys = [derive_component_key(k, "v") for k in keys]

        # K-only layout descriptor (group 0 of the staging object).
        k_layout = MemoryLayoutDesc(
            shapes=[shapes[0]],
            dtypes=[dtypes[0]],
        )

        # Try the K-child slab first.  The slab carries pre-allocated
        # K-shape ``TensorMemoryObj``s with itself as parent_allocator,
        # so L1Manager skips ``L1MemoryManager.allocate``'s address-
        # manager scan and the eventual L1Manager.delete returns the
        # buffer to the slab's deque (no fragmentation in the address
        # manager).  Falls back to the default reserve_write path if
        # the slab is exhausted or shape-mismatched.
        slab = self._ensure_k_child_slab(shapes[0], dtypes[0])
        slab_objs = (
            slab.batched_allocate([shapes[0]], [dtypes[0]], len(k_child_keys))
            if slab is not None
            else None
        )
        if slab_objs is not None and len(slab_objs) == len(k_child_keys):
            results = self._l1_manager.reserve_external_writes(
                keys=k_child_keys,
                memory_objs=slab_objs,
                is_temporary=[False] * len(k_child_keys),
                tag=_L1_WRITE_TAG,
            )
            # If reserve_external_writes hit a collision on any key
            # (KEY_NOT_WRITABLE -- the K-child already exists from a
            # prior incomplete store), recycle the unused slab objects
            # so they don't leak.
            # slab_objs is non-None only when slab was non-None above.
            assert slab is not None
            for k_child_key, obj in zip(k_child_keys, slab_objs, strict=True):
                r = results.get(k_child_key)
                if r is None or r[0] != L1Error.SUCCESS:
                    slab.free(obj)
        else:
            results = self._l1_manager.reserve_write(
                keys=k_child_keys,
                is_temporary=[False] * len(k_child_keys),
                layout_desc=k_layout,
                tag=_L1_WRITE_TAG,
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
            return [], [], {}

        # Copy K bytes from each staging object's group-0 tensor into
        # the K child.  torch's .copy_() between CPU tensors releases
        # the GIL, so we fan the per-key copies across a small thread
        # pool to overlap them on multi-CPU hosts (the StoreController
        # submits batched calls so N>1 is common).
        def _copy_one(idx: int) -> bool:
            try:
                k_child_obj = results[k_child_keys[idx]][1]
                if k_child_obj is None:
                    return False
                src_k = objects[idx].get_tensor(0)
                dst_k = k_child_obj.get_tensor(0)
                if src_k is None or dst_k is None:
                    return False
                dst_k.copy_(src_k)
                return True
            except Exception:
                logger.exception(
                    "Serde wrapper: split-tier K copy failed for key %s",
                    keys[idx] if idx < len(keys) else idx,
                )
                return False

        if self._split_tier_copy_pool is not None and len(keys) > 1:
            ok_list = list(self._split_tier_copy_pool.map(_copy_one, range(len(keys))))
        else:
            # Single-key batch or no pool: avoid the dispatch overhead.
            ok_list = [_copy_one(i) for i in range(len(keys))]
        if not all(ok_list):
            self._release_split_tier_k_children(successful)
            return [], [], {}

        # Register the manifest BEFORE finish_write so the
        # StoreController's listener filter (is_k_child_key) sees
        # the K-child key when the L1 manager's on_l1_keys_write_finished
        # fires.  Otherwise the K-child key races into the pending
        # queue and gets routed back into submit_store_task as a
        # bare 1-shape MemoryObj, failing the multi-group check.
        #
        # All-or-nothing: register_pending raises on a key with an
        # ACTIVE generation (concurrent duplicate store, or paired
        # cleanup mid-delete).  Roll back the keys registered so far
        # and fail the whole task.  Order matters: release the K
        # children FIRST (a no-drain release that does not depend on
        # the side-set) and only THEN drop the manifest entries for
        # the generations we own, so a concurrent generation's entry
        # is never dropped by this rollback.
        generations: dict[ObjectKey, int] = {}
        try:
            for logical_key in keys:
                generations[logical_key] = self._split_tier_manifest.register_pending(
                    logical_key
                )
        except ValueError:
            logger.warning(
                "Serde wrapper: split-tier registration collided on an "
                "in-flight generation (%d/%d keys registered); failing "
                "the store task",
                len(generations),
                len(keys),
            )
            self._release_split_tier_k_children(successful)
            for logical_key, generation in generations.items():
                self._split_tier_manifest.drop(logical_key, generation)
            return [], [], {}

        # Commit the K-child writes.  After this they are read-only
        # cache residents under their child keys.
        self._l1_manager.finish_write(k_child_keys, tag=_L1_WRITE_TAG)
        return k_child_keys, v_child_keys, generations

    def _alloc_v_scratch_and_copy(
        self,
        objects: list[MemoryObj],
    ) -> list[torch.Tensor]:
        """Borrow V-scratch tensors from the slab and copy V bytes from
        each producer-side logical entry's group-1 tensor.

        After this returns successfully, V is no longer needed from the
        logical L1 entries -- the V codec will read V exclusively from
        the slab tensors via :class:`_VScratchSlot`.

        Args:
            objects: Producer-side logical L1 ``MemoryObj`` instances
                that ``submit_store_task`` received.  Each must have a
                group-1 (V) tensor; the wrapper has already validated
                grouping in :meth:`_alloc_split_tier_children`.

        Returns:
            A list of CPU torch tensors (one per input object) holding
            a private copy of V.  Returns ``[]`` if any object's V
            tensor is unavailable; partial copies are returned to the
            slab before the failure path.
        """
        if not objects:
            return []
        out: list[torch.Tensor] = []
        try:
            # Pool construction allocates tensor metadata too, so it belongs
            # to the same rollback boundary as acquiring and copying slots.
            first_v = objects[0].get_tensor(1)
            if first_v is None:
                return []
            slab = self._ensure_v_scratch_slab(first_v.shape, first_v.dtype)
            for obj in objects:
                v = obj.get_tensor(1)
                if v is None or v.shape != first_v.shape or v.dtype != first_v.dtype:
                    raise ValueError(
                        "Serde wrapper: one store batch must have a uniform "
                        "V shape and dtype"
                    )
                scratch = slab.acquire()
                out.append(scratch)
                scratch.copy_(v)
        except Exception:
            logger.exception("Serde wrapper: V-scratch allocation/copy failed")
            self._return_v_scratch(out)
            return []
        return out

    def _ensure_k_child_slab(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> _KChildSlab | None:
        """Build the K-child slab on first split-tier store.

        Shape-pinned at first use; subsequent batches with the same
        K shape hit the slab, others fall through to the regular
        reserve_write path (the slab's :meth:`_KChildSlab.allocate`
        ignores the requested shape because we *also* check
        ``shapes[0]`` matches the slab's pinned shape here -- a
        mismatch returns ``None`` to make the caller take the regular
        path).
        """
        slab = self._k_child_slab
        if slab is not None:
            if slab.shape == shape and slab.dtype == dtype:
                return slab
            # Shape mismatch (a fresh wrapper would see one shape for
            # its lifetime; this branch is defensive).
            return None
        with self._k_child_init_lock:
            if self._k_child_slab is None:
                self._k_child_slab = _KChildSlab(
                    shape=shape,
                    dtype=dtype,
                    max_slots=self._k_child_max_slots,
                )
                # Make the slab's bytes visible to L1's get_memory_usage
                # so the eviction policy reads true L1 pressure (the
                # slab provides backing for K-child residents that
                # bypass L1Manager's own allocator -- without this hook
                # those bytes are invisible to LRU and eviction never
                # fires under K-child pressure).
                try:
                    self._l1_manager.register_external_memory_provider(
                        self._k_child_slab
                    )
                except Exception:
                    # Older L1Manager without the hook -- log and
                    # continue.  The slab still functions; only the
                    # accounting visibility is lost.
                    logger.warning(
                        "Serde wrapper: L1Manager has no "
                        "register_external_memory_provider hook; "
                        "K-child slab bytes will not be visible to "
                        "the eviction policy"
                    )
            return self._k_child_slab

    def _ensure_v_scratch_slab(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> _VScratchSlab:
        """Return the V-scratch slab for an exact shape/dtype layout.

        A wrapper can receive distinct model/layout buckets over its
        lifetime.  Pools are keyed by exact shape and dtype so no store is
        broadcast or numerically converted before serialization.
        """
        key = (tuple(shape), dtype)
        slab = self._v_scratch_slabs.get(key)
        if slab is not None:
            return slab
        with self._v_scratch_init_lock:
            slab = self._v_scratch_slabs.get(key)
            if slab is None:
                slab = _VScratchSlab(
                    shape=shape,
                    dtype=dtype,
                    max_slots=self._v_scratch_max_slots,
                )
                self._v_scratch_slabs[key] = slab
                # Register V-scratch with L1's usage aggregation too.
                # V-scratch tensors live for the duration of the V codec
                # + L2 write; under saturation a meaningful number are
                # in flight at once (~1.5 GiB at 3000 chunks * 512 KiB),
                # and the LRU policy needs to see them as L1 pressure.
                try:
                    self._l1_manager.register_external_memory_provider(slab)
                except Exception:
                    logger.warning(
                        "Serde wrapper: L1Manager has no "
                        "register_external_memory_provider hook; "
                        "V-scratch slab bytes will not be visible to "
                        "the eviction policy"
                    )
            return slab

    def _return_v_scratch(self, tensors: list[torch.Tensor]) -> None:
        """Return slab-borrowed V-scratch tensors after the codec is done."""
        if not tensors:
            return
        for t in tensors:
            slab = self._v_scratch_slabs.get((tuple(t.shape), t.dtype))
            if slab is not None:
                slab.release(t)

    def _release_split_tier_k_children(self, k_child_keys: list[ObjectKey]) -> None:
        """Release K-child L1 entries on a partial / failed store path
        so K children don't orphan in L1 with no V partner.

        Critically this must NOT fire the StoreController's L1→L2
        drain listener (``on_l1_keys_write_finished``).  A plain
        ``finish_write`` would fire it, and on the failure paths that
        call this the ``is_k_child_key`` side-set may not yet be
        populated (this runs before :meth:`register_pending` in the
        K-child alloc/copy-failure branches) or already dropped (the
        registration-collision rollback), so the drain would route
        these write-locked K children to L2 unfiltered -- the exact
        multi-group re-entry the side-set exists to prevent, plus a
        read-locked K-child leak if the store loop wins the race.

        Staged children are discarded atomically without making them
        resident or notifying the listener. K children already committed
        by a successful ``finish_write`` are removed by the subsequent
        ``delete`` call.
        """
        if not k_child_keys:
            return
        try:
            # Staged children are discarded atomically without becoming
            # resident or emitting write-finished notifications. Resident
            # children return KEY_NOT_EXIST here and are removed below.
            self._l1_manager.finish_write_and_delete(k_child_keys, tag=_L1_WRITE_TAG)
        except Exception:
            logger.debug(
                "Serde wrapper split-tier release: staging discard raised; "
                "continuing to delete %d K children",
                len(k_child_keys),
            )
        try:
            self._l1_manager.delete(k_child_keys)
        except Exception:
            logger.exception(
                "Serde wrapper split-tier release: delete raised for %d K children",
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

        Each ``mark_complete`` is guarded by the generation this task
        registered, so a completion racing a re-store never marks the
        newer generation's entry COMPLETE off this task's stale ack.
        """
        completed_models: Counter[str] = Counter()
        for logical_key in state.keys:
            generation = state.split_tier_generations.get(logical_key)
            if generation is None:
                continue
            try:
                self._split_tier_manifest.mark_complete(logical_key, generation)
            except ValueError:
                # Preempted by an invalidate from the eviction
                # controller, or the entry was reclaimed by a
                # newer generation.  The K child + manifest entry
                # cleanup happens through the invalidation path (for
                # the entry we owned) or belongs to the newer
                # generation; here we just refuse to mark COMPLETE.
                logger.debug(
                    "Serde wrapper split-tier: logical key already "
                    "invalidated / superseded before COMPLETE; skipping"
                )
                continue
            completed_models[logical_key.model_name] += 1
        if completed_models:
            get_event_bus().publish(
                Event(
                    event_type=EventType.SPLIT_TIER_STORE_COMPLETED,
                    metadata={
                        "count": sum(completed_models.values()),
                        "model_names": completed_models,
                    },
                )
            )

    def _invalidate_split_tier_pending(self, state: _StoreTaskState) -> None:
        """Fence and clean the physical children of a failed split-tier store.

        Each generation first enters ``DELETE_IN_FLIGHT`` atomically.  New
        stores cannot reuse the generation-less child names while a blocking
        inner V delete or K-child release is outstanding.  After both children
        are gone, dropping the fenced manifest entry permits a later retry.

        Every manifest operation is generation guarded.  If ownership has
        already changed, no physical child belonging to the replacement is
        touched.

        The inner V-child delete is issued once inner submission was attempted
        (``phase == INNER_STORE``), including an exception from
        ``submit_store_task`` because the adapter may have produced side
        effects before raising.  A serialize failure remains pre-submit and
        skips the blocking delete because no V bytes could have landed.
        """
        inner_store_submitted = state.phase == _StorePhase.INNER_STORE
        owned_keys: list[ObjectKey] = []
        for logical_key in state.keys:
            generation = state.split_tier_generations.get(logical_key)
            if generation is None:
                continue
            previous = self._split_tier_manifest.begin_physical_cleanup(
                logical_key,
                generation,
                (SplitTierState.STORE_IN_FLIGHT, SplitTierState.INVALIDATED),
            )
            if previous is not None:
                owned_keys.append(logical_key)
        if owned_keys:
            self._publish_split_tier_store_invalidated(
                owned_keys,
                "inner_store" if inner_store_submitted else "pre_submit",
            )
        if owned_keys and inner_store_submitted:
            # The inner store was submitted and failed, but a failure
            # result doesn't guarantee no bytes landed -- delete the V
            # child names so a partial blob can't be mistaken for a
            # valid V by a future generation's load.  Restrict to keys
            # whose cleanup fence we acquired; a superseding generation is
            # never touched.  Best-effort: adapter failures leave tolerable
            # cold-tier waste, but the manifest still fails closed.
            v_child_keys = [derive_component_key(k, "v") for k in owned_keys]
            try:
                self._inner.delete(v_child_keys)
            except Exception:
                logger.exception(
                    "Serde wrapper split-tier cleanup: inner delete "
                    "raised for %d V children",
                    len(v_child_keys),
                )
            get_event_bus().publish(
                Event(
                    event_type=EventType.SPLIT_TIER_V_CHILD_DELETED,
                    metadata={
                        "count": len(v_child_keys),
                        "trigger": "store_failure",
                    },
                )
            )
        owned_k_children = [derive_component_key(k, "k") for k in owned_keys]
        self._release_split_tier_k_children(owned_k_children)
        for logical_key in owned_keys:
            generation = state.split_tier_generations.get(logical_key)
            if generation is not None:
                self._split_tier_manifest.drop(logical_key, generation)

    def _drop_split_tier_pending(
        self, keys: list[ObjectKey], generations: dict[ObjectKey, int]
    ) -> None:
        """Submit-side failure AFTER ``register_pending`` but BEFORE
        any inner L2 submission: the K children are released by the
        caller and no V bytes were ever sent, so no physical state
        remains -- drop the entries outright so the logical keys can
        be stored again.

        Each drop is guarded by the generation this task registered
        (from :meth:`_alloc_split_tier_children`), so it never removes
        an entry a concurrent re-store has already reclaimed.

        Args:
            keys: Logical keys whose pending manifest entries to drop.
            generations: Per-key generation ids from
                ``register_pending``.
        """
        for logical_key in keys:
            generation = generations.get(logical_key)
            if generation is not None:
                self._split_tier_manifest.drop(logical_key, generation)

    @staticmethod
    def _publish_split_tier_store_invalidated(
        keys: list[ObjectKey], reason: str
    ) -> None:
        """Publish one terminal split-tier failure outcome per logical key.

        Args:
            keys: Logical keys whose store attempt failed.
            reason: Stable metric reason label such as ``pre_submit`` or
                ``inner_store``.
        """
        if not keys:
            return
        invalidated_models: Counter[str] = Counter(k.model_name for k in keys)
        get_event_bus().publish(
            Event(
                event_type=EventType.SPLIT_TIER_STORE_INVALIDATED,
                metadata={
                    "count": len(keys),
                    "reason": reason,
                    "model_names": invalidated_models,
                },
            )
        )

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
        invalidate the manifest entry here (paired
        eviction) is responsible for that lifecycle.
        """
        # Surviving indices = those still set in the bitmap after the
        # manifest mask.
        surviving = [i for i in range(len(state.keys)) if bitmap.test(i)]
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
                if src_k.shape != dst_k.shape or src_k.dtype != dst_k.dtype:
                    logger.warning(
                        "Serde wrapper split-tier load: K child layout "
                        "mismatch for logical key %s (stored %s/%s, "
                        "destination %s/%s); treating as miss",
                        state.keys[idx],
                        tuple(src_k.shape),
                        src_k.dtype,
                        tuple(dst_k.shape),
                        dst_k.dtype,
                    )
                    bitmap.clear(idx)
                    continue
                dst_k.copy_(src_k)
            except Exception:
                logger.exception(
                    "Serde wrapper split-tier load: K copy failed for logical key %s",
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
        return [  # type: ignore[return-value]
            tuple(
                GroupSlotView(obj, idx) if idx is not None else None for idx in mapping
            )
            for obj in objects
        ]

    def _serde_mapping_covers_objects(
        self,
        objects: list[MemoryObj],
        mapping: tuple[int | None, ...] | None,
        *,
        role: str,
    ) -> bool:
        """Validate that a KV-together serde covers every parent group.

        A multi-output mapping such as ``(0, 1)`` describes exactly one
        K/V pair.  The layout policy can produce ``[K, V, K, V, ...]`` for
        multiple kernel groups; silently serializing only indexes 0 and 1
        would report a hit with later groups untouched.  Single-tensor
        serdes have no slot mapping and retain their existing contract.

        Args:
            objects: Parent memory objects submitted for store or load.
            mapping: Serde slot-to-parent mapping, or ``None``.
            role: Human-readable operation side for diagnostics.

        Returns:
            ``True`` when every object's groups are covered exactly once.
        """
        if mapping is None:
            return True
        referenced = [idx for idx in mapping if idx is not None]
        for obj in objects:
            group_count = len(obj.get_shapes())
            if sorted(referenced) != list(range(group_count)):
                logger.error(
                    "Serde wrapper: %s mapping %r does not cover all %d "
                    "parent groups exactly once; refusing an incomplete "
                    "KV_TOGETHER operation",
                    role,
                    mapping,
                    group_count,
                )
                return False
        return True

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
        return [  # type: ignore[return-value]
            tuple(
                GroupSlotView(obj, idx) if idx is not None else None for idx in mapping
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
