# SPDX-License-Identifier: Apache-2.0
"""
Storage placement policy: where does each component of a multi-output
serde's output live, and what is the lifecycle of the components in
L1?

This is a different decision from
:mod:`lmcache.v1.distributed.storage_layout`, which decides the
*shape* of L1 ``MemoryObj`` allocations (single packed tensor vs
multi-group K and V). Placement decides the *placement and
lifecycle* of those components after the serde encodes them.

Two modes today:

* :attr:`StoragePlacementMode.KV_TOGETHER` — the serde emits one
  encoded blob containing K + V (current default for fp8 and asym
  Mode 1 K16/V8). Wrapper stores the single blob to L2; L1 retains
  the original full K+V allocation under the logical key (the rest
  of LMCache's lookup/eviction machinery treats it as one object).

* :attr:`StoragePlacementMode.KV_SPLIT_TIER` — Mode 2 V-only. After
  the wrapper serializes, the lifecycle drives:

  - V child object → L2 under a derived V child key.
  - K child object → retained in L1 under a derived K child key
    (the K bytes of the original staging object, narrowed).
  - Original full K+V staging object → deleted in L1, so L1
    footprint per cached chunk drops from K+V → K-only.

  Lookup under the logical key returns a "composite hit" iff BOTH
  the K child is in L1 AND the V child is in L2. Load reassembles
  the full logical object from the two children.

Why placement is a separate concept from layout (codex push-back,
2026-05-31): ``KV_COMPONENT_GROUPS`` only declares that K and V are
typed sub-objects in the L1 allocation. Whether the two children
get routed to different storage tiers, retained under separate
child keys, and have independent lifecycles is a placement /
state-machine decision. The two are orthogonal: a future serde
could produce K+V components but choose to ship both to L2 (still
"together" from placement's view), and split-tier could in theory
work even on a packed shape if someone bothered to encode an
in-place split.

The split-tier state machine
-----------------------------

For each logical KV chunk under :attr:`KV_SPLIT_TIER`:

1. ``Producer`` writes full logical K+V into a staging
   ``TensorMemoryObj`` in L1, keyed by the **logical** key
   (no change from the producer's perspective).
2. ``Wrapper.submit_store_task`` runs the V-only serde, materializes
   the K child under :func:`derive_component_key` (role ``"k"``),
   submits the V child to L2 under :func:`derive_component_key`
   (role ``"v"``), then deletes the original staging object in L1.
3. ``Lookup`` of the logical key resolves through the manifest:
   ``COMPLETE`` iff K child is present in L1 and V child is
   reachable via the L2 adapter.
4. ``Load`` re-assembles the full logical object by copying K
   bytes from L1 and decoding V bytes from L2 into a freshly
   allocated staging object handed back to vLLM.
5. ``L1 eviction`` invalidates the manifest entry first (lookup
   immediately returns miss), then deletes the K child in L1 and
   enqueues the paired V child for L2 deletion.

True cross-tier atomicity is not available through the current
``L2AdapterInterface`` — adapters' ``delete()`` is optional and may
be a no-op. The "atomic" eviction policy is therefore implemented
as **logical invalidation first, physical cleanup second**: the
manifest entry transitions to ``INVALIDATED`` synchronously, so
readers see consistent state; the K child and V child are
reclaimed in the background (best-effort for V on adapters without
``delete()``, accepting orphaned cold-tier waste).
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.l2_adapters.config import L2AdapterConfigBase


class StoragePlacementMode(Enum):
    """Per-StorageManager placement policy for serde-encoded output."""

    KV_TOGETHER = "kv_together"
    """The serde emits one encoded blob covering both K and V (the
    default for ``fp8`` and Mode 1 ``asym_k16_v8``).  The wrapper
    stores the single blob under the logical key and L1 retains the
    original full K+V allocation as before."""

    KV_SPLIT_TIER = "kv_split_tier"
    """Mode 2 V-only.  The wrapper drives the split-tier state
    machine: K child retained in L1 under a derived key, V child
    stored to L2 under a derived key, original logical staging
    object deleted in L1.  Lookup is a composite check; load
    reassembles.  L1 eviction is paired and logically atomic."""


# Domain-separation marker bytes appended to ``ObjectKey.chunk_hash``
# when deriving K / V child keys.  Original logical keys never carry
# a trailing role byte, so the length of the chunk_hash field
# (typically 32 bytes for SHA-256) uniquely distinguishes child keys
# from logical keys -- no collision is possible regardless of the
# original hash content.
_K_CHILD_MARKER = b"\x01"
_V_CHILD_MARKER = b"\x02"


def derive_component_key(logical_key: ObjectKey, role: str) -> ObjectKey:
    """Derive a child :class:`ObjectKey` from a logical key + role.

    Preserves every identity field of the logical key --
    ``model_name``, ``kv_rank``, ``object_group_id``, and
    ``cache_salt`` (per-tenant adapter accounting and quotas depend
    on ``cache_salt``; multi-object-group models depend on
    ``object_group_id`` -- do NOT drop either).  Domain-separates via
    the ``chunk_hash`` field by appending a 1-byte role marker.

    Args:
        logical_key: The original logical chunk key.
        role: ``"k"`` or ``"v"`` -- selects which child key is
            derived.

    Returns:
        A new :class:`ObjectKey` whose ``chunk_hash`` is
        ``logical_key.chunk_hash + role_marker``.  Length differs
        from the logical key's hash by exactly one byte, so child
        keys never collide with logical keys regardless of the
        original hash content.

    Raises:
        ValueError: if ``role`` is not ``"k"`` or ``"v"``.
    """
    if role == "k":
        marker = _K_CHILD_MARKER
    elif role == "v":
        marker = _V_CHILD_MARKER
    else:
        raise ValueError(f"derive_component_key: role must be 'k' or 'v', got {role!r}")
    return ObjectKey(
        chunk_hash=logical_key.chunk_hash + marker,
        model_name=logical_key.model_name,
        kv_rank=logical_key.kv_rank,
        object_group_id=logical_key.object_group_id,
        cache_salt=logical_key.cache_salt,
    )


def reverse_component_key(
    child_key: ObjectKey,
) -> Optional[tuple[ObjectKey, str]]:
    """Inverse of :func:`derive_component_key`.

    Given a possibly-child :class:`ObjectKey`, return
    ``(logical_key, role)`` if its trailing byte matches a known
    role marker, otherwise return ``None``.  Used by the L1
    eviction controller to identify K-child victims and look up
    the matching V child for paired cleanup (PR-3').

    The contract: a logical key's ``chunk_hash`` never carries a
    role marker as its trailing byte, so length is the
    discriminator -- a 33-byte hash ending in 0x01 or 0x02 is
    unambiguously a child key; anything else is a logical key
    (or a stray hash from another mode).

    Args:
        child_key: A key that may or may not be a derived child.

    Returns:
        ``(logical, role)`` if ``child_key`` is a recognizable
        K or V child; otherwise ``None``.
    """
    h = child_key.chunk_hash
    if len(h) < 2:
        return None
    marker = h[-1:]
    if marker == _K_CHILD_MARKER:
        role = "k"
    elif marker == _V_CHILD_MARKER:
        role = "v"
    else:
        return None
    logical = ObjectKey(
        chunk_hash=h[:-1],
        model_name=child_key.model_name,
        kv_rank=child_key.kv_rank,
        object_group_id=child_key.object_group_id,
        cache_salt=child_key.cache_salt,
    )
    return logical, role


@dataclass
class _ManifestEntry:
    """One logical key's manifest state plus the generation that owns
    it.

    ``generation`` is a process-unique, monotonically increasing id
    assigned by :meth:`SplitTierManifest.register_pending`.  It lets
    every cleanup path act on *only* the generation it started for:
    a stale delete/invalidate for generation ``G`` becomes a no-op
    the instant a fresh :meth:`register_pending` has reclaimed the
    key under generation ``G+1``.  Without it, a cleanup racing a
    re-store of the same logical key could tear down the brand-new
    composite it never owned.
    """

    state: "SplitTierState"
    generation: int


class SplitTierManifest:
    """Thread-safe manifest of split-tier state per logical
    :class:`ObjectKey`.

    Owned by :class:`StorageManager` (it lives there, not in the
    wrapper, so it composes with eviction and quota accounting).
    The wrapper queries / mutates it through the StorageManager's
    interface during store and load.

    Operations are linear-forward WITHIN one store generation
    (mirrors :class:`SplitTierState`): ``STORE_IN_FLIGHT`` →
    ``COMPLETE`` → ``INVALIDATED`` → ``DELETE_IN_FLIGHT``.  A key
    never moves to an earlier state while its generation is active;
    invalid transitions raise :class:`ValueError` so a race in the
    wrapper surfaces immediately rather than silently corrupting
    the lifecycle.  A NEW generation begins when
    :meth:`register_pending` reclaims a key whose previous
    generation ended (``COMPLETE`` gone stale or ``INVALIDATED``
    after failed-store cleanup) -- a store failure must never
    permanently block a key from being stored again.

    **Generation guard.**  :meth:`register_pending` returns a
    process-unique, monotonically increasing generation id, and
    every mutator that a *cleanup* path uses
    (:meth:`mark_complete`, :meth:`mark_invalidated`,
    :meth:`mark_delete_in_flight`, :meth:`drop`) takes that
    generation and only acts when it still matches the entry's
    current generation.  This closes the class of races where a
    delete/invalidate started for one store transition lands *after*
    the same logical key has been re-registered by a newer store
    (the eviction controller deleting the old K child, then a fresh
    store re-registering under the same key before the eviction's
    teardown runs).  A generation-mismatched cleanup is a silent
    no-op: it never touches the newer generation's state.

    Lookup against an unregistered logical key returns ``None``;
    callers MUST treat that as "miss" (the composite cache entry
    does not exist).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[ObjectKey, _ManifestEntry] = {}
        # Monotonic, process-unique generation counter.  Never reset
        # (not even when an entry is dropped) so a generation number
        # is never reused -- a stale cleanup for an old generation can
        # therefore never collide with a fresh entry.
        self._next_generation = 1
        # Side set of currently-tracked K child keys, populated when
        # the wrapper registers a logical key as STORE_IN_FLIGHT.  The
        # StoreController consults this set to skip K_child write
        # completions (K_child is L1-only by design; it must never be
        # routed to L2, otherwise the wrapper's finish_write triggers
        # a recursive submit_store_task that fails on the K-only
        # single-shape MemoryObj).
        self._k_child_keys: set[ObjectKey] = set()

    def register_pending(self, logical_key: ObjectKey) -> int:
        """Mark a logical key as ``STORE_IN_FLIGHT`` under a fresh
        generation and return that generation id.

        Also records the derived K-child key in
        :meth:`is_k_child_key`'s lookup set so the StoreController
        can suppress L2 routing for that K child (the K stays in L1
        as the canonical tier).

        Keys whose previous store generation has ENDED are reclaimed
        rather than rejected: a stale ``COMPLETE`` (the K child was
        cleared or evicted out from under the manifest -- lookups
        already miss because the composite requires the K child) or
        ``INVALIDATED`` (a failed store whose cleanup finished)
        entry is overwritten by the new generation.  A permanent
        refusal here is worse than the race it guards against: it
        turns one failed store into a per-key denial of caching for
        the lifetime of the process.

        Returns:
            The generation id assigned to this store transition.  The
            caller MUST pass it back to :meth:`mark_complete`,
            :meth:`mark_invalidated`, :meth:`mark_delete_in_flight`,
            and :meth:`drop` so those cleanup paths act only on this
            generation.

        Raises:
            ValueError: if the key has an ACTIVE operation in flight
                (``STORE_IN_FLIGHT``: a concurrent duplicate store is
                a wrapper bug -- surface it loudly;
                ``DELETE_IN_FLIGHT``: paired cleanup is actively
                deleting the previous generation's children and a new
                write would race it -- the caller fails this store
                and a later retry succeeds after :meth:`drop`).
        """
        with self._lock:
            cur = self._entries.get(logical_key)
            if cur is not None and cur.state in (
                SplitTierState.STORE_IN_FLIGHT,
                SplitTierState.DELETE_IN_FLIGHT,
            ):
                raise ValueError(
                    f"SplitTierManifest.register_pending: key already "
                    f"tracked in state {cur.state.name}"
                )
            generation = self._next_generation
            self._next_generation += 1
            self._entries[logical_key] = _ManifestEntry(
                state=SplitTierState.STORE_IN_FLIGHT,
                generation=generation,
            )
            self._k_child_keys.add(derive_component_key(logical_key, "k"))
            return generation

    def mark_complete(self, logical_key: ObjectKey, generation: int) -> None:
        """Transition ``STORE_IN_FLIGHT`` → ``COMPLETE`` for a
        specific generation.

        The inner L2 store has acknowledged; lookup may now resolve
        as a composite hit.

        Args:
            logical_key: The logical key to complete.
            generation: The generation returned by the matching
                :meth:`register_pending`.

        Raises:
            ValueError: if the key is untracked, is owned by a
                different (newer) generation, or is not in
                ``STORE_IN_FLIGHT``.  A generation mismatch means
                this store was superseded by a newer one and must
                not resurrect a stale composite.
        """
        with self._lock:
            cur = self._entries.get(logical_key)
            if cur is None or cur.generation != generation:
                raise ValueError(
                    f"SplitTierManifest.mark_complete: key untracked or "
                    f"owned by generation "
                    f"{cur.generation if cur else 'NONE'}; expected "
                    f"generation {generation}"
                )
            if cur.state != SplitTierState.STORE_IN_FLIGHT:
                raise ValueError(
                    f"SplitTierManifest.mark_complete: key in state "
                    f"{cur.state.name}; expected STORE_IN_FLIGHT"
                )
            cur.state = SplitTierState.COMPLETE

    def mark_invalidated(self, logical_key: ObjectKey, generation: int) -> bool:
        """Invalidate a logical key iff ``generation`` still owns it.

        Compare-and-set on the generation: transitions the entry to
        ``INVALIDATED`` only when the tracked generation matches the
        caller's.  Lookup returns "miss" from that point on; physical
        cleanup (K child in L1 + V child on L2) proceeds in the
        background per the caller's policy.  Acceptable to invalidate
        from any state at the matching generation (including
        ``STORE_IN_FLIGHT`` if a failed store needs to clean up);
        already-``INVALIDATED`` / ``DELETE_IN_FLIGHT`` at the same
        generation is idempotent.

        Args:
            logical_key: The logical key to invalidate.
            generation: The generation the caller believes it is
                cleaning up (from :meth:`register_pending`, or
                captured via :meth:`lookup_entry`).

        Returns:
            ``True`` if this call left the caller's generation
            invalidated (either it transitioned it now, or it was
            already invalidated / delete-in-flight at the SAME
            generation).  ``False`` if the key is untracked or has
            been reclaimed by a newer generation -- in which case the
            caller MUST NOT run any paired physical cleanup, since the
            resources now belong to that newer generation.
        """
        with self._lock:
            cur = self._entries.get(logical_key)
            if cur is None or cur.generation != generation:
                return False
            if cur.state in (
                SplitTierState.INVALIDATED,
                SplitTierState.DELETE_IN_FLIGHT,
            ):
                # Already invalidated / being deleted; idempotent.
                return True
            cur.state = SplitTierState.INVALIDATED
            return True

    def mark_delete_in_flight(self, logical_key: ObjectKey, generation: int) -> None:
        """Transition ``INVALIDATED`` → ``DELETE_IN_FLIGHT`` for a
        specific generation.

        K child deletion in L1 has completed; the V child delete is
        enqueued against the L2 adapter.  The manifest entry is
        removed via :meth:`drop` after the L2 delete acknowledges
        (or after a cleanup timeout for adapters without ``delete()``;
        the V orphan is tolerable cold-tier waste).

        Args:
            logical_key: The logical key being deleted.
            generation: The generation the caller is cleaning up.

        Raises:
            ValueError: if the key is untracked, owned by a different
                generation, or not in ``INVALIDATED``.
        """
        with self._lock:
            cur = self._entries.get(logical_key)
            if cur is None or cur.generation != generation:
                raise ValueError(
                    f"SplitTierManifest.mark_delete_in_flight: key "
                    f"untracked or owned by generation "
                    f"{cur.generation if cur else 'NONE'}; expected "
                    f"generation {generation}"
                )
            if cur.state != SplitTierState.INVALIDATED:
                raise ValueError(
                    f"SplitTierManifest.mark_delete_in_flight: key in "
                    f"state {cur.state.name}; expected INVALIDATED"
                )
            cur.state = SplitTierState.DELETE_IN_FLIGHT

    def drop(self, logical_key: ObjectKey, generation: int) -> None:
        """Remove the manifest entry for ``logical_key`` iff
        ``generation`` still owns it.

        Called after physical cleanup completes; no further state
        transitions are possible.  Idempotent.  Also drops the
        side-set entry for the K-child key.

        A generation mismatch is a silent no-op: the key has been
        reclaimed by a newer store, whose entry and K-child side-set
        membership MUST be preserved.

        Args:
            logical_key: The logical key whose entry to remove.
            generation: The generation the caller is cleaning up.
        """
        with self._lock:
            cur = self._entries.get(logical_key)
            if cur is None or cur.generation != generation:
                return
            self._entries.pop(logical_key, None)
            self._k_child_keys.discard(derive_component_key(logical_key, "k"))

    def is_k_child_key(self, key: ObjectKey) -> bool:
        """``True`` iff ``key`` is currently tracked as a K-child of
        a logical key in this manifest.

        The StoreController consults this to skip L2 routing for
        K-child write completions -- the K child is L1-canonical
        and must never be sent to L2 (the wrapper's
        ``submit_store_task`` would re-enter with a single-shape
        K-only object and fail the multi-group sanity check).
        """
        with self._lock:
            return key in self._k_child_keys

    def lookup(self, logical_key: ObjectKey) -> Optional["SplitTierState"]:
        """Return the manifest state for ``logical_key``, or ``None``.

        Threadsafe snapshot; callers that need to act on the result
        atomically should call the generation-guarded ``mark_*`` /
        :meth:`drop` immediately (the transitions are themselves
        atomic and locked, and no-op on a generation mismatch).
        """
        with self._lock:
            cur = self._entries.get(logical_key)
            return cur.state if cur is not None else None

    def lookup_entry(
        self, logical_key: ObjectKey
    ) -> Optional[tuple["SplitTierState", int]]:
        """Return ``(state, generation)`` for ``logical_key``, or
        ``None`` if untracked.

        Cleanup paths that do not own a generation (the eviction
        controller, :meth:`StorageManager.clear`) capture the
        generation here and pass it to the generation-guarded
        mutators so they only tear down the generation they observed.
        """
        with self._lock:
            cur = self._entries.get(logical_key)
            if cur is None:
                return None
            return cur.state, cur.generation

    def is_complete(self, logical_key: ObjectKey) -> bool:
        """Convenience: ``True`` iff lookup resolves to a composite hit."""
        return self.lookup(logical_key) == SplitTierState.COMPLETE

    def tracked_keys(self) -> list[ObjectKey]:
        """Return a snapshot of every tracked logical key.

        Used by :meth:`StorageManager.clear` to sweep entries whose
        K children were just removed from L1.  The snapshot is
        consistent at the time of the call; callers must tolerate
        entries transitioning (or being dropped) between the
        snapshot and any follow-up per-key operation.
        """
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class SplitTierState(Enum):
    """Manifest-entry state machine for a logical key under
    :attr:`StoragePlacementMode.KV_SPLIT_TIER`.

    Transitions are linear forward; an entry never moves back to
    an earlier state.  Lookup returns "composite hit" only in the
    :attr:`COMPLETE` state.
    """

    STORE_IN_FLIGHT = "store_in_flight"
    """The wrapper has begun the store transition (V submitted to
    L2, K child allocated in L1) but the L2 store has not yet
    completed.  Lookup returns miss; load is invalid."""

    COMPLETE = "complete"
    """Both K child is in L1 and V child is reachable on L2.
    Lookup returns composite hit; load is valid."""

    INVALIDATED = "invalidated"
    """The L1 eviction controller (or an explicit delete) has
    flagged this manifest entry as no longer addressable.  Lookup
    returns miss immediately; physical cleanup of K child + V child
    proceeds in the background (best-effort for V if the adapter
    has no ``delete()`` semantics)."""

    DELETE_IN_FLIGHT = "delete_in_flight"
    """K child deletion in L1 completed; V child delete is enqueued
    against the L2 adapter.  Manifest entry is removed when the L2
    delete acknowledges or after a configurable cleanup timeout (V
    orphans on no-delete adapters are tolerable)."""


def derive_storage_placement_mode(
    adapter_configs: list["L2AdapterConfigBase"],
) -> StoragePlacementMode:
    """Derive the canonical placement mode from configured L2 adapters.

    Inspects each adapter's ``serde_config`` and queries the
    corresponding :class:`SerdeProcessor`'s ``input_slot_mapping``.
    Today the placement rule is straightforward:

    * Any adapter using a multi-output serde whose mapping has at
      least one ``None`` slot (e.g. V-only's ``(None, 1)``) demands
      :attr:`KV_SPLIT_TIER`: the absent slot is the L1-canonical
      tier.
    * Everything else (no serde, single-tensor serde, multi-output
      serde with all slots non-None) demands :attr:`KV_TOGETHER`.

    All adapters must agree.  Mixing :attr:`KV_TOGETHER` and
    :attr:`KV_SPLIT_TIER` on the same StorageManager is rejected:
    the canonical L1 lifecycle differs and the wrapper cannot
    support both shapes at the same time.

    Args:
        adapter_configs: Sequence of L2 adapter configurations.
            Empty list returns :attr:`KV_TOGETHER`.

    Returns:
        The canonical placement mode.

    Raises:
        ValueError: if adapters demand incompatible modes.
    """
    # First Party
    from lmcache.v1.distributed.serde import create_serde_processor

    modes: set[StoragePlacementMode] = set()
    for ac in adapter_configs:
        sc = getattr(ac, "serde_config", None)
        if sc is None:
            modes.add(StoragePlacementMode.KV_TOGETHER)
            continue
        processor = create_serde_processor(sc)
        try:
            mapping = processor.input_slot_mapping()
        finally:
            processor.close()
        if mapping is None:
            modes.add(StoragePlacementMode.KV_TOGETHER)
            continue
        # Multi-output: if any slot is None, the absent slot is the
        # L1-canonical tier -> split-tier placement.  Otherwise
        # (e.g. asym Mode 1's (0, 1) identity) the serde packs both
        # into one blob -> together.
        if any(s is None for s in mapping):
            modes.add(StoragePlacementMode.KV_SPLIT_TIER)
        else:
            modes.add(StoragePlacementMode.KV_TOGETHER)

    if not modes:
        return StoragePlacementMode.KV_TOGETHER
    if len(modes) > 1:
        names = sorted(m.value for m in modes)
        raise ValueError(
            f"Incompatible L2 adapter storage placement modes: {names}. "
            f"All adapters must share one canonical placement / lifecycle; "
            f"a kv_together serde (e.g. fp8, asym_k16_v8 Mode 1) cannot "
            f"coexist with a kv_split_tier serde (e.g. asym_k16_v8_v_only) "
            f"on the same StorageManager."
        )
    return modes.pop()
