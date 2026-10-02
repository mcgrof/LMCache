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
  K16/V8). Wrapper stores the single blob to L2; L1 retains
  the original full K+V allocation under the logical key (the rest
  of LMCache's lookup/eviction machinery treats it as one object).

* :attr:`StoragePlacementMode.KV_SPLIT_TIER` — V-only. After
  the wrapper serializes, the lifecycle drives:

  - V child object → L2 under a derived V child key.
  - K child object → retained in L1 under a derived K child key
    (a separate copy of the original staging object's K bytes).
  - Original full K+V staging object → deleted in L1, so L1
    footprint per cached chunk drops from K+V → K-only.

  Lookup under the logical key returns a "composite hit" iff BOTH
  the K child is in L1 AND the V child is in L2. Load reassembles
  the full logical object from the two children.

Placement is a separate concept from layout.  ``KV_COMPONENT_GROUPS``
only declares that K and V are
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
   deletes the paired V child from L2.

True cross-tier atomicity is not available through the current
``L2AdapterInterface`` — adapters' ``delete()`` is optional and may
be a no-op. Split-tier admission therefore restricts this mode to the
filesystem adapter, whose ``delete()`` return is a terminal physical
completion boundary. The "atomic" eviction policy is implemented as
**logical invalidation first, physical cleanup second**: the manifest
entry transitions to ``INVALIDATED`` synchronously, so readers see
consistent state, and its generation fence remains until both children
reach terminal cleanup.
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


class SplitTierConfigError(ValueError):
    """A configuration is outside the KV_SPLIT_TIER (V-only) support
    matrix.

    Subclasses :class:`ValueError` so existing ``except ValueError``
    handlers still catch it, but gives callers a precise type to
    distinguish a *deterministic* split-tier config/placement conflict
    (retrying the identical config cannot help) from a transient
    failure.  The runtime P2P adapter-discovery loop uses this to stop
    re-attempting a peer whose adapter would violate the matrix.
    """


class StoragePlacementMode(Enum):
    """Per-StorageManager placement policy for serde-encoded output."""

    KV_TOGETHER = "kv_together"
    """The serde emits one encoded blob covering both K and V (the
    default for ``fp8`` and ``asym_k16_v8``).  The wrapper
    stores the single blob under the logical key and L1 retains the
    original full K+V allocation as before."""

    KV_SPLIT_TIER = "kv_split_tier"
    """V-only.  The wrapper drives the split-tier state
    machine: K child retained in L1 under a derived key, V child
    stored to L2 under a derived key, original logical staging
    object deleted in L1.  Lookup is a composite check; load
    reassembles.  L1 eviction is paired and logically atomic."""


# Domain-separation marker bytes appended to ``ObjectKey.chunk_hash``
# when deriving K / V child keys. ``SplitTierManifest`` enforces one
# logical-hash width for every live entry and records child ownership.
# The width invariant is required because ``ObjectKey`` itself permits
# arbitrary byte lengths: without it, logical hash ``H + marker`` would
# alias the child derived from logical hash ``H``.
_K_CHILD_MARKER = b"\x01"
_V_CHILD_MARKER = b"\x02"

# Codec-scheme discriminator woven into the child key so a scale-aware
# (COMPUTED) V blob and a byte-through (RAW_UNIT) V blob for the
# SAME logical chunk cannot collide on one L2 key and silently overwrite
# or mis-serve each other.  The legacy format (role byte only) is
# retained verbatim for COMPUTED so every already-stored object and
# every existing caller stays byte-identical; a non-legacy scheme gets a
# magic-tagged suffix instead.
#
# Versioned suffix layout (RAW_UNIT and any future non-legacy scheme):
#     chunk_hash = logical_hash + MAGIC + scheme_byte + role_byte
# The role byte is kept LAST so a reader that predates this format (its
# ``reverse`` strips only the trailing role byte) yields
# ``logical_hash + MAGIC + scheme_byte`` as the "logical" key -- which
# matches no same-width logical key, so it fails closed rather than aliasing a
# genuine object. ``SplitTierManifest`` provides the authoritative membership
# check; this structural suffix is used only after a caller already knows it
# has a component key. ``chunk_hash`` is hex-encoded into FS filenames, so
# arbitrary marker bytes are transport-safe.
_SCHEME_MARKER_MAGIC = b"\xffKVSCHM\xff"


class ComponentKeyScheme(Enum):
    """Which V-plane codec scheme a child key belongs to.

    Mirrors :class:`lmcache.v1.kv_codec.ScaleScheme` at the storage-key
    layer; the two MUST stay in correspondence
    (``COMPUTED_LEGACY`` <-> ``COMPUTED_PER_TENSOR``,
    ``RAW_UNIT`` <-> ``RAW_UNIT``).  Kept as a separate key-layer enum so
    the placement module does not import the codec module.

    ``COMPUTED_LEGACY`` uses the pre-existing role-only key format (no
    magic) -- byte-identical to every key written before this
    discriminator existed.  ``RAW_UNIT`` uses the magic-tagged format and
    therefore can never reuse a legacy key.
    """

    COMPUTED_LEGACY = 0
    RAW_UNIT = 1


# Scheme <-> byte map used only inside the versioned (magic) suffix.
# COMPUTED_LEGACY is byte 0 for correspondence with ScaleScheme, but it
# is never emitted in the magic form (it is the legacy role-only key);
# a magic suffix that claims byte 0 is therefore malformed.
_SCHEME_TO_BYTE = {
    ComponentKeyScheme.COMPUTED_LEGACY: b"\x00",
    ComponentKeyScheme.RAW_UNIT: b"\x01",
}
_BYTE_TO_SCHEME = {v: k for k, v in _SCHEME_TO_BYTE.items()}


def _role_from_marker(marker: bytes) -> Optional[str]:
    """Map a 1-byte role marker to ``"k"`` / ``"v"`` (or ``None``)."""
    if marker == _K_CHILD_MARKER:
        return "k"
    if marker == _V_CHILD_MARKER:
        return "v"
    return None


def derive_component_key(
    logical_key: ObjectKey,
    role: str,
    scheme: ComponentKeyScheme = ComponentKeyScheme.COMPUTED_LEGACY,
) -> ObjectKey:
    """Derive a child :class:`ObjectKey` from a logical key + role.

    Preserves every identity field of the logical key --
    ``model_name``, ``kv_rank``, ``object_group_id``, and
    ``cache_salt`` (per-tenant adapter accounting and quotas depend
    on ``cache_salt``; multi-object-group models depend on
    ``object_group_id`` -- do NOT drop either).  Domain-separates via
    the ``chunk_hash`` field.

    Args:
        logical_key: The original logical chunk key.
        role: ``"k"`` or ``"v"`` -- selects which child key is
            derived.
        scheme: which codec scheme owns this child.
            ``COMPUTED_LEGACY`` (default) reproduces the historical
            role-only key exactly, so existing callers and stored
            objects are unaffected.  ``RAW_UNIT`` emits the
            magic-tagged key so byte-through V blobs never collide with
            scale-aware V blobs on L2.

    Returns:
        A new :class:`ObjectKey` whose ``chunk_hash`` is the logical
        hash plus a role marker (legacy scheme) or plus
        ``MAGIC + scheme_byte + role_marker`` (non-legacy scheme). The
        owning :class:`SplitTierManifest` enforces a fixed logical-hash
        width, which keeps both derived namespaces disjoint from its
        live logical-key namespace.

    Raises:
        ValueError: if ``role`` is not ``"k"`` or ``"v"``.
    """
    if role == "k":
        marker = _K_CHILD_MARKER
    elif role == "v":
        marker = _V_CHILD_MARKER
    else:
        raise ValueError(f"derive_component_key: role must be 'k' or 'v', got {role!r}")

    if scheme == ComponentKeyScheme.COMPUTED_LEGACY:
        suffix = marker
    else:
        # role kept LAST (see _SCHEME_MARKER_MAGIC note): old readers
        # fail closed instead of aliasing a real logical key.
        suffix = _SCHEME_MARKER_MAGIC + _SCHEME_TO_BYTE[scheme] + marker

    return ObjectKey(
        chunk_hash=logical_key.chunk_hash + suffix,
        model_name=logical_key.model_name,
        kv_rank=logical_key.kv_rank,
        object_group_id=logical_key.object_group_id,
        cache_salt=logical_key.cache_salt,
    )


def decode_component_key(
    child_key: ObjectKey,
) -> Optional[tuple[ObjectKey, str, ComponentKeyScheme]]:
    """Full inverse of :func:`derive_component_key`, scheme included.

    This is a structural decoder for keys already known to be children;
    it must not be used to classify arbitrary keys. A logical hash may
    itself end in a role marker. Operational callers classify K children
    through :meth:`SplitTierManifest.logical_for_k_child`, which uses
    explicit membership instead of suffix inspection.
    It recognizes both the versioned (magic-tagged) key and the legacy
    role-only key. A key that matches neither structure returns ``None``.

    Args:
        child_key: A key that may or may not be a derived child.

    Returns:
        ``(logical, role, scheme)`` for a recognizable child;
        ``None`` for a logical key, an unknown marker, or a magic
        suffix that is present but malformed (fail closed).
    """
    h = child_key.chunk_hash
    magic = _SCHEME_MARKER_MAGIC
    versioned_suffix_len = len(magic) + 2  # MAGIC + scheme_byte + role_byte

    # Versioned form: if the magic is present at the suffix offset, this
    # is (or claims to be) a versioned key -- decode it or fail closed;
    # never fall back to the legacy interpretation, which would alias a
    # bogus logical key. Operational classification uses manifest membership,
    # not this suffix parser.
    if len(h) >= versioned_suffix_len:
        window = h[-versioned_suffix_len:]
        if window[: len(magic)] == magic:
            logical_hash = h[:-versioned_suffix_len]
            scheme = _BYTE_TO_SCHEME.get(window[len(magic) : len(magic) + 1])
            role = _role_from_marker(window[-1:])
            # Fail closed on: empty logical hash, malformed scheme/role,
            # or a magic suffix claiming the legacy scheme (legacy never
            # uses the magic form).
            if (
                len(logical_hash) < 1
                or role is None
                or scheme is None
                or scheme == ComponentKeyScheme.COMPUTED_LEGACY
            ):
                return None
            logical = ObjectKey(
                chunk_hash=logical_hash,
                model_name=child_key.model_name,
                kv_rank=child_key.kv_rank,
                object_group_id=child_key.object_group_id,
                cache_salt=child_key.cache_salt,
            )
            return logical, role, scheme

    # Legacy role-only form.
    if len(h) < 2:
        return None
    role = _role_from_marker(h[-1:])
    if role is None:
        return None
    logical = ObjectKey(
        chunk_hash=h[:-1],
        model_name=child_key.model_name,
        kv_rank=child_key.kv_rank,
        object_group_id=child_key.object_group_id,
        cache_salt=child_key.cache_salt,
    )
    return logical, role, ComponentKeyScheme.COMPUTED_LEGACY


def reverse_component_key(
    child_key: ObjectKey,
) -> Optional[tuple[ObjectKey, str]]:
    """Inverse of :func:`derive_component_key` (role only).

    Back-compat 2-tuple view over :func:`decode_component_key` for the
    L1 eviction controller and other callers that only need the
    logical key + role.  Recognizes both the legacy role-only key and
    the versioned magic-tagged key; the scheme is dropped.  Callers
    that need the scheme (to re-derive the matching child key) must use
    :func:`decode_component_key`.

    Args:
        child_key: A key that may or may not be a derived child.

    Returns:
        ``(logical, role)`` if ``child_key`` is a recognizable
        K or V child; otherwise ``None``.
    """
    decoded = decode_component_key(child_key)
    if decoded is None:
        return None
    logical, role, _scheme = decoded
    return logical, role


@dataclass
class _ManifestEntry:
    """One logical key's manifest state plus the generation that owns
    it.

    ``generation`` is a manifest-instance-local, monotonically increasing id
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

    A store generation progresses from ``STORE_IN_FLIGHT`` to
    ``COMPLETE``. Cleanup claims ``DELETE_IN_FLIGHT`` before touching
    physical children; cancellation may restore its previous state if
    no physical deletion occurred. Invalid transitions raise
    :class:`ValueError` rather than silently corrupting the lifecycle.
    A new generation begins when
    :meth:`register_pending` reclaims a key whose previous
    generation ended (``COMPLETE`` gone stale or ``INVALIDATED``
    after failed-store cleanup) -- a store failure must never
    permanently block a key from being stored again.

    **Generation guard.**  :meth:`register_pending` returns a
    manifest-instance-local, monotonically increasing generation id, and
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

    def __init__(
        self,
        component_key_scheme: ComponentKeyScheme = ComponentKeyScheme.COMPUTED_LEGACY,
    ) -> None:
        self._lock = threading.Lock()
        self._entries: dict[ObjectKey, _ManifestEntry] = {}
        # Monotonic, manifest-instance-local generation counter.  Never reset
        # (not even when an entry is dropped) so a generation number
        # is never reused -- a stale cleanup for an old generation can
        # therefore never collide with a fresh entry.
        self._next_generation = 1
        # Every live logical key has the same hash width. TokenHasher emits
        # fixed-width values for a configured algorithm, but ObjectKey accepts
        # arbitrary bytes; enforcing the invariant here prevents logical
        # ``H + marker`` from aliasing the child derived from ``H``.
        self._logical_hash_length: Optional[int] = None
        # Per-engine child-key scheme.  MUST match the scheme the
        # wrapper allocates the K child under, or the K-child side map
        # below records a different key than the one that actually lands
        # in L1 -- then is_k_child_key misses the real K child and the
        # StoreController wrongly routes it to L2.
        self._component_key_scheme = component_key_scheme
        # Side maps of currently tracked component keys, populated when
        # the wrapper registers a logical key as STORE_IN_FLIGHT.  The
        # StoreController consults this set to skip K_child write
        # completions (K_child is L1-only by design; it must never be
        # routed to L2, otherwise the wrapper's finish_write triggers
        # a recursive submit_store_task that fails on the K-only
        # single-shape MemoryObj).
        # Mapping instead of suffix decoding also prevents an ordinary logical
        # hash ending in a role byte from being mistaken for a child.
        self._k_child_owners: dict[ObjectKey, ObjectKey] = {}
        self._v_child_owners: dict[ObjectKey, ObjectKey] = {}

    @property
    def component_key_scheme(self) -> ComponentKeyScheme:
        """The child-key scheme this manifest derives K children under.

        A consumer that also derives child keys (the wrapper) MUST match
        this, or its K key and this manifest's ``is_k_child_key`` side-set
        disagree.
        """
        return self._component_key_scheme

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
                and a later retry succeeds after :meth:`drop`), or if
                its hash width differs from other live manifest entries.
        """
        with self._lock:
            hash_length = len(logical_key.chunk_hash)
            if (
                self._logical_hash_length is not None
                and hash_length != self._logical_hash_length
            ):
                raise ValueError(
                    "SplitTierManifest.register_pending: all live logical "
                    "keys must use the same chunk_hash length; "
                    f"expected {self._logical_hash_length}, got {hash_length}"
                )
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
            self._logical_hash_length = hash_length
            k_child = derive_component_key(
                logical_key, "k", scheme=self._component_key_scheme
            )
            owner = self._k_child_owners.get(k_child)
            if owner is not None and owner != logical_key:
                self._entries.pop(logical_key, None)
                if not self._entries:
                    self._logical_hash_length = None
                raise ValueError(
                    "SplitTierManifest.register_pending: derived K child "
                    "collides with another logical key"
                )
            self._k_child_owners[k_child] = logical_key
            self._v_child_owners[
                derive_component_key(
                    logical_key, "v", scheme=self._component_key_scheme
                )
            ] = logical_key
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

        K child deletion in L1 has completed; the V child delete runs
        against the L2 adapter while this state prevents child-name reuse.
        The manifest entry is removed via :meth:`drop` only after the L2
        delete reaches its terminal completion boundary.

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

    def begin_physical_cleanup(
        self,
        logical_key: ObjectKey,
        generation: int,
        expected_states: tuple["SplitTierState", ...],
    ) -> Optional["SplitTierState"]:
        """Atomically fence a generation before deleting physical children.

        Child keys do not encode a generation. A cleanup must therefore enter
        ``DELETE_IN_FLIGHT`` before a blocking or delayed delete so a replacement
        cannot publish under the same physical key.

        Args:
            logical_key: Logical composite whose children will be deleted.
            generation: Generation captured or owned by the cleanup caller.
            expected_states: States from which this caller may claim cleanup.

        Returns:
            The prior state when the fence was acquired, otherwise ``None``.
        """
        with self._lock:
            cur = self._entries.get(logical_key)
            if (
                cur is None
                or cur.generation != generation
                or cur.state not in expected_states
            ):
                return None
            previous = cur.state
            cur.state = SplitTierState.DELETE_IN_FLIGHT
            return previous

    def cancel_physical_cleanup(
        self,
        logical_key: ObjectKey,
        generation: int,
        restore_state: "SplitTierState",
    ) -> bool:
        """Release a cleanup fence when physical deletion did not occur.

        Args:
            logical_key: Logical composite whose delete was cancelled.
            generation: Generation that acquired the fence.
            restore_state: State returned by :meth:`begin_physical_cleanup`.

        Returns:
            ``True`` if the matching fence was released.

        Raises:
            ValueError: If ``restore_state`` is ``DELETE_IN_FLIGHT``.
        """
        if restore_state == SplitTierState.DELETE_IN_FLIGHT:
            raise ValueError("cannot restore DELETE_IN_FLIGHT as a prior state")
        with self._lock:
            cur = self._entries.get(logical_key)
            if (
                cur is None
                or cur.generation != generation
                or cur.state != SplitTierState.DELETE_IN_FLIGHT
            ):
                return False
            cur.state = restore_state
            return True

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
            self._k_child_owners.pop(
                derive_component_key(
                    logical_key, "k", scheme=self._component_key_scheme
                ),
                None,
            )
            self._v_child_owners.pop(
                derive_component_key(
                    logical_key, "v", scheme=self._component_key_scheme
                ),
                None,
            )
            if not self._entries:
                self._logical_hash_length = None

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
            return key in self._k_child_owners

    def logical_for_k_child(self, key: ObjectKey) -> Optional[ObjectKey]:
        """Return the tracked logical owner of a K child, if any.

        This membership lookup is the authoritative K-child classifier.
        Suffix decoding is insufficient because a valid logical hash may end
        in the same byte used as a child-role marker.

        Args:
            key: Candidate K-child key.

        Returns:
            The owning logical key while the child is tracked, otherwise
            ``None``.
        """
        with self._lock:
            return self._k_child_owners.get(key)

    def logical_for_v_child(self, key: ObjectKey) -> Optional[ObjectKey]:
        """Return the tracked logical owner of a V child, if any.

        Args:
            key: Candidate V-child key.

        Returns:
            The owning logical key while the child is tracked, otherwise
            ``None``.
        """
        with self._lock:
            return self._v_child_owners.get(key)

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

    def state_counts(self) -> dict["SplitTierState", int]:
        """Snapshot the number of tracked entries in each state.

        Every :class:`SplitTierState` is present in the result (zero when
        no entry is in that state), so the observability gauge emits a
        stable set of series.  A rising ``STORE_IN_FLIGHT`` /
        ``DELETE_IN_FLIGHT`` count or a monotonically growing total is the
        operator's signal for a stuck transition or an entry leak.

        Returns:
            A mapping from each :class:`SplitTierState` to the count of
            tracked logical keys currently in that state.
        """
        with self._lock:
            counts: dict[SplitTierState, int] = {state: 0 for state in SplitTierState}
            for entry in self._entries.values():
                counts[entry.state] += 1
            return counts


class SplitTierState(Enum):
    """Manifest-entry state machine for a logical key under
    :attr:`StoragePlacementMode.KV_SPLIT_TIER`.

    Lookup returns "composite hit" only in :attr:`COMPLETE`. A cancelled
    cleanup restores the prior state while the same generation still owns
    the intact children.
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
    proceeds in the background."""

    DELETE_IN_FLIGHT = "delete_in_flight"
    """Physical child deletion is active.  Manifest entry removal waits
    for the terminal L2 delete boundary so a replacement cannot reuse the
    generation-less child names while an old unlink remains pending."""


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
        # (e.g. asym together-mode (0, 1) identity) the serde packs both
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
            f"a kv_together serde (e.g. fp8, asym_k16_v8) cannot "
            f"coexist with a kv_split_tier serde (e.g. asym_k16_v8_v_only) "
            f"on the same StorageManager."
        )
    return modes.pop()


# Serde factory type names whose V blobs are byte-through RAW_UNIT fp8
# codes (implicit unit scale, no stored scales).  MUST stay in sync with
# the RAW_UNIT factory registrations in
# ``lmcache/v1/distributed/serde/asym_k16_v8.py``.  Any serde not listed
# here is treated as scale-aware (COMPUTED_LEGACY), which keeps every
# pre-existing config on the legacy (role-only) child key.
_RAW_UNIT_SERDE_TYPES = frozenset(
    {
        # V-only byte-through -> KV_SPLIT_TIER (K in L1, single-process).
        "asym_bytethrough_k16_v8_v_only",
        # Both-plane byte-through -> KV_TOGETHER (K+V in one durable L2
        # object, cross-process/restart reusable).  RAW_UNIT here drives
        # the fp8 V-component dtype + the heterogeneous pre-split layout
        # pass-through; the KV_TOGETHER placement (from its (0,1) input
        # slot mapping) keeps the split-tier manifest inert.
        "asym_bytethrough_k16_v8",
    }
)


def derive_component_key_scheme(
    adapter_configs: list["L2AdapterConfigBase"],
) -> ComponentKeyScheme:
    """Derive the per-engine :class:`ComponentKeyScheme` from L2 adapters.

    The scheme domain-separates byte-through (``RAW_UNIT``) V blobs from
    scale-aware (``COMPUTED_LEGACY``) ones at the storage-key layer so the
    two cannot collide on one L2 key (see :func:`derive_component_key`).
    It keys off each adapter's ``serde_config.type``: a byte-through serde
    (see :data:`_RAW_UNIT_SERDE_TYPES`) yields
    :attr:`ComponentKeyScheme.RAW_UNIT`; everything else -- no serde, a
    single-tensor serde, or a scale-aware multi-output serde -- yields
    :attr:`ComponentKeyScheme.COMPUTED_LEGACY`.

    One StorageManager is a single-scheme engine: the child-key domain,
    the lifecycle, and the manifest K-child side-set are all keyed on one
    scheme.  A mixed adapter set is rejected, mirroring
    :func:`derive_storage_placement_mode`.

    Args:
        adapter_configs: Sequence of L2 adapter configurations.  Empty
            list returns :attr:`ComponentKeyScheme.COMPUTED_LEGACY`.

    Returns:
        The single canonical :class:`ComponentKeyScheme`.

    Raises:
        ValueError: if adapters demand different schemes.
    """
    schemes: set[ComponentKeyScheme] = set()
    for ac in adapter_configs:
        sc = getattr(ac, "serde_config", None)
        serde_type = getattr(sc, "type", None) if sc is not None else None
        if serde_type in _RAW_UNIT_SERDE_TYPES:
            schemes.add(ComponentKeyScheme.RAW_UNIT)
        else:
            schemes.add(ComponentKeyScheme.COMPUTED_LEGACY)

    if not schemes:
        return ComponentKeyScheme.COMPUTED_LEGACY
    if len(schemes) > 1:
        names = sorted(s.name for s in schemes)
        raise ValueError(
            f"Incompatible L2 adapter component-key schemes: {names}. "
            f"One StorageManager is a single-scheme engine; a byte-through "
            f"(RAW_UNIT) serde cannot coexist with a scale-aware "
            f"(COMPUTED_LEGACY) serde on the same StorageManager."
        )
    return schemes.pop()
