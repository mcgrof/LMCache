# SPDX-License-Identifier: Apache-2.0
"""Pool CPU K children and temporary V tensors for split-tier storage."""

# Future
from __future__ import annotations

# Standard
from collections import deque
import math
import threading

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)


class _VScratchSlab:
    """Linux-kernel-slab-style pool of CPU V-scratch tensors.

    Used by split-tier store to hold a private copy of V bytes
    extracted from the producer-side logical L1 entry.  The copy
    lets the wrapper release the logical's read lock the moment
    ``submit_store_task`` returns; the V codec then encodes from
    the private scratch tensor instead of reading V directly out
    of L1.

    Reuse tensors through a deque to avoid routing private scratch
    allocations through the L1 catalog lock.

    The pool starts empty and lazily allocates tensors on demand
    up to ``max_slots``.  Once a tensor is freed it goes back to
    the deque for reuse.  ``max_slots`` is sized at construction
    time; over-subscription is logged but not fatal (we just
    allocate a one-shot tensor outside the pool).
    """

    def __init__(self, shape: torch.Size, dtype: torch.dtype, max_slots: int) -> None:
        """Construct a slab.

        Args:
            shape: Shape of each pooled tensor (the V tensor shape from
                the producer-side logical L1 layout).
            dtype: Dtype of each pooled tensor (typically bf16 / fp16,
                matching the producer-side V).
            max_slots: Upper bound on the number of tensors the pool
                will lazily allocate.  Beyond this, ``acquire`` returns
                a one-shot tensor (allocated, never returned to the
                pool); a debug log notes the over-subscription.
        """
        self._shape = shape
        self._dtype = dtype
        self._max_slots = max_slots
        self._element_size = torch.tensor([], dtype=dtype).element_size()
        self._phy_size = math.prod(shape) * self._element_size
        self._free: deque[torch.Tensor] = deque()
        self._allocated_count = 0
        self._over_subscriptions = 0
        # In-flight count: tensors handed out via ``acquire`` and not
        # yet ``release``-d, pool-backed and one-shot alike.  See
        # ``get_used_capacity_bytes`` for why both are counted.
        self._in_flight = 0
        # Object identities make release idempotent and prevent a foreign
        # same-shape tensor from corrupting accounting or entering the pool.
        self._active_ids: set[int] = set()
        # One lock covers deque membership and every related counter. Atomic
        # deque methods alone are insufficient because pop/append and the
        # in-flight/cap checks form one invariant.
        self._lock = threading.Lock()

    def acquire(self) -> torch.Tensor:
        """Pop a tensor from the free deque, or allocate a fresh one.

        Returns:
            A torch.Tensor on CPU with the configured shape + dtype.
            The buffer is uninitialized; callers should overwrite
            with ``copy_`` before reading.
        """
        with self._lock:
            if self._free:
                t = self._free.popleft()
                self._active_ids.add(id(t))
                self._in_flight += 1
                return t
            if self._allocated_count < self._max_slots:
                t = torch.empty(self._shape, dtype=self._dtype, device="cpu")
                self._allocated_count += 1
                self._active_ids.add(id(t))
                self._in_flight += 1
                return t
            self._over_subscriptions += 1

        # Over-subscription: allocate a one-shot tensor that won't be
        # returned if the pool is full. Allocate outside the lock so a burst
        # cannot stall releases behind a slow host allocation.
        t = torch.empty(self._shape, dtype=self._dtype, device="cpu")
        with self._lock:
            self._active_ids.add(id(t))
            self._in_flight += 1
            return t

    def release(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool for reuse.

        Args:
            tensor: A tensor previously acquired via :meth:`acquire`.
                Shape / dtype mismatches are silently dropped (treated
                as one-shot tensors -- they GC normally).
        """
        with self._lock:
            tensor_id = id(tensor)
            if (
                tensor.shape != self._shape
                or tensor.dtype != self._dtype
                or tensor_id not in self._active_ids
            ):
                return
            self._active_ids.remove(tensor_id)
            self._in_flight -= 1
            # Cap the queue at ``max_slots``; anything beyond gets dropped
            # so a transient burst doesn't grow the pool permanently.
            if len(self._free) < self._max_slots:
                self._free.append(tensor)

    def stats(self) -> dict[str, int]:
        """Snapshot of pool counters for diagnostic logging.

        Returns:
            ``{"allocated": N, "free": N, "over_subscriptions": N,
                "in_flight": N}``.
        """
        with self._lock:
            return {
                "allocated": self._allocated_count,
                "free": len(self._free),
                "over_subscriptions": self._over_subscriptions,
                "in_flight": self._in_flight,
            }

    # ----- L1MemoryUsageProvider protocol -----

    def get_used_capacity_bytes(self) -> tuple[int, int]:
        """Live tensor bytes + configured pool cap.

        ``used`` counts *every* outstanding acquire — pool-backed and
        over-subscription one-shots — so the LRU policy sees true
        V-scratch memory pressure, not just the pooled subset.

        ``capacity`` is the configured upper bound for the pool's
        in-pool tensors.  Under heavy over-subscription, ``used`` may
        exceed ``capacity`` and the policy's used/capacity ratio
        crosses 1.0 — that correctly signals "evict harder".
        """
        with self._lock:
            used = self._in_flight * self._phy_size
            capacity = self._max_slots * self._phy_size
            return used, capacity


class _KChildSlab(MemoryAllocatorInterface):
    """K-child memory provider: kmem_cache-style pool of pre-allocated
    K-shaped backing buffers.

    K children are L1 residents for V-only storage. Allocate their CPU
    backing lazily and recycle it through a deque, avoiding a scan of the
    general allocator's free ranges for each store. The wrapper registers
    these objects through ``L1Manager.reserve_external_writes`` so normal
    L1 locking, reads and paired eviction still govern their lifetime.

    Shape and dtype are fixed at construction; the wrapper must select a
    matching pool before allocating. At most ``max_slots`` freed objects
    are retained for reuse. Concurrent allocations beyond the configured
    slot count receive additional tensors, which are discarded when the
    free deque is full. One lock protects deque membership, live-object
    identity and the counters used for memory-pressure accounting.
    """

    def __init__(self, shape: torch.Size, dtype: torch.dtype, max_slots: int) -> None:
        """Construct a K-child slab.

        Args:
            shape: K tensor shape (single-group; the slab does not
                serve multi-group layouts).
            dtype: K tensor dtype.
            max_slots: Pool upper bound.  Beyond this, ``allocate``
                returns one-shot objects that are GC'd, not pooled.
        """
        self._shape = shape
        self._dtype = dtype
        self._max_slots = max_slots
        self._element_size = torch.tensor([], dtype=dtype).element_size()
        self._phy_size = math.prod(shape) * self._element_size
        self._free: deque[TensorMemoryObj] = deque()
        self._allocated_count = 0
        self._over_subscriptions = 0
        # In-flight count covers every outstanding MemoryObj — pooled
        # and one-shot.  Without it, ``get_used_capacity_bytes``
        # under-reports during over-subscription bursts and the LRU
        # policy sees the complete live K-child footprint.
        self._in_flight = 0
        self._active_ids: set[int] = set()
        self._lock = threading.Lock()

    @property
    def shape(self) -> torch.Size:
        """Per-slot tensor shape, pinned at construction."""
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        """Per-slot dtype, pinned at construction."""
        return self._dtype

    def _build_memory_obj(self) -> TensorMemoryObj:
        """Allocate a fresh CPU tensor + wrap as a TensorMemoryObj."""
        tensor = torch.empty(self._shape, dtype=self._dtype, device="cpu")
        raw_data = tensor.view(torch.uint8).flatten()
        metadata = MemoryObjMetadata(
            shape=self._shape,
            dtype=self._dtype,
            address=tensor.data_ptr(),
            phy_size=self._phy_size,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.KV_2LTD,
            shapes=[self._shape],
            dtypes=[self._dtype],
        )
        return TensorMemoryObj(
            raw_data=raw_data,
            metadata=metadata,
            parent_allocator=self,
        )

    def _rebind_for_reuse(self, obj: TensorMemoryObj) -> TensorMemoryObj:
        """Reset per-allocation metadata on a recycled MemoryObj."""
        obj.valid = True
        obj.meta.ref_count = 1
        obj.meta.pin_count = 0
        obj.reset_used_size()
        # group_prefix_sum + shape are pinned at construction (single-
        # shape slab); nothing else to reset.
        return obj

    # ----- MemoryAllocatorInterface API -----

    def allocate(
        self,
        shapes: "torch.Size | list[torch.Size] | None",
        dtypes: "torch.dtype | list[torch.dtype] | None",
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: "str | None" = None,
    ) -> TensorMemoryObj | None:
        """Return a single MemoryObj for one K-child slot.

        The slab is shape-pinned at construction; ``shapes`` /
        ``dtypes`` are accepted (and may be ``None``) only to satisfy
        the :class:`MemoryAllocatorInterface` call shape -- callers
        verify the pinned shape via :attr:`shape` / :attr:`dtype`
        before allocating.
        """
        return self._allocate_one()

    def batched_allocate(  # type: ignore[override]
        self,
        shapes: "torch.Size | list[torch.Size] | None",
        dtypes: "torch.dtype | list[torch.dtype] | None",
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: "str | None" = None,
    ) -> list[TensorMemoryObj]:
        """Return a list of MemoryObjs for ``batch_size`` K-child slots.

        Always returns ``batch_size`` items: pool hits are O(1),
        over-subscription allocates fresh one-shot objects.  Callers
        get the same length list they asked for; never ``None``.
        """
        return [self._allocate_one() for _ in range(batch_size)]

    def _allocate_one(self) -> TensorMemoryObj:
        with self._lock:
            if self._free:
                obj = self._rebind_for_reuse(self._free.popleft())
                self._active_ids.add(id(obj))
                self._in_flight += 1
                return obj
            if self._allocated_count < self._max_slots:
                obj = self._build_memory_obj()
                self._allocated_count += 1
                self._active_ids.add(id(obj))
                self._in_flight += 1
                return obj
            self._over_subscriptions += 1

        # Over-subscription: fresh one-shot, not pooled when the queue is
        # full. Allocate outside the lock so releases stay responsive.
        obj = self._build_memory_obj()
        with self._lock:
            self._active_ids.add(id(obj))
            self._in_flight += 1
        return obj

    def free(self, memory_obj: MemoryObj, allocator_type: "str | None" = None) -> None:
        """Return a slab-borrowed MemoryObj to the deque.

        No-op if the deque is already at ``max_slots`` (one-shot
        objects GC naturally).  Marks the object invalid before
        pooling so :meth:`TensorMemoryObj.__del__`'s safety-net
        ``parent_allocator.free`` call doesn't re-enter and re-pool.
        """
        with self._lock:
            obj_id = id(memory_obj)
            if (
                not isinstance(memory_obj, TensorMemoryObj)
                or memory_obj.meta.shapes != [self._shape]
                or memory_obj.meta.dtypes != [self._dtype]
                or obj_id not in self._active_ids
            ):
                return
            self._active_ids.remove(obj_id)
            memory_obj.valid = False
            self._in_flight -= 1
            if len(self._free) < self._max_slots:
                self._free.append(memory_obj)

    def batched_free(
        self,
        memory_objs: list[MemoryObj],
        allocator_type=None,
        update_stats: bool = True,
    ) -> None:
        for obj in memory_objs:
            self.free(obj)

    def close(self) -> None:
        """Drop all pooled MemoryObjs; the slab cannot be reused after."""
        with self._lock:
            self._free.clear()
            self._allocated_count = 0

    def memcheck(self) -> bool:
        return True

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "allocated": self._allocated_count,
                "free": len(self._free),
                "over_subscriptions": self._over_subscriptions,
                "in_flight": self._in_flight,
            }

    # ----- L1MemoryUsageProvider protocol -----

    def get_used_capacity_bytes(self) -> tuple[int, int]:
        """Bytes held by live K-child MemoryObjs + the configured cap.

        ``used`` counts every outstanding ``allocate`` -- pool-backed
        and over-subscription one-shots -- so the LRU policy sees true
        K-child memory pressure, including allocations beyond the configured
        pool size.

        ``capacity`` is the configured pool upper bound.  Under
        over-subscription, ``used`` may exceed ``capacity`` and the
        policy's used/capacity ratio crosses 1.0 -- this correctly
        signals "evict harder" rather than papering over the
        pressure.
        """
        with self._lock:
            used = self._in_flight * self._phy_size
            capacity = self._max_slots * self._phy_size
            return used, capacity


class _VScratchSlot:
    """MemoryObj-like adapter exposing a slab-borrowed tensor.

    Lightweight: just exposes ``.tensor`` so the V-only codec can
    read V bytes off it without touching the underlying logical L1
    entry.  No allocator hooks, no ref-counting -- the slab owns
    the lifecycle.
    """

    __slots__ = ("_tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor
