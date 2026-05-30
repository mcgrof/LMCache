# SPDX-License-Identifier: Apache-2.0
"""
Multi-output serde extensions.

The :class:`Serializer` and :class:`Deserializer` classes in
``serde/base.py`` operate on one typed tensor at each endpoint:
one tensor in, one byte buffer out on serialize, and the reverse
on deserialize. That shape works when K and V share a single data
type -- the serde just sees them as one combined tensor. It does
not work in two cases:

* **K and V at different data types.** A typed tensor has one
  dtype, so a serde that wants K at one dtype (e.g. fp16/bf16) and
  V at another (e.g. FP8) cannot carry both -- one of them would
  have to be reinterpreted bytewise, losing the typed-tensor
  contract every other consumer relies on.

* **One side absent.** For tier-split placements where K or V is
  held outside this serde's data path -- e.g. K kept in L1 (CPU
  pinned host memory) while V flows to L2 (durable storage) -- the
  serialize input has no tensor for the absent slot and the
  deserialize output has no destination for it.

This module defines the additive extension. It does NOT modify the
single-tensor interfaces or any of their existing callers; the
async processor, factory registry, L2 adapter wrapper, and built-in
``fp8`` serde all keep their current behavior. A serde implementation
that needs multiple tensors at an endpoint mixes in the multi-output
ABC alongside the existing ``Serializer`` / ``Deserializer``, and the
async wiring around it is added in a follow-up change once a concrete
multi-output serde lands.

The contracts kept deliberately narrow here:

* A :class:`MemoryObjGroup` is a fixed-length tuple of
  ``Optional[MemoryObj]``. ``None`` at a position means "this slot is
  absent": on the serialize input side the corresponding tensor is
  not provided (e.g., V-only writes), and on the deserialize output
  side the caller does not want that tensor materialized (e.g.,
  V-only retrieval). Implementations decide per slot whether absence
  is permitted.
* :class:`MultiSerializer` and :class:`MultiDeserializer` declare
  ``group_size`` so the async layer can pre-validate group lengths
  and so callers can introspect the contract without instantiating
  one tensor of the tuple.
* A single-element ``MemoryObjGroup`` is the trivial bridge to the
  existing single-tensor API: :func:`single_to_multi_serializer` and
  :func:`single_to_multi_deserializer` adapt an existing
  ``Serializer`` / ``Deserializer`` so callers that work in groups can
  invoke them uniformly. The adapter is layout-equivalent: a single
  non-None group element delegates to the underlying ``serialize`` /
  ``deserialize`` and the existing per-byte semantics are unchanged.
"""

# Future
from __future__ import annotations

# Standard
from typing import Optional, Sequence, Tuple
import abc

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.base import Deserializer, Serializer
from lmcache.v1.memory_management import MemoryObj

# A fixed-length tuple of optional MemoryObjs. ``None`` denotes an
# absent slot whose semantics are defined per-implementation: on the
# serialize input side it means the caller is not supplying that
# tensor (V-only writes); on the deserialize output side it means
# the caller does not want that tensor materialized (V-only reads).
MemoryObjGroup = Tuple[Optional[MemoryObj], ...]

# Parallel layout-description tuple used by size estimators when one
# or more positions are absent.
LayoutDescGroup = Tuple[Optional[MemoryLayoutDesc], ...]


class MultiSerializer(abc.ABC):
    """Sync serializer with a fixed-length tuple as input.

    Implementations MUST document, at minimum:

    * The fixed value of :attr:`group_size`.
    * The semantic carried by each slot (e.g., ``slot 0 = K``,
      ``slot 1 = V``).
    * Which slots are required and which may be ``None``. A slot that
      may be ``None`` MUST be tolerated by ``estimate_serialized_size``
      with a ``None`` layout descriptor at the same index.

    The single-tensor :class:`Serializer` ABC remains the canonical
    interface for one-in-one-out serdes; this class is purely
    additive.
    """

    @property
    @abc.abstractmethod
    def group_size(self) -> int:
        """Fixed input-tuple length expected by this serializer.

        The async layer validates submitted groups against this value
        before dispatch; raising the abstract property keeps the
        invariant visible without instantiating a tuple of tensors.
        """
        raise NotImplementedError

    def input_slot_mapping(self) -> Tuple[Optional[int], ...]:
        """Mapping from this serializer's slots to the parent
        grouped-:class:`MemoryObj`'s group indexes.

        The :class:`~lmcache.v1.distributed.l2_adapters.serde_wrapper.SerdeL2AdapterWrapper`
        receives a list of single grouped ``MemoryObj`` per key (the
        canonical LMCache shape: one ``MemoryObj`` carries all groups
        for a logical KV chunk).  Multi-output serdes need a tuple of
        per-slot views over those groups.  This method declares which
        parent group, if any, feeds each of the serializer's slots:

        * ``input_slot_mapping()[i] = j`` (an int) — slot ``i`` reads
          parent group ``j``.
        * ``input_slot_mapping()[i] = None`` — slot ``i`` is absent
          (the serializer must tolerate ``None`` at that position;
          typically the data lives outside this serde's path, e.g.
          K kept in L1 for split-tier V-only).

        Default is the identity mapping ``(0, 1, ..., group_size - 1)``
        — the common case where slot ``i`` is exactly parent group
        ``i``.  Subclasses that skip slots (e.g. V-only) override.

        Returns:
            Tuple of length :attr:`group_size`.
        """
        return tuple(range(self.group_size))

    @abc.abstractmethod
    def serialize(self, src: MemoryObjGroup, dst: MemoryObj) -> int:
        """Serialize ``src`` group into ``dst`` byte buffer.

        Args:
            src: Source group with length :attr:`group_size`. Slots
                may be ``None`` to indicate absence; the implementation
                decides whether absence is admissible per slot and
                MUST raise ``ValueError`` when a required slot is
                missing.
            dst: Destination MemoryObj byte buffer (write-locked).
                Capacity MUST be at least the value previously
                returned by :meth:`estimate_serialized_size` for the
                same layout group.

        Returns:
            The number of bytes written to ``dst``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_serialized_size(
        self,
        layout_descs: LayoutDescGroup,
    ) -> int:
        """Upper bound on bytes produced by :meth:`serialize`.

        Args:
            layout_descs: Group of optional layout descriptors with
                length :attr:`group_size`. ``None`` slots MUST mirror
                the absence pattern that :meth:`serialize` will see.

        Returns:
            Number of bytes to allocate for the serialized buffer,
            including any safety margin the implementation requires.
        """
        raise NotImplementedError


class MultiDeserializer(abc.ABC):
    """Sync deserializer with a fixed-length tuple as output.

    Implementations MUST document the same per-slot semantics as
    :class:`MultiSerializer`. ``None`` at a destination slot means the
    caller does not want that output materialized; the implementation
    MUST NOT touch the missing slot and MUST NOT fail solely because
    a slot is ``None`` if the schema permits it.
    """

    @property
    @abc.abstractmethod
    def group_size(self) -> int:
        """Fixed output-tuple length produced by this deserializer."""
        raise NotImplementedError

    def output_slot_mapping(self) -> Tuple[Optional[int], ...]:
        """Mapping from this deserializer's slots to the destination
        grouped-:class:`MemoryObj`'s group indexes.

        Mirrors :meth:`MultiSerializer.input_slot_mapping` for the
        load side: the wrapper materializes a destination grouped
        ``MemoryObj`` per key, and this method declares which parent
        group each slot writes back to.

        Default is the identity mapping.  Subclasses that leave a slot
        absent (e.g. V-only deserialize that returns only V, leaving
        K alone in L1) override.

        Returns:
            Tuple of length :attr:`group_size`.
        """
        return tuple(range(self.group_size))

    @abc.abstractmethod
    def deserialize(self, src: MemoryObj, dst: MemoryObjGroup) -> None:
        """Deserialize ``src`` byte buffer into ``dst`` group.

        Args:
            src: Source byte-buffer MemoryObj.
            dst: Destination group with length :attr:`group_size`.
                ``None`` slots MUST be left untouched.
        """
        raise NotImplementedError


# ----- Adapters bridging single-tensor and tuple interfaces -----


class _SingleAsMultiSerializer(MultiSerializer):
    """Adapt an existing single-tensor :class:`Serializer` to the
    multi-output interface as a length-1 group.

    Used by callers that want to invoke single- and multi-tensor
    serdes through the same group-shaped call site without forcing
    every existing serde to subclass the multi ABC.
    """

    def __init__(self, inner: Serializer) -> None:
        self._inner = inner

    @property
    def group_size(self) -> int:
        return 1

    def serialize(self, src: MemoryObjGroup, dst: MemoryObj) -> int:
        if len(src) != 1:
            raise ValueError(
                f"_SingleAsMultiSerializer expected group of size 1, got {len(src)}"
            )
        single = src[0]
        if single is None:
            raise ValueError(
                "_SingleAsMultiSerializer: single-tensor serializer "
                "does not admit a None src slot"
            )
        return self._inner.serialize(single, dst)

    def estimate_serialized_size(
        self,
        layout_descs: LayoutDescGroup,
    ) -> int:
        if len(layout_descs) != 1:
            raise ValueError(
                f"_SingleAsMultiSerializer expected layout group of "
                f"size 1, got {len(layout_descs)}"
            )
        single = layout_descs[0]
        if single is None:
            raise ValueError(
                "_SingleAsMultiSerializer: single-tensor serializer "
                "requires a non-None layout descriptor"
            )
        return self._inner.estimate_serialized_size(single)


class _SingleAsMultiDeserializer(MultiDeserializer):
    """Adapt an existing single-tensor :class:`Deserializer` to the
    multi-output interface as a length-1 group.
    """

    def __init__(self, inner: Deserializer) -> None:
        self._inner = inner

    @property
    def group_size(self) -> int:
        return 1

    def deserialize(self, src: MemoryObj, dst: MemoryObjGroup) -> None:
        if len(dst) != 1:
            raise ValueError(
                f"_SingleAsMultiDeserializer expected group of size 1, got {len(dst)}"
            )
        single = dst[0]
        if single is None:
            # A length-1 group with a None slot is a no-op rather than
            # an error: it lets callers pass-through groups uniformly
            # even when they have already decided not to materialize
            # the only output.
            return
        self._inner.deserialize(src, single)


def single_to_multi_serializer(inner: Serializer) -> MultiSerializer:
    """Wrap a single-tensor :class:`Serializer` as a length-1
    :class:`MultiSerializer`.

    The wrapper is layout-equivalent: identical bytes are written to
    ``dst`` for the same input as a direct call to ``inner.serialize``.
    """
    return _SingleAsMultiSerializer(inner)


def single_to_multi_deserializer(inner: Deserializer) -> MultiDeserializer:
    """Wrap a single-tensor :class:`Deserializer` as a length-1
    :class:`MultiDeserializer`.

    The wrapper is layout-equivalent: identical bytes are written to
    the dst tensor for the same input as a direct call to
    ``inner.deserialize``. A length-1 group whose only slot is
    ``None`` is treated as a deliberate skip rather than an error.
    """
    return _SingleAsMultiDeserializer(inner)


def layout_desc_to_group(
    layout_desc: MemoryLayoutDesc,
    slot_mapping: Tuple[Optional[int], ...],
) -> LayoutDescGroup:
    """Convert a multi-group :class:`MemoryLayoutDesc` into a
    :data:`LayoutDescGroup` using a slot mapping.

    The parent ``layout_desc`` carries ``shapes`` / ``dtypes`` lists with
    one entry per group (the canonical LMCache shape: one
    ``MemoryLayoutDesc`` per logical KV chunk, with groups for K and V).
    Multi-output serializers want a tuple of single-group descriptors
    matching their slots, with ``None`` for absent slots.

    Args:
        layout_desc: Parent layout with one entry per group in its
            ``shapes`` and ``dtypes`` lists.
        slot_mapping: Per-slot index into the parent's groups, or
            ``None`` for absent slots.  Length defines the returned
            group's length.

    Returns:
        Tuple of :class:`MemoryLayoutDesc`-or-``None`` matching
        ``slot_mapping``.

    Raises:
        ValueError: if a slot index is out of range for the parent
            layout's groups.
    """
    n_groups = len(layout_desc.shapes)
    result: list[Optional[MemoryLayoutDesc]] = []
    for slot_idx in slot_mapping:
        if slot_idx is None:
            result.append(None)
            continue
        if slot_idx < 0 or slot_idx >= n_groups:
            raise ValueError(
                f"slot mapping references parent group {slot_idx} but "
                f"parent layout has only {n_groups} groups"
            )
        result.append(
            MemoryLayoutDesc(
                shapes=[layout_desc.shapes[slot_idx]],
                dtypes=[layout_desc.dtypes[slot_idx]],
            )
        )
    return tuple(result)


class GroupSlotView:
    """Minimal MemoryObj-shaped view over a single group slot of a
    parent grouped :class:`MemoryObj`.

    Used by :class:`SerdeL2AdapterWrapper` to construct
    :data:`MemoryObjGroup` tuples for multi-output serdes without
    allocating new :class:`MemoryObj` instances.  Exposes only the
    ``.tensor`` property -- the multi-output serializers and
    deserializers in this module only read or write that attribute.
    Writes via ``tensor.copy_(...)`` share storage with the parent's
    ``raw_data`` (the typed tensor view returned by
    ``parent.get_tensor(i)`` aliases the parent's buffer).

    Lifecycle: the parent ``MemoryObj`` owns the buffer; this view
    holds only a back-reference and an index.  No ref count, no
    pin, no allocator parent -- the wrapper retains the parent for
    the duration of the serde task.

    Not a registered :class:`MemoryObj` subclass -- duck typing is
    sufficient because the only attribute the multi-output serdes
    access on group slots is ``.tensor``.
    """

    __slots__ = ("_parent", "_index")

    def __init__(self, parent: MemoryObj, group_index: int) -> None:
        self._parent = parent
        self._index = group_index

    @property
    def tensor(self):
        return self._parent.get_tensor(self._index)


def validate_group_size(
    group: Sequence[Optional[MemoryObj]],
    expected: int,
    *,
    role: str,
) -> None:
    """Validation helper used by callers and implementations alike.

    Raises ``ValueError`` if ``group`` is not exactly ``expected``
    long. ``role`` appears in the error message ("src" / "dst") so
    test failures and runtime errors point at the offending side.
    """
    if len(group) != expected:
        raise ValueError(
            f"MultiSerde {role} group length {len(group)} does not "
            f"match expected {expected}"
        )
