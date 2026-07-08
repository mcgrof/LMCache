# SPDX-License-Identifier: Apache-2.0
"""Derive the canonical L1 MemoryObj shape from the configured L2 adapters.

LMCache stores KV chunks as a single packed tensor with leading dim 2 =
(K, V).  Single-tensor serdes such as ``fp8`` cast the whole packed tensor
uniformly and never look at K and V separately.  A multi-output serde such
as ``asym_k16_v8`` needs K and V as distinct typed sub-objects so it can
quantize V to FP8 while leaving K bit-exact.  This module is the single
place that decides, from each adapter's ``serde_config``, which canonical
layout the storage path builds.

The layout is opt-in per serde: ``fp8`` and any other single-tensor serde
keep the packed layout, and only a serde whose ``SerdeProcessor`` reports a
non-``None`` ``input_slot_mapping`` selects the KV-component-groups layout.
Derive the layout here rather than in the vLLM or SGLang connectors so both
inherit the policy without duplicating serde-shape branching, and reject
adapters that demand conflicting layouts at config time so one canonical L1
layout serves every downstream adapter.

K and V share the model's native dtype on the storage side: the asym serde
quantizes V to FP8 internally during serialize, so the packed buffer's K
bytes followed by V bytes are byte-identical to a two-group layout's, and
only the typed view through ``TensorMemoryObj.get_tensor(i)`` changes -- no
transfer-kernel change is needed.  Heterogeneous K/V dtypes (native K BF16
with V FP8) are out of scope and need transfer-kernel work first.
"""

# Future
from __future__ import annotations

# Standard
from enum import Enum
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.l2_adapters.config import L2AdapterConfigBase


class StorageLayoutMode(Enum):
    """Canonical L1 ``MemoryObj`` shape selected by the configured serdes."""

    PACKED = "packed"
    """Single-group ``MemoryObj`` with the K|V tensor packed along
    leading dim 2 (LMCache's historical layout).  Used when every
    configured serde is single-tensor (``Serializer``) or no serde
    is configured.  ``fp8`` operates on this layout."""

    KV_COMPONENT_GROUPS = "kv_component_groups"
    """Two-group ``MemoryObj`` with K and V as separate typed
    sub-objects (group 0 = K, group 1 = V).  Used when any
    configured serde is multi-output (``MultiSerializer``).
    ``TensorMemoryObj.get_tensor(0)`` returns the K view;
    ``get_tensor(1)`` returns the V view.  Both groups carry the
    same dtype today; heterogeneous K / V dtypes are future work."""


def derive_storage_layout_mode(
    adapter_configs: list["L2AdapterConfigBase"],
) -> StorageLayoutMode:
    """Derive the canonical L1 storage layout from configured L2 adapters.

    Inspects each adapter's ``serde_config``: builds the corresponding
    ``SerdeProcessor`` and queries its ``input_slot_mapping``.  A
    non-``None`` mapping means the serde is multi-output and demands
    the KV-component-groups layout; ``None`` is single-tensor.

    All adapters must agree on the layout mode.  Mixed configurations
    (e.g. one adapter with ``fp8`` plus another with ``asym_k16_v8``)
    are rejected because they cannot share one canonical L1
    ``MemoryObj`` shape.

    Args:
        adapter_configs: List of ``L2AdapterConfigBase`` instances
            (each may carry a ``serde_config``).  An empty list
            returns the default :attr:`StorageLayoutMode.PACKED`.

    Returns:
        The single canonical layout mode for the storage path.

    Raises:
        ValueError: If adapters demand incompatible modes.
    """
    # First Party
    from lmcache.v1.distributed.serde import create_serde_processor

    modes: set[StorageLayoutMode] = set()
    for ac in adapter_configs:
        sc = getattr(ac, "serde_config", None)
        if sc is None:
            modes.add(StorageLayoutMode.PACKED)
            continue
        processor = create_serde_processor(sc)
        try:
            mapping = processor.input_slot_mapping()
        finally:
            processor.close()
        if mapping is None:
            modes.add(StorageLayoutMode.PACKED)
        else:
            modes.add(StorageLayoutMode.KV_COMPONENT_GROUPS)

    if not modes:
        return StorageLayoutMode.PACKED
    if len(modes) > 1:
        names = sorted(m.value for m in modes)
        raise ValueError(
            f"Incompatible L2 adapter storage layout modes: {names}. "
            f"All adapters must share one canonical L1 MemoryObj shape; "
            f"a single-tensor serde (e.g. fp8) cannot coexist with a "
            f"multi-output serde (e.g. asym_k16_v8) on the same "
            f"StorageManager."
        )
    return modes.pop()


def apply_kv_component_split(
    layout_desc: MemoryLayoutDesc,
    *,
    k_dtype: torch.dtype | None = None,
    v_dtype: torch.dtype | None = None,
) -> MemoryLayoutDesc:
    """Split a packed ``[2, ...]`` KV layout into K and V component groups.

    The packed convention is that each input group has leading dim 2 =
    (K, V), with K at index 0 and V at index 1 of that dim.  This
    transform drops the leading 2 and emits two component groups of
    shape ``[...]`` each.  By default both children inherit the input
    group's dtype (the symmetric, byte-identical case).  For asymmetric
    K/V (e.g. bf16 K + fp8 V) pass ``k_dtype`` / ``v_dtype`` to override
    the K child and V child dtype respectively; passing neither reproduces
    the pre-existing same-dtype behaviour exactly.  The element *count*
    per child is unchanged; the V child's byte size follows its (possibly
    smaller) dtype.

    For a multi-input layout (e.g. MLA or other configurations where
    ``shapes`` already has more than one entry), every input group is
    expanded in order: ``[g0_K, g0_V, g1_K, g1_V, ...]``.

    Args:
        layout_desc: Input layout where each group has leading dim 2.
        k_dtype: Optional dtype override for every K child. ``None``
            keeps the input group's dtype (symmetric default).
        v_dtype: Optional dtype override for every V child. ``None``
            keeps the input group's dtype (symmetric default). For
            asymmetric K16/V8 this is the fp8 value dtype.

    Returns:
        New ``MemoryLayoutDesc`` with 2× the groups, each carrying
        the leading-2 stripped from the corresponding input group.

    Raises:
        ValueError: If any input group's leading dim is not 2.
    """
    new_shapes: list[torch.Size] = []
    new_dtypes: list[torch.dtype] = []
    for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
        if len(shape) < 1 or shape[0] != 2:
            raise ValueError(
                f"apply_kv_component_split: input group shape {tuple(shape)} "
                f"must have leading dim 2 (K|V packed); got "
                f"{shape[0] if len(shape) else 'empty'}"
            )
        component_shape = torch.Size(shape[1:])
        new_shapes.append(component_shape)
        new_dtypes.append(k_dtype if k_dtype is not None else dtype)
        new_shapes.append(component_shape)
        new_dtypes.append(v_dtype if v_dtype is not None else dtype)
    return MemoryLayoutDesc(shapes=new_shapes, dtypes=new_dtypes)


def apply_layout_policy(
    layout_desc: MemoryLayoutDesc,
    mode: StorageLayoutMode,
    *,
    k_dtype: torch.dtype | None = None,
    v_dtype: torch.dtype | None = None,
) -> MemoryLayoutDesc:
    """Convert a packed layout to the canonical shape for ``mode``.

    For :attr:`StorageLayoutMode.PACKED`, returns ``layout_desc``
    unchanged.  For :attr:`StorageLayoutMode.KV_COMPONENT_GROUPS`,
    splits each input group via :func:`apply_kv_component_split`,
    forwarding any ``k_dtype`` / ``v_dtype`` override for asymmetric
    K/V (e.g. bf16 K + fp8 V).

    Args:
        layout_desc: The packed layout the upstream (transfer-kernel
            side) produced.
        mode: The canonical storage layout mode for this
            ``StorageManager``.
        k_dtype: Optional K-child dtype override (KV_COMPONENT_GROUPS
            only). ``None`` keeps the packed dtype.
        v_dtype: Optional V-child dtype override (KV_COMPONENT_GROUPS
            only). ``None`` keeps the packed dtype.

    Returns:
        The layout to pass to ``StorageManager.reserve_write``.

    Raises:
        ValueError: For an unknown ``mode``, or if a dtype override is
            supplied for :attr:`StorageLayoutMode.PACKED` (which cannot
            express heterogeneous K/V dtypes).
    """
    if mode == StorageLayoutMode.PACKED:
        if k_dtype is not None or v_dtype is not None:
            raise ValueError(
                "apply_layout_policy: k_dtype/v_dtype overrides are not valid "
                "for StorageLayoutMode.PACKED (a single packed group cannot "
                "carry heterogeneous K/V dtypes); use KV_COMPONENT_GROUPS."
            )
        return layout_desc
    if mode == StorageLayoutMode.KV_COMPONENT_GROUPS:
        return apply_kv_component_split(layout_desc, k_dtype=k_dtype, v_dtype=v_dtype)
    raise ValueError(f"Unknown StorageLayoutMode: {mode!r}")
