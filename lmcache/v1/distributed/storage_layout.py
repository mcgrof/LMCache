# SPDX-License-Identifier: Apache-2.0
"""
Storage layout policy: derive the canonical L1 MemoryObj shape from the
configured L2 adapters' serdes.

LMCache historically stores KV chunks as a single packed tensor with
leading dim 2 = (K, V).  Single-tensor serdes (e.g. ``fp8``) cast the
whole packed tensor uniformly and never look at K vs V separately.
Multi-output serdes (e.g. ``asym_k16_v8``) need K and V as distinct
typed sub-objects so they can quantize V to FP8 while leaving K
bit-exact.  This module is the single source of truth that decides,
from the configured ``serde_config`` on each L2 adapter, which canonical
layout the storage path should construct.

Design (per codex/gpt-5.5/xhigh, 2026-05-30):

* The choice is opt-in per serde, not universal.  ``fp8`` and any
  future single-tensor serdes keep the packed layout unchanged.  Only
  serdes whose ``SerdeProcessor`` reports a non-``None``
  ``input_slot_mapping`` opt into the KV-component-groups layout.
* The query lives in LMCache (not in the vLLM / SGLang connectors)
  so both integrations inherit the policy without duplicating
  serde-shape branching.
* Multiple L2 adapters with conflicting layout requirements are
  rejected at config time.  One canonical L1 object layout serves
  all downstream adapters.

Scope (Phase 1):

* Same-dtype K and V (BF16 / FP16) — the asym serde quantizes V to
  FP8 internally during serialize, so the **storage**-side K and V
  share the model's native dtype.  The packed buffer's K bytes
  followed by V bytes are byte-identical to a two-group layout's
  K bytes followed by V bytes — only the typed view via
  ``TensorMemoryObj.get_tensor(i)`` changes.  No transfer-kernel
  change required.
* Heterogeneous K / V dtypes (e.g. vllm-asym's native K BF16 + V FP8)
  are out of scope here.  They need transfer-kernel work first.
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


def apply_kv_component_split(layout_desc: MemoryLayoutDesc) -> MemoryLayoutDesc:
    """Split a packed ``[2, ...]`` KV layout into K and V component groups.

    The packed convention is that each input group has leading dim 2 =
    (K, V), with K at index 0 and V at index 1 of that dim.  This
    transform drops the leading 2 and emits two component groups of
    shape ``[...]`` each, both at the input group's dtype.  Total bytes
    are unchanged; only the typed view changes.

    For a multi-input layout (e.g. MLA or other configurations where
    ``shapes`` already has more than one entry), every input group is
    expanded in order: ``[g0_K, g0_V, g1_K, g1_V, ...]``.

    Args:
        layout_desc: Input layout where each group has leading dim 2.

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
        new_dtypes.append(dtype)
        new_shapes.append(component_shape)
        new_dtypes.append(dtype)
    return MemoryLayoutDesc(shapes=new_shapes, dtypes=new_dtypes)


def apply_layout_policy(
    layout_desc: MemoryLayoutDesc,
    mode: StorageLayoutMode,
) -> MemoryLayoutDesc:
    """Convert a packed layout to the canonical shape for ``mode``.

    For :attr:`StorageLayoutMode.PACKED`, returns ``layout_desc``
    unchanged.  For :attr:`StorageLayoutMode.KV_COMPONENT_GROUPS`,
    splits each input group via :func:`apply_kv_component_split`.

    Args:
        layout_desc: The packed layout the upstream (transfer-kernel
            side) produced.
        mode: The canonical storage layout mode for this
            ``StorageManager``.

    Returns:
        The layout to pass to ``StorageManager.reserve_write``.
    """
    if mode == StorageLayoutMode.PACKED:
        return layout_desc
    if mode == StorageLayoutMode.KV_COMPONENT_GROUPS:
        return apply_kv_component_split(layout_desc)
    raise ValueError(f"Unknown StorageLayoutMode: {mode!r}")
