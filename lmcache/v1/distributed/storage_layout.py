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
    from lmcache.v1.distributed.storage_placement import ComponentKeyScheme


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


class StorageInputForm(Enum):
    """How the transfer side presents K and V to the layout policy.

    Selects which transform the ``KV_COMPONENT_GROUPS`` layout mode
    applies.  It is a *structural* classification only; the chosen
    transform then independently validates dtype, canonical order and
    the per-engine ``component_key_scheme`` (see
    :func:`apply_layout_policy`).  Shape never decides K vs V role."""

    PACKED_KV_UNIFORM = "packed_kv_uniform"
    """One group per KV pair, leading dim ``2`` = (K, V), with K and V
    sharing one dtype / element size.  LMCache splits it into K and V
    component groups (and, for the byte-through V-only path, overrides
    the V child dtype).  Mechanism B (LMCache self-quantizes V) and the
    packed synthetic serde tests produce this form."""

    PRESPLIT_COMPONENTS = "presplit_components"
    """K and V arrive as already-separate component groups (leading dim
    ``1``, ``kv_size == 1`` each), in canonical ``[K, V, ...]`` order,
    carrying their own heterogeneous dtypes (bf16 K, fp8 V).  vLLM
    mechanism-A stores V as its own fp8 plane, so the live MP
    byte-through connector presents this form.  LMCache passes it
    through unchanged after validation -- no split, no dtype override,
    no dequant."""


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


def classify_input_form(layout_desc: MemoryLayoutDesc) -> StorageInputForm:
    """Classify the transfer-side input as packed or pre-split, by structure.

    Every input group must share one form.  A layout that mixes a
    leading-dim-2 packed group with a leading-dim-1 pre-split component,
    or carries any other leading dim, is rejected: a real registration
    never produces a mix, and guessing per-group would risk a silent
    K/V mis-pairing.

    This only SELECTS which transform runs; the chosen transform then
    independently validates dtype, canonical order and scheme.  Shape is
    never used to decide K vs V role.

    Args:
        layout_desc: The transfer-side layout (one entry per kernel
            group, in the kernel groups' declared order).

    Returns:
        :attr:`StorageInputForm.PACKED_KV_UNIFORM` if every group has
        leading dim ``2``; :attr:`StorageInputForm.PRESPLIT_COMPONENTS`
        if every group has leading dim ``1``.

    Raises:
        ValueError: On an empty layout, a 0-d group, a mixed layout, or
            any leading dim other than 1 or 2.
    """
    if not layout_desc.shapes:
        raise ValueError("classify_input_form: empty layout_desc")
    leading: list[int] = []
    for shape in layout_desc.shapes:
        if len(shape) < 1:
            raise ValueError(
                f"classify_input_form: 0-d group shape {tuple(shape)} has no "
                "leading K|V / component dim"
            )
        leading.append(shape[0])
    if all(d == 2 for d in leading):
        return StorageInputForm.PACKED_KV_UNIFORM
    if all(d == 1 for d in leading):
        return StorageInputForm.PRESPLIT_COMPONENTS
    raise ValueError(
        f"classify_input_form: layout mixes or has unexpected leading dims "
        f"{leading}; expected all 2 (packed K|V) or all 1 (pre-split "
        "components)"
    )


def pass_through_presplit_component_groups(
    layout_desc: MemoryLayoutDesc,
    *,
    v_dtype: torch.dtype,
) -> MemoryLayoutDesc:
    """Validate and pass through already-split K/V component groups.

    The live MP byte-through path (vLLM mechanism-A) hands K and V as
    separate component groups -- K bf16, V fp8 -- in canonical
    ``[K, V, K, V, ...]`` order (one (K, V) pair per attention layer
    group).  LMCache stores them as they are: the split already happened
    in vLLM and V is already fp8, so there is nothing to split,
    re-quantize or upcast.

    Role is NOT inferred from dtype.  The order is the authoritative
    contract (K at even index, V at odd index, fixed by the connector's
    plane registration).  This function VALIDATES that the tensor at each
    position carries the dtype that position requires, which also makes a
    K/V order inversion fail closed -- K and V have different dtypes in
    the asymmetric case, so a swap trips the dtype check.  No dtype is
    overridden and total bytes are unchanged.

    Args:
        layout_desc: Pre-split component groups (each leading dim ``1``),
            an even number of them, in ``[K, V]`` pair order.
        v_dtype: The required V-component dtype (``float8_e4m3fn`` for the
            byte-through path).  Every V position must equal it; no K
            position may.

    Returns:
        ``layout_desc`` unchanged (validated).

    Raises:
        ValueError: On an odd / zero group count, a group whose leading
            dim is not 1, a V position whose dtype != ``v_dtype``, or a K
            position whose dtype == ``v_dtype`` (either means K and V were
            swapped or a component was wrongly typed).
    """
    n = len(layout_desc.shapes)
    if n == 0 or n % 2 != 0:
        raise ValueError(
            "pass_through_presplit_component_groups: expected an even, "
            f"non-zero number of [K, V] component groups; got {n}"
        )
    for idx, (shape, dtype) in enumerate(
        zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
    ):
        if len(shape) < 1 or shape[0] != 1:
            raise ValueError(
                f"pass_through_presplit_component_groups: component group "
                f"{idx} shape {tuple(shape)} must have leading dim 1 "
                "(kv_size == 1, already split)"
            )
        is_v_position = idx % 2 == 1
        if is_v_position and dtype != v_dtype:
            raise ValueError(
                f"pass_through_presplit_component_groups: V component at index "
                f"{idx} has dtype {dtype}, expected {v_dtype} (byte-through V "
                "must already be fp8; a mismatch means a K/V order inversion "
                "or an un-quantized V)"
            )
        if not is_v_position and dtype == v_dtype:
            raise ValueError(
                f"pass_through_presplit_component_groups: K component at index "
                f"{idx} has the V dtype {v_dtype}; K must not be fp8 (a K/V "
                "order inversion would corrupt decode silently)"
            )
    return layout_desc


def apply_layout_policy(
    layout_desc: MemoryLayoutDesc,
    mode: StorageLayoutMode,
    *,
    k_dtype: torch.dtype | None = None,
    v_dtype: torch.dtype | None = None,
    scheme: "ComponentKeyScheme | None" = None,
) -> MemoryLayoutDesc:
    """Convert the transfer-side layout to the canonical shape for ``mode``.

    For :attr:`StorageLayoutMode.PACKED`, returns ``layout_desc``
    unchanged.  For :attr:`StorageLayoutMode.KV_COMPONENT_GROUPS`, the
    input is first classified (:func:`classify_input_form`):

    * :attr:`StorageInputForm.PACKED_KV_UNIFORM` -- one packed
      leading-dim-2 group -- is split via :func:`apply_kv_component_split`,
      forwarding any ``k_dtype`` / ``v_dtype`` override (mechanism B and
      the packed byte-through tests).
    * :attr:`StorageInputForm.PRESPLIT_COMPONENTS` -- already-separate K
      (bf16) and V (fp8) groups -- is validated and passed through via
      :func:`pass_through_presplit_component_groups` with NO split and NO
      dtype override (the live MP byte-through path).  This form is only
      valid under the ``RAW_UNIT`` scheme; ``COMPUTED_LEGACY`` must
      receive packed input because it splits and quantizes V itself.

    The pass-through decision is gated on the explicit per-engine
    ``scheme`` and validated by dtype + canonical order; shape only
    classifies, it never assigns K vs V role.

    Args:
        layout_desc: The layout the upstream (transfer-kernel side)
            produced -- one entry per kernel group.
        mode: The canonical storage layout mode for this
            ``StorageManager``.
        k_dtype: Optional K-child dtype override (packed split only).
        v_dtype: Optional V-child dtype override (packed split) AND the
            required V dtype for pre-split validation (``float8_e4m3fn``
            on the byte-through path).
        scheme: The per-engine ``component_key_scheme``.  Required to be
            ``RAW_UNIT`` for a pre-split input; ignored for a packed input.

    Returns:
        The layout to pass to ``StorageManager.reserve_write`` /
        ``submit_prefetch_task``.

    Raises:
        ValueError: For an unknown ``mode``; a dtype override on
            :attr:`StorageLayoutMode.PACKED`; a pre-split input under a
            non-``RAW_UNIT`` scheme or without ``v_dtype``; or any layout /
            dtype / order validation failure from the delegates.
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
        input_form = classify_input_form(layout_desc)
        if input_form == StorageInputForm.PACKED_KV_UNIFORM:
            return apply_kv_component_split(
                layout_desc, k_dtype=k_dtype, v_dtype=v_dtype
            )
        # PRESPLIT_COMPONENTS: only the byte-through (RAW_UNIT) scheme
        # presents pre-split heterogeneous K/V.  COMPUTED_LEGACY must
        # receive packed input (it splits and quantizes V itself), so a
        # pre-split input under it is a contract violation -> fail closed.
        # First Party
        from lmcache.v1.distributed.storage_placement import ComponentKeyScheme

        if scheme is not ComponentKeyScheme.RAW_UNIT:
            raise ValueError(
                "apply_layout_policy: pre-split component input requires the "
                "RAW_UNIT (byte-through) component_key_scheme; got "
                f"{scheme!r}. COMPUTED_LEGACY expects packed leading-dim-2 "
                "input to split and quantize V itself."
            )
        if v_dtype is None:
            raise ValueError(
                "apply_layout_policy: RAW_UNIT pre-split pass-through requires "
                "v_dtype (the fp8 V-component dtype) to validate the V groups."
            )
        return pass_through_presplit_component_groups(layout_desc, v_dtype=v_dtype)
    raise ValueError(f"Unknown StorageLayoutMode: {mode!r}")
