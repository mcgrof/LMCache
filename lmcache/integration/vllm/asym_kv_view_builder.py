# SPDX-License-Identifier: Apache-2.0
"""Build an `AsymKVView` from vLLM's per-layer state.

This helper is the integration point between the asymmetric-kv-
plumbing branch of vLLM and LMCache's asymmetric KV codec.  When
vLLM is running with `kv_cache_dtype = ("auto", "fp8_e4m3")`, it
holds K and V in two separate buffers at different dtypes and
exposes a per-layer `_v_scale_float`.  The codec needs all three
(K, V_fp8, scales) to round-trip without re-quantization.

Because vLLM's paged-cache layout is internal and not part of a
stable public contract, the extraction logic lives in one place
(this file) where it can be updated against a specific vLLM
revision without changing the codec or the serde.

Public API:

    build_asym_kv_view(attention_layer, kv_layer, attn_metadata)
        -> Optional[AsymKVView]

    Returns an AsymKVView if the attention layer is asymmetric and
    K/V/scales can be extracted; returns None if the layer is
    symmetric (caller falls back to the regular FP16 path).

The function is conservative: any failure to extract the expected
state surfaces a warning and returns None rather than raising,
so the caller can degrade gracefully instead of crashing the
forward pass.
"""

# Standard
from __future__ import annotations

from typing import Any, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.naive_serde.asym_serde import (
    AsymKVView,
    detect_native_asym_capability,
)


logger = init_logger(__name__)


def build_asym_kv_view(
    attention_layer: Any,
    kv_layer: torch.Tensor,
    attn_metadata: Any = None,
    *,
    strict: bool = False,
) -> Optional[AsymKVView]:
    """Try to build an AsymKVView from vLLM's per-layer state.

    Args:
        attention_layer: the vLLM Attention object for this layer.
            Inspected for `kv_cache_dtype` (tuple form indicates
            asymmetric) and `_v_scale_float` / `_v_scale`.
        kv_layer: the paged KV tensor vLLM hands to LMCache via
            `save_kv_layer`.  In the asymmetric branch, this is a
            `(K_buffer, V_buffer)` tuple, OR a single tensor whose
            leading dim is 2 with K and V at the SAME dtype if the
            branch is symmetric.  The function distinguishes by
            calling `detect_native_asym_capability`.
        attn_metadata: optional, currently unused; kept for future
            extension when we need to read `seq_lens` or
            `slot_mapping` to slice partial chunks.
        strict: when True, mismatches raise; when False (default),
            mismatches log a warning and return None.

    Returns:
        AsymKVView if the layer is asymmetric and extraction
        succeeded, else None.
    """
    if not detect_native_asym_capability(attention_layer):
        # Symmetric vLLM or partially-asymmetric branch — caller
        # should use the regular FP16 path.
        return None

    # 1. Extract K and V from kv_layer.
    try:
        k, v_fp8 = _split_kv_layer(kv_layer, attention_layer)
    except Exception as e:
        msg = (
            f"build_asym_kv_view: failed to split kv_layer of "
            f"shape={getattr(kv_layer, 'shape', '?')} "
            f"dtype={getattr(kv_layer, 'dtype', '?')}: {e}"
        )
        if strict:
            raise
        logger.warning(msg)
        return None

    # Sanity: K must be a 2-byte float, V must be FP8.
    if k.dtype not in (torch.float16, torch.bfloat16):
        msg = (
            f"build_asym_kv_view: K dtype is {k.dtype}, expected "
            f"float16 or bfloat16; refusing to build view"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return None
    if v_fp8.dtype != torch.float8_e4m3fn:
        msg = (
            f"build_asym_kv_view: V dtype is {v_fp8.dtype}, "
            f"expected float8_e4m3fn; refusing to build view"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return None

    # 2. Extract V scales.
    scales = _extract_v_scales(attention_layer, v_fp8.shape, v_fp8.device)
    if scales is None:
        msg = "build_asym_kv_view: could not extract V scales from layer"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return None

    return AsymKVView(k=k, v_fp8=v_fp8, v_scales=scales)


def _split_kv_layer(
    kv_layer: Any, attention_layer: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull K and V tensors out of vLLM's kv_layer.

    The asymmetric vLLM branch can pass kv_layer in any of a few
    shapes:

    a) Tuple/list `(K, V)` where K is FP16/BF16 and V is FP8.
    b) A single tensor where dim 0 splits K and V (legal only when
       both halves share a dtype — only valid here when V is also
       FP16/BF16, which means the layer is actually symmetric).
    c) An attention layer attribute exposing K and V buffers
       separately; check `attention_layer.kv_cache` (a tuple).

    Try (a), then (c) as a fallback.  Reject (b) for asymmetric
    layers since that would imply same-dtype K and V.
    """
    # (a) Tuple/list passed directly.
    if isinstance(kv_layer, (tuple, list)) and len(kv_layer) == 2:
        return kv_layer[0], kv_layer[1]

    # (c) Attribute on the attention layer.
    layer_kv = getattr(attention_layer, "kv_cache", None)
    if isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 2:
        return layer_kv[0], layer_kv[1]

    # If we got a single tensor, the asymmetric branch should have
    # split K and V already; same-dtype "[2, ...]" is the symmetric
    # representation and shouldn't reach this code path.
    raise ValueError(
        f"kv_layer has unexpected shape: type={type(kv_layer).__name__}, "
        f"and attention_layer.kv_cache is also not a 2-tuple of tensors"
    )


def _extract_v_scales(
    attention_layer: Any,
    v_shape: tuple,
    v_device: torch.device,
) -> Optional[torch.Tensor]:
    """Read V scales off the attention layer.

    Two attribute names supported:
      - `_v_scale_float`: a Python float (per-layer per-tensor).
      - `_v_scale`: a torch tensor (per-layer per-head, shape
        `(num_kv_heads,)` or `()`).

    Returns an FP32 torch tensor on `v_device`, shaped per the
    detected scope.  Returns None if neither attribute is present.
    """
    # Prefer the tensor form (richer scope).
    v_scale_t = getattr(attention_layer, "_v_scale", None)
    if v_scale_t is not None and isinstance(v_scale_t, torch.Tensor):
        return v_scale_t.detach().to(torch.float32).to(v_device)

    v_scale_f = getattr(attention_layer, "_v_scale_float", None)
    if v_scale_f is not None:
        return torch.tensor(
            float(v_scale_f), dtype=torch.float32, device=v_device
        ).reshape(())

    return None
