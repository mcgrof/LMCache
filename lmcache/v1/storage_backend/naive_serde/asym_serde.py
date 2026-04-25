# SPDX-License-Identifier: Apache-2.0
"""Asymmetric K16/V8 serde adapter.

Bridges the codec layer (lmcache/v1/kv_codec) into the
Serializer / Deserializer interface that LMCache's storage path
expects.

Two runtime modes (configured at construction; serde itself is
stateless aside from config):

  storage_only_dequant
      vLLM hands LMCache a fully-materialized FP16/BF16 KV tensor.
      Serializer quantizes V to FP8 on disk; deserializer
      dequantizes V back to the original dtype.  Saves disk
      capacity and disk I/O.  Does not save HBM capacity.

  native_asym  (Phase 4 — not implemented in Phase 2)
      vLLM/FlashInfer already has V in FP8.  Serializer copies
      bytes; deserializer returns K, V_fp8, scales without
      re-expanding.  Saves disk + HBM.

The cross-codec mismatch case (writing under naive, reading under
asym, or vice versa) is handled at the serde *factory* level
(naive_serde/__init__.py): each MemoryObj's MemoryFormat marks
which codec wrote it; reading the wrong codec raises
CodecMismatchError before any byte interpretation happens.
"""

# Standard
from __future__ import annotations

from typing import Any

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    CodecHashes,
    CodecMismatchError,
    EncodedKV,
    ScaleScope,
)
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer


logger = init_logger(__name__)


# Shared by the asymmetric serializer/deserializer pair.  The codec
# is stateless aside from configuration; one instance handles all
# serialize/deserialize calls within an engine.
class _AsymK16V8SerdeBase:
    """Shared state for AsymK16V8Serializer and Deserializer."""

    def __init__(
        self,
        config: Any,
        metadata: Any,
        *,
        runtime_layout: str = "storage_only_dequant",
        scale_scope: ScaleScope = ScaleScope.PER_TENSOR,
    ):
        # Typed loosely to avoid an import cycle through
        # lmcache.v1.metadata; the only attribute we read is
        # `metadata.model_name`.
        if runtime_layout not in ("storage_only_dequant", "native_asym"):
            raise ValueError(
                f"AsymK16V8 serde: unknown runtime_layout {runtime_layout!r}; "
                f"expected 'storage_only_dequant' or 'native_asym'"
            )
        if runtime_layout == "native_asym":
            raise NotImplementedError(
                "native_asym runtime layout is Phase 4; this is Phase 2. "
                "Use storage_only_dequant for now."
            )
        self.runtime_layout = runtime_layout
        self.codec = AsymK16V8Codec(scale_scope=scale_scope)
        # Pull the model identifier from the engine metadata so we
        # can stamp it into the codec header for cross-config
        # poisoning detection.
        self.model_id = getattr(metadata, "model_name", "") or ""
        self.attention_backend = ""
        self.kv_layout = "lmcache_kv_2ltd"
        # Both ends use the same expected hashes; mismatch surfaces
        # at deserialize time when the field on the encoded blob
        # does not match what we configured.
        self.expected_hashes = CodecHashes(
            model_id=self.model_id,
            attention_backend=self.attention_backend,
            kv_layout=self.kv_layout,
        )


class AsymK16V8Serializer(_AsymK16V8SerdeBase, Serializer):
    """Serializer for asymmetric K16/V8 cache layout.

    Input MemoryObj must carry a tensor whose first dim is 2 and
    represents (K, V) split (the standard KV_2LTD / KV_2TD / KV_2D
    formats — anything where ``tensor[0]`` is K and ``tensor[1]`` is V).
    """

    def serialize(self, memory_obj: MemoryObj) -> MemoryObj:
        tensor = memory_obj.tensor
        if tensor is None:
            raise ValueError(
                "AsymK16V8Serializer: input MemoryObj has no tensor; "
                "cannot encode asymmetric KV from a bytes-only object"
            )
        if tensor.ndim < 2 or tensor.shape[0] != 2:
            raise ValueError(
                f"AsymK16V8Serializer: expected leading dim 2 (K/V split), "
                f"got tensor of shape {tuple(tensor.shape)}"
            )
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"AsymK16V8Serializer: input K dtype {tensor.dtype} is not "
                f"FP16 or BF16; this serde does not support quantized inputs "
                f"in storage_only_dequant mode"
            )

        k = tensor[0].contiguous()
        v = tensor[1].contiguous()

        encoded = self.codec.encode(k, v, hashes=self.expected_hashes)
        blob = self.codec.to_bytes(encoded)

        # Wrap in a BytesBufferMemoryObj.  We use BINARY format and
        # carry the original logical shape via the tensor `shapes`
        # field so the deserializer can reconstruct.
        bytes_meta = MemoryObjMetadata(
            shape=torch.Size([len(blob), 0, 0, 0]),
            dtype=None,
            address=0,
            phy_size=0,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.BINARY_BUFFER,
            # Carry the logical (K, V) tensor's full shape and dtype
            # so backends that introspect them get sane answers.
            shapes=[torch.Size(tensor.shape)],
            dtypes=[tensor.dtype],
        )
        return BytesBufferMemoryObj(raw_bytes=blob, metadata=bytes_meta)


class AsymK16V8Deserializer(_AsymK16V8SerdeBase, Deserializer):
    """Deserializer for asymmetric K16/V8 cache layout.

    In `storage_only_dequant` mode this returns a `TensorMemoryObj`
    with a `[2, ...]` tensor at the original FP16/BF16 dtype; V has
    passed through FP8 quant + dequant.
    """

    def deserialize(self, memory_obj: MemoryObj) -> MemoryObj:
        if not isinstance(memory_obj, BytesBufferMemoryObj):
            raise ValueError(
                f"AsymK16V8Deserializer: expected BytesBufferMemoryObj, "
                f"got {type(memory_obj).__name__}"
            )
        blob = memory_obj.raw_data
        if not isinstance(blob, (bytes, bytearray)):
            raise ValueError(
                "AsymK16V8Deserializer: BytesBufferMemoryObj.raw_data is "
                "not bytes"
            )
        # Cross-config gate: codec checks model_id / attention_backend /
        # kv_layout against what we configured.  Mismatch -> exception.
        try:
            encoded = self.codec.from_bytes(
                bytes(blob), expected_hashes=self.expected_hashes
            )
        except CodecMismatchError:
            # Re-raise so the storage-engine layer can decide whether
            # to evict or fail the read.
            raise

        return self._materialize_storage_only(encoded, memory_obj.metadata)

    def _materialize_storage_only(
        self,
        encoded: EncodedKV,
        in_meta: MemoryObjMetadata,
    ) -> MemoryObj:
        # Decode K (native dtype) + V (dequantized to native dtype)
        out_v_dtype = encoded.k_dtype  # match V to K's dtype on the way out
        k, v_dq, _scales = self.codec.decode(
            encoded, out_v_dtype=out_v_dtype
        )

        # Reconstruct the original [2, ...] layout.  We carried the
        # logical shape in shapes[0] at serialize time.
        if in_meta.shapes and len(in_meta.shapes) >= 1:
            target_shape = in_meta.shapes[0]
        else:
            # Fallback: reshape to [2, -1] flat
            target_shape = torch.Size([2, k.numel()])

        # The codec returned flat 1-D buffers; reshape K and V to
        # match the per-half logical shape (target[1:]) and stack.
        per_half_shape = torch.Size(list(target_shape)[1:])
        k = k.reshape(per_half_shape)
        v_dq = v_dq.reshape(per_half_shape)
        out = torch.stack([k, v_dq], dim=0).contiguous()

        out_meta = MemoryObjMetadata(
            shape=torch.Size(out.shape),
            dtype=out.dtype,
            address=0,
            phy_size=out.numel() * out.element_size(),
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.KV_2LTD,
        )
        return TensorMemoryObj(
            raw_data=out, metadata=out_meta, parent_allocator=None
        )


