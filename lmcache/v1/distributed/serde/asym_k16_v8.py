# SPDX-License-Identifier: Apache-2.0
"""
Asymmetric K16/V8 multi-output serde.

Concrete :class:`MultiSerializer` / :class:`MultiDeserializer`
pair that bridges :class:`AsymK16V8Codec` (in
``lmcache/v1/kv_codec``) into the tuple-shaped serde contract
from ``multi.py``.

**Storage-only-dequant mode.** Group of size 2 on both endpoints:

* serialize input ``src = (K, V)`` -- both at the model's native
  dtype (fp16 / bf16). The codec quantizes V to FP8 internally
  and writes a single byte buffer containing K bytes, V FP8
  bytes, and the codec header with V's scale.
* deserialize output ``dst = (K_out, V_out)`` -- both at the
  model's native dtype, with V dequantized from the stored FP8.

Compresses the bytes that ship to L2 storage; the consumer-visible
view on read is unchanged.

**Split-tier / V-only mode.** Group of size 2 on both endpoints;
K is held in L1 (CPU-pinned host memory) and only V flows through
this serde to L2:

* serialize input ``src = (None, V)`` -- the K slot MUST be
  ``None``. Emits an :class:`EncodedKV` with ``k_payload_len = 0``;
  the ``k_dtype`` tag is still recorded so cross-config gating
  works on the eventual restore.
* deserialize output ``dst = (None | K_skip, V_out)`` -- slot 0
  is a no-op regardless of input (K is sourced from L1); slot 1
  is dequantized from the stored FP8.
"""

# Future
from __future__ import annotations

# Standard
import math

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde.multi import (
    LayoutDescGroup,
    MemoryObjGroup,
    MultiDeserializer,
    MultiSerializer,
    validate_group_size,
)
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    EncodedKV,
    ScaleScheme,
    ScaleScope,
)
from lmcache.v1.kv_codec.asym_k16_v8 import (
    _tensor_to_bytes_fast,
    compute_v_scales,
    dequantize_v_fp8,
    quantize_v_fp8,
)
from lmcache.v1.memory_management import MemoryObj

_GROUP_SIZE_STORAGE_ONLY = 2  # (K, V) on both sides for this mode.
_GROUP_SIZE_V_ONLY = 2  # (None, V) on serialize ; (None|K_out, V_out) on deserialize.


def _v_fp8_max_for_dtype(fp8_dtype: torch.dtype) -> float:
    return float(torch.finfo(fp8_dtype).max)


def _is_fp8(dtype: torch.dtype) -> bool:
    """True for any 1-byte floating-point (FP8) dtype.

    Covers every ``float8_*`` variant (e4m3fn, e5m2, and the fnuz
    forms) without enumerating names that may be absent on a given
    torch build: an FP8 dtype is exactly a floating-point dtype whose
    element occupies one byte, whereas fp16/bf16 occupy two.
    """
    return dtype.is_floating_point and dtype.itemsize == 1


class AsymK16V8MultiSerializer(MultiSerializer):
    """Encode ``(K, V)`` into a single asymmetric K16/V8 byte blob.

    Slot semantics:

    * ``slot 0 = K`` (required, fp16 or bf16, native model dtype).
    * ``slot 1 = V`` (required, fp16 or bf16, native model dtype).

    ``None`` is not admitted in either slot for this mode -- both K
    and V must be supplied.

    The codec produces a self-describing blob: header (dtype tags,
    scale scope, scale tensor, optional shape metadata) followed by
    K bytes (native dtype), V bytes (FP8 e4m3), and scale bytes.
    The exact byte layout is owned by ``serialize_header`` /
    ``deserialize_header`` in ``encoded_kv``; this serializer is
    just glue so the codec is reachable through the
    :class:`MultiSerializer` interface.
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_scope: ScaleScope = ScaleScope.PER_TENSOR,
        scale_dtype: torch.dtype = torch.float32,
    ) -> None:
        self._codec = AsymK16V8Codec(
            fp8_dtype=fp8_dtype,
            scale_scope=scale_scope,
            scale_dtype=scale_dtype,
        )
        self._fp8_dtype = fp8_dtype

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE_STORAGE_ONLY

    def serialize(self, src: MemoryObjGroup, dst: MemoryObj, key: ObjectKey) -> int:
        # ``key`` unused: this serde is content-agnostic.
        validate_group_size(src, _GROUP_SIZE_STORAGE_ONLY, role="src")
        k_obj, v_obj = src
        if k_obj is None or v_obj is None:
            raise ValueError(
                "AsymK16V8MultiSerializer (storage-only-dequant): both K "
                "and V must be provided"
            )
        if k_obj.tensor is None or v_obj.tensor is None:
            raise ValueError(
                "AsymK16V8MultiSerializer: src MemoryObjs must have tensors set"
            )
        if dst.tensor is None:
            raise ValueError("AsymK16V8MultiSerializer: dst.tensor is None")
        if _is_fp8(v_obj.tensor.dtype):
            raise ValueError(
                "AsymK16V8MultiSerializer: V is already FP8 "
                f"({v_obj.tensor.dtype}); this serde quantizes a native-dtype "
                "(fp16/bf16) V and would double-quantize an fp8 input.  Use "
                "AsymBytethroughK16V8VOnlyMultiSerializer to store raw fp8 V "
                "bytes through (K stays in L1)."
            )

        enc = self._codec.encode(k_obj.tensor, v_obj.tensor)
        blob = self._codec.to_bytes(enc)
        n = len(blob)
        if dst.tensor.numel() < n:
            raise ValueError(
                f"AsymK16V8MultiSerializer: dst capacity {dst.tensor.numel()} "
                f"below required {n}"
            )
        dst_view = dst.tensor.view(torch.uint8)
        dst_view[:n].copy_(torch.frombuffer(blob, dtype=torch.uint8))
        return n

    def estimate_serialized_size(
        self,
        layout_descs: LayoutDescGroup,
    ) -> int:
        validate_group_size(layout_descs, _GROUP_SIZE_STORAGE_ONLY, role="layout")
        k_layout, v_layout = layout_descs
        if k_layout is None or v_layout is None:
            raise ValueError(
                "AsymK16V8MultiSerializer.estimate_serialized_size "
                "(storage-only-dequant): both K and V layouts required"
            )

        # Bytes accounting:
        #   K: sum(numel(shape)) * itemsize(K_dtype)
        #   V: sum(numel(shape)) * 1          (FP8 = 1 byte / elem)
        #   scales: scope-dependent; only PER_TENSOR and EXTERNAL are
        #           bounded by layout alone. PER_LAYER_HEAD and
        #           PER_PAGE_HEAD need head/page metadata that the
        #           layout descriptor does not carry, so they are
        #           rejected here -- callers with those scopes must
        #           extend the API rather than risk a silent undersize.
        #   header: <= 1 KB allowance for dtype tags + hashes + shape
        scope = self._codec.scale_scope
        scale_dtype = self._codec.scale_dtype
        if scope == ScaleScope.PER_TENSOR:
            scales_bytes = scale_dtype.itemsize
        elif scope == ScaleScope.EXTERNAL:
            scales_bytes = 0
        else:
            raise ValueError(
                f"AsymK16V8MultiSerializer.estimate_serialized_size: "
                f"scale_scope {scope} requires per-head/per-page metadata "
                f"that MemoryLayoutDesc does not carry; only PER_TENSOR "
                f"and EXTERNAL are supported here"
            )
        k_bytes = sum(
            math.prod(s) * d.itemsize
            for s, d in zip(k_layout.shapes, k_layout.dtypes, strict=True)
        )
        v_bytes = sum(math.prod(s) for s in v_layout.shapes)
        header_allowance = 1024
        return header_allowance + k_bytes + v_bytes + scales_bytes


class AsymK16V8MultiDeserializer(MultiDeserializer):
    """Decode an asymmetric K16/V8 byte blob into ``(K_out, V_out)``.

    Slot semantics (storage-only-dequant mode):

    * ``slot 0 = K_out`` — caller-provided MemoryObj at the model's
      native dtype, populated bit-exact from the blob.
    * ``slot 1 = V_out`` — caller-provided MemoryObj at the model's
      native dtype, populated by dequantizing the stored FP8 V.

    A ``None`` slot is treated as a deliberate skip (the corresponding
    output is left untouched), matching the contract documented in
    ``multi.py``. This provides the V-only-read shape: pass
    ``(K_out, None)`` to load K only; pass ``(None, V_out)`` to load
    V only.
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_scope: ScaleScope = ScaleScope.PER_TENSOR,
        scale_dtype: torch.dtype = torch.float32,
    ) -> None:
        self._codec = AsymK16V8Codec(
            fp8_dtype=fp8_dtype,
            scale_scope=scale_scope,
            scale_dtype=scale_dtype,
        )

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE_STORAGE_ONLY

    def deserialize(self, src: MemoryObj, dst: MemoryObjGroup, key: ObjectKey) -> None:
        # ``key`` unused: this serde is content-agnostic.
        validate_group_size(dst, _GROUP_SIZE_STORAGE_ONLY, role="dst")
        if src.tensor is None:
            raise ValueError("AsymK16V8MultiDeserializer: src.tensor is None")

        # Pull blob bytes out of the source uint8 buffer.
        src_view = src.tensor.view(torch.uint8).contiguous()
        blob = src_view.numpy().tobytes()

        # Decode through the codec; ``decode`` returns 1-D flat tensors
        # for K and V, plus the scale tensor.  We reshape into the
        # caller-provided dst shapes.
        k_obj, v_obj = dst
        if k_obj is None and v_obj is None:
            # Both slots skipped — nothing to do.
            return

        # Pick the V output dtype from whichever dst slot is set.  If
        # only K is requested we still decode V to the same dtype so
        # the codec contract (no silent fp16 fallback) stays explicit;
        # the V tensor is then discarded.
        if v_obj is not None and v_obj.tensor is not None:
            target_v_dtype = v_obj.tensor.dtype
        elif k_obj is not None and k_obj.tensor is not None:
            target_v_dtype = k_obj.tensor.dtype
        else:
            raise ValueError(
                "AsymK16V8MultiDeserializer: at least one non-None dst "
                "slot must have a tensor set"
            )

        enc = self._codec.from_bytes(blob)
        if enc.scale_scheme != ScaleScheme.COMPUTED_PER_TENSOR:
            raise ValueError(
                "AsymK16V8MultiDeserializer: blob scale_scheme="
                f"{enc.scale_scheme.name} is not COMPUTED_PER_TENSOR; a "
                "RAW_UNIT (byte-through) blob carries raw fp8 codes with no "
                "scales and must be decoded by "
                "AsymBytethroughK16V8VOnlyMultiDeserializer."
            )
        # Push the K dtype conversion into the codec so the serde
        # doesn't need a post-decode cast.
        if k_obj is not None and k_obj.tensor is not None:
            out_k_dtype = k_obj.tensor.dtype
        else:
            out_k_dtype = None
        k_flat, v_dq_flat, _scales = self._codec.decode(
            enc, out_k_dtype=out_k_dtype, out_v_dtype=target_v_dtype
        )

        if k_obj is not None:
            if k_obj.tensor is None:
                raise ValueError(
                    "AsymK16V8MultiDeserializer: non-None K dst slot "
                    "must have a tensor set"
                )
            target_shape = k_obj.tensor.shape
            if k_flat.numel() != k_obj.tensor.numel():
                raise ValueError(
                    f"AsymK16V8MultiDeserializer: decoded K has "
                    f"{k_flat.numel()} elements, dst K shape "
                    f"{tuple(target_shape)} expects "
                    f"{k_obj.tensor.numel()}"
                )
            k_obj.tensor.copy_(k_flat.reshape(target_shape))

        if v_obj is not None:
            if v_obj.tensor is None:
                raise ValueError(
                    "AsymK16V8MultiDeserializer: non-None V dst slot "
                    "must have a tensor set"
                )
            target_shape = v_obj.tensor.shape
            if v_dq_flat.numel() != v_obj.tensor.numel():
                raise ValueError(
                    f"AsymK16V8MultiDeserializer: decoded V has "
                    f"{v_dq_flat.numel()} elements, dst V shape "
                    f"{tuple(target_shape)} expects "
                    f"{v_obj.tensor.numel()}"
                )
            v_obj.tensor.copy_(v_dq_flat.reshape(target_shape))


class AsymK16V8VOnlyMultiSerializer(MultiSerializer):
    """Encode V-only into a split-tier asymmetric byte blob.

    Split-tier path: K stays in L1 (CPU-pinned host memory) and is
    not written to the byte buffer; only the FP8-quantized V plus
    its scales hit L2.  The blob is a regular :class:`EncodedKV`
    with ``k_payload_len = 0`` and the ``k_dtype`` tag set to
    whatever dtype K would have been (so cross-config gating still
    works on the eventual restore).

    Slot semantics:

    * ``slot 0 = K`` MUST be ``None``.  Passing a tensor here is a
      contract error: this serde does not write K bytes by design.
      Use :class:`AsymK16V8MultiSerializer` instead if you want a
      self-contained restorable object.
    * ``slot 1 = V`` (required, fp16 or bf16, native model dtype).

    Byte ratio vs FP16 KV: ``V_8 / (K_16 + V_16) = 1/4``,
    equivalently ``1/3`` of the corresponding storage-only-dequant
    blob (which carries both K and V).
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_scope: ScaleScope = ScaleScope.PER_TENSOR,
        scale_dtype: torch.dtype = torch.float32,
        # The k_dtype tag is recorded in the header so a future
        # restore that pairs this V blob with its CPU-resident K can
        # cross-check dtype agreement.  Defaults to bfloat16.
        k_dtype_tag: torch.dtype = torch.bfloat16,
    ) -> None:
        # Reuse the same codec instance for header serialization.
        self._codec = AsymK16V8Codec(
            fp8_dtype=fp8_dtype,
            scale_scope=scale_scope,
            scale_dtype=scale_dtype,
        )
        self._fp8_dtype = fp8_dtype
        self._scale_dtype = scale_dtype
        self._scale_scope = scale_scope
        self._k_dtype_tag = k_dtype_tag

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE_V_ONLY

    def input_slot_mapping(self) -> "tuple[int | None, ...]":
        # Split-tier: K stays in L1 / host -- never passed to this
        # serializer.  Slot 0 is always None; slot 1 reads parent
        # group 1 (V).
        return (None, 1)

    def serialize(self, src: MemoryObjGroup, dst: MemoryObj, key: ObjectKey) -> int:
        # ``key`` unused: this serde is content-agnostic.
        validate_group_size(src, _GROUP_SIZE_V_ONLY, role="src")
        k_obj, v_obj = src
        if k_obj is not None:
            raise ValueError(
                "AsymK16V8VOnlyMultiSerializer (split-tier): K slot must "
                "be None.  K stays in host RAM in this mode and is not "
                "written to the byte buffer.  Use "
                "AsymK16V8MultiSerializer for the self-contained "
                "(K, V) write path."
            )
        if v_obj is None or v_obj.tensor is None:
            raise ValueError(
                "AsymK16V8VOnlyMultiSerializer: V slot is required and "
                "must have a tensor set"
            )
        if dst.tensor is None:
            raise ValueError("AsymK16V8VOnlyMultiSerializer: dst.tensor is None")

        v = v_obj.tensor
        if _is_fp8(v.dtype):
            raise ValueError(
                "AsymK16V8VOnlyMultiSerializer: V is already FP8 "
                f"({v.dtype}); this serde quantizes a native-dtype (fp16/bf16) "
                "V and would double-quantize an fp8 input.  Use "
                "AsymBytethroughK16V8VOnlyMultiSerializer to store raw fp8 V "
                "bytes through."
            )

        # Compute per-tensor (or per-scope) V scales and quantize V.
        v_scales = compute_v_scales(
            v,
            self._scale_scope,
            fp8_dtype=self._fp8_dtype,
        ).to(self._scale_dtype)
        v_quant = quantize_v_fp8(
            v,
            v_scales,
            self._scale_scope,
            fp8_dtype=self._fp8_dtype,
        )

        v_cpu = v_quant.detach().to("cpu").contiguous().clone()
        s_cpu = v_scales.detach().to("cpu").contiguous().clone()
        v_bytes = _tensor_to_bytes_fast(v_cpu)
        s_bytes = _tensor_to_bytes_fast(s_cpu)

        enc = EncodedKV(
            k_dtype=self._k_dtype_tag,
            v_dtype=self._fp8_dtype,
            scale_dtype=self._scale_dtype,
            scale_scope=self._scale_scope,
            k_payload_len=0,
            v_payload_len=len(v_bytes),
            scale_payload_len=len(s_bytes),
            scale_shape=tuple(v_scales.shape),
            payload=v_bytes + s_bytes,
        )
        blob = self._codec.to_bytes(enc)
        n = len(blob)
        if dst.tensor.numel() < n:
            raise ValueError(
                f"AsymK16V8VOnlyMultiSerializer: dst capacity "
                f"{dst.tensor.numel()} below required {n}"
            )
        dst_view = dst.tensor.view(torch.uint8)
        dst_view[:n].copy_(torch.frombuffer(blob, dtype=torch.uint8))
        return n

    def estimate_serialized_size(
        self,
        layout_descs: LayoutDescGroup,
    ) -> int:
        validate_group_size(layout_descs, _GROUP_SIZE_V_ONLY, role="layout")
        k_layout, v_layout = layout_descs
        if k_layout is not None:
            raise ValueError(
                "AsymK16V8VOnlyMultiSerializer.estimate_serialized_size: "
                "K layout must be None — this mode does not write K bytes"
            )
        if v_layout is None:
            raise ValueError(
                "AsymK16V8VOnlyMultiSerializer.estimate_serialized_size: "
                "V layout is required"
            )

        # Bytes accounting:
        #   V (fp8): sum(numel(shape)) * 1 byte/elem
        #   scales : scope-dependent; only PER_TENSOR and EXTERNAL are
        #            bounded by layout alone (see the storage-only
        #            estimator for the reasoning).
        #   header : <= 1 KB allowance for dtype tags / hashes / shape
        scope = self._codec.scale_scope
        scale_dtype = self._codec.scale_dtype
        if scope == ScaleScope.PER_TENSOR:
            scales_bytes = scale_dtype.itemsize
        elif scope == ScaleScope.EXTERNAL:
            scales_bytes = 0
        else:
            raise ValueError(
                f"AsymK16V8VOnlyMultiSerializer.estimate_serialized_size: "
                f"scale_scope {scope} requires per-head/per-page metadata "
                f"that MemoryLayoutDesc does not carry; only PER_TENSOR "
                f"and EXTERNAL are supported here"
            )
        v_bytes = sum(math.prod(s) for s in v_layout.shapes)
        header_allowance = 1024
        return header_allowance + v_bytes + scales_bytes


class AsymK16V8VOnlyMultiDeserializer(MultiDeserializer):
    """Decode a split-tier (V-only) asymmetric blob into ``V_out``.

    The blob carries V_fp8 + V_scales only; there is no K payload to
    return.  The K slot is the no-op slot — passing a tensor there
    is permitted but ignored (the slot is left untouched), which
    lets the caller use the same dst-group shape as the storage-only
    deserializer when convenient.

    Slot semantics (split-tier mode):

    * ``slot 0 = K_out`` — caller-provided MemoryObj, **left
      untouched** by this deserializer regardless of whether the
      slot is ``None`` or has a tensor set.  K must be supplied
      from a separate source (typically the host-resident K cache).
    * ``slot 1 = V_out`` — caller-provided MemoryObj at the model's
      native dtype, populated by dequantizing the stored FP8 V.
      ``None`` here is a contract error (fail-closed): V is the only
      payload this mode restores, so a None target is refused rather
      than silently skipped.
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_scope: ScaleScope = ScaleScope.PER_TENSOR,
        scale_dtype: torch.dtype = torch.float32,
    ) -> None:
        self._codec = AsymK16V8Codec(
            fp8_dtype=fp8_dtype,
            scale_scope=scale_scope,
            scale_dtype=scale_dtype,
        )

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE_V_ONLY

    def output_slot_mapping(self) -> "tuple[int | None, ...]":
        # Split-tier: K is sourced from L1 / host, not from this blob.
        # Slot 0 is always None on the dst tuple; slot 1 writes parent
        # group 1 (V).
        return (None, 1)

    def deserialize(self, src: MemoryObj, dst: MemoryObjGroup, key: ObjectKey) -> None:
        # ``key`` unused: this serde is content-agnostic.
        validate_group_size(dst, _GROUP_SIZE_V_ONLY, role="dst")
        if src.tensor is None:
            raise ValueError("AsymK16V8VOnlyMultiDeserializer: src.tensor is None")

        _k_obj, v_obj = dst
        if v_obj is None:
            # Fail closed: V is the only payload this mode restores.  A
            # None V target would silently drop it and still report
            # success, so refuse rather than no-op.
            raise ValueError(
                "AsymK16V8VOnlyMultiDeserializer: V dst slot is required; a "
                "None V target would silently drop the only payload this "
                "serde carries"
            )
        if v_obj.tensor is None:
            raise ValueError(
                "AsymK16V8VOnlyMultiDeserializer: non-None V dst slot "
                "must have a tensor set"
            )

        src_view = src.tensor.view(torch.uint8).contiguous()
        blob = src_view.numpy().tobytes()

        enc = self._codec.from_bytes(blob)
        if enc.scale_scheme != ScaleScheme.COMPUTED_PER_TENSOR:
            raise ValueError(
                "AsymK16V8VOnlyMultiDeserializer: blob scale_scheme="
                f"{enc.scale_scheme.name} is not COMPUTED_PER_TENSOR; a "
                "RAW_UNIT (byte-through) blob carries raw fp8 codes with no "
                "scales and must be decoded by "
                "AsymBytethroughK16V8VOnlyMultiDeserializer."
            )
        if enc.k_payload_len != 0:
            raise ValueError(
                "AsymK16V8VOnlyMultiDeserializer: blob has k_payload_len="
                f"{enc.k_payload_len} (>0); this is a storage-only-dequant "
                "blob.  Use AsymK16V8MultiDeserializer to decode it."
            )

        # Inline the V-only slice of the codec's decode path.  We do
        # NOT call ``self._codec.decode(enc)`` because that codepath
        # tries to materialize a K tensor via ``torch.frombuffer`` on
        # the empty K slice — which raises ``buffer length 0`` rather
        # than returning an empty tensor.
        target_v_dtype = v_obj.tensor.dtype
        v_off = enc.k_payload_len  # = 0 for split-tier blobs
        s_off = v_off + enc.v_payload_len
        v_bytes = enc.payload[v_off:s_off]
        s_bytes = enc.payload[s_off:]
        scale_tensor = (
            torch.frombuffer(s_bytes, dtype=enc.scale_dtype)
            .clone()
            .reshape(enc.scale_shape)
        )
        v_q = torch.frombuffer(v_bytes, dtype=enc.v_dtype).clone()
        if target_v_dtype == enc.v_dtype:
            v_dq_flat = v_q
        else:
            if not target_v_dtype.is_floating_point:
                raise ValueError(
                    "AsymK16V8VOnlyMultiDeserializer: target V dtype "
                    f"{target_v_dtype} is not floating-point"
                )
            v_dq_flat = dequantize_v_fp8(
                v_q,
                scale_tensor,
                enc.scale_scope,
                out_dtype=target_v_dtype,
            )

        target_shape = v_obj.tensor.shape
        if v_dq_flat.numel() != v_obj.tensor.numel():
            raise ValueError(
                f"AsymK16V8VOnlyMultiDeserializer: decoded V has "
                f"{v_dq_flat.numel()} elements, dst V shape "
                f"{tuple(target_shape)} expects {v_obj.tensor.numel()}"
            )
        v_obj.tensor.copy_(v_dq_flat.reshape(target_shape))


class AsymBytethroughK16V8VOnlyMultiSerializer(MultiSerializer):
    """Copy an already-FP8 V straight through, no quantization.

    This is the byte-through (Mode "C") counterpart to
    :class:`AsymK16V8VOnlyMultiSerializer`.  It exists for the vLLM
    live-asymmetric K16/V8 layout, where V is **already** FP8 e4m3 in
    HBM.  There is no full-precision V to compute scales from, so the
    raw e4m3 code bytes are copied byte-for-byte into the blob and no
    scale bytes are stored.  The result is a
    :class:`ScaleScheme.RAW_UNIT` :class:`EncodedKV`
    (``scale_payload_len == 0``, implicit unit scale) written at
    :class:`CodecVersion.V2`, so a scale-aware (V1-only) reader rejects
    it fail-closed instead of misreading the raw codes as a quantized
    payload.

    **Preconditions the caller MUST guarantee** — this serde operates
    on bytes and cannot see the attention layer's ``_v_scale`` scalar,
    so it cannot check them itself:

    * The producing layer's ``_v_scale`` (the external per-layer scalar
      applied symmetrically at store and read) is exactly ``1.0``.  The
      raw e4m3 codes only mean the same values in another process if
      the scale is unit (or the scalar travels out-of-band).  A
      non-unit scale makes a cross-engine reload silently wrong.
    * The restore engine's ``_v_scale`` is likewise ``1.0``.

    Slot semantics (identical to the scale-aware V-only serde):

    * ``slot 0 = K`` MUST be ``None`` (K stays in L1 / host).
    * ``slot 1 = V`` (required, dtype MUST be ``float8_e4m3fn``).
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        # The k_dtype tag is a NON-AUTHORITATIVE hint recorded in the
        # header.  It is NOT the source of truth for pairing this V blob
        # with its CPU-resident K -- that authority belongs to the
        # two-plane logical manifest (LO1), which records the paired K
        # dtype and is checked at re-pair time.  Restore/pairing code
        # MUST NOT gate on this tag.  Defaults to bfloat16 only so the
        # header has a well-formed value.
        k_dtype_tag: torch.dtype = torch.bfloat16,
    ) -> None:
        if fp8_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiSerializer: byte-through only "
                f"supports float8_e4m3fn (got {fp8_dtype}).  The live-asym "
                "K16/V8 layout stores e4m3; other fp8 variants would need a "
                "distinct on-disk tag and reader."
            )
        # The codec is used only for header (de)serialization here; its
        # scale_scope / scale_dtype are irrelevant because no scales are
        # ever computed or stored on the byte-through path.
        self._codec = AsymK16V8Codec(fp8_dtype=fp8_dtype)
        self._fp8_dtype = fp8_dtype
        self._k_dtype_tag = k_dtype_tag

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE_V_ONLY

    def input_slot_mapping(self) -> "tuple[int | None, ...]":
        # Split-tier: K stays in L1 / host -- never passed to this
        # serializer.  Slot 0 is always None; slot 1 reads parent
        # group 1 (V).
        return (None, 1)

    def serialize(self, src: MemoryObjGroup, dst: MemoryObj, key: ObjectKey) -> int:
        # ``key`` unused: this serde is content-agnostic.
        validate_group_size(src, _GROUP_SIZE_V_ONLY, role="src")
        k_obj, v_obj = src
        if k_obj is not None:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiSerializer (split-tier): K "
                "slot must be None.  K stays in host RAM in this mode and is "
                "not written to the byte buffer."
            )
        if v_obj is None or v_obj.tensor is None:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiSerializer: V slot is required "
                "and must have a tensor set"
            )
        if dst.tensor is None:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiSerializer: dst.tensor is None"
            )

        v = v_obj.tensor
        if v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiSerializer: V must already be "
                f"float8_e4m3fn (got {v.dtype}); this serde copies raw e4m3 "
                "codes byte-through and does NOT quantize.  Use "
                "AsymK16V8VOnlyMultiSerializer for a native-dtype V."
            )

        # Raw uint8 byte copy of the fp8 codes -- no numeric op, no
        # scales.  ``_tensor_to_bytes_fast`` reinterprets the storage as
        # uint8 and memcpys it out, so the e4m3 bit patterns are
        # preserved exactly.
        v_cpu = v.detach().to("cpu").contiguous()
        v_bytes = _tensor_to_bytes_fast(v_cpu)

        enc = EncodedKV(
            k_dtype=self._k_dtype_tag,
            v_dtype=self._fp8_dtype,
            # RAW_UNIT stores no scales.  Advertise scale_scope=NONE
            # (not PER_TENSOR) so nothing downstream branches on
            # PER_TENSOR and expects a scalar that was never written.
            # scale_scheme stays the authoritative field.
            scale_dtype=torch.float32,
            scale_scope=ScaleScope.NONE,
            scale_scheme=ScaleScheme.RAW_UNIT,
            k_payload_len=0,
            v_payload_len=len(v_bytes),
            scale_payload_len=0,
            scale_shape=(),
            payload=v_bytes,
        )
        blob = self._codec.to_bytes(enc)
        n = len(blob)
        if dst.tensor.numel() < n:
            raise ValueError(
                f"AsymBytethroughK16V8VOnlyMultiSerializer: dst capacity "
                f"{dst.tensor.numel()} below required {n}"
            )
        dst_view = dst.tensor.view(torch.uint8)
        dst_view[:n].copy_(torch.frombuffer(blob, dtype=torch.uint8))
        return n

    def estimate_serialized_size(
        self,
        layout_descs: LayoutDescGroup,
    ) -> int:
        validate_group_size(layout_descs, _GROUP_SIZE_V_ONLY, role="layout")
        k_layout, v_layout = layout_descs
        if k_layout is not None:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiSerializer."
                "estimate_serialized_size: K layout must be None — this mode "
                "does not write K bytes"
            )
        if v_layout is None:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiSerializer."
                "estimate_serialized_size: V layout is required"
            )

        # Bytes accounting: V (fp8) = 1 byte/elem; no scales (RAW_UNIT);
        # header <= 1 KB for dtype tags / hashes.
        v_bytes = sum(math.prod(s) for s in v_layout.shapes)
        header_allowance = 1024
        return header_allowance + v_bytes


class AsymBytethroughK16V8VOnlyMultiDeserializer(MultiDeserializer):
    """Restore raw FP8 V codes from a byte-through (RAW_UNIT) blob.

    The blob carries only raw e4m3 V codes (no scales, no K).  Restore
    is a straight byte copy back into an ``float8_e4m3fn`` destination;
    there is no dequantization, because the codes were never quantized
    away from a full-precision source in the first place.

    Restoring into a wider dtype (fp16/bf16) is refused: a RAW_UNIT
    blob carries no scale, so widening would require the caller's
    per-layer ``_v_scale`` (== 1.0 by precondition) and a kernel-side
    interpretation that this serde deliberately does not perform.  The
    consumer restores the raw codes into its fp8 HBM V cache and lets
    the attention kernel consume them, exactly as the producer did.

    Slot semantics (split-tier mode):

    * ``slot 0 = K_out`` — left untouched (K comes from L1 / host).
    * ``slot 1 = V_out`` — caller-provided ``float8_e4m3fn`` MemoryObj,
      populated bit-exact from the stored codes.  ``None`` is a
      deliberate skip.
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        if fp8_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiDeserializer: byte-through "
                f"only supports float8_e4m3fn (got {fp8_dtype})."
            )
        self._codec = AsymK16V8Codec(fp8_dtype=fp8_dtype)
        self._fp8_dtype = fp8_dtype

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE_V_ONLY

    def output_slot_mapping(self) -> "tuple[int | None, ...]":
        # Split-tier: K is sourced from L1 / host, not from this blob.
        return (None, 1)

    def deserialize(self, src: MemoryObj, dst: MemoryObjGroup, key: ObjectKey) -> None:
        # ``key`` unused: this serde is content-agnostic.
        validate_group_size(dst, _GROUP_SIZE_V_ONLY, role="dst")
        if src.tensor is None:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiDeserializer: src.tensor is None"
            )

        _k_obj, v_obj = dst
        if v_obj is None:
            # Fail closed: V is the ONLY payload this serde restores.  A
            # None V target means a broken load path would silently drop
            # V and still report success, so refuse rather than no-op.
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiDeserializer: V dst slot is "
                "required; a None V target would silently drop the only "
                "payload this serde carries"
            )
        if v_obj.tensor is None:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiDeserializer: non-None V dst "
                "slot must have a tensor set"
            )

        src_view = src.tensor.view(torch.uint8).contiguous()
        blob = src_view.numpy().tobytes()

        enc = self._codec.from_bytes(blob)
        # Strict RAW_UNIT invariants -- fail closed on anything that is
        # not a pure byte-through V blob.
        if enc.scale_scheme != ScaleScheme.RAW_UNIT:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiDeserializer: blob "
                f"scale_scheme={enc.scale_scheme.name} is not RAW_UNIT; a "
                "COMPUTED_PER_TENSOR blob must be decoded by "
                "AsymK16V8VOnlyMultiDeserializer."
            )
        if enc.scale_payload_len != 0:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiDeserializer: RAW_UNIT blob "
                f"must carry no scales; got scale_payload_len="
                f"{enc.scale_payload_len}"
            )
        if enc.k_payload_len != 0:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiDeserializer: V-only blob must "
                f"have k_payload_len=0; got {enc.k_payload_len}"
            )
        if enc.v_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiDeserializer: byte-through V "
                f"blob must be float8_e4m3fn; got v_dtype={enc.v_dtype}"
            )
        # Exact payload-length check: catch a schema-bit corruption of
        # v_payload_len that survives to here (the header parser slices
        # payload to the declared length, so a shrunk length would
        # otherwise silently truncate V).  CRC covers the payload bytes,
        # not this field, so check it explicitly.
        if enc.v_payload_len != len(enc.payload):
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiDeserializer: v_payload_len="
                f"{enc.v_payload_len} disagrees with payload byte-length "
                f"{len(enc.payload)}"
            )

        target_v_dtype = v_obj.tensor.dtype
        if target_v_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "AsymBytethroughK16V8VOnlyMultiDeserializer: restore target "
                f"must be float8_e4m3fn (raw code restore, no dequant); got "
                f"dst V dtype {target_v_dtype}.  Restoring into a wider dtype "
                "would require a scale, which a RAW_UNIT blob does not carry."
            )

        v_bytes = enc.payload[: enc.v_payload_len]
        v_codes = torch.frombuffer(v_bytes, dtype=torch.float8_e4m3fn).clone()
        target_shape = v_obj.tensor.shape
        if v_codes.numel() != v_obj.tensor.numel():
            raise ValueError(
                f"AsymBytethroughK16V8VOnlyMultiDeserializer: decoded V has "
                f"{v_codes.numel()} elements, dst V shape "
                f"{tuple(target_shape)} expects {v_obj.tensor.numel()}"
            )
        v_obj.tensor.copy_(v_codes.reshape(target_shape))


class AsymBytethroughK16V8MultiSerializer(MultiSerializer):
    """Encode BOTH K (bf16) and V (fp8) byte-through into ONE blob.

    This is the KV_TOGETHER counterpart of
    :class:`AsymBytethroughK16V8VOnlyMultiSerializer`.  The V-only
    byte-through serde returns ``(None, 1)`` and so forces
    :attr:`StoragePlacementMode.KV_SPLIT_TIER` — K is kept in L1 (host
    RAM) and never written to L2, which makes the L2 object unusable by
    any other process (a fresh process has no K bytes).  This serde
    returns the identity mapping ``(0, 1)`` so both planes are written
    into a single self-contained object that resolves to
    :attr:`StoragePlacementMode.KV_TOGETHER`.  That object is durable and
    reusable across processes / restarts via the normal L2 path, with no
    split-tier manifest involved.

    Both planes are copied **raw** — no numeric op, no scales:

    * ``slot 0 = K`` (required): the already-bf16 (or other non-fp8
      native) K codes, stored byte-through.  Its dtype is written
      authoritatively into the header (unlike the V-only mode's
      non-authoritative ``k_dtype`` tag, because here K *is* the payload).
    * ``slot 1 = V`` (required, dtype MUST be ``float8_e4m3fn``): the raw
      e4m3 V codes, stored byte-through exactly as the V-only serde does.

    The blob is a :class:`ScaleScheme.RAW_UNIT` / :class:`CodecVersion.V2`
    object, mutually unreadable by the scale-aware (COMPUTED) decoders.

    **Role-aware, not slot-position:** a K/V order inversion (V handed to
    the K slot) is rejected by the dtype checks — K must not be fp8 and V
    must be fp8 — so a swap fails closed rather than corrupting decode.

    **Precondition the caller MUST guarantee** (this serde sees only
    bytes): the producing and restoring layers' per-layer ``_v_scale``
    scalar is exactly ``1.0``.  A non-unit scale makes a cross-engine
    reload of the raw e4m3 codes silently wrong.
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        if fp8_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "AsymBytethroughK16V8MultiSerializer: byte-through only "
                f"supports float8_e4m3fn (got {fp8_dtype})."
            )
        # The codec is used only for header (de)serialization; no scales
        # are ever computed on the byte-through path.
        self._codec = AsymK16V8Codec(fp8_dtype=fp8_dtype)
        self._fp8_dtype = fp8_dtype

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE_STORAGE_ONLY

    def input_slot_mapping(self) -> "tuple[int | None, ...]":
        # KV_TOGETHER: both planes are written.  Slot 0 = parent group 0
        # (K), slot 1 = parent group 1 (V).  All-non-None -> KV_TOGETHER.
        return (0, 1)

    def serialize(self, src: MemoryObjGroup, dst: MemoryObj, key: ObjectKey) -> int:
        # ``key`` unused: this serde is content-agnostic.
        validate_group_size(src, _GROUP_SIZE_STORAGE_ONLY, role="src")
        k_obj, v_obj = src
        if k_obj is None or k_obj.tensor is None:
            raise ValueError(
                "AsymBytethroughK16V8MultiSerializer: K slot is required and "
                "must have a tensor set (this is the KV_TOGETHER both-plane "
                "mode; use AsymBytethroughK16V8VOnlyMultiSerializer for the "
                "split-tier V-only mode)"
            )
        if v_obj is None or v_obj.tensor is None:
            raise ValueError(
                "AsymBytethroughK16V8MultiSerializer: V slot is required and "
                "must have a tensor set"
            )
        if dst.tensor is None:
            raise ValueError(
                "AsymBytethroughK16V8MultiSerializer: dst.tensor is None"
            )

        k = k_obj.tensor
        v = v_obj.tensor
        # Role-aware fail-closed: a K/V order inversion presents an fp8 K
        # and/or a non-fp8 V.  Reject both so a swap can never silently
        # reinterpret bytes at the wrong dtype on restore.
        if _is_fp8(k.dtype):
            raise ValueError(
                "AsymBytethroughK16V8MultiSerializer: K slot dtype is fp8 "
                f"({k.dtype}); K must be a native (non-fp8) dtype.  This looks "
                "like a K/V slot inversion — refusing to store."
            )
        if v.dtype != self._fp8_dtype:
            raise ValueError(
                "AsymBytethroughK16V8MultiSerializer: V slot must already be "
                f"{self._fp8_dtype} (got {v.dtype}); this serde copies raw "
                "e4m3 codes byte-through and does NOT quantize."
            )

        # Raw uint8 byte copies of both planes -- no numeric op.
        k_cpu = k.detach().to("cpu").contiguous()
        v_cpu = v.detach().to("cpu").contiguous()
        k_bytes = _tensor_to_bytes_fast(k_cpu)
        v_bytes = _tensor_to_bytes_fast(v_cpu)

        enc = EncodedKV(
            # K dtype is AUTHORITATIVE here (K is written), so record the
            # real tensor dtype, not a placeholder tag.
            k_dtype=k.dtype,
            v_dtype=self._fp8_dtype,
            scale_dtype=torch.float32,
            scale_scope=ScaleScope.NONE,
            scale_scheme=ScaleScheme.RAW_UNIT,
            k_payload_len=len(k_bytes),
            v_payload_len=len(v_bytes),
            scale_payload_len=0,
            scale_shape=(),
            payload=k_bytes + v_bytes,
        )
        blob = self._codec.to_bytes(enc)
        n = len(blob)
        if dst.tensor.numel() < n:
            raise ValueError(
                f"AsymBytethroughK16V8MultiSerializer: dst capacity "
                f"{dst.tensor.numel()} below required {n}"
            )
        dst_view = dst.tensor.view(torch.uint8)
        dst_view[:n].copy_(torch.frombuffer(blob, dtype=torch.uint8))
        return n

    def estimate_serialized_size(
        self,
        layout_descs: LayoutDescGroup,
    ) -> int:
        validate_group_size(layout_descs, _GROUP_SIZE_STORAGE_ONLY, role="layout")
        k_layout, v_layout = layout_descs
        if k_layout is None or v_layout is None:
            raise ValueError(
                "AsymBytethroughK16V8MultiSerializer.estimate_serialized_size: "
                "both K and V layouts are required (KV_TOGETHER both-plane mode)"
            )
        # Bytes accounting: K (native) = itemsize/elem; V (fp8) = 1 byte/elem;
        # no scales (RAW_UNIT); header <= 1 KB.
        k_bytes = sum(
            math.prod(s) * d.itemsize
            for s, d in zip(k_layout.shapes, k_layout.dtypes, strict=True)
        )
        v_bytes = sum(math.prod(s) for s in v_layout.shapes)
        header_allowance = 1024
        return header_allowance + k_bytes + v_bytes


class AsymBytethroughK16V8MultiDeserializer(MultiDeserializer):
    """Restore raw K (bf16) + V (fp8) codes from a both-plane RAW_UNIT blob.

    KV_TOGETHER counterpart of
    :class:`AsymBytethroughK16V8VOnlyMultiDeserializer`.  The blob is
    self-contained (both K and V codes present), so a fresh process with
    no prior L1 state can restore the full KV directly from L2 — this is
    what makes the offload cross-process / restart reusable.

    Slot semantics:

    * ``slot 0 = K_out`` (required): restored byte-identically into a
      MemoryObj whose dtype MUST equal the header's stored K dtype.
    * ``slot 1 = V_out`` (required, ``float8_e4m3fn``): restored
      byte-identically from the raw e4m3 codes.

    Both restores are straight byte reinterprets; there is no dequant
    (nothing was quantized on this path).
    """

    def __init__(
        self,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        if fp8_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: byte-through only "
                f"supports float8_e4m3fn (got {fp8_dtype})."
            )
        self._codec = AsymK16V8Codec(fp8_dtype=fp8_dtype)
        self._fp8_dtype = fp8_dtype

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE_STORAGE_ONLY

    def output_slot_mapping(self) -> "tuple[int | None, ...]":
        # KV_TOGETHER: both planes are restored from the blob.
        return (0, 1)

    def deserialize(self, src: MemoryObj, dst: MemoryObjGroup, key: ObjectKey) -> None:
        # ``key`` unused: this serde is content-agnostic.
        validate_group_size(dst, _GROUP_SIZE_STORAGE_ONLY, role="dst")
        if src.tensor is None:
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: src.tensor is None"
            )
        k_obj, v_obj = dst
        if k_obj is None or k_obj.tensor is None:
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: K dst slot is required "
                "and must have a tensor set (both-plane KV_TOGETHER restore)"
            )
        if v_obj is None or v_obj.tensor is None:
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: V dst slot is required "
                "and must have a tensor set"
            )

        src_view = src.tensor.view(torch.uint8).contiguous()
        blob = src_view.numpy().tobytes()
        enc = self._codec.from_bytes(blob)

        # Strict RAW_UNIT invariants -- fail closed on anything that is not
        # a pure both-plane byte-through blob.
        if enc.scale_scheme != ScaleScheme.RAW_UNIT:
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: blob scale_scheme="
                f"{enc.scale_scheme.name} is not RAW_UNIT; a COMPUTED blob "
                "must be decoded by AsymK16V8MultiDeserializer."
            )
        if enc.scale_payload_len != 0:
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: RAW_UNIT blob must "
                f"carry no scales; got scale_payload_len={enc.scale_payload_len}"
            )
        if enc.k_payload_len == 0:
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: both-plane blob must "
                "carry K bytes (k_payload_len>0); a k_payload_len=0 blob is the "
                "V-only split-tier form — decode it with "
                "AsymBytethroughK16V8VOnlyMultiDeserializer."
            )
        if enc.v_dtype != self._fp8_dtype:
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: byte-through V blob "
                f"must be {self._fp8_dtype}; got v_dtype={enc.v_dtype}"
            )
        # Exact payload-length check: the CRC covers payload bytes, not the
        # length fields, so a corrupted k/v_payload_len that still parses
        # would otherwise mis-slice K vs V.  Verify K+V == payload length.
        if enc.k_payload_len + enc.v_payload_len != len(enc.payload):
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: k_payload_len="
                f"{enc.k_payload_len} + v_payload_len={enc.v_payload_len} "
                f"disagrees with payload byte-length {len(enc.payload)}"
            )

        # K restore: byte reinterpret at the stored dtype.  Require the dst
        # K dtype to match the stored dtype exactly -- a raw byte copy is
        # only meaningful at the same dtype, and a mismatch would silently
        # reinterpret the bytes.
        target_k_dtype = k_obj.tensor.dtype
        if target_k_dtype != enc.k_dtype:
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: dst K dtype "
                f"{target_k_dtype} != stored K dtype {enc.k_dtype}; a "
                "byte-through restore requires an identical dtype."
            )
        if _is_fp8(target_k_dtype):
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: dst K dtype is fp8 "
                f"({target_k_dtype}); K must be a native (non-fp8) dtype — this "
                "looks like a K/V slot inversion."
            )
        target_v_dtype = v_obj.tensor.dtype
        if target_v_dtype != self._fp8_dtype:
            raise ValueError(
                "AsymBytethroughK16V8MultiDeserializer: dst V dtype "
                f"{target_v_dtype} must be {self._fp8_dtype} (raw code restore, "
                "no dequant)."
            )

        k_off = 0
        v_off = enc.k_payload_len
        k_bytes = enc.payload[k_off:v_off]
        v_bytes = enc.payload[v_off:]

        k_codes = torch.frombuffer(k_bytes, dtype=enc.k_dtype).clone()
        if k_codes.numel() != k_obj.tensor.numel():
            raise ValueError(
                f"AsymBytethroughK16V8MultiDeserializer: decoded K has "
                f"{k_codes.numel()} elements, dst K shape "
                f"{tuple(k_obj.tensor.shape)} expects {k_obj.tensor.numel()}"
            )
        k_obj.tensor.copy_(k_codes.reshape(k_obj.tensor.shape))

        v_codes = torch.frombuffer(v_bytes, dtype=self._fp8_dtype).clone()
        if v_codes.numel() != v_obj.tensor.numel():
            raise ValueError(
                f"AsymBytethroughK16V8MultiDeserializer: decoded V has "
                f"{v_codes.numel()} elements, dst V shape "
                f"{tuple(v_obj.tensor.shape)} expects {v_obj.tensor.numel()}"
            )
        v_obj.tensor.copy_(v_codes.reshape(v_obj.tensor.shape))


# ============================================================================
# Factory registration (selectable from YAML via serde_config.type)
# ============================================================================
#
# AsyncSerdeProcessor is typed for the single-tensor Serializer /
# Deserializer ABCs, but at runtime its dispatch is duck-typed -- it
# forwards each work item to ``serialize`` / ``deserialize`` unchanged.
# Multi-output serdes plug in via the same processor with ``# type:
# ignore[arg-type]``; the SerdeL2AdapterWrapper detects the
# MultiSerializer / MultiDeserializer at dispatch time and builds
# MemoryObjGroup views over the parent grouped MemoryObj using
# ``input_slot_mapping`` / ``output_slot_mapping``.

# First Party
# Late imports (factory registration): kept below module body to avoid a
# circular import via serde/__init__, which imports this module for its
# registration side effect.
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor  # noqa: E402
from lmcache.v1.distributed.serde.base import SerdeProcessor  # noqa: E402
from lmcache.v1.distributed.serde.factory import register_serde_factory  # noqa: E402


def _resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {name!r}")
    return dtype


def _resolve_scale_scope(name: str) -> ScaleScope:
    try:
        return ScaleScope[name]
    except KeyError as e:
        valid = ", ".join(s.name for s in ScaleScope)
        raise ValueError(f"Unknown ScaleScope {name!r}. Valid: {valid}") from e


def _create_asym_k16_v8_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    """Factory for the storage-only (Mode 1) asym K16/V8 serde.

    Accepted ``kwargs``:

    * ``fp8_dtype`` (str, default ``"float8_e4m3fn"``): torch fp8
      dtype used for V quantization.
    * ``scale_scope`` (str, default ``"PER_TENSOR"``): name of a
      :class:`ScaleScope` member; only ``PER_TENSOR`` and
      ``EXTERNAL`` are supported through ``estimate_serialized_size``
      today.
    * ``scale_dtype`` (str, default ``"float32"``): torch dtype for
      the per-scope scale tensor.
    * ``max_workers`` (int, default ``4``): thread-pool size for the
      drainer-side codec.  The asym K16/V8 codec is CPU-bound.  With a
      single worker the encode and decode run inline on the store path
      and make stores several times slower than storing without a serde
      at single-producer load.  Default to four workers so the codec
      runs off the critical path at realistic multi-producer rates and
      store latency matches the no-serde path, without oversubscribing
      cores.  Raise it on hosts with spare CPU; beyond that the
      producer, not the codec, bounds throughput.

    Returns an :class:`AsyncSerdeProcessor` wrapping the multi-output
    storage-only K16/V8 pair.  The SerdeL2AdapterWrapper consumes the
    processor's ``MultiSerializer`` / ``MultiDeserializer`` shape via
    its ``input_slot_mapping`` / ``output_slot_mapping`` hooks (the
    storage-only mode uses the identity mapping ``(0, 1)``).
    """
    fp8_dtype = _resolve_dtype(str(kwargs.get("fp8_dtype", "float8_e4m3fn")))
    scale_scope = _resolve_scale_scope(str(kwargs.get("scale_scope", "PER_TENSOR")))
    scale_dtype = _resolve_dtype(str(kwargs.get("scale_dtype", "float32")))
    max_workers = int(kwargs.get("max_workers", 4))  # type: ignore[call-overload]
    return AsyncSerdeProcessor(
        AsymK16V8MultiSerializer(  # type: ignore[arg-type]
            fp8_dtype=fp8_dtype,
            scale_scope=scale_scope,
            scale_dtype=scale_dtype,
        ),
        AsymK16V8MultiDeserializer(  # type: ignore[arg-type]
            fp8_dtype=fp8_dtype,
            scale_scope=scale_scope,
            scale_dtype=scale_dtype,
        ),
        max_workers=max_workers,
    )


def _create_asym_k16_v8_v_only_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    """Factory for the V-only split-tier (Mode 2) asym K16/V8 serde.

    Accepted ``kwargs``: same as :func:`_create_asym_k16_v8_serde`,
    plus:

    * ``k_dtype_tag`` (str, default ``"bfloat16"``): torch dtype
      recorded in the header so cross-config gating works on
      restoration (K itself is never written to the byte buffer in
      this mode; the tag identifies what K's dtype would have been).

    Returns an :class:`AsyncSerdeProcessor` wrapping the V-only
    multi-output pair.  ``input_slot_mapping`` returns ``(None, 1)``
    so the wrapper passes no K to the serializer.  Note that
    full split-tier placement (routing K to L1 and V to L2 as
    separate typed child outputs) requires additional wrapper
    work tracked separately; this factory only enables the V-only
    codec path -- with the current wrapper, V is encoded and written
    to L2 as a single blob, and K is expected to be sourced from L1
    by the caller.
    """
    fp8_dtype = _resolve_dtype(str(kwargs.get("fp8_dtype", "float8_e4m3fn")))
    scale_scope = _resolve_scale_scope(str(kwargs.get("scale_scope", "PER_TENSOR")))
    scale_dtype = _resolve_dtype(str(kwargs.get("scale_dtype", "float32")))
    k_dtype_tag = _resolve_dtype(str(kwargs.get("k_dtype_tag", "bfloat16")))
    # See _create_asym_k16_v8_serde for the max_workers=4 rationale
    # (CPU-bound codec; a single worker runs it inline on the store
    # path and is markedly slower than the no-serde path).
    max_workers = int(kwargs.get("max_workers", 4))  # type: ignore[call-overload]
    return AsyncSerdeProcessor(
        AsymK16V8VOnlyMultiSerializer(  # type: ignore[arg-type]
            fp8_dtype=fp8_dtype,
            scale_scope=scale_scope,
            scale_dtype=scale_dtype,
            k_dtype_tag=k_dtype_tag,
        ),
        AsymK16V8VOnlyMultiDeserializer(  # type: ignore[arg-type]
            fp8_dtype=fp8_dtype,
            scale_scope=scale_scope,
            scale_dtype=scale_dtype,
        ),
        max_workers=max_workers,
    )


def _create_asym_bytethrough_k16_v8_v_only_serde(
    kwargs: dict[str, object],
) -> SerdeProcessor:
    """Factory for the byte-through (Mode C) V-only asym K16/V8 serde.

    For the vLLM live-asymmetric K16/V8 layout where V is already FP8
    e4m3 in HBM.  The raw e4m3 codes are copied byte-through with no
    scales; the blob is a :class:`ScaleScheme.RAW_UNIT` /
    :class:`CodecVersion.V2` object.

    Accepted ``kwargs``:

    * ``fp8_dtype`` (str, default ``"float8_e4m3fn"``): the only
      supported value; any other raises.
    * ``k_dtype_tag`` (str, default ``"bfloat16"``): torch dtype
      recorded in the header for cross-config gating on restore (K
      itself is never written in this mode).
    * ``max_workers`` (int, default ``4``): drainer thread-pool size.
      Byte-through does no quant, so its per-item cost is far lower
      than the scale-aware serde; 4 is kept for parity and to hide the
      raw memcpy + L2 write under the async drainer.

    Returns an :class:`AsyncSerdeProcessor` wrapping the byte-through
    V-only pair.  ``input_slot_mapping`` returns ``(None, 1)`` so the
    wrapper passes no K to the serializer.
    """
    fp8_dtype = _resolve_dtype(str(kwargs.get("fp8_dtype", "float8_e4m3fn")))
    k_dtype_tag = _resolve_dtype(str(kwargs.get("k_dtype_tag", "bfloat16")))
    max_workers = int(kwargs.get("max_workers", 4))  # type: ignore[call-overload]
    return AsyncSerdeProcessor(
        AsymBytethroughK16V8VOnlyMultiSerializer(  # type: ignore[arg-type]
            fp8_dtype=fp8_dtype,
            k_dtype_tag=k_dtype_tag,
        ),
        AsymBytethroughK16V8VOnlyMultiDeserializer(  # type: ignore[arg-type]
            fp8_dtype=fp8_dtype,
        ),
        max_workers=max_workers,
    )


def _create_asym_bytethrough_k16_v8_serde(
    kwargs: dict[str, object],
) -> SerdeProcessor:
    """Factory for the both-plane byte-through asym K16/V8 serde.

    The KV_TOGETHER counterpart of the V-only byte-through serde: it
    writes BOTH K (native, byte-through) and V (raw fp8 e4m3,
    byte-through) into a single self-contained
    :class:`ScaleScheme.RAW_UNIT` / :class:`CodecVersion.V2` object.
    Because ``input_slot_mapping`` returns the identity ``(0, 1)`` (both
    slots present), :func:`derive_storage_placement_mode` resolves this
    to :attr:`StoragePlacementMode.KV_TOGETHER` — the object is durable
    and reusable across processes / restarts via the normal L2 path, with
    no split-tier manifest.  Use this (not the ``_v_only`` variant) when
    cross-process / restart reuse of the byte-through offload is required.

    Accepted ``kwargs``:

    * ``fp8_dtype`` (str, default ``"float8_e4m3fn"``): the only
      supported value; any other raises.
    * ``max_workers`` (int, default ``4``): drainer thread-pool size.

    The producing and restoring layers' per-layer ``_v_scale`` MUST be
    ``1.0`` (the serde sees only bytes and cannot check it).
    """
    fp8_dtype = _resolve_dtype(str(kwargs.get("fp8_dtype", "float8_e4m3fn")))
    max_workers = int(kwargs.get("max_workers", 4))  # type: ignore[call-overload]
    return AsyncSerdeProcessor(
        AsymBytethroughK16V8MultiSerializer(  # type: ignore[arg-type]
            fp8_dtype=fp8_dtype,
        ),
        AsymBytethroughK16V8MultiDeserializer(  # type: ignore[arg-type]
            fp8_dtype=fp8_dtype,
        ),
        max_workers=max_workers,
    )


register_serde_factory("asym_k16_v8", _create_asym_k16_v8_serde)
register_serde_factory("asym_k16_v8_v_only", _create_asym_k16_v8_v_only_serde)
register_serde_factory(
    "asym_bytethrough_k16_v8_v_only",
    _create_asym_bytethrough_k16_v8_v_only_serde,
)
register_serde_factory(
    "asym_bytethrough_k16_v8",
    _create_asym_bytethrough_k16_v8_serde,
)
