# SPDX-License-Identifier: Apache-2.0
"""
Asymmetric K16/V8 multi-output serde.

Concrete :class:`MultiSerializer` / :class:`MultiDeserializer` pair
that bridges the existing pure-PyTorch :class:`AsymK16V8Codec` (in
``lmcache/v1/kv_codec``) into the tuple-shaped serde contract from
``multi.py``. This is the first concrete consumer of the multi-output
API; the symmetric ``fp8`` serde in ``serde/fp8.py`` keeps using the
single-tensor API and does not change.

Mode covered by this commit
---------------------------

**Storage-only-dequant.** Group of size 2 on both endpoints:

* serialize input ``src = (K, V)`` — both at the model's native dtype
  (fp16 / bf16). The codec quantizes V to FP8 internally.
* deserialize output ``dst = (K_out, V_out)`` — both back at the
  model's native dtype, with V dequantized from the stored FP8.

This is the path that maps cleanly onto the upstream
``SerdeL2AdapterWrapper`` from PR #3140 once the async-side wiring
for multi-output dst lands; downstream consumers that simply want
"smaller bytes on disk for the same fp16 KV view" use this mode.

Modes deferred to follow-ups
----------------------------

* **Native-asym.** Returns the triple ``(K_native, V_fp8, V_scales)``
  on deserialize so a vLLM asymmetric paged cache can consume V
  without re-expanding it. Requires group_size=3 with mixed dtypes
  and a downstream consumer that knows how to populate
  ``layer._v_scale``. Out of scope here; will land alongside the
  multi-output ``AsyncMultiSerdeProcessor`` analog.
* **V-only writes.** ``serialize`` input ``(None, V)`` for the case
  where K is already resident in HBM. Requires the upstream wrapper
  to express that one slot is absent, which the current
  ``SerdeL2AdapterWrapper.submit_*`` API does not yet do.
"""

# Future
from __future__ import annotations

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.multi import (
    LayoutDescGroup,
    MemoryObjGroup,
    MultiDeserializer,
    MultiSerializer,
    validate_group_size,
)
from lmcache.v1.kv_codec import AsymK16V8Codec, EncodedKV, ScaleScope
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


class AsymK16V8MultiSerializer(MultiSerializer):
    """Encode ``(K, V)`` into a single asymmetric K16/V8 byte blob.

    Slot semantics:

    * ``slot 0 = K`` (required, fp16 or bf16, native model dtype).
    * ``slot 1 = V`` (required, fp16 or bf16, native model dtype).

    ``None`` is not admitted in either slot for storage-only-dequant
    mode; use the native-asym variant (future commit) if K is already
    resident and only V should ship.

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

    def serialize(self, src: MemoryObjGroup, dst: MemoryObj) -> int:
        validate_group_size(src, _GROUP_SIZE_STORAGE_ONLY, role="src")
        k_obj, v_obj = src
        if k_obj is None or v_obj is None:
            raise ValueError(
                "AsymK16V8MultiSerializer (storage-only-dequant): both K "
                "and V must be provided; None slots are reserved for the "
                "native-asym / V-only-writes follow-ups"
            )
        if k_obj.tensor is None or v_obj.tensor is None:
            raise ValueError(
                "AsymK16V8MultiSerializer: src MemoryObjs must have tensors set"
            )
        if dst.tensor is None:
            raise ValueError("AsymK16V8MultiSerializer: dst.tensor is None")

        enc = self._codec.encode(k_obj.tensor, v_obj.tensor)
        blob = self._codec.to_bytes(enc)
        n = len(blob)
        if dst.tensor.numel() < n:
            raise ValueError(
                f"AsymK16V8MultiSerializer: dst capacity {dst.tensor.numel()} "
                f"below required {n}"
            )
        dst_view = dst.tensor.view(torch.uint8)
        dst_view[:n].copy_(torch.frombuffer(bytearray(blob), dtype=torch.uint8))
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
        #   K: numel(K) * itemsize(K_dtype)
        #   V: numel(V) * 1                 (FP8 e4m3 = 1 byte/elem)
        #   scales: per-tensor = 1 elem * itemsize(scale_dtype) = 4 bytes
        #   header: <= 256 bytes for shape+dtype tags+hashes
        #
        # We add a generous 1 KB header allowance plus per-page scale
        # margin so PER_PAGE_HEAD scopes (introduced when the codec
        # gets configured for them) still fit. The tight bound is
        # reported in tests via the encode_blob round-trip.
        def _numel(layout: MemoryLayoutDesc) -> int:
            n = 0
            for shape in layout.shapes:
                m = 1
                for dim in shape:
                    m *= int(dim)
                n += m
            return n

        k_bytes = sum(
            _numel(MemoryLayoutDesc(shapes=[s], dtypes=[d])) * d.itemsize
            for s, d in zip(k_layout.shapes, k_layout.dtypes, strict=True)
        )
        v_bytes = sum(
            _numel(MemoryLayoutDesc(shapes=[s], dtypes=[d])) * 1
            for s, d in zip(v_layout.shapes, v_layout.dtypes, strict=True)
        )
        per_layer_scale_bytes = 4
        n_layer_groups = len(v_layout.shapes)
        scales_bytes = per_layer_scale_bytes * max(n_layer_groups, 1)
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

    def deserialize(self, src: MemoryObj, dst: MemoryObjGroup) -> None:
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
        k_flat, v_dq_flat, _scales = self._codec.decode(enc, out_v_dtype=target_v_dtype)

        if k_obj is not None:
            if k_obj.tensor is None:
                raise ValueError(
                    "AsymK16V8MultiDeserializer: non-None K dst slot "
                    "must have a tensor set"
                )
            target_shape = k_obj.tensor.shape
            if k_flat.numel() != int(torch.tensor(list(target_shape)).prod().item()):
                raise ValueError(
                    f"AsymK16V8MultiDeserializer: decoded K has "
                    f"{k_flat.numel()} elements, dst K shape "
                    f"{tuple(target_shape)} expects "
                    f"{int(torch.tensor(list(target_shape)).prod().item())}"
                )
            k_dec = k_flat.reshape(target_shape)
            if k_obj.tensor.dtype != k_dec.dtype:
                k_dec = k_dec.to(k_obj.tensor.dtype)
            k_obj.tensor.copy_(k_dec)

        if v_obj is not None:
            if v_obj.tensor is None:
                raise ValueError(
                    "AsymK16V8MultiDeserializer: non-None V dst slot "
                    "must have a tensor set"
                )
            target_shape = v_obj.tensor.shape
            if v_dq_flat.numel() != int(torch.tensor(list(target_shape)).prod().item()):
                raise ValueError(
                    f"AsymK16V8MultiDeserializer: decoded V has "
                    f"{v_dq_flat.numel()} elements, dst V shape "
                    f"{tuple(target_shape)} expects "
                    f"{int(torch.tensor(list(target_shape)).prod().item())}"
                )
            v_dec = v_dq_flat.reshape(target_shape)
            if v_obj.tensor.dtype != v_dec.dtype:
                v_dec = v_dec.to(v_obj.tensor.dtype)
            v_obj.tensor.copy_(v_dec)


class AsymK16V8VOnlyMultiSerializer(MultiSerializer):
    """Encode V-only into a split-tier asymmetric byte blob.

    This is the **Mode 2 / split-tier** path: K stays in CPU-pinned
    host memory and is not written to the byte buffer at all; only
    the FP8-quantized V plus its per-tensor scales hit the durable
    storage tier.  The blob is a regular :class:`EncodedKV` with
    ``k_payload_len = 0`` and the ``k_dtype`` tag set to whatever
    dtype K would have been (so cross-config gating still works on
    the eventual restore).

    Slot semantics:

    * ``slot 0 = K`` MUST be ``None``.  Passing a tensor here is a
      contract error: this serde does not write K bytes by design.
      Use :class:`AsymK16V8MultiSerializer` instead if you want a
      self-contained restorable object.
    * ``slot 1 = V`` (required, fp16 or bf16, native model dtype).

    Byte ratio vs FP16 KV (paper Eq.~4): ``V_8 / (K_16 + V_16) =
    1/4``, equivalently ``1/3`` of the all-NVMe Mode 1 object.
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

    def serialize(self, src: MemoryObjGroup, dst: MemoryObj) -> int:
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
        dst_view[:n].copy_(torch.frombuffer(bytearray(blob), dtype=torch.uint8))
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
        #   V (fp8): numel(V) * 1
        #   scales : numel(scales) * itemsize(scale_dtype); for
        #            PER_TENSOR scope this is exactly itemsize bytes.
        #            We allow up to 4 KB of scales for richer scopes.
        #   header : <= 1 KB allowance for dtype tags / hashes / shape
        v_bytes = 0
        for s, _d in zip(v_layout.shapes, v_layout.dtypes, strict=True):
            n = 1
            for dim in s:
                n *= int(dim)
            v_bytes += n * 1  # 1 byte/elem after fp8 cast
        header_allowance = 1024
        scales_allowance = 4096
        return header_allowance + v_bytes + scales_allowance


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
      ``None`` here is a deliberate skip (a no-op call).
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

    def deserialize(self, src: MemoryObj, dst: MemoryObjGroup) -> None:
        validate_group_size(dst, _GROUP_SIZE_V_ONLY, role="dst")
        if src.tensor is None:
            raise ValueError("AsymK16V8VOnlyMultiDeserializer: src.tensor is None")

        _k_obj, v_obj = dst
        if v_obj is None:
            # Nothing to materialize.  The K slot is a no-op for this
            # mode regardless, so a (None, None) call is a deliberate
            # skip rather than an error.
            return
        if v_obj.tensor is None:
            raise ValueError(
                "AsymK16V8VOnlyMultiDeserializer: non-None V dst slot "
                "must have a tensor set"
            )

        src_view = src.tensor.view(torch.uint8).contiguous()
        blob = src_view.numpy().tobytes()

        enc = self._codec.from_bytes(blob)
        if enc.k_payload_len != 0:
            raise ValueError(
                "AsymK16V8VOnlyMultiDeserializer: blob has k_payload_len="
                f"{enc.k_payload_len} (>0); this is a Mode 1 / "
                "storage-only-dequant blob.  Use AsymK16V8MultiDeserializer "
                "to decode it."
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
            torch.frombuffer(bytearray(s_bytes), dtype=enc.scale_dtype)
            .clone()
            .reshape(enc.scale_shape)
        )
        v_q = torch.frombuffer(bytearray(v_bytes), dtype=enc.v_dtype).clone()
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
        expected = int(torch.tensor(list(target_shape)).prod().item())
        if v_dq_flat.numel() != expected:
            raise ValueError(
                f"AsymK16V8VOnlyMultiDeserializer: decoded V has "
                f"{v_dq_flat.numel()} elements, dst V shape "
                f"{tuple(target_shape)} expects {expected}"
            )
        v_dec = v_dq_flat.reshape(target_shape)
        if v_obj.tensor.dtype != v_dec.dtype:
            v_dec = v_dec.to(v_obj.tensor.dtype)
        v_obj.tensor.copy_(v_dec)
