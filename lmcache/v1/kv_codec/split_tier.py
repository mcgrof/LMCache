# SPDX-License-Identifier: Apache-2.0
"""Split-tier placement: K-hot (CPU pinned) / V-cold (NVMe).

This module implements the headline result from the LMCache
asymmetric plan: keys stay in CPU pinned memory at full FP16/BF16
precision (fast tier, serial-critical for attention scores), values
live as FP8 on NVMe (cold tier, bandwidth-tolerant aggregator).

Cache-hit restore traffic from NVMe drops from `K16+V16 = 4 B` per
element pair (FP16 baseline) to `V8 = 1 B` per element pair —
4x reduction in cold-tier read traffic — while K bytes flow
CPU->GPU at roughly an order of magnitude more bandwidth than
NVMe.

The split-tier layer sits ABOVE the codec.  A logical KV chunk is
stored as three physical objects:

    cache_key/layer_<L>/chunk_<C>/K.bin     # FP16/BF16, fast tier
    cache_key/layer_<L>/chunk_<C>/V.fp8     # FP8 e4m3, cold tier
    cache_key/layer_<L>/chunk_<C>/meta.json # codec header, scales

This module owns the path layout, the read-side concurrency
contract, and the byte-accounting that distinguishes "NVMe bytes"
from "CPU bytes" in benchmarks.
"""

# Standard
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple

# Third Party
import torch

# First Party
from lmcache.v1.kv_codec.asym_k16_v8 import AsymK16V8Codec
from lmcache.v1.kv_codec.encoded_kv import (
    CodecHashes,
    EncodedKV,
    ScaleScope,
    deserialize_header,
    serialize_header,
)
from lmcache.v1.kv_codec.errors import (
    CodecError,
    CorruptEncodedKVError,
    UnsupportedConfigError,
)


class PlacementPolicy(IntEnum):
    """Which tier each piece of an asymmetric KV chunk lives on."""

    ALL_NVME = 0
    """K and V both on the disk tier.  Saves disk capacity (75% of
    FP16) but reads both halves from NVMe on a hit."""

    ALL_CPU = 1
    """K and V both in CPU pinned memory.  No disk involvement.
    Fast restore but uses CPU RAM equal to FP16 K + FP8 V bytes."""

    SPLIT_K_CPU_V_NVME = 2
    """The split-tier headline.  K in CPU pinned (fast, full
    precision), V FP8 on NVMe (cold, compressed).  NVMe read on a
    hit is 1 B/element-pair (V only); CPU bytes are 2 B/element-pair
    (K only).  Best for prefix-cache restore under cold-tier pressure."""


# On-disk filenames within a chunk directory.
K_FILENAME = "K.bin"
V_FILENAME = "V.fp8"
META_FILENAME = "meta.bin"  # binary: same EncodedKV header layout


@dataclass
class SplitTierLayout:
    """Where a chunk's three physical pieces live on disk.

    For ALL_CPU, `k_path` and `v_path` are both None and the data
    lives only in pinned-CPU buffers managed by SplitTierStore.
    For SPLIT_K_CPU_V_NVME, k_path is None (K is in CPU memory)
    and v_path points to the FP8 V file.
    """

    chunk_dir: Path
    k_path: Optional[Path]
    v_path: Optional[Path]
    meta_path: Path

    @classmethod
    def for_chunk(
        cls,
        root: Path,
        cache_key: str,
        layer_id: int,
        chunk_id: int,
        policy: PlacementPolicy,
    ) -> "SplitTierLayout":
        chunk_dir = root / cache_key / f"layer_{layer_id:03d}" / f"chunk_{chunk_id:06d}"
        meta_path = chunk_dir / META_FILENAME
        if policy == PlacementPolicy.ALL_NVME:
            return cls(chunk_dir, chunk_dir / K_FILENAME, chunk_dir / V_FILENAME, meta_path)
        if policy == PlacementPolicy.ALL_CPU:
            return cls(chunk_dir, None, None, meta_path)
        if policy == PlacementPolicy.SPLIT_K_CPU_V_NVME:
            return cls(chunk_dir, None, chunk_dir / V_FILENAME, meta_path)
        raise UnsupportedConfigError(f"unknown placement policy: {policy}")


@dataclass
class SplitTierByteCounts:
    """Per-restore byte accounting.  Each field is bytes per cache
    hit, NOT cumulative across hits.  Use these in benchmarks to
    distinguish NVMe read traffic from CPU->GPU copy traffic."""

    nvme_bytes: int = 0
    cpu_bytes: int = 0
    meta_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.nvme_bytes + self.cpu_bytes + self.meta_bytes


class _CPUPinnedKStore:
    """K-store backed by CPU pinned memory.

    Pinned memory is mandatory for fast async H2D copy on the
    restore path.  When a real CUDA build is present we use
    torch.empty(..., pin_memory=True); on CPU-only test boxes we
    fall back to regular CPU memory (the contract is only that
    `.k_for(...)` returns the bytes; pin status only affects async
    copy performance, not correctness).
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self._k_tensors: dict[Tuple[str, int, int], torch.Tensor] = {}

    def can_fit(self, n_bytes: int) -> bool:
        return self.used_bytes + n_bytes <= self.max_bytes

    def put(
        self,
        cache_key: str,
        layer_id: int,
        chunk_id: int,
        k: torch.Tensor,
    ) -> None:
        size = k.numel() * k.element_size()
        if not self.can_fit(size):
            raise SplitTierFull(
                f"CPU pinned budget exceeded: need {size}, "
                f"used {self.used_bytes}/{self.max_bytes}"
            )
        # Store a clone in (preferably pinned) host memory.
        try:
            host = torch.empty(
                k.shape, dtype=k.dtype, pin_memory=torch.cuda.is_available()
            )
        except RuntimeError:
            # No CUDA: fall back to regular CPU memory.
            host = torch.empty(k.shape, dtype=k.dtype)
        host.copy_(k)
        self._k_tensors[(cache_key, layer_id, chunk_id)] = host
        self.used_bytes += size

    def get(self, cache_key: str, layer_id: int, chunk_id: int) -> Optional[torch.Tensor]:
        return self._k_tensors.get((cache_key, layer_id, chunk_id))

    def evict(self, cache_key: str, layer_id: int, chunk_id: int) -> bool:
        t = self._k_tensors.pop((cache_key, layer_id, chunk_id), None)
        if t is None:
            return False
        self.used_bytes -= t.numel() * t.element_size()
        return True

    def keys(self):
        return list(self._k_tensors.keys())


class SplitTierFull(CodecError):
    """The CPU pinned budget cannot hold the next chunk."""


class SplitTierStore:
    """Combined K-CPU + V-NVMe (or other policy) chunk store.

    The K side uses _CPUPinnedKStore; the V side uses the
    filesystem.  Codec metadata header lives next to V on disk so
    that scales are recoverable even if the K-CPU store is cold-
    started after a process restart.
    """

    def __init__(
        self,
        root: Path,
        codec: AsymK16V8Codec,
        *,
        policy: PlacementPolicy = PlacementPolicy.SPLIT_K_CPU_V_NVME,
        cpu_pinned_budget_bytes: int = 16 * 1024 * 1024 * 1024,  # 16 GiB
        on_cpu_full: str = "raise",
        expected_hashes: Optional[CodecHashes] = None,
    ):
        if on_cpu_full not in ("raise", "demote_k_to_nvme"):
            raise UnsupportedConfigError(
                f"unknown on_cpu_full: {on_cpu_full!r}; "
                f"expected 'raise' or 'demote_k_to_nvme'"
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.codec = codec
        self.policy = policy
        self.on_cpu_full = on_cpu_full
        self.expected_hashes = expected_hashes
        self._k_store = _CPUPinnedKStore(cpu_pinned_budget_bytes)

    # -------- write path --------

    def put(
        self,
        cache_key: str,
        layer_id: int,
        chunk_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> SplitTierByteCounts:
        """Store a chunk under the configured policy.  Returns the
        byte counts for this write (NVMe vs CPU vs metadata)."""
        encoded = self.codec.encode(
            k, v,
            hashes=self.expected_hashes or CodecHashes(),
            layer_id=layer_id,
            chunk_id=chunk_id,
        )
        layout = SplitTierLayout.for_chunk(
            self.root, cache_key, layer_id, chunk_id, self.policy
        )
        layout.chunk_dir.mkdir(parents=True, exist_ok=True)

        bytes_written = SplitTierByteCounts()
        # Header (containing K dtype, V dtype, scale shape, hashes,
        # CRC) is written alongside V so scales travel with their
        # values.  Header file is tiny.
        header = encoded.header_bytes
        layout.meta_path.write_bytes(header)
        bytes_written.meta_bytes = len(header)

        # Slice payload back into K_bytes, V_bytes, scale_bytes.
        k_bytes = encoded.payload[: encoded.k_payload_len]
        v_off = encoded.k_payload_len
        v_bytes = encoded.payload[v_off : v_off + encoded.v_payload_len]
        s_off = v_off + encoded.v_payload_len
        s_bytes = encoded.payload[s_off:]

        if self.policy == PlacementPolicy.ALL_NVME:
            assert layout.k_path is not None and layout.v_path is not None
            layout.k_path.write_bytes(k_bytes)
            # V file = V bytes + scales (kept together so a single
            # V read recovers the dequant info).
            layout.v_path.write_bytes(v_bytes + s_bytes)
            bytes_written.nvme_bytes = len(k_bytes) + len(v_bytes) + len(s_bytes)
            return bytes_written

        if self.policy == PlacementPolicy.ALL_CPU:
            self._try_put_k_to_cpu(cache_key, layer_id, chunk_id, k)
            # V also goes to CPU but as raw FP8 — we keep a tensor
            # view in the store too.  Skip writing to disk entirely.
            v_fp8_tensor = torch.frombuffer(
                bytearray(v_bytes), dtype=encoded.v_dtype
            ).clone().reshape(v.shape)
            self._k_store._k_tensors[(cache_key, layer_id, chunk_id, "V")] = v_fp8_tensor  # type: ignore[index]
            scale_tensor = torch.frombuffer(
                bytearray(s_bytes), dtype=encoded.scale_dtype
            ).clone().reshape(encoded.scale_shape)
            self._k_store._k_tensors[(cache_key, layer_id, chunk_id, "S")] = scale_tensor  # type: ignore[index]
            bytes_written.cpu_bytes = (
                k.numel() * k.element_size()
                + len(v_bytes)
                + len(s_bytes)
            )
            return bytes_written

        if self.policy == PlacementPolicy.SPLIT_K_CPU_V_NVME:
            self._try_put_k_to_cpu(cache_key, layer_id, chunk_id, k)
            assert layout.v_path is not None
            layout.v_path.write_bytes(v_bytes + s_bytes)
            bytes_written.cpu_bytes = k.numel() * k.element_size()
            bytes_written.nvme_bytes = len(v_bytes) + len(s_bytes)
            return bytes_written

        raise UnsupportedConfigError(f"unhandled policy: {self.policy}")

    def _try_put_k_to_cpu(
        self,
        cache_key: str,
        layer_id: int,
        chunk_id: int,
        k: torch.Tensor,
    ) -> None:
        size = k.numel() * k.element_size()
        if not self._k_store.can_fit(size):
            if self.on_cpu_full == "raise":
                raise SplitTierFull(
                    f"CPU pinned budget exceeded; configured on_cpu_full="
                    f"'raise'.  Need {size}, used "
                    f"{self._k_store.used_bytes}/{self._k_store.max_bytes}."
                )
            # demote_k_to_nvme: write K to disk under the same
            # chunk_dir.  Caller can read it back from there on the
            # restore path.
            layout = SplitTierLayout.for_chunk(
                self.root, cache_key, layer_id, chunk_id, self.policy
            )
            demoted_path = layout.chunk_dir / "K.demoted.bin"
            layout.chunk_dir.mkdir(parents=True, exist_ok=True)
            k_cpu = k.detach().to("cpu").contiguous().clone()
            demoted_path.write_bytes(bytes(k_cpu.untyped_storage()))
            return
        self._k_store.put(cache_key, layer_id, chunk_id, k)

    # -------- read path --------

    def get(
        self,
        cache_key: str,
        layer_id: int,
        chunk_id: int,
    ) -> Tuple[EncodedKV, SplitTierByteCounts]:
        """Reassemble the chunk's EncodedKV blob from whichever
        physical pieces the policy uses.  Returns the EncodedKV
        plus per-tier byte counts the caller can attribute to NVMe
        read traffic vs CPU memory access."""
        layout = SplitTierLayout.for_chunk(
            self.root, cache_key, layer_id, chunk_id, self.policy
        )
        if not layout.meta_path.exists():
            raise FileNotFoundError(
                f"split-tier chunk metadata missing: {layout.meta_path}"
            )
        bytes_read = SplitTierByteCounts()
        header = layout.meta_path.read_bytes()
        bytes_read.meta_bytes = len(header)

        # Parse header (without its trailing payload — we have
        # nothing yet) by appending zero-byte payload temporarily.
        # We need k_payload_len / v_payload_len / scale_payload_len
        # to compute how much to read where.  deserialize_header
        # requires header + payload; we pass header alone with
        # synthetic zero-length payload offsets.  To avoid that
        # complexity we re-create EncodedKV manually from the header
        # bytes by parsing them with deserialize_header on
        # header + dummy payload of declared length.
        # Simpler: just store header + payload combined and skip
        # ALL_CPU edge case (which has no disk header).  Both ALL_NVME
        # and SPLIT have a sensible meta+V layout we can read.
        if self.policy == PlacementPolicy.ALL_CPU:
            return self._get_from_cpu_store(cache_key, layer_id, chunk_id, header)

        # Parse header to discover payload lengths.
        # Trick: the header is variable-length.  We re-call
        # deserialize_header with header + zeros payload of inferred
        # max length using struct probing.
        enc_skel = self._parse_header_only(header)

        if self.policy == PlacementPolicy.ALL_NVME:
            assert layout.k_path is not None and layout.v_path is not None
            k_bytes = layout.k_path.read_bytes()
            v_plus_scales = layout.v_path.read_bytes()
            bytes_read.nvme_bytes = len(k_bytes) + len(v_plus_scales)
        elif self.policy == PlacementPolicy.SPLIT_K_CPU_V_NVME:
            # K from CPU
            k_tensor = self._k_store.get(cache_key, layer_id, chunk_id)
            if k_tensor is None:
                # Possibly demoted to NVMe if the CPU budget filled up.
                demoted = layout.chunk_dir / "K.demoted.bin"
                if not demoted.exists():
                    raise FileNotFoundError(
                        f"K not in CPU store and no demoted file at {demoted}"
                    )
                k_bytes = demoted.read_bytes()
                bytes_read.nvme_bytes += len(k_bytes)
            else:
                k_bytes = bytes(k_tensor.contiguous().untyped_storage())
                bytes_read.cpu_bytes += len(k_bytes)
            assert layout.v_path is not None
            v_plus_scales = layout.v_path.read_bytes()
            bytes_read.nvme_bytes += len(v_plus_scales)
        else:
            raise UnsupportedConfigError(f"unhandled policy: {self.policy}")

        # Reassemble payload in the canonical K|V|scales order.
        v_bytes_len = enc_skel.v_payload_len
        s_bytes_len = enc_skel.scale_payload_len
        v_bytes = v_plus_scales[:v_bytes_len]
        s_bytes = v_plus_scales[v_bytes_len : v_bytes_len + s_bytes_len]
        full_blob = header + k_bytes + v_bytes + s_bytes

        encoded = deserialize_header(full_blob)
        if self.expected_hashes is not None:
            self.codec._check_hash_match(encoded.hashes, self.expected_hashes)
        return encoded, bytes_read

    def _get_from_cpu_store(
        self,
        cache_key: str,
        layer_id: int,
        chunk_id: int,
        header: bytes,
    ) -> Tuple[EncodedKV, SplitTierByteCounts]:
        bytes_read = SplitTierByteCounts(meta_bytes=len(header))
        k = self._k_store.get(cache_key, layer_id, chunk_id)
        v = self._k_store._k_tensors.get((cache_key, layer_id, chunk_id, "V"))  # type: ignore[index]
        s = self._k_store._k_tensors.get((cache_key, layer_id, chunk_id, "S"))  # type: ignore[index]
        if k is None or v is None or s is None:
            raise FileNotFoundError(
                f"ALL_CPU chunk missing in store: "
                f"{cache_key}/layer_{layer_id}/chunk_{chunk_id}"
            )
        k_bytes = bytes(k.contiguous().untyped_storage())
        v_bytes = bytes(v.contiguous().untyped_storage())
        s_bytes = bytes(s.contiguous().untyped_storage())
        bytes_read.cpu_bytes = len(k_bytes) + len(v_bytes) + len(s_bytes)
        full_blob = header + k_bytes + v_bytes + s_bytes
        encoded = deserialize_header(full_blob)
        return encoded, bytes_read

    @staticmethod
    def _parse_header_only(header: bytes) -> EncodedKV:
        """Header-only parse: reconstruct EncodedKV without payload.

        We accomplish this by appending a synthetic payload of the
        right declared length filled with zeros, then patching the
        CRC.  Cheaper than reimplementing the parser since we just
        want the length fields.
        """
        # First Party
        from lmcache.v1.kv_codec.encoded_kv import _FIXED_HEADER_LEN
        # Standard
        import struct
        import zlib

        # Find scale_shape_n at offset 68 (last int64 in fixed
        # header) to compute where payload_lens start.
        if len(header) < _FIXED_HEADER_LEN + 24 + 2:
            raise CorruptEncodedKVError("header too short to parse lengths")
        (scale_shape_n,) = struct.unpack_from("<q", header, 68)
        if scale_shape_n < 0 or scale_shape_n > 8:
            raise CorruptEncodedKVError(
                f"implausible scale_shape_n: {scale_shape_n}"
            )
        len_off = _FIXED_HEADER_LEN + 8 * scale_shape_n
        k_len, v_len, s_len = struct.unpack_from("<qqq", header, len_off)

        # Synthesize a zero payload + compute its CRC; replace the
        # CRC field at the end of header with that.
        payload = b"\x00" * (k_len + v_len + s_len)
        synthetic_crc = zlib.crc32(payload) & 0xFFFFFFFF
        # CRC is the last 4 bytes of the header.
        patched = header[:-4] + struct.pack("<I", synthetic_crc)
        # Now deserialize_header(patched + payload) succeeds.
        enc = deserialize_header(patched + payload)
        return enc

    # -------- accessors --------

    @property
    def cpu_used_bytes(self) -> int:
        return self._k_store.used_bytes
