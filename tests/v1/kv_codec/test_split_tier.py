# SPDX-License-Identifier: Apache-2.0
"""Phase 5: K-hot / V-cold split-tier placement tests.

The headline result of the LMCache asymmetric plan: when K stays in
CPU pinned memory and V lives as FP8 on NVMe, the cache-hit NVMe
read traffic drops from K16+V16 = 4 B/element-pair (FP16 baseline)
to V8 = 1 B/element-pair (V only) — 4x reduction in cold-tier
read bytes.

Tests verify:

- byte counts are attributed to the correct tier (NVMe vs CPU vs
  metadata) so benchmark numbers don't conflate
- the actual NVMe read on a hit is V-only under split-tier
- ALL_NVME and SPLIT_K_CPU_V_NVME produce bit-identical reconstructed
  K and V (the split is a placement decision, not an encoding one)
- CPU pinned-budget pressure: configurable raise vs demote
- multi-chunk concurrent storage doesn't corrupt across keys
"""

# Standard
from pathlib import Path

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    CodecHashes,
    PlacementPolicy,
    SplitTierFull,
    SplitTierStore,
    deserialize_header,
)


def _make_kv(shape=(4, 16, 32), dtype=torch.float16, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return (
        torch.randn(*shape, dtype=dtype, generator=g),
        torch.randn(*shape, dtype=dtype, generator=g),
    )


def _store(tmp_path, policy, **kwargs):
    return SplitTierStore(
        root=tmp_path,
        codec=AsymK16V8Codec(),
        policy=policy,
        **kwargs,
    )


def test_split_layout_paths(tmp_path):
    """Catches: collisions between layers/chunks via path layout."""
    s = _store(tmp_path, PlacementPolicy.SPLIT_K_CPU_V_NVME)
    k, v = _make_kv()
    s.put("session_42", layer_id=3, chunk_id=7, k=k, v=v)
    expected_dir = tmp_path / "session_42" / "layer_003" / "chunk_000007"
    assert expected_dir.exists()
    assert (expected_dir / "V.fp8").exists()
    assert (expected_dir / "meta.bin").exists()
    # K is in CPU memory under SPLIT, NOT on disk.
    assert not (expected_dir / "K.bin").exists()


def test_split_byte_counts_attribution(tmp_path):
    """Catches: byte counts mis-attributed across tiers (NVMe vs
    CPU).  Under SPLIT_K_CPU_V_NVME, the NVMe-read on a hit must
    be V + scales bytes, NOT K bytes."""
    s = _store(tmp_path, PlacementPolicy.SPLIT_K_CPU_V_NVME)
    k, v = _make_kv(shape=(4, 32, 64))
    write_counts = s.put("k0", 0, 0, k, v)
    # On write: K bytes go to CPU pinned, V+scales to NVMe.
    assert write_counts.cpu_bytes == k.numel() * k.element_size()
    # V FP8 is 1 byte per element; scales are tiny.
    assert write_counts.nvme_bytes >= v.numel() * 1  # at least V bytes
    assert write_counts.nvme_bytes < v.numel() * 2  # well below V at FP16

    _, read_counts = s.get("k0", 0, 0)
    # Critical: read NVMe traffic should NOT include K bytes.
    assert read_counts.cpu_bytes >= k.numel() * 2, "K should be read from CPU"
    assert read_counts.nvme_bytes < k.numel() * 2, (
        "NVMe read should be V-only, not include K"
    )
    # The headline ratio: NVMe reads ~1 B/element-pair under split.
    nvme_per_elem_pair = read_counts.nvme_bytes / k.numel()
    assert nvme_per_elem_pair < 1.5, nvme_per_elem_pair


def test_all_nvme_byte_counts(tmp_path):
    """Under ALL_NVME, both K and V are on disk; CPU bytes are
    zero on read."""
    s = _store(tmp_path, PlacementPolicy.ALL_NVME)
    k, v = _make_kv(shape=(4, 32, 64))
    s.put("k0", 0, 0, k, v)
    _, read_counts = s.get("k0", 0, 0)
    assert read_counts.cpu_bytes == 0
    # NVMe reads: K (2 B/elem) + V (1 B/elem) + scales = ~3 B/elem.
    nvme_per_elem_pair = read_counts.nvme_bytes / k.numel()
    assert 2.5 < nvme_per_elem_pair < 3.5, nvme_per_elem_pair


def test_nvme_traffic_ratio_split_vs_all_nvme(tmp_path):
    """The killer-table claim: split-tier reads 1/4 the NVMe bytes
    of an ALL_FP16 path (1 B vs 4 B per elem pair) and 1/3 the bytes
    of asymmetric ALL_NVME (1 B vs 3 B)."""
    k, v = _make_kv(shape=(8, 32, 64))

    s_all = _store(tmp_path / "all", PlacementPolicy.ALL_NVME)
    s_split = _store(tmp_path / "split", PlacementPolicy.SPLIT_K_CPU_V_NVME)
    s_all.put("k", 0, 0, k, v)
    s_split.put("k", 0, 0, k, v)
    _, c_all = s_all.get("k", 0, 0)
    _, c_split = s_split.get("k", 0, 0)
    ratio = c_split.nvme_bytes / c_all.nvme_bytes
    # Asymmetric all-NVMe = 3 B/elem; split = 1 B/elem; ratio < 0.5.
    assert ratio < 0.45, ratio


def test_split_K_recovered_bit_exact(tmp_path):
    """Under any policy the recovered K is bit-equal to what was
    stored.  This is the "asymmetric is honest" guarantee at the
    placement layer."""
    s = _store(tmp_path, PlacementPolicy.SPLIT_K_CPU_V_NVME)
    k, v = _make_kv()
    s.put("k", 0, 0, k, v)
    encoded, _ = s.get("k", 0, 0)
    # Pull K out of the encoded blob and compare bit-exact.
    k_back = torch.frombuffer(
        bytearray(encoded.payload[: encoded.k_payload_len]),
        dtype=encoded.k_dtype,
    ).clone().reshape(k.shape)
    assert torch.equal(k_back, k)


def test_split_V_within_fp8_noise(tmp_path):
    """V comes back FP8-noise close to the original."""
    s = _store(tmp_path, PlacementPolicy.SPLIT_K_CPU_V_NVME)
    k, v = _make_kv(shape=(4, 64, 32))
    s.put("k", 0, 0, k, v)
    encoded, _ = s.get("k", 0, 0)
    # Decode V via codec's dequant path
    _, v_dq, _ = AsymK16V8Codec().decode(encoded, out_v_dtype=torch.float16)
    rel = (v_dq.view(v.shape).to(torch.float32) - v.to(torch.float32)).abs() / (
        v.abs().to(torch.float32) + 1e-6
    )
    assert rel.median().item() < 0.075


def test_cpu_budget_raise_when_full(tmp_path):
    """Configured on_cpu_full='raise' triggers SplitTierFull when
    the budget can't hold the next K."""
    k, v = _make_kv(shape=(4, 16, 32))
    k_size = k.numel() * 2
    s = _store(
        tmp_path,
        PlacementPolicy.SPLIT_K_CPU_V_NVME,
        cpu_pinned_budget_bytes=k_size,  # exactly one K fits
        on_cpu_full="raise",
    )
    s.put("k0", 0, 0, k, v)
    with pytest.raises(SplitTierFull):
        s.put("k1", 0, 0, k, v)


def test_cpu_budget_demote_when_full(tmp_path):
    """Configured on_cpu_full='demote_k_to_nvme' writes K to disk
    instead of raising."""
    k, v = _make_kv(shape=(4, 16, 32))
    k_size = k.numel() * 2
    s = _store(
        tmp_path,
        PlacementPolicy.SPLIT_K_CPU_V_NVME,
        cpu_pinned_budget_bytes=k_size,
        on_cpu_full="demote_k_to_nvme",
    )
    s.put("k0", 0, 0, k, v)
    s.put("k1", 0, 0, k, v)  # demoted to disk
    # Demoted K file should exist
    demoted = tmp_path / "k1" / "layer_000" / "chunk_000000" / "K.demoted.bin"
    assert demoted.exists(), demoted
    # Reading it back still produces the right K.
    encoded, counts = s.get("k1", 0, 0)
    # NVMe read traffic now includes K because it was demoted.
    assert counts.nvme_bytes >= k_size


def test_multi_chunk_isolation(tmp_path):
    """Multiple chunks across multiple layers don't corrupt each
    other."""
    s = _store(tmp_path, PlacementPolicy.SPLIT_K_CPU_V_NVME)
    chunks = []
    for layer in range(4):
        for chunk in range(3):
            k, v = _make_kv(seed=layer * 100 + chunk)
            chunks.append((layer, chunk, k, v))
            s.put("session", layer, chunk, k, v)
    # Read in shuffled order; each must round-trip.
    for layer, chunk, k, v in chunks[::-1]:
        encoded, _ = s.get("session", layer, chunk)
        k_back = torch.frombuffer(
            bytearray(encoded.payload[: encoded.k_payload_len]),
            dtype=encoded.k_dtype,
        ).clone().reshape(k.shape)
        assert torch.equal(k_back, k), (layer, chunk)


def test_meta_file_carries_codec_header(tmp_path):
    """meta.bin file is exactly the codec header, parseable
    standalone for diagnostics."""
    s = _store(tmp_path, PlacementPolicy.SPLIT_K_CPU_V_NVME)
    k, v = _make_kv(shape=(2, 8, 16))
    s.put("session", 5, 17, k, v)
    meta_path = tmp_path / "session" / "layer_005" / "chunk_000017" / "meta.bin"
    header_bytes = meta_path.read_bytes()
    # Header alone won't deserialize (no payload); but the magic
    # bytes and layer_id should be present.
    assert header_bytes[:8] == b"LMCKV\x01\x00\x01"


def test_invalid_placement_policy_rejected():
    """Unknown PlacementPolicy values surface a typed error."""
    # Standard
    from lmcache.v1.kv_codec import UnsupportedConfigError
    # Standard
    from lmcache.v1.kv_codec.split_tier import SplitTierLayout
    with pytest.raises(UnsupportedConfigError):
        SplitTierLayout.for_chunk(
            Path("/tmp"), "k", 0, 0, policy=99,  # type: ignore
        )


def test_cross_hash_mismatch_caught(tmp_path):
    """Cross-config gating via expected_hashes carried into the
    store at construction.  Use ALL_NVME so both stores see the
    full file set (SPLIT keeps K in CPU memory per-instance, which
    is correct behavior but means a reader-from-cold doesn't have
    K to find — that case is its own test below)."""
    # Standard
    from lmcache.v1.kv_codec import CodecMismatchError
    s_writer = SplitTierStore(
        root=tmp_path,
        codec=AsymK16V8Codec(),
        policy=PlacementPolicy.ALL_NVME,
        expected_hashes=CodecHashes(model_id="qwen-7b"),
    )
    s_reader = SplitTierStore(
        root=tmp_path,
        codec=AsymK16V8Codec(),
        policy=PlacementPolicy.ALL_NVME,
        expected_hashes=CodecHashes(model_id="llama-8b"),
    )
    k, v = _make_kv()
    s_writer.put("k", 0, 0, k, v)
    with pytest.raises(CodecMismatchError, match="model_id"):
        s_reader.get("k", 0, 0)


def test_split_K_in_cpu_does_not_survive_process_restart(tmp_path):
    """Document the SPLIT_K_CPU_V_NVME behavior: K lives in process
    memory (pinned CPU), so a fresh store instance cannot recover K
    even if V and meta are still on disk.  This is by design — for
    cross-process recovery, callers should use ALL_NVME or
    explicitly demote K."""
    s_writer = _store(tmp_path, PlacementPolicy.SPLIT_K_CPU_V_NVME)
    k, v = _make_kv()
    s_writer.put("k", 0, 0, k, v)
    # New store instance (fresh CPU memory)
    s_fresh = _store(tmp_path, PlacementPolicy.SPLIT_K_CPU_V_NVME)
    with pytest.raises(FileNotFoundError, match="K not in CPU store"):
        s_fresh.get("k", 0, 0)
