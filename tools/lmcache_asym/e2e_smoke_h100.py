"""End-to-end smoke test: full LMCache asymmetric integration on H100.

Exercises:
  1. Real LMCacheEngineConfig with kv_placement_policy='split_k_cpu_v_nvme'.
  2. CreateStorageBackends factory dispatches SplitTierStorageBackend.
  3. attention_layer registry populates correctly via register_kv_caches'
     attention_layers parameter (mocked since vLLM isn't installed here).
  4. SplitTierStorageBackend.put / get_blocking roundtrip on GPU tensors.
  5. AsymKVMemoryObj decode produces K bit-exact + V within FP8 noise.
  6. GPU memory snapshot during decode confirms no FP16 V allocation.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import torch

if not torch.cuda.is_available():
    print("CUDA required for this smoke test")
    sys.exit(1)

print("=" * 60)
print("LMCache asymmetric E2E smoke on", torch.cuda.get_device_name())
print("=" * 60)

# ---- 1. Build a real config + factory ----
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend import CreateStorageBackends

with tempfile.TemporaryDirectory() as td:
    cfg = LMCacheEngineConfig.from_defaults(
        local_disk=str(td),
        max_local_disk_size=2.0,
        max_local_cpu_size=1.0,
    )
    cfg.kv_placement_policy = "split_k_cpu_v_nvme"
    cfg.kv_storage_codec = "asym_k16_v8_e4m3"
    cfg.kv_runtime_layout = "native_asym"

    from unittest.mock import MagicMock
    md = MagicMock()
    md.role = "worker"
    md.first_rank = 0
    md.worker_id = 0
    md.use_mla = False
    md.model_name = "qwen2.5-7b-test"

    loop = asyncio.new_event_loop()
    backends = CreateStorageBackends(
        config=cfg, metadata=md, loop=loop, dst_device="cuda"
    )
    print(f"\n[1] CreateStorageBackends keys: {list(backends.keys())}")
    assert "SplitTierStorageBackend" in backends, "split-tier missing"
    assert not any("LocalDiskBackend" in k for k in backends), \
        "LocalDiskBackend should be skipped under split-tier"
    print("    SplitTierStorageBackend registered")
    print("    LocalDiskBackend correctly skipped")

    backend = backends["SplitTierStorageBackend"]

    # ---- 2. Attention-layer registry ----
    from lmcache.integration.vllm.asym_kv_view_builder import (
        ATTENTION_LAYER_REGISTRY,
        clear_attention_layer_registry,
        register_attention_layer,
        build_asym_kv_view_for_layer,
    )

    class _MockAsymAttention:
        kv_cache_dtype = ("auto", "fp8_e4m3")
        _v_scale_float = 0.234

    clear_attention_layer_registry()
    for i in range(8):
        register_attention_layer(f"layer.{i}", _MockAsymAttention())
    print(f"\n[2] Registered {len(ATTENTION_LAYER_REGISTRY)} attention layers")
    assert len(ATTENTION_LAYER_REGISTRY) == 8

    # Lookup test
    k_buf = torch.randn(4, 16, 8, 64, dtype=torch.float16, device="cuda")
    v_fp8_buf = torch.zeros(4, 16, 8, 64,
                            dtype=torch.float8_e4m3fn, device="cuda")
    view = build_asym_kv_view_for_layer(
        "layer.3", kv_layer=(k_buf, v_fp8_buf)
    )
    assert view is not None, "registry lookup returned None"
    assert view.k.device.type == "cuda"
    assert view.v_fp8.device.type == "cuda"
    print("    build_asym_kv_view_for_layer returns AsymKVView on cuda")

    # ---- 3. Backend put/get roundtrip ----
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.memory_management import (
        TensorMemoryObj, MemoryObjMetadata, MemoryFormat,
    )

    g = torch.Generator(device="cuda")
    g.manual_seed(42)
    kv_tensor = torch.randn(
        2, 4, 16, 8, 64, dtype=torch.float16,
        device="cuda", generator=g,
    )
    obj = TensorMemoryObj(
        raw_data=kv_tensor,
        metadata=MemoryObjMetadata(
            shape=torch.Size(kv_tensor.shape),
            dtype=kv_tensor.dtype,
            address=0,
            phy_size=kv_tensor.numel() * kv_tensor.element_size(),
            ref_count=1, pin_count=0, fmt=MemoryFormat.KV_2LTD,
        ),
        parent_allocator=None,
    )
    cache_key = CacheEngineKey(
        model_name="qwen2.5-7b-test", world_size=1,
        worker_id=0, chunk_hash=hash("smoke") & 0x7FFFFFFFFFFFFFFF,
        dtype=torch.float16,
    )
    futures = backend.batched_submit_put_task([cache_key], [obj])
    from concurrent.futures import wait as wait_for
    wait_for(futures)
    assert backend.contains(cache_key), "put didn't land"
    print("\n[3] Backend put + contains: OK")

    out = backend.get_blocking(cache_key)
    assert out is not None, "get_blocking returned None"
    print(f"    get_blocking returned BytesBufferMemoryObj with "
          f"{len(out.raw_data)} bytes")

    # Decode and verify
    encoded = backend.codec.from_bytes(bytes(out.raw_data))
    k_back, v_back, scales = backend.codec.decode(
        encoded, out_v_dtype=torch.float16, device=torch.device("cuda")
    )
    per_half = torch.Size([4, 16, 8, 64])
    k_back = k_back.reshape(per_half)
    v_back = v_back.reshape(per_half)
    assert torch.equal(k_back, kv_tensor[0]), "K not bit-exact"
    rel = (v_back.to(torch.float32) - kv_tensor[1].to(torch.float32)).abs() \
          / (kv_tensor[1].abs().to(torch.float32) + 1e-6)
    print(f"    K bit-exact: True")
    print(f"    V rel err median: {rel.median():.4f} "
          f"(bound: 0.075)")
    assert rel.median().item() < 0.075

    # ---- 4. Byte counts attribution ----
    writes, reads = backend.get_byte_counts()
    print(f"\n[4] Byte counts:")
    print(f"    Write: NVMe={writes.nvme_bytes:>10,}  "
          f"CPU={writes.cpu_bytes:>10,}  meta={writes.meta_bytes:>4,}")
    print(f"    Read:  NVMe={reads.nvme_bytes:>10,}  "
          f"CPU={reads.cpu_bytes:>10,}  meta={reads.meta_bytes:>4,}")
    n_elem = 4 * 16 * 8 * 64
    print(f"    K bytes (FP16): {n_elem * 2:>10,} (CPU expected)")
    print(f"    V bytes (FP8):  {n_elem:>10,} (NVMe expected)")
    assert writes.cpu_bytes == n_elem * 2, "K not on CPU"

    # ---- 5. GPU memory snapshot during decode ----
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    parsed = backend.codec.from_bytes(bytes(out.raw_data))
    k2, v2, s2 = backend.codec.decode(
        parsed, out_v_dtype=None, device=torch.device("cuda")
    )
    torch.cuda.synchronize()
    peak_transient = torch.cuda.max_memory_allocated() - start

    fp16_v_bytes = n_elem * 2
    legitimate = n_elem * 2 + n_elem * 1 + 4
    silent_fp16 = legitimate + fp16_v_bytes
    print(f"\n[5] GPU memory snapshot (native_asym decode):")
    print(f"    transient_peak:   {peak_transient:>12,} bytes")
    print(f"    legitimate:       {legitimate:>12,} bytes")
    print(f"    silent_fp16_thr:  {silent_fp16:>12,} bytes")
    print(f"    peak < silent:    {peak_transient < silent_fp16}")
    assert peak_transient < silent_fp16, \
        "Silent FP16 V allocation detected on GPU"

    backend.close()

print("\n" + "=" * 60)
print("ALL E2E SMOKE CHECKS PASSED")
print("=" * 60)
