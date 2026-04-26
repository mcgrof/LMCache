#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 6 GPU bench: codec encode/decode timing on CUDA across
realistic chunk sizes.  Measures wall-clock latency, peak GPU
memory, and the no-FP16-V-allocation property at scale."""

# Standard
import argparse
import json
import statistics
import time
from pathlib import Path

# Third Party
import torch

# First Party
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    ScaleScope,
    compute_v_scales,
    quantize_v_fp8,
)


def bench_one(*, n_pages, page_size, n_heads, head_dim, dtype, n_iters=10):
    """Encode + decode roundtrip timing on CUDA."""
    shape = (n_pages, page_size, n_heads, head_dim)
    g = torch.Generator(device="cuda")
    g.manual_seed(42)

    # Pre-build inputs on GPU.
    k = torch.randn(*shape, dtype=dtype, device="cuda", generator=g)
    v_fp16 = torch.randn(*shape, dtype=dtype, device="cuda", generator=g)
    scales = compute_v_scales(v_fp16, ScaleScope.PER_TENSOR).to(torch.float32)
    v_fp8 = quantize_v_fp8(v_fp16, scales, ScaleScope.PER_TENSOR)
    fp16_v_bytes = v_fp16.numel() * 2
    del v_fp16
    torch.cuda.empty_cache()

    codec = AsymK16V8Codec()

    # Encode timings (native_asym path): use precomputed K/V/scales.
    enc_times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        encoded = codec.encode(
            k,
            torch.zeros(v_fp8.shape, dtype=k.dtype, device="cuda"),
            precomputed_v_quant=v_fp8,
            precomputed_v_scales=scales,
        )
        blob = codec.to_bytes(encoded)
        torch.cuda.synchronize()
        enc_times.append(time.time() - t0)

    # Decode timings (native_asym): no FP16 V materialization.
    dec_times = []
    transient_peaks = []
    for _ in range(n_iters):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        t0 = time.time()
        parsed = codec.from_bytes(blob)
        k_back, v_back, s_back = codec.decode(
            parsed, out_v_dtype=None, device=torch.device("cuda")
        )
        torch.cuda.synchronize()
        dec_times.append(time.time() - t0)
        peak = torch.cuda.max_memory_allocated()
        transient_peaks.append(peak - start)
        del k_back, v_back, s_back

    # Encoded blob size
    encoded_bytes = len(blob)
    fp16_baseline_bytes = (k.numel() * 2) * 2  # K + V both at fp16

    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "encoded_bytes": encoded_bytes,
        "fp16_baseline_bytes": fp16_baseline_bytes,
        "storage_ratio": encoded_bytes / fp16_baseline_bytes,
        "encode_ms_median": statistics.median(enc_times) * 1000,
        "decode_ms_median": statistics.median(dec_times) * 1000,
        "decode_ms_p95": sorted(dec_times)[int(len(dec_times) * 0.95) - 1] * 1000,
        "transient_peak_bytes_median": statistics.median(transient_peaks),
        "fp16_v_size_bytes": fp16_v_bytes,
        "transient_peak_below_fp16_v": (
            statistics.median(transient_peaks) < fp16_v_bytes
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_gpu_codec.json")
    ap.add_argument("--n-iters", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; cannot run GPU bench")
        return 1

    grid = []
    # Realistic chunk sizes for paged caches.
    for n_pages in (4, 16, 64):
        for page_size in (16, 32):
            for n_heads in (8, 16, 32):
                for head_dim in (64, 128):
                    grid.append(dict(
                        n_pages=n_pages, page_size=page_size,
                        n_heads=n_heads, head_dim=head_dim,
                    ))

    rows = []
    for cfg in grid:
        for dtype in (torch.float16, torch.bfloat16):
            print(
                f"  {cfg['n_pages']:3d} {cfg['page_size']:3d} "
                f"{cfg['n_heads']:3d} {cfg['head_dim']:4d} {dtype}",
                flush=True,
            )
            rows.append(bench_one(**cfg, dtype=dtype, n_iters=args.n_iters))

    storage_ratios = [r["storage_ratio"] for r in rows]
    decode_ms = [r["decode_ms_median"] for r in rows]
    no_fp16_alloc = sum(1 for r in rows if r["transient_peak_below_fp16_v"])

    summary = {
        "gpu": torch.cuda.get_device_name(),
        "n_configurations": len(rows),
        "storage_ratio_vs_fp16": {
            "mean": statistics.mean(storage_ratios),
            "median": statistics.median(storage_ratios),
            "min": min(storage_ratios),
            "max": max(storage_ratios),
        },
        "decode_latency_ms": {
            "median": statistics.median(decode_ms),
            "min": min(decode_ms),
            "max": max(decode_ms),
        },
        "no_fp16_v_alloc_passing": f"{no_fp16_alloc}/{len(rows)}",
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")
    print(f"GPU: {summary['gpu']}")
    print(
        f"asym storage ratio vs fp16: median={summary['storage_ratio_vs_fp16']['median']:.4f}"
    )
    print(
        f"decode latency:             median={summary['decode_latency_ms']['median']:.2f} ms"
    )
    print(
        f"no-FP16-V-alloc property:   {summary['no_fp16_v_alloc_passing']} configs pass"
    )


if __name__ == "__main__":
    main()
