#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure byte-accounting benchmark for the asymmetric KV codec.

Measures encoded blob size for FP16-baseline / asym all-NVMe / asym
split-tier across a synthetic KV grid and reports the storage ratios
that go in the paper.  CPU-only.

Usage:
    python3 bench_codec_bytes.py --out RESULTS.json
"""

# Standard
import argparse
import json
import statistics
from pathlib import Path

# Third Party
import torch

# First Party
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    PlacementPolicy,
    SplitTierStore,
)


def _kv(shape, dtype):
    g = torch.Generator()
    g.manual_seed(42)
    return (
        torch.randn(*shape, dtype=dtype, generator=g),
        torch.randn(*shape, dtype=dtype, generator=g),
    )


def _fp16_baseline_bytes(k, v):
    """Bytes a naive FP16 KV serialization would write (no header)."""
    return k.numel() * k.element_size() + v.numel() * v.element_size()


def measure_one(
    *,
    n_pages,
    page_size,
    n_heads,
    head_dim,
    dtype,
    tmpdir,
):
    shape = (n_pages, page_size, n_heads, head_dim)
    k, v = _kv(shape, dtype)
    fp16_bytes = _fp16_baseline_bytes(k, v)

    codec = AsymK16V8Codec()
    encoded = codec.encode(k, v)
    asym_blob_bytes = len(codec.to_bytes(encoded))

    # Split-tier byte accounting via SplitTierStore.
    store_split = SplitTierStore(
        root=tmpdir / "split",
        codec=AsymK16V8Codec(),
        policy=PlacementPolicy.SPLIT_K_CPU_V_NVME,
    )
    counts_split = store_split.put("k", 0, 0, k, v)

    store_all = SplitTierStore(
        root=tmpdir / "all",
        codec=AsymK16V8Codec(),
        policy=PlacementPolicy.ALL_NVME,
    )
    counts_all = store_all.put("k", 0, 0, k, v)

    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "fp16_baseline_bytes": fp16_bytes,
        "asym_encoded_bytes": asym_blob_bytes,
        "asym_storage_ratio_vs_fp16": asym_blob_bytes / fp16_bytes,
        "all_nvme": {
            "nvme_bytes_write": counts_all.nvme_bytes,
            "cpu_bytes_write": counts_all.cpu_bytes,
            "meta_bytes_write": counts_all.meta_bytes,
        },
        "split": {
            "nvme_bytes_write": counts_split.nvme_bytes,
            "cpu_bytes_write": counts_split.cpu_bytes,
            "meta_bytes_write": counts_split.meta_bytes,
        },
        # The headline number: NVMe-traffic-on-read for split vs all.
        # On write the layout is symmetric to read; we use the write
        # bytes as a proxy here.
        "split_vs_all_nvme_ratio": (
            counts_split.nvme_bytes / max(counts_all.nvme_bytes, 1)
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_codec_bytes.json")
    args = ap.parse_args()
    out_path = Path(args.out).resolve()

    grid = []
    # Realistic chunk sizes for paged caches.  Kept under 8 MB / shape
    # for CPU-only FP8 emulation (PyTorch's CPU fp8 path is slow at
    # large shapes).  When running on a GPU box, expand by adding
    # more (n_pages) or removing the page_size=16 constraint.
    for n_pages in (4, 16):
        for page_size in (16, 32):
            for n_heads in (8, 32):
                for head_dim in (64, 128):
                    grid.append(
                        dict(
                            n_pages=n_pages,
                            page_size=page_size,
                            n_heads=n_heads,
                            head_dim=head_dim,
                        )
                    )

    # Standard
    import tempfile

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        total = len(grid) * 2
        idx = 0
        for cfg in grid:
            for dtype in (torch.float16, torch.bfloat16):
                idx += 1
                print(
                    f"  [{idx:3d}/{total}] n_pages={cfg['n_pages']:3d} "
                    f"page_size={cfg['page_size']:3d} "
                    f"n_heads={cfg['n_heads']:3d} "
                    f"head_dim={cfg['head_dim']:4d} {dtype}",
                    flush=True,
                )
                rows.append(
                    measure_one(
                        **cfg,
                        dtype=dtype,
                        tmpdir=tmpdir / f"{cfg['n_pages']}_{cfg['page_size']}_{cfg['n_heads']}_{cfg['head_dim']}_{dtype}",
                    )
                )

    storage_ratios = [r["asym_storage_ratio_vs_fp16"] for r in rows]
    split_ratios = [r["split_vs_all_nvme_ratio"] for r in rows]

    summary = {
        "n_configurations": len(rows),
        "storage_ratio_vs_fp16": {
            "mean": statistics.mean(storage_ratios),
            "median": statistics.median(storage_ratios),
            "min": min(storage_ratios),
            "max": max(storage_ratios),
        },
        "split_vs_all_nvme_ratio": {
            "mean": statistics.mean(split_ratios),
            "median": statistics.median(split_ratios),
            "min": min(split_ratios),
            "max": max(split_ratios),
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(
        f"asym storage ratio vs fp16: mean={summary['storage_ratio_vs_fp16']['mean']:.3f}, "
        f"median={summary['storage_ratio_vs_fp16']['median']:.3f}"
    )
    print(
        f"split-tier NVMe ratio vs all-NVMe: mean={summary['split_vs_all_nvme_ratio']['mean']:.3f}, "
        f"median={summary['split_vs_all_nvme_ratio']['median']:.3f}"
    )


if __name__ == "__main__":
    main()
