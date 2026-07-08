# SPDX-License-Identifier: Apache-2.0
"""GPU-free KV-cache-offload IO workload generator.

The KV-cache calculator answers "how many bytes is this model's KV cache?"
This answers the next question -- "what does *offloading* it to disk actually
do?" -- without a GPU or a model.

It takes a real model config (the calculator's ``modelconfig.json`` format),
computes the KV-cache block size for one chunk of tokens with the calculator's
geometry (``kv_geometry.py``), then issues that store/load workload against a
real device through LMCache's ``raw_block`` engine -- POSIX, io_uring, or
io_uring_cmd NVMe passthrough. The KV payload is fake bytes: storage IO geometry
(command count, sizes, total bytes) depends only on the block size and the
device's transfer limit, not on the tensor values, so real model *dimensions* +
fake *content* reproduce the real offload IO pattern.

It reports the per-chunk NVMe-command geometry and measured store/load latency,
and can emit a ``kvio_record.json`` manifest (replayable) and fire LMCache's
``LMCACHE_KVIO_TRACE`` semantic trace for cross-layer validation.

Example (real NVMe passthrough):
    python run_kv_offload_io.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --dtype bfloat16 --chunk-tokens 256 --num-chunks 8 \\
        --device /dev/ng0n1 --engine uring_cmd --record /tmp/kvio_record.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

from kv_geometry import kv_cache_bytes

# LMCache public API (no dependency on the test suite).
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.raw_block import RawBlockCore, RawBlockCoreConfig
from lmcache.v1.storage_backend.raw_block.key_codec import encode_object_key

import torch


def make_memory_obj(payload: bytes) -> TensorMemoryObj:
    data = bytearray(payload)
    raw = torch.frombuffer(data, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([len(data)]), dtype=torch.uint8, address=0,
        phy_size=len(data), fmt=MemoryFormat.BINARY, ref_count=1)
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def make_empty_obj(size_bytes: int) -> TensorMemoryObj:
    raw = torch.zeros(size_bytes, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([size_bytes]), dtype=torch.uint8, address=0,
        phy_size=size_bytes, fmt=MemoryFormat.BINARY, ref_count=1)
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def split_commands(nbytes, xfer, lba):
    """A logical transfer -> ceil(nbytes/xfer) commands, each rounded up to lba."""
    if nbytes <= 0:
        return []
    n = math.ceil(nbytes / xfer)
    cmds = [xfer] * (n - 1) + [nbytes - xfer * (n - 1)]
    return [((c + lba - 1) // lba) * lba for c in cmds]


def project(payload, mdts, header, lba):
    """Per-op NVMe command geometry (store = header op + payload; load = payload)."""
    store = split_commands(header, mdts, lba) + split_commands(payload, mdts, lba)
    load = split_commands(payload, mdts, lba)
    return {"store_cmds": len(store), "store_bytes": sum(store),
            "load_cmds": len(load), "load_bytes": sum(load)}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--modelconfig",
                    default=os.path.join(here, "..", "kv_cache_calculator",
                                         "modelconfig.json"),
                    help="modelconfig.json (calculator format)")
    ap.add_argument("--model", required=True, help="model name (key in modelconfig.json)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16", "int8", "fp8"])
    ap.add_argument("--chunk-tokens", type=int, default=256,
                    help="tokens per KV chunk = one offloaded block (LMCache default 256)")
    ap.add_argument("--num-chunks", type=int, default=8, help="how many chunks to offload")
    ap.add_argument("--device", required=True, help="/dev/ngXnY (uring_cmd) or a file path")
    ap.add_argument("--engine", choices=["posix", "io_uring", "uring_cmd"],
                    default="uring_cmd")
    ap.add_argument("--mdts-bytes", type=int, default=131072,
                    help="device max transfer size (LMCache max_data_transfer_size)")
    ap.add_argument("--block-align", type=int, default=4096)
    ap.add_argument("--header-bytes", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=1, help="passes over the chunk set")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--odirect", action="store_true")
    ap.add_argument("--capacity-gb", type=int, default=8)
    ap.add_argument("--record", help="write a kvio_record.json replay manifest here")
    ap.add_argument("--trace", help="LMCACHE_KVIO_TRACE path (semantic trace)")
    args = ap.parse_args()

    if args.trace:
        os.environ["LMCACHE_KVIO_TRACE"] = args.trace
        open(args.trace, "w").close()

    with open(args.modelconfig) as f:
        configs = json.load(f)
    if args.model not in configs:
        raise SystemExit(f"{args.model!r} not in {args.modelconfig}")
    block_bytes, detail = kv_cache_bytes(args.model, configs[args.model],
                                         args.chunk_tokens, args.dtype)
    geom = project(block_bytes, args.mdts_bytes, args.header_bytes, args.block_align)

    print(f"=== KV-offload IO: {args.model} ({detail['family']}), dtype={args.dtype} ===")
    print(f"  chunk={args.chunk_tokens} tok -> KV block = {block_bytes} B "
          f"({block_bytes / 1024 / 1024:.2f} MiB)  [{detail['total_elements']} elems]")
    print(f"  per chunk: store {geom['store_cmds']} cmds / {geom['store_bytes']} B, "
          f"load {geom['load_cmds']} cmds / {geom['load_bytes']} B "
          f"(MDTS={args.mdts_bytes // 1024} KiB, align={args.block_align})")
    print(f"  workload: {args.num_chunks} chunks, engine={args.engine}, "
          f"O_DIRECT={'on' if args.odirect else 'off'}, dev={args.device}")

    slot = ((block_bytes + args.header_bytes + (1 << 20) - 1) >> 20) << 20
    io_engine = "posix" if args.engine == "posix" else "io_uring"
    cfg = RawBlockCoreConfig(
        device_path=args.device, capacity_bytes=args.capacity_gb * 1024 * 1024 * 1024,
        block_align=args.block_align, header_bytes=args.header_bytes, slot_bytes=slot,
        use_odirect=args.odirect, enable_zero_copy=False, meta_total_bytes=1 * 1024 * 1024,
        meta_magic=b"LMCIDX01", meta_version=1, meta_checkpoint_interval_sec=60,
        meta_idle_quiet_ms=0, meta_enable_periodic=False, meta_verify_on_load=False,
        max_data_transfer_size=args.mdts_bytes, load_checkpoint_on_init=False,
        io_engine=io_engine, iouring_queue_depth=8,
        use_uring_cmd=(args.engine == "uring_cmd"))
    core = RawBlockCore(cfg, key_namespace="object")

    buf = bytes(block_bytes)  # zeros; geometry is content-free
    store_ms, load_ms = [], []
    for it in range(args.warmup + args.iters):
        # fresh keys per pass so every store is a real write (not an index hit)
        keys = [encode_object_key(ObjectKey(
                    chunk_hash=ObjectKey.IntHash2Bytes(it * args.num_chunks + i),
                    model_name="kvoffload", kv_rank=0))
                for i in range(args.num_chunks)]
        st = [0.0] * args.num_chunks
        for i in range(args.num_chunks):
            t0 = time.perf_counter()
            core.put_many([keys[i]], [make_memory_obj(buf)])
            st[i] = (time.perf_counter() - t0) * 1e3
        for i in range(args.num_chunks):
            t2 = time.perf_counter()
            core.load_many_into([keys[i].encoded], [make_empty_obj(block_bytes)])
            dt = (time.perf_counter() - t2) * 1e3
            if it >= args.warmup:
                store_ms.append(st[i]); load_ms.append(dt)
    try:
        core.close()
    except Exception:
        pass

    def line(name, ms, cmds, tbytes):
        mean = sum(ms) / len(ms)
        print(f"  {name:5s}: p50 {pct(ms, .5):7.3f} ms  p99 {pct(ms, .99):7.3f} ms | "
              f"{(block_bytes / (mean / 1e3)) / 1e6:8.1f} MB/s | "
              f"{cmds / (mean / 1e3):9.0f} NVMe cmd/s")
    print("  --- measured (real device I/O) ---")
    line("store", store_ms, geom["store_cmds"], geom["store_bytes"])
    line("load", load_ms, geom["load_cmds"], geom["load_bytes"])

    if args.record:
        rec = {
            "schema_version": 1,
            "source": f"kv_cache_offload_io: {args.model} on {args.device}",
            "model": args.model, "geometry": detail,
            "device_geometry": {
                "engine": args.engine, "use_uring_cmd": args.engine == "uring_cmd",
                "mdts_bytes": args.mdts_bytes, "block_align": args.block_align,
                "header_bytes": args.header_bytes, "slot_bytes": slot,
                "capacity_bytes": args.capacity_gb * 1024 * 1024 * 1024,
            },
            "access_pattern": "store-all-then-load-all",
            "objects": [{"index": i, "part": "kv", "payload_bytes": block_bytes,
                         "ops": ["store", "load"]} for i in range(args.num_chunks)],
        }
        with open(args.record, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  wrote replay manifest: {args.record}")


if __name__ == "__main__":
    main()
