# SPDX-License-Identifier: Apache-2.0
"""Exercise blk_iobuf_pool from the raw_block engine (opt-in).

kvio's normal path (IORING_OP_URING_CMD passthrough with page-aligned user
buffers) never touches blk_iobuf_pool -- the kernel maps those buffers zero-copy,
so iobuf_pool_allocs stays 0 under off/auto/force. The pool is fed only by
io_uring op 38 (IORING_REGISTER_BUFFERS_ALLOC_FOR_FILE), kernel-allocated folios.

This driver calls the ext's ``iobuf_pool_bench`` helper, which registers a
pool-backed fixed buffer via op 38 on a BLOCK device (op 38 rejects the /dev/ng
char device) and issues ReadFixed/WriteFixed against it. It reads iobuf_pool_allocs
before/after so the engagement is attributable.

    python iobuf_pool_bench.py --blkdev /dev/nvme1n1 --buffer-size 131072 --nr-ops 64
    # or derive the block device from the char device kvio uses:
    python iobuf_pool_bench.py --device /dev/ng1n1 --buffer-size 131072 --nr-ops 64
"""
from __future__ import annotations

import argparse
import os
import re

import lmcache_rust_raw_block_io as ext


def blkdev_from(device: str, blkdev: str | None) -> str:
    if blkdev:
        return blkdev
    # /dev/ng1n1 -> /dev/nvme1n1
    base = os.path.basename(device)
    m = re.fullmatch(r"ng(\d+)n(\d+)", base)
    if not m:
        raise SystemExit(f"cannot derive block device from {device!r}; pass --blkdev")
    return f"/dev/nvme{m.group(1)}n{m.group(2)}"


def sysfs_allocs(blkdev: str) -> tuple[int, int, int]:
    q = f"/sys/block/{os.path.basename(blkdev)}/queue"

    def rd(name, default=0):
        try:
            with open(f"{q}/{name}") as f:
                return int(f.read().strip(), 0)
        except (OSError, ValueError):
            return default

    return rd("iobuf_pool_allocs"), rd("iobuf_pool_fallbacks"), rd("iobuf_pool_enabled")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blkdev", help="block device (e.g. /dev/nvme1n1)")
    ap.add_argument("--device", help="char device kvio uses (e.g. /dev/ng1n1); "
                                     "block device is derived from it")
    ap.add_argument("--buffer-size", type=int, default=131072,
                    help="fixed-buffer size in bytes (>= pool folio_size)")
    ap.add_argument("--nr-ops", type=int, default=64)
    ap.add_argument("--write", action="store_true",
                    help="issue WriteFixed (DESTRUCTIVE); default is ReadFixed")
    ap.add_argument("--queue-depth", type=int, default=8)
    ap.add_argument("--span-bytes", type=int, default=0,
                    help="offset span to rotate over (0 = buffer_size * 1024)")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="let the kernel fall back to page alloc if the pool is depleted")
    ap.add_argument("--only", choices=["a", "b"], default=None,
                    help="run only arm A (pool off) or B (pool on); default runs both")
    args = ap.parse_args()

    if not args.blkdev and not args.device:
        raise SystemExit("pass --blkdev or --device")
    blkdev = blkdev_from(args.device or "", args.blkdev)

    _, _, enabled = sysfs_allocs(blkdev)
    dir_str = "write" if args.write else "read"
    print(f"=== iobuf_pool A/B on {blkdev} (pool_enabled={enabled}) ===")
    print(f"  buffer_size={args.buffer_size} nr_ops={args.nr_ops} dir={dir_str} "
          f"(same io_uring {dir_str.capitalize()}Fixed engine, block device)")

    def arm(use_pool: bool, label: str):
        a0, f0, _ = sysfs_allocs(blkdev)
        ms, ops = ext.iobuf_pool_bench(
            blkdev, args.buffer_size, args.nr_ops, args.write, use_pool,
            args.queue_depth, args.span_bytes, use_pool, args.allow_fallback)
        a1, f1, _ = sysfs_allocs(blkdev)
        opss = ops / (ms / 1e3) if ms else 0.0
        print(f"  [{label}] {ops} ops in {ms:8.3f} ms  {opss:8.0f} ops/s  "
              f"| allocs +{a1 - a0}  fallbacks +{f1 - f0}")
        return ms, opss, a1 - a0

    if args.only in (None, "a"):
        a_ms, a_ops, a_alloc = arm(False, "A pool-OFF (user buffer, op 0) ")
    if args.only in (None, "b"):
        b_ms, b_ops, b_alloc = arm(True, "B pool-ON  (op 38 pool folios) ")

    if args.only is None:
        speed = (a_ms / b_ms) if b_ms else float("nan")
        print(f"  --- A/B: pool-ON (B) is {speed:.2f}x faster than pool-OFF (A); "
              f"pool engaged only in B (A allocs +{a_alloc}, B allocs +{b_alloc}) ---")


if __name__ == "__main__":
    main()
