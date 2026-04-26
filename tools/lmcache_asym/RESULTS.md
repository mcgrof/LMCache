# LMCache asymmetric KV — Phase 6 results (CPU-only)

These are the byte-accounting numbers the paper would table.  Run
on CPU with `python3 tools/lmcache_asym/bench_codec_bytes.py`;
output JSON in `bench_codec_bytes.json`.

## Storage compression ratio

Encoded blob size / FP16-baseline K+V byte count, across 32
configurations (n_pages ∈ {4, 16}, page_size ∈ {16, 32}, n_heads ∈
{8, 32}, head_dim ∈ {64, 128}, dtype ∈ {fp16, bf16}):

```
mean   = 0.750
median = 0.750
range  ≈ [0.748, 0.752]
```

This is the paper's "asymmetric K16/V8 is 75% of FP16" claim,
expressed as a unit-test-grade measurement.  The spread is < 1%
because the variation is just header-overhead amortization across
chunk sizes.

## Split-tier NVMe traffic ratio

NVMe bytes written / read on a cache hit, split-tier vs ALL_NVME:

```
mean   = 0.333
median = 0.333
range  ≈ [0.330, 0.336]
```

The paper's killer-table claim is `0.25× of fp16 NVMe = 1/3 of asym
all-NVMe`.  This is the 1/3 number, expressed as a measurement.

Translated to bytes-per-element-pair:
- FP16 all-NVMe baseline: 4 B/elem-pair (K=2, V=2)
- Asym all-NVMe:          3 B/elem-pair (K=2, V=1)
- Asym split-tier:        1 B/elem-pair (V only on NVMe; K from CPU)

## Where these numbers come from

`bench_codec_bytes.py` instantiates the codec, encodes, then
instantiates a `SplitTierStore` under both `ALL_NVME` and
`SPLIT_K_CPU_V_NVME` policies and writes a chunk through each.
The byte counts are the `SplitTierByteCounts` returned by
`store.put(...)`, which attribute writes to `nvme_bytes` /
`cpu_bytes` / `meta_bytes` separately.  Same accounting on the
read path (`store.get(...)` returns the read counts).

Reproduction:

```bash
cd /path/to/lmcache
PYTHONPATH=. python3 tools/lmcache_asym/bench_codec_bytes.py \
    --out bench_codec_bytes.json
cat bench_codec_bytes.json | jq '.storage_ratio_vs_fp16, .split_vs_all_nvme_ratio'
```

## Phases not measured here

This file covers the byte-accounting metrics that are decided at
the codec / placement layer.  Three more measurement axes need
GPU + a real model and live in the Phase 6 GPU follow-up:

- **Cache-hit TTFT** (p50/p95/p99) for FP16 vs asym all-NVMe vs
  split-tier under W1–W5 workloads.  Needs vLLM + a real model.
- **Restore latency per layer** — distinguishes disk-read from
  GPU-copy from dequant.  Needs CUDA timing.
- **Quality preservation** vs in-memory asymmetric:
  WikiText-2 PPL on Qwen2.5-7B and NIAH 16K/32K on Qwen2.5-7B and
  Llama-3.1-8B.  Already measured for the in-memory path in
  `prune:/data/knlp-key-results/qwen-fragility-bundled-20260425/`;
  the cache-hit path needs to reproduce those numbers when the
  KV comes from LMCache rather than fresh prefill.
