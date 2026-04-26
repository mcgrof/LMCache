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

## GPU memory snapshot (H100)

Run on a real H100 80GB on 2026-04-25.  The load-bearing test from
the Phase 4 plan: CUDA allocation snapshot during native_asym
decode confirms no FP16 V buffer is materialized.

Configuration: shape (16, 64, 32, 128), 4,194,304 elements, FP16.

```
legitimate_total:   K_fp16 + V_fp8 + scales = 12,582,916 bytes
observed peak:                                12,583,424 bytes
overhead:           +508 bytes  (allocator alignment)
silent-fp16-V:                                20,971,524 bytes
```

Peak / legitimate ratio = **1.000**.  The codec's native_asym
decode allocates exactly K + V_fp8 + scales on the GPU; nothing
extra.  Silent FP16 V materialization would cost an additional
8.4 MB and is not observed.

Also verified on H100:

- CPU and CUDA FP8 quantization produce **bit-identical** FP8
  bytes for the same input.  This is what makes the CPU test
  ladder valid as a stand-in for GPU behavior.
- `codec.decode(device=cuda)` lands tensors on the requested
  device; no silent CPU fallback.

## What is still GPU-pending (vLLM connector integration)

This file's measurements cover the codec layer.  Three more axes
require the LMCache-vLLM connector glue (Phase 4 GPU integration
still to land), and live as follow-up work:

- **Cache-hit TTFT** (p50/p95/p99) for FP16 vs asym all-NVMe vs
  split-tier under W1–W5 workloads.  Needs the connector calling
  the codec on hit.
- **Restore latency per layer** — distinguishes disk-read from
  GPU-copy from dequant.  Needs CUDA timing inside the connector.
- **Quality preservation** vs in-memory asymmetric:
  WikiText-2 PPL on Qwen2.5-7B and NIAH 16K/32K on Qwen2.5-7B and
  Llama-3.1-8B.  Already measured for the in-memory path in
  `prune:/data/knlp-key-results/qwen-fragility-bundled-20260425/`;
  the cache-hit path needs to reproduce those numbers when the
  KV comes from LMCache rather than fresh prefill.

The codec, serde, native_asym mode, capability detection,
AsymKVMemoryObj, AsymKVView, SplitTierStore, byte-counts API, and
the GPU memory-snapshot test are all in place; the remaining work
is connector wiring.
