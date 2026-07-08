# KV-Cache Offload IO Workload Generator

A GPU-free companion to the [KV Cache Size Calculator](../kv_cache_calculator).

The calculator tells you *how big* a model's KV cache is. This tells you what
**offloading it to storage actually costs** — the NVMe command pattern and
store/load latency — **without a GPU or a model**.

It takes a real model config (the calculator's `modelconfig.json`), computes the
KV-cache byte size for one chunk of tokens using the calculator's exact geometry
(`kv_geometry.py`, a Python port that shares the same math and model families:
MHA/GQA, GQA-with-`head_dim`, DeepSeek MLA, Hunyuan CLA), then issues that
store/load workload against a real device through LMCache's `raw_block` engine —
POSIX, io_uring, or io_uring_cmd NVMe passthrough.

**Why fake KV bytes are enough:** storage IO geometry (command count, sizes,
total bytes) depends only on the block size and the device's transfer limit
(`max_data_transfer_size`), not on the tensor values. So *real model dimensions*
+ *fake content* reproduce the real offload IO pattern. A GPU is only needed to
capture real *access patterns/timing* (which chunk, when, hit vs miss) — the IO
geometry is fully determined here.

## Usage

```bash
# Real NVMe passthrough (needs an NVMe char device, e.g. an empty /dev/ng0n1):
python run_kv_offload_io.py \
    --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 \
    --chunk-tokens 256 --num-chunks 8 \
    --device /dev/ng0n1 --engine uring_cmd \
    --record /tmp/kvio_record.json

# Or against a regular file (no passthrough), for a quick local try:
python run_kv_offload_io.py --model Qwen/Qwen3-32B --device /tmp/l2.bin \
    --engine io_uring --num-chunks 4
```

Key options: `--dtype` (fp16/bf16/int8/fp8), `--chunk-tokens` (tokens per
offloaded block; LMCache default 256), `--num-chunks` (workload size — keep it
small for a compact trace), `--engine`, `--mdts-bytes` (device transfer limit),
`--iters`/`--warmup` (latency sampling), `--record` (write a replay manifest),
`--trace` (fire LMCache's `LMCACHE_KVIO_TRACE` semantic trace).

## Output

Per-chunk NVMe-command geometry (store = header op + payload op; load = payload)
and measured store/load p50/p99 latency + throughput on the real device. With
`--record`, a `kvio_record.json` manifest carrying the model provenance and the
device geometry, so the exact workload can be replayed later.

> Note: this issues fake KV blocks sized from real geometry; it does not yet
> model per-layer object granularity or asymmetric K/V quantization (one block =
> one chunk across all layers). Those are natural follow-ups.
