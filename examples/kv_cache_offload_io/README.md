# KV-Cache Offload IO Workload Generator

A GPU-free companion to the [KV Cache Size Calculator](../kv_cache_calculator).

The calculator tells you *how big* a model's KV cache is. This tells you what
**offloading it to storage actually costs** — the NVMe command pattern and
store/load latency — **without a GPU or a model**.

It takes a real model config, computes the KV-cache byte size for one chunk of
tokens using the calculator's exact geometry (`kv_geometry.py`, a Python port
that shares the same math and model families: MHA/GQA, GQA-with-`head_dim`,
DeepSeek MLA, Hunyuan CLA), then issues that store/load workload against a real
device through LMCache's `raw_block` engine — POSIX, io_uring, or io_uring_cmd
NVMe passthrough — or through a GPU-direct (GDS) data path (`cufile`,
`opends`; see below).

**Any model, not just the catalog.** `--model` accepts either a key in the
calculator's `modelconfig.json` (30 curated models) **or any Hugging Face model
id** — its config is fetched automatically (config JSON only, no weights, no
GPU) and the family is inferred from the config (MLA latent, CLA sharing, or
explicit `head_dim`), so you can size a model the catalog has never seen. For
arbitrary models this needs `transformers` installed; catalog models need
nothing.

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

# Any HF model not in the catalog (config auto-fetched), e.g. a 671B you can't
# run — size its offload IO on real storage with no GPU in the room:
python run_kv_offload_io.py --model deepseek-ai/DeepSeek-V3 --tp 8 \
    --device /dev/ng0n1 --engine uring_cmd --num-chunks 8
```

Key options: `--dtype` (fp16/bf16/int8/fp8), `--chunk-tokens` (tokens per
offloaded block; LMCache default 256), `--num-chunks` (workload size — keep it
small for a compact trace), `--engine`, `--mdts-bytes` (bytes per NVMe
command — LMCache's `max_data_transfer_size`, ≤ the device's MDTS),
`--iters`/`--warmup` (latency sampling), `--record` (write a replay manifest),
`--trace` (fire LMCache's `LMCACHE_KVIO_TRACE` semantic trace).

## GPU-direct engines (GDS)

The kernel engines above land KV in host DRAM. Two more engines move the
*same* byte layout (same slots, same header+payload offsets, same
schema-2 semantic trace) over the GPUDirect-Storage data path instead,
so GDS transports can be compared against io_uring per-object,
apples-to-apples:

```bash
# proprietary: libcufile directly (cudaMalloc buffer, cuFileWrite/Read)
python run_kv_offload_io.py --model meta-llama/Llama-3.1-8B-Instruct \
    --device /mnt/nvme/kv.img --engine cufile --trace /tmp/cufile.jsonl

# open API: OpenDS (github.com/xnvme/opends). Backend variants are
# separate .so files with one ABI, so --gds-backend X loads
# libopends_X.so: 'gds' wraps cuFile (GPU memory), 'ref' is the POSIX
# reference (host memory — runs with no GPU at all), and any future
# variant (e.g. aisio) works unmodified.
python run_kv_offload_io.py --model meta-llama/Llama-3.1-8B-Instruct \
    --device /mnt/nvme/kv.img --engine opends --gds-backend gds
python run_kv_offload_io.py ... --engine opends --gds-backend ref  # no GPU
```

Buffers for `opends` come from `opends_alloc`, so the backend picks the
memory class (GPU for `gds`, host for `ref`) — the generator never
touches CUDA for that engine. `--gds-lib-dir` (or `OPENDS_LIB_DIR`)
points at the OpenDS build directory. GDS engines force O_DIRECT and
need a file on a filesystem (not a raw char device).

Measured on an H100 + Micron 7450 (32 MiB objects, single stream):
`cufile` and `opends:gds` track within a few percent (load p50 ~19-21 ms
= 1.6-1.7 GiB/s), device-level confirmation that the open wrapper adds
nothing to the data path.

> cuFile compat-path footgun (GDS 1.18.1.6): if a process's *first*
> cuFile read is smaller than the 1 MiB bounce-pool slab class — or its
> first op is a write — every read after a write is chopped into 4 KiB
> posix reads for the process lifetime (~47x slower). The engine primes
> the pool with one >=1 MiB read at init; see `gds_engine.py`.

## Whole workloads, not one request

`run_kv_offload_io.py` issues one request's worth of chunks. `kvio_workload.py`
replays a *distribution* of request sizes with a store/load mix set by the cache
hit rate — the shape of real serving traffic:

```bash
# synthetic lognormal traffic (200 requests, median 1K tokens, 60% cache hits)
python kvio_workload.py --model meta-llama/Llama-3.1-8B-Instruct \
    --device /dev/ng0n1 --engine uring_cmd \
    --num-requests 200 --median-tokens 1024 --hit-rate 0.6

# or drive it from a real request trace (JSONL; token count per line)
python kvio_workload.py --model Qwen/Qwen3-8B --device /dev/ng0n1 \
    --trace requests.jsonl --hit-rate 0.7
```

It reports the request-size profile, total KV volume, and store/load p50/p99 +
aggregate GiB/s under the realistic mix. Each object is byte-identical to the
single-request generator's — only the *mix* is new.

## Output

Per-chunk NVMe-command geometry (store = header op + payload op; load = payload)
and measured store/load p50/p99 latency + throughput on the real device. With
`--record`, a `kvio_record.json` manifest carrying the model provenance and the
device geometry, so the exact workload can be replayed later.

> Note: this issues fake KV blocks sized from real geometry; it does not yet
> model per-layer object granularity or asymmetric K/V quantization (one block =
> one chunk across all layers). Those are natural follow-ups.
