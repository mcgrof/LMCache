# Raw-block storage P/D proof of concept

This example runs one vLLM prefiller and one decoder against one dedicated raw
NVMe namespace. Payload I/O is ordinary block `io_uring` with O_DIRECT and
registered GPU dma-bufs. It is not the LMCache `enable_pd=true` NIXL path.

This proof retains every published extent until the writer exits. That prevents
reuse during a decoder read, but it also means the namespace must not wrap and
long-running service use is unsupported. ACK-driven reclamation is future work.

## Safety and prerequisites

The namespace is destructive. `launch_vllm.sh` accepts only a persistent
`/dev/disk/by-id/...` path, and rejects partitions, mounted devices, child block
devices, holders, and recognized filesystem or RAID signatures. Review the
resolved device yourself, then confirm the exact path:

```bash
export LMCACHE_RAW_DEVICE=/dev/disk/by-id/nvme-REPLACE_WITH_DEDICATED_NAMESPACE
export LMCACHE_CONFIRM_RAW_DEVICE_ERASE=$LMCACHE_RAW_DEVICE
```

The kernel must provide dma-buf-backed io_uring fixed-buffer registration, and
the GPU driver must export the staging allocation as dma-buf. Initialization
fails closed if that registration is unavailable; there is no host-buffer
fallback in this mode. Size `LMCACHE_GPU_BUFFER_BYTES` for the largest
simultaneous batch (the default is 4 GiB).

## Launch

From this directory, start the proxy first:

```bash
../../../.venv/bin/python ../disagg_proxy_server.py \
  --host localhost --port 9100 \
  --prefiller-host localhost --prefiller-port 7100 --num-prefillers 1 \
  --decoder-host localhost --decoder-port 7200 \
  --decoder-init-port 7300 --decoder-alloc-port 7400 --num-decoders 1 \
  --proxy-host localhost --proxy-port 7500 --model MODEL \
  --storage-pd --storage-pd-ready-timeout-s 30
```

Then launch the reader before the writer on separate GPUs:

```bash
DECODER_DEVICE_ID=1 ./launch_vllm.sh reader MODEL
PREFILLER_DEVICE_ID=0 ./launch_vllm.sh writer MODEL
```

The launcher disables vLLM prefix caching, preserves the first-token
continuation contract, saves the final partial chunk, and requires strict
dma-buf registration. A valid correctness run still needs a fresh decoder or
deliberately overwritten KV blocks and comparison with a no-LMCache oracle.
