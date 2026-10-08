# Raw-block storage P/D proof of concept

This example runs one vLLM prefiller and one decoder against one dedicated raw
NVMe namespace or one initialized XFS/ext4 file on NVMe. Payload I/O is ordinary `io_uring` with O_DIRECT and
registered GPU dma-bufs. Nothing here uses NIXL.

There are two ways to select this route. `pd_data_path=raw_block` says it
through LMCache's own prefill/decode switch: `enable_pd=true` with that data
path skips the transfer channel entirely, derives the storage role from
`pd_role`, and requires the `raw_block` plugin rather than treating it as an
optional cache tier. The settings that describe the transfer channel, its peer
addresses and buffers, are not needed and not asked for. The older spelling
still works, where `enable_pd` stays off and the plugin settings carry
everything; `launch_vllm.sh` uses the newer one and honours
`LMCACHE_PD_DATA_PATH` if you want the other.

## What releases a published extent, and what a restart means

The writer locks every extent it publishes and releases it only when the
decoder that was told about it says the bytes reached GPU memory. That
acknowledgement goes straight from the decoder to the address the writer
advertises in its READY status, as a bounded request whose reply the writer
sends only after the release has actually happened. The proxy carries the READY
barrier and nothing else: whether a hold is gone is a fact about the engine
holding it, and no relay can state it.

Two consequences are worth stating before a long run.

A decoder that cannot reach its writer keeps asking, in the background, with no
further request needed. Its obligations have one absolute deadline each, and
reaching that deadline is reported as unresolved -- never as a release. The
writer therefore keeps those extents, and it stops accepting new publications
once its unacknowledged holds reach their bound rather than abandoning one to
make room. A run that ends with `refusing to publish` in the writer's log is a
run whose decoder stopped acknowledging; the namespace is intact and the holds
are still held.

Restarting is a whole-group operation. `LMCACHE_STORAGE_PD_SESSION` names one
producer/consumer pair's run and both nodes must carry the same value. The
writer binds the first decoder incarnation that acknowledges under a session
and refuses any other incarnation for it, because a decoder that came back
alone cannot have performed the reads it would be claiming -- its extents may
still be being read by nothing at all, or the writer's index may name bytes the
new process never received. So: to restart, stop both nodes, change the session
value, and start both. Bringing the decoder back by itself is refused, loudly,
and the writer keeps its holds.

This is deliberately not crash recovery. Nothing here reconstructs which
extents an exited decoder had finished with, and nothing persists a hold across
a writer restart: a writer that exits leaves its locks only in memory, and the
next writer rebuilds its free list from the committed index. Durable holds and
recovery after an unplanned exit are separate work and are not attempted.

## Safety and prerequisites

The namespace is destructive. `launch_vllm.sh` accepts only a persistent
`/dev/disk/by-id/...` path, and rejects partitions, mounted devices, child block
devices, holders, and recognized filesystem or RAID signatures. Review the
resolved device yourself, then confirm the exact path:

```bash
export LMCACHE_RAW_DEVICE=/dev/disk/by-id/nvme-REPLACE_WITH_DEDICATED_NAMESPACE
export LMCACHE_CONFIRM_RAW_DEVICE_ERASE=$LMCACHE_RAW_DEVICE
```

Those checks establish that a device is not in use. They cannot establish that
its contents are expendable, and the difference matters: a reference drive kept
untouched for comparison is blank, unmounted, unheld and free of signatures, so
it passes every one of them. List anything that must never be written in
`/etc/lmcache/protected-devices`, one entry per line, each a `by-id` path, a
serial, or a WWID, with `#` starting a comment. The preflight refuses a listed
device before any other check, and refuses to run at all if
`LMCACHE_PROTECTED_DEVICES_FILE` names a file it cannot read.

The kernel must provide dma-buf-backed io_uring fixed-buffer registration, and
the GPU driver must export the staging allocation as dma-buf. Initialization
fails closed if that registration is unavailable; there is no host-buffer
fallback in this mode. The launcher marks the backend required
(`storage_plugin.raw_block.required`), so a device that cannot be opened, a
registration that is refused, or any other construction failure stops startup
instead of leaving an engine that serves without the storage it was configured
to hand off through. Size `LMCACHE_GPU_BUFFER_BYTES` for the largest
simultaneous batch (the default is 4 GiB).

## Launch

Choose one session value and export that same value in the proxy, writer, and
reader shells. From this directory, start the proxy first:

```bash
export LMCACHE_STORAGE_PD_SESSION=qualification-REPLACE_WITH_UNIQUE_RUN_ID
../../../.venv/bin/python ../disagg_proxy_server.py \
  --host localhost --port 9100 \
  --prefiller-host localhost --prefiller-port 7100 --num-prefillers 1 \
  --decoder-host localhost --decoder-port 7200 \
  --decoder-init-port 7300 --decoder-alloc-port 7400 --num-decoders 1 \
  --proxy-host localhost --proxy-port 7500 --model MODEL \
  --storage-pd --storage-pd-session "$LMCACHE_STORAGE_PD_SESSION" \
  --storage-pd-ready-timeout-s 30
```

Then launch the reader before the writer on separate GPUs, with the same
session value on both:

```bash
export LMCACHE_STORAGE_PD_SESSION=qualification-REPLACE_WITH_UNIQUE_RUN_ID
DECODER_DEVICE_ID=1 ./launch_vllm.sh reader MODEL
PREFILLER_DEVICE_ID=0 ./launch_vllm.sh writer MODEL
```

The launcher refuses to start without that value rather than inventing one per
process, which would make every pair a mismatched session. The writer answers
acknowledgements on `LMCACHE_RAW_ACK_PORT` (7600 by default), bound on
`LMCACHE_RAW_ACK_BIND_HOST` and advertised as `LMCACHE_RAW_ACK_ADVERTISE_HOST`;
the two are separate because a wildcard bind is not an address a decoder can
reply to. Extent reuse is required in this configuration, so a port that cannot
be bound stops startup instead of producing a writer that silently never
reclaims.

The launcher disables vLLM prefix caching, preserves the first-token
continuation contract, saves the final partial chunk, and requires strict
dma-buf registration. A valid correctness run still needs a fresh decoder or
deliberately overwritten KV blocks and comparison with a no-LMCache oracle.

## Strict filesystem configuration

Select an existing, exclusively assigned regular file with `LMCACHE_RAW_FILE`
instead of `LMCACHE_RAW_DEVICE`. Confirm that its cache area is expendable:

```bash
export LMCACHE_RAW_FILE=/absolute/path/to/disposable-cache.bin
export LMCACHE_CONFIRM_FILE_OVERWRITE=$LMCACHE_RAW_FILE
export LMCACHE_RAW_CAPACITY_BYTES=285212672
export LMCACHE_RAW_SLOT_BYTES=9437184
export LMCACHE_GPU_BUFFER_BYTES=268435456
export LMCACHE_GPU_BUFFER_PROVIDER=vulkan_rm  # optional RTX provider; see build docs
```

The example capacity is 128 MiB of default metadata plus sixteen 9 MiB slots.
Each slot includes its 4 KiB header; it fits the 8 MiB full BF16 KV chunk of
Llama-3.2-1B-Instruct at 256 tokens and TP=1. Recompute these values for another
model or tensor-parallel shard. The same file/inode must be visible to both
roles. This local-NVMe profile is not a network-filesystem configuration.

Create a **new** file and initialize every byte before launching. For example:

```python
import os

path = "/absolute/path/to/disposable-cache.bin"
capacity = 285212672
fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
try:
    os.posix_fallocate(fd, 0, capacity)
    zeros = bytes(4 * 1024 * 1024)
    for offset in range(0, capacity, len(zeros)):
        data = zeros[:min(len(zeros), capacity - offset)]
        if os.pwrite(fd, data, offset) != len(data):
            raise RuntimeError("short file initialization")
    os.fdatasync(fd)
finally:
    os.close(fd)
```

Strict startup validates the actual opened descriptor and rejects holes,
unwritten space, reflinks, unsupported filesystems and capacity overflow.
DMA-BUF registration remains mandatory. Do not enable
`allow_unsafe_pd_io_for_testing` for this configuration. The file must remain
private and unchanged by other applications while the group runs.

For exact probability comparisons, qualify an explicit vLLM numerical profile.
`VLLM_BATCH_INVARIANT=1` makes the tested BF16 eager text workload reproducible;
without it, no-cache repeats can differ too. Report that selection and compare
all outputs against the same no-cache profile without increasing tolerances.
This does not qualify other models, kernels or serving configurations.

## Causal evidence for reuse

The optional timeline includes committed payload generations and their file
offsets. Publication provenance binds those records to the manifest digest.
The checker joins every range to completed writes from its original request,
including when a later publication reuses it. A later overwrite makes the old
generation invalid. Recovered data without its original trace remains unproved.

A restore with no read CQE is accepted only when the adapter records that every
published token is already resident and no token was restored. Exact publication
adoption and the complete claim/ACK/decode chain are still required. This is
zero storage work, not fabricated DMA-BUF traffic. Journal and request-ledger
checks remain necessary to establish that the trace is complete.
