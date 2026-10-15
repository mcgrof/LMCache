# Raw Block L2 Adapter Design

The adapter implements POSIX and ordinary Rust io_uring storage I/O. MP
controllers hold exact-buffer L1 leases until completion returns ownership.
Unknown completion seals and retains the affinity arena. Eager host L1 pools
provide fixed-buffer or DMA-BUF regions; eager GPU L1 pools provide DMA-BUF
regions. The serde wrapper holds separate leases on encoded intermediate
buffers and propagates codec and storage ownership outcomes. GPU affinity
supports plain payloads or FP8 serde on ordinary io_uring. NVMe passthrough
and FDP require their own hardware qualification.

This document describes the built-in `raw_block` L2 adapter for LMCache MP
mode. It covers the adapter shape, the shared raw-block core, and the recovery
model.

## Overview

`raw_block` is a persistent MP L2 adapter backed by a raw block device or a
dedicated file. It is designed to keep the MP request flow unchanged while
reusing the existing raw-block on-device metadata format and the low-level Rust
raw-device I/O path.

```text
StoreController / PrefetchController
                |
                v
        RawBlockL2Adapter
                |
                v
           RawBlockCore
      (index, locks, slots, checkpoints)
                |
                v
         lmcache_rust_raw_block_io
      (POSIX / ordinary io_uring / NVMe passthrough)
                |
                v
         raw block device / file
```

## Goals

- Support LMCache MP mode using raw block storage as an L2 cache.
- Reuse the same durable metadata and checkpoint model as the existing
  non-MP raw-block backend.
- Reuse the existing Rust raw-device I/O layer.
- Preserve restart recovery semantics.
- Keep the MP controller flow unchanged: store, lookup-and-lock, load, unlock.

## FDP Placement Base

The MP `raw_block` adapter supports NVMe Flexible Data Placement (FDP) status
discovery when `io_engine="io_uring"` and `use_uring_cmd=true`. At startup, it
queries FDP reclaim unit handle status from the device and reports the
discovered mapping. Startup fails if the query fails or the device reports no
placement identifiers.

FDP plumbing is split by layer: `RawBlockL2Adapter` discovers and registers
non-zero placement identifiers, while `RawBlockCore` enforces that explicit
identifier 0 is never used. `fdp_placement_ids` is the KV data placement pool:
explicit data identifiers are rejected if they overlap with
`meta_checkpoint_placement_id` or are not reported by the device. If
`fdp_placement_ids` is omitted, the adapter registers all device-reported
non-zero identifiers except the metadata checkpoint identifier.

The adapter maps KV data writes onto FDP placement identifiers with a
cache-salt prefix policy. It derives a case-insensitive bucket from the part of
`ObjectKey.cache_salt` before `:` only when the separator is present, assigns
buckets to placement identifiers in first-seen order, and reuses the same
identifier for later writes in that bucket. Values without `:` or with an empty
prefix omit the directive; `rag:` is a valid opt-in to the `rag` bucket. The
mapping is availability-first: if the number of buckets exceeds the number of
registered data placement identifiers, extra buckets fall back to no directive
and the adapter emits one warning. Status reporting tracks a fallback count and
a bounded bucket sample rather than retaining all fallback bucket names. Empty
`cache_salt` values also omit the directive. The mapping is process-local and
may change after restart; read correctness is unaffected because `cache_salt` is
part of the object key rather than the read path's FDP directive. Metadata
checkpoint writes can use an explicit configured placement identifier while
defaulting to no directive when unset. User-facing FDP configuration rules live in
`docs/source/mp/l2_storage/raw_block.rst`; low-level NVMe command encoding
details live in `rust/raw_block/README.md`.

When `pid_affinity` slot reuse is enabled, `RawBlockCore` tracks each free
slot's latest placement identifier in memory and prefers a matching slot during
reuse. If no matching slot is available, it falls back to another free slot or
allocates a new slot.

Slot affinity is not checkpointed because FDP placement assignments are
process-local. After recovery, free slots have no recorded affinity until they
are reused.

## Key Design Choice

The implementation is split into:

- `RawBlockCore` in `lmcache/v1/storage_backend/raw_block/`
- `RawBlockL2Adapter` in `lmcache/v1/distributed/l2_adapters/`
- `RustRawBlockBackend` as the legacy non-MP wrapper

`RawBlockCore` owns the durable state and blocking I/O:

- raw device open/close
- in-memory key index
- free-slot tracking
- lock refcounts used by MP lookup/load/unlock
- metadata checkpointing and recovery
- direct reads and writes through the Rust binding

This avoids maintaining separate raw-block implementations for MP and non-MP
mode.

## Adapter Contract

`RawBlockL2Adapter` implements `L2AdapterInterface` directly. It exposes:

- three distinct eventfds: store, lookup, load
- non-blocking task submission APIs
- worker-thread execution for blocking raw-device operations
- result maps keyed by adapter-local task id
- listener notifications for stored, accessed, and deleted keys

The adapter uses caller-provided `MemoryObj` buffers for load operations. It
does not allocate destination buffers on the load path.

## Worker Failure and Task Completion

Recoverable io_uring submission errors (`EAGAIN`, `EINTR`, and `EBUSY`) retry
after completion progress or exponential backoff from 1 to 100 ms. Permanent
errors, or 30 seconds without submission or completion progress, stop new
native I/O admission. `RawBlockCore.report_status()` then reports
`is_healthy=False` and the terminal reason in `worker_error`. An ordinary
per-request I/O error does not mark the worker as failed.

An open adapter reports a known worker failure through its task results, not
through a synchronous exception from `submit_*`. It allocates a task ID and
passes an already-failed future to the existing completion callback without
queuing work in the thread pool:

- lookup and load complete with all-zero bitmaps of the submitted key count;
- store completes with `L2StoreResult(False, 0)` without writing data;
- each completion decrements its in-flight counter, publishes the result, and
  signals the corresponding eventfd. Results are consumed once.

A known worker failure that admits no I/O returns reusable buffers. Native
poison instead means that completion is unknown. The adapter returns that
verdict separately from the store status or load bitmap through
`pop_completed_store_outcomes` and `query_load_outcome`. Legacy result APIs
refuse unknown outcomes without consuming them. A zero bitmap is not a buffer
ownership proof.

Store and prefetch acquire an L1 lease before submitting each batch. The lease
pins exact objects independently of reservation timeouts, eviction or forced
delete. L1 bookkeeping runs before the lease is settled. Known completion
returns deferred frees to the allocator; unknown completion retains the arena,
objects and any external-memory context for the process lifetime. Loads with
unknown completion publish no hits and keep their L2 source locks.

Shutdown stops and joins controller threads before closing adapters. A cached
`L2CloseOutcome` carries the core's native quiescence verdict; repeated close
does not invent new proof. Controllers then settle pending leases and L1 closes
only reusable arenas. A controller-only stop has no adapter-wide proof, so it
retains any buffers whose completion is still pending. Partial initialization
and runtime adapter removal also retain an affinity arena when close fails or
explicitly withholds ownership.

MP supports eager host and GPU L1 buffers. Its `buffer_registration` mode describes
stable regions in `L1MemoryDesc`: classic `fixed` regions use host pointers;
`dmabuf` regions also carry borrowed export FDs and their mapping origins.
The allocator keeps every object's physical extent within one 1 GiB window,
including after free-space coalescing. UDMABUF exports split at those windows
while retaining one contiguous backing memfd mapping. The L1 owner closes all
exports together after adapter shutdown proves reuse.

`RawBlockL2Adapter` registers the descriptor with its ring before accepting
tasks. DMA-BUF mode enables strict filesystem validation and requires all
exports; no registration fallback is allowed. Partial setup returns an
`L2AdapterInitializationError` when native cleanup cannot prove release.
The factory retains the affinity L1 and the failed adapter retains its native
handle for the process lifetime. Immutable task contexts join native operations
to their MP task IDs so diagnostics can distinguish fixed I/O from fallback.

Host DMA-BUF registration uses UDMABUF exports; GPU registration uses the
configured GPU exporter. GPU affinity supports plain payloads and FP8 serde.
The serde wrapper allocates encoded intermediates from the affinity L1 pool
and leases them through codec and storage completion. Unknown completion
retains these buffers with their arena. The fault wrapper propagates ownership
along with its modified load bitmap.

## Locking Model

LMCache MP already uses L1 locks for CPU-memory object lifetime. `raw_block`
adds a separate L2-side lock refcount so a looked-up key cannot be deleted
between `lookup_and_lock` and `load`.

Rules:

- `exists_many(..., lock=True)` increments the refcount for hits
- `unlock_many(keys)` decrements and floors at zero
- `delete(keys)` skips locked entries

## Persistence and Recovery

`RawBlockCore` keeps the existing metadata checkpoint model:

- metadata region reserved on the same device
- periodic checkpointing
- optional checkpoint load on startup
- optional verification on load
- recovery by loading the latest durable checkpoint and rebuilding the in-memory
  index
- POSIX recovery validates per-slot headers with an internal pool of 8 reader
  threads. Both regular `io_uring` and `io_uring_cmd` batch header reads up to
  `iouring_queue_depth`.

The on-device format is intentionally unchanged by the MP adapter work.

Recovered keys are exposed to the shared L2 eviction policy on adapter startup,
so reclaimed slots come from global L2 eviction or explicit `delete()` calls.

## Configuration

The MP adapter is configured through `--l2-adapter` JSON:

```json
{
  "type": "raw_block",
  "device_path": "/dev/nvme0n1",
  "slot_bytes": 1048576,
  "capacity_bytes": 0,
  "use_odirect": true,
  "block_align": 4096,
  "header_bytes": 4096,
  "meta_total_bytes": 268435456,
  "meta_magic": "LMCIDX01",
  "meta_version": 1,
  "meta_checkpoint_interval_sec": 60,
  "meta_enable_periodic": true,
  "load_checkpoint_on_init": true,
  "meta_verify_on_load": true,
  "num_store_workers": 2,
  "num_lookup_workers": 1,
  "num_load_workers": 4
}
```

For FDP configuration examples, see `docs/source/mp/l2_storage/raw_block.rst`.

Important validation rules:

- `block_align` must be a power of two
- `slot_bytes`, `header_bytes`, and `meta_total_bytes` must be aligned to
  `block_align`
- with `use_uring_cmd=true`, `block_align` must be a multiple of the NVMe
  namespace LBA size
- `slot_bytes >= header_bytes + 1`
- `per_tp_device_paths` is rejected in MP mode
- `load_checkpoint_on_init=false` starts with an empty in-memory index instead
  of loading the latest on-device metadata checkpoint
- with `use_odirect=true` or `use_uring_cmd=true`, MP L1 alignment must satisfy
  `l1_align_bytes >= block_align`
- with `use_odirect=true`, raw-block I/O rejects offsets and total I/O lengths
  that are not aligned to `block_align`; misaligned write buffers use an
  aligned bounce buffer

## Relationship to Non-MP Mode

The legacy `RustRawBlockBackend` now acts as a thin facade over `RawBlockCore`.
It preserves non-MP behavior such as prefix-oriented contains/get semantics,
while the MP adapter uses the core's full-bitmap lookup/load API.

## References

- Implementation: `lmcache/v1/distributed/l2_adapters/raw_block_l2_adapter.py`
- Shared core: `lmcache/v1/storage_backend/raw_block/core.py`
- User docs: `docs/source/mp/l2_storage/raw_block.rst`
- Rust device layer: `rust/raw_block/README.md`

## Read-only checkpoint consumers

The adapter's `role` selects the shared core's storage ownership. Writers
allocate slots and checkpoint metadata. Readers open the target read-only,
verify slot headers on loads, reject stores and deletes, and skip periodic
and close-time checkpoint writes. They do not participate in global L2
eviction. The adapter loads its reader checkpoint at startup; it does not
add a request publication/acknowledgement protocol or refresh an index
while requests run. External handoff coordination must preserve source
extents and pair the checkpoint with compatible key and model metadata.

## Independent MP storage owners

`pd_derivation` opts the adapter into `RawBlockMPHandoff`. The descriptor comes
from the MP BLAKE3 hasher and records chunk size and chain root. The HTTP control
surface checks it against the running server, and core checkpoint adoption
checks it against storage. An ordinary reader role still loads only its startup
checkpoint unless the caller uses the opt-in adoption control.

The controller reuses `RawBlockPDRequestTracker`, not a second lease algorithm:
prepare protects all keys before model stores; publish observes all committed
metadata, then the tracker pins extents before the forced checkpoint. READY
therefore follows payload completion and both persistence barriers. Repeated
publish calls share the original future and receipt. Exact claimed ACK releases
the hold; wrong epoch, manifest, session or consumer cannot do so. Deletion and
capacity eviction use the existing core lock references. Reads use the reader's
own registered arena and verify slot headers. Adoption also checks every
committed chunk's tensor shape, dtype and logical bytes against the reader's
registered canonical layout before lookup. Equal byte counts alone do not
make fused/split K/V or different dtypes interchangeable.

The `/storage-pd/*` handlers expose these operations through public adapter and
storage-manager APIs. Model/token addressing is initially restricted to TP=PP=1
and one full-attention group. The caller supplies dispatch and ACK ordering.
Prepared/unacknowledged requests are bounded; failed reads deliberately retain
leases. Shutdown drains the publication executor before closing core resources.
One writer is required; writer restart with live old readers, automatic dead
reader fencing, a router, and crash-safe lease recovery are not implemented.

Prepared cancellation uses the tracker's terminal future, preventing READY
while an in-flight publisher retains its pins until settlement. A completed
publication cannot use this shortcut. Exact unread release reuses the core
tracker's no-claim check; it cannot release a reader's claimed extents. These
paths retire request admission without treating timeouts as proof of safety.
