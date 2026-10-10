# Serde intermediate-buffer ownership

`SerdeL2AdapterWrapper` presents an ordinary L2 adapter while transforming
objects before storage and after reload. For example, the FP8 codec encodes
BF16 KV tensors into byte buffers before writing them and restores BF16 tensors
after reading them. The codec interfaces live in
[the serde design](../serde/README.md). This document describes the implemented
wrapper and its ownership contracts.

The caller consumes only the wrapper's event FDs and task outcomes. A private
poll thread consumes the codec and inner adapter event FDs. Lookup, unlock,
eviction, usage and listener operations pass through to the inner adapter.
Usage reports the inner adapter's storage charges, including its slot overhead.

## Data flow

Store:

1. Reserve temporary byte buffers in the adapter's L1 affinity arena and
   acquire an external I/O lease on those exact objects.
2. Submit serialization from the caller's read-locked KV objects into the
   write-locked intermediates. Narrow each intermediate's logical size to the
   codec's actual encoded length.
3. Consume the codec outcome. On known success, atomically change temporary
   write locks into read locks and verify each returned object is the original
   leased allocation. Submit these encoded objects to the inner storage adapter.
4. Consume the storage outcome. Release temporary locks and settle their lease
   with the ownership proof. Publish the wrapper's store outcome.

Load:

1. Reserve and lease temporary byte buffers in the same affinity arena.
2. Submit a storage read into the intermediates and consume its outcome.
3. Decode only keys with successful reads and known buffer ownership. Keep
   intermediates write-locked and leased while the codec writes the caller's
   reserved destination objects.
4. Consume the codec outcome. Discard intermediates and settle their lease
   with the ownership proof. Publish the retained per-key hit bitmap, or no
   hits if decoding failed or ownership is unknown.

The outer store and prefetch controllers own the original source/destination
leases. The wrapper owns the separate intermediate lease from allocation until
the final codec or storage access ends. GPU gather/scatter holds another lease
until its producer stream completion callback runs.

## Completion proof

`L2StoreOutcome`, `L2LoadOutcome` and `SerdeTaskOutcome` carry success separately
from `buffers_reusable`. Success alone does not permit recycling memory.
`AsyncSerdeProcessor` fences its worker's current stream on each participating
GPU before publishing completion, including after a codec exception. A codec
that uses its own stream must drain that stream before returning or raising.
Legacy processors must also stop all accesses before reporting a result or
returning from `close()`.

Unknown storage completion skips decoding. Unknown codec completion reports no
successful store or load. Query or submission failures that lose proof of drain
also return false ownership. Cleanup releases L1 locks before settling the
lease; false ownership seals and retains the whole affinity arena, mappings and
export FDs for the process lifetime. Force-delete and clear cannot recycle a
leased object. This policy also covers expired temporary reservations and
partial write-to-read transitions.

Callers consume `pop_completed_store_outcomes()` and `query_load_outcome()`.
Legacy result-only methods raise on unknown ownership without consuming it.
The wrapper may report unresolved access even when its inner adapter never does,
because a codec's GPU fence can fail.

## Admission and shutdown

An admission lock serializes submission against shutdown. `close_with_outcome()`
stops admission, joins the poll thread, drains codec workers and closes inner
storage before settling pending intermediate leases. A prior unknown outcome
remains irreversible even if a later close proves quiescence. Repeated close
calls return the same outcome. New submissions after close raise.

An allocation failure before any access is immediately reusable. Store failure
applies to the whole task. Known partial inner load failures preserve the
successful key subset; decoding failure reports no hits for that task.

## Supported storage combinations

Ordinary raw-block io_uring supports serde with eager host fixed buffers, host
DMA-BUF buffers and GPU DMA-BUF L1. GPU L1 currently permits plain payloads or
FP8 serde only. CPU byte-view codecs are rejected for GPU arenas. Encoded
intermediates consume the same L1 capacity as source/destination KV objects;
operators must budget for both. The
[raw-block design](raw_block.md) and
[operator configuration](../../../../source/mp/l2_storage/raw_block.rst)
describe registration and exporter requirements.
