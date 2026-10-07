# GPU staging readiness and ownership

The non-layerwise vLLM V2 and V3 connectors finish a gather before returning
its staging objects to a storage backend. Both CPU and CUDA destinations use
this contract. A registered CUDA buffer can be alive while its producer is
still running; a storage reference count alone does not make its bytes ready.

## Outbound contract

The caller's current CUDA stream on the connector device must contain, or
already wait for, every producer of the selected KV and slot mapping. The
connector joins this stream into its store stream, initializes pointer metadata
on that stream, enqueues the batch, and waits for that stream once. It does not
wait for the whole device. Calls on one connector must be serialized.

The caller keeps the selected source slots, slot mapping and destination objects
alive and unchanged until the call returns. After successful return:

- Original vLLM slots can be reused because the gather has finished.
- A backend can submit reads of the staging objects, including external DMA.
- The backend must retain those objects through terminal storage completion.
  The connector's wait does not cover later storage I/O.

`from_gpu` uses a one-element batch. Empty batches enqueue nothing. Parallel
batch lists must have equal lengths; silent truncation is not allowed.

## Inbound contract

Before `batched_to_gpu`, the caller must finish the storage reads and satisfy
the applicable exporter/device external-write visibility contract. Source
objects and slot mapping must be ready. The connector enqueues scatters on
its load stream and waits before returning, so a custom pool may then recycle
the staging sources and dependent model work may use the restored slots.

This change does not add a universal GPUDirect flush. Required visibility
steps depend on the allocation/exporter/device contract and must be qualified
separately. It does not make an unsupported PCIe route work.

## Errors and compatibility

Both batch directions drain already-enqueued work in `finally` when a later
enqueue raises. If synchronization itself fails, the method propagates that
failure; it does not establish quiescence or permit recycling. Recovery from
a failed CUDA context is outside this connector change.

The MUSA subclass retains its own per-object dispatch. It must not inherit
CUDA-specific batching. Layerwise connectors are unchanged.

## Validation and later overlap

The deterministic regression executes the actual public method bodies with
deferred-stream doubles and observes payloads at return. Native CUDA tests use
the real connector and compiled gather/scatter kernels with delayed producers
and gathers. Optional DMA-BUF filesystem tests check the storage boundary.
These are distinct tests; none alone qualifies end-to-end P/D inference.

A later asynchronous interface can return a recorded gather-ready event to an
I/O worker. It must retain source owners through gather, staging owners through
storage completion, and restored staging owners through scatter. Cancellation
does not discharge those obligations. Such an interface is not implemented here.
