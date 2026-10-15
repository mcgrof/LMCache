# Raw-block storage P/D integration

The raw-block path gives the decoder a durable publication receipt for the
producer's prompt KV. It is a request-scoped handoff, including when ordinary
cache lookup misses or vLLM already has some of the prompt in its GPU cache.

## Ordering and ownership

1. The producer gathers KV into staging and waits for the gather to complete.
   The backend publishes READY only after its request's writes and checkpoint
   meet the durability contract. Source-block completion and publication-extent
   release are separate operations.
2. The decoder scheduler carries READY through allocation even for a local
   prefix hit requiring zero external blocks. Ordinary minimum-retrieve
   thresholds cannot silently skip the publication.
3. Before reading, the worker reserves ACK capacity, claims the producer's
   publication, and adopts a checkpoint matching the receipt's manifest.
4. The decoder restores exactly that manifest's tokens. The proxy can append
   the first generated token to the decoder prompt; it is not part of the
   published KV and must not change the last partial chunk's cache key.
5. The worker acknowledges only after every published token is either in the
   resident prefix or marked successfully restored. Chunk alignment can cause
   restoration to overlap the resident prefix; adding the two counts is not
   proof of complete coverage. Missing tail tokens must fail the handoff.
6. The ACK client owns retries after the inference step returns. The producer
   validates the ACK identity before releasing its extent hold. Failed reads,
   failed delivery, and process restarts do not imply that extents are reusable.

Connector construction owns the manager, notification sender/queue, and ACK
client as soon as each starts. A later construction failure shuts down those
resources before propagating the error.

## Supported limits and validation

This implementation uses synchronous, non-layerwise restore. The raw-block
backend rejects asynchronous loading and layerwise operation in storage P/D
mode. A publication must fit in one restore; a smaller
`lmcache.max_tokens_per_load` is rejected. GPU staging is a separate choice
from the publication and ACK protocol.

CPU tests exercise receipt propagation, local-hit scheduling, exact token
ranges, overlapping-prefix coverage, failure cleanup, and loopback control
messages. They cannot validate CUDA/HIP stream completion, native DMA-BUF I/O,
raw-device durability, or the real vLLM scheduler/worker lifecycle. Qualification
requires the exact built tree on the target GPU, an exclusively assigned
storage target, and an unmodified supported vLLM installation. Initialized
private filesystem targets need their own admission and payload checks;
raw-device qualification needs a separately provisioned raw namespace.
Include partial chunks,
warm local prefixes, concurrent requests, cancellation, and stock serving in
that run. File-backed results qualify only the tested filesystem profile;
they do not establish raw-device correctness. Import workarounds do not
qualify a stock vLLM installation.

## Strict MP consumer prefix

The MP connector's opt-in `lmcache.mp.strict_prefix=true` requires a consumer
role and vLLM's `kv_load_failure_policy=fail`. Ordinary MP lookup prefetches into
L1 before advertising a prefix. If that prefetch fails, returning zero would
bypass vLLM's failed-load policy and recompute. Strict mode instead advertises
the required complete prefix range so worker retrieval must succeed or report
invalid blocks through vLLM's existing failure path. It does not implement a
second abort protocol. The expected range excludes the last prompt token and
is rounded down to the MP chunk boundary. Non-strict requests keep ordinary
cache-miss behavior. Storage receipt adoption and claims happen through the MP
control API before dispatch; ACK follows completion, never a failed restore.
