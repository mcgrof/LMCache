Experimental checked raw context lifecycle
==========================================

KMesh multiprocess model recovery remains disabled. The low-level
``GPUCacheContext(..., managed_raw=True)`` constructor is a resource primitive
for a bounded server integration, rather than a serving configuration.
It accepts only raw CUDA IPC wrappers and requires an inactive GDS slab.
Ordinary context creation and serving keep their existing behavior.

The managed constructor retains its context before its first raw import.
A failed construction stays quarantined until the importing process ends.
A trusted server must stop admission and drain accepted consumers before
calling ``close_raw_checked()``. This operation drains the actual context
stream, permanently retires the raw allocation identities against new imports,
closes checked references, and verifies that no raw-wrapper aliases remain.
Only then does it return the released allocation identities. Successful
repeated calls return the same identities without another native close.

Keep the context and exporter allocations alive after any refusal. A failure
known to precede native close can be retried after its cause is removed.
An ambiguous native import or close outcome requires importing-process teardown;
stream completion cannot make its old pointer safe to submit again.

This check does not cover PyTorch storage, VMM or external driver imports.
The constructor and close also refuse active GDS because the existing staging
buffer deregistration interface logs backend errors. GPU-L1 storage therefore
requires a separately qualified checked lifecycle before admission. A service
release receipt, storage namespace fencing and a model consumer are additional
gates; this primitive does not grant any of them.

Bounded managed raw control service
-----------------------------------

The dedicated experimental entry point composes only managed KMesh controls:

.. code-block:: console

   python -m lmcache.v1.multiprocess.kmesh_raw_server \
       --host 127.0.0.1 --port 5555 --transport zmq --chunk-tokens 256

Use ``--transport grpc`` for the other same-host request transport. The service
uses a CPU DRAM L1 and accepts one raw CUDA context and one ordered rotary
preparation per process lifetime. It exposes no legacy integer registration,
generic store/retrieve, lookup, peer, HTTP or automatic reaping interface.
It does not enable multiprocess vLLM model recovery.

Read ``kmesh_get_server`` first. Call ``kmesh_register_raw_context`` with that
boot identity, a fresh 32-character lowercase hexadecimal nonce, a positive
instance ID, exact raw CUDA wrappers, the model name and ``NHD`` or ``HND``
layout. The service records the receiving client's actual transport identity,
its own context generation and incoming wrappers before construction. The
bounded context accepts contiguous BF16 full-attention tensors at world size
one. Model artifact semantics remain independent worker declarations.

Repeating the same nonce and view metadata returns saved registration status
without importing again. Changing an interior view, model or layout under that
nonce refuses. ``kmesh_raw_context_status`` resolves uncertainty about a reply
but does not acknowledge release. Foreign clients and stale server boots refuse.
A poisoned construction requires importing-process exit.

Use the returned context generation with ``kmesh_register_rotary``. The worker
must retain every rotary input and its producer event without further writes
until the service proves their completed use. Incoming raw owners enter the
copy ledger before import. The service waits for the producer, takes a completed
host snapshot, normalizes it and publishes private uploaded tables. Repeating a
completed identical preparation returns its saved runtime without another import.
A failed preparation cannot be replaced in this bounded profile: drain retained
copy debt with ``kmesh_retry_rotary_drain`` and retire the context.

``kmesh_retire_raw_context`` takes the expected boot, nonce and exact context.
It closes admission, revokes registration holds, refuses unfinished physical
work, removes the exact rotary generation and performs checked raw close.
Temporary rotary allocations are retired against new imports after their
checked reference cleanup. Only after all mappings and bookkeeping finish does
it cache a ``ManagedRawRelease`` containing the exact generations and allocation
identities. Repeating retirement returns this same receipt without native work.
A known pre-native refusal can be retried after cleanup; unknown native effects
still require importer exit.

Keep all worker allocations and producer events until validating that receipt,
or until observing importing-process exit. Missing status, a disconnect, a zero
copy count or a legacy reply is not release evidence. These transport identities
coordinate cooperative clients; they do not authenticate a hostile peer.

Storage admission result prerequisite
-------------------------------------

``StorageManager.finish_write_with_status`` returns each captured key's actual
L1 admission result. A batch can contain successes and failures; callers must
check every requested key before publishing a logical source span. Existing
``finish_write_by_owner`` retains its void interface and event reporting.

Both interfaces require device completion first. Their owner/key groups do not
identify reservation generations. A delayed completion therefore needs unique,
immutable physical keys to avoid admitting a replacement reservation. This
status API alone does not enable managed span execution or model recovery.

Native CPU-L1 span transactions
-------------------------------

The dedicated service can additionally enable bounded native span transfers:

.. code-block:: console

   python -m lmcache.v1.multiprocess.kmesh_raw_server \
       --host 127.0.0.1 --port 5555 --transport zmq --chunk-tokens 256 \
       --dram-bytes 67108864 --enable-span-execution

This selects lazy pinned CPU allocation, whose registration granule is 64 MiB.
Use a semantic identity with ``key_schema="kmesh-span-v1"`` when opening and
binding the worker registration. The executor admits CUDA BF16 full attention,
one worker and one object/kernel group, with exact valid source extents. It does
not accept GPU-L1/GDS or enable the vLLM multiprocess model consumer.

``kmesh_store_span`` takes the bound fence, request ID/generation, source token
IDs, original source position, preceding-context SHA-256, full block table
and recorded producer event. Source positions may begin inside a block.
Block-aligned sources use native object-group transfer; interior positions use
the native slot-map gather for two-plane NHD/HND layouts. Each slot-map target
object must fit within one host-registration range. The service waits for source
writes, gathers exact source slots, drains actual device work and checks every key's L1
admission before publishing an immutable ``SpanHandle``. Each physical key is
unique to that accepted operation; the source is never overwritten.

A final source chunk may be partial. Gather only its valid slots into packed
host strides. After checked device completion, repack them into a complete
canonical object with zero padding before checksum and admission. The immutable
handle records valid tokens only. Restore scatters only that valid extent.
Native key rotation visits a full padded chunk, so a shifted match additionally
requires rotary-table capacity for both padded position ranges. Refuse short
tables before native launch. Partial gathering uses the qualified two-plane
NHD/HND formats and never reads following source slots as padding.

``kmesh_restore_span`` takes that exact handle, a destination position and its
preceding-context digest, reserved block table and destination-ready event.
The source must belong to the accepting namespace. The executor read-locks
actual source objects, pins their buffers against reuse and checks completed
host-byte checksums before native re-RoPE and scatter into private engine slots.
The canonical source is unchanged. Missing or corrupt sources fail safely.

Keep actual scheduler block reservations, allocation references and producer
events until checking the returned operation's exact context, request generation
and block-table digest. Repeated identical requests return cached state.
``kmesh_span_status`` recovers a lost reply; unknown or absent status cannot
prove completion. ``kmesh_drain_span`` retries physical cleanup of known failed
work without re-enqueueing the transfer. An unresolved drain retains the work,
buffers and exact resource borrows even across service shutdown.

``stored`` proves admitted source storage. ``loaded`` proves completed I/O at
the same position and preceding context. ``loaded-needs-repair`` additionally
requires request-local model repair before those tokens become computed state.
On failure, recompute only after ``failed-drained`` or observed importer exit;
``failed-undrained`` still owns the destination blocks. These controls do not
establish model quality, GPU-direct storage latency or collective recovery.

Request-owned model layer sources
---------------------------------

``LMCBlender.create_session(..., layer_source=source)`` can consume an explicit
request-owned layer source through the existing model repair and selection
logic. The default remains the cache engine's layerwise retrieval. A custom
source must expose the exact ``ExplicitTokenSpans`` also supplied to the model;
a mismatched recovery window refuses before retrieval begins.

The source yields a hit count, one input boundary per model layer and a final
hit mask. It publishes actual buffers and mandatory gap positions through the
session's private ``LayerwiseLoadState`` and writes repaired state back before
completion. Its close operation must settle device work or retain unresolved
physical ownership and report failure. This interface does not itself establish
completed placement, engine block lifetime or model numerical qualification.
The ordinary multiprocess connector's KMesh guard remains closed.

Request-owned layer sources currently support serial sessions. Group execution
refuses these sources before creating sessions. ``UnsettledLayerSourceError``
retains the actual owner's physical cleanup operation. Session abort and close
preserve this failure instead of hiding it behind an ordinary model error.
The serial worker retries that operation before reporting blocks for request
recomputation. If physical cleanup still fails, the error escapes the load step;
the worker must retain its reservations and stop normal request continuation.

Serializing this error carries failure text only. The physical cleanup callback
and its live device resources remain in the owning worker.

Experimental registered paged-layer input
-----------------------------------------

``lmcache.integration.vllm.kmesh_span_layers`` binds completed native restore
receipts to actual model tokens and a private engine block table.
``RestoredSpanBatch`` checks the accepting fence, context, request generation,
operation kind, block digest, semantic namespace and exact source content.
Only its completed valid spans become hits. Other declared spans remain gaps.
These cooperative checks do not acquire scheduler blocks or attest model weights.

``RestoredPagedLayerSource`` supplies those actual paged inputs through the
session interface. It reads valid restored slots with the native transfer kernel,
zeroes gaps, and publishes a request-owned GPU buffer through ``LayerwiseLoadState``.
The existing model performs selection and deeper-layer repair. Advancing the
source after each model layer scatters the repaired buffer into the worker's
still-reserved blocks. Cached keys are not rotated a second time.

Close drains actual worker-device work before releasing buffers. Unresolved
cleanup retains the source, raw tensors and cache views, and raises the typed
cleanup failure. ``drain_failed()`` retries a closed source's cleanup without
another transfer. Keep real engine reservations until that operation succeeds.
The primitive remains experimental; model outputs and end-to-end registered
serving have not been qualified by its synthetic layer-input controls.
