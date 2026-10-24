# Consume registered span restores through the existing model repair

Status: design. The managed native span service has storage qualification;
its receipt does not authorize computed model tokens. Keep the ordinary
multiprocess connector's KMesh guard closed until a complete consumer passes.

## Worker ownership and completed placement

Use a dedicated worker-side owner for one actual CUDA context, its immutable
rotary inputs and one receiving transport client. Normalize real engine cache
views without copying their contents. Register raw wrappers and the recorded
rotary producer event with the managed service. Bind the returned server boot,
context generation, rotary generation and runtime facts to separately verified
model/config/tokenizer artifacts and the exact execution profile.

Retain the owner before the first remote import can happen. Initialization
failure, transport uncertainty or a missing retirement reply must not drop
exporter allocations. Resolve known operations by their exact request generation;
absence is not drain evidence. A successful close needs the service's matching
boot, nonce, instance, context, rotary generation and complete allocation set.
Partial construction or ambiguous native import/close still needs importer exit.
Never infer exporter release from a zero pending count or a copy-complete event.

Store only while the real worker's model step still owns its source block
reservation and after recording all source writes. Keep exact token content,
original coordinates and preceding-context identity with each immutable handle.
Restore under the same worker's live private destination reservation, retaining
its original request arguments before submission. Validate every receipt's
fence, request/generation, context, operation kind, block digest and source handle.
A changed metadata retry refuses; an unresolved operation keeps blocks reserved.
A drained failure permits fallback, while unknown or undrained work prevents
request recomputation into those blocks. Revoked registrations do not rearm.

## Layerwise model input and writeback

Add an explicit layer-source interface to CacheBlend sessions. Preserve the
existing cache-engine retrieval as the default. A registered source is bound to
completed placement receipts, exact token content, valid hit spans, physical
engine cache views and the worker's still-owned slot mapping. It must not take a
client-supplied completion boolean or infer readiness from populated pointers.

The registered source reads only the restored valid slots through the existing
native transfer kernel into a request-owned GPU buffer. Publish it through the
session's existing LayerwiseLoadState mapping so the real LMCBlender model and
selection logic consume it. Zero explicit gaps and expose their mandatory
recompute positions. Keys already rotated during the service restore must not
be rotated again. After each model layer repairs the buffer, scatter its result
back into that request's actual reserved engine slots before advancing.

The layer source is single-use. It must preserve the existing prime, one yield
per model layer and final-mask choreography. Closing partway through a layer
must drain actual device work before releasing GPU buffers. A failed drain
retains buffers, slot maps, cache views and the owner, and forbids fallback or
another load until an actual worker-device drain settles it. Logical generator
close or request END cannot substitute for that boundary.

## Bounded model experiment

First use the same pinned Qwen model, token trace, repair ratio and output
capture as the in-process serial reference. Publish sources from actual model
K/V, recycle original source slots, restore shifted spans in the managed cache
process, then consume them through real CacheBlend repair. Preserve partial
final chunks and explicit gaps, including the baseline's 416 recovered tokens.
Record source and destination positions, actual hit masks, raw K/V comparisons,
selection masks and logits. Keep ordinary full recomputation as a separately
labeled diagnostic; it is already numerically different from the old reference.

Test miss, eviction/corruption before scatter, post-enqueue failure, unresolved
physical drain and later safe request-local fallback. Check engine block lifetime
through the complete consuming model step and service retirement after model
cleanup. Keep earlier profiles and source identities intact. A model replay
using extra reference metadata/storage must report that cost before a serving
latency comparison can use it. No legacy lookup hit may stand in for a verified
new source handle or a completed restore.

Only after serial repair/numerical/lifetime gates pass should the same consumer
compare staged NVMe reads against DMA-BUF reads with matched scientific inputs.
Active GPU-L1/GDS staging additionally needs checked buffer teardown. Collective
consumers follow the serial baseline and must preserve its request-local failure
containment. Kernel, storage media and shared runtime qualification remain
separate from this consumer integration.
