# TokenDance-style shared-update recovery

This experimental integration recovers an identical agent update at different
token offsets in otherwise different prompts. The vLLM KMesh connector supplies
content-bound token spans and an admission generation. LMCache looks up those
exact slices and reconstructs request-local key/value state through CacheBlend.
No live destination pages are shared between requests.

Pair this source branch with vLLM's
`2026-10-16-feature-tokendance-reproduction` branch. The compatible upstream bases
are LMCache `499c830b3fda473388a2b7d6bb002dd69f5d7867` and vLLM
`454ea5b52611c933e00581723e2db56f0144cea7`. This uses the in-process connector;
the newer multi-process prefetch and DMA-BUF storage integrations are separate.

## Recovery and ownership

`ExplicitTokenSpans` validates ordered, non-overlapping slices inside a
prompt-origin recovery window. Hashes contain the actual slice and existing
model/dtype/request-tag identity. Offsets describe where to recover content;
they do not substitute for content identity. Gaps in a recovery window require
recomputation.

Every layerwise blend has its own closeable session, load state, lookup pins,
metadata and destination buffers. Construction and retained-call receipts keep
an interrupted caller from losing the exact resource it must finish or release.
Cancellation drains disk reads before releasing buffers. Partial layer sets,
stale generations and failed loads cannot publish externally recovered blocks.
The optional Maru backend rejects ownership-receipt allocation before allocating;
it does not advertise the stronger allocation contract.

The target profile validates both resolved connector roles and the actual model,
token view and packed cache layout. It restricts execution to one device, eager
synchronous scheduling, Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, BF16 and Triton attention. The paired
vLLM profile also pins its model snapshot path. Unsupported configurations fail
before this experimental handoff can run.

## Serial and collective execution

`serial_pic_v1` executes position-independent recovery once per request.
`collective_lockstep_v1` advances independent sessions in layer order.
`collective_batched_position_recovery_v1` combines cached-key position recovery
across eligible sessions. `collective_batched_qkv_v1` additionally concatenates
query/key tensors for rotary encoding and combines check-layer key-difference
computation. It does not batch the query/key/value projections or attention.

Important-position selection, gap repair, attention masks and destination writes
remain request-local. A miss does not join a collective operation. If a shared
operation fails, its participants become failed while independent members can
continue. Cleanup attempts every created session.

Operation counters distinguish request-local calls from participation in group
calls. Shared-call participation counters must not be summed as unique group
invocations. Kernel counts do not establish wall-clock improvement.

## Offline family encoding

`lmcache/v1/distributed/serde/kmesh_family.py` is an opt-in CPU artifact codec on
the multi-output serializer interface. It stores one dense master, private tokens
for each mirror, and raw-bit differences over one content-aligned shared span per
member. Each destination receives its own restored tensor. This codec is not
factory-registered, has no live GPU-page aliasing path, and is intended for
locally generated experiment artifacts rather than untrusted external blobs.

The separate vLLM reference codec compares position-aligned blocks. Do not apply
its block accounting to the content-aligned LMCache format. Encoded-byte results
must include metadata, indexes and payloads and do not imply a serving-memory
capacity result.

## Verification boundary

The session, explicit-span, lookup, allocator, interruption and terminal-cleanup
tests exercise the real ownership implementation with controlled dependencies.
Collective CPU tests compare the real blender tensor operations with serial
recovery using deterministic rotary callables. Those callables are test doubles
for the device rotary implementation.

The retained GPU experiment uses the pinned source/runtime/model envelope and
compares collective recovery with serial recovery. It does not establish
elementwise equivalence to full prompt recomputation or a performance benefit.
The strict eight-arm GPU driver and verifier are separate archived experiment
inputs; they are not shipped in this branch. Their source-identity pins must be
updated before using the dated commits. The vLLM in-tree benchmark only runs the
ordinary and observer controls and cannot replace that driver.
