# Publish and restore immutable spans under an accepted KMesh operation

Status: the owner-routed per-key write-status API is implemented and passes
actual CPU L1 admission/readback controls. Registered-resource native planning
and bounded CPU-L1 span execution are implemented behind an explicit opt-in.
Managed raw registration, ordered rotary preparation and checked release are
separately qualified. The multiprocess model consumer remains disabled.

Registration ownership does not establish storage publication. The ordinary
transfer handler records an event and queues a host callback that admits its
write later. Its owner/key completion method also reports failed keys only on
the event bus. A managed span operation needs physical copy completion and an
explicit admission result before publishing a logical source handle.

Use the exact paged and rotary resources returned by ``bindings.begin`` for the
actual receiving worker fence, request ID and monotonically increasing request
generation. Do not resolve the instance integer again. Keep this accepted
operation, all memory objects, block-ID tables, event imports, native plans and
storage leases until the server proves physical completion.

The first execution profile remains serialized, single-worker, contiguous BF16
full attention with one object/kernel group, full source chunks and CPU DRAM L1.
Use the existing native object-group transfer and positional-retrieve executor;
do not substitute a numerical reference implementation for the measured path.
``BlendModule.prepare_registered_retrieve`` plans against accepted resource
borrows without a legacy instance lookup. Its returned plan is single-use and
does not claim completion when submission returns.
Store waits for the worker producer before reading source slots. Restore waits
for the worker's destination-ready event before writing reserved private slots.

Derive every physical object key from the accepted semantic namespace and a
fresh server operation identity. Keep it immutable and never reuse it for a
replacement reservation, even after failure or eviction. Current L1 completion
groups contain owner IDs and keys, not reservation epochs. A late completion
must therefore never address a later allocation using the same physical key.
Publish an opaque logical span handle only through ``bindings.publish`` after
all bytes complete and every requested key reports successful admission.
A partial admission invalidates the namespace and cannot publish a span.

Add an owner-routed completion method that returns actual per-key results while
preserving the existing void interface for ordinary callers. Its result is not
reservation-epoch evidence; the unique physical key rule supplies that missing
protection for this bounded profile. Captured owner groups must come from the
actual reserved objects, never a new placement lookup.

Restore resolves only a published handle in the accepting operation's namespace.
Use the retained immutable source coordinates and content/layout/rotary facts;
the client supplies destination coordinates and its actual block reservation.
Acquire actual read locks and retain the returned objects through completion.
Use ``acquire_write_transfer_leases`` or ``acquire_read_transfer_leases`` with
``all_buffers=True`` before enqueueing native work against host objects. These
physical leases prevent allocator reuse when logical read/write locks expire.
Ordinary eviction refuses a resident object while its physical lease remains
active; discarded write reservations also withhold their memory until release.
Release these leases only after checked completion. The default GPU-only lease
selection remains available to existing callers.
An absent or evicted source fails safely and lets the worker recompute. No
foreign namespace, stale generation or rejected request may trigger device I/O.
Never rotate shared stored bytes in place; positional transformation uses private
GPU staging before scatter into the worker's reserved destination.

The native retrieve executor has internal copy streams. A partial launch can
fail before their work joins the outer stream, so an outer-stream drain alone
cannot prove cleanup. The first serialized implementation uses an actual drain
of the importing server's device context on both success and failure. Keep all
buffers, exceptions and their still-needed preparation state quarantined if it
cannot drain. Only a later server-observed physical drain can release failed
work; request END, cancellation, heartbeat loss or a client boolean cannot.
Successful physical completion still needs the live publication fence.

The worker must retain actual scheduler block reservations as well as CUDA
allocation references. Whole-allocation registration cannot prevent the engine
from recycling a slot within that allocation. Failed but drained restoration
requires request-local fallback and recomputation before those slots become
computed state. Rotating keys does not repair deeper-layer dependence on the
original preceding context; actual model repair and numerical checks remain a
separate gate before a serving performance claim.

Qualify real native gather and shifted scatter against independent tensors,
exact unchanged values, private canonical source preservation, namespace and
session refusal before I/O, stale publication, failed admission, source eviction,
partial submission, failed physical drain and later cleanup. Exercise the typed
service with separate exporter/importer processes before connecting a model.
GDS, storage-backed cold restoration, model quality, staged/DMA-BUF comparison
and collective consumers each require their later explicit qualification.

The dedicated raw service enables this executor with
``--enable-span-execution``. It selects lazy pinned CPU allocation because the
native positional planner requires that allocator's registered chunks. Neither
the legacy default key grammar nor GPU-backed L1 objects are admitted. Use the
explicit ``kmesh-span-v1`` semantic key schema for unique physical span objects.
The store request records the source's token IDs, original position and declared
preceding-context digest. The resulting opaque handle also binds layout and
canonical rotary facts. Physical objects use BLAKE3 keys derived from namespace,
accepted operation identity and chunk index. They are never overwritten.

``kmesh_store_span`` and ``kmesh_restore_span`` return exact operation status.
Identical request-generation retries return saved facts rather than another
native launch; changed metadata refuses. ``kmesh_span_status`` resolves a lost
reply independently of execution. A status for different blocks, context,
request generation or receiving owner cannot release engine reservations.
``kmesh_drain_span`` retries physical cleanup of known accepted work, including
after revocation, without resubmitting its transfer.

Every source object has a SHA-256 byte checksum captured after the native gather
drains. Restore checks actual read-locked bytes before any GPU scatter. Eviction
or a checksum mismatch returns failure and revokes the accepting registration.
``failed-undrained`` retains accepted resources, exception frames and actual
host I/O leases, even across server teardown. No later execution may reuse the
planner buffers until a real server-device drain settles that work. Only
``failed-drained`` or importing-process exit permits request-local recomputation.

``stored`` proves checked admission; ``loaded`` proves same-position and
same-context physical restoration. ``loaded-needs-repair`` proves only I/O for
changed positions or preceding context. None of these receipts implements the
worker's model repair or marks vLLM tokens computed. Model and tokenizer artifact
digests remain cooperative worker declarations, not server weight attestation.

## Source positions inside a chunk

The first native storage profile admits chunk-aligned source positions. That
restriction cannot represent a shared segment beginning after a private prompt
prefix, for example at source token five with a 256-token storage chunk. Chunk
length is relative to the saved segment; it need not align the segment's first
position to the original prompt. Destination rotation alone cannot fix a gather
that selected the wrong source tokens.

Keep the existing block transfer for aligned sources. For an unaligned source,
build its exact slot map from the retained block table and original positions:
``slot[p] = blocks[p // block_size] * block_size + p % block_size``. Use the
existing native ``multi_layer_kv_transfer`` gather into each complete pinned CPU
object. Its slot-map path supports the two-plane head-major and token-major
layouts needed by the bounded full-attention worker. Refuse other unaligned
formats until qualified, while preserving their existing aligned path.

The accepting operation owns every slot-map tensor, pointer table and actual
host-buffer I/O lease until the importing device context drains. A launch error
can occur before the helper returns its new tensors; retain its exception frames
as well as the accepted operation. Failed drain must withhold buffers and prevent
admission. A later cleanup retries physical drain only, not the gather itself.
No source handle may be published from partial or uncertain bytes.

Qualify a nonzero unaligned source with multiple layers in both supported
layouts, reversed noncontiguous blocks, positive and negative positional shifts,
raw value equality and an independent key rotation. Poison the original source
slots before restore. Inject an exception after real slot-map gather and refuse
its first drain; retirement must wait, and later physical drain must settle the
operation without another gather. This storage extension still supplies no
model-repair consumer or DMA-BUF performance result.
