# Publish and restore immutable spans under an accepted KMesh operation

Status: the owner-routed per-key write-status API is implemented and passes
actual CPU L1 admission/readback controls. Native registered-resource planning
and span execution remain design only. Managed raw registration, ordered rotary
preparation and checked release are separately qualified. The multiprocess
model consumer remains disabled.

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
Expose a public registered-resource planning interface rather than reaching into
another class's private planner or invoking its legacy instance lookup handler.
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
