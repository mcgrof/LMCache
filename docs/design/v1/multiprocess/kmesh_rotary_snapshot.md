# Register a rotary snapshot before admitting device discovery

Status: ordered registration, private canonical publication and scoped copy
draining are implemented. Device snapshots require explicit opt-in and their
public-server CUDA qualification is pending. Exporter acknowledgement remains
proposed. Multiprocess KMesh engine admission remains disabled.

A cache server must not normalize a worker's rotary cache before that worker
has finished producing it. Waiting only when hashing the result is too late:
the first normalization read may already have raced. For example, a worker
initializing its cache on a non-default stream must supply an event recorded
after that initialization, and the server must wait before reading any values.
The worker must keep inputs unchanged until download completes. A producer
event orders earlier writes; it does not exclude later writers or provide a
kernel-enforced atomic snapshot.

Use a dedicated ordered registration operation alongside the legacy blend
registration API. Select and borrow the exact paged context for the receiving
client and its expected generation. Keep it borrowed through the whole
preparation. Validate wrapper metadata and group geometry without reading tensor
values; map the raw rotary tensors into a provisional, server-owned generation.
No provisional generation may be discovered, bound or used for recovery.

The typed controls are ``kmesh_get_context``, ``kmesh_register_rotary`` and
``kmesh_retry_rotary_drain``. CPU controls require ``--enable-kmesh-control``
on a blend server. Device snapshots additionally require
``--enable-kmesh-device-snapshots``. The device input profile currently accepts
PyTorch CUDA IPC rotary wrappers; raw-driver and virtual-memory wrappers are
refused before import. Generate the gRPC bindings again after schema changes.
An absent value and error string report refusal on both request transports,
including a CPU input whose shared-memory name has already disappeared.
Each PyTorch CUDA IPC import must receive a fresh export wrapper, including
after a failed copy has drained. Reusing an already-consumed export reuses its
reference-counter token. Do not create an unused export just to test disabled
admission: the receiver cannot consume or close a handle it deliberately never
imports. This control prototype supplies no general outgoing-export cancellation
or positive exporter-teardown acknowledgement. A complete worker path needs
those lifecycle guarantees before engine admission can open.

Give the retained-copy manager a separate paged borrow and a provisional rotary
borrow. Wait for the producer on the actual registered stream, copy the raw
rotary caches into independent host snapshots and prove that stream has drained.
An uncertain copy retains its buffers and borrows without publishing a state.
Normalize and validate the completed snapshots on CPU, preserving the declared
dtype and the existing scale-removal rule. This gives registration a concrete
snapshot boundary, rather than assuming the exporter remains immutable forever.

For a device context, allocate new server-owned rotary tensors and upload the
normalized host snapshots on the same registered stream. Retain each destination
before submitting its copy, and retain all host inputs and the paged context
until upload physically completes. Use the retained-copy completion/quarantine
contract in both directions. Never release a failed upload merely because its
request ended. CPU contexts publish independent host snapshots directly.

Only after both copies complete may the server publish the final rotary
generation through the existing retirement and compare-and-replace rules.
Retirement of the paged context during preparation prevents successful
registration. Outstanding work from an old rotary generation must drain before
replacement; failed retirement must not overwrite it. Bind the final generation
to its original paged context, canonical CPU snapshots and actual server-owned
device tensors. A client-provided boolean or digest echo is not this evidence.

Device runtime discovery may hash those canonical CPU snapshots only for a
generation created by this ordered server path. Its GPU tensors are private to
the server and never exported for worker mutation. Legacy GPU rotary
registrations keep the current discovery refusal. Derive layout from the exact
paged context and compare independent worker model/tokenizer/weight declarations
as before; a rotary snapshot does not attest model weights or cached KV values.

Track preparation, download and upload debt in server lifecycle controls.
Logical reset may revoke admissions but must refuse physical store clearing
while any such work remains. Shutdown closes new preparation before checking
that debt and must preserve uncertain resources. A later drain is a server
observation, not a completion assertion accepted from a client.

Exporter teardown requires a distinct positive acknowledgement tied to the
exact paged and rotary generations and server boot. A zero transaction count,
missing instance lookup, legacy unregister reply or lost connection does not
prove that the server closed its imported mappings. Retirement must report
actual close success, retain failures for retry and never acknowledge a newer
instance under an older generation's identity.

Qualify CPU snapshots, normalization, stale/foreign clients, retirement during
preparation, replacement and lifecycle refusal through both typed transports.
Then use separate CUDA exporter/server processes with a pending producer,
independent byte comparisons, upload/readiness checks and injected failures
that retain resources until a later physical drain. Keep namespace/data
execution, shifted-model validation and collective recovery as subsequent gates.
