# Managed raw context registration and release

Status: the managed GPU context resource primitive is implemented and passes
CPU admission and separate-process CUDA controls. A dedicated raw-only service
implements typed managed registration, raw rotary preparation and cached release
receipts. Separate-process CUDA controls pass on both transports, including
actual exporter frees while the importing service stays alive. Pinned strict
reproduction is a separate qualification.
Raw rotary copy primitives have separate CPU/CUDA qualification. Multiprocess
KMesh engine admission remains closed.

The ordinary context close path drains its stream but logs raw mapping failures.
It therefore cannot produce an exporter release receipt. The first managed
profile must make ownership precede imports and make release a checked operation.

Bound the first profile to one raw CUDA context per server lifetime, TP=1,
contiguous BF16 full attention and no active GDS slab. Ordinary PyTorch, VMM,
legacy transfer, peer, HTTP debugging and automatic reaping surfaces must not
reach this context. Serving model recovery remains a subsequent gate. The
inactive-GDS restriction is deliberate: existing staging deregistration logs
backend failures and cannot yet provide checked retirement evidence. It must
be qualified before enabling this profile with GPU-L1 storage.

A managed GPU context retains itself before its first import or initialization
submission. Pointer-table initialization on the original current stream must
physically drain before publication; draining only the later transfer stream
cannot prove those earlier writes complete. Partial construction remains
retained until importing-process exit;
no ordinary best-effort rollback may silently convert it into successful release.
The server records the actual receiving owner, expected server boot, nonce,
exact context generation and every incoming mapping owner before construction.
A failed construction poisons that registration. Retry of the same nonce must
return its status, rather than reconstructing ambiguous imports.

Checked raw context close first proves stream completion. Then retire every
raw allocation against new imports, close each owned reference through the
checked wrapper API, and verify that no raw mapping for those allocations
remains. Retain all resources on failure. Retry only known pre-native refusal;
an ambiguous native outcome still requires importing-process teardown. Once
complete, repeated close is idempotent and cannot submit an old native pointer.
This operation does not count PyTorch, VMM or external driver imports.

The managed server should expose only its actual control inventory. Registration
publishes the supplied exact ownership object into the internal transfer module;
the legacy integer registration RPC is not exposed. Rotary preparation consumes
owned raw copies before import, normalizes completed host snapshots and publishes
private uploaded tables. It never mutates shared worker tables in place.

Retirement closes admission before revoking exact-context lease holds. Accepted
operations, preparations, copies and stream work must physically drain. Remove
only the selected paged and rotary generations and finish checked mapping close
before caching an idempotent receipt with the actual server boot, receiving
owner, registration nonce and allocation identities. A lost reply may be looked
up again. Zero pending work, absent lookup state, disconnect or a legacy None
reply is not this receipt. A worker keeps its allocations and producer events
until validating that receipt or observing importer-process exit.

Qualify real exporter/importer processes, exact bytes, pending producer order,
foreign-session and stale-boot refusal, lost-reply lookup, duplicate nonce,
retirement racing an accepted copy, retryable pre-native cleanup refusal and
ambiguous native quarantine. Keep legacy serving and MP engine guards intact
until the actual storage namespace and model consumer pass their own tests.
