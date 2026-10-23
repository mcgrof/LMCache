# Registered rotary-cache ownership

Status: registered rotary generations, borrows and retirement are implemented.
CPU controls use actual shared-memory tensors and simulated receiving-client
context. Device completion and KMesh multiprocess admission remain unqualified.

The server registers rotary cos/sin tensors separately from the paged KV cache.
An instance number alone cannot identify either registration: the same worker
number can later refer to different tensors or a different rotary cache. A
shifted restore must retain both exact registrations until device work completes.

Give each rotary registration a fresh transport-owned generation and bind it
to the paged-cache context identity observed during registration. Expose an
immutable snapshot and a borrow returning the exact rotary state plus its
completion hold. Require the actual receiving request session, the expected
rotary generation, and the caller's already-borrowed paged-context identity.
Ownerless legacy registrations cannot grant these borrows.

Replacement, unregister, instance reaping and module close must first retire
the rotary registration: close admission, invoke lease-revocation callbacks,
and wait for physical holds. A timeout or callback failure retains the state
and refuses replacement. Serialize the final dictionary replacement against
other replacements without holding its lock across callbacks or drain waits.
A stale replacement must not overwrite a concurrently installed generation.

Existing ordinary CacheBlend registration remains usable without KMesh holds.
This change does not turn its legacy unregister response into a device-drain
acknowledgement. A KMesh adapter must retain its separate paged-context hold,
registration holds and exporter references, and release transaction holds only
after real server completion. The new snapshot alone authorizes no work.

CPU qualification must use the public borrow and retirement methods to cover
foreign and ownerless sessions, cache-generation mismatch, rotary replacement,
in-flight drain, callback failure and concurrent replacement. Real device events,
exporter teardown, model parity and storage I/O require separate qualification.
