# Consume registered spans in a bounded model group

Status: prototype. Complete-group metadata, worker ownership and an explicit
layer-major CacheBlend consumer are implemented. CPU consistency and synthetic
session checks pass. Native group inputs, the registered serial consumer and
the matched storage comparison still require their real GPU/model gates.
Ordinary multiprocess KMesh serving remains disabled.

The serial worker permits one live model input because that input owns engine
blocks through layer repair and checked physical cleanup. A collective model
step needs several inputs alive at once. Removing that guard would allow one
request to restore into another request's still-live blocks. Instead, admit one
complete group under the existing worker owner.

First complete every member's native restores while the scheduler still owns
all destination blocks. Validate every receipt, exact token content, receiving
fence and registered context before creating any model input. Require distinct
request/generation identities, pairwise disjoint full block tables and unique
restore operation IDs across the group. Sharing an immutable canonical source
is allowed; sharing mutable destination blocks is not.

Bind the complete ordered model request set to that group before creating any
CacheBlend session. The group must reject a changed request, generation, token
window, span list or member order without advancing a retrieval generator.
Individual sources remain single-use and retain their existing actual-cache
views. Store, restore and retirement remain blocked while any source is live
or owes physical completion.

Start with layer-major lockstep and request-local model operations. Each member
gathers only its restored valid positions, publishes its own repair buffer and
scatters repaired state back into its own engine slots. Cached keys were already
rotated by native restoration: do not apply cached-key position recovery again.
Batching model-side query/key rotation and selection differences is a separate
extension after this ownership and numerical reference passes.

Close every created session and every group source before permitting request
fallback. Attempt all member cleanups even when an earlier member refuses a
device drain. Retain an aggregate cleanup owner and report unresolved physical
work as a typed error; a per-member failure result or logical generator close
cannot authorize reuse of those blocks. Restart or actual importer exit remains
the final boundary when completion cannot be established.

CPU consistency tests can exercise group identity, receipt uniqueness and block
alias rejection. Real GPU tests must cover simultaneous layer inputs, private
writeback, early close and refused drain across multiple members. The real model
must then match the qualified serial K/V, masks, selection, logits and tokens,
including a failing member followed by safe recomputation. Keep external
retention and storage-read coalescing distinct from model batching: fewer reads
are not evidence of faster collective model operations.
