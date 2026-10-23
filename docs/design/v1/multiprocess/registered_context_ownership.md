# Registered context ownership

Status: proposed. KMesh MP admission remains closed. These rules extend the
domain registry's accepted-work debt to the actual registered cache context.

A worker instance number can survive deregistration and be reused for different
tensors. The actual context registration therefore needs a fresh opaque
generation and the transport-derived session that created it. A no-op refresh
of an existing registration must retain that identity rather than bless the
refreshing caller as a new owner. Direct or legacy registration without a
session remains usable by ordinary transfers but cannot authorize KMesh work.

An opt-in caller borrows the exact registered context, matching its generation
and the current request's actual session. It also supplies a revocation callback
for the domain lease. The borrow returns a hold that the server completion path
must retain until accepted work physically finishes. Looking up the instance
number again during execution is insufficient: it could name a new context.

Retirement closes the context to new borrows, invokes revocation callbacks
outside the context lock, and waits for outstanding holds. Only after all holds
drain may context close synchronize streams and release mappings. The cache
registry's existing leaf lock must not be held across callbacks, waits or device
cleanup. A borrow racing a pop is either retained by retirement or refused after
retirement begins; it cannot attach to already-released memory.

An undrained timeout or a failed revocation callback quarantines the context
through the existing retained-context failure path. A timeout does not release
holds. Callback failure remains uncertain even if the physical holds later
drain. Calling END or close repeatedly cannot erase that uncertainty. A hold
release is idempotent; a caller reporting physical completion too early remains
an integration error, not something a reference counter can detect.

Normal transfers that never borrow add no accepted-work holds. Their existing
stream-completion shutdown remains in force. This increment must not change
store/prefetch retention, failure chains or metadata lookup ownership.

CPU tests must cover ownerless and mismatched-session refusal, fresh generation
after replacement, close/borrow races, multiple accepted holds, repeated close,
and failed revocation. Real registration tests must use actual cache contexts
and request dispatch. These are prerequisites for serial position-recovery
execution, not evidence of model equivalence or device-I/O correctness.
