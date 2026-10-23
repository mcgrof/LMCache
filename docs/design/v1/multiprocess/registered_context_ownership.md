# Registered context ownership

Status: registration identity, context holds and retirement are implemented and
tested with actual CPU cache registrations through both request transports.
KMesh MP admission remains closed. Binding these holds to KMesh transactions
and actual device completion remains required.

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

A legacy unregister response is not a KMesh drain acknowledgement. Before an
exporter releases its tensors, the KMesh adapter must establish transaction
completion and receive its own positive teardown acknowledgement. Retaining a
server-side context after failure does not prove an exporter can safely exit.
Arbitrary exporter death with outstanding device work remains unqualified.

`tests/v1/test_context_ownership.py` covers ownerless and mismatched-session
refusal, fresh generation, multiple holds, repeated close, failed revocation
and concurrent retirement while a callback is incomplete.
`tests/v1/test_registered_context_ownership.py` uses actual shared CPU tensors,
an eight-MiB DRAM store and both request transports. It checks foreign-client
refusal, no-op refresh ownership, deregistration/reaping with pending holds and
replacement with distinct tensor contents. Completion is controlled by the CPU
test; it does not attest a device event. These are prerequisites for serial
position-recovery, not evidence of model equivalence or device-I/O correctness.
