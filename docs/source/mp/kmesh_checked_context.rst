Experimental checked raw context lifecycle
==========================================

KMesh multiprocess model recovery remains disabled. The low-level
``GPUCacheContext(..., managed_raw=True)`` constructor is a resource primitive
for a bounded server integration, rather than a serving configuration.
It accepts only raw CUDA IPC wrappers and requires an inactive GDS slab.
Ordinary context creation and serving keep their existing behavior.

The managed constructor retains its context before its first raw import.
A failed construction stays quarantined until the importing process ends.
A trusted server must stop admission and drain accepted consumers before
calling ``close_raw_checked()``. This operation drains the actual context
stream, permanently retires the raw allocation identities against new imports,
closes checked references, and verifies that no raw-wrapper aliases remain.
Only then does it return the released allocation identities. Successful
repeated calls return the same identities without another native close.

Keep the context and exporter allocations alive after any refusal. A failure
known to precede native close can be retried after its cause is removed.
An ambiguous native import or close outcome requires importing-process teardown;
stream completion cannot make its old pointer safe to submit again.

This check does not cover PyTorch storage, VMM or external driver imports.
The constructor and close also refuse active GDS because the existing staging
buffer deregistration interface logs backend errors. GPU-L1 storage therefore
requires a separately qualified checked lifecycle before admission. A service
release receipt, storage namespace fencing and a model consumer are additional
gates; this primitive does not grant any of them.
