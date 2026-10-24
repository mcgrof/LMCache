# Acknowledge imported-memory retirement only after verified release

Status: proposed. Ordered rotary snapshots have separate copy-completion
controls. No positive exporter-teardown acknowledgement is implemented, and
multiprocess KMesh engine admission remains closed.

A completed copy does not prove an imported mapping has closed. Legacy context
unregistration can retain a failed context while returning an ordinary reply.
The GPU context's wrapper cleanup also logs failures and drops its wrapper list.
The raw CUDA IPC wrapper removes its mapping record before checking the driver's
close result. These behaviors cannot supply a truthful positive release receipt.

Start with raw CUDA IPC memory handles and one tensor-parallel worker. Keep
PyTorch storage IPC and virtual-memory imports outside the positive-acknowledgement
profile until their own release contracts are implemented. A raw wrapper's tensors
use unowned imported memory: they must never be dereferenced after the last
mapping reference closes, even if Python tensor objects still exist.

The [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html)
specifies reference-counted memory import and close. The final close unmaps the
importer's mapping; it does not free the exporter's allocation or close other
processes' imports. Exported memory must remain allocated until importer close.
A driver error is not positive evidence that release completed.

Add an explicit checked-reference release to the raw wrapper. Preserve the
mapping record, owning wrapper and reference obligation when driver close fails.
Do not decrement the final reference or mark a wrapper closed before success.
Retries require an actual successful driver result; an absent record or an
invalid-handle error must not be reinterpreted as successful close. Keep legacy
best-effort cleanup from silently erasing the checked path's retained debt.

Reference release and allocation retirement are distinct operations. A completed
rotary download may release its temporary raw import while a paged-cache alias
still uses the same underlying CUDA allocation. It cannot acknowledge the whole
allocation as unmapped. During final retirement, close admission for the exact
allocation handles before draining and closing their registered references.
Reject new imports of retired handles. Existing aliases must each release their
own physical obligations; a remaining alias prevents allocation-level success.

Bind server retirement to the expected paged and rotary generations and actual
receiving client. Revoke their leases, stop new preparation and retain accepted
device work until physical completion. Retire only those exact generations;
replacement under the same instance number must survive an older request.
Retain the context, mapping owners and any partial-cleanup state in a retryable
server ledger. Cleanup success is monotonic and does not repeat already completed
destructive operations.

For the bounded raw profile, synchronously drain the actual registered stream,
finish staging-buffer deregistration and release every imported mapping through
the checked API. Report success only after all affected allocation aliases are
closed. The final receipt includes the server boot and exact registration
identities. Zero operations, zero copies, a missing instance, a legacy reply,
request cancellation and disconnect are never substitutes for that receipt.
The worker must preserve all exported allocations and producer events until it
receives and validates the receipt; lost replies require idempotent lookup.

Qualify controlled close failures before driver submission, later successful
retry, multiple wrappers on one allocation, concurrent import versus retirement,
and delayed operation completion. Then run real separate CUDA processes and
both public transports, proving retained aliases remain usable until their own
release. Do not dereference a deliberately unmapped pointer to test failure.
Include replacement, lost-reply replay, clear/shutdown refusal and an exporter
that stays alive until the checked receipt. Actual device faults and process
death need separate qualification; a controlled failure does not establish their
recovery semantics.
