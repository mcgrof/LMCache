# Acknowledge imported-memory retirement only after verified release

Status: checked raw-wrapper reference release, import retirement and scoped
mapping inspection pass three scoped CUDA cases on an RTX 4090. The full CUDA
IPC regression module separately passes fourteen cases.
Ordered rotary snapshots have separate copy-completion controls. The integrated
retirement ledger and positive exporter-teardown acknowledgement below remain
proposed. Multiprocess KMesh engine admission remains closed.

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

The raw wrapper exposes ``close_checked``, ``retire_raw_imports`` and
``raw_mapping_released``. The last inspects only this implementation's registry;
PyTorch storage, virtual-memory and external driver imports are not counted.
Retired raw handles remain tombstoned for the process lifetime. This bounded
prototype has no tombstone reclamation policy or integrated server receipt.

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
