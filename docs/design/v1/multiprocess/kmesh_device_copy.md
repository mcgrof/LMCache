# Retain registered tensors through device fingerprint copies

Status: the CPU registration controls are implemented. The device-copy
component described here is proposed. It does not enable device runtime
discovery, storage recovery or multiprocess KMesh inference.

A server deriving a rotary-cache fingerprint must copy actual registered
tensors to host memory. For example, cancelling the request after the first
asynchronous copy does not prove that its destination or source can be freed.
Both the paged-cache and rotary registration must remain borrowed until the
server proves physical completion, including when copy submission fails.

Use a server-owned copy manager that consumes those exact two borrows. Retain
the input tensors, pinned host destinations, producer-event import and registered
stream before submitting copies. Wait for the worker's producer event on that
stream, enqueue the copies and synchronize the same stream before returning
detached host snapshots. An event backend may synchronize through a separate
helper stream, whose failures would require an additional drain contract; do not
introduce that consumer for this synchronous fingerprint copy. CPU snapshots use
synchronous independent copies and need no device event.

If any device operation raises, attempt to drain the actual registered stream.
A successful drain permits releasing the borrows but does not make the failed
copy usable. A failed drain retains every buffer and both borrows in a
process-lifetime quarantine. Neither cancellation, timeout nor object collection
is physical completion. Only a later successful server-side stream drain may
release this debt. Closing the manager stops new admission and refuses to
complete while active or quarantined copies remain.

Use existing platform stream and event interfaces. Do not infer DMA completion
from a transport reply or a storage completion entry. The copy primitive needs
real-device tests in addition to controlled CPU failure injection. Verify raw
host bytes, non-default-stream producer ordering and retention through failed
copy/synchronization and later drain. An actual separate exporter must retain
its allocations until the server acknowledges their completed use.

This component is a prerequisite, not a complete discovery path. Legacy rotary
registration reads and normalizes worker tensors without an explicit producer
event. An ordered registration path must wait before that first read; waiting
only before a later fingerprint cannot repair an earlier race. Bind discovery
to that exact qualified rotary generation, serialize store clearing and shutdown
with all copy debt, and provide an exporter teardown acknowledgement before
enabling positive device discovery. Keep the existing CPU-only discovery guard
and engine-admission refusals until those consumers are implemented and tested.
