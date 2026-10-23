# KMesh registered worker binding

Status: proposed server-side binding. Engine admission remains closed.

A worker restoring the same update at a different token position needs its
actual paged and rotary registrations. A scheduler's model label or digest echo
cannot prove that the receiving server has those tensors, that all ranks agree,
or that accepted device work has drained after a request abort.

Derive a runtime descriptor from exact borrowed registrations. Include resolved
kernel/object-group order, dtype, logical geometry and block stride, rotary
mapping and normalized cache contents, chunk/hash schema, registered model and
world size. Exclude addresses, device placement and allocated block capacity
from semantic compatibility. Retain those exact contexts separately for use.
The first binding accepts uncompressed BF16 full-attention KV without adapters,
recurrent state, quantization or sliding windows. Unsupported cases must refuse.

Compare server-derived layout and rotary digests with the independently supplied
worker semantics before binding its rank in the domain registry. Persistent
registration holds disarm the lease when either tensor registration retires.
Each accepted transaction takes separate paged/rotary holds and consumes a
monotonic request generation. Return exact borrowed resources to the executor;
do not look the worker number up again.

Revocation closes admission and releases registration holds, while accepted
transactions retain their resources until physical completion. Undrained failure
keeps both tickets and holds. Server completion may release debt after lease
revocation, but cannot rearm the lease or authorize publication. End and repeated
revocation are never completion reports. A replacement must not hide old debt.

CPU tests must bind actual shared-memory registrations, compare changed layout
and rotary contents, reject foreign/stale clients and incomplete rank barriers,
and retain exact old tensors through revocation and replacement. Tests simulate
physical completion; a real executor must supply device completion and exporter
teardown evidence. Typed transport methods, reserved storage namespace guards
and engine metadata/span consumption remain required before MP admission opens.
