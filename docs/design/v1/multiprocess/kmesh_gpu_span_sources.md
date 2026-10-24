# GPU-backed canonical span sources

Status: experimental, qualified in a bounded RTX 4090 model replay. Native
arena controls and the registered serial consumer pass their separate gates.
Matched NVMe restores into pinned-host and GPU DMA-BUF arenas preserve the
serial reference's K/V, logits, masks, selections and generated tokens. Seven
alternating pairs show lower read-call medians but mixed complete-restore time.
GPU integrity checking still copies the full canonical payload to the host.
See the [replay and measured limits](https://knlp.io/kmesh-recovery.html#registered-storage).
Lazy CPU-L1 allocation remains the default reference; ordinary multiprocess
KMesh serving stays disabled.

## Use the existing raw-block ownership path

The storage comparison should use the existing raw-block L2 adapter: a pinned
host L1 with ordinary fixed-buffer reads for the staged arm, and a GPU L1 with
DMA-BUF registration for the direct arm. Select the existing explicit
Vulkan/RM allocator where native CUDA export is unsupported. Keep the managed
raw context's `require_inactive_gds` guard: neither arm needs the separate
cuFile/hipFile GDSContext singleton or its slab teardown behavior.

Do not remove that guard to make the experiment run. A future consumer of the
separate GDSContext slab needs checked native deregistration and closure before
it can use managed raw retirement. The raw-block adapter instead has its own
close outcome and retained arena ownership, which must remain authoritative.

Use a fresh caller-selected file on the qualified filesystem, with identical
chunk geometry, slot layout, queue depth and model requests. Do not change
kernels, storage-device formatting or PCI settings. Record the active CPU and
GPU arena costs separately; equal canonical capacity does not make their
physical memory costs identical.

## Preserve canonical bytes and source reservations

Extend native gathering only for actual contiguous BF16 GPU memory objects
from the owning GPU allocator and the same device as the registered context.
Keep the pinned-host checks and registration-boundary checks for CPU sources.
Device pointers must never pass through the executor's host-copy path.

For interior source positions and partial tails, retain the exact packed view,
full canonical object and repacking scratch before the first gather can queue
work. Drain the accepting device before consuming gathered bytes. Repack valid
rows and zero padding on the registered stream, then drain that work before
publishing source admission or handing bytes to the storage controller. A
failed drain retains every object, scratch tensor, borrow and physical lease.

Keep the existing full-payload integrity contract. An initial GPU-object
implementation may copy canonical data to CPU for the existing SHA-256 check.
That cost must be measured and reported: this would exercise direct NVMe reads
but would not demonstrate elimination of every host payload copy. Do not label
an unchecked GPU restore a successful fast path. A GPU integrity implementation
would need its own qualification before replacing the reference check.

## Stage GPU sources without a host copy

The current flat native plan hardcodes host-to-device staging. Preserve that
ABI and its pinned-host behavior. For the bounded GPU case, prepare actual
device-to-device copies into its existing private working slots on the
registered stream, then emit empty native host-staging rows and the unchanged
rotation/scatter rows. Validate source dtype, shape, device and complete object
length before any copy. Retain source and working buffers through device drain.

Initially admit only plans whose slots are all distinct and fit the available
working set. Refuse a wave that would reuse a slot before its earlier rotation
and scatter. The current model replay restores one canonical object per span
operation, so this bound does not require a new multiwave scheduling algorithm.
Rotate the working copies, never the immutable canonical source. Keep the
partial valid extent and padded rotary-capacity rules unchanged.

The eager pinned-host arm uses the same bounded preparation and empty native
host-staging tables, with asynchronous host-to-device copies on the registered
stream. Validate its actual pinned arena and registration-window boundary.
This keeps the working-copy interface matched to the GPU arm without changing
the existing lazy-host planner used by ordinary serving.

## Prove the storage route and consuming model

Observe actual source keys admitted by the owned service. Wait for successful
L2 lookup of those exact keys and release the lookup locks before deleting their
L1 copies. A successful L1 admission, aggregate byte count or elapsed sleep is
not proof of persistence. Use the adapter's public lookup and unlock interfaces;
do not steal another controller's completion receipts.

Before each measured child, remove the relevant L1 copies only after confirmed
storage admission. Require actual NVMe reads on restore, record registration
mode and account for integrity checking, positional recovery and model repair.
Keep the same immutable source, destination positions and serial workload on
both routes. Compare actual model masks, selections, repaired K/V and logits.

Qualify GPU gather/repacking, canonical preservation, shifted restore and
physical drain with synthetic NHD/HND payloads before the model comparison.
Keep corrupt/evicted source and failed-drain controls. A completed I/O receipt
still means loaded-needs-repair, and no unresolved copy may permit request
fallback or block reuse. The registered worker's exact retirement covers engine
imports; the owned server must also settle or retain its separate raw-block
arena until actual process exit.
