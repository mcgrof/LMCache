# Optional Vulkan/RM staging

Implemented for the in-process raw-block GPU pool only. The default native
CUDA/ROCm allocator is unchanged. Explicit `vulkan_rm` selection creates one
dedicated, device-local, non-host-visible Vulkan buffer, with OPAQUE_FD as its
only external handle type. The optional helper matches the CUDA UUID against
Vulkan and RM, cross-checks the PCI identity, and requires the 615.71.09 RM ABI.
It exports the owned object through the RM core `nv_dmabuf` interface, duplicates
the opaque descriptor for CUDA import, and maps that same allocation into CUDA.
No Vulkan queue work or additional payload copy is submitted.

The returned byte tensor has an external-storage deleter which owns the complete
CUDA, RM, Vulkan and descriptor graph. Every tensor alias retains that deleter.
The Python allocator owns a separate duplicate of the core DMA-BUF descriptor
for fixed-buffer registration; it resolves only its logical pool extent. Both
the logical extent, Vulkan allocation and RM physical backing must fit within
1 GiB. RM can round backing to GPU pages; this never widens the exported or
CUDA-visible logical pool extent.

Backend shutdown must first establish storage quiescence and unregister buffers.
Unknown I/O retains the existing whole-backend ownership graph. Allocator close
closes its registration descriptor and stops new suballocations; it does not
destroy delayed tensor views. The final storage deleter joins CUDA work before
freeing the CUDA mapping and external memory, RM objects, then Vulkan buffer
and memory. A failed cleanup stops destruction, retains remaining resources
until process teardown, and rejects further provider initialization.

Gather/scatter still run through the repaired V2/V3 readiness and padding path.
SYNC_MEMOPS is set and checked on the external CUDA mapping. Ordering/flush
capabilities are queried and logged, not forced. Host-observed I/O completion
must precede dependent CUDA work; prequeued GPU polling consumers are not
supported by this integration. Interop allocation is not a stream fence.

There is no automatic provider retry, arbitrary Torch-pointer conversion,
CPU payload fallback, vLLM allocation change, or driver/topology modification.
Unsupported ABI, UUID, extents, dependencies and registrations fail setup.
The helper is built only with explicit opt-in against clean, pinned NVIDIA SDK
headers; those headers are an external build dependency, not vendored source.

CPU tests cover policy selection, size limits and explicit failures. Opt-in
hardware tests cover alias ownership, larger pool extents, paged offsets and
close, ordinary backend store/retrieve, plus the actual
V2/V3 file ordering/media-padding test. Their pass is not stock-vLLM serving,
raw-device P/D, native-CUDA regression on another GPU, reset recovery or
performance qualification. See the [user guide](../../../source/kv_cache/storage_backends/raw_block_dmabuf.rst).
