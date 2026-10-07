Raw-block DMA-BUF offload
=========================

This experimental in-process route stores ordinary LMCache chunks through
``RustRawBlockBackend`` using fixed-buffer, ordinary ``io_uring`` I/O. A paged
GPU staging pool is the connector's allocation endpoint: V2/V3 gather into it,
storage writes that allocation, storage reads into another pool slot, and the
connector scatters from it. There is no mandatory CPU cache tier and no NIXL
connector in this path. This is standalone store/retrieve, not a cross-process
prefill/decode publication protocol.

Requirements and limits
-----------------------

The Rust raw-block extension, a kernel implementing DMA-BUF fixed-buffer
registration for ordinary ``io_uring``, a working GPU exporter, and a qualified
device/filesystem/topology are required. Stock Linux does not provide this
registration ABI. NVIDIA export needs open kernel modules and
``cuMemGetHandleForAddressRange``; ROCm export uses
``hsa_amd_portable_export_dmabuf`` and preserves driver-exported byte offsets
without widening the pool's logical slice bounds.
Exporter support alone does not establish working device I/O.
Registered I/O journals identify the native route, not physical VRAM residency
or the absence of driver-managed migration or staging.

GPU buffers must be page aligned, contiguous, fully exported and fully
registered. The current representation supports one memory-object group;
grouped device objects are refused before storage slots are reserved. Short
registered transfers fail rather than retrying an invalid DMA-BUF offset.
Paged partial chunks retain a full ``physical_tensor`` slot for aligned I/O
while the connector's logical views remain narrowed to the chunk's token count.
Before handing a gathered slot to storage, V2/V3 zero its physical tail on the
store stream and wait for that stream to finish. Aligned writes can therefore
include padding, but that padding must be zero, including after pool reuse.
The logical payload size and physical I/O size remain separate.
GPU staging does not fall back to host-pointer registration, POSIX I/O, or
NVMe ``io_uring_cmd``. Unsupported export or registration fails setup.

The pool is bounded and has no evictor. Exhaustion returns no allocation;
callers can skip a store or recompute a miss. Producer and gather readiness
are fenced before external storage DMA reads the staging bytes. A failed
wait is not a completion fence: unknown native outcomes seal admission and
retain source/destination owners, slots, exports and backing arenas. Native
close must prove quiescence before the GPU allocator can release its exports.

Example configuration
---------------------

.. warning::

   Raw-block storage writes its device/file and checkpoint area. Use only
   explicitly assigned disposable storage, never a mounted or shared device.
   Set capacity and metadata reservations to fit that target.

.. code-block:: yaml

   chunk_size: 256
   local_cpu: false
   max_local_cpu_size: 0
   storage_plugins: [raw_block]
   extra_config:
     storage_plugin.raw_block.module_path: lmcache.v1.storage_backend.plugins.rust_raw_block_backend
     storage_plugin.raw_block.class_name: RustRawBlockBackend
     rust_raw_block.device_path: /path/to/disposable-preallocated-file
     rust_raw_block.io_engine: io_uring
     rust_raw_block.use_odirect: true
     rust_raw_block.gpu_buffer_bytes: 1073741824
     rust_raw_block.gpu_buffer_device: cuda:0
     rust_raw_block.block_align: 4096
     rust_raw_block.header_bytes: 4096
     rust_raw_block.meta_total_bytes: 268435456

The pool size and chunk geometry come from engine metadata, including the
tensor-parallel shard. Each TP rank needs its own explicitly mapped storage
target. ``max_data_transfer_size`` can cap ordinary ``io_uring`` transfers;
alignment constraints still apply. This caps submitted operations; block-layer
splitting and merging mean it does not guarantee one NVMe command per operation.
Default slot geometry includes the header and a full aligned payload.
GPU export alignment follows the host page size, which can differ from the
storage block alignment. A 64 KiB alignment contract test does not qualify a
64 KiB host's exporter or kernel; test that configuration separately.

Host staging is a separate option: ``local_cpu_dmabuf: udmabuf``,
``system_heap``, ``cma_heap``, or an explicit ``/dev/dma_heap/<name>`` exports
the CPU pool. Udmabuf's memfd mapping may be host registered through the
platform interface. DMA-heap PFN mappings remain unpinned. CPU registration
may fall back to ordinary fixed buffers; that does not qualify DMA-BUF I/O.
If partial registration cannot be safely undone, setup fails and retains its
owners instead of falling back. Registration establishes a kernel-owned buffer
table; when device mappings are established depends on the kernel and importer.

Validation
----------

CPU tests cover descriptor unwind, complete ranges, independent packed-layout
references, deferred stream ordering, partial chunks, ownership faults and
allocator shutdown. They do not qualify a GPU, exporter or kernel.

The explicit CUDA/ROCm hardware test uses an exclusive new regular file,
independent per-direction CPU oracles, a nonzero GPU export offset, repeated
generations and warm registration. Partial chunks start in a nonzero-filled
slot; an independent direct read checks every padding byte on storage. The test
checks the native DMA-BUF I/O journal and rejects fallback and launch-blocking
settings:

.. code-block:: bash

   LMCACHE_RUN_DMABUF_ORDERING=1 LMCACHE_DMABUF_TEST_DIR=/qualified/filesystem \
     pytest -xvs tests/v1/gpu_connector/test_vllm_staging_ordering_dmabuf.py

This native connector/file test is not an ordinary engine-level store/retrieve
or raw-device P/D validation. Those workloads require separate runtime gates.
MP's server-owned raw-block adapter and its evolving L1 interfaces are not
qualified as a GPU-staging route by this in-process integration.
