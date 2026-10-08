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
registration ABI. Native NVIDIA export needs open kernel modules and
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

Strict filesystem targets
-------------------------

``rust_raw_block.require_dmabuf_registration: true`` accepts a block device
or a regular file on qualified XFS/ext4 over NVMe. It requires ordinary
``io_uring`` and ``O_DIRECT`` in both cases. File selection does not weaken
registration, GPU ownership or publication durability. No test-only bypass
is needed for the file configuration.

The native preflight checks the opened descriptor, not only its pathname.
The configured capacity must fit the target. For files, FIEMAP checks that
initialized private extents cover the whole cache range. Sparse holes,
unwritten preallocation, reflinks/shared extents, inline or encoded data and
other filesystems are rejected. Allocate and initialize the complete range
before starting either role; ``fallocate`` alone leaves unwritten extents.
Keep the file exclusively assigned to the cache. External truncation, hole
punching, reflinking or replacement while the group is running is unsupported.
The kernel must preserve DMA-BUF direct I/O without a buffered fallback.

This is admission support, not certification of every XFS/ext4 mount or GPU.
The device/exporter pair still needs independent payload checks and serving
qualification. The publication flush contract is the same for files and block
devices. Unplanned process restart and physical reset recovery are not added.

Run the native admission regressions only in assigned scratch space:

.. code-block:: bash

   LMCACHE_TEST_STRICT_FILESYSTEM_DIR=/qualified/filesystem \
     pytest -xvs tests/v1/storage_backend/test_raw_block_filesystem.py

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

Optional Vulkan/RM NVIDIA staging
----------------------------------

``rust_raw_block.gpu_buffer_provider: native`` is the default. An experimental
``vulkan_rm`` option allocates the staging pool through Vulkan, maps the same
owned device-local allocation into CUDA using OPAQUE_FD, and exports a separate
RM core DMA-BUF for storage. This can be useful when native CUDA DMA-BUF export
returns error 801 (``CUDA_ERROR_NOT_SUPPORTED``). It does not re-export arbitrary
Torch allocations, run inference in Vulkan, or add a host payload bounce.
CUDA still runs the existing gather/scatter kernels. No vLLM option or patch
is needed: this selects LMCache's staging allocator, not vLLM's KV allocator.
The helper initializes the selected CUDA primary context before importing the
external allocation, including in a process that has not allocated a Torch
CUDA tensor yet.

The CUDA import contract is described in `NVIDIA's external-memory guide
<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#importing-memory-objects>`_.
The final DMA-BUF export uses a version-coupled RM interface, not a stable CUDA
API. This implementation supports only the NVIDIA 615.71.09 ABI, a single
unlinked GPU identity, and a pool/backing extent of at most 1 GiB. Device,
filesystem and topology qualification remains necessary; this is not a promise
of support for every RTX GPU or NVIDIA driver.

Build the optional helper with CUDA Torch, a matching CUDA toolkit, Vulkan
development headers/loader, and a clean checkout of NVIDIA's public SDK source:

.. code-block:: bash

   git clone --depth 1 --branch 615.71.09 \
     https://github.com/NVIDIA/open-gpu-kernel-modules.git /path/to/nvidia-sdk
   BUILD_WITH_VULKAN_RM=1 LMCACHE_NVIDIA_RM_HEADERS=/path/to/nvidia-sdk \
     pip install --no-build-isolation .

The build verifies SDK commit ``61dcc93722ecb418bb5f2e00923f05b4b8051dd1``.
Default builds do not require Vulkan. The native runtime does not load the
helper or enumerate Vulkan devices. Add the following to the preceding YAML:

.. code-block:: yaml

   extra_config:
     rust_raw_block.gpu_buffer_provider: vulkan_rm
     rust_raw_block.gpu_buffer_bytes: 268435456
     rust_raw_block.gpu_buffer_device: cuda:0

Keep the other raw-block settings; the three lines above are additions, not a
complete backend configuration. This is explicit selection, not automatic
fallback: allocation/export/registration errors fail setup, with no provider
switch after views or I/O have been published. Missing optional dependencies
and unsupported RM versions produce errors. The allocation owner survives all
tensor aliases and storage completion; uncertain I/O or failed cleanup retains
owners rather than recycling potentially active memory.

To exercise the actual provider and V2/V3 kernels on a qualified scratch
filesystem, use exclusive new files with the existing test:

.. code-block:: bash

   LMCACHE_RUN_VULKAN_RM=1 LMCACHE_DMABUF_TEST_DIR=/qualified/filesystem \
     pytest -xvs tests/v1/test_vulkan_rm_staging.py
   LMCACHE_RUN_DMABUF_ORDERING=1 LMCACHE_DMABUF_TEST_PROVIDER=vulkan_rm \
     LMCACHE_DMABUF_TEST_DIR=/qualified/filesystem \
     pytest -xvs tests/v1/gpu_connector/test_vllm_staging_ordering_dmabuf.py

These tests do not establish model-serving, raw-device P/D, reset/recovery or
performance qualification, nor repair an unrelated numerical/ordering failure.
