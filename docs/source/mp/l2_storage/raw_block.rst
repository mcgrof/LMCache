Raw Block (Rust)
================

A built-in L2 adapter for multiprocess (MP) LMCache that stores KV objects in
fixed-size slots on a raw block
device or pre-sized file using the Rust raw-device I/O bindings. It reuses the
existing raw-block metadata checkpoint model and writes directly into the
caller-provided load buffers during prefetch.

Execution supports ``io_engine="posix"`` and the ordinary Rust
``io_engine="io_uring"`` path. Controllers lease each submitted L1 buffer
until the adapter returns ownership, independently of task success. If native
completion is unresolved, the server retains and seals the affected L1 arena
instead of letting another request reuse it. Restart the owning process group
only after establishing device quiescence.

MP can register eager host L1 regions as classic fixed buffers or DMA-BUFs.
Classic registration pins host pages for the ring. DMA-BUF registration
imports exported host memory through the kernel's DMA-BUF direct-I/O path.
DMA-BUF is Linux's file-descriptor interface for sharing memory with drivers.
MP also supports an eager GPU L1 arena through DMA-BUF. The server gathers
engine KV tensors into its own GPU arena through CUDA interprocess sharing,
then submits the exported arena directly to ordinary io_uring. It scatters
reloaded objects back into the engine's GPU tensors. This path uses the
existing :doc:`/kv_cache/storage_backends/raw_block_dmabuf` exporter support.
Both host registration modes and GPU L1 can use serde; the wrapper leases its
intermediate buffers until codec and storage completion return ownership.
NVMe passthrough and Flexible Data Placement require separate device
qualification; placement-policy unit tests do not qualify hardware execution.

Register the host L1 pool
-------------------------

Set ``--l1-buffer-registration fixed`` or ``dmabuf`` together with
``--no-l1-use-lazy``. The default ``none`` keeps unregistered I/O. Registration
requires ordinary ``io_uring``, ``use_odirect=true`` and
``enable_zero_copy=true`` on each attached raw-block adapter. NVMe
``io_uring_cmd`` is rejected for these MP modes.

For example, register a 1 GiB host pool exported by ``/dev/udmabuf``::

    lmcache server --l1-size-gb 1 --eviction-policy LRU \
      --no-l1-use-lazy --shm-name "" --l1-buffer-registration dmabuf \
      --l2-adapter '{"type":"raw_block","device_path":"/mnt/cache/kv.bin","slot_bytes":1048576,"io_engine":"io_uring","use_odirect":true}'

The file must already contain initialized extents across the selected capacity.
Use a dedicated NVMe-backed XFS or ext4 file, or a dedicated NVMe block device,
on a booted kernel implementing this DMA-BUF registration ABI. The underlying
block device must also support DMA-BUF import; an ext4 or XFS mount alone does
not establish that capability. Sparse files, holes,
unsupported filesystems and failed registration reject setup. DMA-BUF mode
does not fall back to classic registration or ordinary I/O during setup.
It requires access to ``/dev/udmabuf`` and disables POSIX shared memory;
the MP server owns a sealable memfd and its exports. Hugepages remain optional
and require a preallocated pool. DMA-BUF alignment cannot exceed the backing
page size: 4 KiB normally or 2 MiB with hugepages.

Classic ``fixed`` mode can use POSIX shared memory with alignment up to 4 KiB.
The process needs enough locked-memory allowance for every ring's registration.
Lazy allocation, Device-DAX and storage-backed L1 tiers cannot use these modes.
For tagged pools, put ``"buffer_registration":"dmabuf"`` and
``"use_lazy":false`` inside that DRAM ``--l1-manager`` JSON object.

The allocator divides the arena into regions of at most 1 GiB to fit the
supported registration ABI. Objects and their physical alignment padding
stay within one region. Each attached ring registers the regions independently;
the L1 manager owns the mapping and all export FDs until every adapter proves
shutdown. A failed constructor also retains the affinity arena if cleanup
cannot prove that it removed a partial registration.

``report_status()["core"]["buffer_registration_mode"]`` reports setup mode.
The ``io_by_request`` native journal distinguishes ``host_fixed`` and
``dmabuf_fixed`` payload operations from ordinary or bounce I/O, using the MP
task ID and a distinct adapter incarnation. Slot headers and checkpoints use
separate buffers and can still bounce. Registration alone is not a bandwidth
result or proof of GPU-direct storage.

Use a GPU L1 arena
------------------

Choose a tagged ``GPU`` manager and ``--supported-transfer-mode lmcache_driven``::

    lmcache server --supported-transfer-mode lmcache_driven \
      --l1-manager '{"type":"GPU","tag":"gpu","size_gb":1,"device":"cuda:0","eviction":{"eviction_policy":"LRU"}}' \
      --l2-adapter '{"type":"raw_block","affinity_tag":"gpu","device_path":"/mnt/cache/kv.bin","slot_bytes":1048576,"io_engine":"io_uring","use_odirect":true,"serde":{"type":"fp8"}}'

The default ``buffer_provider="native"`` requires a working CUDA DMA-BUF
exporter and the kernel's DMA-BUF direct-I/O ABI. Export or registration
failure aborts startup. There is no host-memory fallback. The explicit
experimental ``buffer_provider="vulkan_rm"`` supports a single arena of at
most 1 GiB and requires the exporter dependencies described in the in-process
backend documentation. GPU L1 uses 4096-byte alignment and process-local
memory; it cannot use lazy allocation, host shared memory, hugepages or
Device-DAX. Peer transfer channels are unsupported with GPU L1.

GPU L1 currently supports plain payloads and ``fp8`` serde. CPU byte-view
codecs, including AES-GCM, are rejected for this arena. Codec intermediates
also consume GPU L1 capacity, so budget room for both KV objects and their
encoded buffers. Codec workers fence their GPU stream before reporting a
result, and the adapter fences padding work before storage submission.
The GPU gather/scatter stream holds a separate lease until its completion
callback runs. A failed GPU fence seals and retains the arena and export FDs,
even if native storage shutdown succeeds.

This provides MP GPU staging and encoded-buffer ownership for the ordinary
io_uring path. It does not establish model accuracy for lossy FP8, native
exporter compatibility on every GPU, serving throughput, or networked
prefill/decode deployment performance. Validate the selected GPU, kernel,
filesystem and engine/model configuration together.

**Required fields:**

- ``device_path``: Raw device path or pre-sized file path.
- ``slot_bytes``: Fixed slot size in bytes. Must be aligned to ``block_align``.

**Optional fields:**

- ``capacity_bytes``: Optional cap on the usable device bytes. Default ``0``
  means use the full device/file size.
- ``use_odirect``: ``true`` or ``false`` (default ``true``).
- ``block_align``: Device alignment in bytes (default ``4096``). Must be a
  power of two.
- ``header_bytes``: Per-slot header reservation (default ``4096``).
- ``meta_total_bytes``: Reserved metadata checkpoint region (default ``256MiB``).
- ``meta_magic`` / ``meta_version``: Metadata checkpoint identity/version knobs.
- ``meta_checkpoint_interval_sec`` / ``meta_idle_quiet_ms`` /
  ``meta_enable_periodic`` / ``meta_verify_on_load``: Checkpoint and recovery
  controls carried over from the legacy raw-block backend.
- ``load_checkpoint_on_init``: Load an existing on-device metadata checkpoint
  during startup (default ``true``). Set to ``false`` to start with an empty
  in-memory index instead.
- ``enable_zero_copy``: Try aligned direct-buffer I/O when possible.
- ``io_engine``: Rust raw-block I/O engine. Valid values are ``"posix"``
  (default synchronous ``pread``/``pwrite`` path), ``"io_uring"`` (direct Rust
  io_uring syscall path).
- ``use_uring_cmd``: Enable NVMe passthrough via io_uring command interface
  for direct device access. Requires ``io_engine="io_uring"`` and NVMe
  character device node (e.g., ``/dev/ng0n1``).
- ``iouring_queue_depth``: Queue depth for ``io_engine="io_uring"``.
- ``max_data_transfer_size``: Maximum data transfer size for
  ``use_uring_cmd=true``. Large transfers are split into smaller chunks
  that fit within device limits. When left unset (or ``<= 0``), it is
  auto-detected from the device's sysfs queue limits, bounded by both
  ``max_hw_sectors_kb`` and ``max_segments * page_size`` (the NVMe passthrough
  path uses one scatter-gather segment per page).
- ``fdp_enabled``: Enables NVMe Flexible Data Placement (FDP) discovery
  and non-zero placement identifier registration. ``cache_salt`` values with
  ``":"`` bucket prefixes use FDP placement by default. Requires
  ``io_engine="io_uring"`` and ``use_uring_cmd=true``.
- ``fdp_placement_ids``: Optional non-zero placement identifier list for KV
  data placement. If omitted, the adapter uses all device-reported non-zero
  identifiers except ``meta_checkpoint_placement_id``.
- ``fdp_data_placement_policy``: KV data placement policy. ``"none"`` omits
  FDP directives for KV data writes. ``"cache_salt_prefix"`` assigns
  case-insensitive ``cache_salt`` prefixes to FDP placement identifiers.
  ``"cache_salt_rank"`` keeps that bucket isolation and further separates
  rank-local write streams within each bucket when identifiers are available.
  The default is ``"cache_salt_prefix"`` when ``fdp_enabled=true``.
- ``fdp_slot_reuse_policy``: Free-slot reuse policy. ``"pid_affinity"``
  prefers a free slot last assigned to the same FDP placement identifier before
  falling back to any free slot. ``"none"`` disables affinity-based reuse. The
  default is ``"pid_affinity"`` when ``fdp_enabled=true`` and ``"none"``
  otherwise.
- ``meta_checkpoint_placement_id``: Optional non-zero placement identifier for
  metadata checkpoint payload/header writes. Omit it to keep checkpoint writes
  on default NVMe placement.
- ``num_store_workers`` / ``num_lookup_workers`` / ``num_load_workers``:
  Worker-thread counts for each operation type.

Buffer ownership and shutdown
-----------------------------

A successful or failed task may return reusable buffers. An unresolved task
returns ``buffers_reusable=False`` through ``pop_completed_store_outcomes`` or
``query_load_outcome``. Prefetch publishes no hits from that task and keeps its
L2 source locks. Store and prefetch stop returning arena memory to the allocator
and reject new reads and allocations in the affected L1 manager. Other L1
managers keep their own ownership state.

An external I/O lease protects buffers even after a reservation expires. Forced
L1 deletion cannot reclaim a leased buffer, and ``clear`` refuses while any
lease is active. A controller-only stop has no proof that pending I/O stopped
and retains unresolved buffers. StorageManager shutdown first joins the
controllers, then closes the adapters and settles leases from their completion
or shutdown proofs before closing L1. An adapter close exception also retains
its affinity arena. Repeated StorageManager shutdown has no effect.

The legacy status-only APIs remain available for known completion. They raise
on an unresolved raw-block result without consuming it; callers must use the
ownership API and retain their arena. ``report_status()`` on L1 reports
``backing_resources_retained`` and ``external_io_lease_count``. A retained arena
is unhealthy and remains allocated for the process lifetime.

**Notes:**

- ``raw_block`` is a server-owned MP adapter. It does **not** support
  per-TP device-path mappings in MP mode.
- ``raw_block`` remains ``"type": "raw_block"`` for all supported engines.
- ``raw_block`` owns on-device slot allocation, checkpointing, and recovery
  through ``RawBlockCore``. Slot reclamation is driven by the shared/global
  L2 eviction controller or explicit ``delete()`` calls.
- POSIX restart recovery validates slot headers with an internal pool of 8
  reader threads. Both regular ``io_uring`` and ``io_uring_cmd`` batch header
  reads up to ``iouring_queue_depth``.
- ``slot_bytes``, ``header_bytes``, and ``meta_total_bytes`` must be multiples
  of ``block_align``.
- If ``use_odirect`` or ``use_uring_cmd`` is enabled, the server's
  ``--l1-align-bytes`` must be at least ``block_align``.
- With ``O_DIRECT``, raw-block I/O rejects offsets and total I/O lengths that
  are not multiples of ``block_align``. Misaligned write buffers use an aligned
  bounce buffer.
- ``persist_enabled`` must remain ``true`` for this adapter.
- For ``use_uring_cmd=true``, ``device_path`` must use the NVMe character
  device node (e.g., ``/dev/ng0n1``) instead of the block device node
  (``/dev/nvme0n1``). The character device provides direct NVMe
  command passthrough.
- For ``use_uring_cmd=true``, ``block_align`` must be a multiple of the NVMe
  namespace LBA size. An incompatible value is rejected when the device opens.
- ``use_uring_cmd`` requires ``io_engine="io_uring"`` to be set.
- When ``use_uring_cmd=true``, ``use_odirect`` is ignored for NVMe namespace
  character devices. FDP examples set ``use_odirect=false`` because
  ``io_uring_cmd`` uses NVMe passthrough rather than the POSIX write path.
- FDP registers only non-zero placement identifiers. ``fdp_placement_ids`` is
  the KV data placement pool: if omitted, all discovered non-zero identifiers
  except ``meta_checkpoint_placement_id`` are used; if provided, every
  identifier must be reported by the device and must not contain 0.
- ``meta_checkpoint_placement_id`` must not overlap with ``fdp_placement_ids``.
  Keeping metadata checkpoints and KV data on separate placement identifiers
  avoids mixing long-lived raw-block metadata with cache data buckets.
- With ``fdp_data_placement_policy="cache_salt_prefix"``, ``cache_salt`` values
  containing ``":"`` are bucketed by the prefix before ``":"``. The bucket name
  is case-insensitive, so ``RAG:app1``, ``rag:app2``, and ``rag:`` share the
  same bucket. Values without ``":"`` and values with an empty prefix use no
  FDP directive. Buckets are assigned exclusive FDP placement identifiers in
  first-seen order while identifiers are available.
- If the number of discovered cache-salt buckets exceeds the number of
  registered FDP placement identifiers, additional buckets continue to store
  with no FDP directive and the adapter emits one warning. ``report_status()``
  exposes a fallback count and a bounded bucket sample instead of retaining the
  complete fallback bucket set. Empty ``cache_salt`` values also use default
  NVMe placement with no directive.
- With ``fdp_data_placement_policy="cache_salt_rank"``, placement is keyed by
  the case-insensitive ``cache_salt`` bucket prefix and the local rank encoded
  in ``kv_rank``. The suffix after ``":"`` remains part of the object key but
  does not affect FDP placement. For example, ``batch:app-a`` and
  ``batch:app-b`` share the ``batch`` bucket: local rank 0 from both
  applications shares one placement identifier, local rank 1 shares another,
  and so on. Each bucket can use at most the detected visible GPU count worth
  of placement identifiers. Extra ranks, or buckets that cannot obtain an
  identifier, store with no FDP directive.
- Sharing placement identifiers within a bucket can be intentional because the
  available FDP placement-identifier pool is finite. Operators can assign the
  same prefix to applications with similar cache lifetimes, or use a distinct
  prefix for each application when placement isolation is required. For
  example, ``app01:<suffix>`` through ``app16:<suffix>`` can provide 16
  application buckets with eight rank streams each, consuming up to 128
  placement identifiers when at least 128 identifiers are available and the
  adapter detects at least eight visible GPUs. Applications using the same
  prefix share the placement identifiers for equal local ranks instead of
  receiving separate sets.
- Bucket-to-placement assignments are process-local. Restart recovery does not
  need them for correctness because ``cache_salt`` is part of the object key,
  but first-seen FDP placement assignments may change after adapter restart.
- Slot affinity is process-local and is not stored in metadata checkpoints.
  After recovery, free slots have no recorded affinity until they are reused.
- Store and retrieve requests must use the same ``cache_salt`` to address the
  same object. This is the normal LMCache key identity rule; FDP placement is a
  write directive and is not used to locate data on reads.
- Metadata checkpoint writes use ``meta_checkpoint_placement_id`` when
  configured, otherwise they use default NVMe placement with no directive.

**Configuration examples:**

.. code-block:: bash

    # Basic raw_block with posix I/O
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "slot_bytes": 1048576, "block_align": 4096, "header_bytes": 4096, "meta_total_bytes": 268435456, "use_odirect": true, "num_store_workers": 2, "num_lookup_workers": 1, "num_load_workers": 4}'

    # With io_uring
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "iouring_queue_depth": 256, "use_odirect": true}'

    # With io_uring_cmd (NVMe passthrough)
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "use_uring_cmd": true, "iouring_queue_depth": 256, "max_data_transfer_size": 131072, "use_odirect": false}'

    # With FDP discovery and cache_salt prefix placement enabled
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "use_uring_cmd": true, "fdp_enabled": true, "use_odirect": false}'

    # With FDP discovery and cache_salt/rank placement enabled
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "use_uring_cmd": true, "fdp_enabled": true, "fdp_data_placement_policy": "cache_salt_rank", "use_odirect": false}'

    # With FDP discovery only, keeping KV data writes on default NVMe placement
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "use_uring_cmd": true, "fdp_enabled": true, "fdp_data_placement_policy": "none", "use_odirect": false}'

    # With eviction
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "slot_bytes": 1048576, "load_checkpoint_on_init": false, "eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.9, "eviction_ratio": 0.1}}'

**Failure handling and shutdown:**

Recoverable ``io_uring`` submission errors (``EAGAIN``, ``EINTR``, ``EBUSY``)
retry after completion progress or 1–100 ms exponential backoff. Permanent
errors, or 30 seconds without submission or completion progress, mark the
worker unhealthy and stop new native I/O. The core status exposes the failure
reason as ``worker_error``.

After a known worker failure, new prefetch tasks report failure through their
normal task results and new stores are skipped. The adapter still publishes
completion results and notifications so controllers can release locks and
buffers; the serving engine can recompute missing data instead of receiving a
synchronous storage exception. Valid hits from L1 or other L2 adapters remain
usable. The legacy non-MP raw-block backend also skips new stores after worker
failure.

Already-accepted I/O retains its buffers until completion or successful
cancellation. Close and destructor cleanup release the Python GIL while
waiting, but can still block indefinitely if accepted I/O cannot finish or be
cancelled. The 30-second submission retry budget is not a shutdown timeout.

**Hardware-gated FDP status validation:**

FDP live-device validation is opt-in because it requires an FDP-capable NVMe
namespace character device. The status probe opens the character device through
the Rust raw-block binding and calls ``fetch_fdp_status()`` with a read-only file
descriptor. It does not issue writes, initialize the MP adapter layout, write KV
data, or verify the MP adapter's cache-salt placement policy.

.. code-block:: bash

    LMCACHE_TEST_FDP_CHAR_DEVICE=/dev/ng0n1 \
      pytest -q tests/v1/storage_backend/test_raw_block_fdp_status_probe.py

When the variable is not set, the test skips. If the configured device, kernel,
or controller does not support the FDP status query, the test skips with the
underlying capability error. A passing status probe only confirms that the live
device can answer the FDP status query; full adapter initialization and KV write
placement on hardware are separate validations.
