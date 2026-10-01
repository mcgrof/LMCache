KV Cache Compression
====================

LMCache supports a **per-adapter serde** that transforms KV cache data on
its way to and from an L2 adapter. Typical uses: quantization (shrink
storage footprint), compression, encryption.

.. contents::
   :local:
   :depth: 2


When to use serde
-----------------

- **Save L2 storage or bandwidth.** fp8 quantization halves byte volume
  vs. bf16 with minor accuracy loss — a good fit for disk / remote
  adapters.
- **Encrypt at rest.** Wrap the raw bytes with authenticated encryption
  before they land on disk.
- **Custom compression.** Anything lossless (lz4/zstd) or lossy
  (CacheGen-style) can be plugged in via the ``Serializer`` /
  ``Deserializer`` ABCs.

Serde is **opt-in per adapter**: one ``--l2-adapter`` may use fp8 while
another stores raw bytes. When omitted, the adapter behaves exactly as
if serde did not exist (no extra allocations, no extra threads).


Configuring serde on an L2 adapter
----------------------------------

Add a ``"serde"`` sub-dict to any ``--l2-adapter`` JSON spec. The ``type``
field selects a registered serde; remaining keys are forwarded to the
serde factory.

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 \
        --eviction-policy LRU \
        --l2-adapter '{
            "type": "fs",
            "base_path": "/data/lmcache/l2",
            "serde": {"type": "fp8", "fp8_dtype": "float8_e4m3fn"}
        }'

.. list-table:: Built-in serde types
   :header-rows: 1
   :widths: 22 38 40

   * - ``type``
     - Description
     - Config fields
   * - ``fp8``
     - Quantize each element to 8-bit float; dequantize on load.
       Lossy but highly compressible.
     - ``fp8_dtype`` (default ``float8_e4m3fn``; also accepts
       ``float8_e5m2``), ``max_workers`` (thread pool size,
       default 1)
   * - ``turboquant``
     - Compress KV tensors with TurboQuant presets before L2 store and
       reconstruct them on load.
     - ``preset`` (default ``turboquant_k8v4``), ``head_dim`` (optional,
       default 128), ``block_size`` (default 16), ``max_workers`` (thread
       pool size, default 1)
   * - ``aesgcm``
     - AES-GCM authenticated encryption of KV bytes at rest in L2, keyed
       per ``cache_salt``.
     - ``key_provider`` (default ``hkdf``), ``master_key_path`` (required
       for ``hkdf``), ``aes_bits`` (``128`` default, or ``256``),
       ``max_workers`` (thread pool size, default 1)
   * - ``asym_k16_v8``
     - Store native K and scale-quantized FP8 V together in one durable
       object.
     - ``fp8_dtype`` (default ``float8_e4m3fn``), ``scale_scope``
       (default ``PER_TENSOR``), ``scale_dtype`` (default ``float32``),
       ``max_workers`` (default 4)
   * - ``asym_k16_v8_v_only``
     - Keep native K in L1 and store only scale-quantized FP8 V in L2.
     - Same fields as ``asym_k16_v8``; ``k_dtype_tag`` is a
       non-authoritative header placeholder
   * - ``asym_bytethrough_k16_v8``
     - Store native K and an already-FP8 V together without quantizing or
       dequantizing V.
     - ``fp8_dtype`` (only ``float8_e4m3fn``), ``max_workers`` (default 4),
       optional ``layout_provenance`` mapping
   * - ``asym_bytethrough_k16_v8_v_only``
     - Keep native K in L1 and store an already-FP8 V in L2 without a scale
       payload.
     - ``fp8_dtype`` (only ``float8_e4m3fn``), ``k_dtype_tag`` placeholder,
       ``max_workers`` (default 4)


Asymmetric K16/V8 serde
-----------------------

The asymmetric serdes expose K and V as separate typed groups.  The
``asym_k16_v8*`` variants accept a native fp16/bf16 V and store an FP8 V plus
its scale.  The ``asym_bytethrough_k16_v8*`` variants are for a serving engine
whose V is already ``float8_e4m3fn``; they preserve the raw FP8 code bytes and
therefore require the producing and restoring attention layers' external
``_v_scale`` to be exactly ``1.0``.

Choose the non-``_v_only`` type for a self-contained L2 object that can be
reloaded after a process restart.  For example:

.. code-block:: json

    {
      "type": "fs",
      "base_path": "/data/lmcache/l2",
      "serde": {
        "type": "asym_bytethrough_k16_v8",
        "fp8_dtype": "float8_e4m3fn",
        "layout_provenance": {
          "model_id": "org/model",
          "model_revision_hash": "sha256:...",
          "tokenizer_hash": "sha256:...",
          "rope_config_hash": "sha256:...",
          "attention_backend": "flashinfer",
          "kv_layout": "vllm-block16-kbf16-vfp8-e4m3"
        }
      }
    }

``layout_provenance`` is optional, but a deployment that shares stored data
across engines should set it.  A reader configured with different provenance
rejects the object as a cache miss rather than interpreting incompatible KV
bytes.

Choose an ``_v_only`` type only for the split-tier optimization.  It mirrors K
under an L1 child key and writes V under an L2 child key; an in-memory manifest
makes the logical key visible only after both sides are ready.  For example:

.. code-block:: json

    {
      "type": "fs",
      "base_path": "/data/lmcache/v-only",
      "serde": {
        "type": "asym_k16_v8_v_only",
        "fp8_dtype": "float8_e4m3fn",
        "scale_scope": "PER_TENSOR"
      }
    }

The split-tier path deliberately has a narrow support matrix and fails at
startup outside it:

- CPU pinned-DRAM L1 only; GDS and Device-DAX L1 are unsupported.
- Exactly one filesystem L2 adapter.  S3, Valkey, peer adapters, and runtime
  attachment of a second adapter are unsupported.
- No per-adapter L2 eviction and no ``IsolatedLRU`` quota policy.
- A single object group (``object_group_id == 0``).

Split-tier entries are process-local composites, not restart-persistent cache
objects.  Restarting loses the manifest and L1 K child, so an existing V child
is treated as a miss.  Eviction, clear, adapter removal, and failed stores use
paired cleanup so one child is not exposed as a valid logical hit.

The byte-through (``RAW_UNIT``) formats have an additional filesystem-only
restriction, including together mode. Their decoder rejects trailing bytes,
and S3 and Valkey remain unsupported until they provide the same used-length
contract. A truncated fixed-layout FP8 file is also a miss; only explicitly
marked variable-length byte buffers may narrow their used size.


Serialized-size contracts
-------------------------

Every registered serde declares whether ``estimate_serialized_size`` is exact
or only an upper bound. Upper-bound formats currently require the filesystem
adapter because it reports the object's actual loaded length before
deserialization. This includes the scale-aware ``asym_k16_v8`` formats, whose
self-describing headers are shorter than their allocation allowance. S3 and
Valkey pairings fail during configuration instead of turning every valid object
into a load miss.


TurboQuant serde
----------------

TurboQuant serde can be enabled by setting ``"type": "turboquant"`` in the
adapter serde config. If ``preset`` is omitted, TurboQuant serde defaults to
``turboquant_k8v4``.

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 \
        --eviction-policy LRU \
        --l2-adapter '{
            "type": "fs",
            "base_path": "/data/lmcache/l2",
            "serde": {
                "type": "turboquant",
                "preset": "turboquant_k8v4",
                "block_size": 16
            }
        }'

Supported presets:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Preset
     - Key path
     - Value path
   * - ``turboquant_k8v4``
     - FP8 key
     - 4-bit value quantization
   * - ``turboquant_4bit_nc``
     - 4-bit MSE key with norm correction
     - 4-bit value quantization
   * - ``turboquant_k3v4_nc``
     - 3-bit MSE key with norm correction
     - 4-bit value quantization
   * - ``turboquant_3bit_nc``
     - 3-bit MSE key with norm correction
     - 3-bit value quantization


Encryption serde
----------------

The ``aesgcm`` serde encrypts KV bytes with AES-GCM before they land in L2
and decrypts them on load, so a party who can read the remote storage
(bucket / disk / RESP) cannot recover cache contents. Each tenant's data is
encrypted under a distinct key derived from its ``cache_salt``.

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 \
        --eviction-policy LRU \
        --l2-adapter '{
            "type": "s3",
            "bucket": "my-kv-cache",
            "serde": {
                "type": "aesgcm",
                "key_provider": "hkdf",
                "master_key_path": "/etc/lmcache/keys/master"
            }
        }'

Provide the master key as a file (e.g. a mounted Kubernetes ``Secret``).
The ``hkdf`` provider reads it once at startup and derives a per-``cache_salt``
key via HKDF-SHA256; the master key is never written to L2. ``aes_bits``
defaults to ``128`` (already unbreakable and slightly faster); set ``256`` if a
compliance mandate requires it.

.. warning::

   The ``hkdf`` provider derives every tenant's key from one shared master
   key, so any server holding the master can decrypt any tenant's data
   ("fleet vs. outside" trust). It is not per-tenant access isolation.



Writing a custom serde
----------------------

Implement the two sync ABCs (``Serializer``, ``Deserializer``) with your
transform logic, then register a factory keyed on a name you pick:

.. code-block:: python

    # my_project/my_serde.py
    from lmcache.v1.distributed.serde import (
        AsyncSerdeProcessor,
        Deserializer,
        SerdeSizeContract,
        Serializer,
        register_serde_factory,
    )

    class MySerializer(Serializer):
        def serialize(self, src, dst, key) -> int:
            # Write serialized bytes into dst; return bytes written.
            # ``key`` is the object's ObjectKey; ignore it unless your
            # transform is keyed per object (e.g. encryption keyed on
            # cache_salt).
            ...

        def estimate_serialized_size(self, layout_desc) -> int:
            # Upper bound on serialized byte size for this layout.
            ...

    class MyDeserializer(Deserializer):
        def deserialize(self, src, dst, key) -> None:
            # Read serialized bytes from src, write into dst (KV-shaped).
            # ``key`` mirrors serialize; ignore it unless keyed per object.
            ...

    def _create_mine(config: dict):
        return AsyncSerdeProcessor(MySerializer(), MyDeserializer())

    register_serde_factory(
        "mine",
        _create_mine,
        size_contract=SerdeSizeContract.EXACT,
    )

Reference it from your adapter config:

.. code-block:: json

    {"type": "fs", "base_path": "/data", "serde": {"type": "mine"}}


Notes
-----

- **Buffer size.** ``estimate_serialized_size(layout)`` must return an
  upper bound on the actual serialized output. Register
  ``SerdeSizeContract.EXACT`` only when every successful serialization writes
  exactly that estimate. Otherwise omit the argument and use the safe
  ``UPPER_BOUND`` default; such serdes currently require the filesystem
  backend's actual-used-length load contract.
- **Raw-byte output.** If your serde writes bytes directly (rather than
  through ``MemoryObj.tensor`` like the quantizers), reach the buffer via
  ``MemoryObj.byte_array`` and **cast it to the native format first**:
  ``memoryview(dst.byte_array).cast("B")``. ``byte_array`` is a
  ctypes-backed view with format ``"<B"``, and CPython does not support
  slice assignment into a non-native format (``dst[i:j] = ...`` raises
  ``NotImplementedError: memoryview: unsupported format <B``). The built-in
  ``aesgcm`` serde is an example.
- **Failure handling.** If any step fails (serialize, store, load, or
  deserialize), the whole submitted batch is reported as failed —
  partial success within one batch is not surfaced. Failed keys are
  cleaned up automatically.
- **Thread pool.** ``AsyncSerdeProcessor(max_workers=N)`` controls the
  pool size. Transforms that release the GIL (e.g., torch ops)
  benefit from ``N > 1``; pure-Python transforms do not.


Example
-------

An end-to-end script that starts an lmcache server with fp8 on a disk
adapter, runs vLLM, clears L1, and re-runs the same request to trigger
the L2 prefetch + fp8 deserialize path lives at
:file:`examples/serde/fp8/`. A pytest-based filesystem round-trip test
(no vLLM required) is at
:file:`tests/v1/distributed/serde/test_serde_fs_e2e.py`.

Compression methods
-------------------

Higher-level KV cache compression methods layered on top of the per-adapter
serde mechanism:

.. toctree::
   :maxdepth: 1

   cachegen
