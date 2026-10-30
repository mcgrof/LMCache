Borrowing mapped artifacts
==========================

``lmcache.v1.mapped_artifacts.MappedArtifactStore`` provides a process-local
backing for immutable, published external files. It avoids a file-to-arena
payload copy. This API supplies external artifact bytes, not token-keyed KV
entries or allocator-owned ``MemoryObj`` objects.

The caller supplies an ``ArtifactCatalog`` implementation with immutable
``register_artifact`` and ``lookup_artifact`` operations. Publication verifies
the complete file range before recording it. For example:

.. code-block:: python

   from lmcache.v1.mapped_artifacts import (
       ArtifactVerification,
       MappedArtifactStore,
   )

   store = MappedArtifactStore(
       catalog,
       max_mappings=512,
       verification=ArtifactVerification.PUBLISHED,
   )
   store.publish(artifact_id, manifest, path, sha256, length, offset=0)
   plan = store.prepare(artifact_id)
   with store.lease(artifact_id, plan=plan) as backing:
       consume(backing.byte_array)
   store.close()

``PUBLISHED`` trusts previously verified payload bytes and the catalog record
the store currently admits. The default, ``ON_MAP``, rechecks the catalog
record and the full checksum on each new mapping. Neither mode permits
backing-file changes during use. Published mode does not detect later payload
corruption or catalog edits made behind the catalog API. Consumer identity and tensor layout admission remain the typed
integration's responsibility.

Keep the lease through asynchronous consumption. Never modify borrowed data.
The backing uses a private copy-on-write mapping so tensor libraries can expose
its buffer. Reads share file-backed pages; writing a view creates private pages.
Eviction drops lookup ownership without explicitly closing exported storage.
Retained buffers and tensors keep the mapping alive until their final release.

``max_mappings`` limits lookup-resident entries, not resident RAM or caller-held
views. Linux controls physical page residency. Python 3.13 avoids retaining a
file descriptor per mapping; older Python retains one. ``map_bytes`` counts
logical mapped ranges, ``checksum_bytes`` counts verified bytes, and
``read_bytes`` stays zero because there is no buffered payload copy. These
counters are not physical storage I/O measurements.
