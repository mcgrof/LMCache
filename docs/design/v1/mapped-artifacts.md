# Borrowing immutable file-backed artifacts

`MappedArtifactStore` manages published external artifacts without copying
their payload into an allocator-owned host arena. It shares publication and
read-plan admission with `ArtifactStore`, but yields `MappedArtifactView`
instead of a `MemoryObj`. Do not present mapped storage as an allocator-owned
KV block or pass it to an allocator's free operation.

The catalog binds exact identity, path, offset, length, digest and consumer
manifest. Publication always checks the full digest before registering the
record. Admission caches metadata. On a mapping miss, the store compares the
current catalog record, opens the regular file, checks its extent with fstat,
and maps only the page-aligned containing range. A source descriptor closes
after mapping; Python 3.13's trackfd=False also avoids mmap's duplicate fd.
Earlier Python keeps one fd per retained mapping.

`ArtifactVerification.ON_MAP` hashes every newly mapped payload before exposing
it. `PUBLISHED` trusts the bytes checked during publication and does not scan
the payload during serving. It does not detect corruption after publication.
Neither policy permits in-place writers or truncation of backing files during
mapping use. A private copy-on-write mapping lets tensor consumers expose the
buffer without a read-only-buffer warning; consumers still must not mutate
borrowed data. It does not create a private payload copy on a read.

A lease takes a reference to the backing wrapper before handoff. Active leases
prevent lookup eviction and store close. When idle, eviction drops the lookup
reference. Exported memoryviews and tensors hold the mmap object independently,
so Python releases its address range only after the last reference disappears.
The implementation must never explicitly close an mmap while a tensor can
still refer to it. This also makes exceptional cleanup safe for escaped views.

The map cache is bounded by entry count. It does not bound physical page-cache
RAM, caller-retained views, or mapped virtual bytes. Linux reclaims clean file
pages. Counters distinguish mapped logical bytes, full checksum bytes, hits and
capacity waits; buffered read/copy bytes remain zero. Concurrent first borrows
coalesce under the store lock. Mapping and strict hashing are serialized;
steady borrows release the lock before consumption. Typed tensor construction
and format conversion belong to the consumer, not this backing implementation.

The mapped option is additive. Existing arena constructors, MemoryObj ownership,
miss verification, capacity handling and leases keep their prior behavior.
