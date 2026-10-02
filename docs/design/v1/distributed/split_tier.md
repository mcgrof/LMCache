# Process-local split-tier K/V placement

## Composite identity and cleanup

A split-tier cache entry keeps the exact K tensor in L1 and the encoded V
object in filesystem L2. A logical key derives two child keys while preserving
its model, rank, object group and cache salt. `SplitTierManifest` owns the
process-local generation and state of that composite. It is not persisted.
The manifest also requires one logical hash width for all live entries and
records each derived K child's owner. Those rules keep the suffix-based child
namespace disjoint even though `ObjectKey` permits arbitrary hash bytes and
prevent a logical hash ending in a role marker from being classified as a child.

A store registers a generation, publishes K, then completes only after V is
stored. Readers accept only `COMPLETE`. Physical cleanup must first claim
`DELETE_IN_FLIGHT`; a replacement cannot reuse the child names until cleanup
ends. If K was locked and nothing was deleted, cleanup can restore the prior
state. A stale generation cannot complete or remove a replacement's entry.

## External L1 objects and private tensor pools

The K-child pool allocates separate CPU buffers; it does not narrow or alias
the producer's full K/V allocation. L1 owns the catalog entry and read/write
locks, stamps its manager identity, and returns the buffer to its external
allocator when the entry is removed. V scratch pools hold private tensors for
the codec and are selected by shape and dtype.

Ordinary L1 allocation and external pools cannot satisfy each other's requests.
Eviction therefore uses the maximum individual pool utilization. Summed bytes
remain useful for observability. Pool slot limits bound retained free buffers,
not peak concurrent allocations; overflow uses temporary allocations.

## Early release of the producer's allocation

An adapter implementing `EarlyReleaseStoreAdapter` can claim source keys once
it has copied all source bytes. StoreController releases those read locks once
and arms redundant logical residents for deletion after their final reader.
The deletion marker belongs to that exact resident, not a key-only side table.
An unrelated adapter's completion cannot authorize split-tier deletion.

## Paired eviction

L1 eviction claims the manifest generation before deleting K or V. If a reader
wins the K lock, it cancels the cleanup claim and preserves the composite.
Otherwise it waits for physical V deletion to terminate before dropping the
claim. Adapter removal detaches future paired passes and drains prior passes
before closing the adapter they captured.

## Serialization and restoration

The wrapper copies K into its L1 child and V into private scratch before
allowing early release. Allocation or codec failure rolls back the owned
buffers and generation. Lookup maps a logical key to its V child and masks
incomplete composites. Load checks the exact K layout, copies K into the
caller-provided destination, and decodes V from L2. A missing component is a
miss. Supporting several K/V pairs in one object is outside this interface.
