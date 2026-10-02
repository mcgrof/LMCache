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
