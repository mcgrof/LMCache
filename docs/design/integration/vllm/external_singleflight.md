# Exact-prefix external load coalescing

`lmcache.mp.external_singleflight: true` opts the MP connector into one
external GPU load per compatible text prefix while that load is in flight.
For example, four requests with one externally cached ancestor choose one
leader; followers return vLLM's existing query-again-later response before
allocation. They retry the normal automatic prefix cache (APC), which shares
GPU blocks after the leader publishes them. The registry does not confer
ownership or declare a prefix locally available.

This implementation gates loads after a positive LOOKUP. LOOKUP itself
submits L2 prefetch, so backend reads can still duplicate. Measure storage
reads independently of GPU load counts. This is not a storage-byte or latency
optimization claim.

The identity hashes resolved model name, parallel world size, cache salt,
adapter namespace and exactly the external covered token prefix. Trailing
tokens do not affect it. The connector admits only text token inputs without
LoRA; multimodal, embedded and adapter requests use the normal path. Prefix
caching is required. The feature is disabled by default.

The scheduler loop owns the registry without cross-thread access. A follower
waits at most 30 seconds before becoming a new leader. Leader completion or
cancellation releases the entry. Follower cancellation removes its accounting;
reset and shutdown invalidate all outstanding deferrals. Local eviction before
admission causes an ordinary external retry. A fresh local hit, rather than a
load-submission notification, is the follower's admission criterion.

`get_singleflight_stats()` returns detached cumulative leader, join, saved-load,
retry and failure counts, cumulative join wait seconds, and current inflight
and follower counts. Expiry without observed publication counts as failure;
these diagnostic counters do not replace worker transfer completion or native
backend byte accounting.
