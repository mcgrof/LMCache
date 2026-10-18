# Caller-driven family ancestor preloading

This prototype coordinates declared preloads with MP child admission. It
builds on external-load coalescing: one reactive load can serve siblings,
but its restore still occurs after the first child arrives. An application
that knows an ancestor will be needed can submit a warm request before
releasing children. The engine's automatic prefix cache supplies sharing.

`declare_family_prefetch()` records an exact chunk-aligned base-model text
ancestor under the singleflight identity. The caller triggers the warm
request and later releases the declaration. It does not launch I/O, install
KV or keep blocks resident. With a warm request ID, matching children with
insufficient local coverage defer before LOOKUP. This avoids a competing
prefetch while the warm request owns L1 staging buffers. The warm request
never defers on itself. Completion, cancellation or expiry removes the wait;
a fresh local hit permits admission without waiting for warm completion.

Credit a local child hit only when allocation admits that request. An
external lookup miss cannot erase actual local coverage. Supply the warm
request's internal engine ID as `prefetch_request_id` to exclude it even
when it finds a local hit. Its external load also receives no child credit,
so a preload with no later children is charged as unused on release.

Cache each candidate identity on its request tracker. Base-model text prefixes,
model/parallel configuration and the tracker's tenant salt are immutable for
the request lifetime. Repeated waiting polls therefore do not copy or hash
the prompt again. Normal tracker teardown frees the identities; different
requests and tenant namespaces have independent caches.

The opt-in `lmcache.mp.family_prefetch` defaults to false and requires prefix
caching and lazy LOOKUP (`lmcache.mp.eager_prefetch: false`). Other input
types keep their normal path. Declarations expire after 30 seconds; at most
4096 may be pending. Each declaration's child-ID sets are bounded by its
expected child count. Capacity refusal returns no identity and leaves
serving unchanged. Release, reset and shutdown clear deferrals. Expiry and
unused release increment the wasted counter.

Counters describe declared local hits, deferred children and logical KV
bytes. Measure native reads, warm work and child latency independently.
After a warm request completes, ordinary LOOKUP still starts backend
prefetch even for a locally cached prefix. Declaration counts alone do not
establish storage or serving benefit. A short fork gap needs a concurrent
engine replay; adding an estimated unfinished restore to a warm-child time
is not a measurement of that case.
