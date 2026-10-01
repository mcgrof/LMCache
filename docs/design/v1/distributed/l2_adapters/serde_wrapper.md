# `SerdeL2AdapterWrapper` — Transparent Serde via Adapter Composition

## Scope

Describes how serialization / deserialization is integrated into the
L2 path. The serde package itself
([`docs/design/v1/distributed/serde/README.md`](../serde/README.md))
defines the generic `Serializer` / `Deserializer` / `SerdeProcessor`
interfaces and built-in transforms. **This** doc is about the adapter
that stitches serde into the distributed storage pipeline.

## Design Summary

`SerdeL2AdapterWrapper` implements `L2AdapterInterface` by composing
an inner L2 adapter with a `SerdeProcessor` and an `L1Manager`. The
caller sees the wrapper's public API only; the wrapper's internal
thread is the sole consumer of the inner adapter's and serde's event
fds.

Step numbers below show **call ordering**. Solid `─►` arrows are
direct synchronous calls; dotted `╌►` arrows are eventfd wakeups
consumed by the wrapper's internal thread.

### Store path

```
   caller                                                          
     │                                                             
     │ (1) submit_store_task(keys, objs)                           
     ▼                                                             
   ┌─────────────────┐  (2) reserve_write(tmp)   ┌──────────────┐  
   │  wrapper (API)  │ ─────────────────────────►│  L1Manager   │  
   │                 │ ◄────── tmp_objs ─────────│              │  
   └────────┬────────┘                           └──────────────┘  
            │ (3) submit_serialize(objs, tmp_objs)                 
            ▼                                                      
   ┌─────────────────┐                                             
   │ SerdeProcessor  │   transforms objs → tmp_objs                
   └────────┬────────┘                                             
            ╎ (4) serialize_efd                                    
            ▼                                                      
   ┌─────────────────┐  (5) inner.submit_store_   ┌──────────────┐ 
   │ wrapper thread  │      task(keys, tmp_objs)  │   inner L2   │ 
   │   (_loop)       │ ─────────────────────────► │   adapter    │ 
   └────────┬────────┘                            └──────┬───────┘ 
            ▲                                            │         
            ╎ (6) inner.store_efd ◄──────────────────────┘         
            │                                                      
   ┌────────┴────────┐                                             
   │ wrapper thread  │  (7) finish_read(tmp_objs)  → auto-delete   
   │                 │      signal wrapper.store_efd               
   └────────┬────────┘                                             
            │                                                      
            │ (8) pop_completed_store_tasks() →                   
            │     {wrapped_id: L2StoreResult(success=True,         
            │                                bytes_transferred=N)}
            ▼                                                      
          caller                                                   
```

### Load path

```
   caller                                                          
     │                                                             
     │ (1) submit_load_task(keys, dst_objs)                        
     ▼                                                             
   ┌─────────────────┐  (2) reserve_write(tmp)   ┌──────────────┐  
   │  wrapper (API)  │ ─────────────────────────►│  L1Manager   │  
   │                 │ ◄────── tmp_objs ─────────│              │  
   └────────┬────────┘                           └──────────────┘  
            │ (3) inner.submit_load_task(keys, tmp_objs)           
            ▼                                                      
   ┌─────────────────┐                                             
   │  inner L2       │   loads serialized bytes into tmp_objs      
   │  adapter        │                                             
   └────────┬────────┘                                             
            ╎ (4) inner.load_efd                                   
            ▼                                                      
   ┌─────────────────┐  (5) submit_deserialize    ┌──────────────┐ 
   │ wrapper thread  │      (tmp_objs, dst_objs)  │   Serde      │ 
   │   (_loop)       │ ─────────────────────────► │  Processor   │ 
   └────────┬────────┘                            └──────┬───────┘ 
            ▲                                            │         
            ╎ (6) deserialize_efd ◄──────────────────────┘         
            │                                                      
   ┌────────┴────────┐                                             
   │ wrapper thread  │  (7) finish_write + delete(tmp_objs)        
   │                 │      signal wrapper.load_efd                
   └────────┬────────┘                                             
            │                                                      
            │ (8) query_load_result() → per-key bitmap             
            ▼                                                      
          caller                                                   
```

## Store Path

```mermaid
sequenceDiagram
    participant Controller as StoreController
    participant W as SerdeL2AdapterWrapper
    participant Thread as wrapper internal thread
    participant Serde as SerdeProcessor
    participant L1 as L1Manager
    participant Inner as inner L2 adapter

    Controller->>W: submit_store_task(keys, objs)
    W->>L1: reserve_write(temp_keys, byte layout)
    L1-->>W: temp_objs (write-locked)
    W->>Serde: submit_serialize(objs, temp_objs, keys) → serde_id
    W-->>Controller: wrapped_id

    Note over Serde,Thread: serde thread pool transforms<br/>objs → temp_objs, signals fd

    Serde-->>Thread: serialize_efd fires
    Thread->>Serde: query_serialize_result(serde_id)
    Serde-->>Thread: success
    Thread->>L1: finish_write_and_reserve_read(temp_keys)
    Thread->>Inner: submit_store_task(keys, temp_objs) → inner_id

    Note over Inner: inner stores serialized bytes,<br/>signals its store_efd

    Inner-->>Thread: store_efd fires
    Thread->>Inner: pop_completed_store_tasks()
    Inner-->>Thread: {inner_id: L2StoreResult(ok=True, bytes=N)}
    Thread->>L1: finish_read(temp_keys)   %% auto-deletes temps
    Thread->>W: _finalize_store(wrapped_id, inner_result)
    W-->>Controller: store_efd fires

    Controller->>W: pop_completed_store_tasks()
    W-->>Controller: {wrapped_id: L2StoreResult(ok=True, bytes=N)}
```

## Load Path

```mermaid
sequenceDiagram
    participant Controller as PrefetchController
    participant W as SerdeL2AdapterWrapper
    participant Thread as wrapper internal thread
    participant Inner as inner L2 adapter
    participant Serde as SerdeProcessor
    participant L1 as L1Manager

    Controller->>W: submit_load_task(keys, dst_objs)
    W->>L1: reserve_write(temp_keys, byte layout)
    L1-->>W: temp_objs (write-locked)
    W->>Inner: submit_load_task(keys, temp_objs) → inner_id
    W-->>Controller: wrapped_id

    Note over Inner: inner loads serialized bytes<br/>into temp_objs, signals load_efd

    Inner-->>Thread: load_efd fires
    Thread->>Inner: query_load_result(inner_id)
    Inner-->>Thread: bitmap (per-key success)

    alt all-zeros bitmap
        Thread->>L1: finish_write + delete temp_keys
        Thread->>W: _finalize_load(wrapped_id, bitmap)
    else some keys loaded
        Thread->>Serde: submit_deserialize(temp_objs, dst_objs) → serde_id
        Note over Serde: serde thread transforms<br/>temp_objs → dst_objs
        Serde-->>Thread: deserialize_efd fires
        Thread->>Serde: query_deserialize_result(serde_id)
        alt success
            Thread->>L1: finish_write + delete temp_keys
            Thread->>W: _finalize_load(wrapped_id, bitmap)
        else failure
            Thread->>L1: finish_write + delete temp_keys
            Thread->>W: _finalize_load(wrapped_id, Bitmap(0, len))
            Note right of W: zero bitmap ⇒ controller<br/>treats keys as load failures
        end
    end

    W-->>Controller: load_efd fires
    Controller->>W: query_load_result(wrapped_id)
    W-->>Controller: final bitmap
```

## Lookup / Unlock / Eviction

These paths do not run a transform. Together placement delegates logical keys
directly. Split-tier placement maps logical keys to V-child keys and masks a
reported hit unless the manifest is `COMPLETE` and the matching K child is
resident in L1. The lookup event fd remains the inner adapter's fd, avoiding
an extra thread hop.

| API | Behavior |
|---|---|
| `get_lookup_and_lock_event_fd` | Returns the inner adapter's fd (pass-through) |
| `submit_lookup_and_lock_task` / `query_lookup_and_lock_result` | Direct in together mode; split-tier maps to V children and gates results on manifest + K residency |
| `submit_unlock` | Direct in together mode; split-tier maps logical keys to V children |
| `delete` | Delegates the supplied physical keys; higher-level split-tier cleanup supplies fenced child keys |
| `get_usage` / `supports_global_eviction` | Delegated directly |
| `register_listener` | Registers on the inner adapter (listeners track real storage state) |

The wrapper does **not** maintain its own byte accounting — it
reports whatever the inner adapter reports. This keeps
`L2EvictionController` and per-cache_salt quota logic unchanged:
they see the inner adapter's byte totals through the wrapper's
`get_usage()`.

## Temp Buffer Lifecycle

Temp byte buffers are the only new L1 state the wrapper introduces.
They exist entirely within the wrapper's knowledge — the caller never
sees the temp keys.

**Store path:**

1. `reserve_write(temp_keys, is_temporary=True, layout=ser_layout, mode="new")`
   — temps are write-locked and marked temporary so
   `finish_read` will auto-delete them later.
2. Serialize runs, filling temps.
3. On success: `finish_write_and_reserve_read(temp_keys)` — temps
   become read-locked so `inner.submit_store_task` can safely read
   them.
4. Inner store completes → `finish_read(temp_keys)` — since
   `is_temporary=True`, finish_read also deletes them.
5. On serialize or pre-admission failure,
   `finish_write_and_delete(temp_keys)` atomically discards write-locked
   temps. If inner submission or storage fails after temp admission,
   `finish_read(temp_keys)` releases the read lock and auto-deletes them.

**Load path:**

1. `reserve_write(temp_keys, is_temporary=True, layout=ser_layout, mode="new")`
   — same as store, temps write-locked.
2. `inner.submit_load_task(keys, temp_objs)` — inner loads serialized
   bytes into temps.
3. Inner completes → `submit_deserialize(temp_objs, dst_objs)` — note
   temps stay write-locked (the wrapper owns them; only the wrapper
   reads them during deserialize).
4. Deserialize completes → `finish_write_and_delete(temp_keys)` atomically
   discards temps regardless of deserialize success.

## Split-Tier K/V Lifecycle

An explicit absent K slot (`input_slot_mapping() == (None, 1)`) selects
`KV_SPLIT_TIER`. The wrapper accepts exactly one K/V pair per logical object.
It derives distinct K- and V-child keys under an engine-wide key scheme:
scale-aware objects use `COMPUTED_LEGACY`; byte-through `RAW_UNIT` objects use
a separate tagged namespace so their wire formats cannot collide.

Store ordering is:

1. Allocate and copy the K child into L1.
2. Register a fresh manifest generation before admitting K, so the store
   controller never routes the K child back to L2.
3. Copy V into a scratch pool keyed by exact `(shape, dtype)`.
4. Return the logical key for early read-lock release; serialization reads
   only the copied V scratch.
5. Store the encoded V under its child key and mark the manifest `COMPLETE`.

Load first masks any key not `COMPLETE`, reserves the K child, loads the V
child, and composes both into the caller's destination. Missing K, missing V,
truncation, codec mismatch, or deserialize failure is a cache miss.

The manifest owns a monotonically increasing generation per attempted store.
Failed stores, eviction, clear, and adapter removal acquire a
`DELETE_IN_FLIGHT` fence before deleting physical children. Because child
names do not include the generation, this fence prevents a replacement from
publishing under the same names until the old blocking delete finishes. The
filesystem adapter treats its 30-second threshold as an operational warning
and continues waiting for the unlink to become terminal; returning with a
pending unlink would let that old operation remove a later generation's file.
Cleanup operations are generation-guarded and cannot delete a newer entry.

The logical staging object is deleted after its final reader exits. Deferred
deletion is stored on that exact L1 resident, so a later replacement under the
same logical key does not inherit the marker. K-child and V-scratch pools are
reported as separate, non-fungible L1 pressure domains; eviction responds to
the fullest pool instead of diluting a full K slab with unrelated free bytes.

This mode is intentionally not persistent across restart. A V child without
the process-local manifest and K child is a miss. Configuration validation
also restricts it to CPU pinned memory, one filesystem adapter, no L2
eviction/`IsolatedLRU`, and one object group.

## Failure Policy: All-or-Nothing per Submit

If **any** key's temp allocation fails or `submit_serialize` /
`submit_deserialize` / `inner.submit_*` raises, the **whole wrapped
task fails**:

- Store task: `pop_completed_store_tasks()` returns
  `{wrapped_id: L2StoreResult(success=False, bytes_transferred=0)}`.
- Load task: `query_load_result(wrapped_id)` returns an all-zeros
  `Bitmap(len(keys))`.

This preserves the **coarse-grained success semantic** of
`L2AdapterInterface` — store is task-level (the `L2StoreResult.is_successful()`
flag), load is per-key via bitmap. The alternative ("drop the failed keys,
succeed with the rest") would silently violate the caller's assumption
that every key it passed either succeeded (task-level) or failed
(task-level). Keeping the policy coarse is what lets the controllers
stay untouched.

Partial load failures inside the inner adapter (some keys in the
inner bitmap succeed, some fail) are faithfully preserved: only keys
with `bitmap.test(i) == True` are deserialized, and the wrapper
reports the same bitmap to the controller (zeroed if deserialize
itself fails).

## Homogeneity Invariant

`_alloc_temp_buffers` sizes temps from `objects[0].get_shapes() /
get_dtypes()` under the assumption that **all MemoryObjs in one
submit share a single layout**. The store controller already
shape-groups keys before calling `submit_store_task`; the prefetch
controller uses a single `layout_desc` per request for all write
reservations. The wrapper enforces the invariant with an explicit
`raise ValueError` to catch future regressions.

## Threading Model

One background thread per wrapper instance, started in `__init__`.

- Poll loop over four fds: `inner.store_efd`, `inner.load_efd`,
  `serde.serialize_efd`, `serde.deserialize_efd`.
- Single `threading.Lock` protecting: the task-id counter, the four
  reverse-lookup dicts, the two completion dicts, and the phase flip
  on store tasks.
- `close()` sets a stop flag and joins the thread; the poll timeout
  (`500 ms`) bounds shutdown latency.
- `_finalize_store` / `_finalize_load` write 1 to the wrapper's own
  eventfd to wake the upstream controller's poll loop.

The wrapper does **not** share any state with the inner adapter or
the `SerdeProcessor` beyond the public interface — both are
independent thread-safe components, and the wrapper's thread
interacts with them only through their documented APIs.

## Known Limitations / Follow-ups

- **Temp buffer allocation per submit.** Every store / load call
  issues an `L1Manager.reserve_write` for fresh temp keys. A free-list
  keyed on `(shape, dtype)` would skip the allocator on the hot path.
  Not done in the initial version — correctness first.
- **Bookkeeping-dict duplication.** Six of the existing L2 adapters
  maintain their own `_next_task_id + lock + completion_dict + eventfd`
  quartet; `AsyncSerdeProcessor` is a seventh; this wrapper is an
  eighth. An `EventfdTaskQueue` base helper could dedupe this across
  the tree — a separate cleanup PR.
- **No per-step latency histograms.** `report_status` delegates to the inner
  and adds `{"serde_wrapped": True}`. Split-tier terminal outcomes and
  manifest-state counts are exported, but serialize/copy/store duration is
  not yet measured separately.
