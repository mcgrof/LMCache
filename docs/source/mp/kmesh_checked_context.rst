Experimental checked raw context lifecycle
==========================================

KMesh multiprocess model recovery remains disabled. The low-level
``GPUCacheContext(..., managed_raw=True)`` constructor is a resource primitive
for a bounded server integration, rather than a serving configuration.
It accepts only raw CUDA IPC wrappers and requires an inactive GDS slab.
Ordinary context creation and serving keep their existing behavior.

The managed constructor retains its context before its first raw import.
A failed construction stays quarantined until the importing process ends.
A trusted server must stop admission and drain accepted consumers before
calling ``close_raw_checked()``. This operation drains the actual context
stream, permanently retires the raw allocation identities against new imports,
closes checked references, and verifies that no raw-wrapper aliases remain.
Only then does it return the released allocation identities. Successful
repeated calls return the same identities without another native close.

Keep the context and exporter allocations alive after any refusal. A failure
known to precede native close can be retried after its cause is removed.
An ambiguous native import or close outcome requires importing-process teardown;
stream completion cannot make its old pointer safe to submit again.

This check does not cover PyTorch storage, VMM or external driver imports.
The constructor and close also refuse active GDS because the existing staging
buffer deregistration interface logs backend errors. GPU-L1 storage therefore
requires a separately qualified checked lifecycle before admission. A service
release receipt, storage namespace fencing and a model consumer are additional
gates; this primitive does not grant any of them.

Bounded managed raw control service
-----------------------------------

The dedicated experimental entry point composes only managed KMesh controls:

.. code-block:: console

   python -m lmcache.v1.multiprocess.kmesh_raw_server \
       --host 127.0.0.1 --port 5555 --transport zmq --chunk-tokens 256

Use ``--transport grpc`` for the other same-host request transport. The service
uses a CPU DRAM L1 and accepts one raw CUDA context and one ordered rotary
preparation per process lifetime. It exposes no legacy integer registration,
generic store/retrieve, lookup, peer, HTTP or automatic reaping interface.
It does not enable multiprocess vLLM model recovery.

Read ``kmesh_get_server`` first. Call ``kmesh_register_raw_context`` with that
boot identity, a fresh 32-character lowercase hexadecimal nonce, a positive
instance ID, exact raw CUDA wrappers, the model name and ``NHD`` or ``HND``
layout. The service records the receiving client's actual transport identity,
its own context generation and incoming wrappers before construction. The
bounded context accepts contiguous BF16 full-attention tensors at world size
one. Model artifact semantics remain independent worker declarations.

Repeating the same nonce and view metadata returns saved registration status
without importing again. Changing an interior view, model or layout under that
nonce refuses. ``kmesh_raw_context_status`` resolves uncertainty about a reply
but does not acknowledge release. Foreign clients and stale server boots refuse.
A poisoned construction requires importing-process exit.

Use the returned context generation with ``kmesh_register_rotary``. The worker
must retain every rotary input and its producer event without further writes
until the service proves their completed use. Incoming raw owners enter the
copy ledger before import. The service waits for the producer, takes a completed
host snapshot, normalizes it and publishes private uploaded tables. Repeating a
completed identical preparation returns its saved runtime without another import.
A failed preparation cannot be replaced in this bounded profile: drain retained
copy debt with ``kmesh_retry_rotary_drain`` and retire the context.

``kmesh_retire_raw_context`` takes the expected boot, nonce and exact context.
It closes admission, revokes registration holds, refuses unfinished physical
work, removes the exact rotary generation and performs checked raw close.
Temporary rotary allocations are retired against new imports after their
checked reference cleanup. Only after all mappings and bookkeeping finish does
it cache a ``ManagedRawRelease`` containing the exact generations and allocation
identities. Repeating retirement returns this same receipt without native work.
A known pre-native refusal can be retried after cleanup; unknown native effects
still require importer exit.

Keep all worker allocations and producer events until validating that receipt,
or until observing importing-process exit. Missing status, a disconnect, a zero
copy count or a legacy reply is not release evidence. These transport identities
coordinate cooperative clients; they do not authenticate a hostile peer.
