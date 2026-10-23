# KMesh multiprocess ownership

## Scope of this increment

This design proposes CPU ownership rules for an opt-in KMesh multi-process
data plane. Status: design only; the registry and request adapters are proposed.
Ordinary MP store, lookup, retained reads and completion handling are unchanged.
The public KMesh target admission remains closed for MP execution.

The immediate objective is to preserve the existing serial position-recovery
semantics while moving cache ownership out of the inference process. A server
URL, an instance number, or a successful protocol handshake cannot establish
that two requests refer to compatible KV objects and a live worker session.

## Independent identities

The server descriptor identifies a boot, logical store generation and live
handler inventory. Capability names must come from installed request handlers,
not an advertised feature flag. Changing that inventory disarms registrations.

The semantic identity identifies immutable weights, resolved model and tokenizer
configuration, KV dtype, effective RoPE settings, cache layout/partitions and
chunk/key hashing rules. Every digest producer must include effective defaults
and version its canonical representation. Floating-point configuration values
must use canonical decimal strings. Layout includes attention window/type,
head/layer geometry, strides, group order, scale representation and parallel
partitions. Runtime placement and benchmark execution settings belong in the
separate execution-profile digest. The initial contract admits text-only BF16
attention with no adapters; integrations must explicitly reject multimodal,
LoRA, recurrent and unsupported quantized contexts before constructing evidence.

A domain opens atomically by semantic fingerprint. Different semantics get
different namespaces even if older objects survive deregistration. Object
namespaces also include server boot, store epoch and domain epoch. A restart
requires fresh registration and a fresh namespace; this increment deliberately
does not recover persistent objects across boots. An index/geometry reset
advances the domain epoch even if underlying bytes remain. Ordinary object
eviction is a miss and must not invalidate a domain.

A registration binds one engine invocation and exact rank barrier. Each rank
must independently supply matching semantic and execution evidence after its
actual tensor context is registered. The server issues an opaque worker fence
bound to the actual session and instance. A scheduler's lease copy contains
expectations only; it cannot authorize data-plane access.

## Transactions, revocation and physical drain

No work is accepted before every rank is bound. Request generations strictly
increase per worker/request and are consumed on acceptance. An aborted
generation five can never restore generation two or reuse five. One accepted
ticket owns the entire lookup/read/write transaction. Concurrent replacement of
the same request refuses while accepted work remains.

Each bound rank has its own heartbeat deadline. An incomplete registration also
has a fixed startup deadline. A missed deadline irreversibly revokes the lease;
a late heartbeat cannot revive it. Reconnect requires a new registration and
rank barrier. Changing the store, domain, capabilities or worker binding also
disarms before another operation is accepted.

Revocation prevents new work and publication; it does not cancel a CUDA event,
release a storage lock or make a pending future safe to destroy. Accepted tickets
remain until the actual completion path reports physical completion. An
undrained failure retains the ticket and revokes the registration. Repeated END
or revoke cannot erase this debt. The same engine run or worker instance cannot
be replaced while its accepted work remains. A later physical drain can release
debt but cannot rearm the old registration.

`publish()` serializes a synchronous publication callback with revocation.
Adapters must publish already-completed writes under the supplied namespace;
an unlocked check followed by a later write is insufficient. The callback must
not wait for I/O or re-enter the registry. A callback failure retains the ticket
and invalidates the domain generation, so uncertain partially published bytes
cannot be reached by a subsequent registration for the same semantics.
Reporting success to `finish()` means physical completion only, not permission
to publish data. Reads need corresponding fenced acceptance and completion
ownership in the adapter. Exporter death with outstanding IPC work requires a
failed, quarantined run until the real completion/teardown rule is established.

## Integration gates still required

The registry is intentionally not a promise that MP serving is working. Before
admission opens, the adapter and server must establish all of the following:

1. Derive semantic and execution digests from actual negotiated contexts and
   resolved model artifacts. Validate the first supported model/profile against
   independent worker evidence. A client-provided arbitrary digest is not proof.
2. Carry typed operations through the current method-oriented RequestClient
   API. Derive session identity from the transport. Refuse transports that cannot
   supply a stable actual-session identity. Do not revive retired integer RPCs.
3. Enforce the namespace/fence on every read, write, lookup and index operation.
   Generic legacy handlers must not bypass the reserved KMesh namespace. Existing
   objects written outside the lease must not become valid by relabeling a key.
4. Couple ticket lifetime to real accepted storage/GPU work, including callbacks,
   read-lock retention, failed completions and shutdown. Accepted work must drain
   before its context or buffers can be recycled. Bound memory and retire revoked
   registrations only after all physical obligations are discharged.
5. Exercise separate server and worker processes, reconnect/restart and drift,
   then one serial shifted segment with numerical/output/fallback qualification.
   Only then add collective recovery and compare storage transports.

The contract assumes cooperative processes under one user. Tokens prevent
accidental ownership confusion; they are not an adversarial authentication
mechanism. Cross-node operation and rolling upgrades are outside this increment.
Upgrade by draining, disarming, replacing processes and explicitly rearming.

## CPU validation

The proposed `tests/v1/test_kmesh_domains.py` must test public APIs without native
extensions. It must cover atomic opens, unsupported schemas/adapters, exact
rank/session binding,
the generation rollback regression, startup/heartbeat expiry, failed drain debt,
stale writer refusal, namespace drift across index/store/reboot boundaries,
unrelated registrations and publication serialized against revocation. These
tests will establish the ownership rules only; they will not attest an RPC
session, model, device, storage backend, or kernel transport.

Once implemented, run the CPU suite in an environment with pytest installed:

```sh
uv run --no-project --with pytest python -m pytest -q -ra \
  -p no:cacheprovider -c /dev/null --noconftest \
  tests/v1/test_kmesh_domains.py
```
