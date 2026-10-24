# Retain raw rotary imports before their first mapping attempt

Status: proposed. Checked raw-wrapper reference release passes its separate
three-case CUDA prerequisite. Raw rotary input preparation and integrated
server retirement remain unimplemented; multiprocess engine admission is closed.

Ordered registration accepts PyTorch CUDA IPC inputs today. Extending it to raw
imports needs ownership before calling ``to_tensor``. If importing a later table
fails, earlier wrappers may already hold driver mappings even though no complete
tensor list exists. A retained-copy ledger created only after import cannot own
that partial failure.

Create an unpublished rotary generation containing the raw mapping owners and
validated declared geometry. Give it and the exact paged-context borrow to the
copy manager before any import. Register the preparation in the manager's
physical-obligation ledger, retaining every wrapper, before its first
``to_tensor`` call. Partial imports stay owned by that entry.

After mapping, validate tensor device and the server's registered KV/rotary
metadata without reading values. Wait for the worker producer on the actual
registered stream before copying table contents into private host snapshots.
Keep the input tensors, mappings, producer event and both borrows until this
stream physically completes. Metadata validation must not perform an unordered
device value read.

After a successful drain, close the temporary raw references through
``close_checked`` before releasing their ownership entry. Other aliases of the
same CUDA allocation retain their own references. A temporary reference release
must not retire the whole allocation or ban later snapshots from it.

If stream completion or mapping close fails, retain the entry, buffers, mapping
owners and both borrows. Mark it eligible for a later server-observed drain and
checked-close retry. Do not publish a snapshot merely because its bytes finished
copying while import cleanup remains uncertain. Interruption preserves the same
obligations. Remove the ledger entry only after physical work and all owned raw
reference releases succeed.

An ambiguous native import/close outcome is not a retryable preflight refusal.
The raw wrapper quarantines it until importer teardown. A later copy-stream
drain must not turn that unresolved mapping outcome into a successful release.

CPU snapshots and existing registered-state copy operations keep their current
contracts. An experimental raw-input entry point must require raw wrapper types
and a trusted server metadata validator. It does not grant runtime discovery,
storage execution or positive exporter teardown by itself. Keep the typed raw
registration path refused until it consumes this ownership operation and passes
its own transport and lifecycle tests.

Qualify a genuinely pending producer with exact host bytes, partial import
failure, pre-driver close refusal and later actual successful retry. Include a
submitted-copy error whose drain fails, ensuring mapping close does not run
until later physical completion. Use separate exporter/importer processes and
keep the exporter alive until the importer exits. These controlled cases do not
qualify actual device faults, exporter death or model serving.
