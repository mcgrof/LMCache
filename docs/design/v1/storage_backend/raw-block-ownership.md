# Raw-block ownership and publication

This contract applies to the in-process raw-block backend. The GPU staging
route is qualified separately for the V2/V3 vLLM connectors. It does not
establish support for other connectors, exporters, or physical PCIe routes.

## Buffer ownership

A completion that reports success or failure and proves native quiescence can
release its owners. A timeout or failed synchronization alone cannot. Unknown
outcomes stop admission and retain buffers, exports, allocators, and streams
until process exit. Retention is deliberate: reclaiming those bytes could let
unfinished DMA overwrite a later request.

Host batched I/O retains Python buffer exports, not only the exporter objects.
An object reference prevents destruction but does not prevent a bytearray from
resizing. DMA-BUF pointer-only tensors retain their tensor owner instead.

A sparse fixed-buffer table becomes owned immediately after registration,
including when the next slot update fails. Successful unregister permits
fallback or retry. Failed unregister poisons the device and keeps its backing
owners; ordinary host registration must not hide that failure.

## Partial GPU chunks

Pool slots retain a physical byte extent. Rebinding a slot updates its logical
shapes, group offsets, and size. V2/V3 gather the logical payload and zero the
physical tail on the store stream, then wait for completion before returning
the slot to storage. For a 1,280-byte payload in a 4,096-byte slot, all 2,816
tail bytes must be zero on media. Reuse must preserve this rule.

Other callers of device-buffer raw-block I/O must establish the same readiness
and initialized-padding contract themselves. The CPU cannot safely clear an
opaque GPU address. Export alignment follows host pages; physical transfer
alignment follows storage geometry. These are different constraints.

## Durable publication

Strict publication waits for its KV writes and writes a checkpoint payload.
It then flushes the target, writes the commit header, and flushes again before
returning a receipt. Either flush failure prevents READY. Block devices and
regular files use `fsync`; a character-device NVMe passthrough target needs a
separately implemented and tested NVMe Flush operation.

This guarantees only the persistence contract the underlying storage stack
provides. It does not persist live reader leases. Restart remains a coordinated
whole-group operation; unplanned writer/reader crash recovery is unsupported.

The tracker retires request state under its lock, then resolves public futures
outside that lock. Callbacks can query the tracker without deadlock. A live
lease prevents request-ID reuse even after the bounded result history evicts
that request. Cancellation before receipt delivery releases only an unclaimed
publication; it cannot release bytes a reader claimed.

## Qualification

CPU and native fault-seam tests check ownership, failure propagation, flush
ordering, and bounds. The native V2/V3 fixture additionally reads media with an
independent oracle and checks padding across generations. Actual DMA-BUF
registration, raw-device P/D, and stock-vLLM generation require separate runs
on the exact source tree and newly built native extensions. A file-backed run
does not establish raw-device qualification.
