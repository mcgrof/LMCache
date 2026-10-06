GPU staging ordering
====================

The non-layerwise vLLM V2 and V3 connectors provide a synchronous staging
handoff. Their ``from_gpu`` and ``batched_from_gpu`` calls return only after
the selected KV bytes have reached the CPU or CUDA staging objects.

Before calling, join all KV and slot-mapping producers into the current CUDA
stream on the connector's device. For example:

.. code-block:: python

   current = torch.cuda.current_stream(device)
   current.wait_stream(kv_producer_stream)
   connector.batched_from_gpu(objects, starts, ends, slot_mapping=slots)
   # Objects are now ready for the storage backend to read.
   # Keep them alive and unchanged until storage completion.

Calls on one connector must be serialized. Keep the source slots and all
buffers stable until the gather returns. The connector waits once per
nonempty batch on its transfer stream, not on the whole device. Storage may
remain asynchronous after this handoff. The original KV slots and the
storage staging objects have different reuse points.

For restoration, finish storage reads and the platform's required external-DMA
visibility steps before calling ``batched_to_gpu``. That method waits for the
scatter before returning. Its staging sources can then be recycled.

Both batch directions drain queued transfers if a later enqueue fails. A CUDA
synchronization failure does not prove that buffers are no longer in use;
do not interpret that error as permission to recycle them.

This contract does not enqueue storage into a CUDA stream. It does not provide
cross-stream isolation, power-loss durability, or support for an otherwise
unsupported DMA route. Layerwise and MUSA connectors retain their separate
contracts.
