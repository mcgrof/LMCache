# SPDX-License-Identifier: Apache-2.0
"""Gather exact accepted source positions into complete pinned host chunks."""

# Third Party
import torch

# First Party
from lmcache import device_ops, torch_dev
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.kmesh_bindings import BoundOperation
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
    transfer_kv_per_object_group,
)
import lmcache.lmcache_native as lmcache_native


def gather_registered_span(
    operation: BoundOperation,
    objects: list[MemoryObj],
    source_start: int,
    block_ids: list[list[int]],
    *,
    transfer_key: str,
) -> list[torch.Tensor]:
    """Gather a complete span without rounding its original token coordinates.

    Args:
        operation: Exact accepted paged/rotary borrows retained by the caller.
        objects: Complete CPU chunk reservations with physical I/O leases held.
        source_start: First source position in the original sequence.
        block_ids: Its full private block table, for one full-attention group.
        transfer_key: Accepted transfer identity used by native phase tracing.

    Returns:
        Device index buffers that must remain alive until checked completion.

    Raises:
        ValueError: Unsupported source geometry, objects or coordinates.
        RuntimeError: Native preparation or submission failed. Retain the
            exception frames, all borrows and objects until device-context drain.

    Block-aligned sources keep the object-group path. An interior block start
    uses the existing slot-map kernel for two-plane NHD/HND storage. Returning
    from this helper never establishes physical completion or admission.
    """
    cache = operation.paged.entry.cache_context
    manager = cache.kv_layer_groups_manager
    if (
        cache.device.type != "cuda"
        or len(manager.kernel_groups) != 1
        or manager.num_object_groups != 1
        or not objects
        or len(block_ids) != 1
    ):
        raise ValueError("unsupported registered span gather resources")
    group = manager.kernel_groups[0]
    shape = group.shape_desc
    chunk = cache.lmcache_tokens_per_chunk
    block_size = group.tokens_per_block
    count = len(objects) * chunk
    if (
        group.dtype != torch.bfloat16
        or group.recurrent_state
        or group.sw_size_tokens != -1
        or chunk % block_size
        or type(source_start) is not int
        or source_start < 0
        or source_start + count > len(block_ids[0]) * block_size
        or any(
            type(b) is not int or not 0 <= b < cache.num_blocks for b in block_ids[0]
        )
        or len(set(block_ids[0])) != len(block_ids[0])
    ):
        raise ValueError("invalid registered source gather coordinates")
    with torch_dev.device(cache.device), torch_dev.stream(cache.stream):
        if source_start % block_size == 0:
            selected = [
                block_ids[0][
                    source_start // block_size : (source_start + count) // block_size
                ]
            ]
            staged = downsample_and_stage_block_ids(cache, selected)
            transfer_kv_per_object_group(
                cache,
                staged,
                objects,
                0,
                1,
                0,
                lmcache_native.TransferDirection.D2H,
                transfer_key=transfer_key,
                block_ids_host=selected,
            )
            return staged
        fmt = cache.get_engine_kv_format(0)
        if fmt.name not in ("NL_X_TWO_NB_BS_NH_HS", "NL_X_TWO_NB_NH_BS_HS"):
            raise ValueError("unaligned gather requires two-plane NHD or HND")
        tensors: list[torch.Tensor] = []
        expected = (2, shape.nl, chunk, shape.nh * shape.hs)
        for obj in objects:
            # The native slot kernel maps one registered host pointer and then
            # indexes it linearly; separate pin ranges need separate mappings.
            offset, size = obj.meta.address, obj.get_size()
            pin = LazyMemoryAllocator.PIN_CHUNK_SIZE
            if (
                not isinstance(obj.parent(), LazyMemoryAllocator)
                or offset < 0
                or size < 1
                or offset // pin != (offset + size - 1) // pin
            ):
                raise ValueError("slot gather object crosses a host registration")
            tensor = obj.get_tensor(0)
            if (
                tensor is None
                or tuple(tensor.shape) != expected
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
            ):
                raise ValueError("unaligned gather requires complete host chunks")
            tensors.append(tensor)
        positions = range(source_start, source_start + count)
        slots = torch.tensor(
            [
                block_ids[0][p // block_size] * block_size + p % block_size
                for p in positions
            ],
            dtype=torch.int64,
            device=cache.device,
        )
        pointers = cache.get_kernel_group_kv_pointers(0)
        for index, tensor in enumerate(tensors):
            device_ops.multi_layer_kv_transfer(
                tensor,
                pointers,
                slots[index * chunk : (index + 1) * chunk],
                cache.device,
                cache.num_blocks * block_size,
                lmcache_native.TransferDirection.D2H,
                fmt,
                block_size=block_size,
                head_size=operation.rotary.state.head_size_for_group(
                    group.engine_group_idx
                ),
                block_stride_elems=shape.block_stride_elems,
            )
        return [slots]
