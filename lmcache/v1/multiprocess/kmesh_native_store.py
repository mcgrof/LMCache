# SPDX-License-Identifier: Apache-2.0
"""Gather exact accepted source positions into canonical pinned host chunks."""

# Standard
from dataclasses import dataclass, field

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


@dataclass
class RegisteredSpanGather:
    """Retain native gather inputs and canonicalize a drained partial tail.

    Args:
        index_buffers: Device indexing tensors retained through actual drain.
        partials: Packed and canonical views of each partial host reservation.

    The importing server must establish physical device completion before
    canonicalization. This object supplies neither completion nor admission.
    """

    index_buffers: list[torch.Tensor]
    partials: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
    _canonicalized: bool = field(default=False, init=False)

    def canonicalize(self) -> None:
        """Repack completed host bytes once, with deterministic zero padding.

        Raises:
            RuntimeError: This gather has already consumed its packed bytes.

        The caller must have drained the actual importing device context.
        Moving bytes on the CPU before that boundary would race native writes.
        """
        if self._canonicalized:
            raise RuntimeError("gather bytes have already been canonicalized")
        self._canonicalized = True
        for packed, canonical in self.partials:
            completed = packed.clone()
            canonical.zero_()
            canonical[:, :, : packed.shape[2]].copy_(completed)


def gather_registered_span(
    operation: BoundOperation,
    objects: list[MemoryObj],
    source_start: int,
    block_ids: list[list[int]],
    *,
    token_count: int,
    transfer_key: str,
) -> RegisteredSpanGather:
    """Gather valid slots without rounding either source boundary.

    Args:
        operation: Exact accepted paged/rotary borrows retained by the caller.
        objects: Complete CPU chunk reservations with physical I/O leases held.
        source_start: First source position in the original sequence.
        block_ids: Its full private block table, for one full-attention group.
        token_count: Exact positive valid extent, including a partial final chunk.
        transfer_key: Accepted transfer identity used by native phase tracing.

    Returns:
        Retained native inputs and post-drain canonicalization of the final tail.

    Raises:
        ValueError: Unsupported source geometry, objects or coordinates.
        RuntimeError: Native preparation or submission failed. Retain the
            exception frames, all borrows and objects until device-context drain.

    Complete block-aligned spans keep the object-group path. Interior starts
    and partial tails use the native slot-map kernel for two-plane NHD/HND.
    A tail gathers into packed host strides, then needs CPU canonicalization
    after actual drain. Returning never establishes completion or admission.
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
    count = token_count
    if (
        group.dtype != torch.bfloat16
        or group.recurrent_state
        or group.sw_size_tokens != -1
        or chunk % block_size
        or type(count) is not int
        or count <= 0
        or len(objects) != (count + chunk - 1) // chunk
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
        if source_start % block_size == 0 and count % chunk == 0:
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
            return RegisteredSpanGather(staged)
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
        partials: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, tensor in enumerate(tensors):
            valid = min(chunk, count - index * chunk)
            target = tensor
            if valid < chunk:
                elements = 2 * shape.nl * valid * shape.nh * shape.hs
                target = tensor.view(-1)[:elements].view(
                    2, shape.nl, valid, shape.nh * shape.hs
                )
                partials.append((target, tensor))
            device_ops.multi_layer_kv_transfer(
                target,
                pointers,
                slots[index * chunk : index * chunk + valid],
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
        return RegisteredSpanGather([slots], tuple(partials))
