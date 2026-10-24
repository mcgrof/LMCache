# SPDX-License-Identifier: Apache-2.0
"""Gather exact accepted source positions into canonical host or device chunks."""

# Standard
from dataclasses import dataclass, field

# Third Party
import torch

# First Party
from lmcache import device_ops, torch_dev
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.kmesh_bindings import BoundOperation
from lmcache.v1.multiprocess.kmesh_source_memory import require_registered_span_tensor
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
        partials: Packed and canonical views of each partial reservation.
        scratch: Preallocated repacking tensors, retained through device drain.

    The importing server must establish physical device completion before
    canonicalization. This object supplies neither completion nor admission.
    """

    index_buffers: list[torch.Tensor]
    partials: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
    scratch: tuple[torch.Tensor, ...] = ()
    _canonicalized: bool = field(default=False, init=False)

    def canonicalize(self) -> None:
        """Repack completed bytes once, with deterministic zero padding.

        Raises:
            RuntimeError: This gather has already consumed its packed bytes.

        The caller must have drained the actual importing device context.
        Repacking before that boundary would race native writes. Device copies
        use the caller's current registered stream and require a second drain.
        """
        if self._canonicalized:
            raise RuntimeError("gather bytes have already been canonicalized")
        self._canonicalized = True
        if not self.scratch:
            if any(packed.device.type != "cpu" for packed, _ in self.partials):
                raise RuntimeError("device repacking requires retained scratch")
            self.scratch = tuple(torch.empty_like(p) for p, _ in self.partials)
        for (packed, canonical), completed in zip(
            self.partials, self.scratch, strict=True
        ):
            completed.copy_(packed)
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
        objects: Complete host or device chunks with physical I/O leases held.
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

    Complete block-aligned lazy-host spans keep the object-group path. Interior
    starts and partial tails use the native slot-map kernel for separate or
    fused NHD/HND. Eager sources always use it. A tail gathers into packed strides,
    then needs canonicalization after actual drain. Device canonicalization
    needs another physical drain before admission. Returning establishes neither.
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
        if (
            all(isinstance(obj.parent(), LazyMemoryAllocator) for obj in objects)
            and source_start % block_size == 0
            and count % chunk == 0
        ):
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
        if fmt.name not in (
            "NL_X_TWO_NB_BS_NH_HS",
            "NL_X_TWO_NB_NH_BS_HS",
            "NL_X_NB_BS_NH_CS",
            "NL_X_NB_NH_BS_CS",
        ):
            raise ValueError("unaligned gather requires separate or fused NHD/HND")
        tensors: list[torch.Tensor] = []
        expected = (shape.kv_size, shape.nl, chunk, shape.nh * shape.hs)
        for obj in objects:
            tensor = obj.get_tensor(0)
            if (
                tensor is None
                or tuple(tensor.shape) != expected
                or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
            ):
                raise ValueError("slot gather requires complete contiguous chunks")
            if isinstance(obj.parent(), LazyMemoryAllocator):
                # One host pointer cannot span separate pin registrations.
                offset, size = obj.meta.address, obj.get_size()
                pin = LazyMemoryAllocator.PIN_CHUNK_SIZE
                if (
                    tensor.device.type != "cpu"
                    or offset < 0
                    or size < 1
                    or offset // pin != (offset + size - 1) // pin
                ):
                    raise ValueError("slot gather object crosses a host registration")
            else:
                require_registered_span_tensor(obj, cache.device, tensor.nbytes)
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
        targets: list[torch.Tensor] = []
        for index, tensor in enumerate(tensors):
            valid = min(chunk, count - index * chunk)
            target = tensor
            if valid < chunk:
                elements = shape.kv_size * shape.nl * valid * shape.nh * shape.hs
                target = tensor.view(-1)[:elements].view(
                    shape.kv_size, shape.nl, valid, shape.nh * shape.hs
                )
                partials.append((target, tensor))
            targets.append(target)
        gathered = RegisteredSpanGather(
            [slots],
            tuple(partials),
            tuple(torch.empty_like(packed) for packed, _ in partials),
        )
        for index, target in enumerate(targets):
            valid = target.shape[2]
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
        return gathered
