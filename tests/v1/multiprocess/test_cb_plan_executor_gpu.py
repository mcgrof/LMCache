# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the CB retrieve plan executor's copy/compute overlap.

The executor stages each step on a dedicated copy stream while the previous
step's rope/scatter kernels run on the compute stream, with per-parity CUDA
events ordering tmp-slot reuse (step w's staging may only start once step
w-2's kernels finished reading the same slot half). These tests drive many
steps of slot-half reuse where every wave writes DIFFERENT data into the same
slots — any event-ordering bug surfaces as a bit-exact mismatch against the
sequential tensor-op reference, not as a flake.

Cover fused-packed and separate-plane K/V with head-major and token-major
paged layouts, including current content-width format tags. Grouped cases
also exercise public registration and retrieve handlers with real pinned
allocation, plan construction and kernels, while storage and event-handle
transport remain local test stand-ins.
"""

# Standard
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
import lmcache.lmcache_native as lmcache_native

if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip(
        "CUDA is not available, skipping the test",
        allow_module_level=True,
    )

# First Party
import lmcache.cuda_ops as cuda_ops  # noqa: E402

if not hasattr(cuda_ops, "execute_cb_retrieve_plan_flat"):
    pytest.skip(
        "cuda_ops build lacks execute_cb_retrieve_plan_flat",
        allow_module_level=True,
    )

_NL, _SPC, _NH, _HS = 4, 8, 2, 16
_NB, _BS = 512, 4
_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class _FmtCase:
    """Per-format geometry: chunk plane count, widths, and paged shape."""

    fmt: "lmcache_native.EngineKVFormat"
    kv_size: int  # chunk leading planes (1 = K/V fused, 2 = split)
    hidden: int  # per-plane scalars per token
    head_stride: int  # rope stride between heads in the K plane
    paged_shape: tuple  # per-layer paged tensor shape
    hnd: bool = True


_CASES = {
    # Fused-packed HND: [NB, NH, BS, 2*HS], K is the first HS of each head.
    "packed": _FmtCase(
        fmt=lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
        kv_size=1,
        hidden=_NH * 2 * _HS,
        head_stride=2 * _HS,
        paged_shape=(_NB, _NH, _BS, 2 * _HS),
    ),
    # Un-fused flash-attention HND: [2, NB, NH, BS, HS], separate K/V planes.
    "split": _FmtCase(
        fmt=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        kv_size=2,
        hidden=_NH * _HS,
        head_stride=_HS,
        paged_shape=(2, _NB, _NH, _BS, _HS),
    ),
    "cs_hnd": _FmtCase(
        fmt=lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS,
        kv_size=1,
        hidden=_NH * 2 * _HS,
        head_stride=2 * _HS,
        paged_shape=(_NB, _NH, _BS, 2 * _HS),
    ),
    "cs_nhd": _FmtCase(
        fmt=lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS,
        kv_size=1,
        hidden=_NH * 2 * _HS,
        head_stride=2 * _HS,
        paged_shape=(_NB, _BS, _NH, 2 * _HS),
        hnd=False,
    ),
    "split_nhd": _FmtCase(
        fmt=lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        kv_size=2,
        hidden=_NH * _HS,
        head_stride=_HS,
        paged_shape=(2, _NB, _BS, _NH, _HS),
        hnd=False,
    ),
}


def _reference_scatter(
    case, host_chunks, paged_ptrs, slot_mapping, old_sts, cur_sts, cos_sin
):
    """Sequential per-chunk tensor-op reference (rope then scatter)."""
    dev = slot_mapping.device
    ramp = torch.arange(_SPC, device=dev, dtype=torch.long).repeat(_NL)
    for i, host in enumerate(host_chunks):
        buf = host.to(dev)
        k_view = buf[0].reshape(_NL * _SPC, _NH, case.head_stride)
        cuda_ops.rotary_embedding_k_fused_strided(
            old_sts[i] + ramp,
            cur_sts[i] + ramp,
            k_view,
            _HS,
            case.head_stride,
            cos_sin,
            True,
        )
        cuda_ops.multi_layer_kv_transfer(
            buf,
            paged_ptrs,
            slot_mapping[i * _SPC : (i + 1) * _SPC],
            slot_mapping.device,
            _NB * _BS,
            lmcache_native.TransferDirection.H2D,
            case.fmt,
            block_size=_BS,
            head_size=_HS,
        )
    torch_dev.synchronize()


def _run_plan(
    case,
    n_chunks,
    max_batch,
    host_chunks,
    paged_ptrs,
    slot_mapping,
    old_sts,
    cur_sts,
    cos_sin,
    slots,
):
    """Drive the production flat-table entry point with the planner's
    double-buffer wave layout."""
    chunk_bytes = case.kv_size * _NL * _SPC * case.hidden * _DTYPE.itemsize
    spec = cuda_ops.CBGroupSpec(
        paged_kv_ptrs=paged_ptrs.data_ptr(),
        temp_buffer_ptrs=[s.data_ptr() for s in slots],
        num_layers=_NL,
        slot_tokens=_SPC,
        hidden_elems=case.hidden,
        element_size=_DTYPE.itemsize,
        engine_kv_format=case.fmt,
        page_buffer_size=_NB * _BS,
        block_size=_BS,
        head_size=_HS,
        slot_mapping_base=slot_mapping.data_ptr(),
        slot_mapping_capacity=slot_mapping.numel(),
        cos_sin_cache=cos_sin.data_ptr(),
        rot_dim=_HS,
        rope_num_kv_heads=_NH,
        rope_head_stride=case.head_stride,
        key_scalar_type=15,  # at::ScalarType::BFloat16
        is_neox=True,
    )
    wave = max_batch // 2
    staging, ropes, scatters, step_offsets = [], [], [], []
    for w0 in range(0, n_chunks, wave):
        step_idx = w0 // wave
        base = (step_idx % 2) * wave
        for j in range(min(wave, n_chunks - w0)):
            ci, slot = w0 + j, base + j
            staging.append(
                (slots[slot].data_ptr(), host_chunks[ci].data_ptr(), chunk_bytes, 0)
            )
            ropes.append((0, slot, old_sts[ci], cur_sts[ci]))
            scatters.append((0, slot, ci * _SPC, _SPC))
        step_offsets.append((len(staging), len(ropes), len(scatters)))
    cuda_ops.execute_cb_retrieve_plan_flat(
        slot_mapping.device,
        1 << 26,
        [spec],
        np.asarray(staging, dtype=np.int64),
        np.asarray(ropes, dtype=np.int64),
        np.asarray(scatters, dtype=np.int64),
        np.asarray(step_offsets, dtype=np.int64),
    )
    torch_dev.synchronize()


@pytest.mark.parametrize("fmt_key", sorted(_CASES))
@pytest.mark.parametrize("n_chunks,max_batch", [(12, 4), (96, 16), (44, 16)])
def test_overlap_slot_reuse_is_bit_exact(n_chunks, max_batch, fmt_key):
    """Many steps of slot-half reuse, each wave carrying different data into
    the same slots: an ordering violation (staging overwriting a slot still
    being read, or kernels reading a half-staged slot) breaks bit-exactness.
    Covers full packs, partial tail packs, and deep reuse, on both the
    fused-packed and un-fused split-K/V paged layouts."""
    case = _CASES[fmt_key]
    dev = torch.device(torch_device_type)
    torch.manual_seed(n_chunks)

    paged_ref = [
        torch.zeros(*case.paged_shape, dtype=_DTYPE, device=dev) for _ in range(_NL)
    ]
    paged_new = [torch.zeros_like(t) for t in paged_ref]
    ptrs_ref = torch.tensor(
        [t.data_ptr() for t in paged_ref], dtype=torch.long, device=dev
    )
    ptrs_new = torch.tensor(
        [t.data_ptr() for t in paged_new], dtype=torch.long, device=dev
    )
    host_chunks = [
        torch.randn(case.kv_size, _NL, _SPC, case.hidden, dtype=_DTYPE).pin_memory()
        for _ in range(n_chunks)
    ]
    slots = [
        torch.zeros(case.kv_size, _NL, _SPC, case.hidden, dtype=_DTYPE, device=dev)
        for _ in range(max_batch)
    ]
    cos_sin = torch.randn(8192, _HS, dtype=_DTYPE, device=dev)
    pos = torch.arange(0, n_chunks * _SPC, device=dev, dtype=torch.long)
    block_ids = torch.arange(_NB, device=dev, dtype=torch.long).flip(0)
    slot_mapping = block_ids[pos // _BS] * _BS + pos % _BS
    old_sts = [i * _SPC + 512 for i in range(n_chunks)]
    cur_sts = [i * _SPC for i in range(n_chunks)]

    _reference_scatter(
        case, host_chunks, ptrs_ref, slot_mapping, old_sts, cur_sts, cos_sin
    )
    _run_plan(
        case,
        n_chunks,
        max_batch,
        host_chunks,
        ptrs_new,
        slot_mapping,
        old_sts,
        cur_sts,
        cos_sin,
        slots,
    )

    for layer in range(_NL):
        assert torch.equal(paged_ref[layer], paged_new[layer]), (
            f"layer {layer} mismatch "
            f"(fmt={fmt_key}, n_chunks={n_chunks}, max_batch={max_batch})"
        )


def _logical_kv(
    case: _FmtCase, host: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read logical K and V independently of the native scatter implementation."""
    if case.kv_size == 1:
        rows = host[0].reshape(_NL, _SPC, _NH, 2 * _HS)
        return rows[..., :_HS], rows[..., _HS:]
    rows = host.reshape(2, _NL, _SPC, _NH, _HS)
    return rows[0], rows[1]


def _read_paged_rows(
    case: _FmtCase, paged: torch.Tensor, mapping: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read each physical token using the declared paged layout."""
    blocks, offsets = mapping // _BS, mapping % _BS
    if case.kv_size == 1:
        rows = paged[blocks, :, offsets, :] if case.hnd else paged[blocks, offsets]
        return rows[..., :_HS], rows[..., _HS:]
    if case.hnd:
        return paged[0, blocks, :, offsets, :], paged[1, blocks, :, offsets, :]
    return paged[0, blocks, offsets], paged[1, blocks, offsets]


def _math_recovery(
    key: torch.Tensor, cache: torch.Tensor, old: int, current: int
) -> torch.Tensor:
    """Compute an independent float32 NeoX rotation on bounded BF16 inputs.

    The native BF16 arithmetic rounds intermediate results; this reference
    retains float32 intermediates. The numerical assertion allows 0.03125
    absolute error, while V and bypassed K require identical storage words.
    """
    if old == current:
        return key.float()
    n_tokens = key.shape[1]
    old_rows = cache[old : old + n_tokens].float()[None, :, None]
    new_rows = cache[current : current + n_tokens].float()[None, :, None]
    old_cos, old_sin = old_rows.chunk(2, dim=-1)
    new_cos, new_sin = new_rows.chunk(2, dim=-1)
    x, y = key.float().chunk(2, dim=-1)
    unrotated_x = x * old_cos + y * old_sin
    unrotated_y = y * old_cos - x * old_sin
    return torch.cat(
        (
            unrotated_x * new_cos - unrotated_y * new_sin,
            unrotated_y * new_cos + unrotated_x * new_sin,
        ),
        dim=-1,
    )


def _direct_group_plan(
    case: _FmtCase,
    chunks: list[list[torch.Tensor]],
    paged: list[list[torch.Tensor]],
    mappings: list[torch.Tensor],
    caches: list[torch.Tensor],
    old_starts: list[int],
    lengths: list[int],
    max_batch: int,
) -> None:
    """Execute public flat-plan bindings with two independently owned groups."""
    device = mappings[0].device
    slots = [
        [torch.empty_like(chunk, device=device) for _ in range(max_batch)]
        for chunk in (group[0] for group in chunks)
    ]
    pointers = [
        torch.tensor([layer.data_ptr() for layer in group], device=device)
        for group in paged
    ]
    specs = [
        cuda_ops.CBGroupSpec(
            paged_kv_ptrs=pointers[g].data_ptr(),
            temp_buffer_ptrs=[slot.data_ptr() for slot in slots[g]],
            num_layers=_NL,
            slot_tokens=_SPC,
            hidden_elems=case.hidden,
            element_size=_DTYPE.itemsize,
            engine_kv_format=case.fmt,
            page_buffer_size=_NB * _BS,
            block_size=_BS,
            head_size=_HS,
            slot_mapping_base=mappings[g].data_ptr(),
            slot_mapping_capacity=mappings[g].numel(),
            cos_sin_cache=caches[g].data_ptr(),
            rot_dim=_HS,
            rope_num_kv_heads=_NH,
            rope_head_stride=case.head_stride,
            key_scalar_type=15,
            cache_scalar_type=6,
            is_neox=True,
        )
        for g in range(len(chunks))
    ]
    staging: list[tuple[int, int, int, int]] = []
    ropes: list[tuple[int, int, int, int]] = []
    scatters: list[tuple[int, int, int, int]] = []
    ends: list[tuple[int, int, int]] = []
    wave = max_batch // 2
    for first in range(0, len(lengths), wave):
        parity = len(ends) % 2
        for offset in range(min(wave, len(lengths) - first)):
            index = first + offset
            slot = parity * wave + offset
            for g in range(len(chunks)):
                host = chunks[g][index]
                staging.append(
                    (slots[g][slot].data_ptr(), host.data_ptr(), host.nbytes, 0)
                )
                if old_starts[index] != index * _SPC:
                    ropes.append((g, slot, old_starts[index], index * _SPC))
                scatters.append((g, slot, index * _SPC, lengths[index]))
        ends.append((len(staging), len(ropes), len(scatters)))
    cuda_ops.execute_cb_retrieve_plan_flat(
        device,
        1 << 26,
        specs,
        *(
            np.asarray(table, dtype=np.int64).reshape(-1, width)
            for table, width in [(staging, 4), (ropes, 4), (scatters, 4), (ends, 3)]
        ),
    )
    torch_dev.synchronize()


def _registered_group_plan(
    case: _FmtCase,
    chunks: list[list[torch.Tensor]],
    paged: list[list[torch.Tensor]],
    block_tables: list[torch.Tensor],
    caches: list[torch.Tensor],
    old_starts: list[int],
    lengths: list[int],
    max_batch: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise public registration/retrieve handlers and their real planner.

    Storage and event-handle transport are local stand-ins. Pinned allocation,
    registration geometry, plan construction, CUDA events and kernels are real.
    This checks native correctness, not cross-process transport or ownership.
    """
    # Third Party
    import cupy

    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.kv_layer_groups import ObjectGroupInfo
    from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
    from lmcache.v1.multiprocess.custom_types import (
        CBMatchResult,
        DeviceIPCWrapper,
        IPCCacheServerKey,
    )
    from lmcache.v1.multiprocess.modules.blend.module import BlendModule

    device = paged[0][0].device
    context = MagicMock()
    context.device = device
    context.stream = torch_dev.current_stream()
    context.cupy_stream = cupy.cuda.ExternalStream(
        context.stream.cuda_stream, device.index
    )
    context.max_batch_size = max_batch
    manager = context.kv_layer_groups_manager
    manager.num_kernel_groups = len(chunks)
    manager.object_groups = [
        ObjectGroupInfo(kernel_group_indices=[g]) for g in range(len(chunks))
    ]
    manager.kernel_groups = [
        SimpleNamespace(
            tokens_per_block=_BS,
            slots_per_block=_BS,
            engine_group_idx=g,
            engine_kv_format=case.fmt,
            shape_desc=SimpleNamespace(nb=_NB, nh=_NH, hs=case.head_stride),
        )
        for g in range(len(chunks))
    ]
    manager.get_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1, -1], group_kinds=("attention", "aux")
    )
    pointers = [
        torch.tensor([layer.data_ptr() for layer in group], device=device)
        for group in paged
    ]
    temp = [
        [torch.empty_like(group[0], device=device) for _ in range(max_batch)]
        for group in chunks
    ]
    context.get_temp_kernel_group_buffer.side_effect = lambda slot, g: temp[g][slot]
    context.get_temp_object_group_buffer.side_effect = lambda slot, g: temp[g][
        slot
    ].view(torch.uint8)
    context.get_kernel_group_kv_pointers.side_effect = lambda g: pointers[g]
    context.get_engine_kv_format.side_effect = lambda g: case.fmt

    ready = torch_dev.Event()
    ready.record()
    backend = MagicMock()
    backend.create_event.side_effect = lambda device: torch_dev.Event()
    backend.record_event.side_effect = lambda event, stream: event.record(stream)
    backend.import_event.return_value = ready
    backend.wait_event.side_effect = lambda event, stream: stream.wait_event(event)
    backend.export_event.return_value = b"local-test-completed"
    transfer = MagicMock()
    transfer.get_and_touch_context_entry.return_value = SimpleNamespace(
        cache_context=context, event_backend=backend
    )
    ctx = MagicMock()
    ctx.chunk_size = _SPC
    ctx.session_manager.destroy_listeners = []
    ctx.session_manager.get.return_value = None
    released: list[list[object]] = []
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.modules.blend.retrieve.submit_callback_to_stream",
        lambda stream, name, keys: released.append(keys),
    )
    allocator = LazyMemoryAllocator(1 << 26, 1 << 26)
    objects = []
    module = BlendModule(ctx, transfer)
    try:
        for index in range(len(lengths)):
            for group in chunks:
                host = group[index]
                obj = allocator.allocate(host.shape, host.dtype)
                assert obj is not None
                raw_tensor = obj.raw_tensor
                assert raw_tensor is not None
                raw_tensor.view(host.dtype).reshape(host.shape).copy_(host)
                objects.append(obj)
        ctx.storage_manager.read_prefetched_results.side_effect = (
            lambda *a, **kw: nullcontext(objects)
        )
        wrappers: list[DeviceIPCWrapper] = []
        for cache in caches:
            wrapper = MagicMock(spec=DeviceIPCWrapper)
            wrapper.to_tensor.return_value = cache
            wrappers.append(wrapper)
        module.cb_register_rope(
            1, wrappers, _HS, True, [0, 1], [[0, _HS], [0, _HS]], [_HS, _HS]
        )
        matches = [
            CBMatchResult(
                hash=index.to_bytes(8, "little"),
                old_st=old,
                old_ed=old + length,
                cur_st=index * _SPC,
                cur_ed=index * _SPC + length,
            )
            for index, (old, length) in enumerate(zip(old_starts, lengths, strict=True))
        ]
        n_tokens = sum(lengths)
        key = IPCCacheServerKey(
            model_name="native-geometry",
            world_size=1,
            worker_id=0,
            request_id="registered-layout",
            token_ids=tuple(range(n_tokens)),
            start=0,
            end=n_tokens,
        )
        torch_dev.synchronize()
        handle, success = module.cb_retrieve_pre_computed(
            key,
            matches,
            [table.tolist() for table in block_tables],
            1,
            b"local-test-ready",
        )
        torch_dev.synchronize()
        assert success and handle == b"local-test-completed"
        assert released and len(released[-1]) == len(objects)
        backend.record_event.assert_called()
    finally:
        torch_dev.synchronize()
        module.close()
        for obj in objects:
            obj.ref_count_down()
        allocator.close()


@pytest.mark.parametrize("fmt_key", ["split", "split_nhd", "cs_hnd", "cs_nhd"])
@pytest.mark.parametrize("n_chunks,max_batch", [(11, 4), (69, 64)])
@pytest.mark.parametrize("entry", ["direct", "registered"])
def test_grouped_layout_rotation_preserves_raw_values(
    fmt_key: str,
    n_chunks: int,
    max_batch: int,
    entry: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check modern layouts, two object groups, skipped rotation and ragged tails.

    A 32-chunk wave crosses the native 16-pointer flush boundary. Complete
    paged tensors retain poison outside written rows. Independent CPU indexing
    and rotation check K values; V and unshifted K use raw BF16 storage words,
    including negative zero. No serving or transfer-speed claim follows.
    """
    case = _CASES[fmt_key]
    device = torch.device("cuda", 0)
    rng = torch.Generator().manual_seed(821 + n_chunks)
    chunks = [
        [
            torch.randn(
                case.kv_size, _NL, _SPC, case.hidden, generator=rng, dtype=_DTYPE
            )
            .mul_(0.125)
            .add_(g * 0.5)
            .pin_memory()
            for _ in range(n_chunks)
        ]
        for g in range(2)
    ]
    for group in chunks:
        for host in group:
            _key, value = _logical_kv(case, host)
            value[..., 0] = -0.0
    paged = [
        [
            torch.full(case.paged_shape, -17.0, dtype=_DTYPE, device=device)
            for _ in range(_NL)
        ]
        for _ in range(2)
    ]
    positions = torch.arange(n_chunks * _SPC)
    block_tables = [torch.arange(_NB).roll(g * 19).flip(0) for g in range(2)]
    mappings = [
        (table[positions // _BS] * _BS + positions % _BS).to(device)
        for table in block_tables
    ]
    caches = []
    for g in range(2):
        phases = torch.arange(2048, dtype=torch.float32)[:, None] * torch.exp(
            -torch.arange(_HS // 2) * (0.1 + g * 0.03)
        )
        caches.append(torch.cat((phases.cos(), phases.sin()), dim=1).to(device))
    old_starts = [i * _SPC if i % 3 == 0 else i * _SPC + 513 for i in range(n_chunks)]
    lengths = [_SPC] * n_chunks
    lengths[-1] = _SPC - 3
    if entry == "direct":
        _direct_group_plan(
            case, chunks, paged, mappings, caches, old_starts, lengths, max_batch
        )
    else:
        _registered_group_plan(
            case,
            chunks,
            paged,
            block_tables,
            caches,
            old_starts,
            lengths,
            max_batch,
            monkeypatch,
        )
    for g, group in enumerate(paged):
        cache = caches[g].cpu()
        mapping = mappings[g].cpu()
        for layer, actual_device in enumerate(group):
            actual = actual_device.cpu()
            changed = torch.zeros_like(actual, dtype=torch.bool)
            for index, length in enumerate(lengths):
                ids = mapping[index * _SPC : index * _SPC + length]
                blocks, offsets = ids // _BS, ids % _BS
                if case.kv_size == 1:
                    if case.hnd:
                        changed[blocks, :, offsets, :] = True
                    else:
                        changed[blocks, offsets] = True
                elif case.hnd:
                    changed[:, blocks, :, offsets, :] = True
                else:
                    changed[:, blocks, offsets] = True
                got_k, got_v = _read_paged_rows(case, actual, ids)
                key, value = _logical_kv(case, chunks[g][index])
                assert torch.equal(
                    got_v.view(torch.int16), value[layer, :length].view(torch.int16)
                )
                if old_starts[index] == index * _SPC:
                    assert torch.equal(
                        got_k.view(torch.int16), key[layer, :length].view(torch.int16)
                    )
                expected = _math_recovery(
                    key[:, :length], cache, old_starts[index], index * _SPC
                )
                torch.testing.assert_close(
                    got_k.float(), expected[layer], rtol=0, atol=0.03125
                )
            assert torch.equal(
                actual[~changed], torch.full_like(actual[~changed], -17.0)
            )
