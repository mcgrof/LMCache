# SPDX-License-Identifier: Apache-2.0

"""B42 regression tests for the LMCache-driven object-group KV transfer.

Background (B42)
----------------
PR #3908 collapsed the per-object-group KV transfer into a single
GIL-released native call.  The planner
(:func:`lmcache.v1.multiprocess.modules.lmcache_driven_transfer._run_object_group_transfer_plan`)
and the staging primitives
(:func:`lmcache.v1.gpu_connector.gpu_ops.build_staging_copies`,
:func:`~lmcache.v1.gpu_connector.gpu_ops.lmcache_memcpy_async_h2d` /
``lmcache_memcpy_async_d2h``) resolve each ``MemoryObj`` to a single
contiguous span ``[data_ptr, data_ptr + get_size())`` and copy the whole
object into one flat object-group staging buffer.

The split-tier / asymmetric path reserves a ``KV_COMPONENT_GROUPS``
``MemoryObj``: a *two-group*
:class:`~lmcache.v1.memory_management.TensorMemoryObj` with K and V as
separate typed sub-views (``get_tensor(0)`` = K,
``get_tensor(1)`` = V).  B42 asks whether that whole-object span is still
correct when the object carries 2N component groups rather than a single
packed ``[2, ...]`` tensor.

Source analysis shows it is safe *by construction*: the two component
groups are contiguous sub-views of one flat ``raw_data`` allocation, so
``[data_ptr, +get_size())`` spans ``[K bytes | V bytes]`` exactly, and
``apply_kv_component_split`` preserves both the total byte count and the
K-then-V ordering of the packed layout.  These tests exercise that on a
real CUDA device.

Path selection / ``NO_GPU_EXT``
-------------------------------
When the native ``c_ops`` extension is absent (``NO_GPU_EXT=1`` build),
``lmcache.c_ops`` is transparently wired to the pure-Python fallback, so
``_HAS_NATIVE_OBJECT_GROUP_TRANSFER`` is ``False``.

* ``test_two_group_layout_is_packed_reinterpretation`` and
  ``test_two_group_object_staging_span_roundtrip``
  drive the B42 whole-object span through ``lmcache_memcpy_async_h2d`` /
  ``lmcache_memcpy_async_d2h``.  For a non-lazy ``MemoryObj`` these use a
  plain ``torch`` copy (no ``c_ops``), so they run on a ``NO_GPU_EXT`` pod
  and are the primary B42 coverage there.
* ``test_native_object_group_transfer_roundtrip``
  drives the full ``transfer_kv_per_object_group`` store/retrieve.  Its
  fallback stages through GPU temp buffers whose device pointers are then
  handed to ``multi_layer_block_kv_transfer``, but the fallback of that
  kernel reconstructs object pointers as *host* tensors
  (``_normalize_lmcache_objects`` -> ``_tensor_from_ptr(..., "cpu")`` in
  ``python_ops_fallback``), which is incompatible with GPU staging
  pointers.  It is therefore gated
  on ``_HAS_NATIVE_OBJECT_GROUP_TRANSFER`` and skips cleanly on a
  ``NO_GPU_EXT`` pod instead of crashing.
"""

# Standard
from collections.abc import Sequence

# Third Party
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc  # noqa: E402
from lmcache.v1.distributed.storage_layout import (  # noqa: E402
    apply_kv_component_split,
)
from lmcache.v1.gpu_connector.gpu_ops import (  # noqa: E402
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    TensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo  # noqa: E402
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (  # noqa: E402
    _HAS_NATIVE_OBJECT_GROUP_TRANSFER,
    downsample_and_stage_block_ids,
    get_layout_desc,
    transfer_kv_per_object_group,
)
from lmcache.v1.platform.cuda.cache_context import GPUCacheContext  # noqa: E402
import lmcache.c_ops as lmc_ops  # noqa: E402

_DEVICE = torch.device("cuda")

# Small packed KV layout used throughout: [kv=2, layers, slots, hidden].
_LAYERS = 2
_SLOTS = 8
_HIDDEN = 8
_DTYPE = torch.bfloat16
_HOST_POOL_BYTES = 1 << 20


# ---------------------------------------------------------------------------
# Helpers -- object construction
# ---------------------------------------------------------------------------


def _packed_layout() -> MemoryLayoutDesc:
    """Return the single-group packed ``[2, L, S, D]`` transfer layout."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([2, _LAYERS, _SLOTS, _HIDDEN])],
        dtypes=[_DTYPE],
    )


def _new_allocator() -> TensorMemoryAllocator:
    """Create a host-backed ``TensorMemoryAllocator`` for test MemoryObjs.

    The allocator's objects report ``parent()`` as a plain
    ``TensorMemoryAllocator`` (not a ``LazyMemoryAllocator``), so the
    staging primitives take their non-lazy ``torch``-copy branch and need
    no native ``c_ops``.
    """
    pool = torch.empty(_HOST_POOL_BYTES, dtype=torch.uint8, device="cpu")
    return TensorMemoryAllocator(pool)


def _alloc_object(
    allocator: TensorMemoryAllocator, layout: MemoryLayoutDesc
) -> TensorMemoryObj:
    """Allocate one ``TensorMemoryObj`` for *layout* (1- or 2-group)."""
    obj = allocator.allocate(layout.shapes, layout.dtypes, MemoryFormat.KV_2LTD)
    if obj is None:
        raise AssertionError("allocator returned no object (pool too small)")
    return obj


# ---------------------------------------------------------------------------
# Helpers -- real GPUCacheContext (native end-to-end test only)
# ---------------------------------------------------------------------------


class _GroupSpec:
    """One homogeneous block of KV layers for the synthetic KV cache.

    Mirrors the helper in ``tests/v1/platform/test_gpu_cache_context.py``.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int = 2,
        head_size: int = 4,
        block_size: int = 4,
        dtype: torch.dtype = _DTYPE,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.dtype = dtype


class _FakeIPCWrapper:
    """Test stand-in for ``CudaIPCWrapper`` (same-process IPC can't reopen
    its own handle); ``GPUCacheContext`` only needs ``to_tensor()``."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        """Return the wrapped local CUDA tensor (test-only)."""
        return self._tensor


def _make_kv_tensors(
    specs: Sequence[_GroupSpec], num_blocks: int
) -> list[torch.Tensor]:
    """Build non-MLA per-layer KV tensors shaped ``[2, NB, BS, NH, HS]``."""
    tensors: list[torch.Tensor] = []
    for spec in specs:
        for _ in range(spec.num_layers):
            tensors.append(
                torch.empty(
                    2,
                    num_blocks,
                    spec.block_size,
                    spec.num_heads,
                    spec.head_size,
                    dtype=spec.dtype,
                    device=_DEVICE,
                )
            )
    return tensors


def _make_context(
    specs: Sequence[_GroupSpec],
    chunk_size: int,
    num_blocks: int,
    engine_group_infos: Sequence[EngineGroupInfo] = (),
) -> GPUCacheContext:
    """Build a real ``GPUCacheContext`` via its public constructor."""
    tensors = _make_kv_tensors(specs, num_blocks=num_blocks)
    kv_caches = [_FakeIPCWrapper(t) for t in tensors]
    return GPUCacheContext(
        kv_caches,  # type: ignore[arg-type]
        lmcache_tokens_per_chunk=chunk_size,
        engine_group_infos=engine_group_infos,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTwoGroupKvComponentTransfer:
    """B42: whole-object span correctness for 2-group KV_COMPONENT_GROUPS objects."""

    def test_two_group_layout_is_packed_reinterpretation(self) -> None:
        """A 2-group split object is a byte-exact reinterpretation of the packed one.

        Fills a packed ``[2, L, S, D]`` object, copies its raw bytes into a
        2-group ``KV_COMPONENT_GROUPS`` object, and asserts the split
        object's ``get_tensor(0)`` (K) / ``get_tensor(1)`` (V) equal the
        packed object's ``tensor[0]`` / ``tensor[1]`` and that both report
        the same total size.  This is the invariant that makes the
        whole-object staging span safe -- no GPU transfer involved.
        """
        packed_layout = _packed_layout()
        split_layout = apply_kv_component_split(packed_layout)
        assert len(split_layout.shapes) == 2 * len(packed_layout.shapes)

        allocator = _new_allocator()
        packed_obj = _alloc_object(allocator, packed_layout)
        split_obj = _alloc_object(allocator, split_layout)

        assert packed_obj.get_size() == split_obj.get_size()

        # Seed the packed object with a distinct K and V pattern.
        packed_view = packed_obj.tensor
        assert packed_view is not None
        k_ref = torch.randn(_LAYERS, _SLOTS, _HIDDEN, dtype=_DTYPE)
        v_ref = torch.randn(_LAYERS, _SLOTS, _HIDDEN, dtype=_DTYPE)
        packed_view[0].copy_(k_ref)
        packed_view[1].copy_(v_ref)

        # Reinterpret the same bytes through the 2-group object.
        split_raw = split_obj.raw_tensor
        packed_raw = packed_obj.raw_tensor
        assert split_raw is not None and packed_raw is not None
        split_raw.copy_(packed_raw)

        k_view = split_obj.get_tensor(0)
        v_view = split_obj.get_tensor(1)
        assert k_view is not None and v_view is not None
        assert k_view.shape == torch.Size([_LAYERS, _SLOTS, _HIDDEN])
        assert v_view.shape == torch.Size([_LAYERS, _SLOTS, _HIDDEN])
        assert torch.equal(k_view, packed_view[0])
        assert torch.equal(v_view, packed_view[1])

    def test_two_group_object_staging_span_roundtrip(self) -> None:
        """Stage a 2-group object to/from a GPU object-group buffer, bit-exact.

        Drives the B42 whole-object span primitive
        (``lmcache_memcpy_async_h2d`` / ``lmcache_memcpy_async_d2h``) on a
        real CUDA device:

        1. Seed K into ``get_tensor(0)`` and V into ``get_tensor(1)`` of a
           2-group object; mirror the bytes into a packed object.
        2. H2D-stage both into GPU object-group buffers and assert the
           staged bytes are identical (2-group == packed) and that K
           occupies the first half of the span and V the second half.
        3. D2H-stage back into a fresh 2-group object and assert K and V
           round-trip bit-exact.

        Runs on a ``NO_GPU_EXT`` pod: the object is non-lazy, so the
        staging primitive uses a plain ``torch`` copy (no ``c_ops``).
        """
        packed_layout = _packed_layout()
        split_layout = apply_kv_component_split(packed_layout)

        allocator = _new_allocator()
        split_obj = _alloc_object(allocator, split_layout)
        packed_obj = _alloc_object(allocator, packed_layout)
        total_bytes = split_obj.get_size()

        # Seed distinct K / V content through the component views.
        k_ref = torch.randn(_LAYERS, _SLOTS, _HIDDEN, dtype=_DTYPE)
        v_ref = torch.randn(_LAYERS, _SLOTS, _HIDDEN, dtype=_DTYPE)
        k_view = split_obj.get_tensor(0)
        v_view = split_obj.get_tensor(1)
        assert k_view is not None and v_view is not None
        k_view.copy_(k_ref)
        v_view.copy_(v_ref)
        split_raw = split_obj.raw_tensor
        packed_raw = packed_obj.raw_tensor
        assert split_raw is not None and packed_raw is not None
        packed_raw.copy_(split_raw)

        # (2) H2D stage both objects into GPU object-group buffers.
        gpu_from_split = torch.empty(total_bytes, dtype=torch.uint8, device=_DEVICE)
        gpu_from_packed = torch.empty(total_bytes, dtype=torch.uint8, device=_DEVICE)
        lmcache_memcpy_async_h2d(split_obj, gpu_from_split)
        lmcache_memcpy_async_h2d(packed_obj, gpu_from_packed)
        torch.cuda.synchronize()

        # 2-group and packed produce byte-identical staged bytes (B42 invariant).
        assert torch.equal(gpu_from_split.cpu(), gpu_from_packed.cpu())

        # K occupies the first half of the span, V the second half.
        staged = gpu_from_split.cpu().view(_DTYPE)
        half = _LAYERS * _SLOTS * _HIDDEN
        assert torch.equal(staged[:half].view(_LAYERS, _SLOTS, _HIDDEN), k_ref)
        assert torch.equal(staged[half:].view(_LAYERS, _SLOTS, _HIDDEN), v_ref)

        # (3) D2H round-trip into a fresh 2-group object.
        split_obj_rt = _alloc_object(allocator, split_layout)
        lmcache_memcpy_async_d2h(gpu_from_split, split_obj_rt)
        torch.cuda.synchronize()
        k_rt = split_obj_rt.get_tensor(0)
        v_rt = split_obj_rt.get_tensor(1)
        assert k_rt is not None and v_rt is not None
        assert torch.equal(k_rt, k_ref)
        assert torch.equal(v_rt, v_ref)

    @pytest.mark.skipif(
        not _HAS_NATIVE_OBJECT_GROUP_TRANSFER,
        reason=(
            "native c_ops object-group transfer not compiled; the fallback "
            "stages through GPU temp buffers whose device pointers are "
            "incompatible with the CPU-only fallback of "
            "multi_layer_block_kv_transfer (see module docstring)"
        ),
    )
    def test_native_object_group_transfer_roundtrip(self) -> None:
        """End-to-end native store (D2H) then retrieve (H2D) with a 2-group object.

        Drives the real
        :func:`~lmcache.v1.multiprocess.modules.lmcache_driven_transfer.transfer_kv_per_object_group`
        (native single-call path) against a live ``GPUCacheContext`` and a
        ``KV_COMPONENT_GROUPS`` object built from the context's own layout:

        * D2H-gather the seeded KV cache into both a 2-group and a packed
          object; assert the staged bytes match and the K / V component
          views equal the packed object's K / V halves.
        * Zero the KV cache, H2D-scatter the 2-group object back, and assert
          the KV cache is restored bit-exact (all blocks are gathered, so
          the comparison is format-agnostic).
        """
        chunk_size = _SLOTS  # tokens per lmcache chunk == one object
        spec = _GroupSpec(num_layers=_LAYERS, num_heads=2, head_size=4, block_size=4)
        blocks_per_chunk = chunk_size // spec.block_size
        ctx = _make_context([spec], chunk_size=chunk_size, num_blocks=blocks_per_chunk)
        object_group_id = 0

        # Seed the KV cache the transfer actually reads (via group pointers).
        for kv in ctx.kv_tensors:
            kv.normal_()
        snapshot = [kv.clone() for kv in ctx.kv_tensors]

        # Build objects from the context's real (packed) transfer layout.
        packed_layout = get_layout_desc(ctx, chunk_size, object_group_id)
        split_layout = apply_kv_component_split(packed_layout)
        allocator = _new_allocator()
        split_obj = _alloc_object(allocator, split_layout)
        packed_obj = _alloc_object(allocator, packed_layout)

        # One object == one chunk covering all blocks.
        block_ids = [list(range(blocks_per_chunk))]
        block_ids_gpu = downsample_and_stage_block_ids(ctx, block_ids)

        # D2H (store) into both objects from the same seeded cache.
        for obj in (split_obj, packed_obj):
            transfer_kv_per_object_group(
                ctx,
                block_ids_gpu,
                [obj],
                object_group_id=object_group_id,
                batch_size=1,
                skip_first_n_tokens=0,
                direction=lmc_ops.TransferDirection.D2H,
            )
        torch.cuda.synchronize()

        split_raw = split_obj.raw_tensor
        packed_raw = packed_obj.raw_tensor
        packed_view = packed_obj.tensor
        assert split_raw is not None and packed_raw is not None
        assert packed_view is not None
        # 2-group gather == packed gather (B42 invariant on a real transfer).
        assert torch.equal(split_raw, packed_raw)
        k_view = split_obj.get_tensor(0)
        v_view = split_obj.get_tensor(1)
        assert k_view is not None and v_view is not None
        assert torch.equal(k_view, packed_view[0])
        assert torch.equal(v_view, packed_view[1])

        # H2D (retrieve) the 2-group object back into a zeroed cache.
        for kv in ctx.kv_tensors:
            kv.zero_()
        transfer_kv_per_object_group(
            ctx,
            block_ids_gpu,
            [split_obj],
            object_group_id=object_group_id,
            batch_size=1,
            skip_first_n_tokens=0,
            direction=lmc_ops.TransferDirection.H2D,
        )
        torch.cuda.synchronize()

        for kv, snap in zip(ctx.kv_tensors, snapshot, strict=True):
            assert torch.equal(kv, snap)
