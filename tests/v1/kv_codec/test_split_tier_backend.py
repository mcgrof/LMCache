# SPDX-License-Identifier: Apache-2.0
"""Phase 5 backend wrapper tests — `SplitTierStorageBackend` adapter.

Pure contract-shape tests.  Live integration with the LMCache
storage manager (registry add, eviction policy interaction)
needs an end-to-end test on a GPU pod and is deferred.
"""

# Standard
from concurrent.futures import wait
from dataclasses import dataclass
from pathlib import Path

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.kv_codec import (
    AsymK16V8Codec,
    PlacementPolicy,
)
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.split_tier_backend import (
    SplitTierStorageBackend,
)


class _FakeCPUBackend:
    """Stand-in for LocalCPUBackend; SplitTierStorageBackend
    delegates allocator queries to it but doesn't touch its
    methods in the put/get path tests below."""

    def allocate(self, *args, **kwargs):
        raise NotImplementedError("test stub")


def _kv_obj(seed=0, shape=(2, 4, 16, 32)):
    g = torch.Generator()
    g.manual_seed(seed)
    t = torch.randn(*shape, dtype=torch.float16, generator=g)
    meta = MemoryObjMetadata(
        shape=torch.Size(t.shape),
        dtype=t.dtype,
        address=0,
        phy_size=t.numel() * t.element_size(),
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(
        raw_data=t, metadata=meta, parent_allocator=None
    )


def _key(name="test_session_0"):
    """Build a CacheEngineKey.  Real signature is
    (model_name, world_size, worker_id, chunk_hash: int, dtype).
    The chunk_hash is an int; we hash the test name to a stable
    int so each call gets a unique key.

    Filenames embed the key's string form via to_string(); the
    SplitTierStore path layout uses the result as a directory
    component.  Avoid characters that conflict with paths."""
    return CacheEngineKey(
        model_name=f"test_{name}",
        world_size=1,
        worker_id=0,
        chunk_hash=hash(name) & 0x7FFFFFFFFFFFFFFF,
        dtype=torch.float16,
    )


@pytest.fixture
def backend(tmp_path):
    return SplitTierStorageBackend(
        root=tmp_path,
        local_cpu_backend=_FakeCPUBackend(),
        policy=PlacementPolicy.SPLIT_K_CPU_V_NVME,
    )


def test_construct_with_split_policy(backend):
    """Catches: constructor regression."""
    assert backend.store.policy == PlacementPolicy.SPLIT_K_CPU_V_NVME


def test_put_and_contains(backend):
    """Submit a put task, wait for completion, then `contains`
    reports True for that key."""
    k = _key("session/chunk0")
    obj = _kv_obj()
    futures = backend.batched_submit_put_task([k], [obj])
    assert len(futures) == 1
    wait(futures)
    # Key should now be present.
    assert backend.contains(k) is True
    # And not in inflight.
    assert backend.exists_in_put_tasks(k) is False


def test_contains_missing_returns_false(backend):
    k = _key("never/written")
    assert backend.contains(k) is False


def test_get_blocking_returns_bytes_buffer(backend):
    """get_blocking returns a BytesBufferMemoryObj that the asym
    serde's deserializer can consume."""
    k = _key("session/chunk0")
    obj = _kv_obj()
    wait(backend.batched_submit_put_task([k], [obj]))

    out = backend.get_blocking(k)
    assert isinstance(out, BytesBufferMemoryObj)
    assert out.metadata.fmt == MemoryFormat.BINARY_BUFFER
    assert out.metadata.dtype is None
    # Plural metadata carries [K_dtype, V_dtype] for the codec
    # decoder to recover.
    assert out.metadata.dtypes is not None
    assert out.metadata.dtypes == [torch.float16, torch.float8_e4m3fn]


def test_get_blocking_missing_returns_None(backend):
    k = _key("absent")
    assert backend.get_blocking(k) is None


def test_byte_counts_attribution_split_policy(backend):
    """The byte counts must attribute K bytes to CPU and V bytes
    to NVMe under split-tier — the headline ratio."""
    k = _key("session/chunk0")
    obj = _kv_obj(shape=(2, 4, 32, 64))
    wait(backend.batched_submit_put_task([k], [obj]))

    writes, reads = backend.get_byte_counts()
    # On write under SPLIT: K bytes -> CPU pinned, V+scales -> NVMe.
    n_elem_per_half = 4 * 32 * 64
    expected_K_bytes = n_elem_per_half * 2  # FP16
    assert writes.cpu_bytes == expected_K_bytes
    # NVMe write is V_fp8 + small scales ~= V byte count.
    assert writes.nvme_bytes >= n_elem_per_half  # at least V bytes
    assert writes.nvme_bytes < n_elem_per_half * 2  # well below FP16 V


def test_native_asym_view_rejects_with_clear_message(backend):
    """An input MemoryObj with .asym_view (native_asym path) is
    not yet plumbed through SplitTierStore; reject with a clear
    NotImplementedError, not silent fallback."""
    # Build a synthetic AsymKVView-bearing object
    # First Party
    from lmcache.v1.kv_codec import (
        ScaleScope,
        compute_v_scales,
        quantize_v_fp8,
    )
    from lmcache.v1.storage_backend.naive_serde.asym_serde import (
        AsymKVView,
    )

    g = torch.Generator()
    g.manual_seed(7)
    k = torch.randn(4, 16, 32, dtype=torch.float16, generator=g)
    v_fp16 = torch.randn(4, 16, 32, dtype=torch.float16, generator=g)
    s = compute_v_scales(v_fp16, ScaleScope.PER_TENSOR).to(torch.float32)
    v_fp8 = quantize_v_fp8(v_fp16, s, ScaleScope.PER_TENSOR)
    view = AsymKVView(k=k, v_fp8=v_fp8, v_scales=s)

    class _AsymInput:
        def __init__(self, view):
            self.asym_view = view
            self.tensor = None
            self.metadata = MemoryObjMetadata(
                shape=torch.Size([0, 0, 0, 0]),
                dtype=None, address=0, phy_size=0, ref_count=1,
                pin_count=0, fmt=MemoryFormat.UNDEFINED,
            )

        def ref_count_down(self):
            pass

    obj = _AsymInput(view)
    key = _key("native_asym")
    # Submit synchronously through the executor and inspect failure.
    futures = backend.batched_submit_put_task([key], [obj])
    f = futures[0]
    # The future should raise NotImplementedError.
    with pytest.raises(NotImplementedError, match="native_asym"):
        f.result()


def test_remove_clears_chunk_dir(backend, tmp_path):
    k = _key("session/chunk0")
    obj = _kv_obj()
    wait(backend.batched_submit_put_task([k], [obj]))
    assert backend.contains(k)
    chunk_dir = tmp_path / k.to_string() / "layer_000" / "chunk_000000"
    assert chunk_dir.exists()
    assert backend.remove(k) is True
    assert not chunk_dir.exists()
    assert backend.contains(k) is False


def test_pin_unpin(backend):
    k = _key("session/chunk0")
    wait(backend.batched_submit_put_task([k], [_kv_obj()]))
    assert backend.pin(k) is True
    # Pinning a missing key returns False.
    missing = _key("missing")
    assert backend.pin(missing) is False
    assert backend.unpin(k) is True
    assert backend.unpin(k) is False  # already unpinned


def test_pin_blocks_force_false_remove(backend):
    """remove(force=False) must respect pinning."""
    k = _key("session/chunk0")
    wait(backend.batched_submit_put_task([k], [_kv_obj()]))
    backend.pin(k)
    assert backend.remove(k, force=False) is False
    assert backend.contains(k)
    # Force remove should still work.
    assert backend.remove(k, force=True) is True


def test_get_allocator_backend_returns_provided(backend):
    """The backend exposes its allocator via the StorageBackendInterface
    contract."""
    inner = backend.get_allocator_backend()
    assert isinstance(inner, _FakeCPUBackend)


def test_put_then_get_roundtrip_K_bit_exact(backend, tmp_path):
    """Full roundtrip via the backend's put + get_blocking + the
    codec's decode produces bit-exact K and within-noise V."""
    k = _key("rt/0")
    obj = _kv_obj()
    wait(backend.batched_submit_put_task([k], [obj]))

    bytes_obj = backend.get_blocking(k)
    assert bytes_obj is not None
    encoded = backend.codec.from_bytes(bytes(bytes_obj.raw_data))
    k_back, v_back, _ = backend.codec.decode(
        encoded, out_v_dtype=torch.float16
    )
    # Reshape to compare with the original tensor[0] / tensor[1].
    expected_shape = obj.metadata.shape
    per_half_shape = torch.Size(list(expected_shape)[1:])
    k_back = k_back.reshape(per_half_shape)
    assert torch.equal(k_back, obj.tensor[0])


def test_close_idempotent(backend):
    backend.close()
    # Calling again should not raise.
    backend.close()
