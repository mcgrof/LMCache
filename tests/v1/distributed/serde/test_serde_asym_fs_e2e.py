# SPDX-License-Identifier: Apache-2.0
"""
End-to-end test for the AsymK16V8 multi-output serde with a real
filesystem L2 adapter.

Mirrors ``test_serde_fs_e2e.py`` (which pins the single-tensor / fp8
layout) but exercises the new path through
:func:`StorageManager.apply_layout_policy` -> KV_COMPONENT_GROUPS:

* The integration / call site passes the transfer-side **packed**
  ``[2, ...]`` layout and lets the StorageManager split it into K
  and V component groups.
* L1Manager allocates a two-group MemoryObj; K lives at group 0,
  V at group 1.  Both share the model's native dtype (BF16 here);
  the asym codec quantizes V to FP8 internally during serialize.
* The wrapper's multi-output dispatch (PR-A) builds
  ``GroupSlotView(obj, 0)`` / ``GroupSlotView(obj, 1)`` tuples
  before calling the asym serde.
* On load the symmetric path runs: disk -> deserialize -> typed
  K and V views populated.

Verifies the data round-trips (K bit-exact, V within FP8 quantization
noise) and that no temp buffers leak.
"""

# Standard
import shutil
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_layout import StorageLayoutMode
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.distributed.storage_placement import StoragePlacementMode


# =============================================================================
# Helpers (mirror test_serde_fs_e2e.py)
# =============================================================================


def _make_key(chunk_hash: bytes) -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name="asym-test",
        kv_rank=0,
    )


def wait_for_condition(predicate, timeout=10.0, poll_interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def wait_for_prefetch_status(sm, handle, timeout=15.0, poll_interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = sm.query_prefetch_status(handle)
        if status is not None:
            return status
        time.sleep(poll_interval)
    return None


# =============================================================================
# Asym K16/V8 storage-only Mode 1 round-trip
# =============================================================================


class TestAsymK16V8SerdeFsRoundTrip:
    """Full disk-backed asym K16/V8 serde round-trip through StorageManager
    with the KV-component-groups layout policy."""

    def test_storage_placement_mode_resolves_to_kv_together(self) -> None:
        """asym_k16_v8 Mode 1 packs K + V into one stored blob; the
        placement mode is KV_TOGETHER (orthogonal to the layout mode
        which is KV_COMPONENT_GROUPS)."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_asym_placement_test_")
        try:
            fs_cfg = FSL2AdapterConfig(
                base_path=disk_path,
                relative_tmp_dir=None,
                read_ahead_size=None,
                use_odirect=False,
            )
            fs_cfg.serde_config = SerdeConfig(type="asym_k16_v8")
            sm_cfg = StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=4 << 30,
                        use_lazy=True,
                        init_size_in_bytes=1 << 30,
                    ),
                ),
                eviction_config=EvictionConfig(eviction_policy="LRU"),
                l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
            )
            sm = StorageManager(sm_cfg)
            try:
                assert sm.storage_placement_mode == StoragePlacementMode.KV_TOGETHER
            finally:
                sm.close()
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    def test_v_only_placement_mode_resolves_to_kv_split_tier(self) -> None:
        """asym_k16_v8_v_only's (None, 1) slot mapping marks K absent
        from the L2 path, so the StorageManager resolves placement to
        KV_SPLIT_TIER (split-tier placement drives the state machine)."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_vonly_placement_test_")
        try:
            fs_cfg = FSL2AdapterConfig(
                base_path=disk_path,
                relative_tmp_dir=None,
                read_ahead_size=None,
                use_odirect=False,
            )
            fs_cfg.serde_config = SerdeConfig(type="asym_k16_v8_v_only")
            sm_cfg = StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=4 << 30,
                        use_lazy=True,
                        init_size_in_bytes=1 << 30,
                    ),
                ),
                eviction_config=EvictionConfig(eviction_policy="LRU"),
                l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
            )
            sm = StorageManager(sm_cfg)
            try:
                assert sm.storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER
            finally:
                sm.close()
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    def test_storage_layout_mode_resolved_to_kv_component_groups(self) -> None:
        """An asym_k16_v8 serde_config on the L2 adapter must resolve to
        the KV_COMPONENT_GROUPS storage layout mode."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_asym_layout_test_")
        try:
            fs_cfg = FSL2AdapterConfig(
                base_path=disk_path,
                relative_tmp_dir=None,
                read_ahead_size=None,
                use_odirect=False,
            )
            fs_cfg.serde_config = SerdeConfig(type="asym_k16_v8")
            sm_cfg = StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=4 << 30,
                        use_lazy=True,
                        init_size_in_bytes=1 << 30,
                    ),
                ),
                eviction_config=EvictionConfig(eviction_policy="LRU"),
                l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
            )
            sm = StorageManager(sm_cfg)
            try:
                assert sm.storage_layout_mode == StorageLayoutMode.KV_COMPONENT_GROUPS
            finally:
                sm.close()
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    def test_apply_layout_policy_splits_packed_kv_into_component_groups(self) -> None:
        """For a KV_COMPONENT_GROUPS StorageManager, apply_layout_policy
        turns ``[2, ...]`` BF16 -> two groups of ``[...]`` BF16 each."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_asym_policy_test_")
        try:
            fs_cfg = FSL2AdapterConfig(
                base_path=disk_path,
                relative_tmp_dir=None,
                read_ahead_size=None,
                use_odirect=False,
            )
            fs_cfg.serde_config = SerdeConfig(type="asym_k16_v8")
            sm_cfg = StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=4 << 30,
                        use_lazy=True,
                        init_size_in_bytes=1 << 30,
                    ),
                ),
                eviction_config=EvictionConfig(eviction_policy="LRU"),
                l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
            )
            sm = StorageManager(sm_cfg)
            try:
                packed = MemoryLayoutDesc(
                    shapes=[torch.Size([2, 4, 256, 128])],
                    dtypes=[torch.bfloat16],
                )
                adapted = sm.apply_layout_policy(packed)
                assert len(adapted.shapes) == 2
                assert adapted.shapes[0] == torch.Size([4, 256, 128])
                assert adapted.shapes[1] == torch.Size([4, 256, 128])
                assert adapted.dtypes[0] == torch.bfloat16
                assert adapted.dtypes[1] == torch.bfloat16
            finally:
                sm.close()
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="StorageManager full E2E requires CUDA-backed L1 allocator",
    )
    def test_write_serialize_clear_prefetch_deserialize(self) -> None:
        """Write KV -> asym K16/V8 serialize -> disk -> clear L1 ->
        prefetch -> verify K bit-exact, V within FP8 noise."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_asym_serde_fs_test_")
        try:
            self._run(disk_path)
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    def _run(self, disk_path: str) -> None:
        # ---- Config: file_l2 backend with asym_k16_v8 serde ----
        fs_cfg = FSL2AdapterConfig(
            base_path=disk_path,
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
        fs_cfg.serde_config = SerdeConfig(type="asym_k16_v8")

        sm_cfg = StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4 << 30,
                    use_lazy=True,
                    init_size_in_bytes=1 << 30,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
        )

        sm = StorageManager(sm_cfg)
        assert sm.storage_layout_mode == StorageLayoutMode.KV_COMPONENT_GROUPS

        # Transfer-side packed shape (what the integration would pass).
        kv_shape = torch.Size([2, 4, 256, 128])
        kv_dtype = torch.bfloat16
        packed_layout = MemoryLayoutDesc(shapes=[kv_shape], dtypes=[kv_dtype])
        # Canonical L1 layout: K and V as separate component groups.
        layout = sm.apply_layout_policy(packed_layout)
        assert len(layout.shapes) == 2, "policy did not split into K/V groups"

        keys = [
            _make_key(b"\x00" * 31 + b"\x01"),
            _make_key(b"\x00" * 31 + b"\x02"),
        ]
        torch.manual_seed(0)
        # Per-key originals: separate K and V tensors at native dtype.
        component_shape = torch.Size([4, 256, 128])
        originals = [
            (
                torch.randn(component_shape, dtype=kv_dtype),
                torch.randn(component_shape, dtype=kv_dtype),
            )
            for _ in keys
        ]

        # ---- Step 1: reserve, fill K/V via typed group views ----
        reserved = sm.reserve_write(keys, layout, mode="new")
        assert len(reserved) == len(keys)
        for k, (k_orig, v_orig) in zip(keys, originals, strict=True):
            mem_obj = reserved[k]
            # Multi-group MemoryObj: get_tensor(0) = K, get_tensor(1) = V.
            mem_obj.get_tensor(0).copy_(k_orig)
            mem_obj.get_tensor(1).copy_(v_orig)
        sm.finish_write(keys)

        # ---- Step 2: wait for L2 store ----
        import os

        ok = wait_for_condition(
            lambda: any(e.is_file() for e in os.scandir(disk_path)),
            timeout=10.0,
        )
        assert ok, f"No files appeared under {disk_path}"
        ok = wait_for_condition(
            lambda: sm.report_status()["store_controller"]["in_flight_task_count"] == 0,
            timeout=10.0,
        )
        assert ok, "Store controller did not finish in time"

        # ---- Step 3: clear L1 ----
        sm.clear(force=True)
        assert sm.report_status()["l1_manager"]["total_object_count"] == 0

        # ---- Step 4: prefetch (disk load + asym deserialize) ----
        handle = sm.submit_prefetch_task(keys, layout)
        prefix_hits = wait_for_prefetch_status(sm, handle)
        assert prefix_hits is not None, "Prefetch never completed"
        assert prefix_hits == len(keys), (
            f"Expected {len(keys)} prefix hits, got {prefix_hits}"
        )

        # ---- Step 5: verify asym round-trip: K bit-exact, V FP8-noise ----
        with sm.read_prefetched_results(keys) as mem_objs:
            assert mem_objs is not None
            assert len(mem_objs) == len(keys)
            for (k_orig, v_orig), mem_obj in zip(originals, mem_objs, strict=True):
                k_got = mem_obj.get_tensor(0)
                v_got = mem_obj.get_tensor(1)
                # K is preserved bit-exact (the codec stores it native).
                assert torch.equal(k_got, k_orig), "K is NOT bit-exact"
                # V went through FP8 quant; allow per-tensor relative error.
                v_rel = (
                    (
                        (v_got.float() - v_orig.float()).abs()
                        / (v_orig.float().abs() + 1e-6)
                    )
                    .mean()
                    .item()
                )
                assert v_rel < 0.05, (
                    f"V FP8 round-trip relative error too high: {v_rel:.4f}"
                )

        sm.finish_read_prefetched(keys)

        # ---- Step 6: verify no L1 leak ----
        ok = wait_for_condition(
            lambda: sm.report_status()["l1_manager"]["memory_used_bytes"] == 0,
            timeout=5.0,
        )
        assert ok, (
            f"L1 memory leak: "
            f"{sm.report_status()['l1_manager']['memory_used_bytes']} bytes"
        )

        sm.close()


# =============================================================================
# Mixed-mode rejection (fp8 + asym_k16_v8 on the same StorageManager)
# =============================================================================


# =============================================================================
# V-only split-tier (KV_SPLIT_TIER placement) round-trip
# =============================================================================


class TestAsymK16V8VOnlySplitTierRoundTrip:
    """Full V-only split-tier round-trip:

      * Store a grouped (K, V) MemoryObj under a logical key.
      * Verify that on store completion, L1 contains the K child (under
        derive_component_key(logical, "k")), L2 has the V child blob,
        the original logical key is DELETED from L1, and the manifest
        is COMPLETE.
      * Submit prefetch under the logical key, verify the load
        reassembles K from L1 + V from L2 into a fresh full grouped
        MemoryObj with K bit-exact and V within FP8 noise.

    This is the empirical proof that the split-tier series
    work end-to-end (the xfail target ``test_split_policy_routes_k_to_cpu_v_to_nvme``
    is the contract version of this; this test is the integration
    version through StorageManager + a real file_l2)."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="StorageManager full E2E requires CUDA-backed L1 allocator",
    )
    def test_store_split_tier_drops_original_keeps_k_child(self) -> None:
        """Store path: after the V L2 write completes, L1 holds only
        the K child (and the original logical key is gone)."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_vonly_store_test_")
        try:
            self._run_store_only(disk_path)
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="StorageManager full E2E requires CUDA-backed L1 allocator",
    )
    def test_split_tier_full_round_trip(self) -> None:
        """Store + load round-trip with K bit-exact + V FP8 noise."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_vonly_round_trip_")
        try:
            self._run_full_round_trip(disk_path)
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _build_sm(self, disk_path: str) -> StorageManager:
        fs_cfg = FSL2AdapterConfig(
            base_path=disk_path,
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
        fs_cfg.serde_config = SerdeConfig(type="asym_k16_v8_v_only")
        sm_cfg = StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4 << 30,
                    use_lazy=True,
                    init_size_in_bytes=1 << 30,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
        )
        sm = StorageManager(sm_cfg)
        assert sm.storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER
        return sm

    def _store(self, sm: StorageManager, keys, originals):
        kv_shape = torch.Size([2, 4, 256, 128])
        kv_dtype = torch.bfloat16
        packed_layout = MemoryLayoutDesc(shapes=[kv_shape], dtypes=[kv_dtype])
        layout = sm.apply_layout_policy(packed_layout)
        assert len(layout.shapes) == 2
        reserved = sm.reserve_write(keys, layout, mode="new")
        assert len(reserved) == len(keys)
        for k, (k_orig, v_orig) in zip(keys, originals, strict=True):
            mem_obj = reserved[k]
            mem_obj.get_tensor(0).copy_(k_orig)
            mem_obj.get_tensor(1).copy_(v_orig)
        sm.finish_write(keys)
        return layout

    def _wait_for_l2_drain(self, sm: StorageManager, disk_path: str):
        # Files must appear on disk under the *V child* keys.  Since
        # we don't peek the key derivation here, just wait for any
        # file + drainer completion.
        # First Party
        # Standard
        import os

        ok = wait_for_condition(
            lambda: any(e.is_file() for e in os.scandir(disk_path)),
            timeout=10.0,
        )
        assert ok, f"No V child files appeared under {disk_path}"
        ok = wait_for_condition(
            lambda: sm.report_status()["store_controller"]["in_flight_task_count"] == 0,
            timeout=10.0,
        )
        assert ok, "Store controller did not finish in time"

    def _run_store_only(self, disk_path: str) -> None:
        # First Party
        from lmcache.v1.distributed.storage_placement import (
            SplitTierState,
            derive_component_key,
        )

        sm = self._build_sm(disk_path)
        try:
            kv_shape = torch.Size([4, 256, 128])
            kv_dtype = torch.bfloat16
            keys = [
                _make_key(b"\x10" * 31 + b"\x01"),
                _make_key(b"\x10" * 31 + b"\x02"),
            ]
            torch.manual_seed(0)
            originals = [
                (
                    torch.randn(kv_shape, dtype=kv_dtype),
                    torch.randn(kv_shape, dtype=kv_dtype),
                )
                for _ in keys
            ]
            self._store(sm, keys, originals)
            self._wait_for_l2_drain(sm, disk_path)

            # Each logical key must be COMPLETE in the manifest.
            for k in keys:
                assert sm.split_tier_manifest.is_complete(k), (
                    f"manifest for {k!r} is not COMPLETE: "
                    f"{sm.split_tier_manifest.lookup(k)}"
                )
                # The K child must be in L1; the original logical key
                # must NOT (the wrapper deletes it after V-store ack).
                k_child = derive_component_key(k, "k")
                l1_objects = sm._l1_manager._objects  # type: ignore[attr-defined]
                assert k_child in l1_objects, (
                    "K child missing from L1 after V-only store"
                )
                assert k not in l1_objects, (
                    "original logical entry still in L1 after split-tier store; "
                    "the L1 footprint win did not materialize"
                )
        finally:
            sm.close()

    def _run_full_round_trip(self, disk_path: str) -> None:
        sm = self._build_sm(disk_path)
        try:
            kv_shape = torch.Size([4, 256, 128])
            kv_dtype = torch.bfloat16
            keys = [_make_key(b"\x20" * 31 + b"\x01")]
            torch.manual_seed(1)
            originals = [
                (
                    torch.randn(kv_shape, dtype=kv_dtype),
                    torch.randn(kv_shape, dtype=kv_dtype),
                )
                for _ in keys
            ]
            packed_layout = MemoryLayoutDesc(
                shapes=[torch.Size([2, 4, 256, 128])],
                dtypes=[kv_dtype],
            )
            layout = self._store(sm, keys, originals)
            self._wait_for_l2_drain(sm, disk_path)

            # ---- Submit prefetch under the LOGICAL key ----
            handle = sm.submit_prefetch_task(keys, layout)
            prefix_hits = wait_for_prefetch_status(sm, handle)
            assert prefix_hits == len(keys), (
                f"split-tier prefetch failed: expected {len(keys)} hits, "
                f"got {prefix_hits}"
            )

            # ---- Verify K bit-exact + V FP8 noise via reassembled MemoryObj ----
            with sm.read_prefetched_results(keys) as mem_objs:
                assert mem_objs is not None
                assert len(mem_objs) == len(keys)
                for (k_orig, v_orig), mem_obj in zip(originals, mem_objs, strict=True):
                    k_got = mem_obj.get_tensor(0)
                    v_got = mem_obj.get_tensor(1)
                    assert torch.equal(k_got, k_orig), (
                        "K is NOT bit-exact through split-tier load"
                    )
                    v_rel = (
                        (
                            (v_got.float() - v_orig.float()).abs()
                            / (v_orig.float().abs() + 1e-6)
                        )
                        .mean()
                        .item()
                    )
                    assert v_rel < 0.05, (
                        f"V FP8 round-trip relative error too high: {v_rel:.4f}"
                    )
            sm.finish_read_prefetched(keys)
        finally:
            sm.close()


def test_mixed_single_and_multi_output_serdes_rejected() -> None:
    """A StorageManager with one fp8 adapter and one asym_k16_v8 adapter
    must fail at config time -- the two demand incompatible L1 layouts."""
    disk_path_a = tempfile.mkdtemp(prefix="lmcache_mixed_a_")
    disk_path_b = tempfile.mkdtemp(prefix="lmcache_mixed_b_")
    try:
        fs_cfg_fp8 = FSL2AdapterConfig(
            base_path=disk_path_a,
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
        fs_cfg_fp8.serde_config = SerdeConfig(type="fp8")
        fs_cfg_asym = FSL2AdapterConfig(
            base_path=disk_path_b,
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
        fs_cfg_asym.serde_config = SerdeConfig(type="asym_k16_v8")
        sm_cfg = StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4 << 30,
                    use_lazy=True,
                    init_size_in_bytes=1 << 30,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(
                adapters=[fs_cfg_fp8, fs_cfg_asym]  # type: ignore[list-item]
            ),
        )
        with pytest.raises(ValueError, match="Incompatible L2 adapter storage layout"):
            StorageManager(sm_cfg)
    finally:
        shutil.rmtree(disk_path_a, ignore_errors=True)
        shutil.rmtree(disk_path_b, ignore_errors=True)
