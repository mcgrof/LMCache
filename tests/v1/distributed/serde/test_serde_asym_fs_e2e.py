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
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchRequestSpec,
)
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
from lmcache.v1.distributed.storage_placement import (
    ComponentKeyScheme,
    StoragePlacementMode,
    derive_component_key,
)


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
            # query_prefetch_status returns a found-key Bitmap (over
            # original positions); the prefix hit count is its leading
            # ones (see StorageManager.query_prefetch_status).
            return status.count_leading_ones()
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
        # reserve_write / submit_prefetch_task apply the layout policy
        # internally (the layout choke point), so we pass the PACKED layout
        # and let them split it into K/V component groups.  Sanity-check
        # the policy does split (it must not be pre-applied by the caller).
        assert len(sm.apply_layout_policy(packed_layout).shapes) == 2, (
            "policy did not split into K/V groups"
        )

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
        reserved = sm.reserve_write(keys, packed_layout, mode="new")
        assert len(reserved) == len(keys)
        for k, (k_orig, v_orig) in zip(keys, originals, strict=True):
            mem_obj = reserved[k]
            # Multi-group MemoryObj: get_tensor(0) = K, get_tensor(1) = V.
            k_view = mem_obj.get_tensor(0)
            v_view = mem_obj.get_tensor(1)
            assert k_view is not None and v_view is not None
            k_view.copy_(k_orig)
            v_view.copy_(v_orig)
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
        handle = sm.submit_prefetch_task(
            PrefetchRequestSpec(keys=keys, group_layout_descs={0: packed_layout})
        )
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
                assert k_got is not None and v_got is not None
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
        # reserve_write applies the layout policy internally (the layout choke
        # point); pass the PACKED layout and let it split into K/V groups.
        assert len(sm.apply_layout_policy(packed_layout).shapes) == 2
        reserved = sm.reserve_write(keys, packed_layout, mode="new")
        assert len(reserved) == len(keys)
        for k, (k_orig, v_orig) in zip(keys, originals, strict=True):
            mem_obj = reserved[k]
            k_view = mem_obj.get_tensor(0)
            v_view = mem_obj.get_tensor(1)
            assert k_view is not None and v_view is not None
            k_view.copy_(k_orig)
            v_view.copy_(v_orig)
        sm.finish_write(keys)
        # Return the PACKED layout; submit_prefetch_task applies the policy.
        return packed_layout

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
                assert sm.contains_l1_key(k_child), (
                    "K child missing from L1 after V-only store"
                )
                assert not sm.contains_l1_key(k), (
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
            layout = self._store(sm, keys, originals)
            self._wait_for_l2_drain(sm, disk_path)

            # ---- Submit prefetch under the LOGICAL key ----
            handle = sm.submit_prefetch_task(
                PrefetchRequestSpec(keys=keys, group_layout_descs={0: layout})
            )
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
                    assert k_got is not None and v_got is not None
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


# =============================================================================
# Byte-through V-only split-tier (Mode "C" / RAW_UNIT) round-trip
# =============================================================================


class TestAsymBytethroughK16V8VOnlySplitTierRoundTrip:
    """Byte-through (Mode "C" / RAW_UNIT) split-tier round-trip.

    The live-asymmetric V is **already** fp8 e4m3 in HBM, so this path copies
    the raw e4m3 codes byte-through with no re-quantization.  That makes the
    V round-trip **bit-exact** (contrast the scale-aware V-only path, which
    quantizes bf16 -> fp8 and only round-trips within FP8 noise).

    The CPU tests prove the config -> StorageManager resolution chain without a
    CUDA allocator:

      * placement resolves to KV_SPLIT_TIER, layout to KV_COMPONENT_GROUPS;
      * the ``component_key_scheme`` derived at construction reaches the
        manifest as RAW_UNIT (the live #34 wiring through a real StorageManager);
      * the V component group is overridden to ``float8_e4m3fn`` (LO6): the
        byte-through V must land in L1 as fp8 with no upcast, while K stays
        bf16.

    The full byte-identity round-trip through the real CUDA-backed L1 allocator
    is the GPU capstone (``skipif`` not cuda) -- it is the M2 leg that proves
    the raw e4m3 codes survive store -> L2 -> reload unchanged.
    """

    def _build_sm(self, disk_path: str) -> StorageManager:
        fs_cfg = FSL2AdapterConfig(
            base_path=disk_path,
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
        fs_cfg.serde_config = SerdeConfig(type="asym_bytethrough_k16_v8_v_only")
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
        return StorageManager(sm_cfg)

    # ---- CPU: config -> StorageManager resolution (no allocator needed) ----

    def test_bytethrough_resolves_split_tier_component_groups_raw_unit(self) -> None:
        """The byte-through serde resolves through a real StorageManager to
        KV_SPLIT_TIER + KV_COMPONENT_GROUPS, and the derived key scheme reaches
        the manifest as RAW_UNIT (the #34 wiring, exercised end-to-end)."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_bytethrough_resolve_")
        try:
            sm = self._build_sm(disk_path)
            try:
                assert sm.storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER
                assert sm.storage_layout_mode == StorageLayoutMode.KV_COMPONENT_GROUPS
                assert (
                    sm.split_tier_manifest.component_key_scheme
                    == ComponentKeyScheme.RAW_UNIT
                )
            finally:
                sm.close()
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    def test_bytethrough_v_child_is_fp8_k_child_is_bf16(self) -> None:
        """LO6: the byte-through V component group is overridden to fp8_e4m3fn
        (no upcast) while K stays bf16; element counts are unchanged."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_bytethrough_dtype_")
        try:
            sm = self._build_sm(disk_path)
            try:
                packed = MemoryLayoutDesc(
                    shapes=[torch.Size([2, 4, 256, 128])],
                    dtypes=[torch.bfloat16],
                )
                adapted = sm.apply_layout_policy(packed)
                assert len(adapted.shapes) == 2
                # K child keeps native bf16; V child narrows to fp8 (byte-through).
                assert adapted.dtypes[0] == torch.bfloat16
                assert adapted.dtypes[1] == torch.float8_e4m3fn
                # Only the V dtype changes; the element counts are preserved.
                assert adapted.shapes[0] == torch.Size([4, 256, 128])
                assert adapted.shapes[1] == torch.Size([4, 256, 128])
            finally:
                sm.close()
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    # ---- GPU: byte-identity round-trip through the real allocator ----

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="StorageManager full E2E requires CUDA-backed L1 allocator",
    )
    def test_bytethrough_full_round_trip_is_byte_exact(self) -> None:
        """Store an already-fp8 V byte-through -> L2 -> reload; V must come back
        bit-for-bit identical (raw e4m3 codes), K bit-exact, manifest COMPLETE,
        and the K child present in L1 under the RAW_UNIT scheme."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_bytethrough_round_trip_")
        try:
            self._run_byte_exact_round_trip(disk_path)
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    def _run_byte_exact_round_trip(self, disk_path: str) -> None:
        sm = self._build_sm(disk_path)
        try:
            assert sm.storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER
            component_shape = torch.Size([4, 256, 128])
            kv_dtype = torch.bfloat16
            keys = [_make_key(b"\x30" * 31 + b"\x01")]

            torch.manual_seed(2)
            # K is native bf16; V is ALREADY fp8 e4m3 (the live-asym layout).
            originals = [
                (
                    torch.randn(component_shape, dtype=kv_dtype),
                    torch.randn(component_shape, dtype=kv_dtype).to(
                        torch.float8_e4m3fn
                    ),
                )
                for _ in keys
            ]

            # Transfer-side packed layout; apply_layout_policy splits it into a
            # bf16 K group + an fp8 V group (LO6).
            packed_layout = MemoryLayoutDesc(
                shapes=[torch.Size([2, 4, 256, 128])], dtypes=[kv_dtype]
            )
            adapted = sm.apply_layout_policy(packed_layout)
            assert adapted.dtypes[1] == torch.float8_e4m3fn, (
                "byte-through V group must be fp8 (LO6) before the round-trip"
            )

            reserved = sm.reserve_write(keys, packed_layout, mode="new")
            assert len(reserved) == len(keys)
            for k, (k_orig, v_orig) in zip(keys, originals, strict=True):
                mem_obj = reserved[k]
                k_view = mem_obj.get_tensor(0)
                v_view = mem_obj.get_tensor(1)
                assert k_view is not None and v_view is not None
                assert v_view.dtype == torch.float8_e4m3fn, (
                    "reserved V group is not fp8; the byte-through serde would "
                    "reject a non-fp8 V"
                )
                k_view.copy_(k_orig)
                v_view.copy_(v_orig)
            sm.finish_write(keys)

            # Wait for the V child to drain to L2.
            import os

            ok = wait_for_condition(
                lambda: any(e.is_file() for e in os.scandir(disk_path)),
                timeout=10.0,
            )
            assert ok, f"No V child files appeared under {disk_path}"
            ok = wait_for_condition(
                lambda: sm.report_status()["store_controller"]["in_flight_task_count"]
                == 0,
                timeout=10.0,
            )
            assert ok, "Store controller did not finish in time"

            # Manifest COMPLETE + K child in L1 under the RAW_UNIT scheme.
            for k in keys:
                assert sm.split_tier_manifest.is_complete(k), (
                    f"manifest for {k!r} is not COMPLETE"
                )
                k_child = derive_component_key(
                    k, "k", scheme=ComponentKeyScheme.RAW_UNIT
                )
                assert sm.contains_l1_key(k_child), (
                    "RAW_UNIT K child missing from L1 after byte-through store"
                )

            # Reload under the logical key and verify byte-identity.
            handle = sm.submit_prefetch_task(
                PrefetchRequestSpec(keys=keys, group_layout_descs={0: packed_layout})
            )
            prefix_hits = wait_for_prefetch_status(sm, handle)
            assert prefix_hits == len(keys), (
                f"byte-through prefetch failed: expected {len(keys)} hits, "
                f"got {prefix_hits}"
            )

            with sm.read_prefetched_results(keys) as mem_objs:
                assert mem_objs is not None
                assert len(mem_objs) == len(keys)
                for (k_orig, v_orig), mem_obj in zip(originals, mem_objs, strict=True):
                    k_got = mem_obj.get_tensor(0)
                    v_got = mem_obj.get_tensor(1)
                    assert k_got is not None and v_got is not None
                    assert torch.equal(k_got, k_orig), (
                        "K is NOT bit-exact through byte-through split-tier load"
                    )
                    # Byte-through: compare the raw e4m3 code bytes, not the
                    # decoded float values (fp8 equality via uint8 view is the
                    # unambiguous byte-identity check).
                    assert v_got.dtype == torch.float8_e4m3fn
                    assert torch.equal(
                        v_got.view(torch.uint8), v_orig.view(torch.uint8)
                    ), "V is NOT byte-identical through byte-through load"
            sm.finish_read_prefetched(keys)
        finally:
            sm.close()


class TestAsymBytethroughK16V8KVTogetherRoundTrip:
    """Both-plane byte-through (RAW_UNIT) KV_TOGETHER round-trip.

    The KV_TOGETHER counterpart of the split-tier byte-through class above.
    The ``asym_bytethrough_k16_v8`` serde returns the identity slot mapping
    ``(0, 1)``, so BOTH K (bf16) and V (fp8) are written into ONE
    self-contained L2 object.  Unlike split-tier (K kept L1-only, in-memory
    manifest), this object is durable and reusable across processes /
    restarts -- a fresh StorageManager with an empty L1 and empty manifest
    restores the full KV directly from the shared L2 store.

    CPU tests prove the config -> StorageManager resolution without a CUDA
    allocator:

      * placement resolves to KV_TOGETHER (NOT split-tier), layout to
        KV_COMPONENT_GROUPS, scheme to RAW_UNIT;
      * the split-tier manifest is inert (empty) under KV_TOGETHER;
      * apply_layout_policy still yields the canonical [K bf16, V fp8]
        component pair (pass-through for the pre-split live form; LO6
        fp8-V override for the packed form).

    The cross-process byte-identity round-trip through the real
    CUDA-backed L1 allocator is the GPU capstone (``skipif`` not cuda):
    store in one StorageManager, close it, then read the same keys back in
    a FRESH StorageManager over the same L2 dir -- the reuse that split-tier
    structurally cannot do.
    """

    def _build_sm(self, disk_path: str) -> StorageManager:
        fs_cfg = FSL2AdapterConfig(
            base_path=disk_path,
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
        fs_cfg.serde_config = SerdeConfig(type="asym_bytethrough_k16_v8")
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
        return StorageManager(sm_cfg)

    # ---- CPU: config -> StorageManager resolution (no allocator needed) ----

    def test_bytethrough_together_resolves_kv_together(self) -> None:
        """The both-plane byte-through serde resolves to KV_TOGETHER (durable,
        cross-process) + KV_COMPONENT_GROUPS + RAW_UNIT, and the split-tier
        manifest is inert (empty) because placement is not split-tier."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_bt_together_resolve_")
        try:
            sm = self._build_sm(disk_path)
            try:
                assert sm.storage_placement_mode == StoragePlacementMode.KV_TOGETHER
                assert sm.storage_layout_mode == StorageLayoutMode.KV_COMPONENT_GROUPS
                # The scheme is still RAW_UNIT (drives the fp8 V dtype + the
                # pre-split pass-through), but the manifest is inert because
                # nothing routes to the split-tier state machine under
                # KV_TOGETHER.
                assert (
                    sm.split_tier_manifest.component_key_scheme
                    == ComponentKeyScheme.RAW_UNIT
                )
            finally:
                sm.close()
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    def test_bytethrough_together_presplit_layout_passes_through(self) -> None:
        """The LIVE MP input form -- K (bf16) + V (fp8) as two pre-split
        leading-dim-1 groups -- passes through apply_layout_policy unchanged
        under KV_TOGETHER, exactly as under split-tier."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_bt_together_presplit_")
        try:
            sm = self._build_sm(disk_path)
            try:
                presplit = MemoryLayoutDesc(
                    shapes=[
                        torch.Size([1, 4, 256, 128]),
                        torch.Size([1, 4, 256, 128]),
                    ],
                    dtypes=[torch.bfloat16, torch.float8_e4m3fn],
                )
                adapted = sm.apply_layout_policy(presplit)
                assert adapted.shapes == presplit.shapes
                assert adapted.dtypes == presplit.dtypes
            finally:
                sm.close()
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    def test_bytethrough_together_packed_splits_to_fp8_v(self) -> None:
        """The packed transfer form still splits into [K bf16, V fp8] under
        KV_TOGETHER (the LO6 fp8-V override rides the RAW_UNIT scheme)."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_bt_together_dtype_")
        try:
            sm = self._build_sm(disk_path)
            try:
                packed = MemoryLayoutDesc(
                    shapes=[torch.Size([2, 4, 256, 128])],
                    dtypes=[torch.bfloat16],
                )
                adapted = sm.apply_layout_policy(packed)
                assert len(adapted.shapes) == 2
                assert adapted.dtypes[0] == torch.bfloat16
                assert adapted.dtypes[1] == torch.float8_e4m3fn
                assert adapted.shapes[0] == torch.Size([4, 256, 128])
                assert adapted.shapes[1] == torch.Size([4, 256, 128])
            finally:
                sm.close()
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    # ---- GPU: cross-process byte-identity round-trip (the point) ----

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="StorageManager full E2E requires CUDA-backed L1 allocator",
    )
    def test_bytethrough_together_cross_process_round_trip(self) -> None:
        """Store K+V byte-through into one StorageManager, close it, then read
        the SAME keys back in a FRESH StorageManager over the same L2 dir.

        This is the cross-process / restart reuse that split-tier cannot do:
        the fresh manager has an empty L1 and an empty manifest, so the KV
        must come entirely from the single durable L2 object.  Both K (bf16)
        and V (raw fp8 e4m3) must return byte-identical."""
        disk_path = tempfile.mkdtemp(prefix="lmcache_bt_together_xproc_")
        try:
            import os

            transfer_layout = MemoryLayoutDesc(
                shapes=[torch.Size([1, 4, 256, 128]), torch.Size([1, 4, 256, 128])],
                dtypes=[torch.bfloat16, torch.float8_e4m3fn],
            )
            keys = [_make_key(b"\x41" * 31 + b"\x01")]

            torch.manual_seed(7)
            # Match the transfer_layout component shape ([1, 4, 256, 128]): the
            # reserved K/V views carry the group's unit leading dim, so the
            # originals must too for the byte-exact readback comparison.
            k_orig = torch.randn(1, 4, 256, 128, dtype=torch.bfloat16)
            v_orig = torch.randn(1, 4, 256, 128, dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )

            # --- Writer process ---
            sm_a = self._build_sm(disk_path)
            try:
                assert sm_a.storage_placement_mode == StoragePlacementMode.KV_TOGETHER
                reserved = sm_a.reserve_write(keys, transfer_layout, mode="new")
                mem_obj = reserved[keys[0]]
                k_view = mem_obj.get_tensor(0)
                v_view = mem_obj.get_tensor(1)
                assert k_view is not None and v_view is not None
                assert v_view.dtype == torch.float8_e4m3fn
                k_view.copy_(k_orig)
                v_view.copy_(v_orig)
                sm_a.finish_write(keys)

                ok = wait_for_condition(
                    lambda: any(e.is_file() for e in os.scandir(disk_path)),
                    timeout=10.0,
                )
                assert ok, f"No KV object files appeared under {disk_path}"
                ok = wait_for_condition(
                    lambda: sm_a.report_status()["store_controller"][
                        "in_flight_task_count"
                    ]
                    == 0,
                    timeout=10.0,
                )
                assert ok, "Store controller did not finish in time"
            finally:
                sm_a.close()

            # --- Fresh reader process (empty L1, empty manifest) ---
            sm_b = self._build_sm(disk_path)
            try:
                handle = sm_b.submit_prefetch_task(
                    PrefetchRequestSpec(
                        keys=keys, group_layout_descs={0: transfer_layout}
                    )
                )
                prefix_hits = wait_for_prefetch_status(sm_b, handle)
                assert prefix_hits == len(keys), (
                    "cross-process byte-through prefetch failed: expected "
                    f"{len(keys)} hits from L2, got {prefix_hits}.  A fresh "
                    "process must reconstruct KV from the durable KV_TOGETHER "
                    "object."
                )
                with sm_b.read_prefetched_results(keys) as mem_objs:
                    assert mem_objs is not None
                    mem_obj = mem_objs[0]
                    k_got = mem_obj.get_tensor(0)
                    v_got = mem_obj.get_tensor(1)
                    assert k_got is not None and v_got is not None
                    assert torch.equal(k_got, k_orig), (
                        "K is NOT bit-exact across processes"
                    )
                    assert v_got.dtype == torch.float8_e4m3fn
                    assert torch.equal(
                        v_got.view(torch.uint8), v_orig.view(torch.uint8)
                    ), "V is NOT byte-identical across processes"
                sm_b.finish_read_prefetched(keys)
            finally:
                sm_b.close()
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)


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
