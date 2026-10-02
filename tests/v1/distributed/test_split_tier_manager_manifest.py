# SPDX-License-Identifier: Apache-2.0
"""Exercise the manager-owned manifest and composite clear lifecycle."""

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_placement import (
    SplitTierManifest,
    derive_component_key,
)


def _make_key(chunk_hash: bytes = b"\x00" * 32, *, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name="test-model",
        kv_rank=0,
        cache_salt=salt,
    )


def test_storage_manager_exposes_split_tier_manifest() -> None:
    """Sanity: every StorageManager has a manifest (empty by default).
    This is the integration point the wrapper depends on."""
    # Standard
    import shutil
    import tempfile

    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
    from lmcache.v1.distributed.serde import SerdeConfig
    from lmcache.v1.distributed.storage_manager import StorageManager

    disk_path = tempfile.mkdtemp(prefix="lmcache_manifest_test_")
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
            manifest = sm.split_tier_manifest
            assert isinstance(manifest, SplitTierManifest)
            assert len(manifest) == 0
        finally:
            sm.close()
    finally:
        shutil.rmtree(disk_path, ignore_errors=True)


def test_storage_manager_clear_sweeps_stale_manifest_entries() -> None:
    """POST /cache/clear (tier=l1) removes K children from L1; the
    manifest sweep must drop the now-uncomposable entries (phantom
    COMPLETE hits + permanently blocked re-stores otherwise) while
    keeping entries whose K child survived the clear."""
    # Standard
    import shutil
    import tempfile

    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
    from lmcache.v1.distributed.serde import SerdeConfig
    from lmcache.v1.distributed.storage_manager import StorageManager

    disk_path = tempfile.mkdtemp(prefix="lmcache_clear_sweep_test_")
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
                    size_in_bytes=1 << 30,
                    use_lazy=True,
                    init_size_in_bytes=64 << 20,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
        )
        sm = StorageManager(sm_cfg)
        try:
            manifest = sm.split_tier_manifest

            # Entry A: COMPLETE but its K child is NOT in L1 (the
            # stale-after-clear shape).  It must be swept.
            stale = _make_key(chunk_hash=b"\x51" * 32)
            stale_gen = manifest.register_pending(stale)
            manifest.mark_complete(stale, stale_gen)

            # Entry B: its K child holds an L1 WRITE lock across the
            # clear (an in-flight store), so force=False keeps the L1
            # entry and the sweep must keep the manifest entry.
            live = _make_key(chunk_hash=b"\x52" * 32)
            manifest.register_pending(live)
            live_k_child = derive_component_key(live, "k")
            # Seed the K-child L1 entry via the LOW-LEVEL l1_manager, the
            # way the real split-tier store path reserves single-component
            # K children.  The high-level StorageManager.reserve_write is
            # the layout-policy choke point (it would try to split this
            # single-component [16] buffer as a packed [2, ...] group).
            reserved = sm._l1_manager.reserve_write(
                keys=[live_k_child],
                is_temporary=[False],
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([16])], dtypes=[torch.bfloat16]
                ),
            )
            assert reserved[live_k_child][1] is not None

            sm.clear(force=False)

            assert manifest.lookup(stale) is None, (
                "stale COMPLETE entry survived the clear sweep"
            )
            assert manifest.lookup(live) is not None, (
                "entry with a surviving K child was wrongly swept"
            )
            # The swept key is storable again.
            manifest.register_pending(stale)

            # Release the write lock before close.
            sm._l1_manager.finish_write([live_k_child])
        finally:
            sm.close()
    finally:
        shutil.rmtree(disk_path, ignore_errors=True)
