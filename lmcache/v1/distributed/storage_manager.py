# SPDX-License-Identifier: Apache-2.0
"""
Distributed multi-tier storage manager for MP mode
"""

# Standard
from contextlib import contextmanager
from typing import Iterator, Literal, Optional
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap, PeriodicEventNotifier
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchMode,
    TrimPolicy,
)
from lmcache.v1.distributed.config import EvictionConfig, StorageManagerConfig
from lmcache.v1.distributed.error import L1Error, strerror
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2AdapterListener
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    get_type_name_for_config,
)
from lmcache.v1.distributed.l2_adapters.reconfiguration import (
    L2ReconfigurableAdapter,
    L2ReconfigureError,
)
from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.distributed.serde import create_serde_processor
from lmcache.v1.distributed.storage_controllers import (
    L1EvictionController,
    L2AdapterEvictionState,
    L2EvictionController,
    PrefetchController,
    StoreController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    create_prefetch_policy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    create_store_policy,
)
from lmcache.v1.distributed.storage_layout import (
    StorageLayoutMode,
    apply_layout_policy as _apply_layout_policy,
    derive_storage_layout_mode,
)
from lmcache.v1.distributed.storage_placement import (
    SplitTierConfigError,
    SplitTierManifest,
    StoragePlacementMode,
    derive_component_key,
    derive_storage_placement_mode,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.mp_observability.trace.decorator import (
    enable_tracing,
    is_tracing_enabled,
    publish_call_event,
)
from lmcache.v1.platform import HAS_EVENTFD

logger = init_logger(__name__)


def _reject_unsupported_split_tier_config(config: StorageManagerConfig) -> None:
    """Enforce the KV_SPLIT_TIER (V-only) support matrix at config time.

    The split-tier path composes an L1-resident exact-K child with an
    L2-resident compressed-V child under one logical key.  It is only
    correct for the narrow initial matrix documented in
    ``docs/design/v1/distributed/`` -- CPU pinned-DRAM L1, exactly one
    split-tier L2 adapter, no L2 eviction/quota driving the V child out
    from under a live composite.  Configurations outside that matrix are
    rejected here, before any resource is built, so they fail closed at
    startup rather than silently mis-serving.

    Only call this when the derived placement mode is
    :attr:`StoragePlacementMode.KV_SPLIT_TIER`; KV_TOGETHER configs are
    unrestricted.

    Args:
        config: The fully-normalized storage manager configuration.

    Raises:
        SplitTierConfigError: naming every unsupported combination
            present, so an operator sees all matrix violations at once.
    """
    reasons: list[str] = []

    l1 = config.l1_manager_config
    if l1.gds_l1_config is not None:
        reasons.append(
            "GDS L1 (gds-l1-path): split-tier requires a CPU pinned-DRAM "
            "L1 tier to retain the exact-K child"
        )
    if l1.memory_config.devdax_path:
        reasons.append(
            "Device-DAX L1 (l1-devdax-path): split-tier requires a CPU "
            "pinned-DRAM L1 tier to retain the exact-K child"
        )

    adapters = config.l2_adapter_config.adapters
    if len(adapters) > 1:
        names = ", ".join(get_type_name_for_config(ac) for ac in adapters)
        reasons.append(
            f"multiple L2 adapters ({names}): split-tier supports exactly "
            "one L2 adapter for the V child"
        )

    evicting = [
        get_type_name_for_config(ac)
        for ac in adapters
        if ac.eviction_config is not None
    ]
    if evicting:
        reasons.append(
            f"per-adapter L2 eviction ({', '.join(evicting)}): L2 eviction "
            "can delete a V child out from under a live composite"
        )

    quota_sources: list[str] = []
    if config.eviction_config.eviction_policy == "IsolatedLRU":
        quota_sources.append("L1 eviction policy")
    quota_sources.extend(
        get_type_name_for_config(ac)
        for ac in adapters
        if ac.eviction_config is not None
        and ac.eviction_config.eviction_policy == "IsolatedLRU"
    )
    if quota_sources:
        reasons.append(
            f"IsolatedLRU per-cache_salt quota ({', '.join(quota_sources)}): "
            "quota accounting does not yet model the split K/V children"
        )

    if reasons:
        raise SplitTierConfigError(
            "Unsupported configuration for KV_SPLIT_TIER (V-only) storage "
            "placement:\n  - " + "\n  - ".join(reasons)
        )


class StorageManager:
    def __init__(self, config: StorageManagerConfig):
        # Canonical L1 placement / lifecycle mode, derived from the
        # configured L2 adapters' serdes.  Determines whether the wrapper
        # packs K + V into one stored blob (KV_TOGETHER) or splits them
        # into separately keyed children with K retained in L1 and V
        # routed to L2 (KV_SPLIT_TIER, the Mode 2 V-only path).  Mixed
        # configurations are rejected.  Derived FIRST -- it is pure config
        # inspection -- so the split-tier support matrix can fail closed
        # before any hardware-backed resource (the L1 memory manager, the
        # adapters) is constructed.
        self._storage_placement_mode = derive_storage_placement_mode(
            config.l2_adapter_config.adapters
        )

        # Support matrix (Phase 1a): the V-only split-tier path is only
        # correct for a narrow, documented configuration.  Reject the
        # unsupported combinations before any resource is built so a bad
        # config fails closed at startup instead of silently mis-serving.
        # KV_TOGETHER is unaffected (the check is gated on the mode).
        if self._storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
            _reject_unsupported_split_tier_config(config)

        self._l1_manager = L1Manager(config.l1_manager_config)
        self._event_bus = get_event_bus()

        # L1 eviction controller
        self._eviction_controller = L1EvictionController(
            l1_manager=self._l1_manager,
            eviction_config=config.eviction_config,
        )
        self._eviction_controller.start()

        # Canonical L1 storage layout mode, derived from the configured
        # L2 adapters' serdes.  All adapters must share one mode; mixing
        # a single-tensor serde (e.g. ``fp8``) with a multi-output serde
        # (e.g. ``asym_k16_v8``) on the same StorageManager is rejected
        # at config time.  See lmcache/v1/distributed/storage_layout.py.
        self._storage_layout_mode = derive_storage_layout_mode(
            config.l2_adapter_config.adapters
        )

        # Per-logical-key state machine for KV_SPLIT_TIER.  Always
        # constructed (cheap; empty when placement is KV_TOGETHER).
        # The wrapper queries and mutates this manifest as part of
        # the V-only store / load composition.
        self._split_tier_manifest = SplitTierManifest()

        # L2 adapters and store controller. When an adapter config carries
        # a ``serde_config``, the adapter is wrapped with
        # ``SerdeL2AdapterWrapper`` so controllers see a plain L2 adapter
        # and serde is transparent.
        self._l1_memory_desc = self._l1_manager.get_l1_memory_desc()
        self._next_adapter_id = 0
        # Serializes add_l2_adapter / delete_l2_adapter against each other.
        self._lifecycle_lock = threading.Lock()
        # Guards the _l2_adapters and _adapter_descriptors dicts.
        self._adapters_lock = threading.Lock()
        self._registered_l2_listeners: list[L2AdapterListener] = []
        self._l2_adapters: dict[int, L2AdapterInterface] = {}
        self._adapter_descriptors: dict[int, AdapterDescriptor] = {}
        for ac in config.l2_adapter_config.adapters:
            adapter_id, adapter, descriptor = self._build_l2_adapter(ac)
            self._l2_adapters[adapter_id] = adapter
            self._adapter_descriptors[adapter_id] = descriptor

        PeriodicEventNotifier.create(
            interval_ms=config.periodic_notifier_interval_ms,
            use_eventfd=HAS_EVENTFD,
        )

        # PR-3' wire-up: attach the manifest + L2 adapters to the L1
        # eviction controller (constructed earlier so its L1 listener
        # was registered before any allocate event).  When placement
        # is KV_SPLIT_TIER, this lets the controller drive paired
        # cleanup (mark INVALIDATED, enqueue V child delete to L2)
        # for K-child evictions.  Safe to call always: KV_TOGETHER
        # mode skips the paired path because no K-child keys are in
        # the eviction-candidate set.
        if self._storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
            self._eviction_controller.set_split_tier_paired_eviction(
                self._split_tier_manifest,
                list(self._l2_adapters.values()),
            )
            # Also wire the manifest into L1Manager so its
            # ``is_key_evictable`` gates K-child eviction on manifest
            # state — STORE_IN_FLIGHT K-children stay pinned across
            # the V codec + L2 write window (PR-11').
            self._l1_manager.set_split_tier_manifest(self._split_tier_manifest)

        # Per-cache_salt quota registry. Shared across the L2 eviction
        # controller (reads quotas each cycle) and the HTTP quota
        # endpoints (CRUD). Present even when no adapter uses
        # IsolatedLRU so the HTTP layer has a stable ``quota_manager``
        # reference. No explicit cleanup on close — the registry is
        # just a dict protected by a lock and has no OS resources.
        self._quota_manager = QuotaManager()

        # Unified L2 eviction controller for all adapters with eviction
        # config. Aggregate-usage policies (``LRU``, ``noop``) need
        # ``max_capacity_bytes > 0`` to compute a usage fraction;
        # adapters without capacity are skipped for those. Isolated
        # policies (``IsolatedLRU``) operate on per cache_salt byte
        # counts which the base class tracks regardless of capacity,
        # so they are wired up unconditionally.
        l2_eviction_states: list[L2AdapterEvictionState] = []
        for adapter_id, ac in zip(
            self._l2_adapters, config.l2_adapter_config.adapters, strict=True
        ):
            adapter = self._l2_adapters[adapter_id]
            if self._should_enable_l2_eviction(adapter, ac.eviction_config):
                assert ac.eviction_config is not None  # make linter happy
                l2_eviction_states.append(
                    L2AdapterEvictionState(
                        adapter_id=adapter_id,
                        adapter=adapter,
                        eviction_config=ac.eviction_config,
                    )
                )
        self._l2_eviction_controller = L2EvictionController(
            l2_eviction_states, quota_manager=self._quota_manager
        )
        self._l2_eviction_controller.start()

        # Controllers receive the initial set as ordered lists; they key
        # their own copies by ``descriptor.index`` (== adapter_id) and learn
        # of later changes via add_adapter/request_remove_adapter.
        self._store_controller = StoreController(
            l1_manager=self._l1_manager,
            l2_adapters=list(self._l2_adapters.values()),
            adapter_descriptors=list(self._adapter_descriptors.values()),
            policy=create_store_policy(config.store_policy),
            split_tier_manifest=self._split_tier_manifest,
        )
        self._store_controller.start()

        # Prefetch controller
        self._prefetch_controller = PrefetchController(
            l1_manager=self._l1_manager,
            l2_adapters=list(self._l2_adapters.values()),
            adapter_descriptors=list(self._adapter_descriptors.values()),
            policy=create_prefetch_policy(config.prefetch_policy),
            max_in_flight=config.prefetch_max_in_flight,
        )
        self._prefetch_controller.start()

        # L2 usage gauge — one observation per adapter, tagged by
        # ``l2_name``.  Parallel to L1Manager's ``l1_memory_usage_bytes``.
        register_gauge(
            "lmcache.l2",
            "lmcache_mp.l2_usage_bytes",
            (
                "Bytes currently held in each L2 adapter, tagged by "
                "``l2_name`` (one observation per adapter)."
            ),
            self.get_l2_usages,
        )

    # External APIs for serving engine integration code to call

    @property
    def split_tier_manifest(self) -> SplitTierManifest:
        """The per-logical-key state machine for KV_SPLIT_TIER mode.

        Always present (empty when placement is :attr:`KV_TOGETHER`).
        The wrapper drives transitions during store / load; the
        eviction controller drives invalidation + delete-in-flight
        transitions during paired cleanup (PR-3').

        Exposed as a property so the wrapper can hold a reference
        without needing the whole StorageManager.
        """
        return self._split_tier_manifest

    @property
    def storage_placement_mode(self) -> StoragePlacementMode:
        """The canonical placement / lifecycle mode for this
        StorageManager.

        :attr:`StoragePlacementMode.KV_TOGETHER` packs K + V into one
        stored blob (the default for ``fp8`` and asym Mode 1 K16/V8).
        :attr:`StoragePlacementMode.KV_SPLIT_TIER` drives the V-only
        state machine: K child retained in L1, V child stored to L2,
        original logical staging object deleted in L1.  Derived once
        at construction from the configured serdes; orthogonal to
        :attr:`storage_layout_mode` (which decides the L1 *shape*,
        not the lifecycle).
        """
        return self._storage_placement_mode

    @property
    def storage_layout_mode(self) -> StorageLayoutMode:
        """The canonical L1 ``MemoryObj`` shape this StorageManager uses.

        Derived once at construction from the configured L2 adapters'
        serdes.  Integration layers (vLLM / SGLang connectors, MP
        server) should call :meth:`apply_layout_policy` on the
        transfer-side packed layout before :meth:`reserve_write` so the
        canonical shape is reached regardless of which serde is wired.
        """
        return self._storage_layout_mode

    def apply_layout_policy(self, layout_desc: MemoryLayoutDesc) -> MemoryLayoutDesc:
        """Adapt a transfer-side packed layout to this StorageManager's
        canonical L1 shape.

        For the default ``PACKED`` mode this is a no-op pass-through;
        for ``KV_COMPONENT_GROUPS`` (selected when a multi-output serde
        is configured) this splits each input group's leading-``2`` K|V
        dim into separate K and V component groups so multi-output
        serdes see K and V as distinct typed sub-objects via
        ``TensorMemoryObj.get_tensor(0)`` / ``get_tensor(1)``.

        Total bytes are unchanged.  The transfer kernel writes K bytes
        followed by V bytes either way; only the typed view changes.

        Args:
            layout_desc: Packed layout as produced by the transfer side
                (each input group has leading dim 2 for the K|V pair).

        Returns:
            Layout to pass to :meth:`reserve_write`.
        """
        return _apply_layout_policy(layout_desc, self._storage_layout_mode)

    @enable_tracing()
    def reserve_write(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"],
    ) -> dict[ObjectKey, MemoryObj]:
        """
        Reserve the object for writing into the storage manager.

        Args:
            keys (list[ObjectKey]): List of object keys to reserve for writing.
            layout_desc (MemoryLayoutDesc): Description of the memory layout
                for the objects to be reserved.
            mode (Literal["new", "update", "all"]): Reservation mode.
            - "new": Reserve only new objects that do not exist.
            - "update": Reserve only existing objects for update.
            - "all": Reserve all writable objects regardless of existence.

        Returns:
            dict[ObjectKey, MemoryObj]: A dictionary mapping object keys to their
                reserved memory objects. Note that not all requested keys could be
                reserved (e.g., out of memory or write conflict)

        Raises:
            ValueError: under KV_SPLIT_TIER, if any key carries a non-zero
                ``object_group_id`` -- a fact invisible at config time.
                Multiple object groups (hybrid / sliding-window / MLA
                models emit one key per KV cache group) are outside the
                split-tier support matrix; failing closed here surfaces
                the mismatch to the producer instead of silently building
                independent, untested per-group composites.
        """
        if self._storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
            multi_group = [k for k in keys if k.object_group_id != 0]
            if multi_group:
                raise ValueError(
                    "KV_SPLIT_TIER (V-only) placement supports a single "
                    f"object group, but {len(multi_group)} of {len(keys)} "
                    "key(s) carry object_group_id != 0 (hybrid / "
                    "sliding-window / MLA models emit multiple object "
                    "groups). This model topology is outside the "
                    "split-tier support matrix."
                )
        reserve_result = self._l1_manager.reserve_write(
            keys=keys,
            is_temporary=[False] * len(keys),
            layout_desc=layout_desc,
            mode=mode,
        )

        result = {k: m for k, (e, m) in reserve_result.items() if m is not None}
        successful_keys = list(result.keys())
        failed_keys = [k for k, (e, m) in reserve_result.items() if m is None]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_WRITE_RESERVED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )

        oom_keys = [
            k for k, (e, _) in reserve_result.items() if e == L1Error.OUT_OF_MEMORY
        ]
        if oom_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_ALLOCATION_FAILED,
                    metadata={"during": "l1_store", "keys": oom_keys},
                )
            )

        return result

    @enable_tracing()
    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """
        Finish writing the objects into the storage manager.

        Args:
            keys (list[ObjectKey]): List of object keys that have been written.
        """
        finish_result = self._l1_manager.finish_write(keys)
        successful_keys = [k for k, e in finish_result.items() if e == L1Error.SUCCESS]
        failed_keys = [k for k, e in finish_result.items() if e != L1Error.SUCCESS]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_WRITE_FINISHED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )

        # TODO: global key states update

    @contextmanager
    def read_prefetched_results(
        self,
        keys: list[ObjectKey],
    ) -> Iterator[list[MemoryObj] | None]:
        """
        Read the memory objects from L1 storage that has been prefetched beforehand.
        Yielding an optional list of memory objects corresponding to the requested
        keys. If any the object is not found in L1, None is yielded.

        Args:
            keys (list[ObjectKey]): List of object keys to reserve for reading.

        Returns:
            Iterator[list[MemoryObj] | None]: An iterator yielding an optional list of
                memory objects corresponding to the requested keys.

        Note:
            If any object is not found in L1 storage, None is yielded. In this case,
            this function will release release the read lock of all successfully read
            memory objects when exiting the context.

            If the caller raised exception during the processing of the yielded memory
            objects, this function will ensure that the read locks will be decreased.
        """
        # Manual TRACE_CALL emission for the context manager.  The
        # ``@enable_tracing`` decorator cannot wrap a ``@contextmanager``
        # generator function (it would publish the call to the wrapper
        # rather than to ``__enter__``).  Emit enter/exit events
        # directly, gated on the tracing flag for zero overhead when
        # disabled.
        if is_tracing_enabled():
            publish_call_event(
                "lmcache.v1.distributed.storage_manager."
                "StorageManager.read_prefetched_results.__enter__",
                {"keys": keys},
            )
        read_results = self._l1_manager.unsafe_read(keys)
        good_keys: list[ObjectKey] = []
        good_objs: list[MemoryObj] = []
        bad_keys: list[ObjectKey] = []
        not_found_keys: list[ObjectKey] = []
        write_locked_keys: list[ObjectKey] = []
        all_good = True
        for k, (e, o) in read_results.items():
            if o is None:
                logger.error(
                    "Failed to read prefetched object %s from L1 storage: %s",
                    k,
                    strerror(e),
                )
                bad_keys.append(k)
                all_good = False
                if e == L1Error.KEY_NOT_EXIST:
                    not_found_keys.append(k)
                elif e == L1Error.KEY_NOT_READABLE:
                    write_locked_keys.append(k)
                continue

            good_keys.append(k)
            good_objs.append(o)

        # L1 read-failure anomaly reporting: unsafe_read is required to be
        # called post-reserve_read, so any failure here is a lock/eviction
        # race, not a normal cache miss.
        if not_found_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_READ_FAILED,
                    metadata={
                        "during": "l1_retrieve",
                        "reason": "not_found",
                        "keys": not_found_keys,
                    },
                )
            )
        if write_locked_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_READ_FAILED,
                    metadata={
                        "during": "l1_retrieve",
                        "reason": "write_locked",
                        "keys": write_locked_keys,
                    },
                )
            )

        successfully_yielded = False

        try:
            yield good_objs if all_good else None
            successfully_yielded = True
        except Exception:
            logger.exception(
                "Exception occurred while processing read prefetched results",
            )
            raise
        finally:
            # Decrease the read lock for all successfully read memory objects
            # if None is yielded or exception occurs during caller's processing
            if not all_good or not successfully_yielded:
                self._l1_manager.finish_read(good_keys)
                self._event_bus.publish(
                    Event(
                        event_type=EventType.SM_READ_PREFETCHED_FINISHED,
                        metadata={
                            "succeeded_keys": good_keys,
                            "failed_keys": bad_keys,
                        },
                    )
                )
            if is_tracing_enabled():
                publish_call_event(
                    "lmcache.v1.distributed.storage_manager."
                    "StorageManager.read_prefetched_results.__exit__",
                    {"keys": keys},
                )

    @enable_tracing()
    def finish_read_prefetched(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> None:
        """Finish reading prefetched objects.

        Args:
            keys: Object keys that have been read.
            extra_count: Extra read locks to release per key
                (on top of the default 1).
        """
        finish_result = self._l1_manager.finish_read(keys, extra_count=extra_count)
        successful_keys = [k for k, e in finish_result.items() if e == L1Error.SUCCESS]
        failed_keys = [k for k, e in finish_result.items() if e != L1Error.SUCCESS]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_READ_PREFETCHED_FINISHED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )

    @enable_tracing()
    def submit_prefetch_task(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        extra_count: int = 0,
        external_request_id: str = "",
        policy: TrimPolicy = TrimPolicy.PREFIX,
        attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC,
        skip_l2: bool = False,
        mode: PrefetchMode = PrefetchMode.LOOKUP,
    ) -> PrefetchHandle:
        """Prefetch objects into L1 asynchronously.

        Args:
            keys: Object keys to prefetch.
            layout_desc: Memory layout description.
            extra_count: Extra workers (on top of the default
                1) that will independently retrieve the same
                key.  Total locks = 1 + extra_count.
            external_request_id: Request ID from the caller
                for end-to-end log tracing.
            policy: Which retained-subset policy to apply (see
                :class:`TrimPolicy`).  ``PREFIX`` keeps the contiguous prefix;
                ``SPARSE`` keeps every found key (gap-tolerant).
            attn_desc: Cross-chunk attention windows of all object groups, in
                object-group order.
            skip_l2: If True, do not load from L2. For ``LOOKUP`` only
                already-resident L1 keys are returned; for ``WARM`` nothing is
                loaded and an empty handle is returned.
            mode: The prefetch intent (see :class:`PrefetchMode`).  ``WARM``
                retains loaded keys and pins none; ``LOOKUP`` (default) pins
                them for an imminent reader and follows the policy.

        Returns:
            PrefetchHandle to track the task.
        """
        if mode is PrefetchMode.WARM:
            # Warm path: load all keys, pin none. skip_l2 makes it a no-op.
            prefetch_request_id = -1
            if not skip_l2 and keys and self._l2_adapters:
                prefetch_request_id = self._prefetch_controller.submit_prefetch_request(
                    keys,
                    layout_desc,
                    extra_count=extra_count,
                    policy=policy,
                    mode=mode,
                )
            return PrefetchHandle(
                prefetch_request_id=prefetch_request_id,
                external_request_id=external_request_id,
                l1_found_indices=(),
                total_requested_keys=len(keys),
                submit_time=time.monotonic(),
                l2_orig_indices=(
                    tuple(range(len(keys))) if prefetch_request_id != -1 else ()
                ),
            )

        # NOTE: now we only have L1, so the prefetch is essentially checking how many
        # objects are already in L1, and adding read locks to them.

        l1_read_result = self._l1_manager.reserve_read(keys, extra_count=extra_count)

        if policy is TrimPolicy.SPARSE:
            # SPARSE: retain a read lock on every L1 hit (not just the leading
            # prefix) and send all L1 misses to L2 as one coalesced request.
            # reserve_read locks only SUCCESS keys, so the found-set already
            # equals the locked set -- nothing to release.
            l1_found_indices: list[int] = []
            succeeded_keys: list[ObjectKey] = []
            sparse_l2_indices: list[int] = []
            remaining_keys: list[ObjectKey] = []
            for i, key in enumerate(keys):
                ent = l1_read_result.get(key)
                if ent is not None and ent[0] == L1Error.SUCCESS and ent[1] is not None:
                    l1_found_indices.append(i)
                    succeeded_keys.append(key)
                else:
                    sparse_l2_indices.append(i)
                    remaining_keys.append(key)

            self._event_bus.publish(
                Event(
                    event_type=EventType.SM_READ_PREFETCHED,
                    metadata={
                        "succeeded_keys": succeeded_keys,
                        "failed_keys": remaining_keys,
                    },
                )
            )

            prefetch_request_id = -1
            if remaining_keys and self._has_l2_adapters():
                prefetch_request_id = self._prefetch_controller.submit_prefetch_request(
                    remaining_keys,
                    layout_desc,
                    extra_count=extra_count,
                    policy=TrimPolicy.SPARSE,
                    attn_desc=attn_desc,
                    mode=mode,
                )
            return PrefetchHandle(
                prefetch_request_id=prefetch_request_id,
                external_request_id=external_request_id,
                l1_found_indices=tuple(l1_found_indices),
                total_requested_keys=len(keys),
                submit_time=time.monotonic(),
                l2_orig_indices=tuple(sparse_l2_indices),
            )

        hit_count = 0
        for key in keys:
            entry = l1_read_result.get(key, None)
            if entry is None:
                break

            err, obj = entry
            if err != L1Error.SUCCESS:
                break

            hit_count += 1

        # NOTE: For L1, there will be cases that "object in the middle" is not found.
        # In this case, we need to `finish_read` for the latter objects so that
        # there won't be dangling read locks.
        skipped_keys = []
        for key in keys[hit_count:]:
            if key in l1_read_result and l1_read_result[key][1] is not None:
                # this key is actually reserved, need to release the read lock
                skipped_keys.append(key)

        if skipped_keys:
            self._l1_manager.finish_read(skipped_keys, extra_count=extra_count)

        self._event_bus.publish(
            Event(
                event_type=EventType.SM_READ_PREFETCHED,
                metadata={
                    "succeeded_keys": keys[:hit_count],
                    "failed_keys": keys[hit_count:],
                },
            )
        )

        if skip_l2:
            return PrefetchHandle(
                prefetch_request_id=-1,
                external_request_id=external_request_id,
                l1_found_indices=tuple(range(hit_count)),
                total_requested_keys=len(keys),
                submit_time=time.monotonic(),
                l2_orig_indices=(),
            )

        # Submit remaining keys to L2 prefetch controller
        remaining_keys = keys[hit_count:]
        prefetch_request_id = -1
        l2_orig_indices: tuple[int, ...] = ()
        if remaining_keys and self._has_l2_adapters():
            prefetch_request_id = self._prefetch_controller.submit_prefetch_request(
                remaining_keys,
                layout_desc,
                extra_count=extra_count,
                attn_desc=attn_desc,
                policy=policy,
                mode=mode,
            )
            # The controller indexes its result bitmap over remaining_keys
            # (0-based); map those local indices back to original positions.
            l2_orig_indices = tuple(range(hit_count, len(keys)))

        submit_time = time.monotonic()
        logger.debug(
            "Prefetch request submitted: "
            "%d total keys, %d L1 prefix hits, "
            "%d remaining for L2 "
            "(external_request_id=%s, "
            "prefetch_request_id=%d)",
            len(keys),
            hit_count,
            len(remaining_keys),
            external_request_id,
            prefetch_request_id,
        )

        return PrefetchHandle(
            prefetch_request_id=prefetch_request_id,
            external_request_id=external_request_id,
            l1_found_indices=tuple(range(hit_count)),
            total_requested_keys=len(keys),
            submit_time=submit_time,
            l2_orig_indices=l2_orig_indices,
        )

    def _combine_found(
        self, handle: PrefetchHandle, l2_local: "Bitmap | None"
    ) -> Bitmap:
        """Merge the L1 found indices with an L2 result bitmap into one bitmap
        over the original key positions.

        ``l2_local`` is indexed over the keys submitted to L2 (0-based); its
        set bits are mapped back to original positions via
        ``handle.l2_orig_indices``.
        """
        found = Bitmap(handle.total_requested_keys)
        found.batched_set(handle.l1_found_indices)
        if l2_local is not None:
            # gather maps each L2 set bit i to its original position
            # ``l2_orig_indices[i]``; batched_set drops any position >= size.
            found.batched_set(l2_local.gather(handle.l2_orig_indices))
        return found

    def query_prefetch_lookup_hits(
        self,
        handle: PrefetchHandle,
    ) -> int | None:
        """
        Query the number of prefix-hit chunks for a prefetch task before the
        L2 prefetching is done.

        Args:
            handle (PrefetchHandle): The handle of the lookup task.

        Returns:
            the number of prefix-hit chunks (L1 + L2) if the lookup is done,
            None if it's still in progress or the prefetch task is already done.

        Note:
            This function is designed for the scenario where the caller wants
            to check the L1 prefix hits as soon as possible without waiting for
            the whole prefetch task to be done.
            When the prefetch task is already done and the prefetch task result
            has already been queried by `query_prefetch_status`, this function
            will return None forever for the same prefetch handle.
            Therefore, it's the caller’s responsibility to make sure not calling
            this function after the prefetch task is done.
        """
        # Prefix-path only: l1_found_indices is contiguous, so len() == prefix hits.
        l1_hits = len(handle.l1_found_indices)
        if handle.prefetch_request_id == -1:
            # No L2 request: the L1 prefix hit count is final.
            return l1_hits

        l2_r = self._prefetch_controller.query_lookup_result(handle.prefetch_request_id)
        if l2_r is None:
            # Still in progress, or already consumed by query_prefetch_status.
            return None
        # L2 lookup done: total prefix hits are L1 plus the L2 continuation.
        return l1_hits + l2_r

    def wait_prefetch_status(
        self,
        handle: PrefetchHandle,
        timeout: float,
    ) -> bool:
        """
        Block until the prefetch task for ``handle`` has a result, or timeout.

        L1-only prefetches (``prefetch_request_id == -1``) have no L2 result to
        wait for and return immediately. This lets a caller avoid busy-polling
        query_prefetch_status; the status itself is still retrieved via
        query_prefetch_status afterwards.

        Args:
            handle (PrefetchHandle): The handle of the prefetch task.
            timeout: Maximum number of seconds to wait for the L2 result.

        Returns:
            True if a result is available within the timeout (always True for
            an L1-only prefetch), False if the wait timed out.
        """
        if handle.prefetch_request_id == -1:
            return True
        return self._prefetch_controller.wait_prefetch_result(
            handle.prefetch_request_id, timeout
        )

    def query_prefetch_status(
        self,
        handle: PrefetchHandle,
    ) -> Bitmap | None:
        """
        Query the status of the prefetch task.

        Args:
            handle (PrefetchHandle): The handle of the prefetch task.

        Returns:
            the found-key bitmap (over original positions) if the prefetch is
            done, None if it's still in progress. Derive the prefix hit count
            via ``count_leading_ones``.
        """
        l2_r: Bitmap | None = None
        if handle.prefetch_request_id != -1:
            l2_r = self._prefetch_controller.query_prefetch_result(
                handle.prefetch_request_id
            )
            if l2_r is None:
                return None

        found = self._combine_found(handle, l2_r)
        # popcount (not count_leading_ones) so the log is accurate for
        # non-contiguous policies (SEGMENTED_PREFIX / SPARSE) too.
        total_hits = found.popcount()
        elapsed_ms = (time.monotonic() - handle.submit_time) * 1000

        if total_hits > 0:
            # L1 and L2 sets are disjoint (only L1-misses go to L2).
            l1_hits = len(handle.l1_found_indices)
            l2_hits = l2_r.popcount() if l2_r is not None else 0
            logger.info(
                "Prefetch request completed (L1+L2): "
                "%d/%d retained keys (%d L1, %d L2) in %.1f ms "
                "(external_request_id=%s, prefetch_request_id=%d)",
                total_hits,
                handle.total_requested_keys,
                l1_hits,
                l2_hits,
                elapsed_ms,
                handle.external_request_id,
                handle.prefetch_request_id,
            )
        return found

    def touch_l1_keys(self, keys: list[ObjectKey]):
        """
        Touch the keys in L1 storage, marking the keys
        as accessed(retrieved or stored).

        Args:
            keys (list[ObjectKey]): List of object keys to touch.
        """
        self._l1_manager.touch_keys(keys)

    def delete_l1_keys(
        self, keys: list[ObjectKey], force: bool = False
    ) -> tuple[int, int]:
        """Delete the given keys from L1.

        Args:
            keys (list[ObjectKey]): List of object keys to delete.
            force (bool): When True, delete even read/write-locked keys; else
                skip them.

        Returns:
            tuple[int, int]: ``(deleted, skipped)`` -- the number of keys removed
                and the number refused because they were locked (non-force only).
                Missing keys are a no-op, so the operation is idempotent.
        """
        results = self._l1_manager.delete(keys, force=force)
        deleted = sum(1 for err in results.values() if err == L1Error.SUCCESS)
        skipped = sum(1 for err in results.values() if err == L1Error.KEY_IS_LOCKED)
        return deleted, skipped

    def contains_l1_key(self, key: ObjectKey) -> bool:
        """Whether ``key`` currently has an L1 catalog entry.

        Pure introspection for diagnostics and tests: reports presence
        regardless of lock state and does not lock, touch, or
        otherwise perturb the entry.

        Args:
            key (ObjectKey): The key to probe.

        Returns:
            bool: True if the key has an L1 entry (any lock state).
        """
        return self._l1_manager.get_object_state(key) is not None

    def unsafe_read(
        self, keys: list[ObjectKey]
    ) -> tuple[list[ObjectKey], list[MemoryObj]]:
        """Read already read-locked objects without acquiring new read locks."""
        read_results = self._l1_manager.unsafe_read(keys)
        good_keys: list[ObjectKey] = []
        good_objs: list[MemoryObj] = []
        for key in keys:
            err, obj = read_results.get(key, (L1Error.KEY_NOT_EXIST, None))
            if err != L1Error.SUCCESS or obj is None:
                continue
            good_keys.append(key)
            good_objs.append(obj)
        return good_keys, good_objs

    @property
    def quota_manager(self) -> QuotaManager:
        """Per-cache_salt quota registry.

        Exposed so the HTTP layer can serve CRUD endpoints without
        reaching into private state. Always non-``None`` — the
        storage manager creates the registry at construction time.
        """
        return self._quota_manager

    @property
    def l1_memory_desc(self) -> L1MemoryDesc:
        """Descriptor of the L1 memory buffer backing this storage manager."""
        return self._l1_memory_desc

    def get_l2_usages(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter L2 usage in OTel-observation shape.

        Backing data for the ``lmcache_mp.l2_usage_bytes`` observable
        gauge.  One entry per configured adapter.

        Returns:
            A list of ``(total_bytes_used, {"l2_name": <type_name>})``
            tuples — empty when no L2 adapters are configured.  Adapters
            whose ``get_usage()`` raises are skipped (the gauge prefers
            silence over a poison observation).
        """
        out: list[tuple[int | float, dict[str, object]]] = []
        for _adapter_id, desc, adapter in self._snapshot_adapters():
            try:
                usage = adapter.get_usage()
            except Exception:
                logger.exception(
                    "L2 adapter %s get_usage() failed; skipping in gauge",
                    desc.type_name,
                )
                continue
            out.append((int(usage.total_bytes_used), {"l2_name": desc.type_name}))
        return out

    def get_usage_bytes_by_cache_salt(self) -> dict[str, int]:
        """Aggregate ``cache_salt`` byte usage across every L2 adapter.

        Used by the HTTP quota endpoints to report ``current_usage_gb``
        alongside the configured limit. Aggregation is a simple sum:
        each adapter tracks the same salt independently so the totals
        are additive.
        """
        totals: dict[str, int] = {}
        for _adapter_id, _desc, adapter in self._snapshot_adapters():
            snap = adapter.get_usage().bytes_by_cache_salt
            for salt, used in snap.items():
                totals[salt] = totals.get(salt, 0) + used
        return totals

    # L2 APIs
    def get_l2_adapter_reconfigure_status(self) -> dict:
        """Return status for all runtime-reconfigurable L2 adapters.

        Returns:
            JSON-serializable status. If no reconfigurable adapter is configured,
            ``enabled`` is ``False`` and the adapter list is empty.
        """
        type_names = {
            adapter_id: desc.type_name
            for adapter_id, desc, _ in self._snapshot_adapters()
        }
        adapters = []
        for adapter_index, (
            l2_adapter_index,
            adapter,
        ) in enumerate(self._list_reconfigurable_l2_adapters()):
            status = dict(adapter.reconfigure_status())
            if l2_adapter_index in type_names:
                status["backend"] = type_names[l2_adapter_index]
            status["adapter_index"] = adapter_index
            status["l2_adapter_index"] = l2_adapter_index
            adapters.append(status)

        return {
            "enabled": bool(adapters),
            "num_adapters": len(adapters),
            "adapters": adapters,
        }

    def reconfigurable_l2_backends(self) -> set[str]:
        """Return the ``type_name`` of every L2 adapter that supports runtime
        reconfiguration.

        Returns:
            The set of reconfigurable adapter ``type_name`` strings (empty when
            none are reconfigurable). The ``{backend}`` path parameter the
            ``/reconfigure`` routes expect is the adapter's ``type_name``.
        """
        return {
            desc.type_name
            for _adapter_id, desc, adapter in self._snapshot_adapters()
            if self._unwrap_reconfigurable_l2_adapter(adapter) is not None
        }

    def reconfigure_l2_adapter(
        self,
        adapter_index: int,
        operation: str,
        payload: dict[str, object],
    ) -> dict:
        """Route a runtime reconfiguration request to one L2 adapter.

        Args:
            adapter_index: Zero-based reconfigurable-adapter index.
            operation: Adapter-specific operation name.
            payload: Adapter-specific operation payload.

        Returns:
            JSON-serializable operation result.
        """
        adapter = self._get_reconfigurable_l2_adapter(adapter_index)
        result = adapter.reconfigure(operation, payload)
        result["adapter_index"] = adapter_index
        return result

    def add_l2_adapter(self, config: L2AdapterConfigBase) -> int:
        """Blocking function to add a new L2 adapter at runtime. Thread-safe.

        Args:
            config: The adapter configuration.

        Returns:
            The stable id assigned to the new adapter.

        Raises:
            SplitTierConfigError: if adding this adapter would change the
                frozen placement mode (a mode cannot flip at runtime --
                the manifest / paired-eviction wiring is set up once at
                construction), mix incompatible placement/layout modes
                (this closes the P2P auto-add bypass, B1), or exceed the
                KV_SPLIT_TIER single-adapter limit.
        """
        with self._lifecycle_lock:
            # Runtime support-matrix guard (Phase 1b): re-derive the
            # placement + layout modes over the existing adapters plus
            # the candidate and reject any config that would change the
            # frozen mode or break the split-tier single-adapter rule.
            # A KV_TOGETHER peer adapter attaching to a KV_SPLIT_TIER
            # manager produces a mixed set that the derive rejects.
            with self._adapters_lock:
                candidate_configs = [
                    d.config for d in self._adapter_descriptors.values()
                ] + [config]
            try:
                candidate_placement = derive_storage_placement_mode(candidate_configs)
                derive_storage_layout_mode(candidate_configs)
            except ValueError as e:
                raise SplitTierConfigError(
                    f"runtime add_l2_adapter rejected: incompatible "
                    f"placement/layout with the existing adapters: {e}"
                ) from e
            if candidate_placement != self._storage_placement_mode:
                raise SplitTierConfigError(
                    "runtime add_l2_adapter would change the storage "
                    f"placement mode from {self._storage_placement_mode.value} "
                    f"to {candidate_placement.value}; the mode is frozen at "
                    "construction and cannot flip at runtime"
                )
            if self._storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
                raise SplitTierConfigError(
                    "KV_SPLIT_TIER (V-only) placement supports exactly one "
                    "L2 adapter; cannot add another at runtime"
                )

            adapter_id, adapter, descriptor = self._build_l2_adapter(config)
            for listener in self._registered_l2_listeners:
                adapter.register_listener(listener)
            with self._adapters_lock:
                self._l2_adapters[adapter_id] = adapter
                self._adapter_descriptors[adapter_id] = descriptor
            self._store_controller.add_adapter(adapter_id, adapter, descriptor)
            self._prefetch_controller.add_adapter(adapter_id, adapter, descriptor)
            if self._should_enable_l2_eviction(adapter, config.eviction_config):
                assert config.eviction_config is not None  # make linter happy
                self._l2_eviction_controller.add_adapter_state(
                    L2AdapterEvictionState(
                        adapter_id=adapter_id,
                        adapter=adapter,
                        eviction_config=config.eviction_config,
                    )
                )
            logger.info("Added L2 adapter %d (%s)", adapter_id, descriptor.type_name)
            return adapter_id

    def delete_l2_adapter(self, adapter_id: int, timeout: float = 30.0) -> None:
        """Blocking function to drain the L2 adapter gracefully at runtime.
        Thread-safe.

        Stops routing new stores/prefetches to the adapter, waits for its
        in-flight work to finish, removes it from the controllers, and
        closes it.

        Args:
            adapter_id: Stable id of the adapter to remove.
            timeout: Maximum seconds to wait for in-flight work to drain.

        Raises:
            ValueError: If no adapter with ``adapter_id`` is active.
            TimeoutError: If draining did not complete within ``timeout``;
                the adapter is left active (draining) so the caller can
                retry.
        """
        with self._lifecycle_lock:
            if adapter_id not in self._l2_adapters:
                raise ValueError(f"No L2 adapter with id {adapter_id}")

            deadline = time.monotonic() + timeout
            store_done = self._store_controller.request_remove_adapter(adapter_id)
            prefetch_done = self._prefetch_controller.request_remove_adapter(adapter_id)
            if not store_done.wait(timeout=max(0.0, deadline - time.monotonic())):
                raise LMCacheTimeoutError(
                    f"Timed out draining adapter {adapter_id} from store controller"
                )
            if not prefetch_done.wait(timeout=max(0.0, deadline - time.monotonic())):
                raise LMCacheTimeoutError(
                    f"Timed out draining adapter {adapter_id} from prefetch controller"
                )

            self._l2_eviction_controller.remove_adapter_state(adapter_id)
            with self._adapters_lock:
                adapter = self._l2_adapters.pop(adapter_id)
                self._adapter_descriptors.pop(adapter_id, None)
                remaining = list(self._l2_adapters.values())
            adapter.close()
            if self._storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
                # Refresh the paired-eviction adapter snapshot so the
                # just-closed adapter is never handed a V-child delete
                # (B2/B21/B37).  When the last split-tier adapter is gone,
                # sweep the manifest: its V children are unreachable, so
                # leaving COMPLETE entries would return phantom lookup
                # hits that can never load (B3).
                self._eviction_controller.set_split_tier_paired_eviction(
                    self._split_tier_manifest, remaining
                )
                if not remaining:
                    self._sweep_manifest_no_adapters()
            logger.info("Deleted L2 adapter %d", adapter_id)

    def _sweep_manifest_no_adapters(self) -> None:
        """Invalidate + drop every split-tier manifest entry after the
        last split-tier L2 adapter has been removed.

        The V children are unreachable (their adapter is closed), so no
        L2 delete is issued -- unlike :meth:`clear`'s sweep, the point
        here is purely to stop lookups from returning phantom COMPLETE
        composite hits that can never load.  The K children in L1 become
        ordinary orphans (evictable).  Each transition is generation-
        guarded so a concurrent re-store under a fresh generation is left
        untouched.  Caller holds ``_lifecycle_lock``.
        """
        for logical_key in self._split_tier_manifest.tracked_keys():
            entry = self._split_tier_manifest.lookup_entry(logical_key)
            if entry is None:
                continue
            _state, generation = entry
            if self._split_tier_manifest.mark_invalidated(logical_key, generation):
                self._split_tier_manifest.drop(logical_key, generation)

    def l2_adapters(self) -> list[tuple[AdapterDescriptor, L2AdapterInterface]]:
        """Return all active L2 adapters paired with descriptors, in
        ascending adapter-id order (== configuration order for the initial
        set, then runtime-added adapters). The list is empty when no L2 is
        configured.

        Do not cache the returned pairs — ``reconfigure_l2_adapter``,
        ``add_l2_adapter``, and ``delete_l2_adapter`` may change the set at
        runtime.
        """
        return [
            (desc, adapter) for _adapter_id, desc, adapter in self._snapshot_adapters()
        ]

    # Management APIs
    def clear(self, force: bool = False):
        """
        Clear data in the storage manager.

        Under :attr:`StoragePlacementMode.KV_SPLIT_TIER`, additionally
        sweep the split-tier manifest: any tracked logical key whose
        K child was just removed from L1 can never compose again (the
        composite requires the L1-resident K child), so its entry is
        invalidated and dropped and its V child is best-effort deleted
        from every L2 adapter. Without the sweep, cleared keys stay
        ``COMPLETE`` in the manifest -- phantom lookup hits that fail
        at load, orphaned V children on L2, and (before manifest
        generations) a permanent refusal to re-store those keys.

        With ``force=False`` the underlying L1 clear now preserves
        STORE_IN_FLIGHT K children (see :meth:`L1Manager.clear`), so an
        in-flight split-tier store survives a non-forced clear intact:
        its K child stays resident, the sweep skips its still-present
        entry, and the store completes normally.

        Restart / persistence (Option C): the split-tier manifest is
        purely in-memory and is NOT persisted across process restarts.
        A restart yields an empty manifest; any V children left on L2 by
        a prior process are stale and age out via that adapter's own
        lifecycle (they can never be composed without their L1 K child,
        which did not survive the restart). This is the documented,
        single-process, non-restart-persistent contract.

        Args:
            force: If True, clear ALL objects including locked and
                in-flight ones. This is a hard reset that can abort an
                in-flight store/prefetch (the operator explicitly wiped
                the cache); the manifest sweep still invalidates the
                affected entries so no phantom ``COMPLETE`` remains.
                If False (default), only clear objects eviction itself
                would remove -- unlocked objects that are not pinned by
                an in-flight operation -- so a live composite is never
                torn out from under a store that is still writing it.
        """
        self._l1_manager.clear(force=force)
        if self._storage_placement_mode != StoragePlacementMode.KV_SPLIT_TIER:
            return
        orphaned_v_children: list[ObjectKey] = []
        # (logical_key, generation) pairs held in DELETE_IN_FLIGHT across
        # the batched physical delete below; dropped only after it
        # completes so the reclaim fence covers the whole purge.
        pending_drops: list[tuple[ObjectKey, int]] = []
        for logical_key in self._split_tier_manifest.tracked_keys():
            entry = self._split_tier_manifest.lookup_entry(logical_key)
            if entry is None:
                # Dropped between the snapshot and here; nothing to do.
                continue
            _state, generation = entry
            k_child = derive_component_key(logical_key, "k")
            if self._l1_manager.get_object_state(k_child) is not None:
                # K child survived (locked entry under force=False, or
                # an in-flight store's write-locked child): the
                # composite is still intact -- leave the entry alone.
                continue
            # Generation-guarded compare-and-set: if a fresh store
            # reclaimed this logical key (writing a new K child) after
            # the tracked_keys snapshot, the captured generation no
            # longer matches, mark_invalidated returns False, and we
            # leave the new composite -- and its live V child -- alone.
            if not self._split_tier_manifest.mark_invalidated(logical_key, generation):
                continue
            # Fence the reclaim window BEFORE queuing the V child for the
            # physical delete.  The V key is generation-agnostic, so if we
            # dropped the entry here (reopening the key) a concurrent
            # re-store could write a fresh V child under the same key and
            # the deferred batch delete would clobber it -- a phantom
            # COMPLETE with no V on L2.  INVALIDATED -> DELETE_IN_FLIGHT is
            # refused by register_pending, so holding it across the batch
            # blocks any reclaim; the CAS also catches a reclaim that beat
            # us to this line (raises -> skip this key).
            try:
                self._split_tier_manifest.mark_delete_in_flight(
                    logical_key, generation
                )
            except ValueError:
                continue
            orphaned_v_children.append(derive_component_key(logical_key, "v"))
            pending_drops.append((logical_key, generation))
        if not orphaned_v_children:
            return
        with self._adapters_lock:
            adapters = list(self._l2_adapters.values())
        for adapter in adapters:
            try:
                adapter.delete(orphaned_v_children)
            except Exception:
                logger.exception(
                    "clear: L2 adapter %s delete raised for %d orphaned "
                    "split-tier V children",
                    type(adapter).__name__,
                    len(orphaned_v_children),
                )
        # Physical purge done; release the reclaim fence.
        for logical_key, generation in pending_drops:
            self._split_tier_manifest.drop(logical_key, generation)

    def close(self):
        """
        Close the storage manager and release all resources.
        """
        self._prefetch_controller.stop()
        self._store_controller.stop()
        self._eviction_controller.stop()
        self._l2_eviction_controller.stop()

        PeriodicEventNotifier.shutdown()

        for adapter in self._l2_adapters.values():
            adapter.close()

        self._l1_manager.close()

    def report_status(self) -> dict:
        """Return a status dict aggregating all sub-component statuses."""
        l1 = self._l1_manager.report_status()
        store = self._store_controller.report_status()
        prefetch = self._prefetch_controller.report_status()
        l1_eviction = self._eviction_controller.report_status()
        l2_eviction = self._l2_eviction_controller.report_status()
        adapters = [a.report_status() for _id, _desc, a in self._snapshot_adapters()]
        children = [l1, store, prefetch, l1_eviction, l2_eviction] + adapters
        return {
            "is_healthy": all(c["is_healthy"] for c in children),
            "l1_manager": l1,
            "store_controller": store,
            "prefetch_controller": prefetch,
            "l1_eviction_controller": l1_eviction,
            "l2_eviction_controller": l2_eviction,
            "l2_adapters": adapters,
            "num_l2_adapters": len(adapters),
        }

    def register_l2_listener(self, listener: L2AdapterListener) -> None:
        """Register a listener on all current and future L2 adapters.

        The listener is recorded so that adapters added later via
        :meth:`add_l2_adapter` receive it too.

        Args:
            listener: The listener to register.
        """
        with self._lifecycle_lock:
            self._registered_l2_listeners.append(listener)
            for adapter in self._l2_adapters.values():
                adapter.register_listener(listener)

    # Functions for debugging and testing
    def memcheck(self) -> bool:
        """
        Perform memory check for all storage tiers.

        Returns:
            True if memory is consistent, False otherwise.
        """
        return self._l1_manager.memcheck()

    def _snapshot_adapters(
        self,
    ) -> list[tuple[int, AdapterDescriptor, L2AdapterInterface]]:
        """Snapshot the active adapters under the lock, in ascending
        adapter-id order. Iterate this instead of the live dicts so a
        concurrent add/delete cannot change them mid-iteration.

        Returns:
            A list of ``(adapter_id, descriptor, adapter)`` tuples.
        """
        with self._adapters_lock:
            return [
                (adapter_id, self._adapter_descriptors[adapter_id], adapter)
                for adapter_id, adapter in sorted(self._l2_adapters.items())
            ]

    def _has_l2_adapters(self) -> bool:
        """Return whether any L2 adapter is currently active."""
        with self._adapters_lock:
            return bool(self._l2_adapters)

    def _build_l2_adapter(
        self,
        config: L2AdapterConfigBase,
    ) -> tuple[int, L2AdapterInterface, AdapterDescriptor]:
        """Create a L2 adapter instance based on the config.

        Args:
            config: The adapter configuration.

        Returns:
            A ``(adapter_id, adapter, descriptor)`` tuple. ``adapter_id`` is
            the freshly allocated stable id, ``adapter`` is the new adapter
            instance, and ``descriptor`` is its descriptor carrying that id.
        """
        adapter_id = self._next_adapter_id
        self._next_adapter_id += 1
        adapter: L2AdapterInterface = create_l2_adapter(config, self._l1_memory_desc)
        if config.serde_config is not None:
            adapter = SerdeL2AdapterWrapper(
                inner=adapter,
                serde=create_serde_processor(config.serde_config),
                l1_manager=self._l1_manager,
                placement_mode=self._storage_placement_mode,
                split_tier_manifest=self._split_tier_manifest,
            )
        descriptor = AdapterDescriptor(index=adapter_id, config=config)
        return adapter_id, adapter, descriptor

    def _should_enable_l2_eviction(
        self,
        adapter: L2AdapterInterface,
        eviction_config: EvictionConfig | None,
    ) -> bool:
        """Whether to wire an adapter into the L2 eviction controller.

        Args:
            adapter: The adapter to evaluate.
            eviction_config: The adapter's eviction config, if any.

        Returns:
            True if an eviction state should be created for this adapter.
        """
        if eviction_config is None:
            return False
        policy_name = eviction_config.eviction_policy
        if policy_name != "IsolatedLRU" and not adapter.supports_global_eviction:
            logger.warning(
                "L2 adapter %s configured with '%s' eviction but does "
                "not support global eviction (max_capacity_bytes=0); "
                "skipping aggregate-usage eviction setup.",
                type(adapter).__name__,
                policy_name,
            )
            return False
        return True

    def _unwrap_reconfigurable_l2_adapter(
        self,
        adapter: L2AdapterInterface,
    ) -> Optional[L2ReconfigurableAdapter]:
        if isinstance(adapter, L2ReconfigurableAdapter):
            return adapter

        inner = getattr(adapter, "inner_adapter", None)
        if inner is not None and isinstance(inner, L2ReconfigurableAdapter):
            return inner

        return None

    def _list_reconfigurable_l2_adapters(
        self,
    ) -> list[tuple[int, L2ReconfigurableAdapter]]:
        with self._adapters_lock:
            items = sorted(self._l2_adapters.items())
        adapters: list[tuple[int, L2ReconfigurableAdapter]] = []
        for l2_adapter_index, adapter in items:
            reconfigurable_adapter = self._unwrap_reconfigurable_l2_adapter(adapter)
            if reconfigurable_adapter is not None:
                adapters.append((l2_adapter_index, reconfigurable_adapter))
        return adapters

    def _get_reconfigurable_l2_adapter(
        self,
        adapter_index: int,
    ) -> L2ReconfigurableAdapter:
        adapters = self._list_reconfigurable_l2_adapters()
        if adapter_index < 0 or adapter_index >= len(adapters):
            raise L2ReconfigureError(404, "L2 adapter not reconfigurable")
        return adapters[adapter_index][1]
