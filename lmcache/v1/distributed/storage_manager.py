# SPDX-License-Identifier: Apache-2.0
"""
Distributed multi-tier storage manager for MP mode
"""

# Standard
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import Any, Iterator, Optional, cast
import threading
import time

# Third Party
import torch

# First Party
from lmcache.lmcache_native import PeriodicEventNotifier
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    CapacitySnapshot,
    MemoryLayoutDesc,
    ModuleMemoryCapacity,
    ObjectKey,
    PrefetchHandle,
    PrefetchResult,
    PrefetchTaskSpec,
    Tier,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    StorageManagerConfig,
    requires_single_l1_memory_region,
    unwrap_l2_adapter_config,
)
from lmcache.v1.distributed.error import L1Error, L1ReconfigureError, strerror
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2AdapterListener
from lmcache.v1.distributed.l1_manager import L1Manager, L1OperationResult
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import AdapterUsage, L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    get_type_name_for_config,
)
from lmcache.v1.distributed.l2_adapters.reconfiguration import (
    L2DeviceOwner,
    L2ReconfigurableAdapter,
    L2ReconfigureError,
)
from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.distributed.serde import (
    SerdeSizeContract,
    create_serde_processor,
    get_serde_size_contract,
)
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
    create_store_policy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.distributed.storage_controllers.write_policy import OrderedWritePolicy
from lmcache.v1.distributed.storage_layout import (
    StorageLayoutMode,
)
from lmcache.v1.distributed.storage_layout import (
    apply_layout_policy as _apply_layout_policy,
)
from lmcache.v1.distributed.storage_layout import (
    derive_storage_layout_mode,
)
from lmcache.v1.distributed.storage_placement import (
    ComponentKeyScheme,
    SplitTierConfigError,
    SplitTierManifest,
    SplitTierState,
    StoragePlacementMode,
    derive_component_key,
    derive_component_key_scheme,
    derive_storage_placement_mode,
)
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxArenaStatus,
    DevDaxRemoveMode,
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

# L1 write tag for every object reserved through this manager. Sharing one
# tag makes concurrent stores of the same key exclude each other.
_L1_WRITE_TAG = "storage_manager"

# Internal stream-callback payload. No MemoryObj or device pointer crosses msgpack.
L1WriteCompletion = list[tuple[int, list[ObjectKey]]]


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
    non_fs = [
        get_type_name_for_config(unwrap_l2_adapter_config(ac))
        for ac in adapters
        if get_type_name_for_config(unwrap_l2_adapter_config(ac)) != "fs"
    ]
    if non_fs:
        reasons.append(
            f"L2 backend ({', '.join(non_fs)}): split-tier variable-length "
            "V blobs are currently validated only with the filesystem "
            "adapter's actual-used-length load contract"
        )
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


def _reject_unsupported_raw_unit_backends(
    adapters: list[L2AdapterConfigBase],
) -> None:
    """Reject RAW_UNIT serdes on adapters without a used-length contract.

    RAW_UNIT V2 parsing rejects trailing bytes. The filesystem adapter reports
    the actual loaded length when a serializer's estimate exceeds its output;
    fixed-capacity receivers such as S3 and Valkey do not. Keep all RAW_UNIT
    formats on the filesystem path, including formats with exact estimates,
    until the other adapters' byte-through contracts are qualified.

    Args:
        adapters: Normalized L2 adapter configurations using RAW_UNIT.

    Raises:
        SplitTierConfigError: If any RAW_UNIT adapter is not filesystem-backed.
    """
    unsupported = [
        get_type_name_for_config(unwrap_l2_adapter_config(ac))
        for ac in adapters
        if get_type_name_for_config(unwrap_l2_adapter_config(ac)) != "fs"
    ]
    if unsupported:
        raise SplitTierConfigError(
            "RAW_UNIT byte-through serde requires the filesystem adapter's "
            "actual-used-length load contract; unsupported L2 backend(s): "
            + ", ".join(unsupported)
        )


def _reject_unsupported_serde_size_contracts(
    adapters: list[L2AdapterConfigBase],
) -> None:
    """Reject upper-bound serdes on backends without used-length loads.

    The serde wrapper allocates load buffers from
    ``estimate_serialized_size``. An ``UPPER_BOUND`` serializer may store fewer
    bytes than that capacity, so the backend must report the actual loaded
    length before deserialization. The filesystem adapter supplies that
    contract; the other current backends do not.

    Args:
        adapters: Normalized L2 adapter configurations.

    Raises:
        ValueError: If an upper-bound serde is paired with a backend that
            cannot restore the object's actual used length.
    """
    unsupported: list[str] = []
    for adapter in adapters:
        serde_config = adapter.serde_config
        if serde_config is None:
            continue
        if get_serde_size_contract(serde_config.type) == SerdeSizeContract.EXACT:
            continue
        backend = get_type_name_for_config(unwrap_l2_adapter_config(adapter))
        if backend != "fs":
            unsupported.append(f"{serde_config.type} on {backend}")
    if unsupported:
        raise ValueError(
            "Upper-bound serde output requires the filesystem adapter's "
            "actual-used-length load contract; unsupported pairing(s): "
            + ", ".join(unsupported)
        )


class StorageManager:
    def __init__(
        self,
        config: StorageManagerConfig,
        *,
        _l1_managers: tuple[L1Manager, ...] | None = None,
        _write_policy: OrderedWritePolicy | None = None,
    ) -> None:
        """Create the normal single-L1 service or an internal write-only harness.

        Supplied managers transfer lifecycle ownership to this instance. Multiple
        managers are an allocation foundation only: no serving reads, L2, or
        eviction. The normal configuration and CLI still construct one L1.

        Args:
            config: Existing single-L1 service settings.
            _l1_managers: Internal ordered candidates; None constructs one L1.
            _write_policy: Internal candidate order; defaults to the supplied order.
                Validated and captured once. Later policy mutations do not
                reconfigure this manager.

        Raises:
            ValueError: Candidates are empty/repeated, or multiple managers are
                combined with L2 adapters or eviction, or the policy names
                repeated or unregistered managers.
        """
        # Canonical L1 placement / lifecycle mode, derived from the
        # configured L2 adapters' serdes.  Determines whether the wrapper
        # packs K + V into one stored blob (KV_TOGETHER) or splits them
        # into separately keyed children with K retained in L1 and V
        # routed to L2 (KV_SPLIT_TIER, the V-only path).  Mixed
        # configurations are rejected.  Derived FIRST -- it is pure config
        # inspection -- so the split-tier support matrix can fail closed
        # before any hardware-backed resource (the L1 memory manager, the
        # adapters) is constructed.
        self._storage_placement_mode = derive_storage_placement_mode(
            config.l2_adapter_config.adapters
        )
        # Per-engine child-key scheme (byte-through RAW_UNIT vs scale-aware
        # COMPUTED_LEGACY), derived once from the same adapter configs and
        # frozen for this StorageManager.  Config-only inspection; a mixed
        # adapter set is rejected fail-closed before any resource is built.
        self._component_key_scheme = derive_component_key_scheme(
            config.l2_adapter_config.adapters
        )
        # Layout is also a pure config decision and may reject mixed
        # single-/multi-output adapter sets.  Derive it before constructing
        # L1 or starting controller threads so every invalid configuration
        # fails without leaking resources.
        self._storage_layout_mode = derive_storage_layout_mode(
            config.l2_adapter_config.adapters
        )

        _reject_unsupported_serde_size_contracts(config.l2_adapter_config.adapters)
        if self._component_key_scheme == ComponentKeyScheme.RAW_UNIT:
            _reject_unsupported_raw_unit_backends(config.l2_adapter_config.adapters)

        # Support matrix: the V-only split-tier path is only
        # correct for a narrow, documented configuration.  Reject the
        # unsupported combinations before any resource is built so a bad
        # config fails closed at startup instead of silently mis-serving.
        # KV_TOGETHER is unaffected (the check is gated on the mode).
        if self._storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
            _reject_unsupported_split_tier_config(config)

        if _l1_managers is not None:
            if not _l1_managers or len({m.l1_manager_id for m in _l1_managers}) != len(
                _l1_managers
            ):
                raise ValueError("L1 managers must be nonempty and distinct")
            if len(_l1_managers) > 1 and (
                config.l2_adapter_config.adapters
                or config.eviction_config.eviction_policy != "noop"
            ):
                raise ValueError(
                    "internal L1 overflow requires no L2 and noop eviction"
                )
        managers = (
            _l1_managers
            if _l1_managers is not None
            else (L1Manager(config.l1_manager_config),)
        )
        self._l1_manager = managers[0]
        self._l1_managers_by_id = {m.l1_manager_id: m for m in managers}
        policy = _write_policy or OrderedWritePolicy(tuple(self._l1_managers_by_id))
        candidates = policy.select_write_targets()
        if len(set(candidates)) != len(candidates) or any(
            owner not in self._l1_managers_by_id for owner in candidates
        ):
            if _l1_managers is None:
                self._l1_manager.close()
            raise ValueError("write policy must select distinct registered L1 managers")
        # Topology is fixed; resolve the validated order outside the write path.
        self._write_managers = tuple(
            self._l1_managers_by_id[owner] for owner in candidates
        )
        self._event_bus = get_event_bus()

        # L1 eviction controller
        self._eviction_controller = L1EvictionController(
            l1_manager=self._l1_manager,
            eviction_config=config.eviction_config,
        )
        self._eviction_controller.start()

        # V-child L1 dtype for the byte-through (RAW_UNIT) path. The
        # live-asymmetric V is already fp8 e4m3 in HBM and is stored byte-
        # through, so its L1 component group MUST be allocated as
        # float8_e4m3fn (1 byte/elem, no upcast) -- otherwise reserve_write
        # would give a bf16 V group and the byte-through serializer rejects a
        # non-fp8 V.  The scale-aware (COMPUTED_LEGACY) path keeps the packed
        # native dtype and quantizes V internally, so it gets no override.
        self._v_component_dtype: Optional[torch.dtype] = (
            torch.float8_e4m3fn
            if self._component_key_scheme == ComponentKeyScheme.RAW_UNIT
            else None
        )

        # Per-logical-key state machine for KV_SPLIT_TIER.  Always
        # constructed (cheap; empty when placement is KV_TOGETHER).
        # The wrapper queries and mutates this manifest as part of
        # the V-only store / load composition.
        self._split_tier_manifest = SplitTierManifest(
            component_key_scheme=self._component_key_scheme
        )

        # L2 adapters and store controller. When an adapter config carries
        # a ``serde_config``, the adapter is wrapped with
        # ``SerdeL2AdapterWrapper`` so controllers see a plain L2 adapter
        # and serde is transparent.
        self._l1_memory_desc = self._l1_manager.get_l1_memory_desc()
        self._next_adapter_id = 0
        # Serializes L1/L2 additions and L2 adapter registration/deletion.
        # Held from ownership check through add, before allocator/device locks.
        self._lifecycle_lock = threading.Lock()
        # Keeps capacity snapshots ordered by the point at which they are
        # built. Registration can publish concurrently with runtime changes.
        self._capacity_publish_lock = threading.Lock()
        # Guards the _l2_adapters and _adapter_descriptors dicts.
        self._adapters_lock = threading.Lock()
        self._registered_l2_listeners: list[L2AdapterListener] = []
        self._l2_adapters: dict[int, L2AdapterInterface] = {}
        self._adapter_descriptors: dict[int, L2AdapterDescriptor] = {}
        for ac in config.l2_adapter_config.adapters:
            adapter_id, adapter, descriptor = self._build_l2_adapter(ac)
            self._l2_adapters[adapter_id] = adapter
            self._adapter_descriptors[adapter_id] = descriptor

        PeriodicEventNotifier.create(
            interval_ms=config.periodic_notifier_interval_ms,
            use_eventfd=HAS_EVENTFD,
        )

        # Paired-eviction wire-up: attach the manifest + L2 adapters to the L1
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
                self._component_key_scheme,
            )
            # Also wire the manifest into L1Manager so its
            # ``is_key_evictable`` gates K-child eviction on manifest
            # state — STORE_IN_FLIGHT K-children stay pinned across
            # the V codec + L2 write window.
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
            l1_managers=[self._l1_manager],
            l1_manager_descriptors=[
                L1ManagerDescriptor(index=0, config=config.l1_manager_config)
            ],
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

        # Split-tier manifest state gauge — one observation per
        # SplitTierState, tagged by ``state``.  Registered only under
        # KV_SPLIT_TIER (the manifest is inert / empty otherwise).  The
        # event-driven ``lmcache_mp.split_tier_store_*`` counters report
        # the store lifecycle rate; this gauge reports the live
        # state-machine distribution (a stuck-transition / leak signal).
        if self._storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
            register_gauge(
                "lmcache.split_tier",
                "lmcache_mp.split_tier_manifest_entries",
                (
                    "Number of split-tier logical keys tracked in the "
                    "manifest, tagged by ``state`` (store_in_flight | "
                    "complete | invalidated | delete_in_flight). One "
                    "observation per state."
                ),
                self.get_split_tier_state_counts,
            )

    # External APIs for serving engine integration code to call

    @property
    def split_tier_manifest(self) -> SplitTierManifest:
        """The per-logical-key state machine for KV_SPLIT_TIER mode.

        Always present (empty when placement is :attr:`KV_TOGETHER`).
        The wrapper drives transitions during store / load; the
        eviction controller drives invalidation + delete-in-flight
        transitions during paired cleanup.

        Exposed as a property so the wrapper can hold a reference
        without needing the whole StorageManager.
        """
        return self._split_tier_manifest

    @property
    def storage_placement_mode(self) -> StoragePlacementMode:
        """The canonical placement / lifecycle mode for this
        StorageManager.

        :attr:`StoragePlacementMode.KV_TOGETHER` packs K + V into one
        stored blob (the default for ``fp8`` and asymmetric K16/V8).
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
        serdes.  Callers hand the transfer-side packed layout straight to
        :meth:`reserve_write` and :meth:`submit_prefetch_task`, which
        apply the policy once at their choke point; integration layers
        (vLLM / SGLang connectors, MP server) do not pre-apply it.
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

        For the byte-through (RAW_UNIT) path the transfer side may present
        either form:

        * A **packed** leading-``2`` group (e.g. the packed byte-through
          serde tests): it is split and the V component group is
          overridden to ``float8_e4m3fn`` (``self._v_component_dtype``). The
          already-FP8 V is stored byte-through with no upcast,
          so its L1 group must be fp8, not the packed native dtype.
        * **Pre-split** leading-``1`` K (bf16) and V (fp8) component
          groups (the transfer path registers V as a separate FP8 plane):
          the split already happened upstream, so
          the layout is validated and passed through unchanged, with no
          re-split and no dtype override.

        Which form applies is classified structurally and gated on this
        StorageManager's ``component_key_scheme`` (pre-split requires
        ``RAW_UNIT``); the pass-through validates K/V dtype and canonical
        order so a K/V order inversion fails closed.  Scale-aware
        (``COMPUTED_LEGACY``) serdes get no override and must receive
        packed input.

        With no dtype override, splitting preserves total bytes.  RAW_UNIT's
        FP8 V override intentionally reduces the V plane to one byte per
        element; the transfer registration must describe the same asymmetric
        K/V byte layout.  Pre-split input is passed through byte-for-byte.

        Args:
            layout_desc: Transfer-side layout -- a packed leading-``2``
                group, or pre-split leading-``1`` K/V component groups.

        Returns:
            Canonical layout for inspection or direct L1 use. Do not pass it
            back to :meth:`reserve_write`, which applies this transform itself.
        """
        return _apply_layout_policy(
            layout_desc,
            self._storage_layout_mode,
            v_dtype=self._v_component_dtype,
            scheme=self._component_key_scheme,
        )

    @enable_tracing()
    def reserve_write(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> dict[ObjectKey, MemoryObj]:
        """
        Reserve the object for writing into the storage manager.

        Args:
            keys (list[ObjectKey]): List of object keys to reserve for writing.
            layout_desc (MemoryLayoutDesc): Transfer-side (packed) memory
                layout for the objects to be reserved.  The L1 storage-layout
                policy is applied to it internally (see
                :meth:`apply_layout_policy`); callers pass the packed layout
                and MUST NOT pre-apply the policy (the transform is not
                idempotent).
        Returns:
            dict[ObjectKey, MemoryObj]: A dictionary mapping object keys to their
                reserved memory objects. Note that not all requested keys could be
                reserved (e.g., out of memory or write conflict)

        Only OUT_OF_MEMORY keys advance to the next candidate. Each L1 receives
        the whole pending subset; overflow does not split allocation batches.
        Candidate managers and their order are captured at construction.

        Raises:
            Exception: Allocation exceptions propagate; they are not overflow.
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
        # Apply the L1 storage-layout policy exactly once, at this choke
        # point, so every store path reserves the canonical L1 shape
        # without each caller re-deriving it.  For the default PACKED
        # layout this is a pure identity pass-through (KV_TOGETHER traffic
        # is byte-unchanged); for a multi-output serde
        # (KV_COMPONENT_GROUPS) it splits each [2, ...] group into K and V
        # component groups.  Must run BEFORE the L1 reservation so the
        # reserved object's shape matches what will be stored.  The
        # transform is not idempotent -- callers must NOT pre-apply it.
        layout_desc = self.apply_layout_policy(layout_desc)
        reserve_result: dict[ObjectKey, L1OperationResult] = {
            key: (L1Error.OUT_OF_MEMORY, None) for key in keys
        }
        pending = keys
        for manager in self._write_managers:
            if not pending:
                break
            results = manager.reserve_write(
                keys=pending,
                is_temporary=[False] * len(pending),
                layout_desc=layout_desc,
                tag=_L1_WRITE_TAG,
            )
            reserve_result.update(results)
            # Retry the failed batch subset, never individual allocations or
            # terminal conflicts. Each candidate is visited at most once.
            pending = [k for k in pending if results[k][0] == L1Error.OUT_OF_MEMORY]

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

    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """
        Finish writing the objects into the storage manager.

        Admits the objects reserved by :meth:`reserve_write`: each becomes
        visible to readers unless the key is already resident, in which case
        the reserved copy is dropped.

        Args:
            keys (list[ObjectKey]): List of object keys that have been written.

        Raises:
            ValueError: Multiple managers require captured owner/key groups via
                prepare_write_completion and finish_write_by_owner instead.
        """
        self._require_single_l1()
        self.finish_write_by_owner([(self._l1_manager.l1_manager_id, keys)])

    def prepare_write_completion(
        self, objects: dict[ObjectKey, MemoryObj]
    ) -> L1WriteCompletion:
        """Capture owner/key groups from the actual reserved objects.

        Args:
            objects: Original key/object pairs returned by reserve_write.

        Returns:
            A MessagePack-compatible batch for finish_write_by_owner.

        Raises:
            ValueError: An object has no owner or belongs to another manager set.
        """
        groups: dict[int, list[ObjectKey]] = {}
        for key, obj in objects.items():
            owner = obj.get_l1_manager()
            if owner is None or owner not in self._l1_managers_by_id:
                raise ValueError("write completion requires a registered L1 owner")
            groups.setdefault(owner, []).append(key)
        return list(groups.items())

    def finish_write_by_owner(self, completion: L1WriteCompletion) -> None:
        """Finish captured reservations without rerunning placement policy.

        Args:
            completion: Owner/key groups captured before scheduling completion.

        Raises:
            ValueError: A captured owner is no longer registered.

        The caller must wait for device writes. L1 staging, writer tags, and TTLs
        retain their existing lifetime rules; an owner tag is not a write epoch.
        Single-L1 tracing retains the replayable finish_write(keys) record.
        """
        if any(owner not in self._l1_managers_by_id for owner, _ in completion):
            raise ValueError("write completion requires a registered L1 owner")
        if is_tracing_enabled() and len(self._l1_managers_by_id) == 1:
            # Replay creates fresh manager IDs; keep its existing key-only schema
            # and emit once for both the legacy and owner-routed entry points.
            publish_call_event(
                "lmcache.v1.distributed.storage_manager.StorageManager.finish_write",
                {"keys": [key for _, keys in completion for key in keys]},
            )
        finish_result: dict[ObjectKey, L1Error] = {}
        for owner, keys in completion:
            finish_result.update(
                self._l1_managers_by_id[owner].finish_write(keys, tag=_L1_WRITE_TAG)
            )
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
        self._require_single_l1()
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
        read_locks: int = 1,
    ) -> None:
        """Finish reading prefetched objects.

        Args:
            keys: Object keys that have been read.
            read_locks: Read locks to release per key (the whole
                reservation when releasing a lookup's locks).
        """
        self._require_single_l1()
        finish_result = self._l1_manager.finish_read(keys, read_locks=read_locks)
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
        spec: PrefetchTaskSpec,
        external_request_id: str = "",
        skip_l2: bool = False,
    ) -> PrefetchHandle:
        """Prefetch objects into L1 asynchronously.

        Args:
            spec: The request (see :class:`PrefetchTaskSpec`). Each key-group
                layout in ``spec.key_groups`` is passed
                through the L1 storage-layout policy here; callers MUST NOT
                pre-apply it (not idempotent).
            external_request_id: Caller id for end-to-end log tracing.
            skip_l2: If True, serve from L1 only. The result is available
                as soon as this returns.

        Returns:
            PrefetchHandle to track the task.

        Raises:
            ValueError: Under KV_SPLIT_TIER, if a row or key carries a
                non-zero ``object_group_id``. Multi-group model topologies
                are outside the split-tier support matrix.
        """
        self._require_single_l1()
        if self._storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
            multi_group_rows = [
                row
                for row in spec.key_groups
                if row.object_group_id != 0
                or any(key.object_group_id != 0 for key in row.keys)
            ]
            if multi_group_rows:
                raise ValueError(
                    "KV_SPLIT_TIER (V-only) placement supports a single "
                    f"object group, but {len(multi_group_rows)} of "
                    f"{len(spec.key_groups)} prefetch row(s) carry "
                    "object_group_id != 0. This model topology is outside "
                    "the split-tier support matrix."
                )
        # Apply the L1 storage-layout policy once here (the prefetch choke
        # point), before the controller handles the request, so every caller
        # prefetches into the canonical L1 shape. PACKED is an identity
        # no-op; a multi-output serde (KV_COMPONENT_GROUPS) splits each
        # [2, ...] group into K/V components. Not idempotent -- callers must
        # not pre-apply.
        spec = replace(
            spec,
            key_groups=[
                replace(
                    row,
                    layout_desc=self.apply_layout_policy(row.layout_desc),
                )
                for row in spec.key_groups
            ],
        )
        prefetch_request_id = self._prefetch_controller.submit_prefetch_request(
            spec, skip_l2=skip_l2
        )
        logger.debug(
            "Prefetch request submitted: %d keys in %d groups "
            "(external_request_id=%s, prefetch_request_id=%d, skip_l2=%s)",
            len(spec.key_groups) * spec.group_size,
            len(spec.key_groups),
            external_request_id,
            prefetch_request_id,
            skip_l2,
        )
        return PrefetchHandle(
            prefetch_request_id=prefetch_request_id,
            external_request_id=external_request_id,
            total_requested_keys=len(spec.key_groups) * spec.group_size,
            submit_time=time.monotonic(),
            sliding_windows=tuple(row.sliding_window_size for row in spec.key_groups),
        )

    def query_prefetch_status(self, handle: PrefetchHandle) -> PrefetchResult | None:
        """
        Query the status of the prefetch task.

        Args:
            handle (PrefetchHandle): The handle of the prefetch task.

        Returns:
            The task's result once it has finished, None while it is still
            in progress.

        Note:
            Each result is returned once; later calls for the same handle
            return None.
        """
        if handle.prefetch_request_id == -1:
            return PrefetchResult(hit_cells=[], l1_hit_cells=[], l2_hit_cells=[])
        result = self._prefetch_controller.query_prefetch_result(
            handle.prefetch_request_id
        )
        if result is None:
            return None
        total_hits = sum(row.popcount() for row in result.hit_cells)
        if total_hits > 0:
            elapsed_ms = (time.monotonic() - handle.submit_time) * 1000
            logger.info(
                "Prefetch request completed (L1+L2): "
                "%d/%d retained keys (%d L1, %d L2) in %.1f ms "
                "(external_request_id=%s, prefetch_request_id=%d)",
                total_hits,
                handle.total_requested_keys,
                result.l1_hit_count,
                result.l2_hit_count,
                elapsed_ms,
                handle.external_request_id,
                handle.prefetch_request_id,
            )
        return result

    def wait_prefetch_status(
        self,
        handle: PrefetchHandle,
        timeout: float,
    ) -> bool:
        """
        Block until the prefetch task for ``handle`` has a result, or timeout.

        This lets a caller avoid busy-polling query_prefetch_status; the
        status itself is still retrieved via query_prefetch_status afterwards.

        Args:
            handle (PrefetchHandle): The handle of the prefetch task.
            timeout: Maximum number of seconds to wait for the result.

        Returns:
            True if a result is available within the timeout, False if the
            wait timed out.
        """
        if handle.prefetch_request_id == -1:
            return True
        return self._prefetch_controller.wait_prefetch_result(
            handle.prefetch_request_id, timeout
        )

    def touch_l1_keys(self, keys: list[ObjectKey]):
        """
        Touch the keys in L1 storage, marking the keys
        as accessed(retrieved or stored).

        Args:
            keys (list[ObjectKey]): List of object keys to touch.
        """
        self._require_single_l1()
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
        self._require_single_l1()
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

        Raises:
            ValueError: Multiple managers are configured in the write-only harness.
        """
        self._require_single_l1()
        return self._l1_manager.get_object_state(key) is not None

    def unsafe_read(
        self, keys: list[ObjectKey]
    ) -> tuple[list[ObjectKey], list[MemoryObj]]:
        """Read already read-locked objects without acquiring new read locks."""
        self._require_single_l1()
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
        self._require_single_l1()
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
        return [
            (int(usage.total_bytes_used), {"l2_name": type_name})
            for type_name, usages in self.get_l2_usages_by_type().items()
            for usage in usages
        ]

    def get_l2_usages_by_type(self) -> dict[str, list[AdapterUsage]]:
        """Per-adapter usage snapshots grouped by adapter type name.

        Backing data for usage telemetry's per-connector presence and
        occupancy reporting and for the :meth:`get_l2_usages` gauge.
        Adapters whose ``get_usage()`` raises are skipped.

        Returns:
            Mapping from adapter type name (e.g. ``"dax"``) to one usage
            snapshot per active adapter of that type; empty when no L2
            adapters are configured.
        """
        out_by_type: dict[str, list[AdapterUsage]] = {}
        for _adapter_id, desc, adapter in self._snapshot_adapters():
            try:
                usage = adapter.get_usage()
            except Exception:
                logger.exception(
                    "L2 adapter %s get_usage() failed; skipping in usage snapshot",
                    desc.type_name,
                )
                continue
            out_by_type.setdefault(desc.type_name, []).append(usage)
        return out_by_type

    def publish_capacity(self) -> None:
        """Announce the current capacity topology on the event bus.

        Called after every coordinator registration, so a restarted
        coordinator relearns this server's capacities even if nothing is
        ever reconfigured. Later changes announce themselves.

        Raises:
            ValueError: Multiple managers are configured in the write-only harness.
        """
        self._require_single_l1()
        self._publish_capacity_changed()

    def _build_capacities(self) -> list[ModuleMemoryCapacity]:
        """Assemble one capacity entry per memory compartment.

        Returns:
            L1 per backing medium, then one entry per L2 adapter.
        """
        capacities = [
            ModuleMemoryCapacity(
                tier=Tier.L1,
                backend=backend.value,
                capacity_bytes=configured,
                shared=False,
            )
            for backend, configured in (
                self._l1_manager.get_capacity_bytes_by_backend().items()
            )
        ]
        for _adapter_id, desc, adapter in self._snapshot_adapters():
            try:
                usage = adapter.get_usage()
            except Exception:
                logger.exception(
                    "L2 adapter %s get_usage() failed; omitting from the "
                    "capacity report",
                    desc.type_name,
                )
                continue
            capacities.append(
                ModuleMemoryCapacity(
                    tier=Tier.L2,
                    backend=desc.type_name,
                    capacity_bytes=int(usage.total_capacity_bytes),
                    shared=bool(desc.config.shared),
                )
            )
        return capacities

    def get_l1_usage(self) -> tuple[int, int]:
        """Current occupancy of the L1 memory pool.

        Backing data for usage telemetry's L1 occupancy reporting.

        Returns:
            Tuple of ``(used_bytes, total_bytes)``.

        Raises:
            ValueError: Multiple managers are configured in the write-only harness.
        """
        self._require_single_l1()
        return self._l1_manager.get_memory_usage()

    # L1 reconfiguration APIs
    def get_l1_devdax_arena_statuses(self) -> list[DevDaxArenaStatus]:
        """Return runtime status for every Device-DAX L1 arena.

        Returns:
            One status per mapped arena, in pool order.

        Raises:
            L1ReconfigureError: If L1 is not Device-DAX backed.
            ValueError: Multiple managers are configured in the write-only harness.
        """
        self._require_single_l1()
        return self._l1_manager.get_devdax_arena_statuses()

    def add_l1_devdax_device(
        self,
        device_path: str,
        size_in_bytes: int,
    ) -> DevDaxArenaStatus:
        """Add a Device-DAX device to the L1 arena pool.

        A successful addition publishes the current whole capacity topology.

        The addition is refused while an L2 adapter that registers a single L1
        memory region is configured.
        The compatibility checks and addition are protected by ``_lifecycle_lock``.

        Args:
            device_path: Path of the Device-DAX device to map.
            size_in_bytes: Number of bytes to map.

        Returns:
            Status of the newly added arena.

        Raises:
            L1ReconfigureError: If L1 is not Device-DAX backed, a single-region
                L2 adapter is configured (409), the physical device is already
                mapped by L2 (409), or the request cannot be applied.
            ValueError: Multiple managers are configured in the write-only harness.
        """
        self._require_single_l1()
        with self._lifecycle_lock:
            # Report the L1 backing error before adapter compatibility.
            self._l1_manager.get_devdax_arena_statuses()
            incompatible = self._single_region_adapter_names()
            if incompatible:
                raise L1ReconfigureError(
                    409,
                    "cannot add a Device-DAX L1 arena: L2 adapters that "
                    "register a single L1 memory region are configured "
                    f"({', '.join(incompatible)}); their transfers cover "
                    "only the primary arena",
                )
            device_owners = self._l2_device_owner_names(device_path)
            if device_owners:
                raise L1ReconfigureError(
                    409,
                    "cannot add a Device-DAX L1 arena: the physical device "
                    "is already mapped by L2 adapter(s) "
                    f"({', '.join(device_owners)})",
                )
            status = self._l1_manager.add_devdax_device(device_path, size_in_bytes)
        self._publish_capacity_changed()
        return status

    def remove_l1_devdax_device(
        self,
        device_path: str,
        mode: DevDaxRemoveMode = DevDaxRemoveMode.DRAIN,
    ) -> DevDaxArenaStatus:
        """Remove a Device-DAX device from the L1 arena pool.

        Whenever the call changes usable capacity, it publishes the current
        whole topology. This includes a drain transition whose later device
        cleanup raises an exception.

        Args:
            device_path: Path of the mapped Device-DAX device.
            mode: Removal strategy. Only drain mode is currently supported.

        Returns:
            Status of the arena after the removal request.

        Raises:
            L1ReconfigureError: If L1 is not Device-DAX backed or the request
                cannot be applied.
            RuntimeError: If device synchronization or cleanup fails after the
                drain transition.
            OSError: If unmapping or closing the device fails after the drain
                transition.
            ValueError: Multiple managers are configured in the write-only harness.
        """
        self._require_single_l1()
        target_was_active = self._l1_devdax_arena_is_active(device_path)
        try:
            status = self._l1_manager.remove_devdax_device(device_path, mode)
        except Exception:
            # Draining begins before an empty arena is synchronized and
            # unmapped. If that cleanup raises, usable capacity has still
            # changed and the coordinator must not retain the old topology.
            try:
                if target_was_active and not self._l1_devdax_arena_is_active(
                    device_path
                ):
                    self._publish_capacity_changed()
            except Exception:
                logger.exception(
                    "Failed to reconcile L1 capacity after a Device-DAX remove error"
                )
            raise
        self._publish_capacity_changed()
        return status

    def _single_region_adapter_names(self) -> list[str]:
        """Return type names of registered L2 adapters needing one L1 region.

        The caller holds ``_lifecycle_lock`` so the answer stays valid while
        it acts on it; ``_adapters_lock`` only guards the dict read.
        """
        with self._adapters_lock:
            descriptors = list(self._adapter_descriptors.values())
        return [
            name
            for descriptor in descriptors
            if (name := requires_single_l1_memory_region(descriptor.config)) is not None
        ]

    def _l1_devdax_arena_is_active(self, device_path: str) -> bool:
        """Return whether an ACTIVE Device-DAX arena is mapped at ``device_path``.

        The path must match the one used when adding the arena. ``False``
        when L1 is not Device-DAX backed or nothing is registered there.
        """
        try:
            status = self._l1_manager.get_devdax_arena_status(device_path)
        except L1ReconfigureError:
            return False
        return status.state is DevDaxArenaState.ACTIVE

    def get_split_tier_state_counts(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Split-tier manifest state distribution in OTel-observation shape.

        Backing data for the ``lmcache_mp.split_tier_manifest_entries``
        observable gauge.  One entry per :class:`SplitTierState`, so the
        gauge always emits the full set of series (zero for empty states).
        A rising ``store_in_flight`` / ``delete_in_flight`` count or a
        monotonically growing total is the operator's signal for a stuck
        transition or an entry leak.

        Returns:
            A list of ``(count, {"state": <state_value>})`` tuples, one per
            split-tier state.
        """
        return [
            (count, {"state": state.value})
            for state, count in self._split_tier_manifest.state_counts().items()
        ]

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

    def _publish_capacity_changed(self) -> None:
        """Announce the current capacity topology on the event bus.

        Snapshot construction and enqueue are serialized because registration
        can publish concurrently with runtime reconfiguration. The subscriber
        assigns revisions in queue order, so an older snapshot must not be
        enqueued after a newer one. The locks used by ``_build_capacities`` to
        protect its reads are still required; this lock only orders declarations.

        The event carries the whole topology, not a delta, so a later
        publication can repair a dropped declaration.
        """
        with self._capacity_publish_lock:
            snapshot = CapacitySnapshot(modules=tuple(self._build_capacities()))
            self._event_bus.publish(
                Event(
                    event_type=EventType.SM_CAPACITY_CHANGED,
                    metadata={"snapshot": snapshot},
                )
            )

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
        with self._lifecycle_lock if operation == "add" else nullcontext():
            adapter = self._get_reconfigurable_l2_adapter(adapter_index)
            result = adapter.reconfigure(
                operation,
                payload,
                device_owners=lambda path: self._device_owner_names(path, adapter),
            )
        result["adapter_index"] = adapter_index
        self._publish_capacity_changed()
        return result

    def add_l2_adapter(self, config: L2AdapterConfigBase) -> int:
        """Blocking function to add a new L2 adapter at runtime. Thread-safe.

        Args:
            config: The adapter configuration.

        Returns:
            The stable id assigned to the new adapter.

        Raises:
            ValueError: If the adapter registers a single L1 memory region
                while L1 spans more than one (hybrid DRAM + Device-DAX, or
                more than one Device-DAX arena), or a DAX device is already
                mapped by L1 or another L2 adapter.
            SplitTierConfigError: if adding this adapter would change the
                frozen placement mode (a mode cannot flip at runtime --
                the manifest / paired-eviction wiring is set up once at
                construction), mix incompatible placement/layout modes
                (this closes the P2P auto-add bypass), or exceed the
                KV_SPLIT_TIER single-adapter limit.
        """
        with self._lifecycle_lock:
            # Runtime support-matrix guard: re-derive the
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
                candidate_layout = derive_storage_layout_mode(candidate_configs)
                candidate_scheme = derive_component_key_scheme(candidate_configs)
            except ValueError as e:
                raise SplitTierConfigError(
                    f"runtime add_l2_adapter rejected: incompatible "
                    f"placement/layout/scheme with the existing adapters: {e}"
                ) from e
            if candidate_placement != self._storage_placement_mode:
                raise SplitTierConfigError(
                    "runtime add_l2_adapter would change the storage "
                    f"placement mode from {self._storage_placement_mode.value} "
                    f"to {candidate_placement.value}; the mode is frozen at "
                    "construction and cannot flip at runtime"
                )
            if candidate_layout != self._storage_layout_mode:
                raise SplitTierConfigError(
                    "runtime add_l2_adapter would change the storage "
                    f"layout mode from {self._storage_layout_mode.value} "
                    f"to {candidate_layout.value}; the mode is frozen at "
                    "construction and cannot flip at runtime"
                )
            if candidate_scheme != self._component_key_scheme:
                raise SplitTierConfigError(
                    "runtime add_l2_adapter would change the component-key "
                    f"scheme from {self._component_key_scheme.name} to "
                    f"{candidate_scheme.name}; the scheme is frozen at "
                    "construction and cannot flip at runtime"
                )
            if candidate_scheme == ComponentKeyScheme.RAW_UNIT:
                _reject_unsupported_raw_unit_backends(candidate_configs)
            _reject_unsupported_serde_size_contracts(candidate_configs)
            if self._storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
                raise SplitTierConfigError(
                    "KV_SPLIT_TIER (V-only) placement supports exactly one "
                    "L2 adapter; cannot add another at runtime"
                )

            # Mirror of the check in add_l1_devdax_device: a single-region
            # adapter may only be added while L1 is exactly one memory region.
            adapter_name = requires_single_l1_memory_region(config)
            region_count = self._l1_manager.memory_region_count()
            if adapter_name is not None and region_count > 1:
                raise ValueError(
                    f"{adapter_name} registers a single L1 memory region, but "
                    f"L1 currently spans {region_count} regions (hybrid DRAM + "
                    "Device-DAX, or more than one Device-DAX arena); remove the "
                    "additional Device-DAX regions before adding it"
                )
            # Check all DAX devices before the constructor maps any.
            device_config = unwrap_l2_adapter_config(config)
            if get_type_name_for_config(device_config) == "dax":
                for device in cast(Any, device_config).devices:
                    owners = self._device_owner_names(device.device_path)
                    if owners:
                        raise ValueError(
                            f"device {device.device_path} is already mapped by "
                            f"{', '.join(owners)}"
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
            self._publish_capacity_changed()
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
                adapter = self._l2_adapters[adapter_id]
                remaining = [
                    existing
                    for existing_id, existing in self._l2_adapters.items()
                    if existing_id != adapter_id
                ]
            if self._storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
                # Detach the adapter from future paired passes and wait for
                # passes that already snapshotted it before closing it.
                drained = self._eviction_controller.set_split_tier_paired_eviction(
                    self._split_tier_manifest,
                    remaining,
                    self._component_key_scheme,
                    timeout=max(0.0, deadline - time.monotonic()),
                )
                if not drained:
                    raise LMCacheTimeoutError(
                        f"Timed out draining adapter {adapter_id} from paired "
                        "L1 eviction"
                    )
            with self._adapters_lock:
                self._l2_adapters.pop(adapter_id)
                self._adapter_descriptors.pop(adapter_id, None)
            adapter.close()
            if self._storage_placement_mode == StoragePlacementMode.KV_SPLIT_TIER:
                # When the last split-tier adapter is gone,
                # sweep the manifest: its V children are unreachable, so
                # leaving COMPLETE entries would return phantom lookup
                # hits that can never load.
                if not remaining:
                    self._sweep_manifest_no_adapters()
            logger.info("Deleted L2 adapter %d", adapter_id)
            self._publish_capacity_changed()

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

    def l2_adapters(self) -> list[tuple[L2AdapterDescriptor, L2AdapterInterface]]:
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

        Serialize the complete sweep with adapter reconfiguration and shutdown
        so each physical child deletion retains a live adapter until it returns.

        Under :attr:`StoragePlacementMode.KV_SPLIT_TIER`, additionally
        sweep the split-tier manifest: any tracked logical key whose
        K child was just removed from L1 can never compose again (the
        composite requires the L1-resident K child), so its entry is
        invalidated and its V child is deleted from every L2 adapter before
        the generation fence is dropped. Without the sweep, cleared keys stay
        ``COMPLETE`` in the manifest -- phantom lookup hits that fail
        at load and orphaned V children on L2.

        With ``force=False`` the underlying L1 clear now preserves
        STORE_IN_FLIGHT K children (see :meth:`L1Manager.clear`), so an
        in-flight split-tier store survives a non-forced clear intact:
        its K child stays resident, the sweep skips its still-present
        entry, and the store completes normally.

        Restart / persistence: the split-tier manifest is
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
        with self._lifecycle_lock:
            for manager in self._l1_managers_by_id.values():
                manager.clear(force=force)
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
                k_child = derive_component_key(
                    logical_key, "k", scheme=self._component_key_scheme
                )
                if self._l1_manager.has_object_or_staging(k_child):
                    # K child survived as a resident or a write-reserved staging
                    # object (locked under force=False / still in flight): the
                    # composite is still intact -- leave the entry alone.
                    continue
                # Atomically claim physical cleanup before queuing the
                # generation-blind V-child key. A replacement cannot register
                # until the delete finishes and this fence is dropped.
                previous = self._split_tier_manifest.begin_physical_cleanup(
                    logical_key,
                    generation,
                    (
                        SplitTierState.STORE_IN_FLIGHT,
                        SplitTierState.COMPLETE,
                        SplitTierState.INVALIDATED,
                    ),
                )
                if previous is None:
                    continue
                orphaned_v_children.append(
                    derive_component_key(
                        logical_key, "v", scheme=self._component_key_scheme
                    )
                )
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
            # Every physical delete attempt is terminal; release the reclaim
            # fence. A failed unlink may leave cold-tier waste, but cannot execute
            # later against a replacement generation.
            for logical_key, generation in pending_drops:
                self._split_tier_manifest.drop(logical_key, generation)
            self._event_bus.publish(
                Event(
                    event_type=EventType.SPLIT_TIER_V_CHILD_DELETED,
                    metadata={
                        "count": len(orphaned_v_children),
                        "trigger": "clear",
                    },
                )
            )

    def close(self):
        """
        Close the storage manager and release all resources.
        """
        with self._lifecycle_lock:
            self._prefetch_controller.stop()
            self._store_controller.stop()
            self._eviction_controller.stop()
            self._l2_eviction_controller.stop()

            PeriodicEventNotifier.shutdown()

            for adapter in self._l2_adapters.values():
                adapter.close()

            for manager in self._l1_managers_by_id.values():
                manager.close()

    def report_status(self) -> dict:
        """Return the single-L1 service's sub-component status snapshot.

        Returns:
            Overall is_healthy, statuses for l1_manager, store_controller,
            prefetch_controller, l1_eviction_controller, l2_eviction_controller,
            and the l2_adapters list with its num_l2_adapters count.

        Raises:
            ValueError: Multiple managers are configured in the write-only harness.
        """
        self._require_single_l1()
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
        Check the single L1 manager's memory consistency.

        Returns:
            True if memory is consistent, False otherwise.

        Raises:
            ValueError: Multiple managers are configured in the write-only harness.
        """
        self._require_single_l1()
        return self._l1_manager.memcheck()

    def _require_single_l1(self) -> None:
        """Reject serving operations on the internal overflow-only harness."""
        if len(self._l1_managers_by_id) != 1:
            raise ValueError("internal L1 overflow supports owner-routed writes only")

    def _snapshot_adapters(
        self,
    ) -> list[tuple[int, L2AdapterDescriptor, L2AdapterInterface]]:
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
    ) -> tuple[int, L2AdapterInterface, L2AdapterDescriptor]:
        """Create a L2 adapter instance based on the config.

        Args:
            config: The adapter configuration.

        Returns:
            A ``(adapter_id, adapter, descriptor)`` tuple. ``adapter_id`` is
            the freshly allocated stable id, ``adapter`` is the new adapter
            instance, and ``descriptor`` is its descriptor carrying that id.
        """
        self._require_single_l1()
        adapter_id = self._next_adapter_id
        self._next_adapter_id += 1
        adapter: L2AdapterInterface = create_l2_adapter(config, self._l1_memory_desc)
        if config.serde_config is not None:
            adapter = SerdeL2AdapterWrapper(
                inner=adapter,
                serde=create_serde_processor(config.serde_config),
                l1_manager=self._l1_manager,
                placement_mode=self._storage_placement_mode,
                component_key_scheme=self._component_key_scheme,
                split_tier_manifest=self._split_tier_manifest,
            )
        descriptor = L2AdapterDescriptor(index=adapter_id, config=config)
        # Stamp the registered type name so the adapter's cache events on
        # the observability bus carry their backend identity.
        adapter.set_backend_identity(descriptor.type_name, shared=config.shared)
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

    def _device_owner_names(
        self, device_path: str, exclude: Optional[L2ReconfigurableAdapter] = None
    ) -> list[str]:
        """Return other L1/L2 owners while the caller holds the lifecycle lock."""
        owners = ["L1"] if self._l1_manager.owns_device(device_path) else []
        return owners + self._l2_device_owner_names(device_path, exclude)

    def _l2_device_owner_names(
        self, device_path: str, exclude: Optional[L2ReconfigurableAdapter] = None
    ) -> list[str]:
        """Return L2 type names that own the physical device at a path.

        The caller holds ``_lifecycle_lock`` so registered adapters cannot be
        added or deleted between this check and the mapping attempt.

        Args:
            device_path: Candidate Device-DAX path.
            exclude: L2 adapter whose own mappings are ignored.

        Returns:
            Registered adapter type names whose open device has the same
            physical identity.
        """
        owners: list[str] = []
        for _adapter_id, descriptor, adapter in self._snapshot_adapters():
            owner = self._unwrap_reconfigurable_l2_adapter(adapter)
            if (
                owner is not exclude
                and isinstance(owner, L2DeviceOwner)
                and owner.owns_device(device_path)
            ):
                owners.append(descriptor.type_name)
        return owners

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
