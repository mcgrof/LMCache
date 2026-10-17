# SPDX-License-Identifier: Apache-2.0
"""
LMCacheManager: A unified manager for LMCache internal components.

This module provides a clean interface to manage LMCache components lifecycle,
decoupling the vLLM adapter from internal LMCache implementation details.
"""

# Standard
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    NoReturn,
    Optional,
    TypeVar,
    Union,
    cast,
)
import inspect
import threading
import time
import traceback
import weakref

# Third Party
import torch

# First Party
from lmcache.integration.base_service_factory import (
    BaseServiceFactory,
    ServiceResultPublisher,
)
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import (
    EngineDestroyResult,
    LMCacheEngine,
    LMCacheEngineBuilder,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.construction import (
    drain_lookup_construction_cleanup,
    restore_lookup_construction_error_cause,
    retain_lookup_construction_cleanup,
)
from lmcache.v1.health_monitor.base import HealthMonitor
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.lifecycle import RetainedCall
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer

logger = init_logger(__name__)

_FactoryResultT = TypeVar("_FactoryResultT")
_FACTORY_RESULT_UNPUBLISHED = object()
_FACTORY_RESULT_CLEANED = object()


@dataclass(frozen=True)
class _FactoryResultPublication:
    """One authoritative moving factory-result ownership state."""

    result: object
    cleanup_owner: Optional[Any]
    final: bool


class _PostInitShutdownRequested(RuntimeError):
    """Unwind a same-thread shutdown request out of post-initialization."""


class _EngineOwnershipReceipt:
    """Exact process-local ownership token for one manager and engine.

    Current ownership is represented only by membership of this exact object in
    ``LMCacheManager._engine_ownership_receipts[engine]``.  ``ever_registered``
    distinguishes a completed, therefore absent, receipt from an object that
    never reached the ownership ledger.  It is not a second current-ownership
    flag.
    """

    __slots__ = ("_engine_ref", "ever_registered")

    def __init__(self, engine: LMCacheEngine) -> None:
        self._engine_ref = weakref.ref(engine)
        self.ever_registered = False

    @property
    def engine(self) -> Optional[LMCacheEngine]:
        return self._engine_ref()


class _EngineAcquisitionPhase(Enum):
    """Whether the engine factory can already have observed an invocation."""

    PRECALL = auto()
    INVOCATION_MAY_HAVE_BEGUN = auto()


class _RetainedFactoryCallOwner(Generic[_FactoryResultT]):
    """Exactly-once retained invocation and result of one Manager factory."""

    def __init__(
        self,
        factory_call: Callable[[ServiceResultPublisher], Optional[_FactoryResultT]],
        thread_name: str,
    ) -> None:
        self._factory_call = factory_call
        try:
            parameters = inspect.signature(factory_call).parameters.values()
        except (TypeError, ValueError):
            # Signature opacity is common for C-backed and dynamically
            # wrapped third-party callables. The pre-publisher API was
            # zero-argument, so preserve that compatibility unless one
            # positional publisher can be proven.
            self._factory_accepts_publisher = False
        else:
            self._factory_accepts_publisher = any(
                parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
                for parameter in parameters
            )
        self._publication: object = _FACTORY_RESULT_UNPUBLISHED
        self._publication_lock = threading.RLock()
        self._call = RetainedCall(
            self._invoke,
            thread_name=thread_name,
        )

    def _invoke(self) -> None:
        if self._factory_accepts_publisher:
            result = self._factory_call(self)
        else:
            # Compatibility for third-party factories implementing the older
            # zero-argument interface. This path intentionally has only the
            # historical caller-side publication guarantee; serial_pic_v1's
            # in-tree factory always accepts the exact publisher.
            result = self._factory_call()  # type: ignore[call-arg]
            self.publish_final(result)
        if not self.final_result_published:
            raise RuntimeError("factory returned without publishing its final result")
        if self.result is not result:
            raise RuntimeError("factory returned a different published result")

    def _replace_publication(
        self,
        expected: object,
        publication: object,
    ) -> None:
        """Commit one exact ownership transition despite one interrupt."""
        try:
            self._commit_publication(expected, publication)
        except BaseException:
            # The callee has already acquired the resource. This exact state
            # move is local and idempotent, so replay one interruption instead
            # of letting the resource return without an owner.
            self._commit_publication(expected, publication)

    def _commit_publication(
        self,
        expected: object,
        publication: object,
    ) -> None:
        """Perform one validated authoritative state assignment."""
        with self._publication_lock:
            current = self._publication
            if current is publication:
                return
            if current is not expected:
                raise RuntimeError("factory result ownership changed unexpectedly")
            self._publication = publication

    def publish_intermediate(self, result: object, cleanup_owner: Any) -> None:
        """Publish the first live resource in a service construction chain."""
        publication = _FactoryResultPublication(
            result=result,
            cleanup_owner=cleanup_owner,
            final=False,
        )
        self._replace_publication(_FACTORY_RESULT_UNPUBLISHED, publication)

    def replace_intermediate(
        self,
        expected_result: object,
        result: object,
        cleanup_owner: Any,
    ) -> None:
        """Move exact ownership from one intermediate resource to another."""
        with self._publication_lock:
            current = self._publication
            if not isinstance(current, _FactoryResultPublication):
                raise RuntimeError("factory has no intermediate result to replace")
            if current.final or current.result is not expected_result:
                raise RuntimeError("factory intermediate result identity changed")
        publication = _FactoryResultPublication(
            result=result,
            cleanup_owner=cleanup_owner,
            final=False,
        )
        self._replace_publication(current, publication)

    def publish_final(self, result: object) -> None:
        """Publish or finalize the exact result that the factory will return."""
        expected: object
        with self._publication_lock:
            current = self._publication
            if isinstance(current, _FactoryResultPublication):
                if current.result is not result:
                    raise RuntimeError("factory final result identity changed")
                if current.final:
                    return
                expected = current
            elif current is _FACTORY_RESULT_UNPUBLISHED:
                expected = current
            else:
                raise RuntimeError("factory result was already cleaned")
        publication = _FactoryResultPublication(
            result=result,
            cleanup_owner=None,
            final=True,
        )
        self._replace_publication(expected, publication)

    def __call__(self, result: object) -> None:
        """Accept legacy callback-style direct final publication."""
        self.publish_final(result)

    @property
    def intermediate_result_published(self) -> bool:
        """Whether a non-final live resource is retained by this owner."""
        with self._publication_lock:
            publication = self._publication
            return (
                isinstance(publication, _FactoryResultPublication)
                and not publication.final
            )

    def owns_intermediate(self, result: object) -> bool:
        """Return whether this owner retains the exact non-final resource."""
        with self._publication_lock:
            publication = self._publication
            return (
                isinstance(publication, _FactoryResultPublication)
                and not publication.final
                and publication.result is result
            )

    @property
    def final_result_published(self) -> bool:
        """Whether the exact outer factory return value is retained."""
        with self._publication_lock:
            publication = self._publication
            return (
                isinstance(publication, _FactoryResultPublication) and publication.final
            )

    @property
    def intermediate_cleanup_owner(self) -> Any:
        """Return the exact cleanup owner for a retained intermediate."""
        with self._publication_lock:
            publication = self._publication
            if (
                not isinstance(publication, _FactoryResultPublication)
                or publication.final
                or publication.cleanup_owner is None
            ):
                raise RuntimeError("factory has no intermediate cleanup owner")
            return publication.cleanup_owner

    def mark_intermediate_cleaned(self) -> None:
        """Retire a non-final result after its exact cleanup receipt drains."""
        with self._publication_lock:
            current = self._publication
            if current is _FACTORY_RESULT_CLEANED:
                return
            if not isinstance(current, _FactoryResultPublication) or current.final:
                raise RuntimeError("factory has no intermediate result to retire")
        self._replace_publication(current, _FACTORY_RESULT_CLEANED)

    @property
    def complete(self) -> bool:
        return self._call.complete

    @property
    def result_published(self) -> bool:
        """Whether any exact resource reached this retained owner."""
        with self._publication_lock:
            return isinstance(self._publication, _FactoryResultPublication)

    @property
    def result(self) -> Optional[_FactoryResultT]:
        with self._publication_lock:
            publication = self._publication
            if not isinstance(publication, _FactoryResultPublication):
                raise RuntimeError("factory call has no retained result")
            return cast(Optional[_FactoryResultT], publication.result)

    @property
    def last_error(self) -> Optional[BaseException]:
        return self._call.last_error

    def run(self) -> None:
        self._call.run()


class _EngineAcquisitionReceipt:
    """Prepublished recovery state around the fallible factory engine call."""

    __slots__ = (
        "call_owner",
        "engine",
        "instance_id",
        "ownership_receipt",
        "phase",
        "resolved",
    )

    def __init__(self) -> None:
        self.phase = _EngineAcquisitionPhase.PRECALL
        self.call_owner: Optional[_RetainedFactoryCallOwner[LMCacheEngine]] = None
        self.resolved = False
        self.instance_id: Optional[str] = None
        self.engine: Optional[LMCacheEngine] = None
        self.ownership_receipt: Optional[_EngineOwnershipReceipt] = None


class _FactoryResultAcquisitionReceipt(Generic[_FactoryResultT]):
    """Prepublished recovery state for one optional factory result."""

    __slots__ = ("call_owner", "phase", "resolved", "result")

    def __init__(self) -> None:
        self.phase = _EngineAcquisitionPhase.PRECALL
        self.call_owner: Optional[_RetainedFactoryCallOwner[_FactoryResultT]] = None
        self.resolved = False
        self.result: Optional[_FactoryResultT] = None


class _ConstructionHandoffPhase(Enum):
    """Lifecycle of one exact successful-construction handoff."""

    PREPARED = auto()
    PROVISIONAL = auto()
    RELEASED = auto()
    FAILED_CLEANUP = auto()


class ManagerConstructionHandoffReceipt:
    """Opaque token separating provisional reachability from failed cleanup.

    A receipt is bound to one exact Manager object and can be released or
    promoted only for that object. Object identity, rather than a reusable
    integer generation, makes stale release attempts ABA-safe.
    """

    __slots__ = ("_manager_ref", "_phase")

    def __init__(self, manager: "LMCacheManager") -> None:
        self._manager_ref = weakref.ref(manager)
        self._phase = _ConstructionHandoffPhase.PREPARED

    @property
    def phase(self) -> str:
        """Return a stable diagnostic name for the receipt state."""
        return self._phase.name


class _EngineOwnerCountView:
    """Read-mostly compatibility view derived from exact receipt membership."""

    def __init__(
        self,
        registry: weakref.WeakKeyDictionary[
            LMCacheEngine,
            set[_EngineOwnershipReceipt],
        ],
        lock: threading.RLock,
    ) -> None:
        self._registry = registry
        self._lock = lock

    def __contains__(self, engine: object) -> bool:
        with self._lock:
            owners = self._registry.get(cast(LMCacheEngine, engine))
            return bool(owners)

    def __getitem__(self, engine: LMCacheEngine) -> int:
        with self._lock:
            owners = self._registry.get(engine)
            if not owners:
                raise KeyError(engine)
            return len(owners)

    def get(
        self, engine: LMCacheEngine, default: Optional[int] = None
    ) -> Optional[int]:
        with self._lock:
            owners = self._registry.get(engine)
            return len(owners) if owners else default


class LMCacheManager:
    """
    LMCacheManager manages the lifecycle of LMCache internal components.

    For an integration to utilize the Manager, define a ServiceFactory
    that determines which components to create for which workers.

    This class encapsulates the initialization and shutdown of:
    - LMCacheEngine
    - LookupClient / LookupServer
    - OffloadServer
    - InternalAPIServer
    - RuntimePluginLauncher
    - HealthMonitor
    """

    # A scheduler-bypass manager and a worker manager can receive the same
    # process-local engine object. Builder.destroy() is process-global, so the
    # last manager that owns that exact object is responsible for destroying
    # it. Holding this lock around factory acquisition and final destruction
    # keeps get/register and last-owner destroy atomic with respect to other
    # managers in this process.
    _engine_ownership_lock: ClassVar[threading.RLock] = threading.RLock()
    _engine_ownership_receipts: ClassVar[
        weakref.WeakKeyDictionary[
            LMCacheEngine,
            set[_EngineOwnershipReceipt],
        ]
    ] = weakref.WeakKeyDictionary()
    # Existing tests and diagnostics read this private name.  Keep it as a view
    # computed from receipt membership; it is never lifecycle authority.
    _engine_owner_counts: ClassVar[_EngineOwnerCountView] = _EngineOwnerCountView(
        _engine_ownership_receipts,
        _engine_ownership_lock,
    )
    _incomplete_construction_lock: ClassVar[threading.Lock] = threading.Lock()
    _incomplete_construction_owners: ClassVar[set["LMCacheManager"]] = set()
    # Successful return-publication handoffs need a process-strong reference,
    # but they must never be eligible for public failed-construction cleanup.
    # Exact receipt keys also prevent one stale release from touching a newer
    # handoff of the same Manager.
    _construction_handoff_owners: ClassVar[
        dict[ManagerConstructionHandoffReceipt, "LMCacheManager"]
    ] = {}

    def __init__(
        self,
        config: LMCacheEngineConfig,
        service_factory: BaseServiceFactory,
        connector: Optional[Any] = None,
        construction_result_publisher: Optional[
            Callable[["LMCacheManager"], None]
        ] = None,
    ) -> None:
        """
        Initialize LMCacheManager.

        Args:
            config: LMCache engine configuration.
            service_factory: Factory for creating service components.
            connector: Optional integration connector exposed to the internal
                API server.

        Raises:
            BaseException: A process-level construction failure. Every
                published owner receives terminal cleanup first; an incomplete
                cleanup remains available through
                :meth:`retry_incomplete_construction_cleanup`, and its exact
                failure is chained as the cause.
        """
        self._config = config
        self._service_factory = service_factory
        self._connector: Any = connector

        # Serialize the dynamic lookup lifecycle with terminal shutdown.  The
        # internal API can issue close/create/recreate requests concurrently,
        # and an unprotected create must never overwrite a still-live cleanup
        # owner. RLock lets recreate use the public close/create helpers while
        # keeping the pair atomic.
        self._lookup_lifecycle_lock = threading.RLock()
        self._stop_services_lock = threading.Lock()
        self._post_init_owner_thread_id: Optional[int] = None
        self._post_init_completed = False
        self._post_init_shutdown_deferred = False
        self._stop_services_owner: Optional[tuple[int, object]] = None
        self._shutdown_requested = False
        self._engine_destroyed = False
        self._engine_ownership_receipt: Optional[_EngineOwnershipReceipt] = None
        self._services_stopped = False
        self._last_stop_errors: tuple[BaseException, ...] = ()

        # Flag to track if initialization failed
        self._init_failed = False
        self._init_failed_reason: str = ""

        self._health_monitor: Optional[HealthMonitor] = None
        self._lmcache_engine_metadata: Optional[LMCacheMetadata] = None
        self._lmcache_engine: Optional[LMCacheEngine] = None
        self._engine_acquisition_receipt = _EngineAcquisitionReceipt()
        self._lookup_client_acquisition_receipt = _FactoryResultAcquisitionReceipt[
            LookupClientInterface
        ]()
        self._lookup_server_acquisition_receipt = _FactoryResultAcquisitionReceipt[
            Any
        ]()
        self._lookup_client: Optional[LookupClientInterface] = None
        self._lookup_server: Optional[
            Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]
        ] = None
        self._offload_server: Optional[ZMQOffloadServer] = None
        self._runtime_plugin_launcher: Optional[RuntimePluginLauncher] = None
        self._api_server: Optional[InternalAPIServer] = None

        # Initialize components via service factory
        try:
            try:
                self._lmcache_engine_metadata = service_factory.get_or_create_metadata()
                with type(self)._engine_ownership_lock:
                    acquisition = self._engine_acquisition_receipt
                    instance_id = service_factory.get_engine_instance_id()
                    acquisition.instance_id = instance_id
                    acquisition.call_owner = _RetainedFactoryCallOwner(
                        service_factory.get_or_create_lmcache_engine,
                        thread_name=f"lmcache-engine-factory-{instance_id}",
                    )
                    engine = self._invoke_engine_factory_locked()
                    self._publish_engine_acquisition_locked(engine)
            except BaseException as acquisition_error:
                try:
                    with type(self)._engine_ownership_lock:
                        self._recover_engine_acquisition_locked()
                except BaseException as recovery_error:
                    if isinstance(
                        acquisition_error,
                        Exception,
                    ) and not isinstance(recovery_error, Exception):
                        # A process-control interrupt raised while recovering
                        # an ordinary factory error must not be demoted into
                        # the degraded Exception path below.
                        raise recovery_error from acquisition_error
                    raise acquisition_error from recovery_error
                raise
            self._acquire_lookup_client(service_factory)
            self._acquire_lookup_server(service_factory)
            self._offload_server = service_factory.maybe_create_offload_server()
            self._runtime_plugin_launcher = (
                service_factory.maybe_create_runtime_plugin_launcher()
            )
            self._api_server = service_factory.maybe_create_internal_api_server(
                lmcache_manager=self
            )
        except Exception as e:
            self._init_failed = True
            self._init_failed_reason = "".join(traceback.format_exception(e))
            logger.error(
                "Failed to initialize LMCacheManager components: %s. "
                "System will operate in degraded mode (recompute).",
                self._init_failed_reason,
            )
        except BaseException as init_error:
            self._raise_after_construction_cleanup(init_error)
        if construction_result_publisher is not None:
            try:
                construction_result_publisher(self)
            except BaseException as publication_error:
                # Publication is part of successful construction, not a
                # caller-side epilogue. If it cannot commit, this constructor
                # still owns and must unwind the complete Manager graph.
                self._raise_after_construction_cleanup(publication_error)

    def _raise_after_construction_cleanup(
        self,
        init_error: BaseException,
    ) -> NoReturn:
        """Unwind every Manager owner, then re-raise the exact primary error."""
        cleanup_error: Optional[BaseException] = None
        try:
            self.stop_services()
        except BaseException as error:
            cleanup_error = error

        if self.services_stopped:
            type(self)._discard_incomplete_construction_owner(self)
        else:
            # Publish the strong retry owner only after construction has
            # stopped mutating the object. Exposing an in-progress manager
            # would let a concurrent retry tear it down before a later
            # factory method publishes another resource.
            self.retain_construction_cleanup()
            if cleanup_error is None and self._last_stop_errors:
                cleanup_error = self._last_stop_errors[0]
            if cleanup_error is None:
                cleanup_error = RuntimeError(
                    "LMCache manager construction cleanup is incomplete; "
                    "retry ownership is retained by "
                    "LMCacheManager.retry_incomplete_construction_cleanup()"
                )

        if cleanup_error is not None:
            if cleanup_error is init_error:
                # Recovery can surface the exact sticky construction error
                # again after updating its cleanup cause. Preserve identity
                # without creating a self-cause cycle.
                raise init_error
            raise init_error from cleanup_error
        raise init_error

    def _invoke_engine_factory_locked(self) -> Optional[LMCacheEngine]:
        """Publish possible invocation before entering the factory call.

        The caller holds ``_engine_ownership_lock``. An interruption before
        this helper starts leaves the receipt in ``PRECALL`` and therefore
        proves that no engine can belong to this manager. Once the phase is
        published, recovery resumes the exact retained call even if interruption
        lands immediately before the Python call expression executes. Builder
        recovery is needed only when that retained factory attempt itself fails.
        """
        acquisition = self._engine_acquisition_receipt
        if acquisition.phase is not _EngineAcquisitionPhase.PRECALL:
            raise RuntimeError("engine factory invocation was already admitted")
        owner = acquisition.call_owner
        if owner is None:
            raise RuntimeError("engine factory invocation has no retained owner")
        acquisition.phase = _EngineAcquisitionPhase.INVOCATION_MAY_HAVE_BEGUN
        owner.run()
        return owner.result

    def _publish_engine_acquisition_locked(
        self,
        engine: Optional[LMCacheEngine],
    ) -> None:
        """Publish a recovered factory result and its exact owner token.

        The caller holds ``_engine_ownership_lock``.  Every step is
        idempotent, so the constructor's exception path can replay this helper
        after an interruption at any publication boundary.
        """
        acquisition = self._engine_acquisition_receipt
        if acquisition.engine is not None and acquisition.engine is not engine:
            raise RuntimeError("engine acquisition recovered another generation")

        acquisition.engine = engine
        self._lmcache_engine = engine
        if engine is None:
            acquisition.resolved = True
            return

        receipt = acquisition.ownership_receipt
        if receipt is None:
            receipt = _EngineOwnershipReceipt(engine)
            acquisition.ownership_receipt = receipt
        elif receipt.engine is not engine:
            raise RuntimeError("engine acquisition receipt changed generation")

        self._engine_ownership_receipt = receipt
        owner_receipts = type(self)._engine_ownership_receipts.setdefault(
            engine,
            set(),
        )
        owner_receipts.add(receipt)
        receipt.ever_registered = True
        acquisition.resolved = True

    def _recover_engine_acquisition_locked(self) -> None:
        """Recover a Builder result lost between factory return and adoption."""
        acquisition = self._engine_acquisition_receipt
        if acquisition.resolved:
            return
        if acquisition.phase is _EngineAcquisitionPhase.PRECALL:
            self._publish_engine_acquisition_locked(None)
            return

        owner = acquisition.call_owner
        if owner is None:
            raise RuntimeError("engine factory invocation lost its retained owner")
        if acquisition.instance_id is None:
            raise RuntimeError(
                "engine factory may have been invoked without a recovery ID"
            )
        engine = acquisition.engine
        if engine is None:
            try:
                owner.run()
            except BaseException as call_error:
                if owner.result_published:
                    # The factory returned and published its exact result to
                    # the retained worker before a traced interruption could
                    # publish RetainedCall completion. Adopt that result;
                    # Builder recovery is only for a truly unpublished one.
                    engine = owner.result
                elif owner.last_error is not call_error:
                    # The waiting caller, not the retained factory attempt, was
                    # interrupted. Keep the exact owner unresolved for retry.
                    raise
                else:
                    engine = LMCacheEngineBuilder.recover(
                        acquisition.instance_id,
                        expected_config=self._config,
                        expected_metadata=self._lmcache_engine_metadata,
                    )
            else:
                engine = owner.result
        self._publish_engine_acquisition_locked(engine)

    def _acquire_lookup_client(
        self,
        service_factory: BaseServiceFactory,
    ) -> None:
        """Acquire and publish the exact constructor-time lookup client."""
        self._acquire_retained_factory_result(
            self._lookup_client_acquisition_receipt,
            service_factory.maybe_create_lookup_client,
            self._publish_lookup_client_acquisition,
            "lookup-client",
        )

    def _acquire_lookup_server(
        self,
        service_factory: BaseServiceFactory,
    ) -> None:
        """Acquire and publish the exact constructor-time lookup server."""
        self._acquire_retained_factory_result(
            self._lookup_server_acquisition_receipt,
            service_factory.maybe_create_lookup_server,
            self._publish_lookup_server_acquisition,
            "lookup-server",
        )

    def _acquire_retained_factory_result(
        self,
        acquisition: _FactoryResultAcquisitionReceipt[_FactoryResultT],
        factory_call: Callable[[ServiceResultPublisher], Optional[_FactoryResultT]],
        publish_result: Callable[[Optional[_FactoryResultT]], None],
        resource_name: str,
    ) -> None:
        """Run one factory with an exact result owner across caller interrupts."""
        try:
            acquisition.call_owner = _RetainedFactoryCallOwner(
                factory_call,
                thread_name=f"lmcache-{resource_name}-factory",
            )
            result = self._invoke_retained_factory_result(
                acquisition,
                resource_name,
            )
            publish_result(result)
        except BaseException as acquisition_error:
            try:
                self._recover_retained_factory_result(
                    acquisition,
                    publish_result,
                    resource_name,
                )
            except BaseException as recovery_error:
                if recovery_error is acquisition_error:
                    # A sticky factory error can carry an updated cleanup
                    # cause. Do not create a self-referential exception chain
                    # while preserving that exact primary error identity.
                    raise
                if isinstance(acquisition_error, Exception) and not isinstance(
                    recovery_error,
                    Exception,
                ):
                    raise recovery_error from acquisition_error
                raise acquisition_error from recovery_error
            raise

    def _invoke_retained_factory_result(
        self,
        acquisition: _FactoryResultAcquisitionReceipt[_FactoryResultT],
        resource_name: str,
    ) -> Optional[_FactoryResultT]:
        """Admit one retained call before its exact factory may execute."""
        if acquisition.phase is not _EngineAcquisitionPhase.PRECALL:
            raise RuntimeError(f"{resource_name} factory was already admitted")
        owner = acquisition.call_owner
        if owner is None:
            raise RuntimeError(f"{resource_name} factory has no retained owner")
        acquisition.phase = _EngineAcquisitionPhase.INVOCATION_MAY_HAVE_BEGUN
        owner.run()
        return owner.result

    def _recover_retained_factory_result(
        self,
        acquisition: _FactoryResultAcquisitionReceipt[_FactoryResultT],
        publish_result: Callable[[Optional[_FactoryResultT]], None],
        resource_name: str,
    ) -> None:
        """Recover an admitted result or prove that no result can exist."""
        if acquisition.resolved:
            return
        if acquisition.phase is _EngineAcquisitionPhase.PRECALL:
            publish_result(None)
            return

        owner = acquisition.call_owner
        if owner is None:
            raise RuntimeError(f"{resource_name} factory lost its retained owner")
        try:
            owner.run()
        except BaseException as call_error:
            if owner.final_result_published:
                # The callee marked this exact object as the outer factory
                # result before a later worker interruption. Adopt it even
                # though the retained attempt itself is sticky-failed.
                publish_result(owner.result)
                return
            if owner.last_error is not call_error:
                # The waiter, rather than the retained attempt, was
                # interrupted. Leave the exact call reachable for a retry.
                raise
            if owner.intermediate_result_published:
                # A callee can publish its live transport/service and then be
                # interrupted at RETURN before its caller stores the value.
                # Aggregate that exact prebuilt owner with any nested partial
                # cleanup. Distinct owners can be complementary, while the
                # receipt deduplicates an exact owner object by identity.
                retain_lookup_construction_cleanup(
                    call_error,
                    owner.intermediate_cleanup_owner,
                )
            # A failed lookup factory can retain a partially constructed
            # client/server, transport, raw socket/context, or request thread
            # on this exact sticky exception. Resolve the result as absent
            # only after every attached cleanup owner has quiesced. A cleanup
            # failure leaves this acquisition unresolved, which in turn
            # fences engine destruction during terminal Manager shutdown.
            try:
                cleanup_complete = drain_lookup_construction_cleanup(call_error)
            except BaseException as retained_error:
                if retained_error is not call_error:
                    raise
                cleanup_error = call_error.__cause__
                if (
                    isinstance(call_error, Exception)
                    and isinstance(cleanup_error, BaseException)
                    and not isinstance(cleanup_error, Exception)
                ):
                    # A process-control interrupt during rollback outranks an
                    # ordinary factory failure. Restore the factory's prior
                    # chain first; otherwise raising the cleanup interrupt
                    # from it would create factory<->cleanup cause cycle.
                    restore_lookup_construction_error_cause(call_error)
                    raise cleanup_error from call_error
                raise
            if not cleanup_complete:
                raise RuntimeError(
                    f"{resource_name} construction cleanup remained incomplete"
                ) from call_error
            if owner.intermediate_result_published:
                owner.mark_intermediate_cleaned()
            publish_result(None)
            return
        publish_result(owner.result)

    def _publish_lookup_client_acquisition(
        self,
        client: Optional[LookupClientInterface],
    ) -> None:
        """Idempotently publish one exact lookup-client factory result."""
        with self._lookup_lifecycle_lock:
            acquisition = self._lookup_client_acquisition_receipt
            if acquisition.resolved:
                if acquisition.result is not client:
                    raise RuntimeError("lookup-client factory result changed")
                return
            if acquisition.result is not None and acquisition.result is not client:
                raise RuntimeError("lookup-client factory result changed")
            if self._lookup_client is not None and self._lookup_client is not client:
                raise RuntimeError("another lookup client is already published")
            acquisition.result = client
            self._lookup_client = client
            acquisition.resolved = True

    def _publish_lookup_server_acquisition(self, server: Optional[Any]) -> None:
        """Idempotently publish one exact lookup-server factory result."""
        with self._lookup_lifecycle_lock:
            acquisition = self._lookup_server_acquisition_receipt
            if acquisition.resolved:
                if acquisition.result is not server:
                    raise RuntimeError("lookup-server factory result changed")
                return
            if acquisition.result is not None and acquisition.result is not server:
                raise RuntimeError("lookup-server factory result changed")
            if self._lookup_server is not None and self._lookup_server is not server:
                raise RuntimeError("another lookup server is already published")
            acquisition.result = server
            self._lookup_server = server
            acquisition.resolved = True

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance."""
        return self._lmcache_engine

    @property
    def lmcache_engine_metadata(self) -> Optional[LMCacheMetadata]:
        """Get the LMCache engine metadata."""
        return self._lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional[LookupClientInterface]:
        """Get the lookup client instance."""
        return self._lookup_client

    @property
    def lookup_server(
        self,
    ) -> Optional[Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]]:
        """Get the lookup server instance."""
        return self._lookup_server

    @property
    def offload_server(self) -> Optional[ZMQOffloadServer]:
        """Get the offload server instance."""
        return self._offload_server

    @property
    def api_server(self) -> Optional[InternalAPIServer]:
        """Get the API server instance."""
        return self._api_server

    @property
    def runtime_plugin_launcher(self) -> Optional[RuntimePluginLauncher]:
        """Get the runtime plugin launcher instance.

        Returns:
            The owned launcher, or ``None`` when this role has no launcher.
        """
        return self._runtime_plugin_launcher

    @property
    def health_monitor(self) -> Optional[HealthMonitor]:
        """Get the health monitor instance."""
        return self._health_monitor

    def retain_health_monitor(self, health_monitor: HealthMonitor) -> None:
        """Retain construction ownership of an exact health monitor.

        Factories publish the monitor here before any registry, attachment, or
        startup operation that may fail. An incomplete rollback therefore
        remains reachable by :meth:`stop_services`.

        Args:
            health_monitor: Exact monitor instance under construction.

        Raises:
            RuntimeError: A different monitor is already manager-owned.

        Returns:
            None.
        """
        retained = self._health_monitor
        if retained is not None and retained is not health_monitor:
            raise RuntimeError("a different health monitor is already retained")
        self._health_monitor = health_monitor

    def release_health_monitor(self, health_monitor: HealthMonitor) -> None:
        """Release an exact health monitor after complete rollback.

        Args:
            health_monitor: Exact monitor instance whose rollback completed.

        Returns:
            None.
        """
        if self._health_monitor is health_monitor:
            self._health_monitor = None

    @property
    def _engine_owner_registered(self) -> bool:
        """Compatibility diagnostic derived from exact receipt membership."""
        receipt = self._engine_ownership_receipt
        engine = self._lmcache_engine
        if receipt is None or engine is None:
            return False
        with type(self)._engine_ownership_lock:
            owners = type(self)._engine_ownership_receipts.get(engine)
            return owners is not None and receipt in owners

    @property
    def services_stopped(self) -> bool:
        """Return whether terminal cleanup quiesced every managed service.

        Returns:
            ``True`` only after all resources and engine ownership are
            terminally released; otherwise ``False`` so cleanup can be retried.
        """
        with self._lookup_lifecycle_lock:
            return self._services_stopped

    @property
    def terminal_cleanup_errors(self) -> tuple[BaseException, ...]:
        """Return exact errors from the most recent terminal cleanup attempt.

        Returns:
            An immutable tuple in cleanup-attempt order. A new terminal
            attempt replaces the prior tuple.
        """
        with self._lookup_lifecycle_lock:
            return self._last_stop_errors

    @classmethod
    def incomplete_construction_cleanup_count(cls) -> int:
        """Return the number of rejected constructors retaining cleanup owners.

        Returns:
            Number of managers whose construction escaped or whose degraded
            result was rejected before terminal cleanup fully quiesced its
            resources.
        """
        with cls._incomplete_construction_lock:
            owners = set(cls._incomplete_construction_owners)
            owners.update(
                owner
                for receipt, owner in cls._construction_handoff_owners.items()
                if receipt._phase is _ConstructionHandoffPhase.FAILED_CLEANUP
            )
            return len(owners)

    @classmethod
    def provisional_construction_handoff_count(cls) -> int:
        """Return strong handoffs that are deliberately hidden from retry."""
        with cls._incomplete_construction_lock:
            return sum(
                receipt._phase
                in {
                    _ConstructionHandoffPhase.PREPARED,
                    _ConstructionHandoffPhase.PROVISIONAL,
                }
                for receipt in cls._construction_handoff_owners
            )

    def new_construction_handoff_receipt(
        self,
    ) -> ManagerConstructionHandoffReceipt:
        """Create an exact unpublished receipt for a return handoff."""
        return ManagerConstructionHandoffReceipt(self)

    def retain_construction_handoff(
        self,
        receipt: ManagerConstructionHandoffReceipt,
    ) -> None:
        """Strongly retain one healthy handoff without enabling public retry.

        The caller publishes ``receipt`` in its own recovery owner before this
        method can run. Replaying registration is therefore safe even if an
        interruption lands after the registry insertion.
        """
        if receipt._manager_ref() is not self:
            raise RuntimeError("construction handoff belongs to another Manager")
        with self._lookup_lifecycle_lock:
            if self._services_stopped or self._shutdown_requested:
                raise RuntimeError("cannot hand off a terminal LMCache Manager")
            with type(self)._incomplete_construction_lock:
                if receipt._phase in {
                    _ConstructionHandoffPhase.RELEASED,
                    _ConstructionHandoffPhase.FAILED_CLEANUP,
                }:
                    raise RuntimeError("construction handoff is already terminal")
                owner = type(self)._construction_handoff_owners.get(receipt)
                if owner is not None and owner is not self:
                    raise RuntimeError("construction handoff owner changed")
                type(self)._construction_handoff_owners[receipt] = self
                receipt._phase = _ConstructionHandoffPhase.PROVISIONAL

    def release_construction_handoff(
        self,
        receipt: ManagerConstructionHandoffReceipt,
    ) -> None:
        """Release only this exact successful provisional handoff.

        A failed receipt is intentionally a no-op here: a stale success path
        must never erase failed-cleanup ownership. An adopted receipt remains
        strongly registered until Manager shutdown closes the final return-
        publication gap.
        """
        if receipt._manager_ref() is not self:
            raise RuntimeError("construction handoff belongs to another Manager")
        with self._lookup_lifecycle_lock:
            with type(self)._incomplete_construction_lock:
                owner = type(self)._construction_handoff_owners.get(receipt)
                if owner is not None and owner is not self:
                    raise RuntimeError("construction handoff owner changed")

                if receipt._phase is _ConstructionHandoffPhase.FAILED_CLEANUP:
                    # Failed cleanup owns this exact receipt until terminal
                    # stop succeeds. A stale success replay cannot erase it.
                    return
                if owner is None:
                    if receipt._phase is _ConstructionHandoffPhase.RELEASED:
                        return
                    raise RuntimeError("construction handoff was never retained")

                # Successful adoption stays in the same preallocated registry
                # slot for the Manager lifetime. This closes the final
                # constructor return gap: failure after release can still flip
                # this exact token without allocating or guessing an owner.
                receipt._phase = _ConstructionHandoffPhase.RELEASED
                if self._services_stopped:
                    type(self)._construction_handoff_owners.pop(receipt, None)

    def promote_construction_handoff_failure(
        self,
        receipt: ManagerConstructionHandoffReceipt,
    ) -> None:
        """Atomically make one failed handoff eligible for public retry.

        A retained handoff changes phase in its already-allocated strong
        registry entry. Public retry takes the same lock and filters only the
        failed phase, so it observes either a healthy provisional owner or a
        complete failure publication, never an in-between registry move.
        """
        if receipt._manager_ref() is not self:
            raise RuntimeError("construction handoff belongs to another Manager")
        # Fence future handoff admission before exposing this Manager to the
        # public failed-cleanup drain. The lock order matches healthy retain:
        # self lifecycle first, then the shared exact-token registry. Thus a
        # racing retain either publishes before the failure (and makes retry
        # defer) or observes shutdown_requested and cannot publish afterward.
        with self._lookup_lifecycle_lock:
            self._shutdown_requested = True
            with type(self)._incomplete_construction_lock:
                owner = type(self)._construction_handoff_owners.get(receipt)
                if owner is not None and owner is not self:
                    raise RuntimeError("construction handoff owner changed")
                if receipt._phase is _ConstructionHandoffPhase.FAILED_CLEANUP:
                    # Terminal stop may already have discarded the mapping. Do
                    # not resurrect a completed cleanup on a stale promotion.
                    return
                if owner is None:
                    if receipt._phase is not _ConstructionHandoffPhase.PREPARED:
                        if (
                            receipt._phase is _ConstructionHandoffPhase.RELEASED
                            and self._services_stopped
                        ):
                            return
                        raise RuntimeError("construction handoff owner was lost")
                    # Receipt publication precedes the first retain attempt. If
                    # insertion itself failed, recovery can retry that allocation
                    # here. Once present, the phase transition below allocates
                    # nothing and cannot lose the strong exact owner.
                    type(self)._construction_handoff_owners[receipt] = self
                receipt._phase = _ConstructionHandoffPhase.FAILED_CLEANUP

    def retain_construction_cleanup(self) -> None:
        """Retain a rejected manager for public construction-cleanup retry.

        This registry is failure-only. Healthy return-publication ownership
        uses exact construction-handoff receipts and is never visible to
        :meth:`retry_incomplete_construction_cleanup`. Terminal cleanup
        releases failed ownership only after :attr:`services_stopped` proves
        complete quiescence.

        Returns:
            None.
        """
        with self._lookup_lifecycle_lock:
            if self._services_stopped:
                type(self)._discard_incomplete_construction_owner(self)
                return
            self._shutdown_requested = True
            type(self)._retain_incomplete_construction_owner(self)

    @classmethod
    def retry_incomplete_construction_cleanup(cls) -> int:
        """Retry terminal cleanup for managers whose construction escaped.

        Every retained manager receives one cleanup attempt. Managers are
        released from the process-level retry registry only after their public
        ``services_stopped`` state proves complete quiescence.

        Returns:
            Number of managers that still retain retry ownership.

        Raises:
            BaseException: The first process-level cleanup interruption, after
            every retained manager has received an attempt.
        """
        with cls._incomplete_construction_lock:
            owners = set(cls._incomplete_construction_owners)
            owners.update(
                owner
                for receipt, owner in cls._construction_handoff_owners.items()
                if receipt._phase is _ConstructionHandoffPhase.FAILED_CLEANUP
            )
            # An exact Manager can briefly have an already-failed generation
            # plus a distinct healthy generation that its constructor has not
            # published yet. Do not let the failed drain tear down that shared
            # object until every non-failed strong handoff has transferred.
            healthy_handoff_owners = {
                owner
                for receipt, owner in cls._construction_handoff_owners.items()
                if receipt._phase
                in {
                    _ConstructionHandoffPhase.PREPARED,
                    _ConstructionHandoffPhase.PROVISIONAL,
                }
            }
            owners.difference_update(healthy_handoff_owners)
            owner_snapshot = tuple(owners)

        process_error: Optional[BaseException] = None
        for owner in owner_snapshot:
            try:
                owner.stop_services()
            except BaseException as error:
                logger.error(
                    "Incomplete constructor cleanup retry failed: %s",
                    error,
                )
                if not isinstance(error, Exception) and process_error is None:
                    process_error = error
            finally:
                if owner.services_stopped:
                    cls._discard_incomplete_construction_owner(owner)

        remaining = cls.incomplete_construction_cleanup_count()
        if process_error is not None:
            raise process_error
        return remaining

    @property
    def kv_caches(self) -> dict[str, torch.Tensor]:
        if self._connector is not None and hasattr(self._connector, "kv_caches"):
            return self._connector.kv_caches
        return {}

    @property
    def config(self) -> LMCacheEngineConfig:
        """Get the LMCache engine configuration."""
        return self._config

    def raise_if_init_failed(self) -> None:
        """Raise when component or post-init construction entered degraded mode.

        Ordinary integrations may intentionally use LMCache's historical
        recompute fallback. Narrow correctness profiles call this method at
        their construction boundaries so the same caught exception cannot be
        mistaken for an initialized cache path. Before raising, this method
        publishes a process-level cleanup owner; completed terminal cleanup
        removes that owner automatically.

        Returns:
            None.

        Raises:
            RuntimeError: Component or post-initialization construction failed.
        """
        if self._init_failed:
            # Ordinary LMCache integrations preserve their historical degraded
            # recompute behavior by never calling this correctness boundary.
            # Once a caller rejects the partially constructed manager, retain
            # it before raising so an interrupted or failed adapter unwind
            # cannot drop the only owner of a published resource. Retention is
            # idempotent, and the stopped-state check closes the race with a
            # concurrent terminal cleanup that completed first.
            self.retain_construction_cleanup()
            raise RuntimeError(
                f"LMCache manager initialization failed:\n{self._init_failed_reason}"
            )

    # ==================== Lifecycle Methods ====================

    def start_services(self) -> None:
        """
        Start all managed services.

        Managed services include:
        - InternalAPIServer: HTTP server exposing internal APIs for
          monitoring and management (e.g., cache stats, flush operations).
        - RuntimePluginLauncher: Launches external plugin processes defined
          in the configuration (e.g., custom telemetry, cache warming).
        """
        if self._api_server is not None:
            self._api_server.start()

        if self._runtime_plugin_launcher is not None:
            self._runtime_plugin_launcher.launch_plugins()

    def post_init(self) -> None:
        """Complete post-initialization before terminal shutdown can run.

        Returns:
            None. A completed or terminal lifecycle makes later calls no-ops.

        Raises:
            RuntimeError: The active post-initialization callback recursively
                calls this method, or deferred terminal cleanup is incomplete.
                The outer attempt remains the sole publisher.
            BaseException: A process-level post-initialization or deferred
                cleanup failure, after terminal state has been published.
        """
        # Service construction and terminal teardown are one lifecycle. If
        # shutdown wins the lock, startup must not resurrect any component. If
        # startup wins it, shutdown waits until every new owner is published
        # and can then close it normally.
        deferred_shutdown = False
        post_init_error: Optional[BaseException] = None
        owner_thread_id = threading.get_ident()
        with self._lookup_lifecycle_lock:
            if self._post_init_owner_thread_id == owner_thread_id:
                raise RuntimeError(
                    "recursive LMCacheManager.post_init() is not permitted"
                )
            if self._shutdown_requested or self._post_init_completed:
                logger.warning(
                    "Skipping post_init because the lifecycle is already terminal"
                )
                return

        with self._stop_services_lock:
            with self._lookup_lifecycle_lock:
                if self._shutdown_requested or self._post_init_completed:
                    logger.warning(
                        "Skipping post_init because the lifecycle is already terminal"
                    )
                    return
                self._post_init_owner_thread_id = owner_thread_id
            try:
                self._post_init_before_shutdown()
            except _PostInitShutdownRequested:
                logger.info(
                    "Deferring reentrant terminal shutdown until post_init unwinds"
                )
            except BaseException as error:
                post_init_error = error
            finally:
                with self._lookup_lifecycle_lock:
                    self._post_init_owner_thread_id = None
                    self._post_init_completed = True
                    deferred_shutdown = self._post_init_shutdown_deferred
                    self._post_init_shutdown_deferred = False

        if deferred_shutdown:
            try:
                self.stop_services()
                if not self.services_stopped:
                    raise RuntimeError(
                        "deferred post_init shutdown did not fully quiesce"
                    )
            except BaseException as cleanup_error:
                if post_init_error is not None:
                    raise post_init_error from cleanup_error
                raise
        if post_init_error is not None:
            raise post_init_error

    def _post_init_before_shutdown(self) -> None:
        """Run post-initialization while terminal shutdown is excluded."""
        # If initialization already failed, mark engine and return early
        if self._init_failed:
            if self._lmcache_engine is not None:
                self._lmcache_engine.mark_init_failed(self._init_failed_reason)
            logger.warning("Skipping post_init due to previous initialization failure")
            return

        if self._lmcache_engine is None:
            # Initialize health monitor for scheduler (even without engine)
            self._init_health_monitor()
            return

        try:
            # First Party
            from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
                LMCacheAsyncLookupServer,
            )

            async_lookup_server = None
            if self._config.enable_async_loading and self._lookup_server is not None:
                assert isinstance(self._lookup_server, LMCacheAsyncLookupServer)
                async_lookup_server = self._lookup_server

            self._lmcache_engine.post_init(async_lookup_server=async_lookup_server)

            # Initialize health monitor after engine post_init completes
            self._init_health_monitor()
        except _PostInitShutdownRequested:
            raise
        except Exception as e:
            self._handle_post_init_failure(e)

    def _init_health_monitor(self) -> None:
        """Initialize the health monitor via the service factory.

        Called during post_init after all components are initialized.
        """
        self._health_monitor = self._service_factory.maybe_create_health_monitor(
            lmcache_manager=self
        )

    def _handle_post_init_failure(self, e: Exception) -> None:
        """
        Handle initialization failure during post_init.

        Args:
            e: The exception that caused the failure
        """
        self._init_failed = True
        self._init_failed_reason = "".join(traceback.format_exception(e))
        if self._lmcache_engine is not None:
            self._lmcache_engine.mark_init_failed(self._init_failed_reason)
        logger.error(
            "Failed during post_init: %s. "
            "System will operate in degraded mode (recompute).",
            self._init_failed_reason,
        )

    def stop_services(self) -> None:
        """Stop all managed components gracefully.

        Returns:
            None. Same-thread callback reentry is a bounded redundant no-op;
            callers inspect :attr:`services_stopped` for truthful completion.

        Raises:
            RuntimeError: Called reentrantly by a post-initialization callback.
                The request remains sticky and cleanup runs as soon as the
                callback unwinds.
            BaseException: The first process-level resource interruption,
                after all resources receive a cleanup attempt and retry state
                is published.
        """
        owner_token = object()
        try:
            self._stop_services_impl(owner_token)
        finally:
            with self._lookup_lifecycle_lock:
                owner = self._stop_services_owner
                if owner is not None and owner[1] is owner_token:
                    self._stop_services_owner = None
            if self.services_stopped:
                type(self)._discard_incomplete_construction_owner(self)

    def _stop_services_impl(self, owner_token: object) -> None:
        """Run one serialized terminal cleanup attempt."""
        owner_thread_id = threading.get_ident()
        with self._lookup_lifecycle_lock:
            owner = self._stop_services_owner
            if owner is not None and owner[0] == owner_thread_id:
                # The outer attempt already owns terminal cleanup. A resource
                # callback may redundantly request shutdown; returning here is
                # bounded and does not mark completion early. The outer attempt
                # continues exhaustively and publishes the truthful final state.
                self._shutdown_requested = True
                logger.info("Ignoring redundant same-thread terminal shutdown request")
                return
            if self._post_init_owner_thread_id == owner_thread_id:
                self._shutdown_requested = True
                self._post_init_shutdown_deferred = True
                raise _PostInitShutdownRequested(
                    "terminal shutdown requested by a post_init callback"
                )

        # Only one terminal shutdown attempt runs at a time. This lock is
        # deliberately distinct from the dynamic lookup lifecycle lock:
        # external close callbacks (especially API-server shutdown) must never
        # run while holding the lock an in-flight API request needs to return.
        with self._stop_services_lock:
            with self._lookup_lifecycle_lock:
                if self._services_stopped:
                    logger.info("LMCacheManager services are already stopped")
                    return

                # Sticky even if terminal close must be retried. No internal-
                # API request may resurrect a client or server after this.
                self._shutdown_requested = True
                self._stop_services_owner = (owner_thread_id, owner_token)
                self._last_stop_errors = ()

            logger.info("Stopping LMCacheManager services...")
            start_time = time.time()
            errors: list[tuple[str, Union[str, BaseException]]] = []

            # Construction can be interrupted after the factory publishes a
            # Builder engine but before this manager adopts its exact owner
            # receipt. Recover that publication before terminal cleanup may
            # infer that ``None`` means this role owns no engine. A failed
            # recovery remains unresolved and therefore cannot be mistaken for
            # completed engine destruction below.
            engine_acquisition_resolved = False
            try:
                with type(self)._engine_ownership_lock:
                    self._recover_engine_acquisition_locked()
                    engine_acquisition_resolved = (
                        self._engine_acquisition_receipt.resolved
                    )
            except BaseException as error:
                logger.error("Error recovering cache engine ownership: %s", error)
                errors.append(("cache_engine_acquisition", error))

            lookup_client_acquisition_resolved = False
            try:
                self._recover_retained_factory_result(
                    self._lookup_client_acquisition_receipt,
                    self._publish_lookup_client_acquisition,
                    "lookup-client",
                )
                lookup_client_acquisition_resolved = (
                    self._lookup_client_acquisition_receipt.resolved
                )
            except BaseException as error:
                logger.error("Error recovering lookup client ownership: %s", error)
                errors.append(("lookup_client_acquisition", error))

            lookup_server_acquisition_resolved = False
            try:
                self._recover_retained_factory_result(
                    self._lookup_server_acquisition_receipt,
                    self._publish_lookup_server_acquisition,
                    "lookup-server",
                )
                lookup_server_acquisition_resolved = (
                    self._lookup_server_acquisition_receipt.resolved
                )
            except BaseException as error:
                logger.error("Error recovering lookup server ownership: %s", error)
                errors.append(("lookup_server_acquisition", error))

            def _safe_close(name: str, close_fn) -> bool:
                """Synchronously close a resource and report success."""
                try:
                    logger.info("Closing %s...", name)
                    close_fn()
                    logger.info("%s closed successfully", name)
                    return True
                except BaseException as e:
                    logger.error("Error closing %s: %s", name, e)
                    errors.append((name, e))
                    return False

            # Health checks can be inside an engine/backend call. Their stop
            # method raises unless its periodic thread has actually exited.
            health_monitor = self._health_monitor
            health_monitor_quiesced = health_monitor is None or _safe_close(
                "health_monitor", health_monitor.stop
            )
            if health_monitor is not None and health_monitor_quiesced:
                self._health_monitor = None

            offload_server = self._offload_server
            offload_server_quiesced = offload_server is None or _safe_close(
                "offload_server", offload_server.close
            )
            if offload_server is not None and offload_server_quiesced:
                self._offload_server = None

            plugin_launcher = self._runtime_plugin_launcher
            plugin_launcher_stopped = plugin_launcher is None or _safe_close(
                "runtime_plugin_launcher",
                plugin_launcher.stop_plugins,
            )
            if plugin_launcher is not None and plugin_launcher_stopped:
                self._runtime_plugin_launcher = None

            # Stop ingress before the lookup lifecycle. No lookup lock is held:
            # an API server implementation may wait for an in-flight request
            # that is itself finishing a serialized lifecycle operation.
            api_server = self._api_server
            api_server_quiesced = api_server is None or _safe_close(
                "api_server", api_server.stop
            )
            if api_server is not None and api_server_quiesced:
                self._api_server = None

            with self._lookup_lifecycle_lock:
                server = self._lookup_server
            lookup_server_quiesced = (
                lookup_server_acquisition_resolved and server is None
            )
            if lookup_server_acquisition_resolved and server is not None:
                lookup_server_quiesced = _safe_close(
                    "lookup_server",
                    server.close,
                )
            if server is not None and lookup_server_quiesced:
                with self._lookup_lifecycle_lock:
                    if self._lookup_server is server:
                        self._lookup_server = None

            # A dynamic close is deliberately non-lossy: it may fail while
            # retaining cancellation ownership. Terminal shutdown may then
            # force-quiesce timers and admitted operations because the engine
            # is destroyed immediately afterward. Never clear the pointer
            # before one of those close paths succeeds.
            with self._lookup_lifecycle_lock:
                client = self._lookup_client
            lookup_client_quiesced = (
                lookup_client_acquisition_resolved and client is None
            )
            if lookup_client_acquisition_resolved and client is not None:
                try:
                    client.close()
                    lookup_client_quiesced = True
                    logger.info("lookup_client closed successfully")
                except BaseException as close_error:
                    logger.warning(
                        "Non-lossy lookup client close did not drain during "
                        "terminal shutdown: %s; forcing quiescence",
                        close_error,
                    )
                    if not isinstance(close_error, Exception):
                        errors.append(("lookup_client", close_error))
                    try:
                        client.force_close()
                        lookup_client_quiesced = True
                        logger.info("lookup_client force-closed successfully")
                    except BaseException as force_error:
                        logger.error(
                            "Failed to force-quiesce lookup client: %s",
                            force_error,
                        )
                        errors.append(("lookup_client", force_error))
                if lookup_client_quiesced:
                    with self._lookup_lifecycle_lock:
                        if self._lookup_client is client:
                            self._lookup_client = None

            engine_users_quiesced = (
                health_monitor_quiesced
                and offload_server_quiesced
                and api_server_quiesced
                and lookup_server_quiesced
                and lookup_client_quiesced
            )
            engine_destroyed = self._engine_destroyed
            if not engine_users_quiesced:
                logger.error(
                    "Skipping LMCache engine destruction because managed "
                    "service operations did not quiesce"
                )
                errors.append(("cache_engine", "Managed service still active"))
            elif not engine_destroyed:
                if not engine_acquisition_resolved:
                    ownership_error = RuntimeError(
                        "LMCache engine acquisition is unresolved; refusing "
                        "to claim terminal destruction"
                    )
                    logger.error("Error destroying cache engine: %s", ownership_error)
                    errors.append(("cache_engine", ownership_error))
                elif self._lmcache_engine is None:
                    # Scheduler-only managers do not own a builder entry. In a
                    # co-process topology, destroying by the shared instance ID
                    # here would tear down the worker manager's live engine.
                    engine_destroyed = True
                    logger.info("No LMCache engine owned by this manager")
                elif self._engine_ownership_receipt is None:
                    ownership_error = RuntimeError(
                        "LMCache engine ownership has no exact receipt; refusing "
                        "process-global destruction"
                    )
                    logger.error("Error destroying cache engine: %s", ownership_error)
                    errors.append(("cache_engine", ownership_error))
                else:
                    try:
                        self._release_engine_owner()
                        engine_destroyed = True
                    except BaseException as e:
                        logger.error("Error destroying cache engine: %s", e)
                        errors.append(("cache_engine", e))

            # A failed quiescence/destroy remains retryable, while the sticky
            # shutdown flag continues to reject dynamic creation.
            with self._lookup_lifecycle_lock:
                if engine_destroyed:
                    self._lmcache_engine = None
                self._engine_destroyed = engine_destroyed
                self._services_stopped = (
                    engine_destroyed
                    and health_monitor_quiesced
                    and offload_server_quiesced
                    and plugin_launcher_stopped
                    and api_server_quiesced
                    and lookup_server_quiesced
                    and lookup_client_quiesced
                )
                self._last_stop_errors = tuple(
                    error for _, error in errors if isinstance(error, BaseException)
                )

            elapsed = time.time() - start_time
            if errors:
                logger.warning(
                    "Shutdown completed with %d errors in %.2fs: %s",
                    len(errors),
                    elapsed,
                    errors,
                )
            else:
                logger.info(
                    "LMCacheManager services stopped successfully in %.2fs",
                    elapsed,
                )

            # Process-level interrupts retain their normal control-flow
            # meaning, but only after every resource received a cleanup
            # attempt and retry ownership/stopped state were published.
            for _, recorded_error in errors:
                if isinstance(recorded_error, BaseException) and not isinstance(
                    recorded_error, Exception
                ):
                    raise recorded_error

    def _release_engine_owner(self) -> None:
        """Idempotently release this manager's exact engine-owner receipt."""
        engine = self._lmcache_engine
        receipt = self._engine_ownership_receipt
        if engine is None or receipt is None or receipt.engine is not engine:
            raise RuntimeError("LMCache engine ownership has no exact receipt")

        with type(self)._engine_ownership_lock:
            registry = type(self)._engine_ownership_receipts
            owners = registry.get(engine)
            if owners is None or receipt not in owners:
                if not receipt.ever_registered:
                    raise RuntimeError("LMCache engine ownership was never registered")
                if owners is not None and not owners:
                    registry.pop(engine, None)
                logger.info("LMCache engine ownership was already released")
                return

            # Membership proves registration even if an interruption landed
            # between the atomic set insertion and this diagnostic assignment.
            receipt.ever_registered = True
            owner_count = len(owners)

            if owner_count > 1:
                owners.remove(receipt)
                logger.info(
                    "Released shared LMCache engine owner; %d owner(s) remain",
                    owner_count - 1,
                )
                return

            engine_instance_id = self._service_factory.get_engine_instance_id()
            logger.info("Destroying LMCache engine: %s", engine_instance_id)
            # Keep the sole-owner registration until destroy succeeds so a
            # failed terminal attempt remains retryable.
            destroy_result = LMCacheEngineBuilder.destroy(
                engine_instance_id,
                expected_engine=engine,
            )
            if destroy_result is not EngineDestroyResult.DESTROYED:
                raise RuntimeError(
                    "Builder did not confirm destruction of the owned engine"
                )
            owners.remove(receipt)
            if not owners:
                registry.pop(engine, None)
            logger.info("LMCache engine destroyed successfully")

    # ==================== Lookup Management ====================

    def close_lookup_client(self) -> dict:
        """
        Close the current lookup client.

        Returns:
            dict: Result with old client type info.
        """
        with self._lookup_lifecycle_lock:
            old_type = self._get_lookup_type_str(self._lookup_client)
            if self._shutdown_requested:
                return {
                    "old": old_type,
                    "error": "LMCacheManager shutdown is in progress",
                }
            client = self._lookup_client
            if client is None:
                return {"old": old_type}
            if self._is_async_lookup_client(client):
                return {
                    "old": old_type,
                    "error": "dynamic lifecycle is unsupported for async "
                    "lookup clients",
                }
            try:
                client.close()
                self._lookup_client = None
                logger.info("Closed lookup client: %s", old_type)
            except Exception as e:
                logger.warning("Error closing lookup client: %s", e)
                return {
                    "old": old_type,
                    "error": f"failed to close lookup client: {e}",
                }
            return {"old": old_type}

    def create_lookup_client(self, dryrun: bool = False) -> dict:
        """
        Create a new lookup client using the current configuration.

        Args:
            dryrun: If True, only return what would be created without
                actually creating it.

        Returns:
            dict: Result with new client type info.
        """
        with self._lookup_lifecycle_lock:
            if self._shutdown_requested:
                return {"error": "LMCacheManager shutdown is in progress"}

            # First Party
            from lmcache.v1.lookup_client.factory import LookupClientFactory

            if self._lmcache_engine_metadata is None:
                return {"error": "metadata not available"}
            if self._lookup_client is not None:
                return {
                    "error": "lookup client already available: "
                    + self._get_lookup_type_str(self._lookup_client)
                }
            if self._would_create_async_lookup_client():
                return {
                    "error": "dynamic lifecycle is unsupported for async lookup clients"
                }

            try:
                client = LookupClientFactory.create_lookup_client(
                    self._config,
                    self._lmcache_engine_metadata,
                    self._lmcache_engine,
                )
            except Exception as e:
                logger.warning("Error creating lookup client: %s", e)
                return {"error": f"failed to create lookup client: {e}"}

            new_type = self._get_lookup_type_str(client)
            if dryrun:
                try:
                    client.close()
                except Exception as e:
                    logger.warning("Error closing dry-run lookup client: %s", e)
                    try:
                        client.force_close()
                    except Exception as force_error:
                        logger.error(
                            "Error force-closing dry-run lookup client: %s",
                            force_error,
                        )
                        # The dry-run object is now a real lifecycle owner. Do
                        # not lose it: publishing it makes future create calls
                        # fail and lets terminal shutdown gate engine destroy
                        # on a later successful close/force-close.
                        self._lookup_client = client
                        return {
                            "new": new_type,
                            "dryrun": True,
                            "error": "failed to close dry-run lookup client: "
                            f"{e}; force close also failed: {force_error}; "
                            "client retained for cleanup",
                        }
                    return {
                        "new": new_type,
                        "dryrun": True,
                        "error": f"failed to close dry-run lookup client: {e}",
                    }
                return {"new": new_type, "dryrun": True}

            self._lookup_client = client
            logger.info("Created lookup client: %s", new_type)
            return {"new": new_type}

    def recreate_lookup_client(self) -> dict:
        """
        Recreate the lookup client (close + create).

        Returns:
            dict: Result with old and new client type info.
        """
        with self._lookup_lifecycle_lock:
            if self._shutdown_requested:
                return {"error": "LMCacheManager shutdown is in progress"}
            if self._lookup_client is None:
                return {"error": "lookup client not available"}
            result = self.close_lookup_client()
            if "error" in result:
                return result
            create_result = self.create_lookup_client()
            result.update(create_result)
            return result

    def close_lookup_server(self) -> dict:
        """
        Close the current lookup server.

        Returns:
            dict: Result with old server type info.
        """
        with self._lookup_lifecycle_lock:
            old_type = self._get_lookup_type_str(self._lookup_server)
            if self._shutdown_requested:
                return {
                    "old": old_type,
                    "error": "LMCacheManager shutdown is in progress",
                }
            server = self._lookup_server
            if server is None:
                return {"old": old_type}
            if self._is_async_lookup_server(server):
                return {
                    "old": old_type,
                    "error": "dynamic lifecycle is unsupported for async "
                    "lookup servers",
                }
            try:
                server.close()
                self._lookup_server = None
                logger.info("Closed lookup server: %s", old_type)
            except Exception as e:
                logger.warning("Error closing lookup server: %s", e)
                return {
                    "old": old_type,
                    "error": f"failed to close lookup server: {e}",
                }
            return {"old": old_type}

    def create_lookup_server(self, dryrun: bool = False) -> dict:
        """
        Create a new lookup server using the current configuration.

        Args:
            dryrun: If True, only return what would be created without
                actually creating it.

        Returns:
            dict: Result with new server type info.
        """
        with self._lookup_lifecycle_lock:
            if self._shutdown_requested:
                return {"error": "LMCacheManager shutdown is in progress"}

            # First Party
            from lmcache.v1.lookup_client.factory import LookupClientFactory

            if self._lmcache_engine is None:
                return {"error": "engine not available"}
            if self._lmcache_engine_metadata is None:
                return {"error": "metadata not available"}
            if self._lookup_server is not None:
                return {
                    "error": "lookup server already available: "
                    + self._get_lookup_type_str(self._lookup_server)
                }
            if self._config.enable_async_loading:
                return {
                    "error": "dynamic lifecycle is unsupported for async lookup servers"
                }

            try:
                server = LookupClientFactory.create_lookup_server(
                    self._lmcache_engine,
                    self._lmcache_engine_metadata,
                )
            except Exception as e:
                logger.warning("Error creating lookup server: %s", e)
                return {"error": f"failed to create lookup server: {e}"}

            new_type = self._get_lookup_type_str(server)
            if dryrun:
                try:
                    if server is not None:
                        server.close()
                except Exception as e:
                    logger.warning("Error closing dry-run lookup server: %s", e)
                    # There is no terminal force-close hook for a lookup
                    # server. Retain the constructed object so later create
                    # and engine destroy cannot run past a live server.
                    self._lookup_server = server
                    return {
                        "new": new_type,
                        "dryrun": True,
                        "error": f"failed to close dry-run lookup server: {e}; "
                        "server retained for cleanup",
                    }
                return {"new": new_type, "dryrun": True}

            self._lookup_server = server
            logger.info("Created lookup server: %s", new_type)
            return {"new": new_type}

    def recreate_lookup_server(self) -> dict:
        """
        Recreate the lookup server (close + create).

        Returns:
            dict: Result with old and new server type info.
        """
        with self._lookup_lifecycle_lock:
            if self._shutdown_requested:
                return {"error": "LMCacheManager shutdown is in progress"}
            if self._lookup_server is None:
                return {"error": "lookup server not available"}

            result = self.close_lookup_server()
            if "error" in result:
                return result
            create_result = self.create_lookup_server()
            result.update(create_result)
            return result

    def _get_lookup_type_str(self, obj) -> str:
        """
        Get type string for lookup client/server, including wrapper info.

        Returns format: "OuterWrapper(InnerWrapper(CoreType))" or "None"
        """
        if obj is None:
            return "None"

        parts = []
        current = obj
        while True:
            parts.append(type(current).__name__)
            if hasattr(current, "actual_lookup_client"):
                current = current.actual_lookup_client
            else:
                break

        if len(parts) == 1:
            return parts[0]
        result = parts[-1]
        for wrapper in reversed(parts[:-1]):
            result = "%s(%s)" % (wrapper, result)
        return result

    @staticmethod
    def _get_core_lookup_client(client: LookupClientInterface):
        current = client
        while hasattr(current, "actual_lookup_client"):
            current = current.actual_lookup_client
        return current

    def _is_async_lookup_client(self, client: LookupClientInterface) -> bool:
        # First Party
        from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupClient,
        )

        return isinstance(
            self._get_core_lookup_client(client), LMCacheAsyncLookupClient
        )

    @staticmethod
    def _is_async_lookup_server(server) -> bool:
        # First Party
        from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupServer,
        )

        return isinstance(server, LMCacheAsyncLookupServer)

    def _would_create_async_lookup_client(self) -> bool:
        return bool(
            self._config.enable_async_loading
            and self._config.external_lookup_client is None
            and not (
                self._config.enable_scheduler_bypass_lookup
                and self._lmcache_engine is not None
            )
        )

    def get_lookup_info(self) -> dict:
        """
        Get information about the current lookup client and server.

        Returns:
            dict: Information about lookup client/server types and role.
        """
        with self._lookup_lifecycle_lock:
            return {
                "client": self._get_lookup_type_str(self._lookup_client),
                "server": self._get_lookup_type_str(self._lookup_server),
            }

    # ==================== Health & Info ====================

    def is_healthy(self) -> bool:
        """
        Check if the LMCacheManager is healthy.

        Returns False if:
        - Initialization failed
        - HealthMonitor reports unhealthy
        - Engine reports unhealthy

        Returns:
            bool: True if healthy, False otherwise
        """
        if self._init_failed:
            return False
        if self._lmcache_engine is not None and not self._lmcache_engine.is_healthy():
            return False
        if self._health_monitor is not None:
            return self._health_monitor.is_healthy()
        return True

    def get_inference_info(self) -> dict:
        """Get inference information by delegating to the connector.

        Returns:
            dict: Dictionary containing inference information,
                  or empty dict if connector is not available.
        """
        if self._connector is not None and hasattr(
            self._connector, "get_inference_info"
        ):
            return self._connector.get_inference_info()
        return {}

    @classmethod
    def _retain_incomplete_construction_owner(
        cls,
        manager: "LMCacheManager",
    ) -> None:
        """Retain a rejected manager until cleanup proves complete."""
        with cls._incomplete_construction_lock:
            cls._incomplete_construction_owners.add(manager)

    @classmethod
    def _discard_incomplete_construction_owner(
        cls,
        manager: "LMCacheManager",
    ) -> None:
        """Release a retained manager after terminal cleanup proves complete."""
        with cls._incomplete_construction_lock:
            cls._incomplete_construction_owners.discard(manager)
            terminal_receipts = tuple(
                receipt
                for receipt, owner in cls._construction_handoff_owners.items()
                if owner is manager
                and receipt._phase
                in {
                    _ConstructionHandoffPhase.RELEASED,
                    _ConstructionHandoffPhase.FAILED_CLEANUP,
                }
            )
            for receipt in terminal_receipts:
                cls._construction_handoff_owners.pop(receipt, None)
