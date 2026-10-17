# SPDX-License-Identifier: Apache-2.0
"""Exact construction ownership for external vLLM connector wrappers."""

# Standard
from enum import Enum, auto
from typing import Any, Callable, Generic, Optional, TypeVar, Union, cast

# First Party
from lmcache.v1.lifecycle import RetainedCall

_ConnectorT = TypeVar("_ConnectorT")
_CONNECTOR_UNPUBLISHED = object()


class ConnectorConstructionPhase(Enum):
    """Whether the retained connector constructor has been admitted."""

    PRECALL = auto()
    INVOCATION_MAY_HAVE_BEGUN = auto()


class RetainedConnectorConstructionOwner(Generic[_ConnectorT]):
    """Retain one connector call and its provisionally owned Manager."""

    def __init__(
        self,
        factory_call: Callable[[Callable[[_ConnectorT], None]], _ConnectorT],
        *,
        thread_name: str,
    ) -> None:
        self._factory_call = factory_call
        self._connector: Union[object, _ConnectorT] = _CONNECTOR_UNPUBLISHED
        self._manager: Optional[Any] = None
        self._handoff_receipt: Optional[Any] = None
        self._provisional_manager_retained = False
        self.phase = ConnectorConstructionPhase.PRECALL
        self._failure_promotion_call = RetainedCall(
            self._promote_failed_manager_handoff_once,
            retry_safe_failure=True,
            thread_name=thread_name + "-failure-promotion",
        )
        self._call = RetainedCall(
            self._invoke,
            thread_name=thread_name,
        )

    def _invoke(self) -> None:
        connector = self._factory_call(self.publish_connector)
        if self._published_connector() is not connector:
            raise RuntimeError("connector factory returned another published result")
        manager = getattr(connector, "_manager", None)
        if manager is None:
            raise RuntimeError("constructed connector has no Manager owner")

        # The wrapper's waiting thread can be interrupted before its STORE_ATTR.
        # Keep the Manager globally reachable before the retained worker can
        # publish successful completion and hand the exact connector back.
        self._manager = manager
        receipt = manager.new_construction_handoff_receipt()
        self._handoff_receipt = receipt
        manager.retain_construction_handoff(receipt)
        self._provisional_manager_retained = True

    def publish_connector(self, connector: _ConnectorT) -> None:
        """Accept the exact connector before its constructor can return."""
        try:
            self._connector = connector
        except BaseException:
            self._connector = connector

    @property
    def factory_returned(self) -> bool:
        """Whether the exact connector reached this retained owner."""
        return self._connector is not _CONNECTOR_UNPUBLISHED

    def _published_connector(self) -> _ConnectorT:
        connector = self._connector
        if connector is _CONNECTOR_UNPUBLISHED:
            raise RuntimeError("connector factory has no retained result")
        return cast(_ConnectorT, connector)

    @property
    def connector(self) -> _ConnectorT:
        """Return the exact successfully constructed connector."""
        if not self._call.complete:
            raise RuntimeError("connector construction has no retained result")
        return self._published_connector()

    @property
    def last_error(self) -> Optional[BaseException]:
        """Return the exact sticky connector-construction failure."""
        return self._call.last_error

    @property
    def provisional_manager_retained(self) -> bool:
        """Return whether normal publication still owes an owner transfer."""
        return self._provisional_manager_retained

    def invoke(self) -> None:
        """Admit and run the retained connector constructor exactly once."""
        if self.phase is not ConnectorConstructionPhase.PRECALL:
            raise RuntimeError("connector construction was already admitted")
        self.phase = ConnectorConstructionPhase.INVOCATION_MAY_HAVE_BEGUN
        self._call.run()

    def recover(self) -> Optional[_ConnectorT]:
        """Resume an admitted call, or prove PRECALL constructed nothing."""
        if self.phase is ConnectorConstructionPhase.PRECALL:
            return None
        try:
            self._call.run()
        except BaseException as call_error:
            if self._call.last_error is not call_error or not self.factory_returned:
                raise
            # The connector returned successfully; only provisional Manager
            # retention failed. The owner-held connector is therefore the
            # exact cleanup target even though RetainedCall is sticky-failed.
            return self._published_connector()
        return self.connector

    def release_provisional_manager(self) -> None:
        """Transfer Manager lifetime to a normally published connector."""
        if not self._provisional_manager_retained:
            return
        manager = self._manager
        receipt = self._handoff_receipt
        if manager is None or receipt is None:
            raise RuntimeError("connector construction lost its Manager owner")
        manager.release_construction_handoff(receipt)
        self._provisional_manager_retained = False

    def _promote_failed_manager_handoff_once(self) -> None:
        """Worker body publishing the exact failed Manager receipt once."""
        manager = self._manager
        if manager is None:
            # The connector is published first. If a worker interruption lands
            # before the separate Manager mirror, recover the exact Manager
            # from that already-owned connector rather than dropping it.
            connector = self._published_connector()
            manager = getattr(connector, "_manager", None)
            if manager is None:
                raise RuntimeError("connector construction lost its Manager owner")
            self._manager = manager
        receipt = self._handoff_receipt
        if receipt is None:
            manager.retain_construction_cleanup()
            return
        manager.promote_construction_handoff_failure(receipt)
        self._provisional_manager_retained = False

    def promote_failed_manager_handoff(self) -> None:
        """Retain promotion progress before failed-wrapper cleanup."""
        try:
            self._failure_promotion_call.run()
        except BaseException as promotion_error:
            # Complete the same prepublished attempt before propagating a
            # caller-side interruption. Exact promotion is replay-safe even
            # when the first worker failed just before/after publishing phase.
            try:
                self._failure_promotion_call.run()
            except BaseException as recovery_error:
                if recovery_error is not promotion_error:
                    raise promotion_error from recovery_error
            raise


def cleanup_failed_connector_construction(
    owner: RetainedConnectorConstructionOwner[Any],
    init_error: BaseException,
) -> None:
    """Recover and shut down an exact connector lost by wrapper publication."""
    try:
        connector = owner.recover()
    except BaseException as recovery_error:
        if owner.last_error is init_error:
            # The connector constructor itself raised and owns cleanup of its
            # partially initialized Manager graph.
            return
        raise init_error from recovery_error

    if connector is None:
        return
    try:
        owner.promote_failed_manager_handoff()
        connector.shutdown()
    except BaseException as cleanup_error:
        raise init_error from cleanup_error
