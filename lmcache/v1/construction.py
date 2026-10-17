# SPDX-License-Identifier: Apache-2.0
"""Retry ownership for failed component construction.

Lookup factories must preserve the original construction exception because
callers use its identity to distinguish a failed retained attempt from an
interrupted waiter. Replacing that exception with a cleanup wrapper would
break the retained-call protocol. This module instead attaches an exact,
strong cleanup receipt to the original exception.

The receipt is deliberately independent of ``LMCacheManager`` and of the
lookup-client package. A factory performs one best-effort drain before
re-raising. Manager recovery can later call
:func:`drain_lookup_construction_cleanup` on the same sticky exception and may
resolve the factory result as absent only after it returns.
"""

# Standard
from collections.abc import Callable
from typing import Any, Protocol
import threading

# First Party
from lmcache.v1.lifecycle import RetainedCall, RetryPolicy


class LookupConstructionCleanupOwner(Protocol):
    """One exact resource owner retained after construction failed."""

    def cleanup(self) -> None:
        """Make retry-safe cleanup progress or raise while retaining ownership."""


class LookupConstructionCleanupReceipt:
    """Ordered, exact cleanup ownership attached to one BaseException."""

    def __init__(self, primary_error: BaseException) -> None:
        self._owners: list[LookupConstructionCleanupOwner] = []
        self._lock = threading.RLock()
        # ``raise primary from cleanup`` temporarily replaces these fields.
        # Preserve the construction error's original chain so a successful
        # retry, or promotion of a process-control cleanup interrupt, can
        # restore it without creating a two-object cause cycle.
        self._primary_cause = primary_error.__cause__
        self._primary_suppress_context = primary_error.__suppress_context__

    @property
    def complete(self) -> bool:
        """Return whether every retained owner has completed cleanup."""
        with self._lock:
            return not self._owners

    @property
    def owner_count(self) -> int:
        """Return the number of exact owners still awaiting cleanup."""
        with self._lock:
            return len(self._owners)

    def retain(self, owner: LookupConstructionCleanupOwner) -> None:
        """Retain one exact owner without collapsing distinct ownership."""
        with self._lock:
            if any(retained is owner for retained in self._owners):
                return
            self._owners.append(owner)

    def drain(self) -> None:
        """Drain owners in order, preserving each successful transition.

        Owners remain strongly reachable until their own cleanup call returns.
        A failed owner stays at the front of the receipt so dependent owners
        (for example a private context after its socket) cannot run too early.
        Every owner must make retries idempotent; close-like owners below use a
        retained worker receipt for the call-to-success-publication boundary.
        """
        with self._lock:
            while self._owners:
                owner = self._owners[0]
                owner.cleanup()
                if self._owners and self._owners[0] is owner:
                    self._owners.pop(0)
                else:
                    raise RuntimeError(
                        "lookup construction cleanup owner changed during drain"
                    )

    def restore_primary_cause(self, error: BaseException) -> None:
        """Restore the construction error's pre-cleanup exception chain."""
        error.__cause__ = self._primary_cause
        error.__suppress_context__ = self._primary_suppress_context


class CloseMethodCleanupOwner:
    """Retain one object until a close-like method is proven complete.

    An actual method exception is sticky by default. Callers may opt into a
    retry only after auditing that exact component's close contract. A
    terminal-state probe can additionally prove that a method consumed its
    resource before raising, avoiding an unnecessary second invocation.
    """

    def __init__(
        self,
        resource: Any,
        *,
        method_name: str = "close",
        method_kwargs: dict[str, Any] | None = None,
        thread_name: str = "lmcache-construction-cleanup",
        retry_safe_failure: RetryPolicy = False,
        completion_probe: Callable[[], bool] | None = None,
    ) -> None:
        self.resource = resource
        self._method_name = method_name
        self._method_kwargs = dict(method_kwargs or {})
        self._completion_probe = completion_probe
        self._call = RetainedCall(
            self._invoke,
            retry_safe_failure=retry_safe_failure,
            thread_name=thread_name,
        )

    def _invoke(self) -> None:
        method = getattr(self.resource, self._method_name)
        method(**self._method_kwargs)

    def cleanup(self) -> None:
        """Resume one exact call or an explicitly retry-safe failed attempt."""
        if self._completion_probe is not None and self._completion_probe():
            return
        try:
            self._call.run()
        except BaseException:
            # A terminal method can raise after it consumed the resource but
            # before returning. Prefer its authoritative state over retrying.
            if self._completion_probe is not None and self._completion_probe():
                return
            raise


_RECEIPT_ATTRIBUTE = "_lmcache_lookup_construction_cleanup_receipt"


def get_lookup_construction_cleanup_receipt(
    error: BaseException,
) -> LookupConstructionCleanupReceipt | None:
    """Return the exact cleanup receipt attached to ``error``, if any."""
    receipt = getattr(error, _RECEIPT_ATTRIBUTE, None)
    if receipt is None:
        return None
    if not isinstance(receipt, LookupConstructionCleanupReceipt):
        raise RuntimeError("lookup construction exception has an invalid receipt")
    return receipt


def retain_lookup_construction_cleanup(
    error: BaseException,
    owner: LookupConstructionCleanupOwner,
) -> LookupConstructionCleanupReceipt:
    """Attach and retain ``owner`` despite one publication interruption."""
    try:
        return _commit_lookup_construction_cleanup(error, owner)
    except BaseException:
        # ``owner`` is a strong function argument before the first fallible
        # receipt publication. Replaying closes both sides of every commit
        # boundary: before/after exception-attribute attachment and
        # before/after the identity-deduplicated owner-list append.
        return _commit_lookup_construction_cleanup(error, owner)


def _commit_lookup_construction_cleanup(
    error: BaseException,
    owner: LookupConstructionCleanupOwner,
) -> LookupConstructionCleanupReceipt:
    """Perform one idempotent exact cleanup-owner publication."""
    receipt = get_lookup_construction_cleanup_receipt(error)
    if receipt is None:
        receipt = LookupConstructionCleanupReceipt(error)
        setattr(error, _RECEIPT_ATTRIBUTE, receipt)
    receipt.retain(owner)
    return receipt


def drain_lookup_construction_cleanup(error: BaseException) -> bool:
    """Drain cleanup attached to a sticky factory exception.

    Returns:
        ``True`` when no retained owner remains. A missing receipt means the
        factory failed before it acquired a cleanup-requiring resource.

    Raises:
        BaseException: The *same* ``error`` object, with the current cleanup
        failure as its explicit cause, while retry ownership remains attached.
    """
    receipt = get_lookup_construction_cleanup_receipt(error)
    if receipt is None:
        return True
    try:
        receipt.drain()
    except BaseException as cleanup_error:
        if cleanup_error is error:
            raise
        raise error from cleanup_error
    receipt.restore_primary_cause(error)
    return receipt.complete


def restore_lookup_construction_error_cause(error: BaseException) -> None:
    """Restore ``error`` before promoting an attached cleanup interrupt."""
    receipt = get_lookup_construction_cleanup_receipt(error)
    if receipt is not None:
        receipt.restore_primary_cause(error)
