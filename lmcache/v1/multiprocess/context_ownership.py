# SPDX-License-Identifier: Apache-2.0
"""Retain an exact registered cache context until accepted work drains."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import math
import threading
import uuid

# First Party
from lmcache.v1.multiprocess.transport.request_context import get_request_session


class ContextOwnershipError(RuntimeError):
    """A registration is stale, unowned, retiring, or not proven drained."""


@dataclass(frozen=True, slots=True)
class ContextIdentity:
    """Opaque registration generation and its actual request-client owner."""

    generation: str
    owner_session: str | None


@dataclass(frozen=True, slots=True)
class RopeIdentity:
    """Identify one rotary registration and its original paged-cache context."""

    rotary: ContextIdentity
    context: ContextIdentity | None


class ContextHold:
    """Own one physical-completion obligation for a registered context.

    The server completion path must retain this object until its work physically
    finishes. Dropping the Python object is not evidence of completion and does
    not release the obligation. Releasing a hold is idempotent.
    """

    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        """Report physical completion exactly once, including after revocation."""
        with self._lock:
            if self._released:
                return
            self._released = True
        self._release()


class ContextOwnership:
    """Bind a context's creation session and drain opt-in accepted work.

    Construct while processing actual cache registration. A legacy/direct call
    without request context records no owner and cannot grant opt-in borrows.
    All methods are thread safe. Retirement callbacks run without this object's
    lock and must revoke their users before the underlying context is closed.
    """

    def __init__(self) -> None:
        try:
            session = get_request_session()
        except RuntimeError:
            session = None
        self._identity = ContextIdentity(uuid.uuid4().hex, session)
        self._condition = threading.Condition()
        self._retiring = False
        self._revocation_complete = False
        self._revocation_error: Exception | None = None
        self._holds: dict[str, Callable[[], None]] = {}

    @property
    def identity(self) -> ContextIdentity:
        """Return the immutable registration identity; it does not authorize I/O."""
        return self._identity

    @property
    def pending(self) -> int:
        """Return accepted holds that still require physical completion."""
        with self._condition:
            return len(self._holds)

    def borrow(
        self,
        expected: ContextIdentity,
        actual_session: str,
        revoke: Callable[[], None],
    ) -> ContextHold:
        """Retain this exact generation for its actual owning request client.

        Args:
            expected: Snapshot obtained from the registered context.
            actual_session: Receiving transport's get_request_session() result.
            revoke: Callback that disarms the borrower when retirement starts.

        Raises:
            ContextOwnershipError: Stale generation, missing/mismatched owner,
                or retirement already started.
        """
        with self._condition:
            if (
                self._retiring
                or expected != self._identity
                or self._identity.owner_session is None
                or actual_session != self._identity.owner_session
            ):
                raise ContextOwnershipError("context registration is not borrowable")
            token = uuid.uuid4().hex
            self._holds[token] = revoke
        return ContextHold(lambda: self._finish(token))

    def retire(self, timeout: float = 5.0) -> None:
        """Close admission, revoke borrowers and wait for physical completion.

        Args:
            timeout: Finite nonnegative drain budget in seconds. Zero performs
                a nonblocking check after revocation callbacks are delivered.

        Raises:
            ContextOwnershipError: A callback failed or physical work remains.
                Keep the context quarantined; this never releases its holds.
            ValueError: Invalid timeout.

        Repeated retirement cannot clear failed revocation. A previous timeout
        can finish after all real work drains, but admission never reopens.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("drain timeout must be finite and nonnegative")
        with self._condition:
            deliver_callbacks = not self._retiring
            callbacks = list(self._holds.values()) if deliver_callbacks else []
            self._retiring = True
        failure: Exception | None = None
        for callback in callbacks:
            try:
                callback()
            except Exception as exc:
                if failure is None:
                    failure = exc
        with self._condition:
            if deliver_callbacks:
                self._revocation_error = failure
                self._revocation_complete = True
                self._condition.notify_all()
            if not self._condition.wait_for(
                lambda: self._revocation_complete
                and (self._revocation_error is not None or not self._holds),
                timeout,
            ):
                raise ContextOwnershipError(
                    "context still owns undrained work or revocation"
                )
            if self._revocation_error is not None:
                raise ContextOwnershipError(
                    "context revocation did not complete"
                ) from self._revocation_error

    def _finish(self, token: str) -> None:
        with self._condition:
            del self._holds[token]
            self._condition.notify_all()
