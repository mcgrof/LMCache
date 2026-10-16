# SPDX-License-Identifier: Apache-2.0
"""Small retained receipts for process-local terminal calls.

The caller of a terminal operation can be interrupted after the operation
returns but before it records success.  Retrying the operation directly is
unsafe when the callee is not idempotent.  ``RetainedCall`` runs one attempt
on a dedicated thread and retains that attempt's completion state.  An
interrupt of the waiting caller therefore leaves the original attempt
reachable for a later public retry.
"""

# Standard
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, Union
import threading

# First Party
from lmcache import torch_dev

RetryPolicy = Union[bool, Callable[[BaseException], bool]]
_CALL_PENDING = object()
_CALL_SUCCEEDED = object()
_LockResult = TypeVar("_LockResult")


def _run_under_interrupt_safe_rlock(
    lock: Any,
    operation: Callable[[], _LockResult],
) -> _LockResult:
    """Run one operation while restoring this thread's exact RLock depth.

    A normal ``with lock:`` exit has a CPython bytecode interval after the
    protected body and before ``__exit__`` releases the lock.  A caller-side
    asynchronous exception in that interval can strand one recursion level.

    The inner cleanup below is itself covered by the outer ``try``.  If one
    interruption lands anywhere from acquisition through normal cleanup, the
    outer handler restores the recursion depth observed on entry.  Receipt
    methods can therefore nest on an already-owned RLock without releasing a
    caller's ownership level.

    This helper deliberately requires CPython's ``RLock._recursion_count``:
    owner-aware depth is what distinguishes an interrupted pre-release from
    an interruption after another thread acquired a non-reentrant lock.
    """
    entry_depth = lock._recursion_count()
    try:
        try:
            lock.acquire()
            result = operation()
        finally:
            while lock._recursion_count() > entry_depth:
                lock.release()
    except BaseException:
        # One caller interruption can also land in the normal cleanup above.
        # Its recovery path is then no longer subject to that same one-shot
        # interruption and can restore precisely the levels added here.
        while lock._recursion_count() > entry_depth:
            lock.release()
        raise
    return result


@dataclass
class _CallAttempt:
    """Completion receipt for one invocation of a retained call."""

    done: threading.Event = field(default_factory=threading.Event)
    outcome: object = _CALL_PENDING
    worker: Optional[threading.Thread] = None


class RetainedCall:
    """Run a terminal call once while retaining caller-interrupt progress.

    A failed attempt is sticky unless ``retry_safe_failure`` explicitly says
    that invoking the call again after that exact exception is safe.  This is
    deliberately conservative: an arbitrary ``close()`` implementation may
    raise only after it has already consumed a non-idempotent resource.
    """

    def __init__(
        self,
        call: Callable[[], None],
        *,
        retry_safe_failure: RetryPolicy = False,
        thread_name: str = "lmcache-retained-call",
        on_worker_start: Optional[Callable[[], None]] = None,
        on_worker_finish: Optional[Callable[[], None]] = None,
    ) -> None:
        self._call = call
        self._retry_safe_failure = retry_safe_failure
        self._thread_name = thread_name
        self._on_worker_start = on_worker_start
        self._on_worker_finish = on_worker_finish
        # Accelerator current-device selection is thread-local. Capture it on
        # the owner thread so every retained worker (including nested close
        # workers) restores the same device before touching GPU resources.
        self._accelerator_device: Optional[Any] = (
            torch_dev.current_device() if torch_dev.is_available() else None
        )
        self._attempt: Optional[_CallAttempt] = None
        self._lock = threading.RLock()

    @property
    def complete(self) -> bool:
        """Return whether one retained attempt completed successfully."""
        with self._lock:
            return (
                self._attempt is not None
                and self._attempt.done.is_set()
                and self._attempt.outcome is _CALL_SUCCEEDED
            )

    @property
    def last_error(self) -> Optional[BaseException]:
        """Return the exact sticky or most recent attempt error."""
        with self._lock:
            if self._attempt is None:
                return None
            outcome = self._attempt.outcome
            return outcome if isinstance(outcome, BaseException) else None

    def running_in_current_thread(self) -> bool:
        """Return whether the retained attempt owns the calling thread.

        Terminal owners use this to reject callback-driven reentry instead of
        waiting for the very attempt whose worker is making the callback.
        """
        with self._lock:
            attempt = self._attempt
            return (
                attempt is not None
                and not attempt.done.is_set()
                and attempt.worker is threading.current_thread()
            )

    def _may_retry(self, error: BaseException) -> bool:
        policy = self._retry_safe_failure
        if callable(policy):
            return bool(policy(error))
        return policy

    def _publish_outcome(self, attempt: _CallAttempt, outcome: object) -> None:
        """Commit one callback outcome despite one bookkeeping interrupt."""
        try:
            with self._lock:
                attempt.outcome = outcome
        except BaseException:
            # The callback has already returned or raised. Outcome publication
            # is local, idempotent bookkeeping; replay it instead of turning a
            # completed terminal side effect into an ambiguous sticky failure.
            with self._lock:
                attempt.outcome = outcome

    @staticmethod
    def _publish_done(attempt: _CallAttempt) -> None:
        """Wake waiters despite one interruption of Event publication."""
        try:
            attempt.done.set()
        except BaseException:
            # Event.set() is idempotent. A waiter must never remain blocked
            # after the authoritative outcome has been committed.
            attempt.done.set()

    @staticmethod
    def _run_idempotent_worker_step(step: Callable[[], None]) -> None:
        """Commit one idempotent worker-local step despite interruption.

        The captured accelerator-device restore and in-tree worker hooks are
        idempotent. An asynchronous exception can land immediately before or
        after their exact side effect. Replay that step once so a pre-callback
        interruption cannot become a sticky terminal-call failure and a
        post-callback interruption cannot leave stale worker ownership behind.
        """
        try:
            step()
        except BaseException:
            step()

    def _restore_accelerator_device(self) -> None:
        """Restore the owner thread's exact accelerator selection."""
        device = self._accelerator_device
        if device is not None:
            torch_dev.set_device(device)

    def _run_attempt(self, attempt: _CallAttempt) -> None:
        try:
            if self._accelerator_device is not None:
                self._run_idempotent_worker_step(
                    self._restore_accelerator_device,
                )
            if self._on_worker_start is not None:
                self._run_idempotent_worker_step(self._on_worker_start)
            self._call()
        except BaseException as error:
            self._publish_outcome(attempt, error)
        else:
            self._publish_outcome(attempt, _CALL_SUCCEEDED)
        finally:
            try:
                if self._on_worker_finish is not None:
                    self._run_idempotent_worker_step(self._on_worker_finish)
            except BaseException as finish_error:
                # Never publish success while worker retirement is unresolved.
                # The in-tree hook is idempotent and the replay above handles
                # one asynchronous interruption; a persistent hook failure is
                # therefore the authoritative sticky attempt outcome.
                self._publish_outcome(attempt, finish_error)
            finally:
                self._publish_done(attempt)

    def _new_attempt_locked(self) -> _CallAttempt:
        attempt = _CallAttempt()
        worker = threading.Thread(
            target=self._run_attempt,
            args=(attempt,),
            name=self._thread_name,
        )
        attempt.worker = worker

        # Construct the complete attempt locally before publishing it.  A
        # caller-side interruption before this assignment leaves no shared
        # state and a public retry can build a fresh attempt.  An interruption
        # after it leaves the exact Thread reachable so the retry starts this
        # same attempt instead of invoking the terminal call twice.
        self._attempt = attempt
        self._ensure_attempt_started_locked(attempt)
        return attempt

    def _ensure_attempt_started_locked(self, attempt: _CallAttempt) -> None:
        """Start a published attempt, including after caller interruption.

        The attempt and its ``Thread`` are retained before ``start()``.  If an
        asynchronous exception lands before the OS thread starts, the next
        public retry starts this same attempt.  If it lands after startup,
        ``Thread.ident`` remains set and prevents a duplicate invocation.
        """
        worker = attempt.worker
        if worker is None:
            raise RuntimeError("retained terminal call has no worker")
        if worker.ident is None and not attempt.done.is_set():
            worker.start()

    def run(self) -> None:
        """Wait for the retained attempt or start one when allowed.

        Raises the exact attempt exception.  If the waiting thread is itself
        interrupted, the attempt continues and remains reachable through this
        object.
        """
        with self._lock:
            attempt = self._attempt
            if (
                attempt is not None
                and attempt.done.is_set()
                and attempt.outcome is _CALL_SUCCEEDED
            ):
                return
            if attempt is None:
                attempt = self._new_attempt_locked()
            elif attempt.done.is_set() and isinstance(
                attempt.outcome,
                BaseException,
            ):
                if self._may_retry(attempt.outcome):
                    attempt = self._new_attempt_locked()
            elif not attempt.done.is_set():
                self._ensure_attempt_started_locked(attempt)

        attempt.done.wait()

        with self._lock:
            outcome = attempt.outcome
            if outcome is _CALL_SUCCEEDED:
                return
        if not isinstance(outcome, BaseException):
            raise RuntimeError("retained terminal call ended without a result")
        raise outcome
