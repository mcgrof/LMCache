# SPDX-License-Identifier: Apache-2.0
"""Exact publication-boundary tests for retained terminal calls."""

# Standard
from typing import Any, Callable
import inspect
import sys
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.lifecycle import RetainedCall
from tests.v1.instruction_monitoring import (
    InstructionInterruption,
    instruction_offset,
    requires_instruction_monitoring,
)
import lmcache.v1.lifecycle as lifecycle_module


def _line(function: Callable[..., Any], source: str) -> int:
    lines, start = inspect.getsourcelines(function)
    return next(
        start + offset for offset, text in enumerate(lines) if text.strip() == source
    )


def _one_shot_interrupt(
    function: Callable[..., Any],
    line: int,
    message: str,
) -> Callable[[Any, str, Any], Any]:
    triggered = False

    def trace(frame: Any, event: str, arg: Any) -> Any:
        del arg
        nonlocal triggered
        if (
            not triggered
            and frame.f_code is function.__code__
            and event == "line"
            and frame.f_lineno == line
        ):
            triggered = True
            sys.settrace(None)
            raise KeyboardInterrupt(message)
        return trace

    return trace


def test_interrupt_before_attempt_publication_rebuilds_once() -> None:
    calls = 0

    def terminal_call() -> None:
        nonlocal calls
        calls += 1

    retained = RetainedCall(terminal_call)
    interrupt = _one_shot_interrupt(
        RetainedCall._new_attempt_locked,
        _line(RetainedCall._new_attempt_locked, "self._attempt = attempt"),
        "before retained-attempt publication",
    )

    sys.settrace(interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="before retained-attempt"):
            retained.run()
    finally:
        sys.settrace(None)

    assert retained._attempt is None
    retained.run()
    retained.run()
    assert calls == 1
    assert retained.complete


def test_interrupt_after_attempt_publication_reuses_attached_thread() -> None:
    calls = 0

    def terminal_call() -> None:
        nonlocal calls
        calls += 1

    retained = RetainedCall(terminal_call)
    interrupt = _one_shot_interrupt(
        RetainedCall._new_attempt_locked,
        _line(
            RetainedCall._new_attempt_locked,
            "self._ensure_attempt_started_locked(attempt)",
        ),
        "after retained-attempt publication",
    )

    sys.settrace(interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="after retained-attempt"):
            retained.run()
    finally:
        sys.settrace(None)

    attempt = retained._attempt
    assert attempt is not None
    assert attempt.worker is not None
    assert attempt.worker.ident is None

    retained.run()
    retained.run()
    assert calls == 1
    assert retained.complete


@requires_instruction_monitoring
def test_worker_interrupt_before_success_outcome_commits_success() -> None:
    """A returned callback commits success despite bookkeeping interruption."""
    calls = 0

    def terminal_call() -> None:
        nonlocal calls
        calls += 1

    retained = RetainedCall(
        terminal_call,
        thread_name="test-retained-completion-boundary",
    )
    interrupt = InstructionInterruption(
        RetainedCall._publish_outcome,
        instruction_offset(
            RetainedCall._publish_outcome,
            "STORE_ATTR",
            argval="outcome",
        ),
        KeyboardInterrupt("before retained success outcome"),
        thread_name="test-retained-completion-boundary",
    )

    with interrupt:
        retained.run()
    interrupt.assert_fired_once()

    retained.run()
    assert retained.complete
    assert retained.last_error is None
    assert calls == 1


@requires_instruction_monitoring
def test_worker_interrupt_before_failure_outcome_preserves_error() -> None:
    """Failure bookkeeping replays the callback's exact original exception."""
    callback_error = ValueError("terminal callback failed")

    def terminal_call() -> None:
        raise callback_error

    retained = RetainedCall(
        terminal_call,
        thread_name="test-retained-failure-outcome",
    )
    interrupt = InstructionInterruption(
        RetainedCall._publish_outcome,
        instruction_offset(
            RetainedCall._publish_outcome,
            "STORE_ATTR",
            argval="outcome",
        ),
        KeyboardInterrupt("before retained failure outcome"),
        thread_name="test-retained-failure-outcome",
    )

    with interrupt:
        with pytest.raises(ValueError) as failure:
            retained.run()
    interrupt.assert_fired_once()

    assert failure.value is callback_error
    assert retained.last_error is callback_error
    assert not retained.complete


@requires_instruction_monitoring
def test_worker_interrupt_before_done_wakes_waiter() -> None:
    """A committed outcome always wakes its bounded waiting caller."""
    retained = RetainedCall(
        lambda: None,
        thread_name="test-retained-done-boundary-worker",
    )
    interrupt = InstructionInterruption(
        RetainedCall._publish_done,
        instruction_offset(RetainedCall._publish_done, "CALL"),
        KeyboardInterrupt("before retained done wakeup"),
        thread_name="test-retained-done-boundary-worker",
    )
    failures: list[BaseException] = []

    def wait_for_call() -> None:
        try:
            retained.run()
        except BaseException as error:
            failures.append(error)

    waiter = threading.Thread(target=wait_for_call, name="test-retained-done-waiter")
    try:
        with interrupt:
            waiter.start()
            waiter.join(timeout=5.0)
    finally:
        if waiter.is_alive():
            attempt = retained._attempt
            assert attempt is not None
            attempt.done.set()
            waiter.join(timeout=5.0)

    interrupt.assert_fired_once()
    assert not waiter.is_alive()
    assert failures == []
    assert retained.complete


@pytest.mark.parametrize(
    "after",
    [False, True],
    ids=["before-set-device", "after-set-device"],
)
@requires_instruction_monitoring
def test_worker_device_restore_replays_exact_call_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    after: bool,
) -> None:
    """A one-shot device-restore interrupt cannot suppress the callback."""

    class _Device:
        def __init__(self) -> None:
            self.set_calls = 0

        def is_available(self) -> bool:
            return True

        def current_device(self) -> int:
            return 7

        def set_device(self, device: int) -> None:
            assert device == 7
            self.set_calls += 1

    device = _Device()
    monkeypatch.setattr(lifecycle_module, "torch_dev", device)
    terminal_calls = 0

    def terminal_call() -> None:
        nonlocal terminal_calls
        terminal_calls += 1

    thread_name = f"test-retained-device-restore-{after}"
    retained = RetainedCall(terminal_call, thread_name=thread_name)
    interrupt = InstructionInterruption(
        RetainedCall._restore_accelerator_device,
        instruction_offset(
            RetainedCall._restore_accelerator_device,
            "CALL",
            after=after,
        ),
        KeyboardInterrupt("around retained device restore"),
        thread_name=thread_name,
    )

    with interrupt:
        retained.run()
    interrupt.assert_fired_once()

    assert retained.complete
    assert retained.last_error is None
    assert terminal_calls == 1
    assert device.set_calls == (2 if after else 1)


def test_persistent_worker_device_restore_failure_is_sticky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real setup failure never invokes or ambiguously retries the callback."""
    setup_error = RuntimeError("persistent accelerator restore failure")

    class _Device:
        def __init__(self) -> None:
            self.set_calls = 0

        def is_available(self) -> bool:
            return True

        def current_device(self) -> int:
            return 11

        def set_device(self, device: int) -> None:
            assert device == 11
            self.set_calls += 1
            raise setup_error

    device = _Device()
    monkeypatch.setattr(lifecycle_module, "torch_dev", device)
    terminal_calls = 0

    def terminal_call() -> None:
        nonlocal terminal_calls
        terminal_calls += 1

    retained = RetainedCall(
        terminal_call,
        thread_name="test-retained-persistent-device-restore",
    )
    for _ in range(2):
        with pytest.raises(RuntimeError) as failure:
            retained.run()
        assert failure.value is setup_error

    assert not retained.complete
    assert retained.last_error is setup_error
    assert terminal_calls == 0
    assert device.set_calls == 2


@pytest.mark.parametrize("after", [False, True], ids=["before-add", "after-add"])
@requires_instruction_monitoring
def test_worker_start_hook_replays_exact_call_interrupt(after: bool) -> None:
    """Worker admission cannot become a sticky pre-callback failure."""
    worker_ids: set[int] = set()
    start_calls = 0
    terminal_calls = 0

    def register_worker() -> None:
        nonlocal start_calls
        start_calls += 1
        worker_ids.add(threading.get_ident())

    def terminal_call() -> None:
        nonlocal terminal_calls
        terminal_calls += 1
        assert threading.get_ident() in worker_ids

    def unregister_worker() -> None:
        worker_ids.discard(threading.get_ident())

    thread_name = f"test-retained-worker-start-{after}"
    retained = RetainedCall(
        terminal_call,
        thread_name=thread_name,
        on_worker_start=register_worker,
        on_worker_finish=unregister_worker,
    )
    interrupt = InstructionInterruption(
        register_worker,
        instruction_offset(
            register_worker,
            "CALL",
            occurrence=1,
            after=after,
        ),
        KeyboardInterrupt("around retained worker-start publication"),
        thread_name=thread_name,
    )

    with interrupt:
        retained.run()
    interrupt.assert_fired_once()

    assert retained.complete
    assert retained.last_error is None
    assert start_calls == 2
    assert terminal_calls == 1
    assert worker_ids == set()


@pytest.mark.parametrize(
    "after",
    [False, True],
    ids=["before-discard", "after-discard"],
)
@requires_instruction_monitoring
def test_worker_finish_hook_replays_exact_call_interrupt(after: bool) -> None:
    """A completed callback cannot leave its worker registration live."""
    worker_ids: set[int] = set()
    finish_calls = 0

    def register_worker() -> None:
        worker_ids.add(threading.get_ident())

    def unregister_worker() -> None:
        nonlocal finish_calls
        finish_calls += 1
        worker_ids.discard(threading.get_ident())

    thread_name = f"test-retained-worker-finish-{after}"
    retained = RetainedCall(
        lambda: None,
        thread_name=thread_name,
        on_worker_start=register_worker,
        on_worker_finish=unregister_worker,
    )
    interrupt = InstructionInterruption(
        unregister_worker,
        instruction_offset(
            unregister_worker,
            "CALL",
            occurrence=1,
            after=after,
        ),
        KeyboardInterrupt("around retained worker-finish hook"),
        thread_name=thread_name,
    )

    with interrupt:
        retained.run()
    interrupt.assert_fired_once()

    assert retained.complete
    assert retained.last_error is None
    assert finish_calls == 2
    assert worker_ids == set()


def test_complete_waits_for_worker_finish_hook() -> None:
    """Callback success stays hidden until worker retirement commits."""
    finish_entered = threading.Event()
    release_finish = threading.Event()

    def unregister_worker() -> None:
        finish_entered.set()
        assert release_finish.wait(timeout=5.0)

    retained = RetainedCall(
        lambda: None,
        thread_name="test-retained-complete-after-finish",
        on_worker_finish=unregister_worker,
    )
    failures: list[BaseException] = []

    def wait_for_call() -> None:
        try:
            retained.run()
        except BaseException as error:
            failures.append(error)

    waiter = threading.Thread(
        target=wait_for_call,
        name="test-retained-complete-after-finish-waiter",
    )
    try:
        waiter.start()
        assert finish_entered.wait(timeout=5.0)
        assert not retained.complete
    finally:
        release_finish.set()
        waiter.join(timeout=5.0)

    assert not waiter.is_alive()
    assert failures == []
    assert retained.complete


def test_nested_retained_calls_restore_captured_accelerator_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every fresh worker inherits the owner thread's accelerator device."""

    class _ThreadLocalDevice:
        def __init__(self) -> None:
            self.local = threading.local()
            self.events: list[tuple[str, str, int]] = []
            self.lock = threading.Lock()

        def is_available(self) -> bool:
            return True

        def current_device(self) -> int:
            device = getattr(self.local, "device", 0)
            with self.lock:
                self.events.append(("capture", threading.current_thread().name, device))
            return device

        def set_device(self, device: int) -> None:
            self.local.device = device
            with self.lock:
                self.events.append(("set", threading.current_thread().name, device))

    device = _ThreadLocalDevice()
    device.local.device = 7
    monkeypatch.setattr(lifecycle_module, "torch_dev", device)
    observed: list[tuple[str, int]] = []

    def inner_call() -> None:
        observed.append(("inner", device.current_device()))

    def outer_call() -> None:
        observed.append(("outer", device.current_device()))
        RetainedCall(inner_call, thread_name="nested-device-inner").run()

    RetainedCall(outer_call, thread_name="nested-device-outer").run()

    assert observed == [("outer", 7), ("inner", 7)]
    set_events = [event for event in device.events if event[0] == "set"]
    assert set_events == [
        ("set", "nested-device-outer", 7),
        ("set", "nested-device-inner", 7),
    ]
