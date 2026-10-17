# SPDX-License-Identifier: Apache-2.0
"""Asserted one-shot CPython instruction interruption for lifecycle tests."""

# Standard
from types import CodeType
from typing import Any, Callable, Optional
import dis
import sys
import threading

# Third Party
import pytest

_MONITOR_LOCK = threading.Lock()
requires_instruction_monitoring = pytest.mark.skipif(
    getattr(sys, "monitoring", None) is None,
    reason="exact instruction interruption requires CPython 3.12+",
)


def instruction_offset(
    function: Callable[..., Any],
    opname: str,
    *,
    argval: object = None,
    occurrence: int = 0,
    after: bool = False,
) -> int:
    """Return one exact bytecode instruction offset for ``function``."""
    if getattr(sys, "monitoring", None) is None:
        # Skip decorators consume this harmless value on Python 3.10/3.11.
        # Their bytecode uses a different call family, so do not attempt to
        # resolve a CPython 3.12+ instruction selector during collection.
        return 0
    instructions = tuple(dis.get_instructions(function))
    matches = [
        index
        for index, instruction in enumerate(instructions)
        if instruction.opname == opname
        and (argval is None or instruction.argval == argval)
    ]
    index = matches[occurrence]
    if after:
        index += 1
    return instructions[index].offset


def method_call_offset(
    function: Callable[..., Any], method: str, *, after: bool = False
) -> int:
    """Select a named method's call rather than an unrelated call ordinal."""
    if getattr(sys, "monitoring", None) is None:
        return 0
    instructions = tuple(dis.get_instructions(function))
    load = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.opname in ("LOAD_ATTR", "LOAD_METHOD")
        and instruction.argval == method
    )
    position = instructions[load].positions
    if position is None:
        raise RuntimeError("instruction monitoring requires source positions")
    call = next(
        index
        for index in range(load + 1, len(instructions))
        if instructions[index].opname in ("CALL", "CALL_KW", "CALL_FUNCTION_EX")
        and (candidate := instructions[index].positions) is not None
        and candidate.lineno == position.lineno
        and candidate.col_offset == position.col_offset
    )
    return instructions[call + int(after)].offset


class InstructionInterruption:
    """Raise one exact error before a selected Python instruction executes."""

    def __init__(
        self,
        function: Callable[..., Any],
        offset: int,
        error: BaseException,
        *,
        thread_name: Optional[str] = None,
    ) -> None:
        self._code: CodeType = function.__code__
        self._offset = offset
        self._error = error
        self._thread_name = thread_name
        self._monitoring: Any = None
        self._tool_id: Optional[int] = None
        self.fired = 0

    def __enter__(self) -> "InstructionInterruption":
        monitoring = getattr(sys, "monitoring", None)
        if monitoring is None:
            raise RuntimeError("instruction interruption requires CPython 3.12+")
        _MONITOR_LOCK.acquire()
        try:
            tool_id = next(
                candidate
                for candidate in range(5, -1, -1)
                if monitoring.get_tool(candidate) is None
            )
            monitoring.use_tool_id(tool_id, "lmcache-lifecycle-test")
            monitoring.register_callback(
                tool_id,
                monitoring.events.INSTRUCTION,
                self._callback,
            )
            monitoring.set_local_events(
                tool_id,
                self._code,
                monitoring.events.INSTRUCTION,
            )
        except BaseException:
            _MONITOR_LOCK.release()
            raise
        self._monitoring = monitoring
        self._tool_id = tool_id
        return self

    def _callback(self, code: CodeType, offset: int) -> None:
        if (
            self.fired == 0
            and code is self._code
            and offset == self._offset
            and (
                self._thread_name is None
                or threading.current_thread().name == self._thread_name
            )
        ):
            self.fired = 1
            raise self._error

    def __exit__(self, *_exc_info: object) -> None:
        monitoring = self._monitoring
        tool_id = self._tool_id
        try:
            if monitoring is not None and tool_id is not None:
                monitoring.set_local_events(
                    tool_id,
                    self._code,
                    monitoring.events.NO_EVENTS,
                )
                monitoring.register_callback(
                    tool_id,
                    monitoring.events.INSTRUCTION,
                    None,
                )
                monitoring.free_tool_id(tool_id)
        finally:
            self._monitoring = None
            self._tool_id = None
            _MONITOR_LOCK.release()

    def assert_fired_once(self) -> None:
        """Fail loudly if the selected interpreter never emitted the event."""
        assert self.fired == 1, f"instruction interruption fired {self.fired} times"
