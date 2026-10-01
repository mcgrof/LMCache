# SPDX-License-Identifier: Apache-2.0
"""Drive the worker's submit-failure transitions with a deterministic plan.

The submit-failure paths cannot be reached on a working device: nothing makes
a healthy ring refuse a submission. The engine therefore carries a nondefault
``fault-injection`` feature whose plan replaces what one submit call reports,
before the ring is consulted -- so the worker runs its ordinary code against
an outcome it cannot otherwise be given, and no completion is ever synthesised
for a request the kernel owns.

Substituting before the ring is also what the plan cannot say: with no entry
offered to the kernel, the only take count that is true is zero. A partial
take has to come from the submission queue actually filling up.

The seam is off in a serving build, so these run against a separately built
artifact named by ``LMCACHE_RAW_BLOCK_FAULT_EXT``. The two artifacts are kept
apart on purpose: what is measured here is a build nobody serves with, and
the ordinary depth and readback cases must run on the artifact that is.

Build it with, from ``rust/raw_block``::

    maturin build --release --features fault-injection -i <python> --out <dir>
    python -m zipfile -e <dir>/<wheel> <extdir>
    LMCACHE_RAW_BLOCK_FAULT_EXT=<extdir> pytest <this file>
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap

# Third Party
import pytest

FAULT_EXT = os.environ.get("LMCACHE_RAW_BLOCK_FAULT_EXT", "")

pytestmark = pytest.mark.skipif(
    not FAULT_EXT,
    reason=(
        "needs an engine built with the nondefault fault-injection feature; "
        "set LMCACHE_RAW_BLOCK_FAULT_EXT to an unpacked build of it (see this "
        "module's docstring)"
    ),
)

# Run in a subprocess: one interpreter cannot hold two builds of one native
# module, and the ordinary build is what the rest of the suite imports.
SCENARIO = textwrap.dedent(
    """
    import json, os, sys
    sys.path.insert(0, sys.argv[1])
    from lmcache_rust_raw_block_io import RawBlockDevice

    device_path, plan_spec, ordinal = sys.argv[2], sys.argv[3], int(sys.argv[4])
    dev = RawBlockDevice(
        device_path,
        writable=True,
        use_iouring=True,
        use_odirect=False,
        alignment=4096,
        iouring_queue_depth=8,
    )
    report = {"feature": RawBlockDevice.has_fault_injection()}
    try:
        payload = bytearray(b"x" * 4096)
        # One healthy write first, so the plan's ordinal names a submit the
        # worker makes for the write under test rather than a warm-up.
        dev.wait_iouring(dev.batched_write([0], [payload], [4096], [None]))
        base = dev.submit_call_count()
        dev.inject_submit_faults([(base + ordinal, plan_spec)])
        batch = dev.batched_write([4096], [payload], [4096], [None])
        results, errors = dev.wait_iouring(batch)
        report["results"] = list(results)
        report["errors"] = [str(e) for _, e in errors]
        report["poisoned"] = dev.is_poisoned()
        report["quarantined_batches"] = dev.quarantined_batch_count()
        report["quarantined_owners"] = dev.quarantined_owner_count()
    except BaseException as exc:
        report["raised"] = f"{type(exc).__name__}: {exc}"
        report["poisoned"] = dev.is_poisoned()
        report["quarantined_batches"] = dev.quarantined_batch_count()
        report["quarantined_owners"] = dev.quarantined_owner_count()
    print(json.dumps(report))
    """
)


def _run(tmp_path: Path, plan_spec: str, ordinal: int = 0) -> dict:
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(64 * 1024 * 1024)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            SCENARIO,
            FAULT_EXT,
            str(device),
            plan_spec,
            str(ordinal),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["feature"] is True, "the artifact under test has no fault seam"
    return report


def test_the_artifact_under_test_is_not_the_serving_one() -> None:
    """The two builds are recorded apart, because they are different code.

    A number measured on a build carrying a test seam says nothing about the
    artifact that serves, so the identities are stated rather than assumed to
    be the same.
    """
    fault_so = sorted(Path(FAULT_EXT).rglob("*.so"))
    assert fault_so, f"no extension found under {FAULT_EXT}"
    fault_digest = hashlib.sha256(fault_so[0].read_bytes()).hexdigest()

    # Third Party
    import lmcache_rust_raw_block_io as serving

    serving_so = sorted(Path(serving.__file__).parent.rglob("*.so"))
    assert serving_so, "no serving extension found"
    serving_digest = hashlib.sha256(serving_so[0].read_bytes()).hexdigest()

    assert fault_digest != serving_digest, (
        "the fault build and the serving build are the same artifact; the "
        "seam would then be present in what serves"
    )
    assert serving.RawBlockDevice.has_fault_injection() is False, (
        "the serving build carries the fault seam"
    )
    # And not the scripted ring either. A capability probe answering false
    # while the type still carries the control surface would mean the seam
    # was merely hidden.
    assert not hasattr(serving.RawBlockDevice, "fake_complete"), (
        "the serving build carries the scripted ring's control surface"
    )
    with tempfile.TemporaryDirectory() as directory:
        probe = Path(directory) / "probe.bin"
        with open(probe, "wb") as handle:
            handle.truncate(1024 * 1024)
        # Asked for one, it refuses rather than silently serving a real
        # ring: a caller measuring against a scripted ring that is not
        # there is measuring nothing.
        with pytest.raises(ValueError, match="fault-injection"):
            serving.RawBlockDevice(
                str(probe),
                writable=True,
                use_iouring=True,
                alignment=4096,
                fake_ring_capacity=4,
            )
    print(f"fault build   sha256={fault_digest}")
    print(f"serving build sha256={serving_digest}")


UNPOLLED_OWNER_SCENARIO = textwrap.dedent(
    """
    import gc, json, sys, time, weakref
    sys.path.insert(0, sys.argv[1])
    from lmcache_rust_raw_block_io import RawBlockDevice

    dev = RawBlockDevice(
        sys.argv[2], writable=True, use_iouring=True, use_odirect=False,
        alignment=4096, iouring_queue_depth=8,
    )

    class Payload(bytearray):
        pass

    payload = Payload(b"x" * 4096)
    alive = weakref.ref(payload)

    base = dev.submit_call_count()
    dev.inject_submit_faults([(base, "fatal:5")])
    dev.batched_write([0], [payload], [4096], [None])
    # Deliberately no wait_iouring: this is the caller that never polls.
    deadline = time.monotonic() + 10
    while not dev.is_poisoned() and time.monotonic() < deadline:
        time.sleep(0.01)

    report = {
        "poisoned": dev.is_poisoned(),
        "quarantined_batches": dev.quarantined_batch_count(),
        "owners_without_polling": dev.quarantined_owner_count(),
    }
    del payload
    gc.collect()
    report["alive_after_our_reference_went"] = alive() is not None
    try:
        dev.close()
        report["close"] = "returned"
    except BaseException as exc:
        report["close"] = type(exc).__name__
    del dev
    gc.collect()
    report["alive_after_close_and_drop"] = alive() is not None
    print(json.dumps(report))
    """
)


def _run_scenario(scenario: str, device: Path) -> dict:
    """Run one scenario in its own interpreter and return its report.

    A subprocess per scenario, because one interpreter cannot hold two
    builds of one native module and the rest of the suite imports the
    ordinary one.
    """
    result = subprocess.run(
        [sys.executable, "-c", scenario, FAULT_EXT, str(device)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


UNSWEPT_HEALTHY_OWNER_SCENARIO = textwrap.dedent(
    """
    import gc, json, sys, time, weakref
    sys.path.insert(0, sys.argv[1])
    from lmcache_rust_raw_block_io import RawBlockDevice

    dev = RawBlockDevice(
        sys.argv[2], writable=True, use_iouring=True, use_odirect=False,
        alignment=4096, iouring_queue_depth=8,
    )

    class Payload(bytearray):
        pass

    # A batch that completes normally and that nobody polls. Nothing marks
    # it unknown, so the only thing that can take its owners out of reach
    # is the terminal sweep.
    healthy = Payload(b"h" * 4096)
    healthy_alive = weakref.ref(healthy)
    dev.batched_write([0], [healthy], [4096], [None])
    # Wait for it to land before anything goes wrong. An entry still in
    # flight when the fatal submit happens is failed by that path instead,
    # which is a different rule being tested; the assertion on the owner
    # count below fails loudly if this wait was not enough.
    deadline = time.monotonic() + 10
    while not dev.is_idle() and time.monotonic() < deadline:
        time.sleep(0.01)

    # A second batch whose submit is fatal. That poisons the engine and
    # makes the close refuse -- for a reason that has nothing to do with
    # the first batch.
    doomed = Payload(b"d" * 4096)
    base = dev.submit_call_count()
    dev.inject_submit_faults([(base, "fatal:5")])
    dev.batched_write([4096], [doomed], [4096], [None])
    deadline = time.monotonic() + 10
    while not dev.is_poisoned() and time.monotonic() < deadline:
        time.sleep(0.01)

    report = {
        "poisoned": dev.is_poisoned(),
        # Nothing marked the healthy batch, so before the close only the
        # doomed one's owner has been moved.
        "owners_before_close": dev.quarantined_owner_count(),
    }
    del healthy
    del doomed
    gc.collect()
    try:
        dev.close()
        report["close"] = "returned"
    except BaseException as exc:
        report["close"] = type(exc).__name__
    report["retained_after_close"] = dev.retained_owner_count()
    del dev
    gc.collect()
    report["healthy_alive_after_drop"] = healthy_alive() is not None
    print(json.dumps(report))
    """
)


def test_a_refused_close_sweeps_every_batch_nobody_polled(tmp_path) -> None:
    """The sweep is not only for the batches that were marked.

    A batch can complete normally and never be polled, and then a close
    can refuse for an unrelated reason -- another batch's fatal submit, or
    a registration that would not come back. The registration the engine
    is retaining covers whatever memory it named, including that batch's
    buffers, so the terminal retention takes every batch rather than only
    the ones something had already objected to.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(64 * 1024 * 1024)
    report = _run_scenario(UNSWEPT_HEALTHY_OWNER_SCENARIO, device)
    assert report["poisoned"] is True
    assert report["close"] == "RuntimeError"
    # One marked, one not. The sweep is the only thing that can reach the
    # second, and the owner surviving the drop is what proves it did.
    assert report["owners_before_close"] == 1, report
    assert report["retained_after_close"] == 2, report
    assert report["healthy_alive_after_drop"] is True


def test_an_unpolled_batch_keeps_its_owners_through_close(tmp_path) -> None:
    """Nobody has to poll for the ownership rule to hold.

    A caller whose request was abandoned or cancelled, or whose engine is
    shutting down, never reaches wait_iouring. If a batch's owners moved out
    of reach only there, those callers leave them where the device's own
    destructor frees them -- returning their pool slices to an allocator on
    behalf of a close that has just refused, on the grounds that it could
    not establish what the device was doing.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(64 * 1024 * 1024)
    report = _run_scenario(UNPOLLED_OWNER_SCENARIO, device)

    assert report["poisoned"] is True
    assert report["quarantined_batches"] == 1
    # Moved when the worker learned, not when somebody asked.
    assert report["owners_without_polling"] == 1
    assert report["alive_after_our_reference_went"] is True
    assert report["close"] == "RuntimeError"
    assert report["alive_after_close_and_drop"] is True


def test_a_fatal_submit_poisons_the_engine_and_keeps_the_batch(tmp_path) -> None:
    """A submit that fails fatally leaves what the kernel took unknowable.

    The worker cannot say whether the entries it pushed were consumed, so the
    batch is quarantined with its owners and the device is poisoned before any
    waiter is told anything.
    """
    report = _run(tmp_path, "fatal:5")  # EIO
    assert report["poisoned"] is True
    assert report["quarantined_batches"] >= 1
    assert report["quarantined_owners"] >= 1
    assert report["results"] == [False]


def test_a_submit_that_takes_nothing_keeps_the_entries_resident(tmp_path) -> None:
    """Zero taken is not a failure: the work is still ours and is retried.

    Nothing may be quarantined for it, because nothing was handed over -- and
    treating it as fatal would poison an engine that is fine.
    """
    report = _run(tmp_path, "reports_zero_taken")
    assert report.get("results") == [True], report
    assert report["poisoned"] is False
    assert report["quarantined_batches"] == 0


@pytest.mark.parametrize("errno", [11, 4], ids=["eagain", "eintr"])
def test_a_retryable_submit_error_is_retried_not_quarantined(
    tmp_path, errno: int
) -> None:
    """EAGAIN and EINTR leave the ring unchanged, so the work is still ours."""
    report = _run(tmp_path, f"retryable:{errno}")
    assert report.get("results") == [True], report
    assert report["poisoned"] is False
    assert report["quarantined_batches"] == 0


def test_the_seam_refuses_to_state_a_partial_take(tmp_path) -> None:
    """A count it cannot make true is refused, not accepted and ignored.

    The substitution runs before the ring, so no entry was offered and any
    non-zero count would describe entries that never moved. A seam that
    accepted it would report a partial submit while reproducing the zero
    case -- a test passing against a state the engine was never in.
    """
    report = _run(tmp_path, "partially_taken:1")
    assert report.get("raised", "").startswith("ValueError"), report
    assert "before the ring" in report["raised"]
    assert "reports_zero_taken" in report["raised"]
    assert report["poisoned"] is False
    assert report["quarantined_batches"] == 0


# ---------------------------------------------------------------------------
# The scripted ring.
#
# The submit seam above substitutes what one submit *call reports*, before
# the ring, so it can never deliver a completion -- and the transitions that
# only a completion produces stay out of reach: a short read, a completion
# that never arrives, a request the kernel took and never answered for.
# These drive a ring with no kernel behind it instead, so a completion can
# be delivered for a chosen request or deliberately withheld.
#
# Every one of them also asserts that the ring recorded no ownership
# violation. That list is how the ring refuses: answering for a request it
# does not hold is a fabrication, and a test that passes while fabricating
# one has proved nothing.
# ---------------------------------------------------------------------------

SCRIPTED_PREAMBLE = """
import gc, json, os, sys, tempfile, time, weakref
sys.path.insert(0, sys.argv[1])
from lmcache_rust_raw_block_io import RawBlockDevice

def device(capacity=8):
    return RawBlockDevice(
        sys.argv[2], writable=True, use_iouring=True, use_odirect=False,
        alignment=4096, iouring_queue_depth=8, fake_ring_capacity=capacity,
    )

def wait_until(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()
"""


SCRIPTED_SHORT_COMPLETION = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    dev = device()
    payload = bytearray(b"x" * 4096)
    batch = dev.batched_write([0], [payload], [4096], [None])
    assert wait_until(lambda: bool(dev.fake_owned()))
    first = dev.fake_owned()

    # Half the bytes. A block that is half written is not written.
    dev.fake_complete(first[0], 2048)
    assert wait_until(lambda: bool(dev.fake_owned()))
    remainder = dev.fake_owned()
    dev.fake_complete(remainder[0], 2048)

    results, errors = dev.wait_iouring(batch)
    print(json.dumps({
        "first": first,
        "remainder": remainder,
        "results": list(results),
        "errors": [str(e) for _, e in errors],
        "violations": dev.fake_violations(),
    }))
    """
)


SCRIPTED_UNANSWERED_CLOSE = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    dev = device()

    class Payload(bytearray):
        pass

    payload = Payload(b"x" * 4096)
    alive = weakref.ref(payload)
    dev.batched_write([0], [payload], [4096], [None])
    assert wait_until(lambda: bool(dev.fake_owned()))
    report = {"owned_before_close": dev.fake_owned()}
    del payload
    gc.collect()
    try:
        dev.close()
        report["close"] = "returned"
    except BaseException as exc:
        report["close"] = type(exc).__name__
    report["retained"] = dev.retained_owner_count()
    report["violations"] = dev.fake_violations()
    del dev
    gc.collect()
    report["alive_after_close_and_drop"] = alive() is not None
    print(json.dumps(report))
    """
)


SCRIPTED_SUBMISSION_QUEUE_FULL = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    # One entry fits. The second push is refused, which is the only way a
    # real submission queue reports being full.
    dev = device(capacity=1)
    a = bytearray(b"a" * 4096)
    b = bytearray(b"b" * 4096)
    report = {}
    try:
        batch = dev.batched_write([0, 4096], [a, b], [4096, 4096], [None, None])
        report["submitted"] = True
        assert wait_until(lambda: bool(dev.fake_owned()))
        for held in list(dev.fake_owned()):
            dev.fake_complete(held, 4096)
        results, errors = dev.wait_iouring(batch)
        report["results"] = list(results)
        report["errors"] = [str(e) for _, e in errors]
    except BaseException as exc:
        report["submitted"] = False
        report["raised"] = f"{type(exc).__name__}: {exc}"
    report["violations"] = dev.fake_violations()
    report["poisoned"] = dev.is_poisoned()
    print(json.dumps(report))
    """
)


SCRIPTED_PARTIAL_TAKE = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    dev = device()
    a = bytearray(b"a" * 4096)
    b = bytearray(b"b" * 4096)
    # The ring really hands over a prefix this time, and the suffix really
    # stays the worker's.
    dev.fake_submit_takes(1)
    batch = dev.batched_write([0, 4096], [a, b], [4096, 4096], [None, None])
    assert wait_until(lambda: len(dev.fake_owned()) == 2)
    owned = dev.fake_owned()
    for held in list(owned):
        dev.fake_complete(held, 4096)
    results, errors = dev.wait_iouring(batch)
    print(json.dumps({
        "owned": owned,
        "delivered_once": len(owned) == len(set(owned)),
        "results": list(results),
        "errors": [str(e) for _, e in errors],
        "syncs": dev.fake_syncs(),
        "violations": dev.fake_violations(),
    }))
    """
)


SCRIPTED_INVENTED_COMPLETION = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    # The ring's own refusal, which is what the other cases rest on.
    dev = device()
    report = {}
    try:
        dev.fake_complete(999, 4096)
        report["refused"] = False
    except BaseException as exc:
        report["refused"] = True
        report["reason"] = str(exc)
    report["violations"] = dev.fake_violations()
    print(json.dumps(report))
    """
)


def test_a_short_completion_writes_the_remainder(tmp_path) -> None:
    """A block that is half written is not written.

    Only a completion can say how many bytes moved, so this is the first
    transition the submit seam could not reach at all. The engine pushes the
    remainder for the same logical request rather than reporting the write
    done, and the request is answered for once at each stage.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_SHORT_COMPLETION, device)

    assert report["results"] == [True]
    assert report["errors"] == []
    assert report["first"] == report["remainder"], (
        "the remainder belongs to the same request, not a new one"
    )
    assert report["violations"] == []


def test_a_request_the_kernel_never_answered_for_blocks_the_close(tmp_path) -> None:
    """The case the whole retention rule exists for, driven end to end.

    A request the ring has taken and not answered for is memory the device
    may still be reaching. The close refuses, the owner is retained, and the
    buffer outlives both the close and the device -- and the ring recorded
    no attempt to invent the completion that would have made this look fine.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_UNANSWERED_CLOSE, device)

    assert len(report["owned_before_close"]) == 1
    assert report["close"] == "RuntimeError"
    assert report["retained"] == 1
    assert report["alive_after_close_and_drop"] is True
    assert report["violations"] == []


def test_a_full_submission_queue_is_reported_not_panicked_on(tmp_path) -> None:
    """A push the ring refuses is an error, not a panic.

    The ring has room for one entry and is given two. Whatever the engine
    does with the second, it must not claim the device accepted it, and it
    must not take the process down -- the push error on this path used to be
    an `expect`.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_SUBMISSION_QUEUE_FULL, device)

    assert report["violations"] == []
    if report["submitted"]:
        # The entry that did not fit must not be reported as written.
        assert report["results"].count(True) <= 1, report
    else:
        assert "raised" in report


def test_a_partial_take_delivers_the_suffix_once(tmp_path) -> None:
    """An entry the ring did not take is still resident, and still ours.

    Requeuing it would issue the same I/O twice and point the resident copy
    at a buffer whose owner has been released. So the suffix reaches the
    kernel on a later submit, as the same request, exactly once.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_PARTIAL_TAKE, device)

    assert report["delivered_once"] is True, report
    assert len(report["owned"]) == 2
    assert report["results"] == [True, True]
    assert report["errors"] == []
    assert report["syncs"] > 0, "the worker must have flushed the ring"
    assert report["violations"] == []


def test_the_scripted_ring_refuses_a_completion_it_does_not_hold(tmp_path) -> None:
    """The refusal the other scripted cases rest on.

    If the ring answered for anything it was asked to, the empty violation
    lists above would mean nothing. It refuses, and records the attempt.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_INVENTED_COMPLETION, device)

    assert report["refused"] is True
    assert "does not hold" in report["reason"]
    assert len(report["violations"]) == 1
