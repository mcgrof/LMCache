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
    assert not hasattr(serving.RawBlockDevice, "retire_batch_owners"), (
        "a caller must not be able to retire a batch before native completion"
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


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_partial_dmabuf_registration_cleanup_controls_reuse(
    tmp_path: Path, cleanup_fails: bool
) -> None:
    """A failed slot update permits reuse only after its table is withdrawn."""
    scenario = SCRIPTED_PREAMBLE + textwrap.dedent(
        f"""
        dev = device()
        dev.fake_registration_fails("dmabuf", 1, 22)
        if {cleanup_fails!r}:
            dev.fake_registration_fails("unregister", 0, 5)
        try:
            dev.register_fixed_dmabufs([4096, 8192], [4096, 4096], [7, 8], [0, 0])
            raise AssertionError("slot update unexpectedly succeeded")
        except RuntimeError as exc:
            error = str(exc)
        report = {{"error": error, "poisoned": dev.is_poisoned(),
                   "idle": dev.is_idle(), "table": dev.fake_registrations()}}
        if {cleanup_fails!r}:
            refused = []
            for call in (
                lambda: dev.register_fixed_buffers([4096], [4096]),
                lambda: dev.register_fixed_dmabufs([4096], [4096], [7], [0]),
                lambda: dev.batched_write([0], [bytearray(4096)], [4096]),
            ):
                try:
                    call()
                    refused.append(False)
                except RuntimeError:
                    refused.append(True)
            report["refused"] = refused
            closes = []
            for _ in range(2):
                try:
                    dev.close()
                    closes.append(False)
                except RuntimeError:
                    closes.append(True)
            report["close_retained"] = closes
            report["table_after_close"] = dev.fake_registrations()
        else:
            dev.register_fixed_dmabufs([4096], [4096], [7], [0])
            report["retry_table"] = dev.fake_registrations()
            dev.close()
            report["table_after_close"] = dev.fake_registrations()
        print(json.dumps(report))
        """
    )
    target = tmp_path / "device.bin"
    target.write_bytes(bytes(16384))
    report = _run_scenario(scenario, target)
    assert "slot 1" in report["error"]
    assert report["poisoned"] is cleanup_fails
    assert report["idle"] is not cleanup_fails
    if cleanup_fails:
        assert "unregister also failed" in report["error"]
        assert [record["kind"] for record in report["table"]] == [1, 2]
        assert report["refused"] == [True] * 3
        assert report["close_retained"] == [True, True]
        assert report["table_after_close"] == report["table"]
    else:
        assert [record["kind"] for record in report["table"]] == [3]
        assert [record["kind"] for record in report["retry_table"]] == [3, 1, 2]
        assert [record["kind"] for record in report["table_after_close"]] == [3]


def test_registered_table_cannot_be_replaced_or_registered_after_close(
    tmp_path: Path,
) -> None:
    """A rejected replacement preserves the live table used by fixed I/O."""
    scenario = SCRIPTED_PREAMBLE + textwrap.dedent(
        """
        dev = device()
        dev.register_fixed_dmabufs([4096], [4096], [7], [0])
        original = dev.fake_registrations()
        errors = []
        for call in (
            lambda: dev.register_fixed_buffers([8192], [4096]),
            lambda: dev.register_fixed_dmabufs([8192], [4096], [8], [0]),
        ):
            try:
                call()
            except RuntimeError as exc:
                errors.append(str(exc))
        unchanged = dev.fake_registrations() == original
        dev.close()
        try:
            dev.register_fixed_dmabufs([4096], [4096], [7], [0])
        except RuntimeError as exc:
            errors.append(str(exc))
        print(json.dumps({"errors": errors, "unchanged": unchanged}))
        """
    )
    target = tmp_path / "device.bin"
    target.write_bytes(bytes(8192))
    report = _run_scenario(scenario, target)
    assert report["unchanged"] is True
    assert len(report["errors"]) == 3
    assert all("already registered" in error for error in report["errors"][:2])
    assert "closed" in report["errors"][2]


@pytest.mark.parametrize("direction", ["read", "write"])
@pytest.mark.parametrize("complete", [False, True])
def test_batched_io_pins_resizable_host_buffer_until_known_completion(
    tmp_path: Path, direction: str, complete: bool
) -> None:
    """An exporter reference alone must not permit its backing to be moved."""
    scenario = SCRIPTED_PREAMBLE + textwrap.dedent(
        f"""
        dev = device()
        payload = bytearray(4096)
        batch = dev.batched_{direction}([0], [payload], [4096])
        assert wait_until(lambda: bool(dev.fake_owned()))
        def can_resize():
            try:
                payload.extend(b"x")
                return True
            except BufferError:
                return False
        before = can_resize()
        if {complete!r}:
            dev.fake_complete(dev.fake_owned()[0], 4096)
            assert dev.wait_iouring(batch) == ([True], [])
            dev.close()
        else:
            try:
                dev.close()
            except RuntimeError:
                pass
            assert dev.is_poisoned()
            assert dev.wait_iouring(batch)[0] == [False]
        after = can_resize()
        del dev
        gc.collect()
        print(json.dumps({{"before": before, "after": after,
                          "after_drop": can_resize()}}))
        """
    )
    target = tmp_path / "device.bin"
    target.write_bytes(bytes(8192))
    report = _run_scenario(scenario, target)
    assert report["before"] is False
    assert report["after"] is complete
    assert report["after_drop"] is complete


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
    # real submission queue reports being full. Submits are held so the
    # first entry is still resident when the second arrives -- otherwise
    # the first is gone by then and there is no full queue to report.
    dev = device(capacity=1)
    dev.fake_hold_submits(True)
    a = bytearray(b"a" * 4096)
    b = bytearray(b"b" * 4096)
    batch = dev.batched_write([0, 4096], [a, b], [4096, 4096], [None, None])
    assert wait_until(lambda: len(dev.fake_resident()) == 1)
    report = {"resident_while_full": dev.fake_resident_sqes()}

    # Let the queue drain and answer for whatever the ring took.
    dev.fake_hold_submits(False)
    assert wait_until(lambda: bool(dev.fake_owned()))
    for held in list(dev.fake_owned()):
        dev.fake_complete(held, 4096)
    assert wait_until(lambda: not dev.fake_owned() and not dev.fake_resident())
    results, errors = dev.wait_iouring(batch)
    report["results"] = list(results)
    report["errors"] = [str(e) for _, e in errors]
    report["journal"], report["dropped"] = dev.take_io_journal()
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
    # Hold the submits first, so both entries are resident at once. Without
    # that, batched_write signals each item separately and the worker can
    # make two submissions of one entry each -- where "take one" is a full
    # take and nothing partial has happened.
    dev.fake_hold_submits(True)
    batch = dev.batched_write([0, 4096], [a, b], [4096, 4096], [None, None])
    assert wait_until(lambda: len(dev.fake_resident()) == 2)
    resident_before = dev.fake_resident_sqes()

    # Now one real partial take: the ring hands over a prefix, and the
    # suffix really stays the worker's. The hold stays on, so this is the
    # only submit that takes anything and the suffix can be looked at
    # rather than raced against the worker's next loop.
    dev.fake_submit_takes(1)
    assert wait_until(lambda: len(dev.fake_owned()) == 1)
    accepted = dev.fake_owned_sqes()
    suffix_while_held = dev.fake_resident_sqes()
    dev.fake_complete(accepted[0]["user_data"], 4096)

    # The suffix reaches the kernel on a later submit, unchanged.
    dev.fake_hold_submits(False)
    assert wait_until(lambda: len(dev.fake_owned()) == 1)
    suffix = dev.fake_owned_sqes()
    dev.fake_complete(suffix[0]["user_data"], 4096)
    results, errors = dev.wait_iouring(batch)
    print(json.dumps({
        "journal": dev.take_io_journal()[0],
        "resident_before": resident_before,
        "accepted": accepted,
        "suffix_while_held": suffix_while_held,
        "suffix": suffix,
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
    must not take the process down: a full submission queue is an ordinary
    condition on this path, and the only one that reports it is the push.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_SUBMISSION_QUEUE_FULL, device)

    assert report["violations"] == []
    assert report["poisoned"] is False, report
    # The queue really was full: one entry resident against a capacity of
    # one, which is what makes the second push the refusal under test.
    assert len(report["resident_while_full"]) == 1, report
    assert report["resident_while_full"][0]["len"] == 4096

    # Both operations end, and each ends as itself: the one the ring took is
    # written, and the one it refused is reported as an error against its
    # own index rather than quietly dropped or counted as written.
    assert len(report["results"]) == 2, report
    assert report["results"].count(True) == 1, report
    refused = [index for index, ok in enumerate(report["results"]) if not ok]
    assert len(refused) == 1, report
    assert any("submission queue full" in error for error in report["errors"]), report
    assert report["dropped"] == 0, report
    assert [row["outcome"] for row in report["journal"]] == [
        "submitted",
        "completed",
    ], report


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

    assert report["violations"] == []
    # A real partial take: strictly between nothing and everything. "Take
    # one of one" is a full take, and a test that accepted it would pass
    # against a worker that never partially submitted anything.
    resident_before = report["resident_before"]
    assert len(resident_before) == 2, report
    assert len(report["accepted"]) == 1, report
    assert 0 < len(report["accepted"]) < len(resident_before), report

    # The prefix is the first entry as it was built, and the suffix is the
    # second -- not a re-derived operation that happens to carry the same
    # bytes. Repeated payload bytes make a readback comparison agree with a
    # remainder aimed anywhere, so the geometry is what is compared.
    assert report["accepted"][0] == resident_before[0], report
    assert report["suffix_while_held"] == [resident_before[1]], report
    assert report["suffix"] == [resident_before[1]], report
    assert report["accepted"][0]["user_data"] != report["suffix"][0]["user_data"]
    operations: dict[str, list[str]] = {}
    for row in report["journal"]:
        operations.setdefault(row["operation_id"], []).append(row["outcome"])
    assert len(operations) == 2, report
    assert all(events == ["submitted", "completed"] for events in operations.values())

    assert report["results"] == [True, True]
    assert report["errors"] == []
    assert report["syncs"] > 0, "the worker must have flushed the ring"


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


SCRIPTED_RETRYABLE_THEN_WITHHELD = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    dev = device()
    payload = bytearray(b"x" * 4096)

    # A submit that reports EAGAIN takes nothing, so the entry is still
    # resident and still the worker's. The worker retries, the retry lands,
    # and then the completion is withheld -- which is a different thing
    # again from a submit that failed.
    dev.fake_submit_fails(11)
    dev.batched_write([0], [payload], [4096], [None])
    assert wait_until(lambda: bool(dev.fake_owned())), dev.fake_resident()
    owned = dev.fake_owned()
    report = {
        "owned_after_the_retry": owned,
        "resident_after_the_retry": dev.fake_resident(),
        "poisoned_before_close": dev.is_poisoned(),
    }
    try:
        dev.close()
        report["close"] = "returned"
    except BaseException as exc:
        report["close"] = type(exc).__name__
    report["violations"] = dev.fake_violations()
    print(json.dumps(report))
    """
)


SCRIPTED_SYNCHRONOUS_WRITE = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    import threading

    dev = device()
    payload = bytearray(b"x" * 8192)
    done = {}

    def answer_it():
        # The synchronous route waits inside the call, so its completion has
        # to come from somewhere else.
        assert wait_until(lambda: bool(dev.fake_owned()))
        held = dev.fake_owned()
        done["owned"] = held
        dev.fake_complete(held[0], 4096)

    answering = threading.Thread(target=answer_it)
    answering.start()
    dev.write_uring(0, payload, 4095, 4096, None)
    answering.join(timeout=10)
    print(json.dumps({
        "owned_while_writing": done.get("owned"),
        "owned_after": dev.fake_owned(),
        "violations": dev.fake_violations(),
    }))
    """
)


def test_a_retryable_submit_leaves_the_entry_ours_and_then_lands(tmp_path) -> None:
    """Three different states, told apart.

    A submit reporting EAGAIN took nothing, so the entry is resident and
    still the worker's. The retry hands it to the kernel. The completion
    then never arrives, which is neither of the first two: the kernel holds
    it, and the close has to refuse rather than conclude anything.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_RETRYABLE_THEN_WITHHELD, device)

    assert len(report["owned_after_the_retry"]) == 1, report
    assert report["resident_after_the_retry"] == [], (
        "the retry handed it over; nothing should still be waiting to go"
    )
    assert report["poisoned_before_close"] is False, (
        "a retryable submit is not an unknown outcome"
    )
    assert report["close"] == "RuntimeError"
    assert report["violations"] == []


def test_a_synchronous_write_owns_its_request_until_it_is_answered(tmp_path) -> None:
    """The route that waits inside the call reaches the ring too.

    A padded O_DIRECT payload cannot be expressed as one batched length, so
    it goes one write at a time and waits for its own completion. That route
    has no batch to poll, which is exactly why its ownership had to be
    checked against a ring that can be asked what it holds.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_SYNCHRONOUS_WRITE, device)

    assert report["owned_while_writing"] is not None, report
    assert len(report["owned_while_writing"]) == 1
    assert report["owned_after"] == [], "the answer retired the request"
    assert report["violations"] == []


SCRIPTED_WITHHELD_READ = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    dev = device()

    class Destination(bytearray):
        pass

    # A read's destination is the buffer the device writes *into*. A
    # completion that never arrives leaves the device possibly still
    # writing there, which is the same unknown as a write -- and the one
    # that matters more, because the destination is a pool slice an
    # allocator would otherwise hand to the next request.
    destination = Destination(b"\\x00" * 4096)
    alive = weakref.ref(destination)
    dev.batched_read([0], [destination], [4096])
    assert wait_until(lambda: bool(dev.fake_owned()))
    report = {"owned_before_close": dev.fake_owned()}
    del destination
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
    report["destination_alive_after_close_and_drop"] = alive() is not None
    print(json.dumps(report))
    """
)


def test_a_read_nobody_answered_for_keeps_its_destination(tmp_path) -> None:
    """A read's destination is where the device writes, and it is a pool slice.

    A withheld write completion leaves bytes possibly still leaving a
    buffer; a withheld *read* completion leaves bytes possibly still
    arriving in one. The second is the worse of the two, because the
    destination goes back to an allocator that hands it to the next
    request -- which then reads whatever the device finished writing there.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_WITHHELD_READ, device)

    assert len(report["owned_before_close"]) == 1, report
    assert report["close"] == "RuntimeError"
    assert report["retained"] == 1
    assert report["destination_alive_after_close_and_drop"] is True
    assert report["violations"] == []


SCRIPTED_FATAL_AFTER_WITHHELD = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    dev = device()
    first = bytearray(b"a" * 4096)
    second = bytearray(b"b" * 4096)

    # Work the kernel took and will never answer for.
    earlier = dev.batched_write([0], [first], [4096], [None])
    assert wait_until(lambda: len(dev.fake_owned()) == 1)
    withheld = dev.fake_owned_sqes()

    # Now a submit that fails in a way that says nothing about what the ring
    # already holds. EIO is not EAGAIN: there is no retry that makes this
    # knowable.
    dev.fake_submit_fails(5)
    later = dev.batched_write([4096], [second], [4096], [None])

    # Both waiters have to be answered. The earlier one is the point: its
    # completion is never coming, and before this the only thing that woke
    # it was somebody closing the device from outside.
    #
    # Waited on from a thread with a bound, because an unanswered waiter is
    # exactly what this is testing for: blocking here instead would report a
    # timeout, and a timeout says nothing about which rule was broken.
    import threading

    answered = {}

    def collect(name, handle):
        try:
            results, errors = dev.wait_iouring(handle)
            answered[name] = (list(results), [str(e) for _, e in errors])
        except BaseException as exc:
            answered[name] = ("raised", f"{type(exc).__name__}: {exc}")

    # Daemons: a waiter that is never answered is the defect under test,
    # and a non-daemon thread stuck in it would keep this process alive at
    # exit -- which reports a timeout instead of the rule that was broken.
    waiters = [
        threading.Thread(target=collect, args=("later", later), daemon=True),
        threading.Thread(target=collect, args=("earlier", earlier), daemon=True),
    ]
    for waiter in waiters:
        waiter.start()
    for waiter in waiters:
        waiter.join(timeout=15)

    report = {
        "withheld": withheld,
        "earlier_answered": "earlier" in answered,
        "later_answered": "later" in answered,
        "earlier": answered.get("earlier"),
        "later": answered.get("later"),
        "poisoned": dev.is_poisoned(),
        "quarantined_batches": dev.quarantined_batch_count(),
        "quarantined_owners": dev.quarantined_owner_count(),
        "still_owned": dev.fake_owned_sqes(),
    }
    try:
        dev.close()
        report["close"] = "returned"
    except BaseException as exc:
        report["close"] = type(exc).__name__
    report["retained"] = dev.retained_owner_count()
    report["violations"] = dev.fake_violations()
    print(json.dumps(report))
    sys.stdout.flush()
    # Leave without running interpreter shutdown: a thread deliberately
    # left waiting on an unanswered request would be joined there, and this
    # scenario has already said everything it has to say.
    os._exit(0)
    """
)


SCRIPTED_SYNCHRONOUS_UNANSWERED = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    import threading

    dev = device()
    payload = bytearray(b"x" * 8192)
    other = bytearray(b"y" * 4096)
    outcome = {}

    def write_and_wait():
        # The synchronous route waits inside the call. Nothing answers this
        # request, so what wakes it has to be the engine deciding it cannot
        # say -- not a completion anybody invented.
        try:
            dev.write_uring(0, payload, 4095, 4096, None)
            outcome["raised"] = None
        except BaseException as exc:
            outcome["raised"] = type(exc).__name__

    writing = threading.Thread(target=write_and_wait, daemon=True)
    writing.start()
    assert wait_until(lambda: len(dev.fake_owned()) == 1)
    owned_while_writing = dev.fake_owned_sqes()

    # A fatal submit on different work. The engine can no longer describe
    # what the kernel holds, including the request above.
    dev.fake_submit_fails(5)

    def drive_the_fatal_submit():
        try:
            dev.wait_iouring(dev.batched_write([8192], [other], [4096], [None]))
        except BaseException:
            pass

    driving = threading.Thread(target=drive_the_fatal_submit, daemon=True)
    driving.start()
    driving.join(timeout=15)
    writing.join(timeout=15)

    report = {
        "owned_while_writing": owned_while_writing,
        "writer_finished": not writing.is_alive(),
        "writer_raised": outcome.get("raised"),
        "poisoned": dev.is_poisoned(),
        "quarantined_owners": dev.quarantined_owner_count(),
    }
    try:
        dev.close()
        report["close"] = "returned"
    except BaseException as exc:
        report["close"] = type(exc).__name__
    report["retained"] = dev.retained_owner_count()
    report["violations"] = dev.fake_violations()
    print(json.dumps(report))
    """
)


SCRIPTED_DMABUF_SHORT_TERMINAL = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    import ctypes

    dev = device()
    payload = bytearray(b"x" * 8192)
    address = ctypes.addressof((ctypes.c_char * len(payload)).from_buffer(payload))
    # A nonzero offset inside the registration, which is the case a
    # remainder would get wrong: the SQE address of a registered dma-buf
    # transfer is an offset into the registration, not a process address.
    base = address - 4096
    dev.register_fixed_dmabufs([address], [len(payload)], [7], [base])
    registrations = dev.fake_registrations()

    batch = dev.batched_write([0], [payload], [8192], [None])
    assert wait_until(lambda: len(dev.fake_owned()) == 1)
    submitted = dev.fake_owned_sqes()

    # Half the bytes, and known to be half: a positive short completion.
    dev.fake_complete(submitted[0]["user_data"], 4096)

    # Look at the ring before waiting on the batch. A remainder nobody is
    # going to answer for makes that wait never return, and a test that
    # times out has reported nothing at all.
    remainder = wait_until(
        lambda: bool(dev.fake_resident()) or bool(dev.fake_owned()), seconds=2
    )
    report = {
        "registrations": registrations,
        "submitted": submitted,
        "remainder_appeared": remainder,
        "resident_after": dev.fake_resident_sqes(),
        "owned_after": dev.fake_owned_sqes(),
    }
    if not remainder:
        results, errors = dev.wait_iouring(batch)
        report["results"] = list(results)
        report["errors"] = [str(e) for _, e in errors]
    report["poisoned"] = dev.is_poisoned()
    report["quarantined_owners"] = dev.quarantined_owner_count()
    report["violations"] = dev.fake_violations()
    print(json.dumps(report))
    """
)


def test_a_fatal_submit_answers_work_the_kernel_already_took(tmp_path) -> None:
    """A fatal submit is not news about one batch.

    A request the kernel took earlier, and will now never answer for, has a
    waiter. Failing only the batch whose submit reported the error leaves
    that waiter blocked until somebody closes the device from outside --
    which is not an answer, it is a hang with a cause.

    So every unresolved operation gets a bounded logical failure, and every
    one of them keeps its owners: a logical failure is not a DMA fence, and
    nothing here invents a completion or declares the device finished.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_FATAL_AFTER_WITHHELD, device)

    assert report["violations"] == []
    assert len(report["withheld"]) == 1, report

    # The earlier waiter was answered, and answered as a failure.
    assert report["earlier_answered"] is True, report
    assert report["later_answered"] is True, report
    assert report["earlier"][0] == [False], report
    assert report["earlier"][1], report
    assert report["later"][0] == [False], report

    # And nothing was released on the strength of that answer.
    assert report["poisoned"] is True, report
    assert report["quarantined_batches"] == 2, report
    assert report["quarantined_owners"] >= 2, report
    assert report["close"] == "RuntimeError", report
    assert report["retained"] >= 2, report


def test_a_synchronous_write_nobody_answered_for_is_woken_and_kept(tmp_path) -> None:
    """The route with no batch to poll still has to be answered.

    A synchronous write waits inside its own call, so an unresolved outcome
    there is a thread that never returns. The engine wakes it by deciding it
    cannot say -- not by inventing the completion it is waiting for -- and
    keeps the buffer, because a logical failure says nothing about whether
    the device is still reaching it.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_SYNCHRONOUS_UNANSWERED, device)

    assert report["violations"] == []
    assert len(report["owned_while_writing"]) == 1, report
    assert report["writer_finished"] is True, report
    assert report["writer_raised"] is not None, (
        "a write nobody answered for must not report success"
    )
    assert report["poisoned"] is True, report
    assert report["quarantined_owners"] >= 1, report
    assert report["close"] == "RuntimeError", report
    assert report["retained"] >= 1, report


def test_a_short_registered_dmabuf_transfer_ends_there(tmp_path) -> None:
    """A registered dma-buf transfer has no remainder to retry.

    The SQE address of one is a byte offset inside the registration rather
    than a process address, so advancing it by the bytes transferred aims
    the next submission at an offset nobody registered. There is also no
    bounce buffer to fall back to: the whole point of the registration is
    that the device reaches that memory directly.

    So a known short completion ends the operation. The registration here
    is explicitly synthetic -- no descriptor reached a kernel -- which makes
    this evidence about the engine's own decision and about nothing else.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_DMABUF_SHORT_TERMINAL, device)

    assert report["violations"] == []
    # The registration really went through the shared path: a sparse table
    # and one dma-buf slot.
    kinds = [record["kind"] for record in report["registrations"]]
    assert kinds == [1, 2], report

    # The submission carried the registration, at a nonzero offset inside
    # it, with the registered index rather than a process address.
    assert len(report["submitted"]) == 1, report
    sqe = report["submitted"][0]
    assert sqe["fixed_index"] == 0, report
    assert sqe["dmabuf_offset"] == 4096, report
    assert sqe["addr"] == 4096, report
    assert sqe["len"] == 8192, report

    # It ended there: no remainder was submitted and none is waiting to be.
    assert report["remainder_appeared"] is False, report
    assert report["resident_after"] == [], report
    assert report["owned_after"] == [], report
    assert report["results"] == [False], report
    assert report["errors"], report
    # A known short completion is a known outcome. The device answered for
    # this request, so nothing is withheld on its account.
    assert report["poisoned"] is False, report
    assert report["quarantined_owners"] == 0, report


SCRIPTED_SHORT_REMAINDER_RETRIED = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    dev = device()
    payload = bytearray(b"x" * 8192)
    batch = dev.batched_write([0], [payload], [8192], [None])
    assert wait_until(lambda: bool(dev.fake_owned()))
    first = dev.fake_owned_sqes()

    # Half the bytes. The engine resubmits the rest, and that submit is
    # refused with EAGAIN -- which took nothing, so the remainder is still
    # resident and still the worker's. The hold keeps it there long enough
    # to be looked at; without it the retry lands before anything can see
    # which geometry was pending.
    dev.fake_hold_submits(True)
    dev.fake_submit_fails(11)
    dev.fake_complete(first[0]["user_data"], 4096)
    assert wait_until(lambda: bool(dev.fake_resident()))
    assert wait_until(lambda: dev.fake_submit_errors() == [11])
    resident_after_refusal = dev.fake_resident_sqes()
    report = {
        "first": first,
        "resident_after_refusal": resident_after_refusal,
        "submit_errors": dev.fake_submit_errors(),
        "poisoned_after_refusal": dev.is_poisoned(),
    }

    # The retry hands the same remainder over, and it lands.
    dev.fake_hold_submits(False)
    assert wait_until(lambda: bool(dev.fake_owned()))
    remainder = dev.fake_owned_sqes()
    dev.fake_complete(remainder[0]["user_data"], 4096)
    results, errors = dev.wait_iouring(batch)
    journal, dropped = dev.take_io_journal()
    report.update({
        "journal": journal,
        "dropped": dropped,
        "remainder": remainder,
        "results": list(results),
        "errors": [str(e) for _, e in errors],
        "poisoned": dev.is_poisoned(),
        "quarantined_owners": dev.quarantined_owner_count(),
        "violations": dev.fake_violations(),
    })
    print(json.dumps(report))
    sys.stdout.flush()
    os._exit(0)
    """
)


def test_a_refused_remainder_submit_is_retried_and_lands(tmp_path) -> None:
    """A short write's remainder meets a submit that took nothing.

    Two transitions the inherited matrix names, in the order they happen:
    the remainder the engine builds from a positive short completion, and a
    retryable submit error on *that* submission rather than on an initial
    one. A retryable error consumed nothing, so the remainder is resident
    and still this engine's, and the retry delivers the same geometry.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    report = _run_scenario(SCRIPTED_SHORT_REMAINDER_RETRIED, device)

    assert report["violations"] == []
    # The remainder names the second half, at the offset it belongs at.
    assert len(report["resident_after_refusal"]) == 1, report
    pending = report["resident_after_refusal"][0]
    assert pending["offset"] == 4096, report
    assert pending["len"] == 4096, report
    assert report["submit_errors"] == [11], report
    # A retryable submit error is not an unknown outcome.
    assert report["poisoned_after_refusal"] is False, report
    assert report["remainder"] == report["resident_after_refusal"], report
    assert report["dropped"] == 0, report
    journal = report["journal"]
    assert [row["outcome"] for row in journal] == [
        "submitted",
        "short",
        "submitted",
        "completed",
    ], report
    assert [int(row["bytes"]) for row in journal] == [8192, 4096, 4096, 4096], report
    assert [int(row["attempt"]) for row in journal] == [0, 0, 1, 1], report
    assert len({row["operation_id"] for row in journal}) == 1, report

    assert report["results"] == [True], report
    assert report["errors"] == [], report
    assert report["poisoned"] is False, report
    assert report["quarantined_owners"] == 0, report


SCRIPTED_COMPLETION_DURING_CLOSE = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    class Buffer(bytearray):
        pass

    dev = device()
    payload = Buffer(b"x" * 4096)
    owner = weakref.ref(payload)
    batch = dev.batched_write([0], [payload], [4096], [None])
    del payload
    assert wait_until(lambda: len(dev.fake_owned()) == 1)
    held = dev.fake_owned()[0]
    dev.fake_complete_at_shutdown(held, COMPLETION_RESULT)
    gc.collect()
    before = {
        "owned": dev.fake_owned(),
        "delivered": dev.fake_shutdown_delivered(),
        "owner_alive": owner() is not None,
    }
    dev.close()
    results, errors = dev.wait_iouring(batch)
    gc.collect()
    journal, dropped = dev.take_io_journal()
    print(json.dumps({
        "journal": journal,
        "dropped": dropped,
        "before": before,
        "delivered": dev.fake_shutdown_delivered(),
        "owned": dev.fake_owned(),
        "resident": dev.fake_resident(),
        "results": list(results),
        "errors": [str(e) for _, e in errors],
        "poisoned": dev.is_poisoned(),
        "retained": dev.retained_owner_count(),
        "owner_alive_after_poll": owner() is not None,
        "violations": dev.fake_violations(),
    }))
    """
)


@pytest.mark.parametrize("result", [4096, 2048, -5], ids=["full", "short", "error"])
def test_a_completion_during_close_is_reaped_before_releasing_its_owner(
    tmp_path: Path, result: int
) -> None:
    """Drain a known completion after shutdown begins without a timing race."""
    device = tmp_path / "dev.bin"
    device.write_bytes(bytes(8192))
    report = _run_scenario(
        SCRIPTED_COMPLETION_DURING_CLOSE.replace("COMPLETION_RESULT", str(result)),
        device,
    )

    assert len(report["before"]["owned"]) == 1, report
    assert report["before"]["delivered"] == 0, report
    assert report["before"]["owner_alive"] is True, report
    assert report["delivered"] == 1, report
    assert report["owned"] == report["resident"] == [], report
    assert report["results"] == [result == 4096], report
    assert bool(report["errors"]) == (result != 4096), report
    assert report["poisoned"] is False, report
    assert report["retained"] == 0, report
    assert report["owner_alive_after_poll"] is False, report
    assert report["violations"] == [], report
    assert report["dropped"] == 0, report
    journal = report["journal"]
    outcome = "completed" if result == 4096 else "short" if result >= 0 else "failed"
    assert [row["outcome"] for row in journal] == ["submitted", outcome], report
    assert [int(row["bytes"]) for row in journal] == [4096, result], report
    assert len({(row["operation_id"], row["attempt"]) for row in journal}) == 1, report


SCRIPTED_FATAL_SHORT_REMAINDER = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    import threading

    class Buffer(bytearray):
        pass

    dev = device()
    payload = Buffer(b"x" * 8192)
    owner = weakref.ref(payload)
    batch = dev.batched_write([0], [payload], [8192], [None])
    del payload
    assert wait_until(lambda: len(dev.fake_owned()) == 1)
    first = dev.fake_owned_sqes()[0]
    dev.fake_submit_fails(5)
    dev.fake_complete(first["user_data"], 4096)

    outcome = {}
    def collect():
        results, errors = dev.wait_iouring(batch)
        outcome["results"] = list(results)
        outcome["errors"] = [str(e) for _, e in errors]

    waiting = threading.Thread(target=collect, daemon=True)
    waiting.start()
    waiting.join(timeout=10)
    report = {
        "answered": not waiting.is_alive(),
        "outcome": outcome,
        "first": first,
        "submit_errors": dev.fake_submit_errors(),
        "resident": dev.fake_resident_sqes(),
        "owned": dev.fake_owned(),
        "poisoned": dev.is_poisoned(),
    }
    try:
        dev.close()
        report["close"] = "returned"
    except RuntimeError:
        report["close"] = "refused"
    gc.collect()
    report["owner_alive"] = owner() is not None
    report["retained"] = dev.retained_owner_count()
    report["violations"] = dev.fake_violations()
    print(json.dumps(report))
    sys.stdout.flush()
    os._exit(0)
    """
)


def test_a_fatal_remainder_submit_keeps_the_owner_and_answers_the_waiter(
    tmp_path: Path,
) -> None:
    """A CQE for the prefix does not authorize reuse of an unknown suffix."""
    device = tmp_path / "dev.bin"
    device.write_bytes(bytes(16384))
    report = _run_scenario(SCRIPTED_FATAL_SHORT_REMAINDER, device)

    assert report["answered"] is True, report
    assert report["submit_errors"] == [5], report
    assert report["outcome"]["results"] == [False], report
    assert report["outcome"]["errors"], report
    assert report["owned"] == [], report
    assert len(report["resident"]) == 1, report
    remainder = report["resident"][0]
    assert remainder["offset"] == report["first"]["offset"] + 4096, report
    assert remainder["addr"] == report["first"]["addr"] + 4096, report
    assert remainder["len"] == 4096, report
    assert report["poisoned"] is True, report
    assert report["close"] == "refused", report
    assert report["owner_alive"] is True, report
    assert report["retained"] >= 1, report
    assert report["violations"] == [], report


SCRIPTED_TERMINAL_ADMISSION = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    import threading

    dev = device()
    payload = bytearray(b"x" * 4096)
    if POISONED:
        dev.fake_submit_fails(5)
        batch = dev.batched_write([0], [payload], [4096], [None])
        assert wait_until(dev.is_poisoned)
        dev.wait_iouring(batch)
        try:
            dev.close()
        except RuntimeError:
            pass
        else:
            raise AssertionError("an unknown outcome must refuse close")
    else:
        dev.drain_worker()

    answers = {}
    def attempt(name, operation):
        try:
            operation()
            answers[name] = "accepted"
        except RuntimeError as error:
            answers[name] = str(error)

    calls = {
        "batched_write": lambda: dev.batched_write([0], [payload], [4096], [None]),
        "batched_read": lambda: dev.batched_read([0], [payload], [4096]),
        "write_uring": lambda: dev.write_uring(0, payload, 4096, 4096, None),
    }
    if POISONED:
        calls.update({
            "pwrite": lambda: dev.pwrite_from_buffer(0, payload, 4096, 4096),
            "pread": lambda: dev.pread_into_buffer(0, payload, 4096, 4096),
        })
    for name, operation in calls.items():
        worker = threading.Thread(target=attempt, args=(name, operation), daemon=True)
        worker.start()
        worker.join(timeout=2)
        if worker.is_alive():
            answers[name] = "blocked"
    print(json.dumps({"answers": answers}))
    sys.stdout.flush()
    os._exit(0)
    """
)


@pytest.mark.parametrize("poisoned", [False, True], ids=["drained", "refused-close"])
def test_a_terminal_worker_refuses_new_io(tmp_path: Path, poisoned: bool) -> None:
    """Do not queue a new request after its only completion worker has left."""
    device = tmp_path / "dev.bin"
    device.write_bytes(bytes(8192))
    report = _run_scenario(
        SCRIPTED_TERMINAL_ADMISSION.replace("POISONED", str(poisoned)), device
    )

    assert report["answers"], report
    for method, answer in report["answers"].items():
        if poisoned:
            assert "unknown I/O outcome" in answer, (method, answer)
        else:
            assert "worker" in answer and "stopped" in answer, (method, answer)


SCRIPTED_REJECTED_WRITE = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    class Payload(bytearray):
        pass

    class PointerPayload:
        nbytes = 4096

        def data_ptr(self):
            return 0x100000

    dev = RawBlockDevice(
        sys.argv[2], writable=True, use_iouring=True,
        use_odirect=CASE in ("offset", "length"), alignment=4096,
        iouring_queue_depth=8, fake_ring_capacity=8,
    )
    payload = PointerPayload() if CASE == "pointer" else Payload(b"x" * 4096)
    alive = weakref.ref(payload)
    offset = 1 if CASE == "offset" else 0
    payload_len = 4097 if CASE == "capacity" else 4096
    total_len = {
        "length": 4097, "short_total": 4095, "allocation": 1 << 63,
    }.get(CASE, payload_len)
    placement_id = 0 if CASE == "placement" else None
    report = {}
    try:
        dev.write_uring(offset, payload, payload_len, total_len, placement_id)
        report["error"] = "accepted"
    except (ValueError, RuntimeError) as error:
        report["error"] = str(error)
    if isinstance(payload, bytearray):
        try:
            payload.append(0)
            report["export_released"] = True
        except BufferError:
            report["export_released"] = False
    else:
        report["export_released"] = True
    del payload
    gc.collect()
    report["owner_released"] = alive() is None
    report["owned"] = dev.fake_owned()
    report["violations"] = dev.fake_violations()
    dev.close()
    print(json.dumps(report))
    """
)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("capacity", "input buffer too small"),
        ("short_total", "total_len must be >= payload_len"),
        ("offset", "O_DIRECT requires aligned offset"),
        ("length", "O_DIRECT requires aligned total_len"),
        ("pointer", "pointer-only buffer is not fully covered"),
        ("placement", "placement"),
        ("allocation", "posix_memalign failed"),
    ],
)
def test_rejected_synchronous_write_releases_its_buffer(
    tmp_path: Path, case: str, message: str
) -> None:
    """Release both the exporter and owner when no write reaches the ring."""
    device = tmp_path / "dev.bin"
    device.write_bytes(bytes(8192))
    report = _run_scenario(SCRIPTED_REJECTED_WRITE.replace("CASE", repr(case)), device)

    assert message in report["error"], report
    assert report["export_released"] is True, report
    assert report["owner_released"] is True, report
    assert report["owned"] == [], report
    assert report["violations"] == [], report


SCRIPTED_PADDED_BATCH = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    dev = device()
    payload = bytearray(b"data")
    if POSITIONAL:
        batch = dev.batched_write(
            [0], [payload], [4096], None, [4], request_tag="host-padding",
        )
    else:
        batch = dev.batched_write(
            [0], [payload], [4096], payload_lens=[4], request_tag="host-padding",
        )
    assert wait_until(lambda: bool(dev.fake_owned()))
    dev.fake_complete(dev.fake_owned()[0], 4096)
    results, errors = dev.wait_iouring(batch)
    payload.append(0)
    report = {
        "results": results, "errors": errors,
        "payload": list(payload), "violations": dev.fake_violations(),
    }
    dev.close()
    print(json.dumps(report))
    """
)


@pytest.mark.parametrize("positional", [False, True])
def test_padded_batch_accepts_upstream_payload_lengths(
    tmp_path: Path, positional: bool
) -> None:
    """Accept a short host buffer using either upstream argument spelling."""
    device = tmp_path / "dev.bin"
    device.write_bytes(bytes(8192))
    report = _run_scenario(
        SCRIPTED_PADDED_BATCH.replace("POSITIONAL", str(positional)), device
    )
    assert report["results"] == [True], report
    assert report["errors"] == [], report
    assert report["payload"] == list(b"data\0"), report
    assert report["violations"] == [], report


SCRIPTED_REJECTED_BATCH = SCRIPTED_PREAMBLE + textwrap.dedent(
    """
    class Payload(bytearray):
        pass

    dev = device()
    payload = Payload(b"x" * 4096)
    alive = weakref.ref(payload)
    report = {}
    try:
        if OPERATION == "read":
            dev.batched_read([0, 4096], [payload, object()], [4096, 4096])
        else:
            dev.batched_write([0, 4096], [payload, object()], [4096, 4096])
        report["error"] = "accepted"
    except TypeError:
        report["error"] = "TypeError"
    try:
        payload.append(0)
        report["export_released"] = True
    except BufferError:
        report["export_released"] = False
    del payload
    gc.collect()
    report["owner_released"] = alive() is None
    report["owned"] = dev.fake_owned()
    report["violations"] = dev.fake_violations()
    dev.close()
    print(json.dumps(report))
    """
)


@pytest.mark.parametrize("operation", ["read", "write"])
def test_rejected_batch_releases_already_acquired_buffer_views(
    tmp_path: Path, operation: str
) -> None:
    """Unpin earlier buffers when a later batch item has no buffer interface."""
    device = tmp_path / "dev.bin"
    device.write_bytes(bytes(8192))
    report = _run_scenario(
        SCRIPTED_REJECTED_BATCH.replace("OPERATION", repr(operation)), device
    )

    assert report["error"] == "TypeError", report
    assert report["export_released"] is True, report
    assert report["owner_released"] is True, report
    assert report["owned"] == [], report
    assert report["violations"] == [], report
