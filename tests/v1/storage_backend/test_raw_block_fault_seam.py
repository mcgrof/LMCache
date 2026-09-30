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


def test_an_unpolled_batch_keeps_its_owners_through_close(tmp_path) -> None:
    """Nobody has to poll for the ownership rule to hold.

    A batch's Python owners used to move out of reach only when a caller
    reached wait_iouring. A caller that never polls -- which is every caller
    whose request was abandoned, cancelled, or whose engine is shutting
    down -- left them where the device's own destructor would free them, and
    freeing them returns their pool slices to an allocator. The close that
    did so had just refused, on the grounds that it could not establish what
    the device was doing.
    """
    device = tmp_path / "dev.bin"
    with open(device, "wb") as handle:
        handle.truncate(64 * 1024 * 1024)
    result = subprocess.run(
        [sys.executable, "-c", UNPOLLED_OWNER_SCENARIO, FAULT_EXT, str(device)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])

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
