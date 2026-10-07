# SPDX-License-Identifier: Apache-2.0
"""Metric allocation must not deadlock same-thread memory finalizers."""

# Standard
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import textwrap
import time

# First Party
from lmcache.utils import thread_safe


def test_finalizer_can_update_observability_while_a_collector_is_created():
    # A child bounds the failure on a non-reentrant lock. The finalizer runs
    # synchronously inside collect(), just as it can during metric allocation.
    code = textwrap.dedent("""
        import gc
        from lmcache.utils import thread_safe

        updates = []

        @thread_safe
        def record_release():
            updates.append("released")

        class CyclicOwner:
            def __init__(self):
                self.cycle = self

            def __del__(self):
                record_release()

        gc.disable()
        owner = CyclicOwner()
        del owner

        @thread_safe
        def create_collector():
            gc.collect()

        create_collector()
        assert updates == ["released"]
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_observability_still_excludes_other_threads():
    active = 0
    observed = []

    @thread_safe
    def update():
        nonlocal active
        active += 1
        observed.append(active)
        time.sleep(0.002)
        active -= 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: update(), range(16)))
    assert observed == [1] * 16
    assert active == 0
