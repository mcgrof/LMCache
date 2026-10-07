# SPDX-License-Identifier: Apache-2.0
"""Optional single-host causal receipts for storage-mediated P/D.

The trace is disabled unless ``LMCACHE_STORAGE_PD_TRACE_FILE`` names a file.
Each event is one bounded JSON line appended with one ``os.write`` call, so
the writer, reader, and proxy can share a local file without a collector
process. Linux monotonic timestamps are comparable across processes on the
same host, which is the only scope this receipt claims.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any
import itertools
import json
import os
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_SCHEMA = "lmcache.storage_pd.timeline.v1"
_counter = itertools.count(1)
_lock = threading.Lock()
_warned = False


def trace_storage_pd_event(event: str, **fields: Any) -> None:
    """Append one best-effort causal event when qualification tracing is on.

    Trace failure never changes serving behavior. The checker treats a
    missing event as a failed evidence gate, so swallowing an I/O error here
    cannot turn an incomplete receipt into a pass.
    """
    path = os.environ.get("LMCACHE_STORAGE_PD_TRACE_FILE", "")
    if not path:
        return
    global _warned
    try:
        with _lock:
            sequence = next(_counter)
            row = {
                "schema": _SCHEMA,
                "event": event,
                "monotonic_ns": time.monotonic_ns(),
                "pid": os.getpid(),
                "process_sequence": sequence,
                **fields,
            }
            encoded = (
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                written = os.write(fd, encoded)
            finally:
                os.close(fd)
            if written != len(encoded):
                raise OSError(
                    f"short storage P/D trace write: {written}/{len(encoded)}"
                )
    except Exception:
        if not _warned:
            _warned = True
            logger.warning("Storage P/D causal trace write failed", exc_info=True)
