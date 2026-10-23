# SPDX-License-Identifier: Apache-2.0
"""Retain registered rotary tensors through host copies and uncertain failures.

This copy primitive does not authorize runtime discovery or engine recovery.
Its caller must already have ordered registration's first read against the
worker producer, and must retain the exporter until teardown is acknowledged.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import threading
import uuid

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.kmesh_domains import KMeshFenceError
from lmcache.v1.multiprocess.modules.blend.rope import BorrowedRope
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import BorrowedContext
from lmcache.v1.platform.base.event_ipc import EventIPCBackend
from lmcache.v1.platform.stream import synchronize_stream

logger = init_logger(__name__)

# Preserve the manager even if its server module loses its last reference.
_quarantine: set[RetainedTensorCopies] = set()
_quarantine_lock = threading.Lock()


@dataclass
class _Copy:
    paged: BorrowedContext
    rotary: BorrowedRope
    device: torch.device
    stream: object
    backend: EventIPCBackend | None
    inputs: tuple[torch.Tensor, ...]
    host: list[torch.Tensor] = field(default_factory=list)
    imported: object | None = None
    uncertain: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class RetainedTensorCopies:
    """Own exact registered borrows until a snapshot copy physically completes.

    Successful copies return independent CPU tensors. Failed copies never
    return partially populated snapshots. Uncertain driver failures retain
    their borrows, inputs, destinations and events until a server-side drain.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._copies: dict[str, _Copy] = {}
        self._closing = False

    @property
    def pending(self) -> int:
        """Count active and quarantined copies, including after shutdown refusal."""
        with self._lock:
            return len(self._copies)

    def copy_to_cpu(
        self,
        paged: BorrowedContext,
        rotary: BorrowedRope,
        producer_event: bytes = b"",
    ) -> list[torch.Tensor]:
        """Consume both borrows and copy their exact registered rotary tensors.

        Args:
            paged: Exact borrowed paged registration; ownership transfers here.
            rotary: Its exact borrowed rotary registration; also consumed.
            producer_event: Worker event recorded after the last input write.
                Required for device tensors and unused for CPU snapshots.

        Returns:
            Independent CPU copies after verified physical completion.

        Raises:
            KMeshFenceError: Unsupported input, closing admission, or a failed
                device operation. An undrained failure remains in ``pending``;
                the caller must not release either consumed borrow itself.
        """
        cache = paged.entry.cache_context
        inputs = tuple(rotary.state.cos_sin_caches)
        try:
            if not inputs or any(t.device != cache.device for t in inputs):
                raise KMeshFenceError("copy requires one registered device")
            if rotary.state.context_identity != paged.entry.ownership.identity:
                raise KMeshFenceError("rotary state belongs to another registration")
            cpu = cache.device.type == "cpu"
            if not cpu and (
                not isinstance(producer_event, bytes) or not producer_event
            ):
                raise KMeshFenceError("device copy requires a worker producer event")
            backend = paged.entry.event_backend
            if not cpu and backend is None:
                raise KMeshFenceError("registered context has no event backend")
            copy = _Copy(paged, rotary, cache.device, cache.stream, backend, inputs)
        except BaseException:
            rotary.release()
            paged.release()
            raise

        token = uuid.uuid4().hex
        # Serialize insertion with close; no lock spans driver work or callbacks.
        with self._lock:
            if self._closing:
                rotary.release()
                paged.release()
                raise KMeshFenceError("device-copy admission is closing")
            self._copies[token] = copy
            with _quarantine_lock:
                _quarantine.add(self)
        with copy.lock:
            if cpu:
                try:
                    return [
                        t.detach().clone(memory_format=torch.contiguous_format)
                        for t in inputs
                    ]
                finally:
                    self._release(token, copy)
            assert backend is not None
            try:
                with torch_dev.device(copy.device), torch_dev.stream(copy.stream):
                    backend.check_event_support(copy.device)
                    copy.imported = backend.import_event(producer_event, copy.device)
                    backend.wait_event(copy.imported, copy.stream)
                    for tensor in copy.inputs:
                        host = torch.empty_like(
                            tensor,
                            device="cpu",
                            pin_memory=True,
                            memory_format=torch.contiguous_format,
                        )
                        copy.host.append(host)
                        host.copy_(tensor, non_blocking=True)
                # Event synchronization backends may enqueue on helper streams.
                # Drain the stream owning every submitted read instead.
                synchronize_stream(copy.device, copy.stream)
            except BaseException:
                # Even interruption must not release buffers after queued DMA.
                logger.exception("Registered tensor copy failed; attempt stream drain")
                copy.uncertain = True
                try:
                    synchronize_stream(copy.device, copy.stream)
                except RuntimeError:
                    logger.exception("Retain registered tensor copy after failed drain")
                else:
                    self._release(token, copy)
                raise KMeshFenceError(
                    "registered tensor copy failed; server must check physical debt"
                ) from None
            self._release(token, copy)
            return copy.host

    def drain_quarantined(self) -> int:
        """Attempt physical drain of failed copies and return remaining debt.

        This server-only action does not cancel active copies or authorize their
        outputs. Driver failures leave their resources retained. A zero count
        describes this manager only, not an exporter teardown acknowledgement.
        """
        with self._lock:
            copies = tuple(self._copies.items())
        for token, copy in copies:
            if not copy.lock.acquire(blocking=False):
                continue
            try:
                if not copy.uncertain:
                    continue
                try:
                    synchronize_stream(copy.device, copy.stream)
                except RuntimeError:
                    logger.exception("Registered tensor copy still has uncertain drain")
                else:
                    self._release(token, copy)
            finally:
                copy.lock.release()
        return self.pending

    def close(self) -> None:
        """Stop admission and refuse to finish while any physical debt remains.

        Raises:
            KMeshFenceError: Active or quarantined copies still hold resources.
        """
        with self._lock:
            self._closing = True
        if self.drain_quarantined():
            raise KMeshFenceError("registered tensor copies must drain before shutdown")

    def _release(self, token: str, copy: _Copy) -> None:
        with self._lock:
            if self._copies.get(token) is not copy:
                return
            del self._copies[token]
            if not self._copies:
                with _quarantine_lock:
                    _quarantine.discard(self)
        copy.rotary.release()
        copy.paged.release()
