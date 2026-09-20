# SPDX-License-Identifier: Apache-2.0
"""Wire messages for request-scoped shared-storage P/D handoff."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Literal
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.raw_block.core import RawBlockPublicationReceipt

StoragePDState = Literal["READY", "FAILED", "CANCELLED"]


class StoragePDStatus(msgspec.Struct, tag=True):
    """Terminal producer status for one request and tensor-parallel rank."""

    req_id: str
    producer_instance_id: str
    tp_rank: int
    state: StoragePDState
    writer_epoch: str = ""
    namespace_identity: str = ""
    checkpoint_seq: int = 0
    key_count: int = 0
    manifest_digest: str = ""
    total_logical_bytes: int = 0
    total_padded_bytes: int = 0
    error_stage: str = ""
    error_text: str = ""

    @classmethod
    def ready(
        cls,
        req_id: str,
        tp_rank: int,
        receipt: RawBlockPublicationReceipt,
    ) -> "StoragePDStatus":
        """Build a READY status from a durable publication receipt."""
        return cls(
            req_id=req_id,
            producer_instance_id=receipt.writer_epoch,
            tp_rank=tp_rank,
            state="READY",
            writer_epoch=receipt.writer_epoch,
            namespace_identity=receipt.namespace_identity,
            checkpoint_seq=receipt.checkpoint_seq,
            key_count=receipt.key_count,
            manifest_digest=receipt.manifest_digest,
            total_logical_bytes=receipt.total_logical_bytes,
            total_padded_bytes=receipt.total_padded_bytes,
        )

    def publication_receipt(self) -> RawBlockPublicationReceipt:
        """Convert a READY wire status back to a core receipt."""
        if self.state != "READY":
            raise ValueError(f"storage P/D status is not READY: {self.state}")
        return RawBlockPublicationReceipt(
            writer_epoch=self.writer_epoch,
            checkpoint_seq=self.checkpoint_seq,
            key_count=self.key_count,
            manifest_digest=self.manifest_digest,
            namespace_identity=self.namespace_identity,
            total_logical_bytes=self.total_logical_bytes,
            total_padded_bytes=self.total_padded_bytes,
        )


class StoragePDReadAck(msgspec.Struct, tag=True):
    """Consumer acknowledgement after the advertised KV reached GPU memory."""

    req_id: str
    producer_instance_id: str
    consumer_instance_id: str
    tp_rank: int
    writer_epoch: str
    checkpoint_seq: int
    manifest_digest: str


StoragePDMsg = StoragePDStatus | StoragePDReadAck


def order_storage_pd_ready_statuses(
    statuses: Mapping[int, StoragePDStatus],
    num_tp_ranks: int,
) -> list[StoragePDStatus]:
    """Validate and order one READY status for every expected TP rank.

    Args:
        statuses: Statuses keyed by their claimed tensor-parallel rank.
        num_tp_ranks: Expected tensor-parallel world size.

    Returns:
        Statuses ordered by rank from zero through ``num_tp_ranks - 1``.

    Raises:
        ValueError: If the rank set or a status identity is invalid.
    """
    if num_tp_ranks <= 0:
        raise ValueError("storage P/D requires at least one TP rank")
    expected_ranks = set(range(num_tp_ranks))
    actual_ranks = set(statuses)
    if actual_ranks != expected_ranks:
        raise ValueError(
            "storage P/D READY rank set mismatch: "
            f"expected={sorted(expected_ranks)}, actual={sorted(actual_ranks)}"
        )
    ordered = [statuses[rank] for rank in range(num_tp_ranks)]
    req_ids = {status.req_id for status in ordered}
    if len(req_ids) != 1:
        raise ValueError("storage P/D READY statuses identify different requests")
    for rank, status in enumerate(ordered):
        if status.state != "READY" or status.tp_rank != rank:
            raise ValueError(
                f"storage P/D rank {rank} has invalid {status.state} status "
                f"claiming rank {status.tp_rank}"
            )
    return ordered


class StoragePDStatusSender:
    """Own a PUSH socket on one worker thread and send terminal statuses."""

    def __init__(
        self,
        proxy_host: str,
        proxy_port: int,
        *,
        timeout_s: float = 5.0,
    ) -> None:
        if not proxy_host:
            raise ValueError("storage P/D requires pd_proxy_host")
        if proxy_port <= 0:
            raise ValueError("storage P/D requires a positive pd_proxy_port")
        if timeout_s <= 0:
            raise ValueError("storage P/D status timeout must be positive")
        self._proxy_url = f"{proxy_host}:{proxy_port}"
        self._timeout_s = timeout_s
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="storage-pd-status",
        )
        self._socket: zmq.Socket | None = None
        self._closed = False
        self._lock = threading.Lock()

    def send(self, status: StoragePDMsg) -> None:
        """Synchronously enqueue a status on the owned ZeroMQ socket."""
        with self._lock:
            if self._closed:
                raise RuntimeError("storage P/D status sender is closed")
            future = self._executor.submit(self._send, status)
        try:
            future.result(timeout=self._timeout_s + 1.0)
        except FutureTimeoutError as exc:
            raise LMCacheTimeoutError(
                f"storage P/D status send timed out for {self._proxy_url}"
            ) from exc

    def close(self) -> None:
        """Close the socket on its owner thread and stop the worker."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close_future = self._executor.submit(self._close_socket)
        close_future.result()
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _send(self, status: StoragePDMsg) -> None:
        if self._socket is None:
            context = get_zmq_context(use_asyncio=False)
            self._socket = get_zmq_socket(
                context,
                self._proxy_url,
                "tcp",
                zmq.PUSH,
                "connect",
            )
            self._socket.setsockopt(zmq.SNDTIMEO, int(self._timeout_s * 1000))
        self._socket.send(msgspec.msgpack.encode(status))

    def _close_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=1000)
            self._socket = None
