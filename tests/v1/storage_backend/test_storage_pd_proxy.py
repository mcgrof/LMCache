# SPDX-License-Identifier: Apache-2.0
"""Durable READY barrier tests for the disaggregated prefill proxy."""

# Third Party
import pytest

# First Party
from examples.disagg_prefill import disagg_proxy_server as proxy
from lmcache.v1.storage_backend.raw_block import RawBlockPublicationReceipt
from lmcache.v1.storage_backend.storage_pd_protocol import StoragePDStatus


def _reset_proxy_state() -> None:
    proxy.app.state.finished_reqs.clear()
    proxy.app.state.storage_pd_statuses.clear()
    proxy.app.state.storage_pd_failures.clear()


@pytest.fixture(autouse=True)
def reset_proxy_state():
    _reset_proxy_state()
    yield
    _reset_proxy_state()


@pytest.mark.asyncio
async def test_storage_pd_barrier_returns_exact_ordered_ranks() -> None:
    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    proxy.app.state.storage_pd_statuses["request"] = {
        1: StoragePDStatus.ready("request", 1, receipt),
        0: StoragePDStatus.ready("request", 0, receipt),
    }

    statuses = await proxy.wait_decode_kv_ready(
        "request", 2, storage_pd=True, timeout_s=0.1
    )

    assert [status.tp_rank for status in statuses] == [0, 1]
    assert "request" not in proxy.app.state.storage_pd_statuses


@pytest.mark.asyncio
async def test_storage_pd_barrier_never_accepts_legacy_notification() -> None:
    proxy.app.state.finished_reqs["request"] = 1

    with pytest.raises(TimeoutError, match="timed out waiting for TP ranks"):
        await proxy.wait_decode_kv_ready("request", 1, storage_pd=True, timeout_s=0.002)

    assert "request" not in proxy.app.state.finished_reqs


@pytest.mark.asyncio
async def test_storage_pd_barrier_rejects_unexpected_rank() -> None:
    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    proxy.app.state.storage_pd_statuses["request"] = {
        1: StoragePDStatus.ready("request", 1, receipt)
    }

    with pytest.raises(RuntimeError, match="unexpected TP ranks"):
        await proxy.wait_decode_kv_ready("request", 1, storage_pd=True, timeout_s=0.1)

    assert "request" not in proxy.app.state.storage_pd_statuses


@pytest.mark.asyncio
async def test_legacy_barrier_still_accepts_prefill_notification() -> None:
    proxy.app.state.finished_reqs["request"] = 1

    assert (
        await proxy.wait_decode_kv_ready("request", 1, storage_pd=False, timeout_s=0.1)
        == []
    )


@pytest.mark.asyncio
async def test_client_info_closes_the_client_it_owns() -> None:
    """The proxy holds client records, not clients.

    Shutdown iterates those records, so the close has to belong to the
    record. Without it the call raises and every HTTP client stays open,
    and shutdown never reaches the receiver it was about to stop.
    """

    class RecordingClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    client = RecordingClient()
    info = proxy.ClientInfo(client)  # type: ignore[arg-type]
    await info.aclose()

    assert client.closed
