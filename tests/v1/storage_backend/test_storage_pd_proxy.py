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


@pytest.mark.asyncio
async def test_idle_receiver_stops_when_cancelled_not_when_flagged() -> None:
    """An idle receiver waits inside recv(), where the flag cannot reach it.

    Shutdown used to clear the flag and then wait for the task, so a proxy
    that happened to be idle never shut down at all. Cancelling is what
    wakes a blocked recv(), and the receiver closes its socket on the way
    out whichever route it leaves by.
    """
    # Standard
    import asyncio
    import socket as socket_module
    from types import SimpleNamespace

    probe = socket_module.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    original_args = getattr(proxy, "global_args", None)
    original_flag = proxy.run_proxy
    proxy.global_args = SimpleNamespace(
        proxy_host="127.0.0.1", proxy_port=port, storage_pd=True
    )
    proxy.run_proxy = True

    made: list = []
    original_socket = proxy.zmq_ctx.socket

    def recording_socket(*args, **kwargs):
        sock = original_socket(*args, **kwargs)
        made.append(sock)
        return sock

    proxy.zmq_ctx.socket = recording_socket  # type: ignore[method-assign]
    task = asyncio.create_task(proxy.zmq_pull_server())
    try:
        # Let it bind and settle into recv(). The coroutine body does not
        # run until this point, so the socket factory stays patched until
        # after it has taken one.
        await asyncio.sleep(0.2)
        proxy.zmq_ctx.socket = original_socket  # type: ignore[method-assign]
        assert not task.done()

        proxy.run_proxy = False
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.3)

        task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
        assert task.done()
        assert made, "the receiver never created its socket"
        assert made[0].closed, "a cancelled receiver left its socket open"
    finally:
        proxy.zmq_ctx.socket = original_socket  # type: ignore[method-assign]
        if not task.done():
            task.cancel()
        proxy.run_proxy = original_flag
        if original_args is None:
            del proxy.global_args
        else:
            proxy.global_args = original_args


@pytest.mark.asyncio
async def test_handoff_budget_covers_a_prefiller_that_never_answers() -> None:
    """A stalled prefiller must not hold a request open indefinitely.

    The prefill clients carry no timeout of their own, and the wait for
    READY only starts its clock once the prefill call returns, so a
    prefiller that stops answering used to stall the handoff with nothing
    bounding it.
    """
    # Standard
    import asyncio
    import time
    from types import SimpleNamespace

    original_args = getattr(proxy, "global_args", None)
    original_send = proxy.send_request_to_service
    proxy.global_args = SimpleNamespace(storage_pd=True, storage_pd_ready_timeout_s=0.3)

    async def never_answers(*args, **kwargs):
        await asyncio.sleep(3600)

    proxy.send_request_to_service = never_answers  # type: ignore[assignment]
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="waiting for the prefiller"):
            await proxy.prefill_within_handoff_budget(None, {}, "request")
        assert time.monotonic() - started < 2
    finally:
        proxy.send_request_to_service = original_send  # type: ignore[assignment]
        if original_args is None:
            del proxy.global_args
        else:
            proxy.global_args = original_args


@pytest.mark.asyncio
async def test_handoff_budget_passes_the_remainder_to_the_ready_wait() -> None:
    """What the prefiller spends comes out of the same budget."""
    # Standard
    import asyncio
    from types import SimpleNamespace

    original_args = getattr(proxy, "global_args", None)
    original_send = proxy.send_request_to_service
    proxy.global_args = SimpleNamespace(storage_pd=True, storage_pd_ready_timeout_s=1.0)

    async def answers_slowly(*args, **kwargs):
        await asyncio.sleep(0.3)
        return "response"

    proxy.send_request_to_service = answers_slowly  # type: ignore[assignment]
    try:
        response, remaining = await proxy.prefill_within_handoff_budget(
            None, {}, "request"
        )
        assert response == "response"
        assert 0.4 < remaining < 0.8
    finally:
        proxy.send_request_to_service = original_send  # type: ignore[assignment]
        if original_args is None:
            del proxy.global_args
        else:
            proxy.global_args = original_args
