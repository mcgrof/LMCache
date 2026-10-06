# SPDX-License-Identifier: Apache-2.0
"""Durable READY barrier tests for the disaggregated prefill proxy."""

# Standard
from types import SimpleNamespace
import json
import socket as socketlib
import time

# Third Party
from starlette.requests import ClientDisconnect
import anyio
import pytest

# First Party
from examples.disagg_prefill import disagg_proxy_server as proxy
from lmcache.v1.storage_backend.raw_block import RawBlockPublicationReceipt
from lmcache.v1.storage_backend.storage_pd_ack import (
    StoragePDAckServer,
    StoragePDClaimAnswer,
    StoragePDUnreadAnswer,
    StoragePDUnreadClient,
    StoragePDUnreadRequest,
)
from lmcache.v1.storage_backend.storage_pd_protocol import StoragePDStatus

LOOPBACK = "127.0.0.1"


def _free_port() -> int:
    with socketlib.socket(socketlib.AF_INET, socketlib.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return int(probe.getsockname()[1])


class _RecordingWriter:
    """A writer that records what it was told and releases it."""

    def __init__(self) -> None:
        self.reports: list[StoragePDUnreadRequest] = []

    def unread(self, request: StoragePDUnreadRequest) -> StoragePDUnreadAnswer:
        self.reports.append(request)
        return StoragePDUnreadAnswer(True)


def _reset_proxy_state() -> None:
    proxy.app.state.finished_reqs.clear()
    proxy.app.state.storage_pd_statuses.clear()
    proxy.app.state.storage_pd_failures.clear()
    proxy.app.state.storage_pd_active.clear()


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
        "request", 2, storage_pd=True, deadline=time.monotonic() + 0.1
    )

    assert [status.tp_rank for status in statuses] == [0, 1]
    assert "request" not in proxy.app.state.storage_pd_statuses


@pytest.mark.asyncio
async def test_storage_pd_barrier_never_accepts_legacy_notification() -> None:
    proxy.app.state.finished_reqs["request"] = 1

    with pytest.raises(TimeoutError, match="timed out waiting for TP ranks"):
        await proxy.wait_decode_kv_ready(
            "request", 1, storage_pd=True, deadline=time.monotonic() + 0.002
        )

    assert "request" not in proxy.app.state.finished_reqs


@pytest.mark.asyncio
async def test_storage_pd_barrier_rejects_unexpected_rank() -> None:
    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    proxy.app.state.storage_pd_statuses["request"] = {
        1: StoragePDStatus.ready("request", 1, receipt)
    }

    with pytest.raises(RuntimeError, match="unexpected TP ranks"):
        await proxy.wait_decode_kv_ready(
            "request", 1, storage_pd=True, deadline=time.monotonic() + 0.1
        )

    assert "request" not in proxy.app.state.storage_pd_statuses


@pytest.mark.asyncio
async def test_legacy_barrier_still_accepts_prefill_notification() -> None:
    proxy.app.state.finished_reqs["request"] = 1

    assert (
        await proxy.wait_decode_kv_ready(
            "request", 1, storage_pd=False, deadline=time.monotonic() + 0.1
        )
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
    from types import SimpleNamespace
    import asyncio
    import socket as socket_module

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
    from types import SimpleNamespace
    import asyncio
    import time

    original_args = getattr(proxy, "global_args", None)
    original_send = proxy.send_request_to_service
    proxy.global_args = SimpleNamespace(storage_pd=True, storage_pd_ready_timeout_s=0.3)

    async def never_answers(*args, **kwargs):
        await asyncio.sleep(3600)

    proxy.send_request_to_service = never_answers  # type: ignore[assignment]
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="waiting for the prefiller"):
            await proxy.prefill_within_handoff_budget(
                None, {}, "request", proxy._handoff_deadline()
            )
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
    from types import SimpleNamespace
    import asyncio

    original_args = getattr(proxy, "global_args", None)
    original_send = proxy.send_request_to_service
    proxy.global_args = SimpleNamespace(storage_pd=True, storage_pd_ready_timeout_s=1.0)

    async def answers_slowly(*args, **kwargs):
        await asyncio.sleep(0.3)
        return "response"

    proxy.send_request_to_service = answers_slowly  # type: ignore[assignment]
    try:
        deadline = proxy._handoff_deadline()
        response = await proxy.prefill_within_handoff_budget(
            None, {}, "request", deadline
        )
        assert response == "response"
        # The deadline is absolute, so what the prefiller spent is gone from
        # the barrier's share of it rather than being handed back.
        remaining = deadline - time.monotonic()
        assert 0.4 < remaining < 0.8
    finally:
        proxy.send_request_to_service = original_send  # type: ignore[assignment]
        if original_args is None:
            del proxy.global_args
        else:
            proxy.global_args = original_args


@pytest.mark.asyncio
async def test_barrier_refuses_a_complete_set_once_the_deadline_has_passed() -> None:
    """An exhausted budget must fail, even with every rank reported.

    The barrier checked its deadline only after deciding the set was
    complete, so a request handed an already spent budget still succeeded.
    The deadline is shared with the prefill call, so by then the handoff has
    already taken longer than it was allowed.
    """
    # Standard
    import time

    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    proxy._register_pd_request("request")
    proxy.app.state.storage_pd_statuses["request"] = {
        0: StoragePDStatus.ready("request", 0, receipt),
    }

    with pytest.raises(TimeoutError, match="timed out waiting for TP ranks"):
        await proxy.wait_decode_kv_ready(
            "request", 1, storage_pd=True, deadline=time.monotonic() - 1.0
        )
    # And it cleaned up after itself.
    assert "request" not in proxy.app.state.storage_pd_statuses
    assert not proxy._pd_request_is_active("request")


@pytest.mark.asyncio
async def test_barrier_accepts_a_complete_set_within_the_deadline() -> None:
    """The same set inside the budget still succeeds."""
    # Standard
    import time

    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    proxy._register_pd_request("request")
    proxy.app.state.storage_pd_statuses["request"] = {
        0: StoragePDStatus.ready("request", 0, receipt),
    }

    statuses = await proxy.wait_decode_kv_ready(
        "request", 1, storage_pd=True, deadline=time.monotonic() + 5.0
    )
    assert [status.tp_rank for status in statuses] == [0]


def test_a_status_for_a_finished_request_does_not_recreate_its_state() -> None:
    """Late traffic must not bring back state a barrier just cleared.

    The receiver wrote into defaulting maps, so a READY arriving after the
    barrier finished recreated that request's entry. Nothing clears it
    again, so the proxy grew one entry per late message for as long as it
    ran, and a subsequent request reusing the identifier would inherit it.
    """
    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    status = StoragePDStatus.ready("done-request", 0, receipt)

    # Registered: recorded.
    proxy._register_pd_request("done-request")
    assert proxy.record_storage_pd_status(status) == "recorded"
    assert "done-request" in proxy.app.state.storage_pd_statuses

    # Cleared, as a completed barrier does; a late copy is ignored.
    proxy._clear_pd_request_state("done-request")
    assert proxy.record_storage_pd_status(status) == "ignored"
    assert "done-request" not in proxy.app.state.storage_pd_statuses
    assert "done-request" not in proxy.app.state.storage_pd_failures


def test_a_status_for_an_unknown_request_is_ignored() -> None:
    """A request the proxy never registered has no state to build."""
    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    assert (
        proxy.record_storage_pd_status(StoragePDStatus.ready("never-seen", 0, receipt))
        == "ignored"
    )
    assert proxy.app.state.storage_pd_statuses == {}

    failure = StoragePDStatus(
        req_id="never-seen",
        tp_rank=0,
        state="FAILED",
        error_stage="WRITE_OR_PUBLISH",
        error_text="nope",
    )
    assert proxy.record_storage_pd_status(failure) == "ignored"
    assert proxy.app.state.storage_pd_failures == {}


def test_a_conflicting_second_ready_for_one_rank_fails_the_request() -> None:
    """Two different READYs from one rank cannot both be true."""
    first = StoragePDStatus.ready(
        "request", 0, RawBlockPublicationReceipt("writer", 1, 1, "digest-a")
    )
    second = StoragePDStatus.ready(
        "request", 0, RawBlockPublicationReceipt("writer", 2, 1, "digest-b")
    )
    proxy._register_pd_request("request")
    assert proxy.record_storage_pd_status(first) == "recorded"
    assert proxy.record_storage_pd_status(second) == "conflict"
    assert proxy.app.state.storage_pd_failures["request"].error_stage == "PROXY_BARRIER"

    # An identical duplicate is not a conflict.
    proxy._clear_pd_request_state("request")
    proxy._register_pd_request("request")
    assert proxy.record_storage_pd_status(first) == "recorded"
    assert proxy.record_storage_pd_status(first) == "recorded"
    assert "request" not in proxy.app.state.storage_pd_failures


class _CountingSemaphore:
    """Records every acquire and release so exactly-once can be asserted."""

    def __init__(self) -> None:
        self.held = 0
        self.acquires = 0
        self.releases = 0

    async def acquire(self, slots: int) -> None:
        self.held += slots
        self.acquires += 1

    async def release(self, slots: int) -> None:
        self.held -= slots
        self.releases += 1


def _endpoint_env(monkeypatch, prefill_outcome, *, tokens=(1, 2, 3)):
    """Point the real endpoint at stubs, and hand back the semaphore.

    Only transport and client selection are replaced. The endpoint's own
    control flow, its request registration and its cleanup are the code
    under test.
    """
    # Standard
    from types import SimpleNamespace

    semaphore = _CountingSemaphore()
    client = SimpleNamespace(
        client=None, host="localhost", init_port=[1], alloc_port=[2]
    )

    async def send_request_to_service(_client, endpoint, _data):
        if endpoint == "/tokenize":
            return SimpleNamespace(json=lambda: {"tokens": list(tokens)})
        if isinstance(prefill_outcome, BaseException):
            raise prefill_outcome
        return SimpleNamespace(json=lambda: prefill_outcome)

    monkeypatch.setattr(proxy, "pd_buffer_semaphore", semaphore)
    monkeypatch.setattr(proxy, "send_request_to_service", send_request_to_service)
    monkeypatch.setattr(
        proxy, "pick_up_clients", lambda _request: (client, client, client)
    )
    # global_args only exists once the server's main() has run, so it has to
    # be created here rather than replaced.
    monkeypatch.setattr(
        proxy,
        "global_args",
        SimpleNamespace(
            storage_pd=True, storage_pd_ready_timeout_s=1.0, chunk_size=256
        ),
        raising=False,
    )
    monkeypatch.setattr(proxy, "stats_calculator", SimpleNamespace(add=lambda _v: None))
    return semaphore


def _fake_request(prompt="hello", max_tokens=32):
    # Standard
    from types import SimpleNamespace

    async def json():
        return {"prompt": prompt, "max_tokens": max_tokens}

    return SimpleNamespace(json=json)


@pytest.mark.asyncio
async def test_endpoint_releases_its_request_when_prefill_fails(monkeypatch) -> None:
    """A failure before the barrier must not leave the request registered.

    The endpoint registers the request before contacting the prefiller,
    because a producer can report ready first. Cleanup used to sit around the
    barrier alone, so a request that never reached it stayed registered for
    the life of the proxy, and its status map with it.
    """
    semaphore = _endpoint_env(monkeypatch, RuntimeError("prefill HTTP failure"))

    with pytest.raises(RuntimeError, match="prefill HTTP failure"):
        await proxy.handle_completions(_fake_request())

    assert list(proxy.app.state.storage_pd_active) == []
    assert list(proxy.app.state.storage_pd_statuses) == []
    assert semaphore.held == 0
    assert semaphore.releases == 1


@pytest.mark.asyncio
async def test_endpoint_releases_its_request_when_cancelled(monkeypatch) -> None:
    """Cancellation is not an Exception, so the handler never saw it.

    That left both the registration and an acquired buffer permit behind,
    which is the worse of the two leaks: the permit bounds concurrency, so
    losing them throttles the proxy until it restarts.
    """
    # Standard
    import asyncio

    semaphore = _endpoint_env(monkeypatch, asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await proxy.handle_completions(_fake_request())

    assert list(proxy.app.state.storage_pd_active) == []
    assert list(proxy.app.state.storage_pd_statuses) == []
    assert semaphore.held == 0
    assert semaphore.releases == 1


@pytest.mark.asyncio
async def test_cancellation_after_ready_retains_an_unread_release(monkeypatch) -> None:
    """A READY publication cannot disappear with a cancelled proxy request."""
    # Standard
    import asyncio

    receipt = RawBlockPublicationReceipt(
        "writer",
        1,
        1,
        "digest",
        ack_endpoint="127.0.0.1:9999",
    )
    prefill_response = {
        "id": "cmpl-1",
        "created": 0,
        "model": "probe-model",
        "kv_transfer_params": {"first_tok": 7},
    }
    _endpoint_env(monkeypatch, prefill_response)
    reported: list[StoragePDStatus] = []

    async def wait(req_id, *_args, **_kwargs):
        status = StoragePDStatus.ready(req_id, 0, receipt)
        proxy.app.state.storage_pd_statuses[req_id][0] = status
        raise asyncio.CancelledError()

    async def report(statuses, **_kwargs):
        reported.extend(statuses)
        return []

    monkeypatch.setattr(proxy, "wait_decode_kv_ready", wait)
    monkeypatch.setattr(proxy, "tell_producers_nobody_will_read", report)

    with pytest.raises(asyncio.CancelledError):
        await proxy.handle_completions(_fake_request())

    assert len(reported) == 1
    assert reported[0].publication_receipt() == receipt


@pytest.mark.asyncio
async def test_cancelled_stream_resolves_a_publication_not_claimed_by_decode(
    monkeypatch,
) -> None:
    """The stream owns READY cleanup after the endpoint returns."""
    receipt = RawBlockPublicationReceipt(
        "writer",
        1,
        1,
        "digest",
        ack_endpoint="127.0.0.1:9999",
    )
    prefill_response = {
        "id": "cmpl-1",
        "created": 0,
        "model": "probe-model",
        "kv_transfer_params": {"first_tok": 7},
    }
    _endpoint_env(monkeypatch, prefill_response)
    reported: list[StoragePDStatus] = []

    async def wait(req_id, *_args, **_kwargs):
        return [StoragePDStatus.ready(req_id, 0, receipt)]

    async def report(statuses, **_kwargs):
        reported.extend(statuses)
        return []

    monkeypatch.setattr(proxy, "wait_decode_kv_ready", wait)
    monkeypatch.setattr(proxy, "tell_producers_nobody_will_read", report)

    response = await proxy.handle_completions(_fake_request())
    stream = response.body_iterator
    assert await anext(stream)
    await stream.aclose()

    assert len(reported) == 1
    assert reported[0].publication_receipt() == receipt


def test_decoder_stream_completion_requires_done_without_an_error() -> None:
    """A normal SSE EOF after an engine error is not a completed decode."""
    success = proxy.DecoderStreamCompletion()
    success.feed(b'data: {"choices": [{"finish_reason": "stop"}]}\n\ndata: [DO')
    success.feed(b"NE]\n\n")
    success.finish()
    assert success.succeeded

    failed = proxy.DecoderStreamCompletion()
    failed.feed(b'data: {"err')
    failed.feed(b'or": {"code": 500}}\n\ndata: [DONE]\n\n')
    failed.finish()
    assert failed.saw_done
    assert failed.saw_error
    assert not failed.succeeded

    truncated = proxy.DecoderStreamCompletion()
    truncated.feed(b'data: {"choices": []}\n\n')
    truncated.finish()
    assert not truncated.succeeded

    empty_done = proxy.DecoderStreamCompletion()
    empty_done.feed(b"data: [DONE]\n\n")
    empty_done.finish()
    assert empty_done.saw_done
    assert not empty_done.succeeded


@pytest.mark.asyncio
async def test_decoder_sse_error_resolves_an_unclaimed_publication(
    monkeypatch,
) -> None:
    """An SSE error plus DONE is failure, even though HTTP and EOF are clean."""
    receipt = RawBlockPublicationReceipt(
        "writer",
        1,
        1,
        "digest",
        ack_endpoint="127.0.0.1:9999",
    )
    prefill_response = {
        "id": "cmpl-1",
        "created": 0,
        "model": "probe-model",
        "kv_transfer_params": {"first_tok": 7},
    }
    _endpoint_env(monkeypatch, prefill_response)
    reported: list[StoragePDStatus] = []

    async def wait(req_id, *_args, **_kwargs):
        return [StoragePDStatus.ready(req_id, 0, receipt)]

    async def stream(_client, _endpoint, _data):
        yield b'data: {"error": {"code": 500}}\n\n'
        yield b"data: [DONE]\n\n"

    async def report(statuses, **_kwargs):
        reported.extend(statuses)
        return []

    monkeypatch.setattr(proxy, "wait_decode_kv_ready", wait)
    monkeypatch.setattr(proxy, "stream_service_response", stream)
    monkeypatch.setattr(proxy, "tell_producers_nobody_will_read", report)

    response = await proxy.handle_completions(_fake_request())
    chunks = [chunk async for chunk in response.body_iterator]

    assert any(b'"error"' in chunk for chunk in chunks)
    assert len(reported) == 1
    assert reported[0].publication_receipt() == receipt


@pytest.mark.asyncio
async def test_endpoint_releases_its_permit_once_on_the_successful_path(
    monkeypatch,
) -> None:
    """The successful path must release exactly once, not twice.

    The barrier releases the permit when it completes, and the outer scope
    releases anything still held. Both running would return a permit the
    proxy never took.
    """
    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    prefill_response = {
        "id": "cmpl-1",
        "created": 0,
        "model": "probe-model",
        "kv_transfer_params": {"first_tok": 7},
    }
    semaphore = _endpoint_env(monkeypatch, prefill_response)

    async def stream(_client, _endpoint, _data):
        for chunk in (b"data: {}\n\n",):
            yield chunk

    monkeypatch.setattr(proxy, "stream_service_response", stream)

    # The barrier is satisfied as soon as the expected rank reports.
    real_wait = proxy.wait_decode_kv_ready

    async def wait(req_id, num_tp_rank, *, storage_pd, deadline):
        proxy.app.state.storage_pd_statuses.setdefault(req_id, {})[0] = (
            StoragePDStatus.ready(req_id, 0, receipt)
        )
        return await real_wait(
            req_id, num_tp_rank, storage_pd=storage_pd, deadline=deadline
        )

    monkeypatch.setattr(proxy, "wait_decode_kv_ready", wait)

    response = await proxy.handle_completions(_fake_request())
    assert response is not None

    assert list(proxy.app.state.storage_pd_active) == []
    assert semaphore.held == 0
    assert semaphore.releases == 1, "the permit was released more than once"


def test_a_one_token_request_is_served_rather_than_refused() -> None:
    """One token is a whole answer, and the prefiller produces it.

    Refusing the request was a clearer error than forwarding a zero-token
    decode, but it was still a refusal of something servable. The budget is
    accepted and the decode is skipped instead.
    """
    for spelling in ("max_tokens", "max_completion_tokens"):
        req = {spelling: 1}
        assert proxy.take_prefill_budget(req) == 1
        assert req["max_tokens"] == 1
        assert proxy.producer_answer_is_complete(1, {}) is True


def test_a_budget_below_one_token_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        proxy.take_prefill_budget({"max_tokens": 0})


def test_a_producer_that_stopped_needs_no_decode() -> None:
    """A terminal producer token ends the answer whatever the budget was.

    Asking the decoder to continue past a stop would generate tokens the
    engine had already decided not to.
    """
    stopped = {"choices": [{"finish_reason": "stop"}]}
    ran_out = {"choices": [{"finish_reason": "length"}]}
    unfinished = {"choices": [{"finish_reason": None}]}
    assert proxy.producer_answer_is_complete(32, stopped) is True
    assert proxy.producer_answer_is_complete(32, ran_out) is False
    assert proxy.producer_answer_is_complete(32, unfinished) is False


def test_a_request_with_no_budget_at_all_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="needs max_tokens"):
        proxy.take_prefill_budget({"prompt": [1, 2, 3]})


def test_the_chat_spelling_of_the_budget_is_accepted() -> None:
    """A chat request may carry only max_completion_tokens.

    Reading max_tokens unconditionally raised KeyError and returned 500.
    """
    req = {"max_completion_tokens": 8}
    assert proxy.take_prefill_budget(req) == 8
    # The prefiller is given exactly one token whichever spelling arrived.
    assert req["max_tokens"] == 1


@pytest.mark.parametrize(
    "prefill_output",
    [
        {"kv_transfer_params": {}},
        {},
        {"kv_transfer_params": {"first_tok": None}},
        {"kv_transfer_params": {"first_tok": "7"}},
        {"kv_transfer_params": {"first_tok": True}},
    ],
    ids=["absent-key", "no-params", "null", "string", "bool"],
)
def test_an_unusable_producer_token_fails_before_decode(prefill_output) -> None:
    """Continuing without it is a different answer, not a shorter one.

    The budget has already been spent on a token the decoder is not given, so
    it continues from the wrong prompt and emits a sequence nothing
    generated. Merely declining to append left exactly that.
    """
    req = {"prompt": [1, 2, 3]}
    with pytest.raises(ValueError, match="no usable first token"):
        proxy.adopt_prefill_first_token(req, prefill_output)
    assert req["prompt"] == [1, 2, 3]


def test_the_producer_head_chunk_carries_its_id_and_probabilities() -> None:
    """A head chunk without a probability record cannot be compared.

    The oracle reports one probability per generated token; a head chunk that
    reports none makes the two lists different lengths, which reads as a
    mismatch on a run where nothing mismatched.
    """
    prefill_output = {
        "id": "cmpl-1",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "text": "A",
                "logprobs": {"tokens": ["A"], "token_logprobs": [-0.5]},
                "finish_reason": "stop",
                "stop_reason": None,
            }
        ],
    }
    ongoing = proxy.producer_head_chunk(prefill_output, 7, final=False)
    choice = ongoing["choices"][0]
    assert choice["token_ids"] == [7]
    assert choice["logprobs"]["token_logprobs"] == [-0.5]
    # Not final: the decoder is still to speak, so this chunk ends nothing.
    assert choice["finish_reason"] is None

    final = proxy.producer_head_chunk(prefill_output, 7, final=True)
    assert final["choices"][0]["finish_reason"] == "stop"


def test_a_producer_first_token_is_carried_into_the_prompt() -> None:
    req = {"prompt": [1, 2, 3]}
    first = proxy.adopt_prefill_first_token(
        req, {"kv_transfer_params": {"first_tok": 77}}
    )
    assert first == 77
    assert req["prompt"] == [1, 2, 3, 77]


@pytest.mark.asyncio
async def test_the_proxy_tells_every_rank_a_publication_has_no_reader(
    monkeypatch,
) -> None:
    """A single-token answer leaves real publications nobody will read.

    The proxy is the party that knows no decoder was assigned, and the only
    one holding the statuses that identify the publications, so it is the
    one that says so. The writer is still the one that decides.
    """
    port = _free_port()
    writer = _RecordingWriter()
    server = StoragePDAckServer(
        lambda request: ("REJECTED", "no reads here"),
        claim_handler=lambda request: StoragePDClaimAnswer(False, "no reads here"),
        unread_handler=writer.unread,
        bind_host=LOOPBACK,
        port=port,
        advertise_host=LOOPBACK,
        recv_timeout_ms=50,
    )
    monkeypatch.setattr(
        proxy,
        "global_args",
        SimpleNamespace(storage_pd_session="session-1"),
        raising=False,
    )
    try:
        receipt = RawBlockPublicationReceipt(
            "writer-1", 3, 1, "digest", ack_endpoint=server.endpoint
        )
        statuses = [StoragePDStatus.ready("request-1", 0, receipt)]

        await proxy.tell_producers_nobody_will_read(
            statuses, reason="the producer's single token is the whole answer"
        )
    finally:
        server.close(timeout_s=5.0)

    assert len(writer.reports) == 1
    report = writer.reports[0]
    assert report.status == statuses[0]
    assert report.session_id == "session-1"
    assert "single token" in report.reason


@pytest.mark.asyncio
async def test_the_proxy_says_nothing_about_a_producer_with_no_endpoint(
    monkeypatch,
) -> None:
    """A producer that advertised nowhere cannot be told anything.

    Nothing could ever release that publication either way, and inventing a
    destination would only log a failure that says nothing new.
    """
    receipt = RawBlockPublicationReceipt("writer-1", 3, 1, "digest")
    monkeypatch.setattr(
        proxy,
        "global_args",
        SimpleNamespace(storage_pd_session="session-1"),
        raising=False,
    )

    # Returns rather than raising or blocking on a nonexistent peer.
    await proxy.tell_producers_nobody_will_read(
        [StoragePDStatus.ready("request-1", 0, receipt)], reason="no decoder"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["completions", "chat"])
@pytest.mark.parametrize("storage_pd", [False, True])
async def test_response_identifies_the_actual_storage_pd_publication(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, storage_pd: bool
) -> None:
    """Let the client correlate its response with the publication it consumed."""
    _endpoint_env(
        monkeypatch,
        {
            "id": "cmpl-1",
            "created": 0,
            "model": "probe-model",
            "kv_transfer_params": {"first_tok": 7},
        },
    )
    proxy.global_args.storage_pd = storage_pd
    observed: list[str] = []

    async def wait(req_id: str, *_args, **_kwargs):
        observed.append(req_id)
        return []

    async def body():
        return {"prompt": "hello", "messages": [], "max_tokens": 1}

    monkeypatch.setattr(proxy, "wait_decode_kv_ready", wait)
    handler = (
        proxy.handle_completions
        if endpoint == "completions"
        else proxy.handle_chat_completions
    )
    response = await handler(SimpleNamespace(json=body))
    if storage_pd:
        assert response.headers["X-LMCache-PD-Request-ID"] == observed[0]
    else:
        assert "X-LMCache-PD-Request-ID" not in response.headers
    _streamed = [chunk async for chunk in response.body_iterator]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["completions", "chat"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_response_header_failure_resolves_unstarted_stream(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, cancelled: bool
) -> None:
    """Retain cleanup before the ASGI server advances the body iterator."""
    receipt = RawBlockPublicationReceipt(
        "writer", 1, 1, "digest", ack_endpoint="127.0.0.1:9999"
    )
    prefill_response = {
        "id": "cmpl-1",
        "created": 0,
        "model": "probe-model",
        "kv_transfer_params": {"first_tok": 7},
    }
    _endpoint_env(monkeypatch, prefill_response)
    reported: list[StoragePDStatus] = []

    async def wait(req_id: str, *_args, **_kwargs):
        return [StoragePDStatus.ready(req_id, 0, receipt)]

    async def report(statuses, **_kwargs):
        await anyio.sleep(0)
        reported.extend(statuses)
        return []

    async def body():
        return {"prompt": "hello", "messages": [], "max_tokens": 8}

    async def receive():
        raise AssertionError("ASGI 2.4 need not wait for a disconnect message")

    async def send(message):
        assert message["type"] == "http.response.start"
        if cancelled:
            scope.cancel()
            await anyio.sleep(0)
        raise OSError("client disconnected before response headers")

    monkeypatch.setattr(proxy, "wait_decode_kv_ready", wait)
    monkeypatch.setattr(proxy, "tell_producers_nobody_will_read", report)
    handler = (
        proxy.handle_completions
        if endpoint == "completions"
        else proxy.handle_chat_completions
    )
    response = await handler(SimpleNamespace(json=body))
    with anyio.CancelScope() as scope:
        if cancelled:
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
            )
        else:
            with pytest.raises(ClientDisconnect):
                await response(
                    {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
                )
    assert len(reported) == 1
    assert reported[0].publication_receipt() == receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["completions", "chat"])
@pytest.mark.parametrize(
    "decoder_chunks",
    [
        [b"data: [DONE]\n\n"],
        [b"data: [DO", b"NE]\n\n"],
        [],
        [b'data: {"choices":[{"finish_reason":"error"}]}\n\ndata: [DONE]\n\n'],
        [b'data: {"choices":[{"finish_reason":"abort"}]}\n\ndata: [DONE]\n\n'],
        [b'data: {"choices":[{"finish_reason":"other"}]}\n\ndata: [DONE]\n\n'],
        [b'data: [DONE]\n\ndata: {"choices":[{"finish_reason":"stop"}]}\n\n'],
    ],
    ids=[
        "done-only",
        "split-done",
        "empty-eof",
        "error",
        "abort",
        "unknown",
        "done-before-choice",
    ],
)
async def test_unfinished_decoder_reports_a_stream_error(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, decoder_chunks: list[bytes]
) -> None:
    """A failed restore must not appear to finish after the producer's token."""
    receipt = RawBlockPublicationReceipt(
        "writer", 1, 1, "digest", ack_endpoint="127.0.0.1:9999"
    )
    _endpoint_env(
        monkeypatch,
        {
            "id": "cmpl-1",
            "created": 0,
            "model": "probe-model",
            "kv_transfer_params": {"first_tok": 7},
        },
    )
    reported: list[StoragePDStatus] = []
    decoded: list[StoragePDStatus] = []

    async def wait(req_id: str, *_args, **_kwargs):
        return [StoragePDStatus.ready(req_id, 0, receipt)]

    async def report(statuses, **_kwargs):
        reported.extend(statuses)
        return []

    async def body():
        return {"prompt": "hello", "messages": [], "max_tokens": 8}

    async def decode(*_args):
        for chunk in decoder_chunks:
            yield chunk

    monkeypatch.setattr(proxy, "wait_decode_kv_ready", wait)
    monkeypatch.setattr(proxy, "tell_producers_nobody_will_read", report)
    monkeypatch.setattr(proxy, "stream_service_response", decode)
    monkeypatch.setattr(proxy, "_trace_storage_pd_decode", decoded.extend)
    handler = (
        proxy.handle_completions
        if endpoint == "completions"
        else proxy.handle_chat_completions
    )
    response = await handler(SimpleNamespace(json=body))
    streamed = b"".join([chunk async for chunk in response.body_iterator])
    events = [part.removeprefix(b"data: ") for part in streamed.split(b"\n\n") if part]
    errors = [
        part for part in events if part != b"[DONE]" and "error" in json.loads(part)
    ]
    assert len(errors) == 1
    assert json.loads(errors[0])["error"]["type"] == "server_error"
    if b"[DONE]" in events:
        assert events.index(errors[0]) < events.index(b"[DONE]")
    assert len(reported) == 1
    assert reported[0].publication_receipt() == receipt
    assert decoded == []


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["completions", "chat"])
@pytest.mark.parametrize("token_text", ["answer", ""])
async def test_decode_trace_waits_for_a_generated_token(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, token_text: str
) -> None:
    """Usage, keepalives and an empty choice are not evidence of decode."""
    receipt = RawBlockPublicationReceipt("writer", 1, 1, "digest")
    _endpoint_env(
        monkeypatch,
        {
            "id": "cmpl-1",
            "created": 0,
            "model": "probe-model",
            "kv_transfer_params": {"first_tok": 7},
        },
    )
    events = [
        b": keepalive\n\n",
        b'data: {"choices":[],"usage":{"completion_tokens":0}}\n\n',
        b'data: {"choices":[{"text":"","token_ids":[]}]}\n\n',
        b"data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "text": token_text,
                        "token_ids": [9],
                        "finish_reason": "stop",
                    }
                ]
            }
        ).encode()
        + b"\n\n",
        b"data: [DONE]\n\n",
    ]
    observed_at: list[int] = []
    position = -1

    async def wait(req_id: str, *_args, **_kwargs):
        return [StoragePDStatus.ready(req_id, 0, receipt)]

    async def body():
        return {"prompt": "hello", "messages": [], "max_tokens": 8}

    async def decode(*_args):
        nonlocal position
        for position, event in enumerate(events):
            yield event

    monkeypatch.setattr(proxy, "wait_decode_kv_ready", wait)
    monkeypatch.setattr(proxy, "stream_service_response", decode)
    monkeypatch.setattr(
        proxy,
        "_trace_storage_pd_decode",
        lambda _statuses: observed_at.append(position),
    )
    handler = (
        proxy.handle_completions
        if endpoint == "completions"
        else proxy.handle_chat_completions
    )
    response = await handler(SimpleNamespace(json=body))
    _streamed = [chunk async for chunk in response.body_iterator]
    assert observed_at == [3]


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", [b"\n\n", b"\r\n\r\n"])
@pytest.mark.parametrize("split_bytes", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_decoder_validation_preserves_complete_events(
    monkeypatch: pytest.MonkeyPatch,
    separator: bytes,
    split_bytes: bool,
    finish_reason: str,
) -> None:
    """Keep valid UTF-8 and event boundaries across arbitrary transport chunks."""
    payload = json.dumps(
        {"choices": [{"text": "é", "finish_reason": finish_reason}]}, ensure_ascii=False
    ).encode()
    expected = [b"data: " + payload + separator, b"data: [DONE]" + separator]
    wire = b"".join(expected)
    chunks = (
        [wire[index : index + 1] for index in range(len(wire))]
        if split_bytes
        else [wire]
    )

    async def decode(*_args):
        for chunk in chunks:
            yield chunk

    monkeypatch.setattr(proxy, "stream_service_response", decode)
    completion = proxy.DecoderStreamCompletion()
    streamed = [
        chunk
        async for chunk in proxy.stream_decoder_response(
            None,
            "/v1/completions",
            {},
            completion,  # type: ignore[arg-type]
        )
    ]
    assert completion.succeeded
    assert streamed == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("temporary", [False, True])
async def test_cancelled_unread_admission_closes_only_a_temporary_client(
    monkeypatch: pytest.MonkeyPatch, temporary: bool
) -> None:
    """Do not leave a helper-owned retry worker running after admission cancellation."""
    # Standard
    import asyncio

    client = StoragePDUnreadClient(max_live_obligations=1, poll_interval_s=0.002)
    close_results: list[bool] = []
    admission_blocked = asyncio.Event()
    real_offer = client.offer
    real_close = client.close

    def offer(*args, **kwargs):
        obligation = real_offer(*args, **kwargs)
        if obligation is None:
            admission_blocked.set()
        return obligation

    def close(timeout_s: float = 5.0) -> bool:
        stopped = real_close(timeout_s)
        close_results.append(stopped)
        return stopped

    monkeypatch.setattr(client, "offer", offer)
    monkeypatch.setattr(client, "close", close)
    monkeypatch.setattr(client, "_exchange", lambda *_args: None)
    monkeypatch.setattr(proxy, "StoragePDUnreadClient", lambda **_kwargs: client)
    monkeypatch.setattr(
        proxy.app.state, "storage_pd_unread_client", None if temporary else client
    )
    monkeypatch.setattr(
        proxy,
        "global_args",
        SimpleNamespace(storage_pd_session="session-1"),
        raising=False,
    )
    statuses = [
        StoragePDStatus.ready(
            f"request-{index}",
            0,
            RawBlockPublicationReceipt(
                "writer", index + 1, 1, f"digest-{index}", ack_endpoint="127.0.0.1:9999"
            ),
        )
        for index in range(2)
    ]
    task = asyncio.create_task(
        proxy.tell_producers_nobody_will_read(statuses, reason="no decoder")
    )
    try:
        await asyncio.wait_for(admission_blocked.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert close_results == ([True] if temporary else [])
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        real_close()
