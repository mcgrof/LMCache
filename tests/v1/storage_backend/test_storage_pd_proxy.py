# SPDX-License-Identifier: Apache-2.0
"""Durable READY barrier tests for the disaggregated prefill proxy."""

# Standard
import time

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
        producer_instance_id="pid:1",
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
