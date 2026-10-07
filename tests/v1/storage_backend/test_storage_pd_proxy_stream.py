# SPDX-License-Identifier: Apache-2.0
"""Drive the real proxy's streamed answer, with fake engines behind it.

The comparator that judges a serving run reads the proxy's stream. Its own
fixtures describe the shape they expect, and a fixture is only worth what
its shape is: one written from the docstring rather than from the program
under test passes while the real stream is missing the field the comparison
needs. So the shapes are taken from here -- the actual endpoint, the actual
head-chunk construction, the actual budget arithmetic -- with nothing mocked
but the two engines and the tokenizer.

No model, no GPU and no sockets: the prefiller and the decoder are httpx
mock transports, and the proxy is driven through its ASGI app.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any
import json

# Third Party
from fastapi.testclient import TestClient
import httpx
import pytest

# First Party
from examples.disagg_prefill import disagg_proxy_server as proxy


class _Engines:
    """The two engines and the tokenizer, answering from canned data."""

    def __init__(
        self,
        *,
        prompt_tokens: list[int],
        first_tok_id: Any,
        producer_finish: str | None,
        decoder_chunks: list[dict],
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.first_tok_id = first_tok_id
        self.producer_finish = producer_finish
        self.decoder_chunks = decoder_chunks
        self.prefill_requests: list[dict] = []
        self.decode_requests: list[dict] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": self.prompt_tokens})
        if body.get("stream"):
            self.decode_requests.append(body)
            payload = b"".join(
                b"data: " + json.dumps(chunk).encode() + b"\n\n"
                for chunk in self.decoder_chunks
            )
            return httpx.Response(
                200,
                content=payload + b"data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        self.prefill_requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-producer",
                "object": "text_completion",
                "created": 1700000000,
                "model": "a-test-model",
                "choices": [
                    {
                        "index": 0,
                        "text": "Ok",
                        "finish_reason": self.producer_finish,
                        "logprobs": {
                            "tokens": ["Ok"],
                            "token_logprobs": [-0.25],
                        },
                    }
                ],
                "kv_transfer_params": {"first_tok": self.first_tok_id},
            },
        )


class _AlwaysReady(dict):
    """Reports every request's KV as ready for one rank."""

    def __getitem__(self, key: str) -> int:
        return 1

    def get(self, key: str, default: Any = None) -> int:
        return 1


def _install(engines: _Engines) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(engines.handle),
        base_url="http://engine.invalid",
    )
    info = proxy.ClientInfo(client, "127.0.0.1", [1], [2])
    proxy.app.state.prefill_clients = [info]
    proxy.app.state.decode_clients = [info]
    proxy.app.state.total_clients = [info]


@pytest.fixture(autouse=True)
def proxy_without_a_barrier(monkeypatch):
    """A proxy with real endpoints, no barrier and no buffer accounting.

    The READY barrier has its own tests; what is under test here is the
    answer the proxy composes, which runs the same way either side of it.
    """

    class _Args:
        storage_pd = False
        chunk_size = 256
        storage_pd_ready_timeout_s = 5.0

    monkeypatch.setattr(proxy, "global_args", _Args(), raising=False)
    monkeypatch.setattr(proxy, "pd_buffer_semaphore", None, raising=False)
    # Stands in for the prefiller's "KV is ready" notification, which is
    # what the barrier waits on and has its own tests. Everything either
    # side of the barrier is the real thing.
    monkeypatch.setattr(proxy.app.state, "finished_reqs", _AlwaysReady())
    before = (
        proxy.app.state.prefill_clients,
        proxy.app.state.decode_clients,
        proxy.app.state.total_clients,
    )
    yield
    (
        proxy.app.state.prefill_clients,
        proxy.app.state.decode_clients,
        proxy.app.state.total_clients,
    ) = before


def _stream(body: dict) -> list[dict]:
    """POST one completion and return its streamed chunks, in order.

    The client is used without its context manager on purpose: entering it
    runs the app's lifespan, which would build real engine clients from the
    command line and bind the proxy's notification socket. What is under
    test is the endpoint, with the engines this test installed.
    """
    client = TestClient(proxy.app)
    response = client.post("/v1/completions", json=body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    chunks = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    return chunks


def _decoder_chunk(text: str, token_id: int, finish: str | None = None) -> dict:
    return {
        "id": "cmpl-decoder",
        "object": "text_completion",
        "created": 1700000001,
        "model": "a-test-model",
        "choices": [
            {
                "index": 0,
                "text": text,
                "token_ids": [token_id],
                "finish_reason": finish,
                "logprobs": {"tokens": [text], "token_logprobs": [-0.5]},
            }
        ],
    }


def test_the_producers_token_reaches_the_client_with_its_id():
    """The first token is the producer's, and it must arrive identifiable.

    A comparison of token identity cannot be made from text: two different
    ids can render the same string. The head chunk therefore carries the
    id the producer reported, and the decoder's own chunks follow it.
    """
    engines = _Engines(
        prompt_tokens=[1, 2, 3],
        # What a producer given a one-token budget actually reports: it
        # stopped because the budget ran out, not because the answer ended.
        # Passing that through would end the stream after one token.
        first_tok_id=4242,
        producer_finish="length",
        decoder_chunks=[
            _decoder_chunk(" then", 77),
            _decoder_chunk(" this", 88, finish="stop"),
        ],
    )
    _install(engines)

    chunks = _stream({"prompt": "hello", "max_tokens": 8, "stream": True})

    head = chunks[0]["choices"][0]
    assert head["text"] == "Ok"
    assert head["token_ids"] == [4242], "the producer's token id must be on the wire"
    assert head["logprobs"] == {"tokens": ["Ok"], "token_logprobs": [-0.25]}
    # A continuing answer's head chunk carries no finish reason, even
    # though the producer reported one: its budget ran out, the answer did
    # not, and a reason here ends the stream for a client reading it.
    assert head.get("finish_reason") is None
    assert head.get("stop_reason") is None

    ids = [chunk["choices"][0].get("token_ids") for chunk in chunks]
    assert ids == [[4242], [77], [88]]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    # The decoder was asked for one fewer token, because the producer spent
    # one of them.
    assert engines.decode_requests[0]["max_tokens"] == 7
    assert engines.prefill_requests[0]["max_tokens"] == 1


def test_a_one_token_answer_never_reaches_the_decoder():
    """A budget of one is spent entirely by the producer.

    Forwarding it would ask the decoder for zero tokens, and the answer the
    client is owed is already complete. The stream is the head chunk and
    nothing else -- with its finish reason, because this is the end.
    """
    engines = _Engines(
        prompt_tokens=[5, 6],
        first_tok_id=99,
        producer_finish="length",
        decoder_chunks=[_decoder_chunk(" unreachable", 1)],
    )
    _install(engines)

    chunks = _stream({"prompt": "hi", "max_tokens": 1, "stream": True})

    assert len(chunks) == 1
    head = chunks[0]["choices"][0]
    assert head["token_ids"] == [99]
    assert head["finish_reason"] == "length"
    assert engines.decode_requests == [], "the decoder must not be asked"


def test_a_producer_that_already_stopped_ends_the_answer():
    """A producer reporting anything but length or none has finished.

    Asking the decoder to continue past a stop would generate tokens the
    client did not ask for, and a comparison against an oracle that also
    stopped would then differ on every one of them.
    """
    engines = _Engines(
        prompt_tokens=[7],
        first_tok_id=11,
        producer_finish="stop",
        decoder_chunks=[_decoder_chunk(" unreachable", 2)],
    )
    _install(engines)

    chunks = _stream({"prompt": "hi", "max_tokens": 16, "stream": True})

    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["finish_reason"] == "stop"
    assert engines.decode_requests == []


def test_a_producer_without_a_token_id_is_refused():
    """A missing first token is not an empty one.

    Continuing without it drops a token from the answer, and the client is
    given a shorter completion that looks complete. Refusing is what makes
    that visible.
    """
    engines = _Engines(
        prompt_tokens=[1],
        first_tok_id=None,
        producer_finish=None,
        decoder_chunks=[_decoder_chunk(" x", 3, finish="stop")],
    )
    _install(engines)

    client = TestClient(proxy.app, raise_server_exceptions=False)
    response = client.post(
        "/v1/completions", json={"prompt": "hi", "max_tokens": 4, "stream": True}
    )
    assert response.status_code >= 500
    assert engines.decode_requests == []


def test_a_budget_below_one_is_refused_before_anything_runs():
    """Zero tokens is not a handoff, and neither is a negative budget."""
    engines = _Engines(
        prompt_tokens=[1],
        first_tok_id=5,
        producer_finish=None,
        decoder_chunks=[],
    )
    _install(engines)

    client = TestClient(proxy.app, raise_server_exceptions=False)
    response = client.post(
        "/v1/completions", json={"prompt": "hi", "max_tokens": 0, "stream": True}
    )
    assert response.status_code >= 400
    assert engines.prefill_requests == []


def test_the_stream_carries_everything_a_token_comparison_needs(tmp_path):
    """Record one real stream, in the shape an oracle is compared against.

    A comparator's fixtures are worth what their shapes are: one written
    from prose rather than from the program that emits them passes while
    the real stream is missing the field the comparison rests on. So the
    requirement is asserted here, against the endpoint, and the recording
    is written out for a comparator to be checked against.
    """
    engines = _Engines(
        prompt_tokens=[10, 11],
        first_tok_id=101,
        producer_finish="length",
        decoder_chunks=[
            _decoder_chunk(" b", 102),
            _decoder_chunk(" c", 103, finish="stop"),
        ],
    )
    _install(engines)

    chunks = _stream({"prompt": "a", "max_tokens": 3, "stream": True})

    # One record per generated token, each with an id, a text and a
    # probability, and exactly one terminal reason at the end.
    assert len(chunks) == 3
    for chunk in chunks:
        choice = chunk["choices"][0]
        assert len(choice["token_ids"]) == 1
        assert isinstance(choice["token_ids"][0], int)
        assert not isinstance(choice["token_ids"][0], bool)
        logprobs = choice["logprobs"]
        assert len(logprobs["tokens"]) == 1
        assert len(logprobs["token_logprobs"]) == 1
    reasons = [chunk["choices"][0].get("finish_reason") for chunk in chunks]
    assert reasons == [None, None, "stop"]

    recording = tmp_path / "proxy-stream.jsonl"
    recording.write_text("".join(json.dumps(chunk) + "\n" for chunk in chunks))
    assert recording.read_text().count("\n") == 3
