# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Optional
import argparse
import asyncio
import itertools
import json
import math
import os
import time
import uuid

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import msgspec
import numpy as np
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.pd_backend import (
    PDMsg,
    ProxyNotif,
    StoragePDStatus,
)
from lmcache.v1.storage_backend.storage_pd_ack import (
    StoragePDAckWire,
    StoragePDUnreadReply,
    StoragePDUnreadRequest,
)
from lmcache.v1.storage_backend.storage_pd_protocol import (
    order_storage_pd_ready_statuses,
)

logger = init_logger(__name__)


class WeightedSemaphore:
    """Async semaphore with variable-weight acquire.

    Limits in-flight PD token usage: each request holds ceil(L/chunk_size)
    slots until decoding starts, preventing decoder buffer exhaustion deadlocks.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._available = capacity
        self._lock = asyncio.Condition()

    async def acquire(self, slots: int) -> None:
        """Acquire *slots* from the semaphore, blocking until available.

        Args:
            slots: Number of slots to acquire (must be <= capacity).

        Raises:
            ValueError: If slots exceeds total capacity (would block forever).
        """
        if slots > self._capacity:
            raise ValueError(
                f"Requested {slots} slots exceeds total capacity {self._capacity}"
            )
        async with self._lock:
            await self._lock.wait_for(lambda: self._available >= slots)
            self._available -= slots

    async def release(self, slots: int) -> None:
        """Return *slots* to the semaphore and wake waiting acquirers.

        Args:
            slots: Number of slots to release. No-op if <= 0.
        """
        if slots <= 0:
            return
        async with self._lock:
            self._available += slots
            self._lock.notify_all()

    @property
    def available(self) -> int:
        """Number of slots currently available."""
        return self._available


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize clients

    # Build prefill clients with CSV-based broadcast pairing
    pref_hosts = global_args.prefiller_host
    pref_ports = global_args.prefiller_port

    def pair_hosts_and_ports(hosts, ports, count=None):
        """
        Flexible host-port pairing with expansion strategies.

        Multiple pairing strategies:
        1. Single host + single port + count: Generate incremental ports on same host
        2. Single host + multiple ports: Pair the host with each port
        3. Multiple hosts + single port: Pair each host with the same port
        4. Multiple hosts + multiple ports: Strict one-to-one pairing
           (must have same length)
        """
        # Ensure lists
        if not isinstance(hosts, list):
            hosts = [hosts]
        if not isinstance(ports, list):
            ports = [ports]
        # Single host/port with count -> incremental ports
        if len(hosts) == 1 and len(ports) == 1:
            if count is None or count <= 1:
                return [(hosts[0], ports[0])]
            else:
                return [(hosts[0], ports[0] + i) for i in range(count)]
        # Expand single host to multiple ports
        if len(hosts) == 1:
            return [(hosts[0], p) for p in ports]
        # Expand single port to multiple hosts
        if len(ports) == 1:
            return [(h, ports[0]) for h in hosts]
        # Strict one-to-one pairing when both lists are provided
        if len(hosts) != len(ports):
            raise ValueError(
                "Length mismatch between hosts and ports lists for pairing"
            )
        return list(zip(hosts, ports, strict=False))

    prefill_pairs = pair_hosts_and_ports(
        pref_hosts, pref_ports, global_args.num_prefillers
    )
    for host, port in prefill_pairs:
        prefiller_base_url = f"http://{host}:{int(port)}"
        prefill_client = httpx.AsyncClient(timeout=None, base_url=prefiller_base_url)
        app.state.prefill_clients.append(
            ClientInfo(
                prefill_client,
            )
        )

    # Build decoder clients with CSV-based broadcast pairing
    dec_hosts = global_args.decoder_host
    dec_ports = global_args.decoder_port

    decoder_pairs = pair_hosts_and_ports(dec_hosts, dec_ports, global_args.num_decoders)

    # Whether the ports increase per instances
    # (only when using single host/port with num_decoders > 1)
    incremental_mode = (
        len(dec_hosts) == 1 and len(dec_ports) == 1 and global_args.num_decoders > 1
    )

    for i, (host, port) in enumerate(decoder_pairs):
        decoder_base_url = f"http://{host}:{int(port)}"
        decode_client = httpx.AsyncClient(timeout=None, base_url=decoder_base_url)
        if incremental_mode:
            init_ports = [p + i for p in global_args.decoder_init_port]
            alloc_ports = [p + i for p in global_args.decoder_alloc_port]
        else:
            # Use the provided ports as-is
            # (suitable when different hosts can reuse same port numbers)
            init_ports = list(global_args.decoder_init_port)
            alloc_ports = list(global_args.decoder_alloc_port)

        app.state.decode_clients.append(
            ClientInfo(
                decode_client,
                host,
                init_ports,
                alloc_ports,
            )
        )

    app.state.total_clients = app.state.prefill_clients + app.state.decode_clients

    app.state.zmq_task = asyncio.create_task(zmq_pull_server())

    global pd_buffer_semaphore
    kv_bytes_per_token = compute_kv_bytes_per_token(global_args.model)
    capacity_slots = global_args.pd_buffer_size // (
        kv_bytes_per_token * global_args.chunk_size
    )
    pd_buffer_semaphore = WeightedSemaphore(capacity_slots)
    logger.info(
        "PD buffer semaphore: capacity=%d slots"
        " (%d bytes / (%d bytes/tok * %d chunk_size)) for model %s.",
        capacity_slots,
        global_args.pd_buffer_size,
        kv_bytes_per_token,
        global_args.chunk_size,
        global_args.model,
    )

    yield

    # Shutdown: stop the receiver before the clients it feeds, and close
    # every client even if one of them raises on the way out.
    global run_proxy
    run_proxy = False
    zmq_task = app.state.zmq_task
    try:
        # The receiver spends its idle time inside recv(), which the flag
        # alone cannot interrupt: an idle proxy would wait here forever.
        # Cancelling wakes it, and its own cleanup closes the socket.
        zmq_task.cancel()
        with suppress(asyncio.CancelledError):
            await zmq_task
    finally:
        for client in app.state.total_clients:
            try:
                await client.aclose()
            except Exception:
                logger.exception("Failed to close a proxy HTTP client")


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


class StatsCalculator:
    def __init__(self):
        self._stats = []
        self._last_log_time = time.time()

    def add(self, value):
        self._stats.append(value)
        if time.time() - self._last_log_time > 5:
            self._log_stats()
            self._last_log_time = time.time()

    def _log_stats(self):
        # Print average, median, and 99th percentile
        np_arr = np.array(self._stats) * 1000
        output_str = (
            f"\nNum requests: {len(self._stats)}"
            + "\nPrefill node TTFT stats:"
            + f"\n - Average (ms): {np.mean(np_arr)}"
            + f"\n - Median (ms): {np.median(np_arr)}"
            + f"\n - 99th Percentile (ms): {np.percentile(np_arr, 99)}\n"
        )
        print(
            "===============================",
            output_str,
            "===============================",
        )


stats_calculator = StatsCalculator()
counter = 0


def csv_ints(s):
    return [int(x) for x in s.split(",")]


def csv_strs(s):
    return [x.strip() for x in s.split(",")]


def compute_kv_bytes_per_token(model_name: str) -> int:
    """Return the number of KV cache bytes per token for *model_name*.

    Reads num_hidden_layers, num_key_value_heads, head_dim, and torch_dtype
    from the HuggingFace config without downloading model weights.

    Args:
        model_name: HuggingFace model id or local path.

    Returns:
        Bytes per token across all layers and both K/V tensors.
    """
    # Third Party
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    num_layers: int = cfg.num_hidden_layers
    num_kv_heads: int = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim: int = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    # 4 bytes for float32, 2 bytes for float16/bfloat16 (the common default)
    torch_dtype = str(getattr(cfg, "torch_dtype", "bfloat16"))
    dtype_bytes = 4 if "float32" in torch_dtype else 2
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--prefiller-port", type=csv_ints, default=[8100])
    parser.add_argument("--num-prefillers", type=int, default=1)
    parser.add_argument("--decoder-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--decoder-port", type=csv_ints, default=[8200])
    parser.add_argument("--decoder-init-port", type=csv_ints, default=[8300])
    parser.add_argument("--decoder-alloc-port", type=csv_ints, default=[8400])

    parser.add_argument("--num-decoders", type=int, default=1)
    parser.add_argument("--proxy-host", type=str, default="localhost")
    parser.add_argument("--proxy-port", type=int, default=8500)
    parser.add_argument(
        "--storage-pd",
        action="store_true",
        help="Require durable per-rank raw-block READY statuses before decode.",
    )
    parser.add_argument(
        "--storage-pd-session",
        type=str,
        default=os.environ.get("LMCACHE_STORAGE_PD_SESSION", ""),
        help=(
            "The producer/consumer session identifier this deployment uses. "
            "It must be the same string every participating engine was given "
            "(LMCACHE_STORAGE_PD_SESSION), because a producer only answers "
            "control messages naming its own session."
        ),
    )
    parser.add_argument(
        "--storage-pd-ready-timeout-s",
        type=float,
        default=30.0,
        help=(
            "Maximum time for one storage-P/D handoff, covering both the "
            "prefiller's answer and every READY status after it."
        ),
    )

    # PD buffer concurrency limiting. A weighted semaphore caps in-flight
    # chunk slots to prevent decoder buffer exhaustion deadlocks.
    # capacity_slots = pd_buffer_size // (kv_bytes_per_token * chunk_size)
    # kv_bytes_per_token is derived from the model config automatically.
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help=(
            "HuggingFace model name or local path. Used to derive"
            " kv_bytes_per_token for the PD buffer semaphore capacity."
        ),
    )
    parser.add_argument(
        "--pd-buffer-size",
        type=int,
        default=2 * 1024 * 1024 * 1024,  # 2 GB
        help=(
            "PD transfer buffer size in bytes (must match the decoder's"
            " LMCache config). Used to derive the in-flight slot capacity."
            " Default: 2 GB."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="LMCache chunk size in tokens (must match the LMCache config).",
    )

    args = parser.parse_args()
    if args.storage_pd_ready_timeout_s <= 0:
        parser.error("--storage-pd-ready-timeout-s must be positive")
    if args.storage_pd and not args.storage_pd_session:
        # Without it this proxy cannot resolve a publication nobody will
        # read: the producer answers only for its own session, so the
        # extents would stay held until the writer stopped publishing.
        parser.error(
            "--storage-pd needs --storage-pd-session (or "
            "LMCACHE_STORAGE_PD_SESSION) set to the same value every "
            "participating engine was given"
        )
    return args


@dataclass
class ClientInfo:
    client: httpx.AsyncClient
    host: Optional[str] = None
    init_port: Optional[list[int]] = None
    alloc_port: Optional[list[int]] = None

    async def aclose(self) -> None:
        """Close the HTTP client this record owns."""
        await self.client.aclose()


# Initialize variables to hold the persistent clients
app.state.prefill_clients = []
app.state.decode_clients = []
app.state.total_clients = []

"""
client_request and prefill/decode map
key:   str    - unique id for requests across same conversation
value: tuple  - (tokenization_client, prefiller_client, decoder_client)
"""
app.state.bound_clients = {}

# Keep finished reqs
app.state.finished_reqs = defaultdict(int)
# Requests currently waiting for a storage P/D barrier. A status is only
# admitted while its request is registered here, so traffic that arrives for
# a request already finished, already failed, or never seen cannot bring its
# state back into existence. These are plain dicts for that reason: a
# defaultdict would create an entry for whatever key was read.
app.state.storage_pd_active = {}
app.state.storage_pd_statuses = {}
app.state.storage_pd_failures = {}

pd_buffer_semaphore: Optional[WeightedSemaphore] = None


zmq_ctx = zmq.asyncio.Context()
run_proxy = True  # Shutdown flag


async def tell_producers_nobody_will_read(
    statuses: list[StoragePDStatus],
    *,
    reason: str,
    timeout_ms: int = 2000,
) -> None:
    """Tell each producer rank that its publication has no reader.

    The publication is real and durable; what is missing is a decoder, so
    nothing is ever going to acknowledge it and the producer would hold
    those extents until its own admission bound stopped it publishing. This
    proxy is the party that knows no reader was assigned, so it says so --
    and nothing here fabricates a read acknowledgement, because no read
    happened. The producer still refuses to release a publication some
    consumer claimed, so being wrong about this costs nothing.

    A failure is logged and nothing else: the extents stay held, which is
    the same outcome as not asking.
    """
    for status in statuses:
        if status.state != "READY" or not status.ack_endpoint:
            continue
        request = StoragePDUnreadRequest(
            status=status,
            session_id=global_args.storage_pd_session,
            nonce=uuid.uuid4().hex,
            reason=reason,
        )
        socket = zmq_ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        try:
            socket.connect(f"tcp://{status.ack_endpoint}")
            await socket.send(msgspec.msgpack.encode(request))
            raw = await socket.recv()
            reply = msgspec.msgpack.decode(raw, type=StoragePDAckWire)
            if not isinstance(reply, StoragePDUnreadReply):
                logger.error(
                    "Storage P/D got a %s answering for the unread publication %s",
                    type(reply).__name__,
                    status.req_id,
                )
            elif reply.nonce != request.nonce:
                logger.error(
                    "Storage P/D got an answer about %s while asking about "
                    "%s; discarding it",
                    reply.nonce,
                    request.nonce,
                )
            elif not reply.released:
                logger.error(
                    "Storage P/D producer %s kept the unread publication %s: %s",
                    status.ack_endpoint,
                    status.req_id,
                    reply.reason or "no reason given",
                )
        except (zmq.ZMQError, msgspec.DecodeError, msgspec.ValidationError):
            logger.exception(
                "Storage P/D could not tell %s that %s has no reader; its "
                "extents stay held",
                status.ack_endpoint,
                status.req_id,
            )
        finally:
            socket.close(linger=0)


async def zmq_pull_server():
    socket = zmq_ctx.socket(zmq.PULL)
    # Close the socket from one place that every exit runs through: a bind
    # failure, the shutdown flag, and the cancellation that wakes a blocked
    # recv() all leave this coroutine by a different route.
    try:
        proxy_url = f"{global_args.proxy_host}:{global_args.proxy_port}"
        try:
            socket.bind(f"tcp://{proxy_url}")
        except zmq.ZMQError:
            logger.exception("ZMQ proxy server failed to bind on %s", proxy_url)
            return
        logger.info("ZMQ proxy server started on %s", proxy_url)

        while run_proxy:
            try:
                msg_bytes = await socket.recv()
            except zmq.Again:
                await asyncio.sleep(0.01)  # Avoid busy loop
                continue
            except zmq.ZMQError as exc:
                if exc.errno in (zmq.ETERM, zmq.ENOTSOCK):
                    break
                logger.warning("ZMQ recv error: %s", exc)
                await asyncio.sleep(0.05)
                continue

            try:
                msg = msgspec.msgpack.decode(msg_bytes, type=PDMsg)
            except msgspec.DecodeError as exc:
                logger.warning("ZMQ received non-PD message: %s", exc)
                continue
            except Exception as exc:
                logger.exception("ZMQ message decode failed: %s", exc)
                continue

            if isinstance(msg, StoragePDStatus):
                record_storage_pd_status(msg)
                continue

            if not isinstance(msg, ProxyNotif):
                logger.debug("ZMQ ignored message type: %s", type(msg).__name__)
                continue

            if global_args.storage_pd:
                logger.debug(
                    "Ignoring legacy prefill notification for storage P/D req %s",
                    msg.req_id,
                )
                continue

            req_id = msg.req_id
            app.state.finished_reqs[req_id] += 1
            logger.debug("Prefill of req %s done.", req_id)

    finally:
        socket.close(linger=0)
        logger.info("ZMQ PULL server stopped.")


async def send_request_to_service(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """
    Send a request to a service using a persistent client.
    """

    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    return response


async def stream_service_response(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """
    Asynchronously stream the response from a service using a persistent client.
    """
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    async with client.stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


def round_robin_pick_client(clients, idx):
    return clients[idx % len(clients)]


round_robin_counter = itertools.count()


def round_robin_pick_clients() -> tuple[ClientInfo, ClientInfo, ClientInfo]:
    idx = next(round_robin_counter)
    tokenization_client = round_robin_pick_client(app.state.total_clients, idx)
    prefill_client = round_robin_pick_client(app.state.prefill_clients, idx)
    decode_client = round_robin_pick_client(app.state.decode_clients, idx)
    return tokenization_client, prefill_client, decode_client


def take_prefill_budget(req_data: dict) -> int:
    """Read the caller's token budget and give the prefiller one token.

    A handoff spends one token on the prefiller, so a request asking for a
    single token leaves the decoder none, which the engine rejects. Refuse it
    here, naming the reason, rather than forwarding a request that cannot be
    served. Both spellings of the budget are read, because a chat request
    carries only ``max_completion_tokens`` and a request whose budget this
    cannot find is a 500 rather than a refusal with a reason.
    """
    budget = req_data.get("max_tokens")
    if budget is None:
        budget = req_data.get("max_completion_tokens")
    if budget is None:
        raise ValueError(
            "a prefill/decode handoff needs max_tokens (or "
            "max_completion_tokens) so the decoder's budget can be computed"
        )
    budget = int(budget)
    if budget < 1:
        raise ValueError(
            f"a generation budget must be at least one token; got {budget}"
        )
    req_data["max_tokens"] = 1
    return budget


def adopt_prefill_first_token(req_data: dict, prefill_output: dict) -> int:
    """Carry the prefiller's one token into the decoder's prompt.

    Raises when the producer reported no usable id. Continuing without it is
    not a lesser answer, it is a different one: the budget has already been
    spent on a token the decoder is not given, so the decoder continues from
    the wrong prompt and emits a sequence that was never generated.
    """
    first_tok_id = (prefill_output.get("kv_transfer_params") or {}).get("first_tok")
    if not isinstance(first_tok_id, int) or isinstance(first_tok_id, bool):
        raise ValueError(
            "prefiller reported no usable first token id "
            f"({first_tok_id!r}); the decoder cannot continue from a prompt "
            "missing the token the budget was spent on"
        )
    req_data["prompt"].append(first_tok_id)
    return first_tok_id


def producer_head_chunk(
    prefill_output: dict,
    first_tok_id: int,
    *,
    final: bool,
) -> dict:
    """Render the prefiller's one token as a completion chunk.

    Carries the id and the probability record beside the text. A client
    comparing generated tokens cannot recover an id from text without
    retokenizing, which is a different operation and can disagree; and a
    probability list that omits this token cannot be compared against one
    that includes it.
    """
    choice = (prefill_output.get("choices") or [{}])[0]
    return {
        "id": prefill_output["id"],
        "object": "text_completion",
        "created": prefill_output["created"],
        "model": prefill_output["model"],
        "choices": [
            {
                "index": 0,
                "text": choice.get("text", ""),
                "token_ids": [first_tok_id],
                "logprobs": choice.get("logprobs"),
                "finish_reason": choice.get("finish_reason") if final else None,
                "stop_reason": choice.get("stop_reason") if final else None,
            }
        ],
        "usage": None,
    }


def producer_answer_is_complete(budget: int, prefill_output: dict) -> bool:
    """Whether the prefiller's single token is the whole answer.

    Two ways that happens: the caller asked for one token, or the producer
    stopped of its own accord. Either way the decoder has nothing to
    generate, and asking it for zero tokens is a request the engine refuses.
    """
    if budget <= 1:
        return True
    finish = (prefill_output.get("choices") or [{}])[0].get("finish_reason")
    return finish not in (None, "", "length")


def record_storage_pd_status(msg: StoragePDStatus) -> str:
    """Take one producer status into the barrier's state, or refuse it.

    Returns what was done, for the caller's log and for tests: ``ignored``
    for a request that is not waiting on a barrier, ``failed`` for a
    terminal status, ``conflict`` for a second, different READY from a rank
    that already reported, and ``recorded`` otherwise.
    """
    req_id = msg.req_id
    if not _pd_request_is_active(req_id):
        # Late, duplicate or unknown: the request has already reached a
        # terminal state or was never registered. Recording it would
        # recreate state a barrier just cleared, and nothing would ever
        # clear it again.
        logger.debug(
            "Storage P/D ignoring %s for inactive req %s rank %d",
            msg.state,
            req_id,
            msg.tp_rank,
        )
        return "ignored"
    if msg.state != "READY":
        app.state.storage_pd_failures[req_id] = msg
        logger.error(
            "Storage P/D producer failed req %s rank %d at %s: %s",
            req_id,
            msg.tp_rank,
            msg.error_stage,
            msg.error_text,
        )
        return "failed"
    ranks = app.state.storage_pd_statuses.setdefault(req_id, {})
    previous = ranks.get(msg.tp_rank)
    if previous is not None and previous != msg:
        app.state.storage_pd_failures[req_id] = StoragePDStatus(
            req_id=req_id,
            writer_epoch=msg.writer_epoch,
            tp_rank=msg.tp_rank,
            state="FAILED",
            error_stage="PROXY_BARRIER",
            error_text="conflicting READY statuses for one TP rank",
        )
        return "conflict"
    ranks[msg.tp_rank] = msg
    logger.debug(
        "Storage P/D req %s rank %d published checkpoint %d.",
        req_id,
        msg.tp_rank,
        msg.checkpoint_seq,
    )
    return "recorded"


def _register_pd_request(req_id: str) -> None:
    """Admit statuses for this request from now until it is cleared.

    Registration happens before the prefiller is contacted, because a
    producer can report READY before its HTTP response comes back.
    """
    app.state.storage_pd_active[req_id] = time.monotonic()
    app.state.storage_pd_statuses.setdefault(req_id, {})


def _pd_request_is_active(req_id: str) -> bool:
    """Whether this request is still waiting for its barrier."""
    return req_id in app.state.storage_pd_active


def _clear_pd_request_state(req_id: str) -> None:
    app.state.storage_pd_active.pop(req_id, None)
    app.state.storage_pd_failures.pop(req_id, None)
    app.state.storage_pd_statuses.pop(req_id, None)
    app.state.finished_reqs.pop(req_id, None)


def _handoff_deadline() -> float:
    """One absolute deadline for a whole handoff, fixed when it starts.

    Every wait in the handoff is measured against this instant rather than
    given its own fresh allowance, so no stage can extend the budget by
    starting its clock late or by retrying.
    """
    return time.monotonic() + float(global_args.storage_pd_ready_timeout_s)


async def prefill_within_handoff_budget(
    client: httpx.AsyncClient,
    req_data: dict,
    req_id: str,
    deadline: float,
) -> httpx.Response:
    """Post to the prefiller, bounded by the handoff's absolute deadline.

    The prefill clients carry no timeout of their own, so a prefiller that
    stops answering would hold the request open for as long as it likes,
    and the wait for READY that follows only starts its clock once this
    returns. Both stages share the one deadline.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _clear_pd_request_state(req_id)
        raise TimeoutError(
            f"storage P/D request {req_id} exhausted its handoff budget "
            "before the prefiller was contacted"
        )
    try:
        return await asyncio.wait_for(
            send_request_to_service(client, "/v1/completions", req_data),
            timeout=remaining,
        )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        _clear_pd_request_state(req_id)
        raise TimeoutError(
            f"storage P/D request {req_id} timed out after {remaining:g}s "
            f"waiting for the prefiller to answer"
        ) from exc


async def wait_decode_kv_ready(
    req_id: str,
    num_tp_rank: int,
    *,
    storage_pd: bool,
    deadline: float,
):
    """Wait for every expected rank to report READY, or for the deadline.

    The deadline is absolute and shared with the prefill call, so a barrier
    that is handed an already exhausted budget fails rather than succeeding
    on statuses that arrived too late to be useful.
    """
    expected_ranks = set(range(num_tp_rank))
    while True:
        if storage_pd and time.monotonic() >= deadline:
            missing_ranks = expected_ranks - set(
                app.state.storage_pd_statuses.get(req_id, {})
            )
            _clear_pd_request_state(req_id)
            raise TimeoutError(
                f"storage P/D request {req_id} timed out waiting for TP ranks "
                f"{sorted(missing_ranks)}"
            )
        failure = app.state.storage_pd_failures.get(req_id)
        if failure is not None:
            _clear_pd_request_state(req_id)
            raise RuntimeError(
                f"storage P/D request {req_id} failed on TP rank "
                f"{failure.tp_rank} at {failure.error_stage}: {failure.error_text}"
            )
        statuses = app.state.storage_pd_statuses.get(req_id, {})
        unexpected_ranks = set(statuses) - expected_ranks
        if unexpected_ranks:
            _clear_pd_request_state(req_id)
            raise RuntimeError(
                f"storage P/D request {req_id} reported unexpected TP ranks "
                f"{sorted(unexpected_ranks)}"
            )
        if set(statuses) == expected_ranks:
            try:
                ordered = order_storage_pd_ready_statuses(statuses, num_tp_rank)
            finally:
                _clear_pd_request_state(req_id)
            logger.debug("Storage P/D signaled kv ready for req %s", req_id)
            return ordered
        if not storage_pd and app.state.finished_reqs[req_id] >= num_tp_rank:
            _clear_pd_request_state(req_id)
            logger.debug("Prefill node signaled kv ready for req %s", req_id)
            return []
        await asyncio.sleep(0.001)


BOUND_CLIENTS_MAX_NUM = 1024 * 1024


def pick_up_bound_clients(client_id: str) -> tuple[ClientInfo, ClientInfo, ClientInfo]:
    if client_id not in app.state.bound_clients:
        if len(app.state.bound_clients) >= BOUND_CLIENTS_MAX_NUM:
            # Here simply clear the bound_clients if full
            app.state.bound_clients.clear()
        app.state.bound_clients[client_id] = round_robin_pick_clients()
    return app.state.bound_clients[client_id]


BOUND_CLIENT = os.getenv("CLIENT_BOUND", "false").lower() == "true"
# CLIENT_BOUND_KEY, the field name of the client uid in http request
CLIENT_BOUND_KEY = os.getenv("CLIENT_BOUND_KEY", "session-id")


def pick_up_clients(request: Request) -> tuple[ClientInfo, ClientInfo, ClientInfo]:
    bound_client_id = request.headers.get(CLIENT_BOUND_KEY) if BOUND_CLIENT else None
    if bound_client_id:
        # Use or create a persistent set of clients for the session.
        return pick_up_bound_clients(bound_client_id)
    return round_robin_pick_clients()


@app.post("/v1/completions")
async def handle_completions(request: Request):
    global counter, stats_calculator
    counter += 1
    req_id = uuid.uuid4().hex if global_args.storage_pd else str(counter)

    st = time.time()
    slots = 0  # slots to release on error; set after successful acquire only
    acquired = False
    try:
        req_data = await request.json()

        # Pick tokenization, prefill and decode client
        tokenization_client, prefill_client, decode_client = pick_up_clients(request)

        tokenize_output = await send_request_to_service(
            tokenization_client.client, "/tokenize", {"prompt": req_data["prompt"]}
        )
        tokenize_output = tokenize_output.json()

        org_max_tokens = take_prefill_budget(req_data)
        req_data["prompt"] = tokenize_output["tokens"]

        # Acquire ceil(L/chunk_size) PD buffer slots before prefill.
        slots = math.ceil(len(tokenize_output["tokens"]) / global_args.chunk_size)
        if pd_buffer_semaphore is not None:
            await pd_buffer_semaphore.acquire(slots)
            acquired = True

        disagg_spec = {
            "req_id": req_id,
            "receiver_host": decode_client.host,
            "receiver_init_port": decode_client.init_port,
            "receiver_alloc_port": decode_client.alloc_port,
        }
        num_tp_rank = len(decode_client.init_port or [])

        req_data["kv_transfer_params"] = {
            "ret_first_tok": True,
            "disagg_spec": disagg_spec,
        }

        req_data["stream"] = False
        stream_options = req_data.pop("stream_options", None)

        # Fix the whole handoff's deadline before anything waits, and admit
        # this request's statuses from here: a producer can report READY
        # before its HTTP response comes back.
        handoff_deadline = _handoff_deadline()
        if global_args.storage_pd:
            _register_pd_request(req_id)
            prefill_response = await prefill_within_handoff_budget(
                prefill_client.client, req_data, req_id, handoff_deadline
            )
        else:
            prefill_response = await send_request_to_service(
                prefill_client.client, "/v1/completions", req_data
            )
        prefill_output = prefill_response

        prefill_output = prefill_output.json()

        et = time.time()
        stats_calculator.add(et - st)

        producer_only = producer_answer_is_complete(org_max_tokens, prefill_output)
        first_tok_id = adopt_prefill_first_token(req_data, prefill_output)
        if not producer_only:
            req_data["max_tokens"] = org_max_tokens - 1
        req_data.pop("kv_transfer_params")
        req_data["stream"] = True
        if stream_options is not None:
            req_data["stream_options"] = stream_options

        try:
            statuses = await wait_decode_kv_ready(
                req_id,
                num_tp_rank,
                storage_pd=global_args.storage_pd,
                deadline=handoff_deadline,
            )
            if statuses and producer_only:
                # No decoder is going to read these, so nothing would ever
                # acknowledge them. Say so now, while the statuses that
                # identify the publications are still in hand.
                await tell_producers_nobody_will_read(
                    statuses,
                    reason=(
                        "the producer's single token is the whole answer, so "
                        "no decoder was assigned this publication"
                    ),
                )
            elif statuses:
                req_data["kv_transfer_params"] = {
                    "lmcache.storage_pd_request_id": req_id,
                    "lmcache.storage_pd_statuses": [
                        msgspec.to_builtins(status) for status in statuses
                    ],
                }
        finally:
            # The barrier clears its own state on every path it returns
            # through, but a cancelled or failed request never reaches it.
            if global_args.storage_pd:
                _clear_pd_request_state(req_id)
            if pd_buffer_semaphore is not None:
                acquired = False
                await pd_buffer_semaphore.release(slots)

        # Stream response from decode service
        async def generate_stream():
            yield (
                "data: "
                + json.dumps(
                    producer_head_chunk(
                        prefill_output, first_tok_id, final=producer_only
                    ),
                    separators=(",", ":"),
                )
                + "\n\n"
            ).encode()

            if producer_only:
                # The answer is one token: either that is all the caller
                # asked for, or the producer stopped. The decoder has nothing
                # to generate, and asking it for zero tokens is a request the
                # engine refuses. The publication stands and no reader will
                # claim it, which is the writer's to resolve -- nothing here
                # fabricates a restore acknowledgement for a read that did
                # not happen.
                yield b"data: [DONE]\n\n"
                return

            async for chunk in stream_service_response(
                decode_client.client, "/v1/completions", req_data
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        # Standard
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server - completions endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise
    finally:
        # One ownership scope for the whole request: registration, prefill,
        # parsing and the barrier. The barrier clears its own state and
        # releases the permit on every path it returns through, but a request
        # that fails or is cancelled before reaching it never got there, and
        # cancellation is not an Exception so the handler above never saw it.
        # Both actions are idempotent, so this runs exactly once either way.
        if global_args.storage_pd:
            _clear_pd_request_state(req_id)
        if pd_buffer_semaphore is not None and acquired:
            acquired = False
            await pd_buffer_semaphore.release(slots)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    global counter, stats_calculator
    counter += 1
    req_id = uuid.uuid4().hex if global_args.storage_pd else str(counter)

    st = time.time()
    slots = 0  # slots to release on error; set after successful acquire only
    acquired = False
    try:
        req_data = await request.json()

        # Pick tokenization, prefill and decode client
        tokenization_client, prefill_client, decode_client = pick_up_clients(request)

        # For chat completions, we need to tokenize the messages
        tokenize_output = await send_request_to_service(
            tokenization_client.client, "/tokenize", {"messages": req_data["messages"]}
        )
        tokenize_output = tokenize_output.json()

        org_max_tokens = take_prefill_budget(req_data)
        req_data["prompt"] = tokenize_output["tokens"]

        org_max_completion_tokens = None
        if "max_completion_tokens" in req_data:
            org_max_completion_tokens = req_data["max_completion_tokens"]
            req_data["max_completion_tokens"] = 1

        # Acquire ceil(L/chunk_size) PD buffer slots before prefill.
        slots = math.ceil(len(tokenize_output["tokens"]) / global_args.chunk_size)
        if pd_buffer_semaphore is not None:
            await pd_buffer_semaphore.acquire(slots)
            acquired = True

        disagg_spec = {
            "req_id": req_id,
            "receiver_host": decode_client.host,
            "receiver_init_port": decode_client.init_port,
            "receiver_alloc_port": decode_client.alloc_port,
        }

        num_tp_rank = len(decode_client.init_port or [])

        req_data["kv_transfer_params"] = {
            "ret_first_tok": True,
            "disagg_spec": disagg_spec,
        }

        req_data["stream"] = False
        stream_options = req_data.pop("stream_options", None)

        # Fix the whole handoff's deadline before anything waits, and admit
        # this request's statuses from here: a producer can report READY
        # before its HTTP response comes back.
        handoff_deadline = _handoff_deadline()
        if global_args.storage_pd:
            _register_pd_request(req_id)
            prefill_response = await prefill_within_handoff_budget(
                prefill_client.client, req_data, req_id, handoff_deadline
            )
        else:
            prefill_response = await send_request_to_service(
                prefill_client.client, "/v1/completions", req_data
            )
        prefill_output = prefill_response

        prefill_output = prefill_output.json()

        et = time.time()
        stats_calculator.add(et - st)

        producer_only = producer_answer_is_complete(org_max_tokens, prefill_output)
        if not producer_only:
            req_data["max_tokens"] = org_max_tokens - 1
            if org_max_completion_tokens is not None:
                req_data["max_completion_tokens"] = org_max_completion_tokens - 1

        # Add the first token from prefill to the tokenized messages for decode
        first_tok_id = adopt_prefill_first_token(req_data, prefill_output)

        req_data.pop("kv_transfer_params")
        req_data["stream"] = True
        if stream_options is not None:
            req_data["stream_options"] = stream_options

        try:
            statuses = await wait_decode_kv_ready(
                req_id,
                num_tp_rank,
                storage_pd=global_args.storage_pd,
                deadline=handoff_deadline,
            )
            if statuses and producer_only:
                # No decoder is going to read these, so nothing would ever
                # acknowledge them. Say so now, while the statuses that
                # identify the publications are still in hand.
                await tell_producers_nobody_will_read(
                    statuses,
                    reason=(
                        "the producer's single token is the whole answer, so "
                        "no decoder was assigned this publication"
                    ),
                )
            elif statuses:
                req_data["kv_transfer_params"] = {
                    "lmcache.storage_pd_request_id": req_id,
                    "lmcache.storage_pd_statuses": [
                        msgspec.to_builtins(status) for status in statuses
                    ],
                }
        finally:
            # The barrier clears its own state on every path it returns
            # through, but a cancelled or failed request never reaches it.
            if global_args.storage_pd:
                _clear_pd_request_state(req_id)
            if pd_buffer_semaphore is not None:
                acquired = False
                await pd_buffer_semaphore.release(slots)

        producer_choice = (prefill_output.get("choices") or [{}])[0]

        # Stream response from decode service
        async def generate_stream():
            initial_chunk = {
                "id": prefill_output["id"],
                "object": "chat.completion.chunk",
                "created": prefill_output["created"],
                "model": prefill_output["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield (
                "data: " + json.dumps(initial_chunk, separators=(",", ":")) + "\n\n"
            ).encode()

            head_chunk = {
                "id": prefill_output["id"],
                "object": "chat.completion.chunk",
                "created": prefill_output["created"],
                "model": prefill_output["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": producer_choice.get("text", "")},
                        # The prefiller already decided this token and
                        # reported its integer id. Carry the id and the
                        # probability record, not only the text they render
                        # to: a client comparing generated tokens cannot
                        # recover an id from text without retokenizing, and a
                        # probability list missing this token cannot be
                        # compared against one that includes it.
                        "token_ids": [first_tok_id],
                        "logprobs": producer_choice.get("logprobs"),
                        "finish_reason": (
                            producer_choice.get("finish_reason")
                            if producer_only
                            else None
                        ),
                    }
                ],
            }
            yield (
                "data: " + json.dumps(head_chunk, separators=(",", ":")) + "\n\n"
            ).encode()

            if producer_only:
                # One token is the whole answer, so there is nothing to
                # decode and a zero-token request would be refused. The
                # publication stands unread, which is the writer's to
                # resolve; nothing here fabricates a restore acknowledgement.
                yield b"data: [DONE]\n\n"
                return

            # Stream and convert completion format chunks to chat completion format
            async for chunk in stream_service_response(
                decode_client.client, "/v1/completions", req_data
            ):
                chunk_str = chunk.decode("utf-8")
                if chunk_str.startswith("data: ") and not chunk_str.startswith(
                    "data: [DONE]"
                ):
                    try:
                        json_str = chunk_str[6:].strip()  # Remove 'data: ' prefix
                        if json_str:
                            completion_data = json.loads(json_str)
                            # Decoder can emit non-token chunks (usage, final
                            # metadata, keepalives) with an empty choices list.
                            # Those aren't chat-completion deltas, so pass the
                            # original chunk through unchanged instead of
                            # indexing into an empty list.
                            if not completion_data.get("choices"):
                                yield chunk
                                continue
                            chat_completion_data = {
                                "id": completion_data["id"],
                                "object": "chat.completion.chunk",
                                "created": completion_data["created"],
                                "model": completion_data["model"],
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": completion_data["choices"][0][
                                                "text"
                                            ]
                                        },
                                        "logprobs": completion_data["choices"][0].get(
                                            "logprobs"
                                        ),
                                        # Carried, not dropped. Without this
                                        # the chat lane reports exactly one
                                        # token id -- the synthetic first
                                        # one -- however long the answer is,
                                        # which reads as id-capable while
                                        # comparing nothing.
                                        "token_ids": completion_data["choices"][0].get(
                                            "token_ids"
                                        ),
                                        "finish_reason": completion_data["choices"][
                                            0
                                        ].get("finish_reason"),
                                    }
                                ],
                            }
                            converted_chunk = (
                                "data: "
                                + json.dumps(
                                    chat_completion_data, separators=(",", ":")
                                )
                                + "\n\n"
                            ).encode()
                            yield converted_chunk
                    except (json.JSONDecodeError, KeyError):
                        yield chunk
                else:
                    yield chunk

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        # Standard
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(
            "Error occurred in disagg prefill proxy server  - chat completions endpoint"
        )
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise
    finally:
        # One ownership scope for the whole request: registration, prefill,
        # parsing and the barrier. The barrier clears its own state and
        # releases the permit on every path it returns through, but a request
        # that fails or is cancelled before reaching it never got there, and
        # cancellation is not an Exception so the handler above never saw it.
        # Both actions are idempotent, so this runs exactly once either way.
        if global_args.storage_pd:
            _clear_pd_request_state(req_id)
        if pd_buffer_semaphore is not None and acquired:
            acquired = False
            await pd_buffer_semaphore.release(slots)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    # Third Party
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
