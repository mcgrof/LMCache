# SPDX-License-Identifier: Apache-2.0
"""The control exchange that releases a writer's holds.

A writer keeps every extent it publishes locked until the consumer it told
about them says the bytes reached GPU memory. Nothing else can decide that,
so this exchange is the only thing that ever frees one -- which makes two
properties load-bearing.

The first is that a send is not an answer. A socket accepting a message
says the local queue took it, not that a writer validated it, and certainly
not that a hold was released; a consumer that retires its obligation on a
successful send leaves the writer holding extents nobody will ever reclaim.
So the consumer asks and waits for the writer's own reply, and retires only
on that.

The second is that a lost reply must be safe. The writer may have released
the hold and then failed to answer, so the same request arrives again; it
carries its whole correlation identity, and the writer distinguishes a
retry of something it applied from a different message naming the same
request. A duplicate frees nothing twice and a mismatch frees nothing at
all.

The transport is deliberately small: one bounded request/reply per attempt
on the endpoint the writer already publishes in its READY status. There is
no relay, because a relay's word about an acknowledgement is not the
reader's word, and the writer is the only party that can act on it anyway.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import Optional
import threading
import time
import uuid

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.storage_pd_protocol import StoragePDReadAck

logger = init_logger(__name__)

# What a writer can answer. These are the tracker's outcomes carried on the
# wire; the consumer acts on them and so they are part of the contract.
ACK_APPLIED = "APPLIED"
ACK_ALREADY_APPLIED = "ALREADY_APPLIED"
ACK_REJECTED = "REJECTED"
ACK_UNRESOLVED = "UNRESOLVED"

# A consumer stops offering a request the writer has answered definitively.
# Unresolved is not definitive: it says the writer could not tell, so the
# hold may still stand and the obligation is still owed.
ACK_TERMINAL_OUTCOMES = frozenset({ACK_APPLIED, ACK_ALREADY_APPLIED, ACK_REJECTED})


class StoragePDAckRequest(msgspec.Struct, tag=True):
    """One consumer asking one writer to release one hold.

    ``nonce`` is per attempt and only for correlation: a reply carrying
    another attempt's nonce is a reply to a question this attempt did not
    ask, and answering the current obligation with it would accept an
    answer about something else.
    """

    ack: StoragePDReadAck
    session_id: str
    attempt: int
    nonce: str


class StoragePDAckReply(msgspec.Struct, tag=True):
    """A writer's answer, sent only after it has acted.

    Every identity field is echoed so the consumer can check that this is
    an answer to its own question rather than a well-formed message about
    somebody else's.
    """

    outcome: str
    req_id: str
    writer_epoch: str
    consumer_instance_id: str
    nonce: str
    reason: str = ""


StoragePDAckWire = StoragePDAckRequest | StoragePDAckReply


class StoragePDAckServer:
    """Answer acknowledgement requests for one writer rank.

    One thread owns the socket for its whole life: it binds, receives,
    calls the handler, replies and closes. Nothing else touches it, because
    a reply socket that is written from two places gets out of step with
    its own request/reply alternation and then answers the wrong caller.

    ``handler`` validates the request against what this writer published
    and performs the release. Its return value is what goes back, so it
    must not report success before the release it describes has happened.
    Everything reaching it came off a network and is trusted for nothing
    but its shape.
    """

    def __init__(
        self,
        handler: Callable[[StoragePDAckRequest], tuple[str, str]],
        *,
        bind_host: str,
        port: int,
        advertise_host: str = "",
        recv_timeout_ms: int = 200,
    ) -> None:
        if not bind_host:
            raise ValueError("storage P/D acknowledgements need a bind host")
        if port <= 0:
            raise ValueError("storage P/D acknowledgements need a port")
        # A wildcard bind says where to listen, and says nothing about how
        # to be reached. A consumer given "0.0.0.0" has no address, so the
        # advertised host is separate and must be routable.
        advertised = advertise_host or bind_host
        if advertised in ("0.0.0.0", "::", "*"):
            raise ValueError(
                "storage P/D acknowledgements need an advertisable host: "
                f"{advertised!r} is a bind wildcard, not an address"
            )
        self._handler = handler
        self.endpoint = f"{advertised}:{port}"
        self.bind_endpoint = f"{bind_host}:{port}"
        context = get_zmq_context(use_asyncio=False)
        self._socket = get_zmq_socket(
            context,
            self.bind_endpoint,
            "tcp",
            zmq.REP,
            "bind",
        )
        self._socket.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._stopping = False
        self._quiesced = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="storage-pd-ack-server",
            daemon=True,
        )
        self._thread.start()

    def close(self, timeout_s: float = 5.0) -> bool:
        """Stop answering and close the socket, or report that it did not.

        Returns whether the worker confirmed it had stopped. A join that
        returns is not that confirmation: the thread may have been
        descheduled inside the handler, which reaches the core. The caller
        needs to know which happened, because destroying the core under a
        live handler is the thing this prevents.
        """
        self._stopping = True
        self._thread.join(timeout=timeout_s)
        quiesced = self._quiesced.is_set()
        if not quiesced:
            logger.error(
                "Storage P/D acknowledgement server did not confirm it "
                "stopped within %.1fs; its handler may still be running",
                timeout_s,
            )
            return False
        self._socket.close(linger=0)
        return True

    def _run(self) -> None:
        try:
            while not self._stopping:
                try:
                    raw = self._socket.recv()
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    if self._stopping:
                        return
                    logger.exception("Storage P/D acknowledgement socket failed")
                    return
                self._answer(raw)
        finally:
            # Set last, and only on the way out of the loop, so it means
            # "this thread is no longer in the handler".
            self._quiesced.set()

    def _answer(self, raw: bytes) -> None:
        """Handle one request and send exactly one reply.

        A reply socket owes the caller an answer for every message it
        accepted. Skipping one on a decode failure leaves the socket out of
        step and the caller waiting on an answer that will never come, so
        even an unintelligible request gets a rejection.
        """
        try:
            message = msgspec.msgpack.decode(raw, type=StoragePDAckWire)
        except Exception:
            logger.warning("Storage P/D acknowledgement request was undecodable")
            self._send(
                StoragePDAckReply(
                    outcome=ACK_REJECTED,
                    req_id="",
                    writer_epoch="",
                    consumer_instance_id="",
                    nonce="",
                    reason="undecodable request",
                )
            )
            return
        if not isinstance(message, StoragePDAckRequest):
            self._send(
                StoragePDAckReply(
                    outcome=ACK_REJECTED,
                    req_id="",
                    writer_epoch="",
                    consumer_instance_id="",
                    nonce="",
                    reason=f"expected a request, got a {type(message).__name__}",
                )
            )
            return
        try:
            outcome, reason = self._handler(message)
        except Exception:
            # A handler that raised has not established that it released
            # anything, so the answer is that this writer cannot say.
            logger.exception(
                "Storage P/D acknowledgement for %s was not applied",
                message.ack.req_id,
            )
            outcome, reason = ACK_UNRESOLVED, "the writer could not apply it"
        self._send(
            StoragePDAckReply(
                outcome=outcome,
                req_id=message.ack.req_id,
                writer_epoch=message.ack.writer_epoch,
                consumer_instance_id=message.ack.consumer_instance_id,
                nonce=message.nonce,
                reason=reason,
            )
        )

    def _send(self, reply: StoragePDAckReply) -> None:
        try:
            self._socket.send(msgspec.msgpack.encode(reply))
        except zmq.ZMQError:
            logger.exception(
                "Storage P/D could not answer the acknowledgement for %s",
                reply.req_id,
            )


class StoragePDAckObligation:
    """One logical acknowledgement, owed until a writer answers it.

    The obligation outlives the request that created it and every attempt
    made for it: a consumer whose request has long finished still owes the
    writer this message, and a writer that never hears it holds those
    extents for its own lifetime. It carries one absolute deadline, and
    reaching that deadline is an unresolved control outcome -- never an
    authorization to consider the hold released.
    """

    def __init__(
        self,
        ack: StoragePDReadAck,
        *,
        endpoint: str,
        session_id: str,
        deadline: float,
    ) -> None:
        self.ack = ack
        self.endpoint = endpoint
        self.session_id = session_id
        self.deadline = deadline
        self.attempts = 0
        self._settled = threading.Event()
        self._outcome: Optional[str] = None
        self._reason = ""

    @property
    def settled(self) -> bool:
        return self._settled.is_set()

    @property
    def outcome(self) -> Optional[str]:
        return self._outcome

    @property
    def reason(self) -> str:
        return self._reason

    def wait(self, timeout: Optional[float] = None) -> Optional[str]:
        """Block until this obligation settles, for a caller that can wait."""
        self._settled.wait(timeout=timeout)
        return self._outcome

    def _settle(self, outcome: str, reason: str) -> None:
        if self._settled.is_set():
            return
        self._outcome = outcome
        self._reason = reason
        self._settled.set()


class StoragePDAckClient:
    """Own the acknowledgements a consumer owes, and keep asking.

    A background thread makes progress with no further serving request,
    which is the point: an obligation created by the last restore before a
    quiet period still has to reach the writer, and the restore path is the
    only regular event a consumer has. Making retries depend on a later
    restore means a lost acknowledgement waits for traffic that may never
    come.

    Each attempt is one bounded request/reply on its own socket. On a
    timeout the socket is discarded rather than reused: a request socket
    that timed out is out of step with its own alternation, and its next
    receive could deliver the previous attempt's answer. A reply that does
    not correlate with the attempt that is waiting is discarded for the
    same reason.
    """

    def __init__(
        self,
        *,
        attempt_timeout_ms: int = 2000,
        retry_interval_s: float = 1.0,
        max_live_obligations: int = 1024,
        poll_interval_s: float = 0.05,
    ) -> None:
        if attempt_timeout_ms <= 0:
            raise ValueError("an acknowledgement attempt needs a timeout")
        if max_live_obligations <= 0:
            raise ValueError("an acknowledgement client needs a live bound")
        self._attempt_timeout_ms = attempt_timeout_ms
        self._retry_interval_s = retry_interval_s
        self._max_live = max_live_obligations
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._owed: list[StoragePDAckObligation] = []
        self._next_attempt: dict[int, float] = {}
        self._context = get_zmq_context(use_asyncio=False)
        self._stopping = False
        self._quiesced = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="storage-pd-ack-client",
            daemon=True,
        )
        self._thread.start()

    def live_count(self) -> int:
        with self._lock:
            return len(self._owed)

    def owe(
        self,
        ack: StoragePDReadAck,
        *,
        endpoint: str,
        session_id: str,
        deadline_s: float,
    ) -> Optional[StoragePDAckObligation]:
        """Take on one acknowledgement, or refuse to take on any more.

        Refusing is a real answer. The alternative is to evict an older
        obligation to make room, which silently abandons a writer's hold to
        keep admitting work -- the caller has to stop publishing instead.
        """
        if not endpoint:
            logger.error(
                "Storage P/D cannot acknowledge request %s: the producer "
                "advertised no endpoint, so its extents stay held",
                ack.req_id,
            )
            return None
        obligation = StoragePDAckObligation(
            ack,
            endpoint=endpoint,
            session_id=session_id,
            deadline=time.monotonic() + deadline_s,
        )
        with self._lock:
            if self._stopping:
                return None
            self._owed = [item for item in self._owed if not item.settled]
            if len(self._owed) >= self._max_live:
                logger.error(
                    "Storage P/D already owes %d unanswered acknowledgements; "
                    "refusing to take on request %s rather than abandoning "
                    "one of them",
                    len(self._owed),
                    ack.req_id,
                )
                return None
            self._owed.append(obligation)
        self._wake.set()
        return obligation

    def close(self, timeout_s: float = 5.0) -> bool:
        """Stop attempting and report whether the worker confirmed it had.

        Unsettled obligations are settled as unresolved, because that is
        what they are: nothing here has heard a writer say anything about
        them, and reporting otherwise at shutdown would be the one lie this
        module exists to prevent.
        """
        with self._lock:
            self._stopping = True
        self._wake.set()
        self._thread.join(timeout=timeout_s)
        quiesced = self._quiesced.is_set()
        if not quiesced:
            logger.error(
                "Storage P/D acknowledgement client did not confirm it "
                "stopped within %.1fs",
                timeout_s,
            )
        with self._lock:
            owed = list(self._owed)
        for obligation in owed:
            obligation._settle(
                ACK_UNRESOLVED, "the consumer stopped before a writer answered"
            )
        return quiesced

    def _run(self) -> None:
        try:
            while True:
                with self._lock:
                    if self._stopping:
                        return
                    due = self._due_locked()
                if not due:
                    self._wake.wait(timeout=self._poll_interval_s)
                    self._wake.clear()
                    continue
                for obligation in due:
                    with self._lock:
                        if self._stopping:
                            return
                    self._attempt(obligation)
        finally:
            self._quiesced.set()

    def _due_locked(self) -> list[StoragePDAckObligation]:
        now = time.monotonic()
        due: list[StoragePDAckObligation] = []
        keep: list[StoragePDAckObligation] = []
        for obligation in self._owed:
            if obligation.settled:
                self._next_attempt.pop(id(obligation), None)
                continue
            keep.append(obligation)
            if self._next_attempt.get(id(obligation), 0.0) <= now:
                due.append(obligation)
        self._owed = keep
        return due

    def _attempt(self, obligation: StoragePDAckObligation) -> None:
        now = time.monotonic()
        if now >= obligation.deadline:
            # The deadline is the obligation's, not an attempt's. Reaching
            # it means nobody ever confirmed, which is a control outcome to
            # report -- not a reason to treat the hold as released.
            logger.error(
                "Storage P/D never got an answer about request %s after %d "
                "attempt(s); the producer's extents stay held",
                obligation.ack.req_id,
                obligation.attempts,
            )
            obligation._settle(ACK_UNRESOLVED, "the deadline passed unanswered")
            return
        obligation.attempts += 1
        nonce = uuid.uuid4().hex
        request = StoragePDAckRequest(
            ack=obligation.ack,
            session_id=obligation.session_id,
            attempt=obligation.attempts,
            nonce=nonce,
        )
        reply = self._exchange(obligation.endpoint, request)
        if reply is None:
            self._next_attempt[id(obligation)] = now + self._retry_interval_s
            return
        if not self._correlates(reply, request):
            logger.warning(
                "Storage P/D got an acknowledgement reply for %s/%s while "
                "waiting on %s/%s; discarding it",
                reply.req_id,
                reply.nonce,
                request.ack.req_id,
                nonce,
            )
            self._next_attempt[id(obligation)] = now + self._retry_interval_s
            return
        if reply.outcome in ACK_TERMINAL_OUTCOMES:
            level = logger.info if reply.outcome != ACK_REJECTED else logger.error
            level(
                "Storage P/D acknowledgement for %s: %s%s",
                reply.req_id,
                reply.outcome,
                f" ({reply.reason})" if reply.reason else "",
            )
            obligation._settle(reply.outcome, reply.reason)
            return
        # Unresolved: the writer answered and could not say. Keep asking
        # until the obligation's own deadline.
        self._next_attempt[id(obligation)] = now + self._retry_interval_s

    def _exchange(
        self,
        endpoint: str,
        request: StoragePDAckRequest,
    ) -> Optional[StoragePDAckReply]:
        """Make one bounded attempt on a socket used for nothing else."""
        socket = None
        try:
            socket = get_zmq_socket(
                self._context,
                endpoint,
                "tcp",
                zmq.REQ,
                "connect",
            )
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, self._attempt_timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, self._attempt_timeout_ms)
            socket.send(msgspec.msgpack.encode(request))
            raw = socket.recv()
        except zmq.Again:
            logger.warning(
                "Storage P/D acknowledgement attempt %d for %s timed out against %s",
                request.attempt,
                request.ack.req_id,
                endpoint,
            )
            return None
        except zmq.ZMQError:
            logger.warning(
                "Storage P/D acknowledgement attempt %d for %s could not reach %s",
                request.attempt,
                request.ack.req_id,
                endpoint,
            )
            return None
        finally:
            # One socket per attempt, closed with it. A request socket that
            # timed out is out of step with its own alternation, so it
            # cannot be reused, and closing it means a late reply has
            # nowhere to arrive rather than somewhere to be mistaken for
            # the next answer.
            if socket is not None:
                socket.close(linger=0)
        try:
            message = msgspec.msgpack.decode(raw, type=StoragePDAckWire)
        except Exception:
            logger.warning("Storage P/D acknowledgement reply was undecodable")
            return None
        if not isinstance(message, StoragePDAckReply):
            logger.warning(
                "Storage P/D acknowledgement got a %s in reply",
                type(message).__name__,
            )
            return None
        return message

    @staticmethod
    def _correlates(
        reply: StoragePDAckReply,
        request: StoragePDAckRequest,
    ) -> bool:
        """Whether this reply answers this attempt, and not something else."""
        return (
            reply.nonce == request.nonce
            and reply.req_id == request.ack.req_id
            and reply.writer_epoch == request.ack.writer_epoch
            and reply.consumer_instance_id == request.ack.consumer_instance_id
        )
