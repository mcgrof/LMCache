# SPDX-License-Identifier: Apache-2.0
"""Bind transport-derived logical client identity during request execution."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib

_session: ContextVar[str | None] = ContextVar("lmcache_request_session", default=None)


def get_request_session() -> str:
    """Return the actual handler's logical client-session digest.

    Raises:
        RuntimeError: Outside a transport-bound handler, or if the request has
            no unambiguous identity. Ownership-sensitive handlers must refuse
            this case before accepting work; a peer address is not a fallback.
    """
    session = _session.get()
    if session is None:
        raise RuntimeError("request has no bound client session")
    return session


@contextmanager
def request_session(transport: str, identity: bytes | None) -> Iterator[None]:
    """Scope a request's transport identity, restoring prior context on exit.

    Args:
        transport: Name of the receiving request transport.
        identity: Routing identity or unambiguous per-client binary metadata,
            supplied by the transport rather than an operation's payload.
            None leaves legacy requests without an ownership identity.

    Raises:
        ValueError: Unsupported transport or malformed non-None identity.

    This identifies cooperative client instances, not authenticated peers or
    individual TCP connections. Executors must run a copy of this context.
    """
    if transport not in ("zmq", "grpc"):
        raise ValueError("unsupported request transport")
    if identity is not None and (
        type(identity) is not bytes or not 0 < len(identity) <= 1024
    ):
        raise ValueError("invalid request-client identity")
    session = (
        hashlib.sha256(transport.encode() + b"\0" + identity).hexdigest()
        if identity is not None
        else None
    )
    token = _session.set(session)
    try:
        yield
    finally:
        _session.reset(token)
