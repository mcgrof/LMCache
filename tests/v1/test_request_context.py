# SPDX-License-Identifier: Apache-2.0
"""CPU context isolation independent of optional transport/native packages."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.transport.request_context import (
    get_request_session,
    request_session,
)


def test_nested_contexts_restore_on_return_and_exception() -> None:
    with pytest.raises(RuntimeError, match="no bound"):
        get_request_session()
    with request_session("zmq", b"first"):
        first = get_request_session()
        with pytest.raises(OSError), request_session("grpc", b"second"):
            assert get_request_session() != first
            raise OSError("handler failed")
        assert get_request_session() == first
        with request_session("grpc", None):
            with pytest.raises(RuntimeError, match="no bound"):
                get_request_session()
        assert get_request_session() == first
    with pytest.raises(RuntimeError, match="no bound"):
        get_request_session()


def test_same_client_is_stable_but_transport_and_clients_are_distinct() -> None:
    sessions: list[str] = []
    for transport, identity in (("zmq", b"a"), ("grpc", b"a"), ("zmq", b"b")):
        with request_session(transport, identity):
            sessions.append(get_request_session())
    assert len(set(sessions)) == 3
    with request_session("zmq", b"a"):
        assert get_request_session() == sessions[0]


def test_executor_context_is_explicit_and_never_leaks_to_the_next_job() -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        with request_session("zmq", b"client"):
            expected = get_request_session()
            actual = pool.submit(copy_context().run, get_request_session).result()
            assert actual == expected
        with pytest.raises(RuntimeError, match="no bound"):
            pool.submit(get_request_session).result()


@pytest.mark.parametrize(
    "transport,identity",
    [
        ("unknown", b"a"),
        ("zmq", b""),
        ("grpc", b"x" * 1025),
    ],
)
def test_invalid_transport_context_refuses(transport: str, identity: bytes) -> None:
    with pytest.raises(ValueError), request_session(transport, identity):
        pytest.fail("invalid request context was bound")
