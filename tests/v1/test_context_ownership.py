# SPDX-License-Identifier: Apache-2.0
"""Exercise registration holds and retirement through public CPU APIs."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.context_ownership import (
    ContextOwnership,
    ContextOwnershipError,
)
from lmcache.v1.multiprocess.transport.request_context import (
    get_request_session,
    request_session,
)


def test_ownerless_context_remains_unborrowable() -> None:
    ownership = ContextOwnership()
    assert ownership.identity.owner_session is None
    with pytest.raises(ContextOwnershipError):
        ownership.borrow(ownership.identity, "unrelated", lambda: None)
    ownership.retire(timeout=0)


def test_owner_and_registration_generation_are_both_required() -> None:
    with request_session("zmq", b"client"):
        ownership = ContextOwnership()
        session = get_request_session()
    with pytest.raises(ContextOwnershipError):
        ownership.borrow(ownership.identity, "other-client", lambda: None)
    with pytest.raises(ContextOwnershipError):
        ownership.borrow(
            replace(ownership.identity, generation="stale"), session, lambda: None
        )
    hold = ownership.borrow(ownership.identity, session, lambda: None)
    assert ownership.pending == 1
    hold.release()
    hold.release()
    assert ownership.pending == 0
    ownership.retire(timeout=0)
    with pytest.raises(ContextOwnershipError):
        ownership.borrow(ownership.identity, session, lambda: None)
    with request_session("zmq", b"client"):
        replacement = ContextOwnership()
    assert replacement.identity.generation != ownership.identity.generation
    assert replacement.identity.owner_session == ownership.identity.owner_session


def test_timeout_never_discards_completion_debt_or_reopens_admission() -> None:
    with request_session("grpc", b"client"):
        ownership = ContextOwnership()
        session = get_request_session()
    revoked: list[str] = []
    hold = ownership.borrow(
        ownership.identity, session, lambda: revoked.append("revoke")
    )
    for _ in range(2):
        with pytest.raises(ContextOwnershipError, match="undrained"):
            ownership.retire(timeout=0)
        assert ownership.pending == 1
    assert revoked == ["revoke"]
    with pytest.raises(ContextOwnershipError):
        ownership.borrow(ownership.identity, session, lambda: None)
    hold.release()
    ownership.retire(timeout=0)
    assert ownership.pending == 0


def test_retirement_waits_for_every_physical_hold() -> None:
    with request_session("zmq", b"client"):
        ownership = ContextOwnership()
        session = get_request_session()
    revoked = threading.Event()
    first = ownership.borrow(ownership.identity, session, revoked.set)
    second = ownership.borrow(ownership.identity, session, lambda: None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        retirement = pool.submit(ownership.retire, 5)
        assert revoked.wait(timeout=5)
        first.release()
        assert ownership.pending == 1
        assert not retirement.done()
        second.release()
        retirement.result(timeout=5)
    assert ownership.pending == 0


def test_callback_failure_is_uncertain_even_after_physical_drain() -> None:
    with request_session("zmq", b"client"):
        ownership = ContextOwnership()
        session = get_request_session()

    def fail() -> None:
        raise OSError("revocation failed")

    hold = ownership.borrow(ownership.identity, session, fail)
    with pytest.raises(ContextOwnershipError, match="revocation"):
        ownership.retire(timeout=0)
    hold.release()
    assert ownership.pending == 0
    with pytest.raises(ContextOwnershipError, match="revocation"):
        ownership.retire(timeout=0)


def test_concurrent_retirement_cannot_skip_incomplete_revocation() -> None:
    with request_session("zmq", b"client"):
        ownership = ContextOwnership()
        session = get_request_session()
    entered = threading.Event()
    release_callback = threading.Event()

    def revoke() -> None:
        entered.set()
        if not release_callback.wait(timeout=5):
            raise RuntimeError("revocation was not released")

    hold = ownership.borrow(ownership.identity, session, revoke)
    with ThreadPoolExecutor(max_workers=1) as pool:
        retirement = pool.submit(ownership.retire, 5)
        try:
            assert entered.wait(timeout=5)
            hold.release()
            assert ownership.pending == 0
            with pytest.raises(ContextOwnershipError):
                ownership.retire(timeout=0)
        finally:
            release_callback.set()
        retirement.result(timeout=5)
