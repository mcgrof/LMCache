# SPDX-License-Identifier: Apache-2.0
"""Public ownership contracts; no CUDA, IPC tensors, or storage stand-ins."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.kmesh_domains import (
    Completion,
    DomainRegistry,
    KMeshFenceError,
    RegistrationState,
    SemanticIdentity,
)

PROFILE = "9" * 64


def identity(**changes: object) -> SemanticIdentity:
    """Create a complete synthetic semantic identity for CPU contract tests."""
    return replace(
        SemanticIdentity(
            weights_digest="1" * 64,
            model_config_digest="2" * 64,
            tokenizer_digest="3" * 64,
            rope_digest="4" * 64,
            layout_digest="5" * 64,
            kv_dtype="bfloat16",
            chunk_tokens=256,
            hash_algorithm="sha256",
        ),
        **changes,  # type: ignore[arg-type]  # Deliberately malformed test inputs.
    )


class Clock:
    """Deterministic monotonic clock, with no wall-clock sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_open_is_atomic_and_semantics_are_separate() -> None:
    registry = DomainRegistry(["store", "retrieve"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        domains = list(pool.map(lambda _: registry.open_domain(identity()), range(32)))
    assert len(set(domains)) == 1
    a = domains[0]
    b = registry.open_domain(identity(weights_digest="a" * 64))
    assert a.domain_id != b.domain_id
    assert a.namespace != b.namespace
    assert a == registry.open_domain(identity())


@pytest.mark.parametrize(
    "change",
    [
        {"schema": 2},
        {"schema": True},
        {"adapter_identity": "lora"},
        {"key_schema": "unknown"},
        {"kv_dtype": "float8"},
        {"chunk_tokens": True},
        {"chunk_tokens": 0},
        {"layout_digest": "not-a-digest"},
        {"rope_digest": "A" * 64},
    ],
)
def test_unreviewed_semantics_refuse(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        identity(**change)


def test_barrier_requires_each_rank_and_actual_session() -> None:
    registry = DomainRegistry(["store", "retrieve"])
    domain = registry.open_domain(identity())
    lease = registry.open_registration(domain, "engine", 2, PROFILE)
    rank0 = registry.bind_worker(lease, 0, 10, "session-0", identity(), PROFILE)
    assert registry.state(lease) == RegistrationState.REGISTERING
    with pytest.raises(KMeshFenceError, match="incomplete"):
        registry.acquire(rank0, "session-0", "request", 1)
    registry.bind_worker(lease, 1, 11, "session-1", identity(), PROFILE)
    assert registry.state(lease) == RegistrationState.READY
    with pytest.raises(KMeshFenceError, match="session"):
        registry.acquire(rank0, "reconnected-session", "request", 1)
    ticket = registry.acquire(rank0, "session-0", "request", 1)
    namespaces: list[str] = []
    registry.publish(ticket, "session-0", namespaces.append)
    assert namespaces == [domain.namespace]
    with pytest.raises(KMeshFenceError, match="already published"):
        registry.publish(ticket, "session-0", namespaces.append)
    assert namespaces == [domain.namespace]
    registry.finish(ticket, Completion.SUCCEEDED)


@pytest.mark.parametrize("failure", ["duplicate", "semantics", "profile", "rank"])
def test_inconsistent_rank_revokes_entire_barrier(failure: str) -> None:
    registry = DomainRegistry(["store"])
    lease = registry.open_registration(
        registry.open_domain(identity()), "engine", 2, PROFILE
    )
    first = registry.bind_worker(lease, 0, 10, "s0", identity(), PROFILE)
    rank = 0 if failure == "duplicate" else 3 if failure == "rank" else 1
    semantics = identity(rope_digest="a" * 64) if failure == "semantics" else identity()
    execution = "a" * 64 if failure == "profile" else PROFILE
    with pytest.raises(KMeshFenceError):
        registry.bind_worker(lease, rank, 11, "s1", semantics, execution)
    assert registry.state(lease) == RegistrationState.REVOKED
    with pytest.raises(KMeshFenceError):
        registry.acquire(first, "s0", "request", 1)


def test_abort_never_rolls_generation_five_back_to_two() -> None:
    registry = DomainRegistry(["store"])
    lease = registry.open_registration(
        registry.open_domain(identity()), "engine", 1, PROFILE
    )
    fence = registry.bind_worker(lease, 0, 10, "session", identity(), PROFILE)
    ticket = registry.acquire(fence, "session", "request", 5)
    registry.finish(ticket, Completion.CANCELLED_DRAINED)
    for stale in (2, 5):
        with pytest.raises(KMeshFenceError, match="generation"):
            registry.acquire(fence, "session", "request", stale)
    next_ticket = registry.acquire(fence, "session", "request", 6)
    with pytest.raises(KMeshFenceError, match="accepted work"):
        registry.acquire(fence, "session", "request", 7)
    registry.finish(next_ticket, Completion.SUCCEEDED)
    registry.finish(
        registry.acquire(fence, "session", "request", 7), Completion.SUCCEEDED
    )


def test_expiration_is_not_recoverable_by_a_late_heartbeat() -> None:
    clock = Clock()
    registry = DomainRegistry(["store"], lease_seconds=10, clock=clock)
    domain = registry.open_domain(identity())
    lease = registry.open_registration(domain, "engine", 2, PROFILE)
    rank0 = registry.bind_worker(lease, 0, 10, "s0", identity(), PROFILE)
    registry.bind_worker(lease, 1, 11, "s1", identity(), PROFILE)
    ticket = registry.acquire(rank0, "s0", "request", 1)
    clock.now = 9
    registry.heartbeat(rank0, "s0")
    clock.now = 10  # The other rank missed its heartbeat.
    with pytest.raises(KMeshFenceError):
        registry.heartbeat(rank0, "s0")
    assert registry.state(lease) == RegistrationState.REVOKED
    assert registry.pending(lease) == 1
    registry.finish(ticket, Completion.CANCELLED_DRAINED)
    with pytest.raises(KMeshFenceError):
        registry.acquire(rank0, "s0", "request", 2)


def test_crash_before_ready_expires_registration_grace() -> None:
    clock = Clock()
    registry = DomainRegistry(["store"], lease_seconds=10, clock=clock)
    lease = registry.open_registration(
        registry.open_domain(identity()), "engine", 2, PROFILE
    )
    rank0 = registry.bind_worker(lease, 0, 10, "s0", identity(), PROFILE)
    clock.now = 9
    registry.heartbeat(rank0, "s0")
    clock.now = 10
    with pytest.raises(KMeshFenceError):
        registry.bind_worker(lease, 1, 11, "s1", identity(), PROFILE)
    assert registry.state(lease) == RegistrationState.REVOKED


def test_undrained_failure_survives_end_and_prevents_instance_replacement() -> None:
    registry = DomainRegistry(["store"])
    domain = registry.open_domain(identity())
    lease = registry.open_registration(domain, "engine-a", 1, PROFILE)
    fence = registry.bind_worker(lease, 0, 10, "session", identity(), PROFILE)
    ticket = registry.acquire(fence, "session", "request", 1)
    registry.finish(ticket, Completion.FAILED_UNDRAINED)
    registry.revoke(lease)  # END/revoke cannot erase the failed future.
    assert registry.pending(lease) == 1
    with pytest.raises(KMeshFenceError, match="drain"):
        registry.open_registration(domain, "engine-a", 1, PROFILE)
    replacement = registry.open_registration(domain, "engine-b", 1, PROFILE)
    with pytest.raises(KMeshFenceError, match="drain"):
        registry.bind_worker(replacement, 0, 10, "new-session", identity(), PROFILE)
    with pytest.raises(KMeshFenceError):
        registry.publish(ticket, "session", lambda _: pytest.fail("stale write"))
    registry.finish(ticket, Completion.CANCELLED_DRAINED)
    assert registry.pending(lease) == 0
    replacement = registry.open_registration(domain, "engine-b", 1, PROFILE)
    registry.bind_worker(replacement, 0, 10, "new-session", identity(), PROFILE)
    assert registry.state(replacement) == RegistrationState.READY
    with pytest.raises(KMeshFenceError, match="completed"):
        registry.finish(ticket, Completion.SUCCEEDED)


@pytest.mark.parametrize("invalidate", ["domain", "store", "capabilities"])
def test_drift_fences_writers_but_retains_physical_completion(invalidate: str) -> None:
    registry = DomainRegistry(["store", "retrieve"])
    domain = registry.open_domain(identity())
    lease = registry.open_registration(domain, "engine", 1, PROFILE)
    fence = registry.bind_worker(lease, 0, 10, "session", identity(), PROFILE)
    ticket = registry.acquire(fence, "session", "request", 1)
    if invalidate == "domain":
        registry.invalidate_domain(domain)
    elif invalidate == "store":
        registry.invalidate_store()
    else:
        registry.update_capabilities(["retrieve"])
    with pytest.raises(KMeshFenceError):
        registry.publish(ticket, "session", lambda _: pytest.fail("stale write"))
    assert registry.pending(lease) == 1
    registry.finish(ticket, Completion.SUCCEEDED)
    assert registry.pending(lease) == 0
    current = registry.open_domain(identity())
    if invalidate != "capabilities":
        assert current.namespace != domain.namespace
        with pytest.raises(KMeshFenceError, match="stale domain"):
            registry.open_registration(domain, "stale", 1, PROFILE)


def test_surviving_objects_never_cross_incompatible_or_rebooted_domains() -> None:
    registry = DomainRegistry(["store"])
    domain = registry.open_domain(identity())
    lease = registry.open_registration(domain, "engine-a", 1, PROFILE)
    fence = registry.bind_worker(lease, 0, 10, "session-a", identity(), PROFILE)
    ticket = registry.acquire(fence, "session-a", "request", 1)
    objects: dict[str, str] = {}
    registry.publish(ticket, "session-a", lambda ns: objects.update({ns: "old KV"}))
    registry.finish(ticket, Completion.SUCCEEDED)
    registry.revoke(lease)
    # A subsequent compatible engine may intentionally share the objects.
    assert registry.open_domain(identity()).namespace in objects
    assert (
        registry.open_domain(identity(weights_digest="b" * 64)).namespace not in objects
    )
    rebooted = DomainRegistry(["store"])
    assert rebooted.open_domain(identity()).namespace not in objects
    with pytest.raises(KMeshFenceError):
        rebooted.open_registration(domain, "engine", 1, PROFILE)
    with pytest.raises(KMeshFenceError):
        rebooted.heartbeat(fence, "session-a")


def test_unrelated_registrations_and_inventory_order_do_not_disarm() -> None:
    registry = DomainRegistry(["store", "retrieve"])
    domain = registry.open_domain(identity())
    lease = registry.open_registration(domain, "subject", 1, PROFILE)
    fence = registry.bind_worker(lease, 0, 10, "subject-session", identity(), PROFILE)
    other = registry.open_registration(
        registry.open_domain(identity(layout_digest="a" * 64)), "other", 1, PROFILE
    )
    registry.revoke(other)
    registry.update_capabilities(["retrieve", "store", "store"])
    assert registry.state(lease) == RegistrationState.READY
    registry.finish(
        registry.acquire(fence, "subject-session", "request", 1), Completion.SUCCEEDED
    )


def test_publication_serializes_with_revoke() -> None:
    registry = DomainRegistry(["store"])
    lease = registry.open_registration(
        registry.open_domain(identity()), "engine", 1, PROFILE
    )
    fence = registry.bind_worker(lease, 0, 10, "session", identity(), PROFILE)
    ticket = registry.acquire(fence, "session", "request", 1)
    entered = threading.Event()
    release = threading.Event()
    attempted = threading.Event()
    revoked = threading.Event()

    def commit(namespace: str) -> None:
        assert namespace == lease.domain.namespace
        entered.set()
        assert release.wait(timeout=5)
        assert not revoked.is_set()

    def revoke() -> None:
        attempted.set()
        registry.revoke(lease)
        revoked.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        publishing = pool.submit(registry.publish, ticket, "session", commit)
        assert entered.wait(timeout=5)
        revoking = pool.submit(revoke)
        assert attempted.wait(timeout=5)
        release.set()
        publishing.result(timeout=5)
        revoking.result(timeout=5)
    assert revoked.is_set()
    registry.finish(ticket, Completion.SUCCEEDED)
    assert registry.state(lease) == RegistrationState.REVOKED


def test_publication_failure_retains_debt() -> None:
    registry = DomainRegistry(["store"])
    lease = registry.open_registration(
        registry.open_domain(identity()), "engine", 1, PROFILE
    )
    fence = registry.bind_worker(lease, 0, 10, "session", identity(), PROFILE)
    ticket = registry.acquire(fence, "session", "request", 1)

    def fail(_: str) -> None:
        raise OSError("commit failed")

    with pytest.raises(OSError):
        registry.publish(ticket, "session", fail)
    assert registry.pending(lease) == 1
    assert registry.state(lease) == RegistrationState.REVOKED
    assert registry.open_domain(identity()).namespace != lease.domain.namespace
    registry.finish(ticket, Completion.FAILED_UNDRAINED)
