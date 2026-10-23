# SPDX-License-Identifier: Apache-2.0
"""Check CPU snapshots and inject device failures without allocating a GPU.

The controlled backend models pending reads through weak references. It does
not prove device execution; the separate CUDA tests exercise physical copies.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
import gc
import threading
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess import kmesh_device_copy as copy_module
from lmcache.v1.multiprocess.context_ownership import (
    ContextOwnership,
    ContextOwnershipError,
)
from lmcache.v1.multiprocess.kmesh_device_copy import RetainedTensorCopies
from lmcache.v1.multiprocess.kmesh_domains import KMeshFenceError
from lmcache.v1.multiprocess.modules.blend.rope import BorrowedRope
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    BorrowedContext,
    ContextEntry,
)
from lmcache.v1.multiprocess.transport.request_context import (
    get_request_session,
    request_session,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from tests.v1.test_registered_rope_ownership import rotary_server


@dataclass
class DeviceInput:
    values: torch.Tensor
    device: torch.device = torch.device("cuda:0")


class HostDestination:
    def __init__(self, driver: "ControlledDriver") -> None:
        self.driver = driver
        self.values: torch.Tensor | None = None

    def copy_(self, source: DeviceInput, non_blocking: bool) -> None:
        assert non_blocking
        self.driver.work.append((weakref.ref(self), weakref.ref(source)))
        self.driver.fail("copy")


class ControlledDriver:
    def __init__(self, failure: str = "") -> None:
        self.failure = failure
        self.drain_fails = False
        self.work: list[tuple[weakref.ReferenceType, weakref.ReferenceType]] = []
        self.events: list[str] = []
        self.destinations: list[weakref.ReferenceType] = []

    def fail(self, stage: str) -> None:
        self.events.append(stage)
        if self.failure == stage:
            self.failure = ""
            raise RuntimeError("injected " + stage)

    def check_event_support(self, device: object) -> None:
        self.fail("support")

    def import_event(self, handle: bytes, device: object) -> object:
        assert handle == b"producer-ready"
        self.fail("import")
        return object()

    def wait_event(self, event: object, stream: object) -> None:
        self.fail("wait")

    def allocate(self, tensor: object, **kwargs: Any) -> HostDestination:
        assert kwargs["device"] == "cpu" and kwargs["pin_memory"] is True
        self.fail("allocate")
        target = HostDestination(self)
        self.destinations.append(weakref.ref(target))
        return target

    def drain(self, device: object, stream: object) -> None:
        self.events.append("drain")
        if self.failure == "synchronize":
            self.failure = ""
            raise RuntimeError("injected first stream synchronization")
        if self.drain_fails:
            raise RuntimeError("injected undrained stream")
        self.complete()

    def complete(self) -> None:
        for target_ref, source_ref in self.work:
            target, source = target_ref(), source_ref()
            assert target is not None and source is not None
            target.values = source.values.clone()
        self.work.clear()


def controlled_borrows(
    monkeypatch: pytest.MonkeyPatch, driver: ControlledDriver
) -> tuple[BorrowedContext, BorrowedRope, ContextOwnership, ContextOwnership]:
    """Use real ownership holds with explicitly simulated device metadata."""
    monkeypatch.setattr(
        copy_module,
        "torch_dev",
        SimpleNamespace(device=lambda _: nullcontext(), stream=lambda _: nullcontext()),
    )
    monkeypatch.setattr(copy_module.torch, "empty_like", driver.allocate)
    monkeypatch.setattr(copy_module, "synchronize_stream", driver.drain)
    with request_session("zmq", b"worker"):
        paged_owner, rotary_owner = ContextOwnership(), ContextOwnership()
        paged_hold = paged_owner.borrow(
            paged_owner.identity, get_request_session(), lambda: None
        )
        rotary_hold = rotary_owner.borrow(
            rotary_owner.identity, get_request_session(), lambda: None
        )
    cache = SimpleNamespace(device=torch.device("cuda:0"), stream=object())
    paged = BorrowedContext(
        ContextEntry(
            cast(BaseCacheContext, cache),
            "synthetic-device-control",
            1,
            event_backend=cast(Any, driver),
            ownership=paged_owner,
        ),
        paged_hold,
    )
    state = SimpleNamespace(
        cos_sin_caches=[DeviceInput(torch.arange(16).reshape(2, 8))],
        context_identity=paged_owner.identity,
        ownership=rotary_owner,
    )
    return paged, BorrowedRope(cast(Any, state), rotary_hold), paged_owner, rotary_owner


def test_cpu_snapshot_uses_real_registration_and_independent_host_storage() -> None:
    with rotary_server() as (blend, transfer):
        context, identity = transfer.context_identity(99), blend.rope_identity(99)
        assert context is not None and identity is not None
        with request_session("zmq", b"worker"):
            paged = transfer.borrow_context(99, context, lambda: None)
            rotary = blend.borrow_rope(99, identity, context, lambda: None)
        manager = RetainedTensorCopies()
        snapshot = manager.copy_to_cpu(paged, rotary)
        original = rotary.state.cos_sin_caches[0]
        assert torch.equal(snapshot[0], original)
        snapshot[0].zero_()
        assert original.count_nonzero().item() > 0
        assert manager.pending == 0
        assert paged.entry.ownership.pending == rotary.state.ownership.pending == 0
        manager.close()


@pytest.mark.parametrize(
    "failure",
    [
        "support",
        "import",
        "wait",
        "allocate",
        "copy",
        "synchronize",
    ],
)
def test_failed_device_action_with_verified_drain_releases_consumed_borrows(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    driver = ControlledDriver(failure)
    paged, rotary, paged_owner, rotary_owner = controlled_borrows(monkeypatch, driver)
    manager = RetainedTensorCopies()
    with pytest.raises(KMeshFenceError, match="copy failed"):
        manager.copy_to_cpu(paged, rotary, b"producer-ready")
    assert driver.events[-1] == "drain"
    assert not driver.work and manager.pending == 0
    assert paged_owner.pending == rotary_owner.pending == 0
    manager.close()


@pytest.mark.parametrize("failure", ["copy", "synchronize"])
def test_uncertain_copy_survives_cancellation_collection_and_shutdown_until_drain(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    driver = ControlledDriver(failure)
    driver.drain_fails = True
    paged, rotary, paged_owner, rotary_owner = controlled_borrows(monkeypatch, driver)
    manager = RetainedTensorCopies()
    with pytest.raises(KMeshFenceError, match="copy failed"):
        manager.copy_to_cpu(paged, rotary, b"producer-ready")
    assert manager.pending == paged_owner.pending == rotary_owner.pending == 1
    for owner in (paged_owner, rotary_owner):
        with pytest.raises(ContextOwnershipError):
            owner.retire(timeout=0)
    with pytest.raises(KMeshFenceError, match="before shutdown"):
        manager.close()
    manager_ref = weakref.ref(manager)
    del paged, rotary, manager
    gc.collect()
    retained = manager_ref()
    assert retained is not None and retained.pending == 1
    assert driver.destinations and all(ref() is not None for ref in driver.destinations)
    driver.drain_fails = False
    assert retained.drain_quarantined() == 0
    assert not driver.work and paged_owner.pending == rotary_owner.pending == 0
    retained.close()


def test_successful_device_control_waits_before_copy_and_completion_before_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = ControlledDriver()
    paged, rotary, paged_owner, rotary_owner = controlled_borrows(monkeypatch, driver)
    manager = RetainedTensorCopies()
    snapshots = manager.copy_to_cpu(paged, rotary, b"producer-ready")
    assert torch.equal(snapshots[0].values, torch.arange(16).reshape(2, 8))
    assert driver.events == [
        "support",
        "import",
        "wait",
        "allocate",
        "copy",
        "drain",
    ]
    assert manager.pending == paged_owner.pending == rotary_owner.pending == 0


@pytest.mark.parametrize("refusal", ["producer", "context", "closing"])
def test_pre_submission_refusal_consumes_borrows_without_device_work(
    monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    driver = ControlledDriver()
    paged, rotary, paged_owner, rotary_owner = controlled_borrows(monkeypatch, driver)
    manager = RetainedTensorCopies()
    if refusal == "context":
        rotary.state.context_identity = ContextOwnership().identity
    if refusal == "closing":
        manager.close()
    with pytest.raises(KMeshFenceError):
        manager.copy_to_cpu(
            paged, rotary, b"" if refusal == "producer" else b"producer-ready"
        )
    assert not driver.events and manager.pending == 0
    assert paged_owner.pending == rotary_owner.pending == 0


def test_active_copy_keeps_holds_and_shutdown_refuses_without_waiting_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = ControlledDriver()
    paged, rotary, paged_owner, rotary_owner = controlled_borrows(monkeypatch, driver)
    entered, complete = threading.Event(), threading.Event()

    def synchronize(device: object, stream: object) -> None:
        entered.set()
        assert complete.wait(timeout=5)
        driver.complete()

    monkeypatch.setattr(copy_module, "synchronize_stream", synchronize)
    manager = RetainedTensorCopies()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(manager.copy_to_cpu, paged, rotary, b"producer-ready")
        try:
            assert entered.wait(timeout=5)
            assert not pending.done()
            assert manager.pending == paged_owner.pending == rotary_owner.pending == 1
            assert manager.drain_quarantined() == 1
            with pytest.raises(KMeshFenceError, match="before shutdown"):
                manager.close()
            assert paged_owner.pending == rotary_owner.pending == 1
        finally:
            complete.set()
        assert pending.result(timeout=5)
    assert manager.pending == paged_owner.pending == rotary_owner.pending == 0
    manager.close()


def test_interrupted_copy_keeps_both_holds_until_a_verified_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = ControlledDriver()
    driver.drain_fails = True
    paged, rotary, paged_owner, rotary_owner = controlled_borrows(monkeypatch, driver)

    def interrupt(stage: str) -> None:
        if stage == "copy":
            raise KeyboardInterrupt("injected cancellation after submission")

    monkeypatch.setattr(driver, "fail", interrupt)
    manager = RetainedTensorCopies()
    with pytest.raises(KMeshFenceError, match="copy failed"):
        manager.copy_to_cpu(paged, rotary, b"producer-ready")
    assert manager.pending == paged_owner.pending == rotary_owner.pending == 1
    driver.drain_fails = False
    assert manager.drain_quarantined() == 0
    assert paged_owner.pending == rotary_owner.pending == 0
    manager.close()
