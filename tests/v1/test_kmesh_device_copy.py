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
from lmcache.v1.multiprocess.modules.blend.rope import BorrowedRope, provisional_rope
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

    def copy_(self, source: DeviceInput | torch.Tensor, non_blocking: bool) -> None:
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
        device = torch.device(kwargs["device"])
        assert kwargs["pin_memory"] == (device.type == "cpu")
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
            values = source if isinstance(source, torch.Tensor) else source.values
            target.values = values.clone()
        self.work.clear()


class ControlledRaw:
    """Model import/close debt without creating a CUDA allocation or mapping."""

    def __init__(self, driver: ControlledDriver) -> None:
        self.driver = driver
        self.tensor = DeviceInput(torch.arange(16).reshape(2, 8))
        self.mapped = False
        self.fail_import = False
        self.fail_close = False
        self.interrupt_close = False

    def to_tensor(self) -> DeviceInput:
        self.mapped = True
        if self.fail_import:
            raise RuntimeError("injected partial raw import")
        return self.tensor

    def close_checked(self) -> None:
        self.driver.events.append("raw-close")
        assert not self.driver.work, "mapping close before pending device reads drain"
        if self.interrupt_close:
            self.interrupt_close = False
            raise KeyboardInterrupt("injected raw close interruption")
        if self.fail_close:
            raise RuntimeError("injected checked raw close refusal")
        self.mapped = False


def controlled_raw_borrows(
    monkeypatch: pytest.MonkeyPatch, driver: ControlledDriver, count: int = 1
) -> tuple[
    BorrowedContext,
    BorrowedRope,
    ContextOwnership,
    ContextOwnership,
    list[ControlledRaw],
]:
    # First Party
    from lmcache.v1.platform.devices.cuda import ipc_wrapper

    paged, rotary, paged_owner, rotary_owner = controlled_borrows(monkeypatch, driver)
    owners = [ControlledRaw(driver) for _ in range(count)]
    rotary.state.cos_sin_caches.clear()
    rotary.state.backing_handles = tuple(owners)
    monkeypatch.setattr(ipc_wrapper, "RawCudaIPCWrapper", ControlledRaw)
    monkeypatch.setattr(copy_module, "torch_device_type", "cuda")
    return paged, rotary, paged_owner, rotary_owner, owners


def controlled_borrows(
    monkeypatch: pytest.MonkeyPatch, driver: ControlledDriver, location: str = "device"
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
        cos_sin_caches=[
            torch.arange(16).reshape(2, 8)
            if location == "host"
            else DeviceInput(torch.arange(16).reshape(2, 8))
        ],
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


def test_raw_imports_are_owned_before_mapping_and_closed_after_copy_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = ControlledDriver()
    paged, rotary, paged_owner, rotary_owner, owners = controlled_raw_borrows(
        monkeypatch, driver
    )
    manager = RetainedTensorCopies()

    def validate(actual_paged: BorrowedContext, actual_rotary: BorrowedRope) -> None:
        assert actual_paged is paged and actual_rotary is rotary
        assert manager.pending == paged_owner.pending == rotary_owner.pending == 1
        assert owners[0].mapped

    snapshots = manager.copy_raw_to_cpu(paged, rotary, b"producer-ready", validate)
    assert torch.equal(snapshots[0].values, owners[0].tensor.values)
    assert not owners[0].mapped
    assert manager.pending == paged_owner.pending == rotary_owner.pending == 0
    manager.close()


def test_partial_raw_import_failure_releases_every_opened_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = ControlledDriver()
    paged, rotary, paged_owner, rotary_owner, owners = controlled_raw_borrows(
        monkeypatch, driver, 2
    )
    owners[1].fail_import = True
    manager = RetainedTensorCopies()
    with pytest.raises(KMeshFenceError, match="copy failed"):
        manager.copy_raw_to_cpu(paged, rotary, b"producer-ready", lambda *_: None)
    assert not any(owner.mapped for owner in owners)
    assert manager.pending == paged_owner.pending == rotary_owner.pending == 0
    manager.close()


@pytest.mark.parametrize("failure", ["close", "interruption", "copy-undrained"])
def test_raw_cleanup_debt_retains_borrows_until_later_actual_drain(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    driver = ControlledDriver("copy" if failure == "copy-undrained" else "")
    paged, rotary, paged_owner, rotary_owner, owners = controlled_raw_borrows(
        monkeypatch, driver
    )
    owners[0].fail_close = failure == "close"
    owners[0].interrupt_close = failure == "interruption"
    driver.drain_fails = failure == "copy-undrained"
    manager = RetainedTensorCopies()
    error = KeyboardInterrupt if failure == "interruption" else KMeshFenceError
    with pytest.raises(error):
        manager.copy_raw_to_cpu(paged, rotary, b"producer-ready", lambda *_: None)
    assert manager.pending == paged_owner.pending == rotary_owner.pending == 1
    assert owners[0].mapped
    if failure == "copy-undrained":
        assert "raw-close" not in driver.events
    if failure == "close":
        with pytest.raises(KMeshFenceError, match="must drain"):
            manager.close()
        assert manager.pending == paged_owner.pending == rotary_owner.pending == 1
    owners[0].fail_close = driver.drain_fails = False
    assert manager.drain_quarantined(paged_owner.identity) == 0
    assert not owners[0].mapped
    assert paged_owner.pending == rotary_owner.pending == 0
    manager.close()


def test_raw_metadata_refusal_does_not_return_or_publish_copied_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = ControlledDriver()
    paged, rotary, paged_owner, rotary_owner, owners = controlled_raw_borrows(
        monkeypatch, driver
    )
    manager = RetainedTensorCopies()

    def refuse(*_: object) -> None:
        raise ValueError("unsupported registered layout")

    with pytest.raises(KMeshFenceError, match="copy failed"):
        manager.copy_raw_to_cpu(paged, rotary, b"producer-ready", refuse)
    assert not driver.destinations
    assert not owners[0].mapped
    assert manager.pending == paged_owner.pending == rotary_owner.pending == 0
    manager.close()


def test_raw_copy_refuses_cpu_context_and_releases_only_consumed_borrows() -> None:
    with rotary_server() as (blend, transfer):
        context, identity = transfer.context_identity(99), blend.rope_identity(99)
        assert context is not None and identity is not None
        with request_session("zmq", b"worker"):
            paged = transfer.borrow_context(99, context, lambda: None)
            rotary = blend.borrow_rope(99, identity, context, lambda: None)
        manager = RetainedTensorCopies()
        with pytest.raises(KMeshFenceError, match="unpublished CUDA inputs"):
            manager.copy_raw_to_cpu(paged, rotary, b"", lambda *_: None)
        assert (
            manager.pending
            == paged.entry.ownership.pending
            == rotary.state.ownership.pending
            == 0
        )
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


def test_completed_host_snapshot_prepares_without_replacing_registered_state() -> None:
    with rotary_server() as (blend, transfer):
        context, before = transfer.context_identity(99), blend.rope_identity(99)
        assert context is not None and before is not None
        values = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        with request_session("zmq", b"worker"):
            paged = transfer.borrow_context(99, context, lambda: None)
            owned = provisional_rope(context, [values], 8, True, [0], [], [], "")
        assert owned.state.ownership.identity != before.rotary
        manager = RetainedTensorCopies()
        result = manager.copy_to_device(paged, owned)
        assert torch.equal(result[0], values)
        result[0].zero_()
        assert values.count_nonzero().item() > 0
        assert blend.rope_identity(99) == before
        assert manager.pending == owned.state.ownership.pending == 0
        manager.close()


@pytest.mark.parametrize("owner", [None, b"foreign"])
def test_provisional_generation_requires_the_actual_context_owner(
    owner: bytes | None,
) -> None:
    with rotary_server() as (blend, transfer):
        context, before = transfer.context_identity(99), blend.rope_identity(99)
        assert context is not None
        with request_session("zmq", owner) if owner is not None else nullcontext():
            with pytest.raises(RuntimeError):
                provisional_rope(context, [torch.ones(2, 8)], 8, True, [0], [], [], "")
        assert blend.rope_identity(99) == before


@pytest.mark.parametrize(
    "failure,uncertain",
    [
        ("", False),
        ("copy", False),
        ("synchronize", False),
        ("copy", True),
        ("synchronize", True),
    ],
)
def test_upload_retains_completed_host_sources_and_new_device_destinations(
    monkeypatch: pytest.MonkeyPatch, failure: str, uncertain: bool
) -> None:
    driver = ControlledDriver(failure)
    driver.drain_fails = uncertain
    paged, rotary, paged_owner, rotary_owner = controlled_borrows(
        monkeypatch, driver, "host"
    )
    manager = RetainedTensorCopies()
    if failure:
        with pytest.raises(KMeshFenceError, match="copy failed"):
            manager.copy_to_device(paged, rotary)
        assert (
            manager.pending
            == paged_owner.pending
            == rotary_owner.pending
            == int(uncertain)
        )
    else:
        result = manager.copy_to_device(paged, rotary)
        assert torch.equal(result[0].values, torch.arange(16).reshape(2, 8))
        assert manager.pending == paged_owner.pending == rotary_owner.pending == 0
    assert "import" not in driver.events and "wait" not in driver.events
    driver.drain_fails = False
    assert manager.drain_quarantined() == 0
    assert paged_owner.pending == rotary_owner.pending == 0
    manager.close()


def test_upload_refuses_device_inputs_before_submitting_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = ControlledDriver()
    paged, rotary, paged_owner, rotary_owner = controlled_borrows(monkeypatch, driver)
    manager = RetainedTensorCopies()
    with pytest.raises(KMeshFenceError, match="one registered device"):
        manager.copy_to_device(paged, rotary)
    assert not driver.events and manager.pending == 0
    assert paged_owner.pending == rotary_owner.pending == 0
