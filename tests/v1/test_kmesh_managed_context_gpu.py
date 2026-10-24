# SPDX-License-Identifier: Apache-2.0
"""Qualify checked raw context teardown in a separate importing process.

This exercises the physical resource primitive, not a service release receipt.
Controlled native errors occur before driver effects; actual faults are untested.
"""

# Standard
from multiprocessing.connection import Connection
from types import SimpleNamespace
import multiprocessing as mp
import traceback

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.gpu_connector import gds_context
from lmcache.v1.platform.devices.cuda import cache_context as cache_module
from lmcache.v1.platform.devices.cuda.cache_context import (
    GPUCacheContext,
    retained_managed_raw_contexts,
)
from lmcache.v1.platform.devices.cuda.ipc_wrapper import RawCudaIPCWrapper
from lmcache.v1.platform.devices.cuda.utils import _cuda
from tests.v1.test_kmesh_device_copy_gpu import receive

pytestmark = pytest.mark.skipif(
    not (torch_dev.is_available() and torch_device_type == "cuda"),
    reason="requires raw CUDA IPC and actual context stream completion",
)


def managed_context_child(channel: Connection, mode: str) -> None:
    """Keep construction and every imported view inside the importing process."""
    try:
        exercise_managed_context(channel, mode)
    except BaseException:
        channel.send(("error", traceback.format_exc()))
        raise
    else:
        channel.send(("closed",))
    finally:
        channel.close()


def exercise_managed_context(channel: Connection, mode: str) -> None:
    owners = channel.recv()
    assert retained_managed_raw_contexts() == 0
    native_close = _cuda.runtime.cudaIpcCloseMemHandle
    raw_import = RawCudaIPCWrapper.to_tensor
    closes: list[int] = []

    def observe_close(pointer: int) -> object:
        closes.append(pointer)
        if mode == "unknown-close":
            return (_cuda.runtime.cudaError_t.cudaErrorInvalidValue,)
        return native_close(pointer)

    with pytest.MonkeyPatch.context() as patch:
        if mode == "gds-before":
            patch.setattr(
                gds_context,
                "get_gds_context",
                lambda: SimpleNamespace(initialized=True),
            )
            with pytest.raises(RuntimeError, match="inactive GDS"):
                GPUCacheContext(owners[:1], 8, {"kv_layout": "NHD"}, managed_raw=True)
            assert retained_managed_raw_contexts() == 0
            assert all(owner.raw_mapping_released for owner in owners)
            channel.send(("refused-before-import",))
            return
        if mode == "partial-construction":
            imports = 0

            def partial_import(owner: RawCudaIPCWrapper) -> torch.Tensor:
                nonlocal imports
                imports += 1
                if imports == 2:
                    raise RuntimeError("injected before second import")
                return raw_import(owner)

            patch.setattr(RawCudaIPCWrapper, "to_tensor", partial_import)
            with pytest.raises(RuntimeError, match="second import"):
                GPUCacheContext(owners, 8, {"kv_layout": "NHD"}, managed_raw=True)
            assert imports == 2
            assert retained_managed_raw_contexts() == 1
            assert not owners[0].raw_mapping_released
            channel.send(("requires-importer-exit",))
            return
        if mode == "construction-drain":

            def refuse_initialization_drain(device: object, stream: object) -> None:
                raise RuntimeError("injected initialization stream refusal")

            patch.setattr(
                cache_module, "synchronize_stream", refuse_initialization_drain
            )
            with pytest.raises(RuntimeError, match="initialization stream"):
                GPUCacheContext(owners[:1], 8, {"kv_layout": "NHD"}, managed_raw=True)
            assert retained_managed_raw_contexts() == 1
            assert not owners[0].raw_mapping_released
            channel.send(("requires-importer-exit",))
            return

    cache = GPUCacheContext(owners[:1], 8, {"kv_layout": "NHD"}, managed_raw=True)
    assert retained_managed_raw_contexts() == 1
    assert torch.all(cache.kv_tensors[0] == 7).item()
    alias = owners[1].to_tensor() if mode == "alias" else None
    device = cache.device
    matrix = torch.full((2048, 2048), 1 / 2048, device=device)
    workspace = torch.empty_like(matrix)
    torch_dev.synchronize(device)
    with torch_dev.stream(cache.stream):
        for _ in range(512):
            torch.mm(matrix, matrix, out=workspace)
    assert not cache.stream.query(), "context drain challenge already completed"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_cuda.runtime, "cudaIpcCloseMemHandle", observe_close)
        if mode == "drain-refusal":

            def refuse_drain(stream: object) -> None:
                raise RuntimeError("injected before stream drain")

            patch.setattr(torch.cuda.Stream, "synchronize", refuse_drain)
        elif mode == "close-refusal":

            def refuse_close(owner: RawCudaIPCWrapper) -> None:
                raise RuntimeError("injected before checked close")

            patch.setattr(RawCudaIPCWrapper, "close_checked", refuse_close)
        elif mode == "gds-after":
            patch.setattr(
                gds_context,
                "get_gds_context",
                lambda: SimpleNamespace(initialized=True),
            )
        if mode == "success":
            expected = (owners[0].allocation_identity,)
            assert cache.close_raw_checked() == expected
            assert cache.close_raw_checked() == expected
            cache.close()
            assert len(closes) == 1
        else:
            with pytest.raises(RuntimeError):
                cache.close_raw_checked()
            assert retained_managed_raw_contexts() == 1
            assert not owners[0].raw_mapping_released
            if mode == "alias":
                assert alias is not None
                assert torch.all(alias == 9).item()
                with pytest.raises(RuntimeError, match="retired"):
                    owners[1].to_tensor()
                assert not closes
                owners[1].close_checked()
                assert len(closes) == 1
            elif mode == "unknown-close":
                assert len(closes) == 1
                with pytest.raises(RuntimeError, match="importer teardown"):
                    cache.close_raw_checked()
                assert len(closes) == 1, "ambiguous native pointer was resubmitted"
                channel.send(("requires-importer-exit",))
                return
            else:
                assert not closes
    assert cache.close_raw_checked() == (owners[0].allocation_identity,)
    assert owners[0].raw_mapping_released
    assert retained_managed_raw_contexts() == 0
    channel.send(("released",))


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "alias",
        "drain-refusal",
        "close-refusal",
        "unknown-close",
        "partial-construction",
        "construction-drain",
        "gds-before",
        "gds-after",
    ],
)
def test_checked_managed_raw_context_release(mode: str) -> None:
    device = torch.device("cuda", 0)
    pool = torch.empty(64 * (1 << 20) // 2, device=device, dtype=torch.bfloat16)
    first = pool[:2048].reshape(2, 16, 4, 2, 8)
    second = pool[4096:6144].reshape(2, 16, 4, 2, 8)
    first.fill_(7)
    second.fill_(9)
    torch_dev.synchronize(device)
    owners = [RawCudaIPCWrapper.wrap(tensor) for tensor in (first, second)]
    assert owners[0].allocation_identity == owners[1].allocation_identity
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(
        target=managed_context_child, args=(child, mode)
    )
    process.start()
    child.close()
    try:
        parent.send(owners)
        expected = (
            "refused-before-import"
            if mode == "gds-before"
            else "requires-importer-exit"
            if mode in ("unknown-close", "partial-construction", "construction-drain")
            else "released"
        )
        assert receive(parent) == (expected,)
        assert receive(parent) == ("closed",)
        process.join(timeout=30)
        assert process.exitcode == 0
        assert torch.all(first == 7).item()
        assert torch.all(second == 9).item()
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=30)
        parent.close()
        torch_dev.synchronize(device)
