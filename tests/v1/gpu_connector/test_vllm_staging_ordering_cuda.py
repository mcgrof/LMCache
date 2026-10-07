# SPDX-License-Identifier: Apache-2.0
"""Native CUDA regression tests for the synchronous staging handoff.

Run with LMCACHE_TEST_CUDA_STAGING_ORDERING=1 and a CUDA/native LMCache build.
CUDA_LAUNCH_BLOCKING must be unset: it would conceal the races under test.
These tests use the actual V2/V3 connector and compiled transfer kernel, without
NVMe I/O. They qualify the CUDA handoff, not DMA-BUF visibility or PCIe routing.

All buffers, pointer tables, kernels, streams and events are warmed before the
finite delay is queued. Readiness is checked before any device-to-host copy;
otherwise a synchronizing verifier could make an early return appear correct.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
import os

# Third Party
import pytest

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.gpu_connector.gpu_connectors import (
        VLLMPagedMemGPUConnectorV2,
        VLLMPagedMemGPUConnectorV3,
    )
    from lmcache.v1.memory_management import MemoryObj
else:
    torch = pytest.importorskip("torch")

pytestmark = pytest.mark.no_shared_allocator

LAYERS, BLOCKS, BLOCK_SIZE, HEADS, HEAD_SIZE = 3, 8, 16, 3, 32
CHUNK_SIZE = 8
GENERATIONS = (101, 202, 303)
# A finite device delay, not an event that depends on a blocked host thread.
# About 0.1 s on common GPUs; the pending-event assertion rejects a test whose
# scheduling delay consumed the window before the connector was called.
DELAY_CYCLES = 200_000_000
SLOTS = (1, 18, 35, 52, 69, 86, 103, 120, 4, 21, 38, 55, 72, 89, 106, 123)


@dataclass
class StagingCase:
    """Preallocated native connector, independent oracle and owned buffers."""

    connector: VLLMPagedMemGPUConnectorV2 | VLLMPagedMemGPUConnectorV3
    kv: list[torch.Tensor]
    objects: list[MemoryObj]
    slot_mapping: torch.Tensor
    initial_cpu: torch.Tensor
    initial_gpu: torch.Tensor
    expected_chunks: dict[int, list[torch.Tensor]]
    expected_pools: dict[int, torch.Tensor]
    producer: torch.cuda.Stream
    producer_ready: list[torch.cuda.Event]
    delay_ready: list[torch.cuda.Event]
    starts: list[int]
    ends: list[int]


def assert_payload_bytes(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare every payload byte against an independent CPU oracle.

    Args:
        actual: CUDA tensor, whose readiness the caller already checked.
        expected: CPU tensor containing the expected bytes and shape.

    Raises:
        AssertionError: If any byte, including an untouched slot, differs.
    """
    observed = actual.detach().contiguous().view(torch.uint8).cpu()
    oracle = expected.contiguous().view(torch.uint8)
    assert torch.equal(observed, oracle), (
        f"{int(torch.count_nonzero(observed != oracle))} payload bytes differ"
    )


def finite_cuda_delay() -> None:
    """Queue a bounded native sleep on the current CUDA stream.

    The caller must check a following event remains pending before testing the
    handoff. This makes a scheduling delay that consumed the window a failure.
    """
    # This private PyTorch test primitive deliberately delays device execution;
    # it does not access LMCache internals or replace the native transfer kernel.
    torch.cuda._sleep(DELAY_CYCLES)


def make_staging_case(version: str, intermediate: bool, chunks: int) -> StagingCase:
    """Build and warm the actual native transfer path before delaying work.

    Args:
        version: Connector version, ``v2`` or ``v3``.
        intermediate: Exercise the connector's optional GPU scratch buffer.
        chunks: Number of independently checked chunks, one or two.

    Returns:
        Owned CUDA buffers, connector, warmed streams/events and CPU oracles.

    Raises:
        ValueError: If the requested test geometry is unsupported.
    """
    # First Party
    from lmcache.v1.gpu_connector.gpu_connectors import (
        VLLMPagedMemGPUConnectorV2,
        VLLMPagedMemGPUConnectorV3,
    )
    from lmcache.v1.memory_management import (
        MemoryFormat,
        MemoryObjMetadata,
        TensorMemoryObj,
    )
    from lmcache.v1.metadata import LMCacheMetadata

    if version not in ("v2", "v3") or chunks not in (1, 2):
        raise ValueError("expected v2/v3 and one or two chunks")
    device = torch.device("cuda", torch.cuda.current_device())
    shape = (LAYERS, 2, BLOCKS, BLOCK_SIZE, HEADS, HEAD_SIZE)
    count = LAYERS * 2 * BLOCKS * BLOCK_SIZE * HEADS * HEAD_SIZE
    initial_cpu = (torch.arange(count) % 997).reshape(shape).to(torch.float16)
    initial_gpu = initial_cpu.to(device)
    kv = [initial_gpu[layer].clone() for layer in range(LAYERS)]
    metadata = LMCacheMetadata(
        model_name="native-staging-ordering",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(LAYERS, 2, CHUNK_SIZE, HEADS, HEAD_SIZE),
        chunk_size=CHUNK_SIZE,
    )
    connector: VLLMPagedMemGPUConnectorV2 | VLLMPagedMemGPUConnectorV3
    if version == "v2":
        connector = VLLMPagedMemGPUConnectorV2.from_metadata(
            metadata, use_gpu=intermediate, device=device
        )
    else:
        connector = VLLMPagedMemGPUConnectorV3.from_metadata(
            metadata, use_gpu=intermediate, device=device
        )
    staging_shape = torch.Size((2, LAYERS, CHUNK_SIZE, HEADS * HEAD_SIZE))
    size = staging_shape.numel() * torch.float16.itemsize
    objects: list[MemoryObj] = []
    for _ in range(chunks):
        raw = torch.empty(size, dtype=torch.uint8, device=device)
        obj = TensorMemoryObj(
            raw,
            MemoryObjMetadata(
                shape=staging_shape,
                dtype=torch.float16,
                address=0,
                phy_size=size,
                ref_count=1,
                fmt=MemoryFormat.KV_2LTD,
                shapes=[staging_shape],
                dtypes=[torch.float16],
            ),
            parent_allocator=None,
        )
        objects.append(obj)
    slots = SLOTS[: chunks * CHUNK_SIZE]
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)
    starts = [index * CHUNK_SIZE for index in range(chunks)]
    ends = [start + CHUNK_SIZE for start in starts]
    expected_chunks = {}
    expected_pools = {}
    for generation in GENERATIONS:
        full = initial_cpu + generation
        # CPU-only oracle: explicit slot interpretation independent of the
        # connector and its native kernel. Values are exactly representable.
        expected_chunks[generation] = [
            torch.stack(
                [
                    full[:, :, slot // BLOCK_SIZE, slot % BLOCK_SIZE]
                    for slot in slots[start:end]
                ],
                dim=2,
            )
            .permute(1, 0, 2, 3, 4)
            .reshape(staging_shape)
            for start, end in zip(starts, ends, strict=True)
        ]
        restored = torch.zeros_like(full)
        for slot in slots:
            restored[:, :, slot // BLOCK_SIZE, slot % BLOCK_SIZE] = full[
                :, :, slot // BLOCK_SIZE, slot % BLOCK_SIZE
            ]
        expected_pools[generation] = restored
    case = StagingCase(
        connector,
        kv,
        objects,
        slot_mapping,
        initial_cpu,
        initial_gpu,
        expected_chunks,
        expected_pools,
        torch.cuda.Stream(),
        [torch.cuda.Event() for _ in GENERATIONS],
        [torch.cuda.Event() for _ in GENERATIONS],
        starts,
        ends,
    )
    # Warm both native directions and initialize pointer/group metadata before
    # creating the race window. No allocation, host read or lazy initialization
    # should accidentally supply the dependency that the test is checking.
    connector.batched_from_gpu(
        objects, starts, ends, kvcaches=kv, slot_mapping=slot_mapping
    )
    connector.store_stream.synchronize()
    connector.batched_to_gpu(
        objects, starts, ends, kvcaches=kv, slot_mapping=slot_mapping
    )
    for event in case.producer_ready + case.delay_ready:
        event.record(case.producer)
    with torch.cuda.stream(case.producer):
        torch.cuda._sleep(1)
        for layer, tensor in enumerate(kv):
            tensor.copy_(initial_gpu[layer])
            tensor.add_(0)
    torch.cuda.synchronize()
    return case


@pytest.fixture(autouse=True)
def require_native_cuda() -> None:
    """Skip unavailable hardware/builds, but never silently run a fallback."""
    if os.environ.get("LMCACHE_TEST_CUDA_STAGING_ORDERING") != "1":
        pytest.skip("set LMCACHE_TEST_CUDA_STAGING_ORDERING=1 for native CUDA tests")
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
        pytest.fail("unset CUDA_LAUNCH_BLOCKING: it conceals stream-ordering races")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("requires an NVIDIA CUDA device")
    native = pytest.importorskip("lmcache.cuda_ops", reason="needs native CUDA build")
    # First Party
    from lmcache import device_ops

    device_ops.ensure_native()
    assert device_ops.multi_layer_kv_transfer is native.multi_layer_kv_transfer, (
        "refusing to count the torch fallback as a native CUDA transfer test"
    )
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("this PyTorch build lacks the finite CUDA delay test primitive")


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("resident_tokens", [0, 3, CHUNK_SIZE])
def test_native_restore_preserves_resident_prefix(
    version: str, resident_tokens: int
) -> None:
    """Restore only missing tokens; a fully resident chunk is a native no-op."""
    case = make_staging_case(version, False, 1)
    for obj in case.objects:
        assert obj.tensor is not None
        obj.tensor.fill_(97)
    expected = case.initial_cpu.clone()
    for slot in SLOTS[resident_tokens:CHUNK_SIZE]:
        expected[:, :, slot // BLOCK_SIZE, slot % BLOCK_SIZE].fill_(97)

    case.connector.batched_to_gpu(
        case.objects,
        case.starts,
        case.ends,
        kvcaches=case.kv,
        slot_mapping=case.slot_mapping,
        vllm_cached_tokens=resident_tokens,
    )

    assert_payload_bytes(torch.stack(case.kv), expected)


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("intermediate", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("delay_on", ["producer", "gather"])
def test_native_store_returns_current_generation_ready(
    version: str, intermediate: bool, batched: bool, delay_on: str
) -> None:
    """A CPU I/O submitter may consume every staging byte when store returns."""
    case = make_staging_case(version, intermediate, 2 if batched else 1)
    connector = case.connector
    try:
        for index, generation in enumerate(GENERATIONS):
            for obj in case.objects:
                cast(torch.Tensor, obj.raw_tensor).fill_(0xA5)
            torch.cuda.synchronize()
            with torch.cuda.stream(case.producer):
                if delay_on == "producer":
                    finite_cuda_delay()
                for layer, tensor in enumerate(case.kv):
                    tensor.copy_(case.initial_gpu[layer])
                    tensor.add_(generation)
                case.producer_ready[index].record()
            if delay_on == "gather":
                case.producer_ready[index].synchronize()
                with torch.cuda.stream(connector.store_stream):
                    finite_cuda_delay()
                    case.delay_ready[index].record()
                gate = case.delay_ready[index]
            else:
                gate = case.producer_ready[index]
            assert not gate.query(), "delay ended before test; race was not exercised"
            with torch.cuda.stream(case.producer):
                kwargs = dict(kvcaches=case.kv, slot_mapping=case.slot_mapping)
                if batched:
                    connector.batched_from_gpu(
                        case.objects, case.starts, case.ends, **kwargs
                    )
                else:
                    connector.from_gpu(case.objects[0], 0, CHUNK_SIZE, **kwargs)
            # No .cpu(), item(), global sync or downstream wait before these.
            assert gate.query(), "store returned before its delayed dependency"
            assert case.producer_ready[index].query(), "producer is still running"
            assert connector.store_stream.query(), "gather is still running"
            for obj, expected in zip(
                case.objects, case.expected_chunks[generation], strict=True
            ):
                assert_payload_bytes(cast(torch.Tensor, obj.tensor), expected)
            for layer, tensor in enumerate(case.kv):
                assert_payload_bytes(tensor, case.initial_cpu[layer] + generation)
    finally:
        # A failing old implementation may leave work live. Keep all owned
        # buffers alive until that work drains before letting pytest unwind.
        torch.cuda.synchronize()


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("intermediate", [False, True])
def test_native_restore_finishes_scatter_before_staging_reuse(
    version: str, intermediate: bool
) -> None:
    """Existing batched restore permits immediate staging reuse after return."""
    case = make_staging_case(version, intermediate, 2)
    # Preallocate and upload source payloads before any delayed scatter.
    payloads = {
        generation: [value.to(case.slot_mapping.device) for value in values]
        for generation, values in case.expected_chunks.items()
    }
    try:
        for index, generation in enumerate(GENERATIONS):
            for obj, expected in zip(case.objects, payloads[generation], strict=True):
                cast(torch.Tensor, obj.tensor).copy_(expected)
            for tensor in case.kv:
                tensor.zero_()
            torch.cuda.synchronize()
            with torch.cuda.stream(case.connector.load_stream):
                finite_cuda_delay()
                case.delay_ready[index].record()
            assert not case.delay_ready[index].query(), (
                "delay ended before test; race was not exercised"
            )
            case.connector.batched_to_gpu(
                case.objects,
                case.starts,
                case.ends,
                kvcaches=case.kv,
                slot_mapping=case.slot_mapping,
            )
            # These precede reuse and all synchronizing verification.
            assert case.delay_ready[index].query(), "restore returned before delay"
            assert case.connector.load_stream.query(), "scatter is still running"
            # Simulate a pool owner reusing the bytes immediately after return;
            # source ownership is maintained until the documented handoff ends.
            with torch.cuda.stream(case.producer):
                for obj in case.objects:
                    cast(torch.Tensor, obj.raw_tensor).fill_(0x5A)
            for layer, tensor in enumerate(case.kv):
                assert_payload_bytes(tensor, case.expected_pools[generation][layer])
    finally:
        torch.cuda.synchronize()
