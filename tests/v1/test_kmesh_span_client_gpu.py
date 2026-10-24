# SPDX-License-Identifier: Apache-2.0
"""Qualify actual registered worker operations, reader ownership and retirement."""

# Standard
from typing import Literal
import gc
import math
import multiprocessing as mp

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev
from lmcache.integration.vllm.kmesh_span_client import (
    RegisteredSpanWorker,
    SpanWorkerArtifacts,
)
from lmcache.v1.gpu_connector.load_state import LayerwiseLoadState
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan
from tests.v1 import test_kmesh_span_store_gpu as span_fixture
from tests.v1.test_kmesh_rotary_snapshot_gpu import read_snapshot

pytestmark = span_fixture.pytestmark


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("packed", [False, True])
def test_registered_worker_real_span_and_checked_exporter_retirement(
    transport: Literal["zmq", "grpc"], layout: Literal["NHD", "HND"], packed: bool
) -> None:
    device = torch.device("cuda", 0)
    layers, count, source_start, destination = 24, 9, 5, 3
    pool = torch.empty((64 << 20) // 2, dtype=torch.bfloat16, device=device)
    shape: tuple[int, ...] = (2, 16, 4, 2, 16) if layout == "NHD" else (2, 16, 2, 4, 16)
    if packed:
        shape = (16, 4, 2, 32) if layout == "NHD" else (16, 2, 4, 32)
    kv = pool[: layers * math.prod(shape)].reshape(layers, *shape)
    kv.fill_(-9)
    canonical = (
        torch.arange(layers * 2 * count * 2 * 16, dtype=torch.float32)
        .reshape(layers, 2, count, 2, 16)
        .remainder(53)
        .div(17)
        .to(torch.bfloat16)
    )
    source_gpu = canonical.to(device)
    source_blocks = [2, 4, 6, 8, 10, 12]
    span_fixture.place_source(kv, source_gpu, source_blocks, source_start, layout)
    rotary_pool = torch.empty((64 << 20) // 4, dtype=torch.float32, device=device)
    rotary = rotary_pool[: 64 * 16].reshape(64, 16)
    phases = torch.arange(64, dtype=torch.float32)[:, None] * torch.arange(1, 9) * 0.013
    rotary.copy_(torch.cat((phases.cos(), phases.sin()), dim=1))
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(
        target=span_fixture.span_service, args=(child, transport, "success")
    )
    process.start()
    child.close()
    client = None
    worker = None
    try:
        address = read_snapshot(parent)
        assert isinstance(address, str)
        client = RequestClientFactory.create(address)
        worker = RegisteredSpanWorker.connect(
            client,
            tuple(kv.unbind(0)),
            rotary,
            layout=layout,
            model_name="synthetic-span-worker",
            is_neox_style=True,
            artifacts=SpanWorkerArtifacts("1" * 64, "2" * 64, "3" * 64, "4" * 64),
        )
        slots = tuple(
            source_blocks[p // 4] * 4 + p % 4 for p in range(len(source_blocks) * 4)
        )
        blocks = worker.block_table(slots)
        assert blocks == tuple(source_blocks)
        stored = worker.store(tuple(range(count)), source_start, "5" * 64, blocks)
        assert stored.source_start == source_start and stored.token_count == count
        assert not worker.pending
        kv.fill_(-9)
        destination_blocks = [13, 11, 9, 7]
        window = destination + count + 2
        slots = tuple(destination_blocks[p // 4] * 4 + p % 4 for p in range(window))
        blocks = worker.block_table(slots)
        placed = worker.restore(stored, destination, "6" * 64, blocks)
        assert placed.receipt.state == "loaded-needs-repair" and not worker.pending
        actual = span_fixture.read_window(kv, list(blocks), destination, count, layout)
        expected = span_fixture.rotate_reference(canonical, destination - source_start)
        torch.testing.assert_close(actual[:, 0], expected[:, 0], rtol=0.012, atol=0.025)
        assert torch.equal(
            actual[:, 1].view(torch.uint16), canonical[:, 1].view(torch.uint16)
        )
        tokens = (999,) * destination + tuple(range(count)) + (888,) * 2
        spans = ExplicitTokenSpans(
            window,
            TokenSpan(0, window),
            (TokenSpan(0, 1), TokenSpan(destination, destination + count)),
        )
        unused = worker.layer_source(tokens, blocks, spans, (placed,))
        unused.close()
        assert unused.is_closed and not unused.pending
        source = worker.layer_source(tokens, blocks, spans, (placed,))
        state = LayerwiseLoadState()
        inputs = torch.tensor(tokens, dtype=torch.int64, device=device)
        stream = source.retrieve_layer(inputs, None, load_state=state)
        first = next(stream)
        assert first is not None and int(first.item()) == count
        assert next(stream) is None and worker.pending
        tensor = state.buffer_mapping[0].tensor
        assert tensor is not None
        loaded = (
            tensor[:, destination : destination + count].cpu().reshape(2, count, 2, 16)
        )
        assert torch.equal(loaded.view(torch.uint16), actual[0].view(torch.uint16))
        with pytest.raises(RuntimeError, match="live model source"):
            worker.close()
        stream.close()
        assert source.is_closed and not worker.pending and state.owned_buffer_count == 0
        release = worker.close()
        assert release.context == worker.runtime.context
        assert len(release.allocation_ids) == 2
        assert worker.close() == release

        # Drop every original GPU view after checked retirement. New allocation
        # remains valid while the importing process is still alive.
        del tensor, stream, source, inputs, source_gpu, kv, pool, rotary, rotary_pool
        gc.collect()
        torch_dev.empty_cache()
        fresh = torch.full(((64 << 20) // 2,), 7, dtype=torch.bfloat16, device=device)
        torch_dev.synchronize(device)
        assert torch.all(fresh == 7).item()
        assert process.is_alive() and worker.close() == release
        parent.send("stop")
        assert read_snapshot(parent) == "closed"
        process.join(30)
        assert process.exitcode == 0 and torch.all(fresh == 7).item()
    finally:
        if process.is_alive():
            process.terminate()
        process.join(30)
        if client is not None:
            client.close()
        parent.close()
