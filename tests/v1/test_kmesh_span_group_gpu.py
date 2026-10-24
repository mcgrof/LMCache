# SPDX-License-Identifier: Apache-2.0
"""Qualify simultaneous private GPU inputs with synthetic layer transforms."""

# Standard
from dataclasses import replace
from typing import Literal
import math
import multiprocessing as mp

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import kmesh_span_layers as layer_module
from lmcache.integration.vllm.kmesh_span_client import (
    RegisteredSpanWorker,
    SpanWorkerArtifacts,
)
from lmcache.integration.vllm.kmesh_span_group import (
    RestoredSpanGroup,
    RestoredSpanGroupMember,
)
from lmcache.integration.vllm.kmesh_span_layers import RestoredSpanBatch
from lmcache.v1.compute.blend.layer_source import (
    BlendLayerSourceRequest,
    UnsettledLayerSourceError,
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
@pytest.mark.parametrize("mode", ["writeback", "early-close", "failed-drain"])
def test_registered_group_owns_private_layers_and_all_member_cleanup(
    transport: Literal["zmq", "grpc"],
    layout: Literal["NHD", "HND"],
    packed: bool,
    mode: str,
) -> None:
    device = torch.device("cuda", 0)
    layers, count, source_start = 24, 9, 5
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
    try:
        address = read_snapshot(parent)
        assert isinstance(address, str)
        client = RequestClientFactory.create(address)
        worker = RegisteredSpanWorker.connect(
            client,
            tuple(kv.unbind(0)),
            rotary,
            layout=layout,
            model_name="synthetic-span-group",
            is_neox_style=True,
            artifacts=SpanWorkerArtifacts("1" * 64, "2" * 64, "3" * 64, "4" * 64),
        )
        stored = worker.store(
            tuple(range(count)), source_start, "5" * 64, tuple(source_blocks)
        )
        kv.fill_(-9)
        members, before = [], []
        for index in range(4):
            start = index + 1
            window = start + count + 2
            blocks = tuple(range(index * 4, index * 4 + math.ceil(window / 4)))
            slots = tuple(blocks[p // 4] * 4 + p % 4 for p in range(window))
            assert worker.block_table(slots) == blocks
            placed = worker.restore(stored, start, "6" * 64, blocks)
            batch = RestoredSpanBatch(
                worker.fence,
                worker.runtime.context,
                (999,) * start + tuple(range(count)) + (888,) * 2,
                blocks,
                ExplicitTokenSpans(
                    window,
                    TokenSpan(0, window),
                    (TokenSpan(0, 1), TokenSpan(start, start + count)),
                ),
                (placed,),
            )
            members.append(RestoredSpanGroupMember(f"model-{index}", 1, batch))
            before.append(span_fixture.read_window(kv, list(blocks), 0, window, layout))
            expected = span_fixture.rotate_reference(canonical, start - source_start)
            torch.testing.assert_close(
                before[-1][:, 0, start : start + count],
                expected[:, 0],
                rtol=0.012,
                atol=0.025,
            )
            assert torch.equal(
                before[-1][:, 1, start : start + count].view(torch.uint16),
                canonical[:, 1].view(torch.uint16),
            )
        metadata = RestoredSpanGroup(tuple(members))
        owner = worker.layer_source_group(metadata)
        requests = tuple(
            BlendLayerSourceRequest(
                m.request_id, m.generation, m.batch.token_ids, m.batch.explicit_spans
            )
            for m in members
        )
        invalid = (
            requests[0],
            requests[1],
            requests[2],
            replace(requests[3], generation=2),
        )
        with pytest.raises(ValueError, match="complete registered group"):
            owner.bind(invalid)
        assert not owner.pending and not owner.is_closed
        with pytest.raises(RuntimeError, match="still owns its blocks"):
            worker.layer_source(
                members[0].batch.token_ids,
                members[0].batch.block_ids,
                members[0].batch.explicit_spans,
                members[0].batch.placements,
            )
        with pytest.raises(RuntimeError, match="still owns pending work"):
            worker.restore(stored, 1, "6" * 64, members[0].batch.block_ids)
        sources = owner.bind(requests)
        with pytest.raises(RuntimeError, match="already bound"):
            owner.bind(requests)
        states = [LayerwiseLoadState() for _ in members]
        streams = [
            source.retrieve_layer(
                torch.tensor(m.batch.token_ids, dtype=torch.int64, device=device),
                None,
                load_state=state,
            )
            for source, m, state in zip(sources, members, states, strict=True)
        ]
        for stream in streams:
            hit = next(stream)
            assert hit is not None and int(hit.item()) == count
        for stream in streams:
            assert next(stream) is None
        assert owner.pending and all(state.owned_buffer_count == 1 for state in states)
        with pytest.raises(RuntimeError, match="live model source"):
            owner.close()
        with pytest.raises(RuntimeError, match="live model source"):
            worker.close()
        expected = []
        for m, prior in zip(members, before, strict=True):
            start = m.batch.placements[0].destination_start
            value = torch.zeros_like(prior)
            value[:, :, start : start + count] = prior[:, :, start : start + count]
            expected.append(value)
        if mode == "failed-drain":
            attempts = 0

            def refuse_drain(device: torch.device) -> None:
                nonlocal attempts
                attempts += 1
                raise RuntimeError("synthetic group drain refusal")

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(layer_module, "synchronize_device", refuse_drain)
                for stream in streams:
                    with pytest.raises(UnsettledLayerSourceError):
                        stream.close()
                assert attempts == 4
                with pytest.raises(UnsettledLayerSourceError) as raised:
                    owner.close()
                assert attempts == 8 and owner.pending
                with pytest.raises(UnsettledLayerSourceError):
                    raised.value.drain()
                assert attempts == 12 and all(
                    state.owned_buffer_count == 1 for state in states
                )
            worker.drain_pending()
            assert attempts == 12
        else:
            for layer in range(layers):
                pointers = set()
                for index, (state, m) in enumerate(zip(states, members, strict=True)):
                    tensor = state.buffer_mapping[layer].tensor
                    assert tensor is not None
                    pointers.add(tensor.data_ptr())
                    loaded = tensor.cpu().reshape(2, len(m.batch.token_ids), 2, 16)
                    assert torch.equal(
                        loaded.view(torch.uint16),
                        expected[index][layer].view(torch.uint16),
                    )
                    tensor.add_(layer + index + 1)
                    expected[index][layer].add_(layer + index + 1)
                assert len(pointers) == 4
                if mode == "early-close":
                    for stream in streams:
                        stream.close()
                    break
                if layer + 1 < layers:
                    for stream in streams:
                        assert next(stream) is None
            if mode == "writeback":
                for stream in streams:
                    hit = next(stream)
                    assert hit is not None and int(hit.sum().item()) == count
                    stream.close()
        owner.close()
        assert owner.is_closed and not owner.pending and not worker.pending
        assert all(state.owned_buffer_count == 0 for state in states)
        for index, m in enumerate(members):
            actual = span_fixture.read_window(
                kv, list(m.batch.block_ids), 0, len(m.batch.token_ids), layout
            )
            value = expected[index] if mode == "writeback" else before[index]
            assert torch.equal(actual.view(torch.uint16), value.view(torch.uint16))
            tail = span_fixture.read_window(
                kv,
                list(range(index * 4, (index + 1) * 4)),
                len(m.batch.token_ids),
                16 - len(m.batch.token_ids),
                layout,
            )
            assert torch.all(tail == -9).item()
        receipt = worker.close()
        assert worker.close() == receipt and process.is_alive()
        parent.send("stop")
        assert read_snapshot(parent) == "closed"
        process.join(30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(30)
        if client is not None:
            client.close()
        parent.close()
