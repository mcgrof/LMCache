# SPDX-License-Identifier: Apache-2.0
"""Consume actual remote restores with a synthetic GPU layer transformation.

This checks the native model-input/writeback boundary, not model repair quality.
"""

# Standard
from typing import Literal

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops
from lmcache.integration.vllm import kmesh_span_layers as layer_module
from lmcache.integration.vllm.kmesh_span_layers import (
    CompletedSpanPlacement,
    RestoredPagedLayerSource,
    RestoredSpanBatch,
)
from lmcache.v1.gpu_connector.load_state import LayerwiseLoadState
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan
from tests.v1 import test_kmesh_span_store_gpu as span_fixture
import lmcache.lmcache_native as lmcache_native

pytestmark = span_fixture.pytestmark


def consume_placement(case: span_fixture.SpanConsumerCase, mode: str) -> None:
    """Read actual restored slots, transform a layer and check native writeback."""
    start, count = case.destination_start, case.token_count
    window = start + count + 2
    tokens = (999,) * start + tuple(range(count)) + (888,) * 2
    batch = RestoredSpanBatch(
        case.fence,
        case.runtime.context,
        tokens,
        case.blocks,
        ExplicitTokenSpans(
            window,
            TokenSpan(0, window),
            (TokenSpan(0, 1), TokenSpan(start, start + count)),
        ),
        (CompletedSpanPlacement("destination", 1, start, case.receipt),),
    )
    before = span_fixture.read_window(
        case.kv, list(case.blocks), 0, window, case.layout
    )
    source = RestoredPagedLayerSource(batch, tuple(case.kv.unbind(0)), case.layout)
    state = LayerwiseLoadState()
    inputs = torch.tensor(tokens, dtype=torch.int64, device=case.kv.device)
    stream = source.retrieve_layer(inputs, None, load_state=state)
    first = next(stream)
    assert first is not None and int(first.item()) == count
    if mode == "failed-drain":
        gather = device_ops.multi_layer_kv_transfer
        calls = 0

        def fail_after_gather(
            target: torch.Tensor,
            pointers: torch.Tensor,
            slots: torch.Tensor,
            device: torch.device,
            page_buffer_size: int,
            direction: lmcache_native.TransferDirection,
            fmt: lmcache_native.EngineKVFormat,
            *,
            block_size: int,
            head_size: int,
        ) -> None:
            nonlocal calls
            gather(
                target,
                pointers,
                slots,
                device,
                page_buffer_size,
                direction,
                fmt,
                block_size=block_size,
                head_size=head_size,
            )
            calls += 1
            raise RuntimeError("injected after actual worker-layer gather")

        def refuse_drain(device: torch.device) -> None:
            raise RuntimeError("injected worker-device drain refusal")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(device_ops, "multi_layer_kv_transfer", fail_after_gather)
            patch.setattr(layer_module, "synchronize_device", refuse_drain)
            with pytest.raises(RuntimeError, match="drain refusal"):
                next(stream)
            assert source.pending and state.owned_buffer_count == 1
            with pytest.raises(RuntimeError, match="drain refusal"):
                source.drain_failed()
            assert source.pending and calls == 1
        source.drain_failed()
        assert not source.pending and state.owned_buffer_count == 0 and calls == 1
        return

    expected = torch.zeros_like(before)
    expected[:, :, start : start + count] = before[:, :, start : start + count]
    for layer in range(case.kv.shape[0]):
        assert next(stream) is None
        assert source.pending
        tensor = state.buffer_mapping[layer].tensor
        assert tensor is not None
        loaded = tensor.cpu().reshape(2, window, 2, 16)
        assert torch.equal(
            loaded.view(torch.uint16), expected[layer].view(torch.uint16)
        )
        tensor.add_(layer + 1)
        expected[layer].add_(layer + 1)
        if mode == "early-close":
            stream.close()
            assert not source.pending and state.owned_buffer_count == 0
            unchanged = span_fixture.read_window(
                case.kv, list(case.blocks), 0, window, case.layout
            )
            assert torch.equal(unchanged.view(torch.uint16), before.view(torch.uint16))
            return
    result = next(stream)
    assert result is not None and int(result.sum().item()) == count
    stream.close()
    assert not source.pending and state.owned_buffer_count == 0
    actual = span_fixture.read_window(
        case.kv, list(case.blocks), 0, window, case.layout
    )
    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))
    tail = span_fixture.read_window(
        case.kv, list(case.blocks), window, 28 - window, case.layout
    )
    assert torch.all(tail == -9).item()


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("mode", ["writeback", "early-close", "failed-drain"])
def test_registered_paged_inputs_and_checked_layer_writeback(
    transport: Literal["zmq", "grpc"],
    layout: Literal["NHD", "HND"],
    mode: str,
    packed: bool,
) -> None:
    span_fixture.test_managed_span_native_store_and_restore(
        transport,
        "success",
        5,
        3,
        layout,
        24,
        9,
        consume=lambda case: consume_placement(case, mode),
        packed=packed,
    )
