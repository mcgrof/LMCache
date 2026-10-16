# SPDX-License-Identifier: Apache-2.0
"""Collective cached-K position recovery must match the serial path bitwise.

The fakes mirror the layerwise connector contract: per layer they produce a
deterministic bf16 ``[2, tokens, hidden]`` buffer with per-token source and
target positions, then either rotate inline (serial) or publish a pending
recovery (collective).  The fake model snapshots the buffer bytes it observes
each layer, so a buffer that reaches model compute unrotated, or rotated
differently from the serial path, fails the bitwise comparison.
"""

# Standard
from collections.abc import Generator
from typing import Any, Optional, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.compute.blend.blender import (
    LMCBlender,
    LMCBlendGroupOutcome,
    LMCBlendGroupRequest,
)
from lmcache.v1.gpu_connector.load_state import (
    LayerwiseLoadState,
    PendingPositionRecovery,
)
from lmcache.v1.memory_management import MemoryObj

HIDDEN = 8
HALF = HIDDEN // 2
THETA = 1_000_000.0


class _TorchDeltaRope:
    """Pure-torch stand-in for the fused cached-K rotation kernel."""

    def __init__(self) -> None:
        self.calls = 0
        self.fail_on_call: Optional[int] = None

    def __call__(
        self,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("injected collective rotation failure")
        inv_freq = THETA ** (-torch.arange(0, HALF, dtype=torch.float32) * 2.0 / HIDDEN)
        delta = (new_positions - old_positions).to(torch.float32)
        ang = delta[:, None] * inv_freq[None, :]
        cos, sin = torch.cos(ang), torch.sin(ang)
        kf = k.to(torch.float32)
        x1, x2 = kf[:, :HALF], kf[:, HALF:]
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat([y1, y2], dim=-1).to(k.dtype)


class _FakeBuffer:
    """Owns one [2, tokens, hidden] tensor with MemoryObj release counting."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.release_count = 0

    def ref_count_down(self) -> None:
        self.release_count += 1


class _FakePositionalCacheEngine:
    """Connector-shaped retrieval that defers or applies position recovery."""

    def __init__(
        self,
        num_layers: int,
        rope: _TorchDeltaRope,
        member_specs: dict[str, dict[str, Any]],
    ) -> None:
        self.num_layers = num_layers
        self.rope = rope
        self.member_specs = member_specs
        self.buffers: dict[str, list[_FakeBuffer]] = {}
        self.double_publish: Optional[str] = None

    def _layer_buffer(
        self, request_id: str, layer_id: int, token_count: int
    ) -> torch.Tensor:
        seed = hash((request_id, layer_id)) % (2**31)
        g = torch.Generator().manual_seed(seed)
        return torch.randn(2, token_count, HIDDEN, generator=g).bfloat16()

    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        load_state: LayerwiseLoadState,
        req_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        del mask, kwargs
        request_id = str(req_id)
        spec = self.member_specs[request_id]
        token_count = int(tokens.numel())
        offset = int(spec.get("offset", 0))
        is_miss = bool(spec.get("miss", False))
        gap_positions = torch.tensor(spec.get("gaps", ()), dtype=torch.int64)
        self.buffers[request_id] = []

        load_state.begin()
        try:
            if is_miss:
                yield torch.zeros(token_count, dtype=torch.bool)
                for _ in range(self.num_layers):
                    yield None
                yield torch.zeros(token_count, dtype=torch.bool)
                return

            yield torch.tensor(token_count)
            old_positions = torch.arange(token_count, dtype=torch.int64)
            new_positions = old_positions + offset
            for layer_id in range(self.num_layers):
                buffer = _FakeBuffer(
                    self._layer_buffer(request_id, layer_id, token_count)
                )
                self.buffers[request_id].append(buffer)
                load_state.own_buffer(cast(MemoryObj, buffer))
                load_state.gap_positions = gap_positions
                if load_state.defer_position_recovery:
                    pending = PendingPositionRecovery(
                        layer_id=layer_id,
                        kv_buffer=cast(MemoryObj, buffer),
                        old_positions=old_positions,
                        new_positions=new_positions,
                        gap_positions=gap_positions,
                    )
                    load_state.publish_pending_position_recovery(pending)
                    if self.double_publish == request_id:
                        load_state.publish_pending_position_recovery(pending)
                else:
                    buffer.tensor[0] = self.rope(
                        old_positions, new_positions, buffer.tensor[0]
                    )
                    load_state.record_position_recovery(serial=1)
                    if gap_positions.numel():
                        buffer.tensor[:, gap_positions] = 0.0
                load_state.buffer_mapping[layer_id] = cast(MemoryObj, buffer)
                yield None

            yield torch.ones(token_count, dtype=torch.bool)
        finally:
            load_state.release()


class _SnapshottingModel:
    """Fake layerwise model that snapshots the buffer bytes it observes."""

    def __init__(self, num_layers: int, rope: _TorchDeltaRope) -> None:
        self.num_layers = num_layers
        self.fused_rotary_emb = rope
        self.observed: dict[str, list[torch.Tensor]] = {}

    def compute_layer(self, tokens: torch.Tensor, session) -> Generator:
        del tokens
        return self._compute(session)

    def _compute(self, session) -> Generator[None, None, None]:
        request_id = str(session.request_id)
        self.observed[request_id] = []
        for layer_id in range(self.num_layers):
            buffer = session.load_state.buffer_mapping[layer_id]
            assert session.load_state.pending_position_recovery is None
            self.observed[request_id].append(buffer.tensor.clone())
            yield


def _make_blender(
    num_layers: int, member_specs: dict[str, dict[str, Any]]
) -> tuple[LMCBlender, _FakePositionalCacheEngine, _SnapshottingModel]:
    rope = _TorchDeltaRope()
    blender = LMCBlender.__new__(LMCBlender)
    engine = _FakePositionalCacheEngine(num_layers, rope, member_specs)
    model = _SnapshottingModel(num_layers, rope)
    blender.cache_engine = engine
    blender.layerwise_model = model
    blender.num_layers = num_layers
    return blender, engine, model


def _requests(member_specs: dict[str, dict[str, Any]]) -> list[LMCBlendGroupRequest]:
    return [
        LMCBlendGroupRequest(
            request_id=request_id,
            tokens=list(range(1, int(spec.get("tokens", 16)) + 1)),
        )
        for request_id, spec in member_specs.items()
    ]


BASE_SPECS: dict[str, dict[str, Any]] = {
    "member-a": {"tokens": 16, "offset": 1, "gaps": (0, 5)},
    "member-b": {"tokens": 16, "offset": 2, "gaps": ()},
    "member-c": {"tokens": 16, "offset": 3, "gaps": (15,)},
}


def _run(specs: dict[str, dict[str, Any]], *, collective: bool):
    blender, engine, model = _make_blender(4, specs)
    result = blender.blend_group(
        _requests(specs), collective_position_recovery=collective
    )
    return result, engine, model


def test_collective_matches_serial_bitwise() -> None:
    serial_result, _, serial_model = _run(BASE_SPECS, collective=False)
    collective_result, _, collective_model = _run(BASE_SPECS, collective=True)

    for result in (serial_result, collective_result):
        assert all(
            member.outcome is LMCBlendGroupOutcome.RECOVERED
            for member in result.members
        )
    for request_id in BASE_SPECS:
        serial_layers = serial_model.observed[request_id]
        collective_layers = collective_model.observed[request_id]
        assert len(serial_layers) == len(collective_layers) == 4
        for serial_seen, collective_seen in zip(
            serial_layers, collective_layers, strict=True
        ):
            assert bool(
                serial_seen.view(torch.int16)
                .eq(collective_seen.view(torch.int16))
                .all()
            )

    for member in serial_result.members:
        counts = member.operation_counts
        assert counts.position_recovery_serial_operations == 4
        assert counts.position_recovery_collective_operations == 0
    for member in collective_result.members:
        counts = member.operation_counts
        assert counts.position_recovery_serial_operations == 0
        assert counts.position_recovery_collective_operations == 4

    assert serial_result.collective_passes == ()
    assert len(collective_result.collective_passes) == 4
    for layer_id, collective_pass in enumerate(collective_result.collective_passes):
        assert collective_pass.layer_id == layer_id
        assert collective_pass.member_count == 3
        assert collective_pass.token_count == 48


def test_ragged_token_counts_match_serial_bitwise() -> None:
    specs = {
        "member-a": {"tokens": 16, "offset": 1, "gaps": (2,)},
        "member-b": {"tokens": 24, "offset": 5, "gaps": ()},
        "member-c": {"tokens": 8, "offset": 2, "gaps": (0, 7)},
    }
    _, _, serial_model = _run(specs, collective=False)
    collective_result, _, collective_model = _run(specs, collective=True)
    for request_id in specs:
        for serial_seen, collective_seen in zip(
            serial_model.observed[request_id],
            collective_model.observed[request_id],
            strict=True,
        ):
            assert bool(
                serial_seen.view(torch.int16)
                .eq(collective_seen.view(torch.int16))
                .all()
            )
    assert all(
        collective_pass.token_count == 48
        for collective_pass in collective_result.collective_passes
    )


def test_collective_uses_one_rotation_call_per_layer() -> None:
    _, serial_engine, _ = _run(BASE_SPECS, collective=False)
    assert serial_engine.rope.calls == 3 * 4
    _, collective_engine, _ = _run(BASE_SPECS, collective=True)
    assert collective_engine.rope.calls == 4


def test_miss_member_never_participates() -> None:
    specs = dict(BASE_SPECS)
    specs["member-miss"] = {"tokens": 16, "miss": True}
    result, _, model = _run(specs, collective=True)
    outcomes = {member.request_id: member.outcome for member in result.members}
    assert outcomes["member-miss"] is LMCBlendGroupOutcome.MISS
    assert all(
        outcomes[request_id] is LMCBlendGroupOutcome.RECOVERED
        for request_id in BASE_SPECS
    )
    assert all(
        collective_pass.member_count == 3
        for collective_pass in result.collective_passes
    )
    miss_counts = next(
        member.operation_counts
        for member in result.members
        if member.request_id == "member-miss"
    )
    assert miss_counts.position_recovery_collective_operations == 0
    assert "member-miss" not in model.observed


def test_rotation_failure_poisons_participants_only() -> None:
    specs = dict(BASE_SPECS)
    specs["member-miss"] = {"tokens": 16, "miss": True}
    blender, engine, _ = _make_blender(4, specs)
    engine.rope.fail_on_call = 2
    result = blender.blend_group(_requests(specs), collective_position_recovery=True)
    outcomes = {member.request_id: member.outcome for member in result.members}
    assert outcomes["member-miss"] is LMCBlendGroupOutcome.MISS
    for request_id in BASE_SPECS:
        assert outcomes[request_id] is LMCBlendGroupOutcome.FAILED
    assert len(result.collective_passes) == 1
    for buffers in engine.buffers.values():
        assert all(buffer.release_count == 1 for buffer in buffers)


def test_double_publish_fails_closed() -> None:
    specs = {
        "member-a": {"tokens": 16, "offset": 1, "gaps": ()},
        "member-b": {"tokens": 16, "offset": 2, "gaps": ()},
    }
    blender, engine, _ = _make_blender(4, specs)
    engine.double_publish = "member-b"
    result = blender.blend_group(_requests(specs), collective_position_recovery=True)
    outcomes = {member.request_id: member.outcome for member in result.members}
    assert outcomes["member-b"] is LMCBlendGroupOutcome.FAILED
    assert outcomes["member-a"] is LMCBlendGroupOutcome.RECOVERED


def test_model_step_rejects_unapplied_pending() -> None:
    specs = {"member-a": {"tokens": 16, "offset": 1, "gaps": ()}}
    blender, _, _ = _make_blender(4, specs)
    session = blender.create_session(
        list(range(1, 17)),
        request_id="member-a",
        defer_position_recovery=True,
    )
    session.prime()
    session.step_retrieval()
    with pytest.raises(RuntimeError, match="deferred position recovery"):
        session.step_model()
    assert session.is_closed


def test_step_model_requires_retrieval_first() -> None:
    specs = {"member-a": {"tokens": 16, "offset": 1, "gaps": ()}}
    blender, _, _ = _make_blender(4, specs)
    with blender.create_session(list(range(1, 17)), request_id="member-a") as session:
        session.prime()
        with pytest.raises(RuntimeError, match="retrieval"):
            session.step_model()
        session.step_retrieval()
        with pytest.raises(RuntimeError, match="already advanced"):
            session.step_retrieval()
        session.step_model()


def test_serial_group_never_defers() -> None:
    result, engine, _ = _run(BASE_SPECS, collective=False)
    assert result.collective_passes == ()
    for member in result.members:
        assert member.outcome is LMCBlendGroupOutcome.RECOVERED
        assert member.operation_counts.position_recovery_collective_operations == 0
    del engine


def test_reserved_defer_kwarg_is_rejected() -> None:
    blender, _, _ = _make_blender(4, BASE_SPECS)
    request = LMCBlendGroupRequest(
        request_id="member-a",
        tokens=[1, 2, 3],
        retrieve_kwargs={"defer_position_recovery": True},
    )
    with pytest.raises(ValueError, match="defer_position_recovery"):
        blender.blend_group([request])


def test_load_state_pending_lifecycle() -> None:
    state = LayerwiseLoadState(defer_position_recovery=True)
    state.begin()
    buffer = _FakeBuffer(torch.zeros(2, 4, HIDDEN, dtype=torch.bfloat16))
    pending = PendingPositionRecovery(
        layer_id=0,
        kv_buffer=cast(MemoryObj, buffer),
        old_positions=torch.arange(4, dtype=torch.int64),
        new_positions=torch.arange(4, dtype=torch.int64) + 1,
        gap_positions=torch.empty(0, dtype=torch.int64),
    )
    with pytest.raises(RuntimeError, match="No position recovery"):
        state.consume_pending_position_recovery()
    state.publish_pending_position_recovery(pending)
    with pytest.raises(RuntimeError, match="never applied"):
        state.publish_pending_position_recovery(pending)
    assert state.consume_pending_position_recovery() is pending
    state.publish_pending_position_recovery(pending)
    state.release()
    assert state.pending_position_recovery is None

    undeferred = LayerwiseLoadState()
    undeferred.begin()
    with pytest.raises(RuntimeError, match="not enabled"):
        undeferred.publish_pending_position_recovery(pending)
