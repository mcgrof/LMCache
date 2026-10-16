# SPDX-License-Identifier: Apache-2.0
"""Collective model-side QKV processing must match process_qkv bitwise.

The coordinator batches exactly two per-token elementwise operations — the
rotary application and the check-layer key-difference computation — by
token-axis concatenation.  These tests drive the REAL ``process_qkv`` and
the REAL ``_apply_collective_qkv`` with one shared deterministic rotary
callable and shared cached-KV sources, then compare every output tensor
and every metadata mutation bitwise.
"""

# Standard
from types import SimpleNamespace
from typing import Any, Optional

# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.blend.qkv_handoff import LMCQKVHandoff

HIDDEN = 8
HALF = HIDDEN // 2
THETA = 1_000_000.0


def _delta_rope(positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
    """Deterministic per-token stand-in for the vLLM rotary op."""
    inv_freq = THETA ** (-torch.arange(0, HALF, dtype=torch.float32) * 2.0 / HIDDEN)
    ang = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos, sin = torch.cos(ang), torch.sin(ang)

    def rotate(t: torch.Tensor) -> torch.Tensor:
        tf = t.to(torch.float32)
        x1, x2 = tf[:, :HALF], tf[:, HALF:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(t.dtype)

    return rotate(q), rotate(k)


class _Recorder:
    def __init__(self) -> None:
        self.serial_rope = 0
        self.serial_diff = 0
        self.collective_rope = 0
        self.collective_diff = 0

    def record_serial_operations(self, *, rope_operations=0, diff_operations=0):
        self.serial_rope += rope_operations
        self.serial_diff += diff_operations

    def record_collective_operations(self, *, rope_operations=0, diff_operations=0):
        self.collective_rope += rope_operations
        self.collective_diff += diff_operations


class _Metadata:
    def __init__(self) -> None:
        self.positions: Optional[torch.Tensor] = None
        self.imp_indices: Optional[torch.Tensor] = None


class _AttnMetadata:
    def __init__(self) -> None:
        self.updates: list[torch.Tensor] = []

    def update_from_top_indices(self, top_indices: torch.Tensor) -> None:
        self.updates.append(top_indices.clone())


def _member_session(
    tokens: int, gap: tuple[int, ...] = (), explicit: bool = False
) -> SimpleNamespace:
    recorder = _Recorder()
    session = SimpleNamespace(
        metadata=_Metadata(),
        load_state=SimpleNamespace(
            gap_positions=torch.tensor(gap, dtype=torch.int64),
            pending_position_recovery=None,
        ),
        requires_explicit_gap_recompute=explicit,
        record_serial_operations=recorder.record_serial_operations,
        record_collective_operations=recorder.record_collective_operations,
        recorder=recorder,
    )
    return session


def _make_blender(
    old_kv: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    check_layers: tuple[int, ...] = (1,),
) -> Any:
    # The real blender uses lightweight contract doubles for its dependencies.
    blender: Any = LMCBlender.__new__(LMCBlender)

    def get_kv(layer_id: int, load_state: Any):
        member_index = load_state.member_index
        k, v = old_kv[member_index][layer_id]
        return k, v

    blender.gpu_connector = SimpleNamespace(get_kv=get_kv)
    layer = SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=_delta_rope))
    blender.layerwise_model = SimpleNamespace(
        vllm_model=SimpleNamespace(model=SimpleNamespace(layers=[layer, layer, layer]))
    )
    blender.common_metadata = SimpleNamespace(
        check_layers=list(check_layers),
        recomp_ratios=[0.25],
        thresholds=None,
    )
    return blender


def _fresh_inputs(
    member_index: int, tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(31000 + member_index)
    q = torch.randn(tokens, HIDDEN, generator=g).bfloat16()
    k = torch.randn(tokens, HIDDEN, generator=g).bfloat16()
    v = torch.randn(tokens, HIDDEN, generator=g).bfloat16()
    residual = torch.randn(tokens, HIDDEN, generator=g).bfloat16()
    return q, k, v, residual


def _old_kv(member_index: int, layers: int, tokens: int):
    g = torch.Generator().manual_seed(62000 + member_index)
    return {
        layer: (
            torch.randn(tokens, HIDDEN, generator=g).bfloat16(),
            torch.randn(tokens, HIDDEN, generator=g).bfloat16(),
        )
        for layer in range(layers)
    }


def _run_both(
    layer_id: int,
    specs: list[dict[str, Any]],
    check_layers: tuple[int, ...] = (1,),
):
    """Run serial process_qkv and the collective coordinator on clones."""
    old = {i: _old_kv(i, 3, spec["tokens"]) for i, spec in enumerate(specs)}
    serial_old = {
        i: {layer: (k.clone(), v.clone()) for layer, (k, v) in per_member.items()}
        for i, per_member in old.items()
    }

    serial_out = []
    serial_blender = _make_blender(serial_old, check_layers)
    for i, spec in enumerate(specs):
        session = _member_session(
            spec["tokens"], spec.get("gap", ()), spec.get("explicit", False)
        )
        session.load_state.member_index = i
        q, k, v, residual = _fresh_inputs(i, spec["tokens"])
        out = serial_blender.process_qkv(
            q.clone(),
            k.clone(),
            v.clone(),
            residual.clone(),
            layer_id,
            None,
            _AttnMetadata(),
            session,
        )
        serial_out.append((out, session))

    collective_blender = _make_blender(old, check_layers)
    participants = []
    for i, spec in enumerate(specs):
        session = _member_session(
            spec["tokens"], spec.get("gap", ()), spec.get("explicit", False)
        )
        session.load_state.member_index = i
        session.outcome = None
        q, k, v, residual = _fresh_inputs(i, spec["tokens"])
        handoff = LMCQKVHandoff(
            layer_id=layer_id,
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            residual=residual.clone(),
            attn_output=None,
            attn_metadata=_AttnMetadata(),
        )
        state = SimpleNamespace(session=session, error=None)
        participants.append((state, handoff))
    corrections = collective_blender._apply_collective_qkv(participants)
    assert corrections is not None, [str(state.error) for state, _ in participants]
    return serial_out, participants, corrections, serial_old, old


def _assert_bitwise(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert bool(a.view(torch.int16).eq(b.view(torch.int16)).all()), label


def test_non_check_layer_matches_serial_bitwise() -> None:
    specs = [{"tokens": 16}, {"tokens": 16}, {"tokens": 16}]
    serial_out, participants, corrections, *_ = _run_both(0, specs)
    for (out, s_session), (state, _handoff), correction in zip(
        serial_out, participants, corrections, strict=True
    ):
        q, k, v, residual, attn_output, _meta = out
        _assert_bitwise(correction.q, q, "q")
        _assert_bitwise(correction.k, k, "k")
        _assert_bitwise(correction.v, v, "v")
        _assert_bitwise(correction.residual, residual, "residual")
        assert correction.attn_output.shape == attn_output.shape
        assert state.session.recorder.collective_rope == 1
        assert state.session.recorder.collective_diff == 0
        assert state.session.recorder.serial_rope == 0
        assert s_session.recorder.serial_rope == 1
        _assert_bitwise(
            state.session.metadata.positions,
            s_session.metadata.positions,
            "positions",
        )


def test_check_layer_selection_matches_serial_bitwise() -> None:
    specs = [{"tokens": 16}, {"tokens": 24}, {"tokens": 8}]
    serial_out, participants, corrections, serial_old, old = _run_both(1, specs)
    for i, ((out, s_session), (state, handoff), correction) in enumerate(
        zip(serial_out, participants, corrections, strict=True)
    ):
        q, k, v, residual, attn_output, _meta = out
        _assert_bitwise(correction.q, q, "q")
        _assert_bitwise(correction.k, k, "k(old merged)")
        _assert_bitwise(correction.v, v, "v(old merged)")
        _assert_bitwise(correction.residual, residual, "residual")
        assert torch.equal(
            state.session.metadata.imp_indices, s_session.metadata.imp_indices
        )
        _assert_bitwise(
            state.session.metadata.positions,
            s_session.metadata.positions,
            "positions",
        )
        # cached buffers merged identically
        _assert_bitwise(old[i][1][0], serial_old[i][1][0], "old_k buffer")
        _assert_bitwise(old[i][1][1], serial_old[i][1][1], "old_v buffer")
        assert len(handoff.attn_metadata.updates) == 1
        assert state.session.recorder.collective_diff == 1


def test_explicit_gaps_join_selection_and_repair() -> None:
    specs = [
        {"tokens": 16, "gap": (0, 5), "explicit": True},
        {"tokens": 16, "gap": (15,), "explicit": True},
    ]
    for layer_id in (0, 1):
        serial_out, participants, corrections, serial_old, old = _run_both(
            layer_id, specs
        )
        for i, ((out, _s), (_state, _h), correction) in enumerate(
            zip(serial_out, participants, corrections, strict=True)
        ):
            q, k, v, residual, _ao, _meta = out
            _assert_bitwise(correction.k, k, f"k layer{layer_id}")
            _assert_bitwise(correction.v, v, f"v layer{layer_id}")
            _assert_bitwise(old[i][layer_id][0], serial_old[i][layer_id][0], "old_k")
            _assert_bitwise(old[i][layer_id][1], serial_old[i][layer_id][1], "old_v")


def test_layer_mismatch_poisons_participants() -> None:
    old = {0: _old_kv(0, 3, 8), 1: _old_kv(1, 3, 8)}
    blender = _make_blender(old)
    participants = []
    for i, layer_id in enumerate((0, 1)):
        session = _member_session(8)
        session.load_state.member_index = i
        q, k, v, residual = _fresh_inputs(i, 8)
        participants.append(
            (
                SimpleNamespace(session=session, error=None),
                LMCQKVHandoff(
                    layer_id=layer_id,
                    q=q,
                    k=k,
                    v=v,
                    residual=residual,
                    attn_output=None,
                    attn_metadata=_AttnMetadata(),
                ),
            )
        )
    assert blender._apply_collective_qkv(participants) is None
    assert all(state.error is not None for state, _ in participants)


def test_empty_participants_is_a_noop() -> None:
    blender = _make_blender({0: _old_kv(0, 3, 8)})
    assert blender._apply_collective_qkv([]) is None
