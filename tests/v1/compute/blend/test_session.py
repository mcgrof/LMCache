# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Generator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, cast
import pickle

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.compute.attention.metadata import LMCTritonSparseMetadata
from lmcache.v1.compute.blend import utils as blend_utils
from lmcache.v1.compute.blend.blender import (
    LMCBlender,
    LMCBlendGroupOutcome,
    LMCBlendGroupRequest,
    LMCBlendOutcome,
    LMCBlendSession,
)
from lmcache.v1.compute.blend.layer_source import UnsettledLayerSourceError
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMBufferLayerwiseGPUConnector,
)
from lmcache.v1.gpu_connector.load_state import LayerwiseLoadState
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan
import lmcache.v1.cache_engine as cache_engine_module


class InjectedLayerFailure(RuntimeError):
    """Failure raised by the fake layerwise retrieval path."""


@dataclass
class _FakeBuffer:
    request_id: str
    payload: tuple[Any, ...] = ()
    release_count: int = 0
    fail_release: bool = False

    def ref_count_down(self) -> None:
        self.release_count += 1
        if self.fail_release:
            raise RuntimeError("injected buffer release failure")


@dataclass
class _OperationRecorder:
    rope_operations: int = 0
    diff_operations: int = 0

    def record_serial_operations(
        self,
        *,
        rope_operations: int = 0,
        diff_operations: int = 0,
    ) -> None:
        self.rope_operations += rope_operations
        self.diff_operations += diff_operations


class _FakeCacheEngine:
    def __init__(
        self,
        num_layers: int,
        failure: Optional[tuple[str, int]] = None,
        fatal_failure: Optional[tuple[str, int]] = None,
        events: Optional[list[str]] = None,
    ) -> None:
        self.num_layers = num_layers
        self.failure = failure
        self.fatal_failure = fatal_failure
        self.events = events if events is not None else []
        self.buffers: dict[str, list[_FakeBuffer]] = {}
        self.generators_closed: set[str] = set()
        self.generator_close_counts: dict[str, int] = {}
        self.retrieve_kwargs: dict[str, dict[str, Any]] = {}

    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        load_state: LayerwiseLoadState,
        gap_pattern: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        del mask
        request_id = str(int(tokens[0].item()))
        self.retrieve_kwargs[request_id] = dict(kwargs)
        buffers = [_FakeBuffer(request_id), _FakeBuffer(request_id)]
        self.buffers[request_id] = buffers

        load_state.begin()
        for buffer in buffers:
            load_state.own_buffer(cast(MemoryObj, buffer))
        load_state.gap_positions = torch.tensor(gap_pattern, dtype=torch.int64)

        try:
            self.events.append(f"prime:{request_id}")
            yield torch.tensor(len(tokens))
            for layer_id in range(self.num_layers):
                buffer = buffers[layer_id % len(buffers)]
                buffer.payload = (request_id, layer_id, gap_pattern)
                load_state.buffer_mapping[layer_id] = cast(MemoryObj, buffer)
                self.events.append(f"retrieve:{request_id}:{layer_id}")
                if self.fatal_failure == (request_id, layer_id):
                    raise KeyboardInterrupt(
                        f"injected fatal retrieval failure for {request_id}:{layer_id}"
                    )
                if self.failure == (request_id, layer_id):
                    raise InjectedLayerFailure(
                        f"injected retrieval failure for {request_id}:{layer_id}"
                    )
                yield None

            load_state.buffer_mapping.clear()
            self.events.append(f"finish:{request_id}")
            yield torch.ones(len(tokens), dtype=torch.bool)
        finally:
            self.generators_closed.add(request_id)
            self.generator_close_counts[request_id] = (
                self.generator_close_counts.get(request_id, 0) + 1
            )
            load_state.release()


class _FakeLayerwiseModel:
    def __init__(self, num_layers: int, events: Optional[list[str]] = None) -> None:
        self.num_layers = num_layers
        self.events = events if events is not None else []
        self.outputs: dict[str, list[tuple[Any, ...]]] = {}
        self.generators_closed: set[str] = set()
        self.generator_close_counts: dict[str, int] = {}
        self.generator_calls = 0
        self.layer_calls = 0

    def compute_layer(
        self, tokens: torch.Tensor, session: LMCBlendSession
    ) -> Generator[None, None, None]:
        self.generator_calls += 1
        return self._compute_layer(tokens, session)

    def _compute_layer(
        self, tokens: torch.Tensor, session: LMCBlendSession
    ) -> Generator[None, None, None]:
        request_id = str(session.request_id)
        token_id = int(tokens[0].item())
        self.outputs.setdefault(request_id, [])
        try:
            for layer_id in range(self.num_layers):
                self.layer_calls += 1
                self.events.append(f"model:{request_id}:{layer_id}")
                if session.metadata.positions is None:
                    session.metadata.positions = torch.tensor([token_id])

                buffer = cast(
                    _FakeBuffer, session.load_state.buffer_mapping.pop(layer_id)
                )
                self.outputs[request_id].append(
                    (
                        buffer.payload,
                        int(session.metadata.positions[0].item()),
                    )
                )
                session.metadata.positions = session.metadata.positions + 1
                session.record_serial_operations(
                    rope_operations=1,
                    diff_operations=int(layer_id == 1),
                )
                yield
        finally:
            self.generators_closed.add(request_id)
            self.generator_close_counts[request_id] = (
                self.generator_close_counts.get(request_id, 0) + 1
            )


class _FakeBlender(LMCBlender):
    def __init__(
        self,
        num_layers: int = 3,
        failure: Optional[tuple[str, int]] = None,
        fatal_failure: Optional[tuple[str, int]] = None,
        close_failure: Optional[str] = None,
    ) -> None:
        self.num_layers = num_layers
        self.close_failure = close_failure
        self.events: list[str] = []
        self.cache_engine = _FakeCacheEngine(
            num_layers,
            failure,
            fatal_failure,
            self.events,
        )
        self.layerwise_model = _FakeLayerwiseModel(num_layers, self.events)
        self.sessions: list[LMCBlendSession] = []
        self.session_close_calls: dict[str, int] = {}

    def create_session(self, *args: Any, **kwargs: Any) -> LMCBlendSession:
        session = super().create_session(*args, **kwargs)
        request_id = str(session.request_id)
        original_close = session.close

        def counted_close() -> None:
            self.session_close_calls[request_id] = (
                self.session_close_calls.get(request_id, 0) + 1
            )
            original_close()
            if self.close_failure == request_id:
                raise InjectedLayerFailure(
                    f"injected session close failure for {request_id}"
                )

        cast(Any, session).close = counted_close
        self.sessions.append(session)
        return session


@dataclass
class _ExplicitLayerSource:
    explicit_spans: ExplicitTokenSpans
    engine: _FakeCacheEngine

    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        load_state: LayerwiseLoadState,
    ) -> Generator[torch.Tensor | None, None, None]:
        return self.engine.retrieve_layer(tokens, mask, load_state=load_state)


def test_session_consumes_bound_source_without_legacy_retrieval() -> None:
    blender = _FakeBlender()
    spans = ExplicitTokenSpans(3, TokenSpan(0, 3), (TokenSpan(0, 3),))
    source = _ExplicitLayerSource(spans, _FakeCacheEngine(3))
    with blender.create_session(
        torch.tensor([1, 2, 3]),
        request_id="1",
        layer_source=source,
        explicit_spans=spans,
    ) as session:
        result = session.run()
        assert torch.all(result)
        assert session.requires_explicit_gap_recompute
        assert len(blender.layerwise_model.outputs["1"]) == 3
    assert not blender.cache_engine.retrieve_kwargs
    assert source.engine.generators_closed == {"1"}
    assert all(buffer.release_count == 1 for buffer in source.engine.buffers["1"])


@pytest.mark.parametrize("mismatch", ["absent", "different-window"])
def test_session_refuses_mismatched_source_before_retrieval(mismatch: str) -> None:
    blender = _FakeBlender()
    spans = ExplicitTokenSpans(3, TokenSpan(0, 3), (TokenSpan(0, 3),))
    source = _ExplicitLayerSource(spans, _FakeCacheEngine(3))
    supplied = (
        None
        if mismatch == "absent"
        else ExplicitTokenSpans(3, TokenSpan(0, 2), (TokenSpan(0, 2),))
    )
    with pytest.raises(ValueError, match="source spans"):
        blender.create_session(
            torch.tensor([1, 2, 3]), layer_source=source, explicit_spans=supplied
        )
    assert not source.engine.retrieve_kwargs
    assert not blender.cache_engine.retrieve_kwargs


def test_bound_source_failure_closes_its_request_owned_buffers() -> None:
    blender = _FakeBlender()
    spans = ExplicitTokenSpans(3, TokenSpan(0, 3), (TokenSpan(0, 3),))
    source = _ExplicitLayerSource(spans, _FakeCacheEngine(3, failure=("1", 1)))
    with pytest.raises(InjectedLayerFailure):
        with blender.create_session(
            torch.tensor([1, 2, 3]),
            request_id="1",
            layer_source=source,
            explicit_spans=spans,
        ) as session:
            session.run()
    assert source.engine.generators_closed == {"1"}
    assert all(buffer.release_count == 1 for buffer in source.engine.buffers["1"])
    assert not blender.cache_engine.retrieve_kwargs


@dataclass
class _UnsettledLayerSource(_ExplicitLayerSource):
    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        load_state: LayerwiseLoadState,
    ) -> Generator[torch.Tensor | None, None, None]:
        try:
            yield from self.engine.retrieve_layer(tokens, mask, load_state=load_state)
        finally:
            raise UnsettledLayerSourceError("synthetic unresolved drain", self.drain)

    def drain(self) -> None:
        raise RuntimeError("synthetic cleanup still pending")


@pytest.mark.parametrize("failure", ["retrieval", "caller", "close"])
def test_unsettled_cleanup_escapes_session_error_containment(failure: str) -> None:
    blender = _FakeBlender()
    spans = ExplicitTokenSpans(3, TokenSpan(0, 3), (TokenSpan(0, 3),))
    source = _UnsettledLayerSource(
        spans, _FakeCacheEngine(3, failure=("1", 1) if failure == "retrieval" else None)
    )
    with pytest.raises(UnsettledLayerSourceError) as raised:
        with blender.create_session(
            torch.tensor([1, 2, 3]),
            request_id="1",
            layer_source=source,
            explicit_spans=spans,
        ) as session:
            if failure == "caller":
                session.prime()
                raise ValueError("synthetic caller failure")
            session.run()
    with pytest.raises(RuntimeError, match="still pending"):
        raised.value.drain()


def test_unqualified_group_source_refuses_before_any_session() -> None:
    blender = _FakeBlender()
    spans = ExplicitTokenSpans(3, TokenSpan(0, 3), (TokenSpan(0, 3),))
    source = _ExplicitLayerSource(spans, _FakeCacheEngine(3))
    request = LMCBlendGroupRequest(
        request_id="1",
        tokens=torch.tensor([1, 2, 3]),
        retrieve_kwargs={"layer_source": source, "explicit_spans": spans},
    )
    with pytest.raises(NotImplementedError, match="serial sessions"):
        blender.blend_group((request,))
    assert not blender.sessions and not source.engine.retrieve_kwargs


def test_serialized_cleanup_error_keeps_the_actual_owner_local() -> None:
    attempts = 0

    def drain() -> None:
        nonlocal attempts
        attempts += 1

    original = UnsettledLayerSourceError("synthetic unresolved device work", drain)
    transported = pickle.loads(pickle.dumps(original))
    assert type(transported) is RuntimeError
    assert str(transported) == str(original)
    assert attempts == 0
    original.drain()
    assert attempts == 1


def test_session_marks_only_explicit_spans_for_gap_recompute() -> None:
    blender = _FakeBlender(num_layers=1)
    legacy = blender.create_session(torch.tensor([1]))
    explicit = blender.create_session(
        torch.tensor([2]),
        explicit_spans=object(),
    )
    try:
        assert legacy.requires_explicit_gap_recompute is False
        assert explicit.requires_explicit_gap_recompute is True
    finally:
        legacy.close()
        explicit.close()


def test_gap_positions_are_mandatory_beyond_recompute_budget() -> None:
    """Every zero-filled miss is recomputed before cached KV publication."""
    token_count = 256
    gap_count = 39
    configured_topk = int(token_count * 0.15)
    assert configured_topk == 38

    old_k = torch.zeros((token_count, 1), dtype=torch.float32)
    old_v = torch.zeros((token_count, 1), dtype=torch.float32)
    current_k = torch.zeros_like(old_k)
    current_k[:configured_topk] = 2.0
    current_k[configured_topk] = 1.0
    current_v = torch.arange(token_count, dtype=torch.float32).reshape(-1, 1)

    blender = LMCBlender.__new__(LMCBlender)
    blender.gpu_connector = SimpleNamespace(
        get_kv=lambda layer_id, load_state: (old_k, old_v)
    )

    def rotary(positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        del positions
        return q, k

    layer = SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=rotary))
    blender.layerwise_model = SimpleNamespace(
        vllm_model=SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    )
    cast(Any, blender).common_metadata = SimpleNamespace(
        check_layers=[0],
        recomp_ratios=[0.15],
    )

    load_state = LayerwiseLoadState()
    load_state.gap_positions = torch.arange(gap_count, dtype=torch.int64)
    metadata = SimpleNamespace(positions=None, imp_indices=None)
    operation_recorder = _OperationRecorder()
    session = SimpleNamespace(
        load_state=load_state,
        metadata=metadata,
        requires_explicit_gap_recompute=True,
        record_serial_operations=operation_recorder.record_serial_operations,
    )

    class _AttnMetadata:
        def __init__(self) -> None:
            self.indices: Optional[torch.Tensor] = None

        def update_from_top_indices(self, top_indices: torch.Tensor) -> None:
            self.indices = top_indices

    attn_metadata = _AttnMetadata()
    q = torch.ones((token_count, 1), dtype=torch.float32)
    residual = torch.ones_like(q)
    attn_output = torch.empty_like(q)

    result = blender.process_qkv(
        q,
        current_k,
        current_v,
        residual,
        0,
        attn_output,
        cast(Any, attn_metadata),
        cast(Any, session),
    )

    selected = metadata.imp_indices
    assert selected is not None
    assert torch.equal(selected, torch.arange(gap_count, dtype=torch.int64))
    assert attn_metadata.indices is selected
    assert len(result[0]) == gap_count
    assert len(result[4]) == gap_count
    assert result[1][configured_topk].item() == 1.0
    assert result[2][configured_topk].item() == float(configured_topk)
    assert operation_recorder.rope_operations == 1
    assert operation_recorder.diff_operations == 1


def test_precheck_layer_repairs_gap_kv_before_connector_writeback() -> None:
    """Full-compute layers repair zero gaps in the published old-KV buffer."""
    token_count = 8
    gap_positions = torch.tensor([1, 4], dtype=torch.int64)
    old_k = torch.zeros((token_count, 1), dtype=torch.float32)
    old_v = torch.zeros((token_count, 1), dtype=torch.float32)
    current_k = torch.arange(1, token_count + 1, dtype=torch.float32).reshape(-1, 1)
    current_v = current_k + 100.0

    blender = LMCBlender.__new__(LMCBlender)
    blender.gpu_connector = SimpleNamespace(
        get_kv=lambda layer_id, load_state: (old_k, old_v)
    )

    def rotary(positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        del positions
        return q, k

    layer = SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=rotary))
    blender.layerwise_model = SimpleNamespace(
        vllm_model=SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    )
    cast(Any, blender).common_metadata = SimpleNamespace(
        check_layers=[1],
        recomp_ratios=[0.15],
    )
    load_state = LayerwiseLoadState(gap_positions=gap_positions)
    metadata = SimpleNamespace(positions=None, imp_indices=None)
    operation_recorder = _OperationRecorder()
    session = SimpleNamespace(
        load_state=load_state,
        metadata=metadata,
        requires_explicit_gap_recompute=True,
        record_serial_operations=operation_recorder.record_serial_operations,
    )
    attn_metadata = SimpleNamespace(update_from_top_indices=lambda indices: None)
    q = torch.ones_like(current_k)
    residual = torch.ones_like(current_k)

    result = blender.process_qkv(
        q,
        current_k,
        current_v,
        residual,
        0,
        None,
        cast(Any, attn_metadata),
        cast(Any, session),
    )

    assert metadata.imp_indices is None
    assert torch.equal(result[1], current_k)
    assert torch.equal(result[2], current_v)
    assert torch.equal(old_k[gap_positions], current_k[gap_positions])
    assert torch.equal(old_v[gap_positions], current_v[gap_positions])
    retained_positions = torch.tensor([0, 2, 3, 5, 6, 7])
    assert torch.count_nonzero(old_k[retained_positions]).item() == 0
    assert torch.count_nonzero(old_v[retained_positions]).item() == 0
    assert operation_recorder.rope_operations == 1
    assert operation_recorder.diff_operations == 0


def test_two_layer_explicit_gap_recompute_never_publishes_zeroed_kv() -> None:
    """Pre-check and selection layers both repair every explicit miss."""
    token_count = 256
    gap_positions = torch.arange(39, dtype=torch.int64)
    quality_positions = torch.arange(39, 77, dtype=torch.int64)

    old_k_by_layer = [
        torch.full((token_count, 1), 4.0),
        torch.full((token_count, 1), 7.0),
    ]
    old_v_by_layer = [
        torch.full((token_count, 1), 5.0),
        torch.full((token_count, 1), 9.0),
    ]
    for old_k, old_v in zip(old_k_by_layer, old_v_by_layer, strict=True):
        old_k[gap_positions] = 0
        old_v[gap_positions] = 0

    current_k0 = torch.arange(1, token_count + 1, dtype=torch.float32).reshape(-1, 1)
    current_v0 = current_k0 + 100.0
    current_k1 = old_k_by_layer[1].clone()
    current_v1 = old_v_by_layer[1].clone()
    current_k1[gap_positions] = gap_positions.reshape(-1, 1) + 1.0
    current_v1[gap_positions] = gap_positions.reshape(-1, 1) + 101.0
    current_k1[quality_positions] = quality_positions.reshape(-1, 1) + 1000.0
    current_v1[quality_positions] = quality_positions.reshape(-1, 1) + 2000.0

    blender = LMCBlender.__new__(LMCBlender)
    blender.gpu_connector = SimpleNamespace(
        get_kv=lambda layer_id, load_state: (
            old_k_by_layer[layer_id],
            old_v_by_layer[layer_id],
        )
    )

    def rotary(positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        del positions
        return q, k

    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=rotary)) for _ in range(2)
    ]
    blender.layerwise_model = SimpleNamespace(
        vllm_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )
    cast(Any, blender).common_metadata = SimpleNamespace(
        check_layers=[1],
        recomp_ratios=[0.15],
    )
    load_state = LayerwiseLoadState(gap_positions=gap_positions)
    metadata = SimpleNamespace(positions=None, imp_indices=None)
    operation_recorder = _OperationRecorder()
    session = SimpleNamespace(
        load_state=load_state,
        metadata=metadata,
        requires_explicit_gap_recompute=True,
        record_serial_operations=operation_recorder.record_serial_operations,
    )

    class _AttnMetadata:
        def __init__(self) -> None:
            self.indices: Optional[torch.Tensor] = None

        def update_from_top_indices(self, top_indices: torch.Tensor) -> None:
            self.indices = top_indices

    attn_metadata = _AttnMetadata()
    q = torch.ones((token_count, 1), dtype=torch.float32)
    residual = torch.ones_like(q)

    blender.process_qkv(
        q,
        current_k0,
        current_v0,
        residual,
        0,
        None,
        cast(Any, attn_metadata),
        cast(Any, session),
    )
    assert torch.equal(old_k_by_layer[0][gap_positions], current_k0[gap_positions])
    assert torch.equal(old_v_by_layer[0][gap_positions], current_v0[gap_positions])

    result = blender.process_qkv(
        q,
        current_k1,
        current_v1,
        residual,
        1,
        None,
        cast(Any, attn_metadata),
        cast(Any, session),
    )

    assert attn_metadata.indices is not None
    assert torch.equal(attn_metadata.indices, torch.arange(77))
    assert torch.equal(result[1][gap_positions], current_k1[gap_positions])
    assert torch.equal(result[2][gap_positions], current_v1[gap_positions])
    assert torch.count_nonzero(result[1][gap_positions]).item() == len(gap_positions)
    assert torch.count_nonzero(result[2][gap_positions]).item() == len(gap_positions)
    assert operation_recorder.rope_operations == 2
    assert operation_recorder.diff_operations == 1


@pytest.mark.parametrize(
    ("selection_size", "expected_indices", "expected_indptr"),
    [
        (1, [0], [0, 1]),
        (30, [0], [0, 1]),
        (31, [0], [0, 1]),
        (32, [0], [0, 1]),
        (33, [0, 0, 1], [0, 1, 3]),
        (39, [0, 0, 1], [0, 1, 3]),
        (64, [0, 0, 1], [0, 1, 3]),
        (65, [0, 0, 1, 0, 1, 2], [0, 1, 3, 6]),
    ],
)
def test_triton_sparse_metadata_covers_every_query_row(
    selection_size: int,
    expected_indices: list[int],
    expected_indptr: list[int],
) -> None:
    """Every full or partial compacted-query row gets causal KV blocks."""
    top_indices = torch.arange(selection_size, dtype=torch.int64)
    metadata = LMCTritonSparseMetadata(
        seq_len=256,
        sparse_blk_row_size=32,
        sparse_blk_col_size=32,
    )

    metadata.update_from_top_indices(top_indices)

    assert metadata.query_positions is not top_indices
    assert torch.equal(metadata.query_positions, top_indices)
    assert metadata.block_indices is not None
    assert metadata.block_indptr is not None
    assert torch.equal(
        metadata.block_indices,
        torch.tensor(expected_indices, dtype=torch.int32),
    )
    assert torch.equal(
        metadata.block_indptr,
        torch.tensor(expected_indptr, dtype=torch.int32),
    )


def test_triton_sparse_metadata_covers_single_partial_query_row() -> None:
    """Fewer than 32 selected queries still produce one complete CSR row."""
    top_indices = torch.tensor([7, 91, 200], dtype=torch.int64)
    metadata = LMCTritonSparseMetadata(
        seq_len=256,
        sparse_blk_row_size=32,
        sparse_blk_col_size=32,
    )

    metadata.update_from_top_indices(top_indices)

    assert metadata.query_positions is not top_indices
    assert torch.equal(metadata.query_positions, top_indices)
    assert metadata.block_indices is not None
    assert metadata.block_indptr is not None
    assert torch.equal(metadata.block_indices, torch.arange(7, dtype=torch.int32))
    assert torch.equal(metadata.block_indptr, torch.tensor([0, 7]))

    top_indices.fill_(0)
    assert torch.equal(
        metadata.query_positions,
        torch.tensor([7, 91, 200], dtype=torch.int64),
    )


class _MissCacheEngine:
    """Layerwise miss with either current or legacy-short choreography."""

    def __init__(self, num_layers: int, *, short: bool = False) -> None:
        self.num_layers = num_layers
        self.short = short
        self.generators_closed = 0

    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        load_state: LayerwiseLoadState,
        **kwargs: Any,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        del mask, kwargs
        load_state.begin()
        try:
            # The legacy unhealthy path reported a full zero mask at prime and
            # then terminated. The current no-hit path reports a scalar zero
            # and preserves every layer boundary.
            if self.short:
                yield torch.zeros(len(tokens), dtype=torch.bool)
                return
            yield torch.tensor(0)
            for _ in range(self.num_layers):
                yield None
            yield torch.zeros(len(tokens), dtype=torch.bool)
        finally:
            self.generators_closed += 1
            load_state.release()


class _FakeStatsMonitor:
    def __init__(self) -> None:
        self.finished: list[tuple[int, torch.Tensor]] = []

    def on_retrieve_request(self, num_tokens: int) -> int:
        return num_tokens

    def on_retrieve_finished(
        self, monitor_id: int, retrieved_tokens: torch.Tensor
    ) -> None:
        self.finished.append((monitor_id, retrieved_tokens))


class _FailSecondAllocator:
    def __init__(self, first_buffer: _FakeBuffer) -> None:
        self.first_buffer = first_buffer
        self.allocate_calls = 0

    def allocate(self, *args: Any, **kwargs: Any) -> Optional[MemoryObj]:
        del args, kwargs
        self.allocate_calls += 1
        if self.allocate_calls == 1:
            return cast(MemoryObj, self.first_buffer)
        return None

    def memcheck(self) -> bool:
        return self.first_buffer.release_count == 1


def _run_serial(
    request_id: str,
    gap_pattern: tuple[int, ...],
) -> list[tuple[Any, ...]]:
    blender = _FakeBlender()
    blender.blend(
        torch.tensor([int(request_id)]),
        request_id=request_id,
        gap_pattern=gap_pattern,
    )
    return blender.layerwise_model.outputs[request_id]


@pytest.mark.parametrize("reverse_order", [False, True])
def test_two_sessions_match_serial_when_interleaved(reverse_order: bool) -> None:
    """A/B results remain serial-identical in either interleaving order."""
    request_specs = (("11", (1,)), ("22", (0, 2)))
    expected = {
        request_id: _run_serial(request_id, gaps) for request_id, gaps in request_specs
    }

    blender = _FakeBlender()
    sessions = [
        blender.create_session(
            torch.tensor([int(request_id)]),
            request_id=request_id,
            gap_pattern=gaps,
        )
        for request_id, gaps in request_specs
    ]
    if reverse_order:
        sessions.reverse()

    assert sessions[0].metadata is not sessions[1].metadata
    assert sessions[0].load_state is not sessions[1].load_state
    assert (
        sessions[0].load_state.buffer_mapping
        is not sessions[1].load_state.buffer_mapping
    )

    try:
        for session in sessions:
            session.prime()
        for _ in range(blender.num_layers):
            for session in sessions:
                session.step_layer()
        for session in sessions:
            session.finish()
    finally:
        for session in sessions:
            session.close()

    assert blender.layerwise_model.outputs == expected
    assert blender.cache_engine.generators_closed == {"11", "22"}
    assert blender.layerwise_model.generators_closed == {"11", "22"}
    for buffers in blender.cache_engine.buffers.values():
        assert [buffer.release_count for buffer in buffers] == [1, 1]


def test_group_runner_is_stable_layer_major_with_exact_operation_counts() -> None:
    """The reference runs stable rounds without claiming collective work."""
    blender = _FakeBlender()
    explicit_spans = object()
    result = blender.blend_group(
        (
            LMCBlendGroupRequest(
                request_id="11",
                generation=7,
                tokens=torch.tensor([11]),
                retrieve_kwargs={
                    "explicit_spans": explicit_spans,
                    "gap_pattern": (1,),
                },
            ),
            LMCBlendGroupRequest(
                request_id="22",
                generation=8,
                tokens=torch.tensor([22]),
                retrieve_kwargs={"gap_pattern": (0, 2)},
            ),
        )
    )

    assert [member.request_id for member in result.members] == ["11", "22"]
    assert [member.generation for member in result.members] == [7, 8]
    assert [member.outcome for member in result.members] == [
        LMCBlendGroupOutcome.RECOVERED,
        LMCBlendGroupOutcome.RECOVERED,
    ]
    assert result.failed_members == ()
    assert result.failed_request_ids == ()
    for member in result.members:
        assert member.error is None
        assert member.retrieval_mask is not None
        assert torch.equal(member.retrieval_mask, torch.ones(1, dtype=torch.bool))
        assert member.operation_counts.serial_rope_operations == 3
        assert member.operation_counts.serial_diff_operations == 1
        assert member.operation_counts.collective_rope_operations == 0
        assert member.operation_counts.collective_diff_operations == 0

    assert blender.events == [
        "prime:11",
        "prime:22",
        "retrieve:11:0",
        "model:11:0",
        "retrieve:22:0",
        "model:22:0",
        "retrieve:11:1",
        "model:11:1",
        "retrieve:22:1",
        "model:22:1",
        "retrieve:11:2",
        "model:11:2",
        "retrieve:22:2",
        "model:22:2",
        "finish:11",
        "finish:22",
    ]
    assert (
        blender.cache_engine.retrieve_kwargs["11"]["explicit_spans"] is explicit_spans
    )
    assert blender.session_close_calls == {"11": 1, "22": 1}
    assert all(session.is_closed for session in blender.sessions)
    assert blender.cache_engine.generator_close_counts == {"11": 1, "22": 1}
    assert blender.layerwise_model.generator_close_counts == {"11": 1, "22": 1}
    for buffers in blender.cache_engine.buffers.values():
        assert [buffer.release_count for buffer in buffers] == [1, 1]


def test_group_runner_isolates_member_failure_and_finishes_siblings() -> None:
    """An ordinary member failure is returned for invalidation, not raised."""
    expected_a = _run_serial("41", (2,))
    expected_c = _run_serial("43", (0, 2))
    blender = _FakeBlender(failure=("42", 1))

    result = blender.blend_group(
        (
            LMCBlendGroupRequest(
                request_id="41",
                tokens=torch.tensor([41]),
                retrieve_kwargs={"gap_pattern": (2,)},
            ),
            LMCBlendGroupRequest(
                request_id="42",
                tokens=torch.tensor([42]),
                retrieve_kwargs={"gap_pattern": (0, 1)},
            ),
            LMCBlendGroupRequest(
                request_id="43",
                tokens=torch.tensor([43]),
                retrieve_kwargs={"gap_pattern": (0, 2)},
            ),
        )
    )

    assert [member.outcome for member in result.members] == [
        LMCBlendGroupOutcome.RECOVERED,
        LMCBlendGroupOutcome.FAILED,
        LMCBlendGroupOutcome.RECOVERED,
    ]
    assert result.failed_request_ids == ("42",)
    failed = result.failed_members[0]
    assert isinstance(failed.error, InjectedLayerFailure)
    assert failed.retrieval_mask is None
    assert failed.operation_counts.serial_rope_operations == 1
    assert failed.operation_counts.serial_diff_operations == 0
    assert failed.operation_counts.collective_rope_operations == 0
    assert failed.operation_counts.collective_diff_operations == 0
    assert blender.layerwise_model.outputs["41"] == expected_a
    assert blender.layerwise_model.outputs["43"] == expected_c
    assert "model:42:1" not in blender.events
    assert "retrieve:42:2" not in blender.events
    assert blender.session_close_calls == {"41": 1, "42": 1, "43": 1}
    assert all(session.is_closed for session in blender.sessions)
    assert blender.cache_engine.generator_close_counts == {
        "41": 1,
        "42": 1,
        "43": 1,
    }
    assert blender.layerwise_model.generator_close_counts == {
        "41": 1,
        "42": 1,
        "43": 1,
    }
    for buffers in blender.cache_engine.buffers.values():
        assert [buffer.release_count for buffer in buffers] == [1, 1]


def test_group_runner_surfaces_member_cleanup_failure_for_invalidation() -> None:
    """A close failure invalidates only its member after siblings finish."""
    blender = _FakeBlender(close_failure="82")

    result = blender.blend_group(
        (
            LMCBlendGroupRequest(request_id="81", tokens=torch.tensor([81])),
            LMCBlendGroupRequest(request_id="82", tokens=torch.tensor([82])),
            LMCBlendGroupRequest(request_id="83", tokens=torch.tensor([83])),
        )
    )

    assert [member.outcome for member in result.members] == [
        LMCBlendGroupOutcome.RECOVERED,
        LMCBlendGroupOutcome.FAILED,
        LMCBlendGroupOutcome.RECOVERED,
    ]
    assert result.failed_request_ids == ("82",)
    assert isinstance(result.failed_members[0].error, InjectedLayerFailure)
    assert result.failed_members[0].retrieval_mask is None
    assert blender.session_close_calls == {"81": 1, "82": 1, "83": 1}
    assert all(session.is_closed for session in blender.sessions)
    assert blender.cache_engine.generator_close_counts == {
        "81": 1,
        "82": 1,
        "83": 1,
    }
    for buffers in blender.cache_engine.buffers.values():
        assert [buffer.release_count for buffer in buffers] == [1, 1]


def test_group_runner_reports_misses_without_model_operations() -> None:
    """Miss members preserve lockstep choreography and report zero work."""
    blender = _FakeBlender(num_layers=2)
    miss_engine = _MissCacheEngine(blender.num_layers)
    blender.cache_engine = miss_engine

    result = blender.blend_group(
        (
            LMCBlendGroupRequest(request_id="51", tokens=torch.tensor([51])),
            LMCBlendGroupRequest(request_id="52", tokens=torch.tensor([52])),
        )
    )

    assert [member.outcome for member in result.members] == [
        LMCBlendGroupOutcome.MISS,
        LMCBlendGroupOutcome.MISS,
    ]
    for member in result.members:
        assert member.retrieval_mask is not None
        assert not bool(torch.any(member.retrieval_mask).item())
        assert member.operation_counts.serial_rope_operations == 0
        assert member.operation_counts.serial_diff_operations == 0
        assert member.operation_counts.collective_rope_operations == 0
        assert member.operation_counts.collective_diff_operations == 0
    assert blender.layerwise_model.generator_calls == 0
    assert blender.session_close_calls == {"51": 1, "52": 1}
    assert all(session.is_closed for session in blender.sessions)
    assert miss_engine.generators_closed == 2


def test_group_preflight_finishes_before_any_session_creation() -> None:
    """A later structural error cannot leave an earlier created session."""
    blender = _FakeBlender()
    requests = (
        LMCBlendGroupRequest(request_id="61", tokens=torch.tensor([61])),
        LMCBlendGroupRequest(
            request_id="62",
            tokens=torch.tensor([62]),
            retrieve_kwargs={"load_state": object()},
        ),
    )

    with pytest.raises(ValueError, match="load_state"):
        blender.blend_group(requests)

    assert blender.sessions == []
    assert blender.events == []
    assert blender.cache_engine.buffers == {}


def test_group_runner_propagates_base_exception_after_exhaustive_close() -> None:
    """Process-level failures are not isolated, but every session is closed."""
    blender = _FakeBlender(fatal_failure=("72", 1))
    requests = tuple(
        LMCBlendGroupRequest(
            request_id=request_id,
            tokens=torch.tensor([int(request_id)]),
        )
        for request_id in ("71", "72", "73")
    )

    with pytest.raises(KeyboardInterrupt, match="72:1"):
        blender.blend_group(requests)

    assert blender.session_close_calls == {"71": 1, "72": 1, "73": 1}
    assert all(session.is_closed for session in blender.sessions)
    assert blender.cache_engine.generator_close_counts == {
        "71": 1,
        "72": 1,
        "73": 1,
    }
    for buffers in blender.cache_engine.buffers.values():
        assert [buffer.release_count for buffer in buffers] == [1, 1]


def test_legacy_blend_wrapper_matches_explicit_session_run() -> None:
    """The legacy ``blend`` wrapper and explicit serial session share one path."""
    wrapper = _FakeBlender()
    wrapper_result = wrapper.blend(
        torch.tensor([31]),
        request_id="31",
        gap_pattern=(0, 3),
    )
    assert torch.equal(wrapper_result, torch.ones(1, dtype=torch.bool))

    explicit = _FakeBlender()
    with explicit.create_session(
        torch.tensor([31]),
        request_id="31",
        gap_pattern=(0, 3),
    ) as session:
        result = session.run()
        assert result is not None
        assert session.outcome is LMCBlendOutcome.RECOVERED
        assert torch.equal(result, torch.ones(1, dtype=torch.bool))

    assert explicit.layerwise_model.outputs == wrapper.layerwise_model.outputs
    assert explicit.cache_engine.generators_closed == {"31"}
    assert explicit.layerwise_model.generators_closed == {"31"}


def test_explicit_miss_skips_model_and_returns_zero_mask() -> None:
    """A zero-hit prime drains retrieval without touching model state."""
    blender = _FakeBlender(num_layers=3)
    miss_engine = _MissCacheEngine(blender.num_layers)
    blender.cache_engine = miss_engine

    with blender.create_session(torch.tensor([33, 34]), request_id="33") as session:
        result = session.run()
        assert session.outcome is LMCBlendOutcome.MISS
        assert session.completed_layers == blender.num_layers
        assert torch.equal(result, torch.zeros(2, dtype=torch.bool))
        session.close()
        session.close()

    assert blender.layerwise_model.layer_calls == 0
    assert blender.layerwise_model.generator_calls == 0
    assert blender.layerwise_model.outputs == {}
    assert miss_engine.generators_closed == 1
    assert session.is_closed
    assert session.load_state.is_released

    wrapper = _FakeBlender(num_layers=3)
    wrapper_miss_engine = _MissCacheEngine(wrapper.num_layers)
    wrapper.cache_engine = wrapper_miss_engine
    wrapper_result = wrapper.blend(torch.tensor([36, 37]), request_id="36")
    assert torch.equal(wrapper_result, torch.zeros(2, dtype=torch.bool))
    assert wrapper.layerwise_model.layer_calls == 0
    assert wrapper.layerwise_model.generator_calls == 0
    assert wrapper_miss_engine.generators_closed == 1


def test_legacy_short_miss_preserves_yield_count_without_model_calls() -> None:
    """An old unhealthy retriever remains bounded at prime + N + finish."""
    blender = _FakeBlender(num_layers=4)
    miss_engine = _MissCacheEngine(blender.num_layers, short=True)
    blender.cache_engine = miss_engine

    yields = list(blender.blend_layer(torch.tensor([35]), request_id="35"))

    assert len(yields) == blender.num_layers + 2
    assert blender.layerwise_model.layer_calls == 0
    assert blender.layerwise_model.generator_calls == 0
    assert blender.layerwise_model.outputs == {}
    assert miss_engine.generators_closed == 1


def test_legacy_blend_layer_preserves_yield_count() -> None:
    """The exposed legacy generator still yields prime + N layers + finish."""
    blender = _FakeBlender(num_layers=4)
    layerwise = blender.blend_layer(
        torch.tensor([32]),
        request_id="32",
        gap_pattern=(1,),
    )

    assert len(list(layerwise)) == blender.num_layers + 2
    assert blender.cache_engine.generators_closed == {"32"}
    assert blender.layerwise_model.generators_closed == {"32"}


def test_failed_session_cleans_up_without_corrupting_sibling() -> None:
    """A failed B session releases its state while A remains serial-identical."""
    expected_a = _run_serial("41", (2,))
    blender = _FakeBlender(failure=("42", 1))
    session_a = blender.create_session(
        torch.tensor([41]), request_id="41", gap_pattern=(2,)
    )
    session_b = blender.create_session(
        torch.tensor([42]), request_id="42", gap_pattern=(0, 1)
    )

    try:
        session_a.step_layer()
        session_b.step_layer()
        session_a.step_layer()
        with pytest.raises(InjectedLayerFailure, match="42:1"):
            session_b.step_layer()

        assert session_b.is_closed
        assert session_b.load_state.is_released
        assert session_b.load_state.owned_buffer_count == 0
        assert session_b.load_state.buffer_mapping == {}

        session_a.step_layer()
        session_a.finish()
    finally:
        session_a.close()
        session_b.close()

    assert blender.layerwise_model.outputs["41"] == expected_a
    assert blender.cache_engine.generators_closed == {"41", "42"}
    assert blender.layerwise_model.generators_closed == {"41", "42"}
    for buffers in blender.cache_engine.buffers.values():
        assert [buffer.release_count for buffer in buffers] == [1, 1]


def test_load_state_attempts_every_release_and_is_idempotent() -> None:
    """One release error does not strand the other temporary buffer."""
    state = LayerwiseLoadState()
    good = _FakeBuffer("good")
    bad = _FakeBuffer("bad", fail_release=True)
    state.begin()
    state.own_buffer(cast(MemoryObj, good))
    state.own_buffer(cast(MemoryObj, bad))
    state.buffer_mapping[0] = cast(MemoryObj, good)

    with pytest.raises(RuntimeError, match="Failed to release"):
        state.release()

    assert state.buffer_mapping == {}
    assert state.owned_buffer_count == 0
    assert state.is_released
    assert good.release_count == 1
    assert bad.release_count == 1
    state.release()
    assert good.release_count == 1
    assert bad.release_count == 1


def test_no_hit_layerwise_retrieval_preserves_yields_and_needs_no_consumer() -> None:
    """A no-hit retrieval yields N+2 times without touching an unbound consumer."""

    class EmptyTokenDatabase:
        def process_tokens(self, **kwargs: Any):
            del kwargs
            return iter(())

    class UnexpectedGPUConnector:
        def batched_to_gpu(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise AssertionError("no-hit retrieval must not create a GPU consumer")

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine_state = cast(Any, engine)
    stats_monitor = _FakeStatsMonitor()
    engine_state.num_layers = 3
    engine_state.storage_manager = object()
    engine_state.gpu_connector = UnexpectedGPUConnector()
    engine_state.token_database = EmptyTokenDatabase()
    engine_state.stats_monitor = stats_monitor
    engine_state.is_healthy = lambda: True
    engine_state._get_req_id = lambda kwargs: "no-hit"
    engine_state._is_passive = lambda: True

    yields = list(engine.retrieve_layer(torch.tensor([1, 2, 3])))

    assert torch.equal(yields[0], torch.tensor(0))
    assert yields[1:-1] == [None] * engine.num_layers
    assert torch.equal(yields[-1], torch.zeros(3, dtype=torch.bool))
    assert len(yields) == engine.num_layers + 2
    assert len(stats_monitor.finished) == 1


def test_unhealthy_layerwise_retrieval_preserves_full_choreography() -> None:
    """Health-check misses report zero and retain the legacy N+2 contract."""
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine_state = cast(Any, engine)
    engine_state.num_layers = 3
    engine_state.is_healthy = lambda: False

    yields = list(engine.retrieve_layer(torch.tensor([1, 2])))

    assert torch.equal(yields[0], torch.zeros(2, dtype=torch.bool))
    assert yields[1:-1] == [None] * engine.num_layers
    assert torch.equal(yields[-1], torch.zeros(2, dtype=torch.bool))
    assert len(yields) == engine.num_layers + 2


def test_second_buffer_allocation_failure_releases_first_buffer() -> None:
    """A half-complete double-buffer allocation restores allocator state."""
    connector = VLLMBufferLayerwiseGPUConnector.__new__(VLLMBufferLayerwiseGPUConnector)
    connector_state = cast(Any, connector)
    first_buffer = _FakeBuffer("partial-allocation")
    allocator = _FailSecondAllocator(first_buffer)
    connector_state.cache_positions = False
    connector_state.fused_rotary_emb = lambda *args: args[-1]
    connector_state.dtype = torch.float32
    connector_state.kvcaches = [torch.empty(1)]
    connector_state.gpu_buffer_allocator = allocator
    connector_state.initialize_kvcaches_ptr = lambda **kwargs: None
    connector_state._lazy_initialize_buffer = lambda kv_caches: None
    connector_state.get_shape = lambda num_tokens: torch.Size((2, num_tokens, 1))
    state = LayerwiseLoadState()

    consumer = connector.batched_to_gpu(
        [0],
        [1],
        load_state=state,
        slot_mapping=torch.tensor([0]),
    )
    with pytest.raises(MemoryError, match="load GPU buffer"):
        next(consumer)

    assert allocator.allocate_calls == 2
    assert first_buffer.release_count == 1
    assert allocator.memcheck()
    assert state.is_released
    assert state.owned_buffer_count == 0
    assert state.buffer_mapping == {}


def test_h2d_failure_releases_retrieved_objects(monkeypatch) -> None:
    """Objects returned before an H2D failure are released and unpinned."""

    class FakeMemoryObject:
        def __init__(self) -> None:
            self.release_count = 0
            self.unpin_count = 0
            self._is_pinned = True

        def ref_count_down(self) -> None:
            self.release_count += 1

        @property
        def is_pinned(self) -> bool:
            return self._is_pinned

        def unpin(self) -> None:
            self.unpin_count += 1
            self._is_pinned = False

    class FakeTask:
        def __init__(self, memory_obj: FakeMemoryObject) -> None:
            self.memory_obj = memory_obj

        def result(self) -> list[MemoryObj]:
            return [cast(MemoryObj, self.memory_obj)]

    class FakeStorageManager:
        def __init__(self, memory_obj: FakeMemoryObject) -> None:
            self.memory_obj = memory_obj

        def contains(self, key: CacheEngineKey, locations: Any) -> str:
            del key, locations
            return "LocalCPUBackend"

        def layerwise_batched_get(self, keys: Any, location: str):
            del keys, location
            yield FakeTask(self.memory_obj)

        def layerwise_batched_get_returns_pinned(self, location: str) -> bool:
            del location
            return True

    class OneChunkTokenDatabase:
        def __init__(self, key: CacheEngineKey) -> None:
            self.key = key

        def process_tokens(self, **kwargs: Any):
            del kwargs
            yield 0, 2, self.key

    class FailingGPUConnector:
        def __init__(self) -> None:
            self.closed = False

        def batched_to_gpu(self, starts: Any, ends: Any, **kwargs: Any):
            del starts, ends, kwargs
            try:
                yield
                raise InjectedLayerFailure("injected H2D failure")
            finally:
                self.closed = True

    memory_obj = FakeMemoryObject()
    gpu_connector = FailingGPUConnector()
    key = CacheEngineKey("test", 1, 0, 17, torch.float16)
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine_state = cast(Any, engine)
    engine_state.num_layers = 1
    engine_state.storage_manager = FakeStorageManager(memory_obj)
    engine_state.gpu_connector = gpu_connector
    engine_state.token_database = OneChunkTokenDatabase(key)
    engine_state.stats_monitor = _FakeStatsMonitor()
    engine_state.retrieve_locations = None
    engine_state.is_healthy = lambda: True
    engine_state._get_req_id = lambda kwargs: "h2d-failure"
    engine_state._is_passive = lambda: True
    monkeypatch.setattr(
        cache_engine_module,
        "assert_layerwise_gpu_connector",
        lambda connector: None,
    )

    retriever = engine.retrieve_layer(torch.tensor([1, 2]))
    first_yield = next(retriever)
    assert torch.equal(first_yield, torch.tensor(2))
    with pytest.raises(InjectedLayerFailure, match="H2D"):
        next(retriever)

    assert gpu_connector.closed
    assert memory_obj.release_count == 1
    assert memory_obj.unpin_count == 1


def test_engine_destroy_drops_blender_and_registered_model(monkeypatch) -> None:
    """Destroy/recreate under one engine name cannot reuse stale objects."""

    class FakeBuiltBlender:
        def __init__(self, **kwargs: Any) -> None:
            self.model = kwargs["vllm_model"]
            self.cache_engine = kwargs["cache_engine"]

    instance_id = "test-reentrant-cacheblend-lifecycle"
    monkeypatch.setattr(blend_utils, "LMCBlender", FakeBuiltBlender)
    LMCacheEngineBuilder.destroy(instance_id)

    try:
        first_model = torch.nn.Linear(1, 1)
        first_engine = object()
        VLLMModelTracker.register_model(instance_id, first_model)
        first_blender = blend_utils.LMCBlenderBuilder.get_or_create(
            instance_id,
            cast(Any, first_engine),
            cast(Any, object()),
            cast(Any, object()),
        )
        first_blender_state = cast(Any, first_blender)
        assert first_blender_state.model is first_model
        assert first_blender_state.cache_engine is first_engine

        LMCacheEngineBuilder.destroy(instance_id)
        with pytest.raises(ValueError, match="Blender .* not found"):
            blend_utils.LMCBlenderBuilder.get(instance_id)
        with pytest.raises(ValueError, match="vllm model .* not found"):
            VLLMModelTracker.get_model(instance_id)

        second_model = torch.nn.Linear(1, 1)
        second_engine = object()
        VLLMModelTracker.register_model(instance_id, second_model)
        second_blender = blend_utils.LMCBlenderBuilder.get_or_create(
            instance_id,
            cast(Any, second_engine),
            cast(Any, object()),
            cast(Any, object()),
        )
        assert second_blender is not first_blender
        second_blender_state = cast(Any, second_blender)
        assert second_blender_state.model is second_model
        assert second_blender_state.cache_engine is second_engine
    finally:
        LMCacheEngineBuilder.destroy(instance_id)
