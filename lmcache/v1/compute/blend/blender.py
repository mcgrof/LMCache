# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.load_state import PendingPositionRecovery

logger = init_logger(__name__)


class LMCBlendOutcome(str, Enum):
    """Outcome established at the first layerwise retrieval boundary."""

    RECOVERED = "recovered"
    MISS = "miss"


class LMCBlendGroupOutcome(str, Enum):
    """Final outcome for one member of a lockstep blend group."""

    RECOVERED = "recovered"
    MISS = "miss"
    FAILED = "failed"


@dataclass(frozen=True)
class LMCBlendOperationCounts:
    """Actual tensor-operation calls completed for one request.

    Serial fields count completed request-local rotary and check-layer
    difference calls. Collective fields count group calls in which this
    member participated. Cached-key position recovery has separate serial
    and collective counters. A shared call appears in each participant's
    record, so summing member counters does not count unique group calls.
    These counters do not measure tokens, FLOPs or wall time.
    """

    serial_rope_operations: int = 0
    serial_diff_operations: int = 0
    collective_rope_operations: int = 0
    collective_diff_operations: int = 0
    position_recovery_serial_operations: int = 0
    position_recovery_collective_operations: int = 0


@dataclass(frozen=True)
class LMCBlendCollectivePass:
    """Evidence for one batched position-recovery application."""

    layer_id: int
    member_count: int
    token_count: int


@dataclass(frozen=True)
class LMCBlendGroupRequest:
    """One ordered member of a bounded lockstep CacheBlend run.

    Args:
        request_id: Stable request identity used in results and diagnostics.
        tokens: Complete token window supplied to the request-owned session.
        mask: Optional boolean retrieval mask matching ``tokens``.
        generation: Caller-owned request generation used for invalidation.
        retrieve_kwargs: Request-local arguments forwarded to
            :meth:`LMCBlender.create_session`, including explicit spans.

    Notes:
        :meth:`LMCBlender.blend_group` snapshots mutable list and mapping
        inputs before it creates any session.
    """

    request_id: str
    tokens: Union[torch.Tensor, list[int]]
    mask: Optional[torch.Tensor] = None
    generation: int = 0
    retrieve_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LMCBlendGroupMemberResult:
    """Final result for one ordered member of a lockstep blend group."""

    request_id: str
    generation: int
    outcome: LMCBlendGroupOutcome
    retrieval_mask: Optional[torch.Tensor]
    operation_counts: LMCBlendOperationCounts
    error: Optional[Exception] = None


@dataclass(frozen=True)
class LMCBlendGroupResult:
    """Ordered results and invalidation surface for a lockstep blend group."""

    members: tuple[LMCBlendGroupMemberResult, ...]
    collective_passes: tuple[LMCBlendCollectivePass, ...] = ()

    @property
    def failed_members(self) -> tuple[LMCBlendGroupMemberResult, ...]:
        """Return members whose request-local execution or cleanup failed."""
        return tuple(
            member
            for member in self.members
            if member.outcome is LMCBlendGroupOutcome.FAILED
        )

    @property
    def failed_request_ids(self) -> tuple[str, ...]:
        """Return failed request IDs in the caller's stable input order."""
        return tuple(member.request_id for member in self.failed_members)


@dataclass(frozen=True)
class _PreparedLMCBlendGroupRequest:
    request_id: str
    tokens: Union[torch.Tensor, list[int]]
    mask: Optional[torch.Tensor]
    generation: int
    retrieve_kwargs: dict[str, Any]


@dataclass
class _LMCBlendGroupMemberState:
    request: _PreparedLMCBlendGroupRequest
    session: Optional["LMCBlendSession"] = None
    error: Optional[Exception] = None


class LMCBlendSession:
    """One request-owned, closeable CacheBlend execution.

    The model and retrieval generators intentionally remain separate so a
    caller can advance independent sessions one model layer at a time.  All
    mutable blend metadata and layerwise load state belong to this session.
    Closing the session closes both generators and releases any temporary GPU
    buffers, including when a generator raises partway through a layer.
    """

    def __init__(
        self,
        blender: "LMCBlender",
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        request_id: Optional[str],
        generation: int,
        retrieve_kwargs: dict[str, Any],
        defer_position_recovery: bool = False,
        collective_qkv: bool = False,
    ) -> None:
        # First Party
        # Defer package initialization until the blend module is complete;
        # gpu_connector imports the blend builder as part of its public API.
        from lmcache.v1.gpu_connector.load_state import LayerwiseLoadState

        if type(defer_position_recovery) is not bool:
            raise TypeError("defer_position_recovery must be a boolean")
        if type(collective_qkv) is not bool:
            raise TypeError("collective_qkv must be a boolean")
        self.collective_qkv = collective_qkv
        self.request_id = request_id
        self.generation = generation
        self.metadata = LMCBlendMetadata()
        self.load_state = LayerwiseLoadState(
            defer_position_recovery=defer_position_recovery
        )
        self.num_layers = blender.num_layers
        self.requires_explicit_gap_recompute = (
            retrieve_kwargs.get("explicit_spans") is not None
        )

        self._layerwise_model = blender.layerwise_model
        self._tokens = tokens
        self._model_generator: Optional[Generator[object, object, None]] = None
        self._retrieve_generator: Generator[Optional[torch.Tensor], None, None] = (
            blender.cache_engine.retrieve_layer(
                tokens,
                mask,
                load_state=self.load_state,
                **retrieve_kwargs,
            )
        )
        self._primed = False
        self._completed_layers = 0
        self._retrieval_stepped = False
        self._finished = False
        self._closed = False
        self._retrieve_exhausted = False
        self._serial_rope_operations = 0
        self._serial_diff_operations = 0
        self._collective_rope_operations = 0
        self._collective_diff_operations = 0
        self._collective_position_recovery_passes = 0
        self._miss_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
        self.outcome: Optional[LMCBlendOutcome] = None
        self.result: Optional[torch.Tensor] = None

    def __enter__(self) -> "LMCBlendSession":
        """Enter this session's cleanup scope."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Literal[False]:
        """Close the session without suppressing an active exception."""
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                logger.exception("CacheBlend cleanup failed after an exception")
        return False

    def prime(self) -> LMCBlendOutcome:
        """Advance retrieval to its first layer boundary.

        The cache engine reports how many tokens it can actually recover at
        this boundary.  A zero count (or an all-false compatibility mask) is
        an explicit miss: subsequent layer steps drain retrieval only and do
        not advance the recovery model.

        Raises:
            RuntimeError: If the session is closed or already finished, or if
                the retrieval generator terminates before its first boundary.
        """
        self._ensure_runnable()
        if self._primed:
            assert self.outcome is not None
            return self.outcome
        try:
            prime_result = next(self._retrieve_generator)
            self._primed = True
            if self._prime_is_miss(prime_result):
                self.outcome = LMCBlendOutcome.MISS
            else:
                self.outcome = LMCBlendOutcome.RECOVERED
                if self.collective_qkv:
                    self._model_generator = (
                        self._layerwise_model.compute_layer_collective(
                            self._tokens, self
                        )
                    )
                else:
                    self._model_generator = self._layerwise_model.compute_layer(
                        self._tokens, self
                    )
            return self.outcome
        except StopIteration as error:
            self._abort()
            raise RuntimeError(
                "Layerwise retrieval ended before the first layer"
            ) from error
        except BaseException:
            self._abort()
            raise

    def step_layer(self) -> None:
        """Advance retrieval and model computation by one whole layer.

        Raises:
            RuntimeError: If every layer has already completed or either
                generator terminates before the expected layer boundary.
        """
        self.step_retrieval()
        self.step_model()

    def step_retrieval(self) -> None:
        """Advance retrieval by one layer boundary without model compute.

        A collective blend group uses this sub-step so every member's layer
        buffer exists before one batched position recovery covers the whole
        group.  ``step_model`` must complete the layer before the next
        retrieval sub-step.

        Raises:
            RuntimeError: If every layer has already completed, the layer's
                retrieval already advanced, or the retrieval generator
                terminates before the expected boundary.
        """
        self._ensure_runnable()
        if self._completed_layers >= self.num_layers:
            raise RuntimeError("Every CacheBlend layer has already completed")
        if self._retrieval_stepped:
            raise RuntimeError("CacheBlend retrieval already advanced for this layer")
        if not self._primed:
            self.prime()

        layer_id = self._completed_layers
        if self.outcome is LMCBlendOutcome.MISS:
            self._advance_miss_retrieval()
            self._retrieval_stepped = True
            return

        try:
            next(self._retrieve_generator)
            self._retrieval_stepped = True
        except StopIteration as error:
            self._abort()
            raise RuntimeError(
                f"CacheBlend generator ended while advancing layer {layer_id}"
            ) from error
        except BaseException:
            self._abort()
            raise

    def step_model(self) -> None:
        """Advance model computation for the layer retrieval just loaded.

        Raises:
            RuntimeError: If retrieval has not advanced for this layer, a
                deferred position recovery was never applied, or the model
                generator terminates before the expected boundary.
        """
        self._ensure_runnable()
        if not self._retrieval_stepped:
            raise RuntimeError(
                "CacheBlend model cannot advance before this layer's retrieval"
            )

        layer_id = self._completed_layers
        if self.outcome is LMCBlendOutcome.MISS:
            self._retrieval_stepped = False
            self._completed_layers += 1
            return

        if self.load_state.pending_position_recovery is not None:
            self._abort()
            raise RuntimeError(
                "CacheBlend model cannot advance while a deferred position "
                "recovery is unapplied"
            )

        try:
            assert self._model_generator is not None
            next(self._model_generator)
            self._retrieval_stepped = False
            self._completed_layers += 1
        except StopIteration as error:
            self._abort()
            raise RuntimeError(
                f"CacheBlend generator ended while advancing layer {layer_id}"
            ) from error
        except BaseException:
            self._abort()
            raise

    def step_model_prepare(self) -> object:
        """Advance the collective model to this layer's QKV suspension.

        Returns the member's mid-layer handoff.  Only valid for a
        recovered collective-QKV session whose retrieval sub-step ran and
        whose deferred position recovery was applied.

        Raises:
            RuntimeError: If the session is not in the collective-QKV
                recovered state or a generator ends early.
        """
        self._ensure_runnable()
        if not self.collective_qkv:
            raise RuntimeError("session does not use collective QKV processing")
        if self.outcome is not LMCBlendOutcome.RECOVERED:
            raise RuntimeError("only recovered members expose a QKV handoff")
        if not self._retrieval_stepped:
            raise RuntimeError(
                "CacheBlend model cannot advance before this layer's retrieval"
            )
        if self.load_state.pending_position_recovery is not None:
            self._abort()
            raise RuntimeError(
                "CacheBlend model cannot advance while a deferred position "
                "recovery is unapplied"
            )
        layer_id = self._completed_layers
        try:
            assert self._model_generator is not None
            handoff = next(self._model_generator)
            if handoff is None:
                raise RuntimeError("collective model layer yielded no QKV handoff")
            return handoff
        except StopIteration as error:
            self._abort()
            raise RuntimeError(
                f"CacheBlend generator ended while advancing layer {layer_id}"
            ) from error
        except BaseException:
            self._abort()
            raise

    def step_model_complete(self, correction: object) -> None:
        """Resume the collective model with its group-produced correction.

        Raises:
            RuntimeError: If the resumed layer does not complete cleanly.
        """
        self._ensure_runnable()
        if not self.collective_qkv:
            raise RuntimeError("session does not use collective QKV processing")
        layer_id = self._completed_layers
        try:
            assert self._model_generator is not None
            marker = self._model_generator.send(correction)
            if marker is not None:
                raise RuntimeError(
                    "collective model layer did not complete after resume"
                )
            self._retrieval_stepped = False
            self._completed_layers += 1
        except StopIteration as error:
            self._abort()
            raise RuntimeError(
                f"CacheBlend generator ended while advancing layer {layer_id}"
            ) from error
        except BaseException:
            self._abort()
            raise

    def finish(self) -> torch.Tensor:
        """Finish device writeback and capture the retrieval mask.

        Returns:
            The final token retrieval mask. An explicit miss always returns an
            all-false mask, including for legacy retrievers that terminate
            before their final boundary.

        Raises:
            RuntimeError: If not every model layer has completed, or if the
                retrieval generator terminates before its final boundary.
        """
        self._ensure_runnable()
        if self._completed_layers != self.num_layers:
            raise RuntimeError(
                "Cannot finish CacheBlend before every model layer completes"
            )

        if self.outcome is LMCBlendOutcome.MISS:
            self._advance_miss_retrieval()
            self.result = self._miss_mask
            self._finished = True
            return self.result

        try:
            result = next(self._retrieve_generator)
            if result is None:
                raise RuntimeError(
                    "Layerwise retrieval did not return a final retrieval mask"
                )
            self.result = result
            self._finished = True
            return self.result
        except StopIteration as error:
            self._abort()
            raise RuntimeError(
                "Layerwise retrieval ended before final writeback"
            ) from error
        except BaseException:
            self._abort()
            raise

    def run(self) -> torch.Tensor:
        """Run every layer serially and return the final retrieval mask."""
        self.prime()
        while self._completed_layers < self.num_layers:
            self.step_layer()
        return self.finish()

    def close(self) -> None:
        """Close both generators and release all request-owned state.

        The method is idempotent and attempts every cleanup operation before
        reporting the first failure.

        Raises:
            RuntimeError: If a generator or buffer cleanup operation fails.
        """
        if self._closed:
            return
        self._closed = True

        errors: list[Exception] = []
        generators = (self._model_generator, self._retrieve_generator)
        for generator in generators:
            if generator is None:
                continue
            try:
                generator.close()
            except Exception as error:
                errors.append(error)

        try:
            self.load_state.release()
        except Exception as error:
            errors.append(error)

        self.metadata.clean()
        if errors:
            raise RuntimeError("Failed to close CacheBlend session") from errors[0]

    @property
    def completed_layers(self) -> int:
        """Return how many whole model layers this session completed."""
        return self._completed_layers

    @property
    def is_closed(self) -> bool:
        """Return whether session cleanup has run."""
        return self._closed

    @property
    def operation_counts(self) -> LMCBlendOperationCounts:
        """Return a snapshot of actual operations completed by this session."""
        return LMCBlendOperationCounts(
            serial_rope_operations=self._serial_rope_operations,
            serial_diff_operations=self._serial_diff_operations,
            collective_rope_operations=self._collective_rope_operations,
            collective_diff_operations=self._collective_diff_operations,
            position_recovery_serial_operations=(
                self.load_state.position_recovery_serial_operations
            ),
            position_recovery_collective_operations=(
                self.load_state.position_recovery_collective_operations
            ),
        )

    def record_serial_operations(
        self,
        *,
        rope_operations: int = 0,
        diff_operations: int = 0,
    ) -> None:
        """Record request-local tensor operations after they complete.

        Args:
            rope_operations: Number of completed serial RoPE invocations.
            diff_operations: Number of completed serial diff invocations.

        Raises:
            TypeError: If either count is not an exact integer.
            ValueError: If either count is negative.
        """
        if type(rope_operations) is not int or type(diff_operations) is not int:
            raise TypeError("CacheBlend operation counts must be exact integers")
        if rope_operations < 0 or diff_operations < 0:
            raise ValueError("CacheBlend operation counts cannot be negative")
        self._serial_rope_operations += rope_operations
        self._serial_diff_operations += diff_operations

    def record_collective_operations(
        self,
        *,
        rope_operations: int = 0,
        diff_operations: int = 0,
    ) -> None:
        """Record batched group tensor operations this member took part in.

        Raises:
            TypeError: If either count is not an exact integer.
            ValueError: If either count is negative.
        """
        if type(rope_operations) is not int or type(diff_operations) is not int:
            raise TypeError("CacheBlend operation counts must be exact integers")
        if rope_operations < 0 or diff_operations < 0:
            raise ValueError("CacheBlend operation counts cannot be negative")
        self._collective_rope_operations += rope_operations
        self._collective_diff_operations += diff_operations

    def _ensure_runnable(self) -> None:
        if self._closed:
            raise RuntimeError("CacheBlend session is closed")
        if self._finished:
            raise RuntimeError("CacheBlend session is already finished")

    def _abort(self) -> None:
        try:
            self.close()
        except Exception:
            logger.exception("CacheBlend session cleanup failed during abort")

    @staticmethod
    def _prime_is_miss(prime_result: Optional[torch.Tensor]) -> bool:
        """Return whether the first retrieval boundary reports no actual hit."""
        if prime_result is None:
            # Compatibility with custom retrievers that predate the explicit
            # hit-count boundary. The in-tree engine always reports a tensor.
            return False
        if prime_result.numel() == 0:
            return True
        return not bool(torch.any(prime_result).item())

    def _advance_miss_retrieval(self) -> None:
        """Drain one miss boundary, tolerating legacy short generators."""
        if self._retrieve_exhausted:
            return
        try:
            next(self._retrieve_generator)
        except StopIteration:
            self._retrieve_exhausted = True
        except BaseException:
            self._abort()
            raise


class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)

        # TODO: remove this hardcode
        self.num_layers = len(vllm_model.model.layers)

        # TODO(Jiayi): support threshold-based blending
        # TODO(Jiayi): support different ratios for different layers
        # TODO(Jiayi): support "skipping blending if hit too short"
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
        session: LMCBlendSession,
    ):
        logger.debug("Blender is processing KV for layer %s", layer_id)
        old_k, old_v = self.gpu_connector.get_kv(layer_id, session.load_state)
        metadata = session.metadata

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        # perform positional encoding
        if metadata.positions is None:
            metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        q, k = attn_layer.rotary_emb(metadata.positions, q, k)
        session.record_serial_operations(rope_operations=1)

        mandatory_indices: Optional[torch.Tensor] = None
        if session.requires_explicit_gap_recompute and metadata.imp_indices is None:
            gap_positions = session.load_state.gap_positions
            if gap_positions is None:
                raise RuntimeError(
                    "Explicit CacheBlend load did not publish its gap positions"
                )
            if (
                type(gap_positions) is not torch.Tensor
                or gap_positions.ndim != 1
                or gap_positions.dtype != torch.int64
                or gap_positions.device != k.device
            ):
                raise RuntimeError(
                    "Explicit CacheBlend gap positions have invalid tensor metadata"
                )
            if old_k.shape[0] != k.shape[0] or old_v.shape[0] != v.shape[0]:
                raise RuntimeError(
                    "Explicit CacheBlend buffer does not cover the model token window"
                )
            mandatory_indices = gap_positions
            if bool(
                torch.any(
                    (mandatory_indices < 0)
                    | (mandatory_indices >= old_k.shape[0])
                    | (mandatory_indices >= k.shape[0])
                ).item()
            ):
                raise RuntimeError(
                    "CacheBlend gap position is outside the recovery window"
                )

        if layer_id in self.common_metadata.check_layers:
            diff_k = torch.sum(
                (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
            )
            session.record_serial_operations(diff_operations=1)
            total_len = diff_k.shape[0]

            assert self.common_metadata.recomp_ratios is not None

            # TODO(Jiayi): remove `[0]` hardcode
            topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
            topk_num = max(topk_num, 1)

            top_indices = torch.topk(diff_k, k=topk_num).indices
            if mandatory_indices is not None:
                mandatory_indices = mandatory_indices.to(
                    device=top_indices.device,
                    dtype=top_indices.dtype,
                )
                # Missing explicit spans are zero-filled by the GPU loader.
                # Every such position is mandatory recomputation, even when
                # that exceeds the configured quality-based top-k budget.
                top_indices = torch.unique(
                    torch.cat((top_indices, mandatory_indices)),
                    sorted=True,
                )
            top_indices, _ = torch.sort(top_indices)

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            logger.debug("Number of indices picked: %s", len(top_indices))

            metadata.imp_indices = top_indices
            metadata.positions = metadata.positions[top_indices]
            attn_output = attn_output[: len(top_indices)]

            attn_metadata.update_from_top_indices(top_indices)

        elif metadata.imp_indices is None and mandatory_indices is not None:
            # Layers before the quality-selection checkpoint compute the full
            # sequence, but the connector later publishes ``old_k``/``old_v``
            # to paged KV memory. Repair every zero-filled miss in that buffer
            # before it can be written back or stored.
            old_k[mandatory_indices] = k[mandatory_indices]
            old_v[mandatory_indices] = v[mandatory_indices]

        if metadata.imp_indices is not None:
            old_k[metadata.imp_indices] = k
            old_v[metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata
        else:
            return q, k, v, residual, attn_output, attn_metadata

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to ochestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwiese retrieve + blending.
        """

        # TODO(Jiayi): store is currently not included in this function
        with self.create_session(tokens, mask, **kwargs) as session:
            session.prime()
            yield

            for _ in range(self.num_layers):
                session.step_layer()
                yield

            session.finish()
            yield

    def create_session(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        *,
        request_id: Optional[str] = None,
        generation: int = 0,
        defer_position_recovery: bool = False,
        collective_qkv: bool = False,
        **kwargs: Any,
    ) -> LMCBlendSession:
        """Create one request-owned CacheBlend session.

        Args:
            tokens: Tokens whose cached KV state should be blended.
            mask: Optional retrieval mask for ``tokens``.
            request_id: Stable request identifier for diagnostics.
            generation: Scheduling generation for diagnostics.
            defer_position_recovery: Defer cached-K rotation so a collective
                group applier can batch it; only ``blend_group`` may apply
                deferred rotations, so plain serial callers must leave this
                false.
            **kwargs: Arguments forwarded to ``retrieve_layer``.

        Returns:
            A closeable session that can run serially or advance layerwise.

        Raises:
            ValueError: If a caller attempts to inject a shared load state.
        """
        if "load_state" in kwargs:
            raise ValueError("CacheBlend sessions always own their load state")
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).to(torch_device_type)
        if request_id is None:
            request_id_value = kwargs.get("req_id")
            if request_id_value is not None:
                request_id = str(request_id_value)
        retrieve_kwargs = dict(kwargs)
        if request_id is not None and "req_id" not in retrieve_kwargs:
            retrieve_kwargs["req_id"] = request_id

        return LMCBlendSession(
            blender=self,
            tokens=tokens,
            mask=mask,
            request_id=request_id,
            generation=generation,
            retrieve_kwargs=retrieve_kwargs,
            defer_position_recovery=defer_position_recovery,
            collective_qkv=collective_qkv,
        )

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform blending for the given tokens.
        """

        with self.create_session(tokens, mask, **kwargs) as session:
            return session.run()

    def blend_group(
        self,
        requests: Sequence[LMCBlendGroupRequest],
        *,
        collective_position_recovery: bool = False,
        collective_qkv: bool = False,
    ) -> LMCBlendGroupResult:
        """Run request-owned CacheBlend sessions in bounded lockstep.

        The complete group is structurally validated and snapshotted before
        the first session is created.  All sessions are then created and
        primed in stable input order.  Each round advances every surviving
        member by exactly one whole model layer before the next round starts.
        Ordinary request-local :class:`Exception` failures are isolated so
        siblings can finish; failed members are returned for caller-side cache
        invalidation.  Process-level :class:`BaseException` failures propagate
        only after every created session has been closed.

        With ``collective_position_recovery`` false this is the lockstep
        correctness reference: every tensor operation stays request-local and
        both collective counters remain zero.  With it true, each layer round
        splits into a retrieval sub-step for every member, ONE batched
        cached-K position-recovery application covering every member that
        loaded cached state, and a model sub-step for every member.  Model
        compute never observes an unrotated buffer: the session fails closed
        if a deferred rotation was not applied.  The batched application
        changes tensor shapes only, never per-element arithmetic; per-token
        model-side RoPE and diff work remains request-local and serial.

        Args:
            requests: Ordered, nonempty group of request-local blend inputs.
            collective_position_recovery: Batch cached-K position recovery
                across the group instead of rotating per member.

        Returns:
            Ordered member results, including failed request IDs, exact
            operation counts, and one evidence record per collective pass.

        Raises:
            TypeError: If group metadata has invalid structural types.
            ValueError: If group metadata is empty, ambiguous, or inconsistent.
            BaseException: If a process-level failure interrupts execution or
                session cleanup. Every created session is closed first.
        """
        if type(collective_position_recovery) is not bool:
            raise TypeError("collective_position_recovery must be a boolean")
        if type(collective_qkv) is not bool:
            raise TypeError("collective_qkv must be a boolean")
        if collective_qkv and not collective_position_recovery:
            raise ValueError(
                "collective QKV processing requires collective position "
                "recovery: the group modes share one retrieval/model round "
                "structure"
            )
        prepared = self._prepare_group_requests(requests)
        states = [_LMCBlendGroupMemberState(request=request) for request in prepared]
        collective_passes: list[LMCBlendCollectivePass] = []

        try:
            # Create every request-owned session before retrieval begins.
            for state in states:
                request = state.request
                try:
                    state.session = self.create_session(
                        request.tokens,
                        request.mask,
                        request_id=request.request_id,
                        generation=request.generation,
                        defer_position_recovery=collective_position_recovery,
                        collective_qkv=collective_qkv,
                        **request.retrieve_kwargs,
                    )
                except Exception as error:
                    state.error = error

            # Prime every live member before the first layer-major round.
            for state in states:
                if state.error is not None or state.session is None:
                    continue
                try:
                    state.session.prime()
                except Exception as error:
                    state.error = error

            for _ in range(self.num_layers):
                if collective_position_recovery:
                    for state in states:
                        if state.error is not None or state.session is None:
                            continue
                        try:
                            state.session.step_retrieval()
                        except Exception as error:
                            state.error = error
                    collective_pass = self._apply_collective_position_recovery(states)
                    if collective_pass is not None:
                        collective_passes.append(collective_pass)
                    if collective_qkv:
                        participants: list[tuple[_LMCBlendGroupMemberState, Any]] = []
                        for state in states:
                            if state.error is not None or state.session is None:
                                continue
                            if state.session.outcome is not LMCBlendOutcome.RECOVERED:
                                continue
                            try:
                                participants.append(
                                    (state, state.session.step_model_prepare())
                                )
                            except Exception as error:
                                state.error = error
                        corrections = self._apply_collective_qkv(participants)
                        if corrections is not None:
                            for (state, _handoff), correction in zip(
                                participants, corrections, strict=True
                            ):
                                if state.error is not None:
                                    continue
                                try:
                                    cast(
                                        LMCBlendSession, state.session
                                    ).step_model_complete(correction)
                                except Exception as error:
                                    state.error = error
                        for state in states:
                            if state.error is not None or state.session is None:
                                continue
                            if state.session.outcome is LMCBlendOutcome.RECOVERED:
                                continue
                            try:
                                state.session.step_model()
                            except Exception as error:
                                state.error = error
                    else:
                        for state in states:
                            if state.error is not None or state.session is None:
                                continue
                            try:
                                state.session.step_model()
                            except Exception as error:
                                state.error = error
                else:
                    for state in states:
                        if state.error is not None or state.session is None:
                            continue
                        try:
                            state.session.step_layer()
                        except Exception as error:
                            state.error = error

            for state in states:
                if state.error is not None or state.session is None:
                    continue
                try:
                    state.session.finish()
                except Exception as error:
                    state.error = error
        except BaseException:
            self._close_group_sessions(states, suppress_base_exceptions=True)
            raise

        self._close_group_sessions(states, suppress_base_exceptions=False)
        return self._build_group_result(states, tuple(collective_passes))

    def _apply_collective_position_recovery(
        self, states: Sequence[_LMCBlendGroupMemberState]
    ) -> Optional[LMCBlendCollectivePass]:
        """Apply one batched rotation over every member with a pending layer.

        Members without cached state to rotate (for example explicit misses)
        simply do not participate.  Any application failure poisons exactly
        the participating members so their caches can be invalidated, while
        non-participating members continue.
        """
        participants: list[_LMCBlendGroupMemberState] = []
        for state in states:
            if state.error is not None or state.session is None:
                continue
            if state.session.load_state.pending_position_recovery is not None:
                participants.append(state)
        if not participants:
            return None

        try:
            pendings = [
                cast(
                    "PendingPositionRecovery",
                    cast(
                        LMCBlendSession, state.session
                    ).load_state.pending_position_recovery,
                )
                for state in participants
            ]
            layer_ids = {pending.layer_id for pending in pendings}
            if len(layer_ids) != 1:
                raise RuntimeError(
                    "Collective position recovery requires one lockstep layer; "
                    f"observed layers {sorted(layer_ids)}"
                )
            k_views = []
            for pending in pendings:
                tensor = pending.kv_buffer.tensor
                assert tensor is not None
                k_views.append(tensor[0])
            devices = {tensor.device for tensor in k_views}
            dtypes = {tensor.dtype for tensor in k_views}
            hiddens = {tensor.shape[-1] for tensor in k_views}
            if len(devices) != 1 or len(dtypes) != 1 or len(hiddens) != 1:
                raise RuntimeError(
                    "Collective position recovery requires one device, dtype, "
                    "and hidden size across the group"
                )

            token_counts = [view.shape[0] for view in k_views]
            packed_k = torch.cat(k_views, dim=0)
            packed_old = torch.cat(
                [pending.old_positions for pending in pendings], dim=0
            )
            packed_new = torch.cat(
                [pending.new_positions for pending in pendings], dim=0
            )
            rotated = self.layerwise_model.fused_rotary_emb(
                packed_old, packed_new, packed_k
            )

            offset = 0
            for pending, view, token_count in zip(
                pendings, k_views, token_counts, strict=True
            ):
                view.copy_(rotated[offset : offset + token_count])
                offset += token_count
                tensor = pending.kv_buffer.tensor
                assert tensor is not None
                if pending.gap_positions.numel():
                    tensor[:, pending.gap_positions] = 0.0

            for state in participants:
                assert state.session is not None
                state.session.load_state.consume_pending_position_recovery()
                state.session.load_state.record_position_recovery(collective=1)
            return LMCBlendCollectivePass(
                layer_id=next(iter(layer_ids)),
                member_count=len(participants),
                token_count=sum(token_counts),
            )
        except Exception as error:
            for state in participants:
                if state.error is None:
                    state.error = error
            return None

    def _apply_collective_qkv(
        self,
        participants: "list[tuple[_LMCBlendGroupMemberState, Any]]",
    ) -> "Optional[list[Any]]":
        """Batch one layer's model-side QKV work across the whole group.

        Mirrors :meth:`process_qkv` member for member.  Exactly two
        operations are batched, both per-token elementwise, by token-axis
        concatenation: the rotary application over every member's q and k,
        and the check-layer key-difference computation.  Top-k selection,
        gap repair, metadata mutation, and cached-KV merging stay
        member-local and byte-identical to the serial path.  Any failure
        poisons exactly the participating members; the caller advances
        non-participants independently.
        """
        if not participants:
            return None
        # First Party
        from lmcache.v1.compute.blend.qkv_handoff import LMCQKVCorrection

        try:
            layer_ids = {handoff.layer_id for _, handoff in participants}
            if len(layer_ids) != 1:
                raise RuntimeError(
                    "collective QKV requires one lockstep layer; observed "
                    f"layers {sorted(layer_ids)}"
                )
            layer_id = next(iter(layer_ids))
            layer = self.layerwise_model.vllm_model.model.layers[layer_id]
            attn_layer = layer.self_attn

            members: list[dict[str, Any]] = []
            for state, handoff in participants:
                session = state.session
                assert session is not None
                metadata = session.metadata
                old_k, old_v = self.gpu_connector.get_kv(layer_id, session.load_state)
                attn_output = handoff.attn_output
                if attn_output is None:
                    attn_output = torch.empty(
                        handoff.q.shape,
                        dtype=handoff.q.dtype,
                        device=handoff.q.device,
                    )
                if metadata.positions is None:
                    metadata.positions = torch.arange(
                        handoff.q.shape[0],
                        device=handoff.q.device,
                        dtype=torch.int64,
                    )
                members.append(
                    {
                        "session": session,
                        "metadata": metadata,
                        "q": handoff.q,
                        "k": handoff.k,
                        "v": handoff.v,
                        "residual": handoff.residual,
                        "attn_output": attn_output,
                        "attn_metadata": handoff.attn_metadata,
                        "old_k": old_k,
                        "old_v": old_v,
                    }
                )

            token_counts = [member["q"].shape[0] for member in members]
            packed_positions = torch.cat(
                [member["metadata"].positions for member in members]
            )
            packed_q = torch.cat([member["q"] for member in members])
            packed_k = torch.cat([member["k"] for member in members])
            packed_q, packed_k = attn_layer.rotary_emb(
                packed_positions, packed_q, packed_k
            )
            for member, member_q, member_k in zip(
                members,
                torch.split(packed_q, token_counts),
                torch.split(packed_k, token_counts),
                strict=True,
            ):
                member["q"] = member_q
                member["k"] = member_k
                member["session"].record_collective_operations(rope_operations=1)

            for member in members:
                session = member["session"]
                metadata = member["metadata"]
                k = member["k"]
                v = member["v"]
                old_k = member["old_k"]
                old_v = member["old_v"]
                mandatory_indices: Optional[torch.Tensor] = None
                if (
                    session.requires_explicit_gap_recompute
                    and metadata.imp_indices is None
                ):
                    gap_positions = session.load_state.gap_positions
                    if gap_positions is None:
                        raise RuntimeError(
                            "Explicit CacheBlend load did not publish its gap positions"
                        )
                    if (
                        type(gap_positions) is not torch.Tensor
                        or gap_positions.ndim != 1
                        or gap_positions.dtype != torch.int64
                        or gap_positions.device != k.device
                    ):
                        raise RuntimeError(
                            "Explicit CacheBlend gap positions have invalid "
                            "tensor metadata"
                        )
                    if old_k.shape[0] != k.shape[0] or old_v.shape[0] != v.shape[0]:
                        raise RuntimeError(
                            "Explicit CacheBlend buffer does not cover the "
                            "model token window"
                        )
                    mandatory_indices = gap_positions
                    if bool(
                        torch.any(
                            (mandatory_indices < 0)
                            | (mandatory_indices >= old_k.shape[0])
                            | (mandatory_indices >= k.shape[0])
                        ).item()
                    ):
                        raise RuntimeError(
                            "CacheBlend gap position is outside the recovery window"
                        )
                member["mandatory_indices"] = mandatory_indices

            if layer_id in self.common_metadata.check_layers:
                packed_old_k = torch.cat([member["old_k"] for member in members])
                packed_diff = torch.sum(
                    (packed_k.to(torch.float32) - packed_old_k.to(torch.float32)) ** 2,
                    dim=[1],
                )
                assert self.common_metadata.recomp_ratios is not None
                for member, diff_k in zip(
                    members, torch.split(packed_diff, token_counts), strict=True
                ):
                    metadata = member["metadata"]
                    total_len = diff_k.shape[0]
                    topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
                    topk_num = max(topk_num, 1)
                    top_indices = torch.topk(diff_k, k=topk_num).indices
                    mandatory_indices = member["mandatory_indices"]
                    if mandatory_indices is not None:
                        mandatory_indices = mandatory_indices.to(
                            device=top_indices.device,
                            dtype=top_indices.dtype,
                        )
                        top_indices = torch.unique(
                            torch.cat((top_indices, mandatory_indices)),
                            sorted=True,
                        )
                    top_indices, _ = torch.sort(top_indices)
                    member["k"] = member["k"][top_indices]
                    member["v"] = member["v"][top_indices]
                    member["q"] = member["q"][top_indices]
                    member["residual"] = member["residual"][top_indices]
                    metadata.imp_indices = top_indices
                    metadata.positions = metadata.positions[top_indices]
                    member["attn_output"] = member["attn_output"][: len(top_indices)]
                    member["attn_metadata"].update_from_top_indices(top_indices)
                    member["session"].record_collective_operations(diff_operations=1)
            else:
                for member in members:
                    metadata = member["metadata"]
                    mandatory_indices = member["mandatory_indices"]
                    if metadata.imp_indices is None and mandatory_indices is not None:
                        member["old_k"][mandatory_indices] = member["k"][
                            mandatory_indices
                        ]
                        member["old_v"][mandatory_indices] = member["v"][
                            mandatory_indices
                        ]

            corrections: list[Any] = []
            for member in members:
                metadata = member["metadata"]
                if metadata.imp_indices is not None:
                    member["old_k"][metadata.imp_indices] = member["k"]
                    member["old_v"][metadata.imp_indices] = member["v"]
                    corrections.append(
                        LMCQKVCorrection(
                            q=member["q"],
                            k=member["old_k"],
                            v=member["old_v"],
                            residual=member["residual"],
                            attn_output=member["attn_output"],
                            attn_metadata=member["attn_metadata"],
                        )
                    )
                else:
                    corrections.append(
                        LMCQKVCorrection(
                            q=member["q"],
                            k=member["k"],
                            v=member["v"],
                            residual=member["residual"],
                            attn_output=member["attn_output"],
                            attn_metadata=member["attn_metadata"],
                        )
                    )
            return corrections
        except Exception as error:
            for state, _handoff in participants:
                if state.error is None:
                    state.error = error
            return None

    @staticmethod
    def _prepare_group_requests(
        requests: Sequence[LMCBlendGroupRequest],
    ) -> tuple[_PreparedLMCBlendGroupRequest, ...]:
        request_snapshot = tuple(requests)
        if not request_snapshot:
            raise ValueError("CacheBlend group must contain at least one request")

        prepared: list[_PreparedLMCBlendGroupRequest] = []
        request_ids: set[str] = set()
        reserved_kwargs = {
            "collective_qkv",
            "defer_position_recovery",
            "generation",
            "load_state",
            "mask",
            "request_id",
            "tokens",
        }
        for index, request in enumerate(request_snapshot):
            if not isinstance(request, LMCBlendGroupRequest):
                raise TypeError(
                    f"CacheBlend group member {index} must be LMCBlendGroupRequest"
                )
            if type(request.request_id) is not str or not request.request_id:
                raise TypeError(
                    f"CacheBlend group member {index} request_id must be "
                    "a nonempty exact string"
                )
            if request.request_id in request_ids:
                raise ValueError(
                    f"CacheBlend group request_id is duplicated: {request.request_id}"
                )
            request_ids.add(request.request_id)

            if type(request.generation) is not int:
                raise TypeError(
                    f"CacheBlend group member {index} generation must be "
                    "an exact integer"
                )
            if request.generation < 0:
                raise ValueError(
                    f"CacheBlend group member {index} generation cannot be negative"
                )

            tokens = request.tokens
            if isinstance(tokens, torch.Tensor):
                if tokens.ndim != 1 or tokens.numel() == 0:
                    raise ValueError(
                        f"CacheBlend group member {index} tokens must be "
                        "a nonempty one-dimensional tensor"
                    )
                if (
                    tokens.dtype is torch.bool
                    or tokens.dtype.is_floating_point
                    or tokens.dtype.is_complex
                ):
                    raise TypeError(
                        f"CacheBlend group member {index} tokens must use "
                        "an integer dtype"
                    )
                prepared_tokens: Union[torch.Tensor, list[int]] = tokens
                token_count = tokens.numel()
            elif type(tokens) is list:
                if not tokens:
                    raise ValueError(
                        f"CacheBlend group member {index} tokens cannot be empty"
                    )
                if any(type(token) is not int or token < 0 for token in tokens):
                    raise TypeError(
                        f"CacheBlend group member {index} token list must "
                        "contain only nonnegative exact integers"
                    )
                prepared_tokens = list(tokens)
                token_count = len(tokens)
            else:
                raise TypeError(
                    f"CacheBlend group member {index} tokens must be "
                    "a tensor or list of integers"
                )

            mask = request.mask
            if mask is not None:
                if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool:
                    raise TypeError(
                        f"CacheBlend group member {index} mask must be a boolean tensor"
                    )
                if mask.ndim != 1 or mask.numel() != token_count:
                    raise ValueError(
                        f"CacheBlend group member {index} mask must be "
                        "one-dimensional and match tokens"
                    )

            if not isinstance(request.retrieve_kwargs, Mapping):
                raise TypeError(
                    f"CacheBlend group member {index} retrieve_kwargs must be a mapping"
                )
            retrieve_kwargs = dict(request.retrieve_kwargs)
            if any(type(key) is not str for key in retrieve_kwargs):
                raise TypeError(
                    f"CacheBlend group member {index} retrieve_kwargs keys "
                    "must be exact strings"
                )
            conflicts = sorted(reserved_kwargs.intersection(retrieve_kwargs))
            if conflicts:
                raise ValueError(
                    f"CacheBlend group member {index} retrieve_kwargs cannot "
                    f"override {', '.join(conflicts)}"
                )

            prepared.append(
                _PreparedLMCBlendGroupRequest(
                    request_id=request.request_id,
                    tokens=prepared_tokens,
                    mask=mask,
                    generation=request.generation,
                    retrieve_kwargs=retrieve_kwargs,
                )
            )
        return tuple(prepared)

    @staticmethod
    def _close_group_sessions(
        states: Sequence[_LMCBlendGroupMemberState],
        *,
        suppress_base_exceptions: bool,
    ) -> None:
        fatal_errors: list[BaseException] = []
        for state in states:
            session = state.session
            if session is None or session.is_closed:
                continue
            try:
                session.close()
            except Exception as close_error:
                if state.error is None:
                    state.error = close_error
            except BaseException as caught_fatal_error:
                fatal_errors.append(caught_fatal_error)

        if not fatal_errors:
            return
        if suppress_base_exceptions:
            for cleanup_error in fatal_errors:
                logger.error(
                    "CacheBlend group cleanup failed during process-level abort",
                    exc_info=(
                        type(cleanup_error),
                        cleanup_error,
                        cleanup_error.__traceback__,
                    ),
                )
            return
        first_error = fatal_errors[0]
        raise first_error.with_traceback(first_error.__traceback__)

    @staticmethod
    def _build_group_result(
        states: Sequence[_LMCBlendGroupMemberState],
        collective_passes: tuple[LMCBlendCollectivePass, ...] = (),
    ) -> LMCBlendGroupResult:
        members: list[LMCBlendGroupMemberResult] = []
        for state in states:
            session = state.session
            operation_counts = (
                session.operation_counts
                if session is not None
                else LMCBlendOperationCounts()
            )
            error = state.error
            if error is not None:
                outcome = LMCBlendGroupOutcome.FAILED
                retrieval_mask = None
            elif session is None or session.result is None or session.outcome is None:
                error = RuntimeError(
                    "CacheBlend group member ended without a terminal result"
                )
                outcome = LMCBlendGroupOutcome.FAILED
                retrieval_mask = None
            else:
                outcome = LMCBlendGroupOutcome(session.outcome.value)
                retrieval_mask = session.result

            members.append(
                LMCBlendGroupMemberResult(
                    request_id=state.request.request_id,
                    generation=state.request.generation,
                    outcome=outcome,
                    retrieval_mask=retrieval_mask,
                    operation_counts=operation_counts,
                    error=error,
                )
            )
        return LMCBlendGroupResult(
            members=tuple(members), collective_passes=collective_passes
        )
