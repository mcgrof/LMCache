# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Generator
from enum import Enum
from types import TracebackType
from typing import Any, Literal, Optional, Union

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.load_state import LayerwiseLoadState

logger = init_logger(__name__)


class LMCBlendOutcome(str, Enum):
    """Outcome established at the first layerwise retrieval boundary."""

    RECOVERED = "recovered"
    MISS = "miss"


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
    ) -> None:
        self.request_id = request_id
        self.generation = generation
        self.metadata = LMCBlendMetadata()
        self.load_state = LayerwiseLoadState()
        self.num_layers = blender.num_layers
        self.requires_explicit_gap_recompute = (
            retrieve_kwargs.get("explicit_spans") is not None
        )

        self._layerwise_model = blender.layerwise_model
        self._tokens = tokens
        self._model_generator: Optional[Generator[None, None, None]] = None
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
        self._finished = False
        self._closed = False
        self._retrieve_exhausted = False
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
        self._ensure_runnable()
        if self._completed_layers >= self.num_layers:
            raise RuntimeError("Every CacheBlend layer has already completed")
        if not self._primed:
            self.prime()

        layer_id = self._completed_layers
        if self.outcome is LMCBlendOutcome.MISS:
            self._advance_miss_retrieval()
            self._completed_layers += 1
            return

        try:
            next(self._retrieve_generator)
            assert self._model_generator is not None
            next(self._model_generator)
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
        **kwargs: Any,
    ) -> LMCBlendSession:
        """Create one request-owned CacheBlend session.

        Args:
            tokens: Tokens whose cached KV state should be blended.
            mask: Optional retrieval mask for ``tokens``.
            request_id: Stable request identifier for diagnostics.
            generation: Scheduling generation for diagnostics.
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
