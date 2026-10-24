# SPDX-License-Identifier: Apache-2.0
"""Supply request-owned layer inputs to the existing CacheBlend model."""

# Standard
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

# Third Party
import torch

# First Party
from lmcache.v1.token_span import ExplicitTokenSpans

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.load_state import LayerwiseLoadState


class UnsettledLayerSourceError(RuntimeError):
    """Retain an actual cleanup owner before request-local fallback.

    Args:
        message: Physical cleanup failure, without claiming completion.
        drain: Bound owner operation that retries actual physical cleanup.

    The serial worker must invoke ``drain`` before permitting recomputation.
    A failed retry must escape request-local error containment.
    """

    def __init__(self, message: str, drain: Callable[[], None]) -> None:
        super().__init__(message)
        self._drain = drain

    def drain(self) -> None:
        """Retry the retained owner's cleanup; success permits fallback.

        Raises:
            Exception: The actual owner still cannot establish physical drain.
        """
        self._drain()

    def __reduce__(self) -> tuple[type[RuntimeError], tuple[str]]:
        """Serialize failure text without exporting the worker's cleanup owner.

        The callback can own live CUDA resources. A transported error reports
        failure only; it must not gain the local physical-cleanup capability.
        """
        return RuntimeError, (str(self),)


class BlendLayerSource(Protocol):
    """Provide exact restored spans without replacing model repair.

    A source owns its physical input lifetime and binds its token content,
    destination geometry and completed restoration before session creation.
    It publishes layer buffers and gap positions through the supplied state.
    """

    @property
    def explicit_spans(self) -> ExplicitTokenSpans:
        """Return the exact recovery window and valid source extents."""
        ...

    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        load_state: "LayerwiseLoadState",
    ) -> Generator[torch.Tensor | None, None, None]:
        """Yield a hit count, one boundary per layer, then the final hit mask.

        Args:
            tokens: Actual model input, checked against the bound content.
            mask: Requested token mask; unsupported masks must refuse.
            load_state: This session's private model-input mapping and buffers.

        Returns:
            A closeable generator that publishes actual layer inputs and writes
            repaired state back before its final mask. Close must settle device
            work or retain unresolved physical ownership and report failure.
        """
        ...


@dataclass(frozen=True)
class BlendLayerSourceRequest:
    """Name the exact ordered model input claimed by a group source owner.

    Args:
        request_id: Actual model request identity.
        generation: Caller-owned model scheduling generation.
        token_ids: Immutable complete model token window.
        explicit_spans: Exact recovery window and declared source extents.
    """

    request_id: str
    generation: int
    token_ids: tuple[int, ...]
    explicit_spans: ExplicitTokenSpans

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not str
            or not self.request_id
            or type(self.generation) is not int
            or self.generation < 0
            or type(self.token_ids) is not tuple
            or not self.token_ids
            or any(type(t) is not int or not 0 <= t < 2**63 for t in self.token_ids)
            or type(self.explicit_spans) is not ExplicitTokenSpans
        ):
            raise ValueError("invalid immutable group model input")
        self.explicit_spans.validate_token_count(len(self.token_ids))


class BlendLayerSourceGroup(Protocol):
    """Bind all model members and retain their aggregate cleanup ownership."""

    def bind(
        self, requests: tuple[BlendLayerSourceRequest, ...]
    ) -> tuple[BlendLayerSource, ...]:
        """Validate the complete ordered model input before returning any source.

        Args:
            requests: Actual model identities, token windows and declared spans.

        Returns:
            Single-use sources in exactly the validated member order.

        Raises:
            ValueError: Any member differs from the owner's completed inputs.
            RuntimeError: The owner was already bound, closed or remains live.
        """
        ...

    def close(self) -> None:
        """Attempt every member cleanup and report unresolved physical work.

        Raises:
            UnsettledLayerSourceError: Retained ownership still needs actual
                device drain before fallback or destination reuse.
            RuntimeError: A live consuming generator still owns its model step.
        """
        ...
