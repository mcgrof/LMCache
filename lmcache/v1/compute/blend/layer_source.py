# SPDX-License-Identifier: Apache-2.0
"""Supply request-owned layer inputs to the existing CacheBlend model."""

# Standard
from collections.abc import Generator
from typing import TYPE_CHECKING, Protocol

# Third Party
import torch

# First Party
from lmcache.v1.token_span import ExplicitTokenSpans

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.load_state import LayerwiseLoadState


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
