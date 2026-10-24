# SPDX-License-Identifier: Apache-2.0
"""Bind an entire registered model group before any member consumes its slots."""

# Standard
from dataclasses import dataclass
from typing import Literal

# Third Party
import torch

# First Party
from lmcache.integration.vllm.kmesh_span_layers import (
    RestoredPagedLayerSource,
    RestoredSpanBatch,
)
from lmcache.integration.vllm.kmesh_span_layout import resolve_registered_paged_shape
from lmcache.logging import init_logger
from lmcache.v1.compute.blend.layer_source import (
    BlendLayerSource,
    BlendLayerSourceRequest,
    UnsettledLayerSourceError,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class RestoredSpanGroupMember:
    """Name one model request and its completed, private destination batch.

    Args:
        request_id: Actual model request identity, unique within the group.
        generation: Nonnegative model scheduling generation.
        batch: Exact completed restores and still-owned destination block table.

    This metadata does not acquire a scheduler reservation. The caller retains
    every member's real engine blocks through the complete consuming step.
    """

    request_id: str
    generation: int
    batch: RestoredSpanBatch

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not str
            or not self.request_id
            or type(self.generation) is not int
            or self.generation < 0
            or type(self.batch) is not RestoredSpanBatch
        ):
            raise ValueError("invalid registered model group member")


@dataclass(frozen=True)
class RestoredSpanGroup:
    """Require one owner and disjoint mutable destinations for a whole group.

    Args:
        members: Immutable ordered model requests and completed placement data.

    Shared immutable canonical sources are allowed. Every destination table and
    restore operation belongs to exactly one member. Validation touches metadata
    only and cannot establish device completion beyond the supplied receipts.

    Raises:
        ValueError: Membership, receiving owner, blocks or operations conflict.
    """

    members: tuple[RestoredSpanGroupMember, ...]

    def __post_init__(self) -> None:
        if (
            type(self.members) is not tuple
            or len(self.members) < 2
            or any(
                type(member) is not RestoredSpanGroupMember for member in self.members
            )
        ):
            raise ValueError(
                "registered model group requires ordered immutable members"
            )
        first = self.members[0].batch
        names: set[str] = set()
        blocks: set[int] = set()
        operations: set[str] = set()
        for member in self.members:
            batch = member.batch
            if (
                member.request_id in names
                or batch.fence != first.fence
                or batch.context != first.context
                or blocks.intersection(batch.block_ids)
                or any(
                    placement.receipt.operation_id in operations
                    for placement in batch.placements
                )
            ):
                raise ValueError(
                    "registered model group has aliased or foreign ownership"
                )
            names.add(member.request_id)
            blocks.update(batch.block_ids)
            operations.update(p.receipt.operation_id for p in batch.placements)

    def validate_requests(self, requests: tuple[BlendLayerSourceRequest, ...]) -> None:
        """Check the complete model membership without opening any layer source.

        Args:
            requests: Actual ordered model identities, tokens and declared spans.

        Raises:
            ValueError: Any identity, order or model input differs from the group.
        """
        expected = tuple(
            BlendLayerSourceRequest(
                member.request_id,
                member.generation,
                member.batch.token_ids,
                member.batch.explicit_spans,
            )
            for member in self.members
        )
        if type(requests) is not tuple or requests != expected:
            raise ValueError("model requests differ from the complete registered group")


class RestoredPagedLayerGroup:
    """Own private layer sources for an atomically validated registered group.

    Args:
        group: Complete validated model members and destination reservations.
        kv_caches: The worker's actual registered engine cache views.
        layout: Their physical NHD/HND ordering.

    Construction validates every member's cache bounds before creating any
    source. It does not enqueue device work. The owning worker retains this
    group until every source closes and physical work is settled.
    """

    def __init__(
        self,
        group: RestoredSpanGroup,
        kv_caches: tuple[torch.Tensor, ...],
        layout: Literal["NHD", "HND"],
    ) -> None:
        if not kv_caches:
            raise ValueError("registered model group requires actual cache views")
        geometry = resolve_registered_paged_shape(tuple(kv_caches[0].shape), layout)
        if any(
            max(m.batch.block_ids) >= geometry.blocks
            or len(m.batch.token_ids) > len(m.batch.block_ids) * geometry.block_size
            for m in group.members
        ):
            raise ValueError("registered model group exceeds its reserved cache blocks")
        self._group = group
        self._sources = tuple(
            RestoredPagedLayerSource(member.batch, kv_caches, layout)
            for member in group.members
        )
        self._bound = False

    @property
    def pending(self) -> bool:
        """Report any member still owing checked physical completion."""
        return any(source.pending for source in self._sources)

    @property
    def is_closed(self) -> bool:
        """Report logical retirement of every member, separately from drain."""
        return all(source.is_closed for source in self._sources)

    def bind(
        self, requests: tuple[BlendLayerSourceRequest, ...]
    ) -> tuple[BlendLayerSource, ...]:
        """Match the entire ordered model input before exposing any source.

        Args:
            requests: Actual model request identities, tokens and declared spans.

        Returns:
            Private, single-use model inputs in that exact member order.

        Raises:
            ValueError: Any model input differs from its completed span batch.
            RuntimeError: This group was already bound or closed.
        """
        if self._bound or self.is_closed:
            raise RuntimeError("registered model group is already bound or closed")
        self._group.validate_requests(requests)
        self._bound = True
        return self._sources

    def close(self) -> None:
        """Attempt all member cleanups while retaining unresolved ownership.

        Raises:
            UnsettledLayerSourceError: At least one actual device drain failed.
            RuntimeError: A live generator still owns a consuming model step.
        """
        errors: list[Exception] = []
        for source in self._sources:
            try:
                source.close()
            except Exception as error:
                logger.exception("registered group member cleanup remains unresolved")
                errors.append(error)
        if errors:
            if any(isinstance(error, UnsettledLayerSourceError) for error in errors):
                raise UnsettledLayerSourceError(
                    "registered group still owes physical cleanup", self.close
                ) from errors[0]
            raise RuntimeError(
                "registered group still owns a live model source"
            ) from errors[0]

    def drain_failed(self) -> None:
        """Retry every source cleanup after its consuming generator closes.

        Raises:
            UnsettledLayerSourceError: Actual device drain remains unresolved.
            RuntimeError: A consuming generator is still live.
        """
        self.close()
