# SPDX-License-Identifier: Apache-2.0
"""Bind an entire registered model group before any member consumes its slots."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.integration.vllm.kmesh_span_layers import RestoredSpanBatch


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
