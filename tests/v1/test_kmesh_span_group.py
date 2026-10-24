# SPDX-License-Identifier: Apache-2.0
"""Reject group aliases with real CPU registration and synthetic receipts."""

# Standard
from dataclasses import replace
from typing import Any
import hashlib
import json

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.kmesh_span_group import (
    RestoredSpanGroup,
    RestoredSpanGroupMember,
)
from lmcache.integration.vllm.kmesh_span_layers import RestoredSpanBatch
from lmcache.v1.compute.blend.layer_source import BlendLayerSourceRequest
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan
from tests.v1.test_kmesh_span_layers import completed_batch


def sibling(batch: RestoredSpanBatch, index: int) -> RestoredSpanGroupMember:
    """Give a synthetic receipt a different private table and operation ID."""
    blocks = tuple(block + index * 16 for block in batch.block_ids)
    original = batch.placements[0]
    receipt = replace(
        original.receipt,
        request_id=f"restore-{index}",
        operation_id=f"operation-{index}",
        blocks_digest=hashlib.sha256(
            json.dumps([blocks], separators=(",", ":")).encode()
        ).hexdigest(),
    )
    placed = replace(original, request_id=receipt.request_id, receipt=receipt)
    return RestoredSpanGroupMember(
        f"model-{index}", 1, replace(batch, block_ids=blocks, placements=(placed,))
    )


def test_group_accepts_shared_canonical_source_and_private_destinations() -> None:
    with completed_batch() as batch:
        members = tuple(sibling(batch, index) for index in range(4))
        group = RestoredSpanGroup(members)
        assert group.members == members
        assert len({m.batch.placements[0].receipt.span for m in members}) == 1
        assert len({b for m in members for b in m.batch.block_ids}) == 28


@pytest.mark.parametrize("case", ["request", "blocks", "operation", "owner", "context"])
def test_group_refuses_cross_member_aliases_and_foreign_receipts(case: str) -> None:
    with completed_batch() as batch:
        first, second = sibling(batch, 0), sibling(batch, 1)
        changed = second.batch
        if case == "request":
            second = replace(second, request_id=first.request_id)
        elif case == "blocks":
            second = replace(second, batch=batch)
        elif case == "operation":
            p = changed.placements[0]
            r = replace(
                p.receipt, operation_id=first.batch.placements[0].receipt.operation_id
            )
            second = replace(
                second, batch=replace(changed, placements=(replace(p, receipt=r),))
            )
        else:
            context = replace(changed.context, generation="another-context")
            fence = changed.fence
            if case == "owner":
                fence = replace(fence, session_id="another-worker")
                context = replace(context, owner_session=fence.session_id)
            p = changed.placements[0]
            r = replace(p.receipt, context=context, fence=fence)
            second = replace(
                second,
                batch=replace(
                    changed,
                    context=context,
                    fence=fence,
                    placements=(replace(p, receipt=r),),
                ),
            )
        with pytest.raises(ValueError, match="aliased or foreign"):
            RestoredSpanGroup((first, second))


@pytest.mark.parametrize("case", ["empty", "single", "mutable", "foreign-member"])
def test_group_requires_a_complete_immutable_member_set(case: str) -> None:
    with completed_batch() as batch:
        members = (sibling(batch, 0), sibling(batch, 1))
        value = {
            "empty": (),
            "single": members[:1],
            "mutable": list(members),
            "foreign-member": (members[0], batch),
        }[case]
        with pytest.raises(ValueError, match="ordered immutable"):
            RestoredSpanGroup(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("case", ["empty-id", "bool-generation", "negative-generation"])
def test_group_member_requires_an_exact_model_identity(case: str) -> None:
    with completed_batch() as batch, pytest.raises(ValueError, match="member"):
        RestoredSpanGroupMember(
            "" if case == "empty-id" else "model",
            True
            if case == "bool-generation"
            else -1
            if case == "negative-generation"
            else 0,
            batch,
        )


@pytest.mark.parametrize(
    "case", ["match", "request", "generation", "tokens", "spans", "order", "missing"]
)
def test_bind_checks_every_actual_model_input_before_consumption(case: str) -> None:
    with completed_batch() as batch:
        group = RestoredSpanGroup((sibling(batch, 0), sibling(batch, 1)))
        requests = tuple(
            BlendLayerSourceRequest(
                m.request_id, m.generation, m.batch.token_ids, m.batch.explicit_spans
            )
            for m in group.members
        )
        if case == "match":
            group.validate_requests(requests)
            return
        if case == "order":
            requests = requests[::-1]
        elif case == "missing":
            requests = requests[:1]
        else:
            changes: dict[str, dict[str, Any]] = {
                "request": {"request_id": "other-model"},
                "generation": {"generation": 2},
                "tokens": {"token_ids": (321,) * 14},
                "spans": {
                    "explicit_spans": ExplicitTokenSpans(
                        14, TokenSpan(0, 14), (TokenSpan(4, 12),)
                    )
                },
            }
            changed = changes[case]
            requests = (requests[0], replace(requests[1], **changed))
        with pytest.raises(ValueError, match="complete registered group"):
            group.validate_requests(requests)


@pytest.mark.parametrize(
    "case", ["bool-generation", "mutable-tokens", "bool-token", "bounds"]
)
def test_group_model_input_rejects_mutable_or_ambiguous_values(case: str) -> None:
    spans = ExplicitTokenSpans(3, TokenSpan(0, 3), (TokenSpan(0, 3),))
    values = {
        "bool-generation": (True, (1, 2, 3)),
        "mutable-tokens": (1, [1, 2, 3]),
        "bool-token": (1, (1, True, 3)),
        "bounds": (1, (1, 2)),
    }
    generation, tokens = values[case]
    with pytest.raises(ValueError):
        BlendLayerSourceRequest("model", generation, tokens, spans)  # type: ignore[arg-type]
