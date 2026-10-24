# SPDX-License-Identifier: Apache-2.0
"""Reject inconsistent model handoffs with CPU metadata, without device claims."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json

# Third Party
import blake3
import pytest

# First Party
from lmcache.integration.vllm.kmesh_span_layers import (
    CompletedSpanPlacement,
    RestoredSpanBatch,
)
from lmcache.v1.multiprocess.kmesh_bindings import RegisteredWorkerBindings
from lmcache.v1.multiprocess.kmesh_domains import DomainRegistry
from lmcache.v1.multiprocess.kmesh_types import SpanHandle, SpanOperationStatus
from lmcache.v1.multiprocess.transport.request_context import request_session
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan
from tests.v1.test_kmesh_bindings import semantics
from tests.v1.test_registered_rope_ownership import rotary_server


@contextmanager
def completed_batch() -> Iterator[RestoredSpanBatch]:
    """Use real CPU registration and synthetic receipts solely for consistency."""
    with rotary_server(hash_algorithm="blake3") as (blend, transfer):
        bindings = RegisteredWorkerBindings(DomainRegistry([]), transfer, blend)
        try:
            with request_session("zmq", b"worker"):
                runtime = bindings.describe(99)
                identity = replace(semantics(runtime), key_schema="kmesh-span-v1")
                lease = bindings.open_registration(
                    identity, "layer-inputs", 1, "4" * 64
                )
                fence = bindings.bind(lease, 0, 99, runtime, identity, "4" * 64)
            blocks = (13, 11, 9, 7, 5, 3, 1)
            shared = tuple(range(9))
            tokens = (999,) * 3 + shared + (888,) * 2
            handle = SpanHandle(
                lease.domain.namespace,
                "5" * 32,
                5,
                len(shared),
                blake3.blake3(
                    json.dumps(shared, separators=(",", ":")).encode()
                ).hexdigest(),
                "6" * 64,
                runtime.layout_digest,
                runtime.rope_digest,
            )
            receipt = SpanOperationStatus(
                fence,
                "destination",
                1,
                "7" * 32,
                runtime.context,
                "restore",
                hashlib.sha256(
                    json.dumps([blocks], separators=(",", ":")).encode()
                ).hexdigest(),
                "loaded-needs-repair",
                "",
                handle,
            )
            yield RestoredSpanBatch(
                fence,
                runtime.context,
                tokens,
                blocks,
                ExplicitTokenSpans(14, TokenSpan(0, 14), (TokenSpan(3, 12),)),
                (CompletedSpanPlacement("destination", 1, 3, receipt),),
            )
        finally:
            bindings.close()


def test_completed_batch_preserves_exact_partial_extent() -> None:
    with completed_batch() as batch:
        assert batch.explicit_spans.spans == (TokenSpan(3, 12),)
        receipt = batch.placements[0].receipt
        assert receipt.span is not None and receipt.span.token_count == 9
        assert receipt.state == "loaded-needs-repair"


def test_unrestored_declared_spans_remain_model_gaps() -> None:
    with completed_batch() as batch:
        declared = ExplicitTokenSpans(
            14, TokenSpan(0, 14), (TokenSpan(0, 1), TokenSpan(3, 12))
        )
        partial = replace(batch, explicit_spans=declared)
        assert partial.explicit_spans == declared
        assert partial.valid_spans == (TokenSpan(3, 12),)


@pytest.mark.parametrize(
    "case",
    [
        "request",
        "generation",
        "bool-generation",
        "kind",
        "blocks",
        "stored",
        "executing",
        "undrained",
        "failed",
        "error",
        "operation",
        "source",
    ],
)
def test_model_batch_refuses_mismatched_receipt(case: str) -> None:
    with completed_batch() as batch:
        original = batch.placements[0]
        r = original.receipt
        changed = {
            "request": replace(r, request_id="another-request"),
            "generation": replace(r, generation=2),
            "bool-generation": replace(r, generation=True),
            "kind": replace(r, kind="store"),
            "blocks": replace(r, blocks_digest="0" * 64),
            "stored": replace(r, state="stored"),
            "executing": replace(r, state="executing"),
            "undrained": replace(r, state="failed-undrained"),
            "failed": replace(r, state="failed-drained"),
            "error": replace(r, error="not a successful operation"),
            "operation": replace(r, operation_id=""),
            "source": replace(r, span=None),
        }
        with pytest.raises(ValueError, match="receipt"):
            replace(batch, placements=(replace(original, receipt=changed[case]),))


@pytest.mark.parametrize(
    "case", ["namespace", "layout", "rope", "tokens", "count", "bool-count", "start"]
)
def test_model_batch_refuses_changed_source(case: str) -> None:
    with completed_batch() as batch:
        placement = batch.placements[0]
        source = placement.receipt.span
        assert source is not None
        changed = {
            "namespace": replace(source, namespace="0" * 64),
            "layout": replace(source, layout_digest="0" * 64),
            "rope": replace(source, rope_digest="0" * 64),
            "tokens": replace(source, token_digest="0" * 64),
            "count": replace(source, token_count=8),
            "bool-count": replace(source, token_count=True),
            "start": replace(source, source_start=-1),
        }
        receipt = replace(placement.receipt, span=changed[case])
        with pytest.raises(ValueError, match="receipt"):
            replace(batch, placements=(replace(placement, receipt=receipt),))


@pytest.mark.parametrize(
    "case", ["tokens", "blocks", "owner", "fence", "context", "position", "missing"]
)
def test_model_batch_refuses_changed_input_geometry(case: str) -> None:
    with completed_batch() as batch:
        placement = batch.placements[0]
        with pytest.raises(ValueError):
            if case == "tokens":
                replace(batch, token_ids=(99,) * 14)
            elif case == "blocks":
                replace(batch, block_ids=batch.block_ids[::-1])
            elif case == "owner":
                replace(batch, context=replace(batch.context, owner_session="foreign"))
            elif case == "fence":
                receipt = replace(
                    placement.receipt,
                    fence=replace(batch.fence, token="replacement"),
                )
                replace(batch, placements=(replace(placement, receipt=receipt),))
            elif case == "context":
                receipt = replace(
                    placement.receipt,
                    context=replace(batch.context, generation="replacement"),
                )
                replace(batch, placements=(replace(placement, receipt=receipt),))
            elif case == "position":
                replace(batch, placements=(replace(placement, destination_start=4),))
            else:
                replace(batch, placements=())
