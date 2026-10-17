# SPDX-License-Identifier: Apache-2.0
"""Content binding for idempotent synchronous lookup requests."""

# Standard
from hashlib import sha256
from typing import Optional, Union
import json

# Third Party
import torch

# First Party
from lmcache.v1.token_span import ExplicitTokenSpans


def make_serializable_token_ids(
    token_ids: Union[torch.Tensor, list[int]],
) -> list[int]:
    """Convert tensor/list-like token IDs into a msgpack-safe plain list."""
    if isinstance(token_ids, torch.Tensor):
        return token_ids.tolist()
    if not isinstance(token_ids, list):
        return list(token_ids)
    return token_ids


def make_lookup_request_fingerprint(
    token_ids: list[int],
    request_configs: Optional[dict],
    explicit_spans: Optional[ExplicitTokenSpans],
) -> bytes:
    """Return a stable content fingerprint for one lookup ID binding.

    Empty request configs are normalized to ``None`` because the synchronous
    wire protocol already gives those forms identical meaning. Config keys are
    sorted so insertion order does not create false drift failures.
    """
    explicit_declaration = None
    if explicit_spans is not None:
        explicit_spans.validate_token_count(len(token_ids))
        explicit_declaration = {
            "token_count": explicit_spans.token_count,
            "cached_span": [
                explicit_spans.cached_span.start,
                explicit_spans.cached_span.end,
            ],
            "spans": [[span.start, span.end] for span in explicit_spans.spans],
        }

    canonical_request = json.dumps(
        {
            "tokens": token_ids,
            "request_configs": request_configs or None,
            "explicit_spans": explicit_declaration,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(canonical_request).digest()
