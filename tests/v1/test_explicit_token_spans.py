# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import FrozenInstanceError
from itertools import product
from typing import Any, cast
import random

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import SegmentTokenDatabase
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan
import lmcache.v1.token_database as token_database_module

# Local
from .utils import dumb_metadata


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        del text
        return [1, 99]


class _IntSubclass(int):
    pass


def _segment_database(monkeypatch: pytest.MonkeyPatch) -> SegmentTokenDatabase:
    monkeypatch.setattr(
        token_database_module.AutoTokenizer,
        "from_pretrained",
        lambda model_name: _FakeTokenizer(),
    )
    config = LMCacheEngineConfig.from_legacy(blend_special_str="<sep>")
    return SegmentTokenDatabase(config, dumb_metadata())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TokenSpan(True, 2),
        lambda: TokenSpan(_IntSubclass(0), 2),
        lambda: TokenSpan(cast(Any, np.int64(0)), 2),
        lambda: TokenSpan(cast(Any, torch.tensor(0)), 2),
        lambda: TokenSpan(-1, 2),
        lambda: TokenSpan(2, 2),
        lambda: TokenSpan(3, 2),
        lambda: ExplicitTokenSpans(0, TokenSpan(0, 1), (TokenSpan(0, 1),)),
        lambda: ExplicitTokenSpans(4, TokenSpan(1, 4), (TokenSpan(1, 2),)),
        lambda: ExplicitTokenSpans(4, TokenSpan(0, 5), (TokenSpan(0, 1),)),
        lambda: ExplicitTokenSpans(4, TokenSpan(0, 4), ()),
        lambda: ExplicitTokenSpans(
            4,
            TokenSpan(0, 4),
            cast(Any, [TokenSpan(0, 1)]),
        ),
        lambda: ExplicitTokenSpans(
            4,
            TokenSpan(0, 4),
            cast(Any, (TokenSpan(0, 3), (3, 4))),
        ),
        lambda: ExplicitTokenSpans(
            6,
            TokenSpan(0, 6),
            (TokenSpan(3, 5), TokenSpan(1, 2)),
        ),
        lambda: ExplicitTokenSpans(
            6,
            TokenSpan(0, 6),
            (TokenSpan(1, 4), TokenSpan(3, 5)),
        ),
        lambda: ExplicitTokenSpans(6, TokenSpan(0, 4), (TokenSpan(3, 5),)),
    ],
)
def test_explicit_token_spans_reject_invalid_or_tensor_values(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_explicit_token_spans_are_immutable() -> None:
    spans = ExplicitTokenSpans(
        token_count=4,
        cached_span=TokenSpan(0, 4),
        spans=(TokenSpan(0, 2), TokenSpan(2, 4)),
    )

    with pytest.raises(FrozenInstanceError):
        spans.token_count = 5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        spans.spans[0].end = 3  # type: ignore[misc]


def test_ttsep_and_explicit_spans_produce_identical_keys(monkeypatch) -> None:
    database = _segment_database(monkeypatch)
    tokens = torch.tensor([10, 11, 99, 20, 21, 22, 99, 30])
    request_configs = {"lmcache.tag.tenant": "a"}
    spans = ExplicitTokenSpans(
        token_count=len(tokens),
        cached_span=TokenSpan(0, len(tokens)),
        spans=(TokenSpan(0, 2), TokenSpan(3, 6), TokenSpan(7, 8)),
    )

    ttsep = list(
        database.process_tokens(tokens=tokens, request_configs=request_configs)
    )
    explicit = list(
        database.process_explicit_spans(
            tokens,
            spans,
            request_configs=request_configs,
        )
    )

    assert ttsep == explicit
    for _, _, key in explicit:
        assert isinstance(key, CacheEngineKey)
        assert key.request_configs == request_configs
        assert key.request_configs is not None
        assert "explicit_spans" not in key.request_configs


def test_span_offsets_and_cached_window_do_not_enter_key_identity(monkeypatch) -> None:
    database = _segment_database(monkeypatch)
    tokens = [7, 8, 7, 8]
    first = ExplicitTokenSpans(4, TokenSpan(0, 2), (TokenSpan(0, 2),))
    second = ExplicitTokenSpans(4, TokenSpan(0, 4), (TokenSpan(2, 4),))

    first_key = list(database.process_explicit_spans(tokens, first))[0][2]
    second_key = list(database.process_explicit_spans(tokens, second))[0][2]

    assert first_key == second_key


def _coalesced_spans(bits: tuple[bool, ...]) -> tuple[TokenSpan, ...]:
    spans: list[TokenSpan] = []
    start = None
    for index, selected in enumerate((*bits, False)):
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            spans.append(TokenSpan(start, index))
            start = None
    return tuple(spans)


def _assert_exact_hash_slices(
    database: SegmentTokenDatabase,
    tokens: list[int],
    spans: tuple[TokenSpan, ...],
) -> None:
    captured: list[tuple[int, ...]] = []

    def capture_hash(value) -> int:
        captured.append(value[1])
        return len(captured)

    database.hash_func = capture_hash
    declaration = ExplicitTokenSpans(
        len(tokens),
        TokenSpan(0, len(tokens)),
        spans,
    )
    results = list(database.process_explicit_spans(tokens, declaration, make_key=False))

    assert [(start, end) for start, end, _ in results] == [
        (span.start, span.end) for span in spans
    ]
    assert captured == [tuple(tokens[span.start : span.end]) for span in spans]


def test_exhaustive_and_random_span_partitions_hash_exact_slices(monkeypatch) -> None:
    database = _segment_database(monkeypatch)
    exhaustive_tokens = list(range(5))
    for bits in product((False, True), repeat=len(exhaustive_tokens)):
        spans = _coalesced_spans(bits)
        if spans:
            _assert_exact_hash_slices(database, exhaustive_tokens, spans)

    generator = random.Random(20260731)
    random_tokens = list(range(32))
    for _ in range(100):
        bits = tuple(generator.random() < 0.55 for _ in random_tokens)
        spans = _coalesced_spans(bits)
        if spans:
            _assert_exact_hash_slices(database, random_tokens, spans)


def test_explicit_span_mask_requires_prefix_shape(monkeypatch) -> None:
    database = _segment_database(monkeypatch)
    declaration = ExplicitTokenSpans(
        6,
        TokenSpan(0, 6),
        (TokenSpan(0, 2), TokenSpan(2, 4), TokenSpan(4, 6)),
    )

    mask = torch.tensor([False, False, True, True, True, True])
    assert [
        span[:2]
        for span in database.process_explicit_spans(
            list(range(6)), declaration, mask=mask
        )
    ] == [(2, 4), (4, 6)]

    crossing = ExplicitTokenSpans(
        6,
        TokenSpan(0, 6),
        (TokenSpan(0, 3), TokenSpan(4, 6)),
    )
    with pytest.raises(ValueError, match="cross the mask boundary"):
        list(
            database.process_explicit_spans(
                list(range(6)),
                crossing,
                mask=mask,
            )
        )

    with pytest.raises(ValueError, match="false prefix"):
        list(
            database.process_explicit_spans(
                list(range(6)),
                declaration,
                mask=torch.tensor([True, False, True, True, True, True]),
            )
        )
    with pytest.raises(ValueError, match="token_count"):
        list(database.process_explicit_spans(list(range(5)), declaration))
