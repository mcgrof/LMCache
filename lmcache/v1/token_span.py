# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass


def _is_strict_int(value: object) -> bool:
    return type(value) is int


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """One immutable half-open token range.

    Args:
        start: Inclusive token offset.
        end: Exclusive token offset.

    Raises:
        TypeError: If either offset is not a plain Python integer.
        ValueError: If the range is negative, empty, or reversed.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if not _is_strict_int(self.start) or not _is_strict_int(self.end):
            raise TypeError("token span offsets must be plain Python integers")
        if self.start < 0:
            raise ValueError("token span start must be nonnegative")
        if self.end <= self.start:
            raise ValueError("token span end must be greater than start")


@dataclass(frozen=True, slots=True)
class ExplicitTokenSpans:
    """Validated explicit cache spans for one token sequence.

    ``spans`` selects the exact token slices whose content is hashed. The
    separate ``cached_span`` declares the contiguous recovery window used by
    CacheBlend; gaps inside that window are recomputed. The v0 recovery path
    supports a prompt-origin window only.

    Args:
        token_count: Length of the token sequence this object describes.
        cached_span: Contiguous recovery window. Its start must be zero in v0.
        spans: Nonempty, ordered, nonoverlapping token slices contained in the
            recovery window.

    Raises:
        TypeError: If the representation contains non-primitive values or
            ``spans`` is not a tuple of :class:`TokenSpan` objects.
        ValueError: If a range is out of bounds, the span tuple is empty,
            ranges are unordered/overlapping, or the recovery window does not
            begin at the prompt origin.
    """

    token_count: int
    cached_span: TokenSpan
    spans: tuple[TokenSpan, ...]

    def __post_init__(self) -> None:
        if not _is_strict_int(self.token_count):
            raise TypeError("token_count must be a plain Python integer")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if not isinstance(self.cached_span, TokenSpan):
            raise TypeError("cached_span must be a TokenSpan")
        if self.cached_span.start != 0:
            raise ValueError("cached_span must begin at token zero in v0")
        if self.cached_span.end > self.token_count:
            raise ValueError("cached_span is out of token-sequence bounds")
        if not isinstance(self.spans, tuple):
            raise TypeError("spans must be an immutable tuple")
        if not self.spans:
            raise ValueError("explicit token spans must not be empty")

        previous_end = self.cached_span.start
        for span in self.spans:
            if not isinstance(span, TokenSpan):
                raise TypeError("spans must contain only TokenSpan objects")
            if span.start < previous_end:
                raise ValueError(
                    "explicit token spans must be ordered and nonoverlapping"
                )
            if span.start < self.cached_span.start or span.end > self.cached_span.end:
                raise ValueError("explicit token span is outside cached_span")
            previous_end = span.end

    def validate_token_count(self, actual_token_count: int) -> None:
        """Validate that a supplied token sequence matches this declaration.

        Args:
            actual_token_count: Length of the token sequence supplied to an
                LMCache operation.

        Raises:
            ValueError: If the actual token length differs from ``token_count``.
        """
        if actual_token_count != self.token_count:
            raise ValueError(
                "explicit token span token_count does not match supplied tokens"
            )
