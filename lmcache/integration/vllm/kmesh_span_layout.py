# SPDX-License-Identifier: Apache-2.0
"""Resolve physical paged-cache shapes without changing their storage."""

# Standard
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RegisteredPagedShape:
    """Describe separate K/V planes or per-head fused K/V content.

    Args:
        blocks: Physical block count.
        block_size: Tokens in each physical block.
        kv_heads: Actual K/V heads, excluding the fused K/V content axis.
        head_size: Actual per-key or per-value head size, never doubled.
        format_name: Native physical format matching the original view.
    """

    blocks: int
    block_size: int
    kv_heads: int
    head_size: int
    format_name: str


def resolve_registered_paged_shape(
    shape: tuple[int, ...], layout: Literal["NHD", "HND"]
) -> RegisteredPagedShape:
    """Resolve an actual contiguous two-plane or fused-content tensor shape.

    Args:
        shape: Original physical layer shape, as an immutable integer tuple.
        layout: Engine-provided token-major NHD or head-major HND ordering.

    Returns:
        Checked physical block geometry and the undoubled rotary head size.

    Raises:
        ValueError: Unsupported rank, layout, plane axis or rotary geometry.

    Four-dimensional views pack each head's K and V into its trailing content
    axis. Five-dimensional views keep a leading two-plane K/V axis. Neither
    case authorizes allocating a differently laid-out replacement cache.
    """
    if (
        type(shape) is not tuple
        or layout not in ("NHD", "HND")
        or len(shape) not in (4, 5)
        or any(type(n) is not int or n < 1 for n in shape)
    ):
        raise ValueError("unsupported registered paged shape")
    if len(shape) == 5:
        if shape[0] != 2:
            raise ValueError("registered separate planes require exactly K and V")
        blocks, head_size = shape[1], shape[4]
        block_size, heads = (
            (shape[2], shape[3]) if layout == "NHD" else (shape[3], shape[2])
        )
        fmt = "NL_X_TWO_NB_BS_NH_HS" if layout == "NHD" else "NL_X_TWO_NB_NH_BS_HS"
    else:
        if shape[3] % 2:
            raise ValueError("registered fused content must contain equal K and V")
        blocks, head_size = shape[0], shape[3] // 2
        block_size, heads = (
            (shape[1], shape[2]) if layout == "NHD" else (shape[2], shape[1])
        )
        fmt = "NL_X_NB_BS_NH_CS" if layout == "NHD" else "NL_X_NB_NH_BS_CS"
    if head_size % 2:
        raise ValueError("registered rotary head size must be even")
    return RegisteredPagedShape(blocks, block_size, heads, head_size, fmt)
