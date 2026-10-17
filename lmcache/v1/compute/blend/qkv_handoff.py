# SPDX-License-Identifier: Apache-2.0
"""Mid-layer handoff types for collective model-side QKV processing.

The serial CacheBlend model generator calls ``LMCBlender.process_qkv``
synchronously inside every layer, so per-member rotary encoding and the
check-layer difference pass cannot batch across a group.  The collective
model generator instead suspends at exactly that point: it yields an
``LMCQKVHandoff`` carrying the tensors ``process_qkv`` would have
consumed, and resumes with an ``LMCQKVCorrection`` carrying the tensors
``process_qkv`` would have returned.  The group coordinator between those
two events performs one batched rotary application and, at the check
layer, one batched difference pass covering every member.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Optional

# Third Party
import torch


@dataclass
class LMCQKVHandoff:
    """Tensors one member exposes at the mid-layer suspension point."""

    layer_id: int
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    residual: torch.Tensor
    attn_output: Optional[torch.Tensor]
    attn_metadata: Any

    def __post_init__(self) -> None:
        if type(self.layer_id) is not int or self.layer_id < 0:
            raise ValueError("layer_id must be a nonnegative exact integer")
        for name in ("q", "k", "v", "residual"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                raise ValueError(f"{name} must be a two-dimensional tensor")
        if self.q.shape[0] != self.k.shape[0] or self.k.shape[0] != self.v.shape[0]:
            raise ValueError("q, k, and v must agree on token count")


@dataclass
class LMCQKVCorrection:
    """Tensors the coordinator returns to resume one member's layer."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    residual: torch.Tensor
    attn_output: torch.Tensor
    attn_metadata: Any
