# SPDX-License-Identifier: Apache-2.0
"""Normalize completed host snapshots without reading unordered device values."""

# Standard
import hashlib
import math

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.kmesh_domains import KMeshFenceError


def canonical_rope_cache_facts(
    caches: list[torch.Tensor],
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Hash completed finite CPU caches in their declared dtype and shape.

    Args:
        caches: Completed, privately owned CPU snapshots; no concurrent writes.

    Returns:
        Immutable dtype, shape and raw-content hashes for each cache.

    Raises:
        KMeshFenceError: A cache is not a finite floating CPU matrix.
    """
    facts = []
    for tensor in caches:
        if (
            tensor.device.type != "cpu"
            or tensor.ndim != 2
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all().item()
        ):
            raise KMeshFenceError("rotary snapshot must contain finite CPU matrices")
        host = tensor.detach().contiguous()
        content = hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()
        facts.append((str(host.dtype), tuple(host.shape), content))
    return tuple(facts)


def normalize_rope_snapshot(cache: torch.Tensor) -> torch.Tensor:
    """Remove baked-in rotary scale from a completed host snapshot.

    Args:
        cache: Finite CPU matrix with contiguous cosine and sine halves.

    Returns:
        A unit-magnitude snapshot in the original dtype, using the blend
        registration scale-removal threshold. No device work is submitted.

    Raises:
        KMeshFenceError: The matrix is malformed, non-finite or has zero scale.
    """
    canonical_rope_cache_facts([cache])
    if not cache.shape[0] or not cache.shape[1] or cache.shape[1] % 2:
        raise KMeshFenceError("rotary snapshot requires nonempty cosine/sine halves")
    values = cache.to(torch.float32)
    half = values.shape[1] // 2
    magnitude = float((values[:, :half] ** 2 + values[:, half:] ** 2).mean().sqrt())
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise KMeshFenceError("rotary snapshot has invalid scale")
    normalized = (
        (values / magnitude).to(cache.dtype) if abs(magnitude - 1.0) >= 1e-3 else cache
    )
    canonical_rope_cache_facts([normalized])
    return normalized
