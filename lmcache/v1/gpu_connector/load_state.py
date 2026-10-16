# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj


@dataclass
class LayerwiseLoadState:
    """Own mutable state for one layerwise device load.

    A layerwise CacheBlend load keeps two temporary device buffers alive while a
    model and retrieval generator advance together.  Keeping those buffers and
    the published layer mapping on the connector makes two active loads overwrite
    each other.  This object gives each load one mapping and one ownership list.

    ``release`` is idempotent and attempts every buffer release before reporting
    an error.  Callers may therefore use it from both generator ``finally``
    blocks and higher-level session cleanup.
    """

    buffer_mapping: dict[int, MemoryObj] = field(default_factory=dict)
    gap_positions: Optional[torch.Tensor] = None
    _owned_buffers: list[MemoryObj] = field(default_factory=list, repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    def begin(self) -> None:
        """Prepare this state for one load.

        Raises:
            RuntimeError: If resources from an earlier load remain attached.
        """
        if self.buffer_mapping or self._owned_buffers:
            raise RuntimeError("Layerwise load state still owns resources")
        self.gap_positions = None
        self._released = False

    def own_buffer(self, memory_obj: MemoryObj) -> None:
        """Record one temporary buffer that this load must release.

        Args:
            memory_obj: A newly allocated buffer with one owned reference.

        Raises:
            RuntimeError: If the load state has already been released.
        """
        if self._released:
            raise RuntimeError("Cannot attach a buffer to released load state")
        self._owned_buffers.append(memory_obj)

    def release(self) -> None:
        """Clear published state and release every owned buffer exactly once.

        Raises:
            RuntimeError: If one or more buffer releases fail.  All remaining
                buffers are still attempted before the error is raised.
        """
        if self._released:
            return

        self._released = True
        self.buffer_mapping.clear()
        self.gap_positions = None
        owned_buffers = self._owned_buffers
        self._owned_buffers = []

        first_error: Optional[Exception] = None
        for memory_obj in reversed(owned_buffers):
            try:
                memory_obj.ref_count_down()
            except Exception as error:  # pragma: no cover - defensive GPU cleanup
                if first_error is None:
                    first_error = error

        if first_error is not None:
            raise RuntimeError(
                "Failed to release a layerwise device buffer"
            ) from first_error

    @property
    def owned_buffer_count(self) -> int:
        """Return the number of temporary buffers still owned by this load."""
        return len(self._owned_buffers)

    @property
    def is_released(self) -> bool:
        """Return whether cleanup has completed for this load."""
        return self._released
