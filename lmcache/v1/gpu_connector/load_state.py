# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj


@dataclass(frozen=True)
class PendingPositionRecovery:
    """One deferred cached-K position recovery for exactly one model layer.

    A layerwise load normally re-rotates each layer's cached K to its target
    positions inline.  A collective blend group instead defers that rotation
    so one batched application can cover every group member.  The record
    carries everything the group-level applier needs: the live device buffer
    (``kv_buffer.tensor`` is ``[2, tokens, hidden]``; index 0 is K), the
    per-token source and target positions, and the gap positions that must be
    zeroed strictly after rotation.
    """

    layer_id: int
    kv_buffer: MemoryObj
    old_positions: torch.Tensor
    new_positions: torch.Tensor
    gap_positions: torch.Tensor

    def __post_init__(self) -> None:
        if type(self.layer_id) is not int or self.layer_id < 0:
            raise ValueError("layer_id must be a nonnegative exact integer")
        tensor = self.kv_buffer.tensor
        if tensor is None or tensor.ndim != 3 or tensor.shape[0] != 2:
            raise ValueError("kv_buffer must hold a [2, tokens, hidden] tensor")
        token_count = tensor.shape[1]
        for name in ("old_positions", "new_positions"):
            positions = getattr(self, name)
            if (
                not isinstance(positions, torch.Tensor)
                or positions.ndim != 1
                or positions.dtype != torch.int64
                or positions.numel() != token_count
            ):
                raise ValueError(
                    f"{name} must be a one-dimensional int64 tensor matching "
                    "the buffer token count"
                )
        if (
            not isinstance(self.gap_positions, torch.Tensor)
            or self.gap_positions.ndim != 1
            or self.gap_positions.dtype != torch.int64
        ):
            raise ValueError("gap_positions must be a one-dimensional int64 tensor")


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
    defer_position_recovery: bool = False
    pending_position_recovery: Optional[PendingPositionRecovery] = field(
        default=None, init=False
    )
    position_recovery_serial_operations: int = field(default=0, init=False)
    position_recovery_collective_operations: int = field(default=0, init=False)
    _owned_buffers: list[MemoryObj] = field(default_factory=list, repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    def begin(self) -> None:
        """Prepare this state for one load.

        Raises:
            RuntimeError: If resources from an earlier load remain attached.
        """
        if self.buffer_mapping or self._owned_buffers:
            raise RuntimeError("Layerwise load state still owns resources")
        if self.pending_position_recovery is not None:
            raise RuntimeError(
                "Layerwise load state still holds a pending position recovery"
            )
        self.gap_positions = None
        self._released = False

    def publish_pending_position_recovery(
        self, pending: PendingPositionRecovery
    ) -> None:
        """Record one deferred rotation that a group applier must consume.

        Raises:
            RuntimeError: If the state was released, deferral is not enabled,
                or an earlier pending recovery was never applied.
        """
        if self._released:
            raise RuntimeError("Cannot defer position recovery on released load state")
        if not self.defer_position_recovery:
            raise RuntimeError(
                "Position recovery deferral is not enabled for this load"
            )
        if not isinstance(pending, PendingPositionRecovery):
            raise TypeError("pending must be PendingPositionRecovery")
        if self.pending_position_recovery is not None:
            raise RuntimeError(
                "A deferred position recovery was never applied before the "
                "next layer advanced"
            )
        self.pending_position_recovery = pending

    def consume_pending_position_recovery(self) -> PendingPositionRecovery:
        """Return and clear the pending recovery after it has been applied.

        Raises:
            RuntimeError: If no recovery is pending.
        """
        pending = self.pending_position_recovery
        if pending is None:
            raise RuntimeError("No position recovery is pending")
        self.pending_position_recovery = None
        return pending

    def record_position_recovery(self, *, serial: int = 0, collective: int = 0) -> None:
        """Count completed cached-K position-recovery applications.

        Raises:
            TypeError: If either count is not an exact integer.
            ValueError: If either count is negative.
        """
        if type(serial) is not int or type(collective) is not int:
            raise TypeError("position recovery counts must be exact integers")
        if serial < 0 or collective < 0:
            raise ValueError("position recovery counts cannot be negative")
        self.position_recovery_serial_operations += serial
        self.position_recovery_collective_operations += collective

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
        self.pending_position_recovery = None
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
