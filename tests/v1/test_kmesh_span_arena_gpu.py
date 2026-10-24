# SPDX-License-Identifier: Apache-2.0
"""Qualify registered spans in actual eager pinned and GPU L1 arenas."""

# Standard
from typing import Literal

# Third Party
import pytest

# First Party
from tests.v1 import test_kmesh_span_store_gpu as fixture

pytestmark = fixture.pytestmark


@pytest.mark.parametrize("source_memory", ["eager-host", "vulkan_rm"])
@pytest.mark.parametrize("transport", ["zmq", "grpc"])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("start,count", [(5, 9), (0, 13), (0, 8)])
def test_arena_source_preserves_canonical_bytes_and_shifted_values(
    source_memory: Literal["eager-host", "vulkan_rm"],
    transport: Literal["zmq", "grpc"],
    layout: Literal["NHD", "HND"],
    packed: bool,
    start: int,
    count: int,
) -> None:
    fixture.test_managed_span_native_store_and_restore(
        transport,
        "success",
        start,
        3,
        layout,
        24,
        count,
        packed=packed,
        source_memory=source_memory,
    )


@pytest.mark.parametrize("source_memory", ["eager-host", "vulkan_rm"])
@pytest.mark.parametrize("transport", ["zmq", "grpc"])
@pytest.mark.parametrize(
    "mode", ["corrupt", "evict", "enqueue-failure", "drain-refusal", "source-undrained"]
)
def test_arena_source_failure_requires_physical_drain(
    source_memory: Literal["eager-host", "vulkan_rm"],
    transport: Literal["zmq", "grpc"],
    mode: str,
) -> None:
    fixture.test_managed_span_native_store_and_restore(
        transport,
        mode,
        5,
        3,
        "HND",
        24,
        9,
        packed=True,
        source_memory=source_memory,
    )


@pytest.mark.parametrize("source_memory", ["eager-host", "vulkan_rm"])
@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_arena_source_refuses_more_chunks_than_private_working_slots(
    source_memory: Literal["eager-host", "vulkan_rm"],
    transport: Literal["zmq", "grpc"],
) -> None:
    # Five 8-token chunks exceed the registered context's four private slots.
    fixture.test_managed_span_native_store_and_restore(
        transport,
        "slot-reuse",
        0,
        3,
        "HND",
        24,
        33,
        packed=True,
        source_memory=source_memory,
    )
