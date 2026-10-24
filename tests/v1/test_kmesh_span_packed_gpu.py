# SPDX-License-Identifier: Apache-2.0
"""Qualify actual fused-content canonical sources and shifted native restores."""

# Standard
from typing import Literal

# Third Party
import pytest

# First Party
from tests.v1 import test_kmesh_span_store_gpu as fixture

pytestmark = fixture.pytestmark


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("start,count", [(5, 9), (0, 13)])
def test_fused_partial_source_and_shifted_restore(
    transport: Literal["zmq", "grpc"],
    layout: Literal["NHD", "HND"],
    start: int,
    count: int,
) -> None:
    fixture.test_managed_span_native_store_and_restore(
        transport, "success", start, 3, layout, 24, count, packed=True
    )
