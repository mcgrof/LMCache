# SPDX-License-Identifier: Apache-2.0
"""Check stream completion with real GPU memory and FP8 codec operations."""

# Standard
from contextlib import nullcontext
from unittest.mock import patch
import select

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde import (
    AsyncSerdeProcessor,
    Fp8QuantizationDeserializer,
    Fp8QuantizationSerializer,
)
from lmcache.v1.distributed.serde.base import SerdeTaskOutcome
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator

pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        not torch_dev.is_available(), reason="requires a device runtime"
    ),
]


@pytest.mark.parametrize("operation", ["serialize", "deserialize"])
@pytest.mark.parametrize("fence_failure", [False, True])
def test_gpu_codec_completion_and_unknown_ownership(
    operation: str, fence_failure: bool
) -> None:
    """Reject result-only queries when a completed transform loses its fence."""
    allocator = GPUMemoryAllocator(4 << 20, device="cuda:0", align_bytes=4096)
    source = allocator.allocate(torch.Size([4097]), torch.bfloat16)
    encoded = allocator.allocate(torch.Size([4097]), torch.uint8)
    assert source is not None and encoded is not None
    assert source.tensor is not None
    source.tensor.fill_(2)
    serializer = Fp8QuantizationSerializer()
    key = ObjectKey(ObjectKey.IntHash2Bytes(47), "gpu-codec", 0)
    if operation == "deserialize":
        serializer.serialize(source, encoded, key)
        source.tensor.fill_(7)
    torch_dev.synchronize()
    processor = AsyncSerdeProcessor(serializer, Fp8QuantizationDeserializer())
    try:
        context = (
            patch(
                "lmcache.v1.distributed.serde.async_processor.torch_dev.current_stream"
            )
            if fence_failure
            else nullcontext()
        )
        with context as stream:
            if fence_failure:
                assert stream is not None
                stream.return_value.synchronize.side_effect = RuntimeError(
                    "injected GPU fence failure"
                )
            if operation == "serialize":
                task = processor.submit_serialize([source], [encoded], [key])
                fd = processor.get_serialize_event_fd()
                query_result = processor.query_serialize_result
                query_outcome = processor.query_serialize_outcome
            else:
                task = processor.submit_deserialize([encoded], [source], [key])
                fd = processor.get_deserialize_event_fd()
                query_result = processor.query_deserialize_result
                query_outcome = processor.query_deserialize_outcome
            assert select.select([fd], [], [], 5)[0]
            if fence_failure:
                with pytest.raises(RuntimeError, match="outcome"):
                    query_result(task)
            assert query_outcome(task) == SerdeTaskOutcome(
                not fence_failure, not fence_failure
            )
            assert query_outcome(task) is None
        assert processor.close_with_outcome() is not fence_failure
        if not fence_failure and operation == "deserialize":
            assert torch.all(source.tensor == 2)
    finally:
        processor.close()
        # The injected failure tests reporting; establish real quiescence before
        # freeing these standalone allocations outside the retained L1 contract.
        torch_dev.synchronize()
        allocator.free(source)
        allocator.free(encoded)
        allocator.close()
