# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for layerwise collective buffer preflight."""

# Standard
from typing import Any
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMBufferLayerwiseGPUConnector,
)
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import (
    TensorMemoryAllocator,
)
from lmcache.v1.memory_management import AddressManager


def _new_cpu_connector() -> VLLMBufferLayerwiseGPUConnector:
    # Construction normally creates CUDA streams. The preflight itself only
    # needs the real CPU-fallback allocator chain, so inert stream objects keep
    # this test independent of accelerator availability.
    with patch(
        "lmcache.v1.gpu_connector.gpu_connectors.torch.cuda.Stream",
        return_value=object(),
    ):
        return VLLMBufferLayerwiseGPUConnector(
            hidden_dim_size=8,
            num_layers=2,
            use_gpu=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


def _kv_caches() -> list[torch.Tensor]:
    # Per-layer [K/V, blocks, heads, block tokens, head size]. The resulting
    # 512-token layer occupies 32 KiB, or eight AddressManager alignment units.
    return [torch.empty((2, 32, 2, 16, 4)) for _ in range(2)]


def _initialize_allocator_chain(
    connector: VLLMBufferLayerwiseGPUConnector,
) -> tuple[GPUMemoryAllocator, TensorMemoryAllocator, AddressManager]:
    assert connector.preflight_blend_group_capacity(
        _kv_caches(), num_tokens=64, request_count=4
    )

    gpu_allocator = connector.gpu_buffer_allocator
    assert isinstance(gpu_allocator, GPUMemoryAllocator)
    tensor_allocator = gpu_allocator.allocator
    assert isinstance(tensor_allocator, TensorMemoryAllocator)
    address_manager = tensor_allocator.address_manager
    assert isinstance(address_manager, AddressManager)
    return gpu_allocator, tensor_allocator, address_manager


def test_collective_preflight_uses_real_allocator_chain_without_reserving() -> None:
    connector = _new_cpu_connector()
    gpu_allocator, tensor_allocator, address_manager = _initialize_allocator_chain(
        connector
    )

    assert tensor_allocator.num_active_allocations == 0
    assert tensor_allocator.total_allocated_size == 0
    assert address_manager.get_free_size() == 32 * 1024

    # Four requests require eight ping-pong buffers and exactly fill the arena.
    assert connector.preflight_blend_group_capacity(
        _kv_caches(), num_tokens=64, request_count=4
    )
    # Five requests need ten buffers and cannot fit. Neither query reserves.
    assert not connector.preflight_blend_group_capacity(
        _kv_caches(), num_tokens=64, request_count=5
    )
    assert tensor_allocator.num_active_allocations == 0
    assert tensor_allocator.total_allocated_size == 0
    assert address_manager.get_free_size() == 32 * 1024

    buffers = gpu_allocator.batched_allocate(
        connector.get_shape(64), torch.float32, batch_size=8
    )
    assert buffers is not None
    assert len(buffers) == 8
    assert tensor_allocator.num_active_allocations == 8
    gpu_allocator.batched_free(buffers)
    assert tensor_allocator.num_active_allocations == 0
    assert gpu_allocator.memcheck()


def test_collective_preflight_accounts_for_fragmented_capacity() -> None:
    connector = _new_cpu_connector()
    gpu_allocator, tensor_allocator, address_manager = _initialize_allocator_chain(
        connector
    )
    buffers = gpu_allocator.batched_allocate(
        connector.get_shape(64), torch.float32, batch_size=8
    )
    assert buffers is not None

    # Leave four nonadjacent 4 KiB holes. Their total 16 KiB equals two 8 KiB
    # buffers, but neither required buffer can fit in any individual range.
    gpu_allocator.batched_free(buffers[::2])
    assert address_manager.get_free_size() == 16 * 1024
    assert tensor_allocator.total_allocated_size == 16 * 1024
    assert tensor_allocator.num_active_allocations == 4

    assert not connector.preflight_blend_group_capacity(
        _kv_caches(), num_tokens=128, request_count=1
    )
    assert address_manager.get_free_size() == 16 * 1024
    assert tensor_allocator.total_allocated_size == 16 * 1024
    assert tensor_allocator.num_active_allocations == 4

    gpu_allocator.batched_free(buffers[1::2])
    assert tensor_allocator.num_active_allocations == 0
    assert address_manager.get_free_size() == 32 * 1024
    assert gpu_allocator.memcheck()


@pytest.mark.parametrize("num_tokens", [True, 1.0, 0, -1])
def test_collective_preflight_rejects_nonexact_token_counts(num_tokens: Any) -> None:
    connector = _new_cpu_connector()

    with pytest.raises(ValueError, match="num_tokens must be a positive exact integer"):
        connector.preflight_blend_group_capacity(
            _kv_caches(), num_tokens=num_tokens, request_count=1
        )


@pytest.mark.parametrize("request_count", [True, 1.0, 0, -1])
def test_collective_preflight_rejects_nonexact_request_counts(
    request_count: Any,
) -> None:
    connector = _new_cpu_connector()

    with pytest.raises(
        ValueError, match="request_count must be a positive exact integer"
    ):
        connector.preflight_blend_group_capacity(
            _kv_caches(), num_tokens=1, request_count=request_count
        )
