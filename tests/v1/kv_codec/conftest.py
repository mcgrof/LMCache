# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for kv_codec tests."""

# Standard
import os

# Third Party
import pytest
import torch


# Make tests deterministic
@pytest.fixture(autouse=True)
def _seed_torch():
    torch.manual_seed(0xC0DEC)


@pytest.fixture
def fp8_dtype():
    return torch.float8_e4m3fn


@pytest.fixture
def fp8_max(fp8_dtype):
    return float(torch.finfo(fp8_dtype).max)


@pytest.fixture
def small_kv():
    """A minimal (B=1, T=8, H=2, D=16) FP16 KV pair for fast tests."""
    k = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    v = torch.randn(1, 8, 2, 16, dtype=torch.float16)
    return k, v


@pytest.fixture
def paged_kv():
    """A paged-layout (n_pages=4, page_size=8, n_heads=2, head_dim=16)
    FP16 KV pair, used for PER_PAGE_HEAD tests."""
    n_pages, page_size, n_heads, head_dim = 4, 8, 2, 16
    k = torch.randn(n_pages, page_size, n_heads, head_dim, dtype=torch.float16)
    v = torch.randn(n_pages, page_size, n_heads, head_dim, dtype=torch.float16)
    return k, v
