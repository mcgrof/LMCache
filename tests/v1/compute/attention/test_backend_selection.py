# SPDX-License-Identifier: Apache-2.0
"""Keep optional FlashInfer sparse APIs out of the Triton path."""

# Standard
from types import SimpleNamespace
import sys

# First Party
from lmcache.v1.compute.attention import triton_sparse, utils


def test_cuda_triton_does_not_import_flashinfer_sparse(monkeypatch):
    monkeypatch.setattr(utils, "_is_rocm", lambda: False)
    monkeypatch.setattr(utils, "_flashinfer_available", lambda: True)
    monkeypatch.setitem(
        sys.modules, "lmcache.v1.compute.attention.flash_infer_sparse", None
    )
    monkeypatch.setattr(triton_sparse, "torch_device_type", "cuda")
    monkeypatch.setattr(triton_sparse.torch_dev, "current_device", lambda: 0)
    impl = SimpleNamespace(num_heads=14, num_kv_heads=2, head_size=64, scale=0.125)
    attention = SimpleNamespace(impl=impl)

    backend = utils.infer_attn_backend_from_vllm(attention, enable_sparse=True)

    assert isinstance(backend, triton_sparse.LMCTritonSparseBackend)
    assert backend.vllm_attn is attention
    assert (backend.num_qo_heads, backend.num_kv_heads, backend.head_dim) == (14, 2, 64)
