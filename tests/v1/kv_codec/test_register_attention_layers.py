# SPDX-License-Identifier: Apache-2.0
"""Connector wiring: attention-layer registration on register_kv_caches.

These tests verify that:
- The LMCacheConnectorV1 register_kv_caches call accepts an
  optional `attention_layers` mapping.
- When provided, each layer is recorded in the asym registry,
  so build_asym_kv_view_for_layer(layer_name, ...) returns an
  AsymKVView when called later.
- Symmetric callers (passing None) are unaffected.

We don't boot a real LMCache engine — just exercise the
registration / lookup contract directly with mocked attention
layers.
"""

# Standard
from dataclasses import dataclass

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.asym_kv_view_builder import (
    ATTENTION_LAYER_REGISTRY,
    build_asym_kv_view_for_layer,
    clear_attention_layer_registry,
    register_attention_layer,
)
from lmcache.v1.storage_backend.naive_serde.asym_serde import AsymKVView


class _AsymAttention:
    kv_cache_dtype = ("auto", "fp8_e4m3")
    _v_scale_float: float = 0.42


class _SymAttention:
    kv_cache_dtype = "fp8_e4m3"


@pytest.fixture(autouse=True)
def _reset_registry():
    """Each test starts with a clean registry."""
    clear_attention_layer_registry()
    yield
    clear_attention_layer_registry()


def test_register_records_layer():
    register_attention_layer("layer.0", _AsymAttention())
    assert "layer.0" in ATTENTION_LAYER_REGISTRY


def test_clear_empties_registry():
    register_attention_layer("layer.0", _AsymAttention())
    clear_attention_layer_registry()
    assert len(ATTENTION_LAYER_REGISTRY) == 0


def test_register_idempotent():
    """Re-registering the same name overrides cleanly."""
    a = _AsymAttention()
    b = _AsymAttention()
    register_attention_layer("layer.0", a)
    register_attention_layer("layer.0", b)
    assert ATTENTION_LAYER_REGISTRY["layer.0"] is b


def test_lookup_returns_view_for_registered_asym_layer():
    register_attention_layer("layer.0", _AsymAttention())
    k = torch.randn(4, 16, 8, 64, dtype=torch.float16)
    v_fp8 = torch.zeros(4, 16, 8, 64, dtype=torch.float8_e4m3fn)
    out = build_asym_kv_view_for_layer("layer.0", kv_layer=(k, v_fp8))
    assert isinstance(out, AsymKVView)
    assert out.k.dtype == torch.float16
    assert out.v_fp8.dtype == torch.float8_e4m3fn


def test_lookup_returns_None_for_unregistered_layer():
    """No registration -> None.  Caller falls back to FP16 path."""
    k = torch.randn(4, 16, 8, 64, dtype=torch.float16)
    v_fp8 = torch.zeros(4, 16, 8, 64, dtype=torch.float8_e4m3fn)
    out = build_asym_kv_view_for_layer("layer.0", kv_layer=(k, v_fp8))
    assert out is None


def test_lookup_returns_None_for_registered_symmetric_layer():
    """Even a registered layer that's symmetric returns None;
    the build_asym_kv_view function gates on capability detection."""
    register_attention_layer("layer.0", _SymAttention())
    k = torch.randn(4, 16, 8, 64, dtype=torch.float16)
    v_fp8 = torch.zeros(4, 16, 8, 64, dtype=torch.float8_e4m3fn)
    out = build_asym_kv_view_for_layer("layer.0", kv_layer=(k, v_fp8))
    assert out is None


def test_register_kv_caches_signature_accepts_attention_layers():
    """The public connector API must expose `attention_layers`
    optional parameter.  Catches signature regression.

    Done via source-text inspection because importing
    lmcache.integration.vllm.lmcache_connector_v1 requires `vllm`
    which isn't available in CPU-only test environments.  When vLLM
    IS available, `inspect.signature` would be the real way."""
    # Standard
    from pathlib import Path

    src = Path(
        "lmcache/integration/vllm/lmcache_connector_v1.py"
    ).read_text()
    assert "attention_layers" in src, (
        "lmcache_connector_v1.py must reference `attention_layers` "
        "for the asymmetric KV plumbing hook"
    )
    # Default of None — present in the def line.
    assert "attention_layers: Optional[dict[str, Any]] = None" in src


def test_v1_adapter_register_kv_caches_signature():
    """vllm_v1_adapter mirrors the connector signature."""
    # Standard
    from pathlib import Path

    src = Path(
        "lmcache/integration/vllm/vllm_v1_adapter.py"
    ).read_text()
    assert "attention_layers" in src
    assert "register_attention_layer" in src, (
        "vllm_v1_adapter must call register_attention_layer when "
        "attention_layers is provided to register_kv_caches"
    )


def test_multiple_layers_registered_independently():
    """Many layers, each registered with its own attention object."""
    n_layers = 16
    for i in range(n_layers):
        register_attention_layer(f"layer.{i}", _AsymAttention())
    assert len(ATTENTION_LAYER_REGISTRY) == n_layers
    # Spot-check lookup.
    k = torch.randn(4, 16, 8, 64, dtype=torch.float16)
    v_fp8 = torch.zeros(4, 16, 8, 64, dtype=torch.float8_e4m3fn)
    for i in [0, 7, 15]:
        out = build_asym_kv_view_for_layer(
            f"layer.{i}", kv_layer=(k, v_fp8)
        )
        assert isinstance(out, AsymKVView)
