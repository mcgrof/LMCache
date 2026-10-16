# SPDX-License-Identifier: Apache-2.0
"""Canonical runtime evidence for the bounded KMesh PIC profiles."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import Any, cast
import json

# Third Party
import torch

# First Party
from lmcache.integration.vllm.kmesh import KMeshRuntimeEvidence
from lmcache.integration.vllm.target_profile import (
    KMESH_PIC_V1_PROFILES,
    SERIAL_PIC_V1,
    validate_target_profile,
)
from lmcache.v1.config import LMCacheEngineConfig


def _required(owner: object, owner_name: str, field_name: str) -> object:
    if owner is None or not hasattr(owner, field_name):
        raise ValueError(f"{owner_name}.{field_name} must be present")
    return getattr(owner, field_name)


def _plain_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a nonempty trimmed plain string")
    return value


def _plain_positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive plain integer")
    return value


def _canonical_value(value: object, field_name: str) -> object:
    """Freeze JSON-safe configuration without repr- or subclass-based aliases."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if type(value) in (list, tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        return [
            _canonical_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(sequence)
        ]
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        canonical: dict[str, object] = {}
        keys: list[str] = []
        for raw_key in mapping:
            if type(raw_key) is not str or not raw_key:
                raise TypeError(f"{field_name} keys must be nonempty plain strings")
            keys.append(raw_key)
        for key in sorted(keys):
            canonical[key] = _canonical_value(mapping[key], f"{field_name}.{key}")
        return canonical
    raise TypeError(
        f"{field_name} must contain only exact JSON scalar, list, tuple, or dict values"
    )


def _canonical_json(value: object, field_name: str) -> str:
    return json.dumps(
        _canonical_value(value, field_name),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalized_dtype(value: object, field_name: str) -> str:
    if type(value) is str:
        normalized = value
    elif type(value) is torch.dtype:
        normalized = str(value).removeprefix("torch.")
    else:
        raise TypeError(f"{field_name} must be a plain string or exact torch dtype")
    return _plain_string(normalized, field_name)


def _layer_types(hf_text_config: object) -> tuple[str, ...]:
    raw_layer_types = _required(
        hf_text_config,
        "model_config.hf_text_config",
        "layer_types",
    )
    if not isinstance(raw_layer_types, Sequence) or isinstance(
        raw_layer_types, (str, bytes)
    ):
        raise TypeError("model_config.hf_text_config.layer_types must be a sequence")
    layer_types = tuple(
        _plain_string(layer_type, f"layer_types[{index}]")
        for index, layer_type in enumerate(raw_layer_types)
    )
    if not layer_types:
        raise ValueError("model_config.hf_text_config.layer_types must be nonempty")
    return layer_types


def _attention_backend_payload(attention_config: object) -> dict[str, object]:
    compute_hash = _required(attention_config, "attention_config", "compute_hash")
    if not callable(compute_hash):
        raise TypeError("attention_config.compute_hash must be callable")
    backend = _required(attention_config, "attention_config", "backend")
    backend_name = getattr(backend, "name", backend)
    if backend_name is not None:
        backend_name = _plain_string(backend_name, "attention_config.backend")
    return {
        "backend": backend_name,
        "backend_per_kind": _required(
            attention_config,
            "attention_config",
            "backend_per_kind",
        ),
        "compute_hash": compute_hash(),
        "flash_attn_version": _required(
            attention_config,
            "attention_config",
            "flash_attn_version",
        ),
        "use_prefill_decode_attention": _required(
            attention_config,
            "attention_config",
            "use_prefill_decode_attention",
        ),
    }


def build_kmesh_runtime_evidence(
    vllm_config: Any,
    lmcache_config: LMCacheEngineConfig,
    *,
    role: object,
    target_profile: object = SERIAL_PIC_V1,
) -> KMeshRuntimeEvidence:
    """Revalidate and freeze cache-compatibility evidence for either role.

    Role-local IPC endpoints, LMCache instance IDs, memory capacities, request
    state, generation counters, and segment offsets are intentionally absent.
    The result therefore matches between the scheduler and worker precisely
    when their execution and KV-layout configuration matches.
    """
    if type(target_profile) is not str or target_profile not in KMESH_PIC_V1_PROFILES:
        raise ValueError(f"unsupported KMesh PIC target profile {target_profile!r}")
    validate_target_profile(
        lmcache_config,
        target_profile,
        role=role,
        vllm_config=vllm_config,
    )

    model_config = _required(vllm_config, "vllm_config", "model_config")
    cache_config = _required(vllm_config, "vllm_config", "cache_config")
    parallel_config = _required(vllm_config, "vllm_config", "parallel_config")
    scheduler_config = _required(vllm_config, "vllm_config", "scheduler_config")
    device_config = _required(vllm_config, "vllm_config", "device_config")
    attention_config = _required(vllm_config, "vllm_config", "attention_config")
    kv_transfer_config = _required(
        vllm_config,
        "vllm_config",
        "kv_transfer_config",
    )
    hf_text_config = _required(
        model_config,
        "model_config",
        "hf_text_config",
    )

    model_id = _plain_string(
        _required(model_config, "model_config", "model"),
        "model_config.model",
    )
    model_revision = _plain_string(
        _required(model_config, "model_config", "revision"),
        "model_config.revision",
    )
    tokenizer_id = _plain_string(
        _required(model_config, "model_config", "tokenizer"),
        "model_config.tokenizer",
    )
    tokenizer_revision = _plain_string(
        _required(model_config, "model_config", "tokenizer_revision"),
        "model_config.tokenizer_revision",
    )
    model_dtype = _normalized_dtype(
        _required(model_config, "model_config", "dtype"),
        "model_config.dtype",
    )
    configured_kv_dtype = _plain_string(
        _required(cache_config, "cache_config", "cache_dtype"),
        "cache_config.cache_dtype",
    )
    effective_kv_dtype = (
        model_dtype if configured_kv_dtype == "auto" else configured_kv_dtype
    )

    tensor_parallel_size = _plain_positive_int(
        _required(
            parallel_config,
            "parallel_config",
            "tensor_parallel_size",
        ),
        "parallel_config.tensor_parallel_size",
    )
    pipeline_parallel_size = _plain_positive_int(
        _required(
            parallel_config,
            "parallel_config",
            "pipeline_parallel_size",
        ),
        "parallel_config.pipeline_parallel_size",
    )
    get_num_layers = _required(model_config, "model_config", "get_num_layers")
    get_num_kv_heads = _required(model_config, "model_config", "get_num_kv_heads")
    get_head_size = _required(model_config, "model_config", "get_head_size")
    if not callable(get_num_layers):
        raise TypeError("model layer resolver must be callable")
    if not callable(get_num_kv_heads):
        raise TypeError("model KV-head resolver must be callable")
    if not callable(get_head_size):
        raise TypeError("model head-size resolver must be callable")
    num_layers = _plain_positive_int(
        get_num_layers(parallel_config),
        "model_config.get_num_layers()",
    )
    num_kv_heads = _plain_positive_int(
        get_num_kv_heads(parallel_config),
        "model_config.get_num_kv_heads()",
    )
    head_size = _plain_positive_int(
        get_head_size(),
        "model_config.get_head_size()",
    )
    block_size = _plain_positive_int(
        _required(cache_config, "cache_config", "block_size"),
        "cache_config.block_size",
    )
    chunk_size = _plain_positive_int(lmcache_config.chunk_size, "chunk_size")
    layer_types = _layer_types(hf_text_config)
    attention_kind = "+".join(layer_types)

    rope_parameters = getattr(hf_text_config, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = _required(
            hf_text_config,
            "model_config.hf_text_config",
            "rope_scaling",
        )
    rope_configuration = _canonical_json(
        {
            "max_position_embeddings": _required(
                hf_text_config,
                "model_config.hf_text_config",
                "max_position_embeddings",
            ),
            "rope_parameters": rope_parameters,
        },
        "rope_configuration",
    )

    backend_payload = _attention_backend_payload(attention_config)
    backend_name = backend_payload["backend"] or "auto"
    device_type = _plain_string(
        _required(device_config, "device_config", "device_type"),
        "device_config.device_type",
    )
    attention_backend = f"{backend_name}:{device_type}"
    attention_backend_configuration = _canonical_json(
        backend_payload,
        "attention_backend_configuration",
    )

    kv_layout = _canonical_json(
        {
            "block_size": block_size,
            "chunk_size": chunk_size,
            "head_size": head_size,
            "kv_heads": num_kv_heads,
            "kv_planes": 2,
            "layers": num_layers,
            "layout": "layerwise-separate-kv",
        },
        "kv_layout",
    )
    runtime_configuration = _canonical_json(
        {
            "async_scheduling": _required(
                scheduler_config,
                "scheduler_config",
                "async_scheduling",
            ),
            "blend_check_layers": lmcache_config.blend_check_layers,
            "blend_recompute_ratios": lmcache_config.blend_recompute_ratios,
            "disable_hybrid_kv_cache_manager": _required(
                scheduler_config,
                "scheduler_config",
                "disable_hybrid_kv_cache_manager",
            ),
            "enable_blending": lmcache_config.enable_blending,
            "enable_chunked_prefill": _required(
                scheduler_config,
                "scheduler_config",
                "enable_chunked_prefill",
            ),
            "enforce_eager": _required(
                model_config,
                "model_config",
                "enforce_eager",
            ),
            "extra_config": lmcache_config.extra_config,
            "kv_load_failure_policy": _required(
                kv_transfer_config,
                "kv_transfer_config",
                "kv_load_failure_policy",
            ),
            "max_model_len": _required(
                model_config,
                "model_config",
                "max_model_len",
            ),
            "profile": target_profile,
            "save_unfull_chunk": lmcache_config.save_unfull_chunk,
            "use_layerwise": lmcache_config.use_layerwise,
        },
        "runtime_configuration",
    )
    device_identity = _canonical_json(
        {
            "device_type": device_type,
            "execution_path": "rocm-cuda-compatible",
        },
        "device_identity",
    )

    return KMeshRuntimeEvidence(
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        model_dtype=model_dtype,
        effective_kv_dtype=effective_kv_dtype,
        kv_layout=kv_layout,
        attention_kind=attention_kind,
        rope_configuration=rope_configuration,
        attention_backend=attention_backend,
        attention_backend_configuration=attention_backend_configuration,
        runtime_configuration=runtime_configuration,
        device_identity=device_identity,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )
