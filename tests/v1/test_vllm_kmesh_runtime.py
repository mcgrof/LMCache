# SPDX-License-Identifier: Apache-2.0

# Standard
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import kmesh_runtime
from lmcache.integration.vllm.kmesh_runtime import build_kmesh_runtime_evidence
from lmcache.integration.vllm.target_profile import (
    COLLECTIVE_LOCKSTEP_V1,
    SERIAL_PIC_V1,
)
from lmcache.v1.config import LMCacheEngineConfig


def _lmcache_config() -> LMCacheEngineConfig:
    return cast(
        LMCacheEngineConfig,
        SimpleNamespace(
            chunk_size=256,
            blend_check_layers=[1],
            blend_recompute_ratios=[0.15],
            enable_blending=True,
            extra_config={"enable_sparse": True},
            save_unfull_chunk=True,
            use_layerwise=True,
            lmcache_instance_id="role-local-instance",
            max_local_cpu_size=1,
        ),
    )


def _vllm_config() -> SimpleNamespace:
    hf_text_config = SimpleNamespace(
        layer_types=["full_attention"] * 24,
        max_position_embeddings=32768,
        rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"},
    )
    model_config = SimpleNamespace(
        model="/models/qwen2.5-0.5b/revision",
        revision="revision",
        tokenizer="/models/qwen2.5-0.5b/revision",
        tokenizer_revision="revision",
        dtype="torch.bfloat16",
        enforce_eager=True,
        max_model_len=2048,
        hf_text_config=hf_text_config,
        get_num_layers=lambda _parallel: 24,
        get_num_kv_heads=lambda _parallel: 2,
        get_head_size=lambda: 64,
    )
    attention_config = SimpleNamespace(
        backend=None,
        backend_per_kind={},
        flash_attn_version=None,
        use_non_causal=False,
        compute_hash=lambda: "attention-config-v1",
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=32),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            enable_chunked_prefill=False,
            disable_hybrid_kv_cache_manager=True,
        ),
        device_config=SimpleNamespace(device_type="cuda", device="cuda:0"),
        attention_config=attention_config,
        kv_transfer_config=SimpleNamespace(
            engine_id="scheduler-role-endpoint",
            kv_load_failure_policy="recompute",
            kv_connector_extra_config={"lmcache_rpc_port": 0},
        ),
    )


@pytest.fixture(autouse=True)
def _accept_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kmesh_runtime, "validate_target_profile", lambda *a, **k: None)


def test_runtime_evidence_is_identical_across_role_local_state() -> None:
    scheduler_vllm = _vllm_config()
    worker_vllm = deepcopy(scheduler_vllm)
    worker_vllm.device_config.device = "cuda:7"
    worker_vllm.kv_transfer_config.engine_id = "worker-role-endpoint"
    worker_vllm.kv_transfer_config.kv_connector_extra_config = {
        "lmcache_rpc_port": 65535
    }
    scheduler_lmcache = _lmcache_config()
    worker_lmcache = deepcopy(scheduler_lmcache)
    worker_lmcache.lmcache_instance_id = "worker-only-instance"
    worker_lmcache.max_local_cpu_size = 64

    scheduler = build_kmesh_runtime_evidence(
        scheduler_vllm,
        scheduler_lmcache,
        role="scheduler",
    )
    worker = build_kmesh_runtime_evidence(
        worker_vllm,
        worker_lmcache,
        role="worker",
    )

    assert scheduler == worker
    assert scheduler.fingerprint == worker.fingerprint
    serialized = repr(scheduler.to_canonical_payload())
    for forbidden in (
        "scheduler-role-endpoint",
        "worker-role-endpoint",
        "role-local-instance",
        "worker-only-instance",
        "cuda:0",
        "cuda:7",
        "65535",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: setattr(config.model_config, "revision", "other-revision"),
        lambda config: setattr(config.model_config, "get_head_size", lambda: 128),
        lambda config: setattr(config.cache_config, "block_size", 64),
        lambda config: setattr(config.model_config, "max_model_len", 4096),
        lambda config: setattr(config.attention_config, "compute_hash", lambda: "x"),
        lambda config: setattr(config.device_config, "device_type", "cpu"),
        lambda config: setattr(
            config.model_config.hf_text_config,
            "rope_parameters",
            {"rope_theta": 10000.0, "rope_type": "default"},
        ),
        lambda config: setattr(
            config.model_config.hf_text_config,
            "layer_types",
            ["sliding_attention"] * 24,
        ),
    ],
)
def test_every_execution_or_layout_mutation_changes_runtime_fingerprint(
    mutate,
) -> None:
    baseline_config = _vllm_config()
    changed_config = deepcopy(baseline_config)
    mutate(changed_config)

    baseline = build_kmesh_runtime_evidence(
        baseline_config,
        _lmcache_config(),
        role="scheduler",
    )
    changed = build_kmesh_runtime_evidence(
        changed_config,
        _lmcache_config(),
        role="worker",
    )

    assert changed.fingerprint != baseline.fingerprint


@pytest.mark.parametrize("profile", [SERIAL_PIC_V1, COLLECTIVE_LOCKSTEP_V1])
def test_runtime_builder_revalidates_named_profile_with_exact_role(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    calls: list[tuple[object, object, object, object]] = []

    def validate(
        config: object,
        profile: object,
        *,
        role: object,
        vllm_config: object,
    ) -> None:
        calls.append((config, profile, role, vllm_config))

    monkeypatch.setattr(kmesh_runtime, "validate_target_profile", validate)
    lmcache_config = _lmcache_config()
    vllm_config = _vllm_config()

    build_kmesh_runtime_evidence(
        vllm_config,
        lmcache_config,
        role="worker",
        target_profile=profile,
    )

    assert calls == [(lmcache_config, profile, "worker", vllm_config)]


def test_runtime_profile_identity_distinguishes_collective_from_serial() -> None:
    vllm_config = _vllm_config()
    lmcache_config = _lmcache_config()
    serial = build_kmesh_runtime_evidence(
        vllm_config,
        lmcache_config,
        role="worker",
        target_profile=SERIAL_PIC_V1,
    )
    collective = build_kmesh_runtime_evidence(
        vllm_config,
        lmcache_config,
        role="worker",
        target_profile=COLLECTIVE_LOCKSTEP_V1,
    )

    assert serial.fingerprint != collective.fingerprint
    assert COLLECTIVE_LOCKSTEP_V1 in collective.runtime_configuration


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: setattr(config.model_config, "model", ""),
        lambda config: setattr(config.model_config, "revision", True),
        lambda config: setattr(config.cache_config, "block_size", True),
        lambda config: setattr(
            config.attention_config,
            "backend_per_kind",
            {1: "invalid"},
        ),
        lambda config: setattr(
            config.model_config.hf_text_config,
            "layer_types",
            ["full_attention", True],
        ),
    ],
)
def test_runtime_builder_rejects_ambiguous_or_loose_values(mutation) -> None:
    config = _vllm_config()
    mutation(config)

    with pytest.raises((TypeError, ValueError)):
        build_kmesh_runtime_evidence(
            config,
            _lmcache_config(),
            role=cast(Any, "scheduler"),
        )
