# SPDX-License-Identifier: Apache-2.0
"""Admission tests for the narrow KMesh serial-PIC target profile."""

# Standard
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields
from types import SimpleNamespace
from typing import Any, Callable, Optional, cast
from unittest.mock import MagicMock, patch
import inspect
import math
import os
import sys
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.base_service_factory import BaseServiceFactory
from lmcache.integration.vllm.target_profile import (
    SERIAL_PIC_V1,
    SERIAL_PIC_V1_CONTRACT,
    requested_target_profile,
    validate_target_profile,
)
from lmcache.v1.cache_engine import (
    EngineDestroyResult,
    LMCacheEngine,
    LMCacheEngineBuilder,
)
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.manager import LMCacheManager
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache
from lmcache.v1.storage_backend.storage_manager import StorageManager
from tests.v1.instruction_monitoring import (
    InstructionInterruption,
    instruction_offset,
    requires_instruction_monitoring,
)
from tests.v1.utils import create_test_metadata


class _ConstructionHandoffManagerDouble:
    """Minimal exact-token handoff protocol for adapter Manager doubles."""

    _handoff_events: Optional[list[str]] = None

    def _record_handoff(self, event: str) -> None:
        events = self._handoff_events
        if events is not None:
            events.append(event)

    def new_construction_handoff_receipt(self) -> object:
        return object()

    def retain_construction_handoff(self, receipt: object) -> None:
        self._record_handoff("retain")
        receipts: Optional[set[object]] = getattr(
            self,
            "_handoff_receipts",
            None,
        )
        if receipts is None:
            receipts = set()
            self._handoff_receipts = receipts
        receipts.add(receipt)

    def release_construction_handoff(self, receipt: object) -> None:
        self._record_handoff("release")
        receipts: set[object] = getattr(self, "_handoff_receipts", set())
        receipts.discard(receipt)

    def promote_construction_handoff_failure(self, receipt: object) -> None:
        self._record_handoff("promote")
        receipts: set[object] = getattr(self, "_handoff_receipts", set())
        receipts.discard(receipt)


class _InterruptingConstructionFactory(BaseServiceFactory):
    """Real factory surface that interrupts after publishing early owners."""

    def __init__(
        self,
        lookup_client: LookupClientInterface,
        construction_error: BaseException,
        engine: Optional[LMCacheEngine] = None,
    ) -> None:
        self.lookup_client = lookup_client
        self.construction_error = construction_error
        self.engine = engine
        self.metadata = create_test_metadata()

    def get_engine_instance_id(self) -> str:
        """Return the stable public engine identity used for cleanup."""
        return "target-profile-construction-test"

    def get_or_create_metadata(self) -> LMCacheMetadata:
        """Return test metadata."""
        return self.metadata

    def get_or_create_lmcache_engine(
        self,
        result_publisher: Optional[Callable[[Optional[LMCacheEngine]], None]] = None,
    ) -> Optional[LMCacheEngine]:
        """Return the configured optional engine owner."""
        if result_publisher is not None:
            result_publisher(self.engine)
        return self.engine

    def maybe_create_lookup_client(
        self,
        result_publisher: Optional[
            Callable[[Optional[LookupClientInterface]], None]
        ] = None,
    ) -> LookupClientInterface:
        """Publish the lookup client before the injected interrupt."""
        if result_publisher is not None:
            result_publisher(self.lookup_client)
        return self.lookup_client

    def maybe_create_prometheus_logger(self) -> None:
        """Disable Prometheus construction for this lifecycle probe."""
        return None

    def maybe_create_lookup_server(self, result_publisher=None) -> None:
        """Interrupt construction after the lookup client is published."""
        raise self.construction_error

    def maybe_create_offload_server(self) -> None:
        """Disable the excluded offload service."""
        return None

    def maybe_create_runtime_plugin_launcher(self) -> None:
        """Disable the excluded plugin service."""
        return None

    def maybe_create_internal_api_server(
        self,
        lmcache_manager: LMCacheManager,
    ) -> None:
        """Disable the excluded internal API service."""
        return None

    def maybe_create_health_monitor(
        self,
        lmcache_manager: LMCacheManager,
    ) -> None:
        """Disable post-init monitor creation for a constructor probe."""
        return None


class _TargetSchedulerFactory(BaseServiceFactory):
    """Construct a healthy scheduler graph with an optional forbidden owner."""

    def __init__(
        self,
        lookup_client: Optional[LookupClientInterface],
        *,
        lookup_server: Optional[Any] = None,
    ) -> None:
        self.lookup_client = lookup_client
        self.lookup_server = lookup_server
        self.metadata = create_test_metadata()

    def get_engine_instance_id(self) -> str:
        """Return the stable public engine identity used for cleanup."""
        return "target-profile-scheduler-test"

    def get_or_create_metadata(self) -> LMCacheMetadata:
        """Return test metadata."""
        return self.metadata

    def get_or_create_lmcache_engine(
        self,
        result_publisher: Optional[Callable[[Optional[LMCacheEngine]], None]] = None,
    ) -> Optional[LMCacheEngine]:
        """Construct the target scheduler-only graph."""
        if result_publisher is not None:
            result_publisher(None)
        return None

    def maybe_create_lookup_client(
        self,
        result_publisher: Optional[
            Callable[[Optional[LookupClientInterface]], None]
        ] = None,
    ) -> Optional[LookupClientInterface]:
        """Return the scheduler lookup client."""
        if result_publisher is not None:
            result_publisher(self.lookup_client)
        return self.lookup_client

    def maybe_create_prometheus_logger(self) -> None:
        """Disable Prometheus construction for this lifecycle probe."""
        return None

    def maybe_create_lookup_server(self, result_publisher=None) -> Optional[Any]:
        """Return the optional forbidden worker-only lookup server."""
        if result_publisher is not None:
            result_publisher(self.lookup_server)
        return self.lookup_server

    def maybe_create_offload_server(self) -> None:
        """Disable the excluded offload service."""
        return None

    def maybe_create_runtime_plugin_launcher(self) -> None:
        """Disable the excluded plugin service."""
        return None

    def maybe_create_internal_api_server(
        self,
        lmcache_manager: LMCacheManager,
    ) -> None:
        """Disable the excluded internal API service."""
        return None

    def maybe_create_health_monitor(
        self,
        lmcache_manager: LMCacheManager,
    ) -> None:
        """Disable post-init monitor creation for a constructor probe."""
        return None


class _TargetWorkerFactory(_TargetSchedulerFactory):
    """Construct the exact worker graph for engine-cleanup probes."""

    def __init__(self, engine: LMCacheEngine, lookup_server: Any) -> None:
        self.engine = engine
        self.lookup_client = None
        self.lookup_server = lookup_server
        self.metadata = create_test_metadata()

    def get_engine_instance_id(self) -> str:
        """Return the stable worker engine identity used for cleanup."""
        return "target-profile-worker-test"

    def get_or_create_lmcache_engine(
        self,
        result_publisher: Optional[Callable[[Optional[LMCacheEngine]], None]] = None,
    ) -> Optional[LMCacheEngine]:
        """Return the worker engine owner."""
        if result_publisher is not None:
            result_publisher(self.engine)
        return self.engine

    def maybe_create_lookup_client(
        self,
        result_publisher: Optional[
            Callable[[Optional[LookupClientInterface]], None]
        ] = None,
    ) -> Optional[LookupClientInterface]:
        """Disable the scheduler-only lookup client."""
        if result_publisher is not None:
            result_publisher(None)
        return None


def _valid_config() -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_defaults(
        local_cpu=True,
        max_local_cpu_size=1,
        local_disk=None,
        max_local_disk_size=0,
        remote_url=None,
        use_layerwise=True,
        enable_blending=True,
        blend_recompute_ratios=[0.15],
        blend_thresholds=None,
        blend_check_layers=[1],
        retrieve_locations=["LocalCPUBackend"],
        store_location="LocalCPUBackend",
        enable_async_loading=False,
        enable_scheduler_bypass_lookup=False,
        external_lookup_client=None,
        enable_chunk_statistics=False,
        hit_miss_ratio=None,
        internal_api_server_enabled=False,
        enable_controller=False,
        enable_pd=False,
        enable_p2p=False,
        runtime_plugin_locations=None,
        storage_plugins=None,
        remote_storage_plugins=None,
        gds_path=None,
        gds_buffer_size=None,
        use_gds=False,
        maru_path=None,
        nixl_backends=None,
        nixl_buffer_size=None,
        nixl_buffer_device=None,
        transfer_channel=None,
        lookup_server_worker_ids=None,
        extra_config={"enable_sparse": True},
        save_unfull_chunk=True,
    )
    config.validate()
    return config


def _valid_vllm_config(
    *,
    profile: Any = SERIAL_PIC_V1,
) -> SimpleNamespace:
    connector_extra_config = {
        "kmesh_target_profile": profile,
        "use_native": False,
    }
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
            tokenizer="Qwen/Qwen2.5-0.5B-Instruct",
            tokenizer_revision="7ae557604adf67be50417f59c2c2f167def9a775",
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=2048,
            use_mla=False,
            hf_text_config=SimpleNamespace(
                layer_types=["full_attention"] * 24,
                max_position_embeddings=32768,
                rope_parameters={
                    "rope_theta": 1_000_000.0,
                    "rope_type": "default",
                },
            ),
            get_num_layers=lambda _parallel_config: 24,
            get_num_kv_heads=lambda _parallel_config: 2,
            get_head_size=lambda: 64,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=32),
        speculative_config=None,
        device_config=SimpleNamespace(device="cpu", device_type="cuda"),
        attention_config=SimpleNamespace(
            backend=None,
            backend_per_kind={},
            compute_hash=lambda: "serial-pic-v1-attention",
            flash_attn_version=None,
            use_non_causal=False,
        ),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            engine_id="kmesh-profile-test",
            kv_load_failure_policy="recompute",
            kv_connector_extra_config=connector_extra_config,
            get_from_extra_config=lambda key, default: connector_extra_config.get(
                key, default
            ),
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank=0,
            data_parallel_rank_local=0,
            nnodes=1,
            node_rank=0,
            world_size=1,
            rank=0,
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
            enable_expert_parallel=False,
            enable_elastic_ep=False,
            enable_fault_tolerance=False,
            disable_custom_all_reduce=True,
            distributed_executor_backend="uni",
            enable_dbo=False,
            ubatch_size=0,
            worker_cls="vllm.v1.worker.gpu_worker.Worker",
            worker_extension_cls="",
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            disable_hybrid_kv_cache_manager=True,
            enable_chunked_prefill=False,
        ),
    )


@pytest.mark.parametrize("role", ["scheduler", "worker"])
def test_serial_pic_v1_accepts_exact_role_local_profile(role: str) -> None:
    validate_target_profile(
        _valid_config(),
        SERIAL_PIC_V1,
        role=role,
        vllm_config=_valid_vllm_config(),
    )


def test_serial_pic_v1_builds_nonpaged_receipt_allocator_without_gds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Standard
    import asyncio

    # Third Party
    import torch

    # First Party
    from lmcache.v1.memory_allocators.mixed_memory_allocator import (
        MixedMemoryAllocator,
    )
    from lmcache.v1.memory_allocators.tensor_memory_allocator import (
        TensorMemoryAllocator,
    )
    import lmcache.v1.memory_management as memory_management
    import lmcache.v1.storage_backend as storage_backend_module

    config = _valid_config()
    config.max_local_cpu_size = SERIAL_PIC_V1_CONTRACT.target_chunk_bytes / 1024**3
    metadata = create_test_metadata()
    metadata.role = "worker"
    validate_target_profile(
        config,
        SERIAL_PIC_V1,
        role="worker",
    )

    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *_args, **_kwargs: torch.zeros(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        memory_management,
        "_free_cpu_memory",
        lambda *_args, **_kwargs: None,
    )
    forbidden_gds = MagicMock(
        side_effect=AssertionError("serial_pic_v1 constructed GDS")
    )
    monkeypatch.setattr(storage_backend_module, "GdsBackend", forbidden_gds)

    loop = asyncio.new_event_loop()
    backends: Any = None
    try:
        backends = storage_backend_module.CreateStorageBackends(
            config,
            metadata,
            loop,
        )

        assert tuple(backends) == ("LocalCPUBackend",)
        allocator = backends["LocalCPUBackend"].get_memory_allocator()
        assert isinstance(allocator, MixedMemoryAllocator)
        assert allocator.supports_allocation_receipts()
        assert isinstance(allocator.pin_allocator, TensorMemoryAllocator)
        forbidden_gds.assert_not_called()
    finally:
        if backends is not None:
            for backend in backends.values():
                backend.close()
        loop.close()


def test_serial_pic_v1_accepts_real_resolved_parallel_config() -> None:
    vllm = pytest.importorskip("vllm.config")
    parallel_config = vllm.ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        distributed_executor_backend="uni",
        disable_custom_all_reduce=True,
        enable_dbo=False,
        ubatch_size=0,
        worker_cls="vllm.v1.worker.gpu_worker.Worker",
    )
    missing_fields = sorted(
        field_name
        for field_name in SERIAL_PIC_V1_CONTRACT.exact_parallel
        if not hasattr(parallel_config, field_name)
    )
    if missing_fields:
        pytest.skip(
            "installed vLLM predates the paired KMesh profile fields: "
            f"{missing_fields!r}"
        )

    # ParallelConfig resolves the single-process local rank from None to zero
    # during post-init.  vLLM platform resolution separately replaces the raw
    # "auto" worker selection with this built-in GPU worker before LMCache
    # validation.  Guard both resolved values with the real dataclass.
    assert parallel_config.data_parallel_rank_local == 0
    assert parallel_config.worker_cls == "vllm.v1.worker.gpu_worker.Worker"
    vllm_config = _valid_vllm_config()
    vllm_config.parallel_config = parallel_config
    validate_target_profile(
        _valid_config(),
        SERIAL_PIC_V1,
        role="worker",
        vllm_config=vllm_config,
    )


def test_serial_pic_v1_contract_covers_every_public_config_field() -> None:
    public_fields = {field.name for field in fields(LMCacheEngineConfig)}
    reviewed_fields = set(SERIAL_PIC_V1_CONTRACT.exact_config)
    reviewed_fields.update(SERIAL_PIC_V1_CONTRACT.dynamic_config)

    assert reviewed_fields == public_fields


def _mutated_exact_value(value: Any) -> Any:
    if value is None:
        return "unexpected"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return 1.0 if math.isinf(value) else value + 1.0
    if isinstance(value, str):
        return f"{value}-unexpected"
    if isinstance(value, (list, tuple)):
        return [*value, "unexpected"]
    if isinstance(value, Mapping):
        return {**value, "unexpected": True}
    raise AssertionError(f"no mutation defined for {value!r}")


@pytest.mark.parametrize(
    "field_name",
    sorted(SERIAL_PIC_V1_CONTRACT.exact_config),
)
def test_serial_pic_v1_rejects_mutation_of_every_exact_config_field(
    field_name: str,
) -> None:
    config = _valid_config()
    expected = SERIAL_PIC_V1_CONTRACT.exact_config[field_name]
    setattr(config, field_name, _mutated_exact_value(expected))

    with pytest.raises(ValueError, match=field_name):
        validate_target_profile(config, SERIAL_PIC_V1)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("blend_check_layers", [True]),
        ("extra_config", {"enable_sparse": 1}),
    ],
)
def test_serial_pic_v1_rejects_nested_equality_aliases(
    field_name: str,
    value: Any,
) -> None:
    config = _valid_config()
    setattr(config, field_name, value)
    config.validate()

    with pytest.raises(ValueError, match=field_name):
        validate_target_profile(config, SERIAL_PIC_V1)


@pytest.mark.parametrize(
    "max_local_cpu_size",
    [
        SERIAL_PIC_V1_CONTRACT.target_chunk_bytes / 1024**3,
        0.25,
        1,
        32.0,
    ],
)
def test_serial_pic_v1_accepts_dynamic_positive_local_cpu_size(
    max_local_cpu_size: float,
) -> None:
    config = _valid_config()
    config.max_local_cpu_size = max_local_cpu_size

    validate_target_profile(config, SERIAL_PIC_V1)


@pytest.mark.parametrize(
    "max_local_cpu_size",
    [
        None,
        True,
        0,
        -1,
        "1",
        5e-324,
        (SERIAL_PIC_V1_CONTRACT.target_chunk_bytes - 1) / 1024**3,
        float("nan"),
        float("inf"),
    ],
)
def test_serial_pic_v1_rejects_invalid_local_cpu_size(
    max_local_cpu_size: Any,
) -> None:
    config = _valid_config()
    config.max_local_cpu_size = max_local_cpu_size

    with pytest.raises(ValueError, match="max_local_cpu_size"):
        validate_target_profile(config, SERIAL_PIC_V1)


@pytest.mark.parametrize("instance_id", ["serial-pic-test", "x"])
def test_serial_pic_v1_accepts_dynamic_nonempty_instance_id(instance_id: str) -> None:
    config = _valid_config()
    config.lmcache_instance_id = instance_id

    validate_target_profile(config, SERIAL_PIC_V1)


@pytest.mark.parametrize("instance_id", [None, "", " whitespace ", 1, False])
def test_serial_pic_v1_rejects_invalid_instance_id(instance_id: Any) -> None:
    config = _valid_config()
    config.lmcache_instance_id = instance_id

    with pytest.raises(ValueError, match="lmcache_instance_id"):
        validate_target_profile(config, SERIAL_PIC_V1)


def test_serial_pic_v1_builds_executable_cacheblend_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    from lmcache.v1.compute.blend import blender as blender_module

    layerwise_model = object()
    monkeypatch.setattr(
        blender_module,
        "infer_model_from_vllm",
        lambda *_args: layerwise_model,
    )
    vllm_model = SimpleNamespace(model=SimpleNamespace(layers=[object(), object()]))

    blender = blender_module.LMCBlender(
        cache_engine=object(),
        gpu_connector=object(),
        vllm_model=vllm_model,
        config=_valid_config(),
    )

    assert blender.layerwise_model is layerwise_model
    assert blender.common_metadata.check_layers == [1]
    assert blender.common_metadata.recomp_ratios == [0.15]
    assert blender.common_metadata.thresholds is None
    assert 1 in blender.common_metadata.check_layers
    assert blender.common_metadata.recomp_ratios is not None


def test_target_profile_is_explicitly_opt_in() -> None:
    # None is deliberately a no-op even for the ordinary default config.
    validate_target_profile(LMCacheEngineConfig.from_defaults(), None)


@pytest.mark.parametrize("profile", ["", False, 1, [SERIAL_PIC_V1]])
def test_raw_target_profile_requires_nonempty_string(profile: Any) -> None:
    with pytest.raises(ValueError, match="must be a non-empty string"):
        requested_target_profile(_valid_vllm_config(profile=profile))


def test_unknown_target_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported LMCache target profile"):
        validate_target_profile(_valid_config(), "serial_pic_v2")


def test_explicit_profile_must_match_raw_activation() -> None:
    with pytest.raises(ValueError, match="conflicts with raw"):
        validate_target_profile(
            _valid_config(),
            SERIAL_PIC_V1,
            vllm_config=_valid_vllm_config(profile="serial_pic_v2"),
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("enable_async_loading", True, "enable_async_loading"),
        (
            "enable_scheduler_bypass_lookup",
            True,
            "enable_scheduler_bypass_lookup",
        ),
        ("external_lookup_client", "mooncakestore://test", "external_lookup_client"),
        ("enable_chunk_statistics", True, "enable_chunk_statistics"),
        ("hit_miss_ratio", 0.5, "hit_miss_ratio"),
        ("internal_api_server_enabled", True, "internal_api_server_enabled"),
        ("enable_controller", True, "enable_controller"),
        ("enable_pd", True, "enable_pd"),
        ("enable_p2p", True, "enable_p2p"),
        ("local_cpu_dmabuf", "udmabuf", "local_cpu_dmabuf"),
        ("pd_data_path", "raw_block", "pd_data_path"),
        ("run_script_api_enabled", True, "run_script_api_enabled"),
        ("use_layerwise", False, "use_layerwise"),
        ("enable_blending", False, "enable_blending"),
        ("local_cpu", False, "local_cpu"),
        ("max_local_cpu_size", 0, "max_local_cpu_size"),
        ("local_disk", "/tmp/not-admitted", "local_disk"),
        ("max_local_disk_size", 1, "max_local_disk_size"),
        ("remote_url", "redis://not-admitted", "remote_url"),
        ("retrieve_locations", None, "retrieve_locations"),
        (
            "retrieve_locations",
            ["LocalCPUBackend", "LocalDiskBackend"],
            "retrieve_locations",
        ),
        ("store_location", None, "store_location"),
        ("runtime_plugin_locations", ["plugin.py"], "runtime_plugin_locations"),
        ("storage_plugins", ["custom"], "storage_plugins"),
        ("remote_storage_plugins", ["redis"], "remote_storage_plugins"),
        ("gds_path", "/tmp/not-admitted", "gds_path"),
        ("gds_buffer_size", 1024, "gds_buffer_size"),
        ("use_gds", True, "use_gds"),
        ("maru_path", "/tmp/not-admitted", "maru_path"),
        ("nixl_backends", ["POSIX"], "nixl_backends"),
        ("nixl_buffer_size", 1024, "nixl_buffer_size"),
        ("nixl_buffer_device", "cpu", "nixl_buffer_device"),
        ("transfer_channel", "nixl", "transfer_channel"),
        ("lookup_server_worker_ids", [1], "lookup_server_worker_ids"),
    ],
)
def test_serial_pic_v1_rejects_out_of_profile_config(
    field_name: str,
    value: Any,
    message: str,
) -> None:
    config = _valid_config()
    setattr(config, field_name, value)

    with pytest.raises(ValueError, match=message):
        validate_target_profile(config, SERIAL_PIC_V1, role="worker")


@pytest.mark.parametrize(
    ("extra_config", "message"),
    [
        ({}, "enable_sparse"),
        ({"enable_sparse": False}, "enable_sparse"),
        (
            {"enable_sparse": True, "enable_nixl_storage": True},
            "enable_nixl_storage",
        ),
        (
            {"enable_sparse": True, "audit_backend_enabled": True},
            "audit_backend_enabled",
        ),
    ],
)
def test_serial_pic_v1_rejects_wrong_sparse_or_backend_wrapper_extra_config(
    extra_config: dict[str, Any],
    message: str,
) -> None:
    config = _valid_config()
    config.extra_config = extra_config

    with pytest.raises(ValueError, match=message):
        validate_target_profile(config, SERIAL_PIC_V1)


@pytest.mark.parametrize(
    ("object_name", "field_name", "value", "message"),
    [
        ("kv", "kv_role", "kv_consumer", "kv_role"),
        ("kv", "engine_id", None, "engine_id"),
        ("kv", "kv_parallel_size", 2, "kv_parallel_size"),
        ("kv", "kv_rank", 0, "kv_rank"),
        ("kv_extra", "lmcache_rpc_port", True, "lmcache_rpc_port"),
        ("parallel", "pipeline_parallel_size", 2, "pipeline_parallel_size"),
        ("parallel", "tensor_parallel_size", 2, "tensor_parallel_size"),
        (
            "parallel",
            "prefill_context_parallel_size",
            2,
            "prefill_context_parallel_size",
        ),
        (
            "parallel",
            "decode_context_parallel_size",
            2,
            "decode_context_parallel_size",
        ),
        ("parallel", "data_parallel_size", 2, "data_parallel_size"),
        ("parallel", "data_parallel_size_local", 2, "data_parallel_size_local"),
        ("parallel", "data_parallel_rank", 1, "data_parallel_rank"),
        ("parallel", "data_parallel_rank_local", None, "data_parallel_rank_local"),
        ("parallel", "nnodes", 2, "nnodes"),
        ("parallel", "node_rank", 1, "node_rank"),
        ("parallel", "world_size", 2, "world_size"),
        ("parallel", "rank", 1, "rank"),
        (
            "parallel",
            "data_parallel_external_lb",
            True,
            "data_parallel_external_lb",
        ),
        (
            "parallel",
            "data_parallel_hybrid_lb",
            True,
            "data_parallel_hybrid_lb",
        ),
        ("parallel", "enable_expert_parallel", True, "enable_expert_parallel"),
        ("parallel", "enable_elastic_ep", True, "enable_elastic_ep"),
        ("parallel", "enable_fault_tolerance", True, "enable_fault_tolerance"),
        (
            "parallel",
            "disable_custom_all_reduce",
            False,
            "disable_custom_all_reduce",
        ),
        (
            "parallel",
            "distributed_executor_backend",
            "mp",
            "distributed_executor_backend",
        ),
        ("parallel", "enable_dbo", True, "enable_dbo"),
        ("parallel", "ubatch_size", 1, "ubatch_size"),
        ("parallel", "worker_cls", "auto", "worker_cls"),
        ("parallel", "worker_cls", "custom.Worker", "worker_cls"),
        (
            "parallel",
            "worker_extension_cls",
            "custom.Extension",
            "worker_extension_cls",
        ),
        ("scheduler", "async_scheduling", True, "async_scheduling"),
    ],
)
def test_serial_pic_v1_rejects_nonserial_or_ambiguous_topology(
    object_name: str,
    field_name: str,
    value: Any,
    message: str,
) -> None:
    vllm_config = _valid_vllm_config()
    if object_name == "kv":
        target = vllm_config.kv_transfer_config
    elif object_name == "kv_extra":
        target = vllm_config.kv_transfer_config.kv_connector_extra_config
    elif object_name == "parallel":
        target = vllm_config.parallel_config
    else:
        target = vllm_config.scheduler_config

    if isinstance(target, dict):
        target[field_name] = value
    else:
        setattr(target, field_name, value)

    with pytest.raises(ValueError, match=message):
        validate_target_profile(
            _valid_config(),
            SERIAL_PIC_V1,
            role="scheduler",
            vllm_config=vllm_config,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("skip_last_n_tokens", 1),
        ("discard_partial_chunks", True),
        ("lmcache.max_tokens_per_load", 32),
        ("unexpected_adapter_control", "enabled"),
        (1, "non-string-key"),
    ],
)
def test_serial_pic_v1_rejects_unreviewed_adapter_extra_controls(
    field_name: Any,
    value: Any,
) -> None:
    vllm_config = _valid_vllm_config()
    vllm_config.kv_transfer_config.kv_connector_extra_config[field_name] = value

    with pytest.raises(ValueError, match="unreviewed keys"):
        validate_target_profile(
            _valid_config(),
            SERIAL_PIC_V1,
            role="worker",
            vllm_config=vllm_config,
        )


@pytest.mark.parametrize("use_native", [None, True, 0, "false"])
def test_serial_pic_v1_requires_external_adapter_selection(use_native: Any) -> None:
    vllm_config = _valid_vllm_config()
    if use_native is None:
        del vllm_config.kv_transfer_config.kv_connector_extra_config["use_native"]
    else:
        vllm_config.kv_transfer_config.kv_connector_extra_config["use_native"] = (
            use_native
        )

    with pytest.raises(ValueError, match="use_native"):
        validate_target_profile(
            _valid_config(),
            SERIAL_PIC_V1,
            role="worker",
            vllm_config=vllm_config,
        )


@pytest.mark.parametrize("rpc_port", [0, 1, 65535])
def test_serial_pic_v1_accepts_bounded_rpc_port(rpc_port: int) -> None:
    vllm_config = _valid_vllm_config()
    vllm_config.kv_transfer_config.kv_connector_extra_config["lmcache_rpc_port"] = (
        rpc_port
    )

    validate_target_profile(
        _valid_config(),
        SERIAL_PIC_V1,
        role="worker",
        vllm_config=vllm_config,
    )


@pytest.mark.parametrize("rpc_port", [-1, 65536, 10**200])
def test_serial_pic_v1_rejects_out_of_range_rpc_port(rpc_port: int) -> None:
    vllm_config = _valid_vllm_config()
    vllm_config.kv_transfer_config.kv_connector_extra_config["lmcache_rpc_port"] = (
        rpc_port
    )

    with pytest.raises(ValueError, match="lmcache_rpc_port"):
        validate_target_profile(
            _valid_config(),
            SERIAL_PIC_V1,
            role="worker",
            vllm_config=vllm_config,
        )


@pytest.mark.parametrize(
    "engine_id",
    [" contains-space", "nested/path", "nul\0byte"],
)
def test_serial_pic_v1_rejects_unusable_engine_id(engine_id: str) -> None:
    vllm_config = _valid_vllm_config()
    vllm_config.kv_transfer_config.engine_id = engine_id

    with pytest.raises(ValueError, match="engine_id|IPC endpoint"):
        validate_target_profile(
            _valid_config(),
            SERIAL_PIC_V1,
            role="worker",
            vllm_config=vllm_config,
        )


def test_lmcache_ipc_path_enforces_exact_linux_byte_boundary() -> None:
    base_url = "/tmp"
    fixed_bytes = len(
        os.fsencode(
            get_zmq_rpc_path_lmcache(
                "",
                "lookup",
                0,
                0,
                base_url=base_url,
            )
        )
    )
    exact_engine_id = "x" * (SERIAL_PIC_V1_CONTRACT.ipc_path_max_bytes - fixed_bytes)

    exact_path = get_zmq_rpc_path_lmcache(
        exact_engine_id,
        "lookup",
        0,
        0,
        base_url=base_url,
    )
    assert len(os.fsencode(exact_path)) == 107

    shortened = get_zmq_rpc_path_lmcache(
        exact_engine_id + "x", "lookup", 0, 0, base_url=base_url
    )
    assert len(os.fsencode(shortened)) <= 107
    assert shortened != exact_path
    assert shortened == get_zmq_rpc_path_lmcache(
        exact_engine_id + "x", "lookup", 0, 0, base_url=base_url
    )


@pytest.mark.parametrize(
    ("engine_id", "rpc_port", "base_url"),
    [
        ("nul\0byte", 0, "/tmp"),
        ("engine", 0, "/" + "x" * 100),
    ],
)
def test_lmcache_ipc_path_rejects_unusable_final_endpoint(
    engine_id: str,
    rpc_port: int,
    base_url: str,
) -> None:
    with pytest.raises(ValueError, match="NUL|sockaddr_un"):
        get_zmq_rpc_path_lmcache(
            engine_id,
            "lookup",
            rpc_port,
            0,
            base_url=base_url,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("layers", 23),
        ("kv_heads", 1),
        ("head_size", 128),
        ("dtype", "float32"),
    ],
)
def test_serial_pic_v1_rejects_non_target_kv_chunk_geometry(
    field_name: str,
    value: Any,
) -> None:
    vllm_config = _valid_vllm_config()
    model_config = vllm_config.model_config
    if field_name == "layers":
        model_config.get_num_layers = lambda _parallel_config: value
    elif field_name == "kv_heads":
        model_config.get_num_kv_heads = lambda _parallel_config: value
    elif field_name == "head_size":
        model_config.get_head_size = lambda: value
    else:
        model_config.dtype = value

    with pytest.raises(ValueError, match="target KV chunk"):
        validate_target_profile(
            _valid_config(),
            SERIAL_PIC_V1,
            role="worker",
            vllm_config=vllm_config,
        )


def test_serial_pic_v1_rejects_force_skip_save_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMCACHE_FORCE_SKIP_SAVE", "1")

    with pytest.raises(ValueError, match="LMCACHE_FORCE_SKIP_SAVE"):
        validate_target_profile(
            _valid_config(),
            SERIAL_PIC_V1,
            role="worker",
            vllm_config=_valid_vllm_config(),
        )


@pytest.mark.parametrize(
    ("profile", "expected_force_skip_save"),
    [(SERIAL_PIC_V1, False), (None, True)],
)
def test_adapter_freezes_force_skip_save_only_for_target_profile(
    monkeypatch: pytest.MonkeyPatch,
    profile: Any,
    expected_force_skip_save: bool,
) -> None:
    """Post-admission environment writes cannot disable target stores."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    config = _valid_config()

    class ParentConnector:
        def register_kv_caches(self, kv_caches: dict[str, object]) -> None:
            return None

    class ReadyManager(_ConstructionHandoffManagerDouble):
        lmcache_engine = SimpleNamespace(
            gpu_connector=object(),
            metadata=None,
            drain_pending_store_cleanups=lambda: None,
            pending_store_cleanup_count=lambda: 0,
        )
        lmcache_engine_metadata = object()
        lookup_client = None
        lookup_server = object()
        health_monitor = None
        offload_server = None
        api_server = None
        runtime_plugin_launcher = None
        services_stopped = False

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            publisher = cast(Any, _kwargs.get("construction_result_publisher"))
            if publisher is not None:
                publisher(self)
            return None

        def start_services(self) -> None:
            return None

        def raise_if_init_failed(self) -> None:
            return None

        def is_healthy(self) -> bool:
            return True

        def retain_construction_cleanup(self) -> None:
            return None

        def release_construction_cleanup(self) -> None:
            return None

        def stop_services(self) -> None:
            self.services_stopped = True

    def inject_environment_after_validation(
        *_args: object, **_kwargs: object
    ) -> object:
        monkeypatch.setenv("LMCACHE_FORCE_SKIP_SAVE", "1")
        return object()

    monkeypatch.delenv("LMCACHE_FORCE_SKIP_SAVE", raising=False)
    monkeypatch.setattr(adapter_module, "lmcache_get_or_create_config", lambda: config)
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        inject_environment_after_validation,
    )
    monkeypatch.setattr(adapter_module, "LMCacheManager", ReadyManager)
    monkeypatch.setattr(
        adapter_module.LMCStatsMonitor,
        "GetOrCreate",
        lambda: object(),
    )
    monkeypatch.setattr(
        adapter_module.PrometheusLogger,
        "GetOrCreate",
        lambda *_args, **_kwargs: MagicMock(),
    )

    connector = adapter_module.LMCacheConnectorV1Impl(
        cast(Any, _valid_vllm_config(profile=profile)),
        vllm.KVConnectorRole.WORKER,
        cast(Any, ParentConnector()),
    )

    assert os.environ["LMCACHE_FORCE_SKIP_SAVE"] == "1"
    assert connector.force_skip_save is expected_force_skip_save
    connector.shutdown()


@pytest.mark.parametrize(
    ("object_name", "field_name"),
    [
        ("kv", "kv_role"),
        ("kv", "engine_id"),
        ("kv", "kv_connector_extra_config"),
        ("parallel", "pipeline_parallel_size"),
        ("parallel", "tensor_parallel_size"),
        ("parallel", "prefill_context_parallel_size"),
        ("parallel", "decode_context_parallel_size"),
        ("parallel", "data_parallel_size"),
        ("parallel", "data_parallel_size_local"),
        ("parallel", "data_parallel_rank"),
        ("parallel", "data_parallel_rank_local"),
        ("parallel", "nnodes"),
        ("parallel", "node_rank"),
        ("parallel", "world_size"),
        ("parallel", "rank"),
        ("parallel", "data_parallel_external_lb"),
        ("parallel", "data_parallel_hybrid_lb"),
        ("parallel", "enable_expert_parallel"),
        ("parallel", "enable_elastic_ep"),
        ("parallel", "enable_fault_tolerance"),
        ("parallel", "disable_custom_all_reduce"),
        ("parallel", "distributed_executor_backend"),
        ("parallel", "enable_dbo"),
        ("parallel", "ubatch_size"),
        ("parallel", "worker_cls"),
        ("parallel", "worker_extension_cls"),
        ("scheduler", "async_scheduling"),
    ],
)
def test_serial_pic_v1_rejects_absent_required_topology_field(
    object_name: str,
    field_name: str,
) -> None:
    vllm_config = _valid_vllm_config()
    if object_name == "kv":
        target = vllm_config.kv_transfer_config
    elif object_name == "parallel":
        target = vllm_config.parallel_config
    else:
        target = vllm_config.scheduler_config
    delattr(target, field_name)

    with pytest.raises(ValueError, match=field_name):
        validate_target_profile(
            _valid_config(),
            SERIAL_PIC_V1,
            role="worker",
            vllm_config=vllm_config,
        )


@pytest.mark.parametrize(
    "config_name",
    ["kv_transfer_config", "parallel_config", "scheduler_config"],
)
def test_serial_pic_v1_rejects_absent_required_topology_object(
    config_name: str,
) -> None:
    vllm_config = _valid_vllm_config()
    delattr(vllm_config, config_name)

    with pytest.raises(ValueError, match=config_name):
        validate_target_profile(
            _valid_config(),
            SERIAL_PIC_V1,
            role="scheduler",
            vllm_config=vllm_config,
        )


def test_adapter_revalidates_lmcache_overrides_before_service_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm")

    # First Party (after importorskip)
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    config = LMCacheEngineConfig.from_defaults()
    vllm_config = _valid_vllm_config(profile=None)
    vllm_config.kv_transfer_config.kv_connector_extra_config = {
        "lmcache.enable_controller": True
    }
    factory_calls: list[object] = []
    monkeypatch.setattr(adapter_module, "lmcache_get_or_create_config", lambda: config)
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *args: factory_calls.append(args),
    )

    with pytest.raises(ValueError, match="controller_pull_url"):
        adapter_module.LMCacheConnectorV1Impl(
            cast(Any, vllm_config),
            KVConnectorRole.WORKER,
            cast(Any, object()),
        )

    assert factory_calls == []


def test_adapter_activates_raw_profile_before_service_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm")

    # First Party (after importorskip)
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    # Ordinary defaults are intentionally outside serial_pic_v1.  The raw key
    # must therefore reject before the factory can create any service.
    config = LMCacheEngineConfig.from_defaults()
    factory_calls: list[object] = []
    monkeypatch.setattr(adapter_module, "lmcache_get_or_create_config", lambda: config)
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *args: factory_calls.append(args),
    )

    with pytest.raises(ValueError, match="serial_pic_v1 profile rejected"):
        adapter_module.LMCacheConnectorV1Impl(
            cast(Any, _valid_vllm_config()),
            KVConnectorRole.WORKER,
            cast(Any, object()),
        )

    assert factory_calls == []


def test_adapter_passes_resolved_profile_to_service_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm")

    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    config = _valid_config()
    factory_calls: list[tuple[object, object, str, Any]] = []
    construction_lease_calls: list[str] = []
    factory_sentinel = object()

    def build_factory(
        actual_config: object,
        vllm_config: object,
        role: str,
        *,
        target_profile: Any = None,
    ) -> object:
        factory_calls.append((actual_config, vllm_config, role, target_profile))
        return factory_sentinel

    class FakeManager(_ConstructionHandoffManagerDouble):
        _handoff_events = construction_lease_calls
        lmcache_engine = object()
        lmcache_engine_metadata = object()
        lookup_client = None
        lookup_server = object()
        health_monitor = None
        offload_server = None
        api_server = None
        runtime_plugin_launcher = None
        services_stopped = False

        def start_services(self) -> None:
            return None

        def raise_if_init_failed(self) -> None:
            return None

        def is_healthy(self) -> bool:
            return True

        def retain_construction_cleanup(self) -> None:
            construction_lease_calls.append("retain")

        def release_construction_cleanup(self) -> None:
            construction_lease_calls.append("release")

        def stop_services(self) -> None:
            self.services_stopped = True
            return None

    manager_calls: list[tuple[object, object, object]] = []

    def build_manager(
        actual_config: object,
        service_factory: object,
        *,
        connector: object,
        construction_result_publisher: Callable[[FakeManager], None],
    ) -> FakeManager:
        manager_calls.append((actual_config, service_factory, connector))
        manager = FakeManager()
        construction_result_publisher(manager)
        return manager

    monkeypatch.setattr(adapter_module, "lmcache_get_or_create_config", lambda: config)
    monkeypatch.setattr(adapter_module, "VllmServiceFactory", build_factory)
    monkeypatch.setattr(adapter_module, "LMCacheManager", build_manager)
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_setup_metrics",
        lambda *_args: None,
    )

    vllm_config = _valid_vllm_config()
    connector = adapter_module.LMCacheConnectorV1Impl(
        cast(Any, vllm_config),
        KVConnectorRole.WORKER,
        cast(Any, object()),
    )

    assert factory_calls == [(config, vllm_config, "worker", SERIAL_PIC_V1)]
    assert manager_calls == [(config, factory_sentinel, connector)]
    assert construction_lease_calls == ["retain", "release"]


@pytest.mark.parametrize("persistent_cleanup_failure", [False, True])
def test_serial_pic_v1_real_manager_constructor_baseexception_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    persistent_cleanup_failure: bool,
) -> None:
    """A real manager retains every owner published before an interrupt."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    assert LMCacheManager.provisional_construction_handoff_count() == 0
    lookup_client = MagicMock(spec=LookupClientInterface)
    original_error = KeyboardInterrupt("injected real-manager factory interrupt")
    close_error = RuntimeError("injected close failure")
    force_error = RuntimeError("injected force-close failure")
    factory = _InterruptingConstructionFactory(lookup_client, original_error)
    if persistent_cleanup_failure:
        lookup_client.close.side_effect = close_error
        lookup_client.force_close.side_effect = force_error

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )

    try:
        with pytest.raises(KeyboardInterrupt) as failure:
            adapter_module.LMCacheConnectorV1Impl(
                cast(Any, _valid_vllm_config()),
                vllm.KVConnectorRole.SCHEDULER,
                cast(Any, object()),
            )

        assert failure.value is original_error
        lookup_client.close.assert_called_once_with()
        if persistent_cleanup_failure:
            assert failure.value.__cause__ is force_error
            lookup_client.force_close.assert_called_once_with()
            assert LMCacheManager.incomplete_construction_cleanup_count() == 1

            lookup_client.force_close.side_effect = None
            assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
            assert lookup_client.close.call_count == 2
            assert lookup_client.force_close.call_count == 2
        else:
            assert failure.value.__cause__ is None
            lookup_client.force_close.assert_not_called()
            assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    finally:
        lookup_client.close.side_effect = None
        lookup_client.force_close.side_effect = None
        LMCacheManager.retry_incomplete_construction_cleanup()


def test_serial_pic_v1_degraded_manager_retains_public_retry_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected degraded manager keeps failed cleanup publicly reachable."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    lookup_client = MagicMock(spec=LookupClientInterface)
    construction_error = RuntimeError("injected ordinary construction failure")
    close_error = RuntimeError("injected close failure")
    force_error = RuntimeError("injected force-close failure")
    lookup_client.close.side_effect = close_error
    lookup_client.force_close.side_effect = force_error
    factory = _InterruptingConstructionFactory(lookup_client, construction_error)

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )

    try:
        with pytest.raises(
            RuntimeError, match="manager initialization failed"
        ) as failure:
            adapter_module.LMCacheConnectorV1Impl(
                cast(Any, _valid_vllm_config()),
                vllm.KVConnectorRole.SCHEDULER,
                cast(Any, object()),
            )

        assert "injected ordinary construction failure" in str(failure.value)
        assert failure.value.__cause__ is force_error
        lookup_client.close.assert_called_once_with()
        lookup_client.force_close.assert_called_once_with()
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1

        lookup_client.force_close.side_effect = None
        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        assert lookup_client.close.call_count == 2
        assert lookup_client.force_close.call_count == 2
    finally:
        lookup_client.close.side_effect = None
        lookup_client.force_close.side_effect = None
        LMCacheManager.retry_incomplete_construction_cleanup()


def test_serial_pic_v1_degraded_manager_interrupt_retains_retry_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted adapter unwind cannot erase degraded cleanup ownership."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    lookup_client = MagicMock(spec=LookupClientInterface)
    construction_error = RuntimeError("injected ordinary construction failure")
    cleanup_interrupt = KeyboardInterrupt("injected cleanup interrupt")
    lookup_client.close.side_effect = cleanup_interrupt
    lookup_client.force_close.side_effect = RuntimeError(
        "injected persistent force-close failure"
    )
    factory = _InterruptingConstructionFactory(lookup_client, construction_error)

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )

    try:
        with pytest.raises(
            RuntimeError, match="manager initialization failed"
        ) as failure:
            adapter_module.LMCacheConnectorV1Impl(
                cast(Any, _valid_vllm_config()),
                vllm.KVConnectorRole.SCHEDULER,
                cast(Any, object()),
            )

        assert failure.value.__cause__ is cleanup_interrupt
        lookup_client.close.assert_called_once_with()
        lookup_client.force_close.assert_called_once_with()
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1

        lookup_client.close.side_effect = None
        lookup_client.force_close.side_effect = None
        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        assert lookup_client.close.call_count == 2
        assert lookup_client.force_close.call_count == 1
    finally:
        lookup_client.close.side_effect = None
        lookup_client.force_close.side_effect = None
        LMCacheManager.retry_incomplete_construction_cleanup()


def test_serial_pic_v1_readiness_rejection_retains_public_retry_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy but inexact graph retains every failed cleanup owner."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    lookup_client = MagicMock(spec=LookupClientInterface)
    lookup_server = MagicMock()
    close_error = RuntimeError("injected close failure")
    force_error = RuntimeError("injected force-close failure")
    lookup_client.close.side_effect = close_error
    lookup_client.force_close.side_effect = force_error
    factory = _TargetSchedulerFactory(
        lookup_client,
        lookup_server=lookup_server,
    )

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )

    try:
        with pytest.raises(RuntimeError, match="service graph is not exact") as failure:
            adapter_module.LMCacheConnectorV1Impl(
                cast(Any, _valid_vllm_config()),
                vllm.KVConnectorRole.SCHEDULER,
                cast(Any, object()),
            )

        assert failure.value.__cause__ is force_error
        lookup_server.close.assert_called_once_with()
        lookup_client.close.assert_called_once_with()
        lookup_client.force_close.assert_called_once_with()
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1

        lookup_client.force_close.side_effect = None
        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        lookup_server.close.assert_called_once_with()
        assert lookup_client.close.call_count == 2
        assert lookup_client.force_close.call_count == 2
    finally:
        lookup_client.close.side_effect = None
        lookup_client.force_close.side_effect = None
        LMCacheManager.retry_incomplete_construction_cleanup()


def test_serial_pic_v1_late_init_interrupt_retains_public_retry_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The target lease spans connector state and metrics initialization."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    lookup_client = MagicMock(spec=LookupClientInterface)
    init_interrupt = KeyboardInterrupt("injected metrics interrupt")
    close_error = RuntimeError("injected close failure")
    force_error = RuntimeError("injected force-close failure")
    lookup_client.close.side_effect = close_error
    lookup_client.force_close.side_effect = force_error
    factory = _TargetSchedulerFactory(lookup_client)

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )

    def interrupt_metrics(*_args: object) -> None:
        raise init_interrupt

    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_setup_metrics",
        interrupt_metrics,
    )

    try:
        with pytest.raises(KeyboardInterrupt) as failure:
            adapter_module.LMCacheConnectorV1Impl(
                cast(Any, _valid_vllm_config()),
                vllm.KVConnectorRole.SCHEDULER,
                cast(Any, object()),
            )

        assert failure.value is init_interrupt
        assert failure.value.__cause__ is force_error
        lookup_client.close.assert_called_once_with()
        lookup_client.force_close.assert_called_once_with()
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1

        lookup_client.force_close.side_effect = None
        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        assert lookup_client.close.call_count == 2
        assert lookup_client.force_close.call_count == 2
    finally:
        lookup_client.close.side_effect = None
        lookup_client.force_close.side_effect = None
        LMCacheManager.retry_incomplete_construction_cleanup()


def test_serial_pic_v1_engine_destroy_failure_chains_exact_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected worker retains the exact failed engine teardown owner."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    engine = MagicMock(spec=LMCacheEngine)
    engine.is_healthy.return_value = True
    lookup_server = MagicMock()
    factory = _TargetWorkerFactory(engine, lookup_server)
    init_error = RuntimeError("injected worker metrics failure")
    destroy_error = RuntimeError("injected engine destroy failure")

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )

    def fail_metrics(*_args: object) -> None:
        raise init_error

    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_setup_metrics",
        fail_metrics,
    )

    with patch("lmcache.v1.manager.LMCacheEngineBuilder") as builder:
        builder.destroy.side_effect = destroy_error
        try:
            with pytest.raises(RuntimeError) as failure:
                adapter_module.LMCacheConnectorV1Impl(
                    cast(Any, _valid_vllm_config()),
                    vllm.KVConnectorRole.WORKER,
                    cast(Any, object()),
                )

            assert failure.value is init_error
            assert failure.value.__cause__ is destroy_error
            lookup_server.close.assert_called_once_with()
            builder.destroy.assert_called_once_with(
                "target-profile-worker-test",
                expected_engine=engine,
            )
            assert LMCacheManager.incomplete_construction_cleanup_count() == 1

            builder.destroy.side_effect = None
            builder.destroy.return_value = EngineDestroyResult.DESTROYED
            assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
            assert builder.destroy.call_count == 2
        finally:
            builder.destroy.side_effect = None
            builder.destroy.return_value = EngineDestroyResult.DESTROYED
            LMCacheManager.retry_incomplete_construction_cleanup()


def test_serial_pic_v1_shutdown_retains_real_builder_and_blender_until_retry() -> None:
    """Adapter shutdown retains every real terminal owner until success."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    instance_id = "target-profile-adapter-shutdown-retry"

    class ShutdownWorkerFactory(_TargetWorkerFactory):
        def get_engine_instance_id(self) -> str:
            return instance_id

    config = _valid_config()
    metadata = create_test_metadata()
    engine = MagicMock(spec=LMCacheEngine)
    engine.is_healthy.return_value = True
    engine.drain_pending_store_cleanups.return_value = None
    engine.pending_store_cleanup_count.return_value = 0
    close_error = RuntimeError("persistent engine close failure")
    engine.close.side_effect = close_error
    lookup_server = MagicMock()
    factory = ShutdownWorkerFactory(engine, lookup_server)
    factory.metadata = metadata
    stat_logger = MagicMock()
    blender = object()
    model = MagicMock()

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    assert instance_id not in LMCacheEngineBuilder._instances
    assert instance_id not in LMCBlenderBuilder._blenders
    assert instance_id not in VLLMModelTracker._vllm_models
    LMCacheEngineBuilder._instances[instance_id] = engine
    LMCacheEngineBuilder._cfgs[instance_id] = config
    LMCacheEngineBuilder._metadatas[instance_id] = metadata
    LMCacheEngineBuilder._stat_loggers[instance_id] = stat_logger
    LMCBlenderBuilder._blenders[instance_id] = cast(Any, blender)
    VLLMModelTracker._vllm_models[instance_id] = cast(Any, model)

    manager = LMCacheManager(config, factory)
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    state = cast(Any, connector)
    state._storage_pd_notify_queue = None
    state._storage_pd_status_sender = None
    state._storage_pd_ack_client = None
    state._storage_pd_mode = False
    state._storage_pd_raw_role = "writer"
    state._target_profile = SERIAL_PIC_V1
    state._role = vllm.KVConnectorRole.WORKER
    state._manager = manager
    state._kmesh_store_sessions = {}
    state._kmesh_outstanding_lookup_ids = set()
    state.blender = blender

    try:
        with patch(
            "lmcache.v1.cache_engine.LMCStatsMonitor.DestroyInstance"
        ) as destroy_monitor:
            for expected_close_calls in (1, 2):
                with pytest.raises(RuntimeError) as failure:
                    connector.shutdown()

                assert failure.value is close_error
                assert engine.close.call_count == expected_close_calls
                assert not manager.services_stopped
                assert manager.terminal_cleanup_errors == (close_error,)
                assert manager.lmcache_engine is engine
                assert manager._engine_owner_registered
                assert LMCacheManager._engine_owner_counts[engine] == 1
                assert LMCacheManager.incomplete_construction_cleanup_count() == 1
                with LMCacheManager._incomplete_construction_lock:
                    assert manager in LMCacheManager._incomplete_construction_owners
                assert LMCacheEngineBuilder.get(instance_id) is None
                assert LMCacheEngineBuilder._instances[instance_id] is engine
                assert (
                    LMCacheEngineBuilder._destroy_owners[instance_id].engine is engine
                )
                assert LMCacheEngineBuilder._cfgs[instance_id] is config
                assert LMCacheEngineBuilder._metadatas[instance_id] is metadata
                assert LMCacheEngineBuilder._stat_loggers[instance_id] is stat_logger
                assert LMCBlenderBuilder.get(instance_id) is blender
                assert VLLMModelTracker.get_model(instance_id) is model
                assert connector.blender is blender
                stat_logger.shutdown.assert_not_called()
                destroy_monitor.assert_not_called()

            engine.close.side_effect = None
            connector.shutdown()

            assert engine.close.call_count == 3
            assert manager.services_stopped
            assert manager.terminal_cleanup_errors == ()
            assert manager.lmcache_engine is None
            assert not manager._engine_owner_registered
            assert engine not in LMCacheManager._engine_owner_counts
            assert LMCacheManager.incomplete_construction_cleanup_count() == 0
            assert LMCacheEngineBuilder.get(instance_id) is None
            assert instance_id not in LMCacheEngineBuilder._cfgs
            assert instance_id not in LMCacheEngineBuilder._metadatas
            assert instance_id not in LMCacheEngineBuilder._stat_loggers
            with pytest.raises(ValueError, match="Blender .* not found"):
                LMCBlenderBuilder.get(instance_id)
            with pytest.raises(ValueError, match="model .* not found"):
                VLLMModelTracker.get_model(instance_id)
            assert connector.blender is None
            stat_logger.shutdown.assert_called_once_with()
            destroy_monitor.assert_called_once_with()
            lookup_server.close.assert_called_once_with()
    finally:
        engine.close.side_effect = None
        LMCacheEngineBuilder._instances.pop(instance_id, None)
        LMCacheEngineBuilder._cfgs.pop(instance_id, None)
        LMCacheEngineBuilder._metadatas.pop(instance_id, None)
        LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)
        LMCBlenderBuilder._blenders.pop(instance_id, None)
        VLLMModelTracker._vllm_models.pop(instance_id, None)
        with LMCacheManager._engine_ownership_lock:
            LMCacheManager._engine_ownership_receipts.pop(engine, None)
        with LMCacheManager._incomplete_construction_lock:
            LMCacheManager._incomplete_construction_owners.discard(manager)


def test_serial_pic_v1_storage_backend_close_failure_retains_real_owner() -> None:
    """A real storage failure remains truthful through Builder and Manager."""
    instance_id = "target-profile-storage-close-retry"

    class _RetryCloseBackend:
        def __init__(self, close_failures: int) -> None:
            self.close_failures = close_failures
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls <= self.close_failures:
                raise RuntimeError("persistent LocalCPUBackend close failure")

        def close_failure_is_retry_safe(self, error: BaseException) -> bool:
            return isinstance(error, RuntimeError)

    class _StoppedLoop:
        def __init__(self) -> None:
            self.closed = False

        def is_running(self) -> bool:
            return False

        def is_closed(self) -> bool:
            return self.closed

        def close(self) -> None:
            self.closed = True

    class _StoppedThread:
        def is_alive(self) -> bool:
            return False

    class _StorageWorkerFactory(_TargetWorkerFactory):
        def get_engine_instance_id(self) -> str:
            return instance_id

    backend = _RetryCloseBackend(close_failures=2)
    storage_manager = StorageManager.__new__(StorageManager)
    storage_state = cast(Any, storage_manager)
    storage_state.internal_copy_stream = None
    storage_state.storage_backends = OrderedDict([("LocalCPUBackend", backend)])
    storage_state._put_cleanup_lock = threading.RLock()
    storage_state._put_cleanup_owners = {}
    storage_state._put_cleanup_closed = False
    storage_state.loop = _StoppedLoop()
    storage_state.thread = _StoppedThread()

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine_state = cast(Any, engine)
    engine_state._store_cleanup_lock = threading.RLock()
    engine_state._store_cleanup_owners = {}
    engine_state._store_cleanup_closed = False
    engine_state.gpu_connector = None
    engine_state.storage_manager = storage_manager
    engine_state._close_lookup_cleanups = lambda: None
    engine_state.hidden_state_store = None
    engine_state.lmcache_worker = None

    config = _valid_config()
    metadata = create_test_metadata()
    stat_logger = MagicMock()
    lookup_server = MagicMock()
    factory = _StorageWorkerFactory(engine, lookup_server)
    factory.metadata = metadata

    assert instance_id not in LMCacheEngineBuilder._instances
    LMCacheEngineBuilder._instances[instance_id] = engine
    LMCacheEngineBuilder._cfgs[instance_id] = config
    LMCacheEngineBuilder._metadatas[instance_id] = metadata
    LMCacheEngineBuilder._stat_loggers[instance_id] = stat_logger
    manager = LMCacheManager(config, factory)

    try:
        for expected_close_calls in (1, 2):
            manager.stop_services()

            assert backend.close_calls == expected_close_calls
            assert not manager.services_stopped
            assert len(manager.terminal_cleanup_errors) == 1
            assert str(manager.terminal_cleanup_errors[0]) == (
                "persistent LocalCPUBackend close failure"
            )
            assert manager.lmcache_engine is engine
            assert manager._engine_owner_registered
            assert LMCacheManager._engine_owner_counts[engine] == 1
            assert LMCacheEngineBuilder.get(instance_id) is None
            assert LMCacheEngineBuilder._instances[instance_id] is engine
            assert LMCacheEngineBuilder._destroy_owners[instance_id].engine is engine
            assert LMCacheEngineBuilder._cfgs[instance_id] is config
            assert LMCacheEngineBuilder._metadatas[instance_id] is metadata
            assert LMCacheEngineBuilder._stat_loggers[instance_id] is stat_logger
            stat_logger.shutdown.assert_not_called()

        manager.stop_services()

        assert backend.close_calls == 3
        assert manager.services_stopped
        assert manager.terminal_cleanup_errors == ()
        assert manager.lmcache_engine is None
        assert not manager._engine_owner_registered
        assert engine not in LMCacheManager._engine_owner_counts
        assert LMCacheEngineBuilder.get(instance_id) is None
        assert instance_id not in LMCacheEngineBuilder._cfgs
        assert instance_id not in LMCacheEngineBuilder._metadatas
        assert instance_id not in LMCacheEngineBuilder._stat_loggers
        stat_logger.shutdown.assert_called_once_with()
        lookup_server.close.assert_called_once_with()
    finally:
        LMCacheEngineBuilder._instances.pop(instance_id, None)
        LMCacheEngineBuilder._cfgs.pop(instance_id, None)
        LMCacheEngineBuilder._metadatas.pop(instance_id, None)
        LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)
        with LMCacheManager._engine_ownership_lock:
            LMCacheManager._engine_ownership_receipts.pop(engine, None)
        with LMCacheManager._incomplete_construction_lock:
            LMCacheManager._incomplete_construction_owners.discard(manager)


def test_serial_pic_v1_healthy_construction_is_not_publicly_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public failure retry cannot stop a healthy adapter under construction."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    assert LMCacheManager.provisional_construction_handoff_count() == 0
    lookup_client = MagicMock(spec=LookupClientInterface)
    factory = _TargetSchedulerFactory(lookup_client)
    init_entered = threading.Event()
    allow_init = threading.Event()
    connectors: list[Any] = []
    construction_errors: list[BaseException] = []

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )

    def block_connector_state(*_args: object) -> None:
        init_entered.set()
        assert allow_init.wait(timeout=5)

    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_init_connector_state",
        block_connector_state,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_setup_metrics",
        lambda *_args: None,
    )

    def construct_adapter() -> None:
        try:
            connectors.append(
                adapter_module.LMCacheConnectorV1Impl(
                    cast(Any, _valid_vllm_config()),
                    vllm.KVConnectorRole.SCHEDULER,
                    cast(Any, object()),
                )
            )
        except BaseException as error:
            construction_errors.append(error)

    construction_thread = threading.Thread(target=construct_adapter)
    construction_thread_started = False
    try:
        construction_thread.start()
        construction_thread_started = True
        assert init_entered.wait(timeout=5)

        assert LMCacheManager.incomplete_construction_cleanup_count() == 0
        assert LMCacheManager.provisional_construction_handoff_count() == 1
        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        assert LMCacheManager.provisional_construction_handoff_count() == 1
        lookup_client.close.assert_not_called()

        allow_init.set()
        construction_thread.join(timeout=5)
        assert not construction_thread.is_alive()
        assert construction_errors == []
        assert len(connectors) == 1
        assert connectors[0].lookup_client is lookup_client
        assert LMCacheManager.incomplete_construction_cleanup_count() == 0
        assert LMCacheManager.provisional_construction_handoff_count() == 0

        connectors[0].shutdown()
        lookup_client.close.assert_called_once_with()
    finally:
        allow_init.set()
        if construction_thread_started:
            construction_thread.join(timeout=5)
        if connectors and lookup_client.close.call_count == 0:
            connectors[0].shutdown()
        LMCacheManager.retry_incomplete_construction_cleanup()


def test_serial_pic_v1_post_release_interrupt_promotes_exact_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter release keeps an allocation-free owner through __init__ return."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    lookup_client = MagicMock(spec=LookupClientInterface)
    close_error = RuntimeError("injected adopted Manager close failure")
    force_error = RuntimeError("injected adopted Manager force-close failure")
    lookup_client.close.side_effect = close_error
    lookup_client.force_close.side_effect = force_error
    factory = _TargetSchedulerFactory(lookup_client)
    boundary_error = KeyboardInterrupt("after successful Manager handoff release")
    released_receipts: list[Any] = []
    original_release = LMCacheManager.release_construction_handoff

    def record_release(manager: LMCacheManager, receipt: object) -> None:
        original_release(manager, cast(Any, receipt))
        released_receipts.append(receipt)

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_setup_metrics",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        LMCacheManager,
        "release_construction_handoff",
        record_release,
    )

    target = adapter_module.LMCacheConnectorV1Impl._release_target_manager_handoff
    source_lines, source_start = inspect.getsourcelines(target)
    clear_owner_line = next(
        source_start + offset
        for offset, source in enumerate(source_lines)
        if source.strip() == "self._manager_construction_owner = None"
    )
    triggered = False

    def interrupt_after_release(frame: Any, event: str, arg: Any) -> Any:
        del arg
        nonlocal triggered
        if (
            not triggered
            and frame.f_code is target.__code__
            and event == "line"
            and frame.f_lineno == clear_owner_line
        ):
            triggered = True
            sys.settrace(None)
            raise boundary_error
        return interrupt_after_release

    try:
        sys.settrace(interrupt_after_release)
        try:
            with pytest.raises(KeyboardInterrupt) as failure:
                adapter_module.LMCacheConnectorV1Impl(
                    cast(Any, _valid_vllm_config()),
                    vllm.KVConnectorRole.SCHEDULER,
                    cast(Any, object()),
                )
        finally:
            sys.settrace(None)

        assert triggered
        assert failure.value is boundary_error
        assert failure.value.__cause__ is force_error
        assert len(released_receipts) == 1
        receipt = released_receipts[0]
        assert receipt.phase == "FAILED_CLEANUP"
        assert LMCacheManager.provisional_construction_handoff_count() == 0
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1

        lookup_client.close.side_effect = None
        lookup_client.force_close.side_effect = None
        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        assert lookup_client.close.call_count == 2
        assert lookup_client.force_close.call_count == 1
        with LMCacheManager._incomplete_construction_lock:
            assert receipt not in LMCacheManager._construction_handoff_owners
    finally:
        sys.settrace(None)
        lookup_client.close.side_effect = None
        lookup_client.force_close.side_effect = None
        LMCacheManager.retry_incomplete_construction_cleanup()


def test_serial_pic_v1_success_has_no_construction_cleanup_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy adapter never enters the failed-construction registry."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    lookup_client = MagicMock(spec=LookupClientInterface)
    factory = _TargetSchedulerFactory(lookup_client)

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_setup_metrics",
        lambda *_args: None,
    )

    connector = adapter_module.LMCacheConnectorV1Impl(
        cast(Any, _valid_vllm_config()),
        vllm.KVConnectorRole.SCHEDULER,
        cast(Any, object()),
    )

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    connector.shutdown()
    lookup_client.close.assert_called_once_with()
    assert LMCacheManager.incomplete_construction_cleanup_count() == 0


@requires_instruction_monitoring
def test_serial_pic_v1_manager_publisher_store_replays_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact Manager reaches its retained owner at the publisher STORE."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    lookup_client = MagicMock(spec=LookupClientInterface)
    factory = _TargetSchedulerFactory(lookup_client)
    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_setup_metrics",
        lambda *_args: None,
    )
    target = adapter_module._RetainedManagerConstructionOwner._publish_manager
    injected = KeyboardInterrupt("before retained Manager STORE_ATTR")
    interrupt = InstructionInterruption(
        target,
        instruction_offset(target, "STORE_ATTR", argval="_manager"),
        injected,
        thread_name="lmcache-vllm-manager-construction-scheduler",
    )

    connector = None
    try:
        with interrupt:
            connector = adapter_module.LMCacheConnectorV1Impl(
                cast(Any, _valid_vllm_config()),
                vllm.KVConnectorRole.SCHEDULER,
                cast(Any, object()),
            )
        interrupt.assert_fired_once()
        assert connector._manager is not None
        lookup_client.close.assert_not_called()
    finally:
        if connector is not None:
            connector.shutdown()
        LMCacheManager.retry_incomplete_construction_cleanup()

    lookup_client.close.assert_called_once_with()
    assert LMCacheManager.incomplete_construction_cleanup_count() == 0


@requires_instruction_monitoring
def test_serial_pic_v1_real_manager_return_interrupt_is_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real Manager is owned before its constructor can return."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    lookup_client = MagicMock(spec=LookupClientInterface)
    factory = _TargetSchedulerFactory(lookup_client)
    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_setup_metrics",
        lambda *_args: None,
    )
    injected = KeyboardInterrupt("after real Manager __init__ returned")
    interrupt = InstructionInterruption(
        LMCacheManager.__init__,
        instruction_offset(LMCacheManager.__init__, "RETURN_CONST"),
        injected,
        thread_name="lmcache-vllm-manager-construction-scheduler",
    )

    try:
        with interrupt, pytest.raises(KeyboardInterrupt) as failure:
            adapter_module.LMCacheConnectorV1Impl(
                cast(Any, _valid_vllm_config()),
                vllm.KVConnectorRole.SCHEDULER,
                cast(Any, object()),
            )
        interrupt.assert_fired_once()
        assert failure.value is injected
        lookup_client.close.assert_called_once_with()
        assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    finally:
        LMCacheManager.retry_incomplete_construction_cleanup()


@requires_instruction_monitoring
def test_serial_pic_v1_real_adapter_return_interrupt_is_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outer owner retains the real adapter before __init__ returns."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.lmcache_connector_v1 as wrapper_module
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    lookup_client = MagicMock(spec=LookupClientInterface)
    factory = _TargetSchedulerFactory(lookup_client)
    wrappers: list[Any] = []
    monkeypatch.setattr(
        wrapper_module.KVConnectorBase_V1,
        "__init__",
        lambda wrapper, *_args, **_kwargs: wrappers.append(wrapper),
    )
    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_setup_metrics",
        lambda *_args: None,
    )
    injected = KeyboardInterrupt("after real adapter __init__ returned")
    interrupt = InstructionInterruption(
        adapter_module.LMCacheConnectorV1Impl.__init__,
        instruction_offset(
            adapter_module.LMCacheConnectorV1Impl.__init__,
            "RETURN_CONST",
        ),
        injected,
        thread_name="lmcache-vllm-wrapper-construction-scheduler",
    )

    try:
        with interrupt, pytest.raises(KeyboardInterrupt) as failure:
            wrapper_module.LMCacheConnectorV1Dynamic(
                cast(Any, _valid_vllm_config()),
                vllm.KVConnectorRole.SCHEDULER,
            )
        interrupt.assert_fired_once()
        assert failure.value is injected
        assert len(wrappers) == 1
        assert lookup_client.close.call_count == 1
        assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    finally:
        LMCacheManager.retry_incomplete_construction_cleanup()


def test_serial_pic_v1_manager_return_before_adapter_publication_is_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact returned Manager survives interruption before STORE_ATTR."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    boundary_error = KeyboardInterrupt(
        "injected after Manager return before adapter publication"
    )
    cleanup_calls: list[str] = []
    manager_calls: list[tuple[object, object, object]] = []

    class ExactManager(_ConstructionHandoffManagerDouble):
        _handoff_events = cleanup_calls
        services_stopped = False
        terminal_cleanup_errors: tuple[BaseException, ...] = ()

        def retain_construction_cleanup(self) -> None:
            cleanup_calls.append("retain")

        def release_construction_cleanup(self) -> None:
            cleanup_calls.append("release")

        def stop_services(self) -> None:
            cleanup_calls.append("stop")
            self.services_stopped = True

    manager = ExactManager()

    def build_manager(
        config: object,
        service_factory: object,
        *,
        connector: object,
        construction_result_publisher: Callable[[ExactManager], None],
    ) -> ExactManager:
        manager_calls.append((config, service_factory, connector))
        construction_result_publisher(manager)
        return manager

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(adapter_module, "LMCacheManager", build_manager)

    target = adapter_module.LMCacheConnectorV1Impl._acquire_target_manager
    source_lines, source_start = inspect.getsourcelines(target)
    assignment_line = next(
        source_start + offset
        for offset, source in enumerate(source_lines)
        if source.strip() == "self._manager = manager"
    )
    triggered = False

    def interrupt_before_manager_store(
        frame: Any,
        event: str,
        arg: Any,
    ) -> Any:
        del arg
        nonlocal triggered
        if (
            not triggered
            and frame.f_code is target.__code__
            and event == "line"
            and frame.f_lineno == assignment_line
        ):
            triggered = True
            sys.settrace(None)
            raise boundary_error
        return interrupt_before_manager_store

    sys.settrace(interrupt_before_manager_store)
    try:
        with pytest.raises(KeyboardInterrupt) as failure:
            adapter_module.LMCacheConnectorV1Impl(
                cast(Any, _valid_vllm_config()),
                vllm.KVConnectorRole.WORKER,
                cast(Any, object()),
            )
    finally:
        sys.settrace(None)

    assert triggered
    assert failure.value is boundary_error
    assert failure.value.__cause__ is None
    assert len(manager_calls) == 1
    assert cleanup_calls == ["retain", "promote", "stop"]
    assert manager.services_stopped


def test_serial_pic_v1_manager_recovered_when_provisional_retain_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned Manager survives failure before its first registry add."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0

    lookup_client = MagicMock(spec=LookupClientInterface)
    factory = _TargetSchedulerFactory(lookup_client)
    retain_error = KeyboardInterrupt(
        "injected before provisional Manager registry publication"
    )
    retained_managers: list[LMCacheManager] = []
    original_retain_handoff = LMCacheManager.retain_construction_handoff

    def fail_first_retain(
        manager: LMCacheManager,
        receipt: object,
    ) -> None:
        retained_managers.append(manager)
        if len(retained_managers) == 1:
            raise retain_error
        original_retain_handoff(manager, cast(Any, receipt))

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )
    monkeypatch.setattr(
        LMCacheManager,
        "retain_construction_handoff",
        fail_first_retain,
    )

    try:
        with pytest.raises(KeyboardInterrupt) as failure:
            adapter_module.LMCacheConnectorV1Impl(
                cast(Any, _valid_vllm_config()),
                vllm.KVConnectorRole.SCHEDULER,
                cast(Any, object()),
            )

        assert failure.value is retain_error
        assert failure.value.__cause__ is None
        assert len(retained_managers) == 1
        lookup_client.close.assert_called_once_with()
        lookup_client.force_close.assert_not_called()
        assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    finally:
        LMCacheManager.retry_incomplete_construction_cleanup()


def test_serial_pic_v1_manager_owner_uses_published_result_not_later_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker interrupt after Manager publication still stops that Manager."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    events: list[str] = []
    manager_calls: list[tuple[object, object, object]] = []
    worker_error = KeyboardInterrupt("after retained Manager publication")

    class ExactManager(_ConstructionHandoffManagerDouble):
        services_stopped = False
        terminal_cleanup_errors: tuple[BaseException, ...] = ()

        def retain_construction_cleanup(self) -> None:
            events.append("retain")

        def stop_services(self) -> None:
            events.append("stop")
            self.services_stopped = True

    manager = ExactManager()

    def build_manager(
        config: object,
        service_factory: object,
        *,
        connector: object,
        construction_result_publisher: Callable[[ExactManager], None],
    ) -> ExactManager:
        manager_calls.append((config, service_factory, connector))
        construction_result_publisher(manager)
        return manager

    monkeypatch.setattr(adapter_module, "LMCacheManager", build_manager)
    connector = adapter_module.LMCacheConnectorV1Impl.__new__(
        adapter_module.LMCacheConnectorV1Impl
    )
    owner = adapter_module._RetainedManagerConstructionOwner(
        cast(Any, object()),
        object(),
        connector,
        vllm.KVConnectorRole.SCHEDULER,
    )
    target = adapter_module._RetainedManagerConstructionOwner._invoke
    source_lines, source_start = inspect.getsourcelines(target)
    post_publication_line = next(
        source_start + offset
        for offset, source in enumerate(source_lines)
        if source.strip() == "receipt = manager.new_construction_handoff_receipt()"
    )
    thread_name = "lmcache-vllm-manager-construction-scheduler"
    triggered = False

    def interrupt_after_manager_store(
        frame: Any,
        event: str,
        arg: Any,
    ) -> Any:
        del arg
        nonlocal triggered
        if (
            not triggered
            and frame.f_code is target.__code__
            and event == "line"
            and frame.f_lineno == post_publication_line
            and threading.current_thread().name == thread_name
        ):
            triggered = True
            sys.settrace(None)
            raise worker_error
        return interrupt_after_manager_store

    threading.settrace(interrupt_after_manager_store)
    try:
        with pytest.raises(KeyboardInterrupt) as failure:
            owner.run()
    finally:
        threading.settrace(cast(Any, None))

    assert failure.value is worker_error
    assert owner.manager_returned
    assert owner.manager is manager
    assert owner.last_error is worker_error

    connector._target_profile = SERIAL_PIC_V1
    connector._manager_construction_owner = owner
    connector._cleanup_target_manager_after_failure("construction", worker_error)

    assert len(manager_calls) == 1
    assert events == ["retain", "stop"]
    assert manager.services_stopped


def test_serial_pic_v1_manager_publication_cleanup_interrupt_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker's provisional lease survives an interrupted unwind."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    assert LMCacheManager.incomplete_construction_cleanup_count() == 0
    lookup_client = MagicMock(spec=LookupClientInterface)
    factory = _TargetSchedulerFactory(lookup_client)
    boundary_error = KeyboardInterrupt(
        "injected after Manager return before adapter publication"
    )
    cleanup_interrupt = KeyboardInterrupt("injected after failed receipt promotion")
    promote_calls = 0
    original_promote = LMCacheManager.promote_construction_handoff_failure

    def interrupt_after_promotion(
        manager: LMCacheManager,
        receipt: object,
    ) -> None:
        nonlocal promote_calls
        promote_calls += 1
        original_promote(manager, cast(Any, receipt))
        if promote_calls == 1:
            raise cleanup_interrupt

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: factory,
    )
    monkeypatch.setattr(
        LMCacheManager,
        "promote_construction_handoff_failure",
        interrupt_after_promotion,
    )

    target = adapter_module.LMCacheConnectorV1Impl._acquire_target_manager
    source_lines, source_start = inspect.getsourcelines(target)
    assignment_line = next(
        source_start + offset
        for offset, source in enumerate(source_lines)
        if source.strip() == "self._manager = manager"
    )
    triggered = False

    def interrupt_before_manager_store(
        frame: Any,
        event: str,
        arg: Any,
    ) -> Any:
        del arg
        nonlocal triggered
        if (
            not triggered
            and frame.f_code is target.__code__
            and event == "line"
            and frame.f_lineno == assignment_line
        ):
            triggered = True
            sys.settrace(None)
            raise boundary_error
        return interrupt_before_manager_store

    try:
        sys.settrace(interrupt_before_manager_store)
        try:
            with pytest.raises(KeyboardInterrupt) as failure:
                adapter_module.LMCacheConnectorV1Impl(
                    cast(Any, _valid_vllm_config()),
                    vllm.KVConnectorRole.SCHEDULER,
                    cast(Any, object()),
                )
        finally:
            sys.settrace(None)

        assert triggered
        assert failure.value is boundary_error
        assert failure.value.__cause__ is cleanup_interrupt
        # The retained promotion attempt replays the same exact receipt after
        # the injected worker failure. The original cleanup interruption is
        # preserved, while the monotonic FAILED phase remains publicly owned.
        assert promote_calls == 2
        assert LMCacheManager.incomplete_construction_cleanup_count() == 1
        lookup_client.close.assert_not_called()

        monkeypatch.setattr(
            LMCacheManager,
            "promote_construction_handoff_failure",
            original_promote,
        )
        assert LMCacheManager.retry_incomplete_construction_cleanup() == 0
        lookup_client.close.assert_called_once_with()
        lookup_client.force_close.assert_not_called()
    finally:
        monkeypatch.setattr(
            LMCacheManager,
            "promote_construction_handoff_failure",
            original_promote,
        )
        LMCacheManager.retry_incomplete_construction_cleanup()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_serial_pic_v1_adapter_construction_cleans_baseexception(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fails: bool,
) -> None:
    """Public construction preserves an interrupt and cleanup failure."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    original_error = KeyboardInterrupt("injected adapter initialization interrupt")
    cleanup_error = RuntimeError("injected construction cleanup failure")
    stop_calls: list[str] = []

    class InterruptingManager(_ConstructionHandoffManagerDouble):
        lmcache_engine = object()
        lmcache_engine_metadata = object()
        lookup_client = None
        lookup_server = object()
        health_monitor = None
        offload_server = None
        api_server = None
        runtime_plugin_launcher = None
        services_stopped = False

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            publisher = cast(Any, _kwargs.get("construction_result_publisher"))
            if publisher is not None:
                publisher(self)
            return None

        def start_services(self) -> None:
            raise original_error

        def raise_if_init_failed(self) -> None:
            return None

        def is_healthy(self) -> bool:
            return True

        def retain_construction_cleanup(self) -> None:
            return None

        def release_construction_cleanup(self) -> None:
            return None

        def stop_services(self) -> None:
            stop_calls.append("stop")
            if cleanup_fails:
                raise cleanup_error
            self.services_stopped = True

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(
        adapter_module, "VllmServiceFactory", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(adapter_module, "LMCacheManager", InterruptingManager)

    with pytest.raises(KeyboardInterrupt) as failure:
        adapter_module.LMCacheConnectorV1Impl(
            cast(Any, _valid_vllm_config()),
            vllm.KVConnectorRole.WORKER,
            cast(Any, object()),
        )

    assert failure.value is original_error
    assert failure.value.__cause__ is (cleanup_error if cleanup_fails else None)
    assert stop_calls == ["stop"]


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_serial_pic_v1_register_rolls_back_baseexception(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fails: bool,
) -> None:
    """Public KV registration rolls back publication and preserves failures."""
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    config = _valid_config()
    original_error = KeyboardInterrupt("injected register interrupt")
    cleanup_error = RuntimeError("injected register cleanup failure")
    stop_calls: list[str] = []

    class ParentConnector:
        def register_kv_caches(self, kv_caches: dict[str, object]) -> None:
            return None

    class PostInitManager(_ConstructionHandoffManagerDouble):
        lmcache_engine = SimpleNamespace(
            gpu_connector=object(),
            metadata=None,
            drain_pending_store_cleanups=lambda: None,
            pending_store_cleanup_count=lambda: 0,
        )
        lmcache_engine_metadata = object()
        lookup_client = None
        lookup_server = object()
        health_monitor = None
        offload_server = None
        api_server = None
        runtime_plugin_launcher = None
        services_stopped = False

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            publisher = cast(Any, _kwargs.get("construction_result_publisher"))
            if publisher is not None:
                publisher(self)
            return None

        def start_services(self) -> None:
            return None

        def post_init(self) -> None:
            raise original_error

        def raise_if_init_failed(self) -> None:
            return None

        def is_healthy(self) -> bool:
            return True

        def retain_construction_cleanup(self) -> None:
            return None

        def release_construction_cleanup(self) -> None:
            return None

        def stop_services(self) -> None:
            stop_calls.append("stop")
            if cleanup_fails:
                raise cleanup_error
            self.services_stopped = True

    monkeypatch.setattr(adapter_module, "lmcache_get_or_create_config", lambda: config)
    monkeypatch.setattr(
        adapter_module,
        "VllmServiceFactory",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(adapter_module, "LMCacheManager", PostInitManager)
    monkeypatch.setattr(
        adapter_module.LMCStatsMonitor,
        "GetOrCreate",
        lambda: object(),
    )
    monkeypatch.setattr(
        adapter_module.PrometheusLogger,
        "GetOrCreate",
        lambda *_args, **_kwargs: MagicMock(),
    )

    connector = adapter_module.LMCacheConnectorV1Impl(
        cast(Any, _valid_vllm_config()),
        vllm.KVConnectorRole.WORKER,
        cast(Any, ParentConnector()),
    )
    original_caches = connector.kv_caches
    published_caches = {"layer.0": cast(Any, object())}

    with pytest.raises(KeyboardInterrupt) as failure:
        connector.register_kv_caches(published_caches)

    assert failure.value is original_error
    assert failure.value.__cause__ is (cleanup_error if cleanup_fails else None)
    assert connector.kv_caches is original_caches
    assert connector.kv_caches == {}
    assert stop_calls == ["stop"]


def test_serial_pic_v1_adapter_rejects_degraded_manager_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    cleanup_calls: list[str] = []

    class FailedManager(_ConstructionHandoffManagerDouble):
        lmcache_engine = None
        lmcache_engine_metadata = None
        lookup_client = None
        lookup_server = None
        health_monitor = None
        offload_server = None
        api_server = None
        runtime_plugin_launcher = None
        services_stopped = False

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            publisher = cast(Any, _kwargs.get("construction_result_publisher"))
            if publisher is not None:
                publisher(self)
            return None

        def start_services(self) -> None:
            return None

        def raise_if_init_failed(self) -> None:
            raise RuntimeError("injected lookup-client construction failure")

        def is_healthy(self) -> bool:
            return False

        def retain_construction_cleanup(self) -> None:
            return None

        def release_construction_cleanup(self) -> None:
            return None

        def stop_services(self) -> None:
            cleanup_calls.append("stop")
            self.services_stopped = True

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(adapter_module, "LMCacheManager", FailedManager)
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_setup_metrics",
        lambda *_args: None,
    )

    with pytest.raises(RuntimeError, match="lookup-client construction failure"):
        adapter_module.LMCacheConnectorV1Impl(
            cast(Any, _valid_vllm_config()),
            vllm.KVConnectorRole.WORKER,
            cast(Any, object()),
        )

    assert cleanup_calls == ["stop"]


@pytest.mark.parametrize(
    "failure_mode",
    ["direct_start", "missing_worker_server", "excluded_api"],
)
def test_serial_pic_v1_adapter_rejects_incomplete_startup_graph_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    cleanup_calls: list[str] = []
    start_calls: list[str] = []

    class IncompleteManager(_ConstructionHandoffManagerDouble):
        lmcache_engine = object()
        lmcache_engine_metadata = object()
        lookup_client = None
        lookup_server = object()
        health_monitor = None
        offload_server = None
        api_server = None
        runtime_plugin_launcher = None
        services_stopped = False

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            publisher = cast(Any, _kwargs.get("construction_result_publisher"))
            if publisher is not None:
                publisher(self)
            if failure_mode == "missing_worker_server":
                self.lookup_server = None
            if failure_mode == "excluded_api":
                self.api_server = object()

        def start_services(self) -> None:
            start_calls.append("start")
            if failure_mode == "direct_start":
                raise RuntimeError("injected direct start failure")

        def raise_if_init_failed(self) -> None:
            return None

        def is_healthy(self) -> bool:
            return True

        def retain_construction_cleanup(self) -> None:
            return None

        def release_construction_cleanup(self) -> None:
            return None

        def stop_services(self) -> None:
            cleanup_calls.append("stop")
            self.services_stopped = True

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(adapter_module, "LMCacheManager", IncompleteManager)
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        adapter_module.LMCacheConnectorV1Impl,
        "_setup_metrics",
        lambda *_args: None,
    )

    with pytest.raises(RuntimeError):
        adapter_module.LMCacheConnectorV1Impl(
            cast(Any, _valid_vllm_config()),
            vllm.KVConnectorRole.WORKER,
            cast(Any, object()),
        )

    expected_start_calls = ["start"] if failure_mode == "direct_start" else []
    assert start_calls == expected_start_calls
    assert cleanup_calls == ["stop"]


def test_serial_pic_v1_adapter_surfaces_incomplete_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

    # First Party
    import lmcache.integration.vllm.vllm_v1_adapter as adapter_module

    class UncleanManager(_ConstructionHandoffManagerDouble):
        lmcache_engine = object()
        lmcache_engine_metadata = object()
        lookup_client = None
        lookup_server = None
        health_monitor = None
        offload_server = None
        api_server = None
        runtime_plugin_launcher = None
        services_stopped = False
        terminal_cleanup_errors: tuple[BaseException, ...] = ()

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            publisher = cast(Any, _kwargs.get("construction_result_publisher"))
            if publisher is not None:
                publisher(self)
            return None

        def start_services(self) -> None:
            raise AssertionError("pre-start graph validation must fail first")

        def raise_if_init_failed(self) -> None:
            return None

        def is_healthy(self) -> bool:
            return True

        def retain_construction_cleanup(self) -> None:
            return None

        def release_construction_cleanup(self) -> None:
            return None

        def stop_services(self) -> None:
            return None

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        _valid_config,
    )
    monkeypatch.setattr(adapter_module, "LMCacheManager", UncleanManager)

    with pytest.raises(RuntimeError, match="worker engine") as failure:
        adapter_module.LMCacheConnectorV1Impl(
            cast(Any, _valid_vllm_config()),
            vllm.KVConnectorRole.WORKER,
            cast(Any, object()),
        )

    assert isinstance(failure.value.__cause__, RuntimeError)
    assert "did not fully quiesce" in str(failure.value.__cause__)


def test_serial_pic_v1_service_factory_suppresses_excluded_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    import lmcache.integration.vllm.vllm_service_factory as factory_module

    factory = factory_module.VllmServiceFactory(
        _valid_config(),
        cast(Any, _valid_vllm_config()),
        "worker",
        target_profile=SERIAL_PIC_V1,
    )

    def forbidden_constructor(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("excluded service constructor was called")

    monkeypatch.setattr(factory_module, "ZMQOffloadServer", forbidden_constructor)
    monkeypatch.setattr(
        factory_module,
        "RuntimePluginLauncher",
        forbidden_constructor,
    )
    monkeypatch.setattr(
        factory_module,
        "InternalAPIServer",
        forbidden_constructor,
    )

    assert factory.maybe_create_offload_server() is None
    assert factory.maybe_create_runtime_plugin_launcher() is None
    assert factory.maybe_create_internal_api_server(cast(Any, object())) is None


def test_no_profile_service_factory_preserves_offload_server_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm")

    # Third Party
    import vllm.distributed.parallel_state as parallel_state

    # First Party
    import lmcache.integration.vllm.vllm_service_factory as factory_module

    engine = object()
    server = object()
    calls: list[tuple[object, int]] = []

    def build_server(actual_engine: object, rank: int) -> object:
        calls.append((actual_engine, rank))
        return server

    monkeypatch.setattr(
        parallel_state,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(factory_module, "ZMQOffloadServer", build_server)
    factory = factory_module.VllmServiceFactory(
        _valid_config(),
        cast(Any, _valid_vllm_config(profile=None)),
        "worker",
    )
    factory.lmcache_engine = engine  # type: ignore[assignment]

    assert factory.maybe_create_offload_server() is server
    assert calls == [(engine, 0)]


def test_no_profile_service_factory_preserves_plugin_launcher_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    import lmcache.integration.vllm.utils as vllm_utils
    import lmcache.integration.vllm.vllm_service_factory as factory_module

    launcher = object()
    calls: list[tuple[object, str, int, int]] = []

    def build_launcher(
        config: object,
        role: str,
        worker_count: int,
        worker_id: int,
    ) -> object:
        calls.append((config, role, worker_count, worker_id))
        return launcher

    monkeypatch.setattr(vllm_utils, "is_dp_rank0", lambda _config: True)
    monkeypatch.setattr(factory_module, "RuntimePluginLauncher", build_launcher)
    config = _valid_config()
    factory = factory_module.VllmServiceFactory(
        config,
        cast(Any, _valid_vllm_config(profile=None)),
        "worker",
    )
    factory.lmcache_engine = object()  # type: ignore[assignment]
    factory.metadata = cast(Any, SimpleNamespace(worker_id=0))

    assert factory.maybe_create_runtime_plugin_launcher() is launcher
    assert calls == [(config, "worker", 1, 0)]


def test_no_profile_service_factory_preserves_internal_api_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    import lmcache.integration.vllm.utils as vllm_utils
    import lmcache.integration.vllm.vllm_service_factory as factory_module

    server = object()
    manager = object()
    calls: list[object] = []

    def build_server(actual_manager: object) -> object:
        calls.append(actual_manager)
        return server

    monkeypatch.setattr(vllm_utils, "is_dp_rank0", lambda _config: True)
    monkeypatch.setattr(factory_module, "InternalAPIServer", build_server)
    factory = factory_module.VllmServiceFactory(
        _valid_config(),
        cast(Any, _valid_vllm_config(profile=None)),
        "worker",
    )

    assert factory.maybe_create_internal_api_server(cast(Any, manager)) is server
    assert calls == [manager]
