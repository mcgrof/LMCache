# SPDX-License-Identifier: Apache-2.0
"""Fail-closed validation for narrowly supported vLLM target profiles.

These profiles are admission checks for experimental integrations.  Passing a
profile says only that the resolved configuration and the role-local topology
are inside the named test envelope.  It does not establish that services from
another role have been constructed, nor does it make any teardown guarantee.
"""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Optional
import math
import os
import re

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache


def _freeze_contract_value(value: Any) -> Any:
    """Recursively copy a target-profile value into immutable containers."""
    if type(value) is dict:
        return MappingProxyType(
            {
                _freeze_contract_value(key): _freeze_contract_value(item)
                for key, item in value.items()
            }
        )
    if type(value) in (list, tuple):
        return tuple(_freeze_contract_value(item) for item in value)
    if type(value) in (set, frozenset):
        return frozenset(_freeze_contract_value(item) for item in value)
    return value


KMESH_TARGET_PROFILE_KEY = "kmesh_target_profile"
SERIAL_PIC_V1 = "serial_pic_v1"


@dataclass(frozen=True)
class TargetProfileContract:
    """Immutable public description of a target-profile admission contract.

    Attributes:
        exact_config: Exact values for statically pinned LMCache fields.
        dynamic_config: LMCache fields validated by dynamic bounds.
        exact_parallel: Exact values for the paired vLLM parallel topology.
        target_chunk_bytes: Required byte size of one target KV chunk.
        ipc_path_max_bytes: Maximum admitted Linux UNIX-socket path length.
    """

    exact_config: Mapping[str, Any]
    dynamic_config: frozenset[str]
    exact_parallel: Mapping[str, Any]
    target_chunk_bytes: int
    ipc_path_max_bytes: int


# Qwen2.5-0.5B-Instruct, BF16 KV, 24 layers, 2 K/V planes, 256 tokens,
# 2 KV heads and a 64-element head: 24 * 2 * 256 * 2 * 64 * 2 bytes.
# The paired vLLM profile pins that exact runtime identity.  Keeping the byte
# floor here prevents a positive subnormal GiB value from becoming a zero-byte
# allocator (or an allocator too small to hold even one target chunk).
_SERIAL_PIC_V1_TARGET_CHUNK_BYTES = 3 * 1024 * 1024
_LINUX_UNIX_SOCKET_PATH_MAX_BYTES = 107
_SAFE_ENGINE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


# Every public LMCacheEngineConfig field admitted by serial_pic_v1 is listed
# here, rather than inheriting whatever defaults a future LMCache release adds.
# A new config field therefore rejects until it is explicitly reviewed and
# pinned.  max_local_cpu_size and lmcache_instance_id are the only intentionally
# dynamic values and are validated separately below.
_SERIAL_PIC_V1_EXACT_CONFIG: dict[str, Any] = {
    "chunk_size": 256,
    "local_cpu": True,
    "local_cpu_use_hugepages": False,
    "reserve_local_cpu_size": 0.0,
    "local_disk": None,
    "local_disk_path_sharding": "by_gpu",
    "max_local_disk_size": 0.0,
    "remote_url": None,
    "remote_serde": "naive",
    "use_layerwise": True,
    "save_decode_cache": False,
    "pre_caching_hash_algorithm": "builtin",
    "enable_blending": True,
    "blend_recompute_ratios": [0.15],
    "blend_thresholds": None,
    "blend_check_layers": [1],
    "blend_min_tokens": 256,
    "blend_special_str": " # # ",
    "retrieve_locations": ["LocalCPUBackend"],
    "store_location": "LocalCPUBackend",
    "enable_p2p": False,
    "p2p_host": None,
    "p2p_init_ports": None,
    "p2p_lookup_ports": None,
    "mp_host": None,
    "mp_port": 5555,
    "enable_controller": False,
    "controller_pull_url": None,
    "controller_reply_url": None,
    "lmcache_worker_ports": None,
    "lmcache_worker_ids": None,
    "lmcache_worker_heartbeat_delay_time": 10,
    "lmcache_worker_heartbeat_time": None,
    "enable_pd": False,
    "pd_role": None,
    "pd_buffer_size": None,
    "pd_buffer_device": None,
    "pd_peer_host": None,
    "pd_peer_init_port": None,
    "pd_peer_alloc_port": None,
    "pd_peer_query_port": None,
    "pd_proxy_host": None,
    "pd_proxy_port": None,
    "pd_allocation_timeout_sec": float("inf"),
    "pd_shutdown_timeout_sec": 5.0,
    "pd_condition_poll_interval_sec": 0.005,
    "pd_max_prefill_len": 0,
    "pd_backend_mode": "async",
    "pd_skip_proxy_notification": False,
    "pd_bidirectional": False,
    "transfer_channel": None,
    "nixl_backends": None,
    "nixl_buffer_size": None,
    "nixl_buffer_device": None,
    "gds_path": None,
    "gds_path_sharding": "by_gpu",
    "gds_buffer_size": None,
    "maru_path": None,
    "maru_pool_size": 4.0,
    "use_gds": False,
    "gds_backend": "cufile",
    "audit_actual_remote_url": None,
    "internal_api_server_host": "0.0.0.0",
    "extra_config": {"enable_sparse": True},
    "save_unfull_chunk": True,
    "blocking_timeout_secs": 10,
    "external_lookup_client": None,
    "py_enable_gc": True,
    "cache_policy": "LRU",
    "numa_mode": None,
    "enable_async_loading": False,
    "internal_api_server_enabled": False,
    "internal_api_server_port_start": 6999,
    "priority_limit": None,
    "internal_api_server_include_index_list": None,
    "internal_api_server_socket_path_prefix": None,
    "runtime_plugin_locations": None,
    "storage_plugins": None,
    "remote_storage_plugins": None,
    "lookup_timeout_ms": 3000,
    "min_retrieve_tokens": 0,
    "hit_miss_ratio": None,
    "lookup_server_worker_ids": None,
    "enable_scheduler_bypass_lookup": False,
    "script_allowed_imports": None,
    "enable_lazy_memory_allocator": False,
    "lazy_memory_initial_ratio": 0.2,
    "lazy_memory_expand_trigger_ratio": 0.5,
    "lazy_memory_step_ratio": 0.1,
    "lazy_memory_safe_size": 0.0,
    "enable_chunk_statistics": False,
    "chunk_statistics_auto_start_statistics": False,
    "chunk_statistics_auto_exit_timeout_hours": 0.0,
    "chunk_statistics_auto_exit_target_unique_chunks": 0,
    "chunk_statistics_strategy": "memory_bloom_filter",
    "enable_kv_events": False,
    "use_gpu_connector_v3": False,
    "pin_timeout_sec": 300,
    "pin_check_interval_sec": 30,
    "remote_config_url": None,
    "app_id": None,
    "enable_hidden_state_cache": False,
    "max_hidden_state_cpu_size": 2.0,
    "hidden_state_layers": None,
    "hidden_states_retrieve_mode": "prefix_strict",
}
_SERIAL_PIC_V1_DYNAMIC_CONFIG = frozenset({"max_local_cpu_size", "lmcache_instance_id"})

# Mirror the vLLM-owned topology portion of the paired serial-PIC profile.  The
# LMCache adapter receives a role-local child VllmConfig after MultiConnector
# qualification; validating the resolved values again here prevents a later
# service-factory read from widening that bounded single-process topology.
_SERIAL_PIC_V1_EXACT_PARALLEL: dict[str, Any] = {
    "pipeline_parallel_size": 1,
    "tensor_parallel_size": 1,
    "prefill_context_parallel_size": 1,
    "decode_context_parallel_size": 1,
    "data_parallel_size": 1,
    "data_parallel_size_local": 1,
    "data_parallel_rank": 0,
    "data_parallel_rank_local": 0,
    "nnodes": 1,
    "node_rank": 0,
    "world_size": 1,
    "rank": 0,
    "data_parallel_external_lb": False,
    "data_parallel_hybrid_lb": False,
    "enable_expert_parallel": False,
    "enable_elastic_ep": False,
    "enable_fault_tolerance": False,
    "disable_custom_all_reduce": True,
    "distributed_executor_backend": "uni",
    "enable_dbo": False,
    "ubatch_size": 0,
    "worker_cls": "vllm.v1.worker.gpu_worker.Worker",
    "worker_extension_cls": "",
}

SERIAL_PIC_V1_CONTRACT: TargetProfileContract = TargetProfileContract(
    exact_config=MappingProxyType(
        {
            field_name: _freeze_contract_value(expected)
            for field_name, expected in _SERIAL_PIC_V1_EXACT_CONFIG.items()
        }
    ),
    dynamic_config=_SERIAL_PIC_V1_DYNAMIC_CONFIG,
    exact_parallel=MappingProxyType(
        {
            field_name: _freeze_contract_value(expected)
            for field_name, expected in _SERIAL_PIC_V1_EXACT_PARALLEL.items()
        }
    ),
    target_chunk_bytes=_SERIAL_PIC_V1_TARGET_CHUNK_BYTES,
    ipc_path_max_bytes=_LINUX_UNIX_SOCKET_PATH_MAX_BYTES,
)


def requested_target_profile(vllm_config: Any) -> Optional[str]:
    """Return the raw target-profile activation from a vLLM config.

    Args:
        vllm_config: Resolved vLLM configuration to inspect.

    Returns:
        The requested non-empty profile name, or ``None`` when no profile was
        requested.

    Raises:
        ValueError: The profile activation exists but is not a non-empty
        string.
    """
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    connector_config = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    if not isinstance(connector_config, dict):
        return None

    profile = connector_config.get(KMESH_TARGET_PROFILE_KEY)
    if profile is None:
        return None
    if not isinstance(profile, str) or not profile:
        raise ValueError(
            f"{KMESH_TARGET_PROFILE_KEY} must be a non-empty string, got {profile!r}"
        )
    return profile


def _normalized_role(role: Any) -> Optional[str]:
    if role is None:
        return None
    name = getattr(role, "name", role)
    if not isinstance(name, str):
        raise ValueError(f"target profile role must be a string or enum, got {role!r}")
    normalized = name.lower()
    if normalized not in ("scheduler", "worker"):
        raise ValueError(
            f"serial_pic_v1 role must be 'scheduler' or 'worker', got {name!r}"
        )
    return normalized


def _matches_exact_value(actual: Any, expected: Any) -> bool:
    """Compare profile values recursively without equality coercion."""
    if type(actual) is not type(expected):
        return False

    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _matches_exact_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )

    if isinstance(expected, dict):
        if len(actual) != len(expected):
            return False
        unmatched_items = list(actual.items())
        for expected_key, expected_value in expected.items():
            for index, (actual_key, actual_value) in enumerate(unmatched_items):
                if not _matches_exact_value(actual_key, expected_key):
                    continue
                if not _matches_exact_value(actual_value, expected_value):
                    continue
                unmatched_items.pop(index)
                break
            else:
                return False
        return not unmatched_items

    if isinstance(expected, (set, frozenset)):
        if len(actual) != len(expected):
            return False
        unmatched_items = list(actual)
        for expected_item in expected:
            for index, actual_item in enumerate(unmatched_items):
                if _matches_exact_value(actual_item, expected_item):
                    unmatched_items.pop(index)
                    break
            else:
                return False
        return not unmatched_items

    return bool(actual == expected)


def _validate_serial_pic_config(config: LMCacheEngineConfig) -> list[str]:
    problems: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    public_fields = {field.name for field in fields(config)}
    reviewed_fields = set(_SERIAL_PIC_V1_EXACT_CONFIG)
    reviewed_fields.update(_SERIAL_PIC_V1_DYNAMIC_CONFIG)
    for field_name in sorted(public_fields - reviewed_fields):
        problems.append(
            f"unreviewed LMCacheEngineConfig field {field_name!r} is not admitted"
        )
    for field_name in sorted(reviewed_fields - public_fields):
        problems.append(
            f"reviewed LMCacheEngineConfig field {field_name!r} is unavailable"
        )

    for field_name, expected in _SERIAL_PIC_V1_EXACT_CONFIG.items():
        if field_name not in public_fields:
            continue
        actual = getattr(config, field_name)
        require(
            _matches_exact_value(actual, expected),
            f"{field_name} must equal {expected!r}, got {actual!r}",
        )

    cpu_size = config.max_local_cpu_size
    cpu_size_bytes = -1
    cpu_size_is_number = False
    if type(cpu_size) in (int, float):
        try:
            cpu_size_float = float(cpu_size)
            cpu_size_is_number = math.isfinite(cpu_size_float)
            if cpu_size_is_number:
                cpu_size_bytes = int(cpu_size_float * 1024**3)
        except (OverflowError, ValueError):
            pass
    require(
        cpu_size_is_number and cpu_size_bytes >= _SERIAL_PIC_V1_TARGET_CHUNK_BYTES,
        "max_local_cpu_size must be a finite number large enough for at least "
        f"one {_SERIAL_PIC_V1_TARGET_CHUNK_BYTES}-byte target KV chunk",
    )
    instance_id = config.lmcache_instance_id
    require(
        isinstance(instance_id, str)
        and bool(instance_id)
        and instance_id == instance_id.strip(),
        "lmcache_instance_id must be a non-empty trimmed string",
    )
    return problems


def _validate_serial_pic_topology(
    config: LMCacheEngineConfig,
    *,
    role: Any,
    vllm_config: Any,
) -> list[str]:
    problems: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    def required_attr(owner: Any, owner_name: str, field_name: str) -> Any:
        if not hasattr(owner, field_name):
            problems.append(f"vLLM {owner_name}.{field_name} must be present")
            return None
        return getattr(owner, field_name)

    normalized_role = _normalized_role(role)
    # Normalizing a non-null role is itself the role-local check.  Do not
    # inspect a peer process or require either role to have been constructed.
    assert normalized_role is None or normalized_role in ("scheduler", "worker")

    if vllm_config is None:
        return problems

    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        problems.append("vllm_config.kv_transfer_config must be present")
        return problems

    kv_role = required_attr(kv_transfer_config, "kv_transfer_config", "kv_role")
    require(
        kv_role == "kv_both",
        "vLLM kv_role must equal 'kv_both'",
    )
    engine_id = required_attr(kv_transfer_config, "kv_transfer_config", "engine_id")
    require(
        type(engine_id) is str and _SAFE_ENGINE_ID.fullmatch(engine_id) is not None,
        "vLLM engine_id must be a non-empty safe IPC path component",
    )
    kv_parallel_size = required_attr(
        kv_transfer_config, "kv_transfer_config", "kv_parallel_size"
    )
    require(
        _matches_exact_value(kv_parallel_size, 1),
        "vLLM kv_parallel_size must equal 1",
    )
    kv_rank = required_attr(kv_transfer_config, "kv_transfer_config", "kv_rank")
    require(
        kv_rank is None,
        "vLLM kv_rank must be unset",
    )

    connector_config = required_attr(
        kv_transfer_config,
        "kv_transfer_config",
        "kv_connector_extra_config",
    )
    require(
        isinstance(connector_config, dict),
        "vLLM kv_connector_extra_config must be a dictionary",
    )
    if isinstance(connector_config, dict):
        # These values are consumed directly by the vLLM adapter after the
        # LMCacheEngineConfig has been validated.  Keep this namespace closed:
        # otherwise load capping, token skipping, or partial-chunk transforms
        # can silently widen the supposedly exact profile.
        admitted_extra_keys = {
            KMESH_TARGET_PROFILE_KEY,
            "use_native",
            "lmcache_rpc_port",
        }
        unexpected_extra_keys = sorted(
            repr(key)
            for key in connector_config
            if type(key) is not str or key not in admitted_extra_keys
        )
        require(
            not unexpected_extra_keys,
            "vLLM kv_connector_extra_config contains unreviewed keys "
            f"{unexpected_extra_keys!r}",
        )
        require(
            connector_config.get("use_native") is False,
            "vLLM use_native must be exactly false",
        )
        rpc_port = connector_config.get("lmcache_rpc_port", 0)
        if "lmcache_rpc_port" in connector_config:
            require(
                type(rpc_port) is int and 0 <= rpc_port <= 65535,
                "lmcache_rpc_port must be an integer from 0 through 65535",
            )
        if (
            type(engine_id) is str
            and _SAFE_ENGINE_ID.fullmatch(engine_id) is not None
            and type(rpc_port) is int
            and 0 <= rpc_port <= 65535
        ):
            try:
                endpoint = get_zmq_rpc_path_lmcache(
                    engine_id,
                    "lookup",
                    rpc_port,
                    0,
                )
            except ValueError as error:
                problems.append(f"LMCache lookup IPC endpoint is invalid: {error}")
            else:
                endpoint_bytes = len(os.fsencode(endpoint))
                require(
                    endpoint_bytes <= _LINUX_UNIX_SOCKET_PATH_MAX_BYTES,
                    "LMCache lookup IPC endpoint must fit Linux sockaddr_un "
                    f"({_LINUX_UNIX_SOCKET_PATH_MAX_BYTES} bytes), got "
                    f"{endpoint_bytes} bytes for {endpoint!r}",
                )

    # This adapter-only environment switch is read after config validation and
    # is therefore not covered by the closed LMCacheEngineConfig contract.
    require(
        "LMCACHE_FORCE_SKIP_SAVE" not in os.environ,
        "LMCACHE_FORCE_SKIP_SAVE must be unset",
    )

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        problems.append("vllm_config.parallel_config must be present")
    else:
        for field_name, expected in _SERIAL_PIC_V1_EXACT_PARALLEL.items():
            actual = required_attr(parallel_config, "parallel_config", field_name)
            require(
                _matches_exact_value(actual, expected),
                f"vLLM {field_name} must equal {expected!r}",
            )

    model_config = getattr(vllm_config, "model_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    if model_config is None:
        problems.append("vllm_config.model_config must be present")
    if cache_config is None:
        problems.append("vllm_config.cache_config must be present")
    if (
        model_config is not None
        and cache_config is not None
        and parallel_config is not None
    ):
        try:
            # Third Party
            import torch

            try:
                # Third Party
                from vllm.utils.torch_utils import get_kv_cache_torch_dtype
            except ImportError:
                # Third Party
                from vllm.utils import get_kv_cache_torch_dtype
            # First Party
            from lmcache.integration.vllm.utils import (
                calculate_draft_layers,
                mla_enabled,
            )

            kv_dtype = get_kv_cache_torch_dtype(
                cache_config.cache_dtype,
                model_config.dtype,
            )
            num_layers = model_config.get_num_layers(parallel_config)
            num_layers += calculate_draft_layers(vllm_config)
            kv_planes = 1 if mla_enabled(model_config) else 2
            target_chunk_bytes = (
                num_layers
                * kv_planes
                * config.chunk_size
                * model_config.get_num_kv_heads(parallel_config)
                * model_config.get_head_size()
                * torch.empty((), dtype=kv_dtype).element_size()
            )
        except Exception as error:
            problems.append(f"cannot derive vLLM target KV chunk size: {error}")
        else:
            require(
                target_chunk_bytes == _SERIAL_PIC_V1_TARGET_CHUNK_BYTES,
                "vLLM target KV chunk must equal "
                f"{_SERIAL_PIC_V1_TARGET_CHUNK_BYTES} bytes, got "
                f"{target_chunk_bytes}",
            )

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if scheduler_config is None:
        problems.append("vllm_config.scheduler_config must be present")
    else:
        async_scheduling = required_attr(
            scheduler_config, "scheduler_config", "async_scheduling"
        )
        require(
            async_scheduling is False,
            "vLLM async_scheduling must be false",
        )

    return problems


def validate_target_profile(
    config: LMCacheEngineConfig,
    profile: Optional[str],
    *,
    role: Any = None,
    vllm_config: Any = None,
) -> None:
    """Validate an explicitly requested, fully resolved target profile.

    ``None`` is an intentional no-op so ordinary LMCache users do not enter an
    experimental profile.  Any non-null unknown value fails closed.  If a raw
    vLLM activation is observable, it must agree with ``profile``.

    Args:
        config: Fully resolved LMCache configuration to validate.
        profile: Explicit target profile, or ``None`` for ordinary LMCache.
        role: Optional role string or enum for role-local validation.
        vllm_config: Optional resolved vLLM configuration for paired topology
            and model-geometry validation.

    Returns:
        None.

    Raises:
        ValueError: The requested profile is unknown, conflicts with its raw
        activation, or any resolved value falls outside the closed contract.
    """
    if profile is None:
        return
    if profile != SERIAL_PIC_V1:
        raise ValueError(f"unsupported LMCache target profile {profile!r}")

    raw_profile = (
        requested_target_profile(vllm_config) if vllm_config is not None else None
    )
    if raw_profile is not None and raw_profile != profile:
        raise ValueError(
            f"target profile argument {profile!r} conflicts with raw "
            f"{KMESH_TARGET_PROFILE_KEY}={raw_profile!r}"
        )

    problems = _validate_serial_pic_config(config)
    problems.extend(
        _validate_serial_pic_topology(
            config,
            role=role,
            vllm_config=vllm_config,
        )
    )
    if problems:
        details = "; ".join(problems)
        raise ValueError(f"LMCache {SERIAL_PIC_V1} profile rejected: {details}")
