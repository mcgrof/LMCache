# SPDX-License-Identifier: Apache-2.0
"""Fail-closed tests for descriptor-issued multiprocess cache keys."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import (
    ExternalKVKeys,
    IPCCacheEngineKey,
    reject_external_kv_keys,
    resolve_external_chunk_hashes,
)


def make_key(
    external_keys: ExternalKVKeys | None,
    *,
    start: int = 0,
    end: int = 512,
) -> IPCCacheEngineKey:
    """Build a two-chunk IPC key for validation tests."""
    return IPCCacheEngineKey.from_token_ids(
        model_name="model",
        world_size=1,
        worker_id=0,
        token_ids=list(range(512)),
        start=start,
        end=end,
        request_id="request",
        external_keys=external_keys,
    )


def test_external_keys_replace_token_hashes() -> None:
    """The server returns opaque keys unchanged and performs no token hashing."""
    keys = ExternalKVKeys((b"a" * 32, b"b" * 32))

    assert resolve_external_chunk_hashes(make_key(keys), 256) == list(keys.keys)


def test_legacy_mode_is_explicit() -> None:
    """Only an absent external-key object selects the legacy token namespace."""
    assert resolve_external_chunk_hashes(make_key(None), 256) is None


@pytest.mark.parametrize(
    ("keys", "start", "end", "message"),
    [
        (ExternalKVKeys((b"a" * 32,)), 0, 512, "count"),
        (ExternalKVKeys((b"a" * 32,)), 1, 257, "aligned"),
        (ExternalKVKeys((b"a" * 32,)), 0, 768, "range"),
    ],
)
def test_external_key_mismatch_never_falls_back(
    keys: ExternalKVKeys,
    start: int,
    end: int,
    message: str,
) -> None:
    """Invalid external metadata is rejected before storage lookup."""
    with pytest.raises(ValueError, match=message):
        resolve_external_chunk_hashes(make_key(keys, start=start, end=end), 256)


def test_required_load_store_op_rejects_missing_keys() -> None:
    """Losing provenance at scheduler-to-worker crossing fails closed."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

    with pytest.raises(ValueError, match="provenance-required"):
        LoadStoreOp(
            token_ids=list(range(256)),
            block_ids=list(range(16)),
            end=256,
            require_provenance=True,
        )


def test_required_load_store_op_keeps_frozen_keys() -> None:
    """The operation carries one immutable protocol object to the worker."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

    keys = ExternalKVKeys((b"a" * 32,))
    op = LoadStoreOp(
        token_ids=list(range(256)),
        block_ids=list(range(16)),
        end=256,
        external_keys=keys,
        require_provenance=True,
    )

    assert op.external_keys is keys
    with pytest.raises(AttributeError):
        op.external_keys = None  # type: ignore[misc]


def test_server_helper_does_not_consult_a_hasher() -> None:
    """External-key resolution is independent of TokenHasher state."""
    hasher = MagicMock()
    keys = ExternalKVKeys((b"a" * 32, b"b" * 32))

    resolved = resolve_external_chunk_hashes(make_key(keys), 256)

    assert resolved == list(keys.keys)
    hasher.assert_not_called()


def test_scheduler_retry_and_reconnect_keep_external_identity() -> None:
    """Session/request IDs and adapter instances do not perturb opaque keys."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )

    keys = ExternalKVKeys((b"a" * 32,))
    first = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    first.model_name = "model"
    first._world_size = 1
    first.parallel_strategy = None
    reconnected = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    reconnected.model_name = "model"
    reconnected._world_size = 1
    reconnected.parallel_strategy = None

    retry_key = first._create_key(list(range(256)), 0, 256, "retry", external_keys=keys)
    reconnect_key = reconnected._create_key(
        list(range(256)), 0, 256, "reconnect", external_keys=keys
    )

    assert retry_key == reconnect_key
    assert retry_key.external_keys is keys


def test_worker_key_keeps_scheduler_metadata_keys() -> None:
    """Worker-to-server construction preserves the scheduler's key object."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPWorkerAdapter,
    )

    keys = ExternalKVKeys((b"a" * 32,))
    worker = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    worker.model_name = "model"
    worker._world_size = 1
    worker._worker_id = 0
    worker.parallel_strategy = None

    key = worker._create_key(list(range(256)), 0, 256, "worker", external_keys=keys)

    assert key.external_keys is keys
    assert key.worker_id == 0


def test_unsupported_blend_path_rejects_external_keys() -> None:
    """Blend must not silently turn descriptor keys back into token hashes."""
    key = make_key(ExternalKVKeys((b"a" * 32, b"b" * 32)))

    with pytest.raises(ValueError, match="does not support"):
        reject_external_kv_keys(key, "Blend lookup")
