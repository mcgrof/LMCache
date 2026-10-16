# SPDX-License-Identifier: Apache-2.0
"""Outer-wrapper lookup/close linearization regressions."""

# Standard
from unittest.mock import MagicMock, patch
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.chunk_statistics_lookup_client import (
    ChunkStatisticsLookupClient,
)
from lmcache.v1.lookup_client.hit_limit_lookup_client import HitLimitLookupClient


class _BlockingTokens:
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self.entered = entered
        self.release = release

    def __len__(self) -> int:
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        return 1


@pytest.mark.parametrize("close_name", ["close", "force_close"])
def test_hit_limit_close_waits_for_post_inner_lookup(
    close_name: str,
) -> None:
    inner = MagicMock(spec=LookupClientInterface)
    inner.lookup.return_value = 1
    config = MagicMock()
    config.hit_miss_ratio = 0.0
    config.chunk_size = 256
    wrapper = HitLimitLookupClient(inner, config)
    post_inner_entered = threading.Event()
    post_inner_release = threading.Event()
    tokens = _BlockingTokens(post_inner_entered, post_inner_release)
    lookup_result = []

    lookup_thread = threading.Thread(
        target=lambda: lookup_result.append(wrapper.lookup(tokens, "lookup"))
    )
    lookup_thread.start()
    assert post_inner_entered.wait(timeout=2.0)

    close_thread = threading.Thread(target=getattr(wrapper, close_name))
    close_thread.start()
    close_thread.join(timeout=0.05)
    assert close_thread.is_alive()
    getattr(inner, close_name).assert_not_called()

    post_inner_release.set()
    lookup_thread.join(timeout=2.0)
    close_thread.join(timeout=2.0)
    assert not lookup_thread.is_alive()
    assert not close_thread.is_alive()
    assert lookup_result == [1]
    getattr(inner, close_name).assert_called_once()
    with pytest.raises(RuntimeError, match="closing or closed"):
        wrapper.lookup([], "late")


@pytest.mark.parametrize("close_name", ["close", "force_close"])
def test_chunk_statistics_close_waits_for_post_inner_lookup(
    close_name: str,
) -> None:
    inner = MagicMock(spec=LookupClientInterface)
    inner.lookup.return_value = 0
    config = LMCacheEngineConfig(
        enable_chunk_statistics=True,
        chunk_statistics_strategy="memory_bloom_filter",
        extra_config={
            "chunk_statistics_mem_bf_expected_chunks": 1000,
            "chunk_statistics_mem_bf_false_positive_rate": 0.01,
        },
    )
    wrapper = ChunkStatisticsLookupClient(inner, config)
    wrapper.enabled = True
    post_inner_entered = threading.Event()
    post_inner_release = threading.Event()

    def blocked_record(*_args) -> None:
        post_inner_entered.set()
        assert post_inner_release.wait(timeout=2.0)

    lookup_result = []
    with patch.object(wrapper.recorder, "record_async", side_effect=blocked_record):
        lookup_thread = threading.Thread(
            target=lambda: lookup_result.append(wrapper.lookup([1], "lookup"))
        )
        lookup_thread.start()
        assert post_inner_entered.wait(timeout=2.0)

        close_thread = threading.Thread(target=getattr(wrapper, close_name))
        close_thread.start()
        close_thread.join(timeout=0.05)
        assert close_thread.is_alive()
        getattr(inner, close_name).assert_not_called()

        post_inner_release.set()
        lookup_thread.join(timeout=2.0)
        close_thread.join(timeout=2.0)

    assert not lookup_thread.is_alive()
    assert not close_thread.is_alive()
    assert lookup_result == [0]
    getattr(inner, close_name).assert_called_once()
    with pytest.raises(RuntimeError, match="closing or closed"):
        wrapper.lookup([1], "late")
