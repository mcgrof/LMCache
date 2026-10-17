# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    SaveSpec,
)


class _FakeParent:
    def __init__(self, metadata):
        self._connector_metadata = metadata

    def _get_connector_metadata(self):
        return self._connector_metadata


class _FakeEngine:
    def __init__(self):
        self.unpinned: list[str] = []
        self.registered: list[str] = []
        self.store_steps: dict[str, int] = {}
        self.store_calls: list[str] = []
        self.fail_on_step: dict[str, int] = {}

    def register_lookup_cleanup(self, req_id: str) -> None:
        self.registered.append(req_id)

    def schedule_lookup_unpin(self, req_id: str) -> None:
        self.lookup_unpin(req_id)

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store_layer(self, token_ids, **kwargs):
        req_id = kwargs["req_id"]
        self.store_calls.append(req_id)
        self.store_steps.setdefault(req_id, 0)

        def _storer():
            while True:
                if self.fail_on_step.get(req_id) == self.store_steps[req_id] + 1:
                    raise RuntimeError(f"injected storer failure for {req_id}")
                self.store_steps[req_id] += 1
                yield None

        return _storer()


class _FakeManager:
    def __init__(self, engine: _FakeEngine):
        self.lmcache_engine = engine


def _make_req(
    req_id: str,
    can_save: bool = True,
    lookup_id: str | None = None,
):
    return SimpleNamespace(
        req_id=req_id,
        lookup_id=lookup_id,
        token_ids=[1, 2, 3, 4],
        slot_mapping=torch.arange(4, dtype=torch.long),
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=can_save),
    )


def _make_connector(requests):
    metadata = LMCacheConnectorMetadata(requests=requests)
    engine = _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)
    connector._manager = _FakeManager(engine)
    connector.kv_role = "kv_producer"
    connector.use_layerwise = True
    connector.device = "cpu"
    connector._lmcache_chunk_size = 8
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector._layerwise_save_storers = {}
    connector.layerwise_retrievers = []
    connector.current_layer = 0
    connector.num_layers = 1
    return connector, metadata, engine


def test_layerwise_storer_is_request_scoped_across_interleaved_finalize() -> None:
    connector, metadata, engine = _make_connector(
        [_make_req("req-1"), _make_req("req-2")]
    )

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == ["req-1", "req-2"]
    assert engine.store_steps["req-1"] == 1
    assert engine.store_steps["req-2"] == 1

    metadata.requests = [_make_req("req-1")]
    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert engine.store_steps["req-2"] == 1
    assert engine.unpinned == ["req-1"]
    assert set(connector._layerwise_save_storers.keys()) == {"req-2"}

    metadata.requests = [_make_req("req-2")]
    connector.wait_for_save()
    assert engine.store_steps["req-2"] == 2
    assert engine.unpinned == ["req-1", "req-2"]
    assert connector._layerwise_save_storers == {}


def test_wait_for_save_repeated_call_does_not_readvance_finalized_storer() -> None:
    connector, metadata, engine = _make_connector([_make_req("req-1")])
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_steps["req-1"] == 1

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert connector._layerwise_save_storers == {}

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2


def test_multi_layer_save_and_finalize() -> None:
    connector, _, engine = _make_connector([_make_req("req-1"), _make_req("req-2")])
    num_layers = 4

    for _ in range(num_layers):
        connector.save_kv_layer("layer_x", torch.zeros(1), None)

    assert engine.store_steps["req-1"] == num_layers
    assert engine.store_steps["req-2"] == num_layers

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == num_layers + 1
    assert engine.store_steps["req-2"] == num_layers + 1
    assert connector._layerwise_save_storers == {}


def test_layerwise_save_skips_requests_that_cannot_save() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", can_save=False)])
    connector.kv_role = "kv_both"
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == []
    assert connector._layerwise_save_storers == {}


def test_layerwise_save_kv_producer_ignores_can_save_flag() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", can_save=False)])

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == ["req-1"]
    assert engine.store_steps["req-1"] == 1
    assert set(connector._layerwise_save_storers.keys()) == {"req-1"}

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert connector._layerwise_save_storers == {}


def test_save_storer_failure_schedules_every_transferred_lookup() -> None:
    connector, _, engine = _make_connector(
        [
            _make_req("req-1", lookup_id="lookup-1"),
            _make_req("req-2", lookup_id="lookup-2"),
        ]
    )
    engine.fail_on_step["req-1"] = 1

    with pytest.raises(RuntimeError, match="req-1"):
        connector.save_kv_layer("layer0", torch.zeros(1), None)

    assert engine.registered[:2] == ["lookup-1", "lookup-2"]
    assert engine.unpinned == ["lookup-1", "lookup-2"]


def test_wait_for_save_isolates_storer_failure_and_cleans_later_id() -> None:
    connector, _, engine = _make_connector(
        [
            _make_req("req-1", lookup_id="lookup-1"),
            _make_req("req-2", lookup_id="lookup-2"),
        ]
    )
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    engine.fail_on_step["req-1"] = 2

    with pytest.raises(RuntimeError, match="req-1"):
        connector.wait_for_save()

    assert engine.unpinned == ["lookup-1", "lookup-2"]
    assert engine.store_steps["req-1"] == 1
    assert engine.store_steps["req-2"] == 2


def test_wait_for_layer_load_failure_schedules_every_transferred_lookup() -> None:
    connector, _, engine = _make_connector(
        [
            _make_req("req-1", lookup_id="lookup-1"),
            _make_req("req-2", lookup_id="lookup-2"),
        ]
    )

    def failing_retriever():
        raise RuntimeError("injected layer retriever failure")
        yield None

    connector.layerwise_retrievers = [failing_retriever()]

    with pytest.raises(RuntimeError, match="retriever failure"):
        connector.wait_for_layer_load("layer0")

    assert engine.unpinned == ["lookup-1", "lookup-2"]


def test_no_forward_metadata_schedules_transferred_lookup_cleanup() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", lookup_id="lookup-1")])

    connector.start_load_kv(SimpleNamespace(attn_metadata=None))

    assert engine.registered == ["lookup-1"]
    assert engine.unpinned == ["lookup-1"]


def test_metadata_getter_failure_uses_raw_metadata_for_cleanup() -> None:
    connector, metadata, engine = _make_connector(
        [_make_req("req-1", lookup_id="lookup-1")]
    )

    class _FailingParent(_FakeParent):
        def _get_connector_metadata(self):
            raise RuntimeError("injected metadata getter failure")

    connector._parent = _FailingParent(metadata)

    with pytest.raises(RuntimeError, match="metadata getter failure"):
        connector.start_load_kv(SimpleNamespace(attn_metadata=object()))

    assert engine.registered == ["lookup-1"]
    assert engine.unpinned == ["lookup-1"]
