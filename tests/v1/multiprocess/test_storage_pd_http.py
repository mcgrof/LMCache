# SPDX-License-Identifier: Apache-2.0
"""MP handoff HTTP contracts with real storage and a registered test layout."""

# Standard
from pathlib import Path
from types import SimpleNamespace
import os
import select

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    EncodedObjectKey,
    GroupKind,
    MemoryLayoutDesc,
)
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.multiprocess.http_apis.storage_pd_api import router
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from tests.v1.distributed.test_raw_block_l2_adapter import (
    _create_memory_obj,
    requires_raw_block_ext,
)


@requires_raw_block_ext
def test_http_receipt_adoption_and_exact_ack(tmp_path: Path) -> None:
    """HTTP READY, adoption and ACK use the same on-device committed generation."""
    path = tmp_path / "shared.bin"
    path.write_bytes(bytes(1 << 20))
    hasher = TokenHasher(256)
    owners = [
        RawBlockL2Adapter(
            RawBlockL2AdapterConfig(
                device_path=str(path),
                role=role,
                pd_derivation=hasher.raw_block_derivation(),
                capacity_bytes=1 << 20,
                slot_bytes=65536,
                meta_total_bytes=65536,
                use_odirect=False,
                meta_enable_periodic=False,
            )
        )
        for role in ("writer", "reader")
    ]
    clients = []
    try:
        for owner in owners:
            app = FastAPI()
            app.include_router(router)
            engine = SimpleNamespace(
                storage_manager=SimpleNamespace(
                    l2_adapters=lambda owner=owner: [(None, owner)]
                ),
                context=SimpleNamespace(
                    token_hasher=hasher,
                    layout_desc_registry=SimpleNamespace(
                        find_group_layout_descs=lambda *args: {
                            0: MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
                        },
                        find_attn_desc=lambda *args: AttnWindowDesc([-1]),
                    ),
                ),
            )
            app.state.context = SimpleNamespace(engine=engine)
            clients.append(TestClient(app))
        writer, reader = clients
        manifest = {
            "request_id": "http-request",
            "model_name": "test-model",
            "token_ids": list(range(513)),
        }
        cancelled = dict(manifest, request_id="cancelled")
        assert writer.post("/storage-pd/prepare", json=cancelled).status_code == 200
        assert writer.post(
            "/storage-pd/cancel", json={"request_id": "cancelled"}
        ).json() == {"cancelled": True}
        assert (
            writer.post(
                "/storage-pd/publish", json={"request_id": "cancelled"}
            ).status_code
            == 409
        )
        ready = writer.post("/storage-pd/prepare", json=manifest)
        assert ready.status_code == 200
        assert (
            writer.post(
                "/storage-pd/publish",
                json={"request_id": "http-request", "timeout_s": 0.01},
            ).status_code
            == 504
        )
        keys = [EncodedObjectKey(**key).to_object_key() for key in ready.json()["keys"]]
        task = owners[0].submit_store_task(keys, [_create_memory_obj() for _ in keys])
        poll = select.poll()
        poll.register(owners[0].get_store_event_fd(), select.POLLIN)
        assert poll.poll(5000)
        os.eventfd_read(owners[0].get_store_event_fd())
        assert owners[0].pop_completed_store_tasks()[task].is_successful()
        response = writer.post(
            "/storage-pd/publish", json={"request_id": "http-request"}
        )
        assert response.status_code == 200
        receipt = response.json()["receipt"]
        assert (
            reader.post(
                "/storage-pd/adopt", json={"manifest": manifest, "receipt": receipt}
            ).status_code
            == 200
        )
        bad = dict(receipt, namespace_identity="different-namespace")
        assert (
            reader.post(
                "/storage-pd/adopt",
                json={"manifest": manifest, "receipt": bad, "timeout_ms": 0},
            ).status_code
            == 409
        )
        lease = {
            "identity": {
                "req_id": "http-request",
                "consumer_instance_id": "reader",
                "tp_rank": 0,
                "writer_epoch": receipt["writer_epoch"],
                "checkpoint_seq": receipt["checkpoint_seq"],
                "manifest_digest": receipt["manifest_digest"],
            },
            "session_id": "group",
        }
        assert (
            writer.post("/storage-pd/ack", json=lease).json()["outcome"] == "REJECTED"
        )
        assert writer.post("/storage-pd/claim", json=lease).status_code == 200
        unread = {
            "request_id": "http-request",
            "receipt": receipt,
            "session_id": "group",
        }
        assert not writer.post("/storage-pd/unread", json=unread).json()["released"]
        assert writer.post("/storage-pd/ack", json=lease).json()["outcome"] == "APPLIED"
        assert (
            writer.post("/storage-pd/ack", json=lease).json()["outcome"]
            == "ALREADY_APPLIED"
        )
    finally:
        for client in clients:
            client.close()
        for owner in reversed(owners):
            owner.close()


@pytest.mark.parametrize(
    "groups,windows,kinds,removed,expected",
    [
        ({1: object()}, [-1], (), False, 409),
        ({0: object()}, [2], (), False, 409),
        ({0: object()}, [-1], ("recurrent",), False, 409),
        ({0: object()}, [-1], (), True, 503),
        ({}, [-1], (), False, 503),
    ],
)
def test_http_refuses_unsupported_or_reaped_registration(
    groups: dict[int, object],
    windows: list[int],
    kinds: tuple[GroupKind, ...],
    removed: bool,
    expected: int,
) -> None:
    """Unsupported groups and a concurrent worker reap fail before storage access."""

    def attention(*args: object) -> AttnWindowDesc:
        if removed:
            raise ValueError("model reaped")
        return AttnWindowDesc(windows, group_kinds=kinds)

    app = FastAPI()
    app.include_router(router)
    app.state.context = SimpleNamespace(
        engine=SimpleNamespace(
            context=SimpleNamespace(
                layout_desc_registry=SimpleNamespace(
                    find_group_layout_descs=lambda *args: groups,
                    find_attn_desc=attention,
                ),
            )
        )
    )
    with TestClient(app) as client:
        response = client.post(
            "/storage-pd/prepare",
            json={
                "request_id": "unsupported",
                "model_name": "test-model",
                "token_ids": list(range(513)),
            },
        )
        assert response.status_code == expected
