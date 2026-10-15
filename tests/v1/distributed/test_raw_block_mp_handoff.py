# SPDX-License-Identifier: Apache-2.0
"""Public MP adapter handoff contracts on actual shared file storage."""

# Standard
from dataclasses import replace
from pathlib import Path
import json
import os
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.storage_backend.raw_block import (
    RawBlockDerivationDescriptor,
    ReadAckIdentity,
    ReadAckOutcome,
)
from lmcache.v1.utils.json_utils import make_json_safe
from tests.v1.distributed.test_raw_block_l2_adapter import (
    _create_memory_obj,
    _create_object_key,
    requires_raw_block_ext,
)

pytestmark = [pytest.mark.no_shared_allocator, requires_raw_block_ext]
DERIVATION = RawBlockDerivationDescriptor(
    hash_algorithm="blake3",
    hash_implementation="contract-test",
    hash_seed="",
    chain_root="00",
    key_namespace="object",
)


def adapter(path: Path, role: str = "writer") -> RawBlockL2Adapter:
    """Open one independent owner of the same four-slot namespace."""
    return RawBlockL2Adapter(
        RawBlockL2AdapterConfig(
            device_path=str(path),
            role=role,
            pd_derivation=DERIVATION,
            capacity_bytes=5 * 65536,
            slot_bytes=65536,
            meta_total_bytes=65536,
            use_odirect=False,
            meta_enable_periodic=False,
        )
    )


def store(owner: RawBlockL2Adapter, chunk: int) -> bool:
    """Wait for an actual POSIX payload completion through the public adapter."""
    task = owner.submit_store_task(
        [_create_object_key(chunk)], [_create_memory_obj(1024, float(chunk))]
    )
    poll = select.poll()
    poll.register(owner.get_store_event_fd(), select.POLLIN)
    assert poll.poll(5000)
    os.eventfd_read(owner.get_store_event_fd())
    result = owner.pop_completed_store_tasks()
    assert task in result
    return result[task].is_successful()


def test_publication_adoption_and_ack_protect_then_reclaim(tmp_path: Path) -> None:
    """A live reader holds four extents until an exact claimed ACK releases them."""
    path = tmp_path / "shared.bin"
    path.write_bytes(bytes(5 * 65536))
    writer = adapter(path)
    reader = adapter(path, "reader")
    keys = [_create_object_key(i) for i in range(4)]
    layout = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
    try:
        writer.handoff.prepare("request", keys)
        writer.handoff.prepare("request", keys)
        with pytest.raises(TimeoutError):
            writer.handoff.publish("request", 0.01)
        for i in range(4):
            assert store(writer, i)
        receipt = writer.handoff.publish("request", 2)
        assert writer.handoff.publish("request", 2) == receipt
        assert reader.handoff.adopt(receipt, keys, 0, layout)
        assert not reader.handoff.adopt(
            replace(receipt, writer_epoch="stale"), keys, 0, layout
        )
        assert not reader.handoff.adopt(
            replace(receipt, manifest_digest="bad"), keys, 0, layout
        )
        assert not reader.handoff.adopt(
            receipt, keys, 0, MemoryLayoutDesc([torch.Size([512, 2])], [torch.float32])
        )
        assert not reader.handoff.adopt(
            receipt, keys, 0, MemoryLayoutDesc([torch.Size([1024])], [torch.int32])
        )
        identity = ReadAckIdentity(
            "request",
            "reader-incarnation",
            0,
            receipt.writer_epoch,
            receipt.checkpoint_seq,
            receipt.manifest_digest,
        )
        assert (
            writer.handoff.acknowledge(identity, "session") == ReadAckOutcome.REJECTED
        )
        assert writer.handoff.claim(identity, "session").granted
        writer.delete(keys)
        assert writer.report_status()["core"]["indexed_key_count"] == 4
        assert not store(writer, 9)
        assert (
            writer.handoff.acknowledge(
                replace(identity, consumer_instance_id="impostor"), "session"
            )
            == ReadAckOutcome.REJECTED
        )
        assert writer.handoff.report_status()["live_lease_count"] == 1
        assert writer.handoff.acknowledge(identity, "session") == ReadAckOutcome.APPLIED
        assert (
            writer.handoff.acknowledge(identity, "session")
            == ReadAckOutcome.ALREADY_APPLIED
        )
        assert store(writer, 9)
        assert writer.handoff.report_status()["live_lease_count"] == 0
    finally:
        reader.close()
        writer.close()


def test_changed_retry_and_shutdown_refuse_publication(tmp_path: Path) -> None:
    """No changed manifest or post-shutdown operation is admitted."""
    path = tmp_path / "shared.bin"
    path.write_bytes(bytes(5 * 65536))
    writer = adapter(path)
    try:
        writer.handoff.prepare("request", [_create_object_key(0)])
        with pytest.raises(RuntimeError, match="changed"):
            writer.handoff.prepare("request", [_create_object_key(1)])
        writer.handoff.close()
        with pytest.raises(RuntimeError, match="closed"):
            writer.handoff.publish("request", 1)
    finally:
        writer.close()


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_publication_requires_finite_positive_deadline(
    tmp_path: Path, timeout: float
) -> None:
    """Invalid deadlines cannot turn an incomplete publication into an endless wait."""
    path = tmp_path / "shared.bin"
    path.write_bytes(bytes(5 * 65536))
    writer = adapter(path)
    try:
        writer.handoff.prepare("request", [_create_object_key(0)])
        with pytest.raises(ValueError, match="finite and positive"):
            writer.handoff.publish("request", timeout)
    finally:
        writer.close()


def test_pd_derivation_config_survives_json_projection() -> None:
    """The descriptor returned in MP config output can be supplied as JSON again."""
    config = RawBlockL2AdapterConfig(
        device_path="/unused/config-only",
        slot_bytes=65536,
        pd_derivation=DERIVATION,
        use_odirect=False,
    )
    projected = json.loads(json.dumps(make_json_safe(vars(config))))
    assert isinstance(projected["pd_derivation"], dict)
    restored = RawBlockL2AdapterConfig.from_dict(projected)
    assert restored.to_core_config().derivation == config.to_core_config().derivation


def test_cancel_and_unread_release_do_not_drop_claimed_extents(tmp_path: Path) -> None:
    """Abandoned prepares retire; only exact never-claimed receipts release leases."""
    path = tmp_path / "shared.bin"
    path.write_bytes(bytes(5 * 65536))
    writer = adapter(path)
    keys = [_create_object_key(0)]
    try:
        writer.handoff.prepare("cancelled", keys)
        assert writer.handoff.cancel_prepared("cancelled")
        assert not writer.handoff.cancel_prepared("cancelled")
        assert writer.handoff.report_status()["request_count"] == 0
        assert writer.report_status()["core"]["publication_protected_key_count"] == 0
        with pytest.raises(KeyError):
            writer.handoff.publish("cancelled", 1)
        writer.handoff.prepare("unread", keys)
        assert store(writer, 0)
        receipt = writer.handoff.publish("unread", 1)
        assert not writer.handoff.cancel_prepared("unread")
        assert not writer.handoff.release_unread(
            "unread", replace(receipt, namespace_identity="different"), "group"
        ).released
        assert writer.handoff.report_status()["live_lease_count"] == 1
        assert writer.handoff.release_unread("unread", receipt, "group").released
        assert writer.handoff.release_unread("unread", receipt, "group").released
        writer.handoff.prepare("claimed", keys)
        claimed_receipt = writer.handoff.publish("claimed", 1)
        identity = ReadAckIdentity(
            "claimed",
            "reader",
            0,
            claimed_receipt.writer_epoch,
            claimed_receipt.checkpoint_seq,
            claimed_receipt.manifest_digest,
        )
        assert writer.handoff.claim(identity, "group").granted
        assert not writer.handoff.release_unread(
            "claimed", claimed_receipt, "group"
        ).released
        assert writer.handoff.acknowledge(identity, "group") == ReadAckOutcome.APPLIED
    finally:
        writer.close()
