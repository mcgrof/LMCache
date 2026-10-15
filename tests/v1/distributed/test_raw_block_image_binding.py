# SPDX-License-Identifier: Apache-2.0
"""A strict shared namespace is bound to the image it was provisioned on.

An operator's ``namespace_identity`` says which nodes share a namespace; it
does not say which bytes they share. The first strict writer therefore
provisions the image behind its device path: it mints an image identity,
records it with the namespace it was provisioned for in a checkpoint, and
reads that checkpoint back before it serves. Receipts name the image, strict
owners refuse an image provisioned for another namespace, and a reader
refuses a receipt from another image even when the operator name, manifest,
writer epoch and checkpoint sequence all match.
"""

# Standard
from dataclasses import asdict, replace
from pathlib import Path
import os
import shutil

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.storage_backend.raw_block import ReadAckIdentity, ReadAckOutcome
from lmcache.v1.storage_backend.raw_block.core import (
    RawBlockCore,
    RawBlockPublicationReceipt,
)
from lmcache.v1.storage_backend.storage_pd_protocol import StoragePDStatus
from tests.v1.distributed.test_raw_block_l2_adapter import (
    _create_object_key,
    requires_raw_block_ext,
)
from tests.v1.distributed.test_raw_block_mp_handoff import DERIVATION, store

IDENTITY = "test:shared-namespace:0001"
OTHER_IDENTITY = "test:other-namespace:0002"
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
SLOTS = 5


def adapter(
    path: Path, role: str = "writer", identity: str = IDENTITY
) -> RawBlockL2Adapter:
    """Open one owner of a four-slot shared namespace."""
    return RawBlockL2Adapter(
        RawBlockL2AdapterConfig(
            device_path=str(path),
            role=role,
            pd_derivation=DERIVATION,
            namespace_identity=identity,
            capacity_bytes=SLOTS * 65536,
            slot_bytes=65536,
            meta_total_bytes=65536,
            use_odirect=False,
            meta_enable_periodic=False,
        )
    )


def core_status(owner: RawBlockL2Adapter) -> dict:
    """Return the raw-block core's part of an owner's status."""
    return owner.report_status()["core"]


def blank(path: Path) -> Path:
    path.write_bytes(bytes(SLOTS * 65536))
    return path


def publish(writer: RawBlockL2Adapter, count: int):
    """Store and publish ``count`` objects as one request."""
    keys = [_create_object_key(i) for i in range(count)]
    writer.handoff.prepare("request", keys)
    assert all(store(writer, i) for i in range(count))
    return keys, writer.handoff.publish("request", 2)


def ack_identity(receipt) -> ReadAckIdentity:
    return ReadAckIdentity(
        "request",
        "reader",
        0,
        receipt.writer_epoch,
        receipt.checkpoint_seq,
        receipt.manifest_digest,
    )


@requires_raw_block_ext
def test_strict_writer_provisions_the_image_at_open(tmp_path: Path) -> None:
    """Opening a blank image writes a checkpoint that names the image."""
    path = blank(tmp_path / "shared.bin")
    writer = adapter(path)
    try:
        status = core_status(writer)
        assert len(status["image_id"]) == 32
        assert status["metadata_seq"] == 1
        # A fresh owner reads the same identity back from the device.
        reader = adapter(path, "reader")
        try:
            assert core_status(reader)["image_id"] == status["image_id"]
        finally:
            reader.close()
    finally:
        writer.close()


@requires_raw_block_ext
def test_restarted_writer_keeps_the_image_identity(tmp_path: Path) -> None:
    """The image identity outlives the writer; the writer epoch does not."""
    path = blank(tmp_path / "shared.bin")
    first = adapter(path)
    before = core_status(first)
    first.close()
    second = adapter(path)
    try:
        after = core_status(second)
        assert after["image_id"] == before["image_id"]
        assert after["writer_epoch"] != before["writer_epoch"]
    finally:
        second.close()


@requires_raw_block_ext
def test_owners_refuse_an_image_provisioned_for_another_namespace(
    tmp_path: Path,
) -> None:
    """Neither role may open an image that another namespace provisioned."""
    path = blank(tmp_path / "shared.bin")
    writer = adapter(path)
    try:
        with pytest.raises(RuntimeError, match="namespace_identity mismatch"):
            adapter(path, "reader", identity=OTHER_IDENTITY)
    finally:
        writer.close()
    with pytest.raises(RuntimeError, match="namespace_identity mismatch"):
        adapter(path, "writer", identity=OTHER_IDENTITY)
    # The namespace the image was provisioned for still opens it.
    adapter(path).close()


@requires_raw_block_ext
def test_receipt_from_another_image_is_refused_by_the_binding_alone(
    tmp_path: Path,
) -> None:
    """A receipt naming another image is refused even when all else matches.

    The operator name, manifest, writer epoch and checkpoint sequence all
    agree with image b's own receipt; only the provisioned image differs.
    """
    writer_a = adapter(blank(tmp_path / "a.bin"))
    image_b = blank(tmp_path / "b.bin")
    writer_b = adapter(image_b)
    reader_b = adapter(image_b, "reader")
    try:
        keys, receipt_a = publish(writer_a, 2)
        _, receipt_b = publish(writer_b, 2)
        assert receipt_a.image_id == core_status(writer_a)["image_id"]
        assert receipt_b.image_id == core_status(writer_b)["image_id"]
        assert receipt_a.image_id != receipt_b.image_id
        assert receipt_a.manifest_digest == receipt_b.manifest_digest
        identity = ack_identity(receipt_b)
        assert writer_b.handoff.claim(identity, "session").granted
        forged = replace(receipt_b, image_id=receipt_a.image_id)
        assert not reader_b.handoff.adopt(forged, keys, 100, LAYOUT)
        rewritten = replace(
            receipt_a,
            writer_epoch=receipt_b.writer_epoch,
            checkpoint_seq=receipt_b.checkpoint_seq,
        )
        assert not reader_b.handoff.adopt(rewritten, keys, 100, LAYOUT)
        assert reader_b.handoff.adopt(receipt_b, keys, 100, LAYOUT)
        assert core_status(reader_b)["image_id"] == receipt_b.image_id
        assert (
            writer_b.handoff.acknowledge(identity, "session") == ReadAckOutcome.APPLIED
        )
    finally:
        reader_b.close()
        writer_b.close()
        writer_a.close()


@requires_raw_block_ext
def test_receipt_image_survives_both_wire_encodings(tmp_path: Path) -> None:
    """The HTTP route's scalar encoding and the READY status keep the image."""
    writer = adapter(blank(tmp_path / "shared.bin"))
    try:
        _, receipt = publish(writer, 1)
        wire = asdict(receipt)
        wire.pop("payload_provenance")
        assert wire["image_id"] == core_status(writer)["image_id"]
        assert RawBlockPublicationReceipt(**wire) == receipt
        status = StoragePDStatus.ready("request", 0, receipt)
        encoded = msgspec.msgpack.encode(status)
        decoded = msgspec.msgpack.decode(encoded, type=StoragePDStatus)
        assert decoded.publication_receipt() == receipt
    finally:
        writer.close()


@requires_raw_block_ext
def test_provisioning_readback_mismatch_refuses_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer must read its provisioning checkpoint back from the device."""
    # A readback disagrees with the write only when the device loses or
    # redirects it; no public seam can produce that, so substitute one.
    monkeypatch.setattr(
        RawBlockCore, "_read_provisioned_image_id", lambda self: "elsewhere"
    )
    with pytest.raises(RuntimeError, match="readback mismatch"):
        adapter(blank(tmp_path / "shared.bin"))


@requires_raw_block_ext
def test_byte_identical_copy_keeps_the_image_identity(tmp_path: Path) -> None:
    """Documented limit: the binding names the provisioned image, not a device.

    A copy of the image carries the same identity and checkpoint, so a reader
    reaching the copy through the same path string adopts it. The sequence,
    manifest and per-slot header checks still bound what such a copy can
    serve.
    """
    original = blank(tmp_path / "original.bin")
    link = tmp_path / "namespace.img"
    os.symlink(original, link)
    writer = adapter(link)
    try:
        keys, receipt = publish(writer, 2)
    finally:
        writer.close()
    copy = tmp_path / "copy.bin"
    shutil.copyfile(original, copy)
    os.unlink(link)
    os.symlink(copy, link)
    reader = adapter(link, "reader")
    try:
        assert core_status(reader)["image_id"] == receipt.image_id
        assert reader.handoff.adopt(receipt, keys, 100, LAYOUT)
    finally:
        reader.close()
