# SPDX-License-Identifier: Apache-2.0
"""An explicit namespace identity flows from the MP adapter config to receipts."""

# Standard
from dataclasses import replace
from pathlib import Path
import json

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.storage_backend.raw_block import ReadAckIdentity, ReadAckOutcome
from lmcache.v1.utils.json_utils import make_json_safe
from tests.v1.distributed.test_raw_block_l2_adapter import (
    _create_object_key,
    requires_raw_block_ext,
)
from tests.v1.distributed.test_raw_block_mp_handoff import DERIVATION, store

IDENTITY = "test:shared-namespace:0001"
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])


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
            capacity_bytes=5 * 65536,
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
    path.write_bytes(bytes(5 * 65536))
    return path


def ack_identity(receipt) -> ReadAckIdentity:
    return ReadAckIdentity(
        "request",
        "reader",
        0,
        receipt.writer_epoch,
        receipt.checkpoint_seq,
        receipt.manifest_digest,
    )


def test_config_parses_validates_and_forwards_namespace_identity() -> None:
    """The JSON field is parsed, validated and forwarded to the core."""
    base = {"device_path": "/unused/config-only", "slot_bytes": 65536}
    config = RawBlockL2AdapterConfig.from_dict(
        {**base, "namespace_identity": "test:ns"}
    )
    assert config.namespace_identity == "test:ns"
    assert config.to_core_config().namespace_identity == "test:ns"
    default = RawBlockL2AdapterConfig.from_dict(base)
    assert default.to_core_config().namespace_identity == ""
    assert "namespace_identity" in RawBlockL2AdapterConfig.help()
    for value in (7, None):
        with pytest.raises(ValueError, match="string"):
            RawBlockL2AdapterConfig.from_dict({**base, "namespace_identity": value})
    with pytest.raises(ValueError, match="shareable"):
        RawBlockL2AdapterConfig(
            device_path="/unused/config-only",
            slot_bytes=65536,
            # A block device without a hardware identity is known only to the
            # host that opened it, so its derived identity cannot be shared.
            namespace_identity="block-local:8:0",
        )


def test_namespace_identity_survives_json_projection() -> None:
    """The MP config output can be supplied as JSON again without losing it."""
    config = RawBlockL2AdapterConfig(
        device_path="/unused/config-only",
        slot_bytes=65536,
        pd_derivation=DERIVATION,
        namespace_identity=IDENTITY,
        use_odirect=False,
    )
    projected = json.loads(json.dumps(make_json_safe(vars(config))))
    restored = RawBlockL2AdapterConfig.from_dict(projected)
    assert restored.namespace_identity == IDENTITY
    assert restored.to_core_config().namespace_identity == IDENTITY


@requires_raw_block_ext
def test_default_identity_is_still_derived_from_the_file(tmp_path: Path) -> None:
    """An empty field keeps the device and inode identity of a regular file."""
    writer = adapter(blank(tmp_path / "shared.bin"), identity="")
    try:
        assert core_status(writer)["namespace_identity"].startswith("file:")
    finally:
        writer.close()


@requires_raw_block_ext
def test_explicit_identity_is_carried_in_receipts_and_checked_on_adoption(
    tmp_path: Path,
) -> None:
    """Writer and reader agree on the operator identity; altered receipts fail."""
    path = blank(tmp_path / "shared.bin")
    writer = adapter(path)
    reader = adapter(path, "reader")
    keys = [_create_object_key(i) for i in range(2)]
    try:
        assert core_status(writer)["namespace_identity"] == IDENTITY
        assert core_status(reader)["namespace_identity"] == IDENTITY
        writer.handoff.prepare("request", keys)
        assert all(store(writer, i) for i in range(2))
        receipt = writer.handoff.publish("request", 2)
        assert receipt.namespace_identity == IDENTITY
        identity = ack_identity(receipt)
        assert writer.handoff.claim(identity, "session").granted
        assert reader.handoff.adopt(receipt, keys, 0, LAYOUT)
        altered = replace(receipt, namespace_identity="file:0:0")
        assert not reader.handoff.adopt(altered, keys, 0, LAYOUT)
        assert writer.handoff.acknowledge(identity, "session") == ReadAckOutcome.APPLIED
    finally:
        reader.close()
        writer.close()


@requires_raw_block_ext
def test_reader_with_another_identity_refuses_the_same_file(tmp_path: Path) -> None:
    """Two operators naming the same bytes differently do not share receipts.

    The writer provisions the image for its identity at open, so a reader
    configured with another identity is refused when it opens the image.
    """
    path = blank(tmp_path / "shared.bin")
    writer = adapter(path)
    try:
        with pytest.raises(RuntimeError, match="namespace_identity mismatch"):
            adapter(path, "reader", identity="test:other-namespace")
    finally:
        writer.close()


@requires_raw_block_ext
def test_reused_identifier_on_an_independent_image_is_refused(tmp_path: Path) -> None:
    """An operator identifier is an assertion, not proof of the image.

    Two independently initialized images given the same identifier still
    refuse each other's receipts and claims.
    """
    writer_a = adapter(blank(tmp_path / "a.bin"))
    image_b = blank(tmp_path / "b.bin")
    writer_b = adapter(image_b)
    reader_b = adapter(image_b, "reader")
    keys = [_create_object_key(i) for i in range(2)]
    try:
        writer_a.handoff.prepare("request", keys)
        assert all(store(writer_a, i) for i in range(2))
        receipt_a = writer_a.handoff.publish("request", 2)
        assert receipt_a.namespace_identity == IDENTITY
        # Same identifier, but image b has published nothing: refused.
        assert not reader_b.handoff.adopt(receipt_a, keys, 100, LAYOUT)
        writer_b.handoff.prepare("request", keys)
        assert all(store(writer_b, i) for i in range(2))
        receipt_b = writer_b.handoff.publish("request", 2)
        # Same identifier and manifest on both images: image b adopts only
        # its own receipt, and its writer grants no claim on image a's.
        assert receipt_b.namespace_identity == receipt_a.namespace_identity
        assert receipt_b.manifest_digest == receipt_a.manifest_digest
        assert not reader_b.handoff.adopt(receipt_a, keys, 100, LAYOUT)
        assert not writer_b.handoff.claim(ack_identity(receipt_a), "session").granted
        assert writer_b.handoff.claim(ack_identity(receipt_b), "session").granted
        assert reader_b.handoff.adopt(receipt_b, keys, 100, LAYOUT)
    finally:
        reader_b.close()
        writer_b.close()
        writer_a.close()
