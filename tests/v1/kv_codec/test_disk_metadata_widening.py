# SPDX-License-Identifier: Apache-2.0
"""DiskCacheMetadata widening for asymmetric KV.

These tests verify that when a `BytesBufferMemoryObj` carrying an
asymmetric encoded blob is inserted into the local-disk-style index,
the plural `metadata.shapes` / `metadata.dtypes` ride through to
`DiskCacheMetadata.shapes` / `.dtypes`, while the singular `dtype`
stays None (because the encoded blob is opaque-byte at the codec
boundary).

The actual disk-write codepath is heavyweight; we test the metadata
plumbing in isolation by calling `insert_key` directly with synthetic
inputs.  Real backend integration is covered by the existing local-
disk-roundtrip tests at the codec layer.
"""

# Standard
from dataclasses import dataclass

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.memory_management import MemoryFormat


def test_disk_cache_metadata_carries_plural_shapes_and_dtypes():
    """Catches: DiskCacheMetadata regression that drops the plural
    fields.  This is the storage-side dual of the
    MemoryObjMetadata.shapes/.dtypes additions."""
    md = DiskCacheMetadata(
        path="/tmp/foo.bin",
        size=1024,
        shape=torch.Size([1024, 0, 0, 0]),
        dtype=None,
        fmt=MemoryFormat.BINARY_BUFFER,
        shapes=[torch.Size([2, 4, 16, 32])],
        dtypes=[torch.float16, torch.float8_e4m3fn],
    )
    assert md.dtype is None
    assert md.dtypes == [torch.float16, torch.float8_e4m3fn]
    assert md.shapes == [torch.Size([2, 4, 16, 32])]


def test_disk_cache_metadata_plural_default_none():
    """Catches: required-field regression when the plural fields
    are omitted.  Existing call sites that pass only positional
    args to DiskCacheMetadata must keep working."""
    md = DiskCacheMetadata(
        path="/tmp/foo.bin",
        size=1024,
    )
    assert md.shapes is None
    assert md.dtypes is None
    # Singular fields also default sensibly.
    assert md.shape is None
    assert md.dtype is None
    assert md.fmt is None


def test_disk_cache_metadata_singular_only_still_works():
    """Backwards compat: existing storage backends call
    DiskCacheMetadata with positional shape/dtype/etc. and no
    knowledge of the plural fields.  The dataclass must accept that."""
    md = DiskCacheMetadata(
        "/tmp/foo.bin",  # path
        1024,            # size
        torch.Size([2, 4, 16, 32]),  # shape
        torch.float16,   # dtype
        None,            # cached_positions
        MemoryFormat.KV_2LTD,  # fmt
        0,               # pin_count
    )
    assert md.dtype == torch.float16
    assert md.shapes is None
    assert md.dtypes is None


def test_disk_cache_metadata_pin_unpin_unchanged():
    """Pin/unpin behavior must not regress with the new fields."""
    md = DiskCacheMetadata(
        path="/tmp/foo.bin",
        size=1024,
        shapes=[torch.Size([2, 4, 16, 32])],
        dtypes=[torch.float16, torch.float8_e4m3fn],
    )
    assert not md.is_pinned
    md.pin()
    assert md.is_pinned
    md.unpin()
    assert not md.is_pinned


def test_local_disk_insert_key_signature_accepts_plural():
    """Catches: local_disk_backend.insert_key signature regression
    that breaks the asym KV write path."""
    # First Party
    from lmcache.v1.storage_backend.local_disk_backend import (
        LocalDiskBackend,
    )

    # We don't construct a full LocalDiskBackend (heavy deps);
    # just inspect the method signature.
    # Standard
    import inspect

    sig = inspect.signature(LocalDiskBackend.insert_key)
    params = sig.parameters
    assert "shapes" in params, (
        "LocalDiskBackend.insert_key should accept `shapes` for "
        "asymmetric KV; the asym serde uses metadata.shapes plural "
        "to carry the (K, V) layout."
    )
    assert "dtypes" in params, (
        "LocalDiskBackend.insert_key should accept `dtypes` for "
        "asymmetric KV; metadata.dtypes plural carries [K_dtype, "
        "V_dtype]."
    )
    # The plural defaults must be Optional/None for backwards compat.
    assert params["shapes"].default is None
    assert params["dtypes"].default is None
