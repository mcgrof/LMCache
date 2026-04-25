# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Tuple

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.naive_serde.asym_serde import (
    AsymK16V8Deserializer,
    AsymK16V8Serializer,
)
from lmcache.v1.storage_backend.naive_serde.cachegen_decoder import CacheGenDeserializer
from lmcache.v1.storage_backend.naive_serde.cachegen_encoder import CacheGenSerializer
from lmcache.v1.storage_backend.naive_serde.kivi_serde import (
    KIVIDeserializer,
    KIVISerializer,
)
from lmcache.v1.storage_backend.naive_serde.naive_serde import (
    NaiveDeserializer,
    NaiveSerializer,
)
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer


def CreateSerde(
    serde_type: str,
    metadata: LMCacheMetadata,
    config: LMCacheEngineConfig,
) -> Tuple[Serializer, Deserializer]:
    s: Optional[Serializer] = None
    d: Optional[Deserializer] = None

    if serde_type == "naive":
        s, d = NaiveSerializer(), NaiveDeserializer()
    elif serde_type == "kivi":
        s, d = KIVISerializer(), KIVIDeserializer()
    elif serde_type == "cachegen":
        s, d = (
            CacheGenSerializer(config, metadata),
            CacheGenDeserializer(config, metadata),
        )
    elif serde_type == "asym_k16_v8_e4m3":
        # Storage-only mode: V is FP8 on disk, dequantized back to
        # the input dtype on read.  Native_asym mode is Phase 4.
        s, d = (
            AsymK16V8Serializer(config, metadata),
            AsymK16V8Deserializer(config, metadata),
        )
    else:
        raise ValueError(f"Invalid type: {serde_type}")

    return s, d


__all__ = [
    "Serializer",
    "Deserializer",
    "KIVISerializer",
    "KIVIDeserializer",
    "AsymK16V8Serializer",
    "AsymK16V8Deserializer",
    "CreateSerde",
]
