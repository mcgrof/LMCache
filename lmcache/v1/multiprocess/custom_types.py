# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Any, Callable
import pickle
import threading

# Third Party
import msgspec
import torch

"""
Defines the types and the customized encoder/decoders for inter-process
communications.

Key Types:
- IPCCacheEngineKey: Token-based cache key
  - Contains token_ids, start, end, request_id, and optional descriptor-issued keys
  - Converted to ObjectKey for storage operations via ipc_key_to_object_keys()
"""


class CudaIPCWrapper:
    _discovered_device_mapping: dict[str, int] = {}
    _device_mapping_lock = threading.Lock()

    @staticmethod
    def _get_device_uuid(device_index: int) -> str:
        """Get the UUID of a GPU device given its index."""
        return str(torch.cuda.get_device_properties(device_index).uuid)

    @staticmethod
    def _discover_gpu_devices():
        """Discover all available GPU devices and map their UUIDs to
        the physical device ordinals.
        """
        if not torch.cuda.is_available():
            return

        num_devices = torch.cuda.device_count()
        with CudaIPCWrapper._device_mapping_lock:
            if CudaIPCWrapper._discovered_device_mapping:
                return  # Already discovered

            for i in range(num_devices):
                device_uuid = CudaIPCWrapper._get_device_uuid(i)
                CudaIPCWrapper._discovered_device_mapping[device_uuid] = i

    @staticmethod
    def _get_device_index_from_uuid(device_uuid: str) -> int:
        """Get the physical device ordinal from its UUID."""
        CudaIPCWrapper._discover_gpu_devices()

        with CudaIPCWrapper._device_mapping_lock:
            device_index = CudaIPCWrapper._discovered_device_mapping.get(
                device_uuid, None
            )

        if device_index is None:
            raise RuntimeError(
                f"Device UUID {device_uuid} not found in the discovered devices."
                "Please make sure the process can see all the GPU devices"
            )
        return device_index

    def __init__(self, tensor: torch.Tensor):
        # First Party
        from lmcache.v1.gpu_connector.utils import assert_contiguous

        assert_contiguous(tensor)

        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = CudaIPCWrapper._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """
        Note:
            This function may break if torch cuda is not initialized.
            We should call `torch.cuda.init()` before using this function.
        """
        device_index = CudaIPCWrapper._get_device_index_from_uuid(self.device_uuid)

        storage = torch.UntypedStorage._new_shared_cuda(  # noqa: SLF001
            device_index, *self.handle[1:]
        )

        t = torch.empty((), device=f"cuda:{device_index}", dtype=self.dtype)
        t.set_(storage, self.storage_offset, self.shape, self.stride)
        return t

    def __eq__(self, other):
        if not isinstance(other, CudaIPCWrapper):
            return False
        return (
            self.handle == other.handle
            and self.dtype == other.dtype
            and self.shape == other.shape
            and self.stride == other.stride
            and self.storage_offset == other.storage_offset
            and self.device_uuid == other.device_uuid
        )

    @staticmethod
    def Serialize(obj: "CudaIPCWrapper") -> bytes:
        return pickle.dumps(obj)

    @staticmethod
    def Deserialize(data: bytes) -> "CudaIPCWrapper":
        return pickle.loads(data)


@dataclass(order=True, frozen=True)
class ExternalKVKeys:
    """Opaque cache keys issued by a resolved KV request descriptor.

    The connector and cache server must not reconstruct these values from
    tokens.  Keeping the keys in one frozen protocol object makes presence
    explicit across both process boundaries and lets consumers fail closed
    when a provenance-required operation loses the object.
    """

    keys: tuple[bytes, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported external KV key schema version: {self.schema_version}"
            )
        for key in self.keys:
            if not isinstance(key, bytes) or len(key) != 32:
                raise ValueError("external KV keys must be 32-byte values")


@dataclass(order=True, frozen=True)
class IPCCacheEngineKey:
    """Cache key for the IPC (multiprocess) protocol.

    This key type is sent by the client over ZMQ (serialized via msgspec).

    In legacy mode the server computes chunk hashes from token_ids.  When
    external_keys is present, those opaque descriptor-issued values are the
    only storage address material; malformed external keys are errors and
    never fall back to token hashing.

    The request_id field is for session tracking and is NOT included
    in equality/hash comparisons (two keys with same content but different
    request_ids are considered equal for cache purposes).
    """

    model_name: str
    world_size: int
    worker_id: int | None

    token_ids: tuple[int, ...]  # frozen tuple for hashability
    start: int
    end: int

    # === Session tracking (not part of cache identity) ===
    request_id: str = field(compare=False)
    external_keys: ExternalKVKeys | None = None

    # Helper function for unit tests only
    @classmethod
    def from_token_ids(
        cls,
        model_name: str,
        world_size: int,
        worker_id: int | None,
        token_ids: list[int],
        start: int = 0,
        end: int = 0,
        request_id: str = "",
        external_keys: ExternalKVKeys | None = None,
    ) -> "IPCCacheEngineKey":
        """Create a key from token ids. Only used by the tests."""
        return cls(
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            external_keys=external_keys,
        )

    def no_worker_id_version(self) -> "IPCCacheEngineKey":
        """Create a copy with worker_id=None for lookup requests."""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            token_ids=self.token_ids,
            start=self.start,
            end=self.end,
            request_id=self.request_id,
            external_keys=self.external_keys,
        )


def resolve_external_chunk_hashes(
    key: IPCCacheEngineKey,
    chunk_size: int,
) -> list[bytes] | None:
    """Validate descriptor-issued keys for an IPC operation.

    ``None`` selects the explicit legacy token-key mode.  Once an external key
    object is present, every shape or value error is fatal; callers must never
    recover by hashing tokens because that would address a different namespace.
    """
    if key.external_keys is None:
        return None
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not 0 <= key.start <= key.end <= len(key.token_ids):
        raise ValueError(
            "external KV key range must satisfy 0 <= start <= end <= len(token_ids)"
        )
    if key.start % chunk_size != 0 or key.end % chunk_size != 0:
        raise ValueError("external KV key range must be chunk aligned")
    expected = (key.end - key.start) // chunk_size
    if len(key.external_keys.keys) != expected:
        raise ValueError(
            "external KV key count does not match the operation range: "
            f"expected {expected}, got {len(key.external_keys.keys)}"
        )
    return list(key.external_keys.keys)


def reject_external_kv_keys(key: IPCCacheEngineKey, operation: str) -> None:
    """Prevent an unsupported path from silently rehashing tokens."""
    if key.external_keys is not None:
        raise ValueError(
            f"{operation} does not support descriptor-issued external KV keys"
        )


# Type exports
KVCache = list[CudaIPCWrapper]


@dataclass
class CustomizedSerdeConfig:
    serializer: Callable[[Any], bytes]
    deserializer: Callable[[bytes], Any]
    code: int


_CUSTOMERIZED_SERIALIZERS = {
    CudaIPCWrapper: CustomizedSerdeConfig(
        serializer=CudaIPCWrapper.Serialize,
        deserializer=CudaIPCWrapper.Deserialize,
        code=1,
    ),
}


def get_customized_encoder(type: Any) -> msgspec.msgpack.Encoder:
    # TODO: `type` is not used here
    def enc_hook(obj: Any) -> Any:
        for supported_type, cfg in _CUSTOMERIZED_SERIALIZERS.items():
            if isinstance(obj, supported_type):
                data = cfg.serializer(obj)
                return msgspec.msgpack.Ext(cfg.code, data)
        raise TypeError(f"Unsupported type for serialization: {type(obj)}")

    return msgspec.msgpack.Encoder(enc_hook=enc_hook)


def get_customized_decoder(type: Any) -> msgspec.msgpack.Decoder:
    def ext_hook(code: int, data: bytes) -> Any:
        for cfg in _CUSTOMERIZED_SERIALIZERS.values():
            if cfg.code == code:
                return cfg.deserializer(data)
        raise TypeError(f"Unsupported ext code for deserialization: {code}")

    return msgspec.msgpack.Decoder(ext_hook=ext_hook, type=type)


@dataclass
class BlockAllocationRecord:
    """A single per-request GPU block allocation delta from vLLM."""

    req_id: str
    new_block_ids: list[int]
    new_token_ids: list[int]


@dataclass
class CBMatchResult:
    """Result of a sub-sequence match from BlendTokenRangeMatcher.

    Attributes:
        old_st: Start position in the originally registered (stored) sequence.
        old_ed: End position in the originally registered (stored) sequence.
        cur_st: Start position in the query sequence where the match was found.
        cur_ed: End position in the query sequence where the match was found.
        hash: Token hash bytes (from registration) used as the storage key.
    """

    old_st: int
    old_ed: int
    cur_st: int
    cur_ed: int
    hash: bytes
