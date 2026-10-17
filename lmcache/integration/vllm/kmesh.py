# SPDX-License-Identifier: Apache-2.0
"""Strict LMCache-local parsing for the vLLM KMesh request sidecar.

This module deliberately mirrors only the immutable wire contract.  It does
not import KMesh implementation types from vLLM, so LMCache can validate an
external connector's evidence without coupling its process lifetime to vLLM's
Python modules.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias, cast
import hashlib
import json
import math

# First Party
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan

KMESH_REQUEST_SIDECAR_PROVIDER_ID = "vllm.kmesh.request_lookup.v1"
KMESH_REQUEST_LOOKUP_WIRE_TAG = "kmesh-request-lookup-v1"

KMESH_DOMAIN_TAG_KEY = "lmcache.tag.kmesh.domain"
KMESH_RUNTIME_TAG_KEY = "lmcache.tag.kmesh.runtime"
KMESH_RESERVED_TAG_KEYS = frozenset((KMESH_DOMAIN_TAG_KEY, KMESH_RUNTIME_TAG_KEY))

_VERSIONED_SHA256_PREFIX = "sha256-v1:"
_MAX_IDENTIFIER_LENGTH = 256
_MAX_RUNTIME_EVIDENCE_LENGTH = 4096

KMeshWireValue: TypeAlias = str | int | bool | None | tuple["KMeshWireValue", ...]
KMeshRequestConfigValue: TypeAlias = str | int | float | bool | None


class KMeshRequestLookupIntent(str, Enum):
    """Actionability declared by the request-sidecar provider."""

    OBSERVE_ONLY = "observe_only"
    SERIAL_PIC = "serial_pic"


class KMeshSegmentKind(str, Enum):
    """Segment kinds accepted by the v1 KMesh wire contract."""

    PRIVATE_HISTORY = "private_history"
    SHARED_UPDATE = "shared_update"
    TASK = "task"
    SYSTEM = "system"
    THOUGHT = "thought"
    TOOL_RESULT = "tool_result"
    OTHER = "other"


class KMeshFallbackReason(str, Enum):
    """Fallback values accepted by the v1 KMesh wire contract."""

    DISABLED = "disabled"
    NO_METADATA = "no_metadata"
    INVALID_METADATA = "invalid_metadata"
    UNSUPPORTED_MODEL = "unsupported_model"
    GROUP_KIND_UNSUPPORTED = "group_kind_unsupported"
    STATE_KIND_UNSUPPORTED = "state_kind_unsupported"
    MISSING_COMPATIBILITY_EVIDENCE = "missing_compatibility_evidence"
    MISSING_PROMPT_TOKEN_EVIDENCE = "missing_prompt_token_evidence"
    PROMPT_TOKEN_LENGTH_MISMATCH = "prompt_token_length_mismatch"
    SEGMENT_CONTENT_MISMATCH = "segment_content_mismatch"
    EXECUTION_ID_MISMATCH = "execution_id_mismatch"
    GROUP_ID_MISMATCH = "group_id_mismatch"
    ROUND_MISMATCH = "round_mismatch"
    MODEL_MISMATCH = "model_mismatch"
    TOKENIZER_MISMATCH = "tokenizer_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    KV_CACHE_DTYPE_MISMATCH = "kv_cache_dtype_mismatch"
    KV_LAYOUT_MISMATCH = "kv_layout_mismatch"
    ATTENTION_KIND_MISMATCH = "attention_kind_mismatch"
    ROPE_CONFIGURATION_MISMATCH = "rope_configuration_mismatch"
    ATTENTION_BACKEND_MISMATCH = "attention_backend_mismatch"
    TENSOR_PARALLEL_MISMATCH = "tensor_parallel_mismatch"
    PIPELINE_PARALLEL_MISMATCH = "pipeline_parallel_mismatch"
    LORA_MISMATCH = "lora_mismatch"
    CACHE_SALT_MISMATCH = "cache_salt_mismatch"
    CACHE_DOMAIN_MISMATCH = "cache_domain_mismatch"
    ACTIVE_LENGTH_MISMATCH = "active_length_mismatch"
    CACHED_SPAN_MISMATCH = "cached_span_mismatch"
    PREFILL_CHUNK_MISMATCH = "prefill_chunk_mismatch"
    NO_SHARED_SEGMENTS = "no_shared_segments"
    SINGLETON_GROUP = "singleton_group"
    SLOT_MAPPING_OVERLAP = "slot_mapping_overlap"
    CHUNKED_PREFILL_UNSUPPORTED = "chunked_prefill_unsupported"
    ASYNC_SCHEDULING_UNSUPPORTED = "async_scheduling_unsupported"
    PIC_BACKEND_UNAVAILABLE = "pic_backend_unavailable"
    PIC_CACHE_MISS = "pic_cache_miss"
    GROUP_TIMEOUT = "group_timeout"
    INTERNAL_ERROR = "internal_error"


def _wire_tuple(
    value: object, field_name: str, *, length: int | None = None
) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple")
    if length is not None and len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} items")
    return value


def _strict_string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    return value


def _strict_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    return value


def _identifier(value: object, field_name: str) -> str:
    value = _strict_string(value, field_name)
    if (
        not value
        or value != value.strip()
        or len(value) > _MAX_IDENTIFIER_LENGTH
        or any(
            ord(character) < 32
            or ord(character) == 127
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    ):
        raise ValueError(
            f"{field_name} must be nonempty, trimmed, bounded, and contain "
            "no controls or surrogates"
        )
    return value


def _optional_identifier(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, field_name)


def _runtime_string(value: object, field_name: str) -> str:
    value = _strict_string(value, field_name)
    if (
        not value
        or value != value.strip()
        or len(value) > _MAX_RUNTIME_EVIDENCE_LENGTH
        or any(
            ord(character) < 32
            or ord(character) == 127
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    ):
        raise ValueError(
            f"{field_name} must be nonempty, trimmed, bounded runtime evidence"
        )
    return value


def _versioned_sha256(domain: bytes, payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(domain + b"\0" + canonical).hexdigest()
    return f"{_VERSIONED_SHA256_PREFIX}{digest}"


def kmesh_token_slice_digest(token_ids: tuple[int, ...]) -> str:
    """Recompute the exact vLLM ``sha256-v1`` token-slice digest."""
    validated = _token_tuple(token_ids, "token_ids")
    return _versioned_sha256(b"kmesh-token-slice-v1", list(validated))


def _token_tuple(value: object, field_name: str) -> tuple[int, ...]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{field_name} must be a list or tuple of token IDs")
    values = cast(list[object] | tuple[object, ...], value)
    validated: list[int] = []
    for index, token_id in enumerate(values):
        token_id = _strict_int(token_id, f"{field_name}[{index}]")
        if token_id < 0:
            raise ValueError(f"{field_name}[{index}] must be nonnegative")
        validated.append(token_id)
    return tuple(validated)


def _enum_value(value: object, enum_type: type[Enum], field_name: str) -> Enum:
    value = _strict_string(value, field_name)
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} has an unknown value {value!r}") from exc


@dataclass(frozen=True, slots=True)
class KMeshRequestLookupSpan:
    """One strict LMCache-local copy of an admitted KMesh span."""

    segment_id: str
    kind: KMeshSegmentKind
    token_length: int
    source_id: str | None
    lineage_id: str | None
    shareable: bool
    content_digest: str
    token_start: int
    token_end: int
    logical_order: int

    def __post_init__(self) -> None:
        _identifier(self.segment_id, "span.segment_id")
        if type(self.kind) is not KMeshSegmentKind:
            raise TypeError("span.kind must be KMeshSegmentKind")
        if type(self.token_length) is not int:
            raise TypeError("span.token_length must be an integer")
        if self.token_length <= 0:
            raise ValueError("span.token_length must be positive")
        _optional_identifier(self.source_id, "span.source_id")
        _optional_identifier(self.lineage_id, "span.lineage_id")
        if self.shareable is not True:
            raise ValueError("span.shareable must be true")
        if type(self.content_digest) is not str:
            raise TypeError("span.content_digest must be a string")
        if (
            not self.content_digest.startswith(_VERSIONED_SHA256_PREFIX)
            or len(self.content_digest) != len(_VERSIONED_SHA256_PREFIX) + 64
        ):
            raise ValueError("span.content_digest must be a versioned SHA-256 digest")
        digest_hex = self.content_digest[len(_VERSIONED_SHA256_PREFIX) :]
        if any(character not in "0123456789abcdef" for character in digest_hex):
            raise ValueError("span.content_digest must use lowercase hexadecimal")
        for field_name in ("token_start", "token_end", "logical_order"):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"span.{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"span.{field_name} must be nonnegative")
        if self.token_end <= self.token_start:
            raise ValueError("span token range must be nonempty and ordered")
        if self.token_end - self.token_start != self.token_length:
            raise ValueError("span token range must match span.token_length")

    @classmethod
    def from_wire_payload(cls, payload: object) -> KMeshRequestLookupSpan:
        fields = _wire_tuple(payload, "span", length=10)
        (
            segment_id,
            kind,
            token_length,
            source_id,
            lineage_id,
            shareable,
            content_digest,
            token_start,
            token_end,
            logical_order,
        ) = fields
        if type(shareable) is not bool:
            raise TypeError("span.shareable must be a boolean")
        return cls(
            segment_id=_identifier(segment_id, "span.segment_id"),
            kind=_enum_value(kind, KMeshSegmentKind, "span.kind"),  # type: ignore[arg-type]
            token_length=_strict_int(token_length, "span.token_length"),
            source_id=_optional_identifier(source_id, "span.source_id"),
            lineage_id=_optional_identifier(lineage_id, "span.lineage_id"),
            shareable=shareable,
            content_digest=_strict_string(content_digest, "span.content_digest"),
            token_start=_strict_int(token_start, "span.token_start"),
            token_end=_strict_int(token_end, "span.token_end"),
            logical_order=_strict_int(logical_order, "span.logical_order"),
        )

    def to_wire_payload(self) -> KMeshWireValue:
        return (
            self.segment_id,
            self.kind.value,
            self.token_length,
            self.source_id,
            self.lineage_id,
            self.shareable,
            self.content_digest,
            self.token_start,
            self.token_end,
            self.logical_order,
        )


@dataclass(frozen=True, slots=True)
class KMeshRequestLookupDescriptor:
    """Parsed, immutable request admission evidence."""

    lookup_intent: KMeshRequestLookupIntent
    request_id: str
    admission_generation: int
    prompt_token_count: int
    cache_domain: str | None
    admission_fallback: KMeshFallbackReason | None
    spans: tuple[KMeshRequestLookupSpan, ...]

    def __post_init__(self) -> None:
        if type(self.lookup_intent) is not KMeshRequestLookupIntent:
            raise TypeError("descriptor.lookup_intent must be KMeshRequestLookupIntent")
        _identifier(self.request_id, "descriptor.request_id")
        if type(self.admission_generation) is not int:
            raise TypeError("descriptor.admission_generation must be an integer")
        if self.admission_generation <= 0:
            raise ValueError("descriptor.admission_generation must be positive")
        if type(self.prompt_token_count) is not int:
            raise TypeError("descriptor.prompt_token_count must be an integer")
        if self.prompt_token_count < 0:
            raise ValueError("descriptor.prompt_token_count must be nonnegative")
        _optional_identifier(self.cache_domain, "descriptor.cache_domain")
        if (
            self.admission_fallback is not None
            and type(self.admission_fallback) is not KMeshFallbackReason
        ):
            raise TypeError(
                "descriptor.admission_fallback must be KMeshFallbackReason or null"
            )
        if type(self.spans) is not tuple:
            raise TypeError("descriptor.spans must be an immutable tuple")

        if self.admission_fallback is not None:
            if self.spans:
                raise ValueError("a fallback descriptor cannot carry admitted spans")
            return
        if self.cache_domain is None:
            raise ValueError("an admitted descriptor requires a cache domain")
        if not self.spans:
            raise ValueError("an admitted descriptor requires spans")

        seen_segment_ids: set[str] = set()
        previous: KMeshRequestLookupSpan | None = None
        for index, span in enumerate(self.spans):
            if type(span) is not KMeshRequestLookupSpan:
                raise TypeError(
                    f"descriptor.spans[{index}] must be KMeshRequestLookupSpan"
                )
            if span.segment_id in seen_segment_ids:
                raise ValueError("descriptor span IDs must be unique")
            seen_segment_ids.add(span.segment_id)
            if span.token_end > self.prompt_token_count:
                raise ValueError("descriptor spans must fit within the prompt")
            if previous is not None:
                if span.token_start < previous.token_end:
                    raise ValueError(
                        "descriptor spans must be ordered and non-overlapping"
                    )
                if span.logical_order <= previous.logical_order:
                    raise ValueError(
                        "descriptor span logical order must strictly increase"
                    )
            previous = span

    @classmethod
    def from_wire_payload(cls, payload: object) -> KMeshRequestLookupDescriptor:
        fields = _wire_tuple(payload, "descriptor", length=8)
        (
            wire_tag,
            lookup_intent,
            request_id,
            admission_generation,
            prompt_token_count,
            cache_domain,
            admission_fallback,
            raw_spans,
        ) = fields
        wire_tag = _strict_string(wire_tag, "descriptor wire tag")
        if wire_tag != KMESH_REQUEST_LOOKUP_WIRE_TAG:
            raise ValueError("descriptor has an unsupported wire tag")
        intent = _enum_value(
            lookup_intent,
            KMeshRequestLookupIntent,
            "descriptor.lookup_intent",
        )
        if admission_fallback is None:
            fallback = None
        else:
            fallback = _enum_value(
                admission_fallback,
                KMeshFallbackReason,
                "descriptor.admission_fallback",
            )
        spans_payload = _wire_tuple(raw_spans, "descriptor.spans")
        return cls(
            lookup_intent=intent,  # type: ignore[arg-type]
            request_id=_identifier(request_id, "descriptor.request_id"),
            admission_generation=_strict_int(
                admission_generation, "descriptor.admission_generation"
            ),
            prompt_token_count=_strict_int(
                prompt_token_count, "descriptor.prompt_token_count"
            ),
            cache_domain=_optional_identifier(cache_domain, "descriptor.cache_domain"),
            admission_fallback=fallback,  # type: ignore[arg-type]
            spans=tuple(
                KMeshRequestLookupSpan.from_wire_payload(span) for span in spans_payload
            ),
        )

    @property
    def is_admitted(self) -> bool:
        return self.admission_fallback is None

    @property
    def permits_serial_pic(self) -> bool:
        return (
            self.is_admitted
            and self.lookup_intent is KMeshRequestLookupIntent.SERIAL_PIC
        )

    @property
    def fingerprint(self) -> str:
        """Fingerprint every exact descriptor field, including offsets."""
        return _versioned_sha256(
            b"kmesh-request-lookup-descriptor-v1", self.to_wire_payload()
        )

    def to_wire_payload(self) -> KMeshWireValue:
        return (
            KMESH_REQUEST_LOOKUP_WIRE_TAG,
            self.lookup_intent.value,
            self.request_id,
            self.admission_generation,
            self.prompt_token_count,
            self.cache_domain,
            (
                self.admission_fallback.value
                if self.admission_fallback is not None
                else None
            ),
            tuple(span.to_wire_payload() for span in self.spans),
        )

    def explicit_token_spans(self) -> ExplicitTokenSpans:
        """Return the exact scheduler-admitted ranges for serial PIC."""
        if not self.permits_serial_pic:
            raise ValueError("only admitted serial_pic descriptors are actionable")
        return ExplicitTokenSpans(
            token_count=self.prompt_token_count,
            cached_span=TokenSpan(0, self.prompt_token_count),
            spans=tuple(
                TokenSpan(span.token_start, span.token_end) for span in self.spans
            ),
        )


@dataclass(frozen=True, slots=True)
class KMeshRuntimeEvidence:
    """Complete, canonical local runtime evidence used for cache isolation."""

    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    model_dtype: str
    effective_kv_dtype: str
    kv_layout: str
    attention_kind: str
    rope_configuration: str
    attention_backend: str
    attention_backend_configuration: str
    runtime_configuration: str
    device_identity: str
    tensor_parallel_size: int
    pipeline_parallel_size: int

    def __post_init__(self) -> None:
        for field_name in (
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "model_dtype",
            "effective_kv_dtype",
            "kv_layout",
            "attention_kind",
            "rope_configuration",
            "attention_backend",
            "attention_backend_configuration",
            "runtime_configuration",
            "device_identity",
        ):
            _runtime_string(getattr(self, field_name), field_name)
        for field_name in ("tensor_parallel_size", "pipeline_parallel_size"):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")

    def to_canonical_payload(self) -> tuple[object, ...]:
        return (
            "kmesh-runtime-evidence-v1",
            self.model_id,
            self.model_revision,
            self.tokenizer_id,
            self.tokenizer_revision,
            self.model_dtype,
            self.effective_kv_dtype,
            self.kv_layout,
            self.attention_kind,
            self.rope_configuration,
            self.attention_backend,
            self.attention_backend_configuration,
            self.runtime_configuration,
            self.device_identity,
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
        )

    @property
    def fingerprint(self) -> str:
        return _versioned_sha256(
            b"kmesh-runtime-evidence-v1", self.to_canonical_payload()
        )


def derive_kmesh_cache_tags(
    cache_domain: object,
    runtime_evidence: KMeshRuntimeEvidence,
) -> tuple[tuple[str, str], ...]:
    """Derive stable tags without request, generation, or offset metadata."""
    cache_domain = _identifier(cache_domain, "cache_domain")
    if type(runtime_evidence) is not KMeshRuntimeEvidence:
        raise TypeError("runtime_evidence must be exactly KMeshRuntimeEvidence")
    tags = (
        (
            KMESH_DOMAIN_TAG_KEY,
            _versioned_sha256(b"kmesh-cache-domain-v1", cache_domain),
        ),
        (KMESH_RUNTIME_TAG_KEY, runtime_evidence.fingerprint),
    )
    return tuple(sorted(tags))


def _request_config_items(
    request_configs: Mapping[str, object] | None,
) -> tuple[tuple[str, KMeshRequestConfigValue], ...]:
    if request_configs is None:
        return ()
    if not isinstance(request_configs, Mapping):
        raise TypeError("request_configs must be a mapping or null")
    items: list[tuple[str, KMeshRequestConfigValue]] = []
    for raw_key, raw_value in request_configs.items():
        if type(raw_key) is not str or not raw_key:
            raise TypeError("request config keys must be nonempty plain strings")
        if raw_key.startswith("lmcache.tag."):
            if raw_key in KMESH_RESERVED_TAG_KEYS:
                raise ValueError(f"request config conflicts with reserved {raw_key}")
            raise ValueError("user-supplied lmcache.tag.* configs are not allowed")
        if raw_value is not None and type(raw_value) not in (
            str,
            int,
            float,
            bool,
        ):
            raise TypeError(
                f"request config {raw_key!r} must have a scalar primitive value"
            )
        if type(raw_value) is float and not math.isfinite(raw_value):
            raise ValueError(f"request config {raw_key!r} must be finite")
        items.append((raw_key, raw_value))  # type: ignore[arg-type]
    return tuple(sorted(items))


def build_kmesh_request_configs(
    request_configs: Mapping[str, object] | None,
    cache_domain: object,
    runtime_evidence: KMeshRuntimeEvidence,
) -> tuple[tuple[str, KMeshRequestConfigValue], ...]:
    """Merge validated user configs with exact, sorted KMesh cache tags."""
    cache_tags = derive_kmesh_cache_tags(cache_domain, runtime_evidence)
    merged = (*_request_config_items(request_configs), *cache_tags)
    return tuple(sorted(merged))


def parse_kmesh_request_sidecar(
    provider_id: object, payload: object
) -> KMeshRequestLookupDescriptor:
    """Parse one exact KMesh provider/payload pair."""
    provider_id = _strict_string(provider_id, "sidecar.provider_id")
    if provider_id != KMESH_REQUEST_SIDECAR_PROVIDER_ID:
        raise ValueError("sidecar has an unsupported provider_id")
    return KMeshRequestLookupDescriptor.from_wire_payload(payload)


def _request_evidence(request: object) -> tuple[str, tuple[int, ...], int, object]:
    if request is None:
        raise TypeError("request must be a live request object")
    try:
        request_id = request.request_id  # type: ignore[attr-defined]
        prompt_token_ids = request.prompt_token_ids  # type: ignore[attr-defined]
        prompt_token_count = request.num_prompt_tokens  # type: ignore[attr-defined]
        cache_salt = request.cache_salt  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError(
            "request must expose request_id, prompt_token_ids, "
            "num_prompt_tokens, and cache_salt"
        ) from exc
    return (
        _identifier(request_id, "request.request_id"),
        _token_tuple(prompt_token_ids, "request.prompt_token_ids"),
        _strict_int(prompt_token_count, "request.num_prompt_tokens"),
        cache_salt,
    )


def validate_kmesh_request_lookup_evidence(
    descriptor: KMeshRequestLookupDescriptor,
    request: object,
) -> tuple[str, tuple[int, ...], int]:
    """Validate request identity and prompt evidence for any descriptor outcome."""
    if type(descriptor) is not KMeshRequestLookupDescriptor:
        raise TypeError("descriptor must be exactly KMeshRequestLookupDescriptor")
    request_id, prompt_token_ids, prompt_token_count, cache_salt = _request_evidence(
        request
    )
    if cache_salt is not None:
        raise ValueError("cache_salt is unsupported by the v1 KMesh sidecar")
    if request_id != descriptor.request_id:
        raise ValueError("request ID drifted from the KMesh descriptor")
    if prompt_token_count != len(prompt_token_ids):
        raise ValueError("request prompt count does not match its token evidence")
    if prompt_token_count != descriptor.prompt_token_count:
        raise ValueError("request prompt count drifted from the KMesh descriptor")
    for index, span in enumerate(descriptor.spans):
        actual_digest = kmesh_token_slice_digest(
            prompt_token_ids[span.token_start : span.token_end]
        )
        if actual_digest != span.content_digest:
            raise ValueError(
                f"request token slice for descriptor.spans[{index}] drifted"
            )
    return request_id, prompt_token_ids, prompt_token_count


def _validate_descriptor_against_request(
    descriptor: KMeshRequestLookupDescriptor,
    request: object,
    admission_generation: object,
    cache_domain: object,
) -> tuple[str, tuple[int, ...], int, str]:
    if type(descriptor) is not KMeshRequestLookupDescriptor:
        raise TypeError("descriptor must be exactly KMeshRequestLookupDescriptor")
    if not descriptor.permits_serial_pic:
        raise ValueError("descriptor does not permit serial PIC")
    request_id, prompt_token_ids, prompt_token_count = (
        validate_kmesh_request_lookup_evidence(descriptor, request)
    )
    admission_generation = _strict_int(admission_generation, "admission_generation")
    cache_domain = _identifier(cache_domain, "cache_domain")
    if admission_generation != descriptor.admission_generation:
        raise ValueError("admission generation drifted from the KMesh descriptor")
    if cache_domain != descriptor.cache_domain:
        raise ValueError("cache domain drifted from the KMesh descriptor")
    return request_id, prompt_token_ids, prompt_token_count, cache_domain


def _validate_explicit_token_spans_type(
    explicit_spans: object,
) -> ExplicitTokenSpans:
    """Validate the concrete DTO graph before any dataclass equality check."""
    if type(explicit_spans) is not ExplicitTokenSpans:
        raise TypeError("explicit_spans must be exactly ExplicitTokenSpans")
    validated = cast(ExplicitTokenSpans, explicit_spans)
    if type(validated.token_count) is not int:
        raise TypeError("explicit_spans.token_count must be an integer")
    if type(validated.cached_span) is not TokenSpan:
        raise TypeError("explicit_spans.cached_span must be exactly TokenSpan")
    for field_name in ("start", "end"):
        if type(getattr(validated.cached_span, field_name)) is not int:
            raise TypeError(
                f"explicit_spans.cached_span.{field_name} must be an integer"
            )
    if type(validated.spans) is not tuple:
        raise TypeError("explicit_spans.spans must be an immutable tuple")
    for index, span in enumerate(validated.spans):
        if type(span) is not TokenSpan:
            raise TypeError(f"explicit_spans.spans[{index}] must be exactly TokenSpan")
        for field_name in ("start", "end"):
            if type(getattr(span, field_name)) is not int:
                raise TypeError(
                    f"explicit_spans.spans[{index}].{field_name} must be an integer"
                )
    return validated


def _validate_cache_tag_tuple(
    cache_tags: object,
) -> tuple[tuple[str, str], ...]:
    """Validate the concrete cache-tag representation before comparison."""
    if type(cache_tags) is not tuple:
        raise TypeError("cache_tags must be an immutable tuple")
    for index, item in enumerate(cache_tags):
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("cache_tags must contain two-item tuples")
        key, value = item
        if type(key) is not str or type(value) is not str:
            raise TypeError(
                f"cache_tags[{index}] must contain plain string keys and values"
            )
    return cache_tags


def _validate_request_config_tuple(
    request_configs: object,
) -> tuple[tuple[str, KMeshRequestConfigValue], ...]:
    """Validate a frozen config snapshot without loose equality aliases."""
    if type(request_configs) is not tuple:
        raise TypeError("request_configs must be an immutable tuple")
    keys: list[str] = []
    for item in request_configs:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("request_configs must contain two-item tuples")
        key, value = item
        if type(key) is not str or not key:
            raise TypeError("request config keys must be nonempty plain strings")
        if value is not None and type(value) not in (
            str,
            int,
            float,
            bool,
        ):
            raise TypeError("request config values must be scalar primitives")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("request config values must be finite")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError("request_configs must not contain duplicate keys")
    if keys != sorted(keys):
        raise ValueError("request_configs must be sorted")
    if any(
        key.startswith("lmcache.tag.") and key not in KMESH_RESERVED_TAG_KEYS
        for key in keys
    ):
        raise ValueError("request_configs contain an untrusted cache tag")
    return request_configs


@dataclass(frozen=True, slots=True)
class KMeshRequestLookupBinding:
    """Exact live-request binding for one actionable sidecar descriptor."""

    descriptor: KMeshRequestLookupDescriptor
    request_id: str
    request_object: object = field(repr=False, compare=False, hash=False)
    prompt_token_ids: tuple[int, ...]
    prompt_token_count: int
    admission_generation: int
    cache_domain: str
    descriptor_fingerprint: str
    explicit_spans: ExplicitTokenSpans
    runtime_evidence: KMeshRuntimeEvidence
    cache_tags: tuple[tuple[str, str], ...]
    request_configs: tuple[tuple[str, KMeshRequestConfigValue], ...]

    def __post_init__(self) -> None:
        if type(self.descriptor) is not KMeshRequestLookupDescriptor:
            raise TypeError("descriptor must be exactly KMeshRequestLookupDescriptor")
        request_id = _identifier(self.request_id, "binding.request_id")
        if type(self.prompt_token_ids) is not tuple:
            raise TypeError("binding.prompt_token_ids must be an immutable tuple")
        prompt_token_ids = _token_tuple(
            self.prompt_token_ids, "binding.prompt_token_ids"
        )
        prompt_token_count = _strict_int(
            self.prompt_token_count, "binding.prompt_token_count"
        )
        if prompt_token_count < 0:
            raise ValueError("binding.prompt_token_count must be nonnegative")
        admission_generation = _strict_int(
            self.admission_generation, "binding.admission_generation"
        )
        if admission_generation <= 0:
            raise ValueError("binding.admission_generation must be positive")
        cache_domain = _identifier(self.cache_domain, "binding.cache_domain")
        descriptor_fingerprint = _strict_string(
            self.descriptor_fingerprint, "binding.descriptor_fingerprint"
        )
        explicit_spans = _validate_explicit_token_spans_type(self.explicit_spans)
        if type(self.runtime_evidence) is not KMeshRuntimeEvidence:
            raise TypeError("runtime_evidence must be exactly KMeshRuntimeEvidence")
        cache_tags = _validate_cache_tag_tuple(self.cache_tags)
        request_configs = _validate_request_config_tuple(self.request_configs)

        evidence = _validate_descriptor_against_request(
            self.descriptor,
            self.request_object,
            admission_generation,
            cache_domain,
        )
        if evidence != (
            request_id,
            prompt_token_ids,
            prompt_token_count,
            cache_domain,
        ):
            raise ValueError("binding fields do not match live request evidence")
        if descriptor_fingerprint != self.descriptor.fingerprint:
            raise ValueError("binding has an invalid descriptor fingerprint")
        if explicit_spans != self.descriptor.explicit_token_spans():
            raise ValueError("binding has invalid explicit token spans")
        expected_tags = derive_kmesh_cache_tags(cache_domain, self.runtime_evidence)
        if cache_tags != expected_tags:
            raise ValueError("binding has invalid cache tags")
        request_config_map = dict(request_configs)
        for key, value in expected_tags:
            if request_config_map.get(key) != value:
                raise ValueError("request_configs must contain exact KMesh tags")

    def as_request_configs(self) -> dict[str, KMeshRequestConfigValue]:
        return dict(self.request_configs)

    def validate_bound_request(
        self,
        request: object,
        *,
        admission_generation: object,
        cache_domain: object,
    ) -> None:
        """Reject object replacement or any prompt/admission evidence drift."""
        if request is not self.request_object:
            raise ValueError("request object identity drifted from its KMesh binding")
        evidence = _validate_descriptor_against_request(
            self.descriptor,
            request,
            admission_generation,
            cache_domain,
        )
        if evidence != (
            self.request_id,
            self.prompt_token_ids,
            self.prompt_token_count,
            self.cache_domain,
        ):
            raise ValueError("request evidence drifted from its KMesh binding")
        if self.descriptor.fingerprint != self.descriptor_fingerprint:
            raise ValueError("descriptor fingerprint drifted from its KMesh binding")
        if derive_kmesh_cache_tags(self.cache_domain, self.runtime_evidence) != (
            self.cache_tags
        ):
            raise ValueError("runtime cache tags drifted from their KMesh binding")


@dataclass(frozen=True, slots=True)
class KMeshRequestExecutionEvidence:
    """Pickle-safe worker evidence with no scheduler request reference.

    The scheduler-side binding deliberately retains the live request object so
    repeated lookup probes can detect identity drift.  That object must never
    cross the scheduler/worker boundary.  This DTO contains the independently
    validated immutable evidence needed by the worker to compare its KMesh
    peer metadata and derive the exact LMCache key namespace.
    """

    descriptor: KMeshRequestLookupDescriptor
    prompt_token_ids: tuple[int, ...]
    descriptor_fingerprint: str
    explicit_spans: ExplicitTokenSpans
    runtime_evidence: KMeshRuntimeEvidence
    cache_tags: tuple[tuple[str, str], ...]
    request_configs: tuple[tuple[str, KMeshRequestConfigValue], ...]

    def __post_init__(self) -> None:
        if type(self.descriptor) is not KMeshRequestLookupDescriptor:
            raise TypeError("descriptor must be exactly KMeshRequestLookupDescriptor")
        if not self.descriptor.permits_serial_pic:
            raise ValueError("execution evidence requires an actionable descriptor")
        if type(self.prompt_token_ids) is not tuple:
            raise TypeError("prompt_token_ids must be an immutable tuple")
        prompt_token_ids = _token_tuple(
            self.prompt_token_ids, "execution.prompt_token_ids"
        )
        if len(prompt_token_ids) != self.descriptor.prompt_token_count:
            raise ValueError("execution prompt tokens do not match descriptor length")
        for index, span in enumerate(self.descriptor.spans):
            digest = kmesh_token_slice_digest(
                prompt_token_ids[span.token_start : span.token_end]
            )
            if digest != span.content_digest:
                raise ValueError(
                    f"execution token slice for descriptor.spans[{index}] drifted"
                )

        descriptor_fingerprint = _strict_string(
            self.descriptor_fingerprint,
            "execution.descriptor_fingerprint",
        )
        if descriptor_fingerprint != self.descriptor.fingerprint:
            raise ValueError("execution evidence has an invalid descriptor fingerprint")

        explicit_spans = _validate_explicit_token_spans_type(self.explicit_spans)
        if explicit_spans != self.descriptor.explicit_token_spans():
            raise ValueError("execution evidence has invalid explicit token spans")
        if type(self.runtime_evidence) is not KMeshRuntimeEvidence:
            raise TypeError("runtime_evidence must be exactly KMeshRuntimeEvidence")

        cache_tags = _validate_cache_tag_tuple(self.cache_tags)
        expected_tags = derive_kmesh_cache_tags(
            self.descriptor.cache_domain,
            self.runtime_evidence,
        )
        if cache_tags != expected_tags:
            raise ValueError("execution evidence has invalid cache tags")

        request_configs = _validate_request_config_tuple(self.request_configs)
        request_config_map = dict(request_configs)
        for key, value in expected_tags:
            if request_config_map.get(key) != value:
                raise ValueError(
                    "execution request configs must contain exact KMesh tags"
                )

    @classmethod
    def from_binding(
        cls, binding: KMeshRequestLookupBinding
    ) -> KMeshRequestExecutionEvidence:
        """Freeze one still-valid scheduler binding for worker transport."""
        if type(binding) is not KMeshRequestLookupBinding:
            raise TypeError("binding must be exactly KMeshRequestLookupBinding")
        binding.validate_bound_request(
            binding.request_object,
            admission_generation=binding.admission_generation,
            cache_domain=binding.cache_domain,
        )
        return cls(
            descriptor=binding.descriptor,
            prompt_token_ids=binding.prompt_token_ids,
            descriptor_fingerprint=binding.descriptor_fingerprint,
            explicit_spans=binding.explicit_spans,
            runtime_evidence=binding.runtime_evidence,
            cache_tags=binding.cache_tags,
            request_configs=binding.request_configs,
        )

    @property
    def request_id(self) -> str:
        """Return the descriptor-owned request identifier."""
        return self.descriptor.request_id

    def as_request_configs(self) -> dict[str, KMeshRequestConfigValue]:
        """Return a fresh mutable mapping for the LMCache request API."""
        return dict(self.request_configs)

    def validate_peer_descriptor(
        self, descriptor: KMeshRequestLookupDescriptor
    ) -> None:
        """Require the worker's authoritative peer descriptor to match exactly."""
        if type(descriptor) is not KMeshRequestLookupDescriptor:
            raise TypeError(
                "peer descriptor must be exactly KMeshRequestLookupDescriptor"
            )
        if descriptor.to_wire_payload() != self.descriptor.to_wire_payload():
            raise ValueError("worker KMesh descriptor differs from scheduler evidence")
        if descriptor.fingerprint != self.descriptor_fingerprint:
            raise ValueError("worker KMesh descriptor fingerprint differs")

    def validate_runtime_evidence(self, runtime_evidence: KMeshRuntimeEvidence) -> None:
        """Require a freshly derived worker runtime to equal the scheduler one."""
        if type(runtime_evidence) is not KMeshRuntimeEvidence:
            raise TypeError("runtime_evidence must be exactly KMeshRuntimeEvidence")
        if (
            runtime_evidence.to_canonical_payload()
            != self.runtime_evidence.to_canonical_payload()
        ):
            raise ValueError("worker runtime evidence differs from scheduler evidence")
        if (
            derive_kmesh_cache_tags(
                self.descriptor.cache_domain,
                runtime_evidence,
            )
            != self.cache_tags
        ):
            raise ValueError("worker runtime derives different KMesh cache tags")


def bind_kmesh_request_lookup(
    descriptor: KMeshRequestLookupDescriptor,
    request: object,
    *,
    admission_generation: object,
    cache_domain: object,
    runtime_evidence: KMeshRuntimeEvidence,
    request_configs: Mapping[str, object] | None = None,
) -> KMeshRequestLookupBinding:
    """Bind an actionable descriptor to one exact live vLLM request object."""
    if type(runtime_evidence) is not KMeshRuntimeEvidence:
        raise TypeError("runtime_evidence must be exactly KMeshRuntimeEvidence")
    request_id, prompt_token_ids, prompt_token_count, cache_domain = (
        _validate_descriptor_against_request(
            descriptor,
            request,
            admission_generation,
            cache_domain,
        )
    )
    cache_tags = derive_kmesh_cache_tags(cache_domain, runtime_evidence)
    merged_configs = build_kmesh_request_configs(
        request_configs, cache_domain, runtime_evidence
    )
    return KMeshRequestLookupBinding(
        descriptor=descriptor,
        request_id=request_id,
        request_object=request,
        prompt_token_ids=prompt_token_ids,
        prompt_token_count=prompt_token_count,
        admission_generation=descriptor.admission_generation,
        cache_domain=cache_domain,
        descriptor_fingerprint=descriptor.fingerprint,
        explicit_spans=descriptor.explicit_token_spans(),
        runtime_evidence=runtime_evidence,
        cache_tags=cache_tags,
        request_configs=merged_configs,
    )
