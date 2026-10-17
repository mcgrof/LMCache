# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import FrozenInstanceError, dataclass, replace
from typing import Any, Callable, cast
import hashlib
import json
import pickle

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.kmesh import (
    KMESH_DOMAIN_TAG_KEY,
    KMESH_REQUEST_LOOKUP_WIRE_TAG,
    KMESH_REQUEST_SIDECAR_PROVIDER_ID,
    KMESH_RUNTIME_TAG_KEY,
    KMeshFallbackReason,
    KMeshRequestExecutionEvidence,
    KMeshRequestLookupBinding,
    KMeshRequestLookupDescriptor,
    KMeshRequestLookupIntent,
    KMeshRequestLookupSpan,
    KMeshRuntimeEvidence,
    KMeshSegmentKind,
    bind_kmesh_request_lookup,
    build_kmesh_request_configs,
    derive_kmesh_cache_tags,
    kmesh_token_slice_digest,
    parse_kmesh_request_sidecar,
)
from lmcache.v1.token_span import ExplicitTokenSpans, TokenSpan


class _IntSubclass(int):
    pass


class _StringSubclass(str):
    pass


class _TupleSubclass(tuple):
    pass


class _ListSubclass(list):
    pass


class _TokenSpanSubclass(TokenSpan):
    pass


class _ExplicitTokenSpansSubclass(ExplicitTokenSpans):
    pass


class _ObserveOnlyDescriptorSubclass(KMeshRequestLookupDescriptor):
    @property
    def permits_serial_pic(self) -> bool:
        return True


class _RuntimeEvidenceFingerprintSubclass(KMeshRuntimeEvidence):
    @property
    def fingerprint(self) -> str:
        return _runtime().fingerprint


@dataclass
class _Request:
    request_id: object
    prompt_token_ids: object
    num_prompt_tokens: object
    cache_salt: object = None


TOKENS = (10, 11, 12, 20, 21, 30, 31, 32)


def _independent_digest(token_ids: tuple[int, ...]) -> str:
    payload = json.dumps(
        list(token_ids), separators=(",", ":"), ensure_ascii=True
    ).encode()
    digest = hashlib.sha256(b"kmesh-token-slice-v1\0" + payload).hexdigest()
    return f"sha256-v1:{digest}"


def _span(
    *,
    segment_id: object = "shared-a",
    kind: object = "task",
    token_length: object = 3,
    source_id: object = "source-a",
    lineage_id: object = "lineage-a",
    shareable: object = True,
    content_digest: object | None = None,
    token_start: object = 0,
    token_end: object = 3,
    logical_order: object = 0,
    tokens: tuple[int, ...] = TOKENS,
) -> tuple[object, ...]:
    if content_digest is None and type(token_start) is int and type(token_end) is int:
        content_digest = _independent_digest(tokens[token_start:token_end])
    return (
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
    )


def _payload(
    *,
    wire_tag: object = KMESH_REQUEST_LOOKUP_WIRE_TAG,
    lookup_intent: object = "serial_pic",
    request_id: object = "request-a",
    admission_generation: object = 7,
    prompt_token_count: object = len(TOKENS),
    cache_domain: object = "tenant-a",
    admission_fallback: object = None,
    spans: object | None = None,
) -> tuple[object, ...]:
    if spans is None:
        spans = (
            _span(),
            _span(
                segment_id="shared-b",
                kind="shared_update",
                source_id=None,
                lineage_id="lineage-b",
                token_start=5,
                token_end=8,
                logical_order=2,
            ),
        )
    return (
        wire_tag,
        lookup_intent,
        request_id,
        admission_generation,
        prompt_token_count,
        cache_domain,
        admission_fallback,
        spans,
    )


def _runtime(**changes: object) -> KMeshRuntimeEvidence:
    values: dict[str, object] = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "7ae55760",
        "tokenizer_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "tokenizer_revision": "7ae55760",
        "model_dtype": "bfloat16",
        "effective_kv_dtype": "bfloat16",
        "kv_layout": "NHD",
        "attention_kind": "full",
        "rope_configuration": '{"rope_theta":1000000.0,"rope_type":"default"}',
        "attention_backend": "ROCM_ATTN",
        "attention_backend_configuration": "rocm-attn-v1:no-mla:no-sliding",
        "runtime_configuration": "serial-pic-v1:bf16:eager:max-len-2048",
        "device_identity": "rocm-7.2.3:gfx1100",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    values.update(changes)
    return KMeshRuntimeEvidence(**cast(Any, values))


def _request(
    *,
    request_id: object = "request-a",
    tokens: list[int] | tuple[int, ...] = TOKENS,
    count: object = len(TOKENS),
    cache_salt: object = None,
) -> _Request:
    return _Request(request_id, list(tokens), count, cache_salt)


def _parse(payload: object | None = None) -> KMeshRequestLookupDescriptor:
    return parse_kmesh_request_sidecar(
        KMESH_REQUEST_SIDECAR_PROVIDER_ID,
        _payload() if payload is None else payload,
    )


def _replace_item(value: tuple[object, ...], index: int, item: object) -> tuple:
    return (*value[:index], item, *value[index + 1 :])


def _replace_span(payload: tuple, span_index: int, span: object) -> tuple:
    spans = payload[7]
    assert type(spans) is tuple
    return _replace_item(payload, 7, _replace_item(spans, span_index, span))


def test_parse_exact_wire_into_frozen_local_dtos_and_explicit_spans() -> None:
    descriptor = _parse()

    assert descriptor.lookup_intent is KMeshRequestLookupIntent.SERIAL_PIC
    assert descriptor.request_id == "request-a"
    assert descriptor.admission_generation == 7
    assert descriptor.prompt_token_count == 8
    assert descriptor.cache_domain == "tenant-a"
    assert descriptor.admission_fallback is None
    assert descriptor.permits_serial_pic
    assert descriptor.to_wire_payload() == _payload()
    assert descriptor.spans == (
        KMeshRequestLookupSpan(
            segment_id="shared-a",
            kind=KMeshSegmentKind.TASK,
            token_length=3,
            source_id="source-a",
            lineage_id="lineage-a",
            shareable=True,
            content_digest=_independent_digest(TOKENS[0:3]),
            token_start=0,
            token_end=3,
            logical_order=0,
        ),
        KMeshRequestLookupSpan(
            segment_id="shared-b",
            kind=KMeshSegmentKind.SHARED_UPDATE,
            token_length=3,
            source_id=None,
            lineage_id="lineage-b",
            shareable=True,
            content_digest=_independent_digest(TOKENS[5:8]),
            token_start=5,
            token_end=8,
            logical_order=2,
        ),
    )
    assert descriptor.explicit_token_spans() == ExplicitTokenSpans(
        token_count=8,
        cached_span=TokenSpan(0, 8),
        spans=(TokenSpan(0, 3), TokenSpan(5, 8)),
    )

    with pytest.raises(FrozenInstanceError):
        descriptor.request_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        descriptor.spans[0].token_end = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("provider", "payload"),
    [
        ("unknown", _payload()),
        (True, _payload()),
        (_StringSubclass(KMESH_REQUEST_SIDECAR_PROVIDER_ID), _payload()),
        (KMESH_REQUEST_SIDECAR_PROVIDER_ID, list(_payload())),
        (KMESH_REQUEST_SIDECAR_PROVIDER_ID, _payload()[:-1]),
        (KMESH_REQUEST_SIDECAR_PROVIDER_ID, (*_payload(), None)),
    ],
)
def test_provider_and_descriptor_shape_are_exact(provider, payload) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_kmesh_request_sidecar(provider, payload)


@pytest.mark.parametrize(
    ("index", "invalid"),
    [
        (0, "future-wire"),
        (0, True),
        (0, _StringSubclass(KMESH_REQUEST_LOOKUP_WIRE_TAG)),
        (1, "future_intent"),
        (1, True),
        (1, _StringSubclass("serial_pic")),
        (2, ""),
        (2, " request-a"),
        (2, "request-a\n"),
        (2, "x" * 257),
        (2, True),
        (2, _StringSubclass("request-a")),
        (3, True),
        (3, 0),
        (3, -1),
        (3, _IntSubclass(7)),
        (3, "7"),
        (4, True),
        (4, -1),
        (4, _IntSubclass(len(TOKENS))),
        (4, "8"),
        (5, ""),
        (5, " tenant-a"),
        (5, True),
        (5, _StringSubclass("tenant-a")),
        (6, "future_fallback"),
        (6, True),
        (6, _StringSubclass("disabled")),
        (7, list(cast(tuple[object, ...], _payload()[7]))),
    ],
)
def test_descriptor_fields_reject_loose_primitive_and_value_mutations(
    index: int, invalid: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _parse(_replace_item(_payload(), index, invalid))


def test_every_v1_fallback_enum_is_accepted_only_on_empty_fallback_wire() -> None:
    for fallback in KMeshFallbackReason:
        descriptor = _parse(
            _payload(
                lookup_intent="observe_only",
                cache_domain=None,
                admission_fallback=fallback.value,
                spans=(),
            )
        )
        assert descriptor.admission_fallback is fallback
        assert not descriptor.is_admitted
        assert not descriptor.permits_serial_pic
        with pytest.raises(ValueError, match="actionable"):
            descriptor.explicit_token_spans()

    with pytest.raises(ValueError, match="cannot carry"):
        _parse(_payload(admission_fallback="disabled"))


@pytest.mark.parametrize("kind", [kind.value for kind in KMeshSegmentKind])
def test_every_v1_segment_kind_round_trips(kind: str) -> None:
    payload = _payload(spans=(_span(kind=kind),))
    descriptor = _parse(payload)
    assert descriptor.spans[0].kind.value == kind
    assert descriptor.to_wire_payload() == payload


@pytest.mark.parametrize(
    ("index", "invalid"),
    [
        (0, ""),
        (0, " shared-a"),
        (0, "shared-a\x7f"),
        (0, "x" * 257),
        (0, True),
        (0, _StringSubclass("shared-a")),
        (1, "future_kind"),
        (1, True),
        (1, _StringSubclass("task")),
        (2, True),
        (2, 0),
        (2, -1),
        (2, _IntSubclass(3)),
        (3, ""),
        (3, " source-a"),
        (3, True),
        (3, _StringSubclass("source-a")),
        (4, ""),
        (4, "lineage-a\n"),
        (4, True),
        (4, _StringSubclass("lineage-a")),
        (5, False),
        (5, 1),
        (5, "true"),
        (6, True),
        (6, ""),
        (6, "sha256:" + "0" * 64),
        (6, "sha256-v1:" + "0" * 63),
        (6, "sha256-v1:" + "0" * 65),
        (6, "sha256-v1:" + "A" * 64),
        (6, "sha256-v1:" + "g" * 64),
        (6, _StringSubclass("sha256-v1:" + "0" * 64)),
        (7, True),
        (7, -1),
        (7, _IntSubclass(0)),
        (8, True),
        (8, -1),
        (8, _IntSubclass(3)),
        (9, True),
        (9, -1),
        (9, _IntSubclass(0)),
    ],
)
def test_span_fields_reject_loose_primitive_and_value_mutations(
    index: int, invalid: object
) -> None:
    payload = _payload()
    span = cast(tuple[object, ...], payload[7])[0]
    assert type(span) is tuple
    mutated = _replace_span(payload, 0, _replace_item(span, index, invalid))
    with pytest.raises((TypeError, ValueError)):
        _parse(mutated)


@pytest.mark.parametrize(
    "span",
    [
        list(_span()),
        _span()[:-1],
        (*_span(), None),
        _span(token_start=3, token_end=3, token_length=0),
        _span(token_start=4, token_end=3, token_length=cast(Any, -1)),
        _span(token_start=0, token_end=3, token_length=2),
        _span(token_start=6, token_end=9, tokens=(*TOKENS, 33)),
    ],
)
def test_span_shape_ranges_and_lengths_are_exact(span: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _parse(_payload(spans=(span,)))


def test_descriptor_rejects_empty_duplicate_unordered_overlapping_spans() -> None:
    cases = (
        (),
        (_span(), _span()),
        (
            _span(token_start=5, token_end=8, logical_order=0),
            _span(
                segment_id="shared-b",
                token_start=0,
                token_end=3,
                logical_order=1,
            ),
        ),
        (
            _span(token_start=0, token_end=3, logical_order=0),
            _span(
                segment_id="shared-b",
                token_start=2,
                token_end=5,
                logical_order=1,
            ),
        ),
        (
            _span(token_start=0, token_end=3, logical_order=1),
            _span(
                segment_id="shared-b",
                token_start=5,
                token_end=8,
                logical_order=1,
            ),
        ),
        (
            _span(token_start=0, token_end=3, logical_order=2),
            _span(
                segment_id="shared-b",
                token_start=5,
                token_end=8,
                logical_order=1,
            ),
        ),
    )
    for spans in cases:
        with pytest.raises(ValueError):
            _parse(_payload(spans=spans))


def test_descriptor_requires_domain_for_admission_and_spans_fit_prompt() -> None:
    with pytest.raises(ValueError, match="cache domain"):
        _parse(_payload(cache_domain=None))
    with pytest.raises(ValueError, match="within the prompt"):
        _parse(_payload(prompt_token_count=7))


def test_token_slice_digest_is_exact_and_rejects_loose_tokens() -> None:
    assert kmesh_token_slice_digest((10, 11, 12)) == _independent_digest((10, 11, 12))
    for tokens in (
        [10, True],
        [10, -1],
        [10, _IntSubclass(11)],
        cast(Any, {10, 11}),
        cast(Any, "10"),
    ):
        with pytest.raises((TypeError, ValueError)):
            kmesh_token_slice_digest(tokens)  # type: ignore[arg-type]


def test_binding_freezes_exact_request_evidence_and_cache_inputs() -> None:
    descriptor = _parse()
    request = _request()
    runtime = _runtime()
    binding = bind_kmesh_request_lookup(
        descriptor,
        request,
        admission_generation=7,
        cache_domain="tenant-a",
        runtime_evidence=runtime,
        request_configs={"lmcache.skip_save": False, "lmcache.foo": "bar"},
    )

    assert binding.request_id == request.request_id
    assert binding.request_object is request
    assert binding.prompt_token_ids == TOKENS
    assert type(binding.prompt_token_ids) is tuple
    assert binding.prompt_token_count == len(TOKENS)
    assert binding.admission_generation == 7
    assert binding.cache_domain == "tenant-a"
    assert binding.descriptor_fingerprint == descriptor.fingerprint
    assert binding.explicit_spans == descriptor.explicit_token_spans()
    assert binding.runtime_evidence is runtime
    assert binding.request_configs == tuple(sorted(binding.request_configs))
    assert binding.as_request_configs() == {
        KMESH_DOMAIN_TAG_KEY: binding.cache_tags[0][1],
        KMESH_RUNTIME_TAG_KEY: binding.cache_tags[1][1],
        "lmcache.foo": "bar",
        "lmcache.skip_save": False,
    }
    binding.validate_bound_request(
        request, admission_generation=7, cache_domain="tenant-a"
    )

    with pytest.raises(FrozenInstanceError):
        binding.cache_domain = "tenant-b"  # type: ignore[misc]


@pytest.mark.parametrize(
    "change",
    [
        lambda request: setattr(request, "request_id", "request-b"),
        lambda request: setattr(request, "prompt_token_ids", list(TOKENS[:-1])),
        lambda request: setattr(request, "prompt_token_ids", [99, *TOKENS[1:]]),
        lambda request: setattr(request, "num_prompt_tokens", len(TOKENS) - 1),
        lambda request: setattr(request, "cache_salt", "salt-a"),
    ],
)
def test_bound_request_validation_rejects_live_object_drift(
    change: Callable[[_Request], None],
) -> None:
    request = _request()
    binding = bind_kmesh_request_lookup(
        _parse(),
        request,
        admission_generation=7,
        cache_domain="tenant-a",
        runtime_evidence=_runtime(),
    )
    change(request)
    with pytest.raises(ValueError):
        binding.validate_bound_request(
            request, admission_generation=7, cache_domain="tenant-a"
        )


def test_bound_request_validation_rejects_object_generation_and_domain_drift() -> None:
    request = _request()
    binding = bind_kmesh_request_lookup(
        _parse(),
        request,
        admission_generation=7,
        cache_domain="tenant-a",
        runtime_evidence=_runtime(),
    )
    replacement = _request()
    with pytest.raises(ValueError, match="object identity"):
        binding.validate_bound_request(
            replacement, admission_generation=7, cache_domain="tenant-a"
        )
    with pytest.raises(ValueError, match="generation"):
        binding.validate_bound_request(
            request, admission_generation=8, cache_domain="tenant-a"
        )
    with pytest.raises(ValueError, match="domain"):
        binding.validate_bound_request(
            request, admission_generation=7, cache_domain="tenant-b"
        )


@pytest.mark.parametrize(
    ("bound_request", "generation", "domain", "message"),
    [
        (_request(request_id="request-b"), 7, "tenant-a", "request ID"),
        (_request(tokens=TOKENS[:-1], count=7), 7, "tenant-a", "prompt count"),
        (_request(count=7), 7, "tenant-a", "prompt count"),
        (_request(cache_salt="salt-a"), 7, "tenant-a", "cache_salt"),
        (_request(), 8, "tenant-a", "generation"),
        (_request(), True, "tenant-a", "integer"),
        (_request(), 7, "tenant-b", "domain"),
    ],
)
def test_binding_rejects_request_and_admission_drift(
    bound_request: _Request, generation: object, domain: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        bind_kmesh_request_lookup(
            _parse(),
            bound_request,
            admission_generation=generation,
            cache_domain=domain,
            runtime_evidence=_runtime(),
        )


def test_binding_recomputes_every_admitted_token_slice_digest() -> None:
    descriptor = _parse()
    for token_index in (0, 1, 2, 5, 6, 7):
        tokens = list(TOKENS)
        tokens[token_index] += 1000
        with pytest.raises(ValueError, match="token slice"):
            bind_kmesh_request_lookup(
                descriptor,
                _request(tokens=tokens),
                admission_generation=7,
                cache_domain="tenant-a",
                runtime_evidence=_runtime(),
            )


def test_observer_and_fallback_descriptors_are_never_bound() -> None:
    observe = _parse(_payload(lookup_intent="observe_only"))
    fallback = _parse(
        _payload(
            lookup_intent="serial_pic",
            cache_domain=None,
            admission_fallback="invalid_metadata",
            spans=(),
        )
    )
    for descriptor in (observe, fallback):
        with pytest.raises(ValueError, match="does not permit"):
            bind_kmesh_request_lookup(
                descriptor,
                _request(),
                admission_generation=7,
                cache_domain="tenant-a",
                runtime_evidence=_runtime(),
            )


def test_observer_descriptor_subclass_cannot_override_actionability() -> None:
    observe = _parse(_payload(lookup_intent="observe_only"))
    spoof = _ObserveOnlyDescriptorSubclass(
        lookup_intent=observe.lookup_intent,
        request_id=observe.request_id,
        admission_generation=observe.admission_generation,
        prompt_token_count=observe.prompt_token_count,
        cache_domain=observe.cache_domain,
        admission_fallback=observe.admission_fallback,
        spans=observe.spans,
    )
    assert spoof.permits_serial_pic

    with pytest.raises(TypeError, match="exactly KMeshRequestLookupDescriptor"):
        bind_kmesh_request_lookup(
            spoof,
            _request(),
            admission_generation=7,
            cache_domain="tenant-a",
            runtime_evidence=_runtime(),
        )


def test_descriptor_fingerprint_is_independently_canonical() -> None:
    descriptor = _parse()
    canonical = json.dumps(
        descriptor.to_wire_payload(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    expected = hashlib.sha256(
        b"kmesh-request-lookup-descriptor-v1\0" + canonical
    ).hexdigest()
    assert descriptor.fingerprint == f"sha256-v1:{expected}"
    assert _parse(descriptor.to_wire_payload()).fingerprint == descriptor.fingerprint


def test_descriptor_fingerprint_binds_every_wire_field() -> None:
    payload = _payload()
    first_span = cast(tuple[tuple[object, ...], ...], payload[7])[0]
    valid_other_digest = "sha256-v1:" + "0" * 64
    variants = (
        _replace_item(payload, 1, "observe_only"),
        _replace_item(payload, 2, "request-b"),
        _replace_item(payload, 3, 8),
        _replace_item(payload, 4, 9),
        _replace_item(payload, 5, "tenant-b"),
        _replace_span(payload, 0, _replace_item(first_span, 0, "shared-c")),
        _replace_span(payload, 0, _replace_item(first_span, 1, "system")),
        _replace_span(payload, 0, _replace_item(first_span, 3, "source-c")),
        _replace_span(payload, 0, _replace_item(first_span, 4, "lineage-c")),
        _replace_span(payload, 0, _replace_item(first_span, 6, valid_other_digest)),
        _replace_span(
            payload,
            0,
            _span(
                token_start=1,
                token_end=4,
                logical_order=0,
            ),
        ),
        _replace_span(payload, 0, _replace_item(first_span, 9, 1)),
    )
    fingerprints = {_parse(payload).fingerprint}
    fingerprints.update(_parse(variant).fingerprint for variant in variants)
    assert len(fingerprints) == len(variants) + 1


def test_runtime_evidence_is_frozen_complete_and_canonical() -> None:
    runtime = _runtime()
    canonical = json.dumps(
        runtime.to_canonical_payload(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    expected = hashlib.sha256(b"kmesh-runtime-evidence-v1\0" + canonical).hexdigest()
    assert runtime.fingerprint == f"sha256-v1:{expected}"
    with pytest.raises(FrozenInstanceError):
        runtime.model_revision = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    [
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
    ],
)
@pytest.mark.parametrize("invalid", [None, "", " value", "value\n", True])
def test_runtime_evidence_rejects_every_missing_or_loose_string_field(
    field_name: str, invalid: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _runtime(**{field_name: invalid})


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("tensor_parallel_size", None),
        ("tensor_parallel_size", True),
        ("tensor_parallel_size", 0),
        ("tensor_parallel_size", -1),
        ("tensor_parallel_size", _IntSubclass(1)),
        ("pipeline_parallel_size", None),
        ("pipeline_parallel_size", True),
        ("pipeline_parallel_size", 0),
        ("pipeline_parallel_size", -1),
        ("pipeline_parallel_size", _IntSubclass(1)),
    ],
)
def test_runtime_evidence_rejects_missing_or_loose_parallelism(
    field_name: str, invalid: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _runtime(**{field_name: invalid})


def test_cache_tags_are_sorted_versioned_and_independently_canonical() -> None:
    runtime = _runtime()
    tags = derive_kmesh_cache_tags("tenant-a", runtime)
    domain_payload = json.dumps(
        "tenant-a", ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    expected_domain = hashlib.sha256(
        b"kmesh-cache-domain-v1\0" + domain_payload
    ).hexdigest()

    assert tags == tuple(sorted(tags))
    assert tags == (
        (KMESH_DOMAIN_TAG_KEY, f"sha256-v1:{expected_domain}"),
        (KMESH_RUNTIME_TAG_KEY, runtime.fingerprint),
    )
    assert "tenant-a" not in repr(tags)
    assert all(value.startswith("sha256-v1:") for _, value in tags)
    assert all(len(value) == len("sha256-v1:") + 64 for _, value in tags)


def test_runtime_evidence_subclass_cannot_override_cache_isolation() -> None:
    baseline = _runtime()
    changed = _RuntimeEvidenceFingerprintSubclass(
        model_id=baseline.model_id,
        model_revision=baseline.model_revision + "-changed",
        tokenizer_id=baseline.tokenizer_id,
        tokenizer_revision=baseline.tokenizer_revision,
        model_dtype=baseline.model_dtype,
        effective_kv_dtype=baseline.effective_kv_dtype,
        kv_layout=baseline.kv_layout,
        attention_kind=baseline.attention_kind,
        rope_configuration=baseline.rope_configuration,
        attention_backend=baseline.attention_backend,
        attention_backend_configuration=baseline.attention_backend_configuration,
        runtime_configuration=baseline.runtime_configuration,
        device_identity=baseline.device_identity,
        tensor_parallel_size=baseline.tensor_parallel_size,
        pipeline_parallel_size=baseline.pipeline_parallel_size,
    )
    assert changed.model_revision != baseline.model_revision
    assert changed.fingerprint == baseline.fingerprint

    with pytest.raises(TypeError, match="exactly KMeshRuntimeEvidence"):
        derive_kmesh_cache_tags("tenant-a", changed)
    with pytest.raises(TypeError, match="exactly KMeshRuntimeEvidence"):
        bind_kmesh_request_lookup(
            _parse(),
            _request(),
            admission_generation=7,
            cache_domain="tenant-a",
            runtime_evidence=changed,
        )


def test_cache_tags_do_not_key_request_generation_or_offset_metadata() -> None:
    runtime = _runtime()
    first_request = _request()
    first = bind_kmesh_request_lookup(
        _parse(),
        first_request,
        admission_generation=7,
        cache_domain="tenant-a",
        runtime_evidence=runtime,
    )

    shifted_tokens = (99, *TOKENS)
    shifted_payload = _payload(
        request_id="request-z",
        admission_generation=99,
        prompt_token_count=len(shifted_tokens),
        spans=(
            _span(
                segment_id="shifted-a",
                token_start=1,
                token_end=4,
                logical_order=4,
                tokens=shifted_tokens,
            ),
            _span(
                segment_id="shifted-b",
                token_start=6,
                token_end=9,
                logical_order=8,
                tokens=shifted_tokens,
            ),
        ),
    )
    second_request = _request(
        request_id="request-z", tokens=shifted_tokens, count=len(shifted_tokens)
    )
    second = bind_kmesh_request_lookup(
        _parse(shifted_payload),
        second_request,
        admission_generation=99,
        cache_domain="tenant-a",
        runtime_evidence=runtime,
    )

    assert first.cache_tags == second.cache_tags
    assert first.descriptor_fingerprint != second.descriptor_fingerprint
    for forbidden in ("request-a", "request-z"):
        assert forbidden not in repr(first.cache_tags)
        assert forbidden not in repr(second.cache_tags)


def test_domain_and_every_runtime_field_isolate_cache_tags() -> None:
    runtime = _runtime()
    baseline = dict(derive_kmesh_cache_tags("tenant-a", runtime))
    changed_domain = dict(derive_kmesh_cache_tags("tenant-b", runtime))
    assert changed_domain[KMESH_DOMAIN_TAG_KEY] != baseline[KMESH_DOMAIN_TAG_KEY]
    assert changed_domain[KMESH_RUNTIME_TAG_KEY] == baseline[KMESH_RUNTIME_TAG_KEY]

    string_fields = (
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
    )
    for field_name in string_fields:
        changed_runtime = replace(
            runtime, **{field_name: getattr(runtime, field_name) + "-changed"}
        )
        changed = dict(derive_kmesh_cache_tags("tenant-a", changed_runtime))
        assert changed[KMESH_DOMAIN_TAG_KEY] == baseline[KMESH_DOMAIN_TAG_KEY]
        assert changed[KMESH_RUNTIME_TAG_KEY] != baseline[KMESH_RUNTIME_TAG_KEY]
    for changed_runtime in (
        replace(runtime, tensor_parallel_size=2),
        replace(runtime, pipeline_parallel_size=2),
    ):
        changed = dict(derive_kmesh_cache_tags("tenant-a", changed_runtime))
        assert changed[KMESH_DOMAIN_TAG_KEY] == baseline[KMESH_DOMAIN_TAG_KEY]
        assert changed[KMESH_RUNTIME_TAG_KEY] != baseline[KMESH_RUNTIME_TAG_KEY]


@pytest.mark.parametrize(
    "request_configs",
    [
        {"lmcache.tag.user": "unsafe"},
        {KMESH_DOMAIN_TAG_KEY: "spoofed"},
        {KMESH_RUNTIME_TAG_KEY: "spoofed"},
    ],
)
def test_user_cache_tags_and_reserved_conflicts_are_rejected(
    request_configs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        bind_kmesh_request_lookup(
            _parse(),
            _request(),
            admission_generation=7,
            cache_domain="tenant-a",
            runtime_evidence=_runtime(),
            request_configs=request_configs,
        )


@pytest.mark.parametrize(
    "request_configs",
    [
        cast(Any, [("lmcache.foo", "bar")]),
        cast(Any, {1: "bar"}),
        {"": "bar"},
        {"lmcache.foo": object()},
        {"lmcache.foo": float("inf")},
        {"lmcache.foo": float("nan")},
    ],
)
def test_request_config_snapshot_rejects_ambiguous_values(request_configs) -> None:
    with pytest.raises((TypeError, ValueError)):
        bind_kmesh_request_lookup(
            _parse(),
            _request(),
            admission_generation=7,
            cache_domain="tenant-a",
            runtime_evidence=_runtime(),
            request_configs=request_configs,
        )


def test_request_config_merge_requires_exact_sorted_generated_tags() -> None:
    runtime = _runtime()
    tags = derive_kmesh_cache_tags("tenant-a", runtime)
    assert build_kmesh_request_configs({"z": 1, "a": False}, "tenant-a", runtime) == (
        ("a", False),
        *tags,
        ("z", 1),
    )
    with pytest.raises(ValueError, match="cache_domain"):
        build_kmesh_request_configs(None, "", runtime)
    with pytest.raises(TypeError, match="runtime_evidence"):
        build_kmesh_request_configs(None, "tenant-a", cast(Any, object()))


def test_binding_dto_rejects_tampered_derived_values() -> None:
    request = _request()
    binding = bind_kmesh_request_lookup(
        _parse(),
        request,
        admission_generation=7,
        cache_domain="tenant-a",
        runtime_evidence=_runtime(),
    )
    common: dict[str, Any] = {
        field_name: getattr(binding, field_name)
        for field_name in binding.__dataclass_fields__
    }
    mutations = (
        {"descriptor_fingerprint": "sha256-v1:" + "0" * 64},
        {"explicit_spans": ExplicitTokenSpans(8, TokenSpan(0, 8), (TokenSpan(0, 3),))},
        {"cache_tags": tuple(reversed(binding.cache_tags))},
        {"request_configs": binding.request_configs[:-1]},
    )
    for mutation in mutations:
        with pytest.raises((TypeError, ValueError)):
            KMeshRequestLookupBinding(**{**common, **mutation})


def test_binding_dto_validates_exact_types_before_aliasing_equality() -> None:
    tokens = (1,)
    descriptor = _parse(
        _payload(
            admission_generation=1,
            prompt_token_count=1,
            spans=(
                _span(
                    token_length=1,
                    token_start=0,
                    token_end=1,
                    tokens=tokens,
                ),
            ),
        )
    )
    request = _request(tokens=tokens, count=1)
    binding = bind_kmesh_request_lookup(
        descriptor,
        request,
        admission_generation=1,
        cache_domain="tenant-a",
        runtime_evidence=_runtime(),
        request_configs={"user.config": 1},
    )
    common: dict[str, Any] = {
        field_name: getattr(binding, field_name)
        for field_name in binding.__dataclass_fields__
    }

    descriptor_subclass = _ObserveOnlyDescriptorSubclass(
        lookup_intent=descriptor.lookup_intent,
        request_id=descriptor.request_id,
        admission_generation=descriptor.admission_generation,
        prompt_token_count=descriptor.prompt_token_count,
        cache_domain=descriptor.cache_domain,
        admission_fallback=descriptor.admission_fallback,
        spans=descriptor.spans,
    )
    explicit_subclass = _ExplicitTokenSpansSubclass(
        binding.explicit_spans.token_count,
        binding.explicit_spans.cached_span,
        binding.explicit_spans.spans,
    )
    tuple_spans = ExplicitTokenSpans(
        binding.explicit_spans.token_count,
        binding.explicit_spans.cached_span,
        _TupleSubclass(binding.explicit_spans.spans),
    )
    token_span_subclass = ExplicitTokenSpans(
        binding.explicit_spans.token_count,
        _TokenSpanSubclass(0, 1),
        binding.explicit_spans.spans,
    )
    forged_bool_count = object.__new__(ExplicitTokenSpans)
    object.__setattr__(forged_bool_count, "token_count", True)
    object.__setattr__(
        forged_bool_count, "cached_span", binding.explicit_spans.cached_span
    )
    object.__setattr__(forged_bool_count, "spans", binding.explicit_spans.spans)

    mutations: tuple[dict[str, object], ...] = (
        {"descriptor": descriptor_subclass},
        {"request_id": _StringSubclass(binding.request_id)},
        {"prompt_token_ids": (True,)},
        {"prompt_token_ids": _TupleSubclass(binding.prompt_token_ids)},
        {"prompt_token_ids": _ListSubclass(binding.prompt_token_ids)},
        {"prompt_token_count": True},
        {"prompt_token_count": _IntSubclass(1)},
        {"admission_generation": True},
        {"admission_generation": _IntSubclass(1)},
        {"cache_domain": _StringSubclass(binding.cache_domain)},
        {"descriptor_fingerprint": _StringSubclass(binding.descriptor_fingerprint)},
        {"explicit_spans": explicit_subclass},
        {"explicit_spans": tuple_spans},
        {"explicit_spans": token_span_subclass},
        {"explicit_spans": forged_bool_count},
        {
            "runtime_evidence": _RuntimeEvidenceFingerprintSubclass(
                **{
                    field_name: getattr(binding.runtime_evidence, field_name)
                    for field_name in binding.runtime_evidence.__dataclass_fields__
                }
            )
        },
        {"cache_tags": _TupleSubclass(binding.cache_tags)},
        {
            "cache_tags": (
                _TupleSubclass(binding.cache_tags[0]),
                *binding.cache_tags[1:],
            )
        },
        {
            "cache_tags": (
                (
                    _StringSubclass(binding.cache_tags[0][0]),
                    binding.cache_tags[0][1],
                ),
                *binding.cache_tags[1:],
            )
        },
        {"request_configs": _TupleSubclass(binding.request_configs)},
        {
            "request_configs": (
                _TupleSubclass(binding.request_configs[0]),
                *binding.request_configs[1:],
            )
        },
        {
            "request_configs": (
                (_StringSubclass("user.config"), 1),
                *binding.request_configs[1:],
            )
        },
        {
            "request_configs": (
                ("user.config", _IntSubclass(1)),
                *binding.request_configs[1:],
            )
        },
    )
    for mutation in mutations:
        with pytest.raises(TypeError):
            KMeshRequestLookupBinding(**{**common, **mutation})


def test_execution_evidence_is_pickle_safe_and_drops_live_request() -> None:
    request = _request()
    binding = bind_kmesh_request_lookup(
        _parse(),
        request,
        admission_generation=7,
        cache_domain="tenant-a",
        runtime_evidence=_runtime(),
        request_configs={"user.config": 1},
    )

    evidence = KMeshRequestExecutionEvidence.from_binding(binding)
    rebuilt = pickle.loads(pickle.dumps(evidence))

    assert type(rebuilt) is KMeshRequestExecutionEvidence
    assert rebuilt == evidence
    assert rebuilt.request_id == "request-a"
    assert rebuilt.prompt_token_ids == TOKENS
    assert rebuilt.explicit_spans.token_count == len(TOKENS)
    assert rebuilt.as_request_configs() == dict(binding.request_configs)
    assert not hasattr(rebuilt, "request_object")
    rebuilt.validate_peer_descriptor(_parse())
    rebuilt.validate_runtime_evidence(_runtime())


def test_execution_evidence_revalidates_binding_before_freezing() -> None:
    request = _request()
    binding = bind_kmesh_request_lookup(
        _parse(),
        request,
        admission_generation=7,
        cache_domain="tenant-a",
        runtime_evidence=_runtime(),
    )
    cast(list[int], request.prompt_token_ids)[0] = 99

    with pytest.raises(ValueError, match="token slice"):
        KMeshRequestExecutionEvidence.from_binding(binding)


def test_execution_evidence_rejects_tampering_and_loose_types() -> None:
    binding = bind_kmesh_request_lookup(
        _parse(),
        _request(),
        admission_generation=7,
        cache_domain="tenant-a",
        runtime_evidence=_runtime(),
    )
    evidence = KMeshRequestExecutionEvidence.from_binding(binding)
    common: dict[str, Any] = {
        field_name: getattr(evidence, field_name)
        for field_name in evidence.__dataclass_fields__
    }
    mutations: tuple[dict[str, object], ...] = (
        {"prompt_token_ids": _TupleSubclass(evidence.prompt_token_ids)},
        {"prompt_token_ids": (*evidence.prompt_token_ids[:-1], 99)},
        {"descriptor_fingerprint": "sha256-v1:" + "0" * 64},
        {
            "explicit_spans": ExplicitTokenSpans(
                len(TOKENS),
                TokenSpan(0, len(TOKENS)),
                (TokenSpan(0, 3),),
            )
        },
        {"runtime_evidence": replace(_runtime(), model_revision="changed")},
        {"cache_tags": tuple(reversed(evidence.cache_tags))},
        {"request_configs": evidence.request_configs[:-1]},
    )
    for mutation in mutations:
        with pytest.raises((TypeError, ValueError)):
            KMeshRequestExecutionEvidence(**{**common, **mutation})


def test_execution_evidence_rejects_peer_or_worker_runtime_drift() -> None:
    evidence = KMeshRequestExecutionEvidence.from_binding(
        bind_kmesh_request_lookup(
            _parse(),
            _request(),
            admission_generation=7,
            cache_domain="tenant-a",
            runtime_evidence=_runtime(),
        )
    )

    with pytest.raises(ValueError, match="descriptor differs"):
        evidence.validate_peer_descriptor(_parse(_payload(request_id="request-b")))
    with pytest.raises(TypeError, match="exactly KMeshRequestLookupDescriptor"):
        evidence.validate_peer_descriptor(cast(Any, object()))
    with pytest.raises(ValueError, match="runtime evidence differs"):
        evidence.validate_runtime_evidence(
            replace(_runtime(), attention_backend="changed")
        )
    with pytest.raises(TypeError, match="exactly KMeshRuntimeEvidence"):
        evidence.validate_runtime_evidence(cast(Any, object()))
