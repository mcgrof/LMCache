# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the serde factory + config wiring.

Covers:
- Built-in ``"fp8"`` registration.
- Custom factory registration + dispatch.
- Error paths for missing/unknown ``type``.
- L2 adapter JSON config parses ``serde`` sub-dict.
"""

# Standard
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    add_l2_adapters_args,
    parse_args_to_l2_adapters_config,
)
from lmcache.v1.distributed.serde import (
    AsyncSerdeProcessor,
    SerdeConfig,
    SerdeProcessor,
    create_serde_processor,
    get_registered_serde_types,
    register_serde_factory,
)

# =============================================================================
# Factory registration + dispatch
# =============================================================================


def test_fp8_is_registered_by_default() -> None:
    """The built-in fp8 serde is available without extra setup."""
    assert "fp8" in get_registered_serde_types()


def test_create_fp8_returns_async_processor() -> None:
    """fp8 config produces an AsyncSerdeProcessor with distinct event fds."""
    processor = create_serde_processor(SerdeConfig(type="fp8"))
    try:
        assert isinstance(processor, AsyncSerdeProcessor)
        s_fd = processor.get_serialize_event_fd()
        d_fd = processor.get_deserialize_event_fd()
        assert s_fd != d_fd
    finally:
        processor.close()


def test_create_serde_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown serde type"):
        create_serde_processor(SerdeConfig(type="does-not-exist"))


def test_create_fp8_accepts_float_max_workers() -> None:
    """``max_workers`` from a YAML float (e.g. 2.0) must round to int.

    Regression: the old ``int(str(...))`` parse rejected float-encoded
    integers; direct ``int(...)`` handles ints, floats, and digit
    strings uniformly.
    """
    processor = create_serde_processor(
        SerdeConfig(type="fp8", kwargs={"max_workers": 2.0})
    )
    try:
        assert isinstance(processor, AsyncSerdeProcessor)
    finally:
        processor.close()


def test_register_serde_factory_dispatch() -> None:
    """A custom factory is dispatched by its registered name."""

    seen: dict[str, dict] = {}

    class _DummyProcessor(SerdeProcessor):
        def get_serialize_event_fd(self) -> int:
            return -1

        def get_deserialize_event_fd(self) -> int:
            return -1

        def submit_serialize(self, src_objs, dst_objs):  # type: ignore[no-untyped-def]
            return 0

        def query_serialize_result(self, task_id):  # type: ignore[no-untyped-def]
            return True

        def submit_deserialize(self, src_objs, dst_objs):  # type: ignore[no-untyped-def]
            return 0

        def query_deserialize_result(self, task_id):  # type: ignore[no-untyped-def]
            return True

        def estimate_serialized_size(self, layout_desc) -> int:  # type: ignore[no-untyped-def]
            return 1

        def close(self) -> None:
            pass

    def _factory(kwargs: dict) -> SerdeProcessor:
        seen["kwargs"] = kwargs
        return _DummyProcessor()

    # Use a unique name to avoid collisions if the test runs twice.
    register_serde_factory("test-dummy-ser-de-xyz", _factory)

    processor = create_serde_processor(
        SerdeConfig(type="test-dummy-ser-de-xyz", kwargs={"foo": "bar"})
    )
    assert isinstance(processor, _DummyProcessor)
    # Factory only receives the type-specific kwargs, not the wrapping type.
    assert seen["kwargs"] == {"foo": "bar"}


def test_register_serde_factory_duplicate_raises() -> None:
    """Registering the same name twice is rejected."""

    def _factory(config: dict) -> SerdeProcessor:  # pragma: no cover - not called
        raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        register_serde_factory("fp8", _factory)


# =============================================================================
# L2 adapter JSON config
# =============================================================================


def _parse_adapter(spec: dict):  # type: ignore[no-untyped-def]
    """Helper: run the argparse plumbing on a single adapter JSON spec."""
    # Standard
    import argparse

    parser = argparse.ArgumentParser()
    add_l2_adapters_args(parser)
    args = parser.parse_args(["--l2-adapter", json.dumps(spec)])
    cfg = parse_args_to_l2_adapters_config(args)
    return cfg.adapters[0]


def test_adapter_config_without_serde() -> None:
    adapter = _parse_adapter({"type": "mock", "max_size_gb": 1, "mock_bandwidth_gb": 1})
    assert adapter.serde_config is None


def test_adapter_config_with_serde() -> None:
    serde_spec = {"type": "fp8", "fp8_dtype": "float8_e4m3fn"}
    adapter = _parse_adapter(
        {"type": "mock", "max_size_gb": 1, "mock_bandwidth_gb": 1, "serde": serde_spec}
    )
    assert adapter.serde_config is not None
    assert adapter.serde_config.type == "fp8"
    assert adapter.serde_config.kwargs == {"fp8_dtype": "float8_e4m3fn"}


def test_adapter_config_rejects_non_dict_serde() -> None:
    with pytest.raises(ValueError, match="'serde' must be a dict"):
        _parse_adapter(
            {"type": "mock", "max_size_gb": 1, "mock_bandwidth_gb": 1, "serde": "fp8"}
        )


def test_adapter_config_rejects_serde_without_type() -> None:
    with pytest.raises(ValueError, match="'serde' dict must include a 'type' field"):
        _parse_adapter(
            {
                "type": "mock",
                "max_size_gb": 1,
                "mock_bandwidth_gb": 1,
                "serde": {"fp8_dtype": "float8_e4m3fn"},
            }
        )


# =============================================================================
# Asym K16/V8 multi-output serde factory registration
# =============================================================================


def test_asym_k16_v8_storage_only_is_registered() -> None:
    """Storage-only Mode 1 is selectable from YAML by name."""
    assert "asym_k16_v8" in get_registered_serde_types()


def test_asym_k16_v8_v_only_is_registered() -> None:
    """V-only Mode 2 is selectable from YAML by name."""
    assert "asym_k16_v8_v_only" in get_registered_serde_types()


def test_create_asym_k16_v8_returns_async_processor_with_identity_mapping() -> None:
    """Storage-only Mode 1: identity slot mapping (0, 1) — slot 0 reads
    parent group 0 (K), slot 1 reads parent group 1 (V)."""
    processor = create_serde_processor(SerdeConfig(type="asym_k16_v8"))
    try:
        assert isinstance(processor, AsyncSerdeProcessor)
        assert processor.serializer_group_size() == 2
        assert processor.deserializer_group_size() == 2
        assert processor.input_slot_mapping() == (0, 1)
        assert processor.output_slot_mapping() == (0, 1)
        s_fd = processor.get_serialize_event_fd()
        d_fd = processor.get_deserialize_event_fd()
        assert s_fd != d_fd
    finally:
        processor.close()


def test_create_asym_k16_v8_v_only_returns_async_processor_with_skip_k_mapping() -> None:
    """V-only Mode 2: (None, 1) slot mapping — slot 0 is absent (K stays
    in L1), slot 1 reads parent group 1 (V)."""
    processor = create_serde_processor(SerdeConfig(type="asym_k16_v8_v_only"))
    try:
        assert isinstance(processor, AsyncSerdeProcessor)
        assert processor.serializer_group_size() == 2
        assert processor.input_slot_mapping() == (None, 1)
        assert processor.output_slot_mapping() == (None, 1)
    finally:
        processor.close()


def test_create_asym_k16_v8_honors_custom_kwargs() -> None:
    """fp8_dtype, scale_scope, scale_dtype, max_workers kwargs round-trip
    through the factory without raising."""
    processor = create_serde_processor(
        SerdeConfig(
            type="asym_k16_v8",
            kwargs={
                "fp8_dtype": "float8_e4m3fn",
                "scale_scope": "PER_TENSOR",
                "scale_dtype": "float32",
                "max_workers": 2,
            },
        )
    )
    try:
        assert isinstance(processor, AsyncSerdeProcessor)
    finally:
        processor.close()


def test_create_asym_k16_v8_defaults_max_workers_to_4() -> None:
    """The factory defaults to a multi-threaded drainer because the
    codec is CPU-bound.  Phase 1.5 confirmed that the AsyncSerdeProcessor
    base default of 1 causes a 20x p99 regression at single-producer
    load (the drainer becomes the bottleneck, not the codec itself).
    The asym factory overrides to 4 so the default config does not
    silently land users on the slow path."""
    processor = create_serde_processor(SerdeConfig(type="asym_k16_v8"))
    try:
        assert isinstance(processor, AsyncSerdeProcessor)
        # ThreadPoolExecutor exposes _max_workers; pin the contract so a
        # future refactor doesn't silently flip back to 1.
        assert processor._pool._max_workers == 4  # type: ignore[attr-defined]
    finally:
        processor.close()


def test_create_asym_k16_v8_v_only_defaults_max_workers_to_4() -> None:
    """V-only carries the same single-thread-drainer footgun as the
    storage-only variant, so the factory default is 4 there too."""
    processor = create_serde_processor(SerdeConfig(type="asym_k16_v8_v_only"))
    try:
        assert isinstance(processor, AsyncSerdeProcessor)
        assert processor._pool._max_workers == 4  # type: ignore[attr-defined]
    finally:
        processor.close()


def test_create_asym_k16_v8_rejects_unknown_fp8_dtype() -> None:
    """A typo'd fp8 dtype is rejected at factory time with a clear
    error, not deferred to first use."""
    with pytest.raises(ValueError, match="Unknown torch dtype"):
        create_serde_processor(
            SerdeConfig(type="asym_k16_v8", kwargs={"fp8_dtype": "float8_TYPO"})
        )


def test_create_asym_k16_v8_rejects_unknown_scale_scope() -> None:
    """A typo'd scale_scope is rejected at factory time."""
    with pytest.raises(ValueError, match="Unknown ScaleScope"):
        create_serde_processor(
            SerdeConfig(type="asym_k16_v8", kwargs={"scale_scope": "PER_BANANA"})
        )


def test_single_tensor_processor_returns_none_slot_mapping() -> None:
    """Default base-class slot mapping accessors return None for
    single-tensor serdes (the wrapper uses None to take the
    pass-through dispatch path)."""
    processor = create_serde_processor(SerdeConfig(type="fp8"))
    try:
        assert processor.input_slot_mapping() is None
        assert processor.output_slot_mapping() is None
        assert processor.serializer_group_size() is None
        assert processor.deserializer_group_size() is None
    finally:
        processor.close()
