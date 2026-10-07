# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for the V2/V3 staging handoff contract.

Run directly with ``python tests/v1/gpu_connector/test_vllm_staging_ordering.py``
when torch, native extensions, or pytest are unavailable. These tests also work
under pytest. Set ``LMCACHE_STAGING_ORDERING_SOURCE`` to another checkout's
``gpu_connectors.py`` to run the same contract against a baseline or mutation.

The loader compiles the implementation's actual public method bodies and their
called helpers. It replaces only CUDA/native dependencies and pointer discovery;
it never restates connector control flow. Deferred streams make missing ordering
deterministic: a buffer's bytes stay poisoned until the stream that fills it is
driven. Assertions inspect payload values at the public handoff and after reuse,
not the number of synchronization calls.

This models host-side ordering. It does NOT test CUDA memory visibility, native
kernel correctness, dma-buf registration, NVMe I/O, or fatal CUDA context errors.
Real GPU and storage stress tests are required separately.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast
from unittest.mock import patch
import ast
import gc
import itertools
import os
import unittest
import weakref


def _source_path() -> Path:
    """Return the implementation under test, or the explicit baseline override."""
    override = os.environ.get("LMCACHE_STAGING_ORDERING_SOURCE")
    if override:
        return Path(override)
    return (
        Path(__file__).resolve().parents[3]
        / "lmcache/v1/gpu_connector/gpu_connectors.py"
    )


def _load_connector(version: int, cuda: DeferredCuda) -> type[Connector]:
    """Load actual methods while isolating GPU-only import and initialization.

    Args:
        version: Connector version, 2 or 3.
        cuda: Deferred native transfer and stream environment.

    Returns:
        A class with the original public methods and their called helpers.

    Raises:
        ValueError: The source does not contain the requested connector.
    """
    path = _source_path()
    name = f"VLLMPagedMemGPUConnectorV{version}"
    parsed = ast.parse(path.read_text(), filename=str(path))
    candidates = [
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one {name} in {path}")
    methods = {
        node.name: node
        for node in candidates[0].body
        if isinstance(node, ast.FunctionDef)
    }
    selected = {"from_gpu", "batched_from_gpu", "to_gpu", "batched_to_gpu"}
    # Pointer discovery touches CUDA allocations and native layout machinery.
    # These boundary methods are supplied by the fixture instead.
    boundaries = {"_initialize_pointers", "_initialize_kv_cache_pointers"}
    while True:
        dependencies = {
            call.func.attr
            for method in selected
            for call in ast.walk(methods[method])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr in methods
            and call.func.attr not in boundaries
        }
        if dependencies <= selected:
            break
        selected |= dependencies
    bodies = [node for key, node in methods.items() if key in selected]
    for method in bodies:
        method.decorator_list = [
            decorator
            for decorator in method.decorator_list
            if not (
                isinstance(decorator, ast.Name)
                and decorator.id == "_lmcache_nvtx_annotate"
            )
        ]
    helpers = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_check_staging_ready", "_synchronize_staging"}
    ]
    memory_path = (
        Path(__file__).resolve().parents[3] / "lmcache/v1/memory_management.py"
    )
    retention = next(
        node
        for node in ast.parse(memory_path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "retain_memory_owners"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            retention,
            *helpers,
            ast.ClassDef(
                name=name,
                bases=[],
                keywords=[],
                body=cast(list[ast.stmt], bodies),
                decorator_list=[],
                type_params=[],
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "cast": cast,
        "torch": SimpleNamespace(cuda=cuda, Tensor=Payload),
        "MemoryObj": MemoryObject,
        "_FAILED_STAGING_CONNECTORS": {},
        "_UNCERTAIN_MEMORY_OWNERS": [],
        "MemoryFormat": SimpleNamespace(KV_2LTD="KV_2LTD", KV_MLA_FMT="KV_MLA_FMT"),
        "device_ops": SimpleNamespace(multi_layer_kv_transfer=cuda.transfer),
        "lmcache_native": SimpleNamespace(
            TransferDirection=SimpleNamespace(D2H="gather", H2D="scatter")
        ),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return cast(type[Connector], namespace[name])


def _load_musa_batch(cuda: DeferredCuda) -> type[Connector]:
    """Load MUSA's actual batch dispatch over the actual CUDA parent batch.

    Only ``from_gpu`` is replaced by the test: MUSA's implementation is an
    external-device boundary. This catches an accidental inherited CUDA path
    without importing either torch or torch_musa.
    """
    path = _source_path().with_name("musa_connectors.py")
    name = "VLLMPagedMemMUSAConnectorV2"
    source = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "batched_from_gpu"
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            ast.ClassDef(
                name=name,
                bases=[ast.Name(id="Parent", ctx=ast.Load())],
                keywords=[],
                body=cast(list[ast.stmt], methods) if methods else [ast.Pass()],
                decorator_list=[],
                type_params=[],
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"Parent": _load_connector(2, cuda), "cast": cast}
    exec(compile(module, str(path), "exec"), namespace)
    return cast(type[Connector], namespace[name])


class DeferredStream:
    """A stream whose operations execute only when a completion wait drives it."""

    def __init__(self) -> None:
        self.device = "cuda:0"
        self.operations: list[Callable[[], None]] = []
        self.completed = 0
        self.fail_synchronization = False

    def enqueue(self, operation: Callable[[], None]) -> None:
        """Append work without executing it."""
        self.operations.append(operation)

    def run_until(self, boundary: int) -> None:
        """Execute through a captured stream boundary, propagating failures."""
        while self.completed < boundary:
            operation = self.operations[self.completed]
            self.completed += 1
            operation()

    def wait_stream(self, producer: DeferredStream) -> None:
        """Order later operations after work already queued by the producer."""
        boundary = len(producer.operations)
        self.enqueue(lambda: producer.run_until(boundary))

    def synchronize(self) -> None:
        """Complete this stream and its explicit dependencies."""
        if self.fail_synchronization:
            raise RuntimeError("injected synchronization failure")
        self.run_until(len(self.operations))

    def pending(self) -> int:
        """Return the amount of work that may still access its buffers."""
        return len(self.operations) - self.completed


class DeferredCuda:
    """Replace stream selection and native transfers without importing torch."""

    def __init__(self) -> None:
        self.current = DeferredStream()
        self.transfer_count = 0
        self.fail_after_enqueue = 0

    def current_stream(self, device: object = None) -> DeferredStream:
        """Return the selected stream; the fixture has a single CUDA device."""
        del device
        return self.current

    @contextmanager
    def stream(self, stream: DeferredStream) -> Iterator[None]:
        """Select the stream used to enqueue transfers in the context body."""
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous

    def transfer(
        self,
        staging: Payload,
        cache: Payload,
        slots: list[int],
        device: object,
        page_buffer_size: int,
        direction: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Queue a gather/scatter, optionally raising after work was enqueued.

        The injected exception represents an ordinary submission-side failure;
        the stream remains able to drain. CUDA context loss is outside this model.
        """
        del device, page_buffer_size, args

        def move() -> None:
            if direction == "gather":
                staging.values[:] = [cache.values[index] for index in slots]
            elif direction == "scatter":
                skip = cast(int, kwargs.get("skip_prefix_n_tokens", 0))
                for offset, index in enumerate(slots):
                    if offset >= skip:
                        cache.values[index] = staging.values[offset]
            else:
                raise ValueError(f"Unknown direction: {direction}")

        self.current.enqueue(move)
        self.transfer_count += 1
        if self.transfer_count == self.fail_after_enqueue:
            raise RuntimeError("injected error after native work was enqueued")


class Payload:
    """A token vector with deferred copies and the minimal tensor accessors."""

    def __init__(self, cuda: DeferredCuda, device: str, values: list[int]) -> None:
        self.cuda = cuda
        self.is_cuda = device == "cuda:0"
        self.device = device
        self.values = values.copy()
        self.shape = (2, 1, len(values), 1)

    def __getitem__(self, selection: object) -> Payload:
        """Return the whole chunk-sized temporary view used in these cases."""
        del selection
        return self

    def copy_(self, source: Payload, **kwargs: object) -> None:
        """Queue a copy whose source is read when the stream reaches it."""
        if kwargs.get("non_blocking") is not True:
            raise ValueError("The fixture models nonblocking connector copies")
        self.cuda.current.enqueue(
            lambda: self.values.__setitem__(slice(None), source.values)
        )


class MemoryObject:
    """A staging allocation exposing the connector's public buffer interface."""

    def __init__(self, tensor: Payload) -> None:
        self.tensor = tensor
        self.raw_tensor = tensor
        self.metadata = SimpleNamespace(fmt="KV_2LTD")
        self.ref_count = 1

    def ref_count_up(self) -> None:
        """Acquire an allocation owner, preventing reuse after uncertain work."""
        self.ref_count += 1

    def ref_count_down(self) -> None:
        """Release the caller's allocation ownership."""
        self.ref_count -= 1

    def get_tensor(self, index: int) -> Payload:
        """Return the sole group, rejecting an unexpected group index."""
        if index != 0:
            raise IndexError(index)
        return self.tensor


class SlotMapping(list[int]):
    """A weak-referenceable source operand for the unknown-completion test."""


class Connector(Protocol):
    """The public transfer API exercised by this source-isolated test."""

    store_stream: DeferredStream
    load_stream: DeferredStream

    def from_gpu(
        self, memory_obj: MemoryObject, start: int, end: int, **kwargs: object
    ) -> None: ...

    def batched_from_gpu(
        self,
        memory_objs: list[MemoryObject],
        starts: list[int],
        ends: list[int],
        **kwargs: object,
    ) -> None: ...

    def batched_to_gpu(
        self,
        memory_objs: list[MemoryObject],
        starts: list[int],
        ends: list[int],
        **kwargs: object,
    ) -> None: ...


class Scenario:
    """Two independent CUDA streams and a caller-owned KV/staging fixture."""

    def __init__(self, version: int, target: str, temporary: str, count: int) -> None:
        self.cuda = DeferredCuda()
        self.producer = self.cuda.current
        self.cache = Payload(self.cuda, "cuda:0", [-7] * (4 * count))
        self.expected = [101 + 17 * index for index in range(4 * count)]
        self.objects = [
            MemoryObject(Payload(self.cuda, target, [-99] * 4)) for _ in range(count)
        ]
        self.slots = [base + i for base in range(0, 4 * count, 4) for i in (3, 1, 0, 2)]
        self.starts = list(range(0, 4 * count, 4))
        self.ends = [start + 4 for start in self.starts]
        self.scratch = Payload(self.cuda, "cuda:0", [-88] * 4)
        self.connector = _load_connector(version, self.cuda)()
        # Only construction and GPU pointer-discovery boundaries are replaced.
        # Test actions below call the actual public implementation methods.
        attributes: dict[str, object] = {
            "device": "cuda:0",
            "kvcaches": [self.cache],
            "store_stream": DeferredStream(),
            "load_stream": DeferredStream(),
            "use_mla": False,
            "page_buffer_size": 4 * count,
            "engine_kv_format": "stub",
            "block_size": 2,
            "head_size": 1,
            "block_stride_elems": 0,
            "initialize_kvcaches_ptr": lambda **kwargs: None,
            "_initialize_pointers": lambda kvcaches: self.cache,
            "_initialize_kv_cache_pointers": lambda: None,
            "group_kv_cache_pointers_on_gpu": [self.cache],
            "use_gpu": temporary == "temporary",
            "chunk_size": 4,
            "gpu_buffer": self.scratch if temporary == "temporary" else None,
            "group_tmp_buffer": [self.scratch] if temporary == "temporary" else None,
        }
        for name, value in attributes.items():
            setattr(self.connector, name, value)

    def queue_producer(self) -> None:
        """Leave fresh KV bytes pending on the caller's current stream."""
        self.producer.enqueue(
            lambda: self.cache.values.__setitem__(slice(None), self.expected)
        )

    def store(self, mode: str) -> None:
        """Exercise the single or batched public offload staging operation."""
        if mode == "single":
            self.connector.from_gpu(self.objects[0], 0, 4, slot_mapping=self.slots)
        elif mode == "batch":
            self.connector.batched_from_gpu(
                self.objects, self.starts, self.ends, slot_mapping=self.slots
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def expected_chunks(self) -> list[list[int]]:
        """Return the independent storage oracle in gathered token order."""
        return [
            [self.expected[index] for index in self.slots[start:end]]
            for start, end in zip(self.starts, self.ends, strict=True)
        ]


class TestVLLMStagingOrdering(unittest.TestCase):
    """Verify payload handoff and lifetime through the public connector API."""

    def test_store_handoff_contains_completed_producer_bytes(self) -> None:
        """CPU and CUDA staging are ready at handoff for all ordinary routes."""
        for version, target, temporary, mode, producer in itertools.product(
            (2, 3),
            ("cpu", "cuda:0"),
            ("direct", "temporary"),
            ("single", "batch"),
            ("ready", "pending"),
        ):
            with self.subTest(
                version=version,
                target=target,
                temporary=temporary,
                mode=mode,
                producer=producer,
            ):
                scenario = Scenario(
                    version, target, temporary, 2 if mode == "batch" else 1
                )
                if producer == "pending":
                    scenario.queue_producer()
                else:
                    scenario.cache.values[:] = scenario.expected
                scenario.store(mode)
                # An external I/O engine can read now, without a CUDA call.
                self.assertEqual(
                    [obj.tensor.values for obj in scenario.objects],
                    scenario.expected_chunks(),
                )

    def test_failed_stream_wait_retains_owners_and_refuses_reuse(self) -> None:
        """Unknown GPU completion keeps allocation refs after caller cleanup."""
        for version, direction in itertools.product((2, 3), ("store", "load")):
            with self.subTest(version=version, direction=direction):
                scenario = Scenario(version, "cuda:0", "direct", 2)
                scenario.slots = SlotMapping(scenario.slots)
                mapping_ref = weakref.ref(scenario.slots)
                stream = (
                    scenario.connector.store_stream
                    if direction == "store"
                    else scenario.connector.load_stream
                )
                stream.fail_synchronization = True
                with self.assertRaisesRegex(RuntimeError, "synchronization failure"):
                    if direction == "store":
                        scenario.store("batch")
                    else:
                        scenario.connector.batched_to_gpu(
                            scenario.objects,
                            scenario.starts,
                            scenario.ends,
                            slot_mapping=scenario.slots,
                        )
                self.assertGreater(stream.pending(), 0)
                for obj in scenario.objects:
                    obj.ref_count_down()
                    self.assertEqual(obj.ref_count, 1)
                scenario.slots = []
                gc.collect()
                self.assertIsNotNone(mapping_ref())
                with self.assertRaisesRegex(RuntimeError, "unknown GPU completion"):
                    scenario.store("batch")

    def test_caller_can_join_another_producer_stream(self) -> None:
        """A caller-established dependency reaches the storage handoff."""
        for version in (2, 3):
            with self.subTest(version=version):
                scenario = Scenario(version, "cuda:0", "direct", 2)
                other = DeferredStream()

                def populate_other(case: Scenario = scenario) -> None:
                    case.cache.values[:] = case.expected

                other.enqueue(populate_other)
                scenario.producer.wait_stream(other)
                scenario.store("batch")
                self.assertEqual(
                    [obj.tensor.values for obj in scenario.objects],
                    scenario.expected_chunks(),
                )

    def test_empty_store_does_not_consume_pending_producer_work(self) -> None:
        """An empty batch leaves unrelated producer work pending."""
        for version in (2, 3):
            with self.subTest(version=version):
                scenario = Scenario(version, "cuda:0", "direct", 1)
                scenario.queue_producer()
                scenario.connector.batched_from_gpu([], [], [])
                self.assertEqual(scenario.cache.values, [-7] * 4)

    def test_mismatched_store_batch_is_rejected_before_data_moves(self) -> None:
        """Unequal lists cannot silently store only a prefix of the request."""
        for version in (2, 3):
            with self.subTest(version=version):
                scenario = Scenario(version, "cuda:0", "direct", 2)
                scenario.queue_producer()
                with self.assertRaises(ValueError):
                    scenario.connector.batched_from_gpu(
                        scenario.objects,
                        [0],
                        scenario.ends,
                        slot_mapping=scenario.slots,
                    )
                self.assertEqual(
                    [obj.tensor.values for obj in scenario.objects], [[-99] * 4] * 2
                )

    def test_store_exception_drains_work_already_enqueued(self) -> None:
        """An enqueue error cannot leave staging or temporary-buffer access pending."""
        for version, mode, temporary in itertools.product(
            (2, 3), ("single", "batch"), ("direct", "temporary")
        ):
            with self.subTest(version=version, mode=mode, temporary=temporary):
                count = 2 if mode == "batch" else 1
                scenario = Scenario(version, "cuda:0", temporary, count)
                scenario.queue_producer()
                scenario.cuda.fail_after_enqueue = count
                with self.assertRaisesRegex(RuntimeError, "injected error"):
                    scenario.store(mode)
                expected = scenario.expected_chunks()
                if temporary == "temporary":
                    # The error is raised after the last gather but before its
                    # scratch-to-staging copy. That last scratch write must end
                    # before the failed operation hands the scratch buffer back.
                    self.assertEqual(scenario.scratch.values, expected[-1])
                    expected[-1] = [-99] * 4
                self.assertEqual(
                    [obj.tensor.values for obj in scenario.objects], expected
                )
                self.assertEqual(scenario.connector.store_stream.pending(), 0)
                # Reusing these slots after the exception must not expose a
                # still-running gather that overwrites the next owner's data.
                for obj in scenario.objects:
                    obj.tensor.values[:] = [-555] * 4
                scenario.scratch.values[:] = [-666] * 4
                scenario.connector.store_stream.synchronize()
                self.assertEqual(
                    [obj.tensor.values for obj in scenario.objects],
                    [[-555] * 4] * count,
                )
                self.assertEqual(scenario.scratch.values, [-666] * 4)

    def test_restore_finishes_reading_staging_before_reuse(self) -> None:
        """A completed batch restore no longer accesses its staging sources."""
        for version, target in itertools.product((2, 3), ("cpu", "cuda:0")):
            with self.subTest(version=version, target=target):
                scenario = Scenario(version, target, "direct", 2)
                for obj, values in zip(
                    scenario.objects, scenario.expected_chunks(), strict=True
                ):
                    obj.tensor.values[:] = values
                scenario.connector.batched_to_gpu(
                    scenario.objects,
                    scenario.starts,
                    scenario.ends,
                    slot_mapping=scenario.slots,
                )
                # Model immediate custom-pool reuse by another owner.
                for obj in scenario.objects:
                    obj.tensor.values[:] = [-555] * 4
                scenario.connector.load_stream.synchronize()
                self.assertEqual(scenario.cache.values, scenario.expected)

    def test_restore_exception_drains_scatter_before_propagating(self) -> None:
        """Even a partial batch error ends already-enqueued accesses to staging."""
        for version in (2, 3):
            with self.subTest(version=version):
                scenario = Scenario(version, "cuda:0", "direct", 2)
                for obj, values in zip(
                    scenario.objects, scenario.expected_chunks(), strict=True
                ):
                    obj.tensor.values[:] = values
                scenario.cuda.fail_after_enqueue = 2
                with self.assertRaisesRegex(RuntimeError, "injected error"):
                    scenario.connector.batched_to_gpu(
                        scenario.objects,
                        scenario.starts,
                        scenario.ends,
                        slot_mapping=scenario.slots,
                    )
                for obj in scenario.objects:
                    obj.tensor.values[:] = [-555] * 4
                scenario.connector.load_stream.synchronize()
                self.assertEqual(scenario.cache.values, scenario.expected)

    def test_musa_batch_keeps_its_device_specific_dispatch(self) -> None:
        """MUSA batches reach the MUSA transfer operation without CUDA streams."""
        cuda = DeferredCuda()
        connector = _load_musa_batch(cuda)()
        objects = [MemoryObject(Payload(cuda, "cpu", [-99] * 4)) for _ in range(2)]

        def musa_transfer(
            memory_obj: MemoryObject, start: int, end: int, **kwargs: object
        ) -> None:
            del kwargs
            memory_obj.tensor.values[:] = list(range(100 + start, 100 + end))

        # An instance override is the device-specific public transfer boundary.
        # There deliberately is no CUDA store_stream on this MUSA fixture.
        with patch.object(connector, "from_gpu", musa_transfer):
            connector.batched_from_gpu(objects, [0, 4], [4, 8])
        self.assertEqual(
            [obj.tensor.values for obj in objects],
            [[100, 101, 102, 103], [104, 105, 106, 107]],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
