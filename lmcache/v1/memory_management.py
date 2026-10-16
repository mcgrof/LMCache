# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache, partial, wraps
from types import MappingProxyType
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
import abc
import ctypes
import os
import threading

# Third Party
from sortedcontainers import SortedList
import torch

# First Party
from lmcache import torch_dev
from lmcache import torch_device_type as torch_device_type  # noqa: F401
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.lifecycle import RetainedCall, _run_under_interrupt_safe_rlock
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.platform import current_device_spec as current_device_spec  # noqa: F401
from lmcache.v1.system_detection import NUMAMapping
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


# Cache for ctypes ubyte-array types keyed by length.
#
# ctypes does not cache `(c_ubyte * N)` array types -- each call to the `*`
# operator builds a fresh heap type via PyCArrayType_from_ctype. The heap
# type metadata stays alive forever (held by the type system), so calling
# `(ctypes.c_ubyte * N).from_address(...)` on every TensorMemoryObj.byte_array
# access leaks ~1-2 kB per call. Under long-running remote-backend put/get
# workloads this is the dominant source of monotonic anonymous-memory growth
# (see https://github.com/LMCache/LMCache/issues/3767).
#
# Caching by length is safe: the `from_address(addr)` instance never owns the
# underlying buffer, only the metadata (length, item-type), and that metadata
# depends solely on `N`. ``functools.cache`` provides a thread-safe unbounded
# memoization primitive, so concurrent first-time accesses for the same `N`
# cannot race to create distinct heap types.
@cache
def _get_cached_ubyte_array_type(num_bytes: int) -> type[ctypes.Array[ctypes.c_ubyte]]:
    """Return a cached ``ctypes.c_ubyte * num_bytes`` array type.

    Args:
        num_bytes: The length of the array type in bytes.

    Returns:
        The cached ``ctypes.Array`` subclass for the given length. Subsequent
        calls with the same ``num_bytes`` return the same type object.
    """
    return ctypes.c_ubyte * num_bytes


# Helper functions for thread safety
def synchronized(lock_attr_name):
    """
    Decorator to make a method thread-safe by acquiring the lock
    specified by lock_attr_name on the instance.
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_attr_name)
            with lock:
                return method(self, *args, **kwargs)

        return wrapper

    return decorator


class MemoryFormat(Enum):
    UNDEFINED = 0
    """[2, num_layers, num_tokens, hidden_dim]
    """
    # KV_BLOB = 1
    KV_2LTD = auto()
    """[num_tokens, 2, hidden_dim]
    """
    # LAYER_KV_BLOB = 2
    KV_T2D = auto()
    """[2, num_tokens, hidden_dim]
    """

    KV_2TD = auto()
    """Compressed binary array format
    """
    BINARY = auto()

    BINARY_BUFFER = auto()

    KV_MLA_FMT = auto()
    """[1, num_layers, num_tokens, aligned_head_size]
    """

    # This is for the encoder cache (EC) tensor format
    EC_TD = auto()
    """[num_tokens, hidden_dim]
    """

    # Hidden-state store (HS) tensor format. Same logical shape as EC_TD
    # ([num_tokens, hidden_dim]) but tagged separately so the allocator and
    # any future mp/serialization paths can distinguish encoder-cache entries
    # from hidden-state entries.
    HS_TD = auto()
    """[num_tokens, hidden_dim]
    """

    def token_dim(self) -> int:
        if self == MemoryFormat.KV_2LTD:
            return 2
        elif self == MemoryFormat.KV_T2D:
            return 1
        elif self == MemoryFormat.KV_2TD:
            return 0
        elif self == MemoryFormat.BINARY:
            return 0
        elif self == MemoryFormat.BINARY_BUFFER:
            return 0
        elif self == MemoryFormat.KV_MLA_FMT:
            return 2
        elif self == MemoryFormat.EC_TD:
            return 0
        elif self == MemoryFormat.HS_TD:
            return 0
        return 0


@dataclass
class FreeBlock:
    """Metadata class used by the memory allocators"""

    start: int
    size: int

    def can_be_coalesced(self, succ: "FreeBlock") -> bool:
        return self.start + self.size == succ.start


@dataclass
class MemoryObjMetadata:
    # TODO(chunxiaozheng): use shapes and dtypes to replace shape and dtype
    # The 'logical' shape of the tensor
    shape: torch.Size

    # The 'logical' dtype of the tensor
    dtype: Optional[torch.dtype]

    # The 'physical address' of the tensor
    address: int

    # The 'physical size' in bytes of the allocated memory
    phy_size: int

    # Reference count
    ref_count: int

    # Whether the object is pinned and cannot be evicted
    # lookup pins are temporary
    # cache controller pins are persistent
    pin_count: int = 0

    # The 'logical' format of the tensor
    fmt: MemoryFormat = MemoryFormat.UNDEFINED

    # Positions when the cache is stored
    cached_positions: Optional[torch.Tensor] = None

    # shapes and dtypes should be used in the future
    shapes: Optional[list[torch.Size]] = None
    dtypes: Optional[list[torch.dtype]] = None

    def to_dict(self):
        # Note(Kuntai): this is used for serializing MemoryObjMetadata via
        # msgpack.
        return {
            "__type__": "MemoryObjMetadata",
            "shape": list(self.shape),  # torch.Size -> list
            "dtype": str(self.dtype) if self.dtype else None,
            "address": self.address,
            "phy_size": self.phy_size,
            "ref_count": self.ref_count,
            "fmt": self.fmt.value,
            "shapes": [list(shape) for shape in self.shapes] if self.shapes else None,
            "dtypes": [str(dtype) for dtype in self.dtypes] if self.dtypes else None,
        }

    @staticmethod
    def from_dict(d):
        dtype_str = d["dtype"]
        dtype = getattr(torch, dtype_str.replace("torch.", "")) if dtype_str else None
        shapes_list = d["shapes"]
        shapes = [torch.Size(s) for s in shapes_list] if shapes_list else None
        dtypes_list = d["dtypes"]
        dtypes = (
            [getattr(torch, d_str.replace("torch.", "")) for d_str in dtypes_list]
            if dtypes_list
            else None
        )
        return MemoryObjMetadata(
            shape=torch.Size(d["shape"]),
            dtype=dtype,
            address=d["address"],
            phy_size=d["phy_size"],
            ref_count=d["ref_count"],
            fmt=MemoryFormat(d["fmt"]),
            shapes=shapes,
            dtypes=dtypes,
        )

    def get_size(self) -> int:
        if self.shapes is not None and self.dtypes is not None:
            return get_size_bytes(self.shapes, self.dtypes)
        return self.shape.numel() * self.dtype.itemsize  # type: ignore


@dataclass(frozen=True)
class RefCountUpReceipt:
    """Identity of one exactly-once reference acquisition."""

    allocation_epoch: int
    allocation_token: object
    token: object


@dataclass(frozen=True)
class RefCountDownReceipt:
    """Identity of one already-counted reference release.

    ``allocation_epoch`` prevents a delayed cleanup owner from consuming a
    reference after a page-backed ``TensorMemoryObj`` has been recycled for a
    later allocation. ``allocation_token`` is the stable object identity, and
    ``token`` is committed in the same immutable state assignment as the
    authoritative reference-count decrement.
    """

    allocation_epoch: int
    allocation_token: object
    token: object


@dataclass(frozen=True)
class _TensorRefState:
    """Authoritative, atomically replaced TensorMemoryObj reference state."""

    count: int
    allocation_epoch: int
    allocation_token: object
    up_applied_tokens: frozenset[object]
    up_completed_tokens: frozenset[object]
    applied_tokens: frozenset[object]
    completed_tokens: frozenset[object]


@dataclass(frozen=True)
class _AllocationReceiptState:
    """Atomically replaced ownership state for one address reservation."""

    manager: Optional[Any]
    memory_obj: Optional[Any]
    ref_count_down_receipt: Optional[RefCountDownReceipt]
    address_release_started: bool
    address_release_complete: bool


class AllocationReceipt:
    """Pre-published owner for one exact address allocation.

    A caller creates and retains this receipt *before* entering an allocator.
    ``AddressManager.allocate_once`` binds it before changing free space, and
    the tensor allocator adopts the constructed object with one immutable
    state replacement before returning it. Cleanup can therefore recover
    either the raw reservation or the object's initial reference, including
    when an asynchronous exception lands between those stages.
    """

    def __init__(self) -> None:
        self.token = object()
        self._lock = threading.RLock()
        self._state = _AllocationReceiptState(
            manager=None,
            memory_obj=None,
            ref_count_down_receipt=None,
            address_release_started=False,
            address_release_complete=False,
        )

    @property
    def adopted(self) -> bool:
        with self._lock:
            return self._state.memory_obj is not None

    @property
    def memory_obj(self) -> Optional[Any]:
        with self._lock:
            return self._state.memory_obj

    @property
    def ref_count_down_receipt(self) -> Optional[RefCountDownReceipt]:
        with self._lock:
            return self._state.ref_count_down_receipt

    def _bind_address_manager(self, manager: Any) -> None:
        """Publish the manager binding before it can reserve address space."""
        with self._lock:
            state = self._state
            if state.address_release_started:
                raise RuntimeError("allocation receipt has already been released")
            if state.manager is not None:
                if state.manager is not manager:
                    raise ValueError("allocation receipt belongs to another manager")
                return
            self._state = _AllocationReceiptState(
                manager=manager,
                memory_obj=state.memory_obj,
                ref_count_down_receipt=state.ref_count_down_receipt,
                address_release_started=False,
                address_release_complete=False,
            )

    def _allocation_may_commit(self, manager: Any) -> bool:
        """Recheck cancellation while AddressManager holds its state lock."""
        with self._lock:
            state = self._state
            return state.manager is manager and not state.address_release_started

    def adopt(self, memory_obj: Any) -> RefCountDownReceipt:
        """Atomically replace raw-reservation ownership with object ownership."""
        # Constructing the ref receipt has no ownership side effect. If an
        # interruption lands here, the outer owner still sees the raw address
        # reservation and can release it by this AllocationReceipt.
        ref_receipt = memory_obj.make_ref_count_down_receipt()
        with self._lock:
            state = self._state
            if state.address_release_started:
                raise RuntimeError("cannot adopt a released allocation receipt")
            if state.memory_obj is not None:
                if state.memory_obj is not memory_obj:
                    raise ValueError(
                        "allocation receipt already adopted another object"
                    )
                assert state.ref_count_down_receipt is not None
                return state.ref_count_down_receipt
            self._state = _AllocationReceiptState(
                manager=state.manager,
                memory_obj=memory_obj,
                ref_count_down_receipt=ref_receipt,
                address_release_started=False,
                address_release_complete=False,
            )
            return ref_receipt

    def _begin_address_release(self, manager: Any) -> bool:
        """Publish release intent before the manager changes free space."""
        with self._lock:
            state = self._state
            if state.manager is not None and state.manager is not manager:
                raise ValueError("allocation receipt belongs to another manager")
            if state.address_release_complete:
                return False
            self._state = _AllocationReceiptState(
                manager=manager if state.manager is None else state.manager,
                memory_obj=state.memory_obj,
                ref_count_down_receipt=state.ref_count_down_receipt,
                address_release_started=True,
                address_release_complete=False,
            )
            return True

    def _finish_address_release(self, manager: Any) -> None:
        with self._lock:
            state = self._state
            if state.manager is not manager:
                raise ValueError("allocation receipt belongs to another manager")
            self._state = _AllocationReceiptState(
                manager=state.manager,
                memory_obj=state.memory_obj,
                ref_count_down_receipt=state.ref_count_down_receipt,
                address_release_started=True,
                address_release_complete=True,
            )

    def release_once(self) -> None:
        """Release the owned object reference or an unadopted reservation."""
        with self._lock:
            state = self._state
            memory_obj = state.memory_obj
            ref_receipt = state.ref_count_down_receipt
            manager = state.manager
        if memory_obj is not None:
            assert ref_receipt is not None
            memory_obj.ref_count_down_once(ref_receipt)
            return
        if manager is None:
            # The allocation call never reached AddressManager. Publish a
            # terminal no-reservation state so repeated drains are exact.
            with self._lock:
                state = self._state
                if state.memory_obj is not None:
                    memory_obj = state.memory_obj
                    ref_receipt = state.ref_count_down_receipt
                else:
                    self._state = _AllocationReceiptState(
                        manager=None,
                        memory_obj=None,
                        ref_count_down_receipt=None,
                        address_release_started=True,
                        address_release_complete=True,
                    )
                    return
            assert memory_obj is not None and ref_receipt is not None
            memory_obj.ref_count_down_once(ref_receipt)
            return
        manager.free_once(self)

    def release_complete(self) -> bool:
        """Return whether this owner's exact release has completed."""
        with self._lock:
            state = self._state
            memory_obj = state.memory_obj
            ref_receipt = state.ref_count_down_receipt
            address_complete = state.address_release_complete
        if memory_obj is None:
            return address_complete
        assert ref_receipt is not None
        return memory_obj.ref_count_down_receipt_complete(ref_receipt)


class MemoryObj(metaclass=abc.ABCMeta):
    """
    MemoryObj interface.
    """

    # subclasses should expose raw_data differently
    raw_data: Any

    def __init__(self, metadata: MemoryObjMetadata):
        self.meta = metadata

    def make_ref_count_up_receipt(self) -> RefCountUpReceipt:
        """Create an exact-once acquisition token when supported."""
        raise NotImplementedError(
            f"{type(self).__name__} has no exact ref-count acquisition receipt"
        )

    def ref_count_up_once(self, receipt: RefCountUpReceipt) -> None:
        """Acquire the reference named by ``receipt`` exactly once."""
        del receipt
        raise NotImplementedError(
            f"{type(self).__name__} has no exact ref-count acquisition receipt"
        )

    def ref_count_up_receipt_complete(self, receipt: RefCountUpReceipt) -> bool:
        """Return whether an exact acquisition completed."""
        del receipt
        return False

    def ref_count_up_receipt_applied(self, receipt: RefCountUpReceipt) -> bool:
        """Return whether the acquisition changed authoritative state."""
        del receipt
        return False

    def make_ref_count_down_receipt(self) -> RefCountDownReceipt:
        """Create an exact-once release token when the object supports it.

        Concrete runtime objects override this method.  The base error keeps
        unsupported memory implementations fail-closed instead of silently
        claiming an exact cleanup receipt.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no exact ref-count release receipt"
        )

    def ref_count_down_once(self, receipt: RefCountDownReceipt) -> None:
        """Consume the reference named by ``receipt`` exactly once."""
        del receipt
        raise NotImplementedError(
            f"{type(self).__name__} has no exact ref-count release receipt"
        )

    def ref_count_down_receipt_complete(
        self,
        receipt: RefCountDownReceipt,
    ) -> bool:
        """Return whether ``receipt`` completed for this allocation."""
        del receipt
        return False

    @abc.abstractmethod
    def invalidate(self):
        """
        Invalidate the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_valid(self):
        """
        Check if the MemoryObj is valid.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_size(self) -> int:
        """
        Get the size of the MemoryObj in bytes.
        Note that this number could be smaller than the physical size.
        The physical size is aligned to the allocator's alignment.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self) -> torch.Size:
        """
        Get the shape of the MemoryObj.
        """
        raise NotImplementedError

    def get_dtype(self) -> Optional[torch.dtype]:
        """
        Get the dtype of the MemoryObj.
        """
        return None

    @abc.abstractmethod
    def get_shapes(self) -> list[torch.Size]:
        """
        Get the shapes of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_dtypes(self) -> list[torch.dtype]:
        """
        Get the dtypes of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_memory_format(self) -> MemoryFormat:
        """
        Get the memory format of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_physical_size(self) -> int:
        """
        Get the physical size of the MemoryObj in bytes.
        """
        raise NotImplementedError

    def set_used_size(self, n: int) -> None:  # noqa: B027
        """Narrow this buffer's logical size to the first ``n`` bytes.

        Optional hook for callers that have just written ``n`` bytes
        into a buffer originally allocated with an upper-bound size
        (e.g. the async serde processor, where the destination is sized
        from ``estimate_serialized_size`` but ``serialize`` returns the
        actual ``n`` it wrote). After this call, ``get_size()`` /
        ``byte_array`` / any downstream L2 adapter that reads the
        logical size will see exactly ``n`` bytes.

        Default is a no-op so subclasses without a "used vs allocated"
        distinction (e.g. :class:`BytesBufferMemoryObj`, where the raw
        bytes already are the actual contents) keep working unchanged.

        Args:
            n: bytes actually used in this buffer. Subclasses that
                implement this must validate ``n`` and raise
                ``ValueError`` on out-of-range or unsupported layouts.
        """
        pass

    @abc.abstractmethod
    def pin(self) -> bool:
        """
        Pin the memory obj so that it will not be evicted.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_up(self):
        """
        Increase ref count for the given MemoryObj by one.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unpin(self) -> bool:
        """
        Unpin the memory obj so that it can be evicted.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_down(self):
        """
        Decrease ref count for the given MemoryObj by one.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_ref_count(self) -> int:
        """
        Get ref count for the given MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_tokens(self) -> int:
        """
        Get token number for the given MemoryObj.
        """
        raise NotImplementedError

    @property
    def shm_offset(self) -> int:
        """Return the byte offset of this object inside the SHM pool."""
        return self.meta.address

    @property
    def shm_byte_length(self) -> int:
        """Return the byte length of this object inside the SHM pool."""
        return self.get_size()

    @property
    @abc.abstractmethod
    def metadata(self) -> MemoryObjMetadata:
        """
        Get the metada of the MemoryObj.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tensor(self) -> Optional[torch.Tensor]:
        """
        Get the tensor from the MemoryObj.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def byte_array(self) -> bytes:
        """
        Get the byte array from the MemoryObj.
        The size is will be the physical size instead of the unaligned size.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data_ptr(self) -> int:
        """
        Get the data pointer of the MemoryObj.
        This is used to access the raw data in the memory.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_pinned(self) -> bool:
        """
        Check whether the memory obj is pinned.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def raw_tensor(self) -> Optional[torch.Tensor]:
        """
        Get the raw tensor from the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        """
        Get the tensor from the MemoryObj at the given index(group).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        """
        Get the allocator that allocates this memory object
        """
        raise NotImplementedError


@dataclass
class PinnedAllocFree:
    """Resolved alloc/free function pair for pinned CPU memory."""

    alloc_fn: Any
    alloc_args: tuple
    free_fn: Any
    free_args: tuple

    def alloc(self) -> int:
        """Allocate pinned memory and return the raw pointer."""
        return self.alloc_fn(*self.alloc_args)

    def free(self, ptr: int) -> None:
        """Free a previously allocated pinned-memory pointer."""
        self.free_fn(ptr, *self.free_args)


@dataclass
class _CPUMemoryAllocation:
    """One exact native CPU allocation retained by a public receipt."""

    pointer: Optional[int]
    release_call: RetainedCall
    buffer: Optional[torch.Tensor] = None


class CPUMemoryAllocationReceipt:
    """Retain a native pinned arena across its Tensor return handoff.

    ``_allocate_cpu_memory`` normally returns a Tensor wrapping memory whose
    lifetime is controlled by an LMCache native free routine.  The Tensor does
    not own that free operation.  A caller that publishes this receipt before
    invoking the helper therefore remains able to release the exact native
    allocation even if an interruption lands after the helper returns but
    before the Tensor reaches the caller's normal attribute.
    """

    def __init__(self) -> None:
        self._allocation: Optional[_CPUMemoryAllocation] = None
        self._no_allocation = False
        self._allocation_call: Optional[RetainedCall] = None
        self._lock = threading.RLock()

    @property
    def outcome_known(self) -> bool:
        """Whether allocation either escaped into this receipt or unwound."""
        with self._lock:
            return self._allocation is not None or self._no_allocation

    @property
    def buffer(self) -> Optional[torch.Tensor]:
        """Return the exact Tensor wrapper, once one has been published."""
        with self._lock:
            allocation = self._allocation
            return allocation.buffer if allocation is not None else None

    @property
    def released(self) -> bool:
        """Whether this receipt proves that no native allocation remains."""
        with self._lock:
            allocation = self._allocation
            if allocation is None:
                return self._no_allocation
            return allocation.release_call.complete

    def run_allocation(self, call: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Run one native allocation attempt on a retained worker.

        Native allocation and receipt publication happen on the worker.  A
        caller-side asynchronous exception can therefore only interrupt this
        method's wait; the same attempt remains reachable for terminal drain.
        """
        with self._lock:
            attempt = self._allocation_call
            if attempt is None:
                if self._allocation is not None or self._no_allocation:
                    raise RuntimeError("CPU allocation receipt already has an outcome")

                def run_once() -> None:
                    call()

                attempt = RetainedCall(
                    run_once,
                    thread_name="lmcache-pinned-cpu-allocation",
                )
                # Publish the whole retained attempt before it can start.
                self._allocation_call = attempt
        attempt.run()
        buffer = self.buffer
        if buffer is None:
            raise RuntimeError("CPU allocation completed without a Tensor")
        return buffer

    def finish_allocation_for_close(self) -> None:
        """Finish an interrupted allocation attempt before terminal release."""
        with self._lock:
            attempt = self._allocation_call
            if attempt is None:
                # No retained worker could have acquired memory.  This also
                # covers caller interruption before publication of the local
                # RetainedCall in ``run_allocation``.
                if self._allocation is None:
                    self._no_allocation = True
                return
        try:
            attempt.run()
        except BaseException as error:
            # The worker's construction failure is context, not a terminal
            # cleanup failure.  A newly interrupted waiter is distinct and
            # must leave close retryable.
            if attempt.last_error is not error:
                raise

    def retain_raw(
        self,
        resolved: PinnedAllocFree,
        pointer: int,
    ) -> _CPUMemoryAllocation:
        """Publish exact native free authority before building its Tensor."""
        allocation = _CPUMemoryAllocation(
            pointer=pointer,
            release_call=RetainedCall(
                lambda: resolved.free(pointer),
                thread_name="lmcache-pinned-cpu-free",
            ),
        )
        with self._lock:
            if self._no_allocation or self._allocation is not None:
                raise RuntimeError("CPU allocation receipt already has an outcome")
            # One pointer assignment publishes the exact pointer, its free
            # authority, and the retained free attempt together.
            self._allocation = allocation
        return allocation

    def retain_buffer(
        self,
        buffer: torch.Tensor,
        release_call: Callable[[], None],
    ) -> None:
        """Adopt a returned buffer from a compatibility allocation helper."""
        allocation = _CPUMemoryAllocation(
            pointer=buffer.data_ptr() if buffer.numel() != 0 else None,
            release_call=RetainedCall(
                release_call,
                thread_name="lmcache-pinned-cpu-compat-free",
            ),
            buffer=buffer,
        )
        with self._lock:
            if self._no_allocation or self._allocation is not None:
                raise RuntimeError("CPU allocation receipt already has an outcome")
            self._allocation = allocation

    def publish_buffer(
        self,
        allocation: _CPUMemoryAllocation,
        buffer: torch.Tensor,
    ) -> None:
        """Attach the Tensor wrapper to the exact retained raw allocation."""
        with self._lock:
            if self._allocation is not allocation or allocation.release_call.complete:
                raise RuntimeError("CPU allocation receipt changed before publication")
            allocation.buffer = buffer

    def owns_pointer(self, pointer: int) -> bool:
        """Return whether this receipt owns the supplied native pointer."""
        with self._lock:
            allocation = self._allocation
            return (
                allocation is not None
                and allocation.pointer == pointer
                and not allocation.release_call.complete
            )

    def mark_no_allocation(self) -> None:
        """Publish proof that a failed attempt acquired no native allocation."""
        with self._lock:
            if self._allocation is not None:
                raise RuntimeError("cannot discard a retained CPU allocation")
            self._no_allocation = True

    def release(self) -> None:
        """Release the exact retained allocation, preserving failure state."""
        with self._lock:
            allocation = self._allocation
            no_allocation = self._no_allocation
        if allocation is None:
            if not no_allocation:
                raise RuntimeError("CPU allocation outcome remains unresolved")
            return
        # The native side effect and its success receipt belong to a retained
        # worker.  Caller interruption resumes this same attempt; a native
        # free error is deliberately sticky because replay is not provably
        # safe after an arbitrary allocator failure.
        allocation.release_call.run()


def _resolve_pinned_alloc_free(
    numa_mapping: Optional[NUMAMapping] = None,
    shm_name: Optional[str] = None,
    size: Optional[int] = None,
    use_hugepages: bool = False,
) -> PinnedAllocFree:
    """Resolve the alloc/free function pair based on memory type.

    Returns:
        A PinnedAllocFree with the resolved functions and their extra
        arguments.  Call ``ptr = resolved.alloc()`` and ``resolved.free(ptr)``.
    """
    if shm_name:
        if use_hugepages:
            raise ValueError("Hugepages are not supported with shared memory (shm)")
        return PinnedAllocFree(
            alloc_fn=lmc_ops.alloc_shm_pinned_ptr,
            alloc_args=(size, shm_name),
            free_fn=lmc_ops.free_shm_pinned_ptr,
            free_args=(size, shm_name),
        )
    elif numa_mapping:
        if torch_dev.is_available():
            current_device_id = torch_dev.current_device()
        else:
            current_device_id = 0
        gpu_to_numa_mapping = numa_mapping.gpu_to_numa_mapping
        assert current_device_id in gpu_to_numa_mapping, (
            f"Current device {current_device_id} is not in the GPU NUMA mapping."
        )
        numa_id = gpu_to_numa_mapping[current_device_id]
        if use_hugepages:
            return PinnedAllocFree(
                alloc_fn=lmc_ops.alloc_hugepage_pinned_numa_ptr,
                alloc_args=(size, numa_id),
                free_fn=lmc_ops.free_hugepage_pinned_numa_ptr,
                free_args=(size,),
            )
        else:
            return PinnedAllocFree(
                alloc_fn=lmc_ops.alloc_pinned_numa_ptr,
                alloc_args=(size, numa_id),
                free_fn=lmc_ops.free_pinned_numa_ptr,
                free_args=(size,),
            )
    else:
        flags = 0
        if use_hugepages:
            return PinnedAllocFree(
                alloc_fn=lmc_ops.alloc_hugepage_pinned_ptr,
                alloc_args=(size, flags),
                free_fn=lmc_ops.free_hugepage_pinned_ptr,
                free_args=(size,),
            )
        else:
            return PinnedAllocFree(
                alloc_fn=lmc_ops.alloc_pinned_ptr,
                alloc_args=(size, flags),
                free_fn=lmc_ops.free_pinned_ptr,
                free_args=(),
            )


def _read_hugepage_info() -> Optional[Tuple[int, int, int]]:
    """Read hugepage pool stats from sysfs.

    NOTE: We only use 2 MiB hugepages, so the pool stats are taken from
    the 2 MiB pool directly rather than the system default pool reported in
    ``/proc/meminfo`` (which can be 1 GiB on some hosts).

    Returns:
        ``(nr_hugepages, free_hugepages, page_size_mb)`` for the hugepage
        pool, or ``None`` if the sysfs entries are unavailable.
    """
    base = "/sys/kernel/mm/hugepages/hugepages-2048kB"
    try:
        with open(f"{base}/nr_hugepages") as f:
            total = int(f.read().strip())
        with open(f"{base}/free_hugepages") as f:
            free = int(f.read().strip())
        return total, free, 2
    except (OSError, ValueError):
        return None


def _allocate_cpu_memory(
    size: int,
    numa_mapping: Optional[NUMAMapping] = None,
    shm_name: Optional[str] = None,
    use_hugepages: bool = False,
    ownership_receipt: Optional[CPUMemoryAllocationReceipt] = None,
) -> torch.Tensor:
    if ownership_receipt is not None:
        return ownership_receipt.run_allocation(
            lambda: _allocate_cpu_memory_once(
                size,
                numa_mapping,
                shm_name,
                use_hugepages,
                ownership_receipt,
            )
        )
    return _allocate_cpu_memory_once(
        size,
        numa_mapping,
        shm_name,
        use_hugepages,
        None,
    )


def _allocate_cpu_memory_once(
    size: int,
    numa_mapping: Optional[NUMAMapping],
    shm_name: Optional[str],
    use_hugepages: bool,
    ownership_receipt: Optional[CPUMemoryAllocationReceipt],
) -> torch.Tensor:
    """Allocate and wrap one arena on the current (possibly retained) worker."""
    if size == 0:
        buffer = torch.empty(0, dtype=torch.uint8)
        if ownership_receipt is not None:
            ownership_receipt.retain_buffer(buffer, lambda: None)
        return buffer

    try:
        resolved = _resolve_pinned_alloc_free(
            numa_mapping,
            shm_name,
            size,
            use_hugepages,
        )
    except BaseException:
        if ownership_receipt is not None:
            ownership_receipt.mark_no_allocation()
        raise

    ptr: Optional[int] = None
    allocation: Optional[_CPUMemoryAllocation] = None
    try:
        # Allocation and every ownership-publication step share one rollback
        # region.  In particular, a traced BaseException on the first line
        # after ``alloc`` observes ``ptr`` and cannot strand the native arena.
        ptr = resolved.alloc()
        if ownership_receipt is not None:
            allocation = ownership_receipt.retain_raw(resolved, ptr)
        array_type = ctypes.c_uint8 * size
        buf = array_type.from_address(ptr)
        buffer = torch.frombuffer(buf, dtype=torch.uint8)
        if ownership_receipt is not None:
            assert allocation is not None
            ownership_receipt.publish_buffer(allocation, buffer)
    except BaseException as construction_error:
        if ptr is None:
            if ownership_receipt is not None:
                ownership_receipt.mark_no_allocation()
            if (
                isinstance(construction_error, RuntimeError)
                and use_hugepages
                and "mmap failed" in str(construction_error)
            ):
                diag = _read_hugepage_info()
                if diag is not None:
                    total, free, page_mb = diag
                    page_bytes = page_mb * 1024 * 1024
                    needed = (size + page_bytes - 1) // page_bytes
                    logger.error(
                        "Failed to allocate huge pages. "
                        "Pool has %d pages (%d free, each %d MiB). "
                        "Requested %d bytes (%d pages). "
                        "Please grow the %d MiB hugepage pool.",
                        total,
                        free,
                        page_mb,
                        size,
                        needed,
                        page_mb,
                    )
                else:
                    logger.error(
                        "Failed to allocate huge pages. "
                        "Please grow the 2 MiB hugepage pool."
                    )
            raise

        # The raw pointer either remains local or has already moved into the
        # caller-published receipt.  Reclaim through exactly one authority and
        # retain a failed free in the receipt for outer terminal retry.
        try:
            if ownership_receipt is not None and ownership_receipt.owns_pointer(ptr):
                ownership_receipt.release()
            else:
                resolved.free(ptr)
                if ownership_receipt is not None:
                    ownership_receipt.mark_no_allocation()
        except BaseException as cleanup_error:
            raise construction_error from cleanup_error
        raise

    return buffer


def _free_cpu_memory(
    buffer: torch.Tensor,
    size: int | None = None,
    numa_mapping: Optional[NUMAMapping] = None,
    shm_name: Optional[str] = None,
    use_hugepages: bool = False,
) -> None:
    if torch_dev.is_available():
        torch_dev.synchronize()

    resolved = _resolve_pinned_alloc_free(
        numa_mapping,
        shm_name,
        size,
        use_hugepages,
    )
    resolved.free(buffer.data_ptr())


def _allocate_gpu_memory(
    size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    page_size = os.sysconf("SC_PAGESIZE")

    # Over-allocate
    base_buffer = torch.empty(size + page_size, dtype=torch.uint8, device=device)
    offset = -base_buffer.data_ptr() % page_size

    # Make aligned view
    aligned_buffer = base_buffer[offset : offset + size]

    # Need to return the base buffer as well in order to prevent GC
    return base_buffer, aligned_buffer


class TensorMemoryObj(MemoryObj):
    """
    Wraps a raw flat tensor with some metadata
    """

    monitor = LMCStatsMonitor.GetOrCreate()

    def __init__(
        self,
        raw_data: torch.Tensor,
        metadata: MemoryObjMetadata,
        parent_allocator: Optional["MemoryAllocatorInterface"],
        allocation_receipt: Optional[AllocationReceipt] = None,
    ):
        assert metadata.dtype is not None, "dtype must be specified for TensorMemoryObj"
        super().__init__(metadata)
        self.raw_data = raw_data
        self.valid = True
        # Receipt operations need owner-aware recursion depth so one
        # caller-side asynchronous interruption can restore precisely the
        # ownership level acquired by that operation.
        self.lock = threading.RLock()
        self.parent_allocator = parent_allocator
        self.allocation_receipt = allocation_receipt
        self._ref_state = _TensorRefState(
            count=metadata.ref_count,
            allocation_epoch=0,
            allocation_token=object(),
            up_applied_tokens=frozenset(),
            up_completed_tokens=frozenset(),
            applied_tokens=frozenset(),
            completed_tokens=frozenset(),
        )
        # ``None`` means "use the layout-derived size from
        # group_prefix_sum"; a non-None value narrows the logical view to
        # exactly that many bytes (see set_used_size).  Allocator reuse
        # paths must reset this to None along with the rest of the
        # per-allocation metadata.
        self._used_size_override: Optional[int] = None
        # Calculate the prefix sum of the group sizes
        # If there are two groups, the prefix sum will be
        # [0, size_of_group_1, size_of_group_1 + size_of_group_2]
        self.group_prefix_sum = [0]
        if self.meta.shapes is not None and self.meta.dtypes is not None:
            size_in_bytes = 0
            for shape, dtype in zip(self.meta.shapes, self.meta.dtypes, strict=True):
                size_in_bytes += shape.numel() * dtype.itemsize
                self.group_prefix_sum.append(size_in_bytes)
        else:
            self.group_prefix_sum.append(self.meta.get_size())

    def __del__(self):
        """
        Destructor to ensure memory is released when the object is garbage collected.
        This acts as a safety net to prevent memory leaks if ref_count_down() is not
        called properly somewhere in the code path.
        """
        if self.parent_allocator is not None and self.is_valid():
            ref_count = self._ref_state.count
            if ref_count > 0 or self.meta.pin_count > 0:
                logger.warning(
                    "MemoryObj at %s is being garbage collected "
                    "with ref_count=%d, pin_count=%d. "
                    "This indicates ref_count_down()/unpin() was not called properly.",
                    self.meta.address,
                    ref_count,
                    self.meta.pin_count,
                )
            self.parent_allocator.free(self)

    def invalidate(self):
        self.valid = False

    def is_valid(self):
        return self.valid

    def get_size(self) -> int:
        if self._used_size_override is not None:
            return self._used_size_override
        return self.group_prefix_sum[-1]

    def set_used_size(self, n: int) -> None:
        """Narrow the logical size to ``n`` bytes after a write.

        After this call, ``get_size()`` returns ``n`` and ``byte_array``
        exposes exactly ``n`` bytes from the start of ``raw_data``.  The
        physical allocation (``get_physical_size``) and ``raw_data``
        buffer are unchanged.  Allocator reuse resets this override to
        ``None`` so a recycled block returns to its layout-derived size.

        Note: the ``tensor`` property still derives its shape from
        ``meta.shape``, so accessing ``.tensor`` on a buffer narrowed
        below its layout size will fail to reshape.  Use ``byte_array``
        (or read ``raw_data[: get_size()]`` directly) for downstream
        I/O that must honor the narrowed size.

        Args:
            n: bytes actually written.  Must satisfy
                ``0 <= n <= get_physical_size()``.

        Raises:
            ValueError: if ``n`` is outside the allowed range.
        """
        if n < 0 or n > self.meta.phy_size:
            raise ValueError(
                f"set_used_size: n={n} out of range [0, {self.meta.phy_size}]"
            )
        with self.lock:
            self._used_size_override = n

    # TODO(chunxiaozheng): use get_shapes and get_dtypes to replace
    #  get_shape and get_dtype
    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> torch.dtype:
        assert self.meta.dtype is not None
        return self.meta.dtype

    def get_shapes(self) -> list[torch.Size]:
        assert self.meta.shapes is not None
        return self.meta.shapes

    def get_dtypes(self) -> list[torch.dtype]:
        assert self.meta.dtypes is not None
        return self.meta.dtypes

    def get_memory_format(self) -> MemoryFormat:
        with self.lock:
            return self.meta.fmt

    def get_physical_size(self) -> int:
        return self.meta.phy_size

    def ref_count_up(self):
        with self.lock:
            state = self._ref_state
            next_state = _TensorRefState(
                count=state.count + 1,
                allocation_epoch=state.allocation_epoch,
                allocation_token=state.allocation_token,
                up_applied_tokens=state.up_applied_tokens,
                up_completed_tokens=state.up_completed_tokens,
                applied_tokens=state.applied_tokens,
                completed_tokens=state.completed_tokens,
            )
            self._ref_state = next_state
            self.meta.ref_count = next_state.count

    def ref_count_down(self):
        with self.lock:
            state = self._ref_state
            next_count = state.count - 1
            if next_count < 0:
                logger.warning(
                    f"Ref count of MemoryObj {self.meta.address}"
                    f"is negative: {next_count}."
                    "Double free occurred somewhere."
                    "Setting ref count back to 0 as a hack but please find the bug."
                )
                next_count = 0
            next_state = _TensorRefState(
                count=next_count,
                allocation_epoch=state.allocation_epoch,
                allocation_token=state.allocation_token,
                up_applied_tokens=state.up_applied_tokens,
                up_completed_tokens=state.up_completed_tokens,
                applied_tokens=state.applied_tokens,
                completed_tokens=state.completed_tokens,
            )
            self._ref_state = next_state
            self.meta.ref_count = next_state.count
            if (
                next_state.count == 0
                and self.parent_allocator is not None
                and self.meta.pin_count == 0
            ):
                self.parent_allocator.free(self)

    def get_ref_count(self) -> int:
        return _run_under_interrupt_safe_rlock(
            self.lock,
            lambda: self._ref_state.count,
        )

    def make_ref_count_up_receipt(self) -> RefCountUpReceipt:
        """Name one exact acquisition in the current allocation epoch."""
        return _run_under_interrupt_safe_rlock(
            self.lock,
            self._make_ref_count_up_receipt_locked,
        )

    def _make_ref_count_up_receipt_locked(self) -> RefCountUpReceipt:
        return RefCountUpReceipt(
            allocation_epoch=self._ref_state.allocation_epoch,
            allocation_token=self._ref_state.allocation_token,
            token=object(),
        )

    def _validate_ref_count_up_receipt(
        self,
        receipt: RefCountUpReceipt,
    ) -> _TensorRefState:
        state = self._ref_state
        if receipt.allocation_token is not state.allocation_token:
            raise ValueError("ref-count receipt belongs to another memory object")
        if receipt.allocation_epoch > state.allocation_epoch:
            raise ValueError("ref-count receipt belongs to a future allocation")
        return state

    def ref_count_up_once(self, receipt: RefCountUpReceipt) -> None:
        """Atomically commit one receipt and its reference acquisition."""
        _run_under_interrupt_safe_rlock(
            self.lock,
            partial(self._ref_count_up_once_locked, receipt),
        )

    def _ref_count_up_once_locked(self, receipt: RefCountUpReceipt) -> None:
        state = self._validate_ref_count_up_receipt(receipt)
        if receipt.allocation_epoch < state.allocation_epoch:
            pass
        elif receipt.token in state.up_completed_tokens:
            self.meta.ref_count = state.count
        else:
            if receipt.token not in state.up_applied_tokens:
                state = _TensorRefState(
                    count=state.count + 1,
                    allocation_epoch=state.allocation_epoch,
                    allocation_token=state.allocation_token,
                    up_applied_tokens=state.up_applied_tokens | {receipt.token},
                    up_completed_tokens=state.up_completed_tokens,
                    applied_tokens=state.applied_tokens,
                    completed_tokens=state.completed_tokens,
                )
                self._ref_state = state
            self.meta.ref_count = state.count
            if receipt.token not in state.up_completed_tokens:
                state = _TensorRefState(
                    count=state.count,
                    allocation_epoch=state.allocation_epoch,
                    allocation_token=state.allocation_token,
                    up_applied_tokens=state.up_applied_tokens,
                    up_completed_tokens=state.up_completed_tokens | {receipt.token},
                    applied_tokens=state.applied_tokens,
                    completed_tokens=state.completed_tokens,
                )
                self._ref_state = state
                self.meta.ref_count = state.count

    def ref_count_up_receipt_complete(self, receipt: RefCountUpReceipt) -> bool:
        """Return exact acquisition completion, including after page reuse."""
        return _run_under_interrupt_safe_rlock(
            self.lock,
            partial(self._ref_count_up_receipt_complete_locked, receipt),
        )

    def _ref_count_up_receipt_complete_locked(
        self,
        receipt: RefCountUpReceipt,
    ) -> bool:
        state = self._validate_ref_count_up_receipt(receipt)
        return (
            receipt.allocation_epoch < state.allocation_epoch
            or receipt.token in state.up_completed_tokens
        )

    def ref_count_up_receipt_applied(self, receipt: RefCountUpReceipt) -> bool:
        """Return whether this acquisition incremented its allocation."""
        return _run_under_interrupt_safe_rlock(
            self.lock,
            partial(self._ref_count_up_receipt_applied_locked, receipt),
        )

    def _ref_count_up_receipt_applied_locked(
        self,
        receipt: RefCountUpReceipt,
    ) -> bool:
        state = self._validate_ref_count_up_receipt(receipt)
        return (
            receipt.allocation_epoch == state.allocation_epoch
            and receipt.token in state.up_applied_tokens
        )

    def make_ref_count_down_receipt(self) -> RefCountDownReceipt:
        """Name one reference in the current allocation epoch."""
        return _run_under_interrupt_safe_rlock(
            self.lock,
            self._make_ref_count_down_receipt_locked,
        )

    def _make_ref_count_down_receipt_locked(self) -> RefCountDownReceipt:
        return RefCountDownReceipt(
            allocation_epoch=self._ref_state.allocation_epoch,
            allocation_token=self._ref_state.allocation_token,
            token=object(),
        )

    def _validate_ref_count_down_receipt(
        self,
        receipt: RefCountDownReceipt,
    ) -> _TensorRefState:
        state = self._ref_state
        if receipt.allocation_token is not state.allocation_token:
            raise ValueError("ref-count receipt belongs to another memory object")
        if receipt.allocation_epoch > state.allocation_epoch:
            raise ValueError("ref-count receipt belongs to a future allocation")
        return state

    def ref_count_down_once(self, receipt: RefCountDownReceipt) -> None:
        """Atomically commit one receipt and its reference decrement.

        The first ``_ref_state`` pointer assignment contains both the new
        count and ``receipt.token``.  An interruption can therefore observe
        neither transition or both, never a decrement without ownership proof.
        Reclamation is retryable under the same token and completion is
        published only after the allocator returns.
        """
        _run_under_interrupt_safe_rlock(
            self.lock,
            partial(self._ref_count_down_once_locked, receipt),
        )

    def _ref_count_down_once_locked(self, receipt: RefCountDownReceipt) -> None:
        state = self._validate_ref_count_down_receipt(receipt)
        if receipt.allocation_epoch < state.allocation_epoch:
            pass
        elif receipt.token in state.completed_tokens:
            self.meta.ref_count = state.count
        else:
            if receipt.token not in state.applied_tokens:
                if state.count <= 0:
                    raise RuntimeError(
                        "cannot consume a ref-count receipt from a zero-count object"
                    )
                state = _TensorRefState(
                    count=state.count - 1,
                    allocation_epoch=state.allocation_epoch,
                    allocation_token=state.allocation_token,
                    up_applied_tokens=state.up_applied_tokens,
                    up_completed_tokens=state.up_completed_tokens,
                    applied_tokens=state.applied_tokens | {receipt.token},
                    completed_tokens=state.completed_tokens,
                )
                # Atomic ownership commit: count and receipt move together.
                self._ref_state = state

            # ``meta.ref_count`` is a compatibility mirror, never authority.
            self.meta.ref_count = state.count
            if (
                state.count == 0
                and self.parent_allocator is not None
                and self.meta.pin_count == 0
                and self.is_valid()
            ):
                self.parent_allocator.free(self)

            state = self._ref_state
            if (
                receipt.allocation_epoch >= state.allocation_epoch
                and receipt.token not in state.completed_tokens
            ):
                state = _TensorRefState(
                    count=state.count,
                    allocation_epoch=state.allocation_epoch,
                    allocation_token=state.allocation_token,
                    up_applied_tokens=state.up_applied_tokens,
                    up_completed_tokens=state.up_completed_tokens,
                    applied_tokens=state.applied_tokens,
                    completed_tokens=state.completed_tokens | {receipt.token},
                )
                self._ref_state = state
                self.meta.ref_count = state.count

    def ref_count_down_receipt_complete(
        self,
        receipt: RefCountDownReceipt,
    ) -> bool:
        """Return exact completion, including after page reuse."""
        return _run_under_interrupt_safe_rlock(
            self.lock,
            partial(self._ref_count_down_receipt_complete_locked, receipt),
        )

    def _ref_count_down_receipt_complete_locked(
        self,
        receipt: RefCountDownReceipt,
    ) -> bool:
        state = self._validate_ref_count_down_receipt(receipt)
        return (
            receipt.allocation_epoch < state.allocation_epoch
            or receipt.token in state.completed_tokens
        )

    def reset_ref_count_for_reuse(self, count: int = 1) -> None:
        """Begin a fresh page-backed allocation epoch."""
        if count < 0:
            raise ValueError("allocation ref count cannot be negative")
        with self.lock:
            self._reset_ref_count_for_reuse_locked(count)

    def _reset_ref_count_for_reuse_locked(self, count: int = 1) -> None:
        """Reset a page while the caller holds ``self.lock``."""
        state = self._ref_state
        next_state = _TensorRefState(
            count=count,
            allocation_epoch=state.allocation_epoch + 1,
            # Object identity is stable across page reuse. The epoch names the
            # particular allocation of that object.
            allocation_token=state.allocation_token,
            up_applied_tokens=frozenset(),
            up_completed_tokens=frozenset(),
            applied_tokens=frozenset(),
            completed_tokens=frozenset(),
        )
        self._ref_state = next_state
        self.meta.ref_count = count

    def get_num_tokens(self) -> int:
        with self.lock:
            token_dim = self.meta.fmt.token_dim()
            return self.meta.shape[token_dim]

    def pin(self) -> bool:
        with self.lock:
            first_pin = self.meta.pin_count == 0
            self.meta.pin_count += 1
            # Monitoring is observational: once ownership changes, failures in
            # metrics or timeout bookkeeping must not escape and make callers
            # believe the pin was never published.
            if first_pin:
                try:
                    TensorMemoryObj.monitor.update_pinned_memory_objs_count(1)
                except Exception:
                    logger.exception("Failed to update pinned-memory metrics")
            try:
                PinMonitor.GetOrCreate().on_pin(self)
            except Exception:
                logger.exception("Failed to register pinned memory object")
            return True

    def unpin(self) -> bool:
        with self.lock:
            self.meta.pin_count -= 1

            # Once the count changes, all bookkeeping and reclamation is
            # best-effort so a caller can treat return/raise as an exact
            # ownership boundary.
            if self.meta.pin_count == 0:
                try:
                    TensorMemoryObj.monitor.update_pinned_memory_objs_count(-1)
                except Exception:
                    logger.exception("Failed to update pinned-memory metrics")
                try:
                    PinMonitor.GetOrCreate().on_unpin(self)
                except Exception:
                    logger.exception("Failed to unregister pinned memory object")

            if self.meta.pin_count <= 0 and self._ref_state.count <= 0:
                if self.parent_allocator is None:
                    logger.error(
                        "Parent allocator is None when trying to free MemoryObj."
                        "This could cause memory leak"
                    )
                else:
                    try:
                        self.parent_allocator.free(self)
                    except Exception:
                        logger.exception("Failed to reclaim unpinned memory object")

            if self.meta.pin_count < 0:
                logger.warning(
                    f"Pin count of MemoryObj {self.meta.address}"
                    f"is negative: {self.meta.pin_count}."
                    "Double unpin occurred somewhere."
                    "Setting pin count back to 0 as a hack but please find the bug."
                )
                self.meta.pin_count = 0
            return True

    @property
    def metadata(self) -> MemoryObjMetadata:
        with self.lock:
            # ``_ref_state`` is the ownership authority.  Keep the public
            # compatibility metadata coherent even if an interruption landed
            # after the atomic state commit and before the eager mirror write.
            self.meta.ref_count = self._ref_state.count
            return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        assert self.meta.dtype is not None
        if self._used_size_override is not None:
            # Narrowed byte buffer (see set_used_size): expose exactly
            # the used bytes as a flat uint8 view.  Reshaping to the
            # original meta.shape would raise -- fewer than shape-many
            # bytes are logically present -- so keep the view consistent
            # with get_size()/byte_array/shm_byte_length, which all
            # report the narrowed length.  Consumers that build SHM
            # transport slots from ``tensor.shape`` and
            # ``shm_byte_length`` then stay self-consistent.
            return self.raw_data[: self._used_size_override].view(torch.uint8)
        # TODO(Jiayi): consider caching the `get_size()`
        return (
            self.raw_data[: self.get_size()].view(self.meta.dtype).view(self.meta.shape)
        )

    @property
    def byte_array(self) -> memoryview:
        # TODO: consider using one of the alternatives

        # Alternative 1:
        # # PyTorch tensors support buffer protocol directly for CPU tensors
        # return memoryview(self.raw_data)

        # Alternative 2:
        # assert self.raw_data.device.type == 'cpu',
        #   "byte_array only works with CPU tensors"
        # return memoryview(self.raw_data.contiguous().numpy())

        # Use logical size (get_size) rather than raw_data physical size.
        # The raw_data buffer may include alignment padding (e.g. from
        # batched_allocate) that must not be exposed to callers such as
        # remote-backend put/get which rely on byte_array length matching
        # the metadata length.
        num_bytes = self.get_size()
        ptr = self.raw_data.data_ptr()
        # ctypes does not cache (c_ubyte * N) array types -- each `*` builds a
        # fresh heap type. With this property accessed once per remote put/get,
        # uncached creation leaks ~1-2 kB per call (heap-type metadata is held
        # by the type system and never reclaimed). Cache the array type per
        # size so steady-state usage reuses a fixed set of types. See
        # https://github.com/LMCache/LMCache/issues/3767.
        arr_type = _get_cached_ubyte_array_type(num_bytes)
        ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
        byte_array = arr_type.from_address(ctypes.addressof(ubyte_ptr.contents))
        return memoryview(byte_array)

    @property
    def data_ptr(self) -> int:
        return self.raw_data.data_ptr()

    @property
    def is_pinned(self) -> bool:
        return self.metadata.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        A memory obj can be evicted if it is not pinned and ref_count=1.
        """
        return not self.is_pinned and self.get_ref_count() == 1

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        return self.raw_data

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        assert self.meta.shapes is not None
        assert self.meta.dtypes is not None
        begin = self.group_prefix_sum[index]
        end = self.group_prefix_sum[index + 1]
        return (
            self.raw_data[begin:end]
            .view(self.meta.dtypes[index])
            .view(self.meta.shapes[index])
        )

    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        return self.parent_allocator


class BytesBufferMemoryObj(MemoryObj):
    """
    Wraps a raw flat tensor with some metadata
    """

    def __init__(self, raw_bytes: bytes, metadata: Optional[MemoryObjMetadata] = None):
        self.raw_data = raw_bytes
        if metadata is None:
            bytes_shape = torch.Size([len(self.raw_data), 0, 0, 0])
            metadata = MemoryObjMetadata(
                shape=bytes_shape,
                dtype=None,
                address=0,
                phy_size=0,
                ref_count=1,
                pin_count=0,
                fmt=MemoryFormat.BINARY_BUFFER,
            )
        super().__init__(metadata)
        self.valid = True

    def invalidate(self):
        self.valid = False

    def is_valid(self):
        return self.valid

    def get_size(self) -> int:
        return len(self.raw_data)

    def get_shape(self) -> torch.Size:
        return torch.Size([len(self.raw_data), 0, 0, 0])

    def get_dtype(self) -> Optional[torch.dtype]:
        return None

    def get_shapes(self) -> list[torch.Size]:
        return [self.get_shape()]

    def get_dtypes(self) -> list[torch.dtype]:
        return []

    def get_memory_format(self) -> MemoryFormat:
        return self.metadata.fmt

    def get_physical_size(self) -> int:
        return self.metadata.phy_size

    def pin(self) -> bool:
        self.metadata.pin_count += 1
        return True

    def unpin(self) -> bool:
        self.metadata.pin_count -= 1
        if self.metadata.pin_count < 0:
            logger.warning(
                f"Pin count of MemoryObj {self.meta.address}"
                f"is negative: {self.meta.pin_count}."
                "Double unpin occurred somewhere."
                "Setting pin count back to 0 as a hack but please find the bug."
            )
            self.metadata.pin_count = 0
        return True

    def ref_count_up(self):
        pass

    def make_ref_count_up_receipt(self) -> RefCountUpReceipt:
        """Create a receipt for this non-owning immutable byte buffer."""
        return RefCountUpReceipt(0, self, object())

    def ref_count_up_once(self, receipt: RefCountUpReceipt) -> None:
        if receipt.allocation_token is not self or receipt.allocation_epoch != 0:
            raise ValueError("byte-buffer ref-count receipt does not match object")

    def ref_count_up_receipt_complete(self, receipt: RefCountUpReceipt) -> bool:
        self.ref_count_up_once(receipt)
        return True

    def ref_count_up_receipt_applied(self, receipt: RefCountUpReceipt) -> bool:
        self.ref_count_up_once(receipt)
        return True

    def ref_count_down(self):
        pass

    def make_ref_count_down_receipt(self) -> RefCountDownReceipt:
        """Create a receipt for this non-owning immutable byte buffer."""
        return RefCountDownReceipt(0, self, object())

    def ref_count_down_once(self, receipt: RefCountDownReceipt) -> None:
        if receipt.allocation_token is not self or receipt.allocation_epoch != 0:
            raise ValueError("byte-buffer ref-count receipt does not match object")

    def ref_count_down_receipt_complete(
        self,
        receipt: RefCountDownReceipt,
    ) -> bool:
        self.ref_count_down_once(receipt)
        return True

    def get_ref_count(self) -> int:
        return 1

    def get_num_tokens(self) -> int:
        # TODO(Jiayi): record the number of tokens somehow
        return 1

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        return None

    @property
    def byte_array(self) -> bytes:
        return self.raw_data

    @property
    def data_ptr(self) -> int:
        mv = memoryview(self.raw_data)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
        return addr

    @property
    def is_pinned(self) -> bool:
        return self.metadata.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        A buffer memory obj can be evicted if it is not pinned.
        """
        return not self.is_pinned

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        return None

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        return None

    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        # NOTE: BytesBufferMemoryObj may not be allocated by any allocator,
        # so just return None here
        return None


class GDSMemoryObject(MemoryObj):
    """A slab-anchored ``MemoryObj`` for the GDS L1 tier.

    The bytes live in the GDS slab file, not in host or device memory, so
    this object carries only the slab ``(offset, size)`` (in ``meta.address``
    / ``meta.phy_size``) and is otherwise a placeholder: ``tensor`` is always
    ``None`` and ``byte_array`` / ``data_ptr`` raise.
    """

    def __init__(self, metadata: MemoryObjMetadata) -> None:
        super().__init__(metadata)
        self.valid = True

    @property
    def slab_offset(self) -> int:
        """Byte offset of this chunk within the slab file (== ``meta.address``)."""
        return self.meta.address

    def invalidate(self) -> None:
        self.valid = False

    def is_valid(self) -> bool:
        return self.valid

    def get_size(self) -> int:
        return self.meta.phy_size

    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> Optional[torch.dtype]:
        return self.meta.dtype

    def get_shapes(self) -> list[torch.Size]:
        raise NotImplementedError(
            "GDSMemoryObject.get_shapes: per-group shapes are not tracked on "
            "the GDS path (only the singular meta.shape is); use get_shape()"
        )

    def get_dtypes(self) -> list[torch.dtype]:
        raise NotImplementedError(
            "GDSMemoryObject.get_dtypes: per-group dtypes are not tracked on "
            "the GDS path (only the singular meta.dtype is); use get_dtype()"
        )

    def get_memory_format(self) -> MemoryFormat:
        return self.meta.fmt

    def get_physical_size(self) -> int:
        return self.meta.phy_size

    def ref_count_up(self) -> None:
        raise NotImplementedError(
            "GDSMemoryObject.ref_count_up: not used on the GDS path"
        )

    def ref_count_down(self) -> None:
        raise NotImplementedError(
            "GDSMemoryObject.ref_count_down: not used on the GDS path"
        )

    def get_ref_count(self) -> int:
        raise NotImplementedError(
            "GDSMemoryObject.get_ref_count: not used on the GDS path"
        )

    def get_num_tokens(self) -> int:
        raise NotImplementedError(
            "GDSMemoryObject.get_num_tokens: not used on the GDS path"
        )

    def pin(self) -> bool:
        raise NotImplementedError("GDSMemoryObject.pin: not used on the GDS path")

    def unpin(self) -> bool:
        raise NotImplementedError("GDSMemoryObject.unpin: not used on the GDS path")

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        return None

    @property
    def byte_array(self) -> bytes:
        raise NotImplementedError(
            f"GDSMemoryObject(slab_offset={self.slab_offset}).byte_array is not "
            "supported; bytes live in the GDS slab file and the staging buffer "
            "is registered VRAM (no buffer protocol)."
        )

    @property
    def data_ptr(self) -> int:
        raise NotImplementedError(
            f"GDSMemoryObject(slab_offset={self.slab_offset}).data_ptr is not "
            "supported; GDS reads/writes use gpu_buffer.data_ptr() via the "
            "gpu_ops dispatch, never the MemoryObj's data_ptr."
        )

    @property
    def is_pinned(self) -> bool:
        raise NotImplementedError("GDSMemoryObject.is_pinned: not used on the GDS path")

    @property
    def can_evict(self) -> bool:
        raise NotImplementedError("GDSMemoryObject.can_evict: not used on the GDS path")

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        return None

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        return None

    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        # The GDS slab is not a MemoryAllocatorInterface; dispatch in gpu_ops
        # keys off the GDSMemoryObject type, not the parent allocator.
        return None


class MemoryAllocatorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
        allocation_receipt: Optional[AllocationReceipt] = None,
    ) -> Optional[MemoryObj]:
        """
        Allocates the memory to hold a tensor of the given shape.

        :param torch.Size shapes: The shape of the tensor to allocate.
        :param torch.dtype dtypes: The dtype of the tensor to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.

        :return: A MemoryObj wrapping the allocated memory. Returns
            None if the allocation failed.

        :rtype: Optional[MemoryObj]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
        allocation_receipts: Optional[Sequence[AllocationReceipt]] = None,
    ) -> Optional[List[MemoryObj]]:
        """
        Batched allocate the memory to hold a tensor of the given shape.

        :param torch.Size shapes: The shape of the tensor to allocate.
        :param torch.dtype dtypes: The dtype of the tensor to allocate.
        :param int batch_size: The number of tensors to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.

        :return: A list of MemoryObjs wrapping the allocated memory.
            Returns None if the allocation failed.

        :rtype: Optional[List[MemoryObj]]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ):
        """
        Frees the memory allocated for the given MemoryObj.
        Note that this function shouldn't be explicitly called.
        Instead, use `ref_count_down` to decrease ref count.

        :param MemoryObj memory_obj: The MemoryObj to free.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        """
        Frees the memory allocated for the given list of MemoryObjs.

        :param List[MemoryObj] memory_objs: The list of MemoryObjs
            to free.
        """
        raise NotImplementedError

    def close(self):
        """
        Closes the memory allocator.
        This is called when the LMCacheEngine is closed.
        """
        return

    def memcheck(self) -> bool:
        """
        Checks the memory allocator for consistency.

        Returns:
            True if everything is fine otherwise False
        """
        return True

    # TODO(chunxiaozheng): remove if after all params replaced by shapes/dtypes
    def _adapt_shapes_and_dtypes(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
    ) -> Tuple[list[torch.Size], list[torch.dtype]]:
        if isinstance(shapes, torch.Size):
            shapes = [shapes]

        if isinstance(dtypes, torch.dtype):
            dtypes = [dtypes]

        assert len(shapes) == len(dtypes), (
            f"shapes and dtypes must have the same length, "
            f"got {len(shapes)} and {len(dtypes)}, "
            f"shapes: {shapes}, dtypes: {dtypes}"
        )
        return shapes, dtypes


@dataclass(frozen=True)
class _AddressBlock:
    start: int
    size: int


@dataclass(frozen=True)
class _AddressAllocation:
    start: int
    size: int
    generation: int


@dataclass(frozen=True)
class _AddressManagerState:
    """One immutable snapshot of free space, live identities, and accounting."""

    heap_size: int
    free_blocks: tuple[_AddressBlock, ...]
    live_allocations: Mapping[object, _AddressAllocation]
    total_allocated_size: int
    next_generation: int


class AddressManager:
    """Manage aligned virtual addresses with exact allocation identities.

    Legacy ``allocate``/``free`` remain available. Exact cleanup owners use
    ``allocate_once``/``batched_allocate_once`` and ``free_once`` so a delayed
    retry can only release the generation named by its opaque receipt.
    """

    ALIGN_BYTES = 4096

    def __init__(self, size: int, align_bytes: int = ALIGN_BYTES):
        if size < 0:
            raise ValueError("address-space size cannot be negative")
        if align_bytes <= 0 or align_bytes & (align_bytes - 1) != 0:
            raise ValueError("align_bytes must be a positive power of two")
        self._align = align_bytes
        self._lock = threading.Lock()
        initial_blocks = () if size == 0 else (_AddressBlock(0, size),)
        self._state = _AddressManagerState(
            heap_size=size,
            free_blocks=initial_blocks,
            live_allocations=MappingProxyType({}),
            total_allocated_size=0,
            next_generation=0,
        )

    @property
    def total_allocated_size(self) -> int:
        """Authoritative allocated bytes from the immutable state snapshot."""
        return self._state.total_allocated_size

    @property
    def active_allocation_count(self) -> int:
        """Return the number of exact live allocation generations."""
        return len(self._state.live_allocations)

    @property
    def _explicit_list(self) -> SortedList[FreeBlock]:
        """Compatibility snapshot for diagnostics that inspected the old list."""
        snapshot: SortedList[FreeBlock] = SortedList(key=lambda item: item.start)
        snapshot.update(
            FreeBlock(block.start, block.size) for block in self._state.free_blocks
        )
        return snapshot

    def compute_aligned_size(self, raw_size: int) -> int:
        return (raw_size + self._align - 1) & ~(self._align - 1)

    def can_allocate_batch(self, raw_size: int, batch_size: int) -> bool:
        """Return whether an equal-size batch fits without changing state.

        The calculation follows the same aligned, per-free-range carving rule
        as :meth:`batched_allocate_once`. It is therefore stronger than a
        total-free-byte check when the address space is fragmented.
        """
        if type(raw_size) is not int or raw_size <= 0:
            raise ValueError("allocation size must be a positive exact integer")
        if type(batch_size) is not int or batch_size < 0:
            raise ValueError("batch size must be a nonnegative exact integer")
        if batch_size == 0:
            return True
        aligned_size = self.compute_aligned_size(raw_size)
        with self._lock:
            available = sum(
                block.size // aligned_size for block in self._state.free_blocks
            )
            return available >= batch_size

    def _can_merge_with_prev(
        self, curr_block: FreeBlock, prev_block: FreeBlock
    ) -> bool:
        return prev_block.can_be_coalesced(curr_block)

    def _can_merge_with_succ(
        self, curr_block: FreeBlock, succ_block: FreeBlock
    ) -> bool:
        return curr_block.can_be_coalesced(succ_block)

    def _merge_free_blocks(
        self, blocks: Sequence[_AddressBlock]
    ) -> tuple[_AddressBlock, ...]:
        if not blocks:
            return ()
        ordered = sorted(blocks, key=lambda block: block.start)
        merged: list[_AddressBlock] = []
        for block in ordered:
            if block.size <= 0:
                continue
            if not merged:
                merged.append(block)
                continue
            prev = merged[-1]
            prev_end = prev.start + prev.size
            if block.start < prev_end:
                raise RuntimeError("free ranges overlap")
            if self._can_merge_with_succ(
                FreeBlock(prev.start, prev.size),
                FreeBlock(block.start, block.size),
            ):
                merged[-1] = _AddressBlock(prev.start, prev.size + block.size)
            else:
                merged.append(block)
        return tuple(merged)

    def _new_state(
        self,
        *,
        heap_size: int,
        free_blocks: Sequence[_AddressBlock],
        live_allocations: Mapping[object, _AddressAllocation],
        next_generation: int,
    ) -> _AddressManagerState:
        frozen_blocks = tuple(free_blocks)
        total_allocated = heap_size - sum(block.size for block in frozen_blocks)
        return _AddressManagerState(
            heap_size=heap_size,
            free_blocks=frozen_blocks,
            live_allocations=MappingProxyType(dict(live_allocations)),
            total_allocated_size=total_allocated,
            next_generation=next_generation,
        )

    def _carve_batch(
        self,
        blocks: Sequence[_AddressBlock],
        aligned_size: int,
        batch_size: int,
    ) -> tuple[list[tuple[int, int]], tuple[_AddressBlock, ...]]:
        remaining = batch_size
        results: list[tuple[int, int]] = []
        replacement: list[_AddressBlock] = []
        for block in blocks:
            if remaining <= 0 or block.size < aligned_size:
                replacement.append(block)
                continue
            count = min(remaining, block.size // aligned_size)
            results.extend(
                (block.start + index * aligned_size, aligned_size)
                for index in range(count)
            )
            remaining -= count
            used = count * aligned_size
            if block.size > used:
                replacement.append(_AddressBlock(block.start + used, block.size - used))
        if remaining > 0:
            logger.warning(
                "Failed to batched allocate %d memory blocks of size %d "
                "because no enough memory is available (short by %d blocks)",
                batch_size,
                aligned_size,
                remaining,
            )
            raise RuntimeError(
                f"Failed to batched allocate {batch_size} memory blocks "
                f"of size {aligned_size} because no enough memory is available"
            )
        return results, tuple(replacement)

    @_lmcache_nvtx_annotate
    def allocate_once(self, size: int, receipt: AllocationReceipt) -> tuple[int, int]:
        """Allocate or replay the generation named by ``receipt``."""
        if size <= 0:
            raise ValueError("allocation size must be positive")
        receipt._bind_address_manager(self)
        aligned_size = self.compute_aligned_size(size)
        with self._lock:
            if not receipt._allocation_may_commit(self):
                raise RuntimeError("allocation receipt release has begun")
            state = self._state
            existing = state.live_allocations.get(receipt.token)
            if existing is not None:
                if existing.size != aligned_size:
                    raise ValueError("allocation receipt replay changed size")
                return existing.start, existing.size
            try:
                results, free_blocks = self._carve_batch(
                    state.free_blocks, aligned_size, 1
                )
            except RuntimeError:
                raise RuntimeError(
                    f"Failed to allocate memory block of size {size} "
                    "because no memory is available"
                ) from None
            start, allocated_size = results[0]
            live_allocations = dict(state.live_allocations)
            live_allocations[receipt.token] = _AddressAllocation(
                start=start,
                size=allocated_size,
                generation=state.next_generation,
            )
            next_state = self._new_state(
                heap_size=state.heap_size,
                free_blocks=free_blocks,
                live_allocations=live_allocations,
                next_generation=state.next_generation + 1,
            )
            # Free ranges, generation identity, and accounting publish together.
            self._state = next_state
            return start, allocated_size

    @_lmcache_nvtx_annotate
    def allocate(self, size: int) -> tuple[int, int]:
        """Legacy allocation API; exact allocator paths retain their receipt."""
        return self.allocate_once(size, AllocationReceipt())

    @_lmcache_nvtx_annotate
    def batched_allocate_once(
        self,
        size: int,
        receipts: Sequence[AllocationReceipt],
    ) -> list[tuple[int, int]]:
        """Atomically allocate or replay one generation per receipt."""
        if size <= 0:
            raise ValueError("allocation size must be positive")
        if len({id(receipt) for receipt in receipts}) != len(receipts):
            raise ValueError("batched allocation receipts must be unique")
        for receipt in receipts:
            receipt._bind_address_manager(self)
        if not receipts:
            return []
        aligned_size = self.compute_aligned_size(size)
        with self._lock:
            if not all(receipt._allocation_may_commit(self) for receipt in receipts):
                raise RuntimeError("batched allocation receipt release has begun")
            state = self._state
            existing = [state.live_allocations.get(item.token) for item in receipts]
            if any(item is not None for item in existing):
                if not all(item is not None for item in existing):
                    raise RuntimeError("batched allocation receipt replay is partial")
                replay = []
                for item in existing:
                    assert item is not None
                    if item.size != aligned_size:
                        raise ValueError("allocation receipt replay changed size")
                    replay.append((item.start, item.size))
                return replay

            results, free_blocks = self._carve_batch(
                state.free_blocks, aligned_size, len(receipts)
            )
            live_allocations = dict(state.live_allocations)
            generation = state.next_generation
            for receipt, (start, allocated_size) in zip(receipts, results, strict=True):
                live_allocations[receipt.token] = _AddressAllocation(
                    start=start,
                    size=allocated_size,
                    generation=generation,
                )
                generation += 1
            next_state = self._new_state(
                heap_size=state.heap_size,
                free_blocks=free_blocks,
                live_allocations=live_allocations,
                next_generation=generation,
            )
            # The whole batch has one authoritative publication boundary.
            self._state = next_state
            return results

    @_lmcache_nvtx_annotate
    def batched_allocate(self, size: int, batch_size: int) -> list[tuple[int, int]]:
        if batch_size < 0:
            raise ValueError("batch size cannot be negative")
        return self.batched_allocate_once(
            size, [AllocationReceipt() for _ in range(batch_size)]
        )

    def _free_range_locked(
        self,
        state: _AddressManagerState,
        address: int,
        size: int,
        *,
        exact_token: Optional[object] = None,
    ) -> tuple[_AddressManagerState, bool]:
        end = address + size
        if address < 0 or end > state.heap_size:
            raise ValueError("free range lies outside the address space")
        if any(
            block.start <= address and block.start + block.size >= end
            for block in state.free_blocks
        ):
            return state, False
        for block in state.free_blocks:
            block_end = block.start + block.size
            if address < block_end and end > block.start:
                raise RuntimeError("free range partially overlaps existing free space")

        live_allocations = dict(state.live_allocations)
        if exact_token is not None:
            allocation = live_allocations.get(exact_token)
            if allocation is None:
                return state, False
            if allocation.start != address or allocation.size != size:
                raise RuntimeError("allocation receipt range changed unexpectedly")
            del live_allocations[exact_token]
        else:
            overlapping_tokens: list[object] = []
            for token, allocation in live_allocations.items():
                allocation_end = allocation.start + allocation.size
                if address < allocation_end and end > allocation.start:
                    if allocation.start < address or allocation_end > end:
                        raise RuntimeError("free range splits a live allocation")
                    overlapping_tokens.append(token)
            if not overlapping_tokens:
                raise RuntimeError("free range does not name a live allocation")
            for token in overlapping_tokens:
                del live_allocations[token]

        free_blocks = self._merge_free_blocks(
            (*state.free_blocks, _AddressBlock(address, size))
        )
        return (
            self._new_state(
                heap_size=state.heap_size,
                free_blocks=free_blocks,
                live_allocations=live_allocations,
                next_generation=state.next_generation,
            ),
            True,
        )

    @_lmcache_nvtx_annotate
    def free_once(self, receipt: AllocationReceipt) -> bool:
        """Free only the live generation named by ``receipt``."""
        if not receipt._begin_address_release(self):
            return False
        changed = False
        with self._lock:
            state = self._state
            allocation = state.live_allocations.get(receipt.token)
            if allocation is not None:
                next_state, changed = self._free_range_locked(
                    state,
                    allocation.start,
                    allocation.size,
                    exact_token=receipt.token,
                )
                # Token removal, free-space publication, and accounting are one
                # pointer assignment, so delayed retries cannot free reuse.
                self._state = next_state
        receipt._finish_address_release(self)
        return changed

    @_lmcache_nvtx_annotate
    def free(self, address: int, size: int) -> None:
        """Legacy range free, atomically retiring every covered generation."""
        if size <= 0:
            raise ValueError("free size must be positive")
        with self._lock:
            next_state, changed = self._free_range_locked(self._state, address, size)
            if changed:
                self._state = next_state

    def sbrk(self, size: int) -> None:
        size = self.compute_aligned_size(size)
        if size <= 0:
            return
        with self._lock:
            state = self._state
            free_blocks = self._merge_free_blocks(
                (*state.free_blocks, _AddressBlock(state.heap_size, size))
            )
            self._state = self._new_state(
                heap_size=state.heap_size + size,
                free_blocks=free_blocks,
                live_allocations=state.live_allocations,
                next_generation=state.next_generation,
            )

    def get_heap_size(self) -> int:
        return self._state.heap_size

    def get_free_size(self) -> int:
        state = self._state
        return state.heap_size - state.total_allocated_size

    def check_consistency(self) -> bool:
        state = self._state
        for prev, succ in zip(
            state.free_blocks[:-1], state.free_blocks[1:], strict=False
        ):
            if prev.start + prev.size >= succ.start:
                return False
        total_free_size = sum(block.size for block in state.free_blocks)
        if total_free_size + state.total_allocated_size != state.heap_size:
            return False
        allocations = sorted(
            state.live_allocations.values(), key=lambda allocation: allocation.start
        )
        if (
            sum(allocation.size for allocation in allocations)
            != state.total_allocated_size
        ):
            return False
        for prev_allocation, succ_allocation in zip(
            allocations[:-1], allocations[1:], strict=False
        ):
            if prev_allocation.start + prev_allocation.size > succ_allocation.start:
                return False
        return True


_CORE_EXPORTS = [
    "AllocationReceipt",
    "AddressManager",
    "BytesBufferMemoryObj",
    "FreeBlock",
    "GDSMemoryObject",
    "MemoryAllocatorInterface",
    "MemoryFormat",
    "MemoryObj",
    "MemoryObjMetadata",
    "TensorMemoryObj",
    "torch_device_type",
]

__all__ = list(_CORE_EXPORTS)
