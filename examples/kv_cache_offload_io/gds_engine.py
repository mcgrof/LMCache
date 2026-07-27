# SPDX-License-Identifier: Apache-2.0
"""GPU-direct KV-offload engines for run_kv_offload_io.py.

Two engines that move the same KV-shaped byte layout as LMCache's
``raw_block`` core (1 MiB metadata region, then fixed slots: header at
the slot base, payload at ``slot + header_bytes``) but through the
GPUDirect-Storage data path instead of host DRAM:

- ``cufile``: NVIDIA's proprietary libcufile directly (cudaMalloc buffer
  + cuFileWrite/Read).
- ``opends``: the open OpenDS API (https://github.com/xnvme/opends). Its
  backend variants are separate shared libraries with one ABI, so
  ``--gds-backend X`` simply loads ``libopends_X.so``: ``gds`` (wraps
  cuFile, GPU memory), ``ref`` (POSIX reference, host memory -- runs
  without a GPU), and future backends (e.g. ``aisio``) work unmodified.
  Buffers come from ``opends_alloc`` so the *backend* picks the right
  memory class; this file never touches CUDA for the opends engine.

Both emit the same schema-2 ``LMCACHE_KVIO_TRACE`` semantic records as
RawBlockCore, so kvio2perfetto.py / kvio_tp_report.py consume GDS runs
unchanged. GDS I/O carries no io_uring user_data, so eBPF-side
attribution joins by slot offset rather than trace_id.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import time


def _round_up(n: int, align: int) -> int:
    return ((n + align - 1) // align) * align


def _find_lib(names, dirs):
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ctypes.CDLL(p)
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    raise OSError(f"cannot locate any of {names} (searched {dirs})")


_CUDA_DIRS = [
    os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "lib64"),
    "/usr/local/cuda/lib64", "/usr/local/cuda-13/lib64",
    "/usr/local/cuda-12/lib64",
]


class _OpendsError(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("dev_err", ctypes.c_int)]


class _CUfileError(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]


class _CUfileHandleU(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("handle", ctypes.c_void_p)]


class _CUfileDescr(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("handle", _CUfileHandleU),
                ("fs_ops", ctypes.c_void_p)]


CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1


class _CufileBackend:
    """Proprietary data path: cudaMalloc + cuFileWrite/Read."""

    name = "cufile"

    def __init__(self, fd: int, buf_bytes: int):
        self._rt = _find_lib(["libcudart.so", "libcudart.so.13",
                              "libcudart.so.12"], _CUDA_DIRS)
        self._cf = _find_lib(["libcufile.so", "libcufile.so.0"], _CUDA_DIRS)
        for fn in ("cuFileRead", "cuFileWrite"):
            f = getattr(self._cf, fn)
            f.restype = ctypes.c_ssize_t
            f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_int64, ctypes.c_int64]
        self._cf.cuFileDriverOpen.restype = _CUfileError
        self._cf.cuFileHandleRegister.restype = _CUfileError
        self._cf.cuFileBufRegister.restype = _CUfileError

        # cudaMalloc first: its implicit primary context is what cuFile
        # binds to.
        ptr = ctypes.c_void_p()
        rc = self._rt.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(buf_bytes))
        if rc != 0:
            raise RuntimeError(f"cudaMalloc({buf_bytes}) -> {rc}")
        self._rt.cudaMemset(ptr, 0, ctypes.c_size_t(buf_bytes))
        self.buf = ptr
        e = self._cf.cuFileDriverOpen()
        if e.err != 0:
            raise RuntimeError(f"cuFileDriverOpen err={e.err} cu={e.cu_err}")
        descr = _CUfileDescr()
        descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD
        descr.handle.fd = fd
        self._fh = ctypes.c_void_p()
        e = self._cf.cuFileHandleRegister(ctypes.byref(self._fh),
                                          ctypes.byref(descr))
        if e.err != 0:
            raise RuntimeError(f"cuFileHandleRegister err={e.err}")
        e = self._cf.cuFileBufRegister(self.buf, ctypes.c_size_t(buf_bytes), 0)
        if e.err != 0:
            raise RuntimeError(f"cuFileBufRegister err={e.err}")

    def write(self, size: int, file_off: int) -> int:
        return self._cf.cuFileWrite(self._fh, self.buf, size, file_off, 0)

    def read(self, size: int, file_off: int) -> int:
        return self._cf.cuFileRead(self._fh, self.buf, size, file_off, 0)

    def close(self):
        try:
            self._cf.cuFileBufDeregister(self.buf)
            self._cf.cuFileHandleDeregister(self._fh)
            self._rt.cudaFree(self.buf)
            self._cf.cuFileDriverClose()
        except Exception:
            pass


class _OpendsBackend:
    """Open data path: one ABI, backend picked by which .so is loaded."""

    def __init__(self, fd: int, buf_bytes: int, backend: str, lib_dir):
        self.name = f"opends-{backend}"
        dirs = [d for d in [lib_dir, os.environ.get("OPENDS_LIB_DIR"),
                            os.path.expanduser("~/opends/build")] if d]
        self._od = _find_lib([f"libopends_{backend}.so"], dirs)
        for fn in ("opends_read", "opends_write"):
            f = getattr(self._od, fn)
            f.restype = ctypes.c_ssize_t
            f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_int64, ctypes.c_int64]
        self._od.opends_driver_open.restype = _OpendsError
        self._od.opends_handle_register.restype = _OpendsError
        self._od.opends_alloc.restype = ctypes.c_void_p
        self._od.opends_alloc.argtypes = [ctypes.c_size_t]
        # Without argtypes ctypes passes pointers as 32-bit ints -- the
        # truncated pointer segfaults inside the library at teardown.
        self._od.opends_free.argtypes = [ctypes.c_void_p]
        self._od.opends_handle_deregister.argtypes = [ctypes.c_void_p]

        e = self._od.opends_driver_open()
        if e.err != 0:
            raise RuntimeError(f"opends_driver_open err={e.err} dev={e.dev_err}")
        self._fh = ctypes.c_void_p()
        e = self._od.opends_handle_register(ctypes.byref(self._fh), fd)
        if e.err != 0:
            raise RuntimeError(f"opends_handle_register err={e.err}")
        # Backend-owned allocation: the gds variant hands back GPU memory,
        # ref hands back host memory -- correct DMA setup either way.
        self.buf = self._od.opends_alloc(ctypes.c_size_t(buf_bytes))
        if not self.buf:
            raise RuntimeError(f"opends_alloc({buf_bytes}) failed")

    def write(self, size: int, file_off: int) -> int:
        return self._od.opends_write(self._fh, self.buf, size, file_off, 0)

    def read(self, size: int, file_off: int) -> int:
        return self._od.opends_read(self._fh, self.buf, size, file_off, 0)

    def close(self):
        try:
            self._od.opends_free(self.buf)
            self._od.opends_handle_deregister(self._fh)
            self._od.opends_driver_close()
        except Exception:
            pass


class GdsKVEngine:
    """RawBlockCore's slot layout + semantic trace over a GDS data path.

    Store = header write at the slot base + payload write at
    ``slot + header_bytes``; load = payload read. Offsets and record
    fields match RawBlockCore so every downstream consumer is shared.
    """

    def __init__(self, path, engine, backend, lib_dir, slot_bytes,
                 header_bytes, block_align, obj_bytes, capacity_bytes,
                 mdts, trace_path=None):
        self.slot_bytes = slot_bytes
        self.header_bytes = header_bytes
        self.data_base = 1 << 20  # RawBlockCore meta_total_bytes
        self.obj_bytes = obj_bytes
        self.io_bytes = _round_up(obj_bytes, block_align)  # O_DIRECT pad
        self.max_slots = (capacity_bytes - self.data_base) // slot_bytes
        if self.max_slots <= 0:
            raise ValueError("capacity too small for one slot")

        flags = os.O_RDWR | os.O_CREAT | os.O_DIRECT
        self.fd = os.open(path, flags, 0o644)
        if os.fstat(self.fd).st_size < capacity_bytes:
            try:  # regular file: materialize blocks up front
                os.posix_fallocate(self.fd, 0, capacity_bytes)
            except OSError:
                pass  # block device
        cls = _CufileBackend if engine == "cufile" else _OpendsBackend
        if engine == "cufile":
            self.be = cls(self.fd, self.io_bytes)
        else:
            self.be = cls(self.fd, self.io_bytes, backend, lib_dir)

        # cuFile compat-path quirk (measured, GDS 1.18.1.6): unless the
        # process's FIRST cuFile read is >= the 1 MiB bounce-pool slab
        # class, every read after a write gets chopped into 4 KiB posix
        # reads for the process lifetime (~47x slower: 912 ms vs 19.5 ms
        # per 32 MiB here; prime sweep: 4K poisoned, 1M/16M/32M fine).
        # One throwaway >=1 MiB read before any write primes the pool
        # read-capable. Harmless on non-cuFile backends.
        try:
            self.be.read(min(1 << 20, self.io_bytes), 0)
        except Exception:
            pass

        # schema-2 semantic trace, same shape RawBlockCore emits
        self._trace = trace_path
        self._tid = 0
        if trace_path:
            self._pid = os.getpid()
            self._inst = os.urandom(4).hex()
            self._emit_raw({
                "event_type": "kvio_meta", "kvio_schema": 2,
                "hostname": os.uname().nodename,
                "pid": self._pid, "instance": self._inst,
                "device_path": str(path),
                "capacity_bytes": int(capacity_bytes),
                "slot_bytes": int(slot_bytes),
                "block_align": int(block_align),
                "header_bytes": int(header_bytes),
                "max_data_transfer_size": int(mdts),
                "io_engine": self.be.name, "use_uring_cmd": False,
                "ts_monotonic": time.monotonic(),
                "ts_realtime": time.time(),
            })

    def _emit_raw(self, rec):
        try:
            with open(self._trace, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def _emit(self, op, key, offset, ts_start, error=None):
        if not self._trace:
            return
        rec = {
            "trace_id": self._tid, "op": op, "key": key, "object_id": key,
            "part": "kv", "bytes": int(self.obj_bytes),
            "slot_offset": int(offset), "ts": time.monotonic(),
            "pid": self._pid, "instance": self._inst, "ts_start": ts_start,
        }
        if error is not None:
            rec["error"] = str(error)[:120]
        self._emit_raw(rec)

    def _slot_offset(self, slot_idx: int) -> int:
        return self.data_base + (slot_idx % self.max_slots) * self.slot_bytes

    def store(self, key: str, slot_idx: int) -> None:
        off = self._slot_offset(slot_idx)
        self._tid += 1
        t0 = time.monotonic()
        n = self.be.write(self.header_bytes, off)
        if n == self.header_bytes:
            n = self.be.write(self.io_bytes, off + self.header_bytes)
            n = self.obj_bytes if n == self.io_bytes else n
        err = None if n == self.obj_bytes else f"short write {n}"
        self._emit("store", key, off, t0, err)
        if err:
            raise RuntimeError(f"{self.be.name} store: {err} @ {off}")

    def load(self, key: str, slot_idx: int) -> None:
        off = self._slot_offset(slot_idx)
        self._tid += 1
        t0 = time.monotonic()
        n = self.be.read(self.io_bytes, off + self.header_bytes)
        err = None if n == self.io_bytes else f"short read {n}"
        self._emit("load", key, off, t0, err)
        if err:
            raise RuntimeError(f"{self.be.name} load: {err} @ {off}")

    def close(self):
        self.be.close()
        os.close(self.fd)
