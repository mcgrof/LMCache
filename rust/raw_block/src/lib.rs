// SPDX-License-Identifier: Apache-2.0
#![allow(unknown_lints)]

//! Raw block device I/O extension for LMCache.
//! Provides direct block device access with optional O_DIRECT support.
//!
//! Design notes (for reviewers unfamiliar with Rust / Linux I/O):
//! - This module exposes a very small surface to Python via PyO3.
//! - We wrap Linux `pread` / `pwrite` on a file descriptor opened from a
//!   block device (e.g., /dev/nvmeXnY) or a regular file.
//! - When O_DIRECT is enabled, Linux requires aligned offsets and I/O sizes.
//!   If Python buffers are aligned, we use them directly; otherwise we fallback
//!   to a bounce buffer (aligned via `posix_memalign`) for safety.
//! - For io_uring case a dedicated worker thread drives the io_uring
//!   submission/completion loop. All alignment checks are performed before
//!   enqueuing; violations result in an immediate Python `ValueError`.

use pyo3::exceptions::{PyMemoryError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyMemoryView};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::CString;
use std::io;
use std::os::unix::io::AsRawFd;
use std::os::unix::io::RawFd;
use std::slice;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use io_uring::cqueue::{Entry, Entry32};
use io_uring::squeue::{Entry as SqueueEntry, Entry128};
use io_uring::types::Fd;
use io_uring::{opcode, IoUring};

// Wrapper enum to support both standard and big io_uring entries
// This allows fallback to standard entries on kernels < 5.19
#[derive(Clone)]
enum IoUringWrapper {
    Standard(Arc<Mutex<IoUring<SqueueEntry, Entry>>>),
    Big(Arc<Mutex<IoUring<Entry128, Entry32>>>),
    /// A ring with no kernel behind it, for the transitions a real one
    /// will not produce on demand. Present only in the nondefault
    /// fault-injection build, and the drain below is a `match`, so adding
    /// it could not silently skip a completion path.
    #[cfg(feature = "fault-injection")]
    Fake(FakeRing),
}

impl IoUringWrapper {
    fn submit(&self) -> io::Result<usize> {
        match self {
            Self::Standard(ring) => ring.lock().unwrap().submitter().submit(),
            Self::Big(ring) => ring.lock().unwrap().submitter().submit(),
            #[cfg(feature = "fault-injection")]
            Self::Fake(ring) => ring.submit(),
        }
    }

    // Get the submission queue length
    fn submission_len(&self) -> usize {
        match self {
            IoUringWrapper::Standard(ring) => {
                let mut ring = ring.lock().unwrap();
                let len = ring.submission().len();
                len
            }
            IoUringWrapper::Big(ring) => {
                let mut ring = ring.lock().unwrap();
                let len = ring.submission().len();
                len
            }
            #[cfg(feature = "fault-injection")]
            IoUringWrapper::Fake(ring) => ring.submission_len(),
        }
    }

    // Sync the submission queue
    fn submission_sync(&self) {
        match self {
            IoUringWrapper::Standard(ring) => {
                let mut ring = ring.lock().unwrap();
                ring.submission().sync();
            }
            IoUringWrapper::Big(ring) => {
                let mut ring = ring.lock().unwrap();
                ring.submission().sync();
            }
            #[cfg(feature = "fault-injection")]
            IoUringWrapper::Fake(ring) => ring.sync(),
        }
    }

    fn cancel_submitted(&self) -> io::Result<()> {
        let timeout = Some(io_uring::types::Timespec::new().sec(1));
        let cancel = io_uring::types::CancelBuilder::any();
        match self {
            Self::Standard(ring) => ring
                .lock()
                .unwrap()
                .submitter()
                .register_sync_cancel(timeout, cancel),
            #[cfg(feature = "fault-injection")]
            Self::Fake(_) => Ok(()),
            Self::Big(ring) => ring
                .lock()
                .unwrap()
                .submitter()
                .register_sync_cancel(timeout, cancel),
        }
    }

    fn reap_without_submitting(&self) -> io::Result<usize> {
        let flags = io_uring::EnterFlags::GETEVENTS.bits();
        match self {
            Self::Standard(ring) => unsafe {
                ring.lock()
                    .unwrap()
                    .submitter()
                    .enter::<libc::sigset_t>(0, 0, flags, None)
            },
            #[cfg(feature = "fault-injection")]
            Self::Fake(_) => Ok(0),
            Self::Big(ring) => unsafe {
                ring.lock()
                    .unwrap()
                    .submitter()
                    .enter::<libc::sigset_t>(0, 0, flags, None)
            },
        }
    }
    /// Push one 128-byte passthrough SQE onto this ring.
    ///
    /// Only the big-entry ring can carry one: a 64-byte SQE has no room for
    /// the NVMe command, so a standard ring refuses rather than truncating.
    fn push_cmd(
        &self,
        sqe: &Entry128,
        user_data: u64,
        #[cfg_attr(not(feature = "fault-injection"), allow(unused_variables))]
        descriptor: SqeDescriptor,
    ) -> Result<(), PyErr> {
        match self {
            IoUringWrapper::Big(ring) => {
                let mut ring = ring.lock().unwrap();
                let pushed = unsafe { ring.submission().push(sqe) };
                pushed.map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "submission queue full pushing request {user_data}"
                    ))
                })
            }
            IoUringWrapper::Standard(_) => Err(PyRuntimeError::new_err(
                "io_uring_cmd requires big entries (kernel 5.19+)",
            )),
            #[cfg(feature = "fault-injection")]
            IoUringWrapper::Fake(ring) => ring.push(descriptor).map_err(|()| {
                PyRuntimeError::new_err(format!(
                    "submission queue full pushing request {user_data}"
                ))
            }),
        }
    }

    /// Push one ordinary read or write SQE onto this ring.
    ///
    /// A big-entry ring takes the same operation widened to 128 bytes; the
    /// trailing space is unused for anything but a passthrough command.
    fn push_regular(
        &self,
        sqe: &SqueueEntry,
        user_data: u64,
        #[cfg_attr(not(feature = "fault-injection"), allow(unused_variables))]
        descriptor: SqeDescriptor,
    ) -> Result<(), PyErr> {
        match self {
            IoUringWrapper::Big(ring) => {
                let widened: Entry128 = sqe.clone().into();
                let mut ring = ring.lock().unwrap();
                let pushed = unsafe { ring.submission().push(&widened) };
                pushed.map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "submission queue full pushing request {user_data}"
                    ))
                })
            }
            IoUringWrapper::Standard(ring) => {
                let mut ring = ring.lock().unwrap();
                let pushed = unsafe { ring.submission().push(sqe) };
                pushed.map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "submission queue full pushing request {user_data}"
                    ))
                })
            }
            #[cfg(feature = "fault-injection")]
            IoUringWrapper::Fake(ring) => ring.push(descriptor).map_err(|()| {
                PyRuntimeError::new_err(format!(
                    "submission queue full pushing request {user_data}"
                ))
            }),
        }
    }

    /// The ring's own descriptor, for an operation that needs it directly.
    ///
    /// A ring with no kernel behind it has no descriptor, and refuses
    /// rather than offering one that addresses nothing.
    fn ring_fd(&self) -> Result<RawFd, PyErr> {
        match self {
            IoUringWrapper::Standard(ring) => Ok(ring.lock().unwrap().as_raw_fd()),
            IoUringWrapper::Big(ring) => Ok(ring.lock().unwrap().as_raw_fd()),
            #[cfg(feature = "fault-injection")]
            IoUringWrapper::Fake(_) => Err(PyRuntimeError::new_err(
                "a fault-injection ring has no descriptor to register against",
            )),
        }
    }

    /// Register host buffers with the kernel as this ring's fixed set.
    fn register_host_buffers(&self, iovecs: &[libc::iovec]) -> io::Result<()> {
        match self {
            IoUringWrapper::Standard(ring) => {
                let ring = ring.lock().unwrap();
                unsafe { ring.submitter().register_buffers(iovecs) }
            }
            IoUringWrapper::Big(ring) => {
                let ring = ring.lock().unwrap();
                unsafe { ring.submitter().register_buffers(iovecs) }
            }
            #[cfg(feature = "fault-injection")]
            IoUringWrapper::Fake(_) => Err(io::Error::from_raw_os_error(libc::ENOTSUP)),
        }
    }

    /// Reserve an empty fixed-buffer table of the given size.
    fn register_sparse_buffers(&self, count: u32) -> io::Result<()> {
        match self {
            IoUringWrapper::Standard(ring) => ring
                .lock()
                .unwrap()
                .submitter()
                .register_buffers_sparse(count),
            IoUringWrapper::Big(ring) => ring
                .lock()
                .unwrap()
                .submitter()
                .register_buffers_sparse(count),
            #[cfg(feature = "fault-injection")]
            IoUringWrapper::Fake(ring) => ring.register("sparse", count, 0),
        }
    }

    /// Install one dma-buf in the fixed-buffer table.
    ///
    /// The descriptor validation and the registration map are built by the
    /// caller, which is what both ring kinds share. This is only the
    /// registration itself: the real rings make the syscall, and the
    /// fault-injection ring records an explicitly synthetic entry so the
    /// path above it can be driven on a machine with no dma-buf to register.
    /// No synthetic record is evidence that a kernel registered anything.
    fn register_dmabuf_slot(
        &self,
        index: u32,
        dmabuf_fd: RawFd,
        device_fd: RawFd,
    ) -> io::Result<()> {
        match self {
            IoUringWrapper::Standard(_) | IoUringWrapper::Big(_) => {
                let ring_fd = self
                    .ring_fd()
                    .map_err(|_| io::Error::from_raw_os_error(libc::EBADF))?;
                io_uring_register_dmabuf(ring_fd, index, dmabuf_fd, device_fd)
            }
            #[cfg(feature = "fault-injection")]
            IoUringWrapper::Fake(ring) => ring.register("dmabuf", 1, index),
        }
    }

    /// How many bytes the dma-buf behind this descriptor covers.
    ///
    /// Also dispatched, because a machine with no dma-buf to register has
    /// none to measure either, and the bounds check above this is part of
    /// what a test needs to reach.
    fn dmabuf_extent(&self, dmabuf_fd: RawFd) -> io::Result<usize> {
        match self {
            IoUringWrapper::Standard(_) | IoUringWrapper::Big(_) => dmabuf_extent_bytes(dmabuf_fd),
            #[cfg(feature = "fault-injection")]
            IoUringWrapper::Fake(ring) => Ok(ring.synthetic_dmabuf_extent()),
        }
    }

    /// Withdraw whatever this ring has registered.
    ///
    /// The one operation whose failure is an unknown outcome rather than a
    /// refusal: a registration the kernel still holds covers memory this
    /// process was about to forget.
    fn unregister_buffers(&self) -> io::Result<()> {
        match self {
            IoUringWrapper::Standard(ring) => ring.lock().unwrap().submitter().unregister_buffers(),
            IoUringWrapper::Big(ring) => ring.lock().unwrap().submitter().unregister_buffers(),
            #[cfg(feature = "fault-injection")]
            IoUringWrapper::Fake(ring) => ring.register("unregister", 0, 0),
        }
    }

    /// Take every completion the kernel has posted on this ring.
    ///
    /// The two ring types carry different CQE types that agree on the only
    /// two fields the worker reads, so they are normalised here and the
    /// worker has one completion path instead of one per ring type.
    ///
    /// This is a `match` rather than a chain of `if let`, so that a ring
    /// variant added later fails to compile here. An unmatched variant
    /// would return no completions for a request the kernel still owns,
    /// which presents as a hang rather than as an error.
    fn take_completions(&self) -> Vec<RingCompletion> {
        match self {
            IoUringWrapper::Standard(ring) => ring
                .lock()
                .unwrap()
                .completion()
                .map(|cqe| RingCompletion {
                    user_data: cqe.user_data(),
                    result: cqe.result(),
                })
                .collect(),
            IoUringWrapper::Big(ring) => ring
                .lock()
                .unwrap()
                .completion()
                .map(|cqe| RingCompletion {
                    user_data: cqe.user_data(),
                    result: cqe.result(),
                })
                .collect(),
            #[cfg(feature = "fault-injection")]
            IoUringWrapper::Fake(ring) => ring.take_completions(),
        }
    }
}

/// One completion, with the ring type it came from erased.
#[derive(Clone, Copy)]
struct RingCompletion {
    user_data: u64,
    result: i32,
}

/// Encode the ordinary SQE passed directly to the submission ring.
fn build_regular_sqe(sub: &IoSubmission, user_data: u64) -> SqueueEntry {
    let ptr = sub.ptr_addr as *mut u8;
    // Regular read/write operations
    // A dma-buf registered buffer has no user address in the
    // kernel's eyes: the SQE address is the byte offset of
    // this buffer inside the registered dma-buf.
    let fixed_addr: *mut u8 = match sub.fixed_dmabuf {
        Some(off) => off as *mut u8,
        None => ptr,
    };
    let sqe = if sub.is_write {
        if let Some(idx) = sub.fixed_buffer_idx {
            opcode::WriteFixed::new(Fd(sub.fd), fixed_addr as *const u8, sub.len as u32, idx)
                .offset(sub.offset)
                .build()
        } else {
            opcode::Write::new(Fd(sub.fd), ptr as *const u8, sub.len as u32)
                .offset(sub.offset)
                .build()
        }
    } else if let Some(idx) = sub.fixed_buffer_idx {
        opcode::ReadFixed::new(Fd(sub.fd), fixed_addr, sub.len as u32, idx)
            .offset(sub.offset)
            .build()
    } else {
        opcode::Read::new(Fd(sub.fd), ptr, sub.len as u32)
            .offset(sub.offset)
            .build()
    };
    sqe.user_data(user_data)
}

/// What one submission asks the device to do, in terms both rings share.
///
/// Built where the SQE is built, so there is one description of an
/// operation's geometry rather than one per ring type. A real ring ignores
/// it -- the SQE it was handed is the request -- and the fault-injection
/// ring records it, because `user_data` alone cannot tell a correct
/// remainder from one pointed at the wrong offset, the wrong address or the
/// wrong length. Two submissions carrying the same bytes compare equal; two
/// carrying the same bytes from different places do not.
#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "fault-injection"), allow(dead_code))]
struct SqeDescriptor {
    user_data: u64,
    /// A passthrough NVMe command rather than an ordinary read or write.
    is_cmd: bool,
    is_write: bool,
    /// Device offset, in bytes.
    offset: u64,
    /// Transfer length, in bytes.
    len: u32,
    /// The address the SQE carries. For a registered dma-buf transfer this
    /// is a byte offset inside the registration, not a process address,
    /// which is exactly the distinction a short-I/O remainder can get wrong.
    addr: u64,
    /// Registered-buffer index, or -1 when this is not a fixed operation.
    fixed_index: i64,
    /// Byte offset inside a registered dma-buf, or -1 when the address
    /// above is an ordinary process address.
    dmabuf_offset: i64,
}

#[cfg_attr(not(feature = "fault-injection"), allow(dead_code))]
impl SqeDescriptor {
    fn describe(sub: &IoSubmission, user_data: u64) -> Self {
        let addr = match sub.fixed_dmabuf {
            Some(offset) => offset as u64,
            None => sub.ptr_addr as u64,
        };
        SqeDescriptor {
            user_data,
            is_cmd: sub.nvme_cmd_data.is_some(),
            is_write: sub.is_write,
            offset: sub.offset,
            len: sub.len as u32,
            addr,
            fixed_index: sub.fixed_buffer_idx.map_or(-1, |idx| idx as i64),
            dmabuf_offset: sub.fixed_dmabuf.map_or(-1, |off| off as i64),
        }
    }
}

/// A submission and completion ring with no kernel behind it.
///
/// The submit seam substitutes what one *submit call reports*, before the
/// ring, so it can never deliver a completion -- which leaves the
/// transitions that only a completion can produce unreachable: a short
/// read, a completion arriving after close, a request taken by the kernel
/// and never answered for. This substitutes the ring itself, so a test can
/// deliver a completion for a chosen request, or deliberately withhold one.
///
/// It is also built to refuse. Every operation checks the ownership rule it
/// implies and records a violation rather than papering over it, so a test
/// asserting that the violation list is empty is asserting that the engine
/// did not take a step a real kernel would not have allowed -- completing a
/// request the kernel does not hold, most of all.
#[cfg(feature = "fault-injection")]
struct FakeRingState {
    /// How many entries fit before a push is refused, which is the only way
    /// a real submission queue reports being full.
    sq_capacity: usize,
    /// Pushed and not yet taken by a submit. This is what `submission_len`
    /// reports, and what residency means. Each entry carries the geometry it
    /// was built with, so a remainder can be checked against where it is
    /// meant to be reading or writing rather than only being counted.
    resident: Vec<SqeDescriptor>,
    /// Taken by a submit and not yet answered for. The kernel's, not ours:
    /// anything still here when the device closes is memory the device may
    /// still be reaching.
    owned: Vec<SqeDescriptor>,
    /// Completions queued for the worker to drain.
    ready: Vec<RingCompletion>,
    /// Completions held until the worker enters its shutdown drain.
    shutdown_completions: Vec<RingCompletion>,
    shutdown_started: bool,
    shutdown_delivered: usize,
    /// How many resident entries the next submit takes. `None` takes all of
    /// them, which is what a healthy ring does.
    take_next: Option<usize>,
    /// While set, every submit takes nothing and reports that it took
    /// nothing. A real submit reports zero when the entries it would have
    /// flushed were not flushed, and the worker has to leave them resident
    /// and come back. Holding it is how a test gets more than one entry
    /// resident at the same time, which is the only way a partial take can
    /// be more than a full take of one entry. The worker already paces
    /// itself when the ring will not drain, so holding does not spin.
    hold_submits: bool,
    /// What the next submit reports instead of taking anything.
    submit_error: Option<i32>,
    submit_errors: Vec<i32>,
    /// Ownership rules this ring saw broken, in the order it saw them.
    violations: Vec<String>,
    /// How many times the worker synced the submission queue, which a test
    /// uses to tell a flush apart from a no-op.
    syncs: usize,
    /// What this ring reports as the extent of any dma-buf it is asked
    /// about. A machine with no dma-buf to register has none to measure,
    /// and the bounds check in the registration path is part of what a test
    /// needs to reach.
    dmabuf_extent: usize,
    /// Registrations this ring was asked to make. Synthetic: no descriptor
    /// is passed to the kernel, and a test says so by reading them from
    /// here rather than from a device.
    registrations: Vec<FakeRegistration>,
    /// Failed registration calls, matched by kind and slot in call order.
    registration_failures: VecDeque<(String, u32, i32)>,
}

/// One resident or owned entry, as a flat map a test can compare exactly.
#[cfg(feature = "fault-injection")]
fn describe_fake_sqe(entry: &SqeDescriptor) -> HashMap<String, i64> {
    let mut described: HashMap<String, i64> = HashMap::new();
    described.insert("user_data".to_string(), entry.user_data as i64);
    described.insert("is_cmd".to_string(), entry.is_cmd as i64);
    described.insert("is_write".to_string(), entry.is_write as i64);
    described.insert("offset".to_string(), entry.offset as i64);
    described.insert("len".to_string(), entry.len as i64);
    described.insert("addr".to_string(), entry.addr as i64);
    described.insert("fixed_index".to_string(), entry.fixed_index);
    described.insert("dmabuf_offset".to_string(), entry.dmabuf_offset);
    described
}

/// One synthetic buffer registration, as the fault-injection ring records it.
#[cfg(feature = "fault-injection")]
#[derive(Clone)]
struct FakeRegistration {
    /// What was asked: "host", "sparse", "dmabuf" or "unregister".
    kind: String,
    /// How many slots the request covers.
    count: u32,
    /// Where in the registration table it starts.
    offset: u32,
}

#[cfg(feature = "fault-injection")]
#[derive(Clone)]
struct FakeRing {
    state: Arc<Mutex<FakeRingState>>,
    /// The same notifier the real rings register their CQ eventfd with, so
    /// a completion wakes the worker through the path it actually uses. A
    /// bare fd here would be a second wake-up mechanism that only the fake
    /// exercises, which is the opposite of the point.
    notify: Arc<UringNotify>,
}

#[cfg(feature = "fault-injection")]
impl FakeRing {
    fn new(sq_capacity: usize, notify: Arc<UringNotify>) -> Self {
        FakeRing {
            state: Arc::new(Mutex::new(FakeRingState {
                sq_capacity,
                resident: Vec::new(),
                owned: Vec::new(),
                ready: Vec::new(),
                shutdown_completions: Vec::new(),
                shutdown_started: false,
                shutdown_delivered: 0,
                take_next: None,
                hold_submits: false,
                submit_error: None,
                submit_errors: Vec::new(),
                violations: Vec::new(),
                syncs: 0,
                dmabuf_extent: 1 << 30,
                registrations: Vec::new(),
                registration_failures: VecDeque::new(),
            })),
            notify,
        }
    }

    fn push(&self, sqe: SqeDescriptor) -> Result<(), ()> {
        let mut state = self.state.lock().unwrap();
        if state.resident.len() >= state.sq_capacity {
            return Err(());
        }
        state.resident.push(sqe);
        Ok(())
    }

    fn submit(&self) -> io::Result<usize> {
        let mut state = self.state.lock().unwrap();
        if let Some(errno) = state.submit_error.take() {
            state.submit_errors.push(errno);
            // A submit that reports an error took nothing. Leaving the
            // entries resident is the whole point: whether the worker then
            // reasons about them correctly is what a test is asking.
            return Err(io::Error::from_raw_os_error(errno));
        }
        // An instruction for one submit outranks the hold, so a test can
        // let exactly one partial take happen and have the ring stop again
        // afterwards -- which is what makes the suffix observable instead
        // of a race against the worker's next loop.
        let instructed = state.take_next.take();
        if instructed.is_none() && state.hold_submits {
            // Took nothing and says so. Everything stays resident, which is
            // what lets a second entry join the first.
            return Ok(0);
        }
        let available = state.resident.len();
        let taking = instructed.unwrap_or(available).min(available);
        let taken: Vec<SqeDescriptor> = state.resident.drain(..taking).collect();
        state.owned.extend(taken);
        Ok(taking)
    }

    fn synthetic_dmabuf_extent(&self) -> usize {
        self.state.lock().unwrap().dmabuf_extent
    }

    /// Record one synthetic registration, or refuse one that makes no sense.
    ///
    /// The descriptor validation and the registration map are built in the
    /// shared code above; this stands in only for the syscall at the end of
    /// it. A record here is explicitly synthetic and is never evidence that
    /// a kernel registered anything.
    fn register(&self, kind: &str, count: u32, offset: u32) -> io::Result<()> {
        let mut state = self.state.lock().unwrap();
        if state
            .registration_failures
            .front()
            .is_some_and(|(expected_kind, expected_offset, _)| {
                expected_kind == kind && *expected_offset == offset
            })
        {
            let (_, _, errno) = state.registration_failures.pop_front().unwrap();
            return Err(io::Error::from_raw_os_error(errno));
        }
        if kind == "unregister" {
            state.registrations.clear();
            state.registrations.push(FakeRegistration {
                kind: kind.to_string(),
                count,
                offset,
            });
            return Ok(());
        }
        if count == 0 {
            return Err(io::Error::from_raw_os_error(libc::EINVAL));
        }
        state.registrations.push(FakeRegistration {
            kind: kind.to_string(),
            count,
            offset,
        });
        Ok(())
    }

    fn complete_at_shutdown(&self, user_data: u64, result: i32) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        if state.shutdown_started
            || !state.owned.iter().any(|entry| entry.user_data == user_data)
            || state
                .shutdown_completions
                .iter()
                .any(|entry| entry.user_data == user_data)
        {
            let complaint = format!("cannot schedule shutdown completion for request {user_data}");
            state.violations.push(complaint.clone());
            return Err(complaint);
        }
        state
            .shutdown_completions
            .push(RingCompletion { user_data, result });
        Ok(())
    }

    fn begin_shutdown(&self) {
        let planned = {
            let mut state = self.state.lock().unwrap();
            state.shutdown_started = true;
            std::mem::take(&mut state.shutdown_completions)
        };
        for completion in planned {
            if self
                .complete(completion.user_data, completion.result)
                .is_ok()
            {
                self.state.lock().unwrap().shutdown_delivered += 1;
            }
        }
    }

    fn take_completions(&self) -> Vec<RingCompletion> {
        let mut state = self.state.lock().unwrap();
        std::mem::take(&mut state.ready)
    }

    fn submission_len(&self) -> usize {
        self.state.lock().unwrap().resident.len()
    }

    fn sync(&self) {
        self.state.lock().unwrap().syncs += 1;
    }

    /// Answer for one request the kernel took, and wake the worker.
    ///
    /// Refuses, loudly and in the violation list, to answer for a request
    /// it does not hold. That is the invariant the whole fault seam exists
    /// to protect: a completion for a request the kernel owns must come
    /// from the kernel, and one for a request it does not own is a
    /// fabrication.
    fn complete(&self, user_data: u64, result: i32) -> Result<(), String> {
        {
            let mut state = self.state.lock().unwrap();
            match state
                .owned
                .iter()
                .position(|held| held.user_data == user_data)
            {
                Some(at) => {
                    state.owned.remove(at);
                    state.ready.push(RingCompletion { user_data, result });
                }
                None => {
                    let resident = state
                        .resident
                        .iter()
                        .any(|entry| entry.user_data == user_data);
                    let complaint = format!(
                        "completion for request {user_data} which this ring \
                         does not hold (resident: {resident})"
                    );
                    state.violations.push(complaint.clone());
                    return Err(complaint);
                }
            }
        }
        // Through the notifier the real rings use, so the worker is woken
        // the way it is woken in production.
        self.notify.signal_producer();
        Ok(())
    }
}

// NVMe identify namespace data structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmeIdNs {
    nsze: u64,
    ncap: u64,
    nuse: u64,
    nsfeat: u8,
    nlbaf: u8,
    flbas: u8,
    mc: u8,
    dpc: u8,
    dps: u8,
    nmic: u8,
    rescap: u8,
    fpi: u8,
    dlfeat: u8,
    nawun: u16,
    nawupf: u16,
    nacwu: u16,
    nabsn: u16,
    nabo: u16,
    nabspf: u16,
    noiob: u16,
    nvmcap: [u8; 16],
    npwg: u16,
    npwa: u16,
    npdg: u16,
    npda: u16,
    nows: u16,
    mssrl: u16,
    mcl: u32,
    msrc: u8,
    rsvd81: [u8; 11],
    anagrpid: u32,
    rsvd96: [u8; 3],
    nsattr: u8,
    nvmsetid: u16,
    endgid: u16,
    nguid: [u8; 16],
    eui64: [u8; 8],
    lbaf: [NvmeLbaf; 64],
    vs: [u8; 3712],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmeLbaf {
    ms: u16,
    ds: u8,
    rp: u8,
}

// NVMe admin opcodes
const NVME_ADMIN_IDENTIFY: u8 = 0x06;

// NVMe identify CNS values
const NVME_IDENTIFY_CNS_NS: u32 = 0x00;

// NVMe I/O opcodes
const NVME_IO_READ: u8 = 0x02;
const NVME_IO_WRITE: u8 = 0x01;
const NVME_IO_MGMT_RECV: u8 = 0x12;

// NVMe uring command structure (80 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmeUringCmd {
    opcode: u8,
    flags: u8,
    rsvd1: u16,
    nsid: u32,
    cdw2: u32,
    cdw3: u32,
    metadata: u64,
    addr: u64,
    metadata_len: u32,
    data_len: u32,
    cdw10: u32,
    cdw11: u32,
    cdw12: u32,
    cdw13: u32,
    cdw14: u32,
    cdw15: u32,
    rsvd2: [u32; 4],
}

// Linux ioctl for NVMe admin command
// Defined in <linux/nvme_ioctl.h>: NVME_IOCTL_ADMIN_CMD _IOWR ('N', 0x41)
const NVME_IOCTL_ADMIN_CMD: libc::c_ulong = 0xC048_4E41;

// Defined in <linux/nvme_ioctl.h>: NVME_IOCTL_IO_CMD _IOWR ('N', 0x43)
const NVME_IOCTL_IO_CMD: libc::c_ulong = 0xC048_4E43;

// NVMe io_uring_cmd opcodes
const NVME_URING_CMD_IO: u32 = 0xC048_4E80;

// Linux ioctl for NVMe namespace ID
// Defined in <linux/nvme_ioctl.h>: NVME_IOCTL_ID _IO ('N', 0x40)
const NVME_IOCTL_ID: libc::c_ulong = 0x4e40;

// Linux ioctl for block device size in bytes.
// Defined in <linux/fs.h>: BLKGETSIZE64 _IOR(0x12,114,size_t)
const BLKGETSIZE64: libc::c_ulong = 0x8008_1272; // ioctl op to query block size

// Buffer protocol flags (from CPython C-API).
const PYBUF_WRITABLE: i32 = 0x0001; // buffer must be writable
const PYBUF_ND: i32 = 0x0008; // request N-dimensional buffer
const PYBUF_STRIDES: i32 = 0x0010 | PYBUF_ND; // request strides info
const PYBUF_ANY_CONTIGUOUS: i32 = 0x0080 | PYBUF_STRIDES; // accept any contiguous layout

// O_DIRECT is Linux-only; define a no-op fallback for other platforms.
#[cfg(target_os = "linux")]
const O_DIRECT: i32 = libc::O_DIRECT;
#[cfg(not(target_os = "linux"))]
const O_DIRECT: i32 = 0;
const RING_SIZE: usize = 256;

fn parse_use_iouring(io_engine: Option<String>, use_iouring: bool) -> PyResult<bool> {
    match io_engine {
        Some(engine) if !engine.is_empty() => match engine.as_str() {
            "posix" => Ok(false),
            "io_uring" => Ok(true),
            other => Err(PyValueError::new_err(format!(
                "io_engine must be one of posix, io_uring; got {other}"
            ))),
        },
        _ => Ok(use_iouring),
    }
}

///Per batch tracking for in flight I/O operation
type BatchTracking = (Arc<AtomicU64>, Arc<Condvar>);
type IoUringCompletionErrors = Vec<(usize, String)>;
type IoUringBatchResults = (Vec<bool>, IoUringCompletionErrors);

/// Round up to nearest multiple of alignment (required for O_DIRECT).
#[allow(clippy::manual_div_ceil)]
// Small helper used to align sizes for O_DIRECT I/O.
fn round_up(x: usize, align: usize) -> usize {
    (x + align - 1) / align * align
}

/// A regular read/write can be retried only after making positive progress.
///
/// A zero completion has no remaining-range advance. Retrying it would
/// resubmit the same request indefinitely.
fn is_retryable_regular_short_io(cqe_result: i32, len: usize, is_uring_cmd: bool) -> bool {
    cqe_result > 0 && (cqe_result as usize) < len && !is_uring_cmd
}

/// Move batch owners somewhere no allocator can reach them.
///
/// `batch_id: None` takes every outstanding batch. Moving handles between
/// Rust collections performs no reference-count operation, so this needs no
/// GIL and is safe to call from the worker thread.
///
/// The two locks are taken and released in sequence, never nested: the
/// worker holds neither when it calls this, and nesting them would be the
/// only ordering hazard here.
fn drain_batch_owners<T>(
    batched: &Mutex<HashMap<u64, Vec<T>>>,
    quarantined: &Mutex<Vec<T>>,
    batch_id: Option<u64>,
) -> usize {
    let taken: Vec<T> = {
        let mut batched = batched.lock().unwrap();
        match batch_id {
            Some(id) => batched.remove(&id).unwrap_or_default(),
            None => batched.drain().flat_map(|(_id, owners)| owners).collect(),
        }
    };
    let moved = taken.len();
    if moved > 0 {
        quarantined.lock().unwrap().extend(taken);
    }
    moved
}

/// A submit outcome a test asks the worker to see instead of the kernel's.
///
/// Reaching the fatal and partial-submit transitions otherwise needs a device
/// that fails on demand. This replaces what one submit call reports, before
/// the ring is consulted, so the worker runs its ordinary code against an
/// outcome it cannot otherwise be given. No completion is ever synthesised:
/// a request the kernel owns is never claimed to have finished.
#[cfg(feature = "fault-injection")]
#[derive(Clone, Copy, Debug)]
enum SubmitFault {
    /// The call reports taking nothing, and nothing was taken.
    ///
    /// This is the only count the seam can state truthfully. Substituting
    /// before the ring means no entry was ever offered, so every entry is
    /// still resident and still the worker's -- which is exactly what a
    /// real submit returning zero leaves behind.
    ReportsZeroTaken,
    /// Retryable: the ring is unchanged and the work is still ours.
    Retryable(i32),
    /// Fatal: what the kernel took is unknowable from here.
    Fatal(i32),
}

/// Faults keyed by which submit call they apply to.
///
/// Keying on the call ordinal rather than on time is what makes a plan
/// reproducible: a test names the third submit, and gets the third submit,
/// however fast or slow the run is.
#[cfg(feature = "fault-injection")]
#[derive(Default)]
struct FaultPlan {
    submits: Mutex<HashMap<u64, SubmitFault>>,
    submit_calls: AtomicU64,
}

#[cfg(feature = "fault-injection")]
impl FaultPlan {
    /// Take the fault for this submit call, if the plan names one.
    fn take_submit_fault(&self) -> Option<SubmitFault> {
        let ordinal = self.submit_calls.fetch_add(1, Ordering::SeqCst);
        self.submits.lock().unwrap().remove(&ordinal)
    }
}

/// Submit the ring, or report what a plan says this call reports.
fn submit_ring(
    ring: &IoUringWrapper,
    #[cfg(feature = "fault-injection")] plan: &Option<Arc<FaultPlan>>,
) -> io::Result<usize> {
    #[cfg(feature = "fault-injection")]
    if let Some(plan) = plan {
        if let Some(fault) = plan.take_submit_fault() {
            // Deliberately before the ring: the entries stay exactly where
            // they were, so the worker's own bookkeeping is what is under
            // test and the kernel is never told anything.
            return match fault {
                SubmitFault::ReportsZeroTaken => Ok(0),
                SubmitFault::Retryable(errno) | SubmitFault::Fatal(errno) => {
                    Err(io::Error::from_raw_os_error(errno))
                }
            };
        }
    }
    ring.submit()
}

/// What to do about a completion that moved fewer bytes than were asked for.
#[derive(Debug, PartialEq, Eq)]
enum ShortIoAction {
    /// Not short, or not a case this engine retries: take the result as it is.
    Complete,
    /// Push the remainder as a new submission for the same operation.
    Retry,
    /// End the operation now, with an error, pushing nothing.
    FailTerminally,
}

/// Decide a short completion before anything is pushed.
///
/// A fixed submission addresses its buffer as a byte offset into a dma-buf
/// registered with the ring rather than as a process address. Retrying the
/// remainder means re-expressing that offset, and getting it wrong sends the
/// rest of a transfer to the start of the registered buffer. This engine
/// does not carry that second addressing path, so such an operation ends at
/// its completion instead.
fn short_io_action(
    cqe_result: i32,
    len: usize,
    is_uring_cmd: bool,
    has_fixed_dmabuf: bool,
) -> ShortIoAction {
    if !is_retryable_regular_short_io(cqe_result, len, is_uring_cmd) {
        return ShortIoAction::Complete;
    }
    if has_fixed_dmabuf {
        return ShortIoAction::FailTerminally;
    }
    ShortIoAction::Retry
}

// Fetch errno for the last libc call on this thread.
fn errno() -> i32 {
    // SAFETY: libc call.
    #[cfg(target_os = "linux")]
    unsafe {
        *libc::__errno_location()
    }
    #[cfg(target_os = "macos")]
    unsafe {
        *libc::__error()
    }
}

// Convert errno to a Python OSError with a message.
fn os_err(msg: &str) -> PyErr {
    PyOSError::new_err((errno(), msg.to_string()))
}

// Convert an NVMe passthrough ioctl return value to a Python result.
// Negative values are syscall errors reported through errno, while positive
// values are NVMe command completion status codes returned by the kernel.
fn check_nvme_ioctl_result(rc: libc::c_int, msg: &str) -> Result<(), PyErr> {
    if rc < 0 {
        return Err(os_err(msg));
    }
    if rc > 0 {
        return Err(PyRuntimeError::new_err(format!(
            "{msg}: NVMe status 0x{rc:x}"
        )));
    }
    Ok(())
}

// Low-level write loop that retries until all bytes are written.
// This isolates the raw syscalls from Python-facing logic.
fn pwrite_from_ptr(
    fd: RawFd,
    mut offset: u64,
    mut ptr: *const u8,
    mut len: usize,
) -> Result<(), PyErr> {
    while len > 0 {
        // SAFETY: caller guarantees ptr is valid for len bytes.
        let chunk = unsafe { slice::from_raw_parts(ptr, len) };
        let n = unsafe {
            libc::pwrite(
                fd,
                chunk.as_ptr() as *const libc::c_void,
                chunk.len(),
                offset as libc::off_t,
            )
        };
        if n < 0 {
            return Err(os_err("pwrite failed"));
        }
        let n = n as usize;
        offset += n as u64;
        // SAFETY: advance ptr by n bytes.
        unsafe {
            ptr = ptr.add(n);
        }
        len -= n;
    }
    Ok(())
}

// Low-level read loop that retries until all bytes are read.
// We treat EOF as an error because the caller expects a full read.
fn pread_into(fd: RawFd, offset: u64, mut dst: *mut u8, mut size: usize) -> Result<(), PyErr> {
    let mut off = offset;
    while size > 0 {
        // SAFETY: pread writes into dst for size bytes.
        let n = unsafe { libc::pread(fd, dst as *mut libc::c_void, size, off as libc::off_t) };
        if n < 0 {
            return Err(os_err("pread failed"));
        }
        if n == 0 {
            return Err(PyRuntimeError::new_err("unexpected EOF"));
        }
        let n = n as usize;
        // SAFETY: advance dst by n bytes.
        unsafe {
            dst = dst.add(n);
        }
        off += n as u64;
        size -= n;
    }
    Ok(())
}

// Determine file/device size in bytes (ioctl for block device, fstat fallback).
fn fd_size_bytes(fd: RawFd) -> Result<u64, PyErr> {
    // Try ioctl first (block device / loop device).
    let mut size: u64 = 0;
    // SAFETY: ioctl expects pointer to u64 for BLKGETSIZE64.
    let rc = unsafe { libc::ioctl(fd, BLKGETSIZE64, &mut size as *mut u64) };
    if rc == 0 {
        return Ok(size);
    }

    // Fallback to fstat for regular files.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let rc2 = unsafe { libc::fstat(fd, &mut st as *mut libc::stat) };
    if rc2 != 0 {
        return Err(os_err("fstat failed"));
    }
    Ok(st.st_size as u64)
}

// Linux FIEMAP UAPI; the bounded array follows the fixed 32-byte header.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FileExtent {
    logical: u64,
    physical: u64,
    length: u64,
    reserved64: [u64; 2],
    flags: u32,
    reserved: [u32; 3],
}

#[repr(C)]
struct FileExtentMap {
    start: u64,
    length: u64,
    flags: u32,
    mapped_extents: u32,
    extent_count: u32,
    reserved: u32,
    extents: [FileExtent; 128],
}

fn validate_initialized_file(fd: RawFd, capacity: u64) -> PyResult<()> {
    let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
    // SAFETY: fs points to a writable statfs with the platform ABI.
    if unsafe { libc::fstatfs(fd, &mut fs) } != 0 {
        return Err(os_err("fstatfs failed"));
    }
    if fs.f_type != 0x58465342 && fs.f_type != 0xef53 {
        return Err(PyValueError::new_err(
            "strict DMA-BUF files require XFS or ext4",
        ));
    }
    let mut covered = 0;
    while covered < capacity {
        let mut map = FileExtentMap {
            start: covered,
            length: capacity - covered,
            flags: 1, // FIEMAP_FLAG_SYNC: resolve pending allocation first.
            mapped_extents: 0,
            extent_count: 128,
            reserved: 0,
            extents: [FileExtent::default(); 128],
        };
        // SAFETY: FS_IOC_FIEMAP receives the UAPI header and 128 extent slots.
        if unsafe { libc::ioctl(fd, 0xc020660b as libc::c_ulong, &mut map) } != 0 {
            return Err(os_err("FIEMAP failed"));
        }
        if map.mapped_extents == 0 {
            return Err(PyValueError::new_err(
                "strict DMA-BUF file contains a hole in the cache range",
            ));
        }
        for extent in &map.extents[..map.mapped_extents as usize] {
            // LAST is descriptive. Other flags identify uninitialized,
            // shared, encoded or otherwise unsupported representations.
            if extent.flags & !1 != 0 {
                return Err(PyValueError::new_err(format!(
                    "strict DMA-BUF file requires initialized private extents; flags={:#x}",
                    extent.flags
                )));
            }
            let end = extent.logical.checked_add(extent.length).ok_or_else(|| {
                PyValueError::new_err("FIEMAP extent exceeds the file offset range")
            })?;
            if extent.logical > covered || end <= covered {
                return Err(PyValueError::new_err(
                    "strict DMA-BUF file contains a hole in the cache range",
                ));
            }
            covered = end.min(capacity);
            if covered == capacity {
                break;
            }
        }
    }
    Ok(())
}

// NVMe helper functions for io_uring command support

// Calculate NVMe namespace size in bytes from identify namespace data
fn nvme_ns_size_bytes(id_ns: &NvmeIdNs, lba_size: u32) -> u64 {
    id_ns.nsze * lba_size as u64
}

/// Check if device path is a character device (e.g., /dev/ng0n1)
fn is_character_device(path: &str) -> Result<bool, PyErr> {
    let cpath = CString::new(path).map_err(|_| PyValueError::new_err("path contains NUL"))?;

    // SAFETY: stat call
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::stat(cpath.as_ptr(), &mut st as *mut libc::stat) };

    if rc != 0 {
        return Err(os_err("stat failed"));
    }

    Ok((st.st_mode & libc::S_IFMT) == libc::S_IFCHR)
}

/// Get namespace ID from NVMe device using ioctl.
fn nvme_get_nsid_from_fd(fd: RawFd) -> Result<u32, PyErr> {
    //SAFETY: ioctl call with request that returns an integer
    let ret = unsafe { libc::ioctl(fd, NVME_IOCTL_ID) };
    if ret < 0 {
        return Err(os_err("Failed to get namespace ID via ioctl"));
    }
    Ok(ret as u32)
}

/// Get LBA shift (log2 of LBA size) from identify namespace data
fn nvme_get_lba_shift(id_ns: &NvmeIdNs) -> Result<u32, PyErr> {
    // Extract LBA format index from FLBAS

    let lbaf_index = if id_ns.nlbaf < 16 {
        (id_ns.flbas & 0x0F) as usize
    } else {
        let lsb = (id_ns.flbas & 0x0F) as usize;
        let msb = ((id_ns.flbas >> 5) & 0x03) as usize;
        lsb + (msb << 4)
    };

    if lbaf_index >= 64 {
        return Err(PyValueError::new_err("Invalid LBA format index"));
    }

    // Get LBA data size from LBAF
    let ds = id_ns.lbaf[lbaf_index].ds;
    if ds == 0 {
        return Err(PyValueError::new_err("Invalid LBA data size"));
    }

    // Check for metadata support
    let ms = id_ns.lbaf[lbaf_index].ms;
    if ms != 0 {
        return Err(PyValueError::new_err(
            "Device is formatted with metadata, can't be supported.",
        ));
    }

    Ok(ds as u32)
}

/// Get LBA size in bytes from identify namespace data
fn nvme_get_lba_size(id_ns: &NvmeIdNs) -> Result<u32, PyErr> {
    let lba_shift = nvme_get_lba_shift(id_ns)?;
    Ok(1u32 << lba_shift)
}

/// NVMe passthrough command structure for ioctl
#[repr(C)]
struct NvmePassthruCmd {
    opcode: u8,
    flags: u8,
    rsvd1: u16,
    nsid: u32,
    cdw2: u32,
    cdw3: u32,
    metadata: u64,
    addr: u64,
    metadata_len: u32,
    data_len: u32,
    cdw10: u32,
    cdw11: u32,
    cdw12: u32,
    cdw13: u32,
    cdw14: u32,
    cdw15: u32,
    timeout_ms: u32,
    result: u32,
}

// NVMe FDP (Flexible Data Placement) reclaim unit handle status descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmeFdpRuhStatusDesc {
    pid: u16,
    ruhid: u16,
    earutr: u32,
    ruamw: u64,
    rsvd16: [u8; 16],
}

// NVMe FDP reclaim unit handle status header.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmeFdpRuhStatus {
    rsvd0: [u8; 14],
    nruhsd: u16,
}

/// Send NVMe identify namespace command via ioctl
fn nvme_identify_ns(fd: RawFd, nsid: u32) -> Result<NvmeIdNs, PyErr> {
    let mut id_ns: NvmeIdNs = unsafe { std::mem::zeroed() };

    let cmd = NvmePassthruCmd {
        opcode: NVME_ADMIN_IDENTIFY,
        nsid,
        addr: &mut id_ns as *mut NvmeIdNs as u64,
        data_len: std::mem::size_of::<NvmeIdNs>() as u32,
        cdw10: NVME_IDENTIFY_CNS_NS,
        timeout_ms: 0,
        ..unsafe { std::mem::zeroed() }
    };

    // SAFETY: ioctl with properly initialized command structure
    let rc = unsafe { libc::ioctl(fd, NVME_IOCTL_ADMIN_CMD, &cmd as *const NvmePassthruCmd) };

    check_nvme_ioctl_result(rc, "NVMe identify namespace ioctl failed")?;

    Ok(id_ns)
}

/// Send NVMe I/O management receive command to fetch FDP RUH status.
fn nvme_fdp_reclaim_unit_handle_status(
    fd: RawFd,
    nsid: u32,
    data_len: u32,
    data: *mut u8,
) -> Result<(), PyErr> {
    let mut cmd = NvmePassthruCmd {
        opcode: NVME_IO_MGMT_RECV,
        nsid,
        addr: data as u64,
        data_len,
        cdw10: 1,
        cdw11: (data_len >> 2) - 1,
        timeout_ms: 0,
        result: 0,
        ..unsafe { std::mem::zeroed() }
    };

    // SAFETY: ioctl with properly initialized command structure.
    let rc = unsafe { libc::ioctl(fd, NVME_IOCTL_IO_CMD, &mut cmd as *mut NvmePassthruCmd) };

    check_nvme_ioctl_result(rc, "NVMe FDP reclaim unit handle status ioctl failed")?;

    Ok(())
}

/// Fetch FDP status descriptors as (placement identifier, RUH identifier).
fn fetch_fdp_status(fd: RawFd, nsid: u32) -> Result<Vec<(u16, u16)>, PyErr> {
    let header_size = std::mem::size_of::<NvmeFdpRuhStatus>();
    let desc_size = std::mem::size_of::<NvmeFdpRuhStatusDesc>();
    let mut header: NvmeFdpRuhStatus = unsafe { std::mem::zeroed() };

    // Header-first query matches nvme-cli behavior and avoids assuming a
    // controller-specific descriptor count.
    nvme_fdp_reclaim_unit_handle_status(
        fd,
        nsid,
        header_size as u32,
        (&mut header as *mut NvmeFdpRuhStatus).cast::<u8>(),
    )?;

    let actual_ruhs = u16::from_le(header.nruhsd) as usize;
    let desc_bytes = actual_ruhs
        .checked_mul(desc_size)
        .ok_or_else(|| PyRuntimeError::new_err("NVMe FDP status descriptor size overflow"))?;
    let bytes = header_size
        .checked_add(desc_bytes)
        .ok_or_else(|| PyRuntimeError::new_err("NVMe FDP status payload size overflow"))?;
    let data_len = u32::try_from(bytes)
        .map_err(|_| PyRuntimeError::new_err("NVMe FDP status payload too large"))?;

    let mut buffer: Vec<u8> = Vec::new();
    buffer.try_reserve_exact(bytes).map_err(|e| {
        PyMemoryError::new_err(format!("failed to allocate NVMe FDP status buffer: {e}"))
    })?;
    buffer.resize(bytes, 0);
    nvme_fdp_reclaim_unit_handle_status(fd, nsid, data_len, buffer.as_mut_ptr())?;

    let mut status: Vec<(u16, u16)> = Vec::new();
    status.try_reserve_exact(actual_ruhs).map_err(|e| {
        PyMemoryError::new_err(format!("failed to allocate NVMe FDP status entries: {e}"))
    })?;
    for i in 0..actual_ruhs {
        let desc_ptr = unsafe {
            buffer
                .as_ptr()
                .add(header_size + i * desc_size)
                .cast::<NvmeFdpRuhStatusDesc>()
        };
        let desc = unsafe { std::ptr::read_unaligned(desc_ptr) };
        status.push((u16::from_le(desc.pid), u16::from_le(desc.ruhid)));
    }

    Ok(status)
}

fn placement_id_to_u16(pid: i32) -> PyResult<u16> {
    if pid == 0 {
        return Err(PyValueError::new_err(
            "placement_id must not be 0; use None to omit the FDP directive",
        ));
    }
    u16::try_from(pid).map_err(|_| PyValueError::new_err("placement_id must be in range 1..=65535"))
}

/// Prepare NVMe uring command for read/write operations
#[allow(clippy::too_many_arguments)]
fn nvme_uring_cmd_prep(
    cmd: &mut NvmeUringCmd,
    is_write: bool,
    nsid: u32,
    offset: u64,
    len: usize,
    lba_shift: u32,
    ptr: *const u8,
    dtype: u8,
    dspec: u16,
) -> Result<(), PyErr> {
    let lba_size = 1usize << lba_shift;

    // Validate offset alignment
    if !(offset as usize).is_multiple_of(lba_size) {
        return Err(PyValueError::new_err(format!(
            "offset must be aligned to LBA size ({} bytes), got offset={}",
            lba_size, offset
        )));
    }

    // Validate length alignment
    if !len.is_multiple_of(lba_size) {
        return Err(PyValueError::new_err(format!(
            "length must be aligned to LBA size ({} bytes), got len={}",
            lba_size, len
        )));
    }

    // Validate non-zero length
    if len == 0 {
        return Err(PyValueError::new_err("length must be non-zero"));
    }

    // Calculate SLBA (Starting LBA) and NLB (Number of LBAs)
    let slba = offset >> lba_shift;
    let nlb = (len >> lba_shift) - 1; // NLB is 0-based

    // Validate NLB fits in NVMe field (16 bits, max 0xFFFF)
    if nlb > 0xFFFF {
        return Err(PyValueError::new_err(format!(
            "NLB ({}) exceeds NVMe field maximum (65535)",
            nlb
        )));
    }

    // Set opcode
    cmd.opcode = if is_write {
        NVME_IO_WRITE
    } else {
        NVME_IO_READ
    };
    cmd.nsid = nsid;

    // Set SLBA in cdw10 and cdw11
    cmd.cdw10 = (slba & 0xFFFFFFFF) as u32;
    cmd.cdw11 = (slba >> 32) as u32;

    // Set NLB in cdw12 (bits 0-15) and dtype in bits 20-23
    cmd.cdw12 = nlb as u32 | ((dtype as u32) << 20);

    // Set dspec in cdw13 bits 16-31
    cmd.cdw13 = (dspec as u32) << 16;

    // Set data address and length
    cmd.addr = ptr as u64;
    cmd.data_len = len as u32;

    // No metadata support for now
    cmd.metadata = 0;
    cmd.metadata_len = 0;

    Ok(())
}

/// NVMe command data for io_uring_cmd submissions.
///
/// This structure contains NVMe-specific information needed for
/// passthrough commands via io_uring_cmd.
#[derive(Clone, Debug)]
struct NvmeCmdData {
    nsid: u32,      // Namespace ID
    lba_shift: u32, // LBA shift (log2 of LBA size)
    dtype: u8,      // Directive Type
    dspec: u16,     // Directive Specific
}

/// Aligned buffer for O_DIRECT I/O.
/// Allocated with posix_memalign so the pointer satisfies alignment requirements.
/// Automatically freed on drop.
struct AlignedBuf {
    ptr: *mut u8,
    #[allow(dead_code)]
    len: usize,
    #[allow(dead_code)]
    align: usize,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    // Allocate an aligned buffer suitable for O_DIRECT.
    fn new(len: usize, align: usize) -> Result<Self, PyErr> {
        let mut p: *mut libc::c_void = std::ptr::null_mut();
        // SAFETY: posix_memalign writes to p.
        let rc = unsafe { libc::posix_memalign(&mut p as *mut *mut libc::c_void, align, len) };
        if rc != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "posix_memalign failed rc={rc}"
            )));
        }
        if p.is_null() {
            return Err(PyRuntimeError::new_err("posix_memalign returned null"));
        }
        Ok(Self {
            ptr: p as *mut u8,
            len,
            align,
        })
    }

    // Mutable pointer for read/write syscalls.
    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    // Const pointer for write syscalls.
    fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                libc::free(self.ptr as *mut libc::c_void);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Source description for one prepared io_uring write.
///
/// Fields:
/// - `ptr_addr`: Address to submit, either the caller buffer or the bounce
/// - `bounce`: Bounce buffer that must stay alive until the write completes
/// - `fixed_buffer_idx`: Registered fixed-buffer index, cleared when bouncing
struct PreparedWriteBuffer {
    ptr_addr: usize,
    bounce: Option<Arc<AlignedBuf>>,
    fixed_buffer_idx: Option<u16>,
    fixed_dmabuf: Option<usize>,
}

// Report whether `len` bytes starting at `ptr_addr + offset` are all zero.
fn buffer_range_is_zero(ptr_addr: usize, offset: usize, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    let ptr = (ptr_addr as *const u8).wrapping_add(offset);
    // SAFETY: callers only pass ranges inside the buffer they hold, so
    // [offset, offset + len) is readable for the lifetime of this call.
    unsafe {
        slice::from_raw_parts(ptr, len)
            .iter()
            .all(|byte| *byte == 0)
    }
}

// Prepare one regular io_uring write so that `total_len` bytes can be submitted
// while reading only within the source buffer bounds.
//
// Host padding [payload_len, total_len) is written as zeroes. The
// caller buffer is submitted directly when it can already satisfy that, which
// keeps the fixed-buffer zero-copy path. A bounce buffer is used when the
// source is shorter than `total_len`, its padding tail is not already zero, or
// O_DIRECT requires an aligned address. The source buffer is never modified.
#[allow(clippy::too_many_arguments)]
fn prepare_iouring_write_buffer(
    ptr_addr: usize,
    cap: usize,
    payload_len: usize,
    total_len: usize,
    use_odirect: bool,
    alignment: usize,
    fixed_buffer_idx: Option<u16>,
    fixed_dmabuf: Option<usize>,
) -> PyResult<PreparedWriteBuffer> {
    if cap < payload_len {
        return Err(PyValueError::new_err(format!(
            "input buffer too small: cap={cap} need={payload_len}"
        )));
    }
    if total_len < payload_len {
        return Err(PyValueError::new_err("total_len must be >= payload_len"));
    }

    // Registered device memory must never be read by the CPU. Its physical
    // slot, including padding, belongs to the caller; only the logical payload
    // is exposed on restore. Host buffers retain the zero-padding contract.
    if fixed_dmabuf.is_some() {
        if cap < total_len || (use_odirect && !ptr_addr.is_multiple_of(alignment)) {
            return Err(PyValueError::new_err(
                "dma-buf registered buffer must be aligned and at least total_len bytes",
            ));
        }
        return Ok(PreparedWriteBuffer {
            ptr_addr,
            bounce: None,
            fixed_buffer_idx,
            fixed_dmabuf,
        });
    }

    let needs_alignment_bounce = use_odirect && !ptr_addr.is_multiple_of(alignment);
    let needs_capacity_bounce = cap < total_len;
    let has_padding = payload_len < total_len;
    let tail_is_zero = !has_padding
        || (!needs_capacity_bounce
            && buffer_range_is_zero(ptr_addr, payload_len, total_len - payload_len));
    let needs_zero_bounce = has_padding && !tail_is_zero;
    let needs_bounce = needs_alignment_bounce || needs_capacity_bounce || needs_zero_bounce;

    if !needs_bounce {
        return Ok(PreparedWriteBuffer {
            ptr_addr,
            bounce: None,
            fixed_buffer_idx,
            fixed_dmabuf: None,
        });
    }

    let bounce = AlignedBuf::new(total_len, alignment)?;
    let bounce_ptr = bounce.as_mut_ptr();
    // SAFETY: the bounce holds total_len bytes and cap >= payload_len was
    // checked above, so both the copy and the zero-fill stay in bounds.
    unsafe {
        if payload_len > 0 {
            std::ptr::copy_nonoverlapping(ptr_addr as *const u8, bounce_ptr, payload_len);
        }
        if total_len > payload_len {
            std::ptr::write_bytes(bounce_ptr.add(payload_len), 0u8, total_len - payload_len);
        }
    }
    let bounce_arc = Arc::new(bounce);
    Ok(PreparedWriteBuffer {
        ptr_addr: bounce_arc.as_ptr() as usize,
        bounce: Some(bounce_arc),
        fixed_buffer_idx: None,
        fixed_dmabuf: None,
    })
}

/// A borrowed view of the bytes behind a Python object.
///
/// Two kinds of object reach the engine: ordinary buffers (bytes, memoryview,
/// CPU tensors) that implement the buffer protocol, and device tensors that
/// do not, because their memory is not host-addressable in the buffer
/// protocol sense.  A device tensor is accepted through the `data_ptr()` /
/// `nbytes` interface torch exposes: the pointer is only ever used as an
/// address the kernel resolves through a registered buffer (a dma-buf), never
/// dereferenced here, so the bounce paths refuse it (see `fixed_dmabuf`).
struct BufRef {
    view: Option<pyo3::ffi::Py_buffer>,
    ptr: *mut u8,
    len: usize,
    readonly: bool,
    host_accessible: bool,
}

/// Keep an export, not just the exporter, while a batched I/O owns its address.
/// A reference to bytearray alone does not prevent resize from freeing memory.
fn batch_buffer_owner(obj: &Bound<'_, PyAny>, host_accessible: bool) -> PyResult<Py<PyAny>> {
    if host_accessible {
        Ok(PyMemoryView::from(obj)?.into_any().unbind())
    } else {
        Ok(obj.clone().unbind())
    }
}

impl BufRef {
    fn release(self) {
        if let Some(mut view) = self.view {
            // SAFETY: view was created by PyObject_GetBuffer.
            unsafe { pyo3::ffi::PyBuffer_Release(&mut view) };
        }
    }
}

// Acquire the bytes behind `obj`: a buffer-protocol view with the requested
// mutability, or, for an object without one that has `data_ptr()` and
// `nbytes` (a torch tensor on any device), the address it reports.
fn get_buffer<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    writable: bool,
) -> Result<BufRef, PyErr> {
    // SAFETY: PyObject_GetBuffer follows CPython buffer protocol.
    unsafe {
        let mut view: pyo3::ffi::Py_buffer = std::mem::zeroed();
        // Request a contiguous byte-view. This lets Rust issue a single syscall
        // against a flat pointer instead of handling Python strides/shapes.
        let flags = if writable {
            PYBUF_WRITABLE | PYBUF_ANY_CONTIGUOUS
        } else {
            PYBUF_ANY_CONTIGUOUS
        };
        let rc = pyo3::ffi::PyObject_GetBuffer(obj.as_ptr(), &mut view, flags);
        if rc == 0 {
            return Ok(BufRef {
                ptr: view.buf as *mut u8,
                len: view.len as usize,
                readonly: view.readonly != 0,
                host_accessible: true,
                view: Some(view),
            });
        }
        let err = PyErr::fetch(py);
        if !obj.hasattr("data_ptr")? || !obj.hasattr("nbytes")? {
            return Err(err);
        }
        if let Ok(is_contig) = obj.call_method0("is_contiguous") {
            if !is_contig.extract::<bool>().unwrap_or(true) {
                return Err(PyValueError::new_err(
                    "tensor must be contiguous for raw block I/O",
                ));
            }
        }
        let ptr = obj.call_method0("data_ptr")?.extract::<usize>()?;
        let len = obj.getattr("nbytes")?.extract::<usize>()?;
        Ok(BufRef {
            view: None,
            ptr: ptr as *mut u8,
            len,
            readonly: false,
            host_accessible: false,
        })
    }
}

/// Completion primitive for synchronous io_uring operations.
///
/// When a Python call needs to wait for an I/O operation to complete,
/// we use this primitive. The worker thread will call `set()` with the
/// result, and the caller calls `wait()` to block until the result is ready.
///
/// Fields:
/// - `result`: Stores the completion status (Ok or Err)
/// - `cvar`: Condition variable for signaling when result is available
struct IoCompletion {
    result: Mutex<Option<PyResult<()>>>,
    cvar: Condvar,
}

impl IoCompletion {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            cvar: Condvar::new(),
        }
    }
    fn set(&self, r: PyResult<()>) {
        let mut guard = self
            .result
            .lock()
            .expect("IoCompletion: mutex poisoned in set()");
        *guard = Some(r);
        self.cvar.notify_one();
    }
    fn wait(&self) -> PyResult<()> {
        let mut guard = self
            .result
            .lock()
            .expect("IoCompletion: mutex poisoned in wait()");
        while guard.is_none() {
            guard = self
                .cvar
                .wait(guard)
                .expect("IoCompletion: condition variable wait failed");
        }
        guard.take().unwrap()
    }
}

// Convert a batch's completion objects into a success bitmap and sparse errors.
fn collect_iouring_completion_results(
    completions: Option<Vec<Arc<IoCompletion>>>,
) -> IoUringBatchResults {
    let Some(completions) = completions else {
        return (Vec::new(), Vec::new());
    };

    let mut results = Vec::with_capacity(completions.len());
    let mut errors = Vec::new();
    for (operation_index, completion) in completions.iter().enumerate() {
        match completion.wait() {
            Ok(()) => results.push(true),
            Err(error) => {
                results.push(false);
                errors.push((operation_index, error.to_string()));
            }
        }
    }
    (results, errors)
}

/// Manages io_uring worker thread notification, using one `epoll` instance
/// over two eventfds to wait on two event sources at once: a producer-side
/// eventfd signalled when Python pushes a new submission, and a CQ-side
/// eventfd signalled by the kernel when a completion is posted.
struct UringNotify {
    /// Epoll instance watching both `producer_efd` and `cq_efd`. A single
    /// `epoll_wait` on this fd blocks until either eventfd becomes readable,
    /// so the worker can react to user-space queue pushes and kernel CQE
    /// posts from the same call site.
    epoll_fd: RawFd,
    /// Eventfd written by Python producer threads after pushing into the
    /// submission queue. Replaces the `Condvar::notify_one` used in the
    /// pre-eventfd design. Also written by `do_close` so the worker can
    /// break out of `epoll_wait` and observe the shutdown flag.
    producer_efd: RawFd,
    /// Eventfd registered with the io_uring instance via
    /// `Submitter::register_eventfd`. The kernel writes to it whenever a
    /// CQE is posted, so the worker is woken without having to drain the
    /// completion queue speculatively.
    cq_efd: RawFd,
}

impl UringNotify {
    /// Builds the three fds (two eventfds + one epoll) and wires the
    /// eventfds into the epoll instance. Cleans up partially-built state
    /// on any error path so no fd leaks if construction fails midway.
    fn new() -> io::Result<Self> {
        // Producer-side eventfd. Counter starts at 0 (no pending events).
        // EFD_CLOEXEC prevents leaking the fd into a child process via
        // execve. EFD_NONBLOCK makes read() return EAGAIN (instead of
        // blocking) when the counter is already 0; wait() relies on this
        // non-blocking behaviour to drain safely without hanging.
        let producer_efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if producer_efd < 0 {
            // Nothing else has been allocated yet -- just bubble the error.
            return Err(io::Error::last_os_error());
        }

        // CQ-side eventfd. The kernel writes to this one after
        // register_eventfd() is called on the io_uring instance.
        let cq_efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if cq_efd < 0 {
            // producer_efd is live -- close it before bubbling the error.
            let e = io::Error::last_os_error();
            unsafe { libc::close(producer_efd) };
            return Err(e);
        }

        // Epoll instance fd. EPOLL_CLOEXEC for the same hygiene reason as
        // EFD_CLOEXEC above.
        let epoll_fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        if epoll_fd < 0 {
            // Both eventfds are live; close them before returning.
            let e = io::Error::last_os_error();
            unsafe {
                libc::close(producer_efd);
                libc::close(cq_efd);
            }
            return Err(e);
        }

        // Register both eventfds with the epoll instance so epoll_wait
        // can block on either source.
        //
        // We use EPOLLIN: wake when the counter becomes non-zero
        // (someone signalled). Alternatives we deliberately don't use:
        //   - EPOLLOUT (writable): pointless -- eventfd is effectively
        //     always writable, so it would just busy-loop epoll_wait.
        //   - EPOLLET (edge-triggered): would require fully draining
        //     every wake-up in one go to avoid missing edges. Default
        //     level-triggered fits our drain pattern in wait().
        //   - EPOLLONESHOT: would auto-disarm after each fire and force
        //     us to re-register; not worth the complexity.
        // u64 stores the fd value itself so wait() can identify which
        // fd fired without keeping a side map.
        for fd in [producer_efd, cq_efd] {
            let mut ev = libc::epoll_event {
                events: libc::EPOLLIN as u32,
                u64: fd as u64,
            };
            let rc = unsafe { libc::epoll_ctl(epoll_fd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
            if rc < 0 {
                // All three fds are live; clean up all of them.
                let e = io::Error::last_os_error();
                unsafe {
                    libc::close(epoll_fd);
                    libc::close(producer_efd);
                    libc::close(cq_efd);
                }
                return Err(e);
            }
        }

        // Ownership of all three fds moves into Self; Drop handles the
        // happy-path close.
        Ok(Self {
            epoll_fd,
            producer_efd,
            cq_efd,
        })
    }

    /// Wakes the worker thread by writing 1 to `producer_efd`. Called by
    /// producers after pushing a submission into the queue and by `do_close`
    /// to break the worker out of `epoll_wait` for shutdown.
    fn signal_producer(&self) {
        let v: u64 = 1;
        // Given EFD_NONBLOCK and counter < u64::MAX - 1, the 8-byte write
        // always succeeds, so the return value is intentionally ignored.
        unsafe {
            libc::write(
                self.producer_efd,
                &v as *const u64 as *const libc::c_void,
                8,
            );
        }
    }

    /// Blocks until an eventfd is readable or the optional timeout expires, then drains
    /// each fired fd. Drain is required because epoll is level-triggered:
    /// without consuming the counter, the next epoll_wait would return
    /// immediately on the same already-handled signal.
    fn wait(&self, timeout: Option<Duration>) {
        // A capacity of 2 is enough: only two fds are registered with this
        // epoll instance, so at most two events can come back per call.
        let mut events = [libc::epoll_event { events: 0, u64: 0 }; 2];

        let timeout_ms = timeout.map_or(-1, |delay| {
            delay.as_millis().max(1).min(i32::MAX as u128) as i32
        });
        let event_count =
            unsafe { libc::epoll_wait(self.epoll_fd, events.as_mut_ptr(), 2, timeout_ms) };
        if event_count <= 0 {
            return;
        }

        // For each event reported, read 8 bytes from the corresponding
        // eventfd to reset its counter to 0. The fd value was stashed in
        // ev.u64 during epoll_ctl registration. We discard the read value
        // (we only care that a signal arrived, not how many).
        let mut buf = [0u8; 8];
        for ev in &events[..event_count as usize] {
            let fd = ev.u64 as RawFd;
            // Discard the result. The wake-up was already delivered by
            // epoll_wait; this read only exists to reset the eventfd
            // counter so epoll stops reporting the fd as readable. If it
            // fails, the worst case is one spurious wake-up next
            // iteration. No work is lost, because the real submissions
            // live in the queue and CQ, not in the bytes we just read.
            unsafe {
                libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, 8);
            }
        }
    }
}

impl Drop for UringNotify {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.epoll_fd);
            libc::close(self.producer_efd);
            libc::close(self.cq_efd);
        }
    }
}

/// Represents a single I/O submission to io_uring.
///
/// This struct is sent from Python threads to the worker thread via a queue.
/// It contains all information needed to perform the I/O operation.
///
/// Fields:
/// - `fd`: File descriptor for the block device
/// - `offset`: Byte offset on the device to read/write
/// - `len`: Number of bytes to transfer
/// - `ptr_addr`: Memory address of the buffer (as usize for Send)
/// - `is_write`: true for write, false for read
/// - `completion`: Shared completion primitive for signaling result
/// - `fixed_buffer_idx`: Index into registered fixed buffers (if using zero-copy)
/// - `bounce`: optional bounce buffer when O_DIRECT requires alignment.
/// - `original_ptr`: For reads with bounce buffer, the original destination pointer.
/// - `payload_len`: For reads with bounce buffer, the actual payload length to copy back.
/// - `batch_id`: The batch ID this submission belongs to (for per-batch tracking)
/// - `nvme_cmd_data`: Optional NVMe command for io_uring_cmd submission
#[derive(Clone)]
struct IoSubmission {
    fd: RawFd,
    offset: u64,
    len: usize,
    ptr_addr: usize,
    is_write: bool,
    completion: Arc<IoCompletion>,
    fixed_buffer_idx: Option<u16>,
    // The fixed buffer is a dma-buf registration: the SQE carries this byte
    // offset into the registered dma-buf instead of a user address.
    fixed_dmabuf: Option<usize>,
    bounce: Option<std::sync::Arc<AlignedBuf>>,
    original_ptr: Option<usize>,        // For bounce buffer reads
    payload_len: Option<usize>,         // For bounce buffer reads
    batch_id: u64,                      // Batch ID for per-batch tracking
    nvme_cmd_data: Option<NvmeCmdData>, // NVMe command data for io_uring_cmd
    // Who this operation is for, as the caller named it when it handed the
    // batch over. Immutable and carried with the submission rather than
    // looked up later: the completion is reaped on the worker thread, long
    // after whatever "current request" a thread-local could have named.
    request_tag: Option<Arc<str>>,
    /// Zero for the initial SQE; incremented for each short-I/O remainder.
    attempt: u64,
}

/// Which buffer path a submission actually took.
///
/// Not which route the Python helper chose, and not what the engine was
/// configured to prefer: this is the flavour of SQE that was built, decided
/// by whether the buffer was found in the registration map and what kind of
/// registration it was. A configured dma-buf pool whose buffers miss the map
/// quietly issues ordinary SQEs, and nothing above this layer can tell.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NativeIoPath {
    /// A passthrough NVMe command.
    UringCmd,
    /// A passthrough NVMe command against a registered buffer.
    UringCmdFixed,
    /// Read/write against a registered dma-buf: the device reaches the
    /// exporting memory directly.
    DmabufFixed,
    /// Read/write against a classic registered host buffer.
    HostFixed,
    /// Read/write through an aligned copy this engine owns.
    Bounce,
    /// Read/write against an ordinary process address.
    Regular,
}

impl NativeIoPath {
    fn of(sub: &IoSubmission) -> Self {
        if sub.nvme_cmd_data.is_some() {
            if sub.fixed_buffer_idx.is_some() {
                return NativeIoPath::UringCmdFixed;
            }
            return NativeIoPath::UringCmd;
        }
        if sub.fixed_dmabuf.is_some() {
            return NativeIoPath::DmabufFixed;
        }
        if sub.fixed_buffer_idx.is_some() {
            return NativeIoPath::HostFixed;
        }
        if sub.bounce.is_some() {
            return NativeIoPath::Bounce;
        }
        NativeIoPath::Regular
    }

    fn name(self) -> &'static str {
        match self {
            NativeIoPath::UringCmd => "uring_cmd",
            NativeIoPath::UringCmdFixed => "uring_cmd_fixed",
            NativeIoPath::DmabufFixed => "dmabuf_fixed",
            NativeIoPath::HostFixed => "host_fixed",
            NativeIoPath::Bounce => "bounce",
            NativeIoPath::Regular => "regular",
        }
    }
}

/// What one physical operation did, and who it was for.
///
/// Recorded after the SQE enters the ring and again when the device answers, so a
/// reader can join the two and see an operation nobody answered for rather
/// than inferring it from a total that happens to match. The request tag is
/// the caller's own immutable identity for the work, carried on the
/// submission: a "current request" global would be read on the worker
/// thread, which is nobody's request.
#[derive(Clone)]
struct NativeIoEvent {
    device_instance_id: u64,
    request_tag: Option<Arc<str>>,
    batch_id: u64,
    operation_id: u64,
    attempt: u64,
    is_write: bool,
    path: &'static str,
    /// "submitted", "completed", "short" or "failed".
    outcome: &'static str,
    bytes: i64,
    offset: u64,
}

/// How many operation records the native journal keeps before dropping the
/// oldest. One entry is a few words; this bounds the diagnostic at roughly a
/// megabyte while covering far more operations than any single request.
const NATIVE_IO_JOURNAL_CAPACITY: usize = 16384;
static NEXT_RAW_BLOCK_DEVICE_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// A bounded record of what the device was asked to do, and what it did.
///
/// Bounded because it is a diagnostic, not a log of record: an engine that
/// kept one entry per operation forever would run out of memory on a long
/// serving run. Dropping is counted, so a reader can see that the record is
/// incomplete rather than trusting a sum that silently lost rows.
struct NativeIoJournal {
    device_instance_id: u64,
    events: Mutex<VecDeque<NativeIoEvent>>,
    capacity: usize,
    dropped: AtomicU64,
}

impl NativeIoJournal {
    fn new(capacity: usize) -> Self {
        NativeIoJournal {
            device_instance_id: NEXT_RAW_BLOCK_DEVICE_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
            events: Mutex::new(VecDeque::with_capacity(capacity.min(1024))),
            capacity,
            dropped: AtomicU64::new(0),
        }
    }

    fn record(&self, event: NativeIoEvent) {
        let mut events = self.events.lock().unwrap();
        if events.len() >= self.capacity {
            events.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        events.push_back(event);
    }

    fn record_submission(&self, sub: &IoSubmission, operation_id: u64) {
        self.record(NativeIoEvent {
            device_instance_id: self.device_instance_id,
            request_tag: sub.request_tag.clone(),
            batch_id: sub.batch_id,
            operation_id,
            attempt: sub.attempt,
            is_write: sub.is_write,
            path: NativeIoPath::of(sub).name(),
            outcome: "submitted",
            bytes: sub.len as i64,
            offset: sub.offset,
        });
    }

    fn record_completion(&self, sub: &IoSubmission, operation_id: u64, result: i32) {
        // NVMe passthrough CQEs carry command status, not a byte count.
        let (outcome, bytes) = if sub.nvme_cmd_data.is_some() {
            if result == 0 {
                ("completed", sub.len as i64)
            } else {
                ("failed", 0)
            }
        } else if result < 0 {
            ("failed", result as i64)
        } else if (result as usize) < sub.len {
            ("short", result as i64)
        } else {
            ("completed", result as i64)
        };
        self.record(NativeIoEvent {
            device_instance_id: self.device_instance_id,
            request_tag: sub.request_tag.clone(),
            batch_id: sub.batch_id,
            operation_id,
            attempt: sub.attempt,
            is_write: sub.is_write,
            path: NativeIoPath::of(sub).name(),
            outcome,
            bytes,
            offset: sub.offset,
        });
    }

    fn drain(&self) -> (Vec<NativeIoEvent>, u64) {
        let mut events = self.events.lock().unwrap();
        // Keep lost rows in the same snapshot as the rows they displaced.
        let dropped = self.dropped.swap(0, Ordering::Relaxed);
        (events.drain(..).collect(), dropped)
    }
}

fn enqueue_if_running(
    queue: &Mutex<Vec<IoSubmission>>,
    shutdown: &AtomicBool,
    in_flight: &AtomicU64,
    submission: IoSubmission,
) -> PyResult<()> {
    let mut queue = queue.lock().unwrap();
    if shutdown.load(Ordering::Relaxed) {
        return Err(PyRuntimeError::new_err("io_uring worker stopped"));
    }
    in_flight.fetch_add(1, Ordering::Relaxed);
    queue.push(submission);
    Ok(())
}

fn stop_submissions(queue: &Mutex<Vec<IoSubmission>>, shutdown: &AtomicBool) {
    let _queue = queue.lock().unwrap();
    shutdown.store(true, Ordering::Relaxed);
}

fn fail_submissions(
    queue: &Mutex<Vec<IoSubmission>>,
    shutdown: &AtomicBool,
    worker_error: &Mutex<Option<String>>,
    error: io::Error,
) {
    let _queue = queue.lock().unwrap();
    *worker_error.lock().unwrap() = Some(format!("io_uring worker submission failed: {error}"));
    shutdown.store(true, Ordering::Relaxed);
}

const SUBMISSION_RETRY_INITIAL_DELAY: Duration = Duration::from_millis(1);
const SUBMISSION_RETRY_MAX_DELAY: Duration = Duration::from_millis(100);
const SUBMISSION_STALL_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct SubmissionRetry {
    stalled_since: Option<Instant>,
    retry_at: Option<Instant>,
    delay: Duration,
}

impl SubmissionRetry {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn remaining_delay(&self, now: Instant) -> Option<Duration> {
        self.retry_at
            .and_then(|deadline| deadline.checked_duration_since(now))
            .filter(|delay| !delay.is_zero())
    }

    fn record_result(&mut self, result: io::Result<usize>, now: Instant) -> io::Result<()> {
        let error = match result {
            Ok(submitted) if submitted > 0 => {
                self.reset();
                return Ok(());
            }
            Ok(_) => io::Error::new(
                io::ErrorKind::WouldBlock,
                "io_uring submit accepted no requests",
            ),
            Err(error)
                if matches!(
                    error.raw_os_error(),
                    Some(libc::EAGAIN) | Some(libc::EINTR) | Some(libc::EBUSY)
                ) =>
            {
                error
            }
            Err(error) => return Err(error),
        };
        let stalled_since = *self.stalled_since.get_or_insert(now);
        if now.duration_since(stalled_since) >= SUBMISSION_STALL_TIMEOUT {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "io_uring submission made no progress for {}s: {error}",
                    SUBMISSION_STALL_TIMEOUT.as_secs()
                ),
            ));
        }
        self.delay = if self.delay.is_zero() {
            SUBMISSION_RETRY_INITIAL_DELAY
        } else {
            (self.delay * 2).min(SUBMISSION_RETRY_MAX_DELAY)
        };
        self.retry_at = Some(now + self.delay);
        Ok(())
    }
}

fn submit_pending(
    ring: &IoUringWrapper,
    pending: &mut VecDeque<u64>,
    #[cfg(feature = "fault-injection")] fault_plan: &Option<Arc<FaultPlan>>,
) -> io::Result<usize> {
    if pending.is_empty() {
        return Ok(0);
    }
    record_submission_result(
        pending,
        submit_ring(
            ring,
            #[cfg(feature = "fault-injection")]
            fault_plan,
        ),
    )
}

fn record_submission_result(
    pending: &mut VecDeque<u64>,
    result: io::Result<usize>,
) -> io::Result<usize> {
    let submitted = result?;
    pending.drain(..submitted.min(pending.len()));
    Ok(submitted)
}

#[cfg(test)]
mod submission_lifecycle_tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn partial_submission_preserves_the_unaccepted_suffix() {
        let mut pending = VecDeque::from([10, 11, 12]);
        record_submission_result(&mut pending, Ok(1)).unwrap();
        assert_eq!(pending, VecDeque::from([11, 12]));
        record_submission_result(&mut pending, Ok(0)).unwrap();
        assert_eq!(pending, VecDeque::from([11, 12]));
        record_submission_result(&mut pending, Ok(2)).unwrap();
        assert!(pending.is_empty());
    }

    #[test]
    fn submission_errors_preserve_pending_entries() {
        for error_code in [libc::EAGAIN, libc::EINTR, libc::EBUSY, libc::EIO] {
            let mut pending = VecDeque::from([10, 11, 12]);
            let error = record_submission_result(
                &mut pending,
                Err(io::Error::from_raw_os_error(error_code)),
            )
            .unwrap_err();
            assert_eq!(error.raw_os_error(), Some(error_code));
            assert_eq!(pending, VecDeque::from([10, 11, 12]));
        }
    }

    #[test]
    fn fatal_error_after_partial_submission_keeps_only_unaccepted_entries_pending() {
        let mut pending = VecDeque::from([10, 11, 12]);
        record_submission_result(&mut pending, Ok(1)).unwrap();
        assert!(record_submission_result(
            &mut pending,
            Err(io::Error::from_raw_os_error(libc::EIO)),
        )
        .is_err());
        assert_eq!(pending, VecDeque::from([11, 12]));
    }

    #[test]
    fn stopped_queue_rejects_without_lifecycle_updates() {
        let queue = Mutex::new(Vec::new());
        let shutdown = AtomicBool::new(false);
        let in_flight = AtomicU64::new(0);
        stop_submissions(&queue, &shutdown);
        let result = enqueue_if_running(&queue, &shutdown, &in_flight, IoSubmission::default());
        assert!(result.is_err());
        assert!(queue.lock().unwrap().is_empty());
        assert_eq!(in_flight.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn stop_and_enqueue_have_a_single_admission_boundary() {
        for _ in 0..64 {
            let queue = Mutex::new(Vec::new());
            let shutdown = AtomicBool::new(false);
            let in_flight = AtomicU64::new(0);
            let barrier = Barrier::new(2);
            thread::scope(|scope| {
                let producer = scope.spawn(|| {
                    barrier.wait();
                    enqueue_if_running(&queue, &shutdown, &in_flight, IoSubmission::default())
                        .is_ok()
                });
                barrier.wait();
                stop_submissions(&queue, &shutdown);
                let accepted = producer.join().unwrap();
                assert_eq!(queue.lock().unwrap().len(), usize::from(accepted));
                assert_eq!(in_flight.load(Ordering::Relaxed), u64::from(accepted));
                assert!(
                    enqueue_if_running(&queue, &shutdown, &in_flight, IoSubmission::default(),)
                        .is_err()
                );
            });
        }
    }
}

type FixedBufferRange = (u16, usize, Option<usize>);
type FixedBufferMap = HashMap<usize, FixedBufferRange>;

fn fixed_buffer_for_range(
    registrations: &FixedBufferMap,
    ptr_addr: usize,
    len: usize,
) -> (Option<u16>, Option<usize>) {
    if let Some((index, size, dmabuf_offset)) = registrations.get(&ptr_addr) {
        if len <= *size {
            return (Some(*index), *dmabuf_offset);
        }
        return (None, None);
    }

    let Some(range_end) = ptr_addr.checked_add(len) else {
        return (None, None);
    };

    for (base, (index, size, dmabuf_offset)) in registrations {
        let Some(base_offset) = *dmabuf_offset else {
            continue;
        };
        let Some(buffer_end) = base.checked_add(*size) else {
            continue;
        };
        if ptr_addr >= *base && range_end <= buffer_end {
            return (Some(*index), Some(base_offset + (ptr_addr - *base)));
        }
    }

    (None, None)
}

#[cfg(test)]
mod fixed_buffer_range_tests {
    use super::*;

    #[test]
    fn dmabuf_slice_keeps_slot_and_adjusts_offset() {
        let mut registrations = HashMap::new();
        registrations.insert(0x1000, (3, 0x4000, Some(0x8000)));

        assert_eq!(
            fixed_buffer_for_range(&registrations, 0x3000, 0x1000),
            (Some(3), Some(0xa000))
        );
    }

    #[test]
    fn ordinary_fixed_buffer_does_not_match_an_interior_slice() {
        let mut registrations = HashMap::new();
        registrations.insert(0x1000, (3, 0x4000, None));

        assert_eq!(
            fixed_buffer_for_range(&registrations, 0x2000, 0x1000),
            (None, None)
        );
    }

    #[test]
    fn range_cannot_extend_beyond_registered_buffer() {
        let mut registrations = HashMap::new();
        registrations.insert(0x1000, (3, 0x4000, Some(0x8000)));

        assert_eq!(
            fixed_buffer_for_range(&registrations, 0x4000, 0x2000),
            (None, None)
        );
    }
}

impl Default for IoSubmission {
    fn default() -> Self {
        IoSubmission {
            fd: -1 as RawFd,
            offset: 0,
            len: 0,
            ptr_addr: 0,
            is_write: false,
            completion: Arc::new(IoCompletion::new()),
            fixed_buffer_idx: None,
            fixed_dmabuf: None,
            bounce: None,
            original_ptr: None,
            payload_len: None,
            batch_id: 0,
            nvme_cmd_data: None,
            request_tag: None,
            attempt: 0,
        }
    }
}

/// io_uring extended buffer update, as introduced by the dma-buf backed
/// registered buffers series: `struct io_uring_rsrc_update2` gained a flags
/// word (IORING_RSRC_UPDATE_EXTENDED) and `data` then points at an array of
/// `struct io_uring_regbuf_desc` instead of iovecs.  The io-uring crate does
/// not expose this yet, so the structures are spelled out here.
#[repr(C)]
struct IoUringRsrcUpdate2 {
    offset: u32,
    flags: u32,
    data: u64,
    tags: u64,
    nr: u32,
    resv2: u32,
}

#[repr(C)]
struct IoUringRegbufDesc {
    type_: u32,
    flags: u32,
    size: u64,
    uaddr: u64,
    dmabuf_fd: i32,
    target_fd: i32,
    resv: [u64; 6],
}

const IORING_REGISTER_BUFFERS_UPDATE: libc::c_uint = 16;
const IORING_RSRC_UPDATE_EXTENDED: u32 = 1 << 1;
const IO_REGBUF_TYPE_DMABUF: u32 = 2;

/// Install `dmabuf_fd`, mapped for I/O on `target_fd`, into sparse
/// registered-buffer slot `index` of ring `ring_fd`.
fn io_uring_register_dmabuf(
    ring_fd: RawFd,
    index: u32,
    dmabuf_fd: i32,
    target_fd: RawFd,
) -> io::Result<()> {
    let desc = IoUringRegbufDesc {
        type_: IO_REGBUF_TYPE_DMABUF,
        flags: 0,
        size: 0,
        uaddr: 0,
        dmabuf_fd,
        target_fd,
        resv: [0; 6],
    };
    let up = IoUringRsrcUpdate2 {
        offset: index,
        flags: IORING_RSRC_UPDATE_EXTENDED,
        data: &desc as *const IoUringRegbufDesc as u64,
        tags: 0,
        nr: 1,
        resv2: 0,
    };
    // For the *_UPDATE register ops the last argument is the size of the
    // update structure, not a count; the count is up.nr.
    let ret = unsafe {
        libc::syscall(
            libc::SYS_io_uring_register,
            ring_fd as libc::c_long,
            IORING_REGISTER_BUFFERS_UPDATE as libc::c_long,
            &up as *const IoUringRsrcUpdate2 as libc::c_long,
            std::mem::size_of::<IoUringRsrcUpdate2>() as libc::c_long,
        )
    };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn dmabuf_extent_bytes(fd: RawFd) -> io::Result<usize> {
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::fstat(fd, &mut stat) };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }
    usize::try_from(stat.st_size).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("dma-buf fd {fd} reports an invalid size {}", stat.st_size),
        )
    })
}

/// Raw block device I/O interface for Python.
///
/// - Synchronous I/O (pread/pwrite) - always available
/// - Asynchronous I/O via io_uring - optional, enabled with use_iouring flag
/// Higher-level policies (slotting, manifests, etc.) live in Python.
#[pyclass]
struct RawBlockDevice {
    fd: RawFd,           // raw file descriptor
    size: u64,           // cached device size in bytes
    closed: AtomicBool,  // avoid double-close
    use_odirect: bool,   // enforce alignment + bypass page cache
    alignment: usize,    // required alignment in bytes
    use_iouring: bool,   // Enable io_uring
    use_uring_cmd: bool, // Enable io_uring_cmd for NVMe passthrough
    // NVMe device data (only when use_uring_cmd=true)
    nvme_nsid: Option<u32>,      // Namespace ID
    nvme_lba_shift: Option<u32>, // LBA shift (log2 of LBA size)
    nvme_lba_size: Option<u32>,  // LBA size in bytes
    // io_uring ring instance (only when use_iouring=true)
    // Uses wrapper to support both standard and big entries for kernel compatibility
    ring: Option<IoUringWrapper>,
    // Queue for sending I/O requests from Python to worker thread
    queue: Option<Arc<Mutex<Vec<IoSubmission>>>>,
    // Background worker thread handle
    worker: Option<thread::JoinHandle<()>>,
    // Shutdown signal for worker thread
    shutdown: Option<Arc<AtomicBool>>,
    worker_error: Arc<Mutex<Option<String>>>,
    // Map from buffer pointer address to registered fixed buffer index
    // Used for zero-copy I/O with pre-registered buffers
    // The tuple is (index, size, dmabuf_offset): dmabuf_offset is Some for a
    // buffer registered through register_fixed_dmabufs(), and is the byte
    // offset of this buffer inside the registered dma-buf, which the SQE
    // carries in place of a user address.
    fixed_buffer_map: Arc<Mutex<FixedBufferMap>>,
    // Flag indicating if fixed buffers have been registered
    fixed_buffers_registered: Arc<AtomicBool>,
    // Count of currently in-flight I/O operations (global)
    // Used for shutdown and cleanup
    in_flight_count: Arc<AtomicU64>,
    // Condition variable for signaling when in_flight_count reaches 0
    in_flight_cvar: Arc<Condvar>,
    // Per-batch in-flight count tracking
    // Maps batch_id -> (in_flight_count, condition_variable)
    batch_in_flight: Arc<Mutex<HashMap<u64, BatchTracking>>>,
    // Worker wake-up: producer eventfd + ring CQ eventfd, both polled via
    // a single epoll_fd. Replaces the previous `Arc<Condvar>` which couldn't
    // be signaled from the kernel-side completion queue.
    batch_ready: Option<Arc<UringNotify>>,
    // Store Python buffer objects for writes, reads to keep them alive until they complete
    // This prevents premature garbage collection while io_uring is using the buffers
    // Keyed by batch_id to isolate concurrent batches
    batched_buffer_objs: Arc<Mutex<HashMap<u64, Vec<Py<PyAny>>>>>,
    // Store IoCompletion objects for batched operations to check for I/O errors
    // Keyed by batch_id to isolate concurrent batches
    batched_completions: Arc<Mutex<HashMap<u64, Vec<Arc<IoCompletion>>>>>,
    // Counter for generating unique batch IDs
    next_batch_id: Arc<AtomicU64>,
    // Batches for which no statement can be made about the device's access.
    // The worker records these before it wakes their waiter, so the waiter
    // sees the unknown outcome rather than a plain failure and can keep the
    // batch's owners instead of releasing them.
    quarantined_batches: Arc<Mutex<HashSet<u64>>>,
    // Set once any outcome has been unknown. Sticky for this incarnation:
    // an engine that cannot say what the kernel is doing cannot serve.
    // Deliberately separate from `closed`, because marking the device closed
    // would make `close()` return without draining or unregistering.
    poisoned: Arc<AtomicBool>,
    // Owners transferred out of `batched_buffer_objs` for a quarantined
    // batch. Never cleared: the Python objects behind them must stay alive
    // while the device may still reach the memory they describe.
    quarantined_owners: Arc<Mutex<Vec<Py<PyAny>>>>,
    // How many owners the terminal retention has taken. It takes them by
    // forgetting the vector, so their count has to be recorded before it
    // becomes unreadable.
    retained_owners: Arc<AtomicUsize>,
    // The scripted ring's state, held apart from `ring` itself. The
    // terminal retention takes the ring and forgets it, and a test asking
    // afterwards whether the engine broke an ownership rule on its way out
    // must still get an answer.
    // What the device was actually asked to do, and what it did, recorded
    // per physical operation. Bounded: a diagnostic, not a log of record.
    io_journal: Arc<NativeIoJournal>,
    #[cfg(feature = "fault-injection")]
    fake_state: Option<Arc<Mutex<FakeRingState>>>,
    #[cfg(feature = "fault-injection")]
    fault_plan: Option<Arc<FaultPlan>>,
}

/// RAII guard for a raw file descriptor
struct FdGuard {
    fd: RawFd,
}

impl FdGuard {
    /// Takes ownership of an open file descriptor.
    ///
    /// Args:
    /// - `fd`: a descriptor returned by a successful `open()`.
    fn new(fd: RawFd) -> Self {
        FdGuard { fd }
    }

    /// Releases ownership of the fd to the caller, disarming the guard so the
    /// descriptor is not closed on drop.
    ///
    /// Returns the raw descriptor, now owned by the caller.
    fn disarm(self) -> RawFd {
        let fd = self.fd;
        std::mem::forget(self);
        fd
    }
}

impl Drop for FdGuard {
    fn drop(&mut self) {
        // SAFETY: `fd` was returned by a successful `open()` and ownership has
        // not been released via `disarm()`, so this is the only close of this
        // descriptor.
        unsafe {
            libc::close(self.fd);
        }
    }
}

impl RawBlockDevice {
    fn ensure_io_available(&self, needs_worker: bool) -> PyResult<()> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("device is closed"));
        }
        if self.outcome_is_unknown() {
            return Err(PyRuntimeError::new_err(
                "device has an unknown I/O outcome and cannot accept new work",
            ));
        }
        if needs_worker
            && self
                .shutdown
                .as_ref()
                .is_some_and(|shutdown| shutdown.load(Ordering::Relaxed))
        {
            return Err(PyRuntimeError::new_err("io_uring worker stopped"));
        }
        if needs_worker && self.worker.is_none() {
            return Err(PyRuntimeError::new_err("io_uring worker has stopped"));
        }
        Ok(())
    }

    /// Internal constructor performs all low level setup.
    #[allow(clippy::too_many_arguments)]
    /// The scripted ring behind this device, or a refusal.
    #[cfg(feature = "fault-injection")]
    fn scripted_ring(&self) -> PyResult<&FakeRing> {
        match &self.ring {
            Some(IoUringWrapper::Fake(ring)) => Ok(ring),
            _ => Err(PyRuntimeError::new_err(
                "this device has no scripted ring to drive (it was not \
                 built with one, or a refused close has retained it)",
            )),
        }
    }

    /// The scripted ring's state, which outlives the ring.
    #[cfg(feature = "fault-injection")]
    fn scripted_state(&self) -> PyResult<&Arc<Mutex<FakeRingState>>> {
        self.fake_state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("this device was not built with a scripted ring")
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_internal(
        path: String,
        writable: bool,
        use_odirect: bool,
        alignment: usize,
        use_iouring: bool,
        use_uring_cmd: bool,
        io_engine: Option<String>,
        iouring_queue_depth: usize,
        fake_ring_capacity: usize,
    ) -> PyResult<Self> {
        if fake_ring_capacity > 0 && !cfg!(feature = "fault-injection") {
            // Said rather than ignored: a caller asking for a scripted ring
            // on a serving build is measuring nothing, and silence would
            // let it believe otherwise.
            return Err(PyValueError::new_err(
                "fake_ring_capacity needs the nondefault fault-injection \
                 feature; this build does not carry it",
            ));
        }
        let use_iouring = parse_use_iouring(io_engine, use_iouring)?;
        let iouring_queue_depth = iouring_queue_depth.max(1);
        // use_uring_cmd requires use_iouring to be enabled
        if use_uring_cmd && !use_iouring {
            return Err(PyValueError::new_err(
                "use_uring_cmd requires use_iouring to be enabled",
            ));
        }
        let cpath =
            CString::new(path.clone()).map_err(|_| PyValueError::new_err("path contains NUL"))?;
        let mut flags = if writable {
            libc::O_RDWR
        } else {
            libc::O_RDONLY
        };
        if use_odirect && !use_uring_cmd {
            flags |= O_DIRECT;
        }
        // SAFETY: open returns fd or -1.
        let fd = unsafe { libc::open(cpath.as_ptr(), flags) };
        if fd < 0 {
            return Err(os_err("open failed"));
        }
        // Take ownership of the fd so it is closed if any fallible setup step
        // below returns early before the fd is moved into the RawBlockDevice.
        // Disarmed once the struct is successfully constructed.
        let fd_guard = FdGuard::new(fd);

        // Initialize NVMe data if io_uring command support is enabled
        let (nvme_nsid, nvme_lba_shift, nvme_lba_size, nvme_id_ns) = if use_uring_cmd {
            // Validate that device is a character device (required for io_uring_cmd)
            let is_char_dev = is_character_device(&path)?;
            if !is_char_dev {
                return Err(PyValueError::new_err(
                    "use_uring_cmd requires an NVMe namespace character device (e.g., /dev/ng0n1)",
                ));
            }

            // Get namespace ID from device path
            let nsid = nvme_get_nsid_from_fd(fd)?;
            // Send identify namespace command to get LBA size
            let id_ns = nvme_identify_ns(fd, nsid)?;
            let lba_shift = nvme_get_lba_shift(&id_ns)?;
            let lba_size = nvme_get_lba_size(&id_ns)?;
            if alignment == 0 || !alignment.is_multiple_of(lba_size as usize) {
                return Err(PyValueError::new_err(format!(
                    "alignment ({alignment}) must be a non-zero multiple of NVMe LBA size \
                     ({lba_size})"
                )));
            }

            (Some(nsid), Some(lba_shift), Some(lba_size), Some(id_ns))
        } else {
            (None, None, None, None)
        };

        // Calculate device size. Use NVMe ns info for character device
        // Use ioctl/fstat for block devices and regular files
        let size = if use_uring_cmd {
            if let (Some(id_ns), Some(lba_size)) = (nvme_id_ns, nvme_lba_size) {
                nvme_ns_size_bytes(&id_ns, lba_size)
            } else {
                0
            }
        } else {
            fd_size_bytes(fd)?
        };

        let worker_error = Arc::new(Mutex::new(None));
        // Created before the branch so that the worker closure and the
        // struct can share one plan without threading it through the tuple.
        #[cfg(feature = "fault-injection")]
        let fault_plan: Option<Arc<FaultPlan>> = Some(Arc::new(FaultPlan::default()));

        // Shared with the worker, because the two halves of an operation are
        // recorded in different places: the submission where the SQE is
        // built, the outcome where the completion is reaped.
        let io_journal = Arc::new(NativeIoJournal::new(NATIVE_IO_JOURNAL_CAPACITY));

        let (
            ring_opt,
            queue_opt,
            shutdown_opt,
            worker_opt,
            batch_ready_opt,
            in_flight_count_opt,
            in_flight_cvar_opt,
            batched_buffer_objs_opt,
            batched_completions_opt,
            next_batch_id_opt,
            batch_in_flight_opt,
            quarantined_batches_opt,
            poisoned_opt,
            quarantined_owners_opt,
        ) = if use_iouring {
            let notify =
                Arc::new(UringNotify::new().map_err(|e| {
                    PyRuntimeError::new_err(format!("UringNotify init failed: {}", e))
                })?);
            // A scripted ring, when one was asked for. It takes the same
            // notifier the real rings register their CQ eventfd with, so a
            // completion wakes the worker through the path production uses.
            #[cfg(feature = "fault-injection")]
            let scripted = if fake_ring_capacity > 0 {
                Some(IoUringWrapper::Fake(FakeRing::new(
                    fake_ring_capacity,
                    Arc::clone(&notify),
                )))
            } else {
                None
            };
            #[cfg(not(feature = "fault-injection"))]
            let scripted: Option<IoUringWrapper> = None;
            // Try to create IoUring with big entries (Entry128/Entry32) first
            // This is required for io_uring_cmd support (kernel 5.19+)
            // If that fails, fall back to standard entries (Entry/Entry) for kernel 5.4-5.18
            let ring = match scripted {
                Some(scripted) => scripted,
                None => match IoUring::<Entry128, Entry32>::builder()
                    .build(iouring_queue_depth as u32)
                {
                    Ok(big_ring) => {
                        // Big entries supported - io_uring_cmd can be used
                        if use_uring_cmd {
                            // Validate that device is a character device (required for io_uring_cmd)
                            let is_char_dev = is_character_device(&path)?;
                            if !is_char_dev {
                                return Err(PyValueError::new_err(
                                "use_uring_cmd requires an NVMe namespace character device (e.g., /dev/ng0n1)",
                            ));
                            }
                        }
                        // Register the CQ eventfd with the ring so the kernel writes to it
                        // whenever a CQE is posted. Must happen before the ring is wrapped
                        // in a Mutex / handed to the worker.
                        big_ring
                            .submitter()
                            .register_eventfd(notify.cq_efd)
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!("register_eventfd failed: {}", e))
                            })?;
                        let big_ring = Arc::new(Mutex::new(big_ring));
                        IoUringWrapper::Big(big_ring)
                    }
                    Err(_) => {
                        // Big entries not supported (kernel < 5.19), fall back to standard entries
                        // io_uring_cmd is not available on these kernels
                        if use_uring_cmd {
                            return Err(PyRuntimeError::new_err(
                            "io_uring_cmd requires kernel 5.19 or later (big SQE/CQE entries not supported)",
                        ));
                        }
                        let std_ring = IoUring::<SqueueEntry, Entry>::builder()
                            .build(iouring_queue_depth as u32)
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!("io_uring init failed: {}", e))
                            })?;
                        // Register the CQ eventfd with the ring so the kernel writes to it
                        // whenever a CQE is posted. Must happen before the ring is wrapped
                        // in a Mutex / handed to the worker.
                        std_ring
                            .submitter()
                            .register_eventfd(notify.cq_efd)
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!("register_eventfd failed: {}", e))
                            })?;
                        let std_ring = Arc::new(Mutex::new(std_ring));
                        IoUringWrapper::Standard(std_ring)
                    }
                },
            };
            let queue = Arc::new(Mutex::new(Vec::<IoSubmission>::new()));
            let shutdown = Arc::new(AtomicBool::new(false));
            let batch_ready = notify;
            let in_flight_count = Arc::new(AtomicU64::new(0));
            let in_flight_cvar = Arc::new(Condvar::new());
            let batched_buffer_objs = Arc::new(Mutex::new(HashMap::<u64, Vec<Py<PyAny>>>::new()));
            let quarantined_owners = Arc::new(Mutex::new(Vec::<Py<PyAny>>::new()));
            let batched_buffer_objs_worker = Arc::clone(&batched_buffer_objs);
            let quarantined_owners_worker = Arc::clone(&quarantined_owners);
            let quarantined_batches = Arc::new(Mutex::new(HashSet::<u64>::new()));
            let poisoned = Arc::new(AtomicBool::new(false));
            let quarantined_batches_worker = Arc::clone(&quarantined_batches);
            let poisoned_worker = Arc::clone(&poisoned);
            let io_journal_worker = Arc::clone(&io_journal);
            #[cfg(feature = "fault-injection")]
            let fault_plan_worker = fault_plan.clone();
            let batched_completions =
                Arc::new(Mutex::new(HashMap::<u64, Vec<Arc<IoCompletion>>>::new()));
            let next_batch_id = Arc::new(AtomicU64::new(1));
            let batch_in_flight = Arc::new(Mutex::new(HashMap::<u64, BatchTracking>::new()));

            let ring_clone = ring.clone();
            let queue_clone = Arc::clone(&queue);
            let shutdown_clone = Arc::clone(&shutdown);
            let worker_error_clone = Arc::clone(&worker_error);
            let batch_ready_clone = Arc::clone(&batch_ready);
            let in_flight_count_clone = Arc::clone(&in_flight_count);
            let in_flight_cvar_clone = Arc::clone(&in_flight_cvar);
            let batch_in_flight_clone = Arc::clone(&batch_in_flight);
            let ring_size = iouring_queue_depth;

            // Helper function to copy data from bounce buffer to original buffer
            fn copy_from_bounce_buffer(bounce: &AlignedBuf, orig_ptr: usize, payload_len: usize) {
                unsafe {
                    libc::memcpy(
                        orig_ptr as *mut libc::c_void,
                        bounce.as_ptr() as *const libc::c_void,
                        payload_len,
                    );
                }
            }

            // Helper function to handle completion result and set IoCompletion
            // Returns Ok(()) for successful completion, Err for errors
            // Note: Short I/O resubmission for regular I/O is handled BEFORE calling this function
            // in the main completion loop. This function only handles:
            // - Full completions
            // - Errors (negative results)
            // - Short I/O during shutdown (cannot resubmit)
            fn handle_completion_result(
                sub: &mut IoSubmission,
                cqe_result: i32,
                is_shutdown: bool,
            ) -> PyResult<()> {
                let is_uring_cmd = sub.nvme_cmd_data.is_some();

                if cqe_result < 0 {
                    let code = -cqe_result;
                    let _ = sub.bounce.take();
                    Err(PyOSError::new_err((code, "io_uring I/O error")))
                } else if is_uring_cmd {
                    // Non-zero result indicates NVMe command error
                    if cqe_result != 0 {
                        let code = cqe_result;
                        let _ = sub.bounce.take();
                        Err(PyOSError::new_err((code, "io_uring_cmd NVMe error")))
                    } else {
                        // io_uring_cmd successful completion (result == 0)
                        // For reads with bounce buffer, copy data back to original buffer
                        if !sub.is_write {
                            if let (Some(bounce), Some(orig_ptr), Some(payload_len)) =
                                (sub.bounce.take(), sub.original_ptr, sub.payload_len)
                            {
                                copy_from_bounce_buffer(&bounce, orig_ptr, payload_len);
                            }
                        } else {
                            let _ = sub.bounce.take();
                        }
                        Ok(())
                    }
                } else {
                    // Regular io_uring read/write
                    let bytes_transferred = cqe_result as usize;
                    if bytes_transferred < sub.len {
                        if is_shutdown {
                            // Short read/write during shutdown: fail the request
                            let _ = sub.bounce.take();
                            Err(PyRuntimeError::new_err(
                                "io_uring worker shutting down: short I/O during shutdown",
                            ))
                        } else {
                            // This should never happen
                            let _ = sub.bounce.take();
                            Err(PyRuntimeError::new_err(
                                "Unexpected short I/O: internal error",
                            ))
                        }
                    } else {
                        // Full completion
                        // For reads with bounce buffer, copy data back to original buffer
                        if !sub.is_write {
                            if let (Some(bounce), Some(orig_ptr), Some(payload_len)) =
                                (sub.bounce.take(), sub.original_ptr, sub.payload_len)
                            {
                                copy_from_bounce_buffer(&bounce, orig_ptr, payload_len);
                            }
                        } else {
                            let _ = sub.bounce.take();
                        }
                        Ok(())
                    }
                }
            }

            // Helper function to decrement in-flight counts and notify condition variables
            fn decrement_in_flight(
                in_flight_count: &Arc<AtomicU64>,
                in_flight_cvar: &Arc<Condvar>,
                batch_in_flight: &Arc<Mutex<HashMap<u64, BatchTracking>>>,
                batch_id: u64,
            ) {
                let prev = in_flight_count.fetch_sub(1, Ordering::Relaxed);
                if prev == 1 {
                    in_flight_cvar.notify_all();
                }
                // Decrement per-batch in-flight count and notify if batch is complete
                if batch_id != 0 {
                    let batch_map = batch_in_flight.lock().unwrap();
                    if let Some((batch_count, batch_cvar)) = batch_map.get(&batch_id) {
                        let prev_batch = batch_count.fetch_sub(1, Ordering::Relaxed);
                        if prev_batch == 1 {
                            batch_cvar.notify_all();
                        }
                    }
                }
            }

            // Helper function to build and submit an SQE for a submission
            fn build_and_submit_sqe(
                ring: &IoUringWrapper,
                journal: &NativeIoJournal,
                sub: &IoSubmission,
                user_data: u64,
            ) -> Result<(), PyErr> {
                let ptr = sub.ptr_addr as *mut u8;
                // Check if this is an io_uring_cmd submission
                if let Some(nvme_data) = &sub.nvme_cmd_data {
                    // Prepare NVMe uring command
                    let mut nvme_cmd: NvmeUringCmd = unsafe { std::mem::zeroed() };
                    nvme_uring_cmd_prep(
                        &mut nvme_cmd,
                        sub.is_write,
                        nvme_data.nsid,
                        sub.offset,
                        sub.len,
                        nvme_data.lba_shift,
                        ptr,
                        nvme_data.dtype,
                        nvme_data.dspec,
                    )?;

                    // Convert NvmeUringCmd to byte array for UringCmd80
                    let cmd_bytes: [u8; 80] = unsafe { std::mem::transmute_copy(&nvme_cmd) };

                    // Build UringCmd80 with big SQE entry
                    let mut uring_cmd =
                        opcode::UringCmd80::new(Fd(sub.fd), NVME_URING_CMD_IO).cmd(cmd_bytes);

                    // Set buf_index if using fixed buffers.  A dma-buf
                    // registration cannot be imported by a passthrough command
                    // (io_uring_cmd_import_fixed() has no dma-buf support), so
                    // register_fixed_dmabufs() refuses use_uring_cmd engines.
                    if let Some(idx) = sub.fixed_buffer_idx {
                        uring_cmd = uring_cmd.buf_index(Some(idx));
                    }

                    let sqe128 = uring_cmd.build().user_data(user_data);
                    ring.push_cmd(&sqe128, user_data, SqeDescriptor::describe(sub, user_data))?;
                } else {
                    let sqe = build_regular_sqe(sub, user_data);
                    ring.push_regular(&sqe, user_data, SqeDescriptor::describe(sub, user_data))?;
                }
                journal.record_submission(sub, user_data);
                Ok(())
            }

            // Worker thread that handles io_uring submissions and completions.
            //
            // Runs a continuous loop that:
            // - Processes completion queue events (CQ) from the kernel
            // - Waits for new I/O requests from Python via the queue
            // - Submits new requests to the kernel (SQ)
            //
            // On shutdown, we must process ALL pending requests to avoid deadlocks:
            // - Requests still in the queue
            // - Requests already submitted to kernel (in_flight)
            // Each waiting Python thread must be woken up with an error.
            //
            // Thread safety: All access to shared state (ring, queue, in_flight)
            // is protected by mutexes. The worker is the only thread that:
            // - Reads from the submission queue
            // - Submits to io_uring
            // - Processes completions
            let worker = thread::Builder::new()
                .name("rust-rawblock-uring".into())
                .spawn(move || {
                    let mut in_flight: HashMap<u64, IoSubmission> = HashMap::new();
                    let mut pending = VecDeque::with_capacity(ring_size);
                    let mut next_user_data: u64 = 1;
                    let mut submission_retry = SubmissionRetry::default();
                    // Operations whose access by the device was never shown to
                    // have ended. Quarantining the whole submission keeps every
                    // owner it holds alive, not just its bounce allocation, and
                    // they outlive the worker rather than return anything to an
                    // allocator while the kernel may still act on it.
                    let mut quarantined: Vec<IoSubmission> = Vec::new();
                    // Record a batch as unknown-outcome, and poison the device,
                    // BEFORE its waiter is woken. The waiter reads this to
                    // decide whether it may release the batch's owners, so the
                    // order matters: waking first would let it release them
                    // while the device may still reach that memory.
                    // Record a batch as unknown-outcome, poison the device,
                    // and take the batch's Python owners out of reach --
                    // here, when the worker learns, rather than whenever
                    // somebody happens to ask. A caller that never polls is
                    // the case that made the difference: owners sat in
                    // `batched_buffer_objs` until `wait_iouring` moved them,
                    // so a close after an unknown outcome dropped them and
                    // returned their pool slices to an allocator while the
                    // device may still have been writing.
                    let mark_unknown = |batch_id: u64| {
                        poisoned_worker.store(true, Ordering::SeqCst);
                        quarantined_batches_worker.lock().unwrap().insert(batch_id);
                        drain_batch_owners(
                            &batched_buffer_objs_worker,
                            &quarantined_owners_worker,
                            Some(batch_id),
                        );
                    };
                    // The whole unknown-outcome sequence, in the only order
                    // that is safe, so no site has to remember it: put the
                    // submission's owners out of reach, record the batch as
                    // unknown and poison the device, and only then tell
                    // anyone. A waiter reads the poison flag to decide
                    // whether it may release what it holds, and a
                    // synchronous caller waits on the completion directly
                    // rather than on the batch counter, so setting the
                    // completion first reads to it as "failed, and nothing
                    // is otherwise wrong".
                    let fail_unknown =
                        |sub: &IoSubmission, quarantined: &mut Vec<IoSubmission>, err: PyErr| {
                            quarantined.push(sub.clone());
                            mark_unknown(sub.batch_id);
                            sub.completion.set(Err(err));
                        };

                    while !shutdown_clone.load(Ordering::Relaxed) {
                        // This drains all completed I/O operations from the completion queue (CQ).
                        // For each completion:
                        //   - Remove the request from our in_flight tracking HashMap
                        //   - Signal the waiting Python thread via IoCompletion
                        //   - Decrement the in_flight_count atomic
                        //   - Wake up any threads waiting for all I/O to complete
                        {
                            for cqe in ring_clone.take_completions() {
                                let user_data = cqe.user_data;
                                if let Some(mut sub) = in_flight.remove(&user_data) {
                                    submission_retry.reset();
                                    let batch_id = sub.batch_id;
                                    let cqe_result = cqe.result;
                                    io_journal_worker
                                        .record_completion(&sub, user_data, cqe_result);

                                    // Decide a short completion before pushing
                                    // anything. io_uring_cmd is never retried, and
                                    // a registered dma-buf transfer ends here.
                                    let short_action = short_io_action(
                                        cqe_result,
                                        sub.len,
                                        sub.nvme_cmd_data.is_some(),
                                        sub.fixed_dmabuf.is_some(),
                                    );
                                    if short_action == ShortIoAction::FailTerminally {
                                        let result =
                                            handle_completion_result(&mut sub, -libc::EIO, false);
                                        sub.completion.set(result);
                                        decrement_in_flight(
                                            &in_flight_count_clone,
                                            &in_flight_cvar_clone,
                                            &batch_in_flight_clone,
                                            batch_id,
                                        );
                                        continue;
                                    }
                                    if short_action == ShortIoAction::Retry {
                                        let bytes_transferred = cqe_result as usize;
                                        sub.offset += bytes_transferred as u64;
                                        sub.len -= bytes_transferred;
                                        sub.attempt += 1;
                                        // Update buffer pointer for writes and direct reads
                                        if sub.is_write || sub.bounce.is_none() {
                                            sub.ptr_addr += bytes_transferred;
                                        }
                                        if !sub.is_write {
                                            if let (
                                                Some(bounce),
                                                Some(orig_ptr),
                                                Some(payload_len),
                                            ) = (
                                                sub.bounce.as_ref(),
                                                sub.original_ptr,
                                                sub.payload_len,
                                            ) {
                                                copy_from_bounce_buffer(
                                                    bounce,
                                                    orig_ptr,
                                                    bytes_transferred.min(payload_len),
                                                );
                                                sub.original_ptr =
                                                    Some(orig_ptr + bytes_transferred);
                                                sub.payload_len = Some(
                                                    payload_len.saturating_sub(bytes_transferred),
                                                );
                                            }
                                        }
                                        in_flight.insert(user_data, sub.clone());
                                        if ring_clone.submission_len() < ring_size {
                                            match build_and_submit_sqe(
                                                &ring_clone,
                                                &io_journal_worker,
                                                &sub,
                                                user_data,
                                            ) {
                                                Ok(()) => pending.push_back(user_data),
                                                Err(error) => {
                                                    in_flight.remove(&user_data);
                                                    sub.completion.set(Err(error));
                                                    decrement_in_flight(
                                                        &in_flight_count_clone,
                                                        &in_flight_cvar_clone,
                                                        &batch_in_flight_clone,
                                                        batch_id,
                                                    );
                                                }
                                            }
                                        } else {
                                            in_flight.remove(&user_data);
                                            let mut queue = queue_clone.lock().unwrap();
                                            queue.push(sub);
                                        }
                                        continue;
                                    }

                                    let result =
                                        handle_completion_result(&mut sub, cqe_result, false);
                                    sub.completion.set(result);
                                    decrement_in_flight(
                                        &in_flight_count_clone,
                                        &in_flight_cvar_clone,
                                        &batch_in_flight_clone,
                                        batch_id,
                                    );
                                }
                            }
                            ring_clone.submission_sync();
                        }

                        if let Some(delay) = submission_retry.remaining_delay(Instant::now()) {
                            let _ = ring_clone.reap_without_submitting();
                            batch_ready_clone.wait(Some(delay));
                            continue;
                        }

                        // Block on epoll only if there's truly nothing pending. The empty +
                        // shutdown checks short-circuit so we don't sleep when a producer or
                        // do_close() already left work for us. Race-free against a late
                        // signal_producer(): eventfd is a counter, so a wake-up between the
                        // check and wait() is buffered, not lost.
                        if pending.is_empty() && !in_flight.is_empty() {
                            let _ = ring_clone.reap_without_submitting();
                        }
                        if !shutdown_clone.load(Ordering::Relaxed)
                            && pending.is_empty()
                            && queue_clone.lock().unwrap().is_empty()
                        {
                            batch_ready_clone.wait(None);
                        }

                        let mut q = queue_clone.lock().unwrap();
                        if !q.is_empty() {
                            // Take all pending requests from our queue and submit them to io_uring.
                            //
                            // - Remove all pending requests from queue
                            // - Check how much space is available in the ring (max 256 entries)
                            // - If batch is larger than available space, put excess back in queue
                            // - Increment in_flight_count for each request we're about to submit
                            // - Build SQE (Submission Queue Entry) for each request
                            // - Push SQEs to the ring
                            // - Call submit() to send them to the kernel
                            //
                            // Fixed Buffer Support:
                            // - If the buffer was pre-registered with register_fixed_buffers(),
                            //   we use ReadFixed/WriteFixed for true zero-copy I/O
                            // - Otherwise we use regular Read/Write with user-space pointers
                            let batch: Vec<IoSubmission> = std::mem::take(&mut *q);
                            let batch_len = batch.len();

                            // Bound both resident and accepted operations: CQ
                            // capacity must cover every request still owned.
                            let available = ring_size.saturating_sub(in_flight.len());
                            let to_submit_count = std::cmp::min(available, batch_len);

                            if to_submit_count < batch_len {
                                let remaining: Vec<_> = batch[to_submit_count..].to_vec();
                                if !remaining.is_empty() {
                                    q.extend(remaining);
                                }
                            }

                            drop(q);

                            for sub in batch.iter().take(to_submit_count) {
                                let user_data = next_user_data;
                                next_user_data = next_user_data.wrapping_add(1);
                                match build_and_submit_sqe(
                                    &ring_clone,
                                    &io_journal_worker,
                                    sub,
                                    user_data,
                                ) {
                                    Ok(()) => {
                                        pending.push_back(user_data);
                                        in_flight.insert(user_data, sub.clone());
                                    }
                                    Err(error) => {
                                        sub.completion.set(Err(error));
                                        decrement_in_flight(
                                            &in_flight_count_clone,
                                            &in_flight_cvar_clone,
                                            &batch_in_flight_clone,
                                            sub.batch_id,
                                        );
                                    }
                                }
                            }
                        } else {
                            drop(q);
                        }
                        if !pending.is_empty() {
                            let result = submit_pending(
                                &ring_clone,
                                &mut pending,
                                #[cfg(feature = "fault-injection")]
                                &fault_plan_worker,
                            );
                            if let Err(error) =
                                submission_retry.record_result(result, Instant::now())
                            {
                                fail_submissions(
                                    &queue_clone,
                                    &shutdown_clone,
                                    &worker_error_clone,
                                    error,
                                );
                                for (_, sub) in in_flight.drain() {
                                    fail_unknown(
                                        &sub,
                                        &mut quarantined,
                                        PyRuntimeError::new_err(
                                            "io_uring submission failed; outcome is unknown",
                                        ),
                                    );
                                    decrement_in_flight(
                                        &in_flight_count_clone,
                                        &in_flight_cvar_clone,
                                        &batch_in_flight_clone,
                                        sub.batch_id,
                                    );
                                }
                                pending.clear();
                            }
                        }
                    }

                    // SHUTDOWN: Wake up all waiting Python threads
                    // Drain the queue and wake up all waiting threads with error
                    {
                        let mut q = queue_clone
                            .lock()
                            .expect("Worker: queue mutex poisoned during shutdown");
                        while let Some(mut sub) = q.pop() {
                            let batch_id = sub.batch_id;
                            let _ = sub.bounce.take();
                            sub.completion.set(Err(PyRuntimeError::new_err(
                                "io_uring worker shutting down",
                            )));
                            decrement_in_flight(
                                &in_flight_count_clone,
                                &in_flight_cvar_clone,
                                &batch_in_flight_clone,
                                batch_id,
                            );
                        }
                    }

                    // Drain what the kernel still owns rather than guess at
                    // a duration. A submitted request points the device at this
                    // process's memory, so releasing that memory on a timer can
                    // let a late completion land in a buffer already reused.
                    #[cfg(feature = "fault-injection")]
                    if let IoUringWrapper::Fake(ring) = &ring_clone {
                        ring.begin_shutdown();
                    }
                    // Unaccepted SQEs stay in the ring. Their allocations must
                    // stay owned until the ring itself can be safely torn down.
                    // Do not submit new device work after the admission boundary.
                    for user_data in pending.drain(..) {
                        if let Some(sub) = in_flight.remove(&user_data) {
                            fail_unknown(
                                &sub,
                                &mut quarantined,
                                PyRuntimeError::new_err(
                                    "io_uring stopped with resident work; outcome is unknown",
                                ),
                            );
                            decrement_in_flight(
                                &in_flight_count_clone,
                                &in_flight_cvar_clone,
                                &batch_in_flight_clone,
                                sub.batch_id,
                            );
                        }
                    }
                    if !in_flight.is_empty() {
                        let _ = ring_clone.cancel_submitted();
                    }
                    let drain_deadline = Instant::now() + Duration::from_secs(5);
                    loop {
                        let _ = ring_clone.reap_without_submitting();
                        let completions = ring_clone.take_completions();
                        let reaped = completions.len();
                        for RingCompletion {
                            user_data,
                            result: cqe_result,
                        } in completions
                        {
                            if let Some(mut sub) = in_flight.remove(&user_data) {
                                let batch_id = sub.batch_id;
                                io_journal_worker.record_completion(&sub, user_data, cqe_result);
                                let result = handle_completion_result(&mut sub, cqe_result, true);
                                sub.completion.set(result);
                                decrement_in_flight(
                                    &in_flight_count_clone,
                                    &in_flight_cvar_clone,
                                    &batch_in_flight_clone,
                                    batch_id,
                                );
                            }
                        }
                        ring_clone.submission_sync();
                        if in_flight.is_empty() || Instant::now() >= drain_deadline {
                            break;
                        }
                        if reaped == 0 {
                            thread::sleep(Duration::from_millis(1));
                        }
                    }

                    // Whatever is still outstanding was never reported complete.
                    // The drain above had a deadline, and a deadline is not a
                    // fence: reaching it says the worker stopped waiting, not
                    // that the device stopped. Tell each waiter its operation
                    // did not finish, which is a statement about the logical
                    // result, and quarantine the submission, which is a
                    // statement about who may touch its memory and its extent.
                    // Neither is a claim that the operation was cancelled.
                    for (_user_data, sub) in in_flight.drain() {
                        let batch_id = sub.batch_id;
                        fail_unknown(
                            &sub,
                            &mut quarantined,
                            PyRuntimeError::new_err(
                                "io_uring worker shut down before this request \
                                 completed; its outcome is unknown",
                            ),
                        );
                        decrement_in_flight(
                            &in_flight_count_clone,
                            &in_flight_cvar_clone,
                            &batch_in_flight_clone,
                            batch_id,
                        );
                    }

                    if !quarantined.is_empty() {
                        // Deliberately never dropped. Each submission keeps its
                        // bounce allocation and its completion. The Python object
                        // behind a zero-copy target is held per batch, and the
                        // batch was marked unknown before its waiter was woken,
                        // so the waiter transfers those owners to quarantine too
                        // rather than releasing them. The device is poisoned for
                        // this incarnation, so nothing further is admitted; the
                        // extents these name are withheld by the core, which the
                        // worker cannot reach from here.
                        eprintln!(
                            "raw_block: quarantining {} operation(s) whose completion \
                             was never observed; their buffers and extents are not \
                             safe to reuse in this process",
                            quarantined.len()
                        );
                        std::mem::forget(quarantined);
                    }

                    // Final notification in case any thread is waiting on in_flight_count
                    in_flight_cvar_clone.notify_all();
                })
                .expect("spawn rust-rawblock-uring worker");

            (
                Some(ring),
                Some(queue),
                Some(shutdown),
                Some(worker),
                Some(batch_ready),
                Some(in_flight_count),
                Some(in_flight_cvar),
                Some(batched_buffer_objs),
                Some(batched_completions),
                Some(next_batch_id),
                Some(batch_in_flight),
                Some(quarantined_batches),
                Some(poisoned),
                Some(quarantined_owners),
            )
        } else {
            (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            )
        };

        // All fallible setup succeeded: hand the fd over to the struct, whose
        // Drop is now responsible for closing it. Disarm so the guard does not
        // close the same descriptor a second time.
        let fd = fd_guard.disarm();

        Ok(Self {
            fd,
            size,
            closed: AtomicBool::new(false),
            use_odirect,
            alignment,
            use_iouring,
            use_uring_cmd,
            nvme_nsid,
            nvme_lba_shift,
            nvme_lba_size,
            #[cfg(feature = "fault-injection")]
            fake_state: match &ring_opt {
                Some(IoUringWrapper::Fake(ring)) => Some(Arc::clone(&ring.state)),
                _ => None,
            },
            ring: ring_opt,
            queue: queue_opt,
            worker: worker_opt,
            shutdown: shutdown_opt,
            worker_error,
            fixed_buffer_map: Arc::new(Mutex::new(HashMap::new())),
            fixed_buffers_registered: Arc::new(AtomicBool::new(false)),
            in_flight_count: in_flight_count_opt.unwrap_or_else(|| Arc::new(AtomicU64::new(0))),
            in_flight_cvar: in_flight_cvar_opt.unwrap_or_else(|| Arc::new(Condvar::new())),
            batch_ready: batch_ready_opt,
            batched_buffer_objs: batched_buffer_objs_opt
                .unwrap_or_else(|| Arc::new(Mutex::new(HashMap::new()))),
            batched_completions: batched_completions_opt
                .unwrap_or_else(|| Arc::new(Mutex::new(HashMap::new()))),
            next_batch_id: next_batch_id_opt.unwrap_or_else(|| Arc::new(AtomicU64::new(1))),
            batch_in_flight: batch_in_flight_opt
                .unwrap_or_else(|| Arc::new(Mutex::new(HashMap::new()))),
            quarantined_batches: quarantined_batches_opt
                .unwrap_or_else(|| Arc::new(Mutex::new(HashSet::new()))),
            poisoned: poisoned_opt.unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
            retained_owners: Arc::new(AtomicUsize::new(0)),
            io_journal,
            quarantined_owners: quarantined_owners_opt
                .unwrap_or_else(|| Arc::new(Mutex::new(Vec::new()))),
            #[cfg(feature = "fault-injection")]
            fault_plan,
        })
    }

    /// Build NVMe command data for io_uring_cmd operations.
    /// Returns None if use_uring_cmd is disabled.
    fn _build_nvme_cmd_data(&self, dtype: u8, dspec: u16) -> PyResult<Option<NvmeCmdData>> {
        if !self.use_uring_cmd {
            return Ok(None);
        }
        Ok(Some(NvmeCmdData {
            nsid: self
                .nvme_nsid
                .ok_or_else(|| PyRuntimeError::new_err("NVMe namespace ID not available"))?,
            lba_shift: self
                .nvme_lba_shift
                .ok_or_else(|| PyRuntimeError::new_err("NVMe LBA shift not available"))?,
            dtype,
            dspec,
        }))
    }
}

#[pymethods]
impl RawBlockDevice {
    #[new]
    #[pyo3(
        signature = (
            path,
            writable,
            use_odirect = false,
            use_iouring = false,
            use_uring_cmd = false,
            alignment = 4096,
            io_engine = None,
            iouring_queue_depth = RING_SIZE,
            fake_ring_capacity = 0
        )
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        path: String,
        writable: bool,
        use_odirect: bool,
        use_iouring: bool,
        use_uring_cmd: bool,
        alignment: usize,
        io_engine: Option<String>,
        iouring_queue_depth: usize,
        fake_ring_capacity: usize,
    ) -> PyResult<Self> {
        Self::new_internal(
            path,
            writable,
            use_odirect,
            alignment,
            use_iouring,
            use_uring_cmd,
            io_engine,
            iouring_queue_depth,
            fake_ring_capacity,
        )
    }

    // Expose cached size to Python.
    fn size_bytes(&self) -> PyResult<u64> {
        Ok(self.size)
    }

    /// Validate the actual opened target before strict DMA-BUF setup.
    ///
    /// Accept block devices and initialized, private XFS/ext4 files. This
    /// preflight does not register a buffer or prevent external file mutation.
    fn validate_dmabuf_target(&self, capacity_bytes: u64) -> PyResult<&'static str> {
        self.ensure_io_available(false)?;
        if !self.use_odirect || self.use_uring_cmd {
            return Err(PyValueError::new_err(
                "strict DMA-BUF targets require ordinary O_DIRECT I/O",
            ));
        }
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: st points to a writable stat with the platform ABI.
        if unsafe { libc::fstat(self.fd, &mut st) } != 0 {
            return Err(os_err("fstat failed"));
        }
        if capacity_bytes == 0 || capacity_bytes > fd_size_bytes(self.fd)? {
            return Err(PyValueError::new_err(
                "strict DMA-BUF capacity must be positive and fit the opened target",
            ));
        }
        match st.st_mode & libc::S_IFMT {
            libc::S_IFBLK => Ok("block"),
            libc::S_IFREG => {
                validate_initialized_file(self.fd, capacity_bytes)?;
                Ok("file")
            }
            _ => Err(PyValueError::new_err(
                "strict DMA-BUF target must be a block device or regular file",
            )),
        }
    }

    /// Return the terminal io_uring submission error, or None if none occurred.
    ///
    /// Recoverable errors are retried with 1-100 ms backoff, waking on completions.
    /// After 30 seconds without submission or completion progress, the worker
    /// stops accepting requests. The error is available before accepted I/O
    /// finishes draining. Normal shutdown and POSIX mode do not set an error.
    fn worker_error(&self) -> Option<String> {
        self.worker_error.lock().unwrap().clone()
    }

    /// Get NVMe namespace ID (only available when use_uring_cmd=true)
    fn nvme_nsid(&self) -> PyResult<u32> {
        self.nvme_nsid.ok_or_else(|| {
            PyRuntimeError::new_err("NVMe namespace ID not available (use_uring_cmd not enabled)")
        })
    }

    /// Get NVMe LBA shift (log2 of LBA size, only available when use_uring_cmd=true)
    fn nvme_lba_shift(&self) -> PyResult<u32> {
        self.nvme_lba_shift.ok_or_else(|| {
            PyRuntimeError::new_err("NVMe LBA shift not available (use_uring_cmd not enabled)")
        })
    }

    /// Get NVMe LBA size in bytes (only available when use_uring_cmd=true)
    fn nvme_lba_size(&self) -> PyResult<u32> {
        self.nvme_lba_size.ok_or_else(|| {
            PyRuntimeError::new_err("NVMe LBA size not available (use_uring_cmd not enabled)")
        })
    }

    /// Fetch FDP status descriptors for the NVMe namespace.
    fn fetch_fdp_status(&self) -> PyResult<Vec<(u16, u16)>> {
        if !self.use_uring_cmd {
            return Err(PyRuntimeError::new_err(
                "fetch_fdp_status requires use_uring_cmd to be enabled",
            ));
        }
        let nsid = self
            .nvme_nsid
            .ok_or_else(|| PyRuntimeError::new_err("NVMe namespace ID not available"))?;
        fetch_fdp_status(self.fd, nsid)
    }

    /// Register fixed buffers for zero-copy io_uring operations.
    ///
    /// - Pre-registering memory buffers with the kernel
    /// - Using indexed descriptors (instead of pointers) in I/O operations
    /// - The kernel then does DMA directly from/to the registered buffers
    ///
    /// This is more efficient than regular I/O because:
    /// - No buffer copying between user space and kernel
    /// - The kernel can pin the memory pages for the duration of I/O
    ///
    /// Registration must happen BEFORE any I/O using these buffers.
    /// The buffers must remain valid (not freed) until unregistered.
    /// Refuses registration on a closed or poisoned device, or while a
    /// fixed-buffer table is already installed.
    #[pyo3(signature = (buffer_ptrs, buffer_sizes))]
    fn register_fixed_buffers(
        &self,
        buffer_ptrs: Vec<usize>,
        buffer_sizes: Vec<usize>,
    ) -> PyResult<()> {
        self.ensure_registration_available()?;
        if !self.use_iouring {
            return Err(PyRuntimeError::new_err("io_uring not enabled"));
        }
        if buffer_ptrs.len() != buffer_sizes.len() {
            return Err(PyValueError::new_err(
                "buffer_ptrs and buffer_sizes must have same length",
            ));
        }
        if buffer_ptrs.is_empty() {
            return Err(PyValueError::new_err(
                "at least one buffer must be provided",
            ));
        }

        if let Some(ring) = &self.ring {
            let mut iovecs: Vec<libc::iovec> = Vec::new();
            for (ptr, size) in buffer_ptrs.iter().zip(buffer_sizes.iter()) {
                iovecs.push(libc::iovec {
                    iov_base: *ptr as *mut libc::c_void,
                    iov_len: *size,
                });
            }
            {
                let result = ring.register_host_buffers(&iovecs);
                match result {
                    Ok(_) => {
                        self.fixed_buffers_registered.store(true, Ordering::Relaxed);
                        let mut map = self.fixed_buffer_map.lock().unwrap();
                        for (idx, (ptr, size)) in
                            buffer_ptrs.iter().zip(buffer_sizes.iter()).enumerate()
                        {
                            map.insert(*ptr, (idx as u16, *size, None));
                        }
                    }
                    Err(e) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "register_buffers failed: {}",
                            e
                        )))
                    }
                }
            }
        }
        Ok(())
    }

    /// Register staging buffers as dma-buf backed fixed buffers.
    ///
    /// Each buffer is described by its user address (the key the I/O paths
    /// look up, exactly as for `register_fixed_buffers`), its size, the
    /// dma-buf file descriptor that exports the memory it lives in (a
    /// udmabuf over the buffer's memfd, or the dma-buf system heap
    /// allocation it was mmap()ed from), and the user address at which that
    /// dma-buf is mapped.  Many buffers may share one dma-buf: the allocator's
    /// paged buffers are slices of one big mapping.  Each distinct dma-buf
    /// takes one registered-buffer slot; each buffer records its byte offset
    /// inside its dma-buf, which the SQE then carries in place of a user
    /// address.
    ///
    /// The dma-buf is registered against this device's fd for later fixed
    /// I/O. The kernel controls when the device mapping is established and
    /// may split a transfer into multiple block requests or commands.
    ///
    /// Requires a kernel with dma-buf backed registered buffers (the
    /// io_uring extended buffer update).  On an older kernel the update
    /// fails with EINVAL; the caller should fall back to
    /// `register_fixed_buffers` only after successful cleanup. If cleanup
    /// fails, the device is poisoned, close retains the registration, and
    /// the caller must retain the backing allocator. Refused on a
    /// `use_uring_cmd` engine: NVMe passthrough cannot import a dma-buf
    /// registration.
    #[pyo3(signature = (buffer_ptrs, buffer_sizes, dmabuf_fds, dmabuf_bases))]
    fn register_fixed_dmabufs(
        &self,
        buffer_ptrs: Vec<usize>,
        buffer_sizes: Vec<usize>,
        dmabuf_fds: Vec<i32>,
        dmabuf_bases: Vec<usize>,
    ) -> PyResult<()> {
        self.ensure_registration_available()?;
        if !self.use_iouring {
            return Err(PyRuntimeError::new_err("io_uring not enabled"));
        }
        if self.use_uring_cmd {
            return Err(PyRuntimeError::new_err(
                "dma-buf fixed buffers are not supported with use_uring_cmd",
            ));
        }
        let n = buffer_ptrs.len();
        if n == 0 {
            return Err(PyValueError::new_err(
                "at least one buffer must be provided",
            ));
        }
        if buffer_sizes.len() != n || dmabuf_fds.len() != n || dmabuf_bases.len() != n {
            return Err(PyValueError::new_err(
                "buffer_ptrs, buffer_sizes, dmabuf_fds and dmabuf_bases must have same length",
            ));
        }
        let ring = match &self.ring {
            Some(ring) => ring,
            None => return Err(PyRuntimeError::new_err("io_uring ring not available")),
        };
        let mut extent_by_fd: HashMap<i32, (usize, usize)> = HashMap::new();
        let mut registration_by_ptr: HashMap<usize, (usize, i32, usize)> = HashMap::new();
        for i in 0..n {
            let fd = dmabuf_fds[i];
            if fd < 0 {
                return Err(PyValueError::new_err(format!(
                    "buffer {i} has an invalid dma-buf fd {fd}"
                )));
            }
            if buffer_sizes[i] == 0 {
                return Err(PyValueError::new_err(format!("buffer {i} has zero size")));
            }
            let base = dmabuf_bases[i];
            let extent = match extent_by_fd.get(&fd) {
                Some((known_base, known_extent)) => {
                    if *known_base != base {
                        return Err(PyValueError::new_err(format!(
                            "dma-buf fd {fd} has inconsistent bases: {known_base:#x} and {base:#x}"
                        )));
                    }
                    *known_extent
                }
                None => {
                    let extent = ring.dmabuf_extent(fd).map_err(|e| {
                        PyValueError::new_err(format!("failed to query dma-buf fd {fd} size: {e}"))
                    })?;
                    if extent == 0 {
                        return Err(PyValueError::new_err(format!(
                            "dma-buf fd {fd} reports a zero-byte extent"
                        )));
                    }
                    extent_by_fd.insert(fd, (base, extent));
                    extent
                }
            };
            if buffer_ptrs[i] < base {
                return Err(PyValueError::new_err(format!(
                    "buffer {} lies below its dma-buf base",
                    i
                )));
            }
            let offset = buffer_ptrs[i] - base;
            let end = offset.checked_add(buffer_sizes[i]).ok_or_else(|| {
                PyValueError::new_err(format!("buffer {i} range overflows usize"))
            })?;
            if end > extent {
                return Err(PyValueError::new_err(format!(
                    "buffer {i} range [{offset}, {end}) exceeds dma-buf fd {fd} extent {extent}"
                )));
            }
            let registration = (buffer_sizes[i], fd, offset);
            if let Some(previous) = registration_by_ptr.insert(buffer_ptrs[i], registration) {
                if previous != registration {
                    return Err(PyValueError::new_err(format!(
                        "buffer pointer {:#x} has conflicting dma-buf registrations",
                        buffer_ptrs[i]
                    )));
                }
            }
        }

        // One slot per distinct dma-buf, in first-seen order.
        let mut slot_of_fd: HashMap<i32, u16> = HashMap::new();
        let mut fds_in_order: Vec<i32> = Vec::new();
        for fd in dmabuf_fds.iter() {
            if !slot_of_fd.contains_key(fd) {
                if fds_in_order.len() >= u16::MAX as usize {
                    return Err(PyValueError::new_err("too many dma-bufs"));
                }
                slot_of_fd.insert(*fd, fds_in_order.len() as u16);
                fds_in_order.push(*fd);
            }
        }

        // A sparse table of the right size, then one extended update per slot.
        let sparse = ring.register_sparse_buffers(fds_in_order.len() as u32);
        if let Err(e) = sparse {
            return Err(PyRuntimeError::new_err(format!(
                "register_buffers_sparse failed: {}",
                e
            )));
        }
        // Even an empty sparse table is a live registration. A failed slot
        // update must not let close forget the table or earlier slots.
        self.fixed_buffers_registered.store(true, Ordering::Relaxed);

        for (idx, fd) in fds_in_order.iter().enumerate() {
            if let Err(e) = ring.register_dmabuf_slot(idx as u32, *fd, self.fd) {
                if let Err(cleanup) = ring.unregister_buffers() {
                    self.poisoned.store(true, Ordering::SeqCst);
                    return Err(PyRuntimeError::new_err(format!(
                        "dma-buf registration of slot {idx} (fd {fd}) failed: {e}; \
                         buffer unregister also failed: {cleanup}; device is poisoned \
                         and its backing allocator must be retained"
                    )));
                }
                self.fixed_buffers_registered
                    .store(false, Ordering::Relaxed);
                return Err(PyRuntimeError::new_err(format!(
                    "dma-buf registration of slot {} (fd {}) failed: {}",
                    idx, fd, e
                )));
            }
        }

        {
            let mut map = self.fixed_buffer_map.lock().unwrap();
            map.clear();
            for i in 0..n {
                let slot = slot_of_fd[&dmabuf_fds[i]];
                map.insert(
                    buffer_ptrs[i],
                    (
                        slot,
                        buffer_sizes[i],
                        Some(buffer_ptrs[i] - dmabuf_bases[i]),
                    ),
                );
            }
        }
        self.fixed_buffers_registered.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Batched write: submit multiple writes at once via io_uring.
    /// All writes are queued to the worker thread, which processes them
    /// in batches to maximize throughput.
    ///
    /// `payload_lens` defaults to `total_lens`. Host padding is zero-filled
    /// without changing the source. Registered dma-buf buffers describe a
    /// complete physical slot, whose padding is not accessed by the CPU.
    /// Host buffer exports remain pinned until the batch is polled after a
    /// known completion. Unknown outcomes retain those exports permanently.
    /// Callers must not modify source contents before completion.
    ///
    /// Returns a batch ID that must be passed to `wait_iouring()` to wait for
    /// completion and obtain a success bitmap plus sparse completion errors.
    /// Validation or request-preparation errors are raised instead of returning
    /// a batch ID.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (offsets, buffers, total_lens, placement_ids = None, payload_lens = None, request_tag = None))]
    fn batched_write(
        &self,
        py: Python<'_>,
        offsets: Vec<u64>,
        buffers: Vec<Bound<'_, PyAny>>,
        total_lens: Vec<usize>,
        placement_ids: Option<Vec<Option<i32>>>,
        payload_lens: Option<Vec<usize>>,
        request_tag: Option<String>,
    ) -> PyResult<u64> {
        let request_tag: Option<Arc<str>> = request_tag.map(|tag| Arc::from(tag.as_str()));
        if !self.use_iouring {
            return Err(PyRuntimeError::new_err("io_uring not enabled"));
        }
        self.ensure_io_available(true)?;

        let n = offsets.len();
        if n == 0 {
            // Return a valid batch_id even for empty batches
            return Ok(self.next_batch_id.fetch_add(1, Ordering::Relaxed));
        }
        if buffers.len() != n || total_lens.len() != n {
            return Err(PyValueError::new_err("All vectors must have same length"));
        }
        let placement_ids = if let Some(pids) = placement_ids {
            if pids.len() != n {
                return Err(PyValueError::new_err(
                    "placement_ids must have same length as offsets",
                ));
            }
            pids.into_iter()
                .map(|pid| pid.map(placement_id_to_u16).transpose())
                .collect::<PyResult<Vec<_>>>()?
        } else {
            vec![None; n]
        };

        let payload_lens = payload_lens.unwrap_or_else(|| total_lens.clone());
        if payload_lens.len() != n {
            return Err(PyValueError::new_err(
                "payload_lens must have same length as offsets",
            ));
        }
        if payload_lens
            .iter()
            .zip(&total_lens)
            .any(|(payload, total)| payload > total)
        {
            return Err(PyValueError::new_err("total_len must be >= payload_len"));
        }

        // Acquire buffer views to keep them alive until wait_iouring() completes
        let mut views: Vec<BufRef> = Vec::with_capacity(n);
        for buffer in &buffers {
            let view = match get_buffer(py, buffer, false) {
                Ok(view) => view,
                Err(error) => {
                    for view in views {
                        view.release();
                    }
                    return Err(error);
                }
            };
            if view.ptr.is_null() {
                for v in views {
                    v.release();
                }
                view.release();
                return Err(PyValueError::new_err("null buffer pointer"));
            }
            views.push(view);
        }

        // Validate buffer capacities before allocating any batch tracking so the
        // error path stays simple (just release the views).
        let mut cap_err: Option<(usize, usize)> = None;
        for (i, view) in views.iter().enumerate() {
            let cap = view.len;
            if cap < payload_lens[i] {
                cap_err = Some((cap, payload_lens[i]));
                break;
            }
        }
        if let Some((cap, need)) = cap_err {
            for v in views {
                v.release();
            }
            return Err(PyValueError::new_err(format!(
                "input buffer too small: cap={cap} need={need}"
            )));
        }

        let mut owners = Vec::with_capacity(n);
        for (buffer, view) in buffers.iter().zip(views.iter()) {
            match batch_buffer_owner(buffer, view.host_accessible) {
                Ok(owner) => owners.push(owner),
                Err(error) => {
                    for view in views {
                        view.release();
                    }
                    return Err(error);
                }
            }
        }

        // Generate a unique batch ID for this batch
        let batch_id = self.next_batch_id.fetch_add(1, Ordering::Relaxed);

        // Initialize per-batch tracking for this batch
        {
            let mut batch_map = self.batch_in_flight.lock().unwrap();
            batch_map.insert(
                batch_id,
                (Arc::new(AtomicU64::new(0)), Arc::new(Condvar::new())),
            );
        }

        // Store buffer objects to keep them alive until they are complete
        {
            let mut stored_objs = self.batched_buffer_objs.lock().unwrap();
            stored_objs.insert(batch_id, owners);
        }

        // Extract pointers and capacities as usize before releasing GIL (raw
        // pointers are not Send).
        let mut ptrs = Vec::with_capacity(n);
        let mut caps = Vec::with_capacity(n);
        let mut host_accessible = Vec::with_capacity(n);
        for view in &views {
            ptrs.push(view.ptr as usize);
            caps.push(view.len);
            host_accessible.push(view.host_accessible);
        }

        for view in views {
            view.release();
        }

        let fd = self.fd;
        let use_odirect = self.use_odirect;
        let alignment = self.alignment;
        let use_uring_cmd = self.use_uring_cmd;
        let fixed_buffers_registered = self.fixed_buffers_registered.load(Ordering::Relaxed);
        // Clone the fixed buffer map before releasing GIL to avoid lock contention
        let fixed_buffer_map: FixedBufferMap = if fixed_buffers_registered {
            let map = self.fixed_buffer_map.lock().unwrap();
            map.clone()
        } else {
            HashMap::new()
        };

        let nvme_cmd_data_base = if use_uring_cmd {
            Some((
                self.nvme_nsid
                    .ok_or_else(|| PyRuntimeError::new_err("NVMe namespace ID not available"))?,
                self.nvme_lba_shift
                    .ok_or_else(|| PyRuntimeError::new_err("NVMe LBA shift not available"))?,
            ))
        } else {
            None
        };
        let in_flight_count = Arc::clone(&self.in_flight_count);
        let queue = Arc::clone(self.queue.as_ref().unwrap());
        let batch_ready = Arc::clone(self.batch_ready.as_ref().unwrap());
        let batched_completions = Arc::clone(&self.batched_completions);
        let batch_in_flight = Arc::clone(&self.batch_in_flight);
        // Additional clones for cleanup on error path
        let batch_in_flight_cleanup = Arc::clone(&batch_in_flight);
        let batched_completions_cleanup = Arc::clone(&batched_completions);
        let batched_buffer_objs_cleanup = Arc::clone(&self.batched_buffer_objs);

        // Release the GIL while submitting I/O operations
        let res = py.allow_threads(move || {
            let mut submissions: Vec<(IoSubmission, Arc<IoCompletion>)> = Vec::with_capacity(n);

            // Prepare all requests, bounce buffers (if needed) and collect submission data.
            for i in 0..n {
                let total_len = total_lens[i];
                let offset = offsets[i];

                let comp = Arc::new(IoCompletion::new());

                // Fixed buffers are pre-registered with io_uring, enabling true zero-copy I/O
                let (fixed_idx, fixed_dmabuf) =
                    fixed_buffer_for_range(&fixed_buffer_map, ptrs[i], total_len);
                if !host_accessible[i] && fixed_dmabuf.is_none() {
                    return Err(PyValueError::new_err(
                        "pointer-only buffer is not fully covered by a dma-buf registration",
                    ));
                }

                if use_odirect {
                    #[allow(clippy::manual_is_multiple_of)]
                    if (offset as usize) % alignment != 0 {
                        return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
                    }
                    #[allow(clippy::manual_is_multiple_of)]
                    if total_len % alignment != 0 {
                        return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
                    }
                }

                let prepared = prepare_iouring_write_buffer(
                    ptrs[i],
                    caps[i],
                    payload_lens[i],
                    total_len,
                    use_odirect,
                    alignment,
                    fixed_idx,
                    fixed_dmabuf,
                )?;

                // Build NVMe command data
                let placement_id_u16 = placement_ids[i];

                // None means no directive. Some(pid) sets the FDP directive.
                // Placement identifier 0 is reserved for default writes and is
                // rejected by placement_id_to_u16().
                let nvme_cmd_data = if let Some((nsid, lba_shift)) = nvme_cmd_data_base {
                    Some(NvmeCmdData {
                        nsid,
                        lba_shift,
                        dtype: if placement_id_u16.is_some() { 0x2 } else { 0x0 },
                        dspec: placement_id_u16.unwrap_or(0),
                    })
                } else {
                    None
                };

                let sub = IoSubmission {
                    fd,
                    offset,
                    len: total_len,
                    ptr_addr: prepared.ptr_addr,
                    is_write: true,
                    completion: comp.clone(),
                    fixed_buffer_idx: prepared.fixed_buffer_idx,
                    fixed_dmabuf: prepared.fixed_dmabuf,
                    bounce: prepared.bounce,
                    original_ptr: None,
                    payload_len: None,
                    batch_id,
                    nvme_cmd_data,
                    request_tag: request_tag.clone(),
                    attempt: 0,
                };

                submissions.push((sub, comp));
            }

            for (sub, comp) in submissions {
                // Increment per-batch in-flight count
                {
                    let batch_map = batch_in_flight.lock().unwrap();
                    if let Some((batch_count, _)) = batch_map.get(&batch_id) {
                        batch_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
                {
                    if let Err(error) = enqueue_if_running(
                        &queue,
                        self.shutdown.as_ref().expect("shutdown must exist"),
                        &in_flight_count,
                        sub,
                    ) {
                        comp.set(Err(error));
                        let batch_map = batch_in_flight.lock().unwrap();
                        if let Some((batch_count, batch_cvar)) = batch_map.get(&batch_id) {
                            if batch_count.fetch_sub(1, Ordering::Relaxed) == 1 {
                                batch_cvar.notify_all();
                            }
                        }
                    }
                }
                batch_ready.signal_producer();

                // Store completion for error checking in wait_iouring
                {
                    let mut completions = batched_completions.lock().unwrap();
                    let batch_completions = completions.entry(batch_id).or_default();
                    batch_completions.push(comp);
                }
            }
            Ok::<(), PyErr>(())
        });

        // Preparation failed, clean up the tracking entries to prevent leaks
        if let Err(e) = res {
            {
                let mut batch_map = batch_in_flight_cleanup.lock().unwrap();
                batch_map.remove(&batch_id);
            }
            {
                let mut stored_objs = batched_buffer_objs_cleanup.lock().unwrap();
                stored_objs.remove(&batch_id);
            }
            {
                let mut completions = batched_completions_cleanup.lock().unwrap();
                completions.remove(&batch_id);
            }
            return Err(e);
        }

        Ok(batch_id)
    }

    /// Whether any outcome on this device has been unknown.
    ///
    /// Sticky for the life of this device: once the engine cannot say what
    /// the kernel is doing, it cannot say it later either. Callers use this
    /// to refuse further strict work rather than recompute or fall back.
    fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst)
    }

    /// How many batches are held because their outcome was never established.
    fn quarantined_batch_count(&self) -> usize {
        self.quarantined_batches.lock().unwrap().len()
    }

    /// How many Python buffer owners are retained for those batches.
    ///
    /// Live count only: the terminal retention takes the vector, so after
    /// a refused close this reads zero. `retained_owner_count()` is the
    /// one that survives it, and a caller sampling after a close wants
    /// that.
    fn quarantined_owner_count(&self) -> usize {
        self.quarantined_owners.lock().unwrap().len()
    }

    /// How many buffer owners a refused close retained, for the process.
    ///
    /// Recorded when the retention takes them, because it takes them by
    /// forgetting the vector and a length read afterwards is zero. A
    /// counter that reports nothing retained, on the one path whose whole
    /// purpose is retaining, is worse than no counter.
    fn retained_owner_count(&self) -> usize {
        self.retained_owners.load(Ordering::SeqCst)
    }

    /// Wait for all in-flight I/O for a specific batch to complete.
    /// The method waits on a per-batch condition variable that gets signaled
    /// when the batch's in-flight count reaches 0.
    ///
    /// Args:
    ///     batch_id: The batch ID returned by batched_write() or batched_read().
    ///               Only completions from this batch are checked.
    ///
    /// Returns a success bitmap aligned with the operations submitted in this
    /// batch and a sparse list of `(operation_index, error_message)` entries.
    /// `batched_read()` and `batched_write()` can raise validation or
    /// request-preparation errors instead of returning a batch ID. After a
    /// batch ID is returned, I/O completion failures are reported in both
    /// returned collections.
    #[pyo3(signature = (batch_id))]
    fn wait_iouring(&self, py: Python<'_>, batch_id: u64) -> PyResult<IoUringBatchResults> {
        if !self.use_iouring {
            return Ok((Vec::new(), Vec::new()));
        }

        // Get the per-batch tracking for this batch
        let (batch_count, batch_cvar) = {
            let batch_map = self.batch_in_flight.lock().unwrap();
            match batch_map.get(&batch_id) {
                Some((count, cvar)) => (Arc::clone(count), Arc::clone(cvar)),
                None => {
                    // Batch not found. This could be an empty batch or already completed
                    // Check if there are any completions for this batch
                    let mut completions = self.batched_completions.lock().unwrap();
                    let batch_completions = completions.remove(&batch_id);
                    drop(completions);
                    let results = collect_iouring_completion_results(batch_completions);
                    self.retire_batch_owners(batch_id);
                    return Ok(results);
                }
            }
        };

        // Release the GIL while waiting for I/O to complete
        py.allow_threads(move || {
            let mutex = Mutex::new(());
            let mut guard = mutex.lock().unwrap();
            while batch_count.load(Ordering::Relaxed) > 0 {
                let (g, _) = batch_cvar
                    .wait_timeout(guard, Duration::from_micros(10))
                    .unwrap();
                guard = g;
            }
        });

        // Check all completion results for errors for this specific batch
        let mut completions = self.batched_completions.lock().unwrap();
        let batch_completions = completions.remove(&batch_id);
        drop(completions);
        let results = collect_iouring_completion_results(batch_completions);

        // Release this batch's owners, or keep them if its outcome is unknown.
        self.retire_batch_owners(batch_id);

        // Clean up per-batch tracking
        let mut batch_map = self.batch_in_flight.lock().unwrap();
        batch_map.remove(&batch_id);

        Ok(results)
    }

    /// Synchronous read using io_uring.
    ///
    /// Deprecated: use ``batched_read()`` followed by ``wait_iouring()`` instead.
    /// Whether this build carries the test-only fault seam.
    ///
    /// A serving build answers false. A test asks before installing a plan so
    /// that it fails with a clear reason rather than silently measuring
    /// nothing on an ordinary artifact.
    #[staticmethod]
    fn has_fault_injection() -> bool {
        cfg!(feature = "fault-injection")
    }

    /// Make one submit call report the named outcome instead of the kernel's.
    ///
    /// `faults` maps a submit-call ordinal to one of `reports_zero_taken`,
    /// `retryable:<errno>` or `fatal:<errno>`. Keying on the ordinal is what
    /// makes a plan reproducible: a test names the third submit and gets the
    /// third submit, however fast the run is.
    ///
    /// The substitution happens before the ring, so the entries stay exactly
    /// where they were and the kernel is told nothing. No completion is ever
    /// synthesised for a request the kernel owns. That is also why there is
    /// no way to state a non-zero partial take here -- see the refusal
    /// below.
    #[cfg(feature = "fault-injection")]
    fn inject_submit_faults(&self, faults: Vec<(u64, String)>) -> PyResult<()> {
        let plan = self
            .fault_plan
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("device has no fault plan"))?;
        let mut submits = plan.submits.lock().unwrap();
        for (ordinal, spec) in faults {
            let fault = if spec == "reports_zero_taken" {
                SubmitFault::ReportsZeroTaken
            } else if spec.starts_with("partially_taken") {
                return Err(PyValueError::new_err(
                    "partially_taken cannot be expressed by this seam: the \
                     substitution happens before the ring, so no entry was \
                     offered to the kernel and any non-zero count would be \
                     a claim about entries that never moved. Use \
                     reports_zero_taken for the resident case; a genuine \
                     partial take comes from filling the submission queue.",
                ));
            } else if let Some(n) = spec.strip_prefix("retryable:") {
                SubmitFault::Retryable(
                    n.parse()
                        .map_err(|_| PyValueError::new_err(format!("bad errno in {spec}")))?,
                )
            } else if let Some(n) = spec.strip_prefix("fatal:") {
                SubmitFault::Fatal(
                    n.parse()
                        .map_err(|_| PyValueError::new_err(format!("bad errno in {spec}")))?,
                )
            } else {
                return Err(PyValueError::new_err(format!("unknown fault {spec}")));
            };
            submits.insert(ordinal, fault);
        }
        Ok(())
    }

    /// Take what the device was asked to do, and what it did, as rows.
    ///
    /// Each row names the request the operation was for, the batch it
    /// belonged to, the buffer path it actually took, the outcome and the
    /// byte count. A submitted row means the SQE entered the ring, which
    /// does not yet prove kernel acceptance. Pair it with a completion by
    /// operation_id and attempt: a short transfer starts another attempt
    /// with the same operation_id. An operation nobody answered for
    /// shows up as the submission with no outcome beside it, rather than
    /// having to be inferred from totals that happen to agree.
    ///
    /// Draining is deliberate. These are consumed by whoever is summing
    /// them, and a reader that took them twice would count them twice.
    /// ``dropped`` is how many rows the bound discarded before this call,
    /// so a sum known to be incomplete says so.
    fn take_io_journal(&self) -> PyResult<(Vec<HashMap<String, String>>, u64)> {
        let (events, dropped) = self.io_journal.drain();
        let rows = events
            .into_iter()
            .map(|event| {
                let mut row: HashMap<String, String> = HashMap::new();
                row.insert(
                    "device_instance_id".to_string(),
                    event.device_instance_id.to_string(),
                );
                row.insert(
                    "request_tag".to_string(),
                    event
                        .request_tag
                        .as_ref()
                        .map_or_else(String::new, |tag| tag.to_string()),
                );
                row.insert("batch_id".to_string(), event.batch_id.to_string());
                row.insert("operation_id".to_string(), event.operation_id.to_string());
                row.insert("attempt".to_string(), event.attempt.to_string());
                row.insert(
                    "direction".to_string(),
                    if event.is_write { "write" } else { "read" }.to_string(),
                );
                row.insert("path".to_string(), event.path.to_string());
                row.insert("outcome".to_string(), event.outcome.to_string());
                row.insert("bytes".to_string(), event.bytes.to_string());
                row.insert("offset".to_string(), event.offset.to_string());
                row
            })
            .collect();
        Ok((rows, dropped))
    }

    /// Entries pushed and not yet taken by a submit.
    #[cfg(feature = "fault-injection")]
    fn fake_resident(&self) -> PyResult<Vec<u64>> {
        Ok(self
            .scripted_state()?
            .lock()
            .unwrap()
            .resident
            .iter()
            .map(|entry| entry.user_data)
            .collect())
    }

    /// Requests a submit handed over and that have not been answered for.
    ///
    /// Anything here when the device closes is memory the device may still
    /// be reaching, which is the state the whole retention rule exists for.
    #[cfg(feature = "fault-injection")]
    fn fake_owned(&self) -> PyResult<Vec<u64>> {
        Ok(self
            .scripted_state()?
            .lock()
            .unwrap()
            .owned
            .iter()
            .map(|entry| entry.user_data)
            .collect())
    }

    /// The geometry of every entry still resident, in submission order.
    ///
    /// Counting entries cannot tell a correct remainder from one aimed at
    /// the wrong offset, address or length -- and repeated payload bytes
    /// make a readback comparison agree with either. This is what the
    /// submission actually asks the device to do.
    #[cfg(feature = "fault-injection")]
    fn fake_resident_sqes(&self) -> PyResult<Vec<HashMap<String, i64>>> {
        let state = self.scripted_state()?;
        let state = state.lock().unwrap();
        Ok(state.resident.iter().map(describe_fake_sqe).collect())
    }

    /// The geometry of every entry the ring handed to its imaginary kernel.
    #[cfg(feature = "fault-injection")]
    fn fake_owned_sqes(&self) -> PyResult<Vec<HashMap<String, i64>>> {
        let state = self.scripted_state()?;
        let state = state.lock().unwrap();
        Ok(state.owned.iter().map(describe_fake_sqe).collect())
    }

    /// Make every submit take nothing and report zero, until released.
    ///
    /// A real submit reports zero when the entries it would have flushed
    /// were not flushed, and the worker has to leave them resident and come
    /// back. Holding it is how more than one entry becomes resident at the
    /// same time, which is the only way a partial take is more than a full
    /// take of one entry.
    #[cfg(feature = "fault-injection")]
    fn fake_hold_submits(&self, held: bool) -> PyResult<()> {
        self.scripted_state()?.lock().unwrap().hold_submits = held;
        Ok(())
    }

    /// Synthetic registrations this ring was asked to make, in order.
    ///
    /// Explicitly synthetic: no descriptor reached a kernel, so a record
    /// here is evidence about the engine's own registration path and about
    /// nothing else.
    #[cfg(feature = "fault-injection")]
    fn fake_registrations(&self) -> PyResult<Vec<HashMap<String, i64>>> {
        let state = self.scripted_state()?;
        let state = state.lock().unwrap();
        Ok(state
            .registrations
            .iter()
            .map(|record| {
                let mut described: HashMap<String, i64> = HashMap::new();
                described.insert("count".to_string(), record.count as i64);
                described.insert("offset".to_string(), record.offset as i64);
                described.insert(
                    "kind".to_string(),
                    match record.kind.as_str() {
                        "host" => 0,
                        "sparse" => 1,
                        "dmabuf" => 2,
                        _ => 3,
                    },
                );
                described
            })
            .collect())
    }

    /// Fail a matching synthetic registration call without changing its table.
    ///
    /// Calls are matched in the order installed, by kind and slot offset.
    /// `kind` is sparse, dmabuf or unregister; `errno` must be positive.
    #[cfg(feature = "fault-injection")]
    fn fake_registration_fails(&self, kind: String, offset: u32, errno: i32) -> PyResult<()> {
        if !matches!(kind.as_str(), "sparse" | "dmabuf" | "unregister") || errno <= 0 {
            return Err(PyValueError::new_err(
                "registration failure requires sparse, dmabuf or unregister and positive errno",
            ));
        }
        self.scripted_state()?
            .lock()
            .unwrap()
            .registration_failures
            .push_back((kind, offset, errno));
        Ok(())
    }

    /// Ownership rules this ring saw broken, in order.
    ///
    /// A test asserting this is empty is asserting that the engine took no
    /// step a real kernel would have refused -- answering for a request the
    /// kernel holds, above all.
    #[cfg(feature = "fault-injection")]
    fn fake_violations(&self) -> PyResult<Vec<String>> {
        Ok(self.scripted_state()?.lock().unwrap().violations.clone())
    }

    /// How many times the worker synced the submission queue.
    #[cfg(feature = "fault-injection")]
    fn fake_syncs(&self) -> PyResult<usize> {
        Ok(self.scripted_state()?.lock().unwrap().syncs)
    }

    /// Answer for one request the ring handed to its imaginary kernel.
    ///
    /// Refuses a request the ring does not hold, and records the attempt,
    /// because a completion for a request the kernel owns can only come
    /// from the kernel and one for a request it does not own is invented.
    #[cfg(feature = "fault-injection")]
    fn fake_complete(&self, user_data: u64, result: i32) -> PyResult<()> {
        self.scripted_ring()?
            .complete(user_data, result)
            .map_err(PyValueError::new_err)
    }

    /// Deliver a held request's completion when the worker starts draining.
    ///
    /// Schedule it before calling close: close holds the device's mutable
    /// Python borrow, so another Python thread cannot inject a completion
    /// while it runs. Delivery uses the same ownership check as fake_complete.
    #[cfg(feature = "fault-injection")]
    fn fake_complete_at_shutdown(&self, user_data: u64, result: i32) -> PyResult<()> {
        self.scripted_ring()?
            .complete_at_shutdown(user_data, result)
            .map_err(PyValueError::new_err)
    }

    /// Count scheduled completions delivered inside the shutdown drain.
    #[cfg(feature = "fault-injection")]
    fn fake_shutdown_delivered(&self) -> PyResult<usize> {
        Ok(self.scripted_state()?.lock().unwrap().shutdown_delivered)
    }

    /// Make the next submit take only the first `count` resident entries.
    ///
    /// This is the partial take the submit seam cannot express: the ring
    /// really does hand over a prefix, and the suffix really does stay
    /// resident and still belong to the worker.
    #[cfg(feature = "fault-injection")]
    fn fake_submit_takes(&self, count: usize) -> PyResult<()> {
        self.scripted_state()?.lock().unwrap().take_next = Some(count);
        Ok(())
    }

    /// Make the next submit report an error and take nothing.
    #[cfg(feature = "fault-injection")]
    fn fake_submit_fails(&self, errno: i32) -> PyResult<()> {
        self.scripted_state()?.lock().unwrap().submit_error = Some(errno);
        Ok(())
    }

    /// Errors actually returned by the scripted ring's submit calls.
    #[cfg(feature = "fault-injection")]
    fn fake_submit_errors(&self) -> PyResult<Vec<i32>> {
        Ok(self.scripted_state()?.lock().unwrap().submit_errors.clone())
    }

    /// How many submit calls the worker has made, for a test to key a plan on.
    #[cfg(feature = "fault-injection")]
    fn submit_call_count(&self) -> u64 {
        self.fault_plan
            .as_ref()
            .map(|plan| plan.submit_calls.load(Ordering::SeqCst))
            .unwrap_or(0)
    }

    /// Refused: a synchronous read owns nothing it could retain.
    ///
    /// This path registered no owner for its destination and could not carry
    /// one through an unknown outcome, so on a device whose worker has given
    /// up it handed the buffer back while a read may still have been landing
    /// in it. `batched_read()` followed by `wait_iouring()` owns its
    /// destinations and withholds them; use those.
    #[pyo3(signature = (offset, data, payload_len, total_len = None))]
    fn read_uring(
        &self,
        offset: u64,
        data: &Bound<'_, PyAny>,
        payload_len: usize,
        total_len: Option<usize>,
    ) -> PyResult<()> {
        let _ = (offset, data, payload_len, total_len);
        Err(PyRuntimeError::new_err(
            "RawBlockDevice.read_uring() is not supported; use \
             batched_read() followed by wait_iouring(), which retains its \
             destination buffers when an outcome cannot be established",
        ))
    }

    /// Synchronous write using io_uring.
    ///
    /// `total_len` defaults to `payload_len`. The source buffer is read only
    /// within its bounds and is never modified, and the padding region
    /// `[payload_len, total_len)` is always written as zeroes.
    /// The argument list mirrors one SQE's worth of parameters plus the
    /// attribution the caller carries, which is what it takes to describe
    /// one padded placement-directed write without a side channel.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (offset, data, payload_len, total_len = None, placement_id = None, request_tag = None))]
    fn write_uring(
        &self,
        py: Python<'_>,
        offset: u64,
        data: &Bound<'_, PyAny>,
        payload_len: usize,
        total_len: Option<usize>,
        placement_id: Option<i32>,
        request_tag: Option<String>,
    ) -> PyResult<()> {
        let request_tag: Option<Arc<str>> = request_tag.map(|tag| Arc::from(tag.as_str()));
        if !self.use_iouring {
            return Err(PyRuntimeError::new_err("io_uring not enabled"));
        }
        self.ensure_io_available(true)?;
        let placement_id_u16 = placement_id.map(placement_id_to_u16).transpose()?;
        let nvme_cmd_data = self._build_nvme_cmd_data(
            if placement_id_u16.is_some() { 0x2 } else { 0x0 },
            placement_id_u16.unwrap_or(0),
        )?;

        let view = get_buffer(py, data, false)?;
        let ptr = view.ptr as *const u8;
        if ptr.is_null() {
            view.release();
            return Err(PyValueError::new_err("null buffer pointer"));
        }

        let cap = view.len;
        let total_len = total_len.unwrap_or(payload_len);
        if cap < payload_len {
            view.release();
            return Err(PyValueError::new_err(format!(
                "input buffer too small: cap={cap} need={payload_len}"
            )));
        }
        if total_len < payload_len {
            view.release();
            return Err(PyValueError::new_err("total_len must be >= payload_len"));
        }

        let align = self.alignment;
        if self.use_odirect {
            #[allow(clippy::manual_is_multiple_of)]
            if (offset as usize) % align != 0 {
                view.release();
                return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
            }
            #[allow(clippy::manual_is_multiple_of)]
            if total_len % align != 0 {
                view.release();
                return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
            }
        }

        // Check if the buffer is aligned for O_DIRECT
        let ptr_aligned = if self.use_odirect {
            (ptr as usize).is_multiple_of(align)
        } else {
            true
        };

        // Fixed buffers are pre-registered with io_uring, enabling true zero-copy I/O
        let use_fixed = self.fixed_buffers_registered.load(Ordering::Relaxed);
        let (fixed_idx, fixed_dmabuf) = if use_fixed && ptr_aligned {
            let map = self.fixed_buffer_map.lock().unwrap();
            fixed_buffer_for_range(&map, ptr as usize, total_len)
        } else {
            (None, None)
        };
        if !view.host_accessible && fixed_dmabuf.is_none() {
            view.release();
            return Err(PyValueError::new_err(
                "pointer-only buffer is not fully covered by a dma-buf registration",
            ));
        }

        let prepared = match prepare_iouring_write_buffer(
            ptr as usize,
            cap,
            payload_len,
            total_len,
            self.use_odirect,
            align,
            fixed_idx,
            fixed_dmabuf,
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                view.release();
                return Err(error);
            }
        };
        let use_bounce = prepared.bounce.is_some();

        // Publish ownership only after preparation succeeds. A rejected write
        // has no completion that could retire its buffer owner.
        let batch_id = self.next_batch_id.fetch_add(1, Ordering::Relaxed);
        self.batched_buffer_objs
            .lock()
            .unwrap()
            .insert(batch_id, vec![data.clone().unbind()]);
        let comp = Arc::new(IoCompletion::new());
        let sub = IoSubmission {
            fd: self.fd,
            offset,
            len: total_len,
            ptr_addr: prepared.ptr_addr,
            is_write: true,
            completion: comp.clone(),
            fixed_buffer_idx: prepared.fixed_buffer_idx,
            fixed_dmabuf: prepared.fixed_dmabuf,
            bounce: prepared.bounce,
            original_ptr: None,
            payload_len: if use_bounce { Some(payload_len) } else { None },
            batch_id,
            request_tag,
            attempt: 0,
            nvme_cmd_data,
        };
        if let Err(error) = enqueue_if_running(
            self.queue.as_ref().expect("queue must exist"),
            self.shutdown.as_ref().expect("shutdown must exist"),
            &self.in_flight_count,
            sub,
        ) {
            self.retire_batch_owners(batch_id);
            view.release();
            return Err(error);
        }
        if let Some(batch_ready) = &self.batch_ready {
            batch_ready.signal_producer();
        }
        let res = py.allow_threads(move || comp.wait());

        self.retire_batch_owners(batch_id);
        // Releasing the export calls PyBuffer_Release, which lets the
        // caller's buffer be reused. The device may still be reading a
        // quarantined batch's buffer, so its export is never released: the
        // memory stays pinned for this engine's lifetime, which is the
        // price of not being able to say when the write finished. BufRef
        // has no Drop impl -- release() is an explicit call -- so
        // withholding that call is the whole of the retention.
        let withheld = self.quarantined_batches.lock().unwrap().contains(&batch_id);
        if !withheld {
            view.release();
        }
        res?;
        Ok(())
    }

    /// Batched read: submit multiple reads at once via io_uring.
    /// All reads are queued to the worker thread, which processes them
    /// in batches to maximize throughput.
    ///
    /// Host buffer exports remain pinned until the batch is polled after a
    /// known completion. Unknown outcomes retain those exports permanently.
    /// Callers must not access destination contents before completion.
    ///
    /// Returns a batch ID that must be passed to `wait_iouring()` to wait for
    /// completion and obtain a success bitmap plus sparse completion errors.
    /// Validation or request-preparation errors are raised instead of returning
    /// a batch ID.
    #[pyo3(signature = (offsets, buffers, total_lens, request_tag = None))]
    fn batched_read(
        &self,
        py: Python<'_>,
        offsets: Vec<u64>,
        buffers: Vec<Bound<'_, PyAny>>,
        total_lens: Vec<usize>,
        request_tag: Option<String>,
    ) -> PyResult<u64> {
        let request_tag: Option<Arc<str>> = request_tag.map(|tag| Arc::from(tag.as_str()));
        if !self.use_iouring {
            return Err(PyRuntimeError::new_err("io_uring not enabled"));
        }
        self.ensure_io_available(true)?;

        let n = offsets.len();
        if n == 0 {
            // Return a valid batch_id even for empty batches
            return Ok(self.next_batch_id.fetch_add(1, Ordering::Relaxed));
        }
        if buffers.len() != n || total_lens.len() != n {
            return Err(PyValueError::new_err("All vectors must have same length"));
        }

        // Acquire buffer views to keep them alive until wait_iouring() completes
        let mut views: Vec<BufRef> = Vec::with_capacity(n);
        let mut caps = Vec::with_capacity(n);
        let mut host_accessible = Vec::with_capacity(n);
        for buffer in &buffers {
            let view = match get_buffer(py, buffer, true) {
                Ok(view) => view,
                Err(error) => {
                    for view in views {
                        view.release();
                    }
                    return Err(error);
                }
            };
            if view.readonly {
                for v in views {
                    v.release();
                }
                view.release();
                return Err(PyValueError::new_err("output buffer is readonly"));
            }
            if view.ptr.is_null() {
                for v in views {
                    v.release();
                }
                view.release();
                return Err(PyValueError::new_err("null buffer pointer"));
            }
            caps.push(view.len);
            host_accessible.push(view.host_accessible);
            views.push(view);
        }

        let mut owners = Vec::with_capacity(n);
        for (buffer, view) in buffers.iter().zip(views.iter()) {
            match batch_buffer_owner(buffer, view.host_accessible) {
                Ok(owner) => owners.push(owner),
                Err(error) => {
                    for view in views {
                        view.release();
                    }
                    return Err(error);
                }
            }
        }

        // Generate a unique batch ID for this batch
        let batch_id = self.next_batch_id.fetch_add(1, Ordering::Relaxed);

        // Initialize per-batch tracking for this batch
        {
            let mut batch_map = self.batch_in_flight.lock().unwrap();
            batch_map.insert(
                batch_id,
                (Arc::new(AtomicU64::new(0)), Arc::new(Condvar::new())),
            );
        }

        // Store buffer objects to keep them alive until they complete
        {
            let mut stored_objs = self.batched_buffer_objs.lock().unwrap();
            stored_objs.insert(batch_id, owners);
        }

        // Extract pointers as usize before releasing GIL (raw pointers are not Send)
        let mut ptrs = Vec::with_capacity(n);
        for view in &views {
            ptrs.push(view.ptr as usize);
        }

        for view in views {
            view.release();
        }

        let fd = self.fd;
        let use_odirect = self.use_odirect;
        let use_uring_cmd = self.use_uring_cmd;
        let alignment = self.alignment;
        let fixed_buffers_registered = self.fixed_buffers_registered.load(Ordering::Relaxed);
        // Clone the fixed buffer map before releasing GIL to avoid lock contention
        let fixed_buffer_map: FixedBufferMap = if fixed_buffers_registered {
            let map = self.fixed_buffer_map.lock().unwrap();
            map.clone()
        } else {
            HashMap::new()
        };
        // Get NVMe data for io_uring_cmd
        let nvme_cmd_data = self._build_nvme_cmd_data(0, 0)?;
        let in_flight_count = Arc::clone(&self.in_flight_count);
        let queue = Arc::clone(self.queue.as_ref().unwrap());
        let batch_ready = Arc::clone(self.batch_ready.as_ref().unwrap());
        let batched_completions = Arc::clone(&self.batched_completions);
        let batch_in_flight = Arc::clone(&self.batch_in_flight);

        // Additional clones for cleanup on error path
        let batch_in_flight_cleanup = Arc::clone(&batch_in_flight);
        let batched_completions_cleanup = Arc::clone(&batched_completions);
        let batched_buffer_objs_cleanup = Arc::clone(&self.batched_buffer_objs);

        // Release the GIL while submitting I/O operations
        let res = py.allow_threads(move || {
            let mut submissions: Vec<(IoSubmission, Arc<IoCompletion>)> = Vec::with_capacity(n);

            // Per-item bounce decision mirrors `read_uring`.
            let needs_align = use_odirect || use_uring_cmd;
            for i in 0..n {
                let total_len = total_lens[i];
                let offset = offsets[i];
                let cap = caps[i];

                if use_odirect {
                    #[allow(clippy::manual_is_multiple_of)]
                    if (offset as usize) % alignment != 0 {
                        return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
                    }
                    #[allow(clippy::manual_is_multiple_of)]
                    if total_len % alignment != 0 {
                        return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
                    }
                }

                // cap < total_len is OK (bounce handles it); cap == 0 is invalid.
                if cap == 0 {
                    return Err(PyValueError::new_err(format!(
                        "output buffer too small: cap={} need={}",
                        cap, total_len
                    )));
                }

                let ptr_aligned = if needs_align {
                    ptrs[i].is_multiple_of(alignment)
                } else {
                    true
                };
                let (fixed_idx, fixed_dmabuf) =
                    fixed_buffer_for_range(&fixed_buffer_map, ptrs[i], total_len);
                if !host_accessible[i] && fixed_dmabuf.is_none() {
                    return Err(PyValueError::new_err(
                        "pointer-only buffer is not fully covered by a dma-buf registration",
                    ));
                }
                let use_bounce = !ptr_aligned || cap < total_len;
                if use_bounce && fixed_dmabuf.is_some() {
                    return Err(PyValueError::new_err(
                        "dma-buf registered buffer must be aligned and at least total_len bytes",
                    ));
                }

                let comp = Arc::new(IoCompletion::new());

                let (
                    ptr_addr,
                    fixed_idx,
                    fixed_dmabuf,
                    bounce_opt,
                    original_ptr_opt,
                    payload_len_opt,
                ) = if use_bounce {
                    let bounce = AlignedBuf::new(total_len, alignment)?;
                    let bounce_arc = Arc::new(bounce);
                    let bounce_ptr = bounce_arc.as_mut_ptr() as usize;
                    // Copy-back bounded by caller capacity.
                    let payload_len = std::cmp::min(cap, total_len);
                    (
                        bounce_ptr,
                        None,
                        None,
                        Some(bounce_arc),
                        Some(ptrs[i]),
                        Some(payload_len),
                    )
                } else {
                    // Fixed buffers are pre-registered with io_uring,
                    // enabling true zero-copy I/O.
                    (ptrs[i], fixed_idx, fixed_dmabuf, None, None, None)
                };

                let sub = IoSubmission {
                    fd,
                    offset,
                    len: total_len,
                    ptr_addr,
                    is_write: false, // read operation
                    completion: comp.clone(),
                    fixed_buffer_idx: fixed_idx,
                    fixed_dmabuf,
                    bounce: bounce_opt,
                    original_ptr: original_ptr_opt,
                    payload_len: payload_len_opt,
                    batch_id,
                    nvme_cmd_data: nvme_cmd_data.clone(),
                    request_tag: request_tag.clone(),
                    attempt: 0,
                };

                submissions.push((sub, comp));
            }

            for (sub, comp) in submissions {
                // Increment per-batch in-flight count
                {
                    let batch_map = batch_in_flight.lock().unwrap();
                    if let Some((batch_count, _)) = batch_map.get(&batch_id) {
                        batch_count.fetch_add(1, Ordering::Relaxed);
                    }
                }

                {
                    if let Err(error) = enqueue_if_running(
                        &queue,
                        self.shutdown.as_ref().expect("shutdown must exist"),
                        &in_flight_count,
                        sub,
                    ) {
                        comp.set(Err(error));
                        let batch_map = batch_in_flight.lock().unwrap();
                        if let Some((batch_count, batch_cvar)) = batch_map.get(&batch_id) {
                            if batch_count.fetch_sub(1, Ordering::Relaxed) == 1 {
                                batch_cvar.notify_all();
                            }
                        }
                    }
                }
                batch_ready.signal_producer();

                // Store completion for error checking in wait_iouring
                {
                    let mut completions = batched_completions.lock().unwrap();
                    let batch_completions = completions.entry(batch_id).or_default();
                    batch_completions.push(comp);
                }
            }
            Ok::<(), PyErr>(())
        });

        // Preparation failed, clean up the tracking entries to prevent leaks
        if let Err(e) = res {
            {
                let mut batch_map = batch_in_flight_cleanup.lock().unwrap();
                batch_map.remove(&batch_id);
            }
            {
                let mut stored_objs = batched_buffer_objs_cleanup.lock().unwrap();
                stored_objs.remove(&batch_id);
            }
            {
                let mut completions = batched_completions_cleanup.lock().unwrap();
                completions.remove(&batch_id);
            }
            return Err(e);
        }

        Ok(batch_id)
    }

    /// Write bytes from any Python buffer object into the device.
    /// For O_DIRECT, we use direct pointer I/O when aligned and fallback to
    /// bounce buffering only for the unaligned/padded tail.
    #[pyo3(signature=(offset, data, payload_len=None, total_len=None))]
    fn pwrite_from_buffer(
        &self,
        py: Python<'_>,
        offset: u64,
        data: &Bound<'_, PyAny>,
        payload_len: Option<usize>,
        total_len: Option<usize>,
    ) -> PyResult<()> {
        self.ensure_io_available(false)?;
        let fd = self.fd;

        let view = get_buffer(py, data, false)?;
        if !view.host_accessible {
            view.release();
            return Err(PyValueError::new_err(
                "pointer-only buffers require io_uring dma-buf registration",
            ));
        }
        let ptr = view.ptr as *const u8;
        let buf_len = view.len;
        if ptr.is_null() {
            view.release();
            return Err(PyValueError::new_err("null buffer pointer"));
        }

        // `payload_len`: user bytes to write.
        // `total_len`: actual I/O length. For O_DIRECT this is often aligned up.
        // Example: payload=4100, align=4096 -> total_len=8192.
        let payload_len = payload_len.unwrap_or(buf_len);
        if payload_len > buf_len {
            view.release();
            return Err(PyValueError::new_err("payload_len exceeds buffer length"));
        }
        let total_len = total_len.unwrap_or(payload_len);
        if total_len < payload_len {
            view.release();
            return Err(PyValueError::new_err("total_len must be >= payload_len"));
        }

        let align = self.alignment;
        if self.use_odirect {
            #[allow(clippy::manual_is_multiple_of)]
            if (offset as usize) % align != 0 {
                view.release();
                return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
            }
            #[allow(clippy::manual_is_multiple_of)]
            if total_len % align != 0 {
                view.release();
                return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
            }
        }

        // Store pointer as integer before releasing the GIL. The closure passed
        // to `allow_threads` must own plain data and cannot borrow `view`.
        // We still keep `view` alive until I/O finishes, then release it below.
        let ptr_usize = ptr as usize;
        let res = py.allow_threads(move || {
            let src = ptr_usize as *const u8;
            let src_aligned = (src as usize).is_multiple_of(align);
            if total_len == payload_len && !self.use_odirect {
                // direct write without padding
                return pwrite_from_ptr(fd, offset, src, payload_len);
            }

            if self.use_odirect && src_aligned {
                if total_len == payload_len {
                    // Fully aligned fast path: no copies.
                    return pwrite_from_ptr(fd, offset, src, total_len);
                }

                // Hybrid path for O_DIRECT with padding:
                // - If the Python pointer is aligned, we avoid copying the large
                //   aligned prefix and write it directly.
                // - Only the tail is copied into an aligned bounce buffer, then
                //   zero-padded to satisfy O_DIRECT full-block writes.
                //
                // This keeps copy cost proportional to tail size, not payload size.
                let aligned_prefix = payload_len / align * align;
                if aligned_prefix > 0 {
                    pwrite_from_ptr(fd, offset, src, aligned_prefix)?;
                }
                let tail_payload = payload_len - aligned_prefix;
                let tail_total = total_len - aligned_prefix;
                if tail_total > 0 {
                    let tail_offset = offset
                        .checked_add(aligned_prefix as u64)
                        .ok_or_else(|| PyValueError::new_err("offset overflow"))?;
                    let bounce = AlignedBuf::new(tail_total, align)?;
                    unsafe {
                        if tail_payload > 0 {
                            libc::memcpy(
                                bounce.as_mut_ptr() as *mut libc::c_void,
                                src.add(aligned_prefix) as *const libc::c_void,
                                tail_payload,
                            );
                        }
                        if tail_total > tail_payload {
                            libc::memset(
                                bounce.as_mut_ptr().add(tail_payload) as *mut libc::c_void,
                                0,
                                tail_total - tail_payload,
                            );
                        }
                    }
                    pwrite_from_ptr(fd, tail_offset, bounce.as_ptr(), tail_total)?;
                }
                return Ok(());
            }

            // Full bounce path:
            // - required when source pointer is not alignment-safe for O_DIRECT.
            // - also used when non-O_DIRECT call asks for padding behavior.
            let bounce = AlignedBuf::new(total_len, align)?;
            unsafe {
                libc::memcpy(
                    bounce.as_mut_ptr() as *mut libc::c_void,
                    src as *const libc::c_void,
                    payload_len,
                );
                if total_len > payload_len {
                    libc::memset(
                        bounce.as_mut_ptr().add(payload_len) as *mut libc::c_void,
                        0,
                        total_len - payload_len,
                    );
                }
            }
            pwrite_from_ptr(fd, offset, bounce.as_ptr(), total_len)
        });
        // Always release the CPython buffer view once the blocking I/O closure
        // completes. This decrements exporter-side view count correctly.
        view.release();
        res?;
        Ok(())
    }

    /// Read exactly `payload_len` bytes into a writable Python buffer.
    /// For O_DIRECT, use direct reads when destination is aligned and fallback
    /// to a hybrid/read-bounce path when needed.
    #[pyo3(signature=(offset, out, payload_len, total_len=None))]
    fn pread_into(
        &self,
        py: Python<'_>,
        offset: u64,
        out: &Bound<'_, PyAny>,
        payload_len: usize,
        total_len: Option<usize>,
    ) -> PyResult<()> {
        self.ensure_io_available(false)?;
        let fd = self.fd;
        let view = get_buffer(py, out, true)?;
        if !view.host_accessible {
            view.release();
            return Err(PyValueError::new_err(
                "pointer-only buffers require io_uring dma-buf registration",
            ));
        }
        if view.readonly {
            view.release();
            return Err(PyValueError::new_err("output buffer is readonly"));
        }
        let cap = view.len;
        if cap < payload_len {
            view.release();
            return Err(PyValueError::new_err(format!(
                "output buffer too small: cap={cap} need={payload_len}"
            )));
        }
        let ptr = view.ptr;
        if ptr.is_null() {
            view.release();
            return Err(PyValueError::new_err("null buffer pointer"));
        }

        // `payload_len`: bytes caller wants copied into `out`.
        // `total_len`: bytes to read from device. For O_DIRECT this is usually
        // aligned up and can be larger than payload_len.
        let total_len = total_len.unwrap_or(payload_len);
        if total_len < payload_len {
            view.release();
            return Err(PyValueError::new_err("total_len must be >= payload_len"));
        }

        let align = self.alignment;
        if self.use_odirect {
            #[allow(clippy::manual_is_multiple_of)]
            if (offset as usize) % align != 0 {
                view.release();
                return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
            }
            #[allow(clippy::manual_is_multiple_of)]
            if total_len % align != 0 {
                view.release();
                return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
            }
        }

        // Same pattern as write path: move raw address into closure-safe value
        // while retaining `view` lifetime until closure completion.
        let dst_usize = ptr as usize;
        let res = py.allow_threads(move || {
            let dst = dst_usize as *mut u8;
            let dst_aligned = (dst as usize).is_multiple_of(align);
            if total_len == payload_len && !self.use_odirect {
                return pread_into(fd, offset, dst, payload_len);
            }

            if self.use_odirect && dst_aligned {
                if cap >= total_len {
                    // Fully aligned fast path: no copies.
                    return pread_into(fd, offset, dst, total_len);
                }

                // Hybrid path for O_DIRECT with smaller destination capacity:
                // - read aligned prefix directly into destination.
                // - read aligned tail into bounce buffer.
                // - copy only payload tail bytes back into destination.
                //
                // This avoids writing beyond Python buffer capacity while still
                // honoring O_DIRECT aligned read requirements.
                let aligned_prefix = payload_len / align * align;
                if aligned_prefix > 0 {
                    pread_into(fd, offset, dst, aligned_prefix)?;
                }
                let tail_payload = payload_len - aligned_prefix;
                let tail_total = total_len - aligned_prefix;
                if tail_total > 0 {
                    let tail_offset = offset
                        .checked_add(aligned_prefix as u64)
                        .ok_or_else(|| PyValueError::new_err("offset overflow"))?;
                    let bounce = AlignedBuf::new(tail_total, align)?;
                    pread_into(fd, tail_offset, bounce.as_mut_ptr(), tail_total)?;
                    unsafe {
                        if tail_payload > 0 {
                            libc::memcpy(
                                dst.add(aligned_prefix) as *mut libc::c_void,
                                bounce.as_ptr() as *const libc::c_void,
                                tail_payload,
                            );
                        }
                    }
                }
                return Ok(());
            }

            // Full bounce read path:
            // read aligned size into temporary aligned memory, then copy the
            // requested payload portion to Python output buffer.
            let bounce = AlignedBuf::new(round_up(total_len, align), align)?;
            pread_into(fd, offset, bounce.as_mut_ptr(), total_len)?;
            unsafe {
                libc::memcpy(
                    dst as *mut libc::c_void,
                    bounce.as_ptr() as *const libc::c_void,
                    payload_len,
                );
            }
            Ok(())
        });
        view.release();
        res?;
        Ok(())
    }

    /// Flush completed writes and device caches to persistent storage.
    ///
    /// Callers must await completion of every write this barrier must make
    /// durable. Unrelated I/O may run concurrently; this does not substitute
    /// for waiting on those operations. The kernel performs the file or block
    /// device fsync without the GIL. Storage must honor flush commands for
    /// the resulting durability guarantee to hold.
    ///
    /// Raises RuntimeError for closed or poisoned devices, and OSError
    /// if the kernel cannot complete the flush.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        self.ensure_io_available(false)?;
        py.allow_threads(|| loop {
            if unsafe { libc::fsync(self.fd) } == 0 {
                return Ok(());
            }
            if errno() != libc::EINTR {
                return Err(os_err("fsync failed"));
            }
        })
    }

    /// Close the device after draining accepted I/O, without holding the GIL.
    ///
    /// Repeated successful calls are harmless. A bounded drain that cannot
    /// prove every operation has stopped raises RuntimeError and retains
    /// the ring and its owners. Raises OSError if the descriptor close fails.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if !self.closed.load(Ordering::Relaxed) {
            py.allow_threads(|| self.do_close())?;
        }
        Ok(())
    }

    /// Report whether every admitted operation has a known terminal outcome.
    fn is_idle(&self) -> bool {
        self.in_flight_count.load(Ordering::SeqCst) == 0 && !self.outcome_is_unknown()
    }

    /// Stop and join the worker while retaining the device and its registrations.
    fn drain_worker(&mut self, py: Python<'_>) {
        py.allow_threads(|| self.do_drain_worker());
    }
}

impl RawBlockDevice {
    /// Release or quarantine a finished batch's Python buffer owners.
    ///
    /// A batch whose outcome the worker could not establish must keep them:
    /// the device may still reach the memory they describe, and handing a
    /// pool slice back to an allocator on the strength of a logical failure
    /// is what this distinction exists to prevent. Logical completion is not
    /// reusable capacity.
    fn retire_batch_owners(&self, batch_id: u64) {
        // The worker already moved this batch's owners if it marked the
        // batch unknown, so that case is a no-op here rather than a second
        // transfer. This is still the only release for a healthy batch, and
        // for one the worker never saw.
        if self.quarantined_batches.lock().unwrap().contains(&batch_id) {
            drain_batch_owners(
                &self.batched_buffer_objs,
                &self.quarantined_owners,
                Some(batch_id),
            );
            return;
        }
        let mut stored_objs = self.batched_buffer_objs.lock().unwrap();
        stored_objs.remove(&batch_id);
    }

    /// Refuse replacing a live table or registering on an unusable device.
    fn ensure_registration_available(&self) -> PyResult<()> {
        self.ensure_io_available(self.use_iouring)?;
        if self.fixed_buffers_registered.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err(
                "fixed buffers are already registered",
            ));
        }
        Ok(())
    }

    /// Internal function to perform the cleanup operation.
    /// Stop the worker and wait for it, without tearing anything down.
    ///
    /// The worker is joined before anything is examined. Quarantining a
    /// submission is itself what decrements the in-flight count this waits
    /// on, so reaching zero is not evidence the device is finished -- it is
    /// reached *by* giving up. Only after the worker has stopped is the
    /// quarantine complete enough to ask about.
    ///
    /// Separated from the teardown so a caller can learn whether the device
    /// is quiet while it still has a device to write with. A writer's final
    /// index is the case that needs this: deciding to publish it from a
    /// question asked before the worker stopped publishes a manifest naming
    /// extents whose writes were never observed.
    fn do_drain_worker(&mut self) {
        // Idempotent, and it has to be: close() drains for a caller that
        // did not ask, and the wait below is on a counter only the worker
        // decrements. Running it a second time with the worker already
        // joined would wait for a decrement that can never come.
        if self.worker.is_none() {
            return;
        }
        if self.use_iouring {
            if let (Some(queue), Some(shutdown)) = (&self.queue, &self.shutdown) {
                stop_submissions(queue, shutdown);
            }
            if let Some(batch_ready) = &self.batch_ready {
                batch_ready.signal_producer();
            }

            let mutex = Mutex::new(());
            let mut guard = mutex.lock().unwrap();
            while self.in_flight_count.load(Ordering::Relaxed) > 0 {
                let (g, _) = self
                    .in_flight_cvar
                    .wait_timeout(guard, Duration::from_millis(10))
                    .unwrap();
                guard = g;
            }
        }
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }

    fn do_close(&mut self) -> Result<(), PyErr> {
        if self.use_iouring {
            self.do_drain_worker();

            if self.outcome_is_unknown() {
                return Err(self.retain_everything(
                    "io_uring engine could not establish what the device is \
                     still doing; its buffer registration, exported buffers \
                     and device descriptor are retained rather than torn \
                     down, and this device is not reusable",
                ));
            }

            if self.fixed_buffers_registered.load(Ordering::Relaxed) {
                let unregistered = match &self.ring {
                    Some(ring) => ring.unregister_buffers(),
                    None => Ok(()),
                };
                if let Err(e) = unregistered {
                    // The kernel still holds a registration this process was
                    // about to forget. Forgetting it is what would let the
                    // memory behind it be reused, so this is the same
                    // situation as an unknown completion and is treated as
                    // one.
                    self.poisoned.store(true, Ordering::SeqCst);
                    return Err(self.retain_everything(&format!(
                        "io_uring buffer unregister failed ({e}); the \
                         registration the kernel still holds is retained \
                         rather than forgotten, and this device is not \
                         reusable"
                    )));
                }
                self.fixed_buffers_registered
                    .store(false, Ordering::Relaxed);
                self.fixed_buffer_map.lock().unwrap().clear();
            }
        } else {
            self.do_drain_worker();
        }

        let rc = unsafe { libc::close(self.fd) };
        if rc != 0 {
            return Err(os_err("close failed"));
        }
        self.closed.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Whether this device has stopped being able to say what it is doing.
    fn outcome_is_unknown(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst) || !self.quarantined_batches.lock().unwrap().is_empty()
    }

    /// Keep every resource an unproven outcome may still be reaching.
    ///
    /// The ring, its registration and the device descriptor stay open,
    /// because closing them is what releases the kernel's attachments to
    /// memory a command may still be landing in. The retained Python owners
    /// are leaked deliberately, for the same reason the worker leaks its
    /// quarantined submissions: their pool slices must not go back to an
    /// allocator that would hand them to the next request.
    ///
    /// `closed` is deliberately left false. This device is finished, but
    /// saying it is closed would invite a caller to conclude its resources
    /// were released.
    fn retain_everything(&mut self, reason: &str) -> PyErr {
        // Everything still held by a batch, not only batches somebody
        // marked. A registration that failed to unregister covers whatever
        // memory it named, and a caller that never polled left its owners
        // where only this sweep will find them. Over-retaining a
        // proven-complete unpolled batch is the deliberate side to err on.
        let swept = drain_batch_owners(&self.batched_buffer_objs, &self.quarantined_owners, None);
        let owners = std::mem::take(&mut *self.quarantined_owners.lock().unwrap());
        let owner_count = owners.len();
        self.retained_owners
            .fetch_add(owner_count, Ordering::SeqCst);
        std::mem::forget(owners);
        let registered = std::mem::take(&mut *self.fixed_buffer_map.lock().unwrap());
        std::mem::forget(registered);
        if let Some(ring) = self.ring.take() {
            std::mem::forget(ring);
        }
        PyRuntimeError::new_err(format!(
            "{reason} (retained {owner_count} buffer owner(s), {swept} of them \
             from batches nobody polled, {batches} quarantined batch(es))",
            batches = self.quarantined_batches.lock().unwrap().len(),
        ))
    }
}

impl Drop for RawBlockDevice {
    fn drop(&mut self) {
        if !self.closed.load(Ordering::Relaxed) {
            Python::with_gil(|py| {
                let _ = py.allow_threads(|| self.do_close());
            });
        }
    }
}

#[pymodule]
fn lmcache_rust_raw_block_io(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RawBlockDevice>()?;
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod short_io_action_tests {
    use super::{short_io_action, ShortIoAction};

    #[test]
    fn a_full_transfer_is_not_short() {
        assert_eq!(
            short_io_action(4096, 4096, false, false),
            ShortIoAction::Complete
        );
    }

    #[test]
    fn an_error_is_not_short() {
        assert_eq!(
            short_io_action(-5, 4096, false, false),
            ShortIoAction::Complete
        );
    }

    #[test]
    fn a_short_regular_transfer_retries_its_remainder() {
        assert_eq!(
            short_io_action(1024, 4096, false, false),
            ShortIoAction::Retry
        );
    }

    #[test]
    fn a_command_passthrough_completion_is_never_retried() {
        assert_eq!(
            short_io_action(1024, 4096, true, false),
            ShortIoAction::Complete
        );
    }

    #[test]
    fn a_short_registered_dmabuf_transfer_ends_terminally() {
        // The retry would have to re-express a byte offset into the
        // registered buffer; this engine does not carry that path, so the
        // operation must fail rather than be pushed again.
        assert_eq!(
            short_io_action(1024, 4096, false, true),
            ShortIoAction::FailTerminally
        );
    }

    #[test]
    fn a_full_registered_dmabuf_transfer_is_unaffected() {
        assert_eq!(
            short_io_action(4096, 4096, false, true),
            ShortIoAction::Complete
        );
    }
}

#[cfg(test)]
mod drain_batch_owners_tests {
    use super::drain_batch_owners;
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[test]
    fn one_batch_moves_and_a_second_call_moves_nothing() {
        let batched = Mutex::new(HashMap::from([(7u64, vec!["a", "b"])]));
        let quarantined = Mutex::new(Vec::new());
        assert_eq!(drain_batch_owners(&batched, &quarantined, Some(7)), 2);
        assert_eq!(quarantined.lock().unwrap().len(), 2);
        // Idempotent: the worker moves a batch when it learns the outcome,
        // and a later poll must not try to move it again.
        assert_eq!(drain_batch_owners(&batched, &quarantined, Some(7)), 0);
        assert_eq!(quarantined.lock().unwrap().len(), 2);
    }

    #[test]
    fn an_unknown_batch_id_moves_nothing() {
        let batched = Mutex::new(HashMap::from([(7u64, vec!["a"])]));
        let quarantined = Mutex::new(Vec::new());
        assert_eq!(drain_batch_owners(&batched, &quarantined, Some(8)), 0);
        assert!(quarantined.lock().unwrap().is_empty());
        assert_eq!(batched.lock().unwrap().len(), 1);
    }

    #[test]
    fn sweeping_takes_every_batch_including_unpolled_ones() {
        let batched = Mutex::new(HashMap::from([(1u64, vec!["a"]), (2u64, vec!["b", "c"])]));
        let quarantined = Mutex::new(vec!["already"]);
        assert_eq!(drain_batch_owners(&batched, &quarantined, None), 3);
        assert!(batched.lock().unwrap().is_empty());
        assert_eq!(quarantined.lock().unwrap().len(), 4);
    }
}

#[cfg(test)]
mod io_journal_tests {
    use super::{IoSubmission, NativeIoJournal, NvmeCmdData};

    #[test]
    fn nvme_success_counts_the_transfer_instead_of_the_status() {
        let journal = NativeIoJournal::new(8);
        let sub = IoSubmission {
            len: 4096,
            nvme_cmd_data: Some(NvmeCmdData {
                nsid: 1,
                lba_shift: 12,
                dtype: 0,
                dspec: 0,
            }),
            ..Default::default()
        };
        journal.record_completion(&sub, 7, 0);
        journal.record_completion(&sub, 8, 1);
        let (events, dropped) = journal.drain();
        assert_eq!(dropped, 0);
        assert_eq!(events[0].outcome, "completed");
        assert_eq!(events[0].bytes, 4096);
        assert_eq!(events[1].outcome, "failed");
        assert_eq!(events[1].bytes, 0);
    }

    #[test]
    fn short_and_zero_byte_completions_keep_their_actual_counts() {
        let journal = NativeIoJournal::new(8);
        let sub = IoSubmission {
            len: 4096,
            ..Default::default()
        };
        journal.record_completion(&sub, 7, 2048);
        journal.record_completion(&sub, 8, 0);
        let (events, dropped) = journal.drain();
        assert_eq!(dropped, 0);
        assert_eq!(events[0].outcome, "short");
        assert_eq!(events[0].bytes, 2048);
        assert_eq!(events[1].outcome, "short");
        assert_eq!(events[1].bytes, 0);
    }

    #[test]
    fn dropped_rows_are_returned_with_the_snapshot_that_lost_them() {
        let journal = NativeIoJournal::new(1);
        let sub = IoSubmission::default();
        journal.record_submission(&sub, 7);
        journal.record_completion(&sub, 7, 0);
        let (events, dropped) = journal.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(dropped, 1);
        let (events, dropped) = journal.drain();
        assert!(events.is_empty());
        assert_eq!(dropped, 0);
    }
}

#[cfg(test)]
mod encoded_sqe_oracle_tests {
    use super::{build_regular_sqe, IoSubmission, SqeDescriptor};
    use io_uring::squeue;
    use std::mem::size_of_val;

    /// Decode the stable first 42 bytes of Linux's encoded 64-byte SQE.
    ///
    /// This deliberately does not call `SqeDescriptor::describe`: the point
    /// is to read back the bytes handed to io_uring and compare the diagnostic
    /// description against an independently decoded kernel ABI structure.
    fn encoded_geometry(entry: &squeue::Entry) -> (u8, i32, u64, u64, u32, u64, u16) {
        let raw = unsafe {
            std::slice::from_raw_parts(
                (entry as *const squeue::Entry).cast::<u8>(),
                size_of_val(entry),
            )
        };
        assert!(raw.len() >= 42);
        (
            raw[0],
            i32::from_ne_bytes(raw[4..8].try_into().unwrap()),
            u64::from_ne_bytes(raw[8..16].try_into().unwrap()),
            u64::from_ne_bytes(raw[16..24].try_into().unwrap()),
            u32::from_ne_bytes(raw[24..28].try_into().unwrap()),
            u64::from_ne_bytes(raw[32..40].try_into().unwrap()),
            u16::from_ne_bytes(raw[40..42].try_into().unwrap()),
        )
    }

    #[test]
    fn descriptor_matches_the_worker_encoded_sqe() {
        // Cover both directions, fixed and ordinary addresses, and a remainder
        // whose offsets have advanced independently from its original buffer.
        for is_write in [false, true] {
            for fixed_buffer_idx in [None, Some(3)] {
                for fixed_dmabuf in [None, Some(12288)] {
                    if fixed_dmabuf.is_some() && fixed_buffer_idx.is_none() {
                        continue;
                    }
                    for advanced in [0, 4096] {
                        let sub = IoSubmission {
                            fd: 17,
                            offset: 8192 + advanced as u64,
                            ptr_addr: 0x10000 + advanced,
                            len: 8192 - advanced,
                            is_write,
                            fixed_buffer_idx,
                            fixed_dmabuf: fixed_dmabuf.map(|offset| offset + advanced),
                            ..Default::default()
                        };
                        let user_data = 0x1122_3344_5566_7788;
                        let entry = build_regular_sqe(&sub, user_data);
                        let (opcode, fd, offset, addr, len, encoded_user_data, index) =
                            encoded_geometry(&entry);
                        let expected_opcode = match (is_write, fixed_buffer_idx.is_some()) {
                            (false, false) => 22, // IORING_OP_READ
                            (true, false) => 23,  // IORING_OP_WRITE
                            (false, true) => 4,   // IORING_OP_READ_FIXED
                            (true, true) => 5,    // IORING_OP_WRITE_FIXED
                        };
                        assert_eq!(opcode, expected_opcode);
                        assert_eq!(fd, 17);
                        assert_eq!(offset, 8192 + advanced as u64);
                        assert_eq!(
                            addr,
                            fixed_dmabuf.unwrap_or(0x10000) as u64 + advanced as u64
                        );
                        assert_eq!(len, 8192 - advanced as u32);
                        assert_eq!(encoded_user_data, user_data);
                        assert_eq!(index, fixed_buffer_idx.unwrap_or(0));

                        let described = SqeDescriptor::describe(&sub, user_data);
                        assert!(!described.is_cmd);
                        assert_eq!(described.is_write, is_write);
                        assert_eq!(described.offset, offset);
                        assert_eq!(described.addr, addr);
                        assert_eq!(described.len, len);
                        assert_eq!(described.user_data, encoded_user_data);
                        assert_eq!(
                            described.fixed_index,
                            fixed_buffer_idx.map_or(-1, i64::from)
                        );
                        assert_eq!(
                            described.dmabuf_offset,
                            sub.fixed_dmabuf.map_or(-1, |offset| offset as i64)
                        );
                    }
                }
            }
        }
    }
}
