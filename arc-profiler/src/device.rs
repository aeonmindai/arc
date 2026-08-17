//! Device-time measurement.
//!
//! # Why this module exists at all
//!
//! CUDA kernel launches are asynchronous. A host timer wrapped around a launch
//! measures the *launch*, which is microseconds, for work that may take
//! milliseconds. A profiler built only on [`std::time::Instant`] therefore
//! reports ~0 for the GPU and blames the host for everything — and it looks
//! authoritative while doing it. That failure mode is worse than having no
//! profiler.
//!
//! So device time is measured with CUDA events recorded **on candle's own
//! stream**, and it is reported in a field that host time can never write to.
//! [`crate::tree`] enforces that separation structurally: a `Host` node has no
//! way to set `device_ns`.
//!
//! # What `device_ns` actually means — read this before quoting a number
//!
//! `cudaEventElapsedTime(a, b)` is the wall time **on the stream** between the
//! two records. If the host is too slow to keep the stream fed, the stream sits
//! idle between kernels and that idle time is *inside* the interval. So
//! `device_ns` is **stream-elapsed time, not kernel-busy time**.
//!
//! This is a feature, not a caveat to apologise for, because of what the tree
//! then gives you for free:
//!
//! * a node's `device_self_ns` (its own stream time minus its children's) is
//!   the stream time that belongs to **no instrumented kernel** — i.e. launch
//!   gaps plus anything uninstrumented;
//! * so `forward.device_self_ns` large while the leaves are small **is** launch
//!   starvation, measured rather than argued.
//!
//! What it is *not*: SM occupancy, achieved bandwidth, or per-kernel time under
//! overlap. Those need Nsight. See `docs/engineering/PROFILING.md` §Limits.
//!
//! # Resolution is deferred on purpose
//!
//! Querying an event forces a synchronise. Doing that inside a span would
//! serialise the pipeline and change the very thing being measured. Instead the
//! records are queued and resolved once, at `end_step`, after a single flush
//! that the engine's own logits D2H already performs anyway.

use std::sync::atomic::{AtomicU64, Ordering};

/// An opaque index into the timer's event pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventHandle(pub u32);

/// A source of device timestamps.
///
/// A trait rather than a concrete CUDA type so the accounting can be tested on
/// a laptop, and — more importantly — so the *failure mode* can be tested: see
/// `LaunchOnlyTimer` in the crate's tests, which imitates a profiler that
/// mistakenly reports launch time as device time, and which the reconciliation
/// check is required to catch.
pub trait DeviceTimer: Send + Sync {
    /// Enqueue a timestamp on the device timeline. `None` if the pool is
    /// exhausted or the backend is unavailable, in which case the caller
    /// degrades to a host-only span rather than reporting a wrong number.
    fn record(&self) -> Option<EventHandle>;

    /// Nanoseconds between two previously recorded handles. Only valid after
    /// [`DeviceTimer::flush`]. `None` if the backend could not resolve them.
    fn resolve(&self, start: EventHandle, end: EventHandle) -> Option<u64>;

    /// Block until everything recorded so far has completed.
    fn flush(&self);

    /// Return handles to the pool for reuse.
    fn release(&self, handles: &[EventHandle]);

    fn backend(&self) -> &'static str;
}

/// The no-GPU backend. Records nothing, resolves nothing.
///
/// It deliberately returns `None` from [`DeviceTimer::record`] rather than
/// pretending to succeed and reporting 0 ns: a device span on a CPU run is
/// *unmeasured*, and the report says so, instead of publishing a zero that
/// reads like "the GPU was instantaneous".
#[derive(Debug, Default)]
pub struct NullTimer;

impl DeviceTimer for NullTimer {
    fn record(&self) -> Option<EventHandle> {
        None
    }
    fn resolve(&self, _s: EventHandle, _e: EventHandle) -> Option<u64> {
        None
    }
    fn flush(&self) {}
    fn release(&self, _h: &[EventHandle]) {}
    fn backend(&self) -> &'static str {
        "null"
    }
}

/// How many events the CUDA backend has created, for the report's overhead
/// section and for spotting a pool that grows without bound.
pub static CUDA_EVENTS_CREATED: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::{DeviceTimer, EventHandle, CUDA_EVENTS_CREATED};
    use std::ffi::c_void;
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;

    // Linked from the CUDA runtime, which every `cuda`-featured build of this
    // workspace already links (see `arc-cuda-graph::decode_forward`, which uses
    // the identical extern block).
    extern "C" {
        fn cudaEventCreate(event: *mut *mut c_void) -> u32;
        fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> u32;
        fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> u32;
        fn cudaStreamSynchronize(stream: *mut c_void) -> u32;
    }

    struct Pool {
        events: Vec<*mut c_void>,
        free: Vec<u32>,
    }

    // SAFETY: CUDA events are process-global handles; every access here is
    // behind the `Mutex` that owns the `Pool`, and the raw pointers are only
    // ever passed back to the CUDA runtime, which is itself thread-safe.
    unsafe impl Send for Pool {}

    pub struct CudaTimer {
        stream: *mut c_void,
        pool: Mutex<Pool>,
    }

    // SAFETY: `stream` is a handle owned by the candle device that outlives the
    // timer; the CUDA runtime permits recording on one stream from several
    // threads.
    unsafe impl Send for CudaTimer {}
    unsafe impl Sync for CudaTimer {}

    impl CudaTimer {
        /// `stream` must be candle's stream for the device being profiled —
        /// events recorded on a *different* stream would time an unrelated
        /// timeline and quietly produce plausible nonsense.
        pub fn new(stream: *mut c_void) -> Self {
            Self {
                stream,
                pool: Mutex::new(Pool {
                    events: Vec::new(),
                    free: Vec::new(),
                }),
            }
        }
    }

    impl DeviceTimer for CudaTimer {
        fn record(&self) -> Option<EventHandle> {
            let mut pool = self.pool.lock().ok()?;
            let idx = match pool.free.pop() {
                Some(i) => i,
                None => {
                    let mut ev: *mut c_void = std::ptr::null_mut();
                    // SAFETY: `ev` is a valid out-pointer; failure is checked.
                    let rc = unsafe { cudaEventCreate(&mut ev) };
                    if rc != 0 || ev.is_null() {
                        return None;
                    }
                    CUDA_EVENTS_CREATED.fetch_add(1, Ordering::Relaxed);
                    pool.events.push(ev);
                    (pool.events.len() - 1) as u32
                }
            };
            let ev = pool.events[idx as usize];
            // SAFETY: `ev` came from `cudaEventCreate`; `self.stream` is a live
            // candle stream.
            let rc = unsafe { cudaEventRecord(ev, self.stream) };
            if rc != 0 {
                pool.free.push(idx);
                return None;
            }
            Some(EventHandle(idx))
        }

        fn resolve(&self, start: EventHandle, end: EventHandle) -> Option<u64> {
            let pool = self.pool.lock().ok()?;
            let a = *pool.events.get(start.0 as usize)?;
            let b = *pool.events.get(end.0 as usize)?;
            let mut ms: f32 = 0.0;
            // SAFETY: both handles were produced by `record` on this pool and
            // `flush` has been called, so both events have completed.
            let rc = unsafe { cudaEventElapsedTime(&mut ms, a, b) };
            if rc != 0 || !ms.is_finite() || ms < 0.0 {
                return None;
            }
            Some((ms as f64 * 1.0e6) as u64)
        }

        fn flush(&self) {
            // SAFETY: `self.stream` is a live candle stream.
            unsafe {
                cudaStreamSynchronize(self.stream);
            }
        }

        fn release(&self, handles: &[EventHandle]) {
            if let Ok(mut pool) = self.pool.lock() {
                for h in handles {
                    pool.free.push(h.0);
                }
            }
        }

        fn backend(&self) -> &'static str {
            "cuda-events"
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::CudaTimer;

/// Build the best timer available for `device`.
///
/// Returns [`NullTimer`] for anything that is not a CUDA device, including
/// Metal — Metal has no equivalent of a stream-recorded event that candle
/// exposes, so Metal device spans are reported as *unmeasured*, not as zero.
pub fn timer_for(device: &candle_core::Device) -> Box<dyn DeviceTimer> {
    #[cfg(feature = "cuda")]
    {
        if let candle_core::Device::Cuda(dev) = device {
            let stream = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;
            if !stream.is_null() {
                return Box::new(CudaTimer::new(stream));
            }
        }
    }
    let _ = device;
    Box::new(NullTimer)
}

/// Count of device spans whose events could not be resolved. Surfaced in the
/// report so "missing" never silently reads as "zero".
pub static UNRESOLVED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn note_unresolved() {
    UNRESOLVED.fetch_add(1, Ordering::Relaxed);
}
