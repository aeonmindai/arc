//! `arc-profiler` — a reusable hierarchical profiler for the token path.
//!
//! # What problem this solves
//!
//! Arc's decode step was, until this crate, described by a four-bucket split
//! (`moe / mla_attn / mhc_attn_pre / mhc_ffn_pre`) and a `STEP_us` line that
//! only exists on the PagedAttention branch — which the V4 path never takes.
//! Four buckets cannot locate a 150x gap, and a `~24 ms/sequence` linear term
//! worth roughly 69% of a B=64 step has stayed unattributed for exactly that
//! reason.
//!
//! So: a **span tree**, not a bucket list. Nested scopes with real parent/child
//! links, covering the whole path — scheduler, input prep, cache in/out,
//! forward, per-layer, attention, MoE, sampling, detokenisation, response
//! dispatch — with every node carrying host time, device time, sync/wait time,
//! call count, and batch geometry.
//!
//! # The three time channels, and why they are not one number
//!
//! | channel | what it is | how it is measured |
//! |---|---|---|
//! | **wall** | wall-clock inside the span | [`Instant`] |
//! | **device** | GPU stream time | CUDA events on candle's stream |
//! | **sync** | host *blocked* on the GPU | [`Instant`], on spans declared [`sync_span`] |
//!
//! For a [`device_span`] the wall clock is the **launch** cost and is typically
//! a thousand times smaller than the device time. A profiler that reported one
//! number would credit the host with all of it. The distinction between "the
//! host was busy" and "the host was waiting" is the answer this tool exists to
//! produce, so it is a structural property of the type system here: a
//! [`NodeKind::Host`] node has no path by which `device_ns` can be written.
//!
//! Read [`device`] before quoting a device number — `device_ns` is
//! stream-*elapsed*, which includes launch gaps, and that is deliberate.
//!
//! # Cost when off
//!
//! One relaxed atomic load and a predictable branch per call site. In
//! particular **no `env::var_os` per timer call** — a previous iteration of the
//! V4 timers did that and paid ~390 environment scans per forward with
//! profiling *disabled*. The gate resolves once, at first use.
//!
//! Both states are measured, not asserted: with `ARC_PROFILE=1` the profiler
//! calibrates itself at startup and the resulting ns/span figures ship in the
//! report's `overhead` block, along with the share of a step they represent.
//! If that share is large the report says the profile is partly measuring
//! itself.
//!
//! # Environment
//!
//! | variable | default | meaning |
//! |---|---|---|
//! | `ARC_PROFILE` | unset | `1` enables. Anything else, or unset, disables. |
//! | `ARC_PROFILE_DEPTH` | `12` | max nesting depth. 12 reaches the deepest node V4 has
//!   (`...moe.experts.experts.fast.experts.gate_proj`); lower runs cheaper. |
//! | `ARC_PROFILE_WARMUP` | `4` | steps discarded before recording starts |
//! | `ARC_PROFILE_STEPS` | unset | auto-write the report after N recorded steps |
//! | `ARC_PROFILE_OUT` | `./arc-profile` | output directory |
//! | `ARC_PROFILE_LABEL` | `run` | file stem and the run's name in the HTML |
//! | `ARC_PROFILE_UNROLL` | unset | `1` gives one node per layer index |
//! | `ARC_PROFILE_NO_CALIBRATE` | unset | `1` skips the startup self-calibration |
//! | `ARC_PROFILE_SELFTEST` | unset | `1` proves at load that device time is not launch time |
//!
//! # Using it
//!
//! ```no_run
//! # fn work() {}
//! let _s = arc_profiler::span("scheduler.select");
//! work();
//! // dropped at end of scope; nesting comes from the drop order
//! ```

pub mod device;
pub mod html;
pub mod report;
#[cfg(test)]
mod tests;
mod tree;

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Instant;

use device::{DeviceTimer, EventHandle, NullTimer};
pub use report::{
    Geometry, Node, NodeKind, Overhead, Profile, Reconciliation, RunHeader, Totals, Unreachable,
    Violation, SCHEMA,
};
use tree::{Registry, NO_PARENT};

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

const S_OFF: u8 = 0;
const S_ON: u8 = 1;
const S_UNINIT: u8 = 2;

static STATE: AtomicU8 = AtomicU8::new(S_UNINIT);

/// Is the profiler recording?
///
/// A single relaxed load plus a branch. Deliberately *not* a `OnceLock<bool>`
/// deref and emphatically not an environment read: this is called at every
/// instrumented site, several hundred times per forward, in a binary that ships
/// with profiling off.
#[inline(always)]
pub fn enabled() -> bool {
    match STATE.load(Relaxed) {
        S_ON => true,
        S_OFF => false,
        _ => init_gate(),
    }
}

#[cold]
fn init_gate() -> bool {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let on = std::env::var("ARC_PROFILE").ok().as_deref() == Some("1");
        MAX_DEPTH.store(env_u32("ARC_PROFILE_DEPTH", 12), Relaxed);
        WARMUP.store(env_u32("ARC_PROFILE_WARMUP", 4) as u64, Relaxed);
        UNROLL.store(
            std::env::var("ARC_PROFILE_UNROLL").ok().as_deref() == Some("1"),
            Relaxed,
        );
        STATE.store(if on { S_ON } else { S_OFF }, Relaxed);
        if on {
            // Calibration must run before any real span so the scratch nodes it
            // creates are wiped by the same warmup reset that clears warmup
            // steps, and so its numbers describe an un-warmed process the same
            // way the shipped binary experiences one.
            if std::env::var("ARC_PROFILE_NO_CALIBRATE").ok().as_deref() != Some("1") {
                calibrate();
            }
            tracing::info!(
                "arc-profiler ON (depth={} warmup={} unroll={}) — device time via CUDA events; \
                 host/device/sync reported separately",
                MAX_DEPTH.load(Relaxed),
                WARMUP.load(Relaxed),
                UNROLL.load(Relaxed),
            );
        }
    });
    STATE.load(Relaxed) == S_ON
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static MAX_DEPTH: AtomicU32 = AtomicU32::new(12);
static WARMUP: AtomicU64 = AtomicU64::new(4);
static UNROLL: AtomicBool = AtomicBool::new(false);

static SPANS_OPENED: AtomicU64 = AtomicU64::new(0);
static SPANS_DROPPED_DEPTH: AtomicU64 = AtomicU64::new(0);
static STEPS_SEEN: AtomicU64 = AtomicU64::new(0);
static STEPS_RECORDED: AtomicU64 = AtomicU64::new(0);
static TOKENS: AtomicU64 = AtomicU64::new(0);
static MISNESTED: AtomicU64 = AtomicU64::new(0);
static WROTE_ONCE: AtomicBool = AtomicBool::new(false);

/// Sentinel node id meaning "this span is not recording".
const INACTIVE: u32 = u32::MAX;

fn registry() -> &'static RwLock<Registry> {
    static R: OnceLock<RwLock<Registry>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(Registry::default()))
}

type Pending = Mutex<Vec<(u32, EventHandle, EventHandle)>>;
fn pending() -> &'static Pending {
    static P: OnceLock<Pending> = OnceLock::new();
    P.get_or_init(|| Mutex::new(Vec::new()))
}

/// Install the CUDA timer for `device`.
///
/// Must be called once, before the first [`device_span`], with the device the
/// model actually runs on. Calling it later, or with the wrong device, leaves
/// device spans unmeasured — which the report shows as *unmeasured*, never as
/// zero.
pub fn attach_device(dev: &candle_core::Device) {
    if !enabled() {
        return;
    }
    let _ = ATTACHED.get_or_init(|| device::timer_for(dev));
    set_meta(|h| h.device = format!("{dev:?}"));
}

static ATTACHED: OnceLock<Box<dyn DeviceTimer>> = OnceLock::new();

/// Tests substitute their own timer here — including one that deliberately
/// reproduces the "I timed the launch, not the kernel" bug, so the checks that
/// are supposed to catch it can be shown to catch it.
#[cfg(test)]
static TEST_TIMER: RwLock<Option<Box<dyn DeviceTimer>>> = RwLock::new(None);

/// Borrow the active device timer.
///
/// A closure rather than a `&'static` because the test override lives behind a
/// lock. In non-test builds this compiles to a null-check and a call.
#[cfg(not(test))]
#[inline]
fn with_timer<R>(f: impl FnOnce(&dyn DeviceTimer) -> R) -> R {
    match ATTACHED.get() {
        Some(t) => f(&**t),
        None => f(&NullTimer),
    }
}

#[cfg(test)]
fn with_timer<R>(f: impl FnOnce(&dyn DeviceTimer) -> R) -> R {
    if let Ok(g) = TEST_TIMER.read() {
        if let Some(t) = g.as_ref() {
            return f(&**t);
        }
    }
    match ATTACHED.get() {
        Some(t) => f(&**t),
        None => f(&NullTimer),
    }
}

fn meta() -> &'static RwLock<RunHeader> {
    static M: OnceLock<RwLock<RunHeader>> = OnceLock::new();
    M.get_or_init(|| RwLock::new(RunHeader::default()))
}

/// Mutate the run header (model name, artifact, gpu, commit, ...).
///
/// Provenance is not decoration: a JSON file that cannot say which artifact and
/// which commit produced it is an orphan number, and this repository has a
/// documented history of orphan numbers outliving the builds that made them.
pub fn set_meta(f: impl FnOnce(&mut RunHeader)) {
    if !enabled() {
        return;
    }
    if let Ok(mut m) = meta().write() {
        f(&mut m);
    }
}

fn unreachable_list() -> &'static Mutex<Vec<Unreachable>> {
    static U: OnceLock<Mutex<Vec<Unreachable>>> = OnceLock::new();
    U.get_or_init(|| Mutex::new(Vec::new()))
}

fn overhead_cell() -> &'static RwLock<Overhead> {
    static O: OnceLock<RwLock<Overhead>> = OnceLock::new();
    O.get_or_init(|| RwLock::new(Overhead::default()))
}

thread_local! {
    /// Open span ids, innermost last. Used only to find a new span's parent.
    /// A span's identity is fixed at *open*, so a span that opens on one thread
    /// and closes on another still accumulates into the right node — it only
    /// costs a `misnested` tick, which the report shows.
    static STACK: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------------------
// Spans
// ---------------------------------------------------------------------------

/// An open span. Closes on drop; nesting comes from drop order.
///
/// `#[must_use]` because `arc_profiler::span("x");` — with no binding — drops
/// immediately and measures nothing. That mistake produces a node with a
/// plausible-looking call count and ~0 time, which is precisely the kind of
/// confidently-wrong output this crate exists to avoid.
#[must_use = "a span must be bound to a variable; an unbound span closes immediately and measures nothing"]
pub struct Span {
    node: u32,
    /// Uninitialised when `node == INACTIVE`, and read nowhere else.
    ///
    /// `MaybeUninit` rather than `Option<Instant>` or a lazily-cached epoch
    /// because the disabled path must cost nothing at all: no clock read, no
    /// atomic, no branch beyond the gate itself. `Instant` has no destructor,
    /// so leaving it uninitialised leaks nothing.
    start: std::mem::MaybeUninit<Instant>,
    kind: NodeKind,
    dev_start: Option<EventHandle>,
}

impl Span {
    /// A span that records nothing. Returned whenever the profiler is off, the
    /// depth limit is hit, or a lock is poisoned.
    #[inline(always)]
    const fn inactive() -> Self {
        Self {
            node: INACTIVE,
            start: std::mem::MaybeUninit::uninit(),
            kind: NodeKind::Host,
            dev_start: None,
        }
    }

    /// Is this span recording? Useful in tests and for conditional detail.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.node != INACTIVE
    }
}

#[inline]
fn open(name: &str, kind: NodeKind) -> Span {
    if !enabled() {
        return Span::inactive();
    }
    open_slow(name, kind)
}

fn open_slow(name: &str, kind: NodeKind) -> Span {
    let parent = STACK.with(|s| s.borrow().last().copied().unwrap_or(NO_PARENT));
    let depth = STACK.with(|s| s.borrow().len()) as u32;
    if depth >= MAX_DEPTH.load(Relaxed) {
        SPANS_DROPPED_DEPTH.fetch_add(1, Relaxed);
        return Span::inactive();
    }

    // Fast path: the node already exists, so only a read lock is needed. After
    // the first step every span takes this branch.
    let existing = registry().read().ok().and_then(|r| r.lookup(parent, name));
    let node = match existing {
        Some(id) => id,
        None => match registry().write() {
            Ok(mut r) => r.get_or_create(parent, name, kind),
            Err(_) => return Span::inactive(),
        },
    };

    STACK.with(|s| s.borrow_mut().push(node));
    SPANS_OPENED.fetch_add(1, Relaxed);

    // Recorded last, so the device interval starts as close to the real work as
    // the instrumentation can put it.
    let dev_start = if kind == NodeKind::Device {
        let h = with_timer(|t| t.record());
        if h.is_none() {
            // No backend, or the pool refused. This span will carry no device
            // time, and the report must say "unmeasured" rather than let the
            // resulting 0 ns read as "the GPU did nothing here".
            device::note_unresolved();
        }
        h
    } else {
        None
    };
    Span {
        node,
        start: std::mem::MaybeUninit::new(Instant::now()),
        kind,
        dev_start,
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if self.node == INACTIVE {
            return;
        }
        // Wall first: the profiler's own event-record cost must not be charged
        // to the code under measurement.
        //
        // SAFETY: `start` is initialised on every path that sets `node` to
        // anything other than `INACTIVE`, and the branch above has already
        // returned for `INACTIVE`.
        let wall = unsafe { self.start.assume_init() }.elapsed().as_nanos() as u64;
        let dev_pair = match self.dev_start.take() {
            Some(s) => match with_timer(|t| t.record()) {
                Some(e) => Some((s, e)),
                None => {
                    device::note_unresolved();
                    with_timer(|t| t.release(&[s]));
                    None
                }
            },
            None => None,
        };

        let sync = if self.kind == NodeKind::Sync { wall } else { 0 };
        if let Ok(r) = registry().read() {
            tree::accumulate(&r.nodes[self.node as usize], wall, sync);
        }
        if let Some(pair) = dev_pair {
            if let Ok(mut p) = pending().lock() {
                p.push((self.node, pair.0, pair.1));
            }
        }

        STACK.with(|s| {
            let mut s = s.borrow_mut();
            if s.last() == Some(&self.node) {
                s.pop();
            } else {
                // Out-of-order or cross-thread close. The accumulation above is
                // still correct (the node id was bound at open); only the
                // parent of *subsequent* spans on this thread would be wrong,
                // so repair the stack and count it.
                MISNESTED.fetch_add(1, Relaxed);
                if let Some(pos) = s.iter().rposition(|x| *x == self.node) {
                    s.truncate(pos);
                }
            }
        });
    }
}

/// Host work. `device_ns` for this node is the roll-up of its children.
#[inline(always)]
pub fn span(name: &'static str) -> Span {
    open(name, NodeKind::Host)
}

/// GPU work. The wall clock of this span is the **launch** cost; its GPU time
/// comes from CUDA events and lands in `device_ns`.
#[inline(always)]
pub fn device_span(name: &'static str) -> Span {
    open(name, NodeKind::Device)
}

/// The host blocking on the device — an explicit `synchronize`, a D2H copy, a
/// `to_vec`. Its wall clock is reported as `sync_ns` so waiting can never be
/// mistaken for computing.
#[inline(always)]
pub fn sync_span(name: &'static str) -> Span {
    open(name, NodeKind::Sync)
}

/// Per-index span, e.g. one node per transformer layer.
///
/// Collapses to plain [`span`] unless `ARC_PROFILE_UNROLL=1`, because 43
/// layers x 10 sub-ops is 430 nodes of mostly-identical information; the
/// aggregate node's `min`/`max` usually answers the same question.
#[inline]
pub fn span_idx(name: &'static str, idx: usize) -> Span {
    if !enabled() {
        return Span::inactive();
    }
    if UNROLL.load(Relaxed) {
        open_slow(&format!("{name}[{idx}]"), NodeKind::Host)
    } else {
        open_slow(name, NodeKind::Host)
    }
}

/// Declare that a code path exists here but is **not taken** in this
/// configuration.
///
/// Creates a striped, zero-time node under the current span and records the
/// reason and the `file.rs:LINE` of the condition that bails. Zero-because-
/// unreached and zero-because-fast must not look the same in a report, and the
/// only way to guarantee that is to say so at the branch itself.
pub fn mark_unreachable(name: &'static str, reason: &str, site: &'static str) {
    if !enabled() {
        return;
    }
    let parent = STACK.with(|s| s.borrow().last().copied().unwrap_or(NO_PARENT));
    // Fast path. Call sites for this live inside per-layer dispatch, so it runs
    // tens of times per step; taking the registry's *write* lock every time
    // would make the profiler's own bookkeeping visible in its own numbers.
    if let Ok(r) = registry().read() {
        if let Some(id) = r.lookup(parent, name) {
            if !r.nodes[id as usize].reachable.load(Relaxed) {
                return;
            }
        }
    }
    let path = match registry().write() {
        Ok(mut r) => {
            let id = r.get_or_create(parent, name, NodeKind::Host);
            r.nodes[id as usize].reachable.store(false, Relaxed);
            if let Ok(mut n) = r.nodes[id as usize].note.write() {
                *n = Some(format!("unreachable: {reason} ({site})"));
            }
            r.nodes[id as usize].path.clone()
        }
        Err(_) => name.to_string(),
    };
    if let Ok(mut u) = unreachable_list().lock() {
        if !u.iter().any(|x| x.path == path) {
            u.push(Unreachable {
                path,
                reason: reason.to_string(),
                site: Some(site.to_string()),
            });
        }
    }
}

/// Record a caveat on the run header — typically a region the tree deliberately
/// does **not** decompose, and why.
///
/// Deduplicated, so it is safe to call from a hot path. A known gap that is
/// written down is a limit; the same gap unwritten is a lie by omission, and
/// this report is going to be used to decide where engineering time goes.
pub fn note(msg: &str) {
    if !enabled() {
        return;
    }
    // Read lock first: `note` is called from inside the decode loop, and a
    // write lock per step is exactly the kind of self-inflicted cost that makes
    // a profiler untrustworthy.
    if let Ok(m) = meta().read() {
        if m.notes.iter().any(|n| n == msg) {
            return;
        }
    }
    if let Ok(mut m) = meta().write() {
        if !m.notes.iter().any(|n| n == msg) {
            m.notes.push(msg.to_string());
        }
    }
}

/// Stamp batch geometry onto the innermost open span, and add its tokens to the
/// run total.
pub fn set_geometry(b: usize, t: usize) {
    if !enabled() {
        return;
    }
    let Some(node) = STACK.with(|s| s.borrow().last().copied()) else {
        return;
    };
    let tokens = (b as u64) * (t as u64);
    if let Ok(r) = registry().read() {
        let n = &r.nodes[node as usize];
        n.geom_b.store(b as u32, Relaxed);
        n.geom_t.store(t as u32, Relaxed);
        n.tokens.fetch_add(tokens, Relaxed);
    }
    TOKENS.fetch_add(tokens, Relaxed);
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

/// One engine step. Opens the root span, and on drop resolves this step's CUDA
/// events, handles warmup, and triggers the auto-write if one is configured.
#[must_use = "the step guard must be bound; dropping it immediately ends the step"]
pub struct StepGuard {
    span: Option<Span>,
    active: bool,
}

/// Begin a step. `name` is typically `"decode_step"` or `"prompt_step"`.
pub fn step_scope(name: &'static str) -> StepGuard {
    if !enabled() {
        return StepGuard {
            span: None,
            active: false,
        };
    }
    StepGuard {
        span: Some(open_slow(name, NodeKind::Host)),
        active: true,
    }
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // Draining the stream happens *inside* the root span, as a sync child,
        // so the time between the last launch and GPU completion is attributed
        // rather than falling outside the tree. The engine's own logits D2H
        // already forces a sync each step, so on the real path this is close to
        // free — but where it is not free, that is a finding, not noise.
        {
            let _drain = sync_span("gpu_drain");
            with_timer(|t| t.flush());
        }
        self.span.take();

        resolve_pending();

        let seen = STEPS_SEEN.fetch_add(1, Relaxed) + 1;
        let warmup = WARMUP.load(Relaxed);
        if seen == warmup {
            // Discard warmup entirely: first steps carry allocator growth,
            // autotune, and lazy kernel loads that no steady-state reader wants
            // averaged in.
            if let Ok(r) = registry().read() {
                r.reset_counters();
            }
            SPANS_OPENED.store(0, Relaxed);
            TOKENS.store(0, Relaxed);
            MISNESTED.store(0, Relaxed);
            device::UNRESOLVED.store(0, Relaxed);
            return;
        }
        if seen <= warmup {
            return;
        }
        let recorded = STEPS_RECORDED.fetch_add(1, Relaxed) + 1;

        if let Some(target) = std::env::var("ARC_PROFILE_STEPS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            if recorded >= target && !WROTE_ONCE.swap(true, Relaxed) {
                match write_outputs() {
                    Ok((j, h)) => tracing::info!(
                        "arc-profiler wrote {} recorded steps to {} and {}",
                        recorded,
                        j.display(),
                        h.display()
                    ),
                    Err(e) => tracing::error!("arc-profiler could not write its report: {e}"),
                }
            }
        }
    }
}

/// Turn this step's queued event pairs into device nanoseconds.
///
/// Called after the stream has been flushed, so every event has completed and
/// `cudaEventElapsedTime` cannot block.
fn resolve_pending() {
    let Ok(mut p) = pending().lock() else { return };
    if p.is_empty() {
        return;
    }
    let mut release: Vec<EventHandle> = Vec::with_capacity(p.len() * 2);
    if let Ok(r) = registry().read() {
        for (node, a, b) in p.drain(..) {
            match with_timer(|t| t.resolve(a, b)) {
                Some(ns) => {
                    r.nodes[node as usize]
                        .device_measured_ns
                        .fetch_add(ns, Relaxed);
                }
                None => device::note_unresolved(),
            }
            release.push(a);
            release.push(b);
        }
    } else {
        p.clear();
    }
    with_timer(|t| t.release(&release));
}

// ---------------------------------------------------------------------------
// Self-calibration: measure the profiler, both states
// ---------------------------------------------------------------------------

/// Measure the cost of the instrumentation itself, on and off, through the real
/// public API.
///
/// Not an estimate and not a code-reading argument: it flips the same gate the
/// hot path reads and times the same [`span`] call, so "off" really is the cost
/// a shipped binary pays at every call site.
fn calibrate() {
    const N: u32 = 20_000;
    // OFF: what every shipped binary pays per instrumented site.
    STATE.store(S_OFF, Relaxed);
    let t0 = Instant::now();
    for _ in 0..N {
        let s = span("calibration.off");
        std::hint::black_box(&s);
    }
    let off_ns = t0.elapsed().as_nanos() as f64 / f64::from(N);

    // ON: open + close, including registry lookup and atomic accumulation.
    STATE.store(S_ON, Relaxed);
    let t1 = Instant::now();
    for _ in 0..N {
        let s = span("calibration.on");
        std::hint::black_box(&s);
    }
    let on_ns = t1.elapsed().as_nanos() as f64 / f64::from(N);

    if let Ok(mut o) = overhead_cell().write() {
        o.disabled_ns_per_span = off_ns;
        o.enabled_ns_per_span = on_ns;
    }
    // Wipe the scratch nodes and the calibration's own span count.
    if let Ok(mut r) = registry().write() {
        *r = Registry::default();
    }
    SPANS_OPENED.store(0, Relaxed);
}

/// Proof that device time is not launch time — run this on the GPU.
///
/// Issues a batch of real GPU work inside one [`device_span`] **without
/// synchronising**, then compares the span's wall clock (the launches) against
/// the CUDA-event interval (the execution). A profiler that timed launches
/// would report these as equal. A correct one reports the device interval as
/// very much larger, and this function returns both so the claim is a
/// measurement rather than an assurance.
///
/// Returns `None` when no CUDA timer is attached, in which case the report says
/// the proof was *not run* rather than implying it passed.
pub fn device_selftest(dev: &candle_core::Device) -> Option<SelfTest> {
    if !enabled() || with_timer(|t| t.backend()) == "null" {
        return None;
    }
    use candle_core::{DType, Tensor};
    let n = 2048;
    let a = Tensor::zeros((n, n), DType::F32, dev).ok()?;
    let b = Tensor::zeros((n, n), DType::F32, dev).ok()?;
    // Warm the kernel so the measured window is execution, not module load.
    let _ = a.matmul(&b).ok()?;
    dev.synchronize().ok()?;

    let node_before = SPANS_OPENED.load(Relaxed);
    let launch_wall_ns;
    {
        let s = device_span("selftest.gpu_matmul");
        let t = Instant::now();
        for _ in 0..32 {
            let _ = a.matmul(&b).ok()?;
        }
        launch_wall_ns = t.elapsed().as_nanos() as u64;
        drop(s);
    }
    let _ = node_before;
    with_timer(|t| t.flush());
    resolve_pending();

    let device_ns = registry().read().ok().and_then(|r| {
        r.nodes
            .iter()
            .find(|n| n.name == "selftest.gpu_matmul")
            .map(|n| n.device_measured_ns.load(Relaxed))
    })?;

    Some(SelfTest {
        launch_wall_ns,
        device_ns,
        ratio: device_ns as f64 / launch_wall_ns.max(1) as f64,
    })
}

/// Run [`device_selftest`] when `ARC_PROFILE_SELFTEST=1`, log the verdict, and
/// carry it into the report's notes.
///
/// Called once from pipeline construction. It exists so the proof is a normal
/// part of a profiled run rather than something a future session has to
/// remember to do by hand — and so a run whose device column is untrustworthy
/// says so in its own artifact.
pub fn maybe_selftest(dev: &candle_core::Device) {
    if !enabled() || std::env::var("ARC_PROFILE_SELFTEST").ok().as_deref() != Some("1") {
        return;
    }
    match device_selftest(dev) {
        Some(st) => {
            let verdict = if st.ratio > 10.0 {
                "PASS: CUDA events are measuring execution, not launches"
            } else {
                "FAIL: device time is indistinguishable from launch time — treat every \
                 device_ns in this report as void"
            };
            let msg = format!(
                "device selftest: launch_wall={} ns, device={} ns, ratio={:.1}x — {}",
                st.launch_wall_ns, st.device_ns, st.ratio, verdict
            );
            if st.ratio > 10.0 {
                tracing::info!("{msg}");
            } else {
                tracing::error!("{msg}");
            }
            note(&msg);
            reset();
        }
        None => {
            let msg = "device selftest: NOT RUN (no CUDA event timer attached) — device_ns \
                       columns in this report are unmeasured, not zero";
            tracing::warn!("{msg}");
            note(msg);
        }
    }
}

/// Result of [`device_selftest`].
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SelfTest {
    /// Host time spent issuing the launches.
    pub launch_wall_ns: u64,
    /// GPU time the same window actually took.
    pub device_ns: u64,
    /// `device_ns / launch_wall_ns`. Near 1.0 means the profiler is timing
    /// launches and every device number in the report is wrong.
    pub ratio: f64,
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Tolerance for the parent/child reconciliation check, in percent.
///
/// Not zero: `Instant` granularity plus the guard's own open/close cost mean
/// children can exceed a parent by a few nanoseconds per call without anything
/// being wrong. Anything beyond this is a real accounting error and is listed
/// in the report.
pub const RECONCILE_TOLERANCE_PCT: f64 = 2.0;

/// Freeze the current state into a [`Profile`].
pub fn snapshot() -> Profile {
    let nodes = registry().read().map(|r| r.snapshot()).unwrap_or_default();

    let steps = STEPS_RECORDED.load(Relaxed);
    let tokens = TOKENS.load(Relaxed);

    let roots: Vec<&Node> = nodes.iter().filter(|n| n.parent.is_none()).collect();
    let totals = Totals {
        wall_ns: roots.iter().map(|n| n.wall_ns).sum(),
        device_ns: roots.iter().map(|n| n.device_ns).sum(),
        sync_ns: nodes.iter().map(|n| n.sync_ns).sum(),
        busy_host_ns: nodes.iter().map(|n| n.busy_self_ns).sum(),
        steps,
        tokens,
    };

    let spans = SPANS_OPENED.load(Relaxed);
    let mut overhead = overhead_cell()
        .read()
        .map(|o| o.clone())
        .unwrap_or_default();
    overhead.spans_per_step = if steps > 0 {
        spans as f64 / steps as f64
    } else {
        0.0
    };
    let mean_step_ns = if steps > 0 {
        totals.wall_ns as f64 / steps as f64
    } else {
        0.0
    };
    overhead.enabled_overhead_pct = if mean_step_ns > 0.0 {
        100.0 * overhead.enabled_ns_per_span * overhead.spans_per_step / mean_step_ns
    } else {
        0.0
    };

    let mut run = meta().read().map(|m| m.clone()).unwrap_or_default();
    run.steps = steps;
    run.tokens = tokens;
    run.profile_depth = MAX_DEPTH.load(Relaxed);
    run.warmup_steps = WARMUP.load(Relaxed) as u32;
    run.unroll_layers = UNROLL.load(Relaxed);
    if run.label.is_empty() {
        run.label = std::env::var("ARC_PROFILE_LABEL").unwrap_or_else(|_| "run".to_string());
    }
    if run.started_unix_ms == 0 {
        run.started_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }
    if run.commit.is_empty() {
        run.commit = option_env!("ARC_COMMIT").unwrap_or("unknown").to_string();
    }
    run.notes.push(format!(
        "device timer backend: {}; device_ns is stream-ELAPSED time (includes launch gaps), \
         not kernel-busy time",
        with_timer(|t| t.backend())
    ));
    if SPANS_DROPPED_DEPTH.load(Relaxed) > 0 {
        run.notes.push(format!(
            "{} spans were dropped for exceeding ARC_PROFILE_DEPTH={} — this tree is truncated",
            SPANS_DROPPED_DEPTH.load(Relaxed),
            MAX_DEPTH.load(Relaxed)
        ));
    }

    let mut profile = Profile {
        schema: SCHEMA.to_string(),
        run,
        totals,
        overhead,
        nodes,
        unreachable: unreachable_list()
            .lock()
            .map(|u| u.clone())
            .unwrap_or_default(),
        reconciliation: Reconciliation {
            tolerance_pct: RECONCILE_TOLERANCE_PCT,
            violations: Vec::new(),
            misnested_spans: MISNESTED.load(Relaxed),
            unresolved_device_spans: device::UNRESOLVED.load(Relaxed),
        },
    };
    profile.reconciliation.violations = profile.recheck(RECONCILE_TOLERANCE_PCT);
    profile
}

/// Write `<out>/<label>.json` and `<out>/<label>.html`.
pub fn write_outputs() -> std::io::Result<(PathBuf, PathBuf)> {
    let profile = snapshot();
    let dir = PathBuf::from(
        std::env::var("ARC_PROFILE_OUT").unwrap_or_else(|_| "./arc-profile".to_string()),
    );
    std::fs::create_dir_all(&dir)?;
    let stem = if profile.run.label.is_empty() {
        "run".to_string()
    } else {
        profile
            .run
            .label
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect()
    };
    let json_path = dir.join(format!("{stem}.json"));
    let html_path = dir.join(format!("{stem}.html"));
    std::fs::write(&json_path, serde_json::to_vec_pretty(&profile)?)?;
    std::fs::write(&html_path, html::render(std::slice::from_ref(&profile)))?;
    Ok((json_path, html_path))
}

/// Wipe every counter and node. Exposed for tests and for tools that profile
/// several configurations in one process.
pub fn reset() {
    if let Ok(mut r) = registry().write() {
        *r = Registry::default();
    }
    if let Ok(mut p) = pending().lock() {
        p.clear();
    }
    if let Ok(mut u) = unreachable_list().lock() {
        u.clear();
    }
    SPANS_OPENED.store(0, Relaxed);
    SPANS_DROPPED_DEPTH.store(0, Relaxed);
    STEPS_SEEN.store(0, Relaxed);
    STEPS_RECORDED.store(0, Relaxed);
    TOKENS.store(0, Relaxed);
    MISNESTED.store(0, Relaxed);
    device::UNRESOLVED.store(0, Relaxed);
    STACK.with(|s| s.borrow_mut().clear());
}

// ---------------------------------------------------------------------------
// Test hooks
// ---------------------------------------------------------------------------

/// Force the gate. **Tests and calibration only** — flipping this in a live
/// server mid-run would produce a tree with holes in it.
#[doc(hidden)]
pub fn __force_enabled(on: bool) {
    // Resolve the environment-driven defaults first, otherwise the first real
    // `enabled()` call would run `init_gate` and overwrite this.
    let _ = enabled();
    STATE.store(if on { S_ON } else { S_OFF }, Relaxed);
}

/// Override the depth limit. Tests only.
#[doc(hidden)]
pub fn __set_depth(d: u32) {
    MAX_DEPTH.store(d, Relaxed);
}

/// Override the warmup count. Tests only.
#[doc(hidden)]
pub fn __set_warmup(w: u64) {
    WARMUP.store(w, Relaxed);
}
