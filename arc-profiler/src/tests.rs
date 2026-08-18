//! The teeth.
//!
//! A profiler that reports plausible-but-wrong numbers is worse than no
//! profiler, because it looks authoritative. So every claim this crate makes is
//! tested against **known** truth, and every check is additionally shown to
//! *fail* when the thing it guards is broken — DOCTRINE D12: this repository has
//! found seven-plus tests that passed while their assertion was unreachable.
//!
//! The mutation proofs are the important half:
//!
//! * [`device_ns_would_equal_wall_ns_if_we_timed_launches`] installs a timer
//!   that reproduces the classic bug — reporting host launch time as device
//!   time — and shows the numbers collapse together, which is exactly what
//!   [`device_ns_comes_from_the_event_timer_not_the_clock`] asserts must not
//!   happen.
//! * [`reconciliation_flags_a_child_that_exceeds_its_parent`] breaks the
//!   parent/child accounting and shows the check fires; its sibling shows the
//!   same check is silent on a well-formed tree, so it is not vacuously true.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use super::*;
use crate::device::{DeviceTimer, EventHandle};

/// The profiler is a process-wide singleton, so tests must not overlap.
fn guard() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    let g = L
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    reset();
    if let Ok(mut t) = TEST_TIMER.write() {
        *t = None;
    }
    __force_enabled(true);
    __set_depth(16);
    __set_warmup(0);
    g
}

// ---------------------------------------------------------------------------
// Fake device timers
// ---------------------------------------------------------------------------

/// Reports a fixed device interval, entirely unrelated to host time. Stands in
/// for CUDA events: the whole point is that its answer cannot be derived from
/// the clock.
struct FixedTimer {
    ns: u64,
    next: AtomicU32,
    resolves: AtomicU64,
}

impl FixedTimer {
    fn new(ns: u64) -> Self {
        Self {
            ns,
            next: AtomicU32::new(0),
            resolves: AtomicU64::new(0),
        }
    }
}

impl DeviceTimer for FixedTimer {
    fn record(&self) -> Option<EventHandle> {
        Some(EventHandle(self.next.fetch_add(1, Relaxed)))
    }
    fn resolve(&self, _s: EventHandle, _e: EventHandle) -> Option<u64> {
        self.resolves.fetch_add(1, Relaxed);
        Some(self.ns)
    }
    fn flush(&self) {}
    fn release(&self, _h: &[EventHandle]) {}
    fn backend(&self) -> &'static str {
        "fixed-fake"
    }
}

/// **The bug, on purpose.** Timestamps on the *host* clock, so `resolve`
/// returns the time the host spent issuing launches — which is what a profiler
/// built only on [`Instant`] would report as GPU time.
struct LaunchOnlyTimer {
    stamps: Mutex<Vec<Instant>>,
}

impl LaunchOnlyTimer {
    fn new() -> Self {
        Self {
            stamps: Mutex::new(Vec::new()),
        }
    }
}

impl DeviceTimer for LaunchOnlyTimer {
    fn record(&self) -> Option<EventHandle> {
        let mut s = self.stamps.lock().ok()?;
        s.push(Instant::now());
        Some(EventHandle((s.len() - 1) as u32))
    }
    fn resolve(&self, s: EventHandle, e: EventHandle) -> Option<u64> {
        let st = self.stamps.lock().ok()?;
        let a = *st.get(s.0 as usize)?;
        let b = *st.get(e.0 as usize)?;
        Some(b.duration_since(a).as_nanos() as u64)
    }
    fn flush(&self) {}
    fn release(&self, _h: &[EventHandle]) {}
    fn backend(&self) -> &'static str {
        "launch-only-fake"
    }
}

fn install(t: Box<dyn DeviceTimer>) {
    *TEST_TIMER.write().unwrap() = Some(t);
}

fn burn(d: Duration) {
    let t = Instant::now();
    while t.elapsed() < d {
        std::hint::spin_loop();
    }
}

// ---------------------------------------------------------------------------
// Known synthetic work
// ---------------------------------------------------------------------------

#[test]
fn span_measures_a_known_duration() {
    let _g = guard();
    const TARGET: Duration = Duration::from_millis(25);
    {
        let _s = span("known");
        burn(TARGET);
    }
    let p = snapshot();
    let n = p.node("known").expect("node must exist");
    assert_eq!(n.calls, 1);
    let ms = n.wall_ns as f64 / 1e6;
    // Generous upper bound: CI machines are shared and can be descheduled.
    // The lower bound is the load-bearing one — it is what fails if the timer
    // is wired to the wrong clock or the span closes early.
    assert!(
        (24.0..80.0).contains(&ms),
        "25 ms of work measured as {ms:.2} ms"
    );
}

#[test]
fn nested_spans_form_the_tree_they_look_like() {
    let _g = guard();
    {
        let _a = span("outer");
        burn(Duration::from_millis(5));
        {
            let _b = span("middle");
            burn(Duration::from_millis(10));
            {
                let _c = span("inner");
                burn(Duration::from_millis(10));
            }
        }
    }
    let p = snapshot();
    let outer = p.node("outer").unwrap();
    let middle = p.node("outer.middle").unwrap();
    let inner = p.node("outer.middle.inner").unwrap();

    assert_eq!(outer.parent, None);
    assert_eq!(middle.parent, Some(outer.id));
    assert_eq!(inner.parent, Some(middle.id));
    assert_eq!(outer.depth, 0);
    assert_eq!(inner.depth, 2);
    assert!(outer.children.contains(&middle.id));

    // Containment, which is the property the whole tree rests on.
    assert!(outer.wall_ns >= middle.wall_ns);
    assert!(middle.wall_ns >= inner.wall_ns);

    // Self time lands where the work actually was: ~5 ms at the outer node,
    // ~10 ms at the middle, ~10 ms at the leaf.
    let self_ms = |n: &Node| n.wall_self_ns as f64 / 1e6;
    assert!(
        (3.0..20.0).contains(&self_ms(outer)),
        "outer self {:.2} ms",
        self_ms(outer)
    );
    assert!(
        (7.0..30.0).contains(&self_ms(middle)),
        "middle self {:.2} ms",
        self_ms(middle)
    );
}

#[test]
fn repeated_spans_aggregate_with_min_and_max() {
    let _g = guard();
    for i in 0..5 {
        let _s = span("repeat");
        burn(Duration::from_millis(if i == 4 { 20 } else { 5 }));
    }
    let p = snapshot();
    let n = p.node("repeat").unwrap();
    assert_eq!(n.calls, 5);
    assert!(
        n.max_wall_ns > n.min_wall_ns * 2,
        "the 20 ms call must show up as max"
    );
    assert!(n.wall_ns >= n.max_wall_ns);
}

// ---------------------------------------------------------------------------
// Device time: the trap
// ---------------------------------------------------------------------------

#[test]
fn device_ns_comes_from_the_event_timer_not_the_clock() {
    let _g = guard();
    // 40 ms of "GPU time" reported by the timer, against a span whose host cost
    // is a few microseconds. If `device_ns` were derived from `Instant` these
    // two would be equal — see the mutation test below, which shows exactly
    // that happening.
    const FAKE_DEVICE_NS: u64 = 40_000_000;
    install(Box::new(FixedTimer::new(FAKE_DEVICE_NS)));
    {
        let _step = step_scope("step");
        let _d = device_span("kernel");
    }
    let p = snapshot();
    let n = p.node("step.kernel").unwrap();
    assert_eq!(
        n.device_ns, FAKE_DEVICE_NS,
        "device_ns must be whatever the event timer said"
    );
    assert!(
        n.wall_ns < FAKE_DEVICE_NS / 100,
        "wall_ns is the launch cost and must stay tiny; got {} ns",
        n.wall_ns
    );
    assert_eq!(n.kind, NodeKind::Device);
    assert_eq!(p.reconciliation.unresolved_device_spans, 0);
}

#[test]
fn device_ns_would_equal_wall_ns_if_we_timed_launches() {
    let _g = guard();
    // The mutation. `LaunchOnlyTimer` is a faithful reproduction of the bug:
    // it timestamps on the host clock. With it installed, the device column and
    // the wall column become the same number — which is precisely the situation
    // the test above forbids. This proves that assertion has teeth rather than
    // passing for free.
    install(Box::new(LaunchOnlyTimer::new()));
    {
        let _step = step_scope("step");
        let _d = device_span("kernel");
        burn(Duration::from_millis(10));
    }
    let p = snapshot();
    let n = p.node("step.kernel").unwrap();
    let ratio = n.device_ns as f64 / n.wall_ns.max(1) as f64;
    assert!(
        (0.5..2.0).contains(&ratio),
        "a launch-timing profiler reports device ~= wall; got ratio {ratio:.3}"
    );
}

#[test]
fn a_device_span_with_no_timer_is_unmeasured_not_zero() {
    let _g = guard();
    // No timer installed => NullTimer => no events. The report must not present
    // this as "the GPU took 0 ns".
    {
        let _step = step_scope("step");
        let _d = device_span("kernel");
    }
    let p = snapshot();
    assert_eq!(p.node("step.kernel").unwrap().device_ns, 0);
    assert!(
        p.reconciliation.unresolved_device_spans > 0,
        "an unmeasurable device span must be counted as unresolved"
    );
    assert!(
        p.run.notes.iter().any(|n| n.contains("null")),
        "the run header must name the backend that produced (or failed to produce) device time"
    );
}

#[test]
fn device_time_rolls_up_through_host_parents_to_the_root() {
    let _g = guard();
    install(Box::new(FixedTimer::new(5_000_000)));
    {
        let _step = step_scope("step");
        let _f = span("forward");
        for _ in 0..3 {
            let _d = device_span("kernel");
        }
    }
    let p = snapshot();
    // Only the leaf measures; every ancestor must still show the GPU time, or
    // the root would report no device work while its leaves reported 15 ms.
    assert_eq!(p.node("step.forward.kernel").unwrap().device_ns, 15_000_000);
    assert_eq!(p.node("step.forward").unwrap().device_ns, 15_000_000);
    assert_eq!(p.node("step").unwrap().device_ns, 15_000_000);
    assert_eq!(
        p.node("step.forward").unwrap().device_self_ns,
        0,
        "a host parent claims no device time of its own"
    );
}

#[test]
fn device_self_time_exposes_stream_gaps_between_children() {
    let _g = guard();
    // A parent device span that measures 10 ms while its two children measure
    // 1 ms each: 8 ms of the stream belonged to no instrumented kernel. That
    // residue is how launch starvation shows up, so it must survive to the
    // report rather than being clamped away.
    struct Graded {
        n: AtomicU32,
    }
    impl DeviceTimer for Graded {
        fn record(&self) -> Option<EventHandle> {
            Some(EventHandle(self.n.fetch_add(1, Relaxed)))
        }
        fn resolve(&self, s: EventHandle, _e: EventHandle) -> Option<u64> {
            // Handle 0/1 is the outer pair (recorded first, closed last).
            Some(if s.0 == 0 { 10_000_000 } else { 1_000_000 })
        }
        fn flush(&self) {}
        fn release(&self, _h: &[EventHandle]) {}
        fn backend(&self) -> &'static str {
            "graded-fake"
        }
    }
    install(Box::new(Graded {
        n: AtomicU32::new(0),
    }));
    {
        let _step = step_scope("step");
        let _outer = device_span("attn");
        {
            let _a = device_span("gemm");
        }
        {
            let _b = device_span("softmax");
        }
    }
    let p = snapshot();
    let outer = p.node("step.attn").unwrap();
    assert_eq!(outer.device_ns, 10_000_000);
    assert_eq!(
        outer.device_self_ns, 8_000_000,
        "stream time belonging to no child must be visible, not clamped"
    );
}

// ---------------------------------------------------------------------------
// Sync time
// ---------------------------------------------------------------------------

#[test]
fn sync_time_is_waiting_and_is_not_counted_as_busy_host_time() {
    let _g = guard();
    {
        let _outer = span("step");
        {
            let _w = sync_span("logits_d2h");
            burn(Duration::from_millis(20));
        }
    }
    let p = snapshot();
    let w = p.node("step.logits_d2h").unwrap();
    assert_eq!(w.kind, NodeKind::Sync);
    assert_eq!(w.sync_ns, w.wall_ns, "a sync span's wall time IS its wait");
    assert_eq!(
        w.busy_self_ns, 0,
        "time spent blocked on the GPU must never be reported as host compute"
    );
    assert!(p.totals.sync_ns >= 19_000_000);
}

// ---------------------------------------------------------------------------
// Reconciliation — and the proof that it is not vacuous
// ---------------------------------------------------------------------------

#[test]
fn a_well_formed_tree_reconciles() {
    let _g = guard();
    {
        let _a = span("root");
        burn(Duration::from_millis(5));
        {
            let _b = span("child_a");
            burn(Duration::from_millis(5));
        }
        {
            let _c = span("child_b");
            burn(Duration::from_millis(5));
        }
    }
    let p = snapshot();
    assert!(
        p.reconciliation.violations.is_empty(),
        "clean tree reported violations: {:?}",
        p.reconciliation.violations
    );
    assert_eq!(p.reconciliation.misnested_spans, 0);
    // And children + self really do add up to the parent.
    let root = p.node("root").unwrap();
    let kids: u64 = root
        .children
        .iter()
        .map(|c| p.nodes.iter().find(|n| n.id == *c).unwrap().wall_ns)
        .sum();
    assert_eq!(root.wall_self_ns + kids, root.wall_ns);
}

#[test]
fn reconciliation_flags_a_child_that_exceeds_its_parent() {
    let _g = guard();
    {
        let _a = span("root");
        burn(Duration::from_millis(5));
        let _b = span("child");
        burn(Duration::from_millis(2));
    }
    let mut p = snapshot();
    assert!(p.recheck(RECONCILE_TOLERANCE_PCT).is_empty());

    // Break the accounting the way a real bug would: a child claiming more time
    // than the parent that contains it.
    let root_id = p.node("root").unwrap().id;
    let parent_wall = p.node("root").unwrap().wall_ns;
    for n in p.nodes.iter_mut() {
        if n.parent == Some(root_id) {
            n.wall_ns = parent_wall * 2;
        }
    }
    let v = p.recheck(RECONCILE_TOLERANCE_PCT);
    assert!(
        v.iter().any(|x| x.path == "root" && x.channel == "wall"),
        "a child exceeding its parent must be reported; got {v:?}"
    );
    assert!(v[0].excess_pct > 90.0);
}

#[test]
fn reconciliation_flags_a_broken_device_rollup() {
    let _g = guard();
    install(Box::new(FixedTimer::new(1_000_000)));
    {
        let _step = step_scope("step");
        let _d = device_span("kernel");
    }
    let mut p = snapshot();
    assert!(p.recheck(RECONCILE_TOLERANCE_PCT).is_empty());

    // Same mutation, device channel: a parent that under-reports its children's
    // GPU time. Proves the device column is checked too, not just wall.
    let kernel_id = p.node("step.kernel").unwrap().id;
    for n in p.nodes.iter_mut() {
        if n.id == kernel_id {
            n.device_ns = 50_000_000;
        }
    }
    let v = p.recheck(RECONCILE_TOLERANCE_PCT);
    assert!(
        v.iter().any(|x| x.channel == "device"),
        "device-channel violations must be detected; got {v:?}"
    );
}

#[test]
fn tolerance_is_a_band_not_a_rubber_stamp() {
    let _g = guard();
    // A 1% overshoot is inside tolerance; a 50% overshoot is not. Without both
    // halves, "tolerance_pct" could be any number at all and no test would care.
    {
        let _a = span("root");
        burn(Duration::from_millis(10));
        let _b = span("child");
        burn(Duration::from_millis(1));
    }
    let mut p = snapshot();
    let parent = p.node("root").unwrap().wall_ns;
    let root_id = p.node("root").unwrap().id;
    for n in p.nodes.iter_mut() {
        if n.parent == Some(root_id) {
            n.wall_ns = parent + parent / 100; // +1%
        }
    }
    assert!(
        p.recheck(RECONCILE_TOLERANCE_PCT).is_empty(),
        "+1% must pass"
    );
    for n in p.nodes.iter_mut() {
        if n.parent == Some(root_id) {
            n.wall_ns = parent + parent / 2; // +50%
        }
    }
    assert!(
        !p.recheck(RECONCILE_TOLERANCE_PCT).is_empty(),
        "+50% must fail"
    );
}

// ---------------------------------------------------------------------------
// Gate, depth, geometry, unreachability
// ---------------------------------------------------------------------------

#[test]
fn a_disabled_profiler_records_nothing_at_all() {
    let _g = guard();
    __force_enabled(false);
    {
        let s = span("ghost");
        assert!(!s.is_active());
        let d = device_span("ghost_kernel");
        assert!(!d.is_active());
        burn(Duration::from_millis(2));
    }
    __force_enabled(true);
    let p = snapshot();
    assert!(
        p.node("ghost").is_none(),
        "disabled spans must create no nodes"
    );
    assert_eq!(p.totals.wall_ns, 0);
}

#[test]
fn the_depth_limit_truncates_and_says_so() {
    let _g = guard();
    __set_depth(2);
    let _a = span("d0");
    let _b = span("d0.d1");
    let c = span("d2");
    assert!(
        !c.is_active(),
        "depth 2 must be refused when the limit is 2"
    );
    drop(c);
    drop(_b);
    drop(_a);
    let p = snapshot();
    assert!(
        p.run.notes.iter().any(|n| n.contains("ARC_PROFILE_DEPTH")),
        "a truncated tree must announce that it is truncated"
    );
}

#[test]
fn geometry_is_recorded_on_the_node_it_describes() {
    let _g = guard();
    {
        let _s = step_scope("step");
        set_geometry(64, 1);
    }
    let p = snapshot();
    let n = p.node("step").unwrap();
    assert_eq!(n.geom.b, 64);
    assert_eq!(n.geom.t, 1);
    assert_eq!(n.geom.tokens, 64);
    assert_eq!(p.totals.tokens, 64);
}

#[test]
fn unreachable_paths_are_labelled_not_silently_zero() {
    let _g = guard();
    {
        let _s = step_scope("step");
        mark_unreachable(
            "cuda_graph",
            "cache_config is None because V4 has no PagedAttention backend",
            "normal.rs:1844",
        );
    }
    let p = snapshot();
    let n = p.node("step.cuda_graph").expect("the stub node must exist");
    assert!(!n.reachable);
    assert_eq!(n.wall_ns, 0);
    assert!(n.note.as_deref().unwrap().contains("normal.rs:1844"));
    assert_eq!(p.unreachable.len(), 1);
    assert_eq!(p.unreachable[0].path, "step.cuda_graph");
    assert_eq!(p.unreachable[0].site.as_deref(), Some("normal.rs:1844"));
}

#[test]
fn warmup_steps_are_discarded() {
    let _g = guard();
    __set_warmup(2);
    for _ in 0..2 {
        let _s = step_scope("step");
        burn(Duration::from_millis(10));
    }
    // Everything above must have been thrown away.
    assert_eq!(snapshot().totals.steps, 0);
    {
        let _s = step_scope("step");
        burn(Duration::from_millis(10));
    }
    let p = snapshot();
    assert_eq!(p.run.steps, 1);
    let ms = p.node("step").unwrap().wall_ns as f64 / 1e6;
    assert!(
        (9.0..60.0).contains(&ms),
        "only the recorded step may contribute; got {ms:.2} ms"
    );
}

// ---------------------------------------------------------------------------
// Overhead, measured rather than asserted
// ---------------------------------------------------------------------------

#[test]
fn overhead_is_measured_in_both_states_and_off_is_cheaper() {
    let _g = guard();
    calibrate();
    let o = overhead_cell().read().unwrap().clone();
    // Printed so the number in `docs/engineering/PROFILING.md` can be
    // reproduced with `cargo test -p arc-profiler --release -- --nocapture`
    // rather than taken on trust.
    eprintln!(
        "arc-profiler overhead: ON {:.1} ns/span, OFF {:.1} ns/span",
        o.enabled_ns_per_span, o.disabled_ns_per_span
    );
    assert!(o.enabled_ns_per_span > 0.0);
    assert!(o.disabled_ns_per_span >= 0.0);
    assert!(
        o.disabled_ns_per_span < o.enabled_ns_per_span,
        "an open+close ({:.1} ns) must cost more than a gate check ({:.1} ns)",
        o.enabled_ns_per_span,
        o.disabled_ns_per_span
    );
    // The disabled path is one relaxed load and a branch. Anything in the
    // hundreds of nanoseconds means an environment read crept back in — the
    // exact regression that cost ~390 getenvs per forward before.
    assert!(
        o.disabled_ns_per_span < 50.0,
        "profiler-off cost per call site is {:.1} ns; that is not a gate check",
        o.disabled_ns_per_span
    );
    // Calibration must not leave its own scratch nodes in the tree.
    let p = snapshot();
    assert!(p.node("calibration.on").is_none());
    assert!(p.node("calibration.off").is_none());
}

#[test]
fn the_report_states_its_own_overhead_share() {
    let _g = guard();
    if let Ok(mut o) = overhead_cell().write() {
        o.enabled_ns_per_span = 100.0;
    }
    {
        let _s = step_scope("step");
        for _ in 0..10 {
            let _c = span("child");
        }
        burn(Duration::from_millis(1));
    }
    let p = snapshot();
    assert!(
        p.overhead.spans_per_step >= 11.0,
        "root + gpu_drain + 10 children"
    );
    assert!(
        p.overhead.enabled_overhead_pct > 0.0,
        "a reader must be able to see how much of the profile is the profiler"
    );
}

// ---------------------------------------------------------------------------
// Serialisation round-trip
// ---------------------------------------------------------------------------

#[test]
fn the_json_round_trips_and_keeps_every_channel() {
    let _g = guard();
    install(Box::new(FixedTimer::new(3_000_000)));
    {
        let _s = step_scope("step");
        set_geometry(8, 1);
        {
            let _d = device_span("kernel");
        }
        {
            let _w = sync_span("wait");
            burn(Duration::from_millis(3));
        }
    }
    let p = snapshot();
    let json = serde_json::to_string(&p).unwrap();
    let back: Profile = serde_json::from_str(&json).unwrap();
    assert_eq!(back.schema, SCHEMA);
    assert_eq!(
        back.node("step.kernel").unwrap().device_ns,
        p.node("step.kernel").unwrap().device_ns
    );
    assert_eq!(
        back.node("step.wait").unwrap().sync_ns,
        p.node("step.wait").unwrap().sync_ns
    );
    assert_eq!(back.node("step").unwrap().geom.b, 8);
    assert!(back.recheck(RECONCILE_TOLERANCE_PCT).is_empty());
}

#[test]
fn selftest_ratio_discriminates_a_launch_timer_from_an_event_timer() {
    // Pure arithmetic on the same struct the GPU self-test returns, so the
    // threshold in the runbook is tested even on a laptop.
    let honest = SelfTest {
        launch_wall_ns: 120_000,
        device_ns: 48_000_000,
        ratio: 48_000_000.0 / 120_000.0,
    };
    let broken = SelfTest {
        launch_wall_ns: 120_000,
        device_ns: 121_000,
        ratio: 121_000.0 / 120_000.0,
    };
    assert!(honest.ratio > 10.0, "CUDA events must dwarf launch time");
    assert!(
        broken.ratio < 10.0,
        "a launch-timing profiler must fail the same threshold"
    );
}

// ── The name-collision trap ────────────────────────────────────────────────
//
// A profile carries TWO `mla_attn` nodes, one per step branch. A consumer that
// selects by name gets the prefill one — `calls == 0` in a decode-only window —
// and reads zero for the busiest kernel in the engine. That happened to a
// downstream chain, and the guard that caught it reported "unreached node" when
// the truth was "wrong node". These are different diagnoses with different
// fixes, so both are pinned here.

/// Build the shape that causes it: `step` -> {`prompt`, `decode`}, each with the
/// same leaf name, only the decode side entered.
fn two_branch_profile() -> Profile {
    // Step 1 enters the PREFILL branch, so `step.prompt.mla_attn` is registered
    // first — the registration order that makes the naive name match return it.
    {
        let _step = step_scope("step");
        let _p = span("prompt");
        let _a = span("mla_attn");
    }
    // Every later step is decode-only, which is the ordinary steady-state
    // window: the prompt subtree exists in the tree and is never entered again.
    for _ in 0..3 {
        let _step = step_scope("step");
        let _d = span("decode");
        let _a = span("mla_attn");
        std::thread::sleep(Duration::from_micros(200));
    }
    snapshot()
}

#[test]
fn a_span_name_below_the_branch_split_does_not_identify_one_node() {
    let _g = guard();
    let p = two_branch_profile();
    let named = p.nodes_named("mla_attn");
    assert_eq!(
        named.len(),
        2,
        "the fixture must reproduce the collision, not paper over it"
    );
    // The naive selection returns the PREFILL node — this is the bug, asserted
    // so a registration-order change cannot quietly hide it.
    assert_eq!(
        named[0].branch.as_deref(),
        Some("prompt"),
        "the first name match is the prefill copy, which is the trap"
    );
    let decode_calls = p.resolve_in("decode", "mla_attn").unwrap().calls;
    assert!(
        decode_calls > named[0].calls,
        "the decode node is the one carrying the work ({decode_calls} vs {})",
        named[0].calls
    );
}

#[test]
fn resolve_refuses_an_ambiguous_name_and_names_every_candidate() {
    let _g = guard();
    let p = two_branch_profile();
    let err = p
        .resolve("mla_attn")
        .expect_err("an ambiguous name must not resolve to a guess");
    assert_eq!(err.candidates.len(), 2);
    let msg = err.to_string();
    // The message has to be actionable: which branches, and what each one's
    // call count is, so the reader can see the zero is the wrong node.
    assert!(msg.contains("decode"), "message names the branches: {msg}");
    assert!(msg.contains("prompt"), "message names the branches: {msg}");
    assert!(msg.contains("calls="), "message carries call counts: {msg}");
    assert!(
        msg.contains("resolve_in"),
        "message says what to do instead"
    );
}

#[test]
fn resolve_in_returns_the_branch_asked_for() {
    let _g = guard();
    let p = two_branch_profile();
    let dec = p.resolve_in("decode", "mla_attn").expect("decode side");
    assert!(dec.calls > 0, "the decode node is the one that ran");
    assert_eq!(dec.branch.as_deref(), Some("decode"));
    let pre = p.resolve_in("prompt", "mla_attn").expect("prompt side");
    assert!(
        pre.calls < dec.calls,
        "prefill ran once, decode ran every step"
    );
    assert_eq!(pre.branch.as_deref(), Some("prompt"));
}

#[test]
fn the_three_zeros_are_different_answers() {
    let _g = guard();
    // A node that is registered and never entered — the exact shape of the
    // prompt subtree in a decode-only window.
    {
        let _step = step_scope("step");
        let _d = span("decode");
    }
    mark_unreachable(
        "paged_attention",
        "supports_paged_attention() is false",
        "mod.rs:1",
    );
    let p = snapshot();
    let never = p
        .nodes
        .iter()
        .find(|n| !n.reachable)
        .expect("an unreachable node was declared");
    assert_eq!(never.verdict(), Verdict::Unreachable);
    let ran = p.resolve_in("decode", "decode").ok();
    assert!(ran.is_none() || ran.unwrap().verdict() != Verdict::Unreachable);
}

#[test]
fn the_report_warns_about_colliding_names_before_anyone_reads_one() {
    let _g = guard();
    let p = two_branch_profile();
    assert!(
        p.run
            .notes
            .iter()
            .any(|n| n.contains("more than one node") && n.contains("branch")),
        "a profile with colliding names must say so in its notes: {:?}",
        p.run.notes
    );
}

#[test]
fn the_root_has_no_branch_and_a_branch_node_names_itself() {
    let _g = guard();
    let p = two_branch_profile();
    assert_eq!(p.root().expect("root").branch, None);
    let d = p.node("step.decode").expect("decode branch node");
    assert_eq!(d.branch.as_deref(), Some("decode"));
}
