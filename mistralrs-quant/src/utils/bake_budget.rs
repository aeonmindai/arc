//! Fail-fast device-memory budget for UQFF bakes.
//!
//! A UQFF bake loads and quantizes one layer at a time and only writes the
//! artifact once every layer is done. That makes a device OOM near the end of
//! the run a *total* loss: wave18 lost a 43-layer V4-Flash bake at layer 28
//! after two hours, with a 4 KB output directory to show for it.
//!
//! The cure for the OOM itself is to stop retaining the quantized expert
//! stacks on the device (see [`crate::bake_isq_to_host`]). This module is the
//! seatbelt for everything that cure does not cover — a different rung, a
//! larger model, a smaller card, an allocator that fragments faster than we
//! modelled. It samples device usage once per quantized layer and refuses to
//! keep burning GPU-hours when the observed growth rate says the bake cannot
//! finish.
//!
//! The projection deliberately measures *device usage*, not the byte size of
//! the tensors we produced: usage includes the CUDA async pool's cached and
//! fragmented blocks, which is precisely the failure mode a tensor-size
//! accounting would miss.

use std::collections::VecDeque;
use std::sync::Mutex;

use candle_core::Result;

/// Fraction of total device memory kept in reserve when deciding whether a
/// bake can finish. Overridable with `ARC_BAKE_HEADROOM`.
const DEFAULT_HEADROOM: f64 = 0.08;

/// Layers of growth the slope is measured over.
///
/// Not "since the start": a bake's early layers include one-time working-set
/// costs, and a slope anchored there notices a late-starting leak far too
/// slowly. A trailing window tracks the current behaviour.
const SLOPE_WINDOW: usize = 4;

/// Consecutive over-budget projections required before the guard stops the bake.
///
/// Deliberately one more than [`SLOPE_WINDOW`]. A single unlucky layer — the
/// CUDA pool grabbing a whole scratch block, say — lifts the windowed slope for
/// exactly `SLOPE_WINDOW` samples and then falls out of the window, so it can
/// never reach this count. Sustained growth can. Killing a healthy two-hour
/// bake on allocator noise would be worse than the failure this guard exists to
/// prevent.
const CONSECUTIVE_TRIPS: usize = SLOPE_WINDOW + 1;

/// Result of projecting a bake's peak device usage from the layers observed
/// so far. Pure data — [`project_bake_peak`] computes it without touching the
/// driver, so the arithmetic is unit-testable without a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BakeProjection {
    /// Device bytes in use right now (includes pool-cached and fragmented
    /// blocks, exactly as `cuMemGetInfo` reports them).
    pub used_now: u64,
    /// Bytes of device usage added per layer, averaged over the trailing
    /// [`SLOPE_WINDOW`] layers.
    pub growth_per_layer: u64,
    /// Layers still to quantize.
    pub remaining_layers: usize,
    /// `used_now + remaining_layers * growth_per_layer`.
    pub projected_peak: u64,
    /// `device_total * (1 - headroom)`.
    pub capacity: u64,
}

impl BakeProjection {
    /// Whether the bake is projected to finish inside the budget.
    pub fn fits(&self) -> bool {
        self.projected_peak <= self.capacity
    }
}

/// Project the peak device usage of a bake from the usage observed so far.
///
/// * `layers_done` — quantized layers completed, including this one.
/// * `total_layers` — layers the bake will quantize in total.
/// * `window_start_used` — device usage `window_spans` layers ago.
/// * `used_now` — device usage after this layer.
/// * `window_spans` — how many layers separate the two samples (at most
///   [`SLOPE_WINDOW`]).
/// * `device_total` — total device memory.
/// * `headroom` — fraction of `device_total` to keep in reserve.
///
/// The slope is the mean growth across the window; the projection assumes it
/// continues for every remaining layer.
pub fn project_bake_peak(
    layers_done: usize,
    total_layers: usize,
    window_start_used: u64,
    used_now: u64,
    window_spans: usize,
    device_total: u64,
    headroom: f64,
) -> BakeProjection {
    let spans = window_spans.max(1) as u64;
    let growth_per_layer = used_now.saturating_sub(window_start_used) / spans;
    let remaining_layers = total_layers.saturating_sub(layers_done);
    let projected_peak =
        used_now.saturating_add(growth_per_layer.saturating_mul(remaining_layers as u64));
    let headroom = headroom.clamp(0.0, 0.9);
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    let capacity = (device_total as f64 * (1.0 - headroom)) as u64;
    BakeProjection {
        used_now,
        growth_per_layer,
        remaining_layers,
        projected_peak,
        capacity,
    }
}

/// What one device-usage sample means for the bake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeBudgetVerdict {
    /// First sample: recorded as the slope's origin, nothing to project yet.
    Baseline,
    /// Projected to fit.
    Ok,
    /// Projected over budget, but not yet for [`CONSECUTIVE_TRIPS`]
    /// consecutive samples — logged, not acted on.
    Watching,
    /// Over budget for [`CONSECUTIVE_TRIPS`] consecutive samples: stop.
    Stop,
}

#[derive(Debug)]
struct BakeBudgetState {
    total_layers: usize,
    headroom: f64,
    layers_done: usize,
    /// Trailing device-usage samples, newest last, at most `SLOPE_WINDOW + 1`
    /// entries — enough for `SLOPE_WINDOW` growth spans.
    window: VecDeque<u64>,
    /// How many consecutive layers have projected over budget.
    consecutive_over: usize,
}

impl BakeBudgetState {
    /// Fold one device-usage sample into the state and return the verdict, plus
    /// the projection it was based on (`None` for the very first sample, which
    /// only establishes the origin).
    ///
    /// Pure apart from `self`, so the window and the debounce are testable
    /// without a device.
    fn step(
        &mut self,
        used_now: u64,
        device_total: u64,
    ) -> (BakeBudgetVerdict, Option<BakeProjection>) {
        self.layers_done += 1;
        self.window.push_back(used_now);
        while self.window.len() > SLOPE_WINDOW + 1 {
            self.window.pop_front();
        }
        if self.window.len() < 2 {
            return (BakeBudgetVerdict::Baseline, None);
        }

        let projection = project_bake_peak(
            self.layers_done,
            self.total_layers,
            *self.window.front().expect("window is non-empty"),
            used_now,
            self.window.len() - 1,
            device_total,
            self.headroom,
        );

        if projection.fits() {
            self.consecutive_over = 0;
            return (BakeBudgetVerdict::Ok, Some(projection));
        }

        self.consecutive_over += 1;
        let verdict = if self.consecutive_over < CONSECUTIVE_TRIPS {
            BakeBudgetVerdict::Watching
        } else {
            BakeBudgetVerdict::Stop
        };
        (verdict, Some(projection))
    }
}

static BAKE_BUDGET: Mutex<Option<BakeBudgetState>> = Mutex::new(None);

fn headroom_from_env() -> f64 {
    std::env::var("ARC_BAKE_HEADROOM")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|f| (0.0..0.9).contains(f))
        .unwrap_or(DEFAULT_HEADROOM)
}

/// Arm the guard for a bake of `total_layers` quantized layers.
///
/// Idempotent per bake: re-arming resets the samples. Call
/// [`disarm_bake_budget`] when the load finishes so a subsequent serve in the
/// same process is not policed by a stale budget.
pub fn arm_bake_budget(total_layers: usize) {
    let headroom = headroom_from_env();
    tracing::info!(
        "UQFF bake memory guard armed: {total_layers} layers, {:.0}% device headroom \
         (set ARC_BAKE_HEADROOM to change).",
        headroom * 100.0
    );
    *BAKE_BUDGET.lock().expect("bake budget poisoned") = Some(BakeBudgetState {
        total_layers,
        headroom,
        layers_done: 0,
        window: VecDeque::with_capacity(SLOPE_WINDOW + 1),
        consecutive_over: 0,
    });
}

/// Disarm the guard. Safe to call when it was never armed.
pub fn disarm_bake_budget() {
    *BAKE_BUDGET.lock().expect("bake budget poisoned") = None;
}

/// Whether the guard is currently armed.
pub fn bake_budget_armed() -> bool {
    BAKE_BUDGET.lock().expect("bake budget poisoned").is_some()
}

/// Sample device memory. `None` when there is nothing to sample (no CUDA
/// feature, no CUDA device, or the driver call failed) — the guard then stays
/// quiet rather than guessing.
#[cfg(feature = "cuda")]
fn sample_device_memory() -> Option<(u64, u64)> {
    use candle_core::cuda::cudarc::driver::result;
    // Same call the PagedAttention KV budget uses
    // (`mistralrs_core::utils::memory_usage`). It needs a bound context; the
    // bake calls this from the thread that just ran the quantize kernels, and
    // if there is none we return `None` and stay quiet rather than guess.
    let (free, total) = result::mem_get_info().ok()?;
    if total == 0 {
        return None;
    }
    Some((total.saturating_sub(free) as u64, total as u64))
}

#[cfg(not(feature = "cuda"))]
fn sample_device_memory() -> Option<(u64, u64)> {
    None
}

/// Record that one more quantized layer is done, and refuse to continue if the
/// observed growth says the bake cannot fit on this device.
///
/// No-op unless [`arm_bake_budget`] was called and device memory is
/// observable. Costs one `cuMemGetInfo` per layer.
pub fn note_bake_layer() -> Result<()> {
    let Some((used_now, device_total)) = sample_device_memory() else {
        return Ok(());
    };
    let mut guard = BAKE_BUDGET.lock().expect("bake budget poisoned");
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };

    let (verdict, projection) = state.step(used_now, device_total);
    let (layers_done, total_layers, headroom) =
        (state.layers_done, state.total_layers, state.headroom);
    let consecutive_over = state.consecutive_over;
    let Some(projection) = projection else {
        return Ok(());
    };

    match verdict {
        BakeBudgetVerdict::Baseline | BakeBudgetVerdict::Ok => return Ok(()),
        BakeBudgetVerdict::Watching => {
            tracing::warn!(
                "UQFF bake memory guard: layer {layers_done}/{total_layers} projects a peak of \
                 {:.1} GiB against a {:.1} GiB budget ({consecutive_over} of \
                 {CONSECUTIVE_TRIPS} consecutive samples over). Watching; a single sample can \
                 be allocator noise.",
                gib(projection.projected_peak),
                gib(projection.capacity),
            );
            return Ok(());
        }
        BakeBudgetVerdict::Stop => {}
    }

    drop(guard);
    disarm_bake_budget();

    candle_core::bail!(
        "UQFF bake will not fit on this device: for {CONSECUTIVE_TRIPS} consecutive layers now, \
         the last at {layers_done}/{total_layers}, the GPU has held {used:.1} GiB and grown by \
         {growth:.2} GiB per layer over the trailing {SLOPE_WINDOW} layers, \
         so the remaining {remaining} layers project a peak of {peak:.1} GiB against a budget \
         of {capacity:.1} GiB ({total_mem:.1} GiB total, {headroom_pct:.0}% headroom).\n\
         Stopping now rather than spending the rest of the run to OOM with nothing written.\n\
         Options: bake on a larger device; split the bake with `--isq-organization moqe`; \
         shrink the per-layer quantize working set with ARC_QTIP_EXPERT_BATCH (default 16) \
         or ARC_VITERBI_SCRATCH_GB (default 6); or, if you know the tail is flat, lower \
         ARC_BAKE_HEADROOM.",
        used = gib(projection.used_now),
        growth = gib(projection.growth_per_layer),
        remaining = projection.remaining_layers,
        peak = gib(projection.projected_peak),
        capacity = gib(projection.capacity),
        total_mem = gib(device_total),
        headroom_pct = headroom * 100.0,
    )
}

#[allow(clippy::cast_precision_loss)]
fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn state(total_layers: usize, headroom: f64) -> BakeBudgetState {
        BakeBudgetState {
            total_layers,
            headroom,
            layers_done: 0,
            window: VecDeque::with_capacity(SLOPE_WINDOW + 1),
            consecutive_over: 0,
        }
    }

    /// Drive a whole bake's worth of samples through the state machine.
    fn run(
        total_layers: usize,
        headroom: f64,
        total_mem: u64,
        used: &[u64],
    ) -> Vec<BakeBudgetVerdict> {
        let mut st = state(total_layers, headroom);
        used.iter().map(|u| st.step(*u, total_mem).0).collect()
    }

    /// The post-fix shape — device usage flat across every layer — must never
    /// even reach `Watching`, on either card size.
    #[test]
    fn a_flat_bake_never_warns() {
        for total_mem in [80 * GIB, 140 * GIB] {
            let verdicts = run(43, 0.08, total_mem, &[11 * GIB; 43]);
            assert_eq!(verdicts[0], BakeBudgetVerdict::Baseline);
            assert!(
                verdicts[1..].iter().all(|v| *v == BakeBudgetVerdict::Ok),
                "a flat bake warned on a {} GiB card: {verdicts:?}",
                total_mem / GIB
            );
        }
    }

    /// The pre-fix shape this PR removes: the artifact accumulating on the card
    /// at 1.61 GiB/layer. On an 80 GB H100 that cannot finish 43 layers, and
    /// the guard must say so early — long before the money is spent.
    #[test]
    fn steady_retention_growth_is_caught_in_the_first_quarter_of_the_bake() {
        let used: Vec<u64> = (0..43)
            .map(|i| 12 * GIB + (i as u64) * (161 * GIB / 100))
            .collect();
        let verdicts = run(43, 0.08, 80 * GIB, &used);
        let stop_at = verdicts
            .iter()
            .position(|v| *v == BakeBudgetVerdict::Stop)
            .expect("a bake that cannot fit must be stopped");
        assert!(
            stop_at < 43 / 4,
            "stopped at layer {stop_at} of 43 — too late to save the run"
        );
    }

    /// One unlucky layer — the CUDA pool grabbing a whole scratch block — must
    /// not kill a healthy bake. The jump lifts the windowed slope for exactly
    /// `SLOPE_WINDOW` samples, one short of `CONSECUTIVE_TRIPS`, and then falls
    /// out of the window.
    #[test]
    fn a_one_off_allocator_jump_never_stops_the_bake() {
        let mut used = vec![10 * GIB; 43];
        for u in used.iter_mut().skip(6) {
            *u = 30 * GIB; // a single 20 GiB step at layer 6, flat thereafter
        }
        let verdicts = run(43, 0.08, 80 * GIB, &used);
        assert!(
            !verdicts.contains(&BakeBudgetVerdict::Stop),
            "a one-off jump stopped a bake that would have finished: {verdicts:?}"
        );
        assert!(
            verdicts.contains(&BakeBudgetVerdict::Watching),
            "the jump should still have been logged: {verdicts:?}"
        );
    }

    /// Sustained unaffordable growth stops the bake — but only after
    /// `CONSECUTIVE_TRIPS` confirmations, never on the first alarming sample.
    #[test]
    fn sustained_growth_stops_only_after_the_debounce() {
        let used: Vec<u64> = (0..43).map(|i| 10 * GIB + (i as u64) * 5 * GIB).collect();
        let verdicts = run(43, 0.08, 140 * GIB, &used);
        let first_over = verdicts
            .iter()
            .position(|v| matches!(v, BakeBudgetVerdict::Watching | BakeBudgetVerdict::Stop))
            .expect("growth this steep must be noticed");
        let stop_at = verdicts
            .iter()
            .position(|v| *v == BakeBudgetVerdict::Stop)
            .expect("growth this steep must be stopped");
        assert_eq!(
            stop_at - first_over,
            CONSECUTIVE_TRIPS - 1,
            "stop must follow exactly {CONSECUTIVE_TRIPS} consecutive over-budget samples: \
             {verdicts:?}"
        );
    }

    /// wave18's own measured curve, for the record. The steady 1.7-1.9 GiB/layer
    /// stretch genuinely projects to ~85-95 GiB on a 140 GB card, i.e. it fits —
    /// the run was killed by a late nonlinearity, not by a trend an early
    /// projection could have seen. This test pins that honest answer so nobody
    /// later assumes the guard would have saved this particular bake.
    #[test]
    fn the_measured_v4_flash_trend_projects_to_fit_on_a_140gb_card() {
        // Trailing window at layer 22: 1.71 GiB/layer, usage 48.7 GiB.
        let at_22 = project_bake_peak(
            22,
            43,
            48 * GIB + (7 * GIB) / 10 - 4 * (171 * GIB / 100),
            48 * GIB + (7 * GIB) / 10,
            SLOPE_WINDOW,
            140 * GIB,
            0.08,
        );
        assert!(
            at_22.fits(),
            "the measured steady trend fits; the guard is a seatbelt for other \
             shapes, not a retro-fix for this one: {at_22:?}"
        );
        // The same slope on an 80 GB card does NOT fit — which is the case the
        // guard is actually for.
        let on_h100 = project_bake_peak(
            22,
            43,
            48 * GIB + (7 * GIB) / 10 - 4 * (171 * GIB / 100),
            48 * GIB + (7 * GIB) / 10,
            SLOPE_WINDOW,
            80 * GIB,
            0.08,
        );
        assert!(!on_h100.fits(), "{on_h100:?}");
    }

    /// Headroom is applied to the device total, and clamped to something sane.
    #[test]
    fn headroom_shrinks_the_capacity() {
        let p = project_bake_peak(2, 3, 0, GIB, 1, 100 * GIB, 0.10);
        assert_eq!(p.capacity, 90 * GIB);
        // An absurd headroom is clamped to 0.9 rather than starving the budget
        // to nothing. (Float rounding puts this a byte under 10 GiB.)
        let clamped = project_bake_peak(2, 3, 0, GIB, 1, 100 * GIB, 5.0);
        assert!(
            clamped.capacity.abs_diff(10 * GIB) <= 1,
            "clamped capacity was {}",
            clamped.capacity
        );
    }

    /// The last layer has nothing left to project, so a bake that is already
    /// over budget at the finish line is not reported as a failure.
    #[test]
    fn final_layer_projects_only_what_is_already_used() {
        let p = project_bake_peak(43, 43, 60 * GIB, 70 * GIB, SLOPE_WINDOW, 140 * GIB, 0.08);
        assert_eq!(p.remaining_layers, 0);
        assert_eq!(p.projected_peak, 70 * GIB);
        assert!(p.fits());
    }

    /// Arming and disarming is observable, so the loader can prove it reset
    /// the guard between a bake and a subsequent serve in the same process.
    #[test]
    fn arm_and_disarm_round_trip() {
        disarm_bake_budget();
        assert!(!bake_budget_armed());
        arm_bake_budget(43);
        assert!(bake_budget_armed());
        // With no CUDA device to sample, the guard is inert rather than noisy.
        note_bake_layer().expect("guard must be a no-op without device memory info");
        disarm_bake_budget();
        assert!(!bake_budget_armed());
    }
}
