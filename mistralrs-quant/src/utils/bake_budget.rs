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

use std::sync::Mutex;

use candle_core::Result;

/// Fraction of total device memory kept in reserve when deciding whether a
/// bake can finish. Overridable with `ARC_BAKE_HEADROOM`.
const DEFAULT_HEADROOM: f64 = 0.08;

/// Minimum number of completed layers before the guard is willing to project.
/// Growth is measured as a delta, so one sample tells us nothing.
const MIN_SAMPLES: usize = 2;

/// Result of projecting a bake's peak device usage from the layers observed
/// so far. Pure data — [`project_bake_peak`] computes it without touching the
/// driver, so the arithmetic is unit-testable without a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BakeProjection {
    /// Device bytes in use right now (includes pool-cached and fragmented
    /// blocks, exactly as `cuMemGetInfo` reports them).
    pub used_now: u64,
    /// Bytes of device usage added per completed layer, taken as the worse of
    /// the run average and the most recent layer.
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
/// * `baseline_used` — device usage sampled right after the *first* layer. Using
///   the first layer as the origin excludes the one-time working set (CUDA
///   context, quantize scratch, non-layer weights) from the per-layer slope.
/// * `used_now` / `last_used` — usage after this layer and after the previous one.
/// * `device_total` — total device memory.
/// * `headroom` — fraction of `device_total` to keep in reserve.
///
/// The slope is `max(average, most recent)`: an allocator that starts
/// fragmenting late accelerates, and the average alone would notice far too
/// slowly to save any money.
pub fn project_bake_peak(
    layers_done: usize,
    total_layers: usize,
    baseline_used: u64,
    used_now: u64,
    last_used: u64,
    device_total: u64,
    headroom: f64,
) -> BakeProjection {
    let spans = layers_done.saturating_sub(1).max(1) as u64;
    let average = used_now.saturating_sub(baseline_used) / spans;
    let recent = used_now.saturating_sub(last_used);
    let growth_per_layer = average.max(recent);
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

#[derive(Debug)]
struct BakeBudgetState {
    total_layers: usize,
    headroom: f64,
    layers_done: usize,
    /// Usage right after the first layer; the origin for the slope.
    baseline_used: u64,
    /// Usage after the previous layer.
    last_used: u64,
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
        baseline_used: 0,
        last_used: 0,
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

    state.layers_done += 1;
    if state.layers_done == 1 {
        state.baseline_used = used_now;
        state.last_used = used_now;
        return Ok(());
    }

    let projection = project_bake_peak(
        state.layers_done,
        state.total_layers,
        state.baseline_used,
        used_now,
        state.last_used,
        device_total,
        state.headroom,
    );
    state.last_used = used_now;

    if state.layers_done < MIN_SAMPLES || projection.fits() {
        return Ok(());
    }

    let (layers_done, total_layers, headroom) =
        (state.layers_done, state.total_layers, state.headroom);
    drop(guard);
    disarm_bake_budget();

    candle_core::bail!(
        "UQFF bake will not fit on this device: after {layers_done}/{total_layers} quantized \
         layers the GPU holds {used:.1} GiB and each further layer is adding {growth:.2} GiB, \
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

    /// The wave18 V4-Flash bake, replayed from its measured samples: 43 layers
    /// on a 140 GB H200, growing ~1.8 GiB/layer up to layer 22 and then
    /// accelerating to ~4.45 GiB/layer. The guard must stay quiet while the
    /// flat trend projects a fit and must fire once the acceleration does not.
    #[test]
    fn wave18_v4_flash_samples_trip_the_guard_at_the_acceleration() {
        let device_total = 140 * GIB;
        let headroom = 0.08;
        // Sample after layer 7 is the baseline origin for this reconstruction.
        let baseline = 21 * GIB + GIB / 2; // 21.5 GiB @ layer 7

        // Layer 22: 48.7 GiB, average slope 1.81 GiB/layer, recent 1.71.
        let flat = project_bake_peak(
            16, // layers 7..=22 inclusive => 16 samples
            43,
            baseline,
            48 * GIB + (7 * GIB) / 10,
            47 * GIB,
            device_total,
            headroom,
        );
        assert!(
            flat.fits(),
            "flat 1.8 GiB/layer trend must not trip the guard: {flat:?}"
        );

        // Layer 24: 57.6 GiB after a 4.45 GiB/layer jump from layer 22.
        let accelerating = project_bake_peak(
            18,
            43,
            baseline,
            57 * GIB + (6 * GIB) / 10,
            53 * GIB + (GIB / 10),
            device_total,
            headroom,
        );
        assert!(
            !accelerating.fits(),
            "the 4.45 GiB/layer acceleration that OOMed at layer 28 must trip the guard: \
             {accelerating:?}"
        );
    }

    /// A bake whose device usage is flat (the shape produced by
    /// `bake_isq_to_host`) must never trip the guard, however many layers.
    #[test]
    fn flat_device_usage_never_trips() {
        let p = project_bake_peak(20, 43, 12 * GIB, 12 * GIB, 12 * GIB, 80 * GIB, 0.08);
        assert_eq!(p.growth_per_layer, 0);
        assert_eq!(p.projected_peak, 12 * GIB);
        assert!(p.fits());
    }

    /// The slope is the worse of average and most-recent, so a late-starting
    /// leak is caught on the layer it appears rather than diluted.
    #[test]
    fn slope_takes_the_worse_of_average_and_recent() {
        // Average over 9 spans is ~1 GiB; the last layer alone added 10 GiB.
        let p = project_bake_peak(10, 20, 0, 19 * GIB, 9 * GIB, 200 * GIB, 0.0);
        assert_eq!(p.growth_per_layer, 10 * GIB);
        assert_eq!(p.remaining_layers, 10);
        assert_eq!(p.projected_peak, 19 * GIB + 100 * GIB);
    }

    /// Headroom is applied to the device total, and clamped to something sane.
    #[test]
    fn headroom_shrinks_the_capacity() {
        let p = project_bake_peak(2, 3, 0, GIB, GIB, 100 * GIB, 0.10);
        assert_eq!(p.capacity, 90 * GIB);
        // An absurd headroom is clamped to 0.9 rather than starving the budget
        // to nothing. (Float rounding puts this a byte under 10 GiB.)
        let clamped = project_bake_peak(2, 3, 0, GIB, GIB, 100 * GIB, 5.0);
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
        let p = project_bake_peak(43, 43, 0, 70 * GIB, 68 * GIB, 140 * GIB, 0.08);
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
