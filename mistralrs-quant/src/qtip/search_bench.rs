//! Head-to-head harness for the two trellis-search knobs (wave13-AD).
//!
//! Today's bake runs an exhaustive Viterbi over a `2^L = 65,536`-state trellis
//! minimising **unweighted** weight-MSE: ~3.6–8.5 min/layer on an H200, ~6 h for
//! a 44-layer model, and the wrong objective. This module measures the two
//! independent fixes side by side so the *table*, not intuition, picks the
//! default:
//!
//! * **Axis A — search.** `TrellisSearch::Beam { width }` keeps the best `W`
//!   states per timestep instead of all `2^L`.
//! * **Axis B — objective.** A diagonal activation Hessian
//!   (`H = (1/N)·XᵀX`, GPTQ/LDLQ/QuIP#/QTIP) replaces `‖w − ŵ‖²`.
//!
//! The grid is `{exhaustive, beam 64/128/256} × {unweighted, hessian-weighted}`
//! — 8 cells, each reporting matmul cosine vs the dense reference, weight NMSE,
//! Hessian-weighted NMSE, and wall-time per row.
//!
//! ## Why these fixtures
//!
//! Pure-Gaussian weights and i.i.d. activations hide exactly the effects under
//! test: a Gaussian row is already matched to the LUT (so the beam loses
//! nothing) and an isotropic activation covariance makes `diag(H)` flat (so the
//! weighted objective is a no-op by construction). Following the
//! `bake_quality_tests` precedent we use:
//!
//! * **Weights** — heavy-tailed Student-t(4) rows snapped to the FP4 (e2m1)
//!   lattice with per-32-column block scales: what "INT4-packed FP4 experts
//!   dequantized to BF16" actually produces.
//! * **Activations** — per-channel log-normal scales with a heavy outlier tail,
//!   the well-documented shape of LLM activations (the premise of AWQ /
//!   SmoothQuant). Calibration and evaluation draws use **independent seeds**,
//!   so a weighted config cannot win by memorising its own calibration set.
//!
//! ## Protocol
//!
//! Each configuration runs the production CPU pipeline op-for-op — rotate
//! (Hadamard, block 128) → scale `max|row|/3` → trellis search → pack → decode
//! → un-rotate — differing *only* in the search strategy and the branch metric.
//! Rows are quantized sequentially (not via rayon) so wall-time per row is a
//! clean per-row number rather than a core-count artefact.
//!
//! ## CUDA kernel status (axis A landed in wave13-AF)
//!
//! `kernels/qtip/qtip_beam.cu` is the GPU twin of [`super::viterbi`]'s beam and
//! is bit-identical to it at the same width
//! (`cuda_beam_matches_cpu_beam_bit_for_bit`). It differs from the plan sketched
//! here in one respect worth recording: no sort is needed anywhere. Because all
//! 16 successors of a beam entry depend on that entry only through its low
//! `L−K` bits, distinct "groups" produce disjoint successor sets, so the
//! CPU's dedup collapses to one 64-bit `atomicMin` per group over a
//! 4,096-entry shared table and the expanded candidate list is *already*
//! duplicate-free. Selection of the best `W` is then a radix-select on the
//! 48-bit `(cost, state)` key rather than a `BlockRadixSort`, and the beam's
//! order is irrelevant (every consumer is an argmin or an explicit parent
//! index). See [`beam_bake_cost_model`] for the byte arithmetic.
//!
//! Axis B (the Hessian-weighted branch metric) is still CPU-only, so
//! `quantize_with_options_concrete_calibrated` continues to refuse `ARC_QTIP_HESSIAN`
//! on CUDA rather than silently baking the unweighted objective under a
//! weighted label.

use super::bake_quality_tests::{
    cosine, decode_packed, gen_fp4_dequant, matmul_t, pack_symbols, Rng,
};
use super::{
    apply_block_rotation, gaussian_lut, hessian_row_weights, quantize_row, TrellisSearch,
    QTIP_ROTATION_SEED, V,
};
use crate::turboquant::wht::generate_signs;

/// Production rotation width (`rotation_block_size` caps at 128).
const ROT_BLOCK: usize = 128;
/// Rotation disabled — the arm that isolates axis B from incoherence
/// processing (see `hessian_weights_are_block_means_under_rotation`).
const NO_ROT: usize = 0;

// ---------------------------------------------------------------------------
// Activation fixture
// ---------------------------------------------------------------------------

/// Per-input-channel activation scales: log-normal bulk plus a 3% outlier tail
/// at ~10x. This is the structure that makes `diag(XᵀX)` informative — without
/// it, axis B is measuring nothing.
fn channel_scales(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..k)
        .map(|_| {
            let base = (rng.normal() * 0.9).exp();
            let outlier = if rng.normal() > 1.88 { 10.0 } else { 1.0 };
            (base * outlier) as f32
        })
        .collect()
}

/// `[batch, k]` activations with the given per-channel scales.
fn gen_activations(batch: usize, scales: &[f32], seed: u64) -> Vec<f32> {
    let k = scales.len();
    let mut rng = Rng::new(seed);
    let mut x = vec![0f32; batch * k];
    for b in 0..batch {
        for j in 0..k {
            x[b * k + j] = (rng.normal() as f32) * scales[j];
        }
    }
    x
}

/// `diag(H) = (1/N) Σ_n x_{n,j}²` — exactly what `ImatrixLayerStats` collects.
fn hessian_diag(x: &[f32], batch: usize, k: usize) -> Vec<f32> {
    let mut h = vec![0f64; k];
    for b in 0..batch {
        for j in 0..k {
            let v = x[b * k + j] as f64;
            h[j] += v * v;
        }
    }
    h.into_iter().map(|v| (v / batch as f64) as f32).collect()
}

// ---------------------------------------------------------------------------
// Configuration under test
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SearchCfg {
    search: TrellisSearch,
    weighted: bool,
}

impl SearchCfg {
    fn label(&self) -> String {
        let s = match self.search {
            TrellisSearch::Exhaustive => "exhaustive".to_string(),
            TrellisSearch::Beam { width } => format!("beam W={width}"),
        };
        let o = if self.weighted {
            "hessian"
        } else {
            "unweighted"
        };
        format!("{s:<13} {o:<10}")
    }
}

struct CellResult {
    matmul_cos: f64,
    weight_nmse: f64,
    hessian_nmse: f64,
    secs_per_row: f64,
}

/// Quantize `[n, k]` through the production CPU pipeline under `cfg`, returning
/// the reconstruction in the ORIGINAL frame plus the sequential wall time.
///
/// Op-for-op identical to `QtipLayer::quantize_with_options_concrete` except
/// that the search strategy / branch metric come from `cfg` and rows are walked
/// sequentially so the timing is per-row.
fn quantize_matrix(
    w: &[f32],
    n: usize,
    k: usize,
    cfg: SearchCfg,
    h_diag: &[f32],
    rot_block: usize,
) -> (Vec<f32>, f64) {
    let lut = gaussian_lut();
    let signs = generate_signs(QTIP_ROTATION_SEED, k);
    let weights = cfg.weighted.then(|| hessian_row_weights(h_diag, rot_block));
    let num_symbols = k / V as usize;

    let mut out = vec![0f32; n * k];
    let start = std::time::Instant::now();
    for row in 0..n {
        let mut rot = w[row * k..(row + 1) * k].to_vec();
        if rot_block >= 2 {
            apply_block_rotation(&mut rot, &signs, rot_block);
        }

        let max_abs = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 3.0 };
        let inv = 1.0 / scale;
        let target: Vec<f32> = rot.iter().map(|&v| v * inv).collect();

        let syms = quantize_row(&target, &lut, cfg.search, weights.as_deref());
        let packed = pack_symbols(&syms);

        let mut recon: Vec<f32> = decode_packed(&packed, num_symbols, &lut)
            .into_iter()
            .map(|c| c * scale)
            .collect();
        if rot_block >= 2 {
            apply_block_rotation(&mut recon, &signs, rot_block);
        }
        out[row * k..(row + 1) * k].copy_from_slice(&recon);
    }
    let secs_per_row = start.elapsed().as_secs_f64() / n as f64;
    (out, secs_per_row)
}

/// Relative (normalised) error `Σ h_j (w_j − ŵ_j)² / Σ h_j w_j²`.
/// With `h ≡ 1` this is the plain weight NMSE.
fn nmse(w: &[f32], w_hat: &[f32], h: Option<&[f32]>, k: usize) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (i, (&a, &b)) in w.iter().zip(w_hat.iter()).enumerate() {
        let hj = h.map_or(1.0, |h| h[i % k] as f64);
        let d = (a - b) as f64;
        num += hj * d * d;
        den += hj * (a as f64) * (a as f64);
    }
    num / den.max(1e-300)
}

/// Everything a cell is scored against: the dense reference weight, the
/// held-out activations, and the calibration Hessian diagonal.
struct EvalSet<'a> {
    w: &'a [f32],
    n: usize,
    k: usize,
    x_eval: &'a [f32],
    batch: usize,
    h_diag: &'a [f32],
}

fn evaluate(ev: &EvalSet<'_>, w_hat: &[f32], secs_per_row: f64) -> CellResult {
    let y = matmul_t(ev.x_eval, ev.w, ev.batch, ev.n, ev.k);
    let y_hat = matmul_t(ev.x_eval, w_hat, ev.batch, ev.n, ev.k);
    CellResult {
        matmul_cos: cosine(&y, &y_hat),
        weight_nmse: nmse(ev.w, w_hat, None, ev.k),
        hessian_nmse: nmse(ev.w, w_hat, Some(ev.h_diag), ev.k),
        secs_per_row,
    }
}

/// The 8-cell grid. `cargo test -p mistralrs-quant --lib search_bench -- --ignored --nocapture`.
///
/// Printed twice: once at the production rotation width (128) and once with
/// rotation disabled. The second table is not decoration — it is the control
/// that tells axis B's two possible failure modes apart. `diag(RᵀHR)` is the
/// per-block MEAN of `diag(H)`, so incoherence processing deliberately
/// flattens the very signal the weighted objective consumes; if the weighted
/// objective only pays off at `rot=0`, the honest conclusion is that Hadamard
/// rotation already captured that win and per-position weights are redundant.
#[test]
#[ignore = "benchmark harness (minutes); run with --ignored --nocapture"]
fn trellis_search_headroom_table() {
    run_grid(64, 2048, 256, 256, ROT_BLOCK, true);
    run_grid(64, 2048, 256, 256, NO_ROT, true);
}

/// A fast, always-on version of the same grid. It does not print a table; it
/// asserts the two invariants the table is supposed to establish, so a
/// regression in either axis fails CI rather than waiting for someone to read
/// a benchmark:
///
/// 1. every beam width stays within a small quality band of the exhaustive
///    baseline on matmul cosine, and
/// 2. every beam is strictly faster than the exhaustive search.
#[test]
fn trellis_search_grid_invariants() {
    let cells = run_grid(4, 512, 16, 16, ROT_BLOCK, false);
    let base = cells
        .iter()
        .find(|(c, _)| matches!(c.search, TrellisSearch::Exhaustive) && !c.weighted)
        .map(|(_, r)| (r.matmul_cos, r.secs_per_row))
        .expect("exhaustive/unweighted baseline missing");

    for (cfg, res) in &cells {
        assert!(
            res.matmul_cos.is_finite() && res.matmul_cos > 0.0,
            "{}: non-finite matmul cosine",
            cfg.label()
        );
        if let TrellisSearch::Beam { width } = cfg.search {
            assert!(
                res.matmul_cos >= base.0 - 0.05,
                "{}: matmul cos {:.4} fell more than 0.05 below the exhaustive baseline {:.4}",
                cfg.label(),
                res.matmul_cos,
                base.0
            );
            assert!(
                res.secs_per_row < base.1,
                "beam W={width} ({:.4}s/row) was not faster than exhaustive ({:.4}s/row)",
                res.secs_per_row,
                base.1
            );
        }
    }
}

fn run_grid(
    n: usize,
    k: usize,
    calib_batch: usize,
    eval_batch: usize,
    rot_block: usize,
    print: bool,
) -> Vec<(SearchCfg, CellResult)> {
    let w = gen_fp4_dequant(n, k, 0.02, 0x0051_EA11);
    let scales = channel_scales(k, 0xC0FF_EE01);
    // Independent draws: the Hessian comes from the calibration set, quality is
    // measured on a held-out set.
    let x_cal = gen_activations(calib_batch, &scales, 0x1234_5678);
    let x_eval = gen_activations(eval_batch, &scales, 0x8765_4321);
    let h = hessian_diag(&x_cal, calib_batch, k);

    let ev = EvalSet {
        w: &w,
        n,
        k,
        x_eval: &x_eval,
        batch: eval_batch,
        h_diag: &h,
    };

    let mut cells = Vec::new();
    for weighted in [false, true] {
        for search in [
            TrellisSearch::Exhaustive,
            TrellisSearch::Beam { width: 64 },
            TrellisSearch::Beam { width: 128 },
            TrellisSearch::Beam { width: 256 },
        ] {
            let cfg = SearchCfg { search, weighted };
            let (w_hat, secs) = quantize_matrix(&w, n, k, cfg, &h, rot_block);
            cells.push((cfg, evaluate(&ev, &w_hat, secs)));
        }
    }

    // Re-measure wall time in ROUND-ROBIN rounds over a short row prefix,
    // keeping the per-config minimum. Timing each config's repetitions in one
    // contiguous block lets a single scheduler or thermal event corrupt every
    // sample for that config; interleaving bounds the damage to one round.
    // The search is deterministic, so this only touches the timing column.
    if print {
        const TIMED_ROWS: usize = 8;
        const ROUNDS: usize = 5;
        let rows = TIMED_ROWS.min(n);
        for _ in 0..ROUNDS {
            for (cfg, res) in cells.iter_mut() {
                let (_, secs) = quantize_matrix(&w, rows, k, *cfg, &h, rot_block);
                res.secs_per_row = res.secs_per_row.min(secs);
            }
        }
    }

    if print {
        let base = cells
            .iter()
            .find(|(c, _)| matches!(c.search, TrellisSearch::Exhaustive) && !c.weighted)
            .map(|(_, r)| (r.matmul_cos, r.weight_nmse, r.hessian_nmse, r.secs_per_row))
            .unwrap();
        let rot = if rot_block >= 2 {
            format!("hadamard-{rot_block}")
        } else {
            "off".to_string()
        };
        println!("\n=== trellis search × objective (n={n}, k={k}, rotation={rot}, FP4-lattice weights, log-normal activations) ===");
        println!(
            "{:<24} | {:>9} | {:>10} | {:>10} | {:>9} | {:>7}",
            "configuration", "matmul_cos", "w_nmse", "h_nmse", "s/row", "speedup"
        );
        for (cfg, r) in &cells {
            println!(
                "{:<24} | {:>9.5} | {:>10.3e} | {:>10.3e} | {:>9.4} | {:>6.1}x",
                cfg.label(),
                r.matmul_cos,
                r.weight_nmse,
                r.hessian_nmse,
                r.secs_per_row,
                base.3 / r.secs_per_row
            );
        }
        println!(
            "\nbaseline (exhaustive/unweighted): cos={:.5} w_nmse={:.3e} h_nmse={:.3e} {:.4}s/row",
            base.0, base.1, base.2, base.3
        );
    }

    cells
}

// ---------------------------------------------------------------------------
// Rotation vs. Hessian: which fixture regime does each one win in?
// ---------------------------------------------------------------------------

/// Per-channel scales with a tunable dispersion: log-normal(0, `sigma`) bulk
/// plus a 10x outlier tail for channels above `outlier_z`.
fn channel_scales_sigma(k: usize, seed: u64, sigma: f64, outlier_z: f64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..k)
        .map(|_| {
            let base = (rng.normal() * sigma).exp();
            let outlier = if rng.normal() > outlier_z { 10.0 } else { 1.0 };
            (base * outlier) as f32
        })
        .collect()
}

/// **The control that stops the headline table being over-read.**
///
/// `trellis_search_headroom_table`'s rotation-off arm reports matmul cos 0.995
/// for the Hessian objective against 0.965 for the shipping recipe
/// (rotation + unweighted). Both arms share one fixture, one calibration draw,
/// one held-out evaluation draw and one dense reference, so those numbers ARE
/// directly comparable — which invites the reading "drop rotation, weight by
/// the Hessian instead". This sweep shows that reading is an artefact of a
/// single fixture's activation dispersion.
///
/// Sweeping the channel-energy dynamic range of `diag(H)` across three weight
/// fixtures shows the crossover sits far above anything an LLM produces:
/// at AWQ-like dispersion (~10³:1) rotation wins by 0.08 cos on FP4-lattice
/// weights, at 10⁴:1 it still wins by 0.03, and only past ~10⁷:1 — where the
/// weakest input channel carries twelve million times less energy than the
/// strongest — does the unrotated Hessian objective overtake it.
///
/// It also reproduces wave3-G independently (FP4-lattice, rotation on vs off,
/// unweighted: 0.957 vs 0.843 here; 0.963 vs 0.860 there) and shows rotation
/// buys nothing on already-Gaussian weights (0.9633 vs 0.9626) — i.e. rotation's
/// job is fixing heavy-tailed/lattice weight distributions, which is a different
/// job from the Hessian's.
///
/// `cargo test -p mistralrs-quant --lib probe_rotation_vs_hessian -- --ignored --nocapture`
#[test]
#[ignore = "evidence probe (~45s); run with --ignored --nocapture"]
fn probe_rotation_vs_hessian_sensitivity() {
    let (n, k, cb, eb) = (16usize, 2048usize, 256usize, 256usize);
    let fixtures: Vec<(&str, Vec<f32>)> = vec![
        (
            "gaussian   ",
            super::bake_quality_tests::gen_gaussian(n, k, 0.02, 1),
        ),
        (
            "student_t4 ",
            super::bake_quality_tests::gen_student_t(n, k, 0.02, 2),
        ),
        ("fp4_dequant", gen_fp4_dequant(n, k, 0.02, 0x0051_EA11)),
    ];
    let spreads = [
        ("flat       s=0.0 out=0%  ", 0.0, 999.0),
        ("mild       s=0.3 out=0%  ", 0.3, 999.0),
        ("awq-like   s=0.3 out=0.3%", 0.3, 2.75),
        ("moderate   s=0.6 out=1%  ", 0.6, 2.33),
        ("PR fixture s=0.9 out=3%  ", 0.9, 1.88),
    ];
    println!("\nfixture     | spread                    | h_range | r128+unw | r128+hes | r0+unw  | r0+hes");
    for (fname, w) in &fixtures {
        for (sname, sigma, oz) in spreads {
            let scales = channel_scales_sigma(k, 0xC0FF_EE01, sigma, oz);
            let x_cal = gen_activations(cb, &scales, 0x1234_5678);
            let x_eval = gen_activations(eb, &scales, 0x8765_4321);
            let h = hessian_diag(&x_cal, cb, k);
            let hmax = h.iter().cloned().fold(0f32, f32::max);
            let hmin = h.iter().cloned().fold(f32::INFINITY, f32::min);
            let ev = EvalSet {
                w,
                n,
                k,
                x_eval: &x_eval,
                batch: eb,
                h_diag: &h,
            };
            let mut cos = [0f64; 4];
            for (i, (rot, wt)) in [
                (ROT_BLOCK, false),
                (ROT_BLOCK, true),
                (NO_ROT, false),
                (NO_ROT, true),
            ]
            .into_iter()
            .enumerate()
            {
                let cfg = SearchCfg {
                    search: TrellisSearch::Exhaustive,
                    weighted: wt,
                };
                let (w_hat, s) = quantize_matrix(w, n, k, cfg, &h, rot);
                cos[i] = evaluate(&ev, &w_hat, s).matmul_cos;
            }
            println!(
                "{fname} | {sname} | {:>7.0} | {:>8.5} | {:>8.5} | {:>7.5} | {:>7.5}",
                (hmax / hmin) as f64,
                cos[0],
                cos[1],
                cos[2],
                cos[3]
            );
        }
        println!("---");
    }
}

// ---------------------------------------------------------------------------
// Bake-cost model for the CUDA trellis kernels
// ---------------------------------------------------------------------------
//
// wave16-AF REPLACED wave13-AF's model here, and the reason matters more than
// the numbers. The old model projected 42-85 s/layer for beam-256 from bytes
// moved and instructions issued. Hardware measured 238 s. It was not off by a
// parameter — it modelled a quantity the kernel is not bound by: it assumed
// that removing the memory wall (which the beam genuinely did, 257x less HBM,
// confirmed by mem=1% telemetry) would leave a kernel running near its issue
// roof. It does not. The kernel is latency-bound at 37.5% occupancy, and a
// bytes-and-instructions model cannot see that.
//
// A wrong model left in-tree is worse than no model, because it gets cited
// later as though it were measured. So this file no longer PREDICTS a wall
// time. It states what was measured, states the instruction counts that can be
// derived from the kernel source, and checks that the two are consistent —
// failing loudly if someone changes the kernel without re-measuring.

/// Measured on H200, session 6. GPU search only: the 241 +/- 1 s/layer marginal
/// figure minus the 3.3 s host INT4 unpack.
pub(crate) const MEASURED_BEAM256_LAYER_SECONDS: f64 = 238.0;
/// Measured on the same box class for the exhaustive prefix-grouped Viterbi.
pub(crate) const MEASURED_EXHAUSTIVE_LAYER_SECONDS: f64 = 510.0;
/// `cuobjdump -res-usage`, sm_90a: `REG:80 STACK:0 SHARED:38992 LOCAL:0`.
/// 256 threads x 80 reg = 20,480 reg/block; 65,536 / 20,480 = 3 blocks/SM;
/// 24 of 64 warps = 37.5% occupancy. Register-limited, not shared-limited
/// (3 x 38,992 B = 114 KiB of the 228 KiB/SM available).
pub(crate) const MEASURED_BEAM_REGISTERS_PER_THREAD: usize = 80;
pub(crate) const MEASURED_BEAM_BLOCKS_PER_SM: usize = 3;

/// H200 issue capacity: 132 SMs x 4 warp schedulers x 1.98 GHz (measured clock).
const H200_WARP_INST_PER_S: f64 = 132.0 * 4.0 * 1.98e9;
/// DeepSeek-V4-Flash: 284 B parameters / 44 layers / V=2 weights per symbol.
/// This is the (row x timestep) count, i.e. the number of block-timesteps.
const V4_ROW_TIMESTEPS_PER_LAYER: f64 = 284.0e9 / 44.0 / 2.0;
/// Threads per block in both quantize kernels.
const QTIP_BLOCK_WARPS: f64 = 8.0;

/// Instructions per thread per timestep, counted from the kernel sources.
///
/// The beam figure uses the radix pass count MEASURED by
/// [`probe_beam_kernel_cost_drivers`] (3.87 on production-shaped rows), not an
/// assumed one — the pass count is data-dependent and is the largest single
/// term.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepInstructions {
    pub group_reduce: f64,
    pub expand: f64,
    pub select: f64,
    pub compact: f64,
    pub trace: f64,
}

impl StepInstructions {
    pub fn total(&self) -> f64 {
        self.group_reduce + self.expand + self.select + self.compact + self.trace
    }
    /// Fraction of the step spent choosing among candidates rather than
    /// evaluating them.
    pub fn selection_fraction(&self) -> f64 {
        self.select / self.total()
    }
}

/// `qtip_beam.cu` as measured: 3.87 radix passes x 16 candidates x ~21
/// instructions dominates everything else.
pub(crate) fn beam_step_instructions(radix_passes: f64) -> StepInstructions {
    StepInstructions {
        group_reduce: 40.0,
        expand: 130.0,
        select: radix_passes * 16.0 * 21.0,
        compact: 190.0,
        trace: 4.0,
    }
}

/// `qtip_quantize.cu`'s prefix-grouped exhaustive DP: phase A is 4096 prefixes
/// x 16 predecessors over 256 threads, phase B is all 2^16 states.
pub(crate) fn exhaustive_step_instructions() -> StepInstructions {
    StepInstructions {
        group_reduce: 0.0,
        expand: 0.0,
        // phase A: 256 iterations x (load + compare + select)
        select: 256.0 * 3.0,
        // phase B: 256 states x (2 LUT loads, 2 sub, 2 mul, add, shared load,
        // add, global store)
        compact: 256.0 * 10.0,
        trace: 16.0,
    }
}

/// Warp-instructions one layer of V4-Flash costs at the given per-thread count.
fn layer_warp_instructions(per_thread: f64) -> f64 {
    V4_ROW_TIMESTEPS_PER_LAYER * QTIP_BLOCK_WARPS * per_thread
}

/// Issue efficiency implied by an instruction count and a MEASURED wall time:
/// the fraction of the machine's warp-issue slots the kernel actually used.
///
/// This is the number the old model had no way to express, and it is where the
/// beam's win went: the beam cut instructions 1.9x and efficiency barely moved.
pub(crate) fn implied_issue_efficiency(per_thread: f64, layer_seconds: f64) -> f64 {
    layer_warp_instructions(per_thread) / (H200_WARP_INST_PER_S * layer_seconds)
}

/// Static shared memory of `qtip_quantize_rows_beam_kernel`, mirroring the
/// declarations in the kernel. `cuobjdump` reports 38,992 B for the pre-wave16
/// kernel; the extra ~1 KiB over this arithmetic is compiler/driver overhead.
pub(crate) fn beam_kernel_smem_bytes(max_beam: usize) -> usize {
    let group_table = 4096 * 8; // u64 atomicMin slot per 2^(L-K) prefix
    let beam = max_beam * (4 + 2 + 2); // cost f32 + state u16 + parent u16
    let groups = max_beam * (2 + 4 + 2); // g u16 + cost f32 + parent u16
    let hist = 256 * 4;
    let scratch = 2 * 8 * 4 + 8 * 8; // double-buffered warp totals + scalars
    group_table + beam + groups + hist + scratch
}

/// Blocks per SM permitted by a register budget on sm_90 (65,536 registers and
/// 228 KiB of shared memory per SM, 256-register-per-warp allocation
/// granularity).
pub(crate) fn blocks_per_sm(regs_per_thread: usize, smem_bytes: usize) -> usize {
    let per_warp = (32 * regs_per_thread).div_ceil(256) * 256;
    let per_block = per_warp * 8;
    let by_regs = 65536 / per_block.max(1);
    let by_smem = 233_472 / smem_bytes.max(1);
    by_regs.min(by_smem).min(8)
}

/// Print the measured picture and the derived breakdown side by side.
///
/// `cargo test -p mistralrs-quant --lib beam_kernel_cost_breakdown -- --nocapture`
#[test]
fn beam_kernel_cost_breakdown() {
    let beam = beam_step_instructions(3.87);
    let exh = exhaustive_step_instructions();
    let e_beam = implied_issue_efficiency(beam.total(), MEASURED_BEAM256_LAYER_SECONDS);
    let e_exh = implied_issue_efficiency(exh.total(), MEASURED_EXHAUSTIVE_LAYER_SECONDS);

    println!("\n### MEASURED (H200, session 6) — not a projection");
    println!(
        "  exhaustive {:.0} s/layer | beam-256 {:.0} s/layer | speedup {:.2}x",
        MEASURED_EXHAUSTIVE_LAYER_SECONDS,
        MEASURED_BEAM256_LAYER_SECONDS,
        MEASURED_EXHAUSTIVE_LAYER_SECONDS / MEASURED_BEAM256_LAYER_SECONDS
    );
    println!(
        "  beam kernel: REG:{} LOCAL:0 SHARED:38992 -> {} blocks/SM, {:.1}% occupancy \
         (register-limited)",
        MEASURED_BEAM_REGISTERS_PER_THREAD,
        MEASURED_BEAM_BLOCKS_PER_SM,
        MEASURED_BEAM_BLOCKS_PER_SM as f64 * 8.0 / 64.0 * 100.0
    );
    println!("  telemetry: sm=100%, mem=1%, 261 W of 700 W (37% of TDP) at 1980 MHz");

    println!("\n### DERIVED from kernel source + the measured radix pass count");
    println!(
        "  instructions/thread/timestep: exhaustive {:.0} | beam {:.0} (ratio {:.2}x)",
        exh.total(),
        beam.total(),
        exh.total() / beam.total()
    );
    println!(
        "  beam breakdown: select {:.0} ({:.0}%), compact {:.0}, expand {:.0}, \
         group {:.0}, trace {:.0}",
        beam.select,
        beam.selection_fraction() * 100.0,
        beam.compact,
        beam.expand,
        beam.group_reduce,
        beam.trace
    );
    println!(
        "  implied issue efficiency: exhaustive {:.1}% | beam {:.1}% (ratio {:.2}x)",
        e_exh * 100.0,
        e_beam * 100.0,
        e_beam / e_exh
    );
    println!(
        "  => predicted speedup {:.2}x x {:.2}x = {:.2}x vs measured {:.2}x",
        exh.total() / beam.total(),
        e_beam / e_exh,
        (exh.total() / beam.total()) * (e_beam / e_exh),
        MEASURED_EXHAUSTIVE_LAYER_SECONDS / MEASURED_BEAM256_LAYER_SECONDS
    );
    println!(
        "\n  NOTE: 37.5% occupancy against 37% of TDP is an independent check on the \
         latency-bound reading."
    );
}

/// The model is descriptive, so its job is to stay consistent with the
/// measurement. If a kernel change breaks these, the numbers in
/// `wave16-AF-beam-perf.md` are stale and must be re-measured, not re-derived.
#[test]
fn cost_model_matches_the_measurement() {
    let beam = beam_step_instructions(3.87);
    let exh = exhaustive_step_instructions();

    // Selection, not state evaluation, is what the beam kernel spends its time
    // on. This is the finding the whole analysis rests on.
    assert!(
        beam.selection_fraction() > 0.70,
        "selection is {:.0}% of the beam step; the analysis assumes it dominates",
        beam.selection_fraction() * 100.0
    );

    // Instruction ratio x efficiency ratio must reproduce the measured speedup.
    let e_beam = implied_issue_efficiency(beam.total(), MEASURED_BEAM256_LAYER_SECONDS);
    let e_exh = implied_issue_efficiency(exh.total(), MEASURED_EXHAUSTIVE_LAYER_SECONDS);
    let predicted = (exh.total() / beam.total()) * (e_beam / e_exh);
    let measured = MEASURED_EXHAUSTIVE_LAYER_SECONDS / MEASURED_BEAM256_LAYER_SECONDS;
    assert!(
        (predicted - measured).abs() / measured < 0.05,
        "model predicts {predicted:.3}x but {measured:.3}x was measured"
    );

    // Both kernels run far below the issue roof: that is the shared diagnosis.
    for (name, eff) in [("exhaustive", e_exh), ("beam", e_beam)] {
        assert!(
            eff < 0.25,
            "{name} issue efficiency {eff:.3} — if a kernel now runs near its roof, \
             the latency-bound analysis no longer applies"
        );
    }

    // Occupancy arithmetic, checked against what cuobjdump actually reported.
    assert_eq!(
        blocks_per_sm(MEASURED_BEAM_REGISTERS_PER_THREAD, 38_992),
        MEASURED_BEAM_BLOCKS_PER_SM,
        "occupancy arithmetic disagrees with the measured cuobjdump result"
    );
    // The wave16 fix targets 4 blocks/SM, which needs <= 64 registers/thread.
    assert_eq!(blocks_per_sm(64, beam_kernel_smem_bytes(256)), 4);
    // Shared memory is not what limits either configuration.
    assert!(beam_kernel_smem_bytes(256) * 4 <= 233_472);
}
// ---------------------------------------------------------------------------
// CUDA beam kernel: measured cost drivers (wave16-AF)
// ---------------------------------------------------------------------------
//
// The hardware run measured 238 s/layer for the beam-256 GPU search against
// 510 s/layer for the exhaustive kernel — 2.1x, where wave13-AF projected
// 6-12x. GPU telemetry during the search read sm=100%, mem=1%, 261 W of 700 W
// at max clocks: resident everywhere, working nowhere. Memory was NOT the
// limit (mem=1% confirms the 257x HBM cut landed), so the cost has to be in
// how the kernel spends issue slots.
//
// Two quantities set that, and neither is visible in the CUDA source because
// both are data-dependent:
//
//   * `ng`, the number of distinct prefix groups in the beam. The kernel maps
//     ONE THREAD PER GROUP (`active = tid < ng` in qtip_beam.cu), so `ng` is
//     literally the number of the block's 256 threads that have work in the
//     expansion phase. If `ng` is small, most of the block idles at a barrier.
//   * the number of 8-bit radix-select passes. Each pass costs 6 block
//     barriers and 16 key rebuilds per thread, so the pass count multiplies
//     the most expensive phase of the step.
//
// Both are properties of the beam search, not of CUDA, so they are measurable
// here. This probe replaces the guesswork in the wave13-AF projection with
// arithmetic over observed values.

/// `cargo test -p mistralrs-quant --lib probe_beam_kernel_cost_drivers -- --ignored --nocapture`
#[test]
#[ignore = "evidence probe (~1-2 min); run with --ignored --nocapture"]
fn probe_beam_kernel_cost_drivers() {
    use super::viterbi::beam_kernel_stats;

    let lut = gaussian_lut();
    // V4-Flash MoE expert shapes: gate/up read hidden_size = 7168, down reads
    // moe_intermediate_size = 2048. Rows are the parallel axis; K sets the
    // trellis length, which is what this probe cares about.
    let shapes: [(&str, usize); 2] = [("gate/up  k=7168", 7168), ("down     k=2048", 2048)];
    let n_rows = 4;

    println!(
        "\n=== CUDA beam kernel cost drivers, measured on the CPU beam \
         (FP4-lattice weights, hadamard-128, W=256) ==="
    );
    println!(
        "{:<16} | {:>7} | {:>9} | {:>7} | {:>7} | {:>10} | ng histogram (1,16,32,64,96,128,160,208+)",
        "shape", "steps", "mean ng", "min ng", "max ng", "mean pass"
    );

    for (label, k) in shapes {
        let w = gen_fp4_dequant(n_rows, k, 0.02, 0x00BE_A115);
        let signs = generate_signs(QTIP_ROTATION_SEED, k);
        let mut agg = super::viterbi::BeamKernelStats {
            group_min: usize::MAX,
            ..Default::default()
        };
        for row in 0..n_rows {
            let mut rot = w[row * k..(row + 1) * k].to_vec();
            apply_block_rotation(&mut rot, &signs, ROT_BLOCK);
            let max_abs = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 3.0 };
            let inv = 1.0 / scale;
            let target: Vec<f32> = rot.iter().map(|&v| v * inv).collect();

            let s = beam_kernel_stats(&target, &lut, 256);
            agg.steps += s.steps;
            agg.group_sum += s.group_sum;
            agg.group_min = agg.group_min.min(s.group_min);
            agg.group_max = agg.group_max.max(s.group_max);
            agg.pass_sum += s.pass_sum;
            agg.unpruned_steps += s.unpruned_steps;
            for i in 0..8 {
                agg.group_hist[i] += s.group_hist[i];
            }
            for i in 0..7 {
                agg.pass_hist[i] += s.pass_hist[i];
            }
            agg.wasted_leading_digits_sum += s.wasted_leading_digits_sum;
            agg.top_differing_bit_sum += s.top_differing_bit_sum;
            agg.skip_pass_sum += s.skip_pass_sum;
        }
        println!(
            "{label:<16} | {:>7} | {:>9.1} | {:>7} | {:>7} | {:>10.2} | {:?}",
            agg.steps,
            agg.mean_groups(),
            agg.group_min,
            agg.group_max,
            agg.mean_passes(),
            agg.group_hist
        );
        println!(
            "{:<16}   radix pass histogram (index = passes): {:?}, unpruned steps {}",
            "", agg.pass_hist, agg.unpruned_steps
        );

        // The derived per-step kernel costs, stated in the units the analysis
        // needs: active threads out of 256, and block barriers per timestep
        // (10 fixed + 6 per radix pass, counted from qtip_beam.cu).
        let occupancy = agg.mean_groups() / 256.0;
        let barriers = 10.0 + 6.0 * agg.mean_passes();
        println!(
            "{:<16}   => expansion-phase thread occupancy {:.1}% of the block; \
             {:.0} block barriers per timestep",
            "",
            occupancy * 100.0,
            barriers
        );
        println!(
            "{:<16}   => mean {:.2} of those passes are WASTED (all candidates in one bin); \
             highest differing key bit {:.1} of 48",
            "",
            agg.mean_wasted_digits(),
            agg.mean_top_differing_bit()
        );
        println!(
            "{:<16}   => starting the scan at the first differing digit: {:.2} passes \
             (vs {:.2}), i.e. {:.0} barriers instead of {:.0}",
            "",
            agg.mean_skip_passes(),
            agg.mean_passes(),
            10.0 + 6.0 * agg.mean_skip_passes(),
            barriers
        );
    }
}

/// **The parity argument for the wave16-AF kernel rewrite, checked on CPU.**
///
/// `qtip_beam.cu` used to radix-select over the 48-bit `(cost, state)`
/// composite key; it now scans the 32-bit cost key and falls back into the
/// state bits only when costs genuinely tie. That is a change to *which bits
/// are examined in what order*, and it must not be a change to *which
/// candidates are selected* — byte-identity with the CPU beam is the whole
/// reason the artifact is trustworthy.
///
/// This replays both scans on every real candidate set produced by beam
/// searching production-shaped rows, and asserts they admit the identical set.
/// It is a much stronger check than comparing thresholds: two different
/// thresholds can still select the same set, and the set is what matters.
///
/// This runs on any machine, so the rewrite is guarded before it ever reaches
/// a GPU. `cuda_beam_matches_cpu_beam_bit_for_bit` remains the hardware gate.
#[test]
fn wave16_split_key_selection_matches_composite_key() {
    use super::viterbi::{for_each_candidate_set, threshold_composite_key, threshold_split_key};

    let lut = gaussian_lut();
    let mut checked = 0usize;
    let mut tie_steps = 0usize;

    // Two shapes, and a width that prunes hard, so the selection actually runs.
    for (k, seed, width) in [
        (2048usize, 0x00BE_A115u64, 256usize),
        (1024, 0x0051_EA11, 64),
    ] {
        let w = gen_fp4_dequant(2, k, 0.02, seed);
        let signs = generate_signs(QTIP_ROTATION_SEED, k);
        for row in 0..2 {
            let mut rot = w[row * k..(row + 1) * k].to_vec();
            apply_block_rotation(&mut rot, &signs, ROT_BLOCK);
            let max_abs = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 3.0 };
            let inv = 1.0 / scale;
            let target: Vec<f32> = rot.iter().map(|&v| v * inv).collect();

            for_each_candidate_set(&target, &lut, width, |keys, wdt| {
                let t_old = threshold_composite_key(keys, wdt);
                let t_new = threshold_split_key(keys, wdt);

                let sel_old = keys.iter().filter(|&&x| x <= t_old).count();
                let sel_new = keys.iter().filter(|&&x| x <= t_new).count();
                assert_eq!(
                    sel_old, wdt,
                    "composite scan must admit exactly the beam width"
                );
                assert_eq!(
                    sel_new, wdt,
                    "split-key scan admitted {sel_new} candidates, not {wdt}"
                );
                // Same COUNT is not enough — it must be the same SET.
                for &key in keys {
                    assert_eq!(
                        key <= t_old,
                        key <= t_new,
                        "split-key scan disagrees on candidate {key:#x} \
                         (composite threshold {t_old:#x}, split {t_new:#x})"
                    );
                }
                // Track how often the state bits are actually needed.
                let cost_hi = (t_old >> 16) as u32;
                if keys
                    .iter()
                    .filter(|&&x| (x >> 16) as u32 == cost_hi)
                    .count()
                    > 1
                {
                    tie_steps += 1;
                }
                checked += 1;
            });
        }
    }

    assert!(
        checked > 3000,
        "only {checked} candidate sets checked — fixture too small to be evidence"
    );
    println!(
        "wave16 split-key selection == composite-key selection on {checked} real candidate \
         sets ({tie_steps} of them needed the state-bit tie-break)"
    );
}
