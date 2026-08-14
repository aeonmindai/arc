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
//! ## CUDA kernel plan (not in this PR)
//!
//! `kernels/qtip/qtip_quantize.cu::qtip_quantize_rows_viterbi_kernel` is the
//! prefix-grouped exhaustive DP: phase A does one 16-way reduction per prefix
//! (4,096 prefixes in shared memory), phase B adds the local error across all
//! 65,536 states, phase C writes a 4,096-byte backtrace per timestep. Porting
//! axis A means replacing the dense `prev[]`/`curr[]` arrays with a sorted beam:
//!
//! 1. Keep the beam in shared memory as `(cost f32, state u16, parent u16)` —
//!    `W = 256` is 2 KiB, so a whole row's beam is register/SMEM resident and
//!    the 620 MB global backtrace scratch (`num_symbols = 9472`) collapses to
//!    ~9.7 MB, removing the `VITERBI_MAX_SCRATCH_BYTES` row-batching entirely.
//! 2. Expansion is `W × 16` candidates — one warp per 32 candidates. Dedup by
//!    successor state needs an atomic min; the cheapest correct form is a
//!    64-bit `atomicMin` on `(cost_as_ordered_u32 << 32) | parent`, keyed by a
//!    `2^L`-entry SMEM-unfriendly table, so instead sort candidates by state
//!    (CUB `BlockRadixSort`, 4,096 keys) and segment-reduce — that also yields
//!    the ascending-state order the CPU version relies on for exact tie-breaks.
//! 3. Selection of the best `W` is a second `BlockRadixSort` on cost, or
//!    `cub::BlockRadixSort` once on a packed `(cost, state)` key.
//! 4. Axis B is a one-line kernel change: the branch metric reads a per-column
//!    weight from a `[K_in]` array in constant/`__ldg` memory.
//!
//! Until that lands, `quantize_with_options_concrete_calibrated` refuses to run
//! on CUDA when either flag is set rather than silently baking the exhaustive
//! unweighted objective under a beam/Hessian label.

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
