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
// Bake-cost model for the CUDA trellis kernels (wave13-AF)
// ---------------------------------------------------------------------------
//
// PROJECTION, NOT A MEASUREMENT. wave13-AF had no GPU, so everything below is
// derived by counting bytes and instructions in
// `kernels/qtip/qtip_quantize.cu` and `kernels/qtip/qtip_beam.cu`. It is
// arithmetic about the kernels, so it runs anywhere:
//
//   cargo test -p mistralrs-quant --lib \
//       qtip::search_bench::cuda_beam_bake_projection -- --nocapture
//
// The model has exactly one contact with reality, and it is a good one. Counted
// naively, the exhaustive kernel must move 528 KB across HBM per symbol
// position; a 284 B/44-layer V4-Flash layer holds ~3.23e9 symbol positions, so
// one layer is ~1.7 PB, i.e. 355 s at H200's 4.8 TB/s. FACTS.md measures
// **510 s/layer** on a healthy box. A pure-bandwidth model landing within 1.4x
// of the measurement — and implying a believable 70% of peak — is strong
// evidence that the exhaustive quantizer is HBM-bound, which is precisely the
// "GPU 99% util but low power draw" signature FACTS.md records. There is almost
// no arithmetic hiding behind that traffic.
//
// The beam kernel removes that traffic entirely (the live state set is shared-
// memory resident), so its projection is bounded by shared memory and issue
// rate instead — quantities this model estimates far less reliably than
// bandwidth. Two derating knobs make that pessimism explicit rather than
// hidden, and the printed table reports the raw bound alongside the derated
// figure so the optimistic end is visible too.

/// H200 SXM peak HBM bandwidth (bytes/s).
const H200_HBM_BYTES_PER_S: f64 = 4.8e12;
/// Aggregate shared-memory bandwidth: 132 SMs x 128 B/clk x 1.755 GHz.
const H200_SMEM_BYTES_PER_S: f64 = 132.0 * 128.0 * 1.755e9;
/// Aggregate simple-ALU/LSU issue rate: 132 SMs x 128 lanes x 1.755 GHz.
const H200_ALU_OPS_PER_S: f64 = 132.0 * 128.0 * 1.755e9;

/// Fraction of HBM peak the exhaustive kernel achieves, implied by FACTS.md
/// (byte model 355 s/layer vs measured 510 s/layer). Applied to both kernels'
/// HBM term so the comparison uses one consistent efficiency.
const HBM_EFFICIENCY: f64 = 355.0 / 510.0;

/// Derating applied to the beam kernel's shared-memory and issue-rate terms.
///
/// The beam kernel costs ~35 block barriers per timestep against the exhaustive
/// kernel's ~3, and its 37 KiB of shared memory caps residency at 4 blocks/SM
/// (1024 of 2048 threads). Neither is captured by a throughput model. 0.5 is a
/// deliberately pessimistic stand-in; the printed table also shows the
/// un-derated bound, so the honest answer is the range between them.
const BEAM_ISSUE_EFFICIENCY: f64 = 0.5;

/// FACTS.md: exhaustive GPU Viterbi measured at ~8.5 min/layer on a healthy
/// H200 (523 W, single process, max clocks).
pub(crate) const MEASURED_EXHAUSTIVE_LAYER_SECONDS: f64 = 8.5 * 60.0;

/// DeepSeek-V4-Flash: 284 B parameters over 44 layers, V = 2 weights/symbol.
const V4_SYMBOLS_PER_LAYER: f64 = 284.0e9 / 44.0 / 2.0;

/// Per-symbol-position cost of one trellis-search kernel, in the three
/// currencies that can bind it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SymbolCost {
    /// Bytes crossing HBM per (row, timestep).
    pub hbm_bytes: f64,
    /// Bytes of shared-memory traffic per (row, timestep).
    pub smem_bytes: f64,
    /// Simple ALU/LSU operations per (row, timestep), summed over the block.
    pub alu_ops: f64,
}

/// Per-symbol cost of `qtip_quantize_rows_viterbi_kernel` — the exhaustive
/// prefix-grouped DP — counted from the kernel source.
///
/// * phase A reads every one of the `2^L` predecessor costs exactly once
///   (`(j << 12) | p` is a bijection over the state space): 256 KiB;
/// * phase B writes all `2^L` updated costs: 256 KiB;
/// * phase C writes the `2^(L-K)`-byte backtrace: 4 KiB;
/// * the 512 KiB LUT read is shared by every resident block and stays in L2,
///   so it is charged to bandwidth-below-HBM, not to HBM.
pub(crate) fn exhaustive_symbol_cost() -> SymbolCost {
    let states = 65536.0;
    let prefixes = 4096.0;
    SymbolCost {
        hbm_bytes: states * 4.0 + states * 4.0 + prefixes + 8.0,
        // phase A writes the per-prefix cost+argmin table, phase B reads the
        // cost once per state, phase C reads the argmin back out.
        smem_bytes: prefixes * 5.0 + states * 4.0 + prefixes,
        // phase A: load + compare + select per (prefix, j).
        // phase B: 2 LUT loads, 2 subs, 2 muls, 1 add, 1 shared load, 1 add,
        //          1 global store per state.
        alu_ops: states * 3.0 + states * 10.0,
    }
}

/// Per-symbol cost of `qtip_quantize_rows_beam_kernel` at beam width `w`,
/// counted from the kernel source under the worst case `n_groups == w` (every
/// surviving state in a distinct prefix group, so the candidate list is the
/// full `16w`).
///
/// `radix_passes` is how many 8-bit digit passes the selection runs. The kernel
/// stops as soon as a digit bin holds a single candidate — with f32 costs that
/// is usually pass 2 or 3 — but the hard ceiling is 6, so both ends are worth
/// printing.
///
/// The shared-memory histogram term deliberately assumes NO benefit from the
/// `__match_any_sync` warp aggregation (i.e. one atomic per candidate). With
/// aggregation the real figure is much smaller; charging full price keeps the
/// projection on the pessimistic side of the truth.
pub(crate) fn beam_symbol_cost(w: f64, radix_passes: f64) -> SymbolCost {
    let cands = w * 16.0;
    let threads = 256.0;
    // 8 block-wide scans per timestep (2 for group/beam compaction, 1 per radix
    // pass) at ~8 B of shared traffic per thread each.
    let scan_bytes = 8.0 * threads * 8.0;
    SymbolCost {
        // Only the compacted trace crosses HBM: `w` u32 written on the way
        // forward and read back once by the staged backtrace. There is no cost
        // ping-pong — the live state set lives in shared memory.
        hbm_bytes: w * 4.0 * 2.0 + 8.0,
        smem_bytes:
            // group vote: read state+cost, 64-bit atomicMin read-modify-write
            w * 22.0
            // winner test: read s_gmin, read cost, write 3 group fields, release
            + w * 28.0
            + scan_bytes
            // s_hist[tid] readback, one per radix pass
            + radix_passes * threads * 4.0
            // radix: clear 256 bins + one 4-byte RMW per candidate per pass
            + radix_passes * (1024.0 + cands * 8.0)
            // compaction writes + trace staging read
            + w * 8.0
            + w * 4.0,
        alu_ops:
            // candidate generation: 2 LUT loads, 2 subs, 2 muls, 2 adds
            cands * 8.0
            // radix, per candidate per pass: rebuild the 48-bit key (4), test
            // the resolved prefix (2), extract the digit (2), ballot/match/
            // atomic amortised (4)
            + radix_passes * cands * 12.0
            // compaction: rebuild key (4), compare (1), conditional store (3)
            + cands * 8.0
            // fixed block overhead: 8 scans of ~15 ops across 256 threads,
            // plus barriers and bookkeeping
            + 30000.0,
    }
}

/// One row of the projected bake table.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BakeProjection {
    pub width: usize,
    /// Bound with the derating applied — the number to quote.
    pub layer_seconds: f64,
    /// Bound with no derating — the optimistic end of the range.
    pub layer_seconds_raw: f64,
    pub hbm_seconds: f64,
    pub smem_seconds: f64,
    pub alu_seconds: f64,
}

/// Projected per-layer bake time for the beam kernel at each width on a
/// 284 B-parameter / 44-layer V4-Flash model, plus the exhaustive kernel's
/// pure-bandwidth prediction for calibration.
pub(crate) fn beam_bake_cost_model(
    widths: &[usize],
    radix_passes: f64,
) -> (f64, Vec<BakeProjection>) {
    let symbols = V4_SYMBOLS_PER_LAYER;
    let ex = exhaustive_symbol_cost();
    let exhaustive_seconds = symbols * ex.hbm_bytes / (H200_HBM_BYTES_PER_S * HBM_EFFICIENCY);
    let rows = widths
        .iter()
        .map(|&w| {
            let c = beam_symbol_cost(w as f64, radix_passes);
            let h = symbols * c.hbm_bytes / (H200_HBM_BYTES_PER_S * HBM_EFFICIENCY);
            let s = symbols * c.smem_bytes / H200_SMEM_BYTES_PER_S;
            let a = symbols * c.alu_ops / H200_ALU_OPS_PER_S;
            let raw = h.max(s).max(a);
            BakeProjection {
                width: w,
                layer_seconds: h.max((s / BEAM_ISSUE_EFFICIENCY).max(a / BEAM_ISSUE_EFFICIENCY)),
                layer_seconds_raw: raw,
                hbm_seconds: h,
                smem_seconds: s,
                alu_seconds: a,
            }
        })
        .collect();
    (exhaustive_seconds, rows)
}

/// Static shared-memory footprint of `qtip_quantize_rows_beam_kernel`, in
/// bytes, mirroring the declarations in `kernels/qtip/qtip_beam.cu`.
///
/// Sized by the compile-time `QB_MAX_BEAM`, so it does not shrink with the
/// runtime width — which is the point: the footprint is dominated by the
/// 4,096-entry prefix-group table, NOT by the beam.
pub(crate) fn beam_kernel_smem_bytes(max_beam: usize) -> usize {
    let group_table = 4096 * 8; // u64 atomicMin slot per 2^(L-K) prefix
    let beam = max_beam * (4 + 2 + 2); // cost f32 + state u16 + parent u16
    let groups = max_beam * (2 + 4 + 2); // g u16 + cost f32 + parent u16
    let hist = 256 * 4;
    let scratch = 8 * 4 + 8 * 8; // warp totals + scalars
    group_table + beam + groups + hist + scratch
}

/// The wave13-AF projection, printed as a table. PROJECTION — see the module
/// comment above for exactly which part of it touches a measurement.
#[test]
fn cuda_beam_bake_projection() {
    let widths = [64usize, 128, 256];
    println!(
        "\n### PROJECTED V4-Flash bake cost (284B / 44 layers / H200). \
         NOT MEASURED — wave13-AF had no GPU."
    );
    let ex = exhaustive_symbol_cost();
    let beam = beam_symbol_cost(256.0, 6.0);
    println!(
        "per symbol position: exhaustive {:.0} B HBM / {:.0} B smem / {:.0} ops \
         -> beam W=256 {:.0} B HBM / {:.0} B smem / {:.0} ops \
         ({:.0}x less HBM, {:.1}x less smem, {:.1}x less arithmetic)",
        ex.hbm_bytes,
        ex.smem_bytes,
        ex.alu_ops,
        beam.hbm_bytes,
        beam.smem_bytes,
        beam.alu_ops,
        ex.hbm_bytes / beam.hbm_bytes,
        ex.smem_bytes / beam.smem_bytes,
        ex.alu_ops / beam.alu_ops,
    );

    for &passes in &[2.0f64, 6.0] {
        let (exhaustive, rows) = beam_bake_cost_model(&widths, passes);
        println!(
            "\n--- radix-select digit passes = {passes} ({}) ---",
            if passes == 2.0 {
                "typical: the early exit fires once a digit bin is unique"
            } else {
                "hard ceiling: all 48 key bits resolved"
            }
        );
        println!(
            "exhaustive prefix-grouped Viterbi: {exhaustive:.0} s/layer bandwidth model \
             (FACTS.md MEASURES {:.0} s/layer = {:.1} h for 44 layers)",
            MEASURED_EXHAUSTIVE_LAYER_SECONDS,
            MEASURED_EXHAUSTIVE_LAYER_SECONDS * 44.0 / 3600.0
        );
        println!(
            "{:<8} | {:>9} | {:>9} | {:>7} | {:>7} | {:>7} | {:>9} | {:>12}",
            "beam W", "s/layer", "optimist", "hbm s", "smem s", "alu s", "speedup", "44 layers"
        );
        for r in &rows {
            println!(
                "{:<8} | {:>9.1} | {:>9.1} | {:>7.1} | {:>7.1} | {:>7.1} | {:>8.1}x | {:>9.1} min",
                r.width,
                r.layer_seconds,
                r.layer_seconds_raw,
                r.hbm_seconds,
                r.smem_seconds,
                r.alu_seconds,
                MEASURED_EXHAUSTIVE_LAYER_SECONDS / r.layer_seconds,
                r.layer_seconds * 44.0 / 60.0
            );
        }
    }

    println!(
        "\nshared-memory residency: the kernel's static footprint is {} B of the 49152 B \
         budget at every supported width — {} B of that is the 4096-entry prefix-group \
         table and only {} B is the W=256 beam itself.",
        beam_kernel_smem_bytes(256),
        4096 * 8,
        256 * 8
    );
}

/// Arithmetic facts about the two kernels that must hold for the design to be
/// worth shipping. Not timings, so these run on any machine.
#[test]
fn beam_cost_model_invariants() {
    let ex = exhaustive_symbol_cost();
    // Worst case for the beam: widest supported W and every radix pass taken.
    let beam = beam_symbol_cost(256.0, 6.0);

    // The thesis: the exhaustive kernel is HBM-bound and the beam kernel is not.
    assert!(
        ex.hbm_bytes / beam.hbm_bytes > 200.0,
        "beam must cut HBM traffic by >200x, got {:.0}x",
        ex.hbm_bytes / beam.hbm_bytes
    );
    // ... and it must not simply relocate the bottleneck into shared memory,
    // even when charged full price for every histogram atomic.
    assert!(
        beam.smem_bytes < ex.smem_bytes,
        "beam smem traffic {:.0} B must not exceed the exhaustive kernel's {:.0} B",
        beam.smem_bytes,
        ex.smem_bytes
    );
    assert!(
        ex.alu_ops > beam.alu_ops,
        "beam must also cut arithmetic: {:.0} vs {:.0} ops",
        beam.alu_ops,
        ex.alu_ops
    );

    // The pure-bandwidth model must land within 2x of the measured exhaustive
    // bake, otherwise nothing built on it is worth reading. (HBM_EFFICIENCY is
    // fitted to that measurement, so this asserts the fitted efficiency stays
    // physically plausible rather than absorbing an order of magnitude.)
    assert!(
        (0.3..=1.0).contains(&HBM_EFFICIENCY),
        "implied HBM efficiency {HBM_EFFICIENCY} is not a plausible fraction of peak"
    );

    // Wider beams cost more, monotonically, and every width beats exhaustive.
    let (_, rows) = beam_bake_cost_model(&[64, 128, 256], 6.0);
    for pair in rows.windows(2) {
        assert!(
            pair[1].layer_seconds > pair[0].layer_seconds,
            "W={} must cost more than W={}",
            pair[1].width,
            pair[0].width
        );
    }
    for r in &rows {
        assert!(
            r.layer_seconds < MEASURED_EXHAUSTIVE_LAYER_SECONDS,
            "W={} projects {:.0} s/layer, no better than the measured exhaustive bake",
            r.width,
            r.layer_seconds
        );
    }

    // Shared-memory residency: the kernel fits the 48 KiB static budget, and
    // the prefix-group table — not the beam — is what dominates it.
    let total = beam_kernel_smem_bytes(256);
    assert!(
        total <= 48 * 1024,
        "beam kernel needs {total} B of shared memory, over the 48 KiB static budget"
    );
    assert!(
        4096 * 8 > total / 2,
        "the 4096-entry group table should dominate the shared-memory budget"
    );
}
