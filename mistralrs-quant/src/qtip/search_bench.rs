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
    quantize_matrix_cb(w, n, k, cfg, h_diag, rot_block, &gaussian_lut())
}

/// As [`quantize_matrix`], with the codebook supplied by the caller.
///
/// Both the search and the reconstruction read the SAME table, so swapping it
/// swaps the format coherently — which is what makes a computed codebook
/// measurable here at all (wave19-AP). The row scale still comes from
/// `max|row| / 3`, so any codebook handed in must already be normalised to the
/// spread that policy assumes; see [`mcg_codebook_v2`].
fn quantize_matrix_cb(
    w: &[f32],
    n: usize,
    k: usize,
    cfg: SearchCfg,
    h_diag: &[f32],
    rot_block: usize,
    lut: &[f32],
) -> (Vec<f32>, f64) {
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

        let syms = quantize_row(&target, lut, cfg.search, weights.as_deref());
        let packed = pack_symbols(&syms);

        let mut recon: Vec<f32> = decode_packed(&packed, num_symbols, lut)
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
// wave19-AP — the computed codebook at K=4 / V=2: what does it cost in quality?
// ---------------------------------------------------------------------------
//
// MEASURED SPEED PRIZE (A30 sm_80, wave19-AP): replacing the 512 KiB Gaussian
// LUT with QTIP's computed codebook is worth **1.81x on the beam kernel**
// (998.0 -> 551.7 ms at 1344 rows x k_in=7168), i.e. 201.5 -> 118.5 s/layer on
// H200 and a 43-layer bake from ~2.9 h to ~85 min. That number is a SPEED
// number. This is the gate it has to pass.
//
// The construction is the one already shipping on the `qtip2b` rung
// (`bitshift.rs::mcg_codeword`, exllamav3 PR #26's spectrally-optimised
// multiplier): one wrapping multiply, one masked XOR, and the two fp16 halves
// of the 32-bit result. At K=2/V=1 the codeword is `hi + lo` — a SUM. Going to
// V=2 forces a choice that does not exist at V=1, and the choice is the whole
// experiment:
//
//   Split — take `hi` and `lo` as the two reproduction values.
//           4 instructions for 2 weights. This is what the CUDA measurement
//           above actually ran. But a masked fp16 half has exponent bits
//           restricted to 12..15, so its magnitude lives in [0.125, 2.0) and
//           **cannot be near zero** — the codebook has a hole exactly where a
//           Gaussian weight distribution has most of its mass.
//   Sum2  — two chained multiplies, each folded to `hi + lo`.
//           10 instructions for 2 weights, and the summed distribution is the
//           one qtip2b already ships and measures (sigma 1.2064).
//   Pair  — `hi + lo` of states `2s` and `2s+1`. Same cost as Sum2, different
//           correlation between the two values of one state.
//
// Every codebook here is normalised to unit sigma, because the row scale is
// `max|row| / 3` and that policy assumes a unit-spread codebook. qtip2b does the
// same thing with a constant (`QTIP2B_SCALE_DIVISOR = 3.0 x 1.2064`); folding it
// into the table instead keeps the comparison to the Gaussian LUT exact and
// costs nothing on GPU (it folds into the target scale). Skipping it would
// measure a scale mismatch and call it a codebook result.

/// Which V=2 codeword pair to derive from one MCG product.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum McgV2 {
    /// `(f16(hi), f16(lo))` — 4 instructions, magnitude hole in (-0.125, 0.125).
    Split,
    /// `(hi+lo of x, hi+lo of x*mult)` — 10 instructions, qtip2b's distribution.
    Sum2,
    /// `(codeword(2s), codeword(2s+1))` — 10 instructions.
    Pair,
}

impl McgV2 {
    fn label(&self) -> &'static str {
        match self {
            McgV2::Split => "mcg-split",
            McgV2::Sum2 => "mcg-sum2 ",
            McgV2::Pair => "mcg-pair ",
        }
    }
}

/// One masked-MCG fp16 pair: `(x & 0x8FFF8FFF) ^ 0x3B603B60`, split into halves.
fn mcg_halves(x: u32) -> (f32, f32) {
    let m = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
    (
        half::f16::from_bits((m >> 16) as u16).to_f32(),
        half::f16::from_bits((m & 0xFFFF) as u16).to_f32(),
    )
}

/// Materialise the K=4/V=2 computed codebook, normalised to unit sigma.
///
/// Same 65,536 x 2 layout as [`gaussian_lut`], so it is a drop-in replacement
/// for both the trellis search and the decode.
fn mcg_codebook_v2(mult: u32, variant: McgV2) -> Vec<f32> {
    let mut cb = Vec::with_capacity((1usize << 16) * 2);
    for state in 0..(1u32 << 16) {
        let x = state.wrapping_mul(mult);
        let (v0, v1) = match variant {
            McgV2::Split => mcg_halves(x),
            McgV2::Sum2 => {
                let (a0, a1) = mcg_halves(x);
                let (b0, b1) = mcg_halves(x.wrapping_mul(mult));
                (a0 + a1, b0 + b1)
            }
            McgV2::Pair => {
                let (a0, a1) = mcg_halves(state.wrapping_mul(2).wrapping_mul(mult));
                let (b0, b1) = mcg_halves(state.wrapping_mul(2).wrapping_add(1).wrapping_mul(mult));
                (a0 + a1, b0 + b1)
            }
        };
        cb.push(v0);
        cb.push(v1);
    }
    let n = cb.len() as f64;
    let mean = cb.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = cb.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let sigma = var.sqrt();
    if sigma > 0.0 {
        for v in cb.iter_mut() {
            *v = (*v as f64 / sigma) as f32;
        }
    }
    cb
}

/// Shape statistics of a codebook: what the search actually has to work with.
fn codebook_stats(cb: &[f32]) -> (f64, f64, f64, f64) {
    let n = cb.len() as f64;
    let mean = cb.iter().map(|&v| v as f64).sum::<f64>() / n;
    let sigma = (cb.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    let min_abs = cb.iter().map(|&v| v.abs() as f64).fold(f64::INFINITY, f64::min);
    // Fraction of the table inside +/- 0.25 sigma — a Gaussian source puts
    // ~19.7% of its mass there, so a codebook that puts ~0% there has a hole
    // exactly where the weights are densest.
    let near = cb.iter().filter(|&&v| (v as f64).abs() < 0.25).count() as f64 / n;
    (mean, sigma, min_abs, near)
}

/// **THE GATE on the 1.81x codebook prize.**
///
/// Fixtures (D12 — the distribution is stated because it has flipped a decision
/// in this repo twice): `gaussian` (control), `student_t4` (heavy-tailed), and
/// `fp4_dequant` (FP4-lattice — the actual source chain of V4's experts, and the
/// one that exposed greedy at 0.675 where Gaussian showed 0.888). THREE
/// independent weight draws per family, because with the unweighted objective
/// the quantization does not depend on the activations at all: varying the
/// activation spread only redraws the evaluation set, so it measures eval noise
/// and NOT search noise. Two activation spreads are kept inside the realistic
/// 1e2-1e4 channel-energy band on top of that.
///
/// Weight NMSE is the primary number here and matmul cos the secondary one,
/// which is the opposite of the rotation-vs-Hessian sweep's emphasis and is
/// deliberate: NMSE is activation-independent, so it isolates what changing the
/// codebook actually did, while cos mixes in the eval draw.
///
/// Production geometry throughout: K=4, V=2, L=16, beam W=256, rotation 128,
/// unweighted objective.
///
/// `cargo test --release -p mistralrs-quant --lib probe_computed_codebook_quality \
///     -- --ignored --nocapture`
#[test]
#[ignore = "evidence probe (~2 min); run with --release --ignored --nocapture"]
fn probe_computed_codebook_quality() {
    let (n, k, cb_batch, eb) = (48usize, 2048usize, 128usize, 128usize);
    let search = TrellisSearch::Beam { width: 256 };

    // exllamav3 PR #26's spectrally-optimised multiplier (what qtip2b ships) and
    // EXL3's original, which is what the CUDA speed run happened to use.
    let mults: [(&str, u32); 2] = [
        ("opt", 0xCAF6_A435),
        ("exl3", 0xCBAC_1FED),
    ];

    let mut books: Vec<(String, Vec<f32>)> = vec![("gaussian-LUT".to_string(), gaussian_lut())];
    for (mname, mult) in mults {
        for variant in [McgV2::Split, McgV2::Sum2, McgV2::Pair] {
            books.push((
                format!("{} {mname}", variant.label()),
                mcg_codebook_v2(mult, variant),
            ));
        }
    }

    println!("\ncodebook shape (all normalised to unit sigma before use)");
    println!(
        "{:<20} | {:>9} | {:>7} | {:>9} | {:>13}",
        "codebook", "mean", "sigma", "min |v|", "frac |v|<0.25"
    );
    for (name, book) in &books {
        let (mean, sigma, min_abs, near) = codebook_stats(book);
        println!(
            "{name:<20} | {mean:>9.5} | {sigma:>7.4} | {min_abs:>9.5} | {:>12.2}%",
            near * 100.0
        );
    }
    println!("(a standard normal puts 19.7% of its mass inside +/-0.25)");

    // Three independent weight draws per family.
    let families: [(&str, u64, u64, u64); 3] = [
        ("gaussian   ", 1, 11, 21),
        ("student_t4 ", 2, 12, 22),
        ("fp4_dequant", 0x0051_EA11, 0x0051_EA12, 0x0051_EA13),
    ];
    let spreads = [("awq-like", 0.3, 2.75), ("moderate", 0.6, 2.33)];

    // [codebook][family] -> (cos samples, nmse samples)
    let mut agg: Vec<Vec<(Vec<f64>, Vec<f64>)>> =
        vec![vec![(Vec::new(), Vec::new()); families.len()]; books.len()];

    for (fi, (fname, s0, s1, s2)) in families.iter().enumerate() {
        for &seed in &[*s0, *s1, *s2] {
            let w = match fi {
                0 => super::bake_quality_tests::gen_gaussian(n, k, 0.02, seed),
                1 => super::bake_quality_tests::gen_student_t(n, k, 0.02, seed),
                _ => gen_fp4_dequant(n, k, 0.02, seed),
            };
            for (sname, sigma, oz) in spreads {
                let _ = sname;
                let scales = channel_scales_sigma(k, 0xC0FF_EE01, sigma, oz);
                let x_cal = gen_activations(cb_batch, &scales, 0x1234_5678);
                let x_eval = gen_activations(eb, &scales, 0x8765_4321);
                let h = hessian_diag(&x_cal, cb_batch, k);
                let ev = EvalSet {
                    w: &w,
                    n,
                    k,
                    x_eval: &x_eval,
                    batch: eb,
                    h_diag: &h,
                };
                for (bi, (_, book)) in books.iter().enumerate() {
                    let cfg = SearchCfg {
                        search,
                        weighted: false,
                    };
                    let (w_hat, secs) = quantize_matrix_cb(&w, n, k, cfg, &h, ROT_BLOCK, book);
                    let r = evaluate(&ev, &w_hat, secs);
                    agg[bi][fi].0.push(r.matmul_cos);
                    agg[bi][fi].1.push(r.weight_nmse);
                }
            }
            let _ = fname;
        }
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let spread_of = |v: &[f64]| {
        v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - v.iter().cloned().fold(f64::INFINITY, f64::min)
    };

    println!(
        "\nbeam W=256, rotation 128, unweighted, n={n} k={k}, 3 weight draws x 2 activation draws"
    );
    println!(
        "{:<20} | {:<12} | {:>9} | {:>9} | {:>10} | {:>10} | {:>9}",
        "codebook", "fixture", "cos", "d(cos)", "w_nmse", "d_rel_nmse", "cos range"
    );
    for (bi, (bname, _)) in books.iter().enumerate() {
        for (fi, (fname, ..)) in families.iter().enumerate() {
            let c = mean(&agg[bi][fi].0);
            let nm = mean(&agg[bi][fi].1);
            let c0 = mean(&agg[0][fi].0);
            let nm0 = mean(&agg[0][fi].1);
            println!(
                "{bname:<20} | {fname:<12} | {c:>9.5} | {:>+9.5} | {nm:>10.6} | {:>+9.2}% | {:>9.5}",
                c - c0,
                (nm / nm0 - 1.0) * 100.0,
                spread_of(&agg[bi][fi].0)
            );
        }
    }

    println!(
        "\n{:<20} | {:>10} | {:>10} | {:>11} | {:>11}",
        "codebook", "mean cos", "d(cos)", "mean w_nmse", "d_rel_nmse"
    );
    let all = |bi: usize, sel: usize| -> f64 {
        let mut v = Vec::new();
        for cell in &agg[bi] {
            let src = if sel == 0 { &cell.0 } else { &cell.1 };
            v.extend(src.iter().cloned());
        }
        mean(&v)
    };
    let lut_cos = all(0, 0);
    let lut_nmse = all(0, 1);
    for (bi, (bname, _)) in books.iter().enumerate() {
        println!(
            "{bname:<20} | {:>10.5} | {:>+10.5} | {:>11.6} | {:>+10.2}%",
            all(bi, 0),
            all(bi, 0) - lut_cos,
            all(bi, 1),
            (all(bi, 1) / lut_nmse - 1.0) * 100.0
        );
    }

    // The gate is not "is it better" — it is that the comparison is a codebook
    // comparison at all. A codebook so mismatched that the search collapses would
    // show a cos far below the ladder, and reporting THAT as a small delta is the
    // failure mode this assertion exists to stop.
    for (bi, (bname, _)) in books.iter().enumerate() {
        assert!(
            all(bi, 0) > 0.9,
            "{bname} scored mean cos {} — the search collapsed rather than traded \
             quality, so its delta against the LUT is not a codebook comparison",
            all(bi, 0)
        );
    }
    // Non-degeneracy of the fixture family: the three weight draws must actually
    // differ, or "3 seeds" is decoration.
    assert!(
        spread_of(&agg[0][2].1) > 1e-6,
        "the three fp4_dequant draws produced identical weight NMSE — they are \
         not independent draws"
    );
}

// ---------------------------------------------------------------------------
// wave19-AP — is exhaustive Viterbi actually better than the beam we ship?
// ---------------------------------------------------------------------------

/// The quality half of the gmin-exhaustive decision.
///
/// The gmin-only CUDA kernel (`kernels/qtip/qtip_gmin.cu`) is byte-identical to
/// `TrellisSearch::Exhaustive` — `gmin_replay_matches_exhaustive_bit_for_bit`
/// proves the recursion, and the hardware gate proves the kernel — so "what
/// does exhaustive cost or buy in quality against beam-256" is the whole
/// question, and it must be answered on the fixture FAMILY rather than on one
/// draw (D12: a pure sinusoid once made the banned greedy search look better
/// than Viterbi, and a single over-dispersed Hessian fixture nearly inverted
/// the rotation default).
///
/// The intuition that exhaustive must win is WRONG in a way worth stating: the
/// beam is not an approximation of the objective being scored here. Both
/// searches minimise the same per-row trellis cost, exhaustive attains its
/// minimum exactly, and the beam attains it or slightly above — but the score
/// is `cos` of a *matmul* against held-out activations, which the trellis cost
/// only proxies. A search that is optimal on the proxy can land marginally
/// worse on the score, and `STATUS.md` already records one fixture where it
/// does (beam W=256 0.96680 vs exhaustive 0.96495).
///
/// `cargo test --release -p mistralrs-quant --lib probe_exhaustive_vs_beam_quality \
///     -- --ignored --nocapture`
#[test]
#[ignore = "evidence probe (minutes); run with --release --ignored --nocapture"]
fn probe_exhaustive_vs_beam_quality() {
    let (n, k, cb, eb) = (16usize, 1024usize, 128usize, 128usize);
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
    // Channel-energy dispersions inside the band real LLM activations occupy
    // (~1e2-1e4), per D11/D12 — not the log-normal fantasy that produced the
    // 1.2e7:1 artefact.
    let spreads = [
        ("mild     s=0.3 out=0%  ", 0.3, 999.0),
        ("awq-like s=0.3 out=0.3%", 0.3, 2.75),
        ("moderate s=0.6 out=1%  ", 0.6, 2.33),
    ];

    println!(
        "\nfixture     | spread                  | exhaustive |  beam W=256 |  beam W=128 | \
         d(256) | d(128) | w_nmse exh | w_nmse b256"
    );
    let mut beam256_wins = 0usize;
    let mut cells = 0usize;
    for (fname, w) in &fixtures {
        for (sname, sigma, oz) in spreads {
            let scales = channel_scales_sigma(k, 0xC0FF_EE01, sigma, oz);
            let x_cal = gen_activations(cb, &scales, 0x1234_5678);
            let x_eval = gen_activations(eb, &scales, 0x8765_4321);
            let h = hessian_diag(&x_cal, cb, k);
            let ev = EvalSet {
                w,
                n,
                k,
                x_eval: &x_eval,
                batch: eb,
                h_diag: &h,
            };
            let mut cos = [0f64; 3];
            let mut wn = [0f64; 3];
            for (i, search) in [
                TrellisSearch::Exhaustive,
                TrellisSearch::Beam { width: 256 },
                TrellisSearch::Beam { width: 128 },
            ]
            .into_iter()
            .enumerate()
            {
                let cfg = SearchCfg {
                    search,
                    weighted: false,
                };
                let (w_hat, s) = quantize_matrix(w, n, k, cfg, &h, ROT_BLOCK);
                let r = evaluate(&ev, &w_hat, s);
                cos[i] = r.matmul_cos;
                wn[i] = r.weight_nmse;
            }
            cells += 1;
            if cos[1] > cos[0] {
                beam256_wins += 1;
            }
            println!(
                "{fname} | {sname} | {:>10.5} | {:>11.5} | {:>11.5} | {:>+6.4} | {:>+6.4} | \
                 {:>10.6} | {:>11.6}",
                cos[0],
                cos[1],
                cos[2],
                cos[1] - cos[0],
                cos[2] - cos[0],
                wn[0],
                wn[1]
            );
        }
    }
    println!(
        "\nbeam W=256 scored HIGHER than exhaustive on {beam256_wins}/{cells} cells \
         (the metric is matmul cos, which the trellis cost only proxies)"
    );

    // Exhaustive minimises the objective the search actually optimises, so its
    // WEIGHT nmse must never be beaten by a pruned search. That is the
    // invariant worth pinning; matmul cos is a downstream proxy and is not.
    for (fname, w) in &fixtures {
        let scales = channel_scales_sigma(k, 0xC0FF_EE01, 0.3, 2.75);
        let x_cal = gen_activations(cb, &scales, 0x1234_5678);
        let h = hessian_diag(&x_cal, cb, k);
        let mut nm = [0f64; 2];
        for (i, search) in [TrellisSearch::Exhaustive, TrellisSearch::Beam { width: 256 }]
            .into_iter()
            .enumerate()
        {
            let cfg = SearchCfg {
                search,
                weighted: false,
            };
            let (w_hat, _) = quantize_matrix(w, n, k, cfg, &h, ROT_BLOCK);
            nm[i] = nmse(w, &w_hat, None, k);
        }
        assert!(
            nm[0] <= nm[1] * 1.000_001,
            "{fname}: exhaustive weight nmse {} is WORSE than beam-256's {} — the \
             exhaustive DP is by construction the minimiser, so this means the \
             searches are not optimising the same cost",
            nm[0],
            nm[1]
        );
    }
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
/// `cuobjdump -res-usage` on sm_90a, BEFORE wave16:
/// `REG:80 STACK:0 SHARED:38992 LOCAL:0`. 256 threads x 80 reg = 20,480
/// reg/block; 65,536 / 20,480 = 3 blocks/SM; 24 of 64 warps = 37.5% occupancy.
/// Register-limited, not shared-limited (3 x 38,992 B = 114 KiB of 228 KiB).
pub(crate) const MEASURED_BEAM_REGISTERS_PER_THREAD: usize = 80;
pub(crate) const MEASURED_BEAM_SHARED_BYTES: usize = 38_992;
pub(crate) const MEASURED_BEAM_BLOCKS_PER_SM: usize = 3;

/// The same command on a wave16 build (branch head `9affb097d`, built in a
/// separate checkout so the running bake was untouched):
/// `REG:64 STACK:0 SHARED:39024 LOCAL:0`.
///
/// `LOCAL:0` is the load-bearing half. `__launch_bounds__(256, 4)` does not
/// refuse to compile when it cannot reach 64 registers — it spills, and a
/// spilled load in the radix loop would run 16 x ~3.87 times per timestep and
/// invert the occupancy gain. nvcc reached the budget without spilling, so the
/// 3 -> 4 blocks/SM step is real.
///
/// This is a property of the exact source that produced it. Any later edit to
/// the radix loop or the candidate registers re-opens the question and the
/// `cuobjdump` gate must be re-run — which is why both endpoints are asserted
/// below rather than left in a document.
pub(crate) const MEASURED_BEAM_REGISTERS_AFTER_WAVE16: usize = 64;
pub(crate) const MEASURED_BEAM_SHARED_BYTES_AFTER_WAVE16: usize = 39_024;
pub(crate) const MEASURED_BEAM_BLOCKS_PER_SM_AFTER_WAVE16: usize = 4;

// Compile-time, because these are facts about constants and a runtime assert
// would only fire if someone happened to run the test.
const _: () = assert!(
    MEASURED_BEAM_BLOCKS_PER_SM_AFTER_WAVE16 > MEASURED_BEAM_BLOCKS_PER_SM,
    "the wave16 launch bounds are supposed to BUY occupancy"
);
const _: () = assert!(
    MEASURED_BEAM_SHARED_BYTES_AFTER_WAVE16 * MEASURED_BEAM_BLOCKS_PER_SM_AFTER_WAVE16 <= 233_472,
    "sm_90 has 228 KiB of shared memory per SM; the target occupancy must fit it"
);

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
        "  beam kernel  (pre-wave16): REG:{} SHARED:{} LOCAL:0 -> {} blocks/SM, \
         {:.1}% occupancy (register-limited)",
        MEASURED_BEAM_REGISTERS_PER_THREAD,
        MEASURED_BEAM_SHARED_BYTES,
        MEASURED_BEAM_BLOCKS_PER_SM,
        MEASURED_BEAM_BLOCKS_PER_SM as f64 * 8.0 / 64.0 * 100.0
    );
    println!(
        "  beam kernel (post-wave16): REG:{} SHARED:{} LOCAL:0 -> {} blocks/SM, \
         {:.1}% occupancy — no spill",
        MEASURED_BEAM_REGISTERS_AFTER_WAVE16,
        MEASURED_BEAM_SHARED_BYTES_AFTER_WAVE16,
        MEASURED_BEAM_BLOCKS_PER_SM_AFTER_WAVE16,
        MEASURED_BEAM_BLOCKS_PER_SM_AFTER_WAVE16 as f64 * 8.0 / 64.0 * 100.0
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

    // Occupancy arithmetic, checked against what `cuobjdump -res-usage`
    // actually reported at BOTH ends of the wave16 change. Encoding both means
    // a future edit that silently costs occupancy fails here rather than in a
    // six-hour bake.
    assert_eq!(
        blocks_per_sm(
            MEASURED_BEAM_REGISTERS_PER_THREAD,
            MEASURED_BEAM_SHARED_BYTES
        ),
        MEASURED_BEAM_BLOCKS_PER_SM,
        "occupancy arithmetic disagrees with the pre-wave16 cuobjdump result"
    );
    assert_eq!(
        blocks_per_sm(
            MEASURED_BEAM_REGISTERS_AFTER_WAVE16,
            MEASURED_BEAM_SHARED_BYTES_AFTER_WAVE16
        ),
        MEASURED_BEAM_BLOCKS_PER_SM_AFTER_WAVE16,
        "occupancy arithmetic disagrees with the post-wave16 cuobjdump result"
    );
    // 64 registers/thread is exactly the budget 4 blocks/SM requires: one more
    // register per thread drops it back to 3, so this is a cliff, not a slope.
    assert_eq!(
        blocks_per_sm(
            MEASURED_BEAM_REGISTERS_AFTER_WAVE16 + 1,
            MEASURED_BEAM_SHARED_BYTES_AFTER_WAVE16
        ),
        3,
        "65 registers/thread should fall off the 4-blocks/SM cliff"
    );
    // Shared memory limits neither configuration; registers do.
    assert!(
        blocks_per_sm(1, MEASURED_BEAM_SHARED_BYTES_AFTER_WAVE16)
            > MEASURED_BEAM_BLOCKS_PER_SM_AFTER_WAVE16,
        "shared memory alone would allow more blocks — the limit is registers"
    );
    // The hand arithmetic in `beam_kernel_smem_bytes` should track the reported
    // figure closely; the gap is compiler/driver overhead, not a miscount.
    assert!(
        MEASURED_BEAM_SHARED_BYTES_AFTER_WAVE16.saturating_sub(beam_kernel_smem_bytes(256)) < 2048,
        "declared shared memory drifted from what cuobjdump reports"
    );
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

/// Does the Guess-Verify-Refine premise actually hold for OUR cost
/// distribution? wave17-AN ranks GVR the top identity-safe lever; this measures
/// its premise before any kernel is written on it.
///
/// `cargo test -p mistralrs-quant --lib probe_gvr_leverage -- --ignored --nocapture`
#[test]
#[ignore = "evidence probe (~1 min); run with --ignored --nocapture"]
fn probe_gvr_leverage() {
    use super::viterbi::measure_gvr_leverage;

    let lut = gaussian_lut();
    println!(
        "\n=== Guess-Verify-Refine leverage on the real beam cost distribution (W=256) ===\n\
         survivors = candidates left after the guess; the refine step's cost is proportional to it."
    );
    for (label, k) in [("gate/up k=7168", 7168usize), ("down    k=2048", 2048)] {
        let w = gen_fp4_dequant(2, k, 0.02, 0x00BE_A115);
        let signs = generate_signs(QTIP_ROTATION_SEED, k);
        let mut agg = super::viterbi::GvrStats::default();
        for row in 0..2 {
            let mut rot = w[row * k..(row + 1) * k].to_vec();
            apply_block_rotation(&mut rot, &signs, ROT_BLOCK);
            let max_abs = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 3.0 };
            let inv = 1.0 / scale;
            let target: Vec<f32> = rot.iter().map(|&v| v * inv).collect();
            let s = measure_gvr_leverage(&target, &lut, 256);
            agg.steps += s.steps;
            agg.n_cand_sum += s.n_cand_sum;
            agg.temporal_exact_hits += s.temporal_exact_hits;
            agg.temporal_high += s.temporal_high;
            agg.temporal_low += s.temporal_low;
            agg.temporal_survivors_sum += s.temporal_survivors_sum;
            agg.delegate1_survivors_sum += s.delegate1_survivors_sum;
            agg.delegate1_valid_steps += s.delegate1_valid_steps;
            agg.delegate2_survivors_sum += s.delegate2_survivors_sum;
            agg.delegate2_valid_steps += s.delegate2_valid_steps;
            agg.delegate4_survivors_sum += s.delegate4_survivors_sum;
            agg.delegate4_valid_steps += s.delegate4_valid_steps;
            agg.tight2_survivors_sum += s.tight2_survivors_sum;
            agg.tight2_steps += s.tight2_steps;
            agg.tight4_survivors_sum += s.tight4_survivors_sum;
            agg.tight4_steps += s.tight4_steps;
            agg.tight2_delegates_sum += s.tight2_delegates_sum;
        }
        let n = agg.mean_n_cand();
        println!(
            "\n{label}: {} steps, mean n_cand {n:.0}, W=256 (k/n = 1/{:.0})",
            agg.steps,
            n / 256.0
        );
        println!(
            "  TEMPORAL extrapolation: exact hits {:.2}% | usable (guess >= truth) {:.1}% | \
             misses low {:.1}%",
            100.0 * agg.temporal_exact_hits as f64 / agg.steps.max(1) as f64,
            100.0 * agg.temporal_high as f64 / agg.steps.max(1) as f64,
            100.0 * agg.temporal_low as f64 / agg.steps.max(1) as f64
        );
        if agg.temporal_high > 0 {
            println!(
                "     survivors when usable: {:.0} of {n:.0} ({:.1}x reduction)",
                agg.mean_temporal_survivors(),
                n / agg.mean_temporal_survivors().max(1.0)
            );
        }
        for beta in [1usize, 2, 4] {
            let valid = match beta {
                1 => agg.delegate1_valid_steps,
                2 => agg.delegate2_valid_steps,
                _ => agg.delegate4_valid_steps,
            };
            if valid == 0 {
                println!("  Dr.Top-k RULE 2 beta={beta}: bound never provable (beta*ng < W)");
                continue;
            }
            let surv = agg.mean_delegate_survivors(beta);
            println!(
                "  Dr.Top-k RULE 2 beta={beta} (LOOSE, max of delegates): provable on {:.0}% \
                 of steps | survivors {surv:.0} of {n:.0} ({:.1}x reduction)",
                100.0 * valid as f64 / agg.steps as f64,
                n / surv.max(1.0)
            );
        }
        for (beta, sum, cnt) in [
            (2usize, agg.tight2_survivors_sum, agg.tight2_steps),
            (4, agg.tight4_survivors_sum, agg.tight4_steps),
        ] {
            if cnt == 0 {
                continue;
            }
            let surv = sum as f64 / cnt as f64;
            println!(
                "  Dr.Top-k RULE 2 beta={beta} (TIGHT, W-th of delegates): provable on {:.0}% \
                 of steps | survivors {surv:.0} of {n:.0} ({:.1}x reduction)",
                100.0 * cnt as f64 / agg.steps as f64,
                n / surv.max(1.0)
            );
        }
        if agg.tight2_steps > 0 {
            println!(
                "     delegate vector is {:.0} elements ({:.1} per thread vs 16) — the refine's \
                 own sub-selection",
                agg.tight2_delegates_sum as f64 / agg.tight2_steps as f64,
                agg.tight2_delegates_sum as f64 / agg.tight2_steps as f64 / 256.0
            );
        }
    }
}

/// The ordered-key transform must be an exact bijection, because wave17-AF's
/// kernel now carries candidates as keys and recovers the float only for the
/// survivors. If the round trip were lossy the beam would store a different
/// cost than the one it selected on, and every later step would drift.
///
/// Exhaustive over all 2^32 bit patterns is too slow for CI; this covers every
/// structurally interesting class plus a large pseudo-random sample.
#[test]
fn order_key_round_trips_bitwise() {
    fn key(x: f32) -> u32 {
        let b = x.to_bits();
        if b & 0x8000_0000 != 0 {
            !b
        } else {
            b | 0x8000_0000
        }
    }
    fn unkey(k: u32) -> f32 {
        f32::from_bits(if k & 0x8000_0000 != 0 {
            k & 0x7FFF_FFFF
        } else {
            !k
        })
    }

    let mut cases: Vec<u32> = vec![
        0x0000_0000, // +0.0
        0x8000_0000, // -0.0
        0x0000_0001, // smallest denormal
        0x0080_0000, // smallest normal
        0x3F80_0000, // 1.0
        0x7F7F_FFFF, // f32::MAX
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
    ];
    for i in 0..200_000u32 {
        let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        cases.push((z >> 32) as u32);
    }

    let mut prev: Option<(u32, f32)> = None;
    for bits in cases {
        let x = f32::from_bits(bits);
        if x.is_nan() {
            continue; // NaN has no defined position in the cost order
        }
        assert_eq!(
            unkey(key(x)).to_bits(),
            x.to_bits(),
            "round trip lost bits for {x} ({bits:#010x})"
        );
        // Ordering must agree with `f32::total_cmp`, which is what the CPU
        // beam's `prune_to_width` selects on.
        if let Some((pk, px)) = prev {
            let k = key(x);
            assert_eq!(
                k.cmp(&pk),
                x.total_cmp(&px),
                "key order disagrees with total_cmp for {x} vs {px}"
            );
        }
        prev = Some((key(x), x));
    }
}
