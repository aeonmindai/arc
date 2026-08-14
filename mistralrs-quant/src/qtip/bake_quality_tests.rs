//! Bake-quality investigation: CPU evidence for the qtip2-vs-q2k perplexity gap
//! (wave3-G, measured on H200: PPL qtip2 = 58.85 vs q2k = 22.50 on the same
//! mini-wikitext corpus and the same FP4-dequantized source weights).
//!
//! ## What this module establishes
//!
//! The production UQFF bake (`mistralrs quantize ... --isq qtip2`, no env vars)
//! quantizes every 3-D MoE expert stack with `QtipMode::Greedy`, which ALSO
//! disables the Hadamard incoherence rotation (`quantize_with_mode`:
//! Greedy → `use_rotation = false`). So the deployed experts were baked with:
//!
//!   greedy trellis walk  +  NO incoherence rotation  +  max|row|/3 scale
//!
//! while every published QTIP/EXL3 number assumes Viterbi + incoherence
//! processing + tuned scales. The fixtures here are deliberately NOT pure
//! Gaussian — pure-Gaussian fixtures are why the existing tests sit at
//! cos ≥ 0.95 while the real model loses 2.6× in perplexity. We model the real
//! source chain: INT4-packed FP4 (e2m1) experts dequantized to BF16, i.e.
//! 16-level lattice values with per-32-column block scales and heavy-tailed
//! (Student-t) block statistics.
//!
//! ## Tests
//!
//! * `bake_quality_regression_default_expert_path` — **encodes the defect**:
//!   quantizes a realistic FP4-lattice expert stack through the exact
//!   production decision (`QtipMode::default_expert_mode()`) and asserts the
//!   matmul cosine floor that the Viterbi+rotation path reaches on the same
//!   fixture. FAILS while the production default is Greedy/no-rotation.
//! * `ls_refine_replay_matches_decoder_and_improves_mse` — CPU port of the
//!   gated CUDA `qtip_refine_scales_kernel` replay (state walk from packed
//!   nibbles). Proves the *algorithm* is sound: replayed codewords are
//!   byte-identical to the production decoder walk and the refined scale
//!   never degrades row MSE. (The cos=0.628 that got the kernel gated off in
//!   commit 0731ecb4e is therefore an integration/launch bug, not math.)
//! * `probe_bake_quality_ladder` (`#[ignore]`) — the full evidence table
//!   across fixtures × configurations. Run with:
//!   `cargo test -p mistralrs-quant --release bake_quality -- --ignored --nocapture`

use super::{
    apply_block_rotation, gaussian_lut, viterbi_quantize_row, QtipMode, ALPHABET, K,
    QTIP_ROTATION_SEED, STATE_MASK, V,
};
use crate::turboquant::wht::generate_signs;

// ---------------------------------------------------------------------------
// Deterministic RNG (splitmix64) + samplers
// ---------------------------------------------------------------------------

pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in (0, 1).
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 2.0)
    }
    /// Standard normal (Box-Muller, no caching — determinism over speed).
    pub(super) fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    /// Student-t with `nu` degrees of freedom (heavy-tailed; nu=4 has the
    /// outlier profile typical of trained LLM expert rows).
    fn student_t(&mut self, nu: usize) -> f64 {
        let z = self.normal();
        let chi2: f64 = (0..nu).map(|_| self.normal().powi(2)).sum();
        z / (chi2 / nu as f64).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// FP4 (e2m1) magnitude lattice — the positive levels representable by the
/// source checkpoint's INT4-packed FP4 experts.
const FP4_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn snap_to_fp4(x: f32) -> f32 {
    let a = x.abs();
    let mut best = FP4_LEVELS[0];
    let mut best_d = f32::INFINITY;
    for &l in &FP4_LEVELS {
        let d = (a - l).abs();
        if d < best_d {
            best_d = d;
            best = l;
        }
    }
    if x < 0.0 {
        -best
    } else {
        best
    }
}

/// Pure Gaussian control fixture (what the existing tests use — the fixture
/// on which everything looks fine).
pub(super) fn gen_gaussian(n: usize, k: usize, sigma: f64, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n * k).map(|_| (rng.normal() * sigma) as f32).collect()
}

/// Heavy-tailed rows: Student-t(4), scaled to std ~sigma.
pub(super) fn gen_student_t(n: usize, k: usize, sigma: f64, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    // Var of t(4) is nu/(nu-2) = 2 → divide by sqrt(2) to hit target sigma.
    (0..n * k)
        .map(|_| (rng.student_t(4) * sigma / 2f64.sqrt()) as f32)
        .collect()
}

/// The realistic source chain: heavy-tailed weights snapped to the FP4 (e2m1)
/// lattice with a per-32-column block scale — exactly what "INT4-packed FP4
/// experts dequantized to BF16" produces. Rows are 16-level lattice mixtures,
/// NOT Gaussian.
pub(super) fn gen_fp4_dequant(n: usize, k: usize, sigma: f64, seed: u64) -> Vec<f32> {
    const BLOCK: usize = 32;
    let base = gen_student_t(n, k, sigma, seed);
    let mut out = vec![0f32; n * k];
    for row in 0..n {
        for b0 in (0..k).step_by(BLOCK) {
            let blk = &base[row * k + b0..row * k + b0 + BLOCK];
            let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            // FP4 block scale maps the block max onto the top lattice level.
            let s = if amax == 0.0 { 1.0 } else { amax / 6.0 };
            for (i, &v) in blk.iter().enumerate() {
                out[row * k + b0 + i] = snap_to_fp4(v / s) * s;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Production-mirror quantization pipeline (CPU)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum ScalePolicy {
    /// Production heuristic: scale = max|row| / 3.0.
    MaxOver3,
    /// Search with max/3, then least-squares refine on the fixed symbol
    /// stream — the CPU equivalent of the gated `qtip_refine_scales_kernel`.
    MaxOver3LsRefine,
    /// Search with an RMS-matched scale (target std = 1.0, the LUT's std),
    /// then LS-refine.
    RmsMatchedLsRefine,
}

#[derive(Clone, Copy, Debug)]
struct Cfg {
    viterbi: bool,
    /// 0 = no rotation (production Greedy). 128 = production Viterbi cap.
    /// k = full-width Hadamard (cap lifted).
    rotation_block: usize,
    policy: ScalePolicy,
}

/// Greedy trellis walk — verbatim port of the production CPU greedy
/// (`quantize_with_options_concrete`) and the CUDA greedy kernel.
fn greedy_quantize_row(target: &[f32], lut: &[f32]) -> Vec<u8> {
    let num_symbols = target.len() / V as usize;
    let mut state: u32 = 0;
    let mut syms = vec![0u8; num_symbols];
    for (t, sym_slot) in syms.iter_mut().enumerate() {
        let target_t = &target[t * V as usize..(t + 1) * V as usize];
        let mut best_sym: u8 = 0;
        let mut best_err = f32::INFINITY;
        for sym in 0..ALPHABET as u32 {
            let next_state = ((state << K) | sym) & STATE_MASK;
            let off = (next_state as usize) * V as usize;
            let mut err = 0f32;
            for v in 0..V as usize {
                let d = lut[off + v] - target_t[v];
                err += d * d;
            }
            if err < best_err {
                best_err = err;
                best_sym = sym as u8;
            }
        }
        state = ((state << K) | best_sym as u32) & STATE_MASK;
        *sym_slot = best_sym;
    }
    syms
}

/// Pack two K=4 symbols per byte, low nibble first (production layout).
pub(super) fn pack_symbols(symbols: &[u8]) -> Vec<u8> {
    let mut packed = vec![0u8; symbols.len() / 2];
    for (i, &s) in symbols.iter().enumerate() {
        if i % 2 == 0 {
            packed[i / 2] = s & 0x0F;
        } else {
            packed[i / 2] |= (s & 0x0F) << 4;
        }
    }
    packed
}

/// Production decoder walk: packed bytes → unscaled codeword sequence.
/// Mirrors `qtip_dequantize_v2_k4_l16_kernel` / `dequantize_weights_f32`.
pub(super) fn decode_packed(packed: &[u8], num_symbols: usize, lut: &[f32]) -> Vec<f32> {
    let mut state: u32 = 0;
    let mut out = Vec::with_capacity(num_symbols * V as usize);
    for t in 0..num_symbols {
        let byte = packed[t / 2];
        let sym = if t % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        state = ((state << K) | sym as u32) & STATE_MASK;
        let off = (state as usize) * V as usize;
        for v in 0..V as usize {
            out.push(lut[off + v]);
        }
    }
    out
}

/// CPU port of the gated CUDA `qtip_refine_scales_kernel`: replay the trellis
/// from the packed nibbles and solve s* = ⟨w, c⟩ / ⟨c, c⟩ (f32 accumulation,
/// like the kernel). Returns the refined scale, falling back to `heur` when
/// the refined value is non-positive (pathological row).
fn ls_refine_scale(
    w_rotated: &[f32],
    packed: &[u8],
    num_symbols: usize,
    lut: &[f32],
    heur: f32,
) -> f32 {
    let mut state: u32 = 0;
    let mut dot_wl = 0f32;
    let mut dot_ll = 0f32;
    for t in 0..num_symbols {
        let byte = packed[t / 2];
        let sym = if t % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        state = ((state << K) | sym as u32) & STATE_MASK;
        let off = (state as usize) * V as usize;
        let l0 = lut[off];
        let l1 = lut[off + 1];
        let w0 = w_rotated[t * 2];
        let w1 = w_rotated[t * 2 + 1];
        dot_wl += w0 * l0 + w1 * l1;
        dot_ll += l0 * l0 + l1 * l1;
    }
    if dot_ll > 0.0 && dot_wl / dot_ll > 0.0 {
        dot_wl / dot_ll
    } else {
        heur
    }
}

/// Quantize an `[n, k]` matrix under `cfg`, mirroring the production CPU
/// pipeline op-for-op, and return the reconstruction in the ORIGINAL frame.
fn quantize_matrix(w: &[f32], n: usize, k: usize, cfg: Cfg) -> Vec<f32> {
    use rayon::prelude::*;
    let lut = gaussian_lut();
    let signs = if cfg.rotation_block >= 2 {
        generate_signs(QTIP_ROTATION_SEED, k)
    } else {
        Vec::new()
    };
    let rows: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|row| {
            let raw = &w[row * k..(row + 1) * k];
            // 1. Rotate (production order: rotate, THEN pick the scale).
            let mut rot = raw.to_vec();
            if cfg.rotation_block >= 2 {
                apply_block_rotation(&mut rot, &signs, cfg.rotation_block);
            }
            // 2. Search scale.
            let max_abs = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let search_scale = match cfg.policy {
                ScalePolicy::MaxOver3 | ScalePolicy::MaxOver3LsRefine => {
                    if max_abs == 0.0 {
                        1.0
                    } else {
                        max_abs / 3.0
                    }
                }
                ScalePolicy::RmsMatchedLsRefine => {
                    let rms = (rot.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / k as f64)
                        .sqrt() as f32;
                    if rms == 0.0 {
                        1.0
                    } else {
                        rms
                    }
                }
            };
            let inv = 1.0 / search_scale;
            let target: Vec<f32> = rot.iter().map(|&v| v * inv).collect();
            // 3. Trellis search.
            let syms = if cfg.viterbi {
                viterbi_quantize_row(&target, &lut)
            } else {
                greedy_quantize_row(&target, &lut)
            };
            let packed = pack_symbols(&syms);
            let num_symbols = k / V as usize;
            // 4. Final scale.
            let final_scale = match cfg.policy {
                ScalePolicy::MaxOver3 => search_scale,
                ScalePolicy::MaxOver3LsRefine | ScalePolicy::RmsMatchedLsRefine => {
                    ls_refine_scale(&rot, &packed, num_symbols, &lut, search_scale)
                }
            };
            // 5. Decode + un-rotate (D·H·D is involutory).
            let mut recon: Vec<f32> = decode_packed(&packed, num_symbols, &lut)
                .into_iter()
                .map(|c| c * final_scale)
                .collect();
            if cfg.rotation_block >= 2 {
                apply_block_rotation(&mut recon, &signs, cfg.rotation_block);
            }
            recon
        })
        .collect();
    rows.into_iter().flatten().collect()
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

pub(super) fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

/// y = x @ W^T for x `[batch, k]`, W `[n, k]` → `[batch, n]`.
pub(super) fn matmul_t(x: &[f32], w: &[f32], batch: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0f32; batch * n];
    for b in 0..batch {
        for r in 0..n {
            let mut acc = 0f64;
            let xb = &x[b * k..(b + 1) * k];
            let wr = &w[r * k..(r + 1) * k];
            for i in 0..k {
                acc += xb[i] as f64 * wr[i] as f64;
            }
            y[b * n + r] = acc as f32;
        }
    }
    y
}

/// Tiny linear-probe proxy perplexity: treat the n output rows as classes,
/// label each activation with the exact-weight argmax, and report
/// exp(CE_quantized − CE_exact). 1.0 = no quality loss; bigger = worse.
fn proxy_ppl_ratio(y_exact: &[f32], y_quant: &[f32], batch: usize, n: usize) -> f64 {
    let ce = |y: &[f32], labels: &[usize]| -> f64 {
        let mut total = 0f64;
        for b in 0..batch {
            let row = &y[b * n..(b + 1) * n];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let logsumexp = m + row
                .iter()
                .map(|&v| ((v as f64) - m).exp())
                .sum::<f64>()
                .ln();
            total += logsumexp - y[b * n + labels[b]] as f64;
        }
        total / batch as f64
    };
    let labels: Vec<usize> = (0..batch)
        .map(|b| {
            let row = &y_exact[b * n..(b + 1) * n];
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
        .collect();
    (ce(y_quant, &labels) - ce(y_exact, &labels)).exp()
}

struct Metrics {
    weight_cos: f64,
    matmul_cos: f64,
    ppl_ratio: f64,
}

fn evaluate(w: &[f32], w_hat: &[f32], n: usize, k: usize) -> Metrics {
    const BATCH: usize = 32;
    let x = gen_gaussian(BATCH, k, 1.0, 0xAC71_5EED_CAFE);
    let y = matmul_t(&x, w, BATCH, n, k);
    let y_hat = matmul_t(&x, w_hat, BATCH, n, k);
    Metrics {
        weight_cos: cosine(w, w_hat),
        matmul_cos: cosine(&y, &y_hat),
        ppl_ratio: proxy_ppl_ratio(&y, &y_hat, BATCH, n),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Suspect 1 (scale refinement): the LS-refine *algorithm* (exact CPU port of
/// the CUDA kernel's replay: packed-nibble unpack → state shift → LUT) is
/// consistent with the production decoder and never degrades row MSE.
/// If this holds, the cos=0.628 that gated the kernel off (0731ecb4e) cannot
/// be the replay math — it must be kernel-launch/integration level.
#[test]
fn ls_refine_replay_matches_decoder_and_improves_mse() {
    let lut = gaussian_lut();
    let n = 8;
    let k = 512;
    let num_symbols = k / V as usize;
    let signs = generate_signs(QTIP_ROTATION_SEED, k);
    for (name, w) in [
        ("gaussian", gen_gaussian(n, k, 0.02, 7)),
        ("student_t4", gen_student_t(n, k, 0.02, 8)),
        ("fp4_dequant", gen_fp4_dequant(n, k, 0.02, 9)),
    ] {
        for row in 0..n {
            let mut rot = w[row * k..(row + 1) * k].to_vec();
            apply_block_rotation(&mut rot, &signs, 128);
            let max_abs = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 3.0 };
            let target: Vec<f32> = rot.iter().map(|&v| v / scale).collect();
            let syms = viterbi_quantize_row(&target, &lut);
            let packed = pack_symbols(&syms);

            // (a) The refine replay's codeword sequence == the decoder's.
            let decoded = decode_packed(&packed, num_symbols, &lut);
            let mut state = 0u32;
            for t in 0..num_symbols {
                let byte = packed[t / 2];
                let sym = if t % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                state = ((state << K) | sym as u32) & STATE_MASK;
                let off = (state as usize) * V as usize;
                assert_eq!(
                    (decoded[t * 2], decoded[t * 2 + 1]),
                    (lut[off], lut[off + 1]),
                    "{name} row {row}: refine replay diverged from decoder at t={t}"
                );
            }

            // (b) The refined scale does not degrade reconstruction MSE.
            let refined = ls_refine_scale(&rot, &packed, num_symbols, &lut, scale);
            assert!(
                refined > 0.0,
                "{name} row {row}: non-positive refined scale"
            );
            let mse = |s: f32| -> f64 {
                rot.iter()
                    .zip(decoded.iter())
                    .map(|(&wv, &c)| ((wv - c * s) as f64).powi(2))
                    .sum::<f64>()
                    / k as f64
            };
            let mse_heur = mse(scale);
            let mse_ref = mse(refined);
            assert!(
                mse_ref <= mse_heur * (1.0 + 1e-6),
                "{name} row {row}: LS refine degraded MSE ({mse_ref:.6e} > {mse_heur:.6e})"
            );
        }
    }
}

/// **The regression test that encodes today's bake defect.**
///
/// Fixture: a stacked-expert weight in the realistic source distribution
/// (FP4/e2m1 lattice values × per-32-block scales over heavy-tailed rows —
/// what the V4-Flash bake actually reads after dequantizing the INT4-packed
/// FP4 experts to BF16).
///
/// Path under test: the exact production decision for 3-D expert stacks —
/// `QtipMode::default_expert_mode()` (what `UnquantLinear::apply_isq` uses for
/// `--isq qtip2`) through `QtipLayer::quantize_with_mode`.
///
/// Floor: the same fixture through Viterbi+rotation reaches matmul cos ≥ 0.95
/// (see `probe_bake_quality_ladder`); q2k-class quality needs at least that.
/// Today the default path is Greedy + NO rotation + max/3 scales and lands
/// around cos ≈ 0.87-0.89 → this test FAILS until the bake default is fixed.
#[test]
fn bake_quality_regression_default_expert_path() {
    use candle_core::{DType, Device, Tensor};

    let (e, n, k) = (2usize, 16usize, 512usize);
    let w_data = gen_fp4_dequant(e * n, k, 0.02, 42);
    let w = Tensor::from_vec(w_data.clone(), (e, n, k), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    // The production ISQ decision point for 3-D MoE expert stacks.
    let mode = QtipMode::default_expert_mode();
    let layer = super::QtipLayer::quantize_with_mode(&w, None, &Device::Cpu, mode).unwrap();
    let w_hat: Vec<f32> = layer
        .dequantize_w()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    // Per-expert matmul cosine on the realistic fixture.
    let mut worst = f64::INFINITY;
    for ex in 0..e {
        let m = evaluate(
            &w_data[ex * n * k..(ex + 1) * n * k],
            &w_hat[ex * n * k..(ex + 1) * n * k],
            n,
            k,
        );
        worst = worst.min(m.matmul_cos);
    }
    assert!(
        worst >= 0.95,
        "bake-quality defect: default expert bake path ({mode:?}) reaches matmul cos {worst:.4} \
         on FP4-lattice heavy-tailed experts; the Viterbi+rotation floor on this fixture is ≥0.95. \
         This is the root cause of PPL qtip2=58.85 vs q2k=22.50 (H200, 2026-08-13). \
         Fix: make 3-D expert stacks default to Viterbi+rotation (mistralrs-quant/src/unquantized/mod.rs)."
    );
}

/// The full evidence ladder. `--ignored --nocapture --release` to run.
#[test]
#[ignore = "evidence-gathering probe (slow); run with --ignored --nocapture --release"]
fn probe_bake_quality_ladder() {
    let n = 48;
    let k = 2048; // power of 2 → full-width Hadamard variant is legal
    let sigma = 0.02;

    let fixtures: Vec<(&str, Vec<f32>)> = vec![
        ("gaussian   ", gen_gaussian(n, k, sigma, 1)),
        ("student_t4 ", gen_student_t(n, k, sigma, 2)),
        ("fp4_dequant", gen_fp4_dequant(n, k, sigma, 3)),
    ];

    let configs: Vec<(&str, Cfg)> = vec![
        (
            "A prod-today  greedy rot=0   max/3    ",
            Cfg {
                viterbi: false,
                rotation_block: 0,
                policy: ScalePolicy::MaxOver3,
            },
        ),
        (
            "B greedy      rot=128 max/3           ",
            Cfg {
                viterbi: false,
                rotation_block: 128,
                policy: ScalePolicy::MaxOver3,
            },
        ),
        (
            "C viterbi     rot=0   max/3           ",
            Cfg {
                viterbi: true,
                rotation_block: 0,
                policy: ScalePolicy::MaxOver3,
            },
        ),
        (
            "D env-path    vit rot=128 max/3       ",
            Cfg {
                viterbi: true,
                rotation_block: 128,
                policy: ScalePolicy::MaxOver3,
            },
        ),
        (
            "E D+LSrefine  vit rot=128 max/3+LS    ",
            Cfg {
                viterbi: true,
                rotation_block: 128,
                policy: ScalePolicy::MaxOver3LsRefine,
            },
        ),
        (
            "F rms-scale   vit rot=128 rms+LS      ",
            Cfg {
                viterbi: true,
                rotation_block: 128,
                policy: ScalePolicy::RmsMatchedLsRefine,
            },
        ),
        (
            "G full-Had    vit rot=2048 max/3+LS   ",
            Cfg {
                viterbi: true,
                rotation_block: 2048,
                policy: ScalePolicy::MaxOver3LsRefine,
            },
        ),
    ];

    println!("\n=== bake-quality ladder (n={n}, k={k}, sigma={sigma}) ===");
    println!("config                                    | fixture     | w_cos   | mm_cos  | pplx");
    for (fname, w) in &fixtures {
        for (cname, cfg) in &configs {
            let w_hat = quantize_matrix(w, n, k, *cfg);
            let m = evaluate(w, &w_hat, n, k);
            println!(
                "{cname} | {fname} | {:.5} | {:.5} | {:.4}",
                m.weight_cos, m.matmul_cos, m.ppl_ratio
            );
        }
        println!("------");
    }
}

/// **The qtip2b twin of `bake_quality_regression_default_expert_path`
/// (wave13-AG).**
///
/// The LUT rung's regression test above was fixed in PR #20; the bitshift rung
/// next to it was missed and kept sending 3-D expert stacks to
/// `QtipMode::Greedy` unless `ARC_QTIP_EXPERT_VITERBI` was set — which also
/// disabled the Hadamard rotation. That is the rung the trellis grouped-GEMM
/// keystone and both GEMV autotune grids are built around, so it was never a
/// dormant path.
///
/// This test goes through the **production ISQ dispatch itself** —
/// `UnquantLinear::apply_isq(Some(IsqType::Qtip2b), ..)`, the exact code
/// `mistralrs quantize --isq qtip2b` runs — rather than calling a quantize
/// helper directly. A twin that called `Qtip2bLayer::quantize_with_mode`
/// would have stayed green through the entire defect, because the defect was
/// in the *dispatch*, not in the quantizer.
///
/// Fixture: FP4/e2m1 lattice values with per-32-column block scales over
/// heavy-tailed rows — the real source chain of the V4-Flash experts after the
/// INT4-packed FP4 weights are dequantized to BF16.
///
/// Measured on this fixture: **matmul cos 0.6794** with the old greedy
/// dispatch (RED), **0.9623** after the fix (GREEN). 0.679 reproduces
/// wave3-G's independently-measured 0.675 for greedy + no-rotation on the
/// FP4-lattice fixtures.
#[test]
fn qtip2b_bake_quality_regression_production_isq_path() {
    use crate::{IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, UnquantLinear};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Linear;
    use std::sync::{atomic::AtomicUsize, Arc};

    let (e, n, k) = (2usize, 16usize, 512usize);
    let w_data = gen_fp4_dequant(e * n, k, 0.02, 42);
    let w = Tensor::from_vec(w_data.clone(), (e, n, k), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let unquant: Arc<UnquantLinear> =
        Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(w, None))).unwrap());
    let layer = unquant
        .apply_isq(
            Some(IsqType::Qtip2b),
            Device::Cpu,
            &AtomicUsize::new(0),
            None,
            QuantizeOntoGuard::new(),
        )
        .expect("production qtip2b ISQ dispatch must quantize a 3-D expert stack");

    let w_hat: Vec<f32> = layer
        .dequantize_w()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut worst = f64::INFINITY;
    for ex in 0..e {
        let m = evaluate(
            &w_data[ex * n * k..(ex + 1) * n * k],
            &w_hat[ex * n * k..(ex + 1) * n * k],
            n,
            k,
        );
        worst = worst.min(m.matmul_cos);
    }
    assert!(
        worst >= 0.95,
        "bake-quality defect: the production `--isq qtip2b` dispatch reaches matmul cos \
         {worst:.4} on FP4-lattice heavy-tailed experts; the trellis+rotation floor on this \
         fixture is ≥0.95. Greedy search and/or a missing Hadamard rotation is the cause — \
         see DOCTRINE D4 and mistralrs-quant/src/unquantized/mod.rs."
    );
}
