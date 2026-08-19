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
    apply_block_rotation, gaussian_lut, gaussian_lut_geo, viterbi_quantize_row, QtipMode, ALPHABET,
    K, L, QTIP_ROTATION_SEED, STATE_MASK, V,
};
use crate::turboquant::wht::generate_signs;

// ---------------------------------------------------------------------------
// Trellis geometry (wave: K8/V4/L12 GO-NO-GO)
// ---------------------------------------------------------------------------

/// A trellis geometry: `l` state bits, `k` bits per symbol, `v` reproduction
/// values per symbol.
///
/// Rate is `k / v` bits per weight, so K=4/V=2 and K=8/V=4 are the SAME 2 bpw.
/// What differs is (a) the reproduction dimension `v` — a `2^k`-entry VQ in
/// `v` dimensions — and (b) the trellis memory depth `l / k` symbols, which is
/// what `Q2B_WARMUP_SYMS = Q2B_L / Q2B_K` names on the sibling rung.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Geo {
    pub l: u32,
    pub k: u32,
    pub v: u32,
}

impl Geo {
    const fn lut_size(self) -> usize {
        1usize << self.l
    }
    const fn alphabet(self) -> usize {
        1usize << self.k
    }
    /// Trellis memory depth in symbols (`L / K`).
    fn depth(self) -> f64 {
        self.l as f64 / self.k as f64
    }
    fn label(self) -> String {
        format!(
            "K{}/V{}/L{:<2} depth={:.2}",
            self.k,
            self.v,
            self.l,
            self.depth()
        )
    }
}

/// The geometry every shipped artifact was baked with.
pub(super) const GEO_SHIPPED: Geo = Geo { l: 16, k: 4, v: 2 };
/// The candidate: 15.125 → 4.375 instructions per weight (3.46×).
const GEO_CANDIDATE: Geo = Geo { l: 12, k: 8, v: 4 };

/// Greedy trellis walk, geometry-parameterised. At [`GEO_SHIPPED`] this is
/// byte-identical to [`greedy_quantize_row`] (pinned by
/// `geo_pipeline_is_byte_identical_at_shipped_geometry`).
fn greedy_quantize_row_geo(target: &[f32], lut: &[f32], geo: Geo) -> Vec<u8> {
    let v = geo.v as usize;
    let mask = (1u32 << geo.l) - 1;
    let num_symbols = target.len() / v;
    let mut state: u32 = 0;
    let mut syms = vec![0u8; num_symbols];
    for (t, sym_slot) in syms.iter_mut().enumerate() {
        let target_t = &target[t * v..(t + 1) * v];
        let mut best_sym: u8 = 0;
        let mut best_err = f32::INFINITY;
        for sym in 0..geo.alphabet() as u32 {
            let next_state = ((state << geo.k) | sym) & mask;
            let off = (next_state as usize) * v;
            let mut err = 0f32;
            for i in 0..v {
                let d = lut[off + i] - target_t[i];
                err += d * d;
            }
            if err < best_err {
                best_err = err;
                best_sym = sym as u8;
            }
        }
        state = ((state << geo.k) | best_sym as u32) & mask;
        *sym_slot = best_sym;
    }
    syms
}

/// Exhaustive group-min Viterbi, geometry-parameterised. Structurally the same
/// dynamic program as `viterbi::exhaustive_quantize_row` (same group-min
/// factorisation, same `0..alphabet` scan order, same strict-`<` tie-break) with
/// `L`/`K`/`V` lifted from compile-time consts to `geo`.
fn viterbi_quantize_row_geo(target_row: &[f32], lut: &[f32], geo: Geo) -> Vec<u8> {
    let v = geo.v as usize;
    let lut_size = geo.lut_size();
    let alphabet = geo.alphabet();
    let num_symbols = target_row.len() / v;
    assert!(num_symbols > 0, "need at least one symbol position");
    assert!(geo.l >= geo.k, "trellis needs L >= K");
    assert!(geo.k <= 8, "symbols are packed as u8");
    let shift = geo.l - geo.k;

    let dec_err = |s: usize, t: &[f32]| -> f32 {
        let off = s * v;
        let mut e = 0f32;
        for i in 0..v {
            let d = lut[off + i] - t[i];
            e += d * d;
        }
        e
    };

    let inf = f32::INFINITY;
    let mut prev_cost = vec![inf; lut_size];
    let mut curr_cost = vec![inf; lut_size];
    let mut backtrace: Vec<Vec<u8>> = Vec::with_capacity(num_symbols);

    // t = 0: the decoder starts from state 0, so only s in [0, alphabet) is
    // reachable after shifting in sym_0.
    let target_first = &target_row[..v];
    for (s, slot) in prev_cost.iter_mut().enumerate().take(alphabet) {
        *slot = dec_err(s, target_first);
    }

    let num_groups = 1usize << shift;
    let mut group_cost = vec![inf; num_groups];
    let mut group_j = vec![0u32; num_groups];

    for t in 1..num_symbols {
        let target_t = &target_row[t * v..(t + 1) * v];
        let mut bt_t = vec![0u8; lut_size];
        // Phase 1: min over the alphabet shared predecessors of each group.
        for g in 0..num_groups {
            let mut best_cost = inf;
            let mut best_j = 0u32;
            for j in 0..alphabet {
                let c = prev_cost[(j << shift) | g];
                if c < best_cost {
                    best_cost = c;
                    best_j = j as u32;
                }
            }
            group_cost[g] = best_cost;
            group_j[g] = best_j;
        }
        // Phase 2: per-state local error + the group's predecessor min.
        for s in 0..lut_size {
            let g = s >> geo.k;
            curr_cost[s] = dec_err(s, target_t) + group_cost[g];
            bt_t[s] = group_j[g] as u8;
        }
        backtrace.push(bt_t);
        std::mem::swap(&mut prev_cost, &mut curr_cost);
    }

    let mut best_final = 0usize;
    let mut best_final_cost = inf;
    for (s, &c) in prev_cost.iter().enumerate() {
        if c < best_final_cost {
            best_final_cost = c;
            best_final = s;
        }
    }

    let sym_mask = (1usize << geo.k) - 1;
    let mut symbols = vec![0u8; num_symbols];
    let mut s = best_final;
    symbols[num_symbols - 1] = (s & sym_mask) as u8;
    for t in (1..num_symbols).rev() {
        let j = backtrace[t - 1][s] as usize;
        let prev_s = (j << shift) | (s >> geo.k);
        symbols[t - 1] = (prev_s & sym_mask) as u8;
        s = prev_s;
    }
    symbols
}

/// Production packing: K=4 → two symbols per byte (low nibble first), K=8 →
/// one symbol per byte. Both are the byte layout the matching decoder reads.
fn pack_symbols_geo(symbols: &[u8], geo: Geo) -> Vec<u8> {
    match geo.k {
        4 => pack_symbols(symbols),
        8 => symbols.to_vec(),
        other => panic!("pack_symbols_geo: no byte layout defined for K={other}"),
    }
}

#[inline]
fn unpack_symbol(packed: &[u8], t: usize, geo: Geo) -> u32 {
    match geo.k {
        4 => {
            let byte = packed[t / 2];
            (if t.is_multiple_of(2) {
                byte & 0x0F
            } else {
                byte >> 4
            }) as u32
        }
        8 => packed[t] as u32,
        other => panic!("unpack_symbol: no byte layout defined for K={other}"),
    }
}

fn decode_packed_geo(packed: &[u8], num_symbols: usize, lut: &[f32], geo: Geo) -> Vec<f32> {
    let v = geo.v as usize;
    let mask = (1u32 << geo.l) - 1;
    let mut state: u32 = 0;
    let mut out = Vec::with_capacity(num_symbols * v);
    for t in 0..num_symbols {
        state = ((state << geo.k) | unpack_symbol(packed, t, geo)) & mask;
        let off = (state as usize) * v;
        out.extend_from_slice(&lut[off..off + v]);
    }
    out
}

fn ls_refine_scale_geo(
    w_rotated: &[f32],
    packed: &[u8],
    num_symbols: usize,
    lut: &[f32],
    heur: f32,
    geo: Geo,
) -> f32 {
    let v = geo.v as usize;
    let mask = (1u32 << geo.l) - 1;
    let mut state: u32 = 0;
    let mut dot_wl = 0f32;
    let mut dot_ll = 0f32;
    for t in 0..num_symbols {
        state = ((state << geo.k) | unpack_symbol(packed, t, geo)) & mask;
        let off = (state as usize) * v;
        // Accumulate the V terms of this symbol first, then add once — the
        // production `ls_refine_scale` does `dot += w0*l0 + w1*l1`, and a
        // per-term `+=` would round differently (measured: 4 ULP on f32).
        let mut wl_t = 0f32;
        let mut ll_t = 0f32;
        for i in 0..v {
            let c = lut[off + i];
            wl_t += w_rotated[t * v + i] * c;
            ll_t += c * c;
        }
        dot_wl += wl_t;
        dot_ll += ll_t;
    }
    if dot_ll > 0.0 && dot_wl / dot_ll > 0.0 {
        dot_wl / dot_ll
    } else {
        heur
    }
}

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
    /// Trellis geometry. [`GEO_SHIPPED`] reproduces the historical numbers
    /// byte-for-byte.
    geo: Geo,
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
    // NOTE: the Gaussian codebook, NOT the `ARC_QTIP_CODEBOOK=mcg` computed
    // codebook the shipped artifact was baked with. See the module tests.
    let lut = gaussian_lut_geo(cfg.geo.l, cfg.geo.v);
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
                viterbi_quantize_row_geo(&target, &lut, cfg.geo)
            } else {
                greedy_quantize_row_geo(&target, &lut, cfg.geo)
            };
            let packed = pack_symbols_geo(&syms, cfg.geo);
            let num_symbols = k / cfg.geo.v as usize;
            // 4. Final scale.
            let final_scale = match cfg.policy {
                ScalePolicy::MaxOver3 => search_scale,
                ScalePolicy::MaxOver3LsRefine | ScalePolicy::RmsMatchedLsRefine => {
                    ls_refine_scale_geo(&rot, &packed, num_symbols, &lut, search_scale, cfg.geo)
                }
            };
            // 5. Decode + un-rotate (D·H·D is involutory).
            let mut recon: Vec<f32> = decode_packed_geo(&packed, num_symbols, &lut, cfg.geo)
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
                geo: GEO_SHIPPED,
            },
        ),
        (
            "B greedy      rot=128 max/3           ",
            Cfg {
                viterbi: false,
                rotation_block: 128,
                policy: ScalePolicy::MaxOver3,
                geo: GEO_SHIPPED,
            },
        ),
        (
            "C viterbi     rot=0   max/3           ",
            Cfg {
                viterbi: true,
                rotation_block: 0,
                policy: ScalePolicy::MaxOver3,
                geo: GEO_SHIPPED,
            },
        ),
        (
            "D env-path    vit rot=128 max/3       ",
            Cfg {
                viterbi: true,
                rotation_block: 128,
                policy: ScalePolicy::MaxOver3,
                geo: GEO_SHIPPED,
            },
        ),
        (
            "E D+LSrefine  vit rot=128 max/3+LS    ",
            Cfg {
                viterbi: true,
                rotation_block: 128,
                policy: ScalePolicy::MaxOver3LsRefine,
                geo: GEO_SHIPPED,
            },
        ),
        (
            "F rms-scale   vit rot=128 rms+LS      ",
            Cfg {
                viterbi: true,
                rotation_block: 128,
                policy: ScalePolicy::RmsMatchedLsRefine,
                geo: GEO_SHIPPED,
            },
        ),
        (
            "G full-Had    vit rot=2048 max/3+LS   ",
            Cfg {
                viterbi: true,
                rotation_block: 2048,
                policy: ScalePolicy::MaxOver3LsRefine,
                geo: GEO_SHIPPED,
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

// ---------------------------------------------------------------------------
// Trellis-geometry sweep (K8/V4/L12 GO-NO-GO)
// ---------------------------------------------------------------------------

/// **Anti-silent-success guard 1.** The geometry-parameterised LUT must
/// reproduce the shipped `gaussian_lut()` at (L=16, V=2), or the control
/// column of the sweep is not the control at all.
///
/// It reproduces it to **≤1 ULP**, not exactly: `box_muller` ends in
/// `theta.sin()` / `theta.cos()`, and LLVM lowers that pair differently in the
/// two (identical-source) compiled copies — measured 1,534 of 131,072 entries
/// off by exactly 1 ULP (~6e-8 relative), zero off by more. That is a property
/// of the *existing* `gaussian_lut`, not of the sweep: it means the shipped
/// Gaussian codebook is only reproducible to 1 ULP across compilations.
/// What matters for the sweep is that this noise cannot move a decision — so
/// the symbol streams are asserted byte-identical under both LUTs.
#[test]
fn gaussian_lut_geo_matches_shipped_lut() {
    let want = gaussian_lut();
    let got = gaussian_lut_geo(L, V);
    assert_eq!(got.len(), want.len());
    let mut worst_ulp = 0i64;
    let mut n_diff = 0usize;
    for (&a, &b) in got.iter().zip(want.iter()) {
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
            // Monotone f32 ordinal distance (both operands are finite, and no
            // pair straddles zero at 1 ULP unless both are denormal).
            let ord = |x: f32| -> i64 {
                let bits = x.to_bits() as i64;
                if x.is_sign_negative() {
                    -(bits & 0x7FFF_FFFF)
                } else {
                    bits
                }
            };
            worst_ulp = worst_ulp.max((ord(a) - ord(b)).abs());
        }
    }
    println!(
        "gaussian_lut_geo vs gaussian_lut: {n_diff}/{} entries differ, worst {worst_ulp} ULP",
        want.len()
    );
    let worst_rel = got
        .iter()
        .zip(want.iter())
        .map(|(&a, &b)| {
            let d = (a as f64 - b as f64).abs();
            let m = (a.abs() as f64).max(b.abs() as f64);
            if m > 0.0 {
                d / m
            } else {
                0.0
            }
        })
        .fold(0f64, f64::max);
    println!("  worst relative delta {worst_rel:.3e}");
    assert!(
        worst_ulp <= 4 && worst_rel < 1e-6,
        "gaussian_lut_geo(L,V) diverged from gaussian_lut() by {worst_ulp} ULP / {worst_rel:.3e} \
         relative — that is a real algorithmic difference, not sin/cos lowering noise"
    );

    // The 1-ULP noise must not move a single trellis decision.
    let k = 512;
    let signs = generate_signs(QTIP_ROTATION_SEED, k);
    for (name, w) in [
        ("gaussian", gen_gaussian(4, k, 0.02, 21)),
        ("fp4_dequant", gen_fp4_dequant(4, k, 0.02, 22)),
    ] {
        for row in 0..4 {
            let mut rot = w[row * k..(row + 1) * k].to_vec();
            apply_block_rotation(&mut rot, &signs, 128);
            let max_abs = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 3.0 };
            let target: Vec<f32> = rot.iter().map(|&v| v / scale).collect();
            assert_eq!(
                viterbi_quantize_row(&target, &want),
                viterbi_quantize_row(&target, &got),
                "{name} row {row}: 1-ULP LUT noise changed the Viterbi symbol stream"
            );
        }
    }

    // The geometries under test really do produce different codebooks.
    for geo in [Geo { l: 12, k: 4, v: 2 }, GEO_CANDIDATE] {
        let other = gaussian_lut_geo(geo.l, geo.v);
        assert_ne!(
            other.len(),
            got.len(),
            "{} produced a shipped-sized LUT",
            geo.label()
        );
    }
    // Every geometry's LUT is unit-sigma (the max/3 scale policy's assumption).
    for geo in [GEO_SHIPPED, Geo { l: 14, k: 8, v: 4 }, GEO_CANDIDATE] {
        let lut = gaussian_lut_geo(geo.l, geo.v);
        let sigma =
            (lut.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / lut.len() as f64).sqrt();
        println!("{} LUT sigma {sigma:.5}", geo.label());
        assert!(
            (sigma - 1.0).abs() < 0.03,
            "{} LUT sigma {sigma:.4} is not ~1.0; the max/3 scale policy would be mis-set",
            geo.label()
        );
    }
}

/// **Anti-silent-success guard 2.** At [`GEO_SHIPPED`] every geometry-lifted
/// stage must be byte-identical to the production stage it generalises, so a
/// difference in the sweep is attributable to the geometry and nothing else.
#[test]
fn geo_pipeline_is_byte_identical_at_shipped_geometry() {
    // One LUT for both sides on purpose: this test pins the geometry-lifted
    // *code*. The codebook generators are compared separately in
    // `gaussian_lut_geo_matches_shipped_lut`.
    let lut = gaussian_lut();
    let geo_lut = &lut;
    let k = 512;
    let num_symbols = k / V as usize;
    let signs = generate_signs(QTIP_ROTATION_SEED, k);
    for (name, w) in [
        ("gaussian", gen_gaussian(4, k, 0.02, 7)),
        ("student_t4", gen_student_t(4, k, 0.02, 8)),
        ("fp4_dequant", gen_fp4_dequant(4, k, 0.02, 9)),
    ] {
        for row in 0..4 {
            let mut rot = w[row * k..(row + 1) * k].to_vec();
            apply_block_rotation(&mut rot, &signs, 128);
            let max_abs = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 3.0 };
            let target: Vec<f32> = rot.iter().map(|&v| v / scale).collect();

            let v_prod = viterbi_quantize_row(&target, &lut);
            let v_geo = viterbi_quantize_row_geo(&target, geo_lut, GEO_SHIPPED);
            assert_eq!(v_prod, v_geo, "{name} row {row}: geo Viterbi diverged");

            let g_prod = greedy_quantize_row(&target, &lut);
            let g_geo = greedy_quantize_row_geo(&target, geo_lut, GEO_SHIPPED);
            assert_eq!(g_prod, g_geo, "{name} row {row}: geo greedy diverged");

            let p_prod = pack_symbols(&v_prod);
            let p_geo = pack_symbols_geo(&v_geo, GEO_SHIPPED);
            assert_eq!(p_prod, p_geo, "{name} row {row}: geo packing diverged");

            let d_prod = decode_packed(&p_prod, num_symbols, &lut);
            let d_geo = decode_packed_geo(&p_geo, num_symbols, geo_lut, GEO_SHIPPED);
            assert_eq!(d_prod, d_geo, "{name} row {row}: geo decode diverged");

            let s_prod = ls_refine_scale(&rot, &p_prod, num_symbols, &lut, scale);
            let s_geo = ls_refine_scale_geo(&rot, &p_geo, num_symbols, geo_lut, scale, GEO_SHIPPED);
            assert_eq!(
                s_prod.to_bits(),
                s_geo.to_bits(),
                "{name} row {row}: geo LS-refine diverged"
            );
        }
    }
}

/// **Anti-silent-success guard 3.** Two geometries must not produce the same
/// reconstruction. A threaded-but-ignored parameter would return an identical
/// number for every row and read as "no quality loss".
#[test]
fn geometry_actually_changes_the_reconstruction() {
    let (n, k) = (4usize, 512usize);
    let w = gen_fp4_dequant(n, k, 0.02, 11);
    let base = Cfg {
        viterbi: true,
        rotation_block: 128,
        policy: ScalePolicy::MaxOver3,
        geo: GEO_SHIPPED,
    };
    let a = quantize_matrix(&w, n, k, base);
    for geo in [
        Geo { l: 14, k: 4, v: 2 },
        Geo { l: 12, k: 4, v: 2 },
        Geo { l: 16, k: 8, v: 4 },
        Geo { l: 14, k: 8, v: 4 },
        GEO_CANDIDATE,
    ] {
        let b = quantize_matrix(&w, n, k, Cfg { geo, ..base });
        assert_eq!(a.len(), b.len());
        let differing = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        assert!(
            differing * 10 > a.len(),
            "{} reconstruction is ~identical to the shipped geometry ({differing}/{} weights \
             differ) — the geometry parameter is not reaching the search",
            geo.label(),
            a.len()
        );
    }
}

/// **The GO/NO-GO evidence table.** Trellis geometry × fixture at the
/// production search config (Viterbi + Hadamard rot=128 + max/3), which is
/// what `QtipMode::default_expert_mode()` bakes today.
///
/// Codebook: the **Gaussian** LUT (`QtipCodebook::DEFAULT`), NOT the
/// `ARC_QTIP_CODEBOOK=mcg` computed codebook the shipped V4 artifact was baked
/// with. Deltas here are Gaussian-vs-Gaussian, which is the correct
/// controlled comparison for a geometry decision; they are NOT directly
/// comparable to an mcg-baked artifact's absolute numbers.
///
/// Run: `cargo test -p mistralrs-quant --release probe_trellis_geometry_ladder
/// -- --ignored --nocapture`
#[test]
#[ignore = "evidence-gathering probe (slow); run with --ignored --nocapture --release"]
fn probe_trellis_geometry_ladder() {
    let n = 48;
    let k = 2048;
    let sigma = 0.02;

    let fixtures: Vec<(&str, Vec<f32>)> = vec![
        ("gaussian   ", gen_gaussian(n, k, sigma, 1)),
        ("student_t4 ", gen_student_t(n, k, sigma, 2)),
        ("fp4_dequant", gen_fp4_dequant(n, k, sigma, 3)),
    ];

    // Ordered by descending trellis memory depth (L/K).
    let geometries = [
        GEO_SHIPPED,               // depth 4.00 — the control
        Geo { l: 14, k: 4, v: 2 }, // depth 3.50
        Geo { l: 12, k: 4, v: 2 }, // depth 3.00 — isolates pure LUT-size loss
        Geo { l: 16, k: 8, v: 4 }, // depth 2.00 — isolates the K/V change
        Geo { l: 14, k: 8, v: 4 }, // depth 1.75
        GEO_CANDIDATE,             // depth 1.50 — the candidate
    ];

    let base = Cfg {
        viterbi: true,
        rotation_block: 128,
        policy: ScalePolicy::MaxOver3,
        geo: GEO_SHIPPED,
    };

    println!("\n=== trellis geometry ladder (n={n}, k={k}, sigma={sigma}, viterbi+rot128+max/3, Gaussian codebook) ===");
    println!("geometry                | fixture     | w_cos   | mm_cos  | pplx   | d(mm_cos) vs K4/V2/L16");
    for (fname, w) in &fixtures {
        let mut control_mm: Option<f64> = None;
        let mut control_recon: Option<Vec<f32>> = None;
        for geo in geometries {
            let w_hat = quantize_matrix(w, n, k, Cfg { geo, ..base });
            let m = evaluate(w, &w_hat, n, k);
            let delta = match control_mm {
                None => {
                    control_mm = Some(m.matmul_cos);
                    control_recon = Some(w_hat.clone());
                    0.0
                }
                Some(c) => m.matmul_cos - c,
            };
            // Guard: this geometry's reconstruction must actually differ from
            // the control's, otherwise the row is a silent-success artifact.
            if let Some(ctrl) = &control_recon {
                if geo != GEO_SHIPPED {
                    let differing = ctrl
                        .iter()
                        .zip(w_hat.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                    assert!(
                        differing * 10 > w_hat.len(),
                        "{} on {fname}: only {differing}/{} weights differ from the control — \
                         the geometry parameter is not reaching the search",
                        geo.label(),
                        w_hat.len()
                    );
                }
            }
            println!(
                "{:<23} | {fname} | {:.5} | {:.5} | {:.4} | {:+.5}",
                geo.label(),
                m.weight_cos,
                m.matmul_cos,
                m.ppl_ratio,
                delta
            );
        }
        println!("------");
    }
}

/// **Noise floor for the geometry ladder.** `probe_trellis_geometry_ladder`
/// reports one weight draw per fixture; this repeats control-vs-geometry over
/// several draws so the reported delta can be compared against its own spread
/// rather than against zero.
#[test]
#[ignore = "evidence-gathering probe (slow); run with --ignored --nocapture --release"]
fn probe_geometry_delta_noise() {
    let n = 48;
    let k = 2048;
    let sigma = 0.02;
    let seeds = [1u64, 101, 202, 303, 404];
    let base = Cfg {
        viterbi: true,
        rotation_block: 128,
        policy: ScalePolicy::MaxOver3,
        geo: GEO_SHIPPED,
    };
    let geometries = [
        Geo { l: 14, k: 4, v: 2 },
        Geo { l: 12, k: 4, v: 2 },
        Geo { l: 16, k: 8, v: 4 },
        Geo { l: 14, k: 8, v: 4 },
        GEO_CANDIDATE,
    ];

    println!(
        "\n=== geometry delta over {} weight draws (n={n}, k={k}) ===",
        seeds.len()
    );
    println!("geometry                | fixture     | mean d(w_cos)  [min,max]        | mean d(mm_cos) [min,max]");
    for (fname, gen) in [
        (
            "gaussian   ",
            gen_gaussian as fn(usize, usize, f64, u64) -> Vec<f32>,
        ),
        ("student_t4 ", gen_student_t),
        ("fp4_dequant", gen_fp4_dequant),
    ] {
        // Control metrics per seed.
        let ctrl: Vec<(f64, f64)> = seeds
            .iter()
            .map(|&s| {
                let w = gen(n, k, sigma, s);
                let m = evaluate(&w, &quantize_matrix(&w, n, k, base), n, k);
                (m.weight_cos, m.matmul_cos)
            })
            .collect();
        for geo in geometries {
            let mut dw = Vec::new();
            let mut dm = Vec::new();
            for (i, &s) in seeds.iter().enumerate() {
                let w = gen(n, k, sigma, s);
                let m = evaluate(&w, &quantize_matrix(&w, n, k, Cfg { geo, ..base }), n, k);
                dw.push(m.weight_cos - ctrl[i].0);
                dm.push(m.matmul_cos - ctrl[i].1);
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let lo = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "{:<23} | {fname} | {:+.5} [{:+.5},{:+.5}] | {:+.5} [{:+.5},{:+.5}]",
                geo.label(),
                mean(&dw),
                lo(&dw),
                hi(&dw),
                mean(&dm),
                lo(&dm),
                hi(&dm)
            );
        }
        println!("------");
    }
}
