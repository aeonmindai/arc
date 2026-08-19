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
    // NOTE: the Gaussian codebook, NOT the `ARC_QTIP_CODEBOOK=mcg` computed
    // codebook the shipped artifact was baked with. See the module tests.
    let lut = gaussian_lut_geo(cfg.geo.l, cfg.geo.v);
    quantize_matrix_with_lut(w, n, k, cfg, &lut)
}

/// [`quantize_matrix`] with the codebook supplied instead of generated, so a
/// *designed* table can be measured against the pseudorandom Gaussian one on
/// exactly the same pipeline. `quantize_matrix` is the `Book::Gaussian` case of
/// this function and nothing else — the split is a pure refactor, pinned by
/// `geo_pipeline_is_byte_identical_at_shipped_geometry` and by
/// `designed_codebook_reaches_the_search`.
fn quantize_matrix_with_lut(w: &[f32], n: usize, k: usize, cfg: Cfg, lut: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    assert_eq!(
        lut.len(),
        cfg.geo.lut_size() * cfg.geo.v as usize,
        "codebook is not 2^L x V for {}",
        cfg.geo.label()
    );
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
                viterbi_quantize_row_geo(&target, lut, cfg.geo)
            } else {
                greedy_quantize_row_geo(&target, lut, cfg.geo)
            };
            let packed = pack_symbols_geo(&syms, cfg.geo);
            let num_symbols = k / cfg.geo.v as usize;
            // 4. Final scale.
            let final_scale = match cfg.policy {
                ScalePolicy::MaxOver3 => search_scale,
                ScalePolicy::MaxOver3LsRefine | ScalePolicy::RmsMatchedLsRefine => {
                    ls_refine_scale_geo(&rot, &packed, num_symbols, lut, search_scale, cfg.geo)
                }
            };
            // 5. Decode + un-rotate (D·H·D is involutory).
            let mut recon: Vec<f32> = decode_packed_geo(&packed, num_symbols, lut, cfg.geo)
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

// ---------------------------------------------------------------------------
// Codebook DESIGN (wave: is the K8/V4/L12 loss a geometry limit or a
// random-codebook limit?)
// ---------------------------------------------------------------------------
//
// The geometry sweep drew every codebook at random (Box-Muller per state), so
// at K8/V4/L12 it scattered 4096 points at random through 4-D space. That
// bounds the loss of a *random* codebook at that geometry, not the geometry's
// own limit. Everything below designs the table instead.
//
// Trellis block structure (the thing a design has to respect): from state `s`
// the reachable next states are `((s << K) | j) & mask` for `j` in
// `0..2^K`, i.e. the contiguous block `{ (g << K) | j }` with
// `g = s mod 2^(L-K)`. So the table is `2^(L-K)` *blocks* of `2^K` entries,
// block index `state >> K`, and the encoder picks one entry per block visit.
// Each block must therefore be a good `2^K`-point V-dim quantizer on its own,
// and the blocks should differ from each other or the trellis memory is dead.
// At K8/V4/L12 that is 16 blocks x 256 entries; at K4/V2/L16, 4096 blocks x 16.

/// A codebook *design*: the map from trellis state to its `V`-dimensional
/// reproduction vector.
///
/// **Every variant here is a stored `2^L x V` table.** They all decode with the
/// identical instruction sequence — one indexed load of `V` contiguous bf16 —
/// so a design change costs nothing at inference time and only changes what the
/// offline bake writes. At K8/V4/L12 the table is `4096 * 4 * 2 B = 32,768 B`,
/// exactly the shared-memory budget that made the geometry worth 4.375 inst/wt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Book {
    /// Box-Muller pseudorandom Gaussian, amplitude untouched: `QtipCodebook::
    /// DEFAULT` and exactly what the geometry sweep measured. The control.
    Gaussian,
    /// Gaussian *shape*, amplitude re-fitted to the search targets. The
    /// cheapest possible "design"; separates a pure scale-calibration win from
    /// a real geometry win.
    GaussianCalibrated,
    /// Best memoryless `2^K`-point V-dim VQ (LBG on the search targets),
    /// replicated into every block. All blocks identical => the trellis memory
    /// is switched off on purpose. This is the "no shaping at all" floor.
    LbgMemoryless,
    /// One *random* block replicated into every block. The paired control for
    /// [`Book::LbgMemoryless`]: with the trellis switched off, LBG must beat
    /// random by a wide margin, or the LBG training is broken and every
    /// "designed loses" row below is an artifact rather than a result.
    GaussianMemoryless,
    /// Two-stage LBG with Ungerboeck set partitioning: `2^K` coarse cells, each
    /// split into `2^(L-K)` sub-centroids; block `g` collects the `g`-th
    /// sub-centroid of every coarse cell. Each block is therefore a full-cover
    /// `2^K`-point quantizer, and the `2^(L-K)` blocks are dithered variants of
    /// one another — the textbook trellis-coded-quantizer construction.
    LbgPartitioned,
    /// One LBG `2^K`-point codebook, each block carrying a different random
    /// 4-D rotation of it. Every block is therefore an *optimal-quality*
    /// `2^K`-point VQ (unlike the random table's 256 scattered points) whose
    /// reach is still the whole space (unlike the set partition, whose 16
    /// alternatives for a symbol all sit inside one coarse cell). This is the
    /// designed structure closest to what the random table accidentally has.
    LbgRotated,
    /// The same centroids, but each block is a contiguous clump of whole coarse
    /// cells instead of a spread. **Negative control**: a partition that
    /// destroys per-block coverage must measure clearly worse, which proves the
    /// partition (not just the centroid values) reaches the search.
    LbgClustered,
    /// D4 — the densest packing and best known quantizer in 4 dimensions
    /// (normalised second moment 0.076603 vs 1/12 for the cubic lattice) —
    /// partitioned into its 16 cosets of 2*D4, which is exactly `2^(L-K)` at
    /// K8/V4/L12. Intra-block minimum distance is doubled by construction.
    D4Coset,
    /// Generalized Lloyd with the *Viterbi trellis encoder* in the loop: encode
    /// the training stream, then move each state's codeword to the centroid of
    /// the targets that landed on it. This is the design that optimises the
    /// thing actually being measured, so it upper-bounds the others.
    TrellisLloyd,
    /// The same generalized Lloyd, but started *from the random Gaussian table*
    /// instead of from a designed one. This is the sharpest question available:
    /// if the exact objective cannot be improved by gradient descent from the
    /// random table, the random table is at a local optimum and there is no
    /// design left to find.
    TrellisLloydFromGaussian,
}

impl Book {
    fn label(self) -> &'static str {
        match self {
            Book::Gaussian => "gaussian (random)",
            Book::GaussianCalibrated => "gaussian + scale fit",
            Book::LbgMemoryless => "LBG memoryless VQ",
            Book::GaussianMemoryless => "random memoryless VQ",
            Book::LbgRotated => "LBG + per-block rot",
            Book::LbgPartitioned => "LBG set-partitioned",
            Book::LbgClustered => "LBG clustered (neg)",
            Book::D4Coset => "D4 lattice cosets",
            Book::TrellisLloyd => "trellis Lloyd <- LBG",
            Book::TrellisLloydFromGaussian => "trellis Lloyd <- random",
        }
    }
    /// Whether the amplitude is re-fitted on held-out calibration data.
    fn calibrated(self) -> bool {
        !matches!(self, Book::Gaussian)
    }
}

/// `(name, generator)` for the three weight fixtures, named so the array type
/// stays readable.
type Fixture = (&'static str, fn(usize, usize, f64, u64) -> Vec<f32>);

/// Rotate and scale exactly as `quantize_matrix` steps 1-2 do, returning the
/// stream of search targets the trellis actually sees. Codebooks are designed
/// on **this** distribution, not on the raw weights — that is what "trained on
/// the actual target distribution" has to mean here.
fn search_targets(w: &[f32], n: usize, k: usize, rotation_block: usize) -> Vec<f32> {
    let signs = if rotation_block >= 2 {
        generate_signs(QTIP_ROTATION_SEED, k)
    } else {
        Vec::new()
    };
    let mut out = Vec::with_capacity(n * k);
    for row in 0..n {
        let mut rot = w[row * k..(row + 1) * k].to_vec();
        if rotation_block >= 2 {
            apply_block_rotation(&mut rot, &signs, rotation_block);
        }
        let max_abs = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = if max_abs == 0.0 { 1.0 } else { max_abs / 3.0 };
        let inv = 1.0 / s;
        out.extend(rot.iter().map(|&v| v * inv));
    }
    out
}

/// The codebook training stream: the mixture of all three fixtures, on seeds
/// disjoint from every evaluation seed. One table ships for all weights, so one
/// table is trained on all of them.
fn codebook_training_targets(rows_per_fixture: usize, k: usize) -> Vec<f32> {
    let sigma = 0.02;
    let mut t = Vec::new();
    for (i, gen) in [
        gen_gaussian as fn(usize, usize, f64, u64) -> Vec<f32>,
        gen_student_t,
        gen_fp4_dequant,
    ]
    .into_iter()
    .enumerate()
    {
        let w = gen(rows_per_fixture, k, sigma, 90_001 + i as u64);
        t.extend(search_targets(&w, rows_per_fixture, k, 128));
    }
    t
}

/// Held-out matrix for amplitude calibration: disjoint from both the training
/// seeds and the evaluation seeds.
fn calibration_matrix(n_per_fixture: usize, k: usize) -> (Vec<f32>, usize) {
    let sigma = 0.02;
    let mut w = Vec::new();
    for (i, gen) in [
        gen_gaussian as fn(usize, usize, f64, u64) -> Vec<f32>,
        gen_student_t,
        gen_fp4_dequant,
    ]
    .into_iter()
    .enumerate()
    {
        w.extend(gen(n_per_fixture, k, sigma, 70_001 + i as u64));
    }
    (w, n_per_fixture * 3)
}

#[inline]
fn nearest_centroid(cents: &[f32], v: usize, x: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for c in 0..cents.len() / v {
        let off = c * v;
        let mut d = 0f32;
        for i in 0..v {
            let t = cents[off + i] - x[i];
            d += t * t;
        }
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    (best, best_d)
}

/// Linde-Buzo-Gray / generalized Lloyd VQ design. Deterministic: the parallel
/// accumulation collects per-chunk partials and folds them in index order, so
/// the result does not depend on rayon's scheduling.
fn lbg(train: &[f32], v: usize, ncent: usize, iters: usize, seed: u64) -> Vec<f32> {
    use rayon::prelude::*;
    let nvec = train.len() / v;
    assert!(
        nvec >= ncent * 8,
        "LBG needs >= 8 training vectors per centroid (have {nvec} for {ncent})"
    );
    let mut rng = Rng::new(seed);
    let mut cents = vec![0f32; ncent * v];
    for c in 0..ncent {
        let idx = (rng.next_u64() % nvec as u64) as usize;
        cents[c * v..(c + 1) * v].copy_from_slice(&train[idx * v..(idx + 1) * v]);
    }
    let chunk = (nvec / 256).max(1024) * v;
    for _ in 0..iters {
        let partials: Vec<(Vec<f64>, Vec<u32>)> = train
            .par_chunks(chunk)
            .map(|ch| {
                let mut s = vec![0f64; ncent * v];
                let mut n = vec![0u32; ncent];
                for x in ch.chunks_exact(v) {
                    let (c, _) = nearest_centroid(&cents, v, x);
                    n[c] += 1;
                    for i in 0..v {
                        s[c * v + i] += x[i] as f64;
                    }
                }
                (s, n)
            })
            .collect();
        let mut sums = vec![0f64; ncent * v];
        let mut counts = vec![0u32; ncent];
        for (s, n) in &partials {
            for i in 0..sums.len() {
                sums[i] += s[i];
            }
            for i in 0..counts.len() {
                counts[i] += n[i];
            }
        }
        let big = (0..ncent).max_by_key(|&i| counts[i]).unwrap_or(0);
        for c in 0..ncent {
            if counts[c] > 0 {
                for i in 0..v {
                    cents[c * v + i] = (sums[c * v + i] / counts[c] as f64) as f32;
                }
            } else {
                // Empty cell: re-seed by splitting the most-populated one.
                for i in 0..v {
                    let jitter = ((rng.uniform() - 0.5) * 0.05) as f32;
                    cents[c * v + i] =
                        (sums[big * v + i] / counts[big].max(1) as f64) as f32 + jitter;
                }
            }
        }
    }
    cents
}

/// Two-stage LBG: `ncoarse` cells, each refined into `nfine` sub-centroids.
/// Returns `(coarse, fine)` with `fine[(c * nfine + f) * v ..]` the `f`-th
/// sub-centroid of coarse cell `c`.
fn two_stage_lbg(
    train: &[f32],
    v: usize,
    ncoarse: usize,
    nfine: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;
    let coarse = lbg(train, v, ncoarse, 30, seed);
    // Bucket the training stream by coarse cell (deterministic order).
    let assign: Vec<usize> = train
        .par_chunks_exact(v)
        .map(|x| nearest_centroid(&coarse, v, x).0)
        .collect();
    let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); ncoarse];
    for (x, &c) in train.chunks_exact(v).zip(assign.iter()) {
        buckets[c].extend_from_slice(x);
    }
    let fine: Vec<Vec<f32>> = buckets
        .par_iter()
        .enumerate()
        .map(|(c, b)| {
            if b.len() / v >= nfine * 8 {
                lbg(
                    b,
                    v,
                    nfine,
                    20,
                    seed ^ (0x5DEE_CE66u64.wrapping_mul(c as u64 + 1)),
                )
            } else {
                // Too few samples to split honestly: jitter the coarse centroid
                // deterministically so the block still differs from its peers.
                let mut r = Rng::new(seed ^ (c as u64) << 17 ^ 0xB5);
                let mut out = vec![0f32; nfine * v];
                for f in 0..nfine {
                    for i in 0..v {
                        out[f * v + i] = coarse[c * v + i] + ((r.uniform() - 0.5) * 0.2) as f32;
                    }
                }
                out
            }
        })
        .collect();
    (coarse, fine.into_iter().flatten().collect())
}

/// D4 = { x in Z^4 : sum(x) even }, partitioned into the 16 cosets of 2*D4.
/// Each coset keeps its `2^K` smallest-norm points, so every trellis block is a
/// sphere-bounded lattice quantizer with minimum distance 2*sqrt(2) (twice D4's
/// own), while the union of blocks tiles at D4's density.
fn d4_coset_lut(geo: Geo) -> Vec<f32> {
    assert_eq!(geo.v, 4, "D4 is a 4-D lattice");
    let per_block = 1usize << geo.k;
    let nblocks = 1usize << (geo.l - geo.k);
    assert_eq!(
        nblocks, 16,
        "the 2*D4 coset partition has exactly 16 classes"
    );
    // Coset label: x = b + 2y with b = x mod 2 (even weight, 8 patterns);
    // x - x' in 2*D4  <=>  b == b' and sum(y) == sum(y') mod 2. 8 * 2 = 16.
    let mut pat_idx = [usize::MAX; 16];
    let mut np = 0usize;
    for p in 0u32..16 {
        if p.count_ones() % 2 == 0 {
            pat_idx[p as usize] = np;
            np += 1;
        }
    }
    assert_eq!(np, 8);
    let r = 12i32;
    let mut cosets: Vec<Vec<(i64, [i32; 4])>> = vec![Vec::new(); nblocks];
    for a in -r..=r {
        for b in -r..=r {
            for c in -r..=r {
                for d in -r..=r {
                    if (a + b + c + d) % 2 != 0 {
                        continue;
                    }
                    let x = [a, b, c, d];
                    let mut pat = 0u32;
                    let mut ysum = 0i32;
                    for (i, &xi) in x.iter().enumerate() {
                        let bi = xi.rem_euclid(2);
                        pat |= (bi as u32) << i;
                        ysum += (xi - bi) / 2;
                    }
                    let label = pat_idx[pat as usize] * 2 + ysum.rem_euclid(2) as usize;
                    let n2 = x.iter().map(|&v| (v as i64) * (v as i64)).sum::<i64>();
                    cosets[label].push((n2, x));
                }
            }
        }
    }
    let mut lut = vec![0f32; geo.lut_size() * geo.v as usize];
    for (g, pts) in cosets.iter_mut().enumerate() {
        // Deterministic: norm first, then lexicographic.
        pts.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        assert!(
            pts.len() >= per_block,
            "coset {g} has only {} points, need {per_block}",
            pts.len()
        );
        for (j, (_, x)) in pts.iter().take(per_block).enumerate() {
            let state = (g << geo.k) | j;
            for i in 0..4 {
                lut[state * 4 + i] = x[i] as f32;
            }
        }
    }
    lut
}

/// Generalized Lloyd with the Viterbi trellis encoder in the loop: encode the
/// training rows with the current table, then move each state's codeword to the
/// centroid of the targets that landed on it. States with few visits are
/// shrunk toward the initial table (ridge `lambda`) so a rarely-reached state
/// cannot be driven by two samples.
/// Returns the trained table and the per-sweep training MSE *before* that
/// sweep's update. The trace is not decoration: if it does not fall, the
/// descent is not running and a flat result would be a silent success rather
/// than a converged optimum.
fn trellis_lloyd(rows: &[&[f32]], geo: Geo, init: &[f32], iters: usize) -> (Vec<f32>, Vec<f64>) {
    use rayon::prelude::*;
    let v = geo.v as usize;
    let n_states = geo.lut_size();
    let mask = (1u32 << geo.l) - 1;
    let lambda = 8.0f64;
    let mut lut = init.to_vec();
    let mut trace = Vec::with_capacity(iters + 1);
    for it in 0..=iters {
        let partials: Vec<(Vec<f64>, Vec<u32>, f64, u64)> = rows
            .par_chunks(16)
            .map(|chunk| {
                let mut s = vec![0f64; n_states * v];
                let mut n = vec![0u32; n_states];
                let mut sse = 0f64;
                let mut cnt = 0u64;
                for row in chunk {
                    let syms = viterbi_quantize_row_geo(row, &lut, geo);
                    let mut state: u32 = 0;
                    for (t, &sym) in syms.iter().enumerate() {
                        state = ((state << geo.k) | sym as u32) & mask;
                        let off = state as usize * v;
                        n[state as usize] += 1;
                        for i in 0..v {
                            let x = row[t * v + i] as f64;
                            s[off + i] += x;
                            let d = x - lut[off + i] as f64;
                            sse += d * d;
                        }
                        cnt += v as u64;
                    }
                }
                (s, n, sse, cnt)
            })
            .collect();
        let mut sums = vec![0f64; n_states * v];
        let mut counts = vec![0u32; n_states];
        let mut sse = 0f64;
        let mut cnt = 0u64;
        for (s, n, e, c) in &partials {
            for i in 0..sums.len() {
                sums[i] += s[i];
            }
            for i in 0..counts.len() {
                counts[i] += n[i];
            }
            sse += e;
            cnt += c;
        }
        trace.push(sse / cnt as f64);
        if it == iters {
            break;
        }
        for st in 0..n_states {
            let w = counts[st] as f64;
            for i in 0..v {
                let ridge = init[st * v + i] as f64 * lambda;
                lut[st * v + i] = ((sums[st * v + i] + ridge) / (w + lambda)) as f32;
            }
        }
    }
    (lut, trace)
}

/// Scale a table to unit RMS, so every design meets the `max|row|/3` policy on
/// the same footing as the unit-sigma Gaussian LUT and the amplitude search
/// below starts from a comparable place.
fn normalize_rms(lut: &mut [f32]) {
    let rms = (lut.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / lut.len() as f64).sqrt();
    if rms > 0.0 {
        let inv = (1.0 / rms) as f32;
        for x in lut.iter_mut() {
            *x *= inv;
        }
    }
}

/// Fit the table amplitude on held-out calibration data (coarse grid, then a
/// fine pass around the winner). This is an offline bake decision, so it is
/// legitimate to fit it — but it is fitted on seeds disjoint from every
/// evaluation seed.
fn calibrate_alpha(base: &[f32], geo: Geo, cal_w: &[f32], cal_n: usize, k: usize) -> f32 {
    let cfg = Cfg {
        viterbi: true,
        rotation_block: 128,
        policy: ScalePolicy::MaxOver3,
        geo,
    };
    let score = |alpha: f32| -> f64 {
        let lut: Vec<f32> = base.iter().map(|&c| c * alpha).collect();
        let w_hat = quantize_matrix_with_lut(cal_w, cal_n, k, cfg, &lut);
        cosine(cal_w, &w_hat)
    };
    let mut best_a = 1.0f32;
    let mut best_s = f64::NEG_INFINITY;
    for i in 0..=10 {
        let a = 0.5 + 0.1 * i as f32;
        let s = score(a);
        if s > best_s {
            best_s = s;
            best_a = a;
        }
    }
    let centre = best_a;
    for i in -5i32..=5 {
        let a = centre + 0.02 * i as f32;
        if a <= 0.0 {
            continue;
        }
        let s = score(a);
        if s > best_s {
            best_s = s;
            best_a = a;
        }
    }
    best_a
}

/// Build a designed codebook for `geo`. Returns the table and the fitted
/// amplitude.
///
/// NOTE the LBG-based variants are only affordable where the table is small
/// enough to train: at K8/V4/L12 it is 4096 states x 4 = 16,384 parameters,
/// at K4/V2/L16 it is 131,072. That asymmetry is itself part of the answer —
/// the design lever exists at the candidate geometry and barely exists at the
/// shipped one.
fn build_book(book: Book, geo: Geo, train: &[f32], k: usize, calibrate: bool) -> (Vec<f32>, f32) {
    let v = geo.v as usize;
    let per_block = 1usize << geo.k;
    let nblocks = 1usize << (geo.l - geo.k);
    let mut lut = match book {
        Book::Gaussian | Book::GaussianCalibrated => gaussian_lut_geo(geo.l, geo.v),
        Book::LbgMemoryless | Book::GaussianMemoryless => {
            let c = if book == Book::LbgMemoryless {
                lbg(train, v, per_block, 30, 0xC0DE_B00C)
            } else {
                // Block 0 of the random table: one 2^K-point random codebook.
                gaussian_lut_geo(geo.l, geo.v)[..per_block * v].to_vec()
            };
            let mut lut = vec![0f32; geo.lut_size() * v];
            for g in 0..nblocks {
                for j in 0..per_block {
                    let st = (g << geo.k) | j;
                    lut[st * v..st * v + v].copy_from_slice(&c[j * v..j * v + v]);
                }
            }
            lut
        }
        Book::LbgRotated => {
            let c = lbg(train, v, per_block, 30, 0xC0DE_B00C);
            let mut rng = Rng::new(0x5A17_0B0B);
            let mut lut = vec![0f32; geo.lut_size() * v];
            for g in 0..nblocks {
                // Gram-Schmidt on a Gaussian matrix: a Haar-random rotation.
                let mut r = vec![0f32; v * v];
                for x in r.iter_mut() {
                    *x = rng.normal() as f32;
                }
                for i in 0..v {
                    for j in 0..i {
                        let dot: f32 = (0..v).map(|t| r[i * v + t] * r[j * v + t]).sum();
                        for t in 0..v {
                            r[i * v + t] -= dot * r[j * v + t];
                        }
                    }
                    let nrm: f32 = (0..v)
                        .map(|t| r[i * v + t] * r[i * v + t])
                        .sum::<f32>()
                        .sqrt();
                    for t in 0..v {
                        r[i * v + t] /= nrm;
                    }
                }
                for j in 0..per_block {
                    let st = (g << geo.k) | j;
                    for i in 0..v {
                        lut[st * v + i] = (0..v).map(|t| r[i * v + t] * c[j * v + t]).sum::<f32>();
                    }
                }
            }
            lut
        }
        Book::LbgPartitioned | Book::LbgClustered => {
            let (_, fine) = two_stage_lbg(train, v, per_block, nblocks, 0xC0DE_B00C);
            let mut lut = vec![0f32; geo.lut_size() * v];
            for coarse in 0..per_block {
                for f in 0..nblocks {
                    let src = (coarse * nblocks + f) * v;
                    // Ungerboeck: block = fine index, slot = coarse index, so
                    // every block holds one representative of every coarse cell.
                    // Clustered: walk the centroid list in order, so a block is
                    // a contiguous clump of whole coarse cells.
                    let st = match book {
                        Book::LbgPartitioned => (f << geo.k) | coarse,
                        _ => coarse * nblocks + f,
                    };
                    lut[st * v..st * v + v].copy_from_slice(&fine[src..src + v]);
                }
            }
            lut
        }
        Book::D4Coset => d4_coset_lut(geo),
        Book::TrellisLloyd | Book::TrellisLloydFromGaussian => {
            let mut init = if book == Book::TrellisLloydFromGaussian {
                gaussian_lut_geo(geo.l, geo.v)
            } else {
                let (_, fine) = two_stage_lbg(train, v, per_block, nblocks, 0xC0DE_B00C);
                let mut init = vec![0f32; geo.lut_size() * v];
                for coarse in 0..per_block {
                    for f in 0..nblocks {
                        let src = (coarse * nblocks + f) * v;
                        let st = (f << geo.k) | coarse;
                        init[st * v..st * v + v].copy_from_slice(&fine[src..src + v]);
                    }
                }
                init
            };
            normalize_rms(&mut init);
            // Train at the amplitude the pipeline will actually use, otherwise
            // the encoder in the loop is not the encoder being measured.
            let mut alpha = 1.0f32;
            if calibrate {
                let (cw, cn) = calibration_matrix(4, k);
                alpha = calibrate_alpha(&init, geo, &cw, cn, k);
                for x in init.iter_mut() {
                    *x *= alpha;
                }
            }
            let rows: Vec<&[f32]> = train.chunks_exact(k).collect();
            let (lut, trace) = trellis_lloyd(&rows, geo, &init, 8);
            let fall = 1.0 - trace[trace.len() - 1] / trace[0];
            println!(
                "  {} training MSE {:.6} -> {:.6} ({:+.2}% over 8 sweeps)",
                book.label(),
                trace[0],
                trace[trace.len() - 1],
                -100.0 * fall
            );
            assert!(
                trace[trace.len() - 1] <= trace[0],
                "{}: generalized Lloyd increased the training MSE — the descent is broken",
                book.label()
            );
            // Lloyd output is already at the trained amplitude; re-normalising
            // or re-fitting it here would undo the thing that was optimised.
            return (lut, alpha);
        }
    };
    if !(calibrate && book.calibrated()) {
        if book != Book::Gaussian {
            normalize_rms(&mut lut);
        }
        return (lut, 1.0);
    }
    normalize_rms(&mut lut);
    let (cal_w, cal_n) = calibration_matrix(4, k);
    let alpha = calibrate_alpha(&lut, geo, &cal_w, cal_n, k);
    for x in lut.iter_mut() {
        *x *= alpha;
    }
    (lut, alpha)
}

/// **Anti-silent-success guard.** Every codebook design must produce a
/// different table AND a different reconstruction. If `build_book` returned the
/// same table twice (a partition bug, a copy-paste in the match) or the table
/// never reached the search, every candidate would report the same delta and it
/// would read as a clean result. Five guards failed on unfixed code today; this
/// one is cheap enough to run unignored.
#[test]
fn designed_codebook_reaches_the_search() {
    let k = 2048;
    let geo = GEO_CANDIDATE;
    let train = codebook_training_targets(24, k);
    let books = [
        Book::Gaussian,
        Book::GaussianMemoryless,
        Book::LbgMemoryless,
        Book::LbgRotated,
        Book::LbgPartitioned,
        Book::LbgClustered,
        Book::D4Coset,
    ];
    // Calibration off: this guard is about distinctness, and the amplitude
    // search is the slow part.
    let luts: Vec<(Book, Vec<f32>)> = books
        .iter()
        .map(|&b| (b, build_book(b, geo, &train, k, false).0))
        .collect();
    for (i, (bi, li)) in luts.iter().enumerate() {
        assert_eq!(li.len(), geo.lut_size() * geo.v as usize);
        assert!(
            li.iter().all(|x| x.is_finite()),
            "{} produced a non-finite codeword",
            bi.label()
        );
        for (bj, lj) in luts.iter().skip(i + 1) {
            let same = li.iter().zip(lj.iter()).filter(|(a, b)| a == b).count();
            assert!(
                same * 10 < li.len(),
                "{} and {} are the same table ({same}/{} entries equal) — a design \
                 is not reaching the codebook",
                bi.label(),
                bj.label(),
                li.len()
            );
        }
    }
    // ...and the tables must move the reconstruction, not just the table.
    let (n, kk) = (4usize, 512usize);
    let w = gen_fp4_dequant(n, kk, 0.02, 13);
    let cfg = Cfg {
        viterbi: true,
        rotation_block: 128,
        policy: ScalePolicy::MaxOver3,
        geo,
    };
    let recons: Vec<(Book, Vec<f32>)> = luts
        .iter()
        .map(|(b, l)| {
            // Rebuild at kk: the table is k-independent, only its length matters.
            (*b, quantize_matrix_with_lut(&w, n, kk, cfg, l))
        })
        .collect();
    for (i, (bi, ri)) in recons.iter().enumerate() {
        for (bj, rj) in recons.iter().skip(i + 1) {
            let differing = ri.iter().zip(rj.iter()).filter(|(a, b)| a != b).count();
            assert!(
                differing * 10 > ri.len(),
                "{} and {} reconstruct almost identically ({differing}/{} weights differ) \
                 — the codebook parameter is not reaching the search",
                bi.label(),
                bj.label(),
                ri.len()
            );
        }
    }
}

/// **The answer table.** Does a *designed* codebook at K8/V4/L12 close the
/// -0.0070 w_cos gap that the random-codebook geometry sweep measured?
///
/// Control, metric and config are deliberately identical to
/// `probe_geometry_delta_noise`: Gaussian codebook at K4/V2/L16, Viterbi +
/// Hadamard rot=128 + max/3, n=48, k=2048, 5 weight draws, `w_cos` as the tight
/// metric. `Book::Gaussian` at K8/V4/L12 reproduces that sweep's -0.00698 row,
/// which is the cross-check that this probe and that one are measuring the same
/// thing.
///
/// Run: `cargo test -p mistralrs-quant --release probe_designed_codebook
/// -- --ignored --nocapture`
#[test]
#[ignore = "evidence-gathering probe (slow); run with --ignored --nocapture --release"]
fn probe_designed_codebook_at_k8v4l12() {
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

    let t0 = std::time::Instant::now();
    let train = codebook_training_targets(96, k);
    println!(
        "training stream: {} vectors of dim {} ({} per state at K8/V4/L12)",
        train.len() / GEO_CANDIDATE.v as usize,
        GEO_CANDIDATE.v,
        train.len() / GEO_CANDIDATE.v as usize / GEO_CANDIDATE.lut_size()
    );

    let books = [
        Book::Gaussian,
        Book::GaussianCalibrated,
        Book::GaussianMemoryless,
        Book::LbgMemoryless,
        Book::LbgRotated,
        Book::LbgPartitioned,
        Book::LbgClustered,
        Book::D4Coset,
        Book::TrellisLloyd,
        Book::TrellisLloydFromGaussian,
    ];
    let mut built: Vec<(Geo, Book, Vec<f32>, f32)> = Vec::new();
    for &b in &books {
        let (lut, alpha) = build_book(b, GEO_CANDIDATE, &train, k, true);
        println!(
            "built {:<22} @ {} alpha={alpha:.3} ({:.1}s)",
            b.label(),
            GEO_CANDIDATE.label(),
            t0.elapsed().as_secs_f64()
        );
        built.push((GEO_CANDIDATE, b, lut, alpha));
    }
    // Control-side headroom: the one design that is affordable at L=16 (the
    // 131,072-parameter table cannot be LBG-trained on any sane budget).
    {
        let (lut, alpha) = build_book(Book::GaussianCalibrated, GEO_SHIPPED, &train, k, true);
        println!(
            "built {:<22} @ {} alpha={alpha:.3} ({:.1}s)",
            Book::GaussianCalibrated.label(),
            GEO_SHIPPED.label(),
            t0.elapsed().as_secs_f64()
        );
        built.push((GEO_SHIPPED, Book::GaussianCalibrated, lut, alpha));
    }
    // The *other* geometry that fits 32,768 B of shared memory as a stored bf16
    // table: K4/V2/L13 (8192 states x 2 x 2 B). Different rung, not a rescue of
    // K8/V4/L12 — compiled at 11.250 inst/wt vs the shipped 15.125, so 1.34x
    // rather than 3.46x. Priced here with the random codebook only, because it
    // is nearly free to add and the ladder had no L13 row.
    {
        let geo13 = Geo { l: 13, k: 4, v: 2 };
        assert_eq!(geo13.lut_size() * geo13.v as usize * 2, 32_768);
        let (lut, alpha) = build_book(Book::Gaussian, geo13, &train, k, false);
        built.push((geo13, Book::Gaussian, lut, alpha));
    }

    // Guard: no two candidate tables may be the same table.
    for i in 0..built.len() {
        for j in (i + 1)..built.len() {
            if built[i].0 != built[j].0 {
                continue;
            }
            let same = built[i]
                .2
                .iter()
                .zip(built[j].2.iter())
                .filter(|(a, b)| a == b)
                .count();
            assert!(
                same * 10 < built[i].2.len(),
                "{} and {} are the same table",
                built[i].1.label(),
                built[j].1.label()
            );
        }
    }

    println!(
        "\n=== designed codebooks vs control (gaussian @ {}), {} draws, n={n} k={k} ===",
        GEO_SHIPPED.label(),
        seeds.len()
    );
    println!(
        "geometry     | codebook               | alpha | fixture     | mean d(w_cos) [min,max]              | mean d(mm_cos)"
    );

    let fixtures: [Fixture; 3] = [
        ("gaussian   ", gen_gaussian),
        ("student_t4 ", gen_student_t),
        ("fp4_dequant", gen_fp4_dequant),
    ];

    let mut overall: Vec<(Geo, Book, f64)> = Vec::new();
    for (fname, gen) in fixtures {
        // Control: the sweep's control, byte-for-byte the same call.
        let ctrl: Vec<(f64, f64, Vec<f32>)> = seeds
            .iter()
            .map(|&s| {
                let w = gen(n, k, sigma, s);
                let w_hat = quantize_matrix(&w, n, k, base);
                let m = evaluate(&w, &w_hat, n, k);
                (m.weight_cos, m.matmul_cos, w_hat)
            })
            .collect();
        for (geo, book, lut, alpha) in &built {
            let cfg = Cfg { geo: *geo, ..base };
            let mut dw = Vec::new();
            let mut dm = Vec::new();
            for (i, &s) in seeds.iter().enumerate() {
                let w = gen(n, k, sigma, s);
                let w_hat = quantize_matrix_with_lut(&w, n, k, cfg, lut);
                // Guard: this candidate must not reproduce the control exactly.
                if !(*geo == GEO_SHIPPED && *book == Book::Gaussian) {
                    let differing = ctrl[i]
                        .2
                        .iter()
                        .zip(w_hat.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                    assert!(
                        differing * 10 > w_hat.len(),
                        "{} @ {} on {fname}: only {differing}/{} weights differ from the \
                         control — the codebook is not reaching the search",
                        book.label(),
                        geo.label(),
                        w_hat.len()
                    );
                }
                let m = evaluate(&w, &w_hat, n, k);
                dw.push(m.weight_cos - ctrl[i].0);
                dm.push(m.matmul_cos - ctrl[i].1);
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let lo = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "{:<12} | {:<22} | {alpha:.3} | {fname} | {:+.5} [{:+.5},{:+.5}] | {:+.5}",
                format!("K{}/V{}/L{}", geo.k, geo.v, geo.l),
                book.label(),
                mean(&dw),
                lo(&dw),
                hi(&dw),
                mean(&dm)
            );
            overall.push((*geo, *book, mean(&dw)));
        }
        println!("------");
    }

    println!("\n=== mean d(w_cos) over all 3 fixtures x 5 draws (ship band is +-0.0008) ===");
    for (geo, book, _) in built.iter().map(|(g, b, _, _)| (*g, *b, 0.0)) {
        let vals: Vec<f64> = overall
            .iter()
            .filter(|(g, b, _)| *g == geo && *b == book)
            .map(|(_, _, d)| *d)
            .collect();
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        println!(
            "K{}/V{}/L{:<2} {:<22} {:+.5}  {}",
            geo.k,
            geo.v,
            geo.l,
            book.label(),
            m,
            if m.abs() <= 0.0008 {
                "INSIDE"
            } else {
                "outside"
            }
        );
    }
    println!("total {:.1}s", t0.elapsed().as_secs_f64());
}

/// **Is the best design converged, or merely unfinished?** `trellis Lloyd <-
/// random` is the strongest candidate in `probe_designed_codebook_at_k8v4l12`
/// and it was still descending after 8 sweeps, so reporting it as a floor would
/// be reporting a budget rather than a limit. This runs the same descent with
/// 4x the training data and 40 sweeps, evaluating the delta against the same
/// control at intervals so the shape of the curve is visible rather than
/// inferred, and re-fits the amplitude at the end.
///
/// Run: `cargo test -p mistralrs-quant --release probe_trellis_lloyd_convergence
/// -- --ignored --nocapture`
#[test]
#[ignore = "evidence-gathering probe (slow); run with --ignored --nocapture --release"]
fn probe_trellis_lloyd_convergence() {
    let n = 48;
    let k = 2048;
    let sigma = 0.02;
    let seeds = [1u64, 101, 202, 303, 404];
    let geo = GEO_CANDIDATE;
    let base = Cfg {
        viterbi: true,
        rotation_block: 128,
        policy: ScalePolicy::MaxOver3,
        geo: GEO_SHIPPED,
    };
    let cfg = Cfg { geo, ..base };
    let t0 = std::time::Instant::now();

    let train = codebook_training_targets(384, k);
    let rows: Vec<&[f32]> = train.chunks_exact(k).collect();
    println!(
        "training stream: {} vectors, {} per state",
        train.len() / geo.v as usize,
        train.len() / geo.v as usize / geo.lut_size()
    );

    // Same starting point as the winning row: random table at its fitted
    // amplitude.
    let mut init = gaussian_lut_geo(geo.l, geo.v);
    normalize_rms(&mut init);
    let (cw, cn) = calibration_matrix(4, k);
    let a0 = calibrate_alpha(&init, geo, &cw, cn, k);
    for x in init.iter_mut() {
        *x *= a0;
    }
    println!("init alpha {a0:.3}");

    // Control, once.
    let fixtures: [Fixture; 3] = [
        ("gaussian   ", gen_gaussian),
        ("student_t4 ", gen_student_t),
        ("fp4_dequant", gen_fp4_dequant),
    ];
    let control: Vec<Vec<f64>> = fixtures
        .iter()
        .map(|(_, gen)| {
            seeds
                .iter()
                .map(|&s| {
                    let w = gen(n, k, sigma, s);
                    evaluate(&w, &quantize_matrix(&w, n, k, base), n, k).weight_cos
                })
                .collect()
        })
        .collect();

    let delta_of = |lut: &[f32]| -> f64 {
        let mut all = Vec::new();
        for (fi, (_, gen)) in fixtures.iter().enumerate() {
            for (si, &s) in seeds.iter().enumerate() {
                let w = gen(n, k, sigma, s);
                let m = evaluate(&w, &quantize_matrix_with_lut(&w, n, k, cfg, lut), n, k);
                all.push(m.weight_cos - control[fi][si]);
            }
        }
        all.iter().sum::<f64>() / all.len() as f64
    };

    println!("\nsweeps | train MSE | mean d(w_cos) vs control | band +-0.0008");
    let mut lut = init.clone();
    let mut total = 0usize;
    let mut prev;
    for &step in &[0usize, 8, 8, 8, 8, 8] {
        if step > 0 {
            let (next, trace) = trellis_lloyd(&rows, geo, &lut, step);
            // The ridge in `trellis_lloyd` pulls toward its own `init`, so
            // restarting from the current table keeps the descent monotone
            // rather than repeatedly re-anchoring on the original.
            lut = next;
            total += step;
            prev = trace[trace.len() - 1];
        } else {
            let (_, trace) = trellis_lloyd(&rows, geo, &lut, 0);
            prev = trace[0];
        }
        let d = delta_of(&lut);
        println!(
            "{total:>6} | {prev:.6}  | {d:+.5}                  | {}  ({:.0}s)",
            if d.abs() <= 0.0008 {
                "INSIDE"
            } else {
                "outside"
            },
            t0.elapsed().as_secs_f64()
        );
    }

    // Best effort: re-fit the amplitude on the converged table.
    let mut refit = lut.clone();
    normalize_rms(&mut refit);
    let a1 = calibrate_alpha(&refit, geo, &cw, cn, k);
    for x in refit.iter_mut() {
        *x *= a1;
    }
    let d_refit = delta_of(&refit);
    println!(
        "\nconverged + amplitude re-fit (alpha {a1:.3}): {d_refit:+.5}  {}",
        if d_refit.abs() <= 0.0008 {
            "INSIDE"
        } else {
            "outside"
        }
    );
    println!("total {:.0}s", t0.elapsed().as_secs_f64());
}
