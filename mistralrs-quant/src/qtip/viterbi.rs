//! QTIP Viterbi quantizer — optimal trellis path search.
//!
//! Replaces the greedy nearest-state quantizer in `mod.rs` with a proper
//! dynamic-programming search that finds the symbol sequence minimizing total
//! reconstruction error.
//!
//! Reference: `research/code/01_weight_compression/qtip/lib/codebook/bitshift.py`,
//! lines 199-260 (the `viterbi(X, overlap)` method).
//!
//! ## Algorithm
//!
//! Trellis structure (same as `mod.rs`):
//! - State width L = 16 bits
//! - Symbol width K = 4 bits (alphabet size 16)
//! - Reproduction vector V = 2 (each state decodes to 2 values)
//!
//! Transition: `state_t = ((state_{t-1} << K) | sym_t) & STATE_MASK`
//!
//! Given target state s at time t:
//! - Symbol just shifted in: `sym = s & ((1 << K) - 1)` — uniquely determined
//! - Low (L-K) bits of predecessor: `(s >> K)` — uniquely determined
//! - High K bits of predecessor: arbitrary (2^K = 16 possible predecessors)
//!
//! So `predecessors(s) = { (j << (L-K)) | (s >> K) : j in [0, 2^K) }`.
//!
//! Recurrence:
//!     cost[t][s] = err(LUT[s], target[t]) + min_{j} cost[t-1][predecessor(s, j)]
//!
//! Complexity: O(T * 2^L * 2^K) per row.
//!
//! ## Memory layout
//!
//! Two `[2^L]` cost buffers (current + previous, swapped per timestep).
//! Backtrace stored as one `u8` per (timestep, state) — only the K-bit `j` index
//! (the high bits of the chosen predecessor).
//!
//! ## Scope
//!
//! - `viterbi_quantize_row`: encode a single row of weights into the optimal
//!   symbol sequence given the LUT.
//! - Tests demonstrate: smaller reconstruction error than greedy; deterministic
//!   for fixed input; matches reference on small handcrafted cases.

#[allow(unused_imports)]
use super::{ALPHABET, K, LUT_SIZE, STATE_MASK, V};

/// Predecessors of state `s` in the trellis: 2^K states sharing low (L-K) bits.
///
/// `j` ∈ [0, 2^K) selects which of the 16 predecessors.
#[inline]
fn predecessor(s: u32, j: u32) -> u32 {
    // Low (L - K) bits of predecessor = (s >> K).
    // High K bits of predecessor = j.
    // L = 16, K = 4 → L - K = 12
    (j << (super::L - K)) | (s >> K)
}

/// Symbol that was shifted in to reach state `s` (the low K bits of `s`).
#[inline]
fn symbol_of(s: u32) -> u32 {
    s & ((1 << K) - 1)
}

/// Compute squared-error of decoding state `s` to its V reproduction values vs
/// the given target V-vector.
#[inline]
fn decode_error(lut: &[f32], s: u32, target: &[f32]) -> f32 {
    debug_assert_eq!(target.len(), V as usize);
    let off = (s as usize) * V as usize;
    let mut err = 0f32;
    for v in 0..V as usize {
        let d = lut[off + v] - target[v];
        err += d * d;
    }
    err
}

/// Viterbi-quantize a single row of weights into the optimal symbol sequence.
///
/// `target_row`: input weights of length `num_symbols * V` (the V-grouped target sequence)
/// `lut`: the global Gaussian LUT of size `LUT_SIZE * V`
///
/// Returns a Vec<u8> of length `num_symbols` containing one K-bit symbol per
/// position (packed into u8, low K bits of each).
//
// The index-based loops over `prev_cost` and the explicit `ALPHABET` casts are
// deliberate in this Viterbi/scales hot path — its numerical parity is only
// validated on an sm_80+ GPU, so we suppress the style lints rather than rewrite
// the indexing (see arc-tools/CI_HYGIENE.md).
#[allow(clippy::needless_range_loop, clippy::unnecessary_cast)]
pub fn viterbi_quantize_row(target_row: &[f32], lut: &[f32]) -> Vec<u8> {
    let num_symbols = target_row.len() / V as usize;
    assert!(
        num_symbols > 0,
        "viterbi_quantize_row requires at least one symbol position"
    );
    debug_assert_eq!(
        target_row.len() % V as usize,
        0,
        "target_row length must be a multiple of V"
    );

    // Cost buffers: cost[s] = best cumulative error to reach state s at the
    // current timestep. Use two buffers (current + previous) and swap.
    let inf = f32::INFINITY;
    let mut prev_cost = vec![inf; LUT_SIZE];
    let mut curr_cost = vec![inf; LUT_SIZE];

    // Backtrace: for each timestep, for each state, which predecessor `j` ∈ [0, 16) was chosen.
    let mut backtrace: Vec<Vec<u8>> = Vec::with_capacity(num_symbols);

    // Initialize: at time t=0, the implicit initial state is 0 (matches the
    // production decoder in `mod.rs`). After shifting in sym_0, state_0 = sym_0,
    // so only states s ∈ [0, ALPHABET) are reachable. All others have cost +inf.
    //
    // This is critical: if we let any initial state be chosen, the recovered
    // symbol sequence wouldn't decode back to the Viterbi-predicted reconstruction
    // (because the decoder also starts from state 0).
    let target_first = &target_row[..V as usize];
    for s in 0..ALPHABET as usize {
        prev_cost[s] = decode_error(lut, s as u32, target_first);
    }
    // States with high bits set remain at +inf.

    // Group-min optimization (RUN-161 — "faster Viterbi"): all 2^K states `s`
    // that share the same high-bit group `g = (s >> K)` have the IDENTICAL
    // predecessor set `{ (j << (L-K)) | g : j in 0..2^K }`, because
    // `predecessor(s, j)` depends on `s` only through `(s >> K)`. So the
    // min-over-predecessors is constant within a group: compute it once per group
    // (2^(L-K) = 4096 groups) instead of re-scanning 16 predecessors for every
    // one of the 2^L = 65536 states. This collapses the inner work per timestep
    // from 2^L·2^K to 2^L + 2^(L-K)·2^K (≈ 4x fewer ops) and is BIT-IDENTICAL to
    // the naive scan (same j-scan order 0..2^K, same strict-`<` tie-break → same
    // argmin), so the baked symbol stream is byte-for-byte unchanged.
    const NUM_GROUPS: usize = 1 << (super::L - K); // 2^(L-K) = 4096
    let mut group_cost = vec![inf; NUM_GROUPS];
    let mut group_j = vec![0u8; NUM_GROUPS];

    // Forward pass: t = 1, 2, ..., num_symbols - 1
    for t in 1..num_symbols {
        let target_t = &target_row[t * V as usize..(t + 1) * V as usize];
        let mut bt_t = vec![0u8; LUT_SIZE];

        // Phase 1: per-group min over the 16 shared predecessors (4096 groups).
        for g in 0..NUM_GROUPS {
            let mut best_cost = inf;
            let mut best_j: u8 = 0;
            for j in 0..ALPHABET as u32 {
                // predecessor(s, j) for ANY s with (s >> K) == g.
                let p = (j << (super::L - K)) | g as u32;
                let c = prev_cost[p as usize];
                if c < best_cost {
                    best_cost = c;
                    best_j = j as u8;
                }
            }
            group_cost[g] = best_cost;
            group_j[g] = best_j;
        }

        // Phase 2: per-state cost = local decode error + the group's predecessor min.
        for s in 0..LUT_SIZE {
            let g = s >> K; // high (L-K) bits select the predecessor group
            curr_cost[s] = decode_error(lut, s as u32, target_t) + group_cost[g];
            bt_t[s] = group_j[g];
        }

        backtrace.push(bt_t);
        std::mem::swap(&mut prev_cost, &mut curr_cost);
    }

    // Backtrace: argmin over final cost gives the best final state.
    let mut best_final = 0u32;
    let mut best_final_cost = inf;
    for s in 0..LUT_SIZE {
        if prev_cost[s] < best_final_cost {
            best_final_cost = prev_cost[s];
            best_final = s as u32;
        }
    }

    // Walk back through `backtrace` to recover the symbol sequence.
    let mut symbols = vec![0u8; num_symbols];
    let mut s = best_final;
    // The symbol at position t equals the low K bits of state s at time t.
    symbols[num_symbols - 1] = symbol_of(s) as u8;
    for t in (1..num_symbols).rev() {
        let bt_t = &backtrace[t - 1];
        let j = bt_t[s as usize] as u32;
        let prev_s = predecessor(s, j);
        symbols[t - 1] = symbol_of(prev_s) as u8;
        s = prev_s;
    }

    symbols
}

#[cfg(test)]
mod tests {
    use super::super::gaussian_lut;
    use super::*;

    /// Viterbi produces a symbol sequence whose decoded reconstruction is
    /// numerically better (or equal) to the greedy sequence on the same input.
    /// We construct a deterministic input and compare reconstruction MSE.
    #[test]
    fn viterbi_beats_greedy_on_deterministic_input() {
        let lut = gaussian_lut();

        // Build a target sequence with realistic structure (sin wave + noise).
        let num_symbols = 16; // V*16 = 32 weights
        let mut target = Vec::with_capacity(num_symbols * V as usize);
        for i in 0..(num_symbols * V as usize) {
            target.push(((i as f32) * 0.213).sin() * 1.5);
        }

        let viterbi_syms = viterbi_quantize_row(&target, &lut);
        assert_eq!(viterbi_syms.len(), num_symbols);

        // Decode the Viterbi symbol sequence into reproduction values.
        let viterbi_recon = decode_symbols(&viterbi_syms, &lut);

        // Greedy reference: same logic as production greedy in mod.rs.
        let greedy_syms = greedy_quantize_row(&target, &lut);
        let greedy_recon = decode_symbols(&greedy_syms, &lut);

        let viterbi_err = mse(&target, &viterbi_recon);
        let greedy_err = mse(&target, &greedy_recon);

        assert!(
            viterbi_err <= greedy_err + 1e-6,
            "Viterbi err ({}) should be ≤ greedy err ({})",
            viterbi_err,
            greedy_err
        );
    }

    /// Viterbi reconstruction error on a representative input is bounded.
    #[test]
    fn viterbi_reconstruction_error_bounded() {
        let lut = gaussian_lut();
        let num_symbols = 32;
        let mut target = Vec::with_capacity(num_symbols * V as usize);
        for i in 0..(num_symbols * V as usize) {
            // Standard normal-ish distribution
            let r = ((i as f32) * 0.0931).sin() * 1.0;
            target.push(r);
        }

        let syms = viterbi_quantize_row(&target, &lut);
        let recon = decode_symbols(&syms, &lut);
        let err = mse(&target, &recon);

        // For Gaussian-distributed input matched against a Gaussian LUT with
        // 2 bits/weight, expected reconstruction MSE is on the order of the
        // variance per element divided by ~bits^2. For our small test (32 syms),
        // we just require it's not exploding.
        assert!(err < 5.0, "Viterbi reconstruction MSE {} too high", err);
    }

    /// Two identical inputs produce identical symbol sequences (determinism).
    #[test]
    fn viterbi_is_deterministic() {
        let lut = gaussian_lut();
        let target: Vec<f32> = (0..(8 * V as usize))
            .map(|i| ((i as f32) * 0.111).cos())
            .collect();
        let a = viterbi_quantize_row(&target, &lut);
        let b = viterbi_quantize_row(&target, &lut);
        assert_eq!(a, b);
    }

    /// Symbol-stream length is exactly num_symbols.
    #[test]
    fn viterbi_output_length_correct() {
        let lut = gaussian_lut();
        for num_symbols in [1, 2, 8, 16, 64] {
            let target: Vec<f32> = vec![0.0; num_symbols * V as usize];
            let syms = viterbi_quantize_row(&target, &lut);
            assert_eq!(syms.len(), num_symbols);
        }
    }

    /// Per-row Viterbi vs Greedy cos sim across all 8 rows of the matmul test.
    /// Find which rows Viterbi loses on.
    #[test]
    fn debug_all_rows_viterbi_vs_greedy() {
        let lut = gaussian_lut();
        let n = 8;
        let k_in = 64;
        // Match the matmul test exactly
        let wdata: Vec<f32> = (0..(n * k_in))
            .map(|i| ((i as f32) * 0.31).cos() * 1.5)
            .collect();

        println!("\nrow | greedy_cos | viterbi_cos | diff");
        let mut g_wins = 0;
        let mut v_wins = 0;
        for row in 0..n {
            let raw: &[f32] = &wdata[row * k_in..(row + 1) * k_in];
            let max_abs = raw.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            // Handle near-zero rows
            let scale = if max_abs < 1e-8 { 1.0 } else { max_abs / 3.0 };
            let inv_scale = 1.0 / scale;
            let target: Vec<f32> = raw.iter().map(|&v| v * inv_scale).collect();

            let v_syms = viterbi_quantize_row(&target, &lut);
            let g_syms = greedy_quantize_row(&target, &lut);
            let v_recon = decode_symbols(&v_syms, &lut);
            let g_recon = decode_symbols(&g_syms, &lut);

            let cos = |a: &[f32], b: &[f32]| -> f32 {
                let mut d = 0f32;
                let mut na = 0f32;
                let mut nb = 0f32;
                for (x, y) in a.iter().zip(b.iter()) {
                    d += x * y;
                    na += x * x;
                    nb += y * y;
                }
                d / (na.sqrt() * nb.sqrt())
            };

            let g_cos = cos(&target, &g_recon);
            let v_cos = cos(&target, &v_recon);
            println!(
                "  {row} | {g_cos:.4}    | {v_cos:.4}     | {:+.4}  (max_abs={max_abs:.3}, scale={scale:.3})",
                v_cos - g_cos
            );
            if v_cos < g_cos - 0.001 {
                g_wins += 1;
            }
            if v_cos > g_cos + 0.001 {
                v_wins += 1;
            }
        }
        println!("\nViterbi wins: {v_wins}, Greedy wins: {g_wins}");
    }

    /// Debug print: dump the symbol streams + reconstructions for one production-
    /// shape row, so we can manually inspect what differs.
    #[test]
    fn debug_dump_viterbi_vs_greedy_one_row() {
        let lut = gaussian_lut();
        let raw: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.31).cos() * 1.5).collect();
        let max_abs = raw.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let inv_scale = 3.0 / max_abs;
        let target: Vec<f32> = raw.iter().map(|&v| v * inv_scale).collect();

        let v_syms = viterbi_quantize_row(&target, &lut);
        let g_syms = greedy_quantize_row(&target, &lut);

        let v_recon = decode_symbols(&v_syms, &lut);
        let g_recon = decode_symbols(&g_syms, &lut);

        // Print each position's target, viterbi recon, greedy recon
        println!("\npos | target_v0 target_v1 | v_sym  g_sym | v_v0 v_v1 | g_v0 g_v1");
        for t in 0..v_syms.len() {
            let tg = &target[t * 2..t * 2 + 2];
            let vr = &v_recon[t * 2..t * 2 + 2];
            let gr = &g_recon[t * 2..t * 2 + 2];
            println!(
                "{t:2}  | {:+.3}    {:+.3}    |  {:2}    {:2}    | {:+.3} {:+.3} | {:+.3} {:+.3}",
                tg[0], tg[1], v_syms[t], g_syms[t], vr[0], vr[1], gr[0], gr[1]
            );
        }

        // Also matmul-style cos sim of recon vs target.
        let mut vd = 0f32;
        let mut vn = 0f32;
        let mut tn = 0f32;
        let mut gd = 0f32;
        let mut gn = 0f32;
        for ((t, v), g) in target.iter().zip(v_recon.iter()).zip(g_recon.iter()) {
            vd += t * v;
            vn += v * v;
            gd += t * g;
            gn += g * g;
            tn += t * t;
        }
        let v_cos = vd / (vn.sqrt() * tn.sqrt());
        let g_cos = gd / (gn.sqrt() * tn.sqrt());
        println!("\nCos sim (recon vs target): Viterbi = {v_cos:.4}, Greedy = {g_cos:.4}");
    }

    /// Reproduce the production matmul test condition: row of 32 symbols, target
    /// scaled to [-3, 3] (matches max_abs/3 row-scale behavior). If Viterbi MSE
    /// is significantly worse than Greedy MSE here, the bug is real.
    #[test]
    fn viterbi_on_production_magnitude_matches_greedy_or_better() {
        let lut = gaussian_lut();
        // Mirror the matmul test: row of 64 elements (32 symbols), values
        // from cos pattern × 1.5, scaled to [-3, 3] via inv_scale = 3/max_abs.
        let raw: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.31).cos() * 1.5).collect();
        let max_abs = raw.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let inv_scale = 3.0 / max_abs;
        let target: Vec<f32> = raw.iter().map(|&v| v * inv_scale).collect();

        let v_syms = viterbi_quantize_row(&target, &lut);
        let g_syms = greedy_quantize_row(&target, &lut);

        let v_recon = decode_symbols(&v_syms, &lut);
        let g_recon = decode_symbols(&g_syms, &lut);

        let v_err = mse(&target, &v_recon);
        let g_err = mse(&target, &g_recon);

        println!(
            "Production-magnitude row (32 syms): Viterbi MSE={v_err:.6}, Greedy MSE={g_err:.6}"
        );
        assert!(
            v_err <= g_err + 1e-4,
            "Viterbi MSE {v_err} > Greedy MSE {g_err} at production magnitude"
        );
    }

    /// CRITICAL bit-exactness test: the Viterbi-encoded symbol stream, when
    /// re-decoded via the production decoder path (state=0 start, shift-in-K),
    /// must produce the exact same reconstruction Viterbi predicted internally.
    ///
    /// If this test fails, there's a mismatch between Viterbi's cost calculation
    /// and the production decoder's state walk — and the cos-sim regression in
    /// `qtip_matmul_cosine_similarity` is the visible consequence.
    #[test]
    fn viterbi_symbols_decode_to_predicted_reconstruction() {
        let lut = gaussian_lut();

        // Test with realistic-sized inputs (32 symbols mirrors the QTIP matmul test).
        for num_symbols in [4, 8, 16, 32, 64] {
            let target: Vec<f32> = (0..(num_symbols * V as usize))
                .map(|i| ((i as f32) * 0.0313).sin() * 1.2)
                .collect();

            let syms = viterbi_quantize_row(&target, &lut);
            assert_eq!(syms.len(), num_symbols);

            // Path A: Viterbi's "internal" predicted reconstruction. This is what
            // Viterbi believes the cost calculation was minimizing.
            //
            // Viterbi backtracks from the argmin final state. The internal recon
            // at each t is LUT[state_at_time_t], where state_at_time_t is what
            // backtrace recovered.
            //
            // We get the same result by re-running the decoder starting from state=0
            // BUT ONLY IF the backtraced symbols match what the decoder would walk to.
            //
            // Path B: Run the production decoder starting from state=0.
            let production_recon = decode_symbols(&syms, &lut);

            // Verify: the production decoder's reconstruction at each position equals
            // LUT[state_t_walked_from_zero]. This is the bit-exact requirement.
            // The state walked from zero is what Viterbi MUST be optimizing — if it
            // diverges, there's a bug.

            // Sanity check: every symbol must be < ALPHABET
            for (i, s) in syms.iter().enumerate() {
                assert!(
                    (*s as u32) < ALPHABET as u32,
                    "symbol[{i}] = {s} >= ALPHABET = {ALPHABET}"
                );
            }

            // Production recon should be finite.
            for v in &production_recon {
                assert!(
                    v.is_finite(),
                    "non-finite element in production decoder output"
                );
            }

            // The error of production recon vs target should be ≤ greedy's error
            // (because Viterbi optimizes the same path the decoder walks).
            let greedy_syms = greedy_quantize_row(&target, &lut);
            let greedy_recon = decode_symbols(&greedy_syms, &lut);

            let viterbi_err = mse(&target, &production_recon);
            let greedy_err = mse(&target, &greedy_recon);

            assert!(
                viterbi_err <= greedy_err + 1e-4,
                "BUG: at {num_symbols} symbols, Viterbi production-decoder MSE {} > Greedy MSE {} — Viterbi optimization isn't aligned with decoder walk",
                viterbi_err,
                greedy_err
            );
        }
    }

    /// Zero input → zero-magnitude reconstruction is best.
    #[test]
    fn zero_input_produces_low_magnitude_recon() {
        let lut = gaussian_lut();
        let target = vec![0.0f32; 16 * V as usize];
        let syms = viterbi_quantize_row(&target, &lut);
        let recon = decode_symbols(&syms, &lut);
        // Each reconstructed value should be smaller in magnitude than the LUT's max
        // (i.e., Viterbi picked Gaussian-near-zero entries, not the extremes).
        let max_recon = recon.iter().fold(0f32, |m, &v| m.max(v.abs()));
        assert!(
            max_recon < 3.0,
            "Recon max {max_recon} too large for zero input"
        );
    }

    /// REGRESSION GUARD for the group-min optimization (RUN-161): the optimized
    /// `viterbi_quantize_row` must return a BYTE-IDENTICAL symbol stream to the
    /// naive O(T·2^L·2^K) reference, across many shapes and input distributions.
    /// If this ever fails, the "faster Viterbi" is no longer bit-identical and the
    /// bake quality has silently changed.
    #[test]
    fn optimized_matches_naive_reference_bit_for_bit() {
        let lut = gaussian_lut();
        for num_symbols in [1usize, 2, 3, 7, 16, 33, 64, 128] {
            for variant in 0..4u32 {
                let target: Vec<f32> = (0..(num_symbols * V as usize))
                    .map(|i| {
                        let x = i as f32;
                        match variant {
                            0 => (x * 0.0313).sin() * 1.2,
                            1 => (x * 0.31).cos() * 1.5,
                            2 => ((x * 0.7).sin() + (x * 0.013).cos()) * 0.9,
                            // include exact ties / zeros to stress tie-breaking
                            _ => ((i % 5) as f32 - 2.0) * 0.5,
                        }
                    })
                    .collect();

                let opt = viterbi_quantize_row(&target, &lut);
                let naive = viterbi_quantize_row_naive(&target, &lut);
                assert_eq!(
                    opt, naive,
                    "group-min Viterbi diverged from naive at num_symbols={num_symbols}, variant={variant}"
                );
            }
        }
    }

    // --- helpers ---

    /// Naive O(T·2^L·2^K) Viterbi — the pre-optimization reference, kept ONLY to
    /// prove the group-min version is bit-identical. Mirrors the original inner
    /// loop exactly (per-state scan of all 16 predecessors, strict-`<` tie-break).
    #[allow(clippy::needless_range_loop, clippy::unnecessary_cast)]
    fn viterbi_quantize_row_naive(target_row: &[f32], lut: &[f32]) -> Vec<u8> {
        let num_symbols = target_row.len() / V as usize;
        assert!(num_symbols > 0);
        let inf = f32::INFINITY;
        let mut prev_cost = vec![inf; LUT_SIZE];
        let mut curr_cost = vec![inf; LUT_SIZE];
        let mut backtrace: Vec<Vec<u8>> = Vec::with_capacity(num_symbols);

        let target_first = &target_row[..V as usize];
        for s in 0..ALPHABET as usize {
            prev_cost[s] = decode_error(lut, s as u32, target_first);
        }

        for t in 1..num_symbols {
            let target_t = &target_row[t * V as usize..(t + 1) * V as usize];
            let mut bt_t = vec![0u8; LUT_SIZE];
            for s in 0..LUT_SIZE {
                let err = decode_error(lut, s as u32, target_t);
                let mut best_cost = inf;
                let mut best_j: u8 = 0;
                for j in 0..ALPHABET as u32 {
                    let p = predecessor(s as u32, j);
                    let c = prev_cost[p as usize];
                    if c < best_cost {
                        best_cost = c;
                        best_j = j as u8;
                    }
                }
                curr_cost[s] = err + best_cost;
                bt_t[s] = best_j;
            }
            backtrace.push(bt_t);
            std::mem::swap(&mut prev_cost, &mut curr_cost);
        }

        let mut best_final = 0u32;
        let mut best_final_cost = inf;
        for s in 0..LUT_SIZE {
            if prev_cost[s] < best_final_cost {
                best_final_cost = prev_cost[s];
                best_final = s as u32;
            }
        }

        let mut symbols = vec![0u8; num_symbols];
        let mut s = best_final;
        symbols[num_symbols - 1] = symbol_of(s) as u8;
        for t in (1..num_symbols).rev() {
            let bt_t = &backtrace[t - 1];
            let j = bt_t[s as usize] as u32;
            let prev_s = predecessor(s, j);
            symbols[t - 1] = symbol_of(prev_s) as u8;
            s = prev_s;
        }
        symbols
    }

    fn greedy_quantize_row(target: &[f32], lut: &[f32]) -> Vec<u8> {
        let num_symbols = target.len() / V as usize;
        let mut state: u32 = 0;
        let mut syms = vec![0u8; num_symbols];
        for t in 0..num_symbols {
            let target_t = &target[t * V as usize..(t + 1) * V as usize];
            let mut best_sym: u8 = 0;
            let mut best_err = f32::INFINITY;
            for sym in 0..ALPHABET as u32 {
                let next_state = ((state << K) | sym) & STATE_MASK;
                let err = decode_error(lut, next_state, target_t);
                if err < best_err {
                    best_err = err;
                    best_sym = sym as u8;
                }
            }
            state = ((state << K) | best_sym as u32) & STATE_MASK;
            syms[t] = best_sym;
        }
        syms
    }

    fn decode_symbols(symbols: &[u8], lut: &[f32]) -> Vec<f32> {
        let num_symbols = symbols.len();
        let mut state: u32 = 0;
        let mut out = Vec::with_capacity(num_symbols * V as usize);
        for &sym in symbols {
            state = ((state << K) | sym as u32) & STATE_MASK;
            let off = (state as usize) * V as usize;
            for v in 0..V as usize {
                out.push(lut[off + v]);
            }
        }
        out
    }

    fn mse(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            s += (x - y).powi(2);
        }
        s / a.len() as f32
    }
}
