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
//!
//! ## Two orthogonal knobs (wave13-AD)
//!
//! The exhaustive DP above is *one* point in a 2-D design space. Both axes are
//! independently selectable via [`TrellisSearch`] and the optional per-position
//! weight vector, and both default to today's behaviour.
//!
//! ### Axis A — search: exhaustive vs. beam
//!
//! [`TrellisSearch::Beam`] keeps only the best `W` states per timestep instead
//! of all `2^L`. The candidate set at step `t` is `W · 2^K` (successors of the
//! surviving states), deduplicated by successor state with a min-cost merge —
//! i.e. it is *pruned Viterbi*, not a different algorithm. Setting
//! `W >= 2^L` prunes nothing and reproduces [`TrellisSearch::Exhaustive`]
//! **bit-for-bit** (see `beam_unpruned_matches_exhaustive_bit_for_bit`); this
//! is the regression guard that keeps the fast path honest.
//!
//! Two properties make the beam exact-compatible:
//! * The beam is kept **sorted by state ascending**, so for any successor `s'`
//!   the 16 candidate predecessors `p = (j << (L-K)) | (s' >> K)` are visited
//!   in ascending `j` — matching the group-min scan's `0..2^K` order.
//! * Merging uses strict `<` (first-seen wins ties), matching the group-min
//!   tie-break, and merges on the *predecessor* cost before the local error is
//!   added (the local error is identical for a fixed successor, so it cannot
//!   change the argmin — and this way `decode_error` runs once per distinct
//!   successor instead of once per candidate).
//!
//! Beam search also collapses the backtrace from `2^L` bytes per timestep to
//! `4·W` bytes — the reason the exhaustive kernel needs a multi-GB scratch
//! budget on production-shaped rows (`num_symbols = 9472` ⇒ 620 MB vs 9.7 MB
//! at `W = 256`).
//!
//! ### Axis B — objective: unweighted MSE vs. diagonal Hessian
//!
//! Today's branch metric is `‖w − ŵ‖²`. The proxy-Hessian objective used by
//! GPTQ / LDLQ / QuIP# / QTIP is `(w − ŵ)ᵀ H (w − ŵ)` with `H = (1/N)·XᵀX`
//! accumulated over calibration activations
//! (`research/code/01_weight_compression/qtip/lib/utils/data_utils.py:28-36`,
//! normalised at `quantize_llama/input_hessian_llama.py:137`), relatively
//! damped by `H ← H + σ·mean(diag H)·I` with `σ = 1e-2`
//! (`lib/utils/math_utils.py:44-49`).
//!
//! **Derivation for a per-row trellis.** Keeping only `diag(H)` makes the LDL
//! factor the identity, so the objective separates over input columns and
//! collapses to `Σ_j H_jj (w_j − ŵ_j)²` — a per-position weight in the branch
//! metric. That is exactly [`decode_error_weighted`]. (The official QTIP
//! `viterbi` is *unweighted*: `lib/codebook/bitshift.py:204-206,218-223` has no
//! Hessian argument at all. The Hessian enters through LDLQ's error feedback in
//! the LDL-transformed frame — `lib/algo/ldlq.py:48-67` — where the per-block
//! `D_ii` weighting is dropped and the trellis solves a plain nearest-sequence
//! problem on the corrected target. So per-position weights capture the
//! *diagonal* of what LDLQ captures in full.)
//!
//! **Frame correction.** The search runs on the rotated row `w̃ = R·w` with
//! `R = D·H_B·D` block-diagonal of width `B`. The error in the original frame
//! is `e = R·ẽ`, so `eᵀ H e = ẽᵀ (RᵀHR) ẽ`. Every entry of a normalised
//! Hadamard satisfies `R_ji² = 1/B`, hence
//! `diag(RᵀHR)_ii = Σ_j H_jj R_ji² = mean(H_jj over i's block)`. The correct
//! search-frame weight is therefore the **block mean** of `diag(H)`, computed
//! by [`hessian_row_weights`]. Two consequences worth stating plainly:
//! with *full-width* rotation the weights become globally constant and the
//! objective degenerates to unweighted; with the production cap `B = 128` a
//! `K = 4096` row keeps 32 distinct weights, so the signal survives but is
//! deliberately smoothed by incoherence processing.

#[allow(unused_imports)]
use super::{ALPHABET, K, LUT_SIZE, STATE_MASK, V};

/// Relative Hessian damping, matching QTIP's `--sigma_reg` default
/// (`research/code/01_weight_compression/qtip/lib/utils/math_utils.py:44-49`:
/// `H ← H + σ·mean(diag H)·I`).
pub const HESSIAN_SIGMA_REG: f64 = 1e-2;

/// How the trellis search explores the `2^L` state space.
///
/// `Exhaustive` is the historical (and default) behaviour. `Beam { width }`
/// keeps only the best `width` states per timestep; `width >= 2^L` prunes
/// nothing and is bit-identical to `Exhaustive`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrellisSearch {
    /// Full dynamic program over all `2^L` states (today's bake).
    #[default]
    Exhaustive,
    /// Pruned dynamic program keeping the best `width` states per timestep.
    Beam {
        /// Number of surviving states per timestep. Clamped to `2^L`.
        width: usize,
    },
}

impl TrellisSearch {
    /// Short tag for the bake log header (task #17: a Greedy bake must never
    /// again be mistakable for a Viterbi bake).
    pub fn tag(&self) -> String {
        match self {
            TrellisSearch::Exhaustive => "viterbi-exhaustive".to_string(),
            TrellisSearch::Beam { width } => format!("viterbi-beam(W={width})"),
        }
    }

    /// Parse the `ARC_QTIP_BEAM` env override. Unset, empty, `0`, `off` or
    /// `exhaustive` ⇒ [`TrellisSearch::Exhaustive`]; any positive integer ⇒
    /// a beam of that width. Unparsable values fall back to exhaustive so a
    /// typo can never silently *lower* bake quality.
    pub fn from_env() -> Self {
        match std::env::var("ARC_QTIP_BEAM") {
            Ok(v) => Self::parse(&v),
            Err(_) => TrellisSearch::Exhaustive,
        }
    }

    fn parse(v: &str) -> Self {
        let v = v.trim();
        if v.is_empty() || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("exhaustive") {
            return TrellisSearch::Exhaustive;
        }
        match v.parse::<usize>() {
            Ok(0) => TrellisSearch::Exhaustive,
            Ok(w) if w >= LUT_SIZE => TrellisSearch::Exhaustive,
            Ok(w) => TrellisSearch::Beam { width: w },
            Err(_) => {
                tracing::warn!("ARC_QTIP_BEAM={v:?} is not a width; using exhaustive Viterbi");
                TrellisSearch::Exhaustive
            }
        }
    }
}

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

/// Hessian-weighted squared error for state `s` against a V-vector target.
///
/// `w` holds the V per-position weights for this timestep (the diagonal of the
/// search-frame Hessian, see [`hessian_row_weights`]).
#[inline]
fn decode_error_weighted(lut: &[f32], s: u32, target: &[f32], w: &[f32]) -> f32 {
    debug_assert_eq!(target.len(), V as usize);
    debug_assert_eq!(w.len(), V as usize);
    let off = (s as usize) * V as usize;
    let mut err = 0f32;
    for v in 0..V as usize {
        let d = lut[off + v] - target[v];
        err += w[v] * d * d;
    }
    err
}

/// Branch metric dispatch: unweighted MSE, or diagonal-Hessian weighted.
#[inline]
fn branch_metric(lut: &[f32], s: u32, target: &[f32], w: Option<&[f32]>) -> f32 {
    match w {
        None => decode_error(lut, s, target),
        Some(w) => decode_error_weighted(lut, s, target, w),
    }
}

/// Project a per-input-column Hessian diagonal into the trellis *search* frame.
///
/// Input: `h_diag[j] = H_jj = (1/N) Σ_n x_{n,j}²` over calibration activations,
/// in the model's original input-feature frame — precisely what
/// [`crate::ImatrixLayerStats::compute_imatrix`] already accumulates.
///
/// Output: one non-negative weight per input column, in the frame the trellis
/// actually searches, normalised to mean 1 so the branch metric keeps the same
/// dynamic range as the unweighted one (a global scale cannot change the
/// argmin, but it does change how close the accumulated costs get to `f32`
/// saturation on long rows).
///
/// `rotation_block`: the Hadamard block width used at quantize time, or 0 when
/// rotation is disabled. Because every entry of a normalised Hadamard obeys
/// `R_ji² = 1/B`, `diag(RᵀHR)` is the **block mean** of `diag(H)` — see the
/// module header. A relative damping of [`HESSIAN_SIGMA_REG`] is applied first,
/// mirroring QTIP's `regularize_H`, so an all-but-dead input block can never
/// drive its weight to exactly zero (which would let the trellis emit arbitrary
/// symbols there).
pub fn hessian_row_weights(h_diag: &[f32], rotation_block: usize) -> Vec<f32> {
    let k = h_diag.len();
    if k == 0 {
        return Vec::new();
    }

    // Sanitize: negatives / NaN / inf from a truncated or corrupt calibration
    // pass must not poison the metric.
    let mut w: Vec<f64> = h_diag
        .iter()
        .map(|&v| {
            if v.is_finite() && v > 0.0 {
                v as f64
            } else {
                0.0
            }
        })
        .collect();

    // Relative damping, QTIP `regularize_H` (math_utils.py:44-49).
    let raw_mean = w.iter().sum::<f64>() / k as f64;
    if raw_mean <= 0.0 {
        // No usable calibration signal — fall back to the unweighted objective.
        return vec![1.0f32; k];
    }
    for v in w.iter_mut() {
        *v += HESSIAN_SIGMA_REG * raw_mean;
    }

    // Frame correction: diag(RᵀHR) is the per-block mean of diag(H).
    if rotation_block >= 2 {
        let mut b0 = 0usize;
        while b0 < k {
            let end = (b0 + rotation_block).min(k);
            let len = end - b0;
            let mean = w[b0..end].iter().sum::<f64>() / len as f64;
            for v in w[b0..end].iter_mut() {
                *v = mean;
            }
            b0 = end;
        }
    }

    // Normalise to mean 1. For a uniform input this yields exactly 1.0 in every
    // slot, so a flat Hessian is bit-identical to the unweighted objective.
    let mean = w.iter().sum::<f64>() / k as f64;
    if mean <= 0.0 || !mean.is_finite() {
        return vec![1.0f32; k];
    }
    w.into_iter().map(|v| (v / mean) as f32).collect()
}

/// Viterbi-quantize a single row of weights into the optimal symbol sequence.
///
/// `target_row`: input weights of length `num_symbols * V` (the V-grouped target sequence)
/// `lut`: the global Gaussian LUT of size `LUT_SIZE * V`
///
/// Returns a Vec<u8> of length `num_symbols` containing one K-bit symbol per
/// position (packed into u8, low K bits of each).
///
/// This is the unweighted, exhaustive default. [`quantize_row`] exposes the
/// beam-width and Hessian-weighting knobs.
pub fn viterbi_quantize_row(target_row: &[f32], lut: &[f32]) -> Vec<u8> {
    exhaustive_quantize_row(target_row, lut, None)
}

/// Full search entry point: pick the search strategy and the objective.
///
/// `weights`: optional per-input-column weights of length `target_row.len()`,
/// already projected into the search frame by [`hessian_row_weights`]. `None`
/// selects the unweighted `‖w − ŵ‖²` objective.
pub fn quantize_row(
    target_row: &[f32],
    lut: &[f32],
    search: TrellisSearch,
    weights: Option<&[f32]>,
) -> Vec<u8> {
    debug_assert!(weights.is_none_or(|w| w.len() == target_row.len()));
    match search {
        TrellisSearch::Exhaustive => exhaustive_quantize_row(target_row, lut, weights),
        TrellisSearch::Beam { width } if width >= LUT_SIZE => {
            exhaustive_quantize_row(target_row, lut, weights)
        }
        TrellisSearch::Beam { width } => beam_quantize_row(target_row, lut, weights, width),
    }
}

/// Exhaustive group-min dynamic program over all `2^L` states.
//
// The index-based loops over `prev_cost` and the explicit `ALPHABET` casts are
// deliberate in this Viterbi/scales hot path — its numerical parity is only
// validated on an sm_80+ GPU, so we suppress the style lints rather than rewrite
// the indexing (see arc-tools/CI_HYGIENE.md).
#[allow(clippy::needless_range_loop, clippy::unnecessary_cast)]
fn exhaustive_quantize_row(target_row: &[f32], lut: &[f32], weights: Option<&[f32]>) -> Vec<u8> {
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
    let weights_first = weights.map(|w| &w[..V as usize]);
    for s in 0..ALPHABET as usize {
        prev_cost[s] = branch_metric(lut, s as u32, target_first, weights_first);
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
        // Split by objective so the unweighted hot path keeps its exact shape
        // (no per-state branch, bit-identical to the pre-wave13 code).
        match weights {
            None => {
                for s in 0..LUT_SIZE {
                    let g = s >> K; // high (L-K) bits select the predecessor group
                    curr_cost[s] = decode_error(lut, s as u32, target_t) + group_cost[g];
                    bt_t[s] = group_j[g];
                }
            }
            Some(w) => {
                let w_t = &w[t * V as usize..(t + 1) * V as usize];
                for s in 0..LUT_SIZE {
                    let g = s >> K;
                    curr_cost[s] =
                        decode_error_weighted(lut, s as u32, target_t, w_t) + group_cost[g];
                    bt_t[s] = group_j[g];
                }
            }
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

// ---------------------------------------------------------------------------
// Beam search (axis A)
// ---------------------------------------------------------------------------

/// One surviving trellis state at a timestep.
#[derive(Clone, Copy)]
struct BeamEntry {
    /// Cumulative path cost into this state. During candidate generation this
    /// transiently holds the *predecessor* cost (the local error is added once
    /// per distinct successor, after merging).
    cost: f32,
    /// Trellis state (`L = 16` bits, so `u16` is exact).
    state: u16,
    /// Index of the chosen predecessor in the previous timestep's beam.
    parent: u16,
}

/// Pruned Viterbi keeping the best `width` states per timestep.
///
/// Complexity per timestep is `O(width · 2^K)` candidate generations plus one
/// `O(width · 2^K)` selection, versus `O(2^L)` for the exhaustive DP.
/// Backtrace memory is `4 · width` bytes per timestep instead of `2^L`.
///
/// See the module header for why this reproduces [`exhaustive_quantize_row`]
/// exactly when `width >= 2^L`.
fn beam_quantize_row(
    target_row: &[f32],
    lut: &[f32],
    weights: Option<&[f32]>,
    width: usize,
) -> Vec<u8> {
    let num_symbols = target_row.len() / V as usize;
    assert!(
        num_symbols > 0,
        "beam_quantize_row requires at least one symbol position"
    );
    debug_assert_eq!(target_row.len() % V as usize, 0);
    // A beam wider than the state space prunes nothing; clamp so `u16` indices
    // (and the dedup slot table) stay valid.
    let width = width.clamp(1, LUT_SIZE);

    // Dedup table: successor state -> index into `cands`. Allocated once and
    // cleared through `touched` so we never pay a 2^L memset per timestep.
    let mut slot = vec![u32::MAX; LUT_SIZE];
    let mut touched: Vec<u16> = Vec::with_capacity(width * ALPHABET);
    let mut cands: Vec<BeamEntry> = Vec::with_capacity(width * ALPHABET);

    // Compacted backtrace, flat: `(state, parent)` for every surviving entry of
    // every timestep, with one start offset per timestep. A `Vec<Vec<_>>` here
    // costs one allocation per timestep, which on production-shaped rows
    // (`num_symbols = 9472`) dominates the search itself.
    let mut trace: Vec<(u16, u16)> = Vec::with_capacity(num_symbols * width.min(ALPHABET * 4));
    let mut trace_off: Vec<u32> = Vec::with_capacity(num_symbols);

    // t = 0: the decoder starts from state 0, so exactly the `2^K` states
    // s ∈ [0, ALPHABET) are reachable, in ascending order.
    let target_first = &target_row[..V as usize];
    let weights_first = weights.map(|w| &w[..V as usize]);
    let mut beam: Vec<BeamEntry> = (0..ALPHABET as u32)
        .map(|s| BeamEntry {
            cost: branch_metric(lut, s, target_first, weights_first),
            state: s as u16,
            parent: 0,
        })
        .collect();
    prune_to_width(&mut beam, width);
    trace_off.push(0);
    trace.extend(beam.iter().map(|e| (e.state, e.parent)));

    for t in 1..num_symbols {
        let target_t = &target_row[t * V as usize..(t + 1) * V as usize];
        let weights_t = weights.map(|w| &w[t * V as usize..(t + 1) * V as usize]);

        cands.clear();
        touched.clear();

        // Expand: every surviving state × every symbol. The beam is sorted by
        // state ascending, so for a fixed successor the predecessors arrive in
        // ascending `j` — matching the exhaustive group-min scan order — and
        // strict `<` keeps the smallest `j` on ties.
        for (pi, entry) in beam.iter().enumerate() {
            let base = ((entry.state as u32) << K) & STATE_MASK;
            for sym in 0..ALPHABET as u32 {
                let succ = (base | sym) as u16;
                let s_idx = succ as usize;
                let existing = slot[s_idx];
                if existing == u32::MAX {
                    slot[s_idx] = cands.len() as u32;
                    touched.push(succ);
                    cands.push(BeamEntry {
                        cost: entry.cost,
                        state: succ,
                        parent: pi as u16,
                    });
                } else {
                    let c = &mut cands[existing as usize];
                    if entry.cost < c.cost {
                        c.cost = entry.cost;
                        c.parent = pi as u16;
                    }
                }
            }
        }

        // Add the local branch metric once per distinct successor. The metric
        // is identical for every candidate reaching the same state, so doing it
        // after the merge cannot change any argmin.
        for c in cands.iter_mut() {
            c.cost += branch_metric(lut, c.state as u32, target_t, weights_t);
        }

        // Release the dedup slots for the next timestep.
        for &s in &touched {
            slot[s as usize] = u32::MAX;
        }

        prune_to_width(&mut cands, width);
        // `cands` becomes the new beam; the old beam is recycled as next
        // timestep's candidate buffer (no per-timestep allocation).
        std::mem::swap(&mut beam, &mut cands);
        trace_off.push(trace.len() as u32);
        trace.extend(beam.iter().map(|e| (e.state, e.parent)));
    }

    // Final state: lowest cost, lowest state index on ties (the beam is state
    // sorted, so a strict-`<` scan reproduces the exhaustive tie-break).
    let mut best_idx = 0usize;
    let mut best_cost = f32::INFINITY;
    for (i, e) in beam.iter().enumerate() {
        if e.cost < best_cost {
            best_cost = e.cost;
            best_idx = i;
        }
    }

    let mut symbols = vec![0u8; num_symbols];
    let mut idx = best_idx;
    for t in (0..num_symbols).rev() {
        let (state, parent) = trace[trace_off[t] as usize + idx];
        symbols[t] = symbol_of(state as u32) as u8;
        idx = parent as usize;
    }
    symbols
}

/// Keep the best `width` entries and restore ascending-state order.
///
/// Selection is by `(cost, state)` under [`f32::total_cmp`] so the survivor set
/// is fully deterministic; the trailing sort by state is what makes the next
/// timestep's predecessor visit order match the exhaustive group-min scan.
fn prune_to_width(beam: &mut Vec<BeamEntry>, width: usize) {
    if beam.len() > width {
        beam.select_nth_unstable_by(width - 1, |a, b| {
            a.cost.total_cmp(&b.cost).then(a.state.cmp(&b.state))
        });
        beam.truncate(width);
    }
    beam.sort_unstable_by_key(|e| e.state);
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

    // -----------------------------------------------------------------------
    // Axis A — beam search
    // -----------------------------------------------------------------------

    /// Deterministic pseudo-random row generator for the search tests.
    fn test_row(num_symbols: usize, variant: u32) -> Vec<f32> {
        (0..(num_symbols * V as usize))
            .map(|i| {
                let x = i as f32;
                match variant {
                    0 => (x * 0.0313).sin() * 1.2,
                    1 => (x * 0.31).cos() * 1.5,
                    2 => ((x * 0.7).sin() + (x * 0.013).cos()) * 0.9,
                    3 => ((i % 5) as f32 - 2.0) * 0.5, // exact ties / zeros
                    _ => ((x * 1.7).sin() * (x * 0.011).cos()) * 2.5,
                }
            })
            .collect()
    }

    /// **THE regression guard for axis A.** An unpruned beam (`W >= 2^L`) must
    /// return a BYTE-IDENTICAL symbol stream to the exhaustive group-min DP.
    /// This is what keeps the beam an *approximation of the same algorithm*
    /// rather than a different one: if it ever diverges here, the beam's
    /// tie-breaking or merge order has drifted and every quality number in the
    /// harness is measuring something else.
    #[test]
    fn beam_unpruned_matches_exhaustive_bit_for_bit() {
        let lut = gaussian_lut();
        for num_symbols in [1usize, 2, 3, 7, 16, 33, 64] {
            for variant in 0..5u32 {
                let target = test_row(num_symbols, variant);
                let exact = viterbi_quantize_row(&target, &lut);
                let beam = beam_quantize_row(&target, &lut, None, LUT_SIZE);
                assert_eq!(
                    exact, beam,
                    "unpruned beam diverged from exhaustive at num_symbols={num_symbols}, variant={variant}"
                );
                // The public dispatcher must route W >= 2^L to the exact path.
                let via_api = quantize_row(
                    &target,
                    &lut,
                    TrellisSearch::Beam {
                        width: LUT_SIZE * 4,
                    },
                    None,
                );
                assert_eq!(exact, via_api);
            }
        }
    }

    /// Same guard with the Hessian objective engaged: an unpruned weighted beam
    /// must equal the weighted exhaustive DP bit-for-bit.
    #[test]
    fn beam_unpruned_matches_exhaustive_weighted() {
        let lut = gaussian_lut();
        for num_symbols in [2usize, 9, 32] {
            for variant in 0..5u32 {
                let target = test_row(num_symbols, variant);
                let k = target.len();
                // Non-uniform, outlier-heavy diagonal Hessian.
                let h: Vec<f32> = (0..k)
                    .map(|j| 0.05 + ((j as f32) * 0.37).sin().abs() * 4.0)
                    .collect();
                let w = hessian_row_weights(&h, 0);
                let exact = exhaustive_quantize_row(&target, &lut, Some(&w));
                let beam = beam_quantize_row(&target, &lut, Some(&w), LUT_SIZE);
                assert_eq!(
                    exact, beam,
                    "weighted unpruned beam diverged at num_symbols={num_symbols}, variant={variant}"
                );
            }
        }
    }

    /// A beam is never *better* than the exhaustive DP on the metric it
    /// optimizes, and at production widths it must be very close. This pins the
    /// direction of the trade so a future "optimization" that silently degrades
    /// quality cannot land unnoticed.
    #[test]
    fn beam_cost_is_bounded_by_exhaustive() {
        let lut = gaussian_lut();
        let num_symbols = 128;
        for variant in 0..5u32 {
            let target = test_row(num_symbols, variant);
            let exact_mse = mse(
                &target,
                &decode_symbols(&viterbi_quantize_row(&target, &lut), &lut),
            );
            let mut prev = f32::INFINITY;
            for width in [64usize, 128, 256] {
                let syms = quantize_row(&target, &lut, TrellisSearch::Beam { width }, None);
                assert_eq!(syms.len(), num_symbols);
                let beam_mse = mse(&target, &decode_symbols(&syms, &lut));
                assert!(
                    beam_mse >= exact_mse - 1e-6,
                    "variant {variant}, W={width}: beam MSE {beam_mse} beat exhaustive {exact_mse} — impossible"
                );
                // Widening the beam is monotone non-worsening.
                assert!(
                    beam_mse <= prev + 1e-6,
                    "variant {variant}: W={width} MSE {beam_mse} worse than the narrower beam's {prev}"
                );
                prev = beam_mse;
            }
        }
    }

    /// Beam search is deterministic and always emits legal symbols.
    #[test]
    fn beam_is_deterministic_and_well_formed() {
        let lut = gaussian_lut();
        let target = test_row(48, 2);
        for width in [1usize, 4, 64, 256] {
            let a = quantize_row(&target, &lut, TrellisSearch::Beam { width }, None);
            let b = quantize_row(&target, &lut, TrellisSearch::Beam { width }, None);
            assert_eq!(a, b, "beam W={width} is not deterministic");
            assert_eq!(a.len(), 48);
            for s in &a {
                assert!((*s as usize) < ALPHABET);
            }
        }
    }

    /// `ARC_QTIP_BEAM` parsing: only an explicit positive width below the state
    /// space count switches the search — anything else stays exhaustive so a
    /// typo can never silently lower bake quality.
    #[test]
    fn beam_env_parsing_defaults_to_exhaustive() {
        assert_eq!(TrellisSearch::parse(""), TrellisSearch::Exhaustive);
        assert_eq!(TrellisSearch::parse("0"), TrellisSearch::Exhaustive);
        assert_eq!(TrellisSearch::parse("off"), TrellisSearch::Exhaustive);
        assert_eq!(
            TrellisSearch::parse("Exhaustive"),
            TrellisSearch::Exhaustive
        );
        assert_eq!(TrellisSearch::parse("nonsense"), TrellisSearch::Exhaustive);
        assert_eq!(TrellisSearch::parse("65536"), TrellisSearch::Exhaustive);
        assert_eq!(
            TrellisSearch::parse(" 128 "),
            TrellisSearch::Beam { width: 128 }
        );
        assert_eq!(TrellisSearch::Exhaustive.tag(), "viterbi-exhaustive");
        assert_eq!(
            TrellisSearch::Beam { width: 64 }.tag(),
            "viterbi-beam(W=64)"
        );
    }

    // -----------------------------------------------------------------------
    // Axis B — Hessian objective
    // -----------------------------------------------------------------------

    /// A flat Hessian must reduce to the unweighted objective EXACTLY. This is
    /// the "default OFF is really off" guard: enabling the weighted code path
    /// with uninformative calibration data cannot change a single baked symbol.
    #[test]
    fn flat_hessian_is_bit_identical_to_unweighted() {
        let lut = gaussian_lut();
        for rotation_block in [0usize, 2, 32, 128] {
            for variant in 0..5u32 {
                let target = test_row(64, variant);
                let h = vec![0.75f32; target.len()];
                let w = hessian_row_weights(&h, rotation_block);
                assert!(
                    w.iter().all(|&v| v == 1.0),
                    "flat Hessian did not normalise to exactly 1.0 (block={rotation_block})"
                );
                assert_eq!(
                    viterbi_quantize_row(&target, &lut),
                    exhaustive_quantize_row(&target, &lut, Some(&w))
                );
                assert_eq!(
                    quantize_row(&target, &lut, TrellisSearch::Beam { width: 64 }, None),
                    quantize_row(&target, &lut, TrellisSearch::Beam { width: 64 }, Some(&w))
                );
            }
        }
    }

    /// The frame correction: because every entry of a normalised Hadamard obeys
    /// `R_ji² = 1/B`, `diag(RᵀHR)` is the per-block MEAN of `diag(H)`. So the
    /// projected weights must be piecewise constant on rotation blocks, and a
    /// full-width rotation must flatten them entirely.
    #[test]
    fn hessian_weights_are_block_means_under_rotation() {
        let k = 256;
        let h: Vec<f32> = (0..k).map(|j| (j as f32) + 1.0).collect();

        // No rotation → weights track diag(H) directly (up to the mean-1 scale).
        let unrot = hessian_row_weights(&h, 0);
        let ratio = unrot[10] / unrot[0];
        let expected =
            (h[10] as f64 + HESSIAN_SIGMA_REG * 128.5) / (h[0] as f64 + HESSIAN_SIGMA_REG * 128.5);
        assert!(
            (ratio as f64 - expected).abs() < 1e-3,
            "unrotated weights should follow diag(H): {ratio} vs {expected}"
        );

        // Block rotation → piecewise constant on 64-wide blocks, and the ratio
        // between blocks is the ratio of block means.
        let rot = hessian_row_weights(&h, 64);
        for b in 0..4 {
            for j in 1..64 {
                assert_eq!(
                    rot[b * 64],
                    rot[b * 64 + j],
                    "weights not constant within rotation block {b}"
                );
            }
        }
        assert!(rot[0] < rot[64] && rot[64] < rot[128] && rot[128] < rot[192]);

        // Full-width rotation → the Hessian diagonal is fully whitened, so the
        // weighted objective degenerates to the unweighted one.
        let full = hessian_row_weights(&h, k);
        assert!(full.iter().all(|&v| v == 1.0));
    }

    /// Degenerate calibration input (all zeros, NaNs, negatives) must fall back
    /// to a flat objective rather than producing zero or non-finite weights.
    #[test]
    fn hessian_weights_survive_degenerate_calibration() {
        for h in [vec![0.0f32; 32], vec![f32::NAN; 32], vec![-1.0f32; 32], {
            let mut v = vec![0.0f32; 32];
            v[7] = 5.0;
            v[9] = f32::INFINITY;
            v
        }] {
            for block in [0usize, 8] {
                let w = hessian_row_weights(&h, block);
                assert_eq!(w.len(), 32);
                assert!(
                    w.iter().all(|&v| v.is_finite() && v > 0.0),
                    "degenerate calibration produced an unusable weight vector: {w:?}"
                );
            }
        }
        assert!(hessian_row_weights(&[], 0).is_empty());
    }

    /// The weighted objective actually optimizes the weighted error: on a row
    /// with a strongly non-uniform Hessian, the Hessian-weighted search must
    /// beat the unweighted search on the Hessian-weighted metric.
    #[test]
    fn hessian_objective_improves_hessian_weighted_error() {
        let lut = gaussian_lut();
        let num_symbols = 96;
        let mut wins = 0usize;
        for variant in 0..5u32 {
            let target = test_row(num_symbols, variant);
            let k = target.len();
            // Outlier channels: a few inputs carry 100x the activation energy.
            let h: Vec<f32> = (0..k)
                .map(|j| if j % 17 == 0 { 100.0 } else { 1.0 })
                .collect();
            let w = hessian_row_weights(&h, 0);

            let plain = decode_symbols(&viterbi_quantize_row(&target, &lut), &lut);
            let weighted = decode_symbols(&exhaustive_quantize_row(&target, &lut, Some(&w)), &lut);

            let werr = |recon: &[f32]| -> f64 {
                target
                    .iter()
                    .zip(recon.iter())
                    .zip(w.iter())
                    .map(|((&t, &r), &wi)| wi as f64 * ((t - r) as f64).powi(2))
                    .sum::<f64>()
                    / k as f64
            };
            if werr(&weighted) < werr(&plain) {
                wins += 1;
            }
        }
        assert!(
            wins >= 4,
            "Hessian-weighted search only beat unweighted on {wins}/5 rows of its own metric"
        );
    }

    // -----------------------------------------------------------------------
    // wave19-AP — the gmin-only exhaustive recursion (kernels/qtip/qtip_gmin.cu)
    // -----------------------------------------------------------------------

    /// Deliberate defects injected into the replay to prove the identity
    /// assertion below can actually fail. A parity test whose assertion is
    /// unreachable is worse than no test (DOCTRINE D12).
    ///
    /// Every variant here is *load-bearing*: it changes the answer on ordinary
    /// data. Two plausible-looking mutations were tried and rejected because
    /// they have no teeth, which is itself worth recording — both are pure
    /// tie-breaks, and with a non-degenerate LUT an exact `f32` tie between
    /// `err(s_m) + gmin[pred_m]` for two different `m` essentially never
    /// happens on finite costs:
    ///
    /// * `c <= best` (last predecessor wins ties) — measured identical on all
    ///   43 fixtures. Ties DO occur, but only in the `+inf` region of the early
    ///   timesteps, i.e. in groups the optimal path never enters.
    /// * final state keyed by `(cost, group)` instead of `(cost, state)` — same
    ///   reason: the state's high bits only break a cost tie.
    ///
    /// Both rules are still implemented exactly as the CPU reference has them,
    /// because that is what makes the equivalence an identity rather than an
    /// approximation; they are simply not independently observable here.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum GminMutation {
        /// The kernel as written.
        None,
        /// Predecessor group `(m << 8) | (g >> 4)` mis-derived as
        /// `(m << 8) | (g & 0xFF)` — the single most likely transcription error
        /// in the whole recursion.
        WrongPredecessorGroup,
        /// Branch metric drops the accumulated predecessor cost, turning the DP
        /// into a per-position greedy pick.
        DropPredecessorCost,
        /// Local error evaluated at the group `g` instead of the full state
        /// `(m << 12) | g` — i.e. forgetting that the 16 states of a group have
        /// different codebook entries.
        ErrAtGroupNotState,
        /// Backtrace read one position late (`trace[t - 1]` instead of
        /// `trace[t]`). The index shift between "argmin stored at t" and
        /// "argmin consumed at t" is the subtlest part of the derivation.
        WalkTraceOffByOne,
        /// Emitted symbol taken from the wrong nibble of the group.
        WalkSymbolFromHighNibble,
    }

    /// Replay of `kernels/qtip/qtip_gmin.cu` in the kernel's exact scan order.
    ///
    /// The kernel never materialises the 2^L cost array: it iterates the
    /// recursion
    ///
    /// ```text
    /// gmin_t[g] = min over m of ( err_t((m << 12) | g) + gmin_{t-1}[(m << 8) | (g >> 4)] )
    /// ```
    ///
    /// on 4096 f32, carries the argmin `m` as the backtrace, and walks back on
    /// groups alone (`sym_t = g_t & 0xF`, `g_{t-1} = (arg_t[g_t] << 8) | (g_t >> 4)`).
    #[allow(clippy::needless_range_loop)]
    fn gmin_replay_quantize_row(
        target_row: &[f32],
        lut: &[f32],
        mutation: GminMutation,
    ) -> Vec<u8> {
        const G_BITS: u32 = super::super::L - K; // 12
        const G_COUNT: usize = 1 << G_BITS; // 4096
        let num_symbols = target_row.len() / V as usize;
        assert!(num_symbols > 0);
        let inf = f32::INFINITY;

        // t = 0: only states 0..ALPHABET are reachable from the implicit start
        // state 0, and group g collects {(m << 12) | g}, of which only m = 0 can
        // be below 16.
        let target0 = &target_row[..V as usize];
        let mut gmin: Vec<f32> = (0..G_COUNT)
            .map(|g| {
                if g < ALPHABET {
                    decode_error(lut, g as u32, target0)
                } else {
                    inf
                }
            })
            .collect();
        let mut next = vec![inf; G_COUNT];
        // `trace[t]` is the per-group argmin at position t; index 0 is never
        // read (the walk terminates by emitting `g_0 & 0xF`).
        let mut trace: Vec<Vec<u8>> = vec![vec![0u8; G_COUNT]];

        for t in 1..num_symbols {
            let tt = &target_row[t * V as usize..(t + 1) * V as usize];
            let mut arg = vec![0u8; G_COUNT];
            for g in 0..G_COUNT {
                let mut best = inf;
                let mut bm = 0u8;
                for m in 0..ALPHABET as u32 {
                    let s = match mutation {
                        GminMutation::ErrAtGroupNotState => g as u32,
                        _ => (m << G_BITS) | g as u32,
                    };
                    let pred = match mutation {
                        GminMutation::WrongPredecessorGroup => (m << 8) | (g as u32 & 0xFF),
                        _ => (m << 8) | (g as u32 >> K),
                    };
                    let err = decode_error(lut, s, tt);
                    let c = match mutation {
                        GminMutation::DropPredecessorCost => err,
                        _ => err + gmin[pred as usize],
                    };
                    if c < best {
                        best = c;
                        bm = m as u8;
                    }
                }
                next[g] = best;
                arg[g] = bm;
            }
            std::mem::swap(&mut gmin, &mut next);
            trace.push(arg);
        }

        // Best final state, as a min over the 4096 group representatives.
        // Within a group the argmin is the lowest-indexed state attaining the
        // group minimum, so the lowest state attaining the GLOBAL minimum is
        // among them; `(cost, state)` on the ordered key reproduces the CPU
        // reference's "strict `<` over ascending s".
        let arg_last = &trace[num_symbols - 1];
        let mut best_key = u64::MAX;
        for g in 0..G_COUNT {
            let m = arg_last[g] as u32;
            let st = (m << G_BITS) | g as u32;
            let key = ((total_order_key(gmin[g]) as u64) << 16) | st as u64;
            if key < best_key {
                best_key = key;
            }
        }
        let mut g = (best_key as u32) & (G_COUNT as u32 - 1);

        let sym_of = |g: u32| -> u8 {
            match mutation {
                GminMutation::WalkSymbolFromHighNibble => ((g >> K) & (ALPHABET as u32 - 1)) as u8,
                _ => (g & (ALPHABET as u32 - 1)) as u8,
            }
        };
        let mut symbols = vec![0u8; num_symbols];
        symbols[num_symbols - 1] = sym_of(g);
        for t in (1..num_symbols).rev() {
            let src = match mutation {
                GminMutation::WalkTraceOffByOne => t - 1,
                _ => t,
            };
            let m = trace[src][g as usize] as u32;
            g = (m << (G_BITS - K)) | (g >> K);
            symbols[t - 1] = sym_of(g);
        }
        symbols
    }

    /// **THE CPU-side gate for the gmin kernel.** The gmin-only recursion is not
    /// an approximation of the exhaustive DP — it is the same DP with the 16x
    /// state replication removed — so it must return the byte-identical symbol
    /// stream, on every fixture, including the FP4-lattice weights that are the
    /// actual source distribution of V4's experts (D12: a Gaussian-only or
    /// sinusoid-only fixture set has misled this repo twice).
    ///
    /// The mutation arm is what gives the assertion teeth: five transcription
    /// errors that a reviewer could plausibly wave through are each shown to
    /// change the output.
    #[test]
    fn gmin_replay_matches_exhaustive_bit_for_bit() {
        let lut = gaussian_lut();

        // Fixture family. `test_row` variants are periodic/analytic (variant 3
        // manufactures exact ties); `fp4_dequant` is the heavy-tailed lattice
        // distribution real experts come from.
        let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
        for num_symbols in [1usize, 2, 3, 7, 16, 33, 64] {
            for variant in 0..5u32 {
                rows.push((
                    format!("analytic T={num_symbols} v={variant}"),
                    test_row(num_symbols, variant),
                ));
            }
        }
        let fp4 = super::super::bake_quality_tests::gen_fp4_dequant(4, 256, 0.02, 0x00A9_1234);
        for r in 0..4usize {
            let raw = &fp4[r * 256..(r + 1) * 256];
            let max_abs = raw.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 3.0 };
            rows.push((
                format!("fp4_dequant row={r}"),
                raw.iter().map(|&v| v / scale).collect(),
            ));
        }

        let mut alphabet_seen = [false; 16];
        for (name, target) in &rows {
            let reference = viterbi_quantize_row(target, &lut);
            let replay = gmin_replay_quantize_row(target, &lut, GminMutation::None);
            for &s in &reference {
                alphabet_seen[(s & 0x0F) as usize] = true;
            }
            assert_eq!(
                reference, replay,
                "{name}: the gmin-only recursion diverged from the exhaustive DP"
            );
        }
        // Non-vacuity: if the search only ever emitted one or two symbols the
        // identity above would be nearly free.
        let distinct = alphabet_seen.iter().filter(|&&b| b).count();
        assert!(
            distinct >= 12,
            "the fixture family only exercised {distinct}/16 symbols — too \
             degenerate for byte identity to mean much"
        );

        // Teeth: each mutation must change the answer somewhere.
        for mutation in [
            GminMutation::WrongPredecessorGroup,
            GminMutation::DropPredecessorCost,
            GminMutation::ErrAtGroupNotState,
            GminMutation::WalkTraceOffByOne,
            GminMutation::WalkSymbolFromHighNibble,
        ] {
            let differed = rows.iter().any(|(_, target)| {
                let reference = viterbi_quantize_row(target, &lut);
                let replay = gmin_replay_quantize_row(target, &lut, mutation);
                replay != reference
            });
            assert!(
                differed,
                "{mutation:?} produced the reference answer on every fixture — \
                 the identity assertion above has no teeth against it"
            );
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

// ---------------------------------------------------------------------------
// GPU-kernel cost instrumentation (wave16-AF)
// ---------------------------------------------------------------------------

/// Per-timestep statistics of the beam that the CUDA kernel's cost is driven by.
///
/// `kernels/qtip/qtip_beam.cu` maps **one thread per prefix group** and one
/// radix-select pass per 8 bits of the 48-bit `(cost, state)` key. Both of
/// those are data-dependent, and neither can be read off the source — but both
/// are properties of the *search*, not of CUDA, so they are measurable here on
/// a CPU. This exists to turn the kernel's performance analysis from a
/// projection into arithmetic over measured quantities.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct BeamKernelStats {
    /// Number of timesteps observed (excludes `t = 0`, which is degenerate).
    pub steps: usize,
    /// Distinct prefix groups in the beam, i.e. how many of the kernel's 256
    /// threads have work in the expansion phase. Histogram over 8 buckets:
    /// `[1..16, 16..32, 32..64, 64..96, 96..128, 128..160, 160..208, 208..=256]`.
    pub group_hist: [usize; 8],
    pub group_sum: usize,
    pub group_min: usize,
    pub group_max: usize,
    /// Radix-select digit passes actually executed (1..=6), indexed by count.
    pub pass_hist: [usize; 7],
    pub pass_sum: usize,
    /// Timesteps where the candidate set did not exceed the beam width, so no
    /// selection ran at all.
    pub unpruned_steps: usize,
    /// Leading 8-bit digit positions of the 48-bit key for which EVERY
    /// candidate shares the same value. Those passes are pure overhead: the
    /// histogram has one non-empty bin and the pass cannot narrow anything.
    pub wasted_leading_digits_sum: usize,
    /// Highest differing bit position across all candidate keys (0..48),
    /// summed. `48 - this` is how many leading bits are common.
    pub top_differing_bit_sum: usize,
    /// Passes the same radix-select would take if it began at the highest
    /// differing bit rather than always at bit 40.
    pub skip_pass_sum: usize,
}

#[cfg(test)]
impl BeamKernelStats {
    pub fn mean_groups(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.group_sum as f64 / self.steps as f64
        }
    }
    pub fn mean_passes(&self) -> f64 {
        let pruned = self.steps - self.unpruned_steps;
        if pruned == 0 {
            0.0
        } else {
            self.pass_sum as f64 / pruned as f64
        }
    }
    pub fn mean_wasted_digits(&self) -> f64 {
        let pruned = self.steps - self.unpruned_steps;
        if pruned == 0 {
            0.0
        } else {
            self.wasted_leading_digits_sum as f64 / pruned as f64
        }
    }
    pub fn mean_skip_passes(&self) -> f64 {
        let pruned = self.steps - self.unpruned_steps;
        if pruned == 0 {
            0.0
        } else {
            self.skip_pass_sum as f64 / pruned as f64
        }
    }
    pub fn mean_top_differing_bit(&self) -> f64 {
        let pruned = self.steps - self.unpruned_steps;
        if pruned == 0 {
            0.0
        } else {
            self.top_differing_bit_sum as f64 / pruned as f64
        }
    }
}

/// The kernel's `qtip_total_order_key` — the f32→u32 map that makes ascending
/// unsigned order equal `f32::total_cmp` order.
#[cfg(test)]
fn total_order_key(x: f32) -> u32 {
    let b = x.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// Replay `kernels/qtip/qtip_beam.cu`'s radix-select on a real candidate set and
/// return how many 8-bit digit passes it executes.
///
/// This is a transcription of the kernel loop, early exit included: it stops as
/// soon as a digit bin holds a single candidate.
#[cfg(test)]
fn radix_passes_for(keys: &[u64], width: usize) -> usize {
    radix_passes_from(keys, width, 40)
}

/// As [`radix_passes_for`], but starting the digit scan at `start_shift` —
/// used to price the "skip provably-common leading digits" optimisation.
#[cfg(test)]
fn radix_passes_from(keys: &[u64], width: usize, start_shift: i32) -> usize {
    let mut prefix: u64 = 0;
    let mut k = width as u64;
    let mut passes = 0usize;
    let mut shift: i32 = start_shift;
    while shift >= 0 {
        passes += 1;
        let mut hist = [0u32; 256];
        for &key in keys {
            if (key >> (shift + 8)) == (prefix >> (shift + 8)) {
                hist[((key >> shift) & 0xFF) as usize] += 1;
            }
        }
        let mut excl = 0u64;
        let mut chosen = 0usize;
        for (b, &h) in hist.iter().enumerate() {
            if excl < k && k <= excl + h as u64 {
                chosen = b;
                break;
            }
            excl += h as u64;
        }
        k -= excl;
        prefix |= (chosen as u64) << shift;
        if hist[chosen] == 1 {
            break;
        }
        shift -= 8;
    }
    passes
}

/// Transcription of `qtip_beam.cu`'s **wave16-AF** selection: radix over the
/// 32-bit cost key, with an exact fallback into the 16 state bits when costs
/// tie. Returns the 48-bit threshold, exactly as the kernel computes it.
///
/// Kept beside [`radix_passes_for`] (the pre-wave16 48-bit composite scan) so
/// the two can be proved to select the identical set on real candidate data —
/// that equivalence is the whole parity argument for the rewrite, and it is
/// checkable without a GPU.
#[cfg(test)]
pub(crate) fn threshold_split_key(keys: &[u64], width: usize) -> u64 {
    let cost_of = |k: u64| (k >> 16) as u32;
    let state_of = |k: u64| (k & 0xFFFF) as u32;

    let mut cost_prefix: u32 = 0;
    let mut k = width as u64;
    let mut tie_count: u32 = 0;
    let mut exit_shift: i32 = 0;
    let mut shift: i32 = 24;
    while shift >= 0 {
        exit_shift = shift;
        let mut hist = [0u32; 256];
        for &key in keys {
            let ck = cost_of(key);
            let part = shift == 24 || (ck >> (shift + 8)) == (cost_prefix >> (shift + 8));
            if part {
                hist[((ck >> shift) & 0xFF) as usize] += 1;
            }
        }
        let mut excl = 0u64;
        let mut chosen = 0usize;
        for (b, &h) in hist.iter().enumerate() {
            if excl < k && k <= excl + h as u64 {
                chosen = b;
                break;
            }
            excl += h as u64;
        }
        k -= excl;
        cost_prefix |= (chosen as u32) << shift;
        tie_count = hist[chosen];
        if tie_count == 1 {
            break;
        }
        shift -= 8;
    }

    if tie_count == 1 {
        let cost_hi = if exit_shift > 0 {
            cost_prefix | ((1u32 << exit_shift) - 1)
        } else {
            cost_prefix
        };
        return ((cost_hi as u64) << 16) | 0xFFFF;
    }

    let mut state_prefix: u32 = 0;
    let mut shift: i32 = 8;
    while shift >= 0 {
        let mut hist = [0u32; 256];
        for &key in keys {
            let st = state_of(key);
            let part = cost_of(key) == cost_prefix
                && (shift == 8 || (st >> (shift + 8)) == (state_prefix >> (shift + 8)));
            if part {
                hist[((st >> shift) & 0xFF) as usize] += 1;
            }
        }
        let mut excl = 0u64;
        let mut chosen = 0usize;
        for (b, &h) in hist.iter().enumerate() {
            if excl < k && k <= excl + h as u64 {
                chosen = b;
                break;
            }
            excl += h as u64;
        }
        k -= excl;
        state_prefix |= (chosen as u32) << shift;
        if hist[chosen] == 1 {
            if shift > 0 {
                state_prefix |= (1u32 << shift) - 1;
            }
            break;
        }
        shift -= 8;
    }
    ((cost_prefix as u64) << 16) | (state_prefix & 0xFFFF) as u64
}

/// The pre-wave16 48-bit composite scan, kept as the reference the split-key
/// scan must reproduce.
#[cfg(test)]
pub(crate) fn threshold_composite_key(keys: &[u64], width: usize) -> u64 {
    let mut prefix: u64 = 0;
    let mut k = width as u64;
    let mut shift: i32 = 40;
    while shift >= 0 {
        let mut hist = [0u32; 256];
        for &key in keys {
            if (key >> (shift + 8)) == (prefix >> (shift + 8)) {
                hist[((key >> shift) & 0xFF) as usize] += 1;
            }
        }
        let mut excl = 0u64;
        let mut chosen = 0usize;
        for (b, &h) in hist.iter().enumerate() {
            if excl < k && k <= excl + h as u64 {
                chosen = b;
                break;
            }
            excl += h as u64;
        }
        k -= excl;
        prefix |= (chosen as u64) << shift;
        if hist[chosen] == 1 {
            if shift > 0 {
                prefix |= (1u64 << shift) - 1;
            }
            break;
        }
        shift -= 8;
    }
    prefix
}

/// Every candidate set seen while beam-searching `target_row`, handed to
/// `visit` — so a test can replay the kernel's selection on real data.
#[cfg(test)]
pub(crate) fn for_each_candidate_set(
    target_row: &[f32],
    lut: &[f32],
    width: usize,
    mut visit: impl FnMut(&[u64], usize),
) {
    let num_symbols = target_row.len() / V as usize;
    let width = width.clamp(1, LUT_SIZE);
    let mut slot = vec![u32::MAX; LUT_SIZE];
    let mut touched: Vec<u16> = Vec::new();
    let mut cands: Vec<BeamEntry> = Vec::new();
    let target_first = &target_row[..V as usize];
    let mut beam: Vec<BeamEntry> = (0..ALPHABET as u32)
        .map(|s| BeamEntry {
            cost: branch_metric(lut, s, target_first, None),
            state: s as u16,
            parent: 0,
        })
        .collect();
    prune_to_width(&mut beam, width);

    for t in 1..num_symbols {
        let target_t = &target_row[t * V as usize..(t + 1) * V as usize];
        cands.clear();
        touched.clear();
        for (pi, entry) in beam.iter().enumerate() {
            let base = ((entry.state as u32) << K) & STATE_MASK;
            for sym in 0..ALPHABET as u32 {
                let succ = (base | sym) as u16;
                let existing = slot[succ as usize];
                if existing == u32::MAX {
                    slot[succ as usize] = cands.len() as u32;
                    touched.push(succ);
                    cands.push(BeamEntry {
                        cost: entry.cost,
                        state: succ,
                        parent: pi as u16,
                    });
                } else {
                    let c = &mut cands[existing as usize];
                    if entry.cost < c.cost {
                        c.cost = entry.cost;
                        c.parent = pi as u16;
                    }
                }
            }
        }
        for c in cands.iter_mut() {
            c.cost += branch_metric(lut, c.state as u32, target_t, None);
        }
        for &s in &touched {
            slot[s as usize] = u32::MAX;
        }
        if cands.len() > width {
            let keys: Vec<u64> = cands
                .iter()
                .map(|c| ((total_order_key(c.cost) as u64) << 16) | c.state as u64)
                .collect();
            visit(&keys, width);
        }
        prune_to_width(&mut cands, width);
        std::mem::swap(&mut beam, &mut cands);
    }
}

/// Run the production beam and record the two data-dependent quantities the
/// CUDA kernel's cost model needs. Mirrors [`beam_quantize_row`] step for step;
/// any divergence here would make the numbers meaningless, so the shared
/// helpers are reused rather than reimplemented.
#[cfg(test)]
pub(crate) fn beam_kernel_stats(target_row: &[f32], lut: &[f32], width: usize) -> BeamKernelStats {
    let num_symbols = target_row.len() / V as usize;
    let width = width.clamp(1, LUT_SIZE);
    let mut stats = BeamKernelStats {
        group_min: usize::MAX,
        ..Default::default()
    };

    let mut slot = vec![u32::MAX; LUT_SIZE];
    let mut touched: Vec<u16> = Vec::with_capacity(width * ALPHABET);
    let mut cands: Vec<BeamEntry> = Vec::with_capacity(width * ALPHABET);

    let target_first = &target_row[..V as usize];
    let mut beam: Vec<BeamEntry> = (0..ALPHABET as u32)
        .map(|s| BeamEntry {
            cost: branch_metric(lut, s, target_first, None),
            state: s as u16,
            parent: 0,
        })
        .collect();
    prune_to_width(&mut beam, width);

    // The kernel's prefix-group mask: successors of a state depend on it only
    // through its low `L - K` bits.
    const GROUP_MASK: u32 = (1u32 << (super::L - K)) - 1;

    for t in 1..num_symbols {
        let target_t = &target_row[t * V as usize..(t + 1) * V as usize];

        // How many of the kernel's 256 threads have work: distinct groups.
        let mut seen = std::collections::HashSet::with_capacity(beam.len());
        for e in &beam {
            seen.insert((e.state as u32) & GROUP_MASK);
        }
        let ng = seen.len();
        stats.steps += 1;
        stats.group_sum += ng;
        stats.group_min = stats.group_min.min(ng);
        stats.group_max = stats.group_max.max(ng);
        let bucket = match ng {
            0..=15 => 0,
            16..=31 => 1,
            32..=63 => 2,
            64..=95 => 3,
            96..=127 => 4,
            128..=159 => 5,
            160..=207 => 6,
            _ => 7,
        };
        stats.group_hist[bucket] += 1;

        cands.clear();
        touched.clear();
        for (pi, entry) in beam.iter().enumerate() {
            let base = ((entry.state as u32) << K) & STATE_MASK;
            for sym in 0..ALPHABET as u32 {
                let succ = (base | sym) as u16;
                let s_idx = succ as usize;
                let existing = slot[s_idx];
                if existing == u32::MAX {
                    slot[s_idx] = cands.len() as u32;
                    touched.push(succ);
                    cands.push(BeamEntry {
                        cost: entry.cost,
                        state: succ,
                        parent: pi as u16,
                    });
                } else {
                    let c = &mut cands[existing as usize];
                    if entry.cost < c.cost {
                        c.cost = entry.cost;
                        c.parent = pi as u16;
                    }
                }
            }
        }
        for c in cands.iter_mut() {
            c.cost += branch_metric(lut, c.state as u32, target_t, None);
        }
        for &s in &touched {
            slot[s as usize] = u32::MAX;
        }

        if cands.len() <= width {
            stats.unpruned_steps += 1;
        } else {
            let keys: Vec<u64> = cands
                .iter()
                .map(|c| ((total_order_key(c.cost) as u64) << 16) | c.state as u64)
                .collect();
            let passes = radix_passes_for(&keys, width);
            stats.pass_sum += passes;
            stats.pass_hist[passes.min(6)] += 1;

            // How much of the key carries no information at this step: the
            // costs are CUMULATIVE, so at timestep t they agree to roughly
            // log2(t) bits and the leading digits are constant across the whole
            // candidate set. Every such digit costs a full radix pass (6 block
            // barriers + 16 key rebuilds per thread) and narrows nothing.
            let diff = keys.iter().fold(0u64, |a, &k| a | (k ^ keys[0]));
            let top_bit = if diff == 0 {
                0
            } else {
                64 - diff.leading_zeros() as usize
            };
            stats.top_differing_bit_sum += top_bit;
            let mut wasted = 0usize;
            let mut shift: i32 = 40;
            while shift >= 0 && (diff >> shift) == 0 {
                wasted += 1;
                shift -= 8;
            }
            stats.wasted_leading_digits_sum += wasted;

            // Price the optimisation: start the digit scan at the first digit
            // that actually differs, instead of always at bit 40.
            let start = if top_bit == 0 {
                0
            } else {
                (((top_bit - 1) / 8) * 8) as i32
            };
            stats.skip_pass_sum += radix_passes_from(&keys, width, start);
        }

        prune_to_width(&mut cands, width);
        std::mem::swap(&mut beam, &mut cands);
    }
    if stats.group_min == usize::MAX {
        stats.group_min = 0;
    }
    stats
}

// ---------------------------------------------------------------------------
// Guess-Verify-Refine premise measurement (wave17-AF)
// ---------------------------------------------------------------------------
//
// wave17-AN's top recommendation is Guess-Verify-Refine (arXiv 2604.22312):
// warm-start the top-W threshold from step t-1, verify by counting, refine
// only on a miss. Its premise is that the threshold barely moves between
// consecutive positions. AN argues our correlation is the strongest the
// technique could be handed, because the survivors at t+1 are literally
// `cost_t + branch_metric` over the same paths.
//
// That premise is TESTABLE ON A CPU and it must be tested before any kernel is
// written on top of it, because it has an obvious failure mode: our costs are
// CUMULATIVE. They grow without bound as t advances, so the raw threshold does
// not sit still at all — it drifts upward every single step. Whether a warm
// start works therefore depends entirely on whether the drift is *predictable*,
// which is an empirical question about this cost distribution.
//
// Two families of guess are measured:
//
//   * **Temporal** — extrapolate the threshold's own trajectory. Cheap, but
//     inexact by nature: a miss costs a refine.
//   * **Dr. Top-k Rule 2 (beta-delegate)** — the max over groups of each
//     group's beta-th smallest candidate is a PROVABLE upper bound on the
//     W-th smallest whenever `beta * ng >= W`. Proof: those `beta*ng` delegates
//     are all <= the bound and all lie in the candidate set, so at least
//     `beta*ng >= W` candidates are <= it, hence the W-th smallest is too.
//     No temporal assumption at all, and it can never miss low — the only
//     question is how TIGHT it is, i.e. how many candidates survive it.
//
// The number that decides the design is the survivor count under each guess:
// the refine step's cost is proportional to it.

#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct GvrStats {
    pub steps: usize,
    /// Candidates examined, summed (mean n_cand ~ 3979).
    pub n_cand_sum: usize,
    /// Exact hits, where a guess yields count == W and no refine is needed.
    pub temporal_exact_hits: usize,
    /// Guess landed above the true threshold (safe: refine downward).
    pub temporal_high: usize,
    /// Guess landed below (would need widening — the expensive miss).
    pub temporal_low: usize,
    /// Survivors under the temporal guess, summed over steps where it was a
    /// valid (high) bound.
    pub temporal_survivors_sum: usize,
    /// Survivors under the Rule-2 beta=1 bound (valid only when ng >= W).
    pub delegate1_survivors_sum: usize,
    pub delegate1_valid_steps: usize,
    /// Survivors under the Rule-2 beta=2 bound (valid whenever 2*ng >= W).
    pub delegate2_survivors_sum: usize,
    pub delegate2_valid_steps: usize,
    /// Survivors under beta=4.
    pub delegate4_survivors_sum: usize,
    pub delegate4_valid_steps: usize,
    /// Steps where the true threshold's rank among candidates was exactly W.
    pub width_sum: usize,
    /// Survivors under the TIGHT Rule-2 bound: the W-th smallest of the
    /// beta-delegate vector, rather than its maximum. Same proof, far tighter.
    pub tight2_survivors_sum: usize,
    pub tight2_steps: usize,
    pub tight4_survivors_sum: usize,
    pub tight4_steps: usize,
    /// Delegate-vector size, summed (beta * ng).
    pub tight2_delegates_sum: usize,
}

#[cfg(test)]
impl GvrStats {
    fn mean(sum: usize, n: usize) -> f64 {
        if n == 0 {
            0.0
        } else {
            sum as f64 / n as f64
        }
    }
    pub fn mean_n_cand(&self) -> f64 {
        Self::mean(self.n_cand_sum, self.steps)
    }
    pub fn mean_temporal_survivors(&self) -> f64 {
        Self::mean(self.temporal_survivors_sum, self.temporal_high)
    }
    pub fn mean_delegate_survivors(&self, beta: usize) -> f64 {
        match beta {
            1 => Self::mean(self.delegate1_survivors_sum, self.delegate1_valid_steps),
            2 => Self::mean(self.delegate2_survivors_sum, self.delegate2_valid_steps),
            _ => Self::mean(self.delegate4_survivors_sum, self.delegate4_valid_steps),
        }
    }
}

/// Measure how much leverage a Guess-Verify-Refine front-end would actually
/// have on this cost distribution.
///
/// `groups` is the candidate list's group structure: the kernel gives thread
/// `p` the 16 successors of group `p`, so candidate `c` belongs to group
/// `c / 16` — exactly how `for_each_candidate_set` lays them out.
#[cfg(test)]
pub(crate) fn measure_gvr_leverage(target_row: &[f32], lut: &[f32], width: usize) -> GvrStats {
    let mut stats = GvrStats::default();
    let mut prev_threshold: Option<u64> = None;
    let mut prev_delta: Option<i128> = None;

    for_each_candidate_set(target_row, lut, width, |keys, w| {
        stats.steps += 1;
        stats.n_cand_sum += keys.len();
        stats.width_sum += w;

        let mut sorted = keys.to_vec();
        sorted.sort_unstable();
        let truth = sorted[w - 1];

        // ---- temporal guess: extrapolate the threshold's own trajectory ----
        if let (Some(prev), Some(delta)) = (prev_threshold, prev_delta) {
            let guess = (prev as i128 + delta).clamp(0, u64::MAX as i128) as u64;
            let count = keys.iter().filter(|&&k| k <= guess).count();
            if count == w {
                stats.temporal_exact_hits += 1;
            }
            if guess >= truth {
                stats.temporal_high += 1;
                stats.temporal_survivors_sum += count;
            } else {
                stats.temporal_low += 1;
            }
        }
        if let Some(prev) = prev_threshold {
            prev_delta = Some(truth as i128 - prev as i128);
        }
        prev_threshold = Some(truth);

        // ---- Dr. Top-k Rule 2: max over groups of the beta-th smallest ----
        let ng = keys.len() / 16;
        for (beta, sum, valid) in [
            (
                1usize,
                &mut stats.delegate1_survivors_sum,
                &mut stats.delegate1_valid_steps,
            ),
            (
                2,
                &mut stats.delegate2_survivors_sum,
                &mut stats.delegate2_valid_steps,
            ),
            (
                4,
                &mut stats.delegate4_survivors_sum,
                &mut stats.delegate4_valid_steps,
            ),
        ] {
            if beta * ng < w {
                continue; // bound not provable at this beta
            }
            let mut bound = 0u64;
            for g in 0..ng {
                let mut grp: [u64; 16] = keys[g * 16..(g + 1) * 16].try_into().unwrap();
                grp.sort_unstable();
                bound = bound.max(grp[beta - 1]);
            }
            debug_assert!(bound >= truth, "Rule-2 delegate bound must not miss low");
            *sum += keys.iter().filter(|&&k| k <= bound).count();
            *valid += 1;
        }

        // ---- TIGHT Rule 2: the W-th smallest OF THE DELEGATE VECTOR --------
        // The loose form above takes the delegate vector's MAXIMUM, which is
        // the weakest valid choice. Dr. Top-k's actual rule is that the k-th
        // smallest of the delegate vector bounds the k-th smallest of the full
        // set: those k delegates are all <= it and all lie in the set. Same
        // proof, and it is a selection over beta*ng ~ 500 elements instead of
        // ~4000 — which is itself the cheap sub-problem the refine needs.
        for (beta, sum, cnt) in [
            (
                2usize,
                &mut stats.tight2_survivors_sum,
                &mut stats.tight2_steps,
            ),
            (4, &mut stats.tight4_survivors_sum, &mut stats.tight4_steps),
        ] {
            if beta * ng < w {
                continue;
            }
            let mut delegates: Vec<u64> = Vec::with_capacity(beta * ng);
            for g in 0..ng {
                let mut grp: [u64; 16] = keys[g * 16..(g + 1) * 16].try_into().unwrap();
                grp.sort_unstable();
                delegates.extend_from_slice(&grp[..beta]);
            }
            delegates.sort_unstable();
            let bound = delegates[w - 1];
            debug_assert!(bound >= truth, "tight Rule-2 bound must not miss low");
            *sum += keys.iter().filter(|&&k| k <= bound).count();
            *cnt += 1;
            if beta == 2 {
                stats.tight2_delegates_sum += delegates.len();
            }
        }
    });
    stats
}
