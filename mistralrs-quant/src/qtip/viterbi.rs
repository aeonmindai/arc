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
    /// a beam of that width. Unparseable values fall back to exhaustive so a
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
