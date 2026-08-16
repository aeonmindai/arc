//! QTIP "bitshift trellis" with a **computed codebook** — the `qtip2b` rung.
//!
//! ## Why this format exists (Arc Stage 3)
//!
//! The sibling LUT rung (`qtip2`, K=4/V=2, [`super::QtipLayer`]) decodes each
//! trellis state by gathering from a 512 KB global-memory LUT. Measured on
//! H200 the decode GEMV is **instruction/latency bound, not bandwidth bound**
//! (388 GB/s ≈ 8% of HBM peak; ncu attributes the stall to the data-dependent
//! LUT gather + the serial state chain). The fix — proven by QTIP's own 3INST
//! code and exllamav3's EXL3 — is to *compute* the codeword from the state in
//! registers with ~3 ALU ops, eliminating the LUT load entirely and freeing
//! shared memory for weight staging.
//!
//! ## Format
//!
//! - Trellis: L=16 state bits, K=2 bits/symbol, V=1 weight/symbol
//!   → **2 bits per weight** (same rate as the LUT rung).
//! - Transition: `state_t = ((state_{t-1} << 2) | sym_t) & 0xFFFF`, initial
//!   state 0 (identical recurrence family to the LUT rung, narrower symbol).
//! - Codeword (the "3INST" computed codebook, MCG variant):
//!   ```text
//!   x = state * 0xCAF6A435          (32-bit wrapping multiply — MCG)
//!   m = (x & 0x8FFF8FFF) ^ 0x3B603B60
//!   w = f32(fp16(m >> 16)) + f32(fp16(m & 0xFFFF))
//!   ```
//!   The mask keeps each half's sign bit + 12 low bits; the XOR pins the
//!   exponent window so each half is a small fp16; the sum of the two
//!   quasi-uniform halves is approximately Gaussian. Measured over all 2^16
//!   states: mean ≈ 0, σ = 1.2064, max |w| = 3.97, quantiles matching a
//!   Gaussian of that σ to within ~1%.
//!
//!   The multiplier `0xCAF6A435` is the spectrally-optimized constant from
//!   exllamav3 PR #26: a pure MCG (no additive constant) saves one
//!   instruction vs Cornell's LCG 3INST (`x*89226354 + 64248484`) and
//!   measures 4–8% better KLD at 2 bits. Upstream exllamav3 rejected it only
//!   for legacy-format compatibility, which this fresh format does not have.
//!
//!   The two fp16 halves are summed **in f32** (conversions are exact, the
//!   f32 add is correctly rounded) so the CPU reference and the CUDA kernel
//!   (`__half2float + fadd`) are bit-identical — unlike `__hadd`, whose
//!   single-rounded fp16 sum can differ from the half-crate's
//!   double-rounded emulation in rare ties.
//!
//! - Packing: 4 symbols per byte, LSB-first
//!   (`byte = s0 | s1<<2 | s2<<4 | s3<<6`).
//! - Per-row scale: `max|row| / 3.62` where 3.62 = 3.0 × codebook σ — the
//!   LUT rung's `max/3` policy renormalized to this codebook's spread.
//! - Rotation: the same block-diagonal D·H·D Hadamard incoherence rotation
//!   (same seed, same block policy) as the LUT rung, applied to weight rows
//!   at quantize time and activation rows at forward time.
//!
//! ## Storage / serialization
//!
//! Unlike the LUT rung there is **no codebook tensor**: only the 4-byte MCG
//! multiplier is persisted (forward-compatible with future retuned
//! constants). UQFF tag: [`QuantizedSerdeType::Qtip2b`]. 2-D vs 3-D
//! (stacked-expert) layout is recovered from the self-describing rank of
//! `blocks` — the same mechanism the LUT rung uses (UQFF v0.2.1, PR #3):
//! rank-3 `blocks [E, N, packed_K]` / `row_scales [E, N]` share the 2-D
//! field order with no extra header bytes, and 3-D stacks are bias-free by
//! contract.
//!
//! ## Scope shipped here
//!
//! - CPU: full quantize (Viterbi + greedy, 2-D and 3-D expert stacks),
//!   dequantize, forward, gather_forward, UQFF round-trip. All exercised by
//!   colocated tests that run on macOS.
//! - CUDA: kernels in `kernels/qtip/qtip_bitshift.cu` (in-register decode
//!   GEMV with multiple independent trellis streams per warp, dequantize,
//!   prefix-grouped Viterbi/greedy quantize). Parity tests are gated on the
//!   `cuda` feature so the GPU session can run them; CI's nvcc lanes are the
//!   compile gate.

use std::{
    borrow::Cow,
    io::Cursor,
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{DType, Device, Result, Tensor};

use crate::{
    turboquant::wht::generate_signs,
    utils::{deserialize_tensor, serialize_tensor, version_is_compatible, UQFF_VERSION},
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, QuantizedSerdeType,
};

use super::grouped::ExpertBpwTable;
use super::{
    apply_block_rotation, rotation_block_size, QtipMode, QtipRotation, QtipSearchDetail,
    QtipSearchStamp, TrellisSearch, QTIP_ROTATION_SEED,
};

#[cfg(feature = "cuda")]
use super::cuda_ops;

/// Trellis state width in bits (matches the LUT rung and QTIP paper).
pub const L2B: u32 = 16;
/// Bits per symbol. K=2 with V=1 yields 2 bits per weight.
pub const K2B: u32 = 2;
/// State mask: (1 << L) - 1.
const STATE_MASK_2B: u32 = (1u32 << L2B) - 1;
/// Number of trellis states (= computed-codebook size).
pub(crate) const CB_SIZE_2B: usize = 1usize << L2B;
/// Symbol alphabet size: 2^K = 4.
const ALPHABET_2B: usize = 1usize << K2B;
/// Symbols packed per byte (2 bits each).
const SYMS_PER_BYTE: usize = 4;

/// The spectrally-optimized MCG multiplier (exllamav3 PR #26).
pub const QTIP2B_MCG_MULT: u32 = 0xCAF6_A435;
/// Per-half keep mask: sign bit + 12 low bits of each 16-bit half.
const QTIP2B_MASK: u32 = 0x8FFF_8FFF;
/// Exponent-window XOR pattern (0x3B60 per fp16 half).
const QTIP2B_XOR: u32 = 0x3B60_3B60;

/// Row-scale divisor: scale = max|row| / QTIP2B_SCALE_DIVISOR.
/// 3.62 = 3.0 × 1.2064 (measured codebook σ) — the LUT rung's `max/3`
/// policy renormalized to this codebook's spread so scaled targets see the
/// same effective dynamic range.
pub(crate) const QTIP2B_SCALE_DIVISOR: f32 = 3.62;

/// Compute the codeword for a trellis `state` (low 16 bits significant).
///
/// This is the whole codebook: `IMAD + LOP3 + 2×cvt + FADD` on GPU, and the
/// exact same arithmetic here. See the module docs for the derivation and
/// the f32-sum parity rationale.
#[inline]
pub fn mcg_codeword(state: u32, mult: u32) -> f32 {
    let x = (state & STATE_MASK_2B).wrapping_mul(mult);
    let m = (x & QTIP2B_MASK) ^ QTIP2B_XOR;
    let hi = half::f16::from_bits((m >> 16) as u16);
    let lo = half::f16::from_bits((m & 0xFFFF) as u16);
    hi.to_f32() + lo.to_f32()
}

/// Materialize the full 2^16-entry codebook. Only used at **quantize** time
/// (the Viterbi inner loop wants a flat table); decode paths compute
/// codewords on the fly and never allocate this.
pub(crate) fn computed_codebook(mult: u32) -> Vec<f32> {
    (0..CB_SIZE_2B as u32)
        .map(|s| mcg_codeword(s, mult))
        .collect()
}

// ---------------------------------------------------------------------------
// Symbol pack / unpack (4 × 2-bit symbols per byte, LSB-first).
// ---------------------------------------------------------------------------

fn pack_symbols_2b(symbols: &[u8]) -> Vec<u8> {
    debug_assert_eq!(symbols.len() % SYMS_PER_BYTE, 0);
    let mut packed = vec![0u8; symbols.len() / SYMS_PER_BYTE];
    for (t, &sym) in symbols.iter().enumerate() {
        packed[t / SYMS_PER_BYTE] |= (sym & 0x03) << (2 * (t % SYMS_PER_BYTE));
    }
    packed
}

#[inline]
fn unpack_symbol_2b(packed: &[u8], t: usize) -> u32 {
    ((packed[t / SYMS_PER_BYTE] >> (2 * (t % SYMS_PER_BYTE))) & 0x03) as u32
}

/// Decode one packed row into `out` (length `num_symbols`), applying `scale`.
/// Codewords are computed on the fly — no codebook table involved.
fn decode_row_2b(packed: &[u8], num_symbols: usize, mult: u32, scale: f32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), num_symbols);
    let mut state: u32 = 0;
    for (t, o) in out.iter_mut().enumerate() {
        let sym = unpack_symbol_2b(packed, t);
        state = ((state << K2B) | sym) & STATE_MASK_2B;
        *o = mcg_codeword(state, mult) * scale;
    }
}

// ---------------------------------------------------------------------------
// Viterbi (K=2, V=1) — group-min dynamic program.
// ---------------------------------------------------------------------------

/// Number of predecessor groups: all states sharing high (L-K) bits
/// `g = s >> K` have the identical predecessor set
/// `{ (j << (L-K)) | g : j ∈ [0, 2^K) }` and therefore the identical argmin.
const NUM_GROUPS_2B: usize = 1usize << (L2B - K2B) as usize; // 16384

/// Viterbi-quantize a single (already scaled) row into the optimal symbol
/// sequence for the computed codebook. Mirrors
/// [`super::viterbi::viterbi_quantize_row`] with K=2/V=1 parameters and the
/// same group-min optimization (bit-identical to the naive per-state scan —
/// same j-scan order, same strict-`<` tie-break; see the regression test).
///
/// Backtrace is stored **per group** (16384 entries/timestep) rather than
/// per state, matching the prefix-grouped CUDA kernel: at backtrack time
/// `j = bt[t][s >> K]`, `prev = (j << (L-K)) | (s >> K)`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn viterbi_quantize_row_2b(target_row: &[f32], codebook: &[f32]) -> Vec<u8> {
    let num_symbols = target_row.len();
    assert!(
        num_symbols > 0,
        "viterbi_quantize_row_2b requires at least one symbol position"
    );
    debug_assert_eq!(codebook.len(), CB_SIZE_2B);

    let inf = f32::INFINITY;
    let mut prev_cost = vec![inf; CB_SIZE_2B];
    let mut curr_cost = vec![inf; CB_SIZE_2B];
    let mut backtrace: Vec<Vec<u8>> = Vec::with_capacity(num_symbols.saturating_sub(1));

    // t=0: the initial state is 0, so after shifting in sym_0 only states
    // s ∈ [0, ALPHABET) are reachable — matching the production decoder.
    for s in 0..ALPHABET_2B {
        let d = codebook[s] - target_row[0];
        prev_cost[s] = d * d;
    }

    let mut group_cost = vec![inf; NUM_GROUPS_2B];

    for t in 1..num_symbols {
        let target_t = target_row[t];
        let mut bt_t = vec![0u8; NUM_GROUPS_2B];

        // Phase 1: per-group min over the 4 shared predecessors.
        for g in 0..NUM_GROUPS_2B {
            let mut best_cost = inf;
            let mut best_j: u8 = 0;
            for j in 0..ALPHABET_2B as u32 {
                let p = (j << (L2B - K2B)) | g as u32;
                let c = prev_cost[p as usize];
                if c < best_cost {
                    best_cost = c;
                    best_j = j as u8;
                }
            }
            group_cost[g] = best_cost;
            bt_t[g] = best_j;
        }

        // Phase 2: per-state cost = local decode error + group predecessor min.
        for s in 0..CB_SIZE_2B {
            let d = codebook[s] - target_t;
            curr_cost[s] = d * d + group_cost[s >> K2B];
        }

        backtrace.push(bt_t);
        std::mem::swap(&mut prev_cost, &mut curr_cost);
    }

    // Argmin over final costs.
    let mut best_final = 0u32;
    let mut best_final_cost = inf;
    for s in 0..CB_SIZE_2B {
        if prev_cost[s] < best_final_cost {
            best_final_cost = prev_cost[s];
            best_final = s as u32;
        }
    }

    // Backtrack.
    let mut symbols = vec![0u8; num_symbols];
    let mut s = best_final;
    symbols[num_symbols - 1] = (s & (ALPHABET_2B as u32 - 1)) as u8;
    for t in (1..num_symbols).rev() {
        let g = s >> K2B;
        let j = backtrace[t - 1][g as usize] as u32;
        let prev_s = (j << (L2B - K2B)) | g;
        symbols[t - 1] = (prev_s & (ALPHABET_2B as u32 - 1)) as u8;
        s = prev_s;
    }

    symbols
}

// ---------------------------------------------------------------------------
// Beam search (K=2, V=1) — the pruned twin of `viterbi_quantize_row_2b`.
// ---------------------------------------------------------------------------

/// One surviving trellis state at a timestep. Mirrors
/// [`super::viterbi`]'s `BeamEntry` with K=2 geometry.
#[derive(Clone, Copy)]
struct BeamEntry2b {
    /// Cumulative path cost into this state. During candidate generation this
    /// transiently holds the *predecessor* cost (the local error is added once
    /// per distinct successor, after merging).
    cost: f32,
    /// Trellis state (`L = 16` bits, so `u16` is exact).
    state: u16,
    /// Index of the chosen predecessor in the previous timestep's beam.
    parent: u16,
}

/// Keep the best `width` entries and restore ascending-state order.
///
/// Selection is by `(cost, state)` under [`f32::total_cmp`] so the survivor set
/// is fully deterministic; the trailing sort by state is what makes the next
/// timestep's predecessor visit order match the exhaustive group-min scan (and
/// therefore reproduce its "lowest predecessor state wins a tie" rule).
fn prune_to_width_2b(beam: &mut Vec<BeamEntry2b>, width: usize) {
    if beam.len() > width {
        beam.select_nth_unstable_by(width - 1, |a, b| {
            a.cost.total_cmp(&b.cost).then(a.state.cmp(&b.state))
        });
        beam.truncate(width);
    }
    beam.sort_unstable_by_key(|e| e.state);
}

/// Pruned Viterbi for the K=2/V=1 bitshift trellis, keeping the best `width`
/// states per timestep.
///
/// **This is a strictly worse search than [`viterbi_quantize_row_2b`] and a
/// much cheaper one** (DOCTRINE D4: declared, never hidden). Complexity per
/// timestep is `O(width · 2^K)` candidate generations plus one `O(width · 2^K)`
/// selection, versus `O(2^L)` for the exhaustive DP — at `width = 256` that is
/// 1024 candidates instead of 65536 states, per symbol.
///
/// `width >= 2^L` prunes nothing and reproduces the exhaustive DP bit for bit,
/// which is what `beam_2b_unpruned_matches_exhaustive_bit_for_bit` pins.
///
/// K=2 vs the LUT rung's K=4: 4 successors per predecessor instead of 16, and
/// the dedup key space (successor states reachable from the beam) is
/// `width · 4` instead of `width · 16`. Everything else — the merge rule, the
/// selection order, the tie-break — is identical, because both rungs share the
/// transition `s_t = ((s_{t-1} << K) | sym) & (2^L − 1)`.
pub(crate) fn beam_quantize_row_2b(target_row: &[f32], codebook: &[f32], width: usize) -> Vec<u8> {
    let num_symbols = target_row.len();
    assert!(
        num_symbols > 0,
        "beam_quantize_row_2b requires at least one symbol position"
    );
    debug_assert_eq!(codebook.len(), CB_SIZE_2B);
    // A beam wider than the state space prunes nothing; clamp so `u16` indices
    // (and the dedup slot table) stay valid.
    let width = width.clamp(1, CB_SIZE_2B);

    // Dedup table: successor state -> index into `cands`. Allocated once and
    // cleared through `touched` so we never pay a 2^L memset per timestep.
    let mut slot = vec![u32::MAX; CB_SIZE_2B];
    let mut touched: Vec<u16> = Vec::with_capacity(width * ALPHABET_2B);
    let mut cands: Vec<BeamEntry2b> = Vec::with_capacity(width * ALPHABET_2B);

    // Compacted backtrace, flat: `(state, parent)` for every surviving entry of
    // every timestep, with one start offset per timestep.
    let mut trace: Vec<(u16, u16)> = Vec::with_capacity(num_symbols * width.min(ALPHABET_2B * 4));
    let mut trace_off: Vec<u32> = Vec::with_capacity(num_symbols);

    // t = 0: the decoder starts from state 0, so exactly the `2^K` states
    // s ∈ [0, ALPHABET) are reachable, in ascending order.
    let mut beam: Vec<BeamEntry2b> = (0..ALPHABET_2B as u32)
        .map(|s| {
            let d = codebook[s as usize] - target_row[0];
            BeamEntry2b {
                cost: d * d,
                state: s as u16,
                parent: 0,
            }
        })
        .collect();
    prune_to_width_2b(&mut beam, width);
    trace_off.push(0);
    trace.extend(beam.iter().map(|e| (e.state, e.parent)));

    for &target_t in target_row.iter().skip(1) {
        cands.clear();
        touched.clear();

        // Expand: every surviving state × every symbol. The beam is sorted by
        // state ascending, so for a fixed successor the predecessors arrive in
        // ascending `j` — matching the exhaustive group-min scan order — and
        // strict `<` keeps the smallest `j` on ties.
        for (pi, entry) in beam.iter().enumerate() {
            let base = ((entry.state as u32) << K2B) & STATE_MASK_2B;
            for sym in 0..ALPHABET_2B as u32 {
                let succ = (base | sym) as u16;
                let s_idx = succ as usize;
                let existing = slot[s_idx];
                if existing == u32::MAX {
                    slot[s_idx] = cands.len() as u32;
                    touched.push(succ);
                    cands.push(BeamEntry2b {
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
            let d = codebook[c.state as usize] - target_t;
            c.cost += d * d;
        }

        // Release the dedup slots for the next timestep.
        for &s in &touched {
            slot[s as usize] = u32::MAX;
        }

        prune_to_width_2b(&mut cands, width);
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
        symbols[t] = (state as u32 & (ALPHABET_2B as u32 - 1)) as u8;
        idx = parent as usize;
    }
    symbols
}

/// Search-strategy dispatch for this rung, mirroring
/// [`super::viterbi::quantize_row`].
///
/// [`TrellisSearch::Exhaustive`] is the default and the best quality this rung
/// can produce. A beam is faster and slightly worse, and it is recorded in the
/// artifact (`QtipSearchDetail`) so a checkpoint can never claim a search it
/// did not run.
pub(crate) fn quantize_row_2b(
    target_row: &[f32],
    codebook: &[f32],
    search: TrellisSearch,
) -> Vec<u8> {
    match search {
        TrellisSearch::Exhaustive => viterbi_quantize_row_2b(target_row, codebook),
        // A beam at least as wide as the state space prunes nothing and IS the
        // exhaustive DP; run the cheaper implementation of the same function.
        TrellisSearch::Beam { width } if width >= CB_SIZE_2B => {
            viterbi_quantize_row_2b(target_row, codebook)
        }
        TrellisSearch::Beam { width } => beam_quantize_row_2b(target_row, codebook, width),
    }
}

/// Greedy quantizer: at each step pick the locally-best of the 4 candidate
/// symbols given the current state. Fast, suboptimal — kept for parity with
/// the LUT rung's mode selection (3-D expert stacks default to greedy).
fn greedy_quantize_row_2b(target_row: &[f32], codebook: &[f32]) -> Vec<u8> {
    let mut state: u32 = 0;
    let mut syms = vec![0u8; target_row.len()];
    for (t, &target) in target_row.iter().enumerate() {
        let mut best_sym: u8 = 0;
        let mut best_err = f32::INFINITY;
        for sym in 0..ALPHABET_2B as u32 {
            let next_state = ((state << K2B) | sym) & STATE_MASK_2B;
            let d = codebook[next_state as usize] - target;
            let err = d * d;
            if err < best_err {
                best_err = err;
                best_sym = sym as u8;
            }
        }
        state = ((state << K2B) | best_sym as u32) & STATE_MASK_2B;
        syms[t] = best_sym;
    }
    syms
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// QTIP 2-bit computed-codebook ("bitshift trellis") weight layer — the
/// `qtip2b` ISQ rung. Sibling of [`super::QtipLayer`] (the LUT rung); see
/// the module docs for the format.
///
/// # Storage layout
///
/// ## 2-D mode (`num_experts == None`)
/// - `blocks`:     `[N, K_in / 4]` U8 (four K=2 symbols per byte)
/// - `row_scales`: `[N]` F32
///
/// ## 3-D mode (`num_experts == Some(E)`) — stacked MoE experts
/// - `blocks`:     `[E, N, K_in / 4]` U8
/// - `row_scales`: `[E, N]` F32
///
/// Rotation signs (when present) are shared across experts, exactly as in
/// the LUT rung — the rotation is determined solely by `K_in`.
#[derive(Debug)]
pub struct Qtip2bLayer {
    /// Packed K=2 symbols, four per byte, LSB-first.
    blocks: Tensor,
    /// Per-row scale factor, F32.
    row_scales: Tensor,
    /// Optional bias `[N]` (2-D mode only).
    bias: Option<Tensor>,
    /// Input feature dim.
    in_features: usize,
    /// `Some(E)` for stacked experts, `None` for a plain 2-D linear.
    num_experts: Option<usize>,
    /// Hadamard incoherence rotation signs `[in_features]` F32 ±1.
    rotation_signs: Option<Tensor>,
    /// Block size for the block-diagonal Hadamard rotation. 0 when disabled.
    rotation_block: usize,
    /// MCG multiplier defining the computed codebook. Persisted in UQFF so a
    /// future retuned constant can't silently mis-decode old checkpoints.
    mcg_mult: u32,
    /// Per-expert bit-width descriptors for the grouped-GEMM dispatch
    /// (`Some` iff this is a 3-D expert stack). Uniform 2-bit today; the
    /// table exists so mixed-precision stacks (4-bit hot experts) only have
    /// to extend the per-class launch loop. See `super::grouped`.
    expert_bpw: Option<ExpertBpwTable>,
    /// Which trellis search produced these blocks. Serialized into UQFF from
    /// 0.3.0 and checked at load (DOCTRINE D4 §3).
    search: QtipSearchStamp,
    /// *Which* trellis search (UQFF ≥ 0.3.0 flags byte).
    ///
    /// This field was added while the rung had only
    /// [`viterbi_quantize_row_2b`] — "so both rungs share ONE wire format
    /// rather than diverging the moment qtip2b grows a beam kernel". That
    /// moment is wave46-BX: the rung now has [`beam_quantize_row_2b`] and
    /// `kernels/qtip/qtip2b_beam.cu`, so a bake records
    /// [`QtipSearchDetail::EXHAUSTIVE_MSE`] only when it actually ran the
    /// exhaustive DP and `Known { beam_width: Some(W), .. }` when it ran a
    /// beam. The objective bit stays false by construction: this rung has no
    /// Hessian-weighted branch metric.
    search_detail: QtipSearchDetail,
}

impl Qtip2bLayer {
    /// **The production door** for this rung. Refuses [`QtipMode::Greedy`] in
    /// every build (D4) and takes its rotation from [`QtipRotation::for_mode`],
    /// the single policy table shared with the LUT rung — no local
    /// `matches!(mode, Viterbi)` that a future mode could slip past.
    pub fn quantize_with_mode(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
    ) -> Result<Arc<dyn QuantMethod>> {
        mode.deny_greedy("Qtip2bLayer::quantize_with_mode")?;
        let use_rotation = QtipRotation::for_mode(mode).enabled();
        Self::quantize_with_options(weight, bias, device, mode, use_rotation)
    }

    /// The **search axis**, read from `ARC_QTIP_BEAM` exactly like the LUT
    /// rung's [`super::QtipBakeConfig`].
    ///
    /// Unset (the default) is [`TrellisSearch::Exhaustive`] — the best quality
    /// this rung can produce, and what every artifact baked before the beam
    /// kernel landed contains. A beam is much faster and measurably *worse*
    /// (wave19-AP: exhaustive beat W=256 on 8 of 9 fixture cells on the LUT
    /// rung); it never happens unless an operator asks for it by name, and when
    /// it does the width is stamped into the artifact and re-checked at load.
    fn env_search() -> TrellisSearch {
        TrellisSearch::from_env()
    }

    /// **The greedy fixture door — this crate's tests only (DOCTRINE D4).**
    /// See [`super::QtipLayer::quantize_greedy_fixture`]; compiled only under
    /// `cfg(test)` of `mistralrs-quant`.
    #[cfg(test)]
    pub(crate) fn quantize_greedy_fixture(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Arc<dyn QuantMethod>> {
        Self::quantize_with_options(
            weight,
            bias,
            device,
            QtipMode::Greedy,
            QtipRotation::for_mode(QtipMode::Greedy).enabled(),
        )
    }

    /// Lowest-level quantize entry: explicit mode + explicit rotation flag.
    /// Rank dispatch: 2-D `[N, K]` → per-linear quantize; 3-D `[E, N, K]` →
    /// stacked-expert quantize (bias unsupported there).
    pub fn quantize_with_options(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
    ) -> Result<Arc<dyn QuantMethod>> {
        Self::quantize_with_options_search(
            weight,
            bias,
            device,
            mode,
            use_rotation,
            Self::env_search(),
        )
    }

    /// [`Self::quantize_with_options`] with the trellis search named
    /// explicitly instead of read from the environment. Tests drive this so
    /// they can pin a width without mutating process-global state.
    pub fn quantize_with_options_search(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
        search: TrellisSearch,
    ) -> Result<Arc<dyn QuantMethod>> {
        if weight.dims().len() == 3 {
            if bias.is_some() {
                candle_core::bail!(
                    "qtip2b 3-D quantize: bias not supported for stacked-expert weights"
                );
            }
            return Self::quantize_with_options_3d_search(
                weight,
                device,
                mode,
                use_rotation,
                search,
            );
        }
        let layer = Self::quantize_with_options_concrete_search(
            weight,
            bias,
            device,
            mode,
            use_rotation,
            search,
        )?;
        Ok(Arc::new(layer))
    }

    /// Concrete-typed 2-D quantize. Same role as the LUT rung's
    /// `quantize_with_options_concrete`: used by the 3-D path and by tests
    /// that need typed field access.
    pub fn quantize_with_options_concrete(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
    ) -> Result<Self> {
        Self::quantize_with_options_concrete_search(
            weight,
            bias,
            device,
            mode,
            use_rotation,
            Self::env_search(),
        )
    }

    /// [`Self::quantize_with_options_concrete`] with an explicit trellis
    /// search. On CUDA the plan is put through [`super::cuda_search_plan`]
    /// before it reaches a kernel, so the width recorded in `search_detail` is
    /// what ran, never what was requested.
    pub fn quantize_with_options_concrete_search(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
        search: TrellisSearch,
    ) -> Result<Self> {
        // D4 fixture door: greedy is reachable only from this crate's own
        // `cfg(test)` builds. Production callers arrive via
        // `quantize_with_mode`, which refuses greedy in all builds.
        mode.deny_greedy_outside_tests("Qtip2bLayer::quantize_with_options_concrete")?;
        // GPU fast path. Same hard rule as the LUT rung: when CUDA is
        // compiled in AND the tensor targets CUDA, there is NO CPU fallback —
        // quantize stays on the device the model lives on.
        #[cfg(feature = "cuda")]
        if matches!(device, Device::Cuda(_)) {
            if !super::ffi::HAVE_QTIP_KERNELS {
                candle_core::bail!(
                    "qtip2b quantize: CUDA device but QTIP kernels not compiled in. \
                     Rebuild mistralrs-quant with CUDA + has_qtip_kernels."
                );
            }
            match Self::quantize_with_options_cuda(
                weight,
                bias.clone(),
                device,
                mode,
                use_rotation,
                search,
            )? {
                Some(layer) => return Ok(layer),
                None => candle_core::bail!(
                    "qtip2b quantize: GPU path returned None on a CUDA tensor. \
                     CPU fallback is disabled on CUDA — fix the preconditions \
                     (F32 dtype, contiguous layout, supported rotation block)."
                ),
            }
        }

        let weight_f32 = weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let (n, k_in) = weight_f32.dims2()?;
        super::warn_big_cpu_2d_quantize(n, k_in, "Qtip2bLayer::quantize_with_options_concrete");
        if !k_in.is_multiple_of(SYMS_PER_BYTE) {
            candle_core::bail!(
                "qtip2b quantize: in_features ({k_in}) must be divisible by {SYMS_PER_BYTE} for K=2 packing"
            );
        }
        let num_symbols_per_row = k_in; // V=1: one symbol per weight
        let packed_per_row = num_symbols_per_row / SYMS_PER_BYTE;

        let mcg_mult = QTIP2B_MCG_MULT;
        let codebook = computed_codebook(mcg_mult);
        let weight_data: Vec<f32> = weight_f32.flatten_all()?.to_vec1()?;

        let (rotation_block, rotation_signs_vec) = if use_rotation {
            let block = rotation_block_size(k_in);
            if block >= 2 {
                (block, generate_signs(QTIP_ROTATION_SEED, k_in))
            } else {
                (0usize, Vec::new())
            }
        } else {
            (0usize, Vec::new())
        };

        use rayon::prelude::*;
        let row_results: Vec<(Vec<u8>, f32)> = (0..n)
            .into_par_iter()
            .map(|row| {
                let row_slice = &weight_data[row * k_in..(row + 1) * k_in];

                let rotated_buf: Vec<f32>;
                let working_row: &[f32] = if rotation_block >= 2 {
                    let mut buf = row_slice.to_vec();
                    apply_block_rotation(&mut buf, &rotation_signs_vec, rotation_block);
                    rotated_buf = buf;
                    &rotated_buf[..]
                } else {
                    row_slice
                };

                let max_abs = working_row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let scale = if max_abs == 0.0 {
                    1.0
                } else {
                    max_abs / QTIP2B_SCALE_DIVISOR
                };
                let inv_scale = 1.0 / scale;

                let scaled_target: Vec<f32> = working_row.iter().map(|w| w * inv_scale).collect();

                let symbols: Vec<u8> = match mode {
                    QtipMode::Viterbi => quantize_row_2b(&scaled_target, &codebook, search),
                    QtipMode::Greedy => greedy_quantize_row_2b(&scaled_target, &codebook),
                };

                // Least-squares scale refinement: for the FIXED symbol stream
                // just chosen, the MSE-optimal row scale is
                //   scale* = ⟨w, c⟩ / ⟨c, c⟩
                // where c is the decoded (unscaled) codeword sequence. This is
                // a strict MSE improvement over the max/3.62 search heuristic
                // (which only needs to put the targets in the codebook's
                // dynamic range for the Viterbi search itself). Deterministic:
                // fixed f64 accumulation order.
                let scale = {
                    let mut dot_wc = 0f64;
                    let mut dot_cc = 0f64;
                    let mut state: u32 = 0;
                    for (t, &sym) in symbols.iter().enumerate() {
                        state = ((state << K2B) | sym as u32) & STATE_MASK_2B;
                        let c = codebook[state as usize] as f64;
                        dot_wc += working_row[t] as f64 * c;
                        dot_cc += c * c;
                    }
                    let refined = if dot_cc > 0.0 { dot_wc / dot_cc } else { 0.0 };
                    // Guard: a non-positive refined scale (pathological row)
                    // would flip every sign; keep the search heuristic then.
                    if refined > 0.0 {
                        refined as f32
                    } else {
                        scale
                    }
                };

                (pack_symbols_2b(&symbols), scale)
            })
            .collect();

        let mut all_packed = Vec::with_capacity(n * packed_per_row);
        let mut all_scales = Vec::with_capacity(n);
        for (p, s) in row_results {
            all_packed.extend_from_slice(&p);
            all_scales.push(s);
        }

        let blocks = Tensor::from_vec(all_packed, (n, packed_per_row), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let row_scales = Tensor::from_vec(all_scales, (n,), &Device::Cpu)?.to_device(device)?;
        let bias = bias.map(|b| b.to_device(device)).transpose()?;
        let rotation_signs = if rotation_block >= 2 {
            Some(Tensor::from_vec(rotation_signs_vec, (k_in,), &Device::Cpu)?.to_device(device)?)
        } else {
            None
        };

        Ok(Self {
            blocks,
            row_scales,
            bias,
            in_features: k_in,
            num_experts: None,
            rotation_signs,
            rotation_block,
            mcg_mult,
            expert_bpw: None,
            search: QtipSearchStamp::for_mode(mode),
            // The plan that ran, not the env request. `hessian: false` is
            // earned rather than assumed: this rung has no weighted branch
            // metric at all.
            search_detail: QtipSearchDetail::for_bake(mode, search, false),
        })
    }

    /// GPU fast path for `quantize_with_options_concrete`. Mirrors the LUT
    /// rung's `quantize_with_options_cuda`: rotate on-device, then run the
    /// prefix-grouped Viterbi (or greedy) kernel with the codebook computed
    /// in-register. Returns `Ok(None)` when a precondition means the caller
    /// should surface an error (no CPU fallback on CUDA).
    #[cfg(feature = "cuda")]
    fn quantize_with_options_cuda(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
        search: TrellisSearch,
    ) -> Result<Option<Self>> {
        let (_n, k_in) = match weight.dims2() {
            Ok((n, k)) => (n, k),
            Err(_) => return Ok(None),
        };
        if !k_in.is_multiple_of(SYMS_PER_BYTE) {
            return Ok(None);
        }

        // Device-first transfer, then cast (same rationale as the LUT rung:
        // bf16 over PCIe is half the bytes; widening on-GPU is exact).
        let weight_cuda_f32 = weight.to_device(device)?.to_dtype(DType::F32)?;

        let (rotation_block, rotation_signs_vec) = if use_rotation {
            let block = rotation_block_size(k_in);
            if block >= 2 && matches!(block, 2 | 4 | 8 | 16 | 32 | 64 | 128) {
                (block, generate_signs(QTIP_ROTATION_SEED, k_in))
            } else {
                return Ok(None);
            }
        } else {
            (0usize, Vec::new())
        };

        let weight_rotated = if rotation_block >= 2 {
            let signs_cuda = Tensor::from_vec(rotation_signs_vec.clone(), (k_in,), &Device::Cpu)?
                .to_device(device)?;
            cuda_ops::rotate_weight_rows_cuda(&weight_cuda_f32, &signs_cuda, rotation_block)?
        } else {
            weight_cuda_f32
        };

        // Never substitute a search the kernel cannot run (D4b): a width the
        // beam kernel refuses is an error, not a quietly narrower beam. A width
        // at or above the state space prunes nothing and IS the exhaustive DP.
        let search = super::cuda_search_plan(search, cuda_ops::beam_2b_max_width())?;

        let (blocks, row_scales) =
            cuda_ops::quantize_rows_2b_cuda(&weight_rotated, QTIP2B_MCG_MULT, mode, search)?;

        let bias = bias.map(|b| b.to_device(device)).transpose()?;
        let rotation_signs = if rotation_block >= 2 {
            Some(Tensor::from_vec(rotation_signs_vec, (k_in,), &Device::Cpu)?.to_device(device)?)
        } else {
            None
        };

        Ok(Some(Self {
            blocks,
            row_scales,
            bias,
            in_features: k_in,
            num_experts: None,
            rotation_signs,
            rotation_block,
            mcg_mult: QTIP2B_MCG_MULT,
            expert_bpw: None,
            search: QtipSearchStamp::for_mode(mode),
            // The plan that ran, not the env request. `hessian: false` is
            // earned rather than assumed: this rung has no weighted branch
            // metric at all.
            search_detail: QtipSearchDetail::for_bake(mode, search, false),
        }))
    }

    /// Quantize a 3-D `[E, N, K]` stacked-expert weight tensor. Experts are
    /// processed in batches by flattening `[B, N, K] → [B·N, K]` and reusing
    /// the 2-D pipeline (per-row rotation/scale/search is identical), then
    /// reshaping the packed result back — the same batching strategy as the
    /// LUT rung (RUN-161), including the `ARC_QTIP_EXPERT_BATCH` override.
    pub fn quantize_with_options_3d(
        weight: &Tensor,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
    ) -> Result<Arc<dyn QuantMethod>> {
        Self::quantize_with_options_3d_search(
            weight,
            device,
            mode,
            use_rotation,
            Self::env_search(),
        )
    }

    /// [`Self::quantize_with_options_3d`] with an explicit trellis search.
    pub fn quantize_with_options_3d_search(
        weight: &Tensor,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
        search: TrellisSearch,
    ) -> Result<Arc<dyn QuantMethod>> {
        let (e, n, k_in) = weight.dims3()?;
        if e == 0 || n == 0 || k_in == 0 {
            candle_core::bail!("qtip2b 3-D quantize: zero-sized expert stack ({e}, {n}, {k_in})");
        }
        if !k_in.is_multiple_of(SYMS_PER_BYTE) {
            candle_core::bail!(
                "qtip2b 3-D quantize: in_features ({k_in}) must be divisible by {SYMS_PER_BYTE}"
            );
        }
        let packed_per_row = k_in / SYMS_PER_BYTE;

        // Quantize on GPU when the kernels are available and the stack lives
        // on CPU (the ISQ load path), streaming batches of experts; results
        // come back to `device`. Mirrors the LUT rung's RUN-161 batching.
        // Any reroute to the CPU pipeline while a GPU is plausibly available
        // is counted + warned inside `expert_stack_quant_device` (wave6-Q:
        // the silent version of this gate cost ~20x per layer).
        let quant_device =
            super::expert_stack_quant_device(device, "Qtip2bLayer::quantize_with_options_3d");
        // During a UQFF bake the quantized stack is only serialized, never used
        // for a forward pass, so it is materialized on the host and device usage
        // stays flat across layers instead of growing by the artifact size. See
        // `crate::set_bake_isq_to_host` and the LUT rung's sibling comment.
        // (wave18)
        let out_device = if crate::bake_isq_to_host() {
            Device::Cpu
        } else {
            device.clone()
        };
        let move_back = !quant_device.same_device(&out_device);

        let batch = std::env::var("ARC_QTIP_EXPERT_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16)
            .clamp(1, e);

        let mut blocks_slices: Vec<Tensor> = Vec::with_capacity(e.div_ceil(batch));
        let mut scales_slices: Vec<Tensor> = Vec::with_capacity(e.div_ceil(batch));
        let mut shared_rotation_signs: Option<Tensor> = None;
        let mut shared_rotation_block: usize = 0;
        let mut mcg_mult = QTIP2B_MCG_MULT;

        let mut expert_idx = 0usize;
        while expert_idx < e {
            let this_b = batch.min(e - expert_idx);
            let chunk = weight.narrow(0, expert_idx, this_b)?;
            let rows_2d = chunk.reshape((this_b * n, k_in))?;
            // A narrowed view still carries the WHOLE [E, N, K] storage, and
            // `to_device` copies storage rather than layout — so without this
            // the CUDA path uploads all E experts per chunk. See the LUT rung
            // for the measured cost. (wave18)
            let rows_2d = if this_b < e {
                rows_2d.force_contiguous()?
            } else {
                rows_2d
            };
            // Every chunk is baked with the SAME search the caller handed in,
            // so an expert stack can never be half beam and half exhaustive.
            let layer = Self::quantize_with_options_concrete_search(
                &rows_2d,
                None,
                &quant_device,
                mode,
                use_rotation,
                search,
            )?;

            let blk = if move_back {
                layer.blocks.to_device(&out_device)?
            } else {
                layer.blocks.clone()
            };
            let scl = if move_back {
                layer.row_scales.to_device(&out_device)?
            } else {
                layer.row_scales.clone()
            };
            blocks_slices.push(blk.reshape((this_b, n, packed_per_row))?);
            scales_slices.push(scl.reshape((this_b, n))?);

            if expert_idx == 0 {
                shared_rotation_signs = match layer.rotation_signs.clone() {
                    Some(s) if move_back => Some(s.to_device(&out_device)?),
                    other => other,
                };
                shared_rotation_block = layer.rotation_block;
                mcg_mult = layer.mcg_mult;
            } else {
                debug_assert_eq!(layer.rotation_block, shared_rotation_block);
                debug_assert_eq!(layer.mcg_mult, mcg_mult);
            }
            expert_idx += this_b;
        }

        let blocks_3d = Tensor::cat(&blocks_slices, 0)?;
        let row_scales_2d = Tensor::cat(&scales_slices, 0)?;
        debug_assert_eq!(blocks_3d.dims(), &[e, n, packed_per_row]);
        debug_assert_eq!(row_scales_2d.dims(), &[e, n]);

        Ok(Arc::new(Self {
            blocks: blocks_3d,
            row_scales: row_scales_2d,
            bias: None,
            in_features: k_in,
            num_experts: Some(e),
            rotation_signs: shared_rotation_signs,
            rotation_block: shared_rotation_block,
            mcg_mult,
            expert_bpw: Some(ExpertBpwTable::uniform_2bit(e)),
            search: QtipSearchStamp::for_mode(mode),
            // The plan that ran, not the env request. `hessian: false` is
            // earned rather than assumed: this rung has no weighted branch
            // metric at all.
            search_detail: QtipSearchDetail::for_bake(mode, search, false),
        }))
    }

    // -----------------------------------------------------------------------
    // Dequantize
    // -----------------------------------------------------------------------

    /// Decoded weights *in the rotated frame* (== original frame when
    /// rotation is off). Internal helper for the fused forward path.
    fn dequantize_weights_rotated_f32(&self) -> Result<Vec<f32>> {
        let blocks_cpu = self.blocks.to_device(&Device::Cpu)?;
        let scales_cpu = self.row_scales.to_device(&Device::Cpu)?;
        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<f32> = scales_cpu.flatten_all()?.to_vec1()?;

        let n = self.row_scales.dim(0)?;
        let k_in = self.in_features;
        let packed_per_row = k_in / SYMS_PER_BYTE;

        let mut out = vec![0f32; n * k_in];
        for row in 0..n {
            decode_row_2b(
                &blocks_data[row * packed_per_row..(row + 1) * packed_per_row],
                k_in,
                self.mcg_mult,
                scales_data[row],
                &mut out[row * k_in..(row + 1) * k_in],
            );
        }
        Ok(out)
    }

    fn dequantize_weights_f32(&self) -> Result<Vec<f32>> {
        let mut out = self.dequantize_weights_rotated_f32()?;
        if self.rotation_block >= 2 {
            let signs = self.rotation_signs_cpu()?;
            let n = self.row_scales.dim(0)?;
            let k_in = self.in_features;
            for row in 0..n {
                apply_block_rotation(
                    &mut out[row * k_in..(row + 1) * k_in],
                    &signs,
                    self.rotation_block,
                );
            }
        }
        Ok(out)
    }

    fn rotation_signs_cpu(&self) -> Result<Vec<f32>> {
        match &self.rotation_signs {
            Some(t) => t.to_device(&Device::Cpu)?.to_vec1::<f32>(),
            None => candle_core::bail!(
                "Qtip2bLayer: rotation_block={} but rotation_signs is None",
                self.rotation_block
            ),
        }
    }

    /// Dequantize into a `[N, K_in]` (2-D) or `[E, N, K_in]` (3-D) BF16
    /// tensor in the original (unrotated) frame.
    pub fn dequantize_weights(&self) -> Result<Tensor> {
        if let Some(e) = self.num_experts {
            let n = self.row_scales.dim(1)?;
            let k_in = self.in_features;
            let mut expert_tensors: Vec<Tensor> = Vec::with_capacity(e);
            for expert_idx in 0..e {
                let blocks_e = self
                    .blocks
                    .narrow(0, expert_idx, 1)?
                    .squeeze(0)?
                    .to_device(&Device::Cpu)?
                    .contiguous()?;
                let scales_e = self
                    .row_scales
                    .narrow(0, expert_idx, 1)?
                    .squeeze(0)?
                    .to_device(&Device::Cpu)?
                    .contiguous()?;
                let expert_w = self.dequantize_expert_cpu(&blocks_e, &scales_e, n, k_in)?;
                expert_tensors.push(expert_w);
            }
            return Tensor::stack(&expert_tensors, 0);
        }

        #[cfg(feature = "cuda")]
        {
            if cuda_ops::can_use_qtip_cuda(&self.blocks) {
                let w_rotated = cuda_ops::dequantize_2b_cuda(
                    &self.blocks,
                    &self.row_scales,
                    self.mcg_mult,
                    self.in_features,
                    DType::BF16,
                )?;
                if self.rotation_block >= 2 {
                    let signs = match &self.rotation_signs {
                        Some(t) => t,
                        None => candle_core::bail!(
                            "Qtip2bLayer: rotation_block={} but rotation_signs is None",
                            self.rotation_block
                        ),
                    };
                    return cuda_ops::rotate_x_cuda(&w_rotated, signs, self.rotation_block);
                }
                return Ok(w_rotated);
            }
        }

        let out = self.dequantize_weights_f32()?;
        let n = self.row_scales.dim(0)?;
        let k_in = self.in_features;
        Tensor::from_vec(out, (n, k_in), &Device::Cpu)?
            .to_device(self.blocks.device())?
            .to_dtype(DType::BF16)
    }

    /// Dequantize one expert's `[N, K_in]` slice on CPU (unrotated frame).
    fn dequantize_expert_cpu(
        &self,
        blocks_e: &Tensor,
        scales_e: &Tensor,
        _n: usize,
        k_in: usize,
    ) -> Result<Tensor> {
        let blocks_data: Vec<u8> = blocks_e.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
        let scales_data: Vec<f32> = scales_e.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;

        let n_local = scales_data.len();
        let packed_per_row = k_in / SYMS_PER_BYTE;
        let mut out = vec![0f32; n_local * k_in];
        for row in 0..n_local {
            decode_row_2b(
                &blocks_data[row * packed_per_row..(row + 1) * packed_per_row],
                k_in,
                self.mcg_mult,
                scales_data[row],
                &mut out[row * k_in..(row + 1) * k_in],
            );
        }
        if self.rotation_block >= 2 {
            let signs = self.rotation_signs_cpu()?;
            for row in 0..n_local {
                apply_block_rotation(
                    &mut out[row * k_in..(row + 1) * k_in],
                    &signs,
                    self.rotation_block,
                );
            }
        }
        Tensor::from_vec(out, (n_local, k_in), &Device::Cpu)?
            .to_device(blocks_e.device())?
            .to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    fn forward_dequantize(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_experts.is_some() {
            candle_core::bail!(
                "Qtip2bLayer::forward called on a 3-D stacked-expert layer; \
                 use gather_forward(x, indices) instead"
            );
        }

        let orig_dims = x.dims().to_vec();
        let x_2d = if orig_dims.len() > 2 {
            let features = orig_dims[orig_dims.len() - 1];
            let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
            x.reshape((batch_size, features))?
        } else {
            x.clone()
        };

        let n = self.row_scales.dim(0)?;
        let k_in = self.in_features;

        #[cfg(feature = "cuda")]
        {
            if cuda_ops::can_use_qtip_cuda(&self.blocks) {
                if let Ok(mut result) = self.forward_dequantize_cuda(&x_2d, x.dtype(), k_in) {
                    if let Some(bias) = &self.bias {
                        result = result.broadcast_add(bias)?;
                    }
                    if orig_dims.len() > 2 {
                        let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
                        new_dims.push(result.dim(1)?);
                        result = result.reshape(new_dims)?;
                    }
                    return Ok(result);
                }
            }
        }

        // CPU path — rotated-frame matmul identity `(xR)·(WR)^T = x·W^T`.
        // This materializes the full dequantized weight; at decode shapes the
        // fused GEMV above should have handled this.
        super::warn_dequant_materialize_at_decode(
            x_2d.dim(0)?,
            "Qtip2bLayer::forward_dequantize fallback (full dequantize+matmul)",
        );
        let (x_for_matmul, w_for_matmul) = if self.rotation_block >= 2 {
            let signs = self.rotation_signs_cpu()?;
            let x_cpu = x_2d.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
            let (batch, feat) = x_cpu.dims2()?;
            debug_assert_eq!(feat, k_in);
            let mut x_data: Vec<f32> = x_cpu.flatten_all()?.to_vec1()?;
            for row in 0..batch {
                apply_block_rotation(
                    &mut x_data[row * feat..(row + 1) * feat],
                    &signs,
                    self.rotation_block,
                );
            }
            let x_rot = Tensor::from_vec(x_data, (batch, feat), &Device::Cpu)?
                .to_device(x.device())?
                .to_dtype(x.dtype())?;
            let w_rotated = self.dequantize_weights_rotated_f32()?;
            let w_tensor = Tensor::from_vec(w_rotated, (n, k_in), &Device::Cpu)?
                .to_device(x.device())?
                .to_dtype(x.dtype())?;
            (x_rot, w_tensor)
        } else {
            let w = self.dequantize_weights()?.to_dtype(x.dtype())?;
            (x_2d, w)
        };

        let mut result = x_for_matmul.matmul(&w_for_matmul.t()?)?;
        if let Some(bias) = &self.bias {
            result = result.broadcast_add(bias)?;
        }
        if orig_dims.len() > 2 {
            let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
            new_dims.push(result.dim(1)?);
            result = result.reshape(new_dims)?;
        }
        Ok(result)
    }

    /// GPU forward. Single-token decode uses the fused in-register-decode
    /// GEMV (the whole point of this format); multi-token uses
    /// dequantize+matmul (GEMM amortizes decode across tokens).
    #[cfg(feature = "cuda")]
    fn forward_dequantize_cuda(
        &self,
        x_2d: &Tensor,
        out_dtype: DType,
        k_in: usize,
    ) -> Result<Tensor> {
        if !matches!(out_dtype, DType::BF16 | DType::F16 | DType::F32) {
            candle_core::bail!("qtip2b forward CUDA: unsupported out dtype {out_dtype:?}");
        }
        if !matches!(x_2d.device(), candle_core::Device::Cuda(_)) {
            candle_core::bail!("qtip2b forward CUDA: x must live on CUDA");
        }
        if x_2d.dim(1)? != k_in {
            candle_core::bail!(
                "qtip2b forward CUDA: x.dim(1)={} != in_features={}",
                x_2d.dim(1)?,
                k_in
            );
        }

        let x_rot_dtype = if matches!(x_2d.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            x_2d.dtype()
        } else {
            DType::BF16
        };
        let x_for_rot = if x_2d.dtype() == x_rot_dtype {
            x_2d.contiguous()?
        } else {
            x_2d.to_dtype(x_rot_dtype)?.contiguous()?
        };

        let x_rotated = if self.rotation_block >= 2 {
            let signs = match &self.rotation_signs {
                Some(t) => t,
                None => candle_core::bail!(
                    "Qtip2bLayer: rotation_block={} but rotation_signs is None",
                    self.rotation_block
                ),
            };
            cuda_ops::rotate_x_cuda(&x_for_rot, signs, self.rotation_block)?
        } else {
            x_for_rot
        };

        let n_tokens = x_rotated.dim(0)?;
        // `spec_pin_gemm()` (default off) sends the 1-token case down the
        // multi-token path so a speculative draft and its verify read the same
        // weights through the same kernel — see `tune::spec_pin_gemm`.
        if n_tokens == 1 && !super::tune::spec_pin_gemm() {
            let y = cuda_ops::fused_gemv_2b_cuda(
                &self.blocks,
                &self.row_scales,
                self.mcg_mult,
                &x_rotated,
                k_in,
            )?;
            if y.dtype() != out_dtype {
                return y.to_dtype(out_dtype);
            }
            return Ok(y);
        }

        let w_dtype = x_rotated.dtype();
        let w_rotated = cuda_ops::dequantize_2b_cuda(
            &self.blocks,
            &self.row_scales,
            self.mcg_mult,
            k_in,
            w_dtype,
        )?;
        let y = x_rotated.matmul(&w_rotated.t()?)?;
        if y.dtype() != out_dtype {
            return y.to_dtype(out_dtype);
        }
        Ok(y)
    }

    // -----------------------------------------------------------------------
    // MoE gather forward
    // -----------------------------------------------------------------------

    fn num_experts_count(&self) -> usize {
        self.num_experts.unwrap_or(1)
    }

    fn rows_per_expert(&self) -> Result<usize> {
        match self.num_experts {
            None => self.row_scales.dim(0),
            Some(_) => self.row_scales.dim(1),
        }
    }

    /// Dequantize a single expert's weights into the original frame (CPU
    /// reference used by `gather_forward_cpu`).
    fn dequantize_expert_weights_unrotated(&self, e: usize) -> Result<Tensor> {
        let blocks_e = match self.num_experts {
            None => self.blocks.clone(),
            Some(_) => self.blocks.narrow(0, e, 1)?.squeeze(0)?.contiguous()?,
        };
        let scales_e = match self.num_experts {
            None => self.row_scales.clone(),
            Some(_) => self.row_scales.narrow(0, e, 1)?.squeeze(0)?.contiguous()?,
        };
        let n = scales_e.dim(0)?;
        self.dequantize_expert_cpu(&blocks_e, &scales_e, n, self.in_features)
    }

    /// CPU grouped reference over flattened (token, slot) pairs: dequantize
    /// each **unique** routed expert once (original frame), batch the pairs
    /// per expert, dense matmul per group, scatter results back. This is
    /// exactly the computation the CUDA grouped GEMM performs (loop experts
    /// → dense matmul on dequantized weights) and is the correctness
    /// reference its parity tests compare against.
    ///
    /// `a2d` is `[total_pairs, in_features]`, `indices` is `[total_pairs]`.
    /// Returns `[total_pairs, out_features]` (bias-free — callers attach
    /// the 2-D bias where the shape supports one).
    fn gather_forward_batched_cpu(&self, a2d: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (total_pairs, cols) = a2d.dims2()?;
        debug_assert_eq!(cols, self.in_features);
        let rows = self.rows_per_expert()?;
        let num_experts = self.num_experts_count();

        let idx_cpu: Vec<u32> = indices
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1()?;
        if idx_cpu.len() != total_pairs {
            candle_core::bail!(
                "Qtip2bLayer::gather_forward_batched: indices len {} != total_pairs {total_pairs}",
                idx_cpu.len()
            );
        }

        let mut unique_ids: Vec<usize> = idx_cpu.iter().map(|&v| v as usize).collect();
        unique_ids.sort_unstable();
        unique_ids.dedup();
        for &e in &unique_ids {
            if e >= num_experts {
                candle_core::bail!(
                    "Qtip2bLayer::gather_forward: expert index {e} >= num_experts {num_experts}"
                );
            }
        }

        let mut weight_cache: std::collections::HashMap<usize, Tensor> =
            std::collections::HashMap::with_capacity(unique_ids.len());
        for &e in &unique_ids {
            let w_e = self
                .dequantize_expert_weights_unrotated(e)?
                .to_dtype(a2d.dtype())?;
            weight_cache.insert(e, w_e);
        }

        let mut positions_by_expert: std::collections::HashMap<usize, Vec<u32>> =
            std::collections::HashMap::with_capacity(unique_ids.len());
        for (flat, &e_u32) in idx_cpu.iter().enumerate() {
            positions_by_expert
                .entry(e_u32 as usize)
                .or_default()
                .push(flat as u32);
        }

        let device = a2d.device();
        let mut out_flat = Tensor::zeros((total_pairs, rows), a2d.dtype(), device)?;

        for &e in &unique_ids {
            let positions = positions_by_expert
                .get(&e)
                .expect("positions for expert should be populated");
            let pos_tensor = Tensor::from_vec(positions.clone(), (positions.len(),), device)?;
            let a_e = a2d.index_select(&pos_tensor, 0)?;
            let w_e = weight_cache.get(&e).expect("weight should be cached");
            let y_e = a_e.matmul(&w_e.t()?)?;
            out_flat = out_flat.index_add(&pos_tensor, &y_e, 0)?;
        }

        Ok(out_flat)
    }

    /// CPU sparse-gather forward (portable reference). Mirrors the LUT
    /// rung's `gather_forward_cpu`; the grouped math lives in
    /// [`Self::gather_forward_batched_cpu`] on the flattened pair view.
    fn gather_forward_cpu(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (n_tokens, n_experts_per_tok, cols) = a.dims3()?;
        let (i_tokens, i_k) = indices.dims2()?;
        if i_tokens != n_tokens || i_k != n_experts_per_tok {
            candle_core::bail!(
                "Qtip2bLayer::gather_forward: indices shape ({i_tokens}, {i_k}) doesn't match a shape ({n_tokens}, {n_experts_per_tok}, {cols})"
            );
        }
        if cols != self.in_features {
            candle_core::bail!(
                "Qtip2bLayer::gather_forward: a.dim(-1)={cols} != in_features={}",
                self.in_features
            );
        }
        let rows = self.rows_per_expert()?;

        let total_pairs = n_tokens * n_experts_per_tok;
        let out_flat = self.gather_forward_batched_cpu(
            &a.reshape((total_pairs, cols))?,
            &indices.reshape((total_pairs,))?,
        )?;

        let mut out = out_flat.reshape((n_tokens, n_experts_per_tok, rows))?;
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(&bias.to_dtype(out.dtype())?)?;
        }
        Ok(out)
    }

    /// Batched MoE forward over flattened (token, slot) pairs — the prefill
    /// regime's entry into the **trellis grouped GEMM** (Arc Stage 4,
    /// `kernels/qtip/qtip_grouped_gemm.cu`).
    ///
    /// * `a2d`: `[total_pairs, in_features]` activations (original frame).
    /// * `indices`: `[total_pairs]` router expert ids.
    ///
    /// Returns `[total_pairs, out_features]`.
    ///
    /// On CUDA with 16-bit activations and `in_features % 64 == 0`, pairs
    /// are sorted by expert on-device (histogram → scans/tile-map → grouped
    /// scatter — ZERO host syncs) and a persistent tensor-core kernel walks
    /// the ragged tile list, decoding the 2-bit trellis in registers into
    /// mma.sync fragments. Everywhere else this falls back to the CPU
    /// grouped reference (dequantize each routed expert once, dense matmul
    /// per group).
    pub fn gather_forward_batched(&self, a2d: &Tensor, indices: &Tensor) -> Result<Tensor> {
        if self.num_experts.is_none() {
            candle_core::bail!(
                "Qtip2bLayer::gather_forward_batched requires an expert-stacked 3-D layer"
            );
        }
        let (total_pairs, cols) = a2d.dims2()?;
        if cols != self.in_features {
            candle_core::bail!(
                "Qtip2bLayer::gather_forward_batched: a2d.dim(1)={cols} != in_features={}",
                self.in_features
            );
        }
        if indices.elem_count() != total_pairs {
            candle_core::bail!(
                "Qtip2bLayer::gather_forward_batched: indices len {} != total_pairs {total_pairs}",
                indices.elem_count()
            );
        }
        // Per-expert bit-width dispatch: the 2-bit class is implemented
        // end-to-end; a mixed table partitions its expert groups by class
        // and launches per class once the 4-bit rung lands.
        if let Some(table) = &self.expert_bpw {
            if !table.is_uniform_2bit() {
                candle_core::bail!(
                    "Qtip2bLayer::gather_forward_batched: mixed-bpw expert tables need the \
                     per-class grouped dispatch (4-bit rung); only uniform 2-bit ships today"
                );
            }
        }

        #[cfg(feature = "cuda")]
        {
            if cuda_ops::can_use_qtip_cuda(&self.blocks)
                && matches!(a2d.device(), candle_core::Device::Cuda(_))
                && matches!(a2d.dtype(), DType::BF16 | DType::F16)
                && self
                    .in_features
                    .is_multiple_of(super::grouped::GROUPED_TILE_K)
            {
                let a_contig = a2d.contiguous()?;
                let a_rotated = if self.rotation_block >= 2 {
                    let signs = match &self.rotation_signs {
                        Some(t) => t,
                        None => candle_core::bail!(
                            "Qtip2bLayer: rotation_block={} but rotation_signs is None",
                            self.rotation_block
                        ),
                    };
                    cuda_ops::rotate_x_cuda(&a_contig, signs, self.rotation_block)?
                } else {
                    a_contig
                };
                let idx = indices
                    .reshape((total_pairs,))?
                    .to_dtype(DType::U32)?
                    .contiguous()?;
                return cuda_ops::grouped_gemm_2b_cuda(
                    &self.blocks,
                    &self.row_scales,
                    self.mcg_mult,
                    &a_rotated,
                    &idx,
                    self.in_features,
                );
            }
        }

        self.gather_forward_batched_cpu(a2d, indices)
    }

    /// On-device (sync-free) MoE gather + fused decode GEMV for the decode
    /// regime — the qtip2b sibling of the LUT rung's
    /// `gather_forward_cuda_ondevice`. Reads each pair's expert id on the
    /// GPU so the dispatch is CUDA-graph capturable.
    #[cfg(feature = "cuda")]
    fn gather_forward_cuda_ondevice(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (n_tokens, n_experts_per_tok, cols) = a.dims3()?;
        let rows = self.rows_per_expert()?;

        let a_2d = a
            .reshape((n_tokens * n_experts_per_tok, cols))?
            .contiguous()?;
        let a_rotated = if self.rotation_block >= 2 {
            let signs = match &self.rotation_signs {
                Some(t) => t,
                None => candle_core::bail!(
                    "Qtip2bLayer: rotation_block={} but rotation_signs is None",
                    self.rotation_block
                ),
            };
            cuda_ops::rotate_x_cuda(&a_2d, signs, self.rotation_block)?
        } else {
            a_2d
        };

        let idx_flat = indices
            .reshape((n_tokens * n_experts_per_tok,))?
            .to_dtype(DType::U32)?
            .contiguous()?;

        let y = cuda_ops::gather_gemv_2b_cuda(
            &self.blocks,
            &self.row_scales,
            self.mcg_mult,
            &a_rotated,
            &idx_flat,
            self.in_features,
        )?;

        let mut out = y.reshape((n_tokens, n_experts_per_tok, rows))?;
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(&bias.to_dtype(out.dtype())?)?;
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Accessors (typed test / integration surface)
    // -----------------------------------------------------------------------

    /// `Some(E)` for a stacked-expert 3-D layer, `None` for 2-D.
    pub fn num_experts(&self) -> Option<usize> {
        self.num_experts
    }

    /// Input feature dim.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output rows (per expert in 3-D mode).
    pub fn out_features(&self) -> Result<usize> {
        self.rows_per_expert()
    }

    /// Packed symbol blocks.
    pub fn blocks(&self) -> &Tensor {
        &self.blocks
    }

    /// Per-row scales.
    pub fn row_scales(&self) -> &Tensor {
        &self.row_scales
    }

    /// The MCG multiplier defining the computed codebook.
    pub fn mcg_mult(&self) -> u32 {
        self.mcg_mult
    }

    /// Rotation signs, when rotation is enabled.
    pub fn rotation_signs(&self) -> Option<&Tensor> {
        self.rotation_signs.as_ref()
    }

    /// Hadamard rotation block size (0 = disabled).
    pub fn rotation_block(&self) -> usize {
        self.rotation_block
    }

    /// Which trellis search produced these blocks (DOCTRINE D4 §3).
    pub fn search_stamp(&self) -> QtipSearchStamp {
        self.search
    }

    /// *Which* trellis search: beam width and objective (UQFF ≥ 0.3.0).
    pub fn search_detail(&self) -> QtipSearchDetail {
        self.search_detail
    }

    /// Per-expert bit-width descriptor table (`Some` iff this is a 3-D
    /// expert stack). Consulted by the grouped-GEMM dispatch.
    pub fn expert_bpw(&self) -> Option<&ExpertBpwTable> {
        self.expert_bpw.as_ref()
    }
}

impl QuantMethod for Qtip2bLayer {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Qtip2b {
                blocks,
                row_scales,
                bias,
                in_features,
                mcg_mult,
                rotation_signs,
                rotation_block,
            } => {
                let num_experts = if blocks.rank() == 3 {
                    Some(blocks.dim(0)?)
                } else {
                    None
                };
                Ok(Self {
                    blocks,
                    row_scales,
                    bias,
                    in_features,
                    num_experts,
                    rotation_signs,
                    rotation_block,
                    mcg_mult,
                    expert_bpw: num_experts.map(ExpertBpwTable::uniform_2bit),
                    // Blocks arrive already packed from an unknown producer;
                    // we did not run the search, so we do not claim it (D4).
                    search: QtipSearchStamp::Unstamped,
                    search_detail: QtipSearchDetail::Unknown,
                })
            }
            _ => candle_core::bail!("Qtip2bLayer requires QuantMethodConfig::Qtip2b"),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.dequantize_weights()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_dequantize(x)
    }

    fn gather_forward(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        if self.num_experts.is_none() {
            candle_core::bail!(
                "Qtip2bLayer::gather_forward requires an expert-stacked 3-D layer (got a 2-D layer with no expert dim)"
            );
        }

        let (n_tokens, n_experts_per_tok, cols) = a.dims3()?;
        let (i_tokens, i_k) = indices.dims2()?;
        if i_tokens != n_tokens || i_k != n_experts_per_tok {
            candle_core::bail!(
                "Qtip2bLayer::gather_forward: indices shape ({i_tokens}, {i_k}) doesn't match a shape ({n_tokens}, {n_experts_per_tok}, {cols})"
            );
        }
        if cols != self.in_features {
            candle_core::bail!(
                "Qtip2bLayer::gather_forward: a.dim(-1)={cols} != in_features={}",
                self.in_features
            );
        }

        #[cfg(feature = "cuda")]
        {
            if cuda_ops::can_use_qtip_cuda(&self.blocks)
                && matches!(a.device(), candle_core::Device::Cuda(_))
                && matches!(a.dtype(), DType::BF16 | DType::F16 | DType::F32)
            {
                // On-device only, propagate errors (the CPU fallback would
                // D2H-sync under graph capture and read garbage indices).
                //
                // This rung KEEPS the decode-shaped boundary while the LUT rung
                // derives a much larger one, and the asymmetry is deliberate:
                // above the boundary this rung goes to the trellis grouped GEMM
                // below — tokens sorted by expert on-device, each expert's bytes
                // read once per tile — which is precisely the amortizing kernel
                // the fleet math wants. Raising this cap would replace a kernel
                // whose cost tracks the number of DISTINCT experts with one
                // whose cost is linear in (token, slot) pairs. The LUT rung has
                // no grouped kernel, so its over-boundary path is a
                // dequantize-materialize loop and the same cap was pure loss
                // there. See `super::gather_policy`.
                let ondevice_max_tokens = super::gather_policy::ondevice_max_tokens_override()
                    .unwrap_or(super::DECODE_REGIME_MAX_TOKENS);
                let ondevice_disabled = std::env::var("ARC_NO_QTIP_ONDEVICE_MOE").is_ok();
                if !ondevice_disabled
                    && n_tokens <= ondevice_max_tokens
                    && n_tokens.saturating_mul(n_experts_per_tok)
                        <= super::gather_policy::GATHER_GEMV_MAX_PAIRS
                {
                    return self.gather_forward_cuda_ondevice(a, indices);
                }

                // Prefill regime: the trellis grouped GEMM (Arc Stage 4).
                // Tokens sorted by expert on-device, persistent tensor-core
                // tile loop over the ragged groups — this is where batched
                // 2-bit MoE serving actually lives. 16-bit activations only
                // (the mma.sync pipeline is the point); other dtypes keep
                // the CPU reference below.
                let grouped_disabled = std::env::var("ARC_NO_QTIP_GROUPED_MOE").is_ok();
                if !grouped_disabled
                    && matches!(a.dtype(), DType::BF16 | DType::F16)
                    && self
                        .in_features
                        .is_multiple_of(super::grouped::GROUPED_TILE_K)
                    && self.expert_bpw.as_ref().is_none_or(|t| t.is_uniform_2bit())
                {
                    let total_pairs = n_tokens * n_experts_per_tok;
                    let out_flat = self.gather_forward_batched(
                        &a.reshape((total_pairs, cols))?,
                        &indices.reshape((total_pairs,))?,
                    )?;
                    let mut out =
                        out_flat.reshape((n_tokens, n_experts_per_tok, self.rows_per_expert()?))?;
                    if let Some(bias) = &self.bias {
                        out = out.broadcast_add(&bias.to_dtype(out.dtype())?)?;
                    }
                    return Ok(out);
                }
            }
        }

        super::warn_dequant_materialize_at_decode(n_tokens, "Qtip2bLayer::gather_forward_cpu");
        self.gather_forward_cpu(a, indices)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("Qtip2bLayer does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (DType::BF16, self.blocks.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("Qtip2bLayer does not support ISQ re-quantization")
    }
}

impl Qtip2bLayer {
    /// Concrete-typed UQFF deserialize that returns a `Qtip2bLayer` (plus
    /// the optional bias tensor) instead of `Arc<dyn QuantMethod>`. Shared
    /// body of `deserialize` / `deserialize_ext_bias`; also usable by tests
    /// to inspect typed fields without a `dyn`-downcast.
    ///
    /// Handles both storage modes:
    /// - 2-D: `blocks [N, packed_K]`, `row_scales [N]` → `num_experts: None`
    /// - 3-D: `blocks [E, N, packed_K]`, `row_scales [E, N]` →
    ///   `num_experts: Some(E)`
    ///
    /// The mode is inferred from the self-describing tensor shapes (UQFF
    /// tensor payloads carry rank + dims), so no extra header bytes are
    /// needed — the same mechanism as the LUT rung's `deserialize_concrete`
    /// (UQFF v0.2.1). Each tensor is deserialized in one shot (single host
    /// buffer → single device upload) — no per-expert round-trips at load.
    pub(super) fn deserialize_concrete(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Self, Option<Tensor>)> {
        let (layer, ext_bias) = Self::deserialize_concrete_unchecked(data, device, guard)?;
        // D4 §3 teeth: the load gate every serving path passes through.
        layer
            .search
            .enforce_at_load("qtip2b-layer", layer.rotation_block)?;
        Ok((layer, ext_bias))
    }

    /// Payload parser without the D4 load gate. Private; used by the checked
    /// wrapper above and by serde round-trip tests that round-trip a cheap
    /// greedy fixture. No serving path calls it directly.
    fn deserialize_concrete_unchecked(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Self, Option<Tensor>)> {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }
        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Qtip2b as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected QTIP2B type {}",
                QuantizedSerdeType::Qtip2b as usize
            );
        }
        let has_bias = buffer.read_u8()? != 0;
        let in_features = buffer.read_u32::<LittleEndian>()? as usize;
        let mcg_mult = buffer.read_u32::<LittleEndian>()?;

        let _acquired_load_guard = guard.acquire(device);
        let blocks = deserialize_tensor(&mut buffer, device)?;
        let row_scales = deserialize_tensor(&mut buffer, device)?;

        let num_experts = match blocks.dims().len() {
            2 => None,
            3 => {
                let e = blocks.dim(0)?;
                if row_scales.dims().len() != 2
                    || row_scales.dim(0)? != e
                    || row_scales.dim(1)? != blocks.dim(1)?
                {
                    candle_core::bail!(
                        "Qtip2bLayer: 3-D blocks {:?} require row_scales [E, N]; got {:?}",
                        blocks.dims(),
                        row_scales.dims()
                    );
                }
                if has_bias {
                    // The serializer refuses to attach a bias to a 3-D
                    // stack, so this indicates a corrupt payload.
                    candle_core::bail!("Qtip2bLayer: 3-D stacked-expert payloads are bias-free");
                }
                Some(e)
            }
            other => {
                candle_core::bail!(
                    "Qtip2bLayer: blocks tensor must be rank 2 or 3, got rank {other}"
                )
            }
        };

        let bias = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };
        let ext_bias = bias.clone();

        let (rotation_signs, rotation_block) = match buffer.read_u8() {
            Ok(0) => (None, 0usize),
            Ok(1) => {
                let block = buffer.read_u32::<LittleEndian>()? as usize;
                let signs = deserialize_tensor(&mut buffer, device)?;
                (Some(signs), block)
            }
            Ok(other) => candle_core::bail!(
                "Qtip2bLayer: unexpected rotation-flag byte {other} (expected 0 or 1)"
            ),
            Err(_) => (None, 0usize),
        };

        // D4 §3 search stamp (UQFF ≥ 0.3.0). Pre-0.3.0 payloads end after the
        // rotation section, so EOF means "unstamped".
        let search = match buffer.read_u8() {
            Ok(byte) => QtipSearchStamp::from_wire(byte)?,
            Err(_) => QtipSearchStamp::Unstamped,
        };

        // wave13-AF search-detail flags byte — same wire rule as the LUT rung,
        // deliberately: one format, not one per rung. Mandatory whenever a
        // stamp is present, so a truncated payload fails closed instead of
        // reading as an exhaustive unweighted bake.
        let search_detail = match search {
            QtipSearchStamp::Unstamped => QtipSearchDetail::Unknown,
            stamp => {
                let flags = buffer.read_u8().map_err(|_| {
                    candle_core::Error::Msg(
                        "Qtip2bLayer: payload carries a search stamp but no search-detail \
                         flags byte; refusing a truncated artifact rather than assuming an \
                         exhaustive unweighted bake."
                            .into(),
                    )
                })?;
                QtipSearchDetail::from_wire(flags, stamp, || {
                    buffer.read_u16::<LittleEndian>().map_err(|_| {
                        candle_core::Error::Msg(
                            "Qtip2bLayer: search-detail flags claim a beam but the width is \
                             missing (truncated payload)."
                                .into(),
                        )
                    })
                })?
            }
        };

        Ok((
            Self {
                blocks,
                row_scales,
                bias,
                in_features,
                num_experts,
                rotation_signs,
                rotation_block,
                mcg_mult,
                search,
                search_detail,
                // UQFF today only carries the uniform 2-bit format; a
                // mixed-bpw payload would arrive with the 4-bit rung's
                // serde revision.
                expert_bpw: num_experts.map(ExpertBpwTable::uniform_2bit),
            },
            ext_bias,
        ))
    }
}

impl QuantizedSerde for Qtip2bLayer {
    fn name(&self) -> &'static str {
        "qtip2b-layer"
    }
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.bias.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        // No codebook tensor is persisted — only the 4-byte MCG multiplier.
        // 2-D and 3-D stacked-expert layers share one field order: UQFF
        // tensor payloads are self-describing (rank + dims), so the 3-D
        // `blocks [E, N, packed_K]` / `row_scales [E, N]` round-trip through
        // the same `serialize_tensor` calls as the 2-D layout — the same
        // scheme as the LUT rung (PR #3). The deserializer infers the expert
        // mode from the blocks rank — see `deserialize_concrete`.
        if self.num_experts.is_some() && bias.is_some() {
            // 3-D MoE stacks are bias-free (`quantize_with_options_3d` never
            // attaches one); refuse rather than silently attach a single [N]
            // bias vector to E experts.
            candle_core::bail!("Qtip2bLayer::serialize: 3-D stacked-expert layers are bias-free");
        }
        let mut buffer = Vec::new();
        buffer.extend(&UQFF_VERSION.to_le_bytes());
        buffer.push(QuantizedSerdeType::Qtip2b as u8);
        buffer.push(bias.is_some() as u8);
        buffer.extend(&(self.in_features as u32).to_le_bytes());
        buffer.extend(&self.mcg_mult.to_le_bytes());

        serialize_tensor(&mut buffer, &self.blocks)?;
        serialize_tensor(&mut buffer, &self.row_scales)?;
        if let Some(bias) = &bias {
            serialize_tensor(&mut buffer, bias)?;
        }
        if let Some(signs) = &self.rotation_signs {
            buffer.push(1u8);
            buffer.extend(&(self.rotation_block as u32).to_le_bytes());
            serialize_tensor(&mut buffer, signs)?;
        } else {
            buffer.push(0u8);
        }
        // D4 §3: stamp the trellis search (UQFF 0.3.0).
        match self.search.to_wire() {
            Some(byte) => buffer.push(byte),
            None => candle_core::bail!(
                "Qtip2bLayer::serialize: refusing to write an artifact with unknown search \
                 provenance. This layer came from a pre-0.3.0 payload or a format that carries \
                 no stamp; re-quantize from the source weights so the stamp is earned rather \
                 than assumed (DOCTRINE D4)."
            ),
        }
        // wave13-AF: the search detail beside the stamp. Always
        // exhaustive/unweighted on this rung today, but written through the
        // same encoder so the two rungs cannot drift apart on the wire.
        let (flags, beam_width) = self.search_detail.to_wire(self.search)?;
        buffer.push(flags);
        if let Some(w) = beam_width {
            buffer.extend(&w.to_le_bytes());
        }
        Ok(Cow::from(buffer))
    }
    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        let (layer, _) = Self::deserialize_concrete(data, device, guard)?;
        Ok(Arc::new(layer))
    }
    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        let (layer, ext_bias) = Self::deserialize_concrete(data, device, guard)?;
        Ok((Arc::new(layer), ext_bias))
    }
}

// ===========================================================================
// Tests (CPU — must run on macOS)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers -----------------------------------------------------------

    /// Independent fp16-bits → f32 decoder that does NOT use the `half`
    /// crate. Handles normals, subnormals, zeros (inf/nan can't be produced
    /// by the masked/XORed pattern but are decoded correctly anyway).
    fn f16_bits_to_f32_independent(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
        let exp = ((bits >> 10) & 0x1F) as i32;
        let mant = (bits & 0x3FF) as f32;
        if exp == 0 {
            // Subnormal: mant * 2^-24
            sign * mant * (2.0f32).powi(-24)
        } else if exp == 31 {
            if mant == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        } else {
            sign * (1.0 + mant / 1024.0) * (2.0f32).powi(exp - 15)
        }
    }

    /// Independent reimplementation of the codeword function using u64
    /// arithmetic and the manual fp16 decoder above.
    fn mcg_codeword_independent(state: u32, mult: u32) -> f32 {
        let x = ((state as u64 & 0xFFFF) * mult as u64) & 0xFFFF_FFFF;
        let m = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
        let hi = f16_bits_to_f32_independent((m >> 16) as u16);
        let lo = f16_bits_to_f32_independent((m & 0xFFFF) as u16);
        hi + lo
    }

    fn gaussian_fixture(len: usize, seed_offset: u64, scale: f32) -> Vec<f32> {
        // Same splitmix64 + Box-Muller generator as the LUT rung's
        // "realistic Gaussian" test fixture (mod.rs).
        let mut out = vec![0.0f32; len];
        for (i, v) in out.iter_mut().enumerate() {
            let mut z = (i as u64 + seed_offset).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos() * scale;
        }
        out
    }

    fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
        let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    fn decode_symbols_2b(symbols: &[u8], codebook: &[f32]) -> Vec<f32> {
        let mut state: u32 = 0;
        symbols
            .iter()
            .map(|&sym| {
                state = ((state << K2B) | sym as u32) & STATE_MASK_2B;
                codebook[state as usize]
            })
            .collect()
    }

    fn mse(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            / a.len() as f32
    }

    /// Tuning harness for QTIP2B_SCALE_DIVISOR (ignored; run with
    /// `-- --ignored --nocapture`). Sweeps the search-range divisor across
    /// several Gaussian fixtures. Measured 2026-08: 3.4 and 3.62 tie within
    /// fixture noise (±0.008); 3.62 kept as the principled 3σ choice. The
    /// `iters` loop also evaluates alternating symbols↔scale minimization,
    /// which did NOT improve matmul cosine beyond one LS refinement pass.
    #[test]
    #[ignore]
    fn divisor_sweep() {
        let n = 32;
        let k_in = 256;
        let batch = 4;
        let cb = computed_codebook(QTIP2B_MCG_MULT);
        let signs = generate_signs(QTIP_ROTATION_SEED, k_in);
        let block = rotation_block_size(k_in);

        for seed in [0u64, 555_000, 42_000_000] {
            let wdata = gaussian_fixture(n * k_in, seed, 0.5);
            let xdata = gaussian_fixture(batch * k_in, seed + 1_000_000, 1.0);

            // Dense reference.
            let mut dense = vec![0f32; batch * n];
            for b in 0..batch {
                for r in 0..n {
                    let mut acc = 0f32;
                    for k in 0..k_in {
                        acc += xdata[b * k_in + k] * wdata[r * k_in + k];
                    }
                    dense[b * n + r] = acc;
                }
            }

            for iters in [1usize] {
                for divisor in [3.0f32, 3.1, 3.2, 3.3, 3.4, 3.62] {
                    let mut w_hat = vec![0f32; n * k_in];
                    for r in 0..n {
                        let mut row = wdata[r * k_in..(r + 1) * k_in].to_vec();
                        apply_block_rotation(&mut row, &signs, block);
                        let max_abs = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
                        let mut scale = if max_abs == 0.0 {
                            1.0
                        } else {
                            max_abs / divisor
                        };
                        let mut cvals = vec![0f32; k_in];
                        for _ in 0..iters {
                            let target: Vec<f32> = row.iter().map(|w| w / scale).collect();
                            let syms = viterbi_quantize_row_2b(&target, &cb);
                            let (mut wc, mut cc) = (0f64, 0f64);
                            let mut state = 0u32;
                            for (t, &s) in syms.iter().enumerate() {
                                state = ((state << K2B) | s as u32) & STATE_MASK_2B;
                                let c = cb[state as usize];
                                cvals[t] = c;
                                wc += row[t] as f64 * c as f64;
                                cc += (c as f64) * (c as f64);
                            }
                            if cc > 0.0 && wc > 0.0 {
                                scale = (wc / cc) as f32;
                            }
                        }
                        let mut recon: Vec<f32> = cvals.iter().map(|&c| c * scale).collect();
                        apply_block_rotation(&mut recon, &signs, block);
                        w_hat[r * k_in..(r + 1) * k_in].copy_from_slice(&recon);
                    }
                    let mut q = vec![0f32; batch * n];
                    for b in 0..batch {
                        for r in 0..n {
                            let mut acc = 0f32;
                            for k in 0..k_in {
                                acc += xdata[b * k_in + k] * w_hat[r * k_in + k];
                            }
                            q[b * n + r] = acc;
                        }
                    }
                    println!(
                        "seed {seed} iters {iters} divisor {divisor:.2} -> matmul cos {:.4}",
                        cos_sim(&dense, &q)
                    );
                }
            }
        }
    }

    // -- codebook ----------------------------------------------------------

    /// The codebook generator matches a fully independent reimplementation
    /// (u64 arithmetic + hand-rolled fp16 decode, no `half` crate) at every
    /// one of the 65,536 states, bit for bit.
    #[test]
    fn codebook_matches_independent_reimplementation() {
        for state in 0..CB_SIZE_2B as u32 {
            let a = mcg_codeword(state, QTIP2B_MCG_MULT);
            let b = mcg_codeword_independent(state, QTIP2B_MCG_MULT);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "codeword mismatch at state {state}: {a} vs {b}"
            );
        }
    }

    /// Codebook distribution sanity: mean ≈ 0, σ ≈ 1.206 (measured design
    /// constant), max |w| < 4.1, and no NaN/inf anywhere.
    #[test]
    fn codebook_distribution_is_well_formed() {
        let cb = computed_codebook(QTIP2B_MCG_MULT);
        assert_eq!(cb.len(), CB_SIZE_2B);
        let n = cb.len() as f64;
        let mean: f64 = cb.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var: f64 = cb.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        let max_abs = cb.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        for v in &cb {
            assert!(v.is_finite(), "non-finite codeword");
        }
        assert!(mean.abs() < 0.01, "codebook mean {mean} too far from 0");
        assert!(
            (std - 1.2064).abs() < 0.01,
            "codebook σ {std} != measured design constant 1.2064"
        );
        assert!(max_abs < 4.1, "codebook max |w| {max_abs} out of range");
    }

    // -- viterbi -----------------------------------------------------------

    /// Naive O(T·2^L·2^K) Viterbi reference — proves the group-min version
    /// is bit-identical (same j-scan order, same strict-`<` tie-break).
    /// Index-based loops kept deliberately to mirror the optimized code
    /// line-for-line (same rationale as the LUT rung's naive reference).
    #[allow(clippy::needless_range_loop)]
    fn viterbi_quantize_row_2b_naive(target_row: &[f32], codebook: &[f32]) -> Vec<u8> {
        let num_symbols = target_row.len();
        let inf = f32::INFINITY;
        let mut prev_cost = vec![inf; CB_SIZE_2B];
        let mut curr_cost = vec![inf; CB_SIZE_2B];
        let mut backtrace: Vec<Vec<u8>> = Vec::with_capacity(num_symbols);

        for s in 0..ALPHABET_2B {
            let d = codebook[s] - target_row[0];
            prev_cost[s] = d * d;
        }

        for t in 1..num_symbols {
            let target_t = target_row[t];
            let mut bt_t = vec![0u8; CB_SIZE_2B];
            for s in 0..CB_SIZE_2B {
                let d = codebook[s] - target_t;
                let err = d * d;
                let mut best_cost = inf;
                let mut best_j: u8 = 0;
                for j in 0..ALPHABET_2B as u32 {
                    let p = (j << (L2B - K2B)) | (s as u32 >> K2B);
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
        for s in 0..CB_SIZE_2B {
            if prev_cost[s] < best_final_cost {
                best_final_cost = prev_cost[s];
                best_final = s as u32;
            }
        }

        let mut symbols = vec![0u8; num_symbols];
        let mut s = best_final;
        symbols[num_symbols - 1] = (s & (ALPHABET_2B as u32 - 1)) as u8;
        for t in (1..num_symbols).rev() {
            let j = backtrace[t - 1][s as usize] as u32;
            let prev_s = (j << (L2B - K2B)) | (s >> K2B);
            symbols[t - 1] = (prev_s & (ALPHABET_2B as u32 - 1)) as u8;
            s = prev_s;
        }
        symbols
    }

    /// REGRESSION GUARD: group-min Viterbi is byte-identical to the naive
    /// per-state reference across shapes and distributions (incl. exact ties).
    #[test]
    fn viterbi_group_min_matches_naive_bit_for_bit() {
        let cb = computed_codebook(QTIP2B_MCG_MULT);
        for num_symbols in [1usize, 2, 3, 7, 16, 33, 64] {
            for variant in 0..4u32 {
                let target: Vec<f32> = (0..num_symbols)
                    .map(|i| {
                        let x = i as f32;
                        match variant {
                            0 => (x * 0.0313).sin() * 1.2,
                            1 => (x * 0.31).cos() * 1.5,
                            2 => ((x * 0.7).sin() + (x * 0.013).cos()) * 0.9,
                            _ => ((i % 5) as f32 - 2.0) * 0.5,
                        }
                    })
                    .collect();
                let opt = viterbi_quantize_row_2b(&target, &cb);
                let naive = viterbi_quantize_row_2b_naive(&target, &cb);
                assert_eq!(
                    opt, naive,
                    "group-min diverged at num_symbols={num_symbols}, variant={variant}"
                );
            }
        }
    }

    /// Viterbi reconstruction beats (or ties) greedy on the same input.
    #[test]
    fn viterbi_beats_greedy_2b() {
        let cb = computed_codebook(QTIP2B_MCG_MULT);
        let target = gaussian_fixture(128, 7, 1.0);
        let v_syms = viterbi_quantize_row_2b(&target, &cb);
        let g_syms = greedy_quantize_row_2b(&target, &cb);
        let v_err = mse(&target, &decode_symbols_2b(&v_syms, &cb));
        let g_err = mse(&target, &decode_symbols_2b(&g_syms, &cb));
        println!("qtip2b row MSE: viterbi={v_err:.6}, greedy={g_err:.6}");
        assert!(
            v_err <= g_err + 1e-6,
            "Viterbi MSE {v_err} > greedy MSE {g_err}"
        );
    }

    /// Symbols must decode back through the production state walk to the
    /// reconstruction Viterbi optimized (initial state 0 invariant).
    #[test]
    fn viterbi_symbols_valid_and_deterministic() {
        let cb = computed_codebook(QTIP2B_MCG_MULT);
        let target = gaussian_fixture(64, 3, 1.0);
        let a = viterbi_quantize_row_2b(&target, &cb);
        let b = viterbi_quantize_row_2b(&target, &cb);
        assert_eq!(a, b, "Viterbi not deterministic");
        for (i, s) in a.iter().enumerate() {
            assert!((*s as usize) < ALPHABET_2B, "symbol[{i}]={s} out of range");
        }
        let recon = decode_symbols_2b(&a, &cb);
        for v in &recon {
            assert!(v.is_finite());
        }
    }

    // -- pack/unpack -------------------------------------------------------

    #[test]
    fn pack_unpack_roundtrip() {
        let syms: Vec<u8> = (0..64).map(|i| ((i * 7 + 3) % 4) as u8).collect();
        let packed = pack_symbols_2b(&syms);
        assert_eq!(packed.len(), syms.len() / 4);
        for (t, &expected) in syms.iter().enumerate() {
            assert_eq!(unpack_symbol_2b(&packed, t), expected as u32);
        }
    }

    // -- layer: quantize → dequantize → matmul ------------------------------

    /// Bit-exact CPU determinism: two quantize calls on the same input give
    /// byte-identical blocks and scales.
    #[test]
    fn quantize_is_bit_exact_deterministic() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let k_in = 128;
        let wdata = gaussian_fixture(n * k_in, 42, 0.5);
        let w = Tensor::from_vec(wdata, (n, k_in), &device)?;

        let l1 = Qtip2bLayer::quantize_with_options_concrete(
            &w,
            None,
            &device,
            QtipMode::Viterbi,
            true,
        )?;
        let l2 = Qtip2bLayer::quantize_with_options_concrete(
            &w,
            None,
            &device,
            QtipMode::Viterbi,
            true,
        )?;

        let b1: Vec<u8> = l1.blocks.flatten_all()?.to_vec1()?;
        let b2: Vec<u8> = l2.blocks.flatten_all()?.to_vec1()?;
        assert_eq!(b1, b2, "packed blocks differ between identical runs");
        let s1: Vec<f32> = l1.row_scales.flatten_all()?.to_vec1()?;
        let s2: Vec<f32> = l2.row_scales.flatten_all()?.to_vec1()?;
        assert_eq!(
            s1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            s2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "row scales differ between identical runs"
        );
        Ok(())
    }

    /// Round-trip: quantize → dequantize with bounded relative L2 error.
    #[test]
    fn roundtrip_rel_l2_bounded() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let k_in = 128;
        let wdata = gaussian_fixture(n * k_in, 11, 0.5);
        let w = Tensor::from_vec(wdata.clone(), (n, k_in), &device)?;
        let layer = Qtip2bLayer::quantize_with_mode(&w, None, &device, QtipMode::Viterbi)?;
        let dq: Vec<f32> = layer
            .dequantize_w()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let mut sum_err = 0.0f32;
        let mut sum_orig = 0.0f32;
        for (o, r) in wdata.iter().zip(dq.iter()) {
            sum_err += (o - r).powi(2);
            sum_orig += o.powi(2);
        }
        let rel = (sum_err / sum_orig).sqrt();
        println!("qtip2b Viterbi roundtrip rel L2: {rel:.4}");
        // The LUT rung's Viterbi lands ~0.15–0.3 on comparable fixtures.
        assert!(rel < 0.35, "qtip2b roundtrip rel L2 {rel} > 0.35");
        Ok(())
    }

    /// THE quality bar (mission requirement): on the same synthetic Gaussian
    /// fixtures as the LUT rung's `viterbi_matmul_cosine_similarity_realistic`
    /// test, qtip2b's Viterbi+rotation matmul cosine must (a) clear the
    /// paper-quality 0.95 gate on EVERY fixture and (b) average ≥ the LUT
    /// rung on the identical fixtures.
    ///
    /// Averaged over several seeds because the per-fixture cosine of both
    /// rungs fluctuates by ~±0.008 at this size (n=32, k_in=256) — a
    /// single-seed comparison measures fixture noise, not codebook quality.
    /// Both paths are fully deterministic, so the numbers below are stable.
    #[test]
    fn matmul_cosine_meets_lut_rung_bar_on_gaussian_fixture() -> Result<()> {
        let device = Device::Cpu;
        let n = 32;
        let k_in = 256;
        let batch = 4;

        let mut sum_2b = 0f32;
        let mut sum_lut = 0f32;
        let seeds = [0u64, 555_000, 42_000_000];
        for &seed in &seeds {
            let wdata = gaussian_fixture(n * k_in, seed, 0.5);
            let xdata = gaussian_fixture(batch * k_in, seed + 1_000_000, 1.0);

            let w = Tensor::from_vec(wdata, (n, k_in), &device)?;
            let x = Tensor::from_vec(xdata, (batch, k_in), &device)?;
            let dense_v: Vec<f32> = x.matmul(&w.t()?)?.flatten_all()?.to_vec1()?;

            let run = |layer: Arc<dyn QuantMethod>| -> Result<f32> {
                let qv: Vec<f32> = layer
                    .forward(&x)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1()?;
                Ok(cos_sim(&dense_v, &qv))
            };

            let bitshift_cos = run(Qtip2bLayer::quantize_with_mode(
                &w,
                None,
                &device,
                QtipMode::Viterbi,
            )?)?;
            let lut_cos = run(super::super::QtipLayer::quantize_with_mode(
                &w,
                None,
                &device,
                QtipMode::Viterbi,
            )?)?;

            println!(
                "Gaussian fixture seed {seed} (n={n}, k_in={k_in}, batch={batch}): \
                 qtip2b(computed)={bitshift_cos:.4}, qtip2(LUT)={lut_cos:.4}"
            );

            assert!(
                bitshift_cos >= 0.95,
                "qtip2b Viterbi+rotation cos {bitshift_cos} < 0.95 paper bar (seed {seed})"
            );
            sum_2b += bitshift_cos;
            sum_lut += lut_cos;
        }

        let mean_2b = sum_2b / seeds.len() as f32;
        let mean_lut = sum_lut / seeds.len() as f32;
        println!(
            "mean over {} seeds: qtip2b={mean_2b:.4}, qtip2(LUT)={mean_lut:.4}",
            seeds.len()
        );
        assert!(
            mean_2b >= mean_lut - 0.005,
            "qtip2b mean cos {mean_2b} more than 0.005 below LUT rung mean {mean_lut}"
        );
        Ok(())
    }

    /// Greedy mode also produces a usable (>0.85) matmul cosine.
    #[test]
    fn greedy_matmul_cosine_floor() -> Result<()> {
        let device = Device::Cpu;
        let n = 16;
        let k_in = 256;
        let batch = 2;
        let wdata = gaussian_fixture(n * k_in, 5, 0.5);
        let xdata = gaussian_fixture(batch * k_in, 2_000_000, 1.0);
        let w = Tensor::from_vec(wdata, (n, k_in), &device)?;
        let x = Tensor::from_vec(xdata, (batch, k_in), &device)?;
        let dense_v: Vec<f32> = x.matmul(&w.t()?)?.flatten_all()?.to_vec1()?;

        let layer = Qtip2bLayer::quantize_greedy_fixture(&w, None, &device)?;
        let qv: Vec<f32> = layer
            .forward(&x)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let c = cos_sim(&dense_v, &qv);
        println!("qtip2b greedy matmul cos: {c:.4}");
        assert!(c > 0.85, "qtip2b greedy cos {c} < 0.85");
        Ok(())
    }

    // -- 3-D expert stacks ---------------------------------------------------

    fn build_3d_gaussian_weight(e: usize, n: usize, k_in: usize) -> Result<Tensor> {
        let data = gaussian_fixture(e * n * k_in, 77, 0.5);
        Tensor::from_vec(data, (e, n, k_in), &Device::Cpu)
    }

    /// 3-D quantize per-expert output matches independently quantizing each
    /// expert's 2-D slice (identical packed bytes and scales).
    #[test]
    fn qtip2b_3d_per_expert_matches_2d() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k_in) = (3, 4, 64);
        let w3 = build_3d_gaussian_weight(e, n, k_in)?;

        let layer3 = Qtip2bLayer::quantize_with_options_3d(&w3, &device, QtipMode::Viterbi, true)?;
        let dq3 = layer3.dequantize_w()?; // [E, N, K]

        for expert in 0..e {
            let w2 = w3.narrow(0, expert, 1)?.squeeze(0)?.contiguous()?;
            let layer2 =
                Qtip2bLayer::quantize_with_options(&w2, None, &device, QtipMode::Viterbi, true)?;
            let dq2: Vec<f32> = layer2
                .dequantize_w()?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let dq3_e: Vec<f32> = dq3
                .narrow(0, expert, 1)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            assert_eq!(
                dq2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                dq3_e.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "expert {expert} dequant differs between 3-D and 2-D quantize"
            );
        }
        Ok(())
    }

    /// 3-D forward bails (must route through gather_forward).
    #[test]
    fn qtip2b_3d_forward_bails() -> Result<()> {
        let device = Device::Cpu;
        let w3 = build_3d_gaussian_weight(2, 4, 64)?;
        let layer = Qtip2bLayer::quantize_with_options_3d(&w3, &device, QtipMode::Greedy, false)?;
        let x = Tensor::zeros((1, 64), DType::F32, &device)?;
        assert!(layer.forward(&x).is_err());
        Ok(())
    }

    /// gather_forward CPU matches a manual per-pair dequant+matmul reference.
    #[test]
    fn qtip2b_gather_forward_cpu_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k_in) = (4, 6, 64);
        let w3 = build_3d_gaussian_weight(e, n, k_in)?;
        let layer_dyn =
            Qtip2bLayer::quantize_with_options_3d(&w3, &device, QtipMode::Viterbi, true)?;

        let n_tokens = 3;
        let top_k = 2;
        let a_data = gaussian_fixture(n_tokens * top_k * k_in, 9_000, 1.0);
        let a = Tensor::from_vec(a_data, (n_tokens, top_k, k_in), &device)?;
        let idx: Vec<u32> = vec![0, 1, 2, 3, 1, 0];
        let indices = Tensor::from_vec(idx.clone(), (n_tokens, top_k), &device)?;

        let out = layer_dyn.gather_forward(&a, &indices)?;
        assert_eq!(out.dims(), &[n_tokens, top_k, n]);

        // Reference: full dequant, per-pair manual matmul.
        let dq = layer_dyn.dequantize_w()?.to_dtype(DType::F32)?; // [E, N, K]
        let out_v: Vec<f32> = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let a_v: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let dq_v: Vec<f32> = dq.flatten_all()?.to_vec1()?;

        for (pair, &expert) in idx.iter().enumerate() {
            let a_row = &a_v[pair * k_in..(pair + 1) * k_in];
            for row in 0..n {
                let w_row = &dq_v
                    [(expert as usize * n + row) * k_in..(expert as usize * n + row + 1) * k_in];
                let expected: f32 = a_row.iter().zip(w_row.iter()).map(|(x, w)| x * w).sum();
                let got = out_v[pair * n + row];
                // The BF16 weight round-trip inside gather_forward_cpu loses
                // precision; compare with a tolerance scaled to magnitude.
                let tol = 1e-2f32.max(expected.abs() * 0.02);
                assert!(
                    (expected - got).abs() < tol,
                    "pair {pair} row {row}: expected {expected}, got {got}"
                );
            }
        }
        Ok(())
    }

    /// gather_forward rejects a 2-D layer.
    #[test]
    fn qtip2b_gather_forward_rejects_2d_layer() -> Result<()> {
        let device = Device::Cpu;
        let wdata = gaussian_fixture(4 * 64, 21, 0.5);
        let w = Tensor::from_vec(wdata, (4, 64), &device)?;
        let layer = Qtip2bLayer::quantize_greedy_fixture(&w, None, &device)?;
        let a = Tensor::zeros((1, 2, 64), DType::F32, &device)?;
        let indices = Tensor::zeros((1, 2), DType::U32, &device)?;
        assert!(layer.gather_forward(&a, &indices).is_err());
        Ok(())
    }

    // -- serde ----------------------------------------------------------------

    /// UQFF round-trip: serialize → deserialize preserves dequantized
    /// weights bit-for-bit and all metadata (2-D, with rotation + bias).
    #[test]
    fn uqff_roundtrip_2d() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let k_in = 128;
        let wdata = gaussian_fixture(n * k_in, 33, 0.5);
        let w = Tensor::from_vec(wdata, (n, k_in), &device)?;
        let bias = Tensor::from_vec(gaussian_fixture(n, 99, 0.1), (n,), &device)?;
        let layer = Qtip2bLayer::quantize_with_options_concrete(
            &w,
            Some(bias),
            &device,
            QtipMode::Viterbi,
            true,
        )?;

        let payload = layer.serialize()?;

        // No 512 KB LUT in the payload: blocks (n*k_in/4) + scales (n*4) +
        // bias (n*4) + signs (k_in*4) + tensor headers. Anything over 3× that
        // budget means a codebook tensor snuck in.
        let expected_core = n * k_in / 4 + n * 4 + n * 4 + k_in * 4;
        assert!(
            payload.len() < expected_core * 3 + 1024,
            "payload {} bytes — did a codebook tensor get serialized?",
            payload.len()
        );

        let guard = QuantizeOntoGuard::new();
        let (restored, ext_bias) = Qtip2bLayer::deserialize_ext_bias(payload, &device, guard)?;
        assert!(ext_bias.is_some());

        // wave13-AF: this rung shares the LUT rung's wire rule, so its
        // search-detail flags byte must survive the round-trip too. The bake
        // claim is always exhaustive/unweighted today — `viterbi_quantize_row_2b`
        // IS the exhaustive DP and this rung has no weighted metric — and
        // pinning it is what will catch the claim drifting when qtip2b grows a
        // beam kernel. (The wire half is covered by the byte-exact dequant
        // comparison below: a mis-sized provenance tail desynchronises the
        // parse.)
        assert_eq!(layer.search_detail(), QtipSearchDetail::EXHAUSTIVE_MSE);
        let (typed, _) = Qtip2bLayer::deserialize_concrete(
            layer.serialize()?,
            &device,
            QuantizeOntoGuard::new(),
        )?;
        assert_eq!(
            typed.search_detail(),
            QtipSearchDetail::EXHAUSTIVE_MSE,
            "qtip2b search detail did not survive the UQFF round-trip"
        );

        let dq_orig: Vec<f32> = layer
            .dequantize_weights()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let dq_rest: Vec<f32> = restored
            .dequantize_w()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(
            dq_orig.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            dq_rest.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "dequantized weights differ after UQFF round-trip"
        );
        Ok(())
    }

    /// UQFF round-trip for a 3-D expert stack (rank-recovered num_experts,
    /// PR #3's self-describing mechanism shared with the LUT rung).
    #[test]
    fn uqff_roundtrip_3d() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k_in) = (3, 4, 64);
        let w3 = build_3d_gaussian_weight(e, n, k_in)?;
        let layer_dyn =
            Qtip2bLayer::quantize_with_options_3d(&w3, &device, QtipMode::Viterbi, true)?;

        // 3-D stacks are bias-free by contract: the serializer must refuse
        // to attach a bias rather than silently share one [N] vector across
        // E experts (mirrors the LUT rung's guard).
        let fake_bias = Tensor::zeros((n,), DType::F32, &device)?;
        assert!(
            layer_dyn.serialize_with_bias(Some(fake_bias)).is_err(),
            "serialize_with_bias must refuse a bias on a 3-D expert stack"
        );

        let payload = layer_dyn.serialize()?;

        // Typed deserialize: expert mode + metadata must be recovered from
        // the self-describing tensor ranks alone.
        let (typed, typed_bias) = Qtip2bLayer::deserialize_concrete(
            Cow::Borrowed(&payload),
            &device,
            QuantizeOntoGuard::new(),
        )?;
        assert_eq!(typed.num_experts, Some(e));
        assert_eq!(typed.blocks.dims(), &[e, n, k_in / SYMS_PER_BYTE]);
        assert_eq!(typed.row_scales.dims(), &[e, n]);
        assert!(typed_bias.is_none());

        let guard = QuantizeOntoGuard::new();
        let (restored, _) = Qtip2bLayer::deserialize_ext_bias(payload, &device, guard)?;

        let dq_orig: Vec<f32> = layer_dyn
            .dequantize_w()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let dq_rest: Vec<f32> = restored
            .dequantize_w()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(
            dq_orig.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            dq_rest.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "3-D dequant differs after UQFF round-trip"
        );

        // Restored layer must still gather_forward (num_experts recovered).
        let a = Tensor::zeros((1, 2, k_in), DType::F32, &device)?;
        let indices = Tensor::zeros((1, 2), DType::U32, &device)?;
        restored.gather_forward(&a, &indices)?;
        Ok(())
    }

    // -- grouped GEMM (batched MoE prefill) ---------------------------------

    /// Quantize a 3-D expert stack and hand back the **concrete** layer
    /// (via the UQFF round-trip — `quantize_with_options_3d` returns
    /// `Arc<dyn QuantMethod>` and the trait has no downcast hook).
    fn concrete_3d_layer(
        w3: &Tensor,
        device: &Device,
        mode: QtipMode,
        rotation: bool,
    ) -> Result<Qtip2bLayer> {
        let layer = Qtip2bLayer::quantize_with_options_3d(w3, device, mode, rotation)?;
        let payload = layer.serialize()?;
        // `_unchecked`: several callers build a cheap greedy fixture, which the
        // D4 load gate refuses on purpose. That gate has its own tests
        // (`qtip/greedy_ban_tests.rs`); this helper only needs the typed layer.
        let (concrete, _) =
            Qtip2bLayer::deserialize_concrete_unchecked(payload, device, QuantizeOntoGuard::new())?;
        Ok(concrete)
    }

    /// Ragged (token, slot) pair fixtures for the grouped tests: returns
    /// `(expert ids per pair)` for a handful of adversarial distributions.
    fn ragged_index_fixtures(num_experts: usize) -> Vec<Vec<u32>> {
        let e = num_experts as u32;
        vec![
            // 1 giant expert + many tiny (some experts empty).
            {
                let mut v = vec![0u32; 40];
                v.extend([1, 2, 2, e - 1, 1, e - 2]);
                v
            },
            // All pairs on one expert; every other expert empty. 33 pairs
            // = 2 full m-tiles + 1 single-row tail tile.
            vec![e / 2; 33],
            // Single pair.
            vec![e - 1],
            // Round-robin (every expert non-empty, all groups partial).
            (0..2 * e).map(|i| i % e).collect(),
        ]
    }

    /// Correctness harness (a): the grouped batched output matches a
    /// per-pair dense matmul on the dequantized weights, across ragged
    /// group shapes.
    #[test]
    fn gather_forward_batched_matches_per_pair_dense() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k_in) = (4usize, 24usize, 64usize);
        let w3 = build_3d_gaussian_weight(e, n, k_in)?;
        let layer = concrete_3d_layer(&w3, &device, QtipMode::Viterbi, true)?;

        for (case, idx) in ragged_index_fixtures(e).into_iter().enumerate() {
            let p = idx.len();
            let a_data = gaussian_fixture(p * k_in, 1000 + case as u64, 1.0);
            let a2d = Tensor::from_vec(a_data, (p, k_in), &device)?;
            let indices = Tensor::from_vec(idx.clone(), (p,), &device)?;

            let grouped: Vec<f32> = layer
                .gather_forward_batched(&a2d, &indices)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;

            // Per-pair dense reference: one matmul per pair against that
            // pair's expert, no grouping anywhere.
            let mut dense = vec![0f32; p * n];
            for (pair, &expert) in idx.iter().enumerate() {
                let w_e = layer
                    .dequantize_expert_weights_unrotated(expert as usize)?
                    .to_dtype(DType::F32)?;
                let x_p = a2d.narrow(0, pair, 1)?;
                let y_p: Vec<f32> = x_p.matmul(&w_e.t()?)?.flatten_all()?.to_vec1()?;
                dense[pair * n..(pair + 1) * n].copy_from_slice(&y_p);
            }

            let c = cos_sim(&grouped, &dense);
            assert!(
                c > 0.9999,
                "case {case}: grouped batched cos {c} vs per-pair dense"
            );
        }
        Ok(())
    }

    /// The 3-D `gather_forward` API and the flattened
    /// `gather_forward_batched` API compute the same thing.
    #[test]
    fn gather_forward_batched_matches_gather_forward() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k_in) = (4usize, 16usize, 64usize);
        let w3 = build_3d_gaussian_weight(e, n, k_in)?;
        let layer = concrete_3d_layer(&w3, &device, QtipMode::Viterbi, true)?;

        let (n_tokens, top_k) = (6usize, 2usize);
        let p = n_tokens * top_k;
        let a_data = gaussian_fixture(p * k_in, 4321, 1.0);
        let a3 = Tensor::from_vec(a_data, (n_tokens, top_k, k_in), &device)?;
        let idx: Vec<u32> = (0..p as u32).map(|i| i % e as u32).collect();
        let idx3 = Tensor::from_vec(idx.clone(), (n_tokens, top_k), &device)?;

        let via_gather: Vec<f32> = layer.gather_forward(&a3, &idx3)?.flatten_all()?.to_vec1()?;
        let via_batched: Vec<f32> = layer
            .gather_forward_batched(
                &a3.reshape((p, k_in))?,
                &Tensor::from_vec(idx, (p,), &device)?,
            )?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(
            via_gather.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            via_batched.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "gather_forward and gather_forward_batched diverged"
        );
        Ok(())
    }

    /// Correctness harness (c), CPU half: same input → bit-identical output.
    #[test]
    fn gather_forward_batched_deterministic() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k_in) = (3usize, 8usize, 64usize);
        let w3 = build_3d_gaussian_weight(e, n, k_in)?;
        let layer = concrete_3d_layer(&w3, &device, QtipMode::Greedy, false)?;

        let idx: Vec<u32> = vec![2, 0, 2, 2, 1, 0, 2];
        let p = idx.len();
        let a_data = gaussian_fixture(p * k_in, 99, 1.0);
        let a2d = Tensor::from_vec(a_data, (p, k_in), &device)?;
        let indices = Tensor::from_vec(idx, (p,), &device)?;

        let y1: Vec<f32> = layer
            .gather_forward_batched(&a2d, &indices)?
            .flatten_all()?
            .to_vec1()?;
        let y2: Vec<f32> = layer
            .gather_forward_batched(&a2d, &indices)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(
            y1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "batched gather is not deterministic"
        );
        Ok(())
    }

    /// The batched entry rejects mismatched shapes and 2-D layers.
    #[test]
    fn gather_forward_batched_validates_inputs() -> Result<()> {
        let device = Device::Cpu;
        let (e, _n, k_in) = (2usize, 4usize, 64usize);
        let w3 = build_3d_gaussian_weight(e, 4, k_in)?;
        let layer = concrete_3d_layer(&w3, &device, QtipMode::Greedy, false)?;

        // Wrong K.
        let a_bad = Tensor::zeros((2, k_in / 2), DType::F32, &device)?;
        let idx = Tensor::zeros((2,), DType::U32, &device)?;
        assert!(layer.gather_forward_batched(&a_bad, &idx).is_err());
        // Wrong indices length.
        let a_ok = Tensor::zeros((2, k_in), DType::F32, &device)?;
        let idx_bad = Tensor::zeros((3,), DType::U32, &device)?;
        assert!(layer.gather_forward_batched(&a_ok, &idx_bad).is_err());
        // Out-of-range expert id (CPU path validates).
        let idx_oob = Tensor::from_vec(vec![0u32, e as u32], (2,), &device)?;
        assert!(layer.gather_forward_batched(&a_ok, &idx_oob).is_err());

        // 2-D layers have no expert dim.
        let w2 = Tensor::from_vec(gaussian_fixture(4 * k_in, 5, 0.5), (4, k_in), &device)?;
        let layer2 = Qtip2bLayer::quantize_with_options_concrete(
            &w2,
            None,
            &device,
            QtipMode::Greedy,
            false,
        )?;
        assert!(layer2.gather_forward_batched(&a_ok, &idx).is_err());
        Ok(())
    }

    // -- CUDA parity (compile-gated; run on the GPU session) -----------------

    /// CUDA dequantize matches the CPU decode bit-for-bit modulo the BF16
    /// output rounding (compare in F32 with a tight tolerance).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dequantize_2b_matches_cpu() -> Result<()> {
        if !super::super::ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cuda = Device::new_cuda(0)?;
        let cpu = Device::Cpu;
        let n = 16;
        let k_in = 256;
        let wdata = gaussian_fixture(n * k_in, 4242, 0.5);
        let w_cpu = Tensor::from_vec(wdata, (n, k_in), &cpu)?;

        // Quantize on CPU (reference), then decode on both devices.
        let layer = Qtip2bLayer::quantize_with_options_concrete(
            &w_cpu,
            None,
            &cpu,
            QtipMode::Viterbi,
            true,
        )?;
        let cpu_dq: Vec<f32> = layer
            .dequantize_weights()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        let layer_cuda = Qtip2bLayer {
            blocks: layer.blocks.to_device(&cuda)?,
            row_scales: layer.row_scales.to_device(&cuda)?,
            bias: None,
            in_features: layer.in_features,
            num_experts: None,
            rotation_signs: layer
                .rotation_signs
                .as_ref()
                .map(|t| t.to_device(&cuda))
                .transpose()?,
            rotation_block: layer.rotation_block,
            mcg_mult: layer.mcg_mult,
            expert_bpw: None,
            search: layer.search,
            search_detail: layer.search_detail,
        };
        let cuda_dq: Vec<f32> = layer_cuda
            .dequantize_weights()?
            .to_dtype(DType::F32)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;

        let c = cos_sim(&cpu_dq, &cuda_dq);
        assert!(
            c > 0.999,
            "CUDA dequant cos {c} vs CPU — in-register decode diverged"
        );
        Ok(())
    }

    /// CUDA fused GEMV (single-token decode) matches dequantize+matmul.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fused_gemv_2b_matches_dequant_matmul() -> Result<()> {
        if !super::super::ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cuda = Device::new_cuda(0)?;
        let cpu = Device::Cpu;
        let n = 64;
        let k_in = 512;
        let wdata = gaussian_fixture(n * k_in, 555, 0.5);
        let w_cpu = Tensor::from_vec(wdata, (n, k_in), &cpu)?;
        let layer_cpu = Qtip2bLayer::quantize_with_options_concrete(
            &w_cpu,
            None,
            &cpu,
            QtipMode::Viterbi,
            true,
        )?;
        let layer = Qtip2bLayer {
            blocks: layer_cpu.blocks.to_device(&cuda)?,
            row_scales: layer_cpu.row_scales.to_device(&cuda)?,
            bias: None,
            in_features: layer_cpu.in_features,
            num_experts: None,
            rotation_signs: layer_cpu
                .rotation_signs
                .as_ref()
                .map(|t| t.to_device(&cuda))
                .transpose()?,
            rotation_block: layer_cpu.rotation_block,
            mcg_mult: layer_cpu.mcg_mult,
            expert_bpw: None,
            search: layer_cpu.search,
            search_detail: layer_cpu.search_detail,
        };

        let xdata = gaussian_fixture(k_in, 31337, 1.0);
        let x = Tensor::from_vec(xdata, (1, k_in), &cpu)?
            .to_device(&cuda)?
            .to_dtype(DType::BF16)?;

        // Fused GEMV path (n_tokens == 1).
        let y_fused: Vec<f32> = layer
            .forward(&x)?
            .to_dtype(DType::F32)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;

        // Reference: dequant (original frame) + dense matmul in F32.
        let w_dq = layer.dequantize_weights()?.to_dtype(DType::F32)?;
        let y_ref: Vec<f32> = x
            .to_dtype(DType::F32)?
            .matmul(&w_dq.t()?)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;

        let c = cos_sim(&y_fused, &y_ref);
        assert!(c > 0.995, "fused GEMV cos {c} vs dequant+matmul reference");
        Ok(())
    }

    /// CUDA gather-GEMV (on-device MoE dispatch) matches the CPU gather path.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_gather_gemv_2b_matches_cpu() -> Result<()> {
        if !super::super::ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cuda = Device::new_cuda(0)?;
        let cpu = Device::Cpu;
        let (e, n, k_in) = (4, 32, 256);
        let w3 = build_3d_gaussian_weight(e, n, k_in)?;
        let layer_dyn = Qtip2bLayer::quantize_with_options_3d(&w3, &cpu, QtipMode::Viterbi, true)?;

        // Rebuild a CUDA-resident copy through serde (also exercises UQFF).
        let payload = layer_dyn.serialize()?;
        let guard = QuantizeOntoGuard::new();
        let (layer_cuda, _) = Qtip2bLayer::deserialize_ext_bias(payload, &cuda, guard)?;

        let n_tokens = 2;
        let top_k = 2;
        let a_data = gaussian_fixture(n_tokens * top_k * k_in, 12345, 1.0);
        let a_cpu = Tensor::from_vec(a_data, (n_tokens, top_k, k_in), &cpu)?;
        let idx: Vec<u32> = vec![0, 3, 2, 1];
        let indices_cpu = Tensor::from_vec(idx, (n_tokens, top_k), &cpu)?;

        let out_cpu: Vec<f32> = layer_dyn
            .gather_forward(&a_cpu, &indices_cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        let a_cuda = a_cpu.to_device(&cuda)?.to_dtype(DType::BF16)?;
        let indices_cuda = indices_cpu.to_device(&cuda)?;
        let out_cuda: Vec<f32> = layer_cuda
            .gather_forward(&a_cuda, &indices_cuda)?
            .to_dtype(DType::F32)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;

        let c = cos_sim(&out_cpu, &out_cuda);
        assert!(c > 0.995, "gather GEMV cos {c} vs CPU gather reference");
        Ok(())
    }

    /// CUDA quantize (rotate + Viterbi kernel with in-register codebook)
    /// produces a layer whose dequant matches the CPU-quantized layer's
    /// dequant to high cosine (tie-breaking may differ at fp32 boundaries;
    /// this mirrors the LUT rung's `cuda_quantize_matches_cpu_dequantize_cos_sim`).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_quantize_2b_matches_cpu_dequantize_cos_sim() -> Result<()> {
        if !super::super::ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cuda = Device::new_cuda(0)?;
        let cpu = Device::Cpu;
        let n = 32;
        let k_in = 256;
        let wdata = gaussian_fixture(n * k_in, 888, 0.5);
        let w_cpu = Tensor::from_vec(wdata, (n, k_in), &cpu)?;

        let layer_cpu = Qtip2bLayer::quantize_with_options_concrete(
            &w_cpu,
            None,
            &cpu,
            QtipMode::Viterbi,
            true,
        )?;
        let w_cuda = w_cpu.to_device(&cuda)?;
        let layer_gpu = Qtip2bLayer::quantize_with_options_concrete(
            &w_cuda,
            None,
            &cuda,
            QtipMode::Viterbi,
            true,
        )?;

        let dq_cpu: Vec<f32> = layer_cpu
            .dequantize_weights()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let dq_gpu: Vec<f32> = layer_gpu
            .dequantize_weights()?
            .to_dtype(DType::F32)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;
        let c = cos_sim(&dq_cpu, &dq_gpu);
        assert!(c > 0.99, "GPU-quantized dequant cos {c} vs CPU-quantized");
        Ok(())
    }

    /// Correctness harness (b): the CUDA trellis grouped GEMM matches the
    /// CPU grouped reference (loop experts, dense matmul on dequantized
    /// weights) at cos > 0.9999 across ragged group shapes — 1 giant expert
    /// + many tiny, all-pairs-one-expert (with empty experts), a single
    /// pair, and round-robin partial groups. N = 96 exercises the n-tile
    /// tail (96 = 64 + 32) and 40+ pairs exercise m-tile tails.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_grouped_gemm_matches_cpu_reference_ragged_shapes() -> Result<()> {
        if !super::super::ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cuda = Device::new_cuda(0)?;
        let cpu = Device::Cpu;
        let (e, n, k_in) = (8usize, 96usize, 256usize);
        let w3 = build_3d_gaussian_weight(e, n, k_in)?;
        let layer_cpu = concrete_3d_layer(&w3, &cpu, QtipMode::Viterbi, true)?;
        let payload = layer_cpu.serialize()?;
        let (layer_cuda, _) =
            Qtip2bLayer::deserialize_concrete(payload, &cuda, QuantizeOntoGuard::new())?;

        for (case, idx) in ragged_index_fixtures(e).into_iter().enumerate() {
            let p = idx.len();
            let a_data = gaussian_fixture(p * k_in, 7000 + case as u64, 1.0);
            // Round activations to BF16 once so BOTH sides consume the same
            // inputs; the CPU reference then runs in F32.
            let a_bf16_cpu = Tensor::from_vec(a_data, (p, k_in), &cpu)?.to_dtype(DType::BF16)?;
            let indices_cpu = Tensor::from_vec(idx.clone(), (p,), &cpu)?;

            let reference: Vec<f32> = layer_cpu
                .gather_forward_batched(&a_bf16_cpu.to_dtype(DType::F32)?, &indices_cpu)?
                .flatten_all()?
                .to_vec1()?;

            let a_cuda = a_bf16_cpu.to_device(&cuda)?;
            let indices_cuda = indices_cpu.to_device(&cuda)?;
            let grouped: Vec<f32> = layer_cuda
                .gather_forward_batched(&a_cuda, &indices_cuda)?
                .to_dtype(DType::F32)?
                .to_device(&cpu)?
                .flatten_all()?
                .to_vec1()?;

            let c = cos_sim(&reference, &grouped);
            assert!(
                c > 0.9999,
                "case {case}: grouped GEMM cos {c} vs CPU grouped reference"
            );
        }
        Ok(())
    }

    /// Every compiled autotune variant — gen 1
    /// (kernels/qtip/qtip_bitshift_tune.cu) and gen 2
    /// (kernels/qtip/qtip_bitshift_tune2.cu) alike — meets the SAME parity
    /// contract as the legacy fixed-config kernel: fused single-token GEMV vs
    /// dequant+matmul, and on-device gather GEMV vs the CPU gather reference.
    ///
    /// Fixtures, in order:
    ///   * fused n=64 k=512 (packed_per_row 128),
    ///   * gather e=4 n=32 k=256,
    ///   * a k=4096 (packed_per_row 1024) fused fixture. This one is what
    ///     makes the sweep honest for gen 2: those variants stage whole
    ///     K-tiles of up to 1024 B per K-group, so the small fixtures reject
    ///     them and they would silently fall back to legacy. EVERY compiled
    ///     variant must be applicable here — asserted below — so no variant
    ///     can go unmeasured.
    ///   * an alignment-fallback shape (packed_per_row odd), where every
    ///     variant must be rejected by the tuned launcher and silently fall
    ///     back to the legacy kernel — exercising the production fallback
    ///     path, not just the happy path.
    ///
    /// The forced-variant knob is process-global; that is benign here
    /// because every variant satisfies the same correctness contract the
    /// other CUDA tests assert (a concurrent test at worst runs a different
    /// — equally correct — variant).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_gemv_2b_all_tuned_variants_match_reference() -> Result<()> {
        use super::super::tune::{
            gemv_variant_applicable, set_forced_gemv_variant, QTIP2B_GEMV_VARIANT_LEGACY,
        };
        if !super::super::ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let n_variants = super::super::tune::gemv_num_variants();
        if n_variants == 0 {
            return Ok(());
        }
        struct ResetForced;
        impl Drop for ResetForced {
            fn drop(&mut self) {
                super::super::tune::set_forced_gemv_variant(None);
            }
        }
        let _reset = ResetForced;

        let cuda = Device::new_cuda(0)?;
        let cpu = Device::Cpu;

        // --- fused single-token fixture (n=64, k=512 -> ppr=128, all
        // vector widths + smem stagings applicable) + reference -----------
        let (n, k_in) = (64usize, 512usize);
        let w_cpu = Tensor::from_vec(gaussian_fixture(n * k_in, 555, 0.5), (n, k_in), &cpu)?;
        let fused_cpu = Qtip2bLayer::quantize_with_options_concrete(
            &w_cpu,
            None,
            &cpu,
            QtipMode::Viterbi,
            true,
        )?;
        let fused = Qtip2bLayer {
            blocks: fused_cpu.blocks.to_device(&cuda)?,
            row_scales: fused_cpu.row_scales.to_device(&cuda)?,
            bias: None,
            in_features: fused_cpu.in_features,
            num_experts: None,
            rotation_signs: fused_cpu
                .rotation_signs
                .as_ref()
                .map(|t| t.to_device(&cuda))
                .transpose()?,
            rotation_block: fused_cpu.rotation_block,
            mcg_mult: fused_cpu.mcg_mult,
            expert_bpw: None,
            search: fused_cpu.search,
            search_detail: fused_cpu.search_detail,
        };
        let x = Tensor::from_vec(gaussian_fixture(k_in, 31337, 1.0), (1, k_in), &cpu)?
            .to_device(&cuda)?
            .to_dtype(DType::BF16)?;
        let w_dq = fused.dequantize_weights()?.to_dtype(DType::F32)?;
        let y_ref: Vec<f32> = x
            .to_dtype(DType::F32)?
            .matmul(&w_dq.t()?)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;

        // --- gather fixture (e=4, n=32, k=256) + CPU reference -----------
        let (ge, gn, gk) = (4usize, 32usize, 256usize);
        let w3 = build_3d_gaussian_weight(ge, gn, gk)?;
        let gather_cpu = Qtip2bLayer::quantize_with_options_3d(&w3, &cpu, QtipMode::Viterbi, true)?;
        let payload = gather_cpu.serialize()?;
        let (gather_cuda, _) =
            Qtip2bLayer::deserialize_ext_bias(payload, &cuda, QuantizeOntoGuard::new())?;
        let (n_tokens, top_k) = (2usize, 2usize);
        let a_cpu = Tensor::from_vec(
            gaussian_fixture(n_tokens * top_k * gk, 12345, 1.0),
            (n_tokens, top_k, gk),
            &cpu,
        )?;
        let indices_cpu = Tensor::from_vec(vec![0u32, 3, 2, 1], (n_tokens, top_k), &cpu)?;
        let gather_ref: Vec<f32> = gather_cpu
            .gather_forward(&a_cpu, &indices_cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let a_cuda = a_cpu.to_device(&cuda)?.to_dtype(DType::BF16)?;
        let indices_cuda = indices_cpu.to_device(&cuda)?;

        // --- wide fixture: k=4096 -> packed_per_row=1024, which every
        // compiled variant (both generations) can tile. Greedy quantization
        // keeps the CPU bake cheap; the kernel contract is mode-agnostic. ---
        let (wn, wk) = (64usize, 4096usize);
        let ww_cpu = Tensor::from_vec(gaussian_fixture(wn * wk, 4242, 0.5), (wn, wk), &cpu)?;
        let wide_cpu = Qtip2bLayer::quantize_with_options_concrete(
            &ww_cpu,
            None,
            &cpu,
            QtipMode::Greedy,
            true,
        )?;
        let wide = Qtip2bLayer {
            blocks: wide_cpu.blocks.to_device(&cuda)?,
            row_scales: wide_cpu.row_scales.to_device(&cuda)?,
            bias: None,
            in_features: wide_cpu.in_features,
            num_experts: None,
            rotation_signs: wide_cpu
                .rotation_signs
                .as_ref()
                .map(|t| t.to_device(&cuda))
                .transpose()?,
            rotation_block: wide_cpu.rotation_block,
            mcg_mult: wide_cpu.mcg_mult,
            expert_bpw: None,
            search: wide_cpu.search,
            search_detail: wide_cpu.search_detail,
        };
        let xw = Tensor::from_vec(gaussian_fixture(wk, 24680, 1.0), (1, wk), &cpu)?
            .to_device(&cuda)?
            .to_dtype(DType::BF16)?;
        let ww_dq = wide.dequantize_weights()?.to_dtype(DType::F32)?;
        let yw_ref: Vec<f32> = xw
            .to_dtype(DType::F32)?
            .matmul(&ww_dq.t()?)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;

        // --- alignment-fallback fixture: k=100 -> packed_per_row=25 (odd),
        // no vectorized variant applies; the tuned launcher must return -1
        // and the dispatch must produce legacy-kernel results ---------------
        let (fn_, fk) = (16usize, 100usize);
        let wf_cpu = Tensor::from_vec(gaussian_fixture(fn_ * fk, 777, 0.5), (fn_, fk), &cpu)?;
        let fb_cpu = Qtip2bLayer::quantize_with_options_concrete(
            &wf_cpu,
            None,
            &cpu,
            QtipMode::Greedy,
            false,
        )?;
        let fb = Qtip2bLayer {
            blocks: fb_cpu.blocks.to_device(&cuda)?,
            row_scales: fb_cpu.row_scales.to_device(&cuda)?,
            bias: None,
            in_features: fb_cpu.in_features,
            num_experts: None,
            rotation_signs: None,
            rotation_block: 0,
            mcg_mult: fb_cpu.mcg_mult,
            expert_bpw: None,
            search: fb_cpu.search,
            search_detail: fb_cpu.search_detail,
        };
        let xf = Tensor::from_vec(gaussian_fixture(fk, 999, 1.0), (1, fk), &cpu)?
            .to_device(&cuda)?
            .to_dtype(DType::BF16)?;
        let wf_dq = fb.dequantize_weights()?.to_dtype(DType::F32)?;
        let yf_ref: Vec<f32> = xf
            .to_dtype(DType::F32)?
            .matmul(&wf_dq.t()?)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;

        // Legacy sentinel first, then every variant.
        let mut ids: Vec<u32> = vec![QTIP2B_GEMV_VARIANT_LEGACY];
        ids.extend(0..n_variants as u32);
        for v in ids {
            set_forced_gemv_variant(Some(v));

            let y_fused: Vec<f32> = fused
                .forward(&x)?
                .to_dtype(DType::F32)?
                .to_device(&cpu)?
                .flatten_all()?
                .to_vec1()?;
            let c = cos_sim(&y_fused, &y_ref);
            assert!(
                c > 0.995,
                "variant {v}: fused GEMV cos {c} vs dequant+matmul reference"
            );

            let out_cuda: Vec<f32> = gather_cuda
                .gather_forward(&a_cuda, &indices_cuda)?
                .to_dtype(DType::F32)?
                .to_device(&cpu)?
                .flatten_all()?
                .to_vec1()?;
            let c = cos_sim(&gather_ref, &out_cuda);
            assert!(
                c > 0.995,
                "variant {v}: gather GEMV cos {c} vs CPU gather reference"
            );

            if v != QTIP2B_GEMV_VARIANT_LEGACY {
                // The wide fixture must reach every variant, or the loop
                // below would be measuring the legacy fallback instead.
                assert!(
                    gemv_variant_applicable(v, wn, wk),
                    "variant {v}: must be applicable to the wide fixture (n={wn} k={wk}); \
                     otherwise it is never exercised"
                );
                assert!(
                    !gemv_variant_applicable(v, fn_, fk),
                    "variant {v}: expected inapplicable to odd packed_per_row (k={fk})"
                );
            }
            let yw: Vec<f32> = wide
                .forward(&xw)?
                .to_dtype(DType::F32)?
                .to_device(&cpu)?
                .flatten_all()?
                .to_vec1()?;
            let c = cos_sim(&yw, &yw_ref);
            assert!(
                c > 0.995,
                "variant {v}: wide (k={wk}) GEMV cos {c} vs dequant+matmul reference"
            );
            let yf: Vec<f32> = fb
                .forward(&xf)?
                .to_dtype(DType::F32)?
                .to_device(&cpu)?
                .flatten_all()?
                .to_vec1()?;
            let c = cos_sim(&yf, &yf_ref);
            assert!(
                c > 0.995,
                "variant {v}: odd-ppr fallback GEMV cos {c} vs dequant+matmul reference"
            );
        }
        Ok(())
    }

    /// Correctness harness (c), GPU half: the grouped kernel is bit-
    /// identical across runs. The routing scatter uses atomic cursors (the
    /// order WITHIN an expert group is run-dependent), but every output row
    /// is a pure function of its own pair's activations with a fixed
    /// k-summation order, so placement cannot change a single bit.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_grouped_gemm_deterministic() -> Result<()> {
        if !super::super::ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cuda = Device::new_cuda(0)?;
        let cpu = Device::Cpu;
        let (e, n, k_in) = (4usize, 64usize, 128usize);
        let w3 = build_3d_gaussian_weight(e, n, k_in)?;
        let layer_cpu = concrete_3d_layer(&w3, &cpu, QtipMode::Viterbi, true)?;
        let payload = layer_cpu.serialize()?;
        let (layer_cuda, _) =
            Qtip2bLayer::deserialize_concrete(payload, &cuda, QuantizeOntoGuard::new())?;

        // Heavy collisions on one expert so the atomic scatter order truly
        // varies between runs.
        let mut idx = vec![1u32; 50];
        idx.extend([0, 3, 3, 2, 1, 0]);
        let p = idx.len();
        let a_data = gaussian_fixture(p * k_in, 31415, 1.0);
        let a_cuda = Tensor::from_vec(a_data, (p, k_in), &cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;
        let indices_cuda = Tensor::from_vec(idx, (p,), &cpu)?.to_device(&cuda)?;

        let run = || -> Result<Vec<u32>> {
            Ok(layer_cuda
                .gather_forward_batched(&a_cuda, &indices_cuda)?
                .to_dtype(DType::F32)?
                .to_device(&cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        let y1 = run()?;
        let y2 = run()?;
        assert_eq!(y1, y2, "grouped GEMM output changed between runs");
        Ok(())
    }

    /// wave6-Q regression guard: a 3-D expert-stack quantize whose target
    /// device is the CPU (the bake path) must stream through the GPU
    /// kernels on a CUDA box — a reroute to the CPU Viterbi is a ~20x
    /// per-layer bake regression and must be counted, never silent.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_3d_expert_quantize_does_not_fall_back_to_cpu() -> Result<()> {
        if !super::super::ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        if Device::new_cuda(0).is_err() {
            // No physical GPU (e.g. a compile-gate CI lane running tests):
            // the CPU fallback is correct behavior there, nothing to assert.
            return Ok(());
        }
        let before = crate::gpu_quantize_cpu_fallback_count();
        let (e, n, k_in) = (4usize, 8usize, 256usize);
        let w3 = build_3d_gaussian_weight(e, n, k_in)?;
        let layer =
            Qtip2bLayer::quantize_with_options_3d(&w3, &Device::Cpu, QtipMode::Viterbi, true)?;
        // Quantize really ran: the packed stack dequantizes at full shape.
        assert_eq!(layer.dequantize_w()?.dims(), &[e, n, k_in]);
        let after = crate::gpu_quantize_cpu_fallback_count();
        assert_eq!(
            after,
            before,
            "Qtip2bLayer 3-D expert quantize fell back to the CPU pipeline on a CUDA box \
             ({} new fallback(s)) — check the warn log for the reason",
            after - before
        );
        Ok(())
    }
    // =======================================================================
    // Beam search (wave46-BX) — the K=2 / V=1 port of the LUT rung's beam.
    //
    // 🔴 THIS IS A DECLARED QUALITY TRADE (DOCTRINE D4). Exhaustive search is
    // the best this rung can do and stays the default; the beam exists because
    // wave41-BS measured the exhaustive K=2 bake at >=984 s for layer 0 alone,
    // projecting ~11.75 h / ~$57 for 43 layers. These tests exist to prove the
    // beam is a *legitimate, recorded, slightly worse* search — never a silent
    // one, and never greedy.
    // =======================================================================

    /// **The load-bearing proof.** A beam wide enough to prune nothing must
    /// reproduce [`viterbi_quantize_row_2b`] — the exhaustive 2^L dynamic
    /// program — symbol for symbol. Anything less and "beam" would just mean
    /// "a different algorithm that happens to look similar".
    ///
    /// This is what makes the K=2 successor geometry checkable: 4 successors
    /// per group instead of 16, groups keyed by the low `L-K = 14` bits instead
    /// of the low 12, and a dedup/merge rule that has to reproduce the
    /// exhaustive DP's "lowest predecessor state wins a cost tie" exactly.
    ///
    /// D12 (fixtures lie): a pure Gaussian and the realistic `fp4_dequant`
    /// chain are not enough on their own — mutation testing showed that
    /// flipping the dedup tie-break from `<` to `<=` SURVIVES on both, because
    /// neither produces exact equal-cost predecessor pairs often enough. The
    /// third fixture below fixes that by degenerating the *codebook* rather
    /// than the data: with only four distinct codeword values, equal-cost paths
    /// are everywhere, and the two searches can only agree if they resolve ties
    /// identically ("lowest predecessor state wins").
    #[test]
    fn beam_2b_unpruned_matches_exhaustive_bit_for_bit() {
        use super::super::bake_quality_tests::gen_fp4_dequant;
        let real = computed_codebook(QTIP2B_MCG_MULT);
        // A CONSTANT codebook makes every symbol stream cost exactly the same,
        // so the emitted stream is decided *entirely* by the tie-break rule —
        // "lowest predecessor state wins", in both searches. It is a synthetic
        // codebook precisely because a realistic one cannot be relied on to
        // produce exact f32 ties often enough (measured: `<` -> `<=` survives
        // on gaussian and on fp4_dequant).
        let all_ties: Vec<f32> = vec![0.25f32; CB_SIZE_2B];
        let rows = 4usize;
        let k = 64usize;

        let fixtures: [(&str, Vec<f32>, &Vec<f32>); 3] = [
            ("gaussian", gaussian_fixture(rows * k, 0xBEA3, 1.0), &real),
            ("fp4_dequant", gen_fp4_dequant(rows, k, 1.0, 0xBEA4), &real),
            (
                "constant-codebook(all ties)",
                gaussian_fixture(rows * k, 0xBEA2, 1.0),
                &all_ties,
            ),
        ];

        for (name, data, codebook) in fixtures {
            for row in 0..rows {
                let target = &data[row * k..(row + 1) * k];
                let exhaustive = viterbi_quantize_row_2b(target, codebook);
                let unpruned = beam_quantize_row_2b(target, codebook, CB_SIZE_2B);
                assert_eq!(
                    unpruned, exhaustive,
                    "{name} row {row}: an unpruned beam is not the exhaustive DP — \
                     the K=2 beam is a different search, not a pruned one"
                );
                // `quantize_row_2b` must route a prune-nothing width to the
                // exhaustive implementation, not to the beam.
                assert_eq!(
                    quantize_row_2b(
                        target,
                        codebook,
                        TrellisSearch::Beam {
                            width: CB_SIZE_2B * 2
                        }
                    ),
                    exhaustive,
                    "{name} row {row}: width >= 2^L must dispatch to the exhaustive DP"
                );
            }
        }
    }

    /// D4b on the CPU dispatcher: `quantize_row_2b` must run the width it was
    /// handed, not a width it finds convenient.
    ///
    /// The specific failure this exists for: a beam width above the CUDA
    /// kernel's 256-slot limit being quietly clamped on the CPU path too, so a
    /// CPU bake and a GPU bake of `ARC_QTIP_BEAM=1024` would silently be
    /// different searches (and the GPU one would have errored). Non-vacuity is
    /// the `assert_ne!`: the two widths must actually produce different bytes,
    /// or "the width was honoured" would be unfalsifiable.
    #[test]
    fn quantize_row_2b_never_substitutes_a_beam_width() {
        let codebook = computed_codebook(QTIP2B_MCG_MULT);
        let target = gaussian_fixture(512, 0xBEAE, 0.8);
        for width in [64usize, 256, 1024, 4096] {
            assert_eq!(
                quantize_row_2b(&target, &codebook, TrellisSearch::Beam { width }),
                beam_quantize_row_2b(&target, &codebook, width),
                "W={width}: the dispatcher ran a different width than it was asked for"
            );
        }
        assert_ne!(
            beam_quantize_row_2b(&target, &codebook, 1024),
            beam_quantize_row_2b(&target, &codebook, 256),
            "W=1024 and W=256 produced identical bytes — this row does not \
             discriminate the width, so the assertions above prove nothing"
        );
    }

    /// Non-vacuity for the test above, plus the quality direction.
    ///
    /// At production widths the beam MUST actually prune (otherwise the
    /// unpruned test proves nothing about the pruned kernel), and because
    /// pruning can only discard paths, the beam's reconstruction error must be
    /// at least the exhaustive DP's. Both halves are asserted: a beam that came
    /// out *better* would mean the exhaustive DP is not optimal, i.e. a bug in
    /// the thing we currently ship.
    #[test]
    fn beam_2b_prunes_and_never_beats_exhaustive() {
        use super::super::bake_quality_tests::gen_fp4_dequant;
        let codebook = computed_codebook(QTIP2B_MCG_MULT);
        let rows = 4usize;
        let k = 256usize;
        let data = gen_fp4_dequant(rows, k, 1.0, 0xBEA5);

        let mut any_differed = false;
        for width in [16usize, 64, 256] {
            for row in 0..rows {
                let target = &data[row * k..(row + 1) * k];
                let exhaustive = viterbi_quantize_row_2b(target, &codebook);
                let beam = beam_quantize_row_2b(target, &codebook, width);
                if beam != exhaustive {
                    any_differed = true;
                }
                let e_mse = mse(target, &decode_symbols_2b(&exhaustive, &codebook));
                let b_mse = mse(target, &decode_symbols_2b(&beam, &codebook));
                assert!(
                    b_mse >= e_mse - 1e-6,
                    "W={width} row {row}: beam MSE {b_mse} beat the exhaustive DP's {e_mse} — \
                     the exhaustive search is supposed to be optimal"
                );
            }
        }
        assert!(
            any_differed,
            "the beam reproduced the exhaustive DP at every width on a 256-symbol row — \
             this fixture does not exercise pruning, so bit-identity elsewhere proves nothing"
        );
    }

    /// Same input, same bytes, twice — the beam's selection and its group
    /// dedup are both order-independent by construction and must stay so.
    #[test]
    fn beam_2b_is_deterministic() {
        let codebook = computed_codebook(QTIP2B_MCG_MULT);
        let target = gaussian_fixture(512, 0xBEA6, 0.7);
        for width in [8usize, 64, 256] {
            let a = beam_quantize_row_2b(&target, &codebook, width);
            let b = beam_quantize_row_2b(&target, &codebook, width);
            assert_eq!(a, b, "W={width}: beam is not deterministic");
            assert_eq!(a.len(), target.len());
            assert!(a.iter().all(|&s| (s as usize) < ALPHABET_2B));
        }
    }

    /// 🔴 **The declared cost of the trade, measured on the realistic
    /// distribution** (D4 + D12).
    ///
    /// wave19-AP measured exhaustive beating beam W=256 on 8 of 9 cells for the
    /// LUT rung. This is the same measurement for the K=2 rung, on
    /// `fp4_dequant` (heavy-tailed Student-t(4) snapped to the FP4 lattice with
    /// per-32 block scales — V4's actual source chain) rather than on a pure
    /// Gaussian, which hides exactly this class of effect.
    ///
    /// The assertion is deliberately a *ceiling on the loss*, not a claim of
    /// equality: the beam is allowed to be worse, and is expected to be. The
    /// printed numbers are the deliverable — they are what a W recommendation
    /// has to be argued from.
    #[test]
    fn beam_2b_quality_delta_vs_exhaustive_is_bounded_and_reported() {
        use super::super::bake_quality_tests::gen_fp4_dequant;
        let codebook = computed_codebook(QTIP2B_MCG_MULT);
        let rows = 8usize;
        let k = 1024usize;

        // The trellis searches in the ROTATED frame in production, and the
        // rotation is what makes a heavy-tailed row look Gaussian — so a
        // measurement taken only on the raw rows would overstate the beam's
        // difficulty. Both frames are reported.
        let signs = generate_signs(QTIP_ROTATION_SEED, k);
        let block = rotation_block_size(k);
        let fp4 = gen_fp4_dequant(rows, k, 0.02, 0xBEA7);
        let fp4_rot = {
            let mut out = fp4.clone();
            for row in 0..rows {
                apply_block_rotation(&mut out[row * k..(row + 1) * k], &signs, block);
            }
            out
        };

        for (name, data) in [
            ("fp4_dequant", fp4),
            ("fp4_dequant+hadamard", fp4_rot),
            (
                "gaussian(control)",
                gaussian_fixture(rows * k, 0xBEA8, 0.02),
            ),
        ] {
            for width in [64usize, 128, 256] {
                let mut cos_ex = 0f64;
                let mut cos_bm = 0f64;
                let mut nmse_ex = 0f64;
                let mut nmse_bm = 0f64;
                for row in 0..rows {
                    let raw = &data[row * k..(row + 1) * k];
                    let max_abs = raw.iter().fold(0f32, |m, &v| m.max(v.abs()));
                    let scale = if max_abs == 0.0 {
                        1.0
                    } else {
                        max_abs / QTIP2B_SCALE_DIVISOR
                    };
                    let target: Vec<f32> = raw.iter().map(|w| w / scale).collect();

                    let ex =
                        decode_symbols_2b(&viterbi_quantize_row_2b(&target, &codebook), &codebook);
                    let bm = decode_symbols_2b(
                        &beam_quantize_row_2b(&target, &codebook, width),
                        &codebook,
                    );
                    cos_ex += cos_sim(&target, &ex) as f64;
                    cos_bm += cos_sim(&target, &bm) as f64;
                    let den: f64 = target.iter().map(|&t| (t as f64) * (t as f64)).sum();
                    nmse_ex += mse(&target, &ex) as f64 * k as f64 / den;
                    nmse_bm += mse(&target, &bm) as f64 * k as f64 / den;
                }
                let (cos_ex, cos_bm) = (cos_ex / rows as f64, cos_bm / rows as f64);
                let (nmse_ex, nmse_bm) = (nmse_ex / rows as f64, nmse_bm / rows as f64);
                println!(
                    "[qtip2b search quality] {name:18} W={width:<4} \
                     cos exhaustive={cos_ex:.6} beam={cos_bm:.6} (Δ={:+.6})  \
                     nmse exhaustive={nmse_ex:.6} beam={nmse_bm:.6} (Δ={:+.6})",
                    cos_bm - cos_ex,
                    nmse_bm - nmse_ex
                );
                // The beam may be worse. It may not be *broken*: a 0.01 cosine
                // cliff would mean the pruning is discarding the right path
                // wholesale rather than occasionally.
                assert!(
                    cos_ex - cos_bm < 0.01,
                    "{name} W={width}: beam lost {:.6} cosine to the exhaustive DP — \
                     that is a pruning failure, not a quality trade",
                    cos_ex - cos_bm
                );
            }
        }
    }

    /// The same trade measured on the **whole production pipeline** — block
    /// Hadamard rotation, `max|row|/3.62` search scale, trellis search, then
    /// least-squares scale refinement — rather than on the search in isolation.
    /// This is the number that describes the artifact an operator would ship,
    /// and the one a `W` recommendation has to be argued from.
    #[test]
    fn beam_2b_layer_quality_delta_vs_exhaustive_is_reported() -> Result<()> {
        use super::super::bake_quality_tests::gen_fp4_dequant;
        let device = Device::Cpu;
        let (n, k) = (8usize, 1024usize);
        let raw = gen_fp4_dequant(n, k, 0.02, 0xBEB1);
        let w = Tensor::from_vec(raw.clone(), (n, k), &device)?;

        let quantized = |search: TrellisSearch| -> Result<Vec<f32>> {
            let layer = Qtip2bLayer::quantize_with_options_concrete_search(
                &w,
                None,
                &device,
                QtipMode::Viterbi,
                true,
                search,
            )?;
            layer
                .dequantize_weights()?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()
        };

        let ex = quantized(TrellisSearch::Exhaustive)?;
        let cos_ex = cos_sim(&raw, &ex);
        for width in [64usize, 128, 256] {
            let bm = quantized(TrellisSearch::Beam { width })?;
            let cos_bm = cos_sim(&raw, &bm);
            println!(
                "[qtip2b layer quality] fp4_dequant n={n} k={k} W={width:<4} \
                 weight-cos exhaustive={cos_ex:.6} beam={cos_bm:.6} (Δ={:+.6})",
                cos_bm - cos_ex
            );
            assert!(
                cos_ex - cos_bm < 0.01,
                "W={width}: the beam lost {:.6} weight cosine on the full pipeline",
                cos_ex - cos_bm
            );
        }
        Ok(())
    }

    /// A beam bake must SAY it was a beam bake, at the exact width that ran,
    /// and the claim must survive a UQFF round trip (DOCTRINE D4 §3).
    #[test]
    fn beam_2b_stamps_its_width_into_the_artifact() -> Result<()> {
        let device = Device::Cpu;
        let (n, k) = (8usize, 128usize);
        let w = Tensor::from_vec(gaussian_fixture(n * k, 0xBEA9, 0.4), (n, k), &device)?;

        let exhaustive = Qtip2bLayer::quantize_with_options_concrete_search(
            &w,
            None,
            &device,
            QtipMode::Viterbi,
            true,
            TrellisSearch::Exhaustive,
        )?;
        assert_eq!(exhaustive.search_detail(), QtipSearchDetail::EXHAUSTIVE_MSE);

        for width in [64usize, 256] {
            let beam = Qtip2bLayer::quantize_with_options_concrete_search(
                &w,
                None,
                &device,
                QtipMode::Viterbi,
                true,
                TrellisSearch::Beam { width },
            )?;
            assert_eq!(
                beam.search_detail(),
                QtipSearchDetail::Known {
                    beam_width: Some(width as u16),
                    hessian: false,
                },
                "a W={width} bake did not record W={width}"
            );
            // Non-vacuity: the recorded width describes a genuinely different
            // artifact, not a label stuck on identical bytes.
            let a: Vec<u8> = beam.blocks.flatten_all()?.to_vec1()?;
            let b: Vec<u8> = exhaustive.blocks.flatten_all()?.to_vec1()?;
            assert_ne!(
                a, b,
                "W={width} produced byte-identical blocks to the exhaustive DP — \
                 the search-detail stamp would be describing nothing"
            );

            // Round trip: the claim is on the wire, not just in memory.
            let payload = beam.serialize()?;
            let (back, _) =
                Qtip2bLayer::deserialize_concrete(payload, &device, QuantizeOntoGuard::new())?;
            assert_eq!(
                back.search_detail(),
                QtipSearchDetail::Known {
                    beam_width: Some(width as u16),
                    hessian: false,
                },
                "W={width}: the beam width did not survive UQFF serialization"
            );
        }
        Ok(())
    }

    /// The **3-D expert-stack** path, which is the one a V4 bake actually
    /// takes — all 11.75 h of the wave41-BS projection is MoE expert stacks.
    ///
    /// The stack is quantized in chunks of experts; every chunk must be handed
    /// the SAME search, so a stack can never come out half beam and half
    /// exhaustive, and the stamp on the assembled layer must be the width that
    /// ran on all of them.
    #[test]
    fn beam_2b_stamps_its_width_on_a_3d_expert_stack() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k) = (3usize, 4usize, 128usize);
        let w3 = Tensor::from_vec(gaussian_fixture(e * n * k, 0xBEB2, 0.4), (e, n, k), &device)?;

        let exhaustive = Qtip2bLayer::quantize_with_options_3d_search(
            &w3,
            &device,
            QtipMode::Viterbi,
            true,
            TrellisSearch::Exhaustive,
        )?;
        let ex_w: Vec<f32> = exhaustive
            .dequantize_w()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        let beam = Qtip2bLayer::quantize_with_options_3d_search(
            &w3,
            &device,
            QtipMode::Viterbi,
            true,
            TrellisSearch::Beam { width: 64 },
        )?;
        let bm_w: Vec<f32> = beam
            .dequantize_w()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        assert_ne!(
            ex_w.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            bm_w.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the 3-D beam path produced the exhaustive artifact — the search \
             never reached the per-expert chunks"
        );

        // `quantize_with_options_3d_search` hands back a `dyn QuantMethod`, so
        // read the stamp off the wire, which is what a loader would see anyway.
        let payload = beam.serialize()?;
        let (back, _) =
            Qtip2bLayer::deserialize_concrete(payload, &device, QuantizeOntoGuard::new())?;
        assert_eq!(back.num_experts, Some(e));
        assert_eq!(
            back.search_detail(),
            QtipSearchDetail::Known {
                beam_width: Some(64),
                hessian: false,
            },
            "a 3-D expert-stack beam bake did not record the width it ran"
        );
        Ok(())
    }

    /// D4b: the GPU dispatch may TRANSLATE a search ("a beam at least as wide
    /// as the state space prunes nothing, so run the exhaustive kernel") but it
    /// may never SUBSTITUTE one. A width `qtip2b_beam.cu` cannot run is an
    /// error, never a quietly narrower beam.
    ///
    /// The 2b kernel's `QB2_MAX_BEAM` is its block size, 256 — the same as the
    /// LUT rung's, so an operator's `ARC_QTIP_BEAM` means the same thing on
    /// both rungs. On a CUDA build the constant is read back from the kernel so
    /// this test cannot drift from the compiled limit.
    #[test]
    fn qtip2b_cuda_search_plan_never_substitutes_a_width() {
        use super::super::cuda_search_plan;
        const MAX_W: usize = 256;

        #[cfg(feature = "cuda")]
        if super::super::ffi::HAVE_QTIP_KERNELS {
            assert_eq!(
                super::super::cuda_ops::beam_2b_max_width(),
                MAX_W,
                "the compiled qtip2b beam kernel's max width drifted from this test"
            );
        }

        assert_eq!(
            cuda_search_plan(TrellisSearch::Exhaustive, MAX_W).unwrap(),
            TrellisSearch::Exhaustive
        );
        for w in [1usize, 16, 64, 128, 256] {
            assert_eq!(
                cuda_search_plan(TrellisSearch::Beam { width: w }, MAX_W).unwrap(),
                TrellisSearch::Beam { width: w },
                "width {w} must be honoured exactly"
            );
        }
        for w in [CB_SIZE_2B, CB_SIZE_2B + 1, usize::MAX] {
            assert_eq!(
                cuda_search_plan(TrellisSearch::Beam { width: w }, MAX_W).unwrap(),
                TrellisSearch::Exhaustive
            );
        }
        for w in [MAX_W + 1, 1024, CB_SIZE_2B - 1] {
            let err = cuda_search_plan(TrellisSearch::Beam { width: w }, MAX_W)
                .expect_err("a width beyond the kernel limit must not be silently narrowed");
            assert!(format!("{err}").contains("will not silently substitute"));
        }
        assert!(cuda_search_plan(TrellisSearch::Beam { width: 64 }, 0).is_err());
    }

    /// 🔴 D4 IS ABSOLUTE: growing a beam does NOT open a door for greedy.
    ///
    /// The production entry point must still refuse [`QtipMode::Greedy`] in
    /// every build and at every search setting, and a greedy stamp must still
    /// be unable to carry a beam width (a greedy walk runs no trellis search,
    /// so an artifact claiming both is self-contradictory and is refused on
    /// both the write and the read side).
    #[test]
    fn beam_does_not_make_greedy_reachable() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::from_vec(gaussian_fixture(4 * 64, 0xBEAA, 0.5), (4, 64), &device)?;

        let err = Qtip2bLayer::quantize_with_mode(&w, None, &device, QtipMode::Greedy)
            .expect_err("the production door must refuse greedy");
        assert!(format!("{err}").to_lowercase().contains("greedy"), "{err}");

        // The fixture door (open only inside this crate's own `cfg(test)`
        // build) must not become a way to launder a greedy bake as a beam bake:
        // even with a beam explicitly requested, a greedy layer records no
        // width at all.
        let greedy_fixture = Qtip2bLayer::quantize_with_options_concrete_search(
            &w,
            None,
            &device,
            QtipMode::Greedy,
            QtipRotation::for_mode(QtipMode::Greedy).enabled(),
            TrellisSearch::Beam { width: 256 },
        )?;
        assert_eq!(
            greedy_fixture.search_detail(),
            QtipSearchDetail::EXHAUSTIVE_MSE,
            "a greedy fixture recorded a beam width"
        );
        assert_eq!(greedy_fixture.search, QtipSearchStamp::Greedy);
        // ...and it is still refused at load, beam or no beam.
        let payload = greedy_fixture.serialize()?;
        assert!(
            Qtip2bLayer::deserialize_concrete(payload, &device, QuantizeOntoGuard::new()).is_err(),
            "a greedy-stamped qtip2b artifact must not load"
        );

        assert_eq!(
            QtipSearchDetail::for_bake(QtipMode::Greedy, TrellisSearch::Beam { width: 256 }, true),
            QtipSearchDetail::EXHAUSTIVE_MSE,
            "a greedy bake must never record a beam width or a weighted objective"
        );
        Ok(())
    }

    /// The CUDA beam kernel must emit the **byte-identical** symbol stream the
    /// CPU beam emits at the same width — not a similar one. Cosine similarity
    /// would hide exactly the failure mode that matters: a GPU bake and a CPU
    /// bake of the same weights with the same flag silently producing different
    /// checkpoints.
    ///
    /// Non-vacuity: the same fixture is also baked with the exhaustive kernel
    /// and asserted to DIFFER, so the test cannot pass by the beam happening to
    /// reproduce the full DP.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_beam_2b_matches_cpu_beam_bit_for_bit() -> Result<()> {
        use super::super::cuda_ops;
        if !super::super::ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        let cpu = Device::Cpu;
        let n = 8;
        let k_in = 512; // 512 symbols at V=1: long enough to prune every step
        let wdata = gaussian_fixture(n * k_in, 0xBEAB, 0.5);
        let w_cuda = Tensor::from_vec(wdata.clone(), (n, k_in), &cuda)?;

        // Both sides must see the SAME per-row scale, so take the kernel's.
        let scales: Vec<f32> = cuda_ops::compute_row_scales_2b_cuda(&w_cuda)?
            .to_device(&cpu)?
            .to_vec1()?;
        let codebook = computed_codebook(QTIP2B_MCG_MULT);

        let cpu_reference = |search: TrellisSearch| -> Vec<u8> {
            let mut out = Vec::with_capacity(n * (k_in / SYMS_PER_BYTE));
            for row in 0..n {
                let inv_scale = 1.0f32 / scales[row];
                let target: Vec<f32> = wdata[row * k_in..(row + 1) * k_in]
                    .iter()
                    .map(|w| w * inv_scale)
                    .collect();
                out.extend_from_slice(&pack_symbols_2b(&quantize_row_2b(
                    &target, &codebook, search,
                )));
            }
            out
        };

        let exhaustive: Vec<u8> = cuda_ops::quantize_rows_2b_cuda(
            &w_cuda,
            QTIP2B_MCG_MULT,
            QtipMode::Viterbi,
            TrellisSearch::Exhaustive,
        )?
        .0
        .to_device(&cpu)?
        .flatten_all()?
        .to_vec1()?;

        let mut any_differed = false;
        for width in [64usize, 128, 256] {
            let search = TrellisSearch::Beam { width };
            let gpu: Vec<u8> = cuda_ops::quantize_rows_2b_cuda(
                &w_cuda,
                QTIP2B_MCG_MULT,
                QtipMode::Viterbi,
                search,
            )?
            .0
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;
            let reference = cpu_reference(search);
            assert_eq!(gpu.len(), reference.len());
            let mismatches = gpu
                .iter()
                .zip(reference.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                mismatches,
                0,
                "W={width}: CUDA beam differs from the CPU beam in {mismatches}/{} bytes — \
                 the GPU and CPU bakes of the same weights are not the same checkpoint",
                gpu.len()
            );
            if gpu != exhaustive {
                any_differed = true;
            }
        }
        assert!(
            any_differed,
            "beam and exhaustive produced identical bytes at every width — the fixture \
             does not actually exercise pruning, so bit-identity proves nothing"
        );
        Ok(())
    }

    /// **The load-bearing GPU proof**: a beam wide enough to prune nothing must
    /// reproduce the exhaustive kernel byte for byte.
    ///
    /// `num_symbols = 4` makes that provable rather than incidental. From the
    /// implicit start state 0 the reachable set at K=2 is 4 states at t=0, 16 at
    /// t=1, 64 at t=2 and 256 at t=3 — so at W=256 the beam never drops a
    /// candidate through the whole row, and the exhaustive DP's finite-cost set
    /// is exactly the same 256 states. (The K=2 geometry lets this run 4
    /// timesteps deep where the K=4 rung's twin only reaches 2.)
    ///
    /// Longer rows are covered transitively: `cuda_beam_2b_matches_cpu_beam_
    /// bit_for_bit` pins CUDA to the CPU beam, and
    /// `beam_2b_unpruned_matches_exhaustive_bit_for_bit` pins the unpruned CPU
    /// beam to the CPU exhaustive DP on 64-symbol rows.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_beam_2b_unpruned_matches_exhaustive() -> Result<()> {
        use super::super::cuda_ops;
        if !super::super::ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        let cpu = Device::Cpu;
        let n = 64;
        let k_in = 4; // 4 symbols -> the beam provably prunes nothing at W=256
        let w_cuda = Tensor::from_vec(gaussian_fixture(n * k_in, 0xBEAC, 0.9), (n, k_in), &cuda)?;

        let exhaustive: Vec<u8> = cuda_ops::quantize_rows_2b_cuda(
            &w_cuda,
            QTIP2B_MCG_MULT,
            QtipMode::Viterbi,
            TrellisSearch::Exhaustive,
        )?
        .0
        .to_device(&cpu)?
        .flatten_all()?
        .to_vec1()?;
        let unpruned: Vec<u8> = cuda_ops::quantize_rows_2b_cuda(
            &w_cuda,
            QTIP2B_MCG_MULT,
            QtipMode::Viterbi,
            TrellisSearch::Beam { width: 256 },
        )?
        .0
        .to_device(&cpu)?
        .flatten_all()?
        .to_vec1()?;

        assert_eq!(
            unpruned, exhaustive,
            "an unpruned CUDA beam must be the exhaustive DP, byte for byte"
        );
        Ok(())
    }

    /// wave46-BX also put the exhaustive K=2 kernel onto `qtip_exact_fp.cuh`
    /// (it was contracting `d*d + cost` into an FMA and computing `1.0f/scale`
    /// with a reciprocal approximation under `--use_fast_math`). With those
    /// gone the exhaustive kernel is bit-identical to the CPU DP — which is
    /// what makes the unpruned-beam guard above meaningful, and what makes a
    /// CPU-baked and a GPU-baked qtip2b checkpoint the same artifact.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_exhaustive_2b_matches_cpu_exhaustive_bit_for_bit() -> Result<()> {
        use super::super::cuda_ops;
        if !super::super::ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        let cpu = Device::Cpu;
        let n = 8;
        let k_in = 256;
        let wdata = gaussian_fixture(n * k_in, 0xBEAD, 0.6);
        let w_cuda = Tensor::from_vec(wdata.clone(), (n, k_in), &cuda)?;

        let scales: Vec<f32> = cuda_ops::compute_row_scales_2b_cuda(&w_cuda)?
            .to_device(&cpu)?
            .to_vec1()?;
        let codebook = computed_codebook(QTIP2B_MCG_MULT);

        let gpu: Vec<u8> = cuda_ops::quantize_rows_2b_cuda(
            &w_cuda,
            QTIP2B_MCG_MULT,
            QtipMode::Viterbi,
            TrellisSearch::Exhaustive,
        )?
        .0
        .to_device(&cpu)?
        .flatten_all()?
        .to_vec1()?;

        let mut reference = Vec::with_capacity(gpu.len());
        for row in 0..n {
            let inv_scale = 1.0f32 / scales[row];
            let target: Vec<f32> = wdata[row * k_in..(row + 1) * k_in]
                .iter()
                .map(|w| w * inv_scale)
                .collect();
            reference.extend_from_slice(&pack_symbols_2b(&viterbi_quantize_row_2b(
                &target, &codebook,
            )));
        }
        let mismatches = gpu
            .iter()
            .zip(reference.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches,
            0,
            "the qtip2b exhaustive CUDA kernel differs from the CPU DP in {mismatches}/{} bytes",
            gpu.len()
        );
        Ok(())
    }
}
