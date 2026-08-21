//! Parent system: ArcQuant / ArcKernels
//!
//! TCFRAG-2B — the tensor-core fragment byte order for the **K=2 / V=1 / L=16**
//! (`qtip2b`) trellis rung.
//!
//! # ⚠️ UNVERIFIED ON HARDWARE — never run
//!
//! Nothing in this module, and nothing in `kernels/qtip/qtip2b_tcfrag.cu`, has
//! executed on a GPU. Every performance number below is a **derivation from
//! instruction counts**, not a measurement. The CPU property tests in this file
//! DO run and DO gate the layout; the kernel does not.
//!
//! # Why this exists
//!
//! `qtip2b_gemv_tuned_kernel` (`kernels/qtip/qtip_bitshift_tune.cu`) is the hot
//! trellis path: **69 ms of a 794 ms B=256 step**, and the whole of the b=1
//! decode path, which `gather_policy::lut_fused_gather_preferred` routes to
//! unconditionally for `n_tokens <= 8`.
//!
//! It is not inefficient — it is running flat out on the **wrong roofline**.
//! With the decode stubbed out it reaches ~90% of the H200 *scalar* FMA
//! roofline (60.5 TFLOP/s), while the dense fp16 tensor cores on the same die
//! do 989 TFLOP/s. The kernel contains **two tensor-core instructions in
//! total** despite its comments describing a "persistent tensor-core tile
//! loop". Even a *free* decode buys at most 1.89x on the kernel as structured.
//!
//! # The identity that lets the tensor core do the work
//!
//! The `qtip2b` codeword is, by definition (`bitshift.rs::mcg_codeword`),
//!
//! ```text
//!     x = state * mult                        // one IMAD
//!     m = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60     // one LOP3
//!     w = f32(fp16(m >> 16)) + f32(fp16(m & 0xFFFF))
//! ```
//!
//! `m` is *already* a packed pair of fp16 numbers whose **sum** is the weight.
//! A dot product is linear, so
//!
//! ```text
//!     sum_s (hi_s + lo_s) * x_s  ==  sum_s hi_s * x_s  +  sum_s lo_s * x_s
//! ```
//!
//! Give the two halves of `m` **adjacent k-slots** of an `mma.m16n8k16` A
//! fragment and feed both slots the **same** activation `x_s`, and the tensor
//! core performs the fp16->f32 fold, the k-reduction and the accumulate. The
//! per-row scale leaves the inner loop entirely and becomes one multiply in the
//! epilogue.
//!
//! The LOP3 destination register **is** the mma operand register. Nothing goes
//! between them: no `MOV`, no `PRMT`, no `SHFL`. The K=4/V=2 sibling of this
//! layout measured exactly that on an H200 — 256 of 256 A-operand registers
//! produced directly by `LOP3.LUT`, cosine 1.000000000.
//!
//! ## V=1 costs one instruction per weight more than V=2
//!
//! Stated plainly because it bounds the win. At K=4/V=2 one state yields *two*
//! codewords (the chained `x1 = x0 * mult`), so the state derivation amortises
//! over two weights and the sibling measured **3.56 inst/weight**. At K=2/V=1
//! one state yields *one* codeword, so the four decode instructions
//! (`SHF`, `LOP3` mask-to-16-bits, `IMAD`, `LOP3` mask/xor) are paid **per
//! weight**. See [`derivation`] for the full arithmetic.
//!
//! # The byte layout
//!
//! A row of `S` symbols (2 bits each) becomes `W = ceil(S/16)` words of 32
//! bits. Word `j` holds symbols `16j .. 16j+15` in **reversed 2-bit-field
//! order**:
//!
//! ```text
//!     R[j] bits [2c .. 2c+1]  =  sym[16j + (15 - c)]     for c = 0..15
//! ```
//!
//! so `(R[j-1] << 32) | R[j]` is a continuous **descending**-symbol 2-bit
//! stream. That is the whole trick. The shipped format stores symbols
//! ascending-LSB-first while the trellis recurrence
//! `state = ((state << 2) | sym) & 0xFFFF` reads its 8-symbol window in the
//! opposite order, which is why every shipped decode path pays either a
//! sequential warm-up replay (`qtip_bitshift.cu`, 8 symbols replayed per group)
//! or a `__brev` pair-reversal (`qtip2b_common.cuh::q2b_state_from_window`,
//! ~4 ops). Under TCFRAG-2B the state at **any** symbol is
//!
//! ```text
//!     state(s) = funnelshift_r(R[j], R[j-1], 2 * (15 - (s & 15))) & 0xFFFF
//!                with j = s >> 4  and  R[-1] = 0
//! ```
//!
//! — one `SHF`, one `LOP3`, random access, no replay and no reversal.
//! [`tcfrag2b_state`] is that expression and
//! `tcfrag2b_state_matches_sequential_recurrence` is how a reader validates a
//! candidate repack.
//!
//! # 🔴 D22 — this is a LOAD-TIME permutation, never a baked one
//!
//! TCFRAG-2B reorders stored bits. It does not touch the symbol stream, the
//! trellis, the codebook or the scales, so the decoded weights are
//! **bit-identical** ([`tcfrag2b_decodes_identical_weights`] pins that) and an
//! existing artifact converts at load time via [`tcfrag2b_repack`] — or, on
//! GPU, `launch_qtip2b_tcfrag_repack`.
//!
//! It is **not** baked into the on-disk byte order, by Jish's direct decision
//! (`KERNEL_RULES.md` D22). The k-slot assignment below is specific to
//! `mma.m16n8k16`; Blackwell's `tcgen05` shapes differ. Baking it would mean
//! one artifact per GPU generation, which fragments the byte-format moat (D17)
//! that is the entire reason the GEMM is ours. The cost of doing it at load
//! time is one bandwidth-bound pass — 74 GB read + 74 GB write, ~0.03 s
//! against a ~65 s model load.
//!
//! **The permutation is therefore DATA selected by compute capability**
//! ([`Tcfrag2bLayout::for_compute_cap`]), passed across the CUDA ABI as an
//! integer id, never `#if __CUDA_ARCH__` around the layout. A compute
//! capability with no entry in that table gets `None` and the caller keeps the
//! shipped kernel — it never guesses a layout.
//!
//! # The k-slot assignment
//!
//! For `mma.m16n8k16` the A fragment of lane `t` (`gid = t >> 2`,
//! `i = t & 3`) covers, in its four 32-bit registers,
//!
//! ```text
//!     a[0] <- (row gid  , k 2i  , 2i+1)      a[2] <- (row gid  , k 2i+8, 2i+9)
//!     a[1] <- (row gid+8, k 2i  , 2i+1)      a[3] <- (row gid+8, k 2i+8, 2i+9)
//! ```
//!
//! TCFRAG-2B assigns, within one 8-symbol mma step starting at `s0`,
//!
//! ```text
//!     symbol s0 + p   ->   k-slots 2p, 2p+1        for p = 0..7
//! ```
//!
//! which lands symbol `s0+i` in `a[0]`/`a[1]` and symbol `s0+i+4` in
//! `a[2]`/`a[3]` — because `2(p-4)+8 == 2p`. Both k-slots of a pair receive the
//! **same** activation, so the tensor core computes `hi*x + lo*x == w*x`.
//!
//! The consequence is the point of the whole format: `a[0]` and `a[2]` are the
//! LOP3 outputs of two states of the SAME row, and `a[1]`/`a[3]` the same two
//! states of the row 8 below. **No lane ever needs a value another lane
//! produced**, so no cross-lane movement can appear between the decode and the
//! HMMA.
//!
//! # Activation dtype — a quality gain, and a range risk
//!
//! The A operand is fp16 *by construction* (the mask keeps the sign bit and 12
//! low bits, the XOR pins each half's exponent field), so B must be fp16 too
//! and a bf16 activation has to be converted.
//!
//! Precision: that is a **gain**. fp16 carries 3 more mantissa bits than bf16;
//! the K=4 sibling measured 1.609e-3 relative-L2 for the bf16 path against
//! 2.081e-4 for fp16 on identical weights — 7.7x more accurate.
//!
//! Range: that is a **risk, and it is UNMEASURED on the real model**. fp16
//! overflows at 65504 where bf16 does not. Post-rotation activations in the K=4
//! probe peaked at 3.3, but that number belongs to that probe, not to
//! DeepSeek-V4. If a real `max|x_rotated|` ever exceeds
//! [`TCFRAG2B_MAX_FP16_ACTIVATION`] the mma sees an infinity. The remedy in
//! this change is the kill switch (`ARC_QTIP_TCFRAG=0`), not a silent clamp:
//! clamping would trade a visible infinity for an invisible wrong answer.

use super::bitshift::{mcg_codeword, K2B, L2B};

/// Symbols packed into one TCFRAG-2B word: 32 bits / 2 bits per symbol.
pub const TCFRAG2B_SYMS_PER_WORD: usize = (32 / K2B) as usize;

/// Symbols in one `mma.m16n8k16` step: 16 k-slots at 2 slots per symbol.
pub const TCFRAG2B_SYMS_PER_MMA_K: usize = 8;

/// Rows of one `mma.m16n8k16` A fragment.
pub const TCFRAG2B_MMA_M: usize = 16;

/// Columns (pairs) of one `mma.m16n8k16` B fragment. This is the batch width
/// that "costs the same as b=1".
pub const TCFRAG2B_MMA_N: usize = 8;

/// Largest activation magnitude an fp16 B operand can carry. Above this the
/// tensor core sees `inf`. See the module docs — UNMEASURED on the real model.
pub const TCFRAG2B_MAX_FP16_ACTIVATION: f32 = 65504.0;

/// State mask, derived from the rung's `L` rather than restated.
const STATE_MASK: u32 = (1u32 << L2B) - 1;

/// Symbols whose values are live in the trellis state at any instant: `L / K`.
///
/// This is the blast radius of a single corrupted symbol, and therefore the
/// floor the mutation guard asserts against.
pub const TCFRAG2B_STATE_WINDOW_SYMS: usize = (L2B / K2B) as usize;

// ---------------------------------------------------------------------------
// D22: the layout is data, selected by compute capability.
// ---------------------------------------------------------------------------

/// Which tensor-core fragment order a device's MMA shape needs.
///
/// This enum is the D22 mechanism: the host picks a variant from the device's
/// compute capability and passes [`Tcfrag2bLayout::abi_id`] across the CUDA
/// ABI. The `.cu` side selects a compile-time-instantiated layout from that
/// integer. **No layout is ever chosen by `#if __CUDA_ARCH__`,** and the
/// on-disk artifact carries none of this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tcfrag2bLayout {
    /// The `mma.m16n8k16` fragment order documented above. Available from
    /// sm_80 (Ampere) and retained through Hopper.
    MmaM16n8k16,
}

/// ABI value for "no TCFRAG layout" — the caller must keep the shipped kernel.
pub const TCFRAG2B_LAYOUT_NONE: i32 = -1;

impl Tcfrag2bLayout {
    /// The integer the CUDA ABI carries. Treat as ABI: append, never renumber.
    pub fn abi_id(self) -> i32 {
        match self {
            Tcfrag2bLayout::MmaM16n8k16 => 0,
        }
    }

    /// Symbols per stored 32-bit word under this layout.
    pub fn syms_per_word(self) -> usize {
        match self {
            Tcfrag2bLayout::MmaM16n8k16 => TCFRAG2B_SYMS_PER_WORD,
        }
    }

    /// The layout for a device, or `None` when this build has no fragment
    /// order for it.
    ///
    /// **Refusing beats guessing.** A future architecture whose MMA shape is
    /// not `m16n8k16` needs its own variant and its own repack; until it has
    /// one, `None` sends the caller back to the shipped kernel, which is
    /// correct everywhere. `m16n8k16` exists from sm_80 onward, so today every
    /// arch this crate compiles for maps to the one variant — that is a fact
    /// about `m16n8k16`, not an assumption about future hardware.
    pub fn for_compute_cap(major: u32, minor: u32) -> Option<Self> {
        let cc = major * 10 + minor;
        if cc >= 80 {
            Some(Tcfrag2bLayout::MmaM16n8k16)
        } else {
            None
        }
    }

    /// `abi_id()`, or [`TCFRAG2B_LAYOUT_NONE`] for `None`.
    pub fn abi_id_or_none(this: Option<Self>) -> i32 {
        match this {
            Some(l) => l.abi_id(),
            None => TCFRAG2B_LAYOUT_NONE,
        }
    }
}

// ---------------------------------------------------------------------------
// The layout itself.
// ---------------------------------------------------------------------------

/// Words per row for a row of `num_symbols` symbols.
///
/// A trailing partial word is zero-filled, which is harmless: those bit-fields
/// name symbols past the end of the row, and no state a decode reads depends on
/// them. Every real `qtip2b` shape has `num_symbols` a multiple of 64, so the
/// pad is normally empty.
#[inline]
pub fn tcfrag2b_words_per_row(num_symbols: usize) -> usize {
    num_symbols.div_ceil(TCFRAG2B_SYMS_PER_WORD)
}

/// Read symbol `t` out of a shipped-format packed row (4 symbols/byte,
/// LSB-first). Mirrors `qtip2b_common.cuh::q2b_sym`.
#[inline]
fn shipped_sym(packed: &[u8], t: usize) -> u8 {
    (packed[t >> 2] >> ((t & 3) * 2)) & 0x3
}

/// Repack ONE shipped-format row (`num_symbols` 2-bit symbols, LSB-first) into
/// TCFRAG-2B words. `out` must be exactly [`tcfrag2b_words_per_row`] long.
pub fn tcfrag2b_repack_row(packed: &[u8], num_symbols: usize, out: &mut [u32]) {
    let w = tcfrag2b_words_per_row(num_symbols);
    assert_eq!(out.len(), w, "tcfrag2b_repack_row: out must be {w} words");
    assert!(
        packed.len() >= num_symbols.div_ceil(4),
        "tcfrag2b_repack_row: packed row is shorter than {num_symbols} symbols"
    );
    for (j, slot) in out.iter_mut().enumerate() {
        let mut word = 0u32;
        for c in 0..TCFRAG2B_SYMS_PER_WORD {
            let t = j * TCFRAG2B_SYMS_PER_WORD + (TCFRAG2B_SYMS_PER_WORD - 1 - c);
            if t < num_symbols {
                word |= u32::from(shipped_sym(packed, t)) << (K2B as usize * c);
            }
        }
        *slot = word;
    }
}

/// Inverse of [`tcfrag2b_repack_row`] — the reader a validator uses to prove a
/// TCFRAG-2B blob carries the symbol stream it claims.
pub fn tcfrag2b_unpack_row(words: &[u32], num_symbols: usize, out: &mut [u8]) {
    assert_eq!(out.len(), num_symbols.div_ceil(4));
    out.fill(0);
    for t in 0..num_symbols {
        let j = t / TCFRAG2B_SYMS_PER_WORD;
        let c = TCFRAG2B_SYMS_PER_WORD - 1 - (t % TCFRAG2B_SYMS_PER_WORD);
        let sym = ((words[j] >> (K2B as usize * c)) & 0x3) as u8;
        out[t >> 2] |= sym << ((t & 3) * 2);
    }
}

/// Repack a whole `[n_rows, packed_per_row]` shipped blob into
/// `[n_rows, tcfrag2b_words_per_row]` TCFRAG-2B words.
///
/// Pure permutation — apply at load time, never re-bake (D22). This is the CPU
/// mirror of `launch_qtip2b_tcfrag_repack`; the two are pinned against each
/// other by the CUDA parity test `cuda_tcfrag2b_repack_matches_cpu`.
pub fn tcfrag2b_repack(
    packed: &[u8],
    n_rows: usize,
    packed_per_row: usize,
    num_symbols: usize,
) -> Vec<u32> {
    let w = tcfrag2b_words_per_row(num_symbols);
    let mut out = vec![0u32; n_rows * w];
    for r in 0..n_rows {
        let src = &packed[r * packed_per_row..(r + 1) * packed_per_row];
        tcfrag2b_repack_row(src, num_symbols, &mut out[r * w..(r + 1) * w]);
    }
    out
}

/// The trellis state at symbol `s`, read straight out of TCFRAG-2B words.
///
/// CPU mirror of the kernel's single `SHF` + `LOP3`:
/// `__funnelshift_r(R[j], R[j-1], 2 * (15 - (s & 15))) & 0xFFFF`. Random
/// access: no sequential warm-up replay, no nibble reversal, no `__brev`.
#[inline]
pub fn tcfrag2b_state(words: &[u32], s: usize) -> u32 {
    let j = s / TCFRAG2B_SYMS_PER_WORD;
    let lo = u64::from(words[j]);
    let hi = if j > 0 { u64::from(words[j - 1]) } else { 0 };
    let sh = K2B as usize * (TCFRAG2B_SYMS_PER_WORD - 1 - (s % TCFRAG2B_SYMS_PER_WORD));
    ((((hi << 32) | lo) >> sh) as u32) & STATE_MASK
}

/// Decode the V=1 codeword at symbol `s` from TCFRAG-2B words.
#[inline]
pub fn tcfrag2b_codeword(words: &[u32], s: usize, mult: u32) -> f32 {
    mcg_codeword(tcfrag2b_state(words, s), mult)
}

// ---------------------------------------------------------------------------
// Placement: the row interleave that makes the warp's word load ONE
// transaction instead of sixteen.
// ---------------------------------------------------------------------------

/// Rows in one `mma.m16n8k16` A-fragment tile.
pub const TCFRAG2B_TILE_ROWS: usize = TCFRAG2B_MMA_M;

/// A-fragment tiles needed to cover `n_rows` output rows. The last tile is
/// zero-padded; padded rows decode to *something* but they land in their own
/// accumulator lanes, which the epilogue never writes.
#[inline]
pub fn tcfrag2b_tile_count(n_rows: usize) -> usize {
    n_rows.div_ceil(TCFRAG2B_TILE_ROWS)
}

/// Flat index of word `j` of `row_in_tile` inside `tile`, for one expert.
///
/// The interleave — **tile-major, k-major, row-minor** — is the second half of
/// the format, and it is what makes the load coalesce. A warp's 32 lanes read
/// 16 distinct rows at the *same* `j`; under the shipped row-major order those
/// are 16 addresses `words_per_row` apart, so one `LDG` becomes up to 16
/// transactions. Interleaved, they are 16 consecutive `u32` = one 64-byte
/// transaction.
///
/// This costs nothing because the permutation happens at load time anyway
/// (D22) — it is the same single bandwidth-bound pass either way.
#[inline]
pub fn tcfrag2b_word_index(
    tile: usize,
    j: usize,
    row_in_tile: usize,
    words_per_row: usize,
) -> usize {
    (tile * words_per_row + j) * TCFRAG2B_TILE_ROWS + row_in_tile
}

/// Words one expert's `[n_rows, packed_per_row]` block occupies in the tiled
/// TCFRAG-2B form.
#[inline]
pub fn tcfrag2b_words_per_expert(n_rows: usize, num_symbols: usize) -> usize {
    tcfrag2b_tile_count(n_rows) * tcfrag2b_words_per_row(num_symbols) * TCFRAG2B_TILE_ROWS
}

/// Read a row's word `j` out of a tiled blob.
#[inline]
pub fn tcfrag2b_tiled_word(words: &[u32], row: usize, j: usize, words_per_row: usize) -> u32 {
    words[tcfrag2b_word_index(
        row / TCFRAG2B_TILE_ROWS,
        j,
        row % TCFRAG2B_TILE_ROWS,
        words_per_row,
    )]
}

/// The trellis state at symbol `s` of `row`, read out of a **tiled** blob.
///
/// The kernel's reader. Identical arithmetic to [`tcfrag2b_state`]; only the
/// two word fetches are strided.
#[inline]
pub fn tcfrag2b_state_tiled(words: &[u32], row: usize, s: usize, words_per_row: usize) -> u32 {
    let j = s / TCFRAG2B_SYMS_PER_WORD;
    let lo = u64::from(tcfrag2b_tiled_word(words, row, j, words_per_row));
    let hi = if j > 0 {
        u64::from(tcfrag2b_tiled_word(words, row, j - 1, words_per_row))
    } else {
        0
    };
    let sh = K2B as usize * (TCFRAG2B_SYMS_PER_WORD - 1 - (s % TCFRAG2B_SYMS_PER_WORD));
    ((((hi << 32) | lo) >> sh) as u32) & STATE_MASK
}

/// Repack one expert's `[n_rows, packed_per_row]` shipped block into the
/// **tiled** TCFRAG-2B form the kernel reads.
///
/// CPU mirror of `launch_qtip2b_tcfrag_repack`. Pure permutation (D22).
pub fn tcfrag2b_repack_tiled(
    packed: &[u8],
    n_rows: usize,
    packed_per_row: usize,
    num_symbols: usize,
) -> Vec<u32> {
    let wpr = tcfrag2b_words_per_row(num_symbols);
    let mut out = vec![0u32; tcfrag2b_words_per_expert(n_rows, num_symbols)];
    let mut row_words = vec![0u32; wpr];
    for row in 0..n_rows {
        let src = &packed[row * packed_per_row..(row + 1) * packed_per_row];
        tcfrag2b_repack_row(src, num_symbols, &mut row_words);
        let (tile, rit) = (row / TCFRAG2B_TILE_ROWS, row % TCFRAG2B_TILE_ROWS);
        for (j, &w) in row_words.iter().enumerate() {
            out[tcfrag2b_word_index(tile, j, rit, wpr)] = w;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The bit-exactness harness.
// ---------------------------------------------------------------------------

/// A single disagreement between two f32 streams, reported on the **bit
/// patterns** rather than the values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32BitMismatch {
    pub index: usize,
    pub want_bits: u32,
    pub got_bits: u32,
}

/// First index at which two f32 streams differ **in their bit patterns**.
///
/// Compare here, BEFORE narrowing to bf16/f16. A repack that loses a low
/// mantissa bit is invisible after a `to_dtype(BF16)` — bf16 keeps 8 of the 24
/// bits — so a comparison taken on the narrowed values cannot see the class of
/// bug this format can actually have.
///
/// `NaN` is compared by bit pattern too, deliberately: two NaNs with different
/// payloads ARE a repack disagreement even though `!=` on the values would call
/// them unequal and `==` would call them not-equal-but-not-a-difference.
pub fn first_f32_bit_mismatch(want: &[f32], got: &[f32]) -> Option<F32BitMismatch> {
    assert_eq!(
        want.len(),
        got.len(),
        "first_f32_bit_mismatch: length mismatch {} vs {}",
        want.len(),
        got.len()
    );
    want.iter()
        .zip(got.iter())
        .enumerate()
        .find(|(_, (w, g))| w.to_bits() != g.to_bits())
        .map(|(index, (w, g))| F32BitMismatch {
            index,
            want_bits: w.to_bits(),
            got_bits: g.to_bits(),
        })
}

/// Move `v` by exactly one ULP, away from zero. The control mutation for
/// [`first_f32_bit_mismatch`].
///
/// ⚠️ Perturb the **OUTPUT**, at a known index — never a shared input. A shared
/// input moves both the reference and the candidate together, so the guard can
/// never fire and the test is vacuous no matter what the code does. That trap
/// has already eaten two agents on this rung.
#[inline]
pub fn perturb_one_ulp(v: f32) -> f32 {
    if v.is_nan() {
        return 0.0;
    }
    let bits = v.to_bits();
    // Away from zero in both signs, and never wrapping the exponent field off
    // the top: +1 on the magnitude bits is exactly one ULP for any finite v.
    if v.is_infinite() {
        return f32::from_bits(bits - 1);
    }
    f32::from_bits(bits + 1)
}

// ---------------------------------------------------------------------------
// The derivation. Numbers here are ARITHMETIC, not measurements.
// ---------------------------------------------------------------------------

/// Instruction-count derivation for the TCFRAG-2B GEMV.
///
/// # ⚠️ UNVERIFIED ON HARDWARE — never run
///
/// Every number this module produces is a hand count of the instructions the
/// kernel is written to emit, not an `ncu` reading. It exists so a reviewer can
/// check the claim's arithmetic before a box ever exists, and so the first box
/// has a prediction to falsify rather than a hope to confirm.
pub mod derivation {
    /// Decode instructions per weight under TCFRAG-2B:
    /// `SHF` (funnel) + `LOP3` (mask to 16 bits) + `IMAD` (MCG) + `LOP3`
    /// (mask/xor) = 4. The result register **is** the mma A operand: no
    /// `cvt`, no `FADD`, no scale multiply, no `MOV`.
    pub const TCFRAG_DECODE_INST_PER_WEIGHT: f64 = 4.0;

    /// Decode instructions per weight in the shipped
    /// `qtip2b_gemv_tuned_kernel` inner loop, counted from its source:
    /// symbol extract (`SHF` + `LOP3`) 2, state update (`SHF` + `LOP3`) 2,
    /// `IMAD` 1, mask/xor `LOP3` 1, two `H2F` 2, hi+lo `FADD` 1,
    /// `w * scale` `FMUL` 1, `FFMA` into the accumulator 1 = 11, plus the
    /// vectorised activation load amortised at ~0.25 and loop overhead ~1.
    pub const SHIPPED_DECODE_INST_PER_WEIGHT: f64 = 12.25;

    /// Weights an `mma.m16n8k16` A fragment carries per lane:
    /// 16 rows x 8 symbols / 32 lanes.
    pub const WEIGHTS_PER_LANE_PER_MMA: f64 = 4.0;

    /// Per k-step, per row tile, per lane, on top of the decode: one packed
    /// word load (one new 32-bit word per row per two k-steps, two rows per
    /// lane => 1.0 amortised) and one `HMMA.16816`.
    pub const PER_MMA_OVERHEAD_INST: f64 = 2.0;

    /// Per k-step, per lane, independent of the row tiles: the B fragment is
    /// two activation loads, two `bf16 -> f32 -> f16` conversions and two
    /// splats into `{h,h}` register pairs. Amortised over `m_tiles` mmas.
    /// An f16 activation drops the two conversions and costs 4.
    pub const B_FRAGMENT_INST: f64 = 6.0;

    /// Derived SASS instructions per weight for the TCFRAG-2B kernel at
    /// `m_tiles` row tiles per warp.
    ///
    /// ```text
    ///     (m * (4 * 4 + 2) + 6) / (4 * m)
    /// ```
    pub fn tcfrag_inst_per_weight(m_tiles: usize) -> f64 {
        let m = m_tiles as f64;
        let per_tile =
            TCFRAG_DECODE_INST_PER_WEIGHT * WEIGHTS_PER_LANE_PER_MMA + PER_MMA_OVERHEAD_INST;
        (m * per_tile + B_FRAGMENT_INST) / (WEIGHTS_PER_LANE_PER_MMA * m)
    }

    /// Derived instruction-count ratio, shipped tuned kernel : TCFRAG-2B.
    ///
    /// Both kernels are **issue-bound** (the shipped one measured at 1% of
    /// roofline with the memory controller 4% utilised), so on that measured
    /// premise instruction count is proportional to time. That premise is what
    /// a box has to check first: at ~2.7x fewer instructions the memory side
    /// rises to only ~11% utilisation, so the kernel should still be
    /// issue-bound afterwards — but "should" is the word doing the work.
    pub fn derived_speedup(m_tiles: usize) -> f64 {
        SHIPPED_DECODE_INST_PER_WEIGHT / tcfrag_inst_per_weight(m_tiles)
    }

    /// The shipped kernel's measured share of a B=256 step, in milliseconds.
    /// Provenance: `BUDGET_V4_PREFILL.md` — 69 ms of 794 ms.
    pub const MEASURED_SHIPPED_MS: f64 = 69.0;

    /// Derived kernel time in ms, from [`MEASURED_SHIPPED_MS`] and the
    /// instruction ratio. **A derivation multiplied by a measurement is still
    /// a derivation.**
    pub fn derived_kernel_ms(m_tiles: usize) -> f64 {
        MEASURED_SHIPPED_MS / derived_speedup(m_tiles)
    }

    /// How far the kernel still sits below the dense fp16 tensor-core roofline
    /// after TCFRAG-2B, expressed as instructions issued per `HMMA.16816`.
    ///
    /// H200 dense fp16 peak is 989 TFLOP/s; at 132 SMs and 1.755 GHz that is
    /// ~1067 FLOP per scheduler-cycle, and one `HMMA.16816` is 4096 FLOP — so
    /// tensor peak needs one HMMA every ~3.84 issued instructions. TCFRAG-2B
    /// issues 18 per HMMA. **The remaining gap is the trellis decode itself,
    /// which no tensor core can absorb, plus the n-dimension being 1..8 of 8
    /// at decode.** Naming it is the point: 16.3x was never all available to
    /// this kernel.
    pub const INST_PER_HMMA: f64 = 18.0;
    /// Instructions per HMMA at which the dense fp16 tensor cores saturate.
    pub const INST_PER_HMMA_AT_TENSOR_PEAK: f64 = 3.84;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qtip::QTIP2B_MCG_MULT;

    /// A deterministic symbol stream, plus the shipped-format packing of it.
    fn fixture(num_symbols: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
        let mut s = seed | 1;
        let mut syms = Vec::with_capacity(num_symbols);
        for _ in 0..num_symbols {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            syms.push((s & 0x3) as u8);
        }
        let mut packed = vec![0u8; num_symbols.div_ceil(4)];
        for (t, &sym) in syms.iter().enumerate() {
            packed[t >> 2] |= sym << ((t & 3) * 2);
        }
        (syms, packed)
    }

    /// THE format property. If this fails the byte order is wrong and every
    /// weight the kernel decodes is wrong with it.
    #[test]
    fn tcfrag2b_state_matches_sequential_recurrence() {
        for seed in 1..=32u64 {
            let n = 512;
            let (syms, packed) = fixture(n, seed.wrapping_mul(0x9E37_79B9));
            let mut words = vec![0u32; tcfrag2b_words_per_row(n)];
            tcfrag2b_repack_row(&packed, n, &mut words);

            let mut state = 0u32;
            for (s, &sym) in syms.iter().enumerate() {
                state = ((state << K2B) | u32::from(sym)) & STATE_MASK;
                assert_eq!(
                    tcfrag2b_state(&words, s),
                    state,
                    "seed {seed}: funnel state != recurrence at symbol {s}"
                );
            }
        }
    }

    #[test]
    fn tcfrag2b_roundtrip_preserves_the_symbol_stream() {
        for &n in &[16usize, 64, 128, 7168] {
            let (_, packed) = fixture(n, 0xDEAD_BEEF);
            let mut words = vec![0u32; tcfrag2b_words_per_row(n)];
            tcfrag2b_repack_row(&packed, n, &mut words);
            let mut back = vec![0u8; packed.len()];
            tcfrag2b_unpack_row(&words, n, &mut back);
            assert_eq!(packed, back, "n={n}: repack/unpack is not a bijection");
        }
    }

    /// The claim that makes TCFRAG-2B re-bake-free (D22): the decoded weights
    /// are bit-identical, compared on the **F32 bit patterns before any
    /// narrowing**, with the control living in the next test.
    #[test]
    fn tcfrag2b_decodes_identical_weights() {
        let n = 1024;
        let (syms, packed) = fixture(n, 0x0BAD_C0DE);
        let mut words = vec![0u32; tcfrag2b_words_per_row(n)];
        tcfrag2b_repack_row(&packed, n, &mut words);

        let mut state = 0u32;
        let want: Vec<f32> = syms
            .iter()
            .map(|&sym| {
                state = ((state << K2B) | u32::from(sym)) & STATE_MASK;
                mcg_codeword(state, QTIP2B_MCG_MULT)
            })
            .collect();
        let got: Vec<f32> = (0..n)
            .map(|s| tcfrag2b_codeword(&words, s, QTIP2B_MCG_MULT))
            .collect();

        assert_eq!(
            first_f32_bit_mismatch(&want, &got),
            None,
            "TCFRAG-2B decode is not bit-identical to the sequential recurrence"
        );
    }

    /// ⚠️ THE CONTROL. Perturb the **OUTPUT** by one ULP at a known index and
    /// prove the comparator goes red.
    ///
    /// A shared *input* perturbation moves both sides together and can never
    /// fire; that is the trap, and this test is the proof it was avoided. If
    /// this test ever passes vacuously (i.e. the assert below stops holding),
    /// `tcfrag2b_decodes_identical_weights` is worthless.
    #[test]
    fn tcfrag2b_bit_harness_goes_red_on_a_one_ulp_output_perturbation() {
        const IDX: usize = 613;
        let n = 1024;
        let (_, packed) = fixture(n, 0x0BAD_C0DE);
        let mut words = vec![0u32; tcfrag2b_words_per_row(n)];
        tcfrag2b_repack_row(&packed, n, &mut words);

        let want: Vec<f32> = (0..n)
            .map(|s| tcfrag2b_codeword(&words, s, QTIP2B_MCG_MULT))
            .collect();
        let mut got = want.clone();
        assert_eq!(first_f32_bit_mismatch(&want, &got), None);

        got[IDX] = perturb_one_ulp(got[IDX]);
        assert_ne!(
            got[IDX].to_bits(),
            want[IDX].to_bits(),
            "the perturbation did not move the output — the control is dead"
        );
        let hit = first_f32_bit_mismatch(&want, &got)
            .expect("a 1-ULP output perturbation must be visible in the F32 bit patterns");
        assert_eq!(hit.index, IDX, "the harness found the wrong index");
        assert_eq!(hit.want_bits.abs_diff(hit.got_bits), 1);
    }

    /// A one-symbol edit must move at least the 8 codewords whose state window
    /// covers it, or the two tests above are vacuous for a different reason.
    #[test]
    fn tcfrag2b_guard_is_live_under_a_one_symbol_mutation() {
        let n = 256;
        let (_, packed) = fixture(n, 0x5EED_5EED);
        let mut words = vec![0u32; tcfrag2b_words_per_row(n)];
        tcfrag2b_repack_row(&packed, n, &mut words);
        let clean: Vec<f32> = (0..n)
            .map(|s| tcfrag2b_codeword(&words, s, QTIP2B_MCG_MULT))
            .collect();

        let mut dirty = words.clone();
        // Flip one 2-bit field well inside the row, so the whole window is in
        // range.
        dirty[4] ^= 0x0000_0030;
        let moved = (0..n)
            .map(|s| tcfrag2b_codeword(&dirty, s, QTIP2B_MCG_MULT))
            .zip(clean.iter())
            .filter(|(d, c)| d.to_bits() != c.to_bits())
            .count();
        assert!(
            moved >= TCFRAG2B_STATE_WINDOW_SYMS,
            "a one-symbol mutation moved only {moved} codewords; the state window is \
             {TCFRAG2B_STATE_WINDOW_SYMS} symbols wide so it must move at least that many — \
             the guard is dead"
        );
    }

    /// The exact bit trick `qtip2b_tcfrag_repack_kernel` uses, mirrored here.
    ///
    /// The kernel does not run a field loop like [`tcfrag2b_repack_row`]; it
    /// reads the four packed bytes as one little-endian `u32` and does
    /// `__brev` + an adjacent-bit swap (three ALU ops). `__brev` reverses all
    /// 32 bits, which also swaps the two bits *inside* each 2-bit symbol, and
    /// the swap puts them back.
    ///
    /// ⚠️ This is the ONLY guard on that kernel's arithmetic. The kernel is
    /// UNVERIFIED ON HARDWARE — never run — so if this test is wrong, nothing
    /// downstream catches it.
    fn kernel_pack_word(src: u32) -> u32 {
        let r = src.reverse_bits();
        ((r & 0x5555_5555) << 1) | ((r >> 1) & 0x5555_5555)
    }

    #[test]
    fn kernel_brev_word_formula_matches_the_field_loop() {
        for seed in 1..=16u64 {
            let n = 256;
            let (_, packed) = fixture(n, seed.wrapping_mul(0xA5A5_1234));
            let mut words = vec![0u32; tcfrag2b_words_per_row(n)];
            tcfrag2b_repack_row(&packed, n, &mut words);

            for (j, &want) in words.iter().enumerate() {
                let src = u32::from_le_bytes([
                    packed[4 * j],
                    packed[4 * j + 1],
                    packed[4 * j + 2],
                    packed[4 * j + 3],
                ]);
                assert_eq!(
                    kernel_pack_word(src),
                    want,
                    "seed {seed}, word {j}: the kernel's brev+swap disagrees with the \
                     field loop — qtip2b_tcfrag_repack_kernel would write a wrong layout"
                );
            }
        }
    }

    /// The tiled placement is a pure re-addressing of the row-major words: the
    /// strided reader the kernel uses must see exactly what the flat reader
    /// sees.
    #[test]
    fn tiled_placement_preserves_every_state() {
        let (n_rows, n) = (37usize, 128usize); // deliberately not a multiple of 16
        let ppr = n / 4;
        let mut blob = Vec::with_capacity(n_rows * ppr);
        for r in 0..n_rows {
            let (_, packed) = fixture(n, 0xBEEF_0000 + r as u64);
            blob.extend_from_slice(&packed);
        }
        let tiled = tcfrag2b_repack_tiled(&blob, n_rows, ppr, n);
        assert_eq!(tiled.len(), tcfrag2b_words_per_expert(n_rows, n));
        let wpr = tcfrag2b_words_per_row(n);

        let mut row_words = vec![0u32; wpr];
        for row in 0..n_rows {
            tcfrag2b_repack_row(&blob[row * ppr..(row + 1) * ppr], n, &mut row_words);
            for s in 0..n {
                assert_eq!(
                    tcfrag2b_state_tiled(&tiled, row, s, wpr),
                    tcfrag2b_state(&row_words, s),
                    "row {row} symbol {s}: tiled reader disagrees with the flat one"
                );
            }
        }
        // The padded tail of the last tile must be zero — the kernel decodes it
        // into accumulator lanes the epilogue never writes, so it only has to
        // be in-bounds, but leaving it uninitialised would be a real bug.
        let last_tile = tcfrag2b_tile_count(n_rows) - 1;
        for rit in (n_rows % TCFRAG2B_TILE_ROWS)..TCFRAG2B_TILE_ROWS {
            for j in 0..wpr {
                assert_eq!(tiled[tcfrag2b_word_index(last_tile, j, rit, wpr)], 0);
            }
        }
    }

    /// The whole-blob path lines up with the per-row path.
    #[test]
    fn tcfrag2b_repack_blob_matches_per_row() {
        let (n_rows, n) = (7usize, 128usize);
        let ppr = n / 4;
        let mut blob = Vec::with_capacity(n_rows * ppr);
        for r in 0..n_rows {
            let (_, packed) = fixture(n, 0x1234_0000 + r as u64);
            blob.extend_from_slice(&packed);
        }
        let all = tcfrag2b_repack(&blob, n_rows, ppr, n);
        let w = tcfrag2b_words_per_row(n);
        for r in 0..n_rows {
            let mut one = vec![0u32; w];
            tcfrag2b_repack_row(&blob[r * ppr..(r + 1) * ppr], n, &mut one);
            assert_eq!(&all[r * w..(r + 1) * w], &one[..], "row {r} differs");
        }
    }

    /// D22: the layout is selected by compute capability and REFUSED when
    /// unknown, never guessed.
    #[test]
    fn layout_is_selected_by_compute_capability_and_refuses_when_unknown() {
        assert_eq!(
            Tcfrag2bLayout::for_compute_cap(9, 0),
            Some(Tcfrag2bLayout::MmaM16n8k16)
        );
        assert_eq!(
            Tcfrag2bLayout::for_compute_cap(8, 0),
            Some(Tcfrag2bLayout::MmaM16n8k16)
        );
        // Below sm_80 there is no `has_qtip_kernels` build at all, and no
        // fragment order here for it.
        assert_eq!(Tcfrag2bLayout::for_compute_cap(7, 5), None);
        assert_eq!(
            Tcfrag2bLayout::abi_id_or_none(Tcfrag2bLayout::for_compute_cap(7, 5)),
            TCFRAG2B_LAYOUT_NONE
        );
        assert_eq!(Tcfrag2bLayout::MmaM16n8k16.abi_id(), 0);
    }

    /// The mma step geometry the kernel is written against. If any of these
    /// move, the `.cu` k-slot assignment moves with them.
    #[test]
    fn mma_step_geometry_is_what_the_kernel_assumes() {
        assert_eq!(TCFRAG2B_SYMS_PER_WORD, 16);
        // 16 k-slots / 2 slots per symbol.
        assert_eq!(TCFRAG2B_SYMS_PER_MMA_K, 8);
        // Two mma steps per stored word — that is what lets the kernel keep a
        // rolling (prev, cur) word pair and load one new word per two steps.
        assert_eq!(TCFRAG2B_SYMS_PER_WORD / TCFRAG2B_SYMS_PER_MMA_K, 2);
        assert_eq!(TCFRAG2B_STATE_WINDOW_SYMS, 8);
        // The state window fits inside one mma step's symbol span, which is
        // why `R[j]` and `R[j-1]` are the only two words a step can need.
        const { assert!(TCFRAG2B_STATE_WINDOW_SYMS <= TCFRAG2B_SYMS_PER_WORD) };
    }

    /// `symbol p -> k-slots 2p, 2p+1` must hold for BOTH halves of the A
    /// fragment — `a[2]`/`a[3]` cover k-slots `2i+8, 2i+9`, and `2(i+4) == 2i+8`.
    /// This is the identity that lets the activation duplication be the single
    /// global rule `xd[2t] = xd[2t+1] = x[t]`.
    #[test]
    fn a_fragment_high_half_lands_on_the_same_k_slot_rule() {
        for i in 0..4usize {
            let p_low = i;
            let p_high = i + 4;
            assert_eq!(2 * p_low, 2 * i);
            assert_eq!(2 * p_high, 2 * i + 8);
        }
    }

    /// The derivation is arithmetic, so it is checkable here. This is NOT a
    /// performance test — it pins the numbers the commit message quotes so
    /// they cannot drift from the code that justifies them.
    #[test]
    fn derived_instruction_counts_are_what_the_commit_claims() {
        use derivation::*;
        // The shipped configuration is M_TILES = 2:
        //   (2 * (4*4 + 2) + 6) / (4 * 2) = 42/8 = 5.25
        assert!((tcfrag_inst_per_weight(2) - 5.25).abs() < 1e-12);
        // More row tiles amortise the B fragment further; fewer blocks is the
        // price, which is why the shipped config is 2 and not 4.
        assert!((tcfrag_inst_per_weight(4) - 4.875).abs() < 1e-12);
        assert!((tcfrag_inst_per_weight(1) - 6.0).abs() < 1e-12);
        // 12.25 / 5.25 = 2.333...
        let s = derived_speedup(2);
        assert!((2.32..2.34).contains(&s), "derived speedup drifted: {s}");
        // 69 ms / 2.333 = 29.6 ms
        let ms = derived_kernel_ms(2);
        assert!(
            (29.0..30.0).contains(&ms),
            "derived kernel ms drifted: {ms}"
        );
        // And the honest ceiling statement: still ~4.7x above tensor-peak
        // issue density.
        const { assert!(INST_PER_HMMA / INST_PER_HMMA_AT_TENSOR_PEAK > 4.0) };
    }
}
