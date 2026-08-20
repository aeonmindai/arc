//! Parent system: ArcQuant / ArcFormat (the ArcOverlay serving convention)
//!
//! TCFRAG — the tensor-core fragment byte order for the K=4/V=2/L=16 rung.
//!
//! # Why this exists
//!
//! The shipped trellis GEMV measures **26.31 SASS instructions per weight**
//! against an issue budget of 1.74 at 2 bpw. It is issue-bound, not
//! memory-bound, and a null-fold control (decode made ≈free, MCG chain intact)
//! still measures 10.38 — so **even a zero-cost codebook leaves ~6× the budget
//! on the floor**. The residue is symbol extraction, the fp16→f32 fold, the
//! per-weight scale multiply and the fragment shuffling; the only thing that
//! removes it is a byte order the tensor cores can consume directly.
//!
//! The key observation is that the `sum2` computed codebook already produces,
//! in ONE 32-bit register,
//!
//! ```text
//!     m = ((state * mult) & 0x8FFF_8FFF) ^ 0x3B60_3B60
//! ```
//!
//! whose two 16-bit halves are *already fp16 numbers* and whose codeword is
//! their sum ([`super::mcg_fold`]). A dot product is linear, so
//!
//! ```text
//!     sum_k (hi_k + lo_k) * x_k  ==  sum_k hi_k * x_k  +  sum_k lo_k * x_k
//! ```
//!
//! Give the two halves **adjacent k-slots** of an `mma.m16n8k16` A fragment and
//! duplicate the activation across those two slots, and the tensor core does
//! the fold, the row scale and the k-reduction itself. The LOP3 destination
//! register *is* the mma operand register, with nothing in between.
//!
//! Measured on an H200 (sm_90a, CUDA 12.4), `arc-tools/bench/qtip_mma_frag_probe.cu`:
//!
//! | kernel                                   | inst/weight |
//! |------------------------------------------|-------------|
//! | shipped trellis GEMV (LUT gather)        | 26.31       |
//! | shipped + 512 KiB LUT-gather arm         | 9.38        |
//! | scalar decode on THIS format (no mma)    | 11.69       |
//! | **TCFRAG decode → mma**                  | **3.56**    |
//! | issue budget at 2 bpw                    | 1.74        |
//!
//! and the shuffle count between the decode and the HMMA is **zero**: over
//! every `HMMA` in the hot loop, all 256 A-operand registers are produced
//! directly by `LOP3.LUT`, with no `MOV` / `PRMT` / `SHFL` in between.
//!
//! # The byte layout
//!
//! A row of `S` symbols (4 bits each) is stored as `W = ceil(S/8)` words of 32
//! bits. Word `j` holds symbols `8j..8j+7` in **reversed nibble order**:
//!
//! ```text
//!     R[j] bits [4c .. 4c+3]  =  sym[8j + (7 - c)]        for c = 0..7
//! ```
//!
//! Equivalently, the concatenation `(R[j-1] << 32) | R[j]` is a continuous
//! *descending*-symbol nibble stream. That is the whole trick: the trellis
//! state at symbol `s`,
//!
//! ```text
//!     state(s) = sym[s-3]<<12 | sym[s-2]<<8 | sym[s-1]<<4 | sym[s]
//! ```
//!
//! (the shipped `state = ((state << 4) | sym) & 0xFFFF` recurrence) reads its
//! nibbles in the *opposite* order from the way the shipped format stores them,
//! which is why every shipped decode path pays a nibble/byte reversal —
//! `gg_state_from_window` in `qtip_gather_gemv.cu` (4 ops), or the amortized
//! `__brev` pair-reversal in `qtip_grouped_gemm.cu`. Under TCFRAG the state is
//!
//! ```text
//!     state(s) = funnel_shift_right(R[j], R[j-1], 4*(7 - (s & 7))) & 0xFFFF
//!                with j = s >> 3 and R[-1] = 0
//! ```
//!
//! — one `SHF` and one `LOP3`, at any `s`, with no reversal and no sequential
//! warm-up replay. [`tcfrag_state`] is that expression; the property test
//! `tcfrag_state_matches_sequential_recurrence` is how a reader validates a
//! candidate repack.
//!
//! ## This is a pure permutation — there is no re-bake
//!
//! TCFRAG reorders stored bytes. It does not change the symbol stream, the
//! trellis, the codebook or the scales, so **the decoded weights are
//! bit-identical** and an existing baked artifact can be converted at load
//! time by [`tcfrag_repack`]. `tcfrag_decodes_identical_weights` pins that.
//!
//! ## The k-slot assignment (the half a kernel cannot choose on its own)
//!
//! For `mma.m16n8k16` the A fragment of lane `t` (with `gid = t>>2`,
//! `i = t&3`) covers
//!
//! ```text
//!     a0 <- (row gid  , k 2i  , 2i+1)      a2 <- (row gid  , k 2i+8, 2i+9)
//!     a1 <- (row gid+8, k 2i  , 2i+1)      a3 <- (row gid+8, k 2i+8, 2i+9)
//! ```
//!
//! TCFRAG defines logical weight index → k-slot as
//!
//! ```text
//!     weight 2s   (codeword 0 of symbol s) -> k-slots 2i  , 2i+1
//!     weight 2s+1 (codeword 1 of symbol s) -> k-slots 2i+8, 2i+9
//! ```
//!
//! with `s = 8j + i` on even mma steps and `s = 8j + 4 + i` on odd ones. Both
//! halves of a k-slot pair receive the **same** activation, duplicated, so the
//! tensor core computes `hi*x + lo*x == codeword*x`.
//!
//! The consequence is the point of the whole format: `a0`/`a2` are the two
//! LOP3 outputs of ONE state, and `a1`/`a3` the two of the neighbouring row's
//! state at the same `s`. **No lane ever needs a value another lane produced**,
//! so no cross-lane movement can appear between the decode and the HMMA.
//!
//! ## The activation side
//!
//! The runtime counterpart is that `x` is written duplicated —
//! `xd[2t] = xd[2t+1] = x[t]` — which makes each B fragment pair a single
//! aligned 64-bit load instead of a 32-bit load plus two `PRMT`. That costs
//! 2× the activation scratch (28 KB at k = 7168) and is free inside the D.H.D
//! rotation kernel that already materialises `x_rotated`. Measured neutral in
//! wall clock at the shapes probed so far (the kernel stops being issue-bound
//! before it pays off), so it is described here but not required.
//!
//! ## Activation dtype
//!
//! The A operand is fp16 by construction (the mask/XOR pins each half's fp16
//! exponent), so B must be fp16 too and the bf16 activation has to be
//! converted. That is a quality **gain**, not a cost: measured over
//! 4096 rows × k=7168 against an exact-f32-activation reference on identical
//! weights, the shipped bf16 path is 1.609e-3 relative-L2 and the fp16 path is
//! 2.081e-4 — **7.7× more accurate**, because fp16 carries 3 more mantissa
//! bits. The risk is range, not precision: fp16 overflows at 65504, and
//! post-rotation activations in the probe peaked at 3.3. Any integration must
//! re-measure `max|x|` on the real model rather than inherit that number.

use super::{mcg_codeword_v2, STATE_MASK};

/// Symbols packed into one TCFRAG word. 8 nibbles = 32 bits.
pub const TCFRAG_SYMS_PER_WORD: usize = 8;

/// Words per row for a row of `num_symbols` symbols.
///
/// The trailing partial word (when `num_symbols` is not a multiple of 8) is
/// zero-filled, which is harmless: those nibbles name symbols past the end of
/// the row and no state that a decode reads depends on them. Every real rung
/// shape has `num_symbols` a multiple of 32, so the pad is normally empty.
#[inline]
pub fn tcfrag_words_per_row(num_symbols: usize) -> usize {
    num_symbols.div_ceil(TCFRAG_SYMS_PER_WORD)
}

/// Read symbol `t` out of a shipped-format packed row (low nibble first).
#[inline]
fn shipped_sym(packed: &[u8], t: usize) -> u8 {
    let b = packed[t >> 1];
    if t & 1 == 1 {
        (b >> 4) & 0x0F
    } else {
        b & 0x0F
    }
}

/// Repack ONE shipped-format row (`num_symbols` nibbles, low-nibble-first)
/// into TCFRAG words.
///
/// `out` must be exactly [`tcfrag_words_per_row`] long.
pub fn tcfrag_repack_row(packed: &[u8], num_symbols: usize, out: &mut [u32]) {
    let w = tcfrag_words_per_row(num_symbols);
    assert_eq!(out.len(), w, "tcfrag_repack_row: out must be {w} words");
    assert!(
        packed.len() >= num_symbols.div_ceil(2),
        "tcfrag_repack_row: packed row is shorter than {num_symbols} symbols"
    );
    for (j, slot) in out.iter_mut().enumerate() {
        let mut word = 0u32;
        for c in 0..TCFRAG_SYMS_PER_WORD {
            let t = j * TCFRAG_SYMS_PER_WORD + (TCFRAG_SYMS_PER_WORD - 1 - c);
            if t < num_symbols {
                word |= u32::from(shipped_sym(packed, t)) << (4 * c);
            }
        }
        *slot = word;
    }
}

/// Inverse of [`tcfrag_repack_row`] — the reader a validator uses to prove a
/// TCFRAG blob carries the symbol stream it claims.
pub fn tcfrag_unpack_row(words: &[u32], num_symbols: usize, out: &mut [u8]) {
    assert_eq!(out.len(), num_symbols.div_ceil(2));
    out.fill(0);
    for t in 0..num_symbols {
        let j = t / TCFRAG_SYMS_PER_WORD;
        let c = TCFRAG_SYMS_PER_WORD - 1 - (t % TCFRAG_SYMS_PER_WORD);
        let sym = ((words[j] >> (4 * c)) & 0x0F) as u8;
        if t & 1 == 1 {
            out[t >> 1] |= sym << 4;
        } else {
            out[t >> 1] |= sym;
        }
    }
}

/// Repack a whole `[n_rows, packed_per_row]` shipped blob into
/// `[n_rows, tcfrag_words_per_row]` TCFRAG words.
///
/// Pure permutation: apply at load time, no re-bake.
pub fn tcfrag_repack(
    packed: &[u8],
    n_rows: usize,
    packed_per_row: usize,
    num_symbols: usize,
) -> Vec<u32> {
    let w = tcfrag_words_per_row(num_symbols);
    let mut out = vec![0u32; n_rows * w];
    for r in 0..n_rows {
        let src = &packed[r * packed_per_row..(r + 1) * packed_per_row];
        tcfrag_repack_row(src, num_symbols, &mut out[r * w..(r + 1) * w]);
    }
    out
}

/// The trellis state at symbol `s`, read straight out of TCFRAG words.
///
/// This is the CPU mirror of the kernel's single `SHF` + `LOP3`:
/// `__funnelshift_r(R[j], R[j-1], 4*(7 - (s & 7))) & 0xFFFF`. It is a
/// random-access read — no sequential warm-up replay, no nibble reversal.
#[inline]
pub fn tcfrag_state(words: &[u32], s: usize) -> u32 {
    let j = s / TCFRAG_SYMS_PER_WORD;
    let lo = u64::from(words[j]);
    let hi = if j > 0 { u64::from(words[j - 1]) } else { 0 };
    let sh = 4 * (TCFRAG_SYMS_PER_WORD - 1 - (s % TCFRAG_SYMS_PER_WORD));
    (((hi << 32) | lo) >> sh) as u32 & STATE_MASK
}

/// Decode the V=2 codeword pair at symbol `s` from TCFRAG words.
#[inline]
pub fn tcfrag_codeword(words: &[u32], s: usize, mult: u32) -> (f32, f32) {
    mcg_codeword_v2(tcfrag_state(words, s), mult)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qtip::QTIP_MCG_V2_MULT;

    /// A deterministic symbol stream, plus the shipped-format packing of it.
    fn fixture(num_symbols: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
        let mut s = seed | 1;
        let mut syms = Vec::with_capacity(num_symbols);
        for _ in 0..num_symbols {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            syms.push((s & 0x0F) as u8);
        }
        let mut packed = vec![0u8; num_symbols.div_ceil(2)];
        for (t, &sym) in syms.iter().enumerate() {
            if t & 1 == 1 {
                packed[t >> 1] |= sym << 4;
            } else {
                packed[t >> 1] |= sym;
            }
        }
        (syms, packed)
    }

    /// THE format property. If this fails, the byte order is wrong and every
    /// weight the kernel decodes is wrong with it.
    #[test]
    fn tcfrag_state_matches_sequential_recurrence() {
        for seed in 1..=32u64 {
            let n = 256;
            let (syms, packed) = fixture(n, seed * 0x9E37_79B9);
            let mut words = vec![0u32; tcfrag_words_per_row(n)];
            tcfrag_repack_row(&packed, n, &mut words);

            let mut state = 0u32;
            for (s, &sym) in syms.iter().enumerate() {
                state = ((state << 4) | u32::from(sym)) & STATE_MASK;
                assert_eq!(
                    tcfrag_state(&words, s),
                    state,
                    "seed {seed}: funnel state != recurrence at symbol {s}"
                );
            }
        }
    }

    #[test]
    fn tcfrag_roundtrip_preserves_the_symbol_stream() {
        for &n in &[8usize, 32, 64, 3584] {
            let (_, packed) = fixture(n, 0xDEAD_BEEF);
            let mut words = vec![0u32; tcfrag_words_per_row(n)];
            tcfrag_repack_row(&packed, n, &mut words);
            let mut back = vec![0u8; packed.len()];
            tcfrag_unpack_row(&words, n, &mut back);
            assert_eq!(packed, back, "n={n}: repack/unpack is not a bijection");
        }
    }

    /// The weights are bit-identical, so TCFRAG can be applied to an existing
    /// artifact at load time. This is the claim that makes it re-bake-free.
    #[test]
    fn tcfrag_decodes_identical_weights() {
        let n = 512;
        let (syms, packed) = fixture(n, 0x0BAD_C0DE);
        let mut words = vec![0u32; tcfrag_words_per_row(n)];
        tcfrag_repack_row(&packed, n, &mut words);

        let mut state = 0u32;
        for (s, &sym) in syms.iter().enumerate() {
            state = ((state << 4) | u32::from(sym)) & STATE_MASK;
            let want = mcg_codeword_v2(state, QTIP_MCG_V2_MULT);
            let got = tcfrag_codeword(&words, s, QTIP_MCG_V2_MULT);
            assert_eq!(
                want.0.to_bits(),
                got.0.to_bits(),
                "v0 differs at symbol {s}"
            );
            assert_eq!(
                want.1.to_bits(),
                got.1.to_bits(),
                "v1 differs at symbol {s}"
            );
        }
    }

    /// Prove the guard can go red: a one-nibble edit must be visible in the
    /// decoded weights, or the two tests above are vacuous.
    #[test]
    fn tcfrag_guard_is_live_under_a_one_nibble_mutation() {
        let n = 128;
        let (_, packed) = fixture(n, 0x5EED_5EED);
        let mut words = vec![0u32; tcfrag_words_per_row(n)];
        tcfrag_repack_row(&packed, n, &mut words);
        let clean: Vec<(f32, f32)> = (0..n)
            .map(|s| tcfrag_codeword(&words, s, QTIP_MCG_V2_MULT))
            .collect();

        let mut dirty = words.clone();
        dirty[4] ^= 0x0000_00F0;
        let moved = (0..n)
            .map(|s| tcfrag_codeword(&dirty, s, QTIP_MCG_V2_MULT))
            .zip(clean.iter())
            .filter(|(d, c)| d.0.to_bits() != c.0.to_bits() || d.1.to_bits() != c.1.to_bits())
            .count();
        assert!(
            moved >= 4,
            "a one-nibble mutation moved only {moved} codewords; the state window is \
             4 symbols wide so it must move at least 4 -- the guard is dead"
        );
    }

    /// The whole-blob path lines up with the per-row path.
    #[test]
    fn tcfrag_repack_blob_matches_per_row() {
        let (n_rows, n) = (7usize, 64usize);
        let ppr = n / 2;
        let mut blob = Vec::with_capacity(n_rows * ppr);
        for r in 0..n_rows {
            let (_, packed) = fixture(n, 0x1234_0000 + r as u64);
            blob.extend_from_slice(&packed);
        }
        let all = tcfrag_repack(&blob, n_rows, ppr, n);
        let w = tcfrag_words_per_row(n);
        for r in 0..n_rows {
            let mut one = vec![0u32; w];
            tcfrag_repack_row(&blob[r * ppr..(r + 1) * ppr], n, &mut one);
            assert_eq!(&all[r * w..(r + 1) * w], &one[..], "row {r} differs");
        }
    }
}
