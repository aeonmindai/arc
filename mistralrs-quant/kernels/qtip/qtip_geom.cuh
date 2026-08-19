// Parent system: ArcQuant / QTIP — the trellis GEOMETRY, in one place.
//
// A QTIP rung is fixed by three numbers:
//
//     K   bits per symbol        (alphabet = 2^K successors per state)
//     V   weights per symbol     (reproduction-vector dimension)
//     L   trellis state bits     (state space = 2^L)
//
// and the rate is `K / V` bits per weight. Two rungs therefore sit at the SAME
// 2 bits/weight:
//
//     K=4 / V=2 / L=16   the shipped LUT rung ("qtip2")
//     K=8 / V=4 / L=12   the same rate at materially fewer decode instructions
//                        per weight, because one symbol now carries four
//                        weights instead of two
//
// ⚠ The ABSOLUTE inst/weight figures this header used to quote (4.375 vs
// 15.125, "3.46x") are SUSPENDED and deliberately not restated. They came from
// the SASS census's differential-over-unroll-depth method, whose K=8 replay
// control failed to reproduce: `#pragma unroll` is a HINT, nvcc re-rolled the
// loops, all three unroll depths returned identical counts, and the linearity
// guard reads |d2-d1|/d1 — which is 0.00% precisely when all three are equal.
// The guard passed hardest when the rig was most broken. What survives from
// that census is the clamp DELTA (+1.126 at K=9, +1.125 at K=10), because a
// difference between two kernels in the same mode cancels the re-rolling. The
// direction "one symbol carries four weights, so fewer instructions per weight"
// is structural and does not depend on the census; the ratio does.
//
// and raising K at fixed V=4/L=12 is the ONLY way to buy quality on this
// family, because `bpw = K/V`:
//
//     K=9 / V=4 / L=12   2.25 bits/weight, same 32,768 B codebook, same decode
//                        shape. This is the rung the encoder is being taught to
//                        bake; K=8/V=4/L=12 is quality-CLOSED (Dw_cos -0.00698
//                        with a random codebook and -0.00307 with a converged
//                        trellis-Lloyd one, against a +-0.0008 ship band, over
//                        six codebook designs — trellis freedom binds, not
//                        codebook coverage). K=9 measures +0.00402. Those
//                        numbers and the sweep behind them live in
//                        `qtip/trellis_v4l12.rs`'s module docs; do not restate
//                        them anywhere else.
//
// Before this header each kernel restated `K`/`L`/`V` as its own local
// `constexpr`, and every one of those restatements assumed K=4 somewhere else
// in the file — in the packing (`|= sym << 4` for odd t), in a group table
// sized `2^(L-K)`, in a `cand[16]` register array, in a thread block that
// assumed one thread could own a whole group. Changing the rung meant editing
// numbers that were only implicitly linked. The type below makes the link
// explicit and `static_assert`-checked, so a geometry a kernel cannot serve
// fails to COMPILE rather than silently emitting the wrong bytes.
//
// THE BITSTREAM (this is a WIRE format, not a local convention)
// -------------------------------------------------------------
// Symbol `t` occupies bits `[t*K, t*K + K)` of the packed row, LSB-first, and
// the row is `ceil(num_symbols * K / 8)` bytes. That is exactly what the
// serving side's `qtip/trellis_v4l12.rs::Rung::{pack, extract, packed_bytes}`
// implements, and the bake MUST NOT invent a second convention: the UQFF
// geometry section is `[tag=3, K, L, V]`, so K is an explicit wire field and
// the encoder and the decoder are reading the same bytes at the same offsets.
//
// At K=4 and K=8 the bitstream and the old "8/K whole symbols per byte" packer
// are the SAME bytes — `bit = 4t` puts the even symbol in the low nibble and
// the odd one in the high nibble, and `bit = 8t` is a plain byte store — which
// is why `qtip_pack_symbol` keeps the byte-aligned expression verbatim under
// `if constexpr` rather than trusting a general form to fold back to it.
//
// K=9 is where the two diverge: 8 % 9 != 0, so a symbol straddles two bytes,
// `SYMS_PER_BYTE` is meaningless (integer 8/9 == 0, and the old
// `num_symbols / SYMS_PER_BYTE` divided by zero), and the packer has to write
// `MAX_BYTES_PER_SYMBOL` bytes. See that member for why the bound is 2 and not
// 3.
//
// WHAT IS NOT HERE
// ----------------
// The Rust-side geometry (`qtip/mod.rs::{K, V, L}`) is still three separate
// `pub const`s. This header is deliberately the CUDA half only: the launchers
// in `qtip_beam.cu` take `(k_bits, v_dim, l_bits)` at the ABI boundary and
// REFUSE an unsupported triple, so when the Rust consts become a parameter the
// bake follows with no second parameterisation to reconcile.

#pragma once

#include <cstdint>

template <uint32_t K_, uint32_t V_, uint32_t L_>
struct QtipGeom {
    static constexpr uint32_t K = K_;
    static constexpr uint32_t V = V_;
    static constexpr uint32_t L = L_;

    static constexpr uint32_t ALPHABET      = 1u << K_;          // successors per state
    static constexpr uint32_t SYM_MASK      = ALPHABET - 1u;
    static constexpr uint32_t STATE_MASK    = (1u << L_) - 1u;
    static constexpr uint32_t GROUP_BITS    = L_ - K_;           // predecessor-prefix bits
    static constexpr uint32_t GROUP_COUNT   = 1u << (L_ - K_);
    static constexpr uint32_t GROUP_MASK    = GROUP_COUNT - 1u;
    static constexpr uint32_t LUT_SIZE      = 1u << L_;          // states, not floats

    // Does a whole number of symbols fit a byte? True at K=1,2,4,8; false at
    // K=9. Only the true case has a meaningful `SYMS_PER_BYTE`; the false case
    // is pinned to 0, so the only code that may read it is code guarded by
    // `if constexpr (BYTE_ALIGNED)` — anything else is an obvious divide-by-zero
    // rather than a quietly short row. (`8u / 9u` is 0 by integer division, not
    // UB, so the discarded ternary arm is still well-formed.)
    static constexpr bool     BYTE_ALIGNED  = (8u % K_ == 0u);
    static constexpr uint32_t SYMS_PER_BYTE = BYTE_ALIGNED ? (8u / K_) : 0u;

    // gcd(K, 8). 8 is a power of two, so this is the largest power of two that
    // divides K, capped at 8.
    static constexpr uint32_t GCD8 =
        (K_ % 8u == 0u) ? 8u : ((K_ % 4u == 0u) ? 4u : ((K_ % 2u == 0u) ? 2u : 1u));

    // Bytes ONE symbol can touch, worst case over the bit offsets that are
    // actually REACHABLE.
    //
    // Symbol `t` starts at bit offset `(t*K) mod 8`, which ranges over the
    // multiples of `gcd(K, 8)` and nothing else — so the worst reachable offset
    // is `8 - gcd(K, 8)`, not 7, and the bound is
    // `ceil((8 - gcd(K,8) + K) / 8)`:
    //
    //     K=8  -> gcd 8, offsets {0},          bound 1   (a plain byte store)
    //     K=9  -> gcd 1, offsets {0..7},       bound 2
    //     K=10 -> gcd 2, offsets {0,2,4,6},    bound 2   (naive ceil((7+K)/8)
    //                                                     says 3 and is wrong)
    //
    // This is the same formula as `Rung::max_bytes_per_symbol` on the serving
    // side, and the same correction: `ceil((7+K)/8)` over-reads at K=10.
    static constexpr uint32_t MAX_BYTES_PER_SYMBOL = (8u - GCD8 + K_ + 7u) / 8u;

    // K <= 15 rather than K <= 8: the symbol is no longer required to tile a
    // byte, only to leave the trellis some state (`L_ > K_`) with `L_ <= 16`.
    static_assert(K_ >= 1 && K_ <= 15, "K must leave state in a 16-bit trellis");
    static_assert(L_ > K_, "the trellis must retain state across a symbol");
    // `s_beam_state` is `unsigned short` and every selection key is laid out as
    // `(total_order_key(cost) << 16) | state`, so the state must fit 16 bits.
    static_assert(L_ <= 16, "state must fit the 16 low bits of the selection key");
    static_assert(V_ >= 1, "V is the reproduction-vector dimension");
};

// The shipped LUT rung. 2 bits/weight. (Its inst/weight absolute is suspended
// with the rest of the census — see the header.)
using QtipGeomK4V2L16 = QtipGeom<4, 2, 16>;
// Same 2 bits/weight, fewer decode instructions per weight (absolute suspended).
using QtipGeomK8V4L12 = QtipGeom<8, 4, 12>;
// 2.25 bits/weight on the same table and the same decode shape. The rung the
// quality sweep actually selected; K=8/V=4/L=12 is closed.
using QtipGeomK9V4L12 = QtipGeom<9, 4, 12>;
// K=10/V=4/L=12 (2.5 bpw) needs no new code here — GROUP_COUNT falls to 4, so
// `QbShape::LANES` becomes 64 and `CAND` stays 16 — but it is deliberately NOT
// instantiated: a geometry that has not been through `-res-usage` is a claim,
// not a rung.

// Bytes of packed output one row of `num_symbols` symbols occupies:
// `ceil(num_symbols * K / 8)`.
//
// The byte-aligned branch is the shipped expression, character for character,
// so the K=4 and K=8 instantiations compile from unchanged source rather than
// from a general form that is merely believed to fold back to it. The two
// agree wherever both are defined: at K=4 with `num_symbols` even (which
// `cuda_ops.rs` enforces) `(4n+7)/8 == n/2`, and at K=8 `(8n+7)/8 == n`.
template <class G>
__device__ __forceinline__ int qtip_packed_bytes_per_row(int num_symbols) {
    if constexpr (G::BYTE_ALIGNED) {
        return num_symbols / (int)G::SYMS_PER_BYTE;
    } else {
        return (num_symbols * (int)G::K + 7) / 8;
    }
}

// OR one symbol into the packed row at bits `[t*K, t*K + K)`, LSB-first.
//
// At K=4 this is exactly the shipped `pkd[t/2] |= sym` for even `t` and
// `pkd[t/2] |= sym << 4` for odd `t` (shift = 4 * (t & 1)); at K=8 it is a
// plain byte store (SYMS_PER_BYTE == 1, shift == 0). The caller must have
// zeroed the row first — this ORs, it does not assign.
//
// The straddling branch writes bytes `b0 .. b0 + ceil((off+K)/8) - 1`, i.e.
// only the bytes the symbol's bits actually fall in — NOT a fixed
// `MAX_BYTES_PER_SYMBOL` run. That is what keeps the LAST symbol of a row in
// bounds: its highest bit is `num_symbols*K - 1`, which lives in byte
// `(num_symbols*K - 1) / 8 <= ceil(num_symbols*K/8) - 1`.
//
// Only thread 0 packs (the backtrace walk is serial), so the read-modify-write
// on a byte two adjacent symbols share is not a race.
template <class G>
__device__ __forceinline__ void qtip_pack_symbol(
    uint8_t* __restrict__ pkd, int t, unsigned int sym
) {
    if constexpr (G::BYTE_ALIGNED) {
        const int      slot  = t / (int)G::SYMS_PER_BYTE;
        const unsigned shift = G::K * ((unsigned)t % G::SYMS_PER_BYTE);
        pkd[slot] |= (uint8_t)((sym & G::SYM_MASK) << shift);
    } else {
        const int          bit  = t * (int)G::K;
        const int          b0   = bit >> 3;
        const unsigned int off  = (unsigned int)(bit & 7);
        const unsigned int used = off + G::K;   // bits of `b0`'s window touched
        // `sym < 2^15` and `off <= 7`, so this cannot overflow 32 bits.
        const unsigned int v    = (sym & G::SYM_MASK) << off;
        #pragma unroll
        for (unsigned int b = 0; b < G::MAX_BYTES_PER_SYMBOL; ++b) {
            if (8u * b < used) {
                pkd[b0 + (int)b] |= (uint8_t)((v >> (8u * b)) & 0xFFu);
            }
        }
    }
}
