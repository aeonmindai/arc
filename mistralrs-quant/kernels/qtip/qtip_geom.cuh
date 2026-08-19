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
//     K=8 / V=4 / L=12   the same rate, 3.46x fewer decode instructions per
//                        weight (4.375 vs 15.125), because one symbol now
//                        carries four weights instead of two
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
    static constexpr uint32_t SYMS_PER_BYTE = 8u / K_;

    static_assert(K_ >= 1 && K_ <= 8, "K must fit a byte-packable symbol");
    static_assert(8u % K_ == 0u, "K must divide 8: the packer emits whole bytes");
    static_assert(L_ > K_, "the trellis must retain state across a symbol");
    // `s_beam_state` is `unsigned short` and every selection key is laid out as
    // `(total_order_key(cost) << 16) | state`, so the state must fit 16 bits.
    static_assert(L_ <= 16, "state must fit the 16 low bits of the selection key");
    static_assert(V_ >= 1, "V is the reproduction-vector dimension");
};

// The shipped LUT rung. 2 bits/weight, 15.125 decode instructions/weight.
using QtipGeomK4V2L16 = QtipGeom<4, 2, 16>;
// Same 2 bits/weight at 4.375 decode instructions/weight (3.46x fewer).
using QtipGeomK8V4L12 = QtipGeom<8, 4, 12>;

// Bytes of packed output one row of `num_symbols` symbols occupies.
template <class G>
__device__ __forceinline__ int qtip_packed_bytes_per_row(int num_symbols) {
    return num_symbols / (int)G::SYMS_PER_BYTE;
}

// OR one symbol into the packed row.
//
// At K=4 this is exactly the shipped `pkd[t/2] |= sym` for even `t` and
// `pkd[t/2] |= sym << 4` for odd `t` (shift = 4 * (t & 1)); at K=8 it is a
// plain byte store (SYMS_PER_BYTE == 1, shift == 0). The caller must have
// zeroed the row first — this ORs, it does not assign.
template <class G>
__device__ __forceinline__ void qtip_pack_symbol(
    uint8_t* __restrict__ pkd, int t, unsigned int sym
) {
    const int      slot  = t / (int)G::SYMS_PER_BYTE;
    const unsigned shift = G::K * ((unsigned)t % G::SYMS_PER_BYTE);
    pkd[slot] |= (uint8_t)((sym & G::SYM_MASK) << shift);
}
