// The K=4 / V=2 codebook, in one place.
//
// The LUT rung (`QtipLayer`, K=4/V=2/L=16) historically decoded every trellis
// state by gathering two floats out of a 65,536 x 2 f32 table — 512 KiB, which
// does not fit in shared memory and is therefore a dependent, data-scattered
// trip to L2 on EVERY candidate the search evaluates and EVERY weight the
// decode reconstructs.
//
// This header adds the alternative: compute the two reproduction values from
// the state in registers, with no table anywhere. The construction is the
// "sum2" form of QTIP's 3INST computed code, and it is exactly the arithmetic
// the sibling qtip2b rung already ships at K=2/V=1
// (`qtip2b_common.cuh::q2b_decode`, `bitshift.rs::mcg_codeword`) applied twice:
//
//     x0 = state * mult                  (32-bit wrapping MCG)
//     v0 = f32(fp16(x0_hi_masked)) + f32(fp16(x0_lo_masked))
//     x1 = x0 * mult                     (chained — this is what makes it V=2)
//     v1 = f32(fp16(x1_hi_masked)) + f32(fp16(x1_lo_masked))
//
// where "masked" is the single LOP3 `(x & 0x8FFF8FFF) ^ 0x3B603B60`.
//
// WHY `sum2` AND NOT THE CHEAPER `split`
// --------------------------------------
// At V=1 the codeword IS `hi + lo` and there is nothing to decide. At V=2 you
// need two values per state and the cheap option ("split") is to stop summing
// and take `hi` and `lo` as the pair — 4 instructions per 2 weights instead of
// ~10. Do not. The mask keeps only the sign bit and the low 12 bits of each
// half and the XOR pins each half's exponent into 12..15, so a single masked
// fp16 half **cannot have magnitude below 0.142** — split puts a hole in the
// codebook exactly where a Gaussian weight distribution has most of its mass.
// Measured at K=4/V=2 over 3 fixture families x 3 weight draws (wave19-AP
// part 2, `probe_computed_codebook_quality`):
//
//     sum2   +0.00017 cos, +0.37% weight NMSE vs the Gaussian LUT  -> NEUTRAL
//     split  -0.00174 cos, +3.73% weight NMSE                      -> SHIP NEVER
//
// `qtip_cb_sum2` below is the neutral one. There is no `split` in this file on
// purpose.
//
// SCALE
// -----
// The raw sum2 table has sigma = 1.225552 (measured over all 2^16 states, both
// values; pinned by `computed_codebook_sigma_matches_scale_divisor` on the Rust
// side). The rung's row-scale policy is `max|row| / 3` against a UNIT-spread
// codebook, so the computed codebook is matched by folding sigma into the
// divisor — `max|row| / (3 * sigma)` — exactly as `QTIP2B_SCALE_DIVISOR`
// (3.0 x 1.2064) does for the K=2/V=1 rung. Folding it into the scale rather
// than into the values keeps the decode at zero extra instructions.
//
// PARITY
// ------
// The two fp16 halves are summed in **f32** (`__fadd_rn`), never with `__hadd`:
// the conversions are exact and the f32 add is correctly rounded, so this is
// bit-identical to the Rust reference `mcg_codeword_v2`. `__fadd_rn` also
// cannot be contracted into an FMA by `--use_fast_math`, which is the same
// reason `qtip_exact_fp.cuh` exists.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

// Per-half keep mask (sign bit + 12 low bits) and exponent-window XOR. Shared
// with the qtip2b rung; see `bitshift.rs::QTIP2B_MASK` / `QTIP2B_XOR`.
#define QTIP_CB_MASK 0x8FFF8FFFu
#define QTIP_CB_XOR  0x3B603B60u

// Fold one 32-bit MCG product into a codeword: mask/xor, then sum the two
// fp16 halves in f32.
__device__ __forceinline__ float qtip_cb_fold(unsigned int x) {
    const unsigned int m = (x & QTIP_CB_MASK) ^ QTIP_CB_XOR;
    const float hi = __half2float(__ushort_as_half((unsigned short)(m >> 16)));
    const float lo = __half2float(__ushort_as_half((unsigned short)(m & 0xFFFFu)));
    return __fadd_rn(hi, lo);
}

// The V=2 codeword pair, given the FIRST MCG product `x0 = state * mult`.
//
// Split out from `qtip_cb_sum2` so the beam kernel — which reaches its 16
// successors by strength-reducing `(base|j) * mult` to `base*mult + j*mult`,
// and so already holds `x0` — shares this arithmetic instead of restating it.
// The duplicate cost a real mutation its teeth: changing `x1 = x0 * mult` here
// used to leave the beam kernel untouched, so the bake parity gate could not
// see it (measured, wave24-AU mutation M3).
__device__ __forceinline__ float2 qtip_cb_pair_from_x0(unsigned int x0, unsigned int mult) {
    const unsigned int x1 = x0 * mult;   // chained — this is what makes it V=2
    return make_float2(qtip_cb_fold(x0), qtip_cb_fold(x1));
}

// The V=2 codeword pair for trellis `state`. `mult` is the MCG multiplier
// (`QTIP_MCG_V2_MULT` on the Rust side).
__device__ __forceinline__ float2 qtip_cb_sum2(unsigned int state, unsigned int mult) {
    return qtip_cb_pair_from_x0(state * mult, mult);
}

// ---------------------------------------------------------------------------
// Unified accessor.
//
// `COMPUTED` is a compile-time template parameter so neither path pays for the
// other: the LUT instantiation is byte-for-byte the kernel that shipped before
// this header existed, and the computed instantiation contains no load at all.
// At the ABI boundary the selector is the multiplier itself — `cb_mult == 0`
// means "gather from the stored table", and 0 is not a legal MCG multiplier
// (it must be odd), so the sentinel can never collide with a real codebook.
// ---------------------------------------------------------------------------
template <bool COMPUTED>
__device__ __forceinline__ float2 qtip_cb_value(
    const float* __restrict__ lut, unsigned int state, unsigned int mult
);

template <>
__device__ __forceinline__ float2 qtip_cb_value<false>(
    const float* __restrict__ lut, unsigned int state, unsigned int mult
) {
    (void)mult;
    return make_float2(lut[(size_t)state * 2u + 0u], lut[(size_t)state * 2u + 1u]);
}

template <>
__device__ __forceinline__ float2 qtip_cb_value<true>(
    const float* __restrict__ lut, unsigned int state, unsigned int mult
) {
    (void)lut;
    return qtip_cb_sum2(state, mult);
}

// Read-only-cache variant, for the decode kernels that use `__ldg` on the LUT.
template <bool COMPUTED>
__device__ __forceinline__ float2 qtip_cb_value_ldg(
    const float* __restrict__ lut, unsigned int state, unsigned int mult
);

template <>
__device__ __forceinline__ float2 qtip_cb_value_ldg<false>(
    const float* __restrict__ lut, unsigned int state, unsigned int mult
) {
    (void)mult;
    const float* p = lut + (size_t)state * 2u;
    return make_float2(__ldg(p), __ldg(p + 1));
}

template <>
__device__ __forceinline__ float2 qtip_cb_value_ldg<true>(
    const float* __restrict__ lut, unsigned int state, unsigned int mult
) {
    (void)lut;
    return qtip_cb_sum2(state, mult);
}
