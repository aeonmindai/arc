// QTIP trellis geometry SASS census.
//
// Parent system: ArcQuant / QTIP  (measurement harness, not a shipped kernel)
//
// WHY THIS FILE EXISTS
// --------------------
// The trellis decode ladder has been priced in `inst/weight` from estimates.
// Hand-counting from C++ undercounts SASS by ~2.05x on this exact kernel family
// (a census subagent made that error and had its answer retracted). The only
// honest instrument is nvcc + cuobjdump, and neither needs a GPU: `-cubin`
// stops before host codegen and emits device SASS for the target arch.
//
// METHODOLOGY: DIFFERENTIAL COUNTING
// ----------------------------------
// Counting a whole kernel and dividing by weights/thread silently folds in the
// prologue (LUT staging, row-pointer setup, scale loads) and the epilogue (warp
// reduction, store). Those are O(1) per thread, so at small trip counts they
// dominate and the resulting "inst/weight" is wrong in the pessimistic
// direction.
//
// Instead each geometry is compiled at THREE unroll depths (NG = 8, 16, 24
// groups) and the steady-state marginal cost is
//
//     inst/weight = (SASS(NG=24) - SASS(NG=8)) / ((24 - 8) * WEIGHTS_PER_GROUP)
//
// The prologue/epilogue cancel exactly. NG=16 is compiled purely as a
// LINEARITY CHECK: if the two half-differences disagree by more than a percent,
// the loop is not in steady state (register pressure, spilling, or nvcc
// hoisting across iterations) and the number must not be reported.
//
// WHAT IS HELD CONSTANT ACROSS GEOMETRIES
// ---------------------------------------
// `inst/weight` is only comparable if the geometries do the same amount of
// useful work per group. K=8/V=4 decodes 4 weights per symbol where K=4/V=2
// decodes 2, so SYMBOLS per group are NOT the invariant -- WEIGHTS are.
// WEIGHTS_PER_GROUP is pinned to 8 everywhere and GROUP_SYMS falls out as 8/V.
// For K4/V2 that reproduces the shipped kernel's GROUP=4
// (qtip_gather_gemv.cu:374) exactly.
//
// SCOPE / HONESTY
// ---------------
// * This is a decode GEMV (scalar FMA), which is the B=1 shape. A bf16 LUT
//   entry therefore costs a real bf16->f32 convert per value. The claim that a
//   V=4 bf16 entry is "one LDS.64 = a complete mma B-operand pair, zero
//   conversion" is true only on an `mma`-based GEMM path, NOT here. What this
//   file does test is the LDS.64 half: the V=4 fetch is spelled as a single
//   8-byte shared load so the SASS shows whether nvcc emits one LDS.64.
// * No kernel here runs. SASS instruction count is a static issue-count proxy,
//   validated in-tree against ncu to 1.3% on the GEMV and confirmed to track
//   measured time ~1:1 on the grouped GEMM (qtip_grouped_gemm.cu:125-133).

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../../mistralrs-quant/kernels/qtip/qtip_codebook.cuh"

// Codebook kinds.
#define CB_COMPUTED_SUM2 0  // in-register MCG, what master ships (RUN-161)
#define CB_LUT_BF16      1  // shared-memory bf16 table

// State-entry modes -- this is lever #3.
#define WIN_REPLAY 0  // sequential warm-up replay, what master ships
#define WIN_BREV   1  // random access via __brev + intra-nibble bit swaps
#define WIN_PRMT   2  // random access via PRMT byte-reverse + nibble swap

// ---------------------------------------------------------------------------
// Geometry descriptor.
// ---------------------------------------------------------------------------
template <int K_, int V_, int L_, int CB_>
struct Geom {
    static constexpr int K = K_;
    static constexpr int V = V_;
    static constexpr int L = L_;
    static constexpr int CB = CB_;

    static constexpr uint32_t STATE_MASK = (1u << L_) - 1u;

    // Warm-up symbols needed to fill an L-bit state K bits at a time.
    static constexpr int WARM = (L_ + K_ - 1) / K_;

    // Weights per group is the invariant across geometries (see header note).
    static constexpr int WEIGHTS_PER_GROUP = 8;
    static constexpr int GROUP_SYMS = WEIGHTS_PER_GROUP / V_;

    // bits/weight = K/V (mistralrs-quant/src/qtip/mod.rs:376-381).
    static constexpr int LUT_ENTRIES = 1 << L_;
    static constexpr int LUT_VALUES = LUT_ENTRIES * V_;
    static constexpr int LUT_BYTES = (CB_ == CB_LUT_BF16) ? LUT_VALUES * 2 : 0;
};

// ---------------------------------------------------------------------------
// Symbol extraction. K=8 is byte-aligned, which removes the nibble shift/mask
// entirely -- one of the structural reasons K=8 is cheaper per symbol.
// ---------------------------------------------------------------------------
template <int K>
__device__ __forceinline__ uint32_t census_sym(const uint8_t* __restrict__ p, int t);

template <>
__device__ __forceinline__ uint32_t census_sym<4>(const uint8_t* __restrict__ p, int t) {
    const uint8_t b = p[t >> 1];
    return (t & 1) ? ((uint32_t)b >> 4) : ((uint32_t)b & 0x0Fu);
}

template <>
__device__ __forceinline__ uint32_t census_sym<8>(const uint8_t* __restrict__ p, int t) {
    return (uint32_t)p[t];
}

// ---------------------------------------------------------------------------
// Random-access state reconstruction (lever #3).
//
// The shipped K=2 helper `q2b_state_from_window` (qtip2b_common.cuh:61-64) is a
// PAIR reversal and does not generalise: at K=4 the bijection is a NIBBLE
// reversal, at K=8 a BYTE reversal. Neither exists in the tree -- deriving them
// is part of what this census prices.
//
// Derivation for K=4, L=16. The recurrence is
//     state = ((state << 4) | sym) & 0xFFFF
// so after symbol t:
//     state = sym[t] | sym[t-1]<<4 | sym[t-2]<<8 | sym[t-3]<<12
// The packed stream stores nibbles LSB-first, so the 16-bit window covering
// symbols t-3..t (byte-aligned when t-3 is even) is
//     win = sym[t-3] | sym[t-2]<<4 | sym[t-1]<<8 | sym[t]<<12
// i.e. state is the window with its four NIBBLES reversed, bits within each
// nibble preserved.
//
// PRMT selector 0x4401: result byte0 <- src byte1, byte1 <- src byte0,
// bytes 2,3 <- byte 4 (the low byte of the zero operand). Verified by hand
// against the CUDA __byte_perm contract {y,x} indexed 0..7 from the LSB of x.
// ---------------------------------------------------------------------------

// Route A: full bit-reversal, then undo the reversal inside each nibble by
// swapping adjacent bits and then adjacent bit-pairs.
__device__ __forceinline__ uint32_t nibble_reverse16_brev(uint32_t win) {
    uint32_t r = __brev(win & 0xFFFFu) >> 16;
    r = ((r & 0x5555u) << 1) | ((r >> 1) & 0x5555u);  // swap adjacent bits
    r = ((r & 0x3333u) << 2) | ((r >> 2) & 0x3333u);  // swap adjacent pairs
    return r;
}

// Route B: reverse the two bytes with a single PRMT, then swap the nibbles
// inside each byte. Fewer ops on paper -- the SASS decides.
__device__ __forceinline__ uint32_t nibble_reverse16_prmt(uint32_t win) {
    const uint32_t b = __byte_perm(win, 0u, 0x4401u);
    return ((b & 0x0F0Fu) << 4) | ((b >> 4) & 0x0F0Fu);
}

// K=8: state is a BYTE reversal of the window (L=12 keeps only 12 bits).
__device__ __forceinline__ uint32_t byte_reverse16(uint32_t win) {
    return __byte_perm(win, 0u, 0x4401u);
}

// Load the 16 raw stream bits ending at symbol `t`.
template <int K>
__device__ __forceinline__ uint32_t load_win16(const uint8_t* __restrict__ p, int t);

template <>
__device__ __forceinline__ uint32_t load_win16<4>(const uint8_t* __restrict__ p, int t) {
    const int byte0 = (t - 3) >> 1;  // t is odd at a group boundary => t-3 even
    return (uint32_t)p[byte0] | ((uint32_t)p[byte0 + 1] << 8);
}

template <>
__device__ __forceinline__ uint32_t load_win16<8>(const uint8_t* __restrict__ p, int t) {
    return (uint32_t)p[t - 1] | ((uint32_t)p[t] << 8);
}

template <int G_K, int MODE>
__device__ __forceinline__ uint32_t state_at(const uint8_t* __restrict__ p, int t,
                                             uint32_t mask) {
    if constexpr (MODE == WIN_BREV) {
        return nibble_reverse16_brev(load_win16<G_K>(p, t)) & mask;
    } else if constexpr (G_K == 8) {
        return byte_reverse16(load_win16<G_K>(p, t)) & mask;
    } else {
        return nibble_reverse16_prmt(load_win16<G_K>(p, t)) & mask;
    }
}

// ---------------------------------------------------------------------------
// Codeword fetch. Writes V floats into `w`.
// ---------------------------------------------------------------------------
template <typename G>
__device__ __forceinline__ void decode(const __nv_bfloat16* __restrict__ s_lut,
                                       uint32_t state, unsigned int mult,
                                       float (&w)[G::V]) {
    // `if constexpr` is load-bearing, not style: the dead arms are ill-formed
    // for the other geometries (the V=4 arm writes w[2]/w[3] on a float[2]).
    if constexpr (G::CB == CB_COMPUTED_SUM2) {
        const float2 p = qtip_cb_sum2(state, mult);
        w[0] = p.x;
        w[1] = p.y;
    } else if constexpr (G::V == 2) {
        // One LDS.32: two adjacent bf16.
        const __nv_bfloat162 v =
            *reinterpret_cast<const __nv_bfloat162*>(s_lut + (size_t)state * 2u);
        w[0] = __low2float(v);
        w[1] = __high2float(v);
    } else {
        // One LDS.64: four adjacent bf16, 8-byte aligned because state*4 bf16
        // is an 8-byte stride. Spelled as a uint2 so nvcc cannot split it.
        const uint2 raw =
            *reinterpret_cast<const uint2*>(s_lut + (size_t)state * 4u);
        const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        w[0] = __low2float(p0);
        w[1] = __high2float(p0);
        w[2] = __low2float(p1);
        w[3] = __high2float(p1);
    }
}

// ---------------------------------------------------------------------------
// The census body.
//
// Mirrors the shipped grouped-strided decode (qtip_gather_gemv.cu:381-416):
// register-blocked over ROWS rows, activations loaded once per group and reused
// across rows, one warp reduction at the end.
//
// HOIST is lever #2: false reproduces the shipped `fmaf(w * scale, x, acc)`
// (one FMUL per weight); true accumulates unscaled and applies the row scale
// once after the k-loop.
// ---------------------------------------------------------------------------
template <typename G, int NG, int ROWS, bool HOIST, int MODE>
__device__ __forceinline__ void census_body(
    const uint8_t* __restrict__ packed, const float* __restrict__ x,
    const float* __restrict__ row_scales, const __nv_bfloat16* __restrict__ g_lut,
    float* __restrict__ y, int packed_per_row, unsigned int mult,
    __nv_bfloat16* s_lut) {
    constexpr int GS = G::GROUP_SYMS;
    constexpr int WPG = G::WEIGHTS_PER_GROUP;

    // `if constexpr` again: for the computed geometry `s_lut` is a 1-element
    // dummy and G::LUT_VALUES is 131072, so a runtime `if` would both emit dead
    // staging code (polluting the count) and write far out of bounds.
    if constexpr (G::CB == CB_LUT_BF16) {
        for (int i = threadIdx.x; i < G::LUT_VALUES; i += blockDim.x) {
            s_lut[i] = g_lut[i];
        }
        __syncthreads();
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * (blockDim.x >> 5) + warp) * ROWS;

    const uint8_t* rp[ROWS];
    float scl[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        rp[r] = packed + (size_t)(row0 + r) * packed_per_row;
        scl[r] = row_scales[row0 + r];
    }

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    // Fully unrolled over NG groups -- the knob the differential varies.
    // `lane + 1` keeps every index non-negative at g == 0 without costing an
    // instruction (it folds into the address immediate).
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const int base = (lane + 1 + g * 32) * GS;

        float xg[WPG];
#pragma unroll
        for (int j = 0; j < WPG; ++j) xg[j] = x[base * G::V + j];

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint8_t* p = rp[r];
            const float scale = scl[r];

            uint32_t state;
            if constexpr (MODE == WIN_REPLAY) {
                // Shipped: replay G::WARM symbols to rebuild the state. At
                // K4/V2 WARM == GROUP_SYMS == 4, a 1:1 warm-up-to-useful ratio.
                state = 0u;
#pragma unroll
                for (int t = base - G::WARM; t < base; ++t) {
                    state = ((state << G::K) | census_sym<G::K>(p, t)) & G::STATE_MASK;
                }
            } else {
                state = state_at<G::K, MODE>(p, base - 1, G::STATE_MASK);
            }

            float a = acc[r];
#pragma unroll
            for (int s = 0; s < GS; ++s) {
                state = ((state << G::K) | census_sym<G::K>(p, base + s)) & G::STATE_MASK;
                float w[G::V];
                decode<G>(s_lut, state, mult, w);
#pragma unroll
                for (int i = 0; i < G::V; ++i) {
                    const float xv = xg[s * G::V + i];
                    a = HOIST ? fmaf(w[i], xv, a) : fmaf(w[i] * scale, xv, a);
                }
            }
            acc[r] = a;
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float a = HOIST ? acc[r] * scl[r] : acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xFFFFFFFFu, a, off);
        if (lane == 0) y[row0 + r] = a;
    }
}

// ---------------------------------------------------------------------------
// Instantiations. Each variant gets its own `extern "C"` kernel so cuobjdump
// reports a stable, greppable symbol (no C++ mangling to parse).
// ---------------------------------------------------------------------------
using G_K4V2L16_CB = Geom<4, 2, 16, CB_COMPUTED_SUM2>;  // SHIPPED
using G_K4V2L13_LUT = Geom<4, 2, 13, CB_LUT_BF16>;      // 32 KB bf16 LUT
using G_K8V4L12_LUT = Geom<8, 4, 12, CB_LUT_BF16>;      // 32 KB bf16 LUT

#define CENSUS_KERNEL(NAME, GEOM, NG, ROWS, HOIST, MODE)                       \
    extern "C" __global__ void __launch_bounds__(256)                          \
        census_##NAME(const uint8_t* __restrict__ packed,                      \
                      const float* __restrict__ x,                             \
                      const float* __restrict__ rs,                            \
                      const __nv_bfloat16* __restrict__ lut,                   \
                      float* __restrict__ y, int ppr, unsigned int mult) {     \
        __shared__ __nv_bfloat16 s_lut[GEOM::CB == CB_LUT_BF16 ? GEOM::LUT_VALUES : 1]; \
        census_body<GEOM, NG, ROWS, HOIST, MODE>(packed, x, rs, lut, y, ppr,   \
                                                 mult, s_lut);                 \
    }

// --- Geometry 1: shipped K4/V2/L16, computed codebook, replay, no hoist ------
CENSUS_KERNEL(g1_ng8, G_K4V2L16_CB, 8, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g1_ng16, G_K4V2L16_CB, 16, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g1_ng24, G_K4V2L16_CB, 24, 2, false, WIN_REPLAY)

// --- Geometry 2: K4/V2/L13, 32 KB bf16 LUT, replay, no hoist -----------------
CENSUS_KERNEL(g2_ng8, G_K4V2L13_LUT, 8, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g2_ng16, G_K4V2L13_LUT, 16, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g2_ng24, G_K4V2L13_LUT, 24, 2, false, WIN_REPLAY)

// --- Geometry 3: K8/V4/L12, 32 KB bf16 LUT, replay, no hoist -----------------
CENSUS_KERNEL(g3_ng8, G_K8V4L12_LUT, 8, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g3_ng16, G_K8V4L12_LUT, 16, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g3_ng24, G_K8V4L12_LUT, 24, 2, false, WIN_REPLAY)

// --- Lever #2: row-scale hoist ----------------------------------------------
CENSUS_KERNEL(g1h_ng8, G_K4V2L16_CB, 8, 2, true, WIN_REPLAY)
CENSUS_KERNEL(g1h_ng24, G_K4V2L16_CB, 24, 2, true, WIN_REPLAY)
CENSUS_KERNEL(g3h_ng8, G_K8V4L12_LUT, 8, 2, true, WIN_REPLAY)
CENSUS_KERNEL(g3h_ng24, G_K8V4L12_LUT, 24, 2, true, WIN_REPLAY)

// --- Lever #3: random-access window in place of the warm-up replay -----------
CENSUS_KERNEL(g1w_ng8, G_K4V2L16_CB, 8, 2, false, WIN_PRMT)
CENSUS_KERNEL(g1w_ng24, G_K4V2L16_CB, 24, 2, false, WIN_PRMT)
CENSUS_KERNEL(g1b_ng8, G_K4V2L16_CB, 8, 2, false, WIN_BREV)
CENSUS_KERNEL(g1b_ng24, G_K4V2L16_CB, 24, 2, false, WIN_BREV)
CENSUS_KERNEL(g3w_ng8, G_K8V4L12_LUT, 8, 2, false, WIN_PRMT)
CENSUS_KERNEL(g3w_ng24, G_K8V4L12_LUT, 24, 2, false, WIN_PRMT)

// --- Both levers stacked on the best geometry --------------------------------
CENSUS_KERNEL(g3wh_ng8, G_K8V4L12_LUT, 8, 2, true, WIN_PRMT)
CENSUS_KERNEL(g3wh_ng24, G_K8V4L12_LUT, 24, 2, true, WIN_PRMT)
