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
#define WIN_SEQ    3  // contiguous per-thread slice, state carried across groups

// Symbol-extraction routes. K=4 (nibble) and K=8 (byte) have a native spelling;
// K=9 and K=10 do not divide 8, so the field straddles a byte boundary and the
// route becomes a real design choice. See the "BYTE ALIGNMENT" note below.
#define EXT_NATIVE 0  // K=4 nibble / K=8 single byte -- the aligned control
#define EXT_B2     1  // 2 raw byte loads, no bounds clamp (row stride padded)
#define EXT_B2C    2  // 2 raw byte loads, tail-clamped index (unpadded stride)
#define EXT_B3     3  // 3 raw byte loads -- the generic ceil((K+7)/8) read
#define EXT_FUN    4  // 32-bit word pair + __funnelshift_r
#define EXT_SPLIT  5  // bit-plane split: byte plane + (K-8)-bit plane

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
// BYTE ALIGNMENT -- the thing this census exists to price.
//
// `bpw = K/V`, so at fixed V=4/L=12 the only way to buy quality with bits is to
// raise K. K=8 is a byte; K=9 and K=10 are not. Symbol t then occupies bits
// [K*t, K*t+K) of the packed row, so the extract needs a runtime bit offset and
// a multi-byte read, and THAT is where a cheap decode can evaporate.
//
// Every route below produces the identical value -- verified bit-exact against
// an independent reference bit-extractor on the host, because an instruction
// count for a WRONG extract is worthless. They differ only in how the
// straddling field is assembled, and the SASS decides which is cheapest -- the
// same discipline the PRMT/BREV pair below follows. Nothing here is
// hand-counted.
//
//   EXT_B2    two byte loads.  Correct for K<=9 at any offset, and for K=10 as
//             well: 10*t mod 8 is in {0,2,4,6}, so off+K <= 16 always. Reads
//             one byte past the last symbol of a row, which is why it is the
//             PADDED-STRIDE route.
//   EXT_B2C   EXT_B2 with `min(idx, row_bytes-1)` on each byte index. This is
//             what an unpadded format forces, and the clamp is a real,
//             REMOVABLE cost -- pricing it separately is the point.
//   EXT_B3    the generic ceil((K+7)/8) = 3-byte read, correct for any K<=17 at
//             any offset. K=10's offset set makes this strictly redundant here;
//             it is compiled so a defensively-written serving kernel can see
//             what its generality costs.
//   EXT_FUN   one funnel shift over an aligned uint32 pair. Requires the row
//             stride to be a multiple of 4 (a format decision, free to make).
//   EXT_SPLIT two planes: the low 8 bits of every symbol stay byte-aligned, and
//             the remaining K-8 bits live in a separate packed plane. The
//             byte-plane read is then bit-identical to K=8's.
// ---------------------------------------------------------------------------

// `row_bytes` is only read by EXT_B2C and EXT_SPLIT. It is threaded through
// every route so the aligned control's codegen is unchanged by the existence of
// the others.
template <int K, int EXT>
__device__ __forceinline__ uint32_t census_sym_ext(const uint8_t* __restrict__ p, int t,
                                                   int row_bytes) {
    constexpr uint32_t MASK = (1u << K) - 1u;

    if constexpr (EXT == EXT_NATIVE) {
        (void)row_bytes;
        (void)MASK;
        return census_sym<K>(p, t);
    } else if constexpr (EXT == EXT_SPLIT) {
        // Plane 0 is the low byte of every symbol -- literally K=8's access.
        // Plane 1 packs the top K-8 bits, 8/(K-8) symbols per byte.
        constexpr int HI = K - 8;
        constexpr int PER = 8 / HI;
        const uint8_t* p1 = p + row_bytes;  // plane base; loop-invariant
        const uint32_t lo = (uint32_t)p[t];
        const uint32_t hi = ((uint32_t)p1[t / PER] >> ((t % PER) * HI)) & ((1u << HI) - 1u);
        return lo | (hi << 8);
    } else {
        const int bit = t * K;
        const int off = bit & 7;
        const int b0 = bit >> 3;

        if constexpr (EXT == EXT_FUN) {
            (void)row_bytes;
            (void)off;
            (void)b0;
            // Row stride padded to 4 B, so the row base is uint32-aligned.
            const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
            const int wi = bit >> 5;
            const int woff = bit & 31;
            return __funnelshift_r(w[wi], w[wi + 1], woff) & MASK;
        } else if constexpr (EXT == EXT_B3) {
            (void)row_bytes;
            const uint32_t v = (uint32_t)p[b0] | ((uint32_t)p[b0 + 1] << 8) |
                               ((uint32_t)p[b0 + 2] << 16);
            return (v >> off) & MASK;
        } else if constexpr (EXT == EXT_B2C) {
            // Ternary rather than ::min so this cannot pick up a host overload;
            // ptxas emits IMNMX either way.
            const int last = row_bytes - 1;
            const int i0 = b0 < last ? b0 : last;
            const int i1 = (b0 + 1) < last ? (b0 + 1) : last;
            const uint32_t v = (uint32_t)p[i0] | ((uint32_t)p[i1] << 8);
            return (v >> off) & MASK;
        } else {
            (void)row_bytes;
            const uint32_t v = (uint32_t)p[b0] | ((uint32_t)p[b0 + 1] << 8);
            return (v >> off) & MASK;
        }
    }
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
template <typename G, int NG, int ROWS, bool HOIST, int MODE, int EXT>
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

    // WIN_SEQ: the shape the SERVING kernel actually has
    // (mistralrs-quant/kernels/qtip/qtip_gemv_k8v4l12.cu -- each thread walks a
    // CONTIGUOUS slice of the row's symbols). The trellis state is then
    // loop-carried across groups and the warm-up runs ONCE per thread, in the
    // prologue, where the differential cancels it. That halves the extractions
    // per weight versus WIN_REPLAY, which re-seeds the state for every group --
    // and since the K=9 penalty is paid per EXTRACTION, the two modes bracket
    // the answer rather than agreeing. Both are reported.
    const int slice = (int)(threadIdx.x + blockIdx.x * blockDim.x) * (NG * GS) + G::WARM;
    uint32_t st[ROWS];
    if constexpr (MODE == WIN_SEQ) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            uint32_t s = 0u;
            for (int t = slice - G::WARM; t < slice; ++t) {
                s = ((s << G::K) | census_sym_ext<G::K, EXT>(rp[r], t, packed_per_row)) &
                    G::STATE_MASK;
            }
            st[r] = s;
        }
    }

    // Fully unrolled over NG groups -- the knob the differential varies.
    // `lane + 1` keeps every index non-negative at g == 0 without costing an
    // instruction (it folds into the address immediate).
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const int base = (MODE == WIN_SEQ) ? (slice + g * GS) : ((lane + 1 + g * 32) * GS);

        float xg[WPG];
#pragma unroll
        for (int j = 0; j < WPG; ++j) xg[j] = x[base * G::V + j];

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint8_t* p = rp[r];
            const float scale = scl[r];

            uint32_t state;
            if constexpr (MODE == WIN_SEQ) {
                // State carried from the previous group -- no re-seed at all.
                state = st[r];
            } else if constexpr (MODE == WIN_REPLAY) {
                // Shipped: replay G::WARM symbols to rebuild the state. At
                // K4/V2 WARM == GROUP_SYMS == 4, a 1:1 warm-up-to-useful ratio.
                state = 0u;
#pragma unroll
                for (int t = base - G::WARM; t < base; ++t) {
                    state = ((state << G::K) |
                             census_sym_ext<G::K, EXT>(p, t, packed_per_row)) &
                            G::STATE_MASK;
                }
            } else {
                state = state_at<G::K, MODE>(p, base - 1, G::STATE_MASK);
            }

            float a = acc[r];
#pragma unroll
            for (int s = 0; s < GS; ++s) {
                state = ((state << G::K) |
                         census_sym_ext<G::K, EXT>(p, base + s, packed_per_row)) &
                        G::STATE_MASK;
                float w[G::V];
                decode<G>(s_lut, state, mult, w);
#pragma unroll
                for (int i = 0; i < G::V; ++i) {
                    const float xv = xg[s * G::V + i];
                    a = HOIST ? fmaf(w[i], xv, a) : fmaf(w[i] * scale, xv, a);
                }
            }
            if constexpr (MODE == WIN_SEQ) st[r] = state;
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

// The bit-buying ladder. At fixed V=4/L=12 the table is 2^12 x 4 bf16 = 32,768 B
// for ALL of these -- the codebook does not read K -- so the ONLY thing that
// moves is the bit rate (bpw = K/V) and the cost of pulling a K-bit symbol out
// of the stream. K=9 is 2.25 bpw, K=10 is 2.50 bpw, and each must be scored
// against ITS OWN budget band, never against K=8's.
using G_K9V4L12_LUT = Geom<9, 4, 12, CB_LUT_BF16>;    // 2.25 bpw -- THE DECISION
using G_K10V4L12_LUT = Geom<10, 4, 12, CB_LUT_BF16>;  // 2.50 bpw -- the trend

static_assert(G_K9V4L12_LUT::LUT_VALUES * 2 == 32768, "K=9 must keep the 32 KiB table");
static_assert(G_K10V4L12_LUT::LUT_VALUES * 2 == 32768, "K=10 must keep the 32 KiB table");
static_assert(G_K8V4L12_LUT::WARM == 2 && G_K9V4L12_LUT::WARM == 2 &&
                  G_K10V4L12_LUT::WARM == 2,
              "ceil(L/K) must be 2 for all three, or the extraction COUNT differs "
              "between them and the K delta stops being a pure alignment delta");
static_assert(G_K8V4L12_LUT::GROUP_SYMS == 2 && G_K9V4L12_LUT::GROUP_SYMS == 2 &&
                  G_K10V4L12_LUT::GROUP_SYMS == 2,
              "same symbols per group across the ladder");

#define CENSUS_KERNEL_EXT(NAME, GEOM, NG, ROWS, HOIST, MODE, EXT)              \
    extern "C" __global__ void __launch_bounds__(256)                          \
        census_##NAME(const uint8_t* __restrict__ packed,                      \
                      const float* __restrict__ x,                             \
                      const float* __restrict__ rs,                            \
                      const __nv_bfloat16* __restrict__ lut,                   \
                      float* __restrict__ y, int ppr, unsigned int mult) {     \
        __shared__ __nv_bfloat16 s_lut[GEOM::CB == CB_LUT_BF16 ? GEOM::LUT_VALUES : 1]; \
        census_body<GEOM, NG, ROWS, HOIST, MODE, EXT>(packed, x, rs, lut, y,   \
                                                      ppr, mult, s_lut);       \
    }

// The aligned control spelling. Every pre-existing kernel below goes through
// this, so K=4/K=8 codegen is unchanged by the K=9/K=10 work.
#define CENSUS_KERNEL(NAME, GEOM, NG, ROWS, HOIST, MODE)                       \
    CENSUS_KERNEL_EXT(NAME, GEOM, NG, ROWS, HOIST, MODE, EXT_NATIVE)

// NG LADDER: 4 / 8 / 12.
//
// The first census run used 8/16/24 and hit a HARD 4096-instruction ceiling:
// census_g1_ng24, g1b_ng24, g1h_ng24 and g1w_ng24 all reported EXACTLY 4096 on
// both sm_80 and sm_90, while g2_ng24 came in at 4048 on sm_80 and reported
// clean. Four different kernel bodies landing on the same power of two is a
// clamp, not a coincidence -- beyond it the unroller stops and the count stops
// tracking NG, which destroys the differential. The linearity check caught it
// (92.56% divergence on g1) and the numbers were discarded.
//
// NG=12 keeps the largest body (g1, ~250 inst/group) near 3,000 -- comfortably
// clear of the ceiling -- while still giving an 8-group lever arm.
#define NG_LO 4
#define NG_MID 8
#define NG_HI 12

// --- Geometry 1: shipped K4/V2/L16, computed codebook, replay, no hoist ------
CENSUS_KERNEL(g1_ng4, G_K4V2L16_CB, NG_LO, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g1_ng8, G_K4V2L16_CB, NG_MID, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g1_ng12, G_K4V2L16_CB, NG_HI, 2, false, WIN_REPLAY)

// --- Geometry 2: K4/V2/L13, 32 KB bf16 LUT, replay, no hoist -----------------
CENSUS_KERNEL(g2_ng4, G_K4V2L13_LUT, NG_LO, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g2_ng8, G_K4V2L13_LUT, NG_MID, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g2_ng12, G_K4V2L13_LUT, NG_HI, 2, false, WIN_REPLAY)

// --- Geometry 3: K8/V4/L12, 32 KB bf16 LUT, replay, no hoist -----------------
CENSUS_KERNEL(g3_ng4, G_K8V4L12_LUT, NG_LO, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g3_ng8, G_K8V4L12_LUT, NG_MID, 2, false, WIN_REPLAY)
CENSUS_KERNEL(g3_ng12, G_K8V4L12_LUT, NG_HI, 2, false, WIN_REPLAY)

// --- Lever #2: row-scale hoist ----------------------------------------------
CENSUS_KERNEL(g1h_ng4, G_K4V2L16_CB, NG_LO, 2, true, WIN_REPLAY)
CENSUS_KERNEL(g1h_ng12, G_K4V2L16_CB, NG_HI, 2, true, WIN_REPLAY)
CENSUS_KERNEL(g3h_ng4, G_K8V4L12_LUT, NG_LO, 2, true, WIN_REPLAY)
CENSUS_KERNEL(g3h_ng12, G_K8V4L12_LUT, NG_HI, 2, true, WIN_REPLAY)

// --- Lever #3: random-access window in place of the warm-up replay -----------
CENSUS_KERNEL(g1w_ng4, G_K4V2L16_CB, NG_LO, 2, false, WIN_PRMT)
CENSUS_KERNEL(g1w_ng12, G_K4V2L16_CB, NG_HI, 2, false, WIN_PRMT)
CENSUS_KERNEL(g1b_ng4, G_K4V2L16_CB, NG_LO, 2, false, WIN_BREV)
CENSUS_KERNEL(g1b_ng12, G_K4V2L16_CB, NG_HI, 2, false, WIN_BREV)
CENSUS_KERNEL(g3w_ng4, G_K8V4L12_LUT, NG_LO, 2, false, WIN_PRMT)
CENSUS_KERNEL(g3w_ng12, G_K8V4L12_LUT, NG_HI, 2, false, WIN_PRMT)

// --- Both levers stacked on the best geometry --------------------------------
CENSUS_KERNEL(g3wh_ng4, G_K8V4L12_LUT, NG_LO, 2, true, WIN_PRMT)
CENSUS_KERNEL(g3wh_ng12, G_K8V4L12_LUT, NG_HI, 2, true, WIN_PRMT)

// ===========================================================================
// THE BIT-BUYING LADDER: K = 8 / 9 / 10 at fixed V=4, L=12.
//
// K=8 is the CONTROL and it must reproduce 5.375 (replay, no hoist) and 4.375
// (replay + row-scale hoist). If it does not, the rig is wrong and nothing else
// in this file is usable.
//
// Everything below holds V, L, the table, ROWS, WARM and GROUP_SYMS fixed
// (static_asserted above), so the ONLY difference between the K rungs is how a
// K-bit symbol comes out of the packed stream. Extraction count per group per
// row is WARM + GROUP_SYMS = 4 under WIN_REPLAY and GROUP_SYMS = 2 under
// WIN_SEQ, against 8 weights either way -- so a per-symbol penalty of D shows
// up as +0.50*D inst/weight in replay and +0.25*D in the serving shape.
// ===========================================================================

// --- WIN_REPLAY, no hoist: directly comparable to the g3 control -------------
CENSUS_KERNEL_EXT(g4_ng4, G_K9V4L12_LUT, NG_LO, 2, false, WIN_REPLAY, EXT_B2)
CENSUS_KERNEL_EXT(g4_ng8, G_K9V4L12_LUT, NG_MID, 2, false, WIN_REPLAY, EXT_B2)
CENSUS_KERNEL_EXT(g4_ng12, G_K9V4L12_LUT, NG_HI, 2, false, WIN_REPLAY, EXT_B2)
CENSUS_KERNEL_EXT(g5_ng4, G_K10V4L12_LUT, NG_LO, 2, false, WIN_REPLAY, EXT_B2)
CENSUS_KERNEL_EXT(g5_ng8, G_K10V4L12_LUT, NG_MID, 2, false, WIN_REPLAY, EXT_B2)
CENSUS_KERNEL_EXT(g5_ng12, G_K10V4L12_LUT, NG_HI, 2, false, WIN_REPLAY, EXT_B2)

// --- WIN_REPLAY + row-scale hoist: the 4.375-comparable row -------------------
CENSUS_KERNEL_EXT(g4h_ng4, G_K9V4L12_LUT, NG_LO, 2, true, WIN_REPLAY, EXT_B2)
CENSUS_KERNEL_EXT(g4h_ng12, G_K9V4L12_LUT, NG_HI, 2, true, WIN_REPLAY, EXT_B2)
CENSUS_KERNEL_EXT(g5h_ng4, G_K10V4L12_LUT, NG_LO, 2, true, WIN_REPLAY, EXT_B2)
CENSUS_KERNEL_EXT(g5h_ng12, G_K10V4L12_LUT, NG_HI, 2, true, WIN_REPLAY, EXT_B2)

// --- Extraction-route levers on K=9 (replay + hoist held fixed) ---------------
// EXT_B2C is the tail clamp a serving kernel needs when the row stride is NOT
// padded; the delta B2C - B2 is exactly what padding the stride buys back.
CENSUS_KERNEL_EXT(g4hc_ng4, G_K9V4L12_LUT, NG_LO, 2, true, WIN_REPLAY, EXT_B2C)
CENSUS_KERNEL_EXT(g4hc_ng12, G_K9V4L12_LUT, NG_HI, 2, true, WIN_REPLAY, EXT_B2C)
CENSUS_KERNEL_EXT(g4hf_ng4, G_K9V4L12_LUT, NG_LO, 2, true, WIN_REPLAY, EXT_FUN)
CENSUS_KERNEL_EXT(g4hf_ng12, G_K9V4L12_LUT, NG_HI, 2, true, WIN_REPLAY, EXT_FUN)
CENSUS_KERNEL_EXT(g4hs_ng4, G_K9V4L12_LUT, NG_LO, 2, true, WIN_REPLAY, EXT_SPLIT)
CENSUS_KERNEL_EXT(g4hs_ng12, G_K9V4L12_LUT, NG_HI, 2, true, WIN_REPLAY, EXT_SPLIT)
CENSUS_KERNEL_EXT(g5hf_ng4, G_K10V4L12_LUT, NG_LO, 2, true, WIN_REPLAY, EXT_FUN)
CENSUS_KERNEL_EXT(g5hf_ng12, G_K10V4L12_LUT, NG_HI, 2, true, WIN_REPLAY, EXT_FUN)
CENSUS_KERNEL_EXT(g5h3_ng4, G_K10V4L12_LUT, NG_LO, 2, true, WIN_REPLAY, EXT_B3)
CENSUS_KERNEL_EXT(g5h3_ng12, G_K10V4L12_LUT, NG_HI, 2, true, WIN_REPLAY, EXT_B3)

// --- WIN_SEQ: the shape the serving kernel actually has ----------------------
// Contiguous per-thread symbol slice, state carried, warm-up once in the
// prologue. This is qtip_gemv_k8v4l12.cu's inner loop, and it is where the
// K=9 decision is really taken.
CENSUS_KERNEL(q3_ng4, G_K8V4L12_LUT, NG_LO, 2, false, WIN_SEQ)
CENSUS_KERNEL(q3_ng8, G_K8V4L12_LUT, NG_MID, 2, false, WIN_SEQ)
CENSUS_KERNEL(q3_ng12, G_K8V4L12_LUT, NG_HI, 2, false, WIN_SEQ)
CENSUS_KERNEL(q3h_ng4, G_K8V4L12_LUT, NG_LO, 2, true, WIN_SEQ)
CENSUS_KERNEL(q3h_ng12, G_K8V4L12_LUT, NG_HI, 2, true, WIN_SEQ)

CENSUS_KERNEL_EXT(q4_ng4, G_K9V4L12_LUT, NG_LO, 2, false, WIN_SEQ, EXT_B2)
CENSUS_KERNEL_EXT(q4_ng8, G_K9V4L12_LUT, NG_MID, 2, false, WIN_SEQ, EXT_B2)
CENSUS_KERNEL_EXT(q4_ng12, G_K9V4L12_LUT, NG_HI, 2, false, WIN_SEQ, EXT_B2)
CENSUS_KERNEL_EXT(q4h_ng4, G_K9V4L12_LUT, NG_LO, 2, true, WIN_SEQ, EXT_B2)
CENSUS_KERNEL_EXT(q4h_ng12, G_K9V4L12_LUT, NG_HI, 2, true, WIN_SEQ, EXT_B2)
CENSUS_KERNEL_EXT(q4hc_ng4, G_K9V4L12_LUT, NG_LO, 2, true, WIN_SEQ, EXT_B2C)
CENSUS_KERNEL_EXT(q4hc_ng12, G_K9V4L12_LUT, NG_HI, 2, true, WIN_SEQ, EXT_B2C)
CENSUS_KERNEL_EXT(q4hf_ng4, G_K9V4L12_LUT, NG_LO, 2, true, WIN_SEQ, EXT_FUN)
CENSUS_KERNEL_EXT(q4hf_ng12, G_K9V4L12_LUT, NG_HI, 2, true, WIN_SEQ, EXT_FUN)
CENSUS_KERNEL_EXT(q4hs_ng4, G_K9V4L12_LUT, NG_LO, 2, true, WIN_SEQ, EXT_SPLIT)
CENSUS_KERNEL_EXT(q4hs_ng12, G_K9V4L12_LUT, NG_HI, 2, true, WIN_SEQ, EXT_SPLIT)

CENSUS_KERNEL_EXT(q5_ng4, G_K10V4L12_LUT, NG_LO, 2, false, WIN_SEQ, EXT_B2)
CENSUS_KERNEL_EXT(q5_ng8, G_K10V4L12_LUT, NG_MID, 2, false, WIN_SEQ, EXT_B2)
CENSUS_KERNEL_EXT(q5_ng12, G_K10V4L12_LUT, NG_HI, 2, false, WIN_SEQ, EXT_B2)
CENSUS_KERNEL_EXT(q5h_ng4, G_K10V4L12_LUT, NG_LO, 2, true, WIN_SEQ, EXT_B2)
CENSUS_KERNEL_EXT(q5h_ng12, G_K10V4L12_LUT, NG_HI, 2, true, WIN_SEQ, EXT_B2)
CENSUS_KERNEL_EXT(q5hf_ng4, G_K10V4L12_LUT, NG_LO, 2, true, WIN_SEQ, EXT_FUN)
CENSUS_KERNEL_EXT(q5hf_ng12, G_K10V4L12_LUT, NG_HI, 2, true, WIN_SEQ, EXT_FUN)

// ===========================================================================
// EXTRACTION ISOLATION MICRO-CENSUS
//
// The tables above give inst/WEIGHT with the extraction cost folded inside a
// full decode+FMA body. The decision turns on the extraction ALONE, so this
// section strips everything else away: no table, no codebook, no FMA, no
// shared memory -- just "pull symbol t out of the row" repeated NS times,
// XOR-accumulated so nothing is dead-code eliminated.
//
// The XOR costs exactly one instruction per symbol in EVERY variant, so it
// cancels in the K=9-minus-K=8 delta, which is the number being isolated.
// `base` is a runtime value (as in the real kernel), so address arithmetic is
// paid, not folded into an immediate.
//
// Differencing NS = 8 -> 24 gives inst/SYMBOL. NS=16 is the linearity check.
// The largest body here is ~24 x 12 inst, two orders below the 4096 unroller
// ceiling that invalidated the first census.
// ===========================================================================
template <int K, int EXT, int NS>
__device__ __forceinline__ void extract_body(const uint8_t* __restrict__ packed,
                                             uint32_t* __restrict__ out, int ppr) {
    const uint8_t* p = packed + (size_t)blockIdx.x * ppr;
    const int base = (int)threadIdx.x * NS;
    uint32_t acc = 0u;
#pragma unroll
    for (int s = 0; s < NS; ++s) {
        acc ^= census_sym_ext<K, EXT>(p, base + s, ppr);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

#define EXTRACT_KERNEL(NAME, K, EXT, NS)                                       \
    extern "C" __global__ void __launch_bounds__(256)                          \
        extract_##NAME(const uint8_t* __restrict__ packed,                     \
                       uint32_t* __restrict__ out, int ppr) {                  \
        extract_body<K, EXT, NS>(packed, out, ppr);                            \
    }

#define NS_LO 8
#define NS_MID 16
#define NS_HI 24

// K=8, byte-aligned -- the control every delta is taken against.
EXTRACT_KERNEL(k8_ns8, 8, EXT_NATIVE, NS_LO)
EXTRACT_KERNEL(k8_ns16, 8, EXT_NATIVE, NS_MID)
EXTRACT_KERNEL(k8_ns24, 8, EXT_NATIVE, NS_HI)

// K=4, nibble -- the shipped rung's extract, for scale.
EXTRACT_KERNEL(k4_ns8, 4, EXT_NATIVE, NS_LO)
EXTRACT_KERNEL(k4_ns16, 4, EXT_NATIVE, NS_MID)
EXTRACT_KERNEL(k4_ns24, 4, EXT_NATIVE, NS_HI)

// K=9, four routes.
EXTRACT_KERNEL(k9b2_ns8, 9, EXT_B2, NS_LO)
EXTRACT_KERNEL(k9b2_ns16, 9, EXT_B2, NS_MID)
EXTRACT_KERNEL(k9b2_ns24, 9, EXT_B2, NS_HI)
EXTRACT_KERNEL(k9b2c_ns8, 9, EXT_B2C, NS_LO)
EXTRACT_KERNEL(k9b2c_ns16, 9, EXT_B2C, NS_MID)
EXTRACT_KERNEL(k9b2c_ns24, 9, EXT_B2C, NS_HI)
EXTRACT_KERNEL(k9fun_ns8, 9, EXT_FUN, NS_LO)
EXTRACT_KERNEL(k9fun_ns16, 9, EXT_FUN, NS_MID)
EXTRACT_KERNEL(k9fun_ns24, 9, EXT_FUN, NS_HI)
EXTRACT_KERNEL(k9split_ns8, 9, EXT_SPLIT, NS_LO)
EXTRACT_KERNEL(k9split_ns16, 9, EXT_SPLIT, NS_MID)
EXTRACT_KERNEL(k9split_ns24, 9, EXT_SPLIT, NS_HI)

// K=10, four routes (B3 is the defensive generic read).
EXTRACT_KERNEL(k10b2_ns8, 10, EXT_B2, NS_LO)
EXTRACT_KERNEL(k10b2_ns16, 10, EXT_B2, NS_MID)
EXTRACT_KERNEL(k10b2_ns24, 10, EXT_B2, NS_HI)
EXTRACT_KERNEL(k10b3_ns8, 10, EXT_B3, NS_LO)
EXTRACT_KERNEL(k10b3_ns16, 10, EXT_B3, NS_MID)
EXTRACT_KERNEL(k10b3_ns24, 10, EXT_B3, NS_HI)
EXTRACT_KERNEL(k10fun_ns8, 10, EXT_FUN, NS_LO)
EXTRACT_KERNEL(k10fun_ns16, 10, EXT_FUN, NS_MID)
EXTRACT_KERNEL(k10fun_ns24, 10, EXT_FUN, NS_HI)
EXTRACT_KERNEL(k10split_ns8, 10, EXT_SPLIT, NS_LO)
EXTRACT_KERNEL(k10split_ns16, 10, EXT_SPLIT, NS_MID)
EXTRACT_KERNEL(k10split_ns24, 10, EXT_SPLIT, NS_HI)
