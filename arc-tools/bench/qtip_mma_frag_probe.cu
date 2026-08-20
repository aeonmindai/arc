// ArcQuant / ArcKernels — TCFRAG go/no-go probe.
//
// THESIS UNDER TEST
// -----------------
// The QTIP `sum2` computed codebook produces, in ONE 32-bit register, the value
//     m = (state * mult) & 0x8FFF8FFF ^ 0x3B603B60
// whose two 16-bit halves are *already fp16 numbers* and whose codeword is
// their SUM. The shipped GEMV then spends 2 cvt + 1 FADD + 1 FMUL(scale) +
// 2 FFMA per weight turning that register into an f32 dot-product term.
//
// But a dot product is linear:   sum_k (hi_k + lo_k) * x_k
//                              = sum_k hi_k * x_k + sum_k lo_k * x_k
// so if the two halves are given ADJACENT k-slots of an `mma.m16n8k16`
// fragment and the activation is duplicated across those two slots, the tensor
// core performs the fold, the scale-multiply and the k-reduction itself.
//
// => the LOP3 destination register IS the mma operand register, verbatim.
//    Nothing between the decode and the HMMA. That is what this file measures.
//
// LAYOUT (see the spec block below `pack_rev`) — two bake-time decisions:
//   1. NIBBLE-REVERSED ROW ORDER, so state(s) is a plain funnel-shift of two
//      adjacent u32 words with no __brev / nibble-swap at read time.
//   2. CODEWORD-PAIR -> (a0,a2) / (a1,a3) k-slot assignment, so the V=2 pair a
//      single state produces lands in the two halves of one thread's A
//      fragment without any cross-lane movement.
//
// Build:
//   /usr/local/cuda-12.4/bin/nvcc -arch=sm_90a -O3 --use_fast_math -lineinfo \
//       -o qtip_mma_frag_probe qtip_mma_frag_probe.cu
// SASS:
//   /usr/local/cuda-12.4/bin/cuobjdump -sass qtip_mma_frag_probe

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @ %s:%d -> %s\n", #x, __FILE__, __LINE__, \
            cudaGetErrorString(e_)); exit(2); } } while (0)

// ---------------------------------------------------------------------------
// Format constants (mirror mistralrs-quant/src/qtip/mod.rs + qtip_codebook.cuh)
// ---------------------------------------------------------------------------
static const uint32_t MCG_MULT = 0xCAF6A435u;   // QTIP_MCG_V2_MULT
#define CB_MASK 0x8FFF8FFFu
#define CB_XOR  0x3B603B60u
static const int K_BITS = 4;   // symbol width
static const int V      = 2;   // values per symbol
static const int L_BITS = 16;  // trellis state width

// ---------------------------------------------------------------------------
// Host fp16 (no device intrinsics on host).
// ---------------------------------------------------------------------------
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) & 1u;
    uint32_t e = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)h & 0x3FFu;
    uint32_t out;
    if (e == 0) {
        if (m == 0) { out = s << 31; }
        else {                       // subnormal -> normalize
            e = 1;
            while ((m & 0x400u) == 0) { m <<= 1; e--; }
            m &= 0x3FFu;
            out = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
        }
    } else if (e == 31) {
        out = (s << 31) | 0x7F800000u | (m << 13);
    } else {
        out = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
    }
    float f; memcpy(&f, &out, 4); return f;
}
static uint16_t f2h(float f) {   // round-to-nearest-even, enough for a probe
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000u;
    int32_t  e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7C00u);
    uint32_t h = s | ((uint32_t)e << 10) | (m >> 13);
    uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t)h;
}

// The two fp16 halves of the codeword register, as the GPU sees them.
static void cb_halves(uint32_t x, float* hi, float* lo) {
    uint32_t m = (x & CB_MASK) ^ CB_XOR;
    *hi = h2f((uint16_t)(m >> 16));
    *lo = h2f((uint16_t)(m & 0xFFFFu));
}

// ---------------------------------------------------------------------------
// ===========================  T C F R A G   S P E C  =======================
//
// A row of `S` symbols (4 bits each, S % 8 == 0) is stored as W = S/8 words of
// 32 bits.  Word j holds symbols 8j..8j+7 in REVERSED nibble order:
//
//     R[j] bits [4c .. 4c+3]  =  sym[8j + (7 - c)]        c = 0..7
//
// Equivalently: concatenating (R[j-1] << 32 | R[j]) gives a continuous
// descending-symbol nibble stream, so the 16-bit trellis state at symbol s,
//
//     state(s) = sym[s-3]<<12 | sym[s-2]<<8 | sym[s-1]<<4 | sym[s]
//
// (the shipped `state = ((state<<4)|sym) & 0xFFFF` recurrence) is EXACTLY
//
//     state(s) = __funnelshift_r(R[j], R[j-1], 4*(7 - (s & 7))) & 0xFFFF
//                with j = s >> 3, R[-1] = 0
//
// i.e. ONE SHF + ONE LOP3, with no __brev, no nibble swap, no byte swap. The
// shipped format needs `gg_state_from_window` (4 ops) on top of the load
// because it stores nibbles ASCENDING while the state reads them DESCENDING.
//
// HOW A READER VALIDATES IT: unpack_rev() below is the inverse; the property
// that must hold for any symbol stream is
//     state_recurrence(s) == funnel_state(R, s)  for all s in [0, S)
// which `validate_format()` checks exhaustively on random streams. This is a
// pure bit permutation of the existing artifact -- the decoded weights are
// bit-identical, so it can be applied at LOAD time with no re-bake.
//
// K-SLOT ASSIGNMENT (the half that a kernel cannot choose on its own):
//   For mma.m16n8k16 the A fragment of lane `t` covers, with gid=t>>2, i=t&3,
//       a0 <- (row gid  , k 2i  , 2i+1)      a2 <- (row gid  , k 2i+8, 2i+9)
//       a1 <- (row gid+8, k 2i  , 2i+1)      a3 <- (row gid+8, k 2i+8, 2i+9)
//   We define logical weight index -> k-slot as
//       weight 2s   (= codeword 0 of symbol s) -> k-slots 2i  , 2i+1
//       weight 2s+1 (= codeword 1 of symbol s) -> k-slots 2i+8, 2i+9
//   with s = 8j + i for even mma steps and s = 8j + 4 + i for odd ones.
//   Both halves of a k-slot pair receive the SAME activation, duplicated:
//       B[2i][n] = B[2i+1][n] = x[2s]      B[2i+8][n] = B[2i+9][n] = x[2s+1]
//   so the tensor core computes hi*x + lo*x == codeword*x.
//   Consequence: a0/a2 are the two LOP3 outputs of ONE state, a1/a3 the two of
//   the neighbouring row's state at the same s. No lane ever needs a value
//   another lane produced => zero MOV / PRMT / SHFL between decode and HMMA.
// ===========================================================================
static void pack_rev(const uint8_t* sym, int S, uint32_t* R) {
    int W = S / 8;
    for (int j = 0; j < W; ++j) {
        uint32_t w = 0;
        for (int c = 0; c < 8; ++c) w |= (uint32_t)(sym[8 * j + (7 - c)] & 0xF) << (4 * c);
        R[j] = w;
    }
}
static void unpack_rev(const uint32_t* R, int S, uint8_t* sym) {
    int W = S / 8;
    for (int j = 0; j < W; ++j)
        for (int c = 0; c < 8; ++c) sym[8 * j + (7 - c)] = (uint8_t)((R[j] >> (4 * c)) & 0xF);
}
static uint32_t funnel_state_host(const uint32_t* R, int S, int s) {
    int j = s >> 3;
    uint32_t lo = R[j];
    uint32_t hi = (j > 0) ? R[j - 1] : 0u;
    int sh = 4 * (7 - (s & 7));
    uint64_t cat = ((uint64_t)hi << 32) | (uint64_t)lo;
    return (uint32_t)((cat >> sh) & 0xFFFFull);
}
static void validate_format(int S, int trials) {
    std::mt19937 rng(1234);
    std::vector<uint8_t> sym(S), sym2(S);
    std::vector<uint32_t> R(S / 8);
    for (int t = 0; t < trials; ++t) {
        for (int i = 0; i < S; ++i) sym[i] = (uint8_t)(rng() & 0xF);
        pack_rev(sym.data(), S, R.data());
        unpack_rev(R.data(), S, sym2.data());
        if (memcmp(sym.data(), sym2.data(), S) != 0) { printf("FORMAT: pack/unpack mismatch\n"); exit(1); }
        uint32_t st = 0;
        for (int s = 0; s < S; ++s) {
            st = ((st << K_BITS) | sym[s]) & ((1u << L_BITS) - 1u);
            uint32_t f = funnel_state_host(R.data(), S, s);
            if (f != st) { printf("FORMAT: state mismatch at s=%d: recurrence %04x funnel %04x\n", s, st, f); exit(1); }
        }
    }
    printf("[format] TCFRAG nibble-reversal validated: %d trials x %d symbols, "
           "funnel-shift state == sequential recurrence everywhere.\n", trials, S);
}

// ===========================================================================
// DEVICE
// ===========================================================================

// ---- mma.m16n8k16.row.col.f32.f16.f16.f32 -------------------------------
__device__ __forceinline__ void mma16816(float* acc, uint32_t a0, uint32_t a1,
                                         uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------
// THE KERNEL UNDER TEST.
// One warp per 16-row block (x NB blocks per warp), full k.
// Per j-iteration each THREAD decodes 8 weights (2 rows x 2 mma steps x V=2),
// so per-thread-instructions / 8 is directly comparable to the shipped 26.31.
// ---------------------------------------------------------------------------
template <int NB, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
tcfrag_gemv(const uint32_t* __restrict__ R,      // [n_rows][W] reversed words
            const float*    __restrict__ scales, // [n_rows]
            const __half*   __restrict__ x,      // [2*S]
            float*          __restrict__ y,      // [n_rows]
            int n_rows, int W, int ksplit, uint32_t mult) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gid  = lane >> 2;      // A row group (0..7) / B column
    const int i    = lane & 3;       // k-slot group

    const int rb0 = ((blockIdx.x * WARPS) + warp) * NB * 16;
    if (rb0 >= n_rows) return;

    // k-split: grid.y slices the word axis. State continuity is preserved by
    // seeding `prv` from the real predecessor word, not from zero.
    const int wper = (W + ksplit - 1) / ksplit;
    const int j0   = blockIdx.y * wper;
    const int j1   = min(W, j0 + wper);
    if (j0 >= j1) return;

    float acc[NB][4];
#pragma unroll
    for (int b = 0; b < NB; ++b) { acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f; }

    const int sh0 = 4 * (7 - i);        // symbol 8j + i
    const int sh1 = 4 * (3 - i);        // symbol 8j + 4 + i

    const uint32_t* Rb[NB][2];
    uint32_t prv[NB][2];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        Rb[b][0] = R + (size_t)(rb0 + b * 16 + gid)     * W;
        Rb[b][1] = R + (size_t)(rb0 + b * 16 + gid + 8) * W;
        prv[b][0] = j0 ? __ldg(Rb[b][0] + j0 - 1) : 0u;
        prv[b][1] = j0 ? __ldg(Rb[b][1] + j0 - 1) : 0u;
    }

#pragma unroll 2
    for (int j = j0; j < j1; ++j) {
        // ---- B fragments: activation, duplicated across each k-slot pair.
        //      x index for step 0 is 2*(8j+i) = 16j+2i ; step 1 is 16j+8+2i.
        const uint32_t xa = *reinterpret_cast<const uint32_t*>(x + 16 * j + 2 * i);
        const uint32_t xb = *reinterpret_cast<const uint32_t*>(x + 16 * j + 8 + 2 * i);
        const uint32_t b0_0 = __byte_perm(xa, xa, 0x1010);   // dup low  half
        const uint32_t b1_0 = __byte_perm(xa, xa, 0x3232);   // dup high half
        const uint32_t b0_1 = __byte_perm(xb, xb, 0x1010);
        const uint32_t b1_1 = __byte_perm(xb, xb, 0x3232);

#pragma unroll
        for (int b = 0; b < NB; ++b) {
            // ---- state extraction: 1 SHF + 1 LOP3 each, no reversal.
            const uint32_t c0 = Rb[b][0][j], p0 = prv[b][0];
            const uint32_t c1 = Rb[b][1][j], p1 = prv[b][1];
            prv[b][0] = c0; prv[b][1] = c1;
            const uint32_t s00 = __funnelshift_r(c0, p0, sh0) & 0xFFFFu;
            const uint32_t s10 = __funnelshift_r(c1, p1, sh0) & 0xFFFFu;
            const uint32_t s01 = __funnelshift_r(c0, p0, sh1) & 0xFFFFu;
            const uint32_t s11 = __funnelshift_r(c1, p1, sh1) & 0xFFFFu;

            // ---- decode: IMAD, LOP3 -> the register IS the fragment.
            uint32_t t;
            t = s00 * mult; const uint32_t A00 = (t & CB_MASK) ^ CB_XOR;
            t = t * mult;   const uint32_t A02 = (t & CB_MASK) ^ CB_XOR;
            t = s10 * mult; const uint32_t A01 = (t & CB_MASK) ^ CB_XOR;
            t = t * mult;   const uint32_t A03 = (t & CB_MASK) ^ CB_XOR;
            mma16816(acc[b], A00, A01, A02, A03, b0_0, b1_0);

            t = s01 * mult; const uint32_t A10 = (t & CB_MASK) ^ CB_XOR;
            t = t * mult;   const uint32_t A12 = (t & CB_MASK) ^ CB_XOR;
            t = s11 * mult; const uint32_t A11 = (t & CB_MASK) ^ CB_XOR;
            t = t * mult;   const uint32_t A13 = (t & CB_MASK) ^ CB_XOR;
            mma16816(acc[b], A10, A11, A12, A13, b0_1, b1_1);
        }
    }

    // D column 0 lives in threads with i == 0: c0 = D[gid][0], c2 = D[gid+8][0].
    if (i == 0) {
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int r0 = rb0 + b * 16 + gid, r1 = r0 + 8;
            if (ksplit == 1) {
                if (r0 < n_rows) y[r0] = acc[b][0] * scales[r0];
                if (r1 < n_rows) y[r1] = acc[b][2] * scales[r1];
            } else {
                if (r0 < n_rows) atomicAdd(&y[r0], acc[b][0] * scales[r0]);
                if (r1 < n_rows) atomicAdd(&y[r1], acc[b][2] * scales[r1]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v2: two fixes the v1 SASS exposed, both pure kernel-side.
//
//  (a) `(t & 0x8FFF8FFF) ^ 0x3B603B60` compiled to TWO LOP3 because both
//      constants were immediates and a SASS LOP3 has only one immediate slot.
//      Holding the mask in a register makes it ONE LOP3 (lut 0x6c = (a&c)^b).
//      Measured: 160 -> 96 LOP3 per loop iteration.
//  (b) the R words were 16 scattered LDG.E.32 per iteration (4 useful bytes
//      per 32 B sector). One LDG.E.128 per row per 4 words instead.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t cb_frag(uint32_t t, uint32_t maskreg) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0x6c;" : "=r"(d) : "r"(t), "n"(CB_XOR), "r"(maskreg));
    return d;
}

template <int NB, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
tcfrag_gemv_v2(const uint32_t* __restrict__ R, const float* __restrict__ scales,
               const __half* __restrict__ x, float* __restrict__ y,
               int n_rows, int W, int ksplit, uint32_t mult) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gid  = lane >> 2;
    const int i    = lane & 3;

    const int rb0 = ((blockIdx.x * WARPS) + warp) * NB * 16;
    if (rb0 >= n_rows) return;
    const int wper = ((W / 4 + ksplit - 1) / ksplit) * 4;   // keep uint4 alignment
    const int j0 = blockIdx.y * wper, j1 = min(W, j0 + wper);
    if (j0 >= j1) return;

    uint32_t MREG;
    asm("mov.b32 %0, %1;" : "=r"(MREG) : "n"(CB_MASK));

    float acc[NB][4];
#pragma unroll
    for (int b = 0; b < NB; ++b) { acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f; }
    const int sh0 = 4 * (7 - i), sh1 = 4 * (3 - i);

    const uint4* Rb[NB][2];
    uint32_t prv[NB][2];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        const uint32_t* r0 = R + (size_t)(rb0 + b * 16 + gid) * W;
        const uint32_t* r1 = R + (size_t)(rb0 + b * 16 + gid + 8) * W;
        Rb[b][0] = reinterpret_cast<const uint4*>(r0);
        Rb[b][1] = reinterpret_cast<const uint4*>(r1);
        prv[b][0] = j0 ? __ldg(r0 + j0 - 1) : 0u;
        prv[b][1] = j0 ? __ldg(r1 + j0 - 1) : 0u;
    }

    for (int jb = j0; jb < j1; jb += 4) {
        uint4 v0[NB], v1[NB];
#pragma unroll
        for (int b = 0; b < NB; ++b) { v0[b] = __ldg(Rb[b][0] + (jb >> 2)); v1[b] = __ldg(Rb[b][1] + (jb >> 2)); }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int j = jb + u;
            const uint32_t xa = *reinterpret_cast<const uint32_t*>(x + 16 * j + 2 * i);
            const uint32_t xb = *reinterpret_cast<const uint32_t*>(x + 16 * j + 8 + 2 * i);
            const uint32_t b0_0 = __byte_perm(xa, xa, 0x1010), b1_0 = __byte_perm(xa, xa, 0x3232);
            const uint32_t b0_1 = __byte_perm(xb, xb, 0x1010), b1_1 = __byte_perm(xb, xb, 0x3232);
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                const uint32_t c0 = (u == 0) ? v0[b].x : (u == 1) ? v0[b].y : (u == 2) ? v0[b].z : v0[b].w;
                const uint32_t c1 = (u == 0) ? v1[b].x : (u == 1) ? v1[b].y : (u == 2) ? v1[b].z : v1[b].w;
                const uint32_t p0 = prv[b][0], p1 = prv[b][1];
                prv[b][0] = c0; prv[b][1] = c1;
                const uint32_t s00 = __funnelshift_r(c0, p0, sh0) & 0xFFFFu;
                const uint32_t s10 = __funnelshift_r(c1, p1, sh0) & 0xFFFFu;
                const uint32_t s01 = __funnelshift_r(c0, p0, sh1) & 0xFFFFu;
                const uint32_t s11 = __funnelshift_r(c1, p1, sh1) & 0xFFFFu;
                uint32_t t;
                t = s00 * mult; const uint32_t A00 = cb_frag(t, MREG);
                t = t * mult;   const uint32_t A02 = cb_frag(t, MREG);
                t = s10 * mult; const uint32_t A01 = cb_frag(t, MREG);
                t = t * mult;   const uint32_t A03 = cb_frag(t, MREG);
                mma16816(acc[b], A00, A01, A02, A03, b0_0, b1_0);
                t = s01 * mult; const uint32_t A10 = cb_frag(t, MREG);
                t = t * mult;   const uint32_t A12 = cb_frag(t, MREG);
                t = s11 * mult; const uint32_t A11 = cb_frag(t, MREG);
                t = t * mult;   const uint32_t A13 = cb_frag(t, MREG);
                mma16816(acc[b], A10, A11, A12, A13, b0_1, b1_1);
            }
        }
    }

    if (i == 0) {
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int r0 = rb0 + b * 16 + gid, r1 = r0 + 8;
            if (ksplit == 1) {
                if (r0 < n_rows) y[r0] = acc[b][0] * scales[r0];
                if (r1 < n_rows) y[r1] = acc[b][2] * scales[r1];
            } else {
                if (r0 < n_rows) atomicAdd(&y[r0], acc[b][0] * scales[r0]);
                if (r1 < n_rows) atomicAdd(&y[r1], acc[b][2] * scales[r1]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v3: the ACTIVATION half of the layout.
//
// The B fragment needs each activation duplicated across the two k-slots that
// carry a codeword's (hi, lo) halves. v2 did that at runtime with 2 PRMT per
// fragment pair (0.50 inst/weight at NB=1 -- the single largest non-essential
// term left). Instead, write the activation ONCE in duplicated order:
//     xd[2t] = xd[2t+1] = x[t]
// which is free inside the D.H.D rotation kernel that already materialises
// x_rotated, and costs 2x 14 KB of scratch for k=7168.
//
// Then b0 and b1 for one mma step are ADJACENT 32-bit words, so the whole B
// fragment is a single LDG.E.64: 2 loads and 0 PRMT per step, replacing
// 2 loads + 4 PRMT.
// ---------------------------------------------------------------------------
template <int NB, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
tcfrag_gemv_v3(const uint32_t* __restrict__ R, const float* __restrict__ scales,
               const __half* __restrict__ xd,   // [4*S] duplicated activation
               float* __restrict__ y, int n_rows, int W, int ksplit, uint32_t mult) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gid  = lane >> 2;
    const int i    = lane & 3;
    const int rb0 = ((blockIdx.x * WARPS) + warp) * NB * 16;
    if (rb0 >= n_rows) return;
    const int wper = ((W / 4 + ksplit - 1) / ksplit) * 4;
    const int j0 = blockIdx.y * wper, j1 = min(W, j0 + wper);
    if (j0 >= j1) return;

    uint32_t MREG;
    asm("mov.b32 %0, %1;" : "=r"(MREG) : "n"(CB_MASK));
    float acc[NB][4];
#pragma unroll
    for (int b = 0; b < NB; ++b) { acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f; }
    const int sh0 = 4 * (7 - i), sh1 = 4 * (3 - i);

    const uint4* Rb[NB][2];
    uint32_t prv[NB][2];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        const uint32_t* r0 = R + (size_t)(rb0 + b * 16 + gid) * W;
        const uint32_t* r1 = R + (size_t)(rb0 + b * 16 + gid + 8) * W;
        Rb[b][0] = reinterpret_cast<const uint4*>(r0);
        Rb[b][1] = reinterpret_cast<const uint4*>(r1);
        prv[b][0] = j0 ? __ldg(r0 + j0 - 1) : 0u;
        prv[b][1] = j0 ? __ldg(r1 + j0 - 1) : 0u;
    }

    for (int jb = j0; jb < j1; jb += 4) {
        uint4 v0[NB], v1[NB];
#pragma unroll
        for (int b = 0; b < NB; ++b) { v0[b] = __ldg(Rb[b][0] + (jb >> 2)); v1[b] = __ldg(Rb[b][1] + (jb >> 2)); }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int j = jb + u;
            // symbol s = 8j + i  -> duplicated-x offset 4s = 32j + 4i
            // symbol s = 8j+4+i  -> 4s = 32j + 16 + 4i
            const uint2 B0 = __ldg(reinterpret_cast<const uint2*>(xd + 32 * j + 4 * i));
            const uint2 B1 = __ldg(reinterpret_cast<const uint2*>(xd + 32 * j + 16 + 4 * i));
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                const uint32_t c0 = (u == 0) ? v0[b].x : (u == 1) ? v0[b].y : (u == 2) ? v0[b].z : v0[b].w;
                const uint32_t c1 = (u == 0) ? v1[b].x : (u == 1) ? v1[b].y : (u == 2) ? v1[b].z : v1[b].w;
                const uint32_t p0 = prv[b][0], p1 = prv[b][1];
                prv[b][0] = c0; prv[b][1] = c1;
                const uint32_t s00 = __funnelshift_r(c0, p0, sh0) & 0xFFFFu;
                const uint32_t s10 = __funnelshift_r(c1, p1, sh0) & 0xFFFFu;
                const uint32_t s01 = __funnelshift_r(c0, p0, sh1) & 0xFFFFu;
                const uint32_t s11 = __funnelshift_r(c1, p1, sh1) & 0xFFFFu;
                uint32_t t;
                t = s00 * mult; const uint32_t A00 = cb_frag(t, MREG);
                t = t * mult;   const uint32_t A02 = cb_frag(t, MREG);
                t = s10 * mult; const uint32_t A01 = cb_frag(t, MREG);
                t = t * mult;   const uint32_t A03 = cb_frag(t, MREG);
                mma16816(acc[b], A00, A01, A02, A03, B0.x, B0.y);
                t = s01 * mult; const uint32_t A10 = cb_frag(t, MREG);
                t = t * mult;   const uint32_t A12 = cb_frag(t, MREG);
                t = s11 * mult; const uint32_t A11 = cb_frag(t, MREG);
                t = t * mult;   const uint32_t A13 = cb_frag(t, MREG);
                mma16816(acc[b], A10, A11, A12, A13, B1.x, B1.y);
            }
        }
    }
    if (i == 0) {
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int r0 = rb0 + b * 16 + gid, r1 = r0 + 8;
            if (ksplit == 1) {
                if (r0 < n_rows) y[r0] = acc[b][0] * scales[r0];
                if (r1 < n_rows) y[r1] = acc[b][2] * scales[r1];
            } else {
                if (r0 < n_rows) atomicAdd(&y[r0], acc[b][0] * scales[r0]);
                if (r1 < n_rows) atomicAdd(&y[r1], acc[b][2] * scales[r1]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reference: the shipped scalar decode shape (sequential state, f32 fold,
// per-weight scale FMUL, FFMA accumulate). Used for the A/B time number.
// Reads the SAME reversed words so both kernels touch identical bytes.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
scalar_gemv(const uint32_t* __restrict__ R, const float* __restrict__ scales,
            const __half* __restrict__ x, float* __restrict__ y,
            int n_rows, int W, uint32_t mult) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (row >= n_rows) return;
    const int lane = threadIdx.x & 31;
    const uint32_t* Rr = R + (size_t)row * W;
    const float sc = scales[row];
    float a = 0.f;
    for (int j = lane; j < W; j += 32) {
        const uint32_t c = Rr[j], p = j ? Rr[j - 1] : 0u;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const uint32_t st = __funnelshift_r(c, p, 4 * (7 - u)) & 0xFFFFu;
            uint32_t t0 = st * mult;
            uint32_t m0 = (t0 & CB_MASK) ^ CB_XOR;
            uint32_t t1 = t0 * mult;
            uint32_t m1 = (t1 & CB_MASK) ^ CB_XOR;
            const float w0 = __half2float(__ushort_as_half((unsigned short)(m0 >> 16))) +
                             __half2float(__ushort_as_half((unsigned short)(m0 & 0xFFFFu)));
            const float w1 = __half2float(__ushort_as_half((unsigned short)(m1 >> 16))) +
                             __half2float(__ushort_as_half((unsigned short)(m1 & 0xFFFFu)));
            const int s = 8 * j + u;
            a = fmaf(w0 * sc, __half2float(x[2 * s]), a);
            a = fmaf(w1 * sc, __half2float(x[2 * s + 1]), a);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, off);
    if (lane == 0) y[row] = a;
}

// ===========================================================================
int main(int argc, char** argv) {
    int n_rows = (argc > 1) ? atoi(argv[1]) : 4096;
    int S      = (argc > 2) ? atoi(argv[2]) : 3584;   // symbols per row (k = 2*S)
    int iters  = (argc > 3) ? atoi(argv[3]) : 200;
    if (S % 8) { printf("S must be a multiple of 8\n"); return 1; }
    if (n_rows % 64) { printf("n_rows must be a multiple of 64\n"); return 1; }
    const int W = S / 8, kdim = V * S;

    validate_format(64, 64);

    std::mt19937 rng(7);
    std::vector<uint8_t>  sym((size_t)n_rows * S);
    std::vector<uint32_t> R((size_t)n_rows * W);
    std::vector<float>    scales(n_rows);
    std::vector<uint16_t> xh(kdim);
    for (auto& v : sym) v = (uint8_t)(rng() & 0xF);
    for (int r = 0; r < n_rows; ++r) pack_rev(&sym[(size_t)r * S], S, &R[(size_t)r * W]);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int r = 0; r < n_rows; ++r) scales[r] = 0.02f + 0.001f * (r % 17);
    for (int t = 0; t < kdim; ++t) xh[t] = f2h(nd(rng));

    // ---- QUALITY INSTRUMENT -------------------------------------------------
    // The decoded WEIGHTS are bit-identical (TCFRAG is a pure permutation of
    // the stored bytes). What changes is the arithmetic:
    //   shipped : w = f32(hi)+f32(lo); acc += (w*scale) * f32(bf16 x)
    //   tcfrag  : acc += fp16(hi)*fp16(x) + fp16(lo)*fp16(x)  [f32 acc], *scale
    // so the ONLY quality question is the activation dtype: shipped feeds the
    // FMA a bf16 activation (8-bit mantissa), tcfrag feeds the tensor core an
    // fp16 one (11-bit mantissa, but 6e-5..65504 range). Both are compared
    // below against an exact-f32-activation reference on the same weights.
    std::vector<float> xf(kdim);
    { std::mt19937 r2(99); std::normal_distribution<float> n2(0.f, 1.f);
      for (int t = 0; t < kdim; ++t) xf[t] = n2(r2); }
    for (int t = 0; t < kdim; ++t) xh[t] = f2h(xf[t]);   // dX is filled from xh below
    auto to_bf16 = [](float f) { uint32_t u; memcpy(&u, &f, 4);
        uint32_t r = (u >> 16) & 1u, b = u + 0x7FFFu + r; b &= 0xFFFF0000u;
        float o; memcpy(&o, &b, 4); return o; };
    std::vector<double> q_exact(n_rows), q_bf16(n_rows), q_fp16(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        uint32_t st = 0; double e = 0, b = 0, p = 0;
        for (int s = 0; s < S; ++s) {
            st = ((st << K_BITS) | sym[(size_t)r * S + s]) & 0xFFFFu;
            uint32_t t0 = st * MCG_MULT, t1 = t0 * MCG_MULT;
            float hi0, lo0, hi1, lo1;
            cb_halves(t0, &hi0, &lo0); cb_halves(t1, &hi1, &lo1);
            const float w0 = hi0 + lo0, w1 = hi1 + lo1;
            e += (double)w0 * xf[2*s]              + (double)w1 * xf[2*s+1];
            b += (double)w0 * to_bf16(xf[2*s])     + (double)w1 * to_bf16(xf[2*s+1]);
            p += (double)w0 * h2f(xh[2*s])         + (double)w1 * h2f(xh[2*s+1]);
        }
        q_exact[r] = e * scales[r]; q_bf16[r] = b * scales[r]; q_fp16[r] = p * scales[r];
    }
    auto rel_l2 = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double n = 0, d = 0;
        for (int r = 0; r < n_rows; ++r) { double e = a[r] - b[r]; n += e * e; d += b[r] * b[r]; }
        return sqrt(n / (d + 1e-30));
    };
    printf("[quality] activation dtype vs exact-f32 activation, same weights, %d rows x k=%d:\n", n_rows, kdim);
    printf("[quality]   shipped path (bf16 activation) rel-L2 = %.3e\n", rel_l2(q_bf16, q_exact));
    printf("[quality]   tcfrag path  (fp16 activation) rel-L2 = %.3e   <- %s\n",
           rel_l2(q_fp16, q_exact),
           rel_l2(q_fp16, q_exact) <= rel_l2(q_bf16, q_exact) ? "BETTER than shipped" : "WORSE than shipped");
    { float mx = 0; for (int t = 0; t < kdim; ++t) mx = fmaxf(mx, fabsf(xf[t]));
      printf("[quality]   max |activation| = %.3f (fp16 overflows at 65504)\n", mx); }

    // ---- host reference, in the tensor core's own arithmetic:
    //      acc = sum_k (hi_k * x_k + lo_k * x_k) in f32, then * scale.
    //      Also track the shipped ordering (hi+lo)*x to quantify the delta.
    std::vector<double> ref(n_rows), ref_shipped(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        uint32_t st = 0; double a = 0.0, a2 = 0.0;
        for (int s = 0; s < S; ++s) {
            st = ((st << K_BITS) | sym[(size_t)r * S + s]) & 0xFFFFu;
            uint32_t t0 = st * MCG_MULT, t1 = t0 * MCG_MULT;
            float hi0, lo0, hi1, lo1;
            cb_halves(t0, &hi0, &lo0);
            cb_halves(t1, &hi1, &lo1);
            const float x0 = h2f(xh[2 * s]), x1 = h2f(xh[2 * s + 1]);
            a  += (double)hi0 * x0 + (double)lo0 * x0 + (double)hi1 * x1 + (double)lo1 * x1;
            a2 += (double)(hi0 + lo0) * x0 + (double)(hi1 + lo1) * x1;
        }
        ref[r] = a * scales[r];
        ref_shipped[r] = a2 * scales[r];
    }

    uint32_t *dR; float *dS, *dY; __half* dX;
    CHECK(cudaMalloc(&dR, R.size() * 4));
    CHECK(cudaMalloc(&dS, n_rows * 4));
    CHECK(cudaMalloc(&dY, n_rows * 4));
    CHECK(cudaMalloc(&dX, kdim * 2));
    CHECK(cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dS, scales.data(), n_rows * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dX, xh.data(), kdim * 2, cudaMemcpyHostToDevice));

    std::vector<float> y(n_rows);
    auto check = [&](const char* tag, const std::vector<double>& base) {
        CHECK(cudaMemcpy(y.data(), dY, n_rows * 4, cudaMemcpyDeviceToHost));
        double num = 0, d1 = 0, d2 = 0, maxrel = 0;
        for (int r = 0; r < n_rows; ++r) {
            num += y[r] * base[r]; d1 += (double)y[r] * y[r]; d2 += base[r] * base[r];
            double den = fabs(base[r]) + 1e-6;
            maxrel = fmax(maxrel, fabs(y[r] - base[r]) / den);
        }
        double cos = num / (sqrt(d1) * sqrt(d2) + 1e-30);
        printf("[check] %-22s cosine=%.9f  max_rel=%.3e\n", tag, cos, maxrel);
        return cos;
    };

    constexpr int NB = 4, WARPS = 8;
    const int KS = (argc > 4) ? atoi(argv[4]) : 8;
    const int rows_per_block = 16 * NB * WARPS;
    dim3 g_tc((n_rows + rows_per_block - 1) / rows_per_block, KS, 1);
    auto launch_tc = [&]{ tcfrag_gemv<NB, WARPS><<<g_tc, WARPS * 32>>>(dR, dS, dX, dY, n_rows, W, KS, MCG_MULT); };
    CHECK(cudaMemset(dY, 0, n_rows * 4));
    launch_tc();
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    double c_tc = check("tcfrag v1 vs mma-order", ref);
    check("tcfrag v1 vs shipped-order", ref_shipped);

    auto launch_v2 = [&]{ tcfrag_gemv_v2<NB, WARPS><<<g_tc, WARPS * 32>>>(dR, dS, dX, dY, n_rows, W, KS, MCG_MULT); };
    CHECK(cudaMemset(dY, 0, n_rows * 4));
    launch_v2();
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    double c_v2 = check("tcfrag v2 vs mma-order", ref);
    if (!(c_v2 > 0.999)) { printf("FAIL: v2 cosine %.9f\n", c_v2); return 1; }

    CHECK(cudaMemset(dY, 0, n_rows * 4));
    scalar_gemv<<<(n_rows + 7) / 8, 256>>>(dR, dS, dX, dY, n_rows, W, MCG_MULT);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    check("scalar vs shipped-order", ref_shipped);

    // -------- ENGAGEMENT GUARD: a green run must prove work happened. -------
    if (!(c_tc > 0.999)) {
        printf("FAIL: tcfrag cosine %.9f -- the fragment mapping is wrong.\n", c_tc);
        return 1;
    }
    {   // the guard must be able to go red: perturb one symbol, expect a drop.
        std::vector<uint32_t> Rbad = R;
        Rbad[0] ^= 0xF0u;
        uint32_t* dRb; CHECK(cudaMalloc(&dRb, Rbad.size() * 4));
        CHECK(cudaMemcpy(dRb, Rbad.data(), Rbad.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dY, 0, n_rows * 4));
        tcfrag_gemv<NB, WARPS><<<g_tc, WARPS * 32>>>(dRb, dS, dX, dY, n_rows, W, KS, MCG_MULT);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(y.data(), dY, n_rows * 4, cudaMemcpyDeviceToHost));
        double rel = fabs(y[0] - ref[0]) / (fabs(ref[0]) + 1e-6);
        printf("[guard] one-nibble mutation moves row 0 by %.3e relative %s\n",
               rel, rel > 1e-4 ? "(guard is live)" : "(GUARD IS DEAD)");
        if (!(rel > 1e-4)) { printf("FAIL: guard cannot go red.\n"); return 1; }
        CHECK(cudaFree(dRb));
    }

    // -------- timing --------
    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    auto time_it = [&](const char* tag, auto launch, double weights) {
        CHECK(cudaMemset(dY, 0, n_rows * 4));
        for (int i = 0; i < 20; ++i) launch();
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) launch();
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        double us = (double)ms * 1000.0 / iters;
        printf("[time]  %-22s %8.2f us   %7.1f GB/s (2bpw)  %6.2f Tweight/s\n",
               tag, us, weights * 0.25 / (us * 1e-6) / 1e9, weights / (us * 1e-6) / 1e12);
        return us;
    };
    const double weights = (double)n_rows * kdim;
    double us_tc = time_it("tcfrag v1 mma", launch_tc, weights);
    double us_v2 = time_it("tcfrag v2 mma", launch_v2, weights);
    (void)us_v2;
#define SWEEP(nb, wp)                                                                                  \
    do { const int rpb = 16 * (nb) * (wp);                                                             \
         dim3 g((n_rows + rpb - 1) / rpb, KS, 1);                                                      \
         if (g.x >= 1) { char tag[64]; snprintf(tag, 64, "  v2 NB=%d WARPS=%d", (nb), (wp));           \
           CHECK(cudaMemset(dY, 0, n_rows * 4));                                                       \
           tcfrag_gemv_v2<(nb), (wp)><<<g, (wp) * 32>>>(dR, dS, dX, dY, n_rows, W, KS, MCG_MULT);      \
           CHECK(cudaDeviceSynchronize());                                                             \
           double c = check(tag, ref); if (c < 0.999) { printf("FAIL sweep %s\n", tag); return 1; }    \
           time_it(tag, [&]{ tcfrag_gemv_v2<(nb), (wp)><<<g, (wp) * 32>>>(dR, dS, dX, dY, n_rows, W, KS, MCG_MULT); }, weights); } \
    } while (0)
    SWEEP(2, 4); SWEEP(2, 8); SWEEP(4, 4); SWEEP(8, 4); SWEEP(1, 8); SWEEP(1, 4);
#undef SWEEP

    // ---- v3: pre-duplicated activation (the activation half of the layout).
    std::vector<uint16_t> xdup(2 * kdim);
    for (int t = 0; t < kdim; ++t) { xdup[2 * t] = xh[t]; xdup[2 * t + 1] = xh[t]; }
    __half* dXD; CHECK(cudaMalloc(&dXD, xdup.size() * 2));
    CHECK(cudaMemcpy(dXD, xdup.data(), xdup.size() * 2, cudaMemcpyHostToDevice));
#define SWEEP3(nb, wp)                                                                                 \
    do { const int rpb = 16 * (nb) * (wp);                                                             \
         dim3 g((n_rows + rpb - 1) / rpb, KS, 1);                                                      \
         char tag[64]; snprintf(tag, 64, "  v3 NB=%d WARPS=%d", (nb), (wp));                           \
         CHECK(cudaMemset(dY, 0, n_rows * 4));                                                         \
         tcfrag_gemv_v3<(nb), (wp)><<<g, (wp) * 32>>>(dR, dS, dXD, dY, n_rows, W, KS, MCG_MULT);       \
         CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());                                    \
         double c = check(tag, ref); if (c < 0.999) { printf("FAIL sweep %s\n", tag); return 1; }      \
         time_it(tag, [&]{ tcfrag_gemv_v3<(nb), (wp)><<<g, (wp) * 32>>>(dR, dS, dXD, dY, n_rows, W, KS, MCG_MULT); }, weights); \
    } while (0)
    SWEEP3(1, 8); SWEEP3(1, 4); SWEEP3(2, 8); SWEEP3(4, 8);
#undef SWEEP3
    double us_sc = time_it("scalar (in-file base)", [&]{ scalar_gemv<<<(n_rows + 7) / 8, 256>>>(dR, dS, dX, dY, n_rows, W, MCG_MULT); }, weights);
    printf("[time]  speedup tcfrag / scalar = %.2fx\n", us_sc / us_tc);
    printf("[shape] n_rows=%d k=%d symbols/row=%d words/row=%d NB=%d WARPS=%d ksplit=%d grid=(%d,%d)\n",
           n_rows, kdim, S, W, NB, WARPS, KS, g_tc.x, g_tc.y);
    return 0;
}
