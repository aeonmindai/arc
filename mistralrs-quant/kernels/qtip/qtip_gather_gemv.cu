// QTIP on-device MoE gather + fused decode + gemv (capturable decode path).
//
// This is the CUDA-graph-capturable sibling of `qtip_fused_gemv` in
// `qtip_gemv.cu`. The existing MoE dispatch (`gather_forward_cuda` in mod.rs)
// reads the routing indices to the HOST (`indices.to_device(Cpu).to_vec1()`)
// to decide which experts to dequantize — a device->host sync that aborts any
// CUDA stream capture. This kernel instead reads each (token, slot)'s expert
// id ON-DEVICE (`__ldg(&indices[pair])`) and runs the trellis gemv directly
// against that expert's packed rows, so the whole MoE stays on the stream and
// the decode forward becomes capturable.
//
// Scope: the b=1 (and small-batch / MTP-draft) DECODE regime, where there are
// few (token, slot) pairs. It does one independent gemv per pair, so it does
// NOT reuse a dequantized expert across many tokens — that token-grouping win
// only matters at prefill, which keeps the existing grouped path. For decode
// (n_pairs = n_tokens * top_k, e.g. 1 * 6 = 6) per-pair gemv is optimal and,
// crucially, sync-free.
//
// Layout (3-D stacked experts), matching QtipLayer 3-D mode:
//   * packed     : [E, n_rows, packed_per_row]  u8   (flat: expert*n_rows+row)
//   * row_scales : [E, n_rows]                   f32  (flat: expert*n_rows+row)
//   * lut        : [2^L * V]                     f32  (shared across experts)
//   * x          : [n_pairs, k_in]               T    (already rotated)
//   * indices    : [n_pairs]                     u32  (expert id per pair)
//   * y          : [n_pairs, n_rows]             T
//
// All trellis/correctness invariants are identical to qtip_gemv.cu (state
// recurrence, init state 0, low/high nibble pack order, per-row scale, warmup).
// Activation pre-rotation (D.H.D) must be applied to x BEFORE this kernel.
//
// SM80+ (uses __nv_bfloat16). Gated by `has_qtip_kernels` in build.rs.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// The `sum2` computed codebook. RUN-161 already deleted the LUT *load* from
// this kernel by evaluating the Gaussian in registers — but that costs a
// logf + sqrtf + sincosf per weight on the SFU. The sum2 code replaces all
// three transcendentals with IMAD/LOP3/cvt/FADD.
#include "qtip_codebook.cuh"

namespace {

constexpr uint32_t QTIP_K          = 4;
constexpr uint32_t QTIP_L          = 16;
constexpr uint32_t QTIP_V          = 2;
constexpr uint32_t QTIP_STATE_MASK = (1u << QTIP_L) - 1u;
constexpr uint32_t QTIP_WARMUP_SYMS = QTIP_L / QTIP_K;

template <typename T>
__device__ __forceinline__ T gg_from_f32(float v);
template <>
__device__ __forceinline__ float gg_from_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half gg_from_f32<__half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 gg_from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ float gg_to_f32(T v);
template <>
__device__ __forceinline__ float gg_to_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ float gg_to_f32<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float gg_to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float gg_warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

// Load two consecutive activation elements as one vectorized load (halves the
// LSU instruction count on the x reads). Overloaded per dtype.
__device__ __forceinline__ float2 gg_load2(const float* p) {
    return __ldg(reinterpret_cast<const float2*>(p));
}
__device__ __forceinline__ float2 gg_load2(const __half* p) {
    return __half22float2(__ldg(reinterpret_cast<const __half2*>(p)));
}
__device__ __forceinline__ float2 gg_load2(const __nv_bfloat16* p) {
    return __bfloat1622float2(__ldg(reinterpret_cast<const __nv_bfloat162*>(p)));
}

// ---------------------------------------------------------------------------
// RUN-161: COMPUTED QTIP CODEBOOK (the "trellis is not instruction-bound" fix).
//
// Each 16-bit trellis state decodes to V=2 Gaussian reproduction values. The
// host bakes these with gaussian_lut() (mod.rs:174): splitmix64(state) -> two
// uniforms -> Box-Muller. That is a PURE FUNCTION of `state`, so we evaluate it
// inline in registers instead of gathering from the 512 KB global LUT.
//
// Why this is THE lever: at L=16 the LUT is 2^16 * V * 4 = 512 KB, which does
// NOT fit in 48 KB shared memory, so the prior "stage LUT to shared" path was
// dead and every per-symbol weight lookup was a dependent, data-scattered
// GLOBAL load. ncu attributed the kernel's stall to long_scoreboard (global
// LOAD latency on exactly this gather). Computing the code in-register removes
// that load entirely: ~a dozen integer/FP ops (cache- and bandwidth-free) per
// weight, which is the QTIP-paper computed-code decode that reaches a large
// fraction of HBM peak at M=1.
//
// Bit-faithfulness: mirrors the host math exactly (same constants, same
// operation order). Transcendentals (logf/sincosf) differ from the host libm
// by <~1e-4 ULP-scale, far below quantization error — decode stays numerically
// equivalent to the baked LUT it replaces.
__device__ __forceinline__ float2 qtip_decode_state(uint32_t state) {
    unsigned long long z = (unsigned long long)state * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const uint32_t hi = (uint32_t)(z >> 32);
    const uint32_t lo = (uint32_t)(z & 0xFFFFFFFFu);
    // Host denom = (u32::MAX as f32 + 2.0) == 2^32 exactly in f32.
    const float denom = 4294967296.0f;
    const float u1 = ((float)hi + 1.0f) / denom;
    const float u2 = ((float)lo + 1.0f) / denom;
    const float r = sqrtf(-2.0f * logf(u1));
    const float theta = (2.0f * 3.14159265358979323846f) * u2;
    float s, c;
    sincosf(theta, &s, &c);
    return make_float2(r * c, r * s);  // (g0, g1) == (lut[2*state], lut[2*state+1])
}

// Codebook dispatch for the decode loop. `COMPUTED_CB == false` keeps the
// RUN-161 Gaussian-in-registers behaviour byte-for-byte; `true` is the sum2
// code. `mult` is unused in the Gaussian arm (the Gaussian codebook has no
// tunable constant — it is a pure function of the state).
template <bool COMPUTED_CB>
__device__ __forceinline__ float2 gg_codeword(uint32_t state, unsigned int mult) {
    if (COMPUTED_CB) {
        return qtip_cb_sum2(state, mult);
    }
    return qtip_decode_state(state);
}

// ---------------------------------------------------------------------------
// OPTIMISED warp kernel (trellis chain). Bit-identical to the baseline: it
// computes exactly the same `state` sequence and the same codewords in the same
// order, and touches the accumulator with the same fmaf sequence.
//
// What the SASS said (sm_90, --use_fast_math, the crate's real flags):
//   * the main loop is fully unrolled to GROUP*ROWS_PER_WARP = 8 symbol decodes
//   * 16 x LD.E.U8 per iteration -- *generic* byte loads, not LDS. `stage_packed`
//     is a RUNTIME int, so `rp[r]` is a runtime-selected shared-or-global
//     pointer and the compiler must emit generic loads with a full 64-bit
//     address chain (LEA.HI.X.SX32 x12, IMAD.SHL.U32 x12, IMAD.X x6, IADD3 x23).
//     The shared-staging optimisation was being defeated at the ISA level.
//   * HALF of those 16 byte loads are the warm-up replay, which produces no
//     weights at all: 4 symbols replayed per GROUP=4 symbols decoded, per row.
//
// The three fixes, all bit-identical:
//   (A) `STAGED` becomes a template parameter -> LDS with 32-bit addressing.
//   (B) The L=16 state is just the last 4 nibbles of the stream, so the whole
//       warm-up replay is a WINDOW LOAD: one 16-bit read per 4 symbols covers
//       both the warm-up nibbles and the decoded nibbles. 8 scattered byte
//       loads/row-group -> 2 aligned u16 loads, and the replay's per-symbol
//       load+extract+shift chain collapses to a nibble reversal.
//   (C) the GROUP activations are one 128-bit load instead of GROUP 32-bit ones.
// ---------------------------------------------------------------------------

// Unpack one 32-bit word holding two 16-bit activations. Register-only: the
// vectorised path already has the bytes in registers, and `__ldg` on a stack
// address is illegal (CUDA "operation not supported on global/shared address
// space") -- that is exactly the bug the first draft of this had.
__device__ __forceinline__ float2 gg_unpack2(unsigned int w, const __nv_bfloat16*) {
    return make_float2(__bfloat162float(__ushort_as_bfloat16((unsigned short)(w & 0xFFFFu))),
                       __bfloat162float(__ushort_as_bfloat16((unsigned short)(w >> 16))));
}
__device__ __forceinline__ float2 gg_unpack2(unsigned int w, const __half*) {
    return make_float2(__half2float(__ushort_as_half((unsigned short)(w & 0xFFFFu))),
                       __half2float(__ushort_as_half((unsigned short)(w >> 16))));
}
__device__ __forceinline__ float2 gg_unpack2(unsigned int, const float*) {
    return make_float2(0.0f, 0.0f);   // never taken; float uses the scalar path
}

// Load the GROUP activation pairs. For 16-bit T, GROUP pairs are GROUP*4 bytes,
// and base = lane*GROUP + m*32*GROUP keeps that 16-byte aligned, so it is
// GROUP/4 x LDG.E.128 instead of GROUP x LDG.E.32.
template <typename T, int GROUP>
__device__ __forceinline__ void gg_load_group(const T* p, bool vec_ok, float2* out) {
    if constexpr (sizeof(T) == 2 && (GROUP % 4) == 0) {
        if (vec_ok) {
            #pragma unroll
            for (int q = 0; q < GROUP / 4; ++q) {
                const uint4 v = __ldg(reinterpret_cast<const uint4*>(p) + q);
                out[4 * q + 0] = gg_unpack2(v.x, (const T*)nullptr);
                out[4 * q + 1] = gg_unpack2(v.y, (const T*)nullptr);
                out[4 * q + 2] = gg_unpack2(v.z, (const T*)nullptr);
                out[4 * q + 3] = gg_unpack2(v.w, (const T*)nullptr);
            }
            return;
        }
    }
    #pragma unroll
    for (int j = 0; j < GROUP; ++j) out[j] = gg_load2(p + j * (int)QTIP_V);
}

// Nibble reversal of the low 16 bits: the packed stream stores symbol t of a
// little-endian u16 at bit 4*(t mod 4), while the trellis state holds the OLDEST
// symbol in the HIGH nibble. Same value the replay loop produced, 4 ops instead
// of 4 loads + 4 extracts + 4 shift/or/mask.
__device__ __forceinline__ uint32_t gg_state_from_window(uint32_t h) {
    const uint32_t a = ((h & 0x0F0Fu) << 4) | ((h >> 4) & 0x0F0Fu);
    return ((a & 0x00FFu) << 8) | ((a >> 8) & 0x00FFu);
}

template <typename T, int WARPS_PER_BLOCK, int ROWS_PER_WARP, bool COMPUTED_CB,
          bool STAGED, int GROUP>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
qtip_gather_gemv_warp_kernel(
    const uint8_t*  __restrict__ packed,
    const float*    __restrict__ row_scales,
    const float*    __restrict__ lut,
    const T*        __restrict__ x,
    const uint32_t* __restrict__ indices,
    T*              __restrict__ y,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts,
    unsigned int cb_mult
) {
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    extern __shared__ float s_mem[];
    (void)lut;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int pair = blockIdx.y;
    const int row0_block = blockIdx.x * ROWS_PER_BLOCK;
    if (row0_block >= n_rows || pair >= n_pairs) return;

    const uint32_t expert = __ldg(indices + pair);
    const bool invalid = expert >= (uint32_t)num_experts;

    uint8_t* s_packed = reinterpret_cast<uint8_t*>(s_mem);
    if (STAGED && !invalid) {
        const int n_block_rows = min(ROWS_PER_BLOCK, n_rows - row0_block);
        const long total = (long)n_block_rows * packed_per_row;
        const uint8_t* __restrict__ g =
            packed + ((size_t)expert * n_rows + row0_block) * packed_per_row;
        // Stage 16 B per thread per step instead of 1 B. `packed_per_row` is a
        // multiple of 16 for every real rung shape (it is num_symbols/2 and
        // num_symbols is a multiple of 32), and `g` inherits cudaMalloc's 256 B
        // alignment through a multiple-of-packed_per_row offset, so the uint4
        // view is aligned. The byte loop stays as the fallback for odd shapes.
        if (((packed_per_row & 15) == 0) &&
            ((reinterpret_cast<uintptr_t>(g) & 15u) == 0u)) {
            const uint4* __restrict__ g4 = reinterpret_cast<const uint4*>(g);
            uint4* s4 = reinterpret_cast<uint4*>(s_packed);
            const int total4 = (int)(total >> 4);
            for (int i = threadIdx.x; i < total4; i += WARPS_PER_BLOCK * 32) {
                s4[i] = __ldg(g4 + i);
            }
        } else {
            for (long i = threadIdx.x; i < total; i += WARPS_PER_BLOCK * 32) {
                s_packed[i] = __ldg(g + i);
            }
        }
        __syncthreads();
    }

    const int row0 = row0_block + warp * ROWS_PER_WARP;
    if (invalid) {
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const int row = row0 + r;
                if (row < n_rows) y[(size_t)pair * n_rows + row] = gg_from_f32<T>(0.0f);
            }
        }
        return;
    }

    // Per-row backing. STAGED is compile-time, so the shared form is a 32-bit
    // shared offset (LDS) and the global form a 64-bit pointer (LDG) -- neither
    // is a generic pointer, which is what removes the address chain.
    const uint8_t* rp_g[ROWS_PER_WARP];
    unsigned int   rp_s[ROWS_PER_WARP];
    float scl[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = row0 + r;
        const bool ok = row < n_rows;
        const size_t er = (size_t)expert * n_rows + (ok ? row : 0);
        rp_g[r] = packed + er * packed_per_row;
        rp_s[r] = (unsigned int)((warp * ROWS_PER_WARP + r) * packed_per_row);
        scl[r]  = ok ? __ldg(row_scales + er) : 0.0f;
    }
    const T* __restrict__ x_pair = x + (size_t)pair * num_symbols * QTIP_V;
    const bool xvec = ((reinterpret_cast<uintptr_t>(x_pair) & 15u) == 0u);

    const int gstride = 32 * GROUP;
    // u16 chunks spanning warm-up (QTIP_WARMUP_SYMS nibbles) + GROUP nibbles.
    constexpr int NH = (QTIP_WARMUP_SYMS + GROUP + 3) / 4;

    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    // (D) The `fast` test is per-LANE, not per-warp: only lane 0's first
    //     iteration is ever slow. Leaving the general path inside the loop put a
    //     BSSY/BSYNC divergence region, the `fast` ISETP and the general path's
    //     per-symbol bounds test into EVERY iteration of the hot loop. Peeling it
    //     out leaves a straight-line body. Bit-identical: the peeled loops visit
    //     exactly the same `base` values, in order, and pick exactly the same
    //     path for each -- `fast` is a pure function of `base`.
    auto do_fast = [&](int base) {
        float2 xg[GROUP];
        gg_load_group<T, GROUP>(x_pair + base * (int)QTIP_V, xvec, xg);
        const int b0 = (base - (int)QTIP_WARMUP_SYMS) >> 1;  // even byte index
        // (E) Every row's window load is issued before any row's decode, so the
        //     ROWS_PER_WARP window loads are in flight together.
        uint32_t hh[ROWS_PER_WARP][NH];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            #pragma unroll
            for (int q = 0; q < NH; ++q) {
                if (STAGED) {
                    hh[r][q] = *reinterpret_cast<const unsigned short*>(s_packed + rp_s[r] + b0 + 2 * q);
                } else {
                    hh[r][q] = __ldg(reinterpret_cast<const unsigned short*>(rp_g[r] + b0 + 2 * q));
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const float scale = scl[r];
            uint32_t state = gg_state_from_window(hh[r][0]);
            float a = acc[r];
            #pragma unroll
            for (int j = 0; j < GROUP; ++j) {
                const int t = (int)QTIP_WARMUP_SYMS + j;   // nibble index in the window
                const uint32_t sym = (hh[r][t >> 2] >> (4 * (t & 3))) & 0x0Fu;
                state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;
                const float2 w = gg_codeword<COMPUTED_CB>(state, cb_mult);
                a = fmaf(w.x * scale, xg[j].x, a);
                a = fmaf(w.y * scale, xg[j].y, a);
            }
            acc[r] = a;
        }
    };

    // General path -- identical arithmetic to the original.
    auto do_general = [&](int base) {
        const int gend = min(base + GROUP, num_symbols);
        float2 xg[GROUP];
        gg_load_group<T, GROUP>(x_pair + base * (int)QTIP_V,
                                xvec && (base + GROUP <= num_symbols), xg);
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const float scale = scl[r];
            uint32_t state = 0u;
            float a = acc[r];
            const uint8_t* row_packed = STAGED ? (s_packed + rp_s[r]) : rp_g[r];
            const int w0 = base > (int)QTIP_WARMUP_SYMS ? base - (int)QTIP_WARMUP_SYMS : 0;
            for (int t = w0; t < base; ++t) {
                const uint8_t b = row_packed[t >> 1];
                const uint32_t sym = (t & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
                state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;
            }
            #pragma unroll
            for (int j = 0; j < GROUP; ++j) {
                const int s = base + j;
                if (s >= gend) break;
                const uint8_t b = row_packed[s >> 1];
                const uint32_t sym = (s & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
                state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;
                const float2 w = gg_codeword<COMPUTED_CB>(state, cb_mult);
                a = fmaf(w.x * scale, xg[j].x, a);
                a = fmaf(w.y * scale, xg[j].y, a);
            }
            acc[r] = a;
        }
    };

    int base = lane * GROUP;
    // leading partial warm-up window (lane 0 only, at most one iteration)
    for (; base < num_symbols && base < (int)QTIP_WARMUP_SYMS; base += gstride) do_general(base);
    // hot loop: straight-line, no divergence region
    for (; base + GROUP <= num_symbols; base += gstride) do_fast(base);
    // trailing partial group (runs only when num_symbols is not a multiple of gstride)
    for (; base < num_symbols; base += gstride) do_general(base);

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const float a = gg_warp_reduce_sum(acc[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_rows) y[(size_t)pair * n_rows + row] = gg_from_f32<T>(a);
    }
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

#ifndef GG_GROUP
#define GG_GROUP 4
#endif

#define QTIP_GATHER_GEMV_LAUNCHER(NAME, T)                                            \
    void NAME(const uint8_t*  d_packed,                                               \
              const float*    d_row_scales,                                           \
              const float*    d_lut,                                                  \
              const T*        d_x_rotated,                                            \
              const uint32_t* d_indices,                                              \
              T*              d_y,                                                    \
              int n_rows,                                                             \
              int packed_per_row,                                                     \
              int num_symbols,                                                        \
              int n_pairs,                                                            \
              int num_experts,                                                        \
              unsigned int cb_mult,                                                   \
              cudaStream_t    stream) {                                               \
        constexpr int WARPS_PER_BLOCK = 8;                                            \
        constexpr int ROWS_PER_WARP = 2;                                              \
        constexpr int GRP = GG_GROUP;                                                 \
        constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;               \
        const size_t packed_smem = (size_t)ROWS_PER_BLOCK * packed_per_row;           \
        const bool   stage_packed = packed_smem <= 48 * 1024;                         \
        const size_t SHMEM = stage_packed ? packed_smem : 0;                          \
        dim3 grid((n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, n_pairs, 1);        \
        if (cb_mult != 0u) {                                                          \
            if (stage_packed)                                                         \
                qtip_gather_gemv_warp_kernel<T, WARPS_PER_BLOCK, ROWS_PER_WARP, true, true, GRP>   \
                    <<<grid, WARPS_PER_BLOCK * 32, SHMEM, stream>>>(                  \
                        d_packed, d_row_scales, d_lut, d_x_rotated, d_indices, d_y,   \
                        n_rows, packed_per_row, num_symbols, n_pairs, num_experts, cb_mult); \
            else                                                                      \
                qtip_gather_gemv_warp_kernel<T, WARPS_PER_BLOCK, ROWS_PER_WARP, true, false, GRP>  \
                    <<<grid, WARPS_PER_BLOCK * 32, SHMEM, stream>>>(                  \
                        d_packed, d_row_scales, d_lut, d_x_rotated, d_indices, d_y,   \
                        n_rows, packed_per_row, num_symbols, n_pairs, num_experts, cb_mult); \
        } else {                                                                      \
            if (stage_packed)                                                         \
                qtip_gather_gemv_warp_kernel<T, WARPS_PER_BLOCK, ROWS_PER_WARP, false, true, GRP>  \
                    <<<grid, WARPS_PER_BLOCK * 32, SHMEM, stream>>>(                  \
                        d_packed, d_row_scales, d_lut, d_x_rotated, d_indices, d_y,   \
                        n_rows, packed_per_row, num_symbols, n_pairs, num_experts, 0u); \
            else                                                                      \
                qtip_gather_gemv_warp_kernel<T, WARPS_PER_BLOCK, ROWS_PER_WARP, false, false, GRP> \
                    <<<grid, WARPS_PER_BLOCK * 32, SHMEM, stream>>>(                  \
                        d_packed, d_row_scales, d_lut, d_x_rotated, d_indices, d_y,   \
                        n_rows, packed_per_row, num_symbols, n_pairs, num_experts, 0u); \
        }                                                                             \
    }

QTIP_GATHER_GEMV_LAUNCHER(launch_qtip_gather_gemv_v2_k4_l16_bf16, __nv_bfloat16)
QTIP_GATHER_GEMV_LAUNCHER(launch_qtip_gather_gemv_v2_k4_l16_f16,  __half)
QTIP_GATHER_GEMV_LAUNCHER(launch_qtip_gather_gemv_v2_k4_l16_f32,  float)

#undef QTIP_GATHER_GEMV_LAUNCHER

} // extern "C"
