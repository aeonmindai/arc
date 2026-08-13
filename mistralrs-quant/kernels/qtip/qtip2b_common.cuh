// Shared device functions for the qtip2b bitshift-trellis (computed codebook)
// kernels. Included by qtip_bitshift.cu (dequantize / GEMV / quantize) and
// qtip_grouped_gemm.cu (the batched MoE grouped GEMM) so the 3INST decode is
// defined exactly once.
//
// Format recap (see qtip_bitshift.cu and src/qtip/bitshift.rs for the full
// derivation): V=1, K=2, L=16; `state = ((state << 2) | sym) & 0xFFFF`;
// codeword = f32(fp16(hi)) + f32(fp16(lo)) of the masked/XORed MCG product.
// The fp16 halves are summed in **f32** so the decode is bit-identical to the
// CPU reference (`mcg_codeword`) — never replace with __hadd.
//
// Everything here is `__device__ __forceinline__` / `constexpr`, so including
// this header from multiple translation units creates no linkage issues.

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

constexpr uint32_t Q2B_K            = 2;
constexpr uint32_t Q2B_L            = 16;
constexpr uint32_t Q2B_STATE_MASK   = (1u << Q2B_L) - 1u;
constexpr uint32_t Q2B_ALPHABET     = 1u << Q2B_K;             // 4
constexpr uint32_t Q2B_NSTATES      = 1u << Q2B_L;             // 65536
constexpr uint32_t Q2B_WARMUP_SYMS  = Q2B_L / Q2B_K;           // 8
constexpr uint32_t Q2B_MASK         = 0x8FFF8FFFu;
constexpr uint32_t Q2B_XOR          = 0x3B603B60u;

// ---------------------------------------------------------------------------
// The computed codebook. IMAD + LOP3 + 2x H2F + FADD, all in registers.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float q2b_decode(uint32_t state, uint32_t mult) {
    uint32_t x = state * mult;                    // wrapping 32-bit multiply
    uint32_t m = (x & Q2B_MASK) ^ Q2B_XOR;        // single LOP3 on sm_80+
    float hi = __half2float(__ushort_as_half((unsigned short)(m >> 16)));
    float lo = __half2float(__ushort_as_half((unsigned short)(m & 0xFFFFu)));
    return hi + lo;
}

// 2-bit symbol `t` of a packed row (4 symbols/byte, LSB-first).
__device__ __forceinline__ uint32_t q2b_sym(const uint8_t* __restrict__ packed, int t) {
    return ((uint32_t)packed[t >> 2] >> ((t & 3) * 2)) & 0x3u;
}

// ---------------------------------------------------------------------------
// Random-access state reconstruction (no sequential warm-up replay).
//
// Because the state is a sliding 16-bit window over the symbol stream
// (state pair j = sym[t - j]) while the packed stream stores the same window
// LSB-first (window pair j = sym[t - 7 + j]), the state at symbol `t` is the
// PAIR-REVERSAL of the 16 stream bits ending at bit 2t+1:
//
//     state(t) = pair_reverse_16(stream_bits[2t-14 .. 2t+1])
//
// (bits below stream position 0 are zero, matching the all-zero initial
// state). Pair reversal = full 16-bit bit-reversal followed by swapping
// adjacent bits — ~4 ALU ops via __brev. Mirrored + property-tested against
// the sequential recurrence on the Rust side (`grouped::window_state_2b`).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t q2b_state_from_window(uint32_t win16) {
    uint32_t r = __brev(win16 & 0xFFFFu) >> 16;   // full 16-bit bit-reversal
    return ((r & 0x5555u) << 1) | ((r >> 1) & 0x5555u);
}

template <typename T>
__device__ __forceinline__ T q2b_from_f32(float v);
template <>
__device__ __forceinline__ float q2b_from_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half q2b_from_f32<__half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 q2b_from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ float q2b_to_f32(T v);
template <>
__device__ __forceinline__ float q2b_to_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ float q2b_to_f32<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float q2b_to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float q2b_warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}
