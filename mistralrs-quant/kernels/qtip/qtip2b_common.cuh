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

// ---------------------------------------------------------------------------
// SEQUENTIAL RUN ADVANCE — the cheap half of the same identity.
//
// `q2b_state_from_window` costs ~11 ALU ops (two smem words, two shifts, an
// OR, __brev, and the adjacent-bit swap). That is the price of decoding an
// ISOLATED position. Along a CONTIGUOUS run of symbols the original trellis
// recurrence is far cheaper — one SHF plus one LOP3:
//
//     state(t+1) = ((state(t) << 2) | sym[t+1]) & 0xFFFF
//
// so a run of R symbols costs `window + (R-1)*advance` instead of
// `R * window`. At R = 32 that is ~11.3 ops/weight -> ~2 ops/weight of state
// reconstruction. This matters because the trellis GEMM is ALU-issue-bound on
// decode, not bandwidth-bound (98/98 gen-2 GEMV variants measured: latency
// hiding does not pay, per-symbol decode is the ceiling).
//
// The window identity is still what SEEDS each run, so runs stay independent:
// no cross-thread state chain, no warm-up replay. Mirrored and property-tested
// on the Rust side by `grouped::run_states_2b`.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t q2b_advance_state(uint32_t state, uint32_t sym) {
    return ((state << Q2B_K) | (sym & (Q2B_ALPHABET - 1u))) & Q2B_STATE_MASK;
}

// ---------------------------------------------------------------------------
// cp.async helpers (sm_80+). `valid == false` uses the src-size-0 form, which
// zero-fills the destination without touching the source address — that is
// what makes out-of-range rows / past-the-end tiles free of tail guards.
//
// Shared by the grouped GEMM (qtip_grouped_gemm.cu) and the gen-2 GEMV
// pipeline (qtip_bitshift_tune2.cu). Below sm_80 the bodies degrade to plain
// synchronous copies so the TUs still compile on old arches (the kernels
// using them are unreachable there — `has_qtip_kernels` is off below SM80).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void q2b_cp_async_16(void* smem_dst, const void* gmem_src, bool valid) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(smem_dst);
    const int src_bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(d), "l"(gmem_src), "r"(src_bytes));
#else
    int4* d = reinterpret_cast<int4*>(smem_dst);
    *d = valid ? *reinterpret_cast<const int4*>(gmem_src) : make_int4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void q2b_cp_async_4(void* smem_dst, const void* gmem_src, bool valid) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(smem_dst);
    const int src_bytes = valid ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(d), "l"(gmem_src), "r"(src_bytes));
#else
    uint32_t* d = reinterpret_cast<uint32_t*>(smem_dst);
    *d = valid ? *reinterpret_cast<const uint32_t*>(gmem_src) : 0u;
#endif
}

__device__ __forceinline__ void q2b_cp_commit() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// Wait until at most N committed cp.async groups are still outstanding.
template <int N>
__device__ __forceinline__ void q2b_cp_wait() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
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
