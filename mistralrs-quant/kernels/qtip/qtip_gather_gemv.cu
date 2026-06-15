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

// One block per (output row, pair). Reads the pair's expert id on-device,
// offsets into that expert's packed rows / scales, and accumulates
// y[pair, row] = sum_k W_expert[row,k] * x[pair,k].
//
// Grid:  (n_rows, n_pairs, 1)
// Block: (THREADS, 1, 1)
template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS)
qtip_gather_gemv_v2_k4_l16_kernel(
    const uint8_t*  __restrict__ packed,      // [E, n_rows, packed_per_row]
    const float*    __restrict__ row_scales,  // [E, n_rows]
    const float*    __restrict__ lut,         // [2^L * V]
    const T*        __restrict__ x,           // [n_pairs, k_in]  (rotated)
    const uint32_t* __restrict__ indices,     // [n_pairs]
    T*              __restrict__ y,           // [n_pairs, n_rows]
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts
) {
    const int row  = blockIdx.x;
    const int pair = blockIdx.y;
    if (row >= n_rows || pair >= n_pairs) return;

    const uint32_t expert = __ldg(indices + pair);
    // Defensive: an out-of-range expert id (shouldn't happen from a correct
    // router) writes 0 rather than reading out of bounds.
    if (expert >= (uint32_t)num_experts) {
        if (threadIdx.x == 0) y[(size_t)pair * n_rows + row] = gg_from_f32<T>(0.0f);
        return;
    }

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int N_WARPS = THREADS / 32;
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of warp size");

    const size_t expert_row = (size_t)expert * n_rows + row;
    const uint8_t* row_packed = packed + expert_row * packed_per_row;
    const float scale = __ldg(row_scales + expert_row);
    const T* x_pair = x + (size_t)pair * num_symbols * QTIP_V;

    const int sym_per_thread = (num_symbols + THREADS - 1) / THREADS;
    const int sym_start_raw  = tid * sym_per_thread;
    const int sym_end        = min(num_symbols, sym_start_raw + sym_per_thread);

    float acc = 0.0f;

    if (sym_start_raw < num_symbols) {
        uint32_t state = 0;
        int sym_idx = sym_start_raw;

        if (tid > 0) {
            int warm_start = max(0, sym_start_raw - (int)QTIP_WARMUP_SYMS);
            for (int t = warm_start; t < sym_start_raw; ++t) {
                int byte_idx = t >> 1;
                uint8_t byte = __ldg(row_packed + byte_idx);
                uint32_t sym = (t & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
                state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;
            }
        }

        while (sym_idx < sym_end) {
            int byte_idx = sym_idx >> 1;
            uint8_t byte = __ldg(row_packed + byte_idx);

            bool do_lo = ((sym_idx & 1) == 0);
            uint32_t sym_lo = byte & 0x0F;
            uint32_t sym_hi = (byte >> 4) & 0x0F;

            if (do_lo) {
                state = ((state << QTIP_K) | sym_lo) & QTIP_STATE_MASK;
                const float* lut_p = lut + (size_t)state * QTIP_V;
                float w0 = __ldg(lut_p)     * scale;
                float w1 = __ldg(lut_p + 1) * scale;
                int x_off = sym_idx * (int)QTIP_V;
                float x0 = gg_to_f32<T>(__ldg(x_pair + x_off + 0));
                float x1 = gg_to_f32<T>(__ldg(x_pair + x_off + 1));
                acc = fmaf(w0, x0, acc);
                acc = fmaf(w1, x1, acc);
                ++sym_idx;
                if (sym_idx >= sym_end) break;
            }

            state = ((state << QTIP_K) | sym_hi) & QTIP_STATE_MASK;
            const float* lut_p_hi = lut + (size_t)state * QTIP_V;
            float w0 = __ldg(lut_p_hi)     * scale;
            float w1 = __ldg(lut_p_hi + 1) * scale;
            int x_off = sym_idx * (int)QTIP_V;
            float x0 = gg_to_f32<T>(__ldg(x_pair + x_off + 0));
            float x1 = gg_to_f32<T>(__ldg(x_pair + x_off + 1));
            acc = fmaf(w0, x0, acc);
            acc = fmaf(w1, x1, acc);
            ++sym_idx;
        }
    }

    acc = gg_warp_reduce_sum(acc);

    __shared__ float warp_sums[N_WARPS];
    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < N_WARPS) ? warp_sums[lane] : 0.0f;
        v = gg_warp_reduce_sum(v);
        if (lane == 0) {
            y[(size_t)pair * n_rows + row] = gg_from_f32<T>(v);
        }
    }
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

void launch_qtip_gather_gemv_v2_k4_l16_bf16(
    const uint8_t*       d_packed,
    const float*         d_row_scales,
    const float*         d_lut,
    const __nv_bfloat16* d_x_rotated,
    const uint32_t*      d_indices,
    __nv_bfloat16*       d_y,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts,
    cudaStream_t         stream
) {
    constexpr int THREADS = 128;
    dim3 grid(n_rows, n_pairs, 1);
    qtip_gather_gemv_v2_k4_l16_kernel<__nv_bfloat16, THREADS>
        <<<grid, THREADS, 0, stream>>>(
            d_packed, d_row_scales, d_lut, d_x_rotated, d_indices, d_y,
            n_rows, packed_per_row, num_symbols, n_pairs, num_experts);
}

void launch_qtip_gather_gemv_v2_k4_l16_f16(
    const uint8_t* d_packed,
    const float*   d_row_scales,
    const float*   d_lut,
    const __half*  d_x_rotated,
    const uint32_t* d_indices,
    __half*        d_y,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts,
    cudaStream_t   stream
) {
    constexpr int THREADS = 128;
    dim3 grid(n_rows, n_pairs, 1);
    qtip_gather_gemv_v2_k4_l16_kernel<__half, THREADS>
        <<<grid, THREADS, 0, stream>>>(
            d_packed, d_row_scales, d_lut, d_x_rotated, d_indices, d_y,
            n_rows, packed_per_row, num_symbols, n_pairs, num_experts);
}

void launch_qtip_gather_gemv_v2_k4_l16_f32(
    const uint8_t* d_packed,
    const float*   d_row_scales,
    const float*   d_lut,
    const float*   d_x_rotated,
    const uint32_t* d_indices,
    float*         d_y,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts,
    cudaStream_t   stream
) {
    constexpr int THREADS = 128;
    dim3 grid(n_rows, n_pairs, 1);
    qtip_gather_gemv_v2_k4_l16_kernel<float, THREADS>
        <<<grid, THREADS, 0, stream>>>(
            d_packed, d_row_scales, d_lut, d_x_rotated, d_indices, d_y,
            n_rows, packed_per_row, num_symbols, n_pairs, num_experts);
}

} // extern "C"
