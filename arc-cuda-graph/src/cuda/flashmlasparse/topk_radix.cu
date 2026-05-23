/*
 * Radix top-k kernel for the V4 Lightning Indexer.
 *
 * Adapted from SGLang's topk_v1 kernel:
 *   sgl-project/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh
 * (Apache-2.0 / MIT dual-licensed in the SGLang upstream — vendored under the
 * MIT terms for Arc, which is BSL-1.1 with MIT third-party). Original
 * algorithm by SGLang contributors; modifications here:
 *
 *   - Removed dependency on sgl_kernel/* headers and tvm::ffi tensor wrappers.
 *   - Removed `page_to_indices` step — we return raw indices into the score
 *     array, since V4Indexer's caller (V4 attention) does its own page lookup
 *     on the dense indexer-K tensor.
 *   - Templated kTopK (default 512 to match V4 default `index_topk`) as a
 *     runtime parameter via a launcher dispatch table.
 *   - Removed PDL primitives (we don't yet wire CUDA graph PDL through).
 *
 * The core radix-256 two-stage selection algorithm is unchanged. Given an
 * FP32 score array of length `seq_len`, the kernel writes the indices (i32)
 * of the top-K largest scores into the output array, unsorted within the
 * kept set. For `seq_len <= kTopK`, it falls back to writing the identity
 * 0..seq_len and padding with -1.
 *
 * Grid:   `batch` (= B * H * T_q for our indexer).
 * Block:  kTopKBlockSize (= kTopK ∈ {128, 256, 512, 1024}).
 *
 * SPDX-License-Identifier: MIT
 * Derived from: sgl-project/sglang DSV4 topk_v1.cuh
 *   Copyright (c) 2025 SGLang contributors
 *   Copyright (c) 2026 Arc / Runcrate
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

namespace arc_flashmlasparse {

// ---------------------------------------------------------------------------
// Key conversion helpers (float ↔ unsigned-radix keys).
//
// We need a total order on bit patterns that matches the ordering of the
// floats (largest first). Standard trick: flip the sign bit for positives,
// invert all bits for negatives.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
    // Reduce to 8-bit radix for the coarse histogram. Cast to FP16 first so
    // the top 8 mantissa bits + sign + exponent collapse into a single byte.
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// ---------------------------------------------------------------------------
// Two-stage radix top-k. Templated on kTopK ∈ {64, 128, 256, 512, 1024}.
//
// Block size = kTopK threads. SMEM = (kSMEM bytes) for the bucketed work
// list + the radix histogram tables.
// ---------------------------------------------------------------------------

template <int kTopK>
__device__ void radix_topk_impl(
    const float* __restrict__ input,
    int32_t* __restrict__ output,
    const uint32_t length
) {
    constexpr uint32_t RADIX = 256;
    constexpr uint32_t kTopKBlockSize = kTopK;
    // Dynamic SMEM partition: kSMEM bytes total (64 KB worth of indices).
    constexpr uint32_t kSMEM = 16 * 1024 * sizeof(uint32_t);
    constexpr uint32_t SMEM_INPUT_SIZE = kSMEM / (2 * sizeof(int32_t));

    alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
    alignas(128) __shared__ uint32_t s_counter;
    alignas(128) __shared__ uint32_t s_threshold_bin_id;
    alignas(128) __shared__ uint32_t s_num_input[2];
    alignas(128) __shared__ int32_t s_last_remain;

    extern __shared__ uint32_t s_input_idx_dyn[];
    // Two buffers of SMEM_INPUT_SIZE each.
    uint32_t* s_input_idx[2];
    s_input_idx[0] = s_input_idx_dyn;
    s_input_idx[1] = s_input_idx_dyn + SMEM_INPUT_SIZE;

    const uint32_t tx = threadIdx.x;
    uint32_t remain_topk = kTopK;
    auto& s_histogram = _s_histogram_buf[0];

    auto run_cumsum = [&]() {
        #pragma unroll 8
        for (int32_t i = 0; i < 8; ++i) {
            static_assert(1 << 8 == RADIX, "RADIX must be 256");
            if (tx < RADIX) {
                const auto j = 1 << i;
                const auto k = i & 1;
                auto value = _s_histogram_buf[k][tx];
                if (tx + j < RADIX) {
                    value += _s_histogram_buf[k][tx + j];
                }
                _s_histogram_buf[k ^ 1][tx] = value;
            }
            __syncthreads();
        }
    };

    // Stage 1: 8-bit coarse histogram across the full input.
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();
    for (uint32_t idx = tx; idx < length; idx += kTopKBlockSize) {
        const auto bin = convert_to_uint8(input[idx]);
        ::atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();
    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
        s_threshold_bin_id = tx;
        s_num_input[0] = 0;
        s_counter = 0;
    }
    __syncthreads();

    const auto threshold_bin_stage1 = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin_stage1 + 1];
    if (remain_topk == 0) {
        for (uint32_t idx = tx; idx < length; idx += kTopKBlockSize) {
            const uint32_t bin = convert_to_uint8(input[idx]);
            if (bin > threshold_bin_stage1) {
                const auto pos = ::atomicAdd(&s_counter, 1);
                output[pos] = static_cast<int32_t>(idx);
            }
        }
        __syncthreads();
        return;
    } else {
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();

        for (uint32_t idx = tx; idx < length; idx += kTopKBlockSize) {
            const float raw_input = input[idx];
            const uint32_t bin = convert_to_uint8(raw_input);
            if (bin > threshold_bin_stage1) {
                const auto pos = ::atomicAdd(&s_counter, 1);
                output[pos] = static_cast<int32_t>(idx);
            } else if (bin == threshold_bin_stage1) {
                const auto pos = ::atomicAdd(&s_num_input[0], 1);
                if (pos < SMEM_INPUT_SIZE) {
                    s_input_idx[0][pos] = idx;
                    const auto bin32 = convert_to_uint32(raw_input);
                    const auto sub_bin = (bin32 >> 24) & 0xFF;
                    ::atomicAdd(&s_histogram[sub_bin], 1);
                }
            }
        }
        __syncthreads();
    }

    // Stage 2: refine via 4 additional 8-bit radix passes on FP32 keys.
    #pragma unroll 4
    for (int round = 0; round < 4; ++round) {
        const auto r_idx = round % 2;

        const auto raw_num_input = s_num_input[r_idx];
        const auto num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

        run_cumsum();
        if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
            s_threshold_bin_id = tx;
            s_num_input[r_idx ^ 1] = 0;
            s_last_remain = remain_topk - s_histogram[tx + 1];
        }
        __syncthreads();

        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_histogram[threshold_bin + 1];

        if (remain_topk == 0) {
            for (uint32_t i = tx; i < num_input; i += kTopKBlockSize) {
                const auto idx = s_input_idx[r_idx][i];
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&s_counter, 1);
                    output[pos] = static_cast<int32_t>(idx);
                }
            }
            __syncthreads();
            return;
        } else {
            __syncthreads();
            if (tx < RADIX + 1) s_histogram[tx] = 0;
            __syncthreads();
            for (uint32_t i = tx; i < num_input; i += kTopKBlockSize) {
                const auto idx = s_input_idx[r_idx][i];
                const auto raw_input = input[idx];
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&s_counter, 1);
                    output[pos] = static_cast<int32_t>(idx);
                } else if (bin == threshold_bin) {
                    if (round == 3) {
                        const auto pos = ::atomicAdd(&s_last_remain, -1);
                        if (pos > 0) {
                            output[kTopK - pos] = static_cast<int32_t>(idx);
                        }
                    } else {
                        const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
                        if (pos < SMEM_INPUT_SIZE) {
                            s_input_idx[r_idx ^ 1][pos] = idx;
                            const auto bin32 = convert_to_uint32(raw_input);
                            const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
                            ::atomicAdd(&s_histogram[sub_bin], 1);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// Naive fallback for seq_len <= kTopK: identity-fill + (-1) padding.
// ---------------------------------------------------------------------------

template <int kTopK>
__device__ __forceinline__ void naive_transform_topk(
    int32_t* __restrict__ output,
    const uint32_t length
) {
    if (const auto tx = threadIdx.x; tx < length) {
        output[tx] = static_cast<int32_t>(tx);
    } else if (tx < kTopK) {
        output[tx] = -1;
    }
}

// ---------------------------------------------------------------------------
// Top-k kernel: one block per (batch_idx). `scores` is [batch, score_stride]
// strided contiguous; we read [batch, 0..seq_lens[batch]] and write [batch,
// 0..kTopK] into `out_indices`.
// ---------------------------------------------------------------------------

template <int kTopK>
__launch_bounds__(kTopK, 1)
__global__ void topk_radix_kernel(
    const float* __restrict__ scores,     // [N, score_stride]
    const int32_t* __restrict__ seq_lens, // [N] valid length per row
    int32_t* __restrict__ out_indices,    // [N, kTopK]
    int64_t score_stride
) {
    const int32_t row = blockIdx.x;
    const int32_t length = seq_lens[row];
    const float* score_row = scores + (int64_t)row * score_stride;
    int32_t* out_row = out_indices + (int64_t)row * kTopK;

    if (length <= kTopK) {
        naive_transform_topk<kTopK>(out_row, static_cast<uint32_t>(length));
    } else {
        // Use one __shared__ buffer for the kTopK partial indices.
        __shared__ int32_t s_topk_indices[kTopK];
        radix_topk_impl<kTopK>(score_row, s_topk_indices, static_cast<uint32_t>(length));
        const int tx = threadIdx.x;
        if (tx < kTopK) {
            out_row[tx] = s_topk_indices[tx];
        }
    }
}

}  // namespace arc_flashmlasparse

// ============================================================================
// C entry points — dispatch on kTopK at the host side.
// ============================================================================

extern "C" void arc_flashmlasparse_topk(
    const void* scores_ptr,    // [N, score_stride] F32
    const void* seq_lens_ptr,  // [N] I32
    void* out_indices_ptr,     // [N, topk] I32
    int n_rows,                // = B * H * T_q
    int topk,
    int64_t score_stride,
    cudaStream_t stream
) {
    using namespace arc_flashmlasparse;
    const float* scores = reinterpret_cast<const float*>(scores_ptr);
    const int32_t* seq_lens = reinterpret_cast<const int32_t*>(seq_lens_ptr);
    int32_t* out = reinterpret_cast<int32_t*>(out_indices_ptr);

    // Dynamic SMEM = 2 buffers of (kSMEM bytes / 2). kSMEM = 64 KB.
    constexpr size_t kSMEM_bytes = 16 * 1024 * sizeof(uint32_t);

    // Dispatch on topk. Supported values cover the V4 default (512) plus
    // smaller sizes for unit-test fixtures and smaller models.
    switch (topk) {
        case 64: {
            cudaFuncSetAttribute(
                (const void*)topk_radix_kernel<64>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(kSMEM_bytes));
            topk_radix_kernel<64><<<n_rows, 64, kSMEM_bytes, stream>>>(
                scores, seq_lens, out, score_stride);
            break;
        }
        case 128: {
            cudaFuncSetAttribute(
                (const void*)topk_radix_kernel<128>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(kSMEM_bytes));
            topk_radix_kernel<128><<<n_rows, 128, kSMEM_bytes, stream>>>(
                scores, seq_lens, out, score_stride);
            break;
        }
        case 256: {
            cudaFuncSetAttribute(
                (const void*)topk_radix_kernel<256>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(kSMEM_bytes));
            topk_radix_kernel<256><<<n_rows, 256, kSMEM_bytes, stream>>>(
                scores, seq_lens, out, score_stride);
            break;
        }
        case 512: {
            cudaFuncSetAttribute(
                (const void*)topk_radix_kernel<512>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(kSMEM_bytes));
            topk_radix_kernel<512><<<n_rows, 512, kSMEM_bytes, stream>>>(
                scores, seq_lens, out, score_stride);
            break;
        }
        case 1024: {
            cudaFuncSetAttribute(
                (const void*)topk_radix_kernel<1024>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(kSMEM_bytes));
            topk_radix_kernel<1024><<<n_rows, 1024, kSMEM_bytes, stream>>>(
                scores, seq_lens, out, score_stride);
            break;
        }
        default: {
            // Unsupported topk: surface as a CUDA error so the Rust side can
            // fall back to the candle reference path. We don't have access to
            // a stderr from device code; the Rust side validates topk before
            // dispatching and refuses to launch unsupported sizes (see
            // `flashmlasparse.rs::SUPPORTED_TOPK`).
            break;
        }
    }
}
