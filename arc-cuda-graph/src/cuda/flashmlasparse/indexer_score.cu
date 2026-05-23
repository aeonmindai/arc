/*
 * Lightning Indexer scoring kernel for DeepSeek V4 sparse attention.
 *
 * Computes per-(batch, head, query) dot-product scores between indexer queries
 * and compressed indexer keys, scaled by a per-(batch, head, query) factor
 * from `weights_proj(xs)`. The host then runs a top-k radix selection over the
 * last (T_c) axis of these scores to produce sparse-attention indices.
 *
 * This is a port of SGLang's FlashMLASparse `fp8_paged_mqa_logits` family
 * (see sgl-project/FlashMLA, csrc/sm90/decode/sparse_fp8/* and
 * sgl-project/sglang python/sglang/srt/layers/attention/dsv4/tilelang_kernel.py)
 * specialised to BF16 dense-compressed-K and the Arc V4Indexer interface
 * (matmul + per-head scale only — no FP8 dequantisation, no paged K cache,
 * no positional rope inside the score). Phase-2 (rental day-2) will add the
 * paged FP8 variant; the Rust FFI surface stays stable.
 *
 * Inputs (all CUDA device pointers):
 *   q:               BF16 [B, n_heads, T_q, head_dim]   (contiguous)
 *   k:               BF16 [B, n_heads, T_c, head_dim]   (contiguous)
 *   per_head_scale:  BF16 [B, n_heads, T_q]             (contiguous)
 *
 * Output:
 *   scores:          F32  [B, n_heads, T_q, T_c]        (contiguous)
 *
 * Grid:  (B * n_heads * T_q, 1, 1) — one block per (batch, head, query).
 * Block: WARPS_PER_BLOCK warps of 32 threads. Each warp computes one row of
 *        T_c output values cooperatively.
 *
 * Original FlashMLA / SGLang sources are MIT-licensed (Copyright 2025
 * DeepSeek; see https://github.com/sgl-project/FlashMLA/blob/main/LICENSE).
 * This file is a derived work re-licensed under MIT for the dense-BF16 Arc
 * port; original copyright preserved below.
 *
 * SPDX-License-Identifier: MIT
 * Derived from: sgl-project/FlashMLA (MIT) + sgl-project/sglang DSV4 indexer (MIT).
 *   Copyright (c) 2025 DeepSeek
 *   Copyright (c) 2025 SGLang contributors
 *   Copyright (c) 2026 Arc / Runcrate
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

namespace arc_flashmlasparse {

// Each thread block processes ONE (batch, head, query) cell. Inside the block,
// multiple warps each handle a chunk of T_c. We pick 4 warps (128 threads) as
// a balance between occupancy and per-warp work.
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlockSize = kWarpSize * kWarpsPerBlock;

// Templated head_dim — V4 default is 128. We allow 64 and 128 (the SGLang
// supported head sizes for the indexer).
template <int HEAD_DIM>
__launch_bounds__(kBlockSize, 4)
__global__ void indexer_score_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,           // [B*H, T_q, D]
    const __nv_bfloat16* __restrict__ k,           // [B*H, T_c, D]
    const __nv_bfloat16* __restrict__ per_head_scale,  // [B*H, T_q]
    float* __restrict__ scores,                    // [B*H, T_q, T_c]
    int n_heads_per_batch,                          // = H = n_heads
    int t_q,
    int t_c
) {
    // Block index → (b*H+h, t_q_idx). One block per (batch_head, query).
    const int bh = blockIdx.x;  // batch * n_heads + head
    const int qi = blockIdx.y;  // query index within T_q
    if (qi >= t_q) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // Load the query row [HEAD_DIM] into shared memory (each block uses one
    // query). The data is in BF16 — we cast to FP32 once at load.
    __shared__ float q_smem[HEAD_DIM];

    // Stride into q: q[bh, qi, :]
    const __nv_bfloat16* q_ptr = q + ((int64_t)bh * t_q + qi) * HEAD_DIM;
    #pragma unroll
    for (int d = tid; d < HEAD_DIM; d += kBlockSize) {
        q_smem[d] = __bfloat162float(__ldg(q_ptr + d));
    }

    // Per-head scale (same for all T_c columns of this (bh, qi) cell). Read
    // once into a register replicated across the block.
    __shared__ float scale_smem;
    if (tid == 0) {
        scale_smem = __bfloat162float(__ldg(per_head_scale + (int64_t)bh * t_q + qi));
    }
    __syncthreads();
    const float head_scale = scale_smem;

    // K row pointer for this (batch_head): K[bh, :, :]
    const __nv_bfloat16* k_base = k + (int64_t)bh * t_c * HEAD_DIM;

    // Output row: scores[bh, qi, :]
    float* out_row = scores + ((int64_t)bh * t_q + qi) * t_c;

    // Each warp handles a stride of T_c columns. Within a warp, threads
    // cooperate on the head_dim reduction for ONE column at a time.
    //
    // Layout: column `c` is owned by warp (c % kWarpsPerBlock), each warp
    // strides by kWarpsPerBlock. This gives a round-robin assignment.
    for (int c = warp_id; c < t_c; c += kWarpsPerBlock) {
        const __nv_bfloat16* k_row = k_base + (int64_t)c * HEAD_DIM;

        // Strided BF16 dot product using vectorised uint4 loads (= 8 BF16
        // elements per LDG). HEAD_DIM must be a multiple of 8; for V4 it is
        // (head_dim=128). Each lane processes HEAD_DIM/(32*8) uint4 blocks.
        const int vec_total = HEAD_DIM >> 3;  // # of uint4 across head_dim
        const uint4* qv = reinterpret_cast<const uint4*>(q_smem);
        const uint4* kv = reinterpret_cast<const uint4*>(k_row);

        // For HEAD_DIM=128 → 16 uint4s; with 32 lanes, each lane processes 0
        // or 1. For HEAD_DIM=64 → 8 uint4s; each lane handles 0..1 strided.
        float acc = 0.f;
        for (int i = lane; i < vec_total; i += kWarpSize) {
            // Reinterpret q_smem (FP32) — we stored FP32, so each uint4 holds
            // 4 FP32. We instead reload K as BF16 (uint4 = 8 BF16) and pair
            // with q_smem accessed as FP32 in groups of 4.
            //
            // Hybrid layout: q_smem is FP32 (4 floats per uint4), k_row is
            // BF16 (8 bf16 per uint4). To keep the loop body simple we
            // re-derive offsets in terms of HEAD_DIM elements.
            //
            // For BF16-pair from K: load uint4 of 8 bf16s.
            uint4 ku = __ldg(kv + i);
            const __nv_bfloat16* kp = reinterpret_cast<const __nv_bfloat16*>(&ku);
            // Corresponding q range is [i*8, i*8+8) in q_smem.
            const float* qp = q_smem + i * 8;
            acc += qp[0] * __bfloat162float(kp[0])
                 + qp[1] * __bfloat162float(kp[1])
                 + qp[2] * __bfloat162float(kp[2])
                 + qp[3] * __bfloat162float(kp[3])
                 + qp[4] * __bfloat162float(kp[4])
                 + qp[5] * __bfloat162float(kp[5])
                 + qp[6] * __bfloat162float(kp[6])
                 + qp[7] * __bfloat162float(kp[7]);
        }

        // Warp reduce.
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 16);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 8);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 4);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 2);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 1);

        if (lane == 0) {
            // Scale and store as FP32. The top-k kernel consumes FP32 scores.
            out_row[c] = acc * head_scale;
        }
    }
}

// Variant for HEAD_DIM not in {64, 128}: a slow generic path that walks
// head_dim element-by-element. Useful for unit tests with small synthetic
// shapes (e.g. head_dim=8 in `v4_indexer_forward_shape`).
__launch_bounds__(kBlockSize, 4)
__global__ void indexer_score_bf16_generic_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ per_head_scale,
    float* __restrict__ scores,
    int n_heads_per_batch,
    int t_q,
    int t_c,
    int head_dim
) {
    const int bh = blockIdx.x;
    const int qi = blockIdx.y;
    if (qi >= t_q) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // Dynamic shared memory: head_dim floats for the query.
    extern __shared__ float q_smem_dyn[];

    const __nv_bfloat16* q_ptr = q + ((int64_t)bh * t_q + qi) * head_dim;
    for (int d = tid; d < head_dim; d += kBlockSize) {
        q_smem_dyn[d] = __bfloat162float(__ldg(q_ptr + d));
    }

    __shared__ float scale_smem;
    if (tid == 0) {
        scale_smem = __bfloat162float(__ldg(per_head_scale + (int64_t)bh * t_q + qi));
    }
    __syncthreads();
    const float head_scale = scale_smem;

    const __nv_bfloat16* k_base = k + (int64_t)bh * t_c * head_dim;
    float* out_row = scores + ((int64_t)bh * t_q + qi) * t_c;

    for (int c = warp_id; c < t_c; c += kWarpsPerBlock) {
        const __nv_bfloat16* k_row = k_base + (int64_t)c * head_dim;
        float acc = 0.f;
        for (int d = lane; d < head_dim; d += kWarpSize) {
            acc += q_smem_dyn[d] * __bfloat162float(__ldg(k_row + d));
        }
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 16);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 8);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 4);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 2);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 1);
        if (lane == 0) {
            out_row[c] = acc * head_scale;
        }
    }
}

}  // namespace arc_flashmlasparse

// ============================================================================
// C entry point
// ============================================================================

extern "C" void arc_flashmlasparse_score_bf16(
    const void* q_ptr,           // [B, H, T_q, D] BF16
    const void* k_ptr,           // [B, H, T_c, D] BF16
    const void* scale_ptr,       // [B, H, T_q]   BF16
    void* scores_ptr,            // [B, H, T_q, T_c] F32
    int batch,
    int n_heads,
    int t_q,
    int t_c,
    int head_dim,
    cudaStream_t stream
) {
    using namespace arc_flashmlasparse;
    dim3 grid(batch * n_heads, t_q, 1);
    dim3 block(kBlockSize, 1, 1);
    if (head_dim == 128) {
        indexer_score_bf16_kernel<128><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_ptr),
            reinterpret_cast<const __nv_bfloat16*>(k_ptr),
            reinterpret_cast<const __nv_bfloat16*>(scale_ptr),
            reinterpret_cast<float*>(scores_ptr),
            n_heads, t_q, t_c
        );
    } else if (head_dim == 64) {
        indexer_score_bf16_kernel<64><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_ptr),
            reinterpret_cast<const __nv_bfloat16*>(k_ptr),
            reinterpret_cast<const __nv_bfloat16*>(scale_ptr),
            reinterpret_cast<float*>(scores_ptr),
            n_heads, t_q, t_c
        );
    } else {
        // Generic fallback path — slower but supports arbitrary head_dim
        // (including small synthetic test shapes).
        size_t smem_bytes = head_dim * sizeof(float);
        indexer_score_bf16_generic_kernel<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_ptr),
            reinterpret_cast<const __nv_bfloat16*>(k_ptr),
            reinterpret_cast<const __nv_bfloat16*>(scale_ptr),
            reinterpret_cast<float*>(scores_ptr),
            n_heads, t_q, t_c, head_dim
        );
    }
}

// FP32 input variant: useful for CPU-reference parity tests where Rust holds
// the data in FP32. Same algorithm, different load path.

namespace arc_flashmlasparse {

template <int HEAD_DIM>
__launch_bounds__(kBlockSize, 4)
__global__ void indexer_score_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ per_head_scale,
    float* __restrict__ scores,
    int n_heads_per_batch,
    int t_q,
    int t_c
) {
    const int bh = blockIdx.x;
    const int qi = blockIdx.y;
    if (qi >= t_q) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ float q_smem[HEAD_DIM];
    const float* q_ptr = q + ((int64_t)bh * t_q + qi) * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += kBlockSize) {
        q_smem[d] = __ldg(q_ptr + d);
    }

    __shared__ float scale_smem;
    if (tid == 0) {
        scale_smem = __ldg(per_head_scale + (int64_t)bh * t_q + qi);
    }
    __syncthreads();
    const float head_scale = scale_smem;

    const float* k_base = k + (int64_t)bh * t_c * HEAD_DIM;
    float* out_row = scores + ((int64_t)bh * t_q + qi) * t_c;

    for (int c = warp_id; c < t_c; c += kWarpsPerBlock) {
        const float* k_row = k_base + (int64_t)c * HEAD_DIM;
        float acc = 0.f;
        for (int d = lane; d < HEAD_DIM; d += kWarpSize) {
            acc += q_smem[d] * __ldg(k_row + d);
        }
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 16);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 8);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 4);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 2);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 1);
        if (lane == 0) {
            out_row[c] = acc * head_scale;
        }
    }
}

__launch_bounds__(kBlockSize, 4)
__global__ void indexer_score_f32_generic_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ per_head_scale,
    float* __restrict__ scores,
    int n_heads_per_batch,
    int t_q,
    int t_c,
    int head_dim
) {
    const int bh = blockIdx.x;
    const int qi = blockIdx.y;
    if (qi >= t_q) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    extern __shared__ float q_smem_dyn[];
    const float* q_ptr = q + ((int64_t)bh * t_q + qi) * head_dim;
    for (int d = tid; d < head_dim; d += kBlockSize) {
        q_smem_dyn[d] = __ldg(q_ptr + d);
    }

    __shared__ float scale_smem;
    if (tid == 0) {
        scale_smem = __ldg(per_head_scale + (int64_t)bh * t_q + qi);
    }
    __syncthreads();
    const float head_scale = scale_smem;

    const float* k_base = k + (int64_t)bh * t_c * head_dim;
    float* out_row = scores + ((int64_t)bh * t_q + qi) * t_c;

    for (int c = warp_id; c < t_c; c += kWarpsPerBlock) {
        const float* k_row = k_base + (int64_t)c * head_dim;
        float acc = 0.f;
        for (int d = lane; d < head_dim; d += kWarpSize) {
            acc += q_smem_dyn[d] * __ldg(k_row + d);
        }
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 16);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 8);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 4);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 2);
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, 1);
        if (lane == 0) {
            out_row[c] = acc * head_scale;
        }
    }
}

}  // namespace arc_flashmlasparse

extern "C" void arc_flashmlasparse_score_f32(
    const void* q_ptr,
    const void* k_ptr,
    const void* scale_ptr,
    void* scores_ptr,
    int batch,
    int n_heads,
    int t_q,
    int t_c,
    int head_dim,
    cudaStream_t stream
) {
    using namespace arc_flashmlasparse;
    dim3 grid(batch * n_heads, t_q, 1);
    dim3 block(kBlockSize, 1, 1);
    if (head_dim == 128) {
        indexer_score_f32_kernel<128><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(q_ptr),
            reinterpret_cast<const float*>(k_ptr),
            reinterpret_cast<const float*>(scale_ptr),
            reinterpret_cast<float*>(scores_ptr),
            n_heads, t_q, t_c
        );
    } else if (head_dim == 64) {
        indexer_score_f32_kernel<64><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(q_ptr),
            reinterpret_cast<const float*>(k_ptr),
            reinterpret_cast<const float*>(scale_ptr),
            reinterpret_cast<float*>(scores_ptr),
            n_heads, t_q, t_c
        );
    } else {
        size_t smem_bytes = head_dim * sizeof(float);
        indexer_score_f32_generic_kernel<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const float*>(q_ptr),
            reinterpret_cast<const float*>(k_ptr),
            reinterpret_cast<const float*>(scale_ptr),
            reinterpret_cast<float*>(scores_ptr),
            n_heads, t_q, t_c, head_dim
        );
    }
}
