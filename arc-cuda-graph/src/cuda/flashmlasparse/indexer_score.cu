/*
 * Lightning Indexer logits kernel for DeepSeek V4 sparse attention.
 *
 * Computes, for each (batch, query), ONE logit per compressed key:
 *
 *     logit[b, q, c] = sum_h relu(dot(q[b,h,q,:], k[b,h,c,:])) * w[b,h,q]
 *
 * The host then runs a top-k radix selection over the last (T_c) axis of these
 * logits, producing a SINGLE sparse-attention key set per (batch, query) that
 * every head shares.
 *
 * This is a port of SGLang's FlashMLASparse `fp8_paged_mqa_logits` family
 * (see sgl-project/FlashMLA, csrc/sm90/decode/sparse_fp8/* and
 * sgl-project/sglang python/sglang/srt/layers/attention/dsv4/indexer.py)
 * specialised to BF16 dense-compressed-K and the Arc V4Indexer interface
 * (matmul + relu + weighted head-sum only — no FP8 dequantisation, no paged K
 * cache, no positional rope inside the score). Phase-2 (rental day-2) will add
 * the paged FP8 variant; the Rust FFI surface stays stable.
 *
 * ## The formula, and why each piece matters
 *
 * SGLang's torch reference `fp8_paged_mqa_logits_torch` (indexer.py 84-89):
 *
 *     score = F.linear(kvcache_value, q)   # [T_c, n_heads]  per-head q.k
 *     score = F.relu(score)                # (1) rectify BEFORE mixing heads
 *     score *= q_scale[None, :]            # per-head weight
 *     score = score.sum(dim=1)             # (2) ONE logit per key
 *     score *= kvcache_scale               # (3) per-key FP8 dequant scale
 *
 *   (1) Without the relu, a strongly negative head dot cancels a strongly
 *       positive one and a different key wins the selection.
 *   (2) Without the head sum, top-k runs per head and emits n_heads (= 64)
 *       distinct key sets, multiplying sparse-gather traffic by 64 — the
 *       shared page set is exactly what makes the gather cheap.
 *   (3) `kvcache_scale` is the per-key FP8 dequantisation scale of SGLang's
 *       paged indexer cache. Arc keeps the indexer K in BF16/F32 (no FP8
 *       indexer cache yet), so it is identically 1 and absent here.
 *
 * `weight_scale = softmax_scale * n_heads^-0.5` (SGLang `C4Indexer.__init__`
 * line 519) is folded into `weights` by the caller before launch, mirroring
 * SGLang's `compute_weights` / `fused_scale`. This kernel therefore receives
 * weights that already carry it.
 *
 * Inputs (all CUDA device pointers):
 *   q:        BF16 [B, n_heads, T_q, head_dim]   (contiguous)
 *   k:        BF16 [B, n_heads, T_c, head_dim]   (contiguous)
 *   weights:  BF16 [B, n_heads, T_q]             (contiguous, weight_scale folded in)
 *
 * Output:
 *   logits:   F32  [B, T_q, T_c]                 (contiguous) — NO head axis.
 *
 * Grid:  (B, T_q, 1) — one block per (batch, query).
 * Block: WARPS_PER_BLOCK warps of 32 threads. Each warp owns a fixed subset of
 *        the T_c columns for the whole head loop, so the head accumulation into
 *        `logits` needs no atomics and its order (h ascending) is deterministic.
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

// Each thread block processes ONE (batch, query) cell and loops over all heads
// internally. Inside the block, multiple warps each own a fixed stride of T_c.
// We pick 4 warps (128 threads) as a balance between occupancy and per-warp
// work.
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlockSize = kWarpSize * kWarpsPerBlock;

__device__ __forceinline__ float warp_reduce_sum(float acc) {
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, 1);
    return acc;
}

// Templated head_dim — V4 default is 128. We allow 64 and 128 (the SGLang
// supported head sizes for the indexer).
template <int HEAD_DIM>
__launch_bounds__(kBlockSize, 4)
__global__ void indexer_logits_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,        // [B, H, T_q, D]
    const __nv_bfloat16* __restrict__ k,        // [B, H, T_c, D]
    const __nv_bfloat16* __restrict__ weights,  // [B, H, T_q]
    float* __restrict__ logits,                 // [B, T_q, T_c]
    int n_heads,
    int t_q,
    int t_c
) {
    const int bi = blockIdx.x;  // batch
    const int qi = blockIdx.y;  // query index within T_q
    if (qi >= t_q) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // Shared staging for the current head's query row [HEAD_DIM].
    __shared__ float q_smem[HEAD_DIM];

    // Output row: logits[bi, qi, :]. Accumulated across heads in place; each
    // column is owned by exactly one warp for the whole loop, so no atomics.
    float* out_row = logits + ((int64_t)bi * t_q + qi) * t_c;
    for (int c = tid; c < t_c; c += kBlockSize) {
        out_row[c] = 0.f;
    }
    __syncthreads();

    for (int h = 0; h < n_heads; ++h) {
        const int64_t bh = (int64_t)bi * n_heads + h;

        // Stage q[bi, h, qi, :] — BF16 in memory, cast to FP32 once.
        const __nv_bfloat16* q_ptr = q + (bh * t_q + qi) * HEAD_DIM;
        for (int d = tid; d < HEAD_DIM; d += kBlockSize) {
            q_smem[d] = __bfloat162float(__ldg(q_ptr + d));
        }
        __syncthreads();

        // Per-head weight for this (batch, query) — same for all T_c columns.
        const float w = __bfloat162float(__ldg(weights + bh * t_q + qi));

        // K row pointer for this (batch, head): K[bi, h, :, :]
        const __nv_bfloat16* k_base = k + bh * (int64_t)t_c * HEAD_DIM;

        // Column `c` is owned by warp (c % kWarpsPerBlock); each warp strides
        // by kWarpsPerBlock. Within a warp, lanes cooperate on the head_dim
        // reduction for ONE column at a time.
        for (int c = warp_id; c < t_c; c += kWarpsPerBlock) {
            const __nv_bfloat16* k_row = k_base + (int64_t)c * HEAD_DIM;

            // Strided BF16 dot product using vectorised uint4 loads (= 8 BF16
            // elements per LDG). HEAD_DIM must be a multiple of 8; for V4 it is
            // (head_dim=128). Each lane processes HEAD_DIM/(32*8) uint4 blocks.
            const int vec_total = HEAD_DIM >> 3;  // # of uint4 across head_dim
            const uint4* kv = reinterpret_cast<const uint4*>(k_row);

            float acc = 0.f;
            for (int i = lane; i < vec_total; i += kWarpSize) {
                uint4 ku = __ldg(kv + i);
                const __nv_bfloat16* kp = reinterpret_cast<const __nv_bfloat16*>(&ku);
                // Corresponding q range is [i*8, i*8+8) in q_smem (FP32).
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

            acc = warp_reduce_sum(acc);

            if (lane == 0) {
                // (1) relu BEFORE mixing heads, then (2) weighted head sum.
                out_row[c] += fmaxf(acc, 0.f) * w;
            }
        }
        // All warps must be done reading q_smem before the next head
        // overwrites it.
        __syncthreads();
    }
}

// Variant for HEAD_DIM not in {64, 128}: a slow generic path that walks
// head_dim element-by-element. Useful for unit tests with small synthetic
// shapes (e.g. head_dim=8 in `v4_indexer_forward_shape`).
__launch_bounds__(kBlockSize, 4)
__global__ void indexer_logits_bf16_generic_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ weights,
    float* __restrict__ logits,
    int n_heads,
    int t_q,
    int t_c,
    int head_dim
) {
    const int bi = blockIdx.x;
    const int qi = blockIdx.y;
    if (qi >= t_q) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // Dynamic shared memory: head_dim floats for the query.
    extern __shared__ float q_smem_dyn[];

    float* out_row = logits + ((int64_t)bi * t_q + qi) * t_c;
    for (int c = tid; c < t_c; c += kBlockSize) {
        out_row[c] = 0.f;
    }
    __syncthreads();

    for (int h = 0; h < n_heads; ++h) {
        const int64_t bh = (int64_t)bi * n_heads + h;
        const __nv_bfloat16* q_ptr = q + (bh * t_q + qi) * head_dim;
        for (int d = tid; d < head_dim; d += kBlockSize) {
            q_smem_dyn[d] = __bfloat162float(__ldg(q_ptr + d));
        }
        __syncthreads();

        const float w = __bfloat162float(__ldg(weights + bh * t_q + qi));
        const __nv_bfloat16* k_base = k + bh * (int64_t)t_c * head_dim;

        for (int c = warp_id; c < t_c; c += kWarpsPerBlock) {
            const __nv_bfloat16* k_row = k_base + (int64_t)c * head_dim;
            float acc = 0.f;
            for (int d = lane; d < head_dim; d += kWarpSize) {
                acc += q_smem_dyn[d] * __bfloat162float(__ldg(k_row + d));
            }
            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                out_row[c] += fmaxf(acc, 0.f) * w;
            }
        }
        __syncthreads();
    }
}

}  // namespace arc_flashmlasparse

// ============================================================================
// C entry point
// ============================================================================

extern "C" void arc_flashmlasparse_logits_bf16(
    const void* q_ptr,           // [B, H, T_q, D] BF16
    const void* k_ptr,           // [B, H, T_c, D] BF16
    const void* weights_ptr,     // [B, H, T_q]    BF16 (weight_scale folded in)
    void* logits_ptr,            // [B, T_q, T_c]  F32  (written)
    int batch,
    int n_heads,
    int t_q,
    int t_c,
    int head_dim,
    cudaStream_t stream
) {
    using namespace arc_flashmlasparse;
    dim3 grid(batch, t_q, 1);
    dim3 block(kBlockSize, 1, 1);
    if (head_dim == 128) {
        indexer_logits_bf16_kernel<128><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_ptr),
            reinterpret_cast<const __nv_bfloat16*>(k_ptr),
            reinterpret_cast<const __nv_bfloat16*>(weights_ptr),
            reinterpret_cast<float*>(logits_ptr),
            n_heads, t_q, t_c
        );
    } else if (head_dim == 64) {
        indexer_logits_bf16_kernel<64><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_ptr),
            reinterpret_cast<const __nv_bfloat16*>(k_ptr),
            reinterpret_cast<const __nv_bfloat16*>(weights_ptr),
            reinterpret_cast<float*>(logits_ptr),
            n_heads, t_q, t_c
        );
    } else {
        // Generic fallback path — slower but supports arbitrary head_dim
        // (including small synthetic test shapes).
        size_t smem_bytes = head_dim * sizeof(float);
        indexer_logits_bf16_generic_kernel<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_ptr),
            reinterpret_cast<const __nv_bfloat16*>(k_ptr),
            reinterpret_cast<const __nv_bfloat16*>(weights_ptr),
            reinterpret_cast<float*>(logits_ptr),
            n_heads, t_q, t_c, head_dim
        );
    }
}

// FP32 input variant: useful for CPU-reference parity tests where Rust holds
// the data in FP32. Same algorithm, different load path.

namespace arc_flashmlasparse {

template <int HEAD_DIM>
__launch_bounds__(kBlockSize, 4)
__global__ void indexer_logits_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ weights,
    float* __restrict__ logits,
    int n_heads,
    int t_q,
    int t_c
) {
    const int bi = blockIdx.x;
    const int qi = blockIdx.y;
    if (qi >= t_q) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ float q_smem[HEAD_DIM];

    float* out_row = logits + ((int64_t)bi * t_q + qi) * t_c;
    for (int c = tid; c < t_c; c += kBlockSize) {
        out_row[c] = 0.f;
    }
    __syncthreads();

    for (int h = 0; h < n_heads; ++h) {
        const int64_t bh = (int64_t)bi * n_heads + h;
        const float* q_ptr = q + (bh * t_q + qi) * HEAD_DIM;
        for (int d = tid; d < HEAD_DIM; d += kBlockSize) {
            q_smem[d] = __ldg(q_ptr + d);
        }
        __syncthreads();

        const float w = __ldg(weights + bh * t_q + qi);
        const float* k_base = k + bh * (int64_t)t_c * HEAD_DIM;

        for (int c = warp_id; c < t_c; c += kWarpsPerBlock) {
            const float* k_row = k_base + (int64_t)c * HEAD_DIM;
            float acc = 0.f;
            for (int d = lane; d < HEAD_DIM; d += kWarpSize) {
                acc += q_smem[d] * __ldg(k_row + d);
            }
            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                out_row[c] += fmaxf(acc, 0.f) * w;
            }
        }
        __syncthreads();
    }
}

__launch_bounds__(kBlockSize, 4)
__global__ void indexer_logits_f32_generic_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ weights,
    float* __restrict__ logits,
    int n_heads,
    int t_q,
    int t_c,
    int head_dim
) {
    const int bi = blockIdx.x;
    const int qi = blockIdx.y;
    if (qi >= t_q) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    extern __shared__ float q_smem_dyn[];

    float* out_row = logits + ((int64_t)bi * t_q + qi) * t_c;
    for (int c = tid; c < t_c; c += kBlockSize) {
        out_row[c] = 0.f;
    }
    __syncthreads();

    for (int h = 0; h < n_heads; ++h) {
        const int64_t bh = (int64_t)bi * n_heads + h;
        const float* q_ptr = q + (bh * t_q + qi) * head_dim;
        for (int d = tid; d < head_dim; d += kBlockSize) {
            q_smem_dyn[d] = __ldg(q_ptr + d);
        }
        __syncthreads();

        const float w = __ldg(weights + bh * t_q + qi);
        const float* k_base = k + bh * (int64_t)t_c * head_dim;

        for (int c = warp_id; c < t_c; c += kWarpsPerBlock) {
            const float* k_row = k_base + (int64_t)c * head_dim;
            float acc = 0.f;
            for (int d = lane; d < head_dim; d += kWarpSize) {
                acc += q_smem_dyn[d] * __ldg(k_row + d);
            }
            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                out_row[c] += fmaxf(acc, 0.f) * w;
            }
        }
        __syncthreads();
    }
}

}  // namespace arc_flashmlasparse

extern "C" void arc_flashmlasparse_logits_f32(
    const void* q_ptr,
    const void* k_ptr,
    const void* weights_ptr,
    void* logits_ptr,
    int batch,
    int n_heads,
    int t_q,
    int t_c,
    int head_dim,
    cudaStream_t stream
) {
    using namespace arc_flashmlasparse;
    dim3 grid(batch, t_q, 1);
    dim3 block(kBlockSize, 1, 1);
    if (head_dim == 128) {
        indexer_logits_f32_kernel<128><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(q_ptr),
            reinterpret_cast<const float*>(k_ptr),
            reinterpret_cast<const float*>(weights_ptr),
            reinterpret_cast<float*>(logits_ptr),
            n_heads, t_q, t_c
        );
    } else if (head_dim == 64) {
        indexer_logits_f32_kernel<64><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(q_ptr),
            reinterpret_cast<const float*>(k_ptr),
            reinterpret_cast<const float*>(weights_ptr),
            reinterpret_cast<float*>(logits_ptr),
            n_heads, t_q, t_c
        );
    } else {
        size_t smem_bytes = head_dim * sizeof(float);
        indexer_logits_f32_generic_kernel<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const float*>(q_ptr),
            reinterpret_cast<const float*>(k_ptr),
            reinterpret_cast<const float*>(weights_ptr),
            reinterpret_cast<float*>(logits_ptr),
            n_heads, t_q, t_c, head_dim
        );
    }
}
