/**
 * Fused decode kernels — combine multiple operations into single launches.
 *
 * Saves per-kernel launch overhead (~5μs each) and L2 round-trips between
 * adjacent operations. All kernels are generic (no model assumptions).
 *
 * Fusions:
 * 1. rmsnorm_gemv_bf16: RMSNorm + projection GEMV in one kernel
 *    (input stays in shared memory between norm and matmul)
 * 2. qknorm_rope_bf16: Q norm + K norm + RoPE in one kernel
 *    (qknorm optional via flag for models without it)
 *
 * No cuBLAS, no allocations — fully graph-capture compatible.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// =============================================================================
// 1. Fused RMSNorm + GEMV
// =============================================================================
//
// output[m] = sum_k(weight[m,k] * (input[k] * rsqrt(mean(input^2)+eps) * norm_weight[k]))
//
// Each block:
//   - Loads input into shared memory
//   - Computes rsqrt(mean(input^2)+eps) via block reduction
//   - Writes normalized values back to shared memory
//   - 8 warps each compute 1 output row of the GEMV
//
// Block size: 256 threads (8 warps), 8 rows per block.
// Shared memory: K * sizeof(bf16) for normalized input + 32 floats for reduction.

#define RG_WARP 32
#define RG_WARPS 8
#define RG_BLOCK (RG_WARP * RG_WARPS) // 256
#define RG_ROWS RG_WARPS

__global__ __launch_bounds__(256, 4)
void rmsnorm_gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,        // [K] hidden state
    const __nv_bfloat16* __restrict__ norm_weight,  // [K] norm gain
    const __nv_bfloat16* __restrict__ weight,       // [M, K] projection weight (row-major)
    __nv_bfloat16* __restrict__ output,             // [M] output
    int M, int K, float eps
) {
    extern __shared__ __nv_bfloat16 s_normed[]; // K bf16 elements

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // ---- Phase 1: load input, compute sum of squares ----
    float local_sumsq = 0.0f;
    for (int i = tid; i < K; i += RG_BLOCK) {
        float v = __bfloat162float(input[i]);
        local_sumsq += v * v;
    }

    // Warp-level reduction
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        local_sumsq += __shfl_xor_sync(0xFFFFFFFF, local_sumsq, o);
    }

    // Block-level reduction (8 warps → 1 value)
    __shared__ float s_warp_sums[RG_WARPS];
    if (lane == 0) s_warp_sums[warp] = local_sumsq;
    __syncthreads();

    float rms;
    if (warp == 0) {
        float v = (lane < RG_WARPS) ? s_warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, o);
        }
        if (lane == 0) s_warp_sums[0] = rsqrtf(v / (float)K + eps);
    }
    __syncthreads();
    rms = s_warp_sums[0];

    // ---- Phase 2: write normalized values to shared memory ----
    for (int i = tid; i < K; i += RG_BLOCK) {
        float v = __bfloat162float(input[i]) * rms * __bfloat162float(norm_weight[i]);
        s_normed[i] = __float2bfloat16(v);
    }
    __syncthreads();

    // ---- Phase 3: GEMV — each warp computes 1 row ----
    int row = blockIdx.x * RG_ROWS + warp;
    if (row >= M) return;

    const __nv_bfloat16* w = weight + (int64_t)row * K;
    float acc = 0.0f;

    const int K8 = K >> 3;
    const uint4* wv = (const uint4*)w;
    const uint4* sv = (const uint4*)s_normed;

    for (int i = lane; i < K8; i += RG_WARP) {
        uint4 a = __ldg(wv + i);
        uint4 b = sv[i]; // shared memory read
        const __nv_bfloat16* ap = (const __nv_bfloat16*)&a;
        const __nv_bfloat16* bp = (const __nv_bfloat16*)&b;
        acc += __bfloat162float(ap[0]) * __bfloat162float(bp[0])
             + __bfloat162float(ap[1]) * __bfloat162float(bp[1])
             + __bfloat162float(ap[2]) * __bfloat162float(bp[2])
             + __bfloat162float(ap[3]) * __bfloat162float(bp[3])
             + __bfloat162float(ap[4]) * __bfloat162float(bp[4])
             + __bfloat162float(ap[5]) * __bfloat162float(bp[5])
             + __bfloat162float(ap[6]) * __bfloat162float(bp[6])
             + __bfloat162float(ap[7]) * __bfloat162float(bp[7]);
    }

    // Warp reduction
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    if (lane == 0) output[row] = __float2bfloat16(acc);
}

extern "C" void arc_launch_rmsnorm_gemv_bf16(
    const void* input, const void* norm_weight, const void* weight, void* output,
    int M, int K, float eps, cudaStream_t stream
) {
    dim3 grid((M + RG_ROWS - 1) / RG_ROWS);
    int smem = K * sizeof(__nv_bfloat16);
    rmsnorm_gemv_bf16_kernel<<<grid, RG_BLOCK, smem, stream>>>(
        (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)norm_weight,
        (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)output, M, K, eps
    );
}

// =============================================================================
// 2. Fused Q norm + K norm + RoPE
// =============================================================================
//
// Per layer, these 3 operations all touch Q and K. Combine into one kernel:
//   - Per-head RMSNorm of Q (if has_qknorm)
//   - Per-head RMSNorm of K (if has_qknorm)
//   - Apply RoPE rotation to Q and K
//
// Generic: works with or without qknorm (flag), supports neox and gptj RoPE styles.
// One block per (head, batch). 128 threads per block (head_dim = 128).

__global__ void qknorm_rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,                  // [batch, num_heads * head_dim]
    __nv_bfloat16* __restrict__ k,                  // [batch, num_kv_heads * head_dim]
    const __nv_bfloat16* __restrict__ q_norm_w,     // [head_dim] or NULL
    const __nv_bfloat16* __restrict__ k_norm_w,     // [head_dim] or NULL
    const __nv_bfloat16* __restrict__ cos_table,    // [max_pos, head_dim/2]
    const __nv_bfloat16* __restrict__ sin_table,    // [max_pos, head_dim/2]
    const int32_t* __restrict__ positions,          // [batch]
    int num_heads, int num_kv_heads, int head_dim,
    int batch_size, int is_neox, float eps
) {
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const bool is_q = (head_idx < num_heads);
    const int local_head = is_q ? head_idx : (head_idx - num_heads);
    const int total_heads = is_q ? num_heads : num_kv_heads;

    if (!is_q && local_head >= num_kv_heads) return;

    __nv_bfloat16* tensor = is_q ? q : k;
    const __nv_bfloat16* norm_w = is_q ? q_norm_w : k_norm_w;
    const bool do_norm = (norm_w != nullptr);

    // Pointer to this head's data
    __nv_bfloat16* head_ptr = tensor + (int64_t)batch_idx * total_heads * head_dim
                                     + (int64_t)local_head * head_dim;

    __shared__ float s_data[128];
    if (tid < head_dim) {
        s_data[tid] = __bfloat162float(head_ptr[tid]);
    }
    __syncthreads();

    // ---- RMSNorm (if applicable) ----
    if (do_norm) {
        // Sum of squares (block reduction over head_dim threads)
        float sumsq = (tid < head_dim) ? s_data[tid] * s_data[tid] : 0.0f;
        #pragma unroll
        for (int o = 64; o > 0; o >>= 1) {
            sumsq += __shfl_xor_sync(0xFFFFFFFF, sumsq, o);
        }
        // Note: head_dim=128 means 4 warps. Need block-level reduction.
        __shared__ float s_warp_sumsq[4];
        int warp = tid >> 5;
        int lane = tid & 31;
        if (lane == 0) s_warp_sumsq[warp] = sumsq;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < 4) ? s_warp_sumsq[lane] : 0.0f;
            v += __shfl_xor_sync(0xFFFFFFFF, v, 2);
            v += __shfl_xor_sync(0xFFFFFFFF, v, 1);
            if (lane == 0) s_warp_sumsq[0] = rsqrtf(v / (float)head_dim + eps);
        }
        __syncthreads();
        float rms = s_warp_sumsq[0];

        if (tid < head_dim) {
            s_data[tid] = s_data[tid] * rms * __bfloat162float(norm_w[tid]);
        }
        __syncthreads();
    }

    // ---- RoPE rotation ----
    int pos = positions[batch_idx];
    int rot_dim = head_dim / 2;

    // cos/sin tables: [pos, rot_dim]
    const __nv_bfloat16* cos_row = cos_table + (int64_t)pos * rot_dim;
    const __nv_bfloat16* sin_row = sin_table + (int64_t)pos * rot_dim;

    // GPT-NeoX (half-split): pairs are (i, i+rot_dim)
    // GPT-J (interleaved): pairs are (2i, 2i+1)
    if (tid < rot_dim) {
        float cos_v = __bfloat162float(cos_row[tid]);
        float sin_v = __bfloat162float(sin_row[tid]);

        if (is_neox) {
            float x0 = s_data[tid];
            float x1 = s_data[tid + rot_dim];
            head_ptr[tid] = __float2bfloat16(x0 * cos_v - x1 * sin_v);
            head_ptr[tid + rot_dim] = __float2bfloat16(x0 * sin_v + x1 * cos_v);
        } else {
            float x0 = s_data[2 * tid];
            float x1 = s_data[2 * tid + 1];
            head_ptr[2 * tid] = __float2bfloat16(x0 * cos_v - x1 * sin_v);
            head_ptr[2 * tid + 1] = __float2bfloat16(x0 * sin_v + x1 * cos_v);
        }
    }
}

extern "C" void arc_launch_qknorm_rope_bf16(
    void* q, void* k,
    const void* q_norm_w, const void* k_norm_w, // both NULL if no qknorm
    const void* cos_table, const void* sin_table,
    const int32_t* positions,
    int num_heads, int num_kv_heads, int head_dim,
    int batch_size, int is_neox, float eps,
    cudaStream_t stream
) {
    // Grid: [num_heads + num_kv_heads, batch_size]
    // Each block handles one (head, batch) pair
    dim3 grid(num_heads + num_kv_heads, batch_size);
    qknorm_rope_bf16_kernel<<<grid, 128, 0, stream>>>(
        (__nv_bfloat16*)q, (__nv_bfloat16*)k,
        (const __nv_bfloat16*)q_norm_w, (const __nv_bfloat16*)k_norm_w,
        (const __nv_bfloat16*)cos_table, (const __nv_bfloat16*)sin_table,
        positions,
        num_heads, num_kv_heads, head_dim,
        batch_size, is_neox, eps
    );
}
