/**
 * High-performance BF16 GEMV kernel for the dedicated decode path.
 *
 * output[m] = sum_k(weight[m][k] * input[k])  for m in [0, M)
 * weight: [M, K] row-major BF16
 * input:  [K] BF16
 * output: [M] BF16
 *
 * Optimizations:
 * - Input vector cached in shared memory (read once from HBM, reused by all warps)
 * - 128-bit vectorized loads (8 BF16 per load = 16 bytes)
 * - Multiple rows per thread block (32 warps × 1 row/warp = 32 rows/block)
 * - __ldg for read-only weight access through texture cache
 * - Unrolled inner loop for instruction-level parallelism
 *
 * No cuBLAS dependency — fully graph-capture compatible.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// 32 warps per block = 1024 threads, 32 rows per block
// Maximizes SM occupancy and amortizes input vector shared memory load
#define GEMV_WARP_SIZE 32
#define GEMV_ROWS_PER_BLOCK 32
#define GEMV_BLOCK_SIZE (GEMV_WARP_SIZE * GEMV_ROWS_PER_BLOCK) // 1024

// Max K dimension for shared memory input cache
// Must fit in default 48KB smem: 24576 * 2 = 48KB
// For K > MAX_CACHED_K, use direct HBM reads (still fast via __ldg)
#define MAX_CACHED_K 24576

__global__ void gemv_bf16_cached(
    const __nv_bfloat16* __restrict__ weight, // [M, K]
    const __nv_bfloat16* __restrict__ input,  // [K]
    __nv_bfloat16* __restrict__ output,       // [M]
    int M, int K
) {
    extern __shared__ __nv_bfloat16 s_input[]; // [K] cached input vector

    const int tid = threadIdx.x;
    const int warp_id = tid / GEMV_WARP_SIZE;
    const int lane = tid % GEMV_WARP_SIZE;
    const int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp_id;

    // Cooperatively load input vector into shared memory
    // 1024 threads load K elements — each thread loads K/1024 elements
    for (int i = tid; i < K; i += GEMV_BLOCK_SIZE) {
        s_input[i] = input[i];
    }
    __syncthreads();

    if (row >= M) return;

    const __nv_bfloat16* w = weight + (int64_t)row * K;

    float sum = 0.0f;

    // Main loop: 128-bit vectorized loads from weight, scalar from shared input
    // Weight access is coalesced across lanes within the same warp
    const int K8 = K / 8;
    const uint4* w_vec = (const uint4*)w;

    for (int i = lane; i < K8; i += GEMV_WARP_SIZE) {
        // Load 8 weight elements via __ldg (read-only texture cache)
        uint4 wv = __ldg(w_vec + i);
        const __nv_bfloat16* wp = (const __nv_bfloat16*)&wv;

        // Load 8 input elements from shared memory (no bank conflicts — sequential)
        int base = i * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            sum += __bfloat162float(wp[j]) * __bfloat162float(s_input[base + j]);
        }
    }

    // Remainder
    for (int i = K8 * 8 + lane; i < K; i += GEMV_WARP_SIZE) {
        sum += __bfloat162float(__ldg(w + i)) * __bfloat162float(s_input[i]);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = GEMV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        output[row] = __float2bfloat16(sum);
    }
}

// Fallback for very large K (> MAX_CACHED_K) — no shared memory caching
__global__ void gemv_bf16_direct(
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int M, int K
) {
    const int warp_id = threadIdx.x / GEMV_WARP_SIZE;
    const int lane = threadIdx.x % GEMV_WARP_SIZE;
    const int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp_id;

    if (row >= M) return;

    const __nv_bfloat16* w = weight + (int64_t)row * K;
    float sum = 0.0f;

    const int K8 = K / 8;
    const uint4* w_vec = (const uint4*)w;
    const uint4* in_vec = (const uint4*)input;

    for (int i = lane; i < K8; i += GEMV_WARP_SIZE) {
        uint4 wv = __ldg(w_vec + i);
        uint4 iv = __ldg(in_vec + i);
        const __nv_bfloat16* wp = (const __nv_bfloat16*)&wv;
        const __nv_bfloat16* ip = (const __nv_bfloat16*)&iv;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            sum += __bfloat162float(wp[j]) * __bfloat162float(ip[j]);
        }
    }

    for (int i = K8 * 8 + lane; i < K; i += GEMV_WARP_SIZE) {
        sum += __bfloat162float(__ldg(w + i)) * __bfloat162float(input[i]);
    }

    #pragma unroll
    for (int offset = GEMV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        output[row] = __float2bfloat16(sum);
    }
}

extern "C" void arc_launch_gemv_bf16(
    const void* weight,
    const void* input,
    void* output,
    int M, int K,
    cudaStream_t stream
) {
    dim3 grid((M + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
    dim3 block(GEMV_BLOCK_SIZE);

    if (K <= MAX_CACHED_K) {
        int smem = K * sizeof(__nv_bfloat16); // K * 2 bytes
        gemv_bf16_cached<<<grid, block, smem, stream>>>(
            (const __nv_bfloat16*)weight,
            (const __nv_bfloat16*)input,
            (__nv_bfloat16*)output,
            M, K
        );
    } else {
        gemv_bf16_direct<<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)weight,
            (const __nv_bfloat16*)input,
            (__nv_bfloat16*)output,
            M, K
        );
    }
}
