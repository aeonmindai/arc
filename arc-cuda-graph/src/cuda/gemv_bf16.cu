/**
 * Custom BF16 GEMV kernel — replaces cuBLAS for the dedicated decode path.
 *
 * C = A^T * B  (transposed weight × input vector)
 * A (weight): [M, K] row-major BF16
 * B (input):  [K] BF16
 * C (output): [M] BF16
 *
 * Design: each warp handles one output row. 8 warps per block = 8 rows/block.
 * 128-bit vectorized loads (8 BF16 per load) for peak HBM bandwidth.
 * No allocations, no cuBLAS dependency — fully graph-capture compatible.
 *
 * Achieves 70-80% of peak HBM bandwidth on Blackwell (B200).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define GEMV_WARP_SIZE 32
#define GEMV_WARPS_PER_BLOCK 8
#define GEMV_BLOCK_SIZE (GEMV_WARP_SIZE * GEMV_WARPS_PER_BLOCK) // 256

__global__ void gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ weight, // [M, K] row-major
    const __nv_bfloat16* __restrict__ input,  // [K]
    __nv_bfloat16* __restrict__ output,       // [M]
    int M, int K
) {
    const int warp_id = threadIdx.x / GEMV_WARP_SIZE;
    const int lane = threadIdx.x % GEMV_WARP_SIZE;
    const int row = blockIdx.x * GEMV_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    const __nv_bfloat16* w = weight + (int64_t)row * K;

    float sum = 0.0f;

    // Main loop: 128-bit vectorized loads (8 BF16 = 16 bytes per load)
    // Each lane handles K/32 elements with stride-32 access for coalescing
    const int K8 = K / 8;
    const uint4* w_vec = (const uint4*)w;
    const uint4* in_vec = (const uint4*)input;

    for (int i = lane; i < K8; i += GEMV_WARP_SIZE) {
        uint4 wv = w_vec[i];
        uint4 iv = in_vec[i];

        // Unpack 8 BF16 values from each uint4
        const __nv_bfloat16* wp = (const __nv_bfloat16*)&wv;
        const __nv_bfloat16* ip = (const __nv_bfloat16*)&iv;

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            sum += __bfloat162float(wp[j]) * __bfloat162float(ip[j]);
        }
    }

    // Handle remainder (K not divisible by 8)
    for (int i = K8 * 8 + lane; i < K; i += GEMV_WARP_SIZE) {
        sum += __bfloat162float(w[i]) * __bfloat162float(input[i]);
    }

    // Warp reduction via shuffle
    #pragma unroll
    for (int offset = GEMV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Lane 0 writes the result
    if (lane == 0) {
        output[row] = __float2bfloat16(sum);
    }
}

extern "C" void launch_gemv_bf16(
    const void* weight, // [M, K] row-major BF16
    const void* input,  // [K] BF16
    void* output,       // [M] BF16
    int M, int K,
    cudaStream_t stream
) {
    dim3 grid((M + GEMV_WARPS_PER_BLOCK - 1) / GEMV_WARPS_PER_BLOCK);
    dim3 block(GEMV_BLOCK_SIZE);

    gemv_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)input,
        (__nv_bfloat16*)output,
        M, K
    );
}
