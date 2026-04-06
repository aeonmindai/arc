/**
 * BF16 GEMV — optimized for maximum SM occupancy on Blackwell.
 *
 * 4 warps/block (128 threads), 4 rows/block, up to 16 blocks per SM.
 * High warp count per SM hides HBM latency through warp switching.
 * __ldg for read-only texture cache. No cuBLAS — graph-capture compatible.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define GEMV_WARP 32
#define GEMV_WARPS 4
#define GEMV_ROWS GEMV_WARPS  // 1 row per warp, 4 rows per block
#define GEMV_BLOCK (GEMV_WARP * GEMV_WARPS) // 128 threads

// Allow up to 16 blocks per SM for maximum occupancy
__global__ __launch_bounds__(128, 16)
void gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int M, int K
) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * GEMV_ROWS + (threadIdx.x >> 5);
    if (row >= M) return;

    const __nv_bfloat16* w = weight + (int64_t)row * K;
    float acc = 0.0f;

    const int K8 = K >> 3;
    const uint4* wv = (const uint4*)w;
    const uint4* iv = (const uint4*)input;

    for (int i = lane; i < K8; i += GEMV_WARP) {
        uint4 a = __ldg(wv + i);
        uint4 b = __ldg(iv + i);
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

    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    if (lane == 0) output[row] = __float2bfloat16(acc);
}

extern "C" void arc_launch_gemv_bf16(
    const void* weight, const void* input, void* output,
    int M, int K, cudaStream_t stream
) {
    dim3 grid((M + GEMV_ROWS - 1) / GEMV_ROWS);
    gemv_bf16_kernel<<<grid, GEMV_BLOCK, 0, stream>>>(
        (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)input,
        (__nv_bfloat16*)output, M, K
    );
}
