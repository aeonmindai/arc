/**
 * High-bandwidth BF16 GEMV for the dedicated decode path.
 *
 * output[m] = sum_k(weight[m][k] * input[k])
 *
 * Key optimization: 2 rows per warp. Each warp maintains 2 accumulators
 * and issues 2 independent weight loads per iteration. This doubles the
 * number of outstanding HBM requests per warp, saturating more memory
 * channels and pushing bandwidth utilization from ~47% to 70%+.
 *
 * No cuBLAS — fully graph-capture compatible.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define GEMV_WARP 32
#define GEMV_WARPS 8                        // warps per block
#define GEMV_ROWS_PER_WARP 2                // each warp handles 2 rows
#define GEMV_ROWS (GEMV_WARPS * GEMV_ROWS_PER_WARP) // 16 rows per block
#define GEMV_BLOCK (GEMV_WARP * GEMV_WARPS) // 256 threads

__global__ __launch_bounds__(256, 4)
void gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int M, int K
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int base_row = blockIdx.x * GEMV_ROWS + warp * GEMV_ROWS_PER_WARP;

    const int row0 = base_row;
    const int row1 = base_row + 1;
    const bool valid0 = row0 < M;
    const bool valid1 = row1 < M;
    if (!valid0) return;

    const __nv_bfloat16* w0 = weight + (int64_t)row0 * K;
    const __nv_bfloat16* w1 = valid1 ? (weight + (int64_t)row1 * K) : w0;

    float acc0 = 0.0f, acc1 = 0.0f;

    const int K8 = K >> 3;
    const uint4* iv = (const uint4*)input;
    const uint4* wv0 = (const uint4*)w0;
    const uint4* wv1 = (const uint4*)w1;

    for (int i = lane; i < K8; i += GEMV_WARP) {
        // 1 input load (cached after first access)
        uint4 b = __ldg(iv + i);
        // 2 weight loads from different rows — doubles HBM pressure
        uint4 a0 = __ldg(wv0 + i);
        uint4 a1 = __ldg(wv1 + i);

        const __nv_bfloat16* bp = (const __nv_bfloat16*)&b;
        const __nv_bfloat16* ap0 = (const __nv_bfloat16*)&a0;
        const __nv_bfloat16* ap1 = (const __nv_bfloat16*)&a1;

        acc0 += __bfloat162float(ap0[0]) * __bfloat162float(bp[0])
              + __bfloat162float(ap0[1]) * __bfloat162float(bp[1])
              + __bfloat162float(ap0[2]) * __bfloat162float(bp[2])
              + __bfloat162float(ap0[3]) * __bfloat162float(bp[3])
              + __bfloat162float(ap0[4]) * __bfloat162float(bp[4])
              + __bfloat162float(ap0[5]) * __bfloat162float(bp[5])
              + __bfloat162float(ap0[6]) * __bfloat162float(bp[6])
              + __bfloat162float(ap0[7]) * __bfloat162float(bp[7]);

        acc1 += __bfloat162float(ap1[0]) * __bfloat162float(bp[0])
              + __bfloat162float(ap1[1]) * __bfloat162float(bp[1])
              + __bfloat162float(ap1[2]) * __bfloat162float(bp[2])
              + __bfloat162float(ap1[3]) * __bfloat162float(bp[3])
              + __bfloat162float(ap1[4]) * __bfloat162float(bp[4])
              + __bfloat162float(ap1[5]) * __bfloat162float(bp[5])
              + __bfloat162float(ap1[6]) * __bfloat162float(bp[6])
              + __bfloat162float(ap1[7]) * __bfloat162float(bp[7]);
    }

    // Warp reduce both accumulators
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, 16);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, 16);
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, 8);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, 8);
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, 4);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, 4);
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, 2);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, 2);
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, 1);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, 1);

    if (lane == 0) {
        output[row0] = __float2bfloat16(acc0);
        if (valid1) output[row1] = __float2bfloat16(acc1);
    }
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
