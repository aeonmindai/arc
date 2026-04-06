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
#define GEMV_WARPS 8
#define GEMV_ROWS_PER_WARP 4                // each warp handles 4 rows
#define GEMV_ROWS (GEMV_WARPS * GEMV_ROWS_PER_WARP) // 32 rows per block
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
    if (base_row >= M) return;

    // 4 row pointers — handle M not divisible by 4
    const int r0 = base_row, r1 = base_row+1, r2 = base_row+2, r3 = base_row+3;
    const __nv_bfloat16* w0 = weight + (int64_t)r0 * K;
    const __nv_bfloat16* w1 = (r1<M) ? weight + (int64_t)r1 * K : w0;
    const __nv_bfloat16* w2 = (r2<M) ? weight + (int64_t)r2 * K : w0;
    const __nv_bfloat16* w3 = (r3<M) ? weight + (int64_t)r3 * K : w0;

    float a0=0, a1=0, a2=0, a3=0;
    const int K8 = K >> 3;
    const uint4* iv = (const uint4*)input;

    for (int i = lane; i < K8; i += GEMV_WARP) {
        uint4 b = __ldg(iv + i);
        uint4 v0 = __ldg((const uint4*)w0 + i);
        uint4 v1 = __ldg((const uint4*)w1 + i);
        uint4 v2 = __ldg((const uint4*)w2 + i);
        uint4 v3 = __ldg((const uint4*)w3 + i);
        const __nv_bfloat16* bp = (const __nv_bfloat16*)&b;

        #define DOT8(acc, v) { \
            const __nv_bfloat16* p = (const __nv_bfloat16*)&v; \
            acc += __bfloat162float(p[0])*__bfloat162float(bp[0]) \
                 + __bfloat162float(p[1])*__bfloat162float(bp[1]) \
                 + __bfloat162float(p[2])*__bfloat162float(bp[2]) \
                 + __bfloat162float(p[3])*__bfloat162float(bp[3]) \
                 + __bfloat162float(p[4])*__bfloat162float(bp[4]) \
                 + __bfloat162float(p[5])*__bfloat162float(bp[5]) \
                 + __bfloat162float(p[6])*__bfloat162float(bp[6]) \
                 + __bfloat162float(p[7])*__bfloat162float(bp[7]); \
        }
        DOT8(a0, v0); DOT8(a1, v1); DOT8(a2, v2); DOT8(a3, v3);
        #undef DOT8
    }

    // Reduce all 4 accumulators
    #define WARP_REDUCE(x) \
        x += __shfl_down_sync(0xFFFFFFFF, x, 16); \
        x += __shfl_down_sync(0xFFFFFFFF, x, 8);  \
        x += __shfl_down_sync(0xFFFFFFFF, x, 4);  \
        x += __shfl_down_sync(0xFFFFFFFF, x, 2);  \
        x += __shfl_down_sync(0xFFFFFFFF, x, 1);
    WARP_REDUCE(a0); WARP_REDUCE(a1); WARP_REDUCE(a2); WARP_REDUCE(a3);
    #undef WARP_REDUCE

    if (lane == 0) {
        output[r0] = __float2bfloat16(a0);
        if (r1<M) output[r1] = __float2bfloat16(a1);
        if (r2<M) output[r2] = __float2bfloat16(a2);
        if (r3<M) output[r3] = __float2bfloat16(a3);
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
