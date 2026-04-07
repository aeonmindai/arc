/**
 * BF16 GEMV kernel for the dedicated decode path.
 *
 * output[m] = sum_k(weight[m][k] * input[k])
 * weight: [M, K] row-major BF16
 * input:  [K] BF16
 * output: [M] BF16
 *
 * 8 warps per block, 1 row per warp. __ldg for read-only cache.
 * No cuBLAS — fully graph-capture compatible.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define GEMV_WARP 32
#define GEMV_ROWS 8
#define GEMV_BLOCK (GEMV_WARP * GEMV_ROWS)

// Wide-row variant: 4 warps cooperate on each row, 2 rows per block.
// 4× more blocks than the 1-warp-per-row kernel → much higher SM occupancy
// for small-M, long-K shapes (down_proj, oproj where M ≤ K).
//
// Block: 256 threads = 8 warps. 2 rows × 4 warps/row.
// Each warp processes K/4 of the inner loop, then 4 warp-reductions
// combine via 32B of shared memory per block.
//
// __launch_bounds__(256, 6): allow 6 blocks/SM (vs 4 default). Increases
// in-flight HBM traffic and improves latency hiding for long-K shapes.
// Register budget: 65536 / (256 * 6) = 42 regs/thread (was 64 with 4 blocks).
__global__ __launch_bounds__(256, 6)
void gemv_bf16_w4_kernel(
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;            // 0..7
    const int row_in_block = warp >> 2;   // 0..1
    const int warp_in_row = warp & 3;     // 0..3
    const int row = blockIdx.x * 2 + row_in_block;
    if (row >= M) return;

    const __nv_bfloat16* w = weight + (int64_t)row * K;
    float acc = 0.0f;

    const int K8 = K >> 3;
    const uint4* wv = (const uint4*)w;
    const uint4* iv = (const uint4*)input;

    // Strided across the 4 warps in this row, coalesced within each warp.
    for (int i = warp_in_row * 32 + lane; i < K8; i += 128) {
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

    // Intra-warp reduction
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    // Cross-warp reduction (4 warps per row → shared mem)
    __shared__ float row_acc[2][4];
    if (lane == 0) row_acc[row_in_block][warp_in_row] = acc;
    __syncthreads();

    if (warp_in_row == 0 && lane == 0) {
        float total = row_acc[row_in_block][0] + row_acc[row_in_block][1]
                    + row_acc[row_in_block][2] + row_acc[row_in_block][3];
        output[row] = __float2bfloat16(total);
    }
}

// Fused silu(gate) * up * weight GEMV for the down projection. Eliminates the
// separate silu_mul kernel and the mlp_act buffer round-trip.
//
// Inner loop: for each k, computes act = silu(gate[k]) * up[k] = gate[k]/(1+exp(-gate[k])) * up[k]
// then accumulates W[row, k] * act. gate/up are read once per row, but L2 caches them
// across the M rows in this block (gate+up = 51KB total for Qwen3-32B, fits in L2).
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ __launch_bounds__(256, 6)
void gemv_bf16_w4_silu_mul_kernel(
    const __nv_bfloat16* __restrict__ weight, // [M, K] = down_proj
    const __nv_bfloat16* __restrict__ gate,   // [K]
    const __nv_bfloat16* __restrict__ up,     // [K]
    __nv_bfloat16* __restrict__ output,       // [M]
    int M, int K
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;            // 0..7
    const int row_in_block = warp >> 2;   // 0..1
    const int warp_in_row = warp & 3;     // 0..3
    const int row = blockIdx.x * 2 + row_in_block;
    if (row >= M) return;

    const __nv_bfloat16* w = weight + (int64_t)row * K;
    float acc = 0.0f;

    const int K8 = K >> 3;
    const uint4* wv = (const uint4*)w;
    const uint4* gv = (const uint4*)gate;
    const uint4* uv = (const uint4*)up;

    for (int i = warp_in_row * 32 + lane; i < K8; i += 128) {
        uint4 a = __ldg(wv + i);
        uint4 g = __ldg(gv + i);
        uint4 u = __ldg(uv + i);
        const __nv_bfloat16* ap = (const __nv_bfloat16*)&a;
        const __nv_bfloat16* gp = (const __nv_bfloat16*)&g;
        const __nv_bfloat16* up_p = (const __nv_bfloat16*)&u;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float act = silu_f(__bfloat162float(gp[j])) * __bfloat162float(up_p[j]);
            acc += __bfloat162float(ap[j]) * act;
        }
    }

    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    __shared__ float row_acc[2][4];
    if (lane == 0) row_acc[row_in_block][warp_in_row] = acc;
    __syncthreads();

    if (warp_in_row == 0 && lane == 0) {
        float total = row_acc[row_in_block][0] + row_acc[row_in_block][1]
                    + row_acc[row_in_block][2] + row_acc[row_in_block][3];
        output[row] = __float2bfloat16(total);
    }
}

extern "C" void arc_launch_gemv_bf16_silu_mul_down(
    const void* weight, const void* gate, const void* up, void* output,
    int M, int K, cudaStream_t stream
) {
    dim3 grid((M + 1) / 2);
    gemv_bf16_w4_silu_mul_kernel<<<grid, GEMV_BLOCK, 0, stream>>>(
        (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)gate,
        (const __nv_bfloat16*)up,
        (__nv_bfloat16*)output, M, K
    );
}

__global__ __launch_bounds__(256, 6)
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

// Same kernel but outputs F32 instead of BF16 (for LM head → sampling needs F32)
__global__ __launch_bounds__(256, 4)
void gemv_bf16_f32out_kernel(
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ output,
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

    if (lane == 0) output[row] = acc; // F32 directly, no cast
}

extern "C" void arc_launch_gemv_bf16_f32out(
    const void* weight, const void* input, void* output,
    int M, int K, cudaStream_t stream
) {
    dim3 grid((M + GEMV_ROWS - 1) / GEMV_ROWS);
    gemv_bf16_f32out_kernel<<<grid, GEMV_BLOCK, 0, stream>>>(
        (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)input,
        (float*)output, M, K
    );
}

extern "C" void arc_launch_gemv_bf16(
    const void* weight, const void* input, void* output,
    int M, int K, cudaStream_t stream
) {
    // Dispatch heuristic: when M is small relative to K (i.e. each row has
    // long inner work and there are few rows), the original 1-warp-per-row
    // kernel SM-starves. The wide-row variant uses 4 warps per row, giving
    // 4× more blocks and much higher SM occupancy.
    //
    // For Qwen3-32B: down (M=5120, K=25600), oproj (M=K=5120), gate/up
    // (M=25600, K=5120), qkv (M=10240, K=5120). The wide kernel helps when
    // M/K ≤ ~1, hurts (more redundant launch overhead) when M >> K.
    // Wide kernel wins for small M (down, oproj, qkv): more blocks, better SM occupancy.
    // Original kernel wins for large M (gate, up, lm_head): more rows per block reduces
    // cross-warp reduction overhead. Empirically (Qwen3-32B on B200): wide=45μs vs orig=46μs
    // for down (M=5120), but wide=49μs vs orig=45μs for gate/up (M=25600).
    if (M < 12000) {
        // 4× wide-row variant
        dim3 grid((M + 1) / 2);
        gemv_bf16_w4_kernel<<<grid, GEMV_BLOCK, 0, stream>>>(
            (const __nv_bfloat16*)weight,
            (const __nv_bfloat16*)input,
            (__nv_bfloat16*)output, M, K
        );
    } else {
        dim3 grid((M + GEMV_ROWS - 1) / GEMV_ROWS);
        gemv_bf16_kernel<<<grid, GEMV_BLOCK, 0, stream>>>(
            (const __nv_bfloat16*)weight,
            (const __nv_bfloat16*)input,
            (__nv_bfloat16*)output, M, K
        );
    }
}
