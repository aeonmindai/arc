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
// 4× more blocks than the 1-warp-per-row kernel → higher SM occupancy when
// the model has small M (few rows) and long K (long inner loop).
//
// Block: 256 threads = 8 warps. 2 rows × 4 warps/row.
// Each warp processes K/4 of the inner loop, then 4 warp-reductions combine
// via 32B of shared memory per block.
//
// __launch_bounds__(256, 6) targets ~75% occupancy on modern NVIDIA SMs
// (1536 active threads out of a typical 2048 max), without forcing nvcc to
// over-spill registers. This is a generic heuristic, not GPU-specific.
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

// Dual-weight GEMV: computes gate_out[m] = sum(W_gate[m,k] * input[k]) and
// up_out[m] = sum(W_up[m,k] * input[k]) in a single kernel launch. Each block
// handles a chunk of rows from both matrices. The first M_gate output rows go
// to gate_out, the next M_up rows go to up_out. Saves 1 launch per layer.
//
// We can't physically fuse the weights (would duplicate them in memory),
// so we dispatch via blockIdx: the first ceil(M_a/8) blocks read from
// weight_a, the rest from weight_b.
__global__ __launch_bounds__(256, 6)
void gemv_bf16_dual_kernel(
    const __nv_bfloat16* __restrict__ weight_a,
    const __nv_bfloat16* __restrict__ weight_b,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ out_a,
    __nv_bfloat16* __restrict__ out_b,
    int M_a, int M_b, int K
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // First (M_a + 7) / 8 blocks handle weight_a, the rest handle weight_b.
    const int blocks_a = (M_a + GEMV_ROWS - 1) / GEMV_ROWS;
    bool is_a = blockIdx.x < blocks_a;
    int local_block = is_a ? blockIdx.x : (blockIdx.x - blocks_a);
    int row = local_block * GEMV_ROWS + warp;
    int M_local = is_a ? M_a : M_b;
    if (row >= M_local) return;

    const __nv_bfloat16* w_base = is_a ? weight_a : weight_b;
    __nv_bfloat16* out = is_a ? out_a : out_b;

    const __nv_bfloat16* w = w_base + (int64_t)row * K;
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

    if (lane == 0) out[row] = __float2bfloat16(acc);
}

extern "C" void arc_launch_gemv_bf16_dual(
    const void* weight_a, const void* weight_b,
    const void* input,
    void* out_a, void* out_b,
    int M_a, int M_b, int K, cudaStream_t stream
) {
    int blocks_a = (M_a + GEMV_ROWS - 1) / GEMV_ROWS;
    int blocks_b = (M_b + GEMV_ROWS - 1) / GEMV_ROWS;
    dim3 grid(blocks_a + blocks_b);
    gemv_bf16_dual_kernel<<<grid, GEMV_BLOCK, 0, stream>>>(
        (const __nv_bfloat16*)weight_a, (const __nv_bfloat16*)weight_b,
        (const __nv_bfloat16*)input,
        (__nv_bfloat16*)out_a, (__nv_bfloat16*)out_b,
        M_a, M_b, K
    );
}

// Fused silu(gate) * up * weight GEMV for the down projection. Eliminates the
// separate silu_mul kernel and the mlp_act buffer round-trip.
//
// Inner loop: for each k, computes act = silu(gate[k]) * up[k] = gate[k]/(1+exp(-gate[k])) * up[k]
// then accumulates W[row, k] * act. gate/up are read once per row, but L2 caches them
// across the M rows in this block (gate and up are typically small enough to live in L2).
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

// =============================================================================
// clock64()-instrumented GEMV variants for in-kernel cycle profiling.
// We can't run nsight-compute on Modal (gVisor blocks profiling ioctls), so
// we instrument the kernels ourselves: each warp records cycle counts at
// strategic points (start, after first load, end of inner loop, end of
// reduction) to a per-warp output buffer that the host reads back.
//
// This lets us answer:
//   - Cycles per __ldg → memory latency (bandwidth-bound vs latency-bound)
//   - Cycles per inner-loop iteration → compute vs memory ratio
//   - Cycles in cross-warp reduction → reduction overhead
//   - Cycles per row total → kernel efficiency
// =============================================================================

// One uint64 stamp per (block, warp, phase). 4 phases.
//   phase 0: kernel entry
//   phase 1: after first __ldg pair (memory latency probe)
//   phase 2: end of inner loop
//   phase 3: end of warp reduction (just before output write)
__global__ __launch_bounds__(256, 6)
void gemv_bf16_clocked_kernel(
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned long long* __restrict__ clocks, // [num_blocks * 8 warps * 4 phases]
    int M, int K
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * GEMV_ROWS + warp;
    if (row >= M) return;

    const int n_warps_per_block = GEMV_BLOCK / 32;
    const int stamp_base = (blockIdx.x * n_warps_per_block + warp) * 4;

    // Phase 0: kernel entry
    if (lane == 0) clocks[stamp_base + 0] = clock64();

    const __nv_bfloat16* w = weight + (int64_t)row * K;
    float acc = 0.0f;

    const int K8 = K >> 3;
    const uint4* wv = (const uint4*)w;
    const uint4* iv = (const uint4*)input;

    // First iteration outside the loop so we can probe single-load latency.
    if (lane < K8) {
        uint4 a = __ldg(wv + lane);
        uint4 b = __ldg(iv + lane);
        if (lane == 0) clocks[stamp_base + 1] = clock64();
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
    } else if (lane == 0) {
        clocks[stamp_base + 1] = clock64();
    }

    for (int i = lane + GEMV_WARP; i < K8; i += GEMV_WARP) {
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

    // Phase 2: end of inner loop
    if (lane == 0) clocks[stamp_base + 2] = clock64();

    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    // Phase 3: end of reduction
    if (lane == 0) {
        clocks[stamp_base + 3] = clock64();
        output[row] = __float2bfloat16(acc);
    }
}

extern "C" void arc_launch_gemv_bf16_clocked(
    const void* weight, const void* input, void* output,
    void* clocks_buffer,
    int M, int K, cudaStream_t stream
) {
    dim3 grid((M + GEMV_ROWS - 1) / GEMV_ROWS);
    gemv_bf16_clocked_kernel<<<grid, GEMV_BLOCK, 0, stream>>>(
        (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)input,
        (__nv_bfloat16*)output,
        (unsigned long long*)clocks_buffer,
        M, K
    );
}

extern "C" void arc_launch_gemv_bf16(
    const void* weight, const void* input, void* output,
    int M, int K, int sm_count, cudaStream_t stream
) {
    // Dispatch heuristic, derived from runtime SM count (no GPU-specific magic
    // numbers):
    //   - Original kernel: 1 warp per row, 8 rows per block → blocks = M/8.
    //     With ~6 active blocks/SM (the __launch_bounds__ hint, generic 75% target
    //     occupancy on modern NVIDIA SMs), available "slots" = sm_count * 6.
    //     Wave count for the original kernel ≈ (M/8) / (sm_count * 6).
    //   - Wide kernel: 4 warps per row, 2 rows per block → blocks = M/2 (4× more).
    //
    // Rule: when the original kernel doesn't fill many waves (the per-row inner
    // loop is long but there aren't enough rows to keep all SMs busy), prefer the
    // wide kernel. Empirically the cross-over happens around M ≈ sm_count * 100;
    // above that, the wide kernel's extra cross-warp reduction overhead exceeds
    // its wave-fill benefit.
    int sm = sm_count > 0 ? sm_count : 1;
    if (M < sm * 100) {
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
