/**
 * @brief Optimized FP8 GEMM kernels for blockwise quantized weights.
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace fp8_gemm {

// ============================================================================
// Helper functions
// ============================================================================

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 val) {
  return __half2float(__nv_cvt_fp8_to_halfraw(val.__x, __NV_E4M3));
}

__device__ __forceinline__ float get_scale(const float *__restrict__ scale,
                                           int n, int k, int scale_stride,
                                           int block_size_y, int block_size_x) {
  int sr = n / block_size_y;
  int sc = k / block_size_x;
  return __ldg(&scale[sr * scale_stride + sc]);
}

// ============================================================================
// FP8 Matmul Kernel
// ============================================================================

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void fp8_matmul_tiled(const T *__restrict__ input,
                                 const __nv_fp8_e4m3 *__restrict__ weight,
                                 const float *__restrict__ weight_scale,
                                 T *__restrict__ output, int M, int N, int K,
                                 int scale_row_stride, int block_size_y,
                                 int block_size_x) {
  // Padding is +1, not +4. The inner product reads `s_weight[tx][k]` with `tx`
  // varying fastest within a warp, so consecutive lanes are `BLOCK_K + pad`
  // floats apart. At BLOCK_K=32 a +4 pad gives stride 36; 36 mod 32 = 4, so
  // gcd(4,32)=4 and the warp lands on only 8 of the 32 shared-memory banks --
  // a 4-way conflict on every FMA of the k loop. A +1 pad gives stride 33,
  // which is coprime with 32, so all 32 banks are hit and the access is
  // conflict-free.
  //
  // Bit-identical output: the k loop is bounded by `k < BLOCK_K`, so columns
  // BLOCK_K..BLOCK_K+pad-1 are written by nobody and read by nobody. The pad
  // exists only to change the row stride.
  __shared__ float s_input[BLOCK_M][BLOCK_K + 1];
  __shared__ float s_weight[BLOCK_N][BLOCK_K + 1];

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row = by * BLOCK_M + ty;
  const int col = bx * BLOCK_N + tx;

  float acc = 0.0f;

  const int num_threads = BLOCK_M * BLOCK_N;
  const int tid = ty * BLOCK_N + tx;

  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    for (int i = tid; i < BLOCK_M * BLOCK_K; i += num_threads) {
      int lm = i / BLOCK_K;
      int lk = i % BLOCK_K;
      int gm = by * BLOCK_M + lm;
      int gk = k_tile + lk;

      float val = 0.0f;
      if (gm < M && gk < K) {
        if constexpr (std::is_same_v<T, half>) {
          val = __half2float(__ldg(&input[gm * K + gk]));
        } else {
          val = __bfloat162float(__ldg(&input[gm * K + gk]));
        }
      }
      s_input[lm][lk] = val;
    }

    for (int i = tid; i < BLOCK_N * BLOCK_K; i += num_threads) {
      int ln = i / BLOCK_K;
      int lk = i % BLOCK_K;
      int gn = bx * BLOCK_N + ln;
      int gk = k_tile + lk;

      float val = 0.0f;
      if (gn < N && gk < K) {
        __nv_fp8_e4m3 w;
        w.__x = __ldg(reinterpret_cast<const uint8_t *>(&weight[gn * K + gk]));
        float s = get_scale(weight_scale, gn, gk, scale_row_stride,
                            block_size_y, block_size_x);
        val = fp8_to_float(w) * s;
      }
      s_weight[ln][lk] = val;
    }

    __syncthreads();

    if (row < M && col < N) {
#pragma unroll
      for (int k = 0; k < BLOCK_K; k++) {
        acc += s_input[ty][k] * s_weight[tx][k];
      }
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    if constexpr (std::is_same_v<T, half>) {
      output[row * N + col] = __float2half(acc);
    } else {
      output[row * N + col] = __float2bfloat16(acc);
    }
  }
}

// ============================================================================
// FP8 GEMV kernel (dedicated decode path, M <= 4)
//
// The tiled GEMM above is built for prefill: at M == 1 it launches a full
// 32x32 tile grid where 31/32 of every block's threads do no useful work.
// This kernel is the decode sibling: one warp per output row, FP8 blocks
// dequantized in registers (per-block scales via __ldg), f32 accumulate,
// warp-shuffle reduction, half/bf16 out. Structure adapted from the BF16
// GEMV suite in arc-cuda-graph/src/cuda/gemv_bf16.cu (warp-per-row +
// __launch_bounds__ variants) with the FP8 block-scale inner loop from
// fp8_moe_gemm below.
//
// Preconditions (enforced by the Rust dispatcher in blockwise_fp8/ops.rs):
//   * K % 4 == 0            (32-bit vectorized FP8 loads, row alignment)
//   * block_size_x % 4 == 0 (a 4-wide group never straddles a scale block)
// ============================================================================

template <typename T, int ROWS_PER_BLOCK, int MAX_BLOCKS_PER_SM>
__global__ __launch_bounds__(ROWS_PER_BLOCK * 32, MAX_BLOCKS_PER_SM) void fp8_gemv_warp(
    const T *__restrict__ input,               // [M, K]
    const __nv_fp8_e4m3 *__restrict__ weight,  // [N, K] row-major
    const float *__restrict__ weight_scale,    // [ceil(N/bs_y), ceil(K/bs_x)]
    T *__restrict__ output,                    // [M, N]
    int M, int N, int K, int scale_row_stride, int block_size_y,
    int block_size_x) {
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
  const int m = blockIdx.y;
  if (n >= N || m >= M)
    return;

  const __nv_fp8_e4m3 *w_row = weight + (size_t)n * K;
  const T *in_row = input + (size_t)m * K;

  // Scale row is constant for this output row.
  const int scale_row_offset = (n / block_size_y) * scale_row_stride;

  float acc = 0.0f;

  // Main loop: each lane loads 4 FP8 weights (one 32-bit load) and 4 input
  // values per iteration; the warp covers 128 K-elements per iteration.
  const int K_aligned = (K / 128) * 128;
  for (int k_base = 0; k_base < K_aligned; k_base += 128) {
    const int k = k_base + lane * 4;

    const uint32_t w4 = __ldg(reinterpret_cast<const uint32_t *>(&w_row[k]));

    float i0, i1, i2, i3;
    if constexpr (std::is_same_v<T, half>) {
      const half2 h01 = __ldg(reinterpret_cast<const half2 *>(&in_row[k]));
      const half2 h23 = __ldg(reinterpret_cast<const half2 *>(&in_row[k + 2]));
      i0 = __half2float(h01.x);
      i1 = __half2float(h01.y);
      i2 = __half2float(h23.x);
      i3 = __half2float(h23.y);
    } else {
      const __nv_bfloat162 b01 =
          __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k]));
      const __nv_bfloat162 b23 =
          __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k + 2]));
      i0 = __bfloat162float(b01.x);
      i1 = __bfloat162float(b01.y);
      i2 = __bfloat162float(b23.x);
      i3 = __bfloat162float(b23.y);
    }

    __nv_fp8_e4m3 w0, w1, w2, w3;
    w0.__x = (w4 >> 0) & 0xFF;
    w1.__x = (w4 >> 8) & 0xFF;
    w2.__x = (w4 >> 16) & 0xFF;
    w3.__x = (w4 >> 24) & 0xFF;

    const float scale =
        __ldg(&weight_scale[scale_row_offset + k / block_size_x]);
    acc += scale * (i0 * fp8_to_float(w0) + i1 * fp8_to_float(w1) +
                    i2 * fp8_to_float(w2) + i3 * fp8_to_float(w3));
  }

  // Scalar remainder (K not a multiple of 128; still K % 4 == 0).
  for (int k = K_aligned + lane; k < K; k += 32) {
    float in_val;
    if constexpr (std::is_same_v<T, half>) {
      in_val = __half2float(__ldg(&in_row[k]));
    } else {
      in_val = __bfloat162float(__ldg(&in_row[k]));
    }
    __nv_fp8_e4m3 w;
    w.__x = __ldg(reinterpret_cast<const uint8_t *>(&w_row[k]));
    const float scale =
        __ldg(&weight_scale[scale_row_offset + k / block_size_x]);
    acc += scale * in_val * fp8_to_float(w);
  }

// Warp reduction using shuffle.
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }

  if (lane == 0) {
    if constexpr (std::is_same_v<T, half>) {
      output[(size_t)m * N + n] = __float2half(acc);
    } else {
      output[(size_t)m * N + n] = __float2bfloat16(acc);
    }
  }
}

// ============================================================================
// FP8 MoE GEMM - Warp-parallel kernel with vectorized loads
// Each warp (32 threads) computes one output element collaboratively
// ============================================================================

template <typename T>
__global__ void fp8_moe_gemm(const T *__restrict__ input,
                             const __nv_fp8_e4m3 *__restrict__ weights,
                             const float *__restrict__ weight_scales,
                             const uint32_t *__restrict__ indices,
                             T *__restrict__ output, int num_tokens, int topk,
                             int num_experts, int N, int K,
                             int scale_row_stride, int block_size_y,
                             int block_size_x, bool input_has_topk_dim) {
  // Each warp computes one output element
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane_id = threadIdx.x % 32;

  // Decode warp_id to (token, expert_slot, n_idx)
  const int n_idx = warp_id % N;
  const int temp = warp_id / N;
  const int expert_slot = temp % topk;
  const int token_idx = temp / topk;

  if (token_idx >= num_tokens)
    return;

  const uint32_t expert_idx = __ldg(&indices[token_idx * topk + expert_slot]);
  if (expert_idx >= (uint32_t)num_experts)
    return;

  // Pointers
  const __nv_fp8_e4m3 *w_row =
      weights + (size_t)expert_idx * N * K + (size_t)n_idx * K;
  const int scale_n_dim = CEILDIV(N, block_size_y);
  const int scale_expert_stride = scale_n_dim * scale_row_stride;
  const float *expert_scale =
      weight_scales + (size_t)expert_idx * scale_expert_stride;

  const T *in_row;
  if (input_has_topk_dim) {
    in_row = input + (size_t)token_idx * topk * K + (size_t)expert_slot * K;
  } else {
    in_row = input + (size_t)token_idx * K;
  }

  // Precompute scale row index (constant for this output element)
  const int scale_row = n_idx / block_size_y;
  const int scale_row_offset = scale_row * scale_row_stride;

  float acc = 0.0f;

  // Process 4 elements per thread per iteration using vectorized loads
  // Each warp processes 32*4 = 128 elements per iteration
  const int K_aligned = (K / 128) * 128;

  for (int k_base = 0; k_base < K_aligned; k_base += 128) {
    int k = k_base + lane_id * 4;

    // Load 4 FP8 weights at once (32-bit load)
    uint32_t w4 = __ldg(reinterpret_cast<const uint32_t *>(&w_row[k]));

    // Load 4 input values
    float i0, i1, i2, i3;
    if constexpr (std::is_same_v<T, half>) {
      half2 h01 = __ldg(reinterpret_cast<const half2 *>(&in_row[k]));
      half2 h23 = __ldg(reinterpret_cast<const half2 *>(&in_row[k + 2]));
      i0 = __half2float(h01.x);
      i1 = __half2float(h01.y);
      i2 = __half2float(h23.x);
      i3 = __half2float(h23.y);
    } else {
      __nv_bfloat162 b01 =
          __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k]));
      __nv_bfloat162 b23 =
          __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k + 2]));
      i0 = __bfloat162float(b01.x);
      i1 = __bfloat162float(b01.y);
      i2 = __bfloat162float(b23.x);
      i3 = __bfloat162float(b23.y);
    }

    // Extract 4 FP8 values and convert
    __nv_fp8_e4m3 w0, w1, w2, w3;
    w0.__x = (w4 >> 0) & 0xFF;
    w1.__x = (w4 >> 8) & 0xFF;
    w2.__x = (w4 >> 16) & 0xFF;
    w3.__x = (w4 >> 24) & 0xFF;

    // Get scale
    int scale_col = k / block_size_x;
    float scale = __ldg(&expert_scale[scale_row_offset + scale_col]);

    // Accumulate
    acc += scale * (i0 * fp8_to_float(w0) + i1 * fp8_to_float(w1) +
                    i2 * fp8_to_float(w2) + i3 * fp8_to_float(w3));
  }

  // Handle remainder
  for (int k = K_aligned + lane_id; k < K; k += 32) {
    float in_val;
    if constexpr (std::is_same_v<T, half>) {
      in_val = __half2float(__ldg(&in_row[k]));
    } else {
      in_val = __bfloat162float(__ldg(&in_row[k]));
    }

    __nv_fp8_e4m3 w;
    w.__x = __ldg(reinterpret_cast<const uint8_t *>(&w_row[k]));

    int scale_col = k / block_size_x;
    float scale = __ldg(&expert_scale[scale_row_offset + scale_col]);

    acc += scale * in_val * fp8_to_float(w);
  }

// Warp reduction using shuffle
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }

  // Lane 0 writes the result
  if (lane_id == 0) {
    size_t out_idx =
        (size_t)token_idx * topk * N + (size_t)expert_slot * N + n_idx;
    if constexpr (std::is_same_v<T, half>) {
      output[out_idx] = __float2half(acc);
    } else {
      output[out_idx] = __float2bfloat16(acc);
    }
  }
}

} // namespace fp8_gemm

// ============================================================================
// C API
// ============================================================================

extern "C" void launch_fp8_matmul_f16(const __half *input,
                                      const __nv_fp8_e4m3 *weight,
                                      const float *weight_scale, __half *output,
                                      int M, int N, int K, int scale_row_stride,
                                      int block_size_y, int block_size_x,
                                      cudaStream_t stream) {
  constexpr int TILE = 32;
  constexpr int TILE_K = 32;

  dim3 block(TILE, TILE);
  dim3 grid(CEILDIV(N, TILE), CEILDIV(M, TILE));

  fp8_gemm::fp8_matmul_tiled<half, TILE, TILE, TILE_K>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale, output, M, N, K,
                                   scale_row_stride, block_size_y,
                                   block_size_x);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void
launch_fp8_matmul_bf16(const __nv_bfloat16 *input, const __nv_fp8_e4m3 *weight,
                       const float *weight_scale, __nv_bfloat16 *output, int M,
                       int N, int K, int scale_row_stride, int block_size_y,
                       int block_size_x, cudaStream_t stream) {
  constexpr int TILE = 32;
  constexpr int TILE_K = 32;

  dim3 block(TILE, TILE);
  dim3 grid(CEILDIV(N, TILE), CEILDIV(M, TILE));

  fp8_gemm::fp8_matmul_tiled<__nv_bfloat16, TILE, TILE, TILE_K>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale, output, M, N, K,
                                   scale_row_stride, block_size_y,
                                   block_size_x);
  CUDA_CHECK(cudaGetLastError());
}

// Warp-per-row GEMV launcher for the decode regime (M <= 4). Row-grouping
// mirrors the empirical dispatch of the BF16 GEMV suite (gemv_bf16.cu):
// small-N projection shapes win with 4 warps/block (more bandwidth per
// block when there aren't enough rows to fill every SM), large-N shapes win
// with 1 warp/block (maximum wave fill). grid.y indexes the input row.
template <typename T>
static void launch_fp8_gemv_impl(const T *input, const __nv_fp8_e4m3 *weight,
                                 const float *weight_scale, T *output, int M,
                                 int N, int K, int scale_row_stride,
                                 int block_size_y, int block_size_x,
                                 cudaStream_t stream) {
  if (N < 8192) {
    dim3 grid(CEILDIV(N, 4), M);
    fp8_gemm::fp8_gemv_warp<T, 4, 8><<<grid, 4 * 32, 0, stream>>>(
        input, weight, weight_scale, output, M, N, K, scale_row_stride,
        block_size_y, block_size_x);
  } else {
    dim3 grid(N, M);
    fp8_gemm::fp8_gemv_warp<T, 1, 16><<<grid, 32, 0, stream>>>(
        input, weight, weight_scale, output, M, N, K, scale_row_stride,
        block_size_y, block_size_x);
  }
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_fp8_gemv_f16(const __half *input,
                                    const __nv_fp8_e4m3 *weight,
                                    const float *weight_scale, __half *output,
                                    int M, int N, int K, int scale_row_stride,
                                    int block_size_y, int block_size_x,
                                    cudaStream_t stream) {
  launch_fp8_gemv_impl<half>(input, weight, weight_scale, output, M, N, K,
                             scale_row_stride, block_size_y, block_size_x,
                             stream);
}

extern "C" void launch_fp8_gemv_bf16(const __nv_bfloat16 *input,
                                     const __nv_fp8_e4m3 *weight,
                                     const float *weight_scale,
                                     __nv_bfloat16 *output, int M, int N, int K,
                                     int scale_row_stride, int block_size_y,
                                     int block_size_x, cudaStream_t stream) {
  launch_fp8_gemv_impl<__nv_bfloat16>(input, weight, weight_scale, output, M, N,
                                      K, scale_row_stride, block_size_y,
                                      block_size_x, stream);
}

extern "C" void launch_fp8_indexed_moe_gemm_f16(
    const __half *input, const __nv_fp8_e4m3 *weights,
    const float *weight_scales, const uint32_t *indices, __half *output,
    int num_tokens, int topk, int num_experts, int N, int K,
    int scale_row_stride, int block_size_y, int block_size_x,
    bool input_has_topk_dim, cudaStream_t stream) {
  // Each warp (32 threads) computes one output element
  // Use 512 threads per block (16 warps) for better occupancy
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;

  int total_outputs = num_tokens * topk * N;
  int total_warps = total_outputs;
  int num_blocks = CEILDIV(total_warps, WARPS_PER_BLOCK);

  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(num_blocks);

  fp8_gemm::fp8_moe_gemm<half><<<grid, block, 0, stream>>>(
      input, weights, weight_scales, indices, output, num_tokens, topk,
      num_experts, N, K, scale_row_stride, block_size_y, block_size_x,
      input_has_topk_dim);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_fp8_indexed_moe_gemm_bf16(
    const __nv_bfloat16 *input, const __nv_fp8_e4m3 *weights,
    const float *weight_scales, const uint32_t *indices, __nv_bfloat16 *output,
    int num_tokens, int topk, int num_experts, int N, int K,
    int scale_row_stride, int block_size_y, int block_size_x,
    bool input_has_topk_dim, cudaStream_t stream) {
  // Each warp (32 threads) computes one output element
  // Use 512 threads per block (16 warps) for better occupancy
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;

  int total_outputs = num_tokens * topk * N;
  int total_warps = total_outputs;
  int num_blocks = CEILDIV(total_warps, WARPS_PER_BLOCK);

  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(num_blocks);

  fp8_gemm::fp8_moe_gemm<__nv_bfloat16><<<grid, block, 0, stream>>>(
      input, weights, weight_scales, indices, output, num_tokens, topk,
      num_experts, N, K, scale_row_stride, block_size_y, block_size_x,
      input_has_topk_dim);
  CUDA_CHECK(cudaGetLastError());
}
