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
// FP8 WIDE GEMV — the memory-level-parallelism rewrite of `fp8_gemv_warp`.
//
// Parent system: ArcKernels.
//
// WHY THIS EXISTS
// ---------------
// `fp8_gemv_warp` above owns every b=1 decode call (M <= 4). It is a pure
// streaming problem — each weight byte is read once, multiplied into a vector,
// and discarded — so it should run at HBM speed. It does not. Two structural
// reasons, both visible in its source and both fixed here:
//
// 1. THE INNER LOOP CONTAINS A RUNTIME INTEGER DIVISION.
//    Line ~199: `weight_scale[scale_row_offset + k / block_size_x]`.
//    `block_size_x` is a kernel *argument*, so this is `int / int` by a
//    non-constant divisor. NVIDIA GPUs have no integer-divide instruction;
//    nvcc expands it inline to a reciprocal-and-fixup sequence of roughly
//    15-20 instructions. It executes once per FOUR weight bytes, in the
//    innermost loop, and every one of the 32 lanes computes the *same*
//    quotient (with the shipped `block_size_x = 128`, `k = k_base + 4*lane`
//    and `k_base` a multiple of 128, so `k / 128 == k_base / 128` for all
//    lanes). By instruction count that division is ~40% of the loop body.
//    Here it becomes a single SHF: the host passes `scale_shift =
//    log2(block_size_x)` and the kernel indexes with `k >> scale_shift`.
//    For non-negative `k` and power-of-two `block_size_x` that is *exactly*
//    `k / block_size_x` — the two agree bit for bit, no tolerance involved.
//
// 2. THE WEIGHT LOAD IS 32-BIT.
//    Line ~171: `__ldg(reinterpret_cast<const uint32_t *>(&w_row[k]))` — four
//    bytes per thread per iteration. A bandwidth-bound kernel must hold
//    bandwidth x latency bytes in flight (Little's law). On an H200 at
//    4.8 TB/s with ~600 ns of HBM latency that is 2.88 MB device-wide, i.e.
//    ~21.8 KB per SM across 132 SMs. `fp8_gemv_warp<T,4,8>` puts at most
//    1024 threads on an SM and holds U 4-byte weight loads per thread, so it
//    needs U >= 6 concurrent DRAM loads per thread to cover latency — and its
//    `__launch_bounds__(128, 8)` caps it at 64 registers, which will not hold
//    six iterations of live state. The `<T,1,16>` variant it selects for
//    N >= 8192 is worse still: 16 warps/SM (25% occupancy) x U x 4 B, so even
//    U = 4 reaches only 8 KB/SM = 37% of the requirement.
//
//    This kernel loads `uint4` — 16 bytes per thread per LDG, one warp
//    covering 512 contiguous weight bytes per instruction — and explicitly
//    double-buffers, so two iterations of loads are always outstanding:
//    1024 threads x 2 x 16 B = 32 KB/SM, ~1.5x the Little's-law requirement.
//
// NUMERICS — READ BEFORE CHANGING ANYTHING BELOW
// ----------------------------------------------
// This kernel is NOT bit-identical to `fp8_gemv_warp`, and cannot be: a
// 128-bit load forces lane L to own the 16 contiguous weights at
// `k_base + 16L`, where the 32-bit kernel gives it the 4 at `k_base + 4L`.
// A different lane->k partition means different per-lane partial sums, hence
// different f32 rounding.
//
// The difference is bounded to exactly one thing, and the code below is
// written to keep it that way:
//
//   * Both kernels partition [0, K) into the SAME aligned 4-element groups.
//   * For each group both evaluate `scale * (i0*f(w0) + i1*f(w1) + i2*f(w2)
//     + i3*f(w3))` through the identical expression, in the identical
//     left-to-right order, fused into a running f32 accumulator by one FFMA
//     (`fmaf(scale, group, acc)`). `fp8_group4` below is a verbatim copy of
//     the 32-bit kernel's group expression for precisely this reason.
//   * The scale multiplying each group is bit-identical (see point 1).
//
// So the two kernels accumulate the same multiset of identically-computed
// contributions and differ ONLY in the order of accumulation: the old kernel
// uses one accumulator per lane in ascending k; this one uses four, one per
// group slot within a lane's 16-wide slice. The output therefore differs by
// f32 re-association error alone. Four shorter chains are never worse than
// one long one, so the wide kernel is if anything the more accurate of the
// two against an f64 reference — `fp8_gemv_wide_mirror` in
// `blockwise_fp8/gemv_wide.rs` models both orders in Rust and asserts this.
//
// PRECONDITIONS (all checked on the host in `blockwise_fp8/ops.rs`; a shape
// that fails any of them keeps `fp8_gemv_warp` unchanged):
//   * K % 16 == 0                      — every row starts 16-byte aligned and
//                                        k = k_base + 16*lane is aligned.
//   * block_size_x is a power of two   — `k >> scale_shift == k / block_size_x`.
//   * block_size_x % 16 == 0           — a lane's whole 16-wide slice lies in
//                                        ONE scale block, so one scale load.
//   * weight and input base pointers are 16-byte aligned.
// ============================================================================

// Group of four: byte-for-byte the same expression as the 32-bit kernel's
// inner product. `xlo` holds T-elements 0,1 and `xhi` holds 2,3 of the group.
// `xlo`/`xhi` carry the raw 32 bits of one `half2` / `__nv_bfloat162` each.
// CUDA is little-endian, so bits 0..15 are `.x` and bits 16..31 are `.y`,
// matching the `__ldg(reinterpret_cast<const half2 *>(&in_row[k]))` that
// `fp8_gemv_warp` performs on the same four bytes.
//
// The values arrive in REGISTERS, not memory, so there is nothing to point a
// `reinterpret_cast` at. Two idioms are used instead, both chosen because
// this file already proves they compile and both bit-exact:
//
//   * f16: `__half_raw` carrying the 16 bits, handed to `__half2float` —
//     exactly what `fp8_to_float` above does with the result of
//     `__nv_cvt_fp8_to_halfraw`.
//   * bf16: `__uint_as_float(bits << 16)`. bf16 IS the top 16 bits of an f32,
//     so this is the same bit-move `__bfloat162float` performs, for every
//     pattern including subnormals, infinities and NaN payloads — and it does
//     not depend on a `__ushort_as_bfloat16` overload being present.
template <typename T>
__device__ __forceinline__ float fp8_group4(uint32_t w4, uint32_t xlo,
                                            uint32_t xhi) {
  float i0, i1, i2, i3;
  if constexpr (std::is_same_v<T, half>) {
    __half_raw h;
    h.x = (unsigned short)(xlo & 0xFFFFu);
    i0 = __half2float(h);
    h.x = (unsigned short)(xlo >> 16);
    i1 = __half2float(h);
    h.x = (unsigned short)(xhi & 0xFFFFu);
    i2 = __half2float(h);
    h.x = (unsigned short)(xhi >> 16);
    i3 = __half2float(h);
  } else {
    i0 = __uint_as_float((xlo & 0xFFFFu) << 16);
    i1 = __uint_as_float(xlo & 0xFFFF0000u);
    i2 = __uint_as_float((xhi & 0xFFFFu) << 16);
    i3 = __uint_as_float(xhi & 0xFFFF0000u);
  }

  __nv_fp8_e4m3 w0, w1, w2, w3;
  w0.__x = (w4 >> 0) & 0xFF;
  w1.__x = (w4 >> 8) & 0xFF;
  w2.__x = (w4 >> 16) & 0xFF;
  w3.__x = (w4 >> 24) & 0xFF;

  return i0 * fp8_to_float(w0) + i1 * fp8_to_float(w1) +
         i2 * fp8_to_float(w2) + i3 * fp8_to_float(w3);
}

template <typename T, int ROWS_PER_BLOCK, int MAX_BLOCKS_PER_SM>
__global__ __launch_bounds__(ROWS_PER_BLOCK * 32,
                             MAX_BLOCKS_PER_SM) void fp8_gemv_wide(
    const T *__restrict__ input,               // [M, K]
    const __nv_fp8_e4m3 *__restrict__ weight,  // [N, K] row-major
    const float *__restrict__ weight_scale,    // [ceil(N/bs_y), ceil(K/bs_x)]
    T *__restrict__ output,                    // [M, N]
    int M, int N, int K, int scale_row_stride, int block_size_y,
    int scale_shift) {
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
  const int m = blockIdx.y;
  if (n >= N || m >= M)
    return;

  const __nv_fp8_e4m3 *w_row = weight + (size_t)n * K;
  const T *in_row = input + (size_t)m * K;
  const float *scale_row = weight_scale + (n / block_size_y) * scale_row_stride;

  // Four independent accumulators, one per group slot inside a lane's 16-wide
  // slice. This is what breaks the single-accumulator FFMA dependency chain;
  // it is also the ONLY numerical difference from `fp8_gemv_warp` (see above).
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  // Wide loop: warp covers 32 lanes x 16 bytes = 512 K-elements per iteration.
  const int K_wide = (K / 512) * 512;
  const int lane_off = lane * 16;

  if (K_wide > 0) {
    // ---- Software pipeline ----
    // Each trip issues the NEXT iteration's four loads before consuming the
    // current iteration's values, so >= 2 iterations of loads are outstanding
    // at all times. This is structural, not a hope about nvcc's unroller: the
    // consuming FFMAs below read `w`/`xa`/`xb`, never `wn`/`xan`/`xbn`.
    const int k = lane_off;
    uint4 w = __ldg(reinterpret_cast<const uint4 *>(&w_row[k]));
    uint4 xa = __ldg(reinterpret_cast<const uint4 *>(&in_row[k]));
    uint4 xb = __ldg(reinterpret_cast<const uint4 *>(&in_row[k + 8]));
    float sc = __ldg(&scale_row[k >> scale_shift]);

    for (int k_base = 512; k_base < K_wide; k_base += 512) {
      const int kn = k_base + lane_off;
      const uint4 wn = __ldg(reinterpret_cast<const uint4 *>(&w_row[kn]));
      const uint4 xan = __ldg(reinterpret_cast<const uint4 *>(&in_row[kn]));
      const uint4 xbn = __ldg(reinterpret_cast<const uint4 *>(&in_row[kn + 8]));
      const float scn = __ldg(&scale_row[kn >> scale_shift]);

      acc0 = fmaf(sc, fp8_group4<T>(w.x, xa.x, xa.y), acc0);
      acc1 = fmaf(sc, fp8_group4<T>(w.y, xa.z, xa.w), acc1);
      acc2 = fmaf(sc, fp8_group4<T>(w.z, xb.x, xb.y), acc2);
      acc3 = fmaf(sc, fp8_group4<T>(w.w, xb.z, xb.w), acc3);

      w = wn;
      xa = xan;
      xb = xbn;
      sc = scn;
    }

    // Drain.
    acc0 = fmaf(sc, fp8_group4<T>(w.x, xa.x, xa.y), acc0);
    acc1 = fmaf(sc, fp8_group4<T>(w.y, xa.z, xa.w), acc1);
    acc2 = fmaf(sc, fp8_group4<T>(w.z, xb.x, xb.y), acc2);
    acc3 = fmaf(sc, fp8_group4<T>(w.w, xb.z, xb.w), acc3);
  }

  // ---- Tail ----
  // [K_wide, K) is handled by the 32-bit kernel's own two loops, verbatim, so
  // the tail is not a second place a bug can live. K % 16 == 0 is a host
  // precondition, so the scalar remainder below runs only when 16 | K but
  // 128 does not divide K.
  float acct = 0.0f;
  const int K_aligned = K_wide + ((K - K_wide) / 128) * 128;
  for (int k_base = K_wide; k_base < K_aligned; k_base += 128) {
    const int k = k_base + lane * 4;
    const uint32_t w4 = __ldg(reinterpret_cast<const uint32_t *>(&w_row[k]));
    const uint32_t xlo = __ldg(reinterpret_cast<const uint32_t *>(&in_row[k]));
    const uint32_t xhi =
        __ldg(reinterpret_cast<const uint32_t *>(&in_row[k + 2]));
    const float scale = __ldg(&scale_row[k >> scale_shift]);
    acct = fmaf(scale, fp8_group4<T>(w4, xlo, xhi), acct);
  }
  for (int k = K_aligned + lane; k < K; k += 32) {
    float in_val;
    if constexpr (std::is_same_v<T, half>) {
      in_val = __half2float(__ldg(&in_row[k]));
    } else {
      in_val = __bfloat162float(__ldg(&in_row[k]));
    }
    __nv_fp8_e4m3 w;
    w.__x = __ldg(reinterpret_cast<const uint8_t *>(&w_row[k]));
    const float scale = __ldg(&scale_row[k >> scale_shift]);
    acct += scale * in_val * fp8_to_float(w);
  }

  float acc = ((acc0 + acc1) + (acc2 + acc3)) + acct;

// Warp reduction using shuffle — identical to `fp8_gemv_warp`.
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

// ---------------------------------------------------------------------------
// Wide (128-bit-load) GEMV launcher. Opt-in: reached only when
// `ARC_FP8_GEMV_WIDE=1` AND every precondition in `blockwise_fp8/ops.rs`
// (`fp8_gemv_wide_eligible`) holds. `scale_shift` must be
// `log2(block_size_x)`; the Rust side is the only caller and computes it from
// a verified power-of-two `block_size_x`.
//
// Row grouping: 4 warps/block for EVERY N, not just N < 8192. The 32-bit
// kernel switches to `<T,1,16>` above 8192 rows, which caps residency at 16
// warps/SM; with 16-byte loads that is 512 threads x 2 x 16 B = 16 KB/SM,
// only ~73% of the ~21.8 KB/SM an H200 needs in flight to cover HBM latency,
// where `<T,4,8>` reaches ~32 KB/SM. That choice rests on the Little's-law
// arithmetic in the kernel header and on NOTHING MEASURED — like the rest of
// this kernel it is UNVERIFIED ON HARDWARE, which is why it is opt-in.
template <typename T>
static void launch_fp8_gemv_wide_impl(const T *input,
                                      const __nv_fp8_e4m3 *weight,
                                      const float *weight_scale, T *output,
                                      int M, int N, int K, int scale_row_stride,
                                      int block_size_y, int scale_shift,
                                      cudaStream_t stream) {
  dim3 grid(CEILDIV(N, 4), M);
  fp8_gemm::fp8_gemv_wide<T, 4, 8><<<grid, 4 * 32, 0, stream>>>(
      input, weight, weight_scale, output, M, N, K, scale_row_stride,
      block_size_y, scale_shift);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_fp8_gemv_wide_f16(const __half *input,
                                         const __nv_fp8_e4m3 *weight,
                                         const float *weight_scale,
                                         __half *output, int M, int N, int K,
                                         int scale_row_stride, int block_size_y,
                                         int scale_shift, cudaStream_t stream) {
  launch_fp8_gemv_wide_impl<half>(input, weight, weight_scale, output, M, N, K,
                                  scale_row_stride, block_size_y, scale_shift,
                                  stream);
}

extern "C" void
launch_fp8_gemv_wide_bf16(const __nv_bfloat16 *input,
                          const __nv_fp8_e4m3 *weight,
                          const float *weight_scale, __nv_bfloat16 *output,
                          int M, int N, int K, int scale_row_stride,
                          int block_size_y, int scale_shift,
                          cudaStream_t stream) {
  launch_fp8_gemv_wide_impl<__nv_bfloat16>(input, weight, weight_scale, output,
                                           M, N, K, scale_row_stride,
                                           block_size_y, scale_shift, stream);
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
