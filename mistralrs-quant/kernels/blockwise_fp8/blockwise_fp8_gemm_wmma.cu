/**
 * Parent system: ArcKernels.
 *
 * Blockwise-FP8 GEMM on TENSOR CORES, with an FP32 accumulator promoted at
 * every scale-block boundary along K.
 *
 * ############################################################################
 * # UNVERIFIED ON HARDWARE -- never run.                                     #
 * #                                                                          #
 * # No GPU was available when this file was written. Nothing in it has        #
 * # executed. Every performance number in these comments is a DERIVATION      #
 * # from published machine limits, not a measurement, and is labelled as      #
 * # such. Do not quote any of them as a result. The first box that runs       #
 * # this must A/B it against `fp8_matmul_tiled` (see the kill switch note     #
 * # at the bottom) before anything here is believed.                         #
 * ############################################################################
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * `fp8_matmul_tiled` in blockwise_fp8_gemm.cu is a SCALAR CUDA-core GEMM. Its
 * inner loop is
 *
 *     acc += s_input[ty][k] * s_weight[tx][k];
 *
 * -- two shared-memory float loads to feed ONE FMA, with no register blocking
 * and no tensor-core instruction anywhere in it. Shared memory delivers 32
 * floats/clk/SM against 128 FP32 lanes, so that loop cannot exceed ~1/4 of the
 * FP32 CUDA-core rate no matter how it is tuned; it is a structural ceiling,
 * not a tuning problem. The fix is not a better scalar loop, it is to stop
 * being scalar.
 *
 * THE ONE THING THAT MAKES BLOCKWISE FP8 AWKWARD
 * ----------------------------------------------
 * The weight carries a `[ceil(N/bs_y), ceil(K/bs_x)]` grid of f32 scales
 * (typically 128x128). The scale CHANGES every `bs_x` elements along the
 * reduction axis, so you cannot hoist it out of the K loop, and you must not
 * fold it into the operands: `bs` is an arbitrary f32 and rounding
 * `fp8_value * scale` into bf16/fp16 to feed the tensor core would throw away
 * most of the mantissa the scale is carrying.
 *
 * The structure that resolves this is DeepGEMM's: accumulate the UNSCALED
 * products in an FP32 tensor-core accumulator across one scale block, then
 * PROMOTE -- multiply that accumulator by the block's f32 scale and add it
 * into a second, long-lived FP32 accumulator -- and reset. Hence
 * `K_BLK == bs_x`: one K-tile is exactly one scale block, so the scale is a
 * single scalar for the whole tile and the promotion happens once per tile.
 *
 * Numerically this is BETTER than the scalar kernel it replaces, not a
 * trade. The scalar kernel computes `f32(act) * f32(w * scale)` and rounds
 * every product to f32. Here the FP8 -> fp16/bf16 weight conversion is EXACT
 * (e4m3 carries 3 mantissa bits and an exponent range of 2^-9..2^8; both fp16
 * and bf16 represent every one of the 256 e4m3 values exactly), the tensor
 * core forms each product at full width into an f32 accumulator, and the
 * scale is applied ONCE per 128 K-elements instead of once per element.
 *
 * WHY WMMA AND NOT wgmma
 * ----------------------
 * `wgmma.mma_async` would be worth roughly another 1.3-1.5x on Hopper, and it
 * is reachable here -- cudaforge auto-suffixes sm_90 to sm_90a
 * (cudaforge-0.1.5 `compute_cap.rs::auto_suffix`: `b if b >= 90 =>
 * with_suffix(b, "a")`), which is the target wgmma requires. It is not used
 * because this file cannot be executed before it is committed. Hand-rolled
 * `wgmma` descriptors and swizzled shared-memory layouts are wrong far more
 * often than they are right on the first try, and a wrong GEMM here is
 * silent: it returns plausible logits. `nvcuda::wmma` fixes the fragment
 * layout in the compiler, compiles unchanged for both sm_80 and sm_90a, and
 * already has a working precedent in this tree (kernels/mxfp4/
 * mxfp4_gemm_wmma.cu). That precedent is the reason this is the version that
 * ships first. The wgmma rung is a follow-on, and it should be written
 * against a box that can run the A/B this one sets up.
 *
 * DERIVED COST (a derivation, NOT a measurement)
 * ----------------------------------------------
 * V4 at B=256, the shape this targets: 7 blockwise-FP8 sites/layer x 43
 * layers = 4.238 G params, FLOP = 2 * 256 * 4.238e9 = 2.170 TFLOP.
 *
 *   H200 FP32 non-tensor      67 TFLOP/s  -> 32.4 ms   (what the scalar
 *                                                       kernel is bounded by,
 *                                                       and it achieves ~6.8%
 *                                                       of even that)
 *   H200 BF16/FP16 tensor  989.5 TFLOP/s  ->  2.19 ms  <-- THIS KERNEL'S BOUND
 *   H200 FP8 tensor       1979   TFLOP/s  ->  1.10 ms
 *   HBM at 4.8 TB/s, 4.238 GB of FP8      ->  0.88 ms
 *
 * So this kernel is compute-bound at a DERIVED 2.19 ms floor, against a
 * measured 524 ms for `fp8_matmul_tiled`. A WMMA GEMM of this shape typically
 * realises 50-70% of the tensor-core peak, which would put it at a DERIVED
 * 3.1-4.4 ms. All of that is arithmetic; none of it has run.
 *
 * The remaining 2x to the 1.10 ms FP8-tensor-core number is NOT reachable by
 * tuning this kernel. `mma` with e4m3 operands requires BOTH operands in FP8,
 * so it needs the ACTIVATION quantized to FP8 with its own per-token 128
 * scales -- a numerics change and a separate kernel, deliberately not done
 * here. Named, not built.
 *
 * ALSO CHECKED, AND THE REASON THIS IS HAND-WRITTEN
 * -------------------------------------------------
 * cuBLASLt gained exactly this layout -- `CUBLASLT_MATMUL_DESC_B_SCALE_MODE =
 * CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F` -- and it is documented at
 * compute capability 9.0, i.e. Hopper, our target. But the enum does not
 * exist before CUDA 12.9 (absent in 12.4.1, absent in 12.8.0, present in
 * 12.9.0), and .github/workflows/cuda_compile_check.yaml pins 12.4.1. It also
 * requires FP8 activations with VEC128 per-token scales and TN layout, so it
 * is not the one-attribute change it first looks like. Worth revisiting if
 * the toolkit pin ever moves; it is not available at the pin we build against.
 *
 * --use_fast_math (applied crate-wide by build.rs) -- WHICH SIDE WE ARE ON
 * -----------------------------------------------------------------------
 * Unaffected. This kernel contains no transcendentals, no division and no
 * sqrt, so the flags that rewrite those do not apply. `-fmad=true` only
 * contracts the promotion `acc += c_frag.x[i] * ws` into an FMA, which is
 * more accurate, not less. `--ftz=true` flushes f32 denormals, and nothing
 * here goes near 1e-38: e4m3 spans 2^-9..2^8, the checkpoint scales are
 * normal f32, and the accumulators are sums of such products. Tensor-core
 * MMA is not governed by these flags at all.
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>

using namespace nvcuda::wmma;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace fp8_gemm_wmma {

// ============================================================================
// Tiling
// ============================================================================

constexpr int WMMA_M_DIM = 16;
constexpr int WMMA_N_DIM = 16;
constexpr int WMMA_K_DIM = 16;

// 8 warps: 4 along M, 2 along N; each warp owns two 16-wide N sub-tiles.
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int N_SUB_TILES = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;  // 8
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * 32; // 256

constexpr int M_BLK = WARPS_M * WMMA_M_DIM;             // 64
constexpr int N_BLK = WARPS_N * N_SUB_TILES * WMMA_N_DIM; // 64

// K_BLK is the scale-block length along K. The Rust dispatcher only selects
// this kernel when `block_size_x % K_BLK == 0`, which is what makes the scale
// a single scalar per (block, k-tile) and lets the promotion happen once per
// tile. Changing this constant changes that contract -- update
// `fp8_wmma_eligible` in blockwise_fp8/ops.rs with it.
constexpr int K_BLK = 128;
constexpr int WMMA_K_STEPS = K_BLK / WMMA_K_DIM; // 8

// Shared-memory row padding, in ELEMENTS of T.
//
// 16, and it must stay a multiple of 16. READ THIS BEFORE CHANGING IT -- the
// two NVIDIA documents that govern it disagree, and the laxer one is the one
// people quote.
//
// The CUDA C++ Programming Guide 12.4 §7.24.1 says only:
//
//     "mptr must be a 256-bit aligned pointer ... and [ldm] must be a
//      multiple of 8 for __half element type or multiple of 4 for float
//      element type. (i.e., multiple of 16 bytes in both cases)."
//
// The PTX ISA 12.4 §9.7.13.3.2 is STRICTER, and it is the binding one:
//
//     "The starting address of each instance of the leading dimension (row
//      or column) must be aligned with the size of the corresponding
//      fragment in bytes."
//
// and works the example through for exactly our shape: for
// `wmma.load.a.sync.aligned.row.m16n16k16.f16` the fragment is 32 B (eight
// `.f16x2` elements), so "p is a multiple of 32" and "2*s is a multiple of
// 32" -- i.e. ldm must be a multiple of SIXTEEN __half elements, double what
// the C++ guide states. A +8 pad gives ldm = 136, and 136 * 2 B = 272 B is
// NOT a multiple of 32 B: it satisfies the guide and violates the ISA, which
// is the worst of both worlds because it compiles.
//
// +16 gives ldm = 144 elements = 288 B = 9 * 32 B, which satisfies both, for
// f16 and bf16 alike. Every fragment base pointer this kernel forms is then
// 32-B aligned as well (144 * 2 = 288 and 16 * 2 = 32 are both multiples of
// 32, and B_sh starts 18,432 B into a 16-B-aligned block).
//
// This is also the OPPOSITE of the right answer for the scalar kernel next
// door, where a +1 pad breaks a 4-way bank conflict. There the loads are
// ordinary per-element `LDS.32` with no alignment rule; here they are
// fragment loads that have one. Do not "harmonise" the two.
constexpr int SMEM_PAD = 16;
constexpr int A_LDM = K_BLK + SMEM_PAD; // 144 elements = 288 B = 9 * 32 B
constexpr int B_LDM = K_BLK + SMEM_PAD; // 144 elements
static_assert(A_LDM * 2 % 32 == 0,
              "WMMA leading dimension must be a multiple of 32 bytes (PTX ISA "
              "9.7.13.3.2), not merely of 16 as the C++ guide states");
static_assert(B_LDM * 2 % 32 == 0, "see above");

// One K-tile of A and of B, plus (aliased over A) the f32 output staging tile.
constexpr int A_SMEM_ELEMS = M_BLK * A_LDM; // 64 * 144 = 9216
constexpr int B_SMEM_ELEMS = N_BLK * B_LDM; // 64 * 144 = 9216
constexpr int C_SMEM_ELEMS = M_BLK * N_BLK; // 4096 floats

// Bytes, for a 16-bit T. Static shared memory, so we stay under the 48 KB
// per-block limit that needs no `cudaFuncSetAttribute` opt-in:
//   (9216 + 9216) * 2 = 36,864 B.
// The f32 staging tile is 4096 * 4 = 16,384 B and ALIASES A_sh (18,432 B),
// which is only safe because it is written after the K loop has finished with
// A. The __syncthreads() before that reuse is load-bearing.
constexpr int AB_SMEM_BYTES = (A_SMEM_ELEMS + B_SMEM_ELEMS) * 2;
static_assert(C_SMEM_ELEMS * sizeof(float) <= A_SMEM_ELEMS * 2,
              "f32 output staging tile must fit inside the A tile it aliases");
static_assert(AB_SMEM_BYTES <= 48 * 1024,
              "static shared memory must stay under the 48 KB no-opt-in limit");

using AccFrag = fragment<accumulator, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, float>;

// Number of f32 registers a 16x16 accumulator fragment exposes per lane.
// Asserted rather than assumed: if a future CUDA changes the fragment layout,
// this fails the build instead of silently promoting the wrong registers.
constexpr int ACC_REGS = 8;

// ============================================================================
// FP8 -> T conversion
//
// Built on `__nv_cvt_fp8_to_halfraw`, the same primitive the scalar kernel in
// blockwise_fp8_gemm.cu already uses, so the conversion path is one this tree
// has compiled before. It is a __host__ __device__ function from cuda_fp8.h
// and is available regardless of compute capability -- on pre-sm_89 it is
// emulated in software rather than absent, which is what lets this file
// compile and be correct for the sm_80 CI lane as well as sm_90a.
//
// EXACTNESS: e4m3 has 3 mantissa bits and unbiased exponents in [-9, 8]. fp16
// (10 mantissa bits, exp [-24, 15]) and bf16 (7 mantissa bits, exp
// [-126, 127]) each represent all 256 e4m3 values exactly, so no scale
// information is lost here. The scale itself is deliberately NOT applied at
// this point -- it is applied to the f32 accumulator in the promotion step.
// ============================================================================

__device__ __forceinline__ half fp8_to_half(uint8_t bits) {
  // Deliberately the same expression shape the scalar kernel next door
  // already compiles: `__nv_cvt_fp8_to_halfraw` returns `__half_raw`, and the
  // implicit `__half_raw` -> `__half` conversion is available because
  // build.rs passes `-U__CUDA_NO_HALF_CONVERSIONS__`.
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(bits, __NV_E4M3);
  return hr;
}

template <typename T> __device__ __forceinline__ T fp8_to_T(uint8_t bits);

template <> __device__ __forceinline__ half fp8_to_T<half>(uint8_t bits) {
  return fp8_to_half(bits);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fp8_to_T<__nv_bfloat16>(uint8_t bits) {
  return __float2bfloat16(__half2float(fp8_to_half(bits)));
}

template <typename T> __device__ __forceinline__ T zero_of();
template <> __device__ __forceinline__ half zero_of<half>() {
  return __float2half(0.0f);
}
template <> __device__ __forceinline__ __nv_bfloat16 zero_of<__nv_bfloat16>() {
  return __float2bfloat16(0.0f);
}

template <typename T> __device__ __forceinline__ float T_to_float(T v);
template <> __device__ __forceinline__ float T_to_float<half>(half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float T_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T float_to_T(float v);
template <> __device__ __forceinline__ half float_to_T<half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 float_to_T<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// ============================================================================
// The kernel
//
// C[M, N] = A[M, K] * B[N, K]^T, A in T, B in FP8 e4m3 with a
// [ceil(N/bs_y), ceil(K/bs_x)] f32 scale grid, C in T.
//
// Grid: (CEILDIV(N, N_BLK), CEILDIV(M, M_BLK)). Block: 256 threads.
//
// PRECONDITIONS, enforced by `fp8_wmma_eligible` in blockwise_fp8/ops.rs:
//   * block_size_y % N_BLK == 0   (a 64-wide, 64-aligned N tile never
//                                  straddles a scale row)
//   * block_size_x % K_BLK == 0   (a K tile is exactly one scale block, so
//                                  the scale is one scalar per tile)
// M, N and K are otherwise unconstrained; ragged edges are zero-filled.
// ============================================================================

template <typename T>
__launch_bounds__(BLOCK_THREADS) __global__ void fp8_matmul_wmma(
    const T *__restrict__ input,              // [M, K]
    const __nv_fp8_e4m3 *__restrict__ weight, // [N, K] row-major
    const float *__restrict__ weight_scale,   // [ceil(N/bs_y), ceil(K/bs_x)]
    T *__restrict__ output,                   // [M, N]
    int M, int N, int K, int scale_row_stride, int block_size_y,
    int block_size_x) {
  static_assert(AccFrag::num_elements == ACC_REGS,
                "16x16x16 f32 accumulator fragment is not 8 registers/lane; "
                "the promotion loop below assumes it is");

  // __align__(128), not 16. The PTX ISA rule quoted at SMEM_PAD requires the
  // fragment base pointer itself to be a multiple of 32 B ("p is a multiple
  // of 32"), and every pointer this kernel forms is `base + k*32`, so a
  // 16-B-aligned base would put every one of them on a 16-B boundary and
  // violate it. 128 covers that with room to spare and costs at most 127 B of
  // a 36,864 B allocation.
  __shared__ __align__(128) char smem_raw[AB_SMEM_BYTES];
  T *A_sh = reinterpret_cast<T *>(smem_raw);
  T *B_sh = reinterpret_cast<T *>(smem_raw) + A_SMEM_ELEMS;

  const int threadId = threadIdx.x;
  const int warpId = threadId >> 5;
  const int warp_m_idx = warpId / WARPS_N; // 0..3
  const int warp_n_idx = warpId % WARPS_N; // 0..1

  const int m_base = blockIdx.y * M_BLK;
  const int n_base = blockIdx.x * N_BLK;

  // The scale row is fixed for the whole block: the precondition
  // `block_size_y % N_BLK == 0` guarantees rows [n_base, n_base + N_BLK) all
  // land in scale row n_base / block_size_y.
  const int scale_row_off = (n_base / block_size_y) * scale_row_stride;

  // Long-lived FP32 accumulators. `c_frag` accumulates ONE scale block and is
  // reset after every promotion; `acc` carries the scaled running total for
  // the whole K extent.
  AccFrag c_frag[N_SUB_TILES];
  float acc[N_SUB_TILES][ACC_REGS];
#pragma unroll
  for (int s = 0; s < N_SUB_TILES; ++s) {
    fill_fragment(c_frag[s], 0.0f);
#pragma unroll
    for (int i = 0; i < ACC_REGS; ++i)
      acc[s][i] = 0.0f;
  }

  const T kZero = zero_of<T>();

  for (int k_base = 0; k_base < K; k_base += K_BLK) {
    // ---- Stage A[M_BLK, K_BLK] ------------------------------------------
    // 64 * 128 = 8192 elements over 256 threads = 32 each. Loaded 8 at a time
    // (16 B) when the row is fully in range, which is the common case.
    constexpr int A_ELEMS = M_BLK * K_BLK;
    constexpr int VEC = 8; // 8 * 2 B = 16 B
    for (int i = threadId * VEC; i < A_ELEMS; i += BLOCK_THREADS * VEC) {
      const int lm = i / K_BLK;
      const int lk = i % K_BLK;
      const int gm = m_base + lm;
      const int gk = k_base + lk;
      T *dst = &A_sh[lm * A_LDM + lk];

      if (gm < M && gk + VEC <= K) {
        // A row of `input` is K elements of 2 B; the 16 B load is aligned
        // whenever K % 8 == 0, which the dispatcher does not require, so go
        // through a bytewise-safe vector type only when the address allows.
        const T *src = &input[(size_t)gm * K + gk];
        if ((reinterpret_cast<uintptr_t>(src) & 0xF) == 0) {
          *reinterpret_cast<float4 *>(dst) =
              *reinterpret_cast<const float4 *>(src);
        } else {
#pragma unroll
          for (int v = 0; v < VEC; ++v)
            dst[v] = src[v];
        }
      } else {
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
          const int gk_v = gk + v;
          dst[v] = (gm < M && gk_v < K) ? input[(size_t)gm * K + gk_v] : kZero;
        }
      }
    }

    // ---- Stage B[N_BLK, K_BLK], converted FP8 -> T, UNSCALED -------------
    // 64 * 128 = 8192 FP8 bytes over 256 threads = 32 each, read as 8
    // uint32_t (4 weights per load).
    constexpr int B_ELEMS = N_BLK * K_BLK;
    constexpr int WVEC = 4; // 4 * 1 B = 4 B
    for (int i = threadId * WVEC; i < B_ELEMS; i += BLOCK_THREADS * WVEC) {
      const int ln = i / K_BLK;
      const int lk = i % K_BLK;
      const int gn = n_base + ln;
      const int gk = k_base + lk;
      T *dst = &B_sh[ln * B_LDM + lk];

      if (gn < N && gk + WVEC <= K) {
        const uint8_t *src =
            reinterpret_cast<const uint8_t *>(&weight[(size_t)gn * K + gk]);
        if ((reinterpret_cast<uintptr_t>(src) & 0x3) == 0) {
          const uint32_t w4 = __ldg(reinterpret_cast<const uint32_t *>(src));
#pragma unroll
          for (int v = 0; v < WVEC; ++v)
            dst[v] = fp8_to_T<T>(static_cast<uint8_t>((w4 >> (8 * v)) & 0xFF));
        } else {
#pragma unroll
          for (int v = 0; v < WVEC; ++v)
            dst[v] = fp8_to_T<T>(__ldg(src + v));
        }
      } else {
#pragma unroll
        for (int v = 0; v < WVEC; ++v) {
          const int gk_v = gk + v;
          if (gn < N && gk_v < K) {
            dst[v] = fp8_to_T<T>(__ldg(reinterpret_cast<const uint8_t *>(
                &weight[(size_t)gn * K + gk_v])));
          } else {
            // Zero weight, so the padded lanes contribute nothing to the
            // accumulator and the promotion below stays correct on ragged
            // edges without any masking.
            dst[v] = kZero;
          }
        }
      }
    }

    __syncthreads();

    // ---- Tensor-core accumulation over this one scale block --------------
#pragma unroll
    for (int k_step = 0; k_step < WMMA_K_STEPS; ++k_step) {
      fragment<matrix_a, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, T, row_major>
          a_frag;
      load_matrix_sync(a_frag,
                       A_sh + warp_m_idx * WMMA_M_DIM * A_LDM +
                           k_step * WMMA_K_DIM,
                       A_LDM);

#pragma unroll
      for (int s = 0; s < N_SUB_TILES; ++s) {
        // B_sh is [N][K]; as a K x N operand that is column-major with
        // leading dimension B_LDM.
        fragment<matrix_b, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, T, col_major>
            b_frag;
        load_matrix_sync(b_frag,
                         B_sh +
                             (warp_n_idx * N_SUB_TILES + s) * WMMA_N_DIM * B_LDM +
                             k_step * WMMA_K_DIM,
                         B_LDM);
        mma_sync(c_frag[s], a_frag, b_frag, c_frag[s]);
      }
    }

    // ---- PROMOTE: apply this block's f32 scale, fold in, reset -----------
    // This is the whole reason the kernel is shaped this way. `ws` is one
    // scalar for the entire tile because K_BLK divides block_size_x and
    // N_BLK divides block_size_y.
    {
      const float ws = __ldg(&weight_scale[scale_row_off + k_base / block_size_x]);
#pragma unroll
      for (int s = 0; s < N_SUB_TILES; ++s) {
#pragma unroll
        for (int i = 0; i < ACC_REGS; ++i)
          acc[s][i] += c_frag[s].x[i] * ws;
        fill_fragment(c_frag[s], 0.0f);
      }
    }

    __syncthreads();
  }

  // ---- Store ------------------------------------------------------------
  // Move the promoted f32 totals back into fragments so store_matrix_sync can
  // lay them out, then stage in shared memory and write coalesced.
  //
  // C_sh ALIASES A_sh. The __syncthreads() at the end of the K loop above is
  // what makes that safe -- every warp is done reading A before any warp
  // writes C over it.
  float *C_sh = reinterpret_cast<float *>(smem_raw);
#pragma unroll
  for (int s = 0; s < N_SUB_TILES; ++s) {
#pragma unroll
    for (int i = 0; i < ACC_REGS; ++i)
      c_frag[s].x[i] = acc[s][i];
    store_matrix_sync(C_sh + warp_m_idx * WMMA_M_DIM * N_BLK +
                          (warp_n_idx * N_SUB_TILES + s) * WMMA_N_DIM,
                      c_frag[s], N_BLK, mem_row_major);
  }
  __syncthreads();

  for (int i = threadId; i < C_SMEM_ELEMS; i += BLOCK_THREADS) {
    const int lm = i / N_BLK;
    const int ln = i % N_BLK;
    const int gm = m_base + lm;
    const int gn = n_base + ln;
    if (gm < M && gn < N)
      output[(size_t)gm * N + gn] = float_to_T<T>(C_sh[lm * N_BLK + ln]);
  }
}

} // namespace fp8_gemm_wmma

// ============================================================================
// C API
//
// Signature-identical to `launch_fp8_matmul_{f16,bf16}` in
// blockwise_fp8_gemm.cu, so the Rust dispatcher can swap one for the other
// with no other change. That is deliberate: it is what makes the kill switch
// a single boolean.
// ============================================================================

extern "C" void launch_fp8_matmul_wmma_f16(
    const __half *input, const __nv_fp8_e4m3 *weight, const float *weight_scale,
    __half *output, int M, int N, int K, int scale_row_stride, int block_size_y,
    int block_size_x, cudaStream_t stream) {
  dim3 block(fp8_gemm_wmma::BLOCK_THREADS);
  dim3 grid(CEILDIV(N, fp8_gemm_wmma::N_BLK), CEILDIV(M, fp8_gemm_wmma::M_BLK));

  fp8_gemm_wmma::fp8_matmul_wmma<half><<<grid, block, 0, stream>>>(
      input, weight, weight_scale, output, M, N, K, scale_row_stride,
      block_size_y, block_size_x);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_fp8_matmul_wmma_bf16(
    const __nv_bfloat16 *input, const __nv_fp8_e4m3 *weight,
    const float *weight_scale, __nv_bfloat16 *output, int M, int N, int K,
    int scale_row_stride, int block_size_y, int block_size_x,
    cudaStream_t stream) {
  dim3 block(fp8_gemm_wmma::BLOCK_THREADS);
  dim3 grid(CEILDIV(N, fp8_gemm_wmma::N_BLK), CEILDIV(M, fp8_gemm_wmma::M_BLK));

  fp8_gemm_wmma::fp8_matmul_wmma<__nv_bfloat16><<<grid, block, 0, stream>>>(
      input, weight, weight_scale, output, M, N, K, scale_row_stride,
      block_size_y, block_size_x);
  CUDA_CHECK(cudaGetLastError());
}

// Reported to the Rust side so the dispatcher's eligibility test cannot drift
// from the kernel's actual tiling. See `fp8_wmma_eligible` in
// blockwise_fp8/ops.rs.
extern "C" void fp8_matmul_wmma_tile_dims(int *m_blk, int *n_blk, int *k_blk) {
  *m_blk = fp8_gemm_wmma::M_BLK;
  *n_blk = fp8_gemm_wmma::N_BLK;
  *k_blk = fp8_gemm_wmma::K_BLK;
}
