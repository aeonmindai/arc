// Parent system: ArcInfer / ArcAttention
//
// Fused per-head Q RMS-normalisation for DeepSeek V4 — SIX candle kernels
// collapsed into ONE launch.
//
// ---------------------------------------------------------------------------
// PROVENANCE
// ---------------------------------------------------------------------------
// Structural port of the upstream single-kernel RMSNorm pattern:
//
//   * vLLM, `csrc/layernorm_kernels.cu` — `rms_norm_kernel` (:108-181): one
//     block per row, the sum of squares reduced in shared memory inside the
//     kernel, the normalised row written back in the same launch, with no
//     intermediate tensor for the variance.
//   * SGLang, `sgl-kernel/csrc/.../fused_add_rmsnorm.cuh:57` — same shape:
//     bf16 in, register-level accumulator, bf16 out, one launch.
//
// Both Apache License 2.0:
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
// What is ported is the STRUCTURE (one launch, in-kernel reduction, no
// intermediate tensors), which is the thing Arc lacked. The ARITHMETIC is
// deliberately NOT upstream's: see the contract below.
//
// ---------------------------------------------------------------------------
// BIT-IDENTITY CONTRACT
// ---------------------------------------------------------------------------
// Replaces this chain in `models/deepseek4.rs` (V4 per-head Q RMSNorm,
// reference `inference/model.py:498`):
//
//   let inv_rms = q.sqr()?                          unary.cu  usqr_bf16   x*x
//       .mean_keepdim(D::Minus1)?                   reduce.cu fast_sum_bf16
//                                                   + affine_bf16(1/n, 0)
//       .affine(1.0, rms_norm_eps)?                 affine.cu affine_bf16
//       .recip()?                                   unary.cu  urecip_bf16
//       .sqrt()?;                                   unary.cu  usqrt_bf16
//   q.broadcast_mul(&inv_rms)?                      binary.cu bmul_bf16
//
// Every step above runs in BF16, including the 512-term sum of squares. This
// kernel reproduces that exactly rather than "fixing" it, because the bar for
// this port is bit-identity: changing the accumulator changes generated
// tokens. Three details that a naive rewrite gets wrong:
//
// 1. REDUCTION ORDER. candle's `sum_keepdim` on CUDA runs `fast_sum`
//    (reduce.cu:70) with
//      block_dim = min(1024, el_to_sum_per_block).next_power_of_two()
//    (cuda_backend/mod.rs:370) = 512 for head_dim = 512 — one element per
//    thread, `shr[tid] = 0` then `shr[tid] += x`, then the pairwise tree
//    `shr[t] += shr[t + s]` for s = 256, 128, ..., 1. That is NOT a
//    sequential accumulation and rounds differently from one. `qnorm_row_sum`
//    below replays that exact tree. This is the same trap that made the first
//    sinkhorn kernel fail its H200 A/B; see cuda/sinkhorn.cu.
//
// 2. BF16 ACCUMULATION, NOT FLOAT. Upstream (and every sane implementation)
//    accumulates the sum of squares in `float`. candle accumulates in bf16, so
//    this kernel does too. NOTE FOR A FOLLOW-UP: summing 512 squares with an
//    8-bit mantissa carries roughly sqrt(512) * 2^-8 ~ 9% relative error in
//    the norm. That is a QUALITY defect in the existing code, not something
//    this perf port introduces — and it is deliberately preserved here so the
//    A/B is clean. `ARC_QNORM_F32_ACC=1` switches the accumulator to float for
//    a future perplexity A/B; it is OFF by default and is NOT bit-identical.
//
// 3. THE eps AND 1/n CONSTANTS ARE BF16. candle's `affine` takes `TYPENAME
//    mul, TYPENAME add`, i.e. the host down-converts both to bf16 first
//    (`bf16::from_f64`). They are therefore passed in here as raw u16 bit
//    patterns computed on the Rust side with the same `from_f64`, not as
//    floats converted in-kernel.
//    `x * bf16(1/512) + bf16(0)` and `x * bf16(1.0) + bf16(eps)` are both
//    invariant to FMA contraction (scaling by a power of two is exact, adding
//    zero is exact, multiplying by one is exact), so `--fmad=false` on the
//    IEEE builder cannot change them either way.
//
// FAST MATH: bf16 `operator/` and `hsqrt` both round-trip through float
// internally, so `--use_fast_math` (-prec-div=false, -prec-sqrt=false) would
// perturb them. build.rs compiles this file in the dedicated IEEE builder and
// `assert_ieee_kernel_flags` enforces that; there is no `#error` guard because
// nvcc 12.4 defines no fast-math macro in either pass (see swiglu_clamp.cu).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define QNORM_MAX_BLOCK 1024

// candle's `fast_sum` reduction, replayed exactly: identity-initialised shared
// slot, one `+=` per element, then the pairwise tree. `n` must be <= blockDim.x
// (true whenever head_dim <= 1024, which the host checks).
__device__ __forceinline__ __nv_bfloat16
qnorm_row_sum(__nv_bfloat16 *shr, const __nv_bfloat16 v, const unsigned n) {
  const unsigned tid = threadIdx.x;
  // reduce.cu: `shr[tid] = 0;` then `shr[tid] += src[i]` for the elements this
  // thread owns. Threads past `n` contribute the additive identity.
  __nv_bfloat16 acc = 0;
  if (tid < n) {
    acc = acc + v;
  }
  shr[tid] = acc;
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if ((int)tid < s) {
      shr[tid] = shr[tid] + shr[tid + s];
    }
  }
  __syncthreads();
  return shr[0];
}

// float-accumulator variant (ARC_QNORM_F32_ACC=1). NOT bit-identical to the
// candle chain — kept for the quality A/B described in the contract above.
__device__ __forceinline__ float qnorm_row_sum_f32(float *shr, const float v,
                                                   const unsigned n) {
  const unsigned tid = threadIdx.x;
  shr[tid] = (tid < n) ? v : 0.0f;
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if ((int)tid < s) {
      shr[tid] = shr[tid] + shr[tid + s];
    }
  }
  __syncthreads();
  return shr[0];
}

// One block per row. `cols` = head_dim, `inv_n` = bf16(1/cols),
// `eps` = bf16(rms_norm_eps), both supplied as raw bit patterns from Rust so
// the down-conversion matches candle's `bf16::from_f64` exactly.
__global__ void qnorm_rms_bf16_kernel(const __nv_bfloat16 *__restrict__ x,
                                      __nv_bfloat16 *__restrict__ out,
                                      const unsigned cols,
                                      const unsigned short inv_n_bits,
                                      const unsigned short eps_bits,
                                      const int f32_acc) {
  __shared__ __nv_bfloat16 shr_bf[QNORM_MAX_BLOCK];
  __shared__ float shr_f32[QNORM_MAX_BLOCK];

  const unsigned tid = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * (size_t)cols;

  __nv_bfloat16 xi = 0;
  if (tid < cols) {
    xi = x[base + tid];
  }

  __nv_bfloat16 inv_rms;
  if (f32_acc) {
    // Quality-experiment path. Deliberately different arithmetic.
    const float xf = __bfloat162float(xi);
    const float sum = qnorm_row_sum_f32(shr_f32, xf * xf, cols);
    const float mean = sum / (float)cols;
    inv_rms = __float2bfloat16(1.0f / sqrtf(mean + __bfloat162float(
                                                      __ushort_as_bfloat16(eps_bits))));
  } else {
    // 1. usqr_bf16: x*x in bf16.
    const __nv_bfloat16 sq = xi * xi;
    // 2. fast_sum_bf16 pairwise tree over `cols`, in bf16.
    const __nv_bfloat16 sum = qnorm_row_sum(shr_bf, sq, cols);
    // 3. mean_keepdim's trailing `* (1/n)`: affine_bf16(mul=1/n, add=0).
    const __nv_bfloat16 inv_n = __ushort_as_bfloat16(inv_n_bits);
    const __nv_bfloat16 zero = 0;
    const __nv_bfloat16 mean = sum * inv_n + zero;
    // 4. affine_bf16(mul=1, add=eps).
    const __nv_bfloat16 one = 1;
    const __nv_bfloat16 eps = __ushort_as_bfloat16(eps_bits);
    const __nv_bfloat16 t = mean * one + eps;
    // 5. urecip_bf16: bf16(1) / t.
    const __nv_bfloat16 r = one / t;
    // 6. usqrt_bf16: hsqrt.
    inv_rms = hsqrt(r);
  }

  // 7. bmul_bf16 against the broadcast scalar.
  if (tid < cols) {
    out[base + tid] = xi * inv_rms;
  }
}

static inline unsigned qnorm_next_pow2(unsigned v) {
  unsigned p = 1;
  while (p < v) {
    p <<= 1;
  }
  return p;
}

// Returns 0 on success, 1 if `cols` exceeds what one block can reduce.
extern "C" int qnorm_rms_bf16(const __nv_bfloat16 *x, __nv_bfloat16 *out,
                              uint32_t rows, uint32_t cols,
                              unsigned short inv_n_bits, unsigned short eps_bits,
                              int f32_acc, int64_t stream) {
  if (cols == 0 || cols > QNORM_MAX_BLOCK || rows == 0) {
    return 1;
  }
  // Match candle's block_dim exactly: min(1024, n).next_power_of_two().
  const unsigned threads = qnorm_next_pow2(cols);
  qnorm_rms_bf16_kernel<<<rows, threads, 0, (cudaStream_t)stream>>>(
      x, out, cols, inv_n_bits, eps_bits, f32_acc);
  return 0;
}
