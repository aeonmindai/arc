// Parent system: ArcInfer / ArcMoE
//
// Fused clamped-SwiGLU for DeepSeek V4: `silu(min(gate, L)) * clamp(up, -L, L)`
// in ONE launch, replacing the 7-kernel candle op chain Arc used before.
//
// ---------------------------------------------------------------------------
// PROVENANCE (this is a PORT, not an invention)
// ---------------------------------------------------------------------------
// Ported from, and semantically identical to:
//
//   * vLLM, `csrc/activation_kernels.cu` — `silu_and_mul_clamp_kernel` /
//     `silu_and_mul_clamp` (the launcher at :298), whose Python layer is
//     `vllm/model_executor/layers/activation.py:200`. vLLM wrote that kernel
//     for exactly DeepSeek V4's `swiglu_limit`.
//   * SGLang, `sgl-kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh:55-73`
//     — `silu_and_mul<kApplySwigluLimit = true>`, which cites DeepGEMM
//     `sm100_fp8_fp4_mega_moe.cuh:984-997` as its source of truth.
//
// Both projects are Apache License 2.0:
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
// The upstream reference does the clamp+activation in bf16x2 (`__hmin2`/
// `__hmax2`) with an f32 accumulator. Arc's pre-port candle chain did the
// clamp in f32 after an explicit upcast. This port keeps ARC'S f32 ordering,
// not upstream's bf16 ordering, because the bit-identity contract below is
// against the chain being replaced. The three semantic properties upstream
// pins are preserved exactly:
//   1. the clamp is PRE-activation — the product is never clamped;
//   2. `gate` is clamped ONLY from above (one-sided);
//   3. `up` is clamped on both sides.
//
// ---------------------------------------------------------------------------
// BIT-IDENTITY CONTRACT
// ---------------------------------------------------------------------------
// This kernel must produce bit-identical output to the candle op chain it
// replaces (`moe::swiglu_clamp` + `Tensor::apply(Silu)` + `mul` + `to_dtype`):
//
//   gate_f32 = (float)gate                      cast.cu   cast_bf16_f32
//   gate_f32 = fminf(gate_f32, L)               binary.cu bminimum_f32 (ming)
//   up_f32   = (float)up                        cast.cu   cast_bf16_f32
//   up_f32   = fmaxf(up_f32, -L)                binary.cu bmaximum_f32 (maxg)
//   up_f32   = fminf(up_f32, L)                 binary.cu bminimum_f32
//   act      = gate_f32 / (1.0f + expf(-gate_f32))
//                                               unary.cu  usilu_f32 (silu_fwd)
//   prod     = up_f32 * act                     binary.cu bmul_f32
//   out      = (bf16)prod                       cast.cu   cast_f32_bf16
//
// Two things make that reproducible here and would silently break it:
//
// 1. FAST MATH. candle-kernels build with plain `-O3` and NO
//    `--use_fast_math`, so candle's `expf` is the accurate libdevice
//    `__nv_expf` and its division is `div.rn.f32`. mistralrs-core's build.rs
//    compiles the REST of `src/cuda/*.cu` with `--use_fast_math`, which
//    rewrites `expf -> __expf`, IEEE division -> approximate reciprocal
//    (`-prec-div=false`), and flushes denormals (`-ftz=true`). build.rs
//    therefore compiles THIS FILE in the dedicated no-fast-math IEEE builder
//    (the same one sinkhorn.cu uses).
//
//    Measured PTX for exactly this expression (nvcc 12.4, sm_90, H200):
//      plain -O3            neg.f32 fma.rn.f32(x14) add.f32 div.rn.f32
//      --use_fast_math      mul.ftz.f32 ex2.approx.ftz.f32 add.ftz.f32
//                           div.approx.ftz.f32
//    i.e. the whole accurate-expf expansion is replaced. The drift is real.
//
//    !! There is NO `#error` guard here on purpose. nvcc 12.4 defines NO
//    preprocessor macro for `--use_fast_math` in either the host or the
//    device pass — probed directly on this toolchain, `__USE_FAST_MATH__`,
//    `__CUDA_FAST_MATH__`, `__FAST_MATH__`, `__CUDACC_FAST_MATH__`,
//    `__CUDA_PREC_DIV__` and `__CUDA_FTZ__` are all undefined with and
//    without the flag. sinkhorn.cu's `#if defined(__USE_FAST_MATH__)` guard
//    therefore NEVER FIRES and is not protecting anything. The real guard is
//    `assert_ieee_kernel_flags()` in mistralrs-core/build.rs, which asserts
//    the flag sets at the only place the regression can be introduced.
//
// 2. FMA CONTRACTION. The IEEE builder also passes `--fmad=false`, while
//    candle-kernels build with the default `--fmad=true`. Verified on the same
//    toolchain that this does not perturb the accurate `expf`: libdevice's
//    expansion uses EXPLICIT `fma.rn.f32` (not source-level contraction), and
//    the two PTX instruction streams for this kernel differ only in
//    `add.f32`/`mul.f32` vs `add.rn.f32`/`mul.rn.f32` — the same operations
//    with the same (default) rounding mode. The candle chain has no
//    contractible `a*b+c` anyway: its only add is `1.0f + expf(-x)`, whose
//    operands are a constant and a call result. The explicit `__fadd_rn` /
//    `__fdiv_rn` / `__fmul_rn` below make the guarantee independent of flags.
//
// `mul` is commutative and exact in IEEE-754, so `up * act` (routed-expert
// chain) and `act * up` (shared-expert chain, via ops.cu `fused_glu`) give the
// same bits; this kernel serves both.
//
// The negative-control-backed A/B lives in `cuda/swiglu_clamp.rs mod tests`
// (`#[cfg(feature = "cuda")]`, runs on the GPU).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Element conversions. Explicit intrinsics (not the implicit operators) so the
// file compiles without -U__CUDA_NO_*_CONVERSIONS__, which the IEEE builder
// does not pass. Both directions are round-to-nearest-even, identical to what
// candle's cast.cu `out[i] = inp[i]` emits.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sgc_to_f32(const __nv_bfloat16 x) {
  return __bfloat162float(x);
}
__device__ __forceinline__ float sgc_to_f32(const __half x) {
  return __half2float(x);
}
__device__ __forceinline__ float sgc_to_f32(const float x) { return x; }

__device__ __forceinline__ void sgc_from_f32(__nv_bfloat16 *dst, float v) {
  *dst = __float2bfloat16(v);
}
__device__ __forceinline__ void sgc_from_f32(__half *dst, float v) {
  *dst = __float2half(v);
}
__device__ __forceinline__ void sgc_from_f32(float *dst, float v) { *dst = v; }

// ---------------------------------------------------------------------------
// The arithmetic, in exactly the candle chain's order and rounding.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sgc_swiglu_clamped(float g, float u,
                                                    const float limit) {
  // 1. gate: ONE-SIDED upper clamp (candle `.minimum(limit)`).
  g = fminf(g, limit);
  // 2. up: SYMMETRIC clamp (candle `.clamp(-limit, limit)` == maximum, minimum).
  u = fmaxf(u, -limit);
  u = fminf(u, limit);
  // 3. silu on the clamped gate, in f32 (candle `usilu_f32`: x/(1+expf(-x))).
  const float act = __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
  // 4. product, in f32 (candle `bmul_f32`).
  return __fmul_rn(act, u);
}

template <typename T>
__global__ void
sgc_swiglu_clamp_kernel(const T *__restrict__ gate, const T *__restrict__ up,
                        T *__restrict__ out, const uint32_t n,
                        const float limit) {
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float v =
        sgc_swiglu_clamped(sgc_to_f32(gate[i]), sgc_to_f32(up[i]), limit);
    sgc_from_f32(&out[i], v);
  }
}

// Vector-4 variant. Purely a load/store widening — the per-element arithmetic
// is the same `sgc_swiglu_clamped`, so it is bit-identical to the scalar path.
// Only dispatched when n % 4 == 0 and all three pointers are 4*sizeof(T)
// aligned (checked host-side).
template <typename T, typename V>
__global__ void
sgc_swiglu_clamp_kernel_vec4(const V *__restrict__ gate,
                             const V *__restrict__ up, V *__restrict__ out,
                             const uint32_t n4, const float limit) {
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    V g4 = gate[i];
    V u4 = up[i];
    V o4;
    const T *g = reinterpret_cast<const T *>(&g4);
    const T *u = reinterpret_cast<const T *>(&u4);
    T *o = reinterpret_cast<T *>(&o4);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      sgc_from_f32(
          &o[k], sgc_swiglu_clamped(sgc_to_f32(g[k]), sgc_to_f32(u[k]), limit));
    }
    out[i] = o4;
  }
}

// Grid sizing: cap the grid so tiny decode-time tensors do not launch tens of
// thousands of empty blocks, but stay 1:1 for the shapes that matter.
static inline uint32_t sgc_grid(uint32_t work, uint32_t threads) {
  const uint32_t blocks = (work + threads - 1) / threads;
  return blocks < 65535u ? (blocks == 0u ? 1u : blocks) : 65535u;
}

static inline bool sgc_aligned(const void *a, const void *b, const void *c,
                               size_t bytes) {
  const uintptr_t m = static_cast<uintptr_t>(bytes) - 1;
  return ((reinterpret_cast<uintptr_t>(a) & m) == 0) &&
         ((reinterpret_cast<uintptr_t>(b) & m) == 0) &&
         ((reinterpret_cast<uintptr_t>(c) & m) == 0);
}

#define SGC_LAUNCH(T, V, VEC_BYTES)                                            \
  do {                                                                         \
    const uint32_t threads = 256;                                              \
    if ((n % 4u) == 0u && sgc_aligned(gate, up, out, VEC_BYTES)) {             \
      const uint32_t n4 = n / 4u;                                              \
      sgc_swiglu_clamp_kernel_vec4<T, V>                                       \
          <<<sgc_grid(n4, threads), threads, 0, (cudaStream_t)stream>>>(       \
              reinterpret_cast<const V *>(gate),                               \
              reinterpret_cast<const V *>(up), reinterpret_cast<V *>(out), n4, \
              limit);                                                          \
    } else {                                                                   \
      sgc_swiglu_clamp_kernel<T>                                               \
          <<<sgc_grid(n, threads), threads, 0, (cudaStream_t)stream>>>(        \
              gate, up, out, n, limit);                                        \
    }                                                                          \
  } while (0)

extern "C" void swiglu_clamp_bf16(const __nv_bfloat16 *gate,
                                  const __nv_bfloat16 *up, __nv_bfloat16 *out,
                                  uint32_t n, float limit, int64_t stream) {
  SGC_LAUNCH(__nv_bfloat16, uint64_t, 8);
}

extern "C" void swiglu_clamp_f16(const __half *gate, const __half *up,
                                 __half *out, uint32_t n, float limit,
                                 int64_t stream) {
  SGC_LAUNCH(__half, uint64_t, 8);
}

extern "C" void swiglu_clamp_f32(const float *gate, const float *up, float *out,
                                 uint32_t n, float limit, int64_t stream) {
  SGC_LAUNCH(float, float4, 16);
}
