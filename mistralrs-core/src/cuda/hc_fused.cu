// Parent system: ArcInfer / ArcMoE
//
// Fused router-region kernels for DeepSeek-V4 (mHC pre-step + MoE gate scoring).
//
// ---------------------------------------------------------------------------
// WHY
// ---------------------------------------------------------------------------
// At b=1 the V4 decode step is launch-bound, not bandwidth-bound. Measured on
// an H200 at master ef581e9c8: 7,477.9 `cuLaunchKernel` calls per decode step
// costing 18.976 ms/step of host time, of which the router region is 76 kernels
// per layer (3,268/step, 43.7%) spread over 86 contiguous spans -- exactly two
// per layer (`mhc_attn_pre` and `mhc_ffn_pre`, 43 layers). Those spans operate
// on `[1, 24]` / `[1, 4, 4]` / `[1, 256]` tensors: mean kernel duration 2.72 us,
// 89% of them under 5 us, 28.4% of all kernel time. They are pure launch
// overhead wearing a kernel costume.
//
// MEASURED RESULT of this file plus the sinkhorn fix, same box, same binary,
// toggled with ARC_HC_FUSED: 7,494.3 -> 5,691.0 launches/step (-24.1%),
// 10,892.4 -> 8,918.5 allocations/step, 57.70 -> 49.66 ms/token
// (17.3 -> 20.1 tok/s). The router region itself goes 76 -> 38 kernels/layer
// and 0.207 -> 0.100 ms of GPU time. Generated tokens and logprobs are
// bit-identical across the toggle (6 prompts, 768 logprob values, 0 mismatches).
//
// Two expressions dominate the count and are collapsed here:
//
//  1. `hc_pre` (dsv4_mhc.rs) spells a hand-decomposed RMS statistic in SEVEN
//     launches -- `sqr -> fast_sum -> affine(1/D) -> affine(+eps) -> urecip ->
//     usqrt -> bmul` -- and then a further ELEVEN for the pre/post/comb
//     sigmoid-and-bias scoring. Eighteen launches whose entire data footprint
//     after the reduction is 24 floats.
//
//  2. `MoeGate::forward` (deepseek4.rs) spells `sqrt(softplus(x))` as NINE
//     launches: `zeros_like -> bmaximum -> uabs -> uneg -> uexp -> affine(+1)
//     -> ulog -> badd -> usqrt`.
//
// ---------------------------------------------------------------------------
// BIT-IDENTITY CONTRACT
// ---------------------------------------------------------------------------
// The mHC pre-step feeds `y` into attention and the gate scores decide WHICH
// EXPERTS RUN. A reassociated sum or a contracted FMA can flip an expert choice
// and therefore the generated token, so these kernels are bit-identical to the
// candle op chains they replace, BY CONSTRUCTION, not by tolerance. The same
// three rules that `sinkhorn.cu` documents apply verbatim:
//
// 1. REDUCTION ORDER. candle's `sum_keepdim` on CUDA runs candle-kernels
//    `fast_sum` (reduce.cu) with `block_dim = min(1024, el_to_sum_per_block)
//    .next_power_of_two()` (cuda_backend/mod.rs `FastReduce`). Each thread
//    accumulates a *strided* slice sequentially into a zero-initialized
//    accumulator (`shr[tid] = 0; shr[tid] += src[idx]; idx += blockDim.x`),
//    then a pairwise tree `shr[t] += shr[t + s]` for s = block/2 .. 1. For
//    D = hc_mult * hidden = 16384 that is 1024 threads x 16 elements each, and
//    the tree order is NOT the sequential order. `hc_pre_fused_f32_kernel`
//    replays it exactly, including the `shr[tid] = 0` initialisation (which
//    canonicalises -0.0f to +0.0f) and the identity padding when a thread's
//    strided slice is empty.
//
// 2. UNFUSED, IEEE ROUND-TO-NEAREST ARITHMETIC. candle-kernels build with
//    plain -O3 and NO --use_fast_math, so their division is `div.rn.f32`,
//    their `expf`/`logf` are the accurate libdevice `__nv_expf`/`__nv_logf`,
//    and denormals are not flushed. mistralrs-core's build.rs compiles the
//    rest of src/cuda/*.cu WITH --use_fast_math, which would silently rewrite
//    all three. THIS FILE IS THEREFORE COMPILED BY THE SAME DEDICATED
//    no-fast-math / --fmad=false builder that sinkhorn.cu uses, and carries
//    the same #error guard against a future re-glob. All arithmetic uses
//    __fadd_rn / __fmul_rn / __fsub_rn, which nvcc documents as never merged
//    into an FMA, so the fusion cannot silently change rounding.
//
// 3. EXACT OP TRANSCRIPTION, including the ones that look like no-ops:
//    - candle's `recipg(float a)` is `return 1.0 / a;` -- the literal is a
//      *double*, so the op is `(float)(1.0 / (double)a)`, not `__frcp_rn(a)`.
//      Transcribed literally in `candle_recip`.
//    - `Tensor * f64` / `Tensor + f64` lower to `affine(mul, add)` whose
//      kernel is `x * mul + add`, compiled by candle with the default
//      -fmad=true, i.e. `fmaf(x, mul, add)`. Transcribed as explicit `fmaf`
//      (an explicit fmaf is unaffected by our --fmad=false, which only
//      governs *contraction* of separate mul+add).
//    - `mean_keepdim` is `sum_impl(..)? * (1f64 / len as f64)`, i.e. a
//      `fast_sum` followed by a SEPARATE affine -- the division is not folded
//      into the reduction.
//    - `candle_nn::ops::sigmoid` dispatches to candle-kernels `usigmoid_f32`
//      = `sigmoid_fwd(x)` = `recipg(1.0f + expf(-x))`.
//
// Scalar Rust replicas of both sides (the candle op chain and this kernel) are
// asserted bit-identical over randomized inputs in `cuda/hc_fused.rs`
// `mod tests`, so a future edit to either side trips CPU CI without a GPU. The
// final proof is the on-GPU A/B: `ARC_HC_AB=1` recomputes the candle chain
// alongside the fused kernel and reports any bitwise mismatch.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// ---------------------------------------------------------------------------
// THE COMPILE-TIME FAST-MATH GUARD BELOW IS DEAD. MEASURED, NOT ASSUMED.
// ---------------------------------------------------------------------------
// `nvcc --use_fast_math` defines NEITHER `__USE_FAST_MATH__` NOR `__FAST_MATH__`
// in either the host or the device pass. Probed on this box, CUDA 12.4:
//
//   $ nvcc --use_fast_math -E -dM x.cu | grep -i fast     ->  (no output)
//   $ nvcc --use_fast_math -arch=sm_90 -c guard.cu        ->  compiles clean
//                                        (with #error on both macros present)
//
// So the `#if defined(__USE_FAST_MATH__)` below has never once been capable of
// firing -- in this file or in sinkhorn.cu, which carries the same guard. It is
// left in place ONLY so that this comment sits next to it; do not read it as
// protection. The hazard it was written for is real: the same probe shows
// `--use_fast_math` cutting a float divide from 3 MUFU/RCP instructions to 1,
// i.e. it does reach the device pass and does change the arithmetic.
//
// The guard that actually works is at RUNTIME and on REAL DATA: `ARC_HC_AB=1`
// recomputes the eager candle chain next to every fused kernel and reports the
// first bitwise mismatch (see `ab_check` in hc_fused.rs). It subsumes the
// compile-flag question -- a fast-math build fails it on the first token -- and
// it has been proven RED by injecting a 1-ULP perturbation.
#if defined(__USE_FAST_MATH__)
#error "hc_fused.cu must be compiled WITHOUT --use_fast_math: fast math rewrites expf/logf to the hardware approximations and IEEE division to approximate reciprocals, breaking bit-identity with candle-kernels (which build with plain -O3). See the dedicated no-fast-math builder in mistralrs-core/build.rs."
#endif

namespace {

// candle-kernels cuda_utils.cuh:
//   __device__ __forceinline__ float recipg(float a) { return 1.0 / a; }
// The literal `1.0` is a double, so `a` is promoted, the division is done in
// double, and the result is narrowed on return. Transcribed literally rather
// than "simplified" to __frcp_rn: the two happen to agree for all finite
// inputs, but the point of this file is to not rely on such arguments.
__device__ __forceinline__ float candle_recip(float a) {
    return (float)(1.0 / (double)a);
}

// candle-kernels unary.cu:
//   __device__ __forceinline__ T sigmoid_fwd(T x) {
//       return recipg(static_cast<T>(1) + expg(-x));
//   }
// with expg(float) == expf (libdevice __nv_expf under no-fast-math).
__device__ __forceinline__ float candle_sigmoid(float x) {
    return candle_recip(__fadd_rn(1.0f, expf(-x)));
}

} // namespace

extern "C" {

// ---------------------------------------------------------------------------
// hc_pre: the 18-launch RMS-scale + pre/post/comb scoring block, in one launch.
// ---------------------------------------------------------------------------
//
// Replaces, verbatim (dsv4_mhc.rs `hc_pre`):
//
//   let sq_mean = x_flat.sqr()?.mean_keepdim(D::Minus1)?;              // 3
//   let rsqrt   = (sq_mean + rms_norm_eps)?.recip()?.sqrt()?;          // 3
//   let mixes   = mixes_raw.broadcast_mul(&rsqrt)?;                    // 1
//   let pre     = sigmoid(pre_block *s_pre  +b_pre)? + hc_eps;         // 4
//   let post    = sigmoid(post_block*s_post +b_post)?.affine(2,0);     // 4
//   let comb_pre= comb_block*s_comb + b_comb;                          // 3
//
// The trailing 3 for `comb` include the `.reshape((n, hc, hc))` of a narrowed
// (non-contiguous) view, which candle services with a real copy kernel; here
// the reshape is implicit in the output indexing and costs nothing.
//
// `mixes` itself is never materialised: it is consumed only by the three
// narrows, so the fused kernel keeps it in registers.
//
// Launch contract (enforced host-side in hc_fused.rs):
//   grid  = n blocks (one per row, matching candle's `dst_el` grid)
//   block = min(1024, d).next_power_of_two()  -- candle's FastReduce block_dim
//   shmem = blockDim.x * sizeof(float)
//   m <= blockDim.x, and x_flat / mixes_raw are contiguous F32.
__global__ void hc_pre_fused_f32_kernel(
    const float *__restrict__ x_flat,    // [n, d]      F32 contiguous
    const float *__restrict__ mixes_raw, // [n, m]      F32 contiguous
    const float *__restrict__ hc_scale,  // [3]         F32
    const float *__restrict__ hc_base,   // [m]         F32
    float *__restrict__ pre,             // [n, hc]     out
    float *__restrict__ post,            // [n, hc]     out
    float *__restrict__ comb_pre,        // [n, hc, hc] out
    int d,
    int m,
    int hc,
    float inv_d,   // (float)(1.0 / d) -- candle's mean scale, T::from_f64
    float rms_eps, // (float) rms_norm_eps
    float hc_eps   // (float) hc_eps
) {
    extern __shared__ float shr[];
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int bd = blockDim.x;

    // ---- candle `sqr()` + `fast_sum` replay -------------------------------
    // candle runs these as two kernels; the intermediate `x*x` is rounded to
    // f32 before the accumulation, which __fmul_rn reproduces. The strided
    // walk and the zero-initialised accumulator are fast_sum's, verbatim.
    const float *xrow = x_flat + (size_t)n * (size_t)d;
    float acc = 0.0f;
    for (int idx = tid; idx < d; idx += bd) {
        const float v = xrow[idx];
        acc = __fadd_rn(acc, __fmul_rn(v, v));
    }
    shr[tid] = acc;
    // fast_sum's pairwise tree. The __syncthreads() sits at the TOP of the
    // body there too, so the write above is visible before the first read.
    for (int s = bd >> 1; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            shr[tid] = __fadd_rn(shr[tid], shr[tid + s]);
        }
    }
    __syncthreads();

    // ---- mean_keepdim tail, then `+eps -> recip -> sqrt` ------------------
    // Every thread recomputes this from shr[0]; it is deterministic scalar
    // arithmetic on one value, so all threads agree bit for bit.
    const float sum = shr[0];
    const float mean = fmaf(sum, inv_d, 0.0f);    // mean_keepdim: affine(1/d, 0)
    const float meps = fmaf(mean, 1.0f, rms_eps); // (sq_mean + rms_norm_eps)
    const float rsqrt = sqrtf(candle_recip(meps));

    // ---- broadcast_mul into `mixes`, then the three scoring blocks --------
    if (tid < m) {
        const float mx = __fmul_rn(mixes_raw[(size_t)n * (size_t)m + tid], rsqrt);
        if (tid < hc) {
            // pre = sigmoid(pre_block * s_pre + b_pre) + hc_eps
            const float t = __fadd_rn(__fmul_rn(mx, hc_scale[0]), hc_base[tid]);
            pre[(size_t)n * (size_t)hc + tid] = fmaf(candle_sigmoid(t), 1.0f, hc_eps);
        } else if (tid < 2 * hc) {
            // post = 2 * sigmoid(post_block * s_post + b_post)
            const int j = tid - hc;
            const float t = __fadd_rn(__fmul_rn(mx, hc_scale[1]), hc_base[hc + j]);
            post[(size_t)n * (size_t)hc + j] = fmaf(candle_sigmoid(t), 2.0f, 0.0f);
        } else {
            // comb_pre = comb_block * s_comb + b_comb   (reshape is free here)
            const int j = tid - 2 * hc;
            const float t = __fadd_rn(__fmul_rn(mx, hc_scale[2]), hc_base[2 * hc + j]);
            comb_pre[(size_t)n * (size_t)hc * (size_t)hc + j] = t;
        }
    }
}

void hc_pre_fused_f32(
    const void *x_flat,
    const void *mixes_raw,
    const void *hc_scale,
    const void *hc_base,
    void *pre,
    void *post,
    void *comb_pre,
    int n,
    int d,
    int m,
    int hc,
    int block_dim,
    float inv_d,
    float rms_eps,
    float hc_eps,
    long long stream
) {
    dim3 grid(n, 1, 1);
    dim3 block(block_dim, 1, 1);
    size_t shmem = (size_t)block_dim * sizeof(float);
    hc_pre_fused_f32_kernel<<<grid, block, shmem, (cudaStream_t)stream>>>(
        (const float *)x_flat,
        (const float *)mixes_raw,
        (const float *)hc_scale,
        (const float *)hc_base,
        (float *)pre,
        (float *)post,
        (float *)comb_pre,
        d,
        m,
        hc,
        inv_d,
        rms_eps,
        hc_eps);
}

// ---------------------------------------------------------------------------
// sqrt(softplus(x)): the V4 gate scoring function, in one launch instead of 9.
// ---------------------------------------------------------------------------
//
// Replaces, verbatim (deepseek4.rs `MoeGate::forward`, ScoringFunc::SqrtSoftplus):
//
//   let max0     = logits.maximum(&logits.zeros_like()?)?;   // zeros + bmaximum
//   let abs      = logits.abs()?;                            // uabs
//   let softplus = (max0 + ((abs.neg()?.exp()? + 1.0)?.log()?))?;
//   softplus.sqrt()?
//
// candle op-for-op: bmaximum is `maxg(x, y)` == `fmaxf`; uabs is `fabsf`;
// uneg is `-x`; uexp is `expf`; `+ 1.0` is `affine(1.0, 1.0)` == `fmaf(x,1,1)`;
// ulog is `logf`; the outer `+` is `badd` == `x + y`; usqrt is `sqrtf`.
__global__ void sqrt_softplus_f32_kernel(
    const float *__restrict__ inp,
    float *__restrict__ out,
    long long numel
) {
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    for (long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < numel;
         i += stride) {
        const float l = inp[i];
        const float mx = fmaxf(l, 0.0f);      // maximum(logits, zeros_like)
        const float a = fabsf(l);             // abs()
        const float e = expf(-a);             // neg() then exp()
        const float p1 = fmaf(e, 1.0f, 1.0f); // + 1.0  (affine)
        const float lg = logf(p1);            // log()
        out[i] = sqrtf(__fadd_rn(mx, lg));    // max0 + ... then sqrt()
    }
}

void sqrt_softplus_f32(const void *inp, void *out, long long numel, long long stream) {
    const int block = 256;
    long long blocks = (numel + block - 1) / block;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 65535) {
        blocks = 65535;
    }
    sqrt_softplus_f32_kernel<<<(unsigned)blocks, block, 0, (cudaStream_t)stream>>>(
        (const float *)inp, (float *)out, numel);
}

} // extern "C"

// ===========================================================================
// The mHC epilogue and post-step, one launch each.
// ===========================================================================
//
// After the router-region fusion above, one mHC mixing point is still FOURTEEN
// launches at b=1 -- and there are 86 of them per token (43 layers x {attn,
// ffn}). Counted from an nsys trace on an H200, not from reading the source:
// per mixing point, cast_bf16_f32 x3, cast_f32_bf16 x2, bmul_f32 x2,
// fast_sum_f32, badd_f32, sm80_xmma_gemm, dot_kernel + reduce_1Block (cuBLAS
// splits the [1,16384]x[16384,24] gate GEMV into two), hc_pre_fused, sinkhorn.
//
// The two blocks collapsed here are the nine that are not the gate GEMV:
//
//   hc_pre tail   `pre.unsqueeze(-1).broadcast_mul(x_f32).sum(1)` + narrowing
//                 -- 3 launches to compute a 4-term weighted average.
//   hc_post whole `to_dtype x` + `to_dtype residual` + `broadcast_mul` +
//                 `transpose.contiguous` + `matmul` + `add` + `to_dtype`
//                 -- 6 launches whose real work is 4 multiply-adds per output.
//
// MEASURED RESULT, H200, clocks locked at 1980 MHz, ONE binary toggled with
// ARC_HC_FUSE2 (a two-binary before/after would confound this with the
// rebuild): 6,192.5 -> 5,566.4 launches/step (-626.1, -10.1%) and 40.50 ->
// 38.18 ms/token (24.7 -> 26.2 tok/s). Every kernel that moved is one of these
// nine, each at 86/step: cast_bf16_f32 -172.2, cast_f32_bf16 -172.1,
// bmul_f32 -172.0, fast_sum_f32 -86.1, badd_f32 -86.0, sm80_xmma_gemm -86.0,
// against hc_post_fused +85.9 and hc_y_combine +85.9. GPU time falls too, by
// 1.48 ms/step, because the intermediate [1,4,4096] F32 tensors are never
// materialised.
//
// Both operate on [1, 4, 7168] at b=1: ~114 kB of traffic behind ~9 kernel
// launches. They are launch overhead wearing a kernel costume, the same
// diagnosis this file already applied to the scoring block.
//
// The GEMM in `hc_pre` (`x_flat @ hc_fn.T`, [1,28672]x[28672,24]) is
// DELIBERATELY LEFT ALONE. It is cuBLAS, its K=28672 reduction order is not
// reproducible by construction, and bit-identity is the contract here. Fusing
// it would be the only way to merge `hc_post` with the next `hc_pre` into a
// single cross-layer kernel; that trade -- one launch per seam against a
// tolerance-based result -- is not taken.

namespace {

// Widen to f32. BF16->F32 is an exact widening, so this introduces no rounding
// and the fused path sees exactly the bits candle's `to_dtype(F32)` produced.
__device__ __forceinline__ float hc_widen(float v) { return v; }
__device__ __forceinline__ float hc_widen(__nv_bfloat16 v) { return __bfloat162float(v); }

// Narrow back to the model dtype. candle's cast kernel is
// `static_cast<__nv_bfloat16>(x)`, i.e. `__float2bfloat16`, round-to-nearest-
// even -- the same intrinsic, not an approximation of it.
template <typename T> __device__ __forceinline__ T hc_narrow(float v);
template <> __device__ __forceinline__ float hc_narrow<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 hc_narrow<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// ---------------------------------------------------------------------------
// y = sum_i pre[i] * x[i, :]        (hc_pre tail, 3 launches -> 1)
// ---------------------------------------------------------------------------
//
// Replaces verbatim (dsv4_mhc.rs `hc_pre`):
//   let y     = pre.unsqueeze(-1)?.broadcast_mul(&x_f32)?.sum(1)?;   // 2
//   let y_out = y.reshape(y_shape)?.to_dtype(in_dtype)?;             // 1
//
// REDUCTION ORDER -- read from candle, not remembered. `sum(1)` on [n, hc, h]
// moves the summed dim last (cuda_backend/mod.rs `FastReduce`), giving
// dst_el = n*h and el_to_sum_per_block = hc, hence
// block_dim = min(1024, hc).next_power_of_two() = hc for a power-of-two hc.
// candle-kernels `fast_sum` (reduce.cu:70-103) then runs
//
//     shr[tid] = 0;                       // note: canonicalises -0.0f to +0.0f
//     while (idx < stop) shr[tid] += src[strided_i];   // ONE element per thread
//     for (s = bd/2; s; s >>= 1) { __syncthreads(); if (tid<s) shr[tid]+=shr[tid+s]; }
//
// so for hc = 4 the result is ((0+a0)+(0+a2)) + ((0+a1)+(0+a3)) -- a pairwise
// tree, NOT the sequential a0+a1+a2+a3. Replayed exactly below, including the
// add-to-zero that fixes the sign of zero.
//
// HC is a template parameter so the accumulator lives in registers; a runtime
// bound would put `acc[]` in local memory, which is the spill this file's
// sinkhorn fix already had to undo once.
template <typename OUT_T, int HC>
__global__ void hc_y_combine_kernel(
    const float *__restrict__ x_f32, // [n, hc, h] F32 contiguous
    const float *__restrict__ pre,   // [n, hc]    F32
    OUT_T *__restrict__ y,           // [n, h]     out, model dtype
    int h,
    long long total // n * h
) {
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    for (long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const long long row = i / (long long)h;
        const int col = (int)(i - row * (long long)h);

        const float *prow = pre + row * HC;
        const float *xrow = x_f32 + row * (long long)HC * (long long)h + col;

        float acc[HC];
#pragma unroll
        for (int t = 0; t < HC; ++t) {
            // `shr[tid] = 0` then `shr[tid] += a_t`: the add-to-zero is part of
            // candle's kernel and is NOT a no-op for a_t == -0.0f.
            acc[t] = __fadd_rn(0.0f, __fmul_rn(prow[t], xrow[(long long)t * (long long)h]));
        }
#pragma unroll
        for (int s = HC >> 1; s > 0; s >>= 1) {
#pragma unroll
            for (int t = 0; t < HC; ++t) {
                if (t < s) {
                    acc[t] = __fadd_rn(acc[t], acc[t + s]);
                }
            }
        }
        y[i] = hc_narrow<OUT_T>(acc[0]);
    }
}

// ---------------------------------------------------------------------------
// hc_post: re-expand one branch output into the hc residual streams (6 -> 1)
// ---------------------------------------------------------------------------
//
// Replaces verbatim (dsv4_mhc.rs `hc_post`):
//   let x_n        = x.reshape((n,h))?.to_dtype(F32)?;                    // 1
//   let residual_n = residual.reshape((n,hc,h))?.to_dtype(F32)?;          // 1
//   let term1 = post_n.unsqueeze(-1)?.broadcast_mul(&x_n.unsqueeze(1)?)?; // 1
//   let term2 = comb_n.transpose(1,2)?.matmul(&residual_n)?;              // 2
//                                    (the transpose forces a contiguous copy)
//   let out_n = (term1 + term2)?;                                         // 1
//   out_n.reshape(out_shape)?.to_dtype(in_dtype)                          // 1
//
// term2 is a [hc,hc] x [hc,h] product with K = hc = 4. The K-loop is written
// here as the sequential `acc = fmaf(c, r, acc)` from a zero accumulator, which
// is what a cuBLAS GEMM with K=4 does; unlike the K=28672 GEMM in `hc_pre` this
// is short enough that the order is checkable rather than assumed, and it IS
// checked -- `ARC_HC_AB=1` compares this kernel against the eager chain above
// bit for bit on live decode data.
template <typename T, int HC>
__global__ void hc_post_fused_kernel(
    const T *__restrict__ x,        // [n, h]      model dtype
    const T *__restrict__ residual, // [n, hc, h]  model dtype
    const float *__restrict__ post, // [n, hc]     F32
    const float *__restrict__ comb, // [n, hc, hc] F32
    T *__restrict__ out,            // [n, hc, h]  model dtype
    int h,
    long long total // n * hc * h
) {
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    const long long hh = (long long)HC * (long long)h;
    for (long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const long long row = i / hh;
        const long long rem = i - row * hh;
        const int k = (int)(rem / (long long)h);
        const int col = (int)(rem - (long long)k * (long long)h);

        // term1 = post[k] * x[col]; candle's broadcast_mul is binary.cu `x * y`,
        // a bare multiply with nothing to contract into an FMA.
        const float xv = hc_widen(x[row * (long long)h + col]);
        const float t1 = __fmul_rn(post[row * HC + k], xv);

        // term2 = sum_j comb[j][k] * residual[j][col]
        const float *crow = comb + row * (long long)HC * (long long)HC;
        const T *rrow = residual + row * hh + col;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < HC; ++j) {
            acc = fmaf(crow[j * HC + k], hc_widen(rrow[(long long)j * (long long)h]), acc);
        }

        out[i] = hc_narrow<T>(__fadd_rn(t1, acc));
    }
}

// 256 threads, grid capped so the grid-stride loop stays valid for any size.
inline void hc_launch_dims(long long total, unsigned &blocks, int &block) {
    block = 256;
    long long b = (total + block - 1) / block;
    if (b < 1) {
        b = 1;
    }
    if (b > 65535) {
        b = 65535;
    }
    blocks = (unsigned)b;
}

} // namespace

extern "C" {

// dtype tags shared with hc_fused.rs. Anything else must not reach these.
#define HC_DTYPE_F32 0
#define HC_DTYPE_BF16 1

// Returns 0 on launch, non-zero if (hc, dtype) is outside the specialised set —
// the caller then falls back to the eager chain rather than producing a wrong
// answer with a generic slow path.
int hc_y_combine(
    const void *x_f32,
    const void *pre,
    void *y,
    int hc,
    int h,
    long long total,
    int dtype,
    long long stream
) {
    unsigned blocks;
    int block;
    hc_launch_dims(total, blocks, block);
    cudaStream_t s = (cudaStream_t)stream;
#define HC_YC(T, HCV)                                                                      \
    hc_y_combine_kernel<T, HCV><<<blocks, block, 0, s>>>(                                   \
        (const float *)x_f32, (const float *)pre, (T *)y, h, total)
    if (dtype == HC_DTYPE_F32) {
        switch (hc) {
        case 2: HC_YC(float, 2); return 0;
        case 4: HC_YC(float, 4); return 0;
        case 8: HC_YC(float, 8); return 0;
        case 16: HC_YC(float, 16); return 0;
        default: return 1;
        }
    } else if (dtype == HC_DTYPE_BF16) {
        switch (hc) {
        case 2: HC_YC(__nv_bfloat16, 2); return 0;
        case 4: HC_YC(__nv_bfloat16, 4); return 0;
        case 8: HC_YC(__nv_bfloat16, 8); return 0;
        case 16: HC_YC(__nv_bfloat16, 16); return 0;
        default: return 1;
        }
    }
    return 1;
#undef HC_YC
}

int hc_post_fused(
    const void *x,
    const void *residual,
    const void *post,
    const void *comb,
    void *out,
    int hc,
    int h,
    long long total,
    int dtype,
    long long stream
) {
    unsigned blocks;
    int block;
    hc_launch_dims(total, blocks, block);
    cudaStream_t s = (cudaStream_t)stream;
#define HC_PF(T, HCV)                                                                      \
    hc_post_fused_kernel<T, HCV><<<blocks, block, 0, s>>>(                                  \
        (const T *)x, (const T *)residual, (const float *)post, (const float *)comb,        \
        (T *)out, h, total)
    if (dtype == HC_DTYPE_F32) {
        switch (hc) {
        case 2: HC_PF(float, 2); return 0;
        case 4: HC_PF(float, 4); return 0;
        case 8: HC_PF(float, 8); return 0;
        case 16: HC_PF(float, 16); return 0;
        default: return 1;
        }
    } else if (dtype == HC_DTYPE_BF16) {
        switch (hc) {
        case 2: HC_PF(__nv_bfloat16, 2); return 0;
        case 4: HC_PF(__nv_bfloat16, 4); return 0;
        case 8: HC_PF(__nv_bfloat16, 8); return 0;
        case 16: HC_PF(__nv_bfloat16, 16); return 0;
        default: return 1;
        }
    }
    return 1;
#undef HC_PF
}

} // extern "C"
