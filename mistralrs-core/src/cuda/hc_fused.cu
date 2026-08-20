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

// ---------------------------------------------------------------------------
// THE F32 SEAM (3 kernels below)
// ---------------------------------------------------------------------------
//
// Everything above collapses chains that were *already* F32 end to end. The
// three kernels below close the other half of the problem: expressions that
// are BF16 on both sides but were spelled as "widen -> a few elementwise ops
// in F32 -> narrow". Every entry and exit paid a `cast_*` launch and every
// intermediate was a materialised F32 tensor. Measured on this H200 at
// b424f5cd9 (agent/datamove-fuse), per decode forward:
//
//   cast_bf16_f32 356.7 | cast_f32_bf16 221.1 | bminimum_f32 172.0
//   bmaximum_f32   86.0 | bmul_f32       94.4 | fast_sum_f32   98.8
//
// `bmaximum_f32 = 86.0` and `bminimum_f32 = 172.0` are exact, not approximate:
// 86 = 2 x 43 layers is the number of `swiglu_clamp` calls per forward (one
// routed-expert stage + one shared-expert stage per layer), and each contains
// one `maximum` and two `minimum`. That fingerprint is what identified the
// seam.
//
// A SECOND, INVISIBLE COST. `Tensor::minimum(f64)` / `maximum(f64)` build the
// scalar operand as `Tensor::new(v, Cpu)?.to_dtype()?.to_device(cuda)?` on
// every call (candle tensor.rs `binary_op_scalar!`). That is a 4-byte
// host-to-device memcpy per clamp bound -- 3 per `swiglu_clamp`, 258 per
// forward -- which never appears in a kernel histogram at all.
//
// BIT-IDENTITY. Same contract as the rest of this file: transcription, not
// re-derivation. The op-by-op sources are quoted at each kernel. `ARC_SEAM_AB=1`
// re-runs the candle chain next to each kernel and compares raw bits, and it
// compares them at **F32, before the narrowing** -- a BF16-only comparison has
// 8 mantissa bits and would swallow exactly the reassociation errors that
// matter. `ARC_HC_AB_POISON=1` perturbs one element by 1 ULP and must make
// every comparison fail; if it does not, the comparison is broken, not the
// kernel.

// V4 clamped SwiGLU: 8 launches + 3 H2D memcpys -> 1 kernel.
//
// Replaces verbatim, for the ROUTED experts (moe/experts.rs `swiglu` ->
// `swiglu_clamp`):
//   let gate = gate.to_dtype(F32)?.minimum(limit)?;          // cast + bminimum
//   let up   = up.to_dtype(F32)?.clamp(-limit, limit)?;      // cast + bmaximum + bminimum
//   let act  = gate.apply(&Silu)?;                           // usilu_f32
//   up.mul(&act)?.to_dtype(out_dtype)                        // bmul + cast
//
// and for the SHARED expert (layers.rs `Mlp::gated_act`), whose middle two
// launches are instead one `fused_glu_f32` (mistralrs-quant ops.cu:845-890):
//   T activated = (T)glu_silu(a); output[i] = activated * b[i];
//
// The two spellings are the SAME VALUE, which is why one kernel serves both:
//   * candle `usilu_f32` is `silu_fwd(x) = x / (static_cast<T>(1) + expg(-x))`
//     (candle-kernels unary.cu:59-61, cuda_utils.cuh:148 `expg(float)=expf`);
//   * `glu_silu(x)` is `x / (1.0f + expf(-x))` -- character for character the
//     same expression;
//   * the products differ only in operand order (`up * act` vs `act * up`),
//     and IEEE multiplication is commutative bit-for-bit.
//
// `clamp(lo, hi)` is `maximum(lo)?.minimum(hi)` (candle tensor.rs:1176-1178),
// and candle's `maxg`/`ming` for float are `fmaxf`/`fminf`
// (cuda_utils.cuh:142-145) -- reproduced, not re-derived. The clamp bounds
// arrive as f64 in Rust and are narrowed to f32 by candle's `to_dtype` before
// the compare; `limit` is passed in already narrowed, so `-limit` here is the
// same f32 the eager path compares against.
//
// TIN/TOUT are separate so `ARC_SEAM_AB=1` can ask for the F32 product that
// the eager chain holds *before* its final cast, and compare there.
template <typename TIN, typename TOUT>
__global__ void arc_seam_swiglu_clamp_kernel(
    const TIN *__restrict__ gate, // [total] model dtype
    const TIN *__restrict__ up,   // [total] model dtype
    TOUT *__restrict__ out,       // [total]
    float limit,
    long long total
) {
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    const float neg_limit = -limit;
    for (long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < total; i += stride) {
        // BF16->F32 is an exact widening: these are the bits candle's
        // `to_dtype(F32)` would have written to a temporary.
        float g = hc_widen(gate[i]);
        float u = hc_widen(up[i]);
        g = fminf(g, limit);      // gate is clamped ONLY from above
        u = fmaxf(u, neg_limit);  // up is clamped on both sides, max first
        u = fminf(u, limit);
        // silu_fwd: one add, one accurate-libdevice expf, one div.rn.f32.
        // There is no mul+add pair here, so --fmad=false has nothing to
        // forbid and the result does not depend on the contraction setting.
        const float den = __fadd_rn(1.0f, expf(-g));
        const float act = __fdiv_rn(g, den);
        out[i] = hc_narrow<TOUT>(__fmul_rn(act, u));
    }
}

// The clamp HALF of the same stage: 5 launches + 3 H2D memcpys -> 1 kernel,
// two outputs, both F32.
//
// Why a second, weaker kernel exists. The SHARED expert's activation and
// multiply are `mistralrs_quant::fused_glu`, and mistralrs-quant/build.rs:281
// compiles that translation unit with `--use_fast_math`. Its `expf` is
// therefore `__expf` (ex2.approx) and its `/` is div.approx, not the
// `__nv_expf` + div.rn that candle-kernels' `usilu_f32` uses -- and fast math
// also implies -ftz=true, which no intrinsic in a non-fast-math file can
// reproduce per-op. So `arc_seam_swiglu_clamp_kernel` above is bit-identical to
// the ROUTED chain and would NOT be bit-identical to the shared one. Rather
// than fuse the shared site on a tolerance argument, only its provable half is
// fused: the two widening casts, the `maximum` and the two `minimum`s, plus the
// three host-to-device copies of the bounds. `fused_glu` then runs unchanged on
// the F32 tensors it runs on today.
//
// `ARC_SEAM_AB=1` still evaluates the full fusion at the shared site and
// reports whether the fast-math and IEEE spellings happen to agree; that
// measurement decides whether the remaining 4 launches per shared call can ever
// be taken, and it is a fact in the log rather than an assumption either way.
template <typename TIN>
__global__ void arc_seam_swiglu_clamp_split_kernel(
    const TIN *__restrict__ gate,   // [total] model dtype
    const TIN *__restrict__ up,     // [total] model dtype
    float *__restrict__ gate_out,   // [total] F32, min(g, limit)
    float *__restrict__ up_out,     // [total] F32, clamp(u, -limit, limit)
    float limit,
    long long total
) {
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    const float neg_limit = -limit;
    for (long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const float g = hc_widen(gate[i]);
        float u = hc_widen(up[i]);
        u = fmaxf(u, neg_limit);
        u = fminf(u, limit);
        gate_out[i] = fminf(g, limit);
        up_out[i] = u;
    }
}

// MoE expert combine: 4 launches -> 1.
//
// Replaces verbatim (moe/experts.rs `forward_fast`, the `experts.weighted_sum`
// span):
//   ys.to_dtype(F32)?                                   // cast_bf16_f32
//     .broadcast_mul(&topk_weights.unsqueeze(-1)?)?     // bmul_f32
//     .sum(D::Minus2)?                                  // fast_sum_f32
//     .to_dtype(original_dtype)                         // cast_f32_bf16
//
// REDUCTION ORDER. `sum(D::Minus2)` on `[n, K, h]` gives dst_el = n*h and
// el_to_sum_per_block = K, so candle's `FastReduce` picks
// block_dim = min(1024, K).next_power_of_two() = P (cuda_backend/mod.rs:370).
// For V4's K = 6 that is P = 8: threads 0..5 each take one element, threads
// 6 and 7 take none and keep their zero-initialised accumulator, and the
// pairwise tree then runs over all EIGHT slots. The sum is therefore
//   ((0+a0)+(0+a4)) + ((0+a2)+0) + [ ((0+a1)+(0+a5)) + ((0+a3)+0) ]
// which is neither the sequential sum nor the K=6 pairwise tree. The identity
// padding is part of the answer and is replayed literally below, as is the
// `shr[tid] = 0; shr[tid] += v` (it canonicalises -0.0f to +0.0f).
template <typename TIN, typename TOUT, int K, int P>
__global__ void arc_seam_moe_weighted_sum_kernel(
    const TIN *__restrict__ ys,  // [n, K, h] model dtype
    const float *__restrict__ w, // [n, K]    F32 routing weights
    TOUT *__restrict__ out,      // [n, h]
    int h,
    long long total // n * h
) {
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    for (long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const long long row = i / (long long)h;
        const int col = (int)(i - row * (long long)h);
        const float *wrow = w + row * K;
        const TIN *yrow = ys + row * (long long)K * (long long)h + col;

        float acc[P];
#pragma unroll
        for (int t = 0; t < P; ++t) {
            if (t < K) {
                // `broadcast_mul` is binary.cu `x * y` with x = ys, y = w.
                acc[t] = __fadd_rn(
                    0.0f,
                    __fmul_rn(hc_widen(yrow[(long long)t * (long long)h]), wrow[t]));
            } else {
                acc[t] = 0.0f; // identity-padded lane: shr[tid] = 0, never added to
            }
        }
#pragma unroll
        for (int s = P >> 1; s > 0; s >>= 1) {
#pragma unroll
            for (int t = 0; t < P; ++t) {
                if (t < s) {
                    acc[t] = __fadd_rn(acc[t], acc[t + s]);
                }
            }
        }
        out[i] = hc_narrow<TOUT>(acc[0]);
    }
}

// MoE gate weight renormalise + scale: 4 launches -> 1.
//
// Replaces verbatim (deepseek4.rs `MoeGate::forward`, spans `gate.renormalize`):
//   let denominator = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;  // fast_sum + affine
//   topk_weight = topk_weight.broadcast_div(&denominator)?;            // bdiv_f32
//   topk_weight = (topk_weight * routed_scaling_factor)?;              // affine
//
// `sum_keepdim(D::Minus1)` on `[n, K]` gives el_to_sum_per_block = K, so the
// same P = next_pow2(K) padded tree as above applies.
//
// `Tensor + f64` and `Tensor * f64` both lower to `affine(mul, add)`
// (candle tensor.rs `bin_trait!`: Add -> affine(1., v), Mul -> affine(v, 0.)),
// whose kernel body is `x * mul + add` (affine.cu:18) compiled by candle with
// the DEFAULT -fmad=true, i.e. a single fmaf. Written as an explicit `fmaf`
// here: an explicit fmaf is unaffected by this file's --fmad=false, which only
// forbids *contraction* of a separate mul and add. `fmaf(x, scale, 0.0f)` is
// deliberately not `x * scale` -- they differ in the sign of zero.
template <int K, int P>
__global__ void arc_seam_gate_renorm_kernel(
    const float *__restrict__ w, // [n, K]
    float *__restrict__ out,     // [n, K]
    float eps,                   // the 1e-20 added to the denominator
    float scale,                 // routed_scaling_factor
    int do_renorm,               // cfg.norm_topk_prob && scoring in {sigmoid, sqrtsoftplus}
    long long n
) {
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    for (long long row = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
         row < n; row += stride) {
        const float *wrow = w + row * K;
        float *orow = out + row * K;

        float v[K];
#pragma unroll
        for (int t = 0; t < K; ++t) {
            v[t] = wrow[t];
        }

        float den = 0.0f;
        if (do_renorm) {
            float acc[P];
#pragma unroll
            for (int t = 0; t < P; ++t) {
                acc[t] = (t < K) ? __fadd_rn(0.0f, v[t]) : 0.0f;
            }
#pragma unroll
            for (int s = P >> 1; s > 0; s >>= 1) {
#pragma unroll
                for (int t = 0; t < P; ++t) {
                    if (t < s) {
                        acc[t] = __fadd_rn(acc[t], acc[t + s]);
                    }
                }
            }
            den = fmaf(acc[0], 1.0f, eps); // affine(1., 1e-20)
        }

#pragma unroll
        for (int t = 0; t < K; ++t) {
            const float x = do_renorm ? __fdiv_rn(v[t], den) : v[t];
            orow[t] = fmaf(x, scale, 0.0f); // affine(routed_scaling_factor, 0.)
        }
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

// ---------------------------------------------------------------------------
// F32-seam entry points.
//
// Deliberately fresh symbol names. A stale object file left in the archive by
// an incremental build has already, once, satisfied a link with none of the
// new code in it -- byte-identical output, green build, silent no-op. A name
// that has never existed before turns that failure into a link error.
// ---------------------------------------------------------------------------

// `out = narrow( silu(min(gate, limit)) * clamp(up, -limit, limit) )`.
// `in_dtype` is the dtype of gate/up; `out_dtype` is usually the same, and is
// F32 only when the A/B harness asks for the pre-narrowing product.
int arc_seam_swiglu_clamp(
    const void *gate,
    const void *up,
    void *out,
    float limit,
    long long total,
    int in_dtype,
    int out_dtype,
    long long stream
) {
    unsigned blocks;
    int block;
    hc_launch_dims(total, blocks, block);
    cudaStream_t s = (cudaStream_t)stream;
#define ARC_SWG(TIN, TOUT)                                                                 \
    arc_seam_swiglu_clamp_kernel<TIN, TOUT><<<blocks, block, 0, s>>>(                       \
        (const TIN *)gate, (const TIN *)up, (TOUT *)out, limit, total)
    if (in_dtype == HC_DTYPE_BF16 && out_dtype == HC_DTYPE_BF16) {
        ARC_SWG(__nv_bfloat16, __nv_bfloat16);
        return 0;
    }
    if (in_dtype == HC_DTYPE_BF16 && out_dtype == HC_DTYPE_F32) {
        ARC_SWG(__nv_bfloat16, float);
        return 0;
    }
    if (in_dtype == HC_DTYPE_F32 && out_dtype == HC_DTYPE_F32) {
        ARC_SWG(float, float);
        return 0;
    }
    if (in_dtype == HC_DTYPE_F32 && out_dtype == HC_DTYPE_BF16) {
        ARC_SWG(float, __nv_bfloat16);
        return 0;
    }
    return 1;
#undef ARC_SWG
}

// `gate_out = min(gate, limit)`, `up_out = clamp(up, -limit, limit)`, both F32.
int arc_seam_swiglu_clamp_split(
    const void *gate,
    const void *up,
    void *gate_out,
    void *up_out,
    float limit,
    long long total,
    int in_dtype,
    long long stream
) {
    unsigned blocks;
    int block;
    hc_launch_dims(total, blocks, block);
    cudaStream_t s = (cudaStream_t)stream;
#define ARC_SWG_SPLIT(TIN)                                                                 \
    arc_seam_swiglu_clamp_split_kernel<TIN><<<blocks, block, 0, s>>>(                       \
        (const TIN *)gate, (const TIN *)up, (float *)gate_out, (float *)up_out, limit, total)
    if (in_dtype == HC_DTYPE_BF16) {
        ARC_SWG_SPLIT(__nv_bfloat16);
        return 0;
    }
    if (in_dtype == HC_DTYPE_F32) {
        ARC_SWG_SPLIT(float);
        return 0;
    }
    return 1;
#undef ARC_SWG_SPLIT
}

// `out[n, h] = narrow( tree_sum_K( ys[n, j, h] * w[n, j] ) )`.
int arc_seam_moe_weighted_sum(
    const void *ys,
    const void *w,
    void *out,
    int k,
    int h,
    long long total,
    int in_dtype,
    int out_dtype,
    long long stream
) {
    unsigned blocks;
    int block;
    hc_launch_dims(total, blocks, block);
    cudaStream_t s = (cudaStream_t)stream;
#define ARC_WS(TIN, TOUT, KV, PV)                                                          \
    arc_seam_moe_weighted_sum_kernel<TIN, TOUT, KV, PV><<<blocks, block, 0, s>>>(           \
        (const TIN *)ys, (const float *)w, (TOUT *)out, h, total)
#define ARC_WS_K(TIN, TOUT)                                                                \
    switch (k) {                                                                           \
    case 1: ARC_WS(TIN, TOUT, 1, 1); return 0;                                             \
    case 2: ARC_WS(TIN, TOUT, 2, 2); return 0;                                             \
    case 4: ARC_WS(TIN, TOUT, 4, 4); return 0;                                             \
    case 6: ARC_WS(TIN, TOUT, 6, 8); return 0;                                             \
    case 8: ARC_WS(TIN, TOUT, 8, 8); return 0;                                             \
    default: return 1;                                                                     \
    }
    if (in_dtype == HC_DTYPE_BF16 && out_dtype == HC_DTYPE_BF16) {
        ARC_WS_K(__nv_bfloat16, __nv_bfloat16)
    } else if (in_dtype == HC_DTYPE_BF16 && out_dtype == HC_DTYPE_F32) {
        ARC_WS_K(__nv_bfloat16, float)
    } else if (in_dtype == HC_DTYPE_F32 && out_dtype == HC_DTYPE_F32) {
        ARC_WS_K(float, float)
    } else if (in_dtype == HC_DTYPE_F32 && out_dtype == HC_DTYPE_BF16) {
        ARC_WS_K(float, __nv_bfloat16)
    }
    return 1;
#undef ARC_WS_K
#undef ARC_WS
}

// `out[n, :] = affine( w[n, :] / (tree_sum_K(w[n, :]) + eps), scale )`.
int arc_seam_gate_renorm(
    const void *w,
    void *out,
    float eps,
    float scale,
    int do_renorm,
    int k,
    long long n,
    long long stream
) {
    unsigned blocks;
    int block;
    hc_launch_dims(n, blocks, block);
    cudaStream_t s = (cudaStream_t)stream;
#define ARC_GR(KV, PV)                                                                     \
    arc_seam_gate_renorm_kernel<KV, PV><<<blocks, block, 0, s>>>(                           \
        (const float *)w, (float *)out, eps, scale, do_renorm, n)
    switch (k) {
    case 1: ARC_GR(1, 1); return 0;
    case 2: ARC_GR(2, 2); return 0;
    case 4: ARC_GR(4, 4); return 0;
    case 6: ARC_GR(6, 8); return 0;
    case 8: ARC_GR(8, 8); return 0;
    default: return 1;
    }
#undef ARC_GR
}

} // extern "C"
