// Parent system: ArcInfer / ArcMoE
//
// Fused router-region kernels for DeepSeek-V4 (mHC pre-step + MoE gate scoring).
//
// ---------------------------------------------------------------------------
// WHY
// ---------------------------------------------------------------------------
// At b=1 the V4 decode step is launch-bound, not bandwidth-bound: the measured
// step issues ~7.9k `cuLaunchKernel` calls, of which the router region alone is
// 3,259 (41.8%) spread over 86 contiguous spans -- exactly two per layer
// (`mhc_attn_pre` and `mhc_ffn_pre`, 43 layers). Those spans operate on
// `[1, 24]` / `[1, 4, 4]` / `[1, 256]` tensors: mean kernel duration 2.79 us,
// 89% of them under 5 us. They are pure launch overhead wearing a kernel
// costume.
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
#include <math.h>

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
