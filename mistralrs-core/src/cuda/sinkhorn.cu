// Fused Sinkhorn normalization for V4 mHC (manifold-constrained hyper-connections).
//
// RUN-161 throughput: the eager candle path runs the 20-iteration Sinkhorn as
// ~123 separate tensor ops (each a kernel launch + allocation) on a tiny
// [N, hc, hc] = [1, 4, 4] matrix, twice per layer x 43 layers = ~13,000 serial
// launch-bound micro-kernels per decode token. This kernel collapses the entire
// normalization into ONE launch per call: one block per matrix, `hc` threads
// (one per row), the [hc, hc] tile lives in shared memory, all iterations run
// in-block with no global round-trips.
//
// ---------------------------------------------------------------------------
// BIT-IDENTITY CONTRACT (fix/sinkhorn-bit-identity)
// ---------------------------------------------------------------------------
// This kernel must produce bit-identical f32 output to the candle op chain in
// `sinkhorn_normalize` (models/dsv4_mhc.rs). The first version failed the H200
// A/B (ppl 58.85084 vs 58.88946; 4/6 greedy divergences) for two reasons, both
// fixed here BY CONSTRUCTION. Do not "optimize" any of this away:
//
// 1. Reduction ORDER. candle's `sum_keepdim` / `max_keepdim` on CUDA run
//    candle-kernels/src/reduce.cu `fast_sum` / `fast_max` with
//    block_dim = next_pow2(reduced_len) (cuda_backend/mod.rs FastReduce:
//    `usize::min(1024, el_to_sum_per_block).next_power_of_two()`), one element
//    per thread (identity-padded: `shr[tid] = 0; shr[tid] += v` for sum,
//    `shr[tid] = -INF; shr[tid] = maxg(shr[tid], v)` for max), then a pairwise
//    tree `shr[t] op= shr[t + s]` for s = block/2, ..., 1.
//    For hc = 4 the row sum is therefore (a0+a2)+(a1+a3), NOT the sequential
//    ((a0+a1)+a2)+a3 the old kernel computed — different f32 rounding.
//    `candle_tree_sum` / `candle_tree_max` below replay that exact order,
//    including the `0.0f + v` load (which canonicalizes -0.0f to +0.0f) and
//    the identity padding for non-power-of-two hc.
//
// 2. Unfused, IEEE round-to-nearest arithmetic. candle-kernels are compiled
//    with plain `-O3` and NO --use_fast_math (candle-kernels/build.rs), so its
//    division is `div.rn.f32` and its `expf` is the accurate libdevice
//    `__nv_expf`. mistralrs-core's build.rs compiles the rest of src/cuda/*.cu
//    with --use_fast_math, which silently rewrites expf -> __expf (hardware
//    approximation), IEEE div -> approximate reciprocal (-prec-div=false), and
//    flushes denormals (-ftz=true). Therefore build.rs compiles THIS FILE in a
//    separate no-fast-math builder (with --fmad=false so no FMA contraction is
//    possible either), and the #error guard below turns any future re-glob
//    under fast math into a build failure instead of silent numeric drift.
//    All arithmetic here additionally uses __fadd_rn/__fsub_rn/__fdiv_rn,
//    which nvcc documents as never merged into FMA.
//    (The candle chain itself contains no contractible mul+add: its only
//    fused candidate is the `+ eps` affine `x * 1.0f + eps`, and
//    fmaf(x, 1.0f, eps) rounds x*1.0+eps once, which is exactly
//    __fadd_rn(x, eps).)
//
// 3. Same op sequence as the reference, eps placement included:
//      1. row softmax: x = exp(c - rowmax) / rowsum ; then x = x + eps
//      2. initial col normalize: x = x / (colsum + eps)
//      3. (iters-1) more passes: row x/(rowsum+eps), then col x/(colsum+eps)
//    col_sum is the sum DOWN a column (over rows), matching candle's
//    sum_keepdim(dim=1).
//
// Scalar Rust replicas of both this kernel and the candle-GPU op chain are
// asserted bit-identical against each other in cuda/sinkhorn.rs `mod tests`.
// The remaining platform gap (documented there): libdevice __nv_expf vs host
// libm expf may differ in the last ulp — irrelevant for the GPU A/B because
// both GPU paths call the same __nv_expf. Final proof is the on-GPU A/B
// (arc-tools/quality: run_sinkhorn_ab.py + run_ppl.sh --sinkhorn-ab).

#include <cuda_runtime.h>
#include <math.h>

#if defined(__USE_FAST_MATH__)
#error "sinkhorn.cu must be compiled WITHOUT --use_fast_math: fast math rewrites expf to __expf and IEEE division to approximate reciprocals, breaking bit-identity with candle-kernels (which build with plain -O3). See the dedicated no-fast-math builder in mistralrs-core/build.rs."
#endif

// hc_mult is 4 for V4-Flash; cap at 16 for safety (shared-mem sized at launch).
#define SINKHORN_MAX_HC 16

// ---------------------------------------------------------------------------
// PERFORMANCE NOTE (why this file is templated on `hc`)
// ---------------------------------------------------------------------------
// The first version of this kernel took `hc` as a RUNTIME argument and held the
// per-thread row / tree buffers in `float r[SINKHORN_MAX_HC]` arrays walked by
// runtime-bounded loops. nvcc cannot keep an array in registers unless every
// index is resolvable at compile time, so all three arrays were demoted to
// LOCAL MEMORY -- `ptxas -v` reported `192 bytes stack frame` (48 floats =
// r[16] + buf[16] + col[16]). With hc = 4 the kernel launches ONE block of FOUR
// threads, so there is no occupancy to hide that latency behind: every one of
// the ~1,800 dependent local-memory round trips per call (20 iterations x two
// tree reductions x two arrays, plus the row read/write) is exposed. Measured
// cost: 30.0 us per call on a [1, 4, 4] tensor -- 2.549 ms/step over the 86
// calls, 28% of the whole router region's GPU time, for 16 floats of work.
//
// Templating on `hc` makes every index a compile-time constant, the arrays
// become registers (`0 bytes stack frame`), and the arithmetic is UNCHANGED --
// same ops, same order, same intrinsics -- so bit-identity is preserved by
// construction. The tree reductions are expressed as template recursion rather
// than `#pragma unroll` loops specifically so that the halving step `S` is a
// constant in the type system and cannot silently fall back to a runtime index.
// ---------------------------------------------------------------------------

namespace {

constexpr int next_pow2_ce(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// One level of candle's pairwise tree, with the stride `S` fixed by the type.
// `buf` is taken by array reference (not pointer) so the indices stay
// compile-time and the storage stays in registers.
template <int P, int N> struct TreeSumLevel {
    __device__ __forceinline__ static void run(float (&buf)[N]) {
        constexpr int S = P / 2;
#pragma unroll
        for (int t = 0; t < S; ++t) {
            buf[t] = __fadd_rn(buf[t], buf[t + S]);
        }
        TreeSumLevel<S, N>::run(buf);
    }
};
template <int N> struct TreeSumLevel<1, N> {
    __device__ __forceinline__ static void run(float (&)[N]) {}
};

template <int P, int N> struct TreeMaxLevel {
    __device__ __forceinline__ static void run(float (&buf)[N]) {
        constexpr int S = P / 2;
#pragma unroll
        for (int t = 0; t < S; ++t) {
            buf[t] = fmaxf(buf[t], buf[t + S]);
        }
        TreeMaxLevel<S, N>::run(buf);
    }
};
template <int N> struct TreeMaxLevel<1, N> {
    __device__ __forceinline__ static void run(float (&)[N]) {}
};

// Replays candle-kernels `fast_sum` (reduce.cu) for reduced_len <= 16:
// zero-initialized accumulators, one element per virtual thread
// (block_dim = next_pow2(len) >= len so each thread loads at most one),
// then the pairwise tree. `__fadd_rn(0.0f, v)` mirrors `shr[tid] = 0;
// shr[tid] += v` (note: turns -0.0f into +0.0f, exactly like candle).
template <int HC> __device__ __forceinline__ float candle_tree_sum(const float (&v)[HC]) {
    constexpr int P = next_pow2_ce(HC);
    float buf[P];
#pragma unroll
    for (int t = 0; t < P; ++t) {
        buf[t] = (t < HC) ? __fadd_rn(0.0f, v[t]) : 0.0f;
    }
    TreeSumLevel<P, P>::run(buf);
    return buf[0];
}

// Replays candle-kernels `fast_max` (reduce.cu): -INF init, maxg == fmaxf
// (NaN-ignoring IEEE maxNum), pairwise tree. Order-insensitive for finite
// inputs but mirrored anyway so NaN propagation matches candle exactly.
template <int HC> __device__ __forceinline__ float candle_tree_max(const float (&v)[HC]) {
    constexpr int P = next_pow2_ce(HC);
    float buf[P];
#pragma unroll
    for (int t = 0; t < P; ++t) {
        buf[t] = (t < HC) ? fmaxf(-INFINITY, v[t]) : -INFINITY;
    }
    TreeMaxLevel<P, P>::run(buf);
    return buf[0];
}

// in/out: [n, hc, hc] row-major F32. One block per matrix `n`, `hc` threads.
// `hc` is a template parameter so the per-thread row / tree buffers stay in
// registers; see the PERFORMANCE NOTE above. The arithmetic is identical to the
// runtime-`hc` version this replaced. The kernel lives inside the anonymous
// namespace because a template cannot have C linkage; only the dispatcher below
// is `extern "C"`, which is all the Rust FFI binds to.
template <int HC>
__global__ void sinkhorn_normalize_f32_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n,
    int iters,
    float eps
) {
    int batch = blockIdx.x;
    int row = threadIdx.x;
    if (batch >= n || row >= HC) return;

    __shared__ float mat[HC * HC];
    __shared__ float csum[HC];

    const float* my_in = in + (size_t)batch * HC * HC + (size_t)row * HC;

    // Each thread owns one row in registers.
    float r[HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = my_in[j];

    // ---- 1. stable row softmax, then + eps ----
    // candle: max_keepdim(-1) -> broadcast_sub -> exp -> sum_keepdim(-1)
    //         -> broadcast_div -> affine(+eps)
    const float m = candle_tree_max<HC>(r);
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = expf(__fsub_rn(r[j], m));
    const float rs = candle_tree_sum<HC>(r);
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = __fadd_rn(__fdiv_rn(r[j], rs), eps);

    // publish row to shared
#pragma unroll
    for (int j = 0; j < HC; ++j) mat[row * HC + j] = r[j];
    __syncthreads();

    // ---- 2. initial column normalize: x / (colsum + eps) ----
    // column `row` sum = tree-sum over rows k of mat[k][row]
    {
        float col[HC];
#pragma unroll
        for (int k = 0; k < HC; ++k) col[k] = mat[k * HC + row];
        csum[row] = __fadd_rn(candle_tree_sum<HC>(col), eps);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = __fdiv_rn(mat[row * HC + j], csum[j]);

    // ---- 3. (iters - 1) more row->col passes ----
    for (int it = 0; it < iters - 1; ++it) {
        // row normalize: x / (rowsum + eps)  (r holds this thread's row)
        const float rsum = __fadd_rn(candle_tree_sum<HC>(r), eps);
#pragma unroll
        for (int j = 0; j < HC; ++j) r[j] = __fdiv_rn(r[j], rsum);
#pragma unroll
        for (int j = 0; j < HC; ++j) mat[row * HC + j] = r[j];
        __syncthreads();

        // column normalize: x / (colsum + eps)
        {
            float col[HC];
#pragma unroll
            for (int k = 0; k < HC; ++k) col[k] = mat[k * HC + row];
            csum[row] = __fadd_rn(candle_tree_sum<HC>(col), eps);
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < HC; ++j) r[j] = __fdiv_rn(mat[row * HC + j], csum[j]);
    }

    // ---- write out ----
    float* my_out = out + (size_t)batch * HC * HC + (size_t)row * HC;
#pragma unroll
    for (int j = 0; j < HC; ++j) my_out[j] = r[j];
}

// Every hc in [1, SINKHORN_MAX_HC] gets its own instantiation, so there is no
// runtime-`hc` fallback whose numerics could drift from the templated path.
// V4-Flash uses hc = 4; the rest are cheap (a few hundred bytes of cubin each)
// and keep the Rust-side contract "hc <= 16 is supported" honest.
#define SINKHORN_LAUNCH(HC_)                                                                 \
    case HC_:                                                                                \
        sinkhorn_normalize_f32_kernel<HC_><<<grid, block, 0, (cudaStream_t)stream>>>(         \
            (const float*)in, (float*)out, n, iters, eps);                                   \
        return;

} // namespace

extern "C" {

void sinkhorn_normalize_f32(
    const void* in,
    void* out,
    int n,
    int hc,
    int iters,
    float eps,
    long long stream
) {
    dim3 grid(n, 1, 1);
    dim3 block(hc, 1, 1);
    switch (hc) {
        SINKHORN_LAUNCH(1)
        SINKHORN_LAUNCH(2)
        SINKHORN_LAUNCH(3)
        SINKHORN_LAUNCH(4)
        SINKHORN_LAUNCH(5)
        SINKHORN_LAUNCH(6)
        SINKHORN_LAUNCH(7)
        SINKHORN_LAUNCH(8)
        SINKHORN_LAUNCH(9)
        SINKHORN_LAUNCH(10)
        SINKHORN_LAUNCH(11)
        SINKHORN_LAUNCH(12)
        SINKHORN_LAUNCH(13)
        SINKHORN_LAUNCH(14)
        SINKHORN_LAUNCH(15)
        SINKHORN_LAUNCH(16)
    default:
        // Unreachable: sinkhorn_normalize_cuda rejects hc > SINKHORN_MAX_HC
        // before calling. Leaving `out` untouched here would be a silent wrong
        // answer, so do nothing and let the caller's guard be the contract.
        return;
    }
}
#undef SINKHORN_LAUNCH

} // extern "C"
