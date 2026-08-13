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

namespace {

__device__ __forceinline__ int next_pow2_le16(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// Replays candle-kernels `fast_sum` (reduce.cu) for reduced_len <= 16:
// zero-initialized accumulators, one element per virtual thread
// (block_dim = next_pow2(len) >= len so each thread loads at most one),
// then the pairwise tree. `__fadd_rn(0.0f, v)` mirrors `shr[tid] = 0;
// shr[tid] += v` (note: turns -0.0f into +0.0f, exactly like candle).
__device__ __forceinline__ float candle_tree_sum(const float* v, int len) {
    float buf[SINKHORN_MAX_HC];
    const int p = next_pow2_le16(len);
    for (int t = 0; t < p; ++t) {
        buf[t] = (t < len) ? __fadd_rn(0.0f, v[t]) : 0.0f;
    }
    for (int s = p >> 1; s > 0; s >>= 1) {
        for (int t = 0; t < s; ++t) {
            buf[t] = __fadd_rn(buf[t], buf[t + s]);
        }
    }
    return buf[0];
}

// Replays candle-kernels `fast_max` (reduce.cu): -INF init, maxg == fmaxf
// (NaN-ignoring IEEE maxNum), pairwise tree. Order-insensitive for finite
// inputs but mirrored anyway so NaN propagation matches candle exactly.
__device__ __forceinline__ float candle_tree_max(const float* v, int len) {
    float buf[SINKHORN_MAX_HC];
    const int p = next_pow2_le16(len);
    for (int t = 0; t < p; ++t) {
        buf[t] = (t < len) ? fmaxf(-INFINITY, v[t]) : -INFINITY;
    }
    for (int s = p >> 1; s > 0; s >>= 1) {
        for (int t = 0; t < s; ++t) {
            buf[t] = fmaxf(buf[t], buf[t + s]);
        }
    }
    return buf[0];
}

} // namespace

extern "C" {

// in/out: [n, hc, hc] row-major F32. One block per matrix `n`, `hc` threads.
// Shared memory: hc*hc (matrix tile) + hc (column sums) floats.
__global__ void sinkhorn_normalize_f32_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n,
    int hc,
    int iters,
    float eps
) {
    int batch = blockIdx.x;
    int row = threadIdx.x;
    if (batch >= n || row >= hc) return;

    extern __shared__ float smem[];
    float* mat = smem;            // [hc * hc]
    float* csum = smem + hc * hc; // [hc]

    const float* my_in = in + (size_t)batch * hc * hc + (size_t)row * hc;

    // Each thread owns one row in registers.
    float r[SINKHORN_MAX_HC];
    for (int j = 0; j < hc; ++j) r[j] = my_in[j];

    // ---- 1. stable row softmax, then + eps ----
    // candle: max_keepdim(-1) -> broadcast_sub -> exp -> sum_keepdim(-1)
    //         -> broadcast_div -> affine(+eps)
    const float m = candle_tree_max(r, hc);
    for (int j = 0; j < hc; ++j) r[j] = expf(__fsub_rn(r[j], m));
    const float rs = candle_tree_sum(r, hc);
    for (int j = 0; j < hc; ++j) r[j] = __fadd_rn(__fdiv_rn(r[j], rs), eps);

    // publish row to shared
    for (int j = 0; j < hc; ++j) mat[row * hc + j] = r[j];
    __syncthreads();

    // ---- 2. initial column normalize: x / (colsum + eps) ----
    // column `row` sum = tree-sum over rows k of mat[k][row]
    {
        float col[SINKHORN_MAX_HC];
        for (int k = 0; k < hc; ++k) col[k] = mat[k * hc + row];
        csum[row] = __fadd_rn(candle_tree_sum(col, hc), eps);
    }
    __syncthreads();
    for (int j = 0; j < hc; ++j) r[j] = __fdiv_rn(mat[row * hc + j], csum[j]);

    // ---- 3. (iters - 1) more row->col passes ----
    for (int it = 0; it < iters - 1; ++it) {
        // row normalize: x / (rowsum + eps)  (r holds this thread's row)
        const float rsum = __fadd_rn(candle_tree_sum(r, hc), eps);
        for (int j = 0; j < hc; ++j) r[j] = __fdiv_rn(r[j], rsum);
        for (int j = 0; j < hc; ++j) mat[row * hc + j] = r[j];
        __syncthreads();

        // column normalize: x / (colsum + eps)
        {
            float col[SINKHORN_MAX_HC];
            for (int k = 0; k < hc; ++k) col[k] = mat[k * hc + row];
            csum[row] = __fadd_rn(candle_tree_sum(col, hc), eps);
        }
        __syncthreads();
        for (int j = 0; j < hc; ++j) r[j] = __fdiv_rn(mat[row * hc + j], csum[j]);
    }

    // ---- write out ----
    float* my_out = out + (size_t)batch * hc * hc + (size_t)row * hc;
    for (int j = 0; j < hc; ++j) my_out[j] = r[j];
}

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
    size_t shmem = (size_t)(hc * hc + hc) * sizeof(float);
    sinkhorn_normalize_f32_kernel<<<grid, block, shmem, (cudaStream_t)stream>>>(
        (const float*)in, (float*)out, n, hc, iters, eps);
}

} // extern "C"
