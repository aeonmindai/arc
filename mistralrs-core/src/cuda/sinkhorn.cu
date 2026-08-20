// Parent system: ArcInfer / ArcMoE  (V4 mHC manifold-constrained hyper-connections)
//
// Fused Sinkhorn normalization for V4 mHC.
//
// RUN-161 throughput: the eager candle path runs the 20-iteration Sinkhorn as
// ~123 separate tensor ops (each a kernel launch + allocation) on a tiny
// [N, hc, hc] = [1, 4, 4] matrix, twice per layer x 43 layers = ~13,000 serial
// launch-bound micro-kernels per decode token. This kernel collapses the entire
// normalization into ONE launch per call.
//
// ---------------------------------------------------------------------------
// BIT-IDENTITY CONTRACT (fix/sinkhorn-bit-identity)
// ---------------------------------------------------------------------------
// Every kernel here must produce bit-identical f32 output to the candle op chain
// in `sinkhorn_normalize` (models/dsv4_mhc.rs). The first version failed the
// H200 A/B (ppl 58.85084 vs 58.88946; 4/6 greedy divergences) for two reasons,
// both fixed here BY CONSTRUCTION. Do not "optimize" any of this away:
//
// 1. Reduction ORDER. candle's `sum_keepdim` / `max_keepdim` on CUDA run
//    candle-kernels/src/reduce.cu `fast_sum` / `fast_max` with
//    block_dim = next_pow2(reduced_len) (cuda_backend/mod.rs FastReduce:
//    `usize::min(1024, el_to_sum_per_block).next_power_of_two()`), one element
//    per thread (identity-padded: `shr[tid] = 0; shr[tid] += v` for sum,
//    `shr[tid] = -INF; shr[tid] = maxg(shr[tid], v)` for max), then a pairwise
//    tree `shr[t] op= shr[t + s]` for s = block/2, ..., 1.
//    For hc = 4 the row sum is therefore (a0+a2)+(a1+a3), NOT the sequential
//    ((a0+a1)+a2)+a3 a naive kernel would compute — different f32 rounding.
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
//    possible either), and the guard below turns any future re-glob under fast
//    math into a build failure instead of silent numeric drift.
//    All arithmetic here additionally uses __fadd_rn/__fsub_rn/__fdiv_rn,
//    which nvcc documents as never merged into FMA.
//
// 3. Same op sequence as the reference, eps placement included:
//      1. row softmax: x = exp(c - rowmax) / rowsum ; then x = x + eps
//      2. initial col normalize: x = x / (colsum + eps)
//      3. (iters-1) more passes: row x/(rowsum+eps), then col x/(colsum+eps)
//    col_sum is the sum DOWN a column (over rows), matching candle's
//    sum_keepdim(dim=1).
//
// ---------------------------------------------------------------------------
// PERFORMANCE — why there are two kernels (measured on H200, CUDA 12.4)
// ---------------------------------------------------------------------------
// `legacy` below is the kernel this file shipped with. It takes `hc` as a
// RUNTIME argument and holds the per-thread row / tree buffers in
// `float r[SINKHORN_MAX_HC]` arrays walked by runtime-bounded loops. nvcc
// cannot keep an array in registers unless every index is resolvable at compile
// time, so all three arrays are demoted to LOCAL MEMORY -- `ptxas -v` reports
// `192 bytes stack frame` (48 floats = r[16] + buf[16] + col[16]). With hc = 4
// the kernel launches ONE block of FOUR threads, so there is no second warp to
// hide that latency behind: every one of the ~1,800 dependent local-memory
// round trips per call is fully exposed.
//
//   MEASURED, H200 / nvcc 12.4 / sm_90, n=1 hc=4 iters=20, 86 calls per token:
//     legacy (runtime hc, local memory) : 30.578 us/call = 2.6297 ms/token
//     templated registers + shared/bar  :  9.935 us/call = 0.8544 ms/token  3.08x
//     warp butterfly (this file's default): 7.302 us/call = 0.6280 ms/token  4.19x
//   All three bit-identical; verified against a 1-ULP single-element poison
//   negative control which changes the output of all three identically.
//
// The default `warp` kernel makes every array index a compile-time constant
// (template on `hc`, so `ptxas -v` reports `0 bytes stack frame, 28 registers`)
// AND removes shared memory and both __syncthreads() per iteration by doing the
// column reduction with warp butterfly shuffles. hc <= 16 so all `hc` threads
// are lanes of a single warp and no barrier is needed at all.
//
// The cross-lane butterfly reproduces candle's pairwise tree EXACTLY: at level
// `s`, lane t computes __fadd_rn(buf[t], buf[t + s]) — which is candle's own
// expression for t < s. Lanes t >= s compute the commuted __fadd_rn(buf[t],
// buf[t - s]); IEEE-754 addition is commutative (only associativity fails), so
// every lane ends the level holding bit-identical values and lane 0 holds
// precisely candle's result. Identity padding for non-power-of-two hc is
// preserved by feeding 0.0f from the inactive lanes, exactly as candle's
// `shr[tid] = 0` padding does.
//
// KILL SWITCH: `ARC_NO_SINKHORN_WARP=1` restores the legacy kernel. Default is
// the fast path (house policy: new behaviour ships fast-by-default with a kill
// switch, never opt-in). The two arms are distinguishable in a profile by
// kernel name: legacy is the unmangled extern "C"
// `sinkhorn_normalize_f32_kernel`, the fast path is the mangled template
// `arc_sinkhorn::warp_kernel<N>` — so "which arm ran" is provable from nsys
// output alone and cannot be faked by a stale object file.
//
// Scalar Rust replicas of the kernel and of the candle-GPU op chain are
// asserted bit-identical against each other in cuda/sinkhorn.rs `mod tests`.

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#if defined(__USE_FAST_MATH__)
#error "sinkhorn.cu must be compiled WITHOUT --use_fast_math: fast math rewrites expf to __expf and IEEE division to approximate reciprocals, breaking bit-identity with candle-kernels (which build with plain -O3). See the dedicated no-fast-math builder in mistralrs-core/build.rs."
#endif

// hc_mult is 4 for V4-Flash; cap at 16 for safety (shared-mem sized at launch).
#define SINKHORN_MAX_HC 16

// ===========================================================================
// Legacy kernel — kept reachable via ARC_NO_SINKHORN_WARP=1 as the A/B arm.
// ===========================================================================
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

} // extern "C"

// ===========================================================================
// Fast path — templated registers + warp butterfly, no shared, no barriers.
// ===========================================================================
namespace arc_sinkhorn {

// Compile-time next_pow2 expressed as a template metafunction rather than a
// constexpr function: nvcc rejects calling a constexpr __host__ function from
// __device__ code unless --expt-relaxed-constexpr is passed, and this file must
// build under the plain no-fast-math builder in build.rs with no extra flags.
template <int V, int P = 1, bool Done = (P >= V)> struct NextPow2 {
    enum { value = NextPow2<V, P * 2>::value };
};
template <int V, int P> struct NextPow2<V, P, true> {
    enum { value = P };
};

// One level of candle's pairwise tree with the stride fixed by the type, so the
// indices stay compile-time and the buffer stays in registers. Expressed as
// template recursion rather than a `#pragma unroll` loop specifically so the
// halving step cannot silently fall back to a runtime index (which is what
// demotes the array to local memory).
template <int S, int N> struct TreeSumLevel {
    __device__ __forceinline__ static void run(float (&b)[N]) {
#pragma unroll
        for (int t = 0; t < S; ++t) b[t] = __fadd_rn(b[t], b[t + S]);
        TreeSumLevel<S / 2, N>::run(b);
    }
};
template <int N> struct TreeSumLevel<0, N> {
    __device__ __forceinline__ static void run(float (&)[N]) {}
};

template <int S, int N> struct TreeMaxLevel {
    __device__ __forceinline__ static void run(float (&b)[N]) {
#pragma unroll
        for (int t = 0; t < S; ++t) b[t] = fmaxf(b[t], b[t + S]);
        TreeMaxLevel<S / 2, N>::run(b);
    }
};
template <int N> struct TreeMaxLevel<0, N> {
    __device__ __forceinline__ static void run(float (&)[N]) {}
};

// Register-resident replicas of candle `fast_sum` / `fast_max`, same order and
// same identity padding as the legacy helpers above.
template <int HC> __device__ __forceinline__ float tree_sum(const float (&v)[HC]) {
    float b[NextPow2<HC>::value];
#pragma unroll
    for (int t = 0; t < (int)NextPow2<HC>::value; ++t) {
        b[t] = (t < HC) ? __fadd_rn(0.0f, v[t]) : 0.0f;
    }
    TreeSumLevel<NextPow2<HC>::value / 2, NextPow2<HC>::value>::run(b);
    return b[0];
}

template <int HC> __device__ __forceinline__ float tree_max(const float (&v)[HC]) {
    float b[NextPow2<HC>::value];
#pragma unroll
    for (int t = 0; t < (int)NextPow2<HC>::value; ++t) {
        b[t] = (t < HC) ? fmaxf(-INFINITY, v[t]) : -INFINITY;
    }
    TreeMaxLevel<NextPow2<HC>::value / 2, NextPow2<HC>::value>::run(b);
    return b[0];
}

// Column sums across lanes, in candle's exact pairwise-tree order.
//
// candle level `s` is `buf[t] = __fadd_rn(buf[t], buf[t + s])` for t < s. The
// butterfly gives lane t the value __fadd_rn(buf[t], buf[t ^ s]); for t < s
// that IS candle's expression, and for t >= s it is the commuted form, which is
// bit-identical because IEEE-754 addition is commutative. So after the last
// level every lane holds exactly candle's reduction result.
//
// Inactive lanes (row >= HC, only possible when HC is not a power of two)
// contribute 0.0f, which is precisely candle's `shr[tid] = 0` identity padding.
template <int HC, int P>
__device__ __forceinline__ void col_sums(
    const float (&r)[HC],
    float (&cs)[HC],
    bool active,
    unsigned mask,
    float eps
) {
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        float v = __fadd_rn(0.0f, active ? r[c] : 0.0f);
#pragma unroll
        for (int s = P >> 1; s > 0; s >>= 1) {
            v = __fadd_rn(v, __shfl_xor_sync(mask, v, s));
        }
        cs[c] = __fadd_rn(v, eps);
    }
}

// in/out: [n, HC, HC] row-major F32. One block per matrix, next_pow2(HC)
// threads — all lanes of a single warp, so the column reduction needs no shared
// memory and no __syncthreads().
template <int HC>
__global__ void warp_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n,
    int iters,
    float eps
) {
    const int batch = blockIdx.x;
    const int row = threadIdx.x;
    // Uniform across the block, so it cannot desynchronise the shuffles below.
    if (batch >= n) return;

    const int P = NextPow2<HC>::value;
    const bool active = (row < HC);
    const unsigned mask = (P >= 32) ? 0xffffffffu : ((1u << P) - 1u);

    float r[HC];
    if (active) {
        const float* my_in = in + (size_t)batch * HC * HC + (size_t)row * HC;
#pragma unroll
        for (int j = 0; j < HC; ++j) r[j] = my_in[j];
    } else {
#pragma unroll
        for (int j = 0; j < HC; ++j) r[j] = 0.0f;
    }

    // ---- 1. stable row softmax, then + eps (entirely within this lane) ----
    const float m = tree_max<HC>(r);
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = expf(__fsub_rn(r[j], m));
    const float rs = tree_sum<HC>(r);
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = __fadd_rn(__fdiv_rn(r[j], rs), eps);

    // ---- 2. initial column normalize: x / (colsum + eps) ----
    float cs[HC];
    col_sums<HC, NextPow2<HC>::value>(r, cs, active, mask, eps);
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = __fdiv_rn(r[j], cs[j]);

    // ---- 3. (iters - 1) more row->col passes ----
    for (int it = 0; it < iters - 1; ++it) {
        const float rsum = __fadd_rn(tree_sum<HC>(r), eps);
#pragma unroll
        for (int j = 0; j < HC; ++j) r[j] = __fdiv_rn(r[j], rsum);
        col_sums<HC, NextPow2<HC>::value>(r, cs, active, mask, eps);
#pragma unroll
        for (int j = 0; j < HC; ++j) r[j] = __fdiv_rn(r[j], cs[j]);
    }

    if (active) {
        float* my_out = out + (size_t)batch * HC * HC + (size_t)row * HC;
#pragma unroll
        for (int j = 0; j < HC; ++j) my_out[j] = r[j];
    }
}

// Read once; std::getenv is not cheap and this is on the per-layer decode path.
// Default = fast path. `ARC_NO_SINKHORN_WARP=1` restores the legacy kernel.
inline bool warp_disabled() {
    static const bool disabled = [] {
        const char* v = getenv("ARC_NO_SINKHORN_WARP");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return disabled;
}

} // namespace arc_sinkhorn

extern "C" {

// Every hc in [1, SINKHORN_MAX_HC] gets its own instantiation, so there is no
// runtime-`hc` fallback on the fast path whose numerics could drift from the
// templated one. V4-Flash uses hc = 4; the rest are cheap (a few hundred bytes
// of cubin each) and keep the Rust-side contract "hc <= 16 is supported" honest.
#define SINKHORN_WARP_LAUNCH(HC_)                                                              \
    case HC_:                                                                                  \
        arc_sinkhorn::warp_kernel<HC_>                                                         \
            <<<dim3(n, 1, 1), dim3(arc_sinkhorn::NextPow2<HC_>::value, 1, 1), 0,               \
               (cudaStream_t)stream>>>((const float*)in, (float*)out, n, iters, eps);          \
        return;

void sinkhorn_normalize_f32(
    const void* in,
    void* out,
    int n,
    int hc,
    int iters,
    float eps,
    long long stream
) {
    if (!arc_sinkhorn::warp_disabled()) {
        switch (hc) {
            SINKHORN_WARP_LAUNCH(1)
            SINKHORN_WARP_LAUNCH(2)
            SINKHORN_WARP_LAUNCH(3)
            SINKHORN_WARP_LAUNCH(4)
            SINKHORN_WARP_LAUNCH(5)
            SINKHORN_WARP_LAUNCH(6)
            SINKHORN_WARP_LAUNCH(7)
            SINKHORN_WARP_LAUNCH(8)
            SINKHORN_WARP_LAUNCH(9)
            SINKHORN_WARP_LAUNCH(10)
            SINKHORN_WARP_LAUNCH(11)
            SINKHORN_WARP_LAUNCH(12)
            SINKHORN_WARP_LAUNCH(13)
            SINKHORN_WARP_LAUNCH(14)
            SINKHORN_WARP_LAUNCH(15)
            SINKHORN_WARP_LAUNCH(16)
        default:
            // Unreachable: sinkhorn_normalize_cuda rejects hc > SINKHORN_MAX_HC
            // before calling. Fall through to the legacy kernel rather than
            // leaving `out` untouched, which would be a silent wrong answer.
            break;
        }
    }

    dim3 grid(n, 1, 1);
    dim3 block(hc, 1, 1);
    size_t shmem = (size_t)(hc * hc + hc) * sizeof(float);
    sinkhorn_normalize_f32_kernel<<<grid, block, shmem, (cudaStream_t)stream>>>(
        (const float*)in, (float*)out, n, hc, iters, eps);
}
#undef SINKHORN_WARP_LAUNCH

} // extern "C"
