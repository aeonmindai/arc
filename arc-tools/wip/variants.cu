// Sinkhorn variants bench + bit-identity harness.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#if defined(__USE_FAST_MATH__)
#error "must be built without --use_fast_math"
#endif

#define MAXHC 16

// ---------------- compile-time next_pow2 (no constexpr-fn -> no --expt-relaxed-constexpr) ----
template <int V, int P = 1, bool Done = (P >= V)> struct NextPow2 { enum { value = NextPow2<V, P * 2>::value }; };
template <int V, int P> struct NextPow2<V, P, true> { enum { value = P }; };

// ---------------- register tree reductions (candle fast_sum / fast_max order) ----------------
template <int S, int N> struct TSum {
    __device__ __forceinline__ static void run(float (&b)[N]) {
#pragma unroll
        for (int t = 0; t < S; ++t) b[t] = __fadd_rn(b[t], b[t + S]);
        TSum<S / 2, N>::run(b);
    }
};
template <int N> struct TSum<0, N> { __device__ __forceinline__ static void run(float (&)[N]) {} };

template <int S, int N> struct TMax {
    __device__ __forceinline__ static void run(float (&b)[N]) {
#pragma unroll
        for (int t = 0; t < S; ++t) b[t] = fmaxf(b[t], b[t + S]);
        TMax<S / 2, N>::run(b);
    }
};
template <int N> struct TMax<0, N> { __device__ __forceinline__ static void run(float (&)[N]) {} };

template <int HC> __device__ __forceinline__ float ctree_sum(const float (&v)[HC]) {
    const int P = NextPow2<HC>::value;
    float b[NextPow2<HC>::value];
#pragma unroll
    for (int t = 0; t < P; ++t) b[t] = (t < HC) ? __fadd_rn(0.0f, v[t]) : 0.0f;
    TSum<NextPow2<HC>::value / 2, NextPow2<HC>::value>::run(b);
    return b[0];
}
template <int HC> __device__ __forceinline__ float ctree_max(const float (&v)[HC]) {
    const int P = NextPow2<HC>::value;
    float b[NextPow2<HC>::value];
#pragma unroll
    for (int t = 0; t < P; ++t) b[t] = (t < HC) ? fmaxf(-INFINITY, v[t]) : -INFINITY;
    TMax<NextPow2<HC>::value / 2, NextPow2<HC>::value>::run(b);
    return b[0];
}

// ======================= V0: master (runtime hc, local memory) =======================
namespace v0 {
__device__ __forceinline__ int np2(int v) { int p = 1; while (p < v) p <<= 1; return p; }
__device__ __forceinline__ float tsum(const float* v, int len) {
    float b[MAXHC]; const int p = np2(len);
    for (int t = 0; t < p; ++t) b[t] = (t < len) ? __fadd_rn(0.0f, v[t]) : 0.0f;
    for (int s = p >> 1; s > 0; s >>= 1) for (int t = 0; t < s; ++t) b[t] = __fadd_rn(b[t], b[t + s]);
    return b[0];
}
__device__ __forceinline__ float tmax(const float* v, int len) {
    float b[MAXHC]; const int p = np2(len);
    for (int t = 0; t < p; ++t) b[t] = (t < len) ? fmaxf(-INFINITY, v[t]) : -INFINITY;
    for (int s = p >> 1; s > 0; s >>= 1) for (int t = 0; t < s; ++t) b[t] = fmaxf(b[t], b[t + s]);
    return b[0];
}
__global__ void k(const float* __restrict__ in, float* __restrict__ out, int n, int hc, int iters, float eps) {
    int batch = blockIdx.x, row = threadIdx.x;
    if (batch >= n || row >= hc) return;
    extern __shared__ float smem[];
    float* mat = smem; float* csum = smem + hc * hc;
    const float* my_in = in + (size_t)batch * hc * hc + (size_t)row * hc;
    float r[MAXHC];
    for (int j = 0; j < hc; ++j) r[j] = my_in[j];
    const float m = tmax(r, hc);
    for (int j = 0; j < hc; ++j) r[j] = expf(__fsub_rn(r[j], m));
    const float rs = tsum(r, hc);
    for (int j = 0; j < hc; ++j) r[j] = __fadd_rn(__fdiv_rn(r[j], rs), eps);
    for (int j = 0; j < hc; ++j) mat[row * hc + j] = r[j];
    __syncthreads();
    { float col[MAXHC]; for (int k2 = 0; k2 < hc; ++k2) col[k2] = mat[k2 * hc + row];
      csum[row] = __fadd_rn(tsum(col, hc), eps); }
    __syncthreads();
    for (int j = 0; j < hc; ++j) r[j] = __fdiv_rn(mat[row * hc + j], csum[j]);
    for (int it = 0; it < iters - 1; ++it) {
        const float rsum = __fadd_rn(tsum(r, hc), eps);
        for (int j = 0; j < hc; ++j) r[j] = __fdiv_rn(r[j], rsum);
        for (int j = 0; j < hc; ++j) mat[row * hc + j] = r[j];
        __syncthreads();
        { float col[MAXHC]; for (int k2 = 0; k2 < hc; ++k2) col[k2] = mat[k2 * hc + row];
          csum[row] = __fadd_rn(tsum(col, hc), eps); }
        __syncthreads();
        for (int j = 0; j < hc; ++j) r[j] = __fdiv_rn(mat[row * hc + j], csum[j]);
    }
    float* my_out = out + (size_t)batch * hc * hc + (size_t)row * hc;
    for (int j = 0; j < hc; ++j) my_out[j] = r[j];
}
} // v0

// ======================= V1: templated registers, keeps shared + barriers =======================
namespace v1 {
template <int HC>
__global__ void k(const float* __restrict__ in, float* __restrict__ out, int n, int iters, float eps) {
    int batch = blockIdx.x, row = threadIdx.x;
    if (batch >= n || row >= HC) return;
    __shared__ float mat[HC * HC];
    __shared__ float csum[HC];
    const float* my_in = in + (size_t)batch * HC * HC + (size_t)row * HC;
    float r[HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = my_in[j];
    const float m = ctree_max<HC>(r);
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = expf(__fsub_rn(r[j], m));
    const float rs = ctree_sum<HC>(r);
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = __fadd_rn(__fdiv_rn(r[j], rs), eps);
#pragma unroll
    for (int j = 0; j < HC; ++j) mat[row * HC + j] = r[j];
    __syncthreads();
    { float col[HC];
#pragma unroll
      for (int k2 = 0; k2 < HC; ++k2) col[k2] = mat[k2 * HC + row];
      csum[row] = __fadd_rn(ctree_sum<HC>(col), eps); }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = __fdiv_rn(mat[row * HC + j], csum[j]);
    for (int it = 0; it < iters - 1; ++it) {
        const float rsum = __fadd_rn(ctree_sum<HC>(r), eps);
#pragma unroll
        for (int j = 0; j < HC; ++j) r[j] = __fdiv_rn(r[j], rsum);
#pragma unroll
        for (int j = 0; j < HC; ++j) mat[row * HC + j] = r[j];
        __syncthreads();
        { float col[HC];
#pragma unroll
          for (int k2 = 0; k2 < HC; ++k2) col[k2] = mat[k2 * HC + row];
          csum[row] = __fadd_rn(ctree_sum<HC>(col), eps); }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < HC; ++j) r[j] = __fdiv_rn(mat[row * HC + j], csum[j]);
    }
    float* my_out = out + (size_t)batch * HC * HC + (size_t)row * HC;
#pragma unroll
    for (int j = 0; j < HC; ++j) my_out[j] = r[j];
}
} // v1

// ======================= V2: warp-butterfly, NO shared, NO barriers =======================
namespace v2 {
// Column sums across lanes in candle's exact pairwise-tree order.
// Level t: __fadd_rn(buf[t], buf[t+s]); lanes >= s compute the commuted form,
// which is bit-identical (IEEE add is commutative; only associativity differs).
template <int HC, int P>
__device__ __forceinline__ void col_sums(const float (&r)[HC], float (&cs)[HC],
                                         bool active, unsigned mask, float eps) {
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        float v = __fadd_rn(0.0f, active ? r[c] : 0.0f);   // candle identity padding
#pragma unroll
        for (int s = P >> 1; s > 0; s >>= 1) v = __fadd_rn(v, __shfl_xor_sync(mask, v, s));
        cs[c] = __fadd_rn(v, eps);
    }
}
template <int HC>
__global__ void k(const float* __restrict__ in, float* __restrict__ out, int n, int iters, float eps) {
    const int P = NextPow2<HC>::value;
    const int batch = blockIdx.x, row = threadIdx.x;
    if (batch >= n) return;                       // uniform across block
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
    const float m = ctree_max<HC>(r);
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = expf(__fsub_rn(r[j], m));
    const float rs = ctree_sum<HC>(r);
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = __fadd_rn(__fdiv_rn(r[j], rs), eps);
    float cs[HC];
    col_sums<HC, NextPow2<HC>::value>(r, cs, active, mask, eps);
#pragma unroll
    for (int j = 0; j < HC; ++j) r[j] = __fdiv_rn(r[j], cs[j]);
    for (int it = 0; it < iters - 1; ++it) {
        const float rsum = __fadd_rn(ctree_sum<HC>(r), eps);
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
} // v2

// ======================= driver =======================
static void check(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "CUDA ERROR at %s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 1;
    const int hc = argc > 2 ? atoi(argv[2]) : 4;
    const int iters = argc > 3 ? atoi(argv[3]) : 20;
    const int calls = argc > 4 ? atoi(argv[4]) : 86;
    const int poison = argc > 5 ? atoi(argv[5]) : 0;   // negative control: 1-ULP at index 3
    const float eps = 1e-6f;
    if (hc != 4) { fprintf(stderr, "this harness instantiates HC=4 only\n"); return 2; }

    const size_t elems = (size_t)n * hc * hc;
    std::vector<float> h(elems);
    for (size_t i = 0; i < elems; ++i) h[i] = 0.5f * sinf(0.37f * (float)i) + 0.1f * (float)(i % 7);
    if (poison) {  // single-element 1-ULP perturbation — cannot cancel
        unsigned u; memcpy(&u, &h[3], 4); u += 1u; memcpy(&h[3], &u, 4);
    }

    float *d_in, *d_out; cudaMalloc(&d_in, elems * 4); cudaMalloc(&d_out, elems * 4);
    cudaMemcpy(d_in, h.data(), elems * 4, cudaMemcpyHostToDevice);

    const size_t shm = (size_t)(hc * hc + hc) * sizeof(float);
    const int P = 4;
    struct Res { const char* name; double us; unsigned bits[16]; };
    Res res[3] = {{"v0_master", 0, {0}}, {"v1_tmpl_shared", 0, {0}}, {"v2_warp_shuffle", 0, {0}}};

    for (int variant = 0; variant < 3; ++variant) {
        auto launch = [&]() {
            if (variant == 0) v0::k<<<n, hc, shm>>>(d_in, d_out, n, hc, iters, eps);
            else if (variant == 1) v1::k<4><<<n, hc, 0>>>(d_in, d_out, n, iters, eps);
            else v2::k<4><<<n, P, 0>>>(d_in, d_out, n, iters, eps);
        };
        for (int i = 0; i < 300; ++i) launch();
        cudaDeviceSynchronize(); check(res[variant].name);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        const int reps = 200;
        cudaEventRecord(a);
        for (int r2 = 0; r2 < reps; ++r2) for (int c = 0; c < calls; ++c) launch();
        cudaEventRecord(b); cudaDeviceSynchronize(); check("timing");
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        res[variant].us = (double)ms * 1000.0 / (reps * (double)calls);
        std::vector<float> o(elems);
        cudaMemcpy(o.data(), d_out, elems * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 16; ++i) memcpy(&res[variant].bits[i], &o[i], 4);
    }

    printf("n=%d hc=%d iters=%d calls/token=%d poison=%d\n", n, hc, iters, calls, poison);
    for (int v = 0; v < 3; ++v) {
        printf("%-16s  %8.3f us/call  %8.4f ms/token  speedup_vs_master %5.2fx  BITS",
               res[v].name, res[v].us, res[v].us * calls / 1000.0, res[0].us / res[v].us);
        for (int i = 0; i < 16; ++i) printf(" %08x", res[v].bits[i]);
        printf("\n");
    }
    int ok1 = memcmp(res[0].bits, res[1].bits, 64) == 0;
    int ok2 = memcmp(res[0].bits, res[2].bits, 64) == 0;
    printf("BITIDENTICAL v1_vs_master=%s v2_vs_master=%s\n", ok1 ? "YES" : "NO", ok2 ? "YES" : "NO");
    return (ok1 && ok2) ? 0 : 1;
}
