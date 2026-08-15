// Prices the computed codebook against the stored Gaussian LUT on BOTH sides
// of the pipeline, in one process, on one box.
//
// The 512 KiB table is read by five K=4/V=2 kernels: the beam search and the
// exhaustive DP that BAKE, and the dequantize / fused GEMV / gather GEMV that
// SERVE. wave19-AP measured the bake half (1.81x for the quality-negative
// `split` construction; ~1.68x projected for the shippable `sum2`). Nobody had
// measured the decode half at all, which is the half the fleet claim sells.
//
// Every kernel takes the same selector: `cb_mult == 0` gathers from the table,
// nonzero computes `sum2` in registers with that MCG multiplier.
//
// PARITY BY CONSTRUCTION: the computed arm is compared against the LUT arm run
// with the COMPUTED TABLE UPLOADED. Both then see bit-identical reproduction
// values, so any output difference is a bug in the in-register decode and not a
// property of the codebook. That makes the gate byte identity, not cosine.
//
// Build (kernels are separate TUs — each has its own anonymous namespace):
//
//   nvcc -O3 -std=c++17 --use_fast_math -arch=sm_80 -I<kernels/qtip> \
//        codebook_bench.cu qtip_beam.cu qtip_quantize.cu qtip_dequantize.cu \
//        qtip_gemv.cu qtip_gather_gemv.cu -o codebook_bench
//
// Usage: ./codebook_bench [rows] [k_in] [beam_w] [iters]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CK(x)                                                                  \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            printf("CUDA ERROR %s at %d: %s\n", #x, __LINE__,                  \
                   cudaGetErrorString(e_));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

extern "C" {
int  qtip_beam_max_width();
int  launch_qtip_quantize_rows_beam_f32(const float*, const float*, const float*,
                                        uint8_t*, uint32_t*, int, int, int, int, int,
                                        unsigned int, cudaStream_t);
void launch_qtip_compute_row_scales_f32(const float*, float*, int, int, float, cudaStream_t);
void launch_qtip_dequantize_v2_k4_l16_f32(const uint8_t*, const float*, const float*,
                                          float*, int, int, int, unsigned int, cudaStream_t);
void launch_qtip_fused_gemv_v2_k4_l16_f32(const uint8_t*, const float*, const float*,
                                          const float*, float*, int, int, int,
                                          unsigned int, cudaStream_t);
void launch_qtip_gather_gemv_v2_k4_l16_f32(const uint8_t*, const float*, const float*,
                                           const float*, const uint32_t*, float*,
                                           int, int, int, int, int, unsigned int,
                                           cudaStream_t);
}

// ---------------------------------------------------------------------------
// Host mirrors of the two codebooks. Must match mod.rs bit for bit.
// ---------------------------------------------------------------------------

static const unsigned int MCG_MULT = 0xCAF6A435u;
// 3.0 * 1.225552 (the measured sigma of the sum2 table).
static const float MCG_DIVISOR = 3.676656f;
static const float GAUSS_DIVISOR = 3.0f;

static void gaussian_pair(uint32_t state, float* g0, float* g1) {
    uint64_t z = (uint64_t)state * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    uint32_t hi = (uint32_t)(z >> 32);
    uint32_t lo = (uint32_t)(z & 0xFFFFFFFFu);
    float u1 = ((float)hi + 1.0f) / ((float)UINT32_MAX + 2.0f);
    float u2 = ((float)lo + 1.0f) / ((float)UINT32_MAX + 2.0f);
    float r = sqrtf(-2.0f * logf(u1));
    float th = 2.0f * (float)M_PI * u2;
    *g0 = r * cosf(th);
    *g1 = r * sinf(th);
}

// __half on the host: nvcc gives us __half2float in host code via the
// conversion intrinsics header, but `__ushort_as_half` is device-only, so go
// through a memcpy into __half's storage.
static float half_from_bits(unsigned short bits) {
    __half h;
    memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

static float cb_fold(unsigned int x) {
    unsigned int m = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
    return half_from_bits((unsigned short)(m >> 16)) + half_from_bits((unsigned short)(m & 0xFFFFu));
}

static void mcg_pair(uint32_t state, float* v0, float* v1) {
    unsigned int x0 = state * MCG_MULT;
    unsigned int x1 = x0 * MCG_MULT;
    *v0 = cb_fold(x0);
    *v1 = cb_fold(x1);
}

static uint64_t fnv1a(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

struct Timing { float best, mean; };

template <typename F>
static Timing time_it(F&& f, int iters) {
    f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    float best = 1e30f, sum = 0.0f;
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(t0));
        f();
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
        sum += ms;
    }
    CK(cudaEventDestroy(t0));
    CK(cudaEventDestroy(t1));
    return Timing{best, sum / iters};
}

static size_t count_diff_bytes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t d = 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++d;
    return d;
}

static void report_f32(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    size_t bitdiff = 0;
    double maxabs = 0.0, maxrel = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y;
        memcpy(&x, &a[i], 4);
        memcpy(&y, &b[i], 4);
        if (x != y) ++bitdiff;
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > maxabs) maxabs = d;
        double den = fabs((double)a[i]);
        if (den > 1e-12 && d / den > maxrel) maxrel = d / den;
    }
    printf("PARITY %-14s bit_diff=%zu/%zu  max_abs=%.3e  max_rel=%.3e\n",
           what, bitdiff, a.size(), maxabs, maxrel);
}

int main(int argc, char** argv) {
    const int n_rows = argc > 1 ? atoi(argv[1]) : 512;
    const int k_in   = argc > 2 ? atoi(argv[2]) : 7168;
    const int beam_w = argc > 3 ? atoi(argv[3]) : 256;
    const int iters  = argc > 4 ? atoi(argv[4]) : 4;

    const int num_symbols    = k_in / 2;
    const int packed_per_row = num_symbols / 2;

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  sm_%d%d  SMs=%d  beam_max_w=%d\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, qtip_beam_max_width());
    printf("shape:  rows=%d k_in=%d T=%d W=%d iters=%d\n\n",
           n_rows, k_in, num_symbols, beam_w, iters);

    // ---- codebooks --------------------------------------------------------
    std::vector<float> gauss(65536ull * 2), computed(65536ull * 2);
    for (uint32_t s = 0; s < 65536u; ++s) {
        gaussian_pair(s, &gauss[2 * s], &gauss[2 * s + 1]);
        mcg_pair(s, &computed[2 * s], &computed[2 * s + 1]);
    }
    {
        double mean = 0, var = 0;
        for (float v : computed) mean += v;
        mean /= computed.size();
        for (float v : computed) var += (v - mean) * (v - mean);
        var /= computed.size();
        printf("computed codebook: mean=%.6f sigma=%.6f (3*sigma=%.6f, divisor=%.6f)\n\n",
               mean, sqrt(var), 3.0 * sqrt(var), MCG_DIVISOR);
    }

    // ---- weights ----------------------------------------------------------
    std::vector<float> w((size_t)n_rows * k_in);
    for (int r = 0; r < n_rows; ++r) {
        for (int i = 0; i < k_in; ++i) {
            float a, b;
            gaussian_pair((uint32_t)(r * k_in + i) + 0x9E37u, &a, &b);
            w[(size_t)r * k_in + i] = a * 0.5f;
        }
    }
    std::vector<float> x(k_in);
    for (int i = 0; i < k_in; ++i) {
        float a, b;
        gaussian_pair((uint32_t)i + 0x5EEDu, &a, &b);
        x[i] = a;
    }

    // ---- device buffers ---------------------------------------------------
    float *d_w, *d_gauss, *d_computed, *d_scales, *d_x, *d_deq, *d_y;
    uint8_t *d_packed_lut, *d_packed_cb;
    uint32_t *d_trace, *d_indices;
    CK(cudaMalloc(&d_w, w.size() * sizeof(float)));
    CK(cudaMalloc(&d_gauss, gauss.size() * sizeof(float)));
    CK(cudaMalloc(&d_computed, computed.size() * sizeof(float)));
    CK(cudaMalloc(&d_scales, (size_t)n_rows * sizeof(float)));
    CK(cudaMalloc(&d_x, x.size() * sizeof(float)));
    CK(cudaMalloc(&d_deq, (size_t)n_rows * k_in * sizeof(float)));
    CK(cudaMalloc(&d_y, (size_t)n_rows * sizeof(float)));
    CK(cudaMalloc(&d_packed_lut, (size_t)n_rows * packed_per_row));
    CK(cudaMalloc(&d_packed_cb, (size_t)n_rows * packed_per_row));
    CK(cudaMalloc(&d_trace, (size_t)n_rows * num_symbols * beam_w * sizeof(uint32_t)));
    CK(cudaMalloc(&d_indices, sizeof(uint32_t)));
    CK(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_gauss, gauss.data(), gauss.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_computed, computed.data(), computed.size() * sizeof(float),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    {
        uint32_t zero = 0;
        CK(cudaMemcpy(d_indices, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // ---- row scales -------------------------------------------------------
    // BOTH arms use the same scales, so the comparison isolates the codebook.
    // (The divisor difference is a scale-policy question, priced separately by
    // the CPU quality probe.)
    launch_qtip_compute_row_scales_f32(d_w, d_scales, n_rows, k_in, MCG_DIVISOR, 0);
    CK(cudaDeviceSynchronize());
    {
        std::vector<float> got(n_rows);
        CK(cudaMemcpy(got.data(), d_scales, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (int r = 0; r < n_rows; ++r) {
            float mx = 0.0f;
            for (int i = 0; i < k_in; ++i) mx = fmaxf(mx, fabsf(w[(size_t)r * k_in + i]));
            float want = (mx == 0.0f) ? 1.0f : mx / MCG_DIVISOR;
            uint32_t a, b;
            memcpy(&a, &got[r], 4);
            memcpy(&b, &want, 4);
            if (a != b) ++bad;
        }
        printf("PARITY row_scales   bit_diff=%zu/%d (divisor=%.6f)\n\n", bad, n_rows, MCG_DIVISOR);
    }
    (void)GAUSS_DIVISOR;

    // =======================================================================
    // 1. BAKE — the beam kernel
    // =======================================================================
    Timing bake_lut = time_it([&] {
        launch_qtip_quantize_rows_beam_f32(d_w, d_computed, d_scales, d_packed_lut, d_trace,
                                           n_rows, k_in, num_symbols, 0, beam_w, 0u, 0);
    }, iters);
    Timing bake_cb = time_it([&] {
        launch_qtip_quantize_rows_beam_f32(d_w, d_computed, d_scales, d_packed_cb, d_trace,
                                           n_rows, k_in, num_symbols, 0, beam_w, MCG_MULT, 0);
    }, iters);

    std::vector<uint8_t> pk_lut((size_t)n_rows * packed_per_row), pk_cb(pk_lut.size());
    CK(cudaMemcpy(pk_lut.data(), d_packed_lut, pk_lut.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pk_cb.data(), d_packed_cb, pk_cb.size(), cudaMemcpyDeviceToHost));
    size_t nonzero = 0;
    for (uint8_t b : pk_cb) if (b) ++nonzero;

    printf("BAKE beam W=%d\n", beam_w);
    printf("  lut      best=%9.3f ms  mean=%9.3f ms\n", bake_lut.best, bake_lut.mean);
    printf("  computed best=%9.3f ms  mean=%9.3f ms\n", bake_cb.best, bake_cb.mean);
    printf("  SPEEDUP  %.3fx (best)  %.3fx (mean)\n",
           bake_lut.best / bake_cb.best, bake_lut.mean / bake_cb.mean);
    printf("PARITY bake         byte_diff=%zu/%zu  fnv1a=%016llx  nonzero=%zu\n\n",
           count_diff_bytes(pk_lut, pk_cb), pk_cb.size(),
           (unsigned long long)fnv1a(pk_cb.data(), pk_cb.size()),
           nonzero);

    // The packed symbols the decode benchmarks run on are the real ones.
    // =======================================================================
    // 2. DECODE — dequantize
    // =======================================================================
    std::vector<float> deq_lut((size_t)n_rows * k_in), deq_cb(deq_lut.size());
    Timing deq_t_lut = time_it([&] {
        launch_qtip_dequantize_v2_k4_l16_f32(d_packed_cb, d_scales, d_computed, d_deq,
                                             n_rows, packed_per_row, num_symbols, 0u, 0);
    }, iters);
    CK(cudaMemcpy(deq_lut.data(), d_deq, deq_lut.size() * sizeof(float), cudaMemcpyDeviceToHost));
    Timing deq_t_cb = time_it([&] {
        launch_qtip_dequantize_v2_k4_l16_f32(d_packed_cb, d_scales, d_computed, d_deq,
                                             n_rows, packed_per_row, num_symbols, MCG_MULT, 0);
    }, iters);
    CK(cudaMemcpy(deq_cb.data(), d_deq, deq_cb.size() * sizeof(float), cudaMemcpyDeviceToHost));

    printf("DECODE dequantize\n");
    printf("  lut      best=%9.3f ms  computed best=%9.3f ms  SPEEDUP %.3fx\n",
           deq_t_lut.best, deq_t_cb.best, deq_t_lut.best / deq_t_cb.best);
    report_f32("dequantize", deq_lut, deq_cb);
    printf("\n");

    // =======================================================================
    // 3. DECODE — fused GEMV (the b=1 decode path)
    // =======================================================================
    std::vector<float> y_lut(n_rows), y_cb(n_rows);
    Timing gemv_lut = time_it([&] {
        launch_qtip_fused_gemv_v2_k4_l16_f32(d_packed_cb, d_scales, d_computed, d_x, d_y,
                                             n_rows, packed_per_row, num_symbols, 0u, 0);
    }, iters);
    CK(cudaMemcpy(y_lut.data(), d_y, y_lut.size() * sizeof(float), cudaMemcpyDeviceToHost));
    Timing gemv_cb = time_it([&] {
        launch_qtip_fused_gemv_v2_k4_l16_f32(d_packed_cb, d_scales, d_computed, d_x, d_y,
                                             n_rows, packed_per_row, num_symbols, MCG_MULT, 0);
    }, iters);
    CK(cudaMemcpy(y_cb.data(), d_y, y_cb.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double gbps_lut = (double)n_rows * packed_per_row / (gemv_lut.best * 1e-3) / 1e9;
    double gbps_cb  = (double)n_rows * packed_per_row / (gemv_cb.best * 1e-3) / 1e9;
    printf("DECODE fused_gemv (b=1)\n");
    printf("  lut      best=%9.4f ms  (%.1f GB/s of packed weights)\n", gemv_lut.best, gbps_lut);
    printf("  computed best=%9.4f ms  (%.1f GB/s)\n", gemv_cb.best, gbps_cb);
    printf("  SPEEDUP  %.3fx (best)  %.3fx (mean)\n",
           gemv_lut.best / gemv_cb.best, gemv_lut.mean / gemv_cb.mean);
    report_f32("fused_gemv", y_lut, y_cb);
    printf("\n");

    // =======================================================================
    // 4. DECODE — gather GEMV (the MoE decode path)
    //
    // NOTE the `cb_mult == 0` arm here does NOT read the table: RUN-161 already
    // replaced the gather with an in-register Box-Muller Gaussian. So this
    // measurement is transcendental-Gaussian vs sum2, i.e. what the change is
    // ACTUALLY worth on this kernel today — a smaller prize than deleting a
    // gather, and an honest one.
    // =======================================================================
    std::vector<float> g_gauss(n_rows), g_cb(n_rows);
    Timing gg_lut = time_it([&] {
        launch_qtip_gather_gemv_v2_k4_l16_f32(d_packed_cb, d_scales, d_computed, d_x,
                                              d_indices, d_y, n_rows, packed_per_row,
                                              num_symbols, 1, 1, 0u, 0);
    }, iters);
    CK(cudaMemcpy(g_gauss.data(), d_y, g_gauss.size() * sizeof(float), cudaMemcpyDeviceToHost));
    Timing gg_cb = time_it([&] {
        launch_qtip_gather_gemv_v2_k4_l16_f32(d_packed_cb, d_scales, d_computed, d_x,
                                              d_indices, d_y, n_rows, packed_per_row,
                                              num_symbols, 1, 1, MCG_MULT, 0);
    }, iters);
    CK(cudaMemcpy(g_cb.data(), d_y, g_cb.size() * sizeof(float), cudaMemcpyDeviceToHost));

    printf("DECODE gather_gemv (1 pair; cb_mult=0 arm is the RUN-161 computed GAUSSIAN,\n"
           "                    not a LUT gather — see the note in the source)\n");
    printf("  gaussian best=%9.4f ms  computed best=%9.4f ms  SPEEDUP %.3fx\n",
           gg_lut.best, gg_cb.best, gg_lut.best / gg_cb.best);
    // Different codebooks => different outputs. Report only that both are live.
    double s0 = 0, s1 = 0;
    for (int i = 0; i < n_rows; ++i) { s0 += fabs(g_gauss[i]); s1 += fabs(g_cb[i]); }
    printf("  live check: sum|y_gauss|=%.4f  sum|y_sum2|=%.4f (both must be nonzero)\n\n",
           s0, s1);

    // A real parity gate for the sum2 gather path: it must agree with the
    // dequantize-then-dot reference computed from the same packed bytes.
    {
        double maxrel = 0.0;
        for (int r = 0; r < n_rows; ++r) {
            double acc = 0.0;
            for (int i = 0; i < k_in; ++i) acc += (double)deq_cb[(size_t)r * k_in + i] * x[i];
            double d = fabs(acc - (double)g_cb[r]);
            double den = fabs(acc);
            if (den > 1e-9 && d / den > maxrel) maxrel = d / den;
        }
        printf("PARITY gather_sum2  max_rel_vs_dequantized_dot=%.3e "
               "(f32 accumulation order differs; must be ~1e-5)\n\n", maxrel);
    }

    printf("SUMMARY bake=%.3fx dequantize=%.3fx fused_gemv=%.3fx gather_gemv=%.3fx\n",
           bake_lut.best / bake_cb.best, deq_t_lut.best / deq_t_cb.best,
           gemv_lut.best / gemv_cb.best, gg_lut.best / gg_cb.best);

    cudaFree(d_w); cudaFree(d_gauss); cudaFree(d_computed); cudaFree(d_scales);
    cudaFree(d_x); cudaFree(d_deq); cudaFree(d_y);
    cudaFree(d_packed_lut); cudaFree(d_packed_cb); cudaFree(d_trace); cudaFree(d_indices);
    return 0;
}
