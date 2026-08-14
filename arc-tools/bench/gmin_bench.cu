// Standalone driver comparing the gmin-only exhaustive Viterbi kernel against
// the production beam kernel on ONE box, ONE binary, ONE data set.
//
// All three trellis quantizers are compiled into this translation unit, so the
// comparison has no cross-build confound: the beam number and the exhaustive
// number come out of the same process, from the same weights and the same LUT,
// and byte identity is checked against the kernel that is already proven
// bit-identical to the CPU exhaustive reference
// (`cuda_exhaustive_matches_cpu_exhaustive_bit_for_bit`).
//
// Build:
//     nvcc -O3 -std=c++17 --use_fast_math -arch=sm_80 \
//          -I<repo>/mistralrs-quant/kernels/qtip gmin_bench.cu -o gmin_bench
//
// Usage: ./gmin_bench <n_rows> <k_in> <iters>
//
// The LUT and weights use the SAME deterministic hashes the Rust side uses
// (`gaussian_lut`, splitmix64 + Box-Muller), so the candidate cost distribution
// — and therefore the beam's data-dependent radix pass count, its largest single
// cost term — matches a real bake rather than a synthetic one. This is the same
// generator `beam_bench.cu` uses, so numbers are comparable across both drivers.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "qtip_quantize.cu"
#include "qtip_beam.cu"
#include "qtip_gmin.cu"

#define CK(x)                                                                  \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            printf("CUDA ERROR %s at %d: %s\n", #x, __LINE__,                  \
                   cudaGetErrorString(e_));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Mirrors qtip/mod.rs::hash_to_two_uniforms + box_muller exactly.
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

static uint64_t fnv1a(const std::vector<uint8_t>& v) {
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t b : v) { h ^= b; h *= 1099511628211ULL; }
    return h;
}

struct Data {
    std::vector<float> lut;
    std::vector<float> w;
    std::vector<float> scales;
};

static Data make_data(int n_rows, int k_in) {
    Data d;
    d.lut.resize((size_t)65536 * 2);
    for (uint32_t s = 0; s < 65536u; ++s) gaussian_pair(s, &d.lut[2 * s], &d.lut[2 * s + 1]);
    d.w.resize((size_t)n_rows * k_in);
    d.scales.resize(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        float mx = 0.0f;
        for (int i = 0; i < k_in; ++i) {
            float a, b;
            gaussian_pair((uint32_t)(r * k_in + i) + 0x9E37u, &a, &b);
            float v = a * 0.5f;
            d.w[(size_t)r * k_in + i] = v;
            float av = fabsf(v);
            if (av > mx) mx = av;
        }
        d.scales[r] = (mx == 0.0f) ? 1.0f : mx / 3.0f;
    }
    return d;
}

// ---------------------------------------------------------------------------
// Parity: the gmin kernel must reproduce the existing exhaustive kernel's
// packed bytes exactly. That kernel is the one already gated against the CPU
// exhaustive DP, so byte identity here is byte identity with the CPU reference.
// ---------------------------------------------------------------------------
static int parity_check(int n_rows, int k_in) {
    const int T   = k_in / 2;
    const int ppr = T / 2;
    Data d = make_data(n_rows, k_in);

    float *d_w, *d_lut, *d_scales;
    uint8_t *d_pk_ex, *d_pk_gm;
    CK(cudaMalloc(&d_w, d.w.size() * sizeof(float)));
    CK(cudaMalloc(&d_lut, d.lut.size() * sizeof(float)));
    CK(cudaMalloc(&d_scales, d.scales.size() * sizeof(float)));
    CK(cudaMalloc(&d_pk_ex, (size_t)n_rows * ppr));
    CK(cudaMalloc(&d_pk_gm, (size_t)n_rows * ppr));
    CK(cudaMemcpy(d_w, d.w.data(), d.w.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, d.lut.data(), d.lut.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_scales, d.scales.data(), d.scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Existing exhaustive kernel scratch.
    float *d_ca, *d_cb;
    uint8_t* d_bt;
    CK(cudaMalloc(&d_ca, (size_t)n_rows * 65536 * sizeof(float)));
    CK(cudaMalloc(&d_cb, (size_t)n_rows * 65536 * sizeof(float)));
    CK(cudaMalloc(&d_bt, (size_t)n_rows * T * 4096));
    launch_qtip_quantize_rows_viterbi_f32(d_w, d_lut, d_scales, d_pk_ex,
                                          d_ca, d_cb, d_bt, n_rows, k_in, T, 0, 0);
    CK(cudaDeviceSynchronize());

    // gmin kernel scratch.
    const int words = qtip_gmin_trace_words_per_position();
    uint32_t* d_tr;
    CK(cudaMalloc(&d_tr, (size_t)n_rows * T * words * sizeof(uint32_t)));
    int rc = launch_qtip_quantize_rows_gmin_f32(d_w, d_lut, d_scales, d_pk_gm, d_tr,
                                                n_rows, k_in, T, 0, 0);
    if (rc != 0) { printf("PARITY gmin launch refused rc=%d\n", rc); return 1; }
    CK(cudaDeviceSynchronize());

    std::vector<uint8_t> h_ex((size_t)n_rows * ppr), h_gm((size_t)n_rows * ppr);
    CK(cudaMemcpy(h_ex.data(), d_pk_ex, h_ex.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_gm.data(), d_pk_gm, h_gm.size(), cudaMemcpyDeviceToHost));

    size_t bad = 0;
    for (size_t i = 0; i < h_ex.size(); ++i) if (h_ex[i] != h_gm[i]) ++bad;

    // Non-vacuity: a degenerate fixture (all-equal bytes) would make identity
    // meaningless, so report the byte entropy proxy the caller can eyeball.
    size_t nonzero = 0;
    for (uint8_t b : h_ex) if (b != 0) ++nonzero;

    printf("PARITY rows=%d k_in=%d T=%d bytes=%zu mismatches=%zu nonzero=%zu "
           "exhaustive_fnv1a=%016llx gmin_fnv1a=%016llx %s\n",
           n_rows, k_in, T, h_ex.size(), bad, nonzero,
           (unsigned long long)fnv1a(h_ex), (unsigned long long)fnv1a(h_gm),
           bad == 0 ? "PASS" : "FAIL");

    cudaFree(d_w); cudaFree(d_lut); cudaFree(d_scales);
    cudaFree(d_pk_ex); cudaFree(d_pk_gm);
    cudaFree(d_ca); cudaFree(d_cb); cudaFree(d_bt); cudaFree(d_tr);
    return bad == 0 ? 0 : 1;
}

// A V4-Flash layer is 284e9 / 44 / 2 = 3.2273e9 (row x timestep) units. Report
// in those units so this box's absolute times convert to a per-layer figure the
// same way `beam_bench.cu` does.
static void report(const char* tag, float best_ms, double units) {
    double per_unit_ns = best_ms * 1e6 / units;
    printf("RESULT %-24s best_ms=%9.3f ns_per_row_timestep=%8.4f "
           "extrapolated_layer_s=%8.1f\n",
           tag, best_ms, per_unit_ns, per_unit_ns * 3.2273e9 / 1e9);
}

int main(int argc, char** argv) {
    int n_rows = argc > 1 ? atoi(argv[1]) : 448;
    int k_in   = argc > 2 ? atoi(argv[2]) : 7168;
    int iters  = argc > 3 ? atoi(argv[3]) : 3;

    // Small-shape byte parity first: it is cheap and it is the gate. A failure
    // here makes every timing number below meaningless.
    if (parity_check(8, 512) != 0) return 1;
    if (parity_check(4, 2048) != 0) return 1;

    const int T   = k_in / 2;
    const int ppr = T / 2;
    Data d = make_data(n_rows, k_in);

    float *d_w, *d_lut, *d_scales;
    uint8_t* d_packed;
    CK(cudaMalloc(&d_w, d.w.size() * sizeof(float)));
    CK(cudaMalloc(&d_lut, d.lut.size() * sizeof(float)));
    CK(cudaMalloc(&d_scales, d.scales.size() * sizeof(float)));
    CK(cudaMalloc(&d_packed, (size_t)n_rows * ppr));
    CK(cudaMemcpy(d_w, d.w.data(), d.w.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, d.lut.data(), d.lut.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_scales, d.scales.data(), d.scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int beam_w = 256;
    const int words  = qtip_gmin_trace_words_per_position();
    size_t beam_trace_elems = (size_t)n_rows * T * beam_w;
    size_t gmin_trace_elems = (size_t)n_rows * T * words;
    printf("shape n_rows=%d k_in=%d T=%d | beam trace %.2f GB | gmin trace %.2f GB\n",
           n_rows, k_in, T, beam_trace_elems * 4.0 / 1e9, gmin_trace_elems * 4.0 / 1e9);

    uint32_t* d_trace;
    size_t trace_elems = beam_trace_elems > gmin_trace_elems ? beam_trace_elems : gmin_trace_elems;
    CK(cudaMalloc(&d_trace, trace_elems * sizeof(uint32_t)));

    const double units = (double)n_rows * (double)T;

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));

    std::vector<uint8_t> host((size_t)n_rows * ppr);

    // ---- beam W=256 -------------------------------------------------------
    {
        launch_qtip_quantize_rows_beam_f32(d_w, d_lut, d_scales, d_packed, d_trace,
                                           n_rows, k_in, T, 0, beam_w, 0);
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int it = 0; it < iters; ++it) {
            CK(cudaEventRecord(t0));
            launch_qtip_quantize_rows_beam_f32(d_w, d_lut, d_scales, d_packed, d_trace,
                                               n_rows, k_in, T, 0, beam_w, 0);
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            float ms = 0.0f;
            CK(cudaEventElapsedTime(&ms, t0, t1));
            if (ms < best) best = ms;
        }
        CK(cudaMemcpy(host.data(), d_packed, host.size(), cudaMemcpyDeviceToHost));
        report("beam_w256", best, units);
        printf("       beam_w256 fnv1a=%016llx\n", (unsigned long long)fnv1a(host));
    }

    // ---- gmin exhaustive, three variants ----------------------------------
    const char* names[3] = {"gmin_lb4_traced", "gmin_lb2_traced", "gmin_lb4_notrace"};
    for (int v = 0; v < 3; ++v) {
        int rc = launch_qtip_quantize_rows_gmin_variant_f32(
            d_w, d_lut, d_scales, d_packed, d_trace, n_rows, k_in, T, 0, v, 0);
        if (rc != 0) { printf("gmin variant %d refused rc=%d\n", v, rc); continue; }
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int it = 0; it < iters; ++it) {
            CK(cudaEventRecord(t0));
            launch_qtip_quantize_rows_gmin_variant_f32(
                d_w, d_lut, d_scales, d_packed, d_trace, n_rows, k_in, T, 0, v, 0);
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            float ms = 0.0f;
            CK(cudaEventElapsedTime(&ms, t0, t1));
            if (ms < best) best = ms;
        }
        report(names[v], best, units);
        if (v != 2) {
            CK(cudaMemcpy(host.data(), d_packed, host.size(), cudaMemcpyDeviceToHost));
            printf("       %s fnv1a=%016llx\n", names[v], (unsigned long long)fnv1a(host));
        }
    }

    cudaFree(d_w); cudaFree(d_lut); cudaFree(d_scales);
    cudaFree(d_packed); cudaFree(d_trace);
    return 0;
}
