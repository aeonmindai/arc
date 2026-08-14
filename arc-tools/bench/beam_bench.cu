// Standalone driver for `qtip_quantize_rows_beam_kernel`, for measuring kernel
// wall time and old-vs-new byte identity on a rented GPU without building the
// whole Rust workspace.
//
// Build (the kernel under test is included, so build it twice against two
// different checkouts to get a ratio on the SAME box):
//
//     nvcc -O3 -std=c++17 --use_fast_math -arch=sm_80 \
//          -I<kernels/qtip dir> beam_bench.cu -o bench
//
// Usage: ./bench <n_rows> <k_in> <beam_w> <iters> <out.bin>
//
// The LUT and weights are generated with the SAME deterministic hashes the Rust
// side uses (`gaussian_lut`, splitmix64 + Box-Muller), so the candidate cost
// distribution — and therefore the data-dependent radix pass count, which is
// the largest single term in the kernel's cost — matches a real bake rather
// than a synthetic one.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "qtip_beam.cu"

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

int main(int argc, char** argv) {
    int n_rows  = argc > 1 ? atoi(argv[1]) : 512;
    int k_in    = argc > 2 ? atoi(argv[2]) : 7168;
    int beam_w  = argc > 3 ? atoi(argv[3]) : 256;
    int iters   = argc > 4 ? atoi(argv[4]) : 3;
    const char* out = argc > 5 ? argv[5] : nullptr;

    const int num_symbols   = k_in / 2;            // V = 2
    const int packed_per_row = num_symbols / 2;    // two K=4 symbols per byte

    // ---- host data -------------------------------------------------------
    std::vector<float> lut((size_t)65536 * 2);
    for (uint32_t s = 0; s < 65536u; ++s) gaussian_pair(s, &lut[2 * s], &lut[2 * s + 1]);

    std::vector<float> w((size_t)n_rows * k_in);
    std::vector<float> scales(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        float mx = 0.0f;
        for (int i = 0; i < k_in; ++i) {
            float a, b;
            gaussian_pair((uint32_t)(r * k_in + i) + 0x9E37u, &a, &b);
            float v = a * 0.5f;
            w[(size_t)r * k_in + i] = v;
            float av = fabsf(v);
            if (av > mx) mx = av;
        }
        scales[r] = (mx == 0.0f) ? 1.0f : mx / 3.0f;
    }

    // ---- device buffers ---------------------------------------------------
    float *d_w, *d_lut, *d_scales;
    uint8_t* d_packed;
    uint32_t* d_trace;
    CK(cudaMalloc(&d_w, w.size() * sizeof(float)));
    CK(cudaMalloc(&d_lut, lut.size() * sizeof(float)));
    CK(cudaMalloc(&d_scales, scales.size() * sizeof(float)));
    CK(cudaMalloc(&d_packed, (size_t)n_rows * packed_per_row));
    size_t trace_elems = (size_t)n_rows * num_symbols * beam_w;
    CK(cudaMalloc(&d_trace, trace_elems * sizeof(uint32_t)));
    CK(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    printf("shape n_rows=%d k_in=%d T=%d W=%d | trace %.2f GB\n", n_rows, k_in,
           num_symbols, beam_w, trace_elems * 4.0 / 1e9);

    // ---- warmup + timed runs ---------------------------------------------
    int rc = launch_qtip_quantize_rows_beam_f32(d_w, d_lut, d_scales, d_packed, d_trace,
                                                n_rows, k_in, num_symbols, 0, beam_w, 0);
    if (rc != 0) { printf("launch refused rc=%d\n", rc); return 1; }
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    float best = 1e30f, sum = 0.0f;
    for (int it = 0; it < iters; ++it) {
        CK(cudaEventRecord(t0));
        launch_qtip_quantize_rows_beam_f32(d_w, d_lut, d_scales, d_packed, d_trace,
                                           n_rows, k_in, num_symbols, 0, beam_w, 0);
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
        sum += ms;
        printf("  iter %d: %.2f ms\n", it, ms);
    }

    // Per-layer extrapolation: a V4-Flash layer is 284e9/44/2 = 3.23e9
    // (row x timestep) units. Report in those units so A30 and H200 runs are
    // comparable as RATIOS even though the absolute times are not.
    double units = (double)n_rows * num_symbols;
    double per_unit_ns = best * 1e6 / units;
    printf("RESULT best_ms=%.3f mean_ms=%.3f ns_per_row_timestep=%.4f "
           "extrapolated_layer_s=%.1f\n",
           best, sum / iters, per_unit_ns, per_unit_ns * 3.2273e9 / 1e9);

    // ---- dump packed output for a byte-identity comparison ----------------
    if (out) {
        std::vector<uint8_t> host((size_t)n_rows * packed_per_row);
        CK(cudaMemcpy(host.data(), d_packed, host.size(), cudaMemcpyDeviceToHost));
        FILE* f = fopen(out, "wb");
        fwrite(host.data(), 1, host.size(), f);
        fclose(f);
        uint64_t h = 1469598103934665603ULL;
        for (uint8_t b : host) { h ^= b; h *= 1099511628211ULL; }
        printf("PACKED fnv1a=%016llx bytes=%zu\n", (unsigned long long)h, host.size());
    }

    cudaFree(d_w); cudaFree(d_lut); cudaFree(d_scales);
    cudaFree(d_packed); cudaFree(d_trace);
    return 0;
}
