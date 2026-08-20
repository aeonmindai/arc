// Measured per-op issue throughput on THIS H200 -- because ncu is blocked
// (RmProfilingAdminOnly=1) and the static cubin-counting rig was shown dead.
//
// Each probe runs INDEP independent accumulator chains so latency is hidden and
// what is measured is issue/pipe throughput. Result is reported as ops per SM
// per clock, directly comparable to the CUDA C Programming Guide's Table of
// "Throughput of Native Arithmetic Instructions" for compute capability 9.0.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){printf("cuda %s:%d %s\n",#x,__LINE__,cudaGetErrorString(e_));exit(2);} } while(0)

constexpr int INDEP = 8;
constexpr int INNER = 512;

#define PROBE(NAME, DECL, INIT, BODY, SINK)                                    \
__global__ void probe_##NAME(unsigned int seed, float* out, int loops) {        \
    DECL;                                                                       \
    INIT;                                                                       \
    for (int L = 0; L < loops; ++L) {                                           \
        _Pragma("unroll 8")                                                     \
        for (int i = 0; i < INNER; ++i) { BODY; }                               \
    }                                                                           \
    float s = SINK;                                                             \
    if (s == 12345.678f) out[0] = s;                                            \
}

// 1 op per element per inner iteration in every probe below.
PROBE(ffma, float a[INDEP]; float b = 1.0000001f; float c = 0.9999999f,
      { for (int k=0;k<INDEP;++k) a[k] = (float)(seed + k) * 1e-6f; },
      { _Pragma("unroll") for (int k=0;k<INDEP;++k) a[k] = fmaf(a[k], b, c); },
      (a[0]+a[1]+a[2]+a[3]+a[4]+a[5]+a[6]+a[7]))

PROBE(fadd, float a[INDEP]; float b = 1.0000001f,
      { for (int k=0;k<INDEP;++k) a[k] = (float)(seed + k) * 1e-6f; },
      { _Pragma("unroll") for (int k=0;k<INDEP;++k) a[k] = __fadd_rn(a[k], b); },
      (a[0]+a[1]+a[2]+a[3]+a[4]+a[5]+a[6]+a[7]))

PROBE(imad, unsigned int a[INDEP]; unsigned int m = 0xB5DE9C89u,
      { for (int k=0;k<INDEP;++k) a[k] = seed + k*2654435761u; },
      { _Pragma("unroll") for (int k=0;k<INDEP;++k) a[k] = a[k] * m; },
      (float)(a[0]^a[1]^a[2]^a[3]^a[4]^a[5]^a[6]^a[7]))

PROBE(lop3, unsigned int a[INDEP],
      { for (int k=0;k<INDEP;++k) a[k] = seed + k*2654435761u; },
      { _Pragma("unroll") for (int k=0;k<INDEP;++k) a[k] = (a[k] & 0x8FFF8FFFu) ^ 0x3B603B60u; },
      (float)(a[0]^a[1]^a[2]^a[3]^a[4]^a[5]^a[6]^a[7]))

// half -> float conversion (F2F.F32.F16). One conversion per element.
PROBE(h2f, unsigned int a[INDEP]; float f[INDEP],
      { for (int k=0;k<INDEP;++k) { a[k] = seed + k*2654435761u; f[k]=0.f; } },
      { _Pragma("unroll") for (int k=0;k<INDEP;++k) {
            f[k] += __half2float(__ushort_as_half((unsigned short)(a[k] & 0xFFFFu)));
            a[k] += 0x00010001u; } },
      (f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]))

// int -> float conversion (I2F), for contrast with h2f.
PROBE(i2f, unsigned int a[INDEP]; float f[INDEP],
      { for (int k=0;k<INDEP;++k) { a[k] = seed + k*2654435761u; f[k]=0.f; } },
      { _Pragma("unroll") for (int k=0;k<INDEP;++k) { f[k] += (float)(a[k] & 0xFFFFu); a[k] += 0x00010001u; } },
      (f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]))

// The SHIPPED fold: 1 LOP3 + 2 F2F + 1 FADD, per element.
__device__ __forceinline__ float fold_f2f(unsigned int x) {
    const unsigned int m = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
    return __fadd_rn(__half2float(__ushort_as_half((unsigned short)(m >> 16))),
                     __half2float(__ushort_as_half((unsigned short)(m & 0xFFFFu))));
}
// BIT-IDENTICAL integer reconstruction of the same two f32 values. Legal because
// the mask+xor pins each half's exponent field to {12,13,14,15}: never zero
// (no denormal/zero path) and never 31 (no inf/nan path), so the fp16->fp32
// widening is exactly sign | ((mag << 13) + (112 << 23)).
__device__ __forceinline__ float fold_int(unsigned int x) {
    const unsigned int m = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
    const unsigned int hi = (m & 0x80000000u) | (((m >> 3) & 0x0FFFE000u) + 0x38000000u);
    const unsigned int lo = ((m << 16) & 0x80000000u) | (((m & 0x7FFFu) << 13) + 0x38000000u);
    return __fadd_rn(__int_as_float(hi), __int_as_float(lo));
}

PROBE(fold_f2f, unsigned int a[INDEP]; float f[INDEP],
      { for (int k=0;k<INDEP;++k) { a[k] = seed + k*2654435761u; f[k]=0.f; } },
      { _Pragma("unroll") for (int k=0;k<INDEP;++k) { f[k] += fold_f2f(a[k]); a[k] += 0x9E3779B9u; } },
      (f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]))

PROBE(fold_int, unsigned int a[INDEP]; float f[INDEP],
      { for (int k=0;k<INDEP;++k) { a[k] = seed + k*2654435761u; f[k]=0.f; } },
      { _Pragma("unroll") for (int k=0;k<INDEP;++k) { f[k] += fold_int(a[k]); a[k] += 0x9E3779B9u; } },
      (f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]))

// Whole codeword pair (both folds + the chained MCG) -- the real per-symbol cost.
PROBE(cw_f2f, unsigned int a[INDEP]; float f[INDEP]; unsigned int mu = 0xB5DE9C89u,
      { for (int k=0;k<INDEP;++k) { a[k] = seed + k*2654435761u; f[k]=0.f; } },
      { _Pragma("unroll") for (int k=0;k<INDEP;++k) {
            unsigned int x0 = a[k] * mu; unsigned int x1 = x0 * mu;
            f[k] += fold_f2f(x0) + fold_f2f(x1); a[k] += 0x9E3779B9u; } },
      (f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]))

PROBE(cw_int, unsigned int a[INDEP]; float f[INDEP]; unsigned int mu = 0xB5DE9C89u,
      { for (int k=0;k<INDEP;++k) { a[k] = seed + k*2654435761u; f[k]=0.f; } },
      { _Pragma("unroll") for (int k=0;k<INDEP;++k) {
            unsigned int x0 = a[k] * mu; unsigned int x1 = x0 * mu;
            f[k] += fold_int(x0) + fold_int(x1); a[k] += 0x9E3779B9u; } },
      (f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]))

struct Ent { const char* name; void (*fn)(unsigned int, float*, int); double ops_per_elem; };

int main() {
    cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, 0));
    const int blocks = p.multiProcessorCount * 8, threads = 256, loops = 64;
    printf("# %s SMs=%d clockMax=%.0f MHz | %d blocks x %d thr, %d indep chains, %d inner x %d loops\n",
           p.name, p.multiProcessorCount, p.clockRate / 1000.0, blocks, threads, INDEP, INNER, loops);
    float* d_out; CK(cudaMalloc(&d_out, 4));

    const Ent ents[] = {
        {"FFMA f32",              probe_ffma,     1},
        {"FADD f32",              probe_fadd,     1},
        {"IMAD u32 mul",          probe_imad,     1},
        {"LOP3 (and^xor)",        probe_lop3,     1},
        {"F2F  half->float",      probe_h2f,      1},   // + 1 FADD + 1 IADD carried
        {"I2F  int->float",       probe_i2f,      1},   // + 1 FADD + 1 IADD carried
        {"fold  (SHIPPED, 2xF2F)",probe_fold_f2f, 1},
        {"fold  (int widen)",     probe_fold_int, 1},
        {"codeword pair f32cvt",  probe_cw_f2f,   1},
        {"codeword pair intcvt",  probe_cw_int,   1},
    };
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const double elems = (double)blocks * threads * INDEP * INNER * loops;
    printf("%-26s %10s %14s %12s\n", "probe", "ms", "Gop/s", "ops/SM/clk");
    for (const auto& e : ents) {
        e.fn<<<blocks, threads>>>(1234u, d_out, 4); CK(cudaDeviceSynchronize());
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            CK(cudaEventRecord(e0));
            e.fn<<<blocks, threads>>>(1234u, d_out, loops);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms < best) best = ms;
        }
        const double ops = elems * e.ops_per_elem;
        const double gops = ops / (best * 1e-3) / 1e9;
        const double per_sm_clk = gops * 1e9 / (p.multiProcessorCount * (p.clockRate * 1e3));
        printf("%-26s %10.3f %14.1f %12.1f\n", e.name, best, gops, per_sm_clk);
    }
    printf("\n(ops/SM/clk uses the max clock %.0f MHz; sustained clock is lower, so\n"
           " true per-clock rates are slightly HIGHER than shown.)\n", p.clockRate / 1000.0);
    return 0;
}
