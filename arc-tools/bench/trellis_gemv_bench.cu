// Trellis MoE gather-GEMV microbenchmark (ArcInfer / ArcMoE, ArcQuant/QTIP).
//
// Prices `qtip_gather_gemv_warp_kernel` at the THREE real DeepSeek-V4-Flash
// decode call sites (gate / up / down), with the packed weights read out of a
// rotating >320 MB pool so the 50 MB L2 cannot manufacture fictional bandwidth.
//
// Build:
//   nvcc -O3 -std=c++17 --use_fast_math -arch=sm_90 \
//        -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
//        -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
//        --expt-relaxed-constexpr --expt-extended-lambda \
//        -I. trellis_gemv_bench.cu -o trellis_gemv_bench
//
// Variant 0 is a verbatim transcription of the shipped kernel; every other
// variant is diffed against variant 0's output BYTE FOR BYTE.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "qtip_codebook.cuh"

#define CK(x)                                                                  \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            printf("FATAL cuda: %s at %d: %s\n", #x, __LINE__,                 \
                   cudaGetErrorString(e_));                                    \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

namespace {

constexpr uint32_t QTIP_K           = 4;
constexpr uint32_t QTIP_L           = 16;
constexpr uint32_t QTIP_V           = 2;
constexpr uint32_t QTIP_STATE_MASK  = (1u << QTIP_L) - 1u;
constexpr uint32_t QTIP_WARMUP_SYMS = QTIP_L / QTIP_K;

template <typename T> __device__ __forceinline__ T gg_from_f32(float v);
template <> __device__ __forceinline__ float gg_from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __half gg_from_f32<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 gg_from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float gg_warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

__device__ __forceinline__ float2 gg_load2(const float* p) { return __ldg(reinterpret_cast<const float2*>(p)); }
__device__ __forceinline__ float2 gg_load2(const __half* p) { return __half22float2(__ldg(reinterpret_cast<const __half2*>(p))); }
__device__ __forceinline__ float2 gg_load2(const __nv_bfloat16* p) { return __bfloat1622float2(__ldg(reinterpret_cast<const __nv_bfloat162*>(p))); }

__device__ __forceinline__ float2 qtip_decode_state(uint32_t state) {
    unsigned long long z = (unsigned long long)state * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const uint32_t hi = (uint32_t)(z >> 32);
    const uint32_t lo = (uint32_t)(z & 0xFFFFFFFFu);
    const float denom = 4294967296.0f;
    const float u1 = ((float)hi + 1.0f) / denom;
    const float u2 = ((float)lo + 1.0f) / denom;
    const float r = sqrtf(-2.0f * logf(u1));
    const float theta = (2.0f * 3.14159265358979323846f) * u2;
    float s, c;
    sincosf(theta, &s, &c);
    return make_float2(r * c, r * s);
}

template <bool COMPUTED_CB>
__device__ __forceinline__ float2 gg_codeword(uint32_t state, unsigned int mult) {
    if (COMPUTED_CB) return qtip_cb_sum2(state, mult);
    return qtip_decode_state(state);
}

__device__ __forceinline__ float2 gg_unpack2(unsigned int w, const __nv_bfloat16*) {
    return make_float2(__bfloat162float(__ushort_as_bfloat16((unsigned short)(w & 0xFFFFu))),
                       __bfloat162float(__ushort_as_bfloat16((unsigned short)(w >> 16))));
}
__device__ __forceinline__ float2 gg_unpack2(unsigned int w, const __half*) {
    return make_float2(__half2float(__ushort_as_half((unsigned short)(w & 0xFFFFu))),
                       __half2float(__ushort_as_half((unsigned short)(w >> 16))));
}
__device__ __forceinline__ float2 gg_unpack2(unsigned int, const float*) { return make_float2(0.0f, 0.0f); }

template <typename T, int GROUP>
__device__ __forceinline__ void gg_load_group(const T* p, bool vec_ok, float2* out) {
    if constexpr (sizeof(T) == 2 && (GROUP % 4) == 0) {
        if (vec_ok) {
            #pragma unroll
            for (int q = 0; q < GROUP / 4; ++q) {
                const uint4 v = __ldg(reinterpret_cast<const uint4*>(p) + q);
                out[4 * q + 0] = gg_unpack2(v.x, (const T*)nullptr);
                out[4 * q + 1] = gg_unpack2(v.y, (const T*)nullptr);
                out[4 * q + 2] = gg_unpack2(v.z, (const T*)nullptr);
                out[4 * q + 3] = gg_unpack2(v.w, (const T*)nullptr);
            }
            return;
        }
    }
    #pragma unroll
    for (int j = 0; j < GROUP; ++j) out[j] = gg_load2(p + j * (int)QTIP_V);
}

__device__ __forceinline__ uint32_t gg_state_from_window(uint32_t h) {
    const uint32_t a = ((h & 0x0F0Fu) << 4) | ((h >> 4) & 0x0F0Fu);
    return ((a & 0x00FFu) << 8) | ((a >> 8) & 0x00FFu);
}

// ---------------------------------------------------------------------------
// Parameterised kernel. (STAGE_VEC=0, MLP=0, SCALE_HOIST=0, GROUP=4,
// ROWS_PER_WARP=2, WARPS_PER_BLOCK=8) is the shipped kernel verbatim.
//
//   STAGE_VEC   : stage the packed rows with uint4 loads instead of byte loads
//   MLP         : hoist every row's window load ahead of every row's decode
//   SCALE_HOIST : multiply the row scale in ONCE at the end instead of per
//                 weight (costs bit-exactness -- FP reassociation)
// ---------------------------------------------------------------------------
// ABL: dynamic ablation, to attribute the runtime without a profiler.
//   0 = full kernel
//   1 = codeword replaced by 2 LOP3s  -> the codebook's share
//   2 = no packed loads (window synthesised in registers) -> the memory path's share
//   3 = no fma/activation (accumulate the codeword itself) -> the fma+x share
template <int ABL>
__device__ __forceinline__ float2 abl_codeword(uint32_t state, unsigned int mult) {
    if (ABL == 1) return make_float2(__int_as_float(0x3F800000u | (state & 0x7FFFu)),
                                     __int_as_float(0x3F800000u | ((state >> 1) & 0x7FFFu)));
    return qtip_cb_sum2(state, mult);
}

template <typename T, int WARPS_PER_BLOCK, int ROWS_PER_WARP, bool COMPUTED_CB,
          bool STAGED, int GROUP, int STAGE_VEC, int MLP, int SCALE_HOIST, int PEEL, int ABL = 0>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
tg_kernel(const uint8_t* __restrict__ packed, const float* __restrict__ row_scales,
          const float* __restrict__ lut, const T* __restrict__ x,
          const uint32_t* __restrict__ indices, T* __restrict__ y,
          int n_rows, int packed_per_row, int num_symbols, int n_pairs,
          int num_experts, unsigned int cb_mult) {
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    extern __shared__ float s_mem[];
    (void)lut;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int pair = blockIdx.y;
    const int row0_block = blockIdx.x * ROWS_PER_BLOCK;
    if (row0_block >= n_rows || pair >= n_pairs) return;

    const uint32_t expert = __ldg(indices + pair);
    const bool invalid = expert >= (uint32_t)num_experts;

    uint8_t* s_packed = reinterpret_cast<uint8_t*>(s_mem);
    if (STAGED && !invalid) {
        const int n_block_rows = min(ROWS_PER_BLOCK, n_rows - row0_block);
        const long total = (long)n_block_rows * packed_per_row;
        // ABL 4: every block stages the SAME 16 KB, so the weights live in L2 and
        // DRAM traffic collapses to ~nothing. Instruction stream is unchanged.
        const uint8_t* __restrict__ g = (ABL == 4)
            ? packed
            : packed + ((size_t)expert * n_rows + row0_block) * packed_per_row;
        if (STAGE_VEC && ((packed_per_row & 15) == 0)) {
            const uint4* g4 = reinterpret_cast<const uint4*>(g);
            uint4* s4 = reinterpret_cast<uint4*>(s_packed);
            const int total4 = (int)(total >> 4);
            for (int i = threadIdx.x; i < total4; i += WARPS_PER_BLOCK * 32) s4[i] = __ldg(g4 + i);
        } else {
            for (long i = threadIdx.x; i < total; i += WARPS_PER_BLOCK * 32) s_packed[i] = __ldg(g + i);
        }
        __syncthreads();
    }

    const int row0 = row0_block + warp * ROWS_PER_WARP;
    if (invalid) {
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const int row = row0 + r;
                if (row < n_rows) y[(size_t)pair * n_rows + row] = gg_from_f32<T>(0.0f);
            }
        }
        return;
    }

    const uint8_t* rp_g[ROWS_PER_WARP];
    unsigned int   rp_s[ROWS_PER_WARP];
    float scl[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = row0 + r;
        const bool ok = row < n_rows;
        const size_t er = (size_t)expert * n_rows + (ok ? row : 0);
        rp_g[r] = packed + er * packed_per_row;
        rp_s[r] = (unsigned int)((warp * ROWS_PER_WARP + r) * packed_per_row);
        scl[r]  = ok ? __ldg(row_scales + er) : 0.0f;
    }
    // ABL 5: stage and stop. Reads two staged bytes per row through RUNTIME
    // indices so ptxas cannot narrow or delete the 16 KB staging loop.
    if (ABL == 5) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int row = row0 + r;
            if (lane == 0 && row < n_rows)
                y[(size_t)pair * n_rows + row] = gg_from_f32<T>(
                    (float)s_packed[rp_s[r]] + (float)s_packed[rp_s[r] + packed_per_row - 1]);
        }
        return;
    }

    const T* __restrict__ x_pair = x + (size_t)pair * num_symbols * QTIP_V;
    const bool xvec = ((reinterpret_cast<uintptr_t>(x_pair) & 15u) == 0u);

    const int gstride = 32 * GROUP;
    constexpr int NH = (QTIP_WARMUP_SYMS + GROUP + 3) / 4;

    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    // The two loop bodies, factored so the PEEL and non-PEEL forms share one copy.
    // FAST: full group, full warm-up window in range -- no per-symbol bounds test.
    auto do_fast = [&](int base) {
        float2 xg[GROUP];
        gg_load_group<T, GROUP>(x_pair + base * (int)QTIP_V, xvec, xg);
        const int b0 = (base - (int)QTIP_WARMUP_SYMS) >> 1;
        uint32_t hh[ROWS_PER_WARP][NH];
        #define GG_WIN(RR, QQ)                                                                       \
            ((ABL == 2 || ABL == 7) ? (cb_mult + (unsigned)base + (unsigned)((RR) * 16 + (QQ)))       \
             : STAGED ? (uint32_t)*reinterpret_cast<const unsigned short*>(s_packed + rp_s[RR] + b0 + 2 * (QQ)) \
                      : (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(rp_g[RR] + b0 + 2 * (QQ))))
        if (MLP) {   // every row's window issued before any decode
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                #pragma unroll
                for (int q = 0; q < NH; ++q) hh[r][q] = GG_WIN(r, q);
            }
        }
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            if (!MLP) {
                #pragma unroll
                for (int q = 0; q < NH; ++q) hh[r][q] = GG_WIN(r, q);
            }
            const float scale = scl[r];
            uint32_t state = gg_state_from_window(hh[r][0]);
            float a = acc[r];
            #pragma unroll
            for (int j = 0; j < GROUP; ++j) {
                const int t = (int)QTIP_WARMUP_SYMS + j;
                const uint32_t sym = (hh[r][t >> 2] >> (4 * (t & 3))) & 0x0Fu;
                state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;
                if (ABL == 7) { a += (float)(state & 1u); continue; }   // loop+state only
                const float2 w = abl_codeword<ABL>(state, cb_mult);
                if (ABL == 3) {
                    a = a + w.x + w.y;
                } else if (SCALE_HOIST) {
                    a = fmaf(w.x, xg[j].x, a);
                    a = fmaf(w.y, xg[j].y, a);
                } else {
                    a = fmaf(w.x * scale, xg[j].x, a);
                    a = fmaf(w.y * scale, xg[j].y, a);
                }
            }
            acc[r] = a;
        }
        #undef GG_WIN
    };
    // GENERAL: partial group and/or truncated warm-up. Identical arithmetic to
    // the shipped kernel. Only lane 0's first iteration needs it for V4 shapes.
    auto do_general = [&](int base) {
        const int gend = min(base + GROUP, num_symbols);
        float2 xg[GROUP];
        gg_load_group<T, GROUP>(x_pair + base * (int)QTIP_V,
                                xvec && (base + GROUP <= num_symbols), xg);
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const float scale = scl[r];
            uint32_t state = 0u;
            float a = acc[r];
            const uint8_t* row_packed = STAGED ? (s_packed + rp_s[r]) : rp_g[r];
            const int w0 = base > (int)QTIP_WARMUP_SYMS ? base - (int)QTIP_WARMUP_SYMS : 0;
            for (int t = w0; t < base; ++t) {
                const uint8_t b = row_packed[t >> 1];
                const uint32_t sym = (t & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
                state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;
            }
            #pragma unroll
            for (int j = 0; j < GROUP; ++j) {
                const int s = base + j;
                if (s >= gend) break;
                const uint8_t b = row_packed[s >> 1];
                const uint32_t sym = (s & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
                state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;
                const float2 w = gg_codeword<COMPUTED_CB>(state, cb_mult);
                if (SCALE_HOIST) { a = fmaf(w.x, xg[j].x, a); a = fmaf(w.y, xg[j].y, a); }
                else { a = fmaf(w.x * scale, xg[j].x, a); a = fmaf(w.y * scale, xg[j].y, a); }
            }
            acc[r] = a;
        }
    };

    if (PEEL) {
        // Hot loop carries ONLY the fast body: no `fast` test, no BSSY/BSYNC
        // divergence region, no per-symbol bounds test inside the loop.
        int base = lane * GROUP;
        for (; base < num_symbols && base < (int)QTIP_WARMUP_SYMS; base += gstride) do_general(base);
        for (; base + GROUP <= num_symbols; base += gstride) do_fast(base);
        for (; base < num_symbols; base += gstride) do_general(base);
    } else {
        for (int base = lane * GROUP; base < num_symbols; base += gstride) {
            const bool fast = (base >= (int)QTIP_WARMUP_SYMS) && (base + GROUP <= num_symbols);
            if (fast) do_fast(base); else do_general(base);
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        float a = gg_warp_reduce_sum(acc[r]);
        if (SCALE_HOIST) a *= scl[r];
        const int row = row0 + r;
        if (lane == 0 && row < n_rows) y[(size_t)pair * n_rows + row] = gg_from_f32<T>(a);
    }
}

// Achievable-read-bandwidth reference over the SAME pool, same 12.58 MB slices.
__global__ void stream_read_kernel(const uint4* __restrict__ p, long n4, float* out) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0u, 0u, 0u, 0u);
    for (long j = i; j < n4; j += stride) {
        const uint4 v = __ldg(p + j);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xDEADBEEFu) out[0] = 1.0f;
}

// ---------------------------------------------------------------------------
struct Shape { const char* name; int n_rows; int k_in; };
struct Cfg {
    int id; const char* name;
    int warps; int rows_per_warp; int group; int stage_vec; int mlp; int scale_hoist;
    bool staged; int peel; int abl;
};

using LaunchFn = void (*)(const uint8_t*, const float*, const float*, const __nv_bfloat16*,
                          const uint32_t*, __nv_bfloat16*, int, int, int, int, int,
                          unsigned int, cudaStream_t);

template <int WARPS, int RPW, bool STAGED, int GROUP, int SV, int MLP, int SH, int PL, int ABL>
void launch_variant(const uint8_t* packed, const float* rs, const float* lut,
                    const __nv_bfloat16* x, const uint32_t* idx, __nv_bfloat16* y,
                    int n_rows, int ppr, int num_symbols, int n_pairs, int num_experts,
                    unsigned int cb_mult, cudaStream_t stream) {
    constexpr int ROWS_PER_BLOCK = WARPS * RPW;
    const size_t SHMEM = STAGED ? (size_t)ROWS_PER_BLOCK * ppr : 0;
    dim3 grid((n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, n_pairs, 1);
    tg_kernel<__nv_bfloat16, WARPS, RPW, true, STAGED, GROUP, SV, MLP, SH, PL, ABL>
        <<<grid, WARPS * 32, SHMEM, stream>>>(packed, rs, lut, x, idx, y, n_rows, ppr,
                                              num_symbols, n_pairs, num_experts, cb_mult);
}

} // namespace

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    const int   n_pairs     = (argc > 1) ? atoi(argv[1]) : 6;
    const int   iters       = (argc > 2) ? atoi(argv[2]) : 40;
    const int   reps        = (argc > 3) ? atoi(argv[3]) : 7;
    const int   only_cfg    = (argc > 4) ? atoi(argv[4]) : -1;
    const unsigned CB_MULT  = 0xB5DE9C89u;   // odd MCG multiplier (matches Rust rung)

    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    // Peak HBM bytes/s from the *reported* bus width and memory clock.
    const double peak_bw = 2.0 * (double)prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0);
    printf("# GPU %s  SMs=%d  smem/blk(optin)=%zu  peakHBM=%.2f TB/s\n",
           prop.name, prop.multiProcessorCount, (size_t)prop.sharedMemPerBlockOptin, peak_bw / 1e12);

    // The three real V4 decode call sites, plus one deliberately awkward shape:
    // num_symbols=1088 is NOT a multiple of gstride (128 at GROUP=4, 256 at
    // GROUP=8), so the PEEL form's trailing general-path loop actually runs.
    // Without it the tail is never exercised and PEEL would ship untested.
    const Shape shapes[4] = {
        {"gate", 2048, 4096},
        {"up  ", 2048, 4096},
        {"down", 4096, 2048},
        {"edge", 1024, 2176},
    };

    // ---- rotating pool: 256 experts, the real V4 routed-expert count ----
    const int num_experts = 256;
    size_t pool_bytes = 0;
    for (const auto& s : shapes) {
        const int ppr = (s.k_in / (int)QTIP_V) / 2;
        pool_bytes = std::max(pool_bytes, (size_t)num_experts * s.n_rows * ppr);
    }
    printf("# rotating packed pool = %.1f MB (%d experts)\n", pool_bytes / 1e6, num_experts);

    uint8_t* d_packed = nullptr;  CK(cudaMalloc(&d_packed, pool_bytes));
    float*   d_scales = nullptr;  CK(cudaMalloc(&d_scales, (size_t)num_experts * 4096 * sizeof(float)));
    __nv_bfloat16* d_x = nullptr; CK(cudaMalloc(&d_x, (size_t)n_pairs * 4096 * sizeof(__nv_bfloat16)));
    __nv_bfloat16* d_y = nullptr; CK(cudaMalloc(&d_y, (size_t)n_pairs * 4096 * sizeof(__nv_bfloat16)));
    __nv_bfloat16* d_yref = nullptr; CK(cudaMalloc(&d_yref, (size_t)n_pairs * 4096 * sizeof(__nv_bfloat16)));
    float* d_sink = nullptr;      CK(cudaMalloc(&d_sink, sizeof(float)));

    // rotation index table: iters * n_pairs, walking all 256 experts
    const int nrot = iters;
    std::vector<uint32_t> h_idx((size_t)nrot * n_pairs);
    for (int i = 0; i < nrot; ++i)
        for (int j = 0; j < n_pairs; ++j) h_idx[(size_t)i * n_pairs + j] = (uint32_t)((i * n_pairs + j) % num_experts);
    uint32_t* d_idx = nullptr; CK(cudaMalloc(&d_idx, h_idx.size() * sizeof(uint32_t)));
    CK(cudaMemcpy(d_idx, h_idx.data(), h_idx.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // deterministic fill
    {
        std::vector<uint8_t> h(1 << 22);
        uint64_t st = 0x243F6A8885A308D3ull;
        for (size_t off = 0; off < pool_bytes; off += h.size()) {
            for (size_t i = 0; i < h.size(); ++i) {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                h[i] = (uint8_t)(st >> 24);
            }
            CK(cudaMemcpy(d_packed + off, h.data(), std::min(h.size(), pool_bytes - off), cudaMemcpyHostToDevice));
        }
        std::vector<float> hs((size_t)num_experts * 4096);
        for (size_t i = 0; i < hs.size(); ++i) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; hs[i] = 0.002f + 0.001f * (float)((st >> 40) & 0xFF) / 255.f; }
        CK(cudaMemcpy(d_scales, hs.data(), hs.size() * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<__nv_bfloat16> hx((size_t)n_pairs * 4096);
        for (size_t i = 0; i < hx.size(); ++i) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; hx[i] = __float2bfloat16(((float)((st >> 33) & 0xFFFF) / 32768.f) - 1.f); }
        CK(cudaMemcpy(d_x, hx.data(), hx.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    }

    // ---- achievable read bandwidth on this pool (the honest "bound") ----
    double achievable_bw = 0.0;
    {
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        const long n4 = (long)(pool_bytes / 16);
        for (int w = 0; w < 3; ++w) stream_read_kernel<<<prop.multiProcessorCount * 8, 256>>>((const uint4*)d_packed, n4, d_sink);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 5; ++i) stream_read_kernel<<<prop.multiProcessorCount * 8, 256>>>((const uint4*)d_packed, n4, d_sink);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        achievable_bw = 5.0 * (double)pool_bytes / (ms * 1e-3);
        printf("# measured streaming read = %.2f TB/s (%.0f%% of reported peak)\n",
               achievable_bw / 1e12, 100.0 * achievable_bw / peak_bw);
    }

    // ---- variant table ----
    // id, name,                      warps rpw grp sv mlp sh staged
    const Cfg cfgs[] = {
        //  id  name                          W  RPW  G  sv mlp sh staged peel abl
        {  0, "BASELINE (shipped)",           8,  2,  4, 0, 0, 0, true , 0, 0},
        {  3, "+stage_vec +mlp",              8,  2,  4, 1, 1, 0, true , 0, 0},
        { 30, "+sv +mlp +PEEL",               8,  2,  4, 1, 1, 0, true , 1, 0},
        { 31, "+sv +mlp +PEEL G=4 RPW=4",     8,  4,  4, 1, 1, 0, true , 1, 0},
        { 32, "+sv +mlp +PEEL G=8 RPW=4",     8,  4,  8, 1, 1, 0, true , 1, 0},
        { 33, "  ^ +SCALE_HOIST",             8,  4,  8, 1, 1, 1, true , 1, 0},
        { 34, "+sv +mlp +PEEL G=4 RPW=4 +SH", 8,  4,  4, 1, 1, 1, true , 1, 0},
        // ---- ABLATIONS (attribution, not candidates) ----
        { 20, "ABL codebook->2 LOP3",         8,  2,  4, 1, 1, 0, true , 1, 1},
        { 21, "ABL no packed loads",          8,  2,  4, 1, 1, 0, true , 1, 2},
        { 22, "ABL no fma/activation",        8,  2,  4, 1, 1, 0, true , 1, 3},
        { 24, "ABL weights L2-resident",      8,  2,  4, 1, 1, 0, true , 1, 4},
        { 25, "ABL stage only (no loop)",     8,  2,  4, 1, 1, 0, true , 1, 5},
        { 26, "ABL loop+state only",          8,  2,  4, 1, 1, 0, true , 1, 7},
    };
    const int NCFG = (int)(sizeof(cfgs) / sizeof(cfgs[0]));

    auto dispatch = [&](const Cfg& c, const uint32_t* idx, const Shape& s, int ppr, int nsym, cudaStream_t st) -> bool {
        #define D(W, R, SD, G, SV, ML, SH, PL, AB)                                              \
            if (c.warps == W && c.rows_per_warp == R && c.staged == SD && c.group == G &&        \
                c.stage_vec == SV && c.mlp == ML && c.scale_hoist == SH && c.peel == PL &&       \
                c.abl == AB) {                                                                   \
                launch_variant<W, R, SD, G, SV, ML, SH, PL, AB>(d_packed, d_scales, nullptr,     \
                    d_x, idx, d_y, s.n_rows, ppr, nsym, n_pairs, num_experts, CB_MULT, st); return true; }
        D(8,2,true,4,0,0,0,0,0) D(8,2,true,4,1,1,0,0,0)
        D(8,2,true,4,1,1,0,1,0) D(8,4,true,4,1,1,0,1,0) D(8,4,true,8,1,1,0,1,0)
        D(8,4,true,8,1,1,1,1,0) D(8,4,true,4,1,1,1,1,0)
        D(8,2,true,4,1,1,0,1,1) D(8,2,true,4,1,1,0,1,2) D(8,2,true,4,1,1,0,1,3)
        D(8,2,true,4,1,1,0,1,4) D(8,2,true,4,1,1,0,1,5) D(8,2,true,4,1,1,0,1,7)
        #undef D
        return false;
    };

    cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    std::vector<uint8_t> hy((size_t)n_pairs * 4096 * sizeof(__nv_bfloat16));
    std::vector<uint8_t> hyref(hy.size());

    for (const auto& s : shapes) {
        const int nsym = s.k_in / (int)QTIP_V;
        const int ppr  = nsym / 2;
        const double bytes_per_call = (double)n_pairs * s.n_rows * ppr;
        const double bound_us = bytes_per_call / achievable_bw * 1e6;
        printf("\n=== %s  n_rows=%d k_in=%d  ppr=%d B  pairs=%d | %.2f MB/call  bound(@%.2f TB/s)=%.2f us ===\n",
               s.name, s.n_rows, s.k_in, ppr, n_pairs, bytes_per_call / 1e6, achievable_bw / 1e12, bound_us);
        printf("%-32s %10s %10s %8s %8s %7s\n", "variant", "us/call", "GB/s", "x-bound", "speedup", "bitcmp");

        double base_us = 0.0;
        for (int ci = 0; ci < NCFG; ++ci) {
            const Cfg& c = cfgs[ci];
            if (only_cfg >= 0 && c.id != only_cfg) continue;
            const size_t smem = c.staged ? (size_t)c.warps * c.rows_per_warp * ppr : 0;
            if (smem > 48 * 1024) { printf("%-32s %10s (smem %zu B > 48K)\n", c.name, "SKIP", smem); continue; }

            CK(cudaMemset(d_y, 0, (size_t)n_pairs * 4096 * sizeof(__nv_bfloat16)));
            if (!dispatch(c, d_idx, s, ppr, nsym, 0)) { printf("%-32s NO DISPATCH\n", c.name); continue; }
            cudaError_t le = cudaGetLastError();
            if (le != cudaSuccess) { printf("%-32s LAUNCH FAIL: %s\n", c.name, cudaGetErrorString(le)); continue; }
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hy.data(), d_y, hy.size(), cudaMemcpyDeviceToHost));
            if (c.id == 0) { hyref = hy; CK(cudaMemcpy(d_yref, d_y, hy.size(), cudaMemcpyDeviceToDevice)); }
            const bool bitsame = (memcmp(hy.data(), hyref.data(), (size_t)n_pairs * s.n_rows * 2) == 0);

            // warm
            for (int i = 0; i < 4; ++i) dispatch(c, d_idx + (size_t)(i % nrot) * n_pairs, s, ppr, nsym, 0);
            CK(cudaDeviceSynchronize());

            std::vector<double> samples;
            for (int r = 0; r < reps; ++r) {
                CK(cudaEventRecord(ev0));
                for (int i = 0; i < iters; ++i) dispatch(c, d_idx + (size_t)i * n_pairs, s, ppr, nsym, 0);
                CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
                float ms = 0; CK(cudaEventElapsedTime(&ms, ev0, ev1));
                samples.push_back((double)ms * 1000.0 / iters);
            }
            std::sort(samples.begin(), samples.end());
            const double us = samples[samples.size() / 2];
            const double gbs = bytes_per_call / (us * 1e-6) / 1e9;
            if (c.id == 0) base_us = us;
            printf("%-32s %10.2f %10.1f %8.1f %8.2fx %7s\n", c.name, us, gbs, us / bound_us,
                   base_us > 0 ? base_us / us : 1.0, bitsame ? "SAME" : "DIFF");
        }
    }

    // ---- NEGATIVE CONTROL: 1-ULP perturbation of one activation must show DIFF ----
    {
        const Shape& s = shapes[0];
        const int nsym = s.k_in / (int)QTIP_V, ppr = nsym / 2;
        const Cfg& c = cfgs[0];
        CK(cudaMemset(d_y, 0, (size_t)n_pairs * 4096 * sizeof(__nv_bfloat16)));
        dispatch(c, d_idx, s, ppr, nsym, 0); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hyref.data(), d_y, hyref.size(), cudaMemcpyDeviceToHost));

        unsigned short bits = 0;
        CK(cudaMemcpy(&bits, d_x, 2, cudaMemcpyDeviceToHost));
        const unsigned short bumped = (unsigned short)(bits + 1);
        CK(cudaMemcpy(d_x, &bumped, 2, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_y, 0, (size_t)n_pairs * 4096 * sizeof(__nv_bfloat16)));
        dispatch(c, d_idx, s, ppr, nsym, 0); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hy.data(), d_y, hy.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(d_x, &bits, 2, cudaMemcpyHostToDevice));

        const bool differs = (memcmp(hy.data(), hyref.data(), (size_t)n_pairs * s.n_rows * 2) != 0);
        printf("\n[negative control] x[0] 0x%04X -> 0x%04X : comparator reports %s  => %s\n",
               bits, bumped, differs ? "DIFF" : "SAME", differs ? "COMPARATOR IS LIVE" : "COMPARATOR IS DEAD (FATAL)");
        if (!differs) { printf("FATAL: byte comparator cannot see a 1-ULP input change.\n"); return 3; }
    }
    printf("\nOK\n");
    return 0;
}
