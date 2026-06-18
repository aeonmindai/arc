// QTIP on-device MoE gather + fused decode + gemv (capturable decode path).
//
// This is the CUDA-graph-capturable sibling of `qtip_fused_gemv` in
// `qtip_gemv.cu`. The existing MoE dispatch (`gather_forward_cuda` in mod.rs)
// reads the routing indices to the HOST (`indices.to_device(Cpu).to_vec1()`)
// to decide which experts to dequantize — a device->host sync that aborts any
// CUDA stream capture. This kernel instead reads each (token, slot)'s expert
// id ON-DEVICE (`__ldg(&indices[pair])`) and runs the trellis gemv directly
// against that expert's packed rows, so the whole MoE stays on the stream and
// the decode forward becomes capturable.
//
// Scope: the b=1 (and small-batch / MTP-draft) DECODE regime, where there are
// few (token, slot) pairs. It does one independent gemv per pair, so it does
// NOT reuse a dequantized expert across many tokens — that token-grouping win
// only matters at prefill, which keeps the existing grouped path. For decode
// (n_pairs = n_tokens * top_k, e.g. 1 * 6 = 6) per-pair gemv is optimal and,
// crucially, sync-free.
//
// Layout (3-D stacked experts), matching QtipLayer 3-D mode:
//   * packed     : [E, n_rows, packed_per_row]  u8   (flat: expert*n_rows+row)
//   * row_scales : [E, n_rows]                   f32  (flat: expert*n_rows+row)
//   * lut        : [2^L * V]                     f32  (shared across experts)
//   * x          : [n_pairs, k_in]               T    (already rotated)
//   * indices    : [n_pairs]                     u32  (expert id per pair)
//   * y          : [n_pairs, n_rows]             T
//
// All trellis/correctness invariants are identical to qtip_gemv.cu (state
// recurrence, init state 0, low/high nibble pack order, per-row scale, warmup).
// Activation pre-rotation (D.H.D) must be applied to x BEFORE this kernel.
//
// SM80+ (uses __nv_bfloat16). Gated by `has_qtip_kernels` in build.rs.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr uint32_t QTIP_K          = 4;
constexpr uint32_t QTIP_L          = 16;
constexpr uint32_t QTIP_V          = 2;
constexpr uint32_t QTIP_STATE_MASK = (1u << QTIP_L) - 1u;
constexpr uint32_t QTIP_WARMUP_SYMS = QTIP_L / QTIP_K;

template <typename T>
__device__ __forceinline__ T gg_from_f32(float v);
template <>
__device__ __forceinline__ float gg_from_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half gg_from_f32<__half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 gg_from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ float gg_to_f32(T v);
template <>
__device__ __forceinline__ float gg_to_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ float gg_to_f32<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float gg_to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float gg_warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

// Load two consecutive activation elements as one vectorized load (halves the
// LSU instruction count on the x reads). Overloaded per dtype.
__device__ __forceinline__ float2 gg_load2(const float* p) {
    return __ldg(reinterpret_cast<const float2*>(p));
}
__device__ __forceinline__ float2 gg_load2(const __half* p) {
    return __half22float2(__ldg(reinterpret_cast<const __half2*>(p)));
}
__device__ __forceinline__ float2 gg_load2(const __nv_bfloat16* p) {
    return __bfloat1622float2(__ldg(reinterpret_cast<const __nv_bfloat162*>(p)));
}

// One block per (output row, pair). Reads the pair's expert id on-device,
// offsets into that expert's packed rows / scales, and accumulates
// y[pair, row] = sum_k W_expert[row,k] * x[pair,k].
//
// Grid:  (n_rows, n_pairs, 1)
// Block: (THREADS, 1, 1)
template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS)
qtip_gather_gemv_v2_k4_l16_kernel(
    const uint8_t*  __restrict__ packed,      // [E, n_rows, packed_per_row]
    const float*    __restrict__ row_scales,  // [E, n_rows]
    const float*    __restrict__ lut,         // [2^L * V]
    const T*        __restrict__ x,           // [n_pairs, k_in]  (rotated)
    const uint32_t* __restrict__ indices,     // [n_pairs]
    T*              __restrict__ y,           // [n_pairs, n_rows]
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts
) {
    const int row  = blockIdx.x;
    const int pair = blockIdx.y;
    if (row >= n_rows || pair >= n_pairs) return;

    const uint32_t expert = __ldg(indices + pair);
    // Defensive: an out-of-range expert id (shouldn't happen from a correct
    // router) writes 0 rather than reading out of bounds.
    if (expert >= (uint32_t)num_experts) {
        if (threadIdx.x == 0) y[(size_t)pair * n_rows + row] = gg_from_f32<T>(0.0f);
        return;
    }

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int N_WARPS = THREADS / 32;
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of warp size");

    const size_t expert_row = (size_t)expert * n_rows + row;
    const uint8_t* row_packed = packed + expert_row * packed_per_row;
    const float scale = __ldg(row_scales + expert_row);
    const T* x_pair = x + (size_t)pair * num_symbols * QTIP_V;

    const int sym_per_thread = (num_symbols + THREADS - 1) / THREADS;
    const int sym_start_raw  = tid * sym_per_thread;
    const int sym_end        = min(num_symbols, sym_start_raw + sym_per_thread);

    float acc = 0.0f;

    if (sym_start_raw < num_symbols) {
        uint32_t state = 0;
        int sym_idx = sym_start_raw;

        if (tid > 0) {
            int warm_start = max(0, sym_start_raw - (int)QTIP_WARMUP_SYMS);
            for (int t = warm_start; t < sym_start_raw; ++t) {
                int byte_idx = t >> 1;
                uint8_t byte = __ldg(row_packed + byte_idx);
                uint32_t sym = (t & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
                state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;
            }
        }

        while (sym_idx < sym_end) {
            int byte_idx = sym_idx >> 1;
            uint8_t byte = __ldg(row_packed + byte_idx);

            bool do_lo = ((sym_idx & 1) == 0);
            uint32_t sym_lo = byte & 0x0F;
            uint32_t sym_hi = (byte >> 4) & 0x0F;

            if (do_lo) {
                state = ((state << QTIP_K) | sym_lo) & QTIP_STATE_MASK;
                const float* lut_p = lut + (size_t)state * QTIP_V;
                float w0 = __ldg(lut_p)     * scale;
                float w1 = __ldg(lut_p + 1) * scale;
                int x_off = sym_idx * (int)QTIP_V;
                float x0 = gg_to_f32<T>(__ldg(x_pair + x_off + 0));
                float x1 = gg_to_f32<T>(__ldg(x_pair + x_off + 1));
                acc = fmaf(w0, x0, acc);
                acc = fmaf(w1, x1, acc);
                ++sym_idx;
                if (sym_idx >= sym_end) break;
            }

            state = ((state << QTIP_K) | sym_hi) & QTIP_STATE_MASK;
            const float* lut_p_hi = lut + (size_t)state * QTIP_V;
            float w0 = __ldg(lut_p_hi)     * scale;
            float w1 = __ldg(lut_p_hi + 1) * scale;
            int x_off = sym_idx * (int)QTIP_V;
            float x0 = gg_to_f32<T>(__ldg(x_pair + x_off + 0));
            float x1 = gg_to_f32<T>(__ldg(x_pair + x_off + 1));
            acc = fmaf(w0, x0, acc);
            acc = fmaf(w1, x1, acc);
            ++sym_idx;
        }
    }

    acc = gg_warp_reduce_sum(acc);

    __shared__ float warp_sums[N_WARPS];
    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < N_WARPS) ? warp_sums[lane] : 0.0f;
        v = gg_warp_reduce_sum(v);
        if (lane == 0) {
            y[(size_t)pair * n_rows + row] = gg_from_f32<T>(v);
        }
    }
}

// Warp-per-row strided decode + shared-memory LUT (RUN-161 rewrite v4). One WARP
// computes one output row; its 32 lanes decode symbols STRIDED (lane i -> i,
// i+32, ...), each symbol's L-bit state recovered independently by replaying the
// last QTIP_WARMUP_SYMS symbols. The strided/independent form is deliberate:
// H200 ncu showed this kernel is bound by long_scoreboard (global LOAD LATENCY
// on the per-symbol LUT gather, ~49 cyc/issue) and lg_throttle (~28). Strided
// gives memory-level parallelism (many independent loads in flight) that hides
// latency; the contiguous-segment variant serialized the state->LUT-address
// chain and tanked inst/cyc from 0.7 to 0.15.
//
// THE win here: stage the LUT (2^L * V floats = 2 KB at L=8) into SHARED memory
// once per block. Every per-symbol weight lookup then hits shared (short
// scoreboard, ~tens of cyc) instead of global (long scoreboard, hundreds), and
// the global LUT load that drove lg_throttle disappears. Plus paired loads:
// float2 LUT (both weights, one access) and gg_load2 x (x0,x1, one load).
//
// Grid:  (ceil(n_rows / (WARPS_PER_BLOCK*ROWS_PER_WARP)), n_pairs, 1)
// Block: (WARPS_PER_BLOCK * 32, 1, 1)
// Shared: (2^L * V) floats
//
// RUN-161 rewrite v5: REGISTER BLOCKING over output rows (ROWS_PER_WARP). One
// warp now decodes ROWS_PER_WARP rows of the SAME expert. The activation x_pair
// is identical for every row of an expert, so each lane loads its GROUP of x
// ONCE per group-step and reuses it across all R rows -> R independent decode
// streams in flight. ncu on v4 showed the kernel at inst/cyc 0.71 (of 4), sm
// 66%, NOT bandwidth-bound (dram 2-4%): the limiter is issue/latency, not FLOPs
// or bytes. Roofline says a b=1 GEMV SHOULD be memory-bound here (~10 ops/weight
// vs a ~10^4 op/weight compute budget). R independent rows give the ILP to hide
// long_scoreboard (warm-up global reads) + wait (state-carry dep) and push
// inst/cyc up toward peak, where memory finally becomes the wall.
template <typename T, int WARPS_PER_BLOCK, int ROWS_PER_WARP>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
qtip_gather_gemv_warp_kernel(
    const uint8_t*  __restrict__ packed,
    const float*    __restrict__ row_scales,
    const float*    __restrict__ lut,
    const T*        __restrict__ x,
    const uint32_t* __restrict__ indices,
    T*              __restrict__ y,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts,
    int stage_packed   // 1 => block's packed rows staged to shared (set by launcher)
) {
    // Shared layout: [LUT floats][packed bytes for this block's rows].
    //   - LUT staged when it fits (L=8 -> 2 KB; L>=~13 falls back to global LUT).
    //   - RUN-161 v6: PACKED WEIGHT STAGING. ncu on v5 showed the dominant stall
    //     is long_scoreboard ~5.85 (global latency on the per-symbol + warm-up
    //     packed byte reads) -- NOT the LUT (now shared) and NOT bandwidth (dram
    //     4%). So bulk-copy this block's ROWS_PER_BLOCK packed rows into shared
    //     ONCE (coalesced, large transactions = near-peak for that traffic), then
    //     every warm-up replay + per-symbol read hits shared. This converts the
    //     scattered, warm-up-amplified global loads into cheap shared reads and
    //     collapses long_scoreboard. The global weight bytes are read exactly
    //     once instead of ~1.5x (warm-up) and fully coalesced.
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    extern __shared__ float s_mem[];
    constexpr int LUT_FLOATS = (1 << QTIP_L) * QTIP_V;
    constexpr bool USE_SHARED_LUT = (LUT_FLOATS * (int)sizeof(float)) <= (48 * 1024);

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int pair = blockIdx.y;
    const int row0_block = blockIdx.x * ROWS_PER_BLOCK;
    if (row0_block >= n_rows || pair >= n_pairs) return;

    const uint32_t expert = __ldg(indices + pair);
    const bool invalid = expert >= (uint32_t)num_experts;

    // ---- stage LUT ----
    const float2* s_lut2;
    if (USE_SHARED_LUT) {
        for (int i = threadIdx.x; i < LUT_FLOATS; i += WARPS_PER_BLOCK * 32) {
            s_mem[i] = __ldg(lut + i);  // s_mem front aliased as float LUT
        }
        s_lut2 = reinterpret_cast<const float2*>(s_mem);
    } else {
        s_lut2 = reinterpret_cast<const float2*>(lut);
    }

    // ---- stage packed weight rows (coalesced bulk copy) ----
    uint8_t* s_packed = reinterpret_cast<uint8_t*>(s_mem + (USE_SHARED_LUT ? LUT_FLOATS : 0));
    if (stage_packed && !invalid) {
        const int n_block_rows = min(ROWS_PER_BLOCK, n_rows - row0_block);
        const long total = (long)n_block_rows * packed_per_row;
        const uint8_t* __restrict__ g =
            packed + ((size_t)expert * n_rows + row0_block) * packed_per_row;
        for (long i = threadIdx.x; i < total; i += WARPS_PER_BLOCK * 32) {
            s_packed[i] = __ldg(g + i);
        }
    }
    if (USE_SHARED_LUT || (stage_packed && !invalid)) __syncthreads();

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

    // Per-row backing: packed pointer (shared if staged, else global) + scale.
    // Out-of-range rows get scale 0 so their accumulator stays 0 (not written).
    const uint8_t* rp[ROWS_PER_WARP];
    float scl[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = row0 + r;
        const bool ok = row < n_rows;
        const size_t er = (size_t)expert * n_rows + (ok ? row : 0);
        rp[r]  = stage_packed ? (s_packed + (size_t)(warp * ROWS_PER_WARP + r) * packed_per_row)
                              : (packed + er * packed_per_row);
        scl[r] = ok ? __ldg(row_scales + er) : 0.0f;
    }
    const T* __restrict__ x_pair = x + (size_t)pair * num_symbols * QTIP_V;

    // Grouped-strided decode (GROUP=4: H200 sweep G=1/2/4/8 -> 137/233/325/266
    // GB/s at R=1). Each lane does GROUP contiguous symbols per step, groups
    // strided by 32*GROUP. x for the group is loaded ONCE and reused across all
    // ROWS_PER_WARP rows; the R rows' decode chains are mutually independent,
    // providing the ILP that hides warm-up/state-carry latency.
    constexpr int GROUP = 4;
    const int gstride = 32 * GROUP;

    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    for (int base = lane * GROUP; base < num_symbols; base += gstride) {
        const int gend = min(base + GROUP, num_symbols);
        // Load this lane's GROUP of activations once (shared by all R rows).
        float2 xg[GROUP];
        #pragma unroll
        for (int j = 0; j < GROUP; ++j) {
            const int s = base + j;
            if (s < gend) xg[j] = gg_load2(x_pair + s * (int)QTIP_V);
        }
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const uint8_t* row_packed = rp[r];
            const float scale = scl[r];
            // Warm up the masked L-bit state at `base` for this row.
            uint32_t state = 0u;
            const int w0 = base > (int)QTIP_WARMUP_SYMS ? base - (int)QTIP_WARMUP_SYMS : 0;
            for (int t = w0; t < base; ++t) {
                const uint8_t b = row_packed[t >> 1];
                const uint32_t sym = (t & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
                state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;
            }
            float a = acc[r];
            #pragma unroll
            for (int j = 0; j < GROUP; ++j) {
                const int s = base + j;
                if (s >= gend) break;
                const uint8_t b = row_packed[s >> 1];
                const uint32_t sym = (s & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
                state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;
                const float2 w = s_lut2[state];
                a = fmaf(w.x * scale, xg[j].x, a);
                a = fmaf(w.y * scale, xg[j].y, a);
            }
            acc[r] = a;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const float a = gg_warp_reduce_sum(acc[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_rows) y[(size_t)pair * n_rows + row] = gg_from_f32<T>(a);
    }
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

void launch_qtip_gather_gemv_v2_k4_l16_bf16(
    const uint8_t*       d_packed,
    const float*         d_row_scales,
    const float*         d_lut,
    const __nv_bfloat16* d_x_rotated,
    const uint32_t*      d_indices,
    __nv_bfloat16*       d_y,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts,
    cudaStream_t         stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int ROWS_PER_WARP = 2;   // register-blocking: rows/warp (x reused)
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    constexpr size_t LUT_BYTES = (size_t(1) << QTIP_L) * QTIP_V * sizeof(float);
    const size_t lut_smem    = (LUT_BYTES <= 48 * 1024) ? LUT_BYTES : 0;          // shared LUT only when it fits
    const size_t packed_smem = (size_t)ROWS_PER_BLOCK * packed_per_row;
    const bool   stage_packed = (lut_smem + packed_smem) <= 48 * 1024;            // stage weights when LUT+packed fit
    const size_t SHMEM = stage_packed ? (lut_smem + packed_smem) : lut_smem;
    dim3 grid((n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, n_pairs, 1);
    qtip_gather_gemv_warp_kernel<__nv_bfloat16, WARPS_PER_BLOCK, ROWS_PER_WARP>
        <<<grid, WARPS_PER_BLOCK * 32, SHMEM, stream>>>(
            d_packed, d_row_scales, d_lut, d_x_rotated, d_indices, d_y,
            n_rows, packed_per_row, num_symbols, n_pairs, num_experts, stage_packed ? 1 : 0);
}

void launch_qtip_gather_gemv_v2_k4_l16_f16(
    const uint8_t* d_packed,
    const float*   d_row_scales,
    const float*   d_lut,
    const __half*  d_x_rotated,
    const uint32_t* d_indices,
    __half*        d_y,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts,
    cudaStream_t   stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int ROWS_PER_WARP = 2;   // register-blocking: rows/warp (x reused)
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    constexpr size_t LUT_BYTES = (size_t(1) << QTIP_L) * QTIP_V * sizeof(float);
    const size_t lut_smem    = (LUT_BYTES <= 48 * 1024) ? LUT_BYTES : 0;          // shared LUT only when it fits
    const size_t packed_smem = (size_t)ROWS_PER_BLOCK * packed_per_row;
    const bool   stage_packed = (lut_smem + packed_smem) <= 48 * 1024;            // stage weights when LUT+packed fit
    const size_t SHMEM = stage_packed ? (lut_smem + packed_smem) : lut_smem;
    dim3 grid((n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, n_pairs, 1);
    qtip_gather_gemv_warp_kernel<__half, WARPS_PER_BLOCK, ROWS_PER_WARP>
        <<<grid, WARPS_PER_BLOCK * 32, SHMEM, stream>>>(
            d_packed, d_row_scales, d_lut, d_x_rotated, d_indices, d_y,
            n_rows, packed_per_row, num_symbols, n_pairs, num_experts, stage_packed ? 1 : 0);
}

void launch_qtip_gather_gemv_v2_k4_l16_f32(
    const uint8_t* d_packed,
    const float*   d_row_scales,
    const float*   d_lut,
    const float*   d_x_rotated,
    const uint32_t* d_indices,
    float*         d_y,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts,
    cudaStream_t   stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int ROWS_PER_WARP = 2;   // register-blocking: rows/warp (x reused)
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    constexpr size_t LUT_BYTES = (size_t(1) << QTIP_L) * QTIP_V * sizeof(float);
    const size_t lut_smem    = (LUT_BYTES <= 48 * 1024) ? LUT_BYTES : 0;          // shared LUT only when it fits
    const size_t packed_smem = (size_t)ROWS_PER_BLOCK * packed_per_row;
    const bool   stage_packed = (lut_smem + packed_smem) <= 48 * 1024;            // stage weights when LUT+packed fit
    const size_t SHMEM = stage_packed ? (lut_smem + packed_smem) : lut_smem;
    dim3 grid((n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, n_pairs, 1);
    qtip_gather_gemv_warp_kernel<float, WARPS_PER_BLOCK, ROWS_PER_WARP>
        <<<grid, WARPS_PER_BLOCK * 32, SHMEM, stream>>>(
            d_packed, d_row_scales, d_lut, d_x_rotated, d_indices, d_y,
            n_rows, packed_per_row, num_symbols, n_pairs, num_experts, stage_packed ? 1 : 0);
}

} // extern "C"
