// QTIP bitshift-trellis (computed codebook) CUDA kernels — the `qtip2b` rung.
//
// Format: V=1, K=2, L=16. Each row stores `num_symbols/4` packed bytes
// (four 2-bit symbols per byte, LSB-first). The trellis state updates as
// `state = ((state << 2) | sym) & 0xFFFF`. There is NO LUT: the codeword is
// a pure function of the state,
//
//     x = state * MCG_MULT            // 32-bit wrapping multiply (IMAD)
//     m = (x & 0x8FFF8FFF) ^ 0x3B603B60   // one LOP3
//     w = f32(fp16(m >> 16)) + f32(fp16(m & 0xFFFF))
//
// evaluated in registers. This is the whole point of the format: the LUT
// rung's decode GEMV measured instruction/latency bound on H200 (~8% of HBM
// peak; ncu blamed the data-dependent 512 KB LUT gather + the serial state
// chain). Computing the codeword costs ~5 ALU ops with NO memory access, so
// the kernel's only weight-side traffic is the 2-bit packed bytes — and the
// ILP needed to hide the state-chain dependency comes from decoding
// ROWS_PER_WARP independent rows per warp (R independent trellis streams in
// flight per thread).
//
// The multiplier (0xCAF6A435 by default, passed from the Rust side so a
// retuned constant round-trips through UQFF) is the spectrally-optimized
// MCG from exllamav3 PR #26 — one fewer instruction than Cornell QTIP's LCG
// 3INST and 4-8% better 2-bit KLD.
//
// Parity contract with the CPU reference (`qtip/bitshift.rs::mcg_codeword`):
// the two fp16 halves are summed in **f32** (H2F conversions are exact, the
// f32 add is correctly rounded) — NOT with __hadd, whose single-rounded fp16
// sum can differ from a double-rounded emulation in rare ties. --use_fast_math
// does not affect integer ops, H2F conversions, or IEEE f32 add/mul, so the
// decode is bit-identical to the host.
//
// Kernel groups:
//   1) qtip2b_dequantize_*      — packed -> dtype matrix (rotated frame).
//   2) qtip2b_gemv_*            — fused decode + GEMV, warp-per-rows with
//      register blocking (R independent streams). `indices == nullptr`
//      selects the plain 2-D path (expert 0); non-null gathers each pair's
//      expert id ON-DEVICE (CUDA-graph capturable MoE dispatch).
//   3) qtip2b_compute_row_scales / quantize greedy / quantize Viterbi —
//      quantize-side. The Viterbi kernel is the prefix-grouped DP from
//      qtip_quantize.cu re-parameterized for K=2: PREFIX_COUNT = 2^14 =
//      16384, so the per-prefix cost table (64 KiB f32) no longer fits
//      static shared memory — it lives in a global scratch buffer instead,
//      and per-prefix backtrace bytes are written straight to gmem.
//
// Rotation kernels are NOT duplicated here: the block-diagonal D·H·D
// launchers in qtip_dequantize.cu / qtip_quantize.cu are format-agnostic and
// are reused by the Rust side for qtip2b as well.
//
// SM80+ (uses __nv_bfloat16). Gated by `has_qtip_kernels` in build.rs.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Shared 3INST decode + dtype helpers (also used by qtip_grouped_gemm.cu).
#include "qtip2b_common.cuh"
// Per-operation IEEE f32 for the trellis search (--use_fast_math would
// otherwise contract `d*d + cost` into an FMA and turn `1.0f/scale` into a
// reciprocal approximation, silently diverging from the Rust reference).
#include "qtip_exact_fp.cuh"

namespace {

// ---------------------------------------------------------------------------
// 1) Dequantize kernel: one thread per row (trellis walk is sequential).
// ---------------------------------------------------------------------------
template <typename T, int THREADS>
__global__ void qtip2b_dequantize_kernel(
    const uint8_t* __restrict__ packed,       // [n_rows, packed_per_row]
    const float*   __restrict__ row_scales,   // [n_rows]
    T*             __restrict__ out,          // [n_rows, num_symbols]
    int n_rows,
    int packed_per_row,
    int num_symbols,
    uint32_t mult
) {
    int row = blockIdx.x * THREADS + threadIdx.x;
    if (row >= n_rows) return;

    const uint8_t* my_packed = packed + (size_t)row * packed_per_row;
    T* my_out                = out + (size_t)row * num_symbols;
    const float scale        = row_scales[row];

    uint32_t state = 0;
    #pragma unroll 2
    for (int b = 0; b < packed_per_row; ++b) {
        uint8_t byte = my_packed[b];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t sym = ((uint32_t)byte >> (j * 2)) & 0x3u;
            state = ((state << Q2B_K) | sym) & Q2B_STATE_MASK;
            my_out[b * 4 + j] = q2b_from_f32<T>(q2b_decode(state, mult) * scale);
        }
    }
}

// ---------------------------------------------------------------------------
// 2) Fused decode + GEMV, warp-per-rows with register blocking.
// ---------------------------------------------------------------------------
//
// Structure follows the LUT rung's `qtip_gather_gemv_warp_kernel`
// (qtip_gather_gemv.cu), with the LUT gather replaced by `q2b_decode` and
// V=1 activation loads:
//   * One WARP owns ROWS_PER_WARP output rows of the SAME expert. The
//     activation group is loaded once per lane and reused across all R rows
//     — the R decode chains are mutually independent, which is exactly the
//     ILP that hides the state-carry dependency (the LUT rung measured
//     issue-bound at inst/cyc 0.71 with a single chain).
//   * Lanes process GROUP=8 contiguous symbols (2 packed bytes) strided by
//     32*GROUP; each lane re-derives its L-bit state with an 8-symbol
//     warm-up replay (byte-aligned: base and base-8 are multiples of 8).
//   * Optional staging of the block's packed rows into shared memory turns
//     the warm-up-amplified scattered byte reads into shared-mem reads
//     (same v6 optimization as the LUT rung).
//   * `indices == nullptr` -> plain 2-D layer: every pair reads expert 0.
//
// TUNING (deferred to hardware): GROUP/ROWS_PER_WARP were swept on the LUT
// rung (H200: G=4 best of 1/2/4/8 at R=1); qtip2b has half the bytes/symbol
// and no LUT latency, so the optimum will differ. Start G=8 (one 64-bit
// aligned 2-byte read per lane-step), R=2, and re-sweep with ncu.
//
// Grid:  (ceil(n_rows / (WARPS_PER_BLOCK*ROWS_PER_WARP)), n_pairs, 1)
// Block: (WARPS_PER_BLOCK * 32, 1, 1)
// Shared: staged packed rows when they fit (<= 48 KiB), else none.
template <typename T, int WARPS_PER_BLOCK, int ROWS_PER_WARP>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
qtip2b_gemv_warp_kernel(
    const uint8_t*  __restrict__ packed,      // [E, n_rows, packed_per_row]
    const float*    __restrict__ row_scales,  // [E, n_rows]
    const T*        __restrict__ x,           // [n_pairs, num_symbols] (rotated)
    const uint32_t* __restrict__ indices,     // [n_pairs] or nullptr (2-D)
    T*              __restrict__ y,           // [n_pairs, n_rows]
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts,
    uint32_t mult,
    int stage_packed
) {
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    extern __shared__ float q2b_smem[];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int pair = blockIdx.y;
    const int row0_block = blockIdx.x * ROWS_PER_BLOCK;
    if (row0_block >= n_rows || pair >= n_pairs) return;

    const uint32_t expert = (indices != nullptr) ? __ldg(indices + pair) : 0u;
    const bool invalid = expert >= (uint32_t)num_experts;

    // ---- stage this block's packed rows (coalesced bulk copy) ----
    uint8_t* s_packed = reinterpret_cast<uint8_t*>(q2b_smem);
    if (stage_packed && !invalid) {
        const int n_block_rows = min(ROWS_PER_BLOCK, n_rows - row0_block);
        const long total = (long)n_block_rows * packed_per_row;
        const uint8_t* __restrict__ g =
            packed + ((size_t)expert * n_rows + row0_block) * packed_per_row;
        for (long i = threadIdx.x; i < total; i += WARPS_PER_BLOCK * 32) {
            s_packed[i] = __ldg(g + i);
        }
        __syncthreads();
    }

    const int row0 = row0_block + warp * ROWS_PER_WARP;
    if (invalid) {
        // Out-of-range expert id from a broken router: write zeros, never
        // read out of bounds.
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const int row = row0 + r;
                if (row < n_rows) y[(size_t)pair * n_rows + row] = q2b_from_f32<T>(0.0f);
            }
        }
        return;
    }

    // Per-row packed pointer (shared if staged) + scale. Out-of-range rows
    // get scale 0 so their accumulator stays 0 (and is never written).
    const uint8_t* rp[ROWS_PER_WARP];
    float scl[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = row0 + r;
        const bool ok = row < n_rows;
        const size_t er = (size_t)expert * n_rows + (ok ? row : 0);
        rp[r]  = stage_packed
                     ? (s_packed + (size_t)(warp * ROWS_PER_WARP + r) * packed_per_row)
                     : (packed + er * packed_per_row);
        scl[r] = ok ? __ldg(row_scales + er) : 0.0f;
    }
    const T* __restrict__ x_pair = x + (size_t)pair * num_symbols;

    // GROUP contiguous symbols per lane-step (byte-aligned: GROUP=8 syms =
    // 2 bytes; warm-up window is also 8 symbols = 2 bytes).
    constexpr int GROUP = 8;
    const int gstride = 32 * GROUP;

    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    for (int base = lane * GROUP; base < num_symbols; base += gstride) {
        const int gend = min(base + GROUP, num_symbols);
        // Load this lane's GROUP of activations once (shared by all R rows).
        float xg[GROUP];
        #pragma unroll
        for (int j = 0; j < GROUP; ++j) {
            const int s = base + j;
            if (s < gend) xg[j] = q2b_to_f32<T>(__ldg(x_pair + s));
        }
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const uint8_t* row_packed = rp[r];
            const float scale = scl[r];
            // Warm up the masked L-bit state at `base` for this row. Only
            // the last L/K = 8 symbols influence the state.
            uint32_t state = 0u;
            const int w0 = base > (int)Q2B_WARMUP_SYMS ? base - (int)Q2B_WARMUP_SYMS : 0;
            for (int t = w0; t < base; ++t) {
                state = ((state << Q2B_K) | q2b_sym(row_packed, t)) & Q2B_STATE_MASK;
            }
            float a = acc[r];
            #pragma unroll
            for (int j = 0; j < GROUP; ++j) {
                const int s = base + j;
                if (s >= gend) break;
                state = ((state << Q2B_K) | q2b_sym(row_packed, s)) & Q2B_STATE_MASK;
                const float w = q2b_decode(state, mult);
                a = fmaf(w * scale, xg[j], a);
            }
            acc[r] = a;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const float a = q2b_warp_reduce_sum(acc[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_rows) y[(size_t)pair * n_rows + row] = q2b_from_f32<T>(a);
    }
}

// ---------------------------------------------------------------------------
// 3a) Per-row scale: scale = max(|row|) / divisor (1.0 when max == 0).
// ---------------------------------------------------------------------------
constexpr int Q2B_SCALE_THREADS = 256;

__global__ void qtip2b_row_scale_kernel(
    const float* __restrict__ weight,      // [n_rows, in_features]
    float*       __restrict__ row_scales,  // [n_rows]
    int in_features,
    float divisor
) {
    __shared__ float smax[Q2B_SCALE_THREADS];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* my_row = weight + (size_t)row * in_features;

    float local_max = 0.0f;
    for (int i = tid; i < in_features; i += Q2B_SCALE_THREADS) {
        float a = fabsf(my_row[i]);
        if (a > local_max) local_max = a;
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = Q2B_SCALE_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = smax[tid + stride];
            if (other > smax[tid]) smax[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = smax[0];
        row_scales[row] = (m == 0.0f) ? 1.0f : (m / divisor);
    }
}

// ---------------------------------------------------------------------------
// 3b) Greedy quantize: one warp per row; 4 lanes evaluate the 4 candidates.
// ---------------------------------------------------------------------------
constexpr int Q2B_GREEDY_THREADS = 32;

__global__ void qtip2b_quantize_rows_greedy_kernel(
    const float* __restrict__ weight,        // [n_rows, in_features]
    const float* __restrict__ row_scales,    // [n_rows]
    uint8_t*     __restrict__ packed,        // [n_rows, num_symbols / 4]
    int in_features,
    int num_symbols,
    uint32_t mult
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* my_row = weight + (size_t)row * in_features;
    uint8_t*     my_pkd = packed + (size_t)row * (num_symbols / 4);

    const float inv_scale = 1.0f / row_scales[row];

    uint32_t state = 0u;
    uint32_t my_sym = (uint32_t)tid;  // lanes 0..3 do the work
    uint8_t pending = 0u;

    for (int t = 0; t < num_symbols; ++t) {
        float target = my_row[t] * inv_scale;

        float err = INFINITY;
        if (tid < (int)Q2B_ALPHABET) {
            uint32_t next_state = ((state << Q2B_K) | my_sym) & Q2B_STATE_MASK;
            float d = q2b_decode(next_state, mult) - target;
            err = d * d;
        }

        // Warp min-reduce with lower-symbol tie-break (matches the CPU
        // greedy's ascending scan + strict `<`).
        uint32_t best_sym = my_sym;
        float    best_err = err;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float    other_err = __shfl_xor_sync(0xFFFFFFFFu, best_err, offset);
            uint32_t other_sym = __shfl_xor_sync(0xFFFFFFFFu, best_sym, offset);
            if (other_err < best_err ||
                (other_err == best_err && other_sym < best_sym)) {
                best_err = other_err;
                best_sym = other_sym;
            }
        }

        if (tid == 0) {
            state = ((state << Q2B_K) | best_sym) & Q2B_STATE_MASK;
            pending |= (uint8_t)(best_sym & 0x3u) << ((t & 3) * 2);
            if ((t & 3) == 3) {
                my_pkd[t / 4] = pending;
                pending = 0u;
            }
        }
        state = __shfl_sync(0xFFFFFFFFu, state, 0);
    }
}

// ---------------------------------------------------------------------------
// 3c) Viterbi quantize — prefix-grouped DP, K=2 variant.
// ---------------------------------------------------------------------------
//
// Same algorithm as qtip_quantize.cu's `qtip_quantize_rows_viterbi_kernel`
// with two changes:
//   * decode_err uses the computed codebook (no LUT argument), and
//   * PREFIX_COUNT = 2^(L-K) = 16384, so the per-prefix cost table is 64 KiB
//     of f32 — beyond the 48 KiB static shared-memory limit. The per-prefix
//     min lives in a global `prefix_cost` scratch instead, and best_j is
//     written directly to the (per-prefix) backtrace in gmem. Phase B reads
//     prefix_cost[s >> K], which is fully coalesced (4 consecutive states
//     share one prefix), so the extra gmem round-trip stays bandwidth-cheap
//     relative to the 2^L state sweep it feeds.
//
// Scratch (allocated by the host, per row-in-flight):
//   * cost_a, cost_b:  [BATCH, 2^L]  f32 (ping-pong)
//   * prefix_cost:     [BATCH, 2^14] f32
//   * backtrace:       [BATCH, T, 2^14] u8
constexpr int Q2B_VITERBI_THREADS       = 256;
constexpr uint32_t Q2B_PREFIX_BITS      = Q2B_L - Q2B_K;           // 14
constexpr uint32_t Q2B_PREFIX_COUNT     = 1u << Q2B_PREFIX_BITS;   // 16384

__global__ void qtip2b_quantize_rows_viterbi_kernel(
    const float* __restrict__ weight,        // [n_rows, in_features]
    const float* __restrict__ row_scales,    // [n_rows]
    uint8_t*     __restrict__ packed,        // [n_rows, num_symbols / 4]
    float*       __restrict__ cost_a,        // scratch [BATCH, 2^L]
    float*       __restrict__ cost_b,        // scratch [BATCH, 2^L]
    float*       __restrict__ prefix_cost,   // scratch [BATCH, PREFIX_COUNT]
    uint8_t*     __restrict__ backtrace,     // scratch [BATCH, T, PREFIX_COUNT]
    int in_features,
    int num_symbols,
    int row_offset,
    uint32_t mult
) {
    int local_row = blockIdx.x;
    int row       = row_offset + local_row;
    int tid       = threadIdx.x;

    const float* my_row = weight + (size_t)row * in_features;
    uint8_t*     my_pkd = packed + (size_t)row * (num_symbols / 4);

    float* my_cost_a = cost_a + (size_t)local_row * Q2B_NSTATES;
    float* my_cost_b = cost_b + (size_t)local_row * Q2B_NSTATES;
    float* my_pc     = prefix_cost + (size_t)local_row * Q2B_PREFIX_COUNT;
    uint8_t* my_bt   = backtrace +
                       (size_t)local_row * (size_t)num_symbols * Q2B_PREFIX_COUNT;

    // wave46-BX: exact IEEE ops, so a qtip2b GPU bake and a qtip2b CPU bake of
    // the same weights are the SAME checkpoint — and so "an unpruned beam
    // reproduces the exhaustive DP byte for byte" is testable on GPU. Mirrors
    // what wave13-AF did for the LUT rung.
    const float inv_scale = qtip_inv_scale_exact(row_scales[row]);

    // -------- Init t=0: only states 0..ALPHABET reachable --------
    {
        float t0 = qtip_scaled_target_exact(my_row[0], inv_scale);
        for (uint32_t i = tid; i < Q2B_NSTATES; i += Q2B_VITERBI_THREADS) {
            if (i < Q2B_ALPHABET) {
                float d = __fsub_rn(q2b_decode(i, mult), t0);
                my_cost_a[i] = __fmul_rn(d, d);
            } else {
                my_cost_a[i] = INFINITY;
            }
        }
    }
    __syncthreads();

    // -------- Forward pass t=1..T-1 --------
    float* prev = my_cost_a;
    float* curr = my_cost_b;

    for (int t = 1; t < num_symbols; ++t) {
        float tt = qtip_scaled_target_exact(my_row[t], inv_scale);
        uint8_t* bt_t = my_bt + (size_t)t * Q2B_PREFIX_COUNT;

        // Phase A: per-prefix 4-way reduction, results to gmem scratch.
        // Ascending-j scan + strict `<` matches the CPU tie-break exactly.
        for (uint32_t p = tid; p < Q2B_PREFIX_COUNT; p += Q2B_VITERBI_THREADS) {
            float    best = INFINITY;
            uint32_t bj   = 0u;
            #pragma unroll
            for (uint32_t j = 0; j < Q2B_ALPHABET; ++j) {
                uint32_t pred = (j << Q2B_PREFIX_BITS) | p;
                float c = prev[pred];
                if (c < best) {
                    best = c;
                    bj   = j;
                }
            }
            my_pc[p] = best;
            bt_t[p]  = (uint8_t)bj;
        }
        __syncthreads();

        // Phase B: per-state cost update with the in-register codebook.
        for (uint32_t s = tid; s < Q2B_NSTATES; s += Q2B_VITERBI_THREADS) {
            float d = __fsub_rn(q2b_decode(s, mult), tt);
            curr[s] = __fadd_rn(__fmul_rn(d, d), my_pc[s >> Q2B_K]);
        }
        __syncthreads();

        float* tmp = prev;
        prev = curr;
        curr = tmp;
    }

    // -------- Argmin final state (lowest state wins ties, like the CPU) ----
    __shared__ float    s_amin_cost [Q2B_VITERBI_THREADS];
    __shared__ uint32_t s_amin_state[Q2B_VITERBI_THREADS];

    float    local_best_cost  = INFINITY;
    uint32_t local_best_state = 0u;
    for (uint32_t s = tid; s < Q2B_NSTATES; s += Q2B_VITERBI_THREADS) {
        float c = prev[s];
        if (c < local_best_cost) {
            local_best_cost  = c;
            local_best_state = s;
        }
    }
    s_amin_cost [tid] = local_best_cost;
    s_amin_state[tid] = local_best_state;
    __syncthreads();

    for (int stride = Q2B_VITERBI_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_cost     = s_amin_cost[tid + stride];
            uint32_t other_state = s_amin_state[tid + stride];
            if (other_cost < s_amin_cost[tid] ||
                (other_cost == s_amin_cost[tid] && other_state < s_amin_state[tid])) {
                s_amin_cost [tid] = other_cost;
                s_amin_state[tid] = other_state;
            }
        }
        __syncthreads();
    }

    // -------- Backtrace + pack symbols (single-thread walk) --------
    if (tid == 0) {
        uint32_t s = s_amin_state[0];
        const uint32_t SYM_MASK = Q2B_ALPHABET - 1u;

        int packed_per_row = num_symbols / 4;
        for (int i = 0; i < packed_per_row; ++i) my_pkd[i] = 0u;

        int t = num_symbols - 1;
        uint8_t sym_t = (uint8_t)(s & SYM_MASK);
        my_pkd[t / 4] |= sym_t << ((t & 3) * 2);

        for (int tt2 = num_symbols - 1; tt2 > 0; --tt2) {
            uint8_t* bt_tt = my_bt + (size_t)tt2 * Q2B_PREFIX_COUNT;
            uint32_t prefix = s >> Q2B_K;
            uint32_t j      = (uint32_t)bt_tt[prefix];
            uint32_t prev_s = (j << Q2B_PREFIX_BITS) | prefix;
            uint8_t  sym_prev = (uint8_t)(prev_s & SYM_MASK);
            int prev_t = tt2 - 1;
            my_pkd[prev_t / 4] |= sym_prev << ((prev_t & 3) * 2);
            s = prev_s;
        }
    }
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

// ----- Dequantize -----------------------------------------------------------

#define Q2B_DEQUANT_LAUNCHER(NAME, T)                                          \
    void NAME(const uint8_t* d_packed,                                         \
              const float*   d_row_scales,                                     \
              T*             d_out,                                            \
              int n_rows,                                                      \
              int packed_per_row,                                              \
              int num_symbols,                                                 \
              uint32_t mult,                                                   \
              cudaStream_t stream) {                                           \
        constexpr int THREADS = 64;                                            \
        const int blocks = (n_rows + THREADS - 1) / THREADS;                   \
        qtip2b_dequantize_kernel<T, THREADS>                                   \
            <<<blocks, THREADS, 0, stream>>>(                                  \
                d_packed, d_row_scales, d_out,                                 \
                n_rows, packed_per_row, num_symbols, mult);                    \
    }

Q2B_DEQUANT_LAUNCHER(launch_qtip2b_dequantize_bf16, __nv_bfloat16)
Q2B_DEQUANT_LAUNCHER(launch_qtip2b_dequantize_f16,  __half)
Q2B_DEQUANT_LAUNCHER(launch_qtip2b_dequantize_f32,  float)

#undef Q2B_DEQUANT_LAUNCHER

// ----- Fused GEMV (plain when d_indices == nullptr; MoE gather otherwise) ---

#define Q2B_GEMV_LAUNCHER(NAME, T)                                             \
    void NAME(const uint8_t*  d_packed,                                        \
              const float*    d_row_scales,                                    \
              const T*        d_x_rotated,                                     \
              const uint32_t* d_indices,                                       \
              T*              d_y,                                             \
              int n_rows,                                                      \
              int packed_per_row,                                              \
              int num_symbols,                                                 \
              int n_pairs,                                                     \
              int num_experts,                                                 \
              uint32_t mult,                                                   \
              cudaStream_t stream) {                                           \
        constexpr int WARPS_PER_BLOCK = 8;                                     \
        constexpr int ROWS_PER_WARP = 2;                                       \
        constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;        \
        const size_t packed_smem = (size_t)ROWS_PER_BLOCK * packed_per_row;    \
        const bool   stage_packed = packed_smem <= 48 * 1024;                  \
        const size_t SHMEM = stage_packed ? packed_smem : 0;                   \
        dim3 grid((n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, n_pairs, 1); \
        qtip2b_gemv_warp_kernel<T, WARPS_PER_BLOCK, ROWS_PER_WARP>             \
            <<<grid, WARPS_PER_BLOCK * 32, SHMEM, stream>>>(                   \
                d_packed, d_row_scales, d_x_rotated, d_indices, d_y,           \
                n_rows, packed_per_row, num_symbols, n_pairs, num_experts,     \
                mult, stage_packed ? 1 : 0);                                   \
    }

Q2B_GEMV_LAUNCHER(launch_qtip2b_gemv_bf16, __nv_bfloat16)
Q2B_GEMV_LAUNCHER(launch_qtip2b_gemv_f16,  __half)
Q2B_GEMV_LAUNCHER(launch_qtip2b_gemv_f32,  float)

#undef Q2B_GEMV_LAUNCHER

// ----- Quantize side ---------------------------------------------------------

void launch_qtip2b_compute_row_scales_f32(
    const float*  d_weight,
    float*        d_row_scales,
    int n_rows,
    int in_features,
    float divisor,
    cudaStream_t  stream
) {
    qtip2b_row_scale_kernel<<<n_rows, Q2B_SCALE_THREADS, 0, stream>>>(
        d_weight, d_row_scales, in_features, divisor);
}

void launch_qtip2b_quantize_rows_greedy_f32(
    const float*  d_weight,
    const float*  d_row_scales,
    uint8_t*      d_packed,
    int n_rows,
    int in_features,
    int num_symbols,
    uint32_t mult,
    cudaStream_t  stream
) {
    qtip2b_quantize_rows_greedy_kernel
        <<<n_rows, Q2B_GREEDY_THREADS, 0, stream>>>(
            d_weight, d_row_scales, d_packed,
            in_features, num_symbols, mult);
}

void launch_qtip2b_quantize_rows_viterbi_f32(
    const float*  d_weight,
    const float*  d_row_scales,
    uint8_t*      d_packed,
    float*        d_cost_a,
    float*        d_cost_b,
    float*        d_prefix_cost,
    uint8_t*      d_backtrace,
    int n_rows,
    int in_features,
    int num_symbols,
    int row_offset,
    uint32_t mult,
    cudaStream_t  stream
) {
    qtip2b_quantize_rows_viterbi_kernel
        <<<n_rows, Q2B_VITERBI_THREADS, 0, stream>>>(
            d_weight, d_row_scales, d_packed,
            d_cost_a, d_cost_b, d_prefix_cost, d_backtrace,
            in_features, num_symbols, row_offset, mult);
}

} // extern "C"
