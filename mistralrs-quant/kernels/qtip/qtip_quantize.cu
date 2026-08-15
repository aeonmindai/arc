// QTIP CUDA quantize kernels (RUN-quant-on-gpu).
//
// Three kernels live in this file:
//   1) `qtip_rotate_weight_rows_kernel<BLOCK>` — apply the block-diagonal
//      D·H·D rotation to each row of the weight matrix in-place. Mirrors
//      the activation rotation in `qtip_dequantize.cu` but operates on
//      FP32 weight rows (the quantize-time data type).
//   2) `qtip_quantize_rows_greedy_kernel` — per-row greedy quantize.
//      Sequentially walks the trellis picking the locally-best symbol at
//      each timestep, then packs two K=4 symbols per byte.
//   3) `qtip_quantize_rows_viterbi_kernel` — per-row Viterbi DP quantize.
//      One CUDA block per row. Block-wide cooperation across the 2^L
//      states; T timesteps processed sequentially with block barriers.
//      Forward DP + backtrace + symbol packing all on-GPU.
//
// Format invariants (must match CPU/`mod.rs`):
//   * L=16 trellis bits, K=4 symbol bits, V=2 reproduction values
//   * Row scale = max(|row|) / 3.0 (clamped to 1.0 if max is 0)
//   * Symbol packing: two K=4 symbols per byte, LOW nibble = first symbol
//   * Initial state at t=0 = 0 (state is post-shift, so first state is
//     just the first symbol s.t. s ∈ [0, ALPHABET))
//
// Memory budget (per row in the Viterbi kernel):
//   * 2 × cost arrays of 2^L f32 each = 512 KiB in HBM scratch
//   * 1 × backtrace of T × 2^L u8 = T × 64 KiB in HBM scratch
//   * Backtrace dominates: for T=7168 (Llama 70B intermediate=14336),
//     per-row backtrace is ~470 MB. Host code chooses BATCH such that
//     `BATCH × T × 65536` fits a configurable scratch budget.
//
// SM80+ only (FP32 only, no half-precision intrinsics needed). Gated by
// `has_qtip_kernels` in build.rs, same as the dequantize kernels.

#include <cstdint>
#include <cuda_runtime.h>

// Per-operation IEEE f32 helpers. The crate builds with --use_fast_math, which
// would otherwise contract the branch metric into an FMA and turn `1.0f/scale`
// into an approximate reciprocal — either of which can flip a Viterbi tie and
// make a GPU bake disagree with the CPU reference in `qtip/viterbi.rs`.
// wave13-AF: routing the trellis arithmetic through these intrinsics makes the
// exhaustive kernel, the greedy kernel and the new beam kernel all bit-identical
// to their Rust counterparts. See kernels/qtip/qtip_exact_fp.cuh.
#include "qtip_exact_fp.cuh"
// Codebook selection (stored Gaussian LUT vs in-register sum2 code).
#include "qtip_codebook.cuh"

namespace {

constexpr uint32_t QTIP_K          = 4;
constexpr uint32_t QTIP_L          = 16;
constexpr uint32_t QTIP_V          = 2;
constexpr uint32_t QTIP_STATE_MASK = (1u << QTIP_L) - 1u;
constexpr uint32_t QTIP_ALPHABET   = 1u << QTIP_K;       // 16
constexpr uint32_t QTIP_LUT_SIZE   = 1u << QTIP_L;       // 65536

// Branch metric with the codebook selected at compile time. The `false`
// instantiation is exactly `qtip_decode_err_exact` (two LUT loads, then the
// non-contracting f32 arithmetic); the `true` one derives the same two values
// from the state with no memory access. Both feed the identical
// `qtip_decode_err_exact_lv`, so the DP's rounding behaviour is unchanged and
// the CUDA search stays bit-identical to its Rust twin at either codebook.
template <bool COMPUTED_CB>
__device__ __forceinline__ float qtip_cb_err(
    const float* __restrict__ lut, unsigned int s, float t0, float t1, unsigned int mult
) {
    const float2 c = qtip_cb_value<COMPUTED_CB>(lut, s, mult);
    return qtip_decode_err_exact_lv(c.x, c.y, t0, t1);
}

// ---------------------------------------------------------------------------
// Rotation kernel for weight rows (FP32 in-place block-diagonal D·H·D).
// ---------------------------------------------------------------------------
//
// One CUDA block handles ONE rotation block of BLOCK consecutive elements
// of ONE row. Grid: (n_rows, blocks_per_row). Block: (BLOCK,).
//
// Same math as the activation rotation in `qtip_dequantize.cu`, but the
// input is always FP32 (the quantize-time working dtype).
template <int BLOCK>
__global__ void qtip_rotate_weight_rows_kernel(
    float*       __restrict__ weight,    // [n_rows, in_features], in-place
    const float* __restrict__ signs,     // [in_features]
    int in_features
) {
    static_assert(BLOCK >= 2 && BLOCK <= 256, "BLOCK must be in [2, 256]");

    int row_idx          = blockIdx.x;
    int block_in_row     = blockIdx.y;
    int feature_off      = block_in_row * BLOCK + threadIdx.x;
    if (feature_off >= in_features) return;
    int row_off          = row_idx * in_features + feature_off;

    __shared__ float smem[BLOCK];

    float sign = signs[feature_off];
    float v    = weight[row_off];
    smem[threadIdx.x] = v * sign;
    __syncthreads();

    int lane = threadIdx.x;
    #pragma unroll
    for (int h = 1; h < BLOCK; h *= 2) {
        int pair = lane ^ h;
        float a  = smem[lane];
        float b  = smem[pair];
        __syncthreads();
        smem[lane] = ((lane & h) == 0) ? (a + b) : (b - a);
        __syncthreads();
    }

    float wht_scale = rsqrtf((float)BLOCK);
    float rotated   = smem[threadIdx.x] * wht_scale * sign;

    weight[row_off] = rotated;
}

// ---------------------------------------------------------------------------
// Per-row scale computation kernel.
// ---------------------------------------------------------------------------
//
// One CUDA block per row. Computes scale = max(|row|) / divisor (or 1.0 if
// max is 0) via parallel reduction. The per-row scale is then used by
// the quantize kernel to map weights into the unit-Gaussian frame.
//
// `divisor` is 3.0 for the Gaussian LUT (which is unit-sigma by construction)
// and 3*sigma for the computed sum2 codebook (sigma = 1.225552), which is the
// same `max|row| / 3-sigma` policy `QTIP2B_SCALE_DIVISOR` encodes at K=2/V=1.
// It is passed rather than folded into the values so the decode pays nothing.

constexpr int SCALE_THREADS = 256;

__global__ void qtip_row_scale_kernel(
    const float* __restrict__ weight,    // [n_rows, in_features]
    float*       __restrict__ row_scales, // [n_rows]
    int in_features,
    float divisor                         // 3.0 (Gaussian LUT) or 3*sigma (computed)
) {
    __shared__ float smax[SCALE_THREADS];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* my_row = weight + (size_t)row * in_features;

    float local_max = 0.0f;
    for (int i = tid; i < in_features; i += SCALE_THREADS) {
        float v = my_row[i];
        float a = fabsf(v);
        if (a > local_max) local_max = a;
    }
    smax[tid] = local_max;
    __syncthreads();

    // Tree reduction.
    for (int stride = SCALE_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = smax[tid + stride];
            if (other > smax[tid]) smax[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = smax[0];
        // __fdiv_rn, not `/`: --use_fast_math turns `/` into an approximate
        // reciprocal-multiply, which would make the GPU row scale differ from
        // the CPU's `max_abs / 3.0` and desynchronise every downstream target.
        float s = (m == 0.0f) ? 1.0f : __fdiv_rn(m, divisor);
        row_scales[row] = s;
    }
}

// ---------------------------------------------------------------------------
// Greedy quantize kernel.
// ---------------------------------------------------------------------------
//
// One CUDA block per row. Within each row, the trellis walk is sequential
// (state_t depends on state_{t-1}), so we serialize on a single thread
// per timestep but parallelize the 16-way symbol search across threads
// of a warp.
//
// Layout:
//   * Block size = 32 (one warp). Each thread evaluates one symbol s ∈ [0,16).
//     (Threads 16-31 are inactive — they just participate in shuffles.)
//   * For each of `num_symbols` timesteps:
//       - Each active thread computes err for its own symbol given the
//         current state.
//       - Warp shuffle reduction picks the symbol with min err.
//       - Lane 0 updates the state, writes the symbol, and the packed byte
//         every two timesteps.

constexpr int GREEDY_THREADS = 32;

template <bool COMPUTED_CB>
__global__ void qtip_quantize_rows_greedy_kernel(
    const float*   __restrict__ weight,        // [n_rows, in_features]
    const float*   __restrict__ lut,           // [LUT_SIZE * V] (unused if computed)
    const float*   __restrict__ row_scales,    // [n_rows]
    uint8_t*       __restrict__ packed,        // [n_rows, num_symbols / 2]
    int in_features,
    int num_symbols,
    unsigned int cb_mult
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* my_row = weight + (size_t)row * in_features;
    uint8_t*     my_pkd = packed + (size_t)row * (num_symbols / 2);

    float scale     = row_scales[row];
    float inv_scale = qtip_inv_scale_exact(scale);

    uint32_t state = 0u;

    // Each thread holds the symbol candidate it's responsible for.
    uint32_t my_sym = (uint32_t)tid;  // only first 16 threads do work

    // Packing accumulator for two symbols per byte.
    uint8_t pending_low = 0u;

    for (int t = 0; t < num_symbols; ++t) {
        // Target values for timestep t (V=2 weights).
        float t0 = qtip_scaled_target_exact(my_row[t * 2 + 0], inv_scale);
        float t1 = qtip_scaled_target_exact(my_row[t * 2 + 1], inv_scale);

        // Compute err for this thread's symbol candidate.
        float err = INFINITY;
        if (tid < (int)QTIP_ALPHABET) {
            uint32_t next_state = ((state << QTIP_K) | my_sym) & QTIP_STATE_MASK;
            err = qtip_cb_err<COMPUTED_CB>(lut, next_state, t0, t1, cb_mult);
        }

        // Warp reduction: find min err and its sym.
        // We use bit-mask "encoded" pair: high 16 bits = sym, low 16 = float bits
        // but that loses precision. Instead, do paired shuffle reduction.
        uint32_t best_sym = my_sym;
        float    best_err = err;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float    other_err = __shfl_xor_sync(0xFFFFFFFFu, best_err, offset);
            uint32_t other_sym = __shfl_xor_sync(0xFFFFFFFFu, best_sym, offset);
            // Tie-break: lower symbol wins (matches CPU greedy which iterates
            // 0..16 and keeps strictly less-than).
            if (other_err < best_err ||
                (other_err == best_err && other_sym < best_sym)) {
                best_err = other_err;
                best_sym = other_sym;
            }
        }

        if (tid == 0) {
            uint8_t sym_byte = (uint8_t)(best_sym & 0x0Fu);
            state = ((state << QTIP_K) | best_sym) & QTIP_STATE_MASK;

            if ((t & 1) == 0) {
                pending_low = sym_byte;
            } else {
                my_pkd[t / 2] = pending_low | (sym_byte << 4);
            }
        }
        // Broadcast the new state to all threads so they can compute their
        // candidate next iteration.
        state = __shfl_sync(0xFFFFFFFFu, state, 0);
    }
}

// ---------------------------------------------------------------------------
// Viterbi quantize kernel — prefix-grouped (PG) variant.
// ---------------------------------------------------------------------------
//
// **The key observation that makes this fast on real H100 silicon.**
//
// The trellis transition is `predecessor(s, j) = (j << (L-K)) | (s >> K)`.
// All 16 successor states {p*16 + k : k ∈ [0, 16)} that share a common
// "prefix" p = s >> K have the IDENTICAL set of 16 predecessors —
// {(j << 12) | p : j ∈ [0, 16)} — and therefore the IDENTICAL argmin j.
//
// So instead of doing a 16-way reduction per state (65,536 × 16 ops per
// timestep), we do it once per prefix (4,096 × 16 ops) and reuse the result
// for the 16 successors. This collapses a 9.9-GiOp/row workload to a
// 0.6-GiOp/row one — roughly the difference between "5 min hang on
// down_proj" and "well under one second."
//
// Memory layout (passed as kernel args, allocated by host):
//   * cost_a, cost_b: [n_concurrent_rows, LUT_SIZE] f32 — current/previous
//     cost tables, ping-ponged each timestep.
//   * backtrace:      [n_concurrent_rows, T, PREFIX_COUNT] u8 — for each
//     (timestep, prefix), the K-bit predecessor j shared by all 16
//     successor states of that prefix. PREFIX_COUNT = 1 << (L - K) = 4096,
//     so backtrace is 16× smaller than the naive per-state version.
//
// Algorithm per timestep (block-wide):
//   Phase A: For each prefix p, compute
//     best_cost[p] = min_{j∈[0,16)} prev[(j << 12) | p]
//     best_j   [p] = argmin j
//   into shared memory (4,096 entries = 20 KiB total).
//   Phase B: For each state s, curr[s] = err(s, target_t) + best_cost[s >> K].
//   Phase C: Write best_j[p] for all p into backtrace[t][p].
//
// Backtrace correctness: walking back at time tt with current state s,
// we look up j = bt[tt][s >> K] (same j for all 16 successors of that
// prefix). predecessor = (j << 12) | (s >> K). Symbol at time tt is
// (s & ALPHABET_MASK), exactly as in the CPU reference.
//
// One thread block per row; the block iterates over T timesteps.

constexpr int VITERBI_THREADS         = 256;
constexpr uint32_t QTIP_PREFIX_BITS   = QTIP_L - QTIP_K;          // 12
constexpr uint32_t QTIP_PREFIX_COUNT  = 1u << QTIP_PREFIX_BITS;   // 4096
constexpr uint32_t QTIP_SUFFIX_COUNT  = 1u << QTIP_K;             // 16

// Decode-error of state `s` against the V=2 target vector (already scaled).
// Delegates to the non-contracting helper so the DP's branch metric is
// bit-identical to `qtip/viterbi.rs::decode_error`.
template <bool COMPUTED_CB>
__device__ __forceinline__ float decode_err_fp(
    const float* __restrict__ lut, uint32_t s, float t0, float t1, unsigned int mult
) {
    return qtip_cb_err<COMPUTED_CB>(lut, s, t0, t1, mult);
}

template <bool COMPUTED_CB>
__global__ void qtip_quantize_rows_viterbi_kernel(
    const float*   __restrict__ weight,        // [n_rows, in_features]
    const float*   __restrict__ lut,           // [LUT_SIZE * V] (unused if computed)
    const float*   __restrict__ row_scales,    // [n_rows]
    uint8_t*       __restrict__ packed,        // [n_rows, num_symbols / 2]
    float*         __restrict__ cost_a,        // scratch [BATCH, LUT_SIZE]
    float*         __restrict__ cost_b,        // scratch [BATCH, LUT_SIZE]
    uint8_t*       __restrict__ backtrace,     // scratch [BATCH, T, PREFIX_COUNT]
    int in_features,
    int num_symbols,
    int row_offset,                             // global row index of grid block 0
    unsigned int cb_mult
) {
    int local_row = blockIdx.x;
    int row       = row_offset + local_row;
    int tid       = threadIdx.x;

    const float* my_row = weight + (size_t)row * in_features;
    uint8_t*     my_pkd = packed + (size_t)row * (num_symbols / 2);

    float* my_cost_a = cost_a + (size_t)local_row * QTIP_LUT_SIZE;
    float* my_cost_b = cost_b + (size_t)local_row * QTIP_LUT_SIZE;
    uint8_t* my_bt   = backtrace +
                       (size_t)local_row * (size_t)num_symbols * QTIP_PREFIX_COUNT;

    float scale     = row_scales[row];
    float inv_scale = qtip_inv_scale_exact(scale);

    // Per-prefix tables in shared mem: 4096 × (f32 + u8) = 20 KiB. Plus the
    // argmin reduction tile reused at the end (256 × (f32+u32) = 2 KiB). All
    // well under H100's 48 KiB default per-block shared mem.
    __shared__ float    s_best_cost[QTIP_PREFIX_COUNT];
    __shared__ uint8_t  s_best_j   [QTIP_PREFIX_COUNT];

    // -------- Init t=0 --------
    // Only first ALPHABET states reachable; rest = +inf.
    {
        float t0 = qtip_scaled_target_exact(my_row[0], inv_scale);
        float t1 = qtip_scaled_target_exact(my_row[1], inv_scale);
        for (uint32_t i = tid; i < QTIP_LUT_SIZE; i += VITERBI_THREADS) {
            if (i < QTIP_ALPHABET) {
                my_cost_a[i] = decode_err_fp<COMPUTED_CB>(lut, i, t0, t1, cb_mult);
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
        float tt0 = qtip_scaled_target_exact(my_row[t * 2 + 0], inv_scale);
        float tt1 = qtip_scaled_target_exact(my_row[t * 2 + 1], inv_scale);

        // Phase A: per-prefix 16-way reduction (4096 prefixes, 16 j-values each).
        // Reads prev[] randomly but coalesced within a warp (consecutive prefixes
        // touch consecutive (j<<12)|p positions per j → 4-byte stride per thread).
        for (uint32_t p = tid; p < QTIP_PREFIX_COUNT; p += VITERBI_THREADS) {
            float    best = INFINITY;
            uint32_t bj   = 0u;
            #pragma unroll
            for (uint32_t j = 0; j < QTIP_SUFFIX_COUNT; ++j) {
                uint32_t pred = (j << QTIP_PREFIX_BITS) | p;
                float c = prev[pred];
                if (c < best) {
                    best = c;
                    bj   = j;
                }
            }
            s_best_cost[p] = best;
            s_best_j   [p] = (uint8_t)bj;
        }
        __syncthreads();

        // Phase B: per-state cost update (65536 states). Reads err from LUT,
        // adds shared best_cost[prefix], writes curr[]. All consecutive in s.
        for (uint32_t s = tid; s < QTIP_LUT_SIZE; s += VITERBI_THREADS) {
            float err = decode_err_fp<COMPUTED_CB>(lut, s, tt0, tt1, cb_mult);
            uint32_t p = s >> QTIP_K;
            curr[s] = __fadd_rn(err, s_best_cost[p]);
        }

        // Phase C: emit per-prefix backtrace for this timestep.
        uint8_t* bt_t = my_bt + (size_t)t * QTIP_PREFIX_COUNT;
        for (uint32_t p = tid; p < QTIP_PREFIX_COUNT; p += VITERBI_THREADS) {
            bt_t[p] = s_best_j[p];
        }
        __syncthreads();

        // Per-thread pointer swap (deterministic, same across all threads).
        float* tmp = prev;
        prev = curr;
        curr = tmp;
    }

    // -------- Find argmin final state --------
    // Reuse the front of s_best_cost as the argmin reduction tile (256 entries
    // out of 4096 already-allocated). Cast pointer; no extra shared mem needed.
    __shared__ uint32_t s_argmin_state[VITERBI_THREADS];

    float    local_best_cost  = INFINITY;
    uint32_t local_best_state = 0u;
    for (uint32_t s = tid; s < QTIP_LUT_SIZE; s += VITERBI_THREADS) {
        float c = prev[s];
        if (c < local_best_cost) {
            local_best_cost  = c;
            local_best_state = s;
        }
    }
    s_best_cost   [tid] = local_best_cost;
    s_argmin_state[tid] = local_best_state;
    __syncthreads();

    for (int stride = VITERBI_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_cost     = s_best_cost[tid + stride];
            uint32_t other_state = s_argmin_state[tid + stride];
            if (other_cost < s_best_cost[tid] ||
                (other_cost == s_best_cost[tid] && other_state < s_argmin_state[tid])) {
                s_best_cost   [tid] = other_cost;
                s_argmin_state[tid] = other_state;
            }
        }
        __syncthreads();
    }

    // -------- Backtrace + pack symbols (single-thread) --------
    if (tid == 0) {
        uint32_t s = s_argmin_state[0];
        const uint32_t SYM_MASK = QTIP_ALPHABET - 1u;

        // Zero packed bytes first; we OR low/high nibbles into them as we walk.
        int packed_per_row = num_symbols / 2;
        for (int i = 0; i < packed_per_row; ++i) my_pkd[i] = 0u;

        int t = num_symbols - 1;
        uint8_t sym_t = (uint8_t)(s & SYM_MASK);
        if ((t & 1) == 0) my_pkd[t / 2] |= sym_t;
        else              my_pkd[t / 2] |= (sym_t << 4);

        for (int tt = num_symbols - 1; tt > 0; --tt) {
            uint8_t* bt_tt = my_bt + (size_t)tt * QTIP_PREFIX_COUNT;
            uint32_t prefix = s >> QTIP_K;
            uint32_t j      = (uint32_t)bt_tt[prefix];
            uint32_t prev_s = (j << QTIP_PREFIX_BITS) | prefix;
            uint8_t  sym_prev = (uint8_t)(prev_s & SYM_MASK);
            int prev_t = tt - 1;
            if ((prev_t & 1) == 0) my_pkd[prev_t / 2] |= sym_prev;
            else                   my_pkd[prev_t / 2] |= (sym_prev << 4);
            s = prev_s;
        }
    }
}

// ---------------------------------------------------------------------------
// Least-squares scale refinement kernel.
// ---------------------------------------------------------------------------
//
// After Viterbi/Greedy picks the optimal state sequence, the heuristic scale
// (max|row|/3) is suboptimal.  The MSE-optimal scale for a fixed state
// sequence is:
//
//   s* = dot(w_rotated, lut_values) / dot(lut_values, lut_values)
//
// where lut_values[i] = lut[state_for_i].  This kernel replays the trellis
// from the packed symbols, reconstructs the unscaled LUT values, and
// computes the two dot products in one pass via parallel reduction.
//
// One CUDA block per row, SCALE_THREADS threads.

template <bool COMPUTED_CB>
__global__ void qtip_refine_scales_kernel(
    const float*   __restrict__ weight,      // [n_rows, in_features] rotated
    const uint8_t* __restrict__ packed,       // [n_rows, num_symbols / 2]
    const float*   __restrict__ lut,          // [LUT_SIZE * V] (unused if computed)
    float*         __restrict__ row_scales,   // [n_rows] — updated in-place
    int in_features,
    int num_symbols,
    unsigned int cb_mult
) {
    __shared__ float s_dot_wl[SCALE_THREADS];
    __shared__ float s_dot_ll[SCALE_THREADS];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float*   my_row = weight + (size_t)row * in_features;
    const uint8_t* my_pkd = packed + (size_t)row * (num_symbols / 2);

    // Each thread accumulates partial sums over its strided slice of symbols.
    float dot_wl = 0.0f;
    float dot_ll = 0.0f;

    // Replay the trellis to recover states.  The trellis is sequential
    // (state_t depends on state_{t-1}), but we only need the LUT values
    // which are determined by the state AFTER each symbol.  We can replay
    // serially on thread 0 and broadcast, but that serialises the row.
    //
    // Better: each thread independently replays ALL symbols (cheap — just
    // shift+mask per symbol) but only accumulates its own strided portion.
    // The replay is identical across threads (deterministic), so this is
    // correct and fully parallel for the dot products.

    uint32_t state = 0u;
    for (int t = 0; t < num_symbols; ++t) {
        // Unpack symbol.
        uint8_t byte = my_pkd[t / 2];
        uint32_t sym = ((t & 1) == 0) ? (byte & 0x0Fu) : (byte >> 4);

        state = ((state << QTIP_K) | sym) & QTIP_STATE_MASK;

        // Only accumulate for this thread's slice.
        if ((t % SCALE_THREADS) == tid) {
            const float2 c = qtip_cb_value<COMPUTED_CB>(lut, state, cb_mult);
            float l0 = c.x;
            float l1 = c.y;
            float w0 = my_row[t * 2 + 0];
            float w1 = my_row[t * 2 + 1];
            dot_wl += w0 * l0 + w1 * l1;
            dot_ll += l0 * l0 + l1 * l1;
        }
    }

    s_dot_wl[tid] = dot_wl;
    s_dot_ll[tid] = dot_ll;
    __syncthreads();

    // Tree reduction.
    for (int stride = SCALE_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_dot_wl[tid] += s_dot_wl[tid + stride];
            s_dot_ll[tid] += s_dot_ll[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = s_dot_ll[0];
        float s = (denom > 0.0f) ? (s_dot_wl[0] / denom) : row_scales[row];
        row_scales[row] = s;
    }
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

// ----- Rotate weight rows -----------------------------------------------

#define QTIP_ROT_W_LAUNCHER(NAME, BLOCK)                                       \
    static void NAME(float* d_weight,                                          \
                     const float* d_signs,                                     \
                     int n_rows,                                               \
                     int in_features,                                          \
                     cudaStream_t stream) {                                    \
        const int blocks_per_row = in_features / (BLOCK);                      \
        dim3 grid(n_rows, blocks_per_row);                                     \
        qtip_rotate_weight_rows_kernel<(BLOCK)>                                \
            <<<grid, (BLOCK), 0, stream>>>(d_weight, d_signs, in_features);    \
    }

QTIP_ROT_W_LAUNCHER(rot_w_2,   2)
QTIP_ROT_W_LAUNCHER(rot_w_4,   4)
QTIP_ROT_W_LAUNCHER(rot_w_8,   8)
QTIP_ROT_W_LAUNCHER(rot_w_16,  16)
QTIP_ROT_W_LAUNCHER(rot_w_32,  32)
QTIP_ROT_W_LAUNCHER(rot_w_64,  64)
QTIP_ROT_W_LAUNCHER(rot_w_128, 128)

#undef QTIP_ROT_W_LAUNCHER

void launch_qtip_rotate_weight_rows_f32(
    float*         d_weight,
    const float*   d_signs,
    int n_rows,
    int in_features,
    int block_size,
    cudaStream_t   stream
) {
    switch (block_size) {
        case 2:   rot_w_2  (d_weight, d_signs, n_rows, in_features, stream); break;
        case 4:   rot_w_4  (d_weight, d_signs, n_rows, in_features, stream); break;
        case 8:   rot_w_8  (d_weight, d_signs, n_rows, in_features, stream); break;
        case 16:  rot_w_16 (d_weight, d_signs, n_rows, in_features, stream); break;
        case 32:  rot_w_32 (d_weight, d_signs, n_rows, in_features, stream); break;
        case 64:  rot_w_64 (d_weight, d_signs, n_rows, in_features, stream); break;
        case 128: rot_w_128(d_weight, d_signs, n_rows, in_features, stream); break;
        default: break; // unsupported — caller handles
    }
}

// ----- Per-row scale --------------------------------------------------------

void launch_qtip_compute_row_scales_f32(
    const float*  d_weight,
    float*        d_row_scales,
    int n_rows,
    int in_features,
    float         divisor,
    cudaStream_t  stream
) {
    qtip_row_scale_kernel<<<n_rows, SCALE_THREADS, 0, stream>>>(
        d_weight, d_row_scales, in_features, divisor);
}

// ----- Greedy quantize ------------------------------------------------------

// `cb_mult == 0` selects the stored Gaussian LUT; nonzero selects the computed
// sum2 codebook with that MCG multiplier. See qtip_codebook.cuh.
void launch_qtip_quantize_rows_greedy_f32(
    const float*  d_weight,
    const float*  d_lut,
    const float*  d_row_scales,
    uint8_t*      d_packed,
    int n_rows,
    int in_features,
    int num_symbols,
    unsigned int  cb_mult,
    cudaStream_t  stream
) {
    if (cb_mult != 0u) {
        qtip_quantize_rows_greedy_kernel<true>
            <<<n_rows, GREEDY_THREADS, 0, stream>>>(
                d_weight, d_lut, d_row_scales, d_packed,
                in_features, num_symbols, cb_mult);
    } else {
        qtip_quantize_rows_greedy_kernel<false>
            <<<n_rows, GREEDY_THREADS, 0, stream>>>(
                d_weight, d_lut, d_row_scales, d_packed,
                in_features, num_symbols, 0u);
    }
}

// ----- Viterbi quantize -----------------------------------------------------

void launch_qtip_quantize_rows_viterbi_f32(
    const float*  d_weight,
    const float*  d_lut,
    const float*  d_row_scales,
    uint8_t*      d_packed,
    float*        d_cost_a,
    float*        d_cost_b,
    uint8_t*      d_backtrace,
    int n_rows,
    int in_features,
    int num_symbols,
    int row_offset,
    unsigned int  cb_mult,
    cudaStream_t  stream
) {
    if (cb_mult != 0u) {
        qtip_quantize_rows_viterbi_kernel<true>
            <<<n_rows, VITERBI_THREADS, 0, stream>>>(
                d_weight, d_lut, d_row_scales, d_packed,
                d_cost_a, d_cost_b, d_backtrace,
                in_features, num_symbols, row_offset, cb_mult);
    } else {
        qtip_quantize_rows_viterbi_kernel<false>
            <<<n_rows, VITERBI_THREADS, 0, stream>>>(
                d_weight, d_lut, d_row_scales, d_packed,
                d_cost_a, d_cost_b, d_backtrace,
                in_features, num_symbols, row_offset, 0u);
    }
}

// ----- Scale refinement launch (kernel is in the anonymous namespace above:
// a template cannot have C linkage, so only the launcher lives here) --------

void launch_qtip_refine_scales_f32(
    const float*   d_weight,
    const uint8_t* d_packed,
    const float*   d_lut,
    float*         d_row_scales,
    int n_rows,
    int in_features,
    int num_symbols,
    unsigned int   cb_mult,
    cudaStream_t   stream
) {
    if (cb_mult != 0u) {
        qtip_refine_scales_kernel<true>
            <<<n_rows, SCALE_THREADS, 0, stream>>>(
                d_weight, d_packed, d_lut, d_row_scales,
                in_features, num_symbols, cb_mult);
    } else {
        qtip_refine_scales_kernel<false>
            <<<n_rows, SCALE_THREADS, 0, stream>>>(
                d_weight, d_packed, d_lut, d_row_scales,
                in_features, num_symbols, 0u);
    }
}

} // extern "C"
