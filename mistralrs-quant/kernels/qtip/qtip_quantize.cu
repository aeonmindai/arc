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

namespace {

constexpr uint32_t QTIP_K          = 4;
constexpr uint32_t QTIP_L          = 16;
constexpr uint32_t QTIP_V          = 2;
constexpr uint32_t QTIP_STATE_MASK = (1u << QTIP_L) - 1u;
constexpr uint32_t QTIP_ALPHABET   = 1u << QTIP_K;       // 16
constexpr uint32_t QTIP_LUT_SIZE   = 1u << QTIP_L;       // 65536

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
// One CUDA block per row. Computes scale = max(|row|) / 3.0 (or 1.0 if
// max is 0) via parallel reduction. The per-row scale is then used by
// the quantize kernel to map weights into the unit-Gaussian frame.

constexpr int SCALE_THREADS = 256;

__global__ void qtip_row_scale_kernel(
    const float* __restrict__ weight,    // [n_rows, in_features]
    float*       __restrict__ row_scales, // [n_rows]
    int in_features
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
        float s = (m == 0.0f) ? 1.0f : (m / 3.0f);
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

__global__ void qtip_quantize_rows_greedy_kernel(
    const float*   __restrict__ weight,        // [n_rows, in_features]
    const float*   __restrict__ lut,           // [LUT_SIZE * V]
    const float*   __restrict__ row_scales,    // [n_rows]
    uint8_t*       __restrict__ packed,        // [n_rows, num_symbols / 2]
    int in_features,
    int num_symbols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* my_row = weight + (size_t)row * in_features;
    uint8_t*     my_pkd = packed + (size_t)row * (num_symbols / 2);

    float scale     = row_scales[row];
    float inv_scale = 1.0f / scale;

    uint32_t state = 0u;

    // Each thread holds the symbol candidate it's responsible for.
    uint32_t my_sym = (uint32_t)tid;  // only first 16 threads do work

    // Packing accumulator for two symbols per byte.
    uint8_t pending_low = 0u;

    for (int t = 0; t < num_symbols; ++t) {
        // Target values for timestep t (V=2 weights).
        float t0 = my_row[t * 2 + 0] * inv_scale;
        float t1 = my_row[t * 2 + 1] * inv_scale;

        // Compute err for this thread's symbol candidate.
        float err = INFINITY;
        if (tid < (int)QTIP_ALPHABET) {
            uint32_t next_state = ((state << QTIP_K) | my_sym) & QTIP_STATE_MASK;
            size_t off = (size_t)next_state * QTIP_V;
            float l0 = lut[off + 0];
            float l1 = lut[off + 1];
            float d0 = l0 - t0;
            float d1 = l1 - t1;
            err = d0 * d0 + d1 * d1;
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
// Viterbi quantize kernel.
// ---------------------------------------------------------------------------
//
// One CUDA block per row. Block has up to 1024 threads, each handling a
// strided slice of the 2^L = 65536 states. Timesteps are sequential —
// the block syncs after each timestep.
//
// Memory layout (passed as kernel args, allocated by host):
//   * cost_a, cost_b: [n_concurrent_rows, LUT_SIZE] f32 — current/previous
//     cost tables, ping-ponged each timestep.
//   * backtrace:      [n_concurrent_rows, T, LUT_SIZE] u8 — for each
//     (timestep, state), the K-bit predecessor index `j` that gave the
//     min cost. T = num_symbols (max we'll see in this batch).
//
// Algorithm (mirrors CPU `viterbi_quantize_row` in `viterbi.rs`):
//   * Init (t=0): cost[s] = decode_error(s, target_0) for s ∈ [0, ALPHABET);
//                 cost[s] = +inf otherwise.
//   * Forward (t=1..T-1): for each state s, cost'[s] = err(s, target_t) +
//     min_{j∈[0,16)} cost[predecessor(s, j)]. Record best j in backtrace.
//   * Backtrace: argmin over final cost. Walk back through backtrace to
//     recover symbol sequence. Pack two symbols per byte.
//
// `predecessor(s, j) = (j << (L-K)) | (s >> K)`.
//
// One thread block per row; the block iterates over T timesteps.

constexpr int VITERBI_THREADS = 256;
constexpr int VITERBI_STATES_PER_THREAD = (1 << QTIP_L) / VITERBI_THREADS;  // 256 if 256 threads
// `VITERBI_STATES_PER_THREAD * VITERBI_THREADS == LUT_SIZE` must hold.

// Compute the trellis predecessor of state `s` with predecessor index `j`.
__device__ __forceinline__ uint32_t predecessor(uint32_t s, uint32_t j) {
    return (j << (QTIP_L - QTIP_K)) | (s >> QTIP_K);
}

// Decode-error of state `s` against the V=2 target vector (already scaled).
__device__ __forceinline__ float decode_err_fp(
    const float* __restrict__ lut, uint32_t s, float t0, float t1
) {
    size_t off = (size_t)s * QTIP_V;
    float l0 = lut[off + 0];
    float l1 = lut[off + 1];
    float d0 = l0 - t0;
    float d1 = l1 - t1;
    return d0 * d0 + d1 * d1;
}

__global__ void qtip_quantize_rows_viterbi_kernel(
    const float*   __restrict__ weight,        // [n_rows, in_features]
    const float*   __restrict__ lut,           // [LUT_SIZE * V]
    const float*   __restrict__ row_scales,    // [n_rows]
    uint8_t*       __restrict__ packed,        // [n_rows, num_symbols / 2]
    float*         __restrict__ cost_a,        // scratch [BATCH, LUT_SIZE]
    float*         __restrict__ cost_b,        // scratch [BATCH, LUT_SIZE]
    uint8_t*       __restrict__ backtrace,     // scratch [BATCH, T, LUT_SIZE]
    int in_features,
    int num_symbols,
    int row_offset                              // global row index of grid block 0
) {
    int local_row = blockIdx.x;
    int row       = row_offset + local_row;
    int tid       = threadIdx.x;

    const float* my_row = weight + (size_t)row * in_features;
    uint8_t*     my_pkd = packed + (size_t)row * (num_symbols / 2);

    float* my_cost_a = cost_a + (size_t)local_row * QTIP_LUT_SIZE;
    float* my_cost_b = cost_b + (size_t)local_row * QTIP_LUT_SIZE;
    uint8_t* my_bt   = backtrace +
                       (size_t)local_row * (size_t)num_symbols * QTIP_LUT_SIZE;

    float scale     = row_scales[row];
    float inv_scale = 1.0f / scale;

    // -------- Init t=0 --------
    // Only first ALPHABET states reachable; rest = +inf.
    float t0 = my_row[0] * inv_scale;
    float t1 = my_row[1] * inv_scale;
    for (uint32_t i = tid; i < QTIP_LUT_SIZE; i += VITERBI_THREADS) {
        if (i < QTIP_ALPHABET) {
            my_cost_a[i] = decode_err_fp(lut, i, t0, t1);
        } else {
            my_cost_a[i] = INFINITY;
        }
    }
    __syncthreads();

    // -------- Forward pass t=1..T-1 --------
    // prev = cost_a, curr = cost_b. Swap each timestep.
    float* prev = my_cost_a;
    float* curr = my_cost_b;

    for (int t = 1; t < num_symbols; ++t) {
        float tt0 = my_row[t * 2 + 0] * inv_scale;
        float tt1 = my_row[t * 2 + 1] * inv_scale;

        uint8_t* bt_t = my_bt + (size_t)t * QTIP_LUT_SIZE;

        // Each thread handles a strided slice of states.
        for (uint32_t s = tid; s < QTIP_LUT_SIZE; s += VITERBI_THREADS) {
            float err = decode_err_fp(lut, s, tt0, tt1);

            // Find min over 16 predecessors.
            float    best_cost = INFINITY;
            uint32_t best_j    = 0u;
            #pragma unroll
            for (uint32_t j = 0; j < QTIP_ALPHABET; ++j) {
                uint32_t p = predecessor(s, j);
                float c = prev[p];
                if (c < best_cost) {
                    best_cost = c;
                    best_j    = j;
                }
            }

            curr[s] = err + best_cost;
            bt_t[s] = (uint8_t)best_j;
        }
        __syncthreads();

        // Swap.
        float* tmp = prev;
        prev = curr;
        curr = tmp;
    }

    // -------- Find argmin final state --------
    // Use shared memory tile for the reduction.
    __shared__ float s_best_cost[VITERBI_THREADS];
    __shared__ uint32_t s_best_state[VITERBI_THREADS];

    float    local_best_cost  = INFINITY;
    uint32_t local_best_state = 0u;
    for (uint32_t s = tid; s < QTIP_LUT_SIZE; s += VITERBI_THREADS) {
        float c = prev[s];
        if (c < local_best_cost) {
            local_best_cost  = c;
            local_best_state = s;
        }
    }
    s_best_cost[tid]  = local_best_cost;
    s_best_state[tid] = local_best_state;
    __syncthreads();

    // Tree reduction over the block.
    for (int stride = VITERBI_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_cost     = s_best_cost[tid + stride];
            uint32_t other_state = s_best_state[tid + stride];
            if (other_cost < s_best_cost[tid] ||
                (other_cost == s_best_cost[tid] && other_state < s_best_state[tid])) {
                s_best_cost[tid]  = other_cost;
                s_best_state[tid] = other_state;
            }
        }
        __syncthreads();
    }

    // -------- Backtrace + pack symbols (single-thread) --------
    if (tid == 0) {
        uint32_t s = s_best_state[0];
        // Walk back. The CPU code stores `symbols[t]` for t = 0..num_symbols-1.
        // We need to pack two symbols per byte: low nibble = symbols[2b],
        // high nibble = symbols[2b+1].
        // Strategy: write into a per-row symbol buffer (in registers/local),
        // then pack. To avoid 7k-element local arrays, we instead emit packed
        // bytes inline: first build the symbol sequence by walking back from
        // num_symbols-1 to 0, then pack.
        //
        // Easier: use the backtrace stride to walk symbols in reverse, store
        // them into the packed array (we know packed_per_row = num_symbols/2).
        // We need to write each symbol; but the byte we write to depends on
        // whether it's the low or high nibble.
        //
        // We do this in TWO passes if needed, but the cleanest single-pass
        // approach is: write each symbol to a temp byte location, OR'ing with
        // 0 (for the low nibble) or 0 (high nibble we shift). Since we don't
        // know the order of writes (reverse), we need each byte to be touched
        // twice — once for its low symbol, once for its high.
        //
        // Implementation: zero-init the packed array first, then walk back
        // and OR the symbols into place.

        // Zero packed bytes (single-thread is fine; range is small).
        int packed_per_row = num_symbols / 2;
        for (int i = 0; i < packed_per_row; ++i) my_pkd[i] = 0u;

        // Walk back from t = num_symbols-1 down to 0.
        // The symbol at position t equals (s & ALPHABET_MASK) where s is the
        // state AT time t.
        const uint32_t SYM_MASK = QTIP_ALPHABET - 1u;
        int t = num_symbols - 1;
        uint8_t sym_t = (uint8_t)(s & SYM_MASK);
        if ((t & 1) == 0) {
            my_pkd[t / 2] |= sym_t;
        } else {
            my_pkd[t / 2] |= (sym_t << 4);
        }
        for (int tt = num_symbols - 1; tt > 0; --tt) {
            // Backtrace at time `tt` gives the predecessor `j` chosen when
            // computing cost[tt][s] — that yields state at time `tt - 1`.
            uint8_t* bt_tt = my_bt + (size_t)tt * QTIP_LUT_SIZE;
            uint32_t j = (uint32_t)bt_tt[s];
            uint32_t prev_s = predecessor(s, j);
            uint8_t  sym_prev = (uint8_t)(prev_s & SYM_MASK);
            int prev_t = tt - 1;
            if ((prev_t & 1) == 0) {
                my_pkd[prev_t / 2] |= sym_prev;
            } else {
                my_pkd[prev_t / 2] |= (sym_prev << 4);
            }
            s = prev_s;
        }
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
    cudaStream_t  stream
) {
    qtip_row_scale_kernel<<<n_rows, SCALE_THREADS, 0, stream>>>(
        d_weight, d_row_scales, in_features);
}

// ----- Greedy quantize ------------------------------------------------------

void launch_qtip_quantize_rows_greedy_f32(
    const float*  d_weight,
    const float*  d_lut,
    const float*  d_row_scales,
    uint8_t*      d_packed,
    int n_rows,
    int in_features,
    int num_symbols,
    cudaStream_t  stream
) {
    qtip_quantize_rows_greedy_kernel
        <<<n_rows, GREEDY_THREADS, 0, stream>>>(
            d_weight, d_lut, d_row_scales, d_packed,
            in_features, num_symbols);
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
    cudaStream_t  stream
) {
    qtip_quantize_rows_viterbi_kernel
        <<<n_rows, VITERBI_THREADS, 0, stream>>>(
            d_weight, d_lut, d_row_scales, d_packed,
            d_cost_a, d_cost_b, d_backtrace,
            in_features, num_symbols, row_offset);
}

} // extern "C"
