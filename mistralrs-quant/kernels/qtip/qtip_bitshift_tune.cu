// Autotuner variant table for the qtip2b bitshift-trellis fused decode+GEMV.
//
// qtip_bitshift.cu's `qtip2b_gemv_warp_kernel` ships one fixed configuration
// (WARPS=8, R=2, GROUP=8, no prefetch, runtime smem staging). H200 measured
// it at 153-192 GB/s = 3-4% of the 4.8 TB/s HBM peak with ~18.7 us/call fixed
// overhead — long-scoreboard bound (June ncu narrative): not enough
// independent trellis streams in flight to hide the packed-byte load latency.
// EXL3 proves ~80% of peak is reachable for a computed-codebook trellis GEMV
// on GDDR; HBM needs even more in-flight parallelism.
//
// This TU compile-time instantiates the same kernel contract over the tuning
// axes so a sweep binary (examples/qtip_gemv_tune.rs) can measure every
// variant on real hardware and bake the winner per shape:
//
//   WARPS  warps per block                    (4 / 8 / 16)
//   R      rows (= independent trellis chains) per warp    (1 / 2 / 4)
//   ILP    independent K-segments per lane per row         (1 / 2 / 4)
//          — each segment is its own trellis stream, so a thread keeps
//          R*ILP state chains in flight (register blocking along K).
//   W      packed-word load width in bytes = vectorized load width
//          (2 / 4 / 8 -> GROUP = 4*W symbols decoded per lane-step)
//   MINB   __launch_bounds__ min-blocks-per-SM occupancy hint (1 / 2 / 4)
//   STAGE  stage the block's packed rows in shared memory    (0 / 1)
//   PF     software-pipelined double-buffering of the packed
//          word loads (prefetch depth 2)                     (0 / 1)
//
// Differences vs the legacy kernel (correctness contract identical):
//   * Warm-up replay is replaced by the O(1) random-access state
//     reconstruction `q2b_state_from_window` (qtip2b_common.cuh), reading the
//     two bytes preceding the segment as an aligned u16. Property-tested on
//     the Rust side (`grouped::window_state_2b`) and already used by the
//     grouped GEMM.
//   * Packed weights are loaded as aligned 2/4/8-byte words. A variant is
//     only applicable when `packed_per_row % W == 0` and
//     `num_symbols == 4 * packed_per_row`; the launcher returns -1 otherwise
//     and the Rust dispatch falls back to the legacy kernel.
//   * The decode itself (`q2b_decode`, f32-summed fp16 halves) is untouched,
//     so every variant is bit-identical to the CPU reference per weight;
//     only the dot-product association order differs between variants (each
//     variant is individually deterministic).
//
// The extern "C" surface:
//   qtip2b_gemv_num_variants()                  -> variant count
//   qtip2b_gemv_variant_info(idx, ...)          -> the axis values of a variant
//   launch_qtip2b_gemv_tuned_{bf16,f16,f32}(variant, <legacy gemv args>)
//       -> 0 on launch, -1 when the variant is not applicable to the shape
//          (caller falls back to launch_qtip2b_gemv_*).
//
// SM80+ (uses __nv_bfloat16). Gated by `has_qtip_kernels` in build.rs.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "qtip2b_common.cuh"

namespace {

// Packed-word type per load width. Spelled as the exact builtin types the
// __ldg overload set covers (uint64_t may alias `unsigned long` on LP64,
// which has no __ldg overload).
template <int W> struct q2b_wordty;
template <> struct q2b_wordty<2> { using type = unsigned short; };
template <> struct q2b_wordty<4> { using type = unsigned int; };
template <> struct q2b_wordty<8> { using type = unsigned long long; };

// Aligned word load. Shared-memory pointers must not go through __ldg.
template <bool STAGE, typename WT>
__device__ __forceinline__ WT q2b_load_word(const uint8_t* __restrict__ p) {
    if constexpr (STAGE) {
        return *reinterpret_cast<const WT*>(p);
    } else {
        return __ldg(reinterpret_cast<const WT*>(p));
    }
}

template <bool STAGE>
__device__ __forceinline__ uint32_t q2b_load_win16(const uint8_t* __restrict__ p) {
    if constexpr (STAGE) {
        return (uint32_t)(*reinterpret_cast<const unsigned short*>(p));
    } else {
        return (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p));
    }
}

// ---------------------------------------------------------------------------
// The parameterized fused decode + GEMV kernel.
//
// Grid:  (ceil(n_rows / (WARPS*R)), n_pairs, 1)
// Block: (WARPS * 32, 1, 1)
// Shared: STAGE ? WARPS*R*packed_per_row bytes : 0 (launcher-checked <=48KiB)
//
// Segment layout: iteration i, stream u covers symbols
//   base(u) = lane*GROUP + u*32*GROUP + i*32*GROUP*ILP  ..  +GROUP
// so each warp-step of one stream reads 32*W contiguous bytes (coalesced)
// and the ILP streams are contiguous 32-lane spans. Requires
// num_symbols == 4*packed_per_row and packed_per_row % W == 0 (enforced by
// the launcher), which makes every in-range word load full — no per-symbol
// tail guards inside the unrolled loops.
// ---------------------------------------------------------------------------
template <typename T, int WARPS, int R, int ILP, int W, int MINB, bool STAGE, bool PF>
__global__ void __launch_bounds__(WARPS * 32, MINB)
qtip2b_gemv_tuned_kernel(
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
    uint32_t mult
) {
    using WT = typename q2b_wordty<W>::type;
    constexpr int GROUP   = W * 4;        // symbols per lane-step per stream
    constexpr int RPB     = WARPS * R;    // rows per block
    constexpr int GSTRIDE = 32 * GROUP;   // symbols per warp-step per stream
    constexpr int ISTRIDE = GSTRIDE * ILP;

    extern __shared__ float q2b_tune_smem[];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int pair = blockIdx.y;
    const int row0_block = blockIdx.x * RPB;
    if (row0_block >= n_rows || pair >= n_pairs) return;

    const uint32_t expert = (indices != nullptr) ? __ldg(indices + pair) : 0u;
    const bool invalid = expert >= (uint32_t)num_experts;

    // ---- stage this block's packed rows (coalesced bulk copy) ----
    uint8_t* s_packed = nullptr;
    if constexpr (STAGE) {
        s_packed = reinterpret_cast<uint8_t*>(q2b_tune_smem);
        if (!invalid) {
            const int n_block_rows = min(RPB, n_rows - row0_block);
            const long total = (long)n_block_rows * packed_per_row;
            const uint8_t* __restrict__ g =
                packed + ((size_t)expert * n_rows + row0_block) * (size_t)packed_per_row;
            for (long i = threadIdx.x; i < total; i += WARPS * 32) {
                s_packed[i] = __ldg(g + i);
            }
            __syncthreads();
        }
    }

    const int row0 = row0_block + warp * R;
    if (invalid) {
        // Out-of-range expert id from a broken router: write zeros, never
        // read out of bounds (same contract as the legacy kernel).
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
                if (row < n_rows) y[(size_t)pair * n_rows + row] = q2b_from_f32<T>(0.0f);
            }
        }
        return;
    }

    // Per-row packed pointer (shared if staged) + scale. Out-of-range rows
    // get scale 0 so their accumulator stays 0 (and is never written).
    const uint8_t* rp[R];
    float scl[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        const bool ok = row < n_rows;
        const size_t er = (size_t)expert * n_rows + (ok ? row : 0);
        rp[r]  = STAGE ? (s_packed + (size_t)(warp * R + r) * packed_per_row)
                       : (packed + er * packed_per_row);
        scl[r] = ok ? __ldg(row_scales + er) : 0.0f;
    }
    const T* __restrict__ x_pair = x + (size_t)pair * num_symbols;

    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    // Load all live streams' packed words + initial trellis states for the
    // iteration starting at `base0`.
    auto load_streams = [&](int base0, WT (&w)[ILP][R], uint32_t (&st)[ILP][R]) {
        #pragma unroll
        for (int u = 0; u < ILP; ++u) {
            const int bu = base0 + u * GSTRIDE;
            if (bu < num_symbols) {
                const int boff = bu >> 2; // byte offset, multiple of W
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    w[u][r] = q2b_load_word<STAGE, WT>(rp[r] + boff);
                    // State before symbol `bu`: pair-reversal of the 16
                    // stream bits ending at bit 2*bu-1 = the aligned u16 at
                    // byte boff-2 (bu > 0 implies bu >= GROUP >= 8).
                    st[u][r] = (bu == 0)
                        ? 0u
                        : q2b_state_from_window(q2b_load_win16<STAGE>(rp[r] + boff - 2));
                }
            }
        }
    };

    // Decode + accumulate one iteration's streams. The j-outer / (u, r)-inner
    // order interleaves the R*ILP independent state chains at every symbol
    // step — that is the ILP that hides the state-carry + load latency.
    auto compute_streams = [&](int base0, WT (&w)[ILP][R], uint32_t (&st)[ILP][R]) {
        float xg[ILP][GROUP];
        #pragma unroll
        for (int u = 0; u < ILP; ++u) {
            const int bu = base0 + u * GSTRIDE;
            if (bu < num_symbols) {
                #pragma unroll
                for (int j = 0; j < GROUP; ++j) {
                    xg[u][j] = q2b_to_f32<T>(__ldg(x_pair + bu + j));
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < GROUP; ++j) {
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                if (base0 + u * GSTRIDE < num_symbols) {
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const uint32_t sym = (uint32_t)((w[u][r] >> (2 * j)) & (WT)0x3u);
                        st[u][r] = ((st[u][r] << Q2B_K) | sym) & Q2B_STATE_MASK;
                        acc[r] = fmaf(q2b_decode(st[u][r], mult) * scl[r], xg[u][j], acc[r]);
                    }
                }
            }
        }
    };

    if constexpr (PF) {
        // Software pipeline (depth 2): next iteration's packed words are in
        // flight while the current iteration decodes.
        int base0 = lane * GROUP;
        if (base0 < num_symbols) {
            WT wcur[ILP][R];
            uint32_t scur[ILP][R];
            load_streams(base0, wcur, scur);
            for (; base0 < num_symbols; base0 += ISTRIDE) {
                const int nxt = base0 + ISTRIDE;
                WT wnxt[ILP][R];
                uint32_t snxt[ILP][R];
                if (nxt < num_symbols) load_streams(nxt, wnxt, snxt);
                compute_streams(base0, wcur, scur);
                if (nxt < num_symbols) {
                    #pragma unroll
                    for (int u = 0; u < ILP; ++u) {
                        #pragma unroll
                        for (int r = 0; r < R; ++r) {
                            wcur[u][r] = wnxt[u][r];
                            scur[u][r] = snxt[u][r];
                        }
                    }
                }
            }
        }
    } else {
        for (int base0 = lane * GROUP; base0 < num_symbols; base0 += ISTRIDE) {
            WT wcur[ILP][R];
            uint32_t scur[ILP][R];
            load_streams(base0, wcur, scur);
            compute_streams(base0, wcur, scur);
        }
    }

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const float a = q2b_warp_reduce_sum(acc[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_rows) y[(size_t)pair * n_rows + row] = q2b_from_f32<T>(a);
    }
}

// ---------------------------------------------------------------------------
// Variant table. X(id, WARPS, R, ILP, W, MINB, STAGE, PF) — ids must be
// dense 0..N-1 in listing order (the sweep + baked table key on them, and
// they round-trip through tune_results.json, so treat the ids as ABI: only
// append, never renumber).
//
// Curated grid (~44 variants) around the sweep notes' G=8/R=2 start:
//   0        production-baseline replica (warps=8 R=2 ILP=1 W=2)
//   1..7     R x ILP grid at W=2
//   8..15    R x ILP grid at W=4
//   16..20   R x ILP grid at W=8 (register-pressure-bounded subset)
//   21..26   warps-per-block variation (4 / 16) on promising combos
//   27..30   __launch_bounds__ min-blocks occupancy hints (2 / 4)
//   31..34   no-smem-staging duals
//   35..40   double-buffered prefetch (staged)
//   41..43   double-buffered prefetch straight from gmem (unstaged)
// ---------------------------------------------------------------------------
#define Q2B_TUNE_VARIANTS(X) \
    X( 0,  8, 2, 1, 2, 1, 1, 0) \
    X( 1,  8, 1, 1, 2, 1, 1, 0) \
    X( 2,  8, 1, 2, 2, 1, 1, 0) \
    X( 3,  8, 1, 4, 2, 1, 1, 0) \
    X( 4,  8, 2, 2, 2, 1, 1, 0) \
    X( 5,  8, 2, 4, 2, 1, 1, 0) \
    X( 6,  8, 4, 1, 2, 1, 1, 0) \
    X( 7,  8, 4, 2, 2, 1, 1, 0) \
    X( 8,  8, 1, 1, 4, 1, 1, 0) \
    X( 9,  8, 1, 2, 4, 1, 1, 0) \
    X(10,  8, 1, 4, 4, 1, 1, 0) \
    X(11,  8, 2, 1, 4, 1, 1, 0) \
    X(12,  8, 2, 2, 4, 1, 1, 0) \
    X(13,  8, 2, 4, 4, 1, 1, 0) \
    X(14,  8, 4, 1, 4, 1, 1, 0) \
    X(15,  8, 4, 2, 4, 1, 1, 0) \
    X(16,  8, 1, 1, 8, 1, 1, 0) \
    X(17,  8, 1, 2, 8, 1, 1, 0) \
    X(18,  8, 2, 1, 8, 1, 1, 0) \
    X(19,  8, 2, 2, 8, 1, 1, 0) \
    X(20,  8, 4, 1, 8, 1, 1, 0) \
    X(21,  4, 2, 1, 2, 1, 1, 0) \
    X(22,  4, 2, 2, 4, 1, 1, 0) \
    X(23,  4, 1, 4, 4, 1, 1, 0) \
    X(24, 16, 2, 1, 2, 1, 1, 0) \
    X(25, 16, 2, 2, 4, 1, 1, 0) \
    X(26, 16, 1, 4, 4, 1, 1, 0) \
    X(27,  8, 2, 1, 2, 2, 1, 0) \
    X(28,  8, 2, 1, 2, 4, 1, 0) \
    X(29,  8, 2, 2, 4, 2, 1, 0) \
    X(30,  8, 2, 2, 4, 4, 1, 0) \
    X(31,  8, 2, 1, 2, 1, 0, 0) \
    X(32,  8, 2, 2, 4, 1, 0, 0) \
    X(33,  8, 1, 4, 4, 1, 0, 0) \
    X(34,  8, 4, 1, 2, 1, 0, 0) \
    X(35,  8, 2, 1, 2, 1, 1, 1) \
    X(36,  8, 2, 2, 4, 1, 1, 1) \
    X(37,  8, 1, 2, 4, 1, 1, 1) \
    X(38,  8, 1, 4, 4, 1, 1, 1) \
    X(39,  8, 2, 1, 8, 1, 1, 1) \
    X(40,  8, 4, 1, 4, 1, 1, 1) \
    X(41,  8, 2, 2, 4, 1, 0, 1) \
    X(42,  8, 1, 4, 4, 1, 0, 1) \
    X(43,  8, 4, 1, 8, 1, 0, 1)

#define Q2B_TUNE_COUNT_ONE(id, WARPS, R, ILP, W, MINB, STAGE, PF) +1
constexpr int Q2B_TUNE_NUM_VARIANTS = 0 Q2B_TUNE_VARIANTS(Q2B_TUNE_COUNT_ONE);
#undef Q2B_TUNE_COUNT_ONE

// Launch one variant; -1 when the variant is not applicable to the shape.
template <typename T, int WARPS, int R, int ILP, int W, int MINB, bool STAGE, bool PF>
int q2b_launch_variant(
    const uint8_t* d_packed,
    const float* d_row_scales,
    const T* d_x,
    const uint32_t* d_indices,
    T* d_y,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts,
    uint32_t mult,
    cudaStream_t stream
) {
    constexpr int RPB = WARPS * R;
    // Word loads (and the u16 window loads) need W-aligned rows, and the
    // no-tail-guard inner loops need whole final words.
    if (num_symbols != packed_per_row * 4) return -1;
    if (packed_per_row % W != 0) return -1;
    size_t shmem = 0;
    if (STAGE) {
        shmem = (size_t)RPB * (size_t)packed_per_row;
        if (shmem > 48 * 1024) return -1;
    }
    dim3 grid((n_rows + RPB - 1) / RPB, n_pairs, 1);
    qtip2b_gemv_tuned_kernel<T, WARPS, R, ILP, W, MINB, STAGE, PF>
        <<<grid, WARPS * 32, shmem, stream>>>(
            d_packed, d_row_scales, d_x, d_indices, d_y,
            n_rows, packed_per_row, num_symbols, n_pairs, num_experts, mult);
    return 0;
}

template <typename T>
int q2b_tuned_dispatch(
    int variant,
    const uint8_t* d_packed,
    const float* d_row_scales,
    const T* d_x,
    const uint32_t* d_indices,
    T* d_y,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int n_pairs,
    int num_experts,
    uint32_t mult,
    cudaStream_t stream
) {
    switch (variant) {
#define Q2B_TUNE_CASE(id, WARPS, R, ILP, W, MINB, STAGE, PF)                   \
        case id:                                                               \
            return q2b_launch_variant<T, WARPS, R, ILP, W, MINB, STAGE, PF>(   \
                d_packed, d_row_scales, d_x, d_indices, d_y, n_rows,           \
                packed_per_row, num_symbols, n_pairs, num_experts, mult,       \
                stream);
        Q2B_TUNE_VARIANTS(Q2B_TUNE_CASE)
#undef Q2B_TUNE_CASE
        default:
            return -1;
    }
}

} // anonymous namespace

// ============================================================================
// extern "C" surface
// ============================================================================

extern "C" {

int qtip2b_gemv_num_variants(void) {
    return Q2B_TUNE_NUM_VARIANTS;
}

int qtip2b_gemv_variant_info(
    int idx,
    int* warps,
    int* rows_per_warp,
    int* ilp,
    int* wbytes,
    int* min_blocks,
    int* stage,
    int* prefetch
) {
    switch (idx) {
#define Q2B_TUNE_INFO(id, WARPS, R, ILP, W, MINB, STAGE, PF)                   \
        case id:                                                               \
            *warps = WARPS;                                                    \
            *rows_per_warp = R;                                                \
            *ilp = ILP;                                                        \
            *wbytes = W;                                                       \
            *min_blocks = MINB;                                                \
            *stage = STAGE;                                                    \
            *prefetch = PF;                                                    \
            return 0;
        Q2B_TUNE_VARIANTS(Q2B_TUNE_INFO)
#undef Q2B_TUNE_INFO
        default:
            return -1;
    }
}

#define Q2B_TUNED_GEMV_LAUNCHER(NAME, T)                                       \
    int NAME(int variant,                                                      \
             const uint8_t*  d_packed,                                         \
             const float*    d_row_scales,                                     \
             const T*        d_x_rotated,                                      \
             const uint32_t* d_indices,                                        \
             T*              d_y,                                              \
             int n_rows,                                                       \
             int packed_per_row,                                               \
             int num_symbols,                                                  \
             int n_pairs,                                                      \
             int num_experts,                                                  \
             uint32_t mult,                                                    \
             cudaStream_t stream) {                                            \
        return q2b_tuned_dispatch<T>(                                          \
            variant, d_packed, d_row_scales, d_x_rotated, d_indices, d_y,      \
            n_rows, packed_per_row, num_symbols, n_pairs, num_experts, mult,   \
            stream);                                                           \
    }

Q2B_TUNED_GEMV_LAUNCHER(launch_qtip2b_gemv_tuned_bf16, __nv_bfloat16)
Q2B_TUNED_GEMV_LAUNCHER(launch_qtip2b_gemv_tuned_f16,  __half)
Q2B_TUNED_GEMV_LAUNCHER(launch_qtip2b_gemv_tuned_f32,  float)

#undef Q2B_TUNED_GEMV_LAUNCHER

} // extern "C"
