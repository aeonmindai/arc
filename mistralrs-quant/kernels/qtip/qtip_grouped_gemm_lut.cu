// QTIP **LUT-rung** (K=4 / V=2) GROUPED GEMM — batched 2-bit MoE serving.
//
// This is the sibling of `qtip_grouped_gemm.cu` for the rung that
// `qtip2-*.uqff` actually ships (`QtipLayer`). Same tile geometry, same
// cp.async pipeline, same mma.m16n8k16 accumulation; the ONLY difference is
// the B-fragment decode.
//
// Why it can exist at all
// -----------------------
// The LUT rung's trellis state update is
//
//     state = ((state << 4) | sym) & 0xFFFF          (K=4, L=16)
//
// so `state(t)` is nothing but the last FOUR 4-bit symbols — a 16-bit window
// over the packed nibble stream, exactly as the qtip2b rung's state is a
// 16-bit window over its 2-bit stream. The packing (`byte = lo nibble is
// symbol 2b, hi nibble is symbol 2b+1`) is plain little-endian nibble order,
// so symbol `t` occupies bits `[4t, 4t+3]`. Therefore
//
//     state(t) = nibble_reverse_16( stream_bits[4t-12 .. 4t+3] )
//
// — the K=4 analogue of `q2b_state_from_window`'s pair reversal, and just as
// random-access. There is NO sequential warm-up replay and no cross-thread
// state chain, which is the whole reason a grouped GEMM is reachable here.
//
// The V=2 bonus: one state yields the codeword PAIR covering weights
// `(2t, 2t+1)`, so this rung needs HALF the state reconstructions the qtip2b
// rung needs for the same number of weights.
//
// Byte geometry is identical to the qtip2b rung (both are 2 bits/weight):
// QG_TILE_K = 64 weights = 32 symbols x 4 bits = 16 packed bytes per row per
// k-chunk. That is what lets this file reuse the qtip2b staging and routing
// verbatim — `launch_qtip2b_moe_route` is codebook-agnostic and is called
// directly by the host wrapper rather than duplicated here.
//
// Codebook dispatch mirrors every other K=4/V=2 launcher in the tree:
// `cb_mult == 0` (CB_MULT_GAUSSIAN_LUT) means "gather the reproduction values
// from the stored 2^16 x 2 table"; nonzero means the computed MCG codebook
// decoded in registers. Both are template-specialized so neither pays for the
// other's branch.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "qtip2b_common.cuh"   // cp.async helpers, q2b_from_f32
#include "qtip_codebook.cuh"   // qtip_cb_value<COMPUTED_CB>

namespace {

// --- Tile geometry. Must match qtip_grouped_gemm.cu / grouped.rs. ----------
constexpr int QLG_TILE_M   = 16;  // (token, slot) pairs per m-tile
constexpr int QLG_TILE_N   = 64;  // weight rows per n-tile (4 warps x 16)
constexpr int QLG_TILE_K   = 64;  // WEIGHTS per k-chunk (16 packed bytes/row)
constexpr int QLG_WARPS    = 4;
constexpr int QLG_THREADS  = QLG_WARPS * 32;
// 32-byte staged row: [0,12) unused, [12,16) history prefix, [16,32) chunk.
constexpr int QLG_WP_STRIDE = 32;
// Window base bit: the chunk starts at bit 16*8 = 128, and the state at
// symbol t needs the 16 bits ENDING at t, i.e. starting 12 bits earlier.
// 128 - 12 = 116. (The qtip2b rung's 114 is 128 - 14, its 2-bit analogue.)
constexpr int QLG_WIN_BIT_BASE = 116;
constexpr uint32_t QLG_INVALID_PAIR = 0xFFFFFFFFu;
constexpr int QLG_MAX_GRID = 1024;

// --- mma.m16n8k16, f32 accumulate (same as the qtip2b rung) ---------------
template <typename T>
struct qlg_mma16816;

template <>
struct qlg_mma16816<__nv_bfloat16> {
    static __device__ __forceinline__ void run(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};

template <>
struct qlg_mma16816<__half> {
    static __device__ __forceinline__ void run(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};

template <typename T>
__device__ __forceinline__ uint32_t qlg_pack2(float lo, float hi);

template <>
__device__ __forceinline__ uint32_t qlg_pack2<__nv_bfloat16>(float lo, float hi) {
    return (uint32_t)__bfloat16_as_ushort(__float2bfloat16(lo)) |
           ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16);
}

template <>
__device__ __forceinline__ uint32_t qlg_pack2<__half>(float lo, float hi) {
    return (uint32_t)__half_as_ushort(__float2half_rn(lo)) |
           ((uint32_t)__half_as_ushort(__float2half_rn(hi)) << 16);
}

// state(t) = nibble_reverse_16(window). Byte swap, then nibble swap within
// each byte: n0|n1<<4|n2<<8|n3<<12  ->  n3|n2<<4|n1<<8|n0<<12, which is
// exactly `((s_{t-3}<<12)|(s_{t-2}<<8)|(s_{t-1}<<4)|s_t)`.
__device__ __forceinline__ uint32_t qlg_state_from_window(uint32_t win16) {
    uint32_t w = win16 & 0xFFFFu;
    w = ((w & 0x00FFu) << 8) | ((w & 0xFF00u) >> 8);
    return ((w & 0x0F0Fu) << 4) | ((w & 0xF0F0u) >> 4);
}

// Decode the V=2 codeword pair covering weights (2t, 2t+1) of a staged packed
// row. `t` is the chunk-local SYMBOL index (0..31). The `sh > 16` guard both
// handles the word crossing and keeps the +1 word read inside the 32-byte
// staged row (at sh == 16 the window ends exactly on the word boundary).
template <bool COMPUTED_CB>
__device__ __forceinline__ float2 qlg_decode_pair_smem(
    const uint8_t* s_row, int t, const float* __restrict__ lut, uint32_t cb_mult
) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(s_row);
    const int bit = 4 * t + QLG_WIN_BIT_BASE;
    const int wi = bit >> 5;
    const uint32_t sh = (uint32_t)(bit & 31);
    uint32_t win = w[wi] >> sh;
    if (sh > 16u) win |= w[wi + 1] << (32u - sh);
    return qtip_cb_value<COMPUTED_CB>(lut, qlg_state_from_window(win), cb_mult);
}

// Stage one k-chunk (activations + packed weight rows) into a double buffer.
// Byte-for-byte the qtip2b staging: same 16 bytes/row/chunk, same 4-byte
// history prefix, same zero-fill-on-invalid cp.async trick.
template <typename T>
__device__ __forceinline__ void qlg_stage_chunk(
    int c,
    T*              s_x_buf,   // [QLG_TILE_M * QLG_TILE_K]
    uint8_t*        s_wp_buf,  // [QLG_TILE_N * QLG_WP_STRIDE]
    const uint32_t* s_pairs,   // [QLG_TILE_M]
    const T*        __restrict__ x,
    const uint8_t*  __restrict__ packed,
    uint32_t expert,
    int n0,
    int n_rows,
    int in_features,
    int packed_per_row
) {
    const int tid = threadIdx.x;
    // Activations: 16 rows x 8 sixteen-byte pieces.
    {
        const int slot = tid >> 3;
        const int piece = tid & 7;
        const uint32_t pair = s_pairs[slot];
        const bool valid = pair != QLG_INVALID_PAIR;
        void* dst = (uint8_t*)(s_x_buf + (size_t)slot * QLG_TILE_K) + piece * 16;
        const uint8_t* src =
            (const uint8_t*)(x + (size_t)(valid ? pair : 0u) * in_features + (size_t)c * QLG_TILE_K)
            + piece * 16;
        q2b_cp_async_16(dst, src, valid);
    }
    // Packed weights: 64 rows; threads 0..63 stage the 4-byte history prefix,
    // threads 64..127 the 16-byte chunk.
    {
        const int r = tid & 63;
        const int half = tid >> 6;
        const int n = n0 + r;
        const bool nvalid = n < n_rows;
        const uint8_t* row_base =
            packed + ((size_t)expert * n_rows + (size_t)(nvalid ? n : 0)) * packed_per_row
            + (size_t)c * (QLG_TILE_K / 4);
        uint8_t* srow = s_wp_buf + (size_t)r * QLG_WP_STRIDE;
        if (half == 0) {
            // Chunk 0 has no history: zero-fill => window bits below stream
            // position 0 are zero, matching the all-zero initial state.
            const bool pvalid = nvalid && c > 0;
            q2b_cp_async_4(srow + 12, pvalid ? (row_base - 4) : row_base, pvalid);
        } else {
            q2b_cp_async_16(srow + 16, row_base, nvalid);
        }
    }
}

template <typename T, bool COMPUTED_CB>
__global__ void __launch_bounds__(QLG_THREADS)
qtip_lut_grouped_gemm_kernel(
    const uint8_t*  __restrict__ packed,         // [E, n_rows, packed_per_row]
    const float*    __restrict__ row_scales,     // [E, n_rows]
    const float*    __restrict__ lut,            // [2^16 * 2] (unused if computed)
    const T*        __restrict__ x,              // [n_pairs, in_features] (rotated)
    const uint32_t* __restrict__ sorted_pairs,   // [n_pairs] grouped by expert
    const uint32_t* __restrict__ tile_expert,    // [num_tiles]
    const uint32_t* __restrict__ tile_row_start, // [num_tiles]
    const uint32_t* __restrict__ offsets,        // [E+1]
    const uint32_t* __restrict__ num_tiles,      // [1] (device-side tile count)
    T*              __restrict__ y,              // [n_pairs, n_rows], pre-zeroed
    int n_rows,
    int packed_per_row,
    int in_features,
    uint32_t cb_mult
) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    __shared__ __align__(16) T       s_x[2][QLG_TILE_M][QLG_TILE_K];
    __shared__ __align__(16) uint8_t s_wp[2][QLG_TILE_N][QLG_WP_STRIDE];
    __shared__ uint32_t s_pairs[QLG_TILE_M];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;  // mma groupID
    const int tig  = lane & 3;   // mma threadID-in-group

    const int n_tiles_n = (n_rows + QLG_TILE_N - 1) / QLG_TILE_N;
    const int live      = (int)*num_tiles * n_tiles_n;
    const int n_chunks  = in_features / QLG_TILE_K;

    for (int flat = blockIdx.x; flat < live; flat += gridDim.x) {
        const int mt = flat / n_tiles_n;
        const int nt = flat % n_tiles_n;

        const uint32_t expert = tile_expert[mt];
        const int row_start   = (int)tile_row_start[mt];
        const int group_end   = (int)offsets[expert + 1];
        const int rows_here   = min(QLG_TILE_M, group_end - row_start);
        const int n0          = nt * QLG_TILE_N;

        __syncthreads();
        if (tid < QLG_TILE_M) {
            s_pairs[tid] =
                tid < rows_here ? sorted_pairs[row_start + tid] : QLG_INVALID_PAIR;
        }
        __syncthreads();

        // B-fragment column scales. Out-of-range rows get scale 0, which also
        // zeroes the (finite but nonzero) decode of their zero-filled bytes.
        float sclB[2];
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            const int n = n0 + warp * 16 + f * 8 + g;
            sclB[f] =
                n < n_rows ? __ldg(row_scales + (size_t)expert * n_rows + n) : 0.0f;
        }

        float acc[2][4];
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[f][i] = 0.0f;
        }

        qlg_stage_chunk<T>(0, &s_x[0][0][0], &s_wp[0][0][0], s_pairs, x, packed,
                           expert, n0, n_rows, in_features, packed_per_row);
        q2b_cp_commit();

        for (int c = 0; c < n_chunks; ++c) {
            const int buf = c & 1;
            if (c + 1 < n_chunks) {
                qlg_stage_chunk<T>(c + 1, &s_x[buf ^ 1][0][0], &s_wp[buf ^ 1][0][0],
                                   s_pairs, x, packed, expert, n0, n_rows,
                                   in_features, packed_per_row);
                q2b_cp_commit();
                q2b_cp_wait<1>();
            } else {
                q2b_cp_wait<0>();
            }
            __syncthreads();

            // 4 k-fragments of 16 weights. Every B weight is decoded exactly
            // once block-wide. `kb + tig*2` is always EVEN, so weights
            // (k, k+1) share one trellis state and come back as one V=2 pair
            // — half the state reconstructions the qtip2b rung performs.
            #pragma unroll
            for (int kf = 0; kf < 4; ++kf) {
                const int kb = kf * 16;
                uint32_t a[4];
                a[0] = *reinterpret_cast<const uint32_t*>(&s_x[buf][g    ][kb + tig * 2    ]);
                a[1] = *reinterpret_cast<const uint32_t*>(&s_x[buf][g + 8][kb + tig * 2    ]);
                a[2] = *reinterpret_cast<const uint32_t*>(&s_x[buf][g    ][kb + tig * 2 + 8]);
                a[3] = *reinterpret_cast<const uint32_t*>(&s_x[buf][g + 8][kb + tig * 2 + 8]);
                #pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const uint8_t* srow = &s_wp[buf][warp * 16 + f * 8 + g][0];
                    const float s = sclB[f];
                    const float2 c01 =
                        qlg_decode_pair_smem<COMPUTED_CB>(srow, (kb + tig * 2) >> 1, lut, cb_mult);
                    const float2 c23 =
                        qlg_decode_pair_smem<COMPUTED_CB>(srow, (kb + tig * 2 + 8) >> 1, lut, cb_mult);
                    const uint32_t b0 = qlg_pack2<T>(c01.x * s, c01.y * s);
                    const uint32_t b1 = qlg_pack2<T>(c23.x * s, c23.y * s);
                    qlg_mma16816<T>::run(acc[f], a, b0, b1);
                }
            }
            __syncthreads();
        }

        // Epilogue: scatter the C tile; each output element written once.
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            const int nb = n0 + warp * 16 + f * 8 + tig * 2;
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int slot = g + half * 8;
                if (slot >= rows_here) continue;
                const uint32_t pair = s_pairs[slot];
                T* out = y + (size_t)pair * n_rows + nb;
                if (nb < n_rows)     out[0] = q2b_from_f32<T>(acc[f][half * 2]);
                if (nb + 1 < n_rows) out[1] = q2b_from_f32<T>(acc[f][half * 2 + 1]);
            }
        }
    }
#endif  // __CUDA_ARCH__ >= 800
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers
//
// Routing is shared with the qtip2b rung: the host wrapper calls
// `launch_qtip2b_moe_route` (histogram + scans/tile-map + grouped scatter),
// which depends only on the routing indices and never on the codebook.
// ============================================================================

extern "C" {

#define QLUT_GROUPED_GEMM_LAUNCHER(NAME, T)                                    \
    void NAME(const uint8_t*  d_packed,                                        \
              const float*    d_row_scales,                                    \
              const float*    d_lut,                                           \
              const T*        d_x_rotated,                                     \
              const uint32_t* d_sorted_pairs,                                  \
              const uint32_t* d_tile_expert,                                   \
              const uint32_t* d_tile_row_start,                                \
              const uint32_t* d_offsets,                                       \
              const uint32_t* d_num_tiles,                                     \
              T*              d_y,                                             \
              int n_rows,                                                      \
              int packed_per_row,                                              \
              int in_features,                                                 \
              int max_m_tiles,                                                 \
              uint32_t cb_mult,                                                \
              cudaStream_t stream) {                                           \
        const int n_tiles_n = (n_rows + QLG_TILE_N - 1) / QLG_TILE_N;          \
        const long max_tiles = (long)max_m_tiles * (long)n_tiles_n;            \
        const int grid =                                                       \
            (int)(max_tiles < (long)QLG_MAX_GRID ? max_tiles : (long)QLG_MAX_GRID); \
        if (grid <= 0) return;                                                 \
        if (cb_mult == 0u) {                                                   \
            qtip_lut_grouped_gemm_kernel<T, false><<<grid, QLG_THREADS, 0, stream>>>( \
                d_packed, d_row_scales, d_lut, d_x_rotated, d_sorted_pairs,    \
                d_tile_expert, d_tile_row_start, d_offsets, d_num_tiles, d_y,  \
                n_rows, packed_per_row, in_features, cb_mult);                 \
        } else {                                                               \
            qtip_lut_grouped_gemm_kernel<T, true><<<grid, QLG_THREADS, 0, stream>>>(  \
                d_packed, d_row_scales, d_lut, d_x_rotated, d_sorted_pairs,    \
                d_tile_expert, d_tile_row_start, d_offsets, d_num_tiles, d_y,  \
                n_rows, packed_per_row, in_features, cb_mult);                 \
        }                                                                      \
    }

QLUT_GROUPED_GEMM_LAUNCHER(launch_qtip_lut_grouped_gemm_bf16, __nv_bfloat16)
QLUT_GROUPED_GEMM_LAUNCHER(launch_qtip_lut_grouped_gemm_f16,  __half)

#undef QLUT_GROUPED_GEMM_LAUNCHER

} // extern "C"
