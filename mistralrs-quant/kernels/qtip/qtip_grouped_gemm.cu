// QTIP bitshift-trellis GROUPED GEMM — batched 2-bit MoE serving (Arc Stage 4).
//
// This is the W2A16 trellis grouped-GEMM: tokens are sorted by expert
// ON-DEVICE (zero host syncs anywhere in the routing path), a ragged tile
// map is built over the per-expert groups, and a persistent tensor-core
// kernel walks the tile list — cp.async double-buffered staging of the
// 2-bit packed trellis bytes + BF16 activations, in-register 3INST decode
// into BF16 fragments, mma.sync m16n8k16 (sm_80 / sm_90 HMMA).
//
// Why this wins: at 2 bits/weight the kernel reads 0.25 byte per weight, so
// there is ~4x more tensor-core compute budget per HBM byte than BF16 — the
// decode ALU hides under mma issue. The per-token gather GEMV path
// (qtip_bitshift.cu) re-reads every routed expert's packed bytes once per
// token; this kernel reads each expert's bytes once per TILE of tokens.
//
// -------------------------------------------------------------------------
// The decode trick: random-access state reconstruction
// -------------------------------------------------------------------------
// The bitshift trellis state is a sliding 16-bit window over the symbol
// stream (state pair j = sym[t-j]), and the packed stream stores that same
// window LSB-first. Therefore
//
//     state(t) = pair_reverse_16( stream_bits[2t-14 .. 2t+1] )
//
// — the state at ANY position is ~4 ALU ops away from a 16-bit window load
// (see q2b_state_from_window in qtip2b_common.cuh). No sequential warm-up
// replay, no cross-thread state chain: every (row, k) weight is decoded
// exactly once, by the thread whose mma B fragment needs it.
//
// -------------------------------------------------------------------------
// Pipeline
// -------------------------------------------------------------------------
//  1) qtip2b_moe_histogram : count pairs per expert (atomics).
//  2) qtip2b_moe_build     : exclusive scans (pairs AND m-tiles) + flatten
//                            the ragged per-expert tile list. Ragged groups
//                            are rounded per-expert to TILE_M so a tile is
//                            always a full scheduling unit — tiny experts
//                            can't strand a CTA mid-group (SonicMoE-style).
//  3) qtip2b_moe_scatter   : stable-by-construction grouped scatter of pair
//                            ids into `sorted_pairs` (atomic cursors).
//  4) qtip2b_grouped_gemm  : persistent CTAs grid-stride the flattened
//                            (m-tile x n-tile) domain; per tile:
//                            cp.async-stage activations [TILE_M x TILE_K]
//                            and packed bytes [TILE_N x (4+16)] double-
//                            buffered, decode B fragments in registers,
//                            mma.sync, scatter the C tile to y[pair, n].
//
// DETERMINISM: the scatter's atomic cursors make the order WITHIN an expert
// group run-dependent, but every output row y[pair, :] is computed from
// that pair's activations alone (fixed k-summation order in the mma chain,
// weights decoded exactly, f32 accumulators) and written exactly once — so
// the kernel's output is bit-identical across runs regardless of intra-
// group placement. `y` must be zero-initialized by the host: rows whose
// router id is out of range are dropped by routing and never written.
//
// ALIGNMENT CONTRACT (validated host-side in cuda_ops.rs):
//   * num_symbols % QG_TILE_K == 0 (=> packed row stride % 16 == 0)
//   * x, packed, y are freshly-allocated contiguous device buffers
//     (>= 256 B base alignment), so every cp.async source is 16 B aligned.
//
// -------------------------------------------------------------------------
// TUNING NOTES (GPU box; semantics must not change):
//   * CTA shape sweep: TILE_M 16 -> 32/48 (stack multiple mma m-tiles per
//     CTA — decode cost is per (n, k), independent of m, so bigger m-tiles
//     amortize it), TILE_N 64 -> 128, warps 4 -> 8.
//   * cp.async depth: 2 stages -> 3/4 (smem is tiny at 2 bpw; the pipeline
//     is latency- not capacity-bound). Sweep QG_MAX_GRID vs
//     cudaOccupancyMaxActiveBlocksPerMultiprocessor * SM count.
//   * Decode placement: in-register per-fragment (current) vs cooperative
//     decode-to-smem-BF16 + ldmatrix.x4 for both A and B; the latter frees
//     the b-fragment shuffle pressure but adds an smem round-trip.
//   * R streams: the window trick removed the sequential state chain, so
//     the LUT rung's independent-rows-per-warp ILP workaround is moot; the
//     analogous knob here is k-fragments in flight (unroll depth of the kf
//     loop) — sweep 2/4/8.
//   * s_wp bank conflicts: 32 B row stride = 8-word stride = up to 4-way
//     conflicts across g; try an XOR swizzle on the word index.
//   * Epilogue: pack paired bf16 writes into one 32-bit store when both
//     columns are in range.
//
// SM80+ (mma.sync bf16 + cp.async). Gated by `has_qtip_kernels` in build.rs;
// on older arches the kernel compiles to an empty body (it is unreachable
// at runtime because has_qtip_kernels is off below SM80).

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "qtip2b_common.cuh"

namespace {

// Tile geometry. Mirrored on the Rust side (src/qtip/grouped.rs) — keep in
// sync with GROUPED_TILE_M / GROUPED_TILE_N / GROUPED_TILE_K.
constexpr int QG_TILE_M   = 16;  // (token, slot) pairs per m-tile = one mma M
constexpr int QG_TILE_N   = 64;  // weight rows per n-tile (4 warps x 16)
constexpr int QG_TILE_K   = 64;  // symbols per k-chunk (16 packed bytes/row)
constexpr int QG_WARPS    = 4;
constexpr int QG_THREADS  = QG_WARPS * 32;
// Staged packed-row layout: bytes [12, 16) = 4-byte history prefix
// (stream bytes 16c-4 .. 16c-1), bytes [16, 32) = this chunk's 16 bytes.
// The 32-byte stride keeps every cp.async destination 16 B / 4 B aligned
// while the prefix stays contiguous with the chunk for window loads.
constexpr int QG_WP_STRIDE = 32;
// Buffer byte 12 corresponds to stream byte 16c-4, so the 16-bit window of
// chunk-local symbol ts starts at buffer bit 2*ts + 18 + 96.
constexpr int QG_WIN_BIT_BASE = 114;
constexpr uint32_t QG_INVALID_PAIR = 0xFFFFFFFFu;
// Persistent-CTA grid cap (see tuning notes).
constexpr int QG_MAX_GRID = 1024;

// ---------------------------------------------------------------------------
// cp.async helpers (sm_80). `valid == false` uses the src-size-0 form, which
// zero-fills the destination without touching the source address.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void q2b_cp_async_16(void* smem_dst, const void* gmem_src, bool valid) {
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(smem_dst);
    const int src_bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(d), "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void q2b_cp_async_4(void* smem_dst, const void* gmem_src, bool valid) {
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(smem_dst);
    const int src_bytes = valid ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(d), "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void q2b_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void q2b_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ---------------------------------------------------------------------------
// mma.sync m16n8k16 row.col f32 accumulate, keyed on the 16-bit dtype.
// Fragment layout (PTX ISA, g = lane>>2, tig = lane&3):
//   A (m16 x k16, 4x .b32): a0=(row g,   k 2tig..+1)  a1=(row g+8, k 2tig..+1)
//                           a2=(row g,   k 2tig+8..9) a3=(row g+8, k 2tig+8..9)
//   B (k16 x n8,  2x .b32): b0=(k 2tig..+1,  n g)     b1=(k 2tig+8..9, n g)
//   C (m16 x n8,  4x f32) : c0=(row g, n 2tig) c1=(row g, n 2tig+1)
//                           c2=(row g+8, n 2tig) c3=(row g+8, n 2tig+1)
// ---------------------------------------------------------------------------
template <typename T>
struct q2b_mma16816;

template <>
struct q2b_mma16816<__nv_bfloat16> {
    static __device__ __forceinline__ void run(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};

template <>
struct q2b_mma16816<__half> {
    static __device__ __forceinline__ void run(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};

// Pack two f32 into one .b32 mma fragment register (lower half = lower index).
template <typename T>
__device__ __forceinline__ uint32_t q2b_pack2(float lo, float hi);

template <>
__device__ __forceinline__ uint32_t q2b_pack2<__nv_bfloat16>(float lo, float hi) {
    return (uint32_t)__bfloat16_as_ushort(__float2bfloat16(lo)) |
           ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16);
}

template <>
__device__ __forceinline__ uint32_t q2b_pack2<__half>(float lo, float hi) {
    return (uint32_t)__half_as_ushort(__float2half_rn(lo)) |
           ((uint32_t)__half_as_ushort(__float2half_rn(hi)) << 16);
}

// Decode the weight at chunk-local symbol `ts` from a staged 32-byte packed
// row: aligned u32 window load + pair-reversal + 3INST codeword. The `sh >
// 16` guard both handles the word crossing and keeps the +1 word read in
// bounds (a crossing window always ends inside the staged bytes).
__device__ __forceinline__ float q2b_decode_smem(const uint8_t* s_row, int ts, uint32_t mult) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(s_row);
    const int bit = 2 * ts + QG_WIN_BIT_BASE;
    const int wi = bit >> 5;
    const uint32_t sh = (uint32_t)(bit & 31);
    uint32_t win = w[wi] >> sh;
    if (sh > 16u) win |= w[wi + 1] << (32u - sh);
    return q2b_decode(q2b_state_from_window(win), mult);
}

// ---------------------------------------------------------------------------
// 1..3) Routing: histogram -> scans + tile map -> grouped scatter.
// ---------------------------------------------------------------------------
constexpr int QG_ROUTE_THREADS = 256;

__global__ void qtip2b_moe_histogram_kernel(
    const uint32_t* __restrict__ indices,   // [n_pairs]
    uint32_t*       __restrict__ counts,    // [E], pre-zeroed
    int n_pairs,
    int num_experts
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    const uint32_t e = indices[p];
    // Out-of-range router ids (broken router) are dropped here; their
    // output rows stay at the host's zero initialization.
    if (e < (uint32_t)num_experts) atomicAdd(&counts[e], 1u);
}

__global__ void qtip2b_moe_build_kernel(
    const uint32_t* __restrict__ counts,         // [E]
    uint32_t*       __restrict__ offsets,        // [E+1]
    uint32_t*       __restrict__ tile_prefix,    // [E]
    uint32_t*       __restrict__ tile_expert,    // [max_m_tiles]
    uint32_t*       __restrict__ tile_row_start, // [max_m_tiles]
    uint32_t*       __restrict__ num_tiles,      // [1]
    int num_experts,
    int tile_m
) {
    // Phase 1 (thread 0, serial): exclusive scans over pair counts and
    // m-tile counts. E is small (8..384 for the MoEs we serve); a serial
    // scan is deterministic and negligible next to the GEMM. Mirrored by
    // `grouped::build_group_tile_map` on the Rust side for CPU testing.
    if (threadIdx.x == 0) {
        uint32_t acc = 0u, tacc = 0u;
        for (int e = 0; e < num_experts; ++e) {
            offsets[e] = acc;
            tile_prefix[e] = tacc;
            acc += counts[e];
            tacc += (counts[e] + (uint32_t)tile_m - 1u) / (uint32_t)tile_m;
        }
        offsets[num_experts] = acc;
        *num_tiles = tacc;
    }
    // __syncthreads carries a block-wide memory fence, so thread 0's global
    // writes above are visible to the whole block below.
    __syncthreads();
    // Phase 2 (parallel): flatten the ragged per-expert tile lists.
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        const uint32_t base = tile_prefix[e];
        const uint32_t nt = (counts[e] + (uint32_t)tile_m - 1u) / (uint32_t)tile_m;
        for (uint32_t i = 0; i < nt; ++i) {
            tile_expert[base + i] = (uint32_t)e;
            tile_row_start[base + i] = offsets[e] + i * (uint32_t)tile_m;
        }
    }
}

__global__ void qtip2b_moe_scatter_kernel(
    const uint32_t* __restrict__ indices,      // [n_pairs]
    const uint32_t* __restrict__ offsets,      // [E+1]
    uint32_t*       __restrict__ cursors,      // [E], pre-zeroed
    uint32_t*       __restrict__ sorted_pairs, // [n_pairs]
    int n_pairs,
    int num_experts
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    const uint32_t e = indices[p];
    if (e >= (uint32_t)num_experts) return;
    const uint32_t pos = offsets[e] + atomicAdd(&cursors[e], 1u);
    sorted_pairs[pos] = (uint32_t)p;
}

// ---------------------------------------------------------------------------
// 4) The grouped GEMM.
// ---------------------------------------------------------------------------

// Stage one k-chunk into a double-buffer stage: gather TILE_M activation
// rows (via sorted pair ids) and TILE_N packed rows (+4-byte history
// prefix), all through cp.async. 128 threads issue exactly 2 copies each.
template <typename T>
__device__ __forceinline__ void q2b_stage_chunk(
    int c,
    T*             s_x_buf,   // [QG_TILE_M * QG_TILE_K]
    uint8_t*       s_wp_buf,  // [QG_TILE_N * QG_WP_STRIDE]
    const uint32_t* s_pairs,  // [QG_TILE_M]
    const T*        __restrict__ x,
    const uint8_t*  __restrict__ packed,
    uint32_t expert,
    int n0,
    int n_rows,
    int num_symbols,
    int packed_per_row
) {
    const int tid = threadIdx.x;
    // Activations: 16 rows x 8 sixteen-byte pieces.
    {
        const int slot = tid >> 3;
        const int piece = tid & 7;
        const uint32_t pair = s_pairs[slot];
        const bool valid = pair != QG_INVALID_PAIR;
        void* dst = (uint8_t*)(s_x_buf + (size_t)slot * QG_TILE_K) + piece * 16;
        const uint8_t* src =
            (const uint8_t*)(x + (size_t)(valid ? pair : 0u) * num_symbols + (size_t)c * QG_TILE_K)
            + piece * 16;
        q2b_cp_async_16(dst, src, valid);
    }
    // Packed weights: 64 rows; threads 0..63 stage the 4-byte history
    // prefix, threads 64..127 the 16-byte chunk.
    {
        const int r = tid & 63;
        const int half = tid >> 6;
        const int n = n0 + r;
        const bool nvalid = n < n_rows;
        const uint8_t* row_base =
            packed + ((size_t)expert * n_rows + (size_t)(nvalid ? n : 0)) * packed_per_row
            + (size_t)c * (QG_TILE_K / 4);
        uint8_t* srow = s_wp_buf + (size_t)r * QG_WP_STRIDE;
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

template <typename T>
__global__ void __launch_bounds__(QG_THREADS)
qtip2b_grouped_gemm_kernel(
    const uint8_t*  __restrict__ packed,         // [E, n_rows, packed_per_row]
    const float*    __restrict__ row_scales,     // [E, n_rows]
    const T*        __restrict__ x,              // [n_pairs, num_symbols] (rotated)
    const uint32_t* __restrict__ sorted_pairs,   // [n_pairs] grouped by expert
    const uint32_t* __restrict__ tile_expert,    // [num_tiles]
    const uint32_t* __restrict__ tile_row_start, // [num_tiles]
    const uint32_t* __restrict__ offsets,        // [E+1]
    const uint32_t* __restrict__ num_tiles,      // [1] (device-side tile count)
    T*              __restrict__ y,              // [n_pairs, n_rows], pre-zeroed
    int n_rows,
    int packed_per_row,
    int num_symbols,
    uint32_t mult
) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    __shared__ __align__(16) T       s_x[2][QG_TILE_M][QG_TILE_K];
    __shared__ __align__(16) uint8_t s_wp[2][QG_TILE_N][QG_WP_STRIDE];
    __shared__ uint32_t s_pairs[QG_TILE_M];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;  // mma groupID
    const int tig  = lane & 3;   // mma threadID-in-group

    const int n_tiles_n = (n_rows + QG_TILE_N - 1) / QG_TILE_N;
    const int live      = (int)*num_tiles * n_tiles_n;
    const int n_chunks  = num_symbols / QG_TILE_K;

    // Persistent CTAs: grid-stride the flattened (m-tile, n-tile) domain.
    // The live tile count is read from device memory — the grid was sized
    // from a host-side upper bound, so no host sync was ever needed.
    for (int flat = blockIdx.x; flat < live; flat += gridDim.x) {
        const int mt = flat / n_tiles_n;
        const int nt = flat % n_tiles_n;

        const uint32_t expert = tile_expert[mt];
        const int row_start   = (int)tile_row_start[mt];
        const int group_end   = (int)offsets[expert + 1];
        const int rows_here   = min(QG_TILE_M, group_end - row_start);
        const int n0          = nt * QG_TILE_N;

        // Tile prologue: publish this tile's pair ids. The leading barrier
        // also fences the previous iteration's smem consumers.
        __syncthreads();
        if (tid < QG_TILE_M) {
            s_pairs[tid] =
                tid < rows_here ? sorted_pairs[row_start + tid] : QG_INVALID_PAIR;
        }
        __syncthreads();

        // B-fragment column scales: weight row n0 + warp*16 + f*8 + g.
        // Out-of-range rows get scale 0, which also zeroes the (finite but
        // nonzero) decode of their zero-filled smem bytes.
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

        // Two-stage cp.async pipeline over the k-chunks.
        q2b_stage_chunk<T>(0, &s_x[0][0][0], &s_wp[0][0][0], s_pairs, x, packed,
                           expert, n0, n_rows, num_symbols, packed_per_row);
        q2b_cp_commit();

        for (int c = 0; c < n_chunks; ++c) {
            const int buf = c & 1;
            if (c + 1 < n_chunks) {
                q2b_stage_chunk<T>(c + 1, &s_x[buf ^ 1][0][0], &s_wp[buf ^ 1][0][0],
                                   s_pairs, x, packed, expert, n0, n_rows,
                                   num_symbols, packed_per_row);
                q2b_cp_commit();
                q2b_cp_wait<1>();
            } else {
                q2b_cp_wait<0>();
            }
            __syncthreads();

            // 4 k-fragments of 16 symbols. A is redundant across warps
            // (cheap smem broadcasts); every B weight is decoded exactly
            // once block-wide (tig 0..3 cover k 0..7/8..15 of row g).
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
                    const float w0 = q2b_decode_smem(srow, kb + tig * 2,     mult) * s;
                    const float w1 = q2b_decode_smem(srow, kb + tig * 2 + 1, mult) * s;
                    const float w2 = q2b_decode_smem(srow, kb + tig * 2 + 8, mult) * s;
                    const float w3 = q2b_decode_smem(srow, kb + tig * 2 + 9, mult) * s;
                    const uint32_t b0 = q2b_pack2<T>(w0, w1);
                    const uint32_t b1 = q2b_pack2<T>(w2, w3);
                    q2b_mma16816<T>::run(acc[f], a, b0, b1);
                }
            }
            __syncthreads();
        }

        // Epilogue: scatter the C tile. c0/c1 -> m slot g, c2/c3 -> slot
        // g+8; columns tig*2, tig*2+1 of the f-th 8-wide n block. Each
        // output element is written exactly once (determinism).
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
// ============================================================================

extern "C" {

// Routing: histogram + scans/tile-map + grouped scatter, all on `stream`.
// `d_counts` and `d_cursors` must be zero-initialized by the caller.
void launch_qtip2b_moe_route(
    const uint32_t* d_indices,        // [n_pairs]
    uint32_t*       d_counts,         // [E], zeroed
    uint32_t*       d_offsets,        // [E+1]
    uint32_t*       d_cursors,        // [E], zeroed
    uint32_t*       d_tile_prefix,    // [E]
    uint32_t*       d_tile_expert,    // [max_m_tiles]
    uint32_t*       d_tile_row_start, // [max_m_tiles]
    uint32_t*       d_num_tiles,      // [1]
    uint32_t*       d_sorted_pairs,   // [n_pairs]
    int n_pairs,
    int num_experts,
    int tile_m,
    cudaStream_t stream
) {
    const int blocks_p = (n_pairs + QG_ROUTE_THREADS - 1) / QG_ROUTE_THREADS;
    qtip2b_moe_histogram_kernel<<<blocks_p, QG_ROUTE_THREADS, 0, stream>>>(
        d_indices, d_counts, n_pairs, num_experts);
    qtip2b_moe_build_kernel<<<1, QG_ROUTE_THREADS, 0, stream>>>(
        d_counts, d_offsets, d_tile_prefix, d_tile_expert, d_tile_row_start,
        d_num_tiles, num_experts, tile_m);
    qtip2b_moe_scatter_kernel<<<blocks_p, QG_ROUTE_THREADS, 0, stream>>>(
        d_indices, d_offsets, d_cursors, d_sorted_pairs, n_pairs, num_experts);
}

#define Q2B_GROUPED_GEMM_LAUNCHER(NAME, T)                                     \
    void NAME(const uint8_t*  d_packed,                                        \
              const float*    d_row_scales,                                    \
              const T*        d_x_rotated,                                     \
              const uint32_t* d_sorted_pairs,                                  \
              const uint32_t* d_tile_expert,                                   \
              const uint32_t* d_tile_row_start,                                \
              const uint32_t* d_offsets,                                       \
              const uint32_t* d_num_tiles,                                     \
              T*              d_y,                                             \
              int n_rows,                                                      \
              int packed_per_row,                                              \
              int num_symbols,                                                 \
              int max_m_tiles,                                                 \
              uint32_t mult,                                                   \
              cudaStream_t stream) {                                           \
        const int n_tiles_n = (n_rows + QG_TILE_N - 1) / QG_TILE_N;            \
        const long max_tiles = (long)max_m_tiles * (long)n_tiles_n;            \
        const int grid =                                                       \
            (int)(max_tiles < (long)QG_MAX_GRID ? max_tiles : (long)QG_MAX_GRID); \
        if (grid <= 0) return;                                                 \
        qtip2b_grouped_gemm_kernel<T><<<grid, QG_THREADS, 0, stream>>>(        \
            d_packed, d_row_scales, d_x_rotated, d_sorted_pairs,               \
            d_tile_expert, d_tile_row_start, d_offsets, d_num_tiles, d_y,      \
            n_rows, packed_per_row, num_symbols, mult);                        \
    }

Q2B_GROUPED_GEMM_LAUNCHER(launch_qtip2b_grouped_gemm_bf16, __nv_bfloat16)
Q2B_GROUPED_GEMM_LAUNCHER(launch_qtip2b_grouped_gemm_f16,  __half)

#undef Q2B_GROUPED_GEMM_LAUNCHER

} // extern "C"
