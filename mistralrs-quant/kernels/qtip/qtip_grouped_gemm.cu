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
// -------------------------------------------------------------------------
// ARCH PATHS (D16)
// -------------------------------------------------------------------------
// Two kernels, picked by the device's RUNTIME compute capability in the
// launcher (not by `__CUDA_ARCH__`, which only says what was compiled):
//
//   sm_80 / sm_89  `qtip2b_grouped_gemm_kernel_sm80` — 16-pair m-tile,
//                  per-position window decode straight into mma fragments.
//                  Unchanged; this is the path with 5/5 hardware parity.
//   sm_90 / sm_100 `qtip2b_grouped_gemm_kernel_wg` — 64-pair m-tile,
//                  contiguous-run decode into a shared-memory weight tile,
//                  dynamic shared memory at the arch's real budget.
//   below sm_80    the launcher returns cudaErrorNotSupported. It does NOT
//                  launch: both bodies are compiled out there, and an empty
//                  kernel returns success while writing nothing (D18).
//
// The tuning note below about "TILE_M 16 -> 32/48" is DONE on the SM90+ path
// and is the largest single win in this file — see the header of the
// warpgroup kernel for why m-tile width, not mma throughput, is the lever.
//
// Gated by `has_qtip_kernels` in build.rs (cc >= 80). cudaforge auto-suffixes
// architectures >= 90 to `sm_90a` / `sm_100a`, so the architecture-specific
// MMAs (`wgmma`, `tcgen05`) are reachable from the build when they land — see
// the dispatch comment above the launchers for what is blocking that today.

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
// D16 arch schedule.
//
// SM90+ runs the warpgroup path (`qtip2b_grouped_gemm_kernel_wg`), whose
// m-tile is fixed at 64 by `wgmma.mma_async`'s M=64 shape. SM80/SM89 keep the
// Ampere `mma.sync` path at m-tile 16.
//
// The ROUTING kernel bins pairs into m-tiles, so route and GEMM must agree on
// `tile_m`. That is why this is a host-visible query
// (`qtip2b_grouped_query_schedule`) mirrored by `grouped::grouped_tile_m_for_cc`
// rather than a constant: getting it wrong mis-bins the tile map, and the
// symptom would be wrong numbers, not an error.
// ---------------------------------------------------------------------------
constexpr int QGW_TILE_M = 64;   // wgmma M — pairs per m-tile on SM90+
constexpr int QGW_TILE_N = 64;   // wgmma N — weight rows per n-tile
constexpr int QGW_TILE_K = QG_TILE_K;  // symbols per k-chunk (host gate unchanged)
constexpr int QGW_WARPS   = 4;   // one warpgroup
constexpr int QGW_THREADS = QGW_WARPS * 32;
// Symbols each thread decodes as ONE contiguous run: 64 rows x 64 symbols /
// 128 threads. Mirrored by `grouped::GROUPED_DECODE_RUN`.
constexpr int QGW_RUN = (QGW_TILE_N * QGW_TILE_K) / QGW_THREADS;  // 32
// Symbols packed in one 32-bit word of the staged stream. A run is read as
// exactly two such words, so the decode loop switches source words once.
constexpr int QGW_SYMS_PER_WORD = 32 / (int)Q2B_K;                // 16
static_assert(QGW_RUN == 2 * QGW_SYMS_PER_WORD,
              "the run decoder reads exactly two 32-bit symbol words per run");
static_assert(QGW_TILE_K % QGW_RUN == 0,
              "runs must tile the k-chunk exactly");
static_assert(QGW_THREADS % QGW_TILE_N == 0,
              "every weight row must be covered by a whole number of threads");

__host__ __device__ __forceinline__ int qg_tile_m_for_cc(int cc_major) {
    return cc_major >= 9 ? QGW_TILE_M : QG_TILE_M;
}

// cp.async helpers (q2b_cp_async_16 / _4 / _commit / _wait<N>) live in
// qtip2b_common.cuh — shared with the gen-2 GEMV pipeline.

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
qtip2b_grouped_gemm_kernel_sm80(
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

// ===========================================================================
// SM90 (Hopper) / SM100 (Blackwell) path — D16 arch specialisation.
// ===========================================================================
//
// WHAT THIS CHANGES vs the Ampere kernel above, and WHY.
//
// The trellis GEMM is **ALU-issue-bound on decode**, not bandwidth-bound. That
// is not a model, it is the project's own hardware result: the gen-2 GEMV
// sweep measured 98/98 variants and every latency-hiding axis came back
// negative (split-K, staged width, warp specialization -38%), concluding "the
// ceiling is per-symbol trellis decode serialization — decode FEWER SYMBOLS,
// do not hide latency". So the two levers that matter are (1) decode fewer
// weights for the same output, and (2) spend fewer ALU ops per weight decoded.
//
//  (1) WIDER M-TILE — 16 -> 64 pairs.
//      Total decode work is `(number of m-tiles) x (expert weight count)`:
//      every m-tile re-reads and re-decodes its expert's whole matrix. At 64
//      routed pairs per expert that is 4 m-tiles at TILE_M=16 and 1 at 64 —
//      **4x less decode AND 4x less HBM traffic** for the same output. Below
//      16 pairs/expert the two schedules issue the same single tile, so this
//      is weakly better everywhere and strictly better at batch. It is also
//      exactly `wgmma`'s and `tcgen05`'s M=64, so the schedule is what the
//      arch-native MMA will want (see the note at the launcher).
//      The cost is 4 m-fragments of accumulator (32 f32/thread) — registers,
//      not the bottleneck.
//
//  (2) SEQUENTIAL RUN DECODE — one window seed per 32 symbols, not per symbol.
//      `q2b_state_from_window` is ~11 ALU ops (two smem words, two shifts, an
//      OR, __brev, adjacent-bit swap). The Ampere kernel pays that for EVERY
//      weight because the mma B-fragment gives each thread runs of length 2.
//      Here each thread instead owns a contiguous run of QGW_RUN symbols of
//      one weight row, seeds it once with the window identity, and steps the
//      trellis recurrence (`q2b_advance_state`, ~2 ops). State reconstruction
//      drops from ~11 ops/weight to ~2.3. Runs stay independent — the seed is
//      still the random-access identity — so there is no cross-thread state
//      chain and no warm-up replay. Property-tested on CPU by
//      `grouped::run_states_2b` (D14: that is a unit test, not a measurement).
//
//      Decoding a run forces the weights through shared memory rather than
//      straight into mma fragments. That is the right trade here (smem is on
//      the LSU, which is idle while the ALU saturates) and it is also
//      MANDATORY for the arch-native MMA: `wgmma`'s B operand must live in
//      shared memory, and `tcgen05` takes no register operands at all.
//
//  (3) THE ARCH'S REAL SHARED-MEMORY BUDGET. The tile uses dynamic shared
//      memory and the launcher opts in with `cudaFuncSetAttribute`. Without
//      that opt-in CUDA caps a block at 48 KB regardless of the 227 KB that
//      compute capability 9.0 and 10.x actually allow (8.9 allows only 99 KB —
//      hence the arch gate, not a constant).
//
// THE MOAT IS INTACT: the packed 2-bit bytes are what cross HBM, and decode
// happens inside this kernel into a single 64x64 tile of shared memory that is
// overwritten every k-chunk. Full-size weights never materialise anywhere.
//
// mma.sync.m16n8k16.f32.bf16.bf16 requires sm_80 or higher and follows the
// onion-layer model, so it is valid on sm_90a and sm_100a alike — it is the
// one bf16 tensor-core path that spans Ampere/Hopper/Blackwell. `wgmma` is
// sm_90a-EXCLUSIVE (PTX "Requires sm_90a"; a-suffix targets "do not follow the
// onion layer model" and "cannot be run on later generation devices") and
// `tcgen05` is sm_100a+, so those are two further, mutually exclusive
// specialisations layered on this same producer stage. See the launcher for
// exactly what blocks them today.
// ---------------------------------------------------------------------------

// Padded smem row strides, in ELEMENTS. Both are chosen so the mma fragment
// reads are bank-conflict-free: a lane reads element `stride*(base+g) + 2*tig`
// as one 32-bit word, i.e. bank `(stride/2*g + tig) mod 32`; stride 72 gives
// `(36g + tig) mod 32 = (4g + tig) mod 32`, which is injective over
// g in 0..7, tig in 0..3. The unpadded stride 64 collapses all eight g to the
// same bank (8-way conflict).
constexpr int QGW_X_STRIDE = QGW_TILE_K + 8;  // 72
constexpr int QGW_B_STRIDE = QGW_TILE_K + 8;  // 72
constexpr int QGW_MFRAGS   = QGW_TILE_M / 16; // 4 mma m-fragments per CTA

// Dynamic-smem byte budget for one CTA of the warpgroup kernel.
template <typename T>
__host__ __device__ __forceinline__ size_t qgw_smem_bytes() {
    return (size_t)2 * QGW_TILE_M * QGW_X_STRIDE * sizeof(T)   // s_x  (double buffered)
         + (size_t)2 * QGW_TILE_N * QG_WP_STRIDE               // s_wp (double buffered)
         + (size_t)QGW_TILE_N * QGW_B_STRIDE * sizeof(T)       // s_bw (decoded, single)
         + (size_t)QGW_TILE_M * sizeof(uint32_t);              // s_pairs
}

// Stage one k-chunk: QGW_TILE_M activation rows (gathered through the sorted
// pair ids) and QGW_TILE_N packed weight rows (+4-byte history prefix).
// 128 threads issue 4 activation copies and 1 packed copy each.
template <typename T>
__device__ __forceinline__ void q2bw_stage_chunk(
    int c,
    T*              s_x_buf,   // [QGW_TILE_M][QGW_X_STRIDE]
    uint8_t*        s_wp_buf,  // [QGW_TILE_N][QG_WP_STRIDE]
    const uint32_t* s_pairs,   // [QGW_TILE_M]
    const T*        __restrict__ x,
    const uint8_t*  __restrict__ packed,
    uint32_t expert,
    int n0,
    int n_rows,
    int num_symbols,
    int packed_per_row
) {
    const int tid = threadIdx.x;
    // Activations: QGW_TILE_M rows x 8 sixteen-byte pieces = 512 copies.
    #pragma unroll
    for (int rep = 0; rep < (QGW_TILE_M * QGW_TILE_K * (int)sizeof(T) / 16) / QGW_THREADS; ++rep) {
        const int cid   = tid + rep * QGW_THREADS;
        const int slot  = cid >> 3;
        const int piece = cid & 7;
        const uint32_t pair = s_pairs[slot];
        const bool valid = pair != QG_INVALID_PAIR;
        void* dst = (uint8_t*)(s_x_buf + (size_t)slot * QGW_X_STRIDE) + piece * 16;
        const uint8_t* src =
            (const uint8_t*)(x + (size_t)(valid ? pair : 0u) * num_symbols
                               + (size_t)c * QGW_TILE_K)
            + piece * 16;
        q2b_cp_async_16(dst, src, valid);
    }
    // Packed weights: 64 rows; threads 0..63 stage the 4-byte history prefix,
    // threads 64..127 this chunk's 16 bytes. Identical contract to the Ampere
    // path (chunk 0 has no history and is zero-filled, which matches the
    // all-zero initial trellis state).
    {
        const int r    = tid & (QGW_TILE_N - 1);
        const int half = tid / QGW_TILE_N;
        const int n    = n0 + r;
        const bool nvalid = n < n_rows;
        const uint8_t* row_base =
            packed + ((size_t)expert * n_rows + (size_t)(nvalid ? n : 0)) * packed_per_row
            + (size_t)c * (QGW_TILE_K / 4);
        uint8_t* srow = s_wp_buf + (size_t)r * QG_WP_STRIDE;
        if (half == 0) {
            const bool pvalid = nvalid && c > 0;
            q2b_cp_async_4(srow + 12, pvalid ? (row_base - 4) : row_base, pvalid);
        } else {
            q2b_cp_async_16(srow + 16, row_base, nvalid);
        }
    }
}

// Decode the staged 64x64 packed tile into `s_bw` as 16-bit activations-dtype,
// one contiguous run of QGW_RUN symbols per thread.
//
// Out-of-range weight rows carry scale 0, which zeroes the (finite) decode of
// their cp.async zero-filled bytes — the same tail-guard-free contract the
// Ampere path uses.
template <typename T>
__device__ __forceinline__ void q2bw_decode_tile(
    T*             s_bw,
    const uint8_t* s_wp_buf,
    const float*   __restrict__ row_scales,
    uint32_t       expert,
    int            n0,
    int            n_rows,
    uint32_t       mult
) {
    const int tid = threadIdx.x;
    const int r   = tid / (QGW_THREADS / QGW_TILE_N);   // weight row in the n-tile
    const int h   = tid % (QGW_THREADS / QGW_TILE_N);   // which run of the row
    const int ts0 = h * QGW_RUN;

    const uint32_t* w =
        reinterpret_cast<const uint32_t*>(s_wp_buf + (size_t)r * QG_WP_STRIDE);

    // Seed: the random-access window identity at the first symbol of the run.
    const int bit     = 2 * ts0 + QG_WIN_BIT_BASE;
    const int wi      = bit >> 5;
    const uint32_t sh = (uint32_t)(bit & 31);
    uint32_t win = w[wi] >> sh;
    if (sh > 16u) win |= w[wi + 1] << (32u - sh);
    uint32_t state = q2b_state_from_window(win);

    // Raw symbols of the run. Chunk-local symbol `ts` sits at staged-buffer
    // bit 2*ts + 128 (byte 16 is where this chunk's 16 bytes begin), so a run
    // of 32 symbols is exactly two aligned 32-bit words.
    const int sw = (2 * ts0 + 128) >> 5;
    const uint32_t b_lo = w[sw];
    const uint32_t b_hi = w[sw + 1];
    uint32_t bits = b_lo >> Q2B_K;   // low bits = symbol ts0+1

    const int n = n0 + r;
    const float s =
        n < n_rows ? __ldg(row_scales + (size_t)expert * n_rows + n) : 0.0f;

    T* out = s_bw + (size_t)r * QGW_B_STRIDE + ts0;
    // Both switch conditions are present even though only one can fire at the
    // current unroll factor: the guard is on the SYMBOL INDEX being consumed,
    // so the loop stays correct if the unroll or QGW_RUN is ever retuned.
    // Keying it on loop parity instead would break silently.
    #pragma unroll
    for (int i = 0; i < QGW_RUN; i += 2) {
        const float w0 = q2b_decode(state, mult) * s;
        if (i + 1 == QGW_SYMS_PER_WORD) bits = b_hi;
        state = q2b_advance_state(state, bits);
        bits >>= Q2B_K;
        const float w1 = q2b_decode(state, mult) * s;
        if (i + 2 == QGW_SYMS_PER_WORD) bits = b_hi;
        state = q2b_advance_state(state, bits);
        bits >>= Q2B_K;
        // Both halves land in one 32-bit store (out + i is 4-byte aligned:
        // QGW_B_STRIDE, QGW_RUN and i are all even).
        *reinterpret_cast<uint32_t*>(out + i) = q2b_pack2<T>(w0, w1);
    }
}

template <typename T>
__global__ void __launch_bounds__(QGW_THREADS)
qtip2b_grouped_gemm_kernel_wg(
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
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    extern __shared__ __align__(16) uint8_t qgw_smem[];
    T*        s_x  = reinterpret_cast<T*>(qgw_smem);
    uint8_t*  s_wp = qgw_smem + (size_t)2 * QGW_TILE_M * QGW_X_STRIDE * sizeof(T);
    T*        s_bw = reinterpret_cast<T*>(s_wp + (size_t)2 * QGW_TILE_N * QG_WP_STRIDE);
    uint32_t* s_pairs =
        reinterpret_cast<uint32_t*>(s_bw + (size_t)QGW_TILE_N * QGW_B_STRIDE);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;  // mma groupID
    const int tig  = lane & 3;   // mma threadID-in-group

    const int n_tiles_n = (n_rows + QGW_TILE_N - 1) / QGW_TILE_N;
    const int live      = (int)*num_tiles * n_tiles_n;
    const int n_chunks  = num_symbols / QGW_TILE_K;

    for (int flat = blockIdx.x; flat < live; flat += gridDim.x) {
        const int mt = flat / n_tiles_n;
        const int nt = flat % n_tiles_n;

        const uint32_t expert = tile_expert[mt];
        const int row_start   = (int)tile_row_start[mt];
        const int group_end   = (int)offsets[expert + 1];
        const int rows_here   = min(QGW_TILE_M, group_end - row_start);
        const int n0          = nt * QGW_TILE_N;

        __syncthreads();
        if (tid < QGW_TILE_M) {
            s_pairs[tid] =
                tid < rows_here ? sorted_pairs[row_start + tid] : QG_INVALID_PAIR;
        }
        __syncthreads();

        float acc[QGW_MFRAGS][2][4];
        #pragma unroll
        for (int m = 0; m < QGW_MFRAGS; ++m) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) acc[m][f][i] = 0.0f;
            }
        }

        q2bw_stage_chunk<T>(0, s_x, s_wp, s_pairs, x, packed, expert, n0, n_rows,
                            num_symbols, packed_per_row);
        q2b_cp_commit();

        for (int c = 0; c < n_chunks; ++c) {
            const int buf = c & 1;
            T*       s_x_cur  = s_x  + (size_t)buf * QGW_TILE_M * QGW_X_STRIDE;
            uint8_t* s_wp_cur = s_wp + (size_t)buf * QGW_TILE_N * QG_WP_STRIDE;
            if (c + 1 < n_chunks) {
                q2bw_stage_chunk<T>(
                    c + 1,
                    s_x + (size_t)(buf ^ 1) * QGW_TILE_M * QGW_X_STRIDE,
                    s_wp + (size_t)(buf ^ 1) * QGW_TILE_N * QG_WP_STRIDE,
                    s_pairs, x, packed, expert, n0, n_rows, num_symbols,
                    packed_per_row);
                q2b_cp_commit();
                q2b_cp_wait<1>();
            } else {
                q2b_cp_wait<0>();
            }
            // Also fences the previous chunk's mma reads of s_bw, which is
            // single-buffered (decode is compute, not a copy, so there is
            // nothing to overlap it with — warp specialization measured -38%
            // on this format).
            __syncthreads();

            q2bw_decode_tile<T>(s_bw, s_wp_cur, row_scales, expert, n0, n_rows,
                                mult);
            __syncthreads();

            #pragma unroll
            for (int kf = 0; kf < QGW_TILE_K / 16; ++kf) {
                const int kb = kf * 16;
                // B fragments: this warp's 16 weight rows, two 8-wide blocks.
                // Each 32-bit load is a (k, k+1) pair of one row — exactly the
                // mma B fragment, no shuffling and no re-decode.
                uint32_t b[2][2];
                #pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const T* brow =
                        s_bw + (size_t)(warp * 16 + f * 8 + g) * QGW_B_STRIDE + kb;
                    b[f][0] = *reinterpret_cast<const uint32_t*>(brow + tig * 2);
                    b[f][1] = *reinterpret_cast<const uint32_t*>(brow + tig * 2 + 8);
                }
                #pragma unroll
                for (int m = 0; m < QGW_MFRAGS; ++m) {
                    const int m0 = m * 16;
                    const T* xr0 = s_x_cur + (size_t)(m0 + g) * QGW_X_STRIDE + kb;
                    const T* xr8 = s_x_cur + (size_t)(m0 + g + 8) * QGW_X_STRIDE + kb;
                    uint32_t a[4];
                    a[0] = *reinterpret_cast<const uint32_t*>(xr0 + tig * 2);
                    a[1] = *reinterpret_cast<const uint32_t*>(xr8 + tig * 2);
                    a[2] = *reinterpret_cast<const uint32_t*>(xr0 + tig * 2 + 8);
                    a[3] = *reinterpret_cast<const uint32_t*>(xr8 + tig * 2 + 8);
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        q2b_mma16816<T>::run(acc[m][f], a, b[f][0], b[f][1]);
                    }
                }
            }
            // REQUIRED: the next iteration issues cp.async into the buffer
            // this loop just read, before reaching any other barrier. Without
            // this the staging races the mma reads — the corruption would be
            // silent (wrong activations, no error).
            __syncthreads();
        }

        // Epilogue: same scatter as the Ampere path, over 4 m-fragments.
        #pragma unroll
        for (int m = 0; m < QGW_MFRAGS; ++m) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int nb = n0 + warp * 16 + f * 8 + tig * 2;
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int slot = m * 16 + g + half * 8;
                    if (slot >= rows_here) continue;
                    const uint32_t pair = s_pairs[slot];
                    T* out = y + (size_t)pair * n_rows + nb;
                    if (nb < n_rows)     out[0] = q2b_from_f32<T>(acc[m][f][half * 2]);
                    if (nb + 1 < n_rows) out[1] = q2b_from_f32<T>(acc[m][f][half * 2 + 1]);
                }
            }
        }
    }
#else
    // Compiled for a pre-Hopper target: this entry point is never dispatched
    // (the host picks by runtime compute capability and errors out when no
    // path matches), so an empty body here is unreachable rather than a silent
    // no-op. Reference the parameters so the signature cannot drift.
    (void)packed; (void)row_scales; (void)x; (void)sorted_pairs;
    (void)tile_expert; (void)tile_row_start; (void)offsets; (void)num_tiles;
    (void)y; (void)n_rows; (void)packed_per_row; (void)num_symbols; (void)mult;
#endif  // __CUDA_ARCH__ >= 900
}

// Claim the arch's real per-block shared-memory budget for the warpgroup
// kernel. Without this opt-in CUDA caps a block at 48 KB regardless of the
// 227 KB that compute capability 9.0 and 10.x allow (8.9 allows only 99 KB) --
// PR #94 met the live form of this. The call is unconditional rather than
// gated on "> 48 KB" so the path is exercised on every process and a later
// tile-size increase cannot silently hit the cap.
//
// Templates cannot be given C language linkage, so this lives here rather than
// beside the launchers.
template <typename T>
cudaError_t qgw_prepare_smem() {
    static bool done = false;
    static cudaError_t status = cudaSuccess;
    if (!done) {
        status = cudaFuncSetAttribute(qtip2b_grouped_gemm_kernel_wg<T>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)qgw_smem_bytes<T>());
        done = true;
    }
    return status;
}

// ---------------------------------------------------------------------------
// THE ARCH WITNESS (D18 #12).
//
// The dispatch below picks the warpgroup path from the DEVICE's compute
// capability, but `qtip2b_grouped_gemm_kernel_wg`'s body is `#if __CUDA_ARCH__
// >= 900` -- a property of the BINARY, not the device. Those two can disagree,
// and when they do nothing says so:
//
//   a build whose archive carries only sm_80 SASS, run on an H200, JITs from
//   `compute_80` PTX -- PTX in which the warpgroup body was already compiled
//   away. `cc_major` is 9, so dispatch takes the wide branch, launches a kernel
//   that writes nothing, and `cudaGetLastError()` returns success. The caller
//   receives its `alloc_zeros` buffer as a shape-correct, error-free, ALL-ZERO
//   MoE layer.
//
// That is the same shape as the launch-status hole this PR closes, one level
// up: the absence of a signal read as a specific signal. `ARC_CUDA_ARCHS`
// (PR #108) is what makes the sm_90 cubin exist, but a guarantee that depends
// on another PR having landed, on nobody building without the env var, and on
// no branch being cut from an intermediate state is a sequencing hope. This is
// the mechanical version.
//
// The witness is a real kernel carrying the SAME `__CUDA_ARCH__` guard as the
// one it vouches for, compiled in the same translation unit with the same arch
// flags. It reports what the running binary actually contains: 900 if the
// warpgroup body survived compilation, otherwise the arch it was compiled for.
__global__ void qgw_arch_witness_kernel(int* out) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    if (threadIdx.x == 0) *out = 900;
#else
    if (threadIdx.x == 0) *out = (int)__CUDA_ARCH__;
#endif
}

struct QgwArchWitness {
    cudaError_t status;
    int         arch;  // 900 => the warpgroup body is present in this binary
};

// One launch and one sync, on first use.
//
// It deliberately does NOT default `arch` on failure: a witness that could not
// be taken is `status != cudaSuccess`, and the caller must refuse rather than
// assume. "Could not answer" is not "answered yes".
QgwArchWitness qgw_take_arch_witness() {
    QgwArchWitness w{cudaSuccess, 0};

    // If the legacy stream is capturing, a launch on it is APPENDED TO A GRAPH
    // rather than executed, and the readback below would return the memset 0 --
    // reported as "this binary has no SM90 device code" and blocking a path
    // that works perfectly. A false negative is still a wrong answer, so refuse
    // to answer here and let the caller retry outside capture.
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    const cudaError_t cap_err = cudaStreamIsCapturing(0, &cap);
    if (cap_err != cudaSuccess) {
        w.status = cap_err;
        return w;
    }
    if (cap != cudaStreamCaptureStatusNone) {
        w.status = cudaErrorStreamCaptureUnsupported;
        return w;
    }

    int* d = nullptr;
    w.status = cudaMalloc(&d, sizeof(int));
    if (w.status != cudaSuccess) return w;
    w.status = cudaMemset(d, 0, sizeof(int));
    if (w.status == cudaSuccess) {
        qgw_arch_witness_kernel<<<1, 1>>>(d);
        // Same D18 pairing as every launcher here: a sync returns success for
        // a kernel that never launched, so the launch status is read too.
        const cudaError_t launch_err = cudaGetLastError();
        const cudaError_t sync_err   = cudaStreamSynchronize(0);
        w.status = (launch_err != cudaSuccess) ? launch_err : sync_err;
        if (w.status == cudaSuccess) {
            w.status = cudaMemcpy(&w.arch, d, sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
    cudaFree(d);  // the witness result is already on the host
    return w;
}

// Cached only once it has actually ANSWERED. Caching a failure would be the
// same bug in slow motion: a witness that could not be taken during graph
// capture would poison every later call, and the path would stay refused long
// after the condition cleared. So a non-success result is retried.
//
// Two threads racing the first call may both take the witness. That is benign
// and deliberate: the probe is idempotent and its result is a deterministic
// property of the binary, so both writes store identical bytes. The
// alternative -- a lock on a path that runs per MoE layer -- costs more than
// the duplicate.
const QgwArchWitness& qgw_arch_witness() {
    static QgwArchWitness w{cudaErrorNotReady, 0};  // "not taken yet"
    if (w.status != cudaSuccess) {
        w = qgw_take_arch_witness();
    }
    return w;
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

// ---------------------------------------------------------------------------
// D18: every launcher below returns a `cudaError_t` as `int`, and every one of
// them ends with `cudaGetLastError()`.
//
// Before this, the grouped path had NO launch-status check anywhere (the GEMV
// path's `check_gather_gemv_pairs` guard exists precisely because a discarded
// launch status hands the caller the zero-initialized output buffer — a
// silently all-zero MoE layer, shape-correct and error-free). `cudaGetLastError`
// is the call that distinguishes "the kernel ran" from "the kernel never
// launched": `cudaDeviceSynchronize` returns success for a kernel that never
// started.
// ---------------------------------------------------------------------------

// Which m-tile schedule this device's compute capability selects, and the
// capability itself. Returns a cudaError_t; on failure the out-params are
// untouched and the caller MUST NOT fall back to a default (a wrong `tile_m`
// silently mis-bins the tile map).
//
// Mirrored by `grouped::grouped_tile_m_for_cc` on the Rust side.
int qtip2b_grouped_query_schedule(int* out_cc_major, int* out_cc_minor, int* out_tile_m,
                                  int* out_arch_witness) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return (int)err;
    int major = 0, minor = 0;
    err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (err != cudaSuccess) return (int)err;
    err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    if (err != cudaSuccess) return (int)err;
    // What the DEVICE is, and separately what the BINARY contains. Reporting
    // only the first is what lets an all-zero MoE layer look valid.
    const QgwArchWitness& w = qgw_arch_witness();
    if (w.status != cudaSuccess) return (int)w.status;
    *out_cc_major = major;
    *out_cc_minor = minor;
    *out_tile_m = qg_tile_m_for_cc(major);
    *out_arch_witness = w.arch;
    return (int)cudaSuccess;
}

// Routing: histogram + scans/tile-map + grouped scatter, all on `stream`.
// `d_counts` and `d_cursors` must be zero-initialized by the caller.
int launch_qtip2b_moe_route(
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
    // "Nothing to route" is a legitimate state, but "the grid would have been
    // empty" is not something the caller can distinguish from "it ran" once
    // the zeroed buffers come back, so it is reported.
    if (n_pairs <= 0 || num_experts <= 0 || tile_m <= 0) {
        return (int)cudaErrorInvalidConfiguration;
    }
    const int blocks_p = (n_pairs + QG_ROUTE_THREADS - 1) / QG_ROUTE_THREADS;
    qtip2b_moe_histogram_kernel<<<blocks_p, QG_ROUTE_THREADS, 0, stream>>>(
        d_indices, d_counts, n_pairs, num_experts);
    qtip2b_moe_build_kernel<<<1, QG_ROUTE_THREADS, 0, stream>>>(
        d_counts, d_offsets, d_tile_prefix, d_tile_expert, d_tile_row_start,
        d_num_tiles, num_experts, tile_m);
    qtip2b_moe_scatter_kernel<<<blocks_p, QG_ROUTE_THREADS, 0, stream>>>(
        d_indices, d_offsets, d_cursors, d_sorted_pairs, n_pairs, num_experts);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// D16 DISPATCH. Which arch path a launch takes is decided here, from the
// device's runtime compute capability — not from `__CUDA_ARCH__` alone, which
// only says what was compiled.
//
//   cc >= 9.0  -> `qtip2b_grouped_gemm_kernel_wg`  (64-pair m-tile, run
//                 decode, dynamic smem at the arch's real budget)
//   cc >= 8.0  -> `qtip2b_grouped_gemm_kernel_sm80` (16-pair m-tile)
//   cc <  8.0  -> **error**, never a silent no-op. Both kernel bodies are
//                 `#if`-compiled away below their arch, and a launch of an
//                 empty kernel writes nothing while returning success — the
//                 caller would receive its zero-initialized buffer as a valid,
//                 shape-correct, all-zero MoE output (D18).
//
// WHAT IS NOT HERE YET, stated plainly rather than implied:
// `wgmma.mma_async` (sm_90a) and `tcgen05.mma` (sm_100a/sm_103a) are the
// arch-native MMAs, and this kernel's producer stage — decode into a shared
// -memory tile — is already the shape both of them require (wgmma's B operand
// must be in shared memory; tcgen05 takes no register operands at all). The
// consumer here is still `mma.sync.m16n8k16`, which PTX documents as "requires
// sm_80 or higher" under the onion-layer model and is therefore valid, but not
// native, on sm_90a and sm_100a.
//
// The one thing blocking the swap is the B-operand shared-memory layout: the
// 64-bit matrix descriptor pins swizzle mode, leading-dimension byte offset
// and stride byte offset to a core-matrix layout that the PTX doc specifies
// semantically rather than by a numeric table, and a wrong descriptor produces
// wrong numbers, not an error. That is not something to guess at on the one
// kernel nobody else can adopt. It is settled by a ~30-line descriptor probe
// on any sm_90a box (build a known tile, run one wgmma, diff against this
// kernel's mma.sync result) — minutes of GPU time, and it converts the swap
// from guesswork into a checked change.
// ---------------------------------------------------------------------------

#define Q2B_GROUPED_GEMM_LAUNCHER(NAME, T)                                     \
    int NAME(const uint8_t*  d_packed,                                         \
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
        int dev = 0;                                                           \
        cudaError_t err = cudaGetDevice(&dev);                                 \
        if (err != cudaSuccess) return (int)err;                               \
        int cc_major = 0;                                                      \
        err = cudaDeviceGetAttribute(                                          \
            &cc_major, cudaDevAttrComputeCapabilityMajor, dev);                \
        if (err != cudaSuccess) return (int)err;                               \
        /* Pre-Ampere: BOTH kernel bodies are compiled out, so a launch would  \
           write nothing and still return success. Report it (D18). */         \
        if (cc_major < 8) return (int)cudaErrorNotSupported;                   \
        const bool wide = cc_major >= 9;                                       \
        const int tile_n = wide ? QGW_TILE_N : QG_TILE_N;                      \
        const int n_tiles_n = (n_rows + tile_n - 1) / tile_n;                  \
        const long max_tiles = (long)max_m_tiles * (long)n_tiles_n;            \
        const int grid =                                                       \
            (int)(max_tiles < (long)QG_MAX_GRID ? max_tiles : (long)QG_MAX_GRID); \
        /* An empty grid means the caller's tile bookkeeping disagrees with    \
           ours; that must not be reported as a completed launch (D18). */     \
        if (grid <= 0) return (int)cudaErrorInvalidConfiguration;              \
        if (wide) {                                                            \
            /* D18 #12: the device is SM90+, but is the SM90 kernel BODY in    \
               this binary? If not, the launch below writes nothing and        \
               reports success. Defence in depth -- the Rust caller checks the \
               same witness and can say it in words; this keeps the C ABI safe \
               for any other caller. */                                        \
            const QgwArchWitness& w = qgw_arch_witness();                      \
            if (w.status != cudaSuccess) return (int)w.status;                 \
            if (w.arch < 900) return (int)cudaErrorInvalidDeviceFunction;      \
            err = qgw_prepare_smem<T>();                                       \
            if (err != cudaSuccess) return (int)err;                           \
            const size_t smem = qgw_smem_bytes<T>();                           \
            qtip2b_grouped_gemm_kernel_wg<T>                                   \
                <<<grid, QGW_THREADS, smem, stream>>>(                         \
                    d_packed, d_row_scales, d_x_rotated, d_sorted_pairs,       \
                    d_tile_expert, d_tile_row_start, d_offsets, d_num_tiles,   \
                    d_y, n_rows, packed_per_row, num_symbols, mult);           \
        } else {                                                               \
            qtip2b_grouped_gemm_kernel_sm80<T>                                 \
                <<<grid, QG_THREADS, 0, stream>>>(                             \
                    d_packed, d_row_scales, d_x_rotated, d_sorted_pairs,       \
                    d_tile_expert, d_tile_row_start, d_offsets, d_num_tiles,   \
                    d_y, n_rows, packed_per_row, num_symbols, mult);           \
        }                                                                      \
        return (int)cudaGetLastError();                                        \
    }

Q2B_GROUPED_GEMM_LAUNCHER(launch_qtip2b_grouped_gemm_bf16, __nv_bfloat16)
Q2B_GROUPED_GEMM_LAUNCHER(launch_qtip2b_grouped_gemm_f16,  __half)

#undef Q2B_GROUPED_GEMM_LAUNCHER

} // extern "C"
