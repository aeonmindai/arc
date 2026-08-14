// SECOND-GENERATION autotuner grid for the qtip2b bitshift-trellis fused
// decode+GEMV. Companion TU to qtip_bitshift_tune.cu (gen-1); the two share
// one variant-id space (gen-1 owns 0..43, this file owns 44..N) and one
// dispatch entry (`launch_qtip2b_gemv_tuned_*`, which forwards here).
//
// -------------------------------------------------------------------------
// Why a second grid
// -------------------------------------------------------------------------
// The session-4 H200 sweep of the gen-1 grid put the winner at the SHALLOW
// edge: w4_r2_i1_v2 / w8_r4_i1_v2, 36 us at b=1 and ~467-489 GB/s marginal
// (9.7-10.2% of the 4.8 TB/s peak) — a 2.3x lift over the legacy kernel, but
// latency-hiding is barely engaged. Every deeper gen-1 variant LOST
// monotonically (ILP=2 -> ~7%, ILP=4 -> ~3-4% of peak) because "deeper ILP"
// there meant *more concurrent trellis streams per thread out of GLOBAL
// memory*: each extra stream added a live packed word, a live 16-bit state,
// and a live `xg[ILP][GROUP]` activation block, so register pressure grew as
// ILP*(R*(W+4) + 4*GROUP) bytes — at ILP=4, W=4 that is a 64-float
// activation array per thread. Spills + serial state chains ate the win.
//
// Gen-2 attacks the latency structurally instead of by adding streams:
//
//  1. cp.async DOUBLE/TRIPLE-BUFFERED STAGING (Ampere+). The block's packed
//     rows are copied into smem one K-TILE at a time in 16-byte async
//     chunks, with tile t+STAGES-1 in flight while tile t decodes. This is
//     the Marlin-style pipeline the trellis grouped GEMM already runs
//     (qtip_grouped_gemm.cu), ported to the GEMV. Gen-1's `STAGE=1` was NOT
//     this: it bulk-copied whole rows BYTE-AT-A-TIME in a prologue and then
//     blocked on __syncthreads, so the global latency was fully exposed
//     before any decode started.
//  2. DEEPER ILP DONE RIGHT. All ILP*R packed words + trellis states are
//     loaded up front from STAGED SMEM (independent LDS in flight, ~30-cycle
//     latency instead of ~500), then the decode runs stream-outer /
//     symbol-inner / row-inner, so only ONE stream's activation block
//     (GROUP = 4*W floats) is live at a time. Register cost is now
//     ILP*R*(W+4) + 4*W bytes instead of ILP*(R*(W+4) + 4*GROUP).
//  3. SPLIT-K (KS = 1/2/4). The window-based state reconstruction
//     (`q2b_state_from_window`) makes any K-offset a valid entry point, so
//     the consumer warps are partitioned into KS K-groups that each sweep
//     1/KS of the row and reduce through smem in FIXED kg order (bit-exact
//     run to run — no atomics). Rows/block drops by KS, so blocks/row rises
//     by KS: the N=2048 x b=1 case goes from 128 blocks (< 132 SMs, one
//     under-filled wave) to 256/512.
//  4. WIDER STAGED LOADS (W = 2/4/8 bytes per lane-step, GROUP = 4*W
//     symbols) now served from smem, where a wide load costs no extra HBM
//     latency exposure.
//  5. WARP SPECIALIZATION (PRODW = 0/1): with PRODW=1 the block gets one
//     extra warp that ONLY issues cp.async + waits, so consumer warps never
//     execute an address-generation/copy instruction. Both paths execute the
//     same two __syncthreads per tile, so the barrier structure is identical.
//
// -------------------------------------------------------------------------
// Geometry (all compile-time)
// -------------------------------------------------------------------------
//   GROUP = 4*W                symbols decoded per lane-step per stream
//   TB    = 32*W*ILP           staged bytes per row per tile (one warp pass
//                              over ILP streams covers exactly one tile)
//   SROW  = TB + 16            staged row stride; bytes [12,16) hold the
//                              4-byte history prefix (stream bytes
//                              tile_start-4 .. -1) so the 16-bit state
//                              window of the tile's first symbol is present,
//                              and the 16-byte pad keeps every cp.async
//                              destination 16-byte aligned
//   WPK   = CW/KS              consumer warps per K-group
//   RPB   = WPK*R              rows per block
//   staged rows per stage = KS*RPB = CW*R (independent of KS)
//   smem  = STAGES*CW*R*SROW (+ CW*R f32 when KS>1)   [static, <= 48 KiB]
//
// Grid: (ceil(n_rows / RPB), n_pairs); block: (CW + PRODW) * 32 threads.
//
// -------------------------------------------------------------------------
// Correctness contract — IDENTICAL to the legacy kernel and the gen-1 grid
// -------------------------------------------------------------------------
//   * `q2b_decode` (f32-summed fp16 halves) is untouched, so every weight is
//     bit-identical to the CPU reference; only the dot-product association
//     order differs between variants (each variant is deterministic: the
//     k-order is fixed and the split-K reduction runs in fixed group order).
//   * Out-of-range expert id -> the block writes zeros and returns, never
//     reading out of bounds.
//   * Out-of-range rows -> staged with the cp.async src-size-0 form
//     (zero-fill, source never touched) AND scale 0, and never written.
//   * A variant is applicable only when `num_symbols == 4*packed_per_row`
//     and `packed_per_row % (KS*TB) == 0`; otherwise the launcher returns -1
//     and the Rust dispatch falls back to the legacy kernel. That is the
//     ONLY shape constraint — smem is shape-independent here, unlike gen-1.
//
// SM80+ (cp.async + __nv_bfloat16). Gated by `has_qtip_kernels` in build.rs;
// below SM80 the cp.async helpers degrade to synchronous copies so the TU
// still compiles (the kernels are unreachable there).

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "qtip2b_common.cuh"

namespace {

// Packed-word type per staged load width. Spelled as the exact builtin types
// (uint64_t may alias `unsigned long` on LP64).
template <int W> struct q2b_g2_wordty;
template <> struct q2b_g2_wordty<2> { using type = unsigned short; };
template <> struct q2b_g2_wordty<4> { using type = unsigned int; };
template <> struct q2b_g2_wordty<8> { using type = unsigned long long; };

// Static shared memory ceiling (no cudaFuncSetAttribute opt-in): exceeding it
// is a compile error via the static_assert below, not a runtime surprise.
constexpr int Q2B_G2_SMEM_LIMIT = 48 * 1024;

template <int CW, int R, int ILP, int W, int KS, int STAGES, int PRODW>
struct q2b_g2_geom {
    static constexpr int GROUP   = 4 * W;
    static constexpr int TB      = 32 * W * ILP;
    static constexpr int SROW    = TB + 16;
    static constexpr int PIECES  = TB / 16;          // 16-byte copies per row
    static constexpr int WPK     = CW / KS;
    static constexpr int RPB     = WPK * R;
    static constexpr int NROWBUF = KS * RPB;         // == CW*R
    static constexpr int THREADS = (CW + PRODW) * 32;
    static constexpr int NISS    = (PRODW > 0) ? PRODW * 32 : CW * 32;
    static constexpr int PKBYTES = STAGES * NROWBUF * SROW;
    static constexpr int REDLEN  = (KS > 1) ? NROWBUF : 1;
    static constexpr int SMEM    = PKBYTES + ((KS > 1) ? NROWBUF * 4 : 0);

    static_assert(CW % KS == 0, "consumer warps must split evenly across K-groups");
    static_assert(TB % 16 == 0, "staged tile must be a multiple of the 16-byte cp.async unit");
    static_assert(THREADS <= 1024, "block too large");
    static_assert(SMEM <= Q2B_G2_SMEM_LIMIT, "variant exceeds the 48 KiB static smem budget");
};

// ---------------------------------------------------------------------------
// The gen-2 pipelined kernel.
// ---------------------------------------------------------------------------
template <typename T, int CW, int R, int ILP, int W, int KS, int STAGES, int PRODW, int MINB>
__global__ void __launch_bounds__((CW + PRODW) * 32, MINB)
qtip2b_gemv_gen2_kernel(
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
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    using G  = q2b_g2_geom<CW, R, ILP, W, KS, STAGES, PRODW>;
    using WT = typename q2b_g2_wordty<W>::type;

    __shared__ __align__(16) uint8_t s_pk[G::PKBYTES];
    __shared__ float s_red[G::REDLEN];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int pair = blockIdx.y;
    const int row0_block = blockIdx.x * G::RPB;
    if (row0_block >= n_rows || pair >= n_pairs) return;

    const uint32_t expert = (indices != nullptr) ? __ldg(indices + pair) : 0u;
    if (expert >= (uint32_t)num_experts) {
        // Out-of-range expert id from a broken router: write zeros, never
        // read out of bounds (same contract as the legacy kernel).
        for (int i = tid; i < G::RPB; i += G::THREADS) {
            const int row = row0_block + i;
            if (row < n_rows) y[(size_t)pair * n_rows + row] = q2b_from_f32<T>(0.0f);
        }
        return;
    }

    const size_t erow0  = (size_t)expert * (size_t)n_rows;
    const int    kbytes = packed_per_row / KS;   // packed bytes per K-group
    const int    ntiles = kbytes / G::TB;        // exact (launcher-checked)

    const bool is_producer = (PRODW > 0) ? (warp >= CW) : true;
    const bool is_consumer = (PRODW > 0) ? (warp < CW) : true;
    const int  iss_tid     = (PRODW > 0) ? (tid - CW * 32) : tid;

    // ---- stage one K-tile of every (K-group, row) into buffer `buf` -------
    // Copies past the last tile, and rows past n_rows, use the src-size-0
    // cp.async form: the destination is zero-filled and the source address
    // is never dereferenced (it is still kept in bounds).
    auto stage_tile = [&](int buf, int t) {
        uint8_t* base = s_pk + (size_t)buf * G::NROWBUF * G::SROW;
        const bool live = (t < ntiles);
        const int  tt   = live ? t : 0;
        // Row bodies: 16-byte chunks, consecutive issuing threads take
        // consecutive chunks of the same row -> fully coalesced.
        for (int i = iss_tid; i < G::NROWBUF * G::PIECES; i += G::NISS) {
            const int krow  = i / G::PIECES;
            const int piece = i - krow * G::PIECES;
            const int kg    = krow / G::RPB;
            const int rr    = krow - kg * G::RPB;
            const int row   = row0_block + rr;
            const bool ok   = live && (row < n_rows);
            const size_t goff = (erow0 + (size_t)(ok ? row : 0)) * (size_t)packed_per_row
                              + (size_t)kg * (size_t)kbytes
                              + (size_t)tt * G::TB
                              + (size_t)piece * 16;
            q2b_cp_async_16(base + (size_t)krow * G::SROW + 16 + piece * 16,
                            packed + goff, ok);
        }
        // 4-byte history prefixes (stream bytes tile_start-4 .. tile_start-1).
        // The very first tile of the row has no history: zero-fill, which is
        // exactly the all-zero initial trellis state.
        for (int krow = iss_tid; krow < G::NROWBUF; krow += G::NISS) {
            const int kg  = krow / G::RPB;
            const int rr  = krow - kg * G::RPB;
            const int row = row0_block + rr;
            const int start = kg * kbytes + tt * G::TB;
            const bool ok = live && (row < n_rows) && (start >= 4);
            const size_t goff = (erow0 + (size_t)((row < n_rows) ? row : 0)) * (size_t)packed_per_row
                              + (size_t)(start >= 4 ? start - 4 : 0);
            q2b_cp_async_4(base + (size_t)krow * G::SROW + 12, packed + goff, ok);
        }
    };

    // ---- consumer set-up -------------------------------------------------
    const int kg  = is_consumer ? (warp / G::WPK) : 0;
    const int wi  = is_consumer ? (warp % G::WPK) : 0;
    const int row0 = row0_block + wi * R;

    float scl[R];
    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.0f;
        const int row = row0 + r;
        scl[r] = (is_consumer && row < n_rows) ? __ldg(row_scales + erow0 + row) : 0.0f;
    }
    const T* __restrict__ x_pair = x + (size_t)pair * num_symbols;
    const int xbase_kg = 4 * (kg * kbytes);   // first symbol of this K-group

    // ---- software pipeline ----------------------------------------------
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (is_producer) {
            stage_tile(s, s);
            q2b_cp_commit();
        }
    }

    for (int t = 0; t < ntiles; ++t) {
        if (is_producer) {
            // Always issue (out-of-range tiles are zero-fill no-ops), so the
            // outstanding-group count is uniform and the wait depth is exact.
            stage_tile((t + STAGES - 1) % STAGES, t + STAGES - 1);
            q2b_cp_commit();
            q2b_cp_wait<STAGES - 1>();
        }
        __syncthreads();

        if (is_consumer) {
            const uint8_t* __restrict__ rbase =
                s_pk + (size_t)(t % STAGES) * G::NROWBUF * G::SROW
                     + (size_t)(kg * G::RPB + wi * R) * G::SROW + 16;
            const int xbase_t = xbase_kg + 4 * (t * G::TB);

            // All ILP*R staged words + entry states in flight at once.
            WT       w[ILP][R];
            uint32_t st[ILP][R];
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const int boff = u * (32 * W) + lane * W;
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint8_t* p = rbase + (size_t)r * G::SROW + boff;
                    w[u][r] = *reinterpret_cast<const WT*>(p);
                    // State before this segment's first symbol: pair-reversal
                    // of the 16 stream bits ending at its predecessor, i.e.
                    // the aligned u16 two bytes back (in the history pad when
                    // boff == 0).
                    st[u][r] = q2b_state_from_window(
                        (uint32_t)(*reinterpret_cast<const unsigned short*>(p - 2)));
                }
            }
            // Stream-outer / symbol-inner / row-inner: only ONE stream's
            // GROUP activations are live, and the R rows at each symbol step
            // are independent decode+FMA chains.
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const int xb = xbase_t + 4 * (u * (32 * W) + lane * W);
                float xg[G::GROUP];
                #pragma unroll
                for (int j = 0; j < G::GROUP; ++j) {
                    xg[j] = q2b_to_f32<T>(__ldg(x_pair + xb + j));
                }
                #pragma unroll
                for (int j = 0; j < G::GROUP; ++j) {
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const uint32_t sym = (uint32_t)((w[u][r] >> (2 * j)) & (WT)0x3u);
                        st[u][r] = ((st[u][r] << Q2B_K) | sym) & Q2B_STATE_MASK;
                        acc[r] = fmaf(q2b_decode(st[u][r], mult) * scl[r], xg[j], acc[r]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- epilogue --------------------------------------------------------
    if constexpr (KS == 1) {
        if (is_consumer) {
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const float a = q2b_warp_reduce_sum(acc[r]);
                const int row = row0 + r;
                if (lane == 0 && row < n_rows) {
                    y[(size_t)pair * n_rows + row] = q2b_from_f32<T>(a);
                }
            }
        }
    } else {
        if (is_consumer) {
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const float a = q2b_warp_reduce_sum(acc[r]);
                if (lane == 0) s_red[kg * G::RPB + wi * R + r] = a;
            }
        }
        __syncthreads();
        // Fixed K-group order => bit-identical across runs.
        for (int i = tid; i < G::RPB; i += G::THREADS) {
            float s = 0.0f;
            #pragma unroll
            for (int g = 0; g < KS; ++g) s += s_red[g * G::RPB + i];
            const int row = row0_block + i;
            if (row < n_rows) y[(size_t)pair * n_rows + row] = q2b_from_f32<T>(s);
        }
    }
#endif  // __CUDA_ARCH__ >= 800
}

// ---------------------------------------------------------------------------
// Variant table. X(local_id, CW, R, ILP, W, KS, STAGES, PRODW, MINB) — local
// ids are dense 0..N-1 and are exposed as GLOBAL ids `44 + local_id`
// (Q2B_GEN2_ID_BASE, mirrored by Q2B_TUNE_NUM_VARIANTS in the gen-1 TU).
// The global ids round-trip through tune_results.json and the baked table:
// treat them as ABI — append only, never renumber.
//
//   0..5    gen-1 winner geometries ported onto the cp.async pipeline
//           (STAGES 2 vs 3 duals)
//   6..13   ILP depth at W=2 (staged tile 64 -> 512 B/row)
//   14..21  wider staged loads, W=4
//   22..27  wider staged loads, W=8
//   28..39  split-K (KS 2/4): blocks per row x2 / x4
//   40..47  block width (CW 4/16) + __launch_bounds__ min-blocks hints
//   48..53  warp specialization (one dedicated cp.async producer warp)
// ---------------------------------------------------------------------------
#define Q2B_GEN2_VARIANTS(X) \
    X( 0,  8, 2, 1, 2, 1, 2, 0, 1) \
    X( 1,  8, 2, 1, 2, 1, 3, 0, 1) \
    X( 2,  8, 4, 1, 2, 1, 2, 0, 1) \
    X( 3,  8, 4, 1, 2, 1, 3, 0, 1) \
    X( 4,  4, 2, 1, 2, 1, 2, 0, 1) \
    X( 5,  4, 2, 1, 2, 1, 3, 0, 1) \
    X( 6,  8, 2, 2, 2, 1, 2, 0, 1) \
    X( 7,  8, 2, 2, 2, 1, 3, 0, 1) \
    X( 8,  8, 2, 4, 2, 1, 2, 0, 1) \
    X( 9,  8, 2, 4, 2, 1, 3, 0, 1) \
    X(10,  8, 4, 2, 2, 1, 3, 0, 1) \
    X(11,  8, 1, 4, 2, 1, 3, 0, 1) \
    X(12,  8, 2, 8, 2, 1, 3, 0, 1) \
    X(13,  8, 1, 8, 2, 1, 3, 0, 1) \
    X(14,  8, 2, 1, 4, 1, 2, 0, 1) \
    X(15,  8, 2, 1, 4, 1, 3, 0, 1) \
    X(16,  8, 2, 2, 4, 1, 3, 0, 1) \
    X(17,  8, 2, 4, 4, 1, 3, 0, 1) \
    X(18,  8, 4, 1, 4, 1, 3, 0, 1) \
    X(19,  8, 4, 2, 4, 1, 3, 0, 1) \
    X(20,  4, 2, 2, 4, 1, 3, 0, 1) \
    X(21,  8, 1, 2, 4, 1, 3, 0, 1) \
    X(22,  8, 2, 1, 8, 1, 2, 0, 1) \
    X(23,  8, 2, 1, 8, 1, 3, 0, 1) \
    X(24,  8, 2, 2, 8, 1, 3, 0, 1) \
    X(25,  8, 1, 4, 8, 1, 3, 0, 1) \
    X(26,  8, 4, 1, 8, 1, 3, 0, 1) \
    X(27,  4, 2, 1, 8, 1, 3, 0, 1) \
    X(28,  8, 2, 1, 2, 2, 3, 0, 1) \
    X(29,  8, 2, 1, 2, 4, 3, 0, 1) \
    X(30,  8, 2, 2, 4, 2, 3, 0, 1) \
    X(31,  8, 2, 2, 4, 4, 3, 0, 1) \
    X(32,  8, 4, 1, 2, 2, 3, 0, 1) \
    X(33,  8, 4, 1, 4, 4, 3, 0, 1) \
    X(34, 16, 2, 1, 2, 2, 3, 0, 1) \
    X(35, 16, 2, 1, 4, 4, 3, 0, 1) \
    X(36,  8, 1, 1, 2, 2, 3, 0, 1) \
    X(37,  8, 1, 2, 4, 2, 3, 0, 1) \
    X(38,  8, 2, 1, 8, 2, 3, 0, 1) \
    X(39,  8, 4, 2, 2, 4, 3, 0, 1) \
    X(40, 16, 2, 1, 2, 1, 3, 0, 1) \
    X(41, 16, 4, 1, 2, 1, 3, 0, 1) \
    X(42,  4, 4, 1, 2, 1, 3, 0, 1) \
    X(43,  8, 2, 1, 2, 1, 3, 0, 2) \
    X(44,  8, 2, 2, 4, 1, 3, 0, 2) \
    X(45,  8, 4, 1, 2, 2, 3, 0, 2) \
    X(46, 16, 2, 2, 4, 2, 3, 0, 2) \
    X(47,  8, 2, 1, 4, 1, 2, 0, 2) \
    X(48,  8, 2, 1, 2, 1, 3, 1, 1) \
    X(49,  8, 4, 1, 2, 1, 3, 1, 1) \
    X(50,  8, 2, 2, 4, 1, 3, 1, 1) \
    X(51,  8, 2, 4, 4, 1, 3, 1, 1) \
    X(52, 16, 2, 1, 4, 1, 3, 1, 1) \
    X(53,  8, 2, 1, 8, 1, 3, 1, 1)

#define Q2B_GEN2_COUNT_ONE(id, CW, R, ILP, W, KS, ST, PW, MB) +1
constexpr int Q2B_GEN2_NUM_VARIANTS = 0 Q2B_GEN2_VARIANTS(Q2B_GEN2_COUNT_ONE);
#undef Q2B_GEN2_COUNT_ONE

template <typename T, int CW, int R, int ILP, int W, int KS, int STAGES, int PRODW, int MINB>
int q2b_gen2_launch_variant(
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
    using G = q2b_g2_geom<CW, R, ILP, W, KS, STAGES, PRODW>;
    // V=1 layout + whole tiles per K-group: the only shape constraints (smem
    // is static and shape-independent).
    if (num_symbols != packed_per_row * 4) return -1;
    if (packed_per_row % (KS * G::TB) != 0) return -1;
    dim3 grid((n_rows + G::RPB - 1) / G::RPB, n_pairs, 1);
    qtip2b_gemv_gen2_kernel<T, CW, R, ILP, W, KS, STAGES, PRODW, MINB>
        <<<grid, G::THREADS, 0, stream>>>(
            d_packed, d_row_scales, d_x, d_indices, d_y,
            n_rows, packed_per_row, num_symbols, n_pairs, num_experts, mult);
    return 0;
}

template <typename T>
int q2b_gen2_dispatch(
    int local_variant,
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
    switch (local_variant) {
#define Q2B_GEN2_CASE(id, CW, R, ILP, W, KS, ST, PW, MB)                       \
        case id:                                                               \
            return q2b_gen2_launch_variant<T, CW, R, ILP, W, KS, ST, PW, MB>(  \
                d_packed, d_row_scales, d_x, d_indices, d_y, n_rows,           \
                packed_per_row, num_symbols, n_pairs, num_experts, mult,       \
                stream);
        Q2B_GEN2_VARIANTS(Q2B_GEN2_CASE)
#undef Q2B_GEN2_CASE
        default:
            return -1;
    }
}

} // anonymous namespace

// ============================================================================
// Internal surface consumed by qtip_bitshift_tune.cu (the single dispatch
// entry). Not part of the Rust FFI.
// ============================================================================

extern "C" {

int qtip2b_gemv_gen2_num_variants(void) {
    return Q2B_GEN2_NUM_VARIANTS;
}

int qtip2b_gemv_gen2_variant_info(
    int idx,
    int* consumer_warps,
    int* rows_per_warp,
    int* ilp,
    int* wbytes,
    int* ksplit,
    int* stages,
    int* producer_warps,
    int* min_blocks
) {
    switch (idx) {
#define Q2B_GEN2_INFO(id, CW, R, ILP, W, KS, ST, PW, MB)                       \
        case id:                                                               \
            *consumer_warps = CW;                                              \
            *rows_per_warp = R;                                                \
            *ilp = ILP;                                                        \
            *wbytes = W;                                                       \
            *ksplit = KS;                                                      \
            *stages = ST;                                                      \
            *producer_warps = PW;                                              \
            *min_blocks = MB;                                                  \
            return 0;
        Q2B_GEN2_VARIANTS(Q2B_GEN2_INFO)
#undef Q2B_GEN2_INFO
        default:
            return -1;
    }
}

#define Q2B_GEN2_LAUNCHER(NAME, T)                                             \
    int NAME(int local_variant,                                                \
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
        return q2b_gen2_dispatch<T>(                                           \
            local_variant, d_packed, d_row_scales, d_x_rotated, d_indices,     \
            d_y, n_rows, packed_per_row, num_symbols, n_pairs, num_experts,    \
            mult, stream);                                                     \
    }

Q2B_GEN2_LAUNCHER(qtip2b_gemv_gen2_launch_bf16, __nv_bfloat16)
Q2B_GEN2_LAUNCHER(qtip2b_gemv_gen2_launch_f16,  __half)
Q2B_GEN2_LAUNCHER(qtip2b_gemv_gen2_launch_f32,  float)

#undef Q2B_GEN2_LAUNCHER

} // extern "C"
