// Parent system: ArcQuant / ArcKernels
//
// TCFRAG-2B — the tensor-core trellis GEMV for the K=2 / V=1 / L=16 (`qtip2b`)
// rung.
//
// ============================================================================
// ⚠️  UNVERIFIED ON HARDWARE — NEVER RUN.
//
// No part of this file has executed on a GPU. It has been compiled (nvcc
// sm_80 + sm_90, the repo's free no-GPU CI gate) and nothing more. Every
// performance figure in these comments is a DERIVATION FROM INSTRUCTION
// COUNTS, not a measurement, and is labelled as such at each site. Do not
// quote any of it as a result.
// ============================================================================
//
// WHY THIS EXISTS
// ---------------
// `qtip2b_gemv_tuned_kernel` (qtip_bitshift_tune.cu) is the hot trellis path:
// 69 ms of a 794 ms B=256 step, and the whole of the b=1 decode path, which
// `gather_policy::lut_fused_gather_preferred` routes to unconditionally for
// n_tokens <= 8.
//
// It is not inefficient. It is running flat out on the WRONG ROOFLINE. With
// its decode stubbed to a 2-bit symbol read it reaches ~90% of the H200
// *scalar* FMA roofline (60.5 TFLOP/s) — while the dense fp16 tensor cores on
// the same die do 989 TFLOP/s, 16.3x beyond. The kernel contains exactly TWO
// tensor-core instructions, despite comments describing "a persistent
// tensor-core tile loop". Measured consequence: even a FREE decode buys at
// most 1.89x on the kernel as structured.
//
// THE IDENTITY
// ------------
// The qtip2b codeword is, by definition (`bitshift.rs::mcg_codeword`,
// `qtip2b_common.cuh::q2b_decode`):
//
//     x = state * mult                        // IMAD
//     m = (x & 0x8FFF8FFF) ^ 0x3B603B60       // LOP3
//     w = f32(fp16(m >> 16)) + f32(fp16(m & 0xFFFF))
//
// `m` is ALREADY a packed pair of fp16 numbers whose sum is the weight. A dot
// product is linear, so
//
//     sum_s (hi_s + lo_s) * x_s == sum_s hi_s * x_s + sum_s lo_s * x_s
//
// Give the two halves of `m` adjacent k-slots of an `mma.m16n8k16` A fragment
// and feed both slots the SAME activation, and the tensor core does the
// fp16->f32 fold, the k-reduction and the accumulate. The row scale leaves the
// inner loop entirely and becomes one multiply in the epilogue.
//
// **The LOP3 destination register IS the mma operand register.** Nothing goes
// between them — no MOV, no PRMT, no SHFL, and no lane ever needs a value
// another lane produced.
//
// THE BYTE ORDER (see src/qtip/tcfrag2b.rs for the spec and its property tests)
// ----------------------------------------------------------------------------
// Word j of a row holds symbols 16j..16j+15 in REVERSED 2-bit-field order, so
// (R[j-1] << 32) | R[j] is a descending-symbol stream and
//
//     state(s) = __funnelshift_r(R[j], R[j-1], 2 * (15 - (s & 15))) & 0xFFFF
//
// — one SHF, one LOP3, at any s, with no warm-up replay and no __brev
// reversal. Words are then interleaved tile-major / k-major / row-minor so a
// warp's 16 distinct row reads at one j are 16 CONSECUTIVE u32 = one 64-byte
// transaction instead of 16 scattered ones.
//
// 🔴 D22 — THIS IS A LOAD-TIME PERMUTATION, NEVER A BAKED ONE
// -----------------------------------------------------------
// The permutation is applied by `qtip2b_tcfrag_repack_kernel` after the
// artifact is read, and the on-disk bytes are UNCHANGED. The k-slot assignment
// is specific to `mma.m16n8k16`; Blackwell's tcgen05 shapes differ, and baking
// it would mean one artifact per GPU generation, fragmenting the byte-format
// moat (D17) that is the whole reason this GEMM is ours. Cost of doing it at
// load time: one bandwidth-bound pass, ~74 GB read + 74 GB write, ~0.03 s
// against a ~65 s model load.
//
// Therefore **the layout is DATA, selected by compute capability on the host**
// and carried across the ABI as `layout_id` (see `Tcfrag2bLayout` on the Rust
// side). It is never chosen by `#if __CUDA_ARCH__`. The only `__CUDA_ARCH__`
// guard in this file is around the `mma` INSTRUCTION's availability (sm_80+),
// which is a different thing: it gates whether an opcode exists, not which
// byte order the artifact has.
//
// DERIVED COST (arithmetic, not measurement)
// ------------------------------------------
// Per weight, TCFRAG-2B issues 4 decode instructions — SHF (funnel), LOP3
// (mask to 16 bits), IMAD (MCG), LOP3 (mask/xor) — and the result is the mma
// operand. At K=2/V=1 those four are paid PER WEIGHT, where the K=4/V=2
// sibling amortises them over two (it measured 3.56 inst/weight on an H200).
// Adding the per-mma overhead and the B fragment, at M_TILES = 2:
//
//     (2 * (4*4 + 2) + 6) / (4 * 2) = 5.25 inst/weight
//
// against ~12.25 counted from the shipped tuned kernel's inner loop
// => **derived 2.33x fewer instructions**, and, on the measured premise that
// the kernel is issue-bound (1% of roofline, memory controller 4% utilised),
// a derived 69 ms -> 29.6 ms.
//
// What that does NOT claim: the remaining distance to the tensor-core
// roofline. This kernel issues ~18 instructions per HMMA.16816 where dense
// fp16 peak needs one every ~3.84. The rest of the 16.3x is the trellis decode
// itself, which no tensor core can absorb, plus the n-dimension being 1..8 of
// 8 at decode. Naming it is the point.
//
// SM80+. Gated by `has_qtip_kernels` in build.rs.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "qtip2b_common.cuh"

namespace {

// ---------------------------------------------------------------------------
// Format constants. These MUST agree with src/qtip/tcfrag2b.rs, which owns the
// spec and property-tests it; `mma_step_geometry_is_what_the_kernel_assumes`
// pins them there.
// ---------------------------------------------------------------------------
constexpr int TCF_SYMS_PER_WORD = 32 / (int)Q2B_K;   // 16
constexpr int TCF_SYMS_PER_MMA  = 8;                 // 16 k-slots / 2 per symbol
constexpr int TCF_MMA_M         = 16;
constexpr int TCF_MMA_N         = 8;
constexpr int TCF_TILE_ROWS     = TCF_MMA_M;
constexpr int TCF_STEPS_PER_WORD = TCF_SYMS_PER_WORD / TCF_SYMS_PER_MMA;  // 2

// D22 ABI: the layout id the host computed from the device's compute
// capability. Today one order exists; a future MMA shape appends an id here
// and its own `tcf_*` specialisation. Never a `#if __CUDA_ARCH__`.
constexpr int TCF_LAYOUT_MMA_M16N8K16 = 0;

// Largest number of distinct experts one 8-pair group can contain; also
// `gridDim.z`. One block per (row tile group, pair group, expert pass).
constexpr int TCF_MAX_EXPERT_PASSES = TCF_MMA_N;

// ---------------------------------------------------------------------------
// The whole decode. Four instructions, and the result IS the A operand.
//
//   SHF   funnel R[j]:R[j-1] right by 2*(15 - (s & 15))
//   LOP3  & 0xFFFF                      -> the 16-bit trellis state
//   IMAD  * mult                        -> the MCG product
//   LOP3  (& 0x8FFF8FFF) ^ 0x3B603B60   -> a packed half2 whose SUM is w
//
// The `& 0xFFFF` cannot be folded away: the product is taken mod 2^32, so a
// dirty high half of the shifted word corrupts bits >= 16 of the result, which
// is exactly the half the codeword needs.
// ---------------------------------------------------------------------------
template <int LAYOUT>
__device__ __forceinline__ uint32_t tcf_a_operand(
    uint32_t w_cur, uint32_t w_prev, int sym_in_word, uint32_t mult
);

template <>
__device__ __forceinline__ uint32_t tcf_a_operand<TCF_LAYOUT_MMA_M16N8K16>(
    uint32_t w_cur, uint32_t w_prev, int sym_in_word, uint32_t mult
) {
    const uint32_t sh = (uint32_t)((int)Q2B_K * (TCF_SYMS_PER_WORD - 1 - sym_in_word));
    const uint32_t state = __funnelshift_r(w_cur, w_prev, sh) & Q2B_STATE_MASK;
    return ((state * mult) & Q2B_MASK) ^ Q2B_XOR;
}

// ---------------------------------------------------------------------------
// mma.m16n8k16.row.col.f32.f16.f16.f32, accumulating in place.
//
// Fragment mapping (PTX ISA 9.3 sec. 9.7.15.5.8, verified against the live
// spec — `groupID = laneid >> 2`, `threadID_in_group = laneid % 4`):
//
//   A (16x16, 4 regs): reg0 = (row gid  , k 2i  , 2i+1)
//                      reg1 = (row gid+8, k 2i  , 2i+1)
//                      reg2 = (row gid  , k 2i+8, 2i+9)
//                      reg3 = (row gid+8, k 2i+8, 2i+9)
//   B (16x8,  2 regs): reg0 = (k 2i  , 2i+1 ; col gid)
//                      reg1 = (k 2i+8, 2i+9 ; col gid)
//   C (16x8,  4 f32) : c0,c1 = (row gid  , col 2i, 2i+1)
//                      c2,c3 = (row gid+8, col 2i, 2i+1)
//
// The `__CUDA_ARCH__` guard below is about the OPCODE existing (mma with .f16
// and shape m16n8k16 requires sm_80), not about the byte order — see the D22
// note in the file header. `has_qtip_kernels` already excludes < sm_80, so the
// fallback body is unreachable; it exists so the TU still compiles.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tcf_mma_m16n8k16(
    float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]
) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    // Explicit D and C operand lists (the CUTLASS form) rather than "+f" —
    // the accumulator is read AND written, and spelling both halves out leaves
    // ptxas no room to interpret it differently.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    (void)c; (void)a; (void)b;
    __trap();
#endif
}

// Activation -> the fp16 half-pattern the B fragment carries, duplicated into
// both k-slots of a pair so the tensor core computes hi*x + lo*x == w*x.
//
// Range: fp16 overflows at 65504 where bf16 does not. This conversion is
// deliberately NOT clamped — a clamp would trade a visible infinity for an
// invisible wrong answer. The remedy is the ARC_QTIP_TCFRAG=0 kill switch.
// See the range note in src/qtip/tcfrag2b.rs. UNMEASURED on the real model.
__device__ __forceinline__ uint32_t tcf_b_slot(__nv_bfloat16 v) {
    const unsigned short h = __half_as_ushort(__float2half_rn(__bfloat162float(v)));
    return ((uint32_t)h << 16) | (uint32_t)h;
}
__device__ __forceinline__ uint32_t tcf_b_slot(__half v) {
    const unsigned short h = __half_as_ushort(v);
    return ((uint32_t)h << 16) | (uint32_t)h;
}
__device__ __forceinline__ uint32_t tcf_b_slot(float v) {
    const unsigned short h = __half_as_ushort(__float2half_rn(v));
    return ((uint32_t)h << 16) | (uint32_t)h;
}

// ---------------------------------------------------------------------------
// 1) The load-time repack (D22). Shipped LSB-first bytes -> TCFRAG-2B words.
//
// Per word: one aligned 32-bit read, `__brev` (which reverses all 32 bits and
// therefore swaps the two bits WITHIN each symbol as well), then one
// adjacent-bit swap to put them back. Three ALU ops; the kernel is purely
// bandwidth-bound.
//
// `rows_padded` covers the last partial A-fragment tile, whose words are
// written as zero. Those rows land in their own accumulator lanes and the
// epilogue never writes them.
// ---------------------------------------------------------------------------
template <int LAYOUT>
__device__ __forceinline__ uint32_t tcf_pack_word(uint32_t src);

template <>
__device__ __forceinline__ uint32_t tcf_pack_word<TCF_LAYOUT_MMA_M16N8K16>(uint32_t src) {
    const uint32_t r = __brev(src);
    return ((r & 0x55555555u) << 1) | ((r >> 1) & 0x55555555u);
}

template <int LAYOUT>
__global__ void qtip2b_tcfrag_repack_kernel(
    const uint8_t* __restrict__ packed,   // [E, n_rows, packed_per_row]
    uint32_t*      __restrict__ words,    // [E, tile_count, words_per_row, 16]
    long n_experts,
    int  n_rows,
    int  packed_per_row,
    int  words_per_row,
    int  tile_count
) {
    const long per_expert = (long)tile_count * words_per_row * TCF_TILE_ROWS;
    const long total = per_expert * n_experts;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        const long e   = idx / per_expert;
        const long rem = idx - e * per_expert;
        const int  rit = (int)(rem % TCF_TILE_ROWS);
        const long q   = rem / TCF_TILE_ROWS;
        const int  j    = (int)(q % words_per_row);
        const int  tile = (int)(q / words_per_row);

        const int row = tile * TCF_TILE_ROWS + rit;
        uint32_t out = 0u;
        if (row < n_rows) {
            const uint8_t* p = packed
                + ((size_t)e * (size_t)n_rows + (size_t)row) * (size_t)packed_per_row
                + (size_t)j * 4u;
            out = tcf_pack_word<LAYOUT>(__ldg(reinterpret_cast<const uint32_t*>(p)));
        }
        words[idx] = out;
    }
}

// ---------------------------------------------------------------------------
// 2) The GEMV.
//
// Grid : ( ceil(n_rows / (M_TILES*16)), ceil(n_pairs / 8), TCF_MAX_EXPERT_PASSES )
// Block: WARPS * 32
//
// * blockIdx.x  -> M_TILES consecutive A-fragment tiles (M_TILES*16 rows).
// * blockIdx.y  -> a group of up to 8 (token, slot) pairs = the mma's n
//                  dimension. THIS is the capacity property: as long as the
//                  pairs share an expert, b=8 costs what b=1 costs, because
//                  the decoded A fragment is reused across all 8 columns.
// * blockIdx.z  -> the q-th DISTINCT expert id in that pair group. MoE decode
//                  at top_k=6 puts six different experts in one group; giving
//                  each its own block keeps the GPU fed (a loop would leave
//                  5/6 of the machine idle) and the passes write DISJOINT
//                  columns, so no accumulation can race. Blocks with no q-th
//                  distinct expert exit in a few instructions.
// * warps       -> split the k range (intra-block split-K), reduced through
//                  shared memory in the epilogue. A fixed reduction order, so
//                  the result is deterministic run to run — unlike an
//                  atomicAdd split-K.
//
// The expert ids are read ON DEVICE (`__ldg(indices + pair)`), so this stays
// CUDA-graph capturable, same contract as the kernel it replaces.
// ---------------------------------------------------------------------------
template <typename T, int WARPS, int M_TILES, int LAYOUT>
__global__ void __launch_bounds__(WARPS * 32)
qtip2b_tcfrag_gemv_kernel(
    const uint32_t* __restrict__ words,       // TCFRAG-2B, tiled
    const float*    __restrict__ row_scales,  // [E, n_rows]
    const T*        __restrict__ x,           // [n_pairs, num_symbols] (rotated)
    const uint32_t* __restrict__ indices,     // [n_pairs] or nullptr (2-D)
    T*              __restrict__ y,           // [n_pairs, n_rows]
    int n_rows,
    int num_symbols,
    int n_pairs,
    int num_experts,
    uint32_t mult,
    int words_per_row,
    int tile_count
) {
    constexpr int ACC_PER_TILE = 4;                    // the C fragment
    constexpr int RED_STRIDE   = M_TILES * ACC_PER_TILE + 1;  // +1 kills bank conflicts
    __shared__ float s_red[WARPS][32][RED_STRIDE];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gid  = lane >> 2;        // groupID
    const int tig  = lane & 3;         // threadID_in_group

    const int tile0 = (int)blockIdx.x * M_TILES;
    const int p0    = (int)blockIdx.y * TCF_MMA_N;
    const int pass  = (int)blockIdx.z;
    if (tile0 >= tile_count || p0 >= n_pairs) return;

    // ---- which expert does this pass own, and which columns are its own? ---
    uint32_t eid[TCF_MMA_N];
    bool     live[TCF_MMA_N];
    #pragma unroll
    for (int c = 0; c < TCF_MMA_N; ++c) {
        const int p = p0 + c;
        live[c] = (p < n_pairs);
        eid[c]  = live[c] ? ((indices != nullptr) ? __ldg(indices + p) : 0u) : 0xFFFFFFFFu;
    }
    int      my_first = -1;
    int      seen     = 0;
    #pragma unroll
    for (int c = 0; c < TCF_MMA_N; ++c) {
        if (!live[c] || my_first >= 0) continue;
        bool first = true;
        #pragma unroll
        for (int d = 0; d < TCF_MMA_N; ++d) {
            if (d < c && live[d] && eid[d] == eid[c]) first = false;
        }
        if (!first) continue;
        if (seen == pass) my_first = c;
        else ++seen;
    }
    if (my_first < 0) return;          // no `pass`-th distinct expert here

    const uint32_t expert = eid[my_first];
    uint32_t colmask = 0u;
    #pragma unroll
    for (int c = 0; c < TCF_MMA_N; ++c) {
        if (live[c] && eid[c] == expert) colmask |= (1u << c);
    }
    const bool invalid = expert >= (uint32_t)num_experts;

    // Out-of-range expert id from a broken router: write zeros for this pass's
    // columns, never read out of bounds. Same contract as the shipped kernel.
    if (invalid) {
        if (threadIdx.x < 32) {
            for (int c = 0; c < TCF_MMA_N; ++c) {
                if (!(colmask & (1u << c))) continue;
                const int p = p0 + c;
                for (int m = 0; m < M_TILES; ++m) {
                    for (int r = lane; r < TCF_TILE_ROWS; r += 32) {
                        const int row = (tile0 + m) * TCF_TILE_ROWS + r;
                        if (row < n_rows) y[(size_t)p * n_rows + row] = q2b_from_f32<T>(0.0f);
                    }
                }
            }
        }
        return;
    }

    // ---- k range for this warp (intra-block split-K over whole words) ------
    const int per_warp = (words_per_row + WARPS - 1) / WARPS;
    const int j_begin  = warp * per_warp;
    const int j_end    = min(j_begin + per_warp, words_per_row);

    float acc[M_TILES][ACC_PER_TILE];
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        #pragma unroll
        for (int t = 0; t < ACC_PER_TILE; ++t) acc[m][t] = 0.0f;
    }

    if (j_begin < j_end) {
        // Base of this expert's words, and of each of our M_TILES tiles.
        const uint32_t* __restrict__ ebase =
            words + (size_t)expert * (size_t)tile_count * (size_t)words_per_row * TCF_TILE_ROWS;
        const uint32_t* __restrict__ tbase[M_TILES];
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            const int tile = min(tile0 + m, tile_count - 1);
            tbase[m] = ebase + (size_t)tile * (size_t)words_per_row * TCF_TILE_ROWS;
        }
        // This lane's two rows within a tile: `gid` (A regs 0,2) and `gid+8`
        // (A regs 1,3). One `LDG` over the warp touches 16 consecutive u32.
        const int r_lo = gid;
        const int r_hi = gid + 8;

        // Rolling word pair per tile per row. R[-1] = 0 by definition; at a
        // chunk boundary the predecessor word is a real load.
        uint32_t w_prev[M_TILES][2];
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            if (j_begin == 0) {
                w_prev[m][0] = 0u;
                w_prev[m][1] = 0u;
            } else {
                const uint32_t* p = tbase[m] + (size_t)(j_begin - 1) * TCF_TILE_ROWS;
                w_prev[m][0] = __ldg(p + r_lo);
                w_prev[m][1] = __ldg(p + r_hi);
            }
        }

        // Clamp the column base: a lane whose column is past `n_pairs` has
        // `col_live == false` and never dereferences this, but forming a
        // pointer past the allocation is avoidable, so avoid it.
        const bool col_live = (colmask & (1u << gid)) != 0u;
        const T* __restrict__ x_col =
            x + (size_t)min(p0 + gid, n_pairs - 1) * (size_t)num_symbols;

        for (int j = j_begin; j < j_end; ++j) {
            uint32_t w_cur[M_TILES][2];
            #pragma unroll
            for (int m = 0; m < M_TILES; ++m) {
                const uint32_t* p = tbase[m] + (size_t)j * TCF_TILE_ROWS;
                w_cur[m][0] = __ldg(p + r_lo);
                w_cur[m][1] = __ldg(p + r_hi);
            }

            #pragma unroll
            for (int step = 0; step < TCF_STEPS_PER_WORD; ++step) {
                // Symbols of this mma step, within the word.
                //   A regs 0,1 <- k-slots 2*tig  , 2*tig+1   -> symbol s_lo
                //   A regs 2,3 <- k-slots 2*tig+8, 2*tig+9   -> symbol s_hi
                // and 2*(tig+4) == 2*tig+8, so both obey the single rule
                // "symbol p -> k-slots 2p, 2p+1" that lets one activation be
                // duplicated across a pair.
                const int s_lo = step * TCF_SYMS_PER_MMA + tig;
                const int s_hi = s_lo + 4;

                uint32_t b[2];
                if (col_live) {
                    const int gs = j * TCF_SYMS_PER_WORD;
                    b[0] = tcf_b_slot(__ldg(x_col + gs + s_lo));
                    b[1] = tcf_b_slot(__ldg(x_col + gs + s_hi));
                } else {
                    // A column this pass does not own contributes nothing; its
                    // own pass writes it.
                    b[0] = 0u;
                    b[1] = 0u;
                }

                #pragma unroll
                for (int m = 0; m < M_TILES; ++m) {
                    uint32_t a[4];
                    a[0] = tcf_a_operand<LAYOUT>(w_cur[m][0], w_prev[m][0], s_lo, mult);
                    a[1] = tcf_a_operand<LAYOUT>(w_cur[m][1], w_prev[m][1], s_lo, mult);
                    a[2] = tcf_a_operand<LAYOUT>(w_cur[m][0], w_prev[m][0], s_hi, mult);
                    a[3] = tcf_a_operand<LAYOUT>(w_cur[m][1], w_prev[m][1], s_hi, mult);
                    tcf_mma_m16n8k16(acc[m], a, b);
                }
            }

            #pragma unroll
            for (int m = 0; m < M_TILES; ++m) {
                w_prev[m][0] = w_cur[m][0];
                w_prev[m][1] = w_cur[m][1];
            }
        }
    }

    // ---- reduce the split-K partials across warps (fixed order) -----------
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        #pragma unroll
        for (int t = 0; t < ACC_PER_TILE; ++t) {
            s_red[warp][lane][m * ACC_PER_TILE + t] = acc[m][t];
        }
    }
    __syncthreads();
    if (warp != 0) return;

    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        #pragma unroll
        for (int t = 0; t < ACC_PER_TILE; ++t) {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) v += s_red[w][lane][m * ACC_PER_TILE + t];
            acc[m][t] = v;
        }
    }

    // ---- epilogue: row scale, then narrow and store -----------------------
    //
    // C fragment: c0,c1 = (row gid, col 2*tig, 2*tig+1)
    //             c2,c3 = (row gid+8, col 2*tig, 2*tig+1)
    // The per-row scale, which the shipped kernel pays once PER WEIGHT, is one
    // multiply here.
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        const int tile = tile0 + m;
        if (tile >= tile_count) continue;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = tile * TCF_TILE_ROWS + gid + half * 8;
            if (row >= n_rows) continue;
            const float scale = __ldg(row_scales + (size_t)expert * n_rows + row);
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                const int col = 2 * tig + cc;
                if (!(colmask & (1u << col))) continue;
                const int p = p0 + col;
                if (p >= n_pairs) continue;
                const float v = acc[m][half * 2 + cc] * scale;
                y[(size_t)p * n_rows + row] = q2b_from_f32<T>(v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shipped configuration. ONE live target, not a variant table: the axes that
// matter here (M_TILES trades B-fragment amortisation against block count)
// cannot be swept without hardware, and a table nobody measured is a table
// nobody should trust.
//
// M_TILES = 2, WARPS = 8:
//   * block covers 32 rows, so n_rows = 2048 gives 64 row-tile blocks; times
//     the live expert passes (6 at top_k=6) that is ~384 blocks of 8 warps —
//     roughly 23 warps per SM on a 132-SM H200. A single-tile-per-warp design
//     without the intra-block split-K would have produced 16 blocks and left
//     7/8 of the machine idle. (DERIVED from shapes; not measured.)
//   * derived 5.25 inst/weight (see the header).
// ---------------------------------------------------------------------------
constexpr int TCF_WARPS   = 8;
constexpr int TCF_M_TILES = 2;

// Applicability. A shape that fails any of these keeps the shipped kernel —
// the launcher returns -1 and the Rust dispatch falls back, exactly like the
// tuned launcher's contract.
inline bool tcf_shape_ok(int packed_per_row, int num_symbols) {
    if (num_symbols <= 0 || packed_per_row <= 0) return false;
    if (num_symbols != packed_per_row * 4) return false;       // V=1, 4 syms/byte
    if (num_symbols % TCF_SYMS_PER_WORD != 0) return false;    // whole words only
    return true;
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers.
//
// Every symbol here is NEW. A stale object file from before this change is a
// LINK error, not a silently-old kernel — which is the failure mode the
// `EXPECTED_KERNEL_COUNT` / header-hash gates in build.rs exist to make loud.
// ============================================================================

extern "C" {

// Words one expert occupies in the tiled TCFRAG-2B form. The Rust side sizes
// its buffer from THIS, so the two cannot drift.
long qtip2b_tcfrag_words_per_expert(int n_rows, int num_symbols) {
    if (!tcf_shape_ok(num_symbols / 4, num_symbols)) return -1;
    const int wpr   = num_symbols / TCF_SYMS_PER_WORD;
    const int tiles = (n_rows + TCF_TILE_ROWS - 1) / TCF_TILE_ROWS;
    return (long)tiles * (long)wpr * (long)TCF_TILE_ROWS;
}

// The load-time permutation (D22). `layout_id` is DATA the host derived from
// the device's compute capability; an unknown id is refused, never guessed.
int launch_qtip2b_tcfrag_repack(
    const uint8_t* d_packed,
    uint32_t*      d_words,
    int n_experts,
    int n_rows,
    int packed_per_row,
    int num_symbols,
    int layout_id,
    cudaStream_t stream
) {
    if (!tcf_shape_ok(packed_per_row, num_symbols)) return -1;
    if (layout_id != TCF_LAYOUT_MMA_M16N8K16) return -1;
    if (n_experts <= 0 || n_rows <= 0) return -1;

    const int wpr   = num_symbols / TCF_SYMS_PER_WORD;
    const int tiles = (n_rows + TCF_TILE_ROWS - 1) / TCF_TILE_ROWS;
    const long total = (long)n_experts * tiles * wpr * TCF_TILE_ROWS;

    constexpr int THREADS = 256;
    long blocks = (total + THREADS - 1) / THREADS;
    if (blocks > 65535) blocks = 65535;      // grid-stride loop covers the rest
    qtip2b_tcfrag_repack_kernel<TCF_LAYOUT_MMA_M16N8K16>
        <<<(int)blocks, THREADS, 0, stream>>>(
            d_packed, d_words, (long)n_experts, n_rows, packed_per_row, wpr, tiles);
    return 0;
}

#define Q2B_TCFRAG_GEMV_LAUNCHER(NAME, T)                                      \
    int NAME(const uint32_t* d_words,                                          \
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
             int layout_id,                                                    \
             cudaStream_t stream) {                                            \
        if (!tcf_shape_ok(packed_per_row, num_symbols)) return -1;             \
        if (layout_id != TCF_LAYOUT_MMA_M16N8K16) return -1;                   \
        if (n_pairs <= 0 || n_rows <= 0 || num_experts <= 0) return -1;        \
        const int wpr   = num_symbols / TCF_SYMS_PER_WORD;                     \
        const int tiles = (n_rows + TCF_TILE_ROWS - 1) / TCF_TILE_ROWS;        \
        const int rows_per_block = TCF_M_TILES * TCF_TILE_ROWS;                \
        dim3 grid((n_rows + rows_per_block - 1) / rows_per_block,              \
                  (n_pairs + TCF_MMA_N - 1) / TCF_MMA_N,                       \
                  TCF_MAX_EXPERT_PASSES);                                      \
        qtip2b_tcfrag_gemv_kernel<T, TCF_WARPS, TCF_M_TILES,                   \
                                  TCF_LAYOUT_MMA_M16N8K16>                     \
            <<<grid, TCF_WARPS * 32, 0, stream>>>(                             \
                d_words, d_row_scales, d_x_rotated, d_indices, d_y,            \
                n_rows, num_symbols, n_pairs, num_experts, mult, wpr, tiles);  \
        return 0;                                                              \
    }

Q2B_TCFRAG_GEMV_LAUNCHER(launch_qtip2b_tcfrag_gemv_bf16, __nv_bfloat16)
Q2B_TCFRAG_GEMV_LAUNCHER(launch_qtip2b_tcfrag_gemv_f16,  __half)
Q2B_TCFRAG_GEMV_LAUNCHER(launch_qtip2b_tcfrag_gemv_f32,  float)

#undef Q2B_TCFRAG_GEMV_LAUNCHER

} // extern "C"
