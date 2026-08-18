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
// there is ~4x more tensor-core compute budget per HBM byte than BF16. The
// per-token gather GEMV path (qtip_bitshift.cu) re-reads every routed
// expert's packed bytes once per token; this kernel reads each expert's
// bytes once per TILE of tokens.
//
// 🔴 CORRECTION (2026-08-18): this header used to end that paragraph with
// "the decode ALU hides under mma issue". IT DOES NOT, and the error is a
// factor of ~13. From the measured 8.2 us per m-tile unit (FACTS wave38) at
// 16*N*K = 137.4 MMAC per unit, the tensor pipe runs at 33.5 TFLOP/s = 6.8%
// of H200 dense BF16 — it is idle 93% of the time — while the same 8.2 us
// covers N*K = 8.39M trellis decodes at ~15 ops each = 1.5e13 op/s against
// the SM's 1.67e13 INT32 lane/s. THE KERNEL IS ALU-ISSUE-BOUND ON THE DECODE
// and is already near that roofline. Optimise ops per decoded weight; do not
// optimise bytes moved, and do not expect a better MMA instruction to help
// (a perfect wgmma recovers <10% against a 76% deficit). This reproduces the
// gen-2 GEMV sweep's independent verdict (FACTS: 98/98 variants).
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
//   * ⚠️ TILE_M IS A NON-LEVER IN THE REGIME WE SERVE, despite what this note
//     used to claim. Decode cost is per (n, k) and independent of m, but the
//     number of m-tiles is `sum_e ceil(count_e / TILE_M)`, and at E=256 top-6
//     the average pairs per WOKEN expert is 3.1 even at B=128 (the harness's
//     own `assert_config_is_measurable` refuses anything past TILE_M). Every
//     expert already takes exactly one m-tile, so raising TILE_M changes no
//     decode work at all — it only pads the mma, which is 7% of the time.
//     TILE_N 64 -> 128 and warps 4 -> 8 remain unswept and are still live.
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
//   * s_wp bank conflicts: FIXED in variant 1 by a 48 B stride (this note
//     said "up to 4-way"; the real figure is 2-way — bank = (8g + wi) % 32
//     puts g=0..7 on 4 banks with 2 distinct addresses each). The bigger one
//     it never mentioned is the 8-WAY conflict on the A fragments, also
//     fixed in variant 1. See the VARIANTS block.
//   * Epilogue: pack paired bf16 writes into one 32-bit store when both
//     columns are in range.
//
// -------------------------------------------------------------------------
// VARIANTS (runtime-selected, both compiled in — ARC_QTIP_GROUPED_VARIANT)
// -------------------------------------------------------------------------
// `variant=0` (baseline) is the kernel exactly as first shipped and measured
// (FACTS wave38: 8.2 us per m-tile unit vs the GEMV's 4.4 — a 1.76x per-byte
// deficit). `variant=1` (tuned) is bit-identical in output and differs only
// in how the same weights are reached:
//
// ✅ MEASURED 2026-08-18, H200, E=256 top-6, B=128, U F U N x3 interleaved
// against a null control, both variants compiled in and the arm proved from
// the RUNTIME launch counters (not the build):
//     gate/up 2048x4096   8.16 -> 4.95 us/m-tile   +39.37%  (null 0.09%)
//     down    4096x2048   8.10 -> 4.91 us/m-tile   +39.40%  (null 0.18%)
// Output bit-identical to the baseline over 6.29 M / 12.58 M bytes. The two
// shapes are transposes with identical weight counts and identical decode
// work but different access patterns everywhere else, so agreement to 0.03pp
// is itself evidence the win is ops-per-decoded-weight and not tiling or
// bandwidth. Handicap vs the GEMV: 1.76x -> 1.12x.
//
// `variant=2` adds one `ldmatrix.x4` in place of the four `LDS.32` A-fragment
// loads (and an explicit packed convert that turns out to be a no-op, above).
// MEASURED the same way: **+3.27% / +3.25% over variant 1**, +41.6% / +41.3%
// cumulative over the baseline, bit-identical throughout.
//
// ✅ THE COST MODEL IS NOW CONFIRMED IN BOTH DIRECTIONS by per-thread SASS
// instruction counts (total 896 -> 696 -> 672 for v0/v1/v2):
//   v1->v2: instructions -3.4%, measured time -3.3% — with the bank conflicts
//           gone, time tracks instruction count nearly 1:1, so the kernel is
//           genuinely instruction-issue-bound.
//   v0->v1: instructions -22.3%, measured time -39.4%. The EXCESS over the
//           instruction cut is the bank conflicts, which burn cycles without
//           burning instructions — an independent confirmation of the Phase 1
//           mechanism. (BREV 32 -> 5 confirms the reversal amortization.)
//
// ⏭️ NEXT LEVER, with evidence rather than a guess: IMAD = 115 per thread
// iteration in variant 2, but only ~32 are the codeword's `state * mult`. The
// other ~83 — 12% of all remaining instructions — are shared-memory ADDRESS
// ARITHMETIC recomputed inside the kf/f loops. Hoisting the row base pointers
// out and walking them with immediate offsets is the largest identified item
// left, bigger than anything variant 2 touched.
//
//   1. s_x row stride 64 -> QG_X_STRIDE elements (128 -> 144 B). At a 128 B
//      stride the bank index (byte/4)%32 = (g*32 + ...)%32 is INDEPENDENT of
//      the mma row g = lane>>2, so all eight g-values collide: every A-fragment
//      load was an 8-WAY bank conflict. 144 B makes it (4g + tig + kb/2)%32,
//      which spans all 32 banks exactly once. 144 % 16 == 0, so every cp.async
//      destination stays 16 B aligned.
//   2. s_wp row stride 32 -> 48 B. At 32 B (8 words) the bank is (8g + wi)%32,
//      which takes only 4 values across g=0..7 => a 2-way conflict on every
//      window load. 48 B gives (12g + wi)%32 = 8 distinct banks. 48 keeps the
//      +12 (4 B) and +16 (16 B) cp.async offsets aligned.
//   3. The packed row is PAIR-REVERSED once per row per k-chunk, hoisting the
//      __brev + 4-op pair-swap + mask out of the per-weight decode (~7 of ~15
//      ops) and amortizing them 64:1. Reversing the sixteen 2-bit groups of a
//      word turns the sliding state window into a plain right shift:
//      R word j = pair_reverse_32(staged word 7-j) for j=0..4, after which
//          state(ts) = (R >> (126 - 2*ts)) & 0xFFFF
//      directly — no reversal at read time. This is a pure bit permutation,
//      so the decoded weights are bit-identical (verified host-side against
//      the sequential recurrence over 40,960 states before any GPU time).
//      Rows are warp-private (row r is read only by warp r/16), so the
//      transform needs __syncwarp(), NOT an extra block-wide barrier.
//   4. The four windows a lane needs per (kf, f) are provably all reachable
//      from TWO words: base = 126 - 2*kb - 4*tig has base%32 in {18,22,26,30}
//      for every kb in {0,16,32,48} and tig in 0..3, and the four shifts
//      base-{0,2,16,18} all land in word base>>5 with the 16-bit field ending
//      no later than word (base>>5)+1. So one pair of loads + four
//      __funnelshift_r replaces four independent loads, each of which carried
//      an ALWAYS-TRUE `sh > 16` second-load branch (sh is in {18,22,26,30} in
//      the baseline too, so that guard never once took its cheap path).
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
// Tuned staged strides (see the VARIANTS block above). Both are multiples of
// 16 B, so every cp.async destination keeps its alignment.
constexpr int QG_X_STRIDE_T  = 72;  // elements of T (144 B at 2 B/elem)
constexpr int QG_WP_STRIDE_T = 48;  // bytes
// Buffer byte 12 corresponds to stream byte 16c-4, so the 16-bit window of
// chunk-local symbol ts starts at buffer bit 2*ts + 18 + 96.
constexpr int QG_WIN_BIT_BASE = 114;
// Tuned path: after pair-reversing the staged row into words 0..4, the state
// window of chunk-local symbol ts starts at reversed bit 126 - 2*ts.
constexpr int QG_REV_BIT_BASE = 126;
// Words of the staged row that carry payload (bytes 12..32 => words 3..7),
// and therefore the number of reversed words produced.
constexpr int QG_REV_WORDS = 5;
constexpr uint32_t QG_INVALID_PAIR = 0xFFFFFFFFu;
// Runtime variant selector. Mirrored on the Rust side as
// `QTIP_GROUPED_VARIANT_BASELINE` / `_TUNED` (src/qtip/grouped.rs).
constexpr int QTIP_GROUPED_VARIANT_BASELINE = 0;
constexpr int QTIP_GROUPED_VARIANT_TUNED    = 1;
constexpr int QTIP_GROUPED_VARIANT_LDST     = 2;
// Persistent-CTA grid cap (see tuning notes).
constexpr int QG_MAX_GRID = 1024;

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

// ---------------------------------------------------------------------------
// Variant 2: pack two f32 into one 16-bit-pair register in ONE conversion.
//
// `q2b_pack2` above costs four instructions per pair — two scalar converts, a
// shift and an or — which is ~2 ops per decoded weight, the second largest
// per-weight item after the codeword itself. `__floats2bfloat162_rn` /
// `__floats2half2_rn` are the vector converts for exactly this, and both carry
// the SAME round-to-nearest-even the scalar forms use, so the packed result is
// bit-identical.
//
// ⚠️ "Should be bit-identical" is not evidence. The PTX ISA sections for
// `cvt.rn.bf16x2.f32` and `ldmatrix` are TRUNCATED in the published HTML (the
// same gap the wgmma work hit), so the rounding equivalence is NOT taken from
// a doc we could read. It is taken from the hardware: the A/B harness refuses
// to report a timing unless variant 2's output is byte-for-byte equal to the
// baseline's, over millions of output bytes, with the grouped launch counters
// proving both kernels actually ran. Believe the artefact.
//
// 🔴 AND THE ARTEFACT SAYS THIS ONE BOUGHT NOTHING. `cuobjdump -sass` gives
// F2FP = 24 in variants 0, 1 AND 2 alike: **nvcc was already fusing the two
// scalar converts + shift + or into one `F2FP.PACK_AB`.** The explicit vector
// convert is kept because it states the intent and cannot regress if the
// compiler's pattern match changes, but it is worth ZERO instructions today.
// The prediction that it would save ~1.5 ops/weight was made by costing C++
// source as though it mapped 1:1 to SASS. Disassemble first, then predict.
template <typename T>
__device__ __forceinline__ uint32_t q2b_pack2_fast(float lo, float hi);

template <>
__device__ __forceinline__ uint32_t q2b_pack2_fast<__nv_bfloat16>(float lo, float hi) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);  // .x = lo, .y = hi
    uint32_t r;
    memcpy(&r, &p, sizeof(r));
    return r;
}

template <>
__device__ __forceinline__ uint32_t q2b_pack2_fast<__half>(float lo, float hi) {
    const __half2 p = __floats2half2_rn(lo, hi);
    uint32_t r;
    memcpy(&r, &p, sizeof(r));
    return r;
}

// ---------------------------------------------------------------------------
// Variant 2: the whole A fragment in ONE instruction.
//
// The four `a[i]` are four LDS.32 in variants 0/1. `ldmatrix.x4` fetches all
// four 8x8 b16 sub-matrices in a single LDSM. The mapping is exact, not
// approximate — for `mma.m16n8k16` the A fragment is
//     a0 = (row g,   k 2tig..+1)   a1 = (row g+8, k 2tig..+1)
//     a2 = (row g,   k 2tig+8..9)  a3 = (row g+8, k 2tig+8..9)
// and `ldmatrix.x4` returns sub-matrix i in register i, with each sub-matrix
// distributed as `lane -> row lane/4, cols (lane%4)*2 .. +1`. Sub-matrix 0 is
// rows 0-7 x k 0-7, 1 is rows 8-15 x k 0-7, 2 is rows 0-7 x k 8-15, 3 is rows
// 8-15 x k 8-15 — so register i lands on a[i] with no shuffling, provided
// lane l supplies the address of
//     s_x[(l & 15)][kb + (l >= 16 ? 8 : 0)]
//
// Bank behaviour: ldmatrix resolves 8 rows per phase, and at variant 1's
// 144 B stride the eight addresses of a phase sit at word (36*row) => banks
// (4*row) % 32 = 0,4,...,28 — eight distinct banks, conflict-free. At the
// ORIGINAL 128 B stride all eight would collide on one bank, so this
// instruction is only safe *because* the Phase 1 padding is already there.
__device__ __forceinline__ void q2b_ldmatrix_x4(const void* smem_row, uint32_t* a) {
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addr));
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
// Tuned path (variant 1).
// ---------------------------------------------------------------------------

// Reverse the sixteen 2-bit groups of a word, keeping the two bits inside a
// group in order. __brev reverses all 32 bits; the mask/shift pair puts each
// group's bits back the right way round.
__device__ __forceinline__ uint32_t q2b_pair_reverse_32(uint32_t x) {
    const uint32_t r = __brev(x);
    return ((r & 0x55555555u) << 1) | ((r >> 1) & 0x55555555u);
}

// Rewrite a staged 48-byte packed row in place so the state window becomes a
// plain right shift. Payload is words 3..7 (bytes 12..32); reversed word j
// (j = 0..4) is pair_reverse_32 of payload word 7-j. ONE lane owns a whole
// row, so reading all five sources before writing makes the in-place overlap
// (words 3 and 4) safe with no extra barrier.
__device__ __forceinline__ void q2b_reverse_row(uint8_t* s_row) {
    uint32_t* w = reinterpret_cast<uint32_t*>(s_row);
    uint32_t src[QG_REV_WORDS];
    #pragma unroll
    for (int j = 0; j < QG_REV_WORDS; ++j) src[j] = w[7 - j];
    #pragma unroll
    for (int j = 0; j < QG_REV_WORDS; ++j) w[j] = q2b_pair_reverse_32(src[j]);
}

// The four states a lane needs for one (kf, f): chunk-local symbols
// ts0 + {0, 1, 8, 9}. All four windows start in reversed word `base >> 5` and
// end no later than the next word, so two loads and four funnel shifts cover
// them (proved in the VARIANTS block; asserted host-side in revcheck).
__device__ __forceinline__ void q2b_states_rev(const uint8_t* s_row, int ts0, uint32_t* st) {
    // The two facts the two-word load rests on, both mechanical consequences
    // of the tile geometry — asserted here so a future retile cannot silently
    // walk off the end of the reversed words:
    //   * ts0 = kb + 2*tig with kb a multiple of 16, so 2*ts0 mod 32 = 4*tig
    //     and base mod 32 is in {18,22,26,30} => every shift below is >= 0.
    //   * base is in [18, 126], so base>>5 is in [0,3] and the +1 word is
    //     still inside the QG_REV_WORDS reversed words.
    static_assert(QG_TILE_K == 64 && QG_REV_BIT_BASE == 126 && QG_REV_WORDS == 5,
                  "q2b_states_rev's two-word window proof is tied to this tile geometry");
    const uint32_t* r = reinterpret_cast<const uint32_t*>(s_row);
    const int base = QG_REV_BIT_BASE - 2 * ts0;
    const uint32_t lo = r[base >> 5];
    const uint32_t hi = r[(base >> 5) + 1];
    const unsigned sh = (unsigned)(base & 31);
    st[0] = __funnelshift_r(lo, hi, sh) & Q2B_STATE_MASK;
    st[1] = __funnelshift_r(lo, hi, sh - 2u) & Q2B_STATE_MASK;
    st[2] = __funnelshift_r(lo, hi, sh - 16u) & Q2B_STATE_MASK;
    st[3] = __funnelshift_r(lo, hi, sh - 18u) & Q2B_STATE_MASK;
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
template <typename T, int X_STRIDE, int WP_STRIDE>
__device__ __forceinline__ void q2b_stage_chunk(
    int c,
    T*             s_x_buf,   // [QG_TILE_M * X_STRIDE]
    uint8_t*       s_wp_buf,  // [QG_TILE_N * WP_STRIDE]
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
        void* dst = (uint8_t*)(s_x_buf + (size_t)slot * X_STRIDE) + piece * 16;
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
        uint8_t* srow = s_wp_buf + (size_t)r * WP_STRIDE;
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

template <typename T, int VARIANT>
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
    // Staged strides differ per variant (bank-conflict fix); everything else
    // about the layout, the tile geometry and the math is identical.
    constexpr bool TUNED = VARIANT >= QTIP_GROUPED_VARIANT_TUNED;
    constexpr bool LDST  = VARIANT >= QTIP_GROUPED_VARIANT_LDST;
    constexpr int X_STRIDE  = TUNED ? QG_X_STRIDE_T : QG_TILE_K;
    constexpr int WP_STRIDE = TUNED ? QG_WP_STRIDE_T : QG_WP_STRIDE;
    static_assert(X_STRIDE * sizeof(T) % 16 == 0, "cp.async needs 16 B aligned x rows");
    static_assert(WP_STRIDE % 16 == 0, "cp.async needs 16 B aligned packed rows");

    __shared__ __align__(16) T       s_x[2][QG_TILE_M * X_STRIDE];
    __shared__ __align__(16) uint8_t s_wp[2][QG_TILE_N * WP_STRIDE];
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
        q2b_stage_chunk<T, X_STRIDE, WP_STRIDE>(
            0, &s_x[0][0], &s_wp[0][0], s_pairs, x, packed,
            expert, n0, n_rows, num_symbols, packed_per_row);
        q2b_cp_commit();

        for (int c = 0; c < n_chunks; ++c) {
            const int buf = c & 1;
            if (c + 1 < n_chunks) {
                q2b_stage_chunk<T, X_STRIDE, WP_STRIDE>(
                    c + 1, &s_x[buf ^ 1][0], &s_wp[buf ^ 1][0],
                    s_pairs, x, packed, expert, n0, n_rows,
                    num_symbols, packed_per_row);
                q2b_cp_commit();
                q2b_cp_wait<1>();
            } else {
                q2b_cp_wait<0>();
            }
            __syncthreads();

            // Tuned: pair-reverse this buffer's packed rows so the state
            // window is a plain shift. Rows [warp*16, warp*16+16) are read
            // ONLY by this warp, so __syncwarp is the whole synchronisation
            // cost — no extra block-wide barrier.
            if constexpr (TUNED) {
                if (lane < 16) {
                    q2b_reverse_row(&s_wp[buf][(warp * 16 + lane) * WP_STRIDE]);
                }
                __syncwarp();
            }

            // 4 k-fragments of 16 symbols. A is redundant across warps
            // (cheap smem broadcasts); every B weight is decoded exactly
            // once block-wide (tig 0..3 cover k 0..7/8..15 of row g).
            #pragma unroll
            for (int kf = 0; kf < 4; ++kf) {
                const int kb = kf * 16;
                const T* s_x_buf = &s_x[buf][0];
                uint32_t a[4];
                if constexpr (LDST) {
                    // One LDSM instead of four LDS.32. Lane l addresses
                    // row (l & 15), k-half (l >= 16); see q2b_ldmatrix_x4.
                    q2b_ldmatrix_x4(
                        &s_x_buf[(size_t)(lane & 15) * X_STRIDE + kb + ((lane >= 16) ? 8 : 0)], a);
                } else {
                    a[0] = *reinterpret_cast<const uint32_t*>(&s_x_buf[(size_t)(g    ) * X_STRIDE + kb + tig * 2    ]);
                    a[1] = *reinterpret_cast<const uint32_t*>(&s_x_buf[(size_t)(g + 8) * X_STRIDE + kb + tig * 2    ]);
                    a[2] = *reinterpret_cast<const uint32_t*>(&s_x_buf[(size_t)(g    ) * X_STRIDE + kb + tig * 2 + 8]);
                    a[3] = *reinterpret_cast<const uint32_t*>(&s_x_buf[(size_t)(g + 8) * X_STRIDE + kb + tig * 2 + 8]);
                }
                #pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const uint8_t* srow = &s_wp[buf][(warp * 16 + f * 8 + g) * WP_STRIDE];
                    const float s = sclB[f];
                    float w0, w1, w2, w3;
                    if constexpr (TUNED) {
                        uint32_t st[4];
                        q2b_states_rev(srow, kb + tig * 2, st);
                        w0 = q2b_decode(st[0], mult) * s;
                        w1 = q2b_decode(st[1], mult) * s;
                        w2 = q2b_decode(st[2], mult) * s;
                        w3 = q2b_decode(st[3], mult) * s;
                    } else {
                        w0 = q2b_decode_smem(srow, kb + tig * 2,     mult) * s;
                        w1 = q2b_decode_smem(srow, kb + tig * 2 + 1, mult) * s;
                        w2 = q2b_decode_smem(srow, kb + tig * 2 + 8, mult) * s;
                        w3 = q2b_decode_smem(srow, kb + tig * 2 + 9, mult) * s;
                    }
                    uint32_t b0, b1;
                    if constexpr (LDST) {
                        b0 = q2b_pack2_fast<T>(w0, w1);
                        b1 = q2b_pack2_fast<T>(w2, w3);
                    } else {
                        b0 = q2b_pack2<T>(w0, w1);
                        b1 = q2b_pack2<T>(w2, w3);
                    }
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
              int variant,                                                     \
              cudaStream_t stream) {                                           \
        const int n_tiles_n = (n_rows + QG_TILE_N - 1) / QG_TILE_N;            \
        const long max_tiles = (long)max_m_tiles * (long)n_tiles_n;            \
        const int grid =                                                       \
            (int)(max_tiles < (long)QG_MAX_GRID ? max_tiles : (long)QG_MAX_GRID); \
        if (grid <= 0) return;                                                 \
        switch (variant) {                                                     \
        case QTIP_GROUPED_VARIANT_LDST:                                        \
            qtip2b_grouped_gemm_kernel<T, QTIP_GROUPED_VARIANT_LDST>           \
                <<<grid, QG_THREADS, 0, stream>>>(                             \
                    d_packed, d_row_scales, d_x_rotated, d_sorted_pairs,       \
                    d_tile_expert, d_tile_row_start, d_offsets, d_num_tiles,   \
                    d_y, n_rows, packed_per_row, num_symbols, mult);           \
            break;                                                             \
        case QTIP_GROUPED_VARIANT_TUNED:                                       \
            qtip2b_grouped_gemm_kernel<T, QTIP_GROUPED_VARIANT_TUNED>          \
                <<<grid, QG_THREADS, 0, stream>>>(                             \
                    d_packed, d_row_scales, d_x_rotated, d_sorted_pairs,       \
                    d_tile_expert, d_tile_row_start, d_offsets, d_num_tiles,   \
                    d_y, n_rows, packed_per_row, num_symbols, mult);           \
            break;                                                             \
        default:                                                               \
            qtip2b_grouped_gemm_kernel<T, QTIP_GROUPED_VARIANT_BASELINE>       \
                <<<grid, QG_THREADS, 0, stream>>>(                             \
                    d_packed, d_row_scales, d_x_rotated, d_sorted_pairs,       \
                    d_tile_expert, d_tile_row_start, d_offsets, d_num_tiles,   \
                    d_y, n_rows, packed_per_row, num_symbols, mult);           \
            break;                                                             \
        }                                                                      \
    }

Q2B_GROUPED_GEMM_LAUNCHER(launch_qtip2b_grouped_gemm_bf16, __nv_bfloat16)
Q2B_GROUPED_GEMM_LAUNCHER(launch_qtip2b_grouped_gemm_f16,  __half)

#undef Q2B_GROUPED_GEMM_LAUNCHER

} // extern "C"
