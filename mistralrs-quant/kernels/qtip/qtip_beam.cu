// QTIP beam-search trellis quantize kernel (LUT rung).
//
// This is the GPU twin of `qtip/viterbi.rs::beam_quantize_row` (wave13-AD,
// PR #29). It replaces the exhaustive prefix-grouped DP in `qtip_quantize.cu`
// with a *pruned* dynamic program that keeps only the best `beam_w` states per
// timestep, and it is BIT-IDENTICAL to the CPU beam at the same width — the
// parity test in `qtip/mod.rs` compares packed bytes, not cosine similarity.
//
// GEOMETRY IS A TEMPLATE PARAMETER (see qtip_geom.cuh)
// ---------------------------------------------------
// Three rungs are compiled:
//
//     K=4 / V=2 / L=16   2.00 bpw, the shipped rung
//     K=8 / V=4 / L=12   2.00 bpw, fewer decode instructions per weight
//                        (the ABSOLUTE inst/weight figures are SUSPENDED —
//                        see qtip_geom.cuh; the census's unroll-differential
//                        method did not reproduce its own K=8 control)
//     K=9 / V=4 / L=12   2.25 bpw — THE BAKE TARGET. K=8/V=4/L=12 is
//                        quality-CLOSED and K=9 is not; the sweep, its six
//                        codebook designs and the Dw_cos figures are recorded
//                        once, in `qtip/trellis_v4l12.rs`'s module docs.
//
// The K=8 rung is NOT reachable by bumping the constants. Three things in this
// kernel are shaped by `2^(L-K)` and `2^K`, and at K=8/L=12 all three invert:
//
//   * `2^(L-K)` groups falls 4096 -> 16, so "one thread per group" would leave
//     240 of 256 threads idle;
//   * `2^K` successors per group rises 16 -> 256, so a per-thread `cand[2^K]`
//     register array becomes `cand[256]` — a GUARANTEED local-memory spill, and
//     the spill is where the discredited "~1,700 s/layer" encode figure came
//     from (see the res-usage note below);
//   * the 32 KiB direct-indexed group table falls to 128 B, which is too small
//     to keep doubling as the backtrace staging buffer.
//
// The fix for the first two is one change: **`LANES` threads co-operate on one
// group** instead of one. `LANES = min(2^K, THREADS / 2^(L-K))`, so
//
//     K=4/L=16:  4096 groups >= 256 threads  ->  LANES = 1,  cand[16/1 ] = 16
//     K=8/L=12:    16 groups                 ->  LANES = 16, cand[256/16] = 16
//     K=9/L=12:     8 groups                 ->  LANES = 32, cand[512/32] = 16
//
// The block stays fully occupied at all three geometries and the register array
// stays 16 entries at all three — the candidate count per timestep is `2^L`-
// saturated (`ng * 2^K = 4096`) every time, so the SAME work is simply
// re-blocked. `LANES == 1` collapses every expression below to the shipped
// code, which is why the K=4 rung stays byte-identical.
//
// 🔑 THAT INVARIANCE IS ALSO THE K=9 COST ARGUMENT. Going K=8 -> K=9 at fixed
// V=4/L=12 changes NO per-timestep quantity in this kernel: same 4096
// candidates, same cand[16], same ~3.87 radix passes over the same 32-bit cost
// key, same V=4 branch metric, same `2^K * V * 4 B` of LUT per group over half
// as many groups (65,536 B per timestep either way), same `beam_w` trace
// words, and `num_symbols = in_features / V` is K-independent. The one term
// that does move is the packer: one byte store per symbol at K=8 becomes two
// byte read-modify-writes in thread 0's serial backtrace. So K=9 s/layer is
// K=8 s/layer plus that term — see the encode-cost note below.
//
// The third is handled by sizing the shared scratch as
// `max(2^(L-K) u64, STAGE_U64_MIN)`: at K=4 that is exactly the 4096-entry
// group table that already staged the backtrace (no change at all), and at K=8
// (128 B table) and K=9 (64 B table) it is the same 16 KiB stage over a table
// far too small to serve as one. That is not a smaller staging
// budget in the terms that matter — the backtrace's dependent global loads are
// `num_symbols / window`, and V=4 halves `num_symbols` exactly as the halved
// window doubles it: 9472/32 == 4736/16 == 296 per row, unchanged.
//
// 🔑 WHAT THIS GEOMETRY GIVES THE SERVING PATH — stated precisely, because the
// obvious version of the claim is wrong. Shared memory is a PER-BLOCK resource:
// this kernel giving some back does not hand a budget to a serving kernel that
// never runs beside it. Two separate facts:
//
//   * This kernel's own block falls ~37.1 KiB -> ~19.3 KiB, because the
//     direct-indexed group table goes 32 KiB -> 128 B. That is an encoder
//     occupancy headroom result, nothing more.
//   * The one that matters for serving is a property of the GEOMETRY, not of
//     this file: the K=8/L=12 codebook is 2^12 x V=4 = 16,384 values, i.e.
//     **32,768 B in fp16 — shared-memory resident**. The shipped K=4/L=16 LUT
//     is 2^16 x V=2 f32 = 512 KiB and never could be; it is the "dependent,
//     data-scattered trip to L2 on EVERY candidate" qtip_codebook.cuh opens by
//     describing. A resident codebook removes that trip for the decode side.
//
// ⚠ NOT related to either: the layer-25 bake OOM (FACTS.md:1330). That was
// DEVICE-allocator fragmentation with 72 GiB still free, worked around with
// `ARC_QTIP_EXPERT_BATCH=8`. No shared-memory change touches it.
//
// WHY IT IS FASTER, MEASURED IN BYTES
// -----------------------------------
// The exhaustive kernel moves, per (row, timestep):
//     read  prev[]      2^L * 4 B = 262144 B   (global; the cost ping-pong)
//     write curr[]      2^L * 4 B = 262144 B   (global)
//     write backtrace   2^(L-K)   =   4096 B   (global)
//     read  LUT         2^L * 8 B = 524288 B   (L2-resident, shared by blocks)
// i.e. ~528 KB of HBM traffic per symbol position. Multiplying by the ~3.2e9
// symbol positions in one V4-Flash layer gives ~1.7 PB / layer, which at
// H200's 4.8 TB/s is ~5.9 min — within 1.4x of the 8.5 min/layer FACTS.md
// measures on a healthy box. The exhaustive quantizer is HBM-bound, which is
// exactly the "99% util at 132 W" signature FACTS.md records for a starved
// box: there is almost no arithmetic behind that traffic.
//
// The beam kernel moves, per (row, timestep):
//     write trace       beam_w * 4 B = 1024 B at W=256   (global)
//     read  LUT         n_groups * 2^K * V * 4 B         (L2-resident)
//     read  weight      V * 4 B
// The 512 KB cost ping-pong disappears entirely: the live state set is
// `beam_w` entries, which fit in shared memory with room to spare. HBM traffic
// per symbol position drops ~512x.
//
// WHAT A RUNG COSTS TO ENCODE — AND WHY `(n/V) * W * 2^K` IS NOT IT
// -----------------------------------------------------------------
// That formula predicts K=4 -> K=8 costs 8x MORE and produced the discredited
// "~1,700 s/layer / 20 h / $30" figure. This kernel falsifies it in two lines
// of its own source: the `& G::GROUP_MASK` in step 1a masks the predecessor to
// `2^(L-K)`, so raising K SHRINKS the live group count by exactly the factor by
// which it grows the alphabet, and `n_cand = ng * 2^K` at the end of step 2 is
// therefore saturated at `2^L`. The candidate
// count per timestep is 4096 at K=4/L=16 (256 groups x 16), at K=8/L=12
// (16 x 256) and at K=9/L=12 (8 x 512) alike, and `cand[]` is 16 entries at all
// three. `W` is not a factor either — measured, W=256 = 82.7 s/layer vs
// W=32 = 83.6 s/layer, ~1% (FACTS.md:1320).
//
// What DOES move, per layer:
//
//   (a) timesteps.  `num_symbols = in_features / V`, so V=2 -> V=4 halves them.
//   (b) candidates/timestep, cand[], radix passes, LUT bytes per ROW.  All
//       invariant (see above; LUT is 2^K*V*4 B per group over 2^(L-K) groups,
//       which is 65,536 B/timestep at both V=4 rungs against half as many
//       timesteps as V=2's 32,768 B).
//   (c) branch metric.  V=2 -> V=4 doubles the arithmetic per candidate, but a
//       candidate's cost is dominated by its ~3.87 radix histogram passes and
//       its compaction test, not by the metric — so this is a fraction of 2x.
//   (d) K=8 -> K=9 at fixed V=4/L=12.  (a), (b) and (c) are ALL unchanged. Only
//       the packer moves: one byte store per symbol becomes two byte
//       read-modify-writes, in thread 0's serial backtrace, against an
//       L1-resident row of <= 2 KiB.
//
// ⇒ From the measured anchor `qtip2` beam W=256 = 372.0 s/layer on an A100
//   (FACTS.md:990; harness validated to 0.3% against the published V4 bake's
//   370-376 s/layer, FACTS.md:993):
//
//       K=9/V=4/L=12  ~=  372.0 x 0.5 x [1.0 .. 1.4]  =  186-260 s/layer
//
//   i.e. the 2.25 bpw rung is CHEAPER to bake than the 2.00 bpw rung that
//   ships, not 8x dearer. 43 layers ⇒ 2.2-3.1 h ⇒ $3.3-$4.6 at $1.49/hr, or
//   1.1-1.6 h across two devices at the same total cost (`ARC_BAKE_DEVICES`,
//   measured 1.97x on 2 GPUs, byte-identical — wave29-BD-rung-decision.md:431).
//
// ⚠ PROJECTED, NOT MEASURED. Nothing has run a V=4 bake. This is a division
//   from a measured V=2 anchor by terms this file's own structure fixes; the
//   only honest way to close it is to bake a layer and difference the markers.
//
// SHARED-MEMORY RESIDENCY (the claim wave13-AF was asked to test)
// --------------------------------------------------------------
// At beam_w = 256 the *beam* is 2 KiB (cost f32 + state u16 + parent u16).
// At K=4/L=16 it is not the beam that sets the shared-memory budget — it is
// the 2^(L-K) = 4096-entry group-reduction table below, at 8 B/prefix = 32 KiB.
// Total static shared memory is ~37.1 KiB at W=256, ~35.6 KiB at W=128 and
// ~34.8 KiB at W=64: all three are comfortably inside the 48 KiB static
// budget. At K=8/L=12 the table is 128 B and the 16 KiB backtrace stage is the
// budget instead, for ~19.3 KiB total.
//
// ALGORITHM (mirrors the Rust beam exactly)
// -----------------------------------------
// Successor states of a beam entry with state `p` are
// `((p << K) | sym) & STATE_MASK`, which depends on `p` only through
// `g = p & ((1 << (L-K)) - 1)` — the same prefix-grouping the exhaustive
// kernel exploits, applied to the beam instead of the whole state space.
// Consequences:
//   * All 2^K successors of a group share one predecessor set, so the CPU's
//     "dedup by successor state, keep min predecessor cost, first-seen wins
//     ties" collapses to ONE min per group. The CPU visits the beam in
//     ascending state order, and all predecessors of a fixed successor share
//     their low L-K bits, so "first-seen on ties" == "lowest predecessor
//     state". A 64-bit `atomicMin` over `(total_order_key(cost) << 16) | state`
//     implements that rule exactly and is order-independent.
//   * Distinct groups produce disjoint successor sets, so the expanded
//     candidate list needs no deduplication at all: it is exactly
//     `n_groups * 2^K` entries with pairwise-distinct states — hence pairwise
//     distinct 48-bit selection keys.
//
// The CPU prunes with `select_nth_unstable_by((cost, state))` + `truncate`,
// i.e. "the `beam_w` smallest by `(f32::total_cmp, state)`". Because the keys
// are unique, that set is found here by a radix-select on the 48-bit key
// (6 x 8-bit digit passes with an early exit as soon as a digit bin holds a
// single candidate, which is the common case) followed by a deterministic
// prefix-sum compaction. No sort is needed: nothing downstream depends on the
// beam's order — the group reduction is an atomicMin, the final choice is an
// argmin, and the backtrace follows explicit parent indices. That last property
// is what makes the `LANES > 1` re-blocking safe: splitting a group's
// successors across `LANES` threads changes which beam SLOT a survivor lands
// in, and nothing reads a slot index except as a parent pointer it also wrote.
//
// PARITY
// ------
// Every arithmetic operation on the trellis path goes through
// `qtip_exact_fp.cuh`, which forbids FMA contraction and approximate division
// (both of which `--use_fast_math` would otherwise introduce). See that header
// for the full argument.
//
// LAYOUT / LAUNCH
// ---------------
//   grid  = (rows_in_flight, 1, 1)      one block per row
//   block = (QB_THREADS = 256, 1, 1)    thread t owns beam slot t; for the
//                                       expansion it owns group `t / LANES`
//                                       and that group's `2^K / LANES` symbols
//   trace = [rows_in_flight, num_symbols, beam_w] u32, packed
//           `(state << 16) | parent_slot`. 9.7 MB per row at W=256 and
//           num_symbols=9472 (K=4/V=2); half that at V=4, which halves
//           `num_symbols` for the same row width.
//
// SM80+ (gated by `has_qtip_kernels` in build.rs; uses `__match_any_sync`,
// which needs SM70+).

#include <cstdint>
#include <cuda_runtime.h>

#include "qtip_exact_fp.cuh"
// Codebook selection (stored Gaussian LUT vs in-register sum2 code).
#include "qtip_codebook.cuh"
// K / V / L and the symbol packer.
#include "qtip_geom.cuh"
// Block primitives shared with the qtip2b beam (histogram / scan / radix bin).
#include "qtip_beam_common.cuh"

namespace {

// One thread per beam slot; both the beam width and the number of live groups
// are bounded by the block size. 256 keeps the whole kernel inside the 48 KiB
// static shared-memory budget (see the header) and matches the W=256 that
// PR #29 measured quality-neutral against exhaustive Viterbi.
constexpr int QB_THREADS  = 256;
constexpr int QB_MAX_BEAM = QB_THREADS;
constexpr int QB_WARPS    = QB_THREADS / 32;

constexpr unsigned long long QB_KEY_MAX = ~0ull;

// Block shape implied by a geometry. Every member collapses to the shipped
// K=4/V=2/L=16 value when `LANES == 1`.
template <class G>
struct QbShape {
    // Threads co-operating on ONE group. 1 when the trellis has at least as
    // many groups as the block has threads (K=4/L=16: 4096 >= 256); otherwise
    // the block is split evenly across the groups that exist (K=8/L=12: 16
    // groups x 16 lanes).
    static constexpr int LANES =
        (G::GROUP_COUNT >= (uint32_t)QB_THREADS) ? 1 : (QB_THREADS / (int)G::GROUP_COUNT);
    // Groups the block can expand in one pass.
    static constexpr int GROUPS = QB_THREADS / LANES;
    // Successors each thread evaluates — the `cand[]` register array. 16 at
    // both shipped geometries; the spill landmine below is about THIS number.
    static constexpr int CAND = (int)G::ALPHABET / LANES;
    // Live groups are bounded by both the beam and the trellis.
    static constexpr int MAX_GROUPS =
        (QB_MAX_BEAM < (int)G::GROUP_COUNT) ? QB_MAX_BEAM : (int)G::GROUP_COUNT;

    // Shared u64 scratch: the group-reduction table, which also stages the
    // backtrace. `STAGE_U64_MIN` is the floor that keeps the staged walk worth
    // doing once the group table is too small to serve as the stage — 2048 u64
    // = 4096 u32 slots = a 16-timestep window at W=256, which at V=4 gives the
    // same `num_symbols / window` dependent-load count as the K=4 rung's
    // 32-timestep window at V=2.
    static constexpr int STAGE_U64_MIN = 2048;
    static constexpr int SCRATCH_U64 =
        ((int)G::GROUP_COUNT > STAGE_U64_MIN) ? (int)G::GROUP_COUNT : STAGE_U64_MIN;

    static_assert(QB_THREADS == 256, "the digit histogram is one bin per thread");
    static_assert(QB_MAX_BEAM <= QB_THREADS, "one thread per beam slot");
    static_assert(LANES >= 1 && LANES <= (int)G::ALPHABET,
                  "a group cannot be split across more threads than it has successors");
    static_assert((int)G::ALPHABET % LANES == 0,
                  "the alphabet must divide evenly across a group's lanes");
    static_assert(QB_THREADS % LANES == 0, "lanes must tile the block");
    static_assert(MAX_GROUPS <= GROUPS,
                  "every live group must have a lane team in the block");
    // `cand[]` MUST stay in registers; see the spill note on
    // QB_MIN_BLOCKS_PER_SM. 16 is what all three instantiated geometries
    // produce (K=4: 16/1, K=8: 256/16, K=9: 512/32), and it is the number
    // `cuobjdump -res-usage` has been checked at (LOCAL:0, REG<=62). A geometry
    // that would push it higher must be measured before it is trusted, not
    // reasoned about — hence the ceiling rather than a comment.
    static_assert(CAND >= 1 && CAND <= 16,
                  "cand[] must stay small enough to live in registers");
    // `keep_mask` is a 32-bit bitset over `cand[]`.
    static_assert(CAND <= 32, "keep_mask is 32 bits wide");
    // The backtrace stages trace windows in the shared scratch; a window must
    // hold at least one full timestep or the staged walk cannot advance.
    static_assert(QB_MAX_BEAM <= SCRATCH_U64 * 2,
                  "shared scratch must stage at least one timestep of the trace");
};

// Blocks per SM this kernel is compiled to fit.
//
// 4 caps registers at 65,536 / 4 / 256 = 64 per thread and buys 32 warps
// instead of the 24 that 3 blocks give (+33% occupancy, which for a
// latency-bound kernel is a ~1.33x throughput term).
//
// MEASURED, `cuobjdump -res-usage`, CUDA 12.4, build.rs's exact flags
// (arc-tools/qtip_beam_res_usage_check.sh, which CI runs on every PR — the
// table below is from run 32265671233, both matrix arches):
//
//                              sm_80                        sm_90
//     K=9/V=4/L=12  LUT       REG:64 SHARED:19632 LOCAL:0   REG:64 SHARED:20656 LOCAL:0
//     K=9/V=4/L=12  computed  REG:61 SHARED:19632 LOCAL:0   REG:63 SHARED:20656 LOCAL:0
//     K=8/V=4/L=12  LUT       REG:62 SHARED:19696 LOCAL:0   REG:63 SHARED:20720 LOCAL:0
//     K=8/V=4/L=12  computed  REG:59 SHARED:19696 LOCAL:0   REG:62 SHARED:20720 LOCAL:0
//     K=4/V=2/L=16  LUT       REG:60 SHARED:38000 LOCAL:0   REG:63 SHARED:39024 LOCAL:0
//     K=4/V=2/L=16  computed  REG:60 SHARED:38000 LOCAL:0   REG:63 SHARED:39024 LOCAL:0
//
// STACK:0 on all twelve. Every one is inside the 64-register budget with NO
// spill, so the occupancy is real at all three geometries — `cand[]` stayed in
// registers at K=9, where a naive `cand[2^K]` would have been 512 entries.
//
// Two things this table settles that no argument could:
//
//   * THE K=4 AND K=8 sm_80 NUMBERS ARE UNCHANGED from before K=9 existed
//     (REG:62/59 SHARED:19696 and REG:60/60 SHARED:38000, to the byte). The
//     shipped rung's register allocation and shared-memory layout did not move
//     when the packer became a bitstream — which is a compiled fact about
//     codegen, not an inference from the `if constexpr` keeping its source
//     identical.
//   * K=9's SHARED is exactly 64 B BELOW K=8's at both arches (19,632 vs
//     19,696; 20,656 vs 20,720). That is the group table falling from 16
//     entries to 8 at 8 B each — the predicted number, arrived at
//     independently by the compiler.
//
// ⚠ ZERO REGISTER HEADROOM AT K=9/LUT. It lands on REG:64 at both arches,
// which is exactly the cap `__launch_bounds__(256, 4)` imposes
// (65,536 / 4 / 256). It did NOT spill, so 4 blocks/SM holds — but it is the
// tightest of the six and the next register added to its hot path is a spill,
// not a warning. Anything touching the K=9 expansion, the radix loop or the
// packer must re-run the gate and read the number, not assume the margin that
// K=8 (REG:62/59) still has.
//
// Shared memory is not the constraint at any geometry: 4 x 38,000 B is 148 KiB
// of the 228 KiB an SM has.
//
// (The older `REG:80 ... SHARED:38992` line this note used to quote was a
// pre-wave16 measurement of a kernel that no longer exists — wave16/wave17
// rewrote the selection and the scan buffers. It is superseded by the table
// above, not contradicted by it.)
//
// ⚠ THE FAILURE MODE IS SILENT. `__launch_bounds__` does not refuse to compile
// when it cannot reach the register budget — it SPILLS to local memory, and a
// spilled load inside the radix loop executes 2^K/LANES x ~3.87 times per
// timestep, which would cost more than the occupancy gains. Register allocation
// is nvcc's scheduling decision and CANNOT be proven from source.
//
// REQUIRED CHECK after any change here, on the build box, no GPU needed:
//     cuobjdump -res-usage <obj> | grep -A1 beam_kernel
// `LOCAL:` must remain 0 — for EVERY geometry instantiated below, not just the
// shipped one. This is not a formality: a `cand[]` that spills is precisely the
// mis-scoping that produced the discredited "~1,700 s/layer / 20 h / $30"
// K=8/L=12 encode estimate. `.github/workflows/cuda_compile_check.yaml` runs
// this check on every PR so it cannot be skipped; if it ever fails, set
// QB_MIN_BLOCKS_PER_SM back to 3 rather than trading a real latency regression
// for a nominal occupancy gain — that raises the cap to 65,536/3/256 = 85
// registers, which every geometry in the table above clears with room. (It is
// NOT necessarily allocation-preserving: with a looser cap nvcc is free to
// take more registers than the 59-64 measured at 4 blocks. Re-run the gate and
// read the numbers; do not assume the table still applies.)
//
// Related landmine: `unsigned int cand[CAND]` must stay in registers, which
// requires EVERY loop indexing it to be fully unrolled. Weakening any of those
// `#pragma unroll`s sends the array to local memory on its own.
constexpr int QB_MIN_BLOCKS_PER_SM = 4;

template <class G, bool COMPUTED_CB>
__global__ void __launch_bounds__(QB_THREADS, QB_MIN_BLOCKS_PER_SM)
qtip_quantize_rows_beam_kernel(
    const float*   __restrict__ weight,      // [n_rows, in_features]
    const float*   __restrict__ lut,         // [2^L * V] (unused when computed)
    const float*   __restrict__ row_scales,  // [n_rows]
    uint8_t*       __restrict__ packed,      // [n_rows, num_symbols / (8/K)]
    uint32_t*      __restrict__ trace,       // [BATCH, num_symbols, beam_w]
    int in_features,
    int num_symbols,
    int row_offset,
    int beam_w,
    unsigned int cb_mult
) {
    using QS = QbShape<G>;

    // The group-reduction table occupies the first `2^(L-K)` entries; the whole
    // array doubles as the backtrace staging buffer once the forward pass is
    // done. ~37.1 KiB at K=4/L=16 (SCRATCH_U64 == GROUP_COUNT == 4096, exactly
    // the shipped allocation), ~19.3 KiB at K=8/L=12.
    __shared__ unsigned long long s_gmin[QS::SCRATCH_U64];
    __shared__ float          s_beam_cost  [QB_MAX_BEAM];
    __shared__ unsigned short s_beam_state [QB_MAX_BEAM];
    __shared__ unsigned short s_beam_parent[QB_MAX_BEAM];
    __shared__ unsigned short s_grp_g      [QS::MAX_GROUPS];
    __shared__ float          s_grp_cost   [QS::MAX_GROUPS];
    __shared__ unsigned short s_grp_parent [QS::MAX_GROUPS];
    __shared__ unsigned int   s_hist[256];
    // Two scan scratch buffers, alternated by `qb_scan_slot` (see
    // qb_block_excl_scan): consecutive scans never touch the same words, so no
    // barrier is needed to separate them.
    __shared__ unsigned int   s_warp_tot_buf[2 * QB_WARPS];
    unsigned int qb_scan_slot = 0u;
    __shared__ unsigned int   s_sel_bin;
    __shared__ unsigned int   s_sel_k;
    __shared__ unsigned int   s_sel_cnt;
    __shared__ unsigned long long s_threshold;
    __shared__ unsigned long long s_fin_key;
    __shared__ unsigned int   s_fin_slot;
    __shared__ unsigned int   s_walk_slot;
    __shared__ int            s_beam_n;
    __shared__ int            s_n_groups;

    const int local_row = blockIdx.x;
    const int row       = row_offset + local_row;
    const int tid       = (int)threadIdx.x;

    // Expansion identity: which group this thread serves, and which slice of
    // that group's alphabet. At LANES == 1 this is `(tid, 0)` and every use
    // below folds back to the shipped indexing.
    const int lane_g = tid / QS::LANES;
    const int lane_i = tid % QS::LANES;

    const int pkd_per_row = qtip_packed_bytes_per_row<G>(num_symbols);

    const float* __restrict__ my_row = weight + (size_t)row * (size_t)in_features;
    uint8_t*  __restrict__ my_pkd    = packed + (size_t)row * (size_t)pkd_per_row;
    uint32_t* __restrict__ my_trace  =
        trace + (size_t)local_row * (size_t)num_symbols * (size_t)beam_w;

    const float inv_scale = qtip_inv_scale_exact(row_scales[row]);

    // The group table is cleared once per row: every timestep releases exactly
    // the slots it claimed (one winner per non-empty group), so it re-enters
    // each timestep all-empty without a 2^(L-K)-entry memset.
    for (int i = tid; i < (int)G::GROUP_COUNT; i += QB_THREADS) {
        s_gmin[i] = QB_KEY_MAX;
    }
    // The digit histogram is zeroed ONCE per row. Every radix pass clears its
    // own bin immediately after reading it (see below), so "all zero on entry"
    // is an invariant the passes maintain rather than a cost they re-pay.
    s_hist[tid] = 0u;
    if (tid == 0) {
        s_beam_n   = 0;
        s_n_groups = 0;
    }
    __syncthreads();

    for (int t = 0; t < num_symbols; ++t) {
        // The V targets this symbol must reproduce.
        float tgt[G::V];
        #pragma unroll
        for (uint32_t v = 0; v < G::V; ++v) {
            tgt[v] = qtip_scaled_target_exact(
                my_row[(size_t)t * (size_t)G::V + (size_t)v], inv_scale);
        }

        // ---- 1. Expansion frontier ------------------------------------------
        // t == 0: the decoder starts from state 0, so the reachable states are
        // exactly 0..ALPHABET-1 with cost = branch metric. That is the same
        // shape as "one group g = 0 whose predecessor cost is 0" (and
        // `0.0f + err == err` exactly), so the generic path below covers it.
        if (t == 0) {
            if (tid == 0) {
                s_grp_g[0]      = 0u;
                s_grp_cost[0]   = 0.0f;
                s_grp_parent[0] = 0u;
                s_n_groups      = 1;
            }
            __syncthreads();
        } else {
            const int bn = s_beam_n;

            // 1a. Per-group min over the beam, by (cost, state) — the exact
            //     rule the CPU dedup implements.
            unsigned int       my_g   = 0u;
            unsigned long long my_key = QB_KEY_MAX;
            const bool in_beam = (tid < bn);
            if (in_beam) {
                const unsigned int st = (unsigned int)s_beam_state[tid];
                my_g   = st & G::GROUP_MASK;
                my_key = ((unsigned long long)qtip_total_order_key(s_beam_cost[tid]) << 16)
                       | (unsigned long long)st;
                atomicMin(&s_gmin[my_g], my_key);
            }
            __syncthreads();

            // 1b. Winners self-identify (keys are unique, so exactly one wins
            //     per non-empty group) and compact deterministically.
            const unsigned int win = (in_beam && s_gmin[my_g] == my_key) ? 1u : 0u;
            unsigned int n_groups = 0u;
            const unsigned int pos = qb_block_excl_scan<QB_WARPS>(win, s_warp_tot_buf + QB_WARPS * (qb_scan_slot ^= 1u), &n_groups);
            // The scan's barriers separate the s_gmin reads above from the
            // releases below.
            if (win) {
                s_grp_g[pos]      = (unsigned short)my_g;
                s_grp_cost[pos]   = s_beam_cost[tid];
                s_grp_parent[pos] = (unsigned short)tid;
                s_gmin[my_g]      = QB_KEY_MAX;   // release for the next step
            }
            if (tid == 0) s_n_groups = (int)n_groups;
            __syncthreads();
        }

        // ---- 2. Expand: 2^K successors per group, all distinct ---------------
        const int  ng     = s_n_groups;
        const bool active = (lane_g < ng);
        unsigned int base_state = 0u;
        // This thread's slice of its group's alphabet: symbols
        // `lane_i, lane_i + LANES, lane_i + 2*LANES, ...`. Striding rather than
        // blocking keeps the LANES lanes of a group reading CONSECUTIVE
        // codebook entries, so the LUT path stays one coalesced run per group.
        // At LANES == 1 the slice is `0, 1, .., 2^K-1` — the shipped order.
        //
        // wave17-AF: the candidates are carried as their ORDERED u32 KEYS, not
        // as floats. `qtip_total_order_key` is a bijection, so this loses
        // nothing, and it deletes a key rebuild (2 instructions x CAND
        // candidates) from every radix pass and from compaction — ~5 rebuilds
        // per timestep at the measured 3.87 passes. The float is recovered
        // exactly, once, for the survivors that reach the beam.
        unsigned int cand[QS::CAND];
        if (active) {
            base_state = ((unsigned int)s_grp_g[lane_g]) << G::K;
            const float gcost = s_grp_cost[lane_g];
            if (COMPUTED_CB) {
                // The successors are `base_state | sym` for the strided `sym`
                // above, and base_state's low K bits are zero, so
                // `(base_state|sym)*mult == base_state*mult + sym*mult`: the MCG
                // product advances by one folded constant per step — a single
                // integer add, and no codebook memory traffic at all.
                const unsigned int prod0 =
                    (base_state + (unsigned int)lane_i) * cb_mult;
                const unsigned int step = (unsigned int)QS::LANES * cb_mult;
                #pragma unroll
                for (int m = 0; m < QS::CAND; ++m) {
                    const unsigned int x0 = prod0 + (unsigned int)m * step;
                    const float err = qtip_cb_err_from_x0<G::V>(x0, cb_mult, tgt);
                    cand[m] = qtip_total_order_key(__fadd_rn(gcost, err));
                }
            } else {
                // Consecutive states => one contiguous LUT run per group
                // (2^K * V * 4 B), which the LANES lanes stride through.
                const float* __restrict__ lp =
                    lut + (size_t)(base_state + (unsigned int)lane_i) * (size_t)G::V;
                #pragma unroll
                for (int m = 0; m < QS::CAND; ++m) {
                    const float* __restrict__ cp =
                        lp + (size_t)m * (size_t)(QS::LANES * (int)G::V);
                    const float err = qtip_decode_err_exact_vec<G::V>(cp, tgt);
                    cand[m] = qtip_total_order_key(__fadd_rn(gcost, err));
                }
            }
        } else {
            #pragma unroll
            for (int m = 0; m < QS::CAND; ++m) cand[m] = 0u;
        }
        const int n_cand = ng * (int)G::ALPHABET;

        // ---- 3. Select the beam_w smallest by (cost, state) ------------------
        if (n_cand <= beam_w) {
            // Nothing to prune — this is the case the "unpruned beam reproduces
            // the exhaustive DP" guard exercises.
            if (tid == 0) s_threshold = QB_KEY_MAX;
            __syncthreads();
        } else {
            // wave16-AF: radix-select over the 32-bit COST key, with an exact
            // fallback into the 16 state bits only when costs actually tie.
            //
            // The previous version resolved the full 48-bit `(cost, state)`
            // composite, paying 64-bit shifts and compares on every digit.
            // `probe_beam_kernel_cost_drivers` measured what that buys: the
            // selection terminates inside the cost key in 98.4% of timesteps
            // (the pass histogram reaches the state bits 229 times in 14,328).
            // So the state half of the key is dead weight almost always, and
            // splitting it out turns the hot loop into 32-bit arithmetic
            // without changing which candidate is chosen.
            //
            // Ordering is unchanged and remains exactly `f32::total_cmp` on the
            // cost, then ascending state: `cost_prefix` is resolved first and
            // the tie pass below breaks equal cost keys by state, which is the
            // same lexicographic order the composite key encoded.
            unsigned int cost_prefix = 0u;
            int k = beam_w;
            unsigned int tie_count = 0u;   // candidates sharing the final cost key
            int exit_shift = 0;            // digit at which the scan stopped
            for (int shift = 24; shift >= 0; shift -= 8) {
                exit_shift = shift;

                #pragma unroll
                for (int m = 0; m < QS::CAND; ++m) {
                    unsigned int ck = 0u;
                    bool part = false;
                    if (active) {
                        ck = cand[m];
                        // `shift == 24` makes the guard vacuous; the compiler
                        // folds it away in that unrolled iteration.
                        part = (shift == 24) ||
                               ((ck >> (shift + 8)) == (cost_prefix >> (shift + 8)));
                    }
                    const unsigned int bin = part ? ((ck >> shift) & 0xFFu) : 0u;
                    qb_hist_inc(s_hist, part, bin);
                }
                __syncthreads();

                // Find the digit bin holding the k-th smallest of this prefix.
                qb_select_digit_bin(s_hist, k, &s_sel_bin, &s_sel_k, &s_sel_cnt);
                __syncthreads();

                cost_prefix |= s_sel_bin << shift;
                k = (int)s_sel_k;
                tie_count = s_sel_cnt;
                if (tie_count == 1u) {
                    break;   // unique cost key: the state bits cannot matter
                }
            }

            unsigned long long threshold_key;
            if (tie_count == 1u) {
                // One candidate carries this cost prefix, so it is the beam_w-th
                // smallest outright — but the scan may have stopped before
                // resolving every bit, so the unresolved low cost bits must be
                // saturated too, not just the state half. `key <= threshold`
                // then admits that candidate plus everything strictly cheaper —
                // exactly beam_w candidates.
                const unsigned int cost_hi =
                    (exit_shift > 0) ? (cost_prefix | ((1u << exit_shift) - 1u)) : cost_prefix;
                threshold_key = ((unsigned long long)cost_hi << 16) | 0xFFFFull;
            } else {
                // Genuine cost tie (1.6% of timesteps, measured). `k` of the
                // `tie_count` candidates sharing `cost_prefix` are admitted, and
                // the tie-break is ascending state — identical to what the
                // 48-bit composite did, resolved here in two 8-bit passes over
                // the 16-bit state.
                unsigned int state_prefix = 0u;
                for (int shift = 8; shift >= 0; shift -= 8) {

                    #pragma unroll
                    for (int m = 0; m < QS::CAND; ++m) {
                        bool part = false;
                        unsigned int st = 0u;
                        if (active) {
                            st = base_state
                               | (unsigned int)(lane_i + m * QS::LANES);
                            part = (cand[m] == cost_prefix) &&
                                   ((shift == 8) ||
                                    ((st >> (shift + 8)) == (state_prefix >> (shift + 8))));
                        }
                        const unsigned int bin = part ? ((st >> shift) & 0xFFu) : 0u;
                        qb_hist_inc(s_hist, part, bin);
                    }
                    __syncthreads();

                    qb_select_digit_bin(s_hist, k, &s_sel_bin, &s_sel_k, &s_sel_cnt);
                    __syncthreads();

                    state_prefix |= s_sel_bin << shift;
                    k = (int)s_sel_k;
                    if (s_sel_cnt == 1u) {
                        if (shift > 0) state_prefix |= (1u << shift) - 1u;
                        break;
                    }
                }
                threshold_key = ((unsigned long long)cost_prefix << 16)
                              | (unsigned long long)(state_prefix & 0xFFFFu);
            }
            if (tid == 0) s_threshold = threshold_key;
            __syncthreads();
        }

        // ---- 4. Compact the survivors into the beam --------------------------
        const unsigned long long threshold = s_threshold;
        unsigned int keep_mask = 0u;
        unsigned int keep_cnt  = 0u;
        if (active) {
            #pragma unroll
            for (int m = 0; m < QS::CAND; ++m) {
                const unsigned int st =
                    base_state | (unsigned int)(lane_i + m * QS::LANES);
                const unsigned long long key = ((unsigned long long)cand[m] << 16)
                    | (unsigned long long)st;
                if (key <= threshold) {
                    keep_mask |= (1u << m);
                    ++keep_cnt;
                }
            }
        }
        unsigned int kept = 0u;
        const unsigned int base_slot = qb_block_excl_scan<QB_WARPS>(keep_cnt, s_warp_tot_buf + QB_WARPS * (qb_scan_slot ^= 1u), &kept);
        // After the scan nothing reads the previous beam (the group records
        // already captured every value the expansion needed), so the beam is
        // rewritten in place.
        if (active) {
            const unsigned short parent = s_grp_parent[lane_g];
            unsigned int p = base_slot;
            #pragma unroll
            for (int m = 0; m < QS::CAND; ++m) {
                if (keep_mask & (1u << m)) {
                    // Exact inverse; the bijection makes this the identical f32.
                    s_beam_cost[p]   = qtip_key_to_float(cand[m]);
                    s_beam_state[p]  = (unsigned short)(
                        base_state | (unsigned int)(lane_i + m * QS::LANES));
                    s_beam_parent[p] = parent;
                    ++p;
                }
            }
        }
        if (tid == 0) s_beam_n = (int)kept;
        __syncthreads();

        // ---- 5. Persist this timestep's frontier for the backtrace -----------
        {
            uint32_t* __restrict__ bt = my_trace + (size_t)t * (size_t)beam_w;
            const int bn = s_beam_n;
            for (int i = tid; i < bn; i += QB_THREADS) {
                bt[i] = (((uint32_t)s_beam_state[i]) << 16) | (uint32_t)s_beam_parent[i];
            }
        }
        __syncthreads();
    }

    // ---- 6. Best final state: min (cost, state), like the CPU argmin ---------
    if (tid == 0) s_fin_key = QB_KEY_MAX;
    __syncthreads();
    unsigned long long fin = QB_KEY_MAX;
    const bool fin_active = (tid < s_beam_n);
    if (fin_active) {
        fin = ((unsigned long long)qtip_total_order_key(s_beam_cost[tid]) << 16)
            | (unsigned long long)s_beam_state[tid];
        atomicMin(&s_fin_key, fin);
    }
    __syncthreads();
    if (fin_active && fin == s_fin_key) s_fin_slot = (unsigned int)tid;
    __syncthreads();

    // ---- 7. Backtrace + pack -------------------------------------------------
    //
    // The walk is a dependent pointer chase: slot at t is only known after
    // reading the entry at t+1. Run naively by one thread against global memory
    // it costs `num_symbols` full memory latencies — on a 9472-symbol row that
    // is comparable to the entire forward pass, and it is why the exhaustive
    // kernel's tail is so expensive. Instead reuse the now-dead group table as
    // a staging buffer: the whole block bulk-loads a window of timesteps
    // (coalesced), then thread 0 chases inside shared memory. 32 KiB stages 32
    // timesteps at W=256 (K=4/V=2); the 16 KiB floor stages 16, against half as
    // many timesteps at V=4 — the same dependent-load count either way.
    {
        for (int i = tid; i < pkd_per_row; i += QB_THREADS) my_pkd[i] = 0u;

        uint32_t* s_stage = reinterpret_cast<uint32_t*>(s_gmin);
        const int stage_slots =
            (int)(QS::SCRATCH_U64 * (int)(sizeof(unsigned long long) / sizeof(uint32_t)));
        const int window = stage_slots / beam_w;   // >= 16 for beam_w <= 256

        if (tid == 0) s_walk_slot = s_fin_slot;
        __syncthreads();

        for (int t_hi = num_symbols - 1; t_hi >= 0; t_hi -= window) {
            const int t_lo = (t_hi - window + 1 > 0) ? (t_hi - window + 1) : 0;
            const size_t span = (size_t)(t_hi - t_lo + 1) * (size_t)beam_w;
            const uint32_t* __restrict__ src = my_trace + (size_t)t_lo * (size_t)beam_w;
            for (size_t i = tid; i < span; i += QB_THREADS) s_stage[i] = src[i];
            __syncthreads();

            if (tid == 0) {
                unsigned int slot = s_walk_slot;
                for (int t = t_hi; t >= t_lo; --t) {
                    const uint32_t e = s_stage[(size_t)(t - t_lo) * (size_t)beam_w + (size_t)slot];
                    const unsigned int st = e >> 16;
                    qtip_pack_symbol<G>(my_pkd, t, st);
                    slot = e & 0xFFFFu;
                }
                s_walk_slot = slot;
            }
            __syncthreads();
        }
    }
}

// Launch one geometry. Split out so the ABI-level dispatch below is a plain
// table and the codebook branch is not restated per geometry.
template <class G>
int qb_launch(
    const float*  d_weight,
    const float*  d_lut,
    const float*  d_row_scales,
    uint8_t*      d_packed,
    uint32_t*     d_trace,
    int n_rows,
    int in_features,
    int num_symbols,
    int row_offset,
    int beam_w,
    unsigned int  cb_mult,
    cudaStream_t  stream
) {
    if (beam_w < 1 || beam_w > QB_MAX_BEAM) return -1;
    if (cb_mult != 0u) {
        qtip_quantize_rows_beam_kernel<G, true><<<n_rows, QB_THREADS, 0, stream>>>(
            d_weight, d_lut, d_row_scales, d_packed, d_trace,
            in_features, num_symbols, row_offset, beam_w, cb_mult);
    } else {
        qtip_quantize_rows_beam_kernel<G, false><<<n_rows, QB_THREADS, 0, stream>>>(
            d_weight, d_lut, d_row_scales, d_packed, d_trace,
            in_features, num_symbols, row_offset, beam_w, 0u);
    }
    return 0;
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

// Largest beam width the kernel can run. The Rust side reads this and REFUSES
// a wider beam rather than silently substituting one it can do — a bake must
// never quietly change its search.
int qtip_beam_max_width() { return QB_MAX_BEAM; }

// Beam-quantize `n_rows` rows at the trellis geometry `(k_bits, v_dim, l_bits)`.
//
// The geometry is an ARGUMENT, not a compile-time assumption of the caller: the
// Rust side passes `qtip::{K, V, L}` straight through, so the day those consts
// change the bake follows without a second parameterisation to reconcile. An
// unsupported triple is REFUSED (-2) rather than silently served by the nearest
// compiled kernel — the failure mode that would produce a checkpoint whose
// bytes mean something other than what its header claims.
//
// Returns 0 on launch, -1 when `beam_w` is out of range, -2 when the geometry
// has no compiled kernel.
// `cb_mult == 0` selects the stored Gaussian LUT; nonzero selects the computed
// sum2 codebook with that MCG multiplier. See qtip_codebook.cuh.
int launch_qtip_quantize_rows_beam_geom_f32(
    const float*  d_weight,
    const float*  d_lut,
    const float*  d_row_scales,
    uint8_t*      d_packed,
    uint32_t*     d_trace,
    int n_rows,
    int in_features,
    int num_symbols,
    int row_offset,
    int beam_w,
    int k_bits,
    int v_dim,
    int l_bits,
    unsigned int  cb_mult,
    cudaStream_t  stream
) {
    if (k_bits == 4 && v_dim == 2 && l_bits == 16) {
        return qb_launch<QtipGeomK4V2L16>(
            d_weight, d_lut, d_row_scales, d_packed, d_trace,
            n_rows, in_features, num_symbols, row_offset, beam_w, cb_mult, stream);
    }
    if (k_bits == 8 && v_dim == 4 && l_bits == 12) {
        return qb_launch<QtipGeomK8V4L12>(
            d_weight, d_lut, d_row_scales, d_packed, d_trace,
            n_rows, in_features, num_symbols, row_offset, beam_w, cb_mult, stream);
    }
    // 2.25 bpw. The caller must size `d_packed` as `ceil(num_symbols*9/8)` per
    // row — `num_symbols / (8/K)` is a division by zero at this rung, which is
    // why `qtip_packed_bytes_per_row` is no longer written that way.
    if (k_bits == 9 && v_dim == 4 && l_bits == 12) {
        return qb_launch<QtipGeomK9V4L12>(
            d_weight, d_lut, d_row_scales, d_packed, d_trace,
            n_rows, in_features, num_symbols, row_offset, beam_w, cb_mult, stream);
    }
    return -2;
}

} // extern "C"
