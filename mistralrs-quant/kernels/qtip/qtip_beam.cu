// QTIP beam-search trellis quantize kernel (LUT rung, K=4 / V=2 / L=16).
//
// This is the GPU twin of `qtip/viterbi.rs::beam_quantize_row` (wave13-AD,
// PR #29). It replaces the exhaustive prefix-grouped DP in `qtip_quantize.cu`
// with a *pruned* dynamic program that keeps only the best `beam_w` states per
// timestep, and it is BIT-IDENTICAL to the CPU beam at the same width — the
// parity test in `qtip/mod.rs` compares packed bytes, not cosine similarity.
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
//     read  LUT         n_groups * 128 B <= 32768 B      (L2-resident)
//     read  weight      8 B
// The 512 KB cost ping-pong disappears entirely: the live state set is
// `beam_w` entries, which fit in shared memory with room to spare. HBM traffic
// per symbol position drops ~512x.
//
// SHARED-MEMORY RESIDENCY (the claim wave13-AF was asked to test)
// --------------------------------------------------------------
// At beam_w = 256 the *beam* is 2 KiB (cost f32 + state u16 + parent u16).
// It is not the beam that sets the shared-memory budget — it is the
// 2^(L-K) = 4096-entry group-reduction table below, at 8 B/prefix = 32 KiB.
// Total static shared memory is ~37.1 KiB at W=256, ~35.6 KiB at W=128 and
// ~34.8 KiB at W=64: all three are comfortably inside the 48 KiB static
// budget, all three keep costs and backtrace-of-the-live-set off HBM, and the
// group table (not the beam) is what would have to shrink to go further.
//
// ALGORITHM (mirrors the Rust beam exactly)
// -----------------------------------------
// Successor states of a beam entry with state `p` are
// `((p << K) | sym) & STATE_MASK`, which depends on `p` only through
// `g = p & ((1 << (L-K)) - 1)` — the same prefix-grouping the exhaustive
// kernel exploits, applied to the beam instead of the whole state space.
// Consequences:
//   * All 16 successors of a group share one predecessor set, so the CPU's
//     "dedup by successor state, keep min predecessor cost, first-seen wins
//     ties" collapses to ONE min per group. The CPU visits the beam in
//     ascending state order, and all predecessors of a fixed successor share
//     their low L-K bits, so "first-seen on ties" == "lowest predecessor
//     state". A 64-bit `atomicMin` over `(total_order_key(cost) << 16) | state`
//     implements that rule exactly and is order-independent.
//   * Distinct groups produce disjoint successor sets, so the expanded
//     candidate list needs no deduplication at all: it is exactly
//     `n_groups * 16` entries with pairwise-distinct states — hence pairwise
//     distinct 48-bit selection keys.
//
// The CPU prunes with `select_nth_unstable_by((cost, state))` + `truncate`,
// i.e. "the `beam_w` smallest by `(f32::total_cmp, state)`". Because the keys
// are unique, that set is found here by a radix-select on the 48-bit key
// (6 x 8-bit digit passes with an early exit as soon as a digit bin holds a
// single candidate, which is the common case) followed by a deterministic
// prefix-sum compaction. No sort is needed: nothing downstream depends on the
// beam's order — the group reduction is an atomicMin, the final choice is an
// argmin, and the backtrace follows explicit parent indices.
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
//   block = (QB_THREADS = 256, 1, 1)    thread t owns beam slot t and group t
//   trace = [rows_in_flight, num_symbols, beam_w] u32, packed
//           `(state << 16) | parent_slot`. 9.7 MB per row at W=256 and
//           num_symbols=9472, against 620 MB for a per-state backtrace.
//
// SM80+ (gated by `has_qtip_kernels` in build.rs; uses `__match_any_sync`,
// which needs SM70+).

#include <cstdint>
#include <cuda_runtime.h>

#include "qtip_exact_fp.cuh"
// Codebook selection (stored Gaussian LUT vs in-register sum2 code).
#include "qtip_codebook.cuh"
// Block primitives shared with the qtip2b beam (histogram / scan / radix bin).
#include "qtip_beam_common.cuh"

namespace {

constexpr uint32_t QB_K           = 4;
constexpr uint32_t QB_L           = 16;
constexpr uint32_t QB_ALPHABET    = 1u << QB_K;              // 16
constexpr uint32_t QB_GROUP_BITS  = QB_L - QB_K;             // 12
constexpr uint32_t QB_GROUP_COUNT = 1u << QB_GROUP_BITS;     // 4096
constexpr uint32_t QB_GROUP_MASK  = QB_GROUP_COUNT - 1u;     // 0x0FFF

// One thread per beam slot AND per group; both are bounded by the beam width,
// so the maximum supported width is the block size. 256 keeps the whole kernel
// inside the 48 KiB static shared-memory budget (see the header) and matches
// the W=256 that PR #29 measured quality-neutral against exhaustive Viterbi.
constexpr int QB_THREADS  = 256;
constexpr int QB_MAX_BEAM = QB_THREADS;
constexpr int QB_WARPS    = QB_THREADS / 32;

constexpr unsigned long long QB_KEY_MAX = ~0ull;

// Blocks per SM this kernel is compiled to fit.
//
// MEASURED, not assumed: `cuobjdump -res-usage` on the pre-wave16 kernel
// reported `REG:80 STACK:0 SHARED:38992 LOCAL:0` for sm_90a. 256 threads x 80
// registers = 20,480 per block, and 65,536 / 20,480 = 3 blocks/SM — 24 of 64
// warps, 37.5% occupancy, **register-limited** (shared memory would allow 5:
// 3 x 38,992 B is 114 KiB of the 228 KiB an SM has). The bake drew 261 W of
// 700 W = 37% of TDP against that 37.5% occupancy, which is an independent
// check on the kernel being latency-bound rather than throughput-bound.
//
// Raising this to 4 caps registers at 65,536 / 4 / 256 = 64 per thread and buys
// 32 warps instead of 24 (+33% occupancy, which for a latency-bound kernel is a
// ~1.33x throughput term). Shared memory still fits: 4 x ~39 KiB = 156 KiB.
//
// ⚠ THE FAILURE MODE IS SILENT. `__launch_bounds__` does not refuse to compile
// when it cannot reach the register budget — it SPILLS to local memory, and a
// spilled load inside the radix loop executes 16 x ~3.87 times per timestep,
// which would cost more than the occupancy gains. The 32-bit selection above is
// what is expected to free the 16 registers (it removes the 64-bit key
// temporaries that were live across the whole radix), but register allocation
// is nvcc's scheduling decision and CANNOT be proven from source.
//
// REQUIRED CHECK after any change here, on the build box, no GPU needed:
//     cuobjdump -res-usage <obj> | grep -A1 beam_kernel
// `LOCAL:` must remain 0. If it is not, set QB_MIN_BLOCKS_PER_SM back to 3
// (which reproduces today's measured allocation exactly and is a no-op) rather
// than trading a real latency regression for a nominal occupancy gain.
//
// Related landmine: `float cand[QB_ALPHABET]` must stay in registers, which
// requires EVERY loop indexing it to be fully unrolled. Weakening any of those
// `#pragma unroll`s sends the array to local memory on its own.
constexpr int QB_MIN_BLOCKS_PER_SM = 4;

template <bool COMPUTED_CB>
__global__ void __launch_bounds__(QB_THREADS, QB_MIN_BLOCKS_PER_SM)
qtip_quantize_rows_beam_kernel(
    const float*   __restrict__ weight,      // [n_rows, in_features]
    const float*   __restrict__ lut,         // [2^L * V] (unused when computed)
    const float*   __restrict__ row_scales,  // [n_rows]
    uint8_t*       __restrict__ packed,      // [n_rows, num_symbols / 2]
    uint32_t*      __restrict__ trace,       // [BATCH, num_symbols, beam_w]
    int in_features,
    int num_symbols,
    int row_offset,
    int beam_w,
    unsigned int cb_mult
) {
    // ~37.1 KiB at QB_MAX_BEAM = 256; the 48 KiB static limit is the gate.
    __shared__ unsigned long long s_gmin[QB_GROUP_COUNT];   // 32 KiB
    __shared__ float          s_beam_cost  [QB_MAX_BEAM];
    __shared__ unsigned short s_beam_state [QB_MAX_BEAM];
    __shared__ unsigned short s_beam_parent[QB_MAX_BEAM];
    __shared__ unsigned short s_grp_g      [QB_MAX_BEAM];
    __shared__ float          s_grp_cost   [QB_MAX_BEAM];
    __shared__ unsigned short s_grp_parent [QB_MAX_BEAM];
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

    static_assert(QB_THREADS == 256, "the digit histogram is one bin per thread");
    static_assert(QB_MAX_BEAM <= QB_THREADS, "one thread per beam slot / group");
    // The backtrace stages trace windows in the (dead) group table; a window
    // must hold at least one full timestep or the staged walk cannot advance.
    static_assert(QB_MAX_BEAM <= QB_GROUP_COUNT * 2,
                  "group table must stage at least one timestep of the trace");

    const int local_row = blockIdx.x;
    const int row       = row_offset + local_row;
    const int tid       = (int)threadIdx.x;

    const float* __restrict__ my_row = weight + (size_t)row * (size_t)in_features;
    uint8_t*  __restrict__ my_pkd    = packed + (size_t)row * (size_t)(num_symbols / 2);
    uint32_t* __restrict__ my_trace  =
        trace + (size_t)local_row * (size_t)num_symbols * (size_t)beam_w;

    const float inv_scale = qtip_inv_scale_exact(row_scales[row]);

    // The group table is cleared once per row: every timestep releases exactly
    // the slots it claimed (one winner per non-empty group), so it re-enters
    // each timestep all-empty without a 4096-entry memset.
    for (int i = tid; i < (int)QB_GROUP_COUNT; i += QB_THREADS) {
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
        const float t0 = qtip_scaled_target_exact(my_row[(size_t)t * 2 + 0], inv_scale);
        const float t1 = qtip_scaled_target_exact(my_row[(size_t)t * 2 + 1], inv_scale);

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
                my_g   = st & QB_GROUP_MASK;
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

        // ---- 2. Expand: 16 successors per group, all distinct ----------------
        const int  ng     = s_n_groups;
        const bool active = (tid < ng);
        unsigned int base_state = 0u;
        // wave17-AF: the candidates are carried as their ORDERED u32 KEYS, not
        // as floats. `qtip_total_order_key` is a bijection, so this loses
        // nothing, and it deletes a key rebuild (2 instructions x 16
        // candidates) from every radix pass and from compaction — ~5 rebuilds
        // per timestep at the measured 3.87 passes. The float is recovered
        // exactly, once, for the survivors that reach the beam.
        unsigned int cand[QB_ALPHABET];
        if (active) {
            base_state = ((unsigned int)s_grp_g[tid]) << QB_K;
            const float gcost = s_grp_cost[tid];
            if (COMPUTED_CB) {
                // The 16 successors are `base_state | j` for consecutive j, and
                // base_state's low K bits are zero, so `(base_state|j)*mult ==
                // base_state*mult + j*mult`: the MCG product advances by one
                // folded constant per j — a single integer add, and no codebook
                // memory traffic at all.
                const unsigned int prod0 = base_state * cb_mult;
                #pragma unroll
                for (int j = 0; j < (int)QB_ALPHABET; ++j) {
                    const unsigned int x0 = prod0 + (unsigned int)j * cb_mult;
                    const float2 c = qtip_cb_pair_from_x0(x0, cb_mult);
                    const float err = qtip_decode_err_exact_lv(c.x, c.y, t0, t1);
                    cand[j] = qtip_total_order_key(__fadd_rn(gcost, err));
                }
            } else {
                // 16 consecutive states => one contiguous 128 B LUT run.
                const float* __restrict__ lp = lut + (size_t)base_state * 2u;
                #pragma unroll
                for (int j = 0; j < (int)QB_ALPHABET; ++j) {
                    const float err = qtip_decode_err_exact_lv(lp[2 * j + 0], lp[2 * j + 1], t0, t1);
                    cand[j] = qtip_total_order_key(__fadd_rn(gcost, err));
                }
            }
        } else {
            #pragma unroll
            for (int j = 0; j < (int)QB_ALPHABET; ++j) cand[j] = 0u;
        }
        const int n_cand = ng * (int)QB_ALPHABET;

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
                for (int j = 0; j < (int)QB_ALPHABET; ++j) {
                    unsigned int ck = 0u;
                    bool part = false;
                    if (active) {
                        ck = cand[j];
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
                    for (int j = 0; j < (int)QB_ALPHABET; ++j) {
                        bool part = false;
                        unsigned int st = 0u;
                        if (active) {
                            st = base_state | (unsigned int)j;
                            part = (cand[j] == cost_prefix) &&
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
            for (int j = 0; j < (int)QB_ALPHABET; ++j) {
                const unsigned long long key = ((unsigned long long)cand[j] << 16)
                    | (unsigned long long)(base_state | (unsigned int)j);
                if (key <= threshold) {
                    keep_mask |= (1u << j);
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
            const unsigned short parent = s_grp_parent[tid];
            unsigned int p = base_slot;
            #pragma unroll
            for (int j = 0; j < (int)QB_ALPHABET; ++j) {
                if (keep_mask & (1u << j)) {
                    // Exact inverse; the bijection makes this the identical f32.
                    s_beam_cost[p]   = qtip_key_to_float(cand[j]);
                    s_beam_state[p]  = (unsigned short)(base_state | (unsigned int)j);
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
    // timesteps at W=256, cutting the dependent global loads by that factor.
    {
        const int ppr = num_symbols / 2;
        for (int i = tid; i < ppr; i += QB_THREADS) my_pkd[i] = 0u;

        uint32_t* s_stage = reinterpret_cast<uint32_t*>(s_gmin);
        const int stage_slots = (int)(QB_GROUP_COUNT * (sizeof(unsigned long long) / sizeof(uint32_t)));
        const int window = stage_slots / beam_w;   // >= 32 for beam_w <= 256

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
                    const unsigned int st  = e >> 16;
                    const uint8_t      sym = (uint8_t)(st & (QB_ALPHABET - 1u));
                    if ((t & 1) == 0) my_pkd[t / 2] |= sym;
                    else              my_pkd[t / 2] |= (uint8_t)(sym << 4);
                    slot = e & 0xFFFFu;
                }
                s_walk_slot = slot;
            }
            __syncthreads();
        }
    }
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

// Returns 0 on launch, -1 when `beam_w` is out of range (caller must not
// silently fall back to a different search).
// `cb_mult == 0` selects the stored Gaussian LUT; nonzero selects the computed
// sum2 codebook with that MCG multiplier. See qtip_codebook.cuh.
int launch_qtip_quantize_rows_beam_f32(
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
        qtip_quantize_rows_beam_kernel<true><<<n_rows, QB_THREADS, 0, stream>>>(
            d_weight, d_lut, d_row_scales, d_packed, d_trace,
            in_features, num_symbols, row_offset, beam_w, cb_mult);
    } else {
        qtip_quantize_rows_beam_kernel<false><<<n_rows, QB_THREADS, 0, stream>>>(
            d_weight, d_lut, d_row_scales, d_packed, d_trace,
            in_features, num_symbols, row_offset, beam_w, 0u);
    }
    return 0;
}

} // extern "C"
