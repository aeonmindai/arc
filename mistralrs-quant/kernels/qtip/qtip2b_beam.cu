// QTIP beam-search trellis quantize kernel — the BITSHIFT rung
// (`qtip2b`, K=2 bits/symbol, V=1 weight/symbol, L=16 state bits).
//
// This is the GPU twin of `qtip/bitshift.rs::beam_quantize_row_2b`, and the
// K=2/V=1 port of `qtip_beam.cu` (the K=4/V=2 LUT rung, wave13-AF/wave16-AF/
// wave17-AF). It is BIT-IDENTICAL to the CPU beam at the same width; the parity
// tests in `qtip/bitshift.rs` compare packed bytes, not cosine similarity.
//
// WHY IT EXISTS (wave46-BX)
// -------------------------
// `qtip2b` had no beam at all: `viterbi_quantize_row_2b` and
// `qtip2b_quantize_rows_viterbi_kernel` are the exhaustive DP over all 2^16
// states, re-run once per symbol. Because V=1, a row of `K_in` weights is
// `K_in` symbols — TWICE the timestep count of the LUT rung's `K_in/2` — while
// each timestep still sweeps the whole 2^16 state space. wave41-BS measured the
// consequence on an H200: layer 0 had not finished after 984 s, projecting
// ~11.75 h (~$57) for a 43-layer bake, against 510 s/layer for the LUT rung's
// exhaustive DP. That factor of ~2 is exactly the V=1 timestep doubling.
//
// ⚠ THIS IS A DELIBERATE QUALITY TRADE, NOT A FREE WIN (DOCTRINE D4).
// Exhaustive search is the best-quality option this rung has, and wave19-AP
// measured exhaustive beating beam W=256 on 8 of 9 fixture cells for the LUT
// rung. A beam is a *legitimate* search — unlike greedy, which stays banned and
// unreachable — but it is a slightly worse one, and it is recorded as such:
// `QtipSearchDetail::Known { beam_width: Some(W) }` is stamped into the UQFF
// artifact and checked at load. The default remains exhaustive; a beam only
// happens when `ARC_QTIP_BEAM=<W>` is set explicitly.
//
// THE K=2 / V=1 GEOMETRY (what actually differs from qtip_beam.cu)
// ----------------------------------------------------------------
// The transition is `s_t = ((s_{t-1} << K) | sym) & (2^L - 1)`, so the
// successors of a beam entry `p` depend on `p` only through its low `L-K` bits.
// That is the "group" both rungs exploit. With K=4 there are `2^12 = 4096`
// groups of 16 successors; with K=2 there are **`2^14 = 16384` groups of 4**.
//
//   1. **The group table cannot be directly indexed any more.** The LUT rung
//      keeps `s_gmin[4096]` u64 = 32 KiB in shared memory. The K=2 twin would
//      be 16384 x 8 B = **128 KiB**, past even the 163 KiB opt-in dynamic limit
//      on sm_80 and far past the 48 KiB static one. But the table is only ever
//      *sparse*: at most `beam_w <= 256` groups are live at any timestep. So it
//      becomes an **open-addressed hash table of 1024 slots (12 KiB)** keyed by
//      the 14-bit group, claimed with `atomicCAS` on the tag and reduced with
//      the same 64-bit `atomicMin` on `(total_order_key(cost) << 16) | state`.
//      Net: 12 KiB instead of 32 KiB — the K=2 rung's group structure is
//      CHEAPER in shared memory than the K=4 rung's, not more expensive.
//
//      Determinism (required — this kernel must be bit-identical to the CPU):
//      tags are only ever set during a timestep and only released by the
//      winner afterwards, so the claimed-slot predicate is monotone along a
//      probe path. Every thread of a group therefore converges on the SAME
//      slot — the first slot on `hash(g), hash(g)+1, ...` not owned by another
//      group — regardless of interleaving. The winner is then identified by
//      `s_hmin[slot] == my_key` (keys are unique: distinct beam states give
//      distinct low 16 bits), and the survivors are compacted by a block scan
//      in ASCENDING TID order, which is independent of slot assignment.
//
//   2. **4 successors per group, not 16.** `cand[]` is a 4-entry register array
//      instead of 16, the histogram fill is 4 iterations instead of 16, and the
//      candidate set is `n_groups * 4 <= 1024` instead of 4096. Distinct groups
//      still produce disjoint successor sets (successor `= (g << K) | j` is a
//      bijection of `(g, j)`), so the expanded list needs no dedup and every
//      48-bit selection key is unique — the property the radix-select relies on.
//
//   3. **One weight per symbol.** The branch metric is `(c - t)^2` on a single
//      target, not `(c0-t0)^2 + (c1-t1)^2`, and the accumulation is
//      `__fadd_rn(gcost, err)` — the same two-rounding shape the CPU DP has.
//
//   4. **Computed codebook only.** There is no LUT: the codeword is
//      `f32(fp16(hi)) + f32(fp16(lo))` of the masked/XORed MCG product
//      (`qtip2b_common.cuh::q2b_decode`). Since a group's 4 successors are
//      `base | j` for consecutive `j` and `base`'s low K bits are zero, the MCG
//      product advances by one folded constant per `j`: `base*mult + j*mult`.
//      No codebook memory traffic at all.
//
//   5. **4 symbols per packed byte** (2 bits each, LSB-first) instead of 2.
//
// PARITY
// ------
// Every arithmetic operation on the trellis path goes through
// `qtip_exact_fp.cuh`, which forbids FMA contraction and approximate division
// (both of which `--use_fast_math` would otherwise introduce). The exhaustive
// K=2 kernel in `qtip_bitshift.cu` was moved onto the same intrinsics in this
// same change, which is what makes "an unpruned beam reproduces the exhaustive
// DP byte for byte" a testable claim on GPU rather than a hope.
//
// LAYOUT / LAUNCH
// ---------------
//   grid  = (rows_in_flight, 1, 1)      one block per row
//   block = (QB2_THREADS = 256, 1, 1)   thread t owns beam slot t and group t
//   trace = [rows_in_flight, num_symbols, beam_w] u32, packed
//           `(state << 16) | parent_slot`.
//
// Static shared memory is ~34.1 KiB at W=256 — under the 48 KiB static limit,
// and 4 x 34.1 KiB = 136 KiB fits the 164 KiB an sm_80 SM has, so the
// `__launch_bounds__` occupancy target below is reachable on A100/A30 as well
// as on H100/H200.
//
// SM80+ (gated by `has_qtip_kernels` in build.rs).

#include <cstdint>
#include <cuda_runtime.h>

#include "qtip_exact_fp.cuh"
#include "qtip2b_common.cuh"
#include "qtip_beam_common.cuh"

namespace {

constexpr uint32_t QB2_K           = Q2B_K;                    // 2
constexpr uint32_t QB2_ALPHABET    = Q2B_ALPHABET;             // 4
constexpr uint32_t QB2_GROUP_BITS  = Q2B_L - Q2B_K;            // 14
constexpr uint32_t QB2_GROUP_COUNT = 1u << QB2_GROUP_BITS;     // 16384
constexpr uint32_t QB2_GROUP_MASK  = QB2_GROUP_COUNT - 1u;     // 0x3FFF

// One thread per beam slot AND per group; both are bounded by the beam width,
// so the maximum supported width is the block size. 256 matches the LUT rung's
// `qtip_beam_max_width()` so an operator's `ARC_QTIP_BEAM` means the same thing
// on both rungs.
constexpr int QB2_THREADS  = 256;
constexpr int QB2_MAX_BEAM = QB2_THREADS;
constexpr int QB2_WARPS    = QB2_THREADS / 32;

// Group hash table. 1024 slots for at most 256 live groups is a load factor of
// 0.25 (~1.16 probes expected), and it can never fill — which is what makes the
// CAS claim loop terminate unconditionally.
constexpr int      QB2_HASH_SLOTS = 1024;
constexpr uint32_t QB2_HASH_MASK  = (uint32_t)QB2_HASH_SLOTS - 1u;
constexpr uint32_t QB2_TAG_EMPTY  = 0xFFFF'FFFFu;   // 14-bit groups never reach this

// Backtrace staging window, in u32 slots (16 KiB). See the walk at the bottom:
// a window must hold at least one full timestep of the trace.
constexpr int QB2_STAGE_SLOTS = 4096;

constexpr unsigned long long QB2_KEY_MAX = ~0ull;

// Fibonacci hashing of the 14-bit group onto the table. Any bijection-ish mix
// works; this one spreads the low bits (which are the symbol just shifted in,
// i.e. highly non-uniform across a beam) across the whole table.
__device__ __forceinline__ unsigned int qb2_group_hash(unsigned int g) {
    return (g * 0x9E3779B1u) >> (32 - 10);   // 10 = log2(QB2_HASH_SLOTS)
}

// Codeword from an already-formed MCG product. `q2b_decode(s, mult)` is
// `q2b_codeword_from_x(s * mult)`; splitting it lets the expansion advance the
// product by a constant per successor instead of re-multiplying. The two fp16
// halves are summed in f32 — identical to the CPU `mcg_codeword` and to every
// other decode path in this rung. Never replace with `__hadd`.
__device__ __forceinline__ float qb2_codeword_from_x(uint32_t x) {
    const uint32_t m = (x & Q2B_MASK) ^ Q2B_XOR;
    const float hi = __half2float(__ushort_as_half((unsigned short)(m >> 16)));
    const float lo = __half2float(__ushort_as_half((unsigned short)(m & 0xFFFFu)));
    return __fadd_rn(hi, lo);
}

// Branch metric: `(c - t)^2`, evaluated exactly as the Rust reference does
// (`let d = codebook[s] - target; d * d`). No FMA contraction is possible here
// even in principle, but the subtract and the multiply are pinned anyway so a
// future edit cannot introduce one.
__device__ __forceinline__ float qb2_decode_err_exact(float c, float t) {
    const float d = __fsub_rn(c, t);
    return __fmul_rn(d, d);
}

// Blocks per SM this kernel is compiled to fit.
//
// 4 caps registers at 65,536 / 4 / 256 = 64 per thread. This kernel's live set
// is strictly smaller than the LUT rung's (a 4-entry `cand[]` instead of 16, and
// no LUT pointer arithmetic), and the LUT rung reaches 4 with LOCAL:0, so 4 is
// the conservative choice here rather than an aspirational one.
//
// ⚠ THE FAILURE MODE IS SILENT. `__launch_bounds__` does not refuse to compile
// when it cannot reach the register budget — it SPILLS to local memory, and a
// spilled load inside the radix loop runs ~4 x 3.9 times per timestep, costing
// more than the occupancy it bought.
//
// REQUIRED CHECK after raising this, on the build box, no GPU needed:
//     cuobjdump -res-usage <obj> | grep -A1 qtip2b_quantize_rows_beam_kernel
// `LOCAL:` must remain 0.
//
// Related landmine: `unsigned int cand[QB2_ALPHABET]` must stay in registers,
// which requires EVERY loop indexing it to be fully unrolled. Weakening any of
// those `#pragma unroll`s sends the array to local memory on its own.
constexpr int QB2_MIN_BLOCKS_PER_SM = 4;

__global__ void __launch_bounds__(QB2_THREADS, QB2_MIN_BLOCKS_PER_SM)
qtip2b_quantize_rows_beam_kernel(
    const float*   __restrict__ weight,      // [n_rows, in_features]
    const float*   __restrict__ row_scales,  // [n_rows]
    uint8_t*       __restrict__ packed,      // [n_rows, num_symbols / 4]
    uint32_t*      __restrict__ trace,       // [BATCH, num_symbols, beam_w]
    int in_features,
    int num_symbols,
    int row_offset,
    int beam_w,
    uint32_t mult
) {
    // ~34.1 KiB at QB2_MAX_BEAM = 256; the 48 KiB static limit is the gate.
    __shared__ unsigned long long s_hmin[QB2_HASH_SLOTS];    // 8 KiB
    __shared__ unsigned int       s_htag[QB2_HASH_SLOTS];    // 4 KiB
    __shared__ uint32_t           s_stage[QB2_STAGE_SLOTS];  // 16 KiB
    __shared__ float          s_beam_cost  [QB2_MAX_BEAM];
    __shared__ unsigned short s_beam_state [QB2_MAX_BEAM];
    __shared__ unsigned short s_beam_parent[QB2_MAX_BEAM];
    __shared__ unsigned short s_grp_g      [QB2_MAX_BEAM];
    __shared__ float          s_grp_cost   [QB2_MAX_BEAM];
    __shared__ unsigned short s_grp_parent [QB2_MAX_BEAM];
    __shared__ unsigned int   s_hist[256];
    // Two scan scratch buffers, alternated by `qb2_scan_slot` (see
    // qb_block_excl_scan): consecutive scans never touch the same words, so no
    // barrier is needed to separate them.
    __shared__ unsigned int   s_warp_tot_buf[2 * QB2_WARPS];
    unsigned int qb2_scan_slot = 0u;
    __shared__ unsigned int   s_sel_bin;
    __shared__ unsigned int   s_sel_k;
    __shared__ unsigned int   s_sel_cnt;
    __shared__ unsigned long long s_threshold;
    __shared__ unsigned long long s_fin_key;
    __shared__ unsigned int   s_fin_slot;
    __shared__ unsigned int   s_walk_slot;
    __shared__ int            s_beam_n;
    __shared__ int            s_n_groups;

    static_assert(QB2_THREADS == 256, "the digit histogram is one bin per thread");
    static_assert(QB2_MAX_BEAM <= QB2_THREADS, "one thread per beam slot / group");
    // The CAS claim loop below has no bail-out: it relies on the table never
    // filling, which needs strictly more slots than distinct live groups.
    static_assert(QB2_MAX_BEAM < QB2_HASH_SLOTS, "group hash table must never fill");
    // The staged backtrace walk must fit at least one timestep of the trace.
    static_assert(QB2_STAGE_SLOTS >= QB2_MAX_BEAM,
                  "stage buffer must hold at least one timestep of the trace");

    const int local_row = blockIdx.x;
    const int row       = row_offset + local_row;
    const int tid       = (int)threadIdx.x;

    const float* __restrict__ my_row = weight + (size_t)row * (size_t)in_features;
    uint8_t*  __restrict__ my_pkd    = packed + (size_t)row * (size_t)(num_symbols / 4);
    uint32_t* __restrict__ my_trace  =
        trace + (size_t)local_row * (size_t)num_symbols * (size_t)beam_w;

    const float inv_scale = qtip_inv_scale_exact(row_scales[row]);

    // The group table is cleared once per row: every timestep releases exactly
    // the slots it claimed (one winner per non-empty group), so it re-enters
    // each timestep all-empty without a per-timestep memset.
    for (int i = tid; i < QB2_HASH_SLOTS; i += QB2_THREADS) {
        s_hmin[i] = QB2_KEY_MAX;
        s_htag[i] = QB2_TAG_EMPTY;
    }
    // The digit histogram is zeroed ONCE per row. Every radix pass clears its
    // own bin immediately after reading it (see `qb_select_digit_bin`), so "all
    // zero on entry" is an invariant the passes maintain rather than a cost
    // they re-pay.
    s_hist[tid] = 0u;
    if (tid == 0) {
        s_beam_n   = 0;
        s_n_groups = 0;
    }
    __syncthreads();

    for (int t = 0; t < num_symbols; ++t) {
        const float tt = qtip_scaled_target_exact(my_row[(size_t)t], inv_scale);

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
            //     rule the CPU dedup implements. The CPU visits the beam in
            //     ascending state order and keeps the first entry on a cost
            //     tie; all predecessors of a fixed successor share their low
            //     L-K bits, so "first-seen on ties" == "lowest predecessor
            //     state", which a 64-bit atomicMin over
            //     `(total_order_key(cost) << 16) | state` implements exactly
            //     and order-independently.
            unsigned int       my_slot = 0u;
            unsigned long long my_key  = QB2_KEY_MAX;
            const bool in_beam = (tid < bn);
            if (in_beam) {
                const unsigned int st = (unsigned int)s_beam_state[tid];
                const unsigned int g  = st & QB2_GROUP_MASK;
                my_key = ((unsigned long long)qtip_total_order_key(s_beam_cost[tid]) << 16)
                       | (unsigned long long)st;
                // Claim (or join) this group's slot. Terminates because the
                // table has strictly more slots than possible live groups, and
                // converges to the same slot for every thread of the group
                // because tags are monotone within a timestep.
                unsigned int h = qb2_group_hash(g);
                while (true) {
                    const unsigned int old = atomicCAS(&s_htag[h], QB2_TAG_EMPTY, g);
                    if (old == QB2_TAG_EMPTY || old == g) break;
                    h = (h + 1u) & QB2_HASH_MASK;
                }
                my_slot = h;
                atomicMin(&s_hmin[h], my_key);
            }
            __syncthreads();

            // 1b. Winners self-identify (keys are unique, so exactly one wins
            //     per non-empty group) and compact deterministically by tid.
            const unsigned int win = (in_beam && s_hmin[my_slot] == my_key) ? 1u : 0u;
            unsigned int n_groups = 0u;
            const unsigned int pos = qb_block_excl_scan<QB2_WARPS>(
                win, s_warp_tot_buf + QB2_WARPS * (qb2_scan_slot ^= 1u), &n_groups);
            // The scan's barriers separate the s_hmin reads above from the
            // releases below.
            if (win) {
                s_grp_g[pos]      = (unsigned short)((unsigned int)s_beam_state[tid] & QB2_GROUP_MASK);
                s_grp_cost[pos]   = s_beam_cost[tid];
                s_grp_parent[pos] = (unsigned short)tid;
                s_hmin[my_slot]   = QB2_KEY_MAX;      // release for the next step
                s_htag[my_slot]   = QB2_TAG_EMPTY;
            }
            if (tid == 0) s_n_groups = (int)n_groups;
            __syncthreads();
        }

        // ---- 2. Expand: 4 successors per group, all distinct -----------------
        const int  ng     = s_n_groups;
        const bool active = (tid < ng);
        unsigned int base_state = 0u;
        // The candidates are carried as their ORDERED u32 KEYS, not as floats.
        // `qtip_total_order_key` is a bijection, so this loses nothing, and it
        // deletes a key rebuild from every radix pass and from compaction. The
        // float is recovered exactly, once, for the survivors that reach the
        // beam.
        unsigned int cand[QB2_ALPHABET];
        if (active) {
            // `g < 2^14` so `g << 2 < 2^16`: no masking needed.
            base_state = ((unsigned int)s_grp_g[tid]) << QB2_K;
            const float gcost = s_grp_cost[tid];
            // The 4 successors are `base_state | j` for consecutive j, and
            // base_state's low K bits are zero, so `(base_state|j)*mult ==
            // base_state*mult + j*mult`: the MCG product advances by one folded
            // constant per j — a single integer add, no codebook traffic.
            const unsigned int prod0 = base_state * mult;
            #pragma unroll
            for (int j = 0; j < (int)QB2_ALPHABET; ++j) {
                const float c   = qb2_codeword_from_x(prod0 + (unsigned int)j * mult);
                const float err = qb2_decode_err_exact(c, tt);
                cand[j] = qtip_total_order_key(__fadd_rn(gcost, err));
            }
        } else {
            #pragma unroll
            for (int j = 0; j < (int)QB2_ALPHABET; ++j) cand[j] = 0u;
        }
        const int n_cand = ng * (int)QB2_ALPHABET;

        // ---- 3. Select the beam_w smallest by (cost, state) ------------------
        if (n_cand <= beam_w) {
            // Nothing to prune — this is the case the "unpruned beam reproduces
            // the exhaustive DP" guard exercises.
            if (tid == 0) s_threshold = QB2_KEY_MAX;
            __syncthreads();
        } else {
            // Radix-select over the 32-bit COST key, with an exact fallback into
            // the 16 state bits only when costs actually tie. Ordering is
            // exactly `f32::total_cmp` on the cost, then ascending state — the
            // same lexicographic order a 48-bit composite key would encode, but
            // resolved in 32-bit arithmetic on the overwhelmingly common path.
            unsigned int cost_prefix = 0u;
            int k = beam_w;
            unsigned int tie_count = 0u;   // candidates sharing the final cost key
            int exit_shift = 0;            // digit at which the scan stopped
            for (int shift = 24; shift >= 0; shift -= 8) {
                exit_shift = shift;

                #pragma unroll
                for (int j = 0; j < (int)QB2_ALPHABET; ++j) {
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
                // Genuine cost tie. `k` of the `tie_count` candidates sharing
                // `cost_prefix` are admitted, and the tie-break is ascending
                // state — resolved here in two 8-bit passes over the 16-bit
                // state.
                unsigned int state_prefix = 0u;
                for (int shift = 8; shift >= 0; shift -= 8) {

                    #pragma unroll
                    for (int j = 0; j < (int)QB2_ALPHABET; ++j) {
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
            for (int j = 0; j < (int)QB2_ALPHABET; ++j) {
                const unsigned long long key = ((unsigned long long)cand[j] << 16)
                    | (unsigned long long)(base_state | (unsigned int)j);
                if (key <= threshold) {
                    keep_mask |= (1u << j);
                    ++keep_cnt;
                }
            }
        }
        unsigned int kept = 0u;
        const unsigned int base_slot = qb_block_excl_scan<QB2_WARPS>(
            keep_cnt, s_warp_tot_buf + QB2_WARPS * (qb2_scan_slot ^= 1u), &kept);
        // After the scan nothing reads the previous beam (the group records
        // already captured every value the expansion needed), so the beam is
        // rewritten in place.
        if (active) {
            const unsigned short parent = s_grp_parent[tid];
            unsigned int p = base_slot;
            #pragma unroll
            for (int j = 0; j < (int)QB2_ALPHABET; ++j) {
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
            for (int i = tid; i < bn; i += QB2_THREADS) {
                bt[i] = (((uint32_t)s_beam_state[i]) << 16) | (uint32_t)s_beam_parent[i];
            }
        }
        __syncthreads();
    }

    // ---- 6. Best final state: min (cost, state), like the CPU argmin ---------
    if (tid == 0) s_fin_key = QB2_KEY_MAX;
    __syncthreads();
    unsigned long long fin = QB2_KEY_MAX;
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
    // The walk is a dependent pointer chase: the slot at t is only known after
    // reading the entry at t+1. Run naively by one thread against global memory
    // it costs `num_symbols` full memory latencies. Instead the whole block
    // bulk-loads a window of timesteps into `s_stage` (coalesced), then thread 0
    // chases inside shared memory — 16 timesteps per global round trip at
    // W=256.
    {
        const int ppr = num_symbols / 4;
        for (int i = tid; i < ppr; i += QB2_THREADS) my_pkd[i] = 0u;

        const int window = QB2_STAGE_SLOTS / beam_w;   // >= 16 for beam_w <= 256

        if (tid == 0) s_walk_slot = s_fin_slot;
        __syncthreads();

        for (int t_hi = num_symbols - 1; t_hi >= 0; t_hi -= window) {
            const int t_lo = (t_hi - window + 1 > 0) ? (t_hi - window + 1) : 0;
            const size_t span = (size_t)(t_hi - t_lo + 1) * (size_t)beam_w;
            const uint32_t* __restrict__ src = my_trace + (size_t)t_lo * (size_t)beam_w;
            for (size_t i = tid; i < span; i += QB2_THREADS) s_stage[i] = src[i];
            __syncthreads();

            if (tid == 0) {
                unsigned int slot = s_walk_slot;
                for (int t = t_hi; t >= t_lo; --t) {
                    const uint32_t e = s_stage[(size_t)(t - t_lo) * (size_t)beam_w + (size_t)slot];
                    const unsigned int st  = e >> 16;
                    const uint8_t      sym = (uint8_t)(st & (QB2_ALPHABET - 1u));
                    my_pkd[t >> 2] |= (uint8_t)(sym << ((t & 3) * 2));
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
// never quietly change its search (DOCTRINE D4b).
int qtip2b_beam_max_width() { return QB2_MAX_BEAM; }

// Returns 0 on launch, -1 when `beam_w` is out of range (caller must not
// silently fall back to a different search).
int launch_qtip2b_quantize_rows_beam_f32(
    const float*  d_weight,
    const float*  d_row_scales,
    uint8_t*      d_packed,
    uint32_t*     d_trace,
    int n_rows,
    int in_features,
    int num_symbols,
    int row_offset,
    int beam_w,
    uint32_t      mult,
    cudaStream_t  stream
) {
    if (beam_w < 1 || beam_w > QB2_MAX_BEAM) return -1;
    qtip2b_quantize_rows_beam_kernel<<<n_rows, QB2_THREADS, 0, stream>>>(
        d_weight, d_row_scales, d_packed, d_trace,
        in_features, num_symbols, row_offset, beam_w, mult);
    return 0;
}

} // extern "C"
