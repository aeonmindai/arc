// Block-level primitives shared by the two beam-search trellis quantizers:
//
//   * `qtip_beam.cu`    — the LUT rung (`qtip2`,  K=4 / V=2, 16 successors)
//   * `qtip2b_beam.cu`  — the bitshift rung (`qtip2b`, K=2 / V=1, 4 successors)
//
// These were written and measured for `qtip_beam.cu` (wave16-AF / wave17-AF);
// they moved here verbatim when the qtip2b beam landed (wave46-BX) so the two
// rungs cannot drift. Nothing here knows K, V, the codebook, or the alphabet —
// the only shape assumption is a 256-thread block with a 256-bin digit
// histogram, which both rungs satisfy and both static_assert.
//
// Everything is `__device__ __forceinline__`, so including this header from
// several translation units creates no linkage issues.

#pragma once

#include <cstdint>

// Warp-aggregated shared-memory histogram increment.
//
// The high digits of the selection key are the sign+exponent bits of costs
// that are all within a factor of two of each other, so a plain
// `atomicAdd(&hist[bin], 1)` would put every lane of every warp on the SAME
// address and serialize 32-ways. `__match_any_sync` collapses each warp's
// same-bin lanes into a single atomic. Must be called warp-converged.
// (`__match_any_sync` needs SM70+; `has_qtip_kernels` is off below SM80.)
__device__ __forceinline__ void qb_hist_inc(
    unsigned int* __restrict__ hist, bool part, unsigned int bin
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    const unsigned int amask = __ballot_sync(0xFFFFFFFFu, part);
    if (part) {
        const unsigned int m = __match_any_sync(amask, bin);
        const unsigned int leader = (unsigned int)(__ffs((int)m) - 1);
        if ((threadIdx.x & 31u) == leader) {
            atomicAdd(&hist[bin], (unsigned int)__popc((int)m));
        }
    }
#else
    if (part) {
        atomicAdd(&hist[bin], 1u);
    }
#endif
}

// Block-wide exclusive prefix sum of one value per thread (exactly 32*WARPS
// threads participate). Returns this thread's exclusive prefix; `*total`
// receives the block sum. Contains its own barriers: safe to call back to back.
//
// wave16-AF: the caller passes a scratch buffer that ALTERNATES between two
// WARPS-word regions. The original had to open with a `__syncthreads()` purely
// to stop this call's writes from racing the *previous* call's reads of the
// same words. At ~6 scans per timestep that barrier was ~6 of the 33 per-step
// barriers, on a kernel measured to be latency-bound. Alternating buffers
// removes the hazard structurally — no ordering invariant for a future caller
// to violate — for 32 extra bytes of shared memory.
template <int WARPS>
__device__ __forceinline__ unsigned int qb_block_excl_scan(
    unsigned int v, unsigned int* __restrict__ s_warp_tot, unsigned int* total
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;

    unsigned int x = v;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const unsigned int y = __shfl_up_sync(0xFFFFFFFFu, x, d);
        if (lane >= (unsigned int)d) x += y;
    }
    if (lane == 31u) s_warp_tot[warp] = x;
    __syncthreads();
    if (warp == 0u) {
        unsigned int w = (lane < (unsigned int)WARPS) ? s_warp_tot[lane] : 0u;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const unsigned int y = __shfl_up_sync(0xFFFFFFFFu, w, d);
            if (lane >= (unsigned int)d) w += y;
        }
        if (lane < (unsigned int)WARPS) s_warp_tot[lane] = w;
    }
    __syncthreads();
    const unsigned int warp_excl = (warp == 0u) ? 0u : s_warp_tot[warp - 1u];
    *total = s_warp_tot[WARPS - 1];
    return warp_excl + x - v;
}

// Locate the digit bin containing the k-th smallest, and clear the histogram
// behind it — the whole scan done by ONE warp, with no block barrier inside.
//
// wave17-AF: this replaces a general block-wide exclusive scan that cost three
// `__syncthreads` on a path executed ~3.87 times per timestep. The general scan
// is the wrong tool: we do not need all 256 prefix sums, only the single bin
// where the running total crosses `k`. Warp 0 covers the 256 bins as 32 lanes ×
// 8 bins, warp-scans the 32 lane subtotals with shuffles (no barrier exists
// inside a warp), and the one lane whose range straddles `k` walks its own 8
// bins to pin the bin exactly.
//
// The caller supplies the barrier AFTER this returns; the barrier BEFORE it
// (making the histogram visible) is the caller's histogram-fill barrier. Net
// cost: 2 barriers per radix pass instead of 5.
//
// Clearing is folded in: every bin is read exactly once here, by exactly one
// lane, so zeroing it in the same pass maintains the all-zero invariant with no
// extra traffic and no extra barrier.
__device__ __forceinline__ void qb_select_digit_bin(
    unsigned int* __restrict__ s_hist,
    int k,
    unsigned int* __restrict__ s_sel_bin,
    unsigned int* __restrict__ s_sel_k,
    unsigned int* __restrict__ s_sel_cnt
) {
    if (threadIdx.x >= 32u) return;
    const unsigned int lane = threadIdx.x;
    const unsigned int base = lane * 8u;

    unsigned int c[8];
    unsigned int sub = 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        c[i] = s_hist[base + i];
        s_hist[base + i] = 0u;      // maintain the all-zero invariant
        sub += c[i];
    }

    // Inclusive scan of the 32 lane subtotals.
    unsigned int inc = sub;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const unsigned int y = __shfl_up_sync(0xFFFFFFFFu, inc, d);
        if (lane >= (unsigned int)d) inc += y;
    }
    const unsigned int excl = inc - sub;

    // Exactly one lane's 8-bin range straddles k.
    if (excl < (unsigned int)k && (unsigned int)k <= inc) {
        unsigned int run = excl;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (run < (unsigned int)k && (unsigned int)k <= run + c[i]) {
                *s_sel_bin = base + (unsigned int)i;
                *s_sel_k   = (unsigned int)k - run;
                *s_sel_cnt = c[i];
            }
            run += c[i];
        }
    }
}
