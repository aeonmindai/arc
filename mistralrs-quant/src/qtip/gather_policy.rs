//! Dispatch policy for the fused on-device MoE gather.
//!
//! `QuantMethod::gather_forward` on a stacked-expert QTIP layer picks between
//! a **fused on-device** path (one kernel, routing read on the GPU, no host
//! sync) and a **fallback**. Which fallback depends on the rung, and the two
//! are not remotely equivalent:
//!
//! | rung | over-cap path | what it does |
//! |---|---|---|
//! | LUT (`QtipLayer`, `--isq qtip2`) | `lut_grouped_gemm_cuda`, else `gather_forward_cuda` | the grouped GEMM once its m-tiles amortize (§4); otherwise D2H-syncs the router, dequantizes **every distinct expert** to BF16 in HBM, holds them all live, then one `index_select`+`matmul`+`index_add` per expert |
//! | bitshift (`Qtip2bLayer`, `--isq qtip2b`) | `grouped_gemm_2b_cuda` | sorts pairs by expert on-device and runs the persistent tensor-core tile loop — the *amortizing* kernel |
//!
//! **Both rungs now have a grouped GEMM, and both gate it on tile occupancy**
//! ([`grouped_gemm_tiles_amortize`], §4) rather than on a token count — a raw
//! count switched it on at ~9% tile fill and cost 5.7× end to end.
//!
//! So a single shared cap of 8 tokens was wrong for exactly one of them. The
//! bitshift rung's cap is a genuine crossover between two good kernels. The
//! LUT rung's cap is the boundary past which decode leaves a fused kernel for
//! a host-orchestrated dequantize-materialize loop whose cost grows with the
//! number of *distinct* experts — i.e. with batch. That is the measured
//! inversion (wave26-AX: aggregate 10.31 tok/s at 8 tokens/step, 5.07 at 13).
//!
//! # 1. The structural limit — what the kernel can actually take
//!
//! `qtip_gather_gemv_warp_kernel` (`kernels/qtip/qtip_gather_gemv.cu:253`) and
//! its bitshift sibling (`kernels/qtip/qtip_bitshift.cu:120`) both launch
//!
//! ```text
//! grid  = (ceil(n_rows / (WARPS_PER_BLOCK * ROWS_PER_WARP)), n_pairs, 1)
//! block = (WARPS_PER_BLOCK * 32, 1, 1)
//! smem  = ROWS_PER_BLOCK * packed_per_row   (staged iff <= 48 KiB)
//! ```
//!
//! Everything that could bound the token count:
//!
//! * **Shared memory** — `ROWS_PER_BLOCK * packed_per_row` bytes, a function of
//!   the *weight* shape only. `n_pairs` does not appear. When it exceeds 48 KiB
//!   the launcher passes `stage_packed = 0` and the kernel reads packed bytes
//!   from global instead; it degrades, it does not fail. **Not a token bound.**
//! * **Registers** — per thread: `ROWS_PER_WARP = 2` accumulators, 2 row
//!   pointers, 2 scales, `GROUP = 4` `float2` staged activations, plus the
//!   trellis state. All compile-time sized, none a function of `n_pairs`.
//!   `__launch_bounds__(256)` caps the allocation regardless. **Not a token
//!   bound.**
//! * **Occupancy** — more pairs means *more* blocks, i.e. strictly better
//!   occupancy, never worse. **Not a token bound.**
//! * **`grid.y`** — this is the real one. CUDA's maximum y-dimension of a grid
//!   of thread blocks is **65535** for every compute capability ≥ 2.0 (CUDA C
//!   Programming Guide, "Technical Specifications per Compute Capability").
//!
//! ⇒ the fused kernel is structurally safe up to
//! `n_pairs = n_tokens * n_experts_per_tok <= 65535`, i.e. **8191 tokens** at
//! top-8 routing and **10922** at top-6. The default cap of 8 was three orders
//! of magnitude below anything the kernel could not do.
//!
//! **Exceeding `grid.y` does not crash — it silently returns zeros.** The
//! launchers are `extern "C"` and discard the `<<<>>>` launch status, and the
//! Rust wrappers hand back a `dev.alloc_zeros(..)` buffer. A too-large launch
//! fails with `cudaErrorInvalidConfiguration`, nothing is written, and the MoE
//! output is all-zero. [`check_gather_gemv_pairs`] converts that into an error
//! before the launch.
//!
//! # 2. Where the LUT rung's fused path stops being the better one
//!
//! Per *distinct* expert, in units of that expert's packed 2-bit bytes
//! (`n_rows * in_features / 4`):
//!
//! ```text
//! fused    : 1 packed read per (token, slot) PAIR      -> n_pairs units
//! fallback : dequantize writes BF16 (n*k*2 bytes)
//!            + cuBLAS reads it back (n*k*2 bytes)      -> 16 units per DISTINCT expert
//!            (4*n*k) / (n*k/4) = 16
//! ```
//!
//! Distinct experts woken by `n` tokens under top-`k` of `E` routing is
//! [`expected_distinct_experts`]: `E * (1 - (1 - k/E)^n)`. So the fused path
//! moves less memory exactly while
//!
//! ```text
//! n_tokens * k  <=  16 * E * (1 - (1 - k/E)^n_tokens)
//! ```
//!
//! For `E = 256, k = 8` the left side is linear and the right saturates at
//! 4096, so there is a single crossover, at **~512 tokens**; at `k = 6` it is
//! **~683**. [`lut_fused_gather_preferred`] evaluates this per call — it is a
//! pure function of shapes, so it costs no device sync.
//!
//! **What this model deliberately does not claim.** Two effects it omits push
//! in opposite directions and neither is measured:
//! * *Favouring the fallback:* its 16.8 MB BF16 expert tensor may stay resident
//!   in L2 between the dequantize and the GEMM, so its real HBM traffic can be
//!   closer to 1 unit than 16.
//! * *Favouring the fused path:* the fallback pays a D2H router sync plus
//!   `O(distinct_experts)` kernel launches and large allocations **per MoE
//!   call** (~120 calls/step on V4), while the fused path is one launch. This
//!   term grows with batch and is the most likely explanation for the measured
//!   2× per-sequence cost jump at 13 tokens.
//! * Also unmodelled: the fused GEMV is trellis-decode-serialisation-bound and
//!   reaches only ~9–15% of peak bandwidth (FACTS, wave16-AF), while the
//!   fallback streams. Folding that in alone would move the crossover down to
//!   ~45–60 tokens.
//!
//! The traffic model above is the only one of the four with no free parameter,
//! which is why it is the one in the code. The GPU measurement that settles it
//! is a single B-sweep with `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS` pinned — see
//! `memory/mission/wave28-AZ-moe-token-cap.md`.
//!
//! # 3. What raising the cap does not buy
//!
//! The fused path issues **one GEMV per (token, expert) pair with no dedup**,
//! so its cost is linear in tokens and it never realises the `E(B)`
//! amortisation the fleet thesis needs (8B/E(B) = 4.07× at B=128). Raising the
//! cap replaces a fallback that degrades with batch with a path that is merely
//! *flat* per token. The amortising kernel is the grouped GEMM.
//!
//! # 4. When the grouped GEMM starts paying — **tile occupancy, not tokens**
//!
//! The grouped GEMM sorts the pairs by expert and walks
//! `sum_e ceil(count_e / GROUPED_TILE_M)` m-tiles, staging that expert's packed
//! bytes **once per m-tile**. Its entire advantage over the per-pair GEMV is
//! therefore the average *tile fill*:
//!
//! ```text
//! weight bytes(grouped) / weight bytes(GEMV) = 1 / fill
//! fill = (pairs per WOKEN expert) / GROUPED_TILE_M
//! pairs per woken expert = n*k / (E * (1 - (1 - k/E)^n))
//! ```
//!
//! `n_tokens` alone does not determine fill — `top_k` and `num_experts` are in
//! it too. At V4's `E = 256, k = 6` ([`expected_pairs_per_distinct_expert`]):
//!
//! ```text
//!   n   pairs   woken experts   pairs/expert   tile fill
//!   32    192          136.1          1.41        8.8%
//!  128    768          243.7          3.15       19.7%
//!  512   3072          256.0         12.00       75.0%
//!  683   4098          256.0         16.01      100.0%
//! 1024   6144          256.0         24.00      150.0%
//! ```
//!
//! and the in-tree end-to-end A/B brackets exactly that curve: **1.00× at
//! N=128 and N=512, 2.41× at N=1024**.
//!
//! **This is a measured regression, not a hypothetical.** Shipping the gate as
//! a raw `n_tokens > DECODE_REGIME_MAX_TOKENS` switched the grouped kernel on
//! at n=9, i.e. onto a tile that is ~9% full, and cost 5.7× end to end:
//! aggregate **34.19 tok/s at B=8 → 5.99 tok/s at B=32** on an exclusive H200
//! (nsys, `97a65d643`; the kernel itself ran 7,888 µs/call, 133× above its own
//! bandwidth bound, at 3.5% memory-controller utilisation and 48.2%
//! unallocated warps — saturated doing nothing). The kernel was fine; it was
//! switched on roughly two orders of magnitude too early.
//!
//! So the gate is [`grouped_gemm_tiles_amortize`]: **switch only once each
//! woken expert draws at least `GROUPED_TILE_M` pairs**, i.e. once an average
//! m-tile is full. That is the property the kernel actually needs, and it
//! tracks `top_k`, `num_experts` and `GROUPED_TILE_M` automatically — a
//! hardcoded token count rots the moment any of the three moves.
//!
//! Uniform routing is the **conservative** assumption here: a real router is
//! skewed, which wakes *fewer* distinct experts and so fills tiles *more* than
//! this predicts. The gate therefore errs toward switching late, never early.
//!
//! Note the two boundaries in this module currently coincide — the LUT rung's
//! dequantize fallback is 16× traffic and `GROUPED_TILE_M` is 16 pairs, so both
//! flip at `pairs per woken expert == 16` (~683 tokens at V4's routing). They
//! are independent constants and `grouped_boundary_is_the_fused_gemv_boundary`
//! pins the coincidence so a change to either is a loud test failure rather
//! than a silent dispatch change.

use candle_core::Result;

/// CUDA's maximum y-dimension of a grid of thread blocks. 65535 on every
/// compute capability ≥ 2.0 (CUDA C Programming Guide, Table "Technical
/// Specifications per Compute Capability" — the value has never changed).
pub(crate) const CUDA_MAX_GRID_DIM_Y: usize = 65_535;

/// Structural ceiling on the fused on-device gather: both gather-GEMV kernels
/// map one `(token, slot)` pair to one `grid.y` index.
///
/// Public so a benchmark can refuse to report a GEMV row it could not have
/// measured: past this the launch is rejected outright.
pub const GATHER_GEMV_MAX_PAIRS: usize = CUDA_MAX_GRID_DIM_Y;

/// Env override for the fused-gather token cap (both rungs). Absolute token
/// count; still clamped by [`GATHER_GEMV_MAX_PAIRS`].
///
/// ⚠️ **Read once per process** by [`ondevice_max_tokens_override`]. A caller
/// that flips this variable mid-process to switch MoE kernels is a no-op after
/// the first `gather_forward` — set it before the first call, and switch paths
/// with `ARC_NO_QTIP_ONDEVICE_MOE` / `ARC_NO_QTIP_GROUPED_MOE`, which are read
/// per call. Public so harnesses use this name rather than a string literal.
pub const ONDEVICE_MOE_MAX_TOKENS_ENV: &str = "ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS";

/// HBM traffic of the LUT rung's dequantize-materialize fallback per distinct
/// expert, in units of that expert's packed 2-bit bytes: a BF16 write plus a
/// BF16 read is `4 * n * k` bytes against `n * k / 4` packed bytes.
const DEQUANT_TRAFFIC_RATIO: f64 = 16.0;

/// Reject a fused-gather launch whose `n_pairs` would exceed `grid.y`.
///
/// Without this the launch fails with `cudaErrorInvalidConfiguration`, the
/// `extern "C"` launcher discards that status, and the caller receives the
/// zero-initialised output buffer — a silently all-zero MoE layer. An error is
/// the only acceptable outcome.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn check_gather_gemv_pairs(n_pairs: usize, context: &str) -> Result<()> {
    if n_pairs > GATHER_GEMV_MAX_PAIRS {
        candle_core::bail!(
            "{context}: n_pairs {n_pairs} exceeds the fused gather kernel's grid.y limit \
             {GATHER_GEMV_MAX_PAIRS}; the launch would fail and return zeros. Lower \
             {ONDEVICE_MOE_MAX_TOKENS_ENV} or use the batched path."
        );
    }
    Ok(())
}

/// Expected number of **distinct** experts woken by `n_tokens` tokens each
/// routed to `top_k` of `num_experts`, assuming independent uniform routing:
/// `E * (1 - (1 - k/E)^n)`.
///
/// This is the quantity the whole MoE batching argument turns on: it is what
/// the fallback pays per step, and its ratio to `n_tokens * top_k` is the
/// weight amortisation a grouped kernel can realise.
///
/// Exact for real top-`k` routing (each token draws `k` *distinct* experts):
/// by symmetry a given expert is missed by one token with probability
/// `1 - k/E` regardless of the without-replacement draw, so the product over
/// independent tokens is exactly `(1 - k/E)^n`.
///
/// Public so a benchmark can assert its synthetic routing actually reproduces
/// this curve instead of some other traffic pattern (DOCTRINE D12).
pub fn expected_distinct_experts(n_tokens: usize, top_k: usize, num_experts: usize) -> f64 {
    if num_experts == 0 || n_tokens == 0 || top_k == 0 {
        return 0.0;
    }
    if top_k >= num_experts {
        return num_experts as f64;
    }
    let miss = 1.0 - (top_k as f64) / (num_experts as f64);
    let n = i32::try_from(n_tokens).unwrap_or(i32::MAX);
    (num_experts as f64) * (1.0 - miss.powi(n))
}

/// Expected `(token, slot)` pairs landing on each **woken** expert:
/// `n*k / expected_distinct_experts(n, k, E)`.
///
/// This — not `n_tokens` — is what the grouped GEMM's m-tiles are filled from,
/// so it is the quantity its dispatch gate is written in. Ranges from `1.0`
/// (every pair wakes a fresh expert; every m-tile holds one useful row) up to
/// `n*k/E` once every expert is woken.
///
/// Public so a benchmark can report the fill it actually measured at, instead
/// of a token count that means nothing without `top_k` and `num_experts`.
pub fn expected_pairs_per_distinct_expert(
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
) -> f64 {
    let distinct = expected_distinct_experts(n_tokens, top_k, num_experts);
    if distinct <= 0.0 {
        return 0.0;
    }
    (n_tokens as f64) * (top_k as f64) / distinct
}

/// Average grouped-GEMM m-tile fill for this routing shape, in `[0, ∞)`:
/// `expected_pairs_per_distinct_expert / GROUPED_TILE_M`. `1.0` means a
/// typical m-tile is exactly full; `0.088` is V4 at 32 tokens.
pub fn grouped_gemm_tile_fill(n_tokens: usize, top_k: usize, num_experts: usize) -> f64 {
    expected_pairs_per_distinct_expert(n_tokens, top_k, num_experts)
        / (super::grouped::GROUPED_TILE_M as f64)
}

/// **The grouped-GEMM dispatch gate.** Are its m-tiles full enough for the
/// per-m-tile weight staging to amortize — i.e. does each woken expert draw at
/// least `GROUPED_TILE_M` pairs?
///
/// See module §4 for the derivation and for the 5.7× regression that came from
/// gating on a raw token count instead. Pure function of shapes: no device
/// read, so it is safe inside a CUDA-graph capture.
///
/// Deliberately stated as a *fill* rather than a token threshold. A hardcoded
/// number would have to be re-derived every time `top_k`, `num_experts` or
/// `GROUPED_TILE_M` changes, and the last one that was not re-derived is the
/// regression this replaces.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn grouped_gemm_tiles_amortize(
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
) -> bool {
    expected_pairs_per_distinct_expert(n_tokens, top_k, num_experts)
        >= super::grouped::GROUPED_TILE_M as f64
}

/// Smallest `n_tokens` at which [`grouped_gemm_tiles_amortize`] holds, for logs
/// and tests. Pairs-per-woken-expert is monotone increasing in `n` (the
/// numerator is linear, the woken-expert count saturates at `E`), so the
/// predicate flips at most once and a scan finds it exactly.
///
/// Returns `None` when no token count reaches full tiles — `top_k == 0`, or a
/// routing so wide that `E * GROUPED_TILE_M` pairs never fit inside `grid.y`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn grouped_gemm_min_tokens(top_k: usize, num_experts: usize) -> Option<usize> {
    if top_k == 0 || num_experts == 0 {
        return None;
    }
    // Fill can only reach 1.0 once `n*k >= TILE_M * distinct`, and `distinct`
    // is at most `num_experts`, so this bound is sufficient and the scan is at
    // most a few thousand steps.
    let hi = (super::grouped::GROUPED_TILE_M * num_experts).div_ceil(top_k) + 1;
    (1..=hi).find(|&n| grouped_gemm_tiles_amortize(n, top_k, num_experts))
}

/// Largest token count the fused gather can take before `grid.y` overflows.
pub(crate) fn max_tokens_within_grid(top_k: usize) -> usize {
    if top_k == 0 {
        return 0;
    }
    GATHER_GEMV_MAX_PAIRS / top_k
}

/// LUT rung (`QtipLayer`) dispatch predicate: is the fused on-device gather
/// the path that moves less memory than `gather_forward_cuda`'s per-expert
/// dequantize-materialize fallback?
///
/// See the module docs for the derivation. Pure function of shapes — no device
/// read, so it is safe inside a CUDA-graph capture.
pub(crate) fn lut_fused_gather_preferred(
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
) -> bool {
    let pairs = n_tokens.saturating_mul(top_k);
    if pairs > GATHER_GEMV_MAX_PAIRS {
        return false;
    }
    // RUN-161 floor. Under CUDA-graph capture the fallback's `indices` D2H read
    // is recorded-not-executed, yielding garbage expert ids and an out-of-bounds
    // weight read. The decode regime must therefore stay fused unconditionally,
    // whatever the traffic model says about exotic routing shapes.
    if n_tokens <= super::DECODE_REGIME_MAX_TOKENS {
        return true;
    }
    (pairs as f64)
        <= DEQUANT_TRAFFIC_RATIO * expected_distinct_experts(n_tokens, top_k, num_experts)
}

/// The largest `n_tokens` for which [`lut_fused_gather_preferred`] holds, for
/// logging and tests. `pairs` is linear in `n` and the bound is concave and
/// saturating, so the predicate flips at most once; scanning downward from the
/// point where the saturated bound can no longer hold finds it exactly.
pub(crate) fn lut_fused_gather_max_tokens(top_k: usize, num_experts: usize) -> usize {
    let grid_limit = max_tokens_within_grid(top_k);
    if grid_limit == 0 {
        return 0;
    }
    // `pairs <= RATIO * num_experts` can never hold above this, so the scan is
    // a few thousand iterations at most regardless of the grid limit. The
    // RUN-161 floor is included so a degenerate routing shape cannot report a
    // boundary below the decode regime the dispatcher always fuses.
    let saturated = ((DEQUANT_TRAFFIC_RATIO * num_experts as f64) / top_k as f64).ceil() as usize;
    let hi = grid_limit.min(
        saturated
            .saturating_add(1)
            .max(super::DECODE_REGIME_MAX_TOKENS),
    );
    (1..=hi)
        .rev()
        .find(|&n| lut_fused_gather_preferred(n, top_k, num_experts))
        .unwrap_or(0)
}

/// Announce, once per process, that the LUT rung left the fused gather for the
/// per-expert dequantize-materialize path — with the boundary that decided it.
///
/// This path D2H-syncs the router and materializes every distinct expert to
/// BF16 in HBM. It is a legitimate prefill choice and a decode disaster, and
/// until now nothing in a run's log said which one had happened.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn log_lut_gather_fallback_once(n_tokens: usize, top_k: usize, num_experts: usize) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "QtipLayer::gather_forward: {n_tokens} tokens x top-{top_k} of {num_experts} experts \
             exceeds the fused on-device gather boundary \
             ({} tokens for this routing shape); falling back to the per-expert \
             dequantize+matmul path, which materializes BF16 expert weights to HBM and \
             syncs the router to the host. Set {ONDEVICE_MOE_MAX_TOKENS_ENV} to move the \
             boundary. (warned once per process)",
            lut_fused_gather_max_tokens(top_k, num_experts),
        );
    });
}

/// Announce, once per process, that a rung engaged the trellis grouped GEMM —
/// with the tile fill that decided it.
///
/// Which MoE kernel a run used is worth an order of magnitude of throughput and
/// used to be invisible in the log. Both this and
/// [`log_lut_gather_fallback_once`] exist so a results artifact can be read
/// back to the dispatch that produced it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn log_grouped_gemm_engaged_once(n_tokens: usize, top_k: usize, num_experts: usize) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let fill = grouped_gemm_tile_fill(n_tokens, top_k, num_experts);
        let boundary = grouped_gemm_min_tokens(top_k, num_experts);
        tracing::info!(
            "QTIP MoE: trellis grouped GEMM engaged at {n_tokens} tokens x top-{top_k} of \
             {num_experts} experts — m-tiles {:.0}% full ({:.2} pairs per woken expert against \
             GROUPED_TILE_M={}); the amortization boundary for this routing shape is {} tokens. \
             (logged once per process)",
            fill * 100.0,
            expected_pairs_per_distinct_expert(n_tokens, top_k, num_experts),
            super::grouped::GROUPED_TILE_M,
            match boundary {
                Some(n) => n.to_string(),
                None => "unreachable".to_string(),
            },
        );
    });
}

/// Read [`ONDEVICE_MOE_MAX_TOKENS_ENV`], logging once when it is set.
///
/// The MoE kernel selection has order-of-magnitude performance consequences
/// and used to be chosen by a silent `env::var` that never appeared in any
/// results artifact (wave27-AY §7). A run is now reproducible from its log.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn ondevice_max_tokens_override() -> Option<usize> {
    use std::sync::LazyLock;
    static OVERRIDE: LazyLock<Option<usize>> = LazyLock::new(|| {
        let v = std::env::var(ONDEVICE_MOE_MAX_TOKENS_ENV)
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        if let Some(v) = v {
            tracing::info!(
                "{ONDEVICE_MOE_MAX_TOKENS_ENV}={v}: fused on-device MoE gather capped at {v} \
                 tokens/step (structural kernel limit is {GATHER_GEMV_MAX_PAIRS} pairs)"
            );
        }
        v
    });
    *OVERRIDE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The distinct-expert curve is the load-bearing input to every batching
    /// claim in the project. Pin the values the FACTS table quotes.
    #[test]
    fn expected_distinct_experts_matches_the_published_curve() {
        // FACTS / wave27-AY: E(B) = 256 * (1 - (1 - 8/256)^B) for top-8 of 256.
        for (n, want) in [
            (1usize, 8.0),
            (8, 57.4),
            (32, 163.0),
            (128, 252.0),
            (256, 256.0),
        ] {
            let got = expected_distinct_experts(n, 8, 256);
            assert!((got - want).abs() < 1.0, "E({n}) = {got}, expected ~{want}");
        }
        // V4-Flash is **top-6** of 256, not top-8 (`research/v4_audit.md:263`,
        // `deepseek4.rs` V4_FLASH_CONFIG_JSON `num_experts_per_tok: 6`). The
        // widely-quoted top-8 row above overstates coverage at every batch, and
        // `examples/qtip_grouped_curve.rs` sizes its fixture off THIS curve —
        // pin it here so the harness cannot drift from the dispatch.
        // Exact analytic values, NOT sampled ones — a tolerance wide enough to
        // absorb Monte-Carlo noise would be wide enough to absorb a top-k
        // mix-up at the small batches, which is the whole point of the row.
        for (n, want) in [
            (1usize, 6.00),
            (8, 44.24),
            (16, 80.84),
            (32, 136.15),
            (52, 181.42),
            (64, 199.89),
            (128, 243.70),
        ] {
            let got = expected_distinct_experts(n, 6, 256);
            assert!(
                (got - want).abs() < 0.05,
                "top-6 E({n}) = {got}, expected {want}"
            );
        }
        // The two routings must stay distinguishable at the batch the harness
        // uses for its negative control, or that control proves nothing.
        assert!(
            expected_distinct_experts(16, 8, 256) - expected_distinct_experts(16, 6, 256) > 10.0,
            "top-6 and top-8 coverage at B=16 collapsed together"
        );
        // Degenerate inputs must not produce NaN or a bogus preference.
        assert_eq!(expected_distinct_experts(0, 8, 256), 0.0);
        assert_eq!(expected_distinct_experts(8, 0, 256), 0.0);
        assert_eq!(expected_distinct_experts(8, 256, 256), 256.0);
        assert_eq!(expected_distinct_experts(8, 512, 256), 256.0);
        assert!(expected_distinct_experts(usize::MAX, 8, 256).is_finite());
    }

    /// RUN-161: the decode regime must stay fused for **every** routing shape,
    /// because the fallback's host read of `indices` is invalid under CUDA-graph
    /// capture. Even a shape whose traffic model says otherwise must not flip.
    #[test]
    fn decode_regime_is_fused_unconditionally() {
        // top_k >= num_experts and a tiny expert count: the traffic model alone
        // would reject these (pairs 64 > 16 * 1 distinct expert).
        for n in 1..=super::super::DECODE_REGIME_MAX_TOKENS {
            assert!(
                lut_fused_gather_preferred(n, 8, 1),
                "{n} tokens must stay on the fused path (RUN-161 capture invariant)"
            );
        }
        assert!(!lut_fused_gather_preferred(
            super::super::DECODE_REGIME_MAX_TOKENS + 1,
            8,
            1
        ));
        // The floor never survives past the structural limit.
        assert!(!lut_fused_gather_preferred(
            1,
            GATHER_GEMV_MAX_PAIRS + 1,
            256
        ));
    }

    /// The old cap was 8. The derived boundary must be far above it for the
    /// production V4 shape — that is the entire point of this change — and it
    /// must land where the module's arithmetic says, not somewhere rounder.
    #[test]
    fn lut_boundary_is_where_the_traffic_model_says() {
        // top-8 of 256: crossover at n*8 == 16*E(n).
        let n8 = lut_fused_gather_max_tokens(8, 256);
        assert!(
            (480..=560).contains(&n8),
            "top-8/256 boundary {n8} is not the ~512 the traffic model predicts"
        );
        // top-6 of 256 (V4-Flash): the same solve gives ~683.
        let n6 = lut_fused_gather_max_tokens(6, 256);
        assert!(
            (640..=730).contains(&n6),
            "top-6/256 boundary {n6} is not the ~683 the traffic model predicts"
        );
        // Monotone: preferred below the boundary, not preferred above it.
        assert!(lut_fused_gather_preferred(n8, 8, 256));
        assert!(!lut_fused_gather_preferred(n8 + 1, 8, 256));
        // Every token count the old cap covered is still covered, and so is
        // every batch the measured H200 sweep produced (8 / 13 / 32).
        for n in [1usize, 8, 9, 13, 16, 32, 64, 128] {
            assert!(
                lut_fused_gather_preferred(n, 8, 256),
                "{n} tokens should use the fused path"
            );
        }
    }

    /// The derived boundary must never hand the kernel more pairs than
    /// `grid.y` can address, for any routing shape — including small expert
    /// counts where the saturated bound is loose.
    #[test]
    fn derived_boundary_never_exceeds_the_structural_limit() {
        for top_k in [1usize, 2, 6, 8, 16, 64] {
            for num_experts in [1usize, 8, 64, 128, 256, 1024, 8192] {
                let n = lut_fused_gather_max_tokens(top_k, num_experts);
                assert!(
                    n * top_k <= GATHER_GEMV_MAX_PAIRS,
                    "top_k={top_k} E={num_experts}: {n} tokens => {} pairs > {GATHER_GEMV_MAX_PAIRS}",
                    n * top_k
                );
                assert!(!lut_fused_gather_preferred(
                    max_tokens_within_grid(top_k) + 1,
                    top_k,
                    num_experts
                ));
            }
        }
    }

    /// The guard is the thing standing between a too-large launch and a
    /// silently all-zero MoE layer. It must fail, and it must fail at exactly
    /// the documented pair count.
    #[test]
    fn pair_guard_rejects_exactly_above_the_grid_limit() {
        assert!(check_gather_gemv_pairs(0, "t").is_ok());
        assert!(check_gather_gemv_pairs(GATHER_GEMV_MAX_PAIRS, "t").is_ok());
        let err = check_gather_gemv_pairs(GATHER_GEMV_MAX_PAIRS + 1, "ctx")
            .expect_err("one pair past grid.y must be an error, not zeros");
        let msg = err.to_string();
        assert!(msg.contains("ctx"), "{msg}");
        assert!(msg.contains("65535"), "{msg}");
    }

    /// V4's routing (top-6 of 256) is the shape the 5.7× regression was
    /// measured on. Pin the whole fill curve quoted in module §4 — these are
    /// the numbers the gate is derived from, so if the formula drifts the
    /// derivation in the docs stops being true.
    #[test]
    fn tile_fill_matches_the_v4_curve_in_the_module_docs() {
        // (n_tokens, pairs per woken expert)
        for (n, expected) in [
            (32usize, 1.41f64),
            (128, 3.15),
            (512, 12.00),
            (683, 16.01),
            (1024, 24.00),
        ] {
            let got = expected_pairs_per_distinct_expert(n, 6, 256);
            assert!(
                (got - expected).abs() < 0.01,
                "n={n}: pairs/woken expert {got:.4}, docs say {expected}"
            );
            let fill = grouped_gemm_tile_fill(n, 6, 256);
            assert!(
                (fill - got / 16.0).abs() < 1e-9,
                "fill must be pairs-per-expert / GROUPED_TILE_M"
            );
        }
    }

    /// The regression itself, as an assertion: the grouped GEMM must NOT be
    /// chosen at the batch widths a decode server actually runs at, and must
    /// be chosen once tiles are full. 9 tokens is where the old raw
    /// `n_tokens > DECODE_REGIME_MAX_TOKENS` gate flipped.
    #[test]
    fn grouped_gate_refuses_decode_widths_and_accepts_full_tiles() {
        for n in [9usize, 16, 32, 64, 128, 256, 512] {
            assert!(
                !grouped_gemm_tiles_amortize(n, 6, 256),
                "n={n}: tiles are {:.1}% full, the grouped GEMM must not be selected",
                grouped_gemm_tile_fill(n, 6, 256) * 100.0
            );
        }
        for n in [683usize, 1024, 4096] {
            assert!(
                grouped_gemm_tiles_amortize(n, 6, 256),
                "n={n}: tiles are {:.1}% full, the grouped GEMM should be selected",
                grouped_gemm_tile_fill(n, 6, 256) * 100.0
            );
        }
        // The gate is a fill, not a token count: the SAME token count flips
        // the other way when the routing shape changes. This is the property a
        // hardcoded threshold cannot have.
        assert!(
            grouped_gemm_tiles_amortize(32, 6, 8),
            "32 tokens x top-6 of only 8 experts fills every tile 4x over"
        );
    }

    /// `grouped_gemm_min_tokens` must agree with the predicate it summarises,
    /// exactly at the boundary, for every routing shape we might serve.
    #[test]
    fn grouped_min_tokens_is_the_exact_flip_point() {
        for top_k in [1usize, 2, 6, 8, 16] {
            for num_experts in [1usize, 8, 64, 128, 256, 1024] {
                let n = grouped_gemm_min_tokens(top_k, num_experts)
                    .unwrap_or_else(|| panic!("top_k={top_k} E={num_experts}: no flip point"));
                assert!(
                    grouped_gemm_tiles_amortize(n, top_k, num_experts),
                    "top_k={top_k} E={num_experts}: predicate false at its own boundary {n}"
                );
                assert!(
                    n == 1 || !grouped_gemm_tiles_amortize(n - 1, top_k, num_experts),
                    "top_k={top_k} E={num_experts}: predicate already true below boundary {n}"
                );
            }
        }
        assert_eq!(grouped_gemm_min_tokens(6, 256), Some(683));
    }

    /// The two boundaries in this module are set by two INDEPENDENT constants
    /// that happen to both be 16 — `DEQUANT_TRAFFIC_RATIO` and
    /// `GROUPED_TILE_M`. While they agree, the LUT rung hands off from the
    /// fused GEMV straight to the grouped GEMM with no window in which the
    /// dequantize-materialize fallback runs. If either constant moves that
    /// window opens silently; this test makes it loud instead.
    #[test]
    fn grouped_boundary_is_the_fused_gemv_boundary() {
        assert_eq!(
            DEQUANT_TRAFFIC_RATIO as usize,
            super::super::grouped::GROUPED_TILE_M,
            "the two dispatch boundaries have diverged: the LUT rung now has a token range \
             where the fused GEMV has been left but the grouped GEMM's tiles are not full, \
             so it falls through to the dequantize-materialize path. Re-derive both."
        );
        for top_k in [1usize, 2, 6, 8] {
            for num_experts in [8usize, 64, 256, 1024] {
                let boundary = grouped_gemm_min_tokens(top_k, num_experts).unwrap();
                // Below the boundary: fused GEMV. At and above it: grouped.
                // Both predicates include their own equality case, so the two
                // may overlap by exactly one token count — the dispatcher
                // evaluates `grouped_preferred` first, so grouped wins the tie.
                // What must NEVER happen is a token count where both are false:
                // that is a silent fall-through to the dequantize-materialize
                // path, which is worse than either.
                for n in [boundary.saturating_sub(1).max(1), boundary, boundary + 1] {
                    if n <= super::super::DECODE_REGIME_MAX_TOKENS
                        || n.saturating_mul(top_k) > GATHER_GEMV_MAX_PAIRS
                    {
                        continue;
                    }
                    assert!(
                        lut_fused_gather_preferred(n, top_k, num_experts)
                            || grouped_gemm_tiles_amortize(n, top_k, num_experts),
                        "top_k={top_k} E={num_experts} n={n}: neither the fused GEMV nor the \
                         grouped GEMM is preferred, so dispatch falls through to the \
                         dequantize-materialize path"
                    );
                }
                assert!(
                    grouped_gemm_tiles_amortize(boundary, top_k, num_experts),
                    "top_k={top_k} E={num_experts}: grouped must win at its own boundary"
                );
            }
        }
    }
}
