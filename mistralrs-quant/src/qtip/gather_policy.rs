//! Dispatch policy for the fused on-device MoE gather.
//!
//! `QuantMethod::gather_forward` on a stacked-expert QTIP layer picks between
//! a **fused on-device** path (one kernel, routing read on the GPU, no host
//! sync) and a **fallback**. Which fallback depends on the rung, and the two
//! are not remotely equivalent:
//!
//! | rung | over-cap path | what it does |
//! |---|---|---|
//! | LUT (`QtipLayer`, `--isq qtip2`) | `gather_forward_cuda` | D2H-syncs the router, dequantizes **every distinct expert** to BF16 in HBM, holds them all live, then one `index_select`+`matmul`+`index_add` per expert |
//! | bitshift (`Qtip2bLayer`, `--isq qtip2b`) | `grouped_gemm_2b_cuda` | sorts pairs by expert on-device and runs the persistent tensor-core tile loop — the *amortizing* kernel |
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
//! *flat* per token. The amortising kernel is the grouped GEMM, and only the
//! bitshift rung has one.

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
}
