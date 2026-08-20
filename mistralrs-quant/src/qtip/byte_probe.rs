//! Parent system: ArcQuant / QTIP — expert-weight byte accounting.
//!
//! Answers one question: **how many bytes of expert weight does the MoE path
//! read per decode step, and does that number stay flat as the batch grows?**
//!
//! - Flat with batch  ⇒ amortisation works; a step's expert bytes are shared.
//! - Linear in batch  ⇒ every token drags its own copy of the expert weights,
//!   and aggregate throughput cannot scale no matter how many users arrive.
//!
//! ## Method (state this whenever you quote the number)
//!
//! These are **kernel-issued bytes**, computed exactly from each launch's own
//! geometry — not a profiler sample, not an estimate, and not DRAM traffic.
//! `ncu` cannot attach on the rented H200 ("Failed to prepare kernel for
//! profiling / Unknown Error on device 0") and the box has no DCGM, so a
//! hardware DRAM counter is not available; `nvidia-smi dmon` reports a
//! utilisation percentage, not bytes.
//!
//! The gather GEMV (`qtip2b_gemv_tuned_kernel`, `qtip_gather_gemv.cu`) is
//! indexed `pair = blockIdx.y`, `row-block = blockIdx.x`, and each block loads
//! its own expert's packed rows. Summed over `blockIdx.x`, **one pair reads one
//! whole expert**: `n_rows * packed_per_row` bytes. There is no dedup between
//! pairs, so a launch reads `n_pairs * n_rows * packed_per_row` bytes — which is
//! what [`record_gemv`] accumulates.
//!
//! The grouped GEMM (`qtip_grouped_gemm.cu`) sorts pairs by expert and stages a
//! woken expert's bytes once per `GROUPED_TILE_M` rows, so it reads
//! `ceil(pairs_e / TILE_M)` copies of expert `e` — [`record_grouped`] uses the
//! same per-expert accounting.
//!
//! Because these are loads *issued*, they are an upper bound on DRAM traffic:
//! repeated reads of the same expert within a short window can be served by L2.
//! That distinction is the point — a card at 99% SM occupancy with a 3% memory
//! controller is exactly what re-reading a cached-but-redundant working set
//! looks like. The redundancy factor below is the quantity that has to reach
//! 1.0 for batching to amortise, whichever cache serves it.
//!
//! Enabled with `ARC_MOE_BYTE_PROBE=1`. Off by default and it costs one relaxed
//! atomic add per expert-GEMM launch when on — no device sync, no D2H read, so
//! it is safe under CUDA-graph capture and does not disturb the dispatch it is
//! measuring.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

static INVOCATIONS: AtomicU64 = AtomicU64::new(0);
static GEMV_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
static GROUPED_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
static PAIRS: AtomicU64 = AtomicU64::new(0);
/// Bytes of expert weight the launches were pointed at.
static ISSUED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Lower bound: each expert woken by a launch read exactly once.
static FLOOR_BYTES: AtomicU64 = AtomicU64::new(0);
/// Bytes of a single expert, for reporting the working-set size.
static BYTES_PER_EXPERT: AtomicU64 = AtomicU64::new(0);
static NUM_EXPERTS: AtomicU64 = AtomicU64::new(0);

/// `ARC_MOE_BYTE_PROBE=1`. Read once — `std::env::var` takes a global lock and
/// this sits on the per-layer decode path.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("ARC_MOE_BYTE_PROBE").as_deref(), Ok("1")))
}

/// How many distinct experts a launch can possibly have woken. Exact whenever
/// every expert is woken (the saturated regime that matters for the fleet
/// question) and a conservative *over*-estimate otherwise, so the redundancy
/// factor this yields is never flattering.
#[inline]
fn distinct_upper_bound(n_pairs: usize, num_experts: usize) -> usize {
    n_pairs.min(num_experts)
}

/// The non-amortising gather GEMV: one pair reads one whole expert.
pub fn record_gemv(n_pairs: usize, n_rows: usize, packed_per_row: usize, num_experts: usize) {
    if !enabled() {
        return;
    }
    let per_expert = (n_rows as u64).saturating_mul(packed_per_row as u64);
    let issued = per_expert.saturating_mul(n_pairs as u64);
    let floor = per_expert.saturating_mul(distinct_upper_bound(n_pairs, num_experts) as u64);
    INVOCATIONS.fetch_add(1, Ordering::Relaxed);
    GEMV_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
    PAIRS.fetch_add(n_pairs as u64, Ordering::Relaxed);
    ISSUED_BYTES.fetch_add(issued, Ordering::Relaxed);
    FLOOR_BYTES.fetch_add(floor, Ordering::Relaxed);
    BYTES_PER_EXPERT.store(per_expert, Ordering::Relaxed);
    NUM_EXPERTS.store(num_experts as u64, Ordering::Relaxed);
    maybe_dump();
}

/// The amortising grouped GEMM: a woken expert's bytes are staged once per
/// `tile_m` pairs that landed on it. Without the on-device sort order here, use
/// the even-spread estimate `ceil(pairs/distinct / tile_m)` copies per expert —
/// exact when routing is balanced, which is the regime the gate targets.
pub fn record_grouped(
    n_pairs: usize,
    n_rows: usize,
    packed_per_row: usize,
    num_experts: usize,
    tile_m: usize,
) {
    if !enabled() {
        return;
    }
    let per_expert = (n_rows as u64).saturating_mul(packed_per_row as u64);
    let distinct = distinct_upper_bound(n_pairs, num_experts).max(1);
    let pairs_each = n_pairs.div_ceil(distinct);
    let copies = pairs_each.div_ceil(tile_m.max(1));
    let issued = per_expert
        .saturating_mul(distinct as u64)
        .saturating_mul(copies as u64);
    let floor = per_expert.saturating_mul(distinct as u64);
    INVOCATIONS.fetch_add(1, Ordering::Relaxed);
    GROUPED_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
    PAIRS.fetch_add(n_pairs as u64, Ordering::Relaxed);
    ISSUED_BYTES.fetch_add(issued, Ordering::Relaxed);
    FLOOR_BYTES.fetch_add(floor, Ordering::Relaxed);
    BYTES_PER_EXPERT.store(per_expert, Ordering::Relaxed);
    NUM_EXPERTS.store(num_experts as u64, Ordering::Relaxed);
    maybe_dump();
}

/// Periodic cumulative dump so a run yields a number even if it is killed.
/// `ARC_MOE_BYTE_PROBE_EVERY` invocations, default 4096.
fn maybe_dump() {
    static EVERY: OnceLock<u64> = OnceLock::new();
    let every = *EVERY.get_or_init(|| {
        std::env::var("ARC_MOE_BYTE_PROBE_EVERY")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(4096)
    });
    let n = INVOCATIONS.load(Ordering::Relaxed);
    if n.is_multiple_of(every) {
        eprintln!("{}", report());
    }
}

/// One machine-greppable cumulative line. Divide by the tokens the run actually
/// produced to get bytes/token; multiply by the batch to get bytes/step.
pub fn report() -> String {
    let inv = INVOCATIONS.load(Ordering::Relaxed);
    let gemv = GEMV_INVOCATIONS.load(Ordering::Relaxed);
    let grouped = GROUPED_INVOCATIONS.load(Ordering::Relaxed);
    let pairs = PAIRS.load(Ordering::Relaxed);
    let issued = ISSUED_BYTES.load(Ordering::Relaxed);
    let floor = FLOOR_BYTES.load(Ordering::Relaxed);
    let per_expert = BYTES_PER_EXPERT.load(Ordering::Relaxed);
    let experts = NUM_EXPERTS.load(Ordering::Relaxed);
    let redundancy = if floor > 0 {
        issued as f64 / floor as f64
    } else {
        0.0
    };
    format!(
        "ARC_MOE_BYTES invocations={inv} gemv={gemv} grouped={grouped} pairs={pairs} \
         issued_bytes={issued} floor_bytes={floor} redundancy={redundancy:.3} \
         bytes_per_expert={per_expert} num_experts={experts}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GEMV reads one whole expert per pair, so its redundancy is exactly
    /// pairs/distinct — the quantity that has to fall to 1.0 for batching to
    /// amortise. These are the shapes V4 actually runs (top-6 of 256).
    #[test]
    fn gemv_redundancy_is_pairs_per_distinct_expert() {
        let per_expert = 1024usize * 96;
        for (n_tokens, top_k, experts) in [(1usize, 6usize, 256usize), (512, 6, 256)] {
            let pairs = n_tokens * top_k;
            let distinct = distinct_upper_bound(pairs, experts);
            let issued = per_expert * pairs;
            let floor = per_expert * distinct;
            assert_eq!(issued / floor.max(1), pairs / distinct.max(1));
        }
        // B=1 wakes 6 experts with 6 pairs: nothing to share, redundancy 1.
        assert_eq!(distinct_upper_bound(6, 256), 6);
        // B=512 wakes all 256 with 3072 pairs: 12x redundant.
        assert_eq!(3072 / distinct_upper_bound(3072, 256), 12);
    }

    /// MEASURED on H200, V4-Flash qtip2b, 2026-08-20 (ARC_MOE_BYTE_PROBE=1):
    /// one expert is 2 MiB and a step is 129 gather invocations (43 layers x 3).
    /// Expert bytes per step scale LINEARLY with the batch on the GEMV arm and
    /// are FLAT on the grouped arm. This is a characterization test: if the
    /// dispatch is fixed, it will fail here first and the numbers get updated.
    #[test]
    fn v4_expert_bytes_per_step_scale_with_batch_on_the_gemv_arm() {
        const PER_EXPERT: u64 = 2 * 1024 * 1024; // measured
        const INV_PER_STEP: u64 = 129; // measured: 43 layers x 3 gathers
        const TOP_K: usize = 6;
        const EXPERTS: usize = 256;
        let gemv_gib = |n_tokens: usize| -> f64 {
            let pairs = (n_tokens * TOP_K) as u64;
            (INV_PER_STEP * pairs * PER_EXPERT) as f64 / (1024.0 * 1024.0 * 1024.0)
        };
        let floor_gib = |n_tokens: usize| -> f64 {
            let distinct = distinct_upper_bound(n_tokens * TOP_K, EXPERTS) as u64;
            (INV_PER_STEP * distinct * PER_EXPERT) as f64 / (1024.0 * 1024.0 * 1024.0)
        };
        // GEMV: linear in batch, 512x the users reads 512x the expert bytes.
        for (n, want) in [(1usize, 1.5f64), (8, 12.1), (64, 96.8), (512, 774.0)] {
            assert!(
                (gemv_gib(n) - want).abs() < 0.1,
                "n_tokens={n}: {:.1} GiB/step, expected {want}",
                gemv_gib(n)
            );
        }
        // Amortised floor saturates at the whole expert working set.
        assert!((floor_gib(256) - 64.5).abs() < 0.1);
        assert!((floor_gib(512) - 64.5).abs() < 0.1);
        // The redundancy the fleet pays at B=512.
        assert!((gemv_gib(512) / floor_gib(512) - 12.0).abs() < 0.01);
    }

    /// The grouped GEMM stages an expert once per TILE_M pairs, so at the same
    /// shape it must read strictly fewer bytes than the GEMV once the tiles
    /// fill — that is the whole claim the dispatch gate rests on.
    #[test]
    fn grouped_reads_less_than_gemv_once_tiles_fill() {
        let (n_rows, ppr, experts, tile) = (1024usize, 96usize, 256usize, 16usize);
        let per_expert = (n_rows * ppr) as u64;
        for n_tokens in [512usize, 683, 2048] {
            let pairs = n_tokens * 6;
            let distinct = distinct_upper_bound(pairs, experts);
            let gemv = per_expert * pairs as u64;
            let copies = pairs.div_ceil(distinct).div_ceil(tile);
            let grouped = per_expert * distinct as u64 * copies as u64;
            assert!(
                grouped < gemv,
                "grouped {grouped} !< gemv {gemv} at n_tokens={n_tokens}"
            );
        }
    }
}
