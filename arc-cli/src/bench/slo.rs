//! Service-level-objective (SLO) tier definitions for `arc bench --suite agentperf`.
//!
//! The Artificial Analysis "AA-AgentPerf" methodology defines tiered latency
//! targets that an inference deployment must hold while the concurrent-user
//! load is scaled up. A tier passes at a given concurrency K when:
//!
//!   - P25(output_speed) ≥ tier.min_p25_output_speed_tok_per_s
//!   - P95(TTFT)         ≤ tier.max_p95_ttft_seconds
//!
//! The percentiles are taken over the **steady-state** window of a phase
//! (after a warmup gate of equal length). The scheduler then exponential-
//! ramps K until the first failing tier point, and binary-searches between
//! the last passing K and the first failing K. See `scheduler.rs`.
//!
//! This module is **pure** — no I/O, no async — so it can be unit-tested
//! against synthetic distributions cheaply.

use serde::{Deserialize, Serialize};

/// Tier definition. The Artificial Analysis defaults (mirrored from public
/// AA-AgentPerf docs and our own README) are:
///
/// | Tier | min P25 out tok/s | max P95 TTFT (s) |
/// |------|-------------------|------------------|
/// | 1    |       100         |        1.5       |
/// | 2    |        60         |        2.0       |
/// | 3    |        30         |        3.0       |
/// | 4    |        15         |        5.0       |
///
/// Tier 1 is the strictest; tier 4 the loosest. Tier 2 is the practical
/// "modern reasoning model" default and the one our example agentperf JSON
/// is computed under.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SloTier {
    pub tier: u8,
    pub min_p25_output_speed_tok_per_s: f64,
    pub max_p95_ttft_seconds: f64,
}

impl SloTier {
    /// Look up a tier by integer id. Returns `None` for unknown tiers; the
    /// caller decides how to surface that error.
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            1 => Some(Self {
                tier: 1,
                min_p25_output_speed_tok_per_s: 100.0,
                max_p95_ttft_seconds: 1.5,
            }),
            2 => Some(Self {
                tier: 2,
                min_p25_output_speed_tok_per_s: 60.0,
                max_p95_ttft_seconds: 2.0,
            }),
            3 => Some(Self {
                tier: 3,
                min_p25_output_speed_tok_per_s: 30.0,
                max_p95_ttft_seconds: 3.0,
            }),
            4 => Some(Self {
                tier: 4,
                min_p25_output_speed_tok_per_s: 15.0,
                max_p95_ttft_seconds: 5.0,
            }),
            _ => None,
        }
    }
}

/// Result of evaluating a phase's measurements against a tier.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SloEvaluation {
    pub p25_output_speed_tok_per_s: f64,
    pub p95_ttft_seconds: f64,
    pub p25_pass: bool,
    pub p95_pass: bool,
    pub overall_pass: bool,
}

impl SloEvaluation {
    pub fn evaluate(p25_speed: f64, p95_ttft: f64, tier: SloTier) -> Self {
        let p25_pass = p25_speed >= tier.min_p25_output_speed_tok_per_s;
        let p95_pass = p95_ttft <= tier.max_p95_ttft_seconds;
        Self {
            p25_output_speed_tok_per_s: p25_speed,
            p95_ttft_seconds: p95_ttft,
            p25_pass,
            p95_pass,
            overall_pass: p25_pass && p95_pass,
        }
    }
}

/// Compute the percentile of a slice of `f64`. `q` is in `[0.0, 1.0]`.
///
/// Uses linear interpolation between the two nearest order statistics
/// (NumPy `linear` / R type-7 method). Empty input returns `NaN`; this
/// makes accidental empty-window bugs surface immediately rather than
/// silently reporting 0.
pub fn percentile(samples: &[f64], q: f64) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let h = q * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_lookup_known_ids() {
        let t1 = SloTier::from_id(1).unwrap();
        assert_eq!(t1.tier, 1);
        assert_eq!(t1.min_p25_output_speed_tok_per_s, 100.0);
        assert_eq!(t1.max_p95_ttft_seconds, 1.5);

        let t2 = SloTier::from_id(2).unwrap();
        assert_eq!(t2.min_p25_output_speed_tok_per_s, 60.0);

        assert!(SloTier::from_id(5).is_none());
        assert!(SloTier::from_id(0).is_none());
    }

    #[test]
    fn tiers_are_monotone_relaxing() {
        // Higher tier id = looser SLO. Speed floor drops, TTFT ceiling rises.
        let speeds: Vec<f64> = (1..=4)
            .map(|i| SloTier::from_id(i).unwrap().min_p25_output_speed_tok_per_s)
            .collect();
        let ttfts: Vec<f64> = (1..=4)
            .map(|i| SloTier::from_id(i).unwrap().max_p95_ttft_seconds)
            .collect();
        for w in speeds.windows(2) {
            assert!(w[0] > w[1], "speed floor should monotonically decrease");
        }
        for w in ttfts.windows(2) {
            assert!(w[0] < w[1], "TTFT ceiling should monotonically increase");
        }
    }

    #[test]
    fn percentile_uniform_distribution() {
        // 0..=100 → P25 = 25, P95 = 95.
        let xs: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let p25 = percentile(&xs, 0.25);
        let p95 = percentile(&xs, 0.95);
        assert!((p25 - 25.0).abs() < 1e-9, "P25 should be 25, got {p25}");
        assert!((p95 - 95.0).abs() < 1e-9, "P95 should be 95, got {p95}");
    }

    #[test]
    fn percentile_interpolation() {
        // [1, 2, 3, 4]. P25 = 1.75 (linear between index 0 and 1).
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let p25 = percentile(&xs, 0.25);
        assert!((p25 - 1.75).abs() < 1e-9, "P25 should be 1.75, got {p25}");
    }

    #[test]
    fn percentile_empty_returns_nan() {
        let p = percentile(&[], 0.5);
        assert!(p.is_nan());
    }

    #[test]
    fn percentile_single_element() {
        let p = percentile(&[42.0], 0.0);
        assert_eq!(p, 42.0);
        let p = percentile(&[42.0], 0.5);
        assert_eq!(p, 42.0);
        let p = percentile(&[42.0], 1.0);
        assert_eq!(p, 42.0);
    }

    #[test]
    fn percentile_unsorted_input_handled() {
        // We should not require pre-sorted input.
        let xs = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let p50 = percentile(&xs, 0.5);
        // sorted: [1,1,2,3,4,5,6,9]; index 3.5 → (3+4)/2 = 3.5
        assert!((p50 - 3.5).abs() < 1e-9, "P50 should be 3.5, got {p50}");
    }

    #[test]
    fn slo_evaluation_pass_and_fail() {
        let tier2 = SloTier::from_id(2).unwrap();
        // Pass: 80 tok/s ≥ 60, 1.0s ≤ 2.0s.
        let ev = SloEvaluation::evaluate(80.0, 1.0, tier2);
        assert!(ev.p25_pass);
        assert!(ev.p95_pass);
        assert!(ev.overall_pass);

        // Fail on speed.
        let ev = SloEvaluation::evaluate(30.0, 1.0, tier2);
        assert!(!ev.p25_pass);
        assert!(ev.p95_pass);
        assert!(!ev.overall_pass);

        // Fail on TTFT.
        let ev = SloEvaluation::evaluate(80.0, 3.0, tier2);
        assert!(ev.p25_pass);
        assert!(!ev.p95_pass);
        assert!(!ev.overall_pass);

        // Boundary: exactly equal counts as pass.
        let ev = SloEvaluation::evaluate(60.0, 2.0, tier2);
        assert!(ev.p25_pass);
        assert!(ev.p95_pass);
        assert!(ev.overall_pass);
    }
}
