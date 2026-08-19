//! Concurrency-search scheduler — the heart of `arc bench --suite agentperf`.
//!
//! The algorithm follows the public AA-AgentPerf recipe:
//!
//!   1. **Exponential ramp** — try K = 1, 2, 4, 8, 16, ... until the first K
//!      that fails the SLO. Cap at `--max-users` so we don't melt down.
//!   2. **Binary search** — between the last-pass K and the first-fail K,
//!      bisect until the interval is ≤ 1 user. The largest K whose phase
//!      passes the SLO is reported as the "saturation point" for this tier.
//!   3. **Per-phase pacing** — each K is run for a warmup window (requests
//!      fire but measurements are NOT recorded) followed by a steady-state
//!      window over which P25 / P95 are computed.
//!
//! This module is intentionally separable from the vendor: it asks the
//! vendor for measurements and decides the next K based purely on the
//! pass/fail boolean. The TUI subscribes to a broadcast channel and
//! renders progress live.

use crate::bench::slo::{percentile, SloEvaluation, SloTier};
use crate::bench::vendor::{RequestResult, Vendor};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex};

/// One phase of the search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseRecord {
    pub phase_index: u32,
    pub concurrent_users: u32,
    /// "ramp" or "bisect".
    pub kind: String,
    /// Per-request measurements collected during the **steady-state** window.
    /// (Warmup measurements are not recorded.)
    pub steady_state_samples: u32,
    pub evaluation: SloEvaluation,
    pub duration_seconds: f64,
}

/// Final scheduler outcome — what `report.rs` serialises.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleReport {
    pub tier: SloTier,
    pub max_users_explored: u32,
    pub max_users_cap: u32,
    /// Largest K that passed the SLO. `None` if even K=1 failed.
    pub saturation_users: Option<u32>,
    pub phases: Vec<PhaseRecord>,
    pub total_wall_seconds: f64,
}

/// Configuration knobs for the scheduler.
#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// Upper bound on concurrent users; the ramp stops here.
    pub max_users: u32,
    /// Warmup window per phase. Measurements during this window are
    /// discarded; their job is to fill the engine's prefix cache and let
    /// the scheduler converge on a steady state.
    pub warmup: Duration,
    /// Steady-state window per phase. P25 / P95 are computed over the
    /// requests whose result lands within this window.
    pub steady_state: Duration,
    /// Output-token budget per request. Smaller → more requests per phase,
    /// noisier percentiles. Larger → fewer requests, smoother percentiles.
    pub max_tokens_per_request: u32,
    /// Bisect stops when `first_fail - last_pass <= 1`.
    pub bisect_min_gap: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_users: 256,
            warmup: Duration::from_secs(30),
            steady_state: Duration::from_secs(30),
            max_tokens_per_request: 128,
            bisect_min_gap: 1,
        }
    }
}

/// Event published by the scheduler so the TUI can render progress live.
#[derive(Debug, Clone)]
pub enum Event {
    Started {
        tier: SloTier,
        total_max_users: u32,
    },
    PhaseStarted {
        phase_index: u32,
        concurrent_users: u32,
        kind: String,
        warmup: Duration,
        steady_state: Duration,
    },
    PhaseProgress {
        phase_index: u32,
        elapsed_in_phase: Duration,
        in_warmup: bool,
        steady_state_samples_so_far: u32,
        running_p25_speed: f64,
        running_p95_ttft: f64,
        running_system_throughput_tok_per_s: f64,
        active_requests: u32,
    },
    PhaseFinished {
        record: PhaseRecord,
    },
    Finished {
        report: ScheduleReport,
    },
}

/// Shared state the harness keeps for in-flight request tracking — the
/// TUI's "active requests" counter reads this through the event stream.
#[derive(Default)]
struct PhaseState {
    /// Measurements (TTFT, output speed, ok) from the steady-state window.
    steady_state_results: Vec<RequestResult>,
    /// Measurements from the *whole* phase (warmup + steady). Used for
    /// the running TUI display.
    all_results: Vec<RequestResult>,
    /// How many requests returned an error. A failed request is still pushed
    /// as an `ok: false` datum so the rates stay honest, but the count is what
    /// lets the phase tell "slow" apart from "nothing worked".
    failed: u64,
    /// The first request error, with its full cause chain. Without this the
    /// scheduler recorded a failure as a zeroed `RequestResult` and threw the
    /// reason away — so a run in which every request died (a poisoned CUDA
    /// context, a dead server) produced a complete-looking table of zeros
    /// rather than an error. Absence must not be recorded as a value.
    first_error: Option<String>,
}

/// The scheduler. It owns a vendor (behind an Arc so concurrent requests
/// can clone the handle) plus the event broadcaster.
pub struct Scheduler {
    vendor: Arc<dyn Vendor>,
    cfg: SchedulerConfig,
    tier: SloTier,
    events: broadcast::Sender<Event>,
}

impl Scheduler {
    pub fn new(vendor: Arc<dyn Vendor>, cfg: SchedulerConfig, tier: SloTier) -> Self {
        let (tx, _rx) = broadcast::channel(256);
        Self {
            vendor,
            cfg,
            tier,
            events: tx,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.events.subscribe()
    }

    pub fn vendor_name(&self) -> String {
        self.vendor.name().to_string()
    }

    /// Read the configured tier. Useful for callers that want to render
    /// or log the SLO without round-tripping it through events.
    #[allow(dead_code)]
    pub fn tier(&self) -> SloTier {
        self.tier
    }

    /// Read the scheduler config. Same use case as [`tier`].
    #[allow(dead_code)]
    pub fn cfg(&self) -> SchedulerConfig {
        self.cfg
    }

    /// Run the full agentperf search. Returns the final report.
    pub async fn run(&self) -> Result<ScheduleReport> {
        let started = Instant::now();
        let _ = self.events.send(Event::Started {
            tier: self.tier,
            total_max_users: self.cfg.max_users,
        });

        let mut phases: Vec<PhaseRecord> = Vec::new();
        let mut phase_index: u32 = 0;

        // --- Exponential ramp ----------------------------------------------
        let mut last_pass: Option<u32> = None;
        let mut first_fail: Option<u32> = None;
        let mut k: u32 = 1;
        loop {
            phase_index += 1;
            let record = self.run_phase(phase_index, k, "ramp").await?;
            let pass = record.evaluation.overall_pass;
            let _ = self.events.send(Event::PhaseFinished {
                record: record.clone(),
            });
            phases.push(record);

            if pass {
                last_pass = Some(k);
                if k >= self.cfg.max_users {
                    // Hit ceiling without failing — declare saturation = max.
                    break;
                }
                let next = k.saturating_mul(2).min(self.cfg.max_users);
                if next == k {
                    break;
                }
                k = next;
            } else {
                first_fail = Some(k);
                break;
            }
        }

        // --- Binary search --------------------------------------------------
        // Only meaningful if we have both a last-pass and a first-fail.
        if let (Some(lp), Some(ff)) = (last_pass, first_fail) {
            let mut lo = lp;
            let mut hi = ff;
            while hi - lo > self.cfg.bisect_min_gap {
                let mid = lo + (hi - lo) / 2;
                phase_index += 1;
                let record = self.run_phase(phase_index, mid, "bisect").await?;
                let pass = record.evaluation.overall_pass;
                let _ = self.events.send(Event::PhaseFinished {
                    record: record.clone(),
                });
                phases.push(record);
                if pass {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            last_pass = Some(lo);
        }

        let max_explored = phases.iter().map(|p| p.concurrent_users).max().unwrap_or(0);

        let report = ScheduleReport {
            tier: self.tier,
            max_users_explored: max_explored,
            max_users_cap: self.cfg.max_users,
            saturation_users: last_pass,
            phases,
            total_wall_seconds: started.elapsed().as_secs_f64(),
        };

        let _ = self.events.send(Event::Finished {
            report: report.clone(),
        });

        Ok(report)
    }

    /// Run one phase at concurrency K: warmup window + steady-state window.
    async fn run_phase(&self, phase_index: u32, k: u32, kind: &str) -> Result<PhaseRecord> {
        let warmup = self.cfg.warmup;
        let steady = self.cfg.steady_state;

        let _ = self.events.send(Event::PhaseStarted {
            phase_index,
            concurrent_users: k,
            kind: kind.to_string(),
            warmup,
            steady_state: steady,
        });

        let state: Arc<Mutex<PhaseState>> = Arc::new(Mutex::new(PhaseState::default()));
        let active: Arc<std::sync::atomic::AtomicU32> =
            Arc::new(std::sync::atomic::AtomicU32::new(0));
        let stop: Arc<std::sync::atomic::AtomicBool> =
            Arc::new(std::sync::atomic::AtomicBool::new(false));

        let phase_started = Instant::now();
        let warmup_deadline = phase_started + warmup;
        let phase_deadline = warmup_deadline + steady;

        // Spawn K worker tasks that submit requests in a tight loop until
        // `stop` is set. Each task pulls a clone of the vendor Arc + shared
        // state Arc.
        let mut workers = Vec::with_capacity(k as usize);
        for worker_id in 0..k {
            let vendor = self.vendor.clone();
            let state = state.clone();
            let active = active.clone();
            let stop = stop.clone();
            let max_tokens = self.cfg.max_tokens_per_request;
            workers.push(tokio::spawn(async move {
                let prompt = format!(
                    "Benchmark prompt — agentperf phase worker {worker_id}. \
                     Generate a multi-sentence answer suitable for measuring \
                     end-to-end output token throughput."
                );
                while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                    active.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let r = vendor.run_request(&prompt, max_tokens).await;
                    active.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    let now = Instant::now();
                    let (result, failure) = match r {
                        Ok(rr) => (rr, None),
                        // Keep the zeroed datum so the success/failure rates
                        // stay honest, but NEVER discard the cause: this is the
                        // only place the reason a benchmark request died is
                        // still in hand.
                        Err(e) => (
                            RequestResult {
                                ttft: Duration::ZERO,
                                total: Duration::ZERO,
                                output_tokens: 0,
                                ok: false,
                            },
                            Some(format!("{e:#}")),
                        ),
                    };
                    let mut s = state.lock().await;
                    if let Some(cause) = failure {
                        s.failed += 1;
                        if s.first_error.is_none() {
                            s.first_error = Some(cause);
                        }
                    }
                    s.all_results.push(result);
                    if now >= warmup_deadline && now <= phase_deadline {
                        s.steady_state_results.push(result);
                    }
                    drop(s);
                    if now >= phase_deadline {
                        break;
                    }
                }
            }));
        }

        // Tell the vendor what K we're at. Vendors with synthetic
        // degradation curves use this; real in-process vendors no-op.
        self.vendor.set_concurrent_users(k);

        // Progress emitter — every 200 ms, snapshot the running metrics and
        // emit a PhaseProgress event. Runs until `stop` is set.
        let progress_state = state.clone();
        let progress_active = active.clone();
        let progress_stop = stop.clone();
        let progress_events = self.events.clone();
        let progress = tokio::spawn(async move {
            loop {
                if progress_stop.load(std::sync::atomic::Ordering::SeqCst) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
                let s = progress_state.lock().await;
                let now = Instant::now();
                let in_warmup = now < warmup_deadline;
                let speeds: Vec<f64> = s
                    .all_results
                    .iter()
                    .map(|r| r.output_speed_tok_per_s())
                    .collect();
                let ttfts: Vec<f64> = s
                    .all_results
                    .iter()
                    .map(|r| r.ttft_seconds_for_slo())
                    .collect();
                let total_tokens: u64 = s
                    .all_results
                    .iter()
                    .filter(|r| r.ok)
                    .map(|r| r.output_tokens as u64)
                    .sum();
                let elapsed = phase_started.elapsed().as_secs_f64().max(0.001);
                let throughput = total_tokens as f64 / elapsed;
                let p25 = if speeds.is_empty() {
                    0.0
                } else {
                    percentile(&speeds, 0.25)
                };
                let p95 = if ttfts.is_empty() {
                    0.0
                } else {
                    percentile(&ttfts, 0.95)
                };
                let samples = s.steady_state_results.len() as u32;
                drop(s);

                let _ = progress_events.send(Event::PhaseProgress {
                    phase_index,
                    elapsed_in_phase: phase_started.elapsed(),
                    in_warmup,
                    steady_state_samples_so_far: samples,
                    running_p25_speed: p25,
                    running_p95_ttft: p95,
                    running_system_throughput_tok_per_s: throughput,
                    active_requests: progress_active.load(std::sync::atomic::Ordering::SeqCst),
                });
            }
        });

        // Sleep until the phase ends, then signal workers to stop.
        tokio::time::sleep_until(tokio::time::Instant::from_std(phase_deadline)).await;
        stop.store(true, std::sync::atomic::Ordering::SeqCst);

        // Join workers + progress.
        for w in workers {
            let _ = w.await;
        }
        let _ = progress.await;

        // Build the final phase record from steady-state samples.
        let final_state = state.lock().await;

        // A phase in which NOTHING succeeded is not a slow measurement, it is
        // an absent one. Reporting it as a record yields a full table of zeros
        // — a plausible-looking result that contains no measurement at all,
        // which is how a dead run gets written up as a finding. Fail loudly,
        // and carry the cause we captured rather than a summary of it.
        let succeeded = final_state.all_results.iter().filter(|r| r.ok).count();
        if succeeded == 0 {
            let cause = final_state
                .first_error
                .clone()
                .unwrap_or_else(|| "no request completed and no error was recorded".to_string());
            anyhow::bail!(
                "bench phase {phase_index} (K={k}, {kind}) produced no successful request: \
                 {attempts} attempted, {failed} failed. This is not a measurement -- \
                 refusing to report it as one. First error: {cause}",
                attempts = final_state.all_results.len(),
                failed = final_state.failed,
            );
        }
        let speeds: Vec<f64> = final_state
            .steady_state_results
            .iter()
            .map(|r| r.output_speed_tok_per_s())
            .collect();
        let ttfts: Vec<f64> = final_state
            .steady_state_results
            .iter()
            .map(|r| r.ttft_seconds_for_slo())
            .collect();
        let p25 = if speeds.is_empty() {
            0.0
        } else {
            percentile(&speeds, 0.25)
        };
        let p95 = if ttfts.is_empty() {
            f64::INFINITY
        } else {
            percentile(&ttfts, 0.95)
        };
        let evaluation = SloEvaluation::evaluate(p25, p95, self.tier);

        let record = PhaseRecord {
            phase_index,
            concurrent_users: k,
            kind: kind.to_string(),
            steady_state_samples: final_state.steady_state_results.len() as u32,
            evaluation,
            duration_seconds: phase_started.elapsed().as_secs_f64(),
        };

        Ok(record)
    }
}

// ---------------------------------------------------------------------------
// Pure binary-search helpers — exposed for unit tests so we can validate the
// search algorithm against synthetic SLO functions without spinning up async.
// ---------------------------------------------------------------------------

/// Compute the largest K in `[1, max_k]` for which `slo_passes(k)` returns
/// `true`, using exponential ramp + binary search. This is the **algorithm-
/// only** view of the scheduler — no events, no async, no vendor.
///
/// Returns `None` if even K=1 fails.
///
/// Exported for unit tests (and external probes) of the search algorithm,
/// independent of the async / vendor machinery.
#[allow(dead_code)]
pub fn binary_search_saturation<F>(max_k: u32, mut slo_passes: F) -> Option<u32>
where
    F: FnMut(u32) -> bool,
{
    if max_k == 0 {
        return None;
    }
    let mut last_pass: Option<u32> = None;
    let mut first_fail: Option<u32> = None;
    let mut k: u32 = 1;
    loop {
        let pass = slo_passes(k);
        if pass {
            last_pass = Some(k);
            if k >= max_k {
                break;
            }
            let next = k.saturating_mul(2).min(max_k);
            if next == k {
                break;
            }
            k = next;
        } else {
            first_fail = Some(k);
            break;
        }
    }
    if let (Some(lp), Some(ff)) = (last_pass, first_fail) {
        let mut lo = lp;
        let mut hi = ff;
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if slo_passes(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        last_pass = Some(lo);
    }
    last_pass
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_then_bisect_clean_monotone() {
        // SLO passes iff k <= 73. Expected saturation = 73.
        let sat = binary_search_saturation(256, |k| k <= 73);
        assert_eq!(sat, Some(73));
    }

    #[test]
    fn ramp_first_fail_at_k1() {
        // SLO fails everywhere. Expected: None.
        let sat = binary_search_saturation(256, |_| false);
        assert_eq!(sat, None);
    }

    #[test]
    fn ramp_never_fails_caps_at_max() {
        // SLO always passes. Should bottom out at max_k = 128.
        let sat = binary_search_saturation(128, |_| true);
        assert_eq!(sat, Some(128));
    }

    #[test]
    fn ramp_exactly_at_power_of_two() {
        // Boundary case: SLO passes iff k <= 16. Ramp visits 1,2,4,8,16,32.
        // 32 fails → bisect in [16, 32] → 24, 28, 30, 31. Final: 16.
        let sat = binary_search_saturation(256, |k| k <= 16);
        assert_eq!(sat, Some(16));
    }

    #[test]
    fn ramp_at_user_max_with_no_failure() {
        // Cap = 50; SLO always passes. After ramp: 1, 2, 4, 8, 16, 32, 50.
        // 50 passes and we've hit the cap → saturation = 50.
        let sat = binary_search_saturation(50, |_| true);
        assert_eq!(sat, Some(50));
    }

    #[test]
    fn ramp_at_user_max_with_failure_just_below() {
        // SLO passes iff k <= 7. Cap = 200. Ramp: 1,2,4 (pass), 8 (fail).
        // Bisect [4, 8] → 6 (pass) → [6,8] → 7 (pass) → [7,8] gap=1, done.
        // Final saturation = 7.
        let sat = binary_search_saturation(200, |k| k <= 7);
        assert_eq!(sat, Some(7));
    }

    #[test]
    fn ramp_overflow_safety_with_huge_cap() {
        // Cap = u32::MAX. SLO fails at k=2. Just need not to panic.
        let sat = binary_search_saturation(u32::MAX, |k| k <= 1);
        assert_eq!(sat, Some(1));
    }

    /// "Noisy SLO" — the function's behaviour around the boundary is jittery.
    /// We don't promise an exact reproducible answer in that regime; we
    /// promise the algorithm doesn't loop forever and returns a value within
    /// a few units of the true crossover. (This documents the bisect's
    /// expected behaviour — not idempotent on noisy SLOs, but bounded.)
    #[test]
    fn ramp_noisy_slo_still_terminates() {
        let true_crossover = 50;
        let mut calls = 0usize;
        let sat = binary_search_saturation(512, |k| {
            calls += 1;
            // Noise window of ±2 around the true crossover.
            let jitter = (calls as i64 * 7919) % 5 - 2;
            (k as i64) <= (true_crossover as i64 + jitter)
        });
        assert!(sat.is_some());
        let s = sat.unwrap();
        // Must terminate within O(log max_k) calls = ~20 for u32::MAX.
        assert!(calls < 100, "should not blow up on noisy SLO ({calls})");
        // Saturation should be in a reasonable neighbourhood of the
        // true crossover.
        assert!((32..=64).contains(&s), "noisy result out of band: {s}");
    }

    // ---------------------------------------------------------------------
    // A failed request must not be recorded as a silent zero.
    // ---------------------------------------------------------------------

    /// Vendor whose every request fails with a distinctive, layered cause.
    struct AlwaysFailsVendor;

    #[async_trait::async_trait]
    impl Vendor for AlwaysFailsVendor {
        fn name(&self) -> &str {
            "always-fails"
        }
        async fn run_request(&self, _prompt: &str, _max_tokens: u32) -> Result<RequestResult> {
            Err(
                anyhow::anyhow!("DriverError(CUDA_ERROR_ILLEGAL_INSTRUCTION)")
                    .context("engine returned InternalError"),
            )
        }
    }

    fn fast_cfg() -> SchedulerConfig {
        SchedulerConfig {
            max_users: 1,
            warmup: Duration::from_millis(10),
            steady_state: Duration::from_millis(40),
            max_tokens_per_request: 8,
            bisect_min_gap: 1,
        }
    }

    /// A phase in which every request fails must FAIL, and must carry the
    /// driver's own words out with it.
    ///
    /// Before this, `Err(_)` was mapped to a zeroed `RequestResult { ok: false }`
    /// and the cause was dropped on the floor, so a run where nothing worked
    /// produced a full table of zeros — a plausible-looking result containing
    /// no measurement. Same class as the KV-cache message that replaced a
    /// `DriverError` with a fixed string.
    #[tokio::test]
    async fn a_phase_where_every_request_fails_is_an_error_not_a_row_of_zeros() {
        let tier = SloTier::from_id(1).unwrap();
        let sched = Scheduler::new(Arc::new(AlwaysFailsVendor), fast_cfg(), tier);
        let err = sched
            .run_phase(0, 1, "test")
            .await
            .expect_err("a phase with zero successful requests must not return a PhaseRecord");
        let msg = format!("{err:#}");

        assert!(
            msg.contains("no successful request"),
            "error must say the phase produced no measurement: {msg}"
        );
        assert!(
            msg.contains("CUDA_ERROR_ILLEGAL_INSTRUCTION"),
            "the request's own cause must survive into the phase error, not be \
             replaced by a summary: {msg}"
        );
        assert!(
            msg.contains("engine returned InternalError"),
            "the full cause chain must survive, not just the innermost error: {msg}"
        );
    }

    /// Control: the assertion above discriminates. A vendor that works yields a
    /// record, so the test is not passing merely because everything errors.
    #[tokio::test]
    async fn a_phase_with_working_requests_still_returns_a_record() {
        let vendor = Arc::new(crate::bench::vendor::MockVendor::new(
            crate::bench::vendor::MockVendorConfig::default(),
        ));
        let tier = SloTier::from_id(1).unwrap();
        let sched = Scheduler::new(vendor, fast_cfg(), tier);
        let rec = sched
            .run_phase(0, 1, "test")
            .await
            .expect("a phase with successful requests must return a record");
        assert_eq!(rec.concurrent_users, 1);
    }
}
