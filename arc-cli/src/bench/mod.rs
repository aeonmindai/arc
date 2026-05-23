//! `arc bench --suite agentperf` — the new TUI bench harness.
//!
//! See module-level rustdoc on each submodule for design details:
//!
//! - [`slo`]: tier definitions + percentile math.
//! - [`vendor`]: trait + MockVendor (test) + InProcessVendor (cuda).
//! - [`scheduler`]: exponential-ramp + binary-search engine, broadcasts
//!    `Event`s for the TUI.
//! - [`tui`]: ratatui dashboard.
//! - [`report`]: JSON + markdown writers.
//!
//! The public entry point is [`run`], which `main.rs` calls when the user
//! invokes `arc bench`.

pub mod report;
pub mod scheduler;
pub mod slo;
pub mod tui;
pub mod vendor;

use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use scheduler::{Scheduler, SchedulerConfig};
use slo::SloTier;
use vendor::{MockVendor, MockVendorConfig, Vendor};

/// Bench suite — only `Agentperf` for now, but the enum lets the CLI
/// support `--suite simple` later by routing to `mistralrs bench` or
/// equivalent legacy paths without forcing the agentperf flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Suite {
    Agentperf,
    Simple,
}

impl Suite {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "agentperf" | "aa-agentperf" => Ok(Self::Agentperf),
            "simple" | "legacy" => Ok(Self::Simple),
            other => Err(anyhow!("unknown bench suite `{other}`")),
        }
    }
}

/// Parsed CLI options for `arc bench`.
#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub suite: Suite,
    pub model: String,
    pub slo_tier: u8,
    pub mock: bool,
    pub max_users: u32,
    pub output: PathBuf,
    /// Override the default 30s warmup / 30s steady-state (for `--mock`
    /// integration tests which would otherwise take minutes).
    pub warmup: Duration,
    pub steady_state: Duration,
    /// Disable the TUI even on a TTY. Used by integration tests so they
    /// don't try to grab the terminal.
    pub headless: bool,
}

impl BenchOptions {
    /// Construct the canonical AA-AgentPerf defaults: tier 2, 256 max users,
    /// 30s warmup + 30s steady-state. Useful for callers that want to start
    /// from defaults and tweak one or two knobs. Not used by the CLI path
    /// (which builds `BenchOptions` from clap-parsed flags directly), so
    /// dead-code is allowed for this constructor.
    #[allow(dead_code)]
    pub fn agentperf_defaults(model: String) -> Self {
        Self {
            suite: Suite::Agentperf,
            model,
            slo_tier: 2,
            mock: false,
            max_users: 256,
            output: PathBuf::from("tests/results/agentperf.json"),
            warmup: Duration::from_secs(30),
            steady_state: Duration::from_secs(30),
            headless: false,
        }
    }
}

/// Top-level entry point. Returns the process exit code.
pub fn run(opts: BenchOptions) -> Result<i32> {
    match opts.suite {
        Suite::Agentperf => run_agentperf(opts),
        Suite::Simple => Err(anyhow!(
            "the `simple` suite is intentionally not implemented in this binary; \
             run `mistralrs bench` directly for the legacy single-process loop."
        )),
    }
}

fn run_agentperf(opts: BenchOptions) -> Result<i32> {
    let tier = SloTier::from_id(opts.slo_tier).ok_or_else(|| {
        anyhow!(
            "unknown SLO tier `{}`. Supported: 1 (strictest), 2, 3, 4 (loosest).",
            opts.slo_tier
        )
    })?;
    let cfg = SchedulerConfig {
        max_users: opts.max_users,
        warmup: opts.warmup,
        steady_state: opts.steady_state,
        ..Default::default()
    };

    let vendor: Arc<dyn Vendor> = if opts.mock {
        Arc::new(MockVendor::new(MockVendorConfig::default()))
    } else {
        #[cfg(feature = "cuda")]
        {
            // Real path — load the model into mistralrs-core in-process.
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("failed to build tokio runtime for vendor init")?;
            let v: Arc<dyn Vendor> = rt.block_on(async {
                let v = vendor::in_process::InProcessVendor::new(&opts.model, None)
                    .await
                    .context("failed to construct InProcessVendor")?;
                anyhow::Ok(Arc::new(v) as Arc<dyn Vendor>)
            })?;
            // Re-use the runtime for the scheduler too — drop after `run`.
            return run_with_runtime(rt, v, cfg, tier, &opts);
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err(anyhow!(
                "real-vendor bench requires `--features cuda`. Pass `--mock` for the offline / CI path."
            ));
        }
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build tokio runtime for bench")?;
    run_with_runtime(rt, vendor, cfg, tier, &opts)
}

fn run_with_runtime(
    rt: tokio::runtime::Runtime,
    vendor: Arc<dyn Vendor>,
    cfg: SchedulerConfig,
    tier: SloTier,
    opts: &BenchOptions,
) -> Result<i32> {
    let model_id = opts.model.clone();
    let output = opts.output.clone();
    let suite_name = match opts.suite {
        Suite::Agentperf => "agentperf",
        Suite::Simple => "simple",
    };
    let mode = if opts.mock { "mock" } else { "gpu" };
    let headless = opts.headless;

    let exit_code = rt.block_on(async move {
        let sched = Scheduler::new(vendor.clone(), cfg, tier);
        let rx = sched.subscribe();
        let vendor_name = sched.vendor_name();
        let sched_arc = Arc::new(sched);

        // Spawn the TUI (unless headless).
        let model_for_tui = model_id.clone();
        let vendor_for_tui = vendor_name.clone();
        let tui_handle: Option<tokio::task::JoinHandle<Result<()>>> = if headless {
            // In headless mode, drain events to a small console summary so
            // we still print something humans can read.
            let mut rx2 = rx;
            Some(tokio::spawn(async move {
                use scheduler::Event;
                while let Ok(ev) = rx2.recv().await {
                    match ev {
                        Event::PhaseStarted {
                            phase_index,
                            concurrent_users,
                            kind,
                            ..
                        } => {
                            eprintln!("[bench] phase #{phase_index} kind={kind} K={concurrent_users}");
                        }
                        Event::PhaseFinished { record } => {
                            eprintln!(
                                "[bench]   -> K={} {} P25={:.1} P95={:.2} samples={} pass={}",
                                record.concurrent_users,
                                record.kind,
                                record.evaluation.p25_output_speed_tok_per_s,
                                record.evaluation.p95_ttft_seconds,
                                record.steady_state_samples,
                                record.evaluation.overall_pass
                            );
                        }
                        Event::Finished { report } => {
                            eprintln!(
                                "[bench] DONE — saturation={:?} explored_max={}",
                                report.saturation_users, report.max_users_explored
                            );
                            break;
                        }
                        _ => {}
                    }
                }
                Ok(())
            }))
        } else {
            Some(tokio::spawn(async move {
                tui::run_tui(&model_for_tui, &vendor_for_tui, rx).await
            }))
        };

        // Run the scheduler to completion.
        let report = sched_arc.run().await?;

        // Wait for the TUI to finish drawing (it'll exit on Finished event).
        if let Some(h) = tui_handle {
            let _ = h.await;
        }

        // Write reports.
        let bench_report = report::BenchReport {
            model: model_id.clone(),
            vendor_name: vendor_name.clone(),
            suite: suite_name.to_string(),
            mode: mode.to_string(),
            schedule: report.clone(),
        };
        let written = report::write_reports(&bench_report, &output)?;

        eprintln!();
        eprintln!("arc bench — agentperf complete");
        eprintln!("  model:           {}", model_id);
        eprintln!("  vendor:          {}", vendor_name);
        eprintln!("  SLO tier:        {}", tier.tier);
        eprintln!("  phases:          {}", report.phases.len());
        eprintln!(
            "  max users explored / cap: {} / {}",
            report.max_users_explored, report.max_users_cap
        );
        match report.saturation_users {
            Some(k) => {
                eprintln!("  saturation:      K={k} (largest concurrent users passing tier {})", tier.tier);
            }
            None => {
                eprintln!("  saturation:      FAIL — even K=1 did not pass tier {}", tier.tier);
            }
        }
        eprintln!("  reports written:");
        for p in &written {
            eprintln!("    {}", p.display());
        }

        let exit = if report.saturation_users.is_some() { 0 } else { 1 };
        anyhow::Ok(exit)
    })?;

    Ok(exit_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_parse_known_and_unknown() {
        assert_eq!(Suite::parse("agentperf").unwrap(), Suite::Agentperf);
        assert_eq!(Suite::parse("AGENTPERF").unwrap(), Suite::Agentperf);
        assert_eq!(Suite::parse("simple").unwrap(), Suite::Simple);
        assert!(Suite::parse("garbage").is_err());
    }

    #[test]
    fn agentperf_defaults_have_tier2_and_256_max() {
        let opts = BenchOptions::agentperf_defaults("foo".into());
        assert_eq!(opts.suite, Suite::Agentperf);
        assert_eq!(opts.slo_tier, 2);
        assert_eq!(opts.max_users, 256);
    }

    /// End-to-end mock integration test. We use very short warmup / steady
    /// windows so the whole thing runs in < 5 s.
    #[test]
    fn mock_end_to_end_produces_json_and_md() {
        let stem = std::env::temp_dir()
            .join(format!("arc-bench-integration-{}", std::process::id()));
        let json_path = stem.with_extension("json");
        let md_path = stem.with_extension("md");
        let _ = std::fs::remove_file(&json_path);
        let _ = std::fs::remove_file(&md_path);

        let opts = BenchOptions {
            suite: Suite::Agentperf,
            model: "deepseek-ai/DeepSeek-V4-Flash".into(),
            slo_tier: 2,
            mock: true,
            max_users: 32,
            output: stem.clone(),
            warmup: Duration::from_millis(100),
            steady_state: Duration::from_millis(200),
            headless: true, // do NOT grab the terminal during tests
        };
        let exit = run(opts).expect("bench run should succeed");
        assert!(
            exit == 0 || exit == 1,
            "exit must be 0 (pass) or 1 (no saturation), got {exit}"
        );
        assert!(json_path.exists(), "JSON report missing at {}", json_path.display());
        assert!(md_path.exists(), "markdown report missing at {}", md_path.display());

        let parsed: report::BenchReport =
            serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
        assert_eq!(parsed.mode, "mock");
        assert_eq!(parsed.suite, "agentperf");
        assert!(!parsed.schedule.phases.is_empty(), "must have at least one phase");

        // Cleanup.
        let _ = std::fs::remove_file(json_path);
        let _ = std::fs::remove_file(md_path);
    }
}
