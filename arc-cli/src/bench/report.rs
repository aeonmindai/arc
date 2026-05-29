//! Output writers for `arc bench --suite agentperf`.
//!
//! The harness writes two files when `--output <path>` is passed:
//!
//!   * `<path>.json` — the structured `BenchReport`. Stable schema for CI.
//!   * `<path>.md`  — a human-readable summary, suitable for pasting into
//!                    PRs / Linear comments.
//!
//! If the user passes an explicit extension (`.json` or `.md`), we honor it
//! and write only that file. Otherwise we write both.

use crate::bench::scheduler::ScheduleReport;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub model: String,
    pub vendor_name: String,
    pub suite: String,
    pub mode: String, // "mock" | "gpu"
    pub schedule: ScheduleReport,
}

/// Resolve the output stem. If `path` ends with `.json` or `.md`, we use it
/// verbatim. Otherwise we treat it as a stem and append both extensions.
fn resolve_paths(path: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    match path.extension().and_then(|s| s.to_str()) {
        Some("json") => (Some(path.to_path_buf()), None),
        Some("md") => (None, Some(path.to_path_buf())),
        _ => {
            let mut json = path.to_path_buf();
            json.set_extension("json");
            let mut md = path.to_path_buf();
            md.set_extension("md");
            (Some(json), Some(md))
        }
    }
}

/// Ensure the parent directory of `p` exists.
fn ensure_parent(p: &Path) -> Result<()> {
    if let Some(parent) = p.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create dir {}", parent.display()))?;
        }
    }
    Ok(())
}

pub fn write_reports(report: &BenchReport, path: &Path) -> Result<Vec<PathBuf>> {
    let (json_p, md_p) = resolve_paths(path);
    let mut written = Vec::new();
    if let Some(p) = json_p {
        ensure_parent(&p)?;
        let body = serde_json::to_string_pretty(report)
            .context("failed to serialise bench report to JSON")?;
        std::fs::write(&p, body)
            .with_context(|| format!("failed to write JSON report to {}", p.display()))?;
        written.push(p);
    }
    if let Some(p) = md_p {
        ensure_parent(&p)?;
        std::fs::write(&p, render_markdown(report))
            .with_context(|| format!("failed to write markdown report to {}", p.display()))?;
        written.push(p);
    }
    Ok(written)
}

/// Render a compact markdown summary. Designed for PR descriptions: header,
/// SLO summary, then a phase table.
pub fn render_markdown(report: &BenchReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# arc bench — {} ({})\n\n",
        report.suite, report.mode
    ));
    s.push_str(&format!("- **Model**: `{}`\n", report.model));
    s.push_str(&format!("- **Vendor**: `{}`\n", report.vendor_name));
    s.push_str(&format!(
        "- **SLO tier**: {} (P25 output ≥ {:.0} tok/s, P95 TTFT ≤ {:.2} s)\n",
        report.schedule.tier.tier,
        report.schedule.tier.min_p25_output_speed_tok_per_s,
        report.schedule.tier.max_p95_ttft_seconds
    ));
    s.push_str(&format!(
        "- **Max users explored / cap**: {} / {}\n",
        report.schedule.max_users_explored, report.schedule.max_users_cap
    ));
    match report.schedule.saturation_users {
        Some(k) => s.push_str(&format!(
            "- **Saturation point**: **{k} concurrent users** (largest K passing tier {})\n",
            report.schedule.tier.tier
        )),
        None => s.push_str("- **Saturation point**: FAIL — even K=1 did not pass the SLO\n"),
    }
    s.push_str(&format!(
        "- **Total wall time**: {:.1} s\n\n",
        report.schedule.total_wall_seconds
    ));
    s.push_str("## Phases\n\n");
    s.push_str("| # | K | kind | P25 out tok/s | P95 TTFT (s) | samples | dur (s) | pass |\n");
    s.push_str("|---|---|------|---------------|--------------|---------|---------|------|\n");
    for p in &report.schedule.phases {
        let pass = if p.evaluation.overall_pass {
            "PASS"
        } else {
            "FAIL"
        };
        s.push_str(&format!(
            "| {} | {} | {} | {:.1} | {:.2} | {} | {:.1} | {pass} |\n",
            p.phase_index,
            p.concurrent_users,
            p.kind,
            p.evaluation.p25_output_speed_tok_per_s,
            p.evaluation.p95_ttft_seconds,
            p.steady_state_samples,
            p.duration_seconds
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::scheduler::PhaseRecord;
    use crate::bench::slo::{SloEvaluation, SloTier};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("arc-bench-test-{pid}-{n}"))
    }

    fn synth_report() -> BenchReport {
        let tier = SloTier::from_id(2).unwrap();
        let phase = PhaseRecord {
            phase_index: 1,
            concurrent_users: 1,
            kind: "ramp".into(),
            steady_state_samples: 30,
            evaluation: SloEvaluation::evaluate(120.0, 0.5, tier),
            duration_seconds: 60.0,
        };
        BenchReport {
            model: "deepseek-ai/DeepSeek-V4-Flash".into(),
            vendor_name: "mock".into(),
            suite: "agentperf".into(),
            mode: "mock".into(),
            schedule: crate::bench::scheduler::ScheduleReport {
                tier,
                max_users_explored: 4,
                max_users_cap: 256,
                saturation_users: Some(2),
                phases: vec![phase],
                total_wall_seconds: 60.5,
            },
        }
    }

    #[test]
    fn resolve_paths_explicit_json() {
        let (j, m) = resolve_paths(Path::new("/tmp/foo.json"));
        assert_eq!(j.as_deref(), Some(Path::new("/tmp/foo.json")));
        assert!(m.is_none());
    }

    #[test]
    fn resolve_paths_explicit_md() {
        let (j, m) = resolve_paths(Path::new("/tmp/foo.md"));
        assert!(j.is_none());
        assert_eq!(m.as_deref(), Some(Path::new("/tmp/foo.md")));
    }

    #[test]
    fn resolve_paths_stem_produces_both() {
        let (j, m) = resolve_paths(Path::new("/tmp/foo"));
        assert_eq!(j.as_deref(), Some(Path::new("/tmp/foo.json")));
        assert_eq!(m.as_deref(), Some(Path::new("/tmp/foo.md")));
    }

    #[test]
    fn markdown_contains_phase_row() {
        let r = synth_report();
        let md = render_markdown(&r);
        assert!(md.contains("deepseek-ai/DeepSeek-V4-Flash"));
        assert!(md.contains("Saturation point"));
        assert!(md.contains("PASS"));
        assert!(md.contains("| 1 | 1 | ramp"));
    }

    #[test]
    fn write_reports_round_trips_json() {
        let stem = unique_tmp();
        let r = synth_report();
        let paths = write_reports(&r, &stem).unwrap();
        assert_eq!(paths.len(), 2);
        let json_path = paths
            .iter()
            .find(|p| p.extension().unwrap() == "json")
            .unwrap();
        let parsed: BenchReport =
            serde_json::from_str(&std::fs::read_to_string(json_path).unwrap()).unwrap();
        assert_eq!(parsed.model, r.model);
        assert_eq!(parsed.schedule.tier.tier, r.schedule.tier.tier);
        assert_eq!(
            parsed.schedule.saturation_users,
            r.schedule.saturation_users
        );
        // Cleanup.
        for p in paths {
            let _ = std::fs::remove_file(p);
        }
    }
}
