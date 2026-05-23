//! Schema validation across every committed trajectory JSON file.
//!
//! These tests intentionally do not depend on the `dataset_authoring`
//! module — they read the on-disk JSON files just like a downstream
//! consumer would, so a regression in the schema (e.g. missing field,
//! wrong tool_call_id) is caught even if `dataset_authoring` happens
//! to compile.

use arc_bench::{Dataset, Role, Trajectory};
use std::path::PathBuf;

fn dataset_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join("agentperf_tuning")
}

#[test]
fn all_fifty_trajectories_parse_and_validate() {
    let root = dataset_root();
    let traj_dir = root.join("trajectories");
    let mut count = 0;
    let mut total_input = 0u64;
    let mut total_output = 0u64;
    for entry in std::fs::read_dir(&traj_dir).unwrap_or_else(|e| {
        panic!(
            "expected trajectory dir at {} (have you run the generator? cargo run -p arc-bench --bin generate-agentperf-tuning): {}",
            traj_dir.display(),
            e
        )
    }) {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        let traj = Trajectory::from_path(&path)
            .unwrap_or_else(|e| panic!("parsing {}: {}", path.display(), e));
        // Lightweight sanity asserts on every trajectory.
        assert!(!traj.id.is_empty(), "{} empty id", path.display());
        assert!(!traj.language.is_empty(), "{} empty lang", path.display());
        assert!(!traj.source.repo.is_empty(), "{} empty repo", path.display());
        assert!(
            !traj.source.license.is_empty(),
            "{} empty license",
            path.display()
        );
        assert!(
            traj.turns.iter().any(|t| matches!(t.role, Role::Assistant)),
            "{} has no assistant turn",
            path.display()
        );
        total_input += traj.total_input_tokens_est();
        total_output += traj.total_output_tokens_est();
        count += 1;
    }
    assert_eq!(count, 50, "expected exactly 50 trajectories");
    // Sanity: total input should dwarf total output (long contexts +
    // short replies match AA-AgentPerf's OSL median ~150).
    assert!(
        total_input > 5 * total_output,
        "expected long-context regime (input>>output); got input={} output={}",
        total_input,
        total_output
    );
}

#[test]
fn token_distribution_within_plausible_bounds() {
    let dataset = Dataset::load(dataset_root()).expect("load dataset");
    let (input, output) = dataset.token_totals();
    // Per-trajectory upper bound is generous: 131K * 50 = ~6.5M
    // tokens; with our padded representative trajectories the
    // bundled subset should comfortably sit below 1M.
    assert!(input < 6_500_000, "input total too high: {}", input);
    assert!(output < 1_000_000, "output total too high: {}", output);
    // And bounded below — every trajectory carries a system-prompt
    // overhead of 600 tokens for the user turn alone, so a 50-trajectory
    // subset should clear ~30K input tokens.
    assert!(input > 30_000, "input total too low: {}", input);
    assert!(output > 1_000, "output total too low: {}", output);
}

#[test]
fn known_distinct_ids() {
    let dataset = Dataset::load(dataset_root()).expect("load dataset");
    let mut ids: Vec<_> = dataset.trajectories.iter().map(|t| t.id.clone()).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), dataset.len(), "trajectory ids must be unique");
}
