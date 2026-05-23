//! Generate the bundled `agentperf_tuning` trajectory dataset.
//!
//! Run with: `cargo run -p arc-bench --bin generate-agentperf-tuning`.
//!
//! This binary is a *build tool*, not part of the public API. Its job
//! is to materialise the 50 trajectory JSON files under
//! `arc-bench/datasets/agentperf_tuning/trajectories/` from the
//! source-of-truth tables in `arc_bench::dataset_authoring`. The
//! resulting JSON files are committed to the repo so consumers do not
//! need to re-run the generator.

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let crate_root: PathBuf = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("arc-bench"));
    let out_dir = crate_root
        .join("datasets")
        .join("agentperf_tuning")
        .join("trajectories");
    std::fs::create_dir_all(&out_dir)?;

    let trajectories = arc_bench::dataset_authoring::all_tuning_trajectories();
    eprintln!(
        "writing {} trajectories to {}",
        trajectories.len(),
        out_dir.display()
    );

    for trajectory in &trajectories {
        let filename = format!("{}.json", trajectory.id);
        let path = out_dir.join(&filename);
        let json = serde_json::to_string_pretty(trajectory)?;
        std::fs::write(&path, json)?;
    }

    eprintln!("ok");
    Ok(())
}
