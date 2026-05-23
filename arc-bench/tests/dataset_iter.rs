//! Dataset-loader / iteration tests.

use arc_bench::Dataset;
use std::collections::BTreeSet;
use std::path::PathBuf;

fn dataset_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("datasets/agentperf_tuning")
}

#[test]
fn loader_returns_exactly_fifty_trajectories() {
    let ds = Dataset::load(dataset_root()).expect("load");
    assert_eq!(ds.len(), 50);
    assert!(!ds.is_empty());
    assert_eq!(ds.name, "agentperf_tuning");
}

#[test]
fn covers_all_twelve_languages() {
    let ds = Dataset::load(dataset_root()).expect("load");
    let per_lang = ds.per_language();
    let expected: BTreeSet<&str> = [
        "rust",
        "python",
        "typescript",
        "go",
        "cpp",
        "java",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "scala",
        "csharp",
    ]
    .into_iter()
    .collect();
    let actual: BTreeSet<&str> = per_lang.keys().map(|k| k.as_str()).collect();
    assert_eq!(actual, expected, "missing or extra language: {:?}", per_lang);
    for (lang, count) in &per_lang {
        assert!(
            *count >= 3 && *count <= 5,
            "language {lang} has {count} trajectories (expected 3-5)"
        );
    }
}

#[test]
fn length_distribution_has_at_least_five_long_trajectories() {
    let ds = Dataset::load(dataset_root()).expect("load");
    let dist = ds.length_distribution();
    assert!(
        dist.long >= 5,
        "expected >=5 long trajectories, got {:?}",
        dist
    );
    assert!(
        dist.short + dist.medium + dist.long == ds.len() as u32,
        "bucket counts don't add up"
    );
    // Distribution should be skewed toward short/medium per
    // AA-AgentPerf, but we cannot pin this too tightly.
    assert!(dist.short + dist.medium >= 30);
}

#[test]
fn iteration_is_id_sorted() {
    let ds = Dataset::load(dataset_root()).expect("load");
    let ids: Vec<&str> = ds.iter().map(|t| t.id.as_str()).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted, "iteration order must be id-sorted");
}
