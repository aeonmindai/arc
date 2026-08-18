//! Parent system: ArcGate (build-time correctness gate for ArcKernels)
//!
//! Guards the fix for a silent-failure bug: `build.rs` discovers CUDA kernels
//! with the glob `kernels/*/*.cu`, but a glob is evaluated only when the build
//! script RUNS, and Cargo re-runs a build script only for the paths it named
//! via `rerun-if-changed`. `cudaforge` names one path per *already-resolved*
//! source, so a kernel that does not exist yet can never be named. A newly
//! added `.cu` therefore did not re-run the script, was never compiled, and
//! the build was still green -- `extern "C"` declarations fail only at LINK
//! time and an rlib build does not link.
//!
//! The fix is two lines in `build.rs`: watch the glob root as a directory, and
//! hard-fail when the discovered kernel count leaves its checked-in value.
//! Both live inside `#[cfg(feature = "cuda")]`, so they are invisible to every
//! CI lane that has no CUDA -- which is why these checks are assertions over
//! `build.rs` as TEXT. They run everywhere, GPU-free, and they go red the
//! moment the trigger is deleted.
//!
//! A fix to a silent-failure bug must not itself be able to fail silently.

const BUILD_RS: &str = include_str!("../build.rs");

const EXPECTED_COUNT_FILE: &str = "kernels/EXPECTED_KERNEL_COUNT";

fn manifest_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The value of a `const NAME: &str = "...";` declared in `build.rs`.
fn build_rs_str_const(name: &str) -> String {
    let needle = format!("const {name}: &str = \"");
    let start = BUILD_RS
        .find(&needle)
        .unwrap_or_else(|| panic!("build.rs no longer declares `const {name}: &str`"))
        + needle.len();
    let rest = &BUILD_RS[start..];
    let end = rest
        .find('"')
        .unwrap_or_else(|| panic!("unterminated string literal for `{name}` in build.rs"));
    rest[..end].to_string()
}

/// Every `rerun-if-changed=` target emitted by `build.rs`, with `{CONST}`
/// interpolations resolved. Comment lines are skipped so that prose *about*
/// the directive can never be mistaken for the directive itself.
fn rerun_if_changed_targets() -> Vec<String> {
    let glob_root = build_rs_str_const("KERNEL_GLOB_ROOT");
    BUILD_RS
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with("//"))
        .filter_map(|line| {
            let idx = line.find("rerun-if-changed=")?;
            let rest = &line[idx + "rerun-if-changed=".len()..];
            let end = rest.find('"')?;
            Some(rest[..end].replace("{KERNEL_GLOB_ROOT}", &glob_root))
        })
        .collect()
}

/// `.cu` files exactly one directory below `kernels/` -- i.e. precisely what
/// `kernels/*/*.cu` matches.
fn discover_kernel_sources() -> Vec<std::path::PathBuf> {
    let root = manifest_dir().join(build_rs_str_const("KERNEL_GLOB_ROOT"));
    let mut found = Vec::new();
    for entry in std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", root.display()))
        .flatten()
    {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let Ok(inner) = std::fs::read_dir(entry.path()) else {
            continue;
        };
        for file in inner.flatten() {
            let path = file.path();
            if path.extension().is_some_and(|e| e == "cu") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// Every `.cu` under `kernels/`, at any depth.
fn all_kernel_sources_any_depth(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            all_kernel_sources_any_depth(&path, out);
        } else if path.extension().is_some_and(|e| e == "cu") {
            out.push(path);
        }
    }
}

fn expected_count() -> usize {
    let path = manifest_dir().join(EXPECTED_COUNT_FILE);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    raw.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .and_then(|l| l.parse::<usize>().ok())
        .unwrap_or_else(|| panic!("{} has no bare integer line", path.display()))
}

/// THE regression test. Delete the `rerun-if-changed` for the kernel directory
/// from `build.rs` and this goes red.
#[test]
fn build_rs_watches_the_kernel_glob_root() {
    let glob_root = build_rs_str_const("KERNEL_GLOB_ROOT");
    let targets = rerun_if_changed_targets();

    assert!(
        targets.iter().any(|t| t == &glob_root),
        "build.rs must emit `cargo::rerun-if-changed={glob_root}` -- naming the kernel \
         DIRECTORY, not individual files.\n\
         Without it Cargo never re-runs build.rs when a `.cu` is added under \
         `{glob_root}/`, the `{glob_root}/*/*.cu` glob is never re-evaluated, the new \
         kernel is never compiled, and the build is STILL GREEN.\n\
         Emitted rerun-if-changed targets were: {targets:?}"
    );

    // A per-file trigger is NOT a substitute: a file that does not exist yet
    // cannot be named. Catch anyone "fixing" this by listing kernels instead.
    assert!(
        !targets
            .iter()
            .any(|t| t.starts_with(&format!("{glob_root}/"))),
        "build.rs names individual paths under `{glob_root}/`. That cannot detect a \
         NEWLY ADDED kernel -- watch the directory instead. Targets: {targets:?}"
    );
}

/// Goes red when a kernel is added or removed without updating the checked-in
/// count, which is what makes the build-time gate in `build.rs` meaningful.
#[test]
fn expected_kernel_count_matches_disk() {
    let discovered = discover_kernel_sources();
    let expected = expected_count();
    assert_eq!(
        discovered.len(),
        expected,
        "`{EXPECTED_COUNT_FILE}` says {expected} CUDA kernel sources but \
         `kernels/*/*.cu` matched {}.\nIf the change is intended, update that file.\n\
         Discovered:\n{}",
        discovered.len(),
        discovered
            .iter()
            .map(|p| format!("  {}", p.display()))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    assert!(
        expected > 0,
        "expected kernel count is 0 -- the guard would pass against an empty build"
    );
}

/// The build-time gate is only real if `build.rs` actually consults the file.
#[test]
fn build_rs_consults_the_expected_count_file() {
    assert!(
        BUILD_RS.contains(EXPECTED_COUNT_FILE),
        "build.rs no longer references `{EXPECTED_COUNT_FILE}`; the checked-in kernel \
         count is orphaned and gates nothing"
    );
    // Match the CALL, not the substring: `fn assert_kernel_set_intact() {` also
    // contains the name, so a `contains()` here would stay green after the call
    // site was deleted -- the exact silent-pass this whole file exists to stop.
    assert!(
        BUILD_RS
            .lines()
            .map(str::trim)
            .any(|l| l == "assert_kernel_set_intact();"),
        "build.rs no longer CALLS `assert_kernel_set_intact();` (defining it is not \
         enough); an added or dropped kernel is silent again"
    );
}

/// `kernels/*/*.cu` does not recurse. A kernel at `kernels/a/b/c.cu` is
/// silently skipped by the glob, which is the same class of bug one level down.
#[test]
fn no_kernel_is_nested_below_the_glob_reach() {
    let root = manifest_dir().join(build_rs_str_const("KERNEL_GLOB_ROOT"));
    let mut all = Vec::new();
    all_kernel_sources_any_depth(&root, &mut all);
    all.sort();

    let reachable = discover_kernel_sources();
    let missed: Vec<_> = all
        .iter()
        .filter(|p| !reachable.contains(p))
        .map(|p| p.display().to_string())
        .collect();

    assert!(
        missed.is_empty(),
        "these `.cu` files are under `kernels/` but are NOT matched by \
         `kernels/*/*.cu`, so they are silently excluded from the CUDA build:\n  {}\n\
         Move them exactly one directory below `kernels/`, or widen the glob in \
         build.rs (and update {EXPECTED_COUNT_FILE}).",
        missed.join("\n  ")
    );
    assert!(
        !all.is_empty(),
        "no `.cu` found under {} -- this test would pass vacuously",
        root.display()
    );
}
