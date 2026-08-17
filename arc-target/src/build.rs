//! The build-script half of ArcTarget.
//!
//! Two jobs, in this order:
//!
//! 1. Turn `ARC_CUDA_ARCHS` into the `-gencode` arguments that make nvcc emit a
//!    fat binary (`cudaforge` emits exactly one target on its own, so every
//!    additional architecture has to be appended explicitly).
//! 2. **Verify the archive that came out.** `cuobjdump --list-elf` is run
//!    against the produced static library and every requested architecture must
//!    appear, or the build fails.
//!
//! Step 2 is the part that is not optional. On 2026-08-17 a build on a live
//! H200 asked for `90,100,103`, exited 0, and produced an archive containing
//! only `sm_90a`. Nothing failed; the compile passing was read as the arches
//! being present. An assertion here is the difference between shipping a
//! Blackwell claim and shipping Blackwell code.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{parse_arch_list, render_arch_list, ArchError, CudaArch};

/// Environment variable naming the architectures to build.
///
/// Comma-separated, e.g. `ARC_CUDA_ARCHS=80,90,100,103`. Unset means "whatever
/// this box's GPU is", which is the right default for a rental where every
/// extra architecture is a full recompile of every `.cu` — and the wrong
/// default for anything published, which is why release builds set it.
pub const ARCHS_ENV: &str = "ARC_CUDA_ARCHS";

/// Environment variable a build script exports recording the architectures the
/// produced archive was **observed** to contain.
///
/// Observed, not requested: the runtime's report is derived from `cuobjdump`
/// output, so it cannot inherit a false claim from the build's intent.
pub const BUILT_ENV: &str = "ARC_CUDA_ARCHS_BUILT";

/// What `cuobjdump` was able to tell us about an archive.
///
/// Three states on purpose. `Unavailable` is not a pass and not a failure —
/// collapsing it into either is the bug this module exists to prevent, one
/// level up.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Observation {
    /// `cuobjdump` ran and listed at least one cubin.
    Observed(Vec<CudaArch>),
    /// `cuobjdump` could not be run, or ran and listed nothing.
    Unavailable(String),
}

/// Read [`ARCHS_ENV`], emitting the `rerun-if-env-changed` line for it.
///
/// `Ok(None)` means the variable is unset: the build keeps cudaforge's
/// single-architecture autodetect and makes no multi-arch claim.
pub fn requested_archs() -> Result<Option<Vec<CudaArch>>, ArchError> {
    println!("cargo:rerun-if-env-changed={ARCHS_ENV}");
    match std::env::var(ARCHS_ENV) {
        Err(_) => Ok(None),
        Ok(v) if v.trim().is_empty() => Ok(None),
        Ok(v) => parse_arch_list(&v).map(Some),
    }
}

/// Split a requested list into the one target `cudaforge` will emit itself and
/// the rest, which have to be appended as extra `-gencode` arguments.
///
/// The primary is the first target cudaforge can express verbatim. If it can
/// express none, this returns `Err` rather than letting cudaforge quietly fall
/// back to `nvidia-smi` autodetect and build for something nobody asked for.
pub fn split_primary(archs: &[CudaArch]) -> Result<(String, Vec<String>), ArchError> {
    let primary_idx = archs
        .iter()
        .position(|a| a.cudaforge_arch().is_some())
        .ok_or_else(|| ArchError::FamilyTargetNotRequestable(render_arch_list(archs)))?;
    let primary = archs[primary_idx]
        .cudaforge_arch()
        .expect("position() only matches expressible targets");
    let extra = archs
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != primary_idx)
        .map(|(_, a)| a.gencode())
        .collect();
    Ok((primary, extra))
}

/// Parse `cuobjdump --list-elf` output into the architectures it names.
///
/// The tool prints one line per cubin, of the shape
/// `ELF file    1: qtip_grouped_gemm.1.sm_90a.cubin`. Rather than pin the
/// column layout, this scans each line for an `sm_<digits>[a|f]` token, which
/// is the part whose spelling nvcc actually guarantees.
pub fn parse_list_elf(stdout: &str) -> Vec<CudaArch> {
    let mut out: Vec<CudaArch> = Vec::new();
    for line in stdout.lines() {
        let mut rest = line;
        while let Some(pos) = rest.find("sm_") {
            let tail = &rest[pos + 3..];
            let end = tail
                .find(|c: char| !c.is_ascii_alphanumeric())
                .unwrap_or(tail.len());
            let token = &tail[..end];
            // `sm_90a` is a target; `sm_90abc` is something else entirely, so
            // only accept digits with at most one trailing a/f.
            let digits_end = token
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(token.len());
            let ok = digits_end > 0 && token.len() <= digits_end + 1;
            if ok {
                if let Ok(arch) = CudaArch::parse(&format!("sm_{token}")) {
                    if !out.contains(&arch) {
                        out.push(arch);
                    }
                }
            }
            rest = &tail[end..];
        }
    }
    out.sort();
    out
}

/// Locate `cuobjdump`.
///
/// Preference order: an explicit `CUOBJDUMP`, then the toolkit root the rest of
/// the build already uses, then whatever is next to `nvcc`, then `PATH`.
fn cuobjdump_path() -> PathBuf {
    if let Ok(p) = std::env::var("CUOBJDUMP") {
        return PathBuf::from(p);
    }
    for root_var in [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
    ] {
        if let Ok(root) = std::env::var(root_var) {
            let candidate = Path::new(&root).join("bin").join("cuobjdump");
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    if let Ok(nvcc) = std::env::var("NVCC") {
        if let Some(dir) = Path::new(&nvcc).parent() {
            let candidate = dir.join("cuobjdump");
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    let default = Path::new("/usr/local/cuda/bin/cuobjdump");
    if default.is_file() {
        return default.to_path_buf();
    }
    PathBuf::from("cuobjdump")
}

/// Run `cuobjdump --list-elf` against a built archive.
///
/// An empty listing is reported as [`Observation::Unavailable`], not as an
/// empty [`Observation::Observed`]: a tool that succeeds while telling us
/// nothing has not observed anything.
pub fn observe_archive(archive: &Path) -> Observation {
    if !archive.exists() {
        return Observation::Unavailable(format!("{} does not exist", archive.display()));
    }
    let tool = cuobjdump_path();
    let output = match Command::new(&tool).arg("--list-elf").arg(archive).output() {
        Ok(o) => o,
        Err(e) => {
            return Observation::Unavailable(format!("could not run {}: {e}", tool.display()));
        }
    };
    if !output.status.success() {
        return Observation::Unavailable(format!(
            "{} --list-elf exited {}: {}",
            tool.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let archs = parse_list_elf(&stdout);
    if archs.is_empty() {
        return Observation::Unavailable(format!(
            "{} --list-elf listed no cubins for {}",
            tool.display(),
            archive.display()
        ));
    }
    Observation::Observed(archs)
}

/// Verify a freshly built archive against what was requested, then export
/// [`BUILT_ENV`] for the runtime.
///
/// * When architectures were **requested**, every one of them must appear in
///   the archive. A missing architecture, or an archive that cannot be
///   inspected at all, panics — the build made a claim it could not back.
/// * When none were requested, no claim was made: the observation is exported
///   if it succeeded and the failure is reported as a `cargo:warning` if it did
///   not, so the runtime reports `Unknown` rather than inventing an answer.
///
/// # Panics
///
/// Deliberately, on an unmet request — that is the gate.
pub fn verify_and_export(archive: &Path, requested: Option<&[CudaArch]>) {
    let observation = observe_archive(archive);
    let Some(want) = requested else {
        match observation {
            Observation::Observed(built) => {
                println!("cargo:rustc-env={BUILT_ENV}={}", render_arch_list(&built));
            }
            Observation::Unavailable(why) => {
                // No claim was made, so this is information, not a failure.
                // The empty value makes the runtime say "unknown" instead of
                // guessing.
                println!("cargo:rustc-env={BUILT_ENV}=");
                println!("cargo:warning=ArcTarget: cubin architectures not verified ({why})");
            }
        }
        return;
    };

    let built = match observation {
        Observation::Observed(built) => built,
        Observation::Unavailable(why) => panic!(
            "ArcTarget: {ARCHS_ENV}={} was requested but the produced archive could not be \
             inspected, so the architecture claim cannot be verified: {why}.\n\
             Install the CUDA toolkit's `cuobjdump` (or set CUOBJDUMP=/path/to/cuobjdump). \
             Refusing to ship an unverified multi-architecture claim.",
            render_arch_list(want)
        ),
    };

    let missing: Vec<CudaArch> = want
        .iter()
        .copied()
        .filter(|w| !built.contains(w))
        .collect();
    if !missing.is_empty() {
        panic!(
            "ArcTarget: {ARCHS_ENV}={} was requested but {} contains only {}.\n\
             MISSING: {}.\n\
             The build exited 0 while producing a binary that has no code for those \
             architectures; on such a device every Arc kernel launch fails with \
             cudaErrorNoKernelImageForDevice (Arc embeds no PTX). Check that the toolkit \
             supports them: `nvcc --list-gpu-code`.",
            render_arch_list(want),
            archive.display(),
            render_arch_list(&built),
            render_arch_list(&missing),
        );
    }

    println!("cargo:rustc-env={BUILT_ENV}={}", render_arch_list(&built));
    println!(
        "cargo:warning=ArcTarget: verified cubins {} in {}",
        render_arch_list(&built),
        archive
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArchSuffix;

    #[test]
    fn parses_real_cuobjdump_listing() {
        // Shape of `cuobjdump --list-elf` over a static archive built with
        // three `-gencode` targets.
        let out = "\
ELF file    1: qtip_grouped_gemm.1.sm_80.cubin
ELF file    2: qtip_grouped_gemm.2.sm_90a.cubin
ELF file    3: qtip_grouped_gemm.3.sm_100a.cubin
ELF file    4: qtip2b_beam.1.sm_80.cubin
ELF file    5: qtip2b_beam.2.sm_90a.cubin
ELF file    6: qtip2b_beam.3.sm_100a.cubin
";
        let archs = parse_list_elf(out);
        assert_eq!(render_arch_list(&archs), "sm_80,sm_90a,sm_100a");
    }

    #[test]
    fn the_measured_failure_is_detected() {
        // Exactly what the H200 gate saw: asked for 90,100,103, got sm_90a.
        let built = parse_list_elf("ELF file    1: qtip_grouped_gemm.1.sm_90a.cubin\n");
        let want = parse_arch_list("90,100,103").unwrap();
        let missing: Vec<_> = want.iter().filter(|w| !built.contains(w)).collect();
        assert_eq!(
            missing.iter().map(|a| a.sm()).collect::<Vec<_>>().join(","),
            "sm_100a,sm_103a"
        );
    }

    #[test]
    fn empty_listing_is_not_an_empty_success() {
        assert!(parse_list_elf("").is_empty());
        assert!(parse_list_elf("cuobjdump info : File is not a fatbin\n").is_empty());
    }

    #[test]
    fn does_not_mistake_arbitrary_sm_prefixed_words_for_targets() {
        // `sm_90a` is a target; a symbol name that merely starts with `sm_` is
        // not, and neither is a bare `sm_` or a sub-baseline capability.
        assert!(parse_list_elf("note: sm_scheduler_v2 rebuilt\n").is_empty());
        assert!(parse_list_elf("sm_ sm_x sm_75\n").is_empty());
        let archs = parse_list_elf("a.sm_90a.cubin and sm_90abc\n");
        assert_eq!(render_arch_list(&archs), "sm_90a");
    }

    #[test]
    fn primary_split_hands_cudaforge_something_it_can_express() {
        let archs = parse_arch_list("80,90,100").unwrap();
        let (primary, extra) = split_primary(&archs).unwrap();
        assert_eq!(primary, "80");
        assert_eq!(
            extra,
            vec![
                "-gencode=arch=compute_90a,code=sm_90a".to_string(),
                "-gencode=arch=compute_100a,code=sm_100a".to_string(),
            ]
        );
        // Hopper-only build: the primary is the arch-specific spelling, which
        // cudaforge round-trips.
        let hopper = parse_arch_list("90").unwrap();
        let (primary, extra) = split_primary(&hopper).unwrap();
        assert_eq!(primary, "90a");
        assert!(extra.is_empty());
    }

    #[test]
    fn primary_split_refuses_rather_than_letting_autodetect_win() {
        let family = vec![CudaArch::new(100, ArchSuffix::FamilySpecific)];
        assert!(matches!(
            split_primary(&family),
            Err(ArchError::FamilyTargetNotRequestable(_))
        ));
    }
}
