//! **ArcTarget** — the one place Arc decides which GPU architectures it builds
//! for, and which architecture it is running on.
//!
//! # Why this crate exists
//!
//! Arc used to be *arch-portable by being arch-naive*: every CUDA crate handed
//! architecture selection to `cudaforge`, which derives exactly **one**
//! `-gencode` from `CUDA_COMPUTE_CAP` or the build box's `nvidia-smi`
//! (`cudaforge-0.1.5/src/builder.rs:401-405`). A binary built on an H200
//! therefore contains SM90 SASS and nothing else — while the release matrix,
//! the installer and the docs all claimed Ada/Hopper/Blackwell support.
//!
//! Measured on a live H200 on 2026-08-17: a build asking for
//! `ARC_CUDA_ARCHS=90,100,103` compiled clean and produced an archive whose
//! only cubin was `sm_90a`. The compile passing was read as the arches being
//! present. That is [`D18`] — *the absence of a signal read as a specific
//! signal* — applied to the arch matrix itself.
//!
//! [`D18`]: https://github.com/aeonmindai/arc — `memory/mission/KERNEL_RULES.md`
//!
//! So this crate owns three things and nothing else:
//!
//! 1. [`CudaArch`] — parsing and emitting architecture targets, including the
//!    architecture-specific (`a`) and family-specific (`f`) suffixes that
//!    Hopper and Blackwell instructions require.
//! 2. [`ArcTier`] — the *specialisation* a capability selects: Ampere, Hopper
//!    or Blackwell. This is the enum kernels and dispatch agree on.
//! 3. [`build`] — the build-script half: turn a requested arch list into
//!    `-gencode` arguments, then **verify against the produced archive with
//!    `cuobjdump` and fail the build if a requested arch is missing.** The
//!    verification is the point. A build that cannot prove which cubins it
//!    produced is not allowed to claim any.
//!
//! # The rule this crate mechanises
//!
//! > A green build must prove the cubins exist, not merely that nvcc exited 0.
//!
//! and its corollary, which is why [`Observation`] has three states rather than
//! two: *"cuobjdump could not be run"* is neither a pass nor a failure, and
//! collapsing it into either is the same bug one level up.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::fmt;

// ---------------------------------------------------------------------------
// Architecture targets
// ---------------------------------------------------------------------------

/// Which flavour of cubin a target names.
///
/// nvcc distinguishes three, and the difference is load-bearing for Arc: the
/// Hopper and Blackwell instructions our specialised paths are written against
/// (`wgmma.mma_async`, `tcgen05.mma`) are only available in the
/// architecture-specific variants.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum ArchSuffix {
    /// Plain real architecture, e.g. `sm_80`. Binary-compatible upward within
    /// the same major version (a `sm_80` cubin runs on an `sm_86` device).
    Real,
    /// Architecture-specific, e.g. `sm_90a`. Runs **only** on that exact
    /// capability, and is the only variant exposing the full arch-specific
    /// instruction set.
    ArchSpecific,
    /// Family-specific, e.g. `sm_100f` (CUDA 12.9+). Runs on the same major
    /// family at or above its capability — `compute_100f` covers `sm_100` and
    /// `sm_103`.
    ///
    /// Parsed and understood so that a cubin emitted by a vendored dependency
    /// can be read back correctly, but **not accepted as a build request**:
    /// see [`ArchError::FamilyTargetNotRequestable`].
    FamilySpecific,
}

impl ArchSuffix {
    fn as_str(self) -> &'static str {
        match self {
            ArchSuffix::Real => "",
            ArchSuffix::ArchSpecific => "a",
            ArchSuffix::FamilySpecific => "f",
        }
    }
}

/// A single nvcc architecture target, e.g. `sm_90a`.
///
/// `cc` is the capability with the decimal point removed exactly the way nvcc
/// spells it: `80` is 8.0, `90` is 9.0, `100` is 10.0 and `103` is 10.3. Note
/// that this is *not* `major * 10 + minor` past Hopper — it is `major * 10 +
/// minor` only because Blackwell's major is 10; use [`CudaArch::major`] and
/// [`CudaArch::minor`] rather than doing the arithmetic at call sites.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct CudaArch {
    cc: u32,
    suffix: ArchSuffix,
}

/// Why an architecture string was rejected.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ArchError {
    /// The string was not `[sm_|compute_]<digits>[a|f]`.
    Malformed(String),
    /// The capability is below the Ampere baseline every Arc CUDA kernel is
    /// written against (`has_qtip_kernels` is gated at cc >= 80).
    BelowBaseline(u32),
    /// A family-specific (`f`) target was requested for a build.
    ///
    /// `cudaforge` derives its single primary `-gencode` by parsing the arch
    /// string, and its parser accepts only digits with an optional `a`
    /// (`cudaforge-0.1.5/src/compute_cap.rs:36-70`). Worse, it *silently
    /// ignores* a string it cannot parse and falls back to `nvidia-smi`
    /// autodetect — so a family target would not fail, it would quietly build
    /// for the wrong arch. Request the members explicitly instead
    /// (`100,103` rather than `100f`), which also yields arch-specific cubins
    /// on both rather than the reduced family instruction set.
    FamilyTargetNotRequestable(String),
    /// The list was empty or contained only separators.
    EmptyList,
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchError::Malformed(s) => write!(
                f,
                "malformed CUDA arch {s:?}: expected e.g. `80`, `89`, `90`, `100`, `103`, \
                 `sm_90a` or `compute_100a`"
            ),
            ArchError::BelowBaseline(cc) => write!(
                f,
                "CUDA arch sm_{cc} is below Arc's sm_80 baseline; every Arc kernel is gated \
                 at compute capability 8.0 (`has_qtip_kernels`)"
            ),
            ArchError::FamilyTargetNotRequestable(s) => write!(
                f,
                "family-specific target {s:?} cannot be requested: cudaforge's arch parser \
                 accepts only digits with an optional `a` and SILENTLY falls back to \
                 nvidia-smi autodetect on anything else. List the family members instead \
                 (e.g. `100,103`), which also gives arch-specific cubins on both."
            ),
            ArchError::EmptyList => write!(f, "empty CUDA arch list"),
        }
    }
}

impl std::error::Error for ArchError {}

/// Arc's Ampere baseline. Below this no Arc CUDA kernel is compiled at all.
pub const BASELINE_CC: u32 = 80;

impl CudaArch {
    /// Construct a target directly. Prefer [`CudaArch::parse`].
    pub const fn new(cc: u32, suffix: ArchSuffix) -> Self {
        Self { cc, suffix }
    }

    /// Parse `80`, `90a`, `sm_100a`, `compute_103a`, `sm_100f`, …
    ///
    /// **A capability of 90 or more with no explicit suffix selects the
    /// architecture-specific (`a`) variant.** That matches cudaforge's own
    /// auto-suffixing (`compute_cap.rs:74-79`) so the two halves of the build
    /// cannot disagree, and it matches intent: Arc's Hopper and Blackwell
    /// paths are written against arch-specific instructions, so a plain cubin
    /// would silently be the weaker binary — the exact failure this crate
    /// exists to make impossible.
    pub fn parse(s: &str) -> Result<Self, ArchError> {
        let raw = s.trim();
        let lower = raw.to_ascii_lowercase();
        let body = lower
            .strip_prefix("sm_")
            .or_else(|| lower.strip_prefix("compute_"))
            .unwrap_or(&lower);
        if body.is_empty() {
            return Err(ArchError::Malformed(raw.to_string()));
        }

        let (digits, suffix) = match body.as_bytes()[body.len() - 1] {
            b'a' => (&body[..body.len() - 1], Some(ArchSuffix::ArchSpecific)),
            b'f' => (&body[..body.len() - 1], Some(ArchSuffix::FamilySpecific)),
            _ => (body, None),
        };

        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ArchError::Malformed(raw.to_string()));
        }
        let cc: u32 = digits
            .parse()
            .map_err(|_| ArchError::Malformed(raw.to_string()))?;
        if cc < BASELINE_CC {
            return Err(ArchError::BelowBaseline(cc));
        }

        let suffix = suffix.unwrap_or({
            if cc >= 90 {
                ArchSuffix::ArchSpecific
            } else {
                ArchSuffix::Real
            }
        });
        Ok(Self { cc, suffix })
    }

    /// The capability as nvcc spells it: `80`, `90`, `100`, `103`.
    pub const fn cc(&self) -> u32 {
        self.cc
    }

    /// Which cubin flavour this target names.
    pub const fn suffix(&self) -> ArchSuffix {
        self.suffix
    }

    /// Compute-capability major version (`sm_103` -> 10).
    pub const fn major(&self) -> u32 {
        self.cc / 10
    }

    /// Compute-capability minor version (`sm_103` -> 3).
    pub const fn minor(&self) -> u32 {
        self.cc % 10
    }

    /// `sm_90a`
    pub fn sm(&self) -> String {
        format!("sm_{}{}", self.cc, self.suffix.as_str())
    }

    /// `compute_90a`
    pub fn compute(&self) -> String {
        format!("compute_{}{}", self.cc, self.suffix.as_str())
    }

    /// The nvcc fat-binary argument, e.g.
    /// `-gencode=arch=compute_90a,code=sm_90a`.
    ///
    /// nvcc documents `--generate-code` as repeatable, and repeating it is what
    /// produces a fat binary carrying a cubin per architecture. This is the
    /// only mechanism Arc uses; no PTX is ever embedded, so a device with no
    /// matching cubin gets `cudaErrorNoKernelImageForDevice` at launch rather
    /// than a JIT that might silently disagree with SASS.
    pub fn gencode(&self) -> String {
        format!("-gencode=arch={},code={}", self.compute(), self.sm())
    }

    /// The string form `cudaforge` accepts for its single primary target.
    ///
    /// `None` when cudaforge cannot express this target — see
    /// [`ArchError::FamilyTargetNotRequestable`] for why that has to be a
    /// refusal rather than a fallback.
    pub fn cudaforge_arch(&self) -> Option<String> {
        match self.suffix {
            // cudaforge auto-suffixes >= 90 with `a`, so a bare number is only
            // safe to hand it below 90.
            ArchSuffix::Real if self.cc < 90 => Some(self.cc.to_string()),
            ArchSuffix::Real => None,
            ArchSuffix::ArchSpecific => Some(format!("{}a", self.cc)),
            ArchSuffix::FamilySpecific => None,
        }
    }

    /// Which Arc specialisation this target selects.
    pub const fn tier(&self) -> ArcTier {
        ArcTier::for_cc(self.cc)
    }

    /// Whether a device of capability `device_cc` can execute this cubin, and
    /// on what basis.
    pub fn coverage_of(&self, device_cc: u32) -> Option<Coverage> {
        // Arch-specific cubins are not compatible with anything else, not even
        // a higher minor of the same major.
        if self.suffix == ArchSuffix::ArchSpecific {
            return (self.cc == device_cc).then_some(Coverage::Exact);
        }
        // Plain and family targets both run upward within their own major.
        if self.major() != device_cc / 10 || device_cc < self.cc {
            return None;
        }
        if device_cc == self.cc {
            return Some(Coverage::Exact);
        }
        Some(match self.suffix {
            ArchSuffix::FamilySpecific => Coverage::Family,
            _ => Coverage::MinorCompatible,
        })
    }
}

impl fmt::Display for CudaArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.sm())
    }
}

/// On what basis a device is covered by a cubin in the binary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Coverage {
    /// A cubin targets this exact capability.
    Exact,
    /// A plain cubin of the same major and a lower minor. Documented as
    /// binary-compatible, but Arc's gates demand [`Coverage::Exact`] because a
    /// minor-compatible cubin was, by construction, compiled without the
    /// newer minor's arch-specific instructions.
    MinorCompatible,
    /// A family-specific (`f`) cubin covering this capability.
    Family,
}

// ---------------------------------------------------------------------------
// Tiers — the three specialisations
// ---------------------------------------------------------------------------

/// The specialisation a compute capability selects.
///
/// One system, three specialised paths. `Unspecialised` is deliberately a
/// separate variant rather than being folded into the nearest tier: consumer
/// Blackwell (`sm_120`) has no `tcgen05`, so calling it `Blackwell` would
/// re-create in the type system exactly the over-claim this crate removes from
/// the build.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ArcTier {
    /// sm_80 / sm_86 / sm_89 — `mma.sync` + `cp.async`. The baseline, and the
    /// fallback every other tier degrades to. It must stay correct.
    Ampere,
    /// sm_90 / sm_9x — `wgmma.mma_async`, TMA, and a 227 KB shared-memory
    /// budget that must be opted into with `cudaFuncSetAttribute`.
    Hopper,
    /// sm_100 / sm_103 — `tcgen05` / TMEM and 2-CTA MMA.
    Blackwell,
    /// Anything Arc has no specialised path for (including consumer Blackwell
    /// sm_120, which lacks `tcgen05`). Takes the Ampere path.
    Unspecialised,
}

impl ArcTier {
    /// Map a capability to its specialisation.
    pub const fn for_cc(cc: u32) -> Self {
        match cc {
            80..=89 => ArcTier::Ampere,
            90..=99 => ArcTier::Hopper,
            100..=109 => ArcTier::Blackwell,
            _ => ArcTier::Unspecialised,
        }
    }

    /// Map a `(major, minor)` pair, as reported by the driver, to its
    /// specialisation.
    pub const fn for_capability(major: u32, minor: u32) -> Self {
        Self::for_cc(major * 10 + minor)
    }

    /// Short lowercase name, used in logs and gate output.
    pub const fn name(&self) -> &'static str {
        match self {
            ArcTier::Ampere => "ampere",
            ArcTier::Hopper => "hopper",
            ArcTier::Blackwell => "blackwell",
            ArcTier::Unspecialised => "unspecialised",
        }
    }
}

impl fmt::Display for ArcTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Lists
// ---------------------------------------------------------------------------

/// Parse a comma-separated arch list, e.g. `"80,90,100,103"`.
///
/// Duplicates are collapsed (nvcc rejects a repeated `-gencode` for the same
/// target) and the result is sorted, so the emitted argument order and the
/// exported `ARC_CUDA_ARCHS_BUILT` string are deterministic regardless of how
/// the operator typed the list.
pub fn parse_arch_list(s: &str) -> Result<Vec<CudaArch>, ArchError> {
    let mut out: Vec<CudaArch> = Vec::new();
    for tok in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let arch = CudaArch::parse(tok)?;
        if arch.suffix() == ArchSuffix::FamilySpecific {
            return Err(ArchError::FamilyTargetNotRequestable(tok.to_string()));
        }
        if !out.contains(&arch) {
            out.push(arch);
        }
    }
    if out.is_empty() {
        return Err(ArchError::EmptyList);
    }
    out.sort();
    Ok(out)
}

/// Render a list the way [`parse_arch_list`] and the runtime both read it back.
pub fn render_arch_list(archs: &[CudaArch]) -> String {
    archs.iter().map(|a| a.sm()).collect::<Vec<_>>().join(",")
}

/// How a device is served by the cubins actually present in this binary.
///
/// Three states, not two, and for the [`D18`](crate) reason: `Unknown` means
/// the build could not observe its own output, which is neither "covered" nor
/// "not covered". A caller that treats `Unknown` as either is reintroducing the
/// bug.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DeviceCoverage {
    /// A cubin serves this device; carries which one and on what basis.
    Covered(CudaArch, Coverage),
    /// The binary provably contains no cubin for this device. Arc embeds no
    /// PTX, so every kernel launch will fail with
    /// `cudaErrorNoKernelImageForDevice`.
    Missing {
        /// The cubins that *are* present.
        built: Vec<CudaArch>,
    },
    /// The build did not record which cubins it produced.
    Unknown,
}

/// Decide how `device_cc` is served by `built`.
///
/// Prefers an exact match over a compatible one, so the report names the cubin
/// the driver will actually select.
pub fn device_coverage(built: &[CudaArch], device_cc: u32) -> DeviceCoverage {
    if built.is_empty() {
        return DeviceCoverage::Unknown;
    }
    let mut best: Option<(CudaArch, Coverage)> = None;
    for arch in built {
        let Some(cov) = arch.coverage_of(device_cc) else {
            continue;
        };
        let better = match &best {
            None => true,
            Some((_, Coverage::Exact)) => false,
            Some(_) => cov == Coverage::Exact,
        };
        if better {
            best = Some((*arch, cov));
        }
    }
    match best {
        Some((arch, cov)) => DeviceCoverage::Covered(arch, cov),
        None => DeviceCoverage::Missing {
            built: built.to_vec(),
        },
    }
}

/// Read back the arch list a crate's build script recorded.
///
/// `raw` is the `ARC_CUDA_ARCHS_BUILT` value; an empty or absent value means
/// the build did not observe its own output and yields an empty list, which
/// [`device_coverage`] reports as [`DeviceCoverage::Unknown`].
pub fn parse_built_list(raw: Option<&str>) -> Vec<CudaArch> {
    raw.unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .filter_map(|t| CudaArch::parse(t).ok())
        .collect()
}

// ---------------------------------------------------------------------------
// Build-script half
// ---------------------------------------------------------------------------

pub mod build;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_spelling() {
        for (s, cc, suffix) in [
            ("80", 80, ArchSuffix::Real),
            ("86", 86, ArchSuffix::Real),
            ("89", 89, ArchSuffix::Real),
            ("sm_89", 89, ArchSuffix::Real),
            ("compute_86", 86, ArchSuffix::Real),
            // >= 90 with no explicit suffix is arch-specific.
            ("90", 90, ArchSuffix::ArchSpecific),
            ("sm_90a", 90, ArchSuffix::ArchSpecific),
            ("100", 100, ArchSuffix::ArchSpecific),
            ("sm_103a", 103, ArchSuffix::ArchSpecific),
            ("SM_100A", 100, ArchSuffix::ArchSpecific),
            ("sm_100f", 100, ArchSuffix::FamilySpecific),
            (" 120 ", 120, ArchSuffix::ArchSpecific),
        ] {
            let a = CudaArch::parse(s).unwrap_or_else(|e| panic!("{s}: {e}"));
            assert_eq!((a.cc(), a.suffix()), (cc, suffix), "parsing {s}");
        }
    }

    #[test]
    fn rejects_junk_and_sub_baseline() {
        for s in ["", "sm_", "ninety", "90b", "sm_9x", "a", "-90"] {
            assert!(
                matches!(CudaArch::parse(s), Err(ArchError::Malformed(_))),
                "{s} should be malformed, got {:?}",
                CudaArch::parse(s)
            );
        }
        assert_eq!(CudaArch::parse("75"), Err(ArchError::BelowBaseline(75)));
    }

    #[test]
    fn gencode_and_names_match_nvcc_spelling() {
        let hopper = CudaArch::parse("90").unwrap();
        assert_eq!(hopper.sm(), "sm_90a");
        assert_eq!(hopper.compute(), "compute_90a");
        assert_eq!(hopper.gencode(), "-gencode=arch=compute_90a,code=sm_90a");
        let ampere = CudaArch::parse("80").unwrap();
        assert_eq!(ampere.gencode(), "-gencode=arch=compute_80,code=sm_80");
        let blackwell = CudaArch::parse("103").unwrap();
        assert_eq!(
            blackwell.gencode(),
            "-gencode=arch=compute_103a,code=sm_103a"
        );
    }

    #[test]
    fn major_minor_split_survives_blackwell() {
        let b = CudaArch::parse("103").unwrap();
        assert_eq!((b.major(), b.minor()), (10, 3));
        let h = CudaArch::parse("90").unwrap();
        assert_eq!((h.major(), h.minor()), (9, 0));
        let a = CudaArch::parse("89").unwrap();
        assert_eq!((a.major(), a.minor()), (8, 9));
    }

    #[test]
    fn tiers_are_the_three_specialisations() {
        assert_eq!(ArcTier::for_cc(80), ArcTier::Ampere);
        assert_eq!(ArcTier::for_cc(86), ArcTier::Ampere);
        assert_eq!(ArcTier::for_cc(89), ArcTier::Ampere);
        assert_eq!(ArcTier::for_cc(90), ArcTier::Hopper);
        assert_eq!(ArcTier::for_cc(100), ArcTier::Blackwell);
        assert_eq!(ArcTier::for_cc(103), ArcTier::Blackwell);
        // Consumer Blackwell has no tcgen05: it must NOT claim the Blackwell
        // specialisation.
        assert_eq!(ArcTier::for_cc(120), ArcTier::Unspecialised);
        assert_eq!(ArcTier::for_capability(9, 0), ArcTier::Hopper);
        assert_eq!(ArcTier::for_capability(10, 3), ArcTier::Blackwell);
    }

    #[test]
    fn arch_specific_cubins_cover_only_their_own_capability() {
        let h = CudaArch::parse("90a").unwrap();
        assert_eq!(h.coverage_of(90), Some(Coverage::Exact));
        assert_eq!(h.coverage_of(89), None);
        assert_eq!(h.coverage_of(100), None);
        // The one that mattered: an sm_100a cubin does NOT serve an sm_103
        // device, and vice versa.
        let b0 = CudaArch::parse("100a").unwrap();
        assert_eq!(b0.coverage_of(103), None);
        let b3 = CudaArch::parse("103a").unwrap();
        assert_eq!(b3.coverage_of(100), None);
    }

    #[test]
    fn plain_and_family_cubins_cover_upward_within_a_major() {
        let a80 = CudaArch::parse("80").unwrap();
        assert_eq!(a80.coverage_of(80), Some(Coverage::Exact));
        assert_eq!(a80.coverage_of(86), Some(Coverage::MinorCompatible));
        assert_eq!(a80.coverage_of(89), Some(Coverage::MinorCompatible));
        assert_eq!(a80.coverage_of(90), None, "no cross-major compatibility");
        let f = CudaArch::parse("sm_100f").unwrap();
        assert_eq!(f.coverage_of(100), Some(Coverage::Exact));
        assert_eq!(f.coverage_of(103), Some(Coverage::Family));
        assert_eq!(f.coverage_of(90), None);
    }

    #[test]
    fn arch_lists_dedupe_sort_and_reject_family_requests() {
        let l = parse_arch_list("100, 80,90 , 80").unwrap();
        assert_eq!(render_arch_list(&l), "sm_80,sm_90a,sm_100a");
        assert_eq!(parse_arch_list("  ,, "), Err(ArchError::EmptyList));
        assert!(matches!(
            parse_arch_list("90,100f"),
            Err(ArchError::FamilyTargetNotRequestable(_))
        ));
    }

    #[test]
    fn cudaforge_primary_arch_is_only_offered_when_it_round_trips() {
        // What cudaforge itself would emit, replicated from
        // cudaforge-0.1.5/src/compute_cap.rs:74-99.
        fn cudaforge_would_emit(arch_str: &str) -> String {
            let base: u32 = arch_str.trim_end_matches('a').parse().unwrap();
            if arch_str.ends_with('a') || base >= 90 {
                format!("sm_{base}a")
            } else {
                format!("sm_{base}")
            }
        }
        for s in ["80", "86", "89", "90", "100", "103"] {
            let arch = CudaArch::parse(s).unwrap();
            let handed = arch.cudaforge_arch().expect("requestable");
            assert_eq!(
                cudaforge_would_emit(&handed),
                arch.sm(),
                "cudaforge would emit a different arch than {s} asked for"
            );
        }
        // Not expressible -> None, never a silent substitution.
        assert_eq!(
            CudaArch::new(100, ArchSuffix::FamilySpecific).cudaforge_arch(),
            None
        );
        assert_eq!(CudaArch::new(90, ArchSuffix::Real).cudaforge_arch(), None);
    }

    #[test]
    fn device_coverage_prefers_exact_and_reports_missing() {
        let built = parse_arch_list("80,90,100").unwrap();
        assert_eq!(
            device_coverage(&built, 90),
            DeviceCoverage::Covered(CudaArch::parse("90a").unwrap(), Coverage::Exact)
        );
        assert_eq!(
            device_coverage(&built, 86),
            DeviceCoverage::Covered(CudaArch::parse("80").unwrap(), Coverage::MinorCompatible)
        );
        // The measured failure: a "Blackwell-supported" binary whose only
        // Blackwell-era cubin is sm_100a, run on an sm_103 device.
        assert!(matches!(
            device_coverage(&built, 103),
            DeviceCoverage::Missing { .. }
        ));
        // Nothing observed is NOT "not covered".
        assert_eq!(device_coverage(&[], 90), DeviceCoverage::Unknown);
    }

    #[test]
    fn built_list_round_trips() {
        let built = parse_arch_list("80,90,100,103").unwrap();
        let rendered = render_arch_list(&built);
        assert_eq!(parse_built_list(Some(&rendered)), built);
        assert!(parse_built_list(None).is_empty());
        assert!(parse_built_list(Some("")).is_empty());
    }
}
