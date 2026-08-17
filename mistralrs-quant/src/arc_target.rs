//! **ArcTarget at runtime** — which GPU we are actually on, and whether this
//! binary contains code for it.
//!
//! The build half ([`arc_target::build`]) records the architectures the
//! produced archive was *observed* to contain. This half reads that record
//! back, asks the driver what card it is talking to, and states — once per
//! distinct capability, on every outcome — which of the three specialisations
//! is in play and whether a cubin for it exists.
//!
//! # Why a runtime check at all, when the build already gates
//!
//! Because the build gate and the run happen on different machines. A binary
//! built with `ARC_CUDA_ARCHS=80,90` is *correctly* built; running it on a
//! B200 is still wrong, and Arc embeds no PTX, so the symptom is
//! `cudaErrorNoKernelImageForDevice` from whichever kernel happens to launch
//! first — a long way from the cause. This turns that into one sentence naming
//! the device, the architectures present, and the fix.
//!
//! It is also the [`D18`](arc_target) half of arch dispatch: a dispatch that
//! silently takes the Ampere path on Hopper is indistinguishable from one that
//! took the Hopper path, unless something says which. [`log_once`] says which.

pub use arc_target::{ArcTier, Coverage, CudaArch, DeviceCoverage};

/// The architectures this crate's CUDA archive was **observed** to contain.
///
/// Observed, not requested: the value comes from `cuobjdump --list-elf` run
/// against the archive the build actually produced. An empty list means the
/// build could not inspect its own output — which [`arc_target::device_coverage`]
/// reports as [`DeviceCoverage::Unknown`], never as "not covered".
pub fn built_archs() -> Vec<CudaArch> {
    arc_target::parse_built_list(option_env!("ARC_CUDA_ARCHS_BUILT"))
}

/// What ArcTarget knows about the device a kernel is about to run on.
#[derive(Clone, Debug)]
pub struct TargetReport {
    /// Compute capability as `(major, minor)`, e.g. `(9, 0)` for H100/H200.
    pub capability: (u32, u32),
    /// The specialisation this capability selects.
    pub tier: ArcTier,
    /// Whether a cubin in this binary serves the device.
    pub coverage: DeviceCoverage,
}

impl TargetReport {
    /// A single line naming the device, the tier taken, and the cubin serving
    /// it — the line that makes a silent fallback impossible to mistake for a
    /// specialised run.
    pub fn summary(&self) -> String {
        let (major, minor) = self.capability;
        match &self.coverage {
            DeviceCoverage::Covered(arch, Coverage::Exact) => format!(
                "ArcTarget: device sm_{major}{minor} -> {} path, running {arch} cubin",
                self.tier
            ),
            DeviceCoverage::Covered(arch, basis) => format!(
                "ArcTarget: device sm_{major}{minor} -> {} path, running {arch} cubin \
                 ({basis:?}) — this cubin was compiled WITHOUT sm_{major}{minor}'s \
                 architecture-specific instructions",
                self.tier
            ),
            DeviceCoverage::Missing { built } => format!(
                "ArcTarget: device sm_{major}{minor} ({} class) has NO cubin in this binary \
                 (built: {}). Arc embeds no PTX, so every kernel launch will fail with \
                 cudaErrorNoKernelImageForDevice. Rebuild with \
                 ARC_CUDA_ARCHS={}{}",
                self.tier,
                arc_target::render_arch_list(built),
                if built.is_empty() {
                    String::new()
                } else {
                    format!(
                        "{},",
                        built
                            .iter()
                            .map(|a| a.cc().to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                },
                major * 10 + minor
            ),
            DeviceCoverage::Unknown => format!(
                "ArcTarget: device sm_{major}{minor} -> {} path; the build did not record \
                 which cubins it produced, so architecture coverage is UNVERIFIED (neither \
                 confirmed nor refuted)",
                self.tier
            ),
        }
    }
}

/// Build a report from a capability and the architectures this binary carries.
///
/// Split out from the device query so it is testable without a GPU — the query
/// is the only part that needs hardware.
pub fn report_for_capability(major: u32, minor: u32) -> TargetReport {
    let built = built_archs();
    TargetReport {
        capability: (major, minor),
        tier: ArcTier::for_capability(major, minor),
        coverage: arc_target::device_coverage(&built, major * 10 + minor),
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;
    use candle_core::{CudaDevice, Result};
    use std::sync::{Mutex, OnceLock};

    /// Ask the driver for the device's compute capability.
    pub fn device_capability(dev: &CudaDevice) -> Result<(u32, u32)> {
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        let major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| candle_core::Error::Msg(format!("ArcTarget: cc major query: {e}")))?;
        let minor = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| candle_core::Error::Msg(format!("ArcTarget: cc minor query: {e}")))?;
        if major < 0 || minor < 0 {
            candle_core::bail!("ArcTarget: driver reported compute capability {major}.{minor}");
        }
        Ok((major as u32, minor as u32))
    }

    /// Report on a live device.
    pub fn report(dev: &CudaDevice) -> Result<TargetReport> {
        let (major, minor) = device_capability(dev)?;
        Ok(report_for_capability(major, minor))
    }

    /// State the target once per distinct capability, on **every** outcome.
    ///
    /// Every branch emits a line, including the ones where nothing is wrong and
    /// the one where we could not tell — silence here would mean "no news",
    /// which is precisely the reading that has to be impossible.
    pub fn log_once(dev: &CudaDevice) {
        let report = match report(dev) {
            Ok(r) => r,
            Err(e) => {
                if first_time_for(None) {
                    tracing::warn!("{e}");
                }
                return;
            }
        };
        if !first_time_for(Some(report.capability)) {
            return;
        }
        match report.coverage {
            DeviceCoverage::Covered(_, Coverage::Exact) => tracing::info!("{}", report.summary()),
            DeviceCoverage::Covered(_, _) | DeviceCoverage::Unknown => {
                tracing::warn!("{}", report.summary())
            }
            DeviceCoverage::Missing { .. } => tracing::error!("{}", report.summary()),
        }
    }

    /// `true` the first time a given capability is seen. `None` keys the
    /// "could not query" case, so that failure is reported once rather than on
    /// every kernel launch.
    fn first_time_for(capability: Option<(u32, u32)>) -> bool {
        static SEEN: OnceLock<Mutex<Vec<Option<(u32, u32)>>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(Vec::new()));
        let mut guard = match seen.lock() {
            Ok(g) => g,
            // A poisoned lock must not silence the report.
            Err(poisoned) => poisoned.into_inner(),
        };
        if guard.contains(&capability) {
            return false;
        }
        guard.push(capability);
        true
    }
}

#[cfg(feature = "cuda")]
pub use cuda::{device_capability, log_once, report};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_maps_to_the_right_specialisation() {
        assert_eq!(report_for_capability(8, 0).tier, ArcTier::Ampere);
        assert_eq!(report_for_capability(8, 9).tier, ArcTier::Ampere);
        assert_eq!(report_for_capability(9, 0).tier, ArcTier::Hopper);
        assert_eq!(report_for_capability(10, 0).tier, ArcTier::Blackwell);
        assert_eq!(report_for_capability(10, 3).tier, ArcTier::Blackwell);
    }

    #[test]
    fn an_unrecorded_build_reports_unknown_not_missing() {
        // `built_archs()` is empty in a non-CUDA test build. "We did not look"
        // must never render as "we looked and found nothing".
        let r = report_for_capability(9, 0);
        assert_eq!(r.coverage, DeviceCoverage::Unknown);
        assert!(r.summary().contains("UNVERIFIED"));
    }

    #[test]
    fn every_outcome_produces_a_line_naming_the_tier() {
        let built = arc_target::parse_arch_list("80,90").unwrap();
        for (major, minor) in [(8u32, 0u32), (8, 6), (9, 0), (10, 0)] {
            let r = TargetReport {
                capability: (major, minor),
                tier: ArcTier::for_capability(major, minor),
                coverage: arc_target::device_coverage(&built, major * 10 + minor),
            };
            let s = r.summary();
            assert!(s.starts_with("ArcTarget: "), "{s}");
            assert!(s.contains(&format!("sm_{major}{minor}")), "{s}");
        }
        // The Blackwell case must name the fix, not just the failure.
        let b = TargetReport {
            capability: (10, 0),
            tier: ArcTier::Blackwell,
            coverage: arc_target::device_coverage(&built, 100),
        };
        let s = b.summary();
        assert!(s.contains("NO cubin"), "{s}");
        assert!(s.contains("ARC_CUDA_ARCHS=80,90,100"), "{s}");
    }
}
