//! CUDA peer-to-peer (NVLink P2P) access enablement.
//!
//! Neither candle nor cudarc ever calls `cuCtxEnablePeerAccess`, so a
//! cross-GPU `Tensor::to_device` is staged through host RAM: device -> pinned
//! host -> device, over PCIe, twice. The NVIDIA CUDA Programming Guide
//! ("Multi-GPU Systems" / peer-to-peer memory access) is explicit that
//! peer-to-peer copies fall back to a host staging buffer unless peer access
//! has been enabled for the context pair. On an NVLink box that is the
//! difference between ~450 GB/s and ~20 GB/s per hop.
//!
//! [`enable_peer_access`] walks every ordered pair of the CUDA devices it is
//! given and enables access in both directions. Pairs the driver says cannot
//! peer (no NVLink and no P2P-capable PCIe topology, different IOMMU domains,
//! MIG, ...) are recorded as [`PeerAccessStatus::Unsupported`] and are NOT an
//! error: peer access is an optimization, and a fleet node with a partial
//! topology must still boot.
//!
//! The report type and its formatting are deliberately free of any CUDA types
//! so they can be unit-tested on a host with no GPU.
//!
//! cudarc symbols used (cudarc 0.19.4, the version candle-core 0.9.2 pins):
//! * `driver::sys::cuDeviceCanAccessPeer` — `src/driver/sys/mod.rs:10287`
//! * `driver::sys::cuCtxEnablePeerAccess` — `src/driver/sys/mod.rs:10133`
//! * `driver::sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED` — `:6605`
//! * `driver::result::ctx::{get_current, set_current}` — `src/driver/result.rs:553,546`
//! * `CudaContext::{cu_device, cu_ctx, bind_to_thread}` — `src/driver/safe/core.rs:333,345,350`

use std::fmt;

/// What happened for one ordered `(src, dst)` device pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerAccessStatus {
    /// `cuCtxEnablePeerAccess` succeeded on this call.
    Enabled,
    /// The driver reported `CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED`; someone
    /// (a previous call, NCCL, another library) already enabled this direction.
    AlreadyEnabled,
    /// The driver says this pair cannot peer. Copies stay host-staged.
    Unsupported,
}

impl PeerAccessStatus {
    /// True when the direction ends up peered, however it got there.
    pub fn is_peered(self) -> bool {
        matches!(self, Self::Enabled | Self::AlreadyEnabled)
    }

    /// Stable, log-friendly name for this status.
    pub fn label(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::AlreadyEnabled => "already-enabled",
            Self::Unsupported => "unsupported",
        }
    }
}

/// One ordered pair's outcome: access FROM `src` TO `dst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerPair {
    pub src: usize,
    pub dst: usize,
    pub status: PeerAccessStatus,
}

/// Per-ordered-pair record of what [`enable_peer_access`] did.
///
/// Entries keep insertion order, so the summary is deterministic.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PeerAccessReport {
    pairs: Vec<PeerPair>,
}

impl PeerAccessReport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the outcome for `src -> dst`.
    pub fn record(&mut self, src: usize, dst: usize, status: PeerAccessStatus) {
        self.pairs.push(PeerPair { src, dst, status });
    }

    /// All recorded pairs, in the order they were attempted.
    pub fn pairs(&self) -> &[PeerPair] {
        &self.pairs
    }

    /// Status for one direction, if it was attempted.
    pub fn status(&self, src: usize, dst: usize) -> Option<PeerAccessStatus> {
        self.pairs
            .iter()
            .find(|p| p.src == src && p.dst == dst)
            .map(|p| p.status)
    }

    /// `(enabled, already_enabled, unsupported)` counts.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut counts = (0, 0, 0);
        for pair in &self.pairs {
            match pair.status {
                PeerAccessStatus::Enabled => counts.0 += 1,
                PeerAccessStatus::AlreadyEnabled => counts.1 += 1,
                PeerAccessStatus::Unsupported => counts.2 += 1,
            }
        }
        counts
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// True when every attempted direction is peered. An empty report (fewer
    /// than two CUDA devices) is vacuously fully peered.
    pub fn all_peered(&self) -> bool {
        self.pairs.iter().all(|p| p.status.is_peered())
    }

    /// One-line, log-friendly summary naming every unsupported direction.
    pub fn summary(&self) -> String {
        if self.pairs.is_empty() {
            return "CUDA peer access: no device pairs to enable".to_string();
        }
        let (enabled, already, unsupported) = self.counts();
        let mut out = format!(
            "CUDA peer access: {enabled} enabled, {already} already-enabled, {unsupported} unsupported"
        );
        if unsupported > 0 {
            let names: Vec<String> = self
                .pairs
                .iter()
                .filter(|p| p.status == PeerAccessStatus::Unsupported)
                .map(|p| format!("cuda:{}->cuda:{}", p.src, p.dst))
                .collect();
            out.push_str(&format!(
                " (host-staged: {}) — cross-device copies on those hops go through host RAM",
                names.join(", ")
            ));
        }
        out
    }
}

impl fmt::Display for PeerAccessReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

/// Enable bidirectional CUDA peer access between every pair of `devices`.
///
/// Non-CUDA devices and duplicate ordinals are skipped. A pair the driver
/// cannot peer is recorded as [`PeerAccessStatus::Unsupported`] rather than
/// returned as an error. Idempotent: a second call reports `AlreadyEnabled`.
#[cfg(feature = "cuda")]
pub fn enable_peer_access(
    devices: &[candle_core::Device],
) -> candle_core::Result<PeerAccessReport> {
    use candle_core::cuda::cudarc::driver::{result as cu_result, sys, CudaContext};
    use candle_core::cuda::WrapErr;
    use candle_core::{Device, DeviceLocation};
    use std::sync::Arc;

    // Unique CUDA ordinals, each with the context candle actually allocates on.
    let mut ctxs: Vec<(usize, Arc<CudaContext>)> = Vec::new();
    for device in devices {
        let Device::Cuda(cuda_device) = device else {
            continue;
        };
        let DeviceLocation::Cuda { gpu_id } = device.location() else {
            continue;
        };
        if ctxs.iter().any(|(ordinal, _)| *ordinal == gpu_id) {
            continue;
        }
        ctxs.push((gpu_id, cuda_device.cuda_stream().context().clone()));
    }

    let mut report = PeerAccessReport::new();
    if ctxs.len() < 2 {
        return Ok(report);
    }

    // `cuCtxEnablePeerAccess` acts on the CURRENT context, so we have to bind
    // each source context in turn. Restore whatever was current afterwards so
    // this is invisible to the calling thread.
    let previous = cu_result::ctx::get_current().w()?;

    for (i, (src_ordinal, src_ctx)) in ctxs.iter().enumerate() {
        let mut bound = false;
        for (j, (dst_ordinal, dst_ctx)) in ctxs.iter().enumerate() {
            if i == j {
                continue;
            }

            // Can these two peer at all? A driver error here is treated as
            // "cannot peer" — never fatal, never a panic.
            let mut can_access: std::ffi::c_int = 0;
            let query = unsafe {
                sys::cuDeviceCanAccessPeer(
                    &mut can_access,
                    src_ctx.cu_device(),
                    dst_ctx.cu_device(),
                )
            }
            .result();
            if query.is_err() || can_access == 0 {
                report.record(*src_ordinal, *dst_ordinal, PeerAccessStatus::Unsupported);
                continue;
            }

            if !bound {
                src_ctx.bind_to_thread().w()?;
                bound = true;
            }

            let status = match unsafe { sys::cuCtxEnablePeerAccess(dst_ctx.cu_ctx(), 0) } {
                sys::CUresult::CUDA_SUCCESS => PeerAccessStatus::Enabled,
                sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
                    PeerAccessStatus::AlreadyEnabled
                }
                // Topology / capacity limits: degrade to host staging.
                sys::CUresult::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED
                | sys::CUresult::CUDA_ERROR_TOO_MANY_PEERS
                | sys::CUresult::CUDA_ERROR_INVALID_DEVICE => PeerAccessStatus::Unsupported,
                // Anything else is a real driver fault, not a topology limit.
                other => {
                    return Err(candle_core::Error::Msg(format!(
                        "cuCtxEnablePeerAccess(cuda:{src_ordinal} -> cuda:{dst_ordinal}) failed: {other:?}"
                    ))
                    .bt())
                }
            };
            report.record(*src_ordinal, *dst_ordinal, status);
        }
    }

    if let Some(previous) = previous {
        unsafe { cu_result::ctx::set_current(previous) }.w()?;
    }

    Ok(report)
}

/// Non-CUDA builds: nothing to enable, so report nothing and succeed. Callers
/// can invoke this unconditionally.
#[cfg(not(feature = "cuda"))]
pub fn enable_peer_access(
    _devices: &[candle_core::Device],
) -> candle_core::Result<PeerAccessReport> {
    Ok(PeerAccessReport::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_report_is_vacuously_peered() {
        let report = PeerAccessReport::new();
        assert!(report.is_empty());
        assert_eq!(report.len(), 0);
        assert!(report.all_peered());
        assert_eq!(
            report.summary(),
            "CUDA peer access: no device pairs to enable"
        );
    }

    #[test]
    fn counts_split_the_three_statuses() {
        let mut report = PeerAccessReport::new();
        report.record(0, 1, PeerAccessStatus::Enabled);
        report.record(1, 0, PeerAccessStatus::Enabled);
        report.record(0, 2, PeerAccessStatus::AlreadyEnabled);
        report.record(2, 0, PeerAccessStatus::Unsupported);
        assert_eq!(report.counts(), (2, 1, 1));
        assert_eq!(report.len(), 4);
    }

    #[test]
    fn summary_names_every_unsupported_direction() {
        let mut report = PeerAccessReport::new();
        report.record(0, 1, PeerAccessStatus::Enabled);
        report.record(1, 0, PeerAccessStatus::AlreadyEnabled);
        report.record(0, 3, PeerAccessStatus::Unsupported);
        report.record(3, 0, PeerAccessStatus::Unsupported);
        let summary = report.summary();
        assert!(
            summary.starts_with("CUDA peer access: 1 enabled, 1 already-enabled, 2 unsupported"),
            "unexpected counts in summary: {summary}"
        );
        assert!(
            summary.contains("cuda:0->cuda:3"),
            "summary must name the unsupported direction: {summary}"
        );
        assert!(
            summary.contains("cuda:3->cuda:0"),
            "summary must name the reverse direction too: {summary}"
        );
        // Peered pairs must NOT be listed as host-staged.
        assert!(
            !summary.contains("cuda:1->cuda:0"),
            "peered pairs must not appear in the host-staged list: {summary}"
        );
    }

    #[test]
    fn summary_omits_host_staged_clause_when_all_peered() {
        let mut report = PeerAccessReport::new();
        report.record(0, 1, PeerAccessStatus::Enabled);
        report.record(1, 0, PeerAccessStatus::AlreadyEnabled);
        let summary = report.summary();
        assert!(!summary.contains("host-staged"), "{summary}");
        assert!(report.all_peered());
    }

    #[test]
    fn all_peered_is_false_when_any_direction_is_unsupported() {
        let mut report = PeerAccessReport::new();
        report.record(0, 1, PeerAccessStatus::Enabled);
        report.record(1, 0, PeerAccessStatus::Unsupported);
        assert!(!report.all_peered());
    }

    #[test]
    fn status_lookup_is_direction_sensitive() {
        let mut report = PeerAccessReport::new();
        report.record(0, 1, PeerAccessStatus::Enabled);
        report.record(1, 0, PeerAccessStatus::Unsupported);
        assert_eq!(report.status(0, 1), Some(PeerAccessStatus::Enabled));
        assert_eq!(report.status(1, 0), Some(PeerAccessStatus::Unsupported));
        assert_eq!(report.status(0, 2), None);
    }

    #[test]
    fn already_enabled_counts_as_peered_but_enabled_is_distinct() {
        assert!(PeerAccessStatus::Enabled.is_peered());
        assert!(PeerAccessStatus::AlreadyEnabled.is_peered());
        assert!(!PeerAccessStatus::Unsupported.is_peered());
        assert_ne!(PeerAccessStatus::Enabled, PeerAccessStatus::AlreadyEnabled);
        assert_eq!(PeerAccessStatus::Enabled.label(), "enabled");
        assert_eq!(PeerAccessStatus::AlreadyEnabled.label(), "already-enabled");
        assert_eq!(PeerAccessStatus::Unsupported.label(), "unsupported");
    }

    #[test]
    fn display_matches_summary() {
        let mut report = PeerAccessReport::new();
        report.record(0, 1, PeerAccessStatus::Unsupported);
        assert_eq!(report.to_string(), report.summary());
    }

    /// Without a GPU we cannot exercise the driver, but we CAN pin the
    /// no-CUDA-devices contract: a CPU-only device list must never error and
    /// must report nothing to enable.
    #[test]
    fn cpu_only_device_list_reports_nothing() {
        let report = enable_peer_access(&[candle_core::Device::Cpu, candle_core::Device::Cpu])
            .expect("a CPU-only device list must not error");
        assert!(report.is_empty());
    }
}
