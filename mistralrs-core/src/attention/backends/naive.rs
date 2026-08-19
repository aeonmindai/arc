#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::MemoryUsage;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::MatMul;

use crate::attention::{chunked_attention, SdpaParams};

/// Free-VRAM floor below which [`maybe_synchronize`] performs a full device
/// synchronize.
///
/// The same number bounds CUDA-graph capture, which refuses to start with less
/// than this much headroom — see [`arc_cuda_graph::CAPTURE_SYNC_HEADROOM_BYTES`]
/// and the compile-time assertion below that pins the two together.
#[cfg(target_pointer_width = "64")]
pub(crate) const SYNC_HEADROOM_BYTES: usize = 4 * 1024 * 1024 * 1024;
#[cfg(not(target_pointer_width = "64"))]
pub(crate) const SYNC_HEADROOM_BYTES: usize = usize::MAX;

/// The two thresholds are one decision and must not drift apart. A capture that
/// is allowed to start below the sync floor records a graph whose replays skip
/// a synchronize the eager path performs.
#[cfg(feature = "cuda")]
const _: () = assert!(SYNC_HEADROOM_BYTES == arc_cuda_graph::CAPTURE_SYNC_HEADROOM_BYTES);

/// Whether [`maybe_synchronize`] should issue the device synchronize.
///
/// Split out from the call so the rule is testable without a GPU — the whole
/// defect this guards is that the rule was previously unobservable except by
/// running out of memory on a real box.
///
/// `capturing` is candle's `CudaDevice::capture_mode()`, set by arc around a
/// CUDA-graph capture and around the deferred-free pass that precedes it.
pub(crate) fn should_synchronize(available_bytes: usize, capturing: bool) -> bool {
    !capturing && available_bytes < SYNC_HEADROOM_BYTES
}

/// Not *really* sure why this is necessary but it is.
///
/// # Why this is capture-aware
///
/// `device.synchronize()` is `cuCtxSynchronize`, which CUDA forbids while a
/// stream capture is in flight — it fails the capture outright. This function
/// is called unconditionally from [`naive_sdpa`] and from the cuBLASLt path in
/// `attention/mod.rs`, i.e. from inside the exact region decode capture wants
/// to record, and it fires on a *memory-pressure* condition rather than a
/// deterministic one. That combination is the worst kind of capture blocker:
/// it is invisible on a box with headroom and intermittent on one without.
///
/// So during capture the synchronize is skipped rather than attempted. The
/// other half of the fix lives in `arc-cuda-graph`, which refuses to begin a
/// capture with less than [`SYNC_HEADROOM_BYTES`] free — because a capture
/// taken below that floor would record a graph that skips a synchronize every
/// eager step performs, and the two would silently disagree.
pub(crate) fn maybe_synchronize(device: &Device) -> Result<()> {
    #[allow(unused_mut)]
    let mut capturing = false;
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cd) = device {
        capturing = cd.capture_mode();
    }
    if should_synchronize(MemoryUsage.get_memory_available(device)?, capturing) {
        device.synchronize()?;
    }
    Ok(())
}

/// Computes softmax(QK^T*sqrt(d_k))V
pub(crate) fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    maybe_synchronize(q.device())?;

    // Use chunked attention with a closure that captures the necessary parameters
    chunked_attention(q, k, v, mask, |q_chunk, k, v, mask_chunk| {
        let mut att =
            MatMul.matmul_affine_mul(q_chunk, &k.t()?, sdpa_params.softmax_scale.into())?;

        if let Some(softcap) = sdpa_params.softcap {
            att = (att / softcap as f64)?;
            att = att.tanh()?;
            att = (att * softcap as f64)?;
        }

        if let Some(mask) = mask_chunk {
            att = att.broadcast_add(mask)?;
        }

        att = candle_nn::ops::softmax_last_dim(&att)?;
        MatMul.matmul(&att, v)
    })
}

/// The synchronize rule. Previously unobservable except by running a real box
/// out of memory, which is why a full device synchronize sat inside the region
/// decode capture wants to record without anyone noticing.
#[cfg(test)]
mod sync_policy_tests {
    use super::{should_synchronize, SYNC_HEADROOM_BYTES};

    /// The behaviour that must not regress: under pressure and not capturing,
    /// the synchronize still happens. Upstream says it is necessary and does
    /// not say why, so it is preserved exactly outside capture.
    #[test]
    fn low_memory_outside_capture_still_synchronizes() {
        assert!(should_synchronize(SYNC_HEADROOM_BYTES - 1, false));
        assert!(should_synchronize(0, false));
    }

    /// THE FIX. `cuCtxSynchronize` during a stream capture fails the capture.
    /// Memory pressure must not be able to reach into a capture region.
    #[test]
    fn low_memory_during_capture_does_not_synchronize() {
        assert!(!should_synchronize(SYNC_HEADROOM_BYTES - 1, true));
        assert!(!should_synchronize(0, true));
    }

    /// With headroom the call is free either way — this is the state every
    /// trace so far was taken in, which is exactly why the defect was invisible.
    #[test]
    fn ample_memory_never_synchronizes() {
        assert!(!should_synchronize(SYNC_HEADROOM_BYTES, false));
        assert!(!should_synchronize(SYNC_HEADROOM_BYTES * 4, false));
        assert!(!should_synchronize(SYNC_HEADROOM_BYTES * 4, true));
    }

    /// The comparison is strict `<`, so exactly-at-the-floor does not fire.
    /// Pinned because `arc-cuda-graph`'s capture refusal uses the same strict
    /// comparison and the two must agree on the boundary case.
    #[test]
    fn the_floor_itself_is_not_below_the_floor() {
        assert!(!should_synchronize(SYNC_HEADROOM_BYTES, false));
        assert!(should_synchronize(SYNC_HEADROOM_BYTES - 1, false));
    }
}
