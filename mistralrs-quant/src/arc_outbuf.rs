//! Parent system: ArcKernels
//!
//! Output buffers for kernels contracted to write **every** element.
//!
//! `CudaDevice::alloc_zeros` costs one `cuMemsetD8Async` per buffer. Measured on
//! V4 / H200 with nsys over 160 decode forwards: **610.3 memsets per forward,
//! 3.18 us of HOST time and 1.03 us of GPU time each** — 2.57 ms of a 35.9 ms
//! step. 427.3 of those 610.3 are issued immediately in front of two kernels
//! that *assign* every element of their output:
//!
//! * `fp8_gemv_warp` (299.1/forward) —
//!   `kernels/blockwise_fp8/blockwise_fp8_gemm.cu`. The grid is
//!   `(ceil(N / ROWS_PER_BLOCK), M)`; threads outside `n < N && m < M` return
//!   early, and the surviving lane 0 does `output[m * N + n] = acc`. Every
//!   element of the `[M, N]` allocation is written exactly once. The
//!   `fp8_matmul_tiled` branch that shares the same allocation likewise ends in
//!   `if (row < M && col < N) output[row * N + col] = acc`.
//! * `qtip_gather_gemv_warp_kernel` (128.2/forward) —
//!   `kernels/qtip/qtip_gather_gemv.cu`. `y[pair * n_rows + row] = ...` for
//!   every `pair < n_pairs` and `row < n_rows`; the invalid-expert branch writes
//!   an explicit zero rather than falling through. Every element of the
//!   `[n_pairs, n_rows]` allocation is written.
//!
//! For those the zero fill is dead work: no reader ever observes it. It is also
//! a poor safety net — `gather_gemv_cuda` notes that a *failed* launch would
//! leave the zeros in place, so the memset converts "garbage" into "plausible
//! zeros", which is the silent-success trap, not a guard. The actual guard is
//! `check_gather_gemv_pairs`, which errors before the launch is issued.
//!
//! `ARC_UNINIT_OUT` selects the policy — one binary, three legs:
//!
//! | value      | behaviour              | purpose                             |
//! |------------|------------------------|-------------------------------------|
//! | unset, `1` | uninitialised `alloc`  | **the default** — the saving        |
//! | `0`        | `alloc_zeros`          | KILL SWITCH: byte-identical to before |
//! | `poison`   | `alloc` + fill `0xFF`  | the correctness leg                 |
//!
//! `poison` pays the same memset, so it is not a timing leg. `0xFF` repeated is
//! NaN in BF16, F16 and F32, so any element the kernel fails to write surfaces
//! as NaN instead of a plausible zero. Output that stays bit-identical under
//! poison is positive evidence of full coverage — evidence zeros can never give,
//! because a zero-filled miss is indistinguishable from a correct zero.
//!
//! # Why the saving is the DEFAULT and not opt-in
//!
//! It shipped opt-in ("default off, so the unset binary is byte-identical to
//! before"). That is the configuration nobody runs: the 2.57 ms is measured, so
//! every published number would have described a binary no user gets. A flag
//! here is a kill switch, not a feature request.
//!
//! Both preconditions are discharged above, statically, per kernel: the grid
//! covers the whole allocation and the write is an unconditional `=`, not a
//! `+=`, in every branch including the invalid-expert one. What remains is a
//! *runtime* property no type-check can reach, which is exactly what `poison`
//! is for — so run the poison leg on the first box, before quoting any timing
//! number from this path. If it is ever not bit-identical, `ARC_UNINIT_OUT=0`
//! restores the old behaviour without a rebuild.

#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
#[cfg(feature = "cuda")]
use candle_core::{CudaDevice, Result};
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OutBufPolicy {
    Zeroed,
    Uninit,
    Poison,
}

static POLICY: OnceLock<OutBufPolicy> = OnceLock::new();
#[cfg(feature = "cuda")]
static ENGAGED: AtomicU64 = AtomicU64::new(0);

/// Pure half of [`policy`], so the table can be tested without mutating the
/// process environment.
pub(crate) fn parse_policy(raw: Option<&str>) -> OutBufPolicy {
    match raw.map(str::trim) {
        // KILL SWITCH: restores the pre-change zero fills.
        Some("0") => OutBufPolicy::Zeroed,
        Some("poison") | Some("POISON") => OutBufPolicy::Poison,
        // Unset, "1", or anything else: the fast path. Unknown values land on
        // the DEFAULT rather than on the slow leg, so a typo cannot silently
        // cost 2.57 ms/step while looking like it was disabled on purpose.
        _ => OutBufPolicy::Uninit,
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn policy() -> OutBufPolicy {
    *POLICY.get_or_init(|| {
        let p = parse_policy(std::env::var("ARC_UNINIT_OUT").ok().as_deref());
        // Announced once, by name, in BOTH directions: a timing number from a
        // fast path that never ran is the house fault.
        eprintln!("[arc-uninit-out] policy={p:?}");
        p
    })
}

#[cfg(feature = "cuda")]
fn tick() {
    let n = ENGAGED.fetch_add(1, Ordering::Relaxed) + 1;
    if n % 20_000 == 0 {
        eprintln!("[arc-uninit-out] ENGAGED n={n}");
    }
}

/// Allocate an output buffer for a kernel that writes **every** element of it.
///
/// The caller is asserting full coverage. `ARC_UNINIT_OUT=poison` exists so that
/// assertion stays testable rather than assumed.
#[cfg(feature = "cuda")]
pub(crate) fn alloc_out_fully_written<T: DeviceRepr + ValidAsZeroBits>(
    dev: &CudaDevice,
    len: usize,
) -> Result<CudaSlice<T>> {
    match policy() {
        OutBufPolicy::Zeroed => dev.alloc_zeros::<T>(len),
        OutBufPolicy::Uninit => {
            tick();
            unsafe { dev.alloc::<T>(len) }
        }
        OutBufPolicy::Poison => {
            tick();
            let slice = unsafe { dev.alloc::<T>(len) }?;
            let bytes = len * std::mem::size_of::<T>();
            if bytes > 0 {
                let stream = dev.cuda_stream();
                let (ptr, _guard) = crate::utils::slice_ptr(&slice, 0);
                unsafe {
                    candle_core::cuda::cudarc::driver::result::memset_d8_async(
                        ptr as candle_core::cuda::cudarc::driver::sys::CUdeviceptr,
                        0xFF,
                        bytes,
                        stream.cu_stream(),
                    )
                }
                .map_err(|e| {
                    candle_core::Error::Msg(format!("arc-uninit-out poison memset failed: {e}"))
                })?;
            }
            Ok(slice)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_policy, OutBufPolicy};

    /// The policy table, pinned — most of all its DEFAULT.
    ///
    /// This shipped as `_ => Zeroed` with the saving behind `ARC_UNINIT_OUT=1`,
    /// which meant the measured 2.57 ms/step was unreachable for anyone who did
    /// not know the magic word, and every published number described a binary
    /// no user runs. If someone flips it back, this test says so.
    #[test]
    fn uninit_out_defaults_to_the_fast_path() {
        // THE POINT: no flag set == the saving.
        assert_eq!(parse_policy(None), OutBufPolicy::Uninit);
        assert_eq!(parse_policy(Some("1")), OutBufPolicy::Uninit);
        assert_eq!(parse_policy(Some("")), OutBufPolicy::Uninit);

        // "0" is the kill switch, and the ONLY way back to the zero fills.
        assert_eq!(parse_policy(Some("0")), OutBufPolicy::Zeroed);
        assert_eq!(parse_policy(Some(" 0 ")), OutBufPolicy::Zeroed);

        // The correctness leg stays reachable.
        assert_eq!(parse_policy(Some("poison")), OutBufPolicy::Poison);
        assert_eq!(parse_policy(Some("POISON")), OutBufPolicy::Poison);

        // A typo must not silently buy the slow leg back.
        for junk in ["yes", "true", "off", "zeroed", "2"] {
            assert_eq!(
                parse_policy(Some(junk)),
                OutBufPolicy::Uninit,
                "{junk:?} must fall to the default, not to Zeroed"
            );
        }
    }
}
