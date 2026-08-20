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
//! | value      | behaviour              | purpose                          |
//! |------------|------------------------|----------------------------------|
//! | unset, `0` | `alloc_zeros`          | base leg (byte-identical to before) |
//! | `1`        | uninitialised `alloc`  | the saving                       |
//! | `poison`   | `alloc` + fill `0xFF`  | the correctness leg              |
//!
//! `poison` pays the same memset, so it is not a timing leg. `0xFF` repeated is
//! NaN in BF16, F16 and F32, so any element the kernel fails to write surfaces
//! as NaN instead of a plausible zero. Output that stays bit-identical under
//! poison is positive evidence of full coverage — evidence zeros can never give,
//! because a zero-filled miss is indistinguishable from a correct zero.

use candle_core::cuda::cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
use candle_core::{CudaDevice, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OutBufPolicy {
    Zeroed,
    Uninit,
    Poison,
}

static POLICY: OnceLock<OutBufPolicy> = OnceLock::new();
static ENGAGED: AtomicU64 = AtomicU64::new(0);

fn policy() -> OutBufPolicy {
    *POLICY.get_or_init(|| {
        let p = match std::env::var("ARC_UNINIT_OUT").as_deref() {
            Ok("1") => OutBufPolicy::Uninit,
            Ok("poison") => OutBufPolicy::Poison,
            _ => OutBufPolicy::Zeroed,
        };
        // Announced once, by name, in BOTH directions: a timing number from a
        // fast path that never ran is the house fault.
        eprintln!("[arc-uninit-out] policy={p:?}");
        p
    })
}

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
