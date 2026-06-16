use core::ffi::{c_int, c_void};

extern "C" {
    /// Device-indexed in-place KV-cache write. See `kernels/kvwrite/kvwrite.cu`.
    ///
    /// * `all_data`  - [B, H, capacity, D] contiguous cache buffer (mutated).
    /// * `src`       - [B, H, 1, D] contiguous new K/V for this token.
    /// * `positions` - [B] device i32/u32: seq slot to write per batch row.
    /// * `stream`    - candle's capture stream (so the launch is recorded into
    ///                 the CUDA graph). Null falls back to the default stream.
    pub(crate) fn kvcache_write(
        all_data: *mut c_void,
        src: *const c_void,
        positions: *const c_int,
        b: c_int,
        h: c_int,
        capacity: c_int,
        d: c_int,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
        dtype: u32,
    );
}
