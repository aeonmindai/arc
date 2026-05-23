use half::{bf16, f16};

pub(crate) const HAVE_QTIP_KERNELS: bool = cfg!(has_qtip_kernels);

#[cfg(feature = "cuda")]
extern "C" {
    pub(crate) fn launch_qtip_dequantize_v2_k4_l16_bf16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_lut: *const f32,
        d_out: *mut bf16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_dequantize_v2_k4_l16_f16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_lut: *const f32,
        d_out: *mut f16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_dequantize_v2_k4_l16_f32(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_lut: *const f32,
        d_out: *mut f32,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
