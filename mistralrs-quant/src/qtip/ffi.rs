//! FFI declarations for the QTIP CUDA kernels in `kernels/qtip/`.
//!
//! The kernel ABI (extern "C") and grid-launch policies live in
//! `qtip_dequantize.cu`. This file is the type-safe Rust mirror.

#[cfg(feature = "cuda")]
use half::{bf16, f16};

/// `true` when the QTIP CUDA kernels were compiled in (SM >= 80 at
/// build time). Gated by build.rs based on the detected GPU compute
/// capability.
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

    pub(crate) fn launch_qtip_rotate_x_bf16(
        d_x: *mut bf16,
        d_signs: *const f32,
        batch_size: i32,
        in_features: i32,
        block_size: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_rotate_x_f16(
        d_x: *mut f16,
        d_signs: *const f32,
        batch_size: i32,
        in_features: i32,
        block_size: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_rotate_x_f32(
        d_x: *mut f32,
        d_signs: *const f32,
        batch_size: i32,
        in_features: i32,
        block_size: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // ----- Quantize-side kernels (RUN-quant-on-gpu): rotate weight rows,
    // compute per-row scales, greedy + Viterbi DP quantizers. F32 only — the
    // quantize path always converts the input weight to FP32 first.
    pub(crate) fn launch_qtip_rotate_weight_rows_f32(
        d_weight: *mut f32,
        d_signs: *const f32,
        n_rows: i32,
        in_features: i32,
        block_size: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_compute_row_scales_f32(
        d_weight: *const f32,
        d_row_scales: *mut f32,
        n_rows: i32,
        in_features: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_quantize_rows_greedy_f32(
        d_weight: *const f32,
        d_lut: *const f32,
        d_row_scales: *const f32,
        d_packed: *mut u8,
        n_rows: i32,
        in_features: i32,
        num_symbols: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_quantize_rows_viterbi_f32(
        d_weight: *const f32,
        d_lut: *const f32,
        d_row_scales: *const f32,
        d_packed: *mut u8,
        d_cost_a: *mut f32,
        d_cost_b: *mut f32,
        d_backtrace: *mut u8,
        n_rows: i32,
        in_features: i32,
        num_symbols: i32,
        row_offset: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_refine_scales_f32(
        d_weight: *const f32,
        d_packed: *const u8,
        d_lut: *const f32,
        d_row_scales: *mut f32,
        n_rows: i32,
        in_features: i32,
        num_symbols: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // ----- Fused decode + gemv kernels (single-token decode fast path) -----
    //
    // `x_rotated` MUST already be in the QTIP-rotated frame (caller applies
    // `launch_qtip_rotate_x_*` first). The kernel reads packed bytes once,
    // decodes them in registers, and accumulates `y[row] = sum W[row,k] * x[k]`
    // — saves 2× HBM weight bandwidth vs dequantize-then-matmul.
    pub(crate) fn launch_qtip_fused_gemv_v2_k4_l16_bf16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_lut: *const f32,
        d_x_rotated: *const bf16,
        d_y: *mut bf16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_fused_gemv_v2_k4_l16_f16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_lut: *const f32,
        d_x_rotated: *const f16,
        d_y: *mut f16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_fused_gemv_v2_k4_l16_f32(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_lut: *const f32,
        d_x_rotated: *const f32,
        d_y: *mut f32,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // ----- On-device MoE gather + fused decode + gemv (capturable decode) -----
    //
    // Reads each pair's expert id from `d_indices` ON-DEVICE (no host sync),
    // offsets into the 3-D stacked-expert `d_packed`/`d_row_scales`, and writes
    // `d_y[pair, row] = sum_k W_expert[row,k] * x[pair,k]`. `d_x_rotated` must
    // already be in the QTIP-rotated frame. n_pairs = n_tokens * top_k.
    pub(crate) fn launch_qtip_gather_gemv_v2_k4_l16_bf16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_lut: *const f32,
        d_x_rotated: *const bf16,
        d_indices: *const u32,
        d_y: *mut bf16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        n_pairs: i32,
        num_experts: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_gather_gemv_v2_k4_l16_f16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_lut: *const f32,
        d_x_rotated: *const f16,
        d_indices: *const u32,
        d_y: *mut f16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        n_pairs: i32,
        num_experts: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip_gather_gemv_v2_k4_l16_f32(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_lut: *const f32,
        d_x_rotated: *const f32,
        d_indices: *const u32,
        d_y: *mut f32,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        n_pairs: i32,
        num_experts: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
