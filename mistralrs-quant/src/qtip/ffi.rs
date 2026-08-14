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

    // =========================================================================
    // qtip2b — bitshift-trellis computed-codebook kernels (kernels/qtip/
    // qtip_bitshift.cu). K=2/V=1/L=16; codebook computed in-register from the
    // MCG `mult` (no LUT pointer anywhere in this ABI).
    // =========================================================================

    // ----- Dequantize (rotated frame) -----
    pub(crate) fn launch_qtip2b_dequantize_bf16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_out: *mut bf16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip2b_dequantize_f16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_out: *mut f16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip2b_dequantize_f32(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_out: *mut f32,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // ----- Fused decode + GEMV -----
    //
    // `d_indices == null` selects the plain 2-D path (every pair reads
    // "expert 0", i.e. the whole [n_rows, packed_per_row] matrix); non-null
    // gathers each pair's expert id ON-DEVICE (capturable MoE dispatch).
    // `d_x_rotated` must already be in the QTIP-rotated frame.
    pub(crate) fn launch_qtip2b_gemv_bf16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_x_rotated: *const bf16,
        d_indices: *const u32,
        d_y: *mut bf16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        n_pairs: i32,
        num_experts: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip2b_gemv_f16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_x_rotated: *const f16,
        d_indices: *const u32,
        d_y: *mut f16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        n_pairs: i32,
        num_experts: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip2b_gemv_f32(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_x_rotated: *const f32,
        d_indices: *const u32,
        d_y: *mut f32,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        n_pairs: i32,
        num_experts: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // ----- Autotuned GEMV variants -----------------------------------------
    //
    // Two compile-time instantiated grids behind one dispatch entry:
    //   gen 1 (`qtip_bitshift_tune.cu`, ids 0..43): {warps/block, rows/warp,
    //     ILP streams/thread, vector load width, launch_bounds min-blocks,
    //     smem staging, prefetch depth}.
    //   gen 2 (`qtip_bitshift_tune2.cu`, ids 44..): cp.async double/triple-
    //     buffered staged pipeline over {consumer warps, rows/warp, ILP
    //     streams from smem, staged load width, split-K, pipeline stages,
    //     producer warps, min-blocks}.
    // The tuned launchers return 0 on launch and -1 when the variant is not
    // applicable to the shape (alignment / smem capacity) — the caller must
    // then fall back to `launch_qtip2b_gemv_*`, whose correctness contract
    // the variants share.
    pub(crate) fn qtip2b_gemv_num_variants() -> i32;

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qtip2b_gemv_variant_info(
        idx: i32,
        warps: *mut i32,
        rows_per_warp: *mut i32,
        ilp: *mut i32,
        wbytes: *mut i32,
        min_blocks: *mut i32,
        stage: *mut i32,
        prefetch: *mut i32,
        generation: *mut i32,
        ksplit: *mut i32,
        stages: *mut i32,
        producer_warps: *mut i32,
    ) -> i32;

    pub(crate) fn launch_qtip2b_gemv_tuned_bf16(
        variant: i32,
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_x_rotated: *const bf16,
        d_indices: *const u32,
        d_y: *mut bf16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        n_pairs: i32,
        num_experts: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    ) -> i32;

    pub(crate) fn launch_qtip2b_gemv_tuned_f16(
        variant: i32,
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_x_rotated: *const f16,
        d_indices: *const u32,
        d_y: *mut f16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        n_pairs: i32,
        num_experts: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    ) -> i32;

    pub(crate) fn launch_qtip2b_gemv_tuned_f32(
        variant: i32,
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_x_rotated: *const f32,
        d_indices: *const u32,
        d_y: *mut f32,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        n_pairs: i32,
        num_experts: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    ) -> i32;

    // ----- Quantize side (F32 weights, same contract as the LUT rung) -----
    pub(crate) fn launch_qtip2b_compute_row_scales_f32(
        d_weight: *const f32,
        d_row_scales: *mut f32,
        n_rows: i32,
        in_features: i32,
        divisor: f32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip2b_quantize_rows_greedy_f32(
        d_weight: *const f32,
        d_row_scales: *const f32,
        d_packed: *mut u8,
        n_rows: i32,
        in_features: i32,
        num_symbols: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip2b_quantize_rows_viterbi_f32(
        d_weight: *const f32,
        d_row_scales: *const f32,
        d_packed: *mut u8,
        d_cost_a: *mut f32,
        d_cost_b: *mut f32,
        d_prefix_cost: *mut f32,
        d_backtrace: *mut u8,
        n_rows: i32,
        in_features: i32,
        num_symbols: i32,
        row_offset: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // ----- Grouped GEMM (batched MoE prefill; kernels/qtip/qtip_grouped_gemm.cu) -----
    //
    // On-device routing (histogram + scans/tile-map + grouped scatter) for
    // the trellis grouped GEMM. ZERO host syncs: `d_num_tiles` stays on the
    // device and the GEMM grid is sized from the host-side upper bound
    // `max_m_tiles = ceil(n_pairs / TILE_M) + num_experts`. `d_counts` and
    // `d_cursors` must be zero-initialized.
    pub(crate) fn launch_qtip2b_moe_route(
        d_indices: *const u32,
        d_counts: *mut u32,
        d_offsets: *mut u32,
        d_cursors: *mut u32,
        d_tile_prefix: *mut u32,
        d_tile_expert: *mut u32,
        d_tile_row_start: *mut u32,
        d_num_tiles: *mut u32,
        d_sorted_pairs: *mut u32,
        n_pairs: i32,
        num_experts: i32,
        tile_m: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // W2A16 trellis grouped GEMM over expert-sorted (token, slot) pairs.
    // `d_x_rotated` must already be in the QTIP-rotated frame; `d_y`
    // ([n_pairs, n_rows]) must be zero-initialized (rows dropped by routing
    // are never written). 16-bit activation dtypes only — the mma.sync
    // pipeline is the point of this kernel; F32 callers use the fallback
    // paths.
    pub(crate) fn launch_qtip2b_grouped_gemm_bf16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_x_rotated: *const bf16,
        d_sorted_pairs: *const u32,
        d_tile_expert: *const u32,
        d_tile_row_start: *const u32,
        d_offsets: *const u32,
        d_num_tiles: *const u32,
        d_y: *mut bf16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        max_m_tiles: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_qtip2b_grouped_gemm_f16(
        d_packed: *const u8,
        d_row_scales: *const f32,
        d_x_rotated: *const f16,
        d_sorted_pairs: *const u32,
        d_tile_expert: *const u32,
        d_tile_row_start: *const u32,
        d_offsets: *const u32,
        d_num_tiles: *const u32,
        d_y: *mut f16,
        n_rows: i32,
        packed_per_row: i32,
        num_symbols: i32,
        max_m_tiles: i32,
        mult: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
