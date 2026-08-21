use float8::F8E4M3;
use half::{bf16, f16};

pub(crate) const HAVE_BLOCKWISE_DEQUANT_KERNELS: bool = cfg!(has_blockwise_fp8_kernels);
pub(crate) const HAVE_BLOCKWISE_QUANT_KERNELS: bool = cfg!(has_blockwise_fp8_kernels);
pub(crate) const HAVE_BLOCKWISE_GEMM_KERNELS: bool = cfg!(has_blockwise_fp8_kernels);

extern "C" {
    pub(crate) fn launch_dequant_fp8_blockwise_kernel_f32(
        d_weight: *const F8E4M3,
        d_scale: *const f32,
        d_output: *mut f32,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_dequant_fp8_blockwise_kernel_f16(
        d_weight: *const F8E4M3,
        d_scale: *const f32,
        d_output: *mut f16,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_dequant_fp8_blockwise_kernel_bf16(
        d_weight: *const F8E4M3,
        d_scale: *const f32,
        d_output: *mut bf16,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_quant_fp8_blockwise_kernel_f32(
        d_input: *const f32,
        d_weight: *mut F8E4M3,
        d_scale: *mut f32,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_quant_fp8_blockwise_kernel_f16(
        d_input: *const f16,
        d_weight: *mut F8E4M3,
        d_scale: *mut f32,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_quant_fp8_blockwise_kernel_bf16(
        d_input: *const bf16,
        d_weight: *mut F8E4M3,
        d_scale: *mut f32,
        weight_height: i32,
        weight_width: i32,
        weight_row_stride: i32,
        scale_stride: i32,
        weight_block_size_y: i32,
        weight_block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // FP8 Matmul kernels (for forward method)
    pub(crate) fn launch_fp8_matmul_f16(
        input: *const f16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut f16,
        m: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_fp8_matmul_bf16(
        input: *const bf16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut bf16,
        m: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // ---- Tensor-core blockwise-FP8 GEMM (kernels/blockwise_fp8/
    // blockwise_fp8_gemm_wmma.cu). Signature-identical to
    // `launch_fp8_matmul_*` above so the two are swappable behind one boolean.
    //
    // UNVERIFIED ON HARDWARE -- never run. See the kernel's header comment.
    pub(crate) fn launch_fp8_matmul_wmma_f16(
        input: *const f16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut f16,
        m: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_fp8_matmul_wmma_bf16(
        input: *const bf16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut bf16,
        m: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    /// Report the WMMA kernel's block tiling so the Rust eligibility test
    /// cannot drift from the tiling the kernel was actually compiled with.
    /// The dispatcher's preconditions (`block_size_y % n_blk == 0`,
    /// `block_size_x % k_blk == 0`) are what make the block scale a single
    /// scalar per tile, which is the kernel's whole premise.
    pub(crate) fn fp8_matmul_wmma_tile_dims(m_blk: *mut i32, n_blk: *mut i32, k_blk: *mut i32);

    // FP8 GEMV kernels (dedicated decode path, M <= 4). Warp-per-row,
    // dequant-in-registers with per-block scales, f32 accumulate.
    pub(crate) fn launch_fp8_gemv_f16(
        input: *const f16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut f16,
        m: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_fp8_gemv_bf16(
        input: *const bf16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut bf16,
        m: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // ---- Wide (128-bit-load) FP8 GEMV. Same decode regime as
    // `launch_fp8_gemv_*` above, but 16 bytes of weight per thread per load
    // instead of 4, software-pipelined so two iterations are always in
    // flight, and `scale_shift` (= log2(block_size_x)) in place of the
    // runtime integer division the 32-bit kernel does once per four weight
    // bytes. See the kernel header in
    // `kernels/blockwise_fp8/blockwise_fp8_gemm.cu`.
    //
    // 🔴 UNVERIFIED ON HARDWARE — never run. Opt-in via `ARC_FP8_GEMV_WIDE=1`.
    // NOT bit-identical to `launch_fp8_gemv_*`: it re-associates the f32
    // accumulation (four accumulators per lane instead of one) over an
    // identically-computed set of 4-element groups.
    pub(crate) fn launch_fp8_gemv_wide_f16(
        input: *const f16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut f16,
        m: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        scale_shift: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_fp8_gemv_wide_bf16(
        input: *const bf16,
        weight: *const F8E4M3,
        weight_scale: *const f32,
        output: *mut bf16,
        m: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        scale_shift: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    // FP8 Indexed MoE GEMM kernels (for gather_forward method)
    pub(crate) fn launch_fp8_indexed_moe_gemm_f16(
        input: *const f16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        indices: *const u32,
        output: *mut f16,
        num_tokens: i32,
        topk: i32,
        num_experts: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        input_has_topk_dim: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_fp8_indexed_moe_gemm_bf16(
        input: *const bf16,
        weights: *const F8E4M3,
        weight_scales: *const f32,
        indices: *const u32,
        output: *mut bf16,
        num_tokens: i32,
        topk: i32,
        num_experts: i32,
        n: i32,
        k: i32,
        scale_row_stride: i32,
        block_size_y: i32,
        block_size_x: i32,
        input_has_topk_dim: bool,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
