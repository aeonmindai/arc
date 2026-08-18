//! Pre-allocated GPU buffers for the decode loop.
//!
//! **Every buffer here is read and written by CUDA kernels whose signatures
//! are `int32_t*`** (`cuda/decode_loop.cu`, `cuda/sampling.cu`). The dtype of
//! the Candle tensor therefore has to be 4 bytes wide, or the host and the
//! device disagree about where element `i` lives.

#[cfg(feature = "cuda")]
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::sys::CUstream;

#[cfg(feature = "cuda")]
extern "C" {
    fn cudaMemsetAsync(
        dst: *mut std::ffi::c_void,
        value: i32,
        count: usize,
        stream: CUstream,
    ) -> u32;
    fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: u32,
        stream: CUstream,
    ) -> u32;
}

/// Allocate a zeroed I32 tensor. `Tensor::zeros` on I32 is not supported on
/// every Candle backend (see the same note in `sampling_cuda.rs:296`), so go
/// through `from_vec`.
#[cfg(feature = "cuda")]
fn zeros_i32(
    elems: usize,
    shape: impl Into<candle_core::Shape>,
    device: &Device,
) -> candle_core::Result<Tensor> {
    Tensor::from_vec(vec![0i32; elems], shape, device)
}

/// Pre-allocated decode input buffers at fixed GPU addresses.
#[cfg(feature = "cuda")]
pub struct DecodeInputBuffers {
    pub input_ids: Tensor,     // [padded_bs, 1] u32
    pub positions: Tensor,     // [padded_bs] u32
    pub block_tables: Tensor,  // [padded_bs, max_blocks] u32
    pub context_lens: Tensor,  // [padded_bs] u32
    pub slot_mappings: Tensor, // [padded_bs] i64
}

#[cfg(feature = "cuda")]
impl DecodeInputBuffers {
    pub fn new(
        padded_bs: usize,
        max_blocks_per_seq: usize,
        device: &Device,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            input_ids: Tensor::zeros((padded_bs, 1), DType::U32, device)?,
            positions: Tensor::zeros(padded_bs, DType::U32, device)?,
            block_tables: Tensor::zeros((padded_bs, max_blocks_per_seq), DType::U32, device)?,
            context_lens: Tensor::zeros(padded_bs, DType::U32, device)?,
            slot_mappings: Tensor::zeros(padded_bs, DType::I64, device)?,
        })
    }
}

/// GPU-side decode state that persists across WHILE loop iterations.
///
/// 🔴 These were all `I64` while every kernel that touches them declares
/// `int32_t*` (`decode_loop.cu:19-37`, `sampling.cu:21`). The kernel wrote
/// element `b` at byte `4*b`; Candle read element `b` at byte `8*b`. Nothing
/// lined up: `n_generated`, `finished`, `loop_condition` and the whole
/// `output_tokens` matrix were read back as garbage, and `output_tokens` was
/// additionally indexed with the wrong row stride. The path is gated off
/// before it runs, so nothing ever reported it.
#[cfg(feature = "cuda")]
pub struct DecodeState {
    pub sampled_tokens: Tensor, // [padded_bs] i32
    pub n_generated: Tensor,    // [padded_bs] i32
    pub output_tokens: Tensor,  // [padded_bs, max_tokens] i32
    pub finished: Tensor,       // [padded_bs] i32
    pub loop_condition: Tensor, // [1] i32
    pub max_tokens: usize,
}

#[cfg(feature = "cuda")]
impl DecodeState {
    pub fn new(padded_bs: usize, max_tokens: usize, device: &Device) -> candle_core::Result<Self> {
        let _ = DType::I32; // keep the dtype named in this module
        Ok(Self {
            sampled_tokens: zeros_i32(padded_bs, padded_bs, device)?,
            n_generated: zeros_i32(padded_bs, padded_bs, device)?,
            output_tokens: zeros_i32(padded_bs * max_tokens, (padded_bs, max_tokens), device)?,
            finished: zeros_i32(padded_bs, padded_bs, device)?,
            loop_condition: Tensor::from_vec(vec![1i32], 1, device)?,
            max_tokens,
        })
    }

    /// Reset the per-generation state **in place**, on `stream`.
    ///
    /// 🔴 This used to re-`Tensor::zeros` each field, which allocates NEW
    /// device buffers at NEW addresses. The captured CUDA graph has the OLD
    /// addresses baked into its kernel nodes, so resetting between
    /// generations pointed the graph at freed memory — the reset silently
    /// un-did the capture. A pointer-stable buffer is a hard requirement of
    /// anything that gets captured, so zero the existing allocations instead.
    pub fn reset(&mut self, stream: CUstream) -> candle_core::Result<()> {
        let n_gen = crate::weights::tensor_device_ptr(&self.n_generated)?;
        let fin = crate::weights::tensor_device_ptr(&self.finished)?;
        let cond = crate::weights::tensor_device_ptr(&self.loop_condition)?;
        let bs = self.n_generated.elem_count();
        unsafe {
            let s = cudaMemsetAsync(n_gen as *mut _, 0, bs * 4, stream);
            if s != 0 {
                candle_core::bail!("cudaMemsetAsync(n_generated) failed: {s}");
            }
            let s = cudaMemsetAsync(fin as *mut _, 0, bs * 4, stream);
            if s != 0 {
                candle_core::bail!("cudaMemsetAsync(finished) failed: {s}");
            }
            // loop_condition starts at 1 ("keep going"). memset writes a byte
            // pattern, so 0x01010101 would be wrong; write the word directly.
            let one: i32 = 1;
            let s = cudaMemcpyAsync(
                cond as *mut _,
                &one as *const i32 as *const _,
                4,
                1, // cudaMemcpyHostToDevice
                stream,
            );
            if s != 0 {
                candle_core::bail!("cudaMemcpyAsync(loop_condition) failed: {s}");
            }
        }
        Ok(())
    }
}
