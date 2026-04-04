//! Pre-allocated GPU buffers for the decode loop.

#[cfg(feature = "cuda")]
use candle_core::{DType, Device, Tensor};

/// Pre-allocated decode input buffers at fixed GPU addresses.
#[cfg(feature = "cuda")]
pub struct DecodeInputBuffers {
    pub input_ids: Tensor,      // [padded_bs, 1] u32
    pub positions: Tensor,      // [padded_bs] u32
    pub block_tables: Tensor,   // [padded_bs, max_blocks] u32
    pub context_lens: Tensor,   // [padded_bs] u32
    pub slot_mappings: Tensor,  // [padded_bs] i64
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
#[cfg(feature = "cuda")]
pub struct DecodeState {
    pub sampled_tokens: Tensor,   // [padded_bs] i64
    pub n_generated: Tensor,      // [padded_bs] i64
    pub output_tokens: Tensor,    // [padded_bs, max_tokens] i64
    pub finished: Tensor,         // [padded_bs] i64
    pub loop_condition: Tensor,   // [1] i64
    pub max_tokens: usize,
}

#[cfg(feature = "cuda")]
impl DecodeState {
    pub fn new(
        padded_bs: usize,
        max_tokens: usize,
        device: &Device,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            sampled_tokens: Tensor::zeros(padded_bs, DType::I64, device)?,
            n_generated: Tensor::zeros(padded_bs, DType::I64, device)?,
            output_tokens: Tensor::zeros((padded_bs, max_tokens), DType::I64, device)?,
            finished: Tensor::zeros(padded_bs, DType::I64, device)?,
            loop_condition: Tensor::ones(1, DType::I64, device)?,
            max_tokens,
        })
    }

    pub fn reset(&mut self, device: &Device, padded_bs: usize) -> candle_core::Result<()> {
        self.n_generated = Tensor::zeros(padded_bs, DType::I64, device)?;
        self.finished = Tensor::zeros(padded_bs, DType::I64, device)?;
        self.loop_condition = Tensor::ones(1, DType::I64, device)?;
        Ok(())
    }
}
