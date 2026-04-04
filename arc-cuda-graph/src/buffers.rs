//! Pre-allocated GPU buffers for the decode loop.
//!
//! All decode inputs and state live at fixed GPU addresses.
//! The autonomous WHILE loop reads and writes these buffers
//! without any CPU involvement.

#[cfg(feature = "cuda")]
use candle_core::{DType, Device, Tensor};

/// Pre-allocated decode input buffers at fixed GPU addresses.
/// The forward pass reads from these. The step-update kernel writes to them.
#[cfg(feature = "cuda")]
pub struct DecodeInputBuffers {
    pub input_ids: Tensor,      // [padded_bs] u32
    pub positions: Tensor,      // [padded_bs] i32 (as i64 for Candle compat, kernel reads as i32)
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

    /// Copy actual input values into the fixed buffers.
    /// Call this before launching the graph.
    pub fn update(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        block_tables: &Tensor,
        context_lens: &Tensor,
        slot_mappings: &Tensor,
    ) -> candle_core::Result<()> {
        self.input_ids.copy_(input_ids)?;
        let pos_u32: Vec<u32> = positions.iter().map(|&x| x as u32).collect();
        let pos_tensor = Tensor::from_vec(pos_u32, self.positions.shape(), self.positions.device())?;
        self.positions.copy_(&pos_tensor)?;
        self.block_tables.copy_(block_tables)?;
        self.context_lens.copy_(context_lens)?;
        self.slot_mappings.copy_(slot_mappings)?;
        Ok(())
    }
}

/// GPU-side decode state that persists across WHILE loop iterations.
#[cfg(feature = "cuda")]
pub struct DecodeState {
    pub sampled_tokens: Tensor,   // [padded_bs] i32 — written by sampling kernel
    pub n_generated: Tensor,      // [padded_bs] i32
    pub output_tokens: Tensor,    // [padded_bs, max_tokens] i32
    pub finished: Tensor,         // [padded_bs] i32 (0 = active, 1 = done)
    pub loop_condition: Tensor,   // [1] i32 — read by WHILE conditional node (1 = loop, 0 = stop)
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

    /// Reset state for a new generation.
    pub fn reset(&mut self, device: &Device, padded_bs: usize) -> candle_core::Result<()> {
        self.n_generated = Tensor::zeros(padded_bs, DType::I64, device)?;
        self.finished = Tensor::zeros(padded_bs, DType::I64, device)?;
        self.loop_condition = Tensor::ones(1, DType::I64, device)?;
        Ok(())
    }
}
