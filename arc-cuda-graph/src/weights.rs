//! Weight pointer extraction from loaded Candle models.
//!
//! Extracts raw GPU device pointers from model weight tensors.
//! These pointers are stable (weights don't move after loading)
//! and can be used by the dedicated decode path and CUDA graphs.

#[cfg(feature = "cuda")]
use candle_core::{Storage, Tensor};

/// Raw GPU pointer for a single weight tensor.
/// The pointer is stable as long as the model is alive.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub struct WeightPtr {
    pub ptr: u64,
    pub rows: usize,
    pub cols: usize,
}

/// Weight pointers for one transformer layer.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub input_layernorm: u64,
    pub post_attn_layernorm: u64,
    pub q_proj: WeightPtr,
    pub k_proj: WeightPtr,
    pub v_proj: WeightPtr,
    pub o_proj: WeightPtr,
    pub q_norm: Option<u64>,  // Qwen3 has this, most models don't
    pub k_norm: Option<u64>,
    pub gate_proj: WeightPtr,
    pub up_proj: WeightPtr,
    pub down_proj: WeightPtr,
}

/// All weight pointers needed for the decode path.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct ModelWeights {
    pub embed_tokens: u64,   // [vocab_size, hidden_size]
    pub final_norm: u64,     // [hidden_size]
    pub lm_head: WeightPtr,  // [vocab_size, hidden_size] or tied to embed
    pub layers: Vec<LayerWeights>,
    pub config: DecodeConfig,
}

/// Model configuration for the decode path.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct DecodeConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub has_qk_norm: bool,
}

/// Extract a raw u64 device pointer from a Candle tensor.
/// The tensor must be on a CUDA device.
#[cfg(feature = "cuda")]
pub fn tensor_device_ptr(tensor: &Tensor) -> candle_core::Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cuda(cuda_storage) => {
            let slice = cuda_storage.as_cuda_slice::<u8>()?;
            let (base_ptr, _guard) = slice.device_ptr(slice.stream());
            let offset = layout.start_offset() * tensor.dtype().size_in_bytes();
            Ok(base_ptr + offset as u64)
        }
        _ => candle_core::bail!("tensor_device_ptr requires CUDA tensor"),
    }
}
