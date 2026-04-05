//! Model-agnostic weight pointer extraction from loaded Candle models.
//!
//! Uses IsqModel::get_layers() for projection weights and
//! IsqModel::residual_tensors() for norms and embeddings.
//! Works for ANY decoder-only transformer.

#[cfg(feature = "cuda")]
use candle_core::{Storage, Tensor};

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub struct WeightPtr {
    pub ptr: u64,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub input_layernorm: u64,
    pub post_attn_layernorm: u64,
    pub q_proj: WeightPtr,
    pub k_proj: WeightPtr,
    pub v_proj: WeightPtr,
    pub o_proj: WeightPtr,
    pub q_norm: Option<u64>,
    pub k_norm: Option<u64>,
    pub gate_proj: WeightPtr,
    pub up_proj: WeightPtr,
    pub down_proj: WeightPtr,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct ModelWeights {
    pub embed_tokens: u64,
    pub final_norm: u64,
    pub lm_head: WeightPtr,
    pub layers: Vec<LayerWeights>,
    pub config: DecodeConfig,
}

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
    pub max_position_embeddings: usize,
    pub is_gpt_neox: bool,
}

/// Extract raw u64 device pointer from a Candle tensor (any dtype).
#[cfg(feature = "cuda")]
pub fn tensor_device_ptr(tensor: &Tensor) -> candle_core::Result<u64> {
    use candle_core::cuda::cudarc::driver::DevicePtr;
    use candle_core::DType;

    let (storage, layout) = tensor.storage_and_layout();
    let offset_bytes = layout.start_offset() * tensor.dtype().size_in_bytes();

    match &*storage {
        Storage::Cuda(cuda_storage) => {
            // Match on dtype to get correctly-typed CudaSlice (cudarc requires type match)
            let base_ptr: u64 = match tensor.dtype() {
                DType::BF16 => {
                    let s = cuda_storage.as_cuda_slice::<half::bf16>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::F16 => {
                    let s = cuda_storage.as_cuda_slice::<half::f16>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::F32 => {
                    let s = cuda_storage.as_cuda_slice::<f32>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::U32 => {
                    let s = cuda_storage.as_cuda_slice::<u32>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::I64 => {
                    let s = cuda_storage.as_cuda_slice::<i64>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::U8 => {
                    let s = cuda_storage.as_cuda_slice::<u8>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::F8E4M3 => {
                    // F8E4M3 is stored as u8 in cudarc
                    let s = cuda_storage.as_cuda_slice::<u8>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                dt => candle_core::bail!("tensor_device_ptr: unsupported dtype {dt:?}"),
            };
            Ok(base_ptr + offset_bytes as u64)
        }
        _ => candle_core::bail!("tensor_device_ptr requires CUDA tensor"),
    }
}

/// Extract a WeightPtr from a QuantMethod (uses dequantize_w for BF16 weights).
#[cfg(feature = "cuda")]
pub fn quant_method_ptr(qm: &dyn mistralrs_quant::QuantMethod) -> candle_core::Result<WeightPtr> {
    // Try unquant first (zero-copy), fall back to dequantize
    let tensor = if let Some((w, _)) = qm.unquant_weight_bias() {
        w
    } else {
        qm.dequantize_w()?
    };
    let dims = tensor.dims();
    let (rows, cols) = if dims.len() == 2 {
        (dims[0], dims[1])
    } else {
        (dims.iter().product::<usize>(), 1)
    };
    Ok(WeightPtr {
        ptr: tensor_device_ptr(&tensor)?,
        rows,
        cols,
    })
}

/// Build ModelWeights from IsqModel trait methods.
///
/// `get_layers()` returns projection weights in a fixed order per layer:
///   [lm_head, q_0, k_0, v_0, o_0, gate_0, up_0, down_0, q_1, k_1, ...]
///
/// `residual_tensors()` returns named tensors:
///   model.embed_tokens.weight, model.norm.weight,
///   model.layers.N.input_layernorm.weight, model.layers.N.post_attention_layernorm.weight,
///   model.layers.N.self_attn.q_norm.weight (optional), etc.
#[cfg(feature = "cuda")]
pub fn extract_model_weights(
    layers: &[(&std::sync::Arc<dyn mistralrs_quant::QuantMethod>, Option<usize>)],
    residuals: &[(String, candle_core::Tensor)],
    config: DecodeConfig,
) -> candle_core::Result<ModelWeights> {
    let num_layers = config.num_layers;
    let projs_per_layer = 7; // q, k, v, o, gate, up, down

    // First element is lm_head (layer_idx = None)
    let lm_head = quant_method_ptr(&**layers[0].0)?;

    // Build layer weights from projections
    let mut layer_weights = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let base = 1 + i * projs_per_layer;
        layer_weights.push(LayerWeights {
            input_layernorm: 0,     // filled from residuals below
            post_attn_layernorm: 0, // filled from residuals below
            q_proj: quant_method_ptr(&**layers[base].0)?,
            k_proj: quant_method_ptr(&**layers[base + 1].0)?,
            v_proj: quant_method_ptr(&**layers[base + 2].0)?,
            o_proj: quant_method_ptr(&**layers[base + 3].0)?,
            q_norm: None,
            k_norm: None,
            gate_proj: quant_method_ptr(&**layers[base + 4].0)?,
            up_proj: quant_method_ptr(&**layers[base + 5].0)?,
            down_proj: quant_method_ptr(&**layers[base + 6].0)?,
        });
    }

    // Fill in residual tensors (norms, embeddings)
    let mut embed_tokens: u64 = 0;
    let mut final_norm: u64 = 0;

    for (name, tensor) in residuals {
        let ptr = tensor_device_ptr(tensor)?;

        if name.ends_with("embed_tokens.weight") {
            embed_tokens = ptr;
        } else if name == "model.norm.weight" || name.ends_with(".norm.weight") && !name.contains("layers") {
            final_norm = ptr;
        } else if name.contains("input_layernorm.weight") {
            // Extract layer index: model.layers.N.input_layernorm.weight
            if let Some(idx) = extract_layer_idx(name) {
                if idx < num_layers {
                    layer_weights[idx].input_layernorm = ptr;
                }
            }
        } else if name.contains("post_attention_layernorm.weight") {
            if let Some(idx) = extract_layer_idx(name) {
                if idx < num_layers {
                    layer_weights[idx].post_attn_layernorm = ptr;
                }
            }
        } else if name.contains("q_norm.weight") {
            if let Some(idx) = extract_layer_idx(name) {
                if idx < num_layers {
                    layer_weights[idx].q_norm = Some(ptr);
                }
            }
        } else if name.contains("k_norm.weight") {
            if let Some(idx) = extract_layer_idx(name) {
                if idx < num_layers {
                    layer_weights[idx].k_norm = Some(ptr);
                }
            }
        }
    }

    Ok(ModelWeights {
        embed_tokens,
        final_norm,
        lm_head,
        layers: layer_weights,
        config,
    })
}

/// Extract layer index from a tensor name like "model.layers.42.input_layernorm.weight"
#[cfg(feature = "cuda")]
fn extract_layer_idx(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}
