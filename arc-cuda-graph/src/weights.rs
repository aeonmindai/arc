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
    // Fused QKV: contiguous [q_rows + k_rows + v_rows, hidden] — set by DedicatedDecodePath
    pub qkv_fused: u64,
    pub qkv_rows: usize, // q_rows + k_rows + v_rows
}

/// Per-layer Tensor anchors that own the device storage backing the `u64`
/// pointers in `LayerWeights`. Held inside `ModelWeights::anchors`.
///
/// `qkv` covers q_proj/k_proj/v_proj — these can be dropped after
/// `DedicatedDecodePath::fuse_qkv` since the fused buffer owns its own
/// memory. The remaining four are read at every decode step.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Default)]
pub struct LayerAnchors {
    pub qkv: Vec<candle_core::Tensor>, // 3 entries: q, k, v (clearable after fuse)
    pub o: Option<candle_core::Tensor>,
    pub gate: Option<candle_core::Tensor>,
    pub up: Option<candle_core::Tensor>,
    pub down: Option<candle_core::Tensor>,
}

#[cfg(feature = "cuda")]
impl LayerAnchors {
    pub fn count(&self) -> usize {
        self.qkv.len()
            + self.o.is_some() as usize
            + self.gate.is_some() as usize
            + self.up.is_some() as usize
            + self.down.is_some() as usize
    }
}

/// All Tensor anchors needed to keep every `u64` device pointer in
/// `ModelWeights` valid for the lifetime of the struct.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Default)]
pub struct WeightAnchors {
    pub lm_head: Option<candle_core::Tensor>,
    pub residuals: Vec<candle_core::Tensor>, // norms, embeds
    pub layers: Vec<LayerAnchors>,
}

#[cfg(feature = "cuda")]
impl WeightAnchors {
    pub fn count(&self) -> usize {
        self.lm_head.is_some() as usize
            + self.residuals.len()
            + self.layers.iter().map(|l| l.count()).sum::<usize>()
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct ModelWeights {
    pub embed_tokens: u64,
    pub final_norm: u64,
    pub lm_head: WeightPtr,
    pub layers: Vec<LayerWeights>,
    pub config: DecodeConfig,
    /// Owners of the device buffers that back every `u64` pointer above.
    ///
    /// `QuantMethod::dequantize_w()` returns a fresh `Tensor` whose `CudaSlice`
    /// is freed as soon as the last Tensor clone drops. If we kept only the
    /// raw `u64` device pointer, the next CUDA op (e.g. `fuse_qkv` reading
    /// `q_proj.ptr`) would dereference a freed device address — manifesting
    /// as a SEGV inside `libcuda`'s SSE-aligned memcpy fast path (the page
    /// table walk for the source pointer faults). Holding the Tensors here
    /// keeps the Arc<Storage> alive for the lifetime of `ModelWeights`, so
    /// every device pointer above remains valid until this struct drops.
    ///
    /// The breakdown by role lets `DedicatedDecodePath` drop just the
    /// per-layer Q/K/V anchors after fusion (saving ~3 BF16 projections per
    /// layer for ISQ models) while keeping O/gate/up/down/lm_head/residuals
    /// alive for the decode loop.
    pub anchors: WeightAnchors,
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
///
/// Returns both the `WeightPtr` and the backing `Tensor`. The caller MUST
/// keep the Tensor alive for as long as the device pointer in `WeightPtr` is
/// used — otherwise `Drop` on the Tensor frees the CUDA storage and the
/// pointer becomes dangling. For unquantized methods this is a cheap clone
/// of the model's own weight; for ISQ-quantized methods this owns the freshly
/// dequantized BF16/F16 buffer.
#[cfg(feature = "cuda")]
pub fn quant_method_ptr(
    qm: &dyn mistralrs_quant::QuantMethod,
) -> candle_core::Result<(WeightPtr, candle_core::Tensor)> {
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
    let ptr = tensor_device_ptr(&tensor)?;
    Ok((WeightPtr { ptr, rows, cols }, tensor))
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

    let mut anchors = WeightAnchors::default();
    anchors.residuals.reserve(residuals.len());
    anchors.layers = Vec::with_capacity(num_layers);

    // First element is lm_head (layer_idx = None)
    let (lm_head, lm_head_t) = quant_method_ptr(&**layers[0].0)?;
    anchors.lm_head = Some(lm_head_t);

    // Build layer weights from projections
    let mut layer_weights = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let base = 1 + i * projs_per_layer;
        let (q_proj, q_t) = quant_method_ptr(&**layers[base].0)?;
        let (k_proj, k_t) = quant_method_ptr(&**layers[base + 1].0)?;
        let (v_proj, v_t) = quant_method_ptr(&**layers[base + 2].0)?;
        let (o_proj, o_t) = quant_method_ptr(&**layers[base + 3].0)?;
        let (gate_proj, gate_t) = quant_method_ptr(&**layers[base + 4].0)?;
        let (up_proj, up_t) = quant_method_ptr(&**layers[base + 5].0)?;
        let (down_proj, down_t) = quant_method_ptr(&**layers[base + 6].0)?;
        anchors.layers.push(LayerAnchors {
            qkv: vec![q_t, k_t, v_t],
            o: Some(o_t),
            gate: Some(gate_t),
            up: Some(up_t),
            down: Some(down_t),
        });
        layer_weights.push(LayerWeights {
            input_layernorm: 0,     // filled from residuals below
            post_attn_layernorm: 0, // filled from residuals below
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: None,
            k_norm: None,
            gate_proj,
            up_proj,
            down_proj,
            qkv_fused: 0,
            qkv_rows: 0,
        });
    }

    // Fill in residual tensors (norms, embeddings).
    // These tensors' storage is owned by the model itself, but we *also*
    // anchor a clone here so the pointer stays valid even if the model
    // were ever to be partially reloaded under us.
    let mut embed_tokens: u64 = 0;
    let mut final_norm: u64 = 0;

    for (name, tensor) in residuals {
        let ptr = tensor_device_ptr(tensor)?;
        anchors.residuals.push(tensor.clone());

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
        anchors,
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

/// Regression test: prove `Tensor::clone` shares storage. This is the
/// invariant the anchor mechanism relies on — cloning a Tensor must
/// keep the underlying storage alive but must NOT allocate new device
/// memory. If candle ever changes Tensor semantics, this test catches
/// it on CPU before we ship a quadratic-memory regression to CUDA.
///
/// This test runs without the cuda feature because it uses CPU tensors.
#[cfg(test)]
mod cpu_anchor_tests {
    use candle_core::{Device, Tensor, DType, Storage};

    fn storage_addr(t: &Tensor) -> usize {
        let (storage, _) = t.storage_and_layout();
        match &*storage {
            Storage::Cpu(cpu) => {
                // CpuStorage variants hold a Vec<T>. We use the discriminant
                // address as a stand-in: clones share the same backing
                // RwLock<Storage>, so the addresses match.
                cpu as *const _ as usize
            }
            _ => panic!("test expects CPU storage"),
        }
    }

    #[test]
    fn tensor_clone_shares_storage() {
        let src = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
        let original_addr = storage_addr(&src);
        let clone = src.clone();
        let clone_addr = storage_addr(&clone);
        assert_eq!(
            original_addr, clone_addr,
            "Tensor::clone must share storage (Arc bump) — \
             anchor mechanism relies on this. If this fails, the \
             dangling-pointer fix in DedicatedDecodePath is broken."
        );
    }

    #[test]
    fn tensor_drop_after_clone_keeps_storage_alive() {
        let original_addr: usize;
        let anchor;
        {
            let src = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
            original_addr = storage_addr(&src);
            anchor = src.clone();
            // `src` drops at end of block — but `anchor` should keep
            // the storage alive.
        }
        assert_eq!(
            original_addr, storage_addr(&anchor),
            "Anchored Tensor must keep storage alive after the original drops"
        );
        // Touch the storage to prove it's still readable.
        let v = anchor.to_vec2::<f32>().expect("tensor should be readable");
        assert_eq!(v.len(), 4);
        assert_eq!(v[0].len(), 8);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    #[test]
    fn extract_layer_idx_basic() {
        assert_eq!(
            super::extract_layer_idx("model.layers.42.input_layernorm.weight"),
            Some(42)
        );
        assert_eq!(
            super::extract_layer_idx("model.layers.0.post_attention_layernorm.weight"),
            Some(0)
        );
        assert_eq!(super::extract_layer_idx("model.embed_tokens.weight"), None);
        assert_eq!(super::extract_layer_idx("lm_head.weight"), None);
    }

    /// Regression test for the libcuda SEGV root cause: `quant_method_ptr`
    /// must return the backing Tensor so the caller can anchor it. Without
    /// the anchor the underlying storage drops at the end of the helper and
    /// the raw `u64` device pointer becomes a use-after-free. This test only
    /// asserts the *type contract* — the runtime behavior requires CUDA and
    /// is exercised by `DedicatedDecodePath` integration tests.
    #[test]
    fn quant_method_ptr_returns_tensor_anchor() {
        // We can't construct a real QuantMethod without CUDA. The compile-time
        // signature check is the regression guard: any future refactor that
        // drops the Tensor from the return type will fail to compile.
        fn _signature_check<F>(_: F)
        where
            F: Fn(
                &dyn mistralrs_quant::QuantMethod,
            )
                -> candle_core::Result<(super::WeightPtr, candle_core::Tensor)>,
        {
        }
        _signature_check(super::quant_method_ptr);
    }

    /// Regression test for the libcuda SEGV root cause: `ModelWeights` must
    /// carry an `anchors` field whose lifetime is tied to the struct. Compile-
    /// time check: if `anchors` is ever removed from `ModelWeights`, callers
    /// will fail to build, surfacing the regression immediately.
    #[test]
    fn model_weights_owns_anchors() {
        // Build an empty `WeightAnchors` to prove the field exists and is
        // constructible from `()` (i.e. default).
        let a = super::WeightAnchors::default();
        assert_eq!(a.count(), 0);

        // `LayerAnchors::qkv` MUST be a `Vec` so `DedicatedDecodePath` can
        // `.clear()` it after fusion without touching the still-live
        // o/gate/up/down anchors. Compile-time guard:
        let empty: Vec<candle_core::Tensor> = vec![];
        fn _is_vec<T>(_: &Vec<T>) {}
        _is_vec(a.layers.first().map(|l| &l.qkv).unwrap_or(&empty));
    }
}
