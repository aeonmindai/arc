use std::{
    borrow::Cow,
    io::Cursor,
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{quantized::GgmlDType, DType, Device, DeviceLocation, Result, Shape, Tensor, D};
use candle_nn::Linear;

use crate::{
    calibration::{CalibAccumulator, CalibLayerData, CalibOptions},
    cublaslt::{maybe_init_cublas_lt_wrapper, CUBLASLT_CONTROLLER},
    generate_isq, generate_isq_imatrix,
    hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer, ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    utils::{deserialize_tensor, serialize_tensor, version_is_compatible, UQFF_VERSION},
    AfqBits, AfqGroupSize, AfqLayer, FP8Linear, GgufMatMul, ImatrixLayerStats, IsqType, MatMul,
    QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, QuantizedSerdeType,
};

/// Collapse every leading dimension into the GEMM's `m`, and return the dims
/// needed to restore them.
///
/// `[B, T, hidden] -> ([B*T, hidden], [B, T])`. Kept as a named function so the
/// shape arithmetic is testable on any host — the caller that uses it is
/// device-gated (see [`UnquantLinear::forward`]), and shape logic that only
/// executes on hardware nobody has is shape logic nobody checks.
///
/// `reshape` on a non-contiguous input costs one strided copy rather than
/// failing, which is what makes this safe to apply to an arbitrary activation.
fn flatten_batch_dims(a: &Tensor) -> Result<(Tensor, Vec<usize>)> {
    let batch_dims = a.dims()[..a.rank() - 1].to_vec();
    let rows: usize = batch_dims.iter().product();
    let features = a.dim(D::Minus1)?;
    Ok((a.reshape((rows, features))?, batch_dims))
}

#[derive(Debug)]
pub struct UnquantLinear {
    w: Tensor,
    b: Option<Tensor>,
    stats: Option<ImatrixLayerStats>,
    /// Armed by [`QuantMethod::begin_calibration`] for a forward-only
    /// calibration sweep. Independent of `stats` (the imatrix path).
    calib: Option<CalibAccumulator>,
}

impl QuantMethod for UnquantLinear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. }
            | QuantMethodConfig::NVFP4 { .. }
            | QuantMethodConfig::Qtip { .. }
            | QuantMethodConfig::Qtip2b { .. }
            | QuantMethodConfig::TuckerFactored { .. } => unreachable!(),
            QuantMethodConfig::Unquantized(l) => Ok(Self {
                w: l.weight().clone(),
                b: l.bias().cloned(),
                stats: None,
                calib: None,
            }),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        Ok(self.w.clone())
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        // Batch matrix multiplication
        maybe_init_cublas_lt_wrapper(a.device().clone());

        // Try custom GEMV for single-token decode (batch_size=1).
        //
        // Deliberately tested on the *original* shape, before the flatten
        // below: `should_use_gemv` reads `product(dims[..rank-1])` against
        // `MAX_GEMV_BATCH_SIZE`, so flattening first would not change which
        // side of that threshold a tensor lands on, but keeping the order
        // makes that a fact rather than a coincidence.
        #[cfg(feature = "cuda")]
        if crate::gemv::should_use_gemv(a, &self.w) {
            return crate::gemv::gemv(a, &self.w, self.b.as_ref());
        }

        // 🔴 A rank-3 activation must be flattened to 2-D, or the weight is
        // re-read from HBM once per batch row.
        //
        // Every activation in this engine is rank-3 `[B, T, hidden]`. Left
        // that way, the `match` below took `broadcast_left(B)` — a stride-0
        // view of the weight — and both dispatch arms then treated `B` as a
        // GEMM batch dimension with `stride_b = 0`. At decode `T = 1`, so that
        // is **B independent `m = 1` GEMVs over the same weight**: a batched
        // GEMV wearing a GEMM's name.
        //
        // The cost is not subtle. V4's `lm_head` is `[256, 1, 4096] x
        // [4096, 129280]` BF16, whose bound is
        // `max(271 GFLOP / 989 TFLOP/s, 1.06 GB / 4.8 TB/s) = 0.27 ms`.
        // Measured on an H200 it is **67 ms/step, 8.41% of a 794 ms B=256
        // decode step** — 244x over, and `244 ~= B = 256` is the whole
        // explanation. Back-solving, 256 x 1.06 GB = 271 GB of weight traffic
        // per step at 4.05 TB/s: the kernel is running at 84% of peak
        // bandwidth while doing 256x the necessary reads. The 1.06 GB weight
        // does not fit in L2, so nothing is reused across rows.
        //
        // Flattened, the same work is one `[256, 4096] x [4096, 129280]` GEMM:
        // the weight is streamed once and the tensor cores see `m = 256`
        // instead of `m = 1`.
        //
        // The identical defect was fixed one level up in `BlockwiseFP8Linear`
        // (445b92063) by flattening before delegating here. That fix only
        // covers activations routed through blockwise FP8 — `lm_head` is not,
        // because DeepSeek ships `lm_head.weight` in BF16 with no
        // `weight_scale_inv` and `blockwise_fp8::linear_no_bias` short-circuits
        // straight to `UnquantLinear`. Fixing it here is the general case and
        // subsumes the caller-side workaround.
        //
        // UNVERIFIED ON HARDWARE: derived from shapes and H200 peaks, not
        // measured. The arithmetic above is the claim; the speedup is not.
        //
        // ── Why CPU is excluded, and it is not to dodge a test ──────────────
        //
        // The defect is HBM traffic: a stride-0 batched GEMM streaming a
        // 1.06 GB weight once per batch row. There is no HBM on the CPU path,
        // and `MatMul::matmul` casts to F16 there anyway — it is a reference
        // oracle, not a performance path.
        //
        // Changing its arithmetic is not free. Batched-vs-solo was **bit
        // identical** before this (`max_diff` measured at exactly 0), because
        // `broadcast_left` made the batched case B separate `m = 1` GEMMs —
        // the same shape, and therefore the same accumulation order, as the
        // solo run. Flattening makes the batched case `m = B`, and x86's F16
        // microkernel blocks `m = 2` differently from `m = 1`. That is pure
        // floating-point associativity, but it re-baselines every
        // batched-vs-solo tolerance in the V4 suite at once: CI measured
        // 1.57e-2 relative divergence against a 1e-2 tolerance that had never
        // been exercised, in a test whose job is to catch cross-sequence state
        // leaks — which are O(1), ~100x that tolerance.
        //
        // So the oracle stays byte-stable and the fix lands where the cost is.
        // `two_d_and_rank3_agree_within_the_cpu_f16_budget` pins that the two
        // formulations are the same mathematics, and measures the gap that
        // motivates this line.
        if a.rank() > 2 && !a.device().is_cpu() {
            let (flat, batch_dims) = flatten_batch_dims(a)?;
            let out = self.forward(&flat)?;
            let mut out_dims = batch_dims;
            out_dims.push(out.dim(D::Minus1)?);
            return out.reshape(out_dims);
        }

        let w = match *a.dims() {
            [b1, b2, _, _] => self.w.broadcast_left((b1, b2))?,
            [bsize, _, _] => self.w.broadcast_left(bsize)?,
            _ => self.w.clone(),
        };

        if let Some(stats) = &self.stats {
            stats.process(a)?;
        }
        if let Some(calib) = &self.calib {
            calib.process(a)?;
        }

        if let Some(b) = self.b.as_ref() {
            let mut tgt_shape = a.dims().to_vec();
            tgt_shape[a.dims().len() - 1] = w.dim(D::Minus2)?;
            let b = b.broadcast_as(Shape::from_dims(&tgt_shape))?;

            match a.device().location() {
                DeviceLocation::Cuda { .. } => {
                    // Try to use cublaslt, otherwise fallback to gemm.
                    //
                    // The `rank() >= 3` guard mirrors the no-bias arm below and
                    // is now load-bearing: `batch_matmul` calls `dims3()` on
                    // both operands (`cublaslt/api.rs`), so a 2-D activation
                    // errors out rather than falling back. Before the flatten
                    // above that was unreachable-by-accident — every caller
                    // passed rank 3 — which is exactly the kind of latent arm
                    // a shape change turns into a crash.
                    if let (Device::Cuda(_), Some(cublaslt)) =
                        (a.device(), CUBLASLT_CONTROLLER.get_for_device(a.device()))
                    {
                        if a.rank() >= 3 && w.rank() >= 3 {
                            cublaslt
                                .batch_matmul(
                                    a,
                                    &w,
                                    Some(&b.t()?.contiguous()?),
                                    None,
                                    Some(1.0),
                                    None,
                                    None,
                                )?
                                .t()
                        } else {
                            let matmul_result = MatMul.matmul(a, &w.t()?)?;
                            matmul_result.broadcast_add(&b)
                        }
                    } else {
                        let matmul_result = a.matmul(&w.t()?)?;
                        matmul_result.broadcast_add(&b)
                    }
                }
                DeviceLocation::Metal { .. } => {
                    let matmul_result = a.matmul(&w.t()?)?;
                    matmul_result.broadcast_add(&b)
                }
                DeviceLocation::Cpu => {
                    #[cfg(feature = "accelerate")]
                    {
                        let original_dtype = a.dtype();
                        let a_f32 = a.to_dtype(DType::F32)?;
                        let w_f32 = w.t()?.to_dtype(DType::F32)?;
                        let b_f32 = b.to_dtype(DType::F32)?;
                        let matmul_result = a_f32.matmul(&w_f32)?;
                        matmul_result
                            .broadcast_add(&b_f32)?
                            .to_dtype(original_dtype)
                    }
                    #[cfg(not(feature = "accelerate"))]
                    {
                        let matmul_result = a.matmul(&w.t()?)?;
                        matmul_result.broadcast_add(&b)
                    }
                }
            }
        } else if let (Device::Cuda(_), Some(cublaslt)) =
            (a.device(), CUBLASLT_CONTROLLER.get_for_device(a.device()))
        {
            // cuBLAS batch_matmul requires 3D tensors, fall back to regular matmul for 2D
            if a.rank() >= 3 && w.rank() >= 3 {
                cublaslt
                    .batch_matmul(a, &w, None, None, None, None, None)?
                    .t()
            } else {
                MatMul.matmul(a, &w.t()?)
            }
        } else {
            MatMul.matmul(a, &w.t()?)
        }
    }

    fn gather_forward(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        // Weights are [num_experts, out_features, in_features]
        // For Metal path:
        //   - a: (b_size, seq_len, 1, 1, hidden_dim) - 5D
        //   - indices: (b_size, seq_len, num_experts_per_tok) - 3D
        // For CUDA path:
        //   - a: (num_tokens, 1, hidden_dim) - 3D
        //   - indices: (num_tokens, num_experts_per_tok) - 2D

        let w = &self.w;
        let (_num_experts, out_features, _in_features) = w.dims3()?;

        // Per-expert calibration needs the routing indices, which only this
        // entry point sees — the dense `forward` hook cannot recover them.
        if let Some(calib) = &self.calib {
            calib.process_gather(a, indices)?;
        }

        match a.dims() {
            // Metal path: 5D input (b_size, seq_len, 1, 1, hidden_dim)
            &[b_size, seq_len, 1, 1, hidden_dim] => {
                let (_b, _s, num_experts_per_tok) = indices.dims3()?;
                // Flatten indices to select experts
                let flat_indices = indices.reshape((b_size * seq_len * num_experts_per_tok,))?;

                // Select expert weights: [b*s*k, out_features, in_features]
                let selected_w = w.index_select(&flat_indices, 0)?;

                // Reshape input: [b*s, hidden_dim]
                let a_flat = a.reshape((b_size * seq_len, hidden_dim))?;

                // For each token, we need to compute with each selected expert
                // Broadcast a to match: [b*s, 1, hidden_dim] -> [b*s, k, hidden_dim]
                let a_expanded = a_flat
                    .unsqueeze(1)?
                    .broadcast_as((b_size * seq_len, num_experts_per_tok, hidden_dim))?
                    .reshape((b_size * seq_len * num_experts_per_tok, hidden_dim))?;

                // Matmul: [b*s*k, hidden_dim] @ [b*s*k, hidden_dim, out_features] -> [b*s*k, out_features]
                let result = a_expanded
                    .unsqueeze(1)?
                    .matmul(&selected_w.transpose(1, 2)?)?
                    .squeeze(1)?;

                // Reshape back to [b, s, k, out_features]
                result.reshape((b_size, seq_len, num_experts_per_tok, out_features))
            }
            // CUDA path: 3D input (num_tokens, 1, hidden_dim)
            &[num_tokens, 1, hidden_dim] => {
                let (_, num_experts_per_tok) = indices.dims2()?;

                // Flatten indices
                let flat_indices = indices.reshape((num_tokens * num_experts_per_tok,))?;

                // Select expert weights: [n*k, out_features, in_features]
                let selected_w = w.index_select(&flat_indices, 0)?;

                // Broadcast input: [n, 1, hidden] -> [n, k, hidden] -> [n*k, hidden]
                let a_expanded = a
                    .broadcast_as((num_tokens, num_experts_per_tok, hidden_dim))?
                    .reshape((num_tokens * num_experts_per_tok, hidden_dim))?;

                // Matmul: [n*k, hidden] @ [n*k, hidden, out] -> [n*k, out]
                let result = a_expanded
                    .unsqueeze(1)?
                    .matmul(&selected_w.transpose(1, 2)?)?
                    .squeeze(1)?;

                // Reshape to [n, k, out]
                result.reshape((num_tokens, num_experts_per_tok, out_features))
            }
            dims => {
                candle_core::bail!(
                    "UnquantLinear::gather_forward: unsupported input shape {:?}",
                    dims
                );
            }
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self {
            w: (&self.w + delta)?,
            b: self.b.clone(),
            stats: self.stats.clone(),
            calib: self.calib.clone(),
        }))
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (self.w.dtype(), self.w.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        match dtype {
            /*Some(IsqType::HQQ1 | IsqType::HQQ2 | IsqType::HQQ3 | */
            Some(IsqType::HQQ4 | IsqType::HQQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    // TODO just warn?
                    candle_core::bail!("HQQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::HQQ8 => HqqBits::Eight,
                    IsqType::HQQ4 => HqqBits::Four,
                    // IsqType::HQQ3 => HqqBits::Three,
                    // IsqType::HQQ2 => HqqBits::Two,
                    // IsqType::HQQ1 => HqqBits::One,
                    _ => unreachable!(),
                };
                let cfg = HqqConfig {
                    bits,
                    group_size: ISQ_HQQ_GROUP_SIZE.try_into()?,
                    axis: HqqAxis::Zero,
                    optimization_steps: ISQ_HQQ_DEFAULT_OPT_STEPS,
                    round_zeros: false,
                    channel_wise: true,
                };
                let res = HqqLayer::quantize(&self.w.to_device(&device)?, &device, cfg)?;
                if let Some(bias) = &self.b {
                    let bias = bias
                        .to_device(&device)?
                        .to_dtype(res.dtype_and_device().0)?;
                    Ok(Arc::new(res.with_bias(bias)))
                } else {
                    Ok(Arc::new(res))
                }
            }
            Some(IsqType::AFQ2 | IsqType::AFQ3 | IsqType::AFQ4 | IsqType::AFQ6 | IsqType::AFQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    // TODO just warn?
                    candle_core::bail!("AFQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::AFQ8 => AfqBits::Eight,
                    IsqType::AFQ6 => AfqBits::Six,
                    IsqType::AFQ4 => AfqBits::Four,
                    IsqType::AFQ3 => AfqBits::Three,
                    IsqType::AFQ2 => AfqBits::Two,
                    _ => unreachable!(),
                };

                Ok(Arc::new(AfqLayer::new(QuantMethodConfig::Afq {
                    weight: self.w.to_device(&device)?,
                    bias: self.b.as_ref().map(|b| b.to_device(&device).unwrap()),
                    bits,
                    group_size: AfqGroupSize::default(),
                })?))
            }
            Some(
                IsqType::Q2K
                | IsqType::Q3K
                | IsqType::Q4K
                | IsqType::Q4_0
                | IsqType::Q4_1
                | IsqType::Q5K
                | IsqType::Q5_0
                | IsqType::Q5_1
                | IsqType::Q6K
                | IsqType::Q8K
                | IsqType::Q8_0
                | IsqType::Q8_1,
            ) => {
                let dtype: GgmlDType = dtype.unwrap().try_into()?;
                let res = if let Some(imatrix_weight) = imatrix_weight {
                    generate_isq_imatrix!(self.w, imatrix_weight, device, dtype, n_quantized, guard)
                } else {
                    generate_isq!(self.w, device, dtype, n_quantized, guard)
                };
                Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: res,
                    b: self
                        .b
                        .as_ref()
                        .map(|b| b.to_dtype(DType::F32).unwrap().to_device(&device).unwrap()),
                })?))
            }
            Some(IsqType::F8E4M3) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    // TODO just warn?
                    candle_core::bail!("F8E4M3 does not support imatrix.");
                }

                let w = self.w.to_device(&device)?;
                let b = if let Some(b) = &self.b {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(FP8Linear::new(QuantMethodConfig::FP8 {
                    lin: Linear::new(w, b),
                    dtype: DType::F8E4M3,
                })?))
            }
            Some(IsqType::MXFP4) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("MXFP4 does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let w = self.w.to_device(&device)?;
                let b = self.b.as_ref().map(|b| b.to_device(&device)).transpose()?;
                crate::MXFP4Layer::quantize(&w, b, &device)
            }
            Some(IsqType::NVFP4) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("NVFP4 does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let w = self.w.to_device(&device)?;
                let b = self.b.as_ref().map(|b| b.to_device(&device)).transpose()?;
                crate::NVFP4Layer::quantize(&w, b, &device)
            }
            Some(IsqType::QtipBitshift2) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                // wave13-AD axis B: the imatrix IS the diagonal activation
                // Hessian this rung wants — `ImatrixLayerStats` accumulates
                // `Σ_n x_{n,j}²` per input column, i.e. `N · diag(XᵀX)`, which
                // is `diag(H)` up to the global scale the trellis metric is
                // invariant to. It is only consumed when `ARC_QTIP_HESSIAN=1`;
                // otherwise we keep the historical "unsupported" contract
                // rather than silently ignoring calibration data.
                let hessian_diag = match (&imatrix_weight, crate::QtipBakeConfig::get()?.hessian) {
                    (Some(h), true) => Some(h.as_slice()),
                    (Some(_), false) => candle_core::bail!(
                        "QTIP does not support imatrix. Set ARC_QTIP_HESSIAN=1 to bake with the \
                         diagonal activation-Hessian objective instead."
                    ),
                    (None, _) => None,
                };
                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bias = self.b.as_ref().map(|b| b.to_device(&device)).transpose()?;
                // 3-D weights are MoE expert stacks (e.g. [256, 2048, 4096]).
                // Both ranks bake with the trellis search + Hadamard rotation;
                // greedy is banned (DOCTRINE D4) and there is no env var that
                // brings it back. `default_expert_mode()` is the shared
                // decision point with the bake-quality regression tests
                // (qtip/bake_quality_tests.rs) and with the ISQ thread policy
                // (`IsqType::isq_cpu_thread_policy`).
                let mode = crate::QtipMode::default_expert_mode();
                // For 3D MoE expert stacks ([E, N, K], e.g. [256, 2048, 4096]),
                // keep the full weight on its current device (CPU during ISQ
                // load) and let `quantize_with_options_3d` stream one expert at a
                // time onto `device`. Pre-moving the whole stack to GPU allocates
                // ~4GB of transient BF16 and OOMs the final layers on a single
                // 80GB H100 (the per-expert GPU transient is only ~33MB). 2D
                // weights are small, so move them up front as before. (RUN-161)
                let w_for_quant = if self.w.dims().len() == 3 {
                    self.w.clone()
                } else {
                    self.w.to_device(&device)?
                };
                // One shared policy table, not a local `matches!` — a fourth
                // independent copy of the rotation decision is exactly how it
                // drifts (wave13-AG / PR #29 both flagged this line).
                let use_rotation = crate::QtipRotation::for_mode(mode).enabled();
                crate::QtipLayer::quantize_with_calibration(
                    &w_for_quant,
                    bias,
                    &device,
                    mode,
                    use_rotation,
                    hessian_diag,
                )
            }
            Some(IsqType::Qtip2b) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("QTIP2B does not support imatrix.");
                }
                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bias = self.b.as_ref().map(|b| b.to_device(&device)).transpose()?;
                // Same mode/device policy as the LUT rung above. This arm used
                // to send 3-D expert stacks to `QtipMode::Greedy` unless
                // ARC_QTIP_EXPERT_VITERBI was set — the sibling rung had
                // already been fixed (PR #20) and this one was missed, so the
                // rung the trellis grouped-GEMM and both GEMV autotune grids
                // are built around was still baking at greedy quality
                // (matmul cos 0.843 without rotation vs 0.957 with, wave13-AD).
                // Greedy is now banned outright (DOCTRINE D4): one shared
                // decision point, no env opt-in, no rank-dependent branch.
                let mode = crate::QtipMode::default_expert_mode();
                let w_for_quant = if self.w.dims().len() == 3 {
                    self.w.clone()
                } else {
                    self.w.to_device(&device)?
                };
                crate::Qtip2bLayer::quantize_with_mode(&w_for_quant, bias, &device, mode)
            }
            Some(IsqType::F8Q8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("F8Q8 does not support imatrix.");
                }

                let w = self.w.to_device(&device)?;
                let b = if let Some(b) = &self.b {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(crate::F8Q8Linear::from_weight(&w, b)?))
            }
            None => {
                let _acquired_quantize_guard = guard.acquire(&device);
                // Ignore imatrix altogether

                let w = self.w.to_device(&device)?;
                let b = if let Some(b) = &self.b {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(UnquantLinear::new(
                    QuantMethodConfig::Unquantized(Linear::new(w, b)),
                )?))
            }
        }
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        Some((self.w.clone(), self.b.clone()))
    }

    fn begin_track_stats(&mut self) -> Result<()> {
        self.stats = Some(ImatrixLayerStats::new(&self.w, self.w.device())?);
        Ok(())
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        if let Some(stats) = &self.stats {
            let imatrix = stats.compute_imatrix()?;
            stats.clear()?;
            Ok(imatrix)
        } else {
            candle_core::bail!("`{}` does not support tracking stats.", self.name())
        }
    }

    fn begin_calibration(&mut self, opts: &CalibOptions) -> Result<()> {
        // 2-D `[out, in]` linears and 3-D `[experts, out, in]` MoE stacks both
        // carry the input width in the trailing dimension.
        let dims = self.w.dims();
        let in_features = *dims
            .last()
            .ok_or_else(|| candle_core::Error::Msg("UnquantLinear: scalar weight".to_string()))?;
        let num_experts = if dims.len() == 3 { Some(dims[0]) } else { None };
        self.calib = Some(CalibAccumulator::new(in_features, num_experts, *opts)?);
        Ok(())
    }

    fn end_calibration(&self) -> Result<CalibLayerData> {
        match &self.calib {
            Some(calib) => calib.finish(),
            None => candle_core::bail!("`{}` was not armed for calibration.", self.name()),
        }
    }
}

// Serialization structure:
//
// -----------------------
// UQFF version, u32, little endian
// -----------------------
// ISQ type (1 for unquantized), u8, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Weight tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------
// [OPTIONAL] Bias tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------

impl QuantizedSerde for UnquantLinear {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "unquant-linear"
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.b.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        let mut buffer = Vec::new();

        // Version is always first!

        buffer.extend(&UQFF_VERSION.to_le_bytes());

        // ISQ type for unquant is 1
        buffer.push(QuantizedSerdeType::Unquant as u8);

        // Has bias
        buffer.push(bias.is_some() as u8);

        // Weight
        serialize_tensor(&mut buffer, &self.w)?;

        if let Some(bias) = &bias {
            // Bias
            serialize_tensor(&mut buffer, bias)?;
        }

        Ok(Cow::from(buffer))
    }

    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Unquant as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Unquant as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let _acquired_load_guard = guard.acquire(device);
        let w = deserialize_tensor(&mut buffer, device)?;

        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            w,
            b,
            stats: None,
            calib: None,
        }))
    }
    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Unquant as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Unquant as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let _acquired_load_guard = guard.acquire(device);
        let w = deserialize_tensor(&mut buffer, device)?;

        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        Ok((
            Arc::new(Self {
                w,
                b: None,
                stats: None,
                calib: None,
            }),
            b,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lin(out: usize, inp: usize, bias: bool) -> Result<UnquantLinear> {
        let dev = Device::Cpu;
        // Deterministic, non-symmetric weights: a constant or symmetric matrix
        // would make a transposed or broadcast result compare equal by luck.
        let w: Vec<f32> = (0..out * inp)
            .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
            .collect();
        let w = Tensor::from_vec(w, (out, inp), &dev)?;
        let b = if bias {
            let v: Vec<f32> = (0..out).map(|i| (i % 5) as f32 - 2.0).collect();
            Some(Tensor::from_vec(v, out, &dev)?)
        } else {
            None
        };
        UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(w, b)))
    }

    fn act(rows: usize, inp: usize) -> Result<Tensor> {
        let v: Vec<f32> = (0..rows * inp)
            .map(|i| ((i % 23) as f32 - 11.0) / 32.0)
            .collect();
        Tensor::from_vec(v, (rows, inp), &Device::Cpu)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.max_all()?.to_scalar::<f32>()
    }

    /// **The mathematical claim behind the flatten, and the measurement of what
    /// it costs in precision.**
    ///
    /// `forward` reshapes `[B, T, hidden]` to `[B*T, hidden]` so the weight is
    /// streamed once instead of once per batch row. That is only sound if the
    /// two formulations are the same mathematics. They are — but they are not
    /// the same *arithmetic*: `MatMul::matmul` casts to F16 on CPU, and a
    /// GEMM's accumulation order depends on `m`, so `m = B` and B separate
    /// `m = 1` calls round differently.
    ///
    /// This test asserts agreement inside an F16 budget rather than bit
    /// equality, and that distinction is the reason `forward` excludes CPU: on
    /// x86 the divergence measured 1.57e-2 relative in a V4 batched-vs-solo
    /// test whose tolerance was 1e-2 — a tolerance that had never been
    /// exercised, because before the flatten the two paths were bit identical.
    #[test]
    fn two_d_and_rank3_agree_within_the_cpu_f16_budget() -> Result<()> {
        // Relative to the magnitude present, generous enough for F16
        // re-blocking and far tighter than any structural error (a wrong
        // broadcast or a transposed weight is O(1) here).
        const F16_BUDGET: f32 = 1e-2;
        for bias in [false, true] {
            let l = lin(7, 5, bias)?;
            let flat = act(6, 5)?;
            let want = l.forward(&flat)?;
            let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1.0);

            for shape in [(6usize, 1usize), (3, 2), (2, 3), (1, 6)] {
                let a = flat.reshape((shape.0, shape.1, 5))?;
                let got = l.forward(&a)?;
                assert_eq!(
                    got.dims(),
                    &[shape.0, shape.1, 7],
                    "batch dims must survive (bias={bias})"
                );
                let diff = max_abs_diff(&got.reshape((6, 7))?, &want)?;
                assert!(
                    diff <= F16_BUDGET * scale,
                    "rank-3 {shape:?} disagreed with the 2-D result by {diff} \
                     (budget {}, bias={bias}). That is structural, not rounding.",
                    F16_BUDGET * scale
                );
            }
        }
        Ok(())
    }

    /// **The CPU reference oracle must be byte-stable.**
    ///
    /// Every V4 correctness test in this workspace compares a batched run to a
    /// solo run on CPU, and those comparisons were calibrated against bit
    /// identity. `forward` therefore leaves CPU on its original path, and this
    /// pins it: a rank-3 activation must produce **exactly** what the
    /// pre-change formulation produced — `broadcast_left` into a batched
    /// matmul — element for element.
    #[test]
    fn cpu_forward_still_takes_the_broadcast_path_bit_for_bit() -> Result<()> {
        let l = lin(7, 5, false)?;
        let a = act(6, 5)?.reshape((3, 2, 5))?;

        // The pre-change formulation, spelled out.
        let w = l.w.broadcast_left(3usize)?;
        let want = MatMul.matmul(&a, &w.t()?)?;

        let got = l.forward(&a)?;
        assert_eq!(got.dims(), want.dims());
        assert_eq!(
            max_abs_diff(&got, &want)?,
            0.0,
            "CPU is the oracle the V4 suite is calibrated against; its \
             arithmetic must not move"
        );
        Ok(())
    }

    /// The shape arithmetic the device path depends on, checked where it can
    /// actually be run. Rank 3 and rank 4 both collapse to one `m`, and the
    /// dims handed back are exactly what restores the original shape.
    #[test]
    fn flatten_batch_dims_collapses_every_leading_dim() -> Result<()> {
        let a = act(8, 4)?.reshape((2, 2, 2, 4))?;
        let (flat, dims) = flatten_batch_dims(&a)?;
        assert_eq!(flat.dims(), &[8, 4], "every leading dim folds into m");
        assert_eq!(dims, vec![2, 2, 2]);
        assert_eq!(
            max_abs_diff(&flat.reshape((2, 2, 2, 4))?, &a)?,
            0.0,
            "the flatten must be a pure reshape"
        );

        let b = act(6, 5)?.reshape((3, 2, 5))?;
        let (flat, dims) = flatten_batch_dims(&b)?;
        assert_eq!(flat.dims(), &[6, 5]);
        assert_eq!(dims, vec![3, 2]);
        Ok(())
    }

    /// A non-contiguous activation must reach `reshape`, which candle services
    /// with a strided copy rather than an error. This is the only thing
    /// standing between a transposed view and `matmul` on the device path.
    #[test]
    fn flatten_batch_dims_handles_a_non_contiguous_activation() -> Result<()> {
        // [2, 4, 3] transposed to [2, 3, 4]: the last dim is no longer the
        // fastest-varying axis.
        let base = act(8, 3)?.reshape((2, 4, 3))?;
        let a = base.transpose(1, 2)?;
        assert!(!a.is_contiguous());

        let (flat, dims) = flatten_batch_dims(&a)?;
        assert_eq!(flat.dims(), &[6, 4]);
        assert_eq!(dims, vec![2, 3]);
        assert_eq!(
            max_abs_diff(&flat, &a.contiguous()?.reshape((6, 4))?)?,
            0.0,
            "the strided copy must preserve element order"
        );
        Ok(())
    }
}
