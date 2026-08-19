//! Unified MoE experts layer supporting multiple backends and weight formats.
//!
//! This module provides `MoEExperts`, a flexible experts layer that:
//! - Does NOT carry the gate (router) - gate is external
//! - Supports both per-expert and stacked weight formats
//! - Handles backend selection (fused/fast/slow)
//! - Manages tensor parallelism with all-reduce

use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{
    ExpertSubset, FusedExperts, MatMul, PackedExperts, QuantMethod, QuantizedConfig,
    ShardedVarBuilder, SumAllReduce,
};
use std::sync::Arc;

use crate::cuda::moe;
use crate::layers::Activation;
use crate::moe::expert_parallel::ExpertParallelPlan;
use crate::moe::shard;

/// Apply the trained DeepSeek V4 SwiGLU clamp to the **pre-activation** gate
/// and up projections, returning both in `F32`.
///
/// Authority: `sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh:55-73`
/// (`silu_and_mul<kApplySwigluLimit = true>`, which cites DeepGEMM
/// `sm100_fp8_fp4_mega_moe.cuh:984-997` as source of truth and notes the clamp
/// "must" happen in bf16):
///
/// ```text
/// gate = __hmin2(gate, {limit, limit});    // ONE-SIDED upper clamp
/// up   = __hmax2(up, {-limit, -limit});
/// up   = __hmin2(up, {limit, limit});      // SYMMETRIC clamp
/// ... silu = g / (1 + expf(-g)); val = silu * u;
/// ```
///
/// Three properties this pins, all of which a naive "clamp the output"
/// reading gets wrong:
/// 1. the clamp is **pre-activation** — the product is never clamped;
/// 2. the gate is **not** clamped from below;
/// 3. `up` is clamped on both sides.
///
/// The same device function backs `silu_and_mul_clamp`, which the reference
/// applies to the **shared** expert (`srt/models/deepseek_v2.py:318-323`, with
/// the limit threaded in at `:619` via `swiglu_limit=getattr(config,
/// "swiglu_limit", None)`), so routed and shared experts clamp identically.
/// `deepseek-ai/DeepSeek-V4-Flash`'s published `config.json` carries
/// `"swiglu_limit": 10.0`, so the reference clamps unconditionally — there is
/// no config-dependent escape (wave13-AH).
///
/// Clamping after the `F32` upcast is equivalent to the reference's bf16 clamp:
/// `min`/`max` are monotone, `10.0` is exact in bf16, and the reference upcasts
/// to fp32 for the `silu` anyway.
pub(crate) fn swiglu_clamp(gate: &Tensor, up: &Tensor, limit: f32) -> Result<(Tensor, Tensor)> {
    let limit = limit as f64;
    let gate = gate.to_dtype(DType::F32)?.minimum(limit)?;
    let up = up.to_dtype(DType::F32)?.clamp(-limit, limit)?;
    Ok((gate, up))
}

/// Configuration for MoEExperts
pub struct MoEExpertsConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    /// SwiGLU clamp limit (DeepSeek V4: 10.0). When `Some`, the gate and up
    /// projections are clamped to `[-limit, limit]` before the activation.
    /// V4 was trained with this clamp; without it, experts whose activations
    /// exceed the limit explode (RUN-161). `None` = unclamped (other models).
    pub swiglu_limit: Option<f32>,
}

/// Backend selection for MoE experts
pub enum MoEExpertsBackend {
    /// Use fused CUDA kernels with raw tensors (fastest for CUDA unquantized)
    Fused,
    /// Use gather-based implementation (good for Metal, ISQ)
    Fast,
    /// Use loop-based implementation (fallback for quantized)
    Slow,
}

impl MoEExpertsBackend {
    /// Determine the best backend based on device and quantization settings
    pub fn select(
        device: &Device,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
    ) -> Self {
        // RUN-161 collapse isolation: force the reference-correct per-expert
        // loop (Slow) to test whether the CUDA gather kernel is mis-dispatching
        // experts (collapse). If output stays collapsed under Slow, the bug is
        // in the shared dequant/quant, not the gather kernel.
        if std::env::var_os("ARC_MOE_SLOW").is_some() {
            return Self::Slow;
        }
        let has_immediate_isq = mistralrs_quant::get_immediate_isq().is_some();
        let use_fast = device.is_metal()
            || (device.is_cuda()
                && (loading_isq || quantization_config.is_some() || has_immediate_isq));

        if use_fast {
            Self::Fast
        } else if quantization_config.is_none()
            && !loading_isq
            && !has_immediate_isq
            && device.is_cuda()
        {
            Self::Fused
        } else {
            Self::Slow
        }
    }
}

/// Internal representation of fused expert weights for CUDA kernels
struct FusedExpertsWeights {
    /// gate_up weights: [E, N, K] for standard, [E, K, N] for stacked
    gate_up_w: Tensor,
    /// down weights: [E, N, K] for standard, [E, K, N] for stacked
    down_w: Tensor,
    /// Size of intermediate dimension (after sharding)
    w_size_n: usize,
    /// Whether weights are in stacked format [E, K, N]
    stacked_format: bool,
}

/// Internal representation for gather-based experts (Metal/ISQ)
struct FastExpertsWeights {
    fused_gate_proj: Arc<dyn QuantMethod>,
    fused_up_proj: Arc<dyn QuantMethod>,
    fused_down_proj: Arc<dyn QuantMethod>,
}

/// Internal representation for loop-based experts (quantized fallback)
struct SlowExpertsWeights {
    experts: PackedExperts,
}

/// MoE experts layer without gate
///
/// This struct encapsulates the expert weights and forward logic,
/// but does NOT include the routing gate. The caller is responsible
/// for computing routing weights and topk indices.
pub struct MoEExperts {
    backend: MoEExpertsBackendImpl,
    act: Activation,
    num_experts_per_tok: usize,
    swiglu_limit: Option<f32>,
    all_reduce: SumAllReduce,
    world_size: usize,
    /// This rank's expert-parallel view. `ExpertParallelPlan::single` (the
    /// default) is the pre-EP behaviour: `localize` is the identity, so the
    /// forward is bit-for-bit unchanged.
    ep: ExpertParallelPlan,
    /// Set when the expert weights arrive from a UQFF artifact, which holds
    /// **every** expert: the slice cannot happen at construction, so it must be
    /// applied after `load_from_artifacts`. While this is `Some`, the layer
    /// holds all `E` experts while claiming to own `E / ep_size`, so
    /// [`Self::forward`] refuses to run rather than silently placing the whole
    /// expert set on every card — a capacity bug that would otherwise show up
    /// only as "EP bought nothing".
    pending_expert_subset: Option<ExpertSubset>,
}

enum MoEExpertsBackendImpl {
    Fused(FusedExpertsWeights),
    Fast(FastExpertsWeights),
    Slow(SlowExpertsWeights),
}

impl MoEExperts {
    /// Create MoEExperts with automatic backend selection
    ///
    /// Automatically detects weight format (stacked vs per-expert) and
    /// selects the appropriate backend based on device and quantization.
    pub fn new(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Self> {
        let backend = MoEExpertsBackend::select(&layer_device, loading_isq, quantization_config);
        Self::new_with_backend(
            cfg,
            vb,
            layer_device,
            comm,
            backend,
            quantization_config,
            act,
            &ExpertParallelPlan::single(cfg.num_experts),
        )
    }

    /// Create MoEExperts with automatic backend selection, sharding the routed
    /// experts across the ranks described by `ep`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_expert_parallel(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
        ep: &ExpertParallelPlan,
    ) -> Result<Self> {
        let backend = MoEExpertsBackend::select(&layer_device, loading_isq, quantization_config);
        Self::new_with_backend(
            cfg,
            vb,
            layer_device,
            comm,
            backend,
            quantization_config,
            act,
            ep,
        )
    }

    /// Create MoEExperts with explicit backend selection
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_backend(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        backend: MoEExpertsBackend,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
        ep: &ExpertParallelPlan,
    ) -> Result<Self> {
        if ep.placement().num_experts() != cfg.num_experts {
            candle_core::bail!(
                "expert parallelism: plan covers {} experts but this layer has {}",
                ep.placement().num_experts(),
                cfg.num_experts
            );
        }
        let subset = if ep.is_enabled() {
            ExpertSubset::owned(cfg.num_experts, ep.owned_experts().to_vec())?
        } else {
            ExpertSubset::all(cfg.num_experts)
        };

        let experts_vb = vb.pp("experts").set_device(layer_device.clone());

        // Detect format: stacked has "gate_up_proj", per-expert has "0.gate_proj"
        let is_stacked = experts_vb.contains_tensor("gate_up_proj");
        // The UQFF path builds placeholder layers here and fills them in after
        // `load_from_artifacts`, so the expert slice has to be applied later.
        let deferred_slice = ep.is_enabled() && mistralrs_quant::loading_from_uqff();

        let backend_impl = match backend {
            MoEExpertsBackend::Fused => {
                if is_stacked {
                    MoEExpertsBackendImpl::Fused(Self::load_fused_stacked(
                        cfg, experts_vb, comm, &subset,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Fused(Self::load_fused_standard(
                        cfg, experts_vb, comm, &subset,
                    )?)
                }
            }
            MoEExpertsBackend::Fast => {
                if is_stacked {
                    MoEExpertsBackendImpl::Fast(Self::load_fast_stacked(
                        cfg,
                        vb,
                        quantization_config,
                        &subset,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Fast(Self::load_fast_standard(
                        cfg,
                        vb,
                        quantization_config,
                        &subset,
                    )?)
                }
            }
            MoEExpertsBackend::Slow => MoEExpertsBackendImpl::Slow(Self::load_slow(
                cfg,
                experts_vb,
                comm,
                quantization_config,
                &subset,
            )?),
        };

        Ok(Self {
            backend: backend_impl,
            act,
            num_experts_per_tok: cfg.num_experts_per_tok,
            swiglu_limit: cfg.swiglu_limit,
            all_reduce: SumAllReduce::new(comm),
            world_size: comm.world_size(),
            ep: ep.clone(),
            pending_expert_subset: deferred_slice.then_some(subset),
        })
    }

    /// Load fused weights in standard per-expert format
    fn load_fused_standard(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        subset: &ExpertSubset,
    ) -> Result<FusedExpertsWeights> {
        // EP and TP split this layer on different axes and stage 1 does not
        // combine them: under EP each rank owns whole experts, so the
        // intermediate dimension stays whole. See `PackedExperts::new`.
        let (tp_world, tp_rank) = if subset.is_all() {
            (comm.world_size(), comm.rank())
        } else {
            (1, 0)
        };
        let mut gate_up_experts = Vec::with_capacity(subset.len());
        let mut down_experts = Vec::with_capacity(subset.len());

        for i in subset.ids().iter().copied() {
            let expert_vb = experts_vb.pp(i.to_string());
            // n x k format
            let gate_expert = expert_vb.pp("gate_proj").get_with_hints(
                (cfg.moe_intermediate_size, cfg.hidden_size),
                "weight",
                shard(0, tp_rank, tp_world),
            )?;
            let up_expert = expert_vb.pp("up_proj").get_with_hints(
                (cfg.moe_intermediate_size, cfg.hidden_size),
                "weight",
                shard(0, tp_rank, tp_world),
            )?;
            let down_expert = expert_vb.pp("down_proj").get_with_hints(
                (cfg.hidden_size, cfg.moe_intermediate_size),
                "weight",
                shard(1, tp_rank, tp_world),
            )?;
            // Pack gate_proj and up_proj
            let gate_up_expert = Tensor::cat(&[&gate_expert, &up_expert], 0)?;

            gate_up_experts.push(gate_up_expert);
            down_experts.push(down_expert);
        }

        let gate_up_w = Tensor::stack(&gate_up_experts, 0)?;
        let down_w = Tensor::stack(&down_experts, 0)?;
        let w_size_n = gate_up_w.dim(1)? / 2;

        Ok(FusedExpertsWeights {
            gate_up_w,
            down_w,
            w_size_n,
            stacked_format: false,
        })
    }

    /// Load fused weights in stacked format
    fn load_fused_stacked(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        subset: &ExpertSubset,
    ) -> Result<FusedExpertsWeights> {
        let num_experts = cfg.num_experts;
        // See `load_fused_standard`: EP owns whole experts, so TP is off.
        let (tp_world, tp_rank) = if subset.is_all() {
            (comm.world_size(), comm.rank())
        } else {
            (1, 0)
        };

        // Stacked format has two conventions:
        // Convention A: [num_experts, hidden, inter*2] (CUDA kernel format)
        // Convention B (nn.Linear): [num_experts, inter*2, hidden]
        // Try A first, fall back to B with transpose.
        let gate_up_w = experts_vb
            .get_with_hints(
                (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                "gate_up_proj",
                shard(2, tp_rank, tp_world),
            )
            .or_else(|_| {
                experts_vb
                    .get_with_hints(
                        (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                        "gate_up_proj",
                        shard(1, tp_rank, tp_world),
                    )
                    .and_then(|t| t.transpose(1, 2)?.contiguous())
            })?;

        let down_w = experts_vb
            .get_with_hints(
                (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
                "down_proj",
                shard(1, tp_rank, tp_world),
            )
            .or_else(|_| {
                experts_vb
                    .get_with_hints(
                        (num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
                        "down_proj",
                        shard(2, tp_rank, tp_world),
                    )
                    .and_then(|t| t.transpose(1, 2)?.contiguous())
            })?;

        // Expert parallelism: keep only this rank's slice of the expert axis.
        let gate_up_w = subset.select_dim0(&gate_up_w)?;
        let down_w = subset.select_dim0(&down_w)?;

        let w_size_n = gate_up_w.dim(2)? / 2;

        Ok(FusedExpertsWeights {
            gate_up_w,
            down_w,
            w_size_n,
            stacked_format: true,
        })
    }

    /// Load fast (gather-based) weights in standard per-expert format
    fn load_fast_standard(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        quantization_config: &Option<QuantizedConfig>,
        subset: &ExpertSubset,
    ) -> Result<FastExpertsWeights> {
        let FusedExperts {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        } = FusedExperts::new(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            subset,
            quantization_config,
            vb,
        )?;

        Ok(FastExpertsWeights {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        })
    }

    /// Load fast (gather-based) weights in stacked format
    fn load_fast_stacked(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        quantization_config: &Option<QuantizedConfig>,
        subset: &ExpertSubset,
    ) -> Result<FastExpertsWeights> {
        // FusedExperts auto-detects stacked format
        let FusedExperts {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        } = FusedExperts::new(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            subset,
            quantization_config,
            vb,
        )?;

        Ok(FastExpertsWeights {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        })
    }

    /// Load slow (loop-based) weights using PackedExperts
    fn load_slow(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        quantization_config: &Option<QuantizedConfig>,
        subset: &ExpertSubset,
    ) -> Result<SlowExpertsWeights> {
        let experts = PackedExperts::new(
            subset,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            quantization_config,
            false,
            comm,
            experts_vb,
        )?;

        Ok(SlowExpertsWeights { experts })
    }

    /// Gated activation for one expert step: `act(gate) * up`, with the V4
    /// SwiGLU clamp applied pre-activation when `swiglu_limit` is set.
    ///
    /// Every backend routes its activation stage through here so the clamp
    /// cannot be dropped on one path again (audit finding 5: it was live on
    /// exactly one of five). With `swiglu_limit == None` this is bit-for-bit
    /// the pre-clamp expression, so no non-V4 model changes.
    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let Some(limit) = self.swiglu_limit else {
            return up.mul(&gate.apply(&self.act)?);
        };
        // Fused single-launch path (cuda/swiglu_clamp.cu, ported from vLLM
        // `silu_and_mul_clamp` / SGLang `silu_and_mul<kApplySwigluLimit>`).
        // Bit-identical to the candle chain below; see that file's contract
        // and the on-GPU A/B in cuda/swiglu_clamp.rs. Returns None whenever
        // the call is not eligible, so every other shape/dtype/device keeps
        // the old chain untouched.
        //
        // ARC_NO_FUSED_SWIGLU=1 forces the old chain — this is what the A/B
        // measurement legs toggle, and it must stay a real escape hatch.
        if matches!(self.act, Activation::Silu | Activation::Swish)
            && !crate::cuda::swiglu_clamp::fused_swiglu_disabled()
        {
            if let Some(out) = crate::cuda::swiglu_clamp::swiglu_clamp_cuda(gate, up, limit)? {
                crate::models::deepseek4::v4_stat_dbg(&out, "exp.prod_fused");
                return Ok(out);
            }
        }
        let out_dtype = gate.dtype();
        let (gate, up) = swiglu_clamp(gate, up, limit)?;
        let act_gate = gate.apply(&self.act)?;
        crate::models::deepseek4::v4_stat_dbg(&act_gate, "exp.act_gate");
        up.mul(&act_gate)?.to_dtype(out_dtype)
    }

    /// Split a packed `[.., 2 * w_size_n]` gate_up projection and run
    /// [`Self::swiglu`] over the halves. This is the activation stage of
    /// [`Self::forward_fused`]; it is factored out because the GEMMs around it
    /// are CUDA-only, and this arithmetic is not.
    fn swiglu_gate_up(&self, gate_up: &Tensor, w_size_n: usize) -> Result<Tensor> {
        let gate = gate_up.narrow(D::Minus1, 0, w_size_n)?.contiguous()?;
        let up = gate_up
            .narrow(D::Minus1, w_size_n, w_size_n)?
            .contiguous()?;
        self.swiglu(&gate, &up)
    }

    /// Forward pass through experts
    ///
    /// # Arguments
    /// * `xs` - Input tensor of shape [batch, seq_len, hidden_dim]
    /// * `topk_weights` - Top-k routing weights of shape [num_tokens, num_experts_per_tok]
    /// * `topk_ids` - Top-k expert indices of shape [num_tokens, num_experts_per_tok]
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_dim]
    pub fn forward(&self, xs: &Tensor, topk_weights: Tensor, topk_ids: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        // Prefill = processing multiple tokens; Decode = single token generation
        let is_prefill = seq_len > 1;

        if let Some(pending) = &self.pending_expert_subset {
            candle_core::bail!(
                "expert parallelism: this layer still holds all {} experts — the UQFF expert \
                 slice down to {} was never applied. Running would put the whole expert set on \
                 every rank, so EP would silently buy nothing.",
                pending.global(),
                pending.len()
            );
        }

        // Expert parallelism: rewrite the global expert ids to this rank's
        // local ids and zero the weight of every slot this rank does not own.
        // The unowned slots stay in place (pinned to local expert 0) so every
        // rank runs identical shapes and the combine below is a plain sum.
        let (local_ids, topk_weights) = self.ep.localize(topk_ids, &topk_weights)?;
        let topk_ids = &local_ids;

        // Exactly one of the three runs; the other two are declared unreachable
        // so the report shows which expert backend this build actually took
        // instead of leaving three silent zeros.
        let mut ys = match &self.backend {
            MoEExpertsBackendImpl::Fused(weights) => {
                arc_profiler::mark_unreachable(
                    "experts.fast",
                    "MoEExpertsBackend::select chose Fused (CUDA, unquantised experts)",
                    "moe/experts.rs:103",
                );
                let _s = arc_profiler::device_span("experts.fused");
                self.forward_fused(xs, &topk_weights, topk_ids, weights, is_prefill)?
            }
            MoEExpertsBackendImpl::Fast(weights) => {
                arc_profiler::mark_unreachable(
                    "experts.fused",
                    "MoEExpertsBackend::select chose Fast (CUDA/Metal with ISQ or prequantised \
                     experts) — this is the qtip2b serving path",
                    "moe/experts.rs:97",
                );
                let _s = arc_profiler::device_span("experts.fast");
                self.forward_fast(xs, &topk_weights, topk_ids, weights)?
            }
            MoEExpertsBackendImpl::Slow(weights) => {
                let _s = arc_profiler::device_span("experts.slow");
                self.forward_slow(xs, &topk_weights, topk_ids, weights)?
            }
        };

        // The combine. Under tensor parallelism this sums the intermediate
        // shards; under expert parallelism it sums the per-rank partials
        // `y_r = Σ_{j owned by r} w[t,j] · Expert_{g[t,j]}(x[t])`. Both are the
        // same collective on the same communicator.
        if self.world_size > 1 {
            let _s = arc_profiler::device_span("experts.all_reduce");
            ys = self.all_reduce.sum_all_reduce(&ys)?;
        }

        ys.reshape((b_size, seq_len, hidden_dim))
    }

    /// This rank's expert-parallel view.
    pub fn expert_parallel(&self) -> &ExpertParallelPlan {
        &self.ep
    }

    /// Apply the deferred UQFF expert slice, after `load_from_artifacts` has
    /// replaced the placeholder layers with the deserialized ones.
    ///
    /// Returns the number of layers narrowed. Clears the guard in
    /// [`Self::forward`]; until it is called, an expert-parallel layer loaded
    /// from a UQFF refuses to run.
    pub fn apply_pending_expert_subset(&mut self) -> Result<usize> {
        let Some(subset) = self.pending_expert_subset.take() else {
            return Ok(0);
        };
        let mut narrowed = 0usize;
        for layer in self.get_isq_layers() {
            let sliced = layer.select_experts(&subset.ids())?;
            *layer = sliced;
            narrowed += 1;
        }
        Ok(narrowed)
    }

    /// Fused CUDA kernel forward pass
    fn forward_fused(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        weights: &FusedExpertsWeights,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (_b_size, _seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let (num_tokens, _) = xs.dims2()?;

        // Sort tokens by expert for efficient processing
        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use crate::ops::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        // First GEMM: gate_up projection
        let gate_up = if weights.stacked_format {
            moe::moe_gemm_transposed(
                &xs,
                &weights.gate_up_w,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        } else {
            moe::moe_gemm(
                &xs,
                &weights.gate_up_w,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        };

        // Split and apply the (clamped) gated activation. Audit finding 5:
        // this path used to drop the V4 SwiGLU clamp entirely, and it is the
        // default CUDA path whenever no ISQ/quantization config is present.
        let down_inputs = self
            .swiglu_gate_up(&gate_up, weights.w_size_n)?
            .reshape(((), weights.w_size_n))?;

        // Second GEMM: down projection with weight aggregation
        let ys = if weights.stacked_format {
            moe::moe_gemm_transposed(
                &down_inputs,
                &weights.down_w,
                &Some(topk_weights.clone()),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        } else {
            moe::moe_gemm(
                &down_inputs,
                &weights.down_w,
                &Some(topk_weights.clone()),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        };

        ys.reshape((num_tokens, (), hidden_dim))?.sum(D::Minus2)
    }

    /// Gather-based forward pass (Metal/ISQ)
    fn forward_fast(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        weights: &FastExpertsWeights,
    ) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let num_tokens = b_size * seq_len;

        let xs_flat = xs.reshape((num_tokens, hidden_dim))?;

        let ys = if xs.device().is_cuda() {
            // CUDA path: use indexed_moe_forward compatible shapes.
            // FP8 gather_forward accepts (num_tokens, 1, hidden_dim) and broadcasts
            // internally, but QTIP requires the second dim to match num_experts_per_tok.
            // Expand to (num_tokens, num_experts_per_tok, hidden_dim) for compatibility.
            let xs = xs_flat
                .unsqueeze(1)?
                .expand((num_tokens, self.num_experts_per_tok, hidden_dim))?
                .contiguous()?;
            let gate = {
                let _s = arc_profiler::device_span("experts.gate_proj");
                weights
                    .fused_gate_proj
                    .gather_forward_autocast(&xs, topk_ids)?
            };
            let up = {
                let _s = arc_profiler::device_span("experts.up_proj");
                weights.fused_up_proj.gather_forward_autocast(&xs, topk_ids)?
            };
            crate::models::deepseek4::v4_stat_dbg(&gate, "exp.gate");
            crate::models::deepseek4::v4_stat_dbg(&up, "exp.up");
            // RUN-161 collapse localization: cross-token cosine at each expert
            // projection. xs is [num_tokens, top_k, hidden] (expanded); gate/up
            // are [num_tokens, top_k, inter]. pos_dim=0 = tokens. These print
            // immediately before the matching `L{li}.moe_routed` line, so align
            // by proximity in the log.
            crate::models::deepseek4::v4_collapse_dbg(&xs, "exp.in", 0);
            crate::models::deepseek4::v4_collapse_dbg(&gate, "exp.gate", 0);
            crate::models::deepseek4::v4_collapse_dbg(&up, "exp.up", 0);
            // V4 clamped SwiGLU, computed in F32 for stability: `up` clamped to
            // [-limit, limit], `gate` clamped ONLY on the max side, both BEFORE
            // the activation; `silu(gate)*up` in f32, then back to the model
            // dtype for the down projection. RUN-161.
            //
            // Authority is the CUDA kernel `silu_and_mul_masked_post_quant.cuh:55-73`
            // (see [`swiglu_clamp`]) — NOT `inference/model.py:596-606`, which
            // this comment used to cite and which contains no clamp anywhere
            // (audit §"Prior-art corrections").
            let prod = {
                let _s = arc_profiler::device_span("experts.swiglu");
                self.swiglu(&gate, &up)?
            };
            crate::models::deepseek4::v4_stat_dbg(&prod, "exp.prod");
            let down = {
                let _s = arc_profiler::device_span("experts.down_proj");
                weights
                    .fused_down_proj
                    .gather_forward_autocast(&prod, topk_ids)?
            };
            crate::models::deepseek4::v4_stat_dbg(&down, "exp.down");
            crate::models::deepseek4::v4_collapse_dbg(&prod, "exp.prod", 0);
            crate::models::deepseek4::v4_collapse_dbg(&down, "exp.down", 0);
            down
        } else {
            // Metal path: use broadcast gather shapes
            let xs = xs.reshape((b_size, seq_len, 1, 1, hidden_dim))?;
            let indices = topk_ids.reshape((b_size, seq_len, self.num_experts_per_tok))?;
            let gate = weights
                .fused_gate_proj
                .gather_forward_autocast(&xs, &indices)?;
            let up = weights
                .fused_up_proj
                .gather_forward_autocast(&xs, &indices)?;
            // Audit finding 5: this branch (Metal, and every non-CUDA device)
            // dropped the V4 SwiGLU clamp.
            let xs = weights
                .fused_down_proj
                .gather_forward_autocast(&self.swiglu(&gate, &up)?, &indices)?;
            xs.squeeze(D::Minus2)?
                .reshape((num_tokens, self.num_experts_per_tok, hidden_dim))?
        };

        let _s = arc_profiler::device_span("experts.weighted_sum");
        ys.to_dtype(DType::F32)?
            .broadcast_mul(&topk_weights.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .to_dtype(original_dtype)
    }

    /// Loop-based forward pass (quantized fallback)
    fn forward_slow(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
        weights: &SlowExpertsWeights,
    ) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;

        let routing_weights = topk_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let experts_per_tok = topk_ids.to_vec2::<u32>()?;
        let num_experts = weights.experts.gate_proj.len();

        let mut top_x = vec![vec![]; num_experts];
        let mut selected_experts = vec![vec![]; num_experts];

        for (row_idx, (rw, expert_idxs)) in routing_weights
            .iter()
            .zip(experts_per_tok.iter())
            .enumerate()
        {
            for (&rw, &expert_idx) in rw.iter().zip(expert_idxs.iter()) {
                #[allow(clippy::cast_possible_truncation)]
                top_x[expert_idx as usize].push(row_idx as u32);
                selected_experts[expert_idx as usize].push(rw)
            }
        }

        let mut ys = xs.zeros_like()?;
        for expert_idx in 0..num_experts {
            let top_x_expert = &top_x[expert_idx];
            if top_x_expert.is_empty() {
                continue;
            }
            let top_x_tensor = Tensor::new(top_x_expert.as_slice(), xs.device())?;
            let selected_experts_tensor =
                Tensor::new(selected_experts[expert_idx].as_slice(), xs.device())?
                    .reshape(((), 1))?
                    .to_dtype(xs.dtype())?;
            let current_state = xs
                .index_select(&top_x_tensor, 0)?
                .reshape(((), hidden_dim))?;

            // Forward through expert MLP
            let original_dtype = current_state.dtype();
            let mut expert_input = current_state.clone();
            if let Some(t) = weights.experts.gate_proj[expert_idx].quantized_act_type() {
                expert_input = expert_input.to_dtype(t)?;
            }
            let gate_out =
                MatMul.qmethod_matmul(&expert_input, &*weights.experts.gate_proj[expert_idx])?;
            let up_out =
                MatMul.qmethod_matmul(&expert_input, &*weights.experts.up_proj[expert_idx])?;
            // Audit finding 5: this path applied the activation to the raw gate
            // with no V4 SwiGLU clamp. It is the CPU fallback, the quantized
            // fallback, and what `ARC_MOE_SLOW=1` forces.
            let mut current_hidden_states = MatMul.qmethod_matmul(
                &self.swiglu(&gate_out, &up_out)?,
                &*weights.experts.down_proj[expert_idx],
            )?;
            if weights.experts.gate_proj[expert_idx]
                .quantized_act_type()
                .is_some()
            {
                current_hidden_states = current_hidden_states.to_dtype(original_dtype)?;
            }

            let current_hidden_states =
                current_hidden_states.broadcast_mul(&selected_experts_tensor)?;
            ys = ys.index_add(&top_x_tensor, &current_hidden_states, 0)?;
        }

        ys.reshape((b_size * seq_len, hidden_dim))
    }

    /// Get mutable references to quantizable layers for ISQ
    pub fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        match &mut self.backend {
            MoEExpertsBackendImpl::Fused(_) => vec![],
            MoEExpertsBackendImpl::Fast(weights) => {
                vec![
                    &mut weights.fused_gate_proj,
                    &mut weights.fused_up_proj,
                    &mut weights.fused_down_proj,
                ]
            }
            MoEExpertsBackendImpl::Slow(weights) => {
                let mut layers = Vec::new();
                for (gate, (up, down)) in weights.experts.gate_proj.iter_mut().zip(
                    weights
                        .experts
                        .up_proj
                        .iter_mut()
                        .zip(weights.experts.down_proj.iter_mut()),
                ) {
                    layers.push(gate);
                    layers.push(up);
                    layers.push(down);
                }
                layers
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::expert_parallel::ExpertPlacement;
    use candle_core::Device;
    use mistralrs_quant::safetensors::ShardedSafeTensors;
    use std::collections::HashMap;

    /// The V4 clamp limit used throughout these tests. Deliberately 1.0 rather
    /// than the shipped 10.0 so the fixture's pre-activation values (5.0 gate,
    /// 3.0 up) sit *well past* it — an assertion against an unclamped
    /// implementation must move by ~20x, not by rounding.
    const LIMIT: f32 = 1.0;

    /// silu(1.0) * 1.0 — what the reference produces for a gate pre-activation
    /// of 5.0 and an up pre-activation of 3.0 at `limit = 1.0`.
    const CLAMPED: f32 = 0.731_058_6;
    /// silu(5.0) * 3.0 — what an unclamped implementation produces for the
    /// same inputs. 20.4x the clamped value: no tolerance can hide the gap.
    const UNCLAMPED: f32 = 14.899_607;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn comm(device: &Device) -> Arc<mistralrs_quant::Comm> {
        Arc::new(
            mistralrs_quant::Comm::from_device(mistralrs_quant::Id::new(), device, 0, 1).unwrap(),
        )
    }

    fn wrap(tensors: HashMap<String, Tensor>, device: &Device) -> ShardedVarBuilder {
        let backend: Box<dyn candle_nn::var_builder::SimpleBackend + 'static> = Box::new(tensors);
        ShardedSafeTensors::wrap(backend, DType::F32, device.clone())
    }

    /// Stacked-format expert weights: every expert projects a `hidden`-wide
    /// vector of ones onto `gate_pre` / `up_pre` in each of `inter` channels,
    /// and `down_proj` is the identity so the test reads the gated activation
    /// straight out of the layer output.
    fn stacked_experts_vb(
        num_experts: usize,
        hidden: usize,
        inter: usize,
        gate_pre: f32,
        up_pre: f32,
        device: &Device,
    ) -> ShardedVarBuilder {
        assert_eq!(hidden, inter, "identity down_proj requires hidden == inter");
        // [E, hidden, 2*inter]: first `inter` columns feed the gate, rest the up.
        let mut gate_up = Vec::with_capacity(num_experts * hidden * inter * 2);
        for _ in 0..num_experts {
            for _ in 0..hidden {
                for _ in 0..inter {
                    gate_up.push(gate_pre / hidden as f32);
                }
                for _ in 0..inter {
                    gate_up.push(up_pre / hidden as f32);
                }
            }
        }
        // [E, inter, hidden] identity.
        let mut down = Vec::with_capacity(num_experts * inter * hidden);
        for _ in 0..num_experts {
            for i in 0..inter {
                for h in 0..hidden {
                    down.push(if i == h { 1.0f32 } else { 0.0 });
                }
            }
        }
        let mut tensors = HashMap::new();
        tensors.insert(
            "experts.gate_up_proj".to_string(),
            Tensor::from_vec(gate_up, (num_experts, hidden, inter * 2), device).unwrap(),
        );
        tensors.insert(
            "experts.down_proj".to_string(),
            Tensor::from_vec(down, (num_experts, inter, hidden), device).unwrap(),
        );
        wrap(tensors, device)
    }

    fn experts_cfg(
        num_experts: usize,
        hidden: usize,
        inter: usize,
        limit: Option<f32>,
    ) -> MoEExpertsConfig {
        MoEExpertsConfig {
            num_experts,
            num_experts_per_tok: 1,
            hidden_size: hidden,
            moe_intermediate_size: inter,
            swiglu_limit: limit,
        }
    }

    fn build_experts(backend: MoEExpertsBackend, limit: Option<f32>) -> MoEExperts {
        let device = cpu();
        let (num_experts, hidden, inter) = (2usize, 2usize, 2usize);
        let vb = stacked_experts_vb(num_experts, hidden, inter, 5.0, 3.0, &device);
        MoEExperts::new_with_backend(
            &experts_cfg(num_experts, hidden, inter, limit),
            vb,
            device,
            &comm(&cpu()),
            backend,
            &None,
            Activation::Silu,
            &ExpertParallelPlan::single(num_experts),
        )
        .expect("experts load")
    }

    fn max_abs(t: &Tensor) -> f32 {
        t.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap()
            .into_iter()
            .fold(0.0f32, |acc, v| acc.max(v.abs()))
    }

    /// Relative tolerance. `MatMul::matmul` downcasts CPU matmuls to F16
    /// (`mistralrs-quant/src/lib.rs:595-599`), so any end-to-end CPU expert
    /// forward carries ~5e-4 relative error. That is four orders of magnitude
    /// smaller than the 20.4x gap between CLAMPED and UNCLAMPED, so it cannot
    /// blur the two.
    fn all_close(t: &Tensor, expected: f32) -> bool {
        t.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap()
            .into_iter()
            .all(|v| (v - expected).abs() / expected.abs() < 2e-3)
    }

    /// The clamp is asymmetric by design, and getting it wrong in either
    /// direction is a silent numerical divergence against the trained weights.
    ///
    /// Reference `silu_and_mul_masked_post_quant.cuh:55-73`:
    /// `gate = __hmin2(gate, limit)` (upper only), `up = __hmin2(__hmax2(up,
    /// -limit), limit)` (both sides), both PRE-activation.
    #[test]
    fn swiglu_clamp_is_one_sided_on_gate_and_symmetric_on_up() {
        let device = cpu();
        let vals = vec![-50.0f32, -5.0, 5.0, 50.0];
        let gate = Tensor::from_vec(vals.clone(), (4,), &device).unwrap();
        let up = Tensor::from_vec(vals, (4,), &device).unwrap();

        let (g, u) = swiglu_clamp(&gate, &up, 10.0).unwrap();

        // Gate: upper clamp only — large negatives survive untouched.
        assert_eq!(g.to_vec1::<f32>().unwrap(), vec![-50.0, -5.0, 5.0, 10.0]);
        // Up: symmetric.
        assert_eq!(u.to_vec1::<f32>().unwrap(), vec![-10.0, -5.0, 5.0, 10.0]);
    }

    /// `forward_slow` — the CPU fallback, the quantized fallback, and what
    /// `ARC_MOE_SLOW=1` forces. Audit finding 5 row 3. End-to-end through the
    /// real `MoEExperts::forward`.
    #[test]
    fn slow_backend_applies_swiglu_clamp() {
        let device = cpu();
        let experts = build_experts(MoEExpertsBackend::Slow, Some(LIMIT));
        let xs = Tensor::ones((1, 1, 2), DType::F32, &device).unwrap();
        let topk_weights = Tensor::ones((1, 1), DType::F32, &device).unwrap();
        let topk_ids = Tensor::from_vec(vec![0u32], (1, 1), &device).unwrap();

        let out = experts.forward(&xs, topk_weights, &topk_ids).unwrap();
        assert_eq!(out.dims(), &[1, 1, 2]);
        assert!(
            all_close(&out, CLAMPED),
            "slow backend dropped the swiglu clamp: {:?}",
            out.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    /// The other half of the previous test: with no limit the SAME fixture
    /// must produce the unclamped value. Without this, a `swiglu_clamp` that
    /// (say) always clamped to 1.0 regardless of input would pass, and every
    /// non-V4 model would silently change.
    #[test]
    fn slow_backend_without_a_limit_is_unclamped() {
        let device = cpu();
        let experts = build_experts(MoEExpertsBackend::Slow, None);
        let xs = Tensor::ones((1, 1, 2), DType::F32, &device).unwrap();
        let topk_weights = Tensor::ones((1, 1), DType::F32, &device).unwrap();
        let topk_ids = Tensor::from_vec(vec![0u32], (1, 1), &device).unwrap();

        let out = experts.forward(&xs, topk_weights, &topk_ids).unwrap();
        assert!(
            all_close(&out, UNCLAMPED),
            "unlimited experts changed: {:?}",
            out.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    /// `forward_fused` — the default CUDA path whenever no ISQ or quantization
    /// config is present (`MoEExpertsBackend::select`). Audit finding 5 row 1.
    ///
    /// The GEMMs around it are CUDA-only, so this drives the activation stage
    /// `forward_fused` actually calls, including the gate/up split offsets.
    #[test]
    fn fused_backend_activation_stage_applies_swiglu_clamp() {
        let device = cpu();
        let experts = build_experts(MoEExpertsBackend::Slow, Some(LIMIT));
        // What the first fused GEMM emits: [tokens, 2 * w_size_n], gate half
        // first. Pre-activation 5.0 / 3.0, as in the end-to-end fixture.
        let gate_up = Tensor::from_vec(vec![5.0f32, 5.0, 3.0, 3.0], (1, 4), &device).unwrap();

        let out = experts.swiglu_gate_up(&gate_up, 2).unwrap();
        assert_eq!(out.dims(), &[1, 2]);
        assert!(all_close(&out, CLAMPED), "fused activation stage unclamped");

        let unlimited = build_experts(MoEExpertsBackend::Slow, None);
        assert!(
            all_close(&unlimited.swiglu_gate_up(&gate_up, 2).unwrap(), UNCLAMPED),
            "limitless fused activation stage changed"
        );
    }

    /// `forward_fast` — both branches (CUDA gather, and Metal/other-device
    /// gather) hand their gate/up projections to `swiglu`. Audit finding 5
    /// rows 2 and 4: the Metal branch dropped the clamp; the CUDA branch had
    /// it and must keep it.
    ///
    /// The gathers themselves need CUDA or Metal, so this drives the shared
    /// activation both branches call.
    #[test]
    fn fast_backend_activation_stage_applies_swiglu_clamp() {
        let device = cpu();
        let experts = build_experts(MoEExpertsBackend::Slow, Some(LIMIT));
        let gate = Tensor::from_vec(vec![5.0f32, 5.0], (1, 2), &device).unwrap();
        let up = Tensor::from_vec(vec![3.0f32, 3.0], (1, 2), &device).unwrap();

        assert!(
            all_close(&experts.swiglu(&gate, &up).unwrap(), CLAMPED),
            "fast activation stage unclamped"
        );

        let unlimited = build_experts(MoEExpertsBackend::Slow, None);
        assert!(
            all_close(&unlimited.swiglu(&gate, &up).unwrap(), UNCLAMPED),
            "limitless fast activation stage changed"
        );
    }

    // ---------------------------------------------------------------------
    // Expert parallelism
    //
    // These are UNIT tests of the sharding contract, run in-process with one
    // `MoEExperts` per simulated rank and the combine done by the test. They
    // do NOT validate expert parallelism: the question EP exists to answer is
    // per-collective latency on a real NVLink fabric, and nothing on a CPU can
    // speak to that. What they do establish is that the arithmetic is right,
    // so a hardware run measures speed rather than debugging correctness.
    // ---------------------------------------------------------------------

    /// Stacked expert weights where **every expert computes something
    /// different**: expert `e` projects onto a gate pre-activation of
    /// `gate_pre[e]`, with `up = 1` and an identity `down`.
    ///
    /// Per-expert distinctness is the whole point (DOCTRINE D12): with
    /// identical experts, routing to the wrong one — the exact failure the
    /// global→local remap could introduce — would be invisible.
    fn distinct_stacked_experts_vb(
        gate_pre: &[f32],
        hidden: usize,
        inter: usize,
        device: &Device,
    ) -> ShardedVarBuilder {
        assert_eq!(hidden, inter, "identity down_proj requires hidden == inter");
        let num_experts = gate_pre.len();
        let mut gate_up = Vec::with_capacity(num_experts * hidden * inter * 2);
        for &g in gate_pre {
            for _ in 0..hidden {
                for _ in 0..inter {
                    gate_up.push(g / hidden as f32);
                }
                for _ in 0..inter {
                    gate_up.push(1.0 / hidden as f32);
                }
            }
        }
        let mut down = Vec::with_capacity(num_experts * inter * hidden);
        for _ in 0..num_experts {
            for i in 0..inter {
                for h in 0..hidden {
                    down.push(if i == h { 1.0f32 } else { 0.0 });
                }
            }
        }
        let mut tensors = HashMap::new();
        tensors.insert(
            "experts.gate_up_proj".to_string(),
            Tensor::from_vec(gate_up, (num_experts, hidden, inter * 2), device).unwrap(),
        );
        tensors.insert(
            "experts.down_proj".to_string(),
            Tensor::from_vec(down, (num_experts, inter, hidden), device).unwrap(),
        );
        wrap(tensors, device)
    }

    const EP_GATES: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    const EP_HIDDEN: usize = 2;
    const EP_TOP_K: usize = 2;

    fn ep_experts(plan: &ExpertParallelPlan) -> MoEExperts {
        let device = cpu();
        let vb = distinct_stacked_experts_vb(&EP_GATES, EP_HIDDEN, EP_HIDDEN, &device);
        let cfg = MoEExpertsConfig {
            num_experts: EP_GATES.len(),
            num_experts_per_tok: EP_TOP_K,
            hidden_size: EP_HIDDEN,
            moe_intermediate_size: EP_HIDDEN,
            swiglu_limit: None,
        };
        MoEExperts::new_with_backend(
            &cfg,
            vb,
            device,
            &comm(&cpu()),
            MoEExpertsBackend::Slow,
            &None,
            Activation::Silu,
            plan,
        )
        .expect("experts load")
    }

    /// Three tokens, top-2 routing chosen so every token touches BOTH halves
    /// of the expert set — a routing that stayed inside one rank would make
    /// the combine untested.
    fn ep_routing() -> (Tensor, Tensor, Tensor) {
        let device = cpu();
        let xs = Tensor::ones((1, 3, EP_HIDDEN), DType::F32, &device).unwrap();
        let ids = Tensor::from_vec(vec![0u32, 2, 1, 3, 3, 0], (3, EP_TOP_K), &device).unwrap();
        let w = Tensor::from_vec(
            vec![0.6f32, 0.4, 0.25, 0.75, 0.5, 0.5],
            (3, EP_TOP_K),
            &device,
        )
        .unwrap();
        (xs, w, ids)
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap()
    }

    /// Sum the per-rank partials, as the combine collective does.
    fn ep_combined(placement: &Arc<ExpertPlacement>) -> Vec<f32> {
        let (xs, w, ids) = ep_routing();
        let mut total: Option<Vec<f32>> = None;
        for rank in 0..placement.ep_size() {
            let plan = ExpertParallelPlan::new(placement.clone(), rank, &cpu()).expect("plan");
            let partial = flat(&ep_experts(&plan).forward(&xs, w.clone(), &ids).unwrap());
            total = Some(match total {
                None => partial,
                Some(acc) => acc.iter().zip(&partial).map(|(a, b)| a + b).collect(),
            });
        }
        total.expect("at least one rank")
    }

    fn ep_reference() -> Vec<f32> {
        let (xs, w, ids) = ep_routing();
        let plan = ExpertParallelPlan::single(EP_GATES.len());
        flat(&ep_experts(&plan).forward(&xs, w, &ids).unwrap())
    }

    /// Close enough that no argmax can flip. `MatMul::matmul` downcasts CPU
    /// matmuls to F16 (`mistralrs-quant/src/lib.rs`), so ~5e-4 relative is the
    /// floor even for two runs of the SAME code; the EP re-association adds
    /// nothing on top of that.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// **The acceptance test for the sharding contract**: EP=2's combined
    /// output equals EP=1's.
    ///
    /// Not a claim about expert parallelism's value — that is a hardware
    /// question. A claim that sharding the experts and summing the partials
    /// reproduces the unsharded arithmetic, which is the precondition for any
    /// hardware run to be measuring speed rather than a bug.
    #[test]
    fn ep2_reproduces_ep1_output() {
        let reference = ep_reference();
        let placement = Arc::new(ExpertPlacement::contiguous(EP_GATES.len(), 2).unwrap());
        let combined = ep_combined(&placement);
        assert!(
            max_abs_diff(&reference, &combined) < 1e-3,
            "EP=2 diverged from EP=1:\n  ep1 = {reference:?}\n  ep2 = {combined:?}"
        );
        // The fixture must actually be capable of showing a difference.
        assert!(
            reference.iter().any(|v| v.abs() > 0.1),
            "fixture cannot discriminate: the reference output is ~zero"
        );
    }

    /// The same, with a **permuted** placement (rank 0 owns experts 1 and 3).
    /// Permutation is what the tid2eid-derived balanced placement produces, and
    /// it exercises the global→local remap non-trivially: local index 0 on
    /// rank 0 is global expert 1, so an implementation that forgot to remap
    /// would compute the wrong expert and this test would catch it.
    #[test]
    fn ep2_reproduces_ep1_under_a_permuted_placement() {
        let reference = ep_reference();
        let placement =
            Arc::new(ExpertPlacement::from_expert_to_rank(vec![1, 0, 1, 0], 2).unwrap());
        assert!(!placement.is_contiguous());
        let combined = ep_combined(&placement);
        assert!(
            max_abs_diff(&reference, &combined) < 1e-3,
            "permuted EP=2 diverged from EP=1:\n  ep1 = {reference:?}\n  ep2 = {combined:?}"
        );
    }

    /// EP=4 — one expert per rank — is the sharpest version of the same
    /// contract, and pins that the partition holds when every rank owns
    /// exactly one expert.
    #[test]
    fn ep4_reproduces_ep1_output() {
        let reference = ep_reference();
        let placement = Arc::new(ExpertPlacement::contiguous(EP_GATES.len(), 4).unwrap());
        let combined = ep_combined(&placement);
        assert!(
            max_abs_diff(&reference, &combined) < 1e-3,
            "EP=4 diverged from EP=1:\n  ep1 = {reference:?}\n  ep4 = {combined:?}"
        );
    }

    /// **Mutation proof — dropping a rank's contribution.** This is the
    /// failure mode the whole EP suite exists for: the result stays finite and
    /// fluent, it is simply missing terms. If `ep2_reproduces_ep1_output`
    /// could pass without a working combine, this assertion would fail.
    #[test]
    fn dropping_one_ranks_partial_changes_the_answer() {
        let reference = ep_reference();
        let placement = Arc::new(ExpertPlacement::contiguous(EP_GATES.len(), 2).unwrap());
        let (xs, w, ids) = ep_routing();
        let rank0 = ExpertParallelPlan::new(placement, 0, &cpu()).unwrap();
        let only_rank0 = flat(&ep_experts(&rank0).forward(&xs, w, &ids).unwrap());

        assert!(
            only_rank0.iter().all(|v| v.is_finite()),
            "a dropped rank must still look plausible — that is why it is dangerous"
        );
        assert!(
            max_abs_diff(&reference, &only_rank0) > 1e-2,
            "rank 0 alone matched the full result, so the combine is untested:\n  \
             ep1 = {reference:?}\n  r0 = {only_rank0:?}"
        );
    }

    /// **Mutation proof — a broken expert→device map.** Running rank 0's
    /// placement on both ranks double-counts rank 0's experts and drops
    /// rank 1's. It must not reproduce EP=1.
    #[test]
    fn a_placement_that_is_not_a_partition_changes_the_answer() {
        let reference = ep_reference();
        let placement = Arc::new(ExpertPlacement::contiguous(EP_GATES.len(), 2).unwrap());
        let (xs, w, ids) = ep_routing();
        let rank0 = ExpertParallelPlan::new(placement, 0, &cpu()).unwrap();
        let partial = flat(&ep_experts(&rank0).forward(&xs, w.clone(), &ids).unwrap());
        let doubled: Vec<f32> = partial.iter().map(|v| v * 2.0).collect();
        assert!(
            max_abs_diff(&reference, &doubled) > 1e-2,
            "double-counting one rank matched the correct answer, so the partition \
             property is not being tested"
        );
    }

    /// **Mutation proof — skipping the global→local remap.** Feeding a rank
    /// the raw global ids (what an implementation that sharded the weights but
    /// forgot to renumber the routing would do) selects the wrong experts.
    #[test]
    fn skipping_the_global_to_local_remap_changes_the_answer() {
        let placement = Arc::new(ExpertPlacement::contiguous(EP_GATES.len(), 2).unwrap());
        let (xs, w, ids) = ep_routing();

        let rank1 = ExpertParallelPlan::new(placement, 1, &cpu()).unwrap();
        let correct = flat(&ep_experts(&rank1).forward(&xs, w.clone(), &ids).unwrap());

        // Same rank-1 weights, but routed with the global ids clamped into
        // range instead of remapped — i.e. the remap silently skipped.
        let unmapped_ids = Tensor::from_vec(
            ids.flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()
                .into_iter()
                .map(|g| g % 2)
                .collect::<Vec<u32>>(),
            ids.shape().clone(),
            &cpu(),
        )
        .unwrap();
        let single = ExpertParallelPlan::single(EP_GATES.len());
        let two_expert_cfg_experts = ep_experts(&single);
        let unmapped = flat(
            &two_expert_cfg_experts
                .forward(&xs, w, &unmapped_ids)
                .unwrap(),
        );
        assert!(
            max_abs_diff(&correct, &unmapped) > 1e-2,
            "the remap makes no difference on this fixture, so it is not being tested"
        );
    }

    /// A UQFF-loaded expert-parallel layer must refuse to run until the
    /// deferred slice is applied. Silently running would put all 256 experts
    /// on every card and EP would appear to work while buying nothing.
    #[test]
    fn a_pending_uqff_expert_slice_refuses_to_run() {
        let placement = Arc::new(ExpertPlacement::contiguous(EP_GATES.len(), 2).unwrap());
        let plan = ExpertParallelPlan::new(placement, 0, &cpu()).unwrap();
        let mut experts = ep_experts(&plan);
        experts.pending_expert_subset =
            Some(ExpertSubset::owned(EP_GATES.len(), vec![0, 1]).unwrap());

        let (xs, w, ids) = ep_routing();
        let err = experts.forward(&xs, w.clone(), &ids).unwrap_err();
        assert!(
            format!("{err}").contains("was never applied"),
            "unexpected error: {err}"
        );

        // Applying the slice clears the refusal. (These weights were already
        // narrowed at load, so `select_experts` is not exercised here — that
        // path belongs to the quantized layers; see `QtipLayer` /
        // `Qtip2bLayer::select_experts_concrete`.)
        experts.pending_expert_subset = None;
        assert!(experts.forward(&xs, w, &ids).is_ok());
    }

    /// The clamp must not change the output dtype: the down projection is fed
    /// the model dtype, not F32. (The clamp computes in F32 internally, as the
    /// reference does for the silu.)
    #[test]
    fn swiglu_preserves_the_input_dtype() {
        let device = cpu();
        let experts = build_experts(MoEExpertsBackend::Slow, Some(LIMIT));
        let gate = Tensor::from_vec(vec![5.0f32, 5.0], (1, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let up = Tensor::from_vec(vec![3.0f32, 3.0], (1, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let out = experts.swiglu(&gate, &up).unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        assert!(max_abs(&(out.to_dtype(DType::F32).unwrap() - CLAMPED as f64).unwrap()) < 1e-2);
        // Sanity: the assertion above is not satisfied by the unclamped value.
        assert!((UNCLAMPED - CLAMPED).abs() > 1.0);
    }
}
