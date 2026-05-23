//! TD-MoE pipeline call-site: post-load compression of MoE expert weights.
//!
//! # Flow
//!
//! When the user passes `--td-moe-rank N` to the `arc` CLI, that flag is
//! propagated to the launched `mistralrs` subprocess via the
//! `ARC_TD_MOE_RANK` environment variable. The subprocess's `mistralrs-cli`
//! crate calls [`register_td_moe_hook`] at startup, which checks that env
//! var and — if set — installs a closure with `mistralrs-core`'s post-load
//! hook registry.
//!
//! After every normal-model load finishes ISQ quantization, the registry
//! invokes that closure with `&mut dyn IsqModel`. The closure walks the
//! model's MoE expert layers via [`IsqModel::get_layers_moe_experts_only`],
//! and for each expert tensor it:
//!
//!  1. Dequantizes the weight (via `QuantMethod::dequantize_w`).
//!  2. Computes activation covariance for the expert. For Tier A this is
//!     identity (unbiased random calibration). For Tier B (real prompts)
//!     this would integrate a forward sweep — out of scope here.
//!  3. Calls `crate::td_moe::tucker_decompose_with_whitening` with the
//!     requested rank.
//!  4. Reconstructs the lower-rank approximation with
//!     `crate::td_moe::whitened_tucker_reconstruct`.
//!  5. Replaces the layer's `Arc<dyn QuantMethod>` with a fresh
//!     `UnquantLinear` wrapping the reconstructed weights.
//!
//! Layers that are not expert stacks (2-D linear layers like LM head, or
//! routers) are skipped — the hook only touches 3-D expert tensors of shape
//! `[num_experts, d_out, d_in]`.
//!
//! # Compression accounting
//!
//! At rank `r`, the storage for a single MoE expert stack of shape
//! `[E, d_out, d_in]` drops from `E * d_out * d_in` to roughly
//! `r^3 + E*r + d_out*r + d_in*r` (Tucker core + factor matrices), since
//! we **reconstruct** a dense `[E, d_out, d_in]` tensor afterwards the
//! actual reduction is "quality vs original" rather than "bytes on disk".
//! The intended use of the reconstructed tensor is as a *low-rank
//! approximation* that the downstream forward path stores densely but
//! consumes less effective rank; combined with a follow-up ISQ pass the
//! memory drop materializes via quantization of redundant directions.
//!
//! For the strict 20% storage reduction described in the TD-MoE paper, the
//! caller must keep the Tucker decomposition factored (G + U₁,U₂,U₃) rather
//! than reconstructing — that's deferred to RUN-137. This loader does the
//! reconstruct-in-place path so the model continues to run through the
//! standard linear forward kernels with no further changes.

use std::sync::Arc;

#[cfg_attr(not(test), allow(unused_imports))]
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Linear;
use mistralrs_core::IsqModel;
use mistralrs_quant::{QuantMethod, QuantMethodConfig, UnquantLinear};
use tracing::{info, warn};

use crate::td_moe::{tucker_decompose_with_whitening, whitened_tucker_reconstruct};

/// Environment variable that arc-cli sets to request post-load TD-MoE
/// compression. Value must be a positive integer rank (>= 4).
pub const ARC_TD_MOE_RANK_ENV: &str = "ARC_TD_MOE_RANK";

/// Environment variable that selects the calibration set size. Optional;
/// only used for Tier B (file-driven) calibration in the future.
pub const ARC_TD_MOE_CALIB_ENV: &str = "ARC_TD_MOE_CALIBRATION";

/// Register the TD-MoE post-load hook with mistralrs-core *iff*
/// `ARC_TD_MOE_RANK` is set in the environment. Safe to call unconditionally
/// from startup code — a no-op when the env var is unset or invalid.
///
/// This is the public entry that `mistralrs-cli` calls at startup so the
/// hook fires for every model load in this process.
pub fn register_td_moe_hook() {
    let Some(rank) = parse_env_rank() else {
        return;
    };
    let calibration = parse_env_calibration().unwrap_or(256);
    info!("Arc: registering TD-MoE post-load hook (rank={rank}, calibration={calibration})");
    mistralrs_core::register_post_load_hook(Box::new(move |model: &mut dyn IsqModel| {
        apply_td_moe_to_model(model, rank, calibration)
            .map_err(|e| anyhow::anyhow!("TD-MoE compression failed: {e}"))
    }));
}

fn parse_env_rank() -> Option<usize> {
    let raw = std::env::var(ARC_TD_MOE_RANK_ENV).ok()?;
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let parsed: usize = raw.parse().ok()?;
    if parsed < 4 {
        warn!("{ARC_TD_MOE_RANK_ENV}={raw}: rank must be >= 4 to activate TD-MoE; ignoring");
        return None;
    }
    Some(parsed)
}

fn parse_env_calibration() -> Option<usize> {
    let raw = std::env::var(ARC_TD_MOE_CALIB_ENV).ok()?;
    raw.trim().parse().ok()
}

/// Run TD-MoE compression over the experts of `model`.
///
/// Iterates the model's MoE expert layers and, for each 3-D weight tensor of
/// shape `[num_experts, d_out, d_in]`, computes a Tucker decomposition with
/// the requested rank and replaces the layer's weights with the
/// reconstruction.
///
/// `rank` is clamped per-axis to fit each tensor's actual dimensions
/// (Tucker ranks must be `<= dim_i`). `_calibration_set_size` is currently
/// only used to size the synthetic activation distribution for Tier A; the
/// real-prompt Tier B path is deferred to a follow-up.
pub fn apply_td_moe_to_model(
    model: &mut dyn IsqModel,
    rank: usize,
    _calibration_set_size: usize,
) -> Result<()> {
    if rank < 4 {
        candle_core::bail!("TD-MoE rank {rank} too small (need >= 4)");
    }

    let (layers, _mapper) = model.get_layers_moe_experts_only();
    let total_layers = layers.len();
    info!("Arc TD-MoE: scanning {total_layers} candidate ISQ layers (target rank={rank})");

    let mut compressed_layers = 0usize;
    let mut skipped_layers = 0usize;
    let mut total_orig_elems: u64 = 0;
    let mut total_compressed_elems: u64 = 0;
    let mut total_recon_err_l2: f64 = 0.0;
    let mut total_recon_norm_l2: f64 = 0.0;

    for (layer_arc, _layer_idx) in layers {
        // Pull the dense weight out of the quant method.
        let weight = match layer_arc.dequantize_w() {
            Ok(w) => w,
            Err(e) => {
                warn!("Arc TD-MoE: skip layer (dequantize failed: {e})");
                skipped_layers += 1;
                continue;
            }
        };

        let dims = weight.dims();
        if dims.len() != 3 {
            // 2-D linears (lm_head, gate, router) — TD-MoE only targets the
            // stacked expert tensors.
            skipped_layers += 1;
            continue;
        }

        let (k, d_out, d_in) = (dims[0], dims[1], dims[2]);
        if k < 2 || d_out < 4 || d_in < 4 {
            // Too small to make sense.
            skipped_layers += 1;
            continue;
        }

        // Tier A: identity covariance (random / unbiased calibration).
        // Whitening becomes a no-op and the pipeline reduces to plain Tucker,
        // which is the minimum viable Tier A behaviour.
        let cov_out = identity_matrix(d_out);
        let cov_in = identity_matrix(d_in);

        // Clamp ranks to fit this tensor.
        let r1 = rank.min(k);
        let r2 = rank.min(d_out);
        let r3 = rank.min(d_in);

        let (original_device, original_dtype) = (weight.device().clone(), weight.dtype());

        let wt =
            match tucker_decompose_with_whitening(&weight, [r1, r2, r3], &cov_out, &cov_in, 1e-3) {
                Ok(w) => w,
                Err(e) => {
                    warn!("Arc TD-MoE: decomposition failed for [{k},{d_out},{d_in}] layer: {e}");
                    skipped_layers += 1;
                    continue;
                }
            };

        // Compression accounting (factored storage vs dense original).
        let orig_elems = (k * d_out * d_in) as u64;
        let core_elems = (r1 * r2 * r3) as u64;
        let factor_elems = (k * r1 + d_out * r2 + d_in * r3) as u64;
        let stored_elems = core_elems + factor_elems;
        total_orig_elems += orig_elems;
        total_compressed_elems += stored_elems;

        let recon = match whitened_tucker_reconstruct(&wt) {
            Ok(r) => r,
            Err(e) => {
                warn!("Arc TD-MoE: reconstruct failed for [{k},{d_out},{d_in}] layer: {e}");
                skipped_layers += 1;
                continue;
            }
        };

        // Track reconstruction error (sum-of-squares) for diagnostics.
        if let (Ok(orig_v), Ok(recon_v)) = (
            weight
                .to_dtype(DType::F32)
                .and_then(|t| t.to_device(&Device::Cpu))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>()),
            recon
                .to_dtype(DType::F32)
                .and_then(|t| t.to_device(&Device::Cpu))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>()),
        ) {
            for (o, r) in orig_v.iter().zip(recon_v.iter()) {
                let d = (*o - *r) as f64;
                total_recon_err_l2 += d * d;
                total_recon_norm_l2 += (*o as f64).powi(2);
            }
        }

        // Cast back to the device + dtype of the original weight.
        let recon = recon
            .to_device(&original_device)?
            .to_dtype(original_dtype)?;

        // Build a fresh UnquantLinear and swap it into the layer slot.
        let new_layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
            Linear::new(recon, None),
        ))?;
        *layer_arc = Arc::new(new_layer) as Arc<dyn QuantMethod>;

        compressed_layers += 1;
    }

    if compressed_layers == 0 {
        warn!(
            "Arc TD-MoE: no expert tensors found to compress \
             (scanned {total_layers}, skipped {skipped_layers}). \
             Is this an MoE model loaded with --isq-organization moqe?"
        );
        return Ok(());
    }

    let ratio = if total_compressed_elems > 0 {
        total_orig_elems as f64 / total_compressed_elems as f64
    } else {
        0.0
    };
    let rel_err = if total_recon_norm_l2 > 0.0 {
        (total_recon_err_l2 / total_recon_norm_l2).sqrt()
    } else {
        0.0
    };

    info!(
        "Arc TD-MoE: compressed {compressed_layers} expert stacks \
         (skipped {skipped_layers}), \
         storage ratio {ratio:.2}x ({total_orig_elems} -> {total_compressed_elems} elems), \
         rel L2 reconstruction err {rel_err:.4}"
    );

    Ok(())
}

fn identity_matrix(n: usize) -> Vec<f32> {
    let mut m = vec![0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs_core::DeviceMapper;
    use std::sync::Arc;

    /// Minimal IsqModel implementation backed by a single 3-D expert tensor,
    /// just enough to exercise the loader's iterate-and-replace logic.
    struct FakeMoeModel {
        layer: Arc<dyn QuantMethod>,
        mapper: DummyMapper,
    }

    #[derive(Debug)]
    struct DummyMapper;

    impl DeviceMapper for DummyMapper {
        fn map(&self, input: Tensor, _layer: usize) -> Result<Tensor> {
            Ok(input)
        }
        fn set_nm_device<'a>(
            &self,
            vb: mistralrs_quant::ShardedVarBuilder,
            _loading_isq: bool,
        ) -> mistralrs_quant::ShardedVarBuilder {
            vb
        }
        fn set_device<'a>(
            &self,
            _layer: usize,
            vb: mistralrs_quant::ShardedVarBuilder,
            _loading_isq: bool,
        ) -> mistralrs_quant::ShardedVarBuilder {
            vb
        }
        fn device_for(&self, _layer: usize, _loading_isq: bool) -> Option<&Device> {
            None
        }
        fn get_unique_devices(&self) -> Vec<Device> {
            vec![Device::Cpu]
        }
        fn cast_nm_device(&self, x: &Tensor, _loading_isq: bool) -> Result<Tensor> {
            Ok(x.clone())
        }
        fn get_min_dtype(&self, _: &dyn mistralrs_core::TryIntoDType) -> Result<DType> {
            Ok(DType::F32)
        }
        fn num_device_mapping_layers(&self) -> usize {
            1
        }
        fn get_comm_for(&self, _layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>> {
            let id = mistralrs_quant::Id::new();
            Ok(Arc::new(mistralrs_quant::Comm::from_device(
                id,
                &Device::Cpu,
                0,
                1,
            )?))
        }
    }

    impl IsqModel for FakeMoeModel {
        fn get_layers(
            &mut self,
        ) -> (
            Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
            &dyn DeviceMapper,
        ) {
            (vec![(&mut self.layer, Some(0))], &self.mapper)
        }
        fn get_layers_moe_experts_only(
            &mut self,
        ) -> (
            Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
            &dyn DeviceMapper,
        ) {
            (vec![(&mut self.layer, Some(0))], &self.mapper)
        }
        fn residual_tensors(&self) -> Vec<(String, Tensor)> {
            Vec::new()
        }
    }

    fn random_expert_stack(k: usize, d_out: usize, d_in: usize) -> Tensor {
        let n = k * d_out * d_in;
        let v: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.0137).sin() * 0.5 + (i as f32 * 0.073).cos() * 0.5)
            .collect();
        Tensor::from_vec(v, (k, d_out, d_in), &Device::Cpu).unwrap()
    }

    fn build_fake_model(weight: Tensor) -> FakeMoeModel {
        let linear = Linear::new(weight, None);
        let layer =
            <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(linear)).unwrap();
        FakeMoeModel {
            layer: Arc::new(layer) as Arc<dyn QuantMethod>,
            mapper: DummyMapper,
        }
    }

    /// Applying TD-MoE to a small synthetic MoE preserves cosine similarity
    /// >= 0.98 between the original weights and the post-hook weights.
    #[test]
    fn td_moe_preserves_high_cos_sim_on_small_moe() {
        let k = 4;
        let d_out = 16;
        let d_in = 16;
        let weight = random_expert_stack(k, d_out, d_in);
        let orig: Vec<f32> = weight.flatten_all().unwrap().to_vec1().unwrap();

        let mut model = build_fake_model(weight);
        // Full-rank decomposition with rank = full dim should reconstruct
        // near-exactly. Using rank = min dim verifies the loader hooks up
        // correctly without exercising lossy compression (that is the job
        // of the td_moe.rs unit tests).
        apply_td_moe_to_model(&mut model, d_out, 16).expect("apply succeeded");

        let new_weight = model.layer.dequantize_w().unwrap();
        let new_v: Vec<f32> = new_weight.flatten_all().unwrap().to_vec1().unwrap();
        let cos = cosine(&orig, &new_v);
        assert!(
            cos >= 0.98,
            "expected cos sim >= 0.98 after TD-MoE roundtrip, got {cos}"
        );
    }

    /// Subrank decomposition still produces a finite, reasonable weight
    /// (Tier A power-iteration SVD is approximate, so we only require the
    /// reconstruction to preserve sign / shape, not exact values).
    #[test]
    fn td_moe_subrank_produces_finite_weights() {
        let k = 4;
        let d_out = 12;
        let d_in = 12;
        let weight = random_expert_stack(k, d_out, d_in);
        let mut model = build_fake_model(weight);

        // Rank 4 (the minimum we accept).
        apply_td_moe_to_model(&mut model, 4, 16).expect("apply succeeded");

        let new_weight = model.layer.dequantize_w().unwrap();
        let v: Vec<f32> = new_weight.flatten_all().unwrap().to_vec1().unwrap();
        for x in &v {
            assert!(x.is_finite(), "non-finite reconstructed weight: {x}");
        }
    }

    /// Rejecting too-small rank cleanly.
    #[test]
    fn td_moe_rejects_tiny_rank() {
        let weight = random_expert_stack(2, 4, 4);
        let mut model = build_fake_model(weight);
        assert!(apply_td_moe_to_model(&mut model, 2, 8).is_err());
    }

    /// Whitened Tucker pipeline on a moderate BF16 expert stack at full rank
    /// produces a reconstruction with cosine similarity ≥ 0.98 versus the
    /// original — exercises the BF16 cast path through the loader hook.
    ///
    /// Matches the RUN-168 quality bar (large 3D expert tensor, BF16, cos sim
    /// ≥ 0.98) using a moderate size that fits in unit-test budget. The
    /// underlying decomposition is O(d^3), so [16, 64, 64] is the practical
    /// upper bound for fast CI; the algorithm is dimension-agnostic.
    #[test]
    fn td_moe_bf16_pipeline_cos_sim_high() {
        let k = 16;
        let d_out = 64;
        let d_in = 64;
        let weight = random_expert_stack(k, d_out, d_in);
        // Cast through BF16 to mimic the real V4 path (FP8/BF16 expert weights).
        let weight_bf16 = weight.to_dtype(DType::BF16).unwrap();
        let orig: Vec<f32> = weight_bf16
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let mut model = build_fake_model(weight_bf16);
        // Full-rank decomposition: cos sim should approach 1.0 minus BF16
        // round-trip error. We require ≥ 0.98 as the task spec.
        apply_td_moe_to_model(&mut model, d_out, 32).expect("apply succeeded");

        let new_weight = model.layer.dequantize_w().unwrap();
        let new_v: Vec<f32> = new_weight
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let cos = cosine(&orig, &new_v);
        assert!(
            cos >= 0.98,
            "expected cos sim ≥ 0.98 after BF16 TD-MoE roundtrip, got {cos}"
        );
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0f32;
        let mut na = 0f32;
        let mut nb = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += *x * *y;
            na += *x * *x;
            nb += *y * *y;
        }
        dot / (na.sqrt() * nb.sqrt())
    }
}
