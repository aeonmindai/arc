#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
//! DeepSeek V4 Lightning Indexer (`C4Indexer`).
//!
//! Pure-Rust port of SGLang's
//! `python/sglang/srt/layers/attention/dsv4/indexer.py::C4Indexer`
//! (lines ~469-558) — the per-query top-k selector used on V4 CSA layers
//! (compress_ratio = 4) to replace dense-over-compressed attention with
//! sparse attention over the `topk` most-relevant compressed K entries.
//!
//! ## What it does
//!
//! For V4 CSA layers, given the main attention's pre-`q_b` LoRA-A output
//! `q_a: [B, T_q, q_lora_rank]` and the full K tensor
//! `k_full: [B, n_heads, T_full, head_dim]`, the indexer:
//!   1. Projects `q_a` through `wq_b` to produce `indexer_q` (one query per
//!      indexer head, distinct from main-attention heads).
//!   2. Runs an internal smaller compressor over `k_full` to produce
//!      `indexer_k: [B, n_heads, T_c, head_dim]` (with `T_c = T_full / 4`).
//!   3. Computes per-head dot-product scores `indexer_q @ indexer_k^T`.
//!   4. Scales each head by a learned per-token `weights_proj(x)` factor.
//!   5. Returns the `topk` indices into `T_c` per (batch, head, query).
//!
//! The downstream caller (V4 attention) then gathers only those `topk`
//! compressed-K entries for the actual sparse attention reduction.
//!
//! ## Tensor name auto-detection
//!
//! V4 publishes the indexer at:
//!   - **Native** (DeepSeek): `layers.<i>.attn.indexer.{wq_b, weights_proj,
//!     compressor.{ape, norm, wgate, wkv}}`
//!   - **HF transformers**: `model.layers.<i>.self_attn.indexer.{...}` (with
//!     `compressor.ape.weight` instead of bare `compressor.ape`).
//!
//! The caller is responsible for passing the right `vb.pp("indexer")`
//! prefix; this module only handles the leaf tensor name probe for
//! `ape` (which lacks a `.weight` suffix in native format).
//!
//! ## Phase-2 perf
//!
//! This is a pure-Rust/candle reference. Phase-2 (TileLang fused kernel) will
//! replace the per-step matmul + topk with a single fused
//! `fp8_paged_mqa_logits + topk_transform_512` launch per layer (see SGLang's
//! `dsv4/tilelang_kernel.py`). The interface here is stable; the kernel
//! swap is internal.
//!
//! Reference: SGLang `C4Indexer.__init__` and `compute_q` / `compute_weights`
//! (indexer.py lines 469-538), plus `Compressor.__init__` and `forward`
//! (compressor.py lines 289-392) for the inner compressor structure.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::Module;
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use crate::{
    layers::RmsNorm,
    ops::{TopKLastDimOp, TopKOutput},
};

/// Pure-Rust V4 Lightning Indexer.
///
/// See module-level docs. Fields are `pub` so a downstream
/// integrator (Agent 3 wiring into `Attention::forward_attention`) can
/// inspect them, but the only stable forward path is `forward`.
pub struct V4Indexer {
    /// `q_lora_rank` → `n_heads * head_dim`. Projects the main attention's
    /// LoRA-A Q output into per-indexer-head queries.
    pub wq_b: Arc<dyn QuantMethod>,
    /// `hidden_size` → `n_heads`. Per-token per-head score weighting.
    pub weights_proj: Arc<dyn QuantMethod>,
    /// Inner compressor's absolute-positional-encoding parameter.
    ///
    /// Shape: `[ratio=4, coff*head_dim]`. Added (broadcast) to the gated
    /// compressed-K activations. In SGLang this is also reshaped via
    /// `apply_ape_hotfix` on overlap layers; we preserve the raw shape
    /// here and sum over the ratio dim for the prototype broadcast.
    pub inner_ape: Tensor,
    /// Inner compressor RMSNorm applied to each `head_dim` slice after
    /// gating + ape.
    pub inner_norm: RmsNorm,
    /// Inner-compressor gate linear: `ratio*head_dim` → `coff*head_dim`.
    /// Used as `sigmoid(gate(x_grouped))`.
    pub inner_wgate: Arc<dyn QuantMethod>,
    /// Inner-compressor value linear: `ratio*head_dim` → `coff*head_dim`.
    /// Used as `gate * kv(x_grouped)`.
    pub inner_wkv: Arc<dyn QuantMethod>,
    /// Number of indexer heads (default 64 per V4 Flash).
    pub n_heads: usize,
    /// Per-head dim (default 128 per V4 Flash).
    pub head_dim: usize,
    /// CSA top-k (default 512). Number of compressed K entries to retain.
    pub topk: usize,
    /// Compression ratio used by the inner compressor. Fixed at 4 for the
    /// V4 Lightning Indexer (CSA).
    pub ratio: usize,
    /// Overlap-mode coefficient. `coff = 1 + overlap`. For ratio==4 (CSA),
    /// `overlap = True` and `coff = 2`. See SGLang `Compressor.__init__`.
    pub coff: usize,
}

impl V4Indexer {
    /// Build an indexer from the safetensors at the caller-provided
    /// `vb` prefix (typically `outer_vb.pp("indexer")`).
    pub fn new(
        cfg: &super::deepseek4::DeepSeekV4Config,
        vb: ShardedVarBuilder,
        device: &Device,
        loading_isq: bool,
    ) -> Result<Self> {
        let _ = device;
        let _ = loading_isq;
        let n_heads = cfg.index_n_heads;
        let head_dim = cfg.index_head_dim;
        let topk = cfg.index_topk;
        let q_lora_rank = cfg
            .q_lora_rank
            .ok_or_else(|| candle_core::Error::Msg(
                "V4Indexer requires DeepSeekV4Config.q_lora_rank to be set".into(),
            ))?;

        // CSA always uses ratio=4 with overlap=true → coff=2.
        let ratio = 4usize;
        let coff = 2usize;

        // --- Outer indexer tensors (wq_b, weights_proj) ----------------
        // Both are unquantized (`quant_config=None` in SGLang) per the
        // `ReplicatedLinear` constructors at lines 490-505 of indexer.py.
        // We still go through ReplicatedLayer::new so that ISQ / quant
        // overrides can wrap them transparently if requested.
        let wq_b = ReplicatedLayer::new(
            q_lora_rank,
            n_heads * head_dim,
            &cfg.quantization_config,
            false,
            vb.pp("wq_b"),
        )?;
        let weights_proj = ReplicatedLayer::new(
            cfg.hidden_size,
            n_heads,
            &None,
            false,
            vb.pp("weights_proj"),
        )?;

        // --- Inner compressor (vb.pp("compressor")) --------------------
        // Native checkpoints store ape as a bare parameter ("compressor.ape");
        // HF checkpoints store it as "compressor.ape.weight". Probe both.
        let comp_vb = vb.pp("compressor");

        let inner_ape = if comp_vb.contains_tensor("ape") {
            comp_vb.get((ratio, coff * head_dim), "ape")?
        } else if comp_vb.contains_tensor("ape.weight") {
            comp_vb.get((ratio, coff * head_dim), "ape.weight")?
        } else {
            // Synthetic-weights fallback: zeros (no positional contribution).
            // Real V4 checkpoints publish this tensor; unit tests construct
            // it explicitly via `from_synth` (see tests below).
            Tensor::zeros((ratio, coff * head_dim), DType::F32, comp_vb.device())?
        };

        let inner_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, comp_vb.pp("norm"))?;

        // Inner gate/kv linears: `ratio * head_dim` → `coff * head_dim`.
        //
        // NB: SGLang publishes a single fused `wkv_gate` (`2 * coff *
        // head_dim` out_features) in some checkpoints but the V4 weight
        // schema (arc-engine `weight_schema::v4_indexer_tensors_for_layer`)
        // shows two separate tensors `compressor.wgate.weight` and
        // `compressor.wkv.weight`. We follow the schema.
        let inner_wgate = ReplicatedLayer::new(
            ratio * head_dim,
            coff * head_dim,
            &None,
            false,
            comp_vb.pp("wgate"),
        )?;
        let inner_wkv = ReplicatedLayer::new(
            ratio * head_dim,
            coff * head_dim,
            &None,
            false,
            comp_vb.pp("wkv"),
        )?;

        Ok(Self {
            wq_b,
            weights_proj,
            inner_ape,
            inner_norm,
            inner_wgate,
            inner_wkv,
            n_heads,
            head_dim,
            topk,
            ratio,
            coff,
        })
    }

    /// Score queries against compressed K, return top-k indices.
    ///
    /// Inputs:
    ///   - `q_a`: `[B, T_q, q_lora_rank]` — output of main Q LoRA-A.
    ///   - `k_full`: `[B, n_heads, T_full, head_dim]` — full K tensor.
    ///     `T_full` must be divisible by `ratio` (= 4).
    ///   - `xs`: `[B, T_q, hidden_size]` — original input for `weights_proj`.
    ///
    /// Output:
    ///   - `top_k_indices`: `[B, n_heads, T_q, topk]` indices into
    ///     `T_c = T_full / ratio`.
    ///
    /// SGLang reference: `C4Indexer.compute_q`, `compute_weights`,
    /// and the final scoring step (folded into the fused triton kernel
    /// `fp8_paged_mqa_logits`). This is the unfused candle equivalent.
    pub fn forward(&self, q_a: &Tensor, k_full: &Tensor, xs: &Tensor) -> Result<Tensor> {
        let q_dims = q_a.dims();
        if q_dims.len() != 3 {
            candle_core::bail!("V4Indexer.q_a expected [B, T_q, q_lora_rank], got {:?}", q_dims);
        }
        let (b, t_q, _q_rank) = (q_dims[0], q_dims[1], q_dims[2]);

        let k_dims = k_full.dims();
        if k_dims.len() != 4 {
            candle_core::bail!("V4Indexer.k_full expected [B, H, T_full, D], got {:?}", k_dims);
        }
        let (b_k, h_k, t_full, d_k) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
        if b_k != b {
            candle_core::bail!(
                "V4Indexer: q_a batch {} != k_full batch {}",
                b,
                b_k
            );
        }
        if h_k != self.n_heads {
            candle_core::bail!(
                "V4Indexer: k_full heads {} != n_heads {}",
                h_k,
                self.n_heads
            );
        }
        if d_k != self.head_dim {
            candle_core::bail!(
                "V4Indexer: k_full head_dim {} != head_dim {}",
                d_k,
                self.head_dim
            );
        }
        if t_full % self.ratio != 0 {
            candle_core::bail!(
                "V4Indexer: k_full T_full ({}) must be divisible by ratio ({})",
                t_full,
                self.ratio
            );
        }
        let t_c = t_full / self.ratio;

        let xs_dims = xs.dims();
        if xs_dims.len() != 3 || xs_dims[0] != b || xs_dims[1] != t_q {
            candle_core::bail!(
                "V4Indexer.xs expected [B={}, T_q={}, hidden], got {:?}",
                b,
                t_q,
                xs_dims
            );
        }

        // --- 1. Project query: indexer_q = wq_b(q_a) -------------------
        // [B, T_q, q_lora_rank] → [B, T_q, n_heads*head_dim]
        //                      → [B, T_q, n_heads, head_dim]
        //                      → [B, n_heads, T_q, head_dim]
        let q = self.wq_b.forward_autocast(q_a)?;
        let q = q
            .reshape((b, t_q, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // --- 2. Inner compressor on K ----------------------------------
        // k_full [B, n_heads, T_full, head_dim] →
        //   group along T_full into windows of `ratio`:
        //     [B, n_heads, T_c, ratio*head_dim] →
        //     [B*n_heads*T_c, ratio*head_dim]   (x_grouped)
        let k_grouped = k_full
            .reshape((b, h_k, t_c, self.ratio * self.head_dim))?
            .reshape((b * h_k * t_c, self.ratio * self.head_dim))?;
        let work_dtype = k_grouped.dtype();

        // Linear projections.
        let gate_out = self.inner_wgate.forward_autocast(&k_grouped)?;
        let kv_out = self.inner_wkv.forward_autocast(&k_grouped)?;
        let gated = (candle_nn::ops::sigmoid(&gate_out)? * kv_out)?;

        // Add ape positional bias. inner_ape is [ratio, coff*head_dim].
        // For the prototype we sum over the ratio axis to produce a
        // single broadcastable [coff*head_dim] bias. Phase-2 TileLang
        // kernel will reproduce SGLang's per-position-in-group ape
        // ordering. The bias shape [coff*head_dim] matches gated's last
        // dim and broadcasts correctly across all grouped rows.
        let ape_bias = self
            .inner_ape
            .to_dtype(work_dtype)?
            .sum(0)?; // [coff*head_dim]
        let gated = gated.broadcast_add(&ape_bias)?;

        // Reshape to expose `head_dim` for RMSNorm. SGLang's compressor
        // norm operates per-`head_dim` slice. With coff=2 we have two
        // such slices per output position. Treat them as additional
        // (B*H*T_c)*coff rows of head_dim each.
        let total_groups = b * h_k * t_c;
        let gated = gated.reshape((total_groups * self.coff, self.head_dim))?;
        let normed = gated.apply(&self.inner_norm)?;

        // We need a single [head_dim] output per compressed slot — collapse
        // the `coff` overlap dimension by sum (matches SGLang's overlap
        // handling: the two halves are summed at the kernel level when
        // computing scores; see `compress_forward` triton kernel).
        let normed = normed.reshape((total_groups, self.coff, self.head_dim))?.sum(1)?;
        let indexer_k = normed
            .reshape((b, h_k, t_c, self.head_dim))?
            .contiguous()?;

        // --- 3. Dot-product scores: q @ k^T ----------------------------
        // q:        [B, n_heads, T_q, head_dim]
        // k^T:      [B, n_heads, head_dim, T_c]
        // scores:   [B, n_heads, T_q, T_c]
        let k_t = indexer_k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scores = q.matmul(&k_t)?;

        // --- 4. Per-head scaling from weights_proj(x) ------------------
        // weights_proj: [B, T_q, hidden] → [B, T_q, n_heads]
        //                              → [B, n_heads, T_q, 1] (for broadcast)
        let per_head_scale = self
            .weights_proj
            .forward_autocast(xs)? // [B, T_q, n_heads]
            .transpose(1, 2)? // [B, n_heads, T_q]
            .unsqueeze(D::Minus1)? // [B, n_heads, T_q, 1]
            .contiguous()?;
        let scaled = scores.broadcast_mul(&per_head_scale.to_dtype(scores.dtype())?)?;

        // --- 5. Top-k along T_c (last dim) -----------------------------
        // Returns indices into [0, T_c).
        let scaled = scaled.contiguous()?;
        let k = self.topk.min(t_c);
        let TopKOutput { indices, .. } = scaled.topk_unsorted(k)?;
        Ok(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::var_builder::VarBuilderArgs;
    use mistralrs_quant::ShardedSafeTensors;
    use std::collections::HashMap;

    /// Build a minimal `DeepSeekV4Config` for testing. Only the fields the
    /// indexer reads from are populated; the rest are zeros / defaults.
    fn synth_cfg(
        hidden_size: usize,
        q_lora_rank: usize,
        n_heads: usize,
        head_dim: usize,
        topk: usize,
    ) -> super::super::deepseek4::DeepSeekV4Config {
        // Use serde_json to avoid having to fill every field by hand.
        let json = serde_json::json!({
            "vocab_size": 32,
            "hidden_size": hidden_size,
            "intermediate_size": 64,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "n_shared_experts": null,
            "n_routed_experts": null,
            "num_experts_per_tok": null,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rope_scaling": null,
            "q_lora_rank": q_lora_rank,
            "qk_rope_head_dim": 16,
            "kv_lora_rank": 32,
            "v_head_dim": 16,
            "qk_nope_head_dim": 16,
            "n_group": 1,
            "topk_group": 1,
            "compress_ratios": [4],
            "index_n_heads": n_heads,
            "index_head_dim": head_dim,
            "index_topk": topk,
        });
        serde_json::from_value(json).expect("synth config deserialize")
    }

    /// Build a `ShardedVarBuilder` from a `HashMap<String, Tensor>` for
    /// hermetic unit-test construction of `V4Indexer`. The HashMap
    /// `SimpleBackend` impl comes from candle_nn.
    fn vb_from_map(map: HashMap<String, Tensor>, dtype: DType, dev: &Device) -> ShardedVarBuilder {
        // candle_nn::var_builder::Backend is impl'd on HashMap<String, Tensor>
        // as SimpleBackend, so we can wrap it with ShardedSafeTensors::wrap.
        let backend: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(map);
        let _ = VarBuilderArgs::<'static, ShardedSafeTensors>::new_with_args; // ensure use
        ShardedSafeTensors::wrap(backend, dtype, dev.clone())
    }

    fn make_indexer_tensors(
        cfg: &super::super::deepseek4::DeepSeekV4Config,
        ratio: usize,
        coff: usize,
        device: &Device,
    ) -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        let dtype = DType::F32;

        // wq_b.weight: [n_heads*head_dim, q_lora_rank]
        m.insert(
            "wq_b.weight".to_string(),
            Tensor::randn(
                0.0f32,
                0.02,
                (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank.unwrap()),
                device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap(),
        );
        // weights_proj.weight: [n_heads, hidden_size]
        m.insert(
            "weights_proj.weight".to_string(),
            Tensor::randn(0.0f32, 0.02, (cfg.index_n_heads, cfg.hidden_size), device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap(),
        );
        // compressor.ape: [ratio, coff*head_dim]
        m.insert(
            "compressor.ape".to_string(),
            Tensor::zeros((ratio, coff * cfg.index_head_dim), dtype, device).unwrap(),
        );
        // compressor.norm.weight: [head_dim]
        m.insert(
            "compressor.norm.weight".to_string(),
            Tensor::ones(cfg.index_head_dim, dtype, device).unwrap(),
        );
        // compressor.wgate.weight: [coff*head_dim, ratio*head_dim]
        m.insert(
            "compressor.wgate.weight".to_string(),
            Tensor::randn(
                0.0f32,
                0.02,
                (coff * cfg.index_head_dim, ratio * cfg.index_head_dim),
                device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap(),
        );
        // compressor.wkv.weight: [coff*head_dim, ratio*head_dim]
        m.insert(
            "compressor.wkv.weight".to_string(),
            Tensor::randn(
                0.0f32,
                0.02,
                (coff * cfg.index_head_dim, ratio * cfg.index_head_dim),
                device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap(),
        );
        m
    }

    /// Forward produces output shape [B, n_heads, T_q, topk].
    #[test]
    fn v4_indexer_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let cfg = synth_cfg(/*hidden*/ 32, /*q_lora*/ 24, /*n_heads*/ 4, /*head_dim*/ 8, /*topk*/ 6);
        let tensors = make_indexer_tensors(&cfg, 4, 2, &device);
        let vb = vb_from_map(tensors, DType::F32, &device);
        let indexer = V4Indexer::new(&cfg, vb, &device, false)?;

        let b = 2usize;
        let t_q = 3usize;
        let t_full = 16usize; // T_c = 4
        let q_a = Tensor::randn(0.0f32, 1.0, (b, t_q, cfg.q_lora_rank.unwrap()), &device)?;
        let k_full = Tensor::randn(
            0.0f32,
            1.0,
            (b, cfg.index_n_heads, t_full, cfg.index_head_dim),
            &device,
        )?;
        let xs = Tensor::randn(0.0f32, 1.0, (b, t_q, cfg.hidden_size), &device)?;

        let out = indexer.forward(&q_a, &k_full, &xs)?;
        // Output shape should be [B, n_heads, T_q, topk] where topk
        // is clamped to T_c = 4 (since synth_cfg uses topk=6 > T_c=4).
        let t_c = t_full / 4;
        let expected_k = cfg.index_topk.min(t_c);
        assert_eq!(out.dims(), &[b, cfg.index_n_heads, t_q, expected_k]);
        Ok(())
    }

    /// All returned indices are in [0, T_c).
    #[test]
    fn v4_indexer_indices_in_range() -> Result<()> {
        let device = Device::Cpu;
        let cfg = synth_cfg(32, 24, 4, 8, /*topk*/ 3);
        let tensors = make_indexer_tensors(&cfg, 4, 2, &device);
        let vb = vb_from_map(tensors, DType::F32, &device);
        let indexer = V4Indexer::new(&cfg, vb, &device, false)?;

        let b = 1usize;
        let t_q = 2usize;
        let t_full = 12usize; // T_c = 3
        let q_a = Tensor::randn(0.0f32, 1.0, (b, t_q, cfg.q_lora_rank.unwrap()), &device)?;
        let k_full = Tensor::randn(
            0.0f32,
            1.0,
            (b, cfg.index_n_heads, t_full, cfg.index_head_dim),
            &device,
        )?;
        let xs = Tensor::randn(0.0f32, 1.0, (b, t_q, cfg.hidden_size), &device)?;

        let out = indexer.forward(&q_a, &k_full, &xs)?;
        let t_c = (t_full / 4) as u32;
        // Flatten to Vec<u32> for in-range check.
        let flat: Vec<u32> = out.flatten_all()?.to_vec1()?;
        for idx in &flat {
            assert!(
                *idx < t_c,
                "index {idx} out of range; T_c = {t_c}"
            );
        }
        Ok(())
    }

    /// Output is deterministic across re-runs (no in-forward RNG).
    #[test]
    fn v4_indexer_deterministic() -> Result<()> {
        let device = Device::Cpu;
        let cfg = synth_cfg(16, 12, 2, 4, /*topk*/ 4);
        let tensors = make_indexer_tensors(&cfg, 4, 2, &device);
        let vb = vb_from_map(tensors, DType::F32, &device);
        let indexer = V4Indexer::new(&cfg, vb, &device, false)?;

        let b = 1usize;
        let t_q = 2usize;
        let t_full = 16usize;
        // Use deterministic (non-randn) inputs for value comparison.
        let q_a = Tensor::arange(0u32, (b * t_q * cfg.q_lora_rank.unwrap()) as u32, &device)?
            .reshape((b, t_q, cfg.q_lora_rank.unwrap()))?
            .to_dtype(DType::F32)?;
        let k_full = Tensor::arange(
            0u32,
            (b * cfg.index_n_heads * t_full * cfg.index_head_dim) as u32,
            &device,
        )?
        .reshape((b, cfg.index_n_heads, t_full, cfg.index_head_dim))?
        .to_dtype(DType::F32)?;
        let xs = Tensor::arange(0u32, (b * t_q * cfg.hidden_size) as u32, &device)?
            .reshape((b, t_q, cfg.hidden_size))?
            .to_dtype(DType::F32)?;

        let out_a: Vec<u32> = indexer
            .forward(&q_a, &k_full, &xs)?
            .flatten_all()?
            .to_vec1()?;
        let out_b: Vec<u32> = indexer
            .forward(&q_a, &k_full, &xs)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(out_a, out_b, "indexer must be deterministic across re-runs");
        Ok(())
    }
}
