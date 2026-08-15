#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
// WIP V4 indexer module: the Phase-1 reference `forward` and some config fields
// are not yet wired into the active dispatch path. See arc-tools/CI_HYGIENE.md.
#![allow(dead_code)]
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
//! `q_a: [B, T_q, q_lora_rank]` and the layer's hidden states
//! `xs_full: [B, T_full, hidden_size]`, the indexer:
//!   1. Projects `q_a` through `wq_b` to produce `indexer_q` (one query per
//!      indexer head, distinct from main-attention heads).
//!   2. Runs an internal smaller compressor **over the hidden states** to
//!      produce `indexer_k: [B, T_c, head_dim]` (with `T_c = T_full / 4`) —
//!      **one key per compressed slot, shared by every head (MQA)**.
//!   3. Computes per-head dot-product scores `indexer_q @ indexer_k^T`,
//!      **rectifies** them (`relu`), scales each head by the learned per-token
//!      `weights_proj(x) * weight_scale` factor, and **sums over heads** to
//!      one logit per `(batch, query, compressed-key)`.
//!   4. Returns the `topk` indices into `T_c` per (batch, query) — a **single
//!      key set shared by every head**.
//!
//! The downstream caller (V4 attention) then gathers only those `topk`
//! compressed-K entries for the actual sparse attention reduction.
//!
//! ## Scoring formula (the authoritative one)
//!
//! ```text
//! logit[b, q, c] = kv_scale[c] * Σ_h relu(q[b,h,q,:] · k[b,h,c,:]) * w[b,h,q]
//! w[b,h,q]       = weights_proj(x)[b,q,h] * weight_scale
//! weight_scale   = softmax_scale * n_heads^-0.5,  softmax_scale = head_dim^-0.5
//! topk_idx[b,q]  = argtopk_c(logit[b, q, :])          // one selection, all heads
//! ```
//!
//! This mirrors SGLang's reference implementation of the DSV4 indexer logits,
//! `dsv4/indexer.py::fp8_paged_mqa_logits_torch` (lines 84-89):
//!
//! ```python
//! score = F.linear(kvcache_value, q)   # [T_c, n_heads]  per-head q·k
//! score = F.relu(score)                # (1) rectify BEFORE mixing heads
//! score *= q_scale[None, :]            # per-head weight (weight_scale folded in)
//! score = score.sum(dim=1)             # (2) ONE logit per key
//! score *= kvcache_scale               # (3) per-key FP8 dequant scale
//! ```
//!
//! and `C4Indexer.__init__` line 519 for (3'):
//! `self.weight_scale = self.softmax_scale * self.n_heads**-0.5`, folded into
//! the weights by `compute_weights` / `fused_scale` before the logits kernel.
//!
//! Three properties this buys over a plain per-head `(q·k) * w` + per-head
//! top-k:
//!   1. **ReLU**: without it, a strongly *negative* head dot cancels a strongly
//!      positive one and a different key wins. Selection is not invariant.
//!   2. **Sum over heads → one shared selection**: per-head top-k would emit
//!      `n_heads` (= 64) different key sets, multiplying the sparse-gather
//!      traffic by 64 and defeating the point of the sparse path. The shared
//!      page set is what makes the gather cheap.
//!   3. **`weight_scale`**: a uniform positive factor, so selection-neutral,
//!      but required if the logits are ever consumed directly (debug dumps,
//!      thresholding, calibration).
//!
//! `kv_scale[c]` is the per-key FP8 dequantisation scale of SGLang's paged
//! indexer K cache. Arc keeps the indexer K in BF16/F32 (no FP8 indexer cache
//! yet), so `kv_scale ≡ 1` here and the term is absent; it reappears when the
//! FP8 paged cache lands.
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
//! prefix; the leaf-name probes (`ape` vs `ape.weight`, split `wkv`+`wgate`
//! vs pre-fused `wkv_gate`) live in [`V4Compressor`].
//!
//! ## ⚠️ The inner compressor consumes HIDDEN STATES, not K
//!
//! `C4Indexer` builds its inner compressor from the **same** `Compressor`
//! class the attention path uses, with `head_dim = index_head_dim` and
//! `is_in_indexer = True` (`indexer.py:506-516`), and that class takes the
//! layer input `x` — `wkv_gate = ReplicatedLinear(config.hidden_size,
//! 2 * coff * head_dim)` (`compressor.py:305, 318-325`). So the published
//! weights are `[coff*head_dim, hidden_size]` per half
//! (`deepseek_v4.py:1633-1652` concatenates `cat([wkv, wgate], dim=0)`).
//!
//! This module previously declared them `[coff*head_dim, ratio*head_dim]` and
//! fed the compressor grouped **K** instead. Consequence: the indexer failed
//! to load from Arc's own published UQFF on **every** CSA layer
//! (`shape mismatch for layers.N.attn.indexer.compressor.wgate.weight,
//! expected: [256, 512], got: [256, 4096]`) and silently fell back to
//! dense-over-compressed. Arc's attention-side compressor had already been
//! corrected for exactly this (RUN-161/162); the indexer kept a stale private
//! copy. It now delegates to the shared [`V4Compressor`] so there is one
//! implementation, not two.
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

use candle_core::{Device, Result, Tensor, D};
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use crate::ops::{TopKLastDimOp, TopKOutput};

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
    /// Inner compressor — the **same** `Compressor` module the attention path
    /// uses, instantiated with `head_dim = index_head_dim` and `ratio = 4`
    /// (SGLang `C4Indexer.__init__`, `indexer.py:506-516`). It consumes the
    /// layer's hidden states and emits one `head_dim` key per compressed slot.
    pub compressor: super::deepseek4::V4Compressor,
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
    /// `softmax_scale * n_heads^-0.5` with `softmax_scale = head_dim^-0.5`.
    ///
    /// SGLang `C4Indexer.__init__` line 519. Folded into the per-head weights
    /// (`compute_weights` / `fused_scale`) before the logits kernel runs, so
    /// the kernel and the CPU reference both receive weights that already
    /// carry it. Uniform and positive → selection-neutral, but the logits are
    /// only numerically correct with it.
    pub weight_scale: f64,
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
        // Delegated to the shared `V4Compressor`, which is the corrected
        // (RUN-161/162) port of SGLang's `Compressor`: it takes the layer's
        // hidden states, loads `wkv`/`wgate` at `[coff*head_dim,
        // hidden_size]` (or a pre-fused `wkv_gate` at `[2*coff*head_dim,
        // hidden_size]`), probes both `ape` spellings, and pools with a
        // softmax over the (overlapped) group axis.
        //
        // `head_dim` here is `index_head_dim`, NOT the attention head_dim —
        // `indexer.py:512` passes `head_dim=self.head_dim` where
        // `self.head_dim = config.index_head_dim`.
        let comp_vb = vb.pp("compressor");
        if !super::deepseek4::V4Compressor::has_weights(&comp_vb) {
            candle_core::bail!(
                "V4Indexer: indexer compressor weights are absent — expected either \
                 `compressor.wkv.weight` + `compressor.wgate.weight` (V4 native) or \
                 a pre-fused `compressor.wkv_gate.weight`"
            );
        }
        let compressor =
            super::deepseek4::V4Compressor::new(cfg, comp_vb, ratio, head_dim, device)?;

        // SGLang C4Indexer.__init__:
        //   self.softmax_scale = self.head_dim ** -0.5
        //   self.weight_scale  = self.softmax_scale * self.n_heads ** -0.5
        let weight_scale = (head_dim as f64).powf(-0.5) * (n_heads as f64).powf(-0.5);

        Ok(Self {
            wq_b,
            weights_proj,
            compressor,
            n_heads,
            head_dim,
            topk,
            ratio,
            coff,
            weight_scale,
        })
    }

    /// Score queries against compressed K, return top-k indices.
    ///
    /// Inputs:
    ///   - `q_a`: `[B, T_q, q_lora_rank]` — output of main Q LoRA-A.
    ///   - `xs_full`: `[B, T_full, hidden_size]` — hidden states of **every**
    ///     token the compressed keys are built from. `T_full` must be
    ///     divisible by `ratio` (= 4). This is the compressor's input in the
    ///     reference (`indexer.py:299-304` passes `x=x`), not K.
    ///   - `xs_q`: `[B, T_q, hidden_size]` — hidden states of the *query*
    ///     tokens, for `weights_proj`. In prefill this is the same tensor as
    ///     `xs_full`.
    ///
    /// Output:
    ///   - `top_k_indices`: `[B, T_q, topk]` indices into
    ///     `T_c = T_full / ratio`. **One key set per (batch, query), shared by
    ///     every head** — see the module-level "Scoring formula" section.
    ///
    /// SGLang reference: `C4Indexer.compute_q`, `compute_weights`,
    /// and the final scoring step (`fp8_paged_mqa_logits`, whose torch
    /// reference is `dsv4/indexer.py::fp8_paged_mqa_logits_torch` lines
    /// 84-89). This is the unfused candle equivalent.
    pub fn forward(&self, q_a: &Tensor, xs_full: &Tensor, xs_q: &Tensor) -> Result<Tensor> {
        let q_dims = q_a.dims();
        if q_dims.len() != 3 {
            candle_core::bail!("V4Indexer.q_a expected [B, T_q, q_lora_rank], got {:?}", q_dims);
        }
        let (b, t_q, _q_rank) = (q_dims[0], q_dims[1], q_dims[2]);

        let full_dims = xs_full.dims();
        if full_dims.len() != 3 || full_dims[0] != b {
            candle_core::bail!(
                "V4Indexer.xs_full expected [B={b}, T_full, hidden], got {full_dims:?}"
            );
        }
        let t_full = full_dims[1];
        if t_full % self.ratio != 0 {
            candle_core::bail!(
                "V4Indexer: T_full ({}) must be divisible by ratio ({})",
                t_full,
                self.ratio
            );
        }
        let t_c = t_full / self.ratio;

        let xs_dims = xs_q.dims();
        if xs_dims.len() != 3 || xs_dims[0] != b || xs_dims[1] != t_q {
            candle_core::bail!(
                "V4Indexer.xs_q expected [B={}, T_q={}, hidden], got {:?}",
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

        // --- 2. Inner compressor on the HIDDEN STATES ------------------
        // `[B, T_full, hidden]` → `[B, T_c, head_dim]`: ONE key per
        // compressed slot, shared by every indexer head (SGLang's indexer K
        // cache is MQA, `num_heads_kv = 1`).
        let indexer_k_mqa = self.compressor.forward_from_xs(xs_full)?;

        // The scoring path below (and the CUDA FlashMLASparse kernel) takes
        // `[B, H, T_c, D]`, so broadcast the single key set across heads.
        // This is exact — every head sees the identical key — but it does
        // materialise `n_heads` copies. Collapsing it into a broadcasting
        // matmul is a memory/compute win only; it does not change results.
        let indexer_k = indexer_k_mqa
            .unsqueeze(1)?
            .expand((b, self.n_heads, t_c, self.head_dim))?
            .contiguous()?;

        // --- 3+4+5. Score + relu + weighted head-sum + top-k -------------
        // CUDA fast path (RUN-163): when running on a CUDA device with a
        // supported (topk, head_dim) combo, dispatch to the vendored
        // FlashMLASparse kernels for fused score + radix top-k. The pure
        // Rust path below remains the spec / reference and is also taken
        // on CPU / non-CUDA devices and for unsupported shapes.
        //
        // Per-head weights: weights_proj(xs_q) * weight_scale
        //   → [B, T_q, n_heads] → [B, n_heads, T_q]
        // `weight_scale` is folded in here (not in the kernel), exactly as
        // SGLang folds it in `compute_weights` / `fused_scale` before the
        // logits kernel. The kernel and `cpu_reference` therefore both take
        // weights that already carry it.
        let per_head_scale_3d = self
            .weights_proj
            .forward_autocast(xs_q)? // [B, T_q, n_heads]
            .transpose(1, 2)? // [B, n_heads, T_q]
            .contiguous()?
            .affine(self.weight_scale, 0.0)?;

        #[cfg(feature = "cuda")]
        {
            if matches!(q.device(), candle_core::Device::Cuda(_))
                && q.dtype() == candle_core::DType::BF16
                && arc_cuda_graph::flashmlasparse::SUPPORTED_TOPK.contains(&self.topk.min(t_c))
            {
                // Ensure all three inputs are BF16 contiguous on the same device.
                let q_bf16 = q.to_dtype(candle_core::DType::BF16)?.contiguous()?;
                let k_bf16 = indexer_k.to_dtype(candle_core::DType::BF16)?.contiguous()?;
                let s_bf16 = per_head_scale_3d
                    .to_dtype(candle_core::DType::BF16)?
                    .contiguous()?;
                let topk = self.topk.min(t_c);
                // Returns [B, T_q, topk] — the head dim is reduced inside the
                // kernel, matching `indexer_logits` below.
                let out = arc_cuda_graph::flashmlasparse::score_and_topk_bf16(
                    &q_bf16, &k_bf16, &s_bf16, topk,
                )?;
                // Output dtype matches Candle's `topk_unsorted` (U32).
                return Ok(out);
            }
        }

        // --- Pure-Rust reference path ----------------------------------
        let logits = indexer_logits(&q, &indexer_k, &per_head_scale_3d)?;

        // Top-k along T_c (last dim). Returns indices into [0, T_c).
        let k = self.topk.min(t_c);
        let TopKOutput { indices, .. } = logits.contiguous()?.topk_unsorted(k)?;
        Ok(indices)
    }
}

/// The V4 Lightning Indexer logit reduction — the piece the CUDA kernel
/// (`arc_flashmlasparse_logits_*`) and `arc_cuda_graph::flashmlasparse::
/// cpu_reference` must agree with bit-for-bit in semantics.
///
/// ```text
/// logit[b, q, c] = Σ_h relu(q[b,h,q,:] · k[b,h,c,:]) * w[b,h,q]
/// ```
///
/// Inputs:
///   - `q`: `[B, n_heads, T_q, head_dim]`
///   - `k`: `[B, n_heads, T_c, head_dim]`
///   - `weights`: `[B, n_heads, T_q]` — `weights_proj(x) * weight_scale`,
///     i.e. `weight_scale` is already folded in by the caller (SGLang does
///     the same in `compute_weights`).
///
/// Output: `[B, T_q, T_c]` — **one logit per (query, key)**, no head axis.
///
/// Reference: SGLang `dsv4/indexer.py::fp8_paged_mqa_logits_torch` lines
/// 84-89 (`F.linear` → `F.relu` → `*= q_scale` → `.sum(dim=1)`).
pub fn indexer_logits(q: &Tensor, k: &Tensor, weights: &Tensor) -> Result<Tensor> {
    // q [B, H, T_q, D] @ k^T [B, H, D, T_c] → [B, H, T_q, T_c]
    let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let scores = q.matmul(&k_t)?;

    // (1) ReLU on the per-head dot, BEFORE mixing heads. Without this a
    //     negative head cancels a positive one and a different key wins.
    let scores = scores.relu()?;

    // Per-head weight: [B, H, T_q] → [B, H, T_q, 1] for broadcast.
    let w = weights
        .unsqueeze(D::Minus1)?
        .to_dtype(scores.dtype())?
        .contiguous()?;
    let scaled = scores.broadcast_mul(&w)?;

    // (2) Sum over heads → ONE logit per (batch, query, key), so the
    //     downstream top-k selects a single key set shared by all heads.
    scaled.sum(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
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

    /// Deterministic pseudo-Gaussian tensor (SplitMix64 + Box-Muller): the
    /// same seed yields bit-identical F32 values on every run and platform.
    ///
    /// `make_indexer_tensors` used unseeded `Tensor::randn`, which made
    /// `v4_indexer_agrees_with_cpu_reference` flaky: each test process draws
    /// fresh weights, and occasionally two top-k scores land close enough
    /// that candle's gemm accumulation order and the naive CPU reference
    /// disagree on the ranking (observed failing on the macOS + Windows CI
    /// lanes while ubuntu passed, 2026-08-13). Fixed weights make the test
    /// outcome a constant.
    fn det_randn(seed: u64, mean: f32, std: f32, shape: (usize, usize), device: &Device) -> Tensor {
        let n = shape.0 * shape.1;
        let mut state = seed;
        let mut next = move || {
            // SplitMix64
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut vals = Vec::with_capacity(n + 1);
        while vals.len() < n {
            // Box-Muller from two uniforms in (0, 1]
            let u1 = ((next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
            let u2 = ((next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
            vals.push((mean as f64 + std as f64 * r * c) as f32);
            vals.push((mean as f64 + std as f64 * r * s) as f32);
        }
        vals.truncate(n);
        Tensor::from_vec(vals, shape, device).unwrap()
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
            det_randn(
                0xA11A_0001,
                0.0f32,
                0.02,
                (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank.unwrap()),
                device,
            )
            .to_dtype(dtype)
            .unwrap(),
        );
        // weights_proj.weight: [n_heads, hidden_size]
        m.insert(
            "weights_proj.weight".to_string(),
            det_randn(
                0xA11A_0002,
                0.0f32,
                0.02,
                (cfg.index_n_heads, cfg.hidden_size),
                device,
            )
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
        // compressor.wgate.weight: [coff*head_dim, hidden_size] — the
        // compressor consumes HIDDEN STATES, so its in_features is
        // `hidden_size`, never `ratio*head_dim`. See the module docs and
        // `v4_indexer_loads_published_artifact_shapes`.
        m.insert(
            "compressor.wgate.weight".to_string(),
            det_randn(
                0xA11A_0003,
                0.0f32,
                0.02,
                (coff * cfg.index_head_dim, cfg.hidden_size),
                device,
            )
            .to_dtype(dtype)
            .unwrap(),
        );
        // compressor.wkv.weight: [coff*head_dim, hidden_size]
        m.insert(
            "compressor.wkv.weight".to_string(),
            det_randn(
                0xA11A_0004,
                0.0f32,
                0.02,
                (coff * cfg.index_head_dim, cfg.hidden_size),
                device,
            )
            .to_dtype(dtype)
            .unwrap(),
        );
        m
    }

    /// Forward produces output shape [B, n_heads, T_q, topk].
    #[test]
    fn v4_indexer_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let cfg = synth_cfg(/*hidden*/ 20, /*q_lora*/ 24, /*n_heads*/ 4, /*head_dim*/ 8, /*topk*/ 6);
        let tensors = make_indexer_tensors(&cfg, 4, 2, &device);
        let vb = vb_from_map(tensors, DType::F32, &device);
        let indexer = V4Indexer::new(&cfg, vb, &device, false)?;

        let b = 2usize;
        let t_q = 3usize;
        let t_full = 16usize; // T_c = 4
        let q_a = Tensor::randn(0.0f32, 1.0, (b, t_q, cfg.q_lora_rank.unwrap()), &device)?;
        let xs_full = Tensor::randn(0.0f32, 1.0, (b, t_full, cfg.hidden_size), &device)?;
        let xs_q = Tensor::randn(0.0f32, 1.0, (b, t_q, cfg.hidden_size), &device)?;

        let out = indexer.forward(&q_a, &xs_full, &xs_q)?;
        // Output shape is [B, T_q, topk] — no head axis: the head dot products
        // are relu'd and summed into ONE logit per (query, key), so a single
        // key set is selected and shared by every head. `topk` is clamped to
        // T_c = 4 (synth_cfg uses topk=6 > T_c=4).
        let t_c = t_full / 4;
        let expected_k = cfg.index_topk.min(t_c);
        assert_eq!(out.dims(), &[b, t_q, expected_k]);
        Ok(())
    }

    /// All returned indices are in [0, T_c).
    #[test]
    fn v4_indexer_indices_in_range() -> Result<()> {
        let device = Device::Cpu;
        let cfg = synth_cfg(20, 24, 4, 8, /*topk*/ 3);
        let tensors = make_indexer_tensors(&cfg, 4, 2, &device);
        let vb = vb_from_map(tensors, DType::F32, &device);
        let indexer = V4Indexer::new(&cfg, vb, &device, false)?;

        let b = 1usize;
        let t_q = 2usize;
        let t_full = 12usize; // T_c = 3
        let q_a = Tensor::randn(0.0f32, 1.0, (b, t_q, cfg.q_lora_rank.unwrap()), &device)?;
        let xs_full = Tensor::randn(0.0f32, 1.0, (b, t_full, cfg.hidden_size), &device)?;
        let xs_q = Tensor::randn(0.0f32, 1.0, (b, t_q, cfg.hidden_size), &device)?;

        let out = indexer.forward(&q_a, &xs_full, &xs_q)?;
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
        let cfg = synth_cfg(20, 12, 2, 4, /*topk*/ 4);
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
        let xs_full = Tensor::arange(0u32, (b * t_full * cfg.hidden_size) as u32, &device)?
            .reshape((b, t_full, cfg.hidden_size))?
            .to_dtype(DType::F32)?;
        let xs_q = Tensor::arange(0u32, (b * t_q * cfg.hidden_size) as u32, &device)?
            .reshape((b, t_q, cfg.hidden_size))?
            .to_dtype(DType::F32)?;

        let out_a: Vec<u32> = indexer
            .forward(&q_a, &xs_full, &xs_q)?
            .flatten_all()?
            .to_vec1()?;
        let out_b: Vec<u32> = indexer
            .forward(&q_a, &xs_full, &xs_q)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(out_a, out_b, "indexer must be deterministic across re-runs");
        Ok(())
    }

    /// Agreement test (RUN-163): the V4Indexer's Candle forward path must
    /// agree with a hand-rolled per-(B, T_q) relu + weighted-head-sum + top-k
    /// baseline that mirrors what the CUDA FlashMLASparse kernels compute.
    /// This locks in the spec the CUDA kernels are compared against on a GPU
    /// host.
    ///
    /// The compressor is treated as a black box here — its output is taken
    /// from [`V4Compressor::forward_from_xs`], whose own semantics are pinned
    /// by the tests in `deepseek4.rs` — because the kernels this test exists
    /// to specify compute **only** the score + top-k. Duplicating the
    /// compressor math here is what made the previous version of this test
    /// vacuous: it mirrored the loader's own (wrong) assumption instead of the
    /// checkpoint's.
    #[test]
    fn v4_indexer_agrees_with_cpu_reference() -> Result<()> {
        let device = Device::Cpu;
        // small but realistic shape: 1 batch, 2 query tokens, 3 heads,
        // head_dim=8, ratio=4, t_full=16 → T_c=4; topk=3.
        let cfg = synth_cfg(/*hidden*/ 16, /*q_lora*/ 16, /*n_heads*/ 3, /*head_dim*/ 8, /*topk*/ 3);
        // Fixed-seed deterministic tensor map (see `det_randn`): identical
        // weights on every run and platform, so this agreement check is a
        // constant rather than a per-draw lottery over near-tied scores.
        let tensors = make_indexer_tensors(&cfg, 4, 2, &device);

        // Snapshot the actual weight tensors so the reference is guaranteed
        // to use the very tensors the indexer consumed.
        let wq_b_w = tensors["wq_b.weight"].clone();
        let weights_proj_w = tensors["weights_proj.weight"].clone();

        let vb = vb_from_map(tensors, DType::F32, &device);
        let indexer = V4Indexer::new(&cfg, vb, &device, false)?;

        let b = 1usize;
        let t_q = 2usize;
        let t_full = 16usize;
        let t_c = t_full / 4;
        let topk = cfg.index_topk;

        // Inputs: small distinct values via arange so no two scores tie.
        let q_a = Tensor::arange(0u32, (b * t_q * cfg.q_lora_rank.unwrap()) as u32, &device)?
            .reshape((b, t_q, cfg.q_lora_rank.unwrap()))?
            .to_dtype(DType::F32)?
            .affine(0.013, -0.5)?; // diversify
        let xs_full = Tensor::arange(0u32, (b * t_full * cfg.hidden_size) as u32, &device)?
            .reshape((b, t_full, cfg.hidden_size))?
            .to_dtype(DType::F32)?
            .affine(0.007, -0.25)?;
        let xs_q = Tensor::arange(0u32, (b * t_q * cfg.hidden_size) as u32, &device)?
            .reshape((b, t_q, cfg.hidden_size))?
            .to_dtype(DType::F32)?
            .affine(0.011, -0.3)?;

        // ---- Candle forward (the path under test) ----
        let got: Vec<u32> = indexer
            .forward(&q_a, &xs_full, &xs_q)?
            .flatten_all()?
            .to_vec1()?;

        // ---- Hand-rolled CPU reference (mirror of the candle path) ----
        // 1. q = wq_b @ q_a, reshape, transpose → [B, H, T_q, D]
        let n_heads = cfg.index_n_heads;
        let head_dim = cfg.index_head_dim;
        let q_lora = cfg.q_lora_rank.unwrap();
        let hidden = cfg.hidden_size;

        let q_a_v: Vec<f32> = q_a.flatten_all()?.to_vec1()?;
        let xs_q_v: Vec<f32> = xs_q.flatten_all()?.to_vec1()?;
        let wq_b_v: Vec<f32> = wq_b_w.flatten_all()?.to_vec1()?;
        let weights_proj_v: Vec<f32> = weights_proj_w.flatten_all()?.to_vec1()?;

        // Linear: out[i] = sum_k(w[i,k] * in[k])
        let lin = |w: &[f32], in_dim: usize, out_dim: usize, x: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; x.len() / in_dim * out_dim];
            let rows = x.len() / in_dim;
            for r in 0..rows {
                for o in 0..out_dim {
                    let mut acc = 0.0f32;
                    for k in 0..in_dim {
                        acc += w[o * in_dim + k] * x[r * in_dim + k];
                    }
                    out[r * out_dim + o] = acc;
                }
            }
            out
        };

        // 1. indexer_q = wq_b(q_a) → [B*T_q, n_heads*head_dim]
        let q_flat = lin(&wq_b_v, q_lora, n_heads * head_dim, &q_a_v);
        // Reshape to [B, H, T_q, D] in pure-vec layout: source has
        // [B, T_q, n_heads, head_dim], target needs [B, H, T_q, D].
        let mut q_bhqd = vec![0.0f32; b * n_heads * t_q * head_dim];
        for bi in 0..b {
            for ti in 0..t_q {
                for h in 0..n_heads {
                    for d in 0..head_dim {
                        let src = ((bi * t_q + ti) * n_heads + h) * head_dim + d;
                        let dst = ((bi * n_heads + h) * t_q + ti) * head_dim + d;
                        q_bhqd[dst] = q_flat[src];
                    }
                }
            }
        }

        // 2. Compressed keys: `[B, T_c, head_dim]` — MQA, one key per
        //    compressed slot shared by every head (SGLang's indexer K cache is
        //    `num_heads_kv = 1`). Taken from the shared compressor rather than
        //    re-derived; see the doc comment.
        let indexer_k: Vec<f32> = indexer
            .compressor
            .forward_from_xs(&xs_full)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(indexer_k.len(), b * t_c * head_dim);

        // 3. weights_proj(xs_q) * weight_scale → [B, T_q, n_heads] → [B, H, T_q]
        let weight_scale = ((head_dim as f64).powf(-0.5) * (n_heads as f64).powf(-0.5)) as f32;
        let wp_flat = lin(&weights_proj_v, hidden, n_heads, &xs_q_v);
        let mut scale = vec![0.0f32; b * n_heads * t_q];
        for bi in 0..b {
            for ti in 0..t_q {
                for h in 0..n_heads {
                    let src = (bi * t_q + ti) * n_heads + h;
                    let dst = (bi * n_heads + h) * t_q + ti;
                    scale[dst] = wp_flat[src] * weight_scale;
                }
            }
        }

        // 4. Per-(B, T_q) logits + top-k:
        //      logit[b,q,c] = Σ_h relu(q_h · k_c) * w[b,h,q]
        //    then ONE top-k over c, shared by all heads. Use the same
        //    lowest-index tiebreak as both V4Indexer::forward (candle's
        //    topk_unsorted) and the FlashMLASparse CPU reference.
        let k_top = topk.min(t_c);
        let mut expected = vec![0u32; b * t_q * k_top];
        for bi in 0..b {
            for ti in 0..t_q {
                let mut scored: Vec<(usize, f32)> = (0..t_c)
                    .map(|ci| {
                        let mut logit = 0.0f32;
                        for h in 0..n_heads {
                            let s = scale[(bi * n_heads + h) * t_q + ti];
                            let q_base = ((bi * n_heads + h) * t_q + ti) * head_dim;
                            // MQA: the key set has no head axis.
                            let k_base = (bi * t_c + ci) * head_dim;
                            let mut acc = 0.0f32;
                            for d in 0..head_dim {
                                acc += q_bhqd[q_base + d] * indexer_k[k_base + d];
                            }
                            logit += acc.max(0.0) * s;
                        }
                        (ci, logit)
                    })
                    .collect();
                scored.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                let out_base = (bi * t_q + ti) * k_top;
                for i in 0..k_top {
                    expected[out_base + i] = scored[i].0 as u32;
                }
            }
        }

        // Candle's topk_unsorted returns indices in arbitrary order (no
        // guarantee of sort). The flashmlasparse spec also produces an
        // unsorted set. Compare as sets per (B, T_q) row.
        for row in 0..(b * t_q) {
            let mut g: Vec<u32> = got[row * k_top..(row + 1) * k_top].to_vec();
            let mut e: Vec<u32> = expected[row * k_top..(row + 1) * k_top].to_vec();
            g.sort_unstable();
            e.sort_unstable();
            assert_eq!(
                g, e,
                "row {row}: V4Indexer candle path disagreed with CPU reference; \
                 got={g:?} expected={e:?}"
            );
        }
        Ok(())
    }

    /// **Force-full-selection dense-equivalence.** With `index_topk >= T_c`
    /// the indexer has no choice: it must return *every* compressed key. So
    /// gathering K/V with the selection and attending over the gathered set
    /// must reproduce dense attention over all of `T_c` **bit-exactly**.
    ///
    /// This is the property that makes today's serving path safe (CSA
    /// top-512 covers all compressed entries whenever
    /// `ctx <= index_topk * ratio = 2048`) and it is the free regression
    /// guard the previous per-head formula never had: any selection bug that
    /// drops or duplicates a key breaks it immediately.
    #[test]
    fn v4_indexer_full_selection_is_dense_equivalent() -> Result<()> {
        let device = Device::Cpu;
        let b = 2usize;
        let t_q = 3usize;
        let t_full = 20usize;
        let t_c = t_full / 4; // 5
        let n_heads = 3usize;
        let head_dim = 8usize;
        // index_topk = 8 >= T_c = 5 → forced full selection.
        let cfg = synth_cfg(/*hidden*/ 16, /*q_lora*/ 16, n_heads, head_dim, /*topk*/ 8);
        assert!(cfg.index_topk >= t_c, "test requires index_topk >= T_c");

        let tensors = make_indexer_tensors(&cfg, 4, 2, &device);
        let vb = vb_from_map(tensors, DType::F32, &device);
        let indexer = V4Indexer::new(&cfg, vb, &device, false)?;

        let q_a = det_randn(0xF0FF_0001, 0.0, 1.0, (b * t_q, cfg.q_lora_rank.unwrap()), &device)
            .reshape((b, t_q, cfg.q_lora_rank.unwrap()))?;
        let xs_full = det_randn(0xF0FF_0002, 0.0, 1.0, (b * t_full, cfg.hidden_size), &device)
            .reshape((b, t_full, cfg.hidden_size))?;
        let xs_q = det_randn(0xF0FF_0003, 0.0, 1.0, (b * t_q, cfg.hidden_size), &device)
            .reshape((b, t_q, cfg.hidden_size))?;

        let idx = indexer.forward(&q_a, &xs_full, &xs_q)?;
        assert_eq!(idx.dims(), &[b, t_q, t_c]);

        // (a) The selection is the complete key set for every (batch, query).
        let idx_v: Vec<u32> = idx.flatten_all()?.to_vec1()?;
        for row in 0..(b * t_q) {
            let mut r: Vec<u32> = idx_v[row * t_c..(row + 1) * t_c].to_vec();
            r.sort_unstable();
            let full: Vec<u32> = (0..t_c as u32).collect();
            assert_eq!(
                r, full,
                "row {row}: forced-full selection must return every key exactly once"
            );
        }

        // (b) Gathering main-attention K/V with that selection and attending
        //     over the gathered set reproduces dense attention bit-exactly.
        //     Sorted indices are the identity permutation, so no reordering
        //     can perturb the softmax accumulation order.
        let main_q = det_randn(0xF0FF_0004, 0.0, 1.0, (b * n_heads * t_q, head_dim), &device)
            .reshape((b, n_heads, t_q, head_dim))?;
        let main_k = det_randn(0xF0FF_0005, 0.0, 1.0, (b * n_heads * t_c, head_dim), &device)
            .reshape((b, n_heads, t_c, head_dim))?;
        let main_v = det_randn(0xF0FF_0006, 0.0, 1.0, (b * n_heads * t_c, head_dim), &device)
            .reshape((b, n_heads, t_c, head_dim))?;

        let attend = |k: &Tensor, v: &Tensor| -> Result<Vec<f32>> {
            let s = main_q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
            let p = candle_nn::ops::softmax_last_dim(&s)?;
            p.matmul(v)?.flatten_all()?.to_vec1::<f32>()
        };

        let dense = attend(&main_k, &main_v)?;

        // Gather with the (sorted) selection of query 0 — the selection is
        // identical for every query since it is the full set.
        let mut sel: Vec<u32> = idx_v[0..t_c].to_vec();
        sel.sort_unstable();
        let sel_t = Tensor::from_vec(sel, (t_c,), &device)?;
        let gathered_k = main_k.index_select(&sel_t, 2)?.contiguous()?;
        let gathered_v = main_v.index_select(&sel_t, 2)?.contiguous()?;
        let sparse = attend(&gathered_k, &gathered_v)?;

        assert_eq!(
            dense, sparse,
            "forced-full-selection sparse attention must equal dense attention bit-exactly"
        );
        Ok(())
    }

    /// Pin the scoring semantics on a hand-computed fixture where the OLD
    /// formula (`(q_h · k_c) * w_h`, no relu, top-k **per head**) and the NEW
    /// one (`Σ_h relu(q_h · k_c) * w_h`, one shared top-k) provably select
    /// **different** keys — so a regression to the old formula cannot pass
    /// silently.
    ///
    /// Fixture: B=1, T_q=1, n_heads=2, head_dim=1, T_c=2, w = [1, 1],
    /// `q = 1` on both heads, `k = [[3, -1], [-4, 2]]` (head-major).
    ///
    /// | key | h0 dot | h1 dot | new: Σ relu·w | no-relu sum | old per-head  |
    /// |-----|--------|--------|---------------|-------------|---------------|
    /// | c0  |  +3    |  -4    | 3 + 0 = **3** | 3 - 4 = -1  | h0 picks c0   |
    /// | c1  |  -1    |  +2    | 0 + 2 =   2   | -1 + 2 = +1 | h1 picks c1   |
    ///
    /// New picks **c0**. Dropping the relu (but keeping the head sum) picks
    /// c1. Keeping per-head top-k picks c0 for head 0 and c1 for head 1 — two
    /// different key sets. All three outcomes are distinct.
    #[test]
    fn indexer_logits_relu_and_head_sum_pick_different_keys() -> Result<()> {
        let device = Device::Cpu;
        let (b, n_heads, t_q, t_c, head_dim) = (1usize, 2usize, 1usize, 2usize, 1usize);

        let q = Tensor::from_vec(vec![1.0f32, 1.0], (b, n_heads, t_q, head_dim), &device)?;
        // k[h=0] = [3, -1]; k[h=1] = [-4, 2]
        let k = Tensor::from_vec(
            vec![3.0f32, -1.0, -4.0, 2.0],
            (b, n_heads, t_c, head_dim),
            &device,
        )?;
        let w = Tensor::from_vec(vec![1.0f32, 1.0], (b, n_heads, t_q), &device)?;

        let logits = indexer_logits(&q, &k, &w)?;
        assert_eq!(logits.dims(), &[b, t_q, t_c]);
        let got: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        // relu(3)*1 + relu(-4)*1 = 3 ; relu(-1)*1 + relu(2)*1 = 2
        assert_eq!(got, vec![3.0f32, 2.0], "hand-computed logits");

        let TopKOutput { indices, .. } = logits.contiguous()?.topk_unsorted(1)?;
        let picked: Vec<u32> = indices.flatten_all()?.to_vec1()?;
        assert_eq!(picked, vec![0u32], "new formula must select key c0");

        // --- The old formulas, computed inline, must disagree. ---
        // (a) no relu, but summed over heads → c1 wins (score +1 vs -1).
        let no_relu = q
            .matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
            .broadcast_mul(&w.unsqueeze(D::Minus1)?)?
            .sum(1)?;
        let no_relu_v: Vec<f32> = no_relu.flatten_all()?.to_vec1()?;
        assert_eq!(no_relu_v, vec![-1.0f32, 1.0]);
        let TopKOutput { indices, .. } = no_relu.contiguous()?.topk_unsorted(1)?;
        let old_pick: Vec<u32> = indices.flatten_all()?.to_vec1()?;
        assert_eq!(
            old_pick,
            vec![1u32],
            "no-relu variant must pick a DIFFERENT key than the fixed formula"
        );
        assert_ne!(old_pick, picked);

        // (b) per-head top-k (no relu) → head 0 picks c0, head 1 picks c1:
        //     two different key sets, i.e. n_heads× the gather traffic.
        let per_head = q
            .matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
            .broadcast_mul(&w.unsqueeze(D::Minus1)?)?;
        let TopKOutput { indices, .. } = per_head.contiguous()?.topk_unsorted(1)?;
        let per_head_pick: Vec<u32> = indices.flatten_all()?.to_vec1()?;
        assert_eq!(per_head_pick, vec![0u32, 1u32]);
        assert_ne!(
            per_head_pick[0], per_head_pick[1],
            "fixture must make the heads disagree, or it proves nothing"
        );
        Ok(())
    }

    /// `weight_scale = softmax_scale * n_heads^-0.5` is applied to the
    /// logits (selection-neutral, but the logit magnitudes must be right).
    #[test]
    fn v4_indexer_weight_scale_matches_sglang() -> Result<()> {
        let device = Device::Cpu;
        let cfg = synth_cfg(16, 16, /*n_heads*/ 4, /*head_dim*/ 16, 8);
        let tensors = make_indexer_tensors(&cfg, 4, 2, &device);
        let vb = vb_from_map(tensors, DType::F32, &device);
        let indexer = V4Indexer::new(&cfg, vb, &device, false)?;
        // softmax_scale = head_dim^-0.5 = 16^-0.5 = 0.25
        // weight_scale  = 0.25 * 4^-0.5 = 0.125
        assert!(
            (indexer.weight_scale - 0.125).abs() < 1e-12,
            "weight_scale = {} != 0.125",
            indexer.weight_scale
        );
        Ok(())
    }

    // ===== Round-trip shape contract vs. the PUBLISHED artifact =====
    //
    // Arc's first published artifact, `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2`
    // (baked 2026-08-15), carries
    // `layers.N.attn.indexer.compressor.wgate.weight` at **[256, 4096]** on
    // every CSA layer (2, 4, …, 42). This loader asked for **[256, 512]**, so
    // every CSA layer logged
    //   `V4 CSA layer N: indexer load failed (shape mismatch for
    //    layers.N.attn.indexer.compressor.wgate.weight,
    //    expected: [256, 512], got: [256, 4096])`
    // and silently fell back to dense-over-compressed. The loader was wrong,
    // not the artifact:
    //   * `[256, 4096] == [coff * index_head_dim, hidden_size]`
    //     (`coff = 2`, `index_head_dim = 128`, `hidden_size = 4096`);
    //   * SGLang builds it as `ReplicatedLinear(config.hidden_size,
    //     2 * coff * head_dim)` —
    //     `research/code/06_foundation/sglang/python/sglang/srt/layers/
    //      attention/dsv4/compressor.py:305,318-325`;
    //   * and its V4 loader concatenates two `[coff*head_dim, hidden_size]`
    //     halves — `.../sglang/srt/models/deepseek_v4.py:1633-1652`.
    // Arc's own reference audit had already recorded this
    // (`docs/notes/v4-reference-audit.md:1358-1365`) and it shipped anyway,
    // because the only tests were built from the loader's own assumption.
    //
    // These two tests are the contract in both directions: the artifact's
    // shape MUST load, and the pre-fix shape MUST be rejected.

    /// Real V4-Flash dimensions, taken from `DeepSeekV4Config`'s pinned
    /// `V4_FLASH_CONFIG_JSON` (`deepseek4.rs`): `hidden_size = 4096`,
    /// `index_n_heads = 64`, `index_head_dim = 128`, `index_topk = 512`,
    /// `q_lora_rank = 1024` (`research/v4_audit.md:104`).
    fn v4_flash_indexer_cfg() -> super::super::deepseek4::DeepSeekV4Config {
        synth_cfg(
            /*hidden*/ 4096, /*q_lora*/ 1024, /*n_heads*/ 64, /*head_dim*/ 128,
            /*topk*/ 512,
        )
    }

    /// Tensors at exactly the shapes the published UQFF contains. Zeros —
    /// only the shapes are under test, and zeros keep the ~43 MB fixture
    /// cheap to build.
    fn published_artifact_indexer_tensors(
        cfg: &super::super::deepseek4::DeepSeekV4Config,
        compressor_in_features: usize,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        const COFF: usize = 2;
        const RATIO: usize = 4;
        let hd = cfg.index_head_dim;
        let mut m = HashMap::new();
        let z = |dims: (usize, usize)| Tensor::zeros(dims, DType::F32, device);
        m.insert(
            "wq_b.weight".to_string(),
            z((cfg.index_n_heads * hd, cfg.q_lora_rank.unwrap()))?,
        );
        m.insert(
            "weights_proj.weight".to_string(),
            z((cfg.index_n_heads, cfg.hidden_size))?,
        );
        m.insert("compressor.ape".to_string(), z((RATIO, COFF * hd))?);
        m.insert(
            "compressor.norm.weight".to_string(),
            Tensor::ones(hd, DType::F32, device)?,
        );
        m.insert(
            "compressor.wgate.weight".to_string(),
            z((COFF * hd, compressor_in_features))?,
        );
        m.insert(
            "compressor.wkv.weight".to_string(),
            z((COFF * hd, compressor_in_features))?,
        );
        Ok(m)
    }

    /// The artifact's shapes load. Regression guard for
    /// `indexer load failed (shape mismatch …)` on every CSA layer.
    #[test]
    fn v4_indexer_loads_published_artifact_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cfg = v4_flash_indexer_cfg();

        // What the artifact actually holds: [coff*index_head_dim, hidden_size].
        let in_features = cfg.hidden_size;
        assert_eq!(
            (2 * cfg.index_head_dim, in_features),
            (256, 4096),
            "fixture must reproduce the artifact's observed [256, 4096]"
        );

        let tensors = published_artifact_indexer_tensors(&cfg, in_features, &device)?;
        let vb = vb_from_map(tensors, DType::F32, &device);
        let indexer = V4Indexer::new(&cfg, vb, &device, false)?;

        assert_eq!(indexer.compressor.head_dim, cfg.index_head_dim);
        assert_eq!(indexer.compressor.hidden_size, cfg.hidden_size);
        assert_eq!(indexer.compressor.coff, 2);
        assert_eq!(indexer.compressor.ratio, 4);
        Ok(())
    }

    /// The pre-fix expectation `[coff*head_dim, ratio*head_dim] = [256, 512]`
    /// must now be REJECTED. This is the half that proves the test above is
    /// not vacuous: revert the loader to `ratio * head_dim` and this test
    /// goes green while `v4_indexer_loads_published_artifact_shapes` goes red.
    #[test]
    fn v4_indexer_rejects_pre_fix_compressor_shape() -> Result<()> {
        let device = Device::Cpu;
        let cfg = v4_flash_indexer_cfg();

        // The wrong in_features the loader used to ask for.
        let bad_in_features = 4 * cfg.index_head_dim; // 512
        assert_ne!(
            bad_in_features, cfg.hidden_size,
            "fixture is only meaningful when the two candidate in_features differ"
        );

        let tensors = published_artifact_indexer_tensors(&cfg, bad_in_features, &device)?;
        let vb = vb_from_map(tensors, DType::F32, &device);
        let msg = match V4Indexer::new(&cfg, vb, &device, false) {
            Ok(_) => panic!(
                "[256, 512] is not a V4 indexer compressor weight and must not load — \
                 the loader has drifted back to `ratio * head_dim` in_features"
            ),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("shape mismatch") || msg.contains("wkv"),
            "expected a shape-mismatch error naming the compressor tensor, got: {msg}"
        );
        Ok(())
    }
}
