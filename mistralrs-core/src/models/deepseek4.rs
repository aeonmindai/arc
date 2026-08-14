#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
// V4 Flash/Pro is WIP: MTP/mHC scaffolding (extra methods/fields) is wired up
// incrementally, and the dense model signatures are intentionally verbose. Keep
// these style/dead-code lints quiet here rather than churn this numerically
// sensitive file; CI does not gate mistralrs-core (see arc-tools/CI_HYGIENE.md).
#![allow(dead_code, clippy::type_complexity, clippy::clone_on_copy)]
//! DeepSeek V4 Pro / V4 Flash model loader.
//!
//! ## What this file is
//!
//! V4 is **not** an MLA variant — it's a fundamentally different attention
//! shape:
//!
//! - **Q**: LoRA (`q_a_proj` / `wq_a` → `q_norm` → `q_b_proj` / `wq_b`),
//!   reshape to `[B, n_heads=64, T, head_dim=512]`.
//! - **K/V**: a **single fused** `wkv` projection (`hidden=4096 → head_dim=512`)
//!   that produces ONE MQA head per token. The kernel broadcasts that one
//!   head across all 64 Q heads. There is NO `kv_a_proj_with_mqa`, NO
//!   `kv_b_proj`, NO `kv_lora_rank`.
//! - **RoPE**: applied **in place** to the last `qk_rope_head_dim=64` dims of
//!   each Q head's 512-vector and of K's 512-vector. The first 448 ("nope")
//!   dims of the same 512-vector are left untouched. No split/concat at the
//!   tensor level (a candle-backed implementation does split+rotate+cat
//!   under the hood; the SGLang `fused_norm_rope_inplace` does it in one
//!   pass).
//! - **o_proj**: grouped LoRA — `wo_a` (`n_heads*head_dim/n_groups →
//!   n_groups*o_lora_rank`), then `wo_b` (`n_groups*o_lora_rank →
//!   hidden_size`). With `n_groups=8`, `o_lora_rank=1024`:
//!   `wo_a: [4096→8192]`, `wo_b: [8192→4096]`.
//! - **All 43 layers are MoE** (`first_k_dense_replace=0`), top-6 routed +
//!   1 shared, `scoring_func="sqrtsoftplus"`.
//! - **mHC (Manifold-Constrained Hyper-Connections)** replaces standard
//!   `x + sublayer(norm(x))` residual at every layer. See
//!   [`super::dsv4_mhc::V4MHCLayerParams`].
//! - **Sparse attention dispatch** per layer via `compress_ratios`:
//!   - `0`  → standard MQA (layers 0, 1, 42 only)
//!   - `4`  → CSA (compressor + C4 indexer top-k=512)
//!   - `128` → HCA (compressor only, dense MQA over 128× compressed K)
//! - **FP8 e4m3 + UE8M0 scales**, 128×128 weight blocks. Companion
//!   `.scale` tensor per weight tensor (V4 native naming — HF rewrites
//!   to `.weight_scale_inv`).
//! - **MTP head**: full transformer decoder + 256-expert MoE wrapped with
//!   `h_proj` / `e_proj` / `enorm` / `hnorm` / `norm` + its own
//!   `hc_head_*` / `hc_attn_*` / `hc_ffn_*`.
//!
//! ## References
//!
//! - `/Users/jish/Documents/GitHub/arc/research/v4_audit.md` — the source of
//!   truth for every V4 architectural decision in this file.
//! - SGLang reference: `research/code/06_foundation/sglang/python/sglang/srt/
//!   models/deepseek_v4.py` (MQALayer, DecoderLayer, weight remapping).
//! - V4 Flash config.json: `/tmp/v4_flash_config.json`.
//! - V4 Flash safetensors index: `/tmp/v4_flash_index.json`.
//!
//! ## Out of scope for this file
//!
//! Anything below P1 in the audit (`v4_audit.md §8`). In particular:
//! - Fused CUDA kernels (`fused_norm_rope`, `c4`, `c128_online`) — vendored
//!   into `mistralrs-quant/cuda_kernels/` by a follow-up agent.
//! - `attn_sink` actually fed into the SDPA softmax — the tensor is loaded
//!   and stored on `SdpaParams.sinks` (broadcastable shape), and the
//!   sinks-aware kernel reads it when present.
//! - MTP drafting itself (`MtpSpeculativePipeline` drives it via
//!   `MtpDecodeKit`). This file loads the light `h_proj`/`e_proj` heads
//!   always, and the FULL `mtp.0.*` decoder block ([`MtpBlock`]: attention
//!   + 256-expert MoE + hnorm/enorm/norm) when `--mtp-depth > 0` was
//!   declared before load — plus the block's single-token forward that the
//!   draft chain calls.
//! - FP8 native-to-HF tensor name remap (`*.scale → *.weight_scale_inv`).
//!   Handled by [`mistralrs_quant::attach_rename_rules`] called from
//!   [`crate::pipeline::loaders::normal_loaders::DeepSeekV4Loader::load`]
//!   before this constructor runs. The wrapper is transparent for HF-format
//!   checkpoints (direct lookups still work) and intercepts only missing
//!   `weight_scale_inv` paths to redirect to V4-native `.scale`. Audit §0
//!   P0 #5 + audit §8 item 5.
//! ## RUN-164: 4-D mHC residual threading
//!
//! When the V4 global mHC head (`hc_head_*`) is loaded AND every layer has
//! its per-layer mHC params loaded, the model takes the 4-D end-to-end
//! residual threading path:
//!   1. The embedding output `[B, T, hidden]` is lifted to
//!      `[B, T, hc_mult, hidden]` once at the model entry.
//!   2. Each `DecoderLayer::forward_4d` threads the 4-D state through its
//!      attention + FFN blocks via `hc_pre` (learned per-token stream
//!      collapse) + `hc_post` (re-expand new branch output into streams).
//!   3. The 4-D state is collapsed back to 3-D via the learned
//!      `V4MHCHead::forward` exactly once before `norm` + `lm_head`.
//!
//! The legacy 3-D path (`DecoderLayer::forward` + 3-D bridge from RUN-169)
//! is retained for partial-V4 checkpoints (per-layer mHC present but
//! `hc_head_*` absent) and for V3-style fixtures / synth tests.

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::Deserialize;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        embedding, Activation, CausalMasker, DeepSeekV2RopeConfig, DeepSeekV2RopeScaling,
        DeepSeekV2RotaryEmbedding, Mlp, RmsNorm,
    },
    layers_masker::{masked_fill, PastKvLenCache},
    moe::{MoEExperts, MoEExpertsConfig},
    ops::{SplitOp, TopKLastDimOp, TopKOutput},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

use super::dsv4_mhc::V4MHCLayerParams;

serde_default_fn!(f64, routed_scaling_factor, 1.5);
serde_default_fn!(TopkMethod, topk_method, TopkMethod::NoAuxTc);
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(ScoringFunc, scoring_func, ScoringFunc::SqrtSoftplus);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(usize, default_head_dim, 512);
serde_default_fn!(usize, default_num_kv_heads, 1);
serde_default_fn!(usize, default_num_hash_layers, 3);
serde_default_fn!(usize, default_hc_mult, 4);
serde_default_fn!(f32, default_hc_eps, 1.0e-6);
serde_default_fn!(usize, default_hc_sinkhorn_iters, 20);
serde_default_fn!(f32, default_swiglu_limit, 10.0);
serde_default_fn!(usize, default_n_group, 1);
serde_default_fn!(usize, default_topk_group, 1);

#[derive(Deserialize, Clone, Debug, PartialEq, Eq)]
pub(crate) enum TopkMethod {
    #[serde(rename = "noaux_tc")]
    NoAuxTc,
    #[serde(rename = "greedy")]
    Greedy,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

/// V4 MoE gate scoring function.
///
/// V4 Flash defaults to `sqrtsoftplus(logit) = sqrt(softplus(logit))` per the
/// V4 paper §4.1 ("Reduced-variance routing"). Audit §0 "scoring_func"
/// + §5 lines 463-465.
#[derive(Deserialize, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ScoringFunc {
    #[serde(rename = "softmax")]
    Softmax,
    #[serde(rename = "sigmoid")]
    Sigmoid,
    /// V4-specific: `s = sqrt(softplus(logit))`.
    #[serde(rename = "sqrtsoftplus")]
    SqrtSoftplus,
}

/// V4 Flash config — matches the real `config.json` published at
/// `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash`.
///
/// **Notably absent vs V3**: `kv_lora_rank`, `v_head_dim`, `qk_nope_head_dim`,
/// `intermediate_size`. **Newly present**: `head_dim`, `num_key_value_heads`,
/// `num_hash_layers`, `hc_mult`, `hc_eps`, `hc_sinkhorn_iters`, `swiglu_limit`,
/// `compress_ratios`, `compress_rope_theta`, `o_lora_rank`, `o_groups`,
/// `index_*`. See `v4_audit.md §0 + §5`.
///
/// For backwards-compat with older test fixtures and SGLang's dataclass
/// (which carries V3 defaults), `intermediate_size`, `kv_lora_rank`,
/// `v_head_dim`, `qk_nope_head_dim`, `n_group`, `topk_group` are accepted as
/// optional fields. They're **not used** on the V4 forward path — `head_dim`
/// is the source of truth and `qk_nope_head_dim` is derived as
/// `head_dim - qk_rope_head_dim`.
#[derive(Deserialize, Clone, Debug)]
pub struct DeepSeekV4Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    /// V4: 512. Source of truth for K/V head dim and Q head dim.
    /// Audit §0 + §5 lines 456 ("MISSING `head_dim`").
    #[serde(default = "default_head_dim")]
    pub(crate) head_dim: usize,
    /// V4: not used. Optional for tolerance of old configs. The dense MLP
    /// is unreachable when `first_k_dense_replace=0` (default for V4).
    #[serde(default)]
    pub(crate) intermediate_size: Option<usize>,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    /// V4: 1 (MQA — single KV head, broadcast across all Q heads at the
    /// kernel). Audit §0 + §5 line 457.
    #[serde(default = "default_num_kv_heads")]
    pub(crate) num_key_value_heads: usize,
    pub(crate) n_shared_experts: Option<usize>,
    pub(crate) n_routed_experts: Option<usize>,
    /// V4: 1.5. Audit §0 + §5 line 463.
    #[serde(default = "routed_scaling_factor")]
    pub(crate) routed_scaling_factor: f64,
    /// V4: `noaux_tc`. Audit §0 (default in V4 Flash config).
    #[serde(default = "topk_method")]
    pub(crate) topk_method: TopkMethod,
    pub(crate) num_experts_per_tok: Option<usize>,
    #[serde(default = "moe_layer_freq")]
    pub(crate) moe_layer_freq: usize,
    /// V4: 0 (every layer is MoE). Audit §0 + §5 line 448 ("OK").
    #[serde(default = "first_k_dense_replace")]
    pub(crate) first_k_dense_replace: usize,
    /// V4: `sqrtsoftplus`. Audit §0 + §5 line 464.
    #[serde(default = "scoring_func")]
    pub(crate) scoring_func: ScoringFunc,
    #[serde(default = "hidden_act")]
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) rope_theta: f32,
    pub(crate) rope_scaling: Option<DeepSeekV2RopeScaling>,
    #[serde(default)]
    pub(crate) attention_bias: bool,
    /// V4: 1024. Always present in V4 Flash. Optional only for V3-style
    /// fixtures with a plain Q projection.
    pub(crate) q_lora_rank: Option<usize>,
    /// V4: 64. RoPE is applied to the last `qk_rope_head_dim` dims of each
    /// 512-dim head vector (Q and K both). Audit §3.
    pub(crate) qk_rope_head_dim: usize,
    /// V4: not in real config. Kept for parser tolerance of older fixtures.
    /// Audit §5 line 451 ("`kv_lora_rank` — REMOVE").
    #[serde(default)]
    pub(crate) kv_lora_rank: Option<usize>,
    /// V4: not in real config. Kept for parser tolerance. Audit §5 line 452.
    #[serde(default)]
    pub(crate) v_head_dim: Option<usize>,
    /// V4: derived as `head_dim - qk_rope_head_dim = 512-64 = 448`. Stored
    /// only for parser tolerance; the forward path uses
    /// [`DeepSeekV4Config::qk_nope_head_dim()`] accessor instead.
    /// Audit §5 line 453 ("`qk_nope_head_dim` — REMOVE").
    #[serde(default)]
    pub(crate) qk_nope_head_dim: Option<usize>,
    #[serde(alias = "quantization")]
    pub(crate) quantization_config: Option<QuantizedConfig>,
    /// V4: not in real config. Default 1. Audit §5 line 457
    /// ("`n_group` — Make optional, default 1").
    #[serde(default = "default_n_group")]
    pub(crate) n_group: usize,
    /// V4: not in real config. Default 1.
    #[serde(default = "default_topk_group")]
    pub(crate) topk_group: usize,

    // ----------------- V4-specific fields ---------------------------------
    /// Per-layer compression dispatch: 0 = standard, 4 = CSA, 128 = HCA.
    /// V4 Flash: 44 entries (`num_hidden_layers + 1` to cover the MTP
    /// layer). Audit §1 + §0.
    #[serde(default)]
    pub(crate) compress_ratios: Vec<i32>,
    /// Sliding-window attention size (V4: 128). Used inside the C4
    /// indexer for local-window token selection. Audit §0 ("Sliding window
    /// — OK").
    #[serde(default = "default_sliding_window")]
    pub(crate) sliding_window: usize,
    /// Separate RoPE base for compressed-attention layers (V4: 160000).
    /// Compressed layers use this; standard layers (`compress_ratio == 0`)
    /// use `rope_theta`. Audit §0 + §1 + SGLang `deepseek_v4.py:220`.
    #[serde(default = "default_compress_rope_theta")]
    pub(crate) compress_rope_theta: f32,
    /// Grouped o_proj LoRA: rank per group (V4: 1024). Audit §0 + §5.
    #[serde(default)]
    pub(crate) o_lora_rank: Option<usize>,
    /// Grouped o_proj LoRA: number of groups (V4: 8). Audit §0 + §5.
    #[serde(default)]
    pub(crate) o_groups: Option<usize>,
    /// Lightning Indexer: number of indexer heads (V4: 64).
    #[serde(default = "default_index_n_heads")]
    pub(crate) index_n_heads: usize,
    /// Lightning Indexer: per-head dim (V4: 128).
    #[serde(default = "default_index_head_dim")]
    pub(crate) index_head_dim: usize,
    /// Lightning Indexer: top-k tokens to retain on CSA layers (V4: 512).
    #[serde(default = "default_index_topk")]
    pub(crate) index_topk: usize,
    /// Number of hash routing layers for MoE expert dispatch (V4: 3).
    /// Audit §0 + §5 line 458.
    #[serde(default = "default_num_hash_layers", alias = "n_hash_layers")]
    pub(crate) num_hash_layers: usize,
    /// mHC multiplier: parallel residual streams (V4: 4). Audit §0 + §5.
    #[serde(default = "default_hc_mult")]
    pub(crate) hc_mult: usize,
    /// mHC additive epsilon (V4: 1e-6). Audit §0 + §5.
    #[serde(default = "default_hc_eps")]
    pub(crate) hc_eps: f32,
    /// mHC sinkhorn iteration count (V4: 20). Audit §0 + §5.
    #[serde(default = "default_hc_sinkhorn_iters")]
    pub(crate) hc_sinkhorn_iters: usize,
    /// SwiGLU clamp for MoE expert MLP (V4: 10.0). Audit §0 + §5 line 462.
    #[serde(default = "default_swiglu_limit")]
    pub(crate) swiglu_limit: f32,
}

fn default_sliding_window() -> usize {
    128
}
fn default_compress_rope_theta() -> f32 {
    160000.0
}
fn default_index_n_heads() -> usize {
    64
}
fn default_index_head_dim() -> usize {
    128
}
fn default_index_topk() -> usize {
    512
}

impl DeepSeekV4Config {
    /// Returns the compress ratio for layer `layer_idx`, or 0 (standard) if
    /// the index is out of bounds. Audit §1 layer-to-pattern mapping.
    pub fn layer_compress_ratio(&self, layer_idx: usize) -> i32 {
        self.compress_ratios.get(layer_idx).copied().unwrap_or(0)
    }

    /// V4: the "nope" portion of each head vector — the first
    /// `head_dim - qk_rope_head_dim` dims (no RoPE applied here). In V4
    /// Flash this is `512 - 64 = 448`. Derived per audit §0
    /// "`qk_nope_head_dim`" + §3.
    ///
    /// If the config carries an explicit `qk_nope_head_dim` (legacy/V3-
    /// style), it's honored. Otherwise the derivation
    /// `head_dim - qk_rope_head_dim` is used. Both yield the same answer
    /// for real V4 Flash configs.
    pub fn qk_nope_head_dim(&self) -> usize {
        self.qk_nope_head_dim
            .unwrap_or_else(|| self.head_dim.saturating_sub(self.qk_rope_head_dim))
    }

    /// V4: same as `head_dim`. Kept as a method for symmetry with V3
    /// callers. Audit §3 (Q/K shape arithmetic).
    pub fn q_head_dim(&self) -> usize {
        self.head_dim
    }

    /// MQA broadcast factor: how many Q heads share each K head.
    /// V4 Flash: 64 (Q) / 1 (K) = 64. Audit §3.
    pub fn n_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads.max(1)
    }

    fn softmax_scale(&self) -> f32 {
        // V4 uses a PURE head_dim^-0.5 attention scale with NO YaRN mscale.
        // Reference inference/model.py:464 is `self.softmax_scale = head_dim ** -0.5`
        // and precompute_freqs_cis applies no mscale to the rope magnitudes
        // (torch.polar(ones, freqs)). The previous `* mscale * mscale` (~1.63x for
        // factor=16) over-scaled the attention logits -> softmax too sharp ->
        // over-peaked attention -> degraded generation quality. V2/V3 used mscale
        // here, but V4 does not. (RUN-161)
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// V4 per-layer compression dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressRatio {
    Standard,
    /// CSA — 4× compression with top-k token selection via indexer.
    Csa,
    /// HCA — 128× compression, dense MQA over compressed.
    Hca,
}

impl CompressRatio {
    pub fn ratio(self) -> usize {
        match self {
            Self::Standard => 1,
            Self::Csa => 4,
            Self::Hca => 128,
        }
    }
}

impl From<i32> for CompressRatio {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Standard,
            4 => Self::Csa,
            128 => Self::Hca,
            _ => Self::Standard,
        }
    }
}

/// V4 learned compressor — matches SGLang's `Compressor` at
/// `research/code/06_foundation/sglang/python/sglang/srt/layers/attention/
/// dsv4/compressor.py` lines 289-393. Loaded only on layers with
/// `compress_ratio != 0` (CSA + HCA). Audit §0 + §1 Pattern B/C/D.
///
/// Fields:
/// - `wkv_gate`: `[hidden_size, 2 * coff * head_dim]`. Fused `gate || val`
///   linear acting on the layer input `xs`. Unquantized (bf16) in SGLang.
/// - `norm`: RMSNorm over `head_dim`.
/// - `ape`: `[ratio, coff * head_dim]` absolute positional bias per
///   block position.
/// - `coff`: 2 for ratio=4 (CSA, overlap=true), 1 otherwise (HCA).
#[derive(Debug, Clone)]
pub struct V4Compressor {
    pub wkv_gate: Arc<dyn QuantMethod>,
    pub norm: RmsNorm,
    pub ape: Tensor,
    pub ratio: usize,
    pub head_dim: usize,
    pub coff: usize,
    pub hidden_size: usize,
}

impl V4Compressor {
    /// Auto-detect whether a checkpoint slice carries real compressor
    /// weights. Two layouts are supported (audit §0 + §1 Pattern B/C/D
    /// + §5 line 371 "Compressor wkv_gate fusion"):
    ///
    /// 1. **V4 native (post-fusion)**: a single `wkv_gate.weight` tensor of
    ///    shape `[2*coff*head_dim, hidden_size]`. This is what SGLang
    ///    produces after `load_weights → cache_compressor_weight` (which
    ///    pre-concats `wkv || wgate`); it's also the shape Arc's synthetic
    ///    test fixtures use.
    /// 2. **V4 native (pre-fusion)**: two separate tensors
    ///    `wkv.weight` and `wgate.weight`, each `[coff*head_dim,
    ///    hidden_size]`, exactly as published in the V4 Flash safetensors
    ///    index. The constructor concats `[kv, wgate]` along dim 0 to
    ///    reconstruct the fused linear in-place.
    ///
    /// Returns `true` when either layout is present. Callers that get
    /// `false` should fall back to [`V4Compressor::uniform`] to avoid
    /// silently dropping into a `DummyLayer` on the forward path.
    pub fn has_weights(vb: &ShardedVarBuilder) -> bool {
        vb.contains_tensor("wkv_gate.weight")
            || (vb.contains_tensor("wkv.weight") && vb.contains_tensor("wgate.weight"))
    }

    pub fn new(
        cfg: &DeepSeekV4Config,
        vb: ShardedVarBuilder,
        ratio: usize,
        head_dim: usize,
        real_device: &Device,
    ) -> Result<Self> {
        let overlap = ratio == 4;
        let coff = 1 + usize::from(overlap);

        // Two valid layouts. Both produce the same final `wkv_gate`
        // ReplicatedLayer with shape `[2*coff*head_dim, hidden_size]`.
        // See `V4Compressor::has_weights` for the audit references.
        let wkv_gate = if vb.contains_tensor("wkv_gate.weight") {
            // Pre-fused: load through the standard ReplicatedLayer path so
            // ISQ / quantization config still applies if present.
            ReplicatedLayer::new(
                cfg.hidden_size,
                2 * coff * head_dim,
                &None,
                false,
                vb.pp("wkv_gate"),
            )?
        } else if vb.contains_tensor("wkv.weight") && vb.contains_tensor("wgate.weight") {
            // V4-native dual tensors. SGLang's `cache_compressor_weight`
            // (deepseek_v4.py:1632-1663) does the exact same concat at
            // load time: `wkv_gate.weight = torch.cat([wkv, wgate], dim=0)`.
            // PyTorch Linear stores `[out, in]`, so dim 0 == output channel,
            // and the concat order is wkv-first then wgate (matches the
            // `split([coff*head_dim, coff*head_dim], dim=-1)` decode in
            // [`V4Compressor::forward_from_xs`]).
            let row = coff * head_dim;
            let wkv_w = vb.get((row, cfg.hidden_size), "wkv.weight")?;
            let wgate_w = vb.get((row, cfg.hidden_size), "wgate.weight")?;
            let fused = Tensor::cat(&[&wkv_w, &wgate_w], 0)?;
            // Keep the tiny [2*coff*head_dim, hidden] compressor weight in BF16
            // (unquantized) rather than ISQ-quantizing it. Quantizing saves
            // ~nothing (~4MB/layer) but its qtip quant transient OOMs on the
            // memory-tight tail layers of a single-H100 load (runs at ~78GB
            // resident with <2GB free). The async pool swallows that OOM,
            // silently breaking the compressor -> prefill (which uses the
            // compressor for seqlen>=ratio) then OOMs/fails. Decode and
            // sub-ratio-token prompts use the dense fallback and never touch it,
            // which masked the bug. See RUN-161.
            Arc::new(mistralrs_quant::UnquantLinear::new(
                mistralrs_quant::QuantMethodConfig::Unquantized(candle_nn::Linear::new(
                    fused, None,
                )),
            )?) as Arc<dyn QuantMethod>
        } else {
            // Caller is expected to have gated this case via
            // `V4Compressor::has_weights` — surface a clear error if not.
            candle_core::bail!(
                "V4Compressor::new: neither `wkv_gate.weight` nor (`wkv.weight` + \
                 `wgate.weight`) present in checkpoint. Call \
                 `V4Compressor::has_weights(&vb)` first and fall back to \
                 `V4Compressor::uniform` when it returns false."
            );
        };

        // Load norm and ape on the real device (not the ISQ CPU device)
        // to avoid device mismatch at inference time.
        let norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("norm"))?;
        let norm = RmsNorm::from_w(norm.weight().to_device(real_device)?, cfg.rms_norm_eps)?;

        let ape = if vb.contains_tensor("ape") {
            vb.get((ratio, coff * head_dim), "ape")?.to_device(real_device)?
        } else if vb.contains_tensor("ape.weight") {
            vb.get((ratio, coff * head_dim), "ape.weight")?.to_device(real_device)?
        } else {
            Tensor::zeros((ratio, coff * head_dim), DType::F32, real_device)?
        };

        Ok(Self {
            wkv_gate,
            norm,
            ape,
            ratio,
            head_dim,
            coff,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Synthetic-fallback constructor — yields an "averaging-by-ratio"
    /// compressor used only by unit tests when real V4 weights are absent.
    /// All-ones in → all-ones out, seq_len divided by `ratio`.
    pub fn uniform(ratio: usize, head_dim: usize, device: &Device) -> Result<Self> {
        let overlap = ratio == 4;
        let coff = 1 + usize::from(overlap);
        use mistralrs_quant::UnquantLinear;
        let dummy: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
            mistralrs_quant::QuantMethodConfig::Unquantized(candle_nn::Linear::new(
                Tensor::zeros((2 * coff * head_dim, 1), DType::F32, device)?,
                None,
            )),
        )?);
        let norm_weight = Tensor::ones(head_dim, DType::F32, device)?;
        let norm = RmsNorm::from_w(norm_weight, 1e-6)?;
        let ape = Tensor::zeros((ratio, coff * head_dim), DType::F32, device)?;

        Ok(Self {
            wkv_gate: dummy,
            norm,
            ape,
            ratio,
            head_dim,
            coff,
            hidden_size: 1,
        })
    }

    /// Legacy averaging path — averages `ratio` consecutive K or V entries.
    /// Used by unit tests; not on the real V4 inference path.
    pub fn forward(&self, kv: &Tensor) -> Result<Tensor> {
        let dims = kv.dims();
        if dims.len() != 4 {
            candle_core::bail!("V4Compressor::forward expects [B, H, T, D], got {:?}", dims);
        }
        let (b, h, t, d) = (dims[0], dims[1], dims[2], dims[3]);
        if t % self.ratio != 0 {
            candle_core::bail!("seq_len {t} not divisible by ratio {}", self.ratio);
        }
        if d != self.head_dim {
            candle_core::bail!(
                "V4Compressor::forward: head_dim mismatch {} vs configured {}",
                d,
                self.head_dim
            );
        }
        let t_new = t / self.ratio;
        let reshaped = kv.reshape((b, h, t_new, self.ratio, d))?;
        let summed = reshaped.sum(3)?;
        let inv = 1.0 / self.ratio as f64;
        summed.affine(inv, 0.0)
    }

    /// Real V4 compressor forward, operating directly on the layer input.
    ///
    /// Faithful port of the model's own reference `Compressor.forward`
    /// (prefill / `start_pos == 0` path) in `inference/model.py`. Returns the
    /// **pre-RoPE** compressed KV `[B, T_c, head_dim]` with `T_c = T / ratio`.
    /// Compress-θ RoPE on the last `qk_rope_head_dim` dims (at the strided
    /// compressed positions `[0, ratio, 2*ratio, …]`) is applied by the caller,
    /// which owns the rotary embedding.
    ///
    /// Algorithm (reference lines ~316-380):
    ///   - `wkv_gate(x)` → split into `value = wkv(x)` and `score = wgate(x)`
    ///     (the loader concatenates `cat([wkv, wgate], dim=0)`, so `split[0]`
    ///     is the value and `split[1]` is the score).
    ///   - group `ratio` consecutive tokens; `score += ape`.
    ///   - **overlap** (ratio == 4): re-window each entry over `2*ratio` tokens
    ///     spanning the previous + current group (`overlap_transform`).
    ///   - pool the value by a **softmax over the group axis**:
    ///     `pooled = (value * softmax(score)).sum(group)`.
    ///   - RMSNorm over `head_dim`.
    ///
    /// The previous implementation was wrong on every one of these steps
    /// (value/score swapped, `sigmoid(gate)*val` instead of the value, a plain
    /// sum instead of a softmax-weighted pool, no overlap, a bogus `coff`
    /// collapse) — which corrupted the compressed (distant-context) KV on all
    /// CSA/HCA layers and produced the long-context collapse. RUN-161.
    pub fn forward_from_xs(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        if dims.len() != 3 {
            candle_core::bail!(
                "V4Compressor::forward_from_xs expects [B, T, hidden], got {:?}",
                dims
            );
        }
        let (b, t, _hid) = (dims[0], dims[1], dims[2]);
        if t % self.ratio != 0 {
            candle_core::bail!(
                "V4Compressor: seq_len {t} not divisible by ratio {}",
                self.ratio
            );
        }
        let t_c = t / self.ratio;
        let out_dtype = xs.dtype();
        let dev = xs.device();
        let d = self.head_dim;
        let cd = self.coff * self.head_dim;

        // Compression runs in fp32 (reference: `x = x.float()`).
        let fused = self.wkv_gate.forward_autocast(xs)?.to_dtype(DType::F32)?;
        let split = fused.split(&[cd, cd], D::Minus1)?;
        let value = split[0].reshape((b, t_c, self.ratio, cd))?; // wkv
        let score = split[1].reshape((b, t_c, self.ratio, cd))?; // wgate
        let ape = self
            .ape
            .to_dtype(DType::F32)?
            .reshape((1, 1, self.ratio, cd))?;
        let score = score.broadcast_add(&ape)?;

        // Overlap (ratio == 4, coff == 2): re-window over 2*ratio neighbouring
        // tokens. The channel splits into [overlap-half (first d) | normal-half
        // (second d)]. Non-overlap layers keep the plain `ratio` group.
        let (value, score) = if self.coff == 2 {
            (
                Self::overlap_transform(&value, 0.0, b, t_c, self.ratio, d, dev)?,
                Self::overlap_transform(&score, f64::NEG_INFINITY, b, t_c, self.ratio, d, dev)?,
            )
        } else {
            (value, score)
        };

        // Softmax-weighted pool over the group axis (dim 2).
        let weights = candle_nn::ops::softmax(&score, 2)?;
        let pooled = (value * weights)?.sum(2)?; // [b, t_c, d]

        // Cast back to model dtype, then RMSNorm over head_dim.
        let pooled = pooled.to_dtype(out_dtype)?;
        let flat = pooled.reshape((b * t_c, d))?;
        let normed = flat.apply(&self.norm)?;
        normed.reshape((b, t_c, d))
    }

    /// Overlapping-window transform for the ratio-4 compressor (reference
    /// `Compressor.overlap_transform`). Maps a grouped tensor
    /// `[b, t_c, ratio, 2*d]` → `[b, t_c, 2*ratio, d]`: the last `ratio` slots
    /// take the current group's normal-half (channels `d..2d`), the first
    /// `ratio` slots take the **previous** group's overlap-half (channels
    /// `0..d`). The first group has no predecessor, so its leading `ratio`
    /// slots are filled with `fill` (`0` for values, `-inf` for scores so the
    /// softmax ignores them).
    #[allow(clippy::too_many_arguments)]
    fn overlap_transform(
        grouped: &Tensor,
        fill: f64,
        b: usize,
        t_c: usize,
        ratio: usize,
        d: usize,
        dev: &Device,
    ) -> Result<Tensor> {
        // Current group's normal-half → last `ratio` slots (all groups).
        let normal = grouped.narrow(D::Minus1, d, d)?.contiguous()?; // [b, t_c, ratio, d]
        // Current group's overlap-half (channels 0..d), shifted forward one
        // group so it lands in the *next* group's first `ratio` slots.
        let overlap = grouped.narrow(D::Minus1, 0, d)?; // [b, t_c, ratio, d]
        let pad = Tensor::full(fill, (b, 1, ratio, d), dev)?.to_dtype(grouped.dtype())?;
        let prev = if t_c > 1 {
            let shifted = overlap.narrow(1, 0, t_c - 1)?; // groups 0..t_c-2
            Tensor::cat(&[&pad, &shifted.contiguous()?], 1)?
        } else {
            pad
        };
        Tensor::cat(&[&prev, &normal], 2)
    }
}

/// Q projection — LoRA only (V4 Flash always sets `q_lora_rank=1024`).
/// `QProj::Plain` is preserved for tolerance of legacy non-LoRA Q variants
/// (synthetic tests, older V3-style fixtures).
enum QProj {
    Plain(Arc<dyn QuantMethod>),
    Lora {
        a: Arc<dyn QuantMethod>,
        norm: RmsNorm,
        b: Arc<dyn QuantMethod>,
    },
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Lora { a, norm, b } => {
                b.forward_autocast(&norm.forward(&a.forward_autocast(xs)?)?)
            }
            Self::Plain(lin) => lin.forward_autocast(xs),
        }
    }
}

/// V4 Attention layer — MQA with fused KV projection.
///
/// Audit §0 + §2 (SGLang `MQALayer` lines 172-360). The fundamental
/// shape per token is:
///   - `q`:  `[B, n_heads=64, T, head_dim=512]`
///   - `k`:  `[B, num_kv_heads=1, T, head_dim=512]` (broadcast at kernel)
///   - `v`:  same tensor as `k` (V4 absorbs V into K — audit §3).
struct Attention {
    /// Q LoRA: `wq_a` (hidden → q_lora_rank) → `q_norm` (RMSNorm over
    /// q_lora_rank) → `wq_b` (q_lora_rank → n_heads * head_dim).
    /// Audit §2 (SGLang lines 197-199).
    q: QProj,
    /// **Single fused** K/V projection: `hidden → head_dim`. ONE MQA head.
    /// Replaces V3's `kv_a_proj_with_mqa + kv_b_proj`. Audit §0 + §2
    /// (SGLang line 200).
    wkv: Arc<dyn QuantMethod>,
    /// RMSNorm over `head_dim=512` (NOT `kv_lora_rank`). Audit §0 + §2
    /// (SGLang line 201).
    kv_norm: RmsNorm,
    /// Grouped o_proj part A: `wo_a` shaped to project from
    /// `n_heads * head_dim / n_groups` → `n_groups * o_lora_rank` per the
    /// SGLang ColumnParallelLinear. Audit §0 + §2 (SGLang line 202).
    wo_a: Arc<dyn QuantMethod>,
    /// Grouped o_proj part B: `[n_groups * o_lora_rank, hidden_size]`.
    /// Audit §0 + §2 (SGLang line 203).
    wo_b: Arc<dyn QuantMethod>,
    /// RUN-161 throughput: cache of the dequantized + permuted grouped `wo_a`
    /// weight `(G, D, R)`. The grouped o_proj used to call `wo_a.dequantize_w()`
    /// (~67MB materialize) + a `.contiguous()` permute (another ~67MB) EVERY
    /// forward — for a constant weight. Profiled at ~69ms/token (27% of decode,
    /// the single biggest cost). Built once on first forward, reused after.
    wo_a_t_cache: std::sync::RwLock<Option<Tensor>>,
    /// RoPE — either `rope_theta`-based (standard layers 0/1/42) or
    /// `compress_rope_theta`-based (compress layers). Caller picks at
    /// construction. Audit §0 + §8 P1 item 11.
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    cfg: DeepSeekV4Config,
    /// RUN-161 decode-bug localization: this layer's index, used only to gate
    /// the `V4_TRACE` per-op tensor dumps to a single layer (see `v4_trace_dump`).
    dbg_layer_idx: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    /// Per-TP-rank Q head count.
    num_attention_heads: usize,
    /// Per-TP-rank KV head count (V4: 1 with TP=1).
    num_kv_heads: usize,
    /// CompressRatio for this layer (audit §1). Used by the dispatch in
    /// [`super::dsv4_attention::dsv4_attention`] (RUN-155).
    compress_ratio: CompressRatio,
    /// V4 compressor (loaded only on CSA + HCA layers). Consumed by the
    /// V4 hybrid attention dispatch (RUN-155).
    compressor: Option<V4Compressor>,
    /// V4 sliding-window size (used by CSA + HCA blending; audit §0).
    sliding_window: usize,
    /// V4 attn_sink — learned `[n_heads]` parameter, an extra softmax
    /// column. Stored here for residual_tensors() / IsqModel tracking;
    /// the active forward path threads the same value into
    /// `sdpa_params.sinks` so SDPA picks it up automatically. Audit §0 +
    /// §2 (SGLang line 204 + 288).
    #[allow(dead_code)]
    attn_sink: Option<Tensor>,
    /// V4 Lightning Indexer — only on CSA layers (compress_ratio == 4).
    /// Audit §0 + §2 (SGLang line 195).
    #[allow(dead_code)]
    indexer: Option<super::dsv4_indexer::V4Indexer>,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV4Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_kv_heads = cfg.num_key_value_heads.max(1);

        // ---- Q projection (LoRA only; V3 plain-Q kept for synth tests) ----
        // Audit §0 + §2 (SGLang lines 197-199, 313). Auto-detect V4 native
        // (`wq_a/wq_b`) vs HF (`q_a_proj/q_b_proj`).
        let q = match cfg.q_lora_rank {
            Some(lora_rank) => {
                let q_a_name = if vb.contains_tensor("wq_a.weight") {
                    "wq_a"
                } else {
                    "q_a_proj"
                };
                let q_b_name = if vb.contains_tensor("wq_b.weight") {
                    "wq_b"
                } else {
                    "q_b_proj"
                };
                let a = ReplicatedLayer::new(
                    cfg.hidden_size,
                    lora_rank,
                    &cfg.quantization_config,
                    cfg.attention_bias,
                    mapper.set_device(layer_idx, vb.pp(q_a_name), loading_isq),
                )?;
                // V4 native: `q_norm`. HF: `q_a_layernorm`.
                let q_norm_name = if vb.contains_tensor("q_norm.weight") {
                    "q_norm"
                } else {
                    "q_a_layernorm"
                };
                let norm = RmsNorm::new(
                    lora_rank,
                    cfg.rms_norm_eps,
                    mapper.set_device(layer_idx, vb.pp(q_norm_name), false),
                )?;
                let b = ColumnParallelLayer::new(
                    lora_rank,
                    cfg.num_attention_heads * head_dim,
                    &cfg.quantization_config,
                    false,
                    comm,
                    mapper.set_device(layer_idx, vb.pp(q_b_name), loading_isq),
                )?;
                QProj::Lora { a, norm, b }
            }
            None => QProj::Plain(ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                &cfg.quantization_config,
                false,
                comm,
                mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
            )?),
        };

        // ---- Fused K/V projection (SINGLE wkv: hidden → head_dim) ----
        // V4 native: `wkv` (SGLang line 200). HF-style legacy: `kv_proj`
        // or `kv_a_proj_with_mqa` (V3 fallback). Audit §0 + §5 lines 477-482.
        let wkv_name = if vb.contains_tensor("wkv.weight") {
            "wkv"
        } else if vb.contains_tensor("kv_proj.weight") {
            "kv_proj"
        } else {
            // V3-style legacy. Used by synthetic tests and any older V4
            // republish that flattens the fused projection.
            "kv_a_proj_with_mqa"
        };
        let wkv = ReplicatedLayer::new(
            cfg.hidden_size,
            head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            mapper.set_device(layer_idx, vb.pp(wkv_name), loading_isq),
        )?;

        // ---- KV norm over head_dim (not kv_lora_rank) ----
        // V4 native: `kv_norm` (SGLang line 201). HF: `kv_a_layernorm`.
        // Audit §5 lines 471-475.
        let kv_norm_name = if vb.contains_tensor("kv_norm.weight") {
            "kv_norm"
        } else {
            "kv_a_layernorm"
        };
        let kv_norm = RmsNorm::new(
            head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp(kv_norm_name), false),
        )?;

        // ---- Grouped o_proj LoRA (wo_a + wo_b) ----
        // Audit §0 + §5 line 484 + SGLang lines 202-203.
        //   wo_a: in=n_heads*head_dim/n_groups, out=n_groups*o_lora_rank
        //   wo_b: in=n_groups*o_lora_rank, out=hidden_size
        let o_lora_rank = cfg.o_lora_rank.unwrap_or(cfg.hidden_size);
        let o_groups = cfg.o_groups.unwrap_or(1);
        let o_inner = o_groups * o_lora_rank;

        // Auto-detect V4 native (`wo_a/wo_b`) vs HF (`o_a_proj/o_b_proj`).
        let has_native_o =
            vb.contains_tensor("wo_a.weight") || vb.contains_tensor("wo_b.weight");
        let has_lora_o = has_native_o
            || (vb.contains_tensor("o_a_proj.weight") && vb.contains_tensor("o_b_proj.weight"));
        let has_single_o = vb.contains_tensor("o_proj.weight");

        let (wo_a_name, wo_b_name) = if has_native_o {
            ("wo_a", "wo_b")
        } else {
            ("o_a_proj", "o_b_proj")
        };

        let (wo_a, wo_b) = if has_lora_o || !has_single_o {
            // V4 grouped LoRA path.
            let wo_a_in = cfg.num_attention_heads * head_dim / o_groups.max(1);
            let wo_a = ColumnParallelLayer::new(
                wo_a_in,
                o_inner,
                &cfg.quantization_config,
                cfg.attention_bias,
                comm,
                mapper.set_device(layer_idx, vb.pp(wo_a_name), loading_isq),
            )?;
            let wo_b = RowParallelLayer::new(
                o_inner,
                cfg.hidden_size,
                &cfg.quantization_config,
                cfg.attention_bias,
                comm,
                mapper.set_device(layer_idx, vb.pp(wo_b_name), loading_isq),
            )?;
            (wo_a, wo_b)
        } else {
            // V3-style single o_proj fallback (synth tests). Synthesize a
            // passthrough wo_a.
            let wo_a = ColumnParallelLayer::new(
                cfg.num_attention_heads * head_dim,
                cfg.num_attention_heads * head_dim,
                &cfg.quantization_config,
                false,
                comm,
                mapper.set_device(layer_idx, vb.pp(wo_a_name), loading_isq),
            )?;
            let wo_b = RowParallelLayer::new(
                cfg.num_attention_heads * head_dim,
                cfg.hidden_size,
                &cfg.quantization_config,
                cfg.attention_bias,
                comm,
                mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
            )?;
            (wo_a, wo_b)
        };

        // ---- Per-layer compress dispatch (audit §1) ----
        let ratio_int = cfg.layer_compress_ratio(layer_idx);
        let compress_ratio = CompressRatio::from(ratio_int);

        // ---- V4 compressor (CSA + HCA layers only) ----
        // Auto-detect dispatch: `V4Compressor::has_weights` accepts either
        // the post-fused `wkv_gate.weight` layout (synth / SGLang-cached)
        // or the V4-native pre-fused dual `wkv.weight` + `wgate.weight`
        // layout. Anything else falls back to the uniform-averaging stub so
        // old checkpoints continue to load. Audit §0 + §1 Pattern B/C/D.
        let compressor = if compress_ratio != CompressRatio::Standard {
            // Use the real GPU device (loading_isq=false) for non-quantized
            // tensors (ape, norm) to avoid device mismatch at inference time.
            let real_device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&Device::Cpu);
            let comp_vb = mapper.set_device(layer_idx, vb.pp("compressor"), loading_isq);
            let comp = if V4Compressor::has_weights(&comp_vb) {
                V4Compressor::new(cfg, comp_vb, ratio_int as usize, head_dim, real_device)?
            } else {
                V4Compressor::uniform(ratio_int as usize, head_dim, real_device)?
            };
            Some(comp)
        } else {
            None
        };

        // ---- attn_sink: learned [n_heads] fp32 parameter ----
        // Audit §0 + §2 (SGLang line 288). One scalar per head added as an
        // extra softmax column for the sink token. Optional — synth
        // weights may omit.
        let attn_sink = if vb.contains_tensor("attn_sink") {
            Some(
                mapper
                    .set_device(layer_idx, vb.clone(), false)
                    .get_with_hints_dtype(
                        cfg.num_attention_heads,
                        "attn_sink",
                        Default::default(),
                        DType::F32,
                    )?,
            )
        } else {
            None
        };

        // ---- Lightning Indexer (CSA layers only) ----
        // Audit §0 + §1 Pattern B/D. HCA layers don't have an indexer.
        let indexer = if compress_ratio == CompressRatio::Csa {
            let idx_vb = mapper.set_device(layer_idx, vb.pp("indexer"), loading_isq);
            if idx_vb.contains_tensor("wq_b.weight") {
                let device = mapper
                    .device_for(layer_idx, loading_isq)
                    .unwrap_or(&Device::Cpu);
                match super::dsv4_indexer::V4Indexer::new(cfg, idx_vb, device, loading_isq) {
                    Ok(idx) => Some(idx),
                    Err(e) => {
                        tracing::warn!(
                            "V4 CSA layer {layer_idx}: indexer load failed ({e}), \
                             falling back to dense-over-compressed"
                        );
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // Pre-shape sinks for SDPA: [n_heads] → [1, n_heads, 1, 1].
        // RUN-161 ablation: ARC_DISABLE_SINK=1 drops the attention sink to test
        // whether sink-mass domination (head_dim=512 falls into the unfused
        // softmax_with_sinks path on every layer) is collapsing attention output.
        let sinks_for_sdpa = if std::env::var_os("ARC_DISABLE_SINK").is_some() {
            None
        } else if let Some(ref s) = attn_sink {
            Some(s.reshape((1, cfg.num_attention_heads, 1, 1))?)
        } else {
            None
        };

        let world_size = comm.world_size();
        let num_attention_heads = (cfg.num_attention_heads / world_size).max(1);
        let num_kv_heads_tp = (num_kv_heads / world_size).max(1);
        // MQA broadcast factor: how many Q heads share one KV head.
        // Audit §3.
        let n_kv_groups = (num_attention_heads / num_kv_heads_tp).max(1);

        Ok(Self {
            q,
            wkv,
            kv_norm,
            wo_a,
            wo_b,
            rotary_emb,
            cfg: cfg.clone(),
            dbg_layer_idx: layer_idx,
            paged_attn,
            num_attention_heads,
            num_kv_heads: num_kv_heads_tp,
            sdpa_params: SdpaParams {
                n_kv_groups,
                softcap: None,
                softmax_scale: cfg.softmax_scale(),
                sliding_window: if compress_ratio != CompressRatio::Standard {
                    Some(cfg.sliding_window)
                } else {
                    None
                },
                sinks: sinks_for_sdpa,
            },
            compress_ratio,
            compressor,
            sliding_window: cfg.sliding_window,
            attn_sink,
            indexer,
            wo_a_t_cache: std::sync::RwLock::new(None),
        })
    }

    /// Apply RoPE in-place to the last `qk_rope_head_dim` dims of each
    /// Q-head and K-head's `head_dim`-vector. The first
    /// `qk_nope_head_dim` dims are left untouched. Audit §0 + §3.
    fn apply_rope_inplace(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let qk_nope = self.cfg.qk_nope_head_dim();
        let qk_rope = self.cfg.qk_rope_head_dim;
        let head_dim = self.cfg.head_dim;
        debug_assert_eq!(qk_nope + qk_rope, head_dim);

        // Split-rotate-recombine. Candle-backed equivalent of SGLang's
        // `fused_norm_rope_inplace`. Audit §0 + §8 P0 item 8.
        let q_split = q.split(&[qk_nope, qk_rope], D::Minus1)?;
        let q_nope = q_split[0].clone();
        let mut q_pe = q_split[1].clone();

        let k_split = k.split(&[qk_nope, qk_rope], D::Minus1)?;
        let k_nope = k_split[0].clone();
        let mut k_pe = k_split[1].clone();

        (q_pe, k_pe) = self.rotary_emb.forward(&q_pe, &k_pe, seqlen_offsets)?;

        let q_out = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?.contiguous()?;
        let k_out = Tensor::cat(&[&k_nope, &k_pe], D::Minus1)?.contiguous()?;
        Ok((q_out, k_out))
    }

    /// Compute the compressed (distant-context) KV for a CSA/HCA layer from the
    /// compressor-input history, with compress-θ RoPE applied at the strided
    /// compressed positions. Returns `[B, 1, T_c, head_dim]`, or `None` when:
    ///   * this is a Standard layer / no compressor loaded, or
    ///   * the history is shorter than one `ratio` block (`T_c == 0`) — in which
    ///     case the sliding-window branch alone covers the whole context.
    ///
    /// The largest `ratio`-multiple prefix of the history is compressed; the
    /// remaining `< ratio` recent tokens are covered by the raw sliding-window
    /// branch in [`super::dsv4_attention::dsv4_attention`]. Reference
    /// `inference/model.py` `Attention.forward` (the `kv_compress` path).
    fn compressed_kv(&self, xs_hist: &Tensor) -> Result<Option<Tensor>> {
        if self.compress_ratio == CompressRatio::Standard {
            return Ok(None);
        }
        let Some(compressor) = self.compressor.as_ref() else {
            return Ok(None);
        };
        let ratio = self.compress_ratio.ratio();
        let t_xs = xs_hist.dim(1)?;
        let t_trunc = (t_xs / ratio) * ratio;
        if t_trunc == 0 {
            return Ok(None);
        }
        let xs_trunc = if t_trunc == t_xs {
            xs_hist.clone()
        } else {
            xs_hist.narrow(1, 0, t_trunc)?
        };
        // [B, T_c, head_dim] (pre-RoPE) → [B, 1, T_c, head_dim].
        let comp = compressor.forward_from_xs(&xs_trunc)?.unsqueeze(1)?;
        let t_c = comp.dim(2)?;
        // Compressed entry j sits at absolute position j*ratio. Apply the
        // layer's (compress-θ) RoPE to the last qk_rope_head_dim dims there.
        // NOTE: builds a small arange each call (a host/device sync); fine on
        // the correctness-first dense path — the long-context sparse-gather
        // kernel is the place to precompute this.
        let dev = comp.device();
        let positions = (Tensor::arange(0u32, t_c as u32, dev)?.to_dtype(DType::F32)?
            * (ratio as f64))?
        .to_dtype(DType::U32)?;
        let comp =
            self.rotary_emb
                .forward_at_positions(&comp, self.cfg.qk_rope_head_dim, &positions)?;
        Ok(Some(comp))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        xs_hist_cache: Option<&mut KvCache>,
        metadata: Option<(
            (Tensor, Tensor, Option<Tensor>, Option<Tensor>),
            &PagedAttentionInputMetadata,
        )>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;
        let head_dim = self.cfg.head_dim;
        let mdev = xs.device().clone();

        v4_nan_dbg(xs, "attn.in");
        v4_trace_dump(self.dbg_layer_idx, xs, "00_xs_in");
        // The position values are THE crux of the decode bug: dump them so the
        // diff can compare the offset decode used (row N) vs prefill (0..N).
        v4_trace_text(
            self.dbg_layer_idx,
            "00_meta",
            &format!("seq_len={seq_len}\nseqlen_offsets={seqlen_offsets:?}\n"),
        );

        // R2/R3: the compressor (forward_from_xs) must see the *history* of
        // the attention input during decode, not just the current token. The
        // history is a per-layer, per-SEQUENCE cache slot in the model's
        // NormalCache (indices `num_hidden_layers..`), threaded in by
        // `DeepSeekV4::forward`. Because it lives in the same cache vector as
        // the KV entries, the engine's NormalCacheManager clone_in/clone_out
        // batches it along dim 0 and splits it back per sequence — so
        // multi-sequence batches (Arc Boost `n_votes` voting chains) each keep
        // their own history instead of colliding in one shared buffer (R3;
        // previously a per-model `Mutex<SingleCache>` that crashed/mis-sliced
        // on batch>1 with divergent chains). Storage: `k` holds the xs history
        // `[B, T, hidden]` (seq dim 1); `v` is a `[B, T, 1]` zero marker kept
        // in lockstep because the cache managers require both sides populated.
        // Reset at sequence start (all seqlen_offsets == 0), same as before;
        // the in-layer reset also keeps direct `model.forward` callers
        // (tests / SDK) correct without a cache manager. The q/k/v
        // projections below still use the current `xs`.
        let xs_for_compressor = match xs_hist_cache {
            Some(hist) => {
                if seqlen_offsets.iter().all(|&o| o == 0) {
                    hist.reset();
                }
                let xs3 = xs.contiguous()?;
                let marker = Tensor::zeros((bs, seq_len, 1), xs3.dtype(), xs3.device())?;
                let (hist_xs, _marker) = hist.append(&xs3, &marker)?;
                hist_xs
            }
            None => xs.clone(),
        };
        // 1. Q projection (LoRA). [B, T, hidden] → [B, T, n_heads*head_dim]
        //    → reshape to [B, n_heads, T, head_dim]. Audit §3.
        let q = timed_mla(0, &mdev, || self.q.forward(xs))?;
        v4_nan_dbg(&q, "attn.q_proj");
        let q = q
            .reshape((bs, seq_len, self.num_attention_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        // V4: per-head RMS-normalize Q over head_dim before RoPE. Reference
        // inference/model.py:498 `q *= rsqrt(q.square().mean(-1)+eps)`. This is
        // SEPARATE from q_norm (which normalizes q_lora_rank inside self.q).
        // Missing it leaves Q ~30x too small -> near-uniform attention scores
        // -> the model cannot attend -> word-salad output (RUN-161).
        let q = {
            let inv_rms = q
                .sqr()?
                .mean_keepdim(candle_core::D::Minus1)?
                .affine(1.0, self.cfg.rms_norm_eps)?
                .recip()?
                .sqrt()?;
            q.broadcast_mul(&inv_rms)?
        };
        v4_stat_dbg(&q, "attn.q_normed");
        v4_trace_dump(self.dbg_layer_idx, &q, "10_q_normed");

        // 2. K/V projection: single fused wkv. [B, T, hidden] → [B, T,
        //    head_dim] → kv_norm → reshape to [B, num_kv_heads=1, T,
        //    head_dim]. Audit §0 + §3.
        let kv_raw = timed_mla(1, &mdev, || self.wkv.forward_autocast(xs))?;
        v4_nan_dbg(&kv_raw, "attn.wkv");
        let kv_normed = self.kv_norm.forward(&kv_raw)?;
        v4_nan_dbg(&kv_normed, "attn.kv_norm");
        let k = kv_normed
            .reshape((bs, seq_len, self.num_kv_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // 3. RoPE applied in-place to the last qk_rope_head_dim dims of
        //    each Q-head and K-head's head_dim-vector. Audit §0 + §3.
        let (q, k) = self.apply_rope_inplace(&q, &k, seqlen_offsets)?;
        v4_nan_dbg(&q, "attn.q_rope");
        v4_nan_dbg(&k, "attn.k_rope");
        v4_trace_dump(self.dbg_layer_idx, &q, "20_q_rope");
        v4_trace_dump(self.dbg_layer_idx, &k, "21_k_rope");

        // V4 QAT: FP8-simulate the non-rope dims of K (reference model.py:506
        // `act_quant(kv[..., :-rd], 64, ..., inplace=True)`). The model was
        // trained with the KV non-rope dims round-tripped through block-wise
        // FP8; feeding full BF16 is out-of-distribution. V=K so v inherits it.
        let k = act_quant_kv_nope(&k, self.cfg.qk_rope_head_dim)?;
        v4_stat_dbg(&k, "attn.k_actquant");
        v4_trace_dump(self.dbg_layer_idx, &k, "30_k_actquant");

        // 4. V4: K and V come from the same wkv tensor. The kernel treats
        //    the wkv output as both K (for scores) and V (for weighted
        //    sum). This is the MLA "absorb V into K" trick at scale.
        //    Audit §3 ("absorbed the MLA split into a single fused output").
        //
        // Use copy() not clone(): clone() shares the storage Arc<RwLock>, so k
        // and v alias the same storage. PagedAttention's reshape_and_cache is an
        // in-place op that write-locks and read-locks that storage -> RwLock
        // self-deadlock (the dummy-run hang). copy() gives v its own storage.
        // Cheap here: MQA means 1 KV head. (RUN-161)
        let v = k.copy()?;

        // 5. Attention dispatch (RUN-155 + RUN-167).
        //
        // V4 has two cache backends — a plain in-process `KvCache` (the
        // None paged_attn arm) and the engine-level `PagedAttention` (the
        // Some paged_attn arm). EVERY layer routes through the V4
        // `dsv4_attention` dispatch: `Standard` layers are sliding-window +
        // attn_sink (the reference SWA-onlys ratio-0 layers — SGLang
        // `deepseek_v4_backend.py` attends ratio-0 with `swa_page_indices`
        // only; vLLM `swa_only = compress_ratio <= 1`), and `Csa | Hca`
        // layers add the compressed (distant-context) branch in the same
        // single softmax. The previous wiring sent Standard layers through
        // dense causal SDPA / the dense paged kernel, which fed the
        // window-trained layers 0/1/42 unseen relative distances and
        // collapsed generation once the context crossed the 128-token
        // window (the long-ctx repetition failure; window-only ablation
        // unaffected because these layers are not gated by it).
        //
        // RUN-167 fixes the PagedAttention + CSA/HCA combination: the
        // previous wiring (`kv_cache.append`, then dsv4_attention) wrote
        // into the in-process KvCache that the engine ignores under PA, so
        // the next decode step saw a stale cache. The fix routes through
        // `PagedAttention::cache_write_and_gather`, which writes the new
        // K/V into the paged storage via `reshape_and_cache` and gathers
        // the full (cached + new) sequence back via `gather_kv_cache`. The
        // gathered K/V then feeds `dsv4_attention` exactly like the plain
        // SDPA path. Storing the *compressed* K/V in the paged cache
        // (HBM-saving Tier-B) is left for a follow-up since slot_mapping +
        // block_tables would have to be ratio-aware end-to-end.
        let dsv4_cfg = super::dsv4_attention::Dsv4AttentionConfig {
            compress_ratio: self.compress_ratio,
            sliding_window: self.sliding_window,
        };
        // Faithful V4: the compressed (distant-context) KV, with compress-θ
        // RoPE at the strided compressed positions. `dsv4_attention` runs a
        // single softmax over the union [raw sliding-window KV ++ compressed
        // KV] + attn_sink. `None` on Standard layers / short history (the
        // window alone covers the context). RUN-161.
        //
        // RUN-161 #27 ablation: ARC_V4_WINDOW_ONLY=1 forces the compressed
        // branch off so attention degenerates to pure sliding-window. Used to
        // isolate whether long-ctx sustained-generation degradation lives in
        // the compressed branch (coherent window-only ⇒ yes) vs elsewhere.
        let compressed_kv = if std::env::var_os("ARC_V4_WINDOW_ONLY").is_some() {
            None
        } else {
            self.compressed_kv(&xs_for_compressor)?
        };
        let mut attn_out = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                // PagedAttention (all compress ratios): write to paged
                // cache, gather the full context, then run the V4 dispatch
                // over the gathered K/V. RUN-167. Standard layers also go
                // through `dsv4_attention` — they need the sliding-window
                // mask, which the dense paged kernel cannot apply.
                Some(((mut key_cache, mut value_cache, _, _), input_metadata)) => {
                    let (k_full, v_full) = paged_attn.cache_write_and_gather(
                        &k,
                        &v,
                        &mut key_cache,
                        &mut value_cache,
                        input_metadata,
                        q.dtype(),
                    )?;
                    super::dsv4_attention::dsv4_attention(
                        &q,
                        &k_full,
                        &v_full,
                        compressed_kv.as_ref(),
                        attention_mask,
                        flash_params,
                        &self.sdpa_params,
                        dsv4_cfg,
                    )?
                }
                // Dummy / imatrix path: no paged cache, so just run the
                // dispatch over the live K/V (matches the plain-SDPA arm).
                None => {
                    assert!(attention_mask.is_some());
                    super::dsv4_attention::dsv4_attention(
                        &q,
                        &k,
                        &v,
                        compressed_kv.as_ref(),
                        attention_mask,
                        flash_params,
                        &self.sdpa_params,
                        dsv4_cfg,
                    )?
                }
            },
            // RUN-161 2c: CUDA-graph-capturable decode. When graph-mode device
            // positions are set, write the new K/V at the device slot and read a
            // FIXED `sliding_window`-wide window so every decode step is
            // shape-identical (the caching allocator hits, the graph replays).
            // C == sliding_window forces dsv4_attention's dense SDPA path. The
            // unwritten tail slots are masked via the graph-mode length mask.
            None if crate::layers::has_graph_mode_positions() && seq_len == 1 => {
                let position = crate::layers::graph_mode_positions()
                    .ok_or_else(|| candle_core::Error::Msg("graph positions unset".into()))?
                    .to_dtype(candle_core::DType::U32)?;
                let cap = self.sliding_window.max(1);
                let (k_full, v_full) = kv_cache.append_graph(&k, &v, &position, cap)?;
                // Use ONLY the fixed-width graph mask (matches the C-wide K).
                // The eager `attention_mask` is kv_len-wide (growing) and would
                // both mismatch the fixed window and break shape-constancy.
                // `dsv4_attention` now folds whatever it is handed into its own
                // union mask (it used to discard it), so wiring
                // `set_graph_mode_mask` is all that remains for this path: until
                // then `graph_mode_mask()` is `None` and the unwritten tail slots
                // are attended as zero-padding (finite, not yet correct).
                let gmask = crate::layers::graph_mode_mask();
                super::dsv4_attention::dsv4_attention(
                    &q,
                    &k_full,
                    &v_full,
                    compressed_kv.as_ref(),
                    gmask.as_ref(),
                    flash_params,
                    &self.sdpa_params,
                    dsv4_cfg,
                )?
            }
            None => {
                let (k_cached, v_cached) = kv_cache.append(&k, &v)?;
                // Cache read-back: in decode this is prefill's K (0..N-1) + the
                // new token's K. Diff vs prefill's freshly-computed K splits a
                // cache-storage bug (old rows differ) from a new-token position
                // bug (only the last row differs).
                v4_trace_dump(self.dbg_layer_idx, &k_cached, "40_k_cached");
                v4_trace_dump(self.dbg_layer_idx, &v_cached, "41_v_cached");
                super::dsv4_attention::dsv4_attention(
                    &q,
                    &k_cached,
                    &v_cached,
                    compressed_kv.as_ref(),
                    attention_mask,
                    flash_params,
                    &self.sdpa_params,
                    dsv4_cfg,
                )?
            }
        };

        // 6. Inverse RoPE on attention output's last 64 dims is implicit
        //    here because K and V come from the same source post-RoPE: the
        //    rotation on V cancels with the rotation on K at attention
        //    time. SGLang's explicit `inverse=True` rope call (audit §2,
        //    lines 619-625) applies only when K and V are stored
        //    separately in the FlashMLA-style paged cache. Tracked in
        //    audit §8 P1 item 13.

        // 7. Reshape attention output → flatten heads → grouped o_proj
        //    LoRA. attn_out shape: [B, n_heads, T, head_dim].
        //    Audit §8 P1 item 12 — per-group einsum("tgd,grd->tgr").
        v4_nan_dbg(&attn_out, "attn.sdpa_out");
        // Normalize to [B, H, T, head_dim]. Two input layouts reach here:
        //   * PagedAttention (Standard layers) returns 3-D [B*T, H, head_dim].
        //   * SDPA (`Sdpa::run_attention`) returns 4-D [B, H, T, head_dim] in
        //     BOTH prefill (mask) and decode (no-mask) — every backend
        //     normalizes to [B, H, T, D] (cuBLASLt reshapes to it; flash/metal
        //     transpose to it). The previous code transposed the no-mask decode
        //     output assuming [B, T, H, D]; at seq_len=1 that turned the correct
        //     [B, H, 1, D] into [B, 1, H, D], scrambling the head/dim order fed
        //     to o_proj and collapsing every decode step (RUN-161 decode bug).
        // Without the 3-D case, paged decode transposes [B*T,H,D] -> [.,D,H] and
        // hits forward_inverse_tail with a rank-3 tensor -> "unexpected rank,
        // expected: 4, got: 3". (RUN-161)
        let attn_out_bhtd = if attn_out.rank() == 3 {
            let (_bt, h, hd) = attn_out.dims3()?;
            attn_out.reshape((bs, seq_len, h, hd))?.transpose(1, 2)?
        } else {
            attn_out.clone()
        };
        // V4: inverse-RoPE the last qk_rope_head_dim dims of the attention
        // output. V=K carries the key's RoPE on those dims, so the output must
        // be de-rotated (reference inference/model.py:534). Missing this leaves
        // the value content wrongly position-rotated -> attention output is
        // corrupted on rope dims -> incoherent generation (RUN-161).
        // Normalized [B,H,T,D] SDPA output BEFORE the position-dependent inverse
        // RoPE — diffing this vs 60_after_invtail isolates whether the bug is the
        // attention kernel itself or the inverse-tail de-rotation.
        v4_trace_dump(self.dbg_layer_idx, &attn_out_bhtd, "55_sdpa_bhtd");
        let attn_out_bhtd = self.rotary_emb.forward_inverse_tail(
            &attn_out_bhtd,
            self.cfg.qk_rope_head_dim,
            seqlen_offsets,
        )?;
        v4_trace_dump(self.dbg_layer_idx, &attn_out_bhtd, "60_after_invtail");
        // -> [B, T, H*head_dim]
        attn_out = attn_out_bhtd.transpose(1, 2)?.reshape((bs, seq_len, ()))?;

        let o_groups = self.cfg.o_groups.unwrap_or(1);
        let out = timed_mla(2, &mdev, || {
        let inner = if o_groups > 1 {
            // Grouped o_proj LoRA: each of `o_groups` head-groups gets its
            // own slice of wo_a.  wo_a weight is (G*R, D) where G=o_groups,
            // R=o_lora_rank, D=n_heads*head_dim/G.
            //
            // einsum("tgd,grd->tgr", attn_grouped, wo_a_grouped)
            //   = bmm(attn_grouped.permute(1,0,2), wo_a_grouped.transpose(1,2))
            let n = attn_out.dim(candle_core::D::Minus1)?;
            let per_group = n / o_groups;
            let tokens = bs * seq_len;

            // (bs*seq_len, G, D)
            let attn_grouped = attn_out.reshape((tokens, o_groups, per_group))?;

            // RUN-161: cached dequantized+permuted wo_a as (G, D, R). The weight
            // is constant — dequantizing + permuting it every forward was ~69ms/
            // token (27% of decode). Build once, reuse.
            let wo_a_t = {
                let cached = self.wo_a_t_cache.read().unwrap();
                if let Some(t) = cached.as_ref() {
                    t.clone()
                } else {
                    drop(cached);
                    let wo_a_w = self.wo_a.dequantize_w()?; // (G*R, D)
                    let o_lora_rank = wo_a_w.dim(0)? / o_groups;
                    let wo_a_grouped = wo_a_w.reshape((o_groups, o_lora_rank, per_group))?; // (G, R, D)
                    let t = wo_a_grouped.permute((0, 2, 1))?.contiguous()?; // (G, D, R)
                    *self.wo_a_t_cache.write().unwrap() = Some(t.clone());
                    t
                }
            };
            let o_lora_rank = wo_a_t.dim(2)?; // R

            // bmm: (G, tokens, D) @ (G, D, R) → (G, tokens, R)
            let attn_perm = attn_grouped.permute((1, 0, 2))?.contiguous()?;
            let inner_perm = attn_perm.matmul(&wo_a_t)?;

            // (G, tokens, R) → (tokens, G*R)
            inner_perm
                .permute((1, 0, 2))?
                .contiguous()?
                .reshape((tokens, o_groups * o_lora_rank))?
                .reshape((bs, seq_len, o_groups * o_lora_rank))?
        } else {
            self.wo_a.forward_autocast(&attn_out)?
        };
        v4_nan_dbg(&inner, "attn.wo_a");
        let out = self.wo_b.forward_autocast(&inner)?;
        v4_nan_dbg(&out, "attn.wo_b");
        Ok(out)
        })?;
        v4_trace_dump(self.dbg_layer_idx, &out, "70_o_out");
        Ok(out)
    }
}

/// V4 MoE gate. Inherits V3's NoAuxTc / Greedy / GroupLimitedGreedy dispatch
/// and adds support for `ScoringFunc::SqrtSoftplus` (V4 default — audit §0
/// + §5).
struct MoeGate {
    weight: Tensor,
    cfg: DeepSeekV4Config,
    top_k: usize,
    n_routed_experts: usize,
    e_score_correction_bias: Option<Tensor>,
    /// Hash-routing table for layers `< num_hash_layers` (V4: first 3). Shape
    /// `[vocab_size, top_k]` (I64). For these layers the reference `Gate`
    /// (inference/model.py) selects experts by a fixed token-id lookup
    /// `indices = tid2eid[input_ids]` instead of score top-k; routing weights
    /// still come from the (unbiased) gate scores at those indices. `None` for
    /// score-routed layers (which carry `gate.bias` instead). Audit §0 + §5
    /// line 458.
    tid2eid: Option<Tensor>,
}

impl MoeGate {
    fn new(
        cfg: &DeepSeekV4Config,
        vb: ShardedVarBuilder,
        n_routed_experts: usize,
        layer_idx: usize,
    ) -> Result<Self> {
        let weight = vb.get((n_routed_experts, cfg.hidden_size), "weight")?;
        let top_k = cfg.num_experts_per_tok.unwrap_or(6);
        // Hash routing: layers `< num_hash_layers` ship a fixed token-id ->
        // expert table (`gate.tid2eid`, [vocab_size, top_k]) and NO bias. Load
        // it so `forward` can dispatch hash vs. score routing per layer.
        // Reference: inference/model.py `Gate.hash = layer_id < n_hash_layers`.
        let tid2eid = if layer_idx < cfg.num_hash_layers {
            Some(vb.get_with_hints_dtype(
                (cfg.vocab_size, top_k),
                "tid2eid",
                Default::default(),
                DType::I64,
            )?)
        } else {
            None
        };
        // V4 native publishes the noaux_tc bias as `gate.bias`; HF as
        // `gate.e_score_correction_bias`. Audit §2 (SGLang line 361
        // remap rule).
        let e_score_correction_bias = if matches!(cfg.topk_method, TopkMethod::NoAuxTc) {
            let name = if vb.contains_tensor("bias") {
                Some("bias")
            } else if vb.contains_tensor("e_score_correction_bias") {
                Some("e_score_correction_bias")
            } else {
                // V4 Flash uses TD-MoE (tid2eid) routing — no correction bias.
                None
            };
            name.map(|n| {
                vb.get_with_hints_dtype(
                    n_routed_experts,
                    n,
                    Default::default(),
                    DType::F32,
                )
            })
            .transpose()?
        } else {
            None
        };
        // RUN-161 diagnostic: confirm whether the noaux_tc selection bias is
        // actually loaded for score-routed layers. If it is None on a
        // score-routed layer (layer_idx >= num_hash_layers), expert selection
        // ignores the trained bias and picks the wrong top-k → collapse.
        if std::env::var_os("ARC_COLLAPSE").is_some() {
            match &e_score_correction_bias {
                Some(b) => {
                    let stats = b
                        .to_dtype(DType::F32)
                        .and_then(|t| t.abs())
                        .and_then(|t| t.mean_all())
                        .and_then(|t| t.to_scalar::<f32>());
                    eprintln!(
                        "ARC_ROUTEBIAS [L{layer_idx}] gate.bias LOADED absmean={:?} hash={}",
                        stats,
                        layer_idx < cfg.num_hash_layers
                    );
                }
                None => eprintln!(
                    "ARC_ROUTEBIAS [L{layer_idx}] gate.bias = NONE hash={}",
                    layer_idx < cfg.num_hash_layers
                ),
            }
        }
        Ok(Self {
            weight,
            cfg: cfg.clone(),
            top_k,
            n_routed_experts,
            e_score_correction_bias,
            tid2eid,
        })
    }

    fn forward(&self, xs: &Tensor, input_ids: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (bs, seq_len, h) = xs.dims3()?;
        let xs = xs.reshape(((), h))?;
        let logits = xs
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.weight.t()?.to_dtype(DType::F32)?)?;
        let scores = match self.cfg.scoring_func {
            ScoringFunc::Softmax => candle_nn::ops::softmax_last_dim(&logits)?,
            ScoringFunc::Sigmoid => candle_nn::ops::sigmoid(&logits)?,
            // V4: sqrt(softplus(x)). Stable formulation:
            // softplus(x) = max(x, 0) + log(1 + exp(-|x|)).
            // Audit §8 P1 item 14.
            ScoringFunc::SqrtSoftplus => {
                let max0 = logits.maximum(&logits.zeros_like()?)?;
                let abs = logits.abs()?;
                let softplus = (max0 + ((abs.neg()?.exp()? + 1.0)?.log()?))?;
                softplus.sqrt()?
            }
        };

        let (mut topk_weight, topk_idx) = if let Some(tid2eid) = &self.tid2eid {
            // Hash routing (layers < num_hash_layers): experts are a fixed
            // per-token lookup, NOT score top-k. Reference Gate.forward:
            //   indices = tid2eid[input_ids]
            //   weights = original_scores.gather(1, indices)
            // `scores` here is the (unbiased) sqrtsoftplus output, matching the
            // reference's `original_scores`. input_ids[B*T] aligns row-wise with
            // xs.reshape((B*T, h)) (both row-major over B then T).
            let input_ids = input_ids.ok_or_else(|| {
                candle_core::Error::Msg(
                    "DeepSeek-V4 hash-routing MoE layer requires input_ids; \
                     caller did not thread token ids to the gate."
                        .to_string(),
                )
            })?;
            let ids = input_ids
                .reshape(((),))?
                .to_dtype(DType::U32)?
                .to_device(tid2eid.device())?;
            // [B*T, top_k] expert indices (cast to U32 to match the score path /
            // downstream expert gather).
            let topk_idx = tid2eid.index_select(&ids, 0)?.to_dtype(DType::U32)?;
            let topk_weight = scores.gather(&topk_idx, 1)?;
            (topk_weight, topk_idx)
        } else {
            match self.cfg.topk_method {
            TopkMethod::Greedy => {
                let TopKOutput { values, indices } = scores.topk_unsorted(self.top_k)?;
                (values, indices)
            }
            TopkMethod::NoAuxTc => {
                let scores_flat = scores.reshape((bs * seq_len, ()))?;
                let scores_for_choice = if let Some(bias) = &self.e_score_correction_bias {
                    scores_flat.broadcast_add(&bias.unsqueeze(0)?)?
                } else {
                    // V4 Flash: no correction bias (TD-MoE routing), use raw scores.
                    scores_flat
                };
                // V4 Flash config does NOT publish `n_group` (audit §5 +
                // /tmp/v4_flash_config.json). When n_group==1 the grouping
                // degenerates; fall back to a flat top-k over all experts.
                if self.cfg.n_group > 1 {
                    let group_scores = scores_for_choice
                        .reshape((bs * seq_len, self.cfg.n_group, ()))?
                        .topk(2)?
                        .values
                        .sum(D::Minus1)?;
                    let group_idx = group_scores.topk(self.cfg.topk_group)?.indices;
                    let mut group_mask = group_scores.zeros_like()?;
                    group_mask = group_mask.scatter_add(
                        &group_idx,
                        &group_idx.ones_like()?.to_dtype(group_mask.dtype())?,
                        1,
                    )?;
                    let score_mask = group_mask
                        .unsqueeze(D::Minus1)?
                        .expand((
                            bs * seq_len,
                            self.cfg.n_group,
                            self.n_routed_experts / self.cfg.n_group,
                        ))?
                        .reshape((bs * seq_len, ()))?;
                    let tmp_scores = scores_for_choice.broadcast_mul(&score_mask)?;
                    let topk_idx = tmp_scores.topk(self.top_k)?.indices;
                    (scores.gather(&topk_idx, 1)?, topk_idx)
                } else {
                    let topk_idx = scores_for_choice.topk(self.top_k)?.indices;
                    (scores.gather(&topk_idx, 1)?, topk_idx)
                }
            }
            TopkMethod::GroupLimitedGreedy => {
                let group_scores = scores
                    .reshape((bs * seq_len, self.cfg.n_group, ()))?
                    .max(D::Minus1)?;
                let group_idx = group_scores.topk_unsorted(self.cfg.topk_group)?.indices;
                let mut group_mask = group_scores.zeros_like()?;
                group_mask = group_mask.scatter_add(
                    &group_idx,
                    &group_idx.ones_like()?.to_dtype(group_mask.dtype())?,
                    1,
                )?;
                let score_mask = group_mask
                    .unsqueeze(D::Minus1)?
                    .expand((
                        bs * seq_len,
                        self.cfg.n_group,
                        self.n_routed_experts / self.cfg.n_group,
                    ))?
                    .reshape((bs * seq_len, ()))?;
                let tmp_scores = masked_fill(&score_mask, &(1. - &score_mask.ne(0.)?)?, 0.)?;
                let TopKOutput { values, indices } = tmp_scores.topk_unsorted(self.top_k)?;
                (values, indices)
            }
            }
        };

        if matches!(
            self.cfg.scoring_func,
            ScoringFunc::Sigmoid | ScoringFunc::SqrtSoftplus
        ) {
            let denominator = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = topk_weight.broadcast_div(&denominator)?;
        }

        topk_weight = (topk_weight * self.cfg.routed_scaling_factor)?;

        // RUN-161 diagnostic: are the selected top-k routing weights peaked or
        // near-uniform? Near-uniform weights average 6 redundant experts and
        // amplify their shared common-mode (collapse). Log the spread of the
        // raw gate logits at the selected experts, and the final weight spread.
        if std::env::var_os("ARC_COLLAPSE").is_some() {
            if let Ok(topk_logits) = logits.gather(&topk_idx, 1) {
                v4_stat_dbg(&topk_logits, "gate.topk_logits");
            }
            v4_stat_dbg(&topk_weight, "gate.topk_weight");
        }

        // RUN-161 fix-test: replace the near-uniform normalized-sqrtsoftplus
        // weights with a softmax over the selected experts' RAW logits (peaked
        // routing). If this restores coherence, the routing-weight uniformity is
        // the collapse root cause.
        if std::env::var_os("ARC_SOFTMAX_ROUTE").is_some() {
            if let Ok(topk_logits) = logits.gather(&topk_idx, 1) {
                let sm = candle_nn::ops::softmax_last_dim(&topk_logits)?;
                topk_weight =
                    (sm.to_dtype(topk_weight.dtype())? * self.cfg.routed_scaling_factor)?;
            }
        }

        // RUN-161 diagnostic: collapse is driven by the routed-expert weighted
        // sum at deep layers. ARC_ROUTE_TOP1 keeps only the single
        // largest-weight expert per token (zeroes the rest) to test whether the
        // near-uniform top-6 averaging of redundant deep-layer experts is the
        // common-mode source.
        if std::env::var_os("ARC_ROUTE_TOP1").is_some() {
            let max = topk_weight.max_keepdim(D::Minus1)?;
            let diff = topk_weight.broadcast_sub(&max)?;
            let mask = diff.ge(&diff.zeros_like()?)?.to_dtype(topk_weight.dtype())?;
            topk_weight = (topk_weight * mask)?;
        }
        Ok((topk_idx, topk_weight))
    }
}

struct Moe {
    experts: MoEExperts,
    shared_experts: Option<Mlp>,
    gate: MoeGate,
    layer_idx: usize,
}

impl Moe {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &DeepSeekV4Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        n_shared_experts: Option<usize>,
        n_routed_experts: usize,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(real_device);

        let moe_cfg = MoEExpertsConfig {
            // V4: apply the trained SwiGLU clamp (swiglu_limit, default 10.0) in
            // the routed experts. Without it the experts explode (RUN-161).
            swiglu_limit: Some(cfg.swiglu_limit),
            num_experts: n_routed_experts,
            num_experts_per_tok: cfg.num_experts_per_tok.unwrap_or(6),
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
        };

        let experts = MoEExperts::new(
            &moe_cfg,
            mapper.set_device(layer_idx, vb.clone(), loading_isq),
            layer_device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        let shared_experts = if let Some(n_shared_experts) = n_shared_experts {
            let intermediate_size = cfg.moe_intermediate_size * n_shared_experts;
            Some(Mlp::new(
                mapper.set_device(layer_idx, vb.pp("shared_experts"), loading_isq),
                cfg.hidden_size,
                intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                comm,
            )?)
        } else {
            None
        };
        let gate = MoeGate::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("gate"), false),
            n_routed_experts,
            layer_idx,
        )?;
        Ok(Self {
            experts,
            shared_experts,
            gate,
            layer_idx,
        })
    }

    fn forward(&self, xs: &Tensor, input_ids: Option<&Tensor>) -> Result<Tensor> {
        // Tier-B TD-MoE probe: dump the real expert-input activations for a few
        // representative layers so we can measure activation-aware low-rank
        // error offline. Enable with ARC_CAPTURE_MOE_INPUT=<dir>. Overwrites on
        // each call, so send the calibration prompt last and the file holds the
        // real-data activations (not the warmup dummy run). (RUN-161)
        if let Some(dir) = std::env::var_os("ARC_CAPTURE_MOE_INPUT") {
            if matches!(self.layer_idx, 2 | 20 | 40) {
                if let Ok(x_cpu) = xs
                    .to_dtype(DType::F32)
                    .and_then(|t| t.to_device(&Device::Cpu))
                {
                    let path = format!(
                        "{}/moe_input_L{}.safetensors",
                        dir.to_string_lossy(),
                        self.layer_idx
                    );
                    let _ = x_cpu.save_safetensors("x", &path);
                }
            }
        }
        let identity = xs.clone();
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;

        v4_stat_dbg(xs, "moe.input");
        let (topk_idx, topk_weight) = self.gate.forward(xs, input_ids)?;
        v4_stat_dbg(&topk_weight, "moe.topk_w");
        v4_stat_dbg(&topk_idx, "moe.topk_idx");
        // RUN-161 diagnostic: dump the actual per-token selected expert IDs at a
        // few layers (hash L1, score-onset L7, deep L14) to see whether the 5
        // prompt tokens route to the SAME experts (routing collapse) or stay
        // diverse. If score-routed layers over-concentrate vs the hash layers,
        // the selection bias / scoring is wrong.
        if std::env::var_os("ARC_COLLAPSE").is_some()
            && matches!(self.layer_idx, 1 | 4 | 7 | 14 | 20)
        {
            if let Ok(v) = topk_idx.to_dtype(DType::U32).and_then(|t| t.to_vec2::<u32>()) {
                eprintln!("ARC_TOPKID [L{}] {:?}", self.layer_idx, v);
            }
        }
        let mut y = self.experts.forward(xs, topk_weight, &topk_idx)?;
        y = y.reshape((b_size, seq_len, hidden_dim))?;
        v4_stat_dbg(&y, "moe.routed");
        let li = self.layer_idx;
        v4_collapse_dbg(&y, &format!("L{li}.moe_routed"), 1);
        v4_collapse_dbg(
            &topk_idx.to_dtype(DType::F32).unwrap_or_else(|_| topk_idx.clone()),
            &format!("L{li}.moe_topk_idx"),
            0,
        );

        if let Some(ref shared_experts) = self.shared_experts {
            let shared_out = shared_experts.forward(&identity)?;
            v4_stat_dbg(&shared_out, "moe.shared");
            v4_collapse_dbg(&shared_out, &format!("L{li}.moe_shared"), 1);
            y = (y + shared_out)?;
        }
        v4_stat_dbg(&y, "moe.out");

        Ok(y)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = self.experts.get_isq_layers();
        if let Some(ref mut shared) = self.shared_experts {
            layers.push(&mut shared.gate);
            layers.push(&mut shared.up);
            layers.push(&mut shared.down);
        }
        layers
    }
}

enum MoeOrMlp {
    Moe(Box<Moe>),
    /// Used only when `cfg.first_k_dense_replace > 0` (V3-style fixtures).
    /// V4 Flash sets it to 0 — every layer is MoE. Audit §0 + §5 line 448.
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, input_ids: Option<&Tensor>) -> Result<Tensor> {
        match self {
            // Dense MLP layers (V4 Flash has none) ignore input_ids.
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs, input_ids),
        }
    }
}

/// V4 QAT: block-wise (64) FP8 round-trip on the non-rope dims of K, matching
/// reference inference/model.py:506 `act_quant(kv[..., :-rd], 64, ..., inplace=True)`.
/// Per 64-element block: scale = amax/448 (E4M3 max), quantize to F8E4M3, dequant
/// back. Rope dims (last `rope_dim`) untouched. Computed in F32 (ref does FP32
/// internally). (RUN-161)
fn act_quant_kv_nope(k: &Tensor, rope_dim: usize) -> Result<Tensor> {
    let head_dim = k.dim(D::Minus1)?;
    let nope = head_dim - rope_dim;
    const BLOCK: usize = 64;
    if nope == 0 || nope % BLOCK != 0 {
        return Ok(k.clone());
    }
    let k_nope = k.narrow(D::Minus1, 0, nope)?;
    let k_rope = k.narrow(D::Minus1, nope, rope_dim)?;
    let orig_dims = k_nope.dims().to_vec();
    let mut blk = orig_dims.clone();
    let last = blk.len() - 1;
    blk[last] = nope / BLOCK;
    blk.push(BLOCK);
    let kb = k_nope.to_dtype(DType::F32)?.reshape(blk)?;
    let amax = kb.abs()?.max_keepdim(D::Minus1)?;
    let scale = (amax / 448.0)?.affine(1.0, 1e-12)?;
    let scaled = kb.broadcast_div(&scale)?;
    // candle's F8E4M3 cast has no CUDA kernel ("named symbol not found"), so the
    // exact round-trip runs on CPU — but that forces a GPU<->CPU sync every
    // attention layer (43 syncs/token), serializing decode. ARC_GPU_ACT_QUANT=1
    // does an on-GPU E4M3 round-trip instead (round-to-nearest, 3 mantissa bits,
    // exp range [-6,8], max 448) — removes the sync at a sub-ULP-of-FP8
    // approximation the FP8-trained model tolerates. (RUN-161 throughput)
    let dev = scaled.device().clone();
    let rt = if std::env::var_os("ARC_GPU_ACT_QUANT").is_some() {
        let ln2 = std::f64::consts::LN_2;
        let x = scaled.clamp(-448f32, 448f32)?;
        let ax = x.abs()?.clamp(1e-30f32, 1e30f32)?;
        let e = ax.log()?.affine(1.0 / ln2, 0.0)?.floor()?.clamp(-6f32, 8f32)?; // floor(log2|x|)
        let step = e.affine(ln2, -3.0 * ln2)?.exp()?; // 2^(e-3)
        x.broadcast_div(&step)?.round()?.broadcast_mul(&step)?
    } else {
        scaled
            .to_device(&candle_core::Device::Cpu)?
            .to_dtype(DType::F8E4M3)?
            .to_dtype(DType::F32)?
            .to_device(&dev)?
    };
    let q = rt.broadcast_mul(&scale)?;
    let k_nope_q = q.reshape(orig_dims)?.to_dtype(k.dtype())?;
    Tensor::cat(&[&k_nope_q, &k_rope], D::Minus1)?.contiguous()
}

/// RUN-161 decode profiler. Enabled by `ARC_TIME_DECODE=1`. Accumulates
/// per-component GPU time (sync'd) across all layers of one forward, logged +
/// reset each call. Sync per component => accurate attribution (kills overlap,
/// but decode is launch-bound so there's little overlap to lose). The split
/// tells us which component to fuse next for the throughput cut.
pub(crate) static DECODE_NS: [std::sync::atomic::AtomicU64; 6] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
pub(crate) const DECODE_NAMES: [&str; 6] = [
    "mhc_attn_pre", "mla_attn", "mix_post_attn", "mhc_ffn_pre", "moe", "mix_post_ffn",
];

/// MLA sub-component timers (Phase 1: pin the 50%). SDPA = mla_attn - these 3.
pub(crate) static MLA_NS: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
pub(crate) const MLA_NAMES: [&str; 3] = ["q_proj", "kv_proj_rope", "invrope_oproj"];

#[inline]
pub(crate) fn timed_mla<T>(
    idx: usize,
    dev: &candle_core::Device,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if std::env::var_os("ARC_TIME_DECODE").is_none() {
        return f();
    }
    let _ = dev.synchronize();
    let t0 = std::time::Instant::now();
    let r = f()?;
    let _ = dev.synchronize();
    MLA_NS[idx].fetch_add(
        t0.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    Ok(r)
}

#[inline]
pub(crate) fn timed<T>(
    idx: usize,
    dev: &candle_core::Device,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if std::env::var_os("ARC_TIME_DECODE").is_none() {
        return f();
    }
    let _ = dev.synchronize();
    let t0 = std::time::Instant::now();
    let r = f()?;
    let _ = dev.synchronize();
    DECODE_NS[idx].fetch_add(
        t0.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    Ok(r)
}

/// Debug-only NaN/Inf localizer. Enabled by setting `V4_NAN_DEBUG=1`.
/// Logs (to stderr) the first stage in the forward pass that produces a
/// non-finite value, with shape and the finite value range for context.
/// No-op (single env lookup) when the var is unset.
pub(crate) fn v4_nan_dbg(t: &Tensor, tag: &str) {
    // Also emit magnitude stats under V4_STATS so the existing per-sub-op
    // probes double as a collapse/explosion localizer (RUN-161).
    v4_stat_dbg(t, tag);
    if std::env::var_os("V4_NAN_DEBUG").is_none() {
        return;
    }
    match t
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
    {
        Ok(v) => {
            let n = v.len();
            let nans = v.iter().filter(|x| x.is_nan()).count();
            let infs = v.iter().filter(|x| x.is_infinite()).count();
            if nans > 0 || infs > 0 {
                let fmin = v.iter().copied().filter(|x| x.is_finite()).fold(f32::MAX, f32::min);
                let fmax = v.iter().copied().filter(|x| x.is_finite()).fold(f32::MIN, f32::max);
                let shape = t.dims().to_vec();
                eprintln!(
                    "V4_NAN_DEBUG [{tag}] shape={shape:?} nans={nans} infs={infs}/{n} finite_range=[{fmin:.4},{fmax:.4}]"
                );
            }
        }
        Err(e) => eprintln!("V4_NAN_DEBUG [{tag}] check failed: {e}"),
    }
}

/// Debug-only magnitude logger. Enabled by setting `V4_STATS=1`. Unlike
/// `v4_nan_dbg`, this always logs (abs-mean / std / range), so it can localize
/// a representation collapse (values -> ~0 or constant) that produces a flat
/// logit distribution. Used to bisect which layer/op collapses the hidden state.
pub(crate) fn v4_stat_dbg(t: &Tensor, tag: &str) {
    if std::env::var_os("V4_STATS").is_none() {
        return;
    }
    match t
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
    {
        Ok(v) => {
            let n = v.len().max(1) as f32;
            let naninf = v.iter().filter(|x| !x.is_finite()).count();
            let mean = v.iter().sum::<f32>() / n;
            let absmean = v.iter().map(|x| x.abs()).sum::<f32>() / n;
            let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
            let fmin = v.iter().copied().filter(|x| x.is_finite()).fold(f32::MAX, f32::min);
            let fmax = v.iter().copied().filter(|x| x.is_finite()).fold(f32::MIN, f32::max);
            eprintln!(
                "V4_STATS [{tag}] absmean={absmean:.6} std={:.6} range=[{fmin:.4},{fmax:.4}] nan/inf={naninf}",
                var.sqrt()
            );
        }
        Err(e) => eprintln!("V4_STATS [{tag}] failed: {e}"),
    }
}

/// Cross-position similarity probe (RUN-161 context-collapse localizer).
/// Enabled with `ARC_COLLAPSE=1`. Collapses every axis except `pos_dim` into a
/// per-position feature vector, then reports the mean off-diagonal cosine
/// between positions and cos(first,last). A value trending to ~1.0 means the
/// positions have stopped differentiating (context collapse) — pinpointing the
/// exact layer/op where contextual mixing dies, which whole-tensor std cannot.
pub(crate) fn v4_collapse_dbg(t: &Tensor, tag: &str, pos_dim: usize) {
    if std::env::var_os("ARC_COLLAPSE").is_none() {
        return;
    }
    let run = || -> Result<()> {
        let t = t.to_dtype(DType::F32)?;
        let dims = t.dims().to_vec();
        if pos_dim >= dims.len() {
            return Ok(());
        }
        let tlen = dims[pos_dim];
        if tlen < 2 {
            return Ok(()); // single-token (decode) step: nothing to compare
        }
        // Move pos to front, flatten the rest into a feature vector per position.
        let mut perm: Vec<usize> = vec![pos_dim];
        perm.extend((0..dims.len()).filter(|&d| d != pos_dim));
        let tp = t.permute(perm)?.contiguous()?;
        let f = tp.elem_count() / tlen;
        let m = tp.reshape((tlen, f))?;
        let norm = (m.sqr()?.sum_keepdim(1)?.sqrt()? + 1e-8)?;
        let mn = m.broadcast_div(&norm)?;
        let cos = mn.matmul(&mn.t()?)?; // [T, T]
        let cosv: Vec<f32> = cos.flatten_all()?.to_vec1()?;
        let mut s = 0f32;
        let mut c = 0usize;
        for i in 0..tlen {
            for j in 0..tlen {
                if i != j {
                    s += cosv[i * tlen + j];
                    c += 1;
                }
            }
        }
        let mean_off = s / (c.max(1) as f32);
        let first_last = cosv[tlen - 1]; // cos(pos0, posLast)
        eprintln!(
            "ARC_COLLAPSE [{tag}] T={tlen} mean_offdiag_cos={mean_off:.5} cos(first,last)={first_last:.5}"
        );
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("ARC_COLLAPSE [{tag}] failed: {e}");
    }
}

/// Debug-only per-op tensor dumper for tandem prefill-vs-decode differential
/// tracing (RUN-161 decode bug). Enabled by `V4_TRACE=<dir>`. Writes each tagged
/// tensor to `<dir>/L<layer>.<tag>.npy` (f32). Run prefill and decode into two
/// different dirs, then diff op-by-op: the FIRST op whose cosine drops below 1.0
/// is the bug. Gated to a single layer (`V4_TRACE_LAYER`, default 0) to stay
/// tiny and avoid per-layer overwrite. Position alignment is done in the Python
/// diff (prefill keeps row N, decode keeps row 0). No-op when unset.
pub(crate) fn v4_trace_want_layer() -> Option<usize> {
    if std::env::var_os("V4_TRACE").is_none() {
        return None;
    }
    Some(
        std::env::var("V4_TRACE_LAYER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
    )
}

pub(crate) fn v4_trace_dump(layer: usize, t: &Tensor, tag: &str) {
    if v4_trace_want_layer() != Some(layer) {
        return;
    }
    let dir = match std::env::var("V4_TRACE") {
        Ok(d) => d,
        Err(_) => return,
    };
    let path = format!("{dir}/L{layer}.{tag}.npy");
    if let Err(e) = t
        .to_dtype(DType::F32)
        .and_then(|t| t.contiguous())
        .and_then(|t| t.write_npy(&path))
    {
        eprintln!("V4_TRACE [{tag}] write failed: {e}");
    }
}

pub(crate) fn v4_trace_text(layer: usize, tag: &str, text: &str) {
    if v4_trace_want_layer() != Some(layer) {
        return;
    }
    if let Ok(dir) = std::env::var("V4_TRACE") {
        let _ = std::fs::write(format!("{dir}/L{layer}.{tag}.txt"), text);
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn: Attention,
    moe_or_mlp: MoeOrMlp,
    /// V4 mHC layer-side parameters (loaded if present in checkpoint).
    /// When present, the attn-side and ffn-side residual mixes use the
    /// learned `hc_attn_*` / `hc_ffn_*` weights (RUN-169). Audit §0 + §2
    /// ("DecoderLayer forward — mHC replaces residuals").
    mhc: Option<V4MHCLayerParams>,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV4Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        // V4 native: `attn`, `ffn`, `attn_norm`, `ffn_norm`.
        // HF:        `self_attn`, `mlp`, `input_layernorm`,
        //            `post_attention_layernorm`.
        let uses_native_layer = vb.contains_tensor("attn.wq_a.weight")
            || vb.contains_tensor("attn.wkv.weight")
            || vb.contains_tensor("attn_norm.weight");
        let (attn_subpath, input_ln_path, post_ln_path, mlp_subpath) = if uses_native_layer {
            ("attn", "attn_norm", "ffn_norm", "ffn")
        } else {
            (
                "self_attn",
                "input_layernorm",
                "post_attention_layernorm",
                "mlp",
            )
        };

        let attn = Attention::new(
            rotary_emb,
            cfg,
            vb.pp(attn_subpath),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            comm,
        )?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp(input_ln_path), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp(post_ln_path), false),
        )?;

        // V4: ALL 43 layers are MoE (`first_k_dense_replace=0`). The dense
        // MLP branch is reachable only if a config explicitly sets a
        // non-zero `first_k_dense_replace` AND provides `intermediate_size`.
        // Audit §0 + §5 line 448.
        let mhc_device = real_device.clone();
        let moe_or_mlp = if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
            layer_idx >= cfg.first_k_dense_replace
                && layer_idx.is_multiple_of(cfg.moe_layer_freq)
        }) {
            MoeOrMlp::Moe(Box::new(Moe::new(
                cfg,
                vb.pp(mlp_subpath),
                mapper,
                layer_idx,
                loading_isq,
                cfg.n_shared_experts,
                n_routed_experts,
                comm,
                real_device,
            )?))
        } else if let Some(intermediate_size) = cfg.intermediate_size {
            MoeOrMlp::Mlp(Mlp::new(
                mapper.set_device(layer_idx, vb.pp(mlp_subpath), loading_isq),
                cfg.hidden_size,
                intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                comm,
            )?)
        } else {
            candle_core::bail!(
                "DeepSeekV4 layer {layer_idx} has neither MoE config nor intermediate_size — \
                 invalid V4 config"
            );
        };

        // mHC layer params (loaded at the LAYER root, i.e. `vb` itself).
        // Audit §0 + dsv4_mhc.rs.
        //
        // RUN-169: when a V4-shaped checkpoint is being loaded but the
        // `hc_attn_*` tensors are missing for a given layer, `forward` will
        // fall back to standard residual instead of the learned mHC blend.
        // We surface that fallback once per layer at construction time so it
        // is never silent. We gate the warning on a positive V4-checkpoint
        // signal (`compress_ratios` non-empty AND the layer index is within
        // it) so legacy V3 fixtures and unit-test synthesisers that
        // intentionally omit `hc_attn_*` stay quiet.
        let mhc = V4MHCLayerParams::try_load(cfg, &vb, layer_idx, &mhc_device);
        if mhc.is_none() && !cfg.compress_ratios.is_empty() && layer_idx < cfg.compress_ratios.len()
        {
            tracing::warn!(
                "V4 DecoderLayer {layer_idx}: `hc_attn_*` tensors absent — \
                 falling back to standard residual (xs + sublayer(norm(xs))). \
                 Learned mHC blend (RUN-169) is inactive for this layer."
            );
        }

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attn,
            moe_or_mlp,
            mhc,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        xs_hist_cache: Option<&mut KvCache>,
        metadata: Option<(
            (Tensor, Tensor, Option<Tensor>, Option<Tensor>),
            &PagedAttentionInputMetadata,
        )>,
        flash_params: &FlashParams,
        input_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        // V4 3-D residual fallback path (legacy / partial-V4 checkpoints).
        //
        // This path runs when `DeepSeekV4::forward` keeps the residual as
        // 3-D, i.e. when the global `mhc_head` is absent from the
        // checkpoint. The full 4-D end-to-end threading (RUN-164) is the
        // active path for real V4 checkpoints; it uses `forward_4d` below.
        //
        // When `self.mhc` is loaded (per-layer mHC tensors present), the
        // attn-side and ffn-side residual mixes use the 3-D ↔ 4-D bridge
        // (`mix_attn_3d_bridge` / `mix_ffn_3d_bridge`) which exercises the
        // learned `hc_attn_*` / `hc_ffn_*` weights but pays the cost of
        // re-broadcasting + mean-collapsing the 4-D streams once per layer.
        //
        // When `self.mhc` is absent (e.g. legacy V3-style fixtures or
        // synthesised tests without the `hc_attn_*` tensors) we fall back
        // to standard residual. The construction-time warning lives in
        // `DecoderLayer::new` so it fires once per layer.

        let residual = xs;
        let xs_normed = self.input_layernorm.forward(xs)?;
        let attn_out = self.attn.forward(
            &xs_normed,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            xs_hist_cache,
            metadata,
            flash_params,
        )?;
        let xs = match &self.mhc {
            Some(mhc) => mhc.mix_attn_3d_bridge(residual, &attn_out)?,
            None => (attn_out + residual)?,
        };

        let residual = &xs;
        let ffn_out = self
            .moe_or_mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?, input_ids)?;
        match &self.mhc {
            Some(mhc) => mhc.mix_ffn_3d_bridge(residual, &ffn_out),
            None => residual + ffn_out,
        }
    }

    /// V4 4-D residual end-to-end forward (RUN-164).
    ///
    /// Takes a `[B, T, hc_mult, hidden]` mHC residual stack and threads it
    /// through this layer's attention + FFN blocks without collapsing back
    /// to 3-D. The math mirrors SGLang's `DeepseekV4DecoderLayer.forward`
    /// exactly (audit §3 lines 266-287 / `deepseek_v4.py:910-1001`):
    ///
    /// ```text
    /// residual_4d = xs_4d
    /// y_3d, post, comb = hc_pre(xs_4d, hc_attn_fn, hc_attn_scale, hc_attn_base)
    /// y_normed = input_layernorm(y_3d)
    /// attn_out = self_attn(y_normed)
    /// xs_4d = hc_post(attn_out, residual_4d, post, comb)
    ///
    /// residual_4d = xs_4d
    /// y_3d, post, comb = hc_pre(xs_4d, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
    /// y_normed = post_attention_layernorm(y_3d)
    /// ffn_out = mlp(y_normed)
    /// xs_4d = hc_post(ffn_out, residual_4d, post, comb)
    /// ```
    ///
    /// **Pre-condition**: `self.mhc` MUST be `Some`. The active 4-D path is
    /// only taken when `mhc_head.is_some()` at the model level, which in
    /// turn is gated by V4 checkpoint detection — every real V4 layer has
    /// its mHC tensors. If you reach this function with `self.mhc == None`,
    /// it's a bug in the model-level dispatch; we bail rather than silently
    /// fall back.
    #[allow(clippy::too_many_arguments)]
    fn forward_4d(
        &self,
        xs_4d: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        xs_hist_cache: Option<&mut KvCache>,
        metadata: Option<(
            (Tensor, Tensor, Option<Tensor>, Option<Tensor>),
            &PagedAttentionInputMetadata,
        )>,
        flash_params: &FlashParams,
        layer_idx: usize,
        input_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let li = layer_idx;
        v4_nan_dbg(xs_4d, &format!("L{li}.input_4d"));
        let mhc = self.mhc.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "DecoderLayer::forward_4d called on a layer without loaded `hc_attn_*` / \
                 `hc_ffn_*` tensors. The 4-D end-to-end path requires per-layer mHC weights; \
                 callers must dispatch to `forward` (3-D fallback) when `self.mhc` is None."
                    .to_string(),
            )
        })?;

        let tdev = xs_4d.device().clone();

        // === ATTN BLOCK ===
        let residual_attn = xs_4d;
        let (y_attn, post_attn, comb_attn) = timed(0, &tdev, || mhc.attn_pre(residual_attn))?;
        v4_nan_dbg(&y_attn, &format!("L{li}.attn_pre.y"));
        v4_collapse_dbg(&y_attn, &format!("L{li}.y_attn"), 1);
        v4_nan_dbg(&post_attn, &format!("L{li}.attn_pre.post"));
        v4_nan_dbg(&comb_attn, &format!("L{li}.attn_pre.comb"));
        let y_attn_normed = self.input_layernorm.forward(&y_attn)?;
        v4_nan_dbg(&y_attn_normed, &format!("L{li}.input_layernorm"));
        let attn_out = timed(1, &tdev, || {
            self.attn.forward(
                &y_attn_normed,
                attention_mask,
                seqlen_offsets,
                kv_cache,
                xs_hist_cache,
                metadata,
                flash_params,
            )
        })?;
        v4_nan_dbg(&attn_out, &format!("L{li}.attn_out"));
        v4_collapse_dbg(&attn_out, &format!("L{li}.attn_out"), 1);
        let xs_4d = timed(2, &tdev, || {
            mhc.mix_post_4d(&attn_out, residual_attn, &post_attn, &comb_attn)
        })?;
        v4_nan_dbg(&xs_4d, &format!("L{li}.mix_post_attn"));
        v4_collapse_dbg(&xs_4d, &format!("L{li}.mix_post_attn"), 1);

        // === FFN BLOCK ===
        let residual_ffn = &xs_4d;
        let (y_ffn, post_ffn, comb_ffn) = timed(3, &tdev, || mhc.ffn_pre(residual_ffn))?;
        v4_nan_dbg(&y_ffn, &format!("L{li}.ffn_pre.y"));
        v4_collapse_dbg(&y_ffn, &format!("L{li}.y_ffn"), 1);
        let y_ffn_normed = self.post_attention_layernorm.forward(&y_ffn)?;
        v4_nan_dbg(&y_ffn_normed, &format!("L{li}.post_attention_layernorm"));
        let ffn_out = timed(4, &tdev, || self.moe_or_mlp.forward(&y_ffn_normed, input_ids))?;
        v4_nan_dbg(&ffn_out, &format!("L{li}.ffn_out"));
        v4_collapse_dbg(&ffn_out, &format!("L{li}.ffn_out"), 1);
        let out = timed(5, &tdev, || {
            mhc.mix_post_4d(&ffn_out, residual_ffn, &post_ffn, &comb_ffn)
        })?;
        v4_nan_dbg(&out, &format!("L{li}.mix_post_ffn"));
        v4_collapse_dbg(&out, &format!("L{li}.residual"), 1);
        Ok(out)
    }
}

/// Device-mapper adapter that pins every per-layer call to a fixed layer
/// index. The MTP block lives at virtual layer `num_hidden_layers` (43 for
/// V4 Flash) — an index the real mapper's `mappings` table does not cover
/// (it would panic). Pinning to the LAST mapped layer co-locates the MTP
/// block with the final decoder layer + lm_head, which is where the draft
/// path's inputs (last hidden state, embeddings) already are.
#[derive(Debug)]
struct PinnedLayerMapper<'a> {
    inner: &'a dyn DeviceMapper,
    pin: usize,
}

impl DeviceMapper for PinnedLayerMapper<'_> {
    fn map(&self, input: Tensor, _: usize) -> Result<Tensor> {
        self.inner.map(input, self.pin)
    }
    fn set_device(
        &self,
        _: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder {
        self.inner.set_device(self.pin, varbuilder, loading_isq)
    }
    fn device_for(&self, _: usize, loading_isq: bool) -> Option<&Device> {
        self.inner.device_for(self.pin, loading_isq)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        self.inner.get_unique_devices()
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        self.inner.cast_nm_device(x, loading_isq)
    }
    fn set_nm_device(&self, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder {
        self.inner.set_nm_device(varbuilder, loading_isq)
    }
    fn num_device_mapping_layers(&self) -> usize {
        self.inner.num_device_mapping_layers()
    }
    fn get_comm_for(&self, _: usize) -> Result<Arc<mistralrs_quant::Comm>> {
        self.inner.get_comm_for(self.pin)
    }
    fn get_min_dtype(&self, dtype: &dyn crate::TryIntoDType) -> Result<DType> {
        self.inner.get_min_dtype(dtype)
    }
}

/// V4 MTP transformer block — the FULL decoder layer the checkpoint ships at
/// `mtp.0.*` (native) / `mtp.layers.0.*` (HF): V4-shape attention (LoRA Q +
/// fused wkv + grouped o_proj) + the complete 256-expert MoE, wrapped with
/// the `hnorm`/`enorm` input norms and the `norm` head norm. Audit §0 + §2
/// ("MTP module") + `deepseek_v4_nextn.py:50-201`.
///
/// Loaded only when `--mtp-depth > 0`
/// ([`crate::pipeline::mtp_pipeline::set_mtp_load_depth`]) — ~3GB at FP8 —
/// and consumed by `MtpDecodeKit` in `pipeline/mtp_pipeline.rs`, where the
/// draft chain runs `h/e combine → decoder layer → norm → lm_head`.
pub struct MtpBlock {
    /// RMSNorm over the previous-step hidden state (`mtp.0.hnorm`).
    hnorm: RmsNorm,
    /// RMSNorm over the current-token embedding (`mtp.0.enorm`).
    enorm: RmsNorm,
    /// Final RMSNorm before the shared lm_head (`mtp.0.norm` — SGLang's
    /// `shared_head.norm`).
    out_norm: RmsNorm,
    /// The full V4 decoder layer (attention + MoE + optional per-layer mHC).
    /// `compress_ratio` is Standard per `COMPRESS_RATIO_NEXTN_LAYER = 0`.
    layer: DecoderLayer,
    /// Empty (non-varlen) flash params for the single-token draft steps.
    flash_params: FlashParams,
    /// Device the block's weights live on (for input hand-off).
    device: Device,
    max_seq_len: usize,
}

impl MtpBlock {
    /// Try to load the full MTP decoder block from `vb` (positioned at
    /// `mtp.0` / `mtp.layers.0`). Returns `Ok(None)` when the checkpoint
    /// only ships the light `h_proj`/`e_proj` heads (older exports) — the
    /// caller then keeps the Tier-A projection-only draft path.
    fn try_new(
        cfg: &DeepSeekV4Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
        real_device: &Device,
    ) -> Result<Option<Self>> {
        // Presence probe: the block requires its wrapper norms plus a decoder
        // layer (native `attn_norm` / HF `input_layernorm` naming).
        let has_decoder =
            vb.contains_tensor("attn_norm.weight") || vb.contains_tensor("input_layernorm.weight");
        let has_wrappers = vb.contains_tensor("hnorm.weight")
            && vb.contains_tensor("enorm.weight")
            && vb.contains_tensor("norm.weight");
        if !has_decoder || !has_wrappers {
            return Ok(None);
        }

        // The MTP block is a VIRTUAL extra layer: config decisions (compress
        // ratio, hash-vs-score routing) use index `num_hidden_layers`; device
        // decisions pin to the last real layer (see `PinnedLayerMapper`).
        let layer_idx = cfg.num_hidden_layers;
        let pinned = PinnedLayerMapper {
            inner: mapper,
            pin: cfg.num_hidden_layers.saturating_sub(1),
        };
        let device = pinned
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or_else(|| real_device.clone());

        // MTP attention is standard MQA (`COMPRESS_RATIO_NEXTN_LAYER = 0`,
        // audit §2) → standard-θ RoPE, same YARN handling as the main
        // standard layers (see the rope construction in `DeepSeekV4::new`).
        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: if std::env::var_os("ARC_DISABLE_YARN_STD").is_some() {
                None
            } else {
                cfg.rope_scaling.clone()
            },
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        let rotary_emb = Arc::new(DeepSeekV2RotaryEmbedding::new(
            &rope_cfg,
            vb.dtype(),
            &device,
        )?);
        let comm = pinned.get_comm_for(layer_idx)?;

        let layer = DecoderLayer::new(
            rotary_emb,
            cfg,
            vb.clone(),
            &pinned,
            layer_idx,
            loading_isq,
            // The MTP chain keeps its own in-process `KvCache`; the engine's
            // PagedAttention never manages MTP slots.
            None,
            &comm,
            device.clone(),
        )?;

        let hnorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            pinned.set_device(layer_idx, vb.pp("hnorm"), false),
        )?;
        let enorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            pinned.set_device(layer_idx, vb.pp("enorm"), false),
        )?;
        let out_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            pinned.set_device(layer_idx, vb.pp("norm"), false),
        )?;

        Ok(Some(Self {
            hnorm,
            enorm,
            out_norm,
            layer,
            flash_params: FlashParams {
                max_q: 0,
                max_k: 0,
                cumulative_seqlens_q: HashMap::new(),
                cumulative_seqlens_k: HashMap::new(),
                causal: true,
            },
            device,
            max_seq_len: cfg.max_position_embeddings,
        }))
    }

    /// Fresh per-draft-chain KV cache (dim 2 = the sequence axis, matching
    /// `NormalCache::new`). One chain never sees another chain's (possibly
    /// rejected) draft tokens.
    pub fn new_chain_cache(&self) -> KvCache {
        KvCache::new_normal(2, self.max_seq_len, 16)
    }

    /// `hnorm` over the previous-step hidden state.
    pub fn norm_h(&self, h: &Tensor) -> Result<Tensor> {
        self.hnorm.forward(&h.to_device(&self.device)?)
    }

    /// `enorm` over the current-token embedding.
    pub fn norm_e(&self, e: &Tensor) -> Result<Tensor> {
        self.enorm.forward(&e.to_device(&self.device)?)
    }

    /// Final `norm` (SGLang `shared_head.norm`) before the shared lm_head.
    pub fn norm_out(&self, xs: &Tensor) -> Result<Tensor> {
        self.out_norm.forward(xs)
    }

    /// Run the combined `h_proj + e_proj` state through the real MTP decoder
    /// layer for ONE draft token.
    ///
    /// * `fused`: `[B, 1, hidden]` — `h_proj(hnorm(h)) + e_proj(enorm(e))`.
    /// * `pos`: absolute sequence position of this draft token (RoPE).
    /// * `cache`: the per-chain KV cache from [`Self::new_chain_cache`];
    ///   draft token `i` attends over draft tokens `0..=i` of its chain.
    /// * `input_ids`: `[B, 1]` current token ids — required by the
    ///   hash-routed (`tid2eid`) MoE gate; ignored by score-routed gates.
    ///
    /// Returns the decoder output `[B, 1, hidden]` (pre-`norm`; the caller
    /// applies [`Self::norm_out`] before lm_head and feeds THIS tensor
    /// forward as the next step's hidden state, matching the reference).
    ///
    /// Uses the 3-D decoder path: when the checkpoint ships the per-layer
    /// mHC tensors, `DecoderLayer::forward` applies the learned
    /// `hc_attn_*`/`hc_ffn_*` blend via the 3-D bridge (RUN-169); otherwise
    /// it falls back to standard residuals.
    pub fn forward_step(
        &self,
        fused: &Tensor,
        pos: usize,
        cache: &mut KvCache,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        let in_device = fused.device().clone();
        let xs = fused.to_device(&self.device)?;
        let ids = input_ids.to_device(&self.device)?;
        let seqlen_offsets = [pos];
        let out = self.layer.forward(
            &xs,
            // Single-token step: attends over the whole chain cache, no mask.
            None,
            &seqlen_offsets,
            cache,
            // MTP block is a Standard layer (`COMPRESS_RATIO_NEXTN_LAYER = 0`)
            // — no compressor, no xs history.
            None,
            None,
            &self.flash_params,
            Some(&ids),
        )?;
        out.to_device(&in_device)
    }

    /// ISQ handles for the block's quantizable projections: attention
    /// (wq_a/wq_b or plain q, wkv, wo_a, wo_b) + the MoE experts (routed +
    /// shared). Norms (`hnorm`/`enorm`/`norm`/`attn_norm`/`ffn_norm`/
    /// `q_norm`/`kv_norm`) and the router gate are excluded, matching the
    /// main model's `get_layers`.
    fn isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut tensors = Vec::new();
        match &mut self.layer.attn.q {
            QProj::Plain(q) => tensors.push(q),
            QProj::Lora { a, norm: _, b } => {
                tensors.push(a);
                tensors.push(b);
            }
        }
        tensors.push(&mut self.layer.attn.wkv);
        tensors.push(&mut self.layer.attn.wo_a);
        tensors.push(&mut self.layer.attn.wo_b);
        match &mut self.layer.moe_or_mlp {
            MoeOrMlp::Mlp(mlp) => {
                tensors.push(&mut mlp.gate);
                tensors.push(&mut mlp.up);
                tensors.push(&mut mlp.down);
            }
            MoeOrMlp::Moe(moe) => tensors.extend(moe.get_isq_layers()),
        }
        tensors
    }
}

/// V4 MTP head.
///
/// V4 Flash ships a full transformer decoder layer + 256-expert MoE block
/// at `mtp.0.*`. The two projection layers `h_proj` and `e_proj` are always
/// loaded when present (cheap: 2 × `[hidden, hidden]`); the full decoder
/// block ([`MtpBlock`]) is loaded ONLY when `--mtp-depth > 0` was declared
/// before load (memory: ~3GB FP8 / ~800MB after qtip2 ISQ). Audit §0 + §2
/// ("MTP module").
pub struct MtpHead {
    pub h_proj: Arc<dyn QuantMethod>,
    pub e_proj: Arc<dyn QuantMethod>,
    /// Full MTP decoder block (attention + MoE + hnorm/enorm/norm).
    /// `None` for heads-only loads (older exports or `--mtp-depth 0`).
    pub block: Option<Arc<MtpBlock>>,
}

pub struct DeepSeekV4 {
    lm_head: Arc<dyn QuantMethod>,
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    cache: EitherCache,
    device: Device,
    max_seq_len: usize,
    cfg: ModelConfigMetadata,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    /// V4 MTP head — loaded if `mtp.0.h_proj` (V4 native) or
    /// `mtp.layers.0.h_proj` (HF) tensors present.
    mtp_head: Option<MtpHead>,
    /// V4 global mHC head — applied just before `norm` + `lm_head` to
    /// collapse the 4-D `[B, T, hc_mult, hidden]` residual stack back to
    /// `[B, T, hidden]` via the learned sigmoid-mixed sum across streams.
    /// When `None`, the model takes the 3-D fallback path (RUN-169 bridge
    /// or standard residual). Audit §0 + §2.
    mhc_head: Option<super::dsv4_mhc::V4MHCHead>,
    /// Full parsed config, kept so the MTP decoder block can be re-loaded
    /// from the source checkpoint when a UQFF artifact (baked without
    /// `--mtp-depth`) does not cover it. See
    /// `IsqModel::load_mtp_block_from_source`.
    cfg_full: DeepSeekV4Config,
    /// R3: per-layer slot of the compressor-input history inside the model's
    /// NormalCache. `xs_hist_slots[i] = Some(j)` means layer `i` (CSA/HCA)
    /// keeps its xs history at cache index `num_hidden_layers + j`; `None`
    /// for Standard layers. The history rides the same per-sequence
    /// clone_in/clone_out machinery as the KV entries, which is what makes
    /// multi-sequence batches (voting chains) safe.
    xs_hist_slots: Vec<Option<usize>>,
}

impl DeepSeekV4 {
    /// True if an MTP head was loaded from the checkpoint.
    pub fn has_mtp(&self) -> bool {
        self.mtp_head.is_some()
    }

    pub fn mtp_head(&self) -> Option<&MtpHead> {
        self.mtp_head.as_ref()
    }

    /// Shared input embedding (used as the MTP `e_emb` source). Cheap to
    /// clone — `Embedding` only owns one Tensor handle.
    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    /// Shared output projection (used as the MTP head's `lm_head`).
    pub fn lm_head(&self) -> &Arc<dyn QuantMethod> {
        &self.lm_head
    }

    /// True if mHC global parameters were loaded.
    pub fn has_mhc(&self) -> bool {
        self.mhc_head.is_some()
    }

    /// Emit the OPTIONAL MTP decoder block's non-ISQ tensors (wrapper norms,
    /// decoder-layer norms, router gate) at the HF path `mtp.layers.0.*` so a
    /// UQFF full serialization can reconstruct the block. The block's
    /// quantizable projections are ISQ tensors (serialized as `mtp.<j>` UQFF
    /// artifacts) and `h_proj`/`e_proj` are main ISQ tensors — neither
    /// belongs here. No-op when the block was not loaded.
    fn add_mtp_block_residuals(&self, uvb: &UnVarBuilder) {
        let Some(mtp) = &self.mtp_head else { return };
        let Some(block) = &mtp.block else { return };

        let uvb_b = uvb.pp("mtp").pp("layers").pp("0");
        uvb_b.pp("hnorm").add(&block.hnorm);
        uvb_b.pp("enorm").add(&block.enorm);
        uvb_b.pp("norm").add(&block.out_norm);

        let layer = &block.layer;
        uvb_b.pp("input_layernorm").add(&layer.input_layernorm);
        uvb_b
            .pp("post_attention_layernorm")
            .add(&layer.post_attention_layernorm);
        uvb_b
            .pp("self_attn")
            .pp("kv_a_layernorm")
            .add(&layer.attn.kv_norm);
        if let QProj::Lora { a: _, norm, b: _ } = &layer.attn.q {
            uvb_b.pp("self_attn").pp("q_a_layernorm").add(norm);
        }
        if let MoeOrMlp::Moe(moe) = &layer.moe_or_mlp {
            let uvb_g = uvb_b.pp("mlp").pp("gate");
            uvb_g.add_tensor("weight", moe.gate.weight.clone());
            if let Some(tid2eid) = &moe.gate.tid2eid {
                uvb_g.add_tensor("tid2eid", tid2eid.clone());
            }
            if let Some(bias) = &moe.gate.e_score_correction_bias {
                uvb_g.add_tensor("e_score_correction_bias", bias.clone());
            }
        }
        if let Some(mhc) = &layer.mhc {
            uvb_b.add_tensor("hc_attn_fn", mhc.hc_attn_fn.clone());
            uvb_b.add_tensor("hc_attn_base", mhc.hc_attn_base.clone());
            uvb_b.add_tensor("hc_attn_scale", mhc.hc_attn_scale.clone());
            uvb_b.add_tensor("hc_ffn_fn", mhc.hc_ffn_fn.clone());
            uvb_b.add_tensor("hc_ffn_base", mhc.hc_ffn_base.clone());
            uvb_b.add_tensor("hc_ffn_scale", mhc.hc_ffn_scale.clone());
        }
    }
}

impl DeepSeekV4 {
    pub fn new(
        cfg: &DeepSeekV4Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata.mapper;

        // V4 native publishes globals as `embed.weight`, `norm.weight`,
        // `head.weight` (audit §1). HF format uses
        // `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`.
        let uses_native = vb.contains_tensor("embed.weight")
            && !vb.contains_tensor("model.embed_tokens.weight");

        let (vb_m, embed_path, lm_head_vb, lm_head_path) = if uses_native {
            (vb.clone(), "embed", vb.clone(), "head")
        } else {
            (vb.pp("model"), "embed_tokens", vb.clone(), "lm_head")
        };

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp(embed_path), false),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(
                    lm_head_vb.pp(lm_head_path),
                    normal_loading_metadata.loading_isq,
                ),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(
                    embed_tokens.embeddings(),
                    normal_loading_metadata.loading_isq,
                )?,
                None,
            ))?
        };
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        // ---- Dual RoPE (standard + compress) ----
        // Audit §0 + §8 P1 item 11. Standard layers use `rope_theta`
        // (10000); compress layers use `compress_rope_theta` (160000).
        let mut rope_standard: HashMap<_, Arc<DeepSeekV2RotaryEmbedding>> = HashMap::new();
        let mut rope_compress: HashMap<_, Arc<DeepSeekV2RotaryEmbedding>> = HashMap::new();
        let rope_cfg_standard = DeepSeekV2RopeConfig {
            // RUN-161 ablation: the reference disables YARN on standard layers
            // (`original_seq_len, rope_theta = 0, args.rope_theta`); arc applies
            // it. ARC_DISABLE_YARN_STD=1 matches the reference to test whether
            // YARN-on-standard contributes to the context collapse.
            rope_scaling: if std::env::var_os("ARC_DISABLE_YARN_STD").is_some() {
                None
            } else {
                cfg.rope_scaling.clone()
            },
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        let rope_cfg_compress = DeepSeekV2RopeConfig {
            rope_scaling: cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.compress_rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        let mut need_standard_devices: std::collections::HashSet<_> =
            std::collections::HashSet::new();
        let mut need_compress_devices: std::collections::HashSet<_> =
            std::collections::HashSet::new();
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let loc = device.location();
            match cfg.layer_compress_ratio(i) {
                0 => {
                    need_standard_devices.insert(loc.clone());
                }
                _ => {
                    need_compress_devices.insert(loc.clone());
                }
            }
        }
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let loc = device.location();
            if need_standard_devices.contains(&loc) && !rope_standard.contains_key(&loc) {
                rope_standard.insert(
                    loc.clone(),
                    Arc::new(DeepSeekV2RotaryEmbedding::new(
                        &rope_cfg_standard,
                        vb.dtype(),
                        device,
                    )?),
                );
            }
            if need_compress_devices.contains(&loc) && !rope_compress.contains_key(&loc) {
                rope_compress.insert(
                    loc.clone(),
                    Arc::new(DeepSeekV2RotaryEmbedding::new(
                        &rope_cfg_compress,
                        vb.dtype(),
                        device,
                    )?),
                );
            }
        }

        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let layer_ratio = cfg.layer_compress_ratio(layer_idx);
            let rotary_emb = if layer_ratio == 0 {
                rope_standard
                    .get(&device.location())
                    .expect("No standard RoPE for device location!")
                    .clone()
            } else {
                rope_compress
                    .get(&device.location())
                    .expect("No compress RoPE for device location!")
                    .clone()
            };
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(
                    PagedAttention::new(cfg.head_dim, device, None)
                        .expect("Failed to create PagedAttention"),
                ),
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
                normal_loading_metadata.real_device.clone(),
            )
        })?;

        // ---- MTP head ----
        // Audit §0 + §2 + §5 lines 506-511. Probe both V4 native
        // (`mtp.0.*`) and HF (`mtp.layers.0.*`) paths.
        let mtp_head = {
            let mtp_native_vb = vb.pp("mtp").pp("0");
            let mtp_hf_vb = vb.pp("mtp").pp("layers").pp("0");
            let mtp_vb = if mtp_native_vb.contains_tensor("h_proj.weight") {
                Some(mtp_native_vb)
            } else if mtp_hf_vb.contains_tensor("h_proj.weight") {
                Some(mtp_hf_vb)
            } else {
                None
            };

            if let Some(mtp_vb) = mtp_vb {
                let try_load = |sub: &str| -> Result<Arc<dyn QuantMethod>> {
                    ReplicatedLayer::new(
                        cfg.hidden_size,
                        cfg.hidden_size,
                        &cfg.quantization_config,
                        false,
                        mtp_vb.pp(sub),
                    )
                };
                match (try_load("h_proj"), try_load("e_proj")) {
                    (Ok(h_proj), Ok(e_proj)) => {
                        // Full MTP decoder block — gated on `--mtp-depth > 0`
                        // (declared via `set_mtp_load_depth` before load).
                        // Memory is precious: the block is ~3GB at FP8, so it
                        // is skipped entirely when MTP drafting is disabled.
                        // Exception: a UQFF bake (`set_mtp_uqff_bake`)
                        // force-loads the block so its tensors are quantized
                        // and included in the artifact under `mtp.<j>` names.
                        let block = if crate::pipeline::mtp_pipeline::mtp_load_depth() > 0
                            || crate::pipeline::mtp_pipeline::mtp_uqff_bake()
                        {
                            match MtpBlock::try_new(
                                cfg,
                                mtp_vb.clone(),
                                &*mapper,
                                normal_loading_metadata.loading_isq,
                                &normal_loading_metadata.real_device,
                            ) {
                                Ok(Some(block)) => {
                                    tracing::info!(
                                        "V4 MTP: full decoder block loaded (attention + MoE); \
                                         draft chain will run through the real transformer step"
                                    );
                                    Some(Arc::new(block))
                                }
                                Ok(None) => {
                                    tracing::warn!(
                                        "V4 MTP: --mtp-depth > 0 but the checkpoint only ships \
                                         h_proj/e_proj (no mtp decoder tensors); falling back to \
                                         Tier-A projection-only drafting (~50% acceptance)"
                                    );
                                    None
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "V4 MTP: full decoder block load failed ({e}); falling \
                                         back to Tier-A projection-only drafting"
                                    );
                                    None
                                }
                            }
                        } else {
                            None
                        };
                        Some(MtpHead {
                            h_proj,
                            e_proj,
                            block,
                        })
                    }
                    _ => None,
                }
            } else {
                None
            }
        };

        // ---- Global mHC head ----
        // V4 publishes `hc_head_fn`, `hc_head_base`, `hc_head_scale` at the
        // model root. Audit §1 ("Top-level tensors").
        let mhc_head = super::dsv4_mhc::V4MHCHead::try_load(
            cfg,
            &vb_m,
            &normal_loading_metadata.real_device,
        );

        // R3: one extra per-sequence cache slot per CSA/HCA layer holding the
        // compressor-input history `xs` (`[B, T, hidden]`, seq dim 1). Placed
        // AFTER the `num_hidden_layers` KV entries so `cache[i]` stays the KV
        // cache of layer `i`. The engine's NormalCacheManager iterates
        // `metadata.num_hidden_layers`, which pipelines derive from the CACHE
        // length — so these extra entries are cloned in/out per sequence
        // exactly like KV, giving each sequence in a batch its own history.
        let mut xs_hist_slots = Vec::with_capacity(cfg.num_hidden_layers);
        let mut n_compress_layers = 0usize;
        for i in 0..cfg.num_hidden_layers {
            if cfg.layer_compress_ratio(i) != 0 {
                xs_hist_slots.push(Some(n_compress_layers));
                n_compress_layers += 1;
            } else {
                xs_hist_slots.push(None);
            }
        }
        let cache = NormalCache::new(cfg.num_hidden_layers, cfg.max_position_embeddings);
        {
            let mut guard = cache.lock().unwrap();
            for _ in 0..n_compress_layers {
                // Same geometry as the previous per-model xs_history
                // (`SingleCache::new(1, max_position_embeddings, 256)`):
                // seq dim 1 over `[B, T, hidden]`, initial capacity 256.
                guard
                    .0
                    .push(KvCache::new_normal(1, cfg.max_position_embeddings, 256));
            }
        }

        Ok(Self {
            lm_head,
            embed_tokens,
            norm,
            layers,
            cache: EitherCache::Normal(cache),
            device: normal_loading_metadata.real_device.clone(),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                // V4 MQA: 1 KV head (broadcast across 64 Q heads at the
                // kernel). Audit §0 + §3.
                num_kv_heads: cfg.num_key_value_heads.max(1),
                num_attn_heads: (cfg.num_attention_heads
                    / mapper.get_comm_for(0)?.world_size())
                .max(1),
                sliding_window: None,
                // V4: single 512-dim head for K and V (same fused wkv
                // vector). Audit §3.
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                // V4 is NOT MLA. The KV cache is a flat per-page tensor of
                // shape `[n_kv_heads=1, head_dim=512]`. Audit §5
                // lines 521-523.
                #[cfg(all(feature = "cuda", target_family = "unix"))]
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
                #[cfg(not(all(feature = "cuda", target_family = "unix")))]
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            mtp_head,
            mhc_head,
            cfg_full: cfg.clone(),
            xs_hist_slots,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(
            Vec<(Tensor, Tensor, Option<Tensor>, Option<Tensor>)>,
            &PagedAttentionInputMetadata,
        )>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let xs_embed = self.embed_tokens.forward(input_ids)?;
        let cache = &mut self.cache.normal().0;
        let attention_mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            xs_embed.dtype(),
            self.cfg.num_attn_heads,
        )?;
        let attention_mask = attention_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;

        // R3: split the cache vector into the per-layer KV entries
        // (`0..num_layers`) and the trailing per-layer compressor-input
        // histories (`num_layers..`, CSA/HCA layers only; see
        // `xs_hist_slots`). Two disjoint mutable regions so a layer can
        // borrow both its KV cache and its xs history at once.
        let n_layers = self.layers.len();
        let (layer_caches, xs_caches) = cache.split_at_mut(n_layers);

        // RUN-164: 4-D mHC residual threading end-to-end.
        //
        // Active path when the global mHC head (`hc_head_*`) is loaded AND
        // every layer has its per-layer mHC params loaded — i.e. a real V4
        // checkpoint. In that case:
        //   1. Lift `[B, T, hidden]` → `[B, T, hc_mult, hidden]` at the
        //      embedding output (replicate across the new stream axis).
        //   2. Each layer calls `forward_4d`, which threads the 4-D
        //      residual through attention + FFN using `hc_pre` + `hc_post`
        //      without collapsing back to 3-D.
        //   3. Collapse `[B, T, hc_mult, hidden]` → `[B, T, hidden]` via
        //      the learned `V4MHCHead::forward` (sigmoid-mixed sum across
        //      streams) just before `norm` + `lm_head`.
        //
        // Legacy 3-D path is taken when `mhc_head` is None OR any layer is
        // missing its `mhc` (partial-V4 / V3-style fixtures): the layer's
        // `forward` (3-D bridge per RUN-169) is called and the global head
        // is bypassed.
        let use_4d_mhc = self.mhc_head.is_some() && self.layers.iter().all(|l| l.mhc.is_some());

        let xs = if use_4d_mhc {
            // Lift to 4-D via the head's runtime (carries hc_mult).
            let mhc_head = self.mhc_head.as_ref().unwrap();
            v4_nan_dbg(&xs_embed, "embed");
            v4_stat_dbg(&xs_embed, "embed");
            let mut xs_4d = mhc_head.rt.lift_3d_to_4d(&xs_embed)?;
            v4_nan_dbg(&xs_4d, "lift_3d_to_4d");
            v4_stat_dbg(&xs_4d, "lift_3d_to_4d");
            for (i, layer) in self.layers.iter().enumerate() {
                xs_4d = self.mapper.map(xs_4d, i)?;
                xs_4d = layer.forward_4d(
                    &xs_4d,
                    attention_mask.as_ref().map(|m| m.get(xs_4d.device())),
                    seqlen_offsets,
                    &mut layer_caches[i],
                    self.xs_hist_slots[i].map(|slot| &mut xs_caches[slot]),
                    metadata
                        .as_ref()
                        .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                    flash_params,
                    i,
                    Some(input_ids),
                )?;
                v4_stat_dbg(&xs_4d, &format!("L{i}"));
            }
            if std::env::var_os("ARC_TIME_DECODE").is_some() {
                use std::sync::atomic::Ordering;
                let total: u64 = DECODE_NS.iter().map(|a| a.load(Ordering::Relaxed)).sum();
                let parts: Vec<String> = DECODE_NAMES
                    .iter()
                    .zip(DECODE_NS.iter())
                    .map(|(n, a)| {
                        let ns = a.load(Ordering::Relaxed);
                        format!(
                            "{}={:.2}ms({:.0}%)",
                            n,
                            ns as f64 / 1e6,
                            100.0 * ns as f64 / total.max(1) as f64
                        )
                    })
                    .collect();
                let mla_parts: Vec<String> = MLA_NAMES
                    .iter()
                    .zip(MLA_NS.iter())
                    .map(|(n, a)| format!("{}={:.2}ms", n, a.load(Ordering::Relaxed) as f64 / 1e6))
                    .collect();
                tracing::info!(
                    "ARC_TIME_DECODE forward_total={:.2}ms | {} || MLA[{}] (sdpa=mla_attn-these)",
                    total as f64 / 1e6,
                    parts.join(" "),
                    mla_parts.join(" ")
                );
                for a in DECODE_NS.iter() {
                    a.store(0, Ordering::Relaxed);
                }
                for a in MLA_NS.iter() {
                    a.store(0, Ordering::Relaxed);
                }
            }
            let xs_4d = xs_4d.to_device(&self.device)?;
            // Collapse via the learned global mHC head: 4-D → 3-D.
            let collapsed = mhc_head.forward(&xs_4d)?;
            v4_stat_dbg(&collapsed, "after_mhc_head");
            collapsed
        } else {
            let mut xs = xs_embed;
            for (i, layer) in self.layers.iter().enumerate() {
                xs = self.mapper.map(xs, i)?;
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref().map(|m| m.get(xs.device())),
                    seqlen_offsets,
                    &mut layer_caches[i],
                    self.xs_hist_slots[i].map(|slot| &mut xs_caches[slot]),
                    metadata
                        .as_ref()
                        .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                    flash_params,
                    Some(input_ids),
                )?;
            }
            xs.to_device(&self.device)?
        };

        v4_stat_dbg(&xs, "before_norm");
        let xs = xs.apply(&self.norm)?;
        v4_stat_dbg(&xs, "after_norm");
        let xs = extract_logits(&xs, context_lens)?;

        let logits = self.lm_head.forward_autocast(&xs)?;
        v4_stat_dbg(&logits, "logits");
        Ok(logits)
    }
}

impl IsqModel for DeepSeekV4 {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match &mut layer.attn.q {
                QProj::Plain(q) => {
                    tensors.push((q, Some(i)));
                }
                QProj::Lora { a, norm: _, b } => {
                    tensors.push((a, Some(i)));
                    tensors.push((b, Some(i)));
                }
            }
            // V4: ONE fused wkv (audit §0 + §5 lines 477-481).
            tensors.push((&mut layer.attn.wkv, Some(i)));
            tensors.push((&mut layer.attn.wo_a, Some(i)));
            tensors.push((&mut layer.attn.wo_b, Some(i)));
            // V4 compressor wkv_gate — must be included so ISQ moves it to GPU.
            if let Some(comp) = &mut layer.attn.compressor {
                tensors.push((&mut comp.wkv_gate, Some(i)));
            }
            match &mut layer.moe_or_mlp {
                MoeOrMlp::Mlp(mlp) => {
                    tensors.push((&mut mlp.gate, Some(i)));
                    tensors.push((&mut mlp.up, Some(i)));
                    tensors.push((&mut mlp.down, Some(i)));
                }
                MoeOrMlp::Moe(moe) => {
                    for layer in moe.get_isq_layers() {
                        tensors.push((layer, Some(i)));
                    }
                }
            }
        }
        if let Some(mtp) = &mut self.mtp_head {
            tensors.push((&mut mtp.h_proj, None));
            tensors.push((&mut mtp.e_proj, None));
            // Full MTP block: same ISQ treatment as a regular layer (experts
            // + attention projections; norms/router excluded inside
            // `isq_layers`). The Arc is unique until `mtp_decode_kit()` is
            // first called, which happens only after ISQ.
            if let Some(block) = &mut mtp.block {
                match Arc::get_mut(block) {
                    Some(block) => {
                        for t in block.isq_layers() {
                            tensors.push((t, None));
                        }
                    }
                    None => tracing::warn!(
                        "V4 MTP block is shared (decode kit already handed out); \
                         skipping its tensors for this ISQ pass"
                    ),
                }
            }
        }
        (tensors, &*self.mapper)
    }

    fn get_layers_moe_experts_only(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Include attention weights so they get moved to GPU even in MoeExpertsOnly mode.
            match &mut layer.attn.q {
                QProj::Plain(q) => {
                    tensors.push((q, Some(i)));
                }
                QProj::Lora { a, norm: _, b } => {
                    tensors.push((a, Some(i)));
                    tensors.push((b, Some(i)));
                }
            }
            tensors.push((&mut layer.attn.wkv, Some(i)));
            tensors.push((&mut layer.attn.wo_a, Some(i)));
            tensors.push((&mut layer.attn.wo_b, Some(i)));
            if let Some(comp) = &mut layer.attn.compressor {
                tensors.push((&mut comp.wkv_gate, Some(i)));
            }
            match &mut layer.moe_or_mlp {
                MoeOrMlp::Mlp(mlp) => {
                    tensors.push((&mut mlp.gate, Some(i)));
                    tensors.push((&mut mlp.up, Some(i)));
                    tensors.push((&mut mlp.down, Some(i)));
                }
                MoeOrMlp::Moe(moe) => {
                    for layer in moe.get_isq_layers() {
                        tensors.push((layer, Some(i)));
                    }
                }
            }
        }
        if let Some(mtp) = &mut self.mtp_head {
            tensors.push((&mut mtp.h_proj, None));
            tensors.push((&mut mtp.e_proj, None));
            if let Some(block) = &mut mtp.block {
                match Arc::get_mut(block) {
                    Some(block) => {
                        for t in block.isq_layers() {
                            tensors.push((t, None));
                        }
                    }
                    None => tracing::warn!(
                        "V4 MTP block is shared (decode kit already handed out); \
                         skipping its tensors for this ISQ pass"
                    ),
                }
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            // V4: `kv_norm` operates on head_dim. Emit at the HF-style
            // path `self_attn.kv_a_layernorm` for ISQ round-trip
            // compatibility.
            uvb_l
                .pp("self_attn")
                .pp("kv_a_layernorm")
                .add(&layer.attn.kv_norm);

            match &layer.moe_or_mlp {
                MoeOrMlp::Moe(moe) => {
                    uvb_l
                        .pp("mlp")
                        .pp("gate")
                        .add_tensor("weight", moe.gate.weight.clone());
                }
                MoeOrMlp::Mlp(_) => (),
            }

            match &layer.attn.q {
                QProj::Plain(_) => (),
                QProj::Lora { a: _, norm, b: _ } => {
                    uvb_l.pp("self_attn").pp("q_a_layernorm").add(norm);
                }
            }

            if let Some(comp) = &layer.attn.compressor {
                let uvb_c = uvb_l.pp("self_attn").pp("compressor");
                uvb_c.add(&comp.norm);
                uvb_c.add_tensor("ape", comp.ape.clone());
            }

            if let Some(mhc) = &layer.mhc {
                uvb_l.add_tensor("hc_attn_fn", mhc.hc_attn_fn.clone());
                uvb_l.add_tensor("hc_attn_base", mhc.hc_attn_base.clone());
                uvb_l.add_tensor("hc_attn_scale", mhc.hc_attn_scale.clone());
                uvb_l.add_tensor("hc_ffn_fn", mhc.hc_ffn_fn.clone());
                uvb_l.add_tensor("hc_ffn_base", mhc.hc_ffn_base.clone());
                uvb_l.add_tensor("hc_ffn_scale", mhc.hc_ffn_scale.clone());
            }
        }

        if let Some(mhc_head) = &self.mhc_head {
            uvb_m.add_tensor("hc_head_fn", mhc_head.hc_head_fn.clone());
            uvb_m.add_tensor("hc_head_base", mhc_head.hc_head_base.clone());
            uvb_m.add_tensor("hc_head_scale", mhc_head.hc_head_scale.clone());
        }

        self.add_mtp_block_residuals(&uvb);

        uvb.to_safetensors()
    }

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            uvb_l
                .pp("self_attn")
                .pp("kv_a_layernorm")
                .add(&layer.attn.kv_norm);

            match &layer.moe_or_mlp {
                MoeOrMlp::Moe(moe) => {
                    uvb_l
                        .pp("mlp")
                        .pp("gate")
                        .add_tensor("weight", moe.gate.weight.clone());
                }
                MoeOrMlp::Mlp(_) => (),
            }

            match &layer.attn.q {
                QProj::Plain(q) => {
                    uvb_l.pp("self_attn").pp("q_proj").add(q);
                }
                QProj::Lora { a, norm, b } => {
                    uvb_l.pp("self_attn").pp("q_a_proj").add(a);
                    uvb_l.pp("self_attn").pp("q_a_layernorm").add(norm);
                    uvb_l.pp("self_attn").pp("q_b_proj").add(b);
                }
            }
            // V4: emit at the HF-style fused path `kv_proj` (no longer an
            // a/b split). Audit §5.
            uvb_l.pp("self_attn").pp("kv_proj").add(&layer.attn.wkv);
            uvb_l.pp("self_attn").pp("o_a_proj").add(&layer.attn.wo_a);
            uvb_l.pp("self_attn").pp("o_b_proj").add(&layer.attn.wo_b);

            if let Some(comp) = &layer.attn.compressor {
                let uvb_c = uvb_l.pp("self_attn").pp("compressor");
                uvb_c.add(&comp.norm);
                uvb_c.add_tensor("ape", comp.ape.clone());
            }

            if let Some(mhc) = &layer.mhc {
                uvb_l.add_tensor("hc_attn_fn", mhc.hc_attn_fn.clone());
                uvb_l.add_tensor("hc_attn_base", mhc.hc_attn_base.clone());
                uvb_l.add_tensor("hc_attn_scale", mhc.hc_attn_scale.clone());
                uvb_l.add_tensor("hc_ffn_fn", mhc.hc_ffn_fn.clone());
                uvb_l.add_tensor("hc_ffn_base", mhc.hc_ffn_base.clone());
                uvb_l.add_tensor("hc_ffn_scale", mhc.hc_ffn_scale.clone());
            }
        }

        if let Some(mhc_head) = &self.mhc_head {
            uvb_m.add_tensor("hc_head_fn", mhc_head.hc_head_fn.clone());
            uvb_m.add_tensor("hc_head_base", mhc_head.hc_head_base.clone());
            uvb_m.add_tensor("hc_head_scale", mhc_head.hc_head_scale.clone());
        }

        self.add_mtp_block_residuals(&uvb);

        Some(uvb.to_safetensors())
    }

    fn mtp_isq_tail_len(&mut self) -> usize {
        // Mirrors the tail appended by `get_layers` /
        // `get_layers_moe_experts_only`: the MTP decoder block's ISQ layers
        // (registered only when the Arc is still unique — the same condition
        // under which get_layers pushes them).
        self.mtp_head.as_mut().map_or(0, |mtp| {
            mtp.block.as_mut().map_or(0, |block| {
                Arc::get_mut(block).map_or(0, |block| block.isq_layers().len())
            })
        })
    }

    fn load_mtp_block_from_source(
        &mut self,
        source: &crate::pipeline::UqffSourceWeights<'_>,
    ) -> candle_core::Result<bool> {
        // Only applicable when a block was requested (constructed as
        // DummyLayers under the from-UQFF load) but the artifact did not
        // cover it.
        if !self
            .mtp_head
            .as_ref()
            .is_some_and(|mtp| mtp.block.is_some())
        {
            return Ok(false);
        }

        tracing::info!(
            "V4 MTP: the UQFF artifact has no MTP decoder block tensors (baked without \
             `--mtp-depth`); loading the block UNQUANTIZED (~3GB at BF16/FP8) from the \
             source checkpoint instead. Re-bake with the current `quantize` to embed the \
             quantized block (~800MB) in the artifact."
        );

        let vb = crate::utils::varbuilder_utils::from_mmaped_safetensors(
            source.weight_files.to_vec(),
            Vec::new(),
            Some(source.dtype),
            &self.device,
            vec![None],
            true,
            // No dummy regexes: this reload wants REAL weights.
            None,
            |_| true,
            Arc::new(|_| crate::utils::varbuilder_utils::DeviceForLoadTensor::Base),
        )?;
        // Same FP8 scale-name normalization the V4 loader applies.
        let vb = mistralrs_quant::attach_rename_rules(vb, mistralrs_quant::v4_scale_rename_rules());

        let mtp_native_vb = vb.pp("mtp").pp("0");
        let mtp_hf_vb = vb.pp("mtp").pp("layers").pp("0");
        let mtp_vb = if mtp_native_vb.contains_tensor("h_proj.weight") {
            mtp_native_vb
        } else if mtp_hf_vb.contains_tensor("h_proj.weight") {
            mtp_hf_vb
        } else {
            tracing::warn!(
                "V4 MTP: source weights carry no `mtp.*` tensors; cannot reload the MTP \
                 decoder block (full-serialization UQFF without MTP residuals?)"
            );
            return Ok(false);
        };

        // The per-expert MoE constructor consults the thread-local
        // `loading_from_uqff` flag and would emit DummyLayers again; clear it
        // for the duration of this real-weight reload.
        let prev_uqff_flag = mistralrs_quant::loading_from_uqff();
        mistralrs_quant::set_loading_from_uqff(false);
        let block = MtpBlock::try_new(&self.cfg_full, mtp_vb, &*self.mapper, false, &self.device);
        mistralrs_quant::set_loading_from_uqff(prev_uqff_flag);

        match block? {
            Some(block) => {
                if let Some(mtp) = self.mtp_head.as_mut() {
                    mtp.block = Some(Arc::new(block));
                }
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

impl NormalModel for DeepSeekV4 {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(
            Vec<(Tensor, Tensor, Option<Tensor>, Option<Tensor>)>,
            &PagedAttentionInputMetadata,
        )>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
    /// Expose the MTP decode kit when the checkpoint shipped MTP tensors.
    /// Used by [`crate::pipeline::MtpSpeculativePipeline`] to drive V4's
    /// native single-step speculative draft.
    fn mtp_decode_kit(&self) -> Option<crate::pipeline::mtp_pipeline::MtpDecodeKit> {
        let head = self.mtp_head.as_ref()?;
        Some(crate::pipeline::mtp_pipeline::MtpDecodeKit {
            embed_tokens: self.embed_tokens.clone(),
            lm_head: self.lm_head.clone(),
            h_proj: head.h_proj.clone(),
            e_proj: head.e_proj.clone(),
            block: head.block.clone(),
        })
    }
}

impl AnyMoeBaseModelMixin for DeepSeekV4 {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real V4 Flash config.json (slimmed) parses cleanly through the
    /// rewritten DeepSeekV4Config. Audit §0.
    #[test]
    fn v4_flash_real_config_parses() {
        let cfg_json = r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "vocab_size": 129280,
            "hidden_size": 4096,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "first_k_dense_replace": 0,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000,
            "q_lora_rank": 1024,
            "qk_rope_head_dim": 64,
            "head_dim": 512,
            "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0],
            "sliding_window": 128,
            "compress_rope_theta": 160000,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
            "num_hash_layers": 3,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 20,
            "swiglu_limit": 10.0,
            "routed_scaling_factor": 1.5,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc"
        }"#;

        let cfg: DeepSeekV4Config = serde_json::from_str(cfg_json).expect("V4 config parses");
        // Derived qk_nope_head_dim = head_dim - qk_rope_head_dim
        assert_eq!(cfg.qk_nope_head_dim(), 448);
        assert_eq!(cfg.head_dim, 512);
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.q_lora_rank, Some(1024));
        assert_eq!(cfg.qk_rope_head_dim, 64);
        assert_eq!(cfg.hc_mult, 4);
        assert_eq!(cfg.num_hash_layers, 3);
        assert_eq!(cfg.swiglu_limit, 10.0);
        assert!(matches!(cfg.scoring_func, ScoringFunc::SqrtSoftplus));
        assert!(matches!(cfg.topk_method, TopkMethod::NoAuxTc));
        assert_eq!(cfg.routed_scaling_factor, 1.5);
        // MQA broadcast factor
        assert_eq!(cfg.n_kv_groups(), 64);
        // Compress ratios cover all 43 layers + the MTP slot
        assert_eq!(cfg.compress_ratios.len(), 44);
        // Layer 0/1 = standard, 2 = CSA, 3 = HCA, 43 = standard (MTP).
        assert_eq!(cfg.layer_compress_ratio(0), 0);
        assert_eq!(cfg.layer_compress_ratio(1), 0);
        assert_eq!(cfg.layer_compress_ratio(2), 4);
        assert_eq!(cfg.layer_compress_ratio(3), 128);
        assert_eq!(cfg.layer_compress_ratio(43), 0);
        // Out of bounds → 0
        assert_eq!(cfg.layer_compress_ratio(99), 0);
    }

    /// Backwards-compat: a V3-style fixture (with `kv_lora_rank`,
    /// `v_head_dim`, `qk_nope_head_dim`, `intermediate_size`) still parses,
    /// since those fields are now Optional. Audit §5 "Make optional".
    #[test]
    fn v3_style_v4_config_still_parses() {
        let cfg_json = r#"{
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "n_routed_experts": null,
            "num_experts_per_tok": null,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "q_lora_rank": 8,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 16,
            "kv_lora_rank": 16,
            "v_head_dim": 16,
            "head_dim": 24,
            "compress_ratios": [0, 0]
        }"#;
        let cfg: DeepSeekV4Config = serde_json::from_str(cfg_json).expect("legacy parses");
        // Explicit qk_nope_head_dim is honored when present.
        assert_eq!(cfg.qk_nope_head_dim(), 16);
        assert_eq!(cfg.head_dim, 24);
        assert_eq!(cfg.kv_lora_rank, Some(16));
        assert_eq!(cfg.v_head_dim, Some(16));
        // Defaults when V4 fields absent
        assert_eq!(cfg.hc_mult, 4);
        assert_eq!(cfg.num_key_value_heads, 1);
    }

    /// Config without explicit qk_nope_head_dim derives it from
    /// `head_dim - qk_rope_head_dim`. Audit §0.
    #[test]
    fn qk_nope_head_dim_derives_from_head_dim() {
        let cfg_json = r#"{
            "vocab_size": 16,
            "hidden_size": 16,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "qk_rope_head_dim": 4,
            "head_dim": 16
        }"#;
        let cfg: DeepSeekV4Config = serde_json::from_str(cfg_json).unwrap();
        assert_eq!(cfg.qk_nope_head_dim(), 12); // 16 - 4
        assert_eq!(cfg.head_dim, 16);
        assert_eq!(cfg.q_head_dim(), 16);
    }

    /// V4Compressor::uniform produces averaging behavior.
    #[test]
    fn v4_compressor_uniform_reduces_seq_dim() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 4;
        let ratio = 4;
        let comp = V4Compressor::uniform(ratio, head_dim, &device)?;

        let input = Tensor::ones((1, 2, 8, 4), DType::F32, &device)?;
        let out = comp.forward(&input)?;

        assert_eq!(out.dims(), &[1, 2, 2, 4]);
        let v: Vec<f32> = out.flatten_all()?.to_vec1()?;
        for x in &v {
            assert!((*x - 1.0).abs() < 1e-5);
        }
        Ok(())
    }

    /// V4Compressor with non-divisible seq length errors cleanly.
    #[test]
    fn v4_compressor_rejects_non_divisible() -> Result<()> {
        let device = Device::Cpu;
        let comp = V4Compressor::uniform(4, 8, &device)?;
        let input = Tensor::zeros((1, 1, 7, 8), DType::F32, &device)?;
        assert!(comp.forward(&input).is_err());
        Ok(())
    }

    /// CompressRatio enum maps integer config values correctly.
    #[test]
    fn compress_ratio_enum_matches_integers() {
        assert_eq!(CompressRatio::Standard.ratio(), 1);
        assert_eq!(CompressRatio::Csa.ratio(), 4);
        assert_eq!(CompressRatio::Hca.ratio(), 128);
        assert_eq!(CompressRatio::from(0), CompressRatio::Standard);
        assert_eq!(CompressRatio::from(4), CompressRatio::Csa);
        assert_eq!(CompressRatio::from(128), CompressRatio::Hca);
        assert_eq!(CompressRatio::from(7), CompressRatio::Standard);
    }

    /// CSA-ratio (4×) reduces the seq dim 4×.
    #[test]
    fn compress_kv_csa_reduces_4x() -> Result<()> {
        let device = Device::Cpu;
        let comp = V4Compressor::uniform(4, 4, &device)?;
        let k = Tensor::ones((1, 2, 16, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 2, 16, 4), DType::F32, &device)?;
        let k_c = comp.forward(&k)?;
        let v_c = comp.forward(&v)?;
        assert_eq!(k_c.dims(), &[1, 2, 4, 4]);
        assert_eq!(v_c.dims(), &[1, 2, 4, 4]);
        Ok(())
    }

    /// HCA-ratio (128×) reduces the seq dim 128×.
    #[test]
    fn compress_kv_hca_reduces_128x() -> Result<()> {
        let device = Device::Cpu;
        let comp = V4Compressor::uniform(128, 4, &device)?;
        let k = Tensor::ones((1, 2, 256, 4), DType::F32, &device)?;
        let k_c = comp.forward(&k)?;
        assert_eq!(k_c.dims(), &[1, 2, 2, 4]);
        Ok(())
    }

    /// Compress ratios produce valid shapes across multiple ratios.
    #[test]
    fn compress_kv_multiple_ratios_produce_valid_shapes() -> Result<()> {
        let device = Device::Cpu;
        for ratio in &[4, 128] {
            let comp = V4Compressor::uniform(*ratio, 16, &device)?;
            let t = 128 * ratio;
            let input = Tensor::ones((1, 4, t, 16), DType::F32, &device)?;
            let out = comp.forward(&input)?;
            assert_eq!(out.dim(2)?, t / ratio);
        }
        Ok(())
    }

    /// Config default for compress_rope_theta matches V4 Flash (160000).
    /// Audit §0 + §1 Pattern A/B/C ("compress_rope_theta = 160000").
    #[test]
    fn v4_compress_rope_theta_default_is_160000() {
        assert_eq!(default_compress_rope_theta(), 160000.0);
    }

    /// V4 Flash's sliding_window default matches the audit (128).
    #[test]
    fn v4_sliding_window_default_is_128() {
        assert_eq!(default_sliding_window(), 128);
    }

    // ===== V4Compressor real-weight loading (RUN-162) =====
    //
    // The following tests cover the compress-layer weight loader that
    // replaces the uniform-averaging fallback. They use a
    // `HashMap<String, Tensor>` backing a `ShardedVarBuilder`, mirroring
    // the pattern in `dsv4_indexer::tests`, so the loader is exercised
    // without requiring the real V4 Flash safetensors.

    /// Minimal V4 config sufficient to drive `V4Compressor::new`. Only
    /// `hidden_size` and `rms_norm_eps` are touched by the constructor;
    /// the rest are filled with defaults / synth values.
    fn compressor_test_cfg(hidden_size: usize) -> DeepSeekV4Config {
        let json = serde_json::json!({
            "vocab_size": 32,
            "hidden_size": hidden_size,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "n_shared_experts": null,
            "n_routed_experts": null,
            "num_experts_per_tok": null,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "qk_rope_head_dim": 4,
            "head_dim": 8,
            "compress_ratios": [4, 128],
        });
        serde_json::from_value(json).expect("compressor test cfg deserializes")
    }

    /// Build a `ShardedVarBuilder` from a HashMap. Same helper as
    /// `dsv4_indexer::tests::vb_from_map`; reproduced here so this file
    /// doesn't grow a `pub` test-only API on the indexer module.
    fn vb_from_map(
        map: std::collections::HashMap<String, Tensor>,
        dtype: DType,
        dev: &Device,
    ) -> ShardedVarBuilder {
        let backend: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(map);
        mistralrs_quant::ShardedSafeTensors::wrap(backend, dtype, dev.clone())
    }

    /// Populate the four V4-native compressor tensors (`wkv`, `wgate`,
    /// `norm`, `ape`) with reproducible non-zero values. Returns a HashMap
    /// that can be wrapped by `vb_from_map` and handed to
    /// `V4Compressor::new`. The seed determines the values via a simple
    /// LCG so two calls with different seeds produce different tensors.
    fn make_compressor_dual_tensors(
        hidden_size: usize,
        head_dim: usize,
        ratio: usize,
        coff: usize,
        seed: u32,
        device: &Device,
    ) -> std::collections::HashMap<String, Tensor> {
        let mut m = std::collections::HashMap::new();
        let row = coff * head_dim;
        // Deterministic non-zero pattern: idx * 0.01 + seed-derived offset.
        // Use Tensor::from_vec to avoid pulling rand into deps.
        let make = |rows: usize, cols: usize, off: f32| -> Tensor {
            let mut v = Vec::with_capacity(rows * cols);
            for i in 0..(rows * cols) {
                v.push(((i as f32) * 0.013 + off).sin() * 0.1);
            }
            Tensor::from_vec(v, (rows, cols), device).unwrap()
        };
        m.insert("wkv.weight".to_string(), make(row, hidden_size, seed as f32 * 0.07));
        m.insert(
            "wgate.weight".to_string(),
            make(row, hidden_size, seed as f32 * 0.11 + 0.5),
        );
        m.insert(
            "norm.weight".to_string(),
            Tensor::ones(head_dim, DType::F32, device).unwrap(),
        );
        // ape: small non-zero to influence forward distinctly.
        m.insert(
            "ape".to_string(),
            (Tensor::ones((ratio, coff * head_dim), DType::F32, device).unwrap() * 0.05f64)
                .unwrap(),
        );
        m
    }

    /// `V4Compressor::has_weights` correctly identifies the three valid
    /// layouts: V4-native dual (wkv + wgate), pre-fused (wkv_gate), and
    /// absent.
    #[test]
    fn v4_compressor_has_weights_detects_layouts() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;
        let coff = 2; // ratio=4 → overlap=true

        // Dual layout
        let dual_map = make_compressor_dual_tensors(16, head_dim, 4, coff, 1, &device);
        let dual_vb = vb_from_map(dual_map, DType::F32, &device);
        assert!(
            V4Compressor::has_weights(&dual_vb),
            "dual (wkv + wgate) layout should be detected"
        );

        // Pre-fused layout
        let mut fused_map: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        fused_map.insert(
            "wkv_gate.weight".to_string(),
            Tensor::zeros((2 * coff * head_dim, 16), DType::F32, &device)?,
        );
        let fused_vb = vb_from_map(fused_map, DType::F32, &device);
        assert!(
            V4Compressor::has_weights(&fused_vb),
            "pre-fused (wkv_gate) layout should be detected"
        );

        // Missing both → false
        let empty_vb = vb_from_map(std::collections::HashMap::new(), DType::F32, &device);
        assert!(
            !V4Compressor::has_weights(&empty_vb),
            "empty checkpoint should not be detected as having compressor weights"
        );

        // Partial (only wkv, no wgate) → false (would silently produce a
        // DummyLayer otherwise; this is the bug the auto-detect prevents).
        let mut partial_map: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        partial_map.insert(
            "wkv.weight".to_string(),
            Tensor::zeros((coff * head_dim, 16), DType::F32, &device)?,
        );
        let partial_vb = vb_from_map(partial_map, DType::F32, &device);
        assert!(
            !V4Compressor::has_weights(&partial_vb),
            "wkv.weight alone (without wgate.weight) should not be accepted"
        );

        Ok(())
    }

    /// HCA layer (ratio=128, coff=1): loading `wkv.weight` +
    /// `wgate.weight` produces a compressor whose `forward_from_xs`
    /// output has the right shape and is **not** zero — i.e. the loaded
    /// weights actually drive the math. Audit §0 + §1 Pattern C.
    #[test]
    fn v4_compressor_loads_dual_tensors_hca() -> Result<()> {
        let device = Device::Cpu;
        let hidden_size = 16;
        let head_dim = 8;
        let ratio = 128;
        let coff = 1; // ratio != 4 → overlap=false

        let cfg = compressor_test_cfg(hidden_size);
        let map = make_compressor_dual_tensors(hidden_size, head_dim, ratio, coff, 7, &device);
        let vb = vb_from_map(map, DType::F32, &device);
        let comp = V4Compressor::new(&cfg, vb, ratio, head_dim, &device)?;

        assert_eq!(comp.ratio, ratio);
        assert_eq!(comp.head_dim, head_dim);
        assert_eq!(comp.coff, coff);

        // Run forward on a small input. seq_len = 256 = 2 × ratio.
        let b = 1;
        let t = 256;
        let xs = Tensor::ones((b, t, hidden_size), DType::F32, &device)?;
        let out = comp.forward_from_xs(&xs)?;

        // Expected output shape: [B, T/ratio, head_dim].
        assert_eq!(out.dims(), &[b, t / ratio, head_dim]);

        // Non-trivial output: the loaded wkv + wgate weights are non-zero, so
        // the softmax-pooled (value × softmax(score + ape)) sum is strictly
        // non-zero. Check via max absolute value.
        let flat: Vec<f32> = out.flatten_all()?.to_vec1()?;
        let max_abs = flat.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            max_abs > 1e-6,
            "real-weight forward_from_xs produced an all-zero output; \
             weights were not applied. max_abs={max_abs}"
        );

        Ok(())
    }

    /// CSA layer (ratio=4, coff=2): loading dual tensors with coff=2 fuses
    /// to a wkv_gate of shape `[2*2*head_dim, hidden_size]` and produces
    /// the right output shape.
    #[test]
    fn v4_compressor_loads_dual_tensors_csa() -> Result<()> {
        let device = Device::Cpu;
        let hidden_size = 16;
        let head_dim = 8;
        let ratio = 4;
        let coff = 2; // ratio == 4 → overlap=true

        let cfg = compressor_test_cfg(hidden_size);
        let map = make_compressor_dual_tensors(hidden_size, head_dim, ratio, coff, 3, &device);
        let vb = vb_from_map(map, DType::F32, &device);
        let comp = V4Compressor::new(&cfg, vb, ratio, head_dim, &device)?;
        assert_eq!(comp.coff, 2);

        let b = 2;
        let t = 16; // T_c = 4
        let xs = Tensor::ones((b, t, hidden_size), DType::F32, &device)?;
        let out = comp.forward_from_xs(&xs)?;
        assert_eq!(out.dims(), &[b, t / ratio, head_dim]);
        Ok(())
    }

    /// Differential oracle for the RUN-161 #24 compressor-math fix.
    ///
    /// Pins the EXACT pooling semantics of `forward_from_xs` against an
    /// INDEPENDENT scalar re-derivation of the model's reference algorithm
    /// (inference/model.py `Compressor.forward`):
    ///   value = fused[..., :cd]  (wkv),  score = fused[..., cd:]  (wgate)
    ///   score += ape
    ///   overlap_transform (coff==2): previous group's overlap-half (channels
    ///     0..d) → first `ratio` slots (fill 0 / -inf for group 0); current
    ///     group's normal-half (channels d..2d) → last `ratio` slots
    ///   weights = softmax(score, over the slot axis)        [per channel]
    ///   pooled  = sum(value * weights, over the slot axis)
    ///   out     = RMSNorm(pooled)  over head_dim
    ///
    /// A regression to the pre-fix math (value/score swapped, a sigmoid gate,
    /// a plain sum instead of softmax-pool, or a missing overlap window) moves
    /// the output well outside the 1e-4 tolerance and trips this test on the
    /// CPU — no GPU required. Uses the pre-fused F32 `wkv_gate.weight` layout so
    /// the linear stays F32 (the dual-tensor path stores it in BF16), making the
    /// comparison tight.
    #[test]
    fn v4_compressor_forward_matches_scalar_reference() -> Result<()> {
        // Independent scalar reference. Returns [b, t_c, d] row-major.
        #[allow(clippy::too_many_arguments)]
        fn scalar_reference(
            fused: &[f32], // [b, t, 2*cd] row-major, cd = coff*head_dim
            ape: &[f32],   // [ratio, cd] row-major
            b: usize,
            t: usize,
            ratio: usize,
            coff: usize,
            head_dim: usize,
            eps: f32,
        ) -> Vec<f32> {
            let d = head_dim;
            let cd = coff * head_dim;
            let two_cd = 2 * cd;
            let t_c = t / ratio;
            let n_slots = coff * ratio; // overlap (coff==2) doubles the slot count
            let mut out = vec![0f32; b * t_c * d];
            // Row offset of group `grp`, slot `r` within the fused [b,t,2cd] buffer.
            let row_base = |bi: usize, grp: usize, r: usize| (bi * t + grp * ratio + r) * two_cd;
            for bi in 0..b {
                for g in 0..t_c {
                    let mut slot_val = vec![0f32; n_slots * d];
                    let mut slot_score = vec![0f32; n_slots * d];
                    if coff == 2 {
                        // First `ratio` slots: previous group's overlap-half
                        // (value channels 0..d, score channels 0..d).
                        for r in 0..ratio {
                            for c in 0..d {
                                if g > 0 {
                                    let base = row_base(bi, g - 1, r);
                                    slot_val[r * d + c] = fused[base + c];
                                    slot_score[r * d + c] = fused[base + cd + c] + ape[r * cd + c];
                                } else {
                                    slot_val[r * d + c] = 0.0;
                                    slot_score[r * d + c] = f32::NEG_INFINITY;
                                }
                            }
                        }
                        // Last `ratio` slots: current group's normal-half
                        // (value channels d..2d, score channels d..2d).
                        for r in 0..ratio {
                            let slot = ratio + r;
                            let base = row_base(bi, g, r);
                            for c in 0..d {
                                slot_val[slot * d + c] = fused[base + d + c];
                                slot_score[slot * d + c] =
                                    fused[base + cd + d + c] + ape[r * cd + d + c];
                            }
                        }
                    } else {
                        // No overlap: `ratio` slots, full cd == d channels.
                        for r in 0..ratio {
                            let base = row_base(bi, g, r);
                            for c in 0..d {
                                slot_val[r * d + c] = fused[base + c];
                                slot_score[r * d + c] = fused[base + cd + c] + ape[r * cd + c];
                            }
                        }
                    }
                    // Softmax-pool per channel over the slot axis.
                    let mut pooled = vec![0f32; d];
                    for c in 0..d {
                        let mut maxs = f32::NEG_INFINITY;
                        for s in 0..n_slots {
                            maxs = maxs.max(slot_score[s * d + c]);
                        }
                        let mut denom = 0f32;
                        for s in 0..n_slots {
                            denom += (slot_score[s * d + c] - maxs).exp();
                        }
                        let mut acc = 0f32;
                        for s in 0..n_slots {
                            let w = (slot_score[s * d + c] - maxs).exp() / denom;
                            acc += slot_val[s * d + c] * w;
                        }
                        pooled[c] = acc;
                    }
                    // RMSNorm over head_dim (norm.weight == 1).
                    let mut ms = 0f32;
                    for c in 0..d {
                        ms += pooled[c] * pooled[c];
                    }
                    ms /= d as f32;
                    let inv = 1.0 / (ms + eps).sqrt();
                    for c in 0..d {
                        out[(bi * t_c + g) * d + c] = pooled[c] * inv;
                    }
                }
            }
            out
        }

        let device = Device::Cpu;
        let eps = 1e-6f32;

        // CSA overlap path (ratio=4 → coff=2) and HCA non-overlap (coff=1).
        for &(ratio, t) in &[(4usize, 8usize), (2usize, 6usize)] {
            let coff = if ratio == 4 { 2 } else { 1 };
            let hidden_size = 6;
            let head_dim = 3;
            let cd = coff * head_dim;
            let b = 2;
            let t_c = t / ratio;

            let cfg = compressor_test_cfg(hidden_size);

            // Deterministic F32 weights via the pre-fused layout (stays F32).
            let wcount = 2 * cd * hidden_size;
            let mut wv = Vec::with_capacity(wcount);
            for i in 0..wcount {
                wv.push(((i as f32) * 0.021 + ratio as f32).sin() * 0.2);
            }
            let w = Tensor::from_vec(wv, (2 * cd, hidden_size), &device)?;

            let a_count = ratio * cd;
            let mut av = Vec::with_capacity(a_count);
            for i in 0..a_count {
                av.push(((i as f32) * 0.037).cos() * 0.1);
            }
            let ape = Tensor::from_vec(av.clone(), (ratio, cd), &device)?;

            let mut map: std::collections::HashMap<String, Tensor> =
                std::collections::HashMap::new();
            map.insert("wkv_gate.weight".to_string(), w.clone());
            map.insert(
                "norm.weight".to_string(),
                Tensor::ones(head_dim, DType::F32, &device)?,
            );
            map.insert("ape".to_string(), ape);
            let vb = vb_from_map(map, DType::F32, &device);
            let comp = V4Compressor::new(&cfg, vb, ratio, head_dim, &device)?;
            assert_eq!(comp.coff, coff);

            // Deterministic varied input so the softmax weights are non-uniform.
            let xcount = b * t * hidden_size;
            let mut xv = Vec::with_capacity(xcount);
            for i in 0..xcount {
                xv.push(((i as f32) * 0.013).sin() * 0.5 + 0.1);
            }
            let xs = Tensor::from_vec(xv, (b, t, hidden_size), &device)?;

            // Production path.
            let out = comp.forward_from_xs(&xs)?;
            assert_eq!(out.dims(), &[b, t_c, head_dim]);
            let prod: Vec<f32> = out.flatten_all()?.to_vec1()?;

            // Independent reference: fused via plain F32 matmul, scalar pooling.
            let fused_t = xs.reshape((b * t, hidden_size))?.matmul(&w.t()?)?;
            let fused: Vec<f32> = fused_t.flatten_all()?.to_vec1()?;
            let reference = scalar_reference(&fused, &av, b, t, ratio, coff, head_dim, eps);

            assert_eq!(prod.len(), reference.len());
            let mut max_abs = 0f32;
            let (mut dot, mut np, mut nr) = (0f64, 0f64, 0f64);
            for (p, r) in prod.iter().zip(reference.iter()) {
                max_abs = max_abs.max((p - r).abs());
                dot += (*p as f64) * (*r as f64);
                np += (*p as f64) * (*p as f64);
                nr += (*r as f64) * (*r as f64);
            }
            let cos = dot / (np.sqrt() * nr.sqrt() + 1e-12);
            // cos is the strong discriminator (a pre-fix regression — swapped
            // value/score, sigmoid gate, plain-sum, or missing overlap — drops
            // it far below 0.9999). max_abs guards magnitude; the ~4e-4 floor is
            // forward_autocast dtype rounding vs the pure-F32 reference matmul.
            assert!(
                max_abs < 2e-3 && cos > 0.9999,
                "compressor forward diverged from scalar reference \
                 (ratio={ratio}, coff={coff}): max_abs={max_abs}, cos={cos}"
            );
        }
        Ok(())
    }

    /// Loaded V4Compressor output differs from the uniform-fallback path.
    /// This is the core RUN-162 assertion: real weights produce a real
    /// (non-uniform-averaging) output. We compare `forward_from_xs` on a
    /// real-weight compressor against the `forward` (averaging) path on a
    /// uniform-fallback compressor — same input, two different code paths,
    /// outputs must not match.
    #[test]
    fn v4_compressor_real_weights_differ_from_uniform_fallback() -> Result<()> {
        let device = Device::Cpu;
        let hidden_size = 16;
        let head_dim = 8;
        let ratio = 128;
        let coff = 1;

        // Real-weight compressor via dual tensors.
        let cfg = compressor_test_cfg(hidden_size);
        let map = make_compressor_dual_tensors(hidden_size, head_dim, ratio, coff, 11, &device);
        let vb = vb_from_map(map, DType::F32, &device);
        let real = V4Compressor::new(&cfg, vb, ratio, head_dim, &device)?;

        // Uniform-fallback compressor (the path that was previously used).
        let uniform = V4Compressor::uniform(ratio, head_dim, &device)?;

        // Run both compressors. The real path uses forward_from_xs on
        // `[B, T, hidden]`. The uniform path uses the legacy 4-D forward
        // on `[B, H, T, D]` (its forward_from_xs is unusable because its
        // dummy wkv_gate has shape [_, 1]). The OUTPUT shapes match
        // (`[B, T/ratio, head_dim]` for real vs `[B, H, T/ratio, head_dim]`
        // collapsed), and the OUTPUT VALUES must differ to prove the
        // fallback was replaced.
        let b = 1;
        let t = 256;
        let xs = Tensor::ones((b, t, hidden_size), DType::F32, &device)?;
        let real_out = real.forward_from_xs(&xs)?;
        let real_vec: Vec<f32> = real_out.flatten_all()?.to_vec1()?;

        // Uniform fallback's averaging path on all-ones input yields all
        // ones; the dummy wkv_gate is never invoked here.
        let kv_4d = Tensor::ones((b, 1, t, head_dim), DType::F32, &device)?;
        let uniform_out = uniform.forward(&kv_4d)?;
        let uniform_vec: Vec<f32> = uniform_out.flatten_all()?.to_vec1()?;

        // Real output must NOT equal the uniform all-ones output. Allow a
        // generous tolerance: if even one element differs by > 1e-3, the
        // path is genuinely different.
        assert_eq!(real_vec.len(), uniform_vec.len());
        let diff_count = real_vec
            .iter()
            .zip(uniform_vec.iter())
            .filter(|(r, u)| (*r - *u).abs() > 1e-3)
            .count();
        assert!(
            diff_count > 0,
            "real-weight compressor produced output identical to uniform fallback \
             ({} elements). Loaded weights had no effect on the forward pass.",
            real_vec.len()
        );

        Ok(())
    }

    /// Pre-fused `wkv_gate.weight` layout still loads (backwards-compat
    /// with SGLang's cached-fusion path and Arc's own synthetic
    /// `wkv_gate`-named fixtures).
    #[test]
    fn v4_compressor_loads_prefused_wkv_gate() -> Result<()> {
        let device = Device::Cpu;
        let hidden_size = 16;
        let head_dim = 8;
        let ratio = 128;
        let coff = 1;

        let cfg = compressor_test_cfg(hidden_size);
        let mut map: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        // Non-zero fused weight: [2*coff*head_dim, hidden_size].
        let mut v: Vec<f32> = Vec::with_capacity(2 * coff * head_dim * hidden_size);
        for i in 0..(2 * coff * head_dim * hidden_size) {
            v.push(((i as f32) * 0.017).sin() * 0.1);
        }
        map.insert(
            "wkv_gate.weight".to_string(),
            Tensor::from_vec(v, (2 * coff * head_dim, hidden_size), &device)?,
        );
        map.insert(
            "norm.weight".to_string(),
            Tensor::ones(head_dim, DType::F32, &device)?,
        );
        // No ape → zeros fallback inside V4Compressor::new.

        let vb = vb_from_map(map, DType::F32, &device);
        let comp = V4Compressor::new(&cfg, vb, ratio, head_dim, &device)?;

        let b = 1;
        let t = 256;
        let xs = Tensor::ones((b, t, hidden_size), DType::F32, &device)?;
        let out = comp.forward_from_xs(&xs)?;
        assert_eq!(out.dims(), &[b, t / ratio, head_dim]);
        Ok(())
    }

    /// `V4Compressor::new` returns an Err (rather than silently producing
    /// a DummyLayer) when neither layout is present in the checkpoint.
    /// Callers must gate via `has_weights` and fall back to `uniform`.
    #[test]
    fn v4_compressor_new_errors_without_weights() {
        let device = Device::Cpu;
        let cfg = compressor_test_cfg(16);
        let vb = vb_from_map(std::collections::HashMap::new(), DType::F32, &device);
        let res = V4Compressor::new(&cfg, vb, 128, 8, &device);
        assert!(
            res.is_err(),
            "V4Compressor::new without wkv_gate/wkv+wgate should Err"
        );
    }

    // ============================================================
    // RUN-167: PagedAttention + MLA-cache compress dispatch tests.
    //
    // V4 has two cache backends per Attention::forward: PagedAttention
    // (engine-level batched serving) and KvCache (single-stream / CPU
    // tests). RUN-155 wired CSA/HCA through `dsv4_attention` on the
    // KvCache path; RUN-167 extends that to PagedAttention by writing the
    // K/V into the paged storage and gathering the full sequence back
    // before calling `dsv4_attention`.
    //
    // Both code paths converge on the same downstream call:
    //
    //     dsv4_attention(&q, &k_full, &v_full, ..., compressor.as_ref(),
    //                    &sdpa_params, dsv4_cfg)
    //
    // so the tests below assert the contract that downstream call enforces:
    // for a CSA/HCA layer with a real compressor, the output shape matches
    // the dense-over-compressed + sliding-window-blend pattern, and the
    // result differs measurably from plain SDPA over the uncompressed K/V.
    //
    // The PagedAttention dispatch arm itself is CUDA-only at runtime (the
    // gather kernel only exists in the CUDA + Metal backends), so we cannot
    // exercise it on a CPU build. The dispatch's correctness is therefore
    // verified by:
    //   (a) these tests over the post-gather K/V tensor (the value the
    //       gather kernel would return),
    //   (b) the cu_seqlens helper tests in `paged_attention::cu_seqlens_tests`,
    //   (c) the existing dsv4_attention tests covering the algorithmic core.

    /// Compress dispatch with a CSA layer routes through the compressor:
    /// the output shape and per-element values match
    /// `dsv4_attention(..., compressor.as_ref(), ...)` exactly, and the
    /// output differs from plain SDPA over the uncompressed K/V. This
    /// covers the post-gather path under PagedAttention (RUN-167) since
    /// both the PA and KvCache dispatches converge on the same downstream
    /// call.
    #[test]
    fn paged_attn_compress_dispatch_routes_csa_through_compressor() -> Result<()> {
        use crate::models::dsv4_attention::{dsv4_attention, Dsv4AttentionConfig};
        use crate::attention::SdpaParams;
        use crate::layers::Sdpa;
        use crate::pipeline::text_models_inputs_processor::FlashParams;

        let device = Device::Cpu;
        let head_dim = 16;
        let n_kv_heads = 1;
        let n_q_heads = 1;
        let t_q = 4;
        let t_k = 16; // CSA ratio=4 → 4 compressed entries.

        // Synthetic Q/K/V — the post-`gather_kv_cache` shape `[B, H, T, D]`
        // mirrors what `PagedAttention::cache_write_and_gather` returns.
        let q = Tensor::from_vec(
            (0..(n_q_heads * t_q * head_dim))
                .map(|i| ((i as f32) * 0.13).sin())
                .collect::<Vec<f32>>(),
            (1, n_q_heads, t_q, head_dim),
            &device,
        )?;
        let k = Tensor::from_vec(
            (0..(n_kv_heads * t_k * head_dim))
                .map(|i| ((i as f32) * 0.07).cos())
                .collect::<Vec<f32>>(),
            (1, n_kv_heads, t_k, head_dim),
            &device,
        )?;
        let v = k.clone();

        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };
        let flash_params = FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: std::collections::HashMap::new(),
            cumulative_seqlens_k: std::collections::HashMap::new(),
            causal: false,
        };
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Csa,
            sliding_window: 4,
        };

        // Compressed KV (precomputed by `Attention::compressed_kv` in the real
        // path): T_c = t_k / ratio = 4 entries, shape [B, 1, T_c, D].
        let t_c = t_k / 4;
        let comp = Tensor::from_vec(
            (0..(n_kv_heads * t_c * head_dim))
                .map(|i| ((i as f32) * 0.09).sin())
                .collect::<Vec<f32>>(),
            (1, n_kv_heads, t_c, head_dim),
            &device,
        )?;
        let out = dsv4_attention(
            &q,
            &k,
            &v,
            Some(&comp),
            None,
            &flash_params,
            &sdpa_params,
            cfg,
        )?;
        // Shape contract: query-aligned `[B, H, T_q, D]`. The compressed
        // sequence collapses to T_c=4 internally, but the dispatch projects
        // back to T_q on the output axis (same as plain SDPA's signature).
        assert_eq!(out.dims(), &[1, n_q_heads, t_q, head_dim]);

        // Non-uniform output: with non-uniform Q/K/V (sin/cos seeded
        // tensors) the per-token outputs should differ. A uniform output
        // would mean attention collapsed degenerate-ly — that's the
        // failure mode RUN-167 was supposed to fix.
        let data: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|x| x.is_finite()));
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let range = sorted.last().unwrap() - sorted.first().unwrap();
        assert!(
            range > 1e-6,
            "compress dispatch produced uniform output (range={range})"
        );

        // Differs from plain SDPA over uncompressed K/V: this is the proof
        // that compress IS being applied. If the dispatch fell through to
        // dense SDPA (the pre-RUN-167 bug under PA), `out` would equal
        // `sdpa_dense` byte-for-byte (sliding-window blend would still be
        // applied symmetrically).
        let sdpa_dense = Sdpa.run_attention(&q, &k, &v, None, Some(&flash_params), &sdpa_params)?;
        let a: Vec<f32> = sdpa_dense.flatten_all()?.to_vec1()?;
        let mut max_diff: f32 = 0.0;
        for (ov, av) in data.iter().zip(a.iter()) {
            max_diff = max_diff.max((ov - av).abs());
        }
        assert!(
            max_diff > 1e-5,
            "compress dispatch output matches plain SDPA exactly — compressor was not invoked"
        );

        Ok(())
    }

    /// Same shape contract for HCA (ratio=128). Verifies that the larger
    /// compression ratio still produces a finite, query-aligned output and
    /// that the dispatch invokes the compressor (output differs from dense
    /// SDPA). Mirrors the PagedAttention post-gather path under HCA layers.
    #[test]
    fn paged_attn_compress_dispatch_routes_hca_through_compressor() -> Result<()> {
        use crate::models::dsv4_attention::{dsv4_attention, Dsv4AttentionConfig};
        use crate::attention::SdpaParams;
        use crate::layers::Sdpa;
        use crate::pipeline::text_models_inputs_processor::FlashParams;

        let device = Device::Cpu;
        let head_dim = 16;
        let n_kv_heads = 1;
        let n_q_heads = 1;
        let t_q = 2;
        let t_k = 128; // HCA ratio=128 → 1 compressed entry.

        let q = Tensor::from_vec(
            (0..(n_q_heads * t_q * head_dim))
                .map(|i| ((i as f32) * 0.11).sin())
                .collect::<Vec<f32>>(),
            (1, n_q_heads, t_q, head_dim),
            &device,
        )?;
        let k = Tensor::from_vec(
            (0..(n_kv_heads * t_k * head_dim))
                .map(|i| ((i as f32) * 0.05).cos())
                .collect::<Vec<f32>>(),
            (1, n_kv_heads, t_k, head_dim),
            &device,
        )?;
        let v = k.clone();

        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };
        let flash_params = FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: std::collections::HashMap::new(),
            cumulative_seqlens_k: std::collections::HashMap::new(),
            causal: false,
        };
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Hca,
            sliding_window: 8,
        };

        // Compressed KV: T_c = t_k / ratio = 1 entry, shape [B, 1, T_c, D].
        let t_c = t_k / 128;
        let comp = Tensor::from_vec(
            (0..(n_kv_heads * t_c * head_dim))
                .map(|i| ((i as f32) * 0.03).cos())
                .collect::<Vec<f32>>(),
            (1, n_kv_heads, t_c, head_dim),
            &device,
        )?;
        let out = dsv4_attention(
            &q,
            &k,
            &v,
            Some(&comp),
            None,
            &flash_params,
            &sdpa_params,
            cfg,
        )?;
        assert_eq!(out.dims(), &[1, n_q_heads, t_q, head_dim]);
        let data: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|x| x.is_finite()));

        let sdpa_dense = Sdpa.run_attention(&q, &k, &v, None, Some(&flash_params), &sdpa_params)?;
        let a: Vec<f32> = sdpa_dense.flatten_all()?.to_vec1()?;
        let mut max_diff: f32 = 0.0;
        for (ov, av) in data.iter().zip(a.iter()) {
            max_diff = max_diff.max((ov - av).abs());
        }
        assert!(
            max_diff > 1e-5,
            "HCA dispatch matches plain SDPA exactly — compressor not invoked"
        );
        Ok(())
    }

    /// Standard layers (compress_ratio == 0) must bypass the compressed
    /// branch entirely — they are sliding-window + sink only (the reference
    /// SWA-onlys ratio-0 layers; see `dsv4_attention` module docs). This test
    /// guards against the inverse failure: a Standard layer accidentally
    /// attending over compressed KV.
    #[test]
    fn paged_attn_compress_dispatch_skips_compressor_for_standard_layers() -> Result<()> {
        use crate::models::dsv4_attention::{dsv4_attention, Dsv4AttentionConfig};
        use crate::attention::SdpaParams;
        use crate::layers::Sdpa;
        use crate::pipeline::text_models_inputs_processor::FlashParams;

        let device = Device::Cpu;
        let (b, h, t, d) = (1, 1, 8, 16);
        let q = Tensor::from_vec(
            (0..(b * h * t * d))
                .map(|i| (i as f32) * 0.03)
                .collect::<Vec<f32>>(),
            (b, h, t, d),
            &device,
        )?;
        let k = Tensor::from_vec(
            (0..(b * h * t * d))
                .map(|i| (i as f32) * 0.05)
                .collect::<Vec<f32>>(),
            (b, h, t, d),
            &device,
        )?;
        let v = k.clone();

        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0 / (d as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };
        let flash_params = FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: std::collections::HashMap::new(),
            cumulative_seqlens_k: std::collections::HashMap::new(),
            causal: false,
        };
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Standard,
            sliding_window: 4,
        };

        // Standard layers must bypass the compressed branch entirely: even if
        // a caller (paranoically) supplies compressed KV, the Standard
        // dispatch drops it before the union — the output must be identical
        // to the no-compressed-KV call. (The real `Attention::compressed_kv`
        // returns `None` for Standard layers anyway.)
        let out = dsv4_attention(&q, &k, &v, None, None, &flash_params, &sdpa_params, cfg)?;
        let stray_comp = Tensor::from_vec(
            (0..(b * h * 2 * d))
                .map(|i| (i as f32) * 0.02)
                .collect::<Vec<f32>>(),
            (b, h, 2, d),
            &device,
        )?;
        let out_with_stray = dsv4_attention(
            &q,
            &k,
            &v,
            Some(&stray_comp),
            None,
            &flash_params,
            &sdpa_params,
            cfg,
        )?;
        let a: Vec<f32> = out.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = out_with_stray.flatten_all()?.to_vec1()?;
        for (av, bv) in a.iter().zip(b.iter()) {
            assert!(
                (av - bv).abs() < 1e-6,
                "Standard layer attended over stray compressed KV: {av} vs {bv}"
            );
        }
        // And the windowed semantics hold: with window == 4 < t == 8 the
        // output must NOT equal dense (unwindowed) SDPA.
        let sdpa_dense = Sdpa.run_attention(&q, &k, &v, None, Some(&flash_params), &sdpa_params)?;
        let dn: Vec<f32> = sdpa_dense.flatten_all()?.to_vec1()?;
        let max_diff = a
            .iter()
            .zip(dn.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "Standard layer output matches dense SDPA — sliding window not applied"
        );
        Ok(())
    }
}
