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
//! - MTP transformer-block forward (load-only here; `MtpSpeculativePipeline`
//!   does drafting via `MtpDecodeKit` and only needs `embed_tokens/
//!   lm_head/h_proj/e_proj`).
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
    /// Returns `[B, T_c, head_dim]` with `T_c = T / ratio`. SGLang
    /// `compressor.py:366-392`.
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
        let work_dtype = xs.dtype();

        let wkv = self.wkv_gate.forward_autocast(xs)?;
        let split = wkv.split(
            &[self.coff * self.head_dim, self.coff * self.head_dim],
            D::Minus1,
        )?;
        let gate = split[0].clone();
        let val = split[1].clone();
        let score = (candle_nn::ops::sigmoid(&gate)? * val)?;

        let score_grouped = score.reshape((b, t_c, self.ratio, self.coff * self.head_dim))?;
        let ape = self
            .ape
            .to_dtype(work_dtype)?
            .reshape((1, 1, self.ratio, self.coff * self.head_dim))?;
        let score_grouped = score_grouped.broadcast_add(&ape)?;
        let aggregated = score_grouped.sum(2)?;

        let flat = aggregated.reshape((b * t_c * self.coff, self.head_dim))?;
        let normed = flat.apply(&self.norm)?;
        let collapsed = normed
            .reshape((b * t_c, self.coff, self.head_dim))?
            .sum(1)?;

        collapsed.reshape((b, t_c, self.head_dim))
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
    /// RoPE — either `rope_theta`-based (standard layers 0/1/42) or
    /// `compress_rope_theta`-based (compress layers). Caller picks at
    /// construction. Audit §0 + §8 P1 item 11.
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    cfg: DeepSeekV4Config,
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
    /// R2: per-layer history of the compressor input `xs` (post-layernorm
    /// attention input) so the compressed branch sees compressed *history*
    /// during decode, not just the current token. CSA/HCA only, interior-
    /// mutable, reset at sequence start (seqlen_offset==0). Single-sequence
    /// (non-paged) correctness; paged/multi-seq is the follow-up.
    xs_history: Option<std::sync::Mutex<crate::kv_cache::SingleCache>>,
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
        let sinks_for_sdpa = if let Some(ref s) = attn_sink {
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
            xs_history: if compress_ratio != CompressRatio::Standard {
                Some(std::sync::Mutex::new(crate::kv_cache::SingleCache::new(
                    1,
                    cfg.max_position_embeddings,
                    256,
                )))
            } else {
                None
            },
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

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<(
            (Tensor, Tensor, Option<Tensor>, Option<Tensor>),
            &PagedAttentionInputMetadata,
        )>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;
        let head_dim = self.cfg.head_dim;

        v4_nan_dbg(xs, "attn.in");

        // R2: the compressor (forward_from_xs) must see the *history* of the
        // attention input during decode, not just the current token. Keep a
        // per-layer history of `xs` and feed the full history to the
        // compressed branch; the q/k/v projections below still use the
        // current `xs`. Reset at sequence start (all seqlen_offsets == 0).
        let xs_for_compressor = if let Some(hist) = &self.xs_history {
            let mut h = hist.lock().unwrap();
            if seqlen_offsets.iter().all(|&o| o == 0) {
                h.reset();
            }
            h.append(&xs.contiguous()?)?;
            h.current_data()?.expect("xs_history present after append")
        } else {
            xs.clone()
        };
        // 1. Q projection (LoRA). [B, T, hidden] → [B, T, n_heads*head_dim]
        //    → reshape to [B, n_heads, T, head_dim]. Audit §3.
        let q = self.q.forward(xs)?;
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

        // 2. K/V projection: single fused wkv. [B, T, hidden] → [B, T,
        //    head_dim] → kv_norm → reshape to [B, num_kv_heads=1, T,
        //    head_dim]. Audit §0 + §3.
        let kv_raw = self.wkv.forward_autocast(xs)?;
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

        // V4 QAT: FP8-simulate the non-rope dims of K (reference model.py:506
        // `act_quant(kv[..., :-rd], 64, ..., inplace=True)`). The model was
        // trained with the KV non-rope dims round-tripped through block-wise
        // FP8; feeding full BF16 is out-of-distribution. V=K so v inherits it.
        let k = act_quant_kv_nope(&k, self.cfg.qk_rope_head_dim)?;
        v4_stat_dbg(&k, "attn.k_actquant");

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
        // Some paged_attn arm). For each, `compress_ratio == Standard`
        // routes through dense MLA-style SDPA / paged kernel exactly as in
        // V3, and `compress_ratio == Csa | Hca` routes through the V4
        // hybrid `dsv4_attention` dispatch (compressor.forward over the
        // full K/V + sliding-window local branch, blended 0.5/0.5).
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
        let mut attn_out = match &self.paged_attn {
            Some(paged_attn) if self.compress_ratio == CompressRatio::Standard => match metadata {
                Some(((key_cache, value_cache, _, _), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            Some(paged_attn) => match metadata {
                // CSA / HCA + PagedAttention: write to paged cache, gather
                // the full context, then run the V4 compress dispatch over
                // the gathered K/V. RUN-167.
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
                        &xs_for_compressor,
                        attention_mask,
                        flash_params,
                        self.compressor.as_ref(),
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
                        &xs_for_compressor,
                        attention_mask,
                        flash_params,
                        self.compressor.as_ref(),
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
                // None here => attends over zero-padding (finite, not yet
                // correct) until set_graph_mode_mask is wired.
                let gmask = crate::layers::graph_mode_mask();
                super::dsv4_attention::dsv4_attention(
                    &q,
                    &k_full,
                    &v_full,
                    &xs_for_compressor,
                    gmask.as_ref(),
                    flash_params,
                    self.compressor.as_ref(),
                    &self.sdpa_params,
                    dsv4_cfg,
                )?
            }
            None => {
                let (k_cached, v_cached) = kv_cache.append(&k, &v)?;
                super::dsv4_attention::dsv4_attention(
                    &q,
                    &k_cached,
                    &v_cached,
                    &xs_for_compressor,
                    attention_mask,
                    flash_params,
                    self.compressor.as_ref(),
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
        // Normalize to [B, H, T, head_dim]. Three input layouts reach here:
        //   * PagedAttention (Standard layers) returns 3-D [B*T, H, head_dim].
        //   * SDPA prefill (mask present) is already 4-D [B, H, T, head_dim].
        //   * SDPA decode (no mask) comes back 4-D [B, T, H, head_dim].
        // Without the 3-D case, paged decode transposes [B*T,H,D] -> [.,D,H] and
        // hits forward_inverse_tail with a rank-3 tensor -> "unexpected rank,
        // expected: 4, got: 3". (RUN-161)
        let attn_out_bhtd = if attn_out.rank() == 3 {
            let (_bt, h, hd) = attn_out.dims3()?;
            attn_out.reshape((bs, seq_len, h, hd))?.transpose(1, 2)?
        } else if attention_mask.is_some() {
            attn_out.clone()
        } else {
            attn_out.transpose(1, 2)?
        };
        // V4: inverse-RoPE the last qk_rope_head_dim dims of the attention
        // output. V=K carries the key's RoPE on those dims, so the output must
        // be de-rotated (reference inference/model.py:534). Missing this leaves
        // the value content wrongly position-rotated -> attention output is
        // corrupted on rope dims -> incoherent generation (RUN-161).
        let attn_out_bhtd = self.rotary_emb.forward_inverse_tail(
            &attn_out_bhtd,
            self.cfg.qk_rope_head_dim,
            seqlen_offsets,
        )?;
        // -> [B, T, H*head_dim]
        attn_out = attn_out_bhtd.transpose(1, 2)?.reshape((bs, seq_len, ()))?;

        let o_groups = self.cfg.o_groups.unwrap_or(1);
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

            // Dequantize wo_a weight: (G*R, D) → (G, R, D).
            // Temporary ~67MB per layer, freed after this forward call.
            let wo_a_w = self.wo_a.dequantize_w()?;
            let o_lora_rank = wo_a_w.dim(0)? / o_groups;
            let wo_a_grouped = wo_a_w.reshape((o_groups, o_lora_rank, per_group))?;

            // wo_a_grouped shape: (G, R, D)
            let o_lora_rank = wo_a_grouped.dim(1)?;

            // bmm: (G, tokens, D) @ (G, D, R) → (G, tokens, R)
            let attn_perm = attn_grouped.permute((1, 0, 2))?.contiguous()?;
            let wo_a_t = wo_a_grouped.permute((0, 2, 1))?.contiguous()?;
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
}

impl MoeGate {
    fn new(
        cfg: &DeepSeekV4Config,
        vb: ShardedVarBuilder,
        n_routed_experts: usize,
    ) -> Result<Self> {
        let weight = vb.get((n_routed_experts, cfg.hidden_size), "weight")?;
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
        Ok(Self {
            weight,
            cfg: cfg.clone(),
            top_k: cfg.num_experts_per_tok.unwrap_or(6),
            n_routed_experts,
            e_score_correction_bias,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
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

        let (mut topk_weight, topk_idx) = match self.cfg.topk_method {
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
        };

        if matches!(
            self.cfg.scoring_func,
            ScoringFunc::Sigmoid | ScoringFunc::SqrtSoftplus
        ) {
            let denominator = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = topk_weight.broadcast_div(&denominator)?;
        }

        topk_weight = (topk_weight * self.cfg.routed_scaling_factor)?;
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
        )?;
        Ok(Self {
            experts,
            shared_experts,
            gate,
            layer_idx,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
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
        let (topk_idx, topk_weight) = self.gate.forward(xs)?;
        v4_stat_dbg(&topk_weight, "moe.topk_w");
        v4_stat_dbg(&topk_idx, "moe.topk_idx");
        let mut y = self.experts.forward(xs, topk_weight, &topk_idx)?;
        y = y.reshape((b_size, seq_len, hidden_dim))?;
        v4_stat_dbg(&y, "moe.routed");

        if let Some(ref shared_experts) = self.shared_experts {
            let shared_out = shared_experts.forward(&identity)?;
            v4_stat_dbg(&shared_out, "moe.shared");
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
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
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

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<(
            (Tensor, Tensor, Option<Tensor>, Option<Tensor>),
            &PagedAttentionInputMetadata,
        )>,
        flash_params: &FlashParams,
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
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
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
        metadata: Option<(
            (Tensor, Tensor, Option<Tensor>, Option<Tensor>),
            &PagedAttentionInputMetadata,
        )>,
        flash_params: &FlashParams,
        layer_idx: usize,
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

        // === ATTN BLOCK ===
        let residual_attn = xs_4d;
        let (y_attn, post_attn, comb_attn) = mhc.attn_pre(residual_attn)?;
        v4_nan_dbg(&y_attn, &format!("L{li}.attn_pre.y"));
        v4_nan_dbg(&post_attn, &format!("L{li}.attn_pre.post"));
        v4_nan_dbg(&comb_attn, &format!("L{li}.attn_pre.comb"));
        let y_attn_normed = self.input_layernorm.forward(&y_attn)?;
        v4_nan_dbg(&y_attn_normed, &format!("L{li}.input_layernorm"));
        let attn_out = self.attn.forward(
            &y_attn_normed,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        v4_nan_dbg(&attn_out, &format!("L{li}.attn_out"));
        let xs_4d = mhc.mix_post_4d(&attn_out, residual_attn, &post_attn, &comb_attn)?;
        v4_nan_dbg(&xs_4d, &format!("L{li}.mix_post_attn"));

        // === FFN BLOCK ===
        let residual_ffn = &xs_4d;
        let (y_ffn, post_ffn, comb_ffn) = mhc.ffn_pre(residual_ffn)?;
        v4_nan_dbg(&y_ffn, &format!("L{li}.ffn_pre.y"));
        let y_ffn_normed = self.post_attention_layernorm.forward(&y_ffn)?;
        v4_nan_dbg(&y_ffn_normed, &format!("L{li}.post_attention_layernorm"));
        let ffn_out = self.moe_or_mlp.forward(&y_ffn_normed)?;
        v4_nan_dbg(&ffn_out, &format!("L{li}.ffn_out"));
        let out = mhc.mix_post_4d(&ffn_out, residual_ffn, &post_ffn, &comb_ffn)?;
        v4_nan_dbg(&out, &format!("L{li}.mix_post_ffn"));
        Ok(out)
    }
}

/// V4 MTP head.
///
/// V4 Flash actually ships a full transformer decoder layer + 256-expert
/// MoE block at `mtp.0.*`. We load the two projection layers `h_proj` and
/// `e_proj` — the only tensors `MtpDecodeKit` consumes for the Tier A
/// drafting in `pipeline/mtp_pipeline.rs::MtpDecodeKit::step`.
///
/// The remaining MTP weights (full attn + ffn + hc_*) are loaded lazily by
/// the MTP-pipeline follow-up agent that turns MTP into a real transformer
/// step. Audit §0 + §2 ("MTP module").
pub struct MtpHead {
    pub h_proj: Arc<dyn QuantMethod>,
    pub e_proj: Arc<dyn QuantMethod>,
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
            rope_scaling: cfg.rope_scaling.clone(),
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
                    (Ok(h_proj), Ok(e_proj)) => Some(MtpHead { h_proj, e_proj }),
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

        Ok(Self {
            lm_head,
            embed_tokens,
            norm,
            layers,
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
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
                    &mut cache[i],
                    metadata
                        .as_ref()
                        .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                    flash_params,
                    i,
                )?;
                v4_stat_dbg(&xs_4d, &format!("L{i}"));
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
                    &mut cache[i],
                    metadata
                        .as_ref()
                        .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                    flash_params,
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

        Some(uvb.to_safetensors())
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
    fn mtp_decode_kit(
        &self,
    ) -> Option<crate::pipeline::mtp_pipeline::MtpDecodeKit> {
        let head = self.mtp_head.as_ref()?;
        Some(crate::pipeline::mtp_pipeline::MtpDecodeKit {
            embed_tokens: self.embed_tokens.clone(),
            lm_head: self.lm_head.clone(),
            h_proj: head.h_proj.clone(),
            e_proj: head.e_proj.clone(),
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

        // Non-trivial output: the loaded wkv + wgate weights are non-zero
        // and ape is 0.05, so the post-sigmoid * val * (1 + ape) sum is
        // strictly non-zero. Check via max absolute value.
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

        let compressor = V4Compressor::uniform(4, head_dim, &device)?;
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

        // Compress dispatch. T_xs = t_k (divisible by ratio) drives the real
        // forward_from_xs compressed branch; uniform compressor hidden dim 1.
        let xs = Tensor::zeros((1, t_k, 1), DType::F32, &device)?;
        let out = dsv4_attention(
            &q,
            &k,
            &v,
            &xs,
            None,
            &flash_params,
            Some(&compressor),
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

        let compressor = V4Compressor::uniform(128, head_dim, &device)?;
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

        // T_xs = t_k (divisible by ratio 128) → real forward_from_xs branch.
        let xs = Tensor::zeros((1, t_k, 1), DType::F32, &device)?;
        let out = dsv4_attention(
            &q,
            &k,
            &v,
            &xs,
            None,
            &flash_params,
            Some(&compressor),
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

    /// Standard layers (compress_ratio == 0) under both dispatch arms are
    /// expected to bypass the compressor entirely and route through plain
    /// SDPA. This test guards against the inverse failure: a Standard layer
    /// accidentally going through the compress branch.
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

        // Pass `Some(&compressor)` to be paranoid — Standard layers should
        // never invoke it. Compare bytewise against plain Sdpa.
        let compressor = V4Compressor::uniform(4, d, &device)?;
        // Standard dispatch returns before the compressor; xs is unused.
        let xs = Tensor::zeros((b, t, 1), DType::F32, &device)?;
        let out = dsv4_attention(
            &q,
            &k,
            &v,
            &xs,
            None,
            &flash_params,
            Some(&compressor),
            &sdpa_params,
            cfg,
        )?;
        let sdpa_dense = Sdpa.run_attention(&q, &k, &v, None, Some(&flash_params), &sdpa_params)?;
        let a: Vec<f32> = out.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = sdpa_dense.flatten_all()?.to_vec1()?;
        for (av, bv) in a.iter().zip(b.iter()) {
            assert!(
                (av - bv).abs() < 1e-6,
                "Standard layer diverged from plain SDPA: {av} vs {bv}"
            );
        }
        Ok(())
    }
}
