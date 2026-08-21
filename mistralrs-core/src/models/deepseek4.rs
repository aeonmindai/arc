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
//!   - `0`  → standard MQA (layers 0 and 1, plus slot 43 = the MTP block;
//!     layer 42 is ratio 4, NOT standard)
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
    moe::{ExpertParallelPlan, ExpertPlacement, MoEExperts, MoEExpertsConfig},
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

use crate::kv_cache::XsRollingCache;

use super::dsv4_kv_fp8::V4PackedK;
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
serde_default_fn!(usize, default_ep_size, 1);
serde_default_fn!(usize, default_hc_mult, 4);
serde_default_fn!(f32, default_hc_eps, 1.0e-6);
serde_default_fn!(usize, default_hc_sinkhorn_iters, 20);
serde_default_fn!(f32, default_swiglu_limit, 10.0);
serde_default_fn!(usize, default_n_group, 1);
serde_default_fn!(usize, default_topk_group, 1);
serde_default_fn!(bool, default_norm_topk_prob, true);

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
/// <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json>.
///
/// The published file is reproduced verbatim as `V4_FLASH_CONFIG_JSON` in this
/// module's test block; every "absent"/"present" claim below is asserted there
/// rather than assumed.
///
/// **Verified absent vs V3**: `kv_lora_rank`, `v_head_dim`, `qk_nope_head_dim`,
/// `intermediate_size`, `first_k_dense_replace`, `n_group`, `topk_group`, and
/// (inside `rope_scaling`) `mscale` / `mscale_all_dim`.
/// **Verified present**: `head_dim`, `num_key_value_heads`, `num_hash_layers`
/// (that spelling — *not* `n_hash_layers`), `hc_mult`, `hc_eps`,
/// `hc_sinkhorn_iters`, `swiglu_limit`, `norm_topk_prob`, `compress_ratios`,
/// `compress_rope_theta`, `o_lora_rank`, `o_groups`, `index_*`, `rope_scaling`
/// (yarn, `factor: 16`, `original_max_position_embeddings: 65536`).
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
    /// **Verified absent** from the published V4 Flash `config.json` ⇒ default
    /// 1 ⇒ `MoeGate::forward` takes the ungrouped top-k branch, matching
    /// SGLang's unconditional `use_grouped_topk=False` override for V4.
    #[serde(default = "default_n_group")]
    pub(crate) n_group: usize,
    /// **Verified absent** from the published V4 Flash `config.json`. Default 1.
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
    /// Expert-parallel world size: how many ranks the `n_routed_experts`
    /// routed experts are split across. `1` (the default, and what the
    /// published V4 Flash `config.json` carries) is plain replication and is
    /// bit-for-bit the pre-EP path.
    ///
    /// Published in the HF config as `ep_size`; overridable at run time with
    /// `ARC_EP_SIZE` (see [`DeepSeekV4Config::effective_ep_size`]). This field
    /// was deserialized and never read anywhere in the workspace until
    /// wave60-CK — the twelfth "wired but never invoked" case in this repo.
    #[serde(default = "default_ep_size")]
    pub(crate) ep_size: usize,
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
    ///
    /// **Verified present** in the published `config.json` (`"swiglu_limit":
    /// 10.0`), so the reference's `getattr(config, "swiglu_limit", None)`
    /// resolves and the clamp is mandatory, not optional.
    #[serde(default = "default_swiglu_limit")]
    pub(crate) swiglu_limit: f32,
    /// Whether the selected top-k routing weights are renormalized to sum to
    /// 1 before `routed_scaling_factor` is applied (V4 Flash: `true`).
    ///
    /// The reference passes this straight through as
    /// `renormalize=config.norm_topk_prob` (`deepseek_v2.py:553`). Arc used
    /// to infer it from `scoring_func` instead, which agrees for every real
    /// V4/V3 checkpoint but silently ignored an explicit `false`.
    /// Audit finding 21.
    #[serde(default = "default_norm_topk_prob")]
    pub(crate) norm_topk_prob: bool,
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
    /// The expert-parallel world size actually in force.
    ///
    /// `ARC_EP_SIZE` wins over the config so a run can be sharded without
    /// editing a published `config.json`; an unparsable value is ignored
    /// rather than silently treated as 1, because "EP quietly turned itself
    /// off" is exactly the failure this whole path exists to avoid — it is
    /// reported at plan-construction time by the world-size check instead.
    pub fn effective_ep_size(&self) -> usize {
        match std::env::var("ARC_EP_SIZE") {
            Ok(v) => v.parse::<usize>().unwrap_or(self.ep_size).max(1),
            Err(_) => self.ep_size.max(1),
        }
    }

    /// Returns the compress ratio for layer `layer_idx`, or 0 (standard) if
    /// the index is out of bounds. Audit §1 layer-to-pattern mapping.
    pub fn layer_compress_ratio(&self, layer_idx: usize) -> i32 {
        self.compress_ratios.get(layer_idx).copied().unwrap_or(0)
    }

    /// The RoPE scaling that applies to a layer with compress ratio
    /// `compress_ratio`.
    ///
    /// YaRN is **per layer** in V4, keyed on the compress ratio. Reference
    /// `srt/models/deepseek_v4.py:234-238`:
    ///
    /// ```python
    /// if self.compress_ratio:
    ///     original_seq_len = rope_scaling["original_max_position_embeddings"]
    /// else:
    ///     original_seq_len = 0
    /// ```
    ///
    /// and `precompute_freqs_cis` (`srt/layers/deepseek_v4_rope.py:47-53`)
    /// skips the whole interpolation branch when `original_seq_len == 0`:
    /// ratio-0 (Standard) layers therefore get plain `1/θ^(2i/d)` with
    /// `θ = rope_theta`, no ramp and no `factor` division. The MTP block is
    /// Standard too (`COMPRESS_RATIO_NEXTN_LAYER = 0`,
    /// `deepseek_v4_nextn.py:47`).
    ///
    /// For `deepseek-ai/DeepSeek-V4-Flash` the ratio-0 set is exactly
    /// `{0, 1, 43}` — layers 0 and 1, plus slot 43, the MTP block. (Layer 42
    /// has ratio 4; it is *not* Standard.) With the published `factor: 16`,
    /// applying YaRN there compresses low-frequency rotation ~16×, which is
    /// what `dsv4_attention.rs` blames for the RUN-161 long-context
    /// repetition collapse. Audit finding 6.
    ///
    /// `ARC_YARN_ON_STANDARD_LAYERS=1` restores Arc's pre-fix behaviour (YaRN
    /// on every layer) for A/B work; it is not a supported serving mode.
    pub(crate) fn rope_scaling_for_compress_ratio(
        &self,
        compress_ratio: i32,
    ) -> Option<DeepSeekV2RopeScaling> {
        if compress_ratio == 0
            && !mistralrs_quant::env_flag_is_set("ARC_YARN_ON_STANDARD_LAYERS")
        {
            None
        } else {
            self.rope_scaling.clone()
        }
    }

    /// [`Self::rope_scaling_for_compress_ratio`] for a specific layer index.
    /// `layer_idx == num_hidden_layers` addresses the MTP slot.
    pub(crate) fn rope_scaling_for_layer(&self, layer_idx: usize) -> Option<DeepSeekV2RopeScaling> {
        self.rope_scaling_for_compress_ratio(self.layer_compress_ratio(layer_idx))
    }

    /// RoPE table config for Standard (compress-ratio-0) layers: base
    /// `rope_theta`, and — per [`Self::rope_scaling_for_compress_ratio`] — no
    /// YaRN.
    pub(crate) fn standard_rope_config(&self) -> DeepSeekV2RopeConfig {
        DeepSeekV2RopeConfig {
            rope_scaling: self.rope_scaling_for_compress_ratio(0),
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
            qk_rope_head_dim: self.qk_rope_head_dim,
        }
    }

    /// RoPE table config for compressed layers (ratio 4 = CSA, 128 = HCA):
    /// base `compress_rope_theta`, YaRN applied. Reference
    /// `deepseek_v4.py:220`: `rope_base = config.compress_rope_theta if
    /// self.compress_ratio else rope_theta`.
    pub(crate) fn compress_rope_config(&self) -> DeepSeekV2RopeConfig {
        DeepSeekV2RopeConfig {
            rope_scaling: self.rope_scaling_for_compress_ratio(4),
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.compress_rope_theta,
            qk_rope_head_dim: self.qk_rope_head_dim,
        }
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

/// Guard for the PagedAttention arm of [`Attention::forward`].
///
/// `PagedAttention::cache_write_and_gather`
/// (`paged_attention/layers/paged_attention.rs:391`) returns the gathered K/V
/// as `[1, H, N_total, D]`: the CUDA `gather_kv_cache` produces
/// `[N_total, H, D]` where `N_total = sum_i(seqlen_i)` over every scheduled
/// sequence, and the wrapper lifts it with a bare `unsqueeze(0)` (`:508-509`).
/// That is a **varlen pack**, not a batch.
///
/// [`super::dsv4_attention::dsv4_attention`]'s shape contract is
/// `k, v: [B, 1, T_k, D]` with `T_k` the cached length of ONE sequence: it
/// derives `q0 = t_k - t_q` and both the sliding-window mask and the
/// compressed-block causality mask from `t_k` alone
/// (`dsv4_attention.rs:270-294`). Handing it the pack at `bs > 1` makes every
/// query attend over the concatenation of all sequences' keys, positioned as
/// if they were one stream — wrong output, no error.
///
/// The per-sequence slice + dispatch loop that would fix it is audit §8 P0
/// item 9 and is not implemented (`paged_attention.rs:383-388` records the
/// deferral). Until it exists, refuse the batch loudly rather than serve
/// corrupted tokens. This is only reachable if
/// `DeepSeekV4Loader::supports_paged_attention` is flipped to `true`; see the
/// rationale there.
fn v4_paged_dispatch_precheck(bs: usize) -> Result<()> {
    if bs > 1 {
        candle_core::bail!(
            "V4 PagedAttention dispatch does not support batch_size > 1 (got {bs}): \
             cache_write_and_gather returns a varlen pack [1, H, sum(seqlen), D], \
             but dsv4_attention needs one sequence's [B, 1, T_k, D]. Enabling \
             paged attention for V4 requires the per-sequence slice + dispatch \
             loop (audit §8 P0 item 9) first."
        );
    }
    Ok(())
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
    /// RoPE — either `rope_theta`-based (standard layers 0/1/43) or
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
        let sinks_for_sdpa = if mistralrs_quant::env_flag_is_set("ARC_DISABLE_SINK") {
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
                // Audit §1(e): this was inverted. The value describes the key
                // axis `dsv4_attention` hands the backend, which is the raw
                // window alone on Standard and `[raw ++ compressed]` on
                // CSA/HCA — see `dsv4_attention::sdpa_sliding_window`.
                sliding_window: super::dsv4_attention::sdpa_sliding_window(
                    compress_ratio,
                    cfg.sliding_window,
                ),
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

    /// Decide whether the fused `cuda::qk_norm_rope` kernel may replace the
    /// eager transpose + Q RMS-norm + RoPE chain for this call.
    ///
    /// Returns `Some(pos_offset)` when every precondition for BIT-IDENTITY
    /// holds, `None` otherwise. Each rejection is a real difference in what the
    /// kernel would have to compute, not a convenience:
    ///
    /// * **BF16 only.** In f32 the RoPE expression is FMA-contractible, and
    ///   candle-kernels builds with the default `-fmad=true` while this
    ///   kernel's IEEE builder uses `--fmad=false`; the two would legitimately
    ///   disagree in the last bit. bf16 operators are inline PTX and immune.
    /// * **`head_dim` must equal candle's `fast_sum` block width** (i.e. be a
    ///   power of two ≤ 1024), or the reduction tree — and therefore the
    ///   rounding of the RMS statistic — has a different shape.
    /// * **One shared position offset.** With per-sequence offsets candle takes
    ///   its `Tensor::cat` path, which the kernel does not model.
    /// * **A single KV head**, which is what makes K one extra row rather than
    ///   a second layout.
    ///
    /// A `None` here is logged once (see `note_declined`) so a permanently
    /// disengaged fast path cannot masquerade as a working one.
    fn fused_qk_gate(
        &self,
        q_proj_out: &Tensor,
        kv_normed: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Option<usize> {
        use crate::cuda::qk_norm_rope as qk;
        if !cfg!(feature = "cuda") || !qk::fused_enabled() {
            return None;
        }
        macro_rules! decline {
            ($why:expr) => {{
                qk::note_declined($why);
                return None;
            }};
        }
        if !q_proj_out.device().is_cuda() {
            decline!("not a CUDA device");
        }
        if q_proj_out.dtype() != candle_core::DType::BF16
            || kv_normed.dtype() != candle_core::DType::BF16
        {
            decline!("Q/KV are not BF16");
        }
        let cos = self.rotary_emb.cos();
        let sin = self.rotary_emb.sin();
        if cos.dtype() != candle_core::DType::BF16 || sin.dtype() != candle_core::DType::BF16 {
            decline!("RoPE tables are not BF16");
        }
        // The kernel needs ONE `pos_offset`, not a batch of one row. It reads
        // the table at `pos_offset + t` where `t = blockIdx.y % seq_len`
        // (`qk_norm_rope.cu`), a position that does not depend on `b`, and its
        // grid is `(n_heads + 1, batch * seq_len)` with buffers sized
        // `batch * ...` — so it is already correct at any batch size whose rows
        // share an offset. This used to read `seqlen_offsets.len() != 1`, which
        // tested the length of the vector rather than the distinctness of its
        // values and therefore declined at EVERY batch size above one, leaving
        // the fused path dead in exactly the serving regime it was written for.
        // Rows that genuinely differ still decline, and still take the eager
        // chain. Prove the swap with `ARC_QK_VERIFY=1`, which bit-compares
        // against the eager output at every layer.
        let Some(uniform_offset) = crate::layers::uniform_seqlen_offset(seqlen_offsets) else {
            decline!("per-sequence position offsets");
        };
        if self.num_kv_heads != 1 {
            decline!("more than one KV head");
        }
        let head_dim = self.cfg.head_dim;
        let rope_dim = self.cfg.qk_rope_head_dim;
        // candle's FastReduce block width is `min(1024, n).next_power_of_two()`;
        // the kernel's tree only matches when that equals the row length.
        if head_dim > 1024 || !head_dim.is_power_of_two() {
            decline!("head_dim is not candle's fast_sum block width");
        }
        if rope_dim == 0 || rope_dim % 2 != 0 || rope_dim > head_dim {
            decline!("unusable rope_dim");
        }
        if !q_proj_out.is_contiguous() || !kv_normed.is_contiguous() {
            decline!("Q/KV projection output is not contiguous");
        }
        if !cos.is_contiguous() || !sin.is_contiguous() {
            decline!("RoPE tables are not contiguous");
        }
        // Shapes: q_proj_out is [B, T, n_heads * head_dim], kv_normed is
        // [B, T, head_dim].
        let (qb, qt, qh) = match q_proj_out.dims3() {
            Ok(d) => d,
            Err(_) => decline!("q_proj output is not 3-D"),
        };
        if qh != self.num_attention_heads * head_dim {
            decline!("q_proj output width is not n_heads * head_dim");
        }
        // The batched cohort is OPT-IN. Fixing the length-vs-values gate above
        // makes this kernel live on a path it has never run on, and this repo
        // has paid for exactly that twice this week (TCFRAG, and the fused-512
        // attention path that dropped the mask for four days). `batch == 1` is
        // untouched — already live, already exercised. See
        // `qk::fused_cohort_enabled` for the doctrine and for how to turn the
        // cohort path on once it has been measured on hardware.
        if qb > 1 && !qk::fused_cohort_enabled() {
            decline!("batched cohort is opt-in (set ARC_QK_FUSED_COHORT=1)");
        }
        match kv_normed.dims3() {
            Ok((kb, kt, kh)) if kb == qb && kt == qt && kh == head_dim => {}
            _ => decline!("kv_norm output shape does not match [B, T, head_dim]"),
        }
        let pos_offset = uniform_offset;
        match cos.dims2() {
            Ok((rows, cols)) if cols == rope_dim / 2 && rows >= pos_offset + qt => {}
            _ => decline!("RoPE table does not cover [pos_offset, pos_offset + T)"),
        }
        Some(pos_offset)
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
        let Some(rows) = self.compress_prefix(xs_hist)? else {
            return Ok(None);
        };
        Ok(Some(self.compressed_kv_from_rows(&rows)?))
    }

    /// Compress the largest `ratio`-multiple prefix of a *raw* history in one
    /// shot — the whole-history recompute. Still the prefill path (and the
    /// path for callers with no per-sequence cache), and the reference the
    /// rolling state is tested against.
    fn compress_prefix(&self, xs_hist: &Tensor) -> Result<Option<Tensor>> {
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
        Ok(Some(compressor.forward_from_xs(&xs_trunc)?))
    }

    /// `[B, T_c, head_dim]` pre-RoPE compressed rows → `[B, 1, T_c, head_dim]`
    /// with compress-θ RoPE applied at the strided compressed positions.
    /// Row `j` sits at absolute position `j * ratio` and its rotation depends
    /// on nothing but `j`, which is why the rows can be cached pre-RoPE.
    fn compressed_kv_from_rows(&self, rows: &Tensor) -> Result<Tensor> {
        let ratio = self.compress_ratio.ratio();
        let comp = rows.unsqueeze(1)?;
        let t_c = comp.dim(2)?;
        // Compressed entry j sits at absolute position j*ratio. Apply the
        // layer's (compress-θ) RoPE to the last qk_rope_head_dim dims there.
        //
        // This used to build the positions with `Tensor::arange` on every call.
        // That is a host round trip on the hot path AND — decisively — the
        // reason CUDA-graph capture of the V4 decode forward SIGSEGV'd on its
        // first launch: `arange` uploads a transient host `Vec`, and a captured
        // `cuMemcpyHtoDAsync` records the host POINTER, not the bytes, so the
        // graph re-read a freed `Vec` and handed garbage indices straight to
        // the `index_select` on the next line. `compress_positions` serves a
        // zero-copy view of a table built once, outside capture; see its doc
        // comment for the full mechanism.
        let dev = comp.device();
        let positions = crate::layers::compress_positions(t_c, ratio, dev)?;
        self.rotary_emb
            .forward_at_positions(&comp, self.cfg.qk_rope_head_dim, &positions)
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
        //
        // RUN-170 (wave30): the slot no longer holds the raw `[B, T, hidden]`
        // history. `forward_from_xs` is a strided local reduction — row `j` is
        // final once token `(j+1)*ratio - 1` has arrived and can never change
        // — so what the slot keeps is the *result*: the completed compressed
        // rows `[B, T/ratio, head_dim]` plus the bounded raw tail a future row
        // still needs (`span_groups * ratio + margin` tokens). That is
        // `8 * ratio` times smaller per token (32x on CSA, 1024x on HCA) and
        // removes the O(T) recompute per decode step. `XsRollingCache` owns
        // the token↔row mapping so the engine's token-unit truncations
        // (prefix cacher, MTP verify rollback, speculative rejection) stay
        // correct, and refuses — loudly — the ones it cannot resume.
        let _prof_comp = arc_profiler::device_span("compressor_advance");
        let compressed_rows = match xs_hist_cache {
            Some(hist) => {
                if seqlen_offsets.iter().all(|&o| o == 0) {
                    hist.reset();
                }
                let KvCache::XsRolling(state) = hist else {
                    candle_core::bail!(
                        "V4 compressor layer was handed a {} cache slot; the compressor-input \
                         history must be a KvCache::XsRolling entry (see DeepSeekV4::new).",
                        match hist {
                            KvCache::Normal { .. } => "Normal",
                            KvCache::Rotating { .. } => "Rotating",
                            KvCache::TurboQuant(_) => "TurboQuant",
                            KvCache::XsRolling(_) => unreachable!(),
                        }
                    );
                };
                match self.compressor.as_ref() {
                    Some(compressor) => {
                        debug_assert_eq!(
                            state.span_groups, compressor.coff,
                            "xs rolling cache span must equal the compressor's group span"
                        );
                        let xs3 = xs.contiguous()?;
                        // RUN-161 2c: under graph decode the compressed axis
                        // must be as wide at token 1 as it will ever be, so pin
                        // it before the first row lands. Done here, on every
                        // warmup step, because `pin_comp_capacity` refuses once
                        // rows are live — a regrow would move the address a
                        // captured graph baked in.
                        let graph_rows = (crate::layers::has_graph_mode_positions()
                            && seq_len == 1
                            && self.paged_attn.is_none())
                        .then(|| graph_comp_rows(self.compress_ratio.ratio()));
                        if let Some(rows) = graph_rows {
                            state.pin_comp_capacity(rows)?;
                            // The compressed axis is not the only width that
                            // moves with position: the retained RAW tail is
                            // rebuilt per step at a width that cycles with
                            // `tokens % ratio`, and a capture-time allocation
                            // miss on one of those widths is an unstable graph
                            // memory node — an illegal address on the first
                            // launch, not a slow graph. Pin it too.
                            state.pin_tail_width();
                        }
                        // Always advanced, including under the window-only
                        // ablation below: skipping it would drop the raw tail
                        // and leave a hole the next group cannot be built from.
                        //
                        // 🔴 ARCHITECTURAL LIMIT OF CAPTURE (RUN-161, measured).
                        // This call cannot be served by a replayed graph, for
                        // two independent reasons — neither of which more
                        // warmup, a bigger alloc cache, or pinning a width can
                        // reach:
                        //
                        //  1. `cuGraphLaunch` executes ONLY the recorded
                        //     kernels. No host code runs. `advance` carries the
                        //     compressor history in host-owned Rust state
                        //     (`XsRollingCache::tail`, reassigned to a FRESH
                        //     `Tensor` every step), so under replay the history
                        //     simply stops advancing and the distant-context
                        //     branch freezes at the capture step's content. The
                        //     raw KV half does not have this problem because it
                        //     writes through `write_kv_inplace` into one
                        //     fixed-address buffer at a device-derived slot —
                        //     a recorded kernel mutating stable memory.
                        //  2. The compressor fires only on the 1-in-`ratio`
                        //     steps that complete a block, so consecutive
                        //     decode steps do not even execute the same set of
                        //     kernels. With `compress_ratios` {4, 128} that is
                        //     every 4th step. A graph is a fixed DAG and cannot
                        //     express that branch.
                        //
                        // The fix is (1)'s pattern applied here: `tail` has to
                        // become a fixed-capacity DEVICE ring advanced by a
                        // recorded kernel, exactly like `SingleCache::
                        // append_graph`, and (2) then needs either a
                        // conditional graph node or one graph per phase.
                        // Until then the compressed branch is correct under
                        // capture only where it contributes nothing —
                        // `ARC_V4_WINDOW_ONLY=1`, or a context shorter than one
                        // `ratio` block.
                        let advanced =
                            state.advance(&xs3, |window| compressor.forward_from_xs(window))?;
                        match graph_rows {
                            // Fixed-width read: the same buffer, narrowed at a
                            // constant offset to a constant width, so the
                            // attention geometry does not move with the
                            // sequence. `graph_compressed_mask` masks the
                            // columns no block has reached yet.
                            Some(rows) if advanced.is_some() => {
                                state.compressed_rows_fixed(rows)?
                            }
                            _ => advanced,
                        }
                    }
                    None => None,
                }
            }
            // No per-sequence slot (direct `model.forward` callers — tests,
            // the SDK — and the MTP block, which is a ratio-0 layer): compress
            // the tokens in hand, exactly as before.
            None => self.compress_prefix(xs)?,
        };
        drop(_prof_comp);
        // 1. Q projection (LoRA). [B, T, hidden] → [B, T, n_heads*head_dim]
        //    → reshape to [B, n_heads, T, head_dim]. Audit §3.
        let q_proj_out = {
            let _s = arc_profiler::device_span("q_proj");
            timed_mla(0, &mdev, || self.q.forward(xs))?
        };
        v4_nan_dbg(&q_proj_out, "attn.q_proj");

        // 2. K/V projection: single fused wkv. [B, T, hidden] → [B, T,
        //    head_dim] → kv_norm → reshape to [B, num_kv_heads=1, T,
        //    head_dim]. Audit §0 + §3.
        //
        // wave44: this block used to sit AFTER the Q RMS-norm. It is hoisted
        // above it so the fused Q/K kernel below can take the Q rows and the K
        // row in ONE launch. The two projections read only `xs` and neither
        // reads the other's output, so no arithmetic moves — only the order in
        // which two independent GEMMs are enqueued.
        let kv_raw = {
            let _s = arc_profiler::device_span("kv_proj");
            timed_mla(1, &mdev, || self.wkv.forward_autocast(xs))?
        };
        v4_nan_dbg(&kv_raw, "attn.wkv");
        let kv_normed = {
            let _s = arc_profiler::device_span("kv_norm");
            self.kv_norm.forward(&kv_raw)?
        };
        v4_nan_dbg(&kv_normed, "attn.kv_norm");

        // 3. Head transpose + per-head Q RMS-norm + RoPE + NoPE/PE recombine.
        //
        // The eager spelling below is SIXTEEN candle launches per layer, TEN of
        // them pure data movement (`ucopy_bf16` / `copy2d_bf16`) materialising
        // intermediates that only exist because the chain is expressed as
        // separate ops: a transpose, a `broadcast_mul`, two `.contiguous()`
        // narrows inside `rope_i`, and two two-input `cat`s. Their entire data
        // footprint at decode is 64 KB of Q and 1 KB of K.
        //
        // `cuda::qk_norm_rope` does all of it in one launch of `n_heads + 1`
        // blocks, bit-identically — the transpose becomes an output address,
        // the split/`cat` becomes two write ranges of the same row, and the RMS
        // statistic stays in a register. Toggle with `ARC_QK_FUSED=0`; prove
        // with `ARC_QK_VERIFY=1`.
        let eager_qk = |q_proj_out: &Tensor, kv_normed: &Tensor| -> Result<(Tensor, Tensor)> {
            let q = q_proj_out
                .reshape((bs, seq_len, self.num_attention_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            // V4: per-head RMS-normalize Q over head_dim before RoPE. Reference
            // inference/model.py:498 `q *= rsqrt(q.square().mean(-1)+eps)`. This is
            // SEPARATE from q_norm (which normalizes q_lora_rank inside self.q).
            // Missing it leaves Q ~30x too small -> near-uniform attention scores
            // -> the model cannot attend -> word-salad output (RUN-161).
            let q = {
                let _s = arc_profiler::device_span("q_rmsnorm");
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

            let k = kv_normed
                .reshape((bs, seq_len, self.num_kv_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            // RoPE applied in-place to the last qk_rope_head_dim dims of
            // each Q-head and K-head's head_dim-vector. Audit §0 + §3.
            let _s = arc_profiler::device_span("rope");
            self.apply_rope_inplace(&q, &k, seqlen_offsets)
        };

        let (q, k) = {
            let _s = arc_profiler::device_span("qk_norm_rope");
            match self.fused_qk_gate(&q_proj_out, &kv_normed, seqlen_offsets) {
                Some(pos_offset) => {
                    let (q_f, k_f) = crate::cuda::qk_norm_rope::qk_norm_rope_cuda(
                        &q_proj_out,
                        &kv_normed,
                        self.rotary_emb.cos(),
                        self.rotary_emb.sin(),
                        self.num_attention_heads,
                        bs,
                        seq_len,
                        head_dim,
                        self.cfg.qk_rope_head_dim,
                        self.cfg.rms_norm_eps as f64,
                        pos_offset,
                    )?;
                    crate::cuda::qk_norm_rope::note_engaged(
                        head_dim,
                        self.cfg.qk_rope_head_dim,
                        self.num_attention_heads,
                        "bf16",
                    );
                    if crate::cuda::qk_norm_rope::verify_enabled() {
                        let (q_e, k_e) = eager_qk(&q_proj_out, &kv_normed)?;
                        // The eager NORMALISED Q (pre-RoPE), built on demand so
                        // the rotation candle applied can be solved for exactly.
                        let qn_for_diag = || -> Result<Tensor> {
                            let qn = q_proj_out
                                .reshape((bs, seq_len, self.num_attention_heads, head_dim))?
                                .transpose(1, 2)?
                                .contiguous()?;
                            let inv_rms = qn
                                .sqr()?
                                .mean_keepdim(candle_core::D::Minus1)?
                                .affine(1.0, self.cfg.rms_norm_eps)?
                                .recip()?
                                .sqrt()?;
                            qn.broadcast_mul(&inv_rms)
                        };
                        // Run the row diagnostic on the FAILING call, not on
                        // whichever call happened to be first: position 0 is
                        // the identity rotation, so the first call agrees
                        // trivially and describes nothing.
                        if let Err(e) = crate::cuda::qk_norm_rope::verify_pair("q", &q_f, &q_e) {
                            crate::cuda::qk_norm_rope::diagnose_row(
                                &qn_for_diag()?,
                                &q_e,
                                &q_f,
                                self.rotary_emb.cos(),
                                self.rotary_emb.sin(),
                                head_dim,
                                self.cfg.qk_rope_head_dim,
                                seqlen_offsets[0],
                                seq_len,
                            )?;
                            return Err(e);
                        }
                        crate::cuda::qk_norm_rope::verify_pair("k", &k_f, &k_e)?;
                    }
                    (q_f, k_f)
                }
                None => eager_qk(&q_proj_out, &kv_normed)?,
            }
        };
        v4_nan_dbg(&q, "attn.q_rope");
        v4_nan_dbg(&k, "attn.k_rope");
        v4_trace_dump(self.dbg_layer_idx, &q, "20_q_rope");
        v4_trace_dump(self.dbg_layer_idx, &k, "21_k_rope");

        // V4 QAT: FP8-simulate the non-rope dims of K (reference model.py:506
        // `act_quant(kv[..., :-rd], 64, ..., inplace=True)`). The model was
        // trained with the KV non-rope dims round-tripped through block-wise
        // FP8; feeding full BF16 is out-of-distribution. V=K so v inherits it.
        //
        // wave43: the quantized form is kept, not just its BF16 shadow. The
        // non-paged cache stores the E4M3 codes + each block's `amax` directly
        // (`dsv4_kv_fp8`), which is BIT-EXACT with the BF16 tensor built on the
        // next line — the round trip has already happened by then — at 590
        // B/token/layer instead of 1026.
        let k_packed = {
            let _s = arc_profiler::device_span("kv_fp8_quant");
            super::dsv4_kv_fp8::quantize_k(
                &k,
                self.cfg.qk_rope_head_dim,
                super::dsv4_kv_fp8::KvQuantMode::from_env(),
            )?
        };
        let k = match &k_packed {
            Some(packed) => {
                let _s = arc_profiler::device_span("kv_fp8_dequant");
                packed.dequant(k.dtype())?
            }
            None => k,
        };
        v4_stat_dbg(&k, "attn.k_actquant");
        v4_trace_dump(self.dbg_layer_idx, &k, "30_k_actquant");

        // 4. V4: K and V come from the same wkv tensor. The kernel treats
        //    the wkv output as both K (for scores) and V (for weighted
        //    sum). This is the MLA "absorb V into K" trick at scale.
        //    Audit §3 ("absorbed the MLA split into a single fused output").
        //
        // wave33: because `v` IS `k` — bit for bit, unconditionally — the V
        // half of the KV slot used to store a second full `[B, 1, T, 512]`
        // copy of a tensor the model already has. That doubled the V4 KV
        // footprint for nothing: 2 * 512 * 2 B = 2048 B/token/layer where the
        // reference stores one K (SGLang
        // `mem_cache/deepseek_v4_memory_pool.py:93-111`, 584 B/token/layer).
        // The V half is now a 1-wide zero marker (`append_v_marker`), exactly
        // the device this same file already used for the `xs` history slot
        // (see the R2/R3 comment above: "`v` is a `[B, T, 1]` zero marker kept
        // in lockstep because the cache managers require both sides
        // populated"). Nothing downstream reads it: every consumer of the
        // cached V is `dsv4_attention`, which is handed the cached K.
        //
        // A materialised `v` survives only on the PagedAttention arm, which
        // writes through `reshape_and_cache` into engine-owned block storage
        // that this model does not control. It is built there, not here, so
        // the live (non-paged — `DeepSeekV4Loader::supports_paged_attention`
        // returns false) decode path no longer pays the copy either.

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
        // window-trained layers 0/1/43 unseen relative distances and
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
            raw_prefix: 0,
            row_q0: None,
            graph_positions: None,
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
        let compressed_kv = {
            let _s = arc_profiler::device_span("compressed_kv_build");
            match compressed_rows {
                Some(rows) if !mistralrs_quant::env_flag_is_set("ARC_V4_WINDOW_ONLY") => {
                    Some(self.compressed_kv_from_rows(&rows)?)
                }
                _ => None,
            }
        };
        let mut attn_out = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                // PagedAttention (all compress ratios): write to paged
                // cache, gather the full context, then run the V4 dispatch
                // over the gathered K/V. RUN-167. Standard layers also go
                // through `dsv4_attention` — they need the sliding-window
                // mask, which the dense paged kernel cannot apply.
                Some(((mut key_cache, mut value_cache, _, _), input_metadata)) => {
                    v4_paged_dispatch_precheck(bs)?;
                    // Use copy() not clone(): clone() shares the storage
                    // Arc<RwLock>, so k and v alias the same storage.
                    // PagedAttention's reshape_and_cache is an in-place op that
                    // write-locks and read-locks that storage -> RwLock
                    // self-deadlock (the dummy-run hang). copy() gives v its own
                    // storage. Cheap here: MQA means 1 KV head. (RUN-161)
                    let v = k.copy()?;
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
                        &k,
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
                // Bisect slots, in execution order, per graph-arm layer:
                //   k       — the new post-RoPE key, BEFORE it is written.
                //             Diverges ⇒ the key itself is computed wrong
                //             (RoPE / projections), upstream of the cache.
                //   k_full  — the fixed window read back AFTER the write.
                //             Clean `k` + dirty `k_full` ⇒ the write landed in
                //             the wrong slot, or the wrong window is read.
                //   attn    — clean `k`/`k_full` + dirty `attn` ⇒ the mask or
                //             the compressed branch, not the KV ring.
                crate::layers::arc_layer_trace_push(&k);
                let k_full = append_graph_kv_mqa(kv_cache, &k, &position, cap)?;
                crate::layers::arc_layer_trace_push(&k_full);
                // Use ONLY the fixed-width graph mask (matches the C-wide K).
                // The eager `attention_mask` is kv_len-wide (growing) and would
                // both mismatch the fixed window and break shape-constancy.
                // `dsv4_attention` folds whatever it is handed into its own
                // union mask.
                //
                // 🔴 REFUSED, not defaulted. Of the `cap` slots read back, only
                // `position + 1` were ever written; the rest are the zeros the
                // buffer was allocated with. A zero K row is NOT a masked row —
                // it scores logit 0 and takes `exp(0)/Z` of the softmax weight,
                // diluting every real key and contributing its zero V. This arm
                // previously ran with `graph_mode_mask() == None` and called the
                // result "finite, not yet correct"; a finite wrong answer is the
                // failure mode nothing downstream catches, so it now errors.
                // `DeepSeekV4::forward` publishes the mask once per step.
                let gmask = crate::layers::graph_mode_mask().ok_or_else(|| {
                    candle_core::Error::Msg(
                        "V4 graph-mode decode: the fixed-capacity length mask is unset. \
                         Reading a constant window of which only `position + 1` slots are \
                         written and attending the rest as zero-padding is a wrong answer, \
                         not a degraded one. `DeepSeekV4::forward` must call \
                         `layers::set_graph_mode_mask` whenever graph-mode positions are set."
                            .into(),
                    )
                })?;
                let graph_attn = super::dsv4_attention::dsv4_attention(
                    &q,
                    &k_full,
                    &k_full,
                    compressed_kv.as_ref(),
                    Some(&gmask),
                    flash_params,
                    &self.sdpa_params,
                    super::dsv4_attention::Dsv4AttentionConfig {
                        // The compressed branch's block-causality threshold is
                        // an ABSOLUTE position, and under fixed capacity the
                        // `q0` inferred from the buffer width is a constant —
                        // it would freeze the distant-context branch at
                        // `floor(sliding_window / ratio)` blocks forever (one,
                        // on HCA). Hand it the device position instead, so a
                        // replay re-derives the threshold from whatever the
                        // position buffer holds.
                        graph_positions: Some(&position),
                        ..dsv4_cfg
                    },
                )?;
                crate::layers::arc_layer_trace_push(&graph_attn);
                graph_attn
            }
            None => {
                // FP8 code storage is opt-in (`ARC_V4_FP8_KV=1`); unset stores
                // the dense BF16 K the model has always stored.
                let cached = {
                    let _s = arc_profiler::device_span("kv_cache_append");
                    append_kv_mqa(
                        kv_cache,
                        &k,
                        k_packed.as_ref().filter(|_| v4_fp8_kv_enabled()),
                        v4_kv_retain_rows(self.sliding_window),
                    )?
                };
                // Two independent prefixes meet here, and they compose:
                //
                //  * `store_base` — what the CACHE no longer holds, because
                //    `retain_window` evicted it. `0` unless
                //    `ARC_V4_KV_WINDOW` is on, and then the whole point: it is
                //    the absolute position of the retained run's first row.
                //  * the read-side narrowing — what this call chooses not to
                //    RECONSTRUCT, because no query row in this block can reach
                //    it (`raw_keep_span`). Packed storage does it here so the
                //    FP8 dequant stays O(window); dense storage leaves it to
                //    `dsv4_attention`, which narrows a view for free.
                //
                // `raw_prefix` is the absolute position of `k_cached[.., 0, ..]`
                // either way, which is exactly what `dsv4_attention` defines it
                // to be. With the window off and dense storage this is `0` and
                // the whole cache, i.e. character-for-character the old call.
                let store_base = kv_cache.first_cached();
                let t_k_full = store_base + cached.seq_len()?;
                let (raw_base, keep) = super::dsv4_attention::raw_keep_span(
                    seq_len,
                    self.sliding_window.max(1),
                    t_k_full,
                );
                let (span_base, span_len, raw_prefix) = match &cached {
                    V4CachedK::Dense(_) => (0, cached.seq_len()?, store_base),
                    V4CachedK::Packed(_) => {
                        let rel = raw_base.checked_sub(store_base).ok_or_else(|| {
                            candle_core::Error::Msg(format!(
                                "V4 KV window: rows in this block reach back to absolute key \
                                 {raw_base}, but the store was capped at {store_base}. The \
                                 retained window is too small for a {seq_len}-row block at \
                                 sliding_window {}.",
                                self.sliding_window
                            ))
                        })?;
                        (rel, keep, raw_base)
                    }
                };
                let k_cached = {
                    let _s = arc_profiler::device_span("kv_cache_span");
                    cached.span(span_base, span_len, k.dtype())?
                };
                // Cache read-back. Diff vs prefill's freshly-computed K splits a
                // cache-storage bug (old rows differ) from a new-token position
                // bug (only the last row differs).
                //
                // NOTE: under packed storage this is the RETAINED SPAN
                // (`[raw_prefix, raw_prefix + keep)`), not `[0, N)` — a decode
                // step dumps the trailing `window` rows, ending with the new
                // token's K. `v4_trace_diff.py` comparisons against pre-wave43
                // decode traces must be re-based by `raw_prefix`; prefill is
                // unchanged (`raw_prefix == 0`, the whole cache).
                v4_trace_dump(self.dbg_layer_idx, &k_cached, "40_k_cached");
                // Same tensor as `40_k_cached` by construction (V4 MQA: V == K).
                // Kept under the historic tag so `v4_trace_diff.py` still lines
                // up against pre-wave33 traces.
                v4_trace_dump(self.dbg_layer_idx, &k_cached, "41_v_cached");
                let _s = arc_profiler::device_span("sdpa");
                super::dsv4_attention::dsv4_attention(
                    &q,
                    &k_cached,
                    &k_cached,
                    compressed_kv.as_ref(),
                    attention_mask,
                    flash_params,
                    &self.sdpa_params,
                    super::dsv4_attention::Dsv4AttentionConfig {
                        raw_prefix,
                        // The dense path is the only one whose K/V rows are a
                        // left-aligned ragged cohort (`front_pad_kv_cache`).
                        // PagedAttention keeps per-request block tables, and
                        // the graph-decode arm reads a FIXED window whose
                        // columns are device slots, not absolute positions —
                        // neither can be masked from `seqlen_offsets`, so both
                        // keep `row_q0: None` and behave exactly as before.
                        row_q0: ragged_row_q0(seqlen_offsets),
                        ..dsv4_cfg
                    },
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
        // NOTE for anyone reading the old four-bucket profile: the inverse tail
        // is OUTSIDE `timed_mla(2)` even though that timer is named
        // `invrope_oproj`, so `sdpa = mla_attn - the three MLA timers`
        // over-attributed this work to SDPA. Here it is its own node.
        let attn_out_bhtd = {
            let _s = arc_profiler::device_span("inv_rope");
            self.rotary_emb.forward_inverse_tail(
                &attn_out_bhtd,
                self.cfg.qk_rope_head_dim,
                seqlen_offsets,
            )?
        };
        v4_trace_dump(self.dbg_layer_idx, &attn_out_bhtd, "60_after_invtail");
        // -> [B, T, H*head_dim]
        attn_out = attn_out_bhtd.transpose(1, 2)?.reshape((bs, seq_len, ()))?;

        let o_groups = self.cfg.o_groups.unwrap_or(1);
        let _prof_oproj = arc_profiler::device_span("o_proj");
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
        drop(_prof_oproj);
        v4_trace_dump(self.dbg_layer_idx, &out, "70_o_out");
        Ok(out)
    }
}

/// V4 MoE gate. Inherits V3's NoAuxTc / Greedy / GroupLimitedGreedy dispatch
/// and adds support for `ScoringFunc::SqrtSoftplus` (V4 default — audit §0
/// + §5).
struct MoeGate {
    /// Router weight exactly as it appears in the checkpoint:
    /// `[n_routed_experts, hidden_size]`, checkpoint dtype.
    ///
    /// Kept **only** so `residual_tensors` can write it back into UQFF under
    /// its original name/shape/dtype. It is deliberately not used by
    /// `forward` — see [`Self::weight_t_f32`].
    weight: Tensor,
    /// The same router weight, pre-transposed and pre-promoted to F32 at load
    /// time — i.e. exactly the operand `forward`'s GEMM consumes, materialised
    /// once instead of once per layer per decode step.
    ///
    /// Stored as `weight.t()?.to_dtype(F32)?`, shape
    /// `[hidden_size, n_routed_experts]`. `forward` used to build this inline
    /// on every call; because `.t()` is non-contiguous that was a *strided*
    /// cast — reading the BF16 weight and writing a fresh F32 buffer, per
    /// layer, per token. The weight is a constant, so all of it was redundant.
    /// vLLM likewise keeps the router weight in fp32 from load
    /// (`router/gate_linear.py:66-68`).
    weight_t_f32: Tensor,
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
        // Hoist the router GEMM's operand out of the decode loop. This is the
        // exact same expression `forward` used to evaluate per call
        // (`weight.t()?.to_dtype(F32)?`) — same input, same deterministic
        // conversion, same resulting layout — just evaluated once at load.
        // Deliberately NOT followed by `.contiguous()`: adding one would give
        // the GEMM a different operand layout than it sees today (relevant
        // when the checkpoint is already F32 and `to_dtype` short-circuits to
        // the transposed view), which could change cuBLAS kernel selection and
        // therefore accumulation order.
        //
        // Cost: the F32 copy is resident for the life of the model rather than
        // being reallocated every step — +4.19 MB per MoE layer, ~180 MB over
        // 43 layers (0.13% of an H200). `weight` itself is retained unchanged
        // for UQFF serialization, which is why both live here.
        let weight_t_f32 = weight.t()?.to_dtype(DType::F32)?;
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
        if mistralrs_quant::env_flag_is_set("ARC_COLLAPSE") {
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
            weight_t_f32,
            cfg: cfg.clone(),
            top_k,
            n_routed_experts,
            e_score_correction_bias,
            tid2eid,
        })
    }

    /// One-kernel spelling of the `gate.renormalize` chain.
    /// `Ok(None)` means the caller must run `eager`.
    fn fused_renorm(
        &self,
        topk_weight: &Tensor,
        do_renorm: bool,
        eager: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<Option<Tensor>> {
        use crate::cuda::hc_fused as seam;

        if !seam::seam_on("ARC_SEAM_GATE") {
            return Ok(None);
        }
        let decline = |reason: &str| -> Result<Option<Tensor>> {
            seam::seam_declined(seam::SEAM_GATE, reason);
            Ok(None)
        };
        if !topk_weight.device().is_cuda() {
            return decline("device is not cuda");
        }
        if topk_weight.dtype() != DType::F32 {
            return decline("topk_weight is not F32");
        }
        let Ok((n, k)) = topk_weight.dims2() else {
            return decline("topk_weight is not rank 2");
        };
        if !matches!(k, 1 | 2 | 4 | 6 | 8) {
            return decline("top_k outside the specialised set");
        }
        if !topk_weight.is_contiguous() {
            return decline("topk_weight is not contiguous");
        }
        if n == 0 {
            return decline("empty gate renorm");
        }
        // The scale is narrowed here exactly as candle's `affine` narrows it:
        // `T::from_f64(routed_scaling_factor)`.
        let out = seam::gate_renorm_cuda(
            topk_weight,
            1e-20f64 as f32,
            self.cfg.routed_scaling_factor as f32,
            do_renorm,
        )?;

        if seam::seam_ab_enabled() {
            // Already F32 on both sides — nothing is narrowed here, so this
            // comparison is the whole contract.
            seam::ab_check("seam.gate_renorm", &out, &eager(topk_weight)?)?;
        }

        seam::seam_engaged(seam::SEAM_GATE);
        Ok(Some(out))
    }

    fn forward(&self, xs: &Tensor, input_ids: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (bs, seq_len, h) = xs.dims3()?;
        let xs = xs.reshape(((), h))?;
        // `weight_t_f32` is already transposed and already F32 (built once in
        // `new`), so the GEMM's weight operand costs nothing here. Only the
        // activation cast below remains, and that one is real work — `xs` is a
        // fresh per-token tensor, not a constant.
        let logits = {
            let _s = arc_profiler::device_span("gate.router_gemm");
            xs.to_dtype(DType::F32)?
                .broadcast_matmul(&self.weight_t_f32)?
        };
        let _prof_score = arc_profiler::device_span("gate.scoring");
        let scores = match self.cfg.scoring_func {
            ScoringFunc::Softmax => candle_nn::ops::softmax_last_dim(&logits)?,
            ScoringFunc::Sigmoid => candle_nn::ops::sigmoid(&logits)?,
            // V4: sqrt(softplus(x)). Stable formulation:
            // softplus(x) = max(x, 0) + log(1 + exp(-|x|)).
            // Audit §8 P1 item 14.
            //
            // The eager form below is NINE kernel launches (`zeros_like`,
            // `bmaximum`, `uabs`, `uneg`, `uexp`, `affine`, `ulog`, `badd`,
            // `usqrt`) on a `[1, n_routed_experts]` = [1, 256] tensor, once per
            // MoE layer per token. `cuda/hc_fused.cu` collapses it to one,
            // bit-identically — this expression decides WHICH EXPERTS RUN, so
            // the fused kernel transcribes candle's ops rather than
            // re-deriving them. `ARC_HC_FUSED=0` restores the chain for A/B.
            ScoringFunc::SqrtSoftplus => {
                if crate::cuda::hc_fused::usable(&logits) {
                    crate::cuda::hc_fused::sqrt_softplus_cuda(&logits)?
                } else {
                    let max0 = logits.maximum(&logits.zeros_like()?)?;
                    let abs = logits.abs()?;
                    let softplus = (max0 + ((abs.neg()?.exp()? + 1.0)?.log()?))?;
                    softplus.sqrt()?
                }
            }
        };
        drop(_prof_score);

        let _prof_topk = arc_profiler::device_span("gate.topk");
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
                // The published V4 Flash `config.json` contains neither
                // `n_group` nor `topk_group` — verified against
                // <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json>
                // and pinned by `v4_flash_config_settles_the_audit_questions`.
                // The serde defaults (1/1) therefore apply and this branch is
                // dead for the real checkpoint, which is the same end state
                // SGLang reaches by force-disabling group limiting for V4
                // (`deepseek_v2.py:572-577`). It stays reachable only for
                // V3-style configs that do publish the keys.
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

        drop(_prof_topk);

        let _prof_renorm = arc_profiler::device_span("gate.renormalize");
        // Renormalize the selected weights to sum to 1. The reference gates
        // this purely on the config flag (`renormalize=config.norm_topk_prob`,
        // `deepseek_v2.py:553` → `topk.py:876-882`); softmax scores are already
        // normalized over the full expert set, so V2/V3-style softmax gates
        // keep the historical Arc behavior of skipping it. V4 Flash publishes
        // `"norm_topk_prob": true` with `sqrtsoftplus`, so this is taken —
        // identical to the previous scoring-function-derived condition.
        // Audit finding 21.
        let do_renorm = self.cfg.norm_topk_prob
            && matches!(
                self.cfg.scoring_func,
                ScoringFunc::Sigmoid | ScoringFunc::SqrtSoftplus
            );
        // FOUR launches over a [1, 6] tensor — `fast_sum`, `affine(+1e-20)`,
        // `bdiv`, `affine(*routed_scaling_factor)` — once per MoE layer per
        // token, i.e. 172 per decode forward for 6 divides and 6 multiplies.
        // Collapsed to one; `arc_seam_gate_renorm_kernel` in cuda/hc_fused.cu
        // replays candle's identity-padded reduction tree and the fmaf that
        // `affine` compiles to. `ARC_F32SEAM=0` / `ARC_SEAM_GATE=0` restore the
        // chain below.
        let eager_renorm = |w: &Tensor| -> Result<Tensor> {
            let w = if do_renorm {
                let denominator = (w.sum_keepdim(D::Minus1)? + 1e-20)?;
                w.broadcast_div(&denominator)?
            } else {
                w.clone()
            };
            w * self.cfg.routed_scaling_factor
        };
        topk_weight = match self.fused_renorm(&topk_weight, do_renorm, &eager_renorm)? {
            Some(w) => w,
            None => eager_renorm(&topk_weight)?,
        };
        drop(_prof_renorm);

        // RUN-161 diagnostic: are the selected top-k routing weights peaked or
        // near-uniform? Near-uniform weights average 6 redundant experts and
        // amplify their shared common-mode (collapse). Log the spread of the
        // raw gate logits at the selected experts, and the final weight spread.
        if mistralrs_quant::env_flag_is_set("ARC_COLLAPSE") {
            if let Ok(topk_logits) = logits.gather(&topk_idx, 1) {
                v4_stat_dbg(&topk_logits, "gate.topk_logits");
            }
            v4_stat_dbg(&topk_weight, "gate.topk_weight");
        }

        // RUN-161 fix-test: replace the near-uniform normalized-sqrtsoftplus
        // weights with a softmax over the selected experts' RAW logits (peaked
        // routing). If this restores coherence, the routing-weight uniformity is
        // the collapse root cause.
        if mistralrs_quant::env_flag_is_set("ARC_SOFTMAX_ROUTE") {
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
        if mistralrs_quant::env_flag_is_set("ARC_ROUTE_TOP1") {
            let max = topk_weight.max_keepdim(D::Minus1)?;
            let diff = topk_weight.broadcast_sub(&max)?;
            let mask = diff.ge(&diff.zeros_like()?)?.to_dtype(topk_weight.dtype())?;
            topk_weight = (topk_weight * mask)?;
        }
        Ok((topk_idx, topk_weight))
    }
}

/// How the routed experts are distributed over the expert-parallel ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EpPlacementMode {
    /// Rank `r` owns `[r·E/N, (r+1)·E/N)`. The only placement that slices the
    /// checkpoint at load time rather than after it.
    Contiguous,
    /// Bin-pack by measured per-expert load. For V4's first `num_hash_layers`
    /// layers the load is a **closed-form property of the tokenizer**: those
    /// layers route by `gate.tid2eid`, a fixed token-id → expert table, so
    /// their distribution can be computed exactly, offline, with no GPU.
    /// Score-routed layers have no such table and fall back to contiguous.
    BalancedFromHashTable,
}

impl EpPlacementMode {
    /// `ARC_EP_PLACEMENT=balanced` opts into the tid2eid-derived placement.
    fn from_env() -> Self {
        match std::env::var("ARC_EP_PLACEMENT").as_deref() {
            Ok("balanced") => Self::BalancedFromHashTable,
            _ => Self::Contiguous,
        }
    }
}

/// Per-expert token counts implied by a V4 hash-routing table.
///
/// `tid2eid` is `[vocab_size, top_k]` (I64): token id → the `top_k` experts it
/// is routed to, unconditionally, in the first `num_hash_layers` layers
/// (`deepseek4.rs` `MoeGate::tid2eid`, reference `inference/model.py`
/// `Gate.hash = layer_id < n_hash_layers`). With a uniform token distribution
/// the per-expert load is just the number of table entries naming that expert.
///
/// This is the offline distribution wave44-BV §3.2 flagged as "free to check
/// and nobody has".
fn tid2eid_expert_loads(tid2eid: &Tensor, num_experts: usize) -> Result<Vec<f64>> {
    let rows = tid2eid.to_dtype(DType::I64)?.to_vec2::<i64>()?;
    let mut loads = vec![0.0f64; num_experts];
    for row in &rows {
        for &e in row {
            if e < 0 {
                candle_core::bail!("gate.tid2eid holds a negative expert id ({e})");
            }
            let e = e as usize;
            if e >= num_experts {
                candle_core::bail!(
                    "gate.tid2eid names expert {e}, but this layer has {num_experts} experts"
                );
            }
            loads[e] += 1.0;
        }
    }
    Ok(loads)
}

/// Build this rank's expert-parallel view of one MoE layer.
///
/// Refuses rather than degrades: an `ep_size` that does not match the
/// communicator's world size, or an expert count that does not divide, is an
/// error. wave44-BV §1.6: "a device list on the wrong kind of run is an error,
/// not a silent no-op".
fn build_expert_parallel_plan(
    cfg: &DeepSeekV4Config,
    n_routed_experts: usize,
    comm: &Arc<mistralrs_quant::Comm>,
    device: &Device,
    tid2eid: Option<&Tensor>,
) -> Result<ExpertParallelPlan> {
    let ep_size = cfg.effective_ep_size();
    if ep_size <= 1 {
        return Ok(ExpertParallelPlan::single(n_routed_experts));
    }
    if comm.world_size() != ep_size {
        candle_core::bail!(
            "expert parallelism: ep_size is {ep_size} but the communicator's world size is {} — \
             stage 1 requires one rank per expert shard (attention stays data-parallel, so there \
             is no second axis to split). Launch with {ep_size} ranks or set ARC_EP_SIZE=1.",
            comm.world_size()
        );
    }
    let placement = match (EpPlacementMode::from_env(), tid2eid) {
        (EpPlacementMode::BalancedFromHashTable, Some(table)) => {
            let loads = tid2eid_expert_loads(table, n_routed_experts)?;
            ExpertPlacement::balanced(&loads, ep_size)?
        }
        _ => ExpertPlacement::contiguous(n_routed_experts, ep_size)?,
    };
    ExpertParallelPlan::new(Arc::new(placement), comm.rank(), device)
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

        // The gate is built first because the expert→rank placement can be
        // derived from its hash-routing table (V4's first `num_hash_layers`
        // layers), and the experts must be loaded already sharded.
        let gate = MoeGate::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("gate"), false),
            n_routed_experts,
            layer_idx,
        )?;
        let ep = build_expert_parallel_plan(
            cfg,
            n_routed_experts,
            comm,
            &layer_device,
            gate.tid2eid.as_ref(),
        )?;

        let experts = MoEExperts::new_expert_parallel(
            &moe_cfg,
            mapper.set_device(layer_idx, vb.clone(), loading_isq),
            layer_device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
            &ep,
        )?;

        let shared_experts = if let Some(n_shared_experts) = n_shared_experts {
            let intermediate_size = cfg.moe_intermediate_size * n_shared_experts;
            Some(
                Mlp::new(
                    mapper.set_device(layer_idx, vb.pp("shared_experts"), loading_isq),
                    cfg.hidden_size,
                    intermediate_size,
                    &cfg.quantization_config,
                    cfg.hidden_act,
                    comm,
                )?
                // V4: the shared expert carries the SAME trained SwiGLU clamp
                // as the routed experts — the reference passes
                // `swiglu_limit=getattr(config, "swiglu_limit", None)` into
                // `DeepseekV2MLP` (`deepseek_v2.py:619`) and V4 Flash publishes
                // `swiglu_limit: 10.0`. Audit finding 5: this was dropped on
                // every device and backend.
                .with_swiglu_limit(cfg.swiglu_limit),
            )
        } else {
            None
        };
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
        let (topk_idx, topk_weight) = {
            let _s = arc_profiler::device_span("moe.gate");
            self.gate.forward(xs, input_ids)?
        };
        v4_stat_dbg(&topk_weight, "moe.topk_w");
        v4_stat_dbg(&topk_idx, "moe.topk_idx");
        // RUN-161 diagnostic: dump the actual per-token selected expert IDs at a
        // few layers (hash L1, score-onset L7, deep L14) to see whether the 5
        // prompt tokens route to the SAME experts (routing collapse) or stay
        // diverse. If score-routed layers over-concentrate vs the hash layers,
        // the selection bias / scoring is wrong.
        if mistralrs_quant::env_flag_is_set("ARC_COLLAPSE")
            && matches!(self.layer_idx, 1 | 4 | 7 | 14 | 20)
        {
            if let Ok(v) = topk_idx.to_dtype(DType::U32).and_then(|t| t.to_vec2::<u32>()) {
                eprintln!("ARC_TOPKID [L{}] {:?}", self.layer_idx, v);
            }
        }
        let mut y = {
            let _s = arc_profiler::device_span("moe.experts");
            self.experts.forward(xs, topk_weight, &topk_idx)?
        };
        y = y.reshape((b_size, seq_len, hidden_dim))?;
        v4_stat_dbg(&y, "moe.routed");
        let li = self.layer_idx;
        v4_collapse_dbg(&y, &format!("L{li}.moe_routed"), 1);
        // `to_dtype` in ARGUMENT position is evaluated eagerly, so this debug
        // telemetry was casting the U32 expert ids to F32 on every MoE layer of
        // every forward — 43 `cast_u32_f32` launches per decoded token — even
        // with `ARC_COLLAPSE` unset. `v4_collapse_dbg` already casts to F32
        // internally, *behind* the env guard, so passing the ids through
        // unchanged is the same telemetry for zero launches.
        v4_collapse_dbg(&topk_idx, &format!("L{li}.moe_topk_idx"), 0);

        if let Some(ref shared_experts) = self.shared_experts {
            let _s = arc_profiler::device_span("moe.shared_expert");
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

/// Width, in elements per token, of the V half of a V4 KV cache slot in the
/// **unpacked** layout (FP8 KV off, or a geometry whose non-rope dims are not
/// whole 64-wide blocks — synthetic fixtures). The packed layout puts the rope
/// tail and the block scales there instead; see [`super::dsv4_kv_fp8`].
///
/// V4 is MQA with a single fused `wkv` projection: `V` **is** `K`, bit for bit
/// (see step 4 of [`Attention::forward`]). Storing a second full `head_dim`
/// copy therefore buys nothing, and it is the whole of V4's KV overshoot
/// against the reference — SGLang's DSV4 pool keeps one K per token
/// (`mem_cache/deepseek_v4_memory_pool.py:93-111`).
///
/// The V half cannot simply be dropped: [`crate::kv_cache::NormalCacheManager`]
/// unwraps `v.all_data` unconditionally when it batches sequences in
/// (`clone_in_cache`) and splits them back out (`clone_out_cache`), so both
/// halves must stay populated and must stay length-synchronised for every
/// truncation path (prefix cacher, MTP verify rollback, speculative rejection)
/// to keep working. One element per token satisfies all of that at
/// `head_dim`× less memory. This is the same device the `xs` compressor-history
/// slot has always used (see the R2/R3 note in [`Attention::forward`]).
pub(crate) const V4_V_MARKER_WIDTH: usize = 1;

/// Build the `[B, n_kv_heads, T, 1]` zero marker stored in place of the
/// duplicate V. See [`V4_V_MARKER_WIDTH`].
fn v4_v_marker(k: &Tensor) -> Result<Tensor> {
    let (b, h, t, _d) = k.dims4()?;
    Tensor::zeros((b, h, t, V4_V_MARKER_WIDTH), k.dtype(), k.device())
}

/// Reject a KV slot that is not a plain [`KvCache::Normal`].
///
/// The marker layout relies on the two halves of a `Normal` slot being
/// independent [`crate::kv_cache::SingleCache`]s that may differ in their last
/// dimension. `Rotating` would silently window the marker against the K half's
/// own offset bookkeeping, and `TurboQuant` would try to quantize a 1-wide
/// vector. Neither is reachable for V4 today (`NormalCache::new` hands out
/// `Normal` slots unless the global TurboQuant head-dim is one of 64/128/256,
/// and V4's is 512) — this exists so a future cache change fails loudly
/// instead of corrupting the V4 decode path.
fn require_normal_kv_slot(kv_cache: &KvCache) -> Result<()> {
    if matches!(kv_cache, KvCache::Normal { .. }) {
        return Ok(());
    }
    candle_core::bail!(
        "V4 attention was handed a {} KV slot; the MQA V-marker layout requires a \
         KvCache::Normal entry (see V4_V_MARKER_WIDTH).",
        match kv_cache {
            KvCache::Normal { .. } => unreachable!(),
            KvCache::Rotating { .. } => "Rotating",
            KvCache::TurboQuant(_) => "TurboQuant",
            KvCache::XsRolling(_) => "XsRolling",
        }
    )
}

/// One layer's cached V4 keys, in whichever layout the slot holds.
///
/// V4's `V` **is** its `K`, bit for bit (step 4 of [`Attention::forward`]), so
/// there is only ever one thing here.
#[derive(Debug)]
pub(crate) enum V4CachedK {
    /// Activation-dtype `[B, H, T, head_dim]`, V half a 1-wide marker.
    Dense(Tensor),
    /// FP8 codes in the K half, rope tail + block `amax` in the V half.
    Packed(Box<V4PackedK>),
}

impl V4CachedK {
    /// Number of cached tokens.
    fn seq_len(&self) -> Result<usize> {
        match self {
            Self::Dense(k) => k.dim(2),
            Self::Packed(p) => p.seq_len(),
        }
    }

    /// Materialise keys `[base, base + len)` at activation precision.
    ///
    /// For the packed layout this is where the dequant happens, and it is why
    /// the caller narrows first: reconstructing the whole context every decode
    /// step would write `T * head_dim` activations per layer, which at large
    /// batch costs more bandwidth than the storage saves. Reconstructing only
    /// [`super::dsv4_attention::raw_keep_span`] makes it `O(window)`.
    fn span(&self, base: usize, len: usize, out_dtype: DType) -> Result<Tensor> {
        match self {
            Self::Dense(k) => {
                if base == 0 && len == k.dim(2)? {
                    Ok(k.clone())
                } else {
                    k.narrow(2, base, len)?.contiguous()
                }
            }
            Self::Packed(p) => p.narrow(base, len)?.dequant(out_dtype),
        }
    }
}

/// Process-wide state of the V4 windowed KV store. `0` = not yet resolved
/// (consult `ARC_V4_KV_WINDOW`), `1` = off, `2` = on.
///
/// An atomic rather than a `OnceLock` so [`set_v4_kv_window`] can drive both
/// settings from one process — the identity guard in
/// `tests/synthetic_load_smoke.rs` has to run the SAME model both ways, and a
/// `OnceLock` would silently measure whichever the test binary latched first.
/// Same shape, and the same reason, as `set_mtp_load_depth`.
static V4_KV_WINDOW: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Turn V4's windowed KV store on or off for every subsequently-appended slot.
///
/// **Opt-in, default off** (`ARC_V4_KV_WINDOW=1`). With it off, V4's KV cache
/// grows for the whole context exactly as it always has and not one byte of
/// any other model's cache moves.
///
/// It is opt-in for the reason written on [`v4_fp8_kv_enabled`] one screen up:
/// the last KV-storage change to ship default-on without a GPU behind it
/// (`ARC_V4_FP8_KV`, wave43-BU) killed every request on the first V4 forward
/// that met a real device. The saving here is arithmetic and the identity is
/// pinned on CPU; the *decode-time* cost of the compaction copy is not, and
/// until an A/B has run, the default is the layout that has actually served.
pub fn set_v4_kv_window(on: bool) {
    V4_KV_WINDOW.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::SeqCst);
}

/// Whether V4 should cap its raw KV store at the attention window.
pub fn v4_kv_window_enabled() -> bool {
    match V4_KV_WINDOW.load(std::sync::atomic::Ordering::SeqCst) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var("ARC_V4_KV_WINDOW").is_ok_and(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            });
            set_v4_kv_window(on);
            on
        }
    }
}

/// How many trailing raw rows a V4 KV slot must retain, given its attention
/// window — or `None` when the store is uncapped.
///
/// `window + margin`, and the margin is not slack: after the compaction that
/// precedes an append, the slot holds exactly `retain` rows, and the forward
/// that follows reaches back `t_q + window - 1`
/// ([`super::dsv4_attention::raw_keep_span`]) over the `retain + t_q` rows
/// that then exist. That is satisfied for any `retain >= window - 1`; the
/// margin is what additionally survives an MTP verify rollback, which lowers
/// `current_seq_len` by up to `--mtp-depth` without giving any rows back.
pub(crate) fn v4_kv_retain_rows(sliding_window: usize) -> Option<usize> {
    v4_kv_window_enabled().then(|| sliding_window.max(1) + crate::kv_cache::RETAIN_WINDOW_MARGIN)
}

/// Append V4's fused K/V. `packed` is the quantized form of `k`, already
/// filtered by the caller: `Some` stores the FP8 codes, `None` stores `k` dense
/// beside a 1-wide marker. Returns the full cached sequence in whichever layout
/// was written.
///
/// The [`v4_fp8_kv_enabled`] gate is applied by the caller, not here, so this
/// function's two layouts are both reachable from a test without mutating the
/// process environment — the `OnceLock` behind that gate resolves once per
/// process, so an env-driven test of the packed path would silently measure the
/// dense one under a different default.
fn append_kv_mqa(
    kv_cache: &mut KvCache,
    k: &Tensor,
    packed: Option<&V4PackedK>,
    retain: Option<usize>,
) -> Result<V4CachedK> {
    require_normal_kv_slot(kv_cache)?;
    // Re-armed per append rather than at cache construction: `NormalCache` is
    // built by `NormalCache::new_plain(len, max_seq_len)` from inside
    // `DeepSeekV4::new`, which has no way to hand a per-layer window to a
    // constructor shared with thirty other models. A `usize` store per layer
    // per step is free next to the append it precedes.
    kv_cache.set_retain_window(retain);
    match packed {
        Some(p) => {
            let (codes, side) = kv_cache.append(&p.codes, &p.side)?;
            Ok(V4CachedK::Packed(Box::new(V4PackedK {
                codes,
                side,
                rope_dim: p.rope_dim,
            })))
        }
        None => {
            let marker = v4_v_marker(k)?;
            let (k_cached, _marker_cached) = kv_cache.append(k, &marker)?;
            Ok(V4CachedK::Dense(k_cached))
        }
    }
}

// The compressed-row positions used to live here as a SINGLE-slot thread-local
// keyed on `(t_c, ratio, device)`. That slot is now `layers::compress_positions`
// (a per-`(ratio, device)` table, chunk-rounded and served as a `narrow` view),
// because a single slot THRASHES on V4: the model interleaves CSA (ratio 4) and
// HCA (ratio 128) layers, so consecutive layers of the same step evicted each
// other and the `arange_step` was paid again on every layer — including inside
// the capture region, where it is a SIGSEGV and not merely a cost.
//
// The arithmetic argument that licensed `arange_step` in the first place is kept
// here, because it is what makes the two spellings interchangeable: `j * ratio`
// is integer arithmetic that used to be laundered through float — `arange(u32)
// -> cast_u32_f32 -> affine(*ratio) -> cast_f32_u32`, three kernel launches to
// compute a strided range `arange_step` produces directly. The values are
// identical: the old chain's `fmaf(j, ratio, 0.0)` is exact for every `j * ratio`
// a sequence can reach (both operands and the product are integers well inside
// f32's exact range), and `to_dtype(U32)` truncated it back. `ratio` is 1 / 4 /
// 128 (`CompressRatio::ratio`), never 0, so `arange_step`'s zero-step bail is
// unreachable. Multiplied in usize so the product cannot wrap before the u32
// narrowing.

/// The FIXED number of compressed rows the CUDA-graph decode arm reads back.
///
/// Fixed capacity is what capture costs. A graph replays the launch geometry it
/// recorded, so the compressed key axis has to be as wide on token 1 as it will
/// ever be — which means choosing, up front, the longest context the graph is
/// allowed to serve. `ARC_V4_GRAPH_MAX_CTX` is that choice (tokens, default
/// 8192); the row count is it divided by the layer's compress ratio, so an HCA
/// layer (ratio 128) pays 64 rows where a CSA layer (ratio 4) pays 2048.
///
/// Deliberately NOT `max_position_embeddings / ratio`: at V4's 163 840-token
/// ceiling that is 40 960 compressed columns per layer per step, which would
/// cost far more than the launch overhead capture is meant to remove. Beyond
/// the pinned context the arm refuses (`compressed_rows_fixed` bails) rather
/// than silently re-capturing or silently truncating the distant context —
/// falling back to eager decode is a slower answer, truncation is a wrong one.
fn graph_comp_rows(ratio: usize) -> usize {
    const DEFAULT_GRAPH_CTX_TOKENS: usize = 8192;
    let ctx = std::env::var("ARC_V4_GRAPH_MAX_CTX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_GRAPH_CTX_TOKENS);
    ctx.div_ceil(ratio.max(1)).max(1)
}

/// [`append_kv_mqa`] for the CUDA-graph decode path: writes at the device-held
/// `position` and reads back a fixed `cap`-wide window.
///
/// Dense only. `mistralrs_quant::kvwrite::write_kv_inplace` is instantiated for
/// F16/BF16/F32, so a U8 code cache cannot be written at a device-held slot;
/// [`v4_fp8_kv_enabled`] is therefore false whenever this arm can run, and the
/// check below turns a future divergence into an explicit error rather than a
/// dtype mismatch deep inside the kernel dispatch.
fn append_graph_kv_mqa(
    kv_cache: &mut KvCache,
    k: &Tensor,
    position: &Tensor,
    cap: usize,
) -> Result<Tensor> {
    require_normal_kv_slot(kv_cache)?;
    if v4_fp8_kv_enabled() {
        candle_core::bail!(
            "V4 CUDA-graph decode needs the dense KV layout, but FP8 KV storage is on. \
             Unset ARC_V4_FP8_KV, or teach mistralrs_quant::kvwrite::write_kv_inplace \
             the U8 code cache."
        );
    }
    // The eager arm arms the retention policy on every append; the graph arm
    // must disarm it, because `write_kv_inplace` addresses an ABSOLUTE device
    // slot and a compacted buffer's column 0 is not absolute 0. Clearing the
    // policy is enough for a slot that has not yet evicted; one that HAS is a
    // real incompatibility and `SingleCache::append_graph` says so by name.
    kv_cache.set_retain_window(None);
    let marker = v4_v_marker(k)?;
    let (k_full, _marker_full) = kv_cache.append_graph(k, &marker, position, cap)?;
    Ok(k_full)
}

/// FP8 storage for the cached K. **Opt-in: off unless `ARC_V4_FP8_KV=1`.**
///
/// wave43-BU shipped this defaulted ON without ever running it on a GPU — its
/// own notes said *"needs a GPU: first CUDA exercise of a U8 `SingleCache`"* —
/// and the first V4 forward on a commit that contained it died, every request,
/// with `dtype mismatch in slice-set, lhs: BF16, rhs: U8` (wave48-BY). The
/// layout is sound and bit-exact (see [`super::dsv4_kv_fp8`]); what was never
/// exercised is the CUDA leg of it. Until a GPU A/B has shown
/// `ARC_V4_FP8_KV=1` and unset producing token-identical greedy output, the
/// default is the layout that has actually served (wave49-BZ).
///
/// Also forced off under `ARC_V4_CAPTURE_PROBE`: the CUDA-graph decode arm
/// writes through `mistralrs_quant::kvwrite::write_kv_inplace`, which is
/// instantiated for F16/BF16/F32 only — a U8 code cache needs a U8 variant of
/// that kernel.
///
/// Resolved once: `deepseek4` used to scan the environment per timer call and
/// paid ~390 `getenv`s per forward for it (wave33).
/// The per-row absolute query positions [`super::dsv4_attention`] needs when
/// the dense batch is a **left-aligned ragged** cohort, or `None` when it is
/// not — which is every request today, because the flag defaults off.
///
/// `seqlen_offsets[i]` is already row `i`'s own past length: the engine threads
/// it to every attention call so RoPE places the row's queries at their true
/// absolute positions. That makes it exactly the `q0` V4's *compressed* branch
/// needs and the `lead` its dead prefix needs, so no new value is plumbed —
/// what changes is that `dsv4_attention` stops collapsing the vector to the
/// single position it inferred from the cache width.
///
/// Gated on `ARC_MTP_PER_SEQ_KV` rather than on a flag of its own:
/// [`crate::kv_cache::front_align_batch`] is the only producer of a ragged
/// dense batch and it runs exclusively under `KvAdvance::PerSequence`, which
/// that flag authorizes. With it off no batch is ever left-aligned, so the
/// offsets would be uniform and `dsv4_attention` would take its scalar path
/// anyway — the gate makes that a guarantee instead of a coincidence.
///
/// Resolved once, for the reason [`v4_fp8_kv_enabled`] documents: this is on
/// the per-layer path, and a `getenv` per layer per step is ~43 per forward.
fn ragged_row_q0(seqlen_offsets: &[usize]) -> Option<&[usize]> {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(crate::pipeline::mtp_pipeline::per_sequence_kv_requested);
    ragged_row_q0_from(enabled, seqlen_offsets)
}

/// The decision itself, separated from the process-wide [`OnceLock`] so it can
/// be tested against both settings rather than against whatever the test
/// runner's environment happened to be (`ragged_row_q0_is_opt_in`). Same shape
/// as [`fp8_kv_enabled_from`], for the same reason.
fn ragged_row_q0_from(enabled: bool, seqlen_offsets: &[usize]) -> Option<&[usize]> {
    enabled.then_some(seqlen_offsets)
}

fn v4_fp8_kv_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        fp8_kv_enabled_from(
            std::env::var("ARC_V4_FP8_KV").ok().as_deref(),
            mistralrs_quant::env_flag_is_set("ARC_V4_CAPTURE_PROBE"),
        )
    })
}

/// The opt-in decision itself, separated from the process-wide [`OnceLock`] so
/// it can be tested against every input rather than against whatever the test
/// runner's environment happened to be (`v4_fp8_kv_is_opt_in`). A `OnceLock`
/// resolves once per process: a test that sets the variable proves nothing
/// about the default, which is exactly the kind of test DOCTRINE D12 counts as
/// worse than none.
fn fp8_kv_enabled_from(var: Option<&str>, capture_probe: bool) -> bool {
    var == Some("1") && !capture_probe
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

/// Log one forward's per-component split and reset the accumulators.
///
/// Two things this used to get wrong, both of which made the resulting profile
/// hard to act on (wave33):
///
/// 1. **It only fired on the 4-D mHC path.** The emit block lived inside the
///    `use_4d_mhc` arm of [`DeepSeekV4::forward`], so a checkpoint without the
///    global mHC head ran with `ARC_TIME_DECODE=1` and printed nothing at all.
///    It is now called for both arms; on the 3-D arm it says so, because the
///    `timed()` wrappers themselves live in `DecoderLayer::forward_4d` and a
///    3-D forward genuinely has no component split to report. Silence and
///    "this path is not instrumented" are different answers.
/// 2. **It carried no batch geometry.** A B=64 profile and a B=1 profile
///    printed identical-looking lines, so a percentage split could not be
///    attributed to a batch size after the fact — and the split that motivated
///    a month of MoE work was taken at b=1 on a build that has since been
///    abandoned. The line now carries `b`, `t`, the token count, and
///    per-token-of-batch time, so B=1 and B=64 runs are self-describing.
fn emit_decode_profile(input_ids: &Tensor, instrumented: bool) {
    use std::sync::atomic::Ordering;
    if !decode_timing_enabled() {
        return;
    }
    let (b, t) = match input_ids.dims2() {
        Ok(bt) => bt,
        // `[B]`-shaped decode inputs and anything else: report what we can
        // rather than dropping the whole profile.
        Err(_) => (input_ids.elem_count(), 1),
    };
    if !instrumented {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                "ARC_TIME_DECODE is set but this model is running the 3-D residual \
                 fallback path, which has no per-component timers (they live in \
                 DecoderLayer::forward_4d). No split will be reported. This means \
                 the checkpoint has no global mHC head — see DeepSeekV4::new."
            );
        });
        return;
    }
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
    let tokens = (b * t).max(1);
    tracing::info!(
        "ARC_TIME_DECODE b={} t={} tokens={} forward_total={:.2}ms ({:.3}ms/token) | {} \
         || MLA[{}] (sdpa=mla_attn-these)",
        b,
        t,
        tokens,
        total as f64 / 1e6,
        total as f64 / 1e6 / tokens as f64,
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

/// Whether `ARC_TIME_DECODE=1` was set at process start.
///
/// Cached: the timers wrap 9 components per layer, so a `var_os` per call was
/// ~390 environment scans per forward *when the profiler is off*. It is also
/// what makes the gate honest — `var_os` on every call meant the profiler could
/// half-enable mid-run.
pub(crate) fn decode_timing_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| mistralrs_quant::env_flag_is_set("ARC_TIME_DECODE"))
}

#[inline]
pub(crate) fn timed_mla<T>(
    idx: usize,
    dev: &candle_core::Device,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !decode_timing_enabled() {
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
    if !decode_timing_enabled() {
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
    if !mistralrs_quant::env_flag_is_set("V4_NAN_DEBUG") {
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
    if !mistralrs_quant::env_flag_is_set("V4_STATS") {
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
    if !mistralrs_quant::env_flag_is_set("ARC_COLLAPSE") {
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
        // The six `timed(..)` wrappers below belong to `ARC_TIME_DECODE`, which
        // synchronises the device twice per call — 774 full syncs per token at
        // 43 layers. The `arc_profiler` spans wrapped around them use CUDA
        // events instead and never synchronise, so both may be enabled at once
        // but only `ARC_TIME_DECODE` perturbs the run.
        let (y_attn, post_attn, comb_attn) = {
            let _s = arc_profiler::device_span("mhc_attn_pre");
            timed(0, &tdev, || mhc.attn_pre(residual_attn))?
        };
        v4_nan_dbg(&y_attn, &format!("L{li}.attn_pre.y"));
        v4_collapse_dbg(&y_attn, &format!("L{li}.y_attn"), 1);
        v4_nan_dbg(&post_attn, &format!("L{li}.attn_pre.post"));
        v4_nan_dbg(&comb_attn, &format!("L{li}.attn_pre.comb"));
        let y_attn_normed = {
            let _s = arc_profiler::device_span("input_layernorm");
            self.input_layernorm.forward(&y_attn)?
        };
        v4_nan_dbg(&y_attn_normed, &format!("L{li}.input_layernorm"));
        let _prof_attn = arc_profiler::device_span("mla_attn");
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
        drop(_prof_attn);
        v4_nan_dbg(&attn_out, &format!("L{li}.attn_out"));
        v4_collapse_dbg(&attn_out, &format!("L{li}.attn_out"), 1);
        let xs_4d = {
            let _s = arc_profiler::device_span("mix_post_attn");
            timed(2, &tdev, || {
                mhc.mix_post_4d(&attn_out, residual_attn, &post_attn, &comb_attn)
            })?
        };
        v4_nan_dbg(&xs_4d, &format!("L{li}.mix_post_attn"));
        v4_collapse_dbg(&xs_4d, &format!("L{li}.mix_post_attn"), 1);

        // === FFN BLOCK ===
        let residual_ffn = &xs_4d;
        let (y_ffn, post_ffn, comb_ffn) = {
            let _s = arc_profiler::device_span("mhc_ffn_pre");
            timed(3, &tdev, || mhc.ffn_pre(residual_ffn))?
        };
        v4_nan_dbg(&y_ffn, &format!("L{li}.ffn_pre.y"));
        v4_collapse_dbg(&y_ffn, &format!("L{li}.y_ffn"), 1);
        let y_ffn_normed = {
            let _s = arc_profiler::device_span("post_attention_layernorm");
            self.post_attention_layernorm.forward(&y_ffn)?
        };
        v4_nan_dbg(&y_ffn_normed, &format!("L{li}.post_attention_layernorm"));
        let ffn_out = {
            let _s = arc_profiler::device_span("moe");
            timed(4, &tdev, || self.moe_or_mlp.forward(&y_ffn_normed, input_ids))?
        };
        v4_nan_dbg(&ffn_out, &format!("L{li}.ffn_out"));
        v4_collapse_dbg(&ffn_out, &format!("L{li}.ffn_out"), 1);
        let out = {
            let _s = arc_profiler::device_span("mix_post_ffn");
            timed(5, &tdev, || {
                mhc.mix_post_4d(&ffn_out, residual_ffn, &post_ffn, &comb_ffn)
            })?
        };
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
        // `deepseek_v4_nextn.py:47`, audit §2) → the **Standard** RoPE table:
        // base `rope_theta`, and NO YaRN (`deepseek_v4.py:234-238` forces
        // `original_seq_len = 0` for ratio-0 layers, which makes
        // `precompute_freqs_cis` skip the interpolation branch entirely).
        //
        // This is the one-liner PR #35 deliberately deferred to #30's owner:
        // #35 fixed the identical defect on the main model's Standard layers
        // but could not touch this file. Until it was applied the MTP block
        // drafted on a YaRN-compressed table while the target attended on an
        // unscaled one, so the draft's positions meant something different
        // from the verifier's — and any acceptance measured on it would have
        // been a measurement of that mismatch.
        let rope_cfg = cfg.standard_rope_config();
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

    /// Fresh **draft KV cache** for one sequence (dim 2 = the sequence axis,
    /// matching `NormalCache::new`).
    ///
    /// This cache is **persistent per sequence and prefilled over the accepted
    /// context**, mirroring the reference: SGLang's EAGLE/NextN worker gives
    /// the draft model its own KV pool (`eagle_worker.py:134-138` — *"Share the
    /// allocator with a target worker. Draft and target worker own their own
    /// KV cache pools."*) which is filled by really running the draft module
    /// over the context (`forward_draft_extend`, `:1094-1128`) and extended
    /// over each verify's accepted tokens (`forward_draft_extend_after_decode`,
    /// `:1134+`).
    ///
    /// The draft KV is **not** a copy or a view of the target's KV: the MTP
    /// block has its own `wkv`, so its keys/values are different numbers for
    /// the same tokens. Entry `i` of this cache is the MTP state of the pair
    /// `(h_i, tok_{i+1})` — the target hidden at position `i` combined with
    /// the *next* token — which is the alignment
    /// `apply_eagle_prefill_input_rotation` (`eagle_utils.py:26-46`) sets up:
    /// *"Each req's slice [t_0..t_{n-1}] -> [t_1..t_{n-1}, t_n] … Aligns
    /// draft's position-i hidden with target's label at i+1."*
    ///
    /// The pipeline owns the lifetime; see
    /// [`crate::pipeline::mtp_pipeline::MtpSpeculativePipeline`].
    pub fn new_draft_cache(&self) -> KvCache {
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
    /// layer for `T` consecutive tokens.
    ///
    /// * `fused`: `[B, T, hidden]` — `h_proj(hnorm(h)) + e_proj(enorm(e))`.
    /// * `start_pos`: absolute sequence position of `fused[:, 0]` (RoPE). The
    ///   draft KV cache must already hold exactly `start_pos` entries, so that
    ///   cache slot `k` is the state of absolute position `k` and the
    ///   attention's own `q0 = t_k - t_q` position arithmetic
    ///   (`dsv4_attention.rs`) lines up with RoPE.
    /// * `cache`: the persistent draft KV from [`Self::new_draft_cache`].
    /// * `input_ids`: `[B, T]` current token ids — required by the hash-routed
    ///   (`tid2eid`) MoE gate; ignored by score-routed gates.
    ///
    /// Returns the decoder output `[B, T, hidden]` (pre-`norm`; the caller
    /// applies [`Self::norm_out`] before lm_head and feeds THIS tensor
    /// forward as the next step's hidden state, matching the reference's
    /// `pre_hc_head` hand-off in `deepseek_v4_nextn.py:186-201`).
    ///
    /// No caller mask is passed: the MTP block is a Standard (ratio-0) layer,
    /// and `dsv4_attention` derives its own causal + sliding-window mask from
    /// absolute positions, so a `T > 1` extend is causal by construction.
    ///
    /// Uses the 3-D decoder path: when the checkpoint ships the per-layer
    /// mHC tensors, `DecoderLayer::forward` applies the learned
    /// `hc_attn_*`/`hc_ffn_*` blend via the 3-D bridge (RUN-169); otherwise
    /// it falls back to standard residuals.
    pub fn forward_tokens(
        &self,
        fused: &Tensor,
        start_pos: usize,
        cache: &mut KvCache,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        self.forward_tokens_rows(fused, &[start_pos], cache, input_ids)
    }

    /// [`Self::forward_tokens`] for a batch whose rows sit at **different**
    /// absolute positions.
    ///
    /// `start_pos` is one position per row, or a single shared one. The draft
    /// KV must already be a **left-aligned** cohort — every row's live run
    /// shifted to end at `max(start_pos)`, which is exactly what
    /// [`crate::kv_cache::front_pad_kv_cache`] produces — so the one shared
    /// append offset `SingleCache::append` writes at is simultaneously correct
    /// for all of them.
    ///
    /// 🔑 Nothing new is plumbed to make the ragged case correct. `start_pos`
    /// *is* `seqlen_offsets`, the vector `DecoderLayer::forward` already threads
    /// to RoPE and (via `ragged_row_q0`) to `dsv4_attention`'s per-row `row_q0`,
    /// which masks each row's dead prefix. What changed is that this function
    /// stops collapsing it to one element.
    ///
    /// A **single-element** `start_pos` is the pre-change path verbatim: RoPE
    /// takes its `seqlen_offsets.len() == 1` fast branch rather than the per-row
    /// loop, and `resolve_ragged_rows` filters a one-element vector that equals
    /// the batch's end column back to `None`. So the uniform group — every group
    /// with per-sequence KV advance off — differs from the old code by control
    /// flow, not by a tolerance.
    pub fn forward_tokens_rows(
        &self,
        fused: &Tensor,
        start_pos: &[usize],
        cache: &mut KvCache,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        draft_step_end_column(start_pos, fused.dim(0)?, cache.current_seq_len())?;
        let in_device = fused.device().clone();
        let xs = fused.to_device(&self.device)?;
        let ids = input_ids.to_device(&self.device)?;
        let out = self.layer.forward(
            &xs,
            None,
            start_pos,
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

    /// Single-token convenience wrapper over [`Self::forward_tokens`].
    pub fn forward_step(
        &self,
        fused: &Tensor,
        pos: usize,
        cache: &mut KvCache,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        self.forward_tokens(fused, pos, cache, input_ids)
    }

    /// [`Self::forward_step`] for a group whose rows sit at different absolute
    /// positions — the single-token wrapper over [`Self::forward_tokens_rows`].
    pub fn forward_step_rows(
        &self,
        fused: &Tensor,
        pos: &[usize],
        cache: &mut KvCache,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        self.forward_tokens_rows(fused, pos, cache, input_ids)
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

/// The shared end column a draft step's rows must present, or why this group is
/// not one an MTP-block forward can run.
///
/// A dense `KvCache` appends at ONE offset, so a group can only share a forward
/// when every row's live run ENDS at the same column — which is exactly what
/// `kv_cache::front_pad_kv_cache` arranges and what makes the batch maximum
/// `max(start_pos)` the cache's own `current_seq_len`. Each refusal below is a
/// layout the caller believes in and the cache does not, and guessing would be
/// a wrong answer nothing downstream catches:
///
/// * no positions at all — nothing to place the queries at;
/// * a vector that is neither `1` nor `b_sz` wide — the caller and the tensor
///   disagree about how many rows there are, and `seqlen_offsets` would be
///   indexed past its end or silently applied to the wrong row;
/// * a cache whose width is not the rows' shared end column — either the group
///   was never left-aligned, or a row is *ahead* of the buffer it is drafting
///   into, so its RoPE position and its draft-KV slot mean different things.
///
/// Extracted from [`MtpBlock::forward_tokens_rows`] so the three refusals can be
/// exercised without standing up a 43-layer model — the same reason
/// `dsv4_attention::resolve_ragged_rows` is a free function.
pub(crate) fn draft_step_end_column(
    start_pos: &[usize],
    b_sz: usize,
    cached: usize,
) -> Result<usize> {
    let Some(&end) = start_pos.iter().max() else {
        candle_core::bail!(
            "MTP draft step was given no absolute position at all; it needs one per row (or a \
             single shared one)."
        );
    };
    if start_pos.len() != 1 && start_pos.len() != b_sz {
        candle_core::bail!(
            "MTP draft step: {} absolute position(s) for a batch of {b_sz}. Pass one per row, \
             or a single shared one.",
            start_pos.len()
        );
    }
    if cached != end {
        candle_core::bail!(
            "MTP draft KV desync: cache holds {cached} entries but the step's rows end at \
             absolute position {end}. Draft-KV slot k must be absolute position k, and a ragged \
             group must be left-aligned to its widest row first (see MtpBlock::new_draft_cache \
             and kv_cache::front_pad_kv_cache)."
        );
    }
    Ok(end)
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
    /// Side-channel carrying this model's pre-`lm_head` hidden states out to
    /// the MTP draft path — the reference's `hidden_states_before_norm` /
    /// `pre_hc_head` capture (`deepseek_v4.py:1168-1175`, consumed at
    /// `deepseek_v4_nextn.py:149-154`). Disarmed (and therefore free) until
    /// [`NormalModel::mtp_decode_kit`] hands a kit out.
    pub hidden_capture: Arc<crate::pipeline::mtp_pipeline::MtpHiddenCapture>,
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
        // Audit finding 6: Standard (ratio-0) layers get NO YaRN — the
        // reference forces `original_seq_len = 0` for them, which skips the
        // interpolation entirely. Compressed layers (ratio 4 / 128) do get it,
        // on the `compress_rope_theta` base. See
        // [`DeepSeekV4Config::rope_scaling_for_compress_ratio`].
        let rope_cfg_standard = cfg.standard_rope_config();
        let rope_cfg_compress = cfg.compress_rope_config();
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
                            hidden_capture: Arc::new(Default::default()),
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
        let mut compress_ratios_in_order = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let ratio = cfg.layer_compress_ratio(i);
            if ratio != 0 {
                xs_hist_slots.push(Some(compress_ratios_in_order.len()));
                compress_ratios_in_order.push(ratio as usize);
            } else {
                xs_hist_slots.push(None);
            }
        }
        // `new_plain`, not `new`: V4's fused-MQA layout stores a 1-wide marker
        // in the V half of every slot (see `V4_V_MARKER_WIDTH`), which
        // `require_normal_kv_slot` enforces. An ambient TurboQuant setting must
        // not turn these into compressed slots.
        let cache = NormalCache::new_plain(cfg.num_hidden_layers, cfg.max_position_embeddings);
        {
            let mut guard = cache.lock().unwrap();
            for ratio in compress_ratios_in_order {
                // wave30: the rolling compressed state, not the raw history.
                // `span_groups` mirrors `V4Compressor::coff` — the ratio-4
                // compressor's `overlap_transform` folds the previous group in,
                // so one row consumes two groups of raw tokens; every other
                // ratio consumes one.
                let span_groups = if ratio == 4 { 2 } else { 1 };
                guard
                    .0
                    .push(KvCache::XsRolling(Box::new(XsRollingCache::new(
                        ratio,
                        span_groups,
                        cfg.head_dim,
                        cfg.max_position_embeddings,
                    ))));
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
        let _prof_model = arc_profiler::span("model");
        if let Ok((b, t)) = input_ids.dims2() {
            arc_profiler::set_geometry(b, t);
        }
        let xs_embed = {
            let _s = arc_profiler::device_span("embed");
            self.embed_tokens.forward(input_ids)?
        };

        // RUN-161 2c: publish the fixed-capacity decode window's length mask.
        //
        // 🔑 Built HERE, and nowhere earlier, for two reasons that both bite:
        //
        //  1. **Replay safety.** `normal.rs` sets the graph-mode *positions*
        //     before `self.model.forward` — i.e. outside the capture region. A
        //     mask computed there would be a constant folded into the recorded
        //     graph, correct for the captured token and wrong for every replayed
        //     one. Computed here it is a device-side compare against the
        //     position buffer, so a replay that mutates that buffer in place
        //     re-derives the right mask with no recapture.
        //  2. **`cap` is a model fact.** `ModelConfigMetadata::sliding_window`
        //     is `None` for V4 (it is not a uniform-window model), so the
        //     pipeline cannot supply the width. `cfg_full.sliding_window` can.
        //
        // Once per step, not once per layer: all 43 layers read the same window
        // width, and the graph path is single-device.
        if crate::layers::has_graph_mode_positions() {
            match crate::layers::graph_mode_positions() {
                Some(pos) => {
                    let cap = self.cfg_full.sliding_window.max(1);
                    let mask = crate::layers::graph_mode_length_mask(&pos, cap, xs_embed.dtype())?;
                    // Trace slot 0: does a REPLAY rebuild the right length mask
                    // from the device position buffer, or is the mask the one
                    // capture baked? Pushed before the layers so the bisect
                    // reads mask → embed → lift → L0.. in execution order.
                    crate::layers::arc_layer_trace_push(&mask);
                    crate::layers::set_graph_mode_mask(Some(mask));
                }
                None => crate::layers::set_graph_mode_mask(None),
            }
        }

        let cache = &mut self.cache.normal().0;
        let _prof_mask = arc_profiler::span("causal_mask");
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
        drop(_prof_mask);

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
            let mut xs_4d = {
                let _s = arc_profiler::device_span("mhc.lift_3d_to_4d");
                mhc_head.rt.lift_3d_to_4d(&xs_embed)?
            };
            v4_nan_dbg(&xs_4d, "lift_3d_to_4d");
            v4_stat_dbg(&xs_4d, "lift_3d_to_4d");
            // Trace slots 1 and 2: the embedding (i.e. did the token id reach
            // the graph) and the mHC lift, both BEFORE any attention. A
            // divergence that starts here is an input; one that starts at L0
            // is inside the layer.
            crate::layers::arc_layer_trace_push(&xs_embed);
            crate::layers::arc_layer_trace_push(&xs_4d);
            let _prof_layers = arc_profiler::span("layers");
            for (i, layer) in self.layers.iter().enumerate() {
                // Aggregated across all layers by default (calls = n_layers,
                // with min/max); `ARC_PROFILE_UNROLL=1` splits by index.
                let _prof_layer = arc_profiler::span_idx("layer", i);
                xs_4d = {
                    let _s = arc_profiler::device_span("device_map.map");
                    self.mapper.map(xs_4d, i)?
                };
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
                crate::layers::arc_layer_trace_push(&xs_4d);
            }
            drop(_prof_layers);
            let xs_4d = xs_4d.to_device(&self.device)?;
            // Collapse via the learned global mHC head: 4-D → 3-D.
            let collapsed = {
                let _s = arc_profiler::device_span("mhc_head");
                mhc_head.forward(&xs_4d)?
            };
            v4_stat_dbg(&collapsed, "after_mhc_head");
            collapsed
        } else {
            let mut xs = xs_embed;
            let _prof_layers = arc_profiler::span("layers");
            for (i, layer) in self.layers.iter().enumerate() {
                let _prof_layer = arc_profiler::span_idx("layer", i);
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
                crate::layers::arc_layer_trace_push(&xs);
            }
            xs.to_device(&self.device)?
        };

        // Per-component decode profile. Emitted for BOTH residual paths and
        // stamped with the batch geometry — see `emit_decode_profile`.
        emit_decode_profile(input_ids, use_4d_mhc);

        v4_stat_dbg(&xs, "before_norm");

        // ---- MTP hidden-state capture (audit finding 1) ----------------
        // The reference hands the MTP head the target's own hidden state, NOT
        // an embedding: `deepseek_v4.py:1168-1175` returns `pre_hc_head` (the
        // residual stack *before* the final norm) and `logits_processor.py:
        // 603-606` makes it the captured `spec_info.hidden_states`, which
        // `deepseek_v4_nextn.py:152-154` feeds to `h_proj(hnorm(·))`.
        //
        // We capture the pre-`norm` state for EVERY input position (before
        // `extract_logits` narrows to the sampled rows), because the draft KV
        // prefill needs `h_i` at every context position, not just the last.
        //
        // Divergence, deliberate and documented: the reference's capture is
        // the *pre-collapse* `[T, hc_mult·d]` stack; Arc's MTP path has no
        // `hc_mult` stream axis (audit finding 5(b), separately owned), so we
        // capture the post-`hc_head`, pre-`norm` `[B, T, hidden]` state — the
        // same tensor in the "pre-norm" sense, one collapse later.
        //
        // 🔑 The capture is tagged with `seqlen_offsets` — the WHOLE vector, not
        // `seqlen_offsets[0]`. Row `i` of this block covers absolute positions
        // `[seqlen_offsets[i], seqlen_offsets[i] + T)`, so collapsing it to row
        // 0's offset made every other row's draft-KV extend read the leading
        // row's idea of where it is. Under the cohort rule the rows agree and
        // `MtpHiddenCapture::store_rows` folds them straight back to one shared
        // offset, which is byte-identical to what the collapse produced.
        if let Some(mtp) = &self.mtp_head {
            if mtp.hidden_capture.is_armed() {
                mtp.hidden_capture.store_rows(seqlen_offsets, &xs);
            }
        }

        let xs = {
            let _s = arc_profiler::device_span("final_norm");
            xs.apply(&self.norm)?
        };
        v4_stat_dbg(&xs, "after_norm");
        let xs = {
            let _s = arc_profiler::device_span("extract_logits");
            extract_logits(&xs, context_lens)?
        };

        let logits = {
            let _s = arc_profiler::device_span("lm_head");
            self.lm_head.forward_autocast(&xs)?
        };
        v4_stat_dbg(&logits, "logits");
        Ok(logits)
    }
}

impl IsqModel for DeepSeekV4 {
    /// Apply the deferred expert-parallel slice to every MoE layer.
    ///
    /// Under EP the routed experts are meant to be split across ranks, but a
    /// UQFF artifact holds all of them, so `MoEExperts` records the subset and
    /// refuses to run until this narrows the deserialized stacks. Returns how
    /// many quantized layers were narrowed (3 per MoE layer: gate, up, down).
    fn apply_pending_expert_parallel_slice(&mut self) -> Result<usize> {
        let mut narrowed = 0usize;
        for layer in self.layers.iter_mut() {
            if let MoeOrMlp::Moe(moe) = &mut layer.moe_or_mlp {
                narrowed += moe.experts.apply_pending_expert_subset()?;
            }
        }
        Ok(narrowed)
    }

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

    fn mtp_isq_floor_len(&mut self) -> usize {
        // `get_layers` pushes `h_proj` and `e_proj` immediately before the
        // block's tensors whenever the MTP head loaded at all, so the draft
        // path is the block's tail PLUS those two. They are excluded from
        // `mtp_isq_tail_len` because that boundary also names UQFF entries.
        let tail = self.mtp_isq_tail_len();
        if self.mtp_head.is_some() {
            tail + 2
        } else {
            0
        }
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
        // Arm the hidden-state side-channel: from here on every forward
        // retains its pre-`norm` hidden state for the draft path. Kept behind
        // an explicit arm so a model loaded with MTP tensors but served
        // without `--mtp-depth` never pays to hold a `[B, T, hidden]` prompt
        // activation alive.
        head.hidden_capture.arm();
        Some(crate::pipeline::mtp_pipeline::MtpDecodeKit {
            embed_tokens: self.embed_tokens.clone(),
            lm_head: self.lm_head.clone(),
            h_proj: head.h_proj.clone(),
            e_proj: head.e_proj.clone(),
            block: head.block.clone(),
            hidden_capture: head.hidden_capture.clone(),
        })
    }
}

impl AnyMoeBaseModelMixin for DeepSeekV4 {}

#[cfg(test)]
mod kv_footprint_tests {
    use super::*;
    use crate::kv_cache::{NormalCache, SingleCache};

    /// Bytes one token occupies in one half of a KV slot.
    ///
    /// `all_data` is over-allocated in `CACHE_GROW_SIZE` steps, so the honest
    /// per-token figure is the size of one row along the sequence dim — every
    /// dimension except `c.dim`, times the element size — not
    /// `all_data.elem_count()`.
    fn per_token_bytes(c: &SingleCache) -> usize {
        let d = c.all_data.as_ref().expect("cache half not materialised");
        let row: usize = d
            .dims()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != c.dim)
            .map(|(_, n)| *n)
            .product();
        row * d.dtype().size_in_bytes()
    }

    /// Drive `tokens` decode steps through the real V4 append path
    /// ([`append_kv_mqa`]) on a slot built exactly as
    /// [`NormalCache::new`] builds V4's, and return
    /// `(k_bytes_per_token, v_bytes_per_token)`.
    ///
    /// `rope_dim = None` stores dense (the fallback layout); `Some(rd)` runs the
    /// real quantizer and stores packed, which is what the live path does.
    fn drive_v4_kv(
        head_dim: usize,
        tokens: usize,
        rope_dim: Option<usize>,
    ) -> Result<(usize, usize, KvCache)> {
        let dev = Device::Cpu;
        let mut slot = KvCache::new_normal(2, 4096, NormalCache::CACHE_GROW_SIZE);
        for t in 0..tokens {
            // `[B=1, n_kv_heads=1, T=1, head_dim]` — V4's MQA K, one token.
            // Non-constant so the quantizer sees real block scales.
            let row: Vec<f32> = (0..head_dim)
                .map(|i| ((t + 1) as f32) * 0.017 + 0.25 * (i % 37) as f32)
                .collect();
            let k = Tensor::from_vec(row, (1, 1, 1, head_dim), &dev)?.to_dtype(DType::BF16)?;
            let packed = match rope_dim {
                Some(rd) => super::super::dsv4_kv_fp8::quantize_k(
                    &k,
                    rd,
                    super::super::dsv4_kv_fp8::KvQuantMode::CpuExact,
                )?,
                None => None,
            };
            append_kv_mqa(&mut slot, &k, packed.as_ref(), None)?;
        }
        let KvCache::Normal { k, v } = &slot else {
            panic!("V4 slot must be KvCache::Normal");
        };
        let (kb, vb) = (per_token_bytes(k), per_token_bytes(v));
        Ok((kb, vb, slot.clone()))
    }

    /// **The byte count.** V4 Flash geometry: MQA (`num_key_value_heads = 1`),
    /// `head_dim = 512`, `qk_rope_head_dim = 64`, BF16 activations, 43 layers.
    ///
    /// The ladder this pins, all measured by this one fixture:
    ///
    /// | layout | K half | V half | B/token/layer |
    /// |---|---|---|---|
    /// | pre-wave33, duplicated V | 512 BF16 | 512 BF16 | 2048 |
    /// | wave33, V a marker | 512 BF16 | 1 BF16 | 1026 |
    /// | **wave43, packed** | **448 U8** | **64 + 7 BF16** | **590** |
    ///
    /// Reference: SGLang's DSV4 pool charges **584** B/token/layer
    /// (`research/code/06_foundation/sglang/python/sglang/srt/mem_cache/
    /// deepseek_v4_memory_pool.py:93-111`, whose own assert spells the layout:
    /// `448` nope FP8 + `64*2` rope BF16 + `7` UE8M0 block scales + `1` pad).
    /// The 6-byte residual is deliberate: Arc keeps each block's `amax` at
    /// activation precision rather than rounding it into a UE8M0 exponent,
    /// which is what makes the round trip bit-exact against what was stored
    /// before (`dsv4_kv_fp8::kv_fp8_roundtrip_is_bit_exact_vs_reference`).
    #[test]
    fn v4_kv_bytes_per_token_per_layer() -> Result<()> {
        const HEAD_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const LAYERS: usize = 43;
        const BF16: usize = 2;
        const NOPE: usize = HEAD_DIM - ROPE_DIM; // 448
        const N_BLOCKS: usize = NOPE / super::super::dsv4_kv_fp8::KV_QUANT_BLOCK; // 7

        let (k_bytes, v_bytes, slot) = drive_v4_kv(HEAD_DIM, 300, Some(ROPE_DIM))?;

        // --- Fixture discrimination (DOCTRINE D12) -------------------------
        // Three ways this fixture could measure a wrong implementation as
        // right, each ruled out before anything is asserted about it:
        //
        //  1. if the code byte were as wide as the activation, the packed and
        //     the dense layouts would weigh the same;
        //  2. if the nope dims were the whole head, the rope tail would cost
        //     nothing and a layout that wrongly quantized the rope dims too
        //     would measure identically;
        //  3. if there were one block per token, per-block scales and one
        //     global scale would be indistinguishable.
        assert!(
            std::mem::size_of::<u8>() < BF16,
            "fixture cannot discriminate: a code byte the width of an \
             activation makes the packed and dense layouts weigh the same"
        );
        assert!(
            ROPE_DIM > 0 && NOPE > 0,
            "fixture cannot discriminate: the rope tail must be non-empty, or \
             quantizing it too would measure the same"
        );
        assert!(
            N_BLOCKS > 1,
            "fixture cannot discriminate: {N_BLOCKS} block(s) per token makes \
             per-block and global scales weigh the same"
        );

        assert_eq!(
            k_bytes, NOPE,
            "K half: {NOPE} E4M3 codes, one byte each"
        );
        assert_eq!(
            v_bytes,
            (ROPE_DIM + N_BLOCKS) * BF16,
            "V half: {ROPE_DIM} rope dims + {N_BLOCKS} block amax, at activation precision"
        );

        let per_token_per_layer = k_bytes + v_bytes;
        assert_eq!(per_token_per_layer, 590);

        // The two layouts this replaced, computed from the same geometry.
        // Asserting the gaps (rather than only the new number) is what makes a
        // silent revert fail here instead of passing quietly.
        let duplicated_v_layout = 2 * HEAD_DIM * BF16;
        let marker_layout = HEAD_DIM * BF16 + V4_V_MARKER_WIDTH * BF16;
        assert_eq!(duplicated_v_layout, 2048);
        assert_eq!(marker_layout, 1026);
        assert!(
            per_token_per_layer < marker_layout,
            "the packed layout must be strictly smaller than the marker layout"
        );

        // And the dense layout is still exactly what it was — measured, not
        // assumed, by driving the same fixture with the quantizer switched off.
        let (dense_k, dense_v, _) = drive_v4_kv(HEAD_DIM, 300, None)?;
        assert_eq!(dense_k + dense_v, marker_layout);

        // Whole-model, all 43 layers.
        assert_eq!(per_token_per_layer * LAYERS, 25_370);
        assert_eq!(marker_layout * LAYERS, 44_118);
        assert_eq!(duplicated_v_layout * LAYERS, 88_064);

        // The two halves must stay length- and capacity-synchronised: the cache
        // managers rebuild BOTH halves from the K half's `current_seq_len` and
        // `capacity_seq_len` (`kv_cache/mod.rs` `clone_in_cache`), so a drift
        // here would silently mis-slice every batched sequence.
        let KvCache::Normal { k, v } = &slot else {
            panic!("V4 slot must be KvCache::Normal");
        };
        assert_eq!(k.current_seq_len, 300);
        assert_eq!(k.current_seq_len, v.current_seq_len);
        assert_eq!(k.capacity_seq_len, v.capacity_seq_len);
        assert_eq!(k.dim, v.dim);
        Ok(())
    }

    /// The cache must give back what went in. Drives the packed store over many
    /// tokens and checks every reconstructed row against the reference round
    /// trip — this is the end-to-end form of the bit-exactness claim, through
    /// the real `KvCache` rather than the quantizer alone.
    #[test]
    fn packed_cache_reconstructs_the_stored_keys() -> Result<()> {
        const HEAD_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        let dev = Device::Cpu;
        let mut slot = KvCache::new_normal(2, 4096, NormalCache::CACHE_GROW_SIZE);
        let mode = super::super::dsv4_kv_fp8::KvQuantMode::CpuExact;

        let mut expected: Vec<u16> = Vec::new();
        let tokens = 6usize;
        for t in 0..tokens {
            let row: Vec<f32> = (0..HEAD_DIM)
                .map(|i| ((t + 1) as f32) * 0.31 - 0.02 * (i % 53) as f32)
                .collect();
            let k = Tensor::from_vec(row, (1, 1, 1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let packed = super::super::dsv4_kv_fp8::quantize_k(&k, ROPE_DIM, mode)?
                .expect("448 nope dims are 7 whole blocks");
            // What the pre-packing path would have stored for this token.
            expected.extend(
                packed
                    .dequant(DType::BF16)?
                    .flatten_all()?
                    .to_vec1::<half::bf16>()?
                    .into_iter()
                    .map(|v| v.to_bits()),
            );
            let cached = append_kv_mqa(&mut slot, &k, Some(&packed), None)?;
            assert!(matches!(cached, V4CachedK::Packed(_)));
            assert_eq!(cached.seq_len()?, t + 1);
        }

        let cached = {
            let KvCache::Normal { k, v } = &slot else {
                panic!("V4 slot must be KvCache::Normal")
            };
            V4CachedK::Packed(Box::new(V4PackedK {
                codes: k.current_data()?.expect("codes materialised"),
                side: v.current_data()?.expect("side materialised"),
                rope_dim: ROPE_DIM,
            }))
        };
        let rebuilt = cached.span(0, tokens, DType::BF16)?;
        assert_eq!(rebuilt.dims(), &[1, 1, tokens, HEAD_DIM]);
        let got: Vec<u16> = rebuilt
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .into_iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(
            got, expected,
            "the packed cache must read back bit-identically to the BF16 tensor \
             the pre-packing path stored"
        );

        // And the span the decode path actually asks for is a suffix of it.
        let (base, keep) = super::super::dsv4_attention::raw_keep_span(1, 3, tokens);
        assert_eq!((base, keep), (3, 3));
        let span = cached.span(base, keep, DType::BF16)?;
        let span_bits: Vec<u16> = span
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .into_iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(span_bits, expected[base * HEAD_DIM..]);
        Ok(())
    }

    /// With FP8 storage declined (no whole 64-wide nope blocks), the dense
    /// fallback must still return K, not the marker: that tensor is what
    /// `dsv4_attention` receives as BOTH K and V.
    #[test]
    fn append_kv_mqa_returns_k_for_both_sides() -> Result<()> {
        let dev = Device::Cpu;
        let head_dim = 8usize;
        let mut slot = KvCache::new_normal(2, 4096, NormalCache::CACHE_GROW_SIZE);

        let mut expected = Vec::new();
        for t in 0..5u32 {
            let row: Vec<f32> = (0..head_dim)
                .map(|i| (t as f32) + 0.25 * i as f32)
                .collect();
            expected.extend(row.iter().copied());
            let k = Tensor::from_vec(row, (1, 1, 1, head_dim), &dev)?;
            let cached = append_kv_mqa(&mut slot, &k, None, None)?;
            assert!(matches!(cached, V4CachedK::Dense(_)));
            let materialised = cached.span(0, cached.seq_len()?, DType::F32)?;
            assert_eq!(materialised.dims(), &[1, 1, (t + 1) as usize, head_dim]);
        }

        let cached = append_kv_mqa(
            &mut slot,
            &Tensor::zeros((1, 1, 1, head_dim), DType::F32, &dev)?,
            None,
            None,
        )?;
        let got: Vec<f32> = cached
            .span(0, 5, DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(
            got, expected,
            "the cached K read back must be K, not the marker"
        );

        // And the marker itself is all zeros and 1-wide — nothing readable.
        let KvCache::Normal { v, .. } = &slot else {
            panic!()
        };
        let marker = v.current_data()?.expect("marker materialised");
        assert_eq!(marker.dims(), &[1, 1, 6, V4_V_MARKER_WIDTH]);
        assert!(marker
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| *x == 0.0));
        Ok(())
    }

    /// A V4 KV slot built **the way the engine builds it for a prompt step**,
    /// which is not how any other test in this file builds one.
    ///
    /// Every test above starts from `KvCache::new_normal`, whose `all_data` is
    /// `None`, so `SingleCache::append` allocates the buffer from the *first*
    /// `src` and can never disagree with it. The live path does the opposite:
    ///
    /// * `engine/add_request.rs:464` allocates a per-sequence buffer
    ///   `[1, num_kv_heads, cap, k_head_dim]` in the **activation dtype** —
    ///   for V4 Flash, BF16 and `head_dim = 512` wide, for BOTH halves
    ///   (`ModelConfigMetadata::v_head_dim` is also `cfg.head_dim`);
    /// * on a prompt step with `token_offset() == 0` the engine issues
    ///   `CacheInstruction::Reset { load_preallocated_cache: true }`
    ///   (`engine/mod.rs:470`), and `NormalCacheManager::set_none_cache`
    ///   installs that buffer as `all_data` (`kv_cache/mod.rs:802`) **before
    ///   the first append**.
    ///
    /// So the first thing V4's append meets is a pre-grown BF16 512-wide
    /// buffer, and neither of V4's two layouts fits it: the packed K half is
    /// U8 448-wide and the dense V half is a 1-wide marker. This helper
    /// reproduces that construction exactly.
    fn preallocated_v4_slot(capacity: usize, head_dim: usize) -> Result<KvCache> {
        let dev = Device::Cpu;
        // `k_head_dim == v_head_dim == cfg.head_dim` (see `ModelConfigMetadata`
        // in `DeepSeekV4::new`), and the dtype is `activation_dtype`.
        let shape = (1usize, 1usize, capacity, head_dim);
        Ok(KvCache::Normal {
            k: SingleCache {
                all_data: Some(Tensor::zeros(shape, DType::BF16, &dev)?),
                dim: 2,
                current_seq_len: 0,
                max_seq_len: 4096,
                capacity_seq_len: capacity,
                first_cached: 0,
                retain_window: None,
            },
            v: SingleCache {
                all_data: Some(Tensor::zeros(shape, DType::BF16, &dev)?),
                dim: 2,
                current_seq_len: 0,
                max_seq_len: 4096,
                capacity_seq_len: capacity,
                first_cached: 0,
                retain_window: None,
            },
        })
    }

    /// **The regression wave48-BY hit on hardware.** Packed storage appending
    /// into the engine's preallocated slot: the K half's `all_data` is BF16 and
    /// the codes are U8, so `SingleCache::append`'s `slice_set` used to fail
    /// with `dtype mismatch in slice-set, lhs: BF16, rhs: U8` — every forward,
    /// including the engine's own dummy run, 0 tokens returned.
    ///
    /// The assertion is not merely "it does not error": a cache that silently
    /// dropped the write, or reallocated and lost the token, would still not
    /// error. It must read back **bit-identically** to the reference round
    /// trip, which is the same claim
    /// [`packed_cache_reconstructs_the_stored_keys`] makes about the
    /// freshly-allocated slot.
    #[test]
    fn packed_append_into_the_engines_preallocated_slot_round_trips() -> Result<()> {
        const HEAD_DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const CAPACITY: usize = NormalCache::CACHE_GROW_SIZE;
        let dev = Device::Cpu;
        let mode = super::super::dsv4_kv_fp8::KvQuantMode::CpuExact;
        let mut slot = preallocated_v4_slot(CAPACITY, HEAD_DIM)?;

        // Fixture discrimination (DOCTRINE D12): the slot must actually
        // disagree with what the packed layout writes, or this test passes for
        // a cache that never had to reallocate anything.
        {
            let KvCache::Normal { k, v } = &slot else {
                panic!("preallocated V4 slot must be Normal")
            };
            let (kd, vd) = (k.all_data.as_ref().unwrap(), v.all_data.as_ref().unwrap());
            assert_eq!(kd.dtype(), DType::BF16);
            assert_ne!(
                kd.dtype(),
                DType::U8,
                "fixture cannot discriminate: the preallocated K half must be \
                 the activation dtype, or there is no dtype contract to break"
            );
            assert_ne!(
                kd.dims()[3],
                HEAD_DIM - ROPE_DIM,
                "fixture cannot discriminate: the preallocated K half must be \
                 head_dim-wide, not nope-wide"
            );
            assert_ne!(
                vd.dims()[3],
                ROPE_DIM + (HEAD_DIM - ROPE_DIM) / super::super::dsv4_kv_fp8::KV_QUANT_BLOCK,
                "fixture cannot discriminate: the preallocated V half must be \
                 head_dim-wide, not [rope ++ amax]-wide"
            );
        }

        let tokens = 5usize;
        let mut expected: Vec<u16> = Vec::new();
        for t in 0..tokens {
            let row: Vec<f32> = (0..HEAD_DIM)
                .map(|i| ((t + 1) as f32) * 0.41 - 0.03 * (i % 47) as f32)
                .collect();
            let k = Tensor::from_vec(row, (1, 1, 1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let packed = super::super::dsv4_kv_fp8::quantize_k(&k, ROPE_DIM, mode)?
                .expect("448 nope dims are 7 whole blocks");
            expected.extend(
                packed
                    .dequant(DType::BF16)?
                    .flatten_all()?
                    .to_vec1::<half::bf16>()?
                    .into_iter()
                    .map(|v| v.to_bits()),
            );
            let cached = append_kv_mqa(&mut slot, &k, Some(&packed), None)?;
            assert!(matches!(cached, V4CachedK::Packed(_)));
            assert_eq!(cached.seq_len()?, t + 1);
        }

        let KvCache::Normal { k, v } = &slot else {
            panic!("V4 slot must be KvCache::Normal")
        };
        // The buffer that survives is the packed one, at the capacity the
        // engine preallocated — the pre-grow is kept, only its dtype/width are
        // corrected.
        let kd = k.all_data.as_ref().expect("codes materialised");
        assert_eq!(kd.dtype(), DType::U8);
        assert_eq!(kd.dims(), &[1, 1, CAPACITY, HEAD_DIM - ROPE_DIM]);
        assert_eq!(k.capacity_seq_len, CAPACITY);
        assert_eq!(k.current_seq_len, tokens);
        assert_eq!(k.current_seq_len, v.current_seq_len);

        let cached = V4CachedK::Packed(Box::new(V4PackedK {
            codes: k.current_data()?.expect("codes materialised"),
            side: v.current_data()?.expect("side materialised"),
            rope_dim: ROPE_DIM,
        }));
        let got: Vec<u16> = cached
            .span(0, tokens, DType::BF16)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .into_iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(
            got, expected,
            "the packed cache must read back bit-identically after being \
             appended into the engine's preallocated slot"
        );
        Ok(())
    }

    /// The same contract for the **default** (opt-out) layout, which is what
    /// `master` now runs: dense K plus the 1-wide V marker. This one never
    /// disagreed on dtype, so it never produced wave48-BY's message — it
    /// disagreed on WIDTH, and `slice_set` checks dtype before shape
    /// (`candle-core/src/tensor_cat.rs:254` vs `:278`), so this failure was
    /// hiding *behind* the FP8 one. Flipping the gate alone would have moved
    /// the error, not removed it.
    #[test]
    fn dense_append_into_the_engines_preallocated_slot_round_trips() -> Result<()> {
        const HEAD_DIM: usize = 512;
        const CAPACITY: usize = NormalCache::CACHE_GROW_SIZE;
        let dev = Device::Cpu;
        let mut slot = preallocated_v4_slot(CAPACITY, HEAD_DIM)?;

        let tokens = 4usize;
        let mut expected: Vec<u16> = Vec::new();
        for t in 0..tokens {
            let row: Vec<f32> = (0..HEAD_DIM)
                .map(|i| ((t + 1) as f32) * 0.13 + 0.05 * (i % 29) as f32)
                .collect();
            let k = Tensor::from_vec(row, (1, 1, 1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            expected.extend(
                k.flatten_all()?
                    .to_vec1::<half::bf16>()?
                    .into_iter()
                    .map(|v| v.to_bits()),
            );
            let cached = append_kv_mqa(&mut slot, &k, None, None)?;
            assert!(matches!(cached, V4CachedK::Dense(_)));
        }

        let KvCache::Normal { k, v } = &slot else {
            panic!("V4 slot must be KvCache::Normal")
        };
        // K keeps the preallocated buffer verbatim (it fits); only V is
        // rebuilt, to the marker width.
        assert_eq!(
            k.all_data.as_ref().unwrap().dims(),
            &[1, 1, CAPACITY, HEAD_DIM]
        );
        assert_eq!(
            v.all_data.as_ref().unwrap().dims(),
            &[1, 1, CAPACITY, V4_V_MARKER_WIDTH]
        );
        assert_eq!(k.current_seq_len, tokens);
        assert_eq!(k.current_seq_len, v.current_seq_len);

        let got: Vec<u16> = k
            .current_data()?
            .expect("dense K materialised")
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .into_iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, expected, "dense K must read back exactly what went in");
        Ok(())
    }

    /// FP8 K storage is **opt-in**. PR #72 shipped `!(v == "0")`, so every
    /// value — and, decisively, *unset* — turned it on; the first GPU that ran
    /// it failed every forward (wave48-BY). Only the literal `1` may enable it,
    /// and `ARC_V4_CAPTURE_PROBE` still vetoes it because the CUDA-graph decode
    /// arm cannot write a U8 cache.
    ///
    /// This tests [`fp8_kv_enabled_from`] rather than [`v4_fp8_kv_enabled`] on
    /// purpose: the latter is a `OnceLock`, so a test that set the variable
    /// would resolve it once and prove nothing about the default.
    #[test]
    fn v4_fp8_kv_is_opt_in() {
        assert!(!fp8_kv_enabled_from(None, false), "unset must be OFF");
        assert!(!fp8_kv_enabled_from(Some("0"), false));
        assert!(!fp8_kv_enabled_from(Some(""), false));
        assert!(!fp8_kv_enabled_from(Some("true"), false));
        assert!(!fp8_kv_enabled_from(Some("on"), false));
        assert!(!fp8_kv_enabled_from(Some("2"), false));
        assert!(fp8_kv_enabled_from(Some("1"), false), "=1 must be ON");
        assert!(
            !fp8_kv_enabled_from(Some("1"), true),
            "ARC_V4_CAPTURE_PROBE must veto it: write_kv_inplace has no U8 arm"
        );
    }

    /// The per-row ragged mask is **opt-in**, and off it hands
    /// `dsv4_attention` nothing at all — not a uniform vector it would have to
    /// recognise, but `None`, so the scalar path is taken structurally.
    ///
    /// Tests [`ragged_row_q0_from`] rather than [`ragged_row_q0`] for the same
    /// reason `v4_fp8_kv_is_opt_in` does: the latter is a `OnceLock`.
    #[test]
    fn ragged_row_q0_is_opt_in() {
        let offsets = [10usize, 8, 5];
        assert!(
            ragged_row_q0_from(false, &offsets).is_none(),
            "with per-sequence KV advance off no batch is ever left-aligned, so the model must \
             not be handed per-row positions at all"
        );
        assert_eq!(
            ragged_row_q0_from(true, &offsets),
            Some(&offsets[..]),
            "on, the row positions are `seqlen_offsets` verbatim — the value the engine already \
             threads for RoPE, not a new one"
        );
    }

    /// A non-`Normal` KV slot must be refused loudly rather than silently
    /// quantizing or windowing a 1-wide marker.
    #[test]
    fn non_normal_kv_slot_is_refused() -> Result<()> {
        let dev = Device::Cpu;
        let mut rotating = KvCache::new_rotating(2, 16, NormalCache::CACHE_GROW_SIZE);
        let k = Tensor::zeros((1, 1, 1, 8), DType::F32, &dev)?;
        let err = append_kv_mqa(&mut rotating, &k, None, None).unwrap_err();
        assert!(
            err.to_string().contains("Rotating"),
            "expected a loud slot-type error, got: {err}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Only the tests drive the pin override; importing it at module scope made
    // it an unused import on every non-test build.
    use crate::kv_cache::xs_rolling;

    /// The **verbatim** `config.json` published at
    /// <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json>
    /// (fetched 2026-08-14, HTTP 200, 1749 bytes; repo is public + MIT).
    ///
    /// Byte-for-byte, keys included that Arc does not parse (`expert_dtype`,
    /// `num_nextn_predict_layers`, `initializer_range`, …) — the whole point
    /// of the fixture is that it is *not* slimmed. The previous fixture was
    /// hand-written and omitted `rope_scaling`, `norm_topk_prob`,
    /// `quantization_config`, `hidden_act`, `attention_bias` and
    /// `tie_word_embeddings`, which made it useless for settling exactly the
    /// questions the V4 reference audit could not close
    /// (`docs/notes/v4-reference-audit.md`).
    const V4_FLASH_CONFIG_JSON: &str = r#"{
  "architectures": [
    "DeepseekV4ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 1,
  "expert_dtype": "fp4",
  "hc_eps": 1e-06,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "head_dim": 512,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 512,
  "initializer_range": 0.02,
  "max_position_embeddings": 1048576,
  "model_type": "deepseek_v4",
  "moe_intermediate_size": 2048,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 64,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 43,
  "num_hash_layers": 3,
  "num_key_value_heads": 1,
  "num_nextn_predict_layers": 1,
  "o_groups": 8,
  "o_lora_rank": 1024,
  "q_lora_rank": 1024,
  "qk_rope_head_dim": 64,
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [
      128,
      128
    ]
  },
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 1.5,
  "scoring_func": "sqrtsoftplus",
  "sliding_window": 128,
  "swiglu_limit": 10.0,
  "tie_word_embeddings": false,
  "topk_method": "noaux_tc",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.57.1",
  "use_cache": true,
  "vocab_size": 129280,
  "compress_rope_theta": 160000,
  "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
}"#;

    /// The real V4 Flash `config.json` parses cleanly through
    /// `DeepSeekV4Config` and yields the dimensions the forward path assumes.
    #[test]
    fn v4_flash_real_config_parses() {
        let cfg: DeepSeekV4Config =
            serde_json::from_str(V4_FLASH_CONFIG_JSON).expect("V4 config parses");
        // Derived qk_nope_head_dim = head_dim - qk_rope_head_dim
        assert_eq!(cfg.qk_nope_head_dim(), 448);
        assert_eq!(cfg.head_dim, 512);
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.q_lora_rank, Some(1024));
        assert_eq!(cfg.qk_rope_head_dim, 64);
        assert_eq!(cfg.hc_mult, 4);
        assert_eq!(cfg.hc_sinkhorn_iters, 20);
        assert_eq!(cfg.hc_eps, 1e-6);
        assert_eq!(cfg.num_hash_layers, 3);
        assert!(matches!(cfg.scoring_func, ScoringFunc::SqrtSoftplus));
        assert!(matches!(cfg.topk_method, TopkMethod::NoAuxTc));
        assert_eq!(cfg.routed_scaling_factor, 1.5);
        assert_eq!(cfg.o_lora_rank, Some(1024));
        assert_eq!(cfg.o_groups, Some(8));
        assert_eq!(cfg.index_n_heads, 64);
        assert_eq!(cfg.index_head_dim, 128);
        assert_eq!(cfg.index_topk, 512);
        assert_eq!(cfg.sliding_window, 128);
        assert!(!cfg.attention_bias);
        assert!(!cfg.tie_word_embeddings);
        assert!(matches!(cfg.hidden_act, Activation::Silu));
        // MQA broadcast factor
        assert_eq!(cfg.n_kv_groups(), 64);

        // Keys the real config genuinely does NOT publish. Each of these is
        // load-bearing: the forward path derives them instead, and the audit
        // premises that depend on their absence are recorded below.
        assert_eq!(cfg.intermediate_size, None);
        assert_eq!(cfg.kv_lora_rank, None);
        assert_eq!(cfg.v_head_dim, None);
        assert_eq!(cfg.qk_nope_head_dim, None);
        // `first_k_dense_replace` is absent ⇒ default 0 ⇒ every one of the 43
        // layers is MoE. Corroborated by `model.safetensors.index.json`, which
        // ships 43 × 256 = 11008 `layers.N.ffn.experts.M.w1.weight` tensors and
        // no dense `mlp.*` tensors at any layer.
        assert_eq!(cfg.first_k_dense_replace, 0);

        // Compress ratios cover all 43 layers + the MTP slot
        assert_eq!(cfg.compress_ratios.len(), 44);
        // Layers 0/1 = standard, even 2..=42 = CSA(4), odd 3..=41 = HCA(128),
        // slot 43 = standard (the MTP block).
        assert_eq!(cfg.layer_compress_ratio(0), 0);
        assert_eq!(cfg.layer_compress_ratio(1), 0);
        assert_eq!(cfg.layer_compress_ratio(2), 4);
        assert_eq!(cfg.layer_compress_ratio(3), 128);
        assert_eq!(cfg.layer_compress_ratio(42), 4);
        assert_eq!(cfg.layer_compress_ratio(43), 0);
        // Exactly three ratio-0 slots: layers 0, 1 and the MTP slot 43. Every
        // other layer is compressed — layer 42 included.
        let standard: Vec<usize> = (0..44)
            .filter(|i| cfg.layer_compress_ratio(*i) == 0)
            .collect();
        assert_eq!(standard, vec![0, 1, 43]);
        // Out of bounds → 0
        assert_eq!(cfg.layer_compress_ratio(99), 0);
    }

    /// The five open questions in `docs/notes/v4-reference-audit.md` that
    /// bottomed out in "we do not have the real config.json". Each assertion
    /// below is the settled answer; do not relax one without re-fetching the
    /// file from the URL quoted on [`V4_FLASH_CONFIG_JSON`].
    #[test]
    fn v4_flash_config_settles_the_audit_questions() {
        let raw: serde_json::Value = serde_json::from_str(V4_FLASH_CONFIG_JSON).unwrap();
        let cfg: DeepSeekV4Config = serde_json::from_str(V4_FLASH_CONFIG_JSON).unwrap();

        // ── Audit finding 11: group-limited routing ──────────────────────
        // The shipped config publishes NEITHER `n_group` NOR `topk_group`, so
        // Arc's serde defaults (1/1) apply and `MoeGate::forward` takes the
        // flat top-k branch — which is what SGLang force-disables group
        // limiting to reach. No divergence, and the group branch is dead code
        // for this checkpoint.
        assert!(raw.get("n_group").is_none(), "real config has no n_group");
        assert!(
            raw.get("topk_group").is_none(),
            "real config has no topk_group"
        );
        assert_eq!(cfg.n_group, 1);
        assert_eq!(cfg.topk_group, 1);

        // ── Audit finding 12: hash-layer key spelling ────────────────────
        // Spelled `num_hash_layers`, NOT `n_hash_layers`. That is the key
        // SGLang's runtime reads (`getattr(config, "num_hash_layers", 0)`),
        // so the reference runs three hash layers too — Arc matches.
        // Corroborated by the tensor index: `layers.{0,1,2}.ffn.gate.tid2eid`
        // exist and are the only three layers WITHOUT a `ffn.gate.bias`.
        assert!(raw.get("num_hash_layers").is_some());
        assert!(
            raw.get("n_hash_layers").is_none(),
            "the SGLang dataclass spelling is not what ships"
        );
        assert_eq!(cfg.num_hash_layers, 3);

        // ── Audit finding 24: rope_scaling / mscale ──────────────────────
        // `rope_scaling` IS present (the old fixture omitted it entirely and
        // could not have been the real config — the reference dereferences
        // `rope_scaling["factor"]` unconditionally). It carries no `mscale`
        // and no `mscale_all_dim`, so Arc's defaults (1.0 / 1.0) apply and
        // the sin/cos magnitude factor is
        // `yarn_get_mscale(f, 1.0) / yarn_get_mscale(f, 1.0) == 1.0` — i.e.
        // Arc's mscale multiply is a no-op, matching the reference's
        // unit-magnitude `torch.polar`.
        let rs = raw.get("rope_scaling").expect("rope_scaling is present");
        assert!(rs.get("mscale").is_none(), "no mscale key ships");
        assert!(
            rs.get("mscale_all_dim").is_none(),
            "no mscale_all_dim ships"
        );
        match cfg.rope_scaling.as_ref().expect("parses as Yarn") {
            DeepSeekV2RopeScaling::Yarn {
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                factor,
                mscale,
                mscale_all_dim,
                ..
            } => {
                assert_eq!(*original_max_position_embeddings, 65536);
                assert_eq!(*beta_fast, 32.0);
                assert_eq!(*beta_slow, 1.0);
                assert_eq!(*factor, 16.0);
                // Equal ⇒ the mscale ratio is exactly 1.0 ⇒ neutral.
                assert_eq!(mscale, mscale_all_dim);
                assert_eq!(*mscale, 1.0);
            }
            other => panic!("expected Yarn rope scaling, got {other:?}"),
        }

        // ── Audit finding 5 (rider i): swiglu_limit ──────────────────────
        // The key IS published, so the reference's
        // `getattr(config, "swiglu_limit", None)` resolves to 10.0 and the
        // reference clamps. Arc's hard default of 10.0 coincides — which
        // means the clamp is REQUIRED, and every Arc expert path that skips
        // it is a genuine divergence, not a config-dependent maybe.
        assert!(raw.get("swiglu_limit").is_some());
        assert_eq!(cfg.swiglu_limit, 10.0);

        // ── compress_rope_theta ──────────────────────────────────────────
        // Published as 160000, matching Arc's default. SGLang's *dataclass*
        // default of 40000 is never reached for this checkpoint.
        assert_eq!(
            raw.get("compress_rope_theta").and_then(|v| v.as_f64()),
            Some(160_000.0)
        );
        assert_eq!(cfg.compress_rope_theta, 160_000.0);
        assert_eq!(cfg.rope_theta, 10_000.0);

        // ── Audit finding 21: norm_topk_prob ─────────────────────────────
        // Published as `true`; now parsed rather than inferred.
        assert!(cfg.norm_topk_prob);

        // ── The fp8 checkpoint block parses (finding 14 context) ─────────
        // `scale_fmt: "ue8m0"` is direct evidence that the shipped weight
        // scales are power-of-two, which is what finding 14 says Arc's
        // FP8 K-cache path does not reproduce.
        assert_eq!(
            raw.pointer("/quantization_config/scale_fmt")
                .and_then(|v| v.as_str()),
            Some("ue8m0")
        );
        match cfg.quantization_config.as_ref().expect("fp8 block parses") {
            QuantizedConfig::Fp8 { weight_block_size } => {
                assert_eq!(weight_block_size.as_deref(), Some(&[128usize, 128][..]));
            }
            other => panic!("expected Fp8 quantization config, got {other:?}"),
        }
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
            raw_prefix: 0,
            row_q0: None,
            graph_positions: None,
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

    // ============================================================
    // Wave 29 (BC): the PagedAttention arm is single-sequence only.
    //
    // `PagedAttention::cache_write_and_gather` returns the gathered K/V as
    // `[1, H, N_total, D]` — the CUDA `gather_kv_cache` emits
    // `[sum_i(seqlen_i), H, D]` and the wrapper lifts it with a bare
    // `unsqueeze(0)` (`paged_attention/layers/paged_attention.rs:508-509`).
    // That is a varlen PACK, not a batch. `dsv4_attention` reads `t_k` as one
    // sequence's cached length and derives `q0 = t_k - t_q` plus the
    // sliding-window and compressed-block masks from it
    // (`dsv4_attention.rs:270-294`), so at `bs > 1` every query attends over
    // the concatenation of all sequences' keys at wrong absolute positions.
    //
    // The two tests below (a) demonstrate that corruption on real tensors and
    // (b) prove the guard that now refuses it.

    /// The varlen pack is NOT a batch: attending over it gives sequence 0 a
    /// different answer than attending over its own keys alone. This is the
    /// concrete failure `v4_paged_dispatch_precheck` exists to prevent — it is
    /// demonstrated here rather than asserted, so the guard is not justified by
    /// a claim nobody checked.
    #[test]
    fn v4_paged_varlen_pack_leaks_keys_across_sequences() -> Result<()> {
        use crate::attention::SdpaParams;
        use crate::models::dsv4_attention::{dsv4_attention, Dsv4AttentionConfig};
        use crate::pipeline::text_models_inputs_processor::FlashParams;

        let device = Device::Cpu;
        let head_dim = 8;
        let t_k = 3; // cached length of EACH sequence
        let n_q_heads = 1;

        let mk = |seed: f32, t: usize| -> Result<Tensor> {
            Tensor::from_vec(
                (0..(t * head_dim))
                    .map(|i| ((i as f32) * 0.31 + seed).sin())
                    .collect::<Vec<f32>>(),
                (1, 1, t, head_dim),
                &device,
            )
        };
        // Per-sequence K/V (V == K on V4) and one decode query each.
        let k0 = mk(0.0, t_k)?;
        let k1 = mk(5.0, t_k)?;
        let q0 = mk(1.0, 1)?;
        let q1 = mk(2.0, 1)?;

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
        // Standard (ratio-0) layer with a window wide enough that masking is
        // not what separates the two results — only the key set is.
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Standard,
            sliding_window: 1024,
            raw_prefix: 0,
            row_q0: None,
            graph_positions: None,
        };

        // Correct: sequence 0 attends over its own 3 keys.
        let want = dsv4_attention(&q0, &k0, &k0, None, None, &flash_params, &sdpa_params, cfg)?;

        // What the paged arm would hand it at bs=2: K/V packed along the
        // sequence axis with B still 1, queries batched with B=2.
        let k_pack = Tensor::cat(&[&k0, &k1], 2)?.contiguous()?;
        let q_batch = Tensor::cat(&[&q0, &q1], 0)?.contiguous()?;
        assert_eq!(k_pack.dims(), &[1, 1, 2 * t_k, head_dim]);
        assert_eq!(q_batch.dims(), &[2, n_q_heads, 1, head_dim]);

        // (a) The mechanism, silently: `dsv4_attention` reads `t_k` as ONE
        //     sequence's cached length, so the pack changes both the key set
        //     sequence 0 sees (6 keys, 3 of them sequence 1's) and its own
        //     absolute position (`q0 = t_k - t_q`). Same query, same weights,
        //     different answer — no error anywhere.
        let leaked = dsv4_attention(
            &q0,
            &k_pack,
            &k_pack,
            None,
            None,
            &flash_params,
            &sdpa_params,
            cfg,
        )?;
        let want_v: Vec<f32> = want.flatten_all()?.to_vec1()?;
        let leaked_v: Vec<f32> = leaked.flatten_all()?.to_vec1()?;
        assert_eq!(want_v.len(), leaked_v.len());
        let max_diff = want_v
            .iter()
            .zip(leaked_v.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            max_diff > 1e-4,
            "the two-sequence pack reproduced sequence 0's own answer exactly \
             (max_diff={max_diff}); if gather ever starts returning a real \
             batch, delete v4_paged_dispatch_precheck rather than keep a guard \
             that no longer guards anything"
        );

        // (b) And the honestly-batched form does not survive at all: with `q`
        //     at B=2 against the B=1 pack, the CPU SDPA backend indexes off the
        //     end of its slice. A crash deep in `attention/backends/cpu.rs` is
        //     not a diagnosis — which is why the guard names the batch size at
        //     the call site instead.
        let batched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dsv4_attention(
                &q_batch,
                &k_pack,
                &k_pack,
                None,
                None,
                &flash_params,
                &sdpa_params,
                cfg,
            )
        }));
        let survived_cleanly = matches!(&batched, Ok(Ok(_)));
        assert!(
            !survived_cleanly,
            "a B=2 query against a B=1 varlen pack returned an output — if the \
             backends have started broadcasting this, re-derive whether the \
             result is per-sequence correct before relaxing the guard"
        );
        Ok(())
    }

    /// The guard actually invoked at the PagedAttention call site in
    /// `Attention::forward`. Mutation check: make `v4_paged_dispatch_precheck`
    /// return `Ok(())` unconditionally and this test fails.
    #[test]
    fn v4_paged_dispatch_precheck_refuses_multi_sequence_batches() {
        assert!(
            v4_paged_dispatch_precheck(1).is_ok(),
            "single-sequence paged decode is the supported case and must pass"
        );
        for bs in [2usize, 8, 32, 128] {
            let err = v4_paged_dispatch_precheck(bs).expect_err(
                "bs>1 must be refused: cache_write_and_gather packs, it does not batch",
            );
            let msg = err.to_string();
            assert!(
                msg.contains("batch_size > 1") && msg.contains(&bs.to_string()),
                "the refusal must name the offending batch size so a paid-box log \
                 says what happened; got: {msg}"
            );
        }
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
            raw_prefix: 0,
            row_q0: None,
            graph_positions: None,
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
            raw_prefix: 0,
            row_q0: None,
            graph_positions: None,
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

    // ---------------------------------------------------------------------
    // Audit finding 5 — the shared expert's SwiGLU clamp.
    // ---------------------------------------------------------------------

    /// A layer config small enough to build a real `Moe` on CPU, carrying the
    /// V4-shaped fields the MoE construction reads. `swiglu_limit` is 1.0 (not
    /// the shipped 10.0) so the fixture's pre-activations sit far past it.
    const MOE_FIXTURE_JSON: &str = r#"{
        "vocab_size": 8,
        "hidden_size": 2,
        "moe_intermediate_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "num_hash_layers": 0,
        "max_position_embeddings": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "qk_rope_head_dim": 2,
        "head_dim": 4,
        "swiglu_limit": 1.0,
        "compress_ratios": [0]
    }"#;

    /// silu(1.0) * 1.0 — the reference output for a gate pre-activation of 5.0
    /// and an up pre-activation of 3.0 clamped at 1.0.
    const SHARED_CLAMPED: f32 = 0.731_058_6;
    /// silu(5.0) * 3.0 — what an unclamped shared expert produces. 20.4x the
    /// clamped value.
    const SHARED_UNCLAMPED: f32 = 14.899_607;

    fn moe_fixture_vb(device: &Device) -> ShardedVarBuilder {
        use std::collections::HashMap;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        // Routed experts: all zero, so `y_routed == 0` and the assertion below
        // reads the shared expert alone.
        t.insert(
            "experts.gate_up_proj".to_string(),
            Tensor::zeros((2, 2, 4), DType::F32, device).unwrap(),
        );
        t.insert(
            "experts.down_proj".to_string(),
            Tensor::zeros((2, 2, 2), DType::F32, device).unwrap(),
        );
        // Shared expert: hidden = inter = 2, input all ones, so the gate
        // pre-activation is 2 * 2.5 = 5.0 and the up pre-activation 2 * 1.5 = 3.0.
        // `down_proj` is the identity, so the layer output IS the gated
        // activation.
        t.insert(
            "shared_experts.gate_proj.weight".to_string(),
            Tensor::from_vec(vec![2.5f32; 4], (2, 2), device).unwrap(),
        );
        t.insert(
            "shared_experts.up_proj.weight".to_string(),
            Tensor::from_vec(vec![1.5f32; 4], (2, 2), device).unwrap(),
        );
        t.insert(
            "shared_experts.down_proj.weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), device).unwrap(),
        );
        t.insert(
            "gate.weight".to_string(),
            Tensor::zeros((2, 2), DType::F32, device).unwrap(),
        );
        let backend: Box<dyn candle_nn::var_builder::SimpleBackend + 'static> = Box::new(t);
        mistralrs_quant::safetensors::ShardedSafeTensors::wrap(backend, DType::F32, device.clone())
    }

    fn build_moe_fixture(cfg: &DeepSeekV4Config) -> Moe {
        let device = Device::Cpu;
        let mapper = crate::device_map::DummyDeviceMapper {
            nm_device: device.clone(),
        };
        let comm = mapper.get_comm_for(0).unwrap();
        Moe::new(
            cfg,
            moe_fixture_vb(&device),
            &mapper,
            0,
            false,
            cfg.n_shared_experts,
            cfg.n_routed_experts.unwrap(),
            &comm,
            device,
        )
        .expect("Moe loads")
    }

    /// Highest-value MoE assertion: the shared expert's clamp is
    /// device-independent and unconditional, so this runs everywhere.
    ///
    /// The reference builds the shared expert as `DeepseekV2MLP(...,
    /// swiglu_limit=getattr(config, "swiglu_limit", None))`
    /// (`srt/models/deepseek_v2.py:613-622`) and applies it at `:318-323`;
    /// V4 Flash publishes `swiglu_limit: 10.0`, so the clamp is mandatory.
    /// SGLang disables an entire fusion optimization to keep it
    /// (`deepseek_v4.py:1266-1271`). Arc dropped it on every backend
    /// (audit finding 5).
    #[test]
    fn v4_shared_expert_swiglu_is_clamped() {
        let cfg: DeepSeekV4Config = serde_json::from_str(MOE_FIXTURE_JSON).unwrap();
        let moe = build_moe_fixture(&cfg);

        // Wiring: the config's limit reaches the shared expert at all.
        assert_eq!(
            moe.shared_experts.as_ref().unwrap().swiglu_limit(),
            Some(1.0),
            "config swiglu_limit not threaded into the shared expert"
        );

        let xs = Tensor::ones((1, 1, 2), DType::F32, &Device::Cpu).unwrap();
        let out = moe.forward(&xs, None).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // Relative tolerance: `MatMul::matmul` downcasts CPU matmuls to F16
        // (mistralrs-quant/src/lib.rs:595-599), so ~5e-4 relative error is
        // structural. The clamped/unclamped gap is 20.4x.
        for v in &vals {
            assert!(
                (v - SHARED_CLAMPED).abs() / SHARED_CLAMPED < 2e-3,
                "shared expert dropped the swiglu clamp: {vals:?} (clamped {SHARED_CLAMPED}, unclamped {SHARED_UNCLAMPED})"
            );
        }
    }

    /// The mirror: with no limit the same fixture must produce the unclamped
    /// value. Without this, a clamp hard-wired to 1.0 would pass the test
    /// above while breaking every model that is not V4.
    #[test]
    fn shared_expert_without_a_swiglu_limit_is_unclamped() {
        let cfg: DeepSeekV4Config = serde_json::from_str(MOE_FIXTURE_JSON).unwrap();
        let moe = build_moe_fixture(&cfg);
        let shared = moe.shared_experts.as_ref().unwrap().clone();
        let unlimited = Mlp::new(
            moe_fixture_vb(&Device::Cpu).pp("shared_experts"),
            2,
            2,
            &None,
            Activation::Silu,
            &crate::device_map::DummyDeviceMapper {
                nm_device: Device::Cpu,
            }
            .get_comm_for(0)
            .unwrap(),
        )
        .unwrap();
        assert_eq!(shared.swiglu_limit(), Some(1.0));
        assert_eq!(unlimited.swiglu_limit(), None);

        let xs = Tensor::ones((1, 2), DType::F32, &Device::Cpu).unwrap();
        let clamped: Vec<f32> = shared
            .forward(&xs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let raw: Vec<f32> = unlimited
            .forward(&xs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for v in &clamped {
            assert!(
                (v - SHARED_CLAMPED).abs() / SHARED_CLAMPED < 2e-3,
                "{clamped:?}"
            );
        }
        for v in &raw {
            assert!(
                (v - SHARED_UNCLAMPED).abs() / SHARED_UNCLAMPED < 2e-3,
                "unlimited Mlp changed: {raw:?}"
            );
        }
    }

    // ---------------------------------------------------------------------
    // Audit finding 6 — YaRN must not be applied to Standard (ratio-0) layers.
    // ---------------------------------------------------------------------

    /// The published `rope_scaling` and `compress_ratios` from
    /// `deepseek-ai/DeepSeek-V4-Flash`'s `config.json` (fetched 2026-08-14,
    /// see `memory/mission/wave13-AH-config.md`), on a table small enough to
    /// build in a unit test. `max_position_embeddings` is reduced from the
    /// shipped 1048576 — it only sets the table's row count, not the ramp.
    const V4_ROPE_JSON: &str = r#"{
        "vocab_size": 129280,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "max_position_embeddings": 16384,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "compress_rope_theta": 160000,
        "qk_rope_head_dim": 64,
        "head_dim": 512,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "type": "yarn"
        },
        "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
    }"#;

    /// Pins **which** layers get YaRN, so a config change or a refactor cannot
    /// silently re-enable it on the ratio-0 set.
    ///
    /// Reference `deepseek_v4.py:234-238` forces `original_seq_len = 0` when
    /// `compress_ratio == 0`, and `deepseek_v4_rope.py:47-53` then skips the
    /// interpolation entirely. For V4 Flash the ratio-0 set is exactly
    /// `{0, 1, 43}` — layers 0 and 1 plus slot 43, the MTP block.
    /// **Layer 42 has ratio 4 and is NOT in the set** (the audit text says
    /// "0, 1, 42"; that is an error, corrected by the published config).
    #[test]
    fn yarn_applies_to_compressed_layers_only() {
        let cfg: DeepSeekV4Config = serde_json::from_str(V4_ROPE_JSON).unwrap();
        assert_eq!(cfg.compress_ratios.len(), 44);
        assert!(
            cfg.rope_scaling.is_some(),
            "fixture must publish rope_scaling"
        );

        let unscaled: Vec<usize> = (0..44)
            .filter(|i| cfg.rope_scaling_for_layer(*i).is_none())
            .collect();
        assert_eq!(
            unscaled,
            vec![0, 1, 43],
            "the no-YaRN set must be exactly layers 0, 1 and the MTP slot (43)"
        );
        // Layer 42 is CSA (ratio 4), so it keeps YaRN — the audit's "0, 1, 42"
        // wording is wrong.
        assert_eq!(cfg.layer_compress_ratio(42), 4);
        assert!(cfg.rope_scaling_for_layer(42).is_some());

        for i in 0..44 {
            assert_eq!(
                cfg.rope_scaling_for_layer(i).is_some(),
                cfg.layer_compress_ratio(i) != 0,
                "layer {i} YaRN decision disagrees with its compress ratio"
            );
        }
        // Ratio histogram from the published config: 21 CSA, 20 HCA, 3 Standard.
        assert_eq!(cfg.compress_ratios.iter().filter(|r| **r == 4).count(), 21);
        assert_eq!(
            cfg.compress_ratios.iter().filter(|r| **r == 128).count(),
            20
        );
        assert_eq!(cfg.compress_ratios.iter().filter(|r| **r == 0).count(), 3);
    }

    /// The numerical half: the Standard-layer rotary table must be plain
    /// `1/θ^(2i/d)` — bit-comparable to an explicitly unscaled config — while
    /// the compressed table must still show the YaRN ramp.
    ///
    /// The mirror assertion matters as much as the first: a "fix" that
    /// disabled YaRN everywhere would pass the first check alone.
    #[test]
    fn standard_layer_rope_is_unscaled_and_compressed_is_not() -> Result<()> {
        let cfg: DeepSeekV4Config = serde_json::from_str(V4_ROPE_JSON).unwrap();
        let dev = Device::Cpu;
        let pos = cfg.max_position_embeddings - 1;
        let shape = (1usize, 1usize, 1usize, cfg.qk_rope_head_dim);
        let q = Tensor::ones(shape, DType::F32, &dev)?;

        let rope_of = |c: &DeepSeekV2RopeConfig| -> Result<Vec<f32>> {
            let r = DeepSeekV2RotaryEmbedding::new(c, DType::F32, &dev)?;
            let (out, _) = r.forward(&q, &q, &[pos])?;
            out.flatten_all()?.to_vec1::<f32>()
        };
        let max_diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };

        // Standard layers: what `DeepSeekV4::new` builds vs the reference's
        // `original_seq_len = 0` table (plain inverse frequencies, base 10000).
        let standard = rope_of(&cfg.standard_rope_config())?;
        let reference_unscaled = rope_of(&DeepSeekV2RopeConfig {
            rope_scaling: None,
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        })?;
        assert!(
            max_diff(&standard, &reference_unscaled) < 1e-6,
            "Standard-layer RoPE is YaRN-scaled (max diff {})",
            max_diff(&standard, &reference_unscaled)
        );

        // Compressed layers: YaRN must still bite, on the compress base.
        let compressed = rope_of(&cfg.compress_rope_config())?;
        let compressed_unscaled = rope_of(&DeepSeekV2RopeConfig {
            rope_scaling: None,
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.compress_rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        })?;
        assert!(
            max_diff(&compressed, &compressed_unscaled) > 1e-3,
            "compressed-layer RoPE lost its YaRN ramp (max diff {})",
            max_diff(&compressed, &compressed_unscaled)
        );
        Ok(())
    }

    /// The **MTP block** takes the Standard (no-YaRN) table too.
    ///
    /// PR #35 fixed the main model's ratio-0 layers and deliberately deferred
    /// the identical one-liner in `MtpBlock::try_new` to #30's owner; it sat
    /// unapplied, so the draft head rotated its queries on a YaRN-compressed
    /// table while the target it has to agree with rotated on an unscaled one.
    /// Speculative decode accepts on exact argmax equality, so a
    /// draft-vs-target position mismatch shows up as rejections and nothing
    /// else — an acceptance rate measured on it would have been measuring the
    /// mismatch, not the head.
    ///
    /// `MtpBlock::try_new` now builds its rotary from
    /// [`DeepSeekV4Config::standard_rope_config`]; this pins both halves of
    /// that: the MTP slot really is ratio-0, and the table that helper yields
    /// is the unscaled one. The mirror assertion (the pre-fix expression is
    /// *measurably different*) is what stops this passing vacuously on a
    /// config where YaRN happens to be a no-op.
    #[test]
    fn mtp_block_takes_the_standard_unscaled_rope_table() -> Result<()> {
        let cfg: DeepSeekV4Config = serde_json::from_str(V4_ROPE_JSON).unwrap();
        let mtp_slot = cfg.num_hidden_layers; // the MTP block's virtual index
        assert_eq!(
            cfg.layer_compress_ratio(mtp_slot),
            0,
            "the MTP slot must be Standard (COMPRESS_RATIO_NEXTN_LAYER = 0, \
             deepseek_v4_nextn.py:47)"
        );
        assert!(
            cfg.rope_scaling_for_layer(mtp_slot).is_none(),
            "the MTP slot must resolve to NO YaRN"
        );

        let dev = Device::Cpu;
        let pos = cfg.max_position_embeddings - 1;
        let shape = (1usize, 1usize, 1usize, cfg.qk_rope_head_dim);
        let q = Tensor::ones(shape, DType::F32, &dev)?;
        let rope_of = |c: &DeepSeekV2RopeConfig| -> Result<Vec<f32>> {
            let r = DeepSeekV2RotaryEmbedding::new(c, DType::F32, &dev)?;
            let (out, _) = r.forward(&q, &q, &[pos])?;
            out.flatten_all()?.to_vec1::<f32>()
        };
        let max_diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };

        // What `MtpBlock::try_new` builds today.
        let mtp = rope_of(&cfg.standard_rope_config())?;
        let reference_unscaled = rope_of(&DeepSeekV2RopeConfig {
            rope_scaling: None,
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        })?;
        assert!(
            max_diff(&mtp, &reference_unscaled) < 1e-6,
            "the MTP block's RoPE table is YaRN-scaled (max diff {})",
            max_diff(&mtp, &reference_unscaled)
        );

        // Mirror: the expression this replaced (`cfg.rope_scaling.clone()`)
        // really does produce a different table, so the assertion above has
        // something to catch.
        let pre_fix = rope_of(&DeepSeekV2RopeConfig {
            rope_scaling: cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        })?;
        assert!(
            max_diff(&mtp, &pre_fix) > 1e-3,
            "the pre-fix MTP RoPE table is indistinguishable from the fixed one \
             (max diff {}) — this test would pass either way",
            max_diff(&mtp, &pre_fix)
        );
        Ok(())
    }

    // ---------------------------------------------------------------------
    // wave30: the rolling compressed `xs` state.
    //
    // The whole memory win rests on one claim: `forward_from_xs` is a strided
    // LOCAL reduction, so a completed compressed row never changes and the raw
    // tokens behind it can be dropped. These tests execute that claim instead
    // of asserting it — feed the compressor one group at a time through
    // `XsRollingCache::advance` and require the result to equal the
    // whole-history recompute the model used to do on every decode step.
    // ---------------------------------------------------------------------

    /// A real `V4Compressor` with deterministic, non-degenerate weights.
    /// Zero weights would make every equivalence assertion below vacuous, so
    /// the fixture asserts its own output is non-trivial.
    fn rolling_test_compressor(
        ratio: usize,
        hidden: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<V4Compressor> {
        let coff = if ratio == 4 { 2 } else { 1 };
        let patterned = |shape: (usize, usize), phase: f32| -> Result<Tensor> {
            let n = shape.0 * shape.1;
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.31 + phase).sin() * 0.5)
                .collect();
            Tensor::from_vec(data, shape, device)
        };
        let wkv_gate: Arc<dyn QuantMethod> = Arc::new(mistralrs_quant::UnquantLinear::new(
            mistralrs_quant::QuantMethodConfig::Unquantized(candle_nn::Linear::new(
                patterned((2 * coff * head_dim, hidden), 0.0)?,
                None,
            )),
        )?);
        let norm = RmsNorm::from_w(
            patterned((1, head_dim), 1.0)?.reshape(head_dim)?.abs()?,
            1e-6,
        )?;
        Ok(V4Compressor {
            wkv_gate,
            norm,
            ape: patterned((ratio, coff * head_dim), 0.7)?,
            ratio,
            head_dim,
            coff,
            hidden_size: hidden,
        })
    }

    fn rolling_test_xs(t: usize, hidden: usize, device: &Device) -> Result<Tensor> {
        let data: Vec<f32> = (0..t * hidden)
            .map(|i| ((i as f32) * 0.017 + 0.3).cos() * 0.8)
            .collect();
        Tensor::from_vec(data, (1, t, hidden), device)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
    }

    /// Feeding the history one token at a time through the rolling state must
    /// reproduce, at every step, the compressed rows the whole-history
    /// recompute produces — for both compressor shapes: ratio 4 (`coff == 2`,
    /// `overlap_transform` folds in the PREVIOUS group, so a row spans two
    /// groups) and ratio 128 (`coff == 1`, one group per row).
    fn rolling_matches_full_recompute(ratio: usize, prefill: usize, steps: usize) -> Result<()> {
        let device = Device::Cpu;
        let hidden = 32;
        let head_dim = 16;
        let compressor = rolling_test_compressor(ratio, hidden, head_dim, &device)?;
        let span_groups = compressor.coff;
        let total = prefill + steps;
        let xs = rolling_test_xs(total, hidden, &device)?;

        let mut state = XsRollingCache::new(ratio, span_groups, head_dim, 4096);
        state.advance(&xs.narrow(1, 0, prefill)?, |w| {
            compressor.forward_from_xs(w)
        })?;

        let mut saw_rows = 0usize;
        for step in 0..steps {
            let tok = xs.narrow(1, prefill + step, 1)?;
            let got = state.advance(&tok, |w| compressor.forward_from_xs(w))?;
            let seen = prefill + step + 1;
            assert_eq!(state.current_seq_len(), seen);

            // Ground truth: compress the whole history from scratch, exactly
            // what `Attention::compress_prefix` does.
            let t_trunc = (seen / ratio) * ratio;
            if t_trunc == 0 {
                assert!(got.is_none(), "no group has completed yet at {seen} tokens");
                continue;
            }
            let want = compressor.forward_from_xs(&xs.narrow(1, 0, t_trunc)?)?;
            let got = got.expect("a completed group must produce rows");
            assert_eq!(
                got.dims(),
                want.dims(),
                "rolling rows have the wrong shape at {seen} tokens"
            );
            let diff = max_abs_diff(&got, &want)?;
            assert!(
                diff <= 1e-5,
                "rolling compressed rows diverged from the whole-history recompute at \
                 {seen} tokens (ratio {ratio}): max abs diff {diff}"
            );
            saw_rows = want.dim(1)?;
            // Non-degenerate: a zero fixture would make the equality vacuous.
            let mag = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(mag > 1e-3, "fixture produced ~zero compressed rows ({mag})");
        }
        assert!(
            saw_rows >= 2,
            "test did not cross two `ratio` strides (only {saw_rows} rows)"
        );
        Ok(())
    }

    #[test]
    fn rolling_xs_matches_full_recompute_csa_ratio4() -> Result<()> {
        // 10 prefill tokens then 14 decode steps: crosses the ratio-4 group
        // boundary five times, and the overlap window (2 groups) three times.
        rolling_matches_full_recompute(4, 10, 14)
    }

    #[test]
    fn rolling_xs_matches_full_recompute_hca_ratio128() -> Result<()> {
        // Two 128-token strides completed during single-token decode.
        rolling_matches_full_recompute(128, 100, 160)
    }

    /// The tail retained after each step must stay bounded — that IS the
    /// memory win. Without this, "rolling" would just be the old raw history
    /// under a new name.
    #[test]
    fn rolling_xs_tail_is_bounded_by_the_compressor_span() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_dim, ratio) = (32usize, 16usize, 4usize);
        let compressor = rolling_test_compressor(ratio, hidden, head_dim, &device)?;
        let mut state = XsRollingCache::new(ratio, compressor.coff, head_dim, 4096);
        let xs = rolling_test_xs(400, hidden, &device)?;
        state.advance(&xs.narrow(1, 0, 200)?, |w| compressor.forward_from_xs(w))?;
        let bound = compressor.coff * ratio + state.margin;
        for step in 0..200 {
            state.advance(&xs.narrow(1, 200 + step, 1)?, |w| {
                compressor.forward_from_xs(w)
            })?;
            let tail_rows = state.tail.as_ref().unwrap().dim(1)?;
            assert!(
                tail_rows <= bound,
                "raw tail grew to {tail_rows} rows at {} tokens (bound {bound}) — the rolling \
                 state is retaining history it can never need",
                state.current_seq_len()
            );
        }
        // ... and the compressed rows are the only thing that grows with T.
        assert_eq!(state.comp.current_seq_len(), 400 / ratio);
        Ok(())
    }

    // ---------------------------------------------------------------------
    // wave63-CO: per-row `xs` state.
    //
    // The claim these execute is the keystone one: a batched compressor cache
    // whose rows sit at DIFFERENT token counts advances each row exactly as if
    // that sequence had been run alone. Until this landed, `tokens`/`base` were
    // one number for the whole batch, so no sequence could advance its
    // compressed KV independently — the blocker PR #90 §6, PR #92 §5.1 and
    // wave29-BC §4b each arrived at from a different direction.
    // ---------------------------------------------------------------------

    /// Assemble one batched `XsRollingCache` from per-sequence ones exactly as
    /// `NormalCacheManager::clone_in_cache` does: the START-anchored compressed
    /// buffers are zero-extended to the widest capacity and stacked on dim 0,
    /// the END-anchored raw windows are **front**-padded to the widest and
    /// stacked, and the row lengths are the concatenation of the sequences'
    /// own — not `seqs[0]`'s repeated.
    fn batch_xs(per_seq: &[XsRollingCache]) -> Result<XsRollingCache> {
        let width = |t: &Tensor| t.dim(1);
        let cap = per_seq
            .iter()
            .map(|x| width(x.comp.all_data.as_ref().unwrap()))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap();
        let w = per_seq
            .iter()
            .map(|x| width(x.tail.as_ref().unwrap()))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap();
        let pad = |t: &Tensor, to: usize, at_front: bool| -> Result<Tensor> {
            let have = t.dim(1)?;
            if have == to {
                return t.contiguous();
            }
            let mut shape = t.dims().to_vec();
            shape[1] = to;
            let z = Tensor::zeros(shape, t.dtype(), t.device())?;
            z.slice_set(&t.contiguous()?, 1, if at_front { to - have } else { 0 })?;
            Ok(z)
        };
        let mut comps = Vec::new();
        let mut tails = Vec::new();
        for x in per_seq {
            comps.push(pad(x.comp.all_data.as_ref().unwrap(), cap, false)?);
            tails.push(pad(x.tail.as_ref().unwrap(), w, true)?);
        }
        let mut out = per_seq[0].clone();
        out.comp.all_data = Some(Tensor::cat(&comps, 0)?.contiguous()?);
        out.comp.capacity_seq_len = cap;
        out.tail = Some(Tensor::cat(&tails, 0)?.contiguous()?);
        let tokens: Vec<usize> = per_seq.iter().map(|x| x.row_lens().0[0]).collect();
        let base: Vec<usize> = per_seq.iter().map(|x| x.row_lens().1[0]).collect();
        out.comp.current_seq_len = tokens.iter().copied().max().unwrap() / out.ratio;
        out.set_row_lens(tokens, base)?;
        Ok(out)
    }

    /// Run `lens.len()` sequences to their own token counts on one stream,
    /// batch them, advance the batch by `t_new` of each row's OWN next tokens,
    /// and require every row to equal the same sequence advanced alone.
    fn ragged_batch_matches_b1(ratio: usize, lens: &[usize], t_new: usize) -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_dim) = (32usize, 16usize);
        let compressor = rolling_test_compressor(ratio, hidden, head_dim, &device)?;
        // A DIFFERENT stream per row. Sharing one stream would make every row's
        // compressed block `j` cover the identical tokens, so a bug that read
        // or wrote the wrong row would be invisible.
        let total = lens.iter().copied().max().unwrap() + t_new + 8;
        let streams = per_row_streams(lens.len(), total, hidden, &device)?;

        let mut refs: Vec<XsRollingCache> = Vec::new();
        for (i, &l) in lens.iter().enumerate() {
            let mut s = XsRollingCache::new(ratio, compressor.coff, head_dim, 4096);
            s.advance(&streams[i].narrow(1, 0, l)?, |w| {
                compressor.forward_from_xs(w)
            })?;
            refs.push(s);
        }
        let mut batched = batch_xs(&refs)?;
        assert_eq!(
            batched.rows_uniform(),
            lens.iter().all(|&l| l == lens[0]),
            "the fixture must exercise the path it claims to"
        );

        // Row `i` gets ITS OWN next tokens, which is what a real batch of
        // independent sequences hands the compressor.
        let parts = lens
            .iter()
            .enumerate()
            .map(|(i, &l)| streams[i].narrow(1, l, t_new))
            .collect::<Result<Vec<_>>>()?;
        let xs_new = Tensor::cat(&parts, 0)?;
        let got = batched
            .advance(&xs_new, |w| compressor.forward_from_xs(w))?
            .expect("a batch past one full group has compressed rows");

        for (i, &l) in lens.iter().enumerate() {
            let want = refs[i]
                .advance(&streams[i].narrow(1, l, t_new)?, |w| {
                    compressor.forward_from_xs(w)
                })?
                .expect("the reference is past one full group too");
            let live = (l + t_new) / ratio;
            assert!(live > 0, "fixture must complete at least one block");
            let mine = got.narrow(0, i, 1)?.narrow(1, 0, live)?;
            let theirs = want.narrow(1, 0, live)?;
            assert_eq!(
                max_abs_diff(&mine, &theirs)?,
                0.0,
                "row {i} (at {l} + {t_new} tokens) diverged from the same sequence run at B=1 \
                 over its {live} live compressed block(s)"
            );
            assert_eq!(
                (batched.row_lens().0[i], batched.row_lens().1[i]),
                (refs[i].row_lens().0[0], refs[i].row_lens().1[0]),
                "row {i}'s token count / resume point must equal the B=1 reference's"
            );
        }
        // Non-degenerate: distinct rows must actually hold distinct values, or
        // an implementation that broadcast one row everywhere would pass.
        if lens.len() > 1 {
            let a = got.narrow(0, 0, 1)?;
            let b = got.narrow(0, 1, 1)?;
            assert!(
                max_abs_diff(&a, &b)? > 1e-6,
                "the fixture's rows are indistinguishable, so this proves nothing"
            );
        }
        Ok(())
    }

    /// `n` deterministic, mutually distinct compressor-input streams.
    fn per_row_streams(n: usize, t: usize, hidden: usize, device: &Device) -> Result<Vec<Tensor>> {
        (0..n)
            .map(|row| {
                let data: Vec<f32> = (0..t * hidden)
                    .map(|i| ((i as f32) * 0.017 + 0.3 + row as f32 * 1.7).cos() * 0.8)
                    .collect();
                Tensor::from_vec(data, (1, t, hidden), device)
            })
            .collect()
    }

    /// 🔑 The keystone claim, on the CSA compressor (ratio 4, overlapping): four
    /// rows at four different residues mod `ratio` — the worst case, since the
    /// window geometry is a function of the residue — advance token-identically
    /// to four B=1 runs. Three of the four complete a new block at three
    /// different window offsets and land on the same absolute block; the fourth
    /// completes none and must be left untouched.
    #[test]
    fn batched_ragged_xs_is_token_identical_to_the_b1_reference_csa() -> Result<()> {
        ragged_batch_matches_b1(4, &[37, 38, 39, 40], 3)
    }

    /// 🔑 The case the `slot_mapping` indirection exists for: two rows that are
    /// at the SAME residue mod `ratio` — so they share one compressor call —
    /// but a whole block apart, so their outputs must land on **different**
    /// `comp` columns. Under MTP this is the common case, not the exotic one:
    /// the verify window is `depth + 1`, and a CSA layer's ratio is 4, so two
    /// sequences a full window apart share a residue on most steps.
    ///
    /// Without the per-row destination — with one shared append offset, which
    /// is all `SingleCache::append` can express — both rows would write the
    /// same column and one of them would read the other's history.
    #[test]
    fn two_rows_one_compressor_call_two_destinations() -> Result<()> {
        ragged_batch_matches_b1(4, &[39, 43, 40, 41], 1)
    }

    /// The same on HCA (ratio 128, non-overlapping), where a step's rows almost
    /// all complete NO block and the scatter must write only the one that does.
    #[test]
    fn batched_ragged_xs_is_token_identical_to_the_b1_reference_hca() -> Result<()> {
        ragged_batch_matches_b1(128, &[1022, 1023, 1024, 1025], 3)
    }

    /// 🔑 The case the H200 died on, and the one every fixture above misses:
    /// a row whose ENTIRE history is shorter than the batched window.
    ///
    /// The window a batch allocates is its greediest row's retained run — 23
    /// columns for these token counts — so the 9-token row is front-padded by
    /// 14 columns that stand for tokens before token 0. Every fixture above
    /// uses rows long enough (37..43, 1022..1025, 30..35) that this never
    /// happens, which is why they stayed green while `ARC_V4_XS_PER_SEQ=1` at
    /// B=8 returned one token and `finish_reason: None` for every request:
    /// `row 3 holds 9 tokens, fewer than the 11-wide retained window`.
    ///
    /// The trigger is prompt-length diversity, not generation length — so this
    /// fails harder under real heterogeneous arrivals than under any
    /// uniform-prompt benchmark.
    #[test]
    fn a_row_shorter_than_the_batched_window_is_token_identical_csa() -> Result<()> {
        // Retained runs 9 / 22 / 23 / 20 → a 23-wide window; three distinct
        // residues mod 4, so three compressor calls, and row 3 completes no
        // block at all.
        ragged_batch_matches_b1(4, &[9, 22, 39, 40], 3)
    }

    /// The same on HCA, where `ratio` is 128 and the retained run can reach
    /// `ratio + margin`: a 129-token row sits inside a 143-wide window.
    #[test]
    fn a_row_shorter_than_the_batched_window_is_token_identical_hca() -> Result<()> {
        ragged_batch_matches_b1(128, &[129, 143, 1024, 1150], 3)
    }

    /// 🔑 Settles the one discrepancy between two independent models of this
    /// buffer, and does it without a GPU.
    ///
    /// The ArcGraph chain measured the reallocation cycling through
    /// `4096 x {18, 19, 20, 21}`. Running the retention rule forward predicts
    /// `{20, 21, 22, 23}`. Both are `ratio`-consecutive, both contain 21, and
    /// they are offset by exactly 2 — so one of the two models is wrong about
    /// *why*, even though the bound holds either way.
    ///
    /// The pre-committed reconciliation was: theirs is the pre-saturation RAMP
    /// (their run generated ~21 tokens, so `base` had not finished advancing
    /// and `W = tokens - base` was still climbing), mine is the steady state.
    /// This checks that claim directly by recording the width from token 1,
    /// and it is the difference between a bound that is right for the right
    /// reason and one that happens to be right — which is the trap the next
    /// context length springs.
    #[test]
    fn the_window_ramps_then_settles_to_ratio_consecutive_sizes() -> Result<()> {
        xs_rolling::pin_test_override::with(false, || {
            let device = Device::Cpu;
            let (hidden, head_dim, ratio) = (32usize, 16usize, 4usize);
            let compressor = rolling_test_compressor(ratio, hidden, head_dim, &device)?;
            let stream = per_row_streams(1, 130, hidden, &device)?.remove(0);
            let mut s = XsRollingCache::new(ratio, compressor.coff, head_dim, 4096);

            let mut widths = Vec::new();
            for i in 0..128 {
                s.advance(&stream.narrow(1, i, 1)?, |w| compressor.forward_from_xs(w))?;
                widths.push(s.tail_width()?);
            }
            let cap = ratio * compressor.coff + crate::kv_cache::XS_TAIL_MARGIN_TOKENS;

            // The ramp: early widths are BELOW the steady-state band, because
            // `base` is still pinned at 0 while `tokens` climbs.
            let ramp: std::collections::BTreeSet<_> = widths[..20].iter().copied().collect();
            let steady: std::collections::BTreeSet<_> = widths[64..].iter().copied().collect();
            assert!(
                ramp.iter().min() < steady.iter().min(),
                "the early widths must be a ramp, not the steady band: ramp {ramp:?} \
                 steady {steady:?}"
            );
            // The steady state is exactly `ratio` consecutive sizes…
            assert_eq!(
                steady.len(),
                ratio,
                "steady-state width must cycle through exactly `ratio` sizes, got {steady:?}"
            );
            let lo = *steady.iter().next().unwrap();
            assert_eq!(
                steady.iter().copied().collect::<Vec<_>>(),
                (lo..lo + ratio).collect::<Vec<_>>(),
                "the steady band must be consecutive, got {steady:?}"
            );
            // …and it is the top of the range the bound allows: [cap-ratio, cap).
            assert_eq!(
                (lo, lo + ratio - 1),
                (cap - ratio, cap - 1),
                "steady band should be [cap-ratio, cap-1] = [{}, {}]",
                cap - ratio,
                cap - 1
            );
            // The bound, over every step including the ramp.
            assert!(
                widths.iter().all(|&w| w < cap),
                "some width reached the pinned capacity {cap}, so pinning there could truncate: \
                 max was {:?}",
                widths.iter().max()
            );
            Ok(())
        })
    }

    /// 🔑 Pinning the retained window must change the ALLOCATION and nothing
    /// else. This is the evidence that the pin is SAFE — the two settings are
    /// run against the same stream for 40 steps and required to agree EXACTLY,
    /// on the compressed rows and on both time bases.
    ///
    /// It was once also offered as the evidence for defaulting
    /// `ARC_V4_XS_PIN_WINDOW` on without a throughput number. It is not that:
    /// it proves the pin changes no answer, which is necessary and not
    /// sufficient. The flag is opt-in until the throughput number exists on
    /// this tree — see `xs_rolling::xs_pin_window_enabled_from`'s FLIP
    /// CONDITION for the experiment that settles it.
    ///
    /// The argument for why they must — every offset is derived from the row's
    /// own token count, so a wider buffer shifts `off` by exactly the widening
    /// and the compressor sees the same absolute tokens — is the kind of
    /// argument that has been wrong on this chain before. So it is checked.
    #[test]
    fn pinning_the_window_is_numerically_inert() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_dim, ratio) = (32usize, 16usize, 4usize);
        let compressor = rolling_test_compressor(ratio, hidden, head_dim, &device)?;
        let stream = per_row_streams(1, 220, hidden, &device)?.remove(0);

        let run = |pinned: bool| -> Result<(Tensor, Vec<usize>, Vec<usize>, Vec<usize>)> {
            xs_rolling::pin_test_override::with(pinned, || {
                let mut s = XsRollingCache::new(ratio, compressor.coff, head_dim, 4096);
                s.advance(&stream.narrow(1, 0, 37)?, |w| compressor.forward_from_xs(w))?;
                let mut widths = Vec::new();
                for i in 0..40 {
                    s.advance(&stream.narrow(1, 37 + i, 1)?, |w| {
                        compressor.forward_from_xs(w)
                    })?;
                    widths.push(s.tail_width()?);
                }
                let comp = s.compressed_rows()?.expect("past one full group");
                let (tok, base) = s.row_lens();
                Ok((comp, tok.to_vec(), base.to_vec(), widths))
            })
        };

        let (comp_off, tok_off, base_off, widths_off) = run(false)?;
        let (comp_on, tok_on, base_on, widths_on) = run(true)?;

        assert_eq!(
            max_abs_diff(&comp_off, &comp_on)?,
            0.0,
            "pinning the window changed the compressed rows — it must only change the allocation"
        );
        assert_eq!(tok_off, tok_on, "token counts diverged");
        assert_eq!(base_off, base_on, "resume points diverged");

        // …and the fixture must actually exercise the difference, or the
        // equality above proves nothing (five fixtures on this chain have
        // survived a mutation by not distinguishing the two answers).
        let distinct_off: std::collections::BTreeSet<_> = widths_off.iter().collect();
        assert!(
            distinct_off.len() > 1,
            "unpinned widths did not vary ({widths_off:?}), so this proves nothing about pinning"
        );
        assert_eq!(
            distinct_off.len(),
            ratio,
            "unpinned width should cycle through `ratio` consecutive sizes, got {distinct_off:?}"
        );
        let cap = ratio * compressor.coff + crate::kv_cache::XS_TAIL_MARGIN_TOKENS;
        assert!(
            widths_on.iter().all(|&w| w == cap),
            "pinned widths must be the constant {cap}, got {widths_on:?}"
        );
        // The bound the pin rests on: the unpinned width never reaches it.
        assert!(
            widths_off.iter().all(|&w| w < cap),
            "the pinned capacity {cap} must exceed every width the retention rule produces, \
             got {widths_off:?}"
        );
        Ok(())
    }

    /// The control the ragged tests need: a UNIFORM batch takes the untouched
    /// scalar path and is also token-identical. If this ever failed, the
    /// "flag off is byte-identical" claim would be false and the ragged
    /// result above would be unattributable.
    #[test]
    fn a_uniform_batch_still_matches_the_b1_reference() -> Result<()> {
        ragged_batch_matches_b1(4, &[40, 40, 40, 40], 3)
    }

    /// One step is not enough: a per-row `base` that drifts by one token per
    /// step would still look right on step 1. Run the ragged batch and the B=1
    /// references in lockstep for 24 steps, re-batching each time — which is
    /// what the engine's `clone_in`/`clone_out` pair does — and require exact
    /// agreement at every step.
    #[test]
    fn a_ragged_batch_stays_token_identical_across_many_steps() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_dim, ratio) = (32usize, 16usize, 4usize);
        let compressor = rolling_test_compressor(ratio, hidden, head_dim, &device)?;
        // Rows 1 and 3 sit at the same residue mod `ratio` a block apart, so on
        // every fourth step they share one compressor call and must land on two
        // different `comp` columns.
        let mut lens = [30usize, 31, 33, 35];
        let streams = per_row_streams(lens.len(), 200, hidden, &device)?;

        let mut refs: Vec<XsRollingCache> = Vec::new();
        for (i, &l) in lens.iter().enumerate() {
            let mut s = XsRollingCache::new(ratio, compressor.coff, head_dim, 4096);
            s.advance(&streams[i].narrow(1, 0, l)?, |w| {
                compressor.forward_from_xs(w)
            })?;
            refs.push(s);
        }

        for step in 0..24 {
            let mut batched = batch_xs(&refs)?;
            assert!(!batched.rows_uniform(), "step {step} lost its raggedness");
            let parts = lens
                .iter()
                .enumerate()
                .map(|(i, &l)| streams[i].narrow(1, l, 1))
                .collect::<Result<Vec<_>>>()?;
            let got = batched
                .advance(&Tensor::cat(&parts, 0)?, |w| compressor.forward_from_xs(w))?
                .unwrap();
            for (i, l) in lens.iter_mut().enumerate() {
                let want = refs[i]
                    .advance(&streams[i].narrow(1, *l, 1)?, |w| {
                        compressor.forward_from_xs(w)
                    })?
                    .unwrap();
                *l += 1;
                let live = *l / ratio;
                assert_eq!(
                    max_abs_diff(
                        &got.narrow(0, i, 1)?.narrow(1, 0, live)?,
                        &want.narrow(1, 0, live)?
                    )?,
                    0.0,
                    "step {step}, row {i}: the batched compressor history left the B=1 reference"
                );
                assert_eq!(batched.row_lens().0[i], refs[i].row_lens().0[0]);
                assert_eq!(batched.row_lens().1[i], refs[i].row_lens().1[0]);
            }
            // The retained window must stay bounded even though it is now the
            // widest row's — otherwise the ragged path quietly reintroduces the
            // unbounded raw history this cache exists to delete.
            let bound = compressor.coff * ratio + batched.margin + (lens[3] - lens[0]);
            assert!(
                batched.tail.as_ref().unwrap().dim(1)? <= bound,
                "step {step}: the shared window grew past {bound}"
            );
        }
        Ok(())
    }

    /// 🔑 The MTP path in full: a sequence that has been rolled back keeps its
    /// `base` where it was (`set_len` moves `tokens`, never `base`), so when it
    /// is batched next to a neighbour the retention floor computed from its new
    /// token count sits BELOW what it actually holds. Clamping to `base` is
    /// what keeps it honest — without it the row would come out of the batch
    /// claiming to be resumable from further back than it is, and the rollback
    /// it should later refuse would succeed and resume the compressor from a
    /// gap. That is the exact failure this cache exists to make impossible, and
    /// it is invisible to any single-step or rollback-free fixture.
    #[test]
    fn a_rolled_back_row_keeps_its_own_resume_point_through_a_batch() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_dim, ratio) = (32usize, 16usize, 4usize);
        let compressor = rolling_test_compressor(ratio, hidden, head_dim, &device)?;
        let streams = per_row_streams(2, 64, hidden, &device)?;

        let mut rolled = XsRollingCache::new(ratio, compressor.coff, head_dim, 4096);
        rolled.advance(&streams[0].narrow(1, 0, 40)?, |w| {
            compressor.forward_from_xs(w)
        })?;
        let base_at_40 = rolled.row_lens().1[0];
        rolled.set_len(36)?;
        assert_eq!(
            rolled.row_lens().1[0],
            base_at_40,
            "the fixture needs `set_len` to leave `base` above the natural floor"
        );

        let mut other = XsRollingCache::new(ratio, compressor.coff, head_dim, 4096);
        other.advance(&streams[1].narrow(1, 0, 38)?, |w| {
            compressor.forward_from_xs(w)
        })?;

        let mut refs = vec![rolled, other];
        let mut batched = batch_xs(&refs)?;
        let parts = [streams[0].narrow(1, 36, 2)?, streams[1].narrow(1, 38, 2)?];
        let got = batched
            .advance(&Tensor::cat(&parts, 0)?, |w| compressor.forward_from_xs(w))?
            .unwrap();

        for (i, (start, stream)) in [(36usize, &streams[0]), (38, &streams[1])]
            .into_iter()
            .enumerate()
        {
            let want = refs[i]
                .advance(&stream.narrow(1, start, 2)?, |w| {
                    compressor.forward_from_xs(w)
                })?
                .unwrap();
            let live = (start + 2) / ratio;
            assert_eq!(
                max_abs_diff(
                    &got.narrow(0, i, 1)?.narrow(1, 0, live)?,
                    &want.narrow(1, 0, live)?
                )?,
                0.0,
                "row {i} diverged from the B=1 reference after a rollback"
            );
            assert_eq!(
                batched.row_lens().1[i],
                refs[i].row_lens().1[0],
                "row {i}'s resume point must not be loosened by being batched — a lower `base` \
                 lets a rollback succeed that would resume the compressor from a gap"
            );
        }
        Ok(())
    }

    /// The inverse of `batch_xs`: splitting a batched row back out must restore
    /// the per-sequence invariant `window width == tokens - base`, taking that
    /// row's share from the END of the shared window. Taking it from the front
    /// would hand the row somebody else's older tokens under its own `base`.
    /// ⚠️ Run with the window pin OFF, deliberately. Its premise is a row
    /// *narrower* than the shared window, which is what a resizing buffer
    /// produces; pinned, every row is the same width and the premise — and the
    /// re-anchoring it checks — cannot be constructed. The re-anchoring still
    /// has to be right when the pin is off, and
    /// `splitting_a_pinned_row_keeps_the_buffer_and_the_resume_point` covers
    /// the pinned side, where the requirement is the opposite one.
    #[test]
    fn splitting_a_batched_row_restores_the_per_sequence_window() -> Result<()> {
        xs_rolling::pin_test_override::with(false, splitting_a_batched_row_inner)
    }

    /// 🔑 The pinned counterpart, and the reason it matters is cost, not
    /// correctness: `clone_out_cache` calls `split_row` once per layer per
    /// sequence on EVERY engine step. A split that re-narrowed to
    /// `tokens - base` would reallocate the buffer every step and undo the pin
    /// precisely on the hot path it exists for. So the split must hand back the
    /// SAME tensor, and must not move the row's resume point while doing it.
    #[test]
    fn splitting_a_pinned_row_keeps_the_buffer_and_the_resume_point() -> Result<()> {
        xs_rolling::pin_test_override::with(true, || {
            let device = Device::Cpu;
            let (hidden, head_dim, ratio) = (32usize, 16usize, 4usize);
            let compressor = rolling_test_compressor(ratio, hidden, head_dim, &device)?;
            let lens = [37usize, 40];
            let streams = per_row_streams(2, 64, hidden, &device)?;
            let mut refs: Vec<XsRollingCache> = Vec::new();
            for (i, &l) in lens.iter().enumerate() {
                let mut s = XsRollingCache::new(ratio, compressor.coff, head_dim, 4096);
                s.advance(&streams[i].narrow(1, 0, l)?, |w| {
                    compressor.forward_from_xs(w)
                })?;
                refs.push(s);
            }
            let cap = ratio * compressor.coff + crate::kv_cache::XS_TAIL_MARGIN_TOKENS;
            for r in &refs {
                assert_eq!(
                    r.tail.as_ref().unwrap().dim(1)?,
                    cap,
                    "a pinned per-sequence window must already be the capacity"
                );
            }
            let batched = batch_xs(&refs)?;
            let shared = batched.tail.as_ref().unwrap().dim(1)?;
            assert_eq!(shared, cap, "pinned rows batch without any padding at all");

            let comps = batched.comp.all_data.as_ref().unwrap().chunk(2, 0)?;
            let tails = batched.tail.as_ref().unwrap().chunk(2, 0)?;
            for i in 0..2 {
                let out = batched.split_row(i, comps[i].clone(), tails[i].clone())?;
                assert_eq!(
                    out.tail.as_ref().unwrap().dim(1)?,
                    cap,
                    "row {i}: the split re-narrowed a pinned buffer, which would reallocate it \
                     on every engine step"
                );
                assert_eq!(
                    (out.row_lens().0[0], out.row_lens().1[0]),
                    (refs[i].row_lens().0[0], refs[i].row_lens().1[0]),
                    "row {i}: keeping the buffer must not move the resume point"
                );
            }
            Ok(())
        })
    }

    fn splitting_a_batched_row_inner() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_dim, ratio) = (32usize, 16usize, 4usize);
        let compressor = rolling_test_compressor(ratio, hidden, head_dim, &device)?;
        let lens = [37usize, 40];
        let streams = per_row_streams(2, 64, hidden, &device)?;
        let mut refs: Vec<XsRollingCache> = Vec::new();
        for (i, &l) in lens.iter().enumerate() {
            let mut s = XsRollingCache::new(ratio, compressor.coff, head_dim, 4096);
            s.advance(&streams[i].narrow(1, 0, l)?, |w| {
                compressor.forward_from_xs(w)
            })?;
            refs.push(s);
        }
        let batched = batch_xs(&refs)?;
        let shared = batched.tail.as_ref().unwrap().dim(1)?;
        assert!(
            refs.iter()
                .any(|r| r.tail.as_ref().unwrap().dim(1).unwrap() < shared),
            "the fixture must have a row narrower than the shared window"
        );

        let comps = batched.comp.all_data.as_ref().unwrap().chunk(2, 0)?;
        let tails = batched.tail.as_ref().unwrap().chunk(2, 0)?;
        for i in 0..2 {
            let out = batched.split_row(i, comps[i].clone(), tails[i].clone())?;
            let (tok, base) = (out.row_lens().0[0], out.row_lens().1[0]);
            assert_eq!(
                (tok, base),
                (refs[i].row_lens().0[0], refs[i].row_lens().1[0])
            );
            assert_eq!(
                out.tail.as_ref().unwrap().dim(1)?,
                tok - base,
                "a per-sequence cache's window is exactly `tokens - base` wide"
            );
            assert_eq!(
                max_abs_diff(out.tail.as_ref().unwrap(), refs[i].tail.as_ref().unwrap())?,
                0.0,
                "row {i}'s retained raw tokens must be its own, taken from the END of the \
                 shared window"
            );
            assert_eq!(out.comp.current_seq_len(), refs[i].comp.current_seq_len());
        }
        Ok(())
    }

    /// A rollback of up to `margin` tokens must be accepted and resumable at
    /// EVERY length — including the ones that cross a group boundary, where
    /// the rolled-back group's compressed row has to be rebuilt from the start
    /// of its group. This is the MTP verify / speculative rejection contract:
    /// unlike the prefix cacher, those callers cannot decline, they error.
    #[test]
    fn rolling_xs_accepts_any_rollback_within_the_margin() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_dim) = (32usize, 16usize);
        // ratio 128 is the case that bites: with `span_groups == 1` the naive
        // "keep the last span*ratio + margin tokens" rule retains only 16
        // tokens before the boundary, but resuming inside the previous group
        // needs all of it.
        for ratio in [4usize, 128] {
            let compressor = rolling_test_compressor(ratio, hidden, head_dim, &device)?;
            let span = compressor.coff;
            let xs = rolling_test_xs(300, hidden, &device)?;
            for tokens in [ratio, ratio + 1, 2 * ratio, 2 * ratio + 3, 300] {
                let mut state = XsRollingCache::new(ratio, span, head_dim, 4096);
                state.advance(&xs.narrow(1, 0, tokens)?, |w| {
                    compressor.forward_from_xs(w)
                })?;
                let drop = state.margin.min(tokens);
                let len = tokens - drop;
                state.set_len(len).map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "ratio {ratio}: rolling back {drop} tokens from {tokens} (the MTP \
                         rollback bound) must be accepted, got: {e}"
                    ))
                })?;
                // Re-feed to the original length and require the whole-history
                // answer: a rollback that "succeeds" but resumes from a gap is
                // exactly the silent corruption this is guarding.
                for i in len..tokens {
                    state.advance(&xs.narrow(1, i, 1)?, |w| compressor.forward_from_xs(w))?;
                }
                let t_trunc = (tokens / ratio) * ratio;
                let want = compressor.forward_from_xs(&xs.narrow(1, 0, t_trunc)?)?;
                let got = state.compressed_rows()?.expect("rows after resume");
                assert!(
                    max_abs_diff(&got, &want)? <= 1e-5,
                    "ratio {ratio}: resuming after a {drop}-token rollback at {tokens} tokens \
                     did not reproduce the whole-history recompute"
                );
                let tail_rows = state.tail.as_ref().unwrap().dim(1)?;
                assert!(
                    tail_rows <= span * ratio + state.margin,
                    "ratio {ratio}: rollback support cost more tail than the bound \
                     ({tail_rows} rows)"
                );
            }
        }
        Ok(())
    }

    /// Token-unit truncation (prefix cacher / MTP verify rollback /
    /// speculative rejection) must land on both time bases, and a rollback
    /// behind the retained raw window must be REFUSED, never silently
    /// resumed from a gap.
    #[test]
    fn rolling_xs_set_len_truncates_both_time_bases() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_dim, ratio) = (32usize, 16usize, 4usize);
        let compressor = rolling_test_compressor(ratio, hidden, head_dim, &device)?;
        let xs = rolling_test_xs(64, hidden, &device)?;
        let feed = |state: &mut XsRollingCache, from: usize, to: usize| -> Result<()> {
            for i in from..to {
                state.advance(&xs.narrow(1, i, 1)?, |w| compressor.forward_from_xs(w))?;
            }
            Ok(())
        };

        let mut state = XsRollingCache::new(ratio, compressor.coff, head_dim, 4096);
        feed(&mut state, 0, 50)?;
        assert_eq!(state.comp.current_seq_len(), 12);

        // A short rollback (the MTP/speculative case) is inside the margin.
        state.set_len(47)?;
        assert_eq!(state.current_seq_len(), 47);
        assert_eq!(state.comp.current_seq_len(), 47 / ratio);
        // Resuming after the rollback must reproduce a from-scratch run of
        // the same 50 tokens: the compressor sees no gap.
        feed(&mut state, 47, 50)?;
        let want = compressor.forward_from_xs(&xs.narrow(1, 0, 48)?)?;
        let got = state.compressed_rows()?.expect("rows after resume");
        assert!(
            max_abs_diff(&got, &want)? <= 1e-5,
            "resuming after a truncation did not reproduce the whole-history recompute"
        );

        // A rollback that lands inside the retained tail but leaves the NEXT
        // compressed row unbuildable (the `coff == 2` predecessor group is
        // gone) must also be refused — `set_len` succeeding here would push
        // the failure into `advance`, one step later and further from the
        // cause. At 50 tokens the tail starts at token `(12-1)*4 - 16 = 28`,
        // so 30 is inside it but resuming would need tokens from 24.
        let inside_tail_but_unresumable = 30;
        assert!(inside_tail_but_unresumable >= state.resumable_from());
        assert!(
            state.try_set_len(inside_tail_but_unresumable).is_err(),
            "a rollback past the compressor's own span must be refused even when the raw tail \
             still covers it"
        );

        // A rollback behind the retained window is refused, by both the
        // check-only and the mutating entry point, and leaves the state intact.
        let too_far = state.resumable_from().saturating_sub(1);
        assert!(
            state.try_set_len(too_far).is_err(),
            "a rollback to {too_far} tokens is behind base {} and must be refused",
            state.resumable_from()
        );
        assert!(state.set_len(too_far).is_err());
        assert_eq!(
            state.current_seq_len(),
            50,
            "a refused rollback must not mutate"
        );
        Ok(())
    }

    // ---------------------------------------------------------------------
    // wave42: the MTP verify rollback, against the cache V4 actually has.
    //
    // The tests above prove `XsRollingCache` honours a truncation *when it is
    // handed one*. What they do not touch is the caller that has to hand it
    // one: the MTP rejection path truncates the whole `NormalCache` by a
    // single `n_drop`, and in V4 that Vec is not homogeneous — 43 K/V entries
    // followed by one `KvCache::XsRolling` per compressed layer, each of which
    // maps the token-unit drop onto two different time bases and is the only
    // entry that can *refuse*.
    //
    // A refusal there is a hard error, not a fallback: unlike the prefix
    // cacher, MTP cannot decline and re-prefill mid-decode. That is what makes
    // this path worth its own test.
    // ---------------------------------------------------------------------

    /// A V4-shaped cache: `n_kv` K/V entries, then one `XsRolling` entry per
    /// `(ratio, span_groups)` — the layout `DeepSeekV4::new` builds.
    fn v4_shaped_cache(
        n_kv: usize,
        compressed: &[(usize, usize)],
        head_dim: usize,
        max_pos: usize,
    ) -> EitherCache {
        // `new_plain`: see the note on the other V4 cache constructor — the
        // V-marker layout requires `KvCache::Normal` slots.
        let cache = NormalCache::new_plain(n_kv, max_pos);
        {
            let mut guard = cache.lock().unwrap();
            for &(ratio, span_groups) in compressed {
                guard.0.push(KvCache::XsRolling(Box::new(XsRollingCache::new(
                    ratio, span_groups, head_dim, max_pos,
                ))));
            }
        }
        EitherCache::Normal(cache)
    }

    /// Append `t` positions of K/V to every `KvCache::Normal` entry, and `t`
    /// tokens of compressor input to every `XsRolling` entry — one forward's
    /// worth, exactly as a decode/verify step would.
    fn feed_v4_cache(
        cache: &EitherCache,
        xs: &Tensor,
        from: usize,
        t: usize,
        compressor: &V4Compressor,
    ) -> Result<()> {
        let EitherCache::Normal(normal) = cache else {
            unreachable!("fixture builds a Normal cache")
        };
        let mut guard = normal.lock().unwrap();
        for entry in &mut *guard.0 {
            match entry {
                KvCache::XsRolling(state) => {
                    let slice = xs.narrow(1, from, t)?;
                    state.advance(&slice, |w| compressor.forward_from_xs(w))?;
                }
                _ => {
                    let kv = Tensor::zeros((1, 1, t, 4), DType::F32, xs.device())?;
                    entry.append(&kv, &kv)?;
                }
            }
        }
        Ok(())
    }

    fn cache_lens(cache: &EitherCache) -> Vec<usize> {
        let EitherCache::Normal(normal) = cache else {
            unreachable!()
        };
        let guard = normal.lock().unwrap();
        guard.0.iter().map(|c| c.current_seq_len()).collect()
    }

    fn compressed_rows_of(cache: &EitherCache, idx: usize) -> Result<Option<Tensor>> {
        let EitherCache::Normal(normal) = cache else {
            unreachable!()
        };
        let guard = normal.lock().unwrap();
        match &guard.0[idx] {
            KvCache::XsRolling(state) => state.compressed_rows(),
            _ => unreachable!("entry {idx} is not an XsRolling cache"),
        }
    }

    /// **A rejected MTP draft must restore the compressor rows exactly.**
    ///
    /// Drives the production rollback — `verify_proposed` →
    /// `n_cache_positions_to_drop` → `truncate_cache_by` — over a V4-shaped
    /// mixed cache, at every draft depth clap accepts (`--mtp-depth 0..=8`)
    /// and every accept/reject shape, and requires after each one that:
    ///
    /// 1. the rollback is *accepted* (a refusal here is the hard-error class
    ///    `XS_TAIL_MARGIN_TOKENS` exists to prevent),
    /// 2. every entry — K/V and compressed alike — lands on the same length,
    ///    and that length is the plain-decode invariant `committed - 1`,
    /// 3. the surviving compressed rows are bit-comparable to a from-scratch
    ///    compress over the committed prefix, and
    /// 4. decoding *onward* from the rolled-back state keeps matching the
    ///    from-scratch reference — i.e. the raw tail was restored too, not
    ///    just the row count.
    ///
    /// Check 4 is the one with teeth. A rollback that trims `comp` but leaves
    /// `tail`/`base` inconsistent passes 1-3 and then silently compresses a gap
    /// on the next completed group, which is a wrong distant-context branch and
    /// not an error anywhere.
    #[test]
    fn mtp_verify_rollback_restores_the_compressor_rows_exactly() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_dim) = (32usize, 16usize);
        let max_pos = 4096usize;
        // Both V4 compressor shapes: ratio 4 (`coff == 2`, a row spans two
        // groups) and ratio 128 (`coff == 1`). 128 is the case the naive
        // retention rule got wrong.
        let c4 = rolling_test_compressor(4, hidden, head_dim, &device)?;
        let c128 = rolling_test_compressor(128, hidden, head_dim, &device)?;
        let xs = rolling_test_xs(1024, hidden, &device)?;

        // (proposed, verifier) covering all-accept, mid-reject, first-reject
        // and the degenerate short-verifier case.
        let shapes: &[(&[u32], &[u32])] = &[
            (&[1], &[1]),
            (&[1], &[9]),
            (&[1, 2], &[1, 2]),
            (&[1, 2], &[1, 9]),
            (&[1, 2], &[9, 2]),
            (&[1, 2, 3, 4], &[1, 2, 9, 4]),
            (&[1, 2, 3, 4, 5, 6, 7, 8], &[1, 2, 3, 4, 5, 6, 7, 8]),
            (&[1, 2, 3, 4, 5, 6, 7, 8], &[9, 2, 3, 4, 5, 6, 7, 8]),
            (&[1, 2, 3, 4, 5, 6, 7, 8], &[1, 2, 3, 9, 5, 6, 7, 8]),
        ];

        let mut saw_a_real_truncation = false;
        let mut saw_a_row_drop = false;

        for (ratio, compressor) in [(4usize, &c4), (128usize, &c128)] {
            // Prefill lengths chosen so `L + depth` straddles a group boundary
            // — the rollback then has to rebuild a compressed row from the
            // START of its group, which is the case a "keep the last
            // span*ratio + margin tokens" rule gets wrong.
            let prefills: &[usize] = if ratio == 4 {
                &[13, 15, 16, 20, 61]
            } else {
                &[130, 250, 255, 256, 383]
            };
            for &committed in prefills {
                // Every prefill must already have completed a group, or the
                // "surviving rows" assertion below has no rows to compare and
                // the case proves nothing.
                assert!(
                    committed > ratio,
                    "ratio {ratio}: prefill {committed} completes no compressed group"
                );
                for (proposed, verifier) in shapes {
                    let depth = proposed.len();
                    let result = crate::pipeline::verify_proposed(proposed, verifier);
                    let n_drop = crate::pipeline::n_cache_positions_to_drop(depth, &result);
                    let commit_len = result.commit_len();

                    let cache = v4_shaped_cache(2, &[(ratio, compressor.coff)], head_dim, max_pos);
                    let xs_idx = 2; // after the two K/V entries

                    // Prefill: the cache holds every committed token but the
                    // last (the plain-decode invariant `step()` enters on).
                    feed_v4_cache(&cache, &xs, 0, committed - 1, compressor)?;
                    // T0 forward (+1), then the verify forward (+depth).
                    feed_v4_cache(&cache, &xs, committed - 1, 1, compressor)?;
                    feed_v4_cache(&cache, &xs, committed, depth, compressor)?;
                    assert_eq!(
                        cache_lens(&cache),
                        vec![committed + depth; 3],
                        "fixture desync before the rollback"
                    );
                    let rows_before = compressed_rows_of(&cache, xs_idx)?
                        .map(|r| r.dim(1))
                        .transpose()?
                        .unwrap_or(0);

                    // ---- the production rollback ----
                    crate::pipeline::truncate_cache_by(&cache, n_drop).map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "ratio {ratio}, {committed} committed, depth {depth}, dropping \
                             {n_drop}: the MTP rejection rollback was REFUSED. \
                             XS_TAIL_MARGIN_TOKENS is supposed to make this impossible for any \
                             --mtp-depth (clap caps it at 8). {e}"
                        ))
                    })?;

                    // (1)+(2) every entry agrees, on the decode invariant.
                    let tokens_after = committed + 1 + commit_len;
                    let want_len = tokens_after - 1;
                    assert_eq!(
                        cache_lens(&cache),
                        vec![want_len; 3],
                        "ratio {ratio}: K/V and compressed entries disagree after a depth-{depth} \
                         rollback of {n_drop} (accepted {}, rejection {})",
                        result.accepted.len(),
                        result.rejection.is_some(),
                    );
                    if n_drop > 0 {
                        saw_a_real_truncation = true;
                    }

                    // (3) the surviving rows are the from-scratch answer.
                    let t_trunc = (want_len / ratio) * ratio;
                    let want = compressor.forward_from_xs(&xs.narrow(1, 0, t_trunc)?)?;
                    let got = compressed_rows_of(&cache, xs_idx)?
                        .expect("the prefill completed at least one group");
                    assert_eq!(got.dim(1)?, want.dim(1)?, "ratio {ratio}: wrong row count");
                    assert!(
                        max_abs_diff(&got, &want)? <= 1e-5,
                        "ratio {ratio}, {committed} committed, depth {depth}: the rolled-back \
                         compressed rows are not the from-scratch answer"
                    );
                    if got.dim(1)? < rows_before {
                        saw_a_row_drop = true;
                    }
                    // D12: a fixture whose compressor emits ~0 would make every
                    // equality above hold for the wrong reason.
                    let mag = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                    assert!(
                        mag > 1e-3,
                        "degenerate fixture: the compressor emits ~zero rows ({mag}), so \
                         'restored exactly' would be satisfied by any state at all"
                    );

                    // (4) decoding onward still matches from-scratch: the raw
                    // tail and `base` were restored, not just the row count.
                    let onward = 2 * ratio;
                    feed_v4_cache(&cache, &xs, want_len, onward, compressor)?;
                    let seen = want_len + onward;
                    let want =
                        compressor.forward_from_xs(&xs.narrow(1, 0, (seen / ratio) * ratio)?)?;
                    let got = compressed_rows_of(&cache, xs_idx)?.expect("rows after resume");
                    assert!(
                        max_abs_diff(&got, &want)? <= 1e-5,
                        "ratio {ratio}, {committed} committed, depth {depth}: resuming after the \
                         rollback compressed a GAP — the rows diverge from the whole-history \
                         recompute at {seen} tokens"
                    );
                }
            }
        }

        assert!(
            saw_a_real_truncation,
            "every case rolled back 0 positions — the test never exercised a truncation"
        );
        assert!(
            saw_a_row_drop,
            "no case ever crossed a group boundary, so the compressed-row rebuild — the part a \
             naive retention rule gets wrong — was never exercised"
        );
        Ok(())
    }

    /// The MTP rollback bound and the retained-tail margin are two constants in
    /// two places that must stay ordered. `--mtp-depth` is clap-capped at 8
    /// (`mistralrs-cli/src/args/mod.rs`), a rejection rolls back at most that
    /// many positions, and [`crate::XS_TAIL_MARGIN_TOKENS`] is what keeps that
    /// rollback inside the retained raw window.
    ///
    /// Raising the depth cap past the margin would not fail to compile and
    /// would not fail any test above — it would fail in production, on the
    /// fraction of rejections that happen to cross a group boundary.
    #[test]
    fn mtp_depth_cap_stays_within_the_retained_rollback_margin() {
        const MAX_MTP_DEPTH: usize = 8; // clap: `.range(0..=8)`
        assert!(
            MAX_MTP_DEPTH <= crate::XS_TAIL_MARGIN_TOKENS,
            "--mtp-depth accepts up to {MAX_MTP_DEPTH}, but the xs rolling cache only retains {} \
             tokens of rollback margin. A deeper draft can be rejected past the retained raw \
             window, and that rollback HARD-ERRORS mid-decode.",
            crate::XS_TAIL_MARGIN_TOKENS
        );
    }

    /// 🔑 The MTP draft step's own precondition, and the three layouts it must
    /// refuse rather than guess at.
    ///
    /// A dense `KvCache` appends at ONE offset, so a group shares a forward only
    /// when every row's live run ends at the same column. The scalar form —
    /// `&[pos]` — is the pre-change path exactly: `max` of a one-element slice
    /// is that element, so the check reduces to the old `cached != start_pos`
    /// character for character.
    #[test]
    fn the_draft_step_end_column_is_the_batch_maximum_and_refuses_every_other_layout() {
        // Scalar: the pre-change behaviour, at any batch size.
        assert_eq!(draft_step_end_column(&[7], 1, 7).unwrap(), 7);
        assert_eq!(draft_step_end_column(&[7], 4, 7).unwrap(), 7);
        let err = draft_step_end_column(&[7], 1, 6).unwrap_err().to_string();
        assert!(
            err.contains("desync") && err.contains("6") && err.contains("7"),
            "a cache that is not at the step's position must say both numbers; got {err}"
        );

        // Ragged: the end column is the batch maximum, which is what
        // `front_pad_kv_cache` left the cache at.
        assert_eq!(draft_step_end_column(&[7, 4, 6], 3, 7).unwrap(), 7);
        let err = draft_step_end_column(&[7, 4, 6], 3, 6)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("left-aligned"),
            "a group that was never left-aligned must be told so; got {err}"
        );

        // A row AHEAD of the buffer it is drafting into: `max` exceeds `cached`,
        // so its RoPE position and its draft-KV slot would mean different things.
        assert!(draft_step_end_column(&[7, 9], 2, 7).is_err());

        // Arity: neither 1 nor `b_sz`.
        let err = draft_step_end_column(&[7, 4], 3, 7)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("2 absolute position(s) for a batch of 3"),
            "a positions vector that does not describe the batch must refuse rather than be \
             indexed past its end; got {err}"
        );
        assert!(draft_step_end_column(&[7, 7, 7, 7], 3, 7).is_err());

        // No positions at all.
        assert!(draft_step_end_column(&[], 1, 0).is_err());
    }

    /// Pins the two-field router-weight contract introduced when the F32
    /// re-cast was hoisted out of `MoeGate::forward`.
    ///
    /// `forward` consumes `weight_t_f32`, which must be *exactly*
    /// `weight.t().to_dtype(F32)` — the expression it replaced — so the GEMM
    /// is bit-identical. `weight` itself must keep the checkpoint's shape and
    /// dtype, because `residual_tensors` serializes it straight back into
    /// UQFF under the name `weight`; if a future refactor drops it or swaps in
    /// the transposed copy, the artifact silently changes format.
    #[test]
    fn moe_gate_router_weight_is_hoisted_without_changing_the_uqff_tensor() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 8;
        let n_routed_experts = 4;
        let cfg = compressor_test_cfg(hidden);

        // Distinct, exactly-BF16-representable values so the comparison below
        // isolates layout/dtype handling rather than rounding.
        let weight_f32 = Tensor::from_vec(
            (0..n_routed_experts * hidden)
                .map(|i| (i as f32) * 0.25 - 3.0)
                .collect::<Vec<_>>(),
            (n_routed_experts, hidden),
            &device,
        )?;
        let mut map = std::collections::HashMap::new();
        map.insert("weight".to_string(), weight_f32.to_dtype(DType::BF16)?);
        let vb = vb_from_map(map, DType::BF16, &device);

        // layer_idx 10 >= default num_hash_layers (3), so no `tid2eid` lookup.
        let gate = MoeGate::new(&cfg, vb, n_routed_experts, 10)?;

        // UQFF-facing tensor: untouched checkpoint shape and dtype.
        assert_eq!(gate.weight.dims(), &[n_routed_experts, hidden]);
        assert_eq!(gate.weight.dtype(), DType::BF16);

        // GEMM-facing operand: transposed and promoted, ready to use.
        assert_eq!(gate.weight_t_f32.dims(), &[hidden, n_routed_experts]);
        assert_eq!(gate.weight_t_f32.dtype(), DType::F32);

        // And it is bit-identical to the per-call expression it replaced.
        let want: Vec<f32> = gate
            .weight
            .t()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let got: Vec<f32> = gate.weight_t_f32.flatten_all()?.to_vec1()?;
        assert_eq!(
            got, want,
            "hoisted router operand must equal `weight.t().to_dtype(F32)` exactly"
        );
        Ok(())
    }
}
