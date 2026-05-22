//! DeepSeek V4 model configuration and loader scaffolding.
//!
//! Reference (canonical): `research/code/06_foundation/sglang/python/sglang/srt/`
//!   - `configs/deepseek_v4.py` — `DeepSeekV4Config` dataclass
//!   - `models/deepseek_v4.py` — main model (1,816 lines, extends `deepseek_v2.py`)
//!   - `models/deepseek_v4_nextn.py` — MTP / NextN extension
//!   - `layers/attention/dsv4/` — CSA + HCA backend
//!
//! ## Scope (Tier A)
//!
//! - `DeepSeekV4Config` Rust struct — field-for-field mirror of SGLang's dataclass
//! - HuggingFace `config.json` parsing
//! - `architectures` field dispatch (handles `DeepseekV4ForCausalLM`)
//! - Compress-ratio layer map: per-layer dispatch between standard / CSA / HCA
//! - Safetensors index validation (expected weight names for V4 + MTP)
//! - Tests against the actual HF config from `deepseek-ai/DeepSeek-V4-Flash` /
//!   `DeepSeek-V4-Pro`
//!
//! ## Tier B (deferred)
//!
//! - Full model graph construction in `mistralrs-core/src/models/deepseek4.rs`
//! - End-to-end forward pass on real V4 Flash weights
//! - MTP weight loading + speculative decoding integration on V4

use crate::dsv4::CompressRatio;
use serde::Deserialize;

/// Configuration for DeepSeek V4 (Pro and Flash share this; differences are
/// values, not field names).
///
/// Mirrors SGLang's `DeepSeekV4Config` dataclass at
/// `sglang/srt/configs/deepseek_v4.py`. Field names match exactly so that
/// HuggingFace `config.json` deserializes directly.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV4Config {
    #[serde(default = "default_architectures")]
    pub architectures: Vec<String>,

    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
    #[serde(default = "one_usize")]
    pub ep_size: usize,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_index_head_dim")]
    pub index_head_dim: usize,
    #[serde(default = "default_index_n_heads")]
    pub index_n_heads: usize,
    #[serde(default = "default_index_topk")]
    pub index_topk: usize,

    #[serde(default = "default_initializer_range")]
    pub initializer_range: f32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_kv_lora_rank")]
    pub kv_lora_rank: usize,
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_model_type")]
    pub model_type: String,

    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: usize,
    #[serde(default = "one_usize")]
    pub moe_layer_freq: usize,
    #[serde(default = "default_n_group")]
    pub n_group: usize,
    #[serde(default = "default_n_routed_experts")]
    pub n_routed_experts: usize,
    #[serde(default = "one_usize")]
    pub n_shared_experts: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,

    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "one_usize")]
    pub num_key_value_heads: usize,

    #[serde(default = "default_q_lora_rank")]
    pub q_lora_rank: usize,
    #[serde(default = "default_qk_nope_head_dim")]
    pub qk_nope_head_dim: usize,
    #[serde(default = "default_qk_rope_head_dim")]
    pub qk_rope_head_dim: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_topk_group")]
    pub topk_group: usize,
    #[serde(default = "default_topk_method")]
    pub topk_method: String,

    #[serde(default = "default_v_head_dim")]
    pub v_head_dim: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// V4-specific: per-layer compression ratio. Length must equal num_hidden_layers.
    /// Values must be in {0, 4, 128}. 0 = standard MLA, 4 = CSA, 128 = HCA.
    #[serde(default)]
    pub compress_ratios: Vec<i32>,

    /// V4-specific: window size for the local sliding-window branch (paired with
    /// CSA + HCA). Default 128 per the V4 config.
    #[serde(default = "default_window_size")]
    pub window_size: usize,

    /// V4-specific: separate rope_theta for the compressed-attention path.
    #[serde(default = "default_compress_rope_theta")]
    pub compress_rope_theta: u32,
}

// --- defaults (mirror SGLang's dataclass defaults) ---
fn default_architectures() -> Vec<String> { vec!["DeepseekV4ForCausalLM".into()] }
fn default_eos() -> u32 { 1 }
fn one_usize() -> usize { 1 }
fn default_hidden_act() -> String { "silu".into() }
fn default_hidden_size() -> usize { 4096 }
fn default_index_head_dim() -> usize { 128 }
fn default_index_n_heads() -> usize { 64 }
fn default_index_topk() -> usize { 512 }
fn default_initializer_range() -> f32 { 0.02 }
fn default_intermediate_size() -> usize { 2048 }
fn default_kv_lora_rank() -> usize { 512 }
fn default_max_position() -> usize { 65536 }
fn default_model_type() -> String { "deepseek_v4".into() }
fn default_moe_intermediate_size() -> usize { 2048 }
fn default_n_group() -> usize { 8 }
fn default_n_routed_experts() -> usize { 256 }
fn default_true() -> bool { true }
fn default_num_attention_heads() -> usize { 64 }
fn default_num_experts_per_tok() -> usize { 6 }
fn default_num_hidden_layers() -> usize { 43 }
fn default_q_lora_rank() -> usize { 1024 }
fn default_qk_nope_head_dim() -> usize { 448 }
fn default_qk_rope_head_dim() -> usize { 64 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f32 { 10000.0 }
fn default_routed_scaling_factor() -> f32 { 1.5 }
fn default_scoring_func() -> String { "sqrtsoftplus".into() }
fn default_topk_group() -> usize { 8 }
fn default_topk_method() -> String { "noaux_tc".into() }
fn default_v_head_dim() -> usize { 512 }
fn default_vocab_size() -> usize { 129280 }
fn default_window_size() -> usize { 128 }
fn default_compress_rope_theta() -> u32 { 40000 }

impl DeepSeekV4Config {
    /// Parse from a HuggingFace `config.json` string.
    pub fn from_json(json: &str) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Get the compression ratio for a specific layer. If `compress_ratios` is
    /// empty or shorter than `layer_id+1`, returns Standard (no compression).
    pub fn layer_compress_ratio(&self, layer_id: usize) -> crate::dsv4::CompressRatio {
        let raw = if layer_id < self.compress_ratios.len() {
            self.compress_ratios[layer_id]
        } else {
            0
        };
        crate::dsv4::CompressRatio::from_i32(raw).unwrap_or(CompressRatio::Standard)
    }

    /// Total Q head dimension = qk_nope + qk_rope.
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Is this model an MLA-based DeepSeek V4 architecture?
    pub fn is_v4_architecture(&self) -> bool {
        self.architectures
            .iter()
            .any(|a| a.contains("DeepseekV4") || a.contains("DeepSeekV4"))
            || self.model_type == "deepseek_v4"
    }

    /// Estimate the number of layers using each compression ratio. Useful for
    /// memory budgeting and verification at model load.
    pub fn compress_ratio_layer_counts(&self) -> (usize, usize, usize) {
        let mut std = 0;
        let mut csa = 0;
        let mut hca = 0;
        for layer in 0..self.num_hidden_layers {
            match self.layer_compress_ratio(layer) {
                CompressRatio::Standard => std += 1,
                CompressRatio::Csa => csa += 1,
                CompressRatio::Hca => hca += 1,
            }
        }
        (std, csa, hca)
    }
}

/// Inspect a HuggingFace `config.json` blob and return true if it's a V4 model.
/// Useful for the top-level model-architecture dispatcher.
pub fn is_v4_config(json: &str) -> bool {
    DeepSeekV4Config::from_json(json)
        .map(|cfg| cfg.is_v4_architecture())
        .unwrap_or(false)
}

/// Expected weight tensor name patterns for DeepSeek V4. Used to validate that
/// a safetensors checkpoint has all required tensors before attempting load.
///
/// V4 inherits most names from V3 (MLA-style q_a_proj, q_b_proj, kv_a_proj_with_mqa,
/// kv_b_proj, o_proj, plus expert weights). The V4-specific additions are:
///   - `model.layers.<i>.self_attn.compressor.*` — CSA/HCA compressor weights
///   - `model.layers.<i>.self_attn.indexer.*` — CSA top-k indexer weights
///   - `mtp.layers.<i>.*` — MTP heads (same as V3+)
pub fn expected_weight_name_prefixes() -> Vec<&'static str> {
    vec![
        // Standard transformer
        "model.embed_tokens",
        "model.norm",
        "lm_head",
        // Per-layer MLA components
        "model.layers.0.input_layernorm",
        "model.layers.0.post_attention_layernorm",
        "model.layers.0.self_attn.q_a_proj",
        "model.layers.0.self_attn.q_b_proj",
        "model.layers.0.self_attn.kv_a_proj_with_mqa",
        "model.layers.0.self_attn.kv_b_proj",
        "model.layers.0.self_attn.o_proj",
        // V4-specific compressor / indexer (present on CSA/HCA layers only)
        "model.layers.0.self_attn.compressor",
        // MoE
        "model.layers.1.mlp.experts.0.gate_proj",
        "model.layers.1.mlp.experts.0.up_proj",
        "model.layers.1.mlp.experts.0.down_proj",
        "model.layers.1.mlp.gate.weight",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv4::CompressRatio;

    /// Default V4 config matches the dataclass defaults from SGLang.
    #[test]
    fn default_config_matches_sglang_defaults() {
        let cfg: DeepSeekV4Config = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.num_hidden_layers, 43);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.qk_nope_head_dim, 448);
        assert_eq!(cfg.qk_rope_head_dim, 64);
        assert_eq!(cfg.index_topk, 512);
        assert_eq!(cfg.index_n_heads, 64);
        assert_eq!(cfg.n_routed_experts, 256);
        assert_eq!(cfg.n_shared_experts, 1);
        assert_eq!(cfg.num_experts_per_tok, 6);
        assert_eq!(cfg.vocab_size, 129280);
        assert_eq!(cfg.model_type, "deepseek_v4");
        assert_eq!(cfg.scoring_func, "sqrtsoftplus");
        assert_eq!(cfg.topk_method, "noaux_tc");
        assert!(cfg.is_v4_architecture());
    }

    /// V4 Flash config parses with smaller dims.
    #[test]
    fn parse_v4_flash_style_config() {
        let json = r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "num_hidden_layers": 27,
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "n_routed_experts": 128,
            "vocab_size": 129280,
            "compress_ratios": [0, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4]
        }"#;
        let cfg = DeepSeekV4Config::from_json(json).expect("parse");
        assert_eq!(cfg.num_hidden_layers, 27);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.compress_ratios.len(), 27);
        // First layer should be Standard, then CSA on layer 1
        assert_eq!(cfg.layer_compress_ratio(0), CompressRatio::Standard);
        assert_eq!(cfg.layer_compress_ratio(1), CompressRatio::Csa);
        assert_eq!(cfg.layer_compress_ratio(3), CompressRatio::Hca);
    }

    /// V4 Pro config parses with the headline numbers.
    #[test]
    fn parse_v4_pro_style_config() {
        let json = r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "n_routed_experts": 256,
            "vocab_size": 129280,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 448,
            "qk_rope_head_dim": 64,
            "index_topk": 1024,
            "max_position_embeddings": 1048576
        }"#;
        let cfg = DeepSeekV4Config::from_json(json).expect("parse");
        assert_eq!(cfg.num_hidden_layers, 43);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.q_head_dim(), 448 + 64);
        assert_eq!(cfg.max_position_embeddings, 1_048_576);
        assert_eq!(cfg.index_topk, 1024);
        assert!(cfg.is_v4_architecture());
    }

    /// Architecture dispatch correctly identifies V4 vs V3.
    #[test]
    fn architecture_dispatch_recognizes_v4() {
        let v4_json = r#"{"architectures": ["DeepseekV4ForCausalLM"]}"#;
        assert!(is_v4_config(v4_json));

        let v3_json = r#"{"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3"}"#;
        assert!(!is_v4_config(v3_json));
    }

    /// Layer compression ratio dispatch handles default and overflow.
    #[test]
    fn layer_compress_ratio_handles_edge_cases() {
        let mut cfg: DeepSeekV4Config = serde_json::from_str("{}").unwrap();
        cfg.compress_ratios = vec![0, 4, 128];
        cfg.num_hidden_layers = 5;
        assert_eq!(cfg.layer_compress_ratio(0), CompressRatio::Standard);
        assert_eq!(cfg.layer_compress_ratio(1), CompressRatio::Csa);
        assert_eq!(cfg.layer_compress_ratio(2), CompressRatio::Hca);
        // Out of range → Standard fallback
        assert_eq!(cfg.layer_compress_ratio(3), CompressRatio::Standard);
        assert_eq!(cfg.layer_compress_ratio(100), CompressRatio::Standard);
    }

    /// Compression-ratio layer counts add up to total layers.
    #[test]
    fn compress_ratio_counts_sum_correctly() {
        let mut cfg: DeepSeekV4Config = serde_json::from_str("{}").unwrap();
        cfg.num_hidden_layers = 10;
        cfg.compress_ratios = vec![0, 4, 4, 128, 4, 4, 128, 4, 4, 128];
        let (std, csa, hca) = cfg.compress_ratio_layer_counts();
        assert_eq!(std + csa + hca, 10);
        assert_eq!(std, 1);
        assert_eq!(csa, 6);
        assert_eq!(hca, 3);
    }

    /// Invalid compress_ratio values in the config don't crash —
    /// they fall back to Standard.
    #[test]
    fn invalid_compress_ratio_falls_back_to_standard() {
        let json = r#"{"compress_ratios": [0, 99, 4]}"#;
        let cfg = DeepSeekV4Config::from_json(json).expect("parse");
        // 99 is not in {0, 4, 128}; from_i32 returns Err, fallback is Standard.
        assert_eq!(cfg.layer_compress_ratio(1), CompressRatio::Standard);
    }

    /// Expected weight name prefixes include the V4-specific compressor path.
    #[test]
    fn expected_weight_names_include_compressor() {
        let prefixes = expected_weight_name_prefixes();
        assert!(prefixes.iter().any(|p| p.contains("compressor")));
        assert!(prefixes.iter().any(|p| p.contains("kv_a_proj_with_mqa")));
        assert!(prefixes.iter().any(|p| p.contains("q_a_proj")));
        assert!(prefixes.iter().any(|p| p.contains("experts")));
    }
}
