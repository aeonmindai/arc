//! Kimi K2.5 / K2.6 model configuration and architecture dispatch.
//!
//! Reference (SGLang): `research/code/06_foundation/sglang/python/sglang/srt/`
//!   - `configs/kimi_k25.py` — `KimiK25Config` wraps a `DeepseekV3Config` for text
//!   - `models/kimi_k25.py` — main loader, **extends `DeepseekV3ForCausalLM`**
//!   - `models/kimi_vl_moonvit.py` — MoonViT vision tower (400M params)
//!
//! ## Key insight
//!
//! Kimi K2.5/K2.6 text architecture **is** DeepSeek V3 architecture (MLA + MoE).
//! The Kimi-specific differences are:
//!   - Vocabulary (160K vs DeepSeek's 129K)
//!   - Expert count (384 vs DeepSeek's varies)
//!   - MoonViT vision tower for image/video tokens
//!   - Kimi-specific config field names (`text_config` wrapper around V3 config)
//!
//! So Arc's existing `deepseek3.rs` is ~90% of what's needed. This module
//! handles the Kimi-specific config wrapping + architecture detection.
//!
//! ## Tier A scope
//!
//! - `KimiK2Config` Rust struct mirroring `KimiK25Config`
//! - Architecture detection from HF `config.json`
//! - Text-config extraction (unwraps the nested DeepSeek V3 config)
//! - Multimodal token handling stubs (placeholder IDs)
//! - Tests
//!
//! ## Tier B (deferred)
//!
//! - Wire `mistralrs-core/src/models/kimi_k2.rs` to use Arc's `deepseek3.rs`
//!   forward path with Kimi-specific config + vocab
//! - MoonViT integration (separate sub-issue if scoped in)
//! - End-to-end load on `moonshotai/Kimi-K2.5-Instruct`

use serde::Deserialize;

/// Top-level Kimi K2.5/K2.6 config. Mirrors SGLang's `KimiK25Config`.
///
/// The text model uses DeepSeek V3 architecture under `text_config`; vision
/// uses MoonViT under `vision_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct KimiK2Config {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: String,

    /// DeepSeek V3 text config nested here (Kimi wraps it).
    /// Falls back to defaults if missing.
    #[serde(default)]
    pub text_config: KimiK2TextConfig,

    /// Vision configuration. Optional — text-only inference works without it.
    #[serde(default)]
    pub vision_config: Option<KimiK2VisionConfig>,

    #[serde(default = "default_media_placeholder")]
    pub media_placeholder_token_id: u32,
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default = "default_ignore_index")]
    pub ignore_index: i32,
}

/// Text-side configuration. Mirrors the DeepSeek V3 portion that Kimi inherits.
#[derive(Debug, Clone, Deserialize)]
pub struct KimiK2TextConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,

    #[serde(default = "default_n_routed_experts")]
    pub n_routed_experts: usize,
    #[serde(default = "default_n_shared_experts")]
    pub n_shared_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,

    #[serde(default = "default_kv_lora_rank")]
    pub kv_lora_rank: usize,
    #[serde(default = "default_q_lora_rank")]
    pub q_lora_rank: usize,
    #[serde(default = "default_qk_nope_head_dim")]
    pub qk_nope_head_dim: usize,
    #[serde(default = "default_qk_rope_head_dim")]
    pub qk_rope_head_dim: usize,
    #[serde(default = "default_v_head_dim")]
    pub v_head_dim: usize,

    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    #[serde(default = "default_topk_method")]
    pub topk_method: String,
    #[serde(default = "default_n_group")]
    pub n_group: usize,
    #[serde(default = "default_topk_group")]
    pub topk_group: usize,
}

impl Default for KimiK2TextConfig {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

/// Kimi MoonViT vision tower configuration.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct KimiK2VisionConfig {
    #[serde(default)]
    pub patch_size: usize,
    #[serde(default)]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default)]
    pub intermediate_size: usize,
}

fn default_media_placeholder() -> u32 {
    163605
}
fn default_ignore_index() -> i32 {
    -100
}

// K2.6 text defaults — 1T params, 32B active, 61 layers
fn default_vocab_size() -> usize {
    160_000
}
fn default_hidden_size() -> usize {
    7168
}
fn default_intermediate_size() -> usize {
    18_432
}
fn default_moe_intermediate_size() -> usize {
    2048
}
fn default_num_hidden_layers() -> usize {
    61
}
fn default_num_attention_heads() -> usize {
    64
}
fn default_num_key_value_heads() -> usize {
    1
} // MLA-style heavy GQA

fn default_n_routed_experts() -> usize {
    384
}
fn default_n_shared_experts() -> usize {
    1
}
fn default_num_experts_per_tok() -> usize {
    8
}

fn default_kv_lora_rank() -> usize {
    512
}
fn default_q_lora_rank() -> usize {
    1536
}
fn default_qk_nope_head_dim() -> usize {
    128
}
fn default_qk_rope_head_dim() -> usize {
    64
}
fn default_v_head_dim() -> usize {
    128
}

fn default_max_position() -> usize {
    262_144
} // 256K
fn default_rope_theta() -> f32 {
    50_000.0
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_hidden_act() -> String {
    "silu".into()
}
fn default_topk_method() -> String {
    "noaux_tc".into()
}
fn default_n_group() -> usize {
    8
}
fn default_topk_group() -> usize {
    4
}

impl KimiK2Config {
    pub fn from_json(json: &str) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn is_kimi_architecture(&self) -> bool {
        if self
            .architectures
            .iter()
            .any(|a| a.to_lowercase().contains("kimi"))
        {
            return true;
        }
        if !self.model_type.is_empty() && self.model_type.to_lowercase().contains("kimi") {
            return true;
        }
        false
    }

    pub fn has_vision(&self) -> bool {
        self.vision_config.is_some()
    }

    pub fn q_head_dim(&self) -> usize {
        self.text_config.qk_nope_head_dim + self.text_config.qk_rope_head_dim
    }
}

/// Inspect a HF `config.json` and return true if it's a Kimi K2 family model.
pub fn is_kimi_k2_config(json: &str) -> bool {
    KimiK2Config::from_json(json)
        .map(|cfg| cfg.is_kimi_architecture())
        .unwrap_or(false)
}

/// HuggingFace repo IDs for known Kimi K2 family models.
pub fn known_kimi_k2_repos() -> &'static [&'static str] {
    &[
        "moonshotai/Kimi-K2.5-Instruct",
        "moonshotai/Kimi-K2.6-Instruct",
        "moonshotai/Kimi-K2-Instruct",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_k26_sizing() {
        // With minimal JSON, defaults apply but architecture is NOT auto-detected
        // (architectures + model_type are empty by default — caller must set them).
        let cfg: KimiK2Config =
            serde_json::from_str(r#"{"architectures": ["KimiK25ForConditionalGeneration"]}"#)
                .unwrap();
        assert_eq!(cfg.text_config.num_hidden_layers, 61);
        assert_eq!(cfg.text_config.hidden_size, 7168);
        assert_eq!(cfg.text_config.num_attention_heads, 64);
        assert_eq!(cfg.text_config.n_routed_experts, 384);
        assert_eq!(cfg.text_config.num_experts_per_tok, 8);
        assert_eq!(cfg.text_config.vocab_size, 160_000);
        assert_eq!(cfg.text_config.max_position_embeddings, 262_144);
        assert!(cfg.is_kimi_architecture());
        assert!(!cfg.has_vision());
    }

    #[test]
    fn parse_explicit_k25_config() {
        let json = r#"{
            "architectures": ["KimiK25ForConditionalGeneration"],
            "model_type": "kimi_k25",
            "text_config": {
                "vocab_size": 160000,
                "num_hidden_layers": 61,
                "hidden_size": 7168,
                "n_routed_experts": 384,
                "num_experts_per_tok": 8,
                "max_position_embeddings": 262144
            },
            "media_placeholder_token_id": 163605
        }"#;
        let cfg = KimiK2Config::from_json(json).expect("parse");
        assert_eq!(cfg.text_config.vocab_size, 160_000);
        assert_eq!(cfg.text_config.n_routed_experts, 384);
        assert_eq!(cfg.media_placeholder_token_id, 163605);
    }

    #[test]
    fn parse_kimi_with_vision_tower() {
        let json = r#"{
            "architectures": ["KimiK25ForConditionalGeneration"],
            "text_config": {},
            "vision_config": {
                "patch_size": 14,
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304
            }
        }"#;
        let cfg = KimiK2Config::from_json(json).expect("parse");
        assert!(cfg.has_vision());
        let vis = cfg.vision_config.as_ref().unwrap();
        assert_eq!(vis.patch_size, 14);
        assert_eq!(vis.num_hidden_layers, 27);
    }

    #[test]
    fn architecture_dispatch_recognizes_kimi() {
        assert!(is_kimi_k2_config(
            r#"{"architectures": ["KimiK25ForConditionalGeneration"]}"#
        ));
        assert!(is_kimi_k2_config(r#"{"model_type": "kimi_k25"}"#));
        assert!(is_kimi_k2_config(
            r#"{"architectures": ["KimiK2ForCausalLM"]}"#
        ));
        assert!(!is_kimi_k2_config(
            r#"{"architectures": ["DeepseekV3ForCausalLM"]}"#
        ));
        assert!(!is_kimi_k2_config(
            r#"{"architectures": ["LlamaForCausalLM"]}"#
        ));
    }

    #[test]
    fn q_head_dim_correct() {
        let cfg: KimiK2Config = serde_json::from_str("{}").unwrap();
        // 128 (nope) + 64 (rope) = 192
        assert_eq!(cfg.q_head_dim(), 192);
    }

    #[test]
    fn known_repos_listed() {
        let repos = known_kimi_k2_repos();
        assert!(repos.iter().any(|r| r.contains("Kimi-K2.5")));
        assert!(repos.iter().any(|r| r.contains("Kimi-K2.6")));
    }
}
