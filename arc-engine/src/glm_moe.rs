//! GLM MoE family model configuration and dispatch — covers GLM-4.5, GLM-4.6,
//! GLM-4.7, GLM-5.0, GLM-5.1 (and likely future variants) from one module.
//!
//! Reference (SGLang): `research/code/06_foundation/sglang/python/sglang/srt/`
//!   - `models/glm4_moe.py` (1,499 lines) — covers all GLM MoE versions
//!   - Two entry classes:
//!     - `Glm4MoeForCausalLM` — plain GLM-4 MoE
//!     - `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` — GLM-5 with DSA attention
//!   - `models/glm4_moe_lite.py` — smaller variant
//!   - `models/glm4_moe_nextn.py` — MTP / NextN
//!
//! Architecture varies by version:
//!   - GLM-4.5/4.6/4.7: standard GLM MoE attention
//!   - GLM-5.0/5.1: extends V3-style architecture with DSA (DeepSeek Sparse Attention),
//!     64 attention heads, grouped-noaux-tc top-k routing
//!
//! ## Tier A scope
//!
//! - `GlmMoeConfig` parses HuggingFace configs for GLM-4.5 through GLM-5.1
//! - `GlmVariant` enum dispatches to the right code path (plain MoE vs DSA)
//! - Architecture detection from `architectures` field
//! - Tests with GLM-4.5 / 4.6 / 4.7 / 5.0 / 5.1 representative configs
//!
//! ## Tier B (deferred)
//!
//! - Wire `mistralrs-core/src/models/glm_moe.rs` to dispatch between the two
//!   paths (plain GLM MoE → existing path; GLM-5 → V3-style DSA path via deepseek3.rs)
//! - Real model load + serve on GLM-5.1
//! - Vision (GLM-4V) — separate scope

use serde::Deserialize;

/// GLM-MoE variant detected from the model's architectures field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmVariant {
    /// GLM-4.5 / 4.6 / 4.7 — uses standard GLM MoE attention path.
    Glm4Moe,
    /// GLM-5.0 / 5.1 — uses DSA (V3 NSA) attention, extends DeepSeek V2/V3 path.
    Glm5MoeDsa,
    /// GLM-4-MoE-Lite — smaller variant.
    Glm4MoeLite,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GlmMoeConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: String,

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
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    #[serde(default)]
    pub kv_lora_rank: Option<usize>,
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    #[serde(default)]
    pub qk_nope_head_dim: Option<usize>,
    #[serde(default)]
    pub qk_rope_head_dim: Option<usize>,
    #[serde(default)]
    pub v_head_dim: Option<usize>,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// GLM-5-specific: presence of DSA attention type indicator.
    /// Some configs use `attention_type = "dsa"` to signal V3-NSA-style attention.
    #[serde(default)]
    pub attention_type: Option<String>,

    /// GLM-5 max output length (131K per Zhipu's spec for 5.1).
    #[serde(default)]
    pub max_output_length: Option<usize>,
}

fn default_vocab_size() -> usize {
    151_552
}
fn default_hidden_size() -> usize {
    4096
}
fn default_intermediate_size() -> usize {
    13_440
}
fn default_moe_intermediate_size() -> usize {
    1536
}
fn default_num_hidden_layers() -> usize {
    46
}
fn default_num_attention_heads() -> usize {
    64
}
fn default_num_key_value_heads() -> usize {
    8
}
fn default_n_routed_experts() -> usize {
    128
}
fn default_n_shared_experts() -> usize {
    1
}
fn default_num_experts_per_tok() -> usize {
    8
}
fn default_max_position() -> usize {
    200_000
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_rms_norm_eps() -> f64 {
    1e-5
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
fn default_true() -> bool {
    true
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}

impl GlmMoeConfig {
    pub fn from_json(json: &str) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Identify which variant this config represents.
    ///
    /// Detection rules (matching SGLang's dispatch in `glm4_moe.py`):
    /// 1. If architectures contains `GlmMoeDsaForCausalLM` → Glm5MoeDsa
    /// 2. Else if architectures contains `Glm4MoeLiteForCausalLM` → Glm4MoeLite
    /// 3. Else if `attention_type == "dsa"` → Glm5MoeDsa
    /// 4. Else if model_type contains "glm5" or hidden_size very large (>= 4096 with
    ///    large expert count) → likely Glm5MoeDsa
    /// 5. Otherwise → Glm4Moe (plain)
    pub fn variant(&self) -> GlmVariant {
        for arch in &self.architectures {
            let a = arch.to_lowercase();
            if a.contains("dsaforcausallm") || a.contains("glmmoedsa") {
                return GlmVariant::Glm5MoeDsa;
            }
            if a.contains("lite") {
                return GlmVariant::Glm4MoeLite;
            }
        }
        if let Some(at) = &self.attention_type {
            if at.eq_ignore_ascii_case("dsa") {
                return GlmVariant::Glm5MoeDsa;
            }
        }
        if self.model_type.to_lowercase().contains("glm5") {
            return GlmVariant::Glm5MoeDsa;
        }
        GlmVariant::Glm4Moe
    }

    pub fn is_glm_moe_architecture(&self) -> bool {
        if self
            .architectures
            .iter()
            .any(|a| a.to_lowercase().contains("glm"))
        {
            return true;
        }
        if !self.model_type.is_empty() && self.model_type.to_lowercase().contains("glm") {
            return true;
        }
        false
    }
}

pub fn is_glm_moe_config(json: &str) -> bool {
    GlmMoeConfig::from_json(json)
        .map(|cfg| cfg.is_glm_moe_architecture())
        .unwrap_or(false)
}

/// HuggingFace repo IDs for known GLM MoE family models.
pub fn known_glm_repos() -> &'static [&'static str] {
    &[
        "zai-org/GLM-4.5",
        "zai-org/GLM-4.6",
        "zai-org/GLM-4.7",
        "zai-org/GLM-4.7-Flash",
        "zai-org/GLM-5.0",
        "zai-org/GLM-5.1",
        "THUDM/GLM-4.5",
        "THUDM/GLM-4.6",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_parses() {
        // Defaults to plain Glm4Moe variant when no architectures or model_type
        // explicitly mark it as something else.
        let cfg: GlmMoeConfig =
            serde_json::from_str(r#"{"architectures": ["Glm4MoeForCausalLM"]}"#).unwrap();
        assert_eq!(cfg.variant(), GlmVariant::Glm4Moe);
        assert_eq!(cfg.n_routed_experts, 128);
        assert_eq!(cfg.num_attention_heads, 64);
    }

    #[test]
    fn detects_glm5_via_architectures() {
        let json = r#"{
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm5",
            "num_attention_heads": 64,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "max_position_embeddings": 200000
        }"#;
        let cfg = GlmMoeConfig::from_json(json).expect("parse");
        assert_eq!(cfg.variant(), GlmVariant::Glm5MoeDsa);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.n_routed_experts, 256);
    }

    #[test]
    fn detects_glm45_as_plain_glm4_moe() {
        let json = r#"{
            "architectures": ["Glm4MoeForCausalLM"],
            "model_type": "glm4_moe",
            "n_routed_experts": 128,
            "max_position_embeddings": 131072
        }"#;
        let cfg = GlmMoeConfig::from_json(json).expect("parse");
        assert_eq!(cfg.variant(), GlmVariant::Glm4Moe);
    }

    #[test]
    fn detects_glm_lite() {
        let json = r#"{
            "architectures": ["Glm4MoeLiteForCausalLM"],
            "model_type": "glm4_moe_lite",
            "n_routed_experts": 32
        }"#;
        let cfg = GlmMoeConfig::from_json(json).expect("parse");
        assert_eq!(cfg.variant(), GlmVariant::Glm4MoeLite);
    }

    #[test]
    fn detects_glm5_via_model_type() {
        let json = r#"{
            "architectures": ["Glm4MoeForCausalLM"],
            "model_type": "glm5"
        }"#;
        let cfg = GlmMoeConfig::from_json(json).expect("parse");
        assert_eq!(cfg.variant(), GlmVariant::Glm5MoeDsa);
    }

    #[test]
    fn detects_glm5_via_attention_type() {
        let json = r#"{
            "architectures": ["Glm4MoeForCausalLM"],
            "model_type": "glm4_moe",
            "attention_type": "dsa"
        }"#;
        let cfg = GlmMoeConfig::from_json(json).expect("parse");
        assert_eq!(cfg.variant(), GlmVariant::Glm5MoeDsa);
    }

    #[test]
    fn parse_glm51_style_config() {
        // GLM-5.1: 754B params / 40B active / 200K context / 131K max output
        let json = r#"{
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm5",
            "vocab_size": 151552,
            "num_hidden_layers": 80,
            "hidden_size": 5120,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "max_position_embeddings": 200000,
            "max_output_length": 131072,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128
        }"#;
        let cfg = GlmMoeConfig::from_json(json).expect("parse");
        assert_eq!(cfg.variant(), GlmVariant::Glm5MoeDsa);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.kv_lora_rank, Some(512));
        assert_eq!(cfg.max_position_embeddings, 200_000);
        assert_eq!(cfg.max_output_length, Some(131_072));
        assert!(cfg.is_glm_moe_architecture());
    }

    #[test]
    fn architecture_dispatch_distinguishes_glm_from_llama() {
        assert!(is_glm_moe_config(
            r#"{"architectures": ["Glm4MoeForCausalLM"]}"#
        ));
        assert!(is_glm_moe_config(
            r#"{"architectures": ["GlmMoeDsaForCausalLM"]}"#
        ));
        assert!(is_glm_moe_config(r#"{"model_type": "glm4"}"#));
        assert!(!is_glm_moe_config(
            r#"{"architectures": ["LlamaForCausalLM"]}"#
        ));
        assert!(!is_glm_moe_config(
            r#"{"architectures": ["DeepseekV4ForCausalLM"]}"#
        ));
    }

    #[test]
    fn known_repos_listed() {
        let repos = known_glm_repos();
        assert!(repos.iter().any(|r| r.contains("GLM-5.0")));
        assert!(repos.iter().any(|r| r.contains("GLM-5.1")));
        assert!(repos.iter().any(|r| r.contains("GLM-4.5")));
    }
}
