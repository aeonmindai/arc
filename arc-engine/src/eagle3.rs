//! EAGLE-3 speculative decoding head loader.
//!
//! Paper:    `research/10_foundation/eagle_v3.pdf`
//! Repo:     `research/code/03_per_token_speed/eagle/`
//! Pre-trained heads on HF: see `research/code/03_per_token_speed/eagle/README.md`
//!                          (Llama-3.1/3.3/4, Qwen3 family, GPT-OSS, GLM, Kimi K2.5, …)
//!
//! ## What
//!
//! EAGLE-3 trains a small per-target-model draft transformer that, given the
//! target's hidden states + the most recent token, proposes the next 1–3 tokens.
//! The target verifies in a single forward via tree attention; accepted tokens
//! commit, the first reject is replaced by the target's correct token.
//!
//! ## Tier A scope (this module)
//!
//! - `Eagle3Config`: depth, hidden_size, vocab_size
//! - `Eagle3DraftHead`: structure (weights loaded by Tier B model integration)
//! - `huggingface_head_for_target()`: HF naming-convention auto-detection
//! - Re-uses `crate::mtp::verify_proposed()` for accept/reject (identical protocol)
//! - Tests: HF detection, structural invariants
//!
//! ## Tier B (deferred)
//!
//! - Load `cnets.py`-shaped weights from HF, wire into model-specific adapters
//!   (llama_eagle3, qwen2_eagle, kimi_k25_eagle3, …)
//! - Tree-attention verification kernel
//! - Per-model integration in `mistralrs-core/src/pipeline/speculative.rs`

/// Configuration of an EAGLE-3 draft head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eagle3Config {
    /// Depth of the draft tree (how many tokens proposed per forward).
    /// EAGLE-3 default is 4 (5 = the target + 4 drafted ahead).
    pub depth: usize,
    /// Hidden dim of the target model.
    pub hidden_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
}

#[derive(Debug, Clone)]
pub struct Eagle3DraftHead {
    pub cfg: Eagle3Config,
    /// HuggingFace repo where the head's weights live (e.g.,
    /// "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B").
    pub hf_repo: String,
}

impl Eagle3DraftHead {
    pub fn new(cfg: Eagle3Config, hf_repo: String) -> Self {
        Self { cfg, hf_repo }
    }
}

/// Auto-detect the published EAGLE-3 head repo for a target HuggingFace model ID.
///
/// Mappings come from `research/code/03_per_token_speed/eagle/README.md`. Returns
/// `Some("yuhuili/EAGLE3-...")` for known targets, `None` otherwise.
///
/// We hard-code the well-known mappings rather than parsing README at runtime
/// because (a) it's stable, (b) it lets Arc serve those models with `--eagle3-head auto`,
/// (c) it makes the supported set discoverable through the type system.
pub fn huggingface_head_for_target(target_hf_id: &str) -> Option<&'static str> {
    match target_hf_id {
        "lmsys/vicuna-13b-v1.3" => Some("yuhuili/EAGLE3-Vicuna1.3-13B"),
        "meta-llama/Llama-3.1-8B-Instruct" => Some("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"),
        "meta-llama/Llama-3.3-70B-Instruct" => Some("yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"),
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" => {
            Some("yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B")
        }
        "meta-llama/Llama-4-Scout-17B-16E-Instruct" => {
            Some("lmsys/sglang-EAGLE3-Llama-4-Scout-17B-16E-Instruct-v1")
        }
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct" => {
            Some("lmsys/sglang-EAGLE3-Llama-4-Maverick-17B-128E-Instruct-v1")
        }
        "Qwen/Qwen3-1.7B" => Some("AngelSlim/Qwen3-1.7B_eagle3"),
        "Qwen/Qwen3-4B" => Some("AngelSlim/Qwen3-4B_eagle3"),
        "Qwen/Qwen3-8B" => Some("Tengyunw/qwen3_8b_eagle3"),
        "Qwen/Qwen3-14B" => Some("AngelSlim/Qwen3-14B_eagle3"),
        "Qwen/Qwen3-32B" => Some("AngelSlim/Qwen3-32B_eagle3"),
        "Qwen/Qwen3-30B-A3B" => Some("Tengyunw/qwen3_30b_moe_eagle3"),
        "Qwen/Qwen3-235B-A22B" => Some("nvidia/Qwen3-235B-A22B-Eagle3"),
        "openai/gpt-oss-120b" => Some("lmsys/EAGLE3-gpt-oss-120b-bf16"),
        "zai-org/GLM-4.7-Flash" => Some("thoughtworks/GLM-4.7-Flash-Eagle3"),
        "moonshotai/Kimi-K2.5-Instruct" => None, // covered by SGLang's wrapper; no standalone HF head
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_detect_finds_llama_3_1_head() {
        assert_eq!(
            huggingface_head_for_target("meta-llama/Llama-3.1-8B-Instruct"),
            Some("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
        );
    }

    #[test]
    fn auto_detect_finds_qwen3_family() {
        for (target, expected) in [
            ("Qwen/Qwen3-8B", "Tengyunw/qwen3_8b_eagle3"),
            ("Qwen/Qwen3-32B", "AngelSlim/Qwen3-32B_eagle3"),
            ("Qwen/Qwen3-235B-A22B", "nvidia/Qwen3-235B-A22B-Eagle3"),
        ] {
            assert_eq!(huggingface_head_for_target(target), Some(expected));
        }
    }

    #[test]
    fn auto_detect_finds_glm_47_flash() {
        assert_eq!(
            huggingface_head_for_target("zai-org/GLM-4.7-Flash"),
            Some("thoughtworks/GLM-4.7-Flash-Eagle3")
        );
    }

    #[test]
    fn auto_detect_returns_none_for_unknown_target() {
        assert_eq!(huggingface_head_for_target("my-org/Unsupported-Model-7B"), None);
        // Plain Llama-2 has no community EAGLE-3 head (only EAGLE-1)
        assert_eq!(huggingface_head_for_target("meta-llama/Llama-2-7B"), None);
    }

    #[test]
    fn draft_head_carries_config_and_repo() {
        let cfg = Eagle3Config {
            depth: 4,
            hidden_size: 4096,
            vocab_size: 128256,
        };
        let head = Eagle3DraftHead::new(cfg, "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B".into());
        assert_eq!(head.cfg, cfg);
        assert_eq!(head.hf_repo, "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B");
    }
}
