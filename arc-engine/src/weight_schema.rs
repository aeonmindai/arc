//! Weight-schema validators for the v2 target model families.
//!
//! The point: given a list of safetensors weight tensor names (from
//! `model.safetensors.index.json`), tell the caller — *offline, before any
//! GPU rental* — exactly which tensors are required, which are present, and
//! which are missing.
//!
//! This is the test that catches "V3 loader can't find V4's `o_a_proj` /
//! `o_b_proj` / `compressor` tensors" without spending B200 hours discovering it.
//!
//! ## Coverage
//!
//! - DeepSeek V4 (Pro + Flash): MLA-LoRA Q + KV, LoRA-decomposed o_proj,
//!   optional CSA/HCA compressor per layer, MTP heads
//! - Kimi K2.5 / K2.6 (text-side, inherits V3): MLA-LoRA Q + KV, single o_proj
//! - GLM-5.0 / 5.1: V3-MLA + DSA, single o_proj
//!
//! For each, the validator reports:
//!   - tensors that match expected names
//!   - tensors that are missing (and what loading will fail on)
//!   - extra tensors that the loader will ignore (informational)
//!   - inferred num_layers and num_experts from the keys

use std::collections::HashSet;

/// Outcome of validating a list of safetensors keys against an architecture's schema.
#[derive(Debug, Clone)]
pub struct WeightValidation {
    /// Required tensor patterns that were found.
    pub found: Vec<String>,
    /// Required tensor patterns that were NOT found — load will fail.
    pub missing: Vec<String>,
    /// Tensors present that aren't in the expected set (informational; may be MTP, vision tower, etc.).
    pub extra: Vec<String>,
    /// Inferred number of transformer layers (from `layers.<N>` prefixes).
    pub num_layers: usize,
    /// Inferred number of MoE experts (from `mlp.experts.<N>` prefixes).
    pub num_experts: Option<usize>,
    /// Whether the architecture's main entrypoint will succeed.
    pub will_load: bool,
}

impl WeightValidation {
    pub fn is_valid(&self) -> bool {
        self.missing.is_empty()
    }

    /// Format a human-readable report.
    pub fn report(&self) -> String {
        let status = if self.will_load { "OK" } else { "FAIL" };
        let mut out = format!(
            "Schema validation: {status}\n  layers: {}\n  experts: {:?}\n  required tensors found: {}\n  required tensors missing: {}\n  extra tensors (ignored): {}\n",
            self.num_layers,
            self.num_experts,
            self.found.len(),
            self.missing.len(),
            self.extra.len()
        );
        if !self.missing.is_empty() {
            out.push_str("\n  Missing (will cause load failure):\n");
            for m in &self.missing {
                out.push_str(&format!("    - {m}\n"));
            }
        }
        out
    }
}

/// Validate a safetensors weight-key list against the DeepSeek V4 schema.
///
/// V4-specific tensors required:
/// - `model.layers.<i>.self_attn.o_a_proj.weight` (LoRA-down for o_proj)
/// - `model.layers.<i>.self_attn.o_b_proj.weight` (LoRA-up for o_proj)
///
/// V4 also has optional per-layer compressor tensors when `compress_ratios[i] != 0`:
/// - `model.layers.<i>.self_attn.compressor.weight`
///
/// And MTP heads (V4 ships these):
/// - `mtp.layers.<i>.*`
pub fn validate_v4_weights<'a, I>(keys: I) -> WeightValidation
where
    I: IntoIterator<Item = &'a str>,
{
    let key_set: HashSet<String> = keys.into_iter().map(String::from).collect();
    validate_v4_against_keyset(&key_set)
}

fn validate_v4_against_keyset(keys: &HashSet<String>) -> WeightValidation {
    let num_layers = infer_num_layers(keys);
    let num_experts = infer_num_experts_at_layer(keys, /*layer=*/ 1);

    // Required per-model tensors.
    let mut required: Vec<String> = vec![
        "model.embed_tokens.weight".into(),
        "model.norm.weight".into(),
        "lm_head.weight".into(),
    ];

    // Per-layer V4 tensors.
    for i in 0..num_layers {
        required.extend([
            format!("model.layers.{i}.input_layernorm.weight"),
            format!("model.layers.{i}.post_attention_layernorm.weight"),
            // MLA Q (LoRA): q_a_proj + q_b_proj
            format!("model.layers.{i}.self_attn.q_a_proj.weight"),
            format!("model.layers.{i}.self_attn.q_b_proj.weight"),
            // MLA KV: kv_a_proj_with_mqa + kv_b_proj
            format!("model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"),
            format!("model.layers.{i}.self_attn.kv_b_proj.weight"),
            // V4-specific: o_proj is LoRA-decomposed (o_a + o_b), NOT a single o_proj
            format!("model.layers.{i}.self_attn.o_a_proj.weight"),
            format!("model.layers.{i}.self_attn.o_b_proj.weight"),
        ]);
    }

    classify(keys, required)
}

/// Validate against Kimi K2 (V3-derived) schema.
///
/// K2.5/K2.6 inherit V3 architecture: single o_proj (not LoRA-decomposed),
/// MLA Q/KV with LoRA decomposition.
pub fn validate_kimi_k2_weights<'a, I>(keys: I) -> WeightValidation
where
    I: IntoIterator<Item = &'a str>,
{
    let key_set: HashSet<String> = keys.into_iter().map(String::from).collect();
    let num_layers = infer_num_layers(&key_set);

    let mut required: Vec<String> = vec![
        "model.embed_tokens.weight".into(),
        "model.norm.weight".into(),
        "lm_head.weight".into(),
    ];
    for i in 0..num_layers {
        required.extend([
            format!("model.layers.{i}.input_layernorm.weight"),
            format!("model.layers.{i}.post_attention_layernorm.weight"),
            format!("model.layers.{i}.self_attn.q_a_proj.weight"),
            format!("model.layers.{i}.self_attn.q_b_proj.weight"),
            format!("model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"),
            format!("model.layers.{i}.self_attn.kv_b_proj.weight"),
            // Kimi uses single o_proj (V3-style)
            format!("model.layers.{i}.self_attn.o_proj.weight"),
        ]);
    }

    classify(&key_set, required)
}

/// Validate against GLM-5 DSA schema (V3-MLA + DSA).
pub fn validate_glm5_dsa_weights<'a, I>(keys: I) -> WeightValidation
where
    I: IntoIterator<Item = &'a str>,
{
    // GLM-5 has the same attention layout as V3 (no o_proj LoRA), but DSA
    // attention with grouped-noaux-tc top-k.
    validate_kimi_k2_weights(keys)
}

fn classify(keys: &HashSet<String>, required: Vec<String>) -> WeightValidation {
    let mut found = Vec::new();
    let mut missing = Vec::new();

    for req in &required {
        if keys.contains(req) {
            found.push(req.clone());
        } else {
            missing.push(req.clone());
        }
    }

    let required_set: HashSet<&str> = required.iter().map(String::as_str).collect();
    let extra: Vec<String> = keys
        .iter()
        .filter(|k| !required_set.contains(k.as_str()))
        .cloned()
        .collect();

    let num_layers = infer_num_layers(keys);
    let num_experts = infer_num_experts_at_layer(keys, 1);

    WeightValidation {
        will_load: missing.is_empty(),
        found,
        missing,
        extra,
        num_layers,
        num_experts,
    }
}

fn infer_num_layers(keys: &HashSet<String>) -> usize {
    let mut max_layer: i32 = -1;
    for key in keys {
        if let Some(rest) = key.strip_prefix("model.layers.") {
            if let Some(end) = rest.find('.') {
                if let Ok(idx) = rest[..end].parse::<i32>() {
                    if idx > max_layer {
                        max_layer = idx;
                    }
                }
            }
        }
    }
    (max_layer + 1) as usize
}

fn infer_num_experts_at_layer(keys: &HashSet<String>, layer: usize) -> Option<usize> {
    let prefix = format!("model.layers.{layer}.mlp.experts.");
    let mut max_e: i32 = -1;
    for key in keys {
        if let Some(rest) = key.strip_prefix(&prefix) {
            if let Some(end) = rest.find('.') {
                if let Ok(idx) = rest[..end].parse::<i32>() {
                    if idx > max_e {
                        max_e = idx;
                    }
                }
            }
        }
    }
    if max_e >= 0 {
        Some((max_e + 1) as usize)
    } else {
        None
    }
}

/// Generate a synthetic weight-key list for a V4 model with given layer/expert counts.
/// Useful for tests that need a complete expected-shaped checkpoint without
/// downloading anything.
pub fn synthesize_v4_weight_keys(
    num_layers: usize,
    num_experts: usize,
    include_mtp: bool,
) -> Vec<String> {
    let mut keys = vec![
        "model.embed_tokens.weight".into(),
        "model.norm.weight".into(),
        "lm_head.weight".into(),
    ];
    for i in 0..num_layers {
        keys.push(format!("model.layers.{i}.input_layernorm.weight"));
        keys.push(format!("model.layers.{i}.post_attention_layernorm.weight"));
        keys.push(format!("model.layers.{i}.self_attn.q_a_proj.weight"));
        keys.push(format!("model.layers.{i}.self_attn.q_b_proj.weight"));
        keys.push(format!("model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"));
        keys.push(format!("model.layers.{i}.self_attn.kv_b_proj.weight"));
        keys.push(format!("model.layers.{i}.self_attn.o_a_proj.weight"));
        keys.push(format!("model.layers.{i}.self_attn.o_b_proj.weight"));
        // V4-specific compressor (in real checkpoints, only on CSA/HCA layers)
        if i % 3 != 0 {
            keys.push(format!("model.layers.{i}.self_attn.compressor.weight"));
        }
        // MoE experts
        for e in 0..num_experts {
            keys.push(format!("model.layers.{i}.mlp.experts.{e}.gate_proj.weight"));
            keys.push(format!("model.layers.{i}.mlp.experts.{e}.up_proj.weight"));
            keys.push(format!("model.layers.{i}.mlp.experts.{e}.down_proj.weight"));
        }
        keys.push(format!("model.layers.{i}.mlp.gate.weight"));
    }
    if include_mtp {
        for i in 0..2 {
            keys.push(format!("mtp.layers.{i}.embed.weight"));
            keys.push(format!("mtp.layers.{i}.transformer.weight"));
            keys.push(format!("mtp.layers.{i}.head.weight"));
        }
    }
    keys
}

/// Generate synthetic weight keys for a Kimi K2 / V3-style model.
pub fn synthesize_kimi_k2_weight_keys(num_layers: usize, num_experts: usize) -> Vec<String> {
    let mut keys = vec![
        "model.embed_tokens.weight".into(),
        "model.norm.weight".into(),
        "lm_head.weight".into(),
    ];
    for i in 0..num_layers {
        keys.push(format!("model.layers.{i}.input_layernorm.weight"));
        keys.push(format!("model.layers.{i}.post_attention_layernorm.weight"));
        keys.push(format!("model.layers.{i}.self_attn.q_a_proj.weight"));
        keys.push(format!("model.layers.{i}.self_attn.q_b_proj.weight"));
        keys.push(format!("model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"));
        keys.push(format!("model.layers.{i}.self_attn.kv_b_proj.weight"));
        keys.push(format!("model.layers.{i}.self_attn.o_proj.weight"));
        for e in 0..num_experts {
            keys.push(format!("model.layers.{i}.mlp.experts.{e}.gate_proj.weight"));
            keys.push(format!("model.layers.{i}.mlp.experts.{e}.up_proj.weight"));
            keys.push(format!("model.layers.{i}.mlp.experts.{e}.down_proj.weight"));
        }
        keys.push(format!("model.layers.{i}.mlp.gate.weight"));
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesized_v4_weights_validate_cleanly() {
        let keys = synthesize_v4_weight_keys(4, 8, true);
        let v = validate_v4_weights(keys.iter().map(String::as_str));
        assert!(v.is_valid(), "Validation report:\n{}", v.report());
        assert_eq!(v.num_layers, 4);
        assert_eq!(v.num_experts, Some(8));
        assert!(v.will_load);
    }

    /// **THIS IS THE TEST THAT CATCHES THE V3-LOADER-FOR-V4 BUG.**
    /// V4 weights have `o_a_proj` + `o_b_proj`. A V3 loader expects `o_proj`.
    /// The Kimi-K2 (V3-style) validator should reject V4 weights with a clear
    /// "missing o_proj.weight" error per layer.
    #[test]
    fn v3_loader_rejects_v4_weights_with_clear_error() {
        let v4_keys = synthesize_v4_weight_keys(4, 8, false);
        let v = validate_kimi_k2_weights(v4_keys.iter().map(String::as_str));
        assert!(!v.is_valid(), "V3 loader should fail on V4 weights");
        // Should report each layer's missing o_proj.weight
        let missing_o_proj = v.missing.iter().filter(|m| m.ends_with(".self_attn.o_proj.weight")).count();
        assert_eq!(missing_o_proj, 4, "Expected 4 missing o_proj entries");
    }

    /// Conversely, V4 validator should reject V3 weights (missing o_a/o_b).
    #[test]
    fn v4_loader_rejects_v3_weights_with_clear_error() {
        let v3_keys = synthesize_kimi_k2_weight_keys(4, 8);
        let v = validate_v4_weights(v3_keys.iter().map(String::as_str));
        assert!(!v.is_valid(), "V4 loader should fail on V3 weights");
        let missing_o_a = v.missing.iter().filter(|m| m.ends_with("o_a_proj.weight")).count();
        let missing_o_b = v.missing.iter().filter(|m| m.ends_with("o_b_proj.weight")).count();
        assert_eq!(missing_o_a, 4);
        assert_eq!(missing_o_b, 4);
    }

    #[test]
    fn kimi_k2_weights_validate_cleanly() {
        let keys = synthesize_kimi_k2_weight_keys(4, 8);
        let v = validate_kimi_k2_weights(keys.iter().map(String::as_str));
        assert!(v.is_valid(), "Validation report:\n{}", v.report());
        assert_eq!(v.num_layers, 4);
        assert_eq!(v.num_experts, Some(8));
    }

    #[test]
    fn glm5_weights_validate_cleanly() {
        // GLM-5 has same attention layout as V3 / Kimi K2
        let keys = synthesize_kimi_k2_weight_keys(4, 8);
        let v = validate_glm5_dsa_weights(keys.iter().map(String::as_str));
        assert!(v.is_valid());
    }

    #[test]
    fn missing_global_tensors_caught() {
        // Just layer weights, no embed_tokens / norm / lm_head
        let mut keys = synthesize_v4_weight_keys(4, 4, false);
        keys.retain(|k| !k.contains("embed_tokens") && !k.contains("model.norm") && !k.contains("lm_head"));
        let v = validate_v4_weights(keys.iter().map(String::as_str));
        assert!(!v.is_valid());
        assert!(v.missing.iter().any(|m| m.contains("embed_tokens")));
        assert!(v.missing.iter().any(|m| m.contains("model.norm")));
        assert!(v.missing.iter().any(|m| m.contains("lm_head")));
    }

    #[test]
    fn num_layers_inferred_correctly() {
        let keys = synthesize_v4_weight_keys(43, 256, false);
        let v = validate_v4_weights(keys.iter().map(String::as_str));
        assert_eq!(v.num_layers, 43);
        assert_eq!(v.num_experts, Some(256));
    }

    #[test]
    fn validation_report_is_informative() {
        let keys = synthesize_kimi_k2_weight_keys(2, 4);
        let v = validate_v4_weights(keys.iter().map(String::as_str));
        let report = v.report();
        assert!(report.contains("FAIL"));
        assert!(report.contains("Missing"));
        assert!(report.contains("o_a_proj") || report.contains("o_b_proj"));
    }

    #[test]
    fn extra_tensors_dont_cause_failure() {
        let mut keys = synthesize_v4_weight_keys(2, 4, false);
        keys.push("model.layers.0.random_extra_tensor.weight".into());
        keys.push("model.unknown_block.foo".into());
        let v = validate_v4_weights(keys.iter().map(String::as_str));
        assert!(v.is_valid(), "Extras should not cause load failure");
        assert!(!v.extra.is_empty());
    }
}
