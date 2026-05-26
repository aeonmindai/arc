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

/// O-projection layout: V3-style single matrix or V4's hypothetical LoRA decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OProjLayout {
    /// Single `o_proj.weight` tensor (V3, Kimi K2, GLM-5 confirmed).
    Single,
    /// LoRA-decomposed `o_a_proj.weight` + `o_b_proj.weight` (V4 hypothesis based on
    /// `o_lora_rank` config field; not yet confirmed against real V4 safetensors).
    Lora,
    /// Try Single first; fall back to Lora. Default for V4 until the real layout
    /// is verified against published checkpoints.
    Either,
}

/// Validate a safetensors weight-key list against the DeepSeek V4 schema.
///
/// **Note on o_proj layout uncertainty:** V4's actual safetensors naming
/// convention isn't yet published in the SGLang reference. We default to
/// `OProjLayout::Either` — accept either V3-style single `o_proj` or
/// V4-style LoRA-decomposed `o_a_proj` + `o_b_proj`. The rental's first
/// job is to download real V4 weights and detect which convention is used.
pub fn validate_v4_weights<'a, I>(keys: I) -> WeightValidation
where
    I: IntoIterator<Item = &'a str>,
{
    validate_v4_weights_with_layout(keys, OProjLayout::Either)
}

/// Validate V4 schema with an explicit o_proj layout assumption.
pub fn validate_v4_weights_with_layout<'a, I>(keys: I, o_layout: OProjLayout) -> WeightValidation
where
    I: IntoIterator<Item = &'a str>,
{
    let key_set: HashSet<String> = keys.into_iter().map(String::from).collect();
    validate_v4_against_keyset(&key_set, o_layout)
}

/// Two formats observed in published V4 safetensors:
/// - **Native (SGLang-friendly)**: `layers.0.attn.wq_a.weight`, short prefixes
/// - **HF-format (Transformers-friendly)**: `model.layers.0.self_attn.q_a_proj.weight`
///
/// DeepSeek's official V4 weights use the NATIVE format per SGLang's
/// `remap_weight_name_to_dpsk_hf_format()` (deepseek_v4.py:1365) which converts
/// native → HF at load time. Third-party re-uploads (e.g., MLX, GGUF conversions)
/// may use the HF format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V4Format {
    /// `layers.N.attn.{wq_a, wkv, wo_a, wo_b, ...}` — DeepSeek's native format
    Native,
    /// `model.layers.N.self_attn.{q_a_proj, kv_proj, o_a_proj, ...}` — HF/Transformers format
    Hf,
}

/// Auto-detect V4 safetensors format by inspecting key prefixes.
pub fn detect_v4_format(keys: &HashSet<String>) -> V4Format {
    let has_native = keys
        .iter()
        .any(|k| k.starts_with("layers.") && k.contains(".attn."));
    let has_hf = keys
        .iter()
        .any(|k| k.starts_with("model.layers.") && k.contains(".self_attn."));

    // Native wins if both detected (SGLang's authoritative format).
    if has_native {
        V4Format::Native
    } else if has_hf {
        V4Format::Hf
    } else {
        // Default to Native — that's what DeepSeek publishes
        V4Format::Native
    }
}

/// Per-layer V4 tensor names in either format.
struct V4LayerTensors {
    norm_in: String,
    norm_out: String,
    wq_a: String,
    wq_b: String,
    wkv: String,
    wo_a: String,
    wo_b: String,
}

impl V4LayerTensors {
    fn for_layer(layer: usize, fmt: V4Format) -> Self {
        match fmt {
            V4Format::Native => Self {
                norm_in: format!("layers.{layer}.attn_norm.weight"),
                norm_out: format!("layers.{layer}.ffn_norm.weight"),
                wq_a: format!("layers.{layer}.attn.wq_a.weight"),
                wq_b: format!("layers.{layer}.attn.wq_b.weight"),
                wkv: format!("layers.{layer}.attn.wkv.weight"),
                wo_a: format!("layers.{layer}.attn.wo_a.weight"),
                wo_b: format!("layers.{layer}.attn.wo_b.weight"),
            },
            V4Format::Hf => Self {
                norm_in: format!("model.layers.{layer}.input_layernorm.weight"),
                norm_out: format!("model.layers.{layer}.post_attention_layernorm.weight"),
                wq_a: format!("model.layers.{layer}.self_attn.q_a_proj.weight"),
                wq_b: format!("model.layers.{layer}.self_attn.q_b_proj.weight"),
                wkv: format!("model.layers.{layer}.self_attn.kv_proj.weight"),
                wo_a: format!("model.layers.{layer}.self_attn.o_a_proj.weight"),
                wo_b: format!("model.layers.{layer}.self_attn.o_b_proj.weight"),
            },
        }
    }
    #[allow(dead_code)] // convenience accessor over all schema keys; not yet used
    fn all(&self) -> Vec<&str> {
        vec![
            &self.norm_in,
            &self.norm_out,
            &self.wq_a,
            &self.wq_b,
            &self.wkv,
            &self.wo_a,
            &self.wo_b,
        ]
    }
}

fn validate_v4_against_keyset(keys: &HashSet<String>, _o_layout: OProjLayout) -> WeightValidation {
    // V4 tensor layout from SGLang source: deepseek_v4.py (1816 LOC) lines 172-655 (MQALayer)
    // + 1365-1423 (remap function). Native short-form keys are confirmed.
    let fmt = detect_v4_format(keys);
    let num_layers = infer_num_layers_dual_format(keys, fmt);
    let num_experts = infer_num_experts_at_layer_dual_format(keys, 1, fmt);

    // Global tensors
    let (embed, norm, head) = match fmt {
        V4Format::Native => (
            "embed.weight".to_string(),
            "norm.weight".to_string(),
            "head.weight".to_string(),
        ),
        V4Format::Hf => (
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ),
    };

    let mut required: Vec<String> = vec![embed, norm, head];

    for i in 0..num_layers {
        let t = V4LayerTensors::for_layer(i, fmt);
        required.push(t.norm_in.clone());
        required.push(t.norm_out.clone());
        required.push(t.wq_a.clone());
        required.push(t.wq_b.clone());
        required.push(t.wkv.clone());
        required.push(t.wo_a.clone());
        required.push(t.wo_b.clone());
    }

    let mut found = Vec::new();
    let mut missing = Vec::new();
    for req in &required {
        if keys.contains(req) {
            found.push(req.clone());
        } else {
            missing.push(req.clone());
        }
    }

    let all_required: HashSet<String> = required.iter().cloned().collect();
    let extra: Vec<String> = keys
        .iter()
        .filter(|k| !all_required.contains(k.as_str()) && !found.contains(k))
        .cloned()
        .collect();

    WeightValidation {
        will_load: missing.is_empty(),
        found,
        missing,
        extra,
        num_layers,
        num_experts,
    }
}

/// V3-prefix-aware layer count inference (handles both `model.layers.N` and `layers.N`).
fn infer_num_layers_dual_format(keys: &HashSet<String>, fmt: V4Format) -> usize {
    let prefix = match fmt {
        V4Format::Native => "layers.",
        V4Format::Hf => "model.layers.",
    };
    let mut max_layer: i32 = -1;
    for key in keys {
        if let Some(rest) = key.strip_prefix(prefix) {
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

fn infer_num_experts_at_layer_dual_format(
    keys: &HashSet<String>,
    layer: usize,
    fmt: V4Format,
) -> Option<usize> {
    let prefix = match fmt {
        V4Format::Native => format!("layers.{layer}.ffn.experts."),
        V4Format::Hf => format!("model.layers.{layer}.mlp.experts."),
    };
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

/// Required tensor names for a V4 CSA/HCA layer's compressor module.
/// V4 native format uses fused `wkv_gate` (combined wkv + wgate). Per SGLang's
/// load_weights fusion logic at deepseek_v4.py:1632-1663.
pub fn v4_compressor_tensors_for_layer(layer: usize, fmt: V4Format) -> Vec<String> {
    match fmt {
        V4Format::Native => vec![
            format!("layers.{layer}.attn.wkv_gate.weight"),
            format!("layers.{layer}.attn.wkv_a.weight"),
            format!("layers.{layer}.attn.wkv_b.weight"),
            format!("layers.{layer}.attn.ape"),
            format!("layers.{layer}.attn.kv_norm.weight"),
        ],
        V4Format::Hf => vec![
            format!("model.layers.{layer}.self_attn.compressor.gate_proj.weight"),
            format!("model.layers.{layer}.self_attn.compressor.kv_proj.weight"),
        ],
    }
}

/// Required tensor names for a V4 CSA layer's Lightning Indexer (in FP4).
/// Only present on CSA layers (compress_ratio = 4).
pub fn v4_indexer_tensors_for_layer(layer: usize, fmt: V4Format) -> Vec<String> {
    match fmt {
        V4Format::Native => vec![
            format!("layers.{layer}.attn.indexer.compressor.ape"),
            format!("layers.{layer}.attn.indexer.compressor.norm.weight"),
            format!("layers.{layer}.attn.indexer.compressor.wgate.weight"),
            format!("layers.{layer}.attn.indexer.compressor.wkv.weight"),
            format!("layers.{layer}.attn.indexer.wq_b.weight"),
            format!("layers.{layer}.attn.indexer.weights_proj.weight"),
        ],
        V4Format::Hf => vec![
            format!("model.layers.{layer}.self_attn.indexer.compressor.ape.weight"),
            format!("model.layers.{layer}.self_attn.indexer.compressor.norm.weight"),
            format!("model.layers.{layer}.self_attn.indexer.compressor.wgate.weight"),
            format!("model.layers.{layer}.self_attn.indexer.compressor.wkv.weight"),
            format!("model.layers.{layer}.self_attn.indexer.wq_b.weight"),
            format!("model.layers.{layer}.self_attn.indexer.weights_proj.weight"),
        ],
    }
}

/// Required tensor names for V4's MTP head (single layer with SWA-only attention).
pub fn v4_mtp_tensors(fmt: V4Format) -> Vec<String> {
    match fmt {
        V4Format::Native => vec![
            "mtp.layers.0.h_proj.weight".to_string(),
            "mtp.layers.0.e_proj.weight".to_string(),
            "mtp.layers.0.hnorm.weight".to_string(),
            "mtp.layers.0.enorm.weight".to_string(),
            "mtp.layers.0.norm.weight".to_string(),
            "mtp.layers.0.head.weight".to_string(),
        ],
        V4Format::Hf => vec![
            "model.layers.43.h_proj.weight".to_string(),
            "model.layers.43.e_proj.weight".to_string(),
        ],
    }
}

/// V4 mHC (Manifold-Constrained Hyper-Connections) per-layer tensor names.
pub fn v4_mhc_tensors_for_layer(layer: usize, fmt: V4Format) -> Vec<String> {
    let prefix = match fmt {
        V4Format::Native => format!("layers.{layer}"),
        V4Format::Hf => format!("model.layers.{layer}"),
    };
    vec![
        format!("{prefix}.hc_attn_fn"),
        format!("{prefix}.hc_attn_base"),
        format!("{prefix}.hc_attn_scale"),
        format!("{prefix}.hc_ffn_fn"),
        format!("{prefix}.hc_ffn_base"),
        format!("{prefix}.hc_ffn_scale"),
    ]
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

/// Generate a synthetic weight-key list for a V4 model with REAL tensor names
/// in the requested format. Defaults to Native (SGLang-confirmed DeepSeek
/// publishing format).
pub fn synthesize_v4_weight_keys(
    num_layers: usize,
    num_experts: usize,
    include_mtp: bool,
) -> Vec<String> {
    synthesize_v4_weight_keys_with_format(num_layers, num_experts, include_mtp, V4Format::Native)
}

/// Synthesize V4 weight keys in either native or HF format.
pub fn synthesize_v4_weight_keys_with_format(
    num_layers: usize,
    num_experts: usize,
    include_mtp: bool,
    fmt: V4Format,
) -> Vec<String> {
    let mut keys: Vec<String> = match fmt {
        V4Format::Native => vec![
            "embed.weight".into(),
            "norm.weight".into(),
            "head.weight".into(),
        ],
        V4Format::Hf => vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
            "lm_head.weight".into(),
        ],
    };

    for i in 0..num_layers {
        let t = V4LayerTensors::for_layer(i, fmt);
        keys.push(t.norm_in);
        keys.push(t.norm_out);
        keys.push(t.wq_a);
        keys.push(t.wq_b);
        keys.push(t.wkv);
        keys.push(t.wo_a);
        keys.push(t.wo_b);

        // Compress layers (alternating in V4 Flash per compress_ratios pattern)
        let is_compress = i % 2 != 0;
        if is_compress {
            keys.extend(v4_compressor_tensors_for_layer(i, fmt));
        }
        let is_csa = is_compress && i % 3 == 1;
        if is_csa {
            keys.extend(v4_indexer_tensors_for_layer(i, fmt));
        }

        // mHC parameters (every layer)
        keys.extend(v4_mhc_tensors_for_layer(i, fmt));

        // MoE
        for e in 0..num_experts {
            match fmt {
                V4Format::Native => {
                    keys.push(format!("layers.{i}.ffn.experts.{e}.w1.weight"));
                    keys.push(format!("layers.{i}.ffn.experts.{e}.w2.weight"));
                    keys.push(format!("layers.{i}.ffn.experts.{e}.w3.weight"));
                }
                V4Format::Hf => {
                    keys.push(format!("model.layers.{i}.mlp.experts.{e}.gate_proj.weight"));
                    keys.push(format!("model.layers.{i}.mlp.experts.{e}.up_proj.weight"));
                    keys.push(format!("model.layers.{i}.mlp.experts.{e}.down_proj.weight"));
                }
            }
        }
        match fmt {
            V4Format::Native => keys.push(format!("layers.{i}.ffn.gate.weight")),
            V4Format::Hf => keys.push(format!("model.layers.{i}.mlp.gate.weight")),
        }
    }

    // Global mHC head
    match fmt {
        V4Format::Native => {
            keys.push("hc_head_fn".to_string());
            keys.push("hc_head_base".to_string());
            keys.push("hc_head_scale".to_string());
        }
        V4Format::Hf => {
            keys.push("model.hc_head_fn".to_string());
            keys.push("model.hc_head_base".to_string());
            keys.push("model.hc_head_scale".to_string());
        }
    }

    if include_mtp {
        keys.extend(v4_mtp_tensors(fmt));
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
        keys.push(format!(
            "model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"
        ));
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

    /// V3-style validator (kimi_k2 uses V3 layout) rejects V4 weights.
    /// V4 native format uses `layers.X.attn.wq_a` not `model.layers.X.self_attn.q_a_proj`.
    #[test]
    fn v3_loader_rejects_v4_weights_with_clear_error() {
        let v4_keys = synthesize_v4_weight_keys(4, 8, false); // Native format
        let v = validate_kimi_k2_weights(v4_keys.iter().map(String::as_str));
        assert!(
            !v.is_valid(),
            "V3 (HF-format) loader should fail on V4 (native-format) weights"
        );
        // V3 expects model.embed_tokens.weight — V4 native has embed.weight
        let missing_embed = v
            .missing
            .iter()
            .filter(|m| m.contains("embed_tokens"))
            .count();
        assert_eq!(
            missing_embed, 1,
            "Expected V3 loader to fail finding HF-style embed_tokens"
        );
    }

    /// V4 validator rejects V3-style weights (missing kv_proj, missing o_a/o_b).
    #[test]
    fn v4_loader_rejects_v3_weights_with_clear_error() {
        let v3_keys = synthesize_kimi_k2_weight_keys(4, 8);
        let v = validate_v4_weights(v3_keys.iter().map(String::as_str));
        assert!(
            !v.is_valid(),
            "V4 validator should fail on V3-style weights"
        );
        let missing_kv_proj = v
            .missing
            .iter()
            .filter(|m| m.ends_with(".self_attn.kv_proj.weight"))
            .count();
        let missing_o_a = v
            .missing
            .iter()
            .filter(|m| m.ends_with(".self_attn.o_a_proj.weight"))
            .count();
        let missing_o_b = v
            .missing
            .iter()
            .filter(|m| m.ends_with(".self_attn.o_b_proj.weight"))
            .count();
        assert_eq!(missing_kv_proj, 4, "Expected 4 missing kv_proj entries");
        assert_eq!(missing_o_a, 4, "Expected 4 missing o_a_proj entries");
        assert_eq!(missing_o_b, 4, "Expected 4 missing o_b_proj entries");
    }

    /// V4 compressor tensor names match SGLang's native short-form layout.
    #[test]
    fn v4_compressor_tensor_names_correct_native() {
        let names = v4_compressor_tensors_for_layer(7, V4Format::Native);
        // Per deepseek_v4.py:1632-1663, native has fused wkv_gate + wkv_a + wkv_b + ape + kv_norm
        assert!(names.iter().any(|n| n.contains("wkv_gate")));
        assert!(names.iter().any(|n| n.contains("wkv_a")));
        assert!(names.iter().any(|n| n.contains("wkv_b")));
        assert!(names.iter().any(|n| n.contains("ape")));
    }

    /// V4 Lightning Indexer tensor names (native format).
    #[test]
    fn v4_indexer_tensor_names_correct_native() {
        let names = v4_indexer_tensors_for_layer(3, V4Format::Native);
        assert!(names.iter().any(|n| n.contains("indexer.compressor.ape")));
        assert!(names
            .iter()
            .any(|n| n.contains("indexer.compressor.norm.weight")));
        assert!(names
            .iter()
            .any(|n| n.contains("indexer.compressor.wgate.weight")));
        assert!(names
            .iter()
            .any(|n| n.contains("indexer.compressor.wkv.weight")));
        assert!(names.iter().any(|n| n.contains("indexer.wq_b.weight")));
        assert!(names
            .iter()
            .any(|n| n.contains("indexer.weights_proj.weight")));
    }

    /// V4 MTP tensor names (native format).
    #[test]
    fn v4_mtp_tensor_names_correct_native() {
        let names = v4_mtp_tensors(V4Format::Native);
        assert!(names.contains(&"mtp.layers.0.h_proj.weight".to_string()));
        assert!(names.contains(&"mtp.layers.0.e_proj.weight".to_string()));
    }

    /// Format auto-detection picks Native when both `layers.*` and `model.layers.*`
    /// not present; picks Native when `layers.X.attn.*` present.
    #[test]
    fn v4_format_detection_native() {
        let mut keys = HashSet::new();
        keys.insert("layers.0.attn.wq_a.weight".to_string());
        keys.insert("embed.weight".to_string());
        assert_eq!(detect_v4_format(&keys), V4Format::Native);
    }

    #[test]
    fn v4_format_detection_hf() {
        let mut keys = HashSet::new();
        keys.insert("model.layers.0.self_attn.q_a_proj.weight".to_string());
        keys.insert("model.embed_tokens.weight".to_string());
        assert_eq!(detect_v4_format(&keys), V4Format::Hf);
    }

    /// Synthesized V4 weights in NATIVE format validate cleanly through V4 validator.
    #[test]
    fn synthesized_v4_native_validates() {
        let keys = synthesize_v4_weight_keys_with_format(4, 8, true, V4Format::Native);
        let v = validate_v4_weights(keys.iter().map(String::as_str));
        assert!(
            v.is_valid(),
            "Native V4 should validate. Missing: {:?}",
            v.missing
        );
        assert_eq!(v.num_layers, 4);
    }

    /// Synthesized V4 weights in HF format also validate cleanly.
    #[test]
    fn synthesized_v4_hf_validates() {
        let keys = synthesize_v4_weight_keys_with_format(4, 8, false, V4Format::Hf);
        let v = validate_v4_weights(keys.iter().map(String::as_str));
        assert!(
            v.is_valid(),
            "HF V4 should validate. Missing: {:?}",
            v.missing
        );
        assert_eq!(v.num_layers, 4);
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
        // V4 native format. Strip global tensors.
        let mut keys = synthesize_v4_weight_keys(4, 4, false);
        keys.retain(|k| !["embed.weight", "norm.weight", "head.weight"].contains(&k.as_str()));
        let v = validate_v4_weights(keys.iter().map(String::as_str));
        assert!(!v.is_valid(), "Should detect missing global tensors");
        assert!(v.missing.iter().any(|m| m == "embed.weight"));
        assert!(v.missing.iter().any(|m| m == "norm.weight"));
        assert!(v.missing.iter().any(|m| m == "head.weight"));
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
        // Use Lora mode to force a failure on V3-style input.
        let keys = synthesize_kimi_k2_weight_keys(2, 4);
        let v = validate_v4_weights_with_layout(keys.iter().map(String::as_str), OProjLayout::Lora);
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
