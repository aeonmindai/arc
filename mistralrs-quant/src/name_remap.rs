//! Tensor name remapping backend for VarBuilder.
//!
//! Provides a [`SimpleBackend`] wrapper that translates tensor-name lookups
//! according to a list of rewrite rules before delegating to an inner backend.
//!
//! ### Primary use case: DeepSeek V4 FP8 scale tensors
//!
//! V4 publishes FP8 block scales under `<weight>.scale` (e.g.
//! `layers.0.attn.wq_a.scale`). Arc's [`BlockwiseFP8Linear`] loader expects
//! the HF-convention name `<weight>.weight_scale_inv`. The
//! [`scale_to_weight_scale_inv_rules`] helper produces the standard V4 ruleset
//! covering `self_attn.*`, `mlp.*`, the V4-native `attn.*` / `ffn.*` prefixes,
//! and a handful of top-level cases (`h_proj.scale`, `e_proj.scale`).
//!
//! ### Layered design
//!
//! Wraps any `SimpleBackend` (including the sharded mmap backend used for
//! real model loads, or a `HashMap<String, Tensor>` for tests). The output is
//! a `Box<dyn SimpleBackend>` you can hand to
//! [`crate::ShardedSafeTensors::wrap`] to obtain a [`ShardedVarBuilder`].
//!
//! ### Semantics
//!
//! For each `get`/`contains_tensor` query with name `N`:
//!
//! 1. Try `N` directly on the inner backend.
//! 2. If miss, apply every rule's `matches(N)` predicate to derive an
//!    alternate name `N'` and retry. The first hit wins.
//!
//! This means a model that uses the HF-convention name (`weight_scale_inv`)
//! transparently reads from the V4-native (`scale`) tensor, without the model
//! code needing to know which naming the on-disk checkpoint uses.
//!
//! [`SimpleBackend`]: candle_nn::var_builder::SimpleBackend
//! [`BlockwiseFP8Linear`]: crate::BlockwiseFP8Linear
//! [`ShardedVarBuilder`]: crate::ShardedVarBuilder

use std::sync::Arc;

use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;

use crate::ShardedVarBuilder;

/// A single rewrite rule. Rewrites a tensor name by replacing the matched
/// `from` suffix at any position where it occurs, producing the alternate
/// candidate name. We use simple suffix-replace (not regex) so the rules
/// are cheap and order-stable.
#[derive(Debug, Clone)]
pub struct RenameRule {
    /// Substring to look for in the original name (e.g. `.weight_scale_inv`).
    pub from: String,
    /// Replacement substring (e.g. `.scale`).
    pub to: String,
    /// If `Some`, the rule only fires when the original name contains this
    /// scoping substring. e.g. `Some("self_attn")` means the rule only fires
    /// for attention tensors. `None` means "always".
    pub scope: Option<String>,
}

impl RenameRule {
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            scope: None,
        }
    }

    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    /// Return the rewritten name if this rule matches; else `None`.
    pub fn apply(&self, name: &str) -> Option<String> {
        if let Some(scope) = &self.scope {
            if !name.contains(scope.as_str()) {
                return None;
            }
        }
        if name.contains(&self.from) {
            Some(name.replace(&self.from, &self.to))
        } else {
            None
        }
    }
}

/// The V4 standard ruleset: `.weight_scale_inv` → `.scale`.
///
/// We use unscoped rules because V4 publishes `<tensor>.scale` for every
/// FP8 weight regardless of which subtree it lives under (attn, MoE experts,
/// shared experts, indexer, MTP, h_proj, e_proj). The single rule covers all
/// the SGLang `remap_weight_name_to_dpsk_hf_format` cases at lines 346, 347,
/// 358, 366.
pub fn v4_scale_rename_rules() -> Vec<RenameRule> {
    vec![
        RenameRule::new(".weight_scale_inv", ".scale"),
        // HF convention → V4 native expert projection names.
        // HF uses gate_proj/up_proj/down_proj; V4 uses w1/w3/w2.
        RenameRule::new(".gate_proj.", ".w1."),
        RenameRule::new(".up_proj.", ".w3."),
        RenameRule::new(".down_proj.", ".w2."),
        // Composite rules: rename rules don't chain, so we need combined
        // entries for expert scale tensors (code asks for e.g.
        // `gate_proj.weight_scale_inv`, disk has `w1.scale`).
        RenameRule::new(".gate_proj.weight_scale_inv", ".w1.scale"),
        RenameRule::new(".up_proj.weight_scale_inv", ".w3.scale"),
        RenameRule::new(".down_proj.weight_scale_inv", ".w2.scale"),
    ]
}

/// Backend that wraps another `SimpleBackend` and applies rename rules
/// when the underlying tensor name is not found directly.
pub struct RenamingBackend {
    inner: Box<dyn SimpleBackend + 'static>,
    rules: Vec<RenameRule>,
}

impl RenamingBackend {
    pub fn new(inner: Box<dyn SimpleBackend + 'static>, rules: Vec<RenameRule>) -> Self {
        Self { inner, rules }
    }

    pub fn into_inner(self) -> Box<dyn SimpleBackend + 'static> {
        self.inner
    }

    /// Walk the rules in order and return the first alternate name whose
    /// rule produces a hit. Returns `None` if no rule applies (in which case
    /// the caller may fall back to the original name).
    fn rewrite(&self, name: &str) -> Option<String> {
        for r in &self.rules {
            if let Some(n) = r.apply(name) {
                return Some(n);
            }
        }
        None
    }
}

/// Backend wrapper that adapts a [`ShardedVarBuilder`] into a
/// [`SimpleBackend`] while applying rename rules. Used by
/// [`crate::safetensors::attach_rename_rules`] to graft rename rules onto
/// an existing var-builder regardless of which backend variant it carries
/// (sharded mmap, hashmap, etc.).
pub(crate) struct VarBuilderRenameWrapper {
    inner: ShardedVarBuilder,
    rules: Arc<Vec<RenameRule>>,
}

impl VarBuilderRenameWrapper {
    pub(crate) fn new(inner: ShardedVarBuilder, rules: Arc<Vec<RenameRule>>) -> Self {
        Self { inner, rules }
    }

    fn rewrite(&self, name: &str) -> Option<String> {
        for r in self.rules.as_ref() {
            if let Some(alt) = r.apply(name) {
                if self.inner.contains_tensor(alt.as_str()) {
                    return Some(alt);
                }
            }
        }
        None
    }

    fn pick(&self, name: &str) -> Option<String> {
        if self.inner.contains_tensor(name) {
            return Some(name.to_string());
        }
        if let Some(alt) = self.rewrite(name) {
            tracing::trace!("safetensors rename: {} -> {}", name, alt);
            return Some(alt);
        }
        None
    }
}

impl SimpleBackend for VarBuilderRenameWrapper {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let resolved = self.pick(name).unwrap_or_else(|| name.to_string());
        let t = self
            .inner
            .get_with_hints_dtype(s, resolved.as_str(), Default::default(), dtype)?;
        // Honor the requested device. The inner var-builder typically
        // already loads on the correct device, but if we're hopping across
        // devices for ISQ staging, ensure the device matches.
        if t.device().same_device(dev) {
            Ok(t)
        } else {
            t.to_device(dev)
        }
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        let resolved = self.pick(name).unwrap_or_else(|| name.to_string());
        let t = self.inner.get_unchecked_dtype(resolved.as_str(), dtype)?;
        if t.device().same_device(dev) {
            Ok(t)
        } else {
            t.to_device(dev)
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.pick(name).is_some()
    }
}

impl SimpleBackend for RenamingBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        if self.inner.contains_tensor(name) {
            return self.inner.get(s, name, h, dtype, dev);
        }
        if let Some(alt) = self.rewrite(name) {
            if self.inner.contains_tensor(alt.as_str()) {
                tracing::trace!("renaming-backend: {} -> {} (load-time rewrite)", name, alt);
                return self.inner.get(s, alt.as_str(), h, dtype, dev);
            }
        }
        // No alt resolved — let the inner backend raise its standard
        // "not found" error.
        self.inner.get(s, name, h, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        if self.inner.contains_tensor(name) {
            return self.inner.get_unchecked(name, dtype, dev);
        }
        if let Some(alt) = self.rewrite(name) {
            if self.inner.contains_tensor(alt.as_str()) {
                tracing::trace!("renaming-backend: {} -> {} (load-time rewrite)", name, alt);
                return self.inner.get_unchecked(alt.as_str(), dtype, dev);
            }
        }
        self.inner.get_unchecked(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        if self.inner.contains_tensor(name) {
            return true;
        }
        if let Some(alt) = self.rewrite(name) {
            return self.inner.contains_tensor(alt.as_str());
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use std::collections::HashMap;

    fn mk_inner() -> Box<dyn SimpleBackend + 'static> {
        let dev = Device::Cpu;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        // V4-native naming: weight + scale
        m.insert(
            "model.layers.0.self_attn.wq_a.weight".to_string(),
            Tensor::zeros((1024, 4096), DType::F32, &dev).unwrap(),
        );
        m.insert(
            "model.layers.0.self_attn.wq_a.scale".to_string(),
            Tensor::ones((8, 32), DType::F32, &dev).unwrap(),
        );
        // Already-HF-native naming: weight + weight_scale_inv (don't break it)
        m.insert(
            "model.layers.0.self_attn.kv_norm.weight".to_string(),
            Tensor::ones((512,), DType::F32, &dev).unwrap(),
        );
        m.insert(
            "model.layers.0.mlp.experts.0.w1.weight".to_string(),
            Tensor::zeros((11264, 4096), DType::F32, &dev).unwrap(),
        );
        m.insert(
            "model.layers.0.mlp.experts.0.w1.scale".to_string(),
            Tensor::ones((88, 32), DType::F32, &dev).unwrap(),
        );
        Box::new(m)
    }

    #[test]
    fn rule_apply_basic() {
        let r = RenameRule::new(".weight_scale_inv", ".scale");
        assert_eq!(
            r.apply("model.layers.0.attn.wq_a.weight_scale_inv"),
            Some("model.layers.0.attn.wq_a.scale".to_string())
        );
        // No match: name doesn't contain the `from` substring
        assert_eq!(r.apply("model.layers.0.attn.wq_a.weight"), None);
    }

    #[test]
    fn rule_apply_with_scope() {
        let r = RenameRule::new(".weight_scale_inv", ".scale").with_scope("self_attn");
        assert!(r
            .apply("model.layers.0.self_attn.wq_a.weight_scale_inv")
            .is_some());
        // Scope mismatch (mlp not self_attn) → None
        assert!(r
            .apply("model.layers.0.mlp.experts.0.w1.weight_scale_inv")
            .is_none());
    }

    #[test]
    fn rename_backend_falls_through_for_existing_name() {
        let inner = mk_inner();
        let rb = RenamingBackend::new(inner, v4_scale_rename_rules());
        assert!(rb.contains_tensor("model.layers.0.self_attn.kv_norm.weight"));
        // Asking for `weight` should NOT trigger rename (it's there
        // directly).
        let t = rb
            .get_unchecked(
                "model.layers.0.self_attn.kv_norm.weight",
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(t.dims(), &[512]);
    }

    #[test]
    fn rename_backend_translates_weight_scale_inv_to_scale() {
        let inner = mk_inner();
        let rb = RenamingBackend::new(inner, v4_scale_rename_rules());
        // V4 model code asks for `weight_scale_inv`; backend serves
        // V4-native `.scale`.
        assert!(rb.contains_tensor("model.layers.0.self_attn.wq_a.weight_scale_inv"));
        let t = rb
            .get_unchecked(
                "model.layers.0.self_attn.wq_a.weight_scale_inv",
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        // Picked up the `.scale` shape [8, 32].
        assert_eq!(t.dims(), &[8, 32]);
    }

    #[test]
    fn rename_backend_works_for_mlp_path() {
        let inner = mk_inner();
        let rb = RenamingBackend::new(inner, v4_scale_rename_rules());
        // MLP path also covered by the unscoped rule.
        assert!(rb.contains_tensor("model.layers.0.mlp.experts.0.w1.weight_scale_inv"));
        let t = rb
            .get_unchecked(
                "model.layers.0.mlp.experts.0.w1.weight_scale_inv",
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(t.dims(), &[88, 32]);
    }

    #[test]
    fn rename_backend_reports_missing_correctly() {
        let inner = mk_inner();
        let rb = RenamingBackend::new(inner, v4_scale_rename_rules());
        assert!(!rb.contains_tensor("model.layers.99.never.exists.weight_scale_inv"));
        assert!(rb
            .get_unchecked(
                "model.layers.99.never.exists.weight_scale_inv",
                DType::F32,
                &Device::Cpu
            )
            .is_err());
    }

    /// End-to-end check: a `ShardedVarBuilder` carrying V4-style `.scale`
    /// tensors becomes usable as a HF-style `.weight_scale_inv` view via
    /// [`crate::attach_rename_rules`].
    ///
    /// This exercises the wrap-the-VarBuilder path that the
    /// `DeepSeekV4Loader` uses in production. It does not call into
    /// `blockwise_fp8_linear_b` directly (that requires F8E4M3 storage,
    /// which we'd need to synthesize from F8E4M3 raw bytes — overkill for
    /// this test), but it does prove the contract:
    ///
    /// > A vb wrapped with `attach_rename_rules` returns the V4-native
    /// > `.scale` tensor when asked for the HF-style `.weight_scale_inv`.
    #[test]
    fn attach_rename_rules_end_to_end() {
        let dev = Device::Cpu;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        // V4-native: an FP8 weight + companion scale.
        m.insert(
            "layers.0.self_attn.wq_a.weight".to_string(),
            Tensor::zeros((1024, 4096), DType::F32, &dev).unwrap(),
        );
        m.insert(
            "layers.0.self_attn.wq_a.scale".to_string(),
            Tensor::ones((8, 32), DType::F32, &dev).unwrap(),
        );
        // mHC tensor that should NOT be remapped (no `.scale` to rewrite,
        // direct hit).
        m.insert(
            "layers.0.hc_attn_scale".to_string(),
            Tensor::ones((3,), DType::F32, &dev).unwrap(),
        );
        let backend: Box<dyn SimpleBackend + 'static> = Box::new(m);
        let vb = crate::ShardedSafeTensors::wrap(backend, DType::F32, dev.clone());

        // Wrap with V4 rename rules — same call the V4 loader makes.
        let vb = crate::attach_rename_rules(vb, v4_scale_rename_rules());

        // 1. Asking for `weight_scale_inv` resolves to V4-native `.scale`.
        let scale = vb
            .get_with_hints_dtype(
                (8, 32),
                "layers.0.self_attn.wq_a.weight_scale_inv",
                Default::default(),
                DType::F32,
            )
            .expect("rename should resolve .weight_scale_inv -> .scale");
        assert_eq!(scale.dims(), &[8, 32]);

        // 2. Asking for the original `weight` is untouched.
        let weight = vb
            .get_with_hints_dtype(
                (1024, 4096),
                "layers.0.self_attn.wq_a.weight",
                Default::default(),
                DType::F32,
            )
            .expect("original weight should still resolve");
        assert_eq!(weight.dims(), &[1024, 4096]);

        // 3. `hc_attn_scale` is a real tensor name — rename rule maps
        //    `.weight_scale_inv` to `.scale`, but `hc_attn_scale` doesn't
        //    contain `.weight_scale_inv` so it's a direct hit.
        let hc = vb
            .get_with_hints_dtype(
                (3,),
                "layers.0.hc_attn_scale",
                Default::default(),
                DType::F32,
            )
            .expect("hc_attn_scale should be a direct hit, no rewrite");
        assert_eq!(hc.dims(), &[3]);
    }

    /// End-to-end check that `blockwise_fp8_linear_b` happily loads through
    /// a V4-rename-wrapped var-builder. This validates the full dispatch
    /// chain from the loader down to the FP8 layer constructor.
    ///
    /// We use a small synthetic weight (8×16 F8E4M3) with a 1×1 scale grid
    /// (block_size [128, 128] but the weight is smaller, so a single scale
    /// covers it). All values are zero — the test asserts only the load
    /// path works, not numerical correctness.
    #[test]
    fn blockwise_fp8_linear_b_loads_via_rename_wrapper() {
        use crate::{blockwise_fp8::blockwise_fp8_linear_b, QuantizedConfig, Shard};
        use float8::F8E4M3;

        let dev = Device::Cpu;
        let mut m: HashMap<String, Tensor> = HashMap::new();

        // Synthesize an FP8 weight tensor: [out=8, in=16].
        // F8E4M3 zeros — we don't need real values for the load-path test.
        let weight_data: Vec<F8E4M3> = vec![F8E4M3::from_f32(0.0); 8 * 16];
        let weight = Tensor::from_vec(weight_data, (8, 16), &dev).unwrap();
        m.insert("layers.0.self_attn.wq_a.weight".to_string(), weight);

        // Block size 128×128 → with weight shape 8×16, the scale grid is
        // 1×1 (because 8.div_ceil(128) = 1 and 16.div_ceil(128) = 1).
        // V4 publishes this as `.scale`.
        let scale = Tensor::ones((1, 1), DType::F32, &dev).unwrap();
        m.insert("layers.0.self_attn.wq_a.scale".to_string(), scale);

        let backend: Box<dyn SimpleBackend + 'static> = Box::new(m);
        let root_vb = crate::ShardedSafeTensors::wrap(backend, DType::F32, dev.clone());
        let root_vb = crate::attach_rename_rules(root_vb, v4_scale_rename_rules());

        // Drop down to the layer's path — same as the V4 model code does.
        let layer_vb = root_vb.pp("layers.0").pp("self_attn").pp("wq_a");

        // Sanity: from this layer-rooted vb, both `weight` and the rewritten
        // `weight_scale_inv` exist.
        assert!(layer_vb.contains_tensor("weight"));
        assert!(layer_vb.contains_tensor("weight_scale_inv"));

        // The full dispatch: `blockwise_fp8_linear_b` looks up `weight` +
        // `weight_scale_inv` at the current vb path. With our rename, the
        // V4-native `.scale` tensor satisfies `weight_scale_inv`.
        let cfg = QuantizedConfig::Fp8 {
            weight_block_size: Some(vec![128, 128]),
        };
        let layer = blockwise_fp8_linear_b(
            16, // in
            8,  // out
            &cfg,
            false,
            Shard::default(),
            layer_vb,
        )
        .expect("blockwise FP8 linear should load through V4 rename wrapper");

        // We got a BlockwiseFP8Linear (vs DummyLayer). Verify by checking
        // dtype.
        let (dtype, _) = layer.dtype_and_device();
        assert_eq!(dtype, DType::F8E4M3);
    }
}
