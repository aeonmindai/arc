//! YOCO (You Only Cache Once) cross-layer KV sharing.
//!
//! Paper: `research/12_long_context/yoco_you_only_cache_once.pdf` (Microsoft)
//! Repo:  `research/code/01_weight_compression/microsoft_unilm/YOCO/`
//!
//! ## What
//!
//! YOCO splits a model into two stages:
//!   1. **Self-decoder** layers (the first N/2): produce per-layer KV as normal.
//!   2. **Cross-decoder** layers (the last N/2): all reuse a single shared KV
//!      computed from the self-decoder's output.
//!
//! Net: per-user KV memory ~2× smaller with <0.1% perplexity drift.
//! Stacks multiplicatively with TurboQuant K4/V3: 4.5× × 2× ≈ 9× total compression.
//!
//! ## Tier A scope
//!
//! - `YocoLayout`: which layers cache, which layers share, sharing mapping
//! - `default_split()`: standard half-and-half layout from the paper
//! - `layer_cache_target()`: for layer i, returns the layer index whose KV it uses
//! - Tests: layout invariants, sharing routes through the self-decoder layer
//!
//! ## Tier B (deferred)
//!
//! - `KvCacheManager` plumbing in `mistralrs-core/src/kv_cache/`
//! - Load Microsoft's YOCO-1B / YOCO-3B research checkpoints
//! - Perplexity reproduction on Wikitext-2

/// Per-layer cache role in a YOCO-laid-out model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerRole {
    /// This layer computes its own KV and stores it.
    /// The stored KV may be shared by one or more cross-decoder layers.
    SelfDecoder { stored_at: usize },
    /// This layer doesn't store KV; it reads from another layer's stored KV.
    CrossDecoder { reads_from: usize },
}

#[derive(Debug, Clone)]
pub struct YocoLayout {
    /// Number of layers in the model.
    pub num_layers: usize,
    /// Per-layer role.
    pub roles: Vec<LayerRole>,
}

impl YocoLayout {
    /// Standard YOCO split: first half are self-decoders, second half are
    /// cross-decoders that all share the LAST self-decoder's KV.
    ///
    /// This matches the paper's Figure 2 layout for 32-layer YOCO-3B:
    /// layers 0..15 cache their own KV; layers 16..31 all read from layer 15.
    pub fn default_split(num_layers: usize) -> Self {
        let half = num_layers / 2;
        let mut roles = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            if i < half {
                roles.push(LayerRole::SelfDecoder { stored_at: i });
            } else {
                // Read from the last self-decoder (layer half - 1).
                let read_from = half.saturating_sub(1);
                roles.push(LayerRole::CrossDecoder {
                    reads_from: read_from,
                });
            }
        }
        Self { num_layers, roles }
    }

    /// "No sharing" layout — every layer caches its own. Equivalent to standard
    /// transformer behavior. Useful as a baseline and for non-YOCO models.
    pub fn none(num_layers: usize) -> Self {
        let roles = (0..num_layers)
            .map(|i| LayerRole::SelfDecoder { stored_at: i })
            .collect();
        Self { num_layers, roles }
    }

    /// For layer `i`, return the layer index whose stored KV it should use.
    /// For self-decoders, this is the layer itself. For cross-decoders, this is
    /// the linked self-decoder.
    pub fn layer_cache_target(&self, i: usize) -> usize {
        match self.roles[i] {
            LayerRole::SelfDecoder { stored_at } => stored_at,
            LayerRole::CrossDecoder { reads_from } => reads_from,
        }
    }

    /// Number of distinct cache slots needed (the count of self-decoders).
    pub fn num_cache_slots(&self) -> usize {
        self.roles
            .iter()
            .filter(|r| matches!(r, LayerRole::SelfDecoder { .. }))
            .count()
    }

    /// KV memory savings ratio: (without YOCO) / (with YOCO).
    /// At standard half-and-half, this is `num_layers / num_cache_slots` ≈ 2.
    pub fn savings_ratio(&self) -> f64 {
        self.num_layers as f64 / self.num_cache_slots() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_split_32_layers_half_and_half() {
        let layout = YocoLayout::default_split(32);
        assert_eq!(layout.num_layers, 32);
        // First 16 self, last 16 cross
        for i in 0..16 {
            assert!(
                matches!(layout.roles[i], LayerRole::SelfDecoder { stored_at } if stored_at == i)
            );
        }
        for i in 16..32 {
            assert!(
                matches!(layout.roles[i], LayerRole::CrossDecoder { reads_from } if reads_from == 15)
            );
        }
    }

    #[test]
    fn cache_target_routes_through_last_self_decoder() {
        let layout = YocoLayout::default_split(16);
        // First 8 are self-decoders, indexed 0..7
        assert_eq!(layout.layer_cache_target(0), 0);
        assert_eq!(layout.layer_cache_target(7), 7);
        // Layers 8..15 all read from layer 7
        for i in 8..16 {
            assert_eq!(layout.layer_cache_target(i), 7);
        }
    }

    #[test]
    fn num_cache_slots_is_half() {
        for &n in &[8, 16, 32, 64, 80] {
            let layout = YocoLayout::default_split(n);
            assert_eq!(layout.num_cache_slots(), n / 2);
        }
    }

    #[test]
    fn savings_ratio_about_2x() {
        let layout = YocoLayout::default_split(32);
        assert!((layout.savings_ratio() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn none_layout_has_no_savings() {
        let layout = YocoLayout::none(32);
        assert_eq!(layout.num_cache_slots(), 32);
        assert!((layout.savings_ratio() - 1.0).abs() < 1e-9);
        for i in 0..32 {
            assert!(
                matches!(layout.roles[i], LayerRole::SelfDecoder { stored_at } if stored_at == i)
            );
        }
    }

    #[test]
    fn handles_odd_layer_count() {
        // 9 layers → 4 self, 5 cross all reading from layer 3
        let layout = YocoLayout::default_split(9);
        assert_eq!(layout.num_cache_slots(), 4);
        assert!(matches!(
            layout.roles[3],
            LayerRole::SelfDecoder { stored_at: 3 }
        ));
        assert!(matches!(
            layout.roles[4],
            LayerRole::CrossDecoder { reads_from: 3 }
        ));
        assert!(matches!(
            layout.roles[8],
            LayerRole::CrossDecoder { reads_from: 3 }
        ));
    }
}
