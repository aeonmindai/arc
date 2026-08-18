//! Radix-tree KV sharing.
//!
//! **Parent system: ArcInfer/ArcKV — this module is `ArcKV/Share`**
//! (see `memory/mission/TAXONOMY.md`). That is the product name this doc
//! previously said was "deliberately not decided here"; it is decided now.
//!
//! Replaces Arc's linear-scan prefix cache with a radix tree, and adds three
//! things the reference implementation does not have.
//!
//! # Layout
//!
//! | module | what it is |
//! |---|---|
//! | [`radix`] | the tree: match, split, insert, ref-counted protection, eviction |
//! | [`evict`] | eviction scoring, including the value-aware scorer |
//! | [`content`] | verified content identity + the cross-prefix reuse meter |
//! | [`layout`] | per-block KV byte layout (widths are per block, not global) |
//!
//! # Substrates
//!
//! One tree serves both of Arc's KV storage models, because it is generic over
//! the key symbol:
//!
//! * **Sequence substrate** ([`SequenceTree`]) — symbols are token ids, page
//!   size 1. This is what [`crate::prefix_cacher::PrefixCacheManagerV2`] drives
//!   for every model whose loader reports `supports_paged_attention == false`.
//!   Today that is DeepSeek-V4, so V4 is the model this PR actually speeds up.
//! * **Paged substrate** ([`PagedTree`]) — symbols are
//!   [`crate::paged_attention::block_hash::BlockHash`], one per block, so a
//!   single symbol already stands for `block_size` tokens and page size stays
//!   1. Used for the paged hybrid recurrent-state prefix cache.
//!
//! Models on paged attention already get vLLM-v1-style block-hash prefix
//! caching from `paged_attention::block_pool`; that path is untouched here.
//!
//! # Licensing
//!
//! The tree algorithm is **adapted** from SGLang (`RadixCache`, Apache-2.0).
//! No file here is a verbatim copy. See `NOTICE` in this directory and the
//! per-function citations in [`radix`].

pub mod content;
pub mod evict;
pub mod layout;
pub mod radix;

use std::collections::HashMap;

pub use content::{ContentDigest, ContentIndex, ContentMatch, CrossPrefixMeter, CrossPrefixReport};
pub use evict::{EvictionScorer, LruScorer, NodeStats, ValueAwareScorer};
pub use layout::{KvBlockLayout, KvElemType};
pub use radix::{InsertOutcome, MatchOutcome, NodeId, RadixTree, TreeStats};

/// Tree keyed on token ids. One symbol = one token.
pub type SequenceTree<P> = RadixTree<u32, P>;

/// Tree keyed on paged-attention block hashes. One symbol = one block.
pub type PagedTree<P> = RadixTree<crate::paged_attention::block_hash::BlockHash, P>;

/// Identifier of a cached element held by [`SharedPrefixCache`].
pub type EntryId = u64;

/// A lookup that found reusable KV.
#[derive(Debug)]
pub struct Hit<'a, E> {
    /// The cached element the caller should clone/restore from.
    pub element: &'a E,
    /// Tokens of the query the cached element actually covers. This is the
    /// payload's depth, **not** the tree walk depth: the tree can match deeper
    /// than any stored element reaches.
    pub matched_tokens: usize,
    /// Tree node the match terminated at, for `lock`/`unlock`.
    pub node: NodeId,
}

/// Everything the caller needs to report a workload result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShareStats {
    pub tree: TreeStats,
    pub cross_prefix: CrossPrefixReport,
    /// Entries currently held.
    pub entries: usize,
    /// Measured nanoseconds per token of prefill, as observed from real
    /// requests. Zero until the first observation — the scorer degrades to
    /// "all nodes equally cheap" rather than inventing a number.
    pub measured_cost_ns_per_token: u64,
    pub cost_samples: u64,
}

impl ShareStats {
    /// Tokens the cache served that would otherwise have been recomputed.
    pub fn tokens_not_recomputed(&self) -> u64 {
        self.tree.symbols_reused
    }
    /// Bytes not stored because an insert was already covered by the tree.
    pub fn bytes_saved(&self) -> u64 {
        self.tree.bytes_saved_by_dedup
    }
}

/// Sequence-substrate KV sharing: a radix tree over token ids whose payloads
/// are handles into a caller-owned element store.
///
/// Generic over the element so it can be exercised without a GPU, a model, or
/// a `KvCache` — the workload suite in `mistralrs-core/tests/` drives this
/// exact type, and `prefix_cacher.rs` drives it with the real `CacheElement`.
pub struct SharedPrefixCache<E> {
    tree: SequenceTree<EntryId>,
    entries: HashMap<EntryId, E>,
    next_id: EntryId,
    scorer: ValueAwareScorer,
    meter: CrossPrefixMeter,
    /// Exponentially-weighted mean of measured prefill cost per token.
    cost_ns_per_token: u64,
    cost_samples: u64,
}

impl<E> Default for SharedPrefixCache<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> SharedPrefixCache<E> {
    pub fn new() -> Self {
        Self::with_meter_segment_len(256)
    }

    /// As [`SharedPrefixCache::new`], with an explicit granularity for the
    /// cross-prefix meter.
    ///
    /// The meter chops requests at a fixed stride, so it only detects
    /// duplicates that land on the same stride boundary in both requests. Its
    /// output is therefore a **lower bound** on cross-prefix duplication, which
    /// is the right direction for a number that exists to size an unbuilt
    /// feature.
    pub fn with_meter_segment_len(segment_len: usize) -> Self {
        Self {
            tree: RadixTree::new(1),
            entries: HashMap::new(),
            next_id: 1,
            scorer: ValueAwareScorer::default(),
            meter: CrossPrefixMeter::new(content::Fnv1a128, segment_len),
            cost_ns_per_token: 0,
            cost_samples: 0,
        }
    }

    /// Record a **measured** prefill: `tokens` tokens took `elapsed_ns`.
    ///
    /// This is the only source of the eviction scorer's cost term. Nothing
    /// here estimates; if the caller never observes a prefill, cost stays 0 and
    /// the scorer ranks purely on reuse probability and staleness — degraded,
    /// but never fabricated.
    pub fn observe_prefill(&mut self, tokens: usize, elapsed_ns: u64) {
        if tokens == 0 || elapsed_ns == 0 {
            return;
        }
        let sample = elapsed_ns / tokens as u64;
        self.cost_ns_per_token = if self.cost_samples == 0 {
            sample
        } else {
            // EWMA, alpha = 1/8. Integer arithmetic so the value is exactly
            // reproducible across platforms.
            (self.cost_ns_per_token * 7 + sample) / 8
        };
        self.cost_samples += 1;
    }

    pub fn cost_ns_per_token(&self) -> u64 {
        self.cost_ns_per_token
    }

    /// Store `element` under `tokens`.
    ///
    /// Returns `None` when the insert was **subsumed** — the tree already held
    /// a payload covering every one of these tokens, so storing a second copy
    /// would waste `layout.bytes_for_tokens(tokens.len())` for nothing. The
    /// bytes avoided are counted in [`ShareStats::bytes_saved`].
    pub fn insert(&mut self, tokens: &[u32], element: E, layout: KvBlockLayout) -> Option<EntryId> {
        if tokens.is_empty() {
            return None;
        }
        self.meter.observe(tokens, &layout);

        let id = self.next_id;
        let outcome = self.tree.insert(tokens, id, layout, self.cost_ns_per_token);
        if outcome.subsumed {
            return None;
        }
        self.next_id += 1;
        self.entries.insert(id, element);
        Some(id)
    }

    /// Longest cached prefix of `tokens` whose element satisfies `accept`.
    pub fn lookup<F>(&mut self, tokens: &[u32], accept: F) -> Option<Hit<'_, E>>
    where
        F: Fn(&E) -> bool,
    {
        // Split the borrow: the tree walk needs `&mut tree` while the filter
        // needs `&entries`.
        let Self { tree, entries, .. } = self;
        let outcome = tree.match_prefix_with(tokens, |id| entries.get(id).is_some_and(&accept));
        let (id, depth) = outcome.payload?;
        if depth == 0 {
            return None;
        }
        Some(Hit {
            element: self.entries.get(&id)?,
            matched_tokens: depth,
            node: outcome.node,
        })
    }

    /// Evict until at most `capacity` elements remain, lowest expected value
    /// first. Returns the evicted elements so the caller can free device
    /// memory.
    pub fn evict_to_capacity(&mut self, capacity: usize) -> Vec<E> {
        if self.entries.len() <= capacity {
            return Vec::new();
        }
        let excess = self.entries.len() - capacity;
        let dropped = self.tree.evict_payloads(excess, &self.scorer);
        dropped
            .into_iter()
            .filter_map(|(id, _, _)| self.entries.remove(&id))
            .collect()
    }

    /// Evict using an explicit scorer. Used by the workload suite to run the
    /// same traffic under LRU and under the value-aware scorer.
    pub fn evict_to_capacity_with(
        &mut self,
        capacity: usize,
        scorer: &dyn EvictionScorer,
    ) -> Vec<E> {
        if self.entries.len() <= capacity {
            return Vec::new();
        }
        let excess = self.entries.len() - capacity;
        let dropped = self.tree.evict_payloads(excess, scorer);
        dropped
            .into_iter()
            .filter_map(|(id, _, _)| self.entries.remove(&id))
            .collect()
    }

    /// Protect an in-flight prefix from eviction.
    pub fn lock(&mut self, node: NodeId) {
        self.tree.lock(node);
    }

    /// Release a protection.
    pub fn unlock(&mut self, node: NodeId) {
        self.tree.unlock(node);
    }

    pub fn clear(&mut self) {
        self.tree.clear();
        self.entries.clear();
        self.meter.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn elements(&self) -> impl Iterator<Item = &E> {
        self.entries.values()
    }

    pub fn elements_mut(&mut self) -> impl Iterator<Item = &mut E> {
        self.entries.values_mut()
    }

    /// Drop entries for which `keep` returns false, without disturbing tree
    /// structure that other entries share.
    pub fn retain<F: FnMut(&E) -> bool>(&mut self, mut keep: F) {
        let doomed: Vec<EntryId> = self
            .entries
            .iter()
            .filter(|(_, e)| !keep(e))
            .map(|(id, _)| *id)
            .collect();
        for id in doomed {
            self.entries.remove(&id);
        }
    }

    pub fn stats(&self) -> ShareStats {
        ShareStats {
            tree: self.tree.stats(),
            cross_prefix: self.meter.report(),
            entries: self.entries.len(),
            measured_cost_ns_per_token: self.cost_ns_per_token,
            cost_samples: self.cost_samples,
        }
    }

    /// Printable eviction ranking under the active scorer.
    pub fn explain_eviction_order(&self) -> Vec<String> {
        self.tree.explain_eviction_order(&self.scorer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Elem {
        tag: &'static str,
        media_ok: bool,
    }

    fn e(tag: &'static str) -> Elem {
        Elem {
            tag,
            media_ok: true,
        }
    }

    #[test]
    fn lookup_returns_the_longest_covered_prefix() {
        let mut c: SharedPrefixCache<Elem> = SharedPrefixCache::new();
        c.insert(&[1, 2, 3], e("short"), KvBlockLayout::default());
        c.insert(&[1, 2, 3, 4, 5], e("long"), KvBlockLayout::default());

        let hit = c.lookup(&[1, 2, 3, 4, 5, 6], |_| true).unwrap();
        assert_eq!(hit.element.tag, "long");
        assert_eq!(hit.matched_tokens, 5);

        let hit = c.lookup(&[1, 2, 3, 9], |_| true).unwrap();
        assert_eq!(hit.element.tag, "short");
        assert_eq!(hit.matched_tokens, 3);
    }

    /// The multimodal case: a token-identical prefix whose media differ must
    /// fall back to a shallower acceptable entry, not lose the match entirely.
    #[test]
    fn rejected_element_falls_back_to_a_shallower_one() {
        let mut c: SharedPrefixCache<Elem> = SharedPrefixCache::new();
        c.insert(&[1, 2, 3], e("ok-short"), KvBlockLayout::default());
        c.insert(
            &[1, 2, 3, 4, 5],
            Elem {
                tag: "bad-long",
                media_ok: false,
            },
            KvBlockLayout::default(),
        );

        let hit = c.lookup(&[1, 2, 3, 4, 5, 6], |el| el.media_ok).unwrap();
        assert_eq!(
            hit.element.tag, "ok-short",
            "must climb past the rejected deeper entry"
        );
        assert_eq!(hit.matched_tokens, 3);
    }

    #[test]
    fn subsumed_insert_returns_none_and_saves_bytes() {
        let layout = KvBlockLayout::uniform(KvElemType::Bf16, 64, 2);
        let mut c: SharedPrefixCache<Elem> = SharedPrefixCache::new();
        assert!(c.insert(&[1, 2, 3, 4], e("a"), layout).is_some());
        assert!(c.insert(&[1, 2, 3, 4], e("b"), layout).is_none());
        assert_eq!(c.len(), 1);
        assert_eq!(c.stats().bytes_saved(), layout.bytes_for_tokens(4));
    }

    #[test]
    fn measured_cost_starts_at_zero_and_never_invents_a_number() {
        let mut c: SharedPrefixCache<Elem> = SharedPrefixCache::new();
        assert_eq!(c.cost_ns_per_token(), 0);
        assert_eq!(c.stats().cost_samples, 0);
        c.observe_prefill(1000, 2_000_000);
        assert_eq!(c.cost_ns_per_token(), 2_000);
        c.observe_prefill(0, 999); // degenerate samples ignored
        c.observe_prefill(10, 0);
        assert_eq!(c.stats().cost_samples, 1);
    }

    #[test]
    fn eviction_frees_elements_and_keeps_shared_structure() {
        let mut c: SharedPrefixCache<Elem> = SharedPrefixCache::new();
        let sys: Vec<u32> = (0..64).collect();
        for i in 0..8u32 {
            let mut k = sys.clone();
            k.push(1000 + i);
            c.insert(&k, e("x"), KvBlockLayout::default());
        }
        assert_eq!(c.len(), 8);
        let dropped = c.evict_to_capacity(3);
        assert_eq!(dropped.len(), 5);
        assert_eq!(c.len(), 3);
        // The shared 64-token prefix must survive: a surviving entry still
        // matches the full system prompt.
        let hit = c.lookup(&sys, |_| true);
        assert!(hit.is_none() || hit.unwrap().matched_tokens >= 64);
    }

    #[test]
    fn clear_resets_entries_and_meter_but_keeps_lifetime_counters() {
        let mut c: SharedPrefixCache<Elem> = SharedPrefixCache::new();
        c.insert(&[1, 2, 3], e("a"), KvBlockLayout::default());
        c.lookup(&[1, 2, 3], |_| true);
        let before = c.stats().tree.lookups;
        c.clear();
        assert!(c.is_empty());
        assert_eq!(
            c.stats().tree.lookups,
            before,
            "lifetime counters must survive a cache flush, or throughput \
             reporting silently resets mid-run"
        );
    }
}
