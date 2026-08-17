//! Eviction scoring.
//!
//! Improvement 3. SGLang ships seven strategies
//! (`python/sglang/srt/mem_cache/evict_policy.py:16-65`) and every one of them
//! is a function of *when* or *how often* a node was touched — `last_access`,
//! `creation_time`, `hit_count`, or a static `priority`. None of them knows
//! what a node cost to build or how much memory it holds.
//!
//! That is the gap: LRU will drop a 100k-token document that took seconds of
//! prefill in favour of a one-line greeting used more recently, because the
//! greeting's timestamp is larger. [`ValueAwareScorer`] scores by
//! **measured recompute cost x estimated reuse probability**, decayed by
//! staleness, and the whole thing prints.
//!
//! [`LruScorer`] is kept because it is the baseline the improvement is measured
//! against, and because a mutation test needs something to be *worse*.

/// Everything a scorer is allowed to see. Deliberately small: a scoring
/// function you cannot print is a scoring function you cannot defend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeStats {
    /// Symbols (tokens, or blocks on the paged substrate) this node holds.
    pub symbols: usize,
    /// Bytes of KV held, computed from the node's own [`super::layout::KvBlockLayout`].
    /// A compressed block reports fewer bytes than an uncompressed one holding
    /// the same tokens — that is the point of improvement 1.
    pub bytes: u64,
    /// **Measured** nanoseconds to rebuild these symbols, from the caller's
    /// real prefill timing. Never a guess, never a constant chosen to make a
    /// benchmark look good.
    pub recompute_ns: u64,
    /// Lookups that matched this node.
    pub hits: u64,
    /// Lookups that reached this node's parent, i.e. chances it had to be hit.
    pub opportunities: u64,
    pub last_access: u64,
    pub created: u64,
    /// Current logical clock.
    pub now: u64,
}

impl NodeStats {
    /// Laplace-smoothed reuse probability.
    ///
    /// `(hits + 1) / (opportunities + 2)`. The smoothing matters: a node
    /// inserted one tick ago has zero hits and zero opportunities, and an
    /// unsmoothed `0/0` would make every fresh insert instantly the best
    /// eviction candidate — which is how you build a cache that evicts exactly
    /// what you just paid to compute.
    #[allow(clippy::cast_precision_loss)]
    pub fn reuse_probability(&self) -> f64 {
        (self.hits as f64 + 1.0) / (self.opportunities as f64 + 2.0)
    }

    /// Logical ticks since last access.
    pub fn age(&self) -> u64 {
        self.now.saturating_sub(self.last_access)
    }
}

/// Lower score is evicted first.
pub trait EvictionScorer {
    fn score(&self, s: &NodeStats) -> f64;
    fn name(&self) -> &'static str;
    /// The terms behind the score, for logs and for
    /// `RadixTree::explain_eviction_order`.
    fn explain(&self, s: &NodeStats) -> String;
}

/// The baseline: evict least-recently-used.
///
/// Equivalent to SGLang's `LRUStrategy` (evict_policy.py:16-18), which returns
/// `node.last_access_time`.
pub struct LruScorer;

impl EvictionScorer for LruScorer {
    #[allow(clippy::cast_precision_loss)]
    fn score(&self, s: &NodeStats) -> f64 {
        s.last_access as f64
    }
    fn name(&self) -> &'static str {
        "lru"
    }
    fn explain(&self, s: &NodeStats) -> String {
        format!("last_access={}", s.last_access)
    }
}

/// Evict by expected value: what it would cost to rebuild this, times how
/// likely we are to need it, discounted by how stale it is.
///
/// ```text
///   score = recompute_ns * reuse_probability * staleness_discount
///
///   recompute_ns          measured at insert from real prefill time
///   reuse_probability     (hits + 1) / (opportunities + 2)
///   staleness_discount    1 / (1 + age / half_life)
/// ```
///
/// Three terms, each with a job:
///
/// * **recompute_ns** is why a 100k-token document outranks a greeting: it is
///   four orders of magnitude more expensive to rebuild. This is the term LRU
///   does not have.
/// * **reuse_probability** stops a big-but-useless entry from being immortal.
///   Every lookup that passes a node without hitting it raises `opportunities`
///   and lowers the estimate.
/// * **staleness_discount** covers the case `reuse_probability` cannot: a node
///   that stops being *probed at all* accumulates no opportunities, so without
///   an age term its score would never move and it would starve the cache.
///   Half-life is in logical ticks (one per tree operation), not wall time, so
///   eviction decisions are reproducible.
///
/// Note what is deliberately **absent**: the score is not divided by `bytes`.
/// A per-byte density would rank the greeting above the document (they have
/// nearly identical cost-per-byte), which is precisely the decision this
/// scorer exists to avoid. `bytes` is reported in [`EvictionScorer::explain`]
/// so the trade is visible, and it is what the caller sums when deciding *how
/// much* to evict.
pub struct ValueAwareScorer {
    /// Ticks after which the staleness discount reaches 1/2.
    pub half_life: f64,
}

impl Default for ValueAwareScorer {
    fn default() -> Self {
        // 4096 tree operations. Long enough that a prefix reused across a
        // multi-turn conversation is not discounted mid-conversation, short
        // enough that a dead prefix falls off within a few thousand requests.
        Self { half_life: 4096.0 }
    }
}

impl ValueAwareScorer {
    #[allow(clippy::cast_precision_loss)]
    fn staleness_discount(&self, s: &NodeStats) -> f64 {
        1.0 / (1.0 + s.age() as f64 / self.half_life.max(1.0))
    }
}

impl EvictionScorer for ValueAwareScorer {
    #[allow(clippy::cast_precision_loss)]
    fn score(&self, s: &NodeStats) -> f64 {
        let v = s.recompute_ns as f64 * s.reuse_probability() * self.staleness_discount(s);
        // Finite by construction, but a NaN here would silently corrupt the
        // eviction heap's ordering, so pin it.
        debug_assert!(v.is_finite(), "eviction score must be finite, got {v}");
        v
    }
    fn name(&self) -> &'static str {
        "value-aware"
    }
    fn explain(&self, s: &NodeStats) -> String {
        format!(
            "recompute_ns={} reuse_p={:.4} stale={:.4} bytes={} age={}",
            s.recompute_ns,
            s.reuse_probability(),
            self.staleness_discount(s),
            s.bytes,
            s.age()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(now: u64) -> NodeStats {
        NodeStats {
            symbols: 100_000,
            bytes: 100_000 * 512,
            // 100k tokens of prefill: seconds, measured.
            recompute_ns: 3_000_000_000,
            hits: 3,
            opportunities: 6,
            last_access: 10,
            created: 1,
            now,
        }
    }

    fn greeting(now: u64) -> NodeStats {
        NodeStats {
            symbols: 8,
            bytes: 8 * 512,
            // 8 tokens: microseconds.
            recompute_ns: 240_000,
            hits: 1,
            opportunities: 2,
            last_access: now,
            created: now - 1,
            now,
        }
    }

    /// The headline claim, as an assertion.
    #[test]
    fn lru_evicts_the_document_and_the_value_scorer_does_not() {
        let now = 100;
        let (d, g) = (doc(now), greeting(now));

        // Fixture must actually discriminate (D12): if the greeting were not
        // more recent, LRU would not be wrong and the test would prove nothing.
        assert!(
            g.last_access > d.last_access,
            "fixture is degenerate: greeting must be more recent than the doc"
        );

        let lru = LruScorer;
        assert!(
            lru.score(&d) < lru.score(&g),
            "LRU must rank the document as the first eviction — that is the bug"
        );

        let va = ValueAwareScorer::default();
        assert!(
            va.score(&d) > va.score(&g),
            "value-aware must keep the document: {} vs {}",
            va.score(&d),
            va.score(&g)
        );
    }

    #[test]
    fn a_big_but_never_reused_entry_decays_below_a_small_reused_one() {
        let now = 100_000;
        let mut dead = doc(now);
        dead.hits = 0;
        dead.opportunities = 5_000; // probed constantly, never matched
        dead.last_access = 0; // and stale

        let live = greeting(now);
        let va = ValueAwareScorer::default();
        assert!(
            va.score(&dead) < va.score(&live),
            "expensive-but-dead must lose: {} vs {}",
            va.score(&dead),
            va.score(&live)
        );
    }

    #[test]
    fn fresh_insert_is_not_instantly_the_best_eviction_candidate() {
        let now = 10;
        let fresh = NodeStats {
            symbols: 100,
            bytes: 100 * 512,
            recompute_ns: 5_000_000,
            hits: 0,
            opportunities: 0,
            last_access: now,
            created: now,
            now,
        };
        // Unsmoothed this would be 0/0; smoothed it is 1/2.
        assert!((fresh.reuse_probability() - 0.5).abs() < 1e-12);
        assert!(ValueAwareScorer::default().score(&fresh) > 0.0);
    }

    #[test]
    fn staleness_discount_is_monotone_and_bounded() {
        let va = ValueAwareScorer::default();
        let base = doc(10);
        let mut prev = f64::INFINITY;
        for now in [10u64, 100, 1_000, 10_000, 100_000] {
            let s = NodeStats { now, ..base };
            let d = va.staleness_discount(&s);
            assert!(d > 0.0 && d <= 1.0, "discount out of range: {d}");
            assert!(d <= prev, "discount must not increase with age");
            prev = d;
        }
    }

    #[test]
    fn explain_shows_every_term() {
        let e = ValueAwareScorer::default().explain(&doc(100));
        for term in ["recompute_ns=", "reuse_p=", "stale=", "bytes=", "age="] {
            assert!(e.contains(term), "explain() is missing {term}: {e}");
        }
    }
}
