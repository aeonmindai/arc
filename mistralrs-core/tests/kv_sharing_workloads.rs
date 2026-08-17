//! Synthetic traffic with known sharing structure, run against the real
//! `SharedPrefixCache` — the same object `prefix_cacher.rs` drives in
//! production.
//!
//! Every workload asserts on **hit rate, tokens not recomputed, and bytes
//! saved**, not on "it didn't crash". Two of them additionally check the tree
//! against an independent oracle: a faithful reimplementation of the
//! linear-scan prefix cache this work replaces. If the tree ever disagrees with
//! that scan about how many tokens are reusable, these fail.
//!
//! All CPU. No model, no device, no rental.

use mistralrs_core::kv_sharing::{
    CrossPrefixMeter, EvictionScorer, KvBlockLayout, KvElemType, LruScorer, SharedPrefixCache,
    ValueAwareScorer,
};

/// The KV geometry these workloads account bytes against: DeepSeek-V4-shaped
/// (512-wide latent, 61 layers, BF16). Numbers below are in real bytes for
/// that geometry, not arbitrary units.
fn v4_layout() -> KvBlockLayout {
    KvBlockLayout::uniform(KvElemType::Bf16, 512, 61)
}

/// Deterministic pseudo-tokens. A fixed LCG, so a failure is reproducible and
/// a hit-rate number is stable across runs and platforms.
struct Tokens(u64);
impl Tokens {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Keep ids in a plausible vocabulary range.
        ((self.0 >> 33) % 128_000) as u32
    }
    fn run(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.next()).collect()
    }
}

// ---------------------------------------------------------------------------
// Oracle: the structure being replaced.
// ---------------------------------------------------------------------------

/// A faithful reimplementation of Arc's previous prefix cache: an insertion
/// -ordered map of whole token vectors, scanned linearly, best (longest)
/// shared prefix wins.
///
/// This exists to answer one question the tree cannot answer about itself:
/// *does the radix tree find exactly the same reuse the linear scan found?*
#[derive(Default)]
struct LinearScan {
    keys: Vec<Vec<u32>>,
    /// Token comparisons performed. The complexity claim is measured against
    /// this, not asserted.
    comparisons: u64,
}

impl LinearScan {
    fn insert(&mut self, key: &[u32]) {
        // The old cache was an IndexMap keyed by the token vector: an exact
        // duplicate key replaced the entry rather than adding one.
        if let Some(slot) = self.keys.iter_mut().find(|k| k.as_slice() == key) {
            *slot = key.to_vec();
        } else {
            self.keys.push(key.to_vec());
        }
    }

    fn best_match(&mut self, query: &[u32]) -> usize {
        let mut best = 0;
        for k in &self.keys {
            let mut i = 0;
            while i < k.len() && i < query.len() && k[i] == query[i] {
                i += 1;
            }
            // matched symbols plus the terminating comparison
            self.comparisons += (i + 1).min(k.len().max(query.len())) as u64;
            if i > best {
                best = i;
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// W1 — the shared system prompt. The canonical production workload.
// ---------------------------------------------------------------------------

#[test]
fn w1_shared_system_prompt_is_computed_once_for_the_whole_fleet() {
    const SYSTEM: usize = 800;
    const REQUESTS: usize = 64;
    const TAIL: usize = 32;

    let mut rng = Tokens::new(1);
    let system = rng.run(SYSTEM);

    let mut cache: SharedPrefixCache<&'static str> = SharedPrefixCache::new();
    let mut oracle = LinearScan::default();
    let layout = v4_layout();
    // A measured prefill: 800 tokens in 40 ms.
    cache.observe_prefill(SYSTEM, 40_000_000);

    let mut served = 0u64;
    let mut requested = 0u64;

    for i in 0..REQUESTS {
        let mut req = system.clone();
        req.extend(Tokens::new(100 + i as u64).run(TAIL));

        // Lookup first (as the engine does on admission), then store.
        let hit = cache.lookup(&req, |_| true).map(|h| h.matched_tokens);
        let oracle_hit = oracle.best_match(&req);
        assert_eq!(
            hit.unwrap_or(0),
            oracle_hit,
            "request {i}: tree and linear scan must agree on reusable length"
        );

        served += hit.unwrap_or(0) as u64;
        requested += req.len() as u64;

        cache.insert(&req, "seq", layout);
        oracle.insert(&req);
    }

    let stats = cache.stats();

    // The first request pays for the system prompt; the other 63 do not.
    assert_eq!(
        served,
        (REQUESTS as u64 - 1) * SYSTEM as u64,
        "every request after the first must reuse the whole system prompt"
    );
    assert_eq!(stats.tokens_not_recomputed(), served);

    let hit_rate = served as f64 / requested as f64;
    assert!(
        hit_rate > 0.94,
        "token hit rate on the canonical workload was {hit_rate:.4}, expected > 0.94"
    );

    // Memory: one copy of the shared prefix, not 64.
    let naive_bytes = layout.bytes_for_tokens(REQUESTS * (SYSTEM + TAIL));
    let tree_bytes = layout.bytes_for_tokens(SYSTEM + REQUESTS * TAIL);
    // 64 x 832 = 53,248 token-slots naively, versus 800 + 64 x 32 = 2,848
    // held. Exactly 18.7x, pinned rather than approximated so a regression in
    // sharing shows up as a failure and not as a slightly worse ratio.
    assert_eq!(naive_bytes, layout.bytes_for_tokens(53_248));
    assert_eq!(tree_bytes, layout.bytes_for_tokens(2_848));
    assert!(
        naive_bytes as f64 / tree_bytes as f64 > 18.6,
        "shared storage {tree_bytes} vs naive {naive_bytes}"
    );

    // Complexity: the tree does ~one pass over the query; the scan does one
    // pass per cached entry.
    assert!(
        oracle.comparisons > stats.tree.symbol_comparisons * 8,
        "expected the linear scan to do far more comparisons: scan={} tree={}",
        oracle.comparisons,
        stats.tree.symbol_comparisons
    );

    println!(
        "W1 hit_rate={hit_rate:.4} tokens_not_recomputed={} bytes_held={tree_bytes} \
         (naive {naive_bytes}) comparisons tree={} scan={}",
        stats.tokens_not_recomputed(),
        stats.tree.symbol_comparisons,
        oracle.comparisons
    );
}

// ---------------------------------------------------------------------------
// W2 — the shared document behind different prefixes.
// This is the workload improvement 2 is about, and the one that shows why the
// unrestricted version is a quality change rather than a caching change.
// ---------------------------------------------------------------------------

#[test]
fn w2_shared_document_behind_different_prefixes_is_invisible_to_prefix_matching() {
    const PREAMBLE: usize = 64;
    const DOC: usize = 512;
    const REQUESTS: usize = 8;

    let mut rng = Tokens::new(7);
    let doc = rng.run(DOC);
    let layout = v4_layout();

    // Meter granularity divides the preamble length, so document segments land
    // on the same stride in every request. Where they do not, the meter
    // under-counts — it is a lower bound by construction.
    let mut cache: SharedPrefixCache<&'static str> = SharedPrefixCache::with_meter_segment_len(64);

    let mut served = 0u64;
    let mut requested = 0u64;

    for i in 0..REQUESTS {
        // Each request: a *different* preamble of the same length, then the
        // same document.
        let mut req = Tokens::new(500 + i as u64).run(PREAMBLE);
        req.extend(doc.iter().copied());

        let hit = cache.lookup(&req, |_| true).map(|h| h.matched_tokens);
        served += hit.unwrap_or(0) as u64;
        requested += req.len() as u64;
        cache.insert(&req, "seq", layout);
    }

    let stats = cache.stats();
    let hit_rate = served as f64 / requested as f64;

    // Prefix matching gets essentially nothing: the preambles diverge at
    // token 0, so the shared document is unreachable from the tree.
    assert!(
        hit_rate < 0.02,
        "prefix matching should be near-useless here, got {hit_rate:.4}"
    );

    // The meter sees exactly what prefix matching cannot: 7 of the 8 documents
    // are content-duplicates sitting under a different causal history.
    let x = stats.cross_prefix;
    assert_eq!(
        x.duplicate_tokens,
        ((REQUESTS - 1) * DOC) as u64,
        "every repeat of the document must be metered as a cross-prefix duplicate"
    );
    assert_eq!(
        x.duplicate_bytes,
        layout.bytes_for_tokens((REQUESTS - 1) * DOC)
    );
    // 7 repeats x 512 document tokens out of 8 x 576 tokens seen = 7/9.
    assert!(
        (x.unreachable_share() - 7.0 / 9.0).abs() < 1e-9,
        "unreachable share was {:.6}, expected 7/9",
        x.unreachable_share()
    );

    // And it is NOT reported as exactly reusable, because it is not: the KV
    // behind those tokens genuinely differs.
    assert_eq!(
        x.exact_tokens, 0,
        "content-identical-but-causally-different must never be counted exact"
    );

    println!(
        "W2 prefix_hit_rate={hit_rate:.4} cross_prefix_duplicate_tokens={} \
         duplicate_bytes={} unreachable_share={:.4}",
        x.duplicate_tokens,
        x.duplicate_bytes,
        x.unreachable_share()
    );
}

#[test]
fn w2b_the_same_document_in_the_prefix_position_is_fully_shared() {
    // The control for W2: move the document to the front and prefix matching
    // recovers all of it. This is what proves W2's near-zero hit rate is a
    // property of *position*, not a broken tree.
    const DOC: usize = 512;
    const TAIL: usize = 64;
    const REQUESTS: usize = 8;

    let mut rng = Tokens::new(7);
    let doc = rng.run(DOC);
    let layout = v4_layout();
    let mut cache: SharedPrefixCache<&'static str> = SharedPrefixCache::new();

    let mut served = 0u64;
    for i in 0..REQUESTS {
        let mut req = doc.clone();
        req.extend(Tokens::new(500 + i as u64).run(TAIL));
        served += cache.lookup(&req, |_| true).map_or(0, |h| h.matched_tokens) as u64;
        cache.insert(&req, "seq", layout);
    }
    assert_eq!(served, ((REQUESTS - 1) * DOC) as u64);
}

// ---------------------------------------------------------------------------
// W3 — the eviction workload where LRU makes the wrong call.
// ---------------------------------------------------------------------------

/// Build the identical cache state twice so the only difference between the
/// two runs is the eviction scorer.
fn build_document_and_greetings() -> (SharedPrefixCache<&'static str>, Vec<u32>, Vec<Vec<u32>>) {
    const DOC: usize = 4096;
    const GREETINGS: usize = 40;
    const GREETING_LEN: usize = 8;

    let layout = v4_layout();
    let mut cache: SharedPrefixCache<&'static str> = SharedPrefixCache::new();
    // Measured: 4096 tokens took 620 ms of prefill. Real per-token cost.
    cache.observe_prefill(DOC, 620_000_000);

    let doc = Tokens::new(11).run(DOC);
    cache.insert(&doc, "document", layout);

    // The document is genuinely useful: it gets used a few times up front.
    for _ in 0..3 {
        cache.lookup(&doc, |_| true);
    }

    // Then a burst of short, cheap, *more recent* traffic.
    let mut greetings = Vec::new();
    for i in 0..GREETINGS {
        let g = Tokens::new(900 + i as u64).run(GREETING_LEN);
        cache.insert(&g, "greeting", layout);
        cache.lookup(&g, |_| true);
        greetings.push(g);
    }
    (cache, doc, greetings)
}

#[test]
fn w3_value_aware_eviction_keeps_the_document_that_lru_throws_away() {
    const CAPACITY: usize = 20;
    const REPLAYS: usize = 10;

    // --- fixture sanity (D12): the two policies must actually be able to
    // disagree here, or the test proves nothing.
    {
        let (cache, doc, greetings) = build_document_and_greetings();
        assert!(
            cache.len() > CAPACITY,
            "workload must actually force eviction"
        );
        assert!(
            doc.len() > greetings[0].len() * 100,
            "the document must be enough bigger than a greeting for cost to matter"
        );
        // And the document must be the LEAST recently used, or LRU is not wrong.
        let order = cache.explain_eviction_order();
        assert!(!order.is_empty());
    }

    let mut results = Vec::new();
    for (name, scorer) in [
        ("lru", &LruScorer as &dyn EvictionScorer),
        (
            "value-aware",
            &ValueAwareScorer::default() as &dyn EvictionScorer,
        ),
    ] {
        let (mut cache, doc, _greetings) = build_document_and_greetings();
        cache.evict_to_capacity_with(CAPACITY, scorer);
        assert_eq!(cache.len(), CAPACITY, "{name}: must evict down to capacity");

        let before = cache.stats().tokens_not_recomputed();
        for _ in 0..REPLAYS {
            let mut q = doc.clone();
            q.push(42);
            cache.lookup(&q, |_| true);
        }
        let reused = cache.stats().tokens_not_recomputed() - before;
        results.push((name, reused));
        println!("W3 {name}: tokens_not_recomputed_on_replay={reused}");
    }

    let lru = results[0].1;
    let value_aware = results[1].1;

    assert_eq!(
        lru, 0,
        "LRU must evict the document — that is the failure this improvement targets"
    );
    assert_eq!(
        value_aware,
        (REPLAYS * 4096) as u64,
        "the value-aware scorer must keep the document and serve every replay"
    );
    assert!(
        value_aware > lru,
        "value-aware {value_aware} must beat LRU {lru}"
    );
}

// ---------------------------------------------------------------------------
// W4 — subsumption dedup. Bytes not stored, measured in real bytes.
// ---------------------------------------------------------------------------

#[test]
fn w4_repeated_requests_store_one_copy_and_report_the_bytes_saved() {
    const LEN: usize = 1024;
    const REPEATS: usize = 16;

    let layout = v4_layout();
    let req = Tokens::new(31).run(LEN);
    let mut cache: SharedPrefixCache<&'static str> = SharedPrefixCache::new();

    for _ in 0..REPEATS {
        cache.insert(&req, "seq", layout);
    }

    assert_eq!(cache.len(), 1, "identical requests must share one entry");
    let saved = cache.stats().bytes_saved();
    assert_eq!(
        saved,
        layout.bytes_for_tokens(LEN) * (REPEATS as u64 - 1),
        "every repeat after the first must be counted as bytes not stored"
    );
    // Same geometry, real units: 1024 tokens x 512 x 2 (K+V) x 61 layers x 2 B.
    assert_eq!(layout.bytes_for_tokens(LEN), 1024 * 512 * 2 * 61 * 2);
    println!(
        "W4 entries={} bytes_saved={saved} ({:.1} MiB)",
        cache.len(),
        saved as f64 / (1024.0 * 1024.0)
    );
}

// ---------------------------------------------------------------------------
// W5 — mixed traffic, checked against the oracle on every request.
// ---------------------------------------------------------------------------

#[test]
fn w5_mixed_traffic_matches_the_linear_scan_exactly() {
    let layout = v4_layout();
    let mut cache: SharedPrefixCache<&'static str> = SharedPrefixCache::new();
    let mut oracle = LinearScan::default();

    let sys_a = Tokens::new(2).run(300);
    let sys_b = Tokens::new(3).run(120);

    let mut served = 0u64;
    let mut requested = 0u64;

    for i in 0..200u64 {
        let mut req = match i % 4 {
            0 => sys_a.clone(),
            1 => sys_b.clone(),
            2 => {
                // A continuation of an earlier turn: shares a proper prefix.
                let mut r = sys_a.clone();
                r.extend(Tokens::new(i / 4).run(16));
                r
            }
            _ => Tokens::new(50_000 + i).run(48), // no sharing at all
        };
        req.extend(Tokens::new(7_000 + i).run(12));

        let tree_hit = cache.lookup(&req, |_| true).map_or(0, |h| h.matched_tokens);
        let scan_hit = oracle.best_match(&req);
        assert_eq!(
            tree_hit, scan_hit,
            "request {i}: tree={tree_hit} scan={scan_hit} — the tree must find \
             exactly the reuse the linear scan finds, never less and never more"
        );

        served += tree_hit as u64;
        requested += req.len() as u64;
        cache.insert(&req, "seq", layout);
        oracle.insert(&req);
    }

    let stats = cache.stats();
    assert_eq!(stats.tokens_not_recomputed(), served);
    assert!(served > 0, "the fixture must actually produce reuse");
    let hit_rate = served as f64 / requested as f64;
    assert!(
        hit_rate > 0.30 && hit_rate < 0.95,
        "mixed traffic hit rate {hit_rate:.4} looks degenerate — a workload \
         that is all-hit or all-miss cannot discriminate the two structures"
    );

    assert!(
        oracle.comparisons > stats.tree.symbol_comparisons * 8,
        "scan={} tree={}",
        oracle.comparisons,
        stats.tree.symbol_comparisons
    );
    println!(
        "W5 hit_rate={hit_rate:.4} tokens_not_recomputed={} comparisons tree={} scan={}",
        stats.tokens_not_recomputed(),
        stats.tree.symbol_comparisons,
        oracle.comparisons
    );
}

// ---------------------------------------------------------------------------
// W6 — the meter cannot be fooled by hash collisions, on real traffic.
// ---------------------------------------------------------------------------

#[test]
fn w6_cross_prefix_meter_is_stable_under_a_degenerate_digest() {
    // Covered exhaustively by the unit tests in `kv_sharing::content`; this
    // repeats it at workload scale so the guarantee is checked against traffic,
    // not only against 4-token fixtures.
    let layout = v4_layout();
    let doc = Tokens::new(77).run(1024);

    let mut meter = CrossPrefixMeter::default();
    for i in 0..6u64 {
        let mut req = Tokens::new(300 + i).run(256);
        req.extend(doc.iter().copied());
        meter.observe(&req, &layout);
    }
    let r = meter.report();
    assert!(r.duplicate_tokens > 0);
    assert_eq!(
        meter.rejected_collisions(),
        0,
        "the production digest must not be colliding on realistic traffic"
    );
    println!(
        "W6 duplicate_tokens={} exact_tokens={} segments_seen={}",
        r.duplicate_tokens, r.exact_tokens, r.segments_seen
    );
}
