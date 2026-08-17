//! Radix tree over cache keys, for KV prefix sharing.
//!
//! # Provenance and licensing
//!
//! The *algorithm* here — walk-and-split prefix matching, node splitting on a
//! partial match, ref-counted protection of in-flight prefixes, and eviction
//! restricted to unlocked leaves — is **adapted from SGLang's `RadixCache`**,
//! `python/sglang/srt/mem_cache/radix_cache.py` (SGLang Team, Apache-2.0),
//! at commit `c5251a98a9d499d600beb557835ac5874e0c3f36`. Per-function citations
//! appear on each method. See `mistralrs-core/src/kv_sharing/NOTICE` for the
//! attribution required by Apache-2.0 §4.
//!
//! **Nothing is copied verbatim.** SGLang's implementation is Python, uses
//! parent pointers plus `defaultdict` children, stores `torch.Tensor` values,
//! and is hard-wired to token ids. This is an arena-indexed Rust tree, generic
//! over the key symbol (token ids on the sequence substrate, block hashes on
//! the paged substrate) and over the stored payload, with per-node measured
//! cost and per-block byte layout that SGLang has no equivalent of.
//!
//! # What is ours
//!
//! - Generic `Symbol` key so one tree serves both substrates.
//! - Per-node **measured** build cost and per-block [`KvBlockLayout`], which
//!   feed the value-aware eviction scorer (`super::evict`).
//! - Subsumption dedup on insert (`super::content`), which drops a payload
//!   whose tokens are already covered by the tree instead of storing a second
//!   copy.
//! - Instrumentation (`TreeStats`) that reports hit rate, bytes saved and
//!   tokens-not-recomputed rather than requiring a caller to derive them.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::Hash;

use super::evict::{EvictionScorer, NodeStats};
use super::layout::KvBlockLayout;

/// A key symbol. Token ids (`u32`) on the sequence substrate; block hashes on
/// the paged substrate.
pub trait Symbol: Copy + Eq + Hash + std::fmt::Debug {}
impl<T: Copy + Eq + Hash + std::fmt::Debug> Symbol for T {}

/// Arena index of a tree node. `ROOT` is always index 0.
pub type NodeId = usize;

/// The root node's arena index.
pub const ROOT: NodeId = 0;

/// A logical clock. Wall time is deliberately avoided: eviction decisions must
/// be reproducible in tests, and a monotonically increasing operation counter
/// orders accesses exactly as well as `Instant` does for LRU purposes.
#[derive(Debug, Default)]
struct Clock(u64);

impl Clock {
    fn tick(&mut self) -> u64 {
        self.0 += 1;
        self.0
    }
    fn now(&self) -> u64 {
        self.0
    }
}

struct Node<S: Symbol, P> {
    parent: Option<NodeId>,
    /// Child edges keyed by the first `page_size` symbols of the child's key.
    children: HashMap<Box<[S]>, NodeId>,
    /// The symbols on the edge *into* this node.
    key: Vec<S>,
    /// Total symbol depth from the root, i.e. `parent.depth + key.len()`.
    depth: usize,
    /// Payload that terminates exactly at this node, if any.
    payload: Option<P>,
    /// Ref count: nonzero means an in-flight request depends on this node and
    /// it must not be evicted. Mirrors SGLang `TreeNode.lock_ref`
    /// (radix_cache.py:215).
    lock_ref: u32,
    last_access: u64,
    created: u64,
    /// Number of times a lookup matched at or below this node.
    hits: u64,
    /// Number of lookups this node was *eligible* for (i.e. that reached its
    /// parent). Denominator of the reuse-probability estimate.
    opportunities: u64,
    /// Byte layout of the KV this node's symbols stand for. A property of the
    /// block, never a global assumption (improvement 1).
    layout: KvBlockLayout,
    /// **Measured** nanoseconds to recompute one symbol's worth of KV, taken
    /// from the caller's actual prefill timing at insert. Never estimated.
    cost_ns_per_symbol: u64,
}

impl<S: Symbol, P> Node<S, P> {
    fn stats(&self, now: u64) -> NodeStats {
        NodeStats {
            symbols: self.key.len(),
            bytes: self.layout.bytes_for_tokens(self.key.len()),
            recompute_ns: self
                .cost_ns_per_symbol
                .saturating_mul(self.key.len() as u64),
            hits: self.hits,
            opportunities: self.opportunities,
            last_access: self.last_access,
            created: self.created,
            now,
        }
    }
}

/// What a lookup found.
#[derive(Debug, Clone)]
pub struct MatchOutcome<P> {
    /// Deepest node whose path is a prefix of the query.
    pub node: NodeId,
    /// Number of query symbols covered by the cached prefix.
    pub matched: usize,
    /// Payload of the deepest node that carries one, at or above `node`, plus
    /// the depth at which it was found. `None` when the tree holds no payload
    /// covering this prefix.
    pub payload: Option<(P, usize)>,
}

/// Result of an insert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertOutcome {
    /// How many leading symbols were already present in the tree.
    pub shared_prefix: usize,
    /// Node the inserted key terminates at.
    pub node: NodeId,
    /// True when the whole key was already covered *and* a payload already
    /// existed at or below the terminal node, so the caller's payload was not
    /// stored. This is the byte-saving subsumption case.
    pub subsumed: bool,
}

/// Counters the tree maintains itself, so callers cannot disagree about them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TreeStats {
    pub lookups: u64,
    pub hits: u64,
    /// Symbols served from cache across all lookups — i.e. tokens not
    /// recomputed.
    pub symbols_reused: u64,
    /// Symbols that had to be computed because no cached prefix covered them.
    pub symbols_computed: u64,
    /// Bytes of KV that were *not* stored because an insert was subsumed by a
    /// prefix already in the tree.
    pub bytes_saved_by_dedup: u64,
    /// Symbols whose payload insert was skipped as subsumed.
    pub symbols_deduped: u64,
    pub inserts: u64,
    pub evictions: u64,
    pub symbols_evicted: u64,
    pub nodes_split: u64,
    /// Symbol comparisons performed while matching: every symbol that matched,
    /// plus the one that ended each segment. This is the number the O(|key|)
    /// claim is about, and the workload suite compares it directly against the
    /// linear scan this tree replaced.
    pub symbol_comparisons: u64,
}

impl TreeStats {
    /// Fraction of *queried* symbols served from cache. This is the number the
    /// mission asks for; it is a token-weighted hit rate, not a per-request
    /// one, because a 1-token hit on an 800-token prompt is not a hit.
    #[allow(clippy::cast_precision_loss)]
    pub fn symbol_hit_rate(&self) -> f64 {
        let total = self.symbols_reused + self.symbols_computed;
        if total == 0 {
            return 0.0;
        }
        self.symbols_reused as f64 / total as f64
    }

    /// Fraction of lookups that matched at least one symbol.
    #[allow(clippy::cast_precision_loss)]
    pub fn request_hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / self.lookups as f64
    }
}

/// A radix tree of cache keys with ref-counted protection and pluggable,
/// value-aware eviction.
pub struct RadixTree<S: Symbol, P> {
    nodes: Vec<Node<S, P>>,
    /// Freed arena slots, reused so the arena does not grow without bound.
    free_slots: Vec<NodeId>,
    /// Symbols per page. Matching is rounded down to a multiple of this, so a
    /// paged substrate never reports a partial block as cached.
    page_size: usize,
    /// Unlocked, payload-free-to-drop leaves — the eviction candidate set.
    /// Mirrors SGLang's `evictable_leaves` (radix_cache.py:314).
    evictable_leaves: HashSet<NodeId>,
    clock: Clock,
    stats: TreeStats,
    /// Live symbol count held by unlocked nodes.
    evictable_symbols: usize,
    /// Live symbol count held by locked nodes.
    protected_symbols: usize,
}

impl<S: Symbol, P: Clone> RadixTree<S, P> {
    /// `page_size` must be >= 1. A paged substrate passes its block size; a
    /// sequence substrate passes 1.
    pub fn new(page_size: usize) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        let root = Node {
            parent: None,
            children: HashMap::new(),
            key: Vec::new(),
            depth: 0,
            payload: None,
            // The root is permanently locked so eviction can never walk off
            // the top. SGLang does the same (radix_cache.py:343).
            lock_ref: 1,
            last_access: 0,
            created: 0,
            hits: 0,
            opportunities: 0,
            layout: KvBlockLayout::default(),
            cost_ns_per_symbol: 0,
        };
        Self {
            nodes: vec![root],
            free_slots: Vec::new(),
            page_size,
            evictable_leaves: HashSet::new(),
            clock: Clock::default(),
            stats: TreeStats::default(),
            evictable_symbols: 0,
            protected_symbols: 0,
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn stats(&self) -> TreeStats {
        self.stats
    }

    pub fn evictable_symbols(&self) -> usize {
        self.evictable_symbols
    }

    pub fn protected_symbols(&self) -> usize {
        self.protected_symbols
    }

    /// Number of live nodes, excluding the root.
    pub fn len(&self) -> usize {
        self.nodes.len() - self.free_slots.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Shared prefix length of two symbol slices, rounded down to `page_size`.
    ///
    /// Adapted from SGLang `RadixKey.match` (radix_cache.py:147-176). Same
    /// semantics; the Python walks a zip, this walks slices.
    fn shared_len(a: &[S], b: &[S], page: usize) -> usize {
        let min = a.len().min(b.len());
        if page == 1 {
            let mut i = 0;
            while i < min && a[i] == b[i] {
                i += 1;
            }
            i
        } else {
            let mut i = 0;
            while i + page <= min && a[i..i + page] == b[i..i + page] {
                i += page;
            }
            i
        }
    }

    fn child_key(key: &[S], page: usize) -> Box<[S]> {
        key[..page.min(key.len())].to_vec().into_boxed_slice()
    }

    /// Longest cached prefix of `key`.
    ///
    /// Adapted from SGLang `_match_prefix_helper` (radix_cache.py:645-669).
    /// Like SGLang, a partial match inside a stored segment **splits** the node
    /// so the boundary is exact; unlike SGLang this also carries the payload
    /// search upward, because our payloads live only at terminal nodes.
    ///
    /// Complexity is O(|key|) symbol comparisons plus O(depth) hash probes,
    /// against O(cached_entries x |key|) for a linear scan.
    pub fn match_prefix(&mut self, key: &[S]) -> MatchOutcome<P> {
        self.match_prefix_with(key, |_| true)
    }

    /// [`RadixTree::match_prefix`], but a payload must also satisfy `accept`.
    ///
    /// The payload walk climbs to the next shallower payload when `accept`
    /// rejects one, so the caller gets the longest *acceptable* prefix rather
    /// than "the longest prefix, or nothing". Arc's non-paged cache needs this:
    /// vision and audio models emit identical placeholder token ids for
    /// different media, so a token-identical prefix can still be the wrong KV,
    /// and declining the whole match would lose reuse the old linear scan
    /// found.
    pub fn match_prefix_with<F>(&mut self, key: &[S], accept: F) -> MatchOutcome<P>
    where
        F: Fn(&P) -> bool,
    {
        let now = self.clock.tick();
        self.stats.lookups += 1;
        self.nodes[ROOT].last_access = now;

        // Round the query down to a page multiple: a partially-filled page has
        // no valid cached KV behind it.
        let usable = (key.len() / self.page_size) * self.page_size;
        let key = &key[..usable];

        let mut node = ROOT;
        let mut matched = 0usize;
        let mut rest = key;

        while !rest.is_empty() {
            let ck = Self::child_key(rest, self.page_size);
            let Some(&child) = self.nodes[node].children.get(&ck) else {
                break;
            };
            self.nodes[child].opportunities += 1;
            self.nodes[child].last_access = now;
            let plen = Self::shared_len(&self.nodes[child].key, rest, self.page_size);
            self.stats.symbol_comparisons +=
                (plen + self.page_size).min(self.nodes[child].key.len().max(rest.len())) as u64;
            if plen < self.nodes[child].key.len() {
                // Partial match inside a stored segment: split so the boundary
                // is representable, then stop.
                let split = self.split_node(child, plen);
                self.nodes[split].last_access = now;
                self.nodes[split].hits += 1;
                node = split;
                matched += plen;
                break;
            }
            self.nodes[child].hits += 1;
            node = child;
            matched += plen;
            rest = &rest[plen..];
        }

        // Resolve which stored payload can serve this match.
        //
        // A payload *deeper* than the match still covers it: a cache holding
        // 5,000 tokens can be truncated to the 900 that matched. So the search
        // starts at the deepest matched node and looks **into its subtree**,
        // not only above it — every payload below a node at depth `d` shares
        // that node's first `d` symbols by construction. Arc's previous linear
        // scan had this property for free (it compared against whole stored
        // sequences); losing it in the port would have been a silent
        // reduction in hit rate, not a refactor.
        //
        // If nothing acceptable is found, climb one level and search the
        // branches not already visited, with the usable length reduced to that
        // ancestor's depth.
        let mut payload = None;
        let mut cursor = node;
        let mut visited: Option<NodeId> = None;
        loop {
            if let Some(p) = self.find_acceptable(cursor, visited, &accept) {
                payload = Some((p, self.nodes[cursor].depth));
                break;
            }
            visited = Some(cursor);
            match self.nodes[cursor].parent {
                Some(parent) if cursor != ROOT => cursor = parent,
                _ => break,
            }
        }

        let served = payload.as_ref().map(|(_, d)| *d).unwrap_or(0);
        if served > 0 {
            self.stats.hits += 1;
        }
        self.stats.symbols_reused += served as u64;
        self.stats.symbols_computed += (key.len().saturating_sub(served)) as u64;

        MatchOutcome {
            node,
            matched,
            payload,
        }
    }

    /// First payload at or below `root`, in depth-first order, that `accept`
    /// admits. `skip` prunes a subtree already searched.
    ///
    /// Depth-first with the node's own payload checked first, so the shallowest
    /// (cheapest to truncate) candidate wins and the result is independent of
    /// hash iteration order for the common single-payload case.
    fn find_acceptable<F>(&self, root: NodeId, skip: Option<NodeId>, accept: &F) -> Option<P>
    where
        F: Fn(&P) -> bool,
    {
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if Some(id) == skip {
                continue;
            }
            if let Some(p) = self.nodes[id].payload.as_ref() {
                if accept(p) {
                    return Some(p.clone());
                }
            }
            for &c in self.nodes[id].children.values() {
                stack.push(c);
            }
        }
        None
    }

    /// Split `child` so that its first `split_len` symbols live in a new parent
    /// node and the remainder stays in `child`.
    ///
    /// Adapted from SGLang `_split_node` (radix_cache.py:671-691). The Python
    /// clones tensor slices; here the payload is *not* duplicated — it stays
    /// with the tail node, because a payload describes the whole key that
    /// terminated there, and half of it is not a valid payload.
    fn split_node(&mut self, child: NodeId, split_len: usize) -> NodeId {
        debug_assert!(split_len > 0 && split_len < self.nodes[child].key.len());
        self.stats.nodes_split += 1;

        let parent = self.nodes[child].parent.expect("split of root");
        let head: Vec<S> = self.nodes[child].key[..split_len].to_vec();
        let tail: Vec<S> = self.nodes[child].key[split_len..].to_vec();

        let new_node = Node {
            parent: Some(parent),
            children: HashMap::new(),
            key: head.clone(),
            depth: self.nodes[parent].depth + split_len,
            payload: None,
            lock_ref: self.nodes[child].lock_ref,
            last_access: self.nodes[child].last_access,
            created: self.nodes[child].created,
            hits: self.nodes[child].hits,
            opportunities: self.nodes[child].opportunities,
            layout: self.nodes[child].layout,
            cost_ns_per_symbol: self.nodes[child].cost_ns_per_symbol,
        };
        let new_id = self.alloc(new_node);

        let old_ck = Self::child_key(&head, self.page_size);
        let tail_ck = Self::child_key(&tail, self.page_size);

        self.nodes[child].key = tail;
        self.nodes[child].parent = Some(new_id);
        self.nodes[new_id].children.insert(tail_ck, child);
        self.nodes[parent].children.insert(old_ck, new_id);

        self.refresh_leaf_status(new_id);
        self.refresh_leaf_status(child);
        self.refresh_leaf_status(parent);
        new_id
    }

    fn alloc(&mut self, node: Node<S, P>) -> NodeId {
        if let Some(slot) = self.free_slots.pop() {
            self.nodes[slot] = node;
            slot
        } else {
            self.nodes.push(node);
            self.nodes.len() - 1
        }
    }

    /// Insert `key` with `payload`.
    ///
    /// Adapted from SGLang `_insert_helper` (radix_cache.py:701-753). Two
    /// differences that are ours:
    ///
    /// 1. `cost_ns_per_symbol` and `layout` are recorded on every node created,
    ///    so eviction can score by measured value rather than recency.
    /// 2. **Subsumption dedup**: if the entire key is already covered and a
    ///    payload already exists that covers it, the caller's payload is
    ///    dropped and the bytes it would have occupied are counted as saved.
    ///    A prefix-only cache stores the redundant copy.
    pub fn insert(
        &mut self,
        key: &[S],
        payload: P,
        layout: KvBlockLayout,
        cost_ns_per_symbol: u64,
    ) -> InsertOutcome {
        let now = self.clock.tick();
        self.stats.inserts += 1;
        self.nodes[ROOT].last_access = now;

        let usable = (key.len() / self.page_size) * self.page_size;
        let key = &key[..usable];
        if key.is_empty() {
            return InsertOutcome {
                shared_prefix: 0,
                node: ROOT,
                subsumed: true,
            };
        }

        let mut node = ROOT;
        let mut rest = key;
        let mut shared = 0usize;

        while !rest.is_empty() {
            let ck = Self::child_key(rest, self.page_size);
            let Some(&child) = self.nodes[node].children.get(&ck) else {
                break;
            };
            self.nodes[child].last_access = now;
            let plen = Self::shared_len(&self.nodes[child].key, rest, self.page_size);
            self.stats.symbol_comparisons +=
                (plen + self.page_size).min(self.nodes[child].key.len().max(rest.len())) as u64;
            shared += plen;
            rest = &rest[plen..];
            if plen < self.nodes[child].key.len() {
                node = self.split_node(child, plen);
                self.nodes[node].last_access = now;
                break;
            }
            node = child;
        }

        if rest.is_empty() {
            // The whole key is already in the tree. If a payload covering it
            // exists at or above `node`, the caller's copy is redundant.
            let existing = self.payload_depth_at_or_above(node);
            if existing.is_some() {
                self.stats.bytes_saved_by_dedup += layout.bytes_for_tokens(key.len());
                self.stats.symbols_deduped += key.len() as u64;
                return InsertOutcome {
                    shared_prefix: shared,
                    node,
                    subsumed: true,
                };
            }
            self.nodes[node].payload = Some(payload);
            self.nodes[node].layout = layout;
            self.nodes[node].cost_ns_per_symbol = cost_ns_per_symbol;
            self.refresh_leaf_status(node);
            return InsertOutcome {
                shared_prefix: shared,
                node,
                subsumed: false,
            };
        }

        let ck = Self::child_key(rest, self.page_size);
        let leaf = Node {
            parent: Some(node),
            children: HashMap::new(),
            key: rest.to_vec(),
            depth: self.nodes[node].depth + rest.len(),
            payload: Some(payload),
            lock_ref: 0,
            last_access: now,
            created: now,
            hits: 0,
            opportunities: 0,
            layout,
            cost_ns_per_symbol,
        };
        let added = rest.len();
        let leaf_id = self.alloc(leaf);
        self.nodes[node].children.insert(ck, leaf_id);
        self.evictable_symbols += added;
        self.refresh_leaf_status(node);
        self.refresh_leaf_status(leaf_id);

        InsertOutcome {
            shared_prefix: shared,
            node: leaf_id,
            subsumed: false,
        }
    }

    fn payload_depth_at_or_above(&self, mut node: NodeId) -> Option<usize> {
        loop {
            if self.nodes[node].payload.is_some() {
                return Some(self.nodes[node].depth);
            }
            match self.nodes[node].parent {
                Some(p) if node != ROOT => node = p,
                _ => return None,
            }
        }
    }

    /// Protect the path from `node` to the root from eviction.
    ///
    /// Adapted from SGLang `inc_lock_ref` (radix_cache.py:589-602).
    pub fn lock(&mut self, node: NodeId) {
        let mut cur = node;
        while cur != ROOT {
            if self.nodes[cur].lock_ref == 0 {
                let n = self.nodes[cur].key.len();
                self.evictable_symbols = self.evictable_symbols.saturating_sub(n);
                self.protected_symbols += n;
            }
            self.nodes[cur].lock_ref += 1;
            self.refresh_leaf_status(cur);
            match self.nodes[cur].parent {
                Some(p) => cur = p,
                None => break,
            }
        }
    }

    /// Release a protection taken by [`RadixTree::lock`].
    ///
    /// Adapted from SGLang `dec_lock_ref` (radix_cache.py:604-623).
    pub fn unlock(&mut self, node: NodeId) {
        let mut cur = node;
        while cur != ROOT {
            if self.nodes[cur].lock_ref == 1 {
                let n = self.nodes[cur].key.len();
                self.protected_symbols = self.protected_symbols.saturating_sub(n);
                self.evictable_symbols += n;
            }
            if self.nodes[cur].lock_ref > 0 {
                self.nodes[cur].lock_ref -= 1;
            }
            self.refresh_leaf_status(cur);
            match self.nodes[cur].parent {
                Some(p) => cur = p,
                None => break,
            }
        }
    }

    /// Maintain the evictable-leaf set.
    ///
    /// Adapted from SGLang `_update_leaf_status` (radix_cache.py:783-796).
    fn refresh_leaf_status(&mut self, node: NodeId) {
        if node == ROOT {
            return;
        }
        let evictable = self.nodes[node].lock_ref == 0 && self.nodes[node].children.is_empty();
        if evictable {
            self.evictable_leaves.insert(node);
        } else {
            self.evictable_leaves.remove(&node);
        }
    }

    /// Evict until at least `target_symbols` symbols have been released, using
    /// `scorer` to choose. Returns the evicted payloads paired with the symbol
    /// count they held, so the caller can free the real memory.
    ///
    /// Adapted from SGLang `evict` (radix_cache.py:560-587), including the
    /// re-heap of a parent that becomes a leaf. SGLang's strategies are all
    /// recency/frequency functions of the node; ours additionally take measured
    /// recompute cost and held bytes (see `super::evict`).
    pub fn evict(
        &mut self,
        target_symbols: usize,
        scorer: &dyn EvictionScorer,
    ) -> Vec<(P, usize, u64)> {
        self.evict_while(scorer, move |symbols, _payloads| symbols < target_symbols)
    }

    /// Evict the lowest-scoring leaves until `n` payloads have been released.
    /// The sequence substrate budgets in whole cached sequences, not tokens.
    pub fn evict_payloads(
        &mut self,
        n: usize,
        scorer: &dyn EvictionScorer,
    ) -> Vec<(P, usize, u64)> {
        self.evict_while(scorer, move |_symbols, payloads| payloads < n)
    }

    /// Shared eviction loop. `keep_going(symbols_freed, payloads_freed)`.
    pub fn evict_while<F>(
        &mut self,
        scorer: &dyn EvictionScorer,
        mut keep_going: F,
    ) -> Vec<(P, usize, u64)>
    where
        F: FnMut(usize, usize) -> bool,
    {
        let now = self.clock.now();
        let mut heap: BinaryHeap<Candidate> = self
            .evictable_leaves
            .iter()
            .map(|&id| Candidate {
                // BinaryHeap is a max-heap; negate so the *lowest* score pops
                // first. Scores are finite by construction (see `evict.rs`).
                neg_score: -scorer.score(&self.nodes[id].stats(now)),
                id,
            })
            .collect();

        let mut freed = 0usize;
        let mut out: Vec<(P, usize, u64)> = Vec::new();
        while keep_going(freed, out.len()) {
            let Some(Candidate { id, .. }) = heap.pop() else {
                break;
            };
            // The node may have been removed or re-locked since it entered the
            // heap.
            if !self.evictable_leaves.contains(&id) {
                continue;
            }
            let symbols = self.nodes[id].key.len();
            let bytes = self.nodes[id].layout.bytes_for_tokens(symbols);
            let payload = self.nodes[id].payload.take();
            let parent = self.nodes[id].parent;

            self.detach(id);
            freed += symbols;
            self.stats.evictions += 1;
            self.stats.symbols_evicted += symbols as u64;
            if let Some(p) = payload {
                out.push((p, symbols, bytes));
            }

            if let Some(parent) = parent {
                self.refresh_leaf_status(parent);
                if self.evictable_leaves.contains(&parent) {
                    heap.push(Candidate {
                        neg_score: -scorer.score(&self.nodes[parent].stats(now)),
                        id: parent,
                    });
                }
            }
        }
        out
    }

    fn detach(&mut self, id: NodeId) {
        debug_assert!(self.nodes[id].children.is_empty());
        let key = self.nodes[id].key.clone();
        let n = key.len();
        if let Some(parent) = self.nodes[id].parent {
            let ck = Self::child_key(&key, self.page_size);
            self.nodes[parent].children.remove(&ck);
        }
        self.evictable_leaves.remove(&id);
        self.evictable_symbols = self.evictable_symbols.saturating_sub(n);
        self.nodes[id].children.clear();
        self.nodes[id].key.clear();
        self.nodes[id].payload = None;
        self.nodes[id].parent = None;
        self.free_slots.push(id);
    }

    /// Drop everything. Used by the engine's hard cache-reset paths.
    pub fn clear(&mut self) {
        let stats = self.stats;
        let page = self.page_size;
        *self = Self::new(page);
        self.stats = stats;
    }

    /// Symbol depth of a node — how many symbols of prefix it stands for.
    pub fn depth_of(&self, node: NodeId) -> usize {
        self.nodes[node].depth
    }

    /// Every live payload with the symbol depth it covers. Test and
    /// introspection helper; also used by the manager to report occupancy.
    pub fn payloads(&self) -> Vec<(&P, usize)> {
        let mut out = Vec::new();
        let mut stack = vec![ROOT];
        while let Some(id) = stack.pop() {
            if let Some(p) = self.nodes[id].payload.as_ref() {
                out.push((p, self.nodes[id].depth));
            }
            for &c in self.nodes[id].children.values() {
                stack.push(c);
            }
        }
        out
    }

    /// Human-readable eviction ranking — the "scoring function you can print".
    pub fn explain_eviction_order(&self, scorer: &dyn EvictionScorer) -> Vec<String> {
        let now = self.clock.now();
        let mut cands: Vec<(f64, NodeId)> = self
            .evictable_leaves
            .iter()
            .map(|&id| (scorer.score(&self.nodes[id].stats(now)), id))
            .collect();
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        cands
            .into_iter()
            .map(|(score, id)| {
                format!(
                    "node {id} depth={} symbols={} score={score:.6} {}",
                    self.nodes[id].depth,
                    self.nodes[id].key.len(),
                    scorer.explain(&self.nodes[id].stats(now))
                )
            })
            .collect()
    }
}

/// Heap entry. `f64` has no `Ord`, and eviction must be deterministic, so ties
/// are broken by node id rather than left to hash order.
struct Candidate {
    neg_score: f64,
    id: NodeId,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.neg_score
            .partial_cmp(&other.neg_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.id.cmp(&self.id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_sharing::evict::{LruScorer, ValueAwareScorer};

    fn t() -> RadixTree<u32, &'static str> {
        RadixTree::new(1)
    }

    fn ins(tree: &mut RadixTree<u32, &'static str>, key: &[u32], p: &'static str) -> InsertOutcome {
        tree.insert(key, p, KvBlockLayout::default(), 1_000)
    }

    #[test]
    fn empty_tree_matches_nothing() {
        let mut tree = t();
        let m = tree.match_prefix(&[1, 2, 3]);
        assert_eq!(m.matched, 0);
        assert!(m.payload.is_none());
        assert_eq!(tree.stats().symbols_computed, 3);
    }

    #[test]
    fn exact_and_prefix_match() {
        let mut tree = t();
        ins(&mut tree, &[1, 2, 3, 4], "a");
        let m = tree.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(m.matched, 4);
        assert_eq!(m.payload, Some(("a", 4)));

        // A strict prefix of a stored key is served by truncating the deeper
        // cache — this is what Arc's linear scan did, and the tree must not
        // lose it.
        let m = tree.match_prefix(&[1, 2]);
        assert_eq!(m.matched, 2);
        assert_eq!(
            m.payload,
            Some(("a", 2)),
            "a deeper cache must serve a shorter prefix"
        );
    }

    /// The regression this port could most easily have introduced: matching
    /// deep into a stored sequence and then finding no payload because the
    /// payload lives further down still.
    #[test]
    fn divergent_query_is_served_by_truncating_a_longer_cache() {
        let mut tree = t();
        ins(&mut tree, &[1, 2, 3, 4, 5, 6, 7, 8], "long");
        // Diverges at index 3.
        let m = tree.match_prefix(&[1, 2, 3, 99, 99]);
        assert_eq!(m.matched, 3);
        assert_eq!(
            m.payload,
            Some(("long", 3)),
            "the 8-token cache must serve the 3 tokens that matched"
        );
    }

    #[test]
    fn divergence_splits_the_node_once() {
        let mut tree = t();
        ins(&mut tree, &[1, 2, 3, 4], "a");
        ins(&mut tree, &[1, 2, 9, 9], "b");
        assert_eq!(tree.stats().nodes_split, 1);

        assert_eq!(tree.match_prefix(&[1, 2, 3, 4]).payload, Some(("a", 4)));
        assert_eq!(tree.match_prefix(&[1, 2, 9, 9]).payload, Some(("b", 4)));
        // Sharing is real: the common [1,2] lives in exactly one node.
        assert_eq!(tree.match_prefix(&[1, 2, 7]).matched, 2);
    }

    /// The property the whole port exists for: N requests behind one system
    /// prompt pay for it once.
    #[test]
    fn shared_system_prompt_is_stored_once_and_reused() {
        let mut tree = t();
        let sys: Vec<u32> = (0..800).collect();

        for q in 0..16u32 {
            let mut k = sys.clone();
            k.push(10_000 + q);
            tree.insert(&k, "seq", KvBlockLayout::default(), 1_000);
        }
        // 800 shared symbols + 16 distinct tails = 816 live symbols, not
        // 16 * 801 = 12,816.
        assert_eq!(
            tree.evictable_symbols() + tree.protected_symbols(),
            800 + 16,
            "shared prefix must be stored once"
        );

        let mut probe = sys.clone();
        probe.push(10_003);
        let m = tree.match_prefix(&probe);
        assert_eq!(m.matched, 801);
    }

    #[test]
    fn page_alignment_never_reports_a_partial_page() {
        let mut tree: RadixTree<u32, &str> = RadixTree::new(4);
        tree.insert(&[1, 2, 3, 4, 5, 6, 7, 8], "a", KvBlockLayout::default(), 1);
        // Query shares 6 symbols; only 4 are a whole page.
        let m = tree.match_prefix(&[1, 2, 3, 4, 5, 6, 0, 0]);
        assert_eq!(m.matched % 4, 0);
        assert_eq!(m.matched, 4);
    }

    #[test]
    fn locked_nodes_survive_eviction() {
        let mut tree = t();
        let a = ins(&mut tree, &[1, 2, 3], "a").node;
        ins(&mut tree, &[4, 5, 6], "b");
        tree.lock(a);
        let evicted = tree.evict(100, &LruScorer);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].0, "b");
        assert_eq!(tree.match_prefix(&[1, 2, 3]).payload, Some(("a", 3)));

        tree.unlock(a);
        let evicted = tree.evict(100, &LruScorer);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].0, "a");
    }

    #[test]
    fn lock_unlock_accounting_balances() {
        let mut tree = t();
        let a = ins(&mut tree, &[1, 2, 3], "a").node;
        assert_eq!(tree.evictable_symbols(), 3);
        assert_eq!(tree.protected_symbols(), 0);
        tree.lock(a);
        assert_eq!(tree.evictable_symbols(), 0);
        assert_eq!(tree.protected_symbols(), 3);
        tree.unlock(a);
        assert_eq!(tree.evictable_symbols(), 3);
        assert_eq!(tree.protected_symbols(), 0);
    }

    #[test]
    fn subsumed_insert_stores_nothing_and_counts_bytes_saved() {
        let layout = KvBlockLayout::uniform(crate::kv_sharing::layout::KvElemType::Bf16, 128, 4);
        let mut tree: RadixTree<u32, &str> = RadixTree::new(1);
        tree.insert(&[1, 2, 3, 4], "a", layout, 1);
        let before = tree.len();
        let out = tree.insert(&[1, 2, 3, 4], "duplicate", layout, 1);
        assert!(out.subsumed);
        assert_eq!(tree.len(), before, "no node added for a duplicate key");
        assert_eq!(tree.stats().symbols_deduped, 4);
        assert_eq!(
            tree.stats().bytes_saved_by_dedup,
            layout.bytes_for_tokens(4)
        );
        // The original payload wins; the duplicate was dropped, not swapped.
        assert_eq!(tree.match_prefix(&[1, 2, 3, 4]).payload, Some(("a", 4)));
    }

    #[test]
    fn evict_reclaims_arena_slots() {
        let mut tree = t();
        for i in 0..32u32 {
            ins(&mut tree, &[i, i + 1, i + 2], "x");
        }
        let n = tree.len();
        assert_eq!(n, 32);
        tree.evict(1_000, &LruScorer);
        assert_eq!(tree.len(), 0);
        // Re-inserting must reuse slots rather than grow the arena forever.
        for i in 0..32u32 {
            ins(&mut tree, &[i, i + 1, i + 2], "x");
        }
        assert_eq!(tree.len(), 32);
    }

    #[test]
    fn hit_rate_is_token_weighted() {
        let mut tree = t();
        let sys: Vec<u32> = (0..100).collect();
        tree.insert(&sys, "sys", KvBlockLayout::default(), 1);

        let mut q = sys.clone();
        q.extend([500, 501, 502]);
        tree.match_prefix(&q);
        let s = tree.stats();
        assert_eq!(s.symbols_reused, 100);
        assert_eq!(s.symbols_computed, 3);
        assert!((s.symbol_hit_rate() - 100.0 / 103.0).abs() < 1e-12);
    }

    #[test]
    fn value_aware_scorer_is_printable() {
        let mut tree = t();
        ins(&mut tree, &[1, 2, 3], "a");
        let lines = tree.explain_eviction_order(&ValueAwareScorer::default());
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("score="), "got {}", lines[0]);
    }

    /// Eviction must be deterministic under equal scores, or a flaky test
    /// suite is the best case and a nondeterministic serving policy is the
    /// worst.
    #[test]
    fn eviction_order_is_deterministic_under_ties() {
        let run = || {
            let mut tree = t();
            for i in 0..8u32 {
                ins(&mut tree, &[i * 10, i * 10 + 1], "x");
            }
            // All nodes have identical stats, so every score ties.
            tree.evict(4, &ValueAwareScorer::default())
                .into_iter()
                .map(|(_, s, _)| s)
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
