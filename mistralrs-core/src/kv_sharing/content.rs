//! Content-addressed segment identity, and the cross-prefix reuse meter.
//!
//! # Improvement 2, and the honest boundary around it
//!
//! A radix tree matches from token zero. Two requests that embed the same
//! 40-page document lose the whole document if one of them has "hi" in front.
//! The brief asks for content hashing so identical blocks are shared "wherever
//! they appear".
//!
//! **Half of that is exact and is shipped here. The other half is not a
//! caching change, it is a quality change, and this module refuses to make it
//! silently.** The reason is causality, not effort:
//!
//! > For a causal transformer, `K_i` and `V_i` at layer `l >= 1` are functions
//! > of the hidden state `h_i^(l-1)`, which is produced by attention over
//! > tokens `0..=i`. Change the prefix and you change every K and V after it,
//! > even though the token ids are identical. RoPE compounds it: the same
//! > token at a different absolute position gets a different rotation.
//!
//! So a block's KV is reusable only where its *entire causal history* matches
//! — which is exactly the condition a prefix radix tree already tests. This is
//! why Radix "structurally cannot" do position-independent sharing: it is a
//! bound from the model, not an oversight in the data structure. Reusing KV
//! across differing prefixes is approximate, and the published techniques that
//! do it (CacheBlend-style selective recomputation, Prompt-Cache-style
//! position-independent modules) buy hit rate by spending quality and need
//! their own quality gate.
//!
//! What this module therefore ships:
//!
//! 1. **[`ContentIndex`] — exact, fully verified segment identity.** A 128-bit
//!    digest narrows the search; **both** the segment tokens **and** the entire
//!    causal prefix are then compared element by element before any verdict is
//!    returned. No hash value ever decides anything on its own, so no
//!    collision — in the content digest or in a causal digest — can produce a
//!    false share.
//! 2. **[`CrossPrefixMeter`] — the measurement.** It counts the tokens that
//!    position-independent reuse *would* have shared, per workload, so the
//!    decision to fund a CacheBlend-class corrected path is made against a
//!    number instead of an intuition. It never shares anything.
//!
//! # A defect this design was chosen to eliminate
//!
//! The first version of this module identified causal history by a *digest* of
//! the preceding tokens and compared digests. That is unsafe in exactly the way
//! this module claims not to be: a causal-digest collision reports
//! [`ContentMatch::ExactReusable`] for two segments with different histories,
//! which is a silent false share. It was caught by
//! `meter_with_a_colliding_digest_still_reports_correctly`. Occurrences now
//! retain a shared handle to their whole sequence and compare the real prefix.

use std::collections::HashMap;
use std::sync::Arc;

/// 128-bit content digest over a token segment.
///
/// Two independent 64-bit FNV-1a streams with different bases and primes. The
/// width matters less than it looks: the digest only narrows the candidate set
/// and **every candidate is then verified token by token**, so hash quality
/// affects performance, never correctness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentDigest {
    pub lo: u64,
    pub hi: u64,
}

/// How a segment's digest is computed. Swappable so tests can inject a
/// degenerate hash and prove verification carries the safety.
pub trait SegmentDigest {
    fn digest(&self, tokens: &[u32]) -> ContentDigest;
}

/// Default digest: two FNV-1a-64 streams.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fnv1a128;

const FNV64_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;
// Second stream: a different basis and a different odd prime, so the two
// streams do not move together.
const FNV64_BASIS_B: u64 = 0x9e37_79b9_7f4a_7c15;
const FNV64_PRIME_B: u64 = 0xff51_afd7_ed55_8ccd;

impl SegmentDigest for Fnv1a128 {
    fn digest(&self, tokens: &[u32]) -> ContentDigest {
        let mut lo = FNV64_BASIS;
        let mut hi = FNV64_BASIS_B;
        // Length is folded in so `[1,2]` and `[1,2,0]` differ.
        lo ^= tokens.len() as u64;
        hi ^= (tokens.len() as u64).rotate_left(32);
        for &t in tokens {
            for b in t.to_le_bytes() {
                lo ^= b as u64;
                lo = lo.wrapping_mul(FNV64_PRIME);
                hi ^= (b as u64).wrapping_add(0x9e);
                hi = hi.wrapping_mul(FNV64_PRIME_B);
            }
        }
        ContentDigest { lo, hi }
    }
}

/// A segment identified by position inside its own sequence.
///
/// Carrying the whole sequence rather than a copy of the segment is what makes
/// causal verification exact: the causal history *is* `context[..offset]`, and
/// it can be compared directly instead of trusted through a hash.
#[derive(Debug, Clone, Copy)]
pub struct SegmentRef<'a> {
    pub context: &'a [u32],
    pub offset: usize,
    pub len: usize,
}

impl<'a> SegmentRef<'a> {
    pub fn new(context: &'a [u32], offset: usize, len: usize) -> Self {
        debug_assert!(offset + len <= context.len());
        Self {
            context,
            offset,
            len,
        }
    }
    pub fn segment(&self) -> &'a [u32] {
        &self.context[self.offset..self.offset + self.len]
    }
    pub fn causal(&self) -> &'a [u32] {
        &self.context[..self.offset]
    }
}

/// One stored occurrence. Holds a shared handle to the full sequence, so both
/// the segment and its entire causal prefix stay available for verification at
/// the cost of one `Arc` per recorded request.
#[derive(Debug, Clone)]
struct Occurrence {
    context: Arc<[u32]>,
    offset: usize,
    len: usize,
}

impl Occurrence {
    fn segment(&self) -> &[u32] {
        &self.context[self.offset..self.offset + self.len]
    }
    fn causal(&self) -> &[u32] {
        &self.context[..self.offset]
    }
}

/// Exact, verified content index over token segments.
pub struct ContentIndex<D: SegmentDigest = Fnv1a128> {
    digest: D,
    by_content: HashMap<ContentDigest, Vec<Occurrence>>,
    verifications: u64,
    rejected_collisions: u64,
}

/// Outcome of an index probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentMatch {
    /// No verified occurrence of this content is known.
    Absent,
    /// Known, with a byte-identical causal history — reuse would be bit-exact.
    ExactReusable,
    /// Known, but every occurrence sits under a different causal prefix.
    /// **Not reusable.** Counted by the meter and nothing more.
    CausallyDifferent { occurrences: usize },
}

impl Default for ContentIndex<Fnv1a128> {
    fn default() -> Self {
        Self::new(Fnv1a128)
    }
}

impl<D: SegmentDigest> ContentIndex<D> {
    pub fn new(digest: D) -> Self {
        Self {
            digest,
            by_content: HashMap::new(),
            verifications: 0,
            rejected_collisions: 0,
        }
    }

    pub fn digest_of(&self, tokens: &[u32]) -> ContentDigest {
        self.digest.digest(tokens)
    }

    /// Digest matches that failed token verification, i.e. real collisions
    /// caught. Nonzero in production is a signal to widen the digest; it is
    /// never a correctness problem.
    pub fn rejected_collisions(&self) -> u64 {
        self.rejected_collisions
    }

    pub fn verifications(&self) -> u64 {
        self.verifications
    }

    pub fn len(&self) -> usize {
        self.by_content.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.by_content.is_empty()
    }

    /// Record a segment of `context`. `context` is cloned into an `Arc` once
    /// per call; pass the same `Arc` for every segment of one request via
    /// [`ContentIndex::record_arc`] to share it.
    pub fn record(&mut self, seg: SegmentRef<'_>) {
        let ctx: Arc<[u32]> = Arc::from(seg.context);
        self.record_arc(&ctx, seg.offset, seg.len);
    }

    /// Record a segment of an already-shared context.
    pub fn record_arc(&mut self, context: &Arc<[u32]>, offset: usize, len: usize) {
        if len == 0 || offset + len > context.len() {
            return;
        }
        let occ = Occurrence {
            context: Arc::clone(context),
            offset,
            len,
        };
        let d = self.digest.digest(occ.segment());
        let bucket = self.by_content.entry(d).or_default();
        for existing in bucket.iter() {
            self.verifications += 1;
            if existing.segment() != occ.segment() {
                self.rejected_collisions += 1;
                continue;
            }
            if existing.causal() == occ.causal() {
                // Already have this exact (content, history) pair.
                return;
            }
        }
        bucket.push(occ);
    }

    /// Probe for `seg`.
    ///
    /// Both the segment tokens and the full causal prefix are compared element
    /// by element on every digest hit, so a colliding digest can only cost a
    /// wasted comparison. It can never return another segment's content, and
    /// it can never call two different histories identical.
    pub fn probe(&mut self, seg: SegmentRef<'_>) -> ContentMatch {
        if seg.len == 0 {
            return ContentMatch::Absent;
        }
        let d = self.digest.digest(seg.segment());
        let Some(bucket) = self.by_content.get(&d) else {
            return ContentMatch::Absent;
        };
        let mut causally_different = 0usize;
        let mut collisions = 0u64;
        let mut verifications = 0u64;
        let mut exact = false;
        for occ in bucket {
            verifications += 1;
            if occ.segment() != seg.segment() {
                collisions += 1;
                continue;
            }
            if occ.causal() == seg.causal() {
                exact = true;
                break;
            }
            causally_different += 1;
        }
        self.verifications += verifications;
        self.rejected_collisions += collisions;

        if exact {
            ContentMatch::ExactReusable
        } else if causally_different > 0 {
            ContentMatch::CausallyDifferent {
                occurrences: causally_different,
            }
        } else {
            ContentMatch::Absent
        }
    }

    /// Distinct absolute offsets at which verified-identical content has been
    /// seen.
    pub fn offsets_of(&mut self, tokens: &[u32]) -> Vec<usize> {
        let d = self.digest.digest(tokens);
        let Some(bucket) = self.by_content.get(&d) else {
            return Vec::new();
        };
        self.verifications += bucket.len() as u64;
        let mut out: Vec<usize> = bucket
            .iter()
            .filter(|o| o.segment() == tokens)
            .map(|o| o.offset)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn clear(&mut self) {
        self.by_content.clear();
    }
}

/// What position-independent reuse *would* buy on the traffic actually seen.
///
/// A measurement instrument, not a cache. It shares nothing. Its output is the
/// number that decides whether a CacheBlend-class corrected reuse path is worth
/// building.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CrossPrefixReport {
    /// Segments whose content was already known but whose causal history
    /// differed — exactly the case a prefix tree structurally misses.
    pub duplicate_segments: u64,
    /// Tokens those segments cover.
    pub duplicate_tokens: u64,
    /// Bytes those tokens would occupy at the layout they were seen with.
    pub duplicate_bytes: u64,
    /// Segments that were content-identical *and* causally identical, i.e.
    /// already reusable exactly.
    pub exact_segments: u64,
    pub exact_tokens: u64,
    pub segments_seen: u64,
    pub tokens_seen: u64,
}

impl CrossPrefixReport {
    /// Fraction of offered tokens that only position-independent (approximate)
    /// reuse could have shared. **An upper bound on an unbuilt feature's
    /// benefit, not a benefit we have.**
    #[allow(clippy::cast_precision_loss)]
    pub fn unreachable_share(&self) -> f64 {
        if self.tokens_seen == 0 {
            return 0.0;
        }
        self.duplicate_tokens as f64 / self.tokens_seen as f64
    }
}

/// Meters cross-prefix duplication over a stream of requests.
pub struct CrossPrefixMeter<D: SegmentDigest = Fnv1a128> {
    index: ContentIndex<D>,
    report: CrossPrefixReport,
    /// Tokens per metered segment. A fixed granularity is what makes "the same
    /// document behind different prefixes" detectable at all.
    segment_len: usize,
}

impl Default for CrossPrefixMeter<Fnv1a128> {
    fn default() -> Self {
        Self::new(Fnv1a128, 256)
    }
}

impl<D: SegmentDigest> CrossPrefixMeter<D> {
    pub fn new(digest: D, segment_len: usize) -> Self {
        assert!(segment_len > 0, "segment_len must be > 0");
        Self {
            index: ContentIndex::new(digest),
            report: CrossPrefixReport::default(),
            segment_len,
        }
    }

    pub fn report(&self) -> CrossPrefixReport {
        self.report
    }

    pub fn rejected_collisions(&self) -> u64 {
        self.index.rejected_collisions()
    }

    /// Offer a whole request's tokens.
    pub fn observe(&mut self, tokens: &[u32], layout: &super::layout::KvBlockLayout) {
        if tokens.len() < self.segment_len {
            return;
        }
        let ctx: Arc<[u32]> = Arc::from(tokens);
        let mut offset = 0usize;
        while offset + self.segment_len <= ctx.len() {
            let seg = SegmentRef::new(&ctx, offset, self.segment_len);
            self.report.segments_seen += 1;
            self.report.tokens_seen += self.segment_len as u64;
            match self.index.probe(seg) {
                ContentMatch::ExactReusable => {
                    self.report.exact_segments += 1;
                    self.report.exact_tokens += self.segment_len as u64;
                }
                ContentMatch::CausallyDifferent { .. } => {
                    self.report.duplicate_segments += 1;
                    self.report.duplicate_tokens += self.segment_len as u64;
                    self.report.duplicate_bytes += layout.bytes_for_tokens(self.segment_len);
                }
                ContentMatch::Absent => {}
            }
            self.index.record_arc(&ctx, offset, self.segment_len);
            offset += self.segment_len;
        }
    }

    pub fn clear(&mut self) {
        self.index.clear();
        self.report = CrossPrefixReport::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_sharing::layout::KvBlockLayout;

    /// A digest that collides on absolutely everything. If verification is
    /// what keeps us safe, every result below must be identical with this in
    /// place — and the safety claim is proved, not asserted.
    struct AlwaysCollide;
    impl SegmentDigest for AlwaysCollide {
        fn digest(&self, _tokens: &[u32]) -> ContentDigest {
            ContentDigest { lo: 7, hi: 7 }
        }
    }

    #[test]
    fn digest_separates_content_and_length() {
        let d = Fnv1a128;
        assert_ne!(d.digest(&[1, 2, 3]), d.digest(&[1, 2, 4]));
        assert_ne!(d.digest(&[1, 2, 3]), d.digest(&[3, 2, 1]));
        assert_ne!(d.digest(&[1, 2]), d.digest(&[1, 2, 0]));
        assert_eq!(d.digest(&[1, 2, 3]), d.digest(&[1, 2, 3]));
    }

    #[test]
    fn exact_reuse_requires_identical_causal_history() {
        let mut idx = ContentIndex::default();
        // Same 4-token document behind two different 3-token prefixes.
        let a: Vec<u32> = vec![1, 2, 3, 10, 11, 12, 13];
        let b: Vec<u32> = vec![9, 9, 9, 10, 11, 12, 13];
        assert_ne!(&a[..3], &b[..3], "fixture must discriminate the prefixes");
        assert_eq!(&a[3..], &b[3..], "fixture must share the document");

        idx.record(SegmentRef::new(&a, 3, 4));
        assert_eq!(
            idx.probe(SegmentRef::new(&a, 3, 4)),
            ContentMatch::ExactReusable
        );
        assert_eq!(
            idx.probe(SegmentRef::new(&b, 3, 4)),
            ContentMatch::CausallyDifferent { occurrences: 1 },
            "same tokens behind a different prefix must NOT be reported reusable"
        );
    }

    /// THE safety proof for improvement 2. With a digest that collides on
    /// everything, the index must still never confuse two segments.
    #[test]
    fn collision_proof_verification_rejects_a_false_share() {
        let mut idx = ContentIndex::new(AlwaysCollide);
        let a = [1u32, 2, 3, 4];
        let b = [5u32, 6, 7, 8];

        idx.record(SegmentRef::new(&a, 0, 4));

        assert_eq!(
            idx.digest_of(&a),
            idx.digest_of(&b),
            "fixture is degenerate unless the digests actually collide"
        );
        assert_eq!(
            idx.probe(SegmentRef::new(&b, 0, 4)),
            ContentMatch::Absent,
            "a colliding digest must never yield another segment's content"
        );
        assert!(
            idx.rejected_collisions() > 0,
            "the collision must be observed and counted, not silently ignored"
        );
        assert_eq!(
            idx.probe(SegmentRef::new(&a, 0, 4)),
            ContentMatch::ExactReusable
        );
    }

    /// The second half of the safety proof, and the one that caught a real
    /// defect: causal histories are compared, never hashed-and-trusted.
    #[test]
    fn colliding_digest_cannot_fake_an_identical_causal_history() {
        let mut idx = ContentIndex::new(AlwaysCollide);
        let a: Vec<u32> = vec![1, 2, 3, 10, 11, 12, 13];
        let b: Vec<u32> = vec![9, 9, 9, 10, 11, 12, 13];
        idx.record(SegmentRef::new(&a, 3, 4));
        assert_eq!(
            idx.probe(SegmentRef::new(&b, 3, 4)),
            ContentMatch::CausallyDifferent { occurrences: 1 },
            "a causal-digest collision must not be able to claim exact reuse"
        );
    }

    #[test]
    fn meter_counts_the_document_behind_different_prefixes() {
        let layout = KvBlockLayout::uniform(crate::kv_sharing::layout::KvElemType::Bf16, 128, 4);
        let mut meter = CrossPrefixMeter::new(Fnv1a128, 4);

        let doc: Vec<u32> = (100..116).collect(); // 16 tokens = 4 segments

        // Request A: doc at offset 0.
        let mut a = doc.clone();
        a.extend([900, 901, 902, 903]);
        meter.observe(&a, &layout);

        // Request B: "hi" (4 tokens) then the same doc. A prefix tree gets
        // nothing from A here.
        let mut b = vec![1u32, 2, 3, 4];
        b.extend(doc.iter().copied());
        meter.observe(&b, &layout);

        let r = meter.report();
        assert_eq!(
            r.duplicate_tokens, 16,
            "all 16 document tokens are content-duplicates under a different prefix"
        );
        assert_eq!(r.duplicate_bytes, layout.bytes_for_tokens(16));
        assert!(r.unreachable_share() > 0.0);
        assert_eq!(
            r.exact_tokens, 0,
            "nothing here is exactly reusable — that is the whole point"
        );
    }

    #[test]
    fn meter_reports_exact_duplicates_as_exact() {
        let layout = KvBlockLayout::default();
        let mut meter = CrossPrefixMeter::new(Fnv1a128, 4);
        let req: Vec<u32> = (0..16).collect();
        meter.observe(&req, &layout);
        meter.observe(&req, &layout);
        let r = meter.report();
        assert_eq!(r.exact_tokens, 16);
        assert_eq!(
            r.duplicate_tokens, 0,
            "an identical request is exactly reusable, not a cross-prefix duplicate"
        );
    }

    /// The counts must be byte-identical under a digest that collides on
    /// everything, because verification — not hashing — decides every verdict.
    #[test]
    fn meter_with_a_colliding_digest_still_reports_correctly() {
        let layout = KvBlockLayout::default();
        let mut good = CrossPrefixMeter::new(Fnv1a128, 4);
        let mut bad = CrossPrefixMeter::new(AlwaysCollide, 4);

        let doc: Vec<u32> = (100..116).collect();
        let mut a = doc.clone();
        a.extend([900, 901, 902, 903]);
        let mut b = vec![1u32, 2, 3, 4];
        b.extend(doc.iter().copied());

        good.observe(&a, &layout);
        good.observe(&b, &layout);
        bad.observe(&a, &layout);
        bad.observe(&b, &layout);

        assert_eq!(good.report(), bad.report());
        assert!(
            bad.rejected_collisions() > 0,
            "the degenerate digest must actually be colliding, or this proves nothing"
        );
        assert_eq!(good.rejected_collisions(), 0);
    }

    #[test]
    fn offsets_of_reports_every_verified_position() {
        let mut idx = ContentIndex::default();
        let seq: Vec<u32> = vec![7, 7, 7, 0, 0, 7, 7, 7];
        idx.record(SegmentRef::new(&seq, 0, 3));
        idx.record(SegmentRef::new(&seq, 5, 3));
        idx.record(SegmentRef::new(&seq, 5, 3)); // dup, suppressed
        assert_eq!(idx.offsets_of(&[7, 7, 7]), vec![0, 5]);
    }
}
