//! Segment tables: the read primitive that is *a list of runs*, not one run.
//!
//! # Why this exists
//!
//! PagedAttention's read contract is `(block_table, context_len)` — **one
//! start, one length**. Every consumer in this tree assumes it:
//! [`super::kv_cache_manager::KVCacheManager::get_block_table`] returns one
//! row per sequence, and [`super::build_cu_seqlens_kv_from_context_lens`]
//! turns one length per sequence into the gather's `cu_seqlens`.
//!
//! That contract cannot express a read of **more than one disjoint region**
//! in a single attention. DeepSeek-V4 needs exactly that: `dsv4_attention`
//! folds a raw sliding window over recent tokens and a compressed summary of
//! everything older into **one** online softmax. It is the only model in the
//! tree that reports `supports_paged_attention == false`
//! (`pipeline/loaders/normal_loaders.rs:3231`, against the `Ok(true)` default
//! at `:143`).
//!
//! The tree already contains a hand-rolled, hardcoded, 2-region special case:
//! [`crate::pipeline::inputs_processor::PagedAttentionInputMetadata`] carries
//! **both** `block_tables`/`context_lens` (windowed) and
//! `full_block_tables`/`full_context_lens` (unwindowed), so a per-layer
//! sliding-window model can pick one *or* the other per layer. Two tables,
//! chosen between — never summed. This module replaces "a pair of tables you
//! choose between" with "a list of segments you read all of".
//!
//! # The load-bearing fact: no new kernel is needed for the gather
//!
//! `gather_kv_cache_kernel` (`mistralrs-paged-attn/src/cuda/`
//! `gather_kv_cache_kernel.cu:40`) takes `block_table [batch, max_blocks]`
//! and `cu_seq_lens [batch + 1]`, binary-searches `cu_seq_lens` for the row,
//! and reads `block_table[row * stride + offset / block_size]`. **Nothing in
//! it requires a row to be a sequence.** A row is an independent
//! (block-table, length) run gathered into a contiguous output slice.
//!
//! Therefore **a segment is a row**. A batch of `B` sequences with `S`
//! segments each flattens to a gather with `B * S` rows and needs *zero*
//! changes to the `.cu` file. See [`SegmentPlan`].
//!
//! # V4 in this vocabulary
//!
//! A V4 CSA/HCA layer reads **two** regions, each individually contiguous, at
//! different strides (`models/dsv4_attention.rs`):
//!
//! * **R1**, raw sliding window — the *suffix* `[t_k_full - keep, t_k_full)`
//!   where `keep = min(t_q + window - 1, t_k_full)` (`raw_keep_span`, `:272`).
//!   Anchored to the end, so its head slides forward: [`SegmentTable::slide`].
//! * **R2**, compressed KV — the *prefix* `[0, t_c)`, read in full and never
//!   windowed, one stored row per `ratio` sequence tokens (`:419-434`).
//!   Anchored to the start, so it only ever grows at the tail.
//!
//! Its attention sink is **not** a third region: it is a per-head learned
//! scalar contributing to the softmax denominator only, with zero cache bytes.
//!
//! Today those two regions are joined by `Tensor::cat(&[k, comp], 2)`
//! (`dsv4_attention.rs:431-432`), materialising a fresh `[B, 1, keep + t_c,
//! 512]` buffer **twice** (once for K, once for V) per layer per step. A
//! segment plan emits the two runs adjacent in one gather output, which is the
//! same bytes without the copy — so making the read primitive segment-shaped
//! deletes that `cat` rather than adding work.
//!
//! # What this deliberately does not cover
//!
//! `XsRollingCache.tail` — the compressor's recompute buffer — is `hidden`-wide
//! (4096), lives on a different layer set, and is bounded by
//! `span_groups * ratio + margin` tokens rather than by context. It is a third
//! storage class, not a KV region, and it cannot ride these block tables. See
//! `memory/mission/wave29-BC-v4-paged-attention.md` §4b.
//!
//! # Coordinates
//!
//! Every index in a [`Segment`] is in that **region's own** token coordinate
//! system, not the sequence's. For a raw region (window, sink, dense) one
//! region token is one sequence token. For a [`RegionKind::Compressed`]
//! region with ratio `r`, one region token is one compressed row summarising
//! `r` sequence tokens. This is why `first_token` is not comparable across
//! segments and why byte accounting is per-segment (see [`Segment::layout`]).

use crate::kv_sharing::KvBlockLayout;

/// What a region stores, which is not the same question as what it is read
/// for.
///
/// # Group ids are namespaced by *stored value*, not by read pattern
///
/// [`RegionKind::group_id`] feeds
/// [`super::block_hash::BlockHashWithGroupId`], which is what keeps two
/// blocks holding the same tokens but different *contents* from aliasing in
/// the prefix cache.
///
/// `Dense`, `Window` and `Sink` all store the raw K/V of the tokens they
/// cover, block-aligned in the same absolute coordinate frame, so a block
/// written by one is bit-identical to a block written by another and they
/// **share group 0 deliberately** — a sliding window can reuse a dense
/// prefix-cache hit. `Compressed` stores the output of a compressor, which
/// differs per ratio, so each ratio gets its own group.
///
/// The resulting rule is short enough to hold in your head: **group id is the
/// compression ratio, and 0 means uncompressed.**
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegionKind {
    /// The whole context as one run. This is the degenerate case — a
    /// [`SegmentTable`] holding exactly one `Dense` segment starting at token
    /// 0 is byte-for-byte the classic `(block_table, context_len)` contract.
    Dense,
    /// Raw K/V over the most recent `size` tokens. The head of the run slides
    /// forward; see [`SegmentTable::slide`].
    Window { size: usize },
    /// Raw K/V over the first `tokens` tokens of the sequence — sink tokens
    /// that are kept resident forever. Never slides, never freed before the
    /// sequence ends.
    ///
    /// **This is not V4's attention sink.** V4's sink is a per-head learned
    /// scalar (`[1, H, 1, 1]` F32) that enters `softmax_with_sinks` as a
    /// denominator-only column — it occupies **zero** cache bytes and needs no
    /// segment (`models/dsv4_attention.rs:149`, scalar reference at `:1008`).
    /// This variant exists for models whose sink genuinely is resident KV.
    Sink { tokens: usize },
    /// One compressed row per `ratio` sequence tokens.
    ///
    /// Construct via [`RegionKind::compressed`]; `ratio` must be at least 2 so
    /// it can never collide with the raw group.
    Compressed { ratio: u32 },
}

impl RegionKind {
    /// A compressed region at `ratio` tokens per stored row.
    ///
    /// Returns `None` for `ratio < 2`. V4 uses ratio 4 (CSA, 21 layers) and
    /// 128 (HCA, 20 layers). Its ratio-0 layers are exactly `{0, 1, 43}` —
    /// layers 0 and 1 plus the MTP slot; **layer 42 is CSA, not ratio 0**
    /// (`deepseek4.rs:377-382`, pinned by `yarn_applies_to_compressed_layers_only`).
    /// Ratio-0 layers are window-only and are modelled as
    /// [`RegionKind::Window`] with no compressed segment at all — never as
    /// `Compressed { ratio: 0 }`.
    pub fn compressed(ratio: u32) -> Option<Self> {
        if ratio < 2 {
            return None;
        }
        Some(RegionKind::Compressed { ratio })
    }

    /// KV cache group id for prefix-cache namespacing. See the type docs for
    /// why raw regions share group 0.
    pub fn group_id(self) -> u32 {
        match self {
            RegionKind::Dense | RegionKind::Window { .. } | RegionKind::Sink { .. } => 0,
            RegionKind::Compressed { ratio } => ratio,
        }
    }

    /// Whether this region stores raw (uncompressed) K/V.
    pub fn is_raw(self) -> bool {
        self.group_id() == 0
    }
}

/// One contiguous run of blocks a sequence reads from.
///
/// Invariants, checked by [`Segment::check`]:
/// * `first_token` is a multiple of the table's block size — a run must begin
///   on a block boundary because the gather kernel derives its intra-block
///   slot as `(token_id - cu[row]) % block_size`.
/// * `lead_pad < block_size` — it exists only to name the tokens the
///   block-alignment forced us to gather and the consumer must mask.
/// * `block_ids.len() == ceil(num_tokens / block_size)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub kind: RegionKind,
    /// Physical block ids, in region order.
    pub block_ids: Vec<usize>,
    /// Region-coordinate index of the first **gathered** token. Block-aligned.
    pub first_token: usize,
    /// Gathered tokens, counting from `first_token`. Includes `lead_pad`.
    pub num_tokens: usize,
    /// Gathered-but-dead tokens at the head of the run, produced by rounding
    /// the region's true start down to a block boundary. The consumer masks
    /// them. This mirrors what `inputs_processor.rs` already does by hand for
    /// sliding-window models (`block_aligned_start`).
    pub lead_pad: usize,
    /// Byte layout of **this** segment's blocks.
    ///
    /// Deliberately per-segment: a compressed segment and a raw segment in the
    /// same sequence hold different element counts per token, and the FP8 K
    /// store changes the width of some blocks and not others. Nothing in this
    /// module ever assumes a global KV element width — that is the rule
    /// [`crate::kv_sharing::layout`] already sets.
    pub layout: KvBlockLayout,
    /// Blocks at the head of `block_ids` already registered in the prefix
    /// cache. Skipped when caching newly-full blocks.
    pub num_cached_blocks: usize,
}

impl Segment {
    /// Live (unmasked) tokens in this run.
    pub fn live_tokens(&self) -> usize {
        self.num_tokens.saturating_sub(self.lead_pad)
    }

    /// Bytes of device KV this run occupies, at its own layout.
    pub fn bytes(&self, block_size: usize) -> u64 {
        self.layout
            .bytes_for_tokens(self.block_ids.len().saturating_mul(block_size))
    }

    /// Returns the first violated invariant, or `None`.
    pub fn check(&self, block_size: usize) -> Option<String> {
        if block_size == 0 {
            return Some("block_size is 0".to_string());
        }
        if !self.first_token.is_multiple_of(block_size) {
            return Some(format!(
                "segment {:?}: first_token {} is not block-aligned (block_size {})",
                self.kind, self.first_token, block_size
            ));
        }
        if self.lead_pad >= block_size && self.lead_pad != 0 {
            return Some(format!(
                "segment {:?}: lead_pad {} >= block_size {}",
                self.kind, self.lead_pad, block_size
            ));
        }
        let want = self.num_tokens.div_ceil(block_size);
        if self.block_ids.len() != want {
            return Some(format!(
                "segment {:?}: {} blocks for {} tokens, expected {}",
                self.kind,
                self.block_ids.len(),
                self.num_tokens,
                want
            ));
        }
        if self.num_cached_blocks > self.block_ids.len() {
            return Some(format!(
                "segment {:?}: num_cached_blocks {} exceeds {} blocks",
                self.kind,
                self.num_cached_blocks,
                self.block_ids.len()
            ));
        }
        None
    }
}

/// Everything one sequence reads, as an ordered list of runs.
///
/// A dense Llama-class model produces a table with exactly one `Dense`
/// segment; V4 produces two (window + compressed) on its CSA/HCA layers and
/// one (window) on its ratio-0 layers. The one-segment case is not a special
/// case bolted on — it is what this type degenerates to, and
/// [`SegmentTable::is_degenerate`] is the predicate every legacy caller is
/// allowed to rely on.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentTable {
    block_size: usize,
    segments: Vec<Segment>,
}

impl SegmentTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            segments: Vec::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub fn segments_mut(&mut self) -> &mut [Segment] {
        &mut self.segments
    }

    pub fn segment(&self, index: usize) -> Option<&Segment> {
        self.segments.get(index)
    }

    pub fn segment_mut(&mut self, index: usize) -> Option<&mut Segment> {
        self.segments.get_mut(index)
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn push(&mut self, segment: Segment) {
        self.segments.push(segment);
    }

    /// True when this table is the classic one-run contract: exactly one
    /// `Dense` segment starting at token 0 with nothing masked.
    ///
    /// Callers still on `(block_table, context_len)` are correct **iff** this
    /// holds; [`super::kv_cache_manager::KVCacheManager::get_block_ids`]
    /// refuses rather than lie when it does not.
    pub fn is_degenerate(&self) -> bool {
        self.segments.len() == 1
            && self.segments[0].kind == RegionKind::Dense
            && self.segments[0].first_token == 0
            && self.segments[0].lead_pad == 0
    }

    /// Every block this sequence holds, across all segments, in region order.
    /// Used for freeing; **not** a block table (a multi-segment concatenation
    /// is not readable as one run).
    pub fn all_block_ids(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.total_blocks());
        for s in &self.segments {
            out.extend_from_slice(&s.block_ids);
        }
        out
    }

    pub fn total_blocks(&self) -> usize {
        self.segments.iter().map(|s| s.block_ids.len()).sum()
    }

    /// Device bytes held, summed at each segment's own layout.
    pub fn bytes(&self) -> u64 {
        self.segments
            .iter()
            .map(|s| s.bytes(self.block_size))
            .fold(0u64, u64::saturating_add)
    }

    /// Returns the first violated invariant across all segments, or `None`.
    pub fn check(&self) -> Option<String> {
        self.segments.iter().find_map(|s| s.check(self.block_size))
    }

    /// Advance the *tail* of segment `index` by `extra` tokens, reporting how
    /// many new blocks are required. Does not allocate; the allocator does.
    pub fn blocks_needed_for_advance(&self, index: usize, extra: usize) -> usize {
        let Some(seg) = self.segments.get(index) else {
            return 0;
        };
        let want = (seg.num_tokens + extra).div_ceil(self.block_size);
        want.saturating_sub(seg.block_ids.len())
    }

    /// Slide segment `index` so it retains at most `keep` live tokens,
    /// returning the block ids that fell out of the window and must be freed.
    ///
    /// This is the operation the one-run contract cannot express: releasing
    /// the *head* of a sequence's allocation while it keeps generating. Under
    /// `(block_table, context_len)` a sliding-window model still pays for
    /// every block it ever wrote, because the single run must stay anchored at
    /// token 0 for the block table indexing to work.
    ///
    /// Blocks are only released once the entire block is outside the window —
    /// a partially-live block stays, and its dead prefix becomes `lead_pad`.
    pub fn slide(&mut self, index: usize, keep: usize) -> Vec<usize> {
        let bs = self.block_size;
        let Some(seg) = self.segments.get_mut(index) else {
            return Vec::new();
        };
        let live_end = seg.first_token + seg.num_tokens;
        let live_start = seg.first_token + seg.lead_pad;
        let live = live_end.saturating_sub(live_start);
        if live <= keep {
            return Vec::new();
        }
        let new_live_start = live_end - keep;
        let new_first = (new_live_start / bs) * bs;
        let drop_blocks = (new_first - seg.first_token) / bs;
        if drop_blocks == 0 {
            // The window shrank but not past a block boundary: nothing to
            // free, only more of the head becomes masked.
            seg.lead_pad = new_live_start - seg.first_token;
            return Vec::new();
        }
        let freed: Vec<usize> = seg.block_ids.drain(..drop_blocks).collect();
        seg.first_token = new_first;
        seg.num_tokens -= drop_blocks * bs;
        seg.lead_pad = new_live_start - new_first;
        seg.num_cached_blocks = seg.num_cached_blocks.saturating_sub(drop_blocks);
        freed
    }
}

/// Which sequence and which of its segments a flattened row came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowOwner {
    pub request_id: usize,
    pub segment_index: usize,
    pub kind: RegionKind,
    /// Rows at the head of this row's gathered output that the consumer must
    /// mask. Carried through so the attention never has to re-derive it.
    pub lead_pad: usize,
}

/// A flattened, kernel-ready read plan for a whole batch.
///
/// `block_table` is `rows * stride` `i32`s and `cu_seqlens` is `rows + 1`
/// `i32`s — **exactly** the two arguments `gather_kv_cache` already takes. For
/// a batch of degenerate tables this is bit-identical to what
/// `KVCacheManager::get_block_table` + `build_cu_seqlens_kv_from_context_lens`
/// produce today, which is the no-regression property in its strongest form:
/// not "close enough", *the same bytes*.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentPlan {
    pub block_table: Vec<i32>,
    pub stride: usize,
    pub cu_seqlens: Vec<i32>,
    pub rows: Vec<RowOwner>,
}

impl SegmentPlan {
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn total_tokens(&self) -> usize {
        self.cu_seqlens.last().copied().unwrap_or(0).max(0) as usize
    }

    /// Half-open output range `[start, end)` of row `row` in the gathered
    /// pack.
    pub fn row_range(&self, row: usize) -> Option<(usize, usize)> {
        let a = *self.cu_seqlens.get(row)?;
        let b = *self.cu_seqlens.get(row + 1)?;
        Some((a.max(0) as usize, b.max(0) as usize))
    }

    /// Row indices belonging to `request_id`, in segment order.
    pub fn rows_for(&self, request_id: usize) -> Vec<usize> {
        self.rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.request_id == request_id)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Flatten a batch of segment tables into one kernel-ready plan.
///
/// `stride` is the block-table row width. Pass `None` to use the widest row
/// present (the minimum legal value); pass `Some` to match a
/// `max_blocks_per_seq` a CUDA-graph capture already froze.
///
/// Rows are emitted sequence-major, segment-minor, so a degenerate batch
/// emits one row per sequence in input order.
pub fn flatten(tables: &[(usize, &SegmentTable)], stride: Option<usize>) -> SegmentPlan {
    let widest = tables
        .iter()
        .flat_map(|(_, t)| t.segments().iter())
        .map(|s| s.block_ids.len())
        .max()
        .unwrap_or(0);
    let stride = stride.unwrap_or(widest).max(widest);

    let num_rows: usize = tables.iter().map(|(_, t)| t.len()).sum();
    let mut block_table = Vec::with_capacity(num_rows * stride);
    let mut cu_seqlens = Vec::with_capacity(num_rows + 1);
    let mut rows = Vec::with_capacity(num_rows);
    cu_seqlens.push(0i32);
    let mut acc: i64 = 0;

    for (request_id, table) in tables {
        for (segment_index, seg) in table.segments().iter().enumerate() {
            let base = block_table.len();
            #[allow(clippy::cast_possible_truncation)]
            block_table.extend(seg.block_ids.iter().map(|&b| b as i32));
            // Pad to stride. Zero matches what `get_block_table` pads with
            // today; padded entries are never addressed because `cu_seqlens`
            // stops the row first.
            block_table.resize(base + stride, 0);
            acc += seg.num_tokens as i64;
            #[allow(clippy::cast_possible_truncation)]
            cu_seqlens.push(acc as i32);
            rows.push(RowOwner {
                request_id: *request_id,
                segment_index,
                kind: seg.kind,
                lead_pad: seg.lead_pad,
            });
        }
    }

    SegmentPlan {
        block_table,
        stride,
        cu_seqlens,
        rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_sharing::{KvBlockLayout, KvElemType};

    fn raw_layout() -> KvBlockLayout {
        KvBlockLayout::uniform(KvElemType::Bf16, 512, 61)
    }

    fn dense_seg(block_ids: Vec<usize>, num_tokens: usize) -> Segment {
        Segment {
            kind: RegionKind::Dense,
            block_ids,
            first_token: 0,
            num_tokens,
            lead_pad: 0,
            layout: raw_layout(),
            num_cached_blocks: 0,
        }
    }

    fn table_of(block_size: usize, segs: Vec<Segment>) -> SegmentTable {
        let mut t = SegmentTable::new(block_size);
        for s in segs {
            t.push(s);
        }
        t
    }

    /// The whole design rests on raw regions being able to share prefix-cache
    /// blocks while compressed regions at different ratios never can. If this
    /// collapses, either sharing is lost or V4 reads a window block as a
    /// compressed one.
    #[test]
    fn group_ids_separate_compressed_ratios_and_unify_raw_regions() {
        assert_eq!(RegionKind::Dense.group_id(), 0);
        assert_eq!(RegionKind::Window { size: 512 }.group_id(), 0);
        assert_eq!(RegionKind::Sink { tokens: 4 }.group_id(), 0);
        // V4's two ratios must not alias each other or the raw group.
        let csa = RegionKind::compressed(4).unwrap();
        let hca = RegionKind::compressed(128).unwrap();
        assert_eq!(csa.group_id(), 4);
        assert_eq!(hca.group_id(), 128);
        assert_ne!(csa.group_id(), hca.group_id());
        assert_ne!(csa.group_id(), RegionKind::Dense.group_id());
        assert!(!csa.is_raw() && RegionKind::Dense.is_raw());
        // Ratio 0/1 are not compression and must be refused, or they would
        // collide with the raw group.
        assert!(RegionKind::compressed(0).is_none());
        assert!(RegionKind::compressed(1).is_none());
    }

    /// The no-regression property in its strongest form: a batch of
    /// one-`Dense`-segment tables flattens to exactly the bytes the legacy
    /// `(block_table, context_len)` path builds. Mirrors
    /// `KVCacheManager::get_block_table` (pad with 0 to `max_blocks`) and
    /// `build_cu_seqlens_kv_from_context_lens` (prepend 0, cumsum).
    #[test]
    fn degenerate_batch_flattens_to_the_legacy_block_table_and_cu_seqlens() {
        let bs = 4;
        let a = table_of(bs, vec![dense_seg(vec![7, 8, 9], 10)]);
        let b = table_of(bs, vec![dense_seg(vec![3, 4], 5)]);
        assert!(a.is_degenerate() && b.is_degenerate());

        let plan = flatten(&[(1, &a), (2, &b)], Some(4));

        // Legacy `get_block_table(req, 4)`: ids then zero-pad to 4.
        assert_eq!(plan.stride, 4);
        assert_eq!(plan.block_table, vec![7, 8, 9, 0, 3, 4, 0, 0]);
        // Legacy cu_seqlens from context_lens [10, 5].
        assert_eq!(plan.cu_seqlens, vec![0, 10, 15]);
        assert_eq!(plan.num_rows(), 2);
        assert_eq!(plan.row_range(0), Some((0, 10)));
        assert_eq!(plan.row_range(1), Some((10, 15)));
        assert_eq!(plan.rows[0].request_id, 1);
        assert_eq!(plan.rows[0].lead_pad, 0);
    }

    /// Two segments per sequence become two rows per sequence, and the second
    /// sequence's rows do not shift the first's — the property that lets one
    /// gather serve a union of regions.
    #[test]
    fn two_segments_per_sequence_flatten_to_two_independent_rows() {
        let bs = 4;
        let window = Segment {
            kind: RegionKind::Window { size: 6 },
            block_ids: vec![20, 21],
            first_token: 8,
            num_tokens: 8,
            lead_pad: 2,
            layout: raw_layout(),
            num_cached_blocks: 0,
        };
        let compressed = Segment {
            kind: RegionKind::compressed(4).unwrap(),
            block_ids: vec![30],
            first_token: 0,
            num_tokens: 2,
            lead_pad: 0,
            layout: KvBlockLayout::uniform(KvElemType::Bf16, 128, 61),
            num_cached_blocks: 0,
        };
        let t = table_of(bs, vec![window, compressed]);
        assert!(!t.is_degenerate());
        assert!(t.check().is_none(), "{:?}", t.check());

        let plan = flatten(&[(9, &t)], None);
        assert_eq!(plan.num_rows(), 2);
        assert_eq!(plan.stride, 2);
        assert_eq!(plan.block_table, vec![20, 21, 30, 0]);
        assert_eq!(plan.cu_seqlens, vec![0, 8, 10]);
        // The window row carries its own mask width; the compressed row does
        // not inherit it.
        assert_eq!(plan.rows[0].lead_pad, 2);
        assert_eq!(plan.rows[1].lead_pad, 0);
        assert_eq!(plan.rows_for(9), vec![0, 1]);
        // The gather kernel derives `slot = (token_id - cu[row]) % block_size`,
        // so each row must begin at a block boundary in its own frame. Assert
        // the arithmetic the kernel will do lands on the block we intended.
        let (start, _end) = plan.row_range(0).unwrap();
        let token_id = start + 5; // 6th gathered token of the window row
        let offset = token_id - start;
        assert_eq!(plan.block_table[offset / bs], 21, "row 0 starts at index 0");
        assert_eq!(offset % bs, 1);
    }

    /// Sliding frees whole blocks and never a partially-live one, and the
    /// dead head is named as `lead_pad` rather than silently read.
    #[test]
    fn sliding_a_window_frees_only_fully_dead_blocks() {
        let bs = 4;
        let mut t = table_of(
            bs,
            vec![Segment {
                kind: RegionKind::Window { size: 6 },
                block_ids: vec![10, 11, 12, 13, 14],
                first_token: 0,
                num_tokens: 20,
                lead_pad: 0,
                layout: raw_layout(),
                num_cached_blocks: 5,
            }],
        );
        // Keep the last 6 of 20 live tokens: live starts at 14, block-aligned
        // down to 12, so blocks covering [0,12) — three of them — are dead.
        let freed = t.slide(0, 6);
        assert_eq!(freed, vec![10, 11, 12]);
        let seg = t.segment(0).unwrap();
        assert_eq!(seg.first_token, 12);
        assert_eq!(seg.num_tokens, 8);
        assert_eq!(seg.lead_pad, 2, "tokens 12,13 are gathered but dead");
        assert_eq!(seg.live_tokens(), 6);
        assert_eq!(seg.block_ids, vec![13, 14]);
        assert_eq!(seg.num_cached_blocks, 2);
        assert!(t.check().is_none(), "{:?}", t.check());

        // A slide that does not cross a block boundary frees nothing but
        // still widens the mask — it must not claim a block back.
        let freed = t.slide(0, 5);
        assert!(freed.is_empty());
        assert_eq!(t.segment(0).unwrap().lead_pad, 3);
        assert_eq!(t.segment(0).unwrap().live_tokens(), 5);

        // Sliding to a window wider than what is live is a no-op, not a
        // negative-length run.
        let freed = t.slide(0, 999);
        assert!(freed.is_empty());
        assert_eq!(t.segment(0).unwrap().live_tokens(), 5);
    }

    /// Byte accounting is per segment. A compressed segment at 1/4 the
    /// elements must not be charged the raw width, or capacity planning for
    /// V4 is wrong by the compression ratio.
    #[test]
    fn bytes_are_accounted_per_segment_not_at_one_global_width() {
        let bs = 4;
        let raw = Segment {
            kind: RegionKind::Window { size: 8 },
            block_ids: vec![1, 2],
            first_token: 0,
            num_tokens: 8,
            lead_pad: 0,
            layout: KvBlockLayout::uniform(KvElemType::Bf16, 512, 2),
            num_cached_blocks: 0,
        };
        let comp = Segment {
            kind: RegionKind::compressed(4).unwrap(),
            block_ids: vec![3],
            first_token: 0,
            num_tokens: 4,
            lead_pad: 0,
            layout: KvBlockLayout::uniform(KvElemType::Fp8E4M3, 512, 2),
            num_cached_blocks: 0,
        };
        let raw_bytes = raw.bytes(bs);
        let comp_bytes = comp.bytes(bs);
        // 8 tokens bf16 vs 4 tokens fp8 at the same element count: 4x.
        assert_eq!(raw_bytes, 8 * 512 * 2 * 2 * 2);
        assert_eq!(comp_bytes, 4 * 512 * 2 * 2);
        assert_eq!(raw_bytes, comp_bytes * 4);
        let t = table_of(bs, vec![raw, comp]);
        assert_eq!(t.bytes(), raw_bytes + comp_bytes);
        // A single global width would have made these equal per token.
        assert_ne!(t.bytes(), raw_bytes * 2);
    }

    /// The invariants are load-bearing for the kernel's index arithmetic, so
    /// each one must actually be detected.
    #[test]
    fn broken_invariants_are_named_not_silently_accepted() {
        let bs = 4;
        let mut s = dense_seg(vec![1, 2], 8);
        assert!(s.check(bs).is_none());

        s.first_token = 2;
        assert!(s.check(bs).unwrap().contains("block-aligned"));
        s.first_token = 0;

        s.num_tokens = 9; // needs 3 blocks, has 2
        assert!(s.check(bs).unwrap().contains("blocks for"));
        s.num_tokens = 8;

        s.lead_pad = 4; // == block_size: a whole dead block should have been freed
        assert!(s.check(bs).unwrap().contains("lead_pad"));
        s.lead_pad = 0;

        s.num_cached_blocks = 3;
        assert!(s.check(bs).unwrap().contains("num_cached_blocks"));
    }

    #[test]
    fn advance_reports_exactly_the_blocks_the_tail_needs() {
        let bs = 4;
        let t = table_of(bs, vec![dense_seg(vec![1, 2], 8)]);
        assert_eq!(t.blocks_needed_for_advance(0, 0), 0);
        assert_eq!(t.blocks_needed_for_advance(0, 1), 1);
        assert_eq!(t.blocks_needed_for_advance(0, 4), 1);
        assert_eq!(t.blocks_needed_for_advance(0, 5), 2);
        // An out-of-range segment asks for nothing rather than panicking in
        // an allocator hot path.
        assert_eq!(t.blocks_needed_for_advance(7, 5), 0);
    }
}
