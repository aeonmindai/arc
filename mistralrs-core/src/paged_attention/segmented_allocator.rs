//! Block allocation over [`SegmentTable`]s.
//!
//! This is [`super::kv_cache_manager::KVCacheManager`]'s per-request tracking,
//! generalised from "one `Vec<usize>` of blocks per request" to "one ordered
//! list of runs per request". It owns no memory: every method takes the
//! caller's [`BlockPool`], so one pool still backs every region of every
//! sequence and the free list stays global.
//!
//! # The two properties that make this the keystone
//!
//! **1. Per-sequence, per-region independent advance.** [`Self::advance`] and
//! [`Self::rollback`] name a `(request_id, segment_index)` pair. No operation
//! reads another request's table, so no request's progress can be pinned by
//! another's. That is the allocator-level precondition for lifting the MTP
//! batch ceiling recorded in `memory/mission/wave59-CJ-mtp-batch.md` §3, where
//! one dense shared cache forces `next_u = u + a + 1 - min_i(u_i + a_i)` and a
//! single sequence rejecting its first draft ratchets every other sequence's
//! uncached tail to the window. **It is a precondition, not the fix** — the
//! fused MTP step still has to stop rolling one shared cache. See
//! `pipeline/mtp_pipeline.rs`.
//!
//! **2. Head release.** [`Self::slide`] returns the blocks a sliding window
//! has left behind, so they go back on the free list while the sequence is
//! still running. The one-run contract cannot do this: its block table has to
//! stay anchored at token 0 for `block_table[offset / block_size]` to address
//! correctly, so every block ever written stays charged to the request.

use std::collections::HashMap;

use crate::kv_sharing::KvBlockLayout;

use super::block_hash::{BlockHash, BlockHashWithGroupId};
use super::block_pool::BlockPool;
use super::segment::{RegionKind, Segment, SegmentPlan, SegmentTable};

/// Per-request segment tables over a caller-owned [`BlockPool`].
pub struct SegmentedAllocator {
    block_size: usize,
    tables: HashMap<usize, SegmentTable>,
}

// `total_blocks`, `open_region`, `advance`, `slide` and `region_key` are the
// **multi-region** half of this API. Nothing in the shipped pipeline calls
// them yet, and that is the honest state of this change: `ARC_SEGMENTED_KV=1`
// today drives only the degenerate one-`Dense`-region path through
// `allocate_dense` / `advance_to` / `rollback` / `free`, which is what makes
// the no-regression claim measurable before any model depends on the rest.
// The multi-region entry points are exercised by this module's tests and are
// the surface the V4 and MTP follow-ons attach to. They are *not* silently
// live: no production call site reaches them, by construction.
#[allow(dead_code)]
impl SegmentedAllocator {
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be non-zero");
        Self {
            block_size,
            tables: HashMap::new(),
        }
    }

    pub fn table(&self, request_id: usize) -> Option<&SegmentTable> {
        self.tables.get(&request_id)
    }

    pub fn has_request(&self, request_id: usize) -> bool {
        self.tables.contains_key(&request_id)
    }

    /// Total blocks held across every request and region.
    pub fn total_blocks(&self) -> usize {
        self.tables.values().map(SegmentTable::total_blocks).sum()
    }

    /// Device bytes held, each segment charged at its own layout. Never
    /// assumes one global KV element width.
    pub fn bytes(&self) -> u64 {
        self.tables
            .values()
            .map(SegmentTable::bytes)
            .fold(0u64, u64::saturating_add)
    }

    /// Open (or extend) a request's single `Dense` region — the degenerate
    /// path that must behave exactly like today's `allocate_slots`.
    ///
    /// `computed_blocks` are prefix-cache hits from
    /// [`super::kv_cache_manager::KVCacheManager::get_computed_blocks`]; they
    /// are touched, not allocated.
    ///
    /// Returns the newly allocated block ids, or `None` if the pool cannot
    /// satisfy the request (in which case nothing is allocated or touched).
    pub fn allocate_dense(
        &mut self,
        pool: &mut BlockPool,
        request_id: usize,
        num_tokens: usize,
        computed_blocks: &[usize],
        layout: KvBlockLayout,
    ) -> Option<Vec<usize>> {
        if self.tables.contains_key(&request_id) {
            return self.advance_to(pool, request_id, 0, num_tokens);
        }

        let num_required = num_tokens.div_ceil(self.block_size);
        let num_computed = computed_blocks.len().min(num_required);
        let num_new = num_required - num_computed;

        // Touching a computed block with ref_cnt 0 pulls it out of the free
        // list, so it counts against capacity exactly like a fresh block.
        let num_evictable = if pool.caching_enabled() {
            computed_blocks[..num_computed]
                .iter()
                .filter(|&&id| pool.block_ref_cnt(id) == 0)
                .count()
        } else {
            0
        };
        if num_new + num_evictable > pool.num_free_blocks() {
            return None;
        }

        if num_computed > 0 && pool.caching_enabled() {
            pool.touch(&computed_blocks[..num_computed]);
        }
        let new_blocks = if num_new > 0 {
            pool.get_new_blocks(num_new)?
        } else {
            Vec::new()
        };

        let mut block_ids = Vec::with_capacity(num_required);
        block_ids.extend_from_slice(&computed_blocks[..num_computed]);
        block_ids.extend_from_slice(&new_blocks);

        let mut table = SegmentTable::new(self.block_size);
        table.push(Segment {
            kind: RegionKind::Dense,
            block_ids,
            first_token: 0,
            num_tokens,
            lead_pad: 0,
            layout,
            num_cached_blocks: num_computed,
        });
        debug_assert!(table.check().is_none(), "{:?}", table.check());
        self.tables.insert(request_id, table);
        Some(new_blocks)
    }

    /// Append a new region to a request, sized for `num_tokens` **region**
    /// tokens (compressed rows for a compressed region).
    ///
    /// Returns the segment index and the newly allocated blocks. `None` if the
    /// pool cannot satisfy it; nothing is allocated in that case.
    pub fn open_region(
        &mut self,
        pool: &mut BlockPool,
        request_id: usize,
        kind: RegionKind,
        num_tokens: usize,
        layout: KvBlockLayout,
    ) -> Option<(usize, Vec<usize>)> {
        let needed = num_tokens.div_ceil(self.block_size);
        if needed > pool.num_free_blocks() {
            return None;
        }
        let block_ids = if needed > 0 {
            pool.get_new_blocks(needed)?
        } else {
            Vec::new()
        };
        let table = self
            .tables
            .entry(request_id)
            .or_insert_with(|| SegmentTable::new(self.block_size));
        let index = table.len();
        table.push(Segment {
            kind,
            block_ids: block_ids.clone(),
            first_token: 0,
            num_tokens,
            lead_pad: 0,
            layout,
            num_cached_blocks: 0,
        });
        debug_assert!(table.check().is_none(), "{:?}", table.check());
        Some((index, block_ids))
    }

    /// Grow segment `segment_index` of `request_id` by `extra` region tokens.
    ///
    /// This is the per-sequence advance. Nothing about it consults, or is
    /// bounded by, any other request.
    pub fn advance(
        &mut self,
        pool: &mut BlockPool,
        request_id: usize,
        segment_index: usize,
        extra: usize,
    ) -> Option<Vec<usize>> {
        let table = self.tables.get(&request_id)?;
        let seg = table.segment(segment_index)?;
        self.advance_to(pool, request_id, segment_index, seg.num_tokens + extra)
    }

    /// Grow segment `segment_index` so it covers `num_tokens` region tokens.
    /// A `num_tokens` at or below the current size is a no-op (use
    /// [`Self::rollback`] to shrink).
    pub fn advance_to(
        &mut self,
        pool: &mut BlockPool,
        request_id: usize,
        segment_index: usize,
        num_tokens: usize,
    ) -> Option<Vec<usize>> {
        let block_size = self.block_size;
        let table = self.tables.get_mut(&request_id)?;
        let seg = table.segment_mut(segment_index)?;
        if num_tokens <= seg.num_tokens {
            return Some(Vec::new());
        }
        let want = num_tokens.div_ceil(block_size);
        let num_new = want.saturating_sub(seg.block_ids.len());
        if num_new == 0 {
            seg.num_tokens = num_tokens;
            return Some(Vec::new());
        }
        // Reborrow after the pool call so the pool is not aliased.
        let new_blocks = pool.get_new_blocks(num_new)?;
        let table = self.tables.get_mut(&request_id)?;
        let seg = table.segment_mut(segment_index)?;
        seg.block_ids.extend_from_slice(&new_blocks);
        seg.num_tokens = num_tokens;
        debug_assert!(table.check().is_none(), "{:?}", table.check());
        Some(new_blocks)
    }

    /// Shrink segment `segment_index` to `num_tokens` region tokens, freeing
    /// whole tail blocks that are no longer covered.
    ///
    /// This is what a speculative verify does when it rejects drafts. Because
    /// it names one request, a rejection by one sequence cannot roll back
    /// another's cache.
    pub fn rollback(
        &mut self,
        pool: &mut BlockPool,
        request_id: usize,
        segment_index: usize,
        num_tokens: usize,
    ) -> usize {
        let block_size = self.block_size;
        let mut freed = {
            let Some(table) = self.tables.get_mut(&request_id) else {
                return 0;
            };
            let Some(seg) = table.segment_mut(segment_index) else {
                return 0;
            };
            if num_tokens >= seg.num_tokens {
                return 0;
            }
            let want = num_tokens.div_ceil(block_size);
            seg.num_tokens = num_tokens;
            if want >= seg.block_ids.len() {
                return 0;
            }
            let freed: Vec<usize> = seg.block_ids.drain(want..).collect();
            seg.num_cached_blocks = seg.num_cached_blocks.min(seg.block_ids.len());
            freed
        };
        // Reverse so the most specific (tail) block is evicted first, matching
        // `KVCacheManager::free`.
        freed.reverse();
        pool.free_blocks(&freed);
        freed.len()
    }

    /// Slide segment `segment_index` to retain at most `keep` live region
    /// tokens, returning the blocks released to the pool.
    ///
    /// Only blocks that are *entirely* outside the window are released; a
    /// partially live block stays and its dead prefix becomes the segment's
    /// `lead_pad`.
    pub fn slide(
        &mut self,
        pool: &mut BlockPool,
        request_id: usize,
        segment_index: usize,
        keep: usize,
    ) -> usize {
        let Some(table) = self.tables.get_mut(&request_id) else {
            return 0;
        };
        let mut freed = table.slide(segment_index, keep);
        debug_assert!(table.check().is_none(), "{:?}", table.check());
        if freed.is_empty() {
            return 0;
        }
        // Head blocks are the least specific: free them oldest-last so the
        // LRU order still evicts the oldest first.
        freed.reverse();
        pool.free_blocks(&freed);
        freed.len()
    }

    /// Release every block of a request, across every region.
    pub fn free(&mut self, pool: &mut BlockPool, request_id: usize) {
        let Some(table) = self.tables.remove(&request_id) else {
            return;
        };
        let mut all = table.all_block_ids();
        all.reverse();
        pool.free_blocks(&all);
    }

    /// Register newly-full blocks of one region in the prefix cache.
    ///
    /// The region's [`RegionKind::group_id`] is the KV cache group, so a
    /// compressed block can never be served to a raw read (or to a compressed
    /// read at a different ratio) — the whole reason `BlockHashWithGroupId`
    /// exists. Raw regions deliberately share group 0 and therefore *do* reuse
    /// each other's blocks.
    pub fn cache_region_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: usize,
        segment_index: usize,
        block_hashes: &[BlockHash],
        num_computed_tokens: usize,
    ) {
        if !pool.caching_enabled() {
            return;
        }
        let block_size = self.block_size;
        let Some(table) = self.tables.get_mut(&request_id) else {
            return;
        };
        let Some(seg) = table.segment_mut(segment_index) else {
            return;
        };
        let num_full = (num_computed_tokens / block_size).min(seg.block_ids.len());
        if seg.num_cached_blocks >= num_full || block_hashes.len() < num_full {
            return;
        }
        pool.cache_full_blocks(
            &seg.block_ids,
            block_hashes,
            seg.num_cached_blocks,
            num_full,
            seg.kind.group_id(),
        );
        seg.num_cached_blocks = num_full;
    }

    /// The prefix-cache key for one block of a region: the content hash
    /// namespaced by the region's group.
    pub fn region_key(kind: RegionKind, block_hash: BlockHash) -> BlockHashWithGroupId {
        BlockHashWithGroupId {
            block_hash,
            group_id: kind.group_id(),
        }
    }

    /// Physical slot indices for `n` region tokens starting at region index
    /// `start`, for one segment. `slot = block_id * block_size + offset`.
    pub fn slot_mapping(
        &self,
        request_id: usize,
        segment_index: usize,
        start: usize,
        n: usize,
    ) -> Option<Vec<i64>> {
        let seg = self.tables.get(&request_id)?.segment(segment_index)?;
        let mut slots = Vec::with_capacity(n);
        for pos in start..start + n {
            // Positions are region-absolute; the run begins at `first_token`.
            let rel = pos.checked_sub(seg.first_token);
            match rel {
                Some(rel) if rel / self.block_size < seg.block_ids.len() => {
                    let block_id = seg.block_ids[rel / self.block_size];
                    #[allow(clippy::cast_possible_wrap)]
                    slots.push((block_id * self.block_size + rel % self.block_size) as i64);
                }
                _ => slots.push(super::_PAD_SLOT_ID),
            }
        }
        Some(slots)
    }

    /// Flatten a batch into one kernel-ready plan. Requests with no table are
    /// skipped; the returned [`SegmentPlan`]'s `rows` say which request and
    /// region each row came from.
    pub fn plan(&self, request_ids: &[usize], stride: Option<usize>) -> SegmentPlan {
        let tables: Vec<(usize, &SegmentTable)> = request_ids
            .iter()
            .filter_map(|id| self.tables.get(id).map(|t| (*id, t)))
            .collect();
        super::segment::flatten(&tables, stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_sharing::{KvBlockLayout, KvElemType};
    use crate::paged_attention::block_hash::compute_block_hashes;
    use crate::paged_attention::kv_cache_manager::KVCacheManager;

    fn layout() -> KvBlockLayout {
        KvBlockLayout::uniform(KvElemType::Bf16, 512, 61)
    }

    fn comp_layout() -> KvBlockLayout {
        KvBlockLayout::uniform(KvElemType::Bf16, 512, 61)
    }

    /// The strongest form of "the dense path does not regress": drive the
    /// legacy `KVCacheManager` and the segmented allocator through the *same*
    /// trace against pools of the same size, and assert the block tables, slot
    /// mappings and free-block counts agree at every step — including after a
    /// prefix-cache hit, which is where the touch/evict accounting is easiest
    /// to get subtly wrong.
    #[test]
    fn degenerate_path_matches_the_legacy_manager_step_for_step() {
        let (n_blocks, bs) = (32, 4);
        let mut legacy = KVCacheManager::new(n_blocks, bs, true, vec![0]);
        let mut seg = SegmentedAllocator::new(bs);
        let mut pool = BlockPool::new(n_blocks, true, bs);

        let tokens: Vec<u32> = (1..=20).collect();
        let hashes = compute_block_hashes(&tokens, bs, &[], &[]);

        // 1. cold prompt of 20 tokens
        let l = legacy.allocate_slots(1, 20, &[]).unwrap();
        let s = seg.allocate_dense(&mut pool, 1, 20, &[], layout()).unwrap();
        assert_eq!(l.len(), s.len(), "same number of new blocks");
        assert_eq!(legacy.num_free_blocks(), pool.num_free_blocks());
        assert_eq!(
            legacy.get_block_table(1, 8).unwrap(),
            seg.plan(&[1], Some(8)).block_table,
            "degenerate plan must be the legacy block table, byte for byte"
        );
        assert_eq!(
            legacy.get_slot_mapping(1, 0, 20).unwrap(),
            seg.slot_mapping(1, 0, 0, 20).unwrap()
        );

        // 2. decode: grow to 24 tokens
        legacy.allocate_slots(1, 24, &[]).unwrap();
        seg.advance_to(&mut pool, 1, 0, 24).unwrap();
        assert_eq!(legacy.num_free_blocks(), pool.num_free_blocks());
        assert_eq!(
            legacy.get_block_table(1, 8).unwrap(),
            seg.plan(&[1], Some(8)).block_table
        );

        // 3. cache and free, then a second request re-uses the prefix
        legacy.cache_blocks(1, &hashes, 20);
        seg.cache_region_blocks(&mut pool, 1, 0, &hashes, 20);
        legacy.free(1);
        seg.free(&mut pool, 1);
        assert_eq!(legacy.num_free_blocks(), pool.num_free_blocks());

        let l_hit = legacy.get_computed_blocks(&hashes, 24);
        assert!(l_hit.num_computed_tokens > 0, "fixture must exercise a hit");
        let s_hit: Vec<usize> = (0..l_hit.block_ids.len())
            .map(|i| {
                pool.get_cached_block(hashes[i], &[RegionKind::Dense.group_id()])
                    .unwrap()[0]
            })
            .collect();
        assert_eq!(l_hit.block_ids, s_hit, "same cached blocks found");

        legacy.allocate_slots(2, 24, &l_hit.block_ids).unwrap();
        seg.allocate_dense(&mut pool, 2, 24, &s_hit, layout())
            .unwrap();
        assert_eq!(
            legacy.num_free_blocks(),
            pool.num_free_blocks(),
            "touch/evict accounting must match after a prefix hit"
        );
        assert_eq!(
            legacy.get_block_table(2, 8).unwrap(),
            seg.plan(&[2], Some(8)).block_table
        );
    }

    /// A batch of degenerate requests flattens to the concatenation of their
    /// legacy rows, in request order — so a caller can swap in the plan
    /// without reordering anything.
    #[test]
    fn degenerate_batch_plan_matches_concatenated_legacy_rows() {
        let (n_blocks, bs) = (64, 4);
        let mut legacy = KVCacheManager::new(n_blocks, bs, false, vec![0]);
        let mut seg = SegmentedAllocator::new(bs);
        let mut pool = BlockPool::new(n_blocks, false, bs);

        for (id, toks) in [(7usize, 9usize), (8, 16), (9, 3)] {
            legacy.allocate_slots(id, toks, &[]).unwrap();
            seg.allocate_dense(&mut pool, id, toks, &[], layout())
                .unwrap();
        }
        let ids = [7usize, 8, 9];
        let mut expected = Vec::new();
        for id in ids {
            expected.extend(legacy.get_block_table(id, 4).unwrap());
        }
        let plan = seg.plan(&ids, Some(4));
        assert_eq!(plan.block_table, expected);
        assert_eq!(plan.cu_seqlens, vec![0, 9, 25, 28]);
        assert_eq!(plan.num_rows(), 3);
    }

    /// The MTP precondition, in the shape of wave59-CJ's
    /// `one_laggard_ratchets_every_other_sequences_tail_to_the_window`: a
    /// sequence that rejects everything must not move any other sequence's
    /// cache length. Under one dense shared cache that is exactly what
    /// happens; under per-request segment tables it cannot.
    #[test]
    fn one_sequences_rollback_does_not_touch_another_sequences_advance() {
        let (n_blocks, bs) = (256, 4);
        let mut seg = SegmentedAllocator::new(bs);
        let mut pool = BlockPool::new(n_blocks, false, bs);

        let batch = [10usize, 11, 12, 13, 14];
        for id in batch {
            seg.allocate_dense(&mut pool, id, 16, &[], layout())
                .unwrap();
        }
        // Sequence 10 is the laggard: every step it drafts 4 and keeps 0.
        // Everyone else keeps all 4.
        for _step in 0..7 {
            for id in batch {
                seg.advance(&mut pool, id, 0, 4).unwrap();
            }
            let before = seg.table(10).unwrap().segment(0).unwrap().num_tokens;
            seg.rollback(&mut pool, 10, 0, before - 4);
        }
        let laggard = seg.table(10).unwrap().segment(0).unwrap().num_tokens;
        assert_eq!(laggard, 16, "the laggard never commits a token");
        for id in [11usize, 12, 13, 14] {
            assert_eq!(
                seg.table(id).unwrap().segment(0).unwrap().num_tokens,
                16 + 7 * 4,
                "seq {id} advanced independently of the laggard"
            );
        }
        // The whole point: the healthy sequences' advance is a function of
        // their own accepts, not of min_i over the batch.
        assert_ne!(
            laggard,
            seg.table(11).unwrap().segment(0).unwrap().num_tokens
        );
    }

    /// Head release. A long sequence on a 128-token window must converge to a
    /// constant block footprint instead of growing without bound — the
    /// capacity win the one-run contract cannot express.
    #[test]
    fn a_sliding_window_returns_its_head_blocks_to_the_pool() {
        let (n_blocks, bs, window) = (64, 4, 16usize);
        let mut seg = SegmentedAllocator::new(bs);
        let mut pool = BlockPool::new(n_blocks, false, bs);
        let free_at_start = pool.num_free_blocks();

        let (idx, _) = seg
            .open_region(
                &mut pool,
                1,
                RegionKind::Window { size: window },
                window,
                layout(),
            )
            .unwrap();

        let mut peak_blocks = 0usize;
        for _ in 0..40 {
            seg.advance(&mut pool, 1, idx, 4).unwrap();
            seg.slide(&mut pool, 1, idx, window);
            peak_blocks = peak_blocks.max(seg.total_blocks());
        }
        // Window of 16 tokens at block 4 => 4 live blocks, plus at most one
        // partially-dead head block.
        assert!(
            peak_blocks <= window / bs + 1,
            "window footprint grew to {peak_blocks} blocks"
        );
        let seg0 = seg.table(1).unwrap().segment(0).unwrap();
        assert!(seg0.live_tokens() <= window);
        assert!(seg0.first_token > 0, "the run's head actually moved");
        // Everything not in the window is back on the free list.
        assert_eq!(
            pool.num_free_blocks(),
            free_at_start - seg.total_blocks(),
            "released head blocks must be reusable, not leaked"
        );
        seg.free(&mut pool, 1);
        assert_eq!(pool.num_free_blocks(), free_at_start, "no block leaked");
    }

    /// A compressed region and a raw region holding the same token content
    /// must not alias in the prefix cache, or a window read is served
    /// compressor output. The two ratios must not alias each other either.
    #[test]
    fn regions_do_not_alias_in_the_prefix_cache() {
        let (n_blocks, bs) = (64, 4);
        let mut seg = SegmentedAllocator::new(bs);
        let mut pool = BlockPool::new(n_blocks, true, bs);

        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, bs, &[], &[]);

        seg.allocate_dense(&mut pool, 1, 8, &[], layout()).unwrap();
        let csa = RegionKind::compressed(4).unwrap();
        let hca = RegionKind::compressed(128).unwrap();
        let (csa_idx, _) = seg
            .open_region(&mut pool, 1, csa, 8, comp_layout())
            .unwrap();
        let (hca_idx, _) = seg
            .open_region(&mut pool, 1, hca, 8, comp_layout())
            .unwrap();

        seg.cache_region_blocks(&mut pool, 1, 0, &hashes, 8);
        seg.cache_region_blocks(&mut pool, 1, csa_idx, &hashes, 8);
        seg.cache_region_blocks(&mut pool, 1, hca_idx, &hashes, 8);

        let raw = pool.get_cached_block(hashes[0], &[0]).unwrap();
        let c4 = pool.get_cached_block(hashes[0], &[4]).unwrap();
        let c128 = pool.get_cached_block(hashes[0], &[128]).unwrap();
        assert_ne!(raw[0], c4[0], "raw and CSA blocks must be distinct");
        assert_ne!(c4[0], c128[0], "CSA and HCA blocks must be distinct");
        assert_ne!(raw[0], c128[0]);
        // A group with no cached block must miss rather than fall back to
        // another group's data.
        assert!(pool.get_cached_block(hashes[0], &[7]).is_none());
        assert_eq!(SegmentedAllocator::region_key(csa, hashes[0]).group_id, 4);
    }

    /// Failing to allocate must leave the pool exactly as it was, or a
    /// scheduler that retries slowly leaks the whole cache.
    #[test]
    fn a_refused_allocation_is_a_no_op_on_the_pool() {
        let (n_blocks, bs) = (6, 4); // 5 usable blocks
        let mut seg = SegmentedAllocator::new(bs);
        let mut pool = BlockPool::new(n_blocks, false, bs);
        let free = pool.num_free_blocks();

        assert!(seg
            .allocate_dense(&mut pool, 1, 400, &[], layout())
            .is_none());
        assert_eq!(pool.num_free_blocks(), free);
        assert!(!seg.has_request(1));

        assert!(seg
            .open_region(&mut pool, 1, RegionKind::Window { size: 8 }, 400, layout())
            .is_none());
        assert_eq!(pool.num_free_blocks(), free);
    }

    /// Slot mappings must respect a slid run's `first_token`: after the head
    /// moves, region position `p` lives at `(p - first_token)` into the run,
    /// not at `p`.
    #[test]
    fn slot_mapping_follows_a_slid_runs_origin() {
        let (n_blocks, bs) = (64, 4);
        let mut seg = SegmentedAllocator::new(bs);
        let mut pool = BlockPool::new(n_blocks, false, bs);
        let (idx, _) = seg
            .open_region(&mut pool, 1, RegionKind::Window { size: 8 }, 20, layout())
            .unwrap();
        seg.slide(&mut pool, 1, idx, 8);
        let s = seg.table(1).unwrap().segment(idx).unwrap();
        assert_eq!(s.first_token, 12);
        let block0 = s.block_ids[0];
        let slots = seg.slot_mapping(1, idx, 12, 4).unwrap();
        assert_eq!(
            slots,
            (0..4).map(|i| (block0 * bs + i) as i64).collect::<Vec<_>>()
        );
        // A position that fell out of the window has no slot and must be
        // named as padding, not silently mapped onto block 0.
        assert_eq!(
            seg.slot_mapping(1, idx, 0, 1).unwrap(),
            vec![super::super::_PAD_SLOT_ID]
        );
    }

    /// Bytes are charged per segment. A V4-shaped table (raw window + HCA
    /// compressed) must not be billed as if the compressed region had the raw
    /// width, or the capacity planner is wrong by the compression ratio.
    #[test]
    fn a_v4_shaped_table_is_billed_per_region() {
        let (n_blocks, bs) = (64, 4);
        let mut seg = SegmentedAllocator::new(bs);
        let mut pool = BlockPool::new(n_blocks, false, bs);
        let raw = KvBlockLayout::uniform(KvElemType::Bf16, 512, 1);
        let comp = KvBlockLayout::uniform(KvElemType::Fp8E4M3, 512, 1);

        seg.open_region(&mut pool, 1, RegionKind::Window { size: 8 }, 8, raw)
            .unwrap();
        seg.open_region(&mut pool, 1, RegionKind::compressed(128).unwrap(), 8, comp)
            .unwrap();

        let total = seg.bytes();
        let uniform = raw.bytes_for_tokens(16);
        assert!(
            total < uniform,
            "{total} should be under the all-raw {uniform}"
        );
        assert_eq!(total, raw.bytes_for_tokens(8) + comp.bytes_for_tokens(8));
    }
}
