//! KV Cache Manager for paged attention.
//!
//! This is a port of vLLM's v1 `KVCacheManager` + `FullAttentionManager`.
//! It manages block allocation, prefix cache lookups, and per-request block tracking.
//!
//! The manager owns a `BlockPool` and provides high-level operations:
//! - `get_computed_blocks`: Find the longest prefix cache hit for a request.
//! - `allocate_slots`: Allocate blocks for new tokens.
//! - `free`: Free blocks when a request completes or is preempted.
//! - `cache_blocks`: Cache newly-full blocks after computation.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::kv_sharing::KvBlockLayout;

use super::block_hash::BlockHash;
use super::block_pool::BlockPool;
use super::segment::{SegmentPlan, SegmentTable};
use super::segmented_allocator::SegmentedAllocator;

/// `ARC_SEGMENTED_KV=1` routes per-request block tracking through
/// [`SegmentedAllocator`] instead of the flat `Vec<usize>` per request.
///
/// **Default OFF.** With it off, not one line of the legacy path changes: the
/// segmented allocator is never constructed and `req_to_blocks` is the same
/// `HashMap` it has always been.
///
/// With it on, a dense model runs on the *degenerate* one-`Dense`-segment
/// table, which flattens to byte-identical block tables and slot mappings
/// (pinned by `degenerate_path_matches_the_legacy_manager_step_for_step` in
/// [`super::segmented_allocator`]). That is deliberate: it makes the new
/// allocator measurable on real hardware with a model that already works,
/// before any multi-region model depends on it.
static SEGMENTED_KV: LazyLock<bool> = LazyLock::new(|| {
    matches!(
        std::env::var("ARC_SEGMENTED_KV").as_deref(),
        Ok("1") | Ok("true")
    )
});

/// Whether the segmented allocator is enabled for this process.
pub fn segmented_kv_enabled() -> bool {
    *SEGMENTED_KV
}

/// Per-request block tracking. Two shapes, one pool.
enum Tracking {
    /// One contiguous run per request — the shipped path.
    Flat(HashMap<usize, RequestBlocks>),
    /// An ordered list of runs per request.
    Segmented(Box<SegmentedAllocator>),
}

/// Result of `get_computed_blocks`: cached block IDs and how many tokens they cover.
#[derive(Debug)]
pub struct ComputedBlocks {
    /// Block IDs from the prefix cache, one per block position.
    /// For models with sliding window layers, early blocks may be `null_block_id`
    /// (placeholder for blocks outside the attention window).
    pub block_ids: Vec<usize>,
    /// Number of tokens covered by the cached blocks.
    /// Always a multiple of `block_size`.
    pub num_computed_tokens: usize,
}

/// Per-request block allocation state.
struct RequestBlocks {
    /// Block IDs allocated for this request, in sequence order.
    block_ids: Vec<usize>,
    /// Number of blocks that are already cached (skip during `cache_blocks`).
    num_cached_blocks: usize,
}

/// KV Cache Manager — manages block allocation and prefix caching.
///
/// Each instance handles one "type" of KV cache layer (e.g., full attention).
/// For models with alternating sliding window layers (Gemma2, GPT-OSS),
/// separate instances manage the full-attention and sliding-window block tables,
/// sharing the same underlying `BlockPool`.
pub struct KVCacheManager {
    block_pool: BlockPool,
    block_size: usize,
    enable_caching: bool,
    /// KV cache group IDs used for prefix cache lookups.
    /// Most models have a single group `[0]`. Models with multiple attention
    /// types (e.g., full + sliding window) use different group IDs per manager.
    kv_cache_group_ids: Vec<u32>,
    /// Per-request block tracking.
    tracking: Tracking,
    /// Byte layout charged to a request's blocks in segmented mode.
    ///
    /// Neutral by default — this type is not told the model's geometry at
    /// construction, and inventing one would make the byte report wrong in a
    /// way nobody could see. Callers that know the geometry set it with
    /// [`KVCacheManager::set_kv_layout`].
    kv_layout: KvBlockLayout,
}

impl KVCacheManager {
    /// Create a new KV cache manager.
    ///
    /// - `num_gpu_blocks`: Total number of physical GPU blocks.
    /// - `block_size`: Tokens per block.
    /// - `enable_caching`: Whether prefix caching is enabled.
    /// - `kv_cache_group_ids`: Group IDs for prefix cache lookups.
    pub fn new(
        num_gpu_blocks: usize,
        block_size: usize,
        enable_caching: bool,
        kv_cache_group_ids: Vec<u32>,
    ) -> Self {
        Self::with_segmentation(
            num_gpu_blocks,
            block_size,
            enable_caching,
            kv_cache_group_ids,
            segmented_kv_enabled(),
        )
    }

    /// As [`KVCacheManager::new`], with the segmented allocator forced on or
    /// off instead of read from `ARC_SEGMENTED_KV`. Tests use this so they do
    /// not depend on process environment.
    pub fn with_segmentation(
        num_gpu_blocks: usize,
        block_size: usize,
        enable_caching: bool,
        kv_cache_group_ids: Vec<u32>,
        segmented: bool,
    ) -> Self {
        // Announce the backing, once per manager. This is the only way an
        // operator (or a validation script) can tell the segmented path
        // actually engaged: without it, a binary built *without* this change
        // silently ignores `ARC_SEGMENTED_KV`, runs the control twice, and
        // reports a perfect 1.00x ratio with identical output — a fake pass
        // that looks exactly like a real one. Assert on this line, not on the
        // environment variable.
        if segmented {
            tracing::info!(
                "KV cache backing: SEGMENTED allocator (ARC_SEGMENTED_KV). \
                 Per-request block tracking is a list of runs; dense models \
                 run the degenerate 1-segment case."
            );
        }
        Self {
            block_pool: BlockPool::new(num_gpu_blocks, enable_caching, block_size),
            block_size,
            enable_caching,
            kv_cache_group_ids,
            tracking: if segmented {
                Tracking::Segmented(Box::new(SegmentedAllocator::new(block_size)))
            } else {
                Tracking::Flat(HashMap::new())
            },
            kv_layout: KvBlockLayout::default(),
        }
    }

    /// Whether this manager is tracking requests as segment tables.
    pub fn is_segmented(&self) -> bool {
        matches!(self.tracking, Tracking::Segmented(_))
    }

    /// Set the byte layout charged to newly allocated blocks. Only affects
    /// accounting, never allocation or addressing.
    pub fn set_kv_layout(&mut self, layout: KvBlockLayout) {
        self.kv_layout = layout;
    }

    /// The segment table for a request, when segmented. `None` in flat mode.
    pub fn segment_table(&self, request_id: usize) -> Option<&SegmentTable> {
        match &self.tracking {
            Tracking::Flat(_) => None,
            Tracking::Segmented(alloc) => alloc.table(request_id),
        }
    }

    /// A kernel-ready multi-region read plan for `request_ids`.
    ///
    /// `stride` is the block-table row width; pass the same
    /// `max_blocks_per_seq` the legacy path uses to get identical padding.
    /// `None` in flat mode — a flat manager has nothing to say about regions.
    pub fn segment_plan(
        &self,
        request_ids: &[usize],
        stride: Option<usize>,
    ) -> Option<SegmentPlan> {
        match &self.tracking {
            Tracking::Flat(_) => None,
            Tracking::Segmented(alloc) => Some(alloc.plan(request_ids, stride)),
        }
    }

    /// Device bytes held by all requests, each region at its own layout.
    /// `None` in flat mode, which has no per-region layout to charge.
    pub fn allocated_bytes(&self) -> Option<u64> {
        match &self.tracking {
            Tracking::Flat(_) => None,
            Tracking::Segmented(alloc) => Some(alloc.bytes()),
        }
    }

    /// Get a reference to the block pool.
    pub fn block_pool(&self) -> &BlockPool {
        &self.block_pool
    }

    /// Get a mutable reference to the block pool.
    pub fn block_pool_mut(&mut self) -> &mut BlockPool {
        &mut self.block_pool
    }

    /// Get the null block ID (placeholder for skipped/unused slots).
    pub fn null_block_id(&self) -> usize {
        self.block_pool.null_block_id()
    }

    /// Get the block size (tokens per block).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get KV cache usage as a fraction [0.0, 1.0].
    pub fn usage(&self) -> f64 {
        self.block_pool.usage()
    }

    /// Get the number of free blocks available.
    pub fn num_free_blocks(&self) -> usize {
        self.block_pool.num_free_blocks()
    }

    /// Whether prefix caching is enabled.
    pub fn caching_enabled(&self) -> bool {
        self.enable_caching
    }

    /// Find the longest prefix cache hit for a request.
    ///
    /// Walks the request's block hashes and checks for cached blocks.
    /// Returns cached block IDs and the number of computed tokens.
    ///
    /// **Important**: When all tokens hit the cache, we must recompute the
    /// last block to produce logits. So `max_length` should be
    /// `num_tokens - 1` (the caller is responsible for this).
    pub fn get_computed_blocks(
        &self,
        block_hashes: &[BlockHash],
        num_tokens: usize,
    ) -> ComputedBlocks {
        if !self.enable_caching || block_hashes.is_empty() {
            return ComputedBlocks {
                block_ids: Vec::new(),
                num_computed_tokens: 0,
            };
        }

        // Max cache hit: at most num_tokens - 1 (need to recompute last token for logits)
        let max_cache_hit_length = num_tokens.saturating_sub(1);
        let max_num_blocks = max_cache_hit_length / self.block_size;

        let mut cached_block_ids = Vec::new();

        for (i, &block_hash) in block_hashes.iter().enumerate() {
            if i >= max_num_blocks {
                break;
            }

            // Look up this block hash across all group IDs
            if let Some(ids) = self
                .block_pool
                .get_cached_block(block_hash, &self.kv_cache_group_ids)
            {
                // For simplicity, take the first group's block.
                // Multi-group support would need to return all group block IDs
                // to construct separate block tables per group.
                debug_assert_eq!(
                    ids.len(),
                    1,
                    "Multi-group prefix cache lookup not yet implemented: found {} groups",
                    ids.len()
                );
                cached_block_ids.push(ids[0]);
            } else {
                // Chain is broken — no further blocks can match
                break;
            }
        }

        let num_computed_tokens = cached_block_ids.len() * self.block_size;

        ComputedBlocks {
            block_ids: cached_block_ids,
            num_computed_tokens,
        }
    }

    /// Allocate blocks for a request.
    ///
    /// This handles both new requests (with optional prefix cache hits) and
    /// running requests that need additional blocks.
    ///
    /// - `request_id`: The sequence ID.
    /// - `num_tokens`: Total number of tokens that need a slot (including
    ///   cached and new tokens).
    /// - `computed_blocks`: Block IDs from prefix cache (from `get_computed_blocks`).
    ///   Empty if no cache hit or if the request is already running.
    ///
    /// Returns `Some(new_block_ids)` on success, `None` if not enough free blocks.
    pub fn allocate_slots(
        &mut self,
        request_id: usize,
        num_tokens: usize,
        computed_blocks: &[usize],
    ) -> Option<Vec<usize>> {
        let num_required_blocks = num_tokens.div_ceil(self.block_size);

        let layout = self.kv_layout;
        let enable_caching = self.enable_caching;
        let Self {
            block_pool,
            tracking,
            ..
        } = self;
        let req_to_blocks = match tracking {
            Tracking::Segmented(alloc) => {
                return alloc.allocate_dense(
                    block_pool,
                    request_id,
                    num_tokens,
                    computed_blocks,
                    layout,
                );
            }
            Tracking::Flat(map) => map,
        };

        if let Some(req) = req_to_blocks.get(&request_id) {
            // Running request — just need to allocate additional blocks
            let num_existing = req.block_ids.len();
            let num_new_blocks = num_required_blocks.saturating_sub(num_existing);

            if num_new_blocks == 0 {
                return Some(Vec::new());
            }

            let new_block_ids = block_pool.get_new_blocks(num_new_blocks)?;
            req_to_blocks
                .get_mut(&request_id)
                .unwrap()
                .block_ids
                .extend_from_slice(&new_block_ids);
            return Some(new_block_ids);
        }

        // New request — incorporate computed blocks + allocate new ones
        let num_computed = computed_blocks.len();
        let num_new_blocks = num_required_blocks.saturating_sub(num_computed);

        // Count evictable blocks among computed blocks (blocks with ref_cnt == 0
        // that are in the free list — touching them will remove them from the
        // free list, so we need to account for this in the capacity check).
        let num_evictable = if enable_caching {
            computed_blocks
                .iter()
                .filter(|&&id| block_pool.block_ref_cnt(id) == 0)
                .count()
        } else {
            0
        };

        let total_needed = num_new_blocks + num_evictable;
        if total_needed > block_pool.num_free_blocks() {
            return None;
        }

        // Touch the computed blocks (increment ref_cnt, remove from free list)
        if !computed_blocks.is_empty() && enable_caching {
            block_pool.touch(computed_blocks);
        }

        // Allocate new blocks
        let new_block_ids = if num_new_blocks > 0 {
            block_pool
                .get_new_blocks(num_new_blocks)
                .expect("Should have enough blocks after capacity check")
        } else {
            Vec::new()
        };

        // Build the full block list: computed + new
        let mut all_block_ids = Vec::with_capacity(num_required_blocks);
        all_block_ids.extend_from_slice(computed_blocks);
        all_block_ids.extend_from_slice(&new_block_ids);

        req_to_blocks.insert(
            request_id,
            RequestBlocks {
                block_ids: all_block_ids,
                num_cached_blocks: num_computed,
            },
        );

        Some(new_block_ids)
    }

    /// Free all blocks for a request.
    ///
    /// Blocks are freed in reverse order so that tail blocks (most specific)
    /// are evicted first when the free list is used for LRU eviction.
    pub fn free(&mut self, request_id: usize) {
        let Self {
            block_pool,
            tracking,
            ..
        } = self;
        match tracking {
            Tracking::Segmented(alloc) => alloc.free(block_pool, request_id),
            Tracking::Flat(map) => {
                if let Some(req) = map.remove(&request_id) {
                    // Free in reverse order for LRU eviction priority
                    let reversed: Vec<usize> = req.block_ids.into_iter().rev().collect();
                    block_pool.free_blocks(&reversed);
                }
            }
        }
    }

    /// Trim a running request's allocation to `num_tokens`.
    ///
    /// This is useful when a speculative path over-allocates temporary lookahead
    /// slots and then needs to release unneeded tail blocks.
    pub fn trim_request_to_num_tokens(&mut self, request_id: usize, num_tokens: usize) {
        let num_required_blocks = num_tokens.div_ceil(self.block_size);

        let Self {
            block_pool,
            tracking,
            ..
        } = self;
        let map = match tracking {
            // A trim is a rollback of the one Dense region. Per-request, so
            // one sequence trimming cannot shorten another's allocation.
            Tracking::Segmented(alloc) => {
                alloc.rollback(block_pool, request_id, 0, num_tokens);
                return;
            }
            Tracking::Flat(map) => map,
        };

        let mut removed_blocks = {
            let Some(req) = map.get_mut(&request_id) else {
                return;
            };

            if num_required_blocks >= req.block_ids.len() {
                req.num_cached_blocks = req.num_cached_blocks.min(req.block_ids.len());
                return;
            }

            let removed = req
                .block_ids
                .drain(num_required_blocks..)
                .collect::<Vec<_>>();
            req.num_cached_blocks = req.num_cached_blocks.min(req.block_ids.len());
            removed
        };

        // Free in reverse order for LRU eviction priority.
        removed_blocks.reverse();
        block_pool.free_blocks(&removed_blocks);
    }

    /// Cache newly-full blocks after tokens are computed.
    ///
    /// Called after each step (prefill or decode) to register full blocks
    /// in the prefix cache hash map so future requests can reuse them.
    ///
    /// - `request_id`: The sequence ID.
    /// - `block_hashes`: The block hashes for the request's token sequence.
    /// - `num_computed_tokens`: Total number of tokens computed so far
    ///   (including cached tokens from prefix hits).
    pub fn cache_blocks(
        &mut self,
        request_id: usize,
        block_hashes: &[BlockHash],
        num_computed_tokens: usize,
    ) {
        if !self.enable_caching {
            return;
        }

        let block_size = self.block_size;
        let group_ids = self.kv_cache_group_ids.clone();
        let Self {
            block_pool,
            tracking,
            ..
        } = self;
        let map = match tracking {
            Tracking::Segmented(alloc) => {
                // Segment 0 is the Dense region; its group id is 0, which is
                // what `kv_cache_group_ids` is for every single-group model.
                alloc.cache_region_blocks(
                    block_pool,
                    request_id,
                    0,
                    block_hashes,
                    num_computed_tokens,
                );
                return;
            }
            Tracking::Flat(map) => map,
        };

        let req = match map.get_mut(&request_id) {
            Some(r) => r,
            None => return,
        };

        let num_full_blocks = num_computed_tokens / block_size;
        if req.num_cached_blocks >= num_full_blocks {
            return;
        }

        // Cache each full block for each group ID
        for group_id in group_ids {
            block_pool.cache_full_blocks(
                &req.block_ids,
                block_hashes,
                req.num_cached_blocks,
                num_full_blocks,
                group_id,
            );
        }

        req.num_cached_blocks = num_full_blocks;
    }

    /// Get the block IDs allocated for a request, as one contiguous run.
    ///
    /// In segmented mode this is only meaningful for a **degenerate** table
    /// (one `Dense` region anchored at token 0). A multi-region request has no
    /// single run, and concatenating its segments would be read by the caller
    /// as one — silently gathering the compressed region as if it were more
    /// window. So this returns `None` rather than lie, and callers still on
    /// the one-run contract must move to
    /// [`KVCacheManager::segment_plan`].
    pub fn get_block_ids(&self, request_id: usize) -> Option<&[usize]> {
        match &self.tracking {
            Tracking::Flat(map) => map.get(&request_id).map(|r| r.block_ids.as_slice()),
            Tracking::Segmented(alloc) => {
                let table = alloc.table(request_id)?;
                if !table.is_degenerate() {
                    return None;
                }
                Some(table.segments()[0].block_ids.as_slice())
            }
        }
    }

    /// Get the number of blocks allocated for a request, across all regions.
    pub fn num_blocks_for_request(&self, request_id: usize) -> usize {
        match &self.tracking {
            Tracking::Flat(map) => map.get(&request_id).map(|r| r.block_ids.len()).unwrap_or(0),
            Tracking::Segmented(alloc) => alloc
                .table(request_id)
                .map(|t| t.total_blocks())
                .unwrap_or(0),
        }
    }

    /// Check if a request has allocated blocks.
    pub fn has_request(&self, request_id: usize) -> bool {
        match &self.tracking {
            Tracking::Flat(map) => map.contains_key(&request_id),
            Tracking::Segmented(alloc) => alloc.has_request(request_id),
        }
    }

    /// Get the number of cached blocks for a request. In segmented mode this
    /// reports the `Dense` region, matching what the flat path counts.
    pub fn num_cached_blocks(&self, request_id: usize) -> usize {
        match &self.tracking {
            Tracking::Flat(map) => map
                .get(&request_id)
                .map(|r| r.num_cached_blocks)
                .unwrap_or(0),
            Tracking::Segmented(alloc) => alloc
                .table(request_id)
                .and_then(|t| t.segment(0))
                .map(|s| s.num_cached_blocks)
                .unwrap_or(0),
        }
    }

    /// Reset the prefix cache. Only succeeds if all blocks are free.
    pub fn reset_prefix_cache(&mut self) -> bool {
        self.block_pool.reset_prefix_cache()
    }

    /// Get the slot mapping for a request's tokens.
    ///
    /// Maps each token position to its physical slot in the KV cache:
    /// `slot = block_id * block_size + offset_within_block`
    ///
    /// - `start_token`: First token position to map (e.g., skip cached tokens).
    /// - `num_tokens`: Number of tokens to map.
    ///
    /// Returns a vector of slot indices, or `None` if the request doesn't exist.
    pub fn get_slot_mapping(
        &self,
        request_id: usize,
        start_token: usize,
        num_tokens: usize,
    ) -> Option<Vec<i64>> {
        let map = match &self.tracking {
            Tracking::Segmented(alloc) => {
                return alloc.slot_mapping(request_id, 0, start_token, num_tokens)
            }
            Tracking::Flat(map) => map,
        };
        let req = map.get(&request_id)?;
        let mut slots = Vec::with_capacity(num_tokens);

        for token_pos in start_token..start_token + num_tokens {
            let block_idx = token_pos / self.block_size;
            let offset = token_pos % self.block_size;

            if block_idx < req.block_ids.len() {
                let block_id = req.block_ids[block_idx];
                slots.push((block_id * self.block_size + offset) as i64);
            } else {
                // Should not happen if blocks are correctly allocated
                slots.push(super::_PAD_SLOT_ID);
            }
        }

        Some(slots)
    }

    /// Build the block table for a request (for the paged attention kernel).
    ///
    /// Returns the block IDs in sequence order, padded to `max_blocks` with 0.
    ///
    /// In segmented mode this is the degenerate case only, for the same reason
    /// [`KVCacheManager::get_block_ids`] is — a one-run block table cannot
    /// address a union of regions.
    pub fn get_block_table(&self, request_id: usize, max_blocks: usize) -> Option<Vec<i32>> {
        let block_ids = self.get_block_ids(request_id)?;
        let mut table = Vec::with_capacity(max_blocks);

        #[allow(clippy::cast_possible_truncation)]
        for &block_id in block_ids {
            table.push(block_id as i32);
        }

        // Pad with zeros
        table.resize(max_blocks, 0);
        Some(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::block_hash::compute_block_hashes;

    #[test]
    fn test_basic_allocation() {
        let mut mgr = KVCacheManager::new(16, 4, false, vec![0]);

        // Allocate for a request with 10 tokens (needs 3 blocks: ceil(10/4))
        let new_blocks = mgr.allocate_slots(1, 10, &[]).unwrap();
        assert_eq!(new_blocks.len(), 3);
        assert_eq!(mgr.num_blocks_for_request(1), 3);
    }

    #[test]
    fn test_running_request_extends() {
        let mut mgr = KVCacheManager::new(16, 4, false, vec![0]);

        // Initial allocation: 8 tokens = 2 blocks
        mgr.allocate_slots(1, 8, &[]).unwrap();
        assert_eq!(mgr.num_blocks_for_request(1), 2);

        // Request grows to 12 tokens = 3 blocks, needs 1 more
        let new_blocks = mgr.allocate_slots(1, 12, &[]).unwrap();
        assert_eq!(new_blocks.len(), 1);
        assert_eq!(mgr.num_blocks_for_request(1), 3);
    }

    #[test]
    fn test_allocation_fails_when_full() {
        let mut mgr = KVCacheManager::new(4, 4, false, vec![0]);
        // 4 blocks total, 1 null = 3 free

        mgr.allocate_slots(1, 12, &[]).unwrap(); // takes all 3
        assert!(mgr.allocate_slots(2, 4, &[]).is_none());
    }

    #[test]
    fn test_free_returns_blocks() {
        let mut mgr = KVCacheManager::new(8, 4, false, vec![0]);

        mgr.allocate_slots(1, 12, &[]).unwrap();
        assert_eq!(mgr.num_free_blocks(), 4); // 8-1null-3alloc = 4

        mgr.free(1);
        assert_eq!(mgr.num_free_blocks(), 7); // 8-1null = 7
        assert!(!mgr.has_request(1));
    }

    #[test]
    fn test_prefix_cache_hit() {
        let mut mgr = KVCacheManager::new(16, 4, true, vec![0]);

        // Request 1: tokens [1,2,3,4,5,6,7,8] = 2 blocks
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);

        mgr.allocate_slots(1, 8, &[]).unwrap();
        mgr.cache_blocks(1, &hashes, 8);

        // Free request 1 (blocks stay in cache)
        mgr.free(1);

        // Request 2: same prefix -> should get cache hit
        let computed = mgr.get_computed_blocks(&hashes, 12);
        assert_eq!(computed.num_computed_tokens, 8);
        assert_eq!(computed.block_ids.len(), 2);

        // Allocate with cached blocks
        let new_blocks = mgr.allocate_slots(2, 12, &computed.block_ids).unwrap();
        assert_eq!(new_blocks.len(), 1); // only 1 new block needed
        assert_eq!(mgr.num_blocks_for_request(2), 3); // 2 cached + 1 new
    }

    #[test]
    fn test_prefix_cache_partial_hit() {
        let mut mgr = KVCacheManager::new(16, 4, true, vec![0]);

        // Cache 2 blocks
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);
        mgr.allocate_slots(1, 8, &[]).unwrap();
        mgr.cache_blocks(1, &hashes, 8);
        mgr.free(1);

        // Request with 3 blocks: first 2 should hit cache, 3rd is new
        let tokens_ext: Vec<u32> = (1..=12).collect();
        let hashes_ext = compute_block_hashes(&tokens_ext, 4, &[], &[]);
        let computed = mgr.get_computed_blocks(&hashes_ext, 12);
        assert_eq!(computed.num_computed_tokens, 8);
    }

    #[test]
    fn test_cache_blocks_incremental() {
        let mut mgr = KVCacheManager::new(16, 4, true, vec![0]);

        let tokens: Vec<u32> = (1..=16).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);

        mgr.allocate_slots(1, 16, &[]).unwrap();

        // Cache first 2 blocks
        mgr.cache_blocks(1, &hashes, 8);
        assert_eq!(mgr.num_cached_blocks(1), 2);

        // Cache all 4 blocks
        mgr.cache_blocks(1, &hashes, 16);
        assert_eq!(mgr.num_cached_blocks(1), 4);
    }

    #[test]
    fn test_slot_mapping() {
        let mut mgr = KVCacheManager::new(16, 4, false, vec![0]);

        mgr.allocate_slots(1, 8, &[]).unwrap();
        let block_ids = mgr.get_block_ids(1).unwrap().to_vec();

        // Map tokens 0..8
        let slots = mgr.get_slot_mapping(1, 0, 8).unwrap();
        assert_eq!(slots.len(), 8);

        // First 4 tokens should be in block_ids[0]
        for (i, slot) in slots.iter().enumerate().take(4) {
            assert_eq!(*slot, (block_ids[0] * 4 + i) as i64);
        }
        // Next 4 in block_ids[1]
        for (i, slot) in slots[4..].iter().enumerate().take(4) {
            assert_eq!(*slot, (block_ids[1] * 4 + i) as i64);
        }
    }

    #[test]
    fn test_slot_mapping_skip_cached() {
        let mut mgr = KVCacheManager::new(16, 4, true, vec![0]);

        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);
        mgr.allocate_slots(1, 8, &[]).unwrap();
        mgr.cache_blocks(1, &hashes, 8);
        mgr.free(1);

        // New request reuses cache
        let computed = mgr.get_computed_blocks(&hashes, 12);
        mgr.allocate_slots(2, 12, &computed.block_ids).unwrap();

        // Slot mapping for only new tokens (starting from token 8)
        let slots = mgr.get_slot_mapping(2, 8, 4).unwrap();
        assert_eq!(slots.len(), 4);
    }

    #[test]
    fn test_block_table() {
        let mut mgr = KVCacheManager::new(16, 4, false, vec![0]);
        mgr.allocate_slots(1, 8, &[]).unwrap();

        let table = mgr.get_block_table(1, 5).unwrap();
        assert_eq!(table.len(), 5);
        // Last entries should be 0 (padding)
        assert_eq!(table[2], 0);
        assert_eq!(table[3], 0);
        assert_eq!(table[4], 0);
    }

    #[test]
    fn test_trim_request_allocation() {
        let mut mgr = KVCacheManager::new(8, 4, false, vec![0]);
        mgr.allocate_slots(1, 12, &[]).unwrap();
        assert_eq!(mgr.num_blocks_for_request(1), 3);
        assert_eq!(mgr.num_free_blocks(), 4); // 8 - 1 null - 3 alloc

        mgr.trim_request_to_num_tokens(1, 8); // 2 blocks
        assert_eq!(mgr.num_blocks_for_request(1), 2);
        assert_eq!(mgr.num_free_blocks(), 5);
    }

    #[test]
    fn test_trim_clamps_cached_blocks() {
        let mut mgr = KVCacheManager::new(16, 4, true, vec![0]);
        let tokens: Vec<u32> = (1..=16).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);

        mgr.allocate_slots(1, 16, &[]).unwrap();
        mgr.cache_blocks(1, &hashes, 16);
        assert_eq!(mgr.num_cached_blocks(1), 4);

        mgr.trim_request_to_num_tokens(1, 8);
        assert_eq!(mgr.num_blocks_for_request(1), 2);
        assert_eq!(mgr.num_cached_blocks(1), 2);
    }

    #[test]
    fn test_get_computed_blocks_caps_at_prompt_minus_one() {
        let mut mgr = KVCacheManager::new(16, 4, true, vec![0]);

        // Cache exactly 2 blocks (8 tokens)
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);
        mgr.allocate_slots(1, 8, &[]).unwrap();
        mgr.cache_blocks(1, &hashes, 8);
        mgr.free(1);

        // If num_tokens == 8, max_cache_hit = 7, which is less than 8 (2 blocks)
        // So we can only use 1 block (4 tokens)
        let computed = mgr.get_computed_blocks(&hashes, 8);
        assert_eq!(computed.num_computed_tokens, 4);
        assert_eq!(computed.block_ids.len(), 1);
    }

    /// Same trace, both backings, timed.
    ///
    /// ⚠️ **This measures host-side allocator bookkeeping and NOTHING ELSE.**
    /// It is not a throughput result, it is not a GPU result, and it must
    /// never be reported as validation of the segmented path (D14). Its only
    /// job is to catch the segmented backing becoming *algorithmically* worse
    /// than the flat one — a `Vec` scan where there was a `HashMap`, an
    /// allocation per decode step, that class of thing.
    ///
    /// Run: `cargo test -p mistralrs-core --lib --release
    /// segmented_backing_allocator_overhead -- --ignored --nocapture`
    #[test]
    #[ignore = "timing benchmark; run explicitly"]
    #[allow(clippy::cast_precision_loss)] // reporting a ratio, not computing one
    fn segmented_backing_allocator_overhead() {
        use std::time::Instant;

        const SEQS: usize = 64;
        const PROMPT: usize = 512;
        const DECODE: usize = 256;
        const BLOCK: usize = 16;
        const BLOCKS: usize = 1 << 16;

        fn drive(segmented: bool) -> (u128, usize) {
            let mut mgr =
                KVCacheManager::with_segmentation(BLOCKS, BLOCK, true, vec![0], segmented);
            let t0 = Instant::now();
            let mut sink = 0usize;
            for id in 0..SEQS {
                mgr.allocate_slots(id, PROMPT, &[]).expect("prompt fits");
            }
            for step in 1..=DECODE {
                for id in 0..SEQS {
                    mgr.allocate_slots(id, PROMPT + step, &[])
                        .expect("decode fits");
                    sink += mgr.num_blocks_for_request(id);
                    if let Some(t) = mgr.get_block_table(id, 64) {
                        sink += t.len();
                    }
                    if let Some(s) = mgr.get_slot_mapping(id, PROMPT + step - 1, 1) {
                        sink += s.len();
                    }
                }
            }
            for id in 0..SEQS {
                mgr.free(id);
            }
            (t0.elapsed().as_nanos(), sink)
        }

        // Warm both, then measure, so allocator warmup is not attributed to
        // whichever ran first.
        let _ = drive(false);
        let _ = drive(true);
        let (flat_ns, a) = drive(false);
        let (seg_ns, b) = drive(true);
        assert_eq!(a, b, "both backings must do the same amount of work");

        let ops = (SEQS * DECODE) as u128;
        println!(
            "host-side allocator, {SEQS} seqs x {DECODE} decode steps (NOT a GPU result):\n  \
             flat      {flat_ns:>10} ns total, {:>6} ns/step-seq\n  \
             segmented {seg_ns:>10} ns total, {:>6} ns/step-seq\n  \
             ratio     {:.2}x",
            flat_ns / ops,
            seg_ns / ops,
            seg_ns as f64 / flat_ns as f64
        );
        // Deliberately loose: this exists to catch a complexity regression,
        // not to police jitter on a shared CI box.
        assert!(
            seg_ns < flat_ns.saturating_mul(4),
            "segmented backing is {:.2}x the flat one — that is a complexity \
             change, not noise",
            seg_ns as f64 / flat_ns as f64
        );
    }

    #[test]
    fn test_reset_prefix_cache() {
        let mut mgr = KVCacheManager::new(8, 4, true, vec![0]);

        let tokens: Vec<u32> = (1..=4).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);
        mgr.allocate_slots(1, 4, &[]).unwrap();
        mgr.cache_blocks(1, &hashes, 4);

        // Can't reset while blocks are in use
        assert!(!mgr.reset_prefix_cache());

        mgr.free(1);
        assert!(mgr.reset_prefix_cache());

        // Cache should be empty now
        let computed = mgr.get_computed_blocks(&hashes, 8);
        assert_eq!(computed.num_computed_tokens, 0);
    }
}
