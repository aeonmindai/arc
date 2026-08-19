use candle_core::{Device, Result};
use itertools::Itertools;
use tracing::info;

use crate::{
    kv_cache::RecurrentStateSnapshot,
    kv_sharing::{
        KvBlockLayout, KvElemType, PagedTree, ShareStats, SharedPrefixCache, ValueAwareScorer,
    },
    paged_attention::block_hash::BlockHash,
    pipeline::KvCache,
    sequence::Sequence,
};

#[derive(Clone)]
struct CacheElement {
    cache: Vec<Option<KvCache>>,
    /// Recurrent state snapshots for hybrid models (one per recurrent layer)
    recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
    audio_hashes: Option<Vec<u64>>,
    image_hashes: Option<Vec<u64>>,
}

impl CacheElement {
    /// Whether this element's KV lives on an accelerator. CPU-resident caches
    /// (TurboQuant) do not consume the device budget and are not evicted for
    /// device pressure.
    fn is_device_resident(&self) -> bool {
        let Some(Some(first)) = self.cache.iter().find_or_first(|x| x.is_some()) else {
            return false;
        };
        let device = match first {
            KvCache::Normal { k, .. } => match k.all_data() {
                Some(d) => d.device().clone(),
                None => return false,
            },
            KvCache::Rotating { k, .. } => match k.all_data() {
                Some(d) => d.device().clone(),
                None => return false,
            },
            // TurboQuant cache is CPU-resident for now.
            KvCache::TurboQuant(_) => Device::Cpu,
            KvCache::XsRolling(xs) => match xs.tail.as_ref() {
                Some(t) => t.device().clone(),
                None => return false,
            },
        };
        !matches!(device, Device::Cpu)
    }

    /// The byte layout this element's KV is actually stored at.
    ///
    /// Read from the live tensors rather than assumed, so a cache written with
    /// `ARC_V4_FP8_KV=1` reports its real (smaller) footprint and a BF16 one
    /// reports its own. Correctness never depends on which it is — only the
    /// accounting and the eviction scorer's byte term change.
    fn layout(&self) -> KvBlockLayout {
        let Some(Some(first)) = self.cache.iter().find_or_first(|x| x.is_some()) else {
            return KvBlockLayout::default();
        };
        // Layer counts and cache widths are small by construction (tens of
        // layers, thousands of elements); the saturating conversions keep the
        // accounting honest if that ever stops being true.
        let layers =
            u32::try_from(self.cache.iter().filter(|x| x.is_some()).count()).unwrap_or(u32::MAX);
        let (k_data, v_data) = match first {
            KvCache::Normal { k, v } => (k.all_data(), v.all_data()),
            KvCache::Rotating { k, v } => (k.all_data(), v.all_data()),
            _ => {
                return KvBlockLayout {
                    layers: layers.max(1),
                    ..KvBlockLayout::default()
                }
            }
        };
        KvBlockLayout {
            k_elem: k_data.map(elem_type_of).unwrap_or(KvElemType::Bf16),
            v_elem: v_data.map(elem_type_of).unwrap_or(KvElemType::Bf16),
            k_elems_per_token: elems_per_token(k_data),
            v_elems_per_token: elems_per_token(v_data),
            layers: layers.max(1),
        }
    }
}

/// Stored element width, read from the live tensor. `U8` is how Arc stores
/// FP8 E4M3 K codes (`ARC_V4_FP8_KV=1`); nothing downstream branches on this,
/// it only makes the byte accounting match reality.
fn elem_type_of(t: &candle_core::Tensor) -> KvElemType {
    match t.dtype() {
        candle_core::DType::F32 => KvElemType::F32,
        candle_core::DType::F16 => KvElemType::F16,
        candle_core::DType::U8 => KvElemType::Fp8E4M3,
        _ => KvElemType::Bf16,
    }
}

/// Stored elements per token per layer, from a `[b, heads, seq, head_dim]`
/// cache tensor.
fn elems_per_token(t: Option<&candle_core::Tensor>) -> u32 {
    t.map(|t| {
        let d = t.dims();
        let elems = if d.len() >= 4 {
            d[1] * d[3]
        } else {
            d.last().copied().unwrap_or(1)
        };
        u32::try_from(elems).unwrap_or(u32::MAX)
    })
    .unwrap_or(0)
}

/// Longest common prefix of two media-hash lists, or `None` when the entry is
/// incompatible with the request.
///
/// Vision/audio models emit repeated placeholder tokens (e.g.
/// `<image_soft_token>`) that are identical regardless of the media behind
/// them, so token-level matching can run straight through a media region even
/// when the content differs. An entry whose overlapping media diverge would
/// hand back stale encoder output and must be refused.
fn media_overlap(input: Option<&[u64]>, cached: Option<&Vec<u64>>) -> Option<usize> {
    let matched = match (input, cached) {
        (Some(i), Some(c)) => i.iter().zip(c).take_while(|(a, b)| a == b).count(),
        _ => 0,
    };
    let input_count = input.map_or(0, |h| h.len());
    let cached_count = cached.map_or(0, |h| h.len());
    if matched < input_count.min(cached_count) {
        None
    } else {
        Some(matched)
    }
}

pub struct PrefixCacheManagerV2 {
    /// Sequence substrate: radix tree over token ids.
    caches: SharedPrefixCache<CacheElement>,
    /// Paged substrate: radix tree over block hashes. Payload carries the key
    /// length so an exact-length match can be required — recurrent state after
    /// N blocks is not the state after M blocks, so unlike KV this must not be
    /// served by truncation.
    paged_recurrent: PagedTree<(Vec<RecurrentStateSnapshot>, usize)>,
    paged_scorer: ValueAwareScorer,
    paged_entries: usize,
    n_on_device: usize,
    no_prefix_cache: bool,
    has_paged_attention: bool,
}

#[derive(Clone)]
pub enum MatchingCache {
    Normal {
        normal: Vec<Option<KvCache>>,
        recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
        images_to_keep: usize,
        audios_to_keep: usize,
        toks: Vec<u32>,
        offset: usize,
    },
}

impl PrefixCacheManagerV2 {
    pub fn new(n_on_device: usize, no_prefix_cache: bool, has_paged_attention: bool) -> Self {
        if !no_prefix_cache && !has_paged_attention {
            info!("Prefix caching enabled (radix tree, sequence-level, non-paged attention). Expect higher multi-turn throughput for both text and multimodal.");
        }
        PrefixCacheManagerV2 {
            caches: SharedPrefixCache::new(),
            paged_recurrent: PagedTree::new(1),
            paged_scorer: ValueAwareScorer::default(),
            paged_entries: 0,
            n_on_device,
            no_prefix_cache,
            has_paged_attention,
        }
    }

    fn paged_recurrent_capacity(&self) -> usize {
        self.n_on_device.max(1).saturating_mul(8)
    }

    /// Sharing statistics: hit rate, tokens not recomputed, bytes saved, and
    /// the cross-prefix reuse meter.
    pub fn share_stats(&self) -> ShareStats {
        self.caches.stats()
    }

    /// Publish a **still-running** sequence's KV as soon as its prefill is
    /// done, instead of waiting for the request to finish generating.
    ///
    /// 🔴 The defect this exists for: [`Self::add_sequence`] has exactly two
    /// callers, both inside `if let Some(reason) = is_done`
    /// (`pipeline/sampling.rs:264`, `:423`), while the lookup happens at
    /// admission (`engine/add_request.rs:650`). So a prompt's KV became visible
    /// only when the request that computed it *completed* — for a 2,048-token
    /// system prompt with a long generation that is tens of seconds after the
    /// KV existed, and every request that arrived in that window paid the full
    /// prefill again. SGLang publishes per chunk (`radix_cache.py:516`
    /// `cache_unfinished_req`); vLLM publishes per step (`block_pool.py:225`).
    ///
    /// Publishing here is safe against the producing sequence continuing to
    /// write, because the payload only ever *claims* tokens that are already
    /// written and `SingleCache::append` only ever writes **forward** of them
    /// (`kv_cache/single_cache.rs:227`). Every other mutation of a sequence's
    /// cache — `strip_pending_lead_pad`, `front_pad_kv_cache`, a capacity grow,
    /// and `clone_in_cache`'s per-step rebatch — allocates a **new** tensor and
    /// leaves the old buffer's contents untouched, so a published snapshot is
    /// never rewritten under a reader.
    ///
    /// The entry is a NEW element, not a mutation of an existing one: the
    /// payload is a snapshot of tensors at one instant, there is no in-place
    /// update path in the radix tree, and half of a longer payload is not a
    /// valid payload (`kv_sharing/radix.rs:429`). When the same sequence later
    /// finishes, its full-length payload is inserted as a second, deeper node;
    /// both are valid and `match_prefix_with` picks whichever covers the query.
    pub fn add_prefilled_sequence(
        &mut self,
        seq: &mut Sequence,
        recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
    ) {
        if self.store(seq, recurrent_snapshots) {
            seq.set_prefill_published();
        }
    }

    /// This always keeps the cache on the device.
    pub fn add_sequence(
        &mut self,
        seq: &mut Sequence,
        recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
    ) {
        self.store(seq, recurrent_snapshots);
    }

    /// The one body behind [`Self::add_sequence`] and
    /// [`Self::add_prefilled_sequence`], so a partial entry cannot drift from a
    /// final one on the key cap or the ragged guard. Returns whether an attempt
    /// to store was actually made (i.e. the guards let it through).
    fn store(
        &mut self,
        seq: &mut Sequence,
        recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
    ) -> bool {
        // Do not cache if prefix caching disabled
        if self.no_prefix_cache {
            return false;
        }

        // For paged attention, prefix caching is handled by the KVCacheManager.
        // PrefixCacheManagerV2 only handles non-paged attention caching.
        if self.has_paged_attention {
            return false;
        }

        // Feed the eviction scorer a MEASURED cost. `total_prompt_time` is set
        // by the engine from the real prefill duration
        // (`engine/mod.rs:523`, `:1015`); when it is absent or rounds to zero
        // milliseconds the observation is dropped rather than guessed at.
        //
        // Observed at most ONCE per sequence: publishing at prefill completion
        // and again at request completion would feed the scorer the same
        // measurement twice and double `cost_samples`.
        if !seq.prefill_published() {
            if let (Some(ms), toks) = (seq.total_prompt_time, seq.prompt_tokens()) {
                let elapsed_ns = u64::try_from(ms)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(1_000_000);
                self.caches.observe_prefill(toks, elapsed_ns);
            }
        }

        // 🔑 THE READER THAT WOULD BE POISONED. A sequence that was part of a
        // front-aligned ragged cohort still carries its zero-filled dead prefix
        // until the next `clone_in_cache` strips it — and for a sequence that
        // has just FINISHED there is no next `clone_in_cache`. Storing it here
        // would put padding into the prefix cache and hand it to every future
        // request that matches this prefix, which is a wrong answer that
        // outlives the request that produced it.
        //
        // `clone_out_cache` deliberately defers the strip (it runs per token,
        // for data only re-read on a membership change), so the deferral is
        // paid here instead — at the read, where it is one sequence rather than
        // `O(B x layers)` per step. A no-op for every non-ragged sequence.
        //
        // This guard covers the mid-flight publication too, and that is why the
        // two paths share this body: a partially-written entry is exactly the
        // hazard it exists for. A sequence that has just PREFILLED out of a
        // front-aligned ragged cohort carries the same zero-filled dead prefix
        // as one that has just finished, and here the strip is not merely
        // deferred but *earlier* than `clone_in_cache` would have paid it —
        // which is equivalent, because `take_pending_lead_pad` removes the
        // record and the later `clone_in_cache` then finds `lead == 0`.
        if let Err(e) = crate::kv_cache::strip_pending_lead_pad(seq, false) {
            tracing::warn!(
                "prefix cache: refusing to store sequence {} — could not drop its ragged \
                 dead prefix ({e}). Storing it would cache zero-filled K/V as content.",
                seq.id()
            );
            return false;
        }

        let cache = seq.normal_cache().to_vec();

        // Key the entry by the token prefix the cache ACTUALLY covers.
        //
        // `search_for_matching_cache` calls `set_len(match_len)` on the
        // stored layers, and `SingleCache::try_set_len` only checks
        // capacity, not validity — so a key longer than the cache is an
        // over-claim that hands out uninitialised K/V. Plain decode is safe
        // by an accident of two facts that happen to cancel: `cache_len ==
        // len - 1`, and a full-length match is rejected by the
        // `new_toks.is_empty()` guard, which caps `match_len` at `len - 1`.
        //
        // Batched MTP breaks the accident: a sequence can finish carrying
        // an uncached committed tail of up to `depth + 1` tokens
        // (`mtp_pipeline.rs`, the window invariant), so `match_len` could
        // reach `len - 1` while the cache stops at `len - u`. Capping the
        // key at `cache_len + 1` restores the guard's arithmetic exactly,
        // and is a no-op whenever `u == 1`.
        //
        // The cap is what makes a PARTIAL entry safe as well: a sequence
        // published mid-flight is keyed by the prefix its cache actually
        // covers, so it can no more over-claim than a finished one can. It is
        // read from the live cache rather than from any notion of "how far
        // this request has got", so it is true whenever it is taken.
        let cache_len = seq.cache_bucket_len();
        let toks = seq.get_toks();
        let key_len = toks.len().min(cache_len + 1);
        let key = toks[..key_len].to_vec();

        let element = CacheElement {
            cache,
            recurrent_snapshots,
            image_hashes: seq.image_hashes().map(|x| x.to_vec()),
            audio_hashes: seq.audio_hashes().map(|x| x.to_vec()),
        };
        let layout = element.layout();
        // A `None` return means the tree already covers every token of this
        // key, so the second copy was not stored. Those bytes show up in
        // `share_stats().bytes_saved()`.
        let _ = self.caches.insert(&key, element, layout);
        true
    }

    /// Evict the caches. This will evict the first k seqs such that the number of sequences on device after the copy is
    /// the maximum allowed. Returns the number of evicted sequences.
    ///
    /// Eviction order is decided by the value-aware scorer
    /// ([`crate::kv_sharing::ValueAwareScorer`]) rather than insertion order:
    /// measured recompute cost times reuse probability, decayed by staleness.
    pub fn evict_caches(&mut self) -> Result<usize> {
        if self.no_prefix_cache || self.has_paged_attention {
            return Ok(0);
        }
        let mut n_evicted = 0;
        loop {
            let n_device = self
                .caches
                .elements()
                .filter(|e| e.is_device_resident())
                .count();
            if n_device <= self.n_on_device {
                break;
            }
            let excess = n_device - self.n_on_device;
            let target = self.caches.len().saturating_sub(excess);
            let dropped = self.caches.evict_to_capacity(target);
            if dropped.is_empty() {
                // Nothing further is evictable (all remaining are locked or
                // the tree is empty); stop rather than spin.
                break;
            }
            n_evicted += dropped.iter().filter(|e| e.is_device_resident()).count();
        }
        Ok(n_evicted)
    }

    /// Evict all the caches.
    pub fn evict_all_caches(&mut self) -> Result<usize> {
        let len = self.caches.len();
        self.caches.clear();
        self.paged_recurrent.clear();
        self.paged_entries = 0;
        Ok(len)
    }

    /// Add a recurrent-state snapshot for a paged-attention block-hash prefix key.
    /// This is used by hybrid models to restore recurrent states alongside paged KV prefix hits.
    pub fn add_paged_recurrent_prefix(
        &mut self,
        key: Vec<BlockHash>,
        snapshots: Vec<RecurrentStateSnapshot>,
    ) {
        if self.no_prefix_cache
            || !self.has_paged_attention
            || key.is_empty()
            || snapshots.is_empty()
        {
            return;
        }

        let key_len = key.len();
        let outcome =
            self.paged_recurrent
                .insert(&key, (snapshots, key_len), KvBlockLayout::default(), 0);
        if !outcome.subsumed {
            self.paged_entries += 1;
        }

        let cap = self.paged_recurrent_capacity();
        while self.paged_entries > cap {
            let dropped = self
                .paged_recurrent
                .evict_payloads(self.paged_entries - cap, &self.paged_scorer);
            if dropped.is_empty() {
                break;
            }
            self.paged_entries -= dropped.len();
        }
    }

    /// Lookup a recurrent-state snapshot for a paged-attention block-hash prefix key.
    /// Returns a cloned snapshot.
    pub fn get_paged_recurrent_prefix(
        &mut self,
        key: &[BlockHash],
    ) -> Option<Vec<RecurrentStateSnapshot>> {
        if self.no_prefix_cache || !self.has_paged_attention || key.is_empty() {
            return None;
        }

        let outcome = self.paged_recurrent.match_prefix(key);
        let ((snapshots, stored_len), covers) = outcome.payload?;
        // Exactness is REQUIRED here, unlike KV. A recurrent state is the
        // state *after* exactly `stored_len` blocks; truncating it to a
        // shorter prefix, or extending a shorter one, is not the same state.
        if stored_len != key.len() || covers != key.len() {
            return None;
        }
        Some(snapshots)
    }

    /// Search for a matching cache given some tokens. Image-containing sequences are now cached too.
    ///
    /// O(|toks|) symbol comparisons through the radix tree. The structure this
    /// replaced compared the request against every cached entry in turn —
    /// O(entries x |toks|).
    pub fn search_for_matching_cache(
        &mut self,
        toks: &[u32],
        image_hashes: Option<&[u64]>,
        audio_hashes: Option<&[u64]>,
    ) -> Result<Option<MatchingCache>> {
        // Do not search if prefix caching disabled or no tokens
        if self.no_prefix_cache || toks.is_empty() {
            return Ok(None);
        }

        if self.has_paged_attention {
            // For paged attention, prefix caching is handled by the KVCacheManager.
            // PrefixCacheManagerV2 only handles non-paged attention caching.
            return Ok(None);
        }

        let hit = self.caches.lookup(toks, |element| {
            media_overlap(image_hashes, element.image_hashes.as_ref()).is_some()
                && media_overlap(audio_hashes, element.audio_hashes.as_ref()).is_some()
        });
        let Some(hit) = hit else {
            return Ok(None);
        };
        let match_len = hit.matched_tokens;
        let mut cache = hit.element.clone();

        let new_toks = toks[match_len.min(toks.len())..].to_vec();
        if new_toks.is_empty() {
            // A full-length match leaves nothing to run, and is also what keeps
            // `match_len` below the cached length; see `add_sequence`.
            return Ok(None);
        }

        let images_match_until = media_overlap(image_hashes, cache.image_hashes.as_ref())
            .expect("filter admitted an incompatible element");
        let audios_match_until = media_overlap(audio_hashes, cache.audio_hashes.as_ref())
            .expect("filter admitted an incompatible element");
        let images_to_keep = image_hashes.map_or(0, |h| h.len().saturating_sub(images_match_until));
        let audios_to_keep = audio_hashes.map_or(0, |h| h.len().saturating_sub(audios_match_until));

        // Validate EVERY layer before mutating ANY of them. The two loops are
        // deliberate: `try_set_len` does not mutate, so a cache that fails
        // validation is left exactly as it was and the request simply runs
        // uncached. There is no state in which some layers have been retruncated
        // and others have not.
        //
        // Both guards below decline loudly. The KV length contract is currently
        // fragile elsewhere in the engine (two decode-path panics in ~1,300
        // requests, owned by `fix/kv-length-invariants`), so a prefix cache that
        // quietly hands back an unexpected length would be indistinguishable
        // from those. Declining is correct; declining *silently* is not.
        for layer in cache.cache.iter_mut().flatten() {
            if layer.try_set_len(match_len).is_err() {
                tracing::warn!(
                    match_len,
                    capacity = layer.current_seq_len(),
                    "prefix cache: declining a match whose length exceeds the \
                     stored cache capacity; serving this request uncached"
                );
                return Ok(None);
            }
            // `try_set_len` checks CAPACITY, not validity — it will happily
            // extend a length into slots that were never written. Declining
            // a match longer than the cache actually holds is the belt to
            // `add_sequence`'s braces (it caps the key at the covered
            // prefix); either alone is sufficient, and serving
            // uninitialised K/V is a wrong answer nothing downstream
            // catches.
            if match_len > layer.current_seq_len() {
                tracing::warn!(
                    match_len,
                    stored = layer.current_seq_len(),
                    "prefix cache: declining a match longer than the KV actually \
                     stored; serving this request uncached"
                );
                return Ok(None);
            }
        }
        for layer in cache.cache.iter_mut().flatten() {
            layer.set_len(match_len)?;
        }
        Ok(Some(MatchingCache::Normal {
            normal: cache.cache,
            recurrent_snapshots: cache.recurrent_snapshots,
            images_to_keep,
            audios_to_keep,
            toks: new_toks,
            offset: match_len,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::block_hash::hash_block_tokens;
    use candle_core::{DType, Tensor};

    /// Real `BlockHash` values, produced the way production produces them, so
    /// the fixture cannot drift from the hashing the engine actually uses.
    fn bh(tok: u32) -> BlockHash {
        hash_block_tokens(None, &[tok], None)
    }

    fn snapshot() -> RecurrentStateSnapshot {
        let t = Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap();
        RecurrentStateSnapshot {
            conv_state: t.clone(),
            recurrent_state: t,
            seqlen_offset: 0,
        }
    }

    fn el(images: Option<Vec<u64>>, audios: Option<Vec<u64>>) -> CacheElement {
        CacheElement {
            cache: Vec::new(),
            recurrent_snapshots: None,
            image_hashes: images,
            audio_hashes: audios,
        }
    }

    #[test]
    fn media_overlap_matches_the_previous_linear_scan_semantics() {
        // No media on either side: compatible, nothing to skip.
        assert_eq!(media_overlap(None, None), Some(0));
        // Request has images, entry has none: compatible; all images are new.
        assert_eq!(media_overlap(Some(&[1, 2]), None), Some(0));
        // Entry has images, request has none: compatible.
        assert_eq!(media_overlap(None, Some(&vec![1, 2])), Some(0));
        // Shared leading image, second diverges: overlap 1 < min(2,2) -> refuse.
        assert_eq!(media_overlap(Some(&[1, 9]), Some(&vec![1, 2])), None);
        // Entry is a strict prefix of the request's media: compatible.
        assert_eq!(media_overlap(Some(&[1, 2, 3]), Some(&vec![1, 2])), Some(2));
        // First image already differs.
        assert_eq!(media_overlap(Some(&[9]), Some(&vec![1])), None);
    }

    #[test]
    fn device_residency_is_false_for_an_empty_cache() {
        assert!(!el(None, None).is_device_resident());
    }

    #[test]
    fn layout_of_an_empty_cache_is_the_neutral_default() {
        assert_eq!(el(None, None).layout(), KvBlockLayout::default());
    }

    #[test]
    fn disabled_manager_never_matches_or_stores() {
        let mut m = PrefixCacheManagerV2::new(4, true, false);
        assert!(m
            .search_for_matching_cache(&[1, 2, 3], None, None)
            .unwrap()
            .is_none());
        assert_eq!(m.share_stats().entries, 0);
    }

    #[test]
    fn paged_manager_defers_to_the_block_pool() {
        let mut m = PrefixCacheManagerV2::new(4, false, true);
        assert!(m
            .search_for_matching_cache(&[1, 2, 3], None, None)
            .unwrap()
            .is_none());
    }

    /// A KV slot built the way the ENGINE builds it: `all_data` preallocated
    /// before the first append, dtype and shape fixed up front.
    ///
    /// D12 / wave49-BZ: every KV test in this repo used `KvCache::new_normal`,
    /// whose `all_data` is `None` and which therefore allocates from the first
    /// `src` and can never disagree with it. That made a whole suite incapable
    /// of reproducing production. This mirrors
    /// `deepseek4.rs::preallocated_v4_slot` instead, so the layout probe reads
    /// a real tensor with a real dtype and a real width.
    fn preallocated_slot(
        capacity: usize,
        seq_len: usize,
        head_dim: usize,
        dtype: DType,
    ) -> KvCache {
        let shape = (1usize, 1usize, capacity, head_dim);
        KvCache::Normal {
            k: crate::kv_cache::SingleCache {
                all_data: Some(Tensor::zeros(shape, dtype, &Device::Cpu).unwrap()),
                dim: 2,
                current_seq_len: seq_len,
                max_seq_len: 4096,
                capacity_seq_len: capacity,
                first_cached: 0,
                retain_window: None,
            },
            v: crate::kv_cache::SingleCache {
                all_data: Some(Tensor::zeros(shape, DType::BF16, &Device::Cpu).unwrap()),
                dim: 2,
                current_seq_len: seq_len,
                max_seq_len: 4096,
                capacity_seq_len: capacity,
                first_cached: 0,
                retain_window: None,
            },
        }
    }

    fn seq_with(tokens: Vec<u32>, cache_len: usize, layers: usize, dtype: DType) -> Sequence {
        use crate::sampler::Sampler;
        use crate::sequence::{SeqStepType, SequenceGroup, SequenceRecognizer};
        use std::sync::Arc;

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            0.0,
            0.0,
            None,
            vec![],
            None,
        )
        .unwrap();
        let group = Arc::new(std::sync::Mutex::new(SequenceGroup::new(
            1, false, false, None,
        )));
        let mut seq = Sequence::new_waiting(
            tokens,
            String::new(),
            0,
            0,
            layers,
            tx,
            sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            vec![],
        );
        let cache = seq.normal_cache();
        cache.clear();
        for _ in 0..layers {
            cache.push(Some(preallocated_slot(512, cache_len, 128, dtype)));
        }
        seq
    }

    /// 🔑 A sequence that finished out of a front-aligned RAGGED cohort must not
    /// put its zero-filled dead prefix into the prefix cache.
    ///
    /// `clone_out_cache` defers the strip — it runs per token, for data only
    /// re-read on a membership change — and a sequence that has just FINISHED
    /// never gets that membership change. So without the strip at this reader,
    /// the padding is stored and handed to every future request matching this
    /// prefix: a wrong answer that outlives the request that produced it, and
    /// one `--prefix-cache-n 0` hides completely.
    ///
    /// Pre-fix this stores `cache_len` positions of which the first `lead` are
    /// zeros, and the entry is keyed as if all of them were real.
    #[test]
    fn a_finished_ragged_sequence_does_not_poison_the_prefix_cache() {
        let mut m = PrefixCacheManagerV2::new(16, false, false);
        let base: Vec<u32> = (0..64).collect();
        let mut seq = seq_with(base.clone(), base.len(), 4, DType::BF16);

        // The sequence was row 1 of a cohort front-aligned to 64: 4 dead
        // columns at the front, 60 real positions behind them.
        crate::kv_cache::test_support::set_pending_lead_pad(*seq.id(), 4);

        m.add_sequence(&mut seq, None);

        for layer in seq.normal_cache().iter().flatten() {
            assert_eq!(
                layer.current_seq_len(),
                60,
                "the dead prefix must be dropped before the cache is read, leaving the \
                 row's 60 REAL positions — storing 64 would cache 4 zero-filled K/V rows \
                 as content"
            );
        }
        assert_eq!(
            m.share_stats().entries,
            1,
            "and it must still be stored, just correctly"
        );
    }

    /// End to end through the real manager: store a finished sequence, then
    /// serve a continuation of it.
    #[test]
    fn end_to_end_store_then_serve_a_continuation() {
        let mut m = PrefixCacheManagerV2::new(16, false, false);
        let base: Vec<u32> = (0..64).collect();
        let mut seq = seq_with(base.clone(), base.len(), 4, DType::BF16);
        m.add_sequence(&mut seq, None);
        assert_eq!(m.share_stats().entries, 1);

        let mut query = base.clone();
        query.extend([900, 901, 902]);
        let hit = m
            .search_for_matching_cache(&query, None, None)
            .unwrap()
            .expect("a 64-token shared prefix must be served");
        let MatchingCache::Normal {
            offset,
            toks,
            normal,
            ..
        } = hit;
        assert_eq!(offset, 64, "must reuse all 64 cached tokens");
        assert_eq!(
            toks,
            vec![900, 901, 902],
            "only the new tokens are returned"
        );
        assert_eq!(normal.len(), 4);
        for layer in normal.iter().flatten() {
            assert_eq!(layer.current_seq_len(), 64);
        }
        let stats = m.share_stats();
        assert_eq!(stats.tokens_not_recomputed(), 64);
    }

    /// 🔴 THE DISCRIMINATOR. Two concurrent requests behind one cold system
    /// prompt: the second must see the first's KV as soon as the first's
    /// PREFILL is done, not when the first request finishes generating.
    ///
    /// Before `add_prefilled_sequence`, `add_sequence` was reachable only from
    /// inside `if let Some(reason) = is_done` (`pipeline/sampling.rs:264`,
    /// `:423`), so nothing at all was published while a request ran and this
    /// lookup returned `None` — K requests on one 2,048-token system prompt
    /// each paid the full 5.90 s prefill instead of `5.90 s + (K-1) x eps`.
    #[test]
    fn a_shared_prefix_is_visible_before_the_producing_request_finishes() {
        let mut m = PrefixCacheManagerV2::new(16, false, false);
        let system: Vec<u32> = (0..64).collect();

        // Request A: the 64-token system prompt plus its own 2-token question,
        // one step past prefill — 66 tokens of KV, and the token it just
        // sampled appended but not yet cached (`cache_len == len - 1`, the
        // invariant `add_sequence`'s key cap is built on).
        let mut a_toks = system.clone();
        a_toks.extend([700, 701, 900]);
        let mut a = seq_with(a_toks, 66, 4, DType::BF16);
        m.add_prefilled_sequence(&mut a, None);
        assert!(
            a.prefill_published(),
            "the sequence must record that its prefill was published, so the \
             measured prefill cost is not observed twice when it finishes"
        );

        // A has NOT finished — `add_sequence` was never called for it.
        // Request B shares the system prompt and diverges after it.
        let mut b_toks = system.clone();
        b_toks.extend([800, 801]);
        let hit = m
            .search_for_matching_cache(&b_toks, None, None)
            .unwrap()
            .expect(
                "a concurrent request sharing the 64-token system prompt must be \
                 served from the still-running request's prefill",
            );
        let MatchingCache::Normal { offset, toks, .. } = hit;
        assert_eq!(
            offset, 64,
            "the whole shared system prompt must be reused, not recomputed"
        );
        assert_eq!(toks, vec![800, 801], "only B's own tokens are left to run");
    }

    /// A partial entry must not over-claim any more than a final one can. The
    /// key is capped at what the cache covers, so a query that shares more
    /// tokens than the producing sequence had computed at publication time is
    /// served the covered prefix — never the uncomputed remainder.
    #[test]
    fn a_partial_entry_never_serves_more_than_it_had_computed() {
        let mut m = PrefixCacheManagerV2::new(16, false, false);
        let base: Vec<u32> = (0..64).collect();
        // Published while only 40 of the 64 tokens were in the cache.
        let mut seq = seq_with(base.clone(), 40, 2, DType::BF16);
        m.add_prefilled_sequence(&mut seq, None);

        let mut query = base.clone();
        query.push(900);
        let hit = m.search_for_matching_cache(&query, None, None).unwrap();
        match hit {
            None => {}
            Some(MatchingCache::Normal { offset, normal, .. }) => {
                assert!(
                    offset <= 40,
                    "a partial entry served {offset} tokens from a 40-token cache"
                );
                for layer in normal.iter().flatten() {
                    assert!(layer.current_seq_len() <= 40);
                }
            }
        }
    }

    /// The ragged dead-prefix guard covers the mid-flight publication too — a
    /// partially-written entry is exactly the hazard it exists for, so the two
    /// paths share one body rather than the new one bypassing it.
    #[test]
    fn a_prefilled_ragged_sequence_does_not_poison_the_prefix_cache() {
        let mut m = PrefixCacheManagerV2::new(16, false, false);
        let base: Vec<u32> = (0..64).collect();
        let mut seq = seq_with(base.clone(), base.len(), 4, DType::BF16);

        // Row 1 of a cohort front-aligned to 64: 4 dead columns at the front.
        crate::kv_cache::test_support::set_pending_lead_pad(*seq.id(), 4);

        m.add_prefilled_sequence(&mut seq, None);

        for layer in seq.normal_cache().iter().flatten() {
            assert_eq!(
                layer.current_seq_len(),
                60,
                "the dead prefix must be dropped before a mid-flight entry is \
                 stored — storing 64 would publish 4 zero-filled K/V rows as \
                 content to every later request matching this prefix"
            );
        }
        assert_eq!(m.share_stats().entries, 1);
    }

    /// The over-claim guard, on a production-shaped cache. A key longer than
    /// the KV it stands for must be refused rather than serving uninitialised
    /// K/V — `try_set_len` checks capacity, not validity.
    #[test]
    fn a_match_longer_than_the_stored_kv_is_refused() {
        let mut m = PrefixCacheManagerV2::new(16, false, false);
        let base: Vec<u32> = (0..64).collect();
        // Cache covers only 40 of the 64 tokens. `add_sequence` caps the key at
        // 41; a query sharing 41 tokens then exceeds the 40 actually stored.
        let mut seq = seq_with(base.clone(), 40, 2, DType::BF16);
        m.add_sequence(&mut seq, None);

        let mut query = base.clone();
        query.extend([900]);
        let hit = m.search_for_matching_cache(&query, None, None).unwrap();
        assert!(
            hit.is_none(),
            "a 41-token match against 40 stored tokens must be refused"
        );
    }

    /// Improvement 1: the byte layout comes from the tensor, so an FP8 K store
    /// is accounted at its real width without any global switch.
    #[test]
    fn layout_is_read_from_the_stored_tensor_not_assumed() {
        let bf16 = CacheElement {
            cache: vec![Some(preallocated_slot(512, 8, 128, DType::BF16))],
            recurrent_snapshots: None,
            image_hashes: None,
            audio_hashes: None,
        };
        let fp8_k = CacheElement {
            cache: vec![Some(preallocated_slot(512, 8, 128, DType::U8))],
            recurrent_snapshots: None,
            image_hashes: None,
            audio_hashes: None,
        };
        assert_eq!(bf16.layout().k_elem, KvElemType::Bf16);
        assert_eq!(fp8_k.layout().k_elem, KvElemType::Fp8E4M3);
        // V stays BF16 in both, which is exactly Arc's K-only FP8 shape.
        assert_eq!(fp8_k.layout().v_elem, KvElemType::Bf16);
        assert!(
            fp8_k.layout().bytes_for_tokens(1000) < bf16.layout().bytes_for_tokens(1000),
            "a compressed block must account for fewer bytes"
        );
        assert_eq!(bf16.layout().k_elems_per_token, 128);
    }

    #[test]
    fn paged_recurrent_prefix_requires_an_exact_key() {
        let mut m = PrefixCacheManagerV2::new(2, false, true);
        let key = vec![bh(1), bh(2)];
        m.add_paged_recurrent_prefix(key.clone(), vec![snapshot()]);

        assert!(m.get_paged_recurrent_prefix(&key).is_some());
        // A strict prefix must MISS: the recurrent state after 1 block is not
        // the state after 2, and serving it by truncation would be wrong.
        assert!(m.get_paged_recurrent_prefix(&key[..1]).is_none());
        // A longer key must also miss.
        let mut longer = key.clone();
        longer.push(bh(3));
        assert!(m.get_paged_recurrent_prefix(&longer).is_none());
    }

    #[test]
    fn paged_recurrent_prefix_evicts_at_capacity() {
        let cap = PrefixCacheManagerV2::new(1, false, true).paged_recurrent_capacity();
        let mut m = PrefixCacheManagerV2::new(1, false, true);
        for i in 0..(u32::try_from(cap).unwrap() + 5) {
            m.add_paged_recurrent_prefix(vec![bh(i), bh(i + 1000)], vec![snapshot()]);
        }
        assert!(
            m.paged_entries <= cap,
            "paged recurrent cache grew past capacity: {} > {cap}",
            m.paged_entries
        );
    }
}
