use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::{Result, Tensor, D};

use crate::{
    get_mut_arcmutex,
    pipeline::{CacheManagerMixin, MetadataMixin},
    sequence::Sequence,
};

mod full_cache;
mod hybrid_cache;
mod rotating_cache;
mod single_cache;
pub mod turboquant_cache;
mod xs_rolling;
/// Thread-local `ARC_V4_XS_PER_SEQ` override, so a test outside this module can
/// exercise both sides of the flag (the production read is a `OnceLock`).
#[cfg(test)]
pub(crate) use xs_rolling::test_override as xs_per_seq_test_override;

pub use full_cache::{EitherCache, LayerCaches};
pub use hybrid_cache::{
    HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    RecurrentStateSnapshot,
};
pub use rotating_cache::RotatingCache;
pub use single_cache::SingleCache;
pub use turboquant_cache::TurboQuantCache;
pub use xs_rolling::{
    request_xs_per_sequence, xs_per_sequence_enabled, XsRollingCache, XS_TAIL_MARGIN_TOKENS,
};

pub trait CacheManager<T: CacheManagerMixin + MetadataMixin + ?Sized> {
    /// Build one dense batched cache from `seqs`' per-sequence caches.
    ///
    /// Fallible on purpose: a batch whose sequences disagree about their cache
    /// length cannot be represented as one dense cache, and the only honest
    /// answer is to refuse. Returning `Err` here lets `Pipeline::step`'s caller
    /// fail the *requests* (`handle_pipeline_forward_error!`) instead of
    /// panicking the engine task and orphaning every other in-flight sequence.
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) -> Result<()>;
    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool);
    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    );
}

#[derive(Debug, Clone)]
pub enum KvCache {
    Normal {
        k: SingleCache,
        v: SingleCache,
    },
    Rotating {
        k: RotatingCache,
        v: RotatingCache,
    },
    TurboQuant(Box<TurboQuantCache>),
    /// DeepSeek V4's rolling compressor state (see [`XsRollingCache`]). Not a
    /// K/V cache: it holds completed compressed rows plus a bounded raw tail,
    /// and reports its length in tokens so the generic truncation paths keep
    /// working.
    XsRolling(Box<XsRollingCache>),
}

impl KvCache {
    pub fn new_normal(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        let k = SingleCache::new(dim, max_seq_len, capacity_seq_len);
        let v = SingleCache::new(dim, max_seq_len, capacity_seq_len);
        Self::Normal { k, v }
    }

    pub fn new_rotating(dim: usize, sliding_window: usize, capacity_seq_len: usize) -> Self {
        let k = RotatingCache::new(dim, sliding_window, capacity_seq_len);
        let v = RotatingCache::new(dim, sliding_window, capacity_seq_len);
        Self::Rotating { k, v }
    }

    pub fn new_turboquant(config: &mistralrs_quant::turboquant::TurboQuantConfig) -> Self {
        Self::TurboQuant(Box::new(TurboQuantCache::new(config)))
    }

    /// Fallible [`Self::new_turboquant`]: reports the geometries TurboQuant
    /// cannot serve instead of panicking inside a model constructor.
    pub fn try_new_turboquant(
        config: &mistralrs_quant::turboquant::TurboQuantConfig,
    ) -> std::result::Result<Self, String> {
        Ok(Self::TurboQuant(Box::new(TurboQuantCache::try_new(
            config,
        )?)))
    }

    /// The variant's name, for diagnostics that have to say which slot refused.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Normal { .. } => "Normal",
            Self::Rotating { .. } => "Rotating",
            Self::TurboQuant(_) => "TurboQuant",
            Self::XsRolling(_) => "XsRolling",
        }
    }

    /// Can this slot hold a length that differs from the rest of its batch?
    ///
    /// One dense batched buffer writes every sequence's new rows at ONE offset
    /// (`SingleCache::append`), so a per-sequence length is representable only
    /// if a ragged cohort can be re-aligned so that every row's live content
    /// *ends* at the same column — [`front_pad_kv_cache`]. Each variant:
    ///
    /// * **`Normal` — yes.** Its content is a flat run of `current_seq_len`
    ///   rows on one dim with no other state, so shifting the run right and
    ///   zero-filling ahead of it is exact.
    /// * **`Rotating` — no.** A ring buffer whose `offset` names where the
    ///   window wrapped; shifting the data would leave `offset` describing the
    ///   wrong rows, and the wrap point is not a function of the length.
    /// * **`TurboQuant` — no.** Content is quantized blocks with their own
    ///   grouping; a row shift is not a block shift.
    /// * **`XsRolling` — yes, under `ARC_V4_XS_PER_SEQ`.** It carries *two*
    ///   time bases (completed compressed rows on `comp`, the retained raw
    ///   window on `tail`), and until wave63-CO both were governed by a single
    ///   `tokens`/`base` pair for the whole batch — the keystone gap
    ///   `wave61-CL` §6 ("the compressor's `xs` history has no block table"),
    ///   PR #92 §5.1 and `wave29-BC` §4b all landed on. They are now per-row:
    ///   `comp` stays start-anchored and takes a `slot_mapping`-shaped scatter
    ///   so each row appends at its own column, and `tail` is end-anchored so
    ///   the one shared append offset serves every row — the same
    ///   left-alignment trick [`front_pad_kv_cache`] uses for K/V. The flag
    ///   gates only whether the engine may *build* such a batch; with it off
    ///   nothing produces more than one row of state and every path is
    ///   byte-identical to before.
    pub fn supports_per_sequence_len(&self) -> bool {
        match self {
            Self::Normal { .. } => true,
            Self::XsRolling(_) => xs_per_sequence_enabled(),
            Self::Rotating { .. } | Self::TurboQuant(_) => false,
        }
    }

    pub fn k(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { k, .. } => k.current_data(),
            Self::Rotating { k, .. } => k.current_data(),
            Self::TurboQuant(tq) => tq.k.current_data(),
            // The compressed rows are this entry's "keys".
            Self::XsRolling(xs) => xs.comp.current_data(),
        }
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { v, .. } => v.current_data(),
            Self::Rotating { v, .. } => v.current_data(),
            Self::TurboQuant(tq) => tq.v.current_data(),
            // The retained raw tail is this entry's "values".
            Self::XsRolling(xs) => Ok(xs.tail.clone()),
        }
    }

    /// CUDA-graph-capturable KV append (RUN-161 2c). Writes the new K/V at the
    /// device-held `position` slot and returns fixed `[B,H,read_capacity,D]`
    /// windows (constant shape across decode steps). Normal (SingleCache) only.
    pub fn append_graph(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        position: &Tensor,
        read_capacity: usize,
    ) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        match self {
            Self::Normal { k: kc, v: vc } => {
                let out_k = kc.append_graph(&k, position, read_capacity)?;
                let out_v = vc.append_graph(&v, position, read_capacity)?;
                Ok((out_k, out_v))
            }
            _ => {
                candle_core::bail!("append_graph: only the Normal KV cache supports graph capture")
            }
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        if let Self::TurboQuant(tq) = self {
            return tq.append(&k, &v);
        }
        let (out_k, out_v) = match self {
            Self::Normal { k: kc, v: vc } => {
                kc.append(&k)?;
                vc.append(&v)?;
                (kc.current_data()?, vc.current_data()?)
            }
            Self::Rotating { k: kc, v: vc } => {
                let out_k = kc.append(&k)?;
                let out_v = vc.append(&v)?;
                (Some(out_k), Some(out_v))
            }
            Self::TurboQuant(_) => unreachable!(),
            Self::XsRolling(_) => candle_core::bail!(
                "KvCache::append: the V4 xs rolling cache is advanced through \
                 `XsRollingCache::advance`, not the K/V append path"
            ),
        };
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                match self {
                    Self::Normal { k, .. } => shape[k.dim] = 0,
                    Self::Rotating { k, .. } => shape[k.dim] = 0,
                    Self::TurboQuant(_) | Self::XsRolling(_) => unreachable!(),
                }
                Tensor::zeros(shape, k.dtype(), k.device())?
            }
            Some(k) => k,
        };
        let v = match out_v {
            None => {
                let mut shape = v.dims().to_vec();
                match self {
                    Self::Normal { v, .. } => shape[v.dim] = 0,
                    Self::Rotating { v, .. } => shape[v.dim] = 0,
                    Self::TurboQuant(_) | Self::XsRolling(_) => unreachable!(),
                }
                Tensor::zeros(shape, v.dtype(), v.device())?
            }
            Some(v) => v,
        };
        Ok((k, v))
    }

    pub fn current_seq_len(&self) -> usize {
        match self {
            Self::Normal { k, .. } => k.current_seq_len(),
            Self::Rotating { k, .. } => k.current_seq_len(),
            Self::TurboQuant(tq) => tq.current_seq_len(),
            Self::XsRolling(xs) => xs.current_seq_len(),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Normal { k, v } => {
                k.reset();
                v.reset();
            }
            Self::Rotating { k, v } => {
                k.reset();
                v.reset();
            }
            Self::TurboQuant(tq) => {
                tq.reset();
            }
            Self::XsRolling(xs) => {
                xs.reset();
            }
        }
    }

    /// Returns Ok if the length reassignment was successful, otherwise returns Err.
    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        match self {
            Self::Normal { k, v } => {
                k.set_len(len)?;
                v.set_len(len)?;
                Ok(())
            }
            Self::Rotating { k, v } => {
                k.set_len(len)?;
                v.set_len(len)?;
                Ok(())
            }
            Self::TurboQuant(tq) => {
                tq.k.set_len(len)?;
                tq.v.set_len(len)?;
                Ok(())
            }
            Self::XsRolling(xs) => xs.set_len(len),
        }
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        match self {
            Self::Normal { k, v } => {
                k.try_set_len(len)?;
                v.try_set_len(len)?;
                Ok(())
            }
            Self::Rotating { k, v } => {
                k.try_set_len(len)?;
                v.try_set_len(len)?;
                Ok(())
            }
            Self::TurboQuant(_) => {
                // TurboQuant doesn't support try_set_len yet
                Ok(())
            }
            Self::XsRolling(xs) => xs.try_set_len(len),
        }
    }

    pub fn is_rotating(&self) -> bool {
        matches!(self, Self::Rotating { .. })
    }
}

#[derive(Debug, Clone)]
pub struct NormalCache(pub Vec<KvCache>);

#[derive(Debug)]
pub enum NormalCacheType {
    Normal { max_seq_len: usize },
    SlidingWindow { window: usize },
}

/// Global TurboQuant geometry for the **eager** (non-paged) KV cache.
///
/// `0` in either slot means "off". Packed as two atomics rather than a
/// `Mutex<Option<..>>` because [`NormalCache::new`] runs on the model-build
/// path for every layer of every model.
///
/// # Why this is a global at all
///
/// `NormalCache::new` is called from inside ~30 model constructors
/// (`models/llama.rs:453`, `models/qwen2.rs:443`, …) which take no cache
/// configuration. Threading a config through all of them is the right
/// long-term shape; the global is what makes the feature reachable without
/// touching every upstream model file, which the fork's merge policy
/// discourages.
static TURBOQUANT_HEAD_DIM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TURBOQUANT_V_HEAD_DIM: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Enable TurboQuant for all subsequently-created [`NormalCache`]s, with equal
/// K and V widths.
pub fn set_turboquant_head_dim(head_dim: usize) {
    set_turboquant_kv_head_dims(head_dim, head_dim);
}

/// Enable TurboQuant with independent K and V widths.
///
/// Set either to `0` to disable. Unlike the original gate this accepts **any**
/// width TurboQuant can serve — `{64, 128, 256}` was a codebook-table limit,
/// not a mathematical one (see `mistralrs_quant::turboquant::generate`), and
/// non-powers-of-two are handled by block decomposition
/// (`mistralrs_quant::turboquant::layout`).
///
/// The geometry is *not* validated here; [`NormalCache::new`] validates and
/// falls back to a plain cache with a warning, because it is the only place
/// that can still produce a working cache if the geometry is unsupported.
pub fn set_turboquant_kv_head_dims(k_head_dim: usize, v_head_dim: usize) {
    TURBOQUANT_HEAD_DIM.store(k_head_dim, std::sync::atomic::Ordering::SeqCst);
    TURBOQUANT_V_HEAD_DIM.store(v_head_dim, std::sync::atomic::Ordering::SeqCst);
}

/// Disable TurboQuant for subsequently-created [`NormalCache`]s.
pub fn clear_turboquant_head_dim() {
    set_turboquant_kv_head_dims(0, 0);
}

/// Why the eager KV cache did or did not end up TurboQuant-compressed.
///
/// Returned by [`resolve_eager_turboquant`] so the loader can log one honest
/// line instead of the caller guessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EagerTurboQuantDecision {
    /// Compress, with these `(k_head_dim, v_head_dim)`.
    Enabled(usize, usize),
    /// Leave the cache uncompressed, for this reason.
    Disabled(String),
}

/// Decide whether the **eager** KV cache should use TurboQuant for a model
/// with this geometry.
///
/// # Why this is opt-in and PagedAttention's TurboQuant is not
///
/// The paged path has a fused kernel (`turbo_paged_attention.cu`) that reads
/// packed blocks directly, so compression there is a straight memory win. The
/// eager path has no such kernel: `TurboQuantSingleCache::current_data`
/// reconstructs every compressed token on the host and ships it back to the
/// device once per layer per decode step. That is a decode-time cost no
/// measurement has yet justified, so enabling it is a deliberate act, set by
/// `ARC_TURBOQUANT_KV=1`.
///
/// This is the same discipline the V4 FP8 KV path settled on
/// (`ARC_V4_FP8_KV`, `models/deepseek4.rs:2405`) after a default-on KV change
/// shipped unmeasured.
///
/// `paged` short-circuits everything: with PagedAttention active the KV lives
/// in the paged cache and `NormalCache` is not the cache being used, so
/// setting the global would only risk compressing something else.
pub fn resolve_eager_turboquant(
    k_head_dim: usize,
    v_head_dim: usize,
    standard_layout: bool,
    paged: bool,
    env_value: Option<&str>,
) -> EagerTurboQuantDecision {
    use EagerTurboQuantDecision::*;
    let requested = match env_value.map(|s| s.trim().to_ascii_lowercase()) {
        Some(v) if matches!(v.as_str(), "1" | "true" | "on" | "yes") => true,
        Some(v) if matches!(v.as_str(), "0" | "false" | "off" | "no" | "") => {
            return Disabled("ARC_TURBOQUANT_KV is set to off".to_string())
        }
        Some(v) => {
            return Disabled(format!(
                "ARC_TURBOQUANT_KV={v:?} is not a recognised boolean; expected 1/0"
            ))
        }
        None => false,
    };
    if !requested {
        return Disabled(
            "not requested (set ARC_TURBOQUANT_KV=1 to compress the eager KV cache)".to_string(),
        );
    }
    if paged {
        return Disabled(
            "PagedAttention is active; KV compression there is selected with \
             --pa-cache-type, not ARC_TURBOQUANT_KV"
                .to_string(),
        );
    }
    if !standard_layout {
        return Disabled(
            "this model does not use the standard [B, H, T, D] KV layout; its K and V \
             halves are not independent head vectors"
                .to_string(),
        );
    }
    match mistralrs_quant::turboquant::TurboQuantConfig::try_new(k_head_dim, v_head_dim) {
        Ok(_) => Enabled(k_head_dim, v_head_dim),
        Err(e) => Disabled(e),
    }
}

/// Apply [`resolve_eager_turboquant`] to the process-wide gate, reading
/// `ARC_TURBOQUANT_KV` from the environment. Returns the decision so the
/// caller can log it.
///
/// Must be called **before** the model is constructed: model constructors call
/// [`NormalCache::new`], which reads the gate.
pub fn configure_eager_turboquant(
    k_head_dim: usize,
    v_head_dim: usize,
    standard_layout: bool,
    paged: bool,
) -> EagerTurboQuantDecision {
    let env = std::env::var("ARC_TURBOQUANT_KV").ok();
    let decision = resolve_eager_turboquant(
        k_head_dim,
        v_head_dim,
        standard_layout,
        paged,
        env.as_deref(),
    );
    match &decision {
        EagerTurboQuantDecision::Enabled(k, v) => set_turboquant_kv_head_dims(*k, *v),
        EagerTurboQuantDecision::Disabled(_) => clear_turboquant_head_dim(),
    }
    decision
}

/// Warn once per process that the eager cache is staying uncompressed.
///
/// Once, not per layer: this runs inside every model constructor, and 60
/// identical warnings would bury the one line that matters.
fn warn_turboquant_unsupported(reason: &str) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "TurboQuant KV was requested but this model's cache geometry cannot be \
             compressed, so the eager KV cache stays uncompressed: {reason}"
        );
    });
}

/// The currently-configured `(k_head_dim, v_head_dim)`, or `None` when off.
pub fn turboquant_head_dims() -> Option<(usize, usize)> {
    let k = TURBOQUANT_HEAD_DIM.load(std::sync::atomic::Ordering::SeqCst);
    let v = TURBOQUANT_V_HEAD_DIM.load(std::sync::atomic::Ordering::SeqCst);
    (k > 0 && v > 0).then_some((k, v))
}

/// Build the TurboQuant config for the ambient geometry, or explain why the
/// eager cache must stay uncompressed.
///
/// Returns `Ok(None)` when TurboQuant is simply off, `Err` when it is on but
/// the geometry cannot be served — which the caller turns into a warning plus
/// a plain cache, never into silent compression of something unsupported.
fn turboquant_config_from_globals(
) -> std::result::Result<Option<mistralrs_quant::turboquant::TurboQuantConfig>, String> {
    let Some((k, v)) = turboquant_head_dims() else {
        return Ok(None);
    };
    mistralrs_quant::turboquant::TurboQuantConfig::try_new(k, v).map(Some)
}

impl NormalCache {
    /// The number of tokens to grow the cache by
    pub const CACHE_GROW_SIZE: usize = 512;

    pub fn new(len: usize, max_seq_len: usize) -> Arc<Mutex<Self>> {
        match turboquant_config_from_globals() {
            Ok(Some(config)) => match KvCache::try_new_turboquant(&config) {
                Ok(proto) => return Arc::new(Mutex::new(Self(vec![proto; len]))),
                Err(e) => warn_turboquant_unsupported(&e),
            },
            Ok(None) => {}
            Err(e) => warn_turboquant_unsupported(&e),
        }
        Self::new_plain(len, max_seq_len)
    }

    /// A [`NormalCache`] of plain [`KvCache::Normal`] slots, ignoring the
    /// ambient TurboQuant setting.
    ///
    /// Models whose attention depends on the two halves of a slot being
    /// independent dense buffers — DeepSeek-V4's fused-MQA V marker is the
    /// live example, see `models/deepseek4.rs::require_normal_kv_slot` — must
    /// use this rather than [`Self::new`].
    pub fn new_plain(len: usize, max_seq_len: usize) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self(vec![
            KvCache::new_normal(
                2,
                max_seq_len,
                Self::CACHE_GROW_SIZE
            );
            len
        ])))
    }

    pub fn new_sliding(
        len: usize,
        max_seq_len: usize,
        sliding_window: Option<usize>,
    ) -> Arc<Mutex<Self>> {
        match sliding_window {
            Some(sliding_window) => Arc::new(Mutex::new(Self(vec![
                KvCache::new_rotating(
                    2,
                    sliding_window,
                    Self::CACHE_GROW_SIZE
                );
                len
            ]))),
            None => Arc::new(Mutex::new(Self(vec![
                KvCache::new_normal(
                    2,
                    max_seq_len,
                    Self::CACHE_GROW_SIZE
                );
                len
            ]))),
        }
    }

    pub fn from_types(types: Vec<NormalCacheType>) -> Arc<Mutex<Self>> {
        // Sliding-window layers stay `Rotating` regardless: `TurboQuantCache`
        // has no windowing, so compressing them would silently drop the
        // rotation. Only the full-attention layers of a mixed model are
        // eligible.
        let turbo = match turboquant_config_from_globals() {
            Ok(cfg) => cfg.and_then(|c| match KvCache::try_new_turboquant(&c) {
                Ok(proto) => Some(proto),
                Err(e) => {
                    warn_turboquant_unsupported(&e);
                    None
                }
            }),
            Err(e) => {
                warn_turboquant_unsupported(&e);
                None
            }
        };
        let mut caches = Vec::new();
        for ty in types {
            match ty {
                NormalCacheType::Normal { max_seq_len } => match &turbo {
                    Some(proto) => caches.push(proto.clone()),
                    None => caches.push(KvCache::new_normal(2, max_seq_len, Self::CACHE_GROW_SIZE)),
                },
                NormalCacheType::SlidingWindow { window } => {
                    caches.push(KvCache::new_rotating(2, window, Self::CACHE_GROW_SIZE));
                }
            }
        }
        Arc::new(Mutex::new(Self(caches)))
    }
}

pub struct NormalCacheManager;

/// Find the first cache slot whose `current_seq_len` disagrees across `seqs`.
///
/// Returns `Some((layer, len_of_seq0, len_of_seq_i, i))` for the first
/// disagreement found, or `None` when every populated slot agrees.
///
/// This is the invariant [`NormalCacheManager::clone_in_cache`] silently
/// assumes: it builds ONE dense batched cache and takes `seqs[0]` as the
/// template for `current_seq_len` / `capacity_seq_len`, and
/// [`SingleCache::append`] then writes every sequence's new K/V at that single
/// shared offset. Two sequences at different lengths therefore write to the
/// wrong slot and attend over the wrong window.
///
/// On a homogeneous K/V-only model it fails *silently* rather than loudly,
/// because `NormalCache::CACHE_GROW_SIZE` is 512: two sequences 100 tokens
/// apart still have identical `all_data` shapes, so the `slice_set` in
/// `clone_in_cache` succeeds and only `current_seq_len` differs.
///
/// 🔑 On DeepSeek V4 it does not fail silently — it *panics*, and that is the
/// wave51-CB serving crash. V4's cache vector is not homogeneous: 43 K/V
/// entries are followed by 41 [`KvCache::XsRolling`] entries, and the two
/// tensors `clone_in_cache` batches for those are
/// `comp.all_data` (`[B, comp.capacity_seq_len, head_dim]`, seq dim **1**) and
/// `tail` (`[B, tokens - base, hidden]`, seq dim **1**). `slice_set` along dim
/// 0 requires an exact match on every other dim, so:
///
/// * the `tail` width is `tokens - base`, an **exact, unquantised** function of
///   the token count — **one token of divergence is enough to panic**, at
///   `clone_in_cache`'s `batch_v.slice_set` line, and
/// * the `comp` capacity steps 64 → 576 → 1088 (init 64 rows, then
///   `CACHE_GROW_SIZE` blocks), so two sequences either side of one growth
///   boundary panic at the `batch_k.slice_set` line with `576 <> 64`.
///
/// The scheduler upholds a *weaker* invariant than that: it buckets on
/// [`crate::sequence::Sequence::cache_bucket_len`], which reads **cache slot 0
/// only** — for V4, the layer-0 K/V cache — and is blind to slots 43..83.
pub(crate) fn first_mismatched_cache_len(
    seqs: &mut [&mut crate::sequence::Sequence],
    modify_draft_cache: bool,
) -> Option<(usize, usize, usize, usize)> {
    first_mismatched_cache_len_inner(seqs, modify_draft_cache, xs_per_sequence_enabled())
}

/// [`first_mismatched_cache_len`] with the `xs` capability passed explicitly,
/// so the skip can be exercised without latching a process-wide env read.
///
/// `xs_per_seq` exempts [`KvCache::XsRolling`] slots: once those carry a token
/// count per batch row the whole reason for this check — that one dense
/// `slice_set` demands identical dims — no longer applies to them.
/// `clone_in_cache` reconciles their two buffers itself (front-padding the
/// end-anchored raw window, zero-extending the start-anchored compressed one)
/// and refuses, by name, any batch it genuinely cannot represent.
pub(crate) fn first_mismatched_cache_len_inner(
    seqs: &mut [&mut crate::sequence::Sequence],
    modify_draft_cache: bool,
    xs_per_seq: bool,
) -> Option<(usize, usize, usize, usize)> {
    if seqs.len() < 2 {
        return None;
    }
    // One allocation for the whole check, not one per sequence: this now runs
    // in release on every batched decode step.
    let template: Vec<Option<usize>> = {
        let cache = if modify_draft_cache {
            seqs[0].normal_draft_cache()
        } else {
            seqs[0].normal_cache()
        };
        cache
            .iter()
            .map(|slot| match slot.as_ref() {
                Some(KvCache::XsRolling(_)) if xs_per_seq => None,
                other => other.map(KvCache::current_seq_len),
            })
            .collect()
    };

    for (i, seq) in seqs.iter_mut().enumerate().skip(1) {
        let cache = if modify_draft_cache {
            seq.normal_draft_cache()
        } else {
            seq.normal_cache()
        };
        for (layer, expected) in template.iter().enumerate() {
            let (Some(expected), Some(got)) = (
                *expected,
                cache
                    .get(layer)
                    .and_then(|s| s.as_ref())
                    .map(KvCache::current_seq_len),
            ) else {
                continue;
            };
            if expected != got {
                return Some((layer, expected, got, i));
            }
        }
    }
    None
}

/// Which of a cache slot's two batched tensors is *preallocation slack* and
/// which is *content*.
///
/// `clone_in_cache` builds one dense tensor per side and `slice_set`s each
/// sequence in along dim 0, which demands an exact match on every other dim.
/// Two different things can make those dims disagree, and they need opposite
/// treatment:
///
/// * **Slack** — `SingleCache::all_data` is `capacity_seq_len` wide along its
///   own `dim`, grown in `CACHE_GROW_SIZE` blocks and *never shrunk*. Two
///   sequences at the identical length can still hold different capacities
///   (capacity tracks the sequence's peak, so any rollback — MTP verify, a
///   prefix-cache truncation — leaves it high). The extra width holds nothing:
///   `current_seq_len` is what says how much is real. Zero-padding the short
///   one up to the batch maximum is therefore exact, and it is what makes the
///   K/V and `comp` halves as slack-tolerant as they were always assumed to be.
/// * **Content** — `XsRollingCache::tail` is `tokens - base` wide and every
///   column is live compressor input; `TurboQuantCache`'s `current_data` is
///   narrowed to the current length. A disagreement there is a genuinely ragged
///   batch. Padding it would fabricate history, so it is refused.
struct BatchSrc {
    k: Tensor,
    v: Tensor,
    /// Dim along which `k` is slack, when it is.
    k_slack_dim: Option<usize>,
    /// Dim along which `v` is slack, when it is.
    v_slack_dim: Option<usize>,
    /// `v`'s content is anchored at its END, so widening it to the batch
    /// maximum has to pad at the FRONT — the same left-alignment
    /// [`front_pad_kv_cache`] applies to a ragged K/V cohort, here applied to
    /// [`XsRollingCache::tail`]. Meaningless unless `v_slack_dim` is set.
    v_slack_at_front: bool,
    /// `(tokens, base)` per batch row, for an [`KvCache::XsRolling`] slot.
    /// Carried alongside the tensors because the batched cache's row lengths
    /// are the concatenation of its sequences', not `seqs[0]`'s repeated.
    xs_rows: Option<(Vec<usize>, Vec<usize>)>,
}

impl BatchSrc {
    fn of(cache: &KvCache, xs_per_seq: bool) -> Result<Self> {
        Ok(match cache {
            KvCache::Normal { k, v } => Self {
                k: k.all_data.clone().ok_or_else(|| {
                    candle_core::Error::msg("kv-cache: normal K half not materialised")
                })?,
                v: v.all_data.clone().ok_or_else(|| {
                    candle_core::Error::msg("kv-cache: normal V half not materialised")
                })?,
                k_slack_dim: Some(k.dim),
                v_slack_dim: Some(v.dim),
                v_slack_at_front: false,
                xs_rows: None,
            },
            KvCache::Rotating { k, v } => Self {
                k: k.all_data.clone().ok_or_else(|| {
                    candle_core::Error::msg("kv-cache: rotating K half not materialised")
                })?,
                v: v.all_data.clone().ok_or_else(|| {
                    candle_core::Error::msg("kv-cache: rotating V half not materialised")
                })?,
                k_slack_dim: Some(k.dim),
                v_slack_dim: Some(v.dim),
                v_slack_at_front: false,
                xs_rows: None,
            },
            KvCache::TurboQuant(tq) => Self {
                k: tq.k.current_data()?.ok_or_else(|| {
                    candle_core::Error::msg("kv-cache: turboquant K half not materialised")
                })?,
                v: tq.v.current_data()?.ok_or_else(|| {
                    candle_core::Error::msg("kv-cache: turboquant V half not materialised")
                })?,
                k_slack_dim: None,
                v_slack_dim: None,
                v_slack_at_front: false,
                xs_rows: None,
            },
            // Compressed rows batch like K (a grown capacity buffer, and
            // start-anchored: column `j` is absolute block `j` for every row).
            // The raw tail is live content — with `ARC_V4_XS_PER_SEQ` off its
            // width is an exact function of the token count and any
            // disagreement is a genuinely ragged batch that must be refused;
            // with it on the tail is end-anchored, so a narrower row is
            // front-padded up to the batch maximum and the tokens it gains are
            // ahead of its own `base`, i.e. never read. Both buffers are
            // materialised by `XsRollingCache::advance`, so a sequence that has
            // been cloned out at least once always has them.
            KvCache::XsRolling(xs) => {
                let (tokens, base) = xs.row_lens();
                Self {
                    k: xs.comp.all_data.clone().ok_or_else(|| {
                        candle_core::Error::msg(
                            "xs rolling cache: compressed rows not materialised",
                        )
                    })?,
                    v: xs.tail.clone().ok_or_else(|| {
                        candle_core::Error::msg("xs rolling cache: raw tail not materialised")
                    })?,
                    k_slack_dim: Some(xs.comp.dim),
                    v_slack_dim: if xs_per_seq { Some(1) } else { None },
                    v_slack_at_front: true,
                    xs_rows: Some((tokens.to_vec(), base.to_vec())),
                }
            }
        })
    }
}

/// Shift one `SingleCache`'s live run right so it **ends** at `target_len`,
/// zero-filling the `target_len - current_seq_len` columns ahead of it.
/// Returns the width of that dead prefix (`lead_pad`).
///
/// This is the whole mechanism by which a dense batched cache can carry
/// per-sequence lengths. `SingleCache::append` writes at one shared offset
/// (`single_cache.rs:225`), so a ragged cohort can only share a forward if
/// every row's live content ends at the same column — then that one offset is
/// simultaneously correct for all of them. Left-alignment buys that; the price
/// is `lead_pad` dead columns per row, which the caller MUST mask
/// ([`crate::layers_masker::RaggedKvLens`]) because a zero K row is not a
/// masked row — it scores logit 0 and takes softmax weight.
///
/// The dead prefix does **not** grow without bound: `lead_pad_i` is
/// `max_j L_j - L_i`, and sequences in one batch accept at statistically the
/// same rate, so the spread is a `sqrt(steps)` random walk, not a linear one.
///
/// ⚠️ That last sentence is only true if the prefix is **taken back out** on
/// the way to the per-sequence caches. `current_seq_len` here counts the dead
/// columns as live (it has to — this is where the shared append offset comes
/// from), so a caller that records this length as the sequence's own makes the
/// next `front_align_batch` pad relative to an already-padded length and the
/// prefix accumulates **linearly**. [`drop_dead_prefix`] is the inverse that
/// keeps the claim honest, and
/// `MtpSpeculativePipeline::step`'s per-sequence commit is the caller that
/// applies it. `the_dead_prefix_does_not_accumulate_across_steps` pins the
/// difference, with the un-stripped variant as its negative control.
fn front_pad_single(sc: &mut SingleCache, target_len: usize) -> Result<usize> {
    let live = sc.current_seq_len;
    if target_len < live {
        candle_core::bail!(
            "kv-cache: front_pad to {target_len} is shorter than the {live} live positions it \
             would have to keep"
        );
    }
    let lead = target_len - live;
    if lead == 0 {
        sc.current_seq_len = target_len;
        return Ok(0);
    }
    let Some(ad) = sc.all_data.as_ref() else {
        candle_core::bail!("kv-cache: front_pad on a slot whose buffer is not materialised");
    };
    let capacity = sc.capacity_seq_len.max(target_len);
    if capacity > sc.max_seq_len {
        candle_core::bail!(
            "kv-cache: front_pad to {target_len} needs capacity {capacity}, past this slot's \
             {} maximum",
            sc.max_seq_len
        );
    }
    let mut shape = ad.dims().to_vec();
    shape[sc.dim] = capacity;
    let grown = Tensor::zeros(shape, ad.dtype(), ad.device())?;
    if live > 0 {
        let src = ad.narrow(sc.dim, 0, live)?.contiguous()?;
        grown.slice_set(&src, sc.dim, lead)?;
    }
    sc.all_data = Some(grown);
    sc.capacity_seq_len = capacity;
    sc.current_seq_len = target_len;
    Ok(lead)
}

/// [`front_pad_single`] over both halves of a slot. Refuses any variant that
/// cannot carry its own length ([`KvCache::supports_per_sequence_len`]).
pub(crate) fn front_pad_kv_cache(cache: &mut KvCache, target_len: usize) -> Result<usize> {
    // 🔑 An `XsRolling` slot is *already* per-row and must NOT be flattened to
    // one length here. Its compressed rows are start-anchored (column `j` is
    // absolute block `j` for every row, so there is nothing to shift) and its
    // raw window is end-anchored and reconciled at batch-assembly time by
    // `clone_in_cache`. Writing `target_len` into it would destroy exactly the
    // per-row token counts this whole path exists to preserve.
    if matches!(cache, KvCache::XsRolling(_)) {
        if !xs_per_sequence_enabled() {
            candle_core::bail!(
                "kv-cache: front_pad is only defined for a `Normal` slot; this one is an \
                 `XsRolling` and `ARC_V4_XS_PER_SEQ` is off. See \
                 `KvCache::supports_per_sequence_len`."
            );
        }
        return Ok(0);
    }
    let KvCache::Normal { k, v } = cache else {
        candle_core::bail!(
            "kv-cache: front_pad is only defined for a `Normal` slot; this one is a `{}`. See \
             `KvCache::supports_per_sequence_len`.",
            cache.kind_name()
        );
    };
    let lead_k = front_pad_single(k, target_len)?;
    let lead_v = front_pad_single(v, target_len)?;
    debug_assert_eq!(
        lead_k, lead_v,
        "the two halves of one slot describe the same positions"
    );
    Ok(lead_k)
}

/// Drop the `lead` dead columns [`front_pad_single`] put ahead of one
/// `SingleCache`'s live run, so the run is left-anchored at column 0 again and
/// `current_seq_len` counts only positions that hold real K/V.
fn drop_front_single(sc: &mut SingleCache, lead: usize) -> Result<()> {
    if lead == 0 {
        return Ok(());
    }
    if lead > sc.current_seq_len {
        candle_core::bail!(
            "kv-cache: cannot drop a {lead}-column dead prefix from a slot that holds only {} \
             position(s) — the caller's idea of this row's padding and the slot's own length \
             disagree",
            sc.current_seq_len
        );
    }
    let Some(ad) = sc.all_data.as_ref() else {
        candle_core::bail!("kv-cache: drop_dead_prefix on a slot whose buffer is not materialised");
    };
    let have = ad.dims()[sc.dim];
    if have < sc.current_seq_len {
        candle_core::bail!(
            "kv-cache: slot claims {} live position(s) but its buffer is only {have} wide",
            sc.current_seq_len
        );
    }
    let kept = have - lead;
    sc.all_data = Some(ad.narrow(sc.dim, lead, kept)?.contiguous()?);
    sc.capacity_seq_len = kept;
    sc.current_seq_len -= lead;
    Ok(())
}

/// The inverse of [`front_pad_kv_cache`]: take a row's dead prefix back out
/// once the dense batched forward that needed it is over.
///
/// 🔑 This is what stops the prefix from accumulating. Left-alignment is the
/// only way a ragged cohort can share one append offset, but the length it
/// leaves behind (`lead + live`) is not the sequence's own length. Recording
/// *that* as the per-sequence length makes the next `front_align_batch` pad
/// relative to it, so every step adds another `max_j c_j - c_i` columns and the
/// buffer grows linearly in steps rather than tracking the `sqrt(steps)` spread
/// of the sequences themselves. Stripping the prefix restores the invariant
/// every other part of the cache assumes — that a slot's live run starts at
/// column 0 — which is exactly what `front_pad_single`'s `narrow(dim, 0, live)`
/// requires the next time round.
///
/// `lead == 0` (every B=1 request, every uniform batch) is a no-op that touches
/// no tensor. An `XsRolling` slot is likewise untouched: `front_pad_kv_cache`
/// never gave it a prefix — its compressed rows are start-anchored and its raw
/// window is re-anchored per row by `XsRollingCache::split_row`.
pub(crate) fn drop_dead_prefix(cache: &mut KvCache, lead: usize) -> Result<()> {
    if matches!(cache, KvCache::XsRolling(_)) || lead == 0 {
        return Ok(());
    }
    let KvCache::Normal { k, v } = cache else {
        candle_core::bail!(
            "kv-cache: drop_dead_prefix is only defined for a `Normal` slot; this one is a `{}`. \
             Nothing front-pads it, so nothing should be stripping it either. See \
             `KvCache::supports_per_sequence_len`.",
            cache.kind_name()
        );
    };
    drop_front_single(k, lead)?;
    drop_front_single(v, lead)?;
    Ok(())
}

/// Left-align a whole ragged cohort so every sequence's live K/V ends at the
/// batch maximum, and report each sequence's dead prefix.
///
/// After this call every sequence reports the same `current_seq_len`, so
/// [`ensure_uniform_batch_cache_lens`] passes and `NormalCacheManager` stacks
/// them unchanged — the dense batching code needs no modification at all.
/// What the caller gains is that the length it hands back on the way *out*
/// (`clone_out_cache` → `KvCache::set_len`) may differ per sequence.
///
/// Returns `lead_pad[i]` per sequence, which is `padded_len - live_len_i`, and
/// which the model's attention MUST mask.
pub(crate) fn front_align_batch(
    seqs: &mut [&mut crate::sequence::Sequence],
    modify_draft_cache: bool,
) -> Result<Vec<usize>> {
    let mut target = 0usize;
    for seq in seqs.iter_mut() {
        let cache = if modify_draft_cache {
            seq.normal_draft_cache()
        } else {
            seq.normal_cache()
        };
        for slot in cache.iter().flatten() {
            target = target.max(slot.current_seq_len());
        }
    }
    let mut lead_pads = Vec::with_capacity(seqs.len());
    for seq in seqs.iter_mut() {
        let cache = if modify_draft_cache {
            seq.normal_draft_cache()
        } else {
            seq.normal_cache()
        };
        // The dead prefix is a property of the K/V run, so it is read from a
        // K/V slot. `XsRolling` slots front-pad to nothing (they are already
        // per-row) and would otherwise overwrite a real `lead` with 0 — on
        // DeepSeek V4, whose 41 compressor slots come LAST, that silently
        // reported "no padding" for every sequence.
        let mut lead = 0usize;
        for slot in cache.iter_mut().flatten() {
            let is_xs = matches!(slot, KvCache::XsRolling(_));
            let got = front_pad_kv_cache(slot, target)?;
            if !is_xs {
                lead = got;
            }
        }
        lead_pads.push(lead);
    }
    Ok(lead_pads)
}

thread_local! {
    /// `lead_pad[i]` — the dead, zero-filled columns at the FRONT of row `i` of
    /// the batched dense cache, left there by [`front_align_batch`].
    ///
    /// This is the channel that lets `CausalMasker` learn a batch is ragged
    /// **without threading an argument through all forty-odd model forwards**.
    /// Same shape as the existing `layers::set_graph_mode_positions`.
    ///
    /// 🔑 It holds the dead prefix and NOT the live lengths, because the prefix
    /// is the part that stays constant. `clone_in_cache` runs only when batch
    /// membership CHANGES (`engine/mod.rs`: `pre_op` is `In` only when
    /// `last_completion_ids != current_completion_ids`, `Nothing` otherwise),
    /// while the batched cache keeps growing by one column per row per step.
    /// So a stored `live` would go stale on the very next token, whereas
    /// `lead_pad` stays true for as long as the cohort is intact — and the
    /// masker recovers `live[i] = past_kv_len - lead_pad[i]` from the cache's
    /// own current length.
    ///
    /// `None` — the overwhelmingly common case — means the batch is uniform and
    /// every path behaves exactly as it always has.
    static RAGGED_LEAD_PAD: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
}

/// Whether THIS pipeline's cache can carry per-sequence lengths, decided once
/// after the model is loaded (`engine/mod.rs`) and read by the scheduler.
///
/// Default **off**: a build that never calls the setter behaves exactly as it
/// always has. The scheduler cannot work this out for itself — it never sees
/// the model — and the answer is a property of the cache variants
/// ([`KvCache::supports_per_sequence_len`]), so it is decided where both are
/// visible and published here. Same shape as `xs_per_sequence_enabled`.
static RAGGED_DECODE_SUPPORTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_ragged_decode_supported(on: bool) {
    RAGGED_DECODE_SUPPORTED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// May the scheduler admit a decode batch whose sequences differ in length?
pub fn ragged_decode_supported() -> bool {
    #[cfg(test)]
    if let Some(on) = ragged_decode_test_override::current() {
        return on;
    }
    RAGGED_DECODE_SUPPORTED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Test-only override for [`ragged_decode_supported`].
///
/// The production flag is process-global because there is one pipeline per
/// process. `cargo test` is the exception: it runs tests in parallel threads in
/// ONE process, so a test that flips the global changes what every concurrently
/// running test sees. That is not hypothetical — it made the pre-existing
/// `scheduler_runs_the_whole_admitted_batch` fail while passing under
/// `--test-threads=1`, which is exactly the shape of a bug that looks like a
/// real regression and is not.
///
/// Thread-local, mirroring `xs_rolling::test_override`.
#[cfg(test)]
pub(crate) mod ragged_decode_test_override {
    use std::cell::Cell;

    thread_local! {
        static STATE: Cell<Option<bool>> = const { Cell::new(None) };
    }

    pub(crate) fn current() -> Option<bool> {
        STATE.with(|s| s.get())
    }

    /// Run `f` with ragged decode forced to `on`, on this thread only.
    pub(crate) fn with<R>(on: bool, f: impl FnOnce() -> R) -> R {
        let prev = STATE.with(|s| s.replace(Some(on)));
        let out = f();
        STATE.with(|s| s.set(prev));
        out
    }
}

pub(crate) fn set_ragged_lead_pad(lead_pad: Option<Vec<usize>>) {
    RAGGED_LEAD_PAD.with(|r| *r.borrow_mut() = lead_pad);
}

thread_local! {
    /// `seq id -> the dead prefix its per-sequence cache still carries`.
    ///
    /// 🔑 This is what makes the strip LAZY. `clone_out_cache` runs on every
    /// decode token, but the per-sequence caches it writes are only ever READ
    /// again when batch membership changes (`engine/mod.rs`: `pre_op` is `In`
    /// only when `last_completion_ids != current_completion_ids`) — in between,
    /// the forward runs off the batched cache the pipeline still holds. So
    /// dropping each row's dead prefix on every step was doing
    /// `O(B x layers x 2)` device copies per token for data nobody looks at
    /// (~2,200 copies/token at B=47).
    ///
    /// Instead `clone_out_cache` records the prefix here — no tensor work — and
    /// `clone_in_cache` pays for it once, on the membership change that
    /// actually re-reads those caches.
    static PENDING_LEAD_PAD: std::cell::RefCell<std::collections::HashMap<usize, usize>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

fn record_pending_lead_pad(seq_id: usize, lead: usize) {
    PENDING_LEAD_PAD.with(|m| {
        let mut m = m.borrow_mut();
        if lead == 0 {
            m.remove(&seq_id);
        } else {
            m.insert(seq_id, lead);
        }
    });
}

fn take_pending_lead_pad(seq_id: usize) -> usize {
    PENDING_LEAD_PAD.with(|m| m.borrow_mut().remove(&seq_id).unwrap_or(0))
}

/// Test-only entry points for the deferred-strip bookkeeping, so tests outside
/// this module can build a sequence in the state a front-aligned cohort leaves
/// behind without making the real setters public.
#[cfg(test)]
pub(crate) mod test_support {
    pub(crate) fn set_pending_lead_pad(seq_id: usize, lead: usize) {
        super::record_pending_lead_pad(seq_id, lead);
    }
}

/// Drop a stale dead prefix from one sequence's own cache, re-anchoring its live
/// run at column 0 the way `SingleCache` requires.
///
/// Called from `clone_in_cache` for each sequence that carries one, i.e. once
/// per membership change rather than once per token.
pub(crate) fn strip_pending_lead_pad(
    seq: &mut crate::sequence::Sequence,
    modify_draft_cache: bool,
) -> Result<()> {
    let seq_id = *seq.id();
    let lead = take_pending_lead_pad(seq_id);
    if lead == 0 {
        return Ok(());
    }
    let cache = if modify_draft_cache {
        seq.normal_draft_cache()
    } else {
        seq.normal_cache()
    };
    for slot in cache.iter_mut().flatten() {
        if let KvCache::Normal { k, v } = slot {
            for sc in [k, v] {
                let Some(ad) = sc.all_data.as_ref() else {
                    continue;
                };
                let keep = sc.current_seq_len.saturating_sub(lead);
                let kept = ad.narrow(sc.dim, lead, keep)?.contiguous()?;
                sc.all_data = Some(kept);
                sc.current_seq_len = keep;
                sc.capacity_seq_len = keep;
            }
        }
    }
    Ok(())
}

/// The current cohort's per-row dead prefix, if it is ragged.
pub(crate) fn ragged_lead_pad() -> Option<Vec<usize>> {
    RAGGED_LEAD_PAD.with(|r| r.borrow().clone())
}

/// Can this batch be front-aligned instead of refused?
///
/// Every slot of every sequence has to be able to carry its own length. A
/// `Rotating` or `TurboQuant` slot cannot ([`KvCache::supports_per_sequence_len`]),
/// and `XsRolling` only when `ARC_V4_XS_PER_SEQ` is on — so a model carrying one
/// keeps the old exact-length bucketing rather than getting a wrong answer.
pub(crate) fn batch_can_be_ragged(
    seqs: &mut [&mut crate::sequence::Sequence],
    modify_draft_cache: bool,
) -> bool {
    seqs.iter_mut().all(|seq| {
        let cache = if modify_draft_cache {
            seq.normal_draft_cache()
        } else {
            seq.normal_cache()
        };
        cache
            .iter()
            .flatten()
            .all(KvCache::supports_per_sequence_len)
    })
}

/// Would [`front_align_batch`] succeed on every slot, checked **before any of
/// them is mutated**?
///
/// 🔑 This exists because `front_align_batch` pads in place, sequence by
/// sequence and slot by slot. If it bailed halfway — an unmaterialised buffer,
/// or a row whose padded capacity would exceed its `max_seq_len` — it would
/// return `Err` having already rewritten the rows ahead of the failure. The
/// blanket refusal it replaced (`ensure_uniform_batch_cache_lens` first, ask
/// questions never) mutated **nothing** on the way out, and that property is
/// part of what makes a refusal safe: the engine can fail those requests
/// without the surviving sequences carrying a half-aligned cache.
///
/// So alignment is only attempted when it is known to complete. When this
/// returns `false` the batch falls through to the original refusal, unmutated,
/// exactly as before.
fn front_align_would_succeed(
    seqs: &mut [&mut crate::sequence::Sequence],
    modify_draft_cache: bool,
) -> bool {
    let mut target = 0usize;
    for seq in seqs.iter_mut() {
        let cache = if modify_draft_cache {
            seq.normal_draft_cache()
        } else {
            seq.normal_cache()
        };
        for slot in cache.iter().flatten() {
            target = target.max(slot.current_seq_len());
        }
    }
    seqs.iter_mut().all(|seq| {
        let cache = if modify_draft_cache {
            seq.normal_draft_cache()
        } else {
            seq.normal_cache()
        };
        cache.iter().flatten().all(|slot| match slot {
            // Already per-row; `front_pad_kv_cache` returns Ok(0) untouched.
            KvCache::XsRolling(_) => xs_per_sequence_enabled(),
            KvCache::Normal { k, v } => [k, v].into_iter().all(|sc| {
                if sc.current_seq_len == target {
                    return true; // lead == 0, nothing to move
                }
                sc.all_data.is_some() && sc.capacity_seq_len.max(target) <= sc.max_seq_len
            }),
            // `supports_per_sequence_len` is false for these, so
            // `batch_can_be_ragged` already refused; belt and braces.
            KvCache::Rotating { .. } | KvCache::TurboQuant(_) => false,
        })
    })
}

/// Zero-extend `src` along `dim` to `width`, at the front when the content is
/// end-anchored. Used only for preallocation slack and for the end-anchored
/// `xs` window (see [`BatchSrc`]), so the added columns are never read.
fn pad_slack(src: &Tensor, dim: usize, width: usize, at_front: bool) -> Result<Tensor> {
    let have = src.dims()[dim];
    if have == width {
        return Ok(src.clone());
    }
    if have > width {
        candle_core::bail!("kv-cache: cannot pad a {have}-wide slot down to {width} on dim {dim}");
    }
    let mut shape = src.dims().to_vec();
    shape[dim] = width;
    let grown = Tensor::zeros(shape, src.dtype(), src.device())?;
    let offset = if at_front { width - have } else { 0 };
    grown.slice_set(&src.contiguous()?, dim, offset)?;
    Ok(grown)
}

/// Reconcile one side of one layer across the batch.
///
/// Returns the dims every sequence's tensor must have once slack is padded, or
/// an error naming the sequence and the dim that is genuinely incompatible —
/// which is the diagnostic the raw candle `shape mismatch on dim 1, 18 <> 22`
/// never gave.
fn reconcile_batch_dims(
    per_seq: &[&Tensor],
    slack_dim: Option<usize>,
    side: &str,
    layer: usize,
) -> Result<Vec<usize>> {
    let mut dims = per_seq[0].dims().to_vec();
    if let Some(d) = slack_dim {
        for t in per_seq {
            dims[d] = dims[d].max(t.dims()[d]);
        }
    }
    for (i, t) in per_seq.iter().enumerate() {
        if t.rank() != dims.len() {
            candle_core::bail!(
                "kv-cache: cannot batch cache slot {layer} ({side} half): seqs[0] is rank {} \
                 {:?}, seqs[{i}] is rank {} {:?}",
                dims.len(),
                dims,
                t.rank(),
                t.dims()
            );
        }
        for (d, (want, got)) in dims.iter().zip(t.dims()).enumerate() {
            if d == 0 || Some(d) == slack_dim {
                continue;
            }
            if want != got {
                candle_core::bail!(
                    "kv-cache: cannot batch cache slot {layer} ({side} half) — the batch is \
                     ragged on dim {d}: seqs[0] is {want} wide, seqs[{i}] is {got}. This dim \
                     is the sequence's live state, not preallocation slack, so the two \
                     sequences are at genuinely different points and cannot share one dense \
                     cache. (On DeepSeek V4 the `xs` rolling tail is `tokens - base` wide, so \
                     this means the batch's token counts diverged — see \
                     `first_mismatched_cache_len`.)"
                );
            }
        }
        if dims[0] != t.dims()[0] {
            candle_core::bail!(
                "kv-cache: cannot batch cache slot {layer} ({side} half): seqs[0] has batch \
                 dim {}, seqs[{i}] has {}",
                dims[0],
                t.dims()[0]
            );
        }
    }
    Ok(dims)
}

/// The contract [`NormalCacheManager::clone_in_cache`] needs from a batch,
/// checked **in release**.
///
/// wave51-CB: this was a `debug_assert!`, so the H200 binary that served the
/// `qtip2b` artifact carried no check at all. What it got instead was
/// `slice_set`'s `shape mismatch on dim 1, …` — a panic, on the engine task,
/// which took every other in-flight request down with it.
pub(crate) fn ensure_uniform_batch_cache_lens(
    seqs: &mut [&mut crate::sequence::Sequence],
    modify_draft_cache: bool,
) -> Result<()> {
    match first_mismatched_cache_len(seqs, modify_draft_cache) {
        None => Ok(()),
        Some((layer, expected, got, i)) => candle_core::bail!(
            "kv-cache: sequences in one batch must share current_seq_len — seqs[0] is the \
             template for the whole dense batched cache, and every sequence's new K/V is \
             written at that one offset. Cache slot {layer}: seqs[0] holds {expected} \
             position(s), seqs[{i}] holds {got}. The scheduler buckets on \
             `Sequence::cache_bucket_len`, which reads cache slot 0 only, so a model whose \
             cache vector carries non-K/V slots (DeepSeek V4's 41 `XsRolling` compressor \
             histories) can reach here with slot 0 in agreement and a later slot not."
        ),
    }
}

/// Reconcile the one quantity [`ensure_uniform_batch_cache_lens`] cannot see:
/// [`XsRollingCache::base`].
///
/// Uniform `current_seq_len` is **not** sufficient for `tail` (which is
/// `tokens - base` wide) to batch, because `base` tracks the sequence's
/// *high-water* length rather than its current one: `set_len` narrows the tail
/// without moving `base`, and `advance` never lowers it. Two sequences at an
/// identical token count can therefore hold tails of different width — see
/// [`XsRollingCache::trim_tail_to`] for the ways that happens.
///
/// This needs no speculative decoding to reach. `prefix_cacher.rs` calls
/// `set_len` on every stored layer, so a sequence restored from a prefix-cache
/// entry that was stored at a greater length holds a narrower tail than one that
/// arrived directly — and `clone_in_cache` then batches the two with
/// `slice_set`, which demands an exact dim match. That is a second, independent
/// route into the same `shape mismatch on dim 1, …` failure that
/// `ensure_uniform_batch_cache_lens` was added to close, and it is on the
/// **plain decode path**.
///
/// Raising every member to the batch maximum is lossless by construction: the
/// member already sitting at `base_max` proves no future compressed row needs a
/// token below it. `trim_tail_to` refuses rather than silently dropping history
/// if that were ever untrue.
///
/// # 🔴 Why this is gated on `xs_per_seq`, and why that is not a port
///
/// `base` means two different things either side of `ARC_V4_XS_PER_SEQ`, and
/// the reconciliation is *required* on one path and a *regression* on the other.
///
/// **Flag off** — `tail` is `tokens - base` wide, so `base` IS the physical left
/// edge of the buffer. `BatchSrc::of` sets `v_slack_dim: None`, meaning
/// `reconcile_batch_dims` demands the widths match exactly. Divergent `base` at
/// equal `tokens` is then the 4-vs-132 `slice_set` failure PR #93 exists to fix,
/// it is reachable on ordinary decode through the prefix cacher, and this
/// function runs exactly as it always has.
///
/// **Flag on** — `tail` is `[B, W, hidden]` and **end-anchored**, with
/// `base[i] >= tokens[i] - W`. `base` is now a *logical resume point*, decoupled
/// from the buffer: `BatchSrc::of` sets `v_slack_dim: Some(1)` with
/// `v_slack_at_front: true`, so a narrower row is front-padded up to the batch
/// maximum and the columns it gains sit ahead of its own `base` — never read.
/// The widths already agree, so the reconciliation buys nothing.
///
/// And it is not merely redundant there, it is **harmful**. Trimming every row
/// to `base_max` raises the shortest-reach row's rollback floor to the batch's,
/// which turns a per-sequence quantity into a batch-wide scalar. That is the
/// same shape as the cohort min-rollback PR #92 removed, and as the
/// dead-prefix-counted-as-content trap #102, #103 and #104 each caught
/// independently. This is the last layer that still holds one; running it under
/// the flag would put it back.
///
/// `xs_per_seq` is a parameter rather than a read of
/// [`xs_per_sequence_enabled`] inside so that "which path reconciles" is itself
/// testable — the two branches agree on every batch either one can serve, so no
/// numeric test could tell which ran.
fn reconcile_xs_bases(
    seqs: &mut [&mut crate::sequence::Sequence],
    layer: usize,
    modify_draft_cache: bool,
    xs_per_seq: bool,
) -> Result<()> {
    if xs_per_seq {
        return Ok(());
    }
    let mut base_max = 0usize;
    let mut any = false;
    for seq in seqs.iter_mut() {
        let cache = if modify_draft_cache {
            seq.normal_draft_cache()
        } else {
            seq.normal_cache()
        };
        if let Some(KvCache::XsRolling(xs)) = cache.get(layer).and_then(|s| s.as_ref()) {
            // `resumable_from()` is `base.iter().max()`, which for the
            // single-row per-sequence caches this function sees is `base[0]` —
            // the same value the pre-#95 field read gave.
            base_max = base_max.max(xs.resumable_from());
            any = true;
        }
    }
    if !any || base_max == 0 {
        return Ok(());
    }
    for (i, seq) in seqs.iter_mut().enumerate() {
        let cache = if modify_draft_cache {
            seq.normal_draft_cache()
        } else {
            seq.normal_cache()
        };
        if let Some(KvCache::XsRolling(xs)) = cache.get_mut(layer).and_then(|s| s.as_mut()) {
            xs.trim_tail_to(base_max).map_err(|e| {
                candle_core::Error::msg(format!(
                    "kv-cache: cannot reconcile cache slot {layer} of seqs[{i}] to the batch's \
                     retained-window start {base_max}: {e}"
                ))
            })?;
        }
    }
    Ok(())
}

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for NormalCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) -> Result<()> {
        // A ragged cohort CAN share one dense cache, as long as every row's
        // live K/V is made to END at the same column — then
        // `SingleCache::append`'s one shared write offset is simultaneously
        // correct for all of them. `front_align_batch` buys that; the price is
        // a dead prefix per row, which attention MUST mask, because a zero K
        // row is not a masked row (it scores logit 0 and takes softmax weight).
        // The prefix is published on `RAGGED_LEAD_PAD` and picked up by
        // `CausalMasker::make_causal_mask_matrix`.
        //
        // Refused rather than aligned when any slot cannot carry its own length
        // (`Rotating`, `TurboQuant`, or `XsRolling` with `ARC_V4_XS_PER_SEQ`
        // off) — those models keep the old exact-length bucketing.
        // Publish the capability from the one place that always holds the real
        // cache. Deciding this at engine construction would be a guess: the
        // scheduler never sees the model, and the cache slots are not
        // necessarily allocated yet. Deciding it here cannot be wrong — the
        // cost is that the very first cohort is still bucketed, after which the
        // scheduler has the answer. PagedAttention never reaches this function,
        // so its scheduler is unaffected.
        // Two checks, deliberately overlapping, with DIFFERENT jobs — noted
        // because a mutation test showed that deleting this one alone changes
        // no behaviour, `front_align_would_succeed` having independently
        // refused the same batches.
        //   * this one answers "can this MODEL ever take ragged decode", and is
        //     what the scheduler reads to stop bucketing;
        //   * `front_align_would_succeed` answers "can THIS batch be aligned
        //     right now, without mutating anything if the answer is no".
        // The capability is pinned directly by
        // `the_capability_predicate_is_what_the_scheduler_reads`.
        // Pay the deferred strip now, once, on the membership change that is
        // about to re-read these caches. Until this runs, a row that was part of
        // a front-aligned cohort still carries its dead prefix and reports a
        // length that includes it — so this must happen BEFORE any length is
        // compared or any alignment decided.
        for seq in seqs.iter_mut() {
            strip_pending_lead_pad(seq, modify_draft_cache)?;
        }

        let can_be_ragged = batch_can_be_ragged(seqs, modify_draft_cache);
        set_ragged_decode_supported(can_be_ragged);

        // Alignment is attempted only when it is known to complete, so the
        // refusal path below stays NON-MUTATING exactly as it was before this
        // change — see `front_align_would_succeed`. There is therefore no exit
        // between here and `slice_set` that leaves a half-aligned batch.
        if first_mismatched_cache_len(seqs, modify_draft_cache).is_some()
            && can_be_ragged
            && front_align_would_succeed(seqs, modify_draft_cache)
        {
            let lead_pad = front_align_batch(seqs, modify_draft_cache)?;
            set_ragged_lead_pad(Some(lead_pad));
        } else {
            set_ragged_lead_pad(None);
        }

        // 🔑 wave56-CG: this was a `debug_assert!`, i.e. absent from the release
        // binary that served on the H200. It now runs in release, and it
        // returns rather than panics, so a batch the scheduler should never
        // have formed costs the requests in it and not the engine task.
        //
        // Still a hard postcondition: front-alignment must have MADE the batch
        // uniform. A cohort that could not be aligned still fails here rather
        // than silently writing every sequence's K/V to the wrong slot.
        ensure_uniform_batch_cache_lens(seqs, modify_draft_cache)?;

        let _prof = arc_profiler::span("clone_in_cache");
        let xs_per_seq = xs_per_sequence_enabled();
        let mut new_k_cache = Vec::new();
        let mut new_v_cache = Vec::new();
        let mut xs_row_lens: Vec<Option<(Vec<usize>, Vec<usize>)>> =
            vec![None; pipeline.get_metadata().num_hidden_layers];

        // `num_hidden_layers` here is the *cache vector* length, not 43: V4
        // appends one `XsRolling` compressor-history slot per CSA/HCA layer
        // after the KV entries, so this loop runs 43 + n_compressed times.
        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            let batch_len = seqs.len();

            // `seqs[0]` still decides whether the slot exists at all — a model
            // that shares this layer's cache (gemma3n) has no tensor for anyone.
            {
                let src_cache = if modify_draft_cache {
                    seqs[0].normal_draft_cache()
                } else {
                    seqs[0].normal_cache()
                };
                if src_cache.get(layer).and_then(|s| s.as_ref()).is_none() {
                    new_k_cache.push(None);
                    new_v_cache.push(None);
                    continue;
                }
            }

            // `tail` is content, not slack, so it cannot be padded — but it
            // *can* be trimmed to a common start, which is lossless. Do that
            // before gathering, so the batch's widths agree by construction
            // rather than by luck.
            reconcile_xs_bases(seqs, layer, modify_draft_cache, xs_per_seq)?;

            // Gather every sequence's two tensors for this slot up front, so
            // the batch's shape is decided by the whole batch instead of by
            // `seqs[0]` alone. `None` marks a sequence with no slot here; it is
            // skipped, exactly as before.
            let mut srcs: Vec<Option<BatchSrc>> = Vec::with_capacity(batch_len);
            for seq in seqs.iter_mut() {
                let src_cache = if modify_draft_cache {
                    seq.normal_draft_cache()
                } else {
                    seq.normal_cache()
                };
                match src_cache.get(layer).and_then(|s| s.as_ref()) {
                    Some(cache) => srcs.push(Some(BatchSrc::of(cache, xs_per_seq)?)),
                    None => srcs.push(None),
                }
            }

            let present: Vec<&BatchSrc> = srcs.iter().flatten().collect();
            let k_slack = present[0].k_slack_dim;
            let v_slack = present[0].v_slack_dim;
            let v_front = present[0].v_slack_at_front;
            // The batched slot's row lengths are every sequence's, in batch
            // order — not `seqs[0]`'s repeated. `None` for a non-`xs` slot.
            let xs_rows: Option<(Vec<usize>, Vec<usize>)> = present[0].xs_rows.as_ref().map(|_| {
                let mut tokens = Vec::with_capacity(present.len());
                let mut base = Vec::with_capacity(present.len());
                for s in &present {
                    if let Some((t, b)) = s.xs_rows.as_ref() {
                        tokens.extend_from_slice(t);
                        base.extend_from_slice(b);
                    }
                }
                (tokens, base)
            });
            xs_row_lens[layer] = xs_rows;
            let ks: Vec<&Tensor> = present.iter().map(|s| &s.k).collect();
            let vs: Vec<&Tensor> = present.iter().map(|s| &s.v).collect();

            // Slack dims widen to the batch maximum; every other dim must
            // already agree, and says so by name if it does not.
            let one_k = reconcile_batch_dims(&ks, k_slack, "K", layer)?;
            let one_v = reconcile_batch_dims(&vs, v_slack, "V", layer)?;

            let mut dims_k = one_k.clone();
            let mut dims_v = one_v.clone();
            dims_k[0] *= batch_len;
            dims_v[0] *= batch_len;
            let (batch_k, batch_v) = {
                // Two fresh device allocations per layer, every step.
                let _s = arc_profiler::device_span("clone_in.alloc");
                (
                    Tensor::zeros(dims_k, present[0].k.dtype(), present[0].k.device())?,
                    Tensor::zeros(dims_v, present[0].v.dtype(), present[0].v.device())?,
                )
            };

            // Fill each sequence's cache slice: two device copies per sequence
            // per layer, i.e. O(B x layers) copies per token.
            let _prof_fill = arc_profiler::device_span("clone_in.slice_set");
            for (i, src) in srcs.iter().enumerate() {
                let Some(src) = src else {
                    // Skip for shared kv cache layers in models like gemma3n
                    continue;
                };
                let src_k = match k_slack {
                    Some(d) => pad_slack(&src.k, d, one_k[d], false)?,
                    None => src.k.clone(),
                };
                let src_v = match v_slack {
                    Some(d) => pad_slack(&src.v, d, one_v[d], v_front)?,
                    None => src.v.clone(),
                };
                // Each side is offset by its OWN batch dim. They are always
                // equal in practice (both sides of a slot describe the same
                // sequences), but the old code reused the K offset for V, which
                // would have silently mis-placed the V half if they ever were not.
                batch_k.slice_set(&src_k, 0, i * one_k[0])?;
                batch_v.slice_set(&src_v, 0, i * one_v[0])?;
            }
            drop(_prof_fill);
            new_k_cache.push(Some(batch_k));
            new_v_cache.push(Some(batch_v));
        }

        let seq0_cache = if modify_draft_cache {
            &*seqs[0].normal_draft_cache()
        } else {
            &*seqs[0].normal_cache()
        };

        let mut caches = Vec::new();
        for (layer_idx, (k_cache, v_cache)) in new_k_cache.into_iter().zip(new_v_cache).enumerate()
        {
            // Use this for the various parameters. Assumes all seqs are from one model.
            let Some(cache_ref) = seq0_cache[layer_idx].as_ref() else {
                // This is hit in gemma3n for the shared kv cache - create dummy cache
                // These layers don't have their own cache because they share another layer's cache
                caches.push(KvCache::Normal {
                    k: SingleCache {
                        all_data: None,
                        dim: 0,
                        current_seq_len: 0,
                        max_seq_len: 0,
                        capacity_seq_len: 0,
                    },
                    v: SingleCache {
                        all_data: None,
                        dim: 0,
                        current_seq_len: 0,
                        max_seq_len: 0,
                        capacity_seq_len: 0,
                    },
                });
                continue;
            };
            // 🔑 The batched buffer may be WIDER than `seqs[0]`'s was: slack
            // dims are widened to the batch maximum above. `capacity_seq_len`
            // must describe the buffer that now exists, not the one the
            // template happened to carry — `SingleCache::append` reallocates
            // from `capacity_seq_len`, so a stale (small) value would try to
            // `slice_set` a wide buffer into a narrow one on the next growth.
            let k_cache = k_cache.map(|x| x.contiguous()).transpose()?;
            let v_cache = v_cache.map(|x| x.contiguous()).transpose()?;
            match cache_ref {
                KvCache::Normal { k: old_k, .. } => {
                    let template_cache_dim = old_k.dim;
                    let template_cache_csl = old_k.current_seq_len;
                    let template_cache_msl = old_k.max_seq_len;
                    // dim 0 is the batch dim (it was multiplied by
                    // `batch_len` above), so only a real seq dim is read back.
                    let capacity = match (template_cache_dim, k_cache.as_ref()) {
                        (0, _) | (_, None) => old_k.capacity_seq_len,
                        (d, Some(x)) => x.dims()[d],
                    };

                    caches.push(KvCache::Normal {
                        k: SingleCache {
                            all_data: k_cache,
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: capacity,
                        },
                        v: SingleCache {
                            all_data: v_cache,
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: capacity,
                        },
                    });
                }
                KvCache::Rotating { k: old_k, .. } => {
                    let template_cache_dim = old_k.dim;
                    let template_cache_csl = old_k.current_seq_len;
                    let template_cache_msl = old_k.max_seq_len;
                    let template_cache_offset = old_k.offset;
                    // dim 0 is the batch dim (it was multiplied by
                    // `batch_len` above), so only a real seq dim is read back.
                    let capacity = match (template_cache_dim, k_cache.as_ref()) {
                        (0, _) | (_, None) => old_k.capacity_seq_len,
                        (d, Some(x)) => x.dims()[d],
                    };

                    caches.push(KvCache::Rotating {
                        k: RotatingCache {
                            all_data: k_cache,
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            offset: template_cache_offset,
                            capacity_seq_len: capacity,
                        },
                        v: RotatingCache {
                            all_data: v_cache,
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            offset: template_cache_offset,
                            capacity_seq_len: capacity,
                        },
                    });
                }
                KvCache::TurboQuant(tq) => {
                    caches.push(KvCache::TurboQuant(tq.clone()));
                }
                KvCache::XsRolling(xs) => {
                    // `ratio`, `span_groups`, `margin` and `head_dim` are model
                    // constants and come from the seq0 template. The token
                    // counts do NOT: they are every sequence's own, in batch
                    // order, so the batched slot carries the raggedness instead
                    // of flattening it onto seqs[0]'s numbers. With
                    // `ARC_V4_XS_PER_SEQ` off every sequence agrees anyway and
                    // this is the same vector repeated.
                    let mut rebuilt = (**xs).clone();
                    if let Some(k) = k_cache.as_ref() {
                        if rebuilt.comp.dim != 0 {
                            rebuilt.comp.capacity_seq_len = k.dims()[rebuilt.comp.dim];
                        }
                    }
                    rebuilt.comp.all_data = k_cache;
                    rebuilt.tail = v_cache;
                    if let Some((tokens, base)) = xs_row_lens[layer_idx].take() {
                        // `comp` is start-anchored, so the batched slot holds
                        // the LONGEST row's completed blocks; a shorter row's
                        // surplus columns are stale and are excluded by the
                        // compressed branch's own causality threshold at that
                        // row's absolute position.
                        rebuilt.comp.current_seq_len =
                            tokens.iter().copied().max().unwrap_or(0) / rebuilt.ratio;
                        rebuilt.set_row_lens(tokens, base)?;
                    }
                    caches.push(KvCache::XsRolling(Box::new(rebuilt)));
                }
            }
        }
        *pipeline.cache().normal() = NormalCache(caches);
        Ok(())
    }
    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        // Runs on EVERY decode step: `post_op` is `CacheInstruction::Out`
        // unconditionally (`engine/mod.rs:397-404`), so the batched cache is
        // split back into B per-sequence caches once per token, per layer,
        // including the compressor-history slots.
        let _prof = arc_profiler::span("clone_out_cache");
        // The cohort's dead prefix, if `clone_in_cache` front-aligned it. Read
        // once rather than per layer per sequence, and recorded per sequence so
        // the strip can be deferred to the next membership change.
        let ragged_lead = ragged_lead_pad();
        if let Some(leads) = ragged_lead.as_ref() {
            for (seq_i, seq) in seqs.iter().enumerate() {
                record_pending_lead_pad(*seq.id(), leads[seq_i]);
            }
        }
        debug_assert!(
            ragged_lead.as_ref().is_none_or(|l| l.len() == seqs.len()),
            "ragged lead_pad must have one entry per sequence in the batch"
        );
        let all_cache = pipeline.cache().normal();
        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            let cache = all_cache.0.get(layer).unwrap();
            // This case for llama 3.2 vision cross attn
            if cache.k().unwrap().is_none() {
                continue;
            }

            let (k_cache, v_cache) = match cache {
                KvCache::Normal { k, v } => {
                    (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                }
                KvCache::Rotating { k, v } => {
                    (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                }
                KvCache::TurboQuant(tq) => (
                    tq.k.current_data().unwrap().unwrap(),
                    tq.v.current_data().unwrap().unwrap(),
                ),
                KvCache::XsRolling(xs) => (
                    xs.comp
                        .all_data
                        .clone()
                        .expect("xs rolling cache: compressed rows not materialised"),
                    xs.tail
                        .clone()
                        .expect("xs rolling cache: raw tail not materialised"),
                ),
            };

            let (k_caches, v_caches) = {
                let _s = arc_profiler::device_span("clone_out.chunk");
                let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
                debug_assert_eq!(k_caches.len(), seqs.len());
                let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
                debug_assert_eq!(v_caches.len(), seqs.len());
                (k_caches, v_caches)
            };

            let _prof_rebuild = arc_profiler::span("clone_out.rebuild_per_seq");
            for (seq_i, seq) in seqs.iter_mut().enumerate() {
                let output_cache = if modify_draft_cache {
                    seq.normal_draft_cache()
                } else {
                    seq.normal_cache()
                };
                let seq_cache = &mut output_cache[layer];
                let k = k_caches.get(seq_i).unwrap().clone();
                let v = v_caches.get(seq_i).unwrap().clone();

                match cache {
                    KvCache::Normal {
                        k: cache_k,
                        v: cache_v,
                    } => {
                        // 🔑 Hand this row back its OWN length, not the batch's.
                        //
                        // On a front-aligned ragged cohort the row's real run is
                        // the SUFFIX `[lead, current_seq_len)` — the columns
                        // ahead of it are the zero-filled dead prefix. Copying
                        // the shared `current_seq_len` here (which is what this
                        // did unconditionally) would hand the sequence a cache
                        // that claims the pad as content, and the next prefill
                        // or re-batch of that sequence would read it as real.
                        // `SingleCache` is start-anchored, so the prefix has to
                        // be dropped, not just excluded by a length.
                        //
                        // `lead == 0` for every row of a uniform batch, which is
                        // the identity — no narrow, no copy, no behaviour change.
                        // The prefix is RECORDED, not dropped — see
                        // `PENDING_LEAD_PAD`. This function runs on every
                        // decode token, but what it writes is only re-read on a
                        // membership change, so paying `O(B x layers x 2)`
                        // device copies per token here was work for data nobody
                        // looks at. `clone_in_cache` strips it once, when it
                        // matters.
                        //
                        // `lead == 0` for every row of a uniform batch, so this
                        // is the identity there and the map stays empty.
                        *seq_cache = Some(KvCache::Normal {
                            k: SingleCache {
                                all_data: Some(k),
                                dim: cache_k.dim,
                                current_seq_len: cache_k.current_seq_len,
                                max_seq_len: cache_k.max_seq_len,
                                capacity_seq_len: cache_k.capacity_seq_len,
                            },
                            v: SingleCache {
                                all_data: Some(v),
                                dim: cache_v.dim,
                                current_seq_len: cache_v.current_seq_len,
                                max_seq_len: cache_v.max_seq_len,
                                capacity_seq_len: cache_v.capacity_seq_len,
                            },
                        });
                    }
                    KvCache::Rotating {
                        k: cache_k,
                        v: cache_v,
                    } => {
                        *seq_cache = Some(KvCache::Rotating {
                            k: RotatingCache {
                                all_data: Some(k),
                                dim: cache_k.dim,
                                current_seq_len: cache_k.current_seq_len,
                                max_seq_len: cache_k.max_seq_len,
                                offset: cache_k.offset,
                                capacity_seq_len: cache_k.capacity_seq_len,
                            },
                            v: RotatingCache {
                                all_data: Some(v),
                                dim: cache_v.dim,
                                current_seq_len: cache_v.current_seq_len,
                                max_seq_len: cache_v.max_seq_len,
                                offset: cache_v.offset,
                                capacity_seq_len: cache_v.capacity_seq_len,
                            },
                        });
                    }
                    KvCache::TurboQuant(tq) => {
                        // Clone the TurboQuant cache as-is
                        *seq_cache = Some(KvCache::TurboQuant(tq.clone()));
                    }
                    KvCache::XsRolling(xs) => {
                        let per_seq = xs
                            .split_row(seq_i, k, v)
                            .expect("xs rolling cache: splitting a batched row back out");
                        *seq_cache = Some(KvCache::XsRolling(Box::new(per_seq)));
                    }
                }
            }
        }
    }
    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        _modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        // The cohort this described is being torn down. Leaving the dead prefix
        // published would hand it to whatever batch runs next — a prompt step
        // takes `CacheInstruction::Reset`, i.e. this function, and would then
        // build a ragged mask from another cohort's geometry.
        set_ragged_lead_pad(None);

        if seqs.iter().any(|seq| seq.preallocated_cache().is_none()) {
            for layer in pipeline.cache().normal().0.iter_mut() {
                layer.reset();
            }
            return;
        }

        let layer_devices = pipeline.device_mapper().map(|device_mapper| {
            let total_layers = pipeline.cache().normal().0.len();
            let mut layer_devices = Vec::with_capacity(total_layers);
            for layer in 0..total_layers {
                // `None` for cache entries beyond the device-mapped layer
                // range (e.g. DeepSeek V4's trailing xs-history entries,
                // which are skipped below before this is consulted).
                let device = device_mapper.device_for(layer, false).cloned();
                layer_devices.push(device);
            }
            layer_devices
        });

        let old_caches = pipeline.cache().normal().0.clone();

        for (layer_idx, layer) in pipeline.cache().normal().0.iter_mut().enumerate() {
            if !load_preallocated_cache {
                layer.reset();
                continue;
            }

            // Auxiliary non-KV cache entries (DeepSeek V4's compressor-input
            // xs histories, seq dim 1 over `[B, T, hidden]`) have no
            // preallocated per-sequence buffer — the preallocated caches are
            // KV-shaped `[1, kv_heads, T, head_dim]`. They always start a
            // fresh sequence empty.
            if matches!(&old_caches[layer_idx], KvCache::Normal { k, .. } if k.dim != 2)
                || matches!(&old_caches[layer_idx], KvCache::XsRolling(_))
            {
                layer.reset();
                continue;
            }

            let mut k_caches = Vec::new();
            let mut v_caches = Vec::new();
            for seq in seqs.iter_mut() {
                let (mut k_preallocated_cache, mut v_preallocated_cache) =
                    (*seq.preallocated_cache().as_ref().unwrap()).clone();
                if let Some(layer_devices) = &layer_devices {
                    let layer_dev = layer_devices[layer_idx]
                        .as_ref()
                        .expect("Internal bug, layer out of range!");
                    k_preallocated_cache = k_preallocated_cache
                        .to_device(layer_dev)
                        .expect("Could not prepare cache");
                    v_preallocated_cache = v_preallocated_cache
                        .to_device(layer_dev)
                        .expect("Could not prepare cache");
                }
                k_caches.push(k_preallocated_cache);
                v_caches.push(v_preallocated_cache);
            }
            let k_cache = if k_caches.len() > 1 {
                Tensor::cat(&k_caches, 0).unwrap()
            } else {
                k_caches[0].clone()
            };
            let v_cache = if v_caches.len() > 1 {
                Tensor::cat(&v_caches, 0).unwrap()
            } else {
                v_caches[0].clone()
            };

            // Use this for the various parameters. Assumes all seqs are from one model.
            match &old_caches[layer_idx] {
                // Unreachable: the xs rolling entries are reset and skipped
                // above (they have no preallocated KV-shaped buffer). Kept as
                // a reset rather than a panic so a future reordering degrades
                // to "start empty", which is always safe for this entry.
                KvCache::XsRolling(_) => {
                    layer.reset();
                }
                KvCache::Normal { k, .. } => {
                    let template_cache_dim = k.dim;
                    let template_cache_msl = k.max_seq_len;

                    let cache = KvCache::Normal {
                        k: SingleCache {
                            all_data: Some(k_cache.zeros_like().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: k_cache.dims()[template_cache_dim],
                        },
                        v: SingleCache {
                            all_data: Some(v_cache.zeros_like().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: k_cache.dims()[template_cache_dim],
                        },
                    };
                    *layer = cache;
                }
                KvCache::Rotating { k, .. } => {
                    let template_cache_dim = k.dim;
                    let template_cache_msl = k.max_seq_len;

                    // Rotating cache is not preallocated.
                    let cache = KvCache::Rotating {
                        k: RotatingCache {
                            all_data: None,
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            offset: 0,
                            capacity_seq_len: 0,
                        },
                        v: RotatingCache {
                            all_data: None,
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            offset: 0,
                            capacity_seq_len: 0,
                        },
                    };
                    *layer = cache;
                }
                KvCache::TurboQuant(tq) => {
                    // Reset TurboQuant cache
                    let mut new_tq = tq.clone();
                    new_tq.reset();
                    *layer = KvCache::TurboQuant(new_tq);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    cache: Arc<Mutex<LayerCaches>>,
    xlora_cache: Option<Arc<Mutex<LayerCaches>>>,
    draft_cache: Arc<Mutex<LayerCaches>>,
    scalings_cache: Option<Arc<Mutex<Option<Tensor>>>>,
}

impl Cache {
    pub(crate) fn new(len: usize, is_xlora: bool) -> Self {
        Self {
            cache: Arc::new(Mutex::new(vec![None; len])),
            xlora_cache: if is_xlora {
                Some(Arc::new(Mutex::new(vec![None; len])))
            } else {
                None
            },
            draft_cache: Arc::new(Mutex::new(vec![None; len])),
            scalings_cache: if is_xlora {
                Some(Arc::new(Mutex::new(None)))
            } else {
                None
            },
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.cache)
    }

    pub(crate) fn draft_lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.draft_cache)
    }

    /// # Panics
    /// If there is no xlora cache
    pub(crate) fn xlora_lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.xlora_cache.as_ref().expect("No X-LoRA cache."))
    }

    /// # Panics
    /// If there is no xlora cache
    pub(crate) fn get_scalings_cache(&self) -> MutexGuard<'_, Option<Tensor>> {
        get_mut_arcmutex!(self
            .scalings_cache
            .as_ref()
            .expect("No X-LoRA scalings cache."))
    }

    pub(crate) fn is_xlora(&self) -> bool {
        self.xlora_cache.is_some()
    }

    /// Update the KV cache and return (k,v)
    pub(crate) fn update_kv_cache(
        cache: &mut Option<(Tensor, Tensor)>,
        k: Tensor,
        v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (k, v) = match &*cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                (k, v)
            }
        };
        *cache = Some((k.clone(), v.clone()));
        Ok((k.contiguous()?, v.contiguous()?))
    }

    /// Update the KV cache and return (k,v,attn_mask)
    pub(crate) fn update_kv_cache_sliding_window(
        cache: &mut Option<(Tensor, Tensor)>,
        k: Tensor,
        v: Tensor,
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let (k, v, attention_mask) = match cache.clone() {
            None => (k, v, attention_mask.cloned()),
            Some((mut prev_k, mut prev_v)) => {
                let mut mask = attention_mask.cloned();
                if let Some(sliding_window) = sliding_window {
                    let kv_seq_len = prev_k.dim(2)?;
                    if kv_seq_len > sliding_window {
                        prev_k = prev_k.narrow(
                            2,
                            kv_seq_len - (sliding_window - 1),
                            sliding_window - 1,
                        )?;
                        prev_v = prev_v.narrow(
                            2,
                            kv_seq_len - (sliding_window - 1),
                            sliding_window - 1,
                        )?;
                        if let Some(ref mut mask) = mask {
                            let mask_len = mask.dim(1)?;
                            *mask = mask.narrow(
                                1,
                                mask_len - (sliding_window - 1),
                                sliding_window - 1,
                            )?;
                            *mask = Tensor::cat(
                                &[&*mask, &mask.narrow(1, mask_len - 1, 1)?.ones_like()?],
                                D::Minus1,
                            )?;
                        }
                    }
                }
                let (k, v) = {
                    let k = Tensor::cat(&[prev_k, k], 2)?.contiguous()?;
                    let v = Tensor::cat(&[prev_v, v], 2)?.contiguous()?;
                    (k, v)
                };
                (k, v, mask)
            }
        };
        *cache = Some((k.clone(), v.clone()));
        Ok((k.contiguous()?, v.contiguous()?, attention_mask))
    }
}

pub struct FullCacheManager;

enum SeqCache {
    Normal,
    XLora,
    Draft,
}

fn clone_in_cache(
    num_hidden_layers: usize,
    cache: &mut LayerCaches,
    seqs: &mut [&mut crate::sequence::Sequence],
    src: SeqCache,
) {
    let mut new_cache = Vec::new();
    'outer: for layer in 0..num_hidden_layers {
        let mut k_vec = Vec::new();
        let mut v_vec = Vec::new();
        for seq in &mut *seqs {
            let src_cache = match src {
                SeqCache::Normal => seq.cache(),
                SeqCache::XLora => seq.xlora_cache(),
                SeqCache::Draft => seq.draft_cache(),
            };
            let cache = src_cache.get(layer).unwrap();
            // This case for llama 3.2 vision cross attn
            if cache.is_none() {
                new_cache.push(None);
                continue 'outer;
            }
            let cache = cache
                .as_ref()
                .expect("Not handling completions in `clone_in_cache`.");
            k_vec.push(cache.0.clone());
            v_vec.push(cache.1.clone());
        }
        new_cache.push(Some((
            if k_vec.len() > 1 {
                Tensor::cat(&k_vec, 0).unwrap()
            } else {
                k_vec[0].clone()
            },
            if v_vec.len() > 1 {
                Tensor::cat(&v_vec, 0).unwrap()
            } else {
                v_vec[0].clone()
            },
        )));
    }
    *cache = new_cache;
}

fn clone_out_cache(
    num_hidden_layers: usize,
    cache: &mut LayerCaches,
    seqs: &mut [&mut crate::sequence::Sequence],
    target: SeqCache,
) {
    for layer in 0..num_hidden_layers {
        let cache = cache.get(layer).unwrap();
        // This case for llama 3.2 vision cross attn
        if cache.is_none() {
            continue;
        }

        let k_cache = cache.as_ref().unwrap().0.clone();
        let v_cache = cache.as_ref().unwrap().1.clone();

        let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
        debug_assert_eq!(k_caches.len(), seqs.len());
        let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
        debug_assert_eq!(v_caches.len(), seqs.len());

        for (seq_i, seq) in seqs.iter_mut().enumerate() {
            let output_cache = match target {
                SeqCache::Normal => seq.cache(),
                SeqCache::XLora => seq.xlora_cache(),
                SeqCache::Draft => seq.draft_cache(),
            };
            let seq_cache = &mut output_cache[layer];
            let k = k_caches.get(seq_i).unwrap().clone();
            let v = v_caches.get(seq_i).unwrap().clone();
            *seq_cache = Some((k, v));
        }
    }
}

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for FullCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) -> Result<()> {
        if modify_draft_cache {
            clone_in_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().lock(),
                seqs,
                SeqCache::Draft,
            );
            return Ok(());
        }
        clone_in_cache(
            pipeline.get_metadata().num_hidden_layers,
            &mut pipeline.cache().full().lock(),
            seqs,
            SeqCache::Normal,
        );
        if pipeline.get_metadata().is_xlora && !pipeline.get_metadata().no_kv_cache {
            clone_in_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().xlora_lock(),
                seqs,
                SeqCache::XLora,
            );
        }
        if pipeline.get_metadata().is_xlora {
            pipeline
                .cache()
                .full()
                .get_scalings_cache()
                .clone_from(seqs[0].scaling_cache());
        }
        Ok(())
    }

    fn clone_out_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) {
        if modify_draft_cache {
            clone_out_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().lock(),
                seqs,
                SeqCache::Draft,
            );
            return;
        }
        clone_out_cache(
            pipeline.get_metadata().num_hidden_layers,
            &mut pipeline.cache().full().lock(),
            seqs,
            SeqCache::Normal,
        );
        if pipeline.get_metadata().is_xlora && !pipeline.get_metadata().no_kv_cache {
            clone_out_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().xlora_lock(),
                seqs,
                SeqCache::XLora,
            );
        }
        if pipeline.get_metadata().is_xlora {
            seqs[0]
                .scaling_cache()
                .clone_from(&pipeline.cache().full().get_scalings_cache());
        }
    }

    fn set_none_cache(
        &self,
        pipeline: &T,
        _seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
        let mut new_cache = Vec::new();
        for _ in 0..pipeline.get_metadata().num_hidden_layers {
            new_cache.push(None);
        }
        pipeline.cache().full().lock().clone_from(&new_cache);
        if modify_draft_cache {
            pipeline.cache().full().draft_lock().clone_from(&new_cache);
        }
        if pipeline.cache().full().is_xlora() {
            *pipeline.cache().full().xlora_lock() = new_cache;
        }
    }
}

/// Cache manager for hybrid models (attention + recurrent layers).
///
/// This implements vLLM-style continuous batching:
/// - Attention layers: Standard KV cache batching (cat on clone_in, chunk on clone_out)
/// - Recurrent layers: Pool-based state management with indexed access
///
/// Each sequence has a `recurrent_state_idx` pointing to its slot in the
/// state pool. The forward pass builds a `state_indices` tensor from these
/// indices and uses gather/scatter operations.
pub struct HybridCacheManager;

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for HybridCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) -> Result<()> {
        let mut hybrid_cache = pipeline.cache().hybrid();
        let num_layers = hybrid_cache.num_layers();

        // Build state_indices for recurrent layers from sequences' recurrent_state_idx
        // Find the device from the first recurrent layer's pool
        let recurrent_device = hybrid_cache.caches.iter().find_map(|c| {
            if let HybridLayerCache::Recurrent(pool) = c {
                Some(pool.device().clone())
            } else {
                None
            }
        });

        // Ensure every sequence has a recurrent slot when using hybrid cache.
        let mut state_index_allocation_failed = false;
        let mut newly_allocated = Vec::new();
        for (seq_idx, seq) in seqs.iter_mut().enumerate() {
            if seq.recurrent_state_idx().is_none() {
                if let Some(slot_idx) = hybrid_cache.allocate_seq() {
                    seq.set_recurrent_state_idx(Some(slot_idx));
                    newly_allocated.push((seq_idx, slot_idx));
                } else {
                    tracing::warn!(
                        "Failed to allocate recurrent state slot for sequence {}, hybrid forward will fail for this batch.",
                        seq.id()
                    );
                    state_index_allocation_failed = true;
                    break;
                }
            }
        }
        if state_index_allocation_failed {
            for (seq_idx, slot_idx) in newly_allocated {
                seqs[seq_idx].set_recurrent_state_idx(None);
                hybrid_cache.free_seq(slot_idx);
            }
        }

        if let Some(device) = recurrent_device {
            if state_index_allocation_failed {
                hybrid_cache.set_state_indices(None);
            } else {
                // Build state_indices tensor from sequences
                let mut indices = Vec::with_capacity(seqs.len());
                for seq in seqs.iter() {
                    if let Some(idx) = seq.recurrent_state_idx() {
                        #[allow(clippy::cast_possible_truncation)]
                        indices.push(idx as u32);
                    } else {
                        tracing::warn!(
                            "Sequence {} missing recurrent_state_idx during hybrid clone_in_cache.",
                            seq.id()
                        );
                        hybrid_cache.set_state_indices(None);
                        return Ok(());
                    }
                }
                if let Ok(state_indices) = Tensor::from_vec(indices, (seqs.len(),), &device) {
                    hybrid_cache.set_state_indices(Some(state_indices));
                } else {
                    hybrid_cache.set_state_indices(None);
                }
            }
        }

        // For attention layers, we still need to batch KV caches
        for layer_idx in 0..num_layers {
            let layer_cache = hybrid_cache.caches.get_mut(layer_idx).unwrap();

            if let HybridLayerCache::Attention(kv_cache) = layer_cache {
                // Batch KV caches from sequences (same as NormalCacheManager)
                let mut k_tensors = Vec::new();
                let mut v_tensors = Vec::new();
                let mut template_cache: Option<KvCache> = None;

                for seq in seqs.iter_mut() {
                    let seq_cache = if modify_draft_cache {
                        seq.normal_draft_cache()
                    } else {
                        seq.normal_cache()
                    };
                    if let Some(Some(ref kv)) = seq_cache.get(layer_idx) {
                        if template_cache.is_none() {
                            template_cache = Some(kv.clone());
                        }
                        if let (Ok(Some(k)), Ok(Some(v))) = (kv.k(), kv.v()) {
                            k_tensors.push(k);
                            v_tensors.push(v);
                        }
                    }
                }

                if !k_tensors.is_empty() {
                    // cat/clone of narrow'd views may be non-contiguous;
                    // all_data must be contiguous for slice_set in SingleCache::append.
                    let batched_k = if k_tensors.len() > 1 {
                        Tensor::cat(&k_tensors, 0).unwrap()
                    } else {
                        k_tensors[0].contiguous().unwrap()
                    };
                    let batched_v = if v_tensors.len() > 1 {
                        Tensor::cat(&v_tensors, 0).unwrap()
                    } else {
                        v_tensors[0].contiguous().unwrap()
                    };

                    if let Some(ref template) = template_cache {
                        match (template, kv_cache) {
                            (KvCache::Normal { k: tk, .. }, KvCache::Normal { k, v }) => {
                                k.all_data = Some(batched_k);
                                k.current_seq_len = tk.current_seq_len;
                                k.capacity_seq_len = tk.current_seq_len;
                                v.all_data = Some(batched_v);
                                v.current_seq_len = tk.current_seq_len;
                                v.capacity_seq_len = tk.current_seq_len;
                            }
                            (KvCache::Rotating { k: tk, .. }, KvCache::Rotating { k, v }) => {
                                k.all_data = Some(batched_k);
                                k.current_seq_len = tk.current_seq_len;
                                k.capacity_seq_len = tk.current_seq_len;
                                k.offset = tk.offset;
                                v.all_data = Some(batched_v);
                                v.current_seq_len = tk.current_seq_len;
                                v.capacity_seq_len = tk.current_seq_len;
                                v.offset = tk.offset;
                            }
                            _ => {}
                        }
                    }
                }
            }
            // For recurrent layers: No copying needed!
            // The pool is accessed directly via state_indices during forward.
        }
        Ok(())
    }

    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        let hybrid_cache = pipeline.cache().hybrid();
        let num_layers = hybrid_cache.num_layers();
        let num_seqs = seqs.len();

        // For attention layers, split batched KV caches back to sequences
        for layer_idx in 0..num_layers {
            let layer_cache = hybrid_cache.caches.get(layer_idx).unwrap();

            if let HybridLayerCache::Attention(kv_cache) = layer_cache {
                if let (Ok(Some(k)), Ok(Some(v))) = (kv_cache.k(), kv_cache.v()) {
                    let k_chunks = k.chunk(num_seqs, 0).unwrap();
                    let v_chunks = v.chunk(num_seqs, 0).unwrap();

                    for (seq_idx, seq) in seqs.iter_mut().enumerate() {
                        // chunk() returns non-contiguous views; all_data must be contiguous.
                        let seq_k = k_chunks.get(seq_idx).unwrap().contiguous().unwrap();
                        let seq_v = v_chunks.get(seq_idx).unwrap().contiguous().unwrap();

                        let seq_cache = if modify_draft_cache {
                            seq.normal_draft_cache()
                        } else {
                            seq.normal_cache()
                        };

                        // Initialize cache if needed
                        if seq_cache.get(layer_idx).is_none() || seq_cache[layer_idx].is_none() {
                            while seq_cache.len() <= layer_idx {
                                seq_cache.push(None);
                            }
                            seq_cache[layer_idx] = Some(kv_cache.clone());
                        }

                        if let Some(ref mut seq_kv) = seq_cache[layer_idx] {
                            match (kv_cache, seq_kv) {
                                (KvCache::Normal { k: src_k, .. }, KvCache::Normal { k, v }) => {
                                    k.all_data = Some(seq_k);
                                    k.current_seq_len = src_k.current_seq_len;
                                    k.capacity_seq_len = src_k.current_seq_len;
                                    v.all_data = Some(seq_v);
                                    v.current_seq_len = src_k.current_seq_len;
                                    v.capacity_seq_len = src_k.current_seq_len;
                                }
                                (
                                    KvCache::Rotating { k: src_k, .. },
                                    KvCache::Rotating { k, v },
                                ) => {
                                    k.all_data = Some(seq_k);
                                    k.current_seq_len = src_k.current_seq_len;
                                    k.capacity_seq_len = src_k.current_seq_len;
                                    k.offset = src_k.offset;
                                    v.all_data = Some(seq_v);
                                    v.current_seq_len = src_k.current_seq_len;
                                    v.capacity_seq_len = src_k.current_seq_len;
                                    v.offset = src_k.offset;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            // For recurrent layers: No splitting needed!
            // The pool was updated in-place during forward via scatter operations.
        }
    }

    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
        // Reset attention KV caches in sequences
        for seq in seqs.iter_mut() {
            let seq_cache = if modify_draft_cache {
                seq.normal_draft_cache()
            } else {
                seq.normal_cache()
            };
            for kv in seq_cache.iter_mut().flatten() {
                kv.reset();
            }
        }
        // Reset the hybrid cache (including recurrent state pools)
        let mut hybrid_cache = pipeline.cache().hybrid();
        hybrid_cache.reset();

        // Build state_indices so the forward pass can access recurrent pool states.
        // Sequences already have slots allocated from add_request.
        let recurrent_device = hybrid_cache.caches.iter().find_map(|c| {
            if let HybridLayerCache::Recurrent(pool) = c {
                Some(pool.device().clone())
            } else {
                None
            }
        });
        if let Some(device) = recurrent_device {
            #[allow(clippy::cast_possible_truncation)]
            let indices: Vec<u32> = seqs
                .iter()
                .filter_map(|seq| seq.recurrent_state_idx().map(|idx| idx as u32))
                .collect();
            if indices.len() == seqs.len() {
                if let Ok(state_indices) = Tensor::from_vec(indices, (seqs.len(),), &device) {
                    hybrid_cache.set_state_indices(Some(state_indices));
                }
            }
        }
    }
}

#[cfg(test)]
mod clone_in_cache_invariant_tests {
    use super::*;
    use crate::sampler::Sampler;
    use crate::sequence::{SeqStepType, SequenceGroup, SequenceRecognizer};
    use candle_core::Device;

    /// A cache slot whose only interesting property is its `current_seq_len`.
    /// `all_data` stays `None` — `first_mismatched_cache_len` reads lengths,
    /// never tensors, which is exactly the point: `CACHE_GROW_SIZE = 512` means
    /// the tensor shapes agree even when the lengths do not, so a shape check
    /// would not catch this.
    fn slot(current_seq_len: usize) -> KvCache {
        KvCache::Normal {
            k: SingleCache {
                all_data: None,
                dim: 2,
                current_seq_len,
                capacity_seq_len: 512,
                max_seq_len: 4096,
            },
            v: SingleCache {
                all_data: None,
                dim: 2,
                current_seq_len,
                capacity_seq_len: 512,
                max_seq_len: 4096,
            },
        }
    }

    /// Minimal sequence carrying `n_layers` normal-cache slots all at
    /// `current_seq_len`. Mirrors `sequence::tests::dummy_seq`: no model, no
    /// engine.
    fn seq_with_cache_len(id: usize, n_layers: usize, current_seq_len: usize) -> Sequence {
        let (dummy_sender, _rx) = tokio::sync::mpsc::channel(1);
        let dummy_sampler = Sampler::new(
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
        )
        .unwrap();
        let group = Arc::new(std::sync::Mutex::new(SequenceGroup::new(
            1, false, false, None,
        )));
        let mut seq = Sequence::new_waiting(
            vec![1u32; current_seq_len.max(1)],
            String::new(),
            id,
            0,
            n_layers,
            dummy_sender,
            dummy_sampler,
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
        for _ in 0..n_layers {
            cache.push(Some(slot(current_seq_len)));
        }
        seq
    }

    /// A `Normal` slot with a REAL buffer, so `front_pad` has tensors to move.
    /// `all_data` is preallocated to `capacity` and the live run is the first
    /// `live` rows, which is exactly how production builds it
    /// (`kv_cache/mod.rs` preallocated cache path) — `SingleCache::append`
    /// writes into a capacity buffer, it does not grow a narrow one.
    fn materialised_slot(live: usize, capacity: usize, mark: f32) -> KvCache {
        let half = |mark: f32| {
            let data: Vec<f32> = (0..capacity).map(|i| mark + i as f32).collect();
            let all = Tensor::from_vec(data, (1, 1, capacity, 1), &Device::Cpu).unwrap();
            SingleCache {
                all_data: Some(all),
                dim: 2,
                current_seq_len: live,
                capacity_seq_len: capacity,
                max_seq_len: 4096,
            }
        };
        KvCache::Normal {
            k: half(mark),
            v: half(mark + 1000.0),
        }
    }

    fn rows(slot: &KvCache) -> Vec<f32> {
        let KvCache::Normal { k, .. } = slot else {
            unreachable!()
        };
        k.all_data
            .as_ref()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// 🔑 The mechanism that lets one dense batched buffer carry per-sequence
    /// lengths: the live run is shifted so it ENDS at the target, which makes
    /// `SingleCache::append`'s one shared write offset correct for every row at
    /// once. Content must survive the move exactly, and the prefix must be
    /// zero — a nonzero prefix would be attended as real keys.
    #[test]
    fn front_pad_moves_the_live_run_to_the_end_and_zeroes_the_prefix() {
        let mut slot = materialised_slot(5, 16, 100.0);
        let before = rows(&slot)[..5].to_vec();
        let lead = front_pad_kv_cache(&mut slot, 9).unwrap();
        assert_eq!(
            lead, 4,
            "a 5-row run inside a 9-wide target has a 4-row prefix"
        );
        assert_eq!(slot.current_seq_len(), 9);
        let after = rows(&slot);
        assert!(
            after[..4].iter().all(|x| *x == 0.0),
            "the dead prefix must be zeroed, not left holding another position's K/V"
        );
        assert_eq!(
            after[4..9],
            before[..],
            "the live run must survive the shift bit-for-bit"
        );
    }

    /// A no-op front pad must not reallocate or perturb anything — this is the
    /// B=1 path, and the uniform-batch path, and it must stay free.
    #[test]
    fn front_pad_to_the_current_length_is_a_no_op() {
        let mut slot = materialised_slot(7, 16, 3.0);
        let before = rows(&slot);
        assert_eq!(front_pad_kv_cache(&mut slot, 7).unwrap(), 0);
        assert_eq!(rows(&slot), before, "a zero-width pad must not touch data");
        assert_eq!(slot.current_seq_len(), 7);
    }

    /// A ragged cohort becomes uniform in `current_seq_len` — so
    /// `ensure_uniform_batch_cache_lens` passes and `NormalCacheManager` stacks
    /// it with no changes at all — while each sequence's dead prefix is
    /// reported so the caller can mask it.
    #[test]
    fn front_align_batch_makes_a_ragged_cohort_uniform_and_reports_each_dead_prefix() {
        let mut a = seq_with_cache_len(0, 2, 1);
        let mut b = seq_with_cache_len(1, 2, 1);
        let mut c = seq_with_cache_len(2, 2, 1);
        for (seq, live) in [(&mut a, 5usize), (&mut b, 3), (&mut c, 8)] {
            let cache = seq.normal_cache();
            cache.clear();
            for l in 0..2 {
                cache.push(Some(materialised_slot(live, 16, 10.0 * (l + 1) as f32)));
            }
        }
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b, &mut c];

        assert!(
            first_mismatched_cache_len(&mut seqs, false).is_some(),
            "the fixture must start ragged or this test proves nothing"
        );
        let lead = front_align_batch(&mut seqs, false).unwrap();
        assert_eq!(
            lead,
            vec![3, 5, 0],
            "each sequence's dead prefix is `max_len - its own live length`"
        );
        assert_eq!(
            first_mismatched_cache_len(&mut seqs, false),
            None,
            "after alignment the dense batcher's own precondition holds again"
        );
        for seq in seqs.iter_mut() {
            for slot in seq.normal_cache().iter().flatten() {
                assert_eq!(slot.current_seq_len(), 8);
            }
        }
    }

    /// 🔑 The inverse round-trip. Front padding then stripping must return the
    /// slot to exactly what it was — same rows, same length — because that is
    /// the only reason a per-sequence length recorded after a dense batched
    /// forward means what it says.
    #[test]
    fn dropping_the_dead_prefix_inverts_front_pad() {
        let mut slot = materialised_slot(5, 16, 100.0);
        let before = rows(&slot)[..5].to_vec();
        let lead = front_pad_kv_cache(&mut slot, 9).unwrap();
        assert_eq!(lead, 4);
        drop_dead_prefix(&mut slot, lead).unwrap();
        assert_eq!(
            slot.current_seq_len(),
            5,
            "after the strip the slot's length is its OWN live run again, not the batch's width"
        );
        assert_eq!(
            rows(&slot)[..5],
            before[..],
            "the live run must survive the round trip bit-for-bit"
        );
    }

    /// 🔴 The claim `front_pad_single`'s doc makes — that the dead prefix
    /// tracks the `sqrt(steps)` spread of the sequences rather than growing
    /// linearly — is a property of the code, not of the batch. It holds only
    /// because the prefix is stripped on the way out.
    ///
    /// The fixture is a two-row cohort that diverges by a fixed 2 positions
    /// every step, which is the *worst* case for the spread and still must not
    /// compound. The negative control is the same loop with the strip removed:
    /// that is the pre-change behaviour, and it grows without bound.
    ///
    /// ⚠️ Arithmetic on real tensors, not a hardware measurement (D14).
    #[test]
    fn the_dead_prefix_does_not_accumulate_across_steps() {
        // One "step": pad both rows up to the batch max, pretend the forward
        // appended `w`, keep `commit` of it, then (optionally) strip.
        let run = |strip: bool, steps: usize| -> (Vec<usize>, Vec<Vec<f32>>) {
            let w = 4usize;
            let mut slots = [
                materialised_slot(4, 512, 1.0),
                materialised_slot(4, 512, 2.0),
            ];
            let commits = [3usize, 1];
            for _ in 0..steps {
                let target = slots
                    .iter()
                    .map(|s| s.current_seq_len())
                    .max()
                    .expect("two slots");
                let leads: Vec<usize> = slots
                    .iter_mut()
                    .map(|s| front_pad_kv_cache(s, target).unwrap())
                    .collect();
                for (i, slot) in slots.iter_mut().enumerate() {
                    // The batched forward writes `w` new positions for every row.
                    slot.set_len(target + w).unwrap();
                    if strip {
                        drop_dead_prefix(slot, leads[i]).unwrap();
                        // The slot's OWN length after the strip, deliberately
                        // not recomputed here: `target + w - lead`. Deriving it
                        // in the test instead would let a strip that moves no
                        // data and updates no length still look right.
                        let live = slot.current_seq_len() - w;
                        slot.set_len(live + commits[i]).unwrap();
                    } else {
                        slot.set_len(target + commits[i]).unwrap();
                    }
                }
            }
            (
                slots.iter().map(|s| s.current_seq_len()).collect(),
                slots.iter().map(|s| rows(s)[..4].to_vec()).collect(),
            )
        };

        // 4 committed to start, then `n` steps of 3 and 1 accepted tokens. The
        // slow row's true length is `4 + n`; anything above that is padding
        // being counted as content.
        for n in [6usize, 12] {
            let (lens, heads) = run(true, n);
            assert_eq!(
                lens,
                vec![4 + n * 3, 4 + n],
                "with the strip, each row's recorded length is exactly the tokens it committed"
            );
            assert_eq!(
                heads,
                vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 3.0, 4.0, 5.0]],
                "and the live run is back at column 0 holding its own K/V — the pad/strip pair \
                 has to be an identity on content, or the length is right about the wrong rows"
            );
        }
        // Negative control — the pre-change behaviour. The error is the slow
        // row's recorded length minus its true one, and it must grow with the
        // step count rather than settling.
        let err_at = |n: usize| run(false, n).0[1] - (4 + n);
        assert_eq!(
            (err_at(6), err_at(12)),
            (10, 22),
            "without the strip the slow row is inflated, and doubling the steps must roughly \
             double the inflation — that is the linear accumulation, not a `sqrt(steps)` spread"
        );
        assert!(
            err_at(12) >= 2 * err_at(6) - 2,
            "the inflation must scale with the step count, not saturate"
        );
    }

    /// A strip wider than the live run means the caller's per-row padding and
    /// the slot disagree. Silently clamping would leave the sequence reading
    /// another row's K/V, so it refuses by name (D18).
    #[test]
    fn dropping_more_dead_prefix_than_the_slot_holds_is_refused() {
        let mut slot = materialised_slot(5, 16, 1.0);
        let err = drop_dead_prefix(&mut slot, 6).unwrap_err().to_string();
        assert!(
            err.contains("dead prefix") && err.contains("disagree"),
            "the refusal must say whose bookkeeping disagrees; got {err}"
        );
    }

    /// A zero-width strip is the B=1 path and the uniform-batch path. It must
    /// not reallocate or perturb anything.
    #[test]
    fn dropping_a_zero_width_dead_prefix_is_a_no_op() {
        let mut slot = materialised_slot(7, 16, 3.0);
        let before = rows(&slot);
        drop_dead_prefix(&mut slot, 0).unwrap();
        assert_eq!(rows(&slot), before);
        assert_eq!(slot.current_seq_len(), 7);
    }

    /// The compressor slot is never front-padded, so it must never be
    /// stripped either — its per-row token counts are the state the whole
    /// path exists to carry.
    #[test]
    fn dropping_the_dead_prefix_leaves_an_xs_slot_untouched() {
        xs_rolling::test_override::with(true, || {
            let mut xs = KvCache::XsRolling(Box::new(XsRollingCache::new(4, 2, 64, 2048)));
            let before = xs.current_seq_len();
            drop_dead_prefix(&mut xs, 3).unwrap();
            assert_eq!(xs.current_seq_len(), before);
        });
    }

    /// With `ARC_V4_XS_PER_SEQ` off the slot that blocks DeepSeek V4 refuses
    /// **by name**. A silent success here would corrupt the compressor's
    /// distant-context branch, which nothing downstream checks.
    #[test]
    fn front_pad_refuses_a_slot_that_cannot_carry_its_own_length() {
        let mut xs = KvCache::XsRolling(Box::new(XsRollingCache::new(4, 2, 64, 2048)));
        assert!(!xs.supports_per_sequence_len());
        let err = front_pad_kv_cache(&mut xs, 8).unwrap_err().to_string();
        assert!(
            err.contains("XsRolling") && err.contains("supports_per_sequence_len"),
            "the refusal must name the slot kind and where the rule lives; got {err}"
        );
    }

    /// 🔑 With the flag ON, an `XsRolling` slot must be left **alone** by front
    /// padding: its compressed rows are start-anchored (nothing to shift) and
    /// its per-row token counts are the state this whole path exists to carry.
    /// Writing `target_len` into it — which is what `front_pad_single` does to
    /// every other slot — would flatten exactly that away.
    #[test]
    fn front_pad_leaves_a_per_row_xs_slot_untouched() {
        xs_rolling::test_override::with(true, || {
            let mut xs = KvCache::XsRolling(Box::new(XsRollingCache::new(4, 2, 64, 2048)));
            assert!(xs.supports_per_sequence_len());
            let before = xs.current_seq_len();
            assert_eq!(front_pad_kv_cache(&mut xs, 32).unwrap(), 0);
            assert_eq!(
                xs.current_seq_len(),
                before,
                "front padding must not rewrite the compressor slot's token count"
            );
        });
    }

    /// 🔑 On DeepSeek V4 the 41 compressor slots come **last**, so a loop that
    /// keeps whatever the final slot returned reports "no dead prefix" for
    /// every sequence — and the caller then builds no mask for a batch that
    /// needs one. The lead has to come from a K/V slot.
    #[test]
    fn front_align_reads_the_dead_prefix_from_a_kv_slot_not_a_trailing_xs_one() {
        xs_rolling::test_override::with(true, || {
            let mut a = seq_with_cache_len(0, 1, 1);
            let mut b = seq_with_cache_len(1, 1, 1);
            for (seq, live) in [(&mut a, 5usize), (&mut b, 8)] {
                let cache = seq.normal_cache();
                cache.clear();
                cache.push(Some(materialised_slot(live, 16, 10.0)));
                // …then the compressor slot, exactly where V4 puts it.
                cache.push(Some(KvCache::XsRolling(Box::new(XsRollingCache::new(
                    4, 2, 64, 2048,
                )))));
            }
            let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
            assert_eq!(
                front_align_batch(&mut seqs, false).unwrap(),
                vec![3, 0],
                "the trailing compressor slot must not overwrite the K/V run's dead prefix"
            );
        });
    }

    /// The compressor slot stops being the reason per-sequence MTP advance is
    /// refused. That refusal is `cache_supports_per_sequence_advance`'s first
    /// failing slot, and it named `XsRolling` in every V4 cache until now.
    #[test]
    fn a_v4_shaped_cache_no_longer_refuses_per_sequence_lengths() {
        let v4 = vec![
            materialised_slot(8, 16, 1.0),
            KvCache::XsRolling(Box::new(XsRollingCache::new(4, 2, 64, 2048))),
        ];
        assert!(
            !v4.iter().all(KvCache::supports_per_sequence_len),
            "with the flag off the compressor slot must still decline"
        );
        xs_rolling::test_override::with(true, || {
            assert!(
                v4.iter().all(KvCache::supports_per_sequence_len),
                "with the flag on every slot of a V4-shaped cache carries its own length"
            );
        });
    }

    /// The uniformity precondition exists because one dense `slice_set` demands
    /// identical dims. Once the compressor slots carry per-row token counts
    /// that reason is gone for them — and only for them: a ragged K/V slot must
    /// still be caught.
    #[test]
    fn the_uniformity_check_exempts_xs_slots_and_nothing_else() {
        let build = |a_kv: usize, b_kv: usize, a_xs: usize, b_xs: usize| {
            let mut a = seq_with_cache_len(0, 1, 1);
            let mut b = seq_with_cache_len(1, 1, 1);
            for (seq, kv, xs_tokens) in [(&mut a, a_kv, a_xs), (&mut b, b_kv, b_xs)] {
                let cache = seq.normal_cache();
                cache.clear();
                cache.push(Some(slot(kv)));
                let mut x = XsRollingCache::new(4, 2, 64, 2048);
                x.assign_row_lens(vec![xs_tokens], vec![0]);
                cache.push(Some(KvCache::XsRolling(Box::new(x))));
            }
            (a, b)
        };

        // Ragged ONLY on the compressor slot.
        let (mut a, mut b) = build(8, 8, 8, 11);
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        assert_eq!(
            first_mismatched_cache_len_inner(&mut seqs, false, false).map(|m| m.0),
            Some(1),
            "with the flag off a ragged compressor slot is still a refusal"
        );
        assert_eq!(
            first_mismatched_cache_len_inner(&mut seqs, false, true),
            None,
            "with per-row state the compressor slot is allowed to disagree"
        );

        // Ragged on the K/V slot: never exempt, under either flag.
        let (mut c, mut d) = build(8, 9, 8, 8);
        let mut seqs: Vec<&mut Sequence> = vec![&mut c, &mut d];
        assert_eq!(
            first_mismatched_cache_len_inner(&mut seqs, false, true).map(|m| m.0),
            Some(0),
            "the exemption must be for compressor slots only"
        );
    }

    /// The end-anchored pad is the `xs` window's half of the left-alignment
    /// trick: a narrower row's live tokens must finish at the same column as
    /// everyone else's, so the one shared append offset serves them all.
    #[test]
    fn end_anchored_padding_puts_the_live_run_at_the_back() {
        let src = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3, 1), &Device::Cpu).unwrap();
        let back = pad_slack(&src, 1, 5, true).unwrap();
        assert_eq!(
            back.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 0.0, 1.0, 2.0, 3.0],
            "an end-anchored window pads at the FRONT"
        );
        let front = pad_slack(&src, 1, 5, false).unwrap();
        assert_eq!(
            front.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 0.0, 0.0],
            "a start-anchored buffer still pads at the back"
        );
        assert!(
            pad_slack(&src, 1, 2, true).is_err(),
            "padding is never a truncation"
        );
    }

    /// Front padding only ever grows. Shortening is `set_len`'s job and has its
    /// own retention rules; conflating the two would let a caller drop rows the
    /// compressor still needs without going through `try_set_len`.
    #[test]
    fn front_pad_refuses_to_shorten() {
        let mut slot = materialised_slot(9, 16, 1.0);
        assert!(front_pad_kv_cache(&mut slot, 4).is_err());
    }

    #[test]
    fn uniform_cache_lens_are_accepted() {
        let mut a = seq_with_cache_len(0, 4, 100);
        let mut b = seq_with_cache_len(1, 4, 100);
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        assert_eq!(
            first_mismatched_cache_len(&mut seqs, false),
            None,
            "sequences at equal cache lengths must be accepted — this is the \
             normal, correct batch the scheduler produces"
        );
    }

    /// The silent-corruption case. Two sequences 100 tokens apart still have
    /// identical `all_data` shapes (CACHE_GROW_SIZE = 512), so `slice_set` in
    /// `clone_in_cache` succeeds and only `current_seq_len` differs — the
    /// shorter sequence would then write its next token at the longer one's
    /// offset and attend over a window of zeros.
    #[test]
    fn mismatched_cache_lens_are_detected() {
        let mut a = seq_with_cache_len(0, 4, 100);
        let mut b = seq_with_cache_len(1, 4, 200);
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        assert_eq!(
            first_mismatched_cache_len(&mut seqs, false),
            Some((0, 100, 200, 1)),
            "a length-mismatched batch must be reported: layer 0, seqs[0]=100 \
             vs seqs[1]=200"
        );
    }

    /// Minimal `CacheManagerMixin + MetadataMixin` so the tests below can call
    /// the real `NormalCacheManager::clone_in_cache` — the point is to exercise
    /// the guard where it actually lives, not a copy of its condition.
    struct StubPipeline {
        cache: crate::kv_cache::EitherCache,
        metadata: Arc<crate::pipeline::GeneralMetadata>,
    }

    impl StubPipeline {
        fn new(n_layers: usize) -> Self {
            Self {
                cache: crate::kv_cache::EitherCache::Normal(NormalCache::new(n_layers, 4096)),
                metadata: Arc::new(crate::pipeline::GeneralMetadata {
                    max_seq_len: 4096,
                    llg_factory: None,
                    no_kv_cache: false,
                    no_prefix_cache: false,
                    num_hidden_layers: n_layers,
                    eos_tok: vec![],
                    kind: crate::pipeline::ModelKind::Normal,
                    is_xlora: false,
                    activation_dtype: candle_core::DType::F32,
                    sliding_window: None,
                    cache_config: None,
                    cache_engine: None,
                    model_metadata: None,
                    modalities: crate::pipeline::Modalities {
                        input: vec![],
                        output: vec![],
                    },
                }),
            }
        }
    }

    impl CacheManagerMixin for StubPipeline {
        fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) -> Result<()> {
            unreachable!("tests drive NormalCacheManager directly")
        }
        fn clone_out_cache(&self, _seqs: &mut [&mut Sequence]) {
            unreachable!("tests drive NormalCacheManager directly")
        }
        fn set_none_cache(
            &self,
            _seqs: &mut [&mut Sequence],
            _reset_non_granular: bool,
            _modify_draft_cache: bool,
            _load_preallocated_cache: bool,
        ) {
            unreachable!("tests drive NormalCacheManager directly")
        }
        fn cache(&self) -> &crate::kv_cache::EitherCache {
            &self.cache
        }
    }

    impl MetadataMixin for StubPipeline {
        fn device(&self) -> candle_core::Device {
            candle_core::Device::Cpu
        }
        fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
            None
        }
        fn name(&self) -> String {
            "stub".to_string()
        }
        fn reset_non_granular_state(&self) {}
        fn get_metadata(&self) -> Arc<crate::pipeline::GeneralMetadata> {
            self.metadata.clone()
        }
        fn device_mapper(&self) -> Option<&dyn crate::device_map::DeviceMapper> {
            None
        }
    }

    /// The guard as actually installed: `clone_in_cache` must refuse a
    /// length-mismatched batch, **in release**, by returning an error.
    ///
    /// 🔑 This test used to be `#[cfg(debug_assertions)]` +
    /// `#[should_panic]`, because the guard it covered was a `debug_assert!`.
    /// That is the wave51-CB hole in one line: the check existed in CI and
    /// **did not exist in the binary that served on the H200**, where the same
    /// batch instead reached `slice_set` and panicked the engine task. Both
    /// halves are now fixed — the check runs in release, and it returns rather
    /// than panics — so the test runs in every profile and asserts a value.
    ///
    /// Mutation check: delete the `ensure_uniform_batch_cache_lens(..)?` call
    /// at the top of `clone_in_cache` and this test fails.
    ///
    /// ⚠️ **The contract narrowed, deliberately.** A length-mismatched batch is
    /// now FRONT-ALIGNED rather than refused whenever every slot can carry its
    /// own length — that is the whole point of ragged decode. The guard this
    /// test exists for is unchanged for every cache that *cannot*, which is
    /// what it now covers: a `Rotating` slot
    /// ([`KvCache::supports_per_sequence_len`] is `false`) must still be
    /// refused, in release, by returning an error rather than reaching
    /// `slice_set` and panicking the engine task.
    #[test]
    fn clone_in_cache_refuses_a_length_mismatched_batch_it_cannot_align() {
        let pipeline = StubPipeline::new(2);
        let rotating = |len: usize| {
            let mut c = KvCache::new_rotating(2, 4096, 8);
            if let KvCache::Rotating { k, v } = &mut c {
                k.current_seq_len = len;
                v.current_seq_len = len;
            }
            c
        };
        let mut a = seq_with_cache_len(0, 2, 100);
        let mut b = seq_with_cache_len(1, 2, 200);
        for (seq, len) in [(&mut a, 100usize), (&mut b, 200)] {
            let cache = seq.normal_cache();
            cache.clear();
            for _ in 0..2 {
                cache.push(Some(rotating(len)));
            }
        }
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        let err = NormalCacheManager
            .clone_in_cache(&pipeline, &mut seqs, false)
            .expect_err("a batch that cannot be front-aligned must still be refused")
            .to_string();
        assert!(err.contains("must share current_seq_len"), "got: {err}");
        assert!(err.contains("100") && err.contains("200"), "got: {err}");
    }

    /// 🔑 INERTNESS. On a UNIFORM batch this whole change must do nothing.
    ///
    /// Asserted rather than assumed, because a measured −9.7% on the B=32
    /// *uniform* cell is unexplained and "uniform means `lead == 0`, so the path
    /// is inert" was reasoning, not evidence. If any of these three counters is
    /// non-zero on a uniform cohort, the path is NOT inert and that is where the
    /// regression lives.
    #[test]
    fn a_uniform_batch_leaves_every_ragged_mechanism_untouched() {
        let pipeline = StubPipeline::new(2);
        let mut a = seq_with_cache_len(0, 2, 5);
        let mut b = seq_with_cache_len(1, 2, 5);
        for seq in [&mut a, &mut b] {
            let cache = seq.normal_cache();
            cache.clear();
            for l in 0..2 {
                cache.push(Some(materialised_slot(5, 8, (l * 100) as f32)));
            }
        }
        set_ragged_lead_pad(None);
        PENDING_LEAD_PAD.with(|m| m.borrow_mut().clear());

        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        NormalCacheManager
            .clone_in_cache(&pipeline, &mut seqs, false)
            .expect("a uniform batch must batch");

        assert_eq!(
            ragged_lead_pad(),
            None,
            "a uniform cohort must publish NO dead prefix — a Some(vec![0,0]) here would send \
             every uniform decode through the ragged mask path and off the flash kernel"
        );
        assert_eq!(
            PENDING_LEAD_PAD.with(|m| m.borrow().len()),
            0,
            "a uniform cohort must leave no deferred strip work"
        );
        // And the mask builder must decline, which is what keeps uniform decode
        // on the unmasked fast path.
        let ids = Tensor::zeros((2, 1), candle_core::DType::U32, &Device::Cpu).unwrap();
        let offsets: &[usize] = &[5, 5];
        assert!(
            crate::layers_masker::CausalMasker
                .make_causal_mask_matrix(&ids, &offsets, candle_core::DType::F32, 1)
                .unwrap()
                .is_none(),
            "a uniform cohort must still get NO mask at tgt_len == 1"
        );
    }

    /// 🔑 `batch_can_be_ragged` is the predicate the SCHEDULER acts on, so it
    /// needs a test of its own.
    ///
    /// Found by mutation: replacing the call with `true` inside
    /// `clone_in_cache` left the whole suite green, because
    /// `front_align_would_succeed` independently refuses the same batches. The
    /// alignment decision is therefore double-covered, but the *capability* —
    /// which is published to the scheduler and decides whether it stops
    /// bucketing at all — was covered by nothing.
    #[test]
    fn the_capability_predicate_is_what_the_scheduler_reads() {
        let mut normal = seq_with_cache_len(0, 2, 4);
        {
            let cache = normal.normal_cache();
            cache.clear();
            for l in 0..2 {
                cache.push(Some(materialised_slot(4, 8, (l * 100) as f32)));
            }
        }
        assert!(
            batch_can_be_ragged(&mut [&mut normal], false),
            "a cache of Normal slots must report that it can carry per-sequence lengths"
        );

        let mut rotating = seq_with_cache_len(1, 2, 4);
        {
            let cache = rotating.normal_cache();
            cache.clear();
            for _ in 0..2 {
                cache.push(Some(KvCache::new_rotating(2, 4096, 8)));
            }
        }
        assert!(
            !batch_can_be_ragged(&mut [&mut rotating], false),
            "a Rotating slot cannot carry its own length, so the scheduler must keep bucketing"
        );

        // One bad slot in one sequence is enough to disqualify the cohort.
        assert!(
            !batch_can_be_ragged(&mut [&mut normal, &mut rotating], false),
            "capability is a property of the WHOLE batch, not of its first sequence"
        );
    }

    /// 🔑 A refusal must leave the batch EXACTLY as it found it.
    ///
    /// The blanket refusal this change replaced ran before anything was
    /// touched. `front_align_batch` pads in place, row by row and slot by slot,
    /// so a naive "align, then assert" would return `Err` having already
    /// rewritten the rows ahead of whichever one failed — and the engine would
    /// fail those requests while the surviving sequences carried a half-aligned
    /// cache. `front_align_would_succeed` is what keeps that from happening.
    ///
    /// The fixture is the exact shape that bails midway: sequence 0's slots are
    /// materialised and paddable, sequence 1's are not (`all_data: None`, which
    /// is what `slot()` builds). Alignment must never start.
    #[test]
    fn a_refusal_does_not_leave_a_half_aligned_batch() {
        let pipeline = StubPipeline::new(2);
        let mut a = seq_with_cache_len(0, 2, 5);
        let mut b = seq_with_cache_len(1, 2, 3);
        {
            let cache = a.normal_cache();
            cache.clear();
            for l in 0..2 {
                cache.push(Some(materialised_slot(5, 8, (l * 100) as f32)));
            }
        }
        // `b` keeps the unmaterialised `slot(3)` from the helper.

        let before: Vec<usize> = [&mut a, &mut b]
            .iter_mut()
            .flat_map(|s| {
                s.normal_cache()
                    .iter()
                    .flatten()
                    .map(KvCache::current_seq_len)
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(before, vec![5, 5, 3, 3], "fixture: ragged and mixed");

        set_ragged_lead_pad(None);
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        let err = NormalCacheManager
            .clone_in_cache(&pipeline, &mut seqs, false)
            .expect_err("an unpaddable batch must be refused")
            .to_string();
        assert!(err.contains("must share current_seq_len"), "got: {err}");
        drop(seqs);

        let after: Vec<usize> = [&mut a, &mut b]
            .iter_mut()
            .flat_map(|s| {
                s.normal_cache()
                    .iter()
                    .flatten()
                    .map(KvCache::current_seq_len)
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            after, before,
            "the refusal mutated the batch; a half-aligned cache survived a failed step"
        );
        assert_eq!(
            ragged_lead_pad(),
            None,
            "a refused batch must not publish a dead prefix for the masker to trust"
        );
    }

    /// 🔑 The other side of that contract: a ragged batch whose slots CAN carry
    /// their own lengths is front-aligned and accepted, and the per-row dead
    /// prefix is published for the masker.
    ///
    /// Without the publication step the batch would run with a zero-filled
    /// prefix that nothing masks — a zero K row scores logit 0 and takes real
    /// softmax weight, so it would be a silent wrong answer rather than a
    /// visible failure.
    #[test]
    fn clone_in_cache_front_aligns_a_ragged_batch_it_can_carry() {
        let pipeline = StubPipeline::new(2);
        let mut a = seq_with_cache_len(0, 2, 5);
        let mut b = seq_with_cache_len(1, 2, 3);
        for (seq, live) in [(&mut a, 5usize), (&mut b, 3)] {
            let cache = seq.normal_cache();
            cache.clear();
            for l in 0..2 {
                cache.push(Some(materialised_slot(live, 8, (l * 100) as f32)));
            }
        }
        set_ragged_lead_pad(None);
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        NormalCacheManager
            .clone_in_cache(&pipeline, &mut seqs, false)
            .expect("a ragged batch of Normal slots must be front-aligned, not refused");

        assert_eq!(
            ragged_lead_pad(),
            Some(vec![0, 2]),
            "the longer row has no dead prefix; the 3-long row carries 5-3 = 2"
        );
        // NOT asserted here: `ragged_decode_supported()`. It is a process-global
        // written by every `clone_in_cache`, so a concurrently running test
        // clobbers it and the assertion is racy — which it duly was, failing in
        // the full suite and passing when filtered. The capability itself is
        // pure and is asserted directly instead.
        assert!(
            batch_can_be_ragged(&mut seqs, false),
            "a batch of Normal slots must be reported as carryable"
        );
        set_ragged_lead_pad(None);
    }

    // =====================================================================
    // wave56-CG — the two `clone_in_cache` panics wave51-CB hit on the H200,
    // on a PRODUCTION-shaped fixture.
    //
    // DOCTRINE D12: `slot()` above is deliberately `all_data: None`, which is
    // exactly the fixture shape that let both of these ship. Everything below
    // builds its caches the way the engine does — a preallocated K/V buffer,
    // and `XsRolling` state produced by running `XsRollingCache::advance`
    // itself — so the tensors can, and do, disagree.
    // =====================================================================

    /// DeepSeek V4's K/V slot as the engine hands it over: the preallocated
    /// `[1, 1, capacity, head_dim]` activation-dtype buffer that
    /// `set_none_cache` installs *before* the first append.
    fn v4_kv_slot(capacity: usize, head_dim: usize, current_seq_len: usize) -> KvCache {
        let dev = candle_core::Device::Cpu;
        let mk = || SingleCache {
            all_data: Some(
                Tensor::zeros(
                    (1usize, 1usize, capacity, head_dim),
                    candle_core::DType::F32,
                    &dev,
                )
                .unwrap(),
            ),
            dim: 2,
            current_seq_len,
            capacity_seq_len: capacity,
            max_seq_len: 4096,
        };
        KvCache::Normal { k: mk(), v: mk() }
    }

    const XS_HIDDEN: usize = 8;
    const XS_HEAD_DIM: usize = 4;

    /// Advance an `XsRollingCache` by `t` tokens through the real `advance` —
    /// the same call the V4 attention layer makes — so `tail`, `base` and
    /// `comp.capacity_seq_len` are whatever production computes.
    fn feed_xs(state: &mut XsRollingCache, t: usize) {
        let dev = candle_core::Device::Cpu;
        let ratio = state.ratio;
        let xs = Tensor::zeros((1usize, t, XS_HIDDEN), candle_core::DType::F32, &dev).unwrap();
        state
            .advance(&xs, |w| {
                let rows = w.dim(1)? / ratio;
                Tensor::zeros((1usize, rows, XS_HEAD_DIM), w.dtype(), w.device())
            })
            .unwrap();
    }

    fn xs_state(ratio: usize, span_groups: usize) -> XsRollingCache {
        XsRollingCache::new(ratio, span_groups, XS_HEAD_DIM, 4096)
    }

    /// A sequence carrying exactly `slots` — a V4-shaped heterogeneous cache
    /// vector (K/V entries first, then the compressor histories).
    fn seq_with_slots(id: usize, tokens: usize, slots: Vec<KvCache>) -> Sequence {
        let mut seq = seq_with_cache_len(id, slots.len(), tokens);
        let cache = seq.normal_cache();
        cache.clear();
        for slot in slots {
            cache.push(Some(slot));
        }
        seq
    }

    /// What `clone_in_cache` used to do, verbatim: allocate the batched buffer
    /// from `seqs[0]`'s dims and `slice_set` every other sequence into it. Kept
    /// so each test below can show that its fixture really does reproduce the
    /// panic seen on hardware, rather than asserting against a guard that
    /// nothing would have tripped (D12).
    fn legacy_batch_error(seq0: &Tensor, seq1: &Tensor) -> String {
        let mut dims = seq0.dims().to_vec();
        dims[0] *= 2;
        let batched = Tensor::zeros(dims, seq0.dtype(), seq0.device()).expect("batch alloc");
        batched.slice_set(seq0, 0, 0).expect("seq0 always fits");
        match batched.slice_set(seq1, 0, seq0.dims()[0]) {
            Ok(()) => String::from("<no error>"),
            Err(e) => e.to_string(),
        }
    }

    /// wave51-CB section 4.1 — `kv_cache/mod.rs:498`, `shape mismatch on dim 1,
    /// 576 <> 64`, on the ORDINARY decode path. Two deaths in ~1,300 GSM8K
    /// requests; zero in the 505-request batch sweep.
    ///
    /// The two numbers are not a head dim and a step count. They are two
    /// `XsRollingCache::comp` **capacities**: `init_rows = 64`, and
    /// `64 + CACHE_GROW_SIZE` = 576 after one growth. The mechanism, and why
    /// the sweep never saw it:
    ///
    /// * `comp` holds one row per `ratio` tokens, so a CSA layer (`ratio = 4`)
    ///   crosses 64 rows at 260 tokens. The sweep's sequences were ~132 tokens
    ///   — every buffer in it was 64 wide, so they always matched. GSM8K
    ///   generated up to 2048.
    /// * `SingleCache::reset` clears `current_seq_len` and `all_data` but
    ///   **not `capacity_seq_len`**, and the next `append` re-allocates at that
    ///   retained capacity. V4's attention layer resets the compressor slot at
    ///   the start of every prompt (`seqlen_offsets.iter().all(|&o| o == 0)`),
    ///   so a prompt batch scheduled after a long-context batch inherits a
    ///   576-wide buffer for brand-new sequences while one scheduled after a
    ///   short batch gets 64.
    ///
    /// So two sequences at the **identical length** can hold different-width
    /// compressed-row buffers. The scheduler cannot prevent it: it agrees on
    /// every length there is to bucket on. The extra width is preallocation
    /// slack holding nothing, so the fix is to widen the batch to fit, not to
    /// refuse.
    #[test]
    fn xs_comp_capacity_slack_does_not_kill_the_batch() {
        // seqs[0]: grew past the 64-row boundary, was reset at a prompt
        // boundary, and is now short again — capacity 576, length 256.
        let mut grown = xs_state(4, 2);
        feed_xs(&mut grown, 264); // 66 rows > 64 -> capacity grows to 576
        assert_eq!(grown.comp.capacity_seq_len, 576);
        grown.reset();
        feed_xs(&mut grown, 256);

        // seqs[1]: a fresh sequence that reached the same length — capacity 64.
        let mut fresh = xs_state(4, 2);
        feed_xs(&mut fresh, 256);

        // --- Fixture discrimination (D12) -------------------------------
        // The lengths AGREE, so this is not the ragged-batch failure; it is
        // purely a capacity disagreement, and it must be reproduced as one.
        assert_eq!(grown.current_seq_len(), fresh.current_seq_len());
        assert_eq!(grown.comp.current_seq_len(), fresh.comp.current_seq_len());
        let grown_k = grown.comp.all_data.clone().unwrap();
        let fresh_k = fresh.comp.all_data.clone().unwrap();
        assert_eq!(
            (grown_k.dims()[1], fresh_k.dims()[1]),
            (576, 64),
            "fixture cannot discriminate: the two compressed-row buffers must \
             straddle exactly one CACHE_GROW_SIZE boundary"
        );
        assert_eq!(
            legacy_batch_error(&grown_k, &fresh_k),
            "shape mismatch on dim 1, 576 <> 64",
            "the fixture must reproduce the exact panic wave51-CB saw at \
             kv_cache/mod.rs:498"
        );

        // --- The fix ----------------------------------------------------
        let pipeline = StubPipeline::new(2);
        let mut a = seq_with_slots(
            0,
            256,
            vec![v4_kv_slot(512, 8, 256), KvCache::XsRolling(Box::new(grown))],
        );
        let mut b = seq_with_slots(
            1,
            256,
            vec![v4_kv_slot(512, 8, 256), KvCache::XsRolling(Box::new(fresh))],
        );
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        NormalCacheManager
            .clone_in_cache(&pipeline, &mut seqs, false)
            .expect("capacity slack is not a ragged batch; it must be padded, not refused");

        let batched = pipeline.cache().normal();
        let KvCache::XsRolling(xs) = &batched.0[1] else {
            panic!("slot 1 must stay an XsRolling entry")
        };
        let k = xs.comp.all_data.as_ref().unwrap();
        assert_eq!(
            k.dims(),
            &[2, 576, XS_HEAD_DIM],
            "batched to the widest buffer"
        );
        assert_eq!(
            xs.comp.capacity_seq_len, 576,
            "capacity_seq_len must describe the buffer that now exists — \
             SingleCache::append reallocates from it, and a stale 64 would try \
             to slice a 576-wide buffer into a 64-wide one"
        );
        assert_eq!(
            xs.comp.current_seq_len, 64,
            "row count is unchanged by padding"
        );
    }

    /// wave51-CB section 3.2 — `kv_cache/mod.rs:499`, `shape mismatch on dim 1,
    /// 18 <> 22`, and `19 <> 23` on a second run. MTP at B=8, reproduced from a
    /// clean engine; B=1 is fine.
    ///
    /// Line 499 is the **V** half, which for an `XsRolling` slot is `xs.tail` —
    /// `[B, tokens - base, hidden]`. Unlike every other tensor
    /// `clone_in_cache` batches, its width is not a `CACHE_GROW_SIZE`-quantised
    /// capacity but an **exact function of the token count**, so *one* token of
    /// divergence is enough. For V4's HCA layers (`ratio = 128`,
    /// `span_groups = 1`, `margin = XS_TAIL_MARGIN_TOKENS = 16`) the width is
    /// `T - 128 * floor((T - 16) / 128)`, which gives 18 at T=274 and 22 at
    /// T=278 — the observed pair, four tokens apart.
    ///
    /// Four is what batched MTP produces: `mtp_pipeline.rs` commits between 1
    /// and `depth + 1` tokens per sequence while the shared cache advances by
    /// the batch minimum ("each sequence's surplus stays committed as TOKENS"),
    /// and `Sequence::cache_bucket_len` deliberately buckets on the *cache*
    /// length so the ragged cohort stays whole. Those two decisions are jointly
    /// unsound the moment a cache slot's batched width tracks tokens.
    ///
    /// This is a genuinely ragged batch: the two sequences hold different
    /// compressor history and cannot share one dense buffer. It must be
    /// refused, by name — never papered over, and never a panic on the engine
    /// task.
    #[test]
    fn ragged_xs_tail_is_refused_by_name_not_panicked() {
        let mut short = xs_state(128, 1);
        feed_xs(&mut short, 274);
        let mut long = xs_state(128, 1);
        feed_xs(&mut long, 278);

        // --- Fixture discrimination (D12) -------------------------------
        let short_v = short.tail.clone().unwrap();
        let long_v = long.tail.clone().unwrap();
        assert_eq!(
            (short_v.dims()[1], long_v.dims()[1]),
            (18, 22),
            "fixture cannot discriminate: it must land on the exact tail widths \
             wave51-CB reported"
        );
        assert_eq!(
            short.comp.all_data.as_ref().unwrap().dims()[1],
            long.comp.all_data.as_ref().unwrap().dims()[1],
            "the K half must AGREE, or this test would be reproducing the \
             capacity bug instead of the ragged-tail one (:499, not :498)"
        );
        assert_eq!(
            legacy_batch_error(&short_v, &long_v),
            "shape mismatch on dim 1, 18 <> 22",
            "the fixture must reproduce the exact panic wave51-CB saw at \
             kv_cache/mod.rs:499"
        );

        // --- The fix ----------------------------------------------------
        let pipeline = StubPipeline::new(2);
        let mut a = seq_with_slots(
            0,
            274,
            vec![v4_kv_slot(512, 8, 274), KvCache::XsRolling(Box::new(short))],
        );
        let mut b = seq_with_slots(
            1,
            278,
            vec![v4_kv_slot(512, 8, 278), KvCache::XsRolling(Box::new(long))],
        );
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        let err = NormalCacheManager
            .clone_in_cache(&pipeline, &mut seqs, false)
            .expect_err("a ragged batch must be refused, not batched and not panicked")
            .to_string();
        assert!(
            err.contains("must share current_seq_len"),
            "the refusal must name the invariant, got: {err}"
        );
        assert!(
            err.contains("274") && err.contains("278"),
            "the refusal must name BOTH lengths so the operator can see which \
             sequences diverged, got: {err}"
        );
    }

    /// The ragged-tail refusal must not depend on the K/V slots noticing
    /// first: on V4 the K/V halves are `[1, 1, capacity, head_dim]`, whose
    /// dim-1 is the head count, so they batch happily at *any* pair of lengths.
    /// Cache slot 0 agreeing is precisely the state the scheduler guarantees
    /// and `clone_in_cache` used to trust.
    #[test]
    fn kv_slots_alone_cannot_see_the_divergence() {
        let a = v4_kv_slot(512, 8, 274);
        let b = v4_kv_slot(512, 8, 278);
        let (KvCache::Normal { k: ka, .. }, KvCache::Normal { k: kb, .. }) = (&a, &b) else {
            unreachable!()
        };
        assert_eq!(
            legacy_batch_error(ka.all_data.as_ref().unwrap(), kb.all_data.as_ref().unwrap()),
            "<no error>",
            "V4's K/V buffers batch fine across a 4-token divergence — which is \
             why the panic only ever appeared on the compressor slots, and why \
             a slot-0 check can never catch it"
        );
    }

    /// A single sequence is trivially self-consistent, and a mismatch below
    /// layer 0 must still be found (the scan must not stop at the first layer).
    #[test]
    fn mismatch_is_found_on_any_layer_and_single_seqs_pass() {
        let mut solo = seq_with_cache_len(0, 4, 7);
        {
            let mut seqs: Vec<&mut Sequence> = vec![&mut solo];
            assert_eq!(first_mismatched_cache_len(&mut seqs, false), None);
        }

        let mut a = seq_with_cache_len(0, 4, 100);
        let mut b = seq_with_cache_len(1, 4, 100);
        // Only layer 2 diverges — e.g. an extra per-sequence slot (V4 stores
        // its compressor `xs` history in slots past the KV entries) drifting
        // out of lockstep with the KV caches.
        b.normal_cache()[2] = Some(slot(101));
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        assert_eq!(
            first_mismatched_cache_len(&mut seqs, false),
            Some((2, 100, 101, 1))
        );
    }

    /// Equal `current_seq_len` is NOT sufficient for `tail` to batch.
    ///
    /// This is the second, independent route into the `shape mismatch on dim 1`
    /// failure that `ensure_uniform_batch_cache_lens` closed only half of — and
    /// it needs **no MTP whatsoever**. `prefix_cacher` calls `set_len` on every
    /// stored layer, so a sequence restored from an entry stored at a greater
    /// length holds a narrower tail than one that reached the same token count
    /// directly.
    ///
    /// Mutation check: delete the `reconcile_xs_bases` call from
    /// `clone_in_cache` and this test fails on the `clone_in_cache` line with
    /// `shape mismatch on dim 1, 4 <> 132`.
    #[test]
    fn xs_base_divergence_at_equal_lengths_is_reconciled_not_refused() {
        // Restored from a prefix-cache entry stored at 300 tokens, truncated to
        // 260 — `base` stays at canonical(300), which is past canonical(260).
        let mut restored = xs_state(128, 1);
        feed_xs(&mut restored, 300);
        assert_eq!(
            restored.resumable_from(),
            256,
            "canonical(300) for ratio 128, margin 16"
        );
        restored.set_len(260).unwrap();

        // Reached 260 directly.
        let mut direct = xs_state(128, 1);
        feed_xs(&mut direct, 260);
        assert_eq!(
            direct.resumable_from(),
            128,
            "canonical(260) sits a group lower"
        );

        // --- Fixture discrimination (D12) -------------------------------
        assert_eq!(
            (restored.current_seq_len(), direct.current_seq_len()),
            (260, 260),
            "the LENGTHS must agree, or this test is reproducing the ragged-length \
             failure instead of the base one — `ensure_uniform_batch_cache_lens` \
             would catch that and never reach the tensor"
        );
        assert_eq!(
            (
                restored.tail.as_ref().unwrap().dims()[1],
                direct.tail.as_ref().unwrap().dims()[1]
            ),
            (4, 132),
            "and the WIDTHS must disagree, or there is nothing to reconcile"
        );
        assert_eq!(
            legacy_batch_error(
                restored.tail.as_ref().unwrap(),
                direct.tail.as_ref().unwrap()
            ),
            "shape mismatch on dim 1, 4 <> 132",
            "uniform lengths still panic the legacy batcher — this is a second, \
             independent way into the slice_set shape mismatch"
        );

        // --- The fix ----------------------------------------------------
        let pipeline = StubPipeline::new(2);
        let mut a = seq_with_slots(
            0,
            260,
            vec![
                v4_kv_slot(512, 8, 260),
                KvCache::XsRolling(Box::new(restored)),
            ],
        );
        let mut b = seq_with_slots(
            1,
            260,
            vec![
                v4_kv_slot(512, 8, 260),
                KvCache::XsRolling(Box::new(direct)),
            ],
        );
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        NormalCacheManager
            .clone_in_cache(&pipeline, &mut seqs, false)
            .expect("equal lengths with unequal retained windows must reconcile");

        let batched = pipeline.cache().normal();
        let KvCache::XsRolling(xs) = &batched.0[1] else {
            panic!("slot 1 must stay an XsRolling entry")
        };
        assert_eq!(
            xs.resumable_from(),
            256,
            "trimmed to the batch's largest retained start"
        );
        assert_eq!(xs.tail.as_ref().unwrap().dims(), &[2, 4, XS_HIDDEN]);
    }

    /// 🔴 **The test that discriminates between gating `reconcile_xs_bases` and
    /// porting it.**
    ///
    /// Same fixture as
    /// `xs_base_divergence_at_equal_lengths_is_reconciled_not_refused` — equal
    /// `tokens`, divergent `base`, widths 4 vs 132 — but with
    /// `ARC_V4_XS_PER_SEQ` **on**. Two things must both hold:
    ///
    /// 1. the batch assembles (the front-padded, end-anchored tail reconciles
    ///    the widths without anyone trimming anything); and
    /// 2. **each sequence keeps its OWN `base`** when the batch is split back
    ///    out — 256 stays 256 and 128 stays 128.
    ///
    /// 🔑 (2) is the whole point, and it is asserted separately from (1) because
    /// the two failure modes are different — verified by running both
    /// mutations, not assumed:
    ///
    /// * **Trim the tensors too** (an unconditional `reconcile_xs_bases`, i.e.
    ///   "port it"): the batched tail comes out `[2, 4, …]` instead of
    ///   `[2, 132, …]`, so the *width* assertion catches it. The shorter row
    ///   loses 128 tokens of rollback reach, and so does the longer one.
    /// * **Reconcile only the logical `base`** (flatten `xs_rows`' `base` to
    ///   `base_max` and leave the front-padded tensors alone — the more likely
    ///   mistake once `base` is per-row): every shape is *exactly right*, which
    ///   is all `clone_in_cache` checks, and the shorter row's rollback floor is
    ///   silently raised from 128 to 256. **Only reading the per-row `base` back
    ///   out catches this one** — it fails here with `[256, 256]` against
    ///   `[256, 128]` after passing every width assertion above.
    ///
    /// The second is the same failure mode as the cohort min-rollback #92
    /// removed and the padding traps #102/#103/#104 each caught: a batch-wide
    /// scalar standing in for a per-row quantity, invisible to every check that
    /// looks at shapes.
    #[test]
    fn per_row_xs_bases_survive_a_batch_round_trip_and_are_not_flattened() {
        xs_rolling::test_override::with(true, || {
            let mut restored = xs_state(128, 1);
            feed_xs(&mut restored, 300);
            restored.set_len(260).unwrap();
            let mut direct = xs_state(128, 1);
            feed_xs(&mut direct, 260);

            // --- Fixture discrimination (D12): this must be the BASE
            // divergence, not the length one, and the widths must really
            // disagree — otherwise there is nothing for either rule to do.
            assert_eq!(
                (restored.resumable_from(), direct.resumable_from()),
                (256, 128),
                "the two rows must start at different resume points"
            );
            assert_eq!(
                (restored.current_seq_len(), direct.current_seq_len()),
                (260, 260),
                "...at the SAME token count, or `ensure_uniform_batch_cache_lens` \
                 refuses before any of this is reached"
            );
            assert_eq!(
                (
                    restored.tail.as_ref().unwrap().dims()[1],
                    direct.tail.as_ref().unwrap().dims()[1]
                ),
                (4, 132),
                "...and holding different-width tails, or the reconciliation is \
                 vacuous either way"
            );

            let pipeline = StubPipeline::new(2);
            let mut a = seq_with_slots(
                0,
                260,
                vec![
                    v4_kv_slot(512, 8, 260),
                    KvCache::XsRolling(Box::new(restored)),
                ],
            );
            let mut b = seq_with_slots(
                1,
                260,
                vec![
                    v4_kv_slot(512, 8, 260),
                    KvCache::XsRolling(Box::new(direct)),
                ],
            );

            // (1) It assembles — the end-anchored tail front-pads to the batch
            //     maximum, and no trim was needed to get there.
            {
                let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
                NormalCacheManager
                    .clone_in_cache(&pipeline, &mut seqs, false)
                    .expect("an end-anchored ragged tail must batch without trimming");
            }
            {
                let batched = pipeline.cache().normal();
                let KvCache::XsRolling(xs) = &batched.0[1] else {
                    panic!("slot 1 must stay an XsRolling entry")
                };
                assert_eq!(
                    xs.tail.as_ref().unwrap().dims(),
                    &[2, 132, XS_HIDDEN],
                    "the narrow row is front-padded up to the batch maximum, not \
                     trimmed down to it"
                );
                let (tokens, base) = xs.row_lens();
                assert_eq!(tokens, &[260, 260]);
                assert_eq!(
                    base,
                    &[256, 128],
                    "🔴 the BATCHED cache must carry both resume points. A \
                     `base_max` flattening reads [256, 256] here and every shape \
                     check above still passes."
                );
            }

            // (2) And each sequence gets its own back.
            {
                let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
                NormalCacheManager.clone_out_cache(&pipeline, &mut seqs, false);
            }
            let got: Vec<usize> = [&mut a, &mut b]
                .iter_mut()
                .map(|seq| match seq.normal_cache()[1].as_ref() {
                    Some(KvCache::XsRolling(xs)) => xs.resumable_from(),
                    _ => panic!("slot 1 must split back out as an XsRolling entry"),
                })
                .collect();
            assert_eq!(
                got,
                vec![256, 128],
                "🔴 each sequence must keep its OWN resume point. Getting \
                 [256, 256] means the batch-wide `base_max` was written back \
                 into the shorter row, raising its rollback floor by 128 tokens \
                 — correct-looking, right-shaped, and wrong."
            );
        });
    }

    /// `trim_tail_to` refuses a multi-row cache rather than reading `base[0]`.
    ///
    /// A scalar `new_base` cannot describe a trim of rows sitting at different
    /// resume points, and proceeding would return a right-shaped tensor — which
    /// is exactly what the caller checks — so it must say so (D18).
    #[test]
    fn trimming_a_batched_xs_cache_by_one_scalar_is_refused() {
        xs_rolling::test_override::with(true, || {
            let mut restored = xs_state(128, 1);
            feed_xs(&mut restored, 300);
            restored.set_len(260).unwrap();
            let mut direct = xs_state(128, 1);
            feed_xs(&mut direct, 260);

            let pipeline = StubPipeline::new(2);
            let mut a = seq_with_slots(
                0,
                260,
                vec![
                    v4_kv_slot(512, 8, 260),
                    KvCache::XsRolling(Box::new(restored)),
                ],
            );
            let mut b = seq_with_slots(
                1,
                260,
                vec![
                    v4_kv_slot(512, 8, 260),
                    KvCache::XsRolling(Box::new(direct)),
                ],
            );
            let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
            NormalCacheManager
                .clone_in_cache(&pipeline, &mut seqs, false)
                .unwrap();

            let mut batched = pipeline.cache().normal().0.clone();
            let KvCache::XsRolling(xs) = &mut batched[1] else {
                panic!("slot 1 must stay an XsRolling entry")
            };
            assert_eq!(xs.rows(), 2, "precondition: this cache is batched");
            let err = xs
                .trim_tail_to(256)
                .expect_err("a scalar trim of a two-row cache must refuse")
                .to_string();
            assert!(
                err.contains("single-row") && err.contains("2 rows"),
                "the refusal must name why it cannot answer, got: {err}"
            );
        });
    }

    /// 🔴 `trim_tail_to` must drop the raw rows from the **front**.
    ///
    /// It raises `base`, so the tokens it discards are the OLDEST ones and the
    /// surviving window is the *suffix* of what was there. Narrowing from column
    /// 0 instead keeps the right *number* of columns holding the wrong tokens —
    /// every length, width and shape assertion in this file still passes, and
    /// the sequence silently resumes from history that is `drop` tokens stale.
    ///
    /// The other tests here feed zero-filled `xs`, so content is
    /// indistinguishable and none of them can see this. This one feeds a ramp
    /// (token `t` carries the value `t`) and reads the actual numbers back.
    #[test]
    fn trimming_the_retained_window_drops_the_oldest_rows_not_the_newest() {
        use candle_core::IndexOp;
        let dev = candle_core::Device::Cpu;
        let mut state = xs_state(128, 1);
        // Token `t` is the constant `t` across its hidden dim, so a column's
        // identity is readable straight off the tensor.
        // `f32::from(u16)` rather than `t as f32`: lossless by the type, so the
        // exact float comparisons below are exact by construction and not by a
        // range argument the reader has to make.
        let n = 260usize;
        let tok = |t: usize| f32::from(u16::try_from(t).expect("fixture is < 65536"));
        let ramp: Vec<f32> = (0..n)
            .flat_map(|t| std::iter::repeat_n(tok(t), XS_HIDDEN))
            .collect();
        let xs = Tensor::from_vec(ramp, (1usize, n, XS_HIDDEN), &dev).unwrap();
        state
            .advance(&xs, |w| {
                let rows = w.dim(1)? / 128;
                Tensor::zeros((1usize, rows, XS_HEAD_DIM), w.dtype(), w.device())
            })
            .unwrap();

        let base_before = state.resumable_from();
        let first_before: f32 = state
            .tail
            .as_ref()
            .unwrap()
            .i((0, 0, 0))
            .unwrap()
            .to_scalar()
            .unwrap();
        assert_eq!(
            first_before,
            tok(base_before),
            "precondition: column 0 holds token `base`, so the fixture can tell \
             the two ends apart"
        );

        let new_base = base_before + 4;
        state.trim_tail_to(new_base).unwrap();

        let tail = state.tail.as_ref().unwrap();
        assert_eq!(
            tail.dims()[1],
            260 - new_base,
            "precondition: the WIDTH is right either way — that is exactly why a \
             width assertion cannot see this bug"
        );
        let first_after: f32 = tail.i((0, 0, 0)).unwrap().to_scalar().unwrap();
        let last_after: f32 = tail
            .i((0, tail.dims()[1] - 1, 0))
            .unwrap()
            .to_scalar()
            .unwrap();
        assert_eq!(
            (first_after, last_after),
            (tok(new_base), tok(n - 1)),
            "the retained window must be the SUFFIX [new_base, tokens). Getting \
             ({base_before}, ..) means the narrow ran from column 0 and the \
             sequence would resume from stale history at the right width."
        );
    }

    /// `trim_tail_to` is lossless only up to what the compressor still needs,
    /// and must refuse — by name — past it, rather than silently dropping the
    /// rows the next compressed row is built from.
    #[test]
    fn trimming_the_retained_window_past_what_the_compressor_needs_is_refused() {
        let mut state = xs_state(4, 2);
        feed_xs(&mut state, 260);
        let needs_from = state.compressor_needs_from();
        assert!(
            needs_from >= state.resumable_from(),
            "an untouched cache always retains what it needs"
        );
        assert!(
            state.trim_tail_to(needs_from).is_ok(),
            "trimming to exactly the needed start is the lossless boundary"
        );
        let err = state
            .trim_tail_to(needs_from + 1)
            .expect_err("one token past it drops history the next row is built from")
            .to_string();
        assert!(
            err.contains("next compressed row is built from"),
            "the refusal must name what it would have destroyed, got: {err}"
        );
    }
}

#[cfg(test)]
mod turboquant_gate_tests {
    use super::EagerTurboQuantDecision::*;
    use super::*;

    fn reason(d: &EagerTurboQuantDecision) -> String {
        match d {
            Enabled(k, v) => panic!("expected Disabled, got Enabled({k}, {v})"),
            Disabled(r) => r.clone(),
        }
    }

    /// The head dims the shipped gate accepted, and the ones it silently
    /// refused. Every one of these must now be accepted — that is the whole
    /// change.
    #[test]
    fn every_real_head_dim_is_accepted_when_requested() {
        for d in [64usize, 80, 96, 112, 128, 192, 256, 512] {
            assert_eq!(
                resolve_eager_turboquant(d, d, true, false, Some("1")),
                Enabled(d, d),
                "head_dim {d} was refused"
            );
        }
        // Asymmetric K/V, which the paged path still cannot do.
        assert_eq!(
            resolve_eager_turboquant(192, 128, true, false, Some("1")),
            Enabled(192, 128)
        );
    }

    /// MUTATION GUARD — the refusals must each name their own mechanism, so a
    /// silent fallback is impossible to mistake for success.
    #[test]
    fn each_refusal_names_its_own_mechanism() {
        // Off by default: the eager path has no fused kernel.
        assert!(
            reason(&resolve_eager_turboquant(128, 128, true, false, None))
                .contains("ARC_TURBOQUANT_KV=1")
        );
        // Explicitly off.
        assert!(
            reason(&resolve_eager_turboquant(128, 128, true, false, Some("0")))
                .contains("set to off")
        );
        // A garbage value is refused rather than treated as truthy.
        assert!(reason(&resolve_eager_turboquant(
            128,
            128,
            true,
            false,
            Some("yes-please")
        ))
        .contains("not a recognised boolean"));
        // Paged wins: the KV is not in a NormalCache at all.
        assert!(
            reason(&resolve_eager_turboquant(128, 128, true, true, Some("1")))
                .contains("PagedAttention")
        );
        // MLA-style layouts have no independent K/V head vectors.
        assert!(
            reason(&resolve_eager_turboquant(128, 128, false, false, Some("1")))
                .contains("standard")
        );
        // A width narrower than one rotation block — V4's 1-wide V marker.
        assert!(
            reason(&resolve_eager_turboquant(512, 1, true, false, Some("1")))
                .contains("narrower than one rotation block")
        );
    }

    /// `new_plain` must ignore the gate entirely, because V4 depends on it,
    /// and `new` must take the branch at head dims the old gate refused.
    #[test]
    fn gate_selects_the_right_slot_kind() {
        // Serialise: the gate is process-wide.
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());

        set_turboquant_kv_head_dims(128, 128);
        let plain = NormalCache::new_plain(3, 4096);
        for slot in plain.lock().unwrap().0.iter() {
            assert!(
                matches!(slot, KvCache::Normal { .. }),
                "new_plain produced a compressed slot"
            );
        }
        // 128 was allowed by the old gate; 512 was not.
        for d in [128usize, 512] {
            set_turboquant_kv_head_dims(d, d);
            let c = NormalCache::new(2, 4096);
            for slot in c.lock().unwrap().0.iter() {
                assert!(
                    matches!(slot, KvCache::TurboQuant(_)),
                    "head_dim {d}: NormalCache::new did not take the TurboQuant branch"
                );
            }
        }
        // An unsupported geometry falls back rather than panicking.
        set_turboquant_kv_head_dims(512, 1);
        let c = NormalCache::new(2, 4096);
        for slot in c.lock().unwrap().0.iter() {
            assert!(matches!(slot, KvCache::Normal { .. }));
        }
        clear_turboquant_head_dim();
        assert_eq!(turboquant_head_dims(), None);
        // With the gate off, `new` is `new_plain`.
        let c = NormalCache::new(2, 4096);
        for slot in c.lock().unwrap().0.iter() {
            assert!(matches!(slot, KvCache::Normal { .. }));
        }
    }
}
