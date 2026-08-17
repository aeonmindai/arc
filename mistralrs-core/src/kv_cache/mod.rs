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
mod v4_turbo;
mod xs_rolling;

pub use full_cache::{EitherCache, LayerCaches};
pub use hybrid_cache::{
    HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    RecurrentStateSnapshot,
};
pub use rotating_cache::RotatingCache;
pub use single_cache::SingleCache;
pub use turboquant_cache::TurboQuantCache;
pub use v4_turbo::{V4TurboKCache, V4_TURBO_TAIL_MARGIN_TOKENS};
pub use xs_rolling::{XsRollingCache, XS_TAIL_MARGIN_TOKENS};

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
    /// DeepSeek V4's fused MQA keys in TurboQuant form (see [`V4TurboKCache`]).
    /// Like [`Self::XsRolling`] it is two regions with one boundary, reports
    /// its length in tokens, and is advanced through its own `append` rather
    /// than the generic K/V one.
    V4Turbo(Box<V4TurboKCache>),
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

    pub fn k(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { k, .. } => k.current_data(),
            Self::Rotating { k, .. } => k.current_data(),
            Self::TurboQuant(tq) => tq.k.current_data(),
            // The compressed rows are this entry's "keys".
            Self::XsRolling(xs) => xs.comp.current_data(),
            // The packed records are this entry's "keys".
            Self::V4Turbo(t) => t.codes.current_data(),
        }
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal { v, .. } => v.current_data(),
            Self::Rotating { v, .. } => v.current_data(),
            Self::TurboQuant(tq) => tq.v.current_data(),
            // The retained raw tail is this entry's "values".
            Self::XsRolling(xs) => Ok(xs.tail.clone()),
            // The retained dense window is this entry's "values".
            Self::V4Turbo(t) => Ok(t.tail.clone()),
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
            Self::V4Turbo(_) => candle_core::bail!(
                "KvCache::append: the V4 TurboQuant key cache is advanced through \
                 `V4TurboKCache::append`, which returns the dense span and its base — the \
                 generic two-tensor append cannot express that"
            ),
        };
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                match self {
                    Self::Normal { k, .. } => shape[k.dim] = 0,
                    Self::Rotating { k, .. } => shape[k.dim] = 0,
                    Self::TurboQuant(_) | Self::XsRolling(_) | Self::V4Turbo(_) => unreachable!(),
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
                    Self::TurboQuant(_) | Self::XsRolling(_) | Self::V4Turbo(_) => unreachable!(),
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
            Self::V4Turbo(t) => t.current_seq_len(),
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
            Self::V4Turbo(t) => {
                t.reset();
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
            Self::V4Turbo(t) => t.set_len(len),
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
            Self::V4Turbo(t) => t.try_set_len(len),
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
            .map(|slot| slot.as_ref().map(KvCache::current_seq_len))
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
}

impl BatchSrc {
    fn of(cache: &KvCache) -> Result<Self> {
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
            },
            // Compressed rows batch like K (a grown capacity buffer), the raw
            // tail like V (live content). Both are materialised by
            // `XsRollingCache::advance`, so a sequence that has been cloned out
            // at least once always has them.
            KvCache::XsRolling(xs) => Self {
                k: xs.comp.all_data.clone().ok_or_else(|| {
                    candle_core::Error::msg("xs rolling cache: compressed rows not materialised")
                })?,
                v: xs.tail.clone().ok_or_else(|| {
                    candle_core::Error::msg("xs rolling cache: raw tail not materialised")
                })?,
                k_slack_dim: Some(xs.comp.dim),
                v_slack_dim: None,
            },
            // Same split as `XsRolling`, and for the same reason: the packed
            // records are a grown capacity buffer, the dense tail is live
            // content whose width IS the sequence's position in its window.
            KvCache::V4Turbo(t) => Self {
                k: t.codes.all_data.clone().ok_or_else(|| {
                    candle_core::Error::msg("v4 turboquant kv cache: code buffer not materialised")
                })?,
                v: t.tail.clone().ok_or_else(|| {
                    candle_core::Error::msg("v4 turboquant kv cache: dense tail not materialised")
                })?,
                k_slack_dim: Some(t.codes.dim),
                v_slack_dim: None,
            },
        })
    }
}

/// Zero-extend `src` along `dim` to `width`. Used only for preallocation slack
/// (see [`BatchSrc`]), so the added columns are never read.
fn pad_slack(src: &Tensor, dim: usize, width: usize) -> Result<Tensor> {
    if src.dims()[dim] == width {
        return Ok(src.clone());
    }
    let mut shape = src.dims().to_vec();
    shape[dim] = width;
    let grown = Tensor::zeros(shape, src.dtype(), src.device())?;
    grown.slice_set(&src.contiguous()?, dim, 0)?;
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
fn reconcile_xs_bases(
    seqs: &mut [&mut crate::sequence::Sequence],
    layer: usize,
    modify_draft_cache: bool,
) -> Result<()> {
    let mut base_max = 0usize;
    let mut any = false;
    for seq in seqs.iter_mut() {
        let cache = if modify_draft_cache {
            seq.normal_draft_cache()
        } else {
            seq.normal_cache()
        };
        if let Some(KvCache::XsRolling(xs)) = cache.get(layer).and_then(|s| s.as_ref()) {
            base_max = base_max.max(xs.base);
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
        // 🔑 wave56-CG: this was a `debug_assert!`, i.e. absent from the release
        // binary that served on the H200. It now runs in release, and it
        // returns rather than panics, so a batch the scheduler should never
        // have formed costs the requests in it and not the engine task.
        ensure_uniform_batch_cache_lens(seqs, modify_draft_cache)?;

        let _prof = arc_profiler::span("clone_in_cache");
        let mut new_k_cache = Vec::new();
        let mut new_v_cache = Vec::new();

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
            reconcile_xs_bases(seqs, layer, modify_draft_cache)?;

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
                    Some(cache) => srcs.push(Some(BatchSrc::of(cache)?)),
                    None => srcs.push(None),
                }
            }

            let present: Vec<&BatchSrc> = srcs.iter().flatten().collect();
            let k_slack = present[0].k_slack_dim;
            let v_slack = present[0].v_slack_dim;
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
                    Some(d) => pad_slack(&src.k, d, one_k[d])?,
                    None => src.k.clone(),
                };
                let src_v = match v_slack {
                    Some(d) => pad_slack(&src.v, d, one_v[d])?,
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
                    // Everything except the two buffers (token count, base,
                    // completed-row count, ratio) is per-batch metadata taken
                    // from the seq0 template. That is only sound because
                    // `ensure_uniform_batch_cache_lens` has already established
                    // that every sequence agrees on `tokens` for this slot —
                    // which is exactly what makes `tail` (`tokens - base` wide)
                    // batchable at all.
                    let mut rebuilt = (**xs).clone();
                    if let Some(k) = k_cache.as_ref() {
                        if rebuilt.comp.dim != 0 {
                            rebuilt.comp.capacity_seq_len = k.dims()[rebuilt.comp.dim];
                        }
                    }
                    rebuilt.comp.all_data = k_cache;
                    rebuilt.tail = v_cache;
                    caches.push(KvCache::XsRolling(Box::new(rebuilt)));
                }
                KvCache::V4Turbo(t) => {
                    // As for `XsRolling`: everything but the two buffers is
                    // per-batch metadata from the seq0 template, sound only
                    // because `ensure_uniform_batch_cache_lens` has already
                    // established that every sequence agrees on `tokens` here
                    // — which is what makes the `tokens - base` wide dense tail
                    // batchable at all.
                    let mut rebuilt = (**t).clone();
                    if let Some(k) = k_cache.as_ref() {
                        rebuilt.codes.capacity_seq_len = k.dims()[rebuilt.codes.dim];
                    }
                    rebuilt.codes.all_data = k_cache;
                    rebuilt.tail = v_cache;
                    caches.push(KvCache::V4Turbo(Box::new(rebuilt)));
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
                KvCache::V4Turbo(t) => (
                    t.codes
                        .all_data
                        .clone()
                        .expect("v4 turboquant kv cache: code buffer not materialised"),
                    t.tail
                        .clone()
                        .expect("v4 turboquant kv cache: dense tail not materialised"),
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
                        let mut per_seq = (**xs).clone();
                        per_seq.comp.all_data = Some(k);
                        per_seq.tail = Some(v);
                        *seq_cache = Some(KvCache::XsRolling(Box::new(per_seq)));
                    }
                    KvCache::V4Turbo(t) => {
                        let mut per_seq = (**t).clone();
                        per_seq.codes.all_data = Some(k);
                        per_seq.tail = Some(v);
                        *seq_cache = Some(KvCache::V4Turbo(Box::new(per_seq)));
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
                // Unreachable for the same reason as `XsRolling`: the V4
                // TurboQuant slot has no preallocated KV-shaped buffer (its
                // code records are a different width and dtype from the dense
                // keys this preallocation is sized for). Reset rather than
                // panic, so a future reordering degrades to "start empty".
                KvCache::V4Turbo(_) => {
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
    use tokio::sync::Mutex as TokioMutex;

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
        let group = Arc::new(TokioMutex::new(SequenceGroup::new(1, false, false, None)));
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
    #[test]
    fn clone_in_cache_refuses_a_length_mismatched_batch() {
        let pipeline = StubPipeline::new(4);
        let mut a = seq_with_cache_len(0, 4, 100);
        let mut b = seq_with_cache_len(1, 4, 200);
        let mut seqs: Vec<&mut Sequence> = vec![&mut a, &mut b];
        let err = NormalCacheManager
            .clone_in_cache(&pipeline, &mut seqs, false)
            .expect_err("a length-mismatched batch must be refused")
            .to_string();
        assert!(err.contains("must share current_seq_len"), "got: {err}");
        assert!(err.contains("100") && err.contains("200"), "got: {err}");
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
            restored.base, 256,
            "canonical(300) for ratio 128, margin 16"
        );
        restored.set_len(260).unwrap();

        // Reached 260 directly.
        let mut direct = xs_state(128, 1);
        feed_xs(&mut direct, 260);
        assert_eq!(direct.base, 128, "canonical(260) sits a group lower");

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
            xs.base, 256,
            "trimmed to the batch's largest retained start"
        );
        assert_eq!(xs.tail.as_ref().unwrap().dims(), &[2, 4, XS_HIDDEN]);
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
            needs_from >= state.base,
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
