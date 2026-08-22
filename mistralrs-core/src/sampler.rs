#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, LazyLock, Mutex},
};

use candle_core::{DType, Device, Error, Result, Tensor, D};
use mistralrs_quant::{CumSumOp, SortOp};
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use rand::distr::{weighted::WeightedIndex, Distribution};
use rand_isaac::Isaac64Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

static DRY_SEQUENCE_BREAKERS: LazyLock<Vec<String>> =
    LazyLock::new(|| ["\n", ":", "\"", "*"].map(String::from).to_vec());

/// Health counters for the big-vocab GPU sampling path.
///
/// Falling back to the CPU sampler costs a full-logits-row D2H plus a
/// full-vocab sort, **per sequence per decode step**. That is a throughput
/// cliff, not a warning-level event — but it used to surface only as a
/// per-token `tracing::warn!`, so a 100%-failure condition (the missing I32
/// arm in `tensor_device_ptr`) shipped and ran in production unnoticed.
///
/// These counters exist so "we are on the slow path" is a number the interval
/// logger prints, not something buried in a log nobody reads.
pub mod gpu_sampling_health {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(super) static GPU_OK: AtomicU64 = AtomicU64::new(0);
    /// The GPU path declined by design (`Ok(None)`) — e.g. `top_k` above the
    /// kernel's dispatch cap. Expected, config-driven, not a defect.
    pub(super) static DECLINED: AtomicU64 = AtomicU64::new(0);
    /// The GPU path errored (`Err`). Always a defect.
    pub(super) static FAILED: AtomicU64 = AtomicU64::new(0);

    /// Cumulative `(gpu_ok, declined, failed)` since process start.
    pub fn stats() -> (u64, u64, u64) {
        (
            GPU_OK.load(Ordering::Relaxed),
            DECLINED.load(Ordering::Relaxed),
            FAILED.load(Ordering::Relaxed),
        )
    }

    /// Reset all counters. Called after warmup so steady-state numbers are clean.
    pub fn reset() {
        GPU_OK.store(0, Ordering::Relaxed);
        DECLINED.store(0, Ordering::Relaxed);
        FAILED.store(0, Ordering::Relaxed);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Stop sequences or ids.
pub enum StopTokens {
    Seqs(Vec<String>),
    Ids(Vec<u32>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Sampling params are used to control sampling.
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    /// Top-nσ sampling (Tang et al. 2024, arXiv:2411.07641): keep only tokens
    /// whose logit is within `n * std_dev` of the maximum logit. Applied on the
    /// raw pre-temperature logits, which makes the kept set provably invariant
    /// to temperature (dividing logits by `T` scales max-gap and σ equally).
    /// `n -> 0` reduces to greedy; larger `n` admits more of the tail.
    pub top_nsigma: Option<f32>,
    pub top_n_logprobs: usize,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub stop_toks: Option<StopTokens>,
    pub max_len: Option<usize>,
    pub logits_bias: Option<HashMap<u32, f32>>,
    pub n_choices: usize,
    pub dry_params: Option<DrySamplingParams>,
    /// Arc Boost (DeepConf-low, simplified): cull this request's sibling vote
    /// chains whose lowest-group confidence falls below `best / frac` in log
    /// space, where `best` is the best sibling chain's lowest-group confidence.
    /// Only meaningful with `n_choices > 1`; see `crate::arc_boost`.
    pub early_stop_confidence: Option<f32>,
    /// Arc Boost budget policy: cap "thinking" tokens at this many generated
    /// tokens. Where a `<think>` structure is active and the chat template's
    /// end-think token is known, the cap is graceful: the end-think token is
    /// injected so the model wraps up and still emits a final answer.
    /// Otherwise this degrades to a hard `max_len`-style cap.
    ///
    /// Empirical basis (Arc GPU session 1, GSM8K n=50): easy-math accuracy
    /// saturates by ~256 thinking tokens, and hard truncation (640-token cap,
    /// 33/50 truncated) accounted for roughly half of the observed GSM8K loss
    /// — hence graceful wrap-up instead of hard truncation.
    pub reasoning_budget: Option<usize>,
}

impl SamplingParams {
    /// This sets up the parameters so that there is:
    /// - No temperature, topk, topp, minp
    /// - No penalties, stop tokens, or logit bias
    /// - No maximum length
    pub fn deterministic() -> Self {
        Self {
            temperature: None,
            top_k: Some(1),
            top_p: None,
            min_p: None,
            top_nsigma: None,
            top_n_logprobs: 0,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop_toks: None,
            max_len: None,
            logits_bias: None,
            n_choices: 1,
            dry_params: None,
            early_stop_confidence: None,
            reasoning_budget: None,
        }
    }
}

/// Parameters for DRY (Don't Repeat Yourself) sampling to reduce repetition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrySamplingParams {
    pub sequence_breakers: Vec<String>,
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
}

impl DrySamplingParams {
    pub fn new_with_defaults(
        multiplier: f32,
        sequence_breakers: Option<Vec<String>>,
        base: Option<f32>,
        allowed_length: Option<usize>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            base: base.unwrap_or(1.75),
            allowed_length: allowed_length.unwrap_or(2),
            sequence_breakers: sequence_breakers.unwrap_or(DRY_SEQUENCE_BREAKERS.clone()),
            multiplier,
        })
    }
}

impl Default for DrySamplingParams {
    fn default() -> Self {
        Self {
            multiplier: 0.0,
            base: 1.75,
            allowed_length: 2,
            sequence_breakers: DRY_SEQUENCE_BREAKERS.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct DrySamplingParamsInner {
    pub sequence_breakers: HashSet<u32>,
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
}

impl DrySamplingParamsInner {
    pub fn from(other: DrySamplingParams, tokenizer: &Tokenizer) -> anyhow::Result<Self> {
        Ok(Self {
            base: other.base,
            allowed_length: other.allowed_length,
            sequence_breakers: HashSet::from_iter(
                other
                    .sequence_breakers
                    .into_iter()
                    .map(|breaker| {
                        tokenizer
                            // Prefix with 'a' to get the correct encoding of the token at the end of a text.
                            //
                            // FIXME: This is a hack. See https://github.com/LostRuins/koboldcpp/pull/982
                            //        for the correct solution which covers multi-token sequence breakers
                            //        and ambiguous encodings.
                            .encode_fast(["a", &breaker].concat(), true)
                            .map_err(anyhow::Error::msg)
                            .map(|enc| {
                                let ids = enc.get_ids();
                                if !ids.is_empty() {
                                    Some(ids[ids.len() - 1])
                                } else {
                                    None
                                }
                            })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
            ),
            multiplier: other.multiplier,
        })
    }
}

/// Customizable logits processor.
///
/// # Example
/// ```rust
/// use std::{sync::Arc, ops::Mul};
/// use mistralrs_core::CustomLogitsProcessor;
/// use candle_core::{Result, Tensor};
///
/// struct ThresholdLogitsProcessor;
/// impl CustomLogitsProcessor for ThresholdLogitsProcessor {
///     fn apply(&self, logits: &Tensor, _context: &[u32]) -> Result<Tensor> {
///         // Mask is 1 for true, 0 for false.
///         let mask = logits.ge(0.5)?;
///         logits.broadcast_mul(&mask.to_dtype(logits.dtype())?)
///     }
/// }
/// let processor1: Arc<dyn CustomLogitsProcessor> = Arc::new(|logits: &Tensor, _context: &[u32]| logits * 1.23);
/// let processor2: Arc<dyn CustomLogitsProcessor> = Arc::new(ThresholdLogitsProcessor);
/// ```
pub trait CustomLogitsProcessor: Send + Sync {
    /// Logits and sequence context (prompt and generated tokens), returning modified tokens.
    fn apply(&self, logits: &Tensor, context: &[u32]) -> Result<Tensor>;
}

impl<T: Fn(&Tensor, &[u32]) -> Result<Tensor> + Send + Sync> CustomLogitsProcessor for T {
    fn apply(&self, logits: &Tensor, context: &[u32]) -> Result<Tensor> {
        self(logits, context)
    }
}

/// Sampler for sampling.
#[derive(Clone)]
pub struct Sampler {
    temperature: Option<f64>,
    top_n_logprobs: usize,
    tokenizer: Option<Arc<Tokenizer>>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    dry_params: Option<DrySamplingParamsInner>,
    top_k: i64,
    top_p: f64,
    min_p: f64,
    top_nsigma: Option<f32>,
    logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    /// OpenAI `logit_bias`: token id -> additive bias on the raw logits.
    logits_bias: Option<Arc<HashMap<u32, f32>>>,
    /// `logits_bias` materialised as a dense vector on the device/dtype of the
    /// logits it was last applied to.
    ///
    /// Caching is sound here in a way it is not for sampling noise: the dense
    /// vector is a pure function of `logits_bias`, which is immutable for the
    /// life of the `Sampler`, and it is rebuilt whenever the incoming logits'
    /// device, dtype or vocab differs from the cached tensor's. Without it,
    /// every decoded token would pay a host allocation plus an H2D copy of the
    /// full vocabulary.
    logits_bias_dense: Arc<Mutex<Option<Tensor>>>,
}

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Top-n logprobs element
pub struct TopLogprob {
    pub token: u32,
    pub logprob: f32,
    pub bytes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logprobs {
    pub token: u32,
    pub logprob: f32,
    pub bytes: Option<String>,
    pub top_logprobs: Option<Vec<TopLogprob>>,
}

/// Comparator for descending order by probability (second element of tuple).
#[inline]
fn cmp_desc_by_prob(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
}

/// Returns the top-k (index, probability) pairs from `probs`, sorted in descending order.
/// Uses partial sort (O(n) + O(k log k)) instead of full sort (O(n log n)).
///
/// If `k >= probs.len()`, returns all elements sorted.
/// Also zeros out elements in `probs` beyond top-k if `zero_rest` is true.
fn partial_sort_top_k(probs: &mut [f32], k: usize, zero_rest: bool) -> Vec<(u32, f32)> {
    #[cfg(test)]
    PARTIAL_SORT_CALLS.with(|calls| calls.set(calls.get() + 1));
    let n = probs.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    // Build (index, probability) pairs
    let mut idx_probs: Vec<(u32, f32)> = (0..n as u32).map(|i| (i, probs[i as usize])).collect();

    let k = k.min(n);

    if k < n {
        // Partial sort: partition so top k elements are in first k positions
        // select_nth_unstable_by places the k-1th largest at position k-1,
        // with all larger elements before it (unsorted) and smaller after
        idx_probs.select_nth_unstable_by(k - 1, cmp_desc_by_prob);

        if zero_rest {
            // Zero out elements beyond top-k
            for (idx, _) in idx_probs[k..].iter() {
                probs[*idx as usize] = 0.0;
            }
        }

        // Truncate to top k
        idx_probs.truncate(k);
    }

    // Sort just the top k elements (descending by probability)
    idx_probs.sort_unstable_by(cmp_desc_by_prob);

    idx_probs
}

/// Invocation counter for [`partial_sort_top_k`], test builds only. Exists so
/// the sort-skip in [`Sampler::sample_top_kp_min_p`] is provable: a test can
/// assert the sort genuinely did not run, not merely that the output looks
/// the same. Thread-local, because `cargo test` runs tests concurrently and
/// several other tests in this file sort — a process-global counter would make
/// every delta assertion racy.
#[cfg(test)]
thread_local! {
    pub(crate) static PARTIAL_SORT_CALLS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

/// Kernel dispatch sizes for the GPU radix top-k sampler path. Must mirror
/// `arc_cuda_graph::flashmlasparse::SUPPORTED_TOPK` (kept as a local const so
/// the candidate-truncation logic and its CPU-parity tests build without the
/// `cuda` feature; the CUDA wrapper re-validates and errors — triggering the
/// CPU fallback — if the tables ever diverge).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const GPU_RADIX_TOPK_SIZES: &[usize] = &[64, 128, 256, 512, 1024];

/// Find the index of the maximum element in a slice. O(n) scan.
#[inline]
/// Index of the maximum value, resolving ties to the **lowest** index.
///
/// # Why this is not `max_by`
///
/// It was. `Iterator::max_by` is documented to return the **last** element when
/// several compare equal, so on tied logits this returned the highest-indexed
/// token — the opposite of the argmax convention every other implementation in
/// the stack and outside it (numpy, PyTorch) follows, which is first-wins.
///
/// That is not academic here, because Arc has three greedy implementations and
/// before this fix all three disagreed on ties:
///
/// | implementation | tie goes to |
/// |---|---|
/// | this, via `max_by` | **highest** index |
/// | candle's `fast_argmax` (`candle-kernels/src/reduce.cu:475-521`) | **whichever thread id is lower**, which is not an index order at all |
/// | `arc_greedy_kernel` (`arc-cuda-graph/src/cuda/sampling_kernel.cu:185-188`) | **lowest** index |
///
/// candle's is the subtle one: its per-thread scan keeps the first max within a
/// thread's own stride, but the tree reduction compares values only and keeps
/// the lower thread id on a tie. Thread `t` scans indices `t, t+B, t+2B, ...`,
/// so a tie between thread 0 (holding index 128) and thread 5 (holding index 5)
/// resolves to **128**. Deterministic, but arbitrary.
///
/// Ties are not rare at this vocabulary size. V4's logits arrive as BF16 —
/// 8 mantissa bits, ~256 distinct values per binade — and the F32 cast in
/// `sampling.rs` is exact, so every tie in the model's output survives into
/// this function across 129,280 candidates.
///
/// Fixing the tie rule here is the half that can be proved without a GPU. It
/// makes the CPU path agree with `arc_greedy_kernel`; candle's CUDA argmax
/// still disagrees with both, which is a reason to move the GPU greedy path
/// onto `arc_greedy_kernel` rather than a reason to leave this alone.
///
/// NaN is skipped rather than compared. A naive strict-`>` scan seeded with the
/// first element would let a NaN at index 0 poison every later comparison
/// (`v > NaN` is false for all `v`) and return 0 regardless of the real
/// maximum. Empty and all-NaN inputs yield 0, the same fallback as before.
fn argmax_f32(values: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best: Option<f32> = None;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        // Strictly greater, so the first of any tied run wins.
        if best.is_none_or(|b| v > b) {
            best = Some(v);
            best_idx = i as u32;
        }
    }
    best_idx
}

impl Sampler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        temperature: Option<f64>,
        top_n_logprobs: usize,
        tokenizer: Option<Arc<Tokenizer>>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        repetition_penalty: Option<f32>,
        dry_params: Option<DrySamplingParams>,
        top_k: i64,
        top_p: f64,
        min_p: f64,
        top_nsigma: Option<f32>,
        logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
        logits_bias: Option<HashMap<u32, f32>>,
    ) -> anyhow::Result<Self> {
        let temperature = if temperature.is_none_or(|v| v < 1e-7) {
            None
        } else {
            temperature
        };
        // Negative n is meaningless (it would filter out even the argmax).
        let top_nsigma = top_nsigma.filter(|n| *n >= 0.0 && n.is_finite());
        let dry_params = if let Some(ref tokenizer) = tokenizer {
            dry_params.map(|params| DrySamplingParamsInner::from(params, tokenizer))
        } else {
            None
        };
        let dry_params = match dry_params {
            Some(fallible) => Some(fallible?),
            None => None,
        };
        Ok(Self {
            temperature,
            top_n_logprobs,
            tokenizer,
            frequency_penalty,
            presence_penalty,
            repetition_penalty,
            dry_params,
            top_k,
            top_p,
            min_p,
            top_nsigma,
            logits_processors,
            logits_bias: logits_bias.filter(|b| !b.is_empty()).map(Arc::new),
            logits_bias_dense: Arc::new(Mutex::new(None)),
        })
    }

    /// Effective temperature. `None` means greedy (temperature was <1e-7).
    pub fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    /// Effective top_p (0.0..=1.0). Values >=1.0 mean no top_p filtering.
    pub fn top_p(&self) -> f64 {
        self.top_p
    }

    /// Effective top_k. Values <=0 mean no top_k filtering.
    pub fn top_k(&self) -> i64 {
        self.top_k
    }

    /// Top-nσ threshold (None means disabled). The autonomous-decode GPU
    /// sampler does not implement this filter, so callers should refuse the
    /// autonomous fast-path when this returns Some.
    pub fn top_nsigma(&self) -> Option<f32> {
        self.top_nsigma
    }

    /// Frequency penalty (None means disabled).
    pub fn frequency_penalty(&self) -> Option<f32> {
        self.frequency_penalty
    }

    /// Presence penalty (None means disabled).
    pub fn presence_penalty(&self) -> Option<f32> {
        self.presence_penalty
    }

    /// True if this sampler is effectively greedy (no temperature → argmax).
    pub fn is_greedy(&self) -> bool {
        self.temperature.is_none()
    }

    /// True if any logits_processors are registered. The autonomous-decode
    /// GPU sampler cannot apply custom CPU-side processors, so callers
    /// should refuse autonomous fast-path when this returns true.
    pub fn has_custom_logits_processors(&self) -> bool {
        !self.logits_processors.is_empty()
    }

    /// True when a penalty would rewrite the logits before the token is picked.
    ///
    /// Single source of truth for the condition: [`Self::sample`]'s GPU fast
    /// path reads it as `!has_penalties()`, and [`Self::is_raw_argmax`] reads it
    /// directly. Two copies of this expression could disagree, and the one that
    /// disagreed would be a silent correctness bug rather than a slow path.
    fn has_penalties(&self) -> bool {
        // DRY counts only when it can actually rewrite a logit. The disabled
        // default (`DrySamplingParams::default()`) is `multiplier: 0.0`, and
        // `apply_dry_penalty` returns without touching the logits when the
        // multiplier is exactly 0.0 — so `dry_params.is_some()` alone was
        // disqualifying the GPU fast path (and failing `is_raw_argmax`) for a
        // penalty that is a proven no-op.
        self.frequency_penalty.unwrap_or(0.0) != 0.0
            || self.presence_penalty.unwrap_or(0.0) != 0.0
            || self.repetition_penalty.unwrap_or(1.0) != 1.0
            || self
                .dry_params
                .as_ref()
                .is_some_and(|params| params.multiplier != 0.0)
    }

    /// Would [`Sampler::sample`] take the pure-argmax path — no temperature and
    /// nothing that has to touch the logits on the host?
    ///
    /// This is the exact precondition for ArcGraph's device decode loop to
    /// substitute its on-device argmax for the host one
    /// (`pipeline/sampling.rs`). It deliberately mirrors the conditions inside
    /// `sample` rather than approximating them: `logits_bias` and `top_nsigma`
    /// are applied *before* the `trivial` gate, so a sampler carrying either
    /// still needs the host even though the gate itself ignores them.
    ///
    /// Greedy is also the only mode where device and host agree *exactly* — the
    /// device sampler runs Splitmix64 per row against the host's Isaac64, so
    /// any stochastic mode would draw a different, equally valid token and
    /// silently break seeded reproducibility.
    pub fn is_greedy_trivial(&self) -> bool {
        self.temperature.is_none()
            && self.logits_processors.is_empty()
            && !self.has_penalties()
            && self.logits_bias.is_none()
            && self.top_nsigma.is_none()
    }

    /// True when this sampler is exactly `argmax` over the model's **raw**
    /// logits — no temperature, no penalties, no custom logits processors.
    ///
    /// This is the precondition for any fast path that substitutes its own
    /// `argmax` for [`Self::sample`], which is what speculative verification
    /// does: `MtpSpeculativePipeline` accepts a draft token when it equals
    /// `argmax(target_logits)`, computed on the raw logits.
    ///
    /// [`Self::is_greedy`] alone is **not** that precondition. With
    /// `temperature: None` and any penalty or processor set, [`Self::sample`]
    /// still runs `apply_penalties` and every `logits_processor` *before*
    /// `sample_argmax` (`sampler.rs`, the CPU path), so the token it returns can
    /// differ from `argmax(raw)`. Top-k / top-p / min-p do **not** need gating:
    /// they are only consulted on the `Some(temperature)` branch. Nor does
    /// top-nσ: it masks tokens *below* the maximum, so the argmax is invariant
    /// under it.
    ///
    /// **`logit_bias` also disqualifies**, and this is a merge hazard worth
    /// naming: `is_raw_argmax` landed on master while `logit_bias` was in
    /// flight, so neither author saw the interaction. [`Self::sample`] applies
    /// the bias to the raw logits *first* (`self.apply_logits_bias(logits)?`,
    /// the first statement of the sampling body), so with a bias set the token
    /// `sample` returns is `argmax(logits + bias)`, which is not
    /// `argmax(logits)`. A speculative verifier that trusted `is_raw_argmax`
    /// here would accept draft tokens judged against unbiased logits while the
    /// real sampler used biased ones — a silent, request-specific wrong-token
    /// bug, not a slow path.
    pub fn is_raw_argmax(&self) -> bool {
        self.is_greedy()
            && !self.has_penalties()
            && !self.has_logits_bias()
            && self.logits_processors.is_empty()
    }

    /// Mirror of [`Self::sample`]'s device fast-path dispatch, answerable
    /// **before** the logits row has been copied anywhere: would a non-CPU row
    /// of `vocab` logits be sampled to completion on the device (token-sized
    /// D2H), rather than falling back to the CPU pipeline (full-row D2H +
    /// full-vocab host work)?
    ///
    /// This is what `ARC_SAMPLE_ON_DEVICE`'s host-copy partitioning
    /// (`pipeline/mod.rs`) consults per sequence: rows answering `true` keep
    /// their device residency; rows answering `false` are gathered into ONE
    /// batched D2H. It must stay in this file, next to the gates it mirrors
    /// ([`Self::has_penalties`] and `sample`'s `trivial` condition): a second
    /// copy that drifted would silently route rows onto the wrong side of the
    /// PCIe bus.
    ///
    /// Assumes non-speculative sampling (`sample_speculative = false`), which
    /// is what every `sample_causal_gen` reachable from the partitioned step
    /// arm passes; the speculative pipelines override `step` entirely and
    /// never see the partition.
    ///
    /// `logit_bias` and top-nσ do **not** disqualify: both are applied as
    /// device tensor ops before the fast-path dispatch. One residual runtime
    /// fallback remains on the CUDA radix path (a pure-top-p nucleus the
    /// candidate set cannot contain declines with `Ok(None)`); it pays a
    /// per-row D2H and is counted by `gpu_sampling_health::DECLINED`.
    pub(crate) fn device_sampling_terminates(
        &self,
        vocab: usize,
        device_is_cuda: bool,
        return_logprobs: bool,
    ) -> bool {
        // `sample`'s `trivial` gate, minus its device term — the caller is
        // deciding device residency, not reacting to it.
        let trivial =
            !return_logprobs && self.logits_processors.is_empty() && !self.has_penalties();
        if !trivial {
            return false;
        }
        // Greedy: a single argmax kernel + 4-byte D2H. Any backend, any vocab.
        if self.temperature.is_none() {
            return true;
        }
        // Temperature path: `sample_fast` stays on device when no sort is
        // needed, or when candle's shared-memory arg_sort can hold the vocab
        // (same bound `sample` checks).
        let needs_sort = self.top_k > 0 || (self.top_p > 0.0 && self.top_p < 1.0);
        if !needs_sort {
            return true;
        }
        if vocab.next_power_of_two().saturating_mul(4) <= 48 * 1024 {
            return true;
        }
        // Big-vocab sort: only the CUDA radix top-k path avoids the CPU
        // fallback, and only when the requested k fits its dispatch table
        // (`sample_fast_topk_gpu`'s own static precondition).
        let max_k = *GPU_RADIX_TOPK_SIZES.last().unwrap();
        let k_needed = if self.top_k > 0 {
            self.top_k as usize
        } else {
            max_k
        };
        cfg!(feature = "cuda") && device_is_cuda && k_needed <= max_k && vocab > max_k
    }

    /// True if an OpenAI `logit_bias` map is set. Any sampling path that does
    /// not route through [`Self::sample`] — notably the GPU-autonomous decode
    /// sampler, which draws from the model's raw logits — must refuse the
    /// request when this returns true, or the bias is silently dropped.
    pub fn has_logits_bias(&self) -> bool {
        self.logits_bias.is_some()
    }

    /// Apply the OpenAI `logit_bias` map: `logits[token] += bias`, on the raw
    /// logits, before any filtering or sampling.
    ///
    /// Matches the reference implementations — vLLM's
    /// `LogitBiasLogitsProcessor::apply` is `logits[slice] += bias_tensor` and
    /// SGLang's is a single `logits.add_()` — including the absence of any
    /// clamping of the biased value.
    fn apply_logits_bias(&self, logits: Tensor) -> Result<Tensor> {
        let Some(bias) = self.logits_bias.as_ref() else {
            return Ok(logits);
        };
        let vocab = logits.dim(D::Minus1)?;

        let mut guard = self
            .logits_bias_dense
            .lock()
            .expect("could not lock logits_bias cache");
        let dense = match guard.as_ref() {
            // Reuse only against logits the cached tensor actually matches.
            Some(cached)
                if cached.device().same_device(logits.device())
                    && cached.dtype() == logits.dtype()
                    && cached.dims1().is_ok_and(|n| n == vocab) =>
            {
                cached.clone()
            }
            _ => {
                let mut host = vec![0f32; vocab];
                let mut out_of_range = 0usize;
                for (token, b) in bias.iter() {
                    match host.get_mut(*token as usize) {
                        Some(slot) => *slot += *b,
                        None => out_of_range += 1,
                    }
                }
                if out_of_range > 0 {
                    // Loud rather than silent: the caller asked to bias tokens
                    // this model cannot emit. Runs at most once per sampler,
                    // since the dense vector is then cached.
                    tracing::warn!(
                        "logit_bias: {out_of_range} token id(s) are outside the model's \
                         vocabulary of {vocab} and were ignored"
                    );
                }
                let dense =
                    Tensor::from_vec(host, vocab, logits.device())?.to_dtype(logits.dtype())?;
                *guard = Some(dense.clone());
                dense
            }
        };
        drop(guard);

        logits.broadcast_add(&dense)
    }

    fn get_top_logprobs(&self, probs: &[f32]) -> Result<Vec<TopLogprob>> {
        let k = self.top_n_logprobs.min(probs.len());
        if k == 0 {
            return Ok(Vec::new());
        }

        // Use partial sort helper (doesn't modify probs since we pass a copy)
        let mut probs_copy = probs.to_vec();
        let top_k = partial_sort_top_k(&mut probs_copy, k, false);

        // Build the result vector with log10 of probabilities and optional decoding
        let mut result = Vec::with_capacity(k);
        if let Some(tokenizer) = &self.tokenizer {
            for (token, prob) in top_k {
                let decoded = tokenizer
                    .decode(&[token], false)
                    .map_err(|e| Error::Msg(e.to_string()))?;
                result.push(TopLogprob {
                    token,
                    logprob: prob.log(10.0),
                    bytes: Some(decoded),
                });
            }
        } else {
            for (token, prob) in top_k {
                result.push(TopLogprob {
                    token,
                    logprob: prob.log(10.0),
                    bytes: None,
                });
            }
        }
        Ok(result)
    }

    fn sample_argmax(&self, logits: Tensor, return_logprobs: bool) -> Result<Logprobs> {
        let probs: Vec<f32> = logits.to_vec1()?;
        let next_token = argmax_f32(&probs);
        let logprob = probs[next_token as usize].log(10.0);

        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(&probs)?)
        } else {
            None
        };

        let bytes = if let Some(tokenizer) = &self.tokenizer {
            Some(
                tokenizer
                    .decode(&[next_token], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            )
        } else {
            None
        };

        Ok(Logprobs {
            token: next_token,
            logprob,
            top_logprobs,
            bytes,
        })
    }

    #[allow(unused)]
    fn sample_fast(
        &self,
        logits: Tensor,
        context: &[u32],
        return_logprobs: bool,
        top_k: i64,
        top_p: f64,
        min_p: f64,
    ) -> Result<Logprobs> {
        let mut probs = logits.to_dtype(DType::F32)?;

        for processor in &self.logits_processors {
            probs = processor.apply(&probs, context)?;
        }

        // The penalty prologue — the context upload, the `counts` histogram and
        // the `presence` mask — feeds nothing except the three `match` arms
        // below. When no penalty is actually configured all three arms are
        // no-ops, so every one of those tensors was computed and discarded.
        //
        // That was not a cheap nothing. It cost, per sequence per decoded
        // token: one *blocking* H2D upload of the sequence's entire token
        // history (`Tensor::new` on a CUDA device — CLAUDE.md pitfall #5 — and
        // it grows as the sequence generates), plus roughly eight vocab-sized
        // kernels (`zeros_like`, `ones_like`, `to_dtype`, `scatter_add`, `gt`,
        // two more `*_like`, `where_cond`) and their allocations. At a 129K
        // vocab that is ~3 MB of pointless device traffic and a full host
        // round trip per sequence per token — on the *fast* path, the one
        // taken by the common no-penalty request.
        //
        // Gate the whole prologue on a penalty being live. When one is, the
        // computation and its order are untouched, so results are unchanged;
        // when none is, the removed tensors had no consumer, so results are
        // bit-identical.
        let freq_active = matches!(self.frequency_penalty, Some(p) if p != 0.);
        let pres_active = matches!(self.presence_penalty, Some(p) if p != 0.);
        let rep_active = matches!(self.repetition_penalty, Some(p) if p != 1.);

        if freq_active || pres_active || rep_active {
            let context = Tensor::new(context, logits.device())?;
            let mut counts = logits.zeros_like()?;
            counts = counts.scatter_add(
                &context,
                &context.ones_like()?.to_dtype(counts.dtype())?,
                D::Minus1,
            )?;

            let presence = counts
                .gt(0.)?
                .where_cond(&counts.ones_like()?, &counts.zeros_like()?)?;

            match self.frequency_penalty {
                Some(freq_penalty) if freq_penalty != 0. => {
                    probs = (probs - (freq_penalty as f64 * counts)?)?;
                }
                _ => (),
            }

            match self.presence_penalty {
                Some(pres_penalty) if pres_penalty != 0. => {
                    probs = (probs - (pres_penalty as f64 * &presence)?)?;
                }
                _ => (),
            }

            match self.repetition_penalty {
                Some(rep_penalty) if rep_penalty != 1. => {
                    let pos_mask = probs.gt(0.)?;
                    let scaled_pos = (&probs / (rep_penalty as f64))?;
                    let scaled_neg = (&probs * (rep_penalty as f64))?;
                    let modified = pos_mask.where_cond(&scaled_pos, &scaled_neg)?;

                    let pres_mask = presence.gt(0.)?;
                    probs = pres_mask.where_cond(&modified, &probs)?;
                }
                _ => (),
            }
        }

        probs = candle_nn::ops::softmax_last_dim(&(probs / self.temperature.unwrap_or(1.))?)?;

        // Top-K
        if top_k > 0 {
            let sorted_values = probs.fast_sort_asc(D::Minus1)?;
            let topk_values = sorted_values.narrow(
                D::Minus1,
                sorted_values.dim(D::Minus1)? - top_k as usize,
                top_k as usize,
            )?;

            // select the kth largest value as threshold
            let threshold = topk_values.get_on_dim(D::Minus1, 0)?.unsqueeze(0)?;
            let mask_topk = probs.broadcast_ge(&threshold)?;
            probs = mask_topk.where_cond(&probs, &Tensor::zeros_like(&probs)?)?;
        }

        // Top-P (nucleus)
        if top_p > 0.0 && top_p < 1.0 {
            let sorted_probs = probs.fast_sort_asc(D::Minus1)?;

            let cumsum = sorted_probs.fast_cumsum(D::Minus1)?;

            let mask_topp = cumsum.le(top_p)?;

            let masked_sorted =
                mask_topp.where_cond(&sorted_probs, &Tensor::zeros_like(&sorted_probs)?)?;

            let threshold = masked_sorted.max(D::Minus1)?;
            let threshold = threshold.unsqueeze(D::Minus1)?;
            let mask_full = probs.broadcast_ge(&threshold)?;
            probs = mask_full.where_cond(&probs, &Tensor::zeros_like(&probs)?)?;
        }

        // Min-P
        if min_p > 0.0 && min_p < 1.0 {
            let max_vals = probs.max(D::Minus1)?;
            let threshold_min = (max_vals.unsqueeze(D::Minus1)? * min_p)?;
            let mask_minp = probs.broadcast_gt(&threshold_min)?;
            probs = mask_minp.where_cond(&probs, &Tensor::zeros_like(&probs)?)?;
        }

        // Sample using the Gumbel-max trick fully on-device.
        let log_probs = probs.log()?;
        // Draw Gumbel noise (-log(-log(u))) FRESH on every call. Caching it
        // across calls is not an allocation optimisation, it is a correctness
        // bug: a `Sampler` lives for one whole request, so reusing one noise
        // draw makes `argmax(log_probs + gumbel)` a fixed ranking and every
        // token of that request comes out identical.
        let gumbel = {
            let uniform = Tensor::rand(0f32, 1f32, log_probs.shape(), log_probs.device())?;
            uniform
                .clamp(1e-20, 1.0)?
                .log()? // ln(u)
                .neg()? // -ln(u)
                .log()? // ln(-ln(u))
                .neg()? // -ln(-ln(u))
        };

        let gumbel_logits = (&log_probs + &gumbel)?;
        let next_token = gumbel_logits.argmax(D::Minus1)?.to_scalar::<u32>()?;

        // Extract the top‑n log‑probs if the caller asked for them.
        let (top_logprobs, logprob) = if return_logprobs {
            let k = self.top_n_logprobs;

            let sorted_values = probs.fast_sort_asc(D::Minus1)?;
            let topk_values = sorted_values
                .narrow(
                    D::Minus1,
                    sorted_values.dim(D::Minus1)? - top_k as usize,
                    top_k as usize,
                )?
                .to_vec1::<f32>()?;

            let sorted_idxs = probs.fast_argsort_asc(D::Minus1)?;
            let topk_idxs = sorted_idxs
                .narrow(
                    D::Minus1,
                    sorted_values.dim(D::Minus1)? - top_k as usize,
                    top_k as usize,
                )?
                .to_vec1::<u32>()?;

            let mut result = Vec::with_capacity(k);
            if let Some(tokenizer) = &self.tokenizer {
                for (prob, token) in topk_values.iter().zip(topk_idxs) {
                    let decoded = tokenizer
                        .decode(&[token], false)
                        .map_err(|e| Error::Msg(e.to_string()))?;
                    result.push(TopLogprob {
                        token,
                        logprob: prob.log(10.0),
                        bytes: Some(decoded),
                    });
                }
            } else {
                for (prob, token) in topk_values.iter().zip(topk_idxs) {
                    result.push(TopLogprob {
                        token,
                        logprob: prob.log(10.0),
                        bytes: None,
                    });
                }
            }

            let logprob = result.last().map(|res| res.logprob).unwrap_or(1.);

            (Some(result), logprob)
        } else {
            (None, 1.)
        };

        let bytes = if let Some(tokenizer) = &self.tokenizer {
            Some(
                tokenizer
                    .decode(&[next_token], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            )
        } else {
            None
        };

        Ok(Logprobs {
            token: next_token,
            logprob,
            top_logprobs,
            bytes,
        })
    }
    /// Truncate a candidate set (token id, full-softmax prob) to the kept
    /// sampling distribution, replicating `sample_top_kp_min_p`'s semantics
    /// exactly: top-k first, then top-p (descending cumsum, the crossing
    /// element kept), then min-p. Each filter is applied whenever its own
    /// parameter is in range; min-p does not depend on top-p.
    ///
    /// `candidates` must contain the top `candidates.len()` tokens of the
    /// full distribution (any order). Returns `None` when exactness cannot
    /// be guaranteed, i.e. when the kept set is not provably closed inside
    /// the candidate set:
    /// - a pure-top-p request (top_k <= 0) whose nucleus is not fully
    ///   contained in the candidates (candidate mass < top_p);
    /// - a min-p request that neither top-k nor top-p has already closed,
    ///   whose smallest candidate is still above the min-p threshold (so
    ///   tokens outside the candidate set may also survive min-p).
    ///
    /// With top_k > 0 containment is structural since the caller selects
    /// `k_sel >= top_k` candidates.
    ///
    /// Returned pairs are sorted descending by probability.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn truncate_topk_candidates(&self, mut candidates: Vec<(u32, f32)>) -> Option<Vec<(u32, f32)>> {
        // Descending by prob; ascending-index tiebreak for determinism.
        candidates.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        // Top-k truncation (candidates are a superset: k_sel >= top_k).
        // `top_k <= len` means the candidate set already is (or is truncated
        // to) the exact top-k set, so no token outside it can survive.
        let topk_closed = self.top_k > 0 && (self.top_k as usize) <= candidates.len();
        if self.top_k > 0 && (self.top_k as usize) < candidates.len() {
            candidates.truncate(self.top_k as usize);
        }

        let top_p = self.top_p as f32;
        let min_p = self.min_p as f32;

        let mut closed = topk_closed;
        if top_p > 0.0 && top_p < 1.0 {
            if self.top_k <= 0 {
                let mass: f32 = candidates.iter().map(|(_, p)| *p).sum();
                if mass < top_p {
                    return None;
                }
            }
            // CPU-path top-p: walk descending, keep while cumsum < top_p (so
            // the element that crosses the threshold is kept), drop the rest.
            let mut cumsum = 0.0f32;
            candidates.retain(|(_, p)| {
                if cumsum >= top_p {
                    false
                } else {
                    cumsum += *p;
                    true
                }
            });
            // The nucleus is contained (mass check above, or structurally
            // when top_k > 0), so the kept set is now closed.
            closed = true;
        }

        // CPU-path min-p (`min_p_threshold >= prob` is dropped). Applied
        // whenever min_p is in range, exactly as `sample_top_kp_min_p` does —
        // independently of top_p.
        if min_p > 0.0 && min_p < 1.0 {
            let max_p = candidates.first().map(|(_, p)| *p).unwrap_or(0.0);
            let threshold = max_p * min_p;
            // If neither top-k nor top-p closed the set, the candidates are
            // just the top `k_sel` of the vocabulary and tokens below the cut
            // may still clear the min-p threshold. Only safe when the smallest
            // candidate is already at/below the threshold (everything outside
            // is <= it, hence also dropped); otherwise refuse and let the
            // caller fall back to the exact full-vocab CPU path.
            if !closed && candidates.last().is_some_and(|(_, p)| *p > threshold) {
                return None;
            }
            candidates.retain(|(_, p)| *p > threshold);
        }

        Some(candidates)
    }

    /// Multinomial-sample from a truncated candidate set produced by
    /// [`Self::truncate_topk_candidates`].
    ///
    /// The kept pairs are re-ordered by ascending token id before building
    /// the `WeightedIndex`, so the f32 cumulative-weight sequence is
    /// identical to the CPU path's full-vocab `WeightedIndex` (zeroed-out
    /// tokens contribute exactly nothing to f32 partial sums) — the same rng
    /// state therefore yields the same token as `sample_top_kp_min_p`.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn sample_from_topk_candidates(
        &self,
        candidates: Vec<(u32, f32)>,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Option<Logprobs>> {
        let Some(mut kept) = self.truncate_topk_candidates(candidates) else {
            return Ok(None);
        };
        if kept.is_empty() {
            return Ok(None);
        }

        kept.sort_unstable_by_key(|(token, _)| *token);
        let weights: Vec<f32> = kept.iter().map(|(_, p)| *p).collect();
        let distr = WeightedIndex::new(&weights).map_err(Error::wrap)?;
        let choice = {
            let mut mut_ref_rng = &mut *rng.lock().expect("could not lock rng mutex");
            distr.sample(&mut mut_ref_rng)
        };
        let (token, prob) = kept[choice];
        let logprob = prob.log(10.0);

        let bytes = if let Some(tokenizer) = &self.tokenizer {
            Some(
                tokenizer
                    .decode(&[token], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            )
        } else {
            None
        };

        Ok(Some(Logprobs {
            token,
            logprob,
            top_logprobs: None,
            bytes,
        }))
    }

    /// GPU radix-select sampling for vocabularies too large for candle's
    /// shared-memory arg_sort (the fallback introduced in 947a9c2fc).
    /// Instead of D2H-ing the whole ~129K-logit tensor and sorting it on the
    /// CPU, select the top `k_sel` candidate tokens on-GPU with the
    /// flashmlasparse radix top-k kernel (multi-pass radix-256 select over
    /// global memory — no shared-memory vocab limit), then run the exact CPU
    /// truncation semantics on the tiny candidate set (<= 1024 entries,
    /// ~8 KB D2H instead of ~516 KB + a full-vocab sort).
    ///
    /// Returns Ok(None) when this path cannot guarantee exactness — top_k
    /// above the kernel cap, or a pure-top-p request whose nucleus is not
    /// contained in the selected candidates — and the caller falls back to
    /// the CPU path.
    #[cfg(feature = "cuda")]
    fn sample_fast_topk_gpu(
        &self,
        logits: &Tensor,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Option<Logprobs>> {
        let vocab = logits.dim(D::Minus1)?;
        let max_k = *GPU_RADIX_TOPK_SIZES.last().unwrap();
        let k_needed = if self.top_k > 0 {
            self.top_k as usize
        } else {
            max_k
        };
        // Round up to the kernel's dispatch table (top-k truncation happens
        // exactly on the candidate set). `vocab <= k_sel` never occurs on
        // this path (the small-vocab sort already fits in shared memory) but
        // is excluded anyway so the kernel's identity-fill mode stays out of
        // scope.
        if k_needed > max_k || vocab <= max_k {
            return Ok(None);
        }
        let Some(k_sel) = GPU_RADIX_TOPK_SIZES
            .iter()
            .copied()
            .find(|&s| s >= k_needed)
        else {
            return Ok(None);
        };

        let logits_f32 = logits.to_dtype(DType::F32)?;
        let probs =
            candle_nn::ops::softmax_last_dim(&(&logits_f32 / self.temperature.unwrap_or(1.))?)?;
        // Select on the raw logits, not the probs: softmax (with positive
        // temperature) is monotonic so the top-k set is identical, but
        // post-softmax values cluster near 0.0 and would collapse into a
        // single bucket of the kernel's coarse fp16 radix pass; logits are
        // well-spread.
        let indices = arc_cuda_graph::flashmlasparse::radix_topk_rows_f32(
            &logits_f32.reshape((1, vocab))?,
            k_sel,
        )?
        .reshape((k_sel,))?;
        let values = probs.gather(&indices, D::Minus1)?;

        let token_ids = indices.to_vec1::<u32>()?;
        let token_probs = values.to_vec1::<f32>()?;
        let candidates: Vec<(u32, f32)> = token_ids
            .into_iter()
            .zip(token_probs)
            .filter(|(token, _)| *token != u32::MAX)
            .collect();

        self.sample_from_topk_candidates(candidates, rng)
    }

    fn sample_speculative_top_kp_min_p(
        &self,
        logits: Tensor,
        return_logprobs: bool,
        top_k: i64,
        top_p: f32,
        min_p: f32,
    ) -> Result<Logprobs> {
        let mut probs: Vec<f32> = logits.to_vec1()?;

        // Determine how many elements we need for partial sort
        let k = if top_k > 0 {
            top_k as usize
        } else {
            probs.len()
        };

        // Get sorted top-k indices with partial sort, zeroing out rest
        let idx_probs = partial_sort_top_k(&mut probs, k, true);

        // TOP P
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for (index, prob) in &idx_probs {
            if cumsum >= top_p {
                probs[*index as usize] = 0.0;
            } else {
                cumsum += prob;
            }
        }

        // Get max_p from first sorted element
        let max_p = idx_probs.first().map(|(_, p)| *p).unwrap_or(0.0);

        // MIN P
        // min-p sampling samples from the tokens whose prob are greater than
        // (max prob of token in dist) * min_p

        // Clamp smaller probabilities to zero.
        let min_p_threshold = max_p * min_p;
        for (index, prob) in &idx_probs {
            if min_p_threshold >= *prob {
                probs[*index as usize] = 0.0;
            }
        }

        // Find argmax directly on the Vec (O(n) scan, no Tensor creation)
        let next_token = argmax_f32(&probs);
        let logprob = probs[next_token as usize].log(10.0);

        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(&probs)?)
        } else {
            None
        };

        let bytes = if let Some(tokenizer) = &self.tokenizer {
            Some(
                tokenizer
                    .decode(&[next_token], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            )
        } else {
            None
        };

        Ok(Logprobs {
            token: next_token,
            logprob,
            top_logprobs,
            bytes,
        })
    }

    fn sample_multinomial(
        &self,
        probs: &[f32],
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Logprobs> {
        let distr = WeightedIndex::new(probs).map_err(Error::wrap)?;

        // Hold the global rng mutex for the draw and NOTHING else. The guard
        // used to live to the end of this function (temporary lifetime
        // extension through the `&mut *lock()` borrow), which serialised every
        // concurrently-sampling sequence in the batch behind this one's
        // `get_top_logprobs` (a partial sort plus up to `top_n` tokenizer
        // decodes) and its own `tokenizer.decode` — none of which touch the
        // rng.
        let next_token = {
            let mut guard = rng.lock().expect("could not lock rng mutex");
            distr.sample(&mut *guard) // "Find the first item which has a weight *higher* than the chosen weight."
        };
        let logprob = probs[next_token].log(10.0);

        let top_logprobs = if return_logprobs {
            Some(self.get_top_logprobs(probs)?)
        } else {
            None
        };

        let bytes = if let Some(tokenizer) = &self.tokenizer {
            Some(
                tokenizer
                    .decode(&[next_token.try_into().unwrap()], false)
                    .map_err(|x| Error::Msg(x.to_string()))?,
            )
        } else {
            None
        };

        Ok(Logprobs {
            token: next_token as u32,
            logprob,
            top_logprobs,
            bytes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_top_kp_min_p(
        &self,
        probs: &mut [f32],
        top_k: i64,
        top_p: f32,
        min_p: f32,
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
    ) -> Result<Logprobs> {
        // Determine how many elements we need for partial sort
        let k = if top_k > 0 {
            top_k as usize
        } else {
            probs.len()
        };

        // The sorted pairs feed exactly three consumers: the zero-rest pass
        // inside `partial_sort_top_k` (only when `k < n`), the top-p cumsum
        // walk, and the min-p threshold scan. Under the server defaults
        // (`top_k = -1`, `top_p = 1.0`, `min_p = 0.0` — `engine/add_request.rs`)
        // none of the three runs: `k = n` zeroes nothing, and both filter
        // blocks below are gated out of (0, 1). That request was still paying
        // a full-vocab `sort_unstable_by` — 129,280 elements for V4, per
        // sequence per token — to produce a Vec nothing read. Skip straight to
        // the multinomial draw; `probs` is untouched either way, so the drawn
        // token (same rng stream) and the reported logprobs are identical.
        let top_p_active = top_p > 0.0 && top_p < 1.0;
        let min_p_active = min_p > 0.0 && min_p < 1.0;
        if k >= probs.len() && !top_p_active && !min_p_active {
            return self.sample_multinomial(probs, return_logprobs, rng);
        }

        // Get sorted top-k indices with partial sort, zeroing out rest
        let idx_probs = partial_sort_top_k(probs, k, true);

        // TOP P

        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".

        // Clamp smaller probabilities to zero.
        if top_p > 0.0 && top_p < 1.0 {
            let mut cumsum = 0.;
            for (index, prob) in &idx_probs {
                if cumsum >= top_p {
                    probs[*index as usize] = 0.0;
                } else {
                    cumsum += prob;
                }
            }
        }

        // MIN P

        // min-p sampling samples from the tokens whose prob are greater than
        // (max prob of token in dist) * min_p
        //
        // This filter is INDEPENDENT of top-p: `top_p = 1.0, min_p = 0.1` is a
        // valid, common request. This block used to sit behind an early return
        // taken whenever top_p was outside (0, 1), which silently discarded the
        // caller's min_p.
        if min_p > 0.0 && min_p < 1.0 {
            // Get max_p from first sorted element. Top-p never zeroes the
            // argmax (its cumsum starts below top_p), so this is the max of
            // the surviving set either way.
            let max_p = idx_probs.first().map(|(_, p)| *p).unwrap_or(0.0);

            // Clamp smaller probabilities to zero.
            let min_p_threshold = max_p * min_p;
            for (index, prob) in &idx_probs {
                if min_p_threshold >= *prob {
                    probs[*index as usize] = 0.0;
                }
            }
        }

        // Sample with clamped probabilities.
        self.sample_multinomial(probs, return_logprobs, rng)
    }

    fn apply_penalties(&self, mut logits: Vec<f32>, context: &[u32]) -> Result<Tensor> {
        if context.is_empty() {
            candle_core::bail!("Penalty context is empty, this should not happen.");
        }

        // Dry penalty
        self.apply_dry_penalty(&mut logits, context)?;

        // Frequency, presence, repetition penalty
        self.apply_freq_pres_rep_penalty(&mut logits, context)?;

        let vocab_size = logits.len();
        Tensor::from_vec(logits, vocab_size, &Device::Cpu)
    }

    fn apply_freq_pres_rep_penalty(&self, logits: &mut [f32], context: &[u32]) -> Result<()> {
        if self.frequency_penalty.is_some()
            || self.presence_penalty.is_some()
            || self.repetition_penalty.is_some()
        {
            let frequency_penalty = self.frequency_penalty.unwrap_or(0.);
            let presence_penalty = self.presence_penalty.unwrap_or(0.);
            let repetition_penalty = self.repetition_penalty.unwrap_or(1.);

            //mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence

            let mut counts = vec![0.0f32; logits.len()];
            for ctx in context.iter() {
                // Llama 3.2 uses a hack triggering this error... we wouldn't want a weight on it anyway
                if *ctx as usize >= logits.len() {
                    continue;
                }
                counts[*ctx as usize] += 1.0;
            }

            for (token_id, logit) in logits.iter_mut().enumerate() {
                let count = counts[token_id];
                *logit = *logit
                    - count * frequency_penalty
                    - if count > 0.0 { 1. } else { 0. } * presence_penalty;

                if repetition_penalty != 1.0 && count > 0.0 {
                    if *logit > 0.0 {
                        *logit /= repetition_penalty;
                    } else {
                        *logit *= repetition_penalty;
                    }
                }
            }
        }
        Ok(())
    }

    /// Threshold for using parallel iteration in dry penalty.
    /// Below this, sequential is faster due to parallel overhead.
    const DRY_PENALTY_PAR_THRESHOLD: usize = 1024;

    fn apply_dry_penalty(&self, logits: &mut [f32], context: &[u32]) -> Result<()> {
        if let Some(ref params) = self.dry_params {
            if params.multiplier == 0. {
                return Ok(());
            }

            let last_token = *context.last().unwrap();

            // Use parallel iteration only for large contexts
            let match_indices: Vec<usize> = if context.len() > Self::DRY_PENALTY_PAR_THRESHOLD {
                context
                    .par_iter()
                    .enumerate()
                    .take(context.len() - 1)
                    .filter(|(_i, x)| last_token == **x)
                    .map(|(i, _)| i)
                    .collect()
            } else {
                context
                    .iter()
                    .enumerate()
                    .take(context.len() - 1)
                    .filter(|(_i, x)| last_token == **x)
                    .map(|(i, _)| i)
                    .collect()
            };

            let mut match_lengths = HashMap::new();

            for i in match_indices {
                let next_token = context[i + 1];

                if params.sequence_breakers.contains(&next_token) {
                    continue;
                }

                let mut match_length = 1;

                // Limit match length to avoid quadratic runtime and potential DoS with adversarial inputs.
                while match_length < 50 {
                    if match_length > i {
                        // Start of input
                        break;
                    }

                    let j = i - match_length;

                    let prev_tok = context[context.len() - (match_length + 1)];
                    if context[j] != prev_tok {
                        // Start of match reached
                        break;
                    }

                    if params.sequence_breakers.contains(&prev_tok) {
                        // Seq breaking tok reached
                        break;
                    }

                    match_length += 1;
                }

                #[allow(clippy::map_entry)]
                if match_lengths.contains_key(&next_token) {
                    match_lengths.insert(next_token, match_length.max(match_lengths[&next_token]));
                } else {
                    match_lengths.insert(next_token, match_length);
                }
            }

            // Actually apply penalties
            for (tok, match_len) in match_lengths {
                if match_len >= params.allowed_length {
                    // Llama 3.2 uses a hack triggering this error... we wouldn't want a weight on it anyway
                    if tok as usize >= logits.len() {
                        continue;
                    }
                    let penalty = params.multiplier
                        * params.base.powf((match_len - params.allowed_length) as f32);
                    logits[tok as usize] -= penalty;
                }
            }
        }
        Ok(())
    }

    /// Top-nσ filtering (Tang et al. 2024, arXiv:2411.07641): mask (to -inf)
    /// every logit below `max - n * σ`, where σ is the standard deviation of
    /// the logits. Runs entirely as device tensor ops (no D2H sync).
    ///
    /// Applied on the raw pre-temperature, pre-penalty logits: since dividing
    /// the logits by a temperature `T` scales both `max - logit` gaps and σ by
    /// exactly `1/T`, the kept set is provably temperature-invariant. `n = 0`
    /// keeps only the argmax (and exact ties); the argmax itself always
    /// survives the filter for any `n >= 0`.
    ///
    /// Statistics are computed over finite entries only, so logits already
    /// masked to -inf (e.g. by constraint biasing) neither poison σ nor get
    /// resurrected.
    fn apply_top_nsigma(&self, logits: &Tensor, nsigma: f32) -> Result<Tensor> {
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let zeros = logits_f32.zeros_like()?;
        // Finite entries: anything >= f32::MIN (excludes -inf and NaN).
        let finite_mask = logits_f32.ge(f32::MIN as f64)?;
        let count = finite_mask.to_dtype(DType::F32)?.sum_all()?.unsqueeze(0)?;
        let safe = finite_mask.where_cond(&logits_f32, &zeros)?;
        let mean = safe.sum_all()?.unsqueeze(0)?.broadcast_div(&count)?;
        // Zero the masked entries *after* subtracting the mean (select, not
        // multiply, so -inf never produces NaN), then square.
        let centered = finite_mask.where_cond(&safe.broadcast_sub(&mean)?, &zeros)?;
        let std = centered
            .sqr()?
            .sum_all()?
            .unsqueeze(0)?
            .broadcast_div(&count)?
            .sqrt()?;
        let max = logits_f32.max(D::Minus1)?.unsqueeze(0)?;
        let threshold = max.broadcast_sub(&(std * nsigma as f64)?)?;
        let keep = logits_f32.broadcast_ge(&threshold)?;
        let neg_inf = (zeros + f64::NEG_INFINITY)?;
        keep.where_cond(&logits_f32, &neg_inf)
    }

    #[allow(unused)]
    /// Sample the provided tokens.
    ///
    /// If the temperature is `None`, argmax sampling is used. Otherwise, the selected sampling is used.
    /// With `top-p` sampling, if the `top-p` value is `<= 0.0` or `>= 1.0`, multinomial sampling is used.
    pub fn sample(
        &self,
        logits: Tensor,
        context: &[u32],
        return_logprobs: bool,
        rng: Arc<Mutex<Isaac64Rng>>,
        sample_speculative: bool,
        multiple_sequences: bool,
    ) -> Result<Logprobs> {
        // `logit_bias` is defined as added to the model's raw logits *prior to
        // sampling*, so it runs before every filter — including top-nσ, whose
        // threshold is computed from the raw logits.
        let logits = self.apply_logits_bias(logits)?;
        // Top-nσ runs first, on the raw logits, so every downstream path
        // (GPU fast path, GPU radix top-k, CPU) sees the filtered set.
        let logits = match self.top_nsigma {
            Some(nsigma) => self.apply_top_nsigma(&logits, nsigma)?,
            None => logits,
        };
        // ── GPU fast path ────────────────────────────────────────────────────
        // For the common request shape (no penalties, no logits processors,
        // no logprobs, no speculative sampling) we can stay entirely on GPU
        // and ship a single u32 token back. This skips the ~5.9 ms/token
        // CPU pipeline (D2H 152K logits + softmax + topk/topp + multinomial).
        let no_penalties = !self.has_penalties();
        let trivial = !sample_speculative
            && !return_logprobs
            && self.logits_processors.is_empty()
            && no_penalties
            && !logits.device().is_cpu();
        if trivial {
            // Greedy: single GPU argmax kernel + 4-byte D2H. Never sorts, so
            // always safe regardless of vocab size.
            if self.temperature.is_none() {
                let next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
                let bytes = if let Some(tok) = &self.tokenizer {
                    Some(
                        tok.decode(&[next_token], false)
                            .map_err(|x| Error::Msg(x.to_string()))?,
                    )
                } else {
                    None
                };
                return Ok(Logprobs {
                    token: next_token,
                    logprob: 0.0,
                    top_logprobs: None,
                    bytes,
                });
            }
            // Temperature sampling: full nucleus sampling stays on GPU via
            // sample_fast. BUT the top-k / top-p branches there call
            // `fast_sort_asc`, which on CUDA dispatches candle's arg_sort kernel.
            // That kernel requests `next_power_of_2(vocab) * 4` bytes of shared
            // memory per block; for large vocabularies (e.g. DeepSeek-V4's
            // ~129K) this is ~1 MB, far over the 48 KB/block hardware limit, and
            // the launch fails with CUDA_ERROR_INVALID_VALUE. Only stay on the
            // GPU sort path when a sort is either not needed (top_k<=0 and
            // top_p>=1) or small enough to fit; otherwise fall through to the
            // correct CPU sampling path below (it pays a D2H of the logits, but
            // only when top-k/top-p is actually requested — the common
            // top_p=1.0/top_k=0 request still stays on GPU).
            let needs_sort = self.top_k > 0 || (self.top_p > 0.0 && self.top_p < 1.0);
            let sort_fits = || -> bool {
                match logits.dim(D::Minus1) {
                    Ok(vocab) => vocab.next_power_of_two().saturating_mul(4) <= 48 * 1024,
                    Err(_) => false,
                }
            };
            if !needs_sort || sort_fits() {
                return self.sample_fast(
                    logits,
                    context,
                    return_logprobs,
                    self.top_k,
                    self.top_p,
                    self.min_p,
                );
            }
            // Big-vocab GPU path: radix-select the top candidate tokens
            // on-GPU (no shared-memory limit) and finish with the exact CPU
            // truncation + multinomial semantics on <= 1024 candidates. Falls
            // through to the full CPU path when exactness cannot be
            // guaranteed (Ok(None)) or on any GPU error.
            #[cfg(feature = "cuda")]
            if logits.device().is_cuda() {
                use std::sync::atomic::Ordering;
                match self.sample_fast_topk_gpu(&logits, rng.clone()) {
                    Ok(Some(result)) => {
                        gpu_sampling_health::GPU_OK.fetch_add(1, Ordering::Relaxed);
                        return Ok(result);
                    }
                    Ok(None) => {
                        gpu_sampling_health::DECLINED.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        // Loud once, then rare. The first failure is an ERROR
                        // because it means every subsequent token pays a
                        // full-vocab D2H + CPU sort; after that we back off to
                        // powers of two so a persistent fault stays visible
                        // without emitting one line per token (the previous
                        // behaviour: ~10 lines/s at B=256, which reads as
                        // noise and got ignored for exactly that reason).
                        let prior = gpu_sampling_health::FAILED.fetch_add(1, Ordering::Relaxed);
                        if prior == 0 {
                            tracing::error!(
                                "GPU radix top-k sampling FAILED — every sampled token now \
                                 falls back to the CPU sampler (full-vocab D2H + sort per \
                                 sequence per step). This is a throughput regression, not a \
                                 transient. Cause: {e}"
                            );
                        } else if (prior + 1).is_power_of_two() {
                            tracing::warn!(
                                "GPU radix top-k sampling still failing: {} CPU fallbacks so \
                                 far. Cause: {e}",
                                prior + 1
                            );
                        }
                    }
                }
            }
            // else: fall through to the CPU sampling path (correct on any vocab).
        }

        let logits = logits.to_vec1()?;
        let mut logits = self.apply_penalties(logits, context)?;
        for processor in &self.logits_processors {
            logits = processor.apply(&logits, context)?;
        }
        let next_token = if sample_speculative {
            // NOTE: this branch is an ARGMAX at every temperature.
            // `sample_speculative_top_kp_min_p` finishes with `argmax_f32`, and
            // dividing by a positive temperature is monotonic, so the
            // `Some(temperature)` arm below selects exactly the same token as
            // the greedy arm — it only rescales the reported logprob. There is
            // no stochastic draw here and never was.
            //
            // That is why speculative *verification* (accept iff the draft
            // token equals this one) is lossless for greedy decoding only, and
            // why `pipeline::sampling::sample_target_sequence_speculative`
            // refuses to speculate when the sequence is not greedy rather than
            // shipping greedy tokens for a request that asked for sampling.
            // Correct temperature speculation needs rejection sampling
            // (`min(1, p/q)` accept + `max(0, p-q)` residual redraw), which is
            // not implemented.
            match self.temperature {
                None => self.sample_speculative_top_kp_min_p(
                    logits,
                    return_logprobs,
                    self.top_k,
                    self.top_p as f32,
                    self.min_p as f32,
                )?,
                Some(temperature) => {
                    let logits = (&logits / temperature)?;
                    let probs = candle_nn::ops::softmax_last_dim(&logits)?;

                    self.sample_speculative_top_kp_min_p(
                        probs,
                        return_logprobs,
                        self.top_k,
                        self.top_p as f32,
                        self.min_p as f32,
                    )?
                }
            }
        } else {
            match self.temperature {
                None => self.sample_argmax(logits, return_logprobs)?,
                Some(temperature) => {
                    let logits = (&logits / temperature)?;
                    let probs = candle_nn::ops::softmax_last_dim(&logits)?;
                    let mut probs: Vec<f32> = probs.to_vec1()?;

                    self.sample_top_kp_min_p(
                        &mut probs,
                        self.top_k,
                        self.top_p as f32,
                        self.min_p as f32,
                        return_logprobs,
                        rng,
                    )?
                }
            }
        };
        Ok(next_token)
    }
}

mod tests {
    #[test]
    fn test_argmax() {
        use super::Sampler;
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::sync::Arc;
        use std::sync::Mutex;

        let sampler = Sampler::new(
            None,
            10,
            None,
            None,
            None,
            None,
            None,
            32,
            0.1,
            0.05,
            None,
            vec![],
            None,
        )
        .unwrap();
        let logits = Tensor::arange(0f32, 1024f32, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let res = sampler
            .sample(
                logits,
                &(0..1024).collect::<Vec<_>>(),
                false,
                rng,
                false,
                false,
            )
            .unwrap();
        assert_eq!(res.token, 1023);
        assert_eq!(res.top_logprobs, None);
        assert_eq!(res.logprob, 1023f64.log(10.) as f32)
    }

    #[test]
    fn test_gumbel_speculative() {
        use super::Sampler;
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::sync::Arc;
        use std::sync::Mutex;

        let sampler = Sampler::new(
            None,
            10,
            None,
            None,
            None,
            None,
            None,
            32,
            0.1,
            0.05,
            None,
            vec![],
            None,
        )
        .unwrap();
        let logits = Tensor::arange(0f32, 1024f32, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let res = sampler
            .sample(
                logits,
                &(0..1024).collect::<Vec<_>>(),
                false,
                rng,
                true,
                false,
            )
            .unwrap();
        assert_eq!(res.token, 1023);
        assert_eq!(res.top_logprobs, None);
        assert_eq!(res.logprob, 1023f64.log(10.) as f32)
    }

    /// Sampler accessors used by the GPU-autonomous decode runner expose the
    /// effective sampling config so the GPU sampler kernel can be configured
    /// to match. Verifying greedy + temperature + top_p + penalties round-trip.
    #[test]
    fn accessors_for_autonomous_decode() {
        use super::Sampler;
        let s_greedy = Sampler::new(
            None, /* temperature = greedy */
            0,    // top_n_logprobs
            None, // tokenizer
            None, // freq_penalty
            None, // pres_penalty
            None, // rep_penalty
            None, // dry
            -1,   // top_k disabled
            1.0,  // top_p disabled (>=1.0)
            0.0,  // min_p
            None, // top_nsigma
            vec![],
            None,
        )
        .unwrap();
        assert!(s_greedy.is_greedy());
        assert_eq!(s_greedy.temperature(), None);
        assert_eq!(s_greedy.top_p(), 1.0);
        assert_eq!(s_greedy.top_k(), -1);
        assert_eq!(s_greedy.frequency_penalty(), None);
        assert_eq!(s_greedy.presence_penalty(), None);
        assert!(!s_greedy.has_custom_logits_processors());

        let s_topp = Sampler::new(
            Some(0.7),
            0,
            None,
            Some(0.1),
            Some(0.2),
            None,
            None,
            40,
            0.95,
            0.0,
            None,
            vec![],
            None,
        )
        .unwrap();
        assert!(!s_topp.is_greedy());
        assert_eq!(s_topp.temperature(), Some(0.7));
        assert_eq!(s_topp.top_p(), 0.95);
        assert_eq!(s_topp.top_k(), 40);
        assert_eq!(s_topp.frequency_penalty(), Some(0.1));
        assert_eq!(s_topp.presence_penalty(), Some(0.2));
        assert_eq!(s_topp.top_nsigma(), None);
    }

    /// `is_raw_argmax` is the precondition for substituting an external
    /// `argmax(raw logits)` for [`Sampler::sample`] — which is exactly what
    /// speculative verification does.
    ///
    /// It is **strictly stronger than `is_greedy`**, and the gap is the point:
    /// with `temperature: None` and a penalty (or a logits processor), `sample`
    /// runs `apply_penalties` and every processor *before* `sample_argmax`, so
    /// the token it returns is `argmax(modified)`, not `argmax(raw)`.
    #[test]
    fn is_raw_argmax_is_stronger_than_is_greedy() {
        let mk = |temperature, freq, pres, rep, top_k, top_p, min_p, nsigma| {
            super::Sampler::new(
                temperature,
                0,
                None,
                freq,
                pres,
                rep,
                None,
                top_k,
                top_p,
                min_p,
                nsigma,
                vec![],
                None,
            )
            .unwrap()
        };

        // Plain greedy: both hold.
        let plain = mk(None, None, None, None, -1, 1.0, 0.0, None);
        assert!(plain.is_greedy());
        assert!(plain.is_raw_argmax());

        // temperature > 0: neither.
        let hot = mk(Some(0.7), None, None, None, -1, 1.0, 0.0, None);
        assert!(!hot.is_greedy());
        assert!(!hot.is_raw_argmax());

        // The gap: greedy, but the logits get rewritten first.
        for (freq, pres, rep) in [
            (Some(0.5), None, None),
            (None, Some(0.5), None),
            (None, None, Some(1.1)),
        ] {
            let penalised = mk(None, freq, pres, rep, -1, 1.0, 0.0, None);
            assert!(penalised.is_greedy(), "still temperature-free");
            assert!(
                !penalised.is_raw_argmax(),
                "a penalty rewrites the logits before argmax, so argmax(raw) is a \
                 different token"
            );
        }

        // Identity penalties are not penalties.
        let identity = mk(None, Some(0.0), Some(0.0), Some(1.0), -1, 1.0, 0.0, None);
        assert!(identity.is_raw_argmax());

        // top-k / top-p / min-p are only consulted on the `Some(temperature)`
        // branch, so they cannot move a greedy argmax…
        let filtered = mk(None, None, None, None, 40, 0.95, 0.05, None);
        assert!(filtered.is_raw_argmax());

        // …and top-nσ masks only tokens BELOW the maximum, so the argmax is
        // invariant under it too.
        let nsigma = mk(None, None, None, None, -1, 1.0, 0.0, Some(1.5));
        assert!(nsigma.is_raw_argmax());

        // A custom logits processor runs before argmax on the CPU path.
        let with_processor = super::Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            None,
            vec![std::sync::Arc::new(
                |logits: &candle_core::Tensor, _: &[u32]| Ok(logits.clone()),
            )],
            None,
        )
        .unwrap();
        assert!(with_processor.is_greedy());
        assert!(
            !with_processor.is_raw_argmax(),
            "a processor may rewrite the logits, and the sampler cannot know it did not"
        );
    }

    /// Build a sampler with only temperature + top-nσ active (no top-k/p/min-p,
    /// no penalties) so the CPU multinomial path samples the full filtered set.
    #[cfg(test)]
    fn make_nsigma_sampler(temperature: f64, top_nsigma: Option<f32>) -> super::Sampler {
        super::Sampler::new(
            Some(temperature),
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            top_nsigma,
            vec![],
            None,
        )
        .unwrap()
    }

    /// Top-nσ with n = 0 keeps only the argmax: sampling is exactly greedy at
    /// any temperature (the n→0 limit of the filter).
    #[test]
    fn top_nsigma_zero_is_greedy_at_any_temperature() {
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::sync::{Arc, Mutex};

        let raw = vec![10.0f32, 9.8, 9.6, 0.0, 0.2, 0.1];
        for temperature in [0.5, 1.0, 4.0] {
            let sampler = make_nsigma_sampler(temperature, Some(0.0));
            let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(7)));
            for _ in 0..50 {
                let logits = Tensor::from_vec(raw.clone(), 6, &Device::Cpu).unwrap();
                let res = sampler
                    .sample(logits, &[0], false, rng.clone(), false, false)
                    .unwrap();
                assert_eq!(res.token, 0, "n=0 must be greedy at T={temperature}");
            }
        }
    }

    /// The kept set {i : logit_i >= max - n*sigma} is computed on the raw
    /// logits, so it is invariant to temperature: at every temperature the
    /// observed support equals the analytic kept set and never includes a
    /// filtered token.
    #[test]
    fn top_nsigma_kept_set_is_temperature_invariant() {
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::collections::HashSet;
        use std::sync::{Arc, Mutex};

        let raw = vec![10.0f32, 9.8, 9.6, 0.0, 0.2, 0.1];
        let nsigma = 1.0f32;

        // Analytic kept set from the same statistics the filter uses.
        let mean = raw.iter().sum::<f32>() / raw.len() as f32;
        let var = raw.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / raw.len() as f32;
        let threshold = raw.iter().cloned().fold(f32::MIN, f32::max) - nsigma * var.sqrt();
        let expected: HashSet<u32> = raw
            .iter()
            .enumerate()
            .filter(|(_, l)| **l >= threshold)
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(expected, HashSet::from([0, 1, 2]), "test fixture sanity");

        for temperature in [0.5, 1.0, 2.0, 5.0] {
            let sampler = make_nsigma_sampler(temperature, Some(nsigma));
            let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
            let mut support = HashSet::new();
            for _ in 0..300 {
                let logits = Tensor::from_vec(raw.clone(), 6, &Device::Cpu).unwrap();
                let res = sampler
                    .sample(logits, &[0], false, rng.clone(), false, false)
                    .unwrap();
                support.insert(res.token);
            }
            assert_eq!(
                support, expected,
                "support at T={temperature} must equal the analytic kept set"
            );
        }
    }

    /// A huge n keeps every token, so sampling is bit-identical (same rng
    /// stream) to a sampler with top-nσ disabled: distribution sanity.
    #[test]
    fn top_nsigma_large_n_is_a_no_op() {
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::sync::{Arc, Mutex};

        let raw = vec![2.0f32, 1.5, 1.0, 0.5, 0.0, -0.5];
        let with = make_nsigma_sampler(1.0, Some(1000.0));
        let without = make_nsigma_sampler(1.0, None);

        let draw = |s: &super::Sampler| -> Vec<u32> {
            let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(1234)));
            (0..50)
                .map(|_| {
                    let logits = Tensor::from_vec(raw.clone(), 6, &Device::Cpu).unwrap();
                    s.sample(logits, &[0], false, rng.clone(), false, false)
                        .unwrap()
                        .token
                })
                .collect()
        };

        assert_eq!(draw(&with), draw(&without));
    }

    /// Production regression test for the chat repetition-loop fix.
    ///
    /// DeepSeek-V4-Flash at 2-bit can emit the correct token and then loop it
    /// forever instead of emitting EOS (e.g. "400.400.400..."). The serve-time
    /// loop-breaker is the frequency / presence / repetition penalty, applied on
    /// the CPU sampling path — which is the only path on a CPU device, and the
    /// path the CUDA arg_sort fix routes penalized requests to on GPU as well.
    ///
    /// This pins that a token a greedy decode would repeat forever is demoted
    /// below an alternative once any one of the three penalties is applied (the
    /// loop is broken), and that without a penalty the loop persists. Greedy
    /// (temperature = None) makes selection a deterministic argmax over the
    /// penalized logits, so the assertions are exact and RNG-independent.
    #[test]
    fn penalties_break_repetition_loop() {
        use super::Sampler;
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::sync::{Arc, Mutex};

        // vocab = 6; token 3 is the degeneration attractor (highest logit) and
        // token 1 is the runner-up. A greedy decode always re-selects token 3.
        let raw = vec![0.5f32, 4.0, 0.5, 5.0, 0.5, 0.5];
        let device = Device::Cpu;
        // Context in which token 3 has already been emitted repeatedly (the loop).
        let context: Vec<u32> = vec![3, 3, 3, 3];
        let rng = || Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0)));
        let logits = || Tensor::from_vec(raw.clone(), 6, &device).unwrap();

        let mk = |freq, pres, rep| {
            // Greedy (None temp), top_k/top_p/min_p disabled: argmax over penalized logits.
            Sampler::new(
                None,
                0,
                None,
                freq,
                pres,
                rep,
                None,
                -1,
                1.0,
                0.0,
                None,
                vec![],
                None,
            )
            .unwrap()
        };
        let sample = |s: &Sampler| {
            s.sample(logits(), &context, false, rng(), false, false)
                .unwrap()
                .token
        };

        // No penalty: the loop continues — greedy re-selects the attractor.
        assert_eq!(
            sample(&mk(None, None, None)),
            3,
            "without a penalty the degenerate token should win"
        );

        // Repetition penalty 2.0: logit[3]=5.0>0 -> /2.0 = 2.5 < logit[1]=4.0.
        assert_eq!(
            sample(&mk(None, None, Some(2.0))),
            1,
            "repetition_penalty must break the loop"
        );

        // Frequency penalty 1.0: logit[3] -= count(4)*1.0 = 1.0 < logit[1]=4.0.
        assert_eq!(
            sample(&mk(Some(1.0), None, None)),
            1,
            "frequency_penalty must break the loop"
        );

        // Presence penalty 2.0: logit[3] -= 2.0 (count>0) = 3.0 < logit[1]=4.0.
        assert_eq!(
            sample(&mk(None, Some(2.0), None)),
            1,
            "presence_penalty must break the loop"
        );
    }

    /// `sample_fast` (the on-device Gumbel-max path) must draw fresh noise on
    /// every call.
    ///
    /// Regression: the noise tensor was drawn once and cached on the `Sampler`,
    /// then cloned for every subsequent call. A `Sampler` is constructed once
    /// per request (`engine/add_request.rs`), so `argmax(log_probs + gumbel)`
    /// with a frozen `gumbel` collapsed to a fixed ranking — every token of a
    /// request was the same token, at any temperature.
    ///
    /// Discriminator: a flat distribution over 64 tokens at temperature 1.0.
    /// Correct sampling covers essentially the whole support in 512 draws;
    /// frozen noise yields exactly one distinct token.
    #[test]
    fn sample_fast_draws_fresh_gumbel_noise_per_call() {
        use super::Sampler;
        use candle_core::{Device, Tensor};
        use std::collections::HashSet;

        const VOCAB: usize = 64;
        const DRAWS: usize = 512;

        // Uniform logits: every token has probability 1/64.
        let logits = Tensor::from_vec(vec![0f32; VOCAB], VOCAB, &Device::Cpu).unwrap();
        // temperature=1.0, top_k/top_p/min_p disabled so no sort runs and the
        // draw is a pure Gumbel-max over the full support.
        let sampler = Sampler::new(
            Some(1.0),
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            None,
            vec![],
            None,
        )
        .unwrap();

        let mut seen = HashSet::new();
        for _ in 0..DRAWS {
            let token = sampler
                .sample_fast(logits.clone(), &[0u32], false, -1, 1.0, 0.0)
                .unwrap()
                .token;
            assert!((token as usize) < VOCAB);
            seen.insert(token);
        }

        // Expected distinct count for 512 uniform draws over 64 tokens is
        // ~64.0 (P(any token missed) ~ 3e-4). Frozen noise gives 1.
        assert!(
            seen.len() >= VOCAB * 3 / 4,
            "sample_fast covered only {} of {VOCAB} tokens in {DRAWS} draws from a flat \
             distribution — Gumbel noise is not being redrawn per call",
            seen.len()
        );
    }

    /// `min_p` must be applied whenever it is in (0, 1), independently of
    /// `top_p`.
    ///
    /// Regression: `sample_top_kp_min_p` returned early whenever `top_p` was
    /// outside (0, 1) — before the min-p block — so `top_p = 1.0, min_p = 0.5`
    /// (a perfectly ordinary request) applied no min-p at all and the tail the
    /// caller paid to exclude stayed samplable.
    ///
    /// Discriminator: softmax([4, 0, 0, 0]) = [0.9479, 0.0174, 0.0174, 0.0174].
    /// min_p = 0.5 puts the threshold at 0.474, so only token 0 survives and
    /// every draw must be 0. With min-p skipped, ~5.2% of draws land on 1..3;
    /// over 400 draws the probability of seeing none of them is ~5e-10.
    #[test]
    fn min_p_applies_when_top_p_is_disabled() {
        use super::Sampler;
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::sync::{Arc, Mutex};

        const DRAWS: usize = 400;
        let raw = vec![4.0f32, 0.0, 0.0, 0.0];
        let logits = || Tensor::from_vec(raw.clone(), 4, &Device::Cpu).unwrap();

        // temperature 1.0, top_k disabled, top_p disabled (1.0), min_p = 0.5.
        let mk = |min_p: f64| {
            Sampler::new(
                Some(1.0),
                0,
                None,
                None,
                None,
                None,
                None,
                -1,
                1.0,
                min_p,
                None,
                vec![],
                None,
            )
            .unwrap()
        };

        let draw = |sampler: &Sampler| {
            // One shared rng so the stream advances across draws.
            let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0xA11CE)));
            let mut tail_hits = 0usize;
            for _ in 0..DRAWS {
                let token = sampler
                    .sample(logits(), &[0u32], false, rng.clone(), false, false)
                    .unwrap()
                    .token;
                if token != 0 {
                    tail_hits += 1;
                }
            }
            tail_hits
        };

        // Control: with min_p off the tail is reachable, so the fixture is live
        // and the assertion below is not vacuous.
        assert!(
            draw(&mk(0.0)) > 0,
            "fixture is inert: the tail was never sampled even with min_p disabled"
        );

        assert_eq!(
            draw(&mk(0.5)),
            0,
            "min_p = 0.5 with top_p = 1.0 must exclude every token below \
             0.5 * max_prob; the tail was still sampled, so min_p was skipped"
        );
    }

    /// `sample(.., sample_speculative = true, ..)` is an ARGMAX at every
    /// temperature — it never draws.
    ///
    /// `sample_speculative_top_kp_min_p` ends in `argmax_f32`, and dividing by
    /// a positive temperature is monotonic, so the `Some(temperature)` arm
    /// picks the same token as the greedy arm. This is the property that makes
    /// accept-on-token-equality verification valid for greedy decoding only,
    /// and it is why `pipeline::sampling::sample_target_sequence_speculative`
    /// refuses to speculate above temperature 0.
    ///
    /// The test is a guard: if this branch is ever made stochastic without
    /// also implementing the `min(1, p/q)` accept test and the `max(0, p-q)`
    /// residual redraw, verification silently stops reproducing the target
    /// distribution — and this fails first.
    #[test]
    fn speculative_branch_is_argmax_at_any_temperature() {
        use super::Sampler;
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::collections::HashSet;
        use std::sync::{Arc, Mutex};

        // Softmax at temperature 2.0 is [0.343, 0.267, 0.208, 0.162]: a real
        // draw spreads over all four, an argmax never leaves token 0.
        let raw = vec![3.0f32, 2.5, 2.0, 1.5];
        let logits = || Tensor::from_vec(raw.clone(), 4, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(4242)));

        for temperature in [Some(0.5), Some(1.0), Some(2.0), None] {
            let sampler = Sampler::new(
                temperature,
                0,
                None,
                None,
                None,
                None,
                None,
                -1,
                1.0,
                0.0,
                None,
                vec![],
                None,
            )
            .unwrap();
            for _ in 0..64 {
                let token = sampler
                    .sample(logits(), &[1u32], false, rng.clone(), true, false)
                    .unwrap()
                    .token;
                assert_eq!(
                    token, 0,
                    "speculative sampling at temperature {temperature:?} returned {token}; it \
                     must be a deterministic argmax, or verification-by-equality is invalid"
                );
            }
        }

        // Control: the same sampler WITHOUT the speculative flag does draw, so
        // the assertion above is about the speculative branch and not about a
        // degenerate fixture.
        let stochastic = Sampler::new(
            Some(2.0),
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            None,
            vec![],
            None,
        )
        .unwrap();
        let mut seen = HashSet::new();
        for _ in 0..64 {
            seen.insert(
                stochastic
                    .sample(logits(), &[1u32], false, rng.clone(), false, false)
                    .unwrap()
                    .token,
            );
        }
        assert!(
            seen.len() > 1,
            "fixture is degenerate: non-speculative sampling never varied"
        );
    }

    /// Build a `logit_bias` sampler over a 4-token vocabulary.
    #[cfg(test)]
    fn bias_sampler(
        temperature: Option<f64>,
        bias: Option<std::collections::HashMap<u32, f32>>,
    ) -> super::Sampler {
        super::Sampler::new(
            temperature,
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            None,
            vec![],
            bias,
        )
        .unwrap()
    }

    /// `logit_bias` must actually reach sampling.
    ///
    /// Regression: `SamplingParams::logits_bias` was written by every API
    /// surface and read by none — `Sampler` had no field for it, `Sampler::new`
    /// no parameter, and `Sampler::sample` no bias term. A request setting it
    /// got 200 OK and completely unbiased output.
    ///
    /// Greedy over raw logits `[0, 0, 0, 5]` selects token 3. A +100 bias on
    /// token 1 must move that to 1; a -100 bias on token 3 must move it off 3.
    #[test]
    fn logit_bias_shifts_the_selected_token() {
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};

        let raw = vec![0.0f32, 0.0, 0.0, 5.0];
        let logits = || Tensor::from_vec(raw.clone(), 4, &Device::Cpu).unwrap();
        let rng = || Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0)));
        let sample = |s: &super::Sampler| {
            s.sample(logits(), &[0u32], false, rng(), false, false)
                .unwrap()
                .token
        };

        // Control: no bias, greedy picks the raw argmax.
        assert_eq!(sample(&bias_sampler(None, None)), 3);

        // Positive bias promotes a token that was not the argmax.
        assert_eq!(
            sample(&bias_sampler(None, Some(HashMap::from([(1u32, 100.0f32)])))),
            1,
            "a +100 logit_bias on token 1 must make it the argmax"
        );

        // Negative bias demotes the argmax.
        assert_ne!(
            sample(&bias_sampler(
                None,
                Some(HashMap::from([(3u32, -100.0f32)]))
            )),
            3,
            "a -100 logit_bias on token 3 must stop it being selected"
        );

        // The bias is applied before the temperature branch too, so a dominant
        // bias pins a stochastic sampler onto the biased token.
        let hot = bias_sampler(Some(1.0), Some(HashMap::from([(1u32, 100.0f32)])));
        let shared_rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(99)));
        for _ in 0..64 {
            assert_eq!(
                hot.sample(logits(), &[0u32], false, shared_rng.clone(), false, false)
                    .unwrap()
                    .token,
                1,
                "logit_bias must apply on the temperature path as well"
            );
        }
    }

    /// The dense-bias cache is keyed on the incoming logits, so the same
    /// `Sampler` used against two vocabulary sizes must bias both correctly
    /// rather than reuse a stale vector (or fail a broadcast).
    #[test]
    fn logit_bias_cache_revalidates_on_vocab_change() {
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};

        let sampler = bias_sampler(None, Some(HashMap::from([(1u32, 100.0f32)])));
        let rng = || Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0)));

        let small = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 5.0], 4, &Device::Cpu).unwrap();
        assert_eq!(
            sampler
                .sample(small, &[0u32], false, rng(), false, false)
                .unwrap()
                .token,
            1
        );

        let mut big_raw = vec![0.0f32; 9];
        big_raw[8] = 5.0;
        let big = Tensor::from_vec(big_raw, 9, &Device::Cpu).unwrap();
        assert_eq!(
            sampler
                .sample(big, &[0u32], false, rng(), false, false)
                .unwrap()
                .token,
            1,
            "the cached bias vector must be rebuilt for a different vocab size"
        );
    }

    /// Token ids past the end of the vocabulary are dropped with a warning
    /// rather than panicking or corrupting the distribution.
    #[test]
    fn logit_bias_ignores_out_of_vocab_ids() {
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};

        let sampler = bias_sampler(
            None,
            Some(HashMap::from([(999u32, 100.0f32), (1u32, 100.0f32)])),
        );
        let logits = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 5.0], 4, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0)));
        assert_eq!(
            sampler
                .sample(logits, &[0u32], false, rng, false, false)
                .unwrap()
                .token,
            1,
            "an out-of-vocab id must be ignored, leaving the in-range bias effective"
        );
    }

    /// `has_logits_bias` is what the GPU-autonomous decode path checks before
    /// taking a fast path that never applies the bias.
    #[test]
    fn has_logits_bias_reports_the_map() {
        use std::collections::HashMap;

        assert!(!bias_sampler(None, None).has_logits_bias());
        assert!(
            !bias_sampler(None, Some(HashMap::new())).has_logits_bias(),
            "an empty map is not a bias and must not disable the GPU fast path"
        );
        assert!(bias_sampler(None, Some(HashMap::from([(1u32, 1.0f32)]))).has_logits_bias());
    }

    /// A biased sampler is greedy but is **not** raw-argmax.
    ///
    /// This is the interaction the merge of this PR with master's
    /// `is_raw_argmax` created: both landed in the same place in `sampler.rs`
    /// and neither author saw the other. `sample` applies the bias to the raw
    /// logits before choosing, so `argmax(raw)` and what `sample` returns can
    /// differ — which is exactly what speculative verification compares. If
    /// `is_raw_argmax` ever answers `true` with a bias set, a spec-decode run
    /// silently emits tokens the user's `logit_bias` was supposed to have
    /// changed.
    #[test]
    fn a_biased_sampler_is_greedy_but_not_raw_argmax() {
        use candle_core::{Device, Tensor};
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};

        let biased = bias_sampler(None, Some(HashMap::from([(0u32, 100.0f32)])));
        assert!(biased.is_greedy(), "no temperature is set, so it is greedy");
        assert!(
            !biased.is_raw_argmax(),
            "a logit_bias rewrites the logits before the pick, so this sampler is \
             NOT argmax over the raw logits and no fast path may substitute one"
        );

        // And demonstrate the divergence the flag is protecting, so the flag is
        // not just asserted against itself: raw argmax here is token 3, but the
        // bias moves the answer to token 0.
        let raw = vec![0.0f32, 0.0, 0.0, 5.0];
        let logits = Tensor::from_vec(raw.clone(), 4, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0)));
        let picked = biased
            .sample(logits, &[0u32], false, rng, false, false)
            .unwrap()
            .token;
        let raw_argmax = 3u32;
        assert_eq!(
            picked, 0,
            "the +100 bias on token 0 must win over the raw maximum at token 3"
        );
        assert_ne!(
            picked, raw_argmax,
            "if these ever agree this test proves nothing — pick a fixture where \
             the bias actually changes the answer"
        );
    }
}

/// CPU-parity tests for the GPU radix top-k sampling path.
///
/// The GPU path is: radix-select the top `k_sel` tokens on-GPU, then run
/// `truncate_topk_candidates` + `sample_from_topk_candidates` on the selected
/// set. These tests drive that exact downstream logic with a CPU reference
/// selection (descending sort, take `k_sel` — the spec for the radix kernel)
/// and pin it against the CPU sampling path (`sample_top_kp_min_p`) at the
/// distribution level: same kept token set, same probabilities, and — because
/// the kept weights are re-ordered by token id — the same drawn token for the
/// same rng seed.
#[cfg(test)]
mod topk_parity_tests {
    use super::*;
    use rand::SeedableRng;

    const VOCAB: usize = 4096;
    const TEMPERATURE: f64 = 0.7;

    /// Deterministic peaked logits in [-6, 6] (tie-free in practice), spread
    /// wide enough that the top-1024 tokens carry >0.99 of the softmax mass
    /// (so pure-top-p nucleus containment holds, as it does for trained-LLM
    /// distributions).
    fn peaked_logits(seed: u64) -> Vec<f32> {
        let mut state: u64 = seed ^ 0xDEAD_BEEF_CAFE_F00D;
        (0..VOCAB)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 32) as u32 as f32 / u32::MAX as f32) * 12.0 - 6.0
            })
            .collect()
    }

    /// Full-vocab softmax probabilities exactly as `Sampler::sample`'s CPU
    /// path computes them (tensor division + `softmax_last_dim`).
    fn softmax_probs(logits: &[f32]) -> Vec<f32> {
        let t = Tensor::from_vec(logits.to_vec(), logits.len(), &Device::Cpu).unwrap();
        let t = (t / TEMPERATURE).unwrap();
        candle_nn::ops::softmax_last_dim(&t)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// CPU reference for the radix kernel: top `k_sel` (token, prob) pairs by
    /// descending probability (ascending-token tiebreak).
    fn reference_topk_candidates(probs: &[f32], k_sel: usize) -> Vec<(u32, f32)> {
        let mut pairs: Vec<(u32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, p)| (i as u32, *p))
            .collect();
        pairs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        pairs.truncate(k_sel.min(pairs.len()));
        pairs
    }

    fn make_sampler(top_k: i64, top_p: f64, min_p: f64) -> Sampler {
        Sampler::new(
            Some(TEMPERATURE),
            0,
            None,
            None,
            None,
            None,
            None,
            top_k,
            top_p,
            min_p,
            None,
            vec![],
            None,
        )
        .unwrap()
    }

    /// Mirror of `sample_fast_topk_gpu`'s candidate-count selection.
    fn k_sel_for(top_k: i64) -> usize {
        let max_k = *GPU_RADIX_TOPK_SIZES.last().unwrap();
        let k_needed = if top_k > 0 { top_k as usize } else { max_k };
        GPU_RADIX_TOPK_SIZES
            .iter()
            .copied()
            .find(|&s| s >= k_needed)
            .unwrap()
    }

    const CASES: &[(i64, f64, f64)] = &[
        (40, 1.0, 0.0),    // pure top-k
        (200, 0.9, 0.0),   // top-k + top-p
        (0, 0.9, 0.0),     // pure top-p (nucleus containment via peaked logits)
        (100, 0.95, 0.05), // top-k + top-p + min-p
        (64, 0.3, 0.0),    // aggressive top-p
        (50, 0.0, 0.2),    // top_p out of (0,1): top-p skipped, min-p still runs
        (64, 1.0, 0.5),    // top_p disabled: min-p must still run
    ];

    /// Same kept set + same probabilities as the CPU path's in-place zeroing.
    #[test]
    fn candidate_truncation_matches_cpu_zeroing() {
        for seed in [1u64, 2, 3] {
            let probs = softmax_probs(&peaked_logits(seed));
            for &(top_k, top_p, min_p) in CASES {
                let sampler = make_sampler(top_k, top_p, min_p);

                // CPU path: sample_top_kp_min_p zeroes everything it drops.
                let mut cpu_probs = probs.clone();
                let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
                sampler
                    .sample_top_kp_min_p(
                        &mut cpu_probs,
                        top_k,
                        top_p as f32,
                        min_p as f32,
                        false,
                        rng,
                    )
                    .unwrap();
                let mut cpu_kept: Vec<(u32, f32)> = cpu_probs
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| **p != 0.0)
                    .map(|(i, p)| (i as u32, *p))
                    .collect();
                cpu_kept.sort_unstable_by_key(|(t, _)| *t);

                // Candidate path with the CPU reference selection.
                let candidates = reference_topk_candidates(&probs, k_sel_for(top_k));
                let mut kept = sampler
                    .truncate_topk_candidates(candidates)
                    .unwrap_or_else(|| {
                        panic!(
                            "seed={seed} case=({top_k},{top_p},{min_p}): guard unexpectedly failed"
                        )
                    });
                kept.sort_unstable_by_key(|(t, _)| *t);

                assert_eq!(
                    kept, cpu_kept,
                    "seed={seed} case=({top_k},{top_p},{min_p}): kept set/probs diverge"
                );
            }
        }
    }

    /// Same rng seed => same sampled token + logprob as the CPU path.
    #[test]
    fn candidate_sampling_matches_cpu_token() {
        for seed in [7u64, 8, 9] {
            let probs = softmax_probs(&peaked_logits(seed));
            for &(top_k, top_p, min_p) in CASES {
                let sampler = make_sampler(top_k, top_p, min_p);

                let rng_cpu = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed * 31 + 1)));
                let mut cpu_probs = probs.clone();
                let expected = sampler
                    .sample_top_kp_min_p(
                        &mut cpu_probs,
                        top_k,
                        top_p as f32,
                        min_p as f32,
                        false,
                        rng_cpu,
                    )
                    .unwrap();

                let rng_gpu = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed * 31 + 1)));
                let candidates = reference_topk_candidates(&probs, k_sel_for(top_k));
                let actual = sampler
                    .sample_from_topk_candidates(candidates, rng_gpu)
                    .unwrap()
                    .unwrap_or_else(|| {
                        panic!(
                            "seed={seed} case=({top_k},{top_p},{min_p}): guard unexpectedly failed"
                        )
                    });

                assert_eq!(
                    actual.token, expected.token,
                    "seed={seed} case=({top_k},{top_p},{min_p}): token diverges"
                );
                assert!(
                    (actual.logprob - expected.logprob).abs() < 1e-6,
                    "seed={seed} case=({top_k},{top_p},{min_p}): logprob {} != {}",
                    actual.logprob,
                    expected.logprob
                );
            }
        }
    }

    /// Pure-top-p whose nucleus exceeds the candidate budget must refuse
    /// (return None) rather than truncate the distribution: near-uniform
    /// logits over 4096 tokens put ~25% of the mass in the top 1024, well
    /// under top_p=0.9.
    /// `top_p = 1.0, min_p > 0` must still truncate on the candidate path.
    ///
    /// Regression: min-p sat inside the `top_p in (0,1)` branch here too (to
    /// mirror `sample_top_kp_min_p`'s early return), so a top-k + min-p
    /// request returned the untouched top-k set. Hand-built candidates make
    /// the expected kept set exact.
    #[test]
    fn min_p_applies_to_candidates_when_top_p_disabled() {
        // Descending, and the smallest candidate is far under the threshold so
        // the containment guard is satisfied.
        let candidates = vec![(3u32, 0.90f32), (0, 0.06), (7, 0.03), (5, 0.01)];
        // top_k = 4 (== candidate count, so top-k closes the set), top_p = 1.0
        // (disabled), min_p = 0.5 => threshold 0.45, only 0.90 survives.
        let sampler = make_sampler(4, 1.0, 0.5);
        assert_eq!(
            sampler.truncate_topk_candidates(candidates).unwrap(),
            vec![(3u32, 0.90f32)],
            "min_p must be applied even though top_p is 1.0"
        );
    }

    /// min-p on a set that neither top-k nor top-p closed must refuse rather
    /// than truncate against an incomplete candidate list: the smallest
    /// candidate is above the min-p threshold, so tokens below the radix cut
    /// could also clear it.
    #[test]
    fn min_p_uncontained_candidates_fall_back() {
        let candidates = vec![(0u32, 0.30f32), (1, 0.28), (2, 0.26)];
        // top_k disabled, top_p disabled: nothing closes the set. Threshold is
        // 0.5 * 0.30 = 0.15 and the smallest candidate (0.26) is above it.
        let sampler = make_sampler(-1, 1.0, 0.5);
        assert!(
            sampler.truncate_topk_candidates(candidates).is_none(),
            "uncontained min-p must trigger the CPU fallback, not silent truncation"
        );
    }

    #[test]
    fn pure_topp_fat_nucleus_falls_back() {
        let logits: Vec<f32> = (0..VOCAB).map(|i| (i % 17) as f32 * 1e-3).collect();
        let probs = softmax_probs(&logits);
        let sampler = make_sampler(0, 0.9, 0.0);
        let candidates = reference_topk_candidates(&probs, k_sel_for(0));
        assert!(
            sampler.truncate_topk_candidates(candidates).is_none(),
            "fat nucleus must trigger the CPU fallback, not silent truncation"
        );
    }

    /// Tests for the `sample_fast` penalty-prologue gate.
    ///
    /// `sample_fast` used to upload the sequence's whole token history to the
    /// device and build a `counts` histogram + `presence` mask on *every* call,
    /// then discard all of it unless a penalty was configured. The prologue is
    /// now gated. These two tests pin the only way that gate can be wrong: the
    /// guard's notion of "no penalty is active" must agree exactly with the
    /// `match` arms it is skipping.
    ///
    /// Both use a sharply peaked logit vector so the on-device Gumbel-max draw
    /// is deterministic — the winning margin (20+ logits) is far outside the
    /// range Gumbel noise can overturn.
    mod sample_fast_penalty_gate {
        use crate::sampler::Sampler;
        use candle_core::{Device, Tensor};

        const VOCAB: usize = 8;

        /// logits[2] is the clear winner; logits[1] is the runner-up.
        fn logits() -> Tensor {
            let mut v = vec![0f32; VOCAB];
            v[1] = 30.0;
            v[2] = 60.0;
            Tensor::from_vec(v, VOCAB, &Device::Cpu).unwrap()
        }

        /// Token 2 occurs five times, so a frequency penalty bites it hard.
        fn context() -> Vec<u32> {
            vec![2, 2, 2, 2, 2]
        }

        fn sampler(
            frequency_penalty: Option<f32>,
            presence_penalty: Option<f32>,
            repetition_penalty: Option<f32>,
        ) -> Sampler {
            Sampler::new(
                None,
                0,
                None,
                frequency_penalty,
                presence_penalty,
                repetition_penalty,
                None,
                0,   // top_k off  -> no fast_sort_asc
                1.0, // top_p off  -> no fast_cumsum
                0.0, // min_p off
                None,
                vec![],
                None, // logits_bias — added by #151 after #160 wrote this fixture
            )
            .unwrap()
        }

        fn sampled_token(s: &Sampler) -> u32 {
            s.sample_fast(logits(), &context(), false, 0, 1.0, 0.0)
                .unwrap()
                .token
        }

        /// Penalties set to their *neutral* values (0.0 / 0.0 / 1.0) must be
        /// indistinguishable from penalties left unset. Before the gate both
        /// spellings built the prologue and then took no `match` arm; after the
        /// gate both skip it. If the guard's neutral test ever drifts from the
        /// arms' (`!= 0.` / `!= 1.`), this is what catches it.
        #[test]
        fn neutral_penalties_behave_exactly_like_unset_penalties() {
            let unset = sampled_token(&sampler(None, None, None));
            let neutral = sampled_token(&sampler(Some(0.0), Some(0.0), Some(1.0)));
            assert_eq!(
                unset, 2,
                "with no penalty active the peaked logit must win outright"
            );
            assert_eq!(
                neutral, unset,
                "neutral penalty values must not change the sampled token"
            );
        }

        /// The complement: a genuinely active penalty must still be applied.
        /// This is the regression the gate could plausibly introduce — skipping
        /// the prologue when it was actually needed. Token 2 carries a 30-logit
        /// lead; five occurrences at a penalty of 10.0 subtract 50, so token 1
        /// must take over.
        #[test]
        fn an_active_frequency_penalty_is_still_applied() {
            assert_eq!(
                sampled_token(&sampler(Some(10.0), None, None)),
                1,
                "an active frequency penalty must still suppress the repeated token"
            );
        }

        /// Presence and repetition penalties independently keep the prologue
        /// alive — the guard is an OR, and each disjunct must work on its own.
        #[test]
        fn presence_and_repetition_penalties_each_keep_the_prologue_alive() {
            // Presence subtracts a flat 40 from any token already seen,
            // dropping token 2 from 60 to 20, below token 1's 30.
            assert_eq!(
                sampled_token(&sampler(None, Some(40.0), None)),
                1,
                "an active presence penalty must still suppress the seen token"
            );
            // Repetition divides positive logits of seen tokens by 3.0,
            // dropping token 2 from 60 to 20, again below token 1's 30.
            assert_eq!(
                sampled_token(&sampler(None, None, Some(3.0))),
                1,
                "an active repetition penalty must still suppress the seen token"
            );
        }
    }
}

/// The server-default sort skip and the penalty/eligibility gates.
#[cfg(test)]
mod fast_path_gate_tests {
    use super::{DrySamplingParams, Sampler, PARTIAL_SORT_CALLS};
    use candle_core::{Device, Tensor};
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::sync::{Arc, Mutex};

    const VOCAB: usize = 512;

    /// Deterministic spread-out logits (same LCG family as the parity tests).
    fn logits() -> Tensor {
        let mut state: u64 = 0x5EED ^ 0xDEAD_BEEF_CAFE_F00D;
        let v: Vec<f32> = (0..VOCAB)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 32) as u32 as f32 / u32::MAX as f32) * 12.0 - 6.0
            })
            .collect();
        Tensor::from_vec(v, VOCAB, &Device::Cpu).unwrap()
    }

    fn sampler(top_k: i64, top_p: f64, min_p: f64) -> Sampler {
        Sampler::new(
            Some(0.7),
            0,
            None,
            None,
            None,
            None,
            None,
            top_k,
            top_p,
            min_p,
            None,
            vec![],
            None,
        )
        .unwrap()
    }

    /// Sampled token + how many times `partial_sort_top_k` ran during the
    /// call. `Sampler::sample` on this thread never crosses threads, so the
    /// thread-local delta is exact.
    fn sample_counting_sorts(s: &Sampler, seed: u64) -> (u32, u64) {
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(seed)));
        let before = PARTIAL_SORT_CALLS.with(|c| c.get());
        let token = s
            .sample(logits(), &[1u32, 2, 3], false, rng, false, true)
            .unwrap()
            .token;
        let sorts = PARTIAL_SORT_CALLS.with(|c| c.get()) - before;
        (token, sorts)
    }

    /// The server defaults (`top_k = -1`, `top_p = 1.0`, `min_p = 0.0` —
    /// `engine/add_request.rs`) must not run the full-vocab sort: nothing
    /// consumes it. Equivalence witness: the PRE-FIX code path replicated
    /// literally — `partial_sort_top_k(probs, n, zero_rest=true)` (which for
    /// `k = n` sorts everything and zeroes nothing) followed by
    /// `sample_multinomial` on the same rng seed. Same token, one sort vs
    /// zero. (`top_k = VOCAB` is not a usable witness: it now — correctly —
    /// takes the skip as well, being the same no-op filter.)
    #[test]
    fn server_default_top_k_skips_the_unconsumed_sort() {
        let s = sampler(-1, 1.0, 0.0);
        let (token_skip, sorts_skip) = sample_counting_sorts(&s, 42);
        assert_eq!(sorts_skip, 0, "server defaults must skip the sort entirely");

        // Pre-fix path, spelled out: softmax(logits / T), full sort with
        // zero-rest, multinomial. Filters are all out of range, so the sort's
        // output feeds nothing — exactly the work the skip removes.
        let probs_t =
            candle_nn::ops::softmax_last_dim(&(logits().to_dtype(candle_core::DType::F32).unwrap()
                / 0.7)
                .unwrap())
            .unwrap();
        let mut probs: Vec<f32> = probs_t.to_vec1().unwrap();
        let n = probs.len();
        let before = PARTIAL_SORT_CALLS.with(|c| c.get());
        let sorted = super::partial_sort_top_k(&mut probs, n, true);
        assert_eq!(
            PARTIAL_SORT_CALLS.with(|c| c.get()) - before,
            1,
            "the witness must actually sort, or this test proves nothing"
        );
        assert_eq!(sorted.len(), VOCAB, "k = n returns every element sorted");
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let token_sort = s.sample_multinomial(&probs, false, rng).unwrap().token;

        assert_eq!(
            token_skip, token_sort,
            "skipping the sort must not change the drawn token (identical probs, \
             identical rng stream)"
        );
        // Pin the drawn token so a silent change to the rng stream (e.g. an
        // extra draw before the multinomial) cannot pass as parity.
        assert_eq!(token_skip, 506, "seeded draw moved — the rng stream shifted");
    }

    /// MUTATION CONTROLS for the skip condition: each filter that consumes the
    /// sorted pairs must, on its own, keep the sort alive.
    #[test]
    fn each_active_filter_keeps_the_sort_alive() {
        // top_p in (0,1): the cumsum walk needs the sorted pairs.
        let (_, sorts) = sample_counting_sorts(&sampler(-1, 0.9, 0.0), 7);
        assert_eq!(sorts, 1, "active top_p must sort");
        // min_p in (0,1): the threshold scan needs the sorted pairs.
        let (_, sorts) = sample_counting_sorts(&sampler(-1, 1.0, 0.1), 7);
        assert_eq!(sorts, 1, "active min_p must sort");
        // top_k < vocab: the zero-rest pass needs the partition.
        let (_, sorts) = sample_counting_sorts(&sampler(32, 1.0, 0.0), 7);
        assert_eq!(sorts, 1, "truncating top_k must sort");
    }

    /// A DRY config left at its disabled default (`multiplier: 0.0`) is a
    /// proven no-op in `apply_dry_penalty`, so it must not count as a penalty
    /// — before this fix `has_penalties()` answered true for it, which
    /// disqualified the GPU fast path (and `is_raw_argmax`) for every request
    /// that merely *carried* the default DRY struct.
    ///
    /// `is_raw_argmax` is the public reader of `has_penalties`; both
    /// directions are pinned: disabled DRY ⇒ still raw-argmax, enabled DRY ⇒
    /// not.
    #[test]
    fn disabled_dry_is_not_a_penalty_and_enabled_dry_is() {
        // Dry params only survive `Sampler::new` when a tokenizer is present.
        let tokenizer = Arc::new(tokenizers::Tokenizer::new(
            tokenizers::models::bpe::BPE::default(),
        ));
        let mk = |dry: Option<DrySamplingParams>| {
            Sampler::new(
                None,
                0,
                Some(tokenizer.clone()),
                None,
                None,
                None,
                dry,
                -1,
                1.0,
                0.0,
                None,
                vec![],
                None,
            )
            .unwrap()
        };

        let none = mk(None);
        assert!(none.is_raw_argmax(), "baseline: no DRY at all");

        let disabled = mk(Some(DrySamplingParams::default()));
        assert!(
            disabled.is_raw_argmax(),
            "multiplier = 0.0 (the disabled default) cannot rewrite a logit; \
             it must not disqualify the fast path"
        );

        let enabled = mk(Some(DrySamplingParams {
            multiplier: 0.8,
            ..Default::default()
        }));
        assert!(
            !enabled.is_raw_argmax(),
            "an active DRY penalty rewrites logits and must disqualify"
        );

        // And the no-op claim itself: with a disabled DRY the sampled token
        // equals the no-DRY token on the same fixture (greedy, so exact).
        let rng = || Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0)));
        let ctx = [5u32, 5, 5, 5];
        let t_none = none
            .sample(logits(), &ctx, false, rng(), false, false)
            .unwrap()
            .token;
        let t_disabled = disabled
            .sample(logits(), &ctx, false, rng(), false, false)
            .unwrap()
            .token;
        assert_eq!(t_disabled, t_none, "disabled DRY must be a behavioural no-op");
    }

    /// `device_sampling_terminates` must mirror `sample`'s dispatch exactly —
    /// each disqualifier flips it independently, and the sort/vocab geometry
    /// matches the fast path's own bounds.
    #[test]
    fn device_eligibility_mirrors_the_fast_path_dispatch() {
        let big_vocab = 129_280; // V4: needs_sort can't fit shared memory
        let small_vocab = 4_096; // 4096.next_power_of_two() * 4 = 16 KB <= 48 KB

        // Greedy, no penalties: eligible at any vocab, any backend.
        let greedy = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            None,
            vec![],
            None,
        )
        .unwrap();
        assert!(greedy.device_sampling_terminates(big_vocab, false, false));
        assert!(greedy.device_sampling_terminates(big_vocab, true, false));
        // return_logprobs needs the CPU pipeline.
        assert!(!greedy.device_sampling_terminates(big_vocab, true, true));

        // Temperature without a sort (server defaults): eligible.
        let no_sort = sampler(-1, 1.0, 0.0);
        assert!(no_sort.device_sampling_terminates(big_vocab, false, false));

        // Temperature with a sort: small vocab fits candle's shared-memory
        // arg_sort on any backend; big vocab needs the CUDA radix path.
        let sorted = sampler(32, 1.0, 0.0);
        assert!(sorted.device_sampling_terminates(small_vocab, false, false));
        assert_eq!(
            sorted.device_sampling_terminates(big_vocab, true, false),
            cfg!(feature = "cuda"),
            "big-vocab sorted sampling terminates on device iff the CUDA radix \
             top-k exists in this build"
        );
        assert!(!sorted.device_sampling_terminates(big_vocab, false, false));

        // top_k above the radix dispatch cap: the kernel declines statically.
        let huge_k = sampler(2048, 1.0, 0.0);
        assert!(!huge_k.device_sampling_terminates(big_vocab, true, false));

        // A penalty disqualifies regardless of geometry.
        let penalised = Sampler::new(
            Some(0.7),
            0,
            None,
            Some(0.5),
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            None,
            vec![],
            None,
        )
        .unwrap();
        assert!(!penalised.device_sampling_terminates(big_vocab, true, false));
    }
}

/// $0 host-cost probe at V4's vocabulary.
///
/// **This times CPU code on a CPU tensor. It is evidence about HOST cost
/// only — it validates nothing about GPU behaviour (D14).** It exists to put
/// a number on what one sequence pays per token when it lands on the CPU
/// sampling pipeline, under (i) bench params (`top_k = 32`) and (ii) the
/// server defaults (`top_k = -1`, `top_p = 1.0`, `min_p = 0.0` —
/// `engine/add_request.rs`), which before the sort-skip fix ran a full
/// 129,280-element `sort_unstable_by` whose result nothing consumed.
///
/// Deliberately `#[ignore]`d: it is a manual instrument, not a gate. Run with
/// `cargo test -p mistralrs-core --release host_cost_probe -- --ignored --nocapture`.
///
/// The parity fixture above stays at `VOCAB = 4096` on purpose: its pure-top-p
/// cases depend on nucleus containment in the top 1024, which does not hold
/// for this spread at 129,280.
#[cfg(test)]
mod host_cost_probe {
    use super::Sampler;
    use candle_core::{Device, Tensor};
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::sync::{Arc, Mutex};

    /// DeepSeek-V4's vocabulary.
    const PROBE_VOCAB: usize = 129_280;

    /// Same LCG as `topk_parity_tests::peaked_logits`, at probe size.
    fn probe_logits(seed: u64) -> Vec<f32> {
        let mut state: u64 = seed ^ 0xDEAD_BEEF_CAFE_F00D;
        (0..PROBE_VOCAB)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 32) as u32 as f32 / u32::MAX as f32) * 12.0 - 6.0
            })
            .collect()
    }

    fn probe_sampler(top_k: i64, top_p: f64, min_p: f64) -> Sampler {
        Sampler::new(
            Some(0.7),
            0,
            None,
            None,
            None,
            None,
            None,
            top_k,
            top_p,
            min_p,
            None,
            vec![],
            None,
        )
        .unwrap()
    }

    fn time_us_per_call(sampler: &Sampler) -> (f64, f64) {
        const WARMUP: usize = 3;
        const ITERS: usize = 30;
        let logits = Tensor::from_vec(probe_logits(7), PROBE_VOCAB, &Device::Cpu).unwrap();
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(42)));
        let ctx = [1u32, 2, 3];
        for _ in 0..WARMUP {
            sampler
                .sample(logits.clone(), &ctx, false, rng.clone(), false, true)
                .unwrap();
        }
        let mut samples_us = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = std::time::Instant::now();
            sampler
                .sample(logits.clone(), &ctx, false, rng.clone(), false, true)
                .unwrap();
            samples_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        let mean = samples_us.iter().sum::<f64>() / ITERS as f64;
        let min = samples_us.iter().cloned().fold(f64::INFINITY, f64::min);
        (mean, min)
    }

    #[test]
    #[ignore = "manual $0 probe: prints per-call CPU sampling cost at V4 vocab; \
                run with --ignored --nocapture. Host-cost evidence only, never \
                GPU validation (D14)."]
    fn time_cpu_sample_at_v4_vocab() {
        let (bench_mean, bench_min) = time_us_per_call(&probe_sampler(32, 1.0, 0.0));
        let (srv_mean, srv_min) = time_us_per_call(&probe_sampler(-1, 1.0, 0.0));
        println!(
            "HOST_PROBE vocab={PROBE_VOCAB} bench(top_k=32): mean={bench_mean:.0}us min={bench_min:.0}us"
        );
        println!(
            "HOST_PROBE vocab={PROBE_VOCAB} server(top_k=-1,top_p=1.0): mean={srv_mean:.0}us min={srv_min:.0}us"
        );
    }
}

/// The greedy tie-break contract.
///
/// Arc has three greedy implementations and they did not agree with each other
/// on tied logits. These tests pin the one that can be proved on any host; the
/// divergence of the other two is documented on [`argmax_f32`].
#[cfg(test)]
mod argmax_tiebreak_tests {
    use super::argmax_f32;

    /// THE FIX. `Iterator::max_by` returns the **last** maximum on ties, so
    /// this returned index 3 where the argmax convention — numpy, PyTorch, and
    /// `arc_greedy_kernel` — returns 1.
    #[test]
    fn ties_resolve_to_the_lowest_index() {
        assert_eq!(argmax_f32(&[1.0, 5.0, 3.0, 5.0, 2.0]), 1);
    }

    /// A run of identical maxima, which is what a BF16 logit vector actually
    /// produces: one winner, chosen by position and nothing else.
    #[test]
    fn a_run_of_equal_maxima_picks_the_first() {
        assert_eq!(argmax_f32(&[0.0, 9.0, 9.0, 9.0, 9.0]), 1);
        assert_eq!(argmax_f32(&[9.0, 9.0, 9.0]), 0);
    }

    /// The ordinary case must not have moved.
    #[test]
    fn a_unique_maximum_is_unaffected() {
        assert_eq!(argmax_f32(&[1.0, 2.0, 7.0, 3.0]), 2);
        assert_eq!(argmax_f32(&[7.0, 2.0, 1.0]), 0);
        assert_eq!(argmax_f32(&[1.0, 2.0, 7.0]), 2);
    }

    /// Every logit masked out. A fully `-inf` row is what a constraint mask
    /// produces when it forbids everything, and it must still name a token
    /// rather than depend on scan order.
    #[test]
    fn all_negative_infinity_returns_the_first_index() {
        assert_eq!(
            argmax_f32(&[f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]),
            0
        );
    }

    /// A NaN must not win, and — the part a strict-`>` scan gets wrong when it
    /// is seeded with element 0 — a NaN in front must not suppress the real
    /// maximum behind it.
    #[test]
    fn nan_never_wins_and_never_poisons_the_scan() {
        assert_eq!(argmax_f32(&[f32::NAN, 5.0, 2.0]), 1);
        assert_eq!(argmax_f32(&[1.0, f32::NAN, 8.0]), 2);
        assert_eq!(argmax_f32(&[f32::NAN, f32::NAN]), 0);
    }

    /// Degenerate inputs keep the old fallback.
    #[test]
    fn empty_and_single_inputs() {
        assert_eq!(argmax_f32(&[]), 0);
        assert_eq!(argmax_f32(&[42.0]), 0);
    }
}
