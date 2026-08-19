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

/// Kernel dispatch sizes for the GPU radix top-k sampler path. Must mirror
/// `arc_cuda_graph::flashmlasparse::SUPPORTED_TOPK` (kept as a local const so
/// the candidate-truncation logic and its CPU-parity tests build without the
/// `cuda` feature; the CUDA wrapper re-validates and errors — triggering the
/// CPU fallback — if the tables ever diverge).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const GPU_RADIX_TOPK_SIZES: &[usize] = &[64, 128, 256, 512, 1024];

/// Find the index of the maximum element in a slice. O(n) scan.
#[inline]
fn argmax_f32(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
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

        let mut mut_ref_rng = &mut *rng.lock().expect("could not lock rng mutex");
        let next_token = distr.sample(&mut mut_ref_rng); // "Find the first item which has a weight *higher* than the chosen weight."
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
        let no_penalties = self.frequency_penalty.unwrap_or(0.0) == 0.0
            && self.presence_penalty.unwrap_or(0.0) == 0.0
            && self.repetition_penalty.unwrap_or(1.0) == 1.0
            && self.dry_params.is_none();
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
}
