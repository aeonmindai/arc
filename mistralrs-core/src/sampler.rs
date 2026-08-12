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
    pub top_n_logprobs: usize,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub stop_toks: Option<StopTokens>,
    pub max_len: Option<usize>,
    pub logits_bias: Option<HashMap<u32, f32>>,
    pub n_choices: usize,
    pub dry_params: Option<DrySamplingParams>,
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
            top_n_logprobs: 0,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop_toks: None,
            max_len: None,
            logits_bias: None,
            n_choices: 1,
            dry_params: None,
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
    logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    /// Cached Gumbel noise tensor to avoid reallocating it.
    gumbel_cache: Arc<Mutex<Option<Tensor>>>,
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
        logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    ) -> anyhow::Result<Self> {
        let temperature = if temperature.is_none_or(|v| v < 1e-7) {
            None
        } else {
            temperature
        };
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
            logits_processors,
            gumbel_cache: Arc::new(Mutex::new(None)),
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
        // Generate cached Gumbel noise (-log(-log(u))) once.
        let gumbel = {
            let mut guard = self.gumbel_cache.lock().unwrap();
            if guard.is_none() {
                let uniform = Tensor::rand(0f32, 1f32, log_probs.shape(), log_probs.device())?;
                let noise = uniform
                    .clamp(1e-20, 1.0)?
                    .log()? // ln(u)
                    .neg()? // -ln(u)
                    .log()? // ln(-ln(u))
                    .neg()?; // -ln(-ln(u))
                *guard = Some(noise);
            }
            guard.as_ref().unwrap().clone()
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

        if top_p <= 0.0 || top_p >= 1.0 {
            return self.sample_multinomial(probs, return_logprobs, rng);
        }

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

        if min_p <= 0.0 || min_p >= 1.0 {
            return self.sample_multinomial(probs, return_logprobs, rng);
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
            vec![],
        )
        .unwrap();
        assert!(!s_topp.is_greedy());
        assert_eq!(s_topp.temperature(), Some(0.7));
        assert_eq!(s_topp.top_p(), 0.95);
        assert_eq!(s_topp.top_k(), 40);
        assert_eq!(s_topp.frequency_penalty(), Some(0.1));
        assert_eq!(s_topp.presence_penalty(), Some(0.2));
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
            Sampler::new(None, 0, None, freq, pres, rep, None, -1, 1.0, 0.0, vec![]).unwrap()
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
}
