#![allow(clippy::cast_precision_loss)] // window sizes are tiny (<< f32 mantissa)

//! Arc Boost tier 1: serving-side, training-free quality stack.
//!
//! This module contains the model-agnostic pieces of the Arc Boost stack:
//!
//! - [`ConfidenceTracker`]: the DeepConf confidence signal (Fu et al., "Deep
//!   Think with Confidence", Meta AI, ICLR'26) — a rolling mean logprob of the
//!   generated tokens plus the *lowest group confidence* (minimum over sliding
//!   windows), which is the strongest known predictor of chain correctness.
//! - [`should_early_stop`]: the DeepConf-low online termination rule,
//!   **simplified**: instead of the paper's offline eta-percentile threshold
//!   computed from a warmup set of finished traces, a chain is culled when its
//!   lowest group confidence falls below `best / frac` in log space, where
//!   `best` is the best lowest-group-confidence seen so far across the sibling
//!   chains of the same request and `frac in (0, 1)` is the per-request
//!   `early_stop_confidence` knob. On the probability scale this culls chains
//!   whose per-token geometric-mean probability over their most uncertain
//!   window drops below the best chain's raised to the power `1/frac`.
//! - [`extract_answer`]: GSM8K-style final answer extraction (custom regex,
//!   then `\boxed{...}`, then last number).
//! - [`apply_vote`]: confidence-weighted / majority voting over the `k`
//!   candidate choices of a [`ChatCompletionResponse`] produced by a
//!   `n_votes: k` request. The k chains run as sibling sequences of one
//!   `SequenceGroup`, so the engine decodes them in the same forward-pass
//!   batch (amortizing MoE expert weight reads across chains).
//!
//! Confidence values throughout this module are **mean log10-probabilities**
//! (matching the `Logprobs::logprob` convention used by the sampler), so they
//! are `<= 0` with higher (closer to zero) meaning more confident. Note that
//! the pure-greedy GPU sampling fast path reports `logprob = 0.0`, which makes
//! the confidence signal degenerate under greedy decoding; voting requires
//! temperature sampling anyway.

use std::collections::VecDeque;

#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::response::ChatCompletionResponse;

/// Default sliding-window size (in generated tokens) for group confidence.
/// DeepConf uses 2048-token groups on long AIME traces; 64 is calibrated for
/// the shorter GSM8K-class traces Arc Boost targets.
pub const DEFAULT_CONFIDENCE_WINDOW: usize = 64;

/// Number of generated tokens before [`should_early_stop`] may fire.
/// Two full windows: one to fill the first group, one to let the group
/// statistic stabilize before it is trusted for culling.
pub const CONFIDENCE_WARMUP_TOKENS: usize = 2 * DEFAULT_CONFIDENCE_WINDOW;

/// Rolling confidence telemetry over the generated tokens of one sequence.
///
/// O(1) per token: a fixed-size window of the last `window_size` token
/// logprobs, their running sum, and the minimum window mean seen so far
/// ("lowest group confidence" in DeepConf terms, over a sliding rather than
/// disjoint grouping).
#[derive(Debug, Clone)]
pub struct ConfidenceTracker {
    window: VecDeque<f32>,
    window_size: usize,
    sum: f32,
    total: usize,
    lowest_group: Option<f32>,
}

impl Default for ConfidenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceTracker {
    pub fn new() -> Self {
        Self::with_window(DEFAULT_CONFIDENCE_WINDOW)
    }

    pub fn with_window(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size.max(1)),
            window_size: window_size.max(1),
            sum: 0.0,
            total: 0,
            lowest_group: None,
        }
    }

    /// Record the logprob (log10) of one generated token.
    pub fn push(&mut self, logprob: f32) {
        if self.window.len() == self.window_size {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
            }
        }
        self.window.push_back(logprob);
        self.sum += logprob;
        self.total += 1;
        if self.window.len() == self.window_size {
            let mean = self.sum / self.window_size as f32;
            self.lowest_group = Some(match self.lowest_group {
                Some(prev) => prev.min(mean),
                None => mean,
            });
        }
    }

    /// Rolling mean logprob over the current window (last `window_size`
    /// generated tokens). `None` before the first generated token.
    pub fn mean(&self) -> Option<f32> {
        if self.window.is_empty() {
            None
        } else {
            Some(self.sum / self.window.len() as f32)
        }
    }

    /// Lowest group confidence: minimum window-mean logprob over all sliding
    /// windows so far. `None` until one full window has been observed.
    pub fn lowest_group(&self) -> Option<f32> {
        self.lowest_group
    }

    /// Total number of tokens recorded.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Whether enough tokens have been observed for the early-stop rule.
    pub fn warmup_complete(&self) -> bool {
        self.total >= 2 * self.window_size
    }
}

/// DeepConf-low early termination rule (simplified — see module docs).
///
/// `chain` and `best` are lowest-group-confidence values (mean log10-probs,
/// `<= 0`); `frac` is the tolerance knob in `(0, 1)`. Returns `true` when the
/// chain should be culled. Degenerate signals (`best >= 0`, e.g. greedy
/// decoding's constant `logprob = 0.0`) never cull, and invalid `frac` values
/// disable the rule entirely.
pub fn should_early_stop(chain: f32, best: f32, frac: f32) -> bool {
    if !(frac > 0.0 && frac < 1.0) {
        return false;
    }
    if best >= 0.0 {
        return false;
    }
    chain < best / frac
}

/// How the winner of a `n_votes: k` request is selected.
#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteMode {
    /// One chain, one vote.
    Majority,
    /// Each chain's vote is weighted by `10^confidence` — the geometric-mean
    /// token probability over its most uncertain window (lowest group
    /// confidence, falling back to the rolling mean). Chains without a
    /// confidence signal weigh `10^0 = 1`, so this degrades gracefully to
    /// majority voting when the signal is absent.
    ConfidenceWeighted,
}

impl std::str::FromStr for VoteMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "majority" => Ok(Self::Majority),
            "confidence_weighted" | "confidence-weighted" => Ok(Self::ConfidenceWeighted),
            other => Err(format!(
                "Unknown vote_mode `{other}`; expected `majority` or `confidence_weighted`."
            )),
        }
    }
}

/// Per-chain voting record, returned in [`VoteOutcome::candidates`].
#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
pub struct VoteCandidate {
    /// The `Choice.index` of this chain.
    pub index: usize,
    /// The normalized extracted answer, if any.
    pub answer: Option<String>,
    /// The chain's confidence (lowest group confidence, falling back to the
    /// rolling mean logprob).
    pub confidence: Option<f32>,
    /// The weight this chain contributed to the tally.
    pub weight: f32,
}

/// Voting metadata attached to a [`ChatCompletionResponse`] by [`apply_vote`].
#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize)]
pub struct VoteOutcome {
    /// The vote mode that was applied.
    pub mode: VoteMode,
    /// `Choice.index` of the winning chain (which is moved to `choices[0]`).
    pub winner_index: usize,
    /// The winning normalized answer, if any chain produced one.
    pub winner_answer: Option<String>,
    /// All chains' voting records, in original choice order.
    pub candidates: Vec<VoteCandidate>,
}

/// Normalize an extracted answer for vote-equality purposes: numeric strings
/// (after comma stripping) are canonicalized through `f64`, everything else is
/// trimmed and lowercased.
pub fn normalize_answer(raw: &str) -> String {
    let trimmed = raw
        .trim()
        .trim_start_matches('$')
        .trim_end_matches('%')
        .trim_end_matches('.')
        .trim();
    let no_commas: String = trimmed.chars().filter(|c| *c != ',').collect();
    if let Ok(v) = no_commas.parse::<f64>() {
        if v.is_finite() {
            return format!("{v}");
        }
    }
    trimmed.to_lowercase()
}

/// Extract the content of the last `\boxed{...}` in `text`, handling nested
/// braces (which a plain regex cannot).
fn extract_last_boxed(text: &str) -> Option<String> {
    const MARKER: &str = "\\boxed{";
    let start = text.rfind(MARKER)?;
    let inner_start = start + MARKER.len();
    let mut depth = 1usize;
    for (i, c) in text[inner_start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[inner_start..inner_start + i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn last_number_regex() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[-+]?\d[\d,]*(?:\.\d+)?").expect("static regex"))
}

/// Validate a user-supplied answer-extraction regex (so API layers can reject
/// bad patterns at request time without depending on the regex crate).
pub fn validate_answer_regex(pattern: &str) -> Result<(), String> {
    Regex::new(pattern).map(|_| ()).map_err(|e| e.to_string())
}

/// Extract a final answer from a chain's output text.
///
/// Order of precedence:
/// 1. `custom` regex, if provided: the **last** match wins; capture group 1 is
///    used when present, otherwise the whole match.
/// 2. The last `\boxed{...}` (brace-matched).
/// 3. The last number in the text (covers GSM8K's `#### N` convention).
///
/// The result is passed through [`normalize_answer`].
pub fn extract_answer(text: &str, custom: Option<&Regex>) -> Option<String> {
    if let Some(re) = custom {
        if let Some(caps) = re.captures_iter(text).last() {
            let m = caps
                .get(1)
                .or_else(|| caps.get(0))
                .map(|m| m.as_str().to_string());
            if let Some(m) = m {
                return Some(normalize_answer(&m));
            }
        }
        // A custom regex that matches nothing falls through to the defaults.
    }
    if let Some(boxed) = extract_last_boxed(text) {
        return Some(normalize_answer(&boxed));
    }
    last_number_regex()
        .find_iter(text)
        .last()
        .map(|m| normalize_answer(m.as_str()))
}

/// The vote orchestrator's decision step: given a completed
/// [`ChatCompletionResponse`] whose `choices` are the `k` sampled chains of a
/// `n_votes: k` request, extract every chain's answer, tally the votes, and
/// rewrite the response so the winning chain is `choices[0]` (original
/// `index` fields are preserved) with the full tally in `response.vote`.
///
/// Winner selection:
/// - The answer with the largest total weight wins (ties: earliest chain).
/// - The winning *chain* is the highest-confidence chain among those voting
///   for the winning answer.
/// - If no chain produced an extractable answer, the highest-confidence chain
///   wins outright (falling back to `choices[0]`).
///
/// `answer_regex` must have been validated by the caller; an invalid pattern
/// is ignored (default extraction applies).
pub fn apply_vote(
    response: &mut ChatCompletionResponse,
    mode: VoteMode,
    answer_regex: Option<&str>,
) {
    let custom = answer_regex.and_then(|r| Regex::new(r).ok());

    let candidates: Vec<VoteCandidate> = response
        .choices
        .iter()
        .map(|choice| {
            let text = choice.message.content.as_deref().unwrap_or("");
            let answer = extract_answer(text, custom.as_ref());
            let confidence = choice.lowest_group_confidence.or(choice.confidence);
            let weight = if answer.is_some() {
                match mode {
                    VoteMode::Majority => 1.0,
                    VoteMode::ConfidenceWeighted => 10f32.powf(confidence.unwrap_or(0.0)),
                }
            } else {
                0.0
            };
            VoteCandidate {
                index: choice.index,
                answer,
                confidence,
                weight,
            }
        })
        .collect();

    // Tally in first-seen order for deterministic tie-breaking.
    let mut tally: Vec<(&str, f32)> = Vec::new();
    for cand in &candidates {
        if let Some(answer) = &cand.answer {
            match tally.iter_mut().find(|(a, _)| a == answer) {
                Some((_, w)) => *w += cand.weight,
                None => tally.push((answer.as_str(), cand.weight)),
            }
        }
    }

    let winner_answer: Option<String> = tally
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(answer, _)| answer.to_string());

    let conf_key = |c: &VoteCandidate| c.confidence.unwrap_or(f32::NEG_INFINITY);
    let winner_pos: usize = match &winner_answer {
        Some(answer) => {
            let mut best_pos = 0usize;
            let mut best_conf = f32::NEG_INFINITY;
            let mut found = false;
            for (pos, cand) in candidates.iter().enumerate() {
                if cand.answer.as_deref() == Some(answer.as_str())
                    && (!found || conf_key(cand) > best_conf)
                {
                    best_pos = pos;
                    best_conf = conf_key(cand);
                    found = true;
                }
            }
            best_pos
        }
        None => candidates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                conf_key(a)
                    .partial_cmp(&conf_key(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(pos, _)| pos)
            .unwrap_or(0),
    };

    let winner_index = candidates.get(winner_pos).map(|c| c.index).unwrap_or(0);

    // Move the winning choice to the front, preserving the relative order of
    // the rest (and everyone's original `index`).
    if winner_pos > 0 && winner_pos < response.choices.len() {
        let winner = response.choices.remove(winner_pos);
        response.choices.insert(0, winner);
    }

    response.vote = Some(VoteOutcome {
        mode,
        winner_index,
        winner_answer,
        candidates,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::{Choice, ResponseMessage, Usage};

    fn make_choice(index: usize, content: &str, confidence: Option<f32>) -> Choice {
        Choice {
            finish_reason: "stop".to_string(),
            index,
            message: ResponseMessage {
                content: Some(content.to_string()),
                role: "assistant".to_string(),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            confidence,
            lowest_group_confidence: confidence,
        }
    }

    fn make_response(choices: Vec<Choice>) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "0".to_string(),
            choices,
            created: 0,
            model: "test".to_string(),
            system_fingerprint: "local".to_string(),
            object: "chat.completion".to_string(),
            usage: Usage {
                completion_tokens: 0,
                prompt_tokens: 0,
                total_tokens: 0,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 0.0,
                total_time_sec: 0.0,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
            },
            vote: None,
        }
    }

    #[test]
    fn tracker_rolling_mean_and_lowest_group() {
        let mut t = ConfidenceTracker::with_window(4);
        assert_eq!(t.mean(), None);
        assert_eq!(t.lowest_group(), None);

        for _ in 0..4 {
            t.push(-1.0);
        }
        assert_eq!(t.mean(), Some(-1.0));
        assert_eq!(t.lowest_group(), Some(-1.0));
        assert!(!t.warmup_complete());

        // A bad stretch drags the window mean down; lowest group records it.
        for _ in 0..4 {
            t.push(-3.0);
        }
        assert_eq!(t.mean(), Some(-3.0));
        assert_eq!(t.lowest_group(), Some(-3.0));
        assert!(t.warmup_complete());

        // Recovery moves the rolling mean back up but lowest group is sticky.
        for _ in 0..8 {
            t.push(-0.5);
        }
        assert_eq!(t.mean(), Some(-0.5));
        assert_eq!(t.lowest_group(), Some(-3.0));
        assert_eq!(t.total(), 16);
    }

    #[test]
    fn tracker_partial_window_mean() {
        let mut t = ConfidenceTracker::with_window(4);
        t.push(-2.0);
        t.push(-4.0);
        assert_eq!(t.mean(), Some(-3.0));
        // No full window yet -> no group statistic.
        assert_eq!(t.lowest_group(), None);
    }

    #[test]
    fn early_stop_threshold_semantics() {
        // best = -0.1, frac = 0.5 -> threshold = -0.2.
        assert!(!should_early_stop(-0.15, -0.1, 0.5));
        assert!(should_early_stop(-0.25, -0.1, 0.5));
        // The best chain itself is never culled.
        assert!(!should_early_stop(-0.1, -0.1, 0.5));
        // Invalid frac disables the rule.
        assert!(!should_early_stop(-10.0, -0.1, 0.0));
        assert!(!should_early_stop(-10.0, -0.1, 1.0));
        assert!(!should_early_stop(-10.0, -0.1, -0.5));
        // Degenerate (greedy) confidence never culls.
        assert!(!should_early_stop(-10.0, 0.0, 0.5));
    }

    #[test]
    fn extract_gsm8k_style_answers() {
        assert_eq!(
            extract_answer("The answer is 42.", None),
            Some("42".to_string())
        );
        assert_eq!(
            extract_answer("She has 3 apples and buys 4 more.\n#### 7", None),
            Some("7".to_string())
        );
        assert_eq!(
            extract_answer("Total cost: $1,234.50", None),
            Some("1234.5".to_string())
        );
        assert_eq!(
            extract_answer("x = -3.5 therefore y = 7", None),
            Some("7".to_string())
        );
        assert_eq!(extract_answer("no numbers here", None), None);
    }

    #[test]
    fn extract_boxed_answers() {
        assert_eq!(
            extract_answer(r"thus \boxed{72} is the result", None),
            Some("72".to_string())
        );
        // Nested braces + boxed preferred over a later plain number.
        assert_eq!(
            extract_answer(r"we get \boxed{\frac{1}{2}} after 99 steps", None),
            Some(r"\frac{1}{2}".to_lowercase())
        );
        // Last boxed wins.
        assert_eq!(
            extract_answer(r"\boxed{1} ... actually \boxed{2}", None),
            Some("2".to_string())
        );
    }

    #[test]
    fn extract_with_custom_regex() {
        let re = Regex::new(r"answer:\s*(\w+)").unwrap();
        assert_eq!(
            extract_answer("answer: foo ... answer: BAR", Some(&re)),
            Some("bar".to_string())
        );
        // Non-matching custom regex falls back to defaults.
        assert_eq!(
            extract_answer("the result is 5", Some(&re)),
            Some("5".to_string())
        );
    }

    #[test]
    fn normalize_numeric_equivalence() {
        assert_eq!(normalize_answer("1,234"), normalize_answer("1234"));
        assert_eq!(normalize_answer("42."), normalize_answer("42"));
        assert_eq!(normalize_answer("42.0"), normalize_answer("42"));
        assert_eq!(normalize_answer("$7"), normalize_answer("7"));
        assert_eq!(normalize_answer(" Foo "), "foo");
    }

    #[test]
    fn majority_vote_picks_most_common_answer() {
        let mut resp = make_response(vec![
            make_choice(0, "I think it is 9.", Some(-0.05)),
            make_choice(1, "The answer is 7.", Some(-0.8)),
            make_choice(2, "So the answer is 7.", Some(-0.9)),
        ]);
        apply_vote(&mut resp, VoteMode::Majority, None);
        let vote = resp.vote.as_ref().unwrap();
        assert_eq!(vote.winner_answer.as_deref(), Some("7"));
        // Winner chain = higher-confidence chain among the 7-voters (index 1).
        assert_eq!(vote.winner_index, 1);
        assert_eq!(resp.choices[0].index, 1);
        // Relative order of the rest preserved.
        assert_eq!(resp.choices[1].index, 0);
        assert_eq!(resp.choices[2].index, 2);
        assert_eq!(vote.candidates.len(), 3);
    }

    #[test]
    fn confidence_weighted_vote_beats_majority() {
        // Two low-confidence chains agree on 9; one high-confidence chain
        // says 7. Weights: 10^-2 + 10^-2 = 0.02 vs 10^-0.1 ~= 0.79.
        let mut resp = make_response(vec![
            make_choice(0, "answer 9", Some(-2.0)),
            make_choice(1, "answer 9", Some(-2.0)),
            make_choice(2, "answer 7", Some(-0.1)),
        ]);
        apply_vote(&mut resp, VoteMode::ConfidenceWeighted, None);
        let vote = resp.vote.as_ref().unwrap();
        assert_eq!(vote.winner_answer.as_deref(), Some("7"));
        assert_eq!(vote.winner_index, 2);
        assert_eq!(resp.choices[0].index, 2);

        // Majority mode flips it.
        let mut resp = make_response(vec![
            make_choice(0, "answer 9", Some(-2.0)),
            make_choice(1, "answer 9", Some(-2.0)),
            make_choice(2, "answer 7", Some(-0.1)),
        ]);
        apply_vote(&mut resp, VoteMode::Majority, None);
        assert_eq!(
            resp.vote.as_ref().unwrap().winner_answer.as_deref(),
            Some("9")
        );
        assert_eq!(resp.choices[0].index, 0);
    }

    #[test]
    fn no_extractable_answers_falls_back_to_confidence() {
        let mut resp = make_response(vec![
            make_choice(0, "no idea", Some(-1.0)),
            make_choice(1, "still no idea", Some(-0.2)),
        ]);
        apply_vote(&mut resp, VoteMode::ConfidenceWeighted, None);
        let vote = resp.vote.as_ref().unwrap();
        assert_eq!(vote.winner_answer, None);
        assert_eq!(vote.winner_index, 1);
        assert_eq!(resp.choices[0].index, 1);
    }

    #[test]
    fn answers_normalized_before_tally() {
        let mut resp = make_response(vec![
            make_choice(0, "It costs $1,234.", None),
            make_choice(1, "The total is 1234", None),
            make_choice(2, "Maybe 99?", None),
        ]);
        apply_vote(&mut resp, VoteMode::Majority, None);
        let vote = resp.vote.as_ref().unwrap();
        assert_eq!(vote.winner_answer.as_deref(), Some("1234"));
    }
}
