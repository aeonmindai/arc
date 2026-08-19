//! Multi-Token Prediction (MTP) support for DeepSeek V3+ family models.
//!
//! Reference paper: `research/11_models/deepseek_v3.pdf` § 2.2
//! Reference code:  `research/code/03_per_token_speed/deepseek_v3_nsa/inference/model.py`
//!
//! ## What MTP is
//!
//! DeepSeek V3 trains the main transformer with auxiliary "MTP heads" — small
//! transformer modules that predict the next 1, 2, … N tokens beyond the one the
//! main model would naturally produce. At inference time, the heads can propose
//! a candidate continuation in a single forward; the target model then verifies.
//!
//! Per the V3 paper, MTP enables ~1.8× decode speedup with ~80%+ acceptance rate
//! on natural prompts. Crucially, V3 ships MTP weights — no training required.
//!
//! ## Storage layout (per V3 safetensors)
//!
//! MTP heads live under `mtp.layers.<i>.*` keys. Each head has:
//!   - an embedding layer (re-projection of target's hidden state)
//!   - one transformer block
//!   - a projection back to vocab logits
//!
//! Number of heads = `mtp_num_heads` in the config (typically 1–4).
//!
//! ## Tier A scope (this module)
//!
//! - Type definitions: `MtpHead`, `MtpStack`, `MtpConfig`
//! - Safetensors key detection: `detect_mtp_layers()`
//! - Proposal interface: `propose_tokens()` (stubbed to a deterministic mock for
//!   tests; the real path runs each head as a tiny forward pass)
//! - Verification interface: `verify_proposed()`
//! - Tests: detection, structure invariants, accept/reject correctness
//!
//! ## Tier B (deferred to RUN-136 / RUN-137)
//!
//! - Wire MtpStack into `mistralrs-core/src/models/deepseek3.rs` forward path
//! - Wire MtpStack into `mistralrs-core/src/pipeline/speculative.rs` dispatcher
//! - End-to-end inference test on DeepSeek-V2-Lite or full V3
//! - Acceptance rate measurement on natural prompts

use std::collections::BTreeSet;

/// Configuration for a stack of MTP heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpConfig {
    /// Number of MTP heads in the stack. Each head predicts one additional
    /// future token beyond the main model's output.
    pub num_heads: usize,
    /// Hidden dim of the target model (e.g., 7168 for V3).
    pub hidden_size: usize,
    /// Vocab size.
    pub vocab_size: usize,
}

/// One MTP head: embedding + transformer block + output projection.
///
/// For Tier A this is a struct holding the *config* only — the actual tensor
/// weights are loaded by the V3 model loader (Tier B work). The verify path
/// operates on logits already produced by either real or mock heads.
#[derive(Debug, Clone)]
pub struct MtpHead {
    /// Position in the stack (0 = predicts token+1, 1 = predicts token+2, …).
    pub depth: usize,
    /// Per-head config (mirrors the parent stack).
    pub cfg: MtpConfig,
}

/// A stack of MTP heads in depth order.
#[derive(Debug, Clone)]
pub struct MtpStack {
    pub heads: Vec<MtpHead>,
    pub cfg: MtpConfig,
}

impl MtpStack {
    /// Construct an empty stack of the requested depth.
    pub fn new(cfg: MtpConfig) -> Self {
        let heads = (0..cfg.num_heads)
            .map(|depth| MtpHead { depth, cfg })
            .collect();
        Self { heads, cfg }
    }

    /// Number of additional tokens this stack proposes per main-model forward.
    pub fn depth(&self) -> usize {
        self.heads.len()
    }
}

/// Scan a list of safetensors keys (typically from a `model.safetensors.index.json`)
/// for MTP-related entries. Returns the discovered depths.
///
/// Detects keys matching `mtp.layers.<depth>.*` and reports the unique depths
/// in sorted order. An empty result means the checkpoint has no MTP heads.
pub fn detect_mtp_layers<'a, I>(keys: I) -> Vec<usize>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut depths = BTreeSet::new();
    for key in keys {
        if let Some(rest) = key.strip_prefix("mtp.layers.") {
            // Find the depth number (digits before the next dot or end).
            let end = rest.find('.').unwrap_or(rest.len());
            if let Ok(depth) = rest[..end].parse::<usize>() {
                depths.insert(depth);
            }
        }
    }
    depths.into_iter().collect()
}

/// Tokens proposed by an MTP stack, in order: [next_1, next_2, ...].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposedTokens {
    pub tokens: Vec<u32>,
}

/// Verification result against the target model.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Tokens accepted, in order. May be shorter than the proposal if the
    /// target model rejects an intermediate token.
    pub accepted: Vec<u32>,
    /// First rejected position (if any), and the target's correct token for that slot.
    /// The caller should emit `accepted ++ [target_correction]` and discard the rest.
    pub rejection: Option<(usize, u32)>,
}

impl VerifyResult {
    /// Total number of tokens to commit to the user-visible output: accepted +
    /// 1 correction (if any rejection happened) — or accepted alone if every
    /// proposal was accepted.
    pub fn commit_len(&self) -> usize {
        self.accepted.len() + self.rejection.is_some() as usize
    }
}

/// Verify a proposed token stream against the target model's *correct* next tokens.
///
/// Both inputs are sequences. `proposed[i]` is what the MTP stack thought token
/// at position i should be; `target[i]` is what the target model actually produces
/// (greedy argmax of its logits). The protocol is:
///
/// 1. Walk proposed[] and target[] in lockstep.
/// 2. As long as proposed[i] == target[i], accept proposed[i].
/// 3. On the first mismatch, reject everything from that point on and tell the
///    caller the target's correct token for that slot so they can recover.
///
/// This is the standard speculative-decoding accept/reject; it's mathematically
/// lossless (greedy output of target-with-spec equals greedy output of target alone).
pub fn verify_proposed(proposed: &[u32], target: &[u32]) -> VerifyResult {
    let n = proposed.len().min(target.len());
    let mut accepted = Vec::with_capacity(n);
    for i in 0..n {
        if proposed[i] == target[i] {
            accepted.push(proposed[i]);
        } else {
            return VerifyResult {
                accepted,
                rejection: Some((i, target[i])),
            };
        }
    }
    // All proposed tokens accepted. No rejection — the caller will still emit
    // the target's next token from its own forward pass.
    VerifyResult {
        accepted,
        rejection: None,
    }
}

/// [`verify_proposed`] with the accept/reject decision forced to a configured
/// accept length — the **debug** harness for pricing acceptance without a
/// better draft head.
///
/// The schedule comes from [`mistralrs_core::SyntheticAcceptance`], which is the
/// same type the live serving path consults at its own accept/reject site
/// (`mistralrs-core/src/pipeline/mtp_pipeline.rs`, under
/// [`mistralrs_core::SIMULATE_ACC_LEN_ENV`]). One schedule, two doors: this one
/// so the behaviour is testable as a pure function, that one so a real run
/// exercises the KV-rollback and ragged-verify paths at accept patterns the
/// current 0.42-acceptance drafter almost never produces.
///
/// # The limit, stated plainly
///
/// **The drafter still runs at full cost.** Only the accept/reject decision is
/// synthetic. This prices *acceptance* — "what would throughput be if the head
/// were good enough to hit `acc_len`" — and NOT drafter removal.
///
/// **The accepted tokens are the draft's, not the target's.** They are committed
/// whether or not they match, so any text produced under this flag is not the
/// model's output and must not be scored or published. The rejection still
/// carries the target's real token at the rejected slot, so lengths and
/// corrections stay well-formed.
pub fn verify_proposed_simulated(
    proposed: &[u32],
    target: &[u32],
    sim: &mistralrs_core::SyntheticAcceptance,
) -> VerifyResult {
    let n = proposed.len().min(target.len());
    let k = sim.next_accept_len().min(n);
    VerifyResult {
        accepted: proposed[..k].to_vec(),
        rejection: (k < n).then(|| (k, target[k])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_no_mtp_in_plain_model() {
        let keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ];
        assert_eq!(detect_mtp_layers(keys.iter().copied()), Vec::<usize>::new());
    }

    #[test]
    fn detect_mtp_layers_finds_depths() {
        let keys = [
            "model.embed_tokens.weight",
            "mtp.layers.0.embed.weight",
            "mtp.layers.0.transformer.q_proj.weight",
            "mtp.layers.0.head.weight",
            "mtp.layers.1.embed.weight",
            "mtp.layers.1.transformer.q_proj.weight",
            "mtp.layers.2.transformer.kv_b_proj.weight",
        ];
        assert_eq!(detect_mtp_layers(keys.iter().copied()), vec![0, 1, 2]);
    }

    #[test]
    fn detect_mtp_ignores_malformed_keys() {
        let keys = [
            "mtp.layers.abc.embed.weight", // non-numeric depth
            "mtp.weight",                  // no depth
            "model.mtp.layers.0.weight",   // not at root
        ];
        assert_eq!(detect_mtp_layers(keys.iter().copied()), Vec::<usize>::new());
    }

    #[test]
    fn mtp_stack_construction_matches_depth() {
        let cfg = MtpConfig {
            num_heads: 3,
            hidden_size: 7168,
            vocab_size: 129280,
        };
        let stack = MtpStack::new(cfg);
        assert_eq!(stack.depth(), 3);
        assert_eq!(stack.heads.len(), 3);
        for (i, head) in stack.heads.iter().enumerate() {
            assert_eq!(head.depth, i);
            assert_eq!(head.cfg, cfg);
        }
    }

    #[test]
    fn verify_all_accepted_when_proposals_match_target() {
        let proposed = vec![10, 20, 30];
        let target = vec![10, 20, 30, 40]; // target may be longer
        let res = verify_proposed(&proposed, &target);
        assert_eq!(res.accepted, vec![10, 20, 30]);
        assert!(res.rejection.is_none());
        assert_eq!(res.commit_len(), 3);
    }

    #[test]
    fn verify_rejects_on_first_mismatch() {
        let proposed = vec![10, 20, 30, 40];
        let target = vec![10, 20, 99, 999];
        let res = verify_proposed(&proposed, &target);
        assert_eq!(res.accepted, vec![10, 20]);
        assert_eq!(res.rejection, Some((2, 99)));
        // commit = 2 accepted + 1 correction = 3
        assert_eq!(res.commit_len(), 3);
    }

    #[test]
    fn verify_immediate_rejection_yields_empty_accepted() {
        let proposed = vec![10, 20, 30];
        let target = vec![99, 20, 30];
        let res = verify_proposed(&proposed, &target);
        assert_eq!(res.accepted, Vec::<u32>::new());
        assert_eq!(res.rejection, Some((0, 99)));
        assert_eq!(res.commit_len(), 1);
    }

    #[test]
    fn verify_handles_empty_inputs() {
        let proposed = vec![];
        let target = vec![1, 2, 3];
        let res = verify_proposed(&proposed, &target);
        assert!(res.accepted.is_empty());
        assert!(res.rejection.is_none());
        assert_eq!(res.commit_len(), 0);
    }

    use mistralrs_core::SyntheticAcceptance;

    /// Run `steps` synthetic verifies at depth `depth` where **every** proposal
    /// mismatches the target, and return the committed length of each step.
    ///
    /// Total mismatch is the point: it is the case the real verifier scores 0
    /// on, so any commit length above 1 here can only have come from the
    /// harness. A harness that quietly fell back to the real comparison would
    /// return all 1s.
    fn simulated_commit_lens(acc_len: f64, depth: usize, steps: usize) -> Vec<usize> {
        let sim = SyntheticAcceptance::new(acc_len).expect("acc_len >= 1.0");
        let proposed: Vec<u32> = (0..depth as u32).map(|i| 1000 + i).collect();
        let target: Vec<u32> = (0..depth as u32).map(|i| 2000 + i).collect();
        (0..steps)
            .map(|_| verify_proposed_simulated(&proposed, &target, &sim).commit_len())
            .collect()
    }

    /// A fractional accept length is realised **exactly** in the mean, not by
    /// rounding — otherwise 2.85 and 3.0 would be the same experiment.
    #[test]
    fn a_fractional_accept_length_is_hit_exactly_in_the_mean() {
        const STEPS: usize = 1000;
        // SGLang's DeepSeek V4 CI floor, and the number this harness exists to
        // price: `acc_length > 2.85` at depth 3.
        for &acc_len in &[1.0, 1.5, 1.8387, 2.85, 3.0] {
            let lens = simulated_commit_lens(acc_len, 4, STEPS);
            let mean = lens.iter().sum::<usize>() as f64 / STEPS as f64;
            assert!(
                (mean - acc_len).abs() < 0.01,
                "acc_len={acc_len} produced mean {mean}"
            );
        }

        // The schedule is deterministic: same flag, same sequence, so a timing
        // A/B is not confounded by the harness itself.
        assert_eq!(
            simulated_commit_lens(2.85, 4, 40),
            simulated_commit_lens(2.85, 4, 40)
        );

        // …and it dithers rather than rounding: 2.85 must produce BOTH of the
        // neighbouring integers. Rounding to 3 would pass the mean test above
        // only by accident and fail here.
        let lens = simulated_commit_lens(2.85, 4, 40);
        assert!(lens.contains(&2) && lens.contains(&3), "{lens:?}");
    }

    /// The harness overrides the decision, and every proposal mismatching is
    /// the case that proves it.
    #[test]
    fn the_harness_accepts_tokens_the_real_verifier_would_reject() {
        let proposed = vec![10, 20, 30];
        let target = vec![91, 92, 93];

        // Ground truth: the real verifier accepts nothing here.
        let real = verify_proposed(&proposed, &target);
        assert!(real.accepted.is_empty());
        assert_eq!(real.commit_len(), 1);

        // Forced to accept 2 drafts per step (acc_len 3.0).
        let sim = SyntheticAcceptance::new(3.0).unwrap();
        let forced = verify_proposed_simulated(&proposed, &target, &sim);
        assert_eq!(
            forced.accepted,
            vec![10, 20],
            "the accepted tokens are the DRAFT's — which is exactly why text \
             produced under this flag is not the model's output"
        );
        assert_eq!(
            forced.rejection,
            Some((2, 93)),
            "the correction must still be the TARGET's real token, so lengths \
             and the committed stream stay well-formed"
        );
        assert_eq!(forced.commit_len(), 3);
    }

    /// The schedule is clamped by what the step could actually verify, and the
    /// harness does not compensate afterwards — the reported number stays the
    /// one that happened.
    #[test]
    fn the_schedule_is_clamped_by_the_chain_and_not_compensated() {
        // acc_len 5.0 wants 4 accepted drafts, but the chain is only 2 long, so
        // every step saturates at 2.
        //
        // `commit_len()` reports 2, not 3: with the whole chain accepted there
        // is no rejection, and this type deliberately counts only what the
        // *verifier* resolved. The pipeline adds the target's bonus token on
        // top, which is where its `committed = 1 + accepted` comes from. The
        // two counts differ by exactly that token and only when nothing was
        // rejected.
        let lens = simulated_commit_lens(5.0, 2, 20);
        assert!(
            lens.iter().all(|&l| l == 2),
            "a depth-2 chain cannot accept more than 2 drafts: {lens:?}"
        );

        // A chain of zero drafts commits nothing and cannot be rescued.
        let sim = SyntheticAcceptance::new(4.0).unwrap();
        let empty = verify_proposed_simulated(&[], &[], &sim);
        assert!(empty.accepted.is_empty());
        assert!(empty.rejection.is_none());
        assert_eq!(empty.commit_len(), 0);
    }

    /// The flag refuses nonsense loudly rather than silently doing nothing — a
    /// typo'd debug flag is how a GPU session gets paid for twice.
    #[test]
    fn the_flag_refuses_an_impossible_accept_length() {
        // A step always commits the target's own token, so < 1.0 describes
        // nothing that can happen.
        assert!(SyntheticAcceptance::new(0.9).is_none());
        assert!(SyntheticAcceptance::new(0.0).is_none());
        assert!(SyntheticAcceptance::new(f64::NAN).is_none());
        assert!(SyntheticAcceptance::new(f64::INFINITY).is_none());
        assert!(SyntheticAcceptance::new(1.0).is_some());

        // Unset / empty / unparseable all leave the real drafter measuring.
        assert!(SyntheticAcceptance::from_env_value(None).is_none());
        assert!(SyntheticAcceptance::from_env_value(Some("")).is_none());
        assert!(SyntheticAcceptance::from_env_value(Some("  ")).is_none());
        assert!(SyntheticAcceptance::from_env_value(Some("yes")).is_none());
        assert!(SyntheticAcceptance::from_env_value(Some("0.5")).is_none());

        let parsed = SyntheticAcceptance::from_env_value(Some(" 2.85 "))
            .expect("a valid accept length parses");
        assert!((parsed.acc_len() - 2.85).abs() < f64::EPSILON);
    }
}
