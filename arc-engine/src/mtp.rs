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
}
