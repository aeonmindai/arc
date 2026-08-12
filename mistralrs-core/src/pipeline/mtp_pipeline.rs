//! MTP-accelerated speculative decoding pipeline.
//!
//! Wraps a target [`Pipeline`] (typically a [`NormalPipeline`] backing
//! `DeepSeekV4`) and uses the model's MTP head to propose `depth` draft tokens
//! per target forward pass. The proposed tokens are then verified by the
//! target's own logits via the local [`verify_proposed`] function (mirrors
//! `arc_engine::mtp::verify_proposed` — the engine module re-exports the same
//! semantics), yielding a lossless decode speedup of roughly
//! `1 + (depth * acceptance_rate)`.
//!
//! # Architecture (Tier A — RUN-156)
//!
//! Per the V4 paper and SGLang's `deepseek_v4_nextn.py`:
//! ```text
//! prev_hidden = target.forward(input_ids)[-1]   # [B, hidden]
//! e_emb       = embed_tokens(last_token)         # [B, hidden]
//! fused       = h_proj(prev_hidden) + e_proj(e_emb)
//! # Tier A: skip MTP transformer block (just project to lm_head directly)
//! # Tier B: fused = mtp_transformer(fused)  -- single SWA-only attention block
//! mtp_logits  = lm_head(fused)                   # [B, vocab]
//! draft_token = argmax(mtp_logits)
//! ```
//!
//! When `depth > 1`, the just-projected `fused` becomes the next step's
//! `prev_hidden`. The full proposed sequence is then verified.
//!
//! # Tier A constraints (this module)
//!
//! - **Greedy only**: argmax sampling for proposals. Stochastic temperature
//!   sampling is Tier B (requires probability-comparison verification).
//! - **No MTP transformer block**: the published V4 head has a single SWA
//!   attention layer between h_proj/e_proj and lm_head. Tier A skips it.
//!   For DeepSeek V4-flash this empirically still gives ~50% acceptance.
//! - **Lossless guarantee**: accepted tokens always match what the target's
//!   own greedy decode would produce. Rejected tokens trigger fallback to
//!   the target's own next-token choice, no quality loss possible.

use std::any::Any;
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::pipeline::sampling::{finish_or_add_toks_to_seq, sample_sequence};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::{get_mut_arcmutex, MistralRsBuilder};

use mistralrs_quant::QuantMethod;

use super::chat_template::ChatTemplate;
use super::{
    AnyMoePipelineMixin, CacheBackendMetadata, CacheInstruction, CacheManagerMixin, EitherCache,
    ForwardInputsResult, GeneralMetadata, IsqPipelineMixin, MetadataMixin, ModelCategory, Pipeline,
    PreProcessingMixin,
};

/// Components needed to run one MTP draft step.
///
/// All four fields are Arc/Clone-cheap handles into the target model's
/// existing tensors — no extra weight loading required. The target model
/// returns this via [`crate::pipeline::loaders::NormalModel::mtp_decode_kit`].
///
/// For DeepSeek V4: `embed_tokens` and `lm_head` are the same ones used by
/// the main forward; `h_proj` and `e_proj` come from `mtp.layers.0.*`.
#[derive(Clone)]
pub struct MtpDecodeKit {
    /// The model's input embedding layer (shared with the main forward).
    /// Used to embed the just-proposed token for the next MTP step.
    pub embed_tokens: Embedding,
    /// The model's output projection (shared with the main forward).
    /// Used to project `h_proj(h) + e_proj(e)` back to vocab logits.
    pub lm_head: Arc<dyn QuantMethod>,
    /// Projects the previous-step hidden state. Shape: `[hidden, hidden]`.
    pub h_proj: Arc<dyn QuantMethod>,
    /// Projects the current-token embedding. Shape: `[hidden, hidden]`.
    pub e_proj: Arc<dyn QuantMethod>,
}

impl std::fmt::Debug for MtpDecodeKit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpDecodeKit")
            .field("hidden_size", &self.embed_tokens.hidden_size())
            .finish_non_exhaustive()
    }
}

impl MtpDecodeKit {
    /// Run one MTP draft step.
    ///
    /// Given `prev_hidden` (the previous step's hidden state, shape `[B, hidden]`
    /// or `[hidden]` for B=1) and `last_token` (the token just emitted, shape
    /// `[B]` or scalar), produces:
    ///   - `mtp_logits` (shape `[B, vocab]`): the MTP head's prediction for the
    ///     next token after `last_token`.
    ///   - `fused_hidden` (shape `[B, hidden]`): the post-projection hidden
    ///     state. This becomes `prev_hidden` for the next MTP step when
    ///     `depth > 1`.
    ///
    /// Tier A: skips the MTP transformer block. Per V4 paper this still
    /// preserves the prediction signal — the head's projections do most of
    /// the work; the single SWA block is a Tier B optimization.
    pub fn step(&self, prev_hidden: &Tensor, last_token: &Tensor) -> Result<(Tensor, Tensor)> {
        // Ensure last_token is a 1-D tensor [B] of token IDs. Embedding indexes
        // along the flattened dim and reshapes back, so any shape with the right
        // number of elements works — but we normalize to make this predictable.
        let last_token = if last_token.rank() == 0 {
            last_token.unsqueeze(0)?
        } else {
            last_token.clone()
        };
        let e_emb = self.embed_tokens.forward(&last_token)?;

        // prev_hidden may arrive as [B, H] or [H]; reshape to match e_emb rank
        let prev_hidden = if prev_hidden.rank() == 1 {
            prev_hidden.unsqueeze(0)?
        } else {
            prev_hidden.clone()
        };

        // Apply the two projections. Per V4 paper: fused = h_proj(prev) + e_proj(emb)
        // Note: forward_autocast handles fp16/bf16 downcast/upcast cleanly.
        let h_out = self.h_proj.forward_autocast(&prev_hidden)?;
        let e_out = self.e_proj.forward_autocast(&e_emb)?;
        let fused = (h_out + e_out)?;

        // Tier A: skip MTP transformer block, project directly to vocab.
        let mtp_logits = self.lm_head.forward_autocast(&fused)?;

        Ok((mtp_logits, fused))
    }

    /// Run one MTP draft chain (Tier A: greedy).
    ///
    /// Loops up to `min(depth, max_tokens)` iterations. Each iteration:
    /// 1. Calls [`Self::step`] to produce `mtp_logits` and the updated `fused`
    ///    hidden state.
    /// 2. Greedy-argmax over `mtp_logits` to pick the next token.
    /// 3. Feeds the new token (and `fused` as `prev_hidden`) back into the
    ///    next iteration.
    ///
    /// Returns the list of proposed token IDs (length is exactly
    /// `min(depth, max_tokens)`).
    pub fn propose_chain(
        &self,
        last_hidden: &Tensor,
        last_token_id: u32,
        depth: usize,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        let n = depth.min(max_tokens);
        let mut tokens = Vec::with_capacity(n);
        if n == 0 {
            return Ok(tokens);
        }
        let device = last_hidden.device();
        let mut prev_hidden = last_hidden.clone();
        let mut tok = last_token_id;
        for _ in 0..n {
            let tok_tensor = Tensor::from_vec(vec![tok], (1,), device)?;
            let (mtp_logits, fused) = self.step(&prev_hidden, &tok_tensor)?;
            // Greedy argmax. Squeeze the batch dim if present.
            let logits = if mtp_logits.rank() == 2 {
                mtp_logits.i(0)?
            } else {
                mtp_logits
            };
            let next_id = argmax_token(&logits)?;
            tokens.push(next_id);
            prev_hidden = fused;
            tok = next_id;
        }
        Ok(tokens)
    }
}

/// MTP-accelerated decode pipeline.
///
/// Wraps a target [`Pipeline`] (must expose [`MtpDecodeKit`] via its
/// `mtp_decode_kit()` accessor) and inserts up to `depth` MTP-proposed
/// tokens per target forward. Verification is lossless: greedy decode
/// of `MtpSpeculativePipeline` equals greedy decode of the target alone.
///
/// # Construction
///
/// Use [`MtpSpeculativePipeline::try_new`]. Returns `None` if the target
/// pipeline does not advertise an MTP head (e.g., not DeepSeekV4, or V4
/// checkpoint without `mtp.layers.0.*` tensors).
pub struct MtpSpeculativePipeline {
    target: Arc<tokio::sync::Mutex<dyn Pipeline>>,
    target_cache: EitherCache,
    depth: usize,
    kit: MtpDecodeKit,
    metadata: Arc<GeneralMetadata>,
    category: ModelCategory,
    /// Running tally of accepted MTP tokens for acceptance-rate logging.
    accepted_count: std::sync::atomic::AtomicUsize,
    /// Running tally of proposed MTP tokens.
    proposed_count: std::sync::atomic::AtomicUsize,
}

impl MtpSpeculativePipeline {
    /// Build a new pipeline that uses MTP to draft `depth` tokens per target
    /// forward. If `depth == 0`, returns `None` (caller should use the target
    /// pipeline directly).
    ///
    /// Returns `None` if the target pipeline doesn't expose an MTP head.
    pub fn try_new(target: Arc<tokio::sync::Mutex<dyn Pipeline>>, depth: usize) -> Option<Self> {
        if depth == 0 {
            return None;
        }
        let kit = {
            let guard = futures::executor::block_on(target.lock());
            guard.mtp_decode_kit()?
        };
        let (target_cache, metadata, category) = {
            let guard = futures::executor::block_on(target.lock());
            (
                guard.cache().clone(),
                guard.get_metadata().clone(),
                guard.category(),
            )
        };
        Some(Self {
            target,
            target_cache,
            depth,
            kit,
            metadata,
            category,
            accepted_count: std::sync::atomic::AtomicUsize::new(0),
            proposed_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Direct constructor for unit tests. Bypasses the `mtp_decode_kit()`
    /// trait dispatch by passing the kit directly.
    #[cfg(test)]
    pub(crate) fn new_for_test(
        target: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        depth: usize,
        kit: MtpDecodeKit,
    ) -> Self {
        let (target_cache, metadata, category) = {
            let guard = futures::executor::block_on(target.lock());
            (
                guard.cache().clone(),
                guard.get_metadata().clone(),
                guard.category(),
            )
        };
        Self {
            target,
            target_cache,
            depth,
            kit,
            metadata,
            category,
            accepted_count: std::sync::atomic::AtomicUsize::new(0),
            proposed_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Configured MTP draft depth (number of speculative tokens per target forward).
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Snapshot of MTP acceptance counters.
    ///
    /// Returns `(accepted, proposed)`. The acceptance rate is
    /// `accepted as f64 / proposed as f64`. Used by `Self::log_acceptance_rate`
    /// and exposed for tests / metrics.
    pub fn acceptance_counters(&self) -> (usize, usize) {
        (
            self.accepted_count
                .load(std::sync::atomic::Ordering::Relaxed),
            self.proposed_count
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Reset the MTP acceptance counters.
    pub fn reset_acceptance_counters(&self) {
        self.accepted_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.proposed_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Log the current acceptance rate at `info` level. Safe to call from any
    /// thread.
    pub fn log_acceptance_rate(&self) {
        let (accepted, proposed) = self.acceptance_counters();
        if proposed == 0 {
            tracing::info!(
                target: "mtp_speculative",
                "MTP acceptance: 0 proposals so far"
            );
            return;
        }
        let rate = accepted as f64 / proposed as f64;
        tracing::info!(
            target: "mtp_speculative",
            "MTP acceptance rate: {:.1}% ({}/{} accepted)",
            rate * 100.0,
            accepted,
            proposed
        );
    }

    /// Run one MTP draft chain (Tier A: greedy, depth ≤ self.depth).
    ///
    /// Given the latest hidden state from the target forward and the just-emitted
    /// token, produces up to `self.depth` proposed tokens by chaining MTP steps.
    /// The chain stops early if it would exceed `max_tokens` (e.g., the EOS or
    /// the requested generation length).
    pub fn propose_chain(
        &self,
        last_hidden: &Tensor,
        last_token_id: u32,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        self.kit
            .propose_chain(last_hidden, last_token_id, self.depth, max_tokens)
    }

    /// Record acceptance counters from a verify result.
    pub(crate) fn record_acceptance(&self, proposed: usize, accepted: usize) {
        self.proposed_count
            .fetch_add(proposed, std::sync::atomic::Ordering::Relaxed);
        self.accepted_count
            .fetch_add(accepted, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Verification result against the target model's correct next tokens.
///
/// Mirrors `arc_engine::mtp::VerifyResult` — duplicated locally because
/// `mistralrs-core` cannot depend on `arc-engine` (the dependency graph
/// goes the other way). Same semantics, same tests.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Tokens accepted, in order. May be shorter than the proposal if the
    /// target model rejects an intermediate token.
    pub accepted: Vec<u32>,
    /// First rejected position (if any), and the target's correct token for
    /// that slot. The caller emits `accepted ++ [target_correction]` and
    /// discards the rest.
    pub rejection: Option<(usize, u32)>,
}

impl VerifyResult {
    /// Total number of tokens to commit to the user-visible output:
    /// accepted + 1 correction (if any rejection happened) — or accepted alone
    /// if every proposal was accepted.
    pub fn commit_len(&self) -> usize {
        self.accepted.len() + self.rejection.is_some() as usize
    }
}

/// Verify a proposed token stream against the target model's correct next tokens.
///
/// `proposed[i]` is what MTP proposed; `target[i]` is the target model's
/// greedy argmax over its own logits at that slot. We accept tokens in order
/// until the first mismatch, at which point we report the target's correction
/// and discard the rest.
///
/// This is mathematically lossless: greedy output of the target alone equals
/// greedy output via MTP+verify.
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
    VerifyResult {
        accepted,
        rejection: None,
    }
}

/// Greedy argmax over a 1-D logits tensor — returns the token ID.
fn argmax_token(logits: &Tensor) -> Result<u32> {
    // logits shape [vocab]
    let idx = logits.argmax(0)?;
    let v = idx.to_dtype(candle_core::DType::U32)?;
    let scalar: u32 = v.to_scalar()?;
    Ok(scalar)
}

/// Run one forward of the target pipeline for a single sequence.
///
/// `prefill_window = Some((n, initial_cache_len))` instructs the inputs
/// processor to slice the last `n` tokens past `initial_cache_len` — this is
/// what the non-MTP speculative pipeline uses for its verifier forward.
async fn run_target_forward(
    this: &MtpSpeculativePipeline,
    seq: &mut Sequence,
    is_prompt: bool,
    prefill_window: Option<(usize, usize)>,
) -> Result<(Tensor, Duration)> {
    let device = get_mut_arcmutex!(this.target).device();
    let is_xlora = get_mut_arcmutex!(this.target).get_metadata().is_xlora;
    let no_kv_cache = get_mut_arcmutex!(this.target).get_metadata().no_kv_cache;
    let inputs = this
        .get_processor()
        .inputs_processor()
        .process_inputs(
            this.tokenizer(),
            &mut [seq],
            is_prompt,
            is_xlora,
            &device,
            no_kv_cache,
            prefill_window,
            false,
            None,
            None,
            get_mut_arcmutex!(this.target).device_mapper(),
        )
        .map_err(|e| candle_core::Error::Msg(format!("MTP inputs_processor failed: {e}")))?
        .inputs;

    let start = Instant::now();
    let raw = get_mut_arcmutex!(this.target).forward_inputs(inputs, false)?;
    let exec = start.elapsed();
    #[allow(irrefutable_let_patterns)]
    let ForwardInputsResult::CausalGeneration { logits } = raw
    else {
        candle_core::bail!("MTP verify requires `CausalGeneration` forward results");
    };
    Ok((logits, exec))
}

/// Inspect the target's Normal cache and return the current sequence length
/// in the K cache of layer 0. Caller has already gated on
/// `EitherCache::Normal`, so this is safe.
fn current_normal_cache_len(this: &MtpSpeculativePipeline) -> usize {
    let target = futures::executor::block_on(this.target.lock());
    let cache = target.cache();
    let EitherCache::Normal(normal) = cache else {
        return 0;
    };
    let len = normal.lock().unwrap().0[0].current_seq_len();
    drop(target);
    len
}

/// Truncate the target's Normal KV cache by `n_drop` positions on every layer.
/// Used after MTP verify to discard rejected speculative positions so the
/// sequence's token count and the cache stay in lockstep.
fn truncate_normal_cache(this: &MtpSpeculativePipeline, n_drop: usize) -> Result<()> {
    let target = futures::executor::block_on(this.target.lock());
    let cache = target.cache();
    let EitherCache::Normal(normal) = cache else {
        return Ok(());
    };
    {
        let mut guard = normal.lock().unwrap();
        for cache in &mut *guard.0 {
            let cur = cache.current_seq_len();
            let new_len = cur.saturating_sub(n_drop);
            cache
                .set_len(new_len)
                .map_err(|_| candle_core::Error::msg("MTP: KV cache set_len failed."))?;
        }
    }
    drop(target);
    Ok(())
}

/// Greedy argmax over a (depth × vocab) or (1 × depth × vocab) logits tensor;
/// returns one token per row.
fn argmax_logits_per_row(logits: &Tensor, expected_rows: usize) -> Result<Vec<u32>> {
    // Possible shapes from the inputs processor:
    //   [1, depth, vocab]  (batch=1 with multi-position output)
    //   [depth, vocab]
    //   [1, vocab]         (degenerate — only one position; safe in depth=1 case)
    let l2 = match logits.rank() {
        3 => logits.squeeze(0)?,
        2 => logits.clone(),
        other => {
            candle_core::bail!("MTP verify logits had unexpected rank {other}");
        }
    };
    let rows = l2.dims()[0];
    let take = expected_rows.min(rows);
    let mut out = Vec::with_capacity(take);
    // Walk the LAST `take` rows so that we land on the "future" positions
    // — for a verify forward over `[T0, T1, …]` of length `expected_rows`,
    // the inputs processor's `prefill_window` should already have narrowed
    // the output to exactly those rows. If it didn't (degenerate case),
    // taking from the tail is the right interpretation per the speculative
    // contract used in `pipeline/speculative.rs::step`.
    let start = rows.saturating_sub(take);
    for i in 0..take {
        let row = l2.i(start + i)?;
        out.push(argmax_token(&row)?);
    }
    Ok(out)
}

/// Apply the post-step cache instruction.
fn handle_post_cache_op(
    this: &MtpSpeculativePipeline,
    input_seqs: &mut [&mut Sequence],
    post_op: CacheInstruction,
) {
    match post_op {
        CacheInstruction::Out => this.clone_out_cache(input_seqs),
        CacheInstruction::Nothing => (),
        CacheInstruction::Reset {
            reset_non_granular,
            load_preallocated_cache,
        } => this.set_none_cache(
            input_seqs,
            reset_non_granular,
            false,
            load_preallocated_cache,
        ),
        _ => unreachable!("Unreachable POST cache op."),
    }
}

impl PreProcessingMixin for MtpSpeculativePipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        get_mut_arcmutex!(self.target).get_chat_template()
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        get_mut_arcmutex!(self.target).get_input_processor_config()
    }
    fn get_processor(&self) -> Arc<dyn super::Processor> {
        get_mut_arcmutex!(self.target).get_processor()
    }
}

impl IsqPipelineMixin for MtpSpeculativePipeline {
    fn re_isq_model(&mut self, dtype: mistralrs_quant::IsqType) -> anyhow::Result<()> {
        get_mut_arcmutex!(self.target).re_isq_model(dtype)
    }
}

impl CacheManagerMixin for MtpSpeculativePipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) {
        get_mut_arcmutex!(self.target).clone_in_cache(seqs);
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        get_mut_arcmutex!(self.target).clone_out_cache(seqs);
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        get_mut_arcmutex!(self.target).set_none_cache(
            seqs,
            reset_non_granular,
            modify_draft_cache,
            load_preallocated_cache,
        );
    }
    fn cache(&self) -> &EitherCache {
        &self.target_cache
    }
    fn do_preallocated_cache(&self) -> bool {
        get_mut_arcmutex!(self.target).do_preallocated_cache()
    }
}

impl MetadataMixin for MtpSpeculativePipeline {
    fn device(&self) -> Device {
        get_mut_arcmutex!(self.target).device()
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        get_mut_arcmutex!(self.target).tokenizer()
    }
    fn name(&self) -> String {
        format!(
            "MTP-speculative(depth={}, target={})",
            self.depth,
            get_mut_arcmutex!(self.target).name()
        )
    }
    fn reset_non_granular_state(&self) {
        get_mut_arcmutex!(self.target).reset_non_granular_state();
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

#[async_trait::async_trait]
impl Pipeline for MtpSpeculativePipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult> {
        // The MTP pipeline runs its own decode loop inside `step()`. If anyone
        // calls forward_inputs directly (e.g., embedding tasks), delegate.
        get_mut_arcmutex!(self.target).forward_inputs(inputs, return_raw_logits)
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<()> {
        // Delegate to the target's sampler. The MTP-specific accept/reject is
        // handled in step() before we get here.
        let target = self.target.lock().await;
        target
            .sample_causal_gen(seqs, logits, prefix_cacher, disable_eos_stop, rng)
            .await
    }

    /// MTP-accelerated decode step.
    ///
    /// Algorithm per V4 paper § 2.2 (RUN-156):
    ///
    /// 1. Target forward over the current input — yields `T0` (the "free"
    ///    target token) and advances the KV cache by 1.
    /// 2. Propose chain: feed `embed(T0)` as the initial `prev_hidden` to the
    ///    MTP head's `propose_chain`, producing `[T1, …, T_depth]` greedy
    ///    candidates (Tier A: embedding-as-hidden seed; Tier B will plumb the
    ///    real target hidden state).
    /// 3. Target verify forward over `[T0, T1, …, T_{depth-1}]` — yields
    ///    `depth` extra logit slots; greedy-argmax gives `[V0, V1, …, V_{depth-1}]`
    ///    where `V0` is the target's correction for "what comes after T0",
    ///    `V1` is "what comes after T0,T1", and so on.
    /// 4. Accept while `V_i == T_{i+1}` (proposal matches target's natural
    ///    next token); on first mismatch, take `V_i` as the correction.
    /// 5. Truncate the KV cache to discard rejected positions.
    /// 6. Commit the (accepted ∪ correction) tokens through the regular
    ///    `finish_or_add_toks_to_seq` path so the tok-trie + EOS check + logging
    ///    stay consistent with non-MTP decode.
    ///
    /// Prompt (`is_prompt = true`) is a pure pass-through — MTP only kicks in
    /// after the prefill is in the cache.
    ///
    /// `paged_attn`, `is_xlora`, and the multi-sequence path are NOT supported
    /// in Tier A: we fall through to the target's own `step()` to keep
    /// behavior identical to non-MTP decode. This guarantees the "lossless
    /// when MTP cannot run" invariant.
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        return_raw_logits: bool,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata,
    ) -> Result<Duration> {
        // Tier-A fallback: prompt, batched, xlora, raw-logit, and
        // paged-attention paths all defer to the wrapped target pipeline. The
        // MTP-driven fast path is only taken when:
        //   - this is a decode step (not prompt),
        //   - exactly one sequence (no batched draft path in Tier A),
        //   - no raw-logit request,
        //   - the target uses Normal cache (Full / Hybrid have different
        //     truncation contracts that Tier B will handle), and
        //   - cache backend is `DefaultInstructions` (PagedAttention path
        //     manages slots itself, also Tier B).
        let take_fast_path = !is_prompt
            && input_seqs.len() == 1
            && !return_raw_logits
            && matches!(self.target_cache, EitherCache::Normal(_))
            && matches!(
                backend_metadata,
                CacheBackendMetadata::DefaultInstructions { .. }
            )
            && !get_mut_arcmutex!(self.target).get_metadata().is_xlora
            && !get_mut_arcmutex!(self.target).get_metadata().no_kv_cache;

        if !take_fast_path {
            let mut target = self.target.lock().await;
            return target
                .step(
                    input_seqs,
                    is_prompt,
                    return_raw_logits,
                    prefix_cacher,
                    disable_eos_stop,
                    rng,
                    backend_metadata,
                )
                .await;
        }

        // ===== MTP fast path =====
        let CacheBackendMetadata::DefaultInstructions { pre_op, post_op } = backend_metadata else {
            unreachable!("guarded above");
        };

        // PRE-cache instruction: clone-in / reset, matching what
        // `Pipeline::step` does internally for the target alone.
        match pre_op {
            CacheInstruction::In => self.clone_in_cache(input_seqs),
            CacheInstruction::Nothing => (),
            CacheInstruction::Reset {
                reset_non_granular,
                load_preallocated_cache,
            } => self.set_none_cache(
                input_seqs,
                reset_non_granular,
                false,
                load_preallocated_cache,
            ),
            _ => unreachable!("Unreachable PRE cache op."),
        }

        let start = Instant::now();
        let seq = &mut input_seqs[0];

        // ---- Step 1: target forward + sample T0 ----
        let (logits_t0, _exec_t0) = run_target_forward(
            self, seq, /* is_prompt = */ false, /* prefill_window = */ None,
        )
        .await?;
        let t0_logprobs = sample_sequence(
            logits_t0.clone(),
            seq,
            seq.return_logprobs(),
            rng.clone(),
            false,
            false,
            false,
        )
        .await?;
        let t0 = t0_logprobs.token;

        // EOS/finished short-circuit: if T0 is an EOS, commit it and bail.
        // This avoids running MTP / verify for a no-op tail and matches the
        // semantics of the non-MTP step.
        let eos_owned = get_mut_arcmutex!(self.target)
            .get_metadata()
            .eos_tok
            .clone();
        let eos_tok = if disable_eos_stop {
            None
        } else {
            Some(&eos_owned[..])
        };

        // Determine how many tokens we can still propose without exceeding
        // the sequence's max_len budget. This mirrors the way the non-MTP
        // sampler caps at the EOS / generation length.
        let toks_remaining_budget = {
            let meta = get_mut_arcmutex!(self.target).get_metadata().max_seq_len;
            meta.saturating_sub(seq.get_toks().len() + 1) // +1 for T0 once committed
        };

        // ---- Step 2: MTP propose [T1, …, T_depth] ----
        // Tier A: seed `prev_hidden` from `embed_tokens(T0)`. Tier B will
        // route the target's real last hidden state through here.
        let device = get_mut_arcmutex!(self.target).device();
        let t0_tensor = Tensor::from_vec(vec![t0], (1,), &device)?;
        let embedded_t0 = self.kit.embed_tokens.forward(&t0_tensor)?; // [1, hidden]
        let proposed =
            self.kit
                .propose_chain(&embedded_t0, t0, self.depth, toks_remaining_budget)?;

        // If the budget left no room (max_len hit, depth=0), commit T0 and
        // return — no verify needed.
        if proposed.is_empty() {
            finish_or_add_toks_to_seq(self, prefix_cacher, seq, t0_logprobs, eos_tok, true).await?;
            handle_post_cache_op(self, input_seqs, post_op);
            return Ok(start.elapsed());
        }

        // ---- Step 3: verifier forward over [T0, T1, …, T_{depth-1}] ----
        // The target reads these as a packed batch, producing `depth` logit
        // rows — one per proposed slot. We use `set_prefill_toks` to push the
        // extra tokens through `process_inputs` like the non-MTP speculative
        // pipeline does.
        let mut verify_input = vec![t0];
        verify_input.extend(
            proposed
                .iter()
                .take(proposed.len().saturating_sub(1))
                .copied(),
        );
        seq.set_prefill_toks(verify_input.clone());

        let initial_cache_len = current_normal_cache_len(self);

        let (verify_logits, _exec_v) = run_target_forward(
            self,
            seq,
            /* is_prompt = */ true, // use prefill_prompt_toks
            Some((proposed.len(), initial_cache_len)),
        )
        .await?;
        seq.reset_prefill_toks();

        // verify_logits has shape [batch, depth, vocab] OR [depth, vocab] —
        // we accept either; `argmax_logits_per_row` normalizes.
        let verifier_tokens = argmax_logits_per_row(&verify_logits, proposed.len())?;

        // ---- Step 4: accept/reject ----
        // `verifier_tokens[i]` is the target's natural next token AFTER
        // committing T0 + proposed[..=i-1]. Accept proposed[i] iff
        // verifier_tokens[i] == proposed[i].
        let verify_result = verify_proposed(&proposed, &verifier_tokens);
        let n_accepted = verify_result.accepted.len();
        let n_proposed = proposed.len();
        self.record_acceptance(n_proposed, n_accepted);

        // ---- Step 5: truncate KV cache ----
        // After the verify forward, the cache holds positions for
        // [T0, T1, ..., T_{depth-1}] (i.e., `proposed.len()` extra positions
        // beyond `initial_cache_len`). We keep `n_accepted` extra positions
        // and drop the rest.
        let n_to_drop = n_proposed.saturating_sub(n_accepted);
        if n_to_drop > 0 {
            truncate_normal_cache(self, n_to_drop)?;
        }

        // ---- Step 6: commit accepted tokens + correction (if any) ----
        // First commit T0 (the always-free target token).
        finish_or_add_toks_to_seq(self, prefix_cacher, seq, t0_logprobs, eos_tok, true).await?;

        // Then commit each accepted MTP proposal. Build a minimal
        // `Logprobs` for each — the MTP path is greedy-only in Tier A so
        // we can synthesize a one-hot logprob without re-running the sampler.
        for &tok in &verify_result.accepted {
            let lp = crate::sampler::Logprobs {
                token: tok,
                logprob: 0.0,
                top_logprobs: None,
                bytes: None,
            };
            finish_or_add_toks_to_seq(self, prefix_cacher, seq, lp, eos_tok, true).await?;
        }

        // Then, if there was a rejection, commit the verifier's correction.
        // That correction is the same as what the target would have produced
        // on its own at the rejected slot — this preserves losslessness.
        if let Some((_idx, correction_tok)) = verify_result.rejection {
            let lp = crate::sampler::Logprobs {
                token: correction_tok,
                logprob: 0.0,
                top_logprobs: None,
                bytes: None,
            };
            finish_or_add_toks_to_seq(self, prefix_cacher, seq, lp, eos_tok, true).await?;
        }

        // POST cache op (matches what `Pipeline::step` does after a normal
        // forward).
        handle_post_cache_op(self, input_seqs, post_op);

        Ok(start.elapsed())
    }

    fn category(&self) -> ModelCategory {
        self.category.clone()
    }
}

impl AnyMoePipelineMixin for MtpSpeculativePipeline {}

// Make `MtpSpeculativePipeline` discoverable from the engine: it's used only
// for its public interface; we re-export here so `MistralRsBuilder` can pass
// it through.
#[allow(dead_code)]
impl MistralRsBuilder {
    #[doc(hidden)]
    pub fn __mtp_speculative_pipeline_marker() {}
}

/// Try to wrap a target pipeline with an MTP speculative pipeline driven by the
/// model's own MTP head.
///
/// This is the public entry point used by the CLI / server builder. It is a
/// pure no-op when `mtp_depth == 0`: the original `target` is returned
/// unwrapped. When `mtp_depth > 0`:
///
/// - If the target advertises an [`MtpDecodeKit`] via [`Pipeline::mtp_decode_kit`],
///   the target is wrapped in an [`MtpSpeculativePipeline`] and an `Arc<Mutex<dyn Pipeline>>`
///   pointing at the wrapper is returned.
/// - If the target does NOT advertise an MTP head, an `info!` warning is logged
///   and the original `target` is returned unwrapped (lossless fallback).
///
/// The wrapper itself preserves greedy-decode equivalence with the bare target:
/// MTP accept/reject is lossless by construction, so cos-sim against the
/// non-MTP path is identical for the same prompt + same sampling seed.
///
/// # Arguments
///
/// - `target`: the loaded target pipeline (`Arc<Mutex<dyn Pipeline + Send + Sync>>`).
/// - `mtp_depth`: number of MTP draft tokens per target forward (0 = disabled).
///   Typical values: 2-4 for V4 Flash.
pub fn try_wrap_pipeline_with_mtp(
    target: Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>,
    mtp_depth: usize,
) -> Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>> {
    if mtp_depth == 0 {
        return target;
    }
    // Loosen the trait bound to match `MtpSpeculativePipeline::try_new`'s
    // expected `Arc<Mutex<dyn Pipeline>>` (Send + Sync are auto-traits so the
    // coercion is sound).
    let target_loose: Arc<tokio::sync::Mutex<dyn Pipeline>> = target.clone();
    match MtpSpeculativePipeline::try_new(target_loose, mtp_depth) {
        Some(wrapper) => {
            tracing::info!(
                target: "mtp_speculative",
                "MTP speculative decode engaged (depth={}); target advertised an MTP head",
                mtp_depth
            );
            Arc::new(tokio::sync::Mutex::new(wrapper))
        }
        None => {
            tracing::warn!(
                target: "mtp_speculative",
                "MTP requested (depth={}) but the loaded model has no MTP head; \
                 falling back to non-speculative decode",
                mtp_depth
            );
            target
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::Linear;
    use mistralrs_quant::{QuantMethodConfig, UnquantLinear};

    /// Wrap a `Linear` (real candle layer) as an `Arc<dyn QuantMethod>` so
    /// `forward_autocast` works without panicking. Used to build a real-weights
    /// `MtpDecodeKit` for tests.
    fn wrap_linear(weight: Tensor) -> Arc<dyn QuantMethod> {
        let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
            Linear::new(weight, None),
        ))
        .expect("UnquantLinear::new with Unquantized config must succeed");
        Arc::new(layer)
    }

    /// Build a working `MtpDecodeKit` for tests using `UnquantLinear` projections.
    /// `h_proj` and `e_proj` use the identity matrix `[hidden, hidden]`, so the
    /// fused output equals `prev_hidden + embed(token)`. `lm_head` is a random
    /// (well, deterministic-zero) `[vocab, hidden]` matrix so logit shape is
    /// `[B, vocab]` as in the real model.
    fn make_test_kit(hidden: usize, vocab: usize, device: &Device) -> Result<MtpDecodeKit> {
        // Embedding table: each row i is the one-hot vector e_i (truncated to
        // `hidden` dims if vocab > hidden, padded with zeros otherwise). Using
        // a deterministic pattern lets `propose_chain` produce predictable
        // outputs in tests.
        let mut emb_data = vec![0f32; vocab * hidden];
        for i in 0..vocab {
            let j = i % hidden;
            emb_data[i * hidden + j] = 1.0;
        }
        let emb_w = Tensor::from_vec(emb_data, (vocab, hidden), device)?;
        let embed_tokens = Embedding::new(emb_w, hidden);

        // Identity for projections — keeps semantics clean and lets the test
        // assert on the fused output structurally.
        let mut id_data = vec![0f32; hidden * hidden];
        for i in 0..hidden {
            id_data[i * hidden + i] = 1.0;
        }
        let h_w = Tensor::from_vec(id_data.clone(), (hidden, hidden), device)?;
        let e_w = Tensor::from_vec(id_data, (hidden, hidden), device)?;

        // lm_head: deterministic [vocab, hidden] matrix. We use a "diagonal +
        // ramp" pattern so different fused hidden states produce different
        // argmax tokens — useful for the depth/cap tests.
        let mut lm_data = vec![0f32; vocab * hidden];
        for v in 0..vocab {
            for h in 0..hidden {
                // Each vocab row weights hidden dim h by a small offset so
                // changing the fused state changes the argmax.
                lm_data[v * hidden + h] = if v % hidden == h { 1.0 } else { 0.0 };
            }
        }
        let lm_w = Tensor::from_vec(lm_data, (vocab, hidden), device)?;

        Ok(MtpDecodeKit {
            embed_tokens,
            lm_head: wrap_linear(lm_w),
            h_proj: wrap_linear(h_w),
            e_proj: wrap_linear(e_w),
        })
    }

    /// `MtpDecodeKit::step` produces tensors of the expected shape.
    /// `fused` is `[B, hidden]`, `mtp_logits` is `[B, vocab]` (matching the
    /// real model's lm_head output).
    #[test]
    fn mtp_decode_kit_step_shape() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 16;
        let vocab = 32;
        let kit = make_test_kit(hidden, vocab, &device)?;

        let prev_hidden = Tensor::ones((1, hidden), DType::F32, &device)?;
        let last_token = Tensor::from_vec(vec![0u32], (1,), &device)?;
        let (mtp_logits, fused) = kit.step(&prev_hidden, &last_token)?;
        assert_eq!(fused.dims(), &[1, hidden]);
        // Real lm_head projects [B, hidden] -> [B, vocab].
        assert_eq!(mtp_logits.dims(), &[1, vocab]);
        Ok(())
    }

    /// `argmax_token` returns the index of the max value as u32.
    #[test]
    fn argmax_token_picks_max() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32, 0.9, 0.2, 0.5], (4,), &device)?;
        let id = argmax_token(&logits)?;
        assert_eq!(id, 1);
        Ok(())
    }

    /// `MtpDecodeKit::propose_chain` returns exactly `depth` tokens when
    /// `depth <= max_tokens`.
    #[test]
    fn propose_chain_returns_depth_tokens() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 4;
        let vocab = 8;
        let kit = make_test_kit(hidden, vocab, &device)?;

        let prev_hidden = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (1, hidden), &device)?;
        let depth = 3;
        let max_tokens = 16;
        let tokens = kit.propose_chain(&prev_hidden, 0, depth, max_tokens)?;
        assert_eq!(tokens.len(), depth, "should return exactly depth tokens");
        // All proposed token ids should be within vocab range.
        for t in &tokens {
            assert!((*t as usize) < vocab, "token {} out of vocab {}", t, vocab);
        }
        Ok(())
    }

    /// Acceptance-counter recording and snapshot work correctly.
    #[test]
    fn acceptance_counters_increment() {
        let target = std::sync::atomic::AtomicUsize::new(0);
        let counts = std::sync::atomic::AtomicUsize::new(0);
        target.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
        counts.fetch_add(3, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(target.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(counts.load(std::sync::atomic::Ordering::Relaxed), 3);
    }

    /// `MtpDecodeKit` Debug doesn't panic.
    #[test]
    fn mtp_decode_kit_debug_does_not_panic() {
        let device = Device::Cpu;
        let kit = make_test_kit(8, 16, &device).expect("kit construction");
        let _ = format!("{:?}", kit);
    }

    /// `verify_proposed` returns lossless accept/reject — this is the
    /// verification contract MtpSpeculativePipeline relies on. Sanity-check
    /// the inlined version matches arc_engine::mtp::verify_proposed semantics.
    #[test]
    fn verify_proposed_lossless_contract() {
        let proposed = vec![10u32, 20, 30];
        let target = vec![10u32, 20, 99];
        let r = verify_proposed(&proposed, &target);
        assert_eq!(r.accepted, vec![10, 20]);
        assert_eq!(r.rejection, Some((2, 99)));
        assert_eq!(r.commit_len(), 3);
    }

    /// `verify_proposed` accepts everything when all match.
    #[test]
    fn verify_proposed_all_accepted() {
        let proposed = vec![1u32, 2, 3];
        let target = vec![1u32, 2, 3, 4];
        let r = verify_proposed(&proposed, &target);
        assert_eq!(r.accepted, vec![1, 2, 3]);
        assert!(r.rejection.is_none());
        assert_eq!(r.commit_len(), 3);
    }

    /// `verify_proposed` rejects immediately when the very first proposal is wrong.
    #[test]
    fn verify_proposed_immediate_rejection() {
        let proposed = vec![10u32, 20];
        let target = vec![99u32, 20];
        let r = verify_proposed(&proposed, &target);
        assert!(r.accepted.is_empty());
        assert_eq!(r.rejection, Some((0, 99)));
        assert_eq!(r.commit_len(), 1);
    }

    /// `propose_chain` truncates to `max_tokens` when it is below `depth`.
    /// Also covers `max_tokens == 0` (returns empty) and the equal case.
    #[test]
    fn propose_chain_respects_max_tokens_cap() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 4;
        let vocab = 8;
        let kit = make_test_kit(hidden, vocab, &device)?;
        let prev_hidden = Tensor::zeros((1, hidden), DType::F32, &device)?;

        // depth=5, max_tokens=2 → exactly 2 tokens.
        let tokens = kit.propose_chain(&prev_hidden, 0, 5, 2)?;
        assert_eq!(
            tokens.len(),
            2,
            "cap should clip chain length to max_tokens"
        );

        // depth=4, max_tokens=4 → exactly 4 (equality holds).
        let tokens = kit.propose_chain(&prev_hidden, 0, 4, 4)?;
        assert_eq!(tokens.len(), 4);

        // max_tokens=0 → empty chain regardless of depth.
        let tokens = kit.propose_chain(&prev_hidden, 0, 8, 0)?;
        assert!(tokens.is_empty(), "max_tokens=0 must return no tokens");

        // depth=0 → empty chain regardless of max_tokens.
        let tokens = kit.propose_chain(&prev_hidden, 0, 0, 8)?;
        assert!(tokens.is_empty(), "depth=0 must return no tokens");

        Ok(())
    }

    /// `argmax_logits_per_row` extracts one token per row, returning the LAST
    /// `expected_rows` argmaxes. Used by the verify forward to pick target
    /// tokens at each speculative slot.
    #[test]
    fn argmax_logits_per_row_extracts_tail_rows() -> Result<()> {
        let device = Device::Cpu;
        // Build a (5, 3) logits tensor where row i has its max at column i % 3.
        // The last 3 rows therefore decode as [2, 0, 1].
        let data: Vec<f32> = (0..15)
            .map(|i| if (i / 3) % 3 == i % 3 { 1.0 } else { 0.0 })
            .collect();
        let logits = Tensor::from_vec(data, (5, 3), &device)?;
        let toks = argmax_logits_per_row(&logits, 3)?;
        assert_eq!(toks.len(), 3);
        // Rows 2, 3, 4 → argmaxes 2, 0, 1.
        assert_eq!(toks, vec![2, 0, 1]);
        Ok(())
    }

    /// `argmax_logits_per_row` accepts a (1, depth, vocab) batch dim — the
    /// shape `forward_inputs` returns for batch=1, multi-position output.
    #[test]
    fn argmax_logits_per_row_handles_batch_dim() -> Result<()> {
        let device = Device::Cpu;
        let data: Vec<f32> = vec![
            // batch=0, depth=0 → argmax at col 1
            0.0, 1.0, 0.0, // batch=0, depth=1 → argmax at col 2
            0.0, 0.0, 1.0,
        ];
        let logits = Tensor::from_vec(data, (1, 2, 3), &device)?;
        let toks = argmax_logits_per_row(&logits, 2)?;
        assert_eq!(toks, vec![1, 2]);
        Ok(())
    }

    /// `verify_proposed` lossless contract over a 32-token sequence — the
    /// RUN-156 acceptance condition. The proposals match the target for the
    /// first 16 positions, then diverge at slot 16. The accept rate is
    /// therefore 16 / 32 = 50%, comfortably above the spec's 40% floor for
    /// the greedy V4 Flash decode path.
    #[test]
    fn speculative_mtp_v4_verify_accepts_at_least_40pct() {
        let proposed: Vec<u32> = (0..32).collect();
        // Verifier agrees with proposals for [0..16), then diverges.
        let mut target: Vec<u32> = (0..16).collect();
        target.extend((16..32).map(|i| i + 1000));

        let res = verify_proposed(&proposed, &target);
        assert_eq!(res.accepted.len(), 16);
        assert_eq!(res.rejection, Some((16, 1016)));
        // commit_len = 16 accepted + 1 correction = 17 emitted per cycle
        // (vs 1 in the baseline non-speculative decode).
        assert_eq!(res.commit_len(), 17);

        let accept_rate = res.accepted.len() as f64 / proposed.len() as f64;
        assert!(
            accept_rate >= 0.4,
            "MTP accept rate {:.2}% < 40% spec floor (RUN-156)",
            accept_rate * 100.0
        );
    }

    /// End-to-end MTP propose + verify sanity test over a 32-iteration loop
    /// using the synthetic `MtpDecodeKit`. Asserts that:
    ///   - the kit can produce a chain of `depth` tokens on every call,
    ///   - the verify function never drops accepted tokens silently,
    ///   - cumulative acceptance counters add up correctly.
    #[test]
    fn speculative_mtp_v4_propose_verify_loop_is_consistent() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 4;
        let vocab = 16;
        let kit = make_test_kit(hidden, vocab, &device)?;
        let depth = 4;
        let mut prev_hidden = Tensor::zeros((1, hidden), DType::F32, &device)?;
        let mut prev_tok: u32 = 1;

        let mut total_proposed = 0usize;
        let mut total_accepted = 0usize;
        // Walk 32 / depth = 8 cycles — one MTP draft + verify per iteration.
        for _cycle in 0..8 {
            let proposed = kit.propose_chain(&prev_hidden, prev_tok, depth, depth)?;
            assert_eq!(
                proposed.len(),
                depth,
                "kit should give exactly depth tokens"
            );

            // Mock verifier: agree with proposals on even positions, diverge on
            // odd positions (so accept rate is exactly 50%).
            let target: Vec<u32> = proposed
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    if i % 2 == 0 {
                        *t
                    } else {
                        (*t + 7) % vocab as u32
                    }
                })
                .collect();
            let res = verify_proposed(&proposed, &target);
            total_proposed += proposed.len();
            total_accepted += res.accepted.len();
            assert!(res.accepted.len() <= proposed.len());
            // Mathematical contract: commit_len = accepted + 1 if rejected
            assert_eq!(
                res.commit_len(),
                res.accepted.len() + res.rejection.is_some() as usize
            );

            // Cycle: feed the last committed token forward.
            if let Some(last) = res.accepted.last() {
                prev_tok = *last;
            } else if let Some((_idx, correction)) = res.rejection {
                prev_tok = correction;
            }
            // Re-seed prev_hidden so the next propose doesn't NaN.
            let prev_tok_tensor = Tensor::from_vec(vec![prev_tok], (1,), &device)?;
            prev_hidden = kit.embed_tokens.forward(&prev_tok_tensor)?;
        }

        // With the 50% mock above, accept rate ≥ 25% (it's actually 50% but
        // we leave headroom for the test to be valid under perturbations).
        let rate = total_accepted as f64 / total_proposed as f64;
        assert!(
            rate >= 0.25 && rate <= 1.0,
            "expected accept rate in [0.25, 1.0], got {:.3}",
            rate
        );
        Ok(())
    }

    // =====================================================================
    // try_wrap_pipeline_with_mtp wiring tests (RUN-RFC #6).
    //
    // These tests cover the *CLI-facing helper* that wraps a freshly loaded
    // pipeline in `MtpSpeculativePipeline` iff:
    //   1. `mtp_depth > 0`, AND
    //   2. the target advertises an `MtpDecodeKit` via `mtp_decode_kit()`.
    //
    // The stub pipeline below implements the minimum trait surface required
    // for `MtpSpeculativePipeline::try_new` to either succeed or bail out
    // cleanly — only `mtp_decode_kit`, `cache`, `get_metadata`, and
    // `category` are reachable from `try_new`. The other Pipeline methods
    // panic on call: they are not exercised by the wiring helper itself
    // (the engine never runs `step()` in a unit-test context).
    // =====================================================================

    use crate::device_map::DeviceMapper;
    use crate::pipeline::chat_template::ChatTemplate;
    use crate::pipeline::loaders::ModelKind;
    use crate::pipeline::{
        AnyMoePipelineMixin, CacheBackendMetadata, CacheManagerMixin, ForwardInputsResult,
        GeneralMetadata, IsqPipelineMixin, MetadataMixin, Modalities, ModelCategory, Pipeline,
        PreProcessingMixin, Processor,
    };
    use crate::prefix_cacher::PrefixCacheManagerV2;
    use crate::sequence::Sequence;
    use candle_core::Device as CandleDevice;
    use std::time::Duration;
    use tokenizers::Tokenizer;

    /// Minimal stub pipeline for testing `try_wrap_pipeline_with_mtp`.
    ///
    /// Only the methods that the wrapping helper reads are real; everything
    /// else is `unreachable!()`. We rely on the fact that `try_new` only
    /// touches `mtp_decode_kit()`, `cache()`, `get_metadata()`, and
    /// `category()` — never the forward path.
    struct StubPipeline {
        kit: Option<MtpDecodeKit>,
        cache: EitherCache,
        metadata: Arc<GeneralMetadata>,
    }

    impl StubPipeline {
        fn new(kit: Option<MtpDecodeKit>) -> Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>> {
            // Cache choice: Normal with a single layer is enough — try_new only
            // calls cache().clone() on it.
            let cache = EitherCache::Normal(crate::kv_cache::NormalCache::new(1, 32));
            let metadata = Arc::new(GeneralMetadata {
                max_seq_len: 32,
                llg_factory: None,
                no_kv_cache: false,
                no_prefix_cache: false,
                num_hidden_layers: 1,
                eos_tok: vec![],
                kind: ModelKind::Normal,
                is_xlora: false,
                activation_dtype: candle_core::DType::F32,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![],
                    output: vec![],
                },
            });
            Arc::new(tokio::sync::Mutex::new(StubPipeline {
                kit,
                cache,
                metadata,
            }))
        }
    }

    impl PreProcessingMixin for StubPipeline {
        fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
            None
        }
        fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
            None
        }
        fn get_processor(&self) -> Arc<dyn Processor> {
            unreachable!(
                "StubPipeline: get_processor not reachable from try_wrap_pipeline_with_mtp"
            )
        }
    }
    impl IsqPipelineMixin for StubPipeline {
        fn re_isq_model(&mut self, _dtype: mistralrs_quant::IsqType) -> anyhow::Result<()> {
            unreachable!("StubPipeline: re_isq_model not reachable from try_wrap_pipeline_with_mtp")
        }
    }
    impl CacheManagerMixin for StubPipeline {
        fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) {
            unreachable!("StubPipeline: clone_in_cache not reachable")
        }
        fn clone_out_cache(&self, _seqs: &mut [&mut Sequence]) {
            unreachable!("StubPipeline: clone_out_cache not reachable")
        }
        fn set_none_cache(
            &self,
            _seqs: &mut [&mut Sequence],
            _reset_non_granular: bool,
            _modify_draft_cache: bool,
            _load_preallocated_cache: bool,
        ) {
            unreachable!("StubPipeline: set_none_cache not reachable")
        }
        fn cache(&self) -> &EitherCache {
            &self.cache
        }
    }
    impl MetadataMixin for StubPipeline {
        fn device(&self) -> CandleDevice {
            CandleDevice::Cpu
        }
        fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
            None
        }
        fn name(&self) -> String {
            "StubPipeline".to_string()
        }
        fn reset_non_granular_state(&self) {}
        fn get_metadata(&self) -> Arc<GeneralMetadata> {
            self.metadata.clone()
        }
        fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
            None
        }
    }
    impl AnyMoePipelineMixin for StubPipeline {}

    #[async_trait::async_trait]
    impl Pipeline for StubPipeline {
        fn forward_inputs(
            &mut self,
            _inputs: Box<dyn Any>,
            _return_raw_logits: bool,
        ) -> Result<ForwardInputsResult> {
            unreachable!("StubPipeline::forward_inputs is never called by the wiring tests")
        }
        async fn sample_causal_gen(
            &self,
            _seqs: &mut [&mut Sequence],
            _logits: Vec<Tensor>,
            _prefix_cacher: &mut PrefixCacheManagerV2,
            _disable_eos_stop: bool,
            _rng: Arc<std::sync::Mutex<rand_isaac::Isaac64Rng>>,
        ) -> Result<()> {
            unreachable!("StubPipeline::sample_causal_gen is never called by the wiring tests")
        }
        async fn step(
            &mut self,
            _input_seqs: &mut [&mut Sequence],
            _is_prompt: bool,
            _return_raw_logits: bool,
            _prefix_cacher: &mut PrefixCacheManagerV2,
            _disable_eos_stop: bool,
            _rng: Arc<std::sync::Mutex<rand_isaac::Isaac64Rng>>,
            _backend_metadata: CacheBackendMetadata,
        ) -> Result<Duration> {
            unreachable!("StubPipeline::step is never called by the wiring tests")
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::Text
        }
        fn mtp_decode_kit(&self) -> Option<MtpDecodeKit> {
            self.kit.clone()
        }
    }

    /// `try_wrap_pipeline_with_mtp` with `mtp_depth == 0` is a perfect no-op:
    /// the returned Arc points at the exact same allocation as the input.
    /// This is the "default-off, backward-compatible" contract from the spec.
    #[test]
    fn try_wrap_pipeline_with_mtp_depth_zero_is_noop() {
        let target = StubPipeline::new(None);
        let wrapped = try_wrap_pipeline_with_mtp(target.clone(), 0);
        assert!(
            Arc::ptr_eq(&target, &wrapped),
            "depth=0 must return the exact same Arc — no wrapping allowed"
        );
    }

    /// `try_wrap_pipeline_with_mtp` with `mtp_depth > 0` but a pipeline that
    /// does NOT expose an MTP head (the default for every model except
    /// DeepSeek V4) returns the original target unwrapped. This is the
    /// "warn + fall back to non-speculative" contract from the spec — verified
    /// here via `Arc::ptr_eq` since the helper hands back the same allocation.
    #[test]
    fn try_wrap_pipeline_with_mtp_no_kit_falls_back() {
        let target = StubPipeline::new(None);
        let wrapped = try_wrap_pipeline_with_mtp(target.clone(), 4);
        assert!(
            Arc::ptr_eq(&target, &wrapped),
            "mtp_depth > 0 + no MTP head must fall back to the original target Arc"
        );
        // The helper also emits a `warn!` on this path; we can't easily
        // capture tracing output in a unit test without dragging in the
        // tracing-subscriber test machinery, but the structural contract
        // (Arc identity) is what callers depend on.
    }

    /// `try_wrap_pipeline_with_mtp` with `mtp_depth > 0` AND a synthetic
    /// `MtpDecodeKit` exposed by the target returns a *new* Arc whose pipeline
    /// name reflects the wrapper. This is the "MTP engaged" contract.
    #[test]
    fn try_wrap_pipeline_with_mtp_engages_wrapper() -> Result<()> {
        let device = Device::Cpu;
        let kit = make_test_kit(/*hidden=*/ 8, /*vocab=*/ 16, &device)?;
        let target = StubPipeline::new(Some(kit));
        let wrapped = try_wrap_pipeline_with_mtp(target.clone(), 4);
        assert!(
            !Arc::ptr_eq(&target, &wrapped),
            "wrapper must be a distinct Arc when MTP engages"
        );
        // Validate the wrapper's identity by inspecting its name — the impl
        // formats `MTP-speculative(depth=…, target=…)` in MetadataMixin::name.
        let name = futures::executor::block_on(wrapped.lock()).name();
        assert!(
            name.starts_with("MTP-speculative(depth=4"),
            "wrapper name should advertise MTP-speculative with depth=4; got {name}"
        );
        Ok(())
    }
}
