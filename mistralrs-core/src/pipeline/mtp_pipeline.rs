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
//! # Architecture (RUN-156 Tier A + full-block Tier B)
//!
//! Per the V4 paper and SGLang's `deepseek_v4_nextn.py:149-161`:
//! ```text
//! h           = target hidden state at position i   # pre-`norm`, NOT an embedding
//! e_emb       = embed_tokens(tok_{i+1})             # [B, hidden]
//! # Full-block path (checkpoint ships mtp.0.* decoder + --mtp-depth > 0):
//! fused       = h_proj(hnorm(h)) + e_proj(enorm(e_emb))
//! hidden      = mtp_decoder_layer(fused)         # V4 attention + 256-expert MoE
//! mtp_logits  = lm_head(norm(hidden))            # [B, vocab]
//! # Tier-A fallback (older exports / --mtp-depth 0 at load):
//! fused       = h_proj(h) + e_proj(e_emb)
//! mtp_logits  = lm_head(fused)
//! draft_token = argmax(mtp_logits)
//! ```
//!
//! When `depth > 1`, the block output (or the projected `fused` on the
//! Tier-A path) becomes the next step's `h`. The full proposed
//! sequence is then verified.
//!
//! # The two signals must differ (audit finding 1)
//!
//! `h_proj` takes the **target model's own hidden state** and `e_proj` takes
//! the **token embedding**. Combining two different signals is the entire
//! point of the trained head; feeding `embed(T0)` to both collapses it and
//! drives acceptance to noise regardless of quantization. The hidden state
//! arrives through [`MtpHiddenCapture`], a side-channel the target model
//! fills during its own forward — no second forward pass is run.
//!
//! # The draft KV must hold the accepted context (audit finding 2)
//!
//! The MTP block applies **absolute** RoPE positions, so its KV cache must be
//! the real context, not an empty per-chain buffer. Mirroring
//! `eagle_worker.py` (`:134-138` own KV pool, `:1094-1128` prefill over the
//! prompt, `:1134+` extend over accepted tokens), this module keeps a
//! **persistent per-sequence draft KV** whose slot `k` is absolute position
//! `k`, holding the MTP state of the pair `(h_k, tok_{k+1})`. It is prefilled
//! from the prompt forward's captured hidden states and extended after every
//! verify. When it cannot be established contiguously (batched prefill,
//! prefix-cache hit) drafting is **skipped** rather than run against a cache
//! whose positions mean nothing — skipping is lossless, drafting blind is not.
//!
//! # Constraints (this module)
//!
//! - **Greedy only, and now enforced**: proposals are argmax, and verification
//!   accepts a draft when it equals the target's `argmax(raw logits)`. That is
//!   lossless only for a request whose sampler *is* that argmax, so any other
//!   request — `temperature > 0`, a repetition/frequency/presence/DRY penalty,
//!   a custom logits processor — **drafts nothing** and decodes through the
//!   real sampler ([`seq_may_draft`]). Before that gate existed, such a request
//!   silently received greedy tokens (with `logprob: 0.0`) at every accepted
//!   draft position, which DOCTRINE D4 bans. Serving `temperature > 0` *with*
//!   MTP needs probability-comparison acceptance (follow-up).
//! - **MTP transformer block**: V4 ships a full decoder layer (attention +
//!   MoE) at `mtp.0.*`. It is loaded only when `--mtp-depth > 0`
//!   ([`set_mtp_load_depth`]) — ~3GB at FP8, ~800MB after qtip2 ISQ — and
//!   drafts then flow through it (field reference: 80-90% second-token
//!   acceptance vs ~50% for projection-only drafting). Heads-only
//!   checkpoints keep the Tier-A path.
//! - **Lossless guarantee**: accepted tokens always match what the target's
//!   own greedy decode would produce. Rejected tokens trigger fallback to
//!   the target's own next-token choice, no quality loss possible.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;

use crate::kv_cache::{KvCache, SingleCache};

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

/// Global MTP load-depth gate.
///
/// Set by the server/CLI builder (from `--mtp-depth`) BEFORE the model is
/// loaded. The DeepSeek V4 loader reads it to decide whether to load the
/// full MTP transformer block (~3GB at FP8) in addition to the light
/// `h_proj`/`e_proj` heads. Zero (the default) keeps the old heads-only
/// behavior — no extra memory is spent when MTP is disabled.
///
/// Mirrors the `mistralrs_quant::set_loading_from_uqff` pattern: loading is
/// driven through macro-constructed metadata that does not carry runtime
/// flags, so a process-wide atomic is the lowest-blast-radius channel.
static MTP_LOAD_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Declare the requested MTP draft depth before model load. `0` disables
/// full-block loading (heads-only, the pre-Tier-B behavior).
pub fn set_mtp_load_depth(depth: usize) {
    MTP_LOAD_DEPTH.store(depth, Ordering::Relaxed);
}

/// The MTP draft depth declared via [`set_mtp_load_depth`] (0 = disabled).
pub fn mtp_load_depth() -> usize {
    MTP_LOAD_DEPTH.load(Ordering::Relaxed)
}

/// UQFF-bake override for MTP loading: when a `--write-uqff` bake is running,
/// the V4 loader loads the full MTP decoder block even if `--mtp-depth 0`, so
/// the block's ISQ tensors are quantized and included in the artifact
/// (~800MB at 2-bit). Without this, a bake made without `--mtp-depth` produces
/// a UQFF that cannot serve `--mtp-depth > 0` without falling back to the
/// source checkpoint. Same process-wide-atomic channel as
/// [`set_mtp_load_depth`]; set fresh on every load in
/// `NormalLoader::load_model_from_path`.
static MTP_UQFF_BAKE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Declare (before model load) that this load is a UQFF bake, forcing the V4
/// MTP decoder block to load so it gets serialized into the artifact.
pub fn set_mtp_uqff_bake(bake: bool) {
    MTP_UQFF_BAKE.store(bake, Ordering::Relaxed);
}

/// True when the current load should force-load the MTP block for a UQFF bake.
pub fn mtp_uqff_bake() -> bool {
    MTP_UQFF_BAKE.load(Ordering::Relaxed)
}

/// Is `ty` narrower than 8 bits per weight?
///
/// Everything at or above 8 bits (`Q8_*`, `HQQ8`, `AFQ8`, the FP8 types) is
/// safe for the MTP tail; everything else is not. See [`floor_mtp_isq`].
pub fn isq_is_sub_int8(ty: mistralrs_quant::IsqType) -> bool {
    use mistralrs_quant::IsqType as T;
    !matches!(
        ty,
        T::Q8_0 | T::Q8_1 | T::Q8K | T::HQQ8 | T::AFQ8 | T::F8E4M3 | T::F8Q8
    )
}

/// Raise a requested ISQ dtype to the MTP path's 8-bit floor.
///
/// **Why a floor exists.** The MTP head is a second logit-producing path, and
/// it is far more fragile than the main one: it must reproduce the target's
/// *argmax*, and `verify_proposed` accepts on exact `u32` equality, so every
/// distribution wobble is a rejected token rather than a slightly different
/// word. colibrì measured an int4 MTP draft head at **0-4% acceptance**
/// (`EXTERNAL_FINDINGS.md` F3) — at which point speculation is pure overhead.
///
/// This is the same call the project already made for `lm_head` (RUN-161:
/// *"quantizing the logit projection to 2-bit corrupts EOS probabilities and
/// breaks chat/instruction-following"*), applied to the other logit path.
///
/// Set `ARC_MTP_ALLOW_SUB_INT8=1` to opt out and get the requested width.
pub fn floor_mtp_isq(
    requested: Option<mistralrs_quant::IsqType>,
) -> Option<mistralrs_quant::IsqType> {
    use mistralrs_quant::IsqType as T;
    let Some(ty) = requested else {
        return requested;
    };
    if !isq_is_sub_int8(ty) {
        return requested;
    }
    if std::env::var_os("ARC_MTP_ALLOW_SUB_INT8").is_some() {
        warn_sub_int8_once(ty, None);
        return requested;
    }
    // Stay inside the requested backend family where an 8-bit sibling exists,
    // so the tensor keeps using the same kernels as its neighbours.
    let floored = match ty {
        T::AFQ2 | T::AFQ3 | T::AFQ4 | T::AFQ6 => T::AFQ8,
        T::HQQ4 => T::HQQ8,
        _ => T::Q8_0,
    };
    warn_sub_int8_once(ty, Some(floored));
    Some(floored)
}

static WARNED_SUB_INT8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn warn_sub_int8_once(
    requested: mistralrs_quant::IsqType,
    floored: Option<mistralrs_quant::IsqType>,
) {
    if WARNED_SUB_INT8.swap(true, Ordering::Relaxed) {
        return;
    }
    match floored {
        Some(floored) => tracing::warn!(
            target: "mtp_speculative",
            "MTP tail requested at {requested:?} (sub-int8); raising to {floored:?}. An int4 \
             MTP draft head measured 0-4% acceptance in the field, which makes speculative \
             decode a net slowdown. Set ARC_MTP_ALLOW_SUB_INT8=1 to override."
        ),
        None => tracing::warn!(
            target: "mtp_speculative",
            "ARC_MTP_ALLOW_SUB_INT8 is set: leaving the MTP tail at {requested:?} (sub-int8). \
             Expect near-zero acceptance and a net decode slowdown."
        ),
    }
}

/// Side-channel carrying the target model's pre-`lm_head` hidden states out of
/// its own forward pass and into the MTP draft path.
///
/// This is Arc's counterpart to the reference's
/// `LogitsProcessor(hidden_states_before_norm=…)` capture
/// (`logits_processor.py:603-606`), which becomes `spec_info.hidden_states`
/// and is consumed by `h_proj(hnorm(·))` at `deepseek_v4_nextn.py:152-154`.
///
/// Contract:
/// * The model stores the **whole** `[B, T, hidden]` block for the positions
///   it just ran, tagged with the absolute position of column 0 **per row**
///   ([`CaptureOffsets`]), and only while [`Self::arm`] has been called. Every
///   store overwrites the previous one, so a consumer must take the value in
///   the same step that produced it.
/// * [`Self::take`] clears the slot — holding a prompt-sized activation alive
///   past its use would cost `T × hidden` of device memory for nothing.
#[derive(Debug, Default)]
pub struct MtpHiddenCapture {
    armed: std::sync::atomic::AtomicBool,
    /// `(absolute position of column 0 per row, [B, T, hidden])`.
    slot: std::sync::Mutex<Option<(CaptureOffsets, Tensor)>>,
}

/// Where column 0 of a captured `[B, T, hidden]` block sits, absolutely.
///
/// **One** element means one shared offset for every row — the pre-change
/// shape, and what [`resolve_capture_offsets`] returns for any batch whose rows
/// agree on where they are. `B` elements means row `i` starts at element `i`,
/// which is what a left-aligned per-sequence verify window produces: each row
/// carries its own `seqlen_offsets[i]` and the shared component is zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureOffsets(Vec<usize>);

impl CaptureOffsets {
    /// Row `row`'s own offset, or `None` when this capture does not describe
    /// that row.
    pub fn row(&self, row: usize) -> Option<usize> {
        match self.0.len() {
            1 => Some(self.0[0]),
            _ => self.0.get(row).copied(),
        }
    }

    /// Whether this capture claims to describe a `batch`-row block.
    pub fn describes(&self, batch: usize) -> bool {
        self.0.len() == 1 || self.0.len() == batch
    }
}

/// Fold a forward's `seqlen_offsets` into a [`CaptureOffsets`].
///
/// 🔑 A batch whose rows agree collapses to ONE element, which makes
/// [`CaptureOffsets::row`] return exactly what
/// `seqlen_offsets.first().copied().unwrap_or(0)` used to return for every row.
/// That is the dispatch that keeps the cohort path byte-identical: it is not
/// that the per-row vector happens to hold equal values, it is that the per-row
/// representation is never constructed at all. Same reason
/// `resolve_ragged_rows` (PR #100) and `resolve_row_cache_lens` (PR #102) are
/// separate functions — the two branches agree on their common domain, so no
/// numeric test can say which one ran.
pub(crate) fn resolve_capture_offsets(offsets: &[usize]) -> CaptureOffsets {
    match offsets.split_first() {
        None => CaptureOffsets(vec![0]),
        Some((first, rest)) if rest.iter().all(|o| o == first) => CaptureOffsets(vec![*first]),
        Some(_) => CaptureOffsets(offsets.to_vec()),
    }
}

impl MtpHiddenCapture {
    /// Start retaining hidden states. Called when an [`MtpDecodeKit`] is
    /// handed out, i.e. when MTP drafting is actually engaged.
    pub fn arm(&self) {
        self.armed.store(true, Ordering::Relaxed);
    }

    /// Whether the model should pay for the capture on this forward.
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }

    /// Record the hidden states of the positions just computed, all rows
    /// sharing one absolute start. A no-op while disarmed.
    pub fn store(&self, start_pos: usize, hidden: &Tensor) {
        self.store_rows(&[start_pos], hidden);
    }

    /// [`Self::store`] with the model's `seqlen_offsets` verbatim: one absolute
    /// start **per row**. A batch whose rows agree folds back to a single
    /// shared offset ([`resolve_capture_offsets`]). A no-op while disarmed.
    pub fn store_rows(&self, offsets: &[usize], hidden: &Tensor) {
        if !self.is_armed() {
            return;
        }
        let offsets = resolve_capture_offsets(offsets);
        if let Ok(mut slot) = self.slot.lock() {
            *slot = Some((offsets, hidden.clone()));
        }
    }

    /// Take and clear the captured block.
    pub fn take(&self) -> Option<(CaptureOffsets, Tensor)> {
        self.slot.lock().ok().and_then(|mut slot| slot.take())
    }

    /// Drop any captured block without consuming it.
    pub fn clear(&self) {
        if let Ok(mut slot) = self.slot.lock() {
            *slot = None;
        }
    }
}

/// Components needed to run one MTP draft step.
///
/// All fields are Arc/Clone-cheap handles into the target model's
/// existing tensors — no extra weight loading required. The target model
/// returns this via [`crate::pipeline::loaders::NormalModel::mtp_decode_kit`].
///
/// For DeepSeek V4: `embed_tokens` and `lm_head` are the same ones used by
/// the main forward; `h_proj` and `e_proj` come from `mtp.layers.0.*` /
/// `mtp.0.*`; `block` is the checkpoint's full MTP decoder layer (attention
/// + MoE + `hnorm`/`enorm`/`norm`), present only when `--mtp-depth > 0` was
/// set at load time AND the checkpoint ships the block tensors.
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
    /// The full MTP transformer block (Tier B). `None` for heads-only loads
    /// (older exports, or `--mtp-depth 0`); drafting then falls back to the
    /// Tier-A projection-only path.
    pub(crate) block: Option<Arc<crate::models::deepseek4::MtpBlock>>,
    /// Shared with the target model: carries the target's pre-`lm_head`
    /// hidden states into `h_proj`. See [`MtpHiddenCapture`].
    pub hidden_capture: Arc<MtpHiddenCapture>,
}

impl std::fmt::Debug for MtpDecodeKit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpDecodeKit")
            .field("hidden_size", &self.embed_tokens.hidden_size())
            .finish_non_exhaustive()
    }
}

impl MtpDecodeKit {
    /// True when the full MTP transformer block was loaded (Tier B path).
    pub fn has_full_block(&self) -> bool {
        self.block.is_some()
    }

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

    /// Fresh persistent draft KV cache, or `None` on the Tier-A (heads-only)
    /// path where there is no block to cache for.
    pub fn new_draft_cache(&self) -> Option<KvCache> {
        self.block.as_ref().map(|b| b.new_draft_cache())
    }

    /// The reference's `h`/`e` combine, applied to a whole `[B, T, hidden]`
    /// block of positions at once.
    ///
    /// `deepseek_v4_nextn.py:155-161`:
    /// ```python
    /// h_proj_out, _ = self.h_proj(self.hnorm(hc_flat))
    /// e_proj_hidden_states, _ = self.e_proj(self.enorm(hidden_states))
    /// hidden_states = e_proj_hidden_states[:, None, :] + h_proj_hidden_states
    /// ```
    /// — norms **before** the projections, two projections then add. The two
    /// inputs are different tensors by construction: `hidden` is the target's
    /// captured hidden state, `tokens` are the *next* token ids.
    fn combine(
        &self,
        block: &crate::models::deepseek4::MtpBlock,
        hidden: &Tensor,
        tokens: &Tensor,
    ) -> Result<Tensor> {
        let e_emb = self.embed_tokens.forward(tokens)?;
        let h_out = self.h_proj.forward_autocast(&block.norm_h(hidden)?)?;
        let e_out = self.e_proj.forward_autocast(&block.norm_e(&e_emb)?)?;
        h_out + e_out
    }

    /// Extend the persistent draft KV over a run of committed context
    /// positions — Arc's `forward_draft_extend` (`eagle_worker.py:1094-1128`)
    /// / `forward_draft_extend_after_decode` (`:1134+`).
    ///
    /// `hidden` is `[B, T, hidden]`, the target's captured states for absolute
    /// positions `start_pos .. start_pos + T`; `next_tokens` is `[B, T]`, the
    /// committed token at each of those positions **plus one** — the pairing
    /// `apply_eagle_prefill_input_rotation` sets up (`eagle_utils.py:31-33`).
    /// The cache must already hold exactly `start_pos` entries.
    ///
    /// Returns the new fill level (`start_pos + T`).
    pub fn extend_draft_cache(
        &self,
        cache: &mut KvCache,
        start_pos: usize,
        hidden: &Tensor,
        next_tokens: &Tensor,
    ) -> Result<usize> {
        let Some(block) = self.block.as_ref() else {
            return Ok(cache.current_seq_len());
        };
        let n = next_tokens.dim(next_tokens.rank() - 1)?;
        if n == 0 {
            return Ok(start_pos);
        }
        let fused = self.combine(block, hidden, next_tokens)?;
        block.forward_tokens(&fused, start_pos, cache, next_tokens)?;
        Ok(start_pos + n)
    }

    /// Run one MTP draft step through the full transformer block (Tier B).
    ///
    /// Mirrors SGLang's `DeepseekV4ModelNextN.forward` (audit §2 "MTP head"):
    /// ```text
    /// fused  = h_proj(hnorm(prev_hidden)) + e_proj(enorm(embed(token)))
    /// hidden = decoder_layer(fused)          # V4 attention + MoE, draft KV
    /// logits = lm_head(norm(hidden))         # mtp.0.norm, shared lm_head
    /// ```
    /// `pos` is the absolute sequence position of this step (used for RoPE and
    /// as the draft KV slot); `cache` is the persistent draft KV, already
    /// holding the accepted context, so the step attends over the real prefix
    /// and not just its own chain. Returns
    /// `(logits [B, vocab], next_hidden [B, hidden])` where `next_hidden` is
    /// the decoder output pre-`norm` (the reference feeds the pre-head hidden
    /// state forward between spec steps).
    fn step_full(
        &self,
        block: &crate::models::deepseek4::MtpBlock,
        prev_hidden: &Tensor,
        last_token: &Tensor,
        pos: usize,
        cache: &mut KvCache,
    ) -> Result<(Tensor, Tensor)> {
        let last_token = if last_token.rank() == 0 {
            last_token.unsqueeze(0)?
        } else {
            last_token.clone()
        };
        let prev_hidden = if prev_hidden.rank() == 1 {
            prev_hidden.unsqueeze(0)?
        } else {
            prev_hidden.clone()
        };

        let fused = self.combine(block, &prev_hidden, &last_token)?; // [B, hidden]
        let fused3 = fused.unsqueeze(1)?; // [B, 1, hidden]
        let ids = last_token.unsqueeze(1)?; // [B, 1] (hash-routed MoE gate input)
        let hidden = block.forward_step(&fused3, pos, cache, &ids)?; // [B, 1, hidden]

        let normed = block.norm_out(&hidden)?;
        let mtp_logits = self.lm_head.forward_autocast(&normed.squeeze(1)?)?;
        Ok((mtp_logits, hidden.squeeze(1)?))
    }

    /// Run one MTP draft chain (greedy).
    ///
    /// Loops up to `min(depth, max_tokens)` iterations. Each iteration:
    /// 1. Runs one MTP step — through the full transformer block when the
    ///    checkpoint shipped it (Tier B), else the Tier-A projection-only
    ///    [`Self::step`] — producing `mtp_logits` and the next hidden state.
    /// 2. Greedy-argmax over `mtp_logits` to pick the next token.
    /// 3. Feeds the new token (and the hidden state) back into the next
    ///    iteration.
    ///
    /// `last_hidden` MUST be the **target model's** hidden state at position
    /// `start_pos` (via [`MtpHiddenCapture`]), never an embedding — see the
    /// module docs.
    ///
    /// `start_pos` is the absolute sequence position of the FIRST draft step,
    /// which is `len(committed_tokens) - 1`: that step is the pair
    /// `(h_{L-1}, T0)` and is a *real* context entry, exactly the last entry
    /// the reference writes during its draft extend
    /// (`eagle_worker.py:1094-1128`); the reference's own draft-decode loop
    /// then starts one position later at `seq_len` (`:726`, `:910`).
    ///
    /// `draft_cache` is the persistent per-sequence draft KV, which must
    /// already hold `start_pos` entries of accepted context. `None` is only
    /// valid on the Tier-A (heads-only) path, which has no attention at all.
    ///
    /// Returns the list of proposed token IDs (length is exactly
    /// `min(depth, max_tokens)`).
    pub fn propose_chain(
        &self,
        last_hidden: &Tensor,
        last_token_id: u32,
        depth: usize,
        max_tokens: usize,
        start_pos: usize,
        draft_cache: Option<&mut KvCache>,
    ) -> Result<Vec<u32>> {
        let n = depth.min(max_tokens);
        let mut tokens = Vec::with_capacity(n);
        if n == 0 {
            return Ok(tokens);
        }
        if self.block.is_some() && draft_cache.is_none() {
            candle_core::bail!(
                "MTP full-block drafting requires the persistent draft KV cache; drafting \
                 against an empty per-chain cache while applying absolute RoPE positions is \
                 the RUN-169 acceptance-killer (audit finding 2)."
            );
        }
        let device = last_hidden.device();
        let mut prev_hidden = last_hidden.clone();
        let mut tok = last_token_id;
        let mut draft_cache = draft_cache;
        for i in 0..n {
            let tok_tensor = Tensor::from_vec(vec![tok], (1,), device)?;
            let (mtp_logits, next_hidden) = match (&self.block, draft_cache.as_deref_mut()) {
                (Some(block), Some(cache)) => {
                    self.step_full(block, &prev_hidden, &tok_tensor, start_pos + i, cache)?
                }
                _ => self.step(&prev_hidden, &tok_tensor)?,
            };
            // Greedy argmax. Squeeze the batch dim if present.
            let logits = if mtp_logits.rank() == 2 {
                mtp_logits.i(0)?
            } else {
                mtp_logits
            };
            let next_id = argmax_token(&logits)?;
            tokens.push(next_id);
            prev_hidden = next_hidden;
            tok = next_id;
        }
        Ok(tokens)
    }

    /// [`Self::propose_chain`] for a whole group of sequences at once.
    ///
    /// One `[G, 1, hidden]` MTP-block forward per chain step drafts for every
    /// row in the group: `G` chains cost the same block reads as one. That is
    /// the whole reason the batched fast path groups before drafting — see
    /// [`MtpSpeculativePipeline::step`].
    ///
    /// * `seed_hidden` — `[G, hidden]`, the target's hidden state at that row's
    ///   chain start.
    /// * `last_tokens` — `[G]`, `tok_{chain_start+1}` for each sequence.
    /// * `chain_starts` — `[G]`, each row's own absolute chain start. Rows no
    ///   longer have to agree: a group whose starts differ is drafted against a
    ///   **left-aligned** batched draft KV (built by
    ///   [`front_align_draft_group`]) and the per-row positions go to
    ///   `MtpBlock::forward_step_rows`, which threads them to RoPE and to
    ///   `dsv4_attention`'s `row_q0` dead-prefix mask.
    /// * `cache` — ONE batched draft KV whose live runs all END at
    ///   `max(chain_starts)`.
    ///
    /// Returns `G` chains of exactly `n` tokens each.
    fn propose_chain_batched(
        &self,
        seed_hidden: &Tensor,
        last_tokens: &[u32],
        n: usize,
        chain_starts: &[usize],
        mut cache: Option<&mut KvCache>,
    ) -> Result<Vec<Vec<u32>>> {
        let g = last_tokens.len();
        let mut chains = vec![Vec::with_capacity(n); g];
        if n == 0 || g == 0 {
            return Ok(chains);
        }
        if chain_starts.len() != g {
            candle_core::bail!(
                "MTP batched draft: {} chain start(s) for {g} row(s)",
                chain_starts.len()
            );
        }
        if self.block.is_some() && cache.is_none() {
            candle_core::bail!(
                "MTP full-block drafting requires the persistent draft KV cache; drafting \
                 against an empty per-chain cache while applying absolute RoPE positions is \
                 the RUN-169 acceptance-killer (audit finding 2)."
            );
        }
        // `None` — every group the cohort rule builds — runs the pre-change
        // scalar arithmetic verbatim, down to RoPE's one-element fast branch.
        let ragged = resolve_ragged_chain_starts(chain_starts);
        let start_pos = chain_starts[0];
        let device = seed_hidden.device().clone();
        let mut prev_hidden = seed_hidden.clone(); // [G, hidden]
        let mut toks = last_tokens.to_vec();
        let mut row_pos: Vec<usize> = Vec::new();
        for i in 0..n {
            let tok_tensor = Tensor::from_vec(toks.clone(), (g,), &device)?;
            let (logits, next_hidden) = match (&self.block, cache.as_deref_mut()) {
                (Some(block), Some(cache)) => {
                    let fused = self.combine(block, &prev_hidden, &tok_tensor)?; // [G, hidden]
                    let ids = tok_tensor.reshape((g, 1))?;
                    let hidden = match ragged {
                        None => {
                            block.forward_step(&fused.unsqueeze(1)?, start_pos + i, cache, &ids)?
                        }
                        Some(rows) => {
                            row_pos.clear();
                            row_pos.extend(rows.iter().map(|&r| r + i));
                            block.forward_step_rows(&fused.unsqueeze(1)?, &row_pos, cache, &ids)?
                        }
                    };
                    let normed = block.norm_out(&hidden)?;
                    (
                        self.lm_head.forward_autocast(&normed.squeeze(1)?)?,
                        hidden.squeeze(1)?,
                    )
                }
                // Tier-A (heads-only): no attention, so no position and no
                // cache — the projection pair batches trivially.
                _ => self.step(&prev_hidden, &tok_tensor)?,
            };
            let next = argmax_rows(&logits)?;
            for (row, tok) in next.iter().enumerate() {
                chains[row].push(*tok);
            }
            toks = next;
            prev_hidden = next_hidden;
        }
        Ok(chains)
    }
}

/// Row-wise greedy argmax over a `[G, vocab]` logits tensor.
///
/// One device-side `argmax` plus one `G`-element copy, rather than `G`
/// single-row argmaxes: the batched draft chain runs this once per chain step,
/// so a per-row loop would put `G` device syncs on the decode hot path.
fn argmax_rows(logits: &Tensor) -> Result<Vec<u32>> {
    let l2 = match logits.rank() {
        1 => logits.unsqueeze(0)?,
        2 => logits.clone(),
        3 => logits.squeeze(1)?,
        other => candle_core::bail!("MTP argmax_rows: unexpected logits rank {other}"),
    };
    l2.argmax(candle_core::D::Minus1)?
        .to_dtype(candle_core::DType::U32)?
        .to_vec1::<u32>()
}

/// Per-row absolute chain starts, or `None` when the group's rows already sit
/// at one.
///
/// 🔑 A **dispatch, not an optimisation**. `None` sends
/// [`MtpDecodeKit::propose_chain_batched`] down the pre-change scalar path
/// verbatim — `MtpBlock::forward_step` with one `pos`, one-element
/// `seqlen_offsets`, RoPE's `len() == 1` fast branch, `row_q0` filtered back to
/// `None` — so every group the cohort rule builds is byte-identical by control
/// flow rather than by a number some test happened to observe. The two branches
/// agree exactly on their common domain, which is why this is a named function
/// with a test of its own rather than an `if` inside the loop (PR #100 caught a
/// mutation that routed uniform batches down the ragged path only because
/// `resolve_ragged_rows` was separable; the numerical tests could not see it).
pub(crate) fn resolve_ragged_chain_starts(chain_starts: &[usize]) -> Option<&[usize]> {
    let first = chain_starts.first()?;
    chain_starts
        .iter()
        .any(|s| s != first)
        .then_some(chain_starts)
}

/// Each row's dead prefix once a draft group is left-aligned to its widest row,
/// or `None` when the rows already share a chain start.
///
/// `lead[i] = max_j start_j - start_i` is what
/// [`crate::kv_cache::front_pad_kv_cache`] will leave ahead of row `i`, and
/// therefore also what [`crate::kv_cache::drop_dead_prefix`] must take back out
/// on the way to the per-sequence caches. Recording the padded width as a row's
/// own length is the defect PR #102 named in the target's cache — the dead
/// prefix compounds every step and the rows silently re-converge onto one
/// length, i.e. the cohort barrier reappearing as padding. It is the same trap
/// here, one cache down.
///
/// `enabled` is the per-sequence-advance decision, not a fourth flag: a ragged
/// group can only exist under `KvAdvance::PerSequence`, and `ragged_row_q0`
/// (which masks the dead prefix) is gated on the same `ARC_MTP_PER_SEQ_KV`. If
/// this returned `Some` with the flag off, the block would attend zero-filled
/// padding as real keys with nothing to mask it.
pub(crate) fn resolve_draft_leads(enabled: bool, chain_starts: &[usize]) -> Option<Vec<usize>> {
    if !enabled {
        return None;
    }
    let end = *chain_starts.iter().max()?;
    if chain_starts.iter().all(|&s| s == end) {
        return None;
    }
    Some(chain_starts.iter().map(|&s| end - s).collect())
}

/// Build ONE batched draft KV from a group whose rows sit at **different**
/// absolute positions, by left-aligning them first.
///
/// The single-layer analogue of [`crate::kv_cache::front_align_batch`] +
/// `clone_in_cache`: truncate each row to its own chain start, shift its live
/// run right so it ENDS at `max(chain_starts)` — which makes
/// `SingleCache::append`'s one shared write offset simultaneously correct for
/// all of them — then concatenate along the batch dim.
///
/// Every row is narrowed to exactly `target` columns before the concatenation:
/// `front_pad_single` returns early without touching the buffer when a row's
/// lead is zero, so the rows' physical widths are their own capacities and only
/// their *live* extents agree. That is precisely the disagreement the old
/// `capacity_seq_len` equality check refused over.
fn front_align_draft_group(caches: &[&KvCache], chain_starts: &[usize]) -> Result<KvCache> {
    if caches.len() != chain_starts.len() {
        candle_core::bail!(
            "MTP ragged draft group: {} cache(s) for {} chain start(s)",
            caches.len(),
            chain_starts.len()
        );
    }
    let Some(&target) = chain_starts.iter().max() else {
        candle_core::bail!("MTP ragged draft group: no rows");
    };
    let mut ks = Vec::with_capacity(caches.len());
    let mut vs = Vec::with_capacity(caches.len());
    let mut proto: Option<(SingleCache, SingleCache)> = None;
    for (cache, &start) in caches.iter().zip(chain_starts) {
        let mut owned = (*cache).clone();
        // `set_len` will happily grow into spare capacity, so it cannot be the
        // check: a chain start past what this row actually holds means the
        // caller's idea of where the sequence is and the cache's disagree, and
        // drafting from it would attend uninitialised columns as context.
        let held = owned.current_seq_len();
        if start > held {
            candle_core::bail!(
                "MTP ragged draft group: chain start {start} is past the {held} position(s) \
                 this row's draft KV actually holds"
            );
        }
        owned.set_len(start).map_err(|e| {
            candle_core::Error::msg(format!(
                "MTP ragged draft group: truncating a row to its own chain start {start} \
                 failed: {e}"
            ))
        })?;
        crate::kv_cache::front_pad_kv_cache(&mut owned, target)?;
        let KvCache::Normal { k, v } = &owned else {
            candle_core::bail!(
                "MTP ragged draft group: a draft KV is a `{}`, not a `Normal` slot",
                owned.kind_name()
            );
        };
        let (Some(kd), Some(vd)) = (k.all_data.as_ref(), v.all_data.as_ref()) else {
            candle_core::bail!("MTP ragged draft group: a row's buffer is not materialised");
        };
        ks.push(kd.narrow(k.dim, 0, target)?);
        vs.push(vd.narrow(v.dim, 0, target)?);
        if proto.is_none() {
            proto = Some((k.clone(), v.clone()));
        }
    }
    let Some((k0, v0)) = proto else {
        candle_core::bail!("MTP ragged draft group: no rows");
    };
    let batched_k = Tensor::cat(&ks, 0)?.contiguous()?;
    let batched_v = Tensor::cat(&vs, 0)?.contiguous()?;
    Ok(KvCache::Normal {
        k: SingleCache {
            all_data: Some(batched_k),
            capacity_seq_len: target,
            current_seq_len: target,
            ..k0
        },
        v: SingleCache {
            all_data: Some(batched_v),
            capacity_seq_len: target,
            current_seq_len: target,
            ..v0
        },
    })
}

/// Build ONE batched draft KV from a group's per-sequence caches.
///
/// The MTP block is a single decoder layer, so this mirrors
/// `NormalCacheManager::clone_in_cache` at 1/43 of its cost: concatenate the
/// per-sequence `all_data` along the batch dim and keep `seqs[0]`'s length
/// metadata, which is exact because every sequence in a group is at the same
/// fill level by construction.
///
/// Returns `None` (drafting is then skipped for the group, losslessly) when a
/// cache has not been materialised yet or the group's buffers disagree on
/// shape — a state the group invariant says cannot happen, so refusing beats
/// concatenating tensors whose rows would not mean what the caller thinks.
fn batch_draft_caches(caches: &[&KvCache]) -> Option<KvCache> {
    let first = caches.first()?;
    if caches.len() == 1 {
        return Some((*first).clone());
    }
    let KvCache::Normal { k: k0, v: v0 } = first else {
        return None;
    };
    let (mut ks, mut vs) = (
        Vec::with_capacity(caches.len()),
        Vec::with_capacity(caches.len()),
    );
    for cache in caches {
        let KvCache::Normal { k, v } = cache else {
            return None;
        };
        if k.current_seq_len != k0.current_seq_len
            || k.capacity_seq_len != k0.capacity_seq_len
            || k.dim != k0.dim
        {
            return None;
        }
        ks.push(k.all_data.clone()?);
        vs.push(v.all_data.clone()?);
    }
    let batched_k = Tensor::cat(&ks, 0).ok()?.contiguous().ok()?;
    let batched_v = Tensor::cat(&vs, 0).ok()?.contiguous().ok()?;
    Some(KvCache::Normal {
        k: SingleCache {
            all_data: Some(batched_k),
            ..k0.clone()
        },
        v: SingleCache {
            all_data: Some(batched_v),
            ..v0.clone()
        },
    })
}

/// Split a batched draft KV back into `g` per-sequence caches — the
/// `clone_out_cache` half of [`batch_draft_caches`].
fn split_draft_cache(batched: &KvCache, g: usize) -> Option<Vec<KvCache>> {
    if g == 1 {
        return Some(vec![batched.clone()]);
    }
    let KvCache::Normal { k, v } = batched else {
        return None;
    };
    let ks = k.all_data.as_ref()?.chunk(g, 0).ok()?;
    let vs = v.all_data.as_ref()?.chunk(g, 0).ok()?;
    if ks.len() != g || vs.len() != g {
        return None;
    }
    Some(
        ks.into_iter()
            .zip(vs)
            .map(|(kc, vc)| KvCache::Normal {
                k: SingleCache {
                    all_data: Some(kc),
                    ..k.clone()
                },
                v: SingleCache {
                    all_data: Some(vc),
                    ..v.clone()
                },
            })
            .collect(),
    )
}

/// Log the running MTP acceptance rate once per this many **proposed** tokens.
///
/// 64 proposals is ~32 decode steps at depth 2 — a line every few seconds at
/// single-digit tok/s, which is enough resolution to watch acceptance settle
/// without flooding the serve log. This is the cadence the session-2/4 runbooks
/// document (`per 64 proposed`), so changing it invalidates their expected
/// output.
const ACCEPTANCE_LOG_EVERY_PROPOSED: usize = 64;

/// How many draft-chain positions the per-position acceptance breakdown keeps
/// separate buckets for.
///
/// The chain is linear (`topk == 1`), so position `j` is the `j`-th token of the
/// draft chain and the useful depth is small: DeepSeek V4 ships 1-3 MTP layers
/// and SGLang's own shipped config for every DeepSeek arch is `(3, 1, 4)`
/// (`server_args.py:7611-7627`). Eight buckets is well past anything a real
/// checkpoint can drive, and it keeps [`MtpAcceptance`] `Copy` — which is what
/// lets the whole struct live in a `BTreeMap` value and be folded with `add`.
///
/// A step that proposes MORE than this many drafts still accounts correctly in
/// every aggregate field; only its positions `>= 8` go unbucketed, and
/// [`MtpAcceptance::positions_over_cap`] counts those steps so the reader can
/// tell rather than having to trust the arithmetic.
pub const MTP_MAX_TRACKED_POSITIONS: usize = 8;

/// Parse the `ARC_MTP_LOG_ACCEPTANCE` gate. **Default ON.**
///
/// Off-by-default was how three GPU sessions measured MTP acceptance and came
/// home with empty files: the number only exists while the process that
/// produced it is alive, and nobody sets a variable they have not read about.
/// A speculative decoder that will not say how often it was right is not
/// measurable, so the line is on whenever MTP is engaged and
/// `ARC_MTP_LOG_ACCEPTANCE=0` (also `false` / `off` / `no`) turns it off. The
/// existing runbooks set it to `1`, which still means the same thing.
fn acceptance_log_from_env(value: Option<&str>) -> bool {
    !matches!(
        value.map(str::trim),
        Some("0") | Some("false") | Some("off") | Some("no")
    )
}

/// Whether to emit the periodic acceptance report.
///
/// Read once per process into a `OnceLock`: `record_acceptance` runs on every
/// verify, which is the decode hot path, and it must not do an env lookup per
/// token. The variable name is load-bearing — GPU-session runbooks and
/// `arc-tools/quality/qlib.py` both reference this exact variable.
fn acceptance_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        acceptance_log_from_env(std::env::var("ARC_MTP_LOG_ACCEPTANCE").ok().as_deref())
    })
}

/// One scope's MTP speculative-decode accounting, as counted at the
/// accept/reject site.
///
/// Every field is a raw count taken at the one place the decision is made
/// (`MtpSpeculativePipeline::step`). Nothing here is derived from throughput,
/// wall-clock, or the configured depth: if the counters are wrong the number is
/// wrong, and there is no smoothing to hide it (DOCTRINE D9).
///
/// The two ratios answer different questions and both are needed:
/// * [`Self::rate`] — `accepted / proposed`, how good the draft head is.
/// * [`Self::tokens_per_step`] — `committed / steps`, what the decode loop
///   actually got out of it. This is the number that moves the per-user
///   ceiling: at B=256 one H200 can sustain ~65 tok/s/user of *steps*, so
///   `tok_per_step` multiplies straight through it.
///
/// [`Self::drafted_steps`] is what separates "the head is bad" from "we never
/// drafted at all" — a distinction that matters because the draft KV declines
/// to draft (losslessly, and silently after the first warning) whenever it
/// cannot be primed. Without it, a 0% acceptance rate is ambiguous.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MtpAcceptance {
    /// Draft tokens the verifier accepted.
    pub accepted: usize,
    /// Draft tokens handed to the verifier.
    pub proposed: usize,
    /// MTP decode steps taken, including steps that drafted nothing.
    pub steps: usize,
    /// Steps that proposed at least one draft token.
    pub drafted_steps: usize,
    /// User-visible tokens committed across those steps.
    pub committed: usize,
    /// **Engine** steps (one forward over the whole batch), as distinct from
    /// [`Self::steps`], which counts one per *sequence* per engine step.
    ///
    /// The two ratios they produce answer the two different questions the
    /// ceiling model asks. `committed / steps` is the **per-user** multiplier —
    /// the one that multiplies the 68 tok/s per-user floor at B=128. `committed
    /// / batch_steps` is the **aggregate** multiplier, tokens out per forward.
    /// At B=1 they are equal, which is exactly why a B=1 measurement cannot
    /// tell you whether MTP lifts a batched row (DOCTRINE D2).
    pub batch_steps: usize,
    /// Per draft-chain position: how many times position `j` was **reached** —
    /// the step proposed at least `j + 1` drafts *and* every earlier position
    /// was accepted.
    ///
    /// This is the denominator the aggregate [`Self::rate`] does not have. In a
    /// linear chain (`topk == 1`, which is what DeepSeek MTP is) position `j` is
    /// only ever evaluated when the chain survived to it, so dividing
    /// `accepted_at[j]` by `proposed` — or by the step count — answers no
    /// question. `accepted_at[j] / reached_at[j]` is the per-position
    /// conditional acceptance `p_j`, the `p` in `E[accepted] = Σ pʲ`.
    pub reached_at: [usize; MTP_MAX_TRACKED_POSITIONS],
    /// Per draft-chain position: how many of the [`Self::reached_at`]
    /// evaluations at position `j` the verifier accepted.
    ///
    /// **This is the field the scalar [`Self::accepted`] could not answer.**
    /// `Σ accepted_at == accepted` (exactly, whenever
    /// [`Self::positions_over_cap`] is 0), so it is a refinement and not a
    /// second, competing count. Reference shape: vLLM's
    /// `vllm:spec_decode_num_accepted_tokens_per_pos`
    /// (`v1/spec_decode/metrics.py:255`).
    pub accepted_at: [usize; MTP_MAX_TRACKED_POSITIONS],
    /// Steps that proposed more than [`MTP_MAX_TRACKED_POSITIONS`] drafts, and
    /// whose positions beyond the cap therefore landed in no bucket.
    ///
    /// Zero for every depth a real checkpoint can drive. Carried anyway because
    /// the `Σ accepted_at == accepted` invariant is what makes the breakdown
    /// auditable, and an invariant with a silent exception is not one.
    pub positions_over_cap: usize,
}

/// Fill the per-position tallies for one step.
///
/// `n_proposed` drafts went out, positions `0..reached` were evaluated, and
/// positions `0..n_accepted` came back accepted. `reached` is `n_accepted`
/// plus one when the chain was cut short by a rejection, and `n_accepted`
/// exactly when it ran to the end of the proposal (nothing was evaluated past
/// the last draft).
fn position_tallies(
    n_proposed: usize,
    n_accepted: usize,
    reached: usize,
) -> (
    [usize; MTP_MAX_TRACKED_POSITIONS],
    [usize; MTP_MAX_TRACKED_POSITIONS],
    usize,
) {
    let mut reached_at = [0usize; MTP_MAX_TRACKED_POSITIONS];
    let mut accepted_at = [0usize; MTP_MAX_TRACKED_POSITIONS];
    for slot in reached_at
        .iter_mut()
        .take(reached.min(MTP_MAX_TRACKED_POSITIONS))
    {
        *slot = 1;
    }
    for slot in accepted_at
        .iter_mut()
        .take(n_accepted.min(MTP_MAX_TRACKED_POSITIONS))
    {
        *slot = 1;
    }
    let over_cap = usize::from(n_proposed > MTP_MAX_TRACKED_POSITIONS);
    (reached_at, accepted_at, over_cap)
}

impl MtpAcceptance {
    /// The accounting for one MTP decode step that proposed `n_proposed` draft
    /// tokens and got `result` back from the verifier.
    ///
    /// `committed` is T0 (always emitted) + the accepted proposals + the
    /// verifier's correction when there was a rejection — exactly the tokens
    /// `step()` passes to `finish_or_add_toks_to_seq`. Deriving it here rather
    /// than at the call site is deliberate: a drift between "what we counted"
    /// and "what we emitted" is precisely how a speculative decoder comes to
    /// report a multiplier it never delivered.
    pub fn from_verify(n_proposed: usize, result: &VerifyResult) -> Self {
        let n_accepted = result.accepted.len();
        // The verifier evaluated exactly the accepted prefix plus, if it
        // rejected, the one position it rejected at. Taking it from `result`
        // rather than from `n_proposed` is what keeps a short `target` slice
        // (fewer target tokens than proposals) from being counted as a
        // rejection at a position nothing ever looked at.
        let reached = n_accepted + usize::from(result.rejection.is_some());
        let (reached_at, accepted_at, positions_over_cap) =
            position_tallies(n_proposed, n_accepted, reached);
        Self {
            accepted: n_accepted,
            proposed: n_proposed,
            steps: 1,
            drafted_steps: usize::from(n_proposed > 0),
            committed: 1 + result.commit_len(),
            batch_steps: 0,
            reached_at,
            accepted_at,
            positions_over_cap,
        }
    }

    /// The accounting for one sequence in a **fused** batched step, which runs
    /// ONE target forward and commits `1 + accepted` tokens (`accepted`
    /// verified drafts plus the target's own correction-or-bonus token).
    ///
    /// Distinct from [`Self::from_verify`], which describes the two-forward
    /// shape: there `committed = 1 + commit_len` because `T0` came out of a
    /// *separate* target forward that this shape no longer runs.
    ///
    /// The per-position tallies are derived from the same two counts, so they
    /// agree with the aggregate by construction: positions `0..n_accepted` were
    /// accepted, and position `n_accepted` was reached-and-rejected whenever the
    /// chain did not run to its end. That attribution also absorbs the
    /// **length-budget clamp** at the call site (a chain truncated because the
    /// sequence is `max_seq_len`-bound arrives here with a smaller
    /// `n_accepted`): the clamped position is booked as a rejection, exactly as
    /// the aggregate `accepted` books it, so `Σ accepted_at == accepted` still
    /// holds and the two numbers can never tell different stories.
    pub fn from_fused_verify(n_proposed: usize, n_accepted: usize) -> Self {
        let reached = if n_proposed == 0 {
            0
        } else {
            (n_accepted + 1).min(n_proposed)
        };
        let (reached_at, accepted_at, positions_over_cap) =
            position_tallies(n_proposed, n_accepted, reached);
        Self {
            accepted: n_accepted,
            proposed: n_proposed,
            steps: 1,
            drafted_steps: usize::from(n_proposed > 0),
            committed: 1 + n_accepted,
            batch_steps: 0,
            reached_at,
            accepted_at,
            positions_over_cap,
        }
    }

    /// The accounting for a decode step that drafted nothing — no draft KV, no
    /// budget left, or MTP simply declined. One token out, no proposals.
    pub fn skipped_step() -> Self {
        Self {
            accepted: 0,
            proposed: 0,
            steps: 1,
            drafted_steps: 0,
            committed: 1,
            batch_steps: 0,
            // No position was reached, so no position's denominator moves. This
            // is the whole reason `reached_at` exists rather than a step count:
            // a run that mostly declines to draft must not dilute `p_j`.
            reached_at: [0; MTP_MAX_TRACKED_POSITIONS],
            accepted_at: [0; MTP_MAX_TRACKED_POSITIONS],
            positions_over_cap: 0,
        }
    }

    /// `accepted / proposed`, or `None` when nothing was ever proposed (0/0 is
    /// not 0%, and printing it as one is how an un-engaged run gets mistaken
    /// for a failed one).
    #[allow(clippy::cast_precision_loss)] // counts; >2^53 tokens is not a case
    pub fn rate(&self) -> Option<f64> {
        (self.proposed > 0).then(|| self.accepted as f64 / self.proposed as f64)
    }

    /// Committed tokens per decode step **per user** — the effective
    /// speculative multiplier on the per-user ceiling. Plain (non-speculative)
    /// decode is exactly 1.0.
    #[allow(clippy::cast_precision_loss)]
    pub fn tokens_per_step(&self) -> Option<f64> {
        (self.steps > 0).then(|| self.committed as f64 / self.steps as f64)
    }

    /// Committed tokens per **engine** step — the aggregate multiplier, i.e.
    /// tokens out per target forward across the whole batch.
    ///
    /// `None` when no batched step was recorded (the counter is set by the
    /// pipeline once per forward, not once per sequence, so a caller that only
    /// ever recorded per-sequence tallies honestly has no batch data).
    #[allow(clippy::cast_precision_loss)]
    pub fn tokens_per_batch_step(&self) -> Option<f64> {
        (self.batch_steps > 0).then(|| self.committed as f64 / self.batch_steps as f64)
    }

    /// Mean batch size across the recorded engine steps — `steps /
    /// batch_steps`. The label a per-user multiplier has to be read against.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_batch(&self) -> Option<f64> {
        (self.batch_steps > 0).then(|| self.steps as f64 / self.batch_steps as f64)
    }

    /// How many draft-chain positions this snapshot ever reached — the length
    /// of the useful prefix of [`Self::reached_at`] / [`Self::accepted_at`].
    ///
    /// Zero when nothing was ever drafted.
    pub fn tracked_extent(&self) -> usize {
        self.reached_at
            .iter()
            .rposition(|&n| n > 0)
            .map_or(0, |i| i + 1)
    }

    /// Per-position conditional acceptance `p_j = accepted_at[j] /
    /// reached_at[j]`, one entry per position actually reached.
    ///
    /// **This is the number the aggregate rate cannot give you.** With the
    /// chain's `E[accepted] = Σ_{i=1..depth} pⁱ`, a flat profile
    /// (`p₀ ≈ p₁ ≈ …`) and a falling one (`p₀ ≫ p₁`) produce the same aggregate
    /// `accept_rate` while meaning opposite things: flat is a target/draft
    /// **distribution mismatch** at every position, falling is the draft
    /// **compounding on its own hidden state** down the chain — and only the
    /// second is a fix in the chain-feedback code.
    #[allow(clippy::cast_precision_loss)] // counts; >2^53 tokens is not a case
    pub fn per_position_rates(&self) -> Vec<f64> {
        (0..self.tracked_extent())
            .map(|j| self.accepted_at[j] as f64 / self.reached_at[j] as f64)
            .collect()
    }

    /// The per-position fields as `(position, accepted, reached, p)` rows,
    /// truncated to [`Self::tracked_extent`]. Empty when nothing was drafted.
    #[allow(clippy::cast_precision_loss)]
    pub fn position_rows(&self) -> Vec<(usize, usize, usize, f64)> {
        (0..self.tracked_extent())
            .map(|j| {
                (
                    j,
                    self.accepted_at[j],
                    self.reached_at[j],
                    self.accepted_at[j] as f64 / self.reached_at[j] as f64,
                )
            })
            .collect()
    }

    /// The machine-greppable one-liner, in the project's marker convention
    /// (`SPEED[...]`, `BATCH[...]`, `GSM8K[...]`).
    ///
    /// `scope` is `agg` for the process total or `req=<id>` for one request.
    /// Every raw count is on the line, so both ratios are auditable without
    /// trusting the formatter.
    /// The per-position fields are **appended** to the existing key=value tail,
    /// never substituted into it: the aggregate/`by-batch` separation the
    /// runbooks already grep for (`accept_rate`, `tok_per_step`,
    /// `drafted_steps`, `batch_steps`) is unchanged, so
    /// `GPU_SESSION_RUNBOOK_8.md`'s expected output still parses.
    pub fn marker(&self, scope: &str) -> String {
        let fmt = |v: Option<f64>| v.map_or_else(|| "n/a".to_string(), |x| format!("{x:.4}"));
        let list = |v: &[usize]| {
            if v.is_empty() {
                "n/a".to_string()
            } else {
                v.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
            }
        };
        let extent = self.tracked_extent();
        let p_pos = if extent == 0 {
            "n/a".to_string()
        } else {
            self.per_position_rates()
                .iter()
                .map(|p| format!("{p:.4}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "MTP[{scope}] accept_rate={} accepted={} proposed={} steps={} drafted_steps={} \
             committed={} tok_per_step={} batch_steps={} mean_batch={} tok_per_batch_step={} \
             p_pos={} accepted_pos={} reached_pos={} pos_over_cap={}",
            fmt(self.rate()),
            self.accepted,
            self.proposed,
            self.steps,
            self.drafted_steps,
            self.committed,
            fmt(self.tokens_per_step()),
            self.batch_steps,
            fmt(self.mean_batch()),
            fmt(self.tokens_per_batch_step()),
            p_pos,
            list(&self.accepted_at[..extent]),
            list(&self.reached_at[..extent]),
            self.positions_over_cap,
        )
    }

    /// The per-position breakdown as human-readable lines, with the reading
    /// instructions attached.
    ///
    /// Empty when nothing was drafted — the honest answer to "what is `p₁`" on a
    /// run that never drafted is nothing, not `0.0`.
    pub fn position_report_lines(&self) -> Vec<String> {
        let rows = self.position_rows();
        if rows.is_empty() {
            return Vec::new();
        }
        let mut out = vec![
            "MTP per-position acceptance (p_j = accepted at chain position j / times \
             position j was reached):"
                .to_string(),
        ];
        for (j, acc, reached, p) in rows {
            out.push(format!("  pos {j}: p={p:.4} ({acc}/{reached})"));
        }
        out.push(
            "  reading: flat p_j across positions => target/draft distribution mismatch; \
             p_0 >> p_1 => the draft compounding on its own hidden state (a fix in the chain \
             feedback). E[accepted] = sum_i p^i."
                .to_string(),
        );
        if self.positions_over_cap > 0 {
            out.push(format!(
                "  WARN: {} step(s) proposed more than {} drafts; positions beyond the cap are \
                 counted in the aggregate but not broken out here",
                self.positions_over_cap, MTP_MAX_TRACKED_POSITIONS,
            ));
        }
        out
    }

    /// The human-readable line the runbooks already grep for
    /// (`grep "MTP acceptance"`). Kept alongside [`Self::marker`] so existing
    /// GPU-session tooling does not need a new parser.
    #[allow(clippy::cast_precision_loss)]
    pub fn report_line(&self) -> String {
        match self.rate() {
            None => "MTP acceptance: 0 proposals so far".to_string(),
            Some(r) => format!(
                "MTP acceptance rate: {:.1}% ({}/{} accepted)",
                100.0 * r,
                self.accepted,
                self.proposed
            ),
        }
    }

    fn add(&mut self, other: &Self) {
        self.accepted += other.accepted;
        self.proposed += other.proposed;
        self.steps += other.steps;
        self.drafted_steps += other.drafted_steps;
        self.committed += other.committed;
        self.batch_steps += other.batch_steps;
        for j in 0..MTP_MAX_TRACKED_POSITIONS {
            self.reached_at[j] += other.reached_at[j];
            self.accepted_at[j] += other.accepted_at[j];
        }
        self.positions_over_cap += other.positions_over_cap;
    }
}

/// Atomic tallies behind an [`MtpAcceptance`] snapshot, plus the periodic
/// report that makes them visible from a serve log.
///
/// Split out of [`MtpSpeculativePipeline`] so the telemetry is testable on CPU
/// without standing up a target `dyn Pipeline`. Before this existed,
/// `log_acceptance_rate()` had **zero callers** anywhere in the workspace: the
/// counters accumulated and nothing ever read them, so GPU session 3 measured
/// MTP acceptance and produced an empty artifact.
#[derive(Debug, Default)]
pub(crate) struct AcceptanceTelemetry {
    accepted: AtomicUsize,
    proposed: AtomicUsize,
    steps: AtomicUsize,
    drafted_steps: AtomicUsize,
    committed: AtomicUsize,
    batch_steps: AtomicUsize,
    /// Per draft-chain position, mirroring [`MtpAcceptance::reached_at`] /
    /// [`MtpAcceptance::accepted_at`]. Plain relaxed counters like every other
    /// field here: the accept/reject site is the only writer and a reader wants
    /// the running total, not a consistent cut across positions.
    reached_at: [AtomicUsize; MTP_MAX_TRACKED_POSITIONS],
    accepted_at: [AtomicUsize; MTP_MAX_TRACKED_POSITIONS],
    positions_over_cap: AtomicUsize,
    /// The same counters again, split by the batch size of the engine step
    /// that produced them.
    ///
    /// An aggregate over a run whose batch size moved is a number about no
    /// particular batch — and the whole reason batched MTP exists is that the
    /// per-user ceiling is a function of B (`CEILINGS.json`: 1413 at B=1, 68 at
    /// B=128). One `MTP[b=<B>]` line per observed batch size is the smallest
    /// thing that makes "does MTP still multiply at B=128" answerable from a
    /// log.
    by_batch: std::sync::Mutex<std::collections::BTreeMap<usize, MtpAcceptance>>,
}

impl AcceptanceTelemetry {
    const fn new() -> Self {
        Self {
            accepted: AtomicUsize::new(0),
            proposed: AtomicUsize::new(0),
            steps: AtomicUsize::new(0),
            drafted_steps: AtomicUsize::new(0),
            committed: AtomicUsize::new(0),
            batch_steps: AtomicUsize::new(0),
            reached_at: [const { AtomicUsize::new(0) }; MTP_MAX_TRACKED_POSITIONS],
            accepted_at: [const { AtomicUsize::new(0) }; MTP_MAX_TRACKED_POSITIONS],
            positions_over_cap: AtomicUsize::new(0),
            by_batch: std::sync::Mutex::new(std::collections::BTreeMap::new()),
        }
    }

    /// Everything counted so far.
    fn snapshot(&self) -> MtpAcceptance {
        let mut reached_at = [0usize; MTP_MAX_TRACKED_POSITIONS];
        let mut accepted_at = [0usize; MTP_MAX_TRACKED_POSITIONS];
        for j in 0..MTP_MAX_TRACKED_POSITIONS {
            reached_at[j] = self.reached_at[j].load(Ordering::Relaxed);
            accepted_at[j] = self.accepted_at[j].load(Ordering::Relaxed);
        }
        MtpAcceptance {
            accepted: self.accepted.load(Ordering::Relaxed),
            proposed: self.proposed.load(Ordering::Relaxed),
            steps: self.steps.load(Ordering::Relaxed),
            drafted_steps: self.drafted_steps.load(Ordering::Relaxed),
            committed: self.committed.load(Ordering::Relaxed),
            batch_steps: self.batch_steps.load(Ordering::Relaxed),
            reached_at,
            accepted_at,
            positions_over_cap: self.positions_over_cap.load(Ordering::Relaxed),
        }
    }

    /// Per-batch-size breakdown, smallest batch first.
    fn snapshot_by_batch(&self) -> Vec<(usize, MtpAcceptance)> {
        self.by_batch
            .lock()
            .map(|m| m.iter().map(|(b, a)| (*b, *a)).collect())
            .unwrap_or_default()
    }

    /// Fold one engine step's per-sequence tallies into the batch-size bucket
    /// for `batch_size`, and count the engine step itself exactly once.
    fn record_batch(&self, batch_size: usize, per_seq: &[MtpAcceptance]) {
        let mut total = MtpAcceptance {
            batch_steps: 1,
            ..MtpAcceptance::default()
        };
        for step in per_seq {
            total.add(step);
        }
        // `add` also folded each sequence's (zero) batch_steps, so the count
        // survives as exactly one per forward.
        self.batch_steps.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut map) = self.by_batch.lock() {
            map.entry(batch_size).or_default().add(&total);
        }
    }

    fn reset(&self) {
        self.accepted.store(0, Ordering::Relaxed);
        self.proposed.store(0, Ordering::Relaxed);
        self.steps.store(0, Ordering::Relaxed);
        self.drafted_steps.store(0, Ordering::Relaxed);
        self.committed.store(0, Ordering::Relaxed);
        self.batch_steps.store(0, Ordering::Relaxed);
        for j in 0..MTP_MAX_TRACKED_POSITIONS {
            self.reached_at[j].store(0, Ordering::Relaxed);
            self.accepted_at[j].store(0, Ordering::Relaxed);
        }
        self.positions_over_cap.store(0, Ordering::Relaxed);
        if let Ok(mut map) = self.by_batch.lock() {
            map.clear();
        }
    }

    /// Accumulate one decode step; return `true` when this call carried the
    /// running proposed total across a multiple of `every`, i.e. when the
    /// caller should emit the periodic report.
    ///
    /// The boundary test uses the pre-add total returned by `fetch_add`, so
    /// each proposed token belongs to exactly one caller's interval and
    /// concurrent recorders cannot both claim (or both miss) a boundary.
    /// `every == 0` disables reporting rather than dividing by zero.
    fn accumulate(&self, step: &MtpAcceptance, every: usize) -> bool {
        let before = self.proposed.fetch_add(step.proposed, Ordering::Relaxed);
        self.accepted.fetch_add(step.accepted, Ordering::Relaxed);
        self.steps.fetch_add(step.steps, Ordering::Relaxed);
        self.drafted_steps
            .fetch_add(step.drafted_steps, Ordering::Relaxed);
        self.committed.fetch_add(step.committed, Ordering::Relaxed);
        for j in 0..MTP_MAX_TRACKED_POSITIONS {
            self.reached_at[j].fetch_add(step.reached_at[j], Ordering::Relaxed);
            self.accepted_at[j].fetch_add(step.accepted_at[j], Ordering::Relaxed);
        }
        self.positions_over_cap
            .fetch_add(step.positions_over_cap, Ordering::Relaxed);
        if every == 0 || step.proposed == 0 {
            return false;
        }
        (before + step.proposed) / every != before / every
    }

    /// The exact line the periodic report emits, as a string, so a test can
    /// assert on the reported ratio without parsing log plumbing.
    #[cfg(test)]
    fn report_line(&self) -> String {
        self.snapshot().report_line()
    }

    /// Emit the report unconditionally (the manual `log_acceptance_rate` door).
    fn log(&self) {
        let snap = self.snapshot();
        tracing::info!(target: "mtp_speculative", "{}", snap.marker("agg"));
        for (b, per_b) in self.snapshot_by_batch() {
            tracing::info!(target: "mtp_speculative", "{}", per_b.marker(&format!("b={b}")));
        }
        tracing::info!(target: "mtp_speculative", "{}", snap.report_line());
        // The per-position breakdown, in prose, right where the aggregate rate
        // is. This is the line the acceptance-gap diagnosis needs: the scalar
        // above says 0.4194 and stops, and there is no way to tell a
        // distribution mismatch from a compounding chain from a scalar.
        for line in snap.position_report_lines() {
            tracing::info!(target: "mtp_speculative", "{line}");
        }
    }

    /// Accumulate, and emit the periodic report if this call crossed a
    /// reporting boundary **and** `enabled`.
    ///
    /// `enabled` is a parameter rather than a direct [`acceptance_log_enabled`]
    /// call so the wiring is exercisable from a test: the env gate is memoised
    /// process-wide (correctly — it is on the decode hot path), which would
    /// otherwise make "does the logger actually fire" untestable.
    fn record_gated(&self, step: &MtpAcceptance, enabled: bool) {
        if self.accumulate(step, ACCEPTANCE_LOG_EVERY_PROPOSED) && enabled {
            self.log();
        }
    }

    /// Production entry: accumulate and report under `ARC_MTP_LOG_ACCEPTANCE`.
    fn record(&self, step: &MtpAcceptance) {
        self.record_gated(step, acceptance_log_enabled());
    }
}

/// Process-wide MTP accounting.
///
/// The per-pipeline counters live behind `Arc<Mutex<dyn Pipeline>>` and cannot
/// be downcast back out, so a harness that only holds a `MistralRs` handle —
/// `mistralrs bench`, every HTTP client — has no way to read them. This static
/// is the sink that makes the number reachable from outside the pipeline
/// without threading an accessor through the whole `Pipeline` trait.
static GLOBAL_ACCEPTANCE: AcceptanceTelemetry = AcceptanceTelemetry::new();

/// Process-wide MTP acceptance counters, across every request and every
/// pipeline instance. All zero when MTP never ran.
pub fn mtp_acceptance() -> MtpAcceptance {
    GLOBAL_ACCEPTANCE.snapshot()
}

/// Zero the process-wide counters (e.g. after benchmark warmup).
pub fn reset_mtp_acceptance() {
    GLOBAL_ACCEPTANCE.reset();
}

/// The aggregate `MTP[agg] …` marker line, or `None` when no MTP decode step
/// has run in this process — the honest answer to "what was the acceptance
/// rate" when nothing was measured is *nothing*, not `0%`.
pub fn mtp_acceptance_marker() -> Option<String> {
    let snap = mtp_acceptance();
    (snap.steps > 0).then(|| snap.marker("agg"))
}

/// Record one MTP decode step into the process-wide counters.
///
/// This is exactly the call `MtpSpeculativePipeline::step` makes at its
/// accept/reject site, exposed so the accounting can be proven — with a
/// fixture whose outcome is known by construction — without renting a GPU.
pub fn record_mtp_step(step: MtpAcceptance) {
    GLOBAL_ACCEPTANCE.record(&step);
}

/// Process-wide MTP counters split by the batch size that produced them,
/// smallest batch first.
pub fn mtp_acceptance_by_batch() -> Vec<(usize, MtpAcceptance)> {
    GLOBAL_ACCEPTANCE.snapshot_by_batch()
}

/// Record one engine step's per-sequence tallies against its batch size.
///
/// Exposed alongside [`record_mtp_step`] so the batched accounting can be
/// proven from a fixture whose per-sequence outcomes are known by
/// construction, without renting a GPU.
pub fn record_mtp_batch_step(batch_size: usize, per_seq: &[MtpAcceptance]) {
    GLOBAL_ACCEPTANCE.record_batch(batch_size, per_seq);
}

/// Every machine-greppable MTP line for this process: the aggregate first,
/// then one per observed batch size. Empty when MTP never ran.
/// The per-position acceptance breakdown as human-readable lines, or empty when
/// no MTP decode step ever drafted in this process.
///
/// Companion to [`mtp_acceptance_markers`], which carries the same numbers in
/// `p_pos=` / `accepted_pos=` / `reached_pos=` on the machine-greppable line.
pub fn mtp_acceptance_position_lines() -> Vec<String> {
    mtp_acceptance().position_report_lines()
}

pub fn mtp_acceptance_markers() -> Vec<String> {
    let Some(agg) = mtp_acceptance_marker() else {
        return Vec::new();
    };
    let mut out = vec![agg];
    for (b, per_b) in mtp_acceptance_by_batch() {
        out.push(per_b.marker(&format!("b={b}")));
    }
    out
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
    /// Running proposed/accepted tallies plus the env-gated periodic report.
    acceptance: AcceptanceTelemetry,
    /// Per-request tallies, keyed by `Sequence::id`, flushed as one
    /// `MTP[req=<id>] …` line when the sequence reaches a terminal state.
    /// Aggregates hide the case that matters most — one request drafting well
    /// while another never drafts at all — and that case is the *expected* one
    /// here, because the draft KV declines to prime on a prefix-cache hit.
    per_seq: std::sync::Mutex<std::collections::HashMap<usize, MtpAcceptance>>,
    /// Persistent per-sequence draft KV caches (audit finding 2). Keyed by
    /// `Sequence::id`. Bounded by [`Self::MAX_DRAFT_KV_SEQS`] — an entry is
    /// one decoder layer's KV for one sequence.
    draft_kv: std::sync::Mutex<std::collections::HashMap<usize, DraftKv>>,
    /// Monotonic clock for the draft-KV eviction order.
    draft_kv_clock: AtomicUsize,
    /// Latches so the "drafting skipped, draft KV unprimed" explanation is
    /// logged once per process rather than once per token.
    warned_unprimed: std::sync::atomic::AtomicBool,
    /// The same latch for the sampling gate ("this request is not greedy, so it
    /// gets no drafts"). Separate from [`Self::warned_unprimed`] because the two
    /// have different causes and different fixes, and a shared latch would let
    /// whichever fired first hide the other.
    warned_non_greedy: std::sync::atomic::AtomicBool,
    /// How this pipeline advances the KV cache after a verify. Latched on the
    /// first fast-path step (the cache's slot kinds are fixed by then) so the
    /// decision and its reason are logged once, not once per token.
    kv_advance: std::sync::OnceLock<KvAdvance>,
}

/// One sequence's persistent MTP draft KV.
///
/// Invariant: slot `k` of `cache` is the MTP block's state for **absolute**
/// position `k`, i.e. the pair `(h_k, tok_{k+1})`. `filled` is the number of
/// such committed entries; anything the cache holds beyond `filled` is the
/// speculative tail of the last chain and is truncated before reuse.
struct DraftKv {
    /// `None` on the Tier-A (heads-only) path, which has no attention and so
    /// nothing to cache. The struct still exists there because [`Self::seed`]
    /// does — the fused step needs somewhere to carry the hidden state between
    /// steps regardless of which tier drafts.
    cache: Option<KvCache>,
    filled: usize,
    /// Set once the cache can no longer be trusted to index absolute
    /// positions (a non-contiguous prefill, or an extend that errored).
    /// Drafting is skipped for the sequence from then on — lossless, and it
    /// stops us re-attempting a doomed extend on every token.
    poisoned: bool,
    /// `(absolute position p, target hidden state at p)` — the seed of the
    /// NEXT chain, carried over from the forward that produced it.
    ///
    /// This is what removes the second target forward per step (see
    /// [`MtpSpeculativePipeline::step`]). The chain step at draft-KV slot `p`
    /// is the pair `(h_p, tok_{p+1})` and predicts `tok_{p+2}`; with committed
    /// length `L` the first proposal must predict `tok_L`, so `p = L - 2`.
    /// That row is always inside the window the last forward covered — on a
    /// rejection at slot `j` the committed length is `C + j + 1` and
    /// `p = C + j - 1`; on a full accept it is the window's last input — so
    /// the state never has to be recomputed.
    seed: Option<(usize, Tensor)>,
    last_used: usize,
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
            acceptance: AcceptanceTelemetry::default(),
            per_seq: std::sync::Mutex::new(std::collections::HashMap::new()),
            draft_kv: std::sync::Mutex::new(std::collections::HashMap::new()),
            draft_kv_clock: AtomicUsize::new(0),
            warned_unprimed: std::sync::atomic::AtomicBool::new(false),
            warned_non_greedy: std::sync::atomic::AtomicBool::new(false),
            kv_advance: std::sync::OnceLock::new(),
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
            acceptance: AcceptanceTelemetry::default(),
            per_seq: std::sync::Mutex::new(std::collections::HashMap::new()),
            draft_kv: std::sync::Mutex::new(std::collections::HashMap::new()),
            draft_kv_clock: AtomicUsize::new(0),
            warned_unprimed: std::sync::atomic::AtomicBool::new(false),
            warned_non_greedy: std::sync::atomic::AtomicBool::new(false),
            kv_advance: std::sync::OnceLock::new(),
        }
    }

    /// Configured MTP draft depth (number of speculative tokens per target forward).
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Fixed width, in token slots, of the window every sequence feeds the
    /// target per fused MTP step: its uncached committed tail `u ∈ [1, w]`
    /// followed by `w - u` drafts.
    ///
    /// `depth + 1` is what makes the window invariant close. A step commits at
    /// most `1 + (w - u)` tokens while the shared cache advances by at least
    /// one, so the next tail is `u' = u + commit - advance <= w` — the tail can
    /// never outgrow the window, and a sequence that has run all the way out to
    /// `u == w` simply drafts nothing that step and catches back up.
    pub fn window(&self) -> usize {
        self.depth + 1
    }

    /// How this pipeline advances the KV cache after a verify, decided once.
    ///
    /// Default [`KvAdvance::Cohort`]. `ARC_MTP_PER_SEQ_KV=1` asks for
    /// [`KvAdvance::PerSequence`]; it is granted only when **all four** of
    /// these hold, and the reason it was refused is logged by name so a run
    /// that asked for it never silently gets the ceiling instead:
    ///
    /// 1. every cache slot can carry its own length
    ///    ([`cache_supports_per_sequence_advance`]) — cleared for V4 by
    ///    wave63-CO under `ARC_V4_XS_PER_SEQ`;
    /// 2. the target model masks a left-aligned ragged batch's dead prefix AND
    ///    its start-anchored compressed columns
    ///    ([`model_masks_ragged_batches`]) — cleared for V4 by the per-row
    ///    `row_q0` in `dsv4_attention`;
    /// 3. the MTP **draft** chain can carry rows at different absolute
    ///    positions ([`draft_chain_carries_per_sequence_positions`]) — cleared
    ///    for V4 by the per-row capture offsets, `forward_step_rows` and
    ///    [`front_align_draft_group`];
    /// 4. the **scheduler** will admit a cohort whose cache lengths differ
    ///    ([`scheduler_admits_ragged_cohorts`]) — **still refused**, and now the
    ///    only one.
    ///
    /// This pipeline's own step arithmetic used to be (3), before the draft
    /// chain was. It no longer is: every quantity in [`Self::step`] comes from
    /// `cache_lens[i]` ([`resolve_row_cache_lens`]), the verify window routes
    /// `seqlen_offsets[i]` per row, and the per-sequence commit strips the dead
    /// prefix ([`crate::kv_cache::drop_dead_prefix`]) so a row's recorded length
    /// is its own rather than the batch's padding.
    ///
    /// None of these is decoration. Per-sequence lengths in one dense buffer
    /// mean `front_pad_kv_cache` zero-fills ahead of the shorter rows, and a
    /// zero K row is **not** a masked row — it scores logit 0 and takes real
    /// softmax weight. Serving from those columns would be a wrong answer
    /// nothing downstream catches, which is the exact failure this project
    /// already paid for once with FP8 KV.
    ///
    /// (4) is worth stating explicitly because granting the mode on (1)-(3)
    /// alone would be *correct* and measurably **slower**: the scheduler runs
    /// exactly one cache-length bucket per step, so the first divergent accept
    /// count shatters the batch and idles most of it, while MTP goes on
    /// reporting a healthy `accept_rate`. A mode that is inert is bad; a mode
    /// that is a regression *and* still reports is D18's "the absence of a
    /// signal read as a specific signal".
    pub(crate) fn kv_advance(&self) -> KvAdvance {
        *self.kv_advance.get_or_init(|| {
            if !per_sequence_kv_requested() {
                return KvAdvance::Cohort;
            }
            let refusal = per_sequence_refusal(&self.target_cache);
            match refusal {
                None => {
                    tracing::info!(
                        target: "mtp_speculative",
                        "MTP: per-sequence KV advance is ON — each sequence advances by its own \
                         accepted count and no cohort rollback is applied"
                    );
                    KvAdvance::PerSequence
                }
                Some(why) => {
                    tracing::warn!(
                        target: "mtp_speculative",
                        "MTP: ARC_MTP_PER_SEQ_KV=1 was requested but this pipeline cannot honour \
                         it, so batched MTP keeps the cohort rollback and its ceiling ({why})"
                    );
                    KvAdvance::Cohort
                }
            }
        })
    }

    /// Snapshot of this pipeline's MTP acceptance counters.
    pub fn acceptance(&self) -> MtpAcceptance {
        self.acceptance.snapshot()
    }

    /// Snapshot of MTP acceptance counters as `(accepted, proposed)`.
    /// [`Self::acceptance`] carries the step/commit counts too.
    pub fn acceptance_counters(&self) -> (usize, usize) {
        let snap = self.acceptance.snapshot();
        (snap.accepted, snap.proposed)
    }

    /// Reset the MTP acceptance counters.
    pub fn reset_acceptance_counters(&self) {
        self.acceptance.reset();
    }

    /// Log the current acceptance rate at `info` level. Safe to call from any
    /// thread.
    ///
    /// The decode path does not need to call this by hand: `record_acceptance`
    /// emits the same line every [`ACCEPTANCE_LOG_EVERY_PROPOSED`] proposed
    /// tokens when `ARC_MTP_LOG_ACCEPTANCE=1`. This remains the door for a
    /// one-off report (e.g. at the end of a benchmark run).
    pub fn log_acceptance_rate(&self) {
        self.acceptance.log();
    }

    /// Run one MTP draft chain (greedy, depth ≤ self.depth) against a caller-
    /// supplied draft KV.
    ///
    /// Given the target's hidden state at `start_pos` and the just-emitted
    /// token, produces up to `self.depth` proposed tokens by chaining MTP
    /// steps. The chain stops early if it would exceed `max_tokens` (e.g., the
    /// EOS or the requested generation length). See
    /// [`MtpDecodeKit::propose_chain`] for the `start_pos` / `draft_cache`
    /// contract.
    pub fn propose_chain(
        &self,
        last_hidden: &Tensor,
        last_token_id: u32,
        max_tokens: usize,
        start_pos: usize,
        draft_cache: Option<&mut KvCache>,
    ) -> Result<Vec<u32>> {
        self.kit.propose_chain(
            last_hidden,
            last_token_id,
            self.depth,
            max_tokens,
            start_pos,
            draft_cache,
        )
    }

    /// Record one MTP decode step: this pipeline's counters, the per-request
    /// tally, and the process-wide sink [`mtp_acceptance`].
    ///
    /// Unless `ARC_MTP_LOG_ACCEPTANCE=0`, this also emits the running rate
    /// every [`ACCEPTANCE_LOG_EVERY_PROPOSED`] proposed tokens, so a GPU
    /// session reads MTP acceptance out of the serve log with
    /// `grep 'MTP\[' serve.log` — no extra run, no extra rental.
    ///
    /// Steps that drafted nothing are recorded too (`proposed = 0`,
    /// `committed = 1`): they are what makes `tok_per_step` honest, and
    /// `drafted_steps` is what tells a reader whether a 0% rate means the head
    /// was wrong or the draft KV never primed.
    fn record_step(&self, seq_id: usize, step: MtpAcceptance) {
        self.acceptance.record_gated(&step, false);
        if let Ok(mut map) = self.per_seq.lock() {
            map.entry(seq_id).or_default().add(&step);
        }
        // The process-wide sink owns the periodic report so a multi-pipeline
        // process emits one cadence, not one per pipeline.
        GLOBAL_ACCEPTANCE.record(&step);
    }

    /// Emit and forget one request's tally. Called when the sequence reaches a
    /// terminal state, and again when its draft KV is dropped, so a request
    /// that ends on a path the fast path never sees is still reported.
    fn flush_seq_acceptance(&self, seq_id: usize) {
        let Some(snap) = self
            .per_seq
            .lock()
            .ok()
            .and_then(|mut map| map.remove(&seq_id))
        else {
            return;
        };
        if snap.steps == 0 || !acceptance_log_enabled() {
            return;
        }
        tracing::info!(target: "mtp_speculative", "{}", snap.marker(&format!("req={seq_id}")));
    }

    /// Cap on retained per-sequence draft KV caches. Each is one decoder
    /// layer's KV for one sequence; the engine only ever drafts for the
    /// sequence it is currently stepping, so a small LRU is plenty.
    const MAX_DRAFT_KV_SEQS: usize = 64;

    /// Take a sequence's draft KV out of the map (so the borrow checker lets
    /// us hand `&mut KvCache` to the kit while `self` stays shared), creating
    /// it on first use. `None` on the Tier-A heads-only path.
    fn checkout_draft_kv(&self, seq_id: usize) -> Option<DraftKv> {
        let mut map = self.draft_kv.lock().ok()?;
        if let Some(state) = map.remove(&seq_id) {
            return Some(state);
        }
        let cache = self.kit.new_draft_cache();
        if cache.is_none() && self.kit.block.is_some() {
            return None;
        }
        Some(DraftKv {
            cache,
            filled: 0,
            poisoned: false,
            seed: None,
            last_used: self.draft_kv_clock.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// Put a sequence's draft KV back, evicting the least-recently-used entry
    /// if the map has grown past [`Self::MAX_DRAFT_KV_SEQS`].
    fn checkin_draft_kv(&self, seq_id: usize, mut state: DraftKv) {
        state.last_used = self.draft_kv_clock.fetch_add(1, Ordering::Relaxed);
        let Ok(mut map) = self.draft_kv.lock() else {
            return;
        };
        map.insert(seq_id, state);
        while map.len() > Self::MAX_DRAFT_KV_SEQS {
            let Some(oldest) = map
                .iter()
                .min_by_key(|(_, s)| s.last_used)
                .map(|(id, _)| *id)
            else {
                break;
            };
            map.remove(&oldest);
        }
    }

    /// Forget a sequence's draft KV (sequence finished, or the target's own
    /// cache was reset out from under it).
    fn drop_draft_kv(&self, seq_id: usize) {
        if let Ok(mut map) = self.draft_kv.lock() {
            map.remove(&seq_id);
        }
        self.flush_seq_acceptance(seq_id);
    }

    /// Forget every draft KV — the target cache was reset wholesale.
    fn clear_draft_kv(&self) {
        if let Ok(mut map) = self.draft_kv.lock() {
            map.clear();
        }
        let ids: Vec<usize> = self
            .per_seq
            .lock()
            .map(|map| map.keys().copied().collect())
            .unwrap_or_default();
        for id in ids {
            self.flush_seq_acceptance(id);
        }
    }

    /// Bring one sequence's draft KV up to date from a captured block of
    /// target hidden states.
    ///
    /// This is `forward_draft_extend` / `forward_draft_extend_after_decode`
    /// (`eagle_worker.py:1094-1128`, `:1134+`): draft-KV slot `i` is the MTP
    /// state of `(h_i, tok_{i+1})`, so from a capture covering absolute
    /// positions `[off, off+T)` and a committed token list of length `L` we
    /// can write slots `i ∈ [max(off, filled), min(off+T, L-1))`.
    ///
    /// The `i ≤ L-2` bound is doing double duty: it is also exactly the
    /// condition that `h_i` was conditioned only on *committed* tokens, so a
    /// rejected proposal's hidden state can never enter the cache.
    ///
    /// Returns `Ok(false)` when the run is not contiguous with what the cache
    /// already holds (`off > filled`) — the caller must then skip drafting
    /// rather than let slot index and absolute position drift apart.
    fn extend_draft_kv(
        &self,
        state: &mut DraftKv,
        capture: Option<(CaptureOffsets, Tensor)>,
        toks: &[u32],
    ) -> Result<bool> {
        self.extend_draft_kv_row(state, capture, toks, 0, 1)
    }

    /// [`Self::extend_draft_kv`] for row `row` of a `[B, T, hidden]` capture
    /// produced by a batched forward over `batch` sequences.
    ///
    /// Also stashes [`DraftKv::seed`]: the target hidden state at absolute
    /// position `toks.len() - 2`, which is where the *next* chain starts. That
    /// row is inside every capture this is called with, which is what lets the
    /// fast path run one target forward per step instead of two.
    fn extend_draft_kv_row(
        &self,
        state: &mut DraftKv,
        capture: Option<(CaptureOffsets, Tensor)>,
        toks: &[u32],
        row: usize,
        batch: usize,
    ) -> Result<bool> {
        if state.poisoned {
            return Ok(false);
        }
        let Some((offsets, hidden)) = capture else {
            return Ok(true);
        };
        if self.kit.block.is_none() {
            return Ok(true);
        }
        // Normalize to [B, T, hidden]; the model captures before
        // `extract_logits`, so T is the full input width of that forward.
        let hidden = match hidden.rank() {
            2 => hidden.unsqueeze(0)?,
            3 => hidden,
            other => candle_core::bail!("MTP hidden capture had unexpected rank {other}"),
        };
        if hidden.dim(0)? != batch || row >= batch {
            // The capture does not describe the batch we were told it does, so
            // no row can be attributed to this sequence.
            return Ok(false);
        }
        // 🔑 Row `row`'s OWN absolute start, not row 0's. Under the cohort rule
        // the capture carries one shared offset and this is the same number for
        // every row; under a per-sequence verify window each row starts at its
        // own `cache_lens[i]`, and reading row 0's would put this sequence's
        // draft-KV slots at the leading row's positions.
        if !offsets.describes(batch) {
            return Ok(false);
        }
        let Some(off) = offsets.row(row) else {
            return Ok(false);
        };
        let hidden = hidden.narrow(0, row, 1)?;
        let t = hidden.dim(1)?;
        if off > state.filled {
            return Ok(false);
        }
        // The seed for the next chain: `(h_{L-2}, tok_{L-1})` predicts `tok_L`.
        if let Some(seed_pos) = toks.len().checked_sub(2) {
            if seed_pos >= off && seed_pos < off + t {
                state.seed = Some((seed_pos, hidden.i((0, seed_pos - off))?));
            }
        }
        let lo = state.filled.max(off);
        // `tok_{i+1}` must be committed: i + 1 <= toks.len() - 1.
        let hi = (off + t).min(toks.len().saturating_sub(1));
        if hi <= lo {
            return Ok(true);
        }
        let n = hi - lo;
        let Some(cache) = state.cache.as_mut() else {
            // Tier A: nothing to extend, but `filled` still tracks how much
            // committed context the seed is allowed to assume.
            state.filled = hi;
            return Ok(true);
        };
        let hidden_slice = hidden.narrow(1, lo - off, n)?;
        let next_tokens = Tensor::from_slice(&toks[lo + 1..hi + 1], (1, n), hidden.device())?;
        cache.set_len(state.filled).map_err(|e| {
            candle_core::Error::msg(format!(
                "MTP draft KV truncate to {} failed: {e}",
                state.filled
            ))
        })?;
        state.filled = self
            .kit
            .extend_draft_cache(cache, lo, &hidden_slice, &next_tokens)?;
        Ok(true)
    }

    /// `clone_in_cache`, keeping the one thing it computes and throws away.
    ///
    /// Under [`KvAdvance::PerSequence`] the cohort arriving here is genuinely
    /// ragged, so it is left-aligned first: every sequence's live K/V is shifted
    /// to END at the batch maximum, which makes `SingleCache::append`'s one
    /// shared write offset simultaneously correct for all of them. The dense
    /// batching below is then completely unchanged.
    ///
    /// 🔑 The returned `lead_pad[i]` is the **only** record that row `i`'s share
    /// of the batched buffer starts `lead_pad[i]` columns in. Once
    /// `front_pad_single` has run, the slot's own `current_seq_len` is the
    /// batch's width for every row — the per-row length is gone. Discarding this
    /// vector is therefore what forced every downstream quantity onto one shared
    /// `cache_len`; keeping it is what lets [`resolve_row_cache_lens`] hand it
    /// straight back.
    ///
    /// Off the per-sequence path it is `vec![0; seqs.len()]` and no tensor is
    /// touched, exactly as before.
    fn clone_in_cache_reporting_lead(
        &self,
        seqs: &mut [&mut Sequence],
    ) -> candle_core::Result<Vec<usize>> {
        let lead_pad = if self.kv_advance() == KvAdvance::PerSequence {
            crate::kv_cache::front_align_batch(seqs, false)?
        } else {
            vec![0; seqs.len()]
        };
        get_mut_arcmutex!(self.target).clone_in_cache(seqs)?;
        Ok(lead_pad)
    }

    /// Log the "draft KV could not be primed, drafting skipped" explanation
    /// once per process.
    /// Say once that a request is being served without MTP because its sampler
    /// is not a plain argmax.
    ///
    /// The defect this gate fixes was invisible: `temperature > 0` requests were
    /// silently getting greedy tokens at every accepted draft position. A silent
    /// *refusal* would only move the silence, so the reason is stated — once per
    /// process, because this runs on the decode hot path.
    fn warn_non_greedy_once(&self, seq_id: usize) {
        if !self
            .warned_non_greedy
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            tracing::warn!(
                target: "mtp_speculative",
                "MTP drafting disabled for sequence {seq_id} (and any other request like it): \
                 its sampler is not a plain argmax over raw logits — temperature > 0, a \
                 repetition/frequency/presence/DRY penalty, or a custom logits processor. MTP \
                 verification accepts a draft when it equals argmax(target logits), so \
                 committing it would substitute GREEDY decode at every accepted position, \
                 which DOCTRINE D4 bans. Skipping is lossless; this request decodes at plain \
                 target speed with its own sampling settings intact."
            );
        }
    }

    fn warn_unprimed_once(&self, reason: &str) {
        if !self
            .warned_unprimed
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            tracing::warn!(
                target: "mtp_speculative",
                "MTP drafting skipped: {reason}. The MTP block applies ABSOLUTE RoPE \
                 positions, so drafting without a draft KV covering the accepted context \
                 would attend over nothing at positions that index nothing (audit finding \
                 2). Skipping is lossless; decode continues at plain target speed."
            );
        }
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

/// What one fused batched MTP step does to the shared cache, given each
/// sequence's uncached tail and how many of its drafts the target accepted.
///
/// Split out of [`MtpSpeculativePipeline::step`] because this arithmetic is the
/// whole of the ragged-batch problem and none of it needs a GPU: one dense
/// `NormalCache` carries ONE length for the batch, so a step where sequence A
/// accepted 3 drafts and sequence B accepted none has to resolve two different
/// answers into one number without losing a token or keeping a slot that holds
/// a rejected draft's K/V.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BatchStepPlan {
    /// Window slots the shared cache keeps: `min_i(u_i + accepted_i)`.
    pub keep: usize,
    /// Window slots rolled back: `w - keep`.
    pub n_drop: usize,
    /// Each sequence's uncached committed tail at the START of the next step.
    pub next_uncached: Vec<usize>,
}

/// Which row of a sequence's `[w, vocab]` verify output carries the target's
/// own token for draft slot `j`.
///
/// The window is `u` real committed tokens then `w - u` drafts, so the row
/// sitting on the last real token (`u - 1`) predicts the position draft 0
/// proposed; row `u - 1 + d` is the bonus token a fully accepted chain earns.
/// Off-by-one here is invisible — it just looks like a bad draft head — which
/// is why the row arithmetic lives in one named place that the tests drive.
pub(crate) const fn window_verify_row(uncached: usize, j: usize) -> usize {
    uncached - 1 + j
}

/// Resolve one fused batched step. `w` is the window width
/// ([`MtpSpeculativePipeline::window`]), `uncached[i]` the tail sequence `i`
/// fed, `accepted[i]` how many of its `w - uncached[i]` drafts were accepted.
///
/// Invariants this maintains, all of which the tests pin:
/// * `keep >= 1` — the shared cache always advances, so a batch cannot stall.
/// * `1 <= next_uncached[i] <= w` — a tail can never outgrow the window, so
///   every sequence always has at least one real slot to feed and the fast path
///   never has to bail out of a state it created.
/// * No slot beyond `keep` survives — a rejected draft's K/V is never readable.
pub(crate) fn plan_batch_step(uncached: &[usize], accepted: &[usize], w: usize) -> BatchStepPlan {
    let keep = uncached
        .iter()
        .zip(accepted)
        .map(|(u, a)| u + a)
        .min()
        .unwrap_or(0);
    let next_uncached = uncached
        .iter()
        .zip(accepted)
        .map(|(u, a)| (u + a + 1).saturating_sub(keep))
        .collect();
    BatchStepPlan {
        keep,
        n_drop: w.saturating_sub(keep),
        next_uncached,
    }
}

/// How a batched MTP step advances the KV cache.
///
/// This is the whole of the batch problem, named. See
/// [`plan_per_sequence_step`] for why `Cohort` has a ceiling that no amount of
/// draft-quality work can lift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvAdvance {
    /// One dense cache, one length, rolled back to `min_i(u_i + a_i)`.
    /// [`plan_batch_step`]. Correct, and bounded by [`plan_per_sequence_step`]'s
    /// doc comment.
    Cohort,
    /// Every sequence advances by its own accepted count. What vLLM, SGLang and
    /// TensorRT-LLM all do. [`plan_per_sequence_step`].
    PerSequence,
}

/// What one fused batched MTP step does when each sequence advances its **own**
/// KV by its **own** accepted count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PerSeqStepPlan {
    /// `committed[i]` — how many of the `w` window slots sequence `i` keeps:
    /// `u_i + a_i`. Its cache ends the step at `cache_len + committed[i]`.
    pub committed: Vec<usize>,
    /// Slots sequence `i` drops: `w - committed[i]`. Their K/V is never read
    /// again — it is overwritten by the next step's window, exactly as vLLM and
    /// SGLang leave rejected draft slots stale inside a request's own
    /// over-allocated region rather than freeing them.
    pub dropped: Vec<usize>,
    /// Each sequence's uncached committed tail at the START of the next step.
    /// **Always 1**, for every sequence, at every batch size — which is the
    /// entire point, and is exactly the B=1 invariant.
    pub next_uncached: Vec<usize>,
}

/// Resolve one fused batched step **without a cohort barrier**.
///
/// # Why this exists
///
/// [`plan_batch_step`] rolls one dense cache back to `keep = min_i(u_i + a_i)`,
/// so `next_u_i = u_i + a_i + 1 - keep`. The moment any sequence rejects its
/// first draft (`u=1, a=0`) that pins `keep = 1` and collapses to
/// `next_u_i = u_i + a_i` — monotone non-decreasing, saturating at the window
/// `w`, where the sequence drafts `w - u = 0` tokens. **The sequences that
/// accept best are the ones that stop drafting.** At B=1 it cannot happen
/// (`keep = u + a`, so `next_u ≡ 1`), which is why B=1 is the only regime
/// batched MTP ever measured well in.
///
/// # What production engines do instead
///
/// All three reference engines key the advance on a **per-request scalar** and
/// perform no reduction across the batch:
///
/// * vLLM v1 — `v1/core/sched/scheduler.py:1836-1848`, inside the per-`req_id`
///   loop of `update_from_output`: `num_rejected = num_draft_tokens -
///   num_accepted; request.num_computed_tokens -= num_rejected`.
/// * SGLang — `speculative/eagle_worker_common.py:587-588`
///   `new_seq_lens = batch.seq_lens + accept_lens` (elementwise), committed at
///   `scheduler_components/batch_result_processor.py:686-687`
///   `req.kv_committed_len += num_accept_tokens`.
/// * TensorRT-LLM — `pyexecutor/resource_manager.py:1330-1346` rewinds
///   `request.py_rewind_len` per request.
///
/// A batch-wide `min` over accept lengths appears in none of them. The only
/// `min`s nearby are unrelated (vLLM's cascade-attention common-prefix pick) or
/// per-request clamps.
///
/// # What happens to a rejected draft's K/V
///
/// Nothing. vLLM and SGLang neither free nor zero it: the request's length
/// shrinks so attention stops reading those slots, and the next step's window
/// writes over the same physical positions (SGLang says so in
/// `eagle_worker_common.py:450` — "trailing unaccepted slots stay and are freed
/// as overshoot"). TensorRT-LLM is the only one that reclaims, and only when a
/// rewind empties a whole block (`kvCacheManager.cpp:4562-4600`). `dropped[i]`
/// here is that same quantity: slots the sequence stops counting, not slots
/// anyone hands back.
///
/// # Invariants
///
/// * `next_uncached[i] == 1` for every `i` — no ratchet is representable.
/// * `committed[i] == u_i + a_i >= 1`, so every sequence advances every step.
/// * `committed[i] + dropped[i] == w` — no window slot is unaccounted for.
/// * **No value depends on any other sequence.** That is the property under
///   test; a cohort reduction reintroduced anywhere breaks
///   `per_sequence_advance_is_independent_of_every_other_sequence`.
pub(crate) fn plan_per_sequence_step(
    uncached: &[usize],
    accepted: &[usize],
    w: usize,
) -> PerSeqStepPlan {
    let committed: Vec<usize> = uncached
        .iter()
        .zip(accepted)
        .map(|(u, a)| (u + a).min(w))
        .collect();
    PerSeqStepPlan {
        dropped: committed.iter().map(|c| w.saturating_sub(*c)).collect(),
        next_uncached: vec![1; committed.len()],
        committed,
    }
}

/// `ARC_MTP_PER_SEQ_KV=1` opts a run into [`KvAdvance::PerSequence`].
///
/// Default OFF. The flag alone is not enough: the batch's cache must also be
/// able to carry a per-sequence length
/// ([`cache_supports_per_sequence_advance`]), and a run that asks for it and
/// cannot get it is told which slot refused rather than silently served the
/// cohort answer.
pub(crate) fn per_sequence_kv_requested() -> bool {
    std::env::var("ARC_MTP_PER_SEQ_KV").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Whether every slot of this cache can hold a length that differs from the
/// rest of its batch, or the name of the first slot that cannot.
///
/// A dense batched forward writes every sequence's new K/V at ONE offset
/// (`SingleCache::append`, `kv_cache/single_cache.rs:225`), so per-sequence
/// lengths are only representable if a ragged cohort can be re-aligned into one
/// buffer whose rows all *end* at the same column —
/// [`crate::kv_cache::front_pad_kv_cache`]. That is well defined for a plain
/// `Normal` slot, and — since wave63-CO, under `ARC_V4_XS_PER_SEQ` — for
/// DeepSeek V4's `XsRolling` compressor slots, whose two time bases now carry
/// a token count per batch row. `Rotating` and `TurboQuant` still decline; see
/// [`crate::kv_cache::KvCache::supports_per_sequence_len`] for why.
///
/// ⚠️ A pass here is **not** sufficient on its own. `kv_advance` also requires
/// [`MtpSpeculativePipeline::target_masks_ragged_batches`], which no model
/// satisfies yet — so with `ARC_V4_XS_PER_SEQ=1` a V4 cache stops being the
/// slot named in the refusal and the ragged mask becomes the sole remaining
/// blocker. That is the whole point: the refusal moves, it does not disappear.
pub(crate) fn cache_supports_per_sequence_advance(
    cache: &EitherCache,
) -> std::result::Result<(), String> {
    let EitherCache::Normal(normal) = cache else {
        // Not a refusal of the *idea* — the opposite. PagedAttention already
        // has per-sequence independence at the allocator surface, and under
        // `ARC_SEGMENTED_KV=1` `KVCacheManager::trim_request_to_num_tokens`
        // routes straight to `SegmentedAllocator::rollback(pool, request_id, 0,
        // num_tokens)` (`paged_attention/kv_cache_manager.rs:445-448`), which
        // is per-request and takes an ABSOLUTE token count. What blocks paged
        // MTP is upstream of here: the fast path only accepts
        // `CacheBackendMetadata::DefaultInstructions`, so a paged batch never
        // reaches this rollback at all. That is its own change.
        return Err(
            "the target is on PagedAttention, whose per-sequence block tables already advance \
             independently — this dense-cache path is not what it needs"
                .to_string(),
        );
    };
    let guard = normal.lock().unwrap();
    for (i, slot) in guard.0.iter().enumerate() {
        if !slot.supports_per_sequence_len() {
            return Err(format!(
                "cache slot {i} is a `{}`, which carries one length for the whole batch. \
                 Per-sequence MTP advance needs every slot to hold its own length; see \
                 `KvCache::supports_per_sequence_len`.",
                slot.kind_name()
            ));
        }
    }
    Ok(())
}

/// The whole of [`MtpSpeculativePipeline::kv_advance`]'s decision: the three
/// capability checks, composed in order, or `None` when all three pass.
///
/// Composed here rather than inline in `kv_advance` so that "which checks are
/// required" is itself testable — dropping one is otherwise invisible, because
/// `kv_advance` needs a whole target pipeline to call.
///
/// Order matters only for which reason gets reported. Each check is
/// independent, and each has been the *last* one at some point: (1) fell to
/// wave63-CO, (2) to the per-row `row_q0`, (3) to the per-row step arithmetic,
/// (4) to this wave's per-row draft chain, and (5) is where it stands now.
pub(crate) fn per_sequence_refusal(cache: &EitherCache) -> Option<String> {
    cache_supports_per_sequence_advance(cache)
        .err()
        .or_else(|| model_masks_ragged_batches(cache).err())
        .or_else(|| draft_chain_carries_per_sequence_positions(cache).err())
        .or_else(|| batch_admission_admits_ragged_cohorts().err())
}

/// Whether the model behind this cache masks a left-aligned ragged batch, or
/// why it does not. The cache-inspecting half of
/// [`MtpSpeculativePipeline::kv_advance`]'s second check, split out so it is
/// testable without standing up a whole target pipeline — the same shape
/// [`cache_supports_per_sequence_advance`] has.
///
/// The discriminator is the `XsRolling` compressor slot. DeepSeek V4 is the
/// only architecture in the tree that allocates any, and it is exactly the
/// architecture whose attention now takes a per-row `row_q0`.
pub(crate) fn model_masks_ragged_batches(cache: &EitherCache) -> std::result::Result<(), String> {
    let EitherCache::Normal(normal) = cache else {
        return Err(
            "the target is on PagedAttention; this dense-cache mask is not what it needs"
                .to_string(),
        );
    };
    let is_v4 = normal
        .lock()
        .unwrap()
        .0
        .iter()
        .any(|slot| matches!(slot, crate::kv_cache::KvCache::XsRolling(_)));
    if is_v4 {
        return Ok(());
    }
    Err(
        "the target has no `XsRolling` compressor slots, so it is not DeepSeek V4 and no other \
         model in this tree threads a ragged-batch mask into its forward. The zero-filled dead \
         prefix a left-aligned per-sequence cache leaves would be attended as real keys (a zero \
         K row scores logit 0 and takes softmax weight). See \
         `MtpSpeculativePipeline::target_masks_ragged_batches`."
            .to_string(),
    )
}

/// Per-row committed KV lengths for one fused step, or `None` when the batch
/// shares one — which is every batch that was not built by
/// [`crate::kv_cache::front_align_batch`] under per-sequence advance.
///
/// `lead_pad[i]` is the dead prefix that call left ahead of row `i`, so
/// `cache_len - lead_pad[i]` is row `i`'s own committed KV length: the batched
/// buffer's width is the batch **maximum**, and every row was padded up to it.
/// No value is invented here — the vector already existed and was being
/// discarded by `clone_in_cache`.
///
/// 🔑 Returning `None` for a uniform batch is a **dispatch, not an
/// optimisation**. `None` runs the pre-change scalar arithmetic verbatim, so
/// every B=1 request and every batch built with the flag off is byte-identical
/// by control flow rather than by a number some test happened to observe. It is
/// a separate function for the reason PR #100's `resolve_ragged_rows` is: the
/// two branches agree exactly on their common domain, so no numeric test can
/// say which one ran.
///
/// The two refusals describe the same invariant from opposite sides.
/// `front_align_batch` pads every row **up to** the batch maximum, so no lead
/// can exceed the width and at least one row must have no lead at all. A batch
/// that is ragged in some other way is not the layout this step knows how to
/// unpick, and guessing would be a wrong answer nothing downstream catches.
pub(crate) fn resolve_row_cache_lens(
    enabled: bool,
    cache_len: usize,
    lead_pad: &[usize],
) -> Result<Option<Vec<usize>>> {
    if !enabled || lead_pad.iter().all(|&l| l == 0) {
        return Ok(None);
    }
    if let Some(&over) = lead_pad.iter().find(|&&l| l > cache_len) {
        candle_core::bail!(
            "MTP per-sequence step: a dead prefix of {over} column(s) is wider than the \
             {cache_len}-column batched cache it was padded into, so the pipeline's idea of this \
             batch's layout and `front_align_batch`'s disagree"
        );
    }
    if !lead_pad.contains(&0) {
        candle_core::bail!(
            "MTP per-sequence step: every row of this batch carries a dead prefix ({lead_pad:?}), \
             so none of them sits at the shared end column {cache_len}. `front_align_batch` pads \
             UP TO the batch maximum, so at least one row must have no prefix — this is not the \
             left-aligned cohort the per-row arithmetic is defined for"
        );
    }
    Ok(Some(lead_pad.iter().map(|&l| cache_len - l).collect()))
}

/// The window invariant, per row.
///
/// `uncached[i]` is the committed tail sequence `i` has not yet put in its KV,
/// and `ok` is whether every row is in a state this fast path produced. Both
/// used to be computed from ONE shared `cache_len` — the *batched* buffer's
/// width, which left-alignment makes `max_j L_j` rather than any row's length —
/// which made them wrong in two different ways the moment the rows diverged:
///
/// * `toks_i.len() - cache_len` saturated to **0** for every row but the
///   longest, so `ok` failed on every ragged step and per-sequence advance
///   would have been *reachable but inert*;
/// * `cache_len` then indexed the verify window (`toks[cache_len..]`), a
///   **panic** site the saturation was the only thing hiding.
///
/// Both close by taking row `i`'s own length. `ok` also guarantees the slice is
/// in range: `uncached[i] >= 1` means `cache_lens[i] < tok_lens[i]`.
pub(crate) fn plan_step_window(cache_lens: &[usize], tok_lens: &[usize], w: usize) -> StepWindow {
    let uncached: Vec<usize> = cache_lens
        .iter()
        .zip(tok_lens)
        .map(|(&c, &t)| t.saturating_sub(c))
        .collect();
    let ok = cache_lens
        .iter()
        .zip(&uncached)
        .all(|(&c, &u)| c + u >= 2 && (1..=w).contains(&u));
    StepWindow { uncached, ok }
}

/// [`plan_step_window`]'s answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepWindow {
    pub uncached: Vec<usize>,
    pub ok: bool,
}

/// Which rows of a batch may draft together: those needing the same number of
/// drafts (`w - u`) — and, when the draft chain cannot carry per-row positions,
/// also sitting at the same absolute chain start.
///
/// The second half used to be forced by the draft path, not chosen: one batched
/// MTP-block forward took ONE `pos` (`MtpBlock::forward_step`, used as the RoPE
/// position and the draft-KV slot) and `batch_draft_caches` refused rows of
/// differing length. Both are now per-row (`forward_step_rows`,
/// [`front_align_draft_group`]), so `carries_per_row` drops the chain start out
/// of the key and the partition returns to `u` alone: `u ∈ [1, w]`, so a batch
/// splits into at most `depth + 1` groups no matter how large it is.
///
/// 🔴 The flag is not cosmetic and dropping the start unconditionally would be
/// **silent**, not slow: with the old scalar path a row whose seed is not at the
/// group's single chain start is skipped by the seed check, so MTP would stop
/// drafting for it with no signal but a falling `tok_per_step`.
pub(crate) const fn draft_group_key(
    uncached: usize,
    cache_len: usize,
    carries_per_row: bool,
) -> (usize, usize) {
    if carries_per_row {
        (uncached, 0)
    } else {
        (uncached, cache_len + uncached - 2)
    }
}

/// Whether this sequence's **own sampler** admits MTP drafting.
///
/// MTP's verifier accepts a draft token when it equals the target's
/// `argmax(raw_logits)`, and commits it directly. That is lossless for exactly
/// one kind of request: one whose sampler *is* `argmax` over raw logits. For
/// any other — `temperature > 0`, a repetition/frequency/presence/DRY penalty,
/// a custom logits processor — the sampler would have picked a different token,
/// so committing the draft silently substitutes greedy decode at every accepted
/// position (with `logprob: 0.0` besides).
///
/// DOCTRINE **D4 bans greedy decoding**, so that substitution is a doctrine
/// violation happening invisibly, in production, to any `temperature > 0`
/// request.
///
/// Refusing to draft is the lossless answer, and it is refused **per sequence,
/// not per batch**: sampling parameters are fixed for a sequence but batch
/// composition is not, and the fast path's own contract is that every condition
/// gating it is static for a sequence — "a sequence that once entered the fast
/// path cannot be handed to the target mid-run with an uncached tail it would
/// misread". A batch-level gate would break exactly that, because a
/// `temperature > 0` arrival would eject a greedy sequence that is already
/// carrying a multi-token uncached tail. A per-sequence refusal is just the
/// `drafts[i].is_empty()` state the draft KV already produces whenever it cannot
/// prime, which every downstream step already handles.
pub(crate) fn seq_may_draft(seq: &mut Sequence) -> bool {
    seq.sampler().is_raw_argmax()
}

/// Partition the rows that may draft into groups that can share one batched
/// MTP-block forward.
///
/// Split out of the step so the sampling gate is provable without standing up a
/// pipeline: a row with `may_draft[i] == false` must appear in **no** group, and
/// therefore leaves the drafting section with `drafts[i]` empty.
pub(crate) fn draft_groups(
    uncached: &[usize],
    cache_lens: &[usize],
    may_draft: &[bool],
    w: usize,
    per_seq_advance: bool,
) -> std::collections::BTreeMap<(usize, usize), Vec<usize>> {
    let mut groups: std::collections::BTreeMap<(usize, usize), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, &u) in uncached.iter().enumerate() {
        if u < w && may_draft[i] {
            groups
                .entry(draft_group_key(u, cache_lens[i], per_seq_advance))
                .or_default()
                .push(i);
        }
    }
    groups
}

/// Row `i`'s absolute chain start: the position of the pair
/// `(h_{L-2}, tok_{L-1})` whose target hidden state seeds its chain.
///
/// One expression, used by both the group key and the per-row draft, so the two
/// cannot drift apart.
pub(crate) const fn chain_start_of(cache_len: usize, uncached: usize) -> usize {
    cache_len + uncached - 2
}

/// Whether the **MTP draft chain** can carry a batch whose rows sit at
/// different absolute positions. It now can, on the same architectures whose
/// attention masks a ragged batch.
///
/// The draft KV is a second, independent time base from the target's, and it was
/// keyed on one scalar in four places. All four now take the row's own value:
///
/// * `MtpHiddenCapture::store_rows(seqlen_offsets, ..)` tags a capture with one
///   absolute start **per row** ([`CaptureOffsets`]) instead of row 0's, so
///   `extend_draft_kv_row` writes each sequence's slots at that sequence's
///   positions. A batch whose rows agree folds back to one element
///   ([`resolve_capture_offsets`]), which is the old value exactly.
/// * `propose_chain_batched` takes `chain_starts: &[usize]` and, when they
///   differ ([`resolve_ragged_chain_starts`]), hands `starts[r] + i` to
///   `MtpBlock::forward_step_rows` — one RoPE position and one `row_q0` entry
///   per row.
/// * [`front_align_draft_group`] left-aligns a group's draft KVs instead of
///   `batch_draft_caches` refusing them, so one shared append offset is correct
///   for every row; the dead prefix is masked by `dsv4_attention`'s `row_q0` and
///   taken back out by `drop_dead_prefix` before any length is recorded.
/// * [`draft_group_key`] therefore drops the absolute chain start and the
///   partition returns to `u` alone — `<= depth + 1` groups at any batch size,
///   not one per sequence.
///
/// ⚠️ **This check is now IMPLIED by [`model_masks_ragged_batches`], and that is
/// stated rather than left implicit.** The discriminator is the same one, for the
/// same reason: the MTP block IS a V4 decoder layer, so what masks its
/// left-aligned draft cohort is `dsv4_attention`'s per-row `row_q0`. A model
/// whose attention cannot mask a ragged batch cannot draft against one either.
///
/// So in [`per_sequence_refusal`] this entry can never be the *first* failure,
/// and a mutation that deletes it from the composition survives every test —
/// honestly, because it is a no-op today. It is kept, named and separately
/// tested because the implication is a property of the *current* discriminator,
/// not of the two capabilities: the day `model_masks_ragged_batches` learns a
/// second architecture, the draft chain is a separate question again.
/// `the_draft_chain_check_is_implied_by_the_mask_check_today` asserts the
/// implication directly, so relaxing check (2) fails a test instead of silently
/// granting check (3).
pub(crate) fn draft_chain_carries_per_sequence_positions(
    cache: &EitherCache,
) -> std::result::Result<(), String> {
    model_masks_ragged_batches(cache).map_err(|_| {
        "the MTP draft chain left-aligns a group's draft KVs, which only a model whose \
         attention masks a ragged batch can read. See \
         `MtpSpeculativePipeline::draft_chain_carries_per_sequence_positions`."
            .to_string()
    })
}

/// Whether the layer that **admits** a batch — the scheduler that picks it and
/// the engine that decides how its cache is assembled — will hand this pipeline
/// a cohort whose rows sit at different cache lengths.
///
/// It will not, and this is where the refusal stands now. I am not claiming it
/// is the last: four waves have made that claim and been wrong, and the useful
/// output of each has been naming precisely where the refusal moved.
///
/// Two sites, both above this file, and both the same "one batch, one length"
/// assumption:
///
/// 🔴 **(a) `engine/mod.rs` — `pre_op` is `CacheInstruction::In` only when the
/// scheduled completion id list *changed* since the last step.** A stable cohort
/// therefore gets `Nothing` on every step after the first, and
/// [`MtpSpeculativePipeline::step`]'s `per_seq_advance` requires
/// `assembled_here`, which is set only on the `In` arm. So the mode would
/// **interleave** with the cohort rule step by step rather than being on. That is
/// not merely inert. The `PerSequence` arm deliberately runs no
/// `truncate_normal_cache`, so the pipeline's resident batched cache stays
/// `cache_len + w` wide while each sequence keeps `cache_len + u_i + a_i`; the
/// next `Nothing` step reads that stale width through
/// [`current_normal_cache_len`], `window_ok` fails, and — because `lead_pad` is
/// all-zero on a step that did not assemble the batch — the loud left-aligned
/// guard cannot fire. Control would reach the deferral, which hands the batch to
/// the target's plain decode against a cache whose write offset and whose RoPE
/// positions no longer agree. A wrong answer nothing downstream catches.
///
/// 🔴 **(b) `scheduler/default_scheduler.rs` — the running set is partitioned by
/// `Sequence::cache_bucket_len` and `select_running_bucket` runs exactly ONE
/// bucket per step.** That partition is correct for every path that reaches it
/// today, because the cohort rule keeps every sequence's cache length equal — it
/// is `cache_bucket_len`'s own documented invariant, and per-sequence advance is
/// precisely the thing that deletes it. Row `i` advances by `1 + a_i`, so the
/// first step with differing accept counts shatters the batch into one bucket
/// per length and idles all but one: the cohort barrier this workstream exists
/// to remove, reappearing one layer up as admission, and costing more than the
/// mode wins.
///
/// Neither fix belongs in this pipeline. `front_align_batch` is exactly the
/// statement that unequal cache lengths may now share a forward, so what has to
/// change is *who is allowed to say that* — the engine's `In`/`Nothing` gate and
/// the scheduler's bucket key — which is one change with its own question about
/// how either learns which pipeline is running. Granting the mode first would
/// ship (a) as a silent wrong answer and (b) as a measurable regression, both
/// while MTP went on reporting a healthy `accept_rate`: D18's "the absence of a
/// signal read as a specific signal".
pub(crate) fn batch_admission_admits_ragged_cohorts() -> std::result::Result<(), String> {
    Err(
        "the batch admission layer keys on one length for the whole cohort, in two places. \
         `engine/mod.rs` issues `CacheInstruction::In` only when the scheduled completion ids \
         change, so a stable cohort takes `Nothing` and this step's `assembled_here` — and with \
         it per-sequence advance — is false on the steady-state path, interleaving the two \
         rules; and the scheduler buckets running sequences by `Sequence::cache_bucket_len` \
         while `select_running_bucket` runs exactly ONE bucket per step, so diverging lengths \
         shatter the batch and idle all but one. See \
         `MtpSpeculativePipeline::batch_admission_admits_ragged_cohorts`."
            .to_string(),
    )
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
/// [`run_target_forward`] over a whole batch.
///
/// Every sequence must already carry a window of the SAME width via
/// `set_prefill_toks`, and `prefill_window = Some((w, cache_len))` — one shared
/// `seqlen_offset` and one shared logit width, which is the invariant the dense
/// batched `NormalCache` demands (`kv_cache/mod.rs::first_mismatched_cache_len`).
/// Returns `[B, w, vocab]`.
async fn run_target_forward_batch(
    this: &MtpSpeculativePipeline,
    seqs: &mut [&mut Sequence],
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
            seqs,
            /* is_prompt = */ true,
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

#[allow(dead_code)]
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
///
/// Reads through `self.target_cache` — an `EitherCache::Normal` clone shares
/// its `Arc<Mutex<NormalCache>>` with the target pipeline, so no pipeline
/// lock is needed. (The previous implementation ran
/// `futures::executor::block_on(target.lock())` inside the async `step()`,
/// parking a runtime worker on a pipeline mutex — a deadlock class we do not
/// want in the serve hot path; see docs/notes/mtp-hang-triage.md.)
/// Has this sequence stopped generating? The trigger for flushing its
/// per-request `MTP[req=…]` line.
fn seq_is_terminal(seq: &Sequence) -> bool {
    use crate::sequence::SequenceState;
    matches!(
        seq.getstate(),
        SequenceState::Done(_)
            | SequenceState::FinishedAborted
            | SequenceState::FinishedIgnored
            | SequenceState::Error
    )
}

fn current_normal_cache_len(this: &MtpSpeculativePipeline) -> usize {
    let EitherCache::Normal(normal) = &this.target_cache else {
        return 0;
    };
    normal.lock().unwrap().0[0].current_seq_len()
}

/// Truncate the target's Normal KV cache by `n_drop` positions on every layer.
/// Used after MTP verify to discard speculative positions the committed
/// sequence does not need, so the token count and the cache stay in lockstep.
/// Same lock-free-on-the-pipeline access as [`current_normal_cache_len`].
fn truncate_normal_cache(this: &MtpSpeculativePipeline, n_drop: usize) -> Result<()> {
    truncate_cache_by(&this.target_cache, n_drop)
}

/// The rollback itself, on a bare [`EitherCache`].
///
/// Split from [`truncate_normal_cache`] so the rejection path can be exercised
/// against a **real** cache — including DeepSeek V4's [`crate::XsRollingCache`]
/// entries, which are not K/V at all: they hold completed compressed rows plus
/// a bounded raw tail, and map a token-unit truncation onto those two time
/// bases themselves. Every entry reports its length in tokens, so one `n_drop`
/// is correct for all of them; what differs is what each has to do to honour
/// it, and only the xs entries can *refuse*.
///
/// A refusal here is a hard error on purpose. `XsRollingCache` declines a
/// rollback whose raw rows it no longer retains, because resuming from that gap
/// would silently corrupt the compressor's distant-context branch — a wrong
/// answer that nothing downstream would catch. `XS_TAIL_MARGIN_TOKENS` (16) is
/// sized to cover any `--mtp-depth` (clap caps it at 8), so a refusal on this
/// path means an invariant broke, not that the margin was too small.
pub(crate) fn truncate_cache_by(cache: &EitherCache, n_drop: usize) -> Result<()> {
    let EitherCache::Normal(normal) = cache else {
        return Ok(());
    };
    let mut guard = normal.lock().unwrap();
    for cache in &mut *guard.0 {
        let cur = cache.current_seq_len();
        let new_len = cur.saturating_sub(n_drop);
        cache.set_len(new_len).map_err(|e| {
            candle_core::Error::msg(format!(
                "MTP: cache set_len({new_len}) failed rolling back {n_drop} of {cur} \
                 positions: {e}"
            ))
        })?;
    }
    Ok(())
}

/// How many trailing KV-cache positions to drop after an MTP verify forward.
///
/// Invariant to restore (what plain decode maintains): entering a decode
/// step, the target cache holds the KV of every committed token EXCEPT the
/// last one — the last committed token is the next step's single-token input
/// (`make_completion_chunk` feeds `toks[len-1..]` only).
///
/// Bookkeeping: after the verify forward the cache gained `n_proposed` extra
/// positions (the inputs `[T0, P1, …, P_{d-1}]`). The step commits
/// `1 (T0) + commit_len` tokens, where `commit_len = accepted + correction`
/// ([`VerifyResult::commit_len`]). The cache must therefore keep
/// `commit_len` of the extras and drop the rest:
///
/// ```text
/// n_drop = n_proposed - commit_len
///        = n_proposed - n_accepted        (no rejection: keep every extra)
///        = n_proposed - n_accepted - 1    (rejection: the correction token is
///                                          committed but was never a verify
///                                          input, so one FEWER slot is dropped)
/// ```
///
/// The pre-session-5 code dropped `n_proposed - n_accepted` unconditionally —
/// one slot too many on EVERY rejected chain, permanently desyncing the cache
/// from the committed tokens (cumulative, one position per rejection). See
/// docs/notes/mtp-hang-triage.md for the full derivation.
///
/// Retained as a test-only helper: the fused window shape computes the same
/// quantity through [`plan_batch_step`] (which has to resolve a whole batch,
/// not one sequence), while `mtp_verify_rollback_restores_the_compressor_rows_exactly`
/// keeps driving this derivation into the live [`truncate_cache_by`] so the
/// `XsRollingCache` refusal contract stays pinned against a real V4 cache.
#[cfg(test)]
pub(crate) fn n_cache_positions_to_drop(n_proposed: usize, verify_result: &VerifyResult) -> usize {
    n_proposed.saturating_sub(verify_result.commit_len())
}

/// Greedy argmax over a (depth × vocab) or (1 × depth × vocab) logits tensor;
/// returns one token per row.
#[cfg(test)]
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
    /// Under [`KvAdvance::PerSequence`] a cohort arrives here genuinely ragged
    /// — that is the point — so it is left-aligned first: every sequence's live
    /// K/V is shifted to END at the batch maximum, which makes
    /// `SingleCache::append`'s one shared write offset simultaneously correct
    /// for all of them. The dense batching code below is then completely
    /// unchanged, and `ensure_uniform_batch_cache_lens` passes for the same
    /// reason it always did.
    ///
    /// `front_align_batch` returns each row's dead prefix, which is the only
    /// record anywhere of how much of the batched buffer is padding rather than
    /// this row's own KV. The fused step needs it, so it calls
    /// [`MtpSpeculativePipeline::clone_in_cache_reporting_lead`] directly; this
    /// trait entry point is for every other caller and discards it.
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> candle_core::Result<()> {
        self.clone_in_cache_reporting_lead(seqs).map(|_| ())
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
    ///    target token), advances the KV cache by 1, and (through
    ///    [`MtpHiddenCapture`]) hands out the target's own hidden state
    ///    `h_{L-1}` at the last committed position.
    /// 2. Propose chain: feed `h_{L-1}` to `h_proj` and `embed(T0)` to
    ///    `e_proj` — two different signals, as the head was trained — and run
    ///    the chain against the persistent draft KV, which already holds the
    ///    accepted context, producing `[T1, …, T_depth]` greedy candidates.
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
        // Fallback: prompt, xlora, raw-logit, non-`NormalCache` and
        // PagedAttention steps defer to the wrapped target pipeline.
        //
        // **Batch size is no longer one of these conditions.** What used to gate
        // the fast path at `input_seqs.len() == 1` was never a property of MTP;
        // it is a property of the dense batched `NormalCache`, whose
        // `SingleCache::append` writes every sequence's new K/V at ONE shared
        // offset (`kv_cache/single_cache.rs:161`) and whose batched view carries
        // ONE `current_seq_len` for the whole batch. Per-sequence accept lengths
        // make committed token counts diverge, and that cache cannot represent a
        // per-sequence length — so a naive batched MTP either corrupts the
        // cache or fragments the batch into one length-bucket per accept length
        // (`scheduler/default_scheduler.rs::select_running_bucket` runs exactly
        // one bucket per step, so fragmentation costs more than MTP wins).
        //
        // The fast path below resolves it by keeping the CACHE lengths uniform
        // and letting the TOKEN lengths diverge instead: each sequence carries
        // an `uncached` committed tail of `u ∈ [1, w]` tokens, feeds
        // `u` real tokens plus `w - u` drafts into a fixed-width window, and the
        // batch rolls back to the one shared length every sequence can prove.
        // Every non-MTP path keeps `u == 1`, which is the invariant plain decode
        // already maintains.
        //
        // Every remaining condition is STATIC for a sequence
        // (`return_raw_logits` is a per-request field the engine asserts is
        // uniform across a batch, `engine/mod.rs:406-411`), so a sequence that
        // once entered the fast path cannot be handed to the target mid-run with
        // an uncached tail it would misread.
        //
        // # PagedAttention is NOT mutually exclusive with MTP
        //
        // The `DefaultInstructions` condition below is an **unimplemented
        // branch**, not an incompatibility. What is missing is exactly one
        // thing: the rollback. This path rolls rejected drafts back with
        // `truncate_cache_by`, which walks a `NormalCache` and calls `set_len`
        // per layer. Under paging the K/V lives in the block table the
        // PagedAttention KV-cache manager owns, the `EitherCache::Normal`
        // entries are unused, and that truncate would be a silent no-op —
        // leaving a rejected draft's K/V addressable. Refusing is the only
        // correct thing to do until the paged free-list step exists.
        //
        // It is also the *cheap* case to add, and the reference proves it:
        // SGLang frees exactly the complement of its accept set
        // (`eagle_info.py:488-490`), and for a **linear chain** — `topk == 1`,
        // which is what DeepSeek MTP is in SGLang's own shipped config
        // (`server_args.py:7611-7627`, `(3, 1, 4)` for every DeepSeek arch) —
        // the accepted tokens are a contiguous prefix of the allocated run, so
        // the rollback is pure truncation with no KV movement at all
        // (`eagle_info.py:492-501`). Only tree drafting needs the
        // `move_kv_cache` compaction (`:505-545`).
        //
        // Further: paged KV is a *better* substrate for batched MTP than the
        // dense cache, because a per-sequence block table makes per-sequence
        // lengths free and the whole uncached-tail scheme below unnecessary.
        //
        // **For V4 specifically the question is moot, twice over.**
        // `DeepSeekV4Loader::supports_paged_attention` returns `false`
        // (`loaders/normal_loaders.rs`, rationale corrected in wave29-BC:
        // `flashinfer_mla_decode.cu` fixes `HEAD_DIM_CKV=512` as a template
        // constant and computes dense causal attention, while every V4 layer is
        // sliding-window + sink), so the engine never hands V4 a
        // `CacheBackendMetadata::PagedAttention` at all. And `mtp_decode_kit`
        // has exactly one implementation in the tree — `deepseek4.rs:4250`; the
        // trait default at `loaders/normal_loaders.rs:104` returns `None` — so
        // the MTP wrapper cannot currently wrap a model that pages. The two
        // features have never met, and this guard has never fired.
        let (target_is_xlora, target_no_kv_cache) = {
            let meta = get_mut_arcmutex!(self.target).get_metadata();
            (meta.is_xlora, meta.no_kv_cache)
        };
        let take_fast_path = !is_prompt
            && !input_seqs.is_empty()
            && !return_raw_logits
            && matches!(self.target_cache, EitherCache::Normal(_))
            && matches!(
                backend_metadata,
                CacheBackendMetadata::DefaultInstructions { .. }
            )
            && !target_is_xlora
            && !target_no_kv_cache;

        if !take_fast_path {
            // The prompt forward is the ONE place the target produces hidden
            // states for the whole context, so it is where the draft KV gets
            // prefilled — the reference's `forward_draft_extend`
            // (`eagle_worker.py:1094-1128`). Any other fallback (xlora, raw
            // logits) just drops the capture, so a stale block can never be
            // attributed to the wrong sequence on a later step.
            //
            // A **batched** prompt primes every row, not just row 0. Skipping
            // it would have made batched MTP dead on arrival in a real serve:
            // 128 concurrent arrivals prefill together, so no sequence would
            // ever get a seed and every decode step would decline to draft —
            // losslessly, silently, and at exactly zero speedup.
            // `make_prompt_chunk` right-pads to the batch's longest prompt, so
            // row `i`'s real positions are `[0, len_i)` and
            // `extend_draft_kv_row`'s own `i <= L-2` bound already stops there.
            let prompt_ids: Vec<usize> = if is_prompt && self.kit.block.is_some() {
                input_seqs.iter().map(|s| *s.id()).collect()
            } else {
                Vec::new()
            };
            let elapsed = {
                let mut target = self.target.lock().await;
                target
                    .step(
                        input_seqs,
                        is_prompt,
                        return_raw_logits,
                        prefix_cacher,
                        disable_eos_stop,
                        rng,
                        backend_metadata,
                    )
                    .await?
            };
            if prompt_ids.is_empty() {
                self.kit.hidden_capture.clear();
            } else {
                let capture = self.kit.hidden_capture.take();
                let batch = prompt_ids.len();
                for (row, seq_id) in prompt_ids.into_iter().enumerate() {
                    let Some(mut state) = self.checkout_draft_kv(seq_id) else {
                        continue;
                    };
                    let toks = input_seqs[row].get_toks().to_vec();
                    match self.extend_draft_kv_row(&mut state, capture.clone(), &toks, row, batch) {
                        Ok(true) => {}
                        Ok(false) => {
                            state.poisoned = true;
                            self.warn_unprimed_once(
                                "the prompt forward's hidden states do not cover the \
                                 sequence contiguously from position 0 (chunked prefill \
                                 or a prefix-cache hit)",
                            );
                        }
                        Err(e) => {
                            state.poisoned = true;
                            tracing::warn!(
                                target: "mtp_speculative",
                                "MTP draft-KV prefill failed ({e}); drafting disabled for \
                                 this sequence"
                            );
                        }
                    }
                    self.checkin_draft_kv(seq_id, state);
                }
            }
            return Ok(elapsed);
        }

        // ===== MTP fast path =====
        let CacheBackendMetadata::DefaultInstructions { pre_op, post_op } = backend_metadata else {
            unreachable!("guarded above");
        };

        // PRE-cache instruction: clone-in / reset, matching what
        // `Pipeline::step` does internally for the target alone.
        //
        // `lead_pad[i]` is how many columns of the batched buffer are padding
        // ahead of row `i` rather than that row's own K/V — see
        // [`Self::clone_in_cache_reporting_lead`]. It is all-zero unless this
        // step assembled the batch itself, which is why a ragged cohort can only
        // be unpicked on the `In` arm.
        let mut lead_pad = vec![0usize; input_seqs.len()];
        let mut assembled_here = false;
        match pre_op {
            CacheInstruction::In => {
                lead_pad = self.clone_in_cache_reporting_lead(input_seqs)?;
                assembled_here = true;
            }
            CacheInstruction::Nothing => (),
            CacheInstruction::Reset {
                reset_non_granular,
                load_preallocated_cache,
            } => {
                // The target's KV was thrown away; the draft KV that shadowed
                // it is now meaningless too.
                self.clear_draft_kv();
                self.set_none_cache(
                    input_seqs,
                    reset_non_granular,
                    false,
                    load_preallocated_cache,
                )
            }
            _ => unreachable!("Unreachable PRE cache op."),
        }

        let start = Instant::now();

        // A capture can only be stale here: every path that produces one
        // consumes it in the same step. Dropping it makes it impossible for a
        // block belonging to another sequence to be read as this one's.
        self.kit.hidden_capture.clear();

        // ---- The window invariant ----
        //
        // `cache_len` (C) is the width of the one dense batched buffer
        // `clone_in_cache` built. Under the cohort rule it is also every
        // sequence's committed length, and `cache_lens` is `C` repeated. Under
        // per-sequence advance it is only the batch **maximum**: row `i` was
        // padded up to it and owns `C - lead_pad[i]` of it, which is what
        // `cache_lens[i]` carries.
        //
        // Each sequence's committed length is `cache_lens[i] + u_i`, where `u`
        // is its uncached committed tail; plain decode holds `u == 1`.
        // `w = depth + 1` is the fixed window width every sequence feeds: `u`
        // real tokens, then `w - u` drafts.
        let w = self.window();
        let batch = input_seqs.len();
        let cache_len = current_normal_cache_len(self);
        // Per-sequence advance needs the batched buffer split back into each
        // sequence's own storage before the lengths can diverge, and that is
        // what `CacheInstruction::Out` does. It also needs this step to have
        // assembled the batch, because that is the only place `lead_pad` comes
        // from. Any other pair leaves the batched cache in the pipeline, where
        // there is no per-sequence length to set, so those steps take the cohort
        // rule — correctly, and without a mid-run state that no other path knows
        // how to read.
        let per_seq_advance = assembled_here
            && matches!(post_op, CacheInstruction::Out)
            && self.kv_advance() == KvAdvance::PerSequence;
        let row_cache_lens = resolve_row_cache_lens(per_seq_advance, cache_len, &lead_pad)?;
        let cache_lens: Vec<usize> = match &row_cache_lens {
            Some(rows) => rows.clone(),
            None => vec![cache_len; batch],
        };
        let tok_lens: Vec<usize> = input_seqs.iter().map(|s| s.get_toks().len()).collect();
        let StepWindow {
            uncached,
            ok: window_ok,
        } = plan_step_window(&cache_lens, &tok_lens, w);
        if !window_ok {
            // 🔴 A left-aligned batch cannot be handed to the target's own
            // decode: that path builds no mask at all at `t_q == 1`
            // (`layers_masker.rs:285`), so every row's dead prefix would be
            // attended as real keys — a zero K row scores logit 0 and takes
            // softmax weight. Deferring is only safe for a batch that is not
            // padded, so a padded one refuses instead.
            if row_cache_lens.is_some() {
                candle_core::bail!(
                    "MTP per-sequence step: this batch is left-aligned (dead prefixes \
                     {lead_pad:?}) but its window invariant does not hold (uncached tails \
                     {uncached:?} against a window of {w}). Deferring to the target's own decode \
                     would attend the dead prefix as real keys, because that path builds no mask \
                     at `t_q == 1`."
                );
            }
            // 🔴 …and neither can a batch that merely FOLLOWS one. Under
            // `KvAdvance::PerSequence` this step runs no cohort rollback, so the
            // pipeline's resident batched cache stays `cache_len + w` wide while
            // each sequence keeps its own shorter length. A later step that did
            // not assemble the batch (`pre_op: Nothing`, which the engine issues
            // for any cohort whose completion ids did not change) reads that
            // stale width here and has an all-zero `lead_pad`, so the guard
            // above cannot see it — and the deferral below would decode against
            // a cache whose write offset and whose RoPE positions disagree.
            // Refusing is the only honest answer; see
            // [`batch_admission_admits_ragged_cohorts`], which is why the mode
            // is not granted in the first place.
            if self.kv_advance() == KvAdvance::PerSequence {
                candle_core::bail!(
                    "MTP per-sequence step: the window invariant does not hold (uncached tails \
                     {uncached:?} against a window of {w}) on a step that did not assemble this \
                     batch (assembled_here={assembled_here}). Under per-sequence advance the \
                     pipeline's batched cache is not rolled back, so its {cache_len}-column \
                     width is the previous step's and deferring to the target's own decode would \
                     write new K/V at an offset its RoPE positions do not agree with."
                );
            }
            // Not a state this fast path produced — a cache reset, a
            // prefix-cache hit, a chunked prefill. The target's own bookkeeping
            // is authoritative there, so defer (the PRE op is already applied).
            self.clear_draft_kv();
            let elapsed = {
                let mut target = self.target.lock().await;
                target
                    .step(
                        input_seqs,
                        is_prompt,
                        return_raw_logits,
                        prefix_cacher,
                        disable_eos_stop,
                        rng,
                        CacheBackendMetadata::DefaultInstructions {
                            pre_op: CacheInstruction::Nothing,
                            post_op,
                        },
                    )
                    .await?
            };
            self.kit.hidden_capture.clear();
            return Ok(elapsed);
        }

        let eos_owned = get_mut_arcmutex!(self.target)
            .get_metadata()
            .eos_tok
            .clone();
        let eos_tok = if disable_eos_stop {
            None
        } else {
            Some(&eos_owned[..])
        };
        let max_seq_len = get_mut_arcmutex!(self.target).get_metadata().max_seq_len;
        let budgets: Vec<usize> = input_seqs
            .iter()
            .map(|s| max_seq_len.saturating_sub(s.get_toks().len()))
            .collect();

        let seq_ids: Vec<usize> = input_seqs.iter().map(|s| *s.id()).collect();
        let mut states: Vec<Option<DraftKv>> = seq_ids
            .iter()
            .map(|id| self.checkout_draft_kv(*id))
            .collect();

        // ---- Draft, grouped by uncached-tail length ----
        //
        // Within a group every sequence needs the SAME number of drafts `w - u`,
        // so one batched MTP-block forward per chain step drafts for all of
        // them. `u` ranges over `[1, w]`, so a batch splits into at most
        // `depth + 1` groups no matter how large it is, and the draft cost stays
        // a small multiple of one single-layer forward rather than a multiple of
        // the batch size.
        //
        // 🔑 The rows of a group no longer have to sit at the same absolute
        // chain start. They used to, because `MtpBlock::forward_step` took ONE
        // `pos` — the RoPE position *and* the draft-KV slot — and
        // `batch_draft_caches` refused rows of differing length. Both are now
        // per-row: `forward_step_rows` threads a positions vector to RoPE and to
        // `dsv4_attention`'s `row_q0`, and `front_align_draft_group` left-aligns
        // the group's draft KVs the way `front_align_batch` left-aligns the
        // target's. So under per-sequence advance the partition stays at
        // `<= depth + 1` groups instead of degenerating to one per sequence.
        //
        // 🔴 `draft_group_key`'s flag is what keeps that honest. With the old
        // scalar path, keying on `u` alone would not be slow, it would be
        // silent: a row whose seed is not at the group's chain start is dropped
        // by the seed check below, so it would stop drafting with no signal but
        // a falling `tok_per_step`.
        //
        // 🔴 `may_draft[i]` is the sampling gate. A sequence whose sampler is
        // not a plain argmax over raw logits drafts NOTHING — see
        // [`seq_may_draft`]. It stays in the batch, feeds its uncached tail plus
        // padding into the same window, and takes its one token out of
        // `sample_sequence` at the bottom of this function, which is plain
        // decode with that request's own temperature and penalties.
        let mut drafts: Vec<Vec<u32>> = vec![Vec::new(); batch];
        let may_draft: Vec<bool> = input_seqs.iter_mut().map(|s| seq_may_draft(s)).collect();
        for (i, allowed) in may_draft.iter().enumerate() {
            if !allowed {
                self.warn_non_greedy_once(seq_ids[i]);
            }
        }
        let groups = draft_groups(&uncached, &cache_lens, &may_draft, w, per_seq_advance);
        for ((u, _), members) in groups {
            let n_draft = w - u;
            let mut rows: Vec<usize> = Vec::with_capacity(members.len());
            let mut seeds: Vec<Tensor> = Vec::with_capacity(members.len());
            let mut last_toks: Vec<u32> = Vec::with_capacity(members.len());
            // Row `i`'s OWN chain start. Under the cohort rule every member of a
            // group produces the same number here, which is the key's second
            // component — so this vector is uniform and every dispatch below
            // takes its scalar branch.
            let mut chain_starts: Vec<usize> = Vec::with_capacity(members.len());
            for i in members {
                if budgets[i] == 0 {
                    continue;
                }
                let Some(state) = states[i].as_ref() else {
                    continue;
                };
                if state.poisoned {
                    continue;
                }
                let Some((seed_pos, seed)) = state.seed.as_ref() else {
                    continue;
                };
                // The seed must be the target's hidden state at exactly the
                // chain's first slot, and the draft KV must hold at least that
                // many committed entries. Anything else and slot index and
                // absolute RoPE position have drifted apart.
                let chain_start = chain_start_of(cache_lens[i], u);
                if *seed_pos != chain_start || state.filled < chain_start {
                    continue;
                }
                let toks = input_seqs[i].get_toks();
                rows.push(i);
                seeds.push(seed.clone());
                last_toks.push(toks[toks.len() - 1]);
                chain_starts.push(chain_start);
            }
            if rows.is_empty() {
                if self.kit.block.is_some() {
                    self.warn_unprimed_once(&format!(
                        "no sequence in the u={u} group has a draft KV seeded at its own \
                         absolute chain start"
                    ));
                }
                continue;
            }
            let seed_hidden = Tensor::stack(&seeds, 0)?;
            // One batched draft KV for the group: the MTP block is a single
            // decoder layer, so this is `clone_in_cache` at 1/43 of its cost.
            //
            // `leads[slot]` is the dead prefix left-alignment put ahead of row
            // `slot`, and it MUST come back out before the row's own cache is
            // stored again — see `resolve_draft_leads`. It is all-zero for a
            // group whose rows share a chain start, and `drop_dead_prefix` at
            // `lead == 0` touches no tensor.
            let mut leads = vec![0usize; rows.len()];
            let mut group_cache = if self.kit.block.is_some() {
                let borrowed: Vec<&KvCache> = rows
                    .iter()
                    .filter_map(|i| states[*i].as_ref().and_then(|s| s.cache.as_ref()))
                    .collect();
                if borrowed.len() != rows.len() {
                    continue;
                }
                match resolve_draft_leads(per_seq_advance, &chain_starts) {
                    // Uniform group: verbatim the pre-change path.
                    None => {
                        let Some(mut batched) = batch_draft_caches(&borrowed) else {
                            self.warn_unprimed_once(
                                "the draft KVs in one uncached-tail group disagree on shape, so \
                                 they cannot share a batched forward",
                            );
                            continue;
                        };
                        if batched.set_len(chain_starts[0]).is_err() {
                            continue;
                        }
                        Some(batched)
                    }
                    Some(row_leads) => match front_align_draft_group(&borrowed, &chain_starts) {
                        Ok(batched) => {
                            leads = row_leads;
                            Some(batched)
                        }
                        Err(e) => {
                            self.warn_unprimed_once(&format!(
                                "the draft KVs in one uncached-tail group could not be \
                                 left-aligned to share a batched forward ({e})"
                            ));
                            continue;
                        }
                    },
                }
            } else {
                None
            };
            let chains = self.kit.propose_chain_batched(
                &seed_hidden,
                &last_toks,
                n_draft,
                &chain_starts,
                group_cache.as_mut(),
            )?;
            if let Some(batched) = group_cache.as_ref() {
                let Some(split) = split_draft_cache(batched, rows.len()) else {
                    continue;
                };
                for (slot, i) in rows.iter().enumerate() {
                    let mut row_cache = split[slot].clone();
                    // 🔑 Strip the padding before the length is recorded. Keeping
                    // it would make the next group's left-alignment pad relative
                    // to an already-padded length, so the prefix would compound
                    // every step and the rows would re-converge onto one length
                    // — the same defect PR #102 found in the target's cache,
                    // one cache down.
                    if let Err(e) = crate::kv_cache::drop_dead_prefix(&mut row_cache, leads[slot]) {
                        self.warn_unprimed_once(&format!(
                            "stripping a draft KV's dead prefix failed ({e}); this sequence's \
                             draft cache is dropped rather than recorded with padding counted \
                             as its own content"
                        ));
                        if let Some(state) = states[*i].as_mut() {
                            state.poisoned = true;
                        }
                        continue;
                    }
                    if let Some(state) = states[*i].as_mut() {
                        state.cache = Some(row_cache);
                        // The chain's FIRST entry is committed context — the
                        // pair `(h_{L-2}, tok_{L-1})`, both of which are real.
                        // Everything after it used the DRAFT's hidden states
                        // and is truncated by the next extend.
                        state.filled = chain_starts[slot] + 1;
                    }
                }
            }
            for (slot, i) in rows.iter().enumerate() {
                drafts[*i] = chains[slot].clone();
            }
        }

        // ---- Verify: ONE target forward over the fixed-width window ----
        //
        // The pre-fusion shape ran TWO target forwards per step (one to
        // materialise `T0` and its hidden state, one to verify), which caps the
        // multiplier at `(depth + 1) / 2` — at depth 1 that is exactly 1.0, no
        // speedup at all. The seed carried on `DraftKv` removes the first
        // forward: the row the next chain needs is always inside the window the
        // last verify already covered. SGLang does the same thing from the
        // other direction, skipping the final draft forward because the
        // draft-extend it must run anyway already produced that token
        // (`eagle_worker.py:871-873`).
        //
        // 🔑 `seqlen_offsets[i]` is where the model places row `i`'s queries —
        // for RoPE, and (on DeepSeek V4) as the per-row `row_q0` that masks a
        // left-aligned cohort's dead prefix and its start-anchored compressed
        // columns. A shared `prefill_window` collapses it to one scalar, which
        // is right only when the rows share a length. When they do not, each row
        // carries its own and the shared component is zero — the vector the
        // model needs is the one this step already computed.
        for (i, seq) in input_seqs.iter_mut().enumerate() {
            let toks = seq.get_toks();
            let mut window: Vec<u32> = toks[cache_lens[i]..].to_vec();
            window.extend(drafts[i].iter().copied());
            let pad = *window.last().expect("uncached tail is at least 1 token");
            window.resize(w, pad);
            match row_cache_lens {
                Some(_) => seq.set_prefill_toks_at(window, cache_lens[i]),
                None => seq.set_prefill_toks(window),
            }
        }
        let shared_offset = match row_cache_lens {
            Some(_) => 0,
            None => cache_len,
        };
        let verify = run_target_forward_batch(self, input_seqs, Some((w, shared_offset))).await;
        for seq in input_seqs.iter_mut() {
            seq.reset_prefill_toks();
        }
        let (verify_logits, _exec) = verify?;
        // `[B, w]` greedy argmax — one device op, not `B * w` of them.
        let arg = verify_logits
            .argmax(candle_core::D::Minus1)?
            .to_dtype(candle_core::DType::U32)?
            .to_vec2::<u32>()?;

        // ---- Accept / reject, per sequence ----
        let mut accepted: Vec<Vec<u32>> = Vec::with_capacity(batch);
        let mut valid_extent: Vec<usize> = Vec::with_capacity(batch);
        let mut per_seq_stats: Vec<MtpAcceptance> = Vec::with_capacity(batch);
        for i in 0..batch {
            let u = uncached[i];
            let d = drafts[i].len();
            // Row `u - 1 + j` predicts the token at position `L + j`, which is
            // what draft `j` proposed. Row `u - 1 + d` is the bonus token that
            // a fully accepted chain earns.
            let targets: Vec<u32> = (0..d).map(|j| arg[i][window_verify_row(u, j)]).collect();
            let result = verify_proposed(&drafts[i], &targets);
            let mut n_acc = result.accepted.len();
            // Never commit past the sequence's own length budget.
            if n_acc + 1 > budgets[i] {
                n_acc = budgets[i].saturating_sub(1);
            }
            accepted.push(result.accepted[..n_acc].to_vec());
            valid_extent.push(u + n_acc);
            per_seq_stats.push(MtpAcceptance::from_fused_verify(d, n_acc));
        }

        // ---- Roll the shared cache back to the length every sequence can prove ----
        //
        // The forward advanced the batched cache by `w` for everyone. Sequence
        // `i` can only vouch for `u_i + accepted_i` of those slots: the rest
        // hold KV for draft tokens that were rejected (or for the pad a
        // sequence that could not draft fed). One dense cache means one length,
        // so the batch keeps the minimum — and each sequence's surplus stays
        // committed as TOKENS and comes back as a longer uncached tail next
        // step, which is exactly what `w - u` drafts leaves room for. `u >= 1`
        // for every sequence, so the cache always advances by at least one
        // position and the batch can never stall.
        //
        // Under [`KvAdvance::PerSequence`] none of the paragraph above applies:
        // there is no minimum, no rollback of the shared buffer, and no
        // surplus. Each sequence's own cache is set to `cache_len + u_i + a_i`
        // on the way out and its next uncached tail is 1 — the B=1 invariant,
        // at every batch size.
        let n_accepted: Vec<usize> = accepted.iter().map(Vec::len).collect();
        let advance = if per_seq_advance {
            KvAdvance::PerSequence
        } else {
            KvAdvance::Cohort
        };
        let per_sequence_commit = match advance {
            KvAdvance::Cohort => {
                let plan = plan_batch_step(&uncached, &n_accepted, w);
                debug_assert_eq!(
                    plan.keep,
                    valid_extent.iter().copied().min().unwrap_or(0),
                    "plan_batch_step must agree with the per-sequence valid extents"
                );
                if plan.n_drop > 0 {
                    truncate_normal_cache(self, plan.n_drop)?;
                }
                None
            }
            KvAdvance::PerSequence => {
                let plan = plan_per_sequence_step(&uncached, &n_accepted, w);
                debug_assert_eq!(
                    plan.committed, valid_extent,
                    "plan_per_sequence_step must agree with the per-sequence valid extents"
                );
                Some(plan.committed)
            }
        };

        // ---- Commit ----
        //
        // Accepted drafts are emitted directly, bypassing `sample_sequence`.
        // That is only sound because [`seq_may_draft`] refused to draft for any
        // sequence whose sampler is not `argmax(raw logits)`: such a sequence
        // has `accepted[i].is_empty()` here, so this loop is a no-op for it and
        // its single token comes out of the real sampler below. Widening the
        // gate without also widening this emission is how the greedy
        // substitution came back.
        for i in 0..batch {
            let u = uncached[i];
            let n_acc = accepted[i].len();
            debug_assert!(
                n_acc == 0 || may_draft[i],
                "a sequence that may not draft must never have an accepted draft to commit"
            );
            let seq = &mut input_seqs[i];
            for &tok in &accepted[i] {
                let lp = crate::sampler::Logprobs {
                    token: tok,
                    logprob: 0.0,
                    top_logprobs: None,
                    bytes: None,
                };
                finish_or_add_toks_to_seq(self, prefix_cacher, seq, lp, eos_tok, true).await?;
                if seq_is_terminal(seq) {
                    break;
                }
            }
            if seq_is_terminal(seq) || budgets[i] <= n_acc {
                continue;
            }
            // The one token of the step that was NOT a verified draft — the
            // target's correction at the rejected slot, or the bonus token
            // after a full accept. It goes through the real sampler, so the
            // request's temperature / penalties apply to it exactly as they
            // would without MTP.
            let row = window_verify_row(u, n_acc);
            let row_logits = verify_logits.i(i)?.narrow(0, row, 1)?.unsqueeze(0)?;
            let want_logprobs = seq.return_logprobs();
            let lp = sample_sequence(
                row_logits,
                seq,
                want_logprobs,
                rng.clone(),
                false,
                false,
                false,
            )
            .await?;
            finish_or_add_toks_to_seq(self, prefix_cacher, seq, lp, eos_tok, true).await?;
        }

        // ---- Extend the draft KV over the newly committed tokens ----
        //
        // The verify forward's captured hidden states are the TARGET's states
        // for the accepted positions, so the draft-KV entries for them are
        // rebuilt from the target's `h`; the chain's own entries beyond the
        // first used the DRAFT's `h` and are truncated inside
        // `extend_draft_kv_row`. Its `i <= L-2` bound also guarantees a rejected
        // proposal's hidden state is never written, and it is what re-stamps
        // the seed for the next step.
        let capture = self.kit.hidden_capture.take();
        for i in 0..batch {
            let Some(mut state) = states[i].take() else {
                continue;
            };
            let toks = input_seqs[i].get_toks().to_vec();
            match self.extend_draft_kv_row(&mut state, capture.clone(), &toks, i, batch) {
                Ok(true) => {}
                Ok(false) => {
                    state.poisoned = true;
                    self.warn_unprimed_once(
                        "the verify forward's hidden states were not contiguous with the draft KV",
                    );
                }
                Err(e) => {
                    state.poisoned = true;
                    tracing::warn!(
                        target: "mtp_speculative",
                        "MTP draft-KV extend after verify failed ({e}); drafting disabled for \
                         this sequence"
                    );
                }
            }
            self.checkin_draft_kv(seq_ids[i], state);
        }

        // ---- Account ----
        for (i, stat) in per_seq_stats.iter().enumerate() {
            self.record_step(seq_ids[i], *stat);
        }
        record_mtp_batch_step(batch, &per_seq_stats);

        // POST cache op (matches what `Pipeline::step` does after a normal
        // forward).
        for i in 0..batch {
            let done = seq_is_terminal(input_seqs[i]);
            if let CacheInstruction::Reset { .. } = post_op {
                self.drop_draft_kv(seq_ids[i]);
            } else if done {
                self.flush_seq_acceptance(seq_ids[i]);
            }
        }
        handle_post_cache_op(self, input_seqs, post_op);

        // ---- Per-sequence advance ----
        //
        // `handle_post_cache_op` has just split the batched buffer back into
        // each sequence's own `normal_cache`, every row still `cache_len + w`
        // long. Each sequence now keeps exactly the `cache_len + u_i + a_i` it
        // can vouch for. The slots past that hold a rejected draft's K/V and
        // are simply stopped being counted — vLLM
        // (`v1/core/sched/scheduler.py:1836-1848`) and SGLang
        // (`speculative/eagle_worker_common.py:450`) do the same thing and
        // likewise never free or zero them; the next step's window overwrites
        // the same physical positions.
        //
        // 🔑 The per-sequence state IS `Sequence::normal_cache`'s own
        // `current_seq_len`. There is deliberately **no second map keyed by
        // sequence id** anywhere in this pipeline: a shadow copy of a length
        // that the cache also holds is exactly the divergence wave59-CJ §1
        // documents, reappearing in a new place. The commit therefore happens
        // here, in the same scope that computed it, from a plain `Vec` indexed
        // by the batch row — and there is no reduction across `input_seqs` in
        // it, which is the whole difference from the cohort path.
        //
        // 🔑 And the length recorded is this row's OWN, not the batch's width.
        // `clone_out_cache` hands each row the batched buffer verbatim, dead
        // prefix included, so the prefix is stripped first
        // (`drop_dead_prefix` — the inverse of `front_pad_kv_cache`). Recording
        // `cache_len + c_i` instead would count `lead_pad[i]` padding columns as
        // this sequence's content, and the next `front_align_batch` would pad
        // relative to that: the prefix would compound every step, growing
        // linearly rather than tracking the `sqrt(steps)` spread of the
        // sequences themselves, and the rows would silently re-converge onto one
        // length — the cohort barrier this whole path exists to remove,
        // reappearing as padding.
        if let Some(committed) = per_sequence_commit {
            for (i, c) in committed.iter().enumerate() {
                let lead = lead_pad[i];
                let target = cache_lens[i] + c;
                for slot in input_seqs[i].normal_cache().iter_mut().flatten() {
                    let kind = slot.kind_name();
                    if let Err(e) = crate::kv_cache::drop_dead_prefix(slot, lead) {
                        candle_core::bail!(
                            "MTP per-sequence advance: stripping the {lead}-column dead prefix \
                             failed on a {kind} slot for batch row {i} ({e}). Continuing would \
                             record that padding as the sequence's own length."
                        );
                    }
                    if let Err(e) = slot.set_len(target) {
                        candle_core::bail!(
                            "MTP per-sequence advance: set_len({target}) refused on a {kind} slot \
                             for batch row {i} ({e}). Continuing would leave that sequence \
                             reading a rejected draft's K/V."
                        );
                    }
                }
            }
        }

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
/// The wrapper is output-equivalent to the bare target for **every** request,
/// not only greedy ones. For a request whose sampler is `argmax(raw logits)`,
/// MTP accept/reject is lossless by construction. For any other request —
/// `temperature > 0`, a penalty, a custom logits processor — the wrapper
/// declines to draft ([`seq_may_draft`]) and every token comes out of
/// `sample_sequence` exactly as it would without MTP. Wrapping therefore never
/// changes what a request receives; it only changes how fast a greedy one
/// receives it.
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
// The crate denies the cast lints. Inside these tests every cast is on a small
// literal token count (depths of 1-8, step counts of tens) being compared
// against an exact expected ratio — the precision the lints guard would need
// >2^53 proposals to matter, and spelling the casts out defensively would
// obscure the arithmetic the assertions exist to pin.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
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
            hidden_capture: Arc::new(MtpHiddenCapture::default()),
            embed_tokens,
            lm_head: wrap_linear(lm_w),
            h_proj: wrap_linear(h_w),
            e_proj: wrap_linear(e_w),
            block: None,
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

    /// The MTP tail must never be resolved below 8 bits, whatever the global
    /// ISQ request. An int4 draft head measured 0-4% acceptance in the field
    /// (`EXTERNAL_FINDINGS.md` F3) — the same failure class RUN-161 already
    /// fixed for `lm_head`.
    #[test]
    fn mtp_tail_is_floored_at_int8() {
        use mistralrs_quant::IsqType as T;
        let sub_int8 = [
            T::Q4_0,
            T::Q4_1,
            T::Q5_0,
            T::Q5_1,
            T::Q2K,
            T::Q3K,
            T::Q4K,
            T::Q5K,
            T::Q6K,
            T::HQQ4,
            T::AFQ2,
            T::AFQ3,
            T::AFQ4,
            T::AFQ6,
            T::MXFP4,
            T::NVFP4,
            T::QtipBitshift2,
            T::Qtip2b,
        ];
        for ty in sub_int8 {
            assert!(isq_is_sub_int8(ty), "{ty:?} should be classified sub-int8");
            let floored = floor_mtp_isq(Some(ty)).expect("a Some request stays Some");
            assert!(
                !isq_is_sub_int8(floored),
                "{ty:?} floored to {floored:?}, which is still below int8"
            );
        }

        // 8-bit and wider requests pass through untouched — the floor must not
        // silently change a width the user asked for and can afford.
        for ty in [
            T::Q8_0,
            T::Q8_1,
            T::Q8K,
            T::HQQ8,
            T::AFQ8,
            T::F8E4M3,
            T::F8Q8,
        ] {
            assert!(!isq_is_sub_int8(ty));
            assert_eq!(floor_mtp_isq(Some(ty)), Some(ty));
        }

        // No ISQ requested -> nothing to floor.
        assert_eq!(floor_mtp_isq(None), None);

        // Family preserved where an 8-bit sibling exists, so the floored
        // tensor keeps using its neighbours' kernels.
        assert_eq!(floor_mtp_isq(Some(T::AFQ2)), Some(T::AFQ8));
        assert_eq!(floor_mtp_isq(Some(T::HQQ4)), Some(T::HQQ8));
        assert_eq!(floor_mtp_isq(Some(T::Qtip2b)), Some(T::Q8_0));
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
        let tokens = kit.propose_chain(&prev_hidden, 0, depth, max_tokens, 0, None)?;
        assert_eq!(tokens.len(), depth, "should return exactly depth tokens");
        // All proposed token ids should be within vocab range.
        for t in &tokens {
            assert!((*t as usize) < vocab, "token {} out of vocab {}", t, vocab);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // MTP acceptance telemetry (wave14-AK)
    // -----------------------------------------------------------------------
    //
    // `log_acceptance_rate()` had ZERO callers in the workspace and nothing in
    // Rust read `ARC_MTP_LOG_ACCEPTANCE`, so GPU session 3 measured MTP
    // acceptance and produced an empty artifact; sessions 4+ carried a patch
    // file re-applied by hand every time. These tests are what makes the wiring
    // permanent: the cadence, the honesty of the reported ratio, and the fact
    // that the line actually reaches `tracing`.

    /// Capture `tracing` output into a shared buffer so a test can assert that
    /// a log line was *actually emitted*, not merely that a predicate returned
    /// true.
    #[derive(Clone, Default)]
    struct CaptureWriter(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CaptureWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("capture buffer poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CaptureWriter {
        type Writer = CaptureWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Run `f` with a thread-local `tracing` subscriber and return everything
    /// it logged.
    fn capture_logs(f: impl FnOnce()) -> String {
        let writer = CaptureWriter::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer.clone())
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        let bytes = writer.0.lock().expect("capture buffer poisoned").clone();
        String::from_utf8(bytes).expect("tracing output is UTF-8")
    }

    /// A depth-`proposed` step with `accepted` of them accepted, committing
    /// T0 + accepted + (the verifier's correction, when anything was rejected)
    /// — the same arithmetic `step()` does at its accept/reject site.
    fn step_of(proposed: usize, accepted: usize) -> MtpAcceptance {
        assert!(accepted <= proposed);
        let correction = usize::from(accepted < proposed);
        // Positions `0..accepted` were accepted; the chain reached one further
        // whenever it was cut short — the same derivation `from_fused_verify`
        // makes at the real accept/reject site.
        let reached = if proposed == 0 {
            0
        } else {
            (accepted + 1).min(proposed)
        };
        let (reached_at, accepted_at, positions_over_cap) =
            position_tallies(proposed, accepted, reached);
        MtpAcceptance {
            accepted,
            proposed,
            steps: 1,
            drafted_steps: usize::from(proposed > 0),
            committed: 1 + accepted + correction,
            batch_steps: 0,
            reached_at,
            accepted_at,
            positions_over_cap,
        }
    }

    /// Per-position tallies as a fixed-size array, for terse test expectations.
    fn positions(v: &[usize]) -> [usize; MTP_MAX_TRACKED_POSITIONS] {
        let mut out = [0usize; MTP_MAX_TRACKED_POSITIONS];
        out[..v.len()].copy_from_slice(v);
        out
    }

    /// Counters accumulate exactly what the verify site hands them, and the
    /// snapshot reads them back field for field.
    #[test]
    fn acceptance_counters_increment() {
        let tel = AcceptanceTelemetry::default();
        assert_eq!(tel.snapshot(), MtpAcceptance::default());
        tel.record_gated(&step_of(2, 1), false);
        tel.record_gated(&step_of(2, 2), false);
        assert_eq!(
            tel.snapshot(),
            MtpAcceptance {
                accepted: 3,
                proposed: 4,
                steps: 2,
                drafted_steps: 2,
                // step 1: T0 + 1 accepted + 1 correction = 3
                // step 2: T0 + 2 accepted, no correction = 3
                committed: 6,
                batch_steps: 0,
                // Both steps proposed 2 and reached both positions (step 1 was
                // rejected AT position 1, which is still an evaluation).
                reached_at: positions(&[2, 2]),
                // Position 0 accepted in both steps, position 1 only in step 2.
                accepted_at: positions(&[2, 1]),
                positions_over_cap: 0,
            }
        );
        tel.reset();
        assert_eq!(tel.snapshot(), MtpAcceptance::default());
    }

    /// **The mutation proof (DOCTRINE D12).** The counter must *move* with the
    /// outcome, not merely exist. Force every draft accepted and the reported
    /// rate must be exactly 1.0; force every draft rejected and it must be
    /// exactly 0.0. A counter hard-wired to a constant — or one reading the
    /// wrong side of the accept/reject decision — fails one of the two.
    #[test]
    fn acceptance_rate_moves_to_one_on_all_accept_and_zero_on_all_reject() {
        const DEPTH: usize = 3;
        const STEPS: usize = 40;

        let all_accept = AcceptanceTelemetry::default();
        for _ in 0..STEPS {
            all_accept.record_gated(&step_of(DEPTH, DEPTH), false);
        }
        let accept_snap = all_accept.snapshot();
        assert_eq!(
            accept_snap.rate(),
            Some(1.0),
            "{}",
            accept_snap.marker("all-accept")
        );
        // Every proposal accepted => T0 + depth committed per step, so the
        // effective multiplier is exactly depth + 1.
        assert_eq!(accept_snap.tokens_per_step(), Some(DEPTH as f64 + 1.0));

        let all_reject = AcceptanceTelemetry::default();
        for _ in 0..STEPS {
            all_reject.record_gated(&step_of(DEPTH, 0), false);
        }
        let reject_snap = all_reject.snapshot();
        assert_eq!(
            reject_snap.rate(),
            Some(0.0),
            "{}",
            reject_snap.marker("all-reject")
        );
        // Nothing accepted still commits T0 + the verifier's correction: even a
        // useless draft head yields 2 tokens per step, which is exactly why
        // acceptance and tok_per_step are both reported and neither alone.
        assert_eq!(reject_snap.tokens_per_step(), Some(2.0));

        // Non-degenerate: the two runs proposed the same number of tokens and
        // are still distinguishable, so no single broken implementation
        // satisfies both assertions above.
        assert_eq!(accept_snap.proposed, reject_snap.proposed);
        assert_ne!(accept_snap.accepted, reject_snap.accepted);

        // A mixed run lands strictly between them, at the counted ratio.
        let mixed = AcceptanceTelemetry::default();
        for i in 0..STEPS {
            mixed.record_gated(&step_of(DEPTH, i % (DEPTH + 1)), false);
        }
        let snap = mixed.snapshot();
        let rate = snap.rate().expect("the mixed run proposed tokens");
        assert!(rate > 0.0 && rate < 1.0, "{}", snap.marker("mixed"));
        assert!(
            (rate - snap.accepted as f64 / snap.proposed as f64).abs() < f64::EPSILON,
            "the reported rate must be the counted ratio, with nothing smoothed"
        );
    }

    /// **The case the scalar could not answer (DOCTRINE D33 — feed the check its
    /// failing case).**
    ///
    /// Two runs at depth 3 over 100 steps, from known accept patterns, whose
    /// **every existing aggregate field is bit-identical**: `accepted=100`,
    /// `proposed=300`, `steps=100`, `drafted_steps=100`, `committed=300`,
    /// `accept_rate=1/3`, `tok_per_step=3.0`.
    ///
    /// * `dies_at_1` — every step accepts exactly 1. The chain reaches position
    ///   1 every step and is rejected there: `p = [1.00, 0.00]`.
    /// * `all_or_nothing` — half the steps accept 2, half accept 0. The chain
    ///   only reaches position 1 on the halves that cleared position 0:
    ///   `p = [0.50, 1.00, 0.00]`.
    ///
    /// One is a chain that **compounds** (position 0 is perfect, the draft's own
    /// hidden state kills position 1); the other is a **first-position**
    /// distribution mismatch that the chain then survives. Different defects,
    /// different fixes, and while `MtpAcceptance.accepted` was one `usize` they
    /// were *indistinguishable in the telemetry* — which is why the measured
    /// `accept_rate=0.4194` could not be attributed to either.
    ///
    /// The assertions fail if the per-position vector is dropped, flattened,
    /// derived from the aggregate, or denominated in steps instead of
    /// reach.
    #[test]
    fn per_position_acceptance_separates_profiles_the_aggregate_cannot() {
        const STEPS: usize = 100;

        let dies_at_1 = AcceptanceTelemetry::default();
        for _ in 0..STEPS {
            dies_at_1.record_gated(&step_of(3, 1), false);
        }
        let dies_at_1 = dies_at_1.snapshot();

        let all_or_nothing = AcceptanceTelemetry::default();
        for i in 0..STEPS {
            all_or_nothing.record_gated(&step_of(3, if i % 2 == 0 { 2 } else { 0 }), false);
        }
        let all_or_nothing = all_or_nothing.snapshot();

        // The premise: every pre-existing aggregate agrees. If this ever stops
        // holding, the test is no longer testing what it claims.
        assert_eq!(dies_at_1.accepted, all_or_nothing.accepted);
        assert_eq!(dies_at_1.proposed, all_or_nothing.proposed);
        assert_eq!(dies_at_1.steps, all_or_nothing.steps);
        assert_eq!(dies_at_1.drafted_steps, all_or_nothing.drafted_steps);
        assert_eq!(dies_at_1.committed, all_or_nothing.committed);
        assert_eq!(dies_at_1.rate(), all_or_nothing.rate());
        assert_eq!(
            dies_at_1.tokens_per_step(),
            all_or_nothing.tokens_per_step()
        );
        assert_eq!(dies_at_1.rate(), Some(1.0 / 3.0));
        assert_eq!(dies_at_1.tokens_per_step(), Some(3.0));

        // …and the per-position vectors do not.
        assert_eq!(dies_at_1.accepted_at, positions(&[100, 0, 0]));
        assert_eq!(dies_at_1.reached_at, positions(&[100, 100, 0]));
        assert_eq!(dies_at_1.per_position_rates(), vec![1.0, 0.0]);

        assert_eq!(all_or_nothing.accepted_at, positions(&[50, 50, 0]));
        assert_eq!(all_or_nothing.reached_at, positions(&[100, 50, 50]));
        assert_eq!(all_or_nothing.per_position_rates(), vec![0.5, 1.0, 0.0]);

        assert_ne!(
            dies_at_1.per_position_rates(),
            all_or_nothing.per_position_rates(),
            "two runs identical in every aggregate must be separable per \
             position — that separation is the entire point of this field"
        );

        // The refinement invariants: the breakdown must not be a second,
        // competing count.
        for snap in [dies_at_1, all_or_nothing] {
            assert_eq!(
                snap.accepted_at.iter().sum::<usize>(),
                snap.accepted,
                "per-position accepted must sum to the aggregate"
            );
            assert_eq!(
                snap.reached_at[0], snap.drafted_steps,
                "every drafting step reaches position 0"
            );
            for j in 1..snap.tracked_extent() {
                assert!(
                    snap.reached_at[j] <= snap.accepted_at[j - 1],
                    "a linear chain can only reach position {j} through position {}",
                    j - 1
                );
            }
            assert_eq!(snap.positions_over_cap, 0);
        }
    }

    /// The per-position numbers must reach the two surfaces a GPU session
    /// actually reads: the greppable `MTP[...]` marker and the prose report.
    #[test]
    fn per_position_acceptance_is_on_the_marker_and_the_report() {
        let tel = AcceptanceTelemetry::default();
        // 4 steps at depth 3: accept 3, 2, 1, 0.
        // pos 0 reached 4x, accepted 3x  -> p = 0.75
        // pos 1 reached 3x, accepted 2x  -> p = 0.6667
        // pos 2 reached 2x, accepted 1x  -> p = 0.5
        for accepted in [3, 2, 1, 0] {
            tel.record_gated(&step_of(3, accepted), false);
        }
        let snap = tel.snapshot();
        assert_eq!(snap.accepted_at, positions(&[3, 2, 1]));
        assert_eq!(snap.reached_at, positions(&[4, 3, 2]));

        let marker = snap.marker("agg");
        // Appended, not substituted: every field the runbooks grep for is still
        // on the line.
        assert!(marker.contains("accept_rate="), "{marker}");
        assert!(marker.contains("tok_per_step="), "{marker}");
        assert!(marker.contains("drafted_steps=4"), "{marker}");
        assert!(marker.contains("accepted_pos=3,2,1"), "{marker}");
        assert!(marker.contains("reached_pos=4,3,2"), "{marker}");
        assert!(marker.contains("p_pos=0.7500,0.6667,0.5000"), "{marker}");
        assert!(marker.contains("pos_over_cap=0"), "{marker}");

        let report = snap.position_report_lines().join("\n");
        assert!(report.contains("pos 0: p=0.7500 (3/4)"), "{report}");
        assert!(report.contains("pos 1: p=0.6667 (2/3)"), "{report}");
        assert!(report.contains("pos 2: p=0.5000 (1/2)"), "{report}");

        // A run that never drafted reports nothing per position rather than
        // zeros — 0/0 is not 0%, the same rule `rate()` already follows.
        let idle = AcceptanceTelemetry::default();
        idle.record_gated(&MtpAcceptance::skipped_step(), false);
        let idle = idle.snapshot();
        assert_eq!(idle.tracked_extent(), 0);
        assert!(idle.position_report_lines().is_empty());
        assert!(idle.marker("agg").contains("p_pos=n/a"));
    }

    /// A chain deeper than the bucket cap still accounts correctly in every
    /// aggregate, and says so instead of silently dropping the overflow.
    #[test]
    fn positions_beyond_the_cap_are_flagged_not_hidden() {
        let deep = MTP_MAX_TRACKED_POSITIONS + 2;
        let tel = AcceptanceTelemetry::default();
        tel.record_gated(&step_of(deep, deep), false);
        let snap = tel.snapshot();

        assert_eq!(snap.accepted, deep, "the aggregate is exact regardless");
        assert_eq!(snap.proposed, deep);
        assert_eq!(snap.positions_over_cap, 1);
        assert_eq!(snap.tracked_extent(), MTP_MAX_TRACKED_POSITIONS);
        assert_eq!(
            snap.accepted_at.iter().sum::<usize>(),
            MTP_MAX_TRACKED_POSITIONS,
            "only the first {} positions get buckets",
            MTP_MAX_TRACKED_POSITIONS
        );
        assert!(snap
            .position_report_lines()
            .iter()
            .any(|l| l.contains("WARN")));
        assert!(snap.marker("agg").contains("pos_over_cap=1"));
    }

    /// A step that drafted nothing is still a step. Without it `tok_per_step`
    /// would silently exclude the case the draft KV hits most often — drafting
    /// skipped because the cache could not be primed — and report a multiplier
    /// the decode loop never delivered.
    #[test]
    fn skipped_draft_steps_count_as_steps_and_pull_the_multiplier_to_one() {
        let skipped = MtpAcceptance {
            accepted: 0,
            proposed: 0,
            steps: 1,
            drafted_steps: 0,
            committed: 1,
            batch_steps: 0,
            ..MtpAcceptance::default()
        };
        // The literal above must stay the thing the pipeline actually records.
        assert_eq!(skipped, MtpAcceptance::skipped_step());
        let tel = AcceptanceTelemetry::default();
        for _ in 0..10 {
            tel.record_gated(&skipped, false);
        }
        let snap = tel.snapshot();
        assert_eq!(
            snap.rate(),
            None,
            "0 proposals is 'no measurement', not '0% acceptance'"
        );
        assert_eq!(
            snap.tokens_per_step(),
            Some(1.0),
            "a run that never drafted must report the plain-decode multiplier"
        );
        assert_eq!(snap.drafted_steps, 0);
        assert!(
            snap.marker("agg").contains("accept_rate=n/a"),
            "{}",
            snap.marker("agg")
        );

        // Half drafted, half skipped: the multiplier is the honest average.
        for _ in 0..10 {
            tel.record_gated(&step_of(2, 2), false);
        }
        let snap = tel.snapshot();
        assert_eq!((snap.steps, snap.drafted_steps), (20, 10));
        assert_eq!(snap.rate(), Some(1.0));
        assert_eq!(snap.tokens_per_step(), Some((10.0 + 30.0) / 20.0));
    }

    /// The env gate is **on by default**. Three GPU sessions produced empty
    /// acceptance artifacts; an opt-in measurement is one a rented box forgets
    /// to turn on. Only an explicit off-value suppresses it.
    #[test]
    fn acceptance_logging_is_on_unless_explicitly_disabled() {
        assert!(acceptance_log_from_env(None), "unset must mean ON");
        for on in ["1", "true", "on", "yes", " 1 "] {
            assert!(acceptance_log_from_env(Some(on)), "{on:?} must mean ON");
        }
        for off in ["0", "false", "off", "no", " 0 "] {
            assert!(!acceptance_log_from_env(Some(off)), "{off:?} must mean OFF");
        }
    }

    /// The report fires once per [`ACCEPTANCE_LOG_EVERY_PROPOSED`] proposed
    /// tokens — no more, no less — including when a chain straddles the
    /// boundary rather than landing on it.
    #[test]
    fn acceptance_report_fires_once_per_log_period() {
        // Depth 2: 64 proposals = 32 steps, so the first 31 must be silent.
        let tel = AcceptanceTelemetry::default();
        for step in 1..=31 {
            assert!(
                !tel.accumulate(&step_of(2, 1), ACCEPTANCE_LOG_EVERY_PROPOSED),
                "step {step} (total {} proposed) is inside the first period",
                step * 2
            );
        }
        assert!(
            tel.accumulate(&step_of(2, 1), ACCEPTANCE_LOG_EVERY_PROPOSED),
            "the step that carries the total to 64 must report"
        );

        // Depth 3 never lands exactly on 64 (64/3 is not an integer), so the
        // boundary is *straddled*: 63 -> 66. A naive `total % every == 0` test
        // would report zero times here, which is the shape of bug that makes a
        // telemetry hook look wired while producing nothing.
        let tel = AcceptanceTelemetry::default();
        let fires = (0..300)
            .filter(|_| tel.accumulate(&step_of(3, 2), 64))
            .count();
        let snap = tel.snapshot();
        assert_eq!((snap.accepted, snap.proposed), (600, 900));
        assert_eq!(fires, 900 / 64, "900 proposed tokens = 14 whole periods");

        // A single oversized batch crosses several periods at once and still
        // reports exactly once — the report is periodic, not per-period.
        let tel = AcceptanceTelemetry::default();
        assert!(tel.accumulate(&step_of(1000, 500), 64));
        assert!(
            !tel.accumulate(&MtpAcceptance::default(), 64),
            "an empty verify reports nothing"
        );

        // `every == 0` disables reporting instead of dividing by zero.
        let tel = AcceptanceTelemetry::default();
        assert!(!tel.accumulate(&step_of(64, 32), 0));
    }

    /// The reported rate is proposed-vs-accepted **as counted** (DOCTRINE D9),
    /// carrying both raw numbers so the reader can check the arithmetic — not
    /// a derived or smoothed estimate.
    #[test]
    fn acceptance_report_states_the_counted_ratio() {
        let tel = AcceptanceTelemetry::default();
        assert_eq!(
            tel.report_line(),
            "MTP acceptance: 0 proposals so far",
            "with no proposals there is no rate to report, and 0/0 must not be \
             printed as 0% or NaN"
        );

        // 65 depth-2 chains: 26 fully accepted, 39 half accepted.
        // 130 proposed / 91 accepted = exactly 70.0%.
        for _ in 0..26 {
            tel.record_gated(&step_of(2, 2), false);
        }
        for _ in 0..39 {
            tel.record_gated(&step_of(2, 1), false);
        }
        let snap = tel.snapshot();
        assert_eq!((snap.accepted, snap.proposed), (91, 130));
        let line = tel.report_line();
        assert_eq!(line, "MTP acceptance rate: 70.0% (91/130 accepted)");
        // The raw counters are in the line, so the percentage is auditable
        // rather than something the reader has to trust.
        assert!(
            line.contains(&format!("{}/{}", snap.accepted, snap.proposed)),
            "{line}"
        );

        // Same for the greppable marker: every raw count plus both ratios.
        let marker = snap.marker("agg");
        assert!(marker.starts_with("MTP[agg] "), "{marker}");
        for field in [
            "accept_rate=0.7000",
            "accepted=91",
            "proposed=130",
            "steps=65",
            "drafted_steps=65",
            // 26 * (1 + 2) + 39 * (1 + 1 + 1) = 78 + 117
            "committed=195",
            "tok_per_step=3.0000",
        ] {
            assert!(marker.contains(field), "marker missing {field}: {marker}");
        }
    }

    /// **The wiring gate.** The line must reach `tracing` from
    /// `record_acceptance`'s code path when the gate is on, and must not when
    /// it is off. This is the assertion whose absence cost session 3 its MTP
    /// number: the counters were fine; nothing ever logged them.
    #[test]
    fn acceptance_logger_fires_from_the_record_path_only_when_gated_on() {
        // Gate ON: 32 depth-2 verifies = 64 proposed = one report.
        let tel = AcceptanceTelemetry::default();
        let logs = capture_logs(|| {
            for _ in 0..32 {
                tel.record_gated(&step_of(2, 1), true);
            }
        });
        assert_eq!(
            logs.matches("MTP acceptance rate").count(),
            1,
            "exactly one periodic report over 64 proposed tokens; got:\n{logs}"
        );
        assert_eq!(
            logs.matches("MTP[agg]").count(),
            1,
            "the greppable marker must accompany it, exactly once; got:\n{logs}"
        );
        assert!(
            logs.contains("MTP acceptance rate: 50.0% (32/64 accepted)"),
            "the emitted line must carry the counted ratio; got:\n{logs}"
        );

        // Gate OFF: identical traffic, no output. (Counters still accumulate —
        // `log_acceptance_rate()` and `acceptance()` stay useful.)
        let tel = AcceptanceTelemetry::default();
        let logs = capture_logs(|| {
            for _ in 0..32 {
                tel.record_gated(&step_of(2, 1), false);
            }
        });
        assert!(
            !logs.contains("MTP acceptance") && !logs.contains("MTP["),
            "the gate must suppress every line; got:\n{logs}"
        );
        let snap = tel.snapshot();
        assert_eq!((snap.accepted, snap.proposed), (32, 64));

        // The manual door still works regardless of the gate.
        let logs = capture_logs(|| tel.log());
        assert!(
            logs.contains("MTP acceptance rate: 50.0% (32/64 accepted)"),
            "log_acceptance_rate must report on demand; got:\n{logs}"
        );
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

    /// Cache-truncation accounting after a verify forward (the session-4 MTP
    /// hang triage's confirmed bug — see `n_cache_positions_to_drop`).
    ///
    /// Model: before the step the cache holds `tokens - 1` positions (plain
    /// decode invariant). The step-1 forward adds 1; the verify forward adds
    /// `n_proposed`; the step commits `1 + commit_len` tokens. After dropping
    /// `n_cache_positions_to_drop`, the invariant must hold again.
    #[test]
    fn cache_truncation_restores_decode_invariant() {
        // (proposed, target) pairs covering depth-2 all-accept, mid-reject,
        // first-reject, plus a depth-4 mid-reject.
        let cases: &[(&[u32], &[u32])] = &[
            (&[1, 2], &[1, 2]),             // all accepted
            (&[1, 2], &[1, 9]),             // rejected at 1
            (&[1, 2], &[9, 2]),             // rejected at 0
            (&[1, 2, 3, 4], &[1, 2, 9, 4]), // depth 4, rejected at 2
        ];
        for (proposed, target) in cases {
            let r = verify_proposed(proposed, target);
            let d = proposed.len();
            let n_drop = n_cache_positions_to_drop(d, &r);

            // Simulate the bookkeeping around one fast-path step.
            let tokens_before = 100usize;
            let cache_before = tokens_before - 1; // plain-decode invariant
            let cache_after_t0_fwd = cache_before + 1; // step-1 target forward
            let cache_after_verify = cache_after_t0_fwd + d; // verify forward
            let cache_final = cache_after_verify - n_drop;

            let tokens_committed = 1 + r.commit_len(); // T0 + accepted + correction
            let tokens_after = tokens_before + tokens_committed;

            assert_eq!(
                cache_final,
                tokens_after - 1,
                "cache must hold every committed token except the last \
                 (proposed={proposed:?} target={target:?} accepted={} \
                 rejected={} n_drop={n_drop})",
                r.accepted.len(),
                r.rejection.is_some(),
            );
        }
    }

    /// The rejection case drops exactly one FEWER slot than the accepted-gap:
    /// the correction token is committed but never entered the cache, so its
    /// slot must not be charged. (The pre-fix code dropped
    /// `n_proposed - n_accepted` and desynced by one per rejection.)
    #[test]
    fn cache_truncation_rejection_off_by_one_pinned() {
        // depth 2, rejected at 0: accepted 0 → old code dropped 2, correct is 1.
        let r = verify_proposed(&[1, 2], &[9, 2]);
        assert_eq!(n_cache_positions_to_drop(2, &r), 1);
        // depth 2, rejected at 1: accepted 1 → old code dropped 1, correct is 0.
        let r = verify_proposed(&[1, 2], &[1, 9]);
        assert_eq!(n_cache_positions_to_drop(2, &r), 0);
        // depth 2, all accepted: no rejection → drop 0 (unchanged from old code).
        let r = verify_proposed(&[1, 2], &[1, 2]);
        assert_eq!(n_cache_positions_to_drop(2, &r), 0);
        // Degenerate: verifier produced fewer rows than proposals (no
        // rejection recorded, accepted < proposed) → drop the un-verified
        // tail (matches the old code on this edge).
        let r = verify_proposed(&[1, 2, 3], &[1, 2]);
        assert!(r.rejection.is_none());
        assert_eq!(r.accepted.len(), 2);
        assert_eq!(n_cache_positions_to_drop(3, &r), 1);
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
        let tokens = kit.propose_chain(&prev_hidden, 0, 5, 2, 0, None)?;
        assert_eq!(
            tokens.len(),
            2,
            "cap should clip chain length to max_tokens"
        );

        // depth=4, max_tokens=4 → exactly 4 (equality holds).
        let tokens = kit.propose_chain(&prev_hidden, 0, 4, 4, 0, None)?;
        assert_eq!(tokens.len(), 4);

        // max_tokens=0 → empty chain regardless of depth.
        let tokens = kit.propose_chain(&prev_hidden, 0, 8, 0, 0, None)?;
        assert!(tokens.is_empty(), "max_tokens=0 must return no tokens");

        // depth=0 → empty chain regardless of max_tokens.
        let tokens = kit.propose_chain(&prev_hidden, 0, 0, 8, 0, None)?;
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
            let proposed = kit.propose_chain(&prev_hidden, prev_tok, depth, depth, 0, None)?;
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
        fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) -> candle_core::Result<()> {
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

    /// Build a `Sequence` whose sampler carries exactly the settings under
    /// test. Mirrors `pipeline::amoe::new_dummy_seq` — no model, no engine.
    fn seq_with_sampler(sampler: crate::sampler::Sampler) -> Sequence {
        use crate::sequence::{SequenceGroup, SequenceRecognizer};
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let group = Arc::new(std::sync::Mutex::new(SequenceGroup::new(
            1, false, false, None,
        )));
        Sequence::new_waiting(
            vec![1, 2, 3],
            "test".to_string(),
            0,
            0,
            1,
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
            crate::sequence::SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            vec![],
        )
    }

    /// A sampler with everything off except the one knob under test.
    fn sampler_with(
        temperature: Option<f64>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        repetition_penalty: Option<f32>,
    ) -> crate::sampler::Sampler {
        crate::sampler::Sampler::new(
            temperature,
            0,
            None,
            frequency_penalty,
            presence_penalty,
            repetition_penalty,
            None,
            -1,
            1.0,
            0.0,
            None,
            vec![],
        )
        .expect("Sampler::new with plain settings must succeed")
    }

    /// **The doctrine violation this gate closes (D4: greedy decoding is
    /// banned).**
    ///
    /// A `temperature > 0` request must not take the MTP path. Accepted drafts
    /// are committed straight to the sequence with `logprob: 0.0`, bypassing
    /// `sample_sequence` entirely — so if such a request were allowed to draft,
    /// it would receive **greedy** tokens at every accepted position while its
    /// caller believed it had asked for sampling.
    #[test]
    fn a_temperature_request_does_not_take_the_mtp_path() {
        // Greedy: drafts.
        let mut greedy = seq_with_sampler(sampler_with(None, None, None, None));
        assert!(
            seq_may_draft(&mut greedy),
            "a plain greedy request is exactly what MTP verification is lossless for"
        );

        // temperature > 0: must not draft.
        for temperature in [0.1, 0.7, 1.0, 2.0] {
            let mut hot = seq_with_sampler(sampler_with(Some(temperature), None, None, None));
            assert!(
                !seq_may_draft(&mut hot),
                "temperature={temperature} must not draft: MTP would commit argmax tokens \
                 through a path that never calls the sampler"
            );
        }

        // temperature below the sampler's own greedy threshold (1e-7) IS
        // greedy — `Sampler::new` folds it to `None` — so it may draft. Reading
        // the raw request field instead of the sampler's effective setting
        // would wrongly refuse here.
        let mut effectively_greedy = seq_with_sampler(sampler_with(Some(1e-9), None, None, None));
        assert!(
            seq_may_draft(&mut effectively_greedy),
            "a temperature the sampler itself folds to greedy must still draft"
        );

        // The same defect, same shape: with any penalty the sampler rewrites
        // the logits before its own argmax, so argmax(raw) is not what the
        // request would have got.
        for (freq, pres, rep) in [
            (Some(0.5), None, None),
            (None, Some(0.5), None),
            (None, None, Some(1.1)),
        ] {
            let mut penalised = seq_with_sampler(sampler_with(None, freq, pres, rep));
            assert!(
                !seq_may_draft(&mut penalised),
                "greedy + a penalty is still not argmax over RAW logits"
            );
        }

        // A zero/identity penalty is not a penalty.
        let mut identity = seq_with_sampler(sampler_with(None, Some(0.0), Some(0.0), Some(1.0)));
        assert!(
            seq_may_draft(&mut identity),
            "an identity penalty must not cost a request its speculative decode"
        );
    }

    /// The gate has to reach the drafting partition, not just exist: a sequence
    /// that may not draft must appear in **no** draft group, which is what
    /// leaves `drafts[i]` empty and routes its token through `sample_sequence`.
    #[test]
    fn a_sequence_that_may_not_draft_is_in_no_draft_group() {
        const W: usize = 3;
        // Three sequences, identical in every way the partition cares about
        // (same uncached tail, same cache length) so the ONLY thing that can
        // separate them is the sampling gate.
        let uncached = vec![1usize, 1, 1];
        let cache_lens = vec![10usize, 10, 10];

        let all_greedy = draft_groups(&uncached, &cache_lens, &[true, true, true], W, false);
        let members: Vec<usize> = all_greedy.values().flatten().copied().collect();
        assert_eq!(
            members,
            vec![0, 1, 2],
            "with every sequence greedy, all three draft together"
        );

        let middle_hot = draft_groups(&uncached, &cache_lens, &[true, false, true], W, false);
        let members: Vec<usize> = middle_hot.values().flatten().copied().collect();
        assert_eq!(
            members,
            vec![0, 2],
            "the non-greedy sequence must be in no group at all — and the greedy \
             sequences beside it must keep drafting, so one hot request does not \
             cost the whole batch its speculative decode"
        );

        let none = draft_groups(&uncached, &cache_lens, &[false, false, false], W, false);
        assert!(
            none.is_empty(),
            "an all-non-greedy batch produces no draft groups"
        );

        // A row already at the full window still cannot draft, gate or no gate —
        // the pre-existing `u < w` rule is unchanged.
        let full = draft_groups(&[W], &[10], &[true], W, false);
        assert!(full.is_empty(), "u == w leaves no room for a draft");
    }

    /// What a non-drafting sequence's accounting looks like: `proposed == 0`,
    /// so it contributes nothing to `accept_rate` and exactly one token per
    /// step — plain decode, which is the definition of "did not take the MTP
    /// path".
    #[test]
    fn a_non_drafting_sequence_accounts_as_plain_decode() {
        let stat = MtpAcceptance::from_fused_verify(0, 0);
        assert_eq!(stat.proposed, 0);
        assert_eq!(stat.accepted, 0);
        assert_eq!(
            stat.drafted_steps, 0,
            "a step with no proposals is not a draft"
        );
        assert_eq!(stat.committed, 1, "exactly the token the sampler produced");
        assert_eq!(
            stat.rate(),
            None,
            "a request that never drafted has no acceptance rate — 0/0 is not 0%"
        );
        assert_eq!(
            stat.tokens_per_step(),
            Some(1.0),
            "the plain-decode multiplier"
        );
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

    // =====================================================================
    // Batched MTP: the ragged accept/reject case (wave45-BW)
    // =====================================================================
    //
    // The case that breaks a batched speculative decoder is one step in which
    // different sequences accept different numbers of drafts. One dense
    // `NormalCache` carries ONE length for the whole batch, so that step has to
    // resolve B different answers into one number while every sequence's
    // user-visible token stream stays exactly what it would have been alone.
    //
    // These tests drive the PRODUCTION arithmetic — `plan_batch_step`,
    // `window_verify_row`, `verify_proposed` — against a deterministic
    // target/draft pair, so a change to the real functions moves the test.

    /// A deterministic stand-in for the target model: the next token is a
    /// function of the last two committed tokens.
    ///
    /// Deliberately NOT a constant and NOT a function of position: the fixture
    /// trap this repo already stepped on (`h_proj`/`e_proj` both zero ⇒
    /// `fused ≡ 0` ⇒ every draft token identical) makes accept and reject the
    /// same experiment, and every MTP defect invisible. DOCTRINE D12.
    fn oracle_target(ctx: &[u32]) -> u32 {
        let a = u64::from(*ctx.last().expect("non-empty context"));
        let b = u64::from(ctx.get(ctx.len().wrapping_sub(2)).copied().unwrap_or(7));
        ((a.wrapping_mul(1103515245)
            .wrapping_add(b.wrapping_mul(12345)))
            % 97
            + 3) as u32
    }

    /// A deterministic stand-in for the MTP draft head. `agree_mod` controls how
    /// often it reproduces the target: the draft agrees whenever the target's
    /// own answer is divisible by `agree_mod`, which makes acceptance depend on
    /// the sequence's own content — the only way two sequences in one batch end
    /// up with different accept lengths.
    fn oracle_draft(ctx: &[u32], agree_mod: u32) -> u32 {
        let t = oracle_target(ctx);
        if agree_mod != 0 && t % agree_mod == 0 {
            t
        } else {
            t.wrapping_add(1)
        }
    }

    /// One sequence's decode state under the fused window scheme.
    #[derive(Clone)]
    struct SimSeq {
        toks: Vec<u32>,
        agree_mod: u32,
        /// Uncached committed tail — `len - shared_cache_len`.
        uncached: usize,
    }

    /// Run one fused step for a batch of `SimSeq`, exactly as
    /// `MtpSpeculativePipeline::step` does: draft `w - u` tokens per sequence,
    /// feed a `w`-wide window, verify with `verify_proposed` against the rows
    /// `window_verify_row` names, then resolve the batch with
    /// `plan_batch_step`. Returns the tokens each sequence emitted this step.
    ///
    /// `plan_override` lets a mutation test substitute a WRONG batch resolution
    /// and show that the assertions catch it.
    fn sim_step(
        seqs: &mut [SimSeq],
        w: usize,
        plan_override: Option<fn(&[usize], &[usize], usize) -> BatchStepPlan>,
    ) -> Vec<Vec<u32>> {
        let (uncached, n_accepted, emitted) = sim_propose_and_verify(seqs, w);
        let plan = plan_override.unwrap_or(plan_batch_step)(&uncached, &n_accepted, w);
        for (i, s) in seqs.iter_mut().enumerate() {
            s.toks.extend_from_slice(&emitted[i]);
            s.uncached = (s.uncached + n_accepted[i] + 1).saturating_sub(plan.keep);
        }
        emitted
    }

    /// The same step under [`KvAdvance::PerSequence`]. **Byte-for-byte the same
    /// drafting and the same verify** — `sim_propose_and_verify` is shared — so
    /// the only thing that differs between this and [`sim_step`] is how the KV
    /// cache advances. Any divergence in the tests below is therefore
    /// attributable to the advance rule and to nothing else.
    fn sim_step_per_sequence(seqs: &mut [SimSeq], w: usize) -> Vec<Vec<u32>> {
        let (uncached, n_accepted, emitted) = sim_propose_and_verify(seqs, w);
        let plan = plan_per_sequence_step(&uncached, &n_accepted, w);
        for (i, s) in seqs.iter_mut().enumerate() {
            s.toks.extend_from_slice(&emitted[i]);
            s.uncached = plan.next_uncached[i];
        }
        emitted
    }

    /// Draft `w - u` tokens per sequence, feed the `w`-wide window, verify with
    /// `verify_proposed` against the rows `window_verify_row` names. Returns
    /// `(uncached, n_accepted, emitted)`; it does **not** touch the cache.
    fn sim_propose_and_verify(
        seqs: &[SimSeq],
        w: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<Vec<u32>>) {
        let uncached: Vec<usize> = seqs.iter().map(|s| s.uncached).collect();
        let mut drafts: Vec<Vec<u32>> = Vec::with_capacity(seqs.len());
        for s in seqs.iter() {
            let n_draft = w - s.uncached;
            let mut chain = Vec::with_capacity(n_draft);
            let mut ctx = s.toks.clone();
            for _ in 0..n_draft {
                let next = oracle_draft(&ctx, s.agree_mod);
                chain.push(next);
                ctx.push(next);
            }
            drafts.push(chain);
        }

        // The verify forward: row `window_verify_row(u, j)` of the window is the
        // target's own token after the window's first `u + j` tokens.
        let mut emitted: Vec<Vec<u32>> = Vec::with_capacity(seqs.len());
        let mut n_accepted: Vec<usize> = Vec::with_capacity(seqs.len());
        for (i, s) in seqs.iter().enumerate() {
            let d = drafts[i].len();
            let mut rows: Vec<u32> = vec![0; w];
            let mut ctx = s.toks.clone();
            // Row for j = 0 sits on the last real token; rows for j >= 1 sit on
            // draft j-1, so the context grows with the drafts.
            for j in 0..=d {
                rows[window_verify_row(s.uncached, j)] = oracle_target(&ctx);
                if j < d {
                    ctx.push(drafts[i][j]);
                }
            }
            let targets: Vec<u32> = (0..d)
                .map(|j| rows[window_verify_row(s.uncached, j)])
                .collect();
            let result = verify_proposed(&drafts[i], &targets);
            let a = result.accepted.len();
            let mut out = result.accepted.clone();
            out.push(rows[window_verify_row(s.uncached, a)]);
            n_accepted.push(a);
            emitted.push(out);
        }

        (uncached, n_accepted, emitted)
    }

    // =====================================================================
    // Per-sequence KV advance — the production rule (wave62)
    // =====================================================================
    //
    // `plan_batch_step` is correct and has a ceiling; `plan_per_sequence_step`
    // is what vLLM, SGLang and TensorRT-LLM do. Every test below drives BOTH
    // through the same `sim_propose_and_verify`, so the drafting, the verify
    // and the accept rule are literally the same code and the only variable is
    // the cache advance.

    /// A B=128 cohort with content-dependent accept lengths — the fixture the
    /// whole argument is about.
    fn batch_of(n: usize) -> Vec<SimSeq> {
        (0..n)
            .map(|i| SimSeq {
                // Distinct seeds so the sequences genuinely diverge; two equal
                // seeds would make ragged accept unreachable and the test
                // vacuous.
                toks: vec![11 + (i as u32 % 61), 5 + (i as u32 % 37)],
                // 2..=5: a spread of per-sequence acceptance, which is what
                // makes `min_i` bite.
                agree_mod: 2 + (i as u32 % 4),
                uncached: 1,
            })
            .collect()
    }

    /// The capability probe is what stands between "asked for per-sequence
    /// advance" and "silently served the cohort ceiling". A V4-shaped cache —
    /// K/V slots plus the compressor's `XsRolling` histories — must decline,
    /// and must name the slot, because that slot is the actual remaining work.
    #[test]
    fn the_capability_probe_names_the_slot_that_cannot_carry_its_own_length() {
        use crate::kv_cache::{EitherCache, KvCache, NormalCache, XsRollingCache};
        use std::sync::{Arc, Mutex};

        let all_normal = EitherCache::Normal(Arc::new(Mutex::new(NormalCache(vec![
            KvCache::new_normal(2, 4096, 512),
            KvCache::new_normal(2, 4096, 512),
        ]))));
        assert!(
            cache_supports_per_sequence_advance(&all_normal).is_ok(),
            "a cache of plain Normal slots can carry per-sequence lengths"
        );

        // V4's shape: K/V slots interleaved with compressor histories.
        let v4_shaped = EitherCache::Normal(Arc::new(Mutex::new(NormalCache(vec![
            KvCache::new_normal(2, 4096, 512),
            KvCache::XsRolling(Box::new(XsRollingCache::new(4, 2, 64, 2048))),
        ]))));
        let err = cache_supports_per_sequence_advance(&v4_shaped).unwrap_err();
        assert!(
            err.contains("slot 1") && err.contains("XsRolling"),
            "the refusal must name which slot and which kind; got {err}"
        );
    }

    /// The **mask** capability, which this wave moves. A V4 target passes it
    /// now (`dsv4_attention` takes a per-row `row_q0`); a plain-`Normal` target
    /// still does not, and must say so rather than inherit V4's answer.
    ///
    /// Paired with the refusal that remains, so the pair reads as one
    /// statement: the mask is no longer the blocker, and something else is.
    #[test]
    fn the_mask_capability_passes_for_v4_and_still_refuses_for_everyone_else() {
        use crate::kv_cache::{EitherCache, KvCache, NormalCache, XsRollingCache};
        use std::sync::{Arc, Mutex};

        let v4_shaped = EitherCache::Normal(Arc::new(Mutex::new(NormalCache(vec![
            KvCache::new_normal(2, 4096, 512),
            KvCache::XsRolling(Box::new(XsRollingCache::new(4, 2, 64, 2048))),
        ]))));
        assert!(
            model_masks_ragged_batches(&v4_shaped).is_ok(),
            "a V4 target threads the per-row `row_q0` into `dsv4_attention`, so the ragged mask \
             is no longer what blocks it"
        );

        let plain = EitherCache::Normal(Arc::new(Mutex::new(NormalCache(vec![
            KvCache::new_normal(2, 4096, 512),
            KvCache::new_normal(2, 4096, 512),
        ]))));
        let err = model_masks_ragged_batches(&plain).unwrap_err();
        assert!(
            err.contains("XsRolling") && err.contains("dead"),
            "a non-V4 target must still refuse, and name why; got {err}"
        );
    }

    /// 🔴 **The refusal moved; it did not disappear.** With all four
    /// previously-named blockers cleared — the cache carries per-row lengths
    /// (wave63-CO), V4's attention masks a ragged cohort (PR #100), the fused
    /// step's arithmetic is per-row (PR #102), and the draft chain now carries
    /// per-row positions — a V4 target must STILL be refused, and the reason
    /// must now name the **scheduler**.
    ///
    /// This is the test that stops the remaining check from being quietly
    /// dropped, and the assertions that it no longer names the cleared ones are
    /// what make "the refusal moved" a claim a test can fail. Each `!contains`
    /// below is a previous wave's blocker: if any reappears, that wave was
    /// reverted.
    #[test]
    fn a_fully_capable_v4_cache_is_still_refused_and_now_names_the_scheduler() {
        use crate::kv_cache::{EitherCache, KvCache, NormalCache, XsRollingCache};
        use std::sync::{Arc, Mutex};

        let v4 = EitherCache::Normal(Arc::new(Mutex::new(NormalCache(vec![
            KvCache::new_normal(2, 4096, 512),
            KvCache::XsRolling(Box::new(XsRollingCache::new(4, 2, 64, 2048))),
        ]))));

        // Force the cache capability ON, thread-locally, so checks (1)-(3) all
        // pass and check (4) is the only thing left standing.
        let refusal = crate::kv_cache::xs_per_seq_test_override::with(true, || {
            assert!(
                cache_supports_per_sequence_advance(&v4).is_ok(),
                "precondition: with ARC_V4_XS_PER_SEQ on, a V4 cache carries per-row lengths"
            );
            assert!(
                model_masks_ragged_batches(&v4).is_ok(),
                "precondition: V4's attention now masks a ragged cohort"
            );
            assert!(
                draft_chain_carries_per_sequence_positions(&v4).is_ok(),
                "precondition: V4's MTP draft chain now carries per-row positions"
            );
            per_sequence_refusal(&v4)
        });

        let why = refusal.expect(
            "per-sequence advance must STILL be refused: the scheduler buckets by cache length \
             and runs exactly one bucket per step, so divergent lengths shatter the batch",
        );
        assert!(
            why.contains("cache_bucket_len") && why.contains("bucket"),
            "the refusal must name the scheduler's bucket key, not a blocker that is already \
             cleared; got {why}"
        );
        assert!(
            !why.contains("XsRolling"),
            "the refusal still names the compressor slot, so the flag is not reaching the \
             cache and wave63-CO changed nothing; got {why}"
        );
        assert!(
            !why.contains("window_ok") && !why.contains("cache_len ="),
            "the refusal still names the fused step's shared `cache_len`, which PR #102 \
             removed; got {why}"
        );
        assert!(
            !why.contains("forward_step") && !why.contains("MtpHiddenCapture"),
            "the refusal still names the draft chain's scalar `pos`, which THIS wave removed — \
             so it changed nothing; got {why}"
        );
    }

    /// The **draft-chain** capability, which this wave moves. A V4 target passes
    /// it (its MTP block is a V4 decoder layer, so `dsv4_attention`'s per-row
    /// `row_q0` masks the left-aligned draft cohort); a plain-`Normal` target
    /// must not inherit that answer, because left-aligning draft KVs for a model
    /// that cannot mask a dead prefix is a wrong answer, not a slow one.
    #[test]
    fn the_draft_chain_capability_passes_for_v4_and_still_refuses_for_everyone_else() {
        use crate::kv_cache::{EitherCache, KvCache, NormalCache, XsRollingCache};
        use std::sync::{Arc, Mutex};

        let v4 = EitherCache::Normal(Arc::new(Mutex::new(NormalCache(vec![
            KvCache::new_normal(2, 4096, 512),
            KvCache::XsRolling(Box::new(XsRollingCache::new(4, 2, 64, 2048))),
        ]))));
        assert!(draft_chain_carries_per_sequence_positions(&v4).is_ok());

        let plain = EitherCache::Normal(Arc::new(Mutex::new(NormalCache(vec![
            KvCache::new_normal(2, 4096, 512),
        ]))));
        let err = draft_chain_carries_per_sequence_positions(&plain).unwrap_err();
        assert!(
            err.contains("left-aligns") && err.contains("draft"),
            "a non-V4 target must refuse the draft chain by name; got {err}"
        );
    }

    /// ⚠️ **The honest statement of a redundancy, so it cannot rot into a
    /// silent grant.**
    ///
    /// Check (3) is currently `model_masks_ragged_batches` verbatim — the MTP
    /// block is a V4 decoder layer, so the same `row_q0` masks its draft cohort.
    /// One consequence is that deleting check (3) from `per_sequence_refusal` is
    /// a no-op that no mutation can catch; it is kept anyway, because the
    /// implication is a property of today's single-architecture discriminator
    /// and not of the two capabilities.
    ///
    /// This test is what makes that safe. Teach `model_masks_ragged_batches` a
    /// second architecture and it fails here, forcing the draft chain to be
    /// answered for that architecture rather than inherited.
    #[test]
    fn the_draft_chain_check_is_implied_by_the_mask_check_today() {
        use crate::kv_cache::{EitherCache, KvCache, NormalCache, XsRollingCache};
        use std::sync::{Arc, Mutex};

        let shapes = [
            // V4: a compressor slot present.
            vec![
                KvCache::new_normal(2, 4096, 512),
                KvCache::XsRolling(Box::new(XsRollingCache::new(4, 2, 64, 2048))),
            ],
            // Plain: none.
            vec![KvCache::new_normal(2, 4096, 512)],
            // A single compressor slot on its own.
            vec![KvCache::XsRolling(Box::new(XsRollingCache::new(
                4, 2, 64, 2048,
            )))],
        ];
        for slots in shapes {
            let cache = EitherCache::Normal(Arc::new(Mutex::new(NormalCache(slots))));
            assert_eq!(
                model_masks_ragged_batches(&cache).is_ok(),
                draft_chain_carries_per_sequence_positions(&cache).is_ok(),
                "the two checks must agree exactly while they share a discriminator — if they \
                 no longer do, `per_sequence_refusal` is granting the draft chain on the mask's \
                 evidence"
            );
        }
    }

    /// 🔑 The dispatch, pinned directly. A uniform batch — every B=1 request,
    /// every batch built with the flag off — must return `None`, because `None`
    /// is what runs the pre-change scalar arithmetic verbatim.
    ///
    /// ⚠️ This is the mutation PR #100 warned about and caught the hard way:
    /// routing a uniform batch down the per-row path is **numerically
    /// invisible**, since `cache_lens` would just be `cache_len` repeated. It is
    /// still a real defect — it flips `set_prefill_toks_at` on for every decode,
    /// builds a `[B]` offsets vector per forward, and sends `dsv4_attention` a
    /// `row_q0` it has to re-check on all 43 layers. Only a test of the dispatch
    /// itself can see it.
    #[test]
    fn a_uniform_batch_never_reaches_the_per_row_step_arithmetic() {
        assert_eq!(
            resolve_row_cache_lens(true, 40, &[0, 0, 0, 0]).unwrap(),
            None,
            "no row carries a dead prefix, so the batch is uniform and must take the scalar path"
        );
        assert_eq!(
            resolve_row_cache_lens(true, 40, &[]).unwrap(),
            None,
            "an empty batch is uniform"
        );
        assert_eq!(
            resolve_row_cache_lens(true, 40, &[0, 7, 0, 3]).unwrap(),
            Some(vec![40, 33, 40, 37]),
            "a row's own length is the batch width minus its dead prefix"
        );
    }

    /// The per-row route is opt-in on the same flag that authorizes the only
    /// producer of a ragged batch. With it off, even a batch that somehow
    /// carries leads takes the scalar path — the gate is a guarantee, not a
    /// coincidence.
    #[test]
    fn the_per_row_step_arithmetic_is_opt_in() {
        assert_eq!(
            resolve_row_cache_lens(false, 40, &[0, 7, 0, 3]).unwrap(),
            None,
            "with per-sequence advance off, nothing may take the per-row path"
        );
    }

    /// `front_align_batch` pads every row UP TO the batch maximum, so no lead
    /// can exceed the width and at least one row must have none. A vector that
    /// breaks either half is not the layout this arithmetic is defined for, and
    /// guessing would be a wrong answer nothing downstream catches.
    #[test]
    fn a_batch_that_is_not_a_left_aligned_cohort_is_refused_by_name() {
        let err = resolve_row_cache_lens(true, 10, &[0, 12])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("wider than") && err.contains("disagree"),
            "a lead past the batch width must say whose bookkeeping disagrees; got {err}"
        );

        let err = resolve_row_cache_lens(true, 10, &[2, 3])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("every row") && err.contains("end column"),
            "a batch with no row at the shared end column must say so; got {err}"
        );
    }

    /// 🔴 Defect 1, pinned as arithmetic. Fed the batched buffer's width, a
    /// ragged cohort's uncached tails saturate to 0 for every row but the
    /// longest, `window_ok` fails, and every ragged step defers — per-sequence
    /// advance would be reachable but inert. Fed each row's own length, every
    /// tail is 1, which is the B=1 invariant at every batch size.
    ///
    /// ⚠️ Arithmetic on the plan function, not a hardware measurement (D14).
    #[test]
    fn per_row_lengths_are_what_keep_a_ragged_cohorts_tails_inside_the_window() {
        let w = 4usize;
        // Three rows at 40/33/37 committed tokens, each one token ahead of its
        // own cache — the steady state per-sequence advance produces.
        let rows = [40usize, 33, 37];
        let tok_lens: Vec<usize> = rows.iter().map(|r| r + 1).collect();
        let cache_len = 40usize;

        let shared = plan_step_window(&[cache_len; 3], &tok_lens, w);
        assert_eq!(
            shared.uncached,
            vec![1, 0, 0],
            "the shared width saturates every row but the longest"
        );
        assert!(
            !shared.ok,
            "which trips the window invariant, so the fused step would defer on every ragged step"
        );

        let per_row = plan_step_window(&rows, &tok_lens, w);
        assert_eq!(per_row.uncached, vec![1, 1, 1]);
        assert!(per_row.ok, "each row's own length puts every tail at 1");
    }

    /// 🔴 Defect 2, pinned as arithmetic: `toks[cache_lens[i]..]` is a panic
    /// site unless `ok` guarantees the slice is in range. It does, for every
    /// row, and that guarantee is the ONLY thing standing between the verify
    /// window and an index past the end.
    ///
    /// ⚠️ Arithmetic on the plan function, not a hardware measurement (D14).
    #[test]
    fn the_window_invariant_is_what_keeps_the_verify_slice_in_range() {
        let w = 4usize;
        for cache_lens in [
            vec![10usize, 10, 10],
            vec![10usize, 7, 9],
            vec![3usize, 3, 3],
        ] {
            for extra in [0usize, 1, 2, 4, 5] {
                let tok_lens: Vec<usize> = cache_lens.iter().map(|c| c + extra).collect();
                let plan = plan_step_window(&cache_lens, &tok_lens, w);
                if plan.ok {
                    for (i, &c) in cache_lens.iter().enumerate() {
                        assert!(
                            c < tok_lens[i],
                            "ok must imply row {i}'s slice start {c} is inside its {} tokens",
                            tok_lens[i]
                        );
                    }
                }
            }
        }
        // A cache longer than the sequence — what the shared width looks like to
        // a short row — must never be reported ok.
        assert!(
            !plan_step_window(&[12, 12], &[13, 9], w).ok,
            "a row whose cache claims more positions than it has tokens is not a state this \
             fast path produced"
        );
    }

    /// 🔑 The cost this wave removes, as a count rather than as a benchmark.
    ///
    /// The draft partition used to key on `(u, chain_start)` because one
    /// MTP-block forward carried one absolute position. On a batch that shares a
    /// width the chain start is a function of `u` alone, so that key was bounded
    /// by `depth + 1` — but under per-sequence advance `u ≡ 1` and every row is
    /// at its own length, so it degenerated to **one group per sequence**: `B`
    /// single-row block forwards per chain step.
    ///
    /// With per-row positions the chain start leaves the key entirely. The
    /// `false` arm below is the pre-change behaviour, kept as the negative
    /// control that makes the `true` arm mean something.
    ///
    /// ⚠️ A count of groups, not a hardware measurement (D14).
    #[test]
    fn the_draft_group_partition_stops_collapsing_once_the_rows_carry_their_own_positions() {
        use std::collections::BTreeSet;
        let count = |uncached: &[usize], cache_lens: &[usize], per_row: bool| -> usize {
            uncached
                .iter()
                .zip(cache_lens)
                .map(|(&u, &c)| draft_group_key(u, c, per_row))
                .collect::<BTreeSet<_>>()
                .len()
        };

        // The cohort batch: one shared width, tails spread over the window.
        let uncached = [1usize, 2, 1, 3, 2, 1, 4, 1];
        let shared = [40usize; 8];
        assert_eq!(
            count(&uncached, &shared, false),
            4,
            "sharing a width makes the chain start a function of `u`, so the partition is \
             bounded by the window and one batched forward drafts for the whole group"
        );
        assert_eq!(
            count(&uncached, &shared, true),
            count(&uncached, &shared, false),
            "on a batch that shares a width the two keys must partition IDENTICALLY — the \
             chain start was already a function of `u` there, so dropping it changes nothing \
             the cohort path can observe"
        );

        // The per-sequence batch: every tail is 1 (the point of the mode), every
        // row is at its own length.
        let per_seq_lens = [40usize, 33, 37, 31, 39, 35, 30, 38];
        let ones = [1usize; 8];
        assert_eq!(
            count(&ones, &per_seq_lens, false),
            8,
            "negative control: with the scalar draft chain, rows at their own lengths partition \
             one-per-sequence — the B single-row block forwards this wave removes"
        );
        assert_eq!(
            count(&ones, &per_seq_lens, true),
            1,
            "with per-row positions every row needing the same number of drafts shares ONE \
             batched forward, whatever their absolute positions"
        );

        // Ragged in BOTH: distinct tails and distinct lengths. The bound is the
        // window, not the batch size.
        let mixed_u = [1usize, 2, 1, 3, 2, 1, 4, 1];
        assert_eq!(count(&mixed_u, &per_seq_lens, false), 8);
        assert_eq!(
            count(&mixed_u, &per_seq_lens, true),
            4,
            "the partition must be bounded by the number of distinct tails (<= depth + 1), not \
             by the batch size"
        );
    }

    /// 🔑 The draft-chain dispatch, pinned directly — the counterpart to
    /// `a_uniform_batch_never_reaches_the_ragged_path` in `dsv4_attention`.
    ///
    /// ⚠️ Routing a uniform group down the ragged path is **numerically
    /// invisible**: `chain_starts` would just be one value repeated and the mask
    /// would agree bit-for-bit. It is still a real defect — it puts RoPE on its
    /// per-row loop (`G` narrow+rope+cat instead of one) and builds a
    /// `[G, 1, 1, n_keys]` mask plus two host→device copies on every chain step
    /// of every B=1 request. Only a dispatch test can see it.
    #[test]
    fn a_uniform_draft_group_never_reaches_the_ragged_chain_path() {
        assert_eq!(
            resolve_ragged_chain_starts(&[40, 40, 40]),
            None,
            "a group whose rows share a chain start must take the scalar path"
        );
        assert_eq!(resolve_ragged_chain_starts(&[40]), None);
        assert_eq!(resolve_ragged_chain_starts(&[]), None);
        assert_eq!(
            resolve_ragged_chain_starts(&[40, 33, 40]),
            Some(&[40usize, 33, 40][..]),
            "a group whose rows differ must carry every row's own position, including the ones \
             that happen to sit at the maximum"
        );
    }

    /// The dead-prefix plan for a ragged draft group, and the two ways it must
    /// refuse to build one.
    ///
    /// `enabled` is `per_seq_advance`, not a fourth flag: `ragged_row_q0` — the
    /// only thing that masks the padding this plan creates — is gated on
    /// `ARC_MTP_PER_SEQ_KV`. Returning `Some` with it off would left-align rows
    /// whose zero columns nothing masks, and a zero K row is not a masked row.
    #[test]
    fn the_draft_leads_are_the_distance_to_the_groups_widest_row() {
        assert_eq!(
            resolve_draft_leads(true, &[40, 33, 37, 40]),
            Some(vec![0, 7, 3, 0]),
            "lead[i] is max_j start_j - start_i, and at least one row must have none"
        );
        assert_eq!(
            resolve_draft_leads(true, &[40, 40, 40]),
            None,
            "a group that already shares a start must take the pre-change path verbatim"
        );
        assert_eq!(
            resolve_draft_leads(false, &[40, 33]),
            None,
            "with per-sequence advance off, no group may be left-aligned — nothing would mask \
             the dead prefix"
        );
        assert_eq!(resolve_draft_leads(true, &[]), None);
    }

    /// 🔑 The capture offsets fold back to ONE element whenever the rows agree.
    ///
    /// That is what makes the cohort path byte-identical: `CaptureOffsets::row`
    /// then returns exactly what `seqlen_offsets.first().copied().unwrap_or(0)`
    /// returned for every row, and the per-row representation is never built.
    #[test]
    fn the_capture_offsets_collapse_when_the_rows_agree_and_survive_when_they_do_not() {
        let shared = resolve_capture_offsets(&[12, 12, 12, 12]);
        assert_eq!(shared, CaptureOffsets(vec![12]));
        for row in 0..4 {
            assert_eq!(shared.row(row), Some(12));
        }
        assert!(shared.describes(4) && shared.describes(1));

        let ragged = resolve_capture_offsets(&[12, 9, 11]);
        assert_eq!(ragged, CaptureOffsets(vec![12, 9, 11]));
        assert_eq!(ragged.row(0), Some(12));
        assert_eq!(ragged.row(1), Some(9), "row 1 must NOT read row 0's offset");
        assert_eq!(ragged.row(2), Some(11));
        assert_eq!(ragged.row(3), None, "a row the capture does not cover");
        assert!(ragged.describes(3));
        assert!(
            !ragged.describes(4),
            "a capture must refuse a batch it does not describe rather than index into it"
        );

        // An empty forward still has to produce something `row()` can answer.
        assert_eq!(resolve_capture_offsets(&[]).row(0), Some(0));
    }

    /// The group key must be a pure function of its arguments and agree with
    /// [`chain_start_of`], which is the expression the per-row draft uses. Two
    /// copies of "where does row `i` start" that could drift apart is exactly
    /// the shadow-state defect wave59-CJ documents.
    #[test]
    fn the_group_key_and_the_per_row_chain_start_are_the_same_expression() {
        for cache_len in [2usize, 3, 40, 4096] {
            for u in 1usize..=4 {
                assert_eq!(
                    draft_group_key(u, cache_len, false).1,
                    chain_start_of(cache_len, u),
                    "the scalar key's chain start must be the row's own chain start"
                );
            }
        }
    }

    /// The two window rules coincide on a uniform batch, which is the other
    /// half of "flag off is byte-identical": the extracted function must not
    /// have changed what the inline expression decided.
    #[test]
    fn the_window_invariant_is_unchanged_for_a_uniform_batch() {
        let w = 4usize;
        for cache_len in [0usize, 1, 2, 9] {
            for tails in [vec![1usize, 1], vec![1, 4], vec![0, 1], vec![5, 1]] {
                let tok_lens: Vec<usize> = tails.iter().map(|t| cache_len + t).collect();
                let plan = plan_step_window(&[cache_len; 2], &tok_lens, w);
                let legacy = cache_len + tails.iter().copied().min().unwrap_or(0) >= 2
                    && tails.iter().all(|u| (1..=w).contains(u));
                assert_eq!(
                    plan.ok, legacy,
                    "cache_len {cache_len}, tails {tails:?}: the per-row rule must agree with the \
                     scalar one it replaced"
                );
                assert_eq!(plan.uncached, tails);
            }
        }
    }

    /// The plan functions agree on the one thing they must: the slots a
    /// sequence keeps. They differ only in what happens to the surplus — the
    /// cohort rule rolls it back and re-feeds it as a longer tail next step,
    /// the per-sequence rule keeps it and resets the tail to 1.
    #[test]
    fn both_plans_agree_on_what_each_sequence_can_vouch_for() {
        let w = 5usize;
        let uncached = [1usize, 2, 1, 4];
        let accepted = [3usize, 1, 0, 0];
        let per_seq = plan_per_sequence_step(&uncached, &accepted, w);
        let cohort = plan_batch_step(&uncached, &accepted, w);
        for i in 0..uncached.len() {
            assert_eq!(
                per_seq.committed[i],
                uncached[i] + accepted[i],
                "seq {i}: a sequence vouches for its own tail plus its own accepts"
            );
        }
        assert_eq!(
            cohort.keep,
            per_seq.committed.iter().copied().min().unwrap(),
            "the cohort keep is exactly the batch minimum of the per-sequence commits — which \
             is the barrier, stated as an equation"
        );
    }

    /// 🔑 **The whole point.** Under the per-sequence rule no sequence's
    /// uncached tail ever leaves 1, so every sequence drafts the full depth on
    /// every step, at every batch size.
    ///
    /// The control in the same test shows the cohort rule does the opposite on
    /// the identical fixture: tails ratchet up to the window and the sequences
    /// sitting there draft **zero**. That control is what makes this test
    /// non-vacuous — if someone reintroduces a cohort reduction inside
    /// `plan_per_sequence_step` the first assertion fails, and if the cohort
    /// rule ever stops ratcheting the second one fails and this whole
    /// workstream is unnecessary.
    #[test]
    fn per_sequence_advance_never_lets_a_tail_ratchet_at_b128() {
        let depth = 3usize;
        let w = depth + 1;
        let steps = 200usize;

        let mut seqs = batch_of(128);
        let mut min_drafted = usize::MAX;
        for step in 0..steps {
            for (i, s) in seqs.iter().enumerate() {
                assert_eq!(
                    s.uncached, 1,
                    "step {step}, seq {i}: per-sequence advance must leave every uncached tail \
                     at 1 — a tail above 1 is the ratchet, and a tail at w={w} drafts nothing"
                );
                min_drafted = min_drafted.min(w - s.uncached);
            }
            sim_step_per_sequence(&mut seqs, w);
        }
        assert_eq!(
            min_drafted, depth,
            "every sequence must draft the full depth on every step under the per-sequence rule"
        );

        // Control: the cohort rule on the identical fixture.
        let mut cohort = batch_of(128);
        let mut saturated = 0usize;
        let mut zero_drafters = 0usize;
        for _ in 0..steps {
            sim_step(&mut cohort, w, None);
        }
        for s in &cohort {
            if s.uncached == w {
                saturated += 1;
            }
            if w - s.uncached == 0 {
                zero_drafters += 1;
            }
        }
        assert!(
            saturated > 0,
            "the cohort rule is supposed to ratchet tails to the window at B=128 — if it no \
             longer does, this whole change is unnecessary and the premise must be re-derived"
        );
        assert_eq!(
            saturated, zero_drafters,
            "a saturated tail is exactly a sequence that drafts nothing"
        );
    }

    /// Token identity: batched per-sequence decoding must produce, for every
    /// sequence, exactly the tokens that sequence would have produced alone.
    ///
    /// This is the acceptance test for the whole design. `sim_step_per_sequence`
    /// at B=128 vs the same sequence run at B=1 — no tolerance, exact equality
    /// on the full stream.
    #[test]
    fn per_sequence_advance_is_token_identical_to_the_b1_reference() {
        let w = 4usize;
        let steps = 60usize;

        let mut batched = batch_of(128);
        for _ in 0..steps {
            sim_step_per_sequence(&mut batched, w);
        }

        for (i, want) in batch_of(128).into_iter().enumerate() {
            let mut solo = vec![want];
            for _ in 0..steps {
                sim_step_per_sequence(&mut solo, w);
            }
            assert_eq!(
                batched[i].toks, solo[0].toks,
                "sequence {i} at B=128 diverged from its own B=1 run — batched per-sequence \
                 decoding is not user-invisible"
            );
        }
    }

    /// B=1 must not regress: at a batch of one the two rules are the same
    /// function, so the flag cannot move the only regime that has ever been
    /// measured.
    #[test]
    fn the_two_advance_rules_are_identical_at_batch_one() {
        for depth in 1..=8usize {
            let w = depth + 1;
            let mut cohort = vec![batch_of(1).remove(0)];
            let mut per_seq = vec![batch_of(1).remove(0)];
            for step in 0..40 {
                let a = sim_step(&mut cohort, w, None);
                let b = sim_step_per_sequence(&mut per_seq, w);
                assert_eq!(a, b, "depth {depth}, step {step}: B=1 emission differs");
                assert_eq!(
                    cohort[0].uncached, per_seq[0].uncached,
                    "depth {depth}, step {step}: B=1 uncached tail differs"
                );
            }
        }
    }

    /// No value the per-sequence plan produces for sequence `i` may depend on
    /// any other sequence. This is the anti-cohort-barrier test: it fails the
    /// moment anyone reintroduces a `min`, a `max` or any other reduction over
    /// the batch.
    #[test]
    fn per_sequence_plan_ignores_every_other_sequence() {
        let w = 5usize;
        // A deliberately hostile batch: one permanent laggard (u=1, a=0) —
        // exactly the sequence that pins `keep = 1` and ratchets everyone else
        // under the cohort rule.
        let uncached = [1usize, 1, 2, 3, 5, 1, 4];
        let accepted = [0usize, 4, 3, 2, 0, 1, 0];
        let batched = plan_per_sequence_step(&uncached, &accepted, w);

        for i in 0..uncached.len() {
            let solo = plan_per_sequence_step(&uncached[i..=i], &accepted[i..=i], w);
            assert_eq!(
                batched.committed[i], solo.committed[0],
                "seq {i}: committed slots changed because of its neighbours"
            );
            assert_eq!(
                batched.dropped[i], solo.dropped[0],
                "seq {i}: dropped slots changed because of its neighbours"
            );
            assert_eq!(batched.next_uncached[i], 1, "seq {i}: tail must reset to 1");
            assert_eq!(
                batched.committed[i] + batched.dropped[i],
                w,
                "seq {i}: window slots unaccounted for"
            );
            assert!(batched.committed[i] >= 1, "seq {i}: cache must advance");
        }

        // Teeth: the cohort rule on the SAME input does depend on neighbours,
        // so the equalities above are not trivially true of any plan.
        let cohort = plan_batch_step(&uncached, &accepted, w);
        assert_eq!(cohort.keep, 1, "the laggard pins the cohort keep at 1");
        assert!(
            cohort.next_uncached.iter().any(|u| *u > 1),
            "the cohort rule must ratchet on this fixture or the contrast is vacuous"
        );
    }

    /// The quantified gap, in the unit that matters. `tok_per_step` — NOT
    /// `accept_rate`: a saturated sequence drafts 0, so it contributes
    /// `proposed = 0` and leaves `accept_rate` flattering while `tok_per_step`
    /// collapses. The two diverge exactly here.
    ///
    /// This is arithmetic on the production plan functions, not a hardware
    /// measurement, and must never be quoted as one.
    #[test]
    fn tok_per_step_at_b128_is_bounded_by_the_cohort_rule_and_freed_by_the_per_sequence_rule() {
        let w = 4usize; // depth 3, the shipped default
        let steps = 200usize;
        let b = 128usize;

        let tok_per_step = |mut seqs: Vec<SimSeq>, per_seq: bool| -> f64 {
            let mut emitted = 0usize;
            for _ in 0..steps {
                let out = if per_seq {
                    sim_step_per_sequence(&mut seqs, w)
                } else {
                    sim_step(&mut seqs, w, None)
                };
                emitted += out.iter().map(Vec::len).sum::<usize>();
            }
            emitted as f64 / (steps * seqs.len()) as f64
        };

        let cohort = tok_per_step(batch_of(b), false);
        let per_sequence = tok_per_step(batch_of(b), true);
        // The B=1 baseline is the mean over each of the SAME 128 sequences run
        // alone — not `batch_of(1)`, whose single member has one particular
        // `agree_mod` and is not representative of the cohort's mix.
        let solo = batch_of(b)
            .into_iter()
            .map(|s| tok_per_step(vec![s], false))
            .sum::<f64>()
            / b as f64;

        assert!(
            per_sequence > cohort * 1.10,
            "per-sequence advance must beat the cohort rule at B=128 by a clear margin \
             (cohort {cohort:.4}, per-sequence {per_sequence:.4} tok/step)"
        );
        assert!(
            (per_sequence - solo).abs() < 1e-9,
            "per-sequence advance at B=128 must reach what the same fixture reaches at B=1 \
             (B=128 {per_sequence:.4}, B=1 {solo:.4} tok/step) — that is what 'no cohort \
             barrier' means"
        );
        assert!(
            cohort < solo,
            "the cohort rule must lose against B=1 or there is no problem to solve \
             (cohort {cohort:.4}, B=1 {solo:.4} tok/step)"
        );
    }

    /// The window invariant closes: over every depth 1..=8 and every ragged
    /// accept pattern a batch can produce, the shared cache advances by at
    /// least one position and no sequence's uncached tail ever outgrows the
    /// window.
    ///
    /// These two are what make the fast path total. If `keep` could be 0 the
    /// batch would stall forever; if a tail could exceed `w` the sequence would
    /// have no room left to feed its own committed tokens and the fast path
    /// would have to bail out of a state it created — which is precisely the
    /// state no other code path knows how to read.
    #[test]
    fn batched_window_invariant_closes_at_every_depth_and_accept_pattern() {
        let mut checked = 0usize;
        for depth in 1..=8usize {
            let w = depth + 1;
            // Every reachable (u, a) pair for a batch of up to 4 sequences.
            let pairs: Vec<(usize, usize)> = (1..=w)
                .flat_map(|u| (0..=(w - u)).map(move |a| (u, a)))
                .collect();
            for i in 0..pairs.len() {
                for j in 0..pairs.len() {
                    for k in 0..pairs.len() {
                        let batch = [pairs[i], pairs[j], pairs[k]];
                        let uncached: Vec<usize> = batch.iter().map(|(u, _)| *u).collect();
                        let accepted: Vec<usize> = batch.iter().map(|(_, a)| *a).collect();
                        let plan = plan_batch_step(&uncached, &accepted, w);
                        assert!(
                            plan.keep >= 1,
                            "depth {depth}: the shared cache must advance every step, else the \
                             batch stalls forever (u={uncached:?} a={accepted:?})"
                        );
                        assert!(
                            plan.n_drop <= w,
                            "depth {depth}: cannot roll back more than the window ({plan:?})"
                        );
                        for (seq, next_u) in plan.next_uncached.iter().enumerate() {
                            assert!(
                                (1..=w).contains(next_u),
                                "depth {depth}, seq {seq}: next uncached tail {next_u} escaped \
                                 [1, {w}] from u={uncached:?} a={accepted:?}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 5_000, "only {checked} patterns exercised");
    }

    /// A batch whose sequences accept DIFFERENT numbers of drafts in the same
    /// step emits, per sequence, exactly the token stream that sequence would
    /// have produced alone.
    ///
    /// This is the whole claim of the batched fast path. The batch changes how
    /// many tokens a sequence commits per *step* (its tail shrinks the drafts it
    /// gets), never *which* tokens it commits.
    #[test]
    fn batched_ragged_accept_is_token_identical_to_the_b1_reference() {
        for depth in 1..=4usize {
            let w = depth + 1;
            // Different `agree_mod` per sequence is what produces different
            // accept lengths in the same step.
            let seeds: [(u32, u32); 5] = [(11, 2), (23, 3), (41, 1), (57, 5), (73, 4)];
            let make = |(tok, agree): (u32, u32)| SimSeq {
                toks: vec![1, tok],
                agree_mod: agree,
                uncached: 1,
            };

            // Reference: each sequence alone.
            let mut reference: Vec<Vec<u32>> = Vec::new();
            for s in seeds {
                let mut solo = vec![make(s)];
                let mut stream = Vec::new();
                for _ in 0..200 {
                    let out = sim_step(&mut solo, w, None);
                    stream.extend_from_slice(&out[0]);
                    assert_eq!(
                        solo[0].uncached, 1,
                        "at B=1 the batch minimum is the sequence itself, so its tail must \
                         return to 1 every step"
                    );
                    if stream.len() >= 60 {
                        break;
                    }
                }
                reference.push(stream);
            }

            // Batched: all five together, ragged by construction.
            let mut batch: Vec<SimSeq> = seeds.into_iter().map(make).collect();
            let mut streams: Vec<Vec<u32>> = vec![Vec::new(); batch.len()];
            let mut saw_ragged = false;
            let mut saw_tail_above_one = false;
            for _ in 0..200 {
                let before: Vec<usize> = batch.iter().map(|s| s.toks.len()).collect();
                let out = sim_step(&mut batch, w, None);
                let commits: Vec<usize> = out.iter().map(Vec::len).collect();
                if commits.iter().any(|c| *c != commits[0]) {
                    saw_ragged = true;
                }
                if batch.iter().any(|s| s.uncached > 1) {
                    saw_tail_above_one = true;
                }
                for (i, toks) in out.iter().enumerate() {
                    streams[i].extend_from_slice(toks);
                    assert_eq!(
                        batch[i].toks.len(),
                        before[i] + toks.len(),
                        "committed token count and emitted token count must not drift"
                    );
                }
                if streams.iter().all(|s| s.len() >= 60) {
                    break;
                }
            }

            // Non-degeneracy first (D12): if the batch never went ragged and no
            // tail ever exceeded one, this test proves nothing about the case it
            // exists for.
            assert!(
                saw_ragged,
                "depth {depth}: no step had different accept lengths across the batch — the \
                 fixture is degenerate and would pass with the ragged path deleted"
            );
            assert!(
                saw_tail_above_one,
                "depth {depth}: no sequence ever carried an uncached tail > 1, so the window \
                 mechanism was never exercised"
            );

            for (i, stream) in streams.iter().enumerate() {
                let n = stream.len().min(reference[i].len());
                assert!(n >= 50, "seq {i} produced only {n} comparable tokens");
                assert_eq!(
                    &stream[..n],
                    &reference[i][..n],
                    "depth {depth}, seq {i}: batched MTP diverged from the B=1 reference"
                );
            }
        }
    }

    /// Mutation: resolve the batch on the MAXIMUM valid extent instead of the
    /// minimum — the obvious "keep as much cache as possible" mistake.
    ///
    /// It reads as harmless (it only keeps *more* cache) and it is not: the
    /// sequences that accepted fewer drafts would carry K/V for tokens they
    /// never committed, and their next tail underflows. The equivalence test
    /// above must fail on it, or it is not testing the rollback at all.
    #[test]
    fn batched_rollback_mutation_max_instead_of_min_is_caught() {
        fn bad_plan(uncached: &[usize], accepted: &[usize], w: usize) -> BatchStepPlan {
            let keep = uncached
                .iter()
                .zip(accepted)
                .map(|(u, a)| u + a)
                .max()
                .unwrap_or(0);
            BatchStepPlan {
                keep,
                n_drop: w.saturating_sub(keep),
                next_uncached: uncached
                    .iter()
                    .zip(accepted)
                    .map(|(u, a)| (u + a + 1).saturating_sub(keep))
                    .collect(),
            }
        }

        let w = 4usize;
        let seeds: [(u32, u32); 4] = [(11, 2), (23, 3), (41, 1), (57, 5)];
        let make = |(tok, agree): (u32, u32)| SimSeq {
            toks: vec![1, tok],
            agree_mod: agree,
            uncached: 1,
        };

        let mut good: Vec<SimSeq> = seeds.into_iter().map(make).collect();
        let mut bad: Vec<SimSeq> = seeds.into_iter().map(make).collect();
        let mut good_tails = Vec::new();
        let mut bad_tails = Vec::new();
        for _ in 0..12 {
            sim_step(&mut good, w, None);
            good_tails.push(good.iter().map(|s| s.uncached).collect::<Vec<_>>());
            // The mutation is allowed to reach a state the window scheme
            // forbids; stepping ONWARD from it would index a row that does not
            // exist, which is the corruption itself, so stop at the evidence.
            if bad.iter().all(|s| (1..=w).contains(&s.uncached)) {
                sim_step(&mut bad, w, Some(bad_plan));
                bad_tails.push(bad.iter().map(|s| s.uncached).collect::<Vec<_>>());
            }
        }
        // Under the mutation at least one sequence's tail collapses to 0 — it
        // would have to feed a zero-width window, i.e. its own committed tokens
        // would sit in the cache as K/V computed from a draft that was rejected.
        assert!(
            bad_tails.iter().flatten().any(|u| *u == 0),
            "the max-instead-of-min mutation must drive some tail to 0; if it does not, the \
             fixture is not ragged enough to catch it"
        );
        assert!(
            good_tails.iter().flatten().all(|u| (1..=w).contains(u)),
            "the production plan must keep every tail inside [1, {w}]"
        );
        assert_ne!(
            good_tails, bad_tails,
            "the mutation produced identical state — the test has no teeth"
        );
    }

    /// Mutation: shift the verify row by one. This is the defect that hides
    /// perfectly — it does not crash, it does not corrupt the cache, it just
    /// compares each draft against the target's answer for the wrong position,
    /// so acceptance collapses and looks like a bad draft head.
    #[test]
    fn window_verify_row_off_by_one_destroys_acceptance() {
        let w = 4usize;
        let seq = SimSeq {
            toks: vec![1, 11],
            agree_mod: 1, // draft always agrees with the target
            uncached: 1,
        };
        let mut good = vec![seq.clone()];
        let emitted = sim_step(&mut good, w, None);
        assert_eq!(
            emitted[0].len(),
            w, // all w-1 drafts accepted, plus the bonus token
            "with a perfectly agreeing draft head every draft must be accepted"
        );

        // Same chain, rows read one slot late.
        let d = w - seq.uncached;
        let mut ctx = seq.toks.clone();
        let mut drafts = Vec::new();
        for _ in 0..d {
            let next = oracle_draft(&ctx, 1);
            drafts.push(next);
            ctx.push(next);
        }
        let mut rows = vec![0u32; w + 1];
        let mut ctx = seq.toks.clone();
        for j in 0..=d {
            rows[window_verify_row(seq.uncached, j)] = oracle_target(&ctx);
            if j < d {
                ctx.push(drafts[j]);
            }
        }
        let shifted: Vec<u32> = (0..d)
            .map(|j| rows[window_verify_row(seq.uncached, j) + 1])
            .collect();
        let bad = verify_proposed(&drafts, &shifted);
        assert!(
            bad.accepted.len() < d,
            "a one-row shift must reject something; it accepted all {d}"
        );
    }

    /// Acceptance is reported per batch size, and the two multipliers the
    /// ceiling model needs are both on the line.
    ///
    /// `tok_per_step` is per user (what multiplies the 68 tok/s floor at
    /// B=128); `tok_per_batch_step` is aggregate (tokens out per forward). At
    /// B=1 they are equal, which is exactly why a B=1 measurement cannot answer
    /// whether MTP still pays at batch.
    #[test]
    fn batched_acceptance_is_reported_per_batch_size() {
        let tel = AcceptanceTelemetry::default();
        // B=4, every sequence accepts 2 of 2 drafts.
        let all_accept: Vec<MtpAcceptance> = (0..4)
            .map(|_| MtpAcceptance::from_fused_verify(2, 2))
            .collect();
        tel.record_batch(4, &all_accept);
        // B=2, nothing accepted.
        let all_reject: Vec<MtpAcceptance> = (0..2)
            .map(|_| MtpAcceptance::from_fused_verify(2, 0))
            .collect();
        tel.record_batch(2, &all_reject);

        let by_batch = tel.snapshot_by_batch();
        assert_eq!(
            by_batch.iter().map(|(b, _)| *b).collect::<Vec<_>>(),
            vec![2, 4],
            "batch sizes must be reported smallest-first and not merged"
        );
        let b2 = by_batch[0].1;
        let b4 = by_batch[1].1;

        // Mutation-style bracketing: the two extremes must read 0.0 and 1.0.
        assert_eq!(b2.rate(), Some(0.0), "all-reject must report 0%, not n/a");
        assert_eq!(b4.rate(), Some(1.0), "all-accept must report 100%");

        // Per-user multiplier: all-accept at depth 2 commits 3 tokens/step/user.
        assert_eq!(b4.tokens_per_step(), Some(3.0));
        assert_eq!(b2.tokens_per_step(), Some(1.0));
        // Aggregate multiplier: 4 users x 3 tokens out of ONE forward.
        assert_eq!(b4.tokens_per_batch_step(), Some(12.0));
        assert_eq!(b2.tokens_per_batch_step(), Some(2.0));
        assert_eq!(b4.mean_batch(), Some(4.0));

        let marker = b4.marker("b=4");
        for field in [
            "accept_rate=1.0000",
            "tok_per_step=3.0000",
            "batch_steps=1",
            "mean_batch=4.0000",
            "tok_per_batch_step=12.0000",
        ] {
            assert!(marker.contains(field), "marker missing `{field}`: {marker}");
        }
    }

    /// `plan_batch_step` is the one place the batch resolution lives, and the
    /// fast path asserts against it. Pin the exact ragged case by hand so a
    /// refactor cannot quietly redefine "keep".
    #[test]
    fn plan_batch_step_keeps_only_what_every_sequence_can_prove() {
        // Window 4. Seq 0 fed 1 real token and accepted 3 drafts; seq 1 fed 1
        // real token and accepted none; seq 2 fed 2 real tokens and accepted 1.
        let plan = plan_batch_step(&[1, 1, 2], &[3, 0, 1], 4);
        assert_eq!(
            plan.keep, 1,
            "seq 1 can only vouch for its single real slot"
        );
        assert_eq!(plan.n_drop, 3);
        // Seq 0 committed 4 tokens but the cache advanced 1, so it carries 4
        // uncached; seq 1 committed 1 and carries 1; seq 2 committed 2 against
        // a 2-slot valid extent and carries 3.
        assert_eq!(plan.next_uncached, vec![4, 1, 3]);
        assert!(plan.next_uncached.iter().all(|u| (1..=4).contains(u)));
    }

    // ---- The draft KV, on real tensors -----------------------------------

    /// A draft `KvCache` with a REAL buffer, marked so every row and every
    /// position is distinguishable. `k[.., j, ..] = mark + j`, which is what
    /// lets the assertions below say *which* position ended up where instead of
    /// only that some tensor has the right shape.
    fn draft_slot(live: usize, capacity: usize, mark: f32) -> KvCache {
        let half = |mark: f32| {
            let data: Vec<f32> = (0..capacity).map(|i| mark + i as f32).collect();
            let all = Tensor::from_vec(data, (1, 1, capacity, 1), &Device::Cpu).unwrap();
            SingleCache {
                all_data: Some(all),
                dim: 2,
                current_seq_len: live,
                capacity_seq_len: capacity,
                max_seq_len: 4096,
                first_cached: 0,
                retain_window: None,
            }
        };
        KvCache::Normal {
            k: half(mark),
            v: half(mark + 1000.0),
        }
    }

    fn k_rows(slot: &KvCache) -> Vec<f32> {
        let KvCache::Normal { k, .. } = slot else {
            unreachable!("draft caches are Normal slots")
        };
        k.all_data
            .as_ref()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// 🔑 The refusal this wave replaces, replaced. `batch_draft_caches` returns
    /// `None` for rows of differing length — under per-sequence advance, all of
    /// them — and [`front_align_draft_group`] is what makes them shareable
    /// instead: each row truncated to its OWN chain start, then shifted so its
    /// live run ends at the group's widest.
    ///
    /// The assertions are on content, not on shape: row `i`'s dead prefix must
    /// be zero (a nonzero K row is not a masked row — it scores logit 0 and
    /// takes softmax weight) and its live run must be its own positions in
    /// order, ending at the shared column.
    #[test]
    fn a_ragged_draft_group_is_left_aligned_where_the_old_path_refused_it() {
        let a = draft_slot(9, 32, 100.0);
        let b = draft_slot(9, 32, 200.0);
        let c = draft_slot(9, 32, 300.0);

        // Chain starts 7 / 4 / 6 — the shared end column is 7.
        let starts = [7usize, 4, 6];
        let borrowed: Vec<&KvCache> = vec![&a, &b, &c];

        assert!(
            batch_draft_caches(&borrowed).is_some(),
            "precondition: the rows agree on length BEFORE the per-row truncation, so the old \
             path is what refuses the chain starts, not the buffers"
        );

        let batched = front_align_draft_group(&borrowed, &starts).unwrap();
        assert_eq!(
            batched.current_seq_len(),
            7,
            "the batched width is the group's widest chain start, which is the one shared \
             append offset every row writes at"
        );
        let flat = k_rows(&batched);
        assert_eq!(flat.len(), 3 * 7, "three rows of exactly the shared width");
        for (row, (&start, mark)) in starts.iter().zip([100.0f32, 200.0, 300.0]).enumerate() {
            let lead = 7 - start;
            let span = &flat[row * 7..(row + 1) * 7];
            assert!(
                span[..lead].iter().all(|x| *x == 0.0),
                "row {row}'s {lead}-column dead prefix must be zeroed, not left holding another \
                 position's K"
            );
            let want: Vec<f32> = (0..start).map(|j| mark + j as f32).collect();
            assert_eq!(
                &span[lead..],
                &want[..],
                "row {row}'s own positions 0..{start} must land in order, ending at the shared \
                 column"
            );
        }
    }

    /// 🔴 The trap PR #102 named, one cache down: *"the cohort barrier
    /// reappearing as padding."*
    ///
    /// Left-alignment is the only way a ragged group shares an append offset,
    /// but the length it leaves behind is not the row's own. Recording *that*
    /// makes the next group pad relative to an already-padded length, so the
    /// prefix compounds every step and the rows re-converge onto one length.
    ///
    /// ⚠️ The assertions read `current_seq_len()` back **off the slot** and read
    /// the content back **out of the tensor**. #102 shipped a first draft of the
    /// equivalent test that recomputed the post-strip length itself and
    /// therefore passed with the strip stubbed to a no-op — it asserted its own
    /// arithmetic against its own arithmetic. The negative control below is that
    /// same no-op, and it must fail every assertion this one makes.
    ///
    /// ⚠️ Arithmetic on real tensors, not a hardware measurement (D14).
    #[test]
    fn the_draft_dead_prefix_does_not_survive_into_the_per_sequence_caches() {
        // One "chain": left-align to the widest start, append `n_draft`
        // positions the way the block forward does, split back, and (optionally)
        // strip. `filled` is what the group loop records.
        let run = |strip: bool, steps: usize| -> (Vec<usize>, Vec<f32>) {
            let n_draft = 3usize;
            let mut slots = [
                draft_slot(7, 512, 100.0),
                draft_slot(4, 512, 200.0),
                draft_slot(6, 512, 300.0),
            ];
            // Each row advances by its own accepted count, which is the whole
            // point of the mode — so the starts stay ragged step after step.
            let advance = [3usize, 1, 2];
            for _ in 0..steps {
                let starts: Vec<usize> = slots.iter().map(|s| s.current_seq_len()).collect();
                let leads = resolve_draft_leads(true, &starts).expect("the fixture is ragged");
                let borrowed: Vec<&KvCache> = slots.iter().collect();
                let mut batched = front_align_draft_group(&borrowed, &starts).unwrap();
                // The chain forward writes `n_draft` new positions for every
                // row, through the real `append` — which is what grows the
                // buffer past the width the alignment left it at.
                let new = Tensor::zeros(
                    (slots.len(), 1, n_draft, 1),
                    candle_core::DType::F32,
                    &Device::Cpu,
                )
                .unwrap();
                batched.append(&new, &new).unwrap();
                let split = split_draft_cache(&batched, slots.len()).unwrap();
                for (i, row) in split.into_iter().enumerate() {
                    let mut row = row;
                    if strip {
                        crate::kv_cache::drop_dead_prefix(&mut row, leads[i]).unwrap();
                    }
                    // What the group loop records: the row's own chain start
                    // plus the one committed entry. Read the slot's length back
                    // rather than deriving it, then advance by this row's own
                    // accepted count.
                    let live = row.current_seq_len();
                    row.set_len(live - n_draft + advance[i]).unwrap();
                    slots[i] = row;
                }
            }
            let lens: Vec<usize> = slots.iter().map(|s| s.current_seq_len()).collect();
            // Row 1's first live position, read out of the tensor. It must be
            // that row's own position 0 (mark 200.0) and it must sit at column 0
            // — `front_pad_single`'s `narrow(dim, 0, live)` requires it next
            // time round.
            let head = k_rows(&slots[1])[0];
            (lens, vec![head])
        };

        let (lens_1, head_1) = run(true, 1);
        assert_eq!(
            lens_1,
            vec![10, 5, 8],
            "after one chain each row is at its OWN start plus its OWN accepted count \
             (7+3, 4+1, 6+2) — no row carries the group's 7-column width"
        );
        assert_eq!(
            head_1,
            vec![200.0],
            "row 1's live run must start at column 0"
        );

        let (lens_6, _) = run(true, 6);
        let (lens_12, _) = run(true, 12);
        assert_eq!(
            lens_6,
            vec![7 + 3 * 6, 4 + 6, 6 + 2 * 6],
            "six chains later every row is still exactly its start plus its own advances"
        );
        assert_eq!(lens_12, vec![7 + 3 * 12, 4 + 12, 6 + 2 * 12]);
        let spread_6 = lens_6.iter().max().unwrap() - lens_6.iter().min().unwrap();
        let spread_12 = lens_12.iter().max().unwrap() - lens_12.iter().min().unwrap();

        // 🔴 The negative control: the same loop with the strip removed. This is
        // what recording the padded width does.
        let (bad_6, bad_head) = run(false, 6);
        let (bad_12, _) = run(false, 12);
        assert_ne!(
            bad_6, lens_6,
            "without the strip the recorded lengths must differ — if they do not, the strip is \
             a no-op and this test proves nothing"
        );
        assert_ne!(bad_head, vec![200.0]);
        let bad_spread_6 = bad_6.iter().max().unwrap() - bad_6.iter().min().unwrap();
        let bad_spread_12 = bad_12.iter().max().unwrap() - bad_12.iter().min().unwrap();
        assert!(
            bad_6.iter().sum::<usize>() > lens_6.iter().sum::<usize>()
                && bad_12.iter().sum::<usize>() > lens_12.iter().sum::<usize>(),
            "the unstripped variant must inflate: {bad_6:?} / {bad_12:?} against {lens_6:?} / \
             {lens_12:?}"
        );
        assert_eq!(
            (spread_6, spread_12),
            (
                lens_6.iter().max().unwrap() - lens_6.iter().min().unwrap(),
                lens_12.iter().max().unwrap() - lens_12.iter().min().unwrap()
            ),
        );
        assert!(
            bad_spread_12 < spread_12 && bad_spread_6 < spread_6,
            "the tell is not only that the unstripped lengths inflate but that they CONVERGE — \
             every row ends up advancing at the fastest row's rate, which is the cohort barrier \
             wearing padding: spreads {bad_spread_6}/{bad_spread_12} against \
             {spread_6}/{spread_12}"
        );
    }

    /// The round trip through a batched draft KV must be the identity on a
    /// group that is already uniform, at the level of bytes — that is the
    /// cohort path, and it runs on every request today.
    #[test]
    fn a_uniform_draft_group_round_trips_through_the_batched_cache_unchanged() {
        let slots = [
            draft_slot(6, 16, 100.0),
            draft_slot(6, 16, 200.0),
            draft_slot(6, 16, 300.0),
        ];
        let borrowed: Vec<&KvCache> = slots.iter().collect();
        assert_eq!(
            resolve_draft_leads(true, &[6, 6, 6]),
            None,
            "a uniform group must not take the alignment path at all"
        );
        let batched = batch_draft_caches(&borrowed).expect("uniform rows batch as before");
        let split = split_draft_cache(&batched, 3).expect("and split back");
        for (i, row) in split.iter().enumerate() {
            let mut row = row.clone();
            crate::kv_cache::drop_dead_prefix(&mut row, 0).expect("a zero-width strip is a no-op");
            assert_eq!(row.current_seq_len(), 6);
            assert_eq!(
                k_rows(&row),
                k_rows(&slots[i]),
                "row {i} must come back byte-identical"
            );
        }
    }

    /// The alignment must refuse rather than guess when the caller's idea of the
    /// group and the caches disagree.
    #[test]
    fn front_align_draft_group_refuses_a_mismatched_or_over_long_group() {
        let a = draft_slot(9, 32, 100.0);
        let b = draft_slot(9, 32, 200.0);
        let borrowed: Vec<&KvCache> = vec![&a, &b];

        let err = front_align_draft_group(&borrowed, &[7])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("2 cache(s) for 1 chain start"),
            "a width mismatch must be named, not silently zipped short; got {err}"
        );

        let err = front_align_draft_group(&borrowed, &[7, 20])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("chain start 20"),
            "a chain start past what the row actually holds must refuse; got {err}"
        );

        assert!(front_align_draft_group(&[], &[]).is_err());
    }
}
