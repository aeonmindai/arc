//! DeepSeek V4 hybrid attention dispatch — Standard / CSA / HCA.
//!
//! This module is the **call-site** of V4 attention: it ports the model's own
//! reference `Attention.forward` (`inference/model.py`) into `mistralrs-core`
//! so the V4 model's `Attention::forward` can route through it.
//!
//! ## Algorithm (faithful to the reference `Attention.forward` / `sparse_attn`)
//!
//! V4 attention is a **single online softmax over a union of key sets**, not a
//! learned gate and not a blend of separate softmaxes:
//!
//! - **Standard (`compress_ratio == 0`, layers 0 and 1 plus slot 43, the MTP
//!   block — NOT layer 42, which is CSA)**: sliding-window
//!   attention over the last `window` raw tokens + `attn_sink`. NOT dense
//!   causal! Both production references window ratio-0 layers exactly like the
//!   raw branch of CSA/HCA, just with no compressed branch:
//!     * SGLang `deepseek_v4_backend.py` (`forward`): `compress_ratio == 0`
//!       leaves `extra_k_cache = None` and attends with
//!       `indices=swa_page_indices` (the last `SWA_WINDOW = 128` positions,
//!       `get_swa_page_indices`), `attn_sink` asserted present;
//!     * vLLM `models/deepseek_v4/attention.py`: `swa_only = compress_ratio <=
//!       1`; SWA-only layers have no own KV cache and decode/prefill attend
//!       over the SWA indices alone (`top_k = 0, N = 0`).
//!   Running these layers dense-causal (the pre-fix behavior) feeds
//!   window-trained heads key/query relative distances they never saw in
//!   training; on hardware this collapsed generation into repetition loops the
//!   moment the context crossed 128 tokens, regardless of the compressed
//!   branch (`ARC_V4_WINDOW_ONLY` made no difference — layers 0/1/43 were the
//!   ones polluting the stream).
//! - **CSA (`compress_ratio == 4`) / HCA (`compress_ratio == 128`)**: one
//!   softmax over `[raw sliding-window KV ++ compressed KV]`, plus the per-head
//!   `attn_sink` as an extra denominator column (supplied via
//!   `sdpa_params.sinks`). The reference selects a sparse top-k subset of keys
//!   (`topk_idxs`); here we attend **densely** over the union with the
//!   equivalent causal masking:
//!     * raw branch: a banded causal **sliding-window** mask (each query sees
//!       the last `window` raw tokens up to and including itself);
//!     * compressed branch: a **causal-over-blocks** mask (query `i` sees
//!       compressed block `b` iff `b < (i+1)/ratio`).
//!
//!   Dense-over-the-union is **mathematically a superset** of the reference's
//!   sparse selection — HCA has no top-k at all, and CSA's top-512 selects
//!   *every* compressed entry whenever `ctx <= index_topk * ratio (= 2048)`. So
//!   this is exact for chat-length context and a correct (just slower) superset
//!   beyond it. The Indexer + sparse-gather kernel is a long-context **speed**
//!   layer to add on top — not required for correctness.
//!
//! ## Shape contract
//!
//! Inputs:
//!   - `q`: `[B, n_heads, T_q, D]` — query after RoPE + per-head RMSNorm.
//!   - `k`, `v`: `[B, 1, T_k, D]` — raw K/V after RoPE (V4 MLA: `v == k`).
//!     `T_k` is the full cached sequence; the current query block is its last
//!     `T_q` tokens.
//!   - `compressed_kv`: `[B, 1, T_c, D]` — compressed (distant-context) KV with
//!     compress-θ RoPE already applied at the strided compressed positions, or
//!     `None` (Standard layers, or history shorter than one `ratio` block).
//!   - `attention_mask`: an **additive** mask over the raw keys, rank 2
//!     `[T_q, T_k]` or rank 4 `[B, H, T_q, T_k]` (a `[1, 1]` tensor is the
//!     flash-attn placeholder and is ignored). It is *folded into* the mask
//!     this module builds, never replaced by it and never dropped — see
//!     `compose_caller_mask`.
//!
//! Returns `[B, n_heads, T_q, D]` matching plain SDPA's output shape so the
//! caller's post-attention inverse-RoPE + grouped `wo_a`/`wo_b` chain works
//! as-is.
//!
//! ## Multi-sequence correctness
//!
//! Every mask this module can compute — raw causality, the sliding window,
//! compressed-block causality — is derived from *absolute positions*, so it is
//! per-query-row and holds identically for every sequence in a batch. What it
//! cannot see is which columns are padding, which is why the caller's mask is
//! composed in rather than discarded. Because the composed mask is always
//! passed to `Sdpa::run_attention`, the call can never be diverted onto
//! `sinks_attn`'s varlen backend, which has no mask parameter (see the routing
//! comment in `attention/backends/sinks.rs`).

use candle_core::{DType, Result, Tensor};
use mistralrs_quant::MatMul;

use super::deepseek4::CompressRatio;
use crate::attention::SdpaParams;
use crate::layers::Sdpa;
use crate::pipeline::text_models_inputs_processor::FlashParams;

/// Kill-switch for the absorbed-MLA decode path (`ARC_V4_NO_ABSORBED_DECODE=1`
/// restores the pre-fix `Sdpa.run_attention` repeat_kv expansion for on-GPU
/// A/B triage, mirroring `ARC_FORCE_NAIVE_SDPA`).
fn absorbed_decode_disabled() -> bool {
    use std::sync::OnceLock;
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        matches!(
            std::env::var("ARC_V4_NO_ABSORBED_DECODE").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Absorbed-MLA decode: single-token (`t_q == 1`), single-KV-head attention
/// computed directly in the shared latent space.
///
/// V4's *weights* are already absorbed — the fused `wkv` projection means the
/// 512-d MQA KV head IS the latent (V = K; there is no `W_UK`/`W_UV` left to
/// fold, unlike V2/V3's `kv_b_proj`). What was still naive is the *runtime*:
/// `Sdpa.run_attention` → `sinks_attn` has no fused kernel at head_dim=512
/// (flash-sinks supports only {64..256}), so it falls to the unfused path and
/// `repeat_kv`-expands K and V to all `n_heads` — materializing
/// `2 * n_heads * T_k * head_dim` elements per layer per decode step just to
/// feed a batched GEMM 64 identical copies of K.
///
/// Here the heads are folded into the GEMM M-dimension instead, so K/V are
/// each read exactly once (`q_latent · kv_latent^T`):
///
/// ```text
/// scores = (Q  [B, 1, H, D]) @ (K^T [B, 1, D, T]) -> [B, 1, H, T]
///          softmax_with_sinks over [B, H, 1, T] (per-head sink + mask)
/// out    = (W  [B, 1, H, T]) @ (V   [B, 1, T, D]) -> [B, 1, H, D]
/// ```
///
/// This is a mathematical identity with the repeat_kv path — the same dot
/// products in the same order — so outputs match near-exactly (pinned by
/// `absorbed_decode_matches_repeat_kv_reference` below and by every decode
/// test in this module, all of which now route through this path).
fn absorbed_mqa_decode(
    q: &Tensor,    // [B, H, 1, D]
    k: &Tensor,    // [B, 1, T_k, D]
    v: &Tensor,    // [B, 1, T_k, D]
    mask: &Tensor, // additive, broadcastable over [B, H, 1, T_k]
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b, h, t_q, d) = q.dims4()?;
    debug_assert_eq!(t_q, 1);
    let t_k = k.dim(2)?;
    let sinks = sdpa_params
        .sinks
        .as_ref()
        .expect("absorbed_mqa_decode requires sinks (caller-gated)");

    // [B, H, 1, D] -> [B, 1, H, D]: at t_q == 1 this is a pure reshape of the
    // same contiguous buffer (no data movement).
    let q_lat = q.contiguous()?.reshape((b, 1, h * t_q, d))?;
    let att = MatMul.matmul_affine_mul(&q_lat, &k.t()?, sdpa_params.softmax_scale.into())?;
    // Per-head layout for the sinks softmax; contiguous, so reshape is free.
    let att = att.reshape((b, h, t_q, t_k))?;
    // Same sink pre-shaping as `sinks_attn_cpu`: [1, H, 1, 1] F32 -> [H] in
    // the logits dtype.
    let sinks = sinks.flatten_all()?.to_dtype(att.dtype())?;
    let att = mistralrs_quant::softmax_with_sinks(&att, &sinks, Some(mask))?;
    let att = att.to_dtype(v.dtype())?;
    // Weighted sum in the latent space: [B, 1, H, T] @ [B, 1, T, D].
    let out = MatMul.matmul(&att.reshape((b, 1, h * t_q, t_k))?, v)?;
    out.reshape((b, h, t_q, d))
}

/// Fold a caller-supplied additive `attention_mask` into the locally built
/// union mask.
///
/// The local mask (`[1, 1, t_q, n_keys]`) encodes only what *this* module
/// knows: raw causality, the sliding window, and compressed-block causality.
/// Everything the *caller* knows — padding columns in a ragged batch, the
/// fixed-width graph-decode length mask, any custom bias — lives in
/// `attention_mask` and used to be thrown away here, so padded/unwritten
/// positions polluted every batched request (reference audit §1(e)).
///
/// Additive masks compose by addition (`0` keeps, `-inf` kills, and `-inf`
/// wins over `0`), so folding is a broadcast add. The only work is lining up
/// the key axis: the caller's mask spans the **raw** keys `[0, t_k)`, while the
/// union axis is `[raw ++ compressed]` of width `n_keys = t_k + t_c`. The
/// caller has no opinion about compressed blocks (they are not tokens in its
/// sequence; their causality is `comp_valid`), so the compressed columns are
/// padded with `0`.
///
/// A width that is neither `t_k_full` nor `t_k_full + t_c` is a contract
/// violation and is reported rather than dropped — silently ignoring a
/// mismatched mask is the exact failure mode this function exists to end.
///
/// `raw_base`/`t_k` describe the raw-key narrowing the caller has already
/// applied to K and V (see the narrowing block in [`dsv4_attention`]): the
/// local mask spans raw keys `[raw_base, raw_base + t_k)`, while the caller's
/// mask always spans the *full* raw cache `[0, t_k_full)`. The caller's mask is
/// therefore sliced to the same window before folding. When nothing was
/// narrowed (`raw_base == 0 && t_k == t_k_full`) the slice is the identity and
/// this behaves exactly as it did before.
fn compose_caller_mask(
    local: &Tensor,
    caller: &Tensor,
    t_q: usize,
    t_k_full: usize,
    raw_base: usize,
    t_k: usize,
) -> Result<Tensor> {
    // `CausalMasker::make_causal_mask_matrix` returns a `[1, 1]` zero
    // placeholder when built with flash-attn on CUDA (there the kernel applies
    // causality itself). It carries no information, so folding it is a no-op.
    if caller.elem_count() == 1 {
        return Ok(local.clone());
    }
    let n_keys = local.dim(3)?;
    let caller = caller.to_dtype(local.dtype())?;
    let caller = match caller.rank() {
        2 => {
            let (q_rows, k_cols) = caller.dims2()?;
            caller.reshape((1, 1, q_rows, k_cols))?
        }
        4 => caller,
        rank => {
            return Err(candle_core::Error::Msg(format!(
                "dsv4_attention: attention_mask must be rank 2 [t_q, t_k] or rank 4 \
                 [b, h, t_q, t_k], got rank {rank} with dims {:?}",
                caller.dims()
            )))
        }
    };
    let q_rows = caller.dim(2)?;
    if q_rows != t_q && q_rows != 1 {
        return Err(candle_core::Error::Msg(format!(
            "dsv4_attention: attention_mask query axis is {q_rows}, expected {t_q} (or 1 to \
             broadcast); dims {:?}",
            caller.dims()
        )));
    }
    let k_cols = caller.dim(3)?;
    // Compressed-column count: the local mask is `[raw(t_k) ++ compressed(t_c)]`.
    let t_c = n_keys - t_k;
    let caller = if k_cols == t_k_full + t_c {
        // Full union mask: slice its raw half to the retained window and keep
        // its compressed half as-is.
        let raw = caller.narrow(3, raw_base, t_k)?;
        if t_c == 0 {
            raw
        } else {
            let comp = caller.narrow(3, t_k_full, t_c)?;
            Tensor::cat(&[&raw, &comp], 3)?
        }
    } else if k_cols == t_k_full {
        // Raw-only mask: slice to the retained window, then neutral (0) over
        // the compressed columns.
        let raw = caller.narrow(3, raw_base, t_k)?;
        if t_c == 0 {
            raw
        } else {
            let (b, h) = (caller.dim(0)?, caller.dim(1)?);
            let pad = Tensor::zeros((b, h, q_rows, t_c), caller.dtype(), caller.device())?;
            Tensor::cat(&[&raw, &pad], 3)?
        }
    } else {
        return Err(candle_core::Error::Msg(format!(
            "dsv4_attention: attention_mask key axis is {k_cols}, expected the raw cache width \
             {t_k_full} or the full union width {}; dims {:?}",
            t_k_full + t_c,
            caller.dims()
        )));
    };
    local.broadcast_add(&caller.contiguous()?)
}

/// The span of raw keys any query row in this block can reach, as
/// `(base, len)` into the full raw cache `[0, t_k_full)`.
///
/// Query row `r` (`r` in `[0, t_q)`) sits at absolute position `q0 + r` where
/// `q0 = t_k_full - t_q`, and attends raw key `j` iff
/// `q0 + r - window < j <= q0 + r`. Taking the union over `r`:
///
/// * the largest reachable `j` is `q0 + t_q - 1 = t_k_full - 1` (row `t_q-1`
///   attending its own position);
/// * the smallest is `q0 - window + 1` (row `0` at its window's far edge).
///
/// So the span is the trailing `t_q + window - 1` keys, clamped to the cache.
/// Every key before `base` is `-inf` on **every** row, which is why dropping
/// them is an identity rather than an approximation.
pub(crate) fn raw_keep_span(t_q: usize, window: usize, t_k_full: usize) -> (usize, usize) {
    let keep = (t_q + window - 1).min(t_k_full);
    (t_k_full - keep, keep)
}

/// Per-call configuration for V4 hybrid attention.
#[derive(Debug, Clone, Copy)]
pub struct Dsv4AttentionConfig {
    /// Per-layer compress ratio (0/4/128 → `Standard`/`Csa`/`Hca`).
    pub compress_ratio: CompressRatio,
    /// Sliding-window size for the always-on local branch (V4 default 128).
    pub sliding_window: usize,
    /// How many raw keys the caller has already dropped from the front of `k`
    /// / `v`, i.e. the absolute position of `k[.., 0, ..]`.
    ///
    /// `0` (the default everywhere except the packed-KV decode path) means `k`
    /// is the whole cached sequence and the absolute query position is inferred
    /// from its length, which is what this module did unconditionally before.
    ///
    /// It exists because the absolute position is the *only* thing this module
    /// needs the dropped prefix for: query row `r` sits at `q0 + r` with
    /// `q0 = raw_prefix + k.dim(2) - t_q`. Passing it lets the caller hand over
    /// just the reachable span — which is what makes the FP8 KV dequant
    /// `O(window)` per decode step instead of `O(context)` (see
    /// [`crate::models::dsv4_kv_fp8`]) — without corrupting the sliding-window
    /// mask or the compressed-block causality threshold.
    ///
    /// The caller may only drop keys this module would have masked out anyway:
    /// `raw_prefix` above [`raw_keep_span`]'s base is a hard error, not a
    /// silent truncation.
    pub raw_prefix: usize,
}

/// V4 hybrid attention dispatch — a single softmax over the union of the raw
/// sliding-window KV and the compressed KV, with the per-head `attn_sink`
/// (carried in `sdpa_params.sinks`) as an extra softmax-denominator column.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    compressed_kv: Option<&Tensor>,
    attention_mask: Option<&Tensor>,
    flash_params: &FlashParams,
    sdpa_params: &SdpaParams,
    cfg: Dsv4AttentionConfig,
) -> Result<Tensor> {
    // ---- Standard layers: sliding-window + sink, same raw branch as CSA/HCA
    // (see module docs — the reference SWA-onlys ratio-0 layers; dense causal
    // here was the long-context collapse). ARC_V4_STANDARD_DENSE=1 restores
    // the pre-fix dense-causal behavior for on-GPU A/B triage only.
    if cfg.compress_ratio == CompressRatio::Standard
        && std::env::var_os("ARC_V4_STANDARD_DENSE").is_some()
    {
        return Sdpa.run_attention(q, k, v, attention_mask, Some(flash_params), sdpa_params);
    }
    // Standard layers have no compressor; never let a stray compressed branch
    // through (the caller's `Attention::compressed_kv` already returns `None`).
    let compressed_kv = if cfg.compress_ratio == CompressRatio::Standard {
        None
    } else {
        compressed_kv
    };

    let (_b, _h, t_q, _d) = q.dims4()?;
    let t_k_given = k.dim(2)?;
    // `k` spans absolute positions `[raw_prefix, t_k_full)`; `raw_prefix == 0`
    // (every path but packed-KV decode) makes this the cache length, exactly as
    // before.
    let t_k_full = cfg.raw_prefix + t_k_given;
    let window = cfg.sliding_window.max(1);
    let dev = q.device();

    // Absolute positions. The current query block is the last `t_q` tokens of
    // the sequence, so query row `r` is at position `q0 + r` with
    // `q0 = t_k_full - t_q` (holds for prefill `t_q == t_k_full` and decode
    // `t_q == 1`).
    let q0 = t_k_full
        .checked_sub(t_q)
        .ok_or_else(|| candle_core::Error::Msg(format!(
            "dsv4_attention: {t_q} query rows against only {t_k_full} keys \
             (raw_prefix {} + {t_k_given} given)",
            cfg.raw_prefix
        )))?;

    // ---- Raw working-set narrowing ----------------------------------------
    // Only the trailing `t_q + window - 1` raw keys are reachable by ANY query
    // row in this block: row `r` sits at absolute position `q0 + r` and attends
    // raw key `j` iff `q0 + r - window < j <= q0 + r`, so over `r` in
    // `[0, t_q)` the union of reachable `j` is `[q0 - window + 1, t_k_full)` —
    // exactly `t_q + window - 1` columns. Every earlier column is `-inf` on
    // every row, so dropping it is an identity, not an approximation.
    //
    // It is worth dropping because the cost of carrying it is not the mask: it
    // is the `Tensor::cat` below, which copies the whole raw cache (twice, once
    // for K and once for V) on every decode step, and the scores GEMM over
    // every one of those columns. At 2048 ctx a decode step (`t_q == 1`) goes
    // from 2048 raw columns to `window` = 128. Prefill (`t_q == t_k_full`) is
    // untouched — there the reachable union is the whole cache.
    //
    // NOTE (wave33/wave43): this is the *read* side, and `cfg.raw_prefix` is
    // how a caller opts into doing part of it itself — the packed-KV decode
    // path narrows before it reconstructs, so it only ever dequantizes the span
    // computed here. The cache itself is still grown to the full sequence
    // (`kv_cache::NormalCache`); capping the *store* at `window` — what SGLang's
    // DSV4 pool does, charging raw KV at `swa_full_tokens_ratio`
    // (`model_executor/pool_configurator.py:397`) — is now unblocked by
    // `raw_prefix` but is a separate change.
    let (raw_base, keep) = raw_keep_span(t_q, window, t_k_full);
    let rel_base = raw_base.checked_sub(cfg.raw_prefix).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "dsv4_attention: caller dropped {} raw keys but rows in this block \
             reach back to absolute key {raw_base}; the dropped keys are NOT \
             unreachable and the window mask would be wrong",
            cfg.raw_prefix
        ))
    })?;
    let (k_owned, v_owned) = if rel_base == 0 && keep == t_k_given {
        (None, None)
    } else {
        (
            Some(k.narrow(2, rel_base, keep)?.contiguous()?),
            Some(v.narrow(2, rel_base, keep)?.contiguous()?),
        )
    };
    let k = k_owned.as_ref().unwrap_or(k);
    let v = v_owned.as_ref().unwrap_or(v);
    let t_k = keep;

    // ---- Raw sliding-window branch mask: [t_q, t_k] -----------------------
    // query r attends raw key j iff (q0+r-window < j <= q0+r): causal AND
    // within the trailing `window` tokens. The diagonal (j == q0+r) is always
    // valid, so no query row is fully masked (no softmax NaN). `kp` carries the
    // ABSOLUTE position of each retained key, so the comparison against `qp` is
    // unchanged by the narrowing above.
    let kp = Tensor::arange(raw_base as u32, (raw_base + t_k) as u32, dev)?
        .to_dtype(DType::F32)?
        .reshape((1, t_k))?;
    let qp = Tensor::arange(q0 as u32, (q0 + t_q) as u32, dev)?
        .to_dtype(DType::F32)?
        .reshape((t_q, 1))?;
    let causal = kp.broadcast_le(&qp)?;
    let lower = (&qp - window as f64)?;
    let in_window = kp.broadcast_gt(&lower)?;
    let raw_valid = (causal * in_window)?; // [t_q, t_k] (u8)

    // ---- Build the union key set + validity mask. -------------------------
    let (k_cat, v_cat, valid) = match compressed_kv {
        Some(comp) => {
            let t_c = comp.dim(2)?;
            // Compressed branch mask: query r attends compressed block b iff
            // b < (q0+r+1)/ratio (causal over fully-completed ratio-blocks).
            let ratio = cfg.compress_ratio.ratio();
            let bp = Tensor::arange(0u32, t_c as u32, dev)?
                .to_dtype(DType::F32)?
                .reshape((1, t_c))?;
            let threshold = ((&qp + 1.0)? / ratio as f64)?.floor()?; // [t_q, 1]
            let comp_valid = bp.broadcast_lt(&threshold)?; // [t_q, t_c] (u8)
            let valid = Tensor::cat(&[&raw_valid, &comp_valid], 1)?;
            let k_cat = Tensor::cat(&[k, comp], 2)?.contiguous()?;
            let v_cat = Tensor::cat(&[v, comp], 2)?.contiguous()?;
            (k_cat, v_cat, valid)
        }
        None => (k.clone(), v.clone(), raw_valid),
    };

    // Boolean validity → additive mask (0 / -inf), broadcast over batch+heads.
    let n_keys = valid.dim(1)?;
    let zeros = Tensor::zeros((t_q, n_keys), q.dtype(), dev)?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (t_q, n_keys), dev)?.to_dtype(q.dtype())?;
    let mask = valid
        .where_cond(&zeros, &neg_inf)?
        .reshape((1, 1, t_q, n_keys))?;

    // Fold in whatever the caller knows that this module cannot see: padding
    // columns in a ragged batch, the graph-decode length mask, custom bias.
    // Dropping it (the pre-fix behavior) let one sequence's padding vote in
    // its neighbours' softmax on every batched request.
    let mask = match attention_mask {
        Some(caller) => compose_caller_mask(&mask, caller, t_q, t_k_full, raw_base, t_k)?,
        None => mask,
    };

    // ---- Absorbed-MLA decode (stage-2 perf) -------------------------------
    // Single-token decode with one KV head and sinks present (the deployment
    // configuration on every V4 layer) computes attention directly in the
    // latent space — no per-head K/V materialization (see
    // `absorbed_mqa_decode`). The sinks gate keeps no-sink callers (synthetic
    // fixtures) on the flash-attn-capable Sdpa dispatch unchanged.
    if t_q == 1 && k_cat.dim(1)? == 1 && sdpa_params.sinks.is_some() && !absorbed_decode_disabled()
    {
        return absorbed_mqa_decode(q, &k_cat, &v_cat, &mask, sdpa_params);
    }

    // Always `Some(&mask)`: V4 owns its causality (raw window ∧ compressed
    // block-causality ∧ the caller's mask), and `sinks_attn` only takes the
    // mask-free varlen path when `mask.is_none()` — so this call can never be
    // routed through a backend that would drop it.
    Sdpa.run_attention(
        q,
        &k_cat,
        &v_cat,
        Some(&mask),
        Some(flash_params),
        sdpa_params,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn empty_flash_params() -> FlashParams {
        FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: HashMap::new(),
            cumulative_seqlens_k: HashMap::new(),
            causal: false,
        }
    }

    /// `FlashParams` carrying `cu_seqlens` for the CPU device — the shape the
    /// engine actually produces on a flash-attn build (`inputs_processor.rs`
    /// populates `cumulative_seqlens_{q,k}` for every unique device whenever
    /// `using_flash_attn()`), and the trigger for `sinks_attn`'s varlen route.
    /// Lengths are the PADDED per-sequence lengths, matching the engine.
    fn varlen_flash_params(seqlens_q: &[u32], seqlens_k: &[u32], dev: &Device) -> FlashParams {
        let cumsum = |lens: &[u32]| -> Tensor {
            let mut acc = 0u32;
            let mut out = vec![0u32];
            for l in lens {
                acc += l;
                out.push(acc);
            }
            Tensor::from_vec(out, lens.len() + 1, dev).unwrap()
        };
        let mut q_map = HashMap::new();
        let mut k_map = HashMap::new();
        q_map.insert(dev.location(), cumsum(seqlens_q));
        k_map.insert(dev.location(), cumsum(seqlens_k));
        FlashParams {
            max_q: seqlens_q.iter().copied().max().unwrap_or(0),
            max_k: seqlens_k.iter().copied().max().unwrap_or(0),
            cumulative_seqlens_q: q_map,
            cumulative_seqlens_k: k_map,
            causal: true,
        }
    }

    /// Additive `[b, 1, t, t]` mask: causal, plus `-inf` on the leading
    /// `left_pad[i]` key columns of batch row `i`. Left padding is the case
    /// causality alone cannot cover — the padded columns sit *before* every
    /// real query, squarely inside the causal (and sliding-window) set — so it
    /// is the honest probe for "is the caller's mask actually applied?".
    fn left_padded_causal_mask(left_pad: &[usize], t: usize, dev: &Device) -> Result<Tensor> {
        let b = left_pad.len();
        let mut data = vec![0f32; b * t * t];
        for (i, &pad) in left_pad.iter().enumerate() {
            for r in 0..t {
                for j in 0..t {
                    if j > r || j < pad {
                        data[(i * t + r) * t + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        Tensor::from_vec(data, (b, 1, t, t), dev)
    }

    /// Replace `k`/`v` (`[b, 1, t, d]`) with loud, distinctive values wherever
    /// `select(i, j)` is true, leaving every other entry bit-identical.
    fn perturb_keys(
        k: &Tensor,
        v: &Tensor,
        select: impl Fn(usize, usize) -> bool,
    ) -> Result<(Tensor, Tensor)> {
        let (b, _kv, t, _d) = k.dims4()?;
        let mut sel = vec![0u8; b * t];
        for (i, row) in sel.chunks_mut(t).enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = u8::from(select(i, j));
            }
        }
        let sel = Tensor::from_vec(sel, (b, 1, t, 1), k.device())?
            .broadcast_as(k.shape())?
            .contiguous()?;
        // Deliberately large: if a masked-out key were attended at all, its
        // logit would dominate the softmax and the output could not survive
        // the equality assertions below.
        let loud = Tensor::full(9.0f32, k.shape(), k.device())?.to_dtype(k.dtype())?;
        Ok((
            sel.where_cond(&loud, k)?.contiguous()?,
            sel.where_cond(&loud.neg()?, v)?.contiguous()?,
        ))
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length mismatch");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    fn sdpa_params(head_dim: usize, n_kv_groups: usize) -> SdpaParams {
        SdpaParams {
            n_kv_groups,
            softcap: None,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        }
    }

    /// SdpaParams carrying a per-head `attn_sink` (shape `[1, n_heads, 1, 1]`
    /// F32, as V4 pre-shapes it). Every CSA/HCA layer in the real model runs
    /// with sinks, so the union finite tests exercise the same `sinks_attn`
    /// path the deployment uses (head_dim=512 → `softmax_with_sinks`). The sink
    /// also supplies a finite softmax denominator, so query rows whose entire
    /// raw-window prefix is masked (e.g. decode at position ≫ window) never NaN.
    fn sdpa_params_with_sinks(
        head_dim: usize,
        n_heads: usize,
        dev: &Device,
    ) -> Result<(SdpaParams, Tensor)> {
        let sinks = Tensor::zeros((1, n_heads, 1, 1), DType::F32, dev)?;
        let params = SdpaParams {
            n_kv_groups: n_heads,
            softcap: None,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: None,
            sinks: Some(sinks.clone()),
        };
        Ok((params, sinks))
    }

    fn mk(b: usize, h: usize, t: usize, d: usize, seed: f32, dev: &Device) -> Result<Tensor> {
        let data: Vec<f32> = (0..(b * h * t * d))
            .map(|i| ((i as f32) * seed).sin())
            .collect();
        Tensor::from_vec(data, (b, h, t, d), dev)
    }

    /// Builds the `[1, 1, t, t]` additive banded causal mask: query `i` sees
    /// key `j` iff `i - window < j <= i`.
    fn banded_causal_mask(t: usize, window: usize, device: &Device) -> Result<Tensor> {
        let mut mask_data = vec![0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                if j > i || j + window <= i {
                    mask_data[i * t + j] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_vec(mask_data, (1, 1, t, t), device)
    }

    /// THE long-context regression test (RUN-161 hardware collapse): Standard
    /// (`compress_ratio == 0`) layers must be **sliding-window** attention,
    /// exactly like the raw branch of CSA/HCA — the reference SWA-onlys
    /// ratio-0 layers (SGLang `deepseek_v4_backend.py`: `indices =
    /// swa_page_indices`; vLLM: `swa_only = compress_ratio <= 1`). The pre-fix
    /// code dense-causal'd them, which collapsed generation the moment the
    /// context crossed the window. This test FAILS on the pre-fix dispatch.
    #[test]
    fn standard_prefill_is_sliding_window_masked() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, t, d, window) = (1, 2, 12, 16, 4);
        let q = mk(b, h, t, d, 0.1, &device)?;
        let k = mk(b, 1, t, d, 0.2, &device)?;
        let v = mk(b, 1, t, d, 0.3, &device)?;

        let sdpa = sdpa_params(d, h);
        let flash = empty_flash_params();
        let windowed_mask = banded_causal_mask(t, window, &device)?;
        let expected = Sdpa.run_attention(&q, &k, &v, Some(&windowed_mask), Some(&flash), &sdpa)?;

        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Standard,
            sliding_window: window,
            raw_prefix: 0,
        };
        let actual = dsv4_attention(&q, &k, &v, None, None, &flash, &sdpa, cfg)?;

        let e: Vec<f32> = expected.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let a: Vec<f32> = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        for (ev, av) in e.iter().zip(a.iter()) {
            assert!((ev - av).abs() < 1e-5, "{ev} != {av}");
        }

        // Sanity: with t > window the windowed output must differ from dense
        // causal — otherwise this test couldn't catch the dense regression.
        let causal_mask = banded_causal_mask(t, t, &device)?;
        let dense = Sdpa.run_attention(&q, &k, &v, Some(&causal_mask), Some(&flash), &sdpa)?;
        let dn: Vec<f32> = dense.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let max_diff = a
            .iter()
            .zip(dn.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "windowed and dense outputs are indistinguishable (max_diff={max_diff}); test has no teeth"
        );
        Ok(())
    }

    /// Standard with a window covering the whole sequence reduces to plain
    /// causal SDPA (the previous `standard_dispatch_matches_plain_sdpa`
    /// contract, now with the correct causal+window semantics).
    #[test]
    fn standard_full_window_matches_causal_sdpa() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, t, d) = (1, 2, 8, 16);
        let q = mk(b, h, t, d, 0.1, &device)?;
        let k = mk(b, 1, t, d, 0.2, &device)?;
        let v = mk(b, 1, t, d, 0.3, &device)?;

        let sdpa = sdpa_params(d, h);
        let flash = empty_flash_params();
        let causal_mask = banded_causal_mask(t, t, &device)?;
        let expected = Sdpa.run_attention(&q, &k, &v, Some(&causal_mask), Some(&flash), &sdpa)?;

        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Standard,
            sliding_window: t,
            raw_prefix: 0,
        };
        let actual = dsv4_attention(&q, &k, &v, None, None, &flash, &sdpa, cfg)?;

        let e: Vec<f32> = expected.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let a: Vec<f32> = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        for (ev, av) in e.iter().zip(a.iter()) {
            assert!((ev - av).abs() < 1e-5, "{ev} != {av}");
        }
        Ok(())
    }

    /// Decode window boundary is exact: at cache length `t_k` (query at
    /// position `t_k - 1`), a Standard layer attends over exactly the last
    /// `window` keys `[t_k - window, t_k - 1]` — verified by comparing against
    /// SDPA over the narrowed K/V at cache lengths crossing the window
    /// boundary (the 128→129 hardware transition, scaled down). An off-by-one
    /// in the window slice fails the equality at every post-window length
    /// (asserted by the shifted-window negative control below).
    /// Runs with (zero) sinks on both sides, matching deployment where
    /// `attn_sink` is always present — and required here: the no-sink CPU
    /// flash path NaNs on rows with a masked prefix (see the HCA test note).
    ///
    /// Tolerance: the dispatch decode now runs the absorbed single-GEMM path
    /// while the reference narrows K/V through `sinks_attn_cpu`'s repeat_kv
    /// GEMM — identical real-number math, but `MatMul`'s CPU path rounds
    /// through F16 and the two GEMM batchings tile (and therefore round)
    /// differently per arch/machine. CI observed exactly-1-F16-ulp diffs
    /// (e.g. 0.46826172 vs 0.46801758 on x86) where this host shows 0, so
    /// the bound is the documented CPU-MatMul F16 noise floor (~1e-3, same
    /// as `union_decode_matches_scalar_reference`), not f32 eps. The window
    /// semantics themselves are shape-level (which keys participate), which
    /// the negative control pins far above this floor.
    #[test]
    fn standard_decode_window_boundary_exact() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, d, window) = (1, 2, 16, 4);
        for t_k in [window - 1, window, window + 1, 2 * window, 4 * window + 1] {
            let q = mk(b, h, 1, d, 0.07, &device)?;
            let k = mk(b, 1, t_k, d, 0.13, &device)?;
            let v = mk(b, 1, t_k, d, 0.19, &device)?;

            let (sdpa, _sinks) = sdpa_params_with_sinks(d, h, &device)?;
            let flash = empty_flash_params();

            // Reference: (unmasked) sinks-SDPA over exactly the last
            // min(window, t_k) keys.
            let n_vis = window.min(t_k);
            let k_win = k.narrow(2, t_k - n_vis, n_vis)?;
            let v_win = v.narrow(2, t_k - n_vis, n_vis)?;
            let expected = Sdpa.run_attention(&q, &k_win, &v_win, None, Some(&flash), &sdpa)?;

            let cfg = Dsv4AttentionConfig {
                compress_ratio: CompressRatio::Standard,
                sliding_window: window,
                raw_prefix: 0,
            };
            let actual = dsv4_attention(&q, &k, &v, None, None, &flash, &sdpa, cfg)?;

            let e: Vec<f32> = expected.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let a: Vec<f32> = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            const TOL: f32 = 1.5e-3;
            for (ev, av) in e.iter().zip(a.iter()) {
                assert!(
                    (ev - av).abs() < TOL,
                    "t_k={t_k}: {ev} != {av} (window slice off)"
                );
            }

            // Negative control (teeth for the F16-floor tolerance): a
            // window slice off by one — the last `window - 1` keys instead
            // of the last `window` — must sit far above TOL. Only
            // meaningful once the window is a strict subset of the cache.
            if t_k > window {
                let k_shift = k.narrow(2, t_k - (n_vis - 1), n_vis - 1)?;
                let v_shift = v.narrow(2, t_k - (n_vis - 1), n_vis - 1)?;
                let shifted =
                    Sdpa.run_attention(&q, &k_shift, &v_shift, None, Some(&flash), &sdpa)?;
                let s: Vec<f32> = shifted.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let max_diff = s
                    .iter()
                    .zip(a.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_diff > 3.0 * TOL,
                    "t_k={t_k}: off-by-one window is indistinguishable at TOL \
                     (max_diff={max_diff}); test has no teeth"
                );
            }
        }
        Ok(())
    }

    /// `raw_prefix` must be a pure accounting change: handing over only the
    /// reachable span, and saying how much was dropped, must produce the
    /// bit-identical output of handing over the whole cache.
    ///
    /// This is the contract the FP8 KV decode path rests on — it narrows the
    /// packed cache *before* reconstructing it, so if `raw_prefix` shifted the
    /// window mask or the compressed-block causality threshold by even one
    /// position, decode would silently attend the wrong keys.
    #[test]
    fn raw_prefix_is_equivalent_to_passing_the_whole_cache() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, d) = (1, 2, 8);
        let window = 6usize;
        let scale = 1.0 / (d as f32).sqrt();
        let sinks = Tensor::from_vec(vec![0.5f32, 0.75], (1, h, 1, 1), &device)?;
        let sdpa = SdpaParams {
            n_kv_groups: h,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: Some(sinks),
        };
        let flash = empty_flash_params();

        let mut narrowed_at_least_once = false;
        for (ratio, t_k, t_q) in [
            (CompressRatio::Standard, 20usize, 1usize),
            (CompressRatio::Csa, 20, 1),
            (CompressRatio::Csa, 20, 3),
            (CompressRatio::Hca, 40, 1),
        ] {
            let q = mk(b, h, t_q, d, 0.03, &device)?;
            let k = mk(b, 1, t_k, d, 0.11, &device)?;
            let comp = mk(b, 1, (t_k / ratio.ratio()).max(1), d, 0.05, &device)?;
            let compressed = (ratio != CompressRatio::Standard).then_some(&comp);

            let whole = dsv4_attention(
                &q,
                &k,
                &k,
                compressed,
                None,
                &flash,
                &sdpa,
                Dsv4AttentionConfig {
                    compress_ratio: ratio,
                    sliding_window: window,
                    raw_prefix: 0,
                },
            )?;

            let (base, keep) = raw_keep_span(t_q, window, t_k);
            if base > 0 {
                narrowed_at_least_once = true;
            }
            let k_span = k.narrow(2, base, keep)?.contiguous()?;
            let spanned = dsv4_attention(
                &q,
                &k_span,
                &k_span,
                compressed,
                None,
                &flash,
                &sdpa,
                Dsv4AttentionConfig {
                    compress_ratio: ratio,
                    sliding_window: window,
                    raw_prefix: base,
                },
            )?;

            let a: Vec<f32> = whole.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let c: Vec<f32> = spanned.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            assert_eq!(
                a, c,
                "{ratio:?} t_k={t_k} t_q={t_q}: raw_prefix={base} changed the output"
            );
        }
        // Without this the sweep could pass while every case had `base == 0`,
        // i.e. while `raw_prefix` was never actually exercised.
        assert!(
            narrowed_at_least_once,
            "fixture cannot discriminate: no case actually dropped a prefix"
        );
        Ok(())
    }

    /// Dropping keys a query row can still reach is a caller bug, and must be
    /// refused rather than silently answered with a truncated window.
    #[test]
    fn raw_prefix_past_the_reachable_span_is_refused() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, d) = (1, 2, 8);
        let q = mk(b, h, 1, d, 0.03, &device)?;
        let k = mk(b, 1, 4, d, 0.11, &device)?;
        let sdpa = SdpaParams {
            n_kv_groups: h,
            softcap: None,
            softmax_scale: 1.0 / (d as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };
        // window 16 > 4 cached keys, so row 0 reaches key 0: nothing is
        // droppable, yet the caller claims to have dropped 4.
        let err = dsv4_attention(
            &q,
            &k,
            &k,
            None,
            None,
            &empty_flash_params(),
            &sdpa,
            Dsv4AttentionConfig {
                compress_ratio: CompressRatio::Standard,
                sliding_window: 16,
                raw_prefix: 4,
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("are NOT"),
            "expected a loud unreachable-prefix error, got: {err}"
        );
        Ok(())
    }

    /// Scalar (loop-based) reference for the full union semantics at decode:
    /// one softmax over [raw window ++ compressed blocks] with the per-head
    /// sink as a denominator-only column (value 0), at cache lengths crossing
    /// 128/256/512 with the production window=128 / ratio=4. Pins:
    ///   * raw window slice `[pos - 127, pos]` (absolute positions, no
    ///     re-basing);
    ///   * compressed-block causality `b < (pos + 1) / ratio`;
    ///   * sink participating in the denominator only.
    #[test]
    fn union_decode_matches_scalar_reference() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, d) = (1, 2, 8);
        let window = 128usize;
        let ratio = 4usize;
        let scale = 1.0 / (d as f32).sqrt();

        for t_k in [127usize, 128, 129, 256, 512, 513] {
            let pos = t_k - 1;
            let t_c = t_k / ratio; // caller compresses the largest ratio-multiple prefix
            let q = mk(b, h, 1, d, 0.03, &device)?;
            let k = mk(b, 1, t_k, d, 0.11, &device)?;
            let v = mk(b, 1, t_k, d, 0.17, &device)?;
            let comp = mk(b, 1, t_c, d, 0.05, &device)?;

            // Per-head sinks, nonzero so a sink bug shifts the output.
            let sinks_data: Vec<f32> = (0..h).map(|i| 0.5 + 0.25 * i as f32).collect();
            let sinks = Tensor::from_vec(sinks_data.clone(), (1, h, 1, 1), &device)?;
            let sdpa = SdpaParams {
                n_kv_groups: h,
                softcap: None,
                softmax_scale: scale,
                sliding_window: None,
                sinks: Some(sinks),
            };
            let flash = empty_flash_params();
            let cfg = Dsv4AttentionConfig {
                compress_ratio: CompressRatio::Csa,
                sliding_window: window,
                raw_prefix: 0,
            };
            let out = dsv4_attention(&q, &k, &v, Some(&comp), None, &flash, &sdpa, cfg)?;
            let out_v: Vec<f32> = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

            // ---- Scalar reference ----
            let qv: Vec<f32> = q.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?; // [h, d]
            let kv: Vec<f32> = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?; // [t_k, d]
            let vv: Vec<f32> = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?; // [t_k, d]
            let cv: Vec<f32> = comp.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?; // [t_c, d]

            let raw_lo = (pos + 1).saturating_sub(window); // first visible raw key
            let n_blocks = (pos + 1) / ratio; // visible compressed blocks
            assert!(n_blocks <= t_c);

            // Scalar reference in f64. `include_sink=false` / `blocks`
            // variants serve as negative controls (buggy semantics).
            let scalar_ref = |raw_lo: usize, n_blocks: usize, include_sink: bool| -> Vec<f64> {
                let dot = |a: &[f32], b: &[f32]| -> f64 {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| *x as f64 * *y as f64)
                        .sum::<f64>()
                        * scale as f64
                };
                let mut out = vec![0f64; h * d];
                for head in 0..h {
                    let qh = &qv[head * d..(head + 1) * d];
                    let mut logits: Vec<f64> = Vec::new();
                    let mut values: Vec<&[f32]> = Vec::new();
                    for j in raw_lo..=pos {
                        logits.push(dot(qh, &kv[j * d..(j + 1) * d]));
                        values.push(&vv[j * d..(j + 1) * d]);
                    }
                    for bidx in 0..n_blocks {
                        let cb = &cv[bidx * d..(bidx + 1) * d];
                        logits.push(dot(qh, cb));
                        values.push(cb);
                    }
                    // Softmax; the sink is a denominator-only extra column.
                    let sink = sinks_data[head] as f64;
                    let m0 = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let m = if include_sink { m0.max(sink) } else { m0 };
                    let mut denom: f64 = logits.iter().map(|&x| (x - m).exp()).sum();
                    if include_sink {
                        denom += (sink - m).exp();
                    }
                    for (lg, val) in logits.iter().zip(values.iter()) {
                        let w = (lg - m).exp() / denom;
                        for (e, x) in out[head * d..(head + 1) * d].iter_mut().zip(val.iter()) {
                            *e += w * *x as f64;
                        }
                    }
                }
                out
            };
            let max_diff = |reference: &[f64]| -> f64 {
                reference
                    .iter()
                    .zip(out_v.iter())
                    .map(|(e, a)| (e - *a as f64).abs())
                    .fold(0f64, f64::max)
            };

            // The dispatch's CPU matmuls round through F16 (MatMul::matmul
            // CPU path), so the noise floor is ~1e-3, not f32 eps.
            let d_correct = max_diff(&scalar_ref(raw_lo, n_blocks, true));
            assert!(
                d_correct < 1.5e-3,
                "t_k={t_k}: dispatch deviates from union semantics (max diff {d_correct})"
            );

            // Discrimination (negative controls) at the shorter lengths,
            // where per-column softmax mass is well above the F16 noise:
            // dropping the last visible compressed block (block-causality
            // off-by-one) or the sink column must move the reference away
            // from the dispatch by a clear margin.
            if t_k <= 129 && n_blocks > 0 {
                let d_dropblock = max_diff(&scalar_ref(raw_lo, n_blocks - 1, true));
                let d_nosink = max_diff(&scalar_ref(raw_lo, n_blocks, false));
                let floor = 3.0 * d_correct;
                assert!(
                    d_dropblock > floor,
                    "t_k={t_k}: dropping a compressed block is indistinguishable \
                     (correct {d_correct} vs dropblock {d_dropblock}) — test has no teeth"
                );
                assert!(
                    d_nosink > floor,
                    "t_k={t_k}: omitting the sink is indistinguishable \
                     (correct {d_correct} vs nosink {d_nosink}) — test has no teeth"
                );
            }
        }
        Ok(())
    }

    /// CSA/HCA with no compressed KV and a window covering the whole sequence
    /// reduces to plain causal SDPA (the windowed-causal mask == causal mask).
    #[test]
    fn csa_no_compressed_full_window_matches_causal() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, t, d) = (1, 2, 6, 16);
        let q = mk(b, h, t, d, 0.1, &device)?;
        let k = mk(b, 1, t, d, 0.2, &device)?;
        let v = mk(b, 1, t, d, 0.3, &device)?;
        let sdpa = sdpa_params(d, h);
        let flash = empty_flash_params();

        // Reference: plain causal SDPA with a [1,1,t,t] additive causal mask.
        let mut mask_data = vec![0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                if j > i {
                    mask_data[i * t + j] = f32::NEG_INFINITY;
                }
            }
        }
        let causal_mask = Tensor::from_vec(mask_data, (1, 1, t, t), &device)?;
        let expected = Sdpa.run_attention(&q, &k, &v, Some(&causal_mask), Some(&flash), &sdpa)?;

        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Csa,
            sliding_window: t, // window covers everything → windowed == causal
            raw_prefix: 0,
        };
        let actual = dsv4_attention(&q, &k, &v, None, None, &flash, &sdpa, cfg)?;

        let e: Vec<f32> = expected.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let a: Vec<f32> = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        for (ev, av) in e.iter().zip(a.iter()) {
            assert!((ev - av).abs() < 1e-4, "{ev} != {av}");
        }
        Ok(())
    }

    /// CSA union (raw window + compressed) emits finite output of the right
    /// shape and obeys the union mask.
    #[test]
    fn csa_union_emits_finite() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, d) = (1, 2, 16);
        let (t_k, t_q, t_c) = (16, 16, 4);
        let q = mk(b, h, t_q, d, 0.1, &device)?;
        let k = mk(b, 1, t_k, d, 0.2, &device)?;
        let v = mk(b, 1, t_k, d, 0.3, &device)?;
        let comp = mk(b, 1, t_c, d, 0.05, &device)?;

        let (sdpa, _sinks) = sdpa_params_with_sinks(d, h, &device)?;
        let flash = empty_flash_params();
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Csa,
            sliding_window: 4,
            raw_prefix: 0,
        };
        let out = dsv4_attention(&q, &k, &v, Some(&comp), None, &flash, &sdpa, cfg)?;
        assert_eq!(out.dims(), &[b, h, t_q, d]);
        let data: Vec<f32> = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|x| x.is_finite()));
        Ok(())
    }

    /// The absorbed decode is a mathematical identity with the naive
    /// repeat_kv expansion: same dot products, same order, only the GEMM
    /// batching differs. Compare `absorbed_mqa_decode` against
    /// `Sdpa.run_attention` (which with sinks routes through
    /// `sinks_attn_cpu`'s repeat_kv path) on identical inputs — nonzero
    /// per-head sinks, a mask with -inf holes, v != k on purpose (the
    /// absorbed path must not assume V aliases K), across batch sizes and
    /// cache lengths.
    ///
    /// Tolerance: `MatMul`'s CPU path rounds through F16, and the two GEMM
    /// batchings ([B*H,1,D] repeat_kv vs [B,H,D] head-folded) tile — and
    /// therefore round — differently per arch/machine. CI observed
    /// exactly-1-F16-ulp diffs (e.g. -0.10424805 vs -0.10418701 on x86,
    /// ulp(0.1)=6.1e-5) where this host shows 0, so the bound is the
    /// documented CPU-MatMul F16 noise floor (~1e-3; see
    /// `union_decode_matches_scalar_reference`), not f32 eps. Any real
    /// semantic divergence (wrong mask column, missing sink, head
    /// scrambling) moves outputs by orders of magnitude more.
    #[test]
    fn absorbed_decode_matches_repeat_kv_reference() -> Result<()> {
        let device = Device::Cpu;
        let (h, d) = (4, 16);
        for b in [1usize, 2] {
            for t_k in [1usize, 5, 33, 128] {
                let q = mk(b, h, 1, d, 0.03, &device)?;
                let k = mk(b, 1, t_k, d, 0.11, &device)?;
                let v = mk(b, 1, t_k, d, 0.17, &device)?;

                let sinks_data: Vec<f32> = (0..h).map(|i| 0.2 + 0.3 * i as f32).collect();
                let sinks = Tensor::from_vec(sinks_data, (1, h, 1, 1), &device)?;
                let sdpa = SdpaParams {
                    n_kv_groups: h,
                    softcap: None,
                    softmax_scale: 1.0 / (d as f32).sqrt(),
                    sliding_window: None,
                    sinks: Some(sinks),
                };

                // Additive mask with some fully-masked columns (never the
                // diagonal/last key, so no all-masked row).
                let mut mask_data = vec![0f32; t_k];
                for (j, mval) in mask_data.iter_mut().enumerate() {
                    if j % 3 == 1 && j + 1 != t_k {
                        *mval = f32::NEG_INFINITY;
                    }
                }
                let mask = Tensor::from_vec(mask_data, (1, 1, 1, t_k), &device)?;

                let flash = empty_flash_params();
                let expected = Sdpa.run_attention(&q, &k, &v, Some(&mask), Some(&flash), &sdpa)?;
                let actual = absorbed_mqa_decode(&q, &k, &v, &mask, &sdpa)?;
                assert_eq!(actual.dims(), &[b, h, 1, d]);

                let e: Vec<f32> = expected.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let a: Vec<f32> = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                for (ev, av) in e.iter().zip(a.iter()) {
                    assert!(
                        (ev - av).abs() < 1.5e-3,
                        "b={b} t_k={t_k}: absorbed {av} != naive {ev}"
                    );
                }
            }
        }
        Ok(())
    }

    /// HCA union (single-token decode) emits finite output.
    #[test]
    fn hca_union_decode_emits_finite() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, d) = (1, 2, 16);
        let (t_k, t_q, t_c) = (300, 1, 2);
        let q = mk(b, h, t_q, d, 0.05, &device)?;
        let k = mk(b, 1, t_k, d, 0.15, &device)?;
        let v = mk(b, 1, t_k, d, 0.25, &device)?;
        let comp = mk(b, 1, t_c, d, 0.02, &device)?;

        // Decode at position 299 with window=128: the entire raw-window prefix
        // (keys 0..171) is masked. This is exactly the case where a flash-style
        // online softmax would NaN on the leading all-masked block — V4's
        // always-on `attn_sink` (sinks_attn path) supplies the finite
        // denominator that keeps it well-defined. Mirror that here.
        let (sdpa, _sinks) = sdpa_params_with_sinks(d, h, &device)?;
        let flash = empty_flash_params();
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Hca,
            sliding_window: 128,
            raw_prefix: 0,
        };
        let out = dsv4_attention(&q, &k, &v, Some(&comp), None, &flash, &sdpa, cfg)?;
        assert_eq!(out.dims(), &[b, h, t_q, d]);
        let data: Vec<f32> = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|x| x.is_finite()));
        Ok(())
    }

    // =======================================================================
    // BATCHED-SERVING CORRECTNESS (reference audit §1(e), findings 3 and 4)
    // =======================================================================

    /// **Batch-vs-solo equivalence.** N sequences of DIFFERENT lengths, run
    /// right-padded as one batch and again individually, must agree per
    /// sequence over that sequence's real rows.
    ///
    /// This is the test that would have caught finding 3. The `FlashParams`
    /// here carry `cu_seqlens` for the query device — which is what the engine
    /// produces on every flash-attn build — and with `b_sz > 1` that used to
    /// send this call into `sinks_attn_varlen`, a backend with no mask
    /// parameter whose CPU fallback attended bidirectionally over a packed
    /// K/V layout the caller never supplies. Pre-fix this test does not merely
    /// disagree: it fails outright inside the varlen path.
    #[test]
    fn ragged_batch_prefill_matches_per_sequence() -> Result<()> {
        let device = Device::Cpu;
        let (h, d, window) = (2, 16, 6);
        // Ragged on purpose, and none of them a multiple of the CSA ratio.
        let lens = [12usize, 7, 5];
        let b = lens.len();
        let t = lens[0];
        let t_c = 3;

        let q = mk(b, h, t, d, 0.11, &device)?;
        let k = mk(b, 1, t, d, 0.23, &device)?;
        let v = mk(b, 1, t, d, 0.31, &device)?;
        let comp = mk(b, 1, t_c, d, 0.07, &device)?;

        let (sdpa, _sinks) = sdpa_params_with_sinks(d, h, &device)?;
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Csa,
            sliding_window: window,
            raw_prefix: 0,
        };

        // The engine pads every sequence to the batch max and reports PADDED
        // lengths in cu_seqlens (see the `FlashParams` doc in
        // `inputs_processor.rs`), so all entries are `t`.
        let padded = vec![t as u32; b];
        let flash = varlen_flash_params(&padded, &padded, &device);
        // Right padding: a plain per-batch causal mask is what `CausalMasker`
        // hands the layer. Rank 4, raw-width — it must be folded in, not
        // dropped, and must not disturb the compressed columns.
        let mask = left_padded_causal_mask(&[0, 0, 0], t, &device)?;

        let batched = dsv4_attention(&q, &k, &v, Some(&comp), Some(&mask), &flash, &sdpa, cfg)?;
        assert_eq!(batched.dims(), &[b, h, t, d]);

        let solo_flash = empty_flash_params();
        let mut solos: Vec<Vec<f32>> = Vec::new();
        for (i, &len) in lens.iter().enumerate() {
            let qi = q.narrow(0, i, 1)?.narrow(2, 0, len)?.contiguous()?;
            let ki = k.narrow(0, i, 1)?.narrow(2, 0, len)?.contiguous()?;
            let vi = v.narrow(0, i, 1)?.narrow(2, 0, len)?.contiguous()?;
            let ci = comp.narrow(0, i, 1)?.contiguous()?;
            let solo = dsv4_attention(&qi, &ki, &vi, Some(&ci), None, &solo_flash, &sdpa, cfg)?;

            let got = flat(&batched.narrow(0, i, 1)?.narrow(2, 0, len)?.contiguous()?);
            let want = flat(&solo);
            // Batch-2/3 and batch-1 GEMMs tile — and therefore round —
            // differently on `MatMul`'s F16 CPU path; same documented noise
            // floor as `union_decode_matches_scalar_reference`.
            let diff = max_abs_diff(&got, &want);
            assert!(
                diff < 1.5e-3,
                "sequence {i} (len {len}) differs between the batch and its solo run \
                 (max abs diff {diff}) — batched serving is not per-sequence correct"
            );
            solos.push(want);
        }

        // Teeth: the sequences must be grossly distinguishable, so agreement
        // above is not the vacuous kind. Compare over the shortest length.
        let short = *lens.iter().min().unwrap();
        let head = |v: &[f32]| v[..h * short * d].to_vec();
        assert!(
            max_abs_diff(&head(&solos[0]), &head(&solos[1])) > 1e-2,
            "fixture sequences are too similar for the equality assertions to have teeth"
        );
        Ok(())
    }

    /// **Causality probe.** Plant a loud, distinctive key/value strictly in
    /// the future of a query row; the row's output must not move at all.
    ///
    /// Run at `b_sz > 1` with `cu_seqlens` present — the configuration whose
    /// backend had no causality whatsoever. The perturbation is bit-exactly
    /// invisible when masking is right (a `-inf` logit gives an exactly-zero
    /// softmax weight, and `0 * x == 0`), so this asserts exact equality; the
    /// negative control shows the same perturbation moves the output by O(1)
    /// the moment it lands inside the causal set.
    #[test]
    fn future_keys_cannot_influence_batched_prefill() -> Result<()> {
        let device = Device::Cpu;
        let (h, d, window) = (2, 16, 32);
        let (b, t, t_c) = (2, 10, 2);
        let cut = 5usize; // rows 0..=cut must not see keys > cut

        let q = mk(b, h, t, d, 0.09, &device)?;
        let k = mk(b, 1, t, d, 0.19, &device)?;
        let v = mk(b, 1, t, d, 0.29, &device)?;
        let comp = mk(b, 1, t_c, d, 0.04, &device)?;

        let (sdpa, _sinks) = sdpa_params_with_sinks(d, h, &device)?;
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Csa,
            sliding_window: window, // window >= t, so ONLY causality is on trial
            raw_prefix: 0,
        };
        let padded = vec![t as u32; b];
        let flash = varlen_flash_params(&padded, &padded, &device);
        let mask = left_padded_causal_mask(&[0; 2], t, &device)?;

        let base = dsv4_attention(&q, &k, &v, Some(&comp), Some(&mask), &flash, &sdpa, cfg)?;

        // Distinctive token late in sequence 0 only.
        let (k_fut, v_fut) = perturb_keys(&k, &v, |i, j| i == 0 && j > cut)?;
        let futured = dsv4_attention(
            &q,
            &k_fut,
            &v_fut,
            Some(&comp),
            Some(&mask),
            &flash,
            &sdpa,
            cfg,
        )?;

        let rows = |x: &Tensor| -> Result<Vec<f32>> {
            Ok(flat(
                &x.narrow(0, 0, 1)?.narrow(2, 0, cut + 1)?.contiguous()?,
            ))
        };
        let diff = max_abs_diff(&rows(&base)?, &rows(&futured)?);
        assert_eq!(
            diff, 0.0,
            "a token planted AFTER position {cut} changed the output at positions 0..={cut} \
             (max abs diff {diff}) — batch prefill is not causal"
        );

        // Negative control: the identical perturbation inside the causal set
        // moves the same rows by a wide margin, so the equality above is a
        // real constraint and not an artifact of an inert perturbation.
        let (k_past, v_past) = perturb_keys(&k, &v, |i, j| i == 0 && j <= cut)?;
        let pasted = dsv4_attention(
            &q,
            &k_past,
            &v_past,
            Some(&comp),
            Some(&mask),
            &flash,
            &sdpa,
            cfg,
        )?;
        let control = max_abs_diff(&rows(&base)?, &rows(&pasted)?);
        assert!(
            control > 1e-2,
            "the probe token is invisible even inside the causal set (diff {control}); \
             the causality assertion has no teeth"
        );

        // Sequence 1 was never touched: batch-mate invariance.
        let mate = |x: &Tensor| -> Result<Vec<f32>> { Ok(flat(&x.narrow(0, 1, 1)?.contiguous()?)) };
        assert_eq!(
            max_abs_diff(&mate(&base)?, &mate(&futured)?),
            0.0,
            "perturbing sequence 0's keys changed sequence 1's output — cross-sequence leak"
        );
        Ok(())
    }

    /// **Padding mask honored.** A caller mask that kills the leading `pad[i]`
    /// key columns of batch row `i` (left padding — the columns causality
    /// cannot exclude) must make those positions unable to influence anything.
    ///
    /// Pre-fix, `dsv4_attention` read `attention_mask` on exactly one
    /// env-gated branch and dropped it on every live path, so this fails by
    /// the width of the negative control below.
    #[test]
    fn caller_padding_mask_is_honored() -> Result<()> {
        let device = Device::Cpu;
        let (h, d, window) = (2, 16, 8);
        let (t, t_c) = (12, 3);
        let pads = [0usize, 4, 6];
        let b = pads.len();

        let q = mk(b, h, t, d, 0.13, &device)?;
        let k = mk(b, 1, t, d, 0.21, &device)?;
        let v = mk(b, 1, t, d, 0.27, &device)?;
        let comp = mk(b, 1, t_c, d, 0.06, &device)?;

        let (sdpa, _sinks) = sdpa_params_with_sinks(d, h, &device)?;
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Csa,
            sliding_window: window,
            raw_prefix: 0,
        };
        let padded = vec![t as u32; b];
        let flash = varlen_flash_params(&padded, &padded, &device);
        let mask = left_padded_causal_mask(&pads, t, &device)?;

        let base = dsv4_attention(&q, &k, &v, Some(&comp), Some(&mask), &flash, &sdpa, cfg)?;
        // Perturb ONLY the padded columns. Every real query row still sits
        // inside the sliding window of some of them, so an unapplied mask
        // leaks immediately.
        let (k_p, v_p) = perturb_keys(&k, &v, |i, j| j < pads[i])?;
        let perturbed =
            dsv4_attention(&q, &k_p, &v_p, Some(&comp), Some(&mask), &flash, &sdpa, cfg)?;

        for (i, &pad) in pads.iter().enumerate() {
            let real = |x: &Tensor| -> Result<Vec<f32>> {
                Ok(flat(
                    &x.narrow(0, i, 1)?.narrow(2, pad, t - pad)?.contiguous()?,
                ))
            };
            let diff = max_abs_diff(&real(&base)?, &real(&perturbed)?);
            assert_eq!(
                diff, 0.0,
                "sequence {i}: changing PADDED key positions [0, {pad}) moved the real \
                 positions' output (max abs diff {diff}) — the caller's padding mask is \
                 not applied"
            );
        }

        // Negative control: without the mask the very same perturbation is
        // loudly visible, i.e. the padded columns really are inside the
        // causal ∧ window set and the assertion above is not vacuous.
        let unmasked_base = dsv4_attention(&q, &k, &v, Some(&comp), None, &flash, &sdpa, cfg)?;
        let unmasked_pert = dsv4_attention(&q, &k_p, &v_p, Some(&comp), None, &flash, &sdpa, cfg)?;
        let control = max_abs_diff(&flat(&unmasked_base), &flat(&unmasked_pert));
        assert!(
            control > 1e-2,
            "padded keys are inert even with no mask at all (diff {control}); this test \
             cannot distinguish an applied mask from a dropped one"
        );
        Ok(())
    }

    /// **Union guard.** Folding the caller's mask must not disturb the
    /// raw∪compressed union semantics: the branches are concatenated, not
    /// deduplicated, so a token that is visible BOTH as a raw key and inside a
    /// visible compressed block contributes twice. The reference audit
    /// confirmed this double counting matches SGLang (`clip_down`/`get_raw_loc`
    /// there are ring-buffer addressing, not masking), so it is a contract to
    /// hold, not a bug to fix. The caller's mask spans the raw axis only and
    /// must leave the compressed columns alone.
    #[test]
    fn caller_mask_preserves_raw_union_compressed_double_counting() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, d) = (1usize, 2usize, 8usize);
        let (t, t_c, ratio) = (8usize, 2usize, 4usize);
        let scale = 1.0 / (d as f32).sqrt();

        let q = mk(b, h, t, d, 0.03, &device)?;
        let k = mk(b, 1, t, d, 0.11, &device)?;
        let v = mk(b, 1, t, d, 0.17, &device)?;
        let comp = mk(b, 1, t_c, d, 0.05, &device)?;

        let sinks_data: Vec<f32> = (0..h).map(|i| 0.5 + 0.25 * i as f32).collect();
        let sinks = Tensor::from_vec(sinks_data.clone(), (1, h, 1, 1), &device)?;
        let sdpa = SdpaParams {
            n_kv_groups: h,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: Some(sinks),
        };
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Csa,
            // Window covers the whole sequence, so every raw key a query can
            // see is ALSO covered by a visible compressed block: maximum
            // overlap, which is what makes the double counting observable.
            sliding_window: t,
            raw_prefix: 0,
        };
        // Rank-4 causal mask (what `CausalMasker` hands the layer), raw width.
        let caller = left_padded_causal_mask(&[0], t, &device)?;
        let out = dsv4_attention(
            &q,
            &k,
            &v,
            Some(&comp),
            Some(&caller),
            &empty_flash_params(),
            &sdpa,
            cfg,
        )?;
        let got = flat(&out);

        let qv = flat(&q);
        let kv = flat(&k);
        let vv = flat(&v);
        let cv = flat(&comp);
        let dot = |a: &[f32], bb: &[f32]| -> f64 {
            a.iter()
                .zip(bb)
                .map(|(x, y)| *x as f64 * *y as f64)
                .sum::<f64>()
                * scale as f64
        };
        // `dedup` collapses the raw/compressed overlap — the "fix" the audit
        // says NOT to make. It serves as the negative control.
        let scalar_ref = |dedup: bool| -> Vec<f64> {
            let mut out = vec![0f64; h * t * d];
            for head in 0..h {
                for r in 0..t {
                    let qh = &qv[(head * t + r) * d..(head * t + r + 1) * d];
                    let mut logits: Vec<f64> = Vec::new();
                    let mut values: Vec<&[f32]> = Vec::new();
                    let n_blocks = (r + 1) / ratio;
                    // Raw branch: causal, window == t so nothing is dropped.
                    let raw_lo = if dedup { n_blocks * ratio } else { 0 };
                    for j in raw_lo..=r {
                        logits.push(dot(qh, &kv[j * d..(j + 1) * d]));
                        values.push(&vv[j * d..(j + 1) * d]);
                    }
                    for bidx in 0..n_blocks {
                        let cb = &cv[bidx * d..(bidx + 1) * d];
                        logits.push(dot(qh, cb));
                        values.push(cb);
                    }
                    let sink = sinks_data[head] as f64;
                    let m = logits
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                        .max(sink);
                    let denom: f64 =
                        logits.iter().map(|&x| (x - m).exp()).sum::<f64>() + (sink - m).exp();
                    let base = (head * t + r) * d;
                    for (lg, val) in logits.iter().zip(values.iter()) {
                        let w = (lg - m).exp() / denom;
                        for (e, x) in out[base..base + d].iter_mut().zip(val.iter()) {
                            *e += w * *x as f64;
                        }
                    }
                }
            }
            out
        };
        let diff = |reference: &[f64]| -> f64 {
            reference
                .iter()
                .zip(got.iter())
                .map(|(e, a)| (e - *a as f64).abs())
                .fold(0f64, f64::max)
        };

        let d_union = diff(&scalar_ref(false));
        assert!(
            d_union < 1.5e-3,
            "prefill union semantics drifted (max diff {d_union}) — raw and compressed \
             must be one softmax over the concatenated key set, with the caller's mask \
             folded into the raw half only"
        );
        let d_dedup = diff(&scalar_ref(true));
        assert!(
            d_dedup > 3.0 * d_union.max(1e-4),
            "de-duplicating the raw/compressed overlap is indistinguishable here \
             (union {d_union} vs dedup {d_dedup}); this guard has no teeth"
        );
        Ok(())
    }

    /// The raw working-set span is exactly the reachable set — no wider (or the
    /// narrowing buys nothing) and no narrower (or it silently drops attended
    /// keys). Derived independently here by brute force over the same
    /// per-row window rule `dsv4_attention` implements, so an edit to
    /// [`raw_keep_span`]'s arithmetic has to survive the definition, not a
    /// restatement of itself.
    #[test]
    fn raw_keep_span_is_exactly_the_reachable_set() {
        let mut narrowed_at_least_once = false;
        for window in [1usize, 2, 4, 8, 128] {
            for t_k_full in [1usize, 3, 8, 129, 256, 2048] {
                for t_q in [1usize, 2, 7, 64] {
                    if t_q > t_k_full {
                        continue;
                    }
                    let q0 = t_k_full - t_q;
                    // Brute-force union of {j : q0+r-window < j <= q0+r, 0<=j<t_k_full}.
                    let reachable: Vec<usize> = (0..t_k_full)
                        .filter(|j| {
                            (0..t_q).any(|r| {
                                let p = q0 + r;
                                *j <= p && p < *j + window
                            })
                        })
                        .collect();
                    let (base, keep) = raw_keep_span(t_q, window, t_k_full);
                    assert_eq!(
                        (base, base + keep),
                        (reachable[0], reachable[reachable.len() - 1] + 1),
                        "window={window} t_k_full={t_k_full} t_q={t_q}"
                    );
                    // The span must be contiguous for a `narrow` to be valid.
                    assert_eq!(reachable.len(), keep);
                    if keep < t_k_full {
                        narrowed_at_least_once = true;
                    }
                }
            }
        }
        // Without this the loop above could pass while `raw_keep_span` was the
        // identity on every case it visits (DOCTRINE D12).
        assert!(
            narrowed_at_least_once,
            "no case actually narrowed; the sweep cannot discriminate a real \
             narrowing from `keep == t_k_full` everywhere"
        );
        // Production decode: window 128, 2048 ctx, one query.
        assert_eq!(raw_keep_span(1, 128, 2048), (1920, 128));
        // Prefill is untouched.
        assert_eq!(raw_keep_span(2048, 128, 2048), (0, 2048));
    }

    /// End-to-end identity: keys outside the retained span cannot influence the
    /// output (so dropping them is lossless), and the key at the retained
    /// boundary CAN (so the span is not over-trimmed). Uses the deliberately
    /// loud `perturb_keys` values — an attended-but-dropped key would dominate
    /// the softmax and could not survive the equality.
    #[test]
    fn keys_outside_the_retained_span_cannot_influence_decode() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, d, window, t_k) = (1, 2, 16, 8usize, 64usize);
        let q = mk(b, h, 1, d, 0.07, &device)?;
        let k = mk(b, 1, t_k, d, 0.13, &device)?;
        let v = mk(b, 1, t_k, d, 0.19, &device)?;
        let (sdpa, _sinks) = sdpa_params_with_sinks(d, h, &device)?;
        let flash = empty_flash_params();
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Standard,
            sliding_window: window,
            raw_prefix: 0,
        };

        let (base, keep) = raw_keep_span(1, window, t_k);
        assert!(keep < t_k, "fixture does not narrow; the test has no teeth");
        assert_eq!((base, keep), (56, 8));

        let baseline = dsv4_attention(&q, &k, &v, None, None, &flash, &sdpa, cfg)?;

        // Every key before the span: output must be bit-identical.
        let (k_out, v_out) = perturb_keys(&k, &v, |_, j| j < base)?;
        let perturbed_outside = dsv4_attention(&q, &k_out, &v_out, None, None, &flash, &sdpa, cfg)?;
        assert_eq!(
            max_abs_diff(&flat(&baseline), &flat(&perturbed_outside)),
            0.0,
            "a key outside the retained span changed the output"
        );

        // The first key INSIDE the span: output must move, or the span is
        // wider than it needs to be and this test proves nothing.
        let (k_in, v_in) = perturb_keys(&k, &v, |_, j| j == base)?;
        let perturbed_inside = dsv4_attention(&q, &k_in, &v_in, None, None, &flash, &sdpa, cfg)?;
        assert!(
            max_abs_diff(&flat(&baseline), &flat(&perturbed_inside)) > 1e-3,
            "the oldest retained key had no effect; the span is over-wide"
        );
        Ok(())
    }

    /// A caller mask whose key axis matches neither the raw cache width nor
    /// the union width is a contract violation, and is reported rather than
    /// silently ignored — silence is what made findings 3 and 4 survive.
    #[test]
    fn mismatched_caller_mask_is_rejected() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, t, d) = (1, 2, 8, 16);
        let q = mk(b, h, t, d, 0.1, &device)?;
        let k = mk(b, 1, t, d, 0.2, &device)?;
        let v = mk(b, 1, t, d, 0.3, &device)?;
        let (sdpa, _sinks) = sdpa_params_with_sinks(d, h, &device)?;
        let flash = empty_flash_params();
        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Standard,
            sliding_window: 4,
            raw_prefix: 0,
        };
        let bad = Tensor::zeros((1, 1, t, t + 3), DType::F32, &device)?;
        let err = dsv4_attention(&q, &k, &v, None, Some(&bad), &flash, &sdpa, cfg).unwrap_err();
        assert!(
            err.to_string().contains("attention_mask key axis"),
            "expected a mask-width error, got: {err}"
        );

        // The flash-attn `[1, 1]` placeholder carries no information and must
        // stay a no-op rather than trip the width check.
        let dummy = Tensor::zeros((1, 1), DType::F32, &device)?;
        let with_dummy = dsv4_attention(&q, &k, &v, None, Some(&dummy), &flash, &sdpa, cfg)?;
        let without = dsv4_attention(&q, &k, &v, None, None, &flash, &sdpa, cfg)?;
        assert_eq!(
            max_abs_diff(&flat(&with_dummy), &flat(&without)),
            0.0,
            "the flash-attn placeholder mask changed the result"
        );
        Ok(())
    }
}
