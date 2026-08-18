#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::{attention::backends::cpu, pipeline::text_models_inputs_processor::FlashParams};

use candle_core::{DType, Device, Result, Tensor};

mod backends;

#[allow(unused)]
pub(crate) use backends::{
    cuda_flash_attn_supported, flash_attn, maybe_synchronize, naive_sdpa, sinks_attn,
};

/// Chunk size for attention computation to avoid OOM on long sequences
pub(crate) const ATTENTION_CHUNK_SIZE: usize = 1024;

/// RUN-161 decode-bug bisection: when `ARC_FORCE_NAIVE_SDPA=1`, the CUDA
/// no-flash path skips the cuBLASLt batch-matmul attention and uses
/// `naive_sdpa` instead. cuBLASLt at M=1 (single-query decode step) on the
/// V4 head_dim=512 shape is the prime suspect for the decode collapse; this
/// gate lets us A/B it at runtime with no rebuild between tests.
fn force_naive_sdpa() -> bool {
    use std::sync::OnceLock;
    static FORCE: OnceLock<bool> = OnceLock::new();
    *FORCE.get_or_init(|| {
        matches!(
            std::env::var("ARC_FORCE_NAIVE_SDPA").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Generic chunked attention computation that can be used by different backends
pub(crate) fn chunked_attention<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attention_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>) -> Result<Tensor>,
{
    let seq_len = q.dim(2)?;

    if seq_len <= ATTENTION_CHUNK_SIZE {
        // For short sequences, use the regular path
        return attention_fn(q, k, v, mask);
    }

    // Chunk the query to avoid OOM on long sequences
    let num_chunks = seq_len.div_ceil(ATTENTION_CHUNK_SIZE);
    let mut attn_chunks = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let offset = chunk_idx * ATTENTION_CHUNK_SIZE;
        let chunk_len = ATTENTION_CHUNK_SIZE.min(seq_len - offset);

        // Extract query chunk
        let q_chunk = q.narrow(2, offset, chunk_len)?;

        // Extract mask chunk if present
        let mask_chunk = mask
            .map(|m| {
                match m.rank() {
                    2 => {
                        // For 2D masks (seq_len, seq_len), narrow along dimension 0
                        m.narrow(0, offset, chunk_len)
                    }
                    3 => {
                        // For 3D masks (batch, seq_len, seq_len), narrow along dimension 1
                        m.narrow(1, offset, chunk_len)
                    }
                    4 => {
                        // For 4D masks (batch, heads, seq_len, seq_len), narrow along dimension 2
                        m.narrow(2, offset, chunk_len)
                    }
                    _ => m.narrow(2, offset, chunk_len), // Default to dimension 2
                }
            })
            .transpose()?;

        // Compute attention for this chunk
        let att_chunk = attention_fn(&q_chunk, k, v, mask_chunk.as_ref())?;

        attn_chunks.push(att_chunk);
    }

    // Concatenate all chunks along the sequence dimension
    Tensor::cat(&attn_chunks, 2)
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

pub struct SdpaParams {
    pub n_kv_groups: usize,
    pub softcap: Option<f32>,
    pub softmax_scale: f32,
    pub sliding_window: Option<usize>,
    pub sinks: Option<Tensor>,
}

/// Does this mask carry information the CUDA flash path would silently discard?
///
/// # Why this predicate has to exist
///
/// `flash_attn(&q, &k, &v, flash_params, sdpa_params)` — the CUDA branch of
/// [`Sdpa::run_attention`] — **takes no mask argument at all**. Anything the
/// caller passed in `mask` is DISCARDED there, with no error. The flash
/// kernels' only masking knobs are `causal` and `window_size_left/right`, and
/// all three are *batch-scalar*: one value for the whole batch. So no mask
/// whose content is not exactly "causal" or "windowed" is expressible there.
///
/// # Why rank >= 3, and not "any mask"
///
/// Rank is a proxy for *batch- or head-varying*, which is exactly the class
/// flash's batch-scalar knobs cannot express. Every mask in this tree that is
/// currently dropped and must not be has rank >= 3 — verified by reading each
/// producer, 2026-08-18:
/// * `CausalMasker::expand_mask` — the `[B, 1, q, k]` (siglip) and
///   `[B, H, q, k]` (idefics3) **PAD patch** masks of a variable-resolution
///   image batch. Dropping them makes every real patch take softmax weight from
///   its own image's padding.
/// * the conformer audio encoder's T5 **relative attention bias**
///   (`vision_models/conformer/encoder.rs:93-101`) — not a mask at all but a
///   learned additive logit bias, and it is passed on EVERY forward
///   (`encoder.rs:641-648`), not only when padding exists.
/// * mllama's aspect-ratio padding mask (`vision_models/mllama/vision.rs:650`)
///   — rank 3 when `bs == 1`, rank 4 otherwise. A rank-4-only predicate misses
///   the first, which is why this is `>= 3`.
/// * `CausalMasker::make_left_padded_causal_mask` — the `[B, 1, 1, k]` ragged
///   decode mask. A zero K row is not a masked row: it scores logit 0 and takes
///   real softmax weight.
///
/// Conversely every rank-2 mask that reaches the CUDA flash path is redundant
/// with a knob flash already has, so diverting it would cost throughput for no
/// correctness gain:
/// * `make_causal_mask_matrix` / `make_sliding_window_causal_mask_matrix` never
///   return a real rank-2 mask there at all — they return the `[1, 1]` sentinel,
///   gated on `using_flash_attn() && device.is_cuda()`.
/// * the `*_as_attn_bias` producers do build real rank-2 masks, and reach flash
///   from exactly one place, `vision_models/gemma3/text.rs:591`. Gemma-3 also
///   sets `sdpa_params.sliding_window` per layer (`gemma3/text.rs:145`), so
///   causality and the window are both already carried. (`gpt_oss.rs:782` uses
///   the same producers but never gets here: `sdpa_params.sinks` is `Some`, so
///   `run_attention` short-circuits into `sinks_attn` first.)
///
/// # Honest limit — what this leaves behind
///
/// `make_chunked_mask_matrix` builds a rank-2 **chunked** mask
/// (`vision_models/llama4/text.rs:694`) which flash cannot express either, and
/// which this predicate therefore still lets be dropped. Fixing it means moving
/// Llama-4 text off flash, a throughput trade that should be measured on a box
/// before it is shipped. Filed rather than bundled in here, precisely so it is
/// not an unmeasured regression smuggled in behind a correctness fix.
///
/// # Scope
///
/// The caller applies this **only to the CUDA conjunct**. The CPU branch of
/// `can_use_flash` routes to `cpu::run_flash_attn_cpu`, which *does* take
/// `mask` and honours it, so diverting it would be a pointless slowdown.
///
/// This supersedes the narrower `seq_len == 1 && rank 4 && dims[0] > 1` form on
/// `feat/dense-ragged-decode` (PR #122): every input that predicate diverts,
/// this one diverts too. Kept under the same name deliberately, so the two
/// collide textually at merge and a human reconciles them rather than one
/// silently winning.
pub(crate) fn mask_must_be_applied_as_bias(mask_dims: Option<&[usize]>) -> bool {
    match mask_dims {
        None => false,
        Some(dims) => dims.len() >= 3,
    }
}

pub struct Sdpa;

impl Sdpa {
    /// Computes softmax(QK^T*sqrt(d_k))V
    ///
    /// Inputs:
    /// - q: (b_sz, n_attn_heads, q_len, head_dim)
    /// - k: (b_sz, n_kv_heads, q_len, head_dim)
    /// - v: (b_sz, n_kv_heads, q_len, head_dim)
    ///
    /// The attention implementation is dispatched as follows:
    /// 1) If using flash attn (CUDA), use a flash attention V2/V3 kernel
    /// 2) If decoding and using a Metal device, use a fused kkernel
    /// 2) Otherwise, use the "naive" SDPA implementation (with optimized mask+softmax+scale application)
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        flash_params: Option<&FlashParams>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        // If sinks are present, dispatch to the sinks backend
        if let Some(sinks) = &sdpa_params.sinks {
            return sinks_attn(q, k, v, sinks, mask, flash_params, sdpa_params);
        }

        let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
        let (_, _, _, k_head_dim) = k.dims4()?;
        let (_, _, _, v_head_dim) = v.dims4()?;

        // The CUDA flash kernels accept a bounded envelope of shapes: FA2 takes
        // any head_dim that is a multiple of 8 up to 256, FA3 takes only
        // {64, 128, 256} and has no softcap kernel. Asking for anything else
        // used to enter the flash path and hard-`bail!` inside candle; now it
        // degrades to non-flash SDPA, which has no such restriction. See
        // `backends::flash` for the envelope and its provenance.
        //
        // (The CPU branch below is a separate, unrestricted implementation and
        // is deliberately not gated on this predicate.)
        //
        // The mask predicate is on the CUDA conjunct ONLY: `flash_attn` takes no
        // mask argument and would drop it, whereas the CPU branch's
        // `run_flash_attn_cpu` takes `mask` and honours it. See
        // `mask_must_be_applied_as_bias`.
        let can_use_flash = q.device().is_cpu()
            || q.device().is_cuda()
                && crate::using_flash_attn()
                && q.dtype() != DType::F32
                && cuda_flash_attn_supported(head_dim, k_head_dim, v_head_dim, sdpa_params.softcap)
                && !mask_must_be_applied_as_bias(mask.map(|m| m.dims()));

        if can_use_flash {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;

            if q.device().is_cpu() {
                match q.dtype() {
                    DType::F32 => {
                        return cpu::run_flash_attn_cpu::<f32>(&q, &k, &v, mask, sdpa_params);
                    }
                    DType::F16 => {
                        return cpu::run_flash_attn_cpu::<half::f16>(&q, &k, &v, mask, sdpa_params)
                    }
                    DType::BF16 => {
                        return cpu::run_flash_attn_cpu::<half::bf16>(
                            &q,
                            &k,
                            &v,
                            mask,
                            sdpa_params,
                        );
                    }
                    _ => {
                        return Err(candle_core::Error::Msg("Unsupported data type".into()));
                    }
                }
            } else {
                return flash_attn(&q, &k, &v, flash_params, sdpa_params)?.transpose(1, 2);
            }
        }

        // Diverting to non-flash SDPA is only safe if a real mask exists.
        //
        // `LayersMasker` skips materializing the causal mask whenever
        // `using_flash_attn() && device.is_cuda()`, handing back a `(1, 1)`
        // zeros placeholder instead (see `layers_masker.rs`) on the assumption
        // that the flash kernel applies causality itself. `naive_sdpa` does
        // `att.broadcast_add(mask)`, so that placeholder broadcasts to zeros and
        // the result is silently NON-causal — the exact failure mode this file
        // exists to prevent.
        //
        // A `(1, 1)` mask with `seq_len > 1` can only be that placeholder: a
        // genuinely bidirectional caller (vision/audio encoders) passes `None`,
        // which stays `None` and is correctly non-causal. So refuse loudly here
        // rather than return wrong numbers. This restores the pre-dispatch
        // behaviour for these shapes, which hard-`bail!`ed inside candle.
        if seq_len > 1
            && q.device().is_cuda()
            && crate::using_flash_attn()
            && mask.is_some_and(|m| m.dims() == [1, 1])
        {
            candle_core::bail!(
                "no CUDA flash backend accepts this shape (head_dim {head_dim}, k {k_head_dim}, \
                 v {v_head_dim}, softcap {:?}) and the causal mask was not materialized because \
                 the build enables flash-attn. Falling back to non-flash SDPA here would compute \
                 NON-causal attention. Rebuild without a flash feature to use this model, or add \
                 `--features flash-attn` if only flash-attn-v3 is enabled.",
                sdpa_params.softcap
            );
        }

        self.run_attention_noflash(q, k, v, mask, sdpa_params)
    }

    /// Same as `run_attention`, but no flash attention
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention_noflash(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
        let (_, _, _, k_head_dim) = k.dims4()?;
        let (_, _, _, v_head_dim) = v.dims4()?;

        // We can use Metal SDPA (vector/full) if the mask is the correct size and head dims match.
        // If the mask is provided, then softcapping isn't allowed - default back to naive SDPA
        // Softcapping is implemented for vector SDPA.
        let all_head_dims_match = head_dim == k_head_dim && k_head_dim == v_head_dim;
        let tgt_mask_shape = vec![b_sz, n_attn_heads, seq_len, k.dim(2)?];
        let can_use_mask = mask.is_none_or(|mask| {
            mask.layout().broadcast_as(tgt_mask_shape.clone()).is_ok()
                && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        });
        let valid_head_dims: &[usize] = if seq_len == 1 {
            &[32, 64, 72, 80, 96, 128, 256]
        } else {
            // Not sure why the full kernel doesn't like 256.
            // [32, 64, 72, 80, 96, 128, 256]
            &[32, 64, 72, 80, 96, 128]
        };
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && all_head_dims_match
            && valid_head_dims.contains(&head_dim)
            && can_use_mask
        {
            let mask = match mask {
                Some(mask) => Some(mask.broadcast_as(tgt_mask_shape)?),
                None => None,
            };
            return candle_nn::ops::sdpa(
                q,
                k,
                v,
                mask.as_ref(),
                false,
                sdpa_params.softmax_scale,
                sdpa_params.softcap.unwrap_or(1.0),
            );
        }

        let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;

        if mask.is_some_and(|x| x.rank() == 2) || mistralrs_quant::distributed::use_nccl() {
            return naive_sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask,
                sdpa_params,
            );
        }

        // TODO: bench?
        #[allow(unused)]
        if let (false, Device::Cuda(_), Some(cublaslt)) = (
            force_naive_sdpa(),
            q.device(),
            mistralrs_quant::cublaslt::CUBLASLT_CONTROLLER.get_for_device(q.device()),
        ) {
            #[cfg(feature = "cuda")]
            {
                maybe_synchronize(q.device())?;

                // Use chunked attention for cuBLASLt path
                let k_flat = k.flatten(0, 1)?;
                let v_flat = v.flatten(0, 1)?;

                chunked_attention(q, &k, &v, mask, |q_chunk, _k, _v, mask_chunk| {
                    // cuBLASLt batch matmul implementation requires inputs to be dims3
                    let (chunk_b_sz, chunk_n_heads, chunk_seq_len, chunk_head_dim) =
                        q_chunk.dims4()?;
                    let q_flat = q_chunk.flatten(0, 1)?;

                    let attention_bias = match mask_chunk {
                        Some(mask) if mask.rank() == 3 && mask.dims()[0] == 1 => {
                            Some(mask.repeat((chunk_n_heads, 1, 1))?)
                        }
                        Some(mask) if mask.rank() == 3 => Some(mask.clone()),
                        Some(mask) if mask.rank() == 4 => {
                            let tgt_shape =
                                vec![chunk_b_sz, chunk_n_heads, chunk_seq_len, k.dim(2)?];
                            Some(mask.broadcast_as(tgt_shape)?.flatten(0, 1)?)
                        }
                        Some(mask) => {
                            candle_core::bail!("cublaslt attn mask: rank must be 3 or 4")
                        }
                        None => None,
                    };

                    // If attention_bias is set, we fuse the add by giving it as the output matrix
                    // and setting beta to 1.0
                    let beta = match attention_bias.is_some() {
                        true => Some(1.0),
                        false => None,
                    };

                    // Batch matrix multiplication
                    // Fuse softmax scale and attention_bias add
                    let mut attention_scores = cublaslt.batch_matmul(
                        &k_flat,
                        &q_flat,
                        attention_bias.as_ref(),
                        Some(sdpa_params.softmax_scale / sdpa_params.softcap.unwrap_or(1.0)),
                        beta,
                        None,
                        None,
                    )?;
                    if let Some(softcap) = sdpa_params.softcap {
                        attention_scores = (attention_scores.tanh()? * softcap as f64)?;
                    }
                    attention_scores = candle_nn::ops::softmax_last_dim(&attention_scores)?;

                    let context_layer = cublaslt.batch_matmul(
                        &v_flat.t()?.contiguous()?,
                        &attention_scores,
                        // We save one allocation
                        Some(&q_flat),
                        None,
                        None,
                        None,
                        None,
                    )?;

                    // Reshape to dims4
                    context_layer.reshape((chunk_b_sz, chunk_n_heads, chunk_seq_len, v_head_dim))
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle_core::bail!("`cuda` feature is not enabled")
            }
        } else {
            naive_sdpa(q, &k, &v, mask, sdpa_params)
        }
    }
}

#[cfg(test)]
mod bias_dispatch_tests {
    use super::mask_must_be_applied_as_bias;

    /// Every shape below is the shape a named call site actually produces. They
    /// are not illustrative: if one of these reaches the CUDA flash path, that
    /// model's mask is dropped and the output is silently wrong.
    ///
    /// Mutation check (run 2026-08-18): make the predicate `false` for `Some(_)`
    /// — master's behaviour — and all seven rows here fail, naming the site.
    #[test]
    fn every_real_masked_call_site_goes_to_the_bias_path() {
        for (site, dims) in [
            // siglip.rs:512-518 — CausalMasker::expand_mask -> [bs, 1, q, k].
            // PAD patches of a variable-resolution image batch.
            ("siglip expand_mask", &[4usize, 1, 64, 64][..]),
            // idefics3/vision.rs:484-505 — expand_mask, then broadcast to
            // [bs, num_heads, q, k].
            ("idefics3 expand_mask", &[4, 16, 64, 64][..]),
            // A single image still has PAD patches inside a fixed patch grid,
            // so bs == 1 must divert too. `dims[0] > 1` would miss this.
            ("siglip, batch of one", &[1, 1, 64, 64][..]),
            // conformer/encoder.rs:93-101 — T5 relative attention bias, a
            // LEARNED additive logit bias, passed on EVERY forward
            // (encoder.rs:641-648), not just when padding exists.
            ("conformer relative bias", &[1, 8, 100, 100][..]),
            // mllama/vision.rs:650-662 — aspect-ratio padding mask. Rank 3 when
            // bs == 1 (no unsqueeze), rank 4 otherwise. A rank-4-only predicate
            // would miss the first of these.
            ("mllama, bs == 1", &[1, 1601, 1601][..]),
            ("mllama, bs > 1", &[2, 1, 1601, 1601][..]),
            // layers_masker.rs:221-256 — make_left_padded_causal_mask, the
            // ragged decode mask (PR #122's case). Subsumed here.
            ("ragged decode", &[8, 1, 1, 500][..]),
        ] {
            assert!(
                mask_must_be_applied_as_bias(Some(dims)),
                "{site}: mask {dims:?} carries information no flash kernel can \
                 express, so it must not reach the flash path"
            );
        }
    }

    /// Everything that must STAY on flash. Diverting any of these would be an
    /// unmeasured throughput regression for no correctness gain.
    ///
    /// Mutation check (run 2026-08-18): relax the predicate to `Some(_) => true`
    /// and this fires — that over-wide version is the one I rejected, because it
    /// moves Gemma-3 text off flash to fix a bug in a different model.
    #[test]
    fn nothing_that_works_today_is_diverted() {
        // Uniform decode / a genuinely bidirectional encoder: no mask at all.
        assert!(!mask_must_be_applied_as_bias(None));
        // The `Tensor::zeros((1, 1))` sentinel `layers_masker` returns *because*
        // flash applies causality itself. It carries no information.
        assert!(!mask_must_be_applied_as_bias(Some(&[1, 1])));
        // gemma3/text.rs:591 — a real rank-2 causal / SWA bias. Redundant on
        // flash: causality comes from the `causal` flag and the window from
        // `sdpa_params.sliding_window`, which gemma3 sets per layer.
        assert!(!mask_must_be_applied_as_bias(Some(&[7, 7])));
        assert!(!mask_must_be_applied_as_bias(Some(&[512, 512])));
    }

    /// The documented limit, pinned so it cannot be forgotten: a rank-2 CHUNKED
    /// mask (`make_chunked_mask_matrix`, used by `llama4/text.rs:694`) is also
    /// inexpressible on flash, and this predicate deliberately does NOT divert
    /// it — that fix needs a measured throughput number first.
    ///
    /// If this assertion ever flips, the limit has been closed and the doc
    /// comment's "Honest limit" section must go with it.
    #[test]
    fn the_chunked_rank2_mask_is_a_known_and_deliberate_gap() {
        assert!(
            !mask_must_be_applied_as_bias(Some(&[512, 512])),
            "llama4's chunked mask is rank 2 and is still dropped on flash — \
             documented, filed, not silently fixed"
        );
    }

    /// This predicate must be at least as strict as the one it supersedes on
    /// `feat/dense-ragged-decode` (PR #122): `seq_len == 1 && rank 4 &&
    /// dims[0] > 1`. Anything that one diverts, this one must divert.
    #[test]
    fn supersedes_the_pr122_predicate() {
        fn pr122(seq_len: usize, mask_dims: Option<&[usize]>) -> bool {
            match mask_dims {
                Some(dims) => seq_len == 1 && dims.len() == 4 && dims[0] > 1,
                None => false,
            }
        }
        for dims in [
            &[8usize, 1, 1, 500][..],
            &[2, 1, 1, 6][..],
            &[1, 1][..],
            &[7, 7][..],
            &[4, 1, 64, 64][..],
            &[1, 1, 1, 500][..],
        ] {
            if pr122(1, Some(dims)) {
                assert!(
                    mask_must_be_applied_as_bias(Some(dims)),
                    "{dims:?}: PR #122 diverts this; the wider predicate must too"
                );
            }
        }
        // And it is STRICTLY wider — at least one input differs, or the two
        // would be the same predicate and this fix would be a no-op.
        let vision = &[4usize, 1, 64, 64][..];
        assert!(
            !pr122(64, Some(vision)),
            "sanity: PR #122 leaves this on flash"
        );
        assert!(
            mask_must_be_applied_as_bias(Some(vision)),
            "the vision padding mask is exactly what widening buys"
        );
    }
}
