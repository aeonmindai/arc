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
//! - **Standard (`compress_ratio == 0`, layers 0/1/42 + MTP)**: sliding-window
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
//!   branch (`ARC_V4_WINDOW_ONLY` made no difference — layers 0/1/42 were the
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
//!
//! Returns `[B, n_heads, T_q, D]` matching plain SDPA's output shape so the
//! caller's post-attention inverse-RoPE + grouped `wo_a`/`wo_b` chain works
//! as-is.

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

/// Per-call configuration for V4 hybrid attention.
#[derive(Debug, Clone, Copy)]
pub struct Dsv4AttentionConfig {
    /// Per-layer compress ratio (0/4/128 → `Standard`/`Csa`/`Hca`).
    pub compress_ratio: CompressRatio,
    /// Sliding-window size for the always-on local branch (V4 default 128).
    pub sliding_window: usize,
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
    let t_k = k.dim(2)?;
    let window = cfg.sliding_window.max(1);
    let dev = q.device();

    // Absolute positions. `k` is the full cached sequence over `[0, t_k)`; the
    // current query block is its last `t_q` tokens, so query row `r` is at
    // position `q0 + r` with `q0 = t_k - t_q` (holds for prefill `t_q == t_k`
    // and decode `t_q == 1`).
    let q0 = t_k - t_q;

    // ---- Raw sliding-window branch mask: [t_q, t_k] -----------------------
    // query r attends raw key j iff (q0+r-window < j <= q0+r): causal AND
    // within the trailing `window` tokens. The diagonal (j == q0+r) is always
    // valid, so no query row is fully masked (no softmax NaN).
    let kp = Tensor::arange(0u32, t_k as u32, dev)?
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
    /// in the window slice fails the equality at every post-window length.
    /// Runs with (zero) sinks on both sides, matching deployment where
    /// `attn_sink` is always present — and required here: the no-sink CPU
    /// flash path NaNs on rows with a masked prefix (see the HCA test note).
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
            };
            let actual = dsv4_attention(&q, &k, &v, None, None, &flash, &sdpa, cfg)?;

            let e: Vec<f32> = expected.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let a: Vec<f32> = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            for (ev, av) in e.iter().zip(a.iter()) {
                assert!(
                    (ev - av).abs() < 1e-5,
                    "t_k={t_k}: {ev} != {av} (window slice off)"
                );
            }
        }
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
                        (ev - av).abs() < 1e-5,
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
        };
        let out = dsv4_attention(&q, &k, &v, Some(&comp), None, &flash, &sdpa, cfg)?;
        assert_eq!(out.dims(), &[b, h, t_q, d]);
        let data: Vec<f32> = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|x| x.is_finite()));
        Ok(())
    }
}
