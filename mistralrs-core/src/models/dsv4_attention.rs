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
//! - **Standard (`compress_ratio == 0`, layers 0/1/42)**: plain causal MLA SDPA
//!   over the full K/V. The caller supplies the causal mask.
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

use super::deepseek4::CompressRatio;
use crate::attention::SdpaParams;
use crate::layers::Sdpa;
use crate::pipeline::text_models_inputs_processor::FlashParams;

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
    // ---- Standard layers: dense causal MLA via plain SDPA. ----------------
    if cfg.compress_ratio == CompressRatio::Standard {
        return Sdpa.run_attention(q, k, v, attention_mask, Some(flash_params), sdpa_params);
    }

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

    Sdpa.run_attention(q, &k_cat, &v_cat, Some(&mask), Some(flash_params), sdpa_params)
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

    /// Standard dispatch is exactly plain SDPA.
    #[test]
    fn standard_dispatch_matches_plain_sdpa() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, t, d) = (1, 2, 8, 16);
        let q = mk(b, h, t, d, 0.1, &device)?;
        let k = mk(b, 1, t, d, 0.2, &device)?;
        let v = mk(b, 1, t, d, 0.3, &device)?;

        let sdpa = sdpa_params(d, h);
        let flash = empty_flash_params();
        let expected = Sdpa.run_attention(&q, &k, &v, None, Some(&flash), &sdpa)?;

        let cfg = Dsv4AttentionConfig {
            compress_ratio: CompressRatio::Standard,
            sliding_window: 4,
        };
        let actual = dsv4_attention(&q, &k, &v, None, None, &flash, &sdpa, cfg)?;

        let e: Vec<f32> = expected.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let a: Vec<f32> = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        for (ev, av) in e.iter().zip(a.iter()) {
            assert!((ev - av).abs() < 1e-5, "{ev} != {av}");
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
