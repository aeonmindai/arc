use candle_core::{Result, Tensor};
use mistralrs_quant::MatMul;

use crate::{
    attention::{repeat_kv, SdpaParams},
    pipeline::text_models_inputs_processor::FlashParams,
};

/// Fused attention with per-head sinks.
///
/// Dispatches to:
///   CUDA  -> flash_attn_sinks / flash_attn_sinks_varlen
///   Metal -> flash_attn_sinks_metal / flash_attn_sinks_varlen_metal
///   CPU   -> unfused matmul + softmax_with_sinks
///
/// Varlen is used when flash_params contains cu_seqlens_k for this device AND
/// q has batch > 1 AND the caller supplied no explicit mask.
///
/// The varlen backends derive causality and windowing *internally* from
/// `cu_seqlens` + `window_size` (bottom-right aligned; see
/// `flash_attn_sinks.cu`), and `sinks_attn_varlen` therefore takes no mask
/// argument at all. Routing a caller that DID supply an additive mask through
/// it silently discards that mask — padding columns in a ragged batch, a
/// model's own block-causality, a fixed-width graph-decode length mask. That
/// is a correctness bug, not a fast path, so an explicit mask pins the call to
/// the regular path, which honors it.
#[allow(unused_variables, clippy::too_many_arguments)]
pub(crate) fn sinks_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    mask: Option<&Tensor>,
    flash_params: Option<&FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b_sz, _n_heads, _q_len, _head_dim) = q.dims4()?;
    let window_size = sdpa_params.sliding_window.unwrap_or(0);

    // Detect varlen: no caller mask to honor AND flash_params has cu_seqlens_k
    // AND batch > 1.
    let is_varlen = mask.is_none()
        && b_sz > 1
        && flash_params
            .is_some_and(|fp| fp.cumulative_seqlens_k.contains_key(&q.device().location()));

    if is_varlen {
        return sinks_attn_varlen(
            q,
            k,
            v,
            sinks,
            flash_params.unwrap(),
            sdpa_params,
            window_size,
        );
    }

    // Non-varlen path
    sinks_attn_regular(q, k, v, sinks, mask, sdpa_params, window_size)
}

/// Non-varlen sinks attention: Q [B, H, q_len, D], K/V [B, kv_H, kv_len, D]
#[allow(unused_variables)]
fn sinks_attn_regular(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
    window_size: usize,
) -> Result<Tensor> {
    // The CUDA flash-sinks kernel now also instantiates head_dim=512 (V4:
    // symmetric 512/512, MQA). METAL DOES NOT — its kernel still stops at 256
    // (`metal/backend/paged_attention.rs:121`), so the Metal branch below is
    // guarded separately and a 512 head on Metal still takes the unfused
    // matmul + softmax_with_sinks path (GPU-capable via
    // SoftmaxWithSinks::cuda_fwd, same math as the old "cpu" fallback).
    let hd = q.dim(candle_core::D::Minus1)?;
    // ⚠️ TWO BACKENDS, TWO ENVELOPES — this was ONE shared flag, and widening it
    // for CUDA silently widened it for Metal too. Metal's sinks kernel stops at
    // 256: `metal_kernels/mod.rs:3025` errors on any other head_dim, and
    // `sdpa_with_sinks.metal` instantiates only {64,80,96,128,256}. Routing a
    // 512 head there turns a working unfused fallback into a hard
    // CompilationError. Each backend now advertises its own set.
    let flash_sinks_ok_cuda = matches!(hd, 64 | 80 | 96 | 112 | 128 | 192 | 256 | 512);
    let flash_sinks_ok_metal = matches!(hd, 64 | 80 | 96 | 112 | 128 | 192 | 256);
    #[cfg(not(feature = "cuda"))]
    let _ = flash_sinks_ok_cuda;
    #[cfg(not(feature = "metal"))]
    let _ = flash_sinks_ok_metal;

    #[cfg(feature = "cuda")]
    if q.device().is_cuda() && flash_sinks_ok_cuda {
        return mistralrs_paged_attn::flash_attn_sinks(
            q,
            k,
            v,
            Some(sinks),
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    #[cfg(feature = "metal")]
    if q.device().is_metal() && flash_sinks_ok_metal {
        return mistralrs_quant::flash_attn_sinks_metal(
            q,
            k,
            v,
            Some(sinks),
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    // CPU fallback: unfused matmul + softmax_with_sinks
    sinks_attn_cpu(q, k, v, sinks, mask, sdpa_params)
}

/// Varlen sinks attention: Q [B, H, max_q, D], K/V packed [total_kv, kv_H, D]
/// or K/V [1, kv_H, total_kv, D] (squeezed+transposed automatically).
#[allow(unused_variables)]
fn sinks_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    flash_params: &FlashParams,
    sdpa_params: &SdpaParams,
    window_size: usize,
) -> Result<Tensor> {
    let device = q.device();

    // Handle K/V shape: 4D [1, kv_H, total, D] -> 3D [total, kv_H, D], or 3D as-is
    let (k_packed, v_packed) = if k.dims().len() == 4 {
        (
            k.squeeze(0)?.transpose(0, 1)?,
            v.squeeze(0)?.transpose(0, 1)?,
        )
    } else {
        (k.clone(), v.clone())
    };

    // Get cu_seqlens from flash_params (already on GPU as U32, no conversion needed)
    let cu_seqlens_q = &flash_params.cumulative_seqlens_q[&device.location()];
    let cu_seqlens_k = &flash_params.cumulative_seqlens_k[&device.location()];

    // R1: head_dim guard (see sinks_attn_regular).
    let hd = q.dim(candle_core::D::Minus1)?;
    // ⚠️ TWO BACKENDS, TWO ENVELOPES — this was ONE shared flag, and widening it
    // for CUDA silently widened it for Metal too. Metal's sinks kernel stops at
    // 256: `metal_kernels/mod.rs:3025` errors on any other head_dim, and
    // `sdpa_with_sinks.metal` instantiates only {64,80,96,128,256}. Routing a
    // 512 head there turns a working unfused fallback into a hard
    // CompilationError. Each backend now advertises its own set.
    let flash_sinks_ok_cuda = matches!(hd, 64 | 80 | 96 | 112 | 128 | 192 | 256 | 512);
    let flash_sinks_ok_metal = matches!(hd, 64 | 80 | 96 | 112 | 128 | 192 | 256);
    #[cfg(not(feature = "cuda"))]
    let _ = flash_sinks_ok_cuda;
    #[cfg(not(feature = "metal"))]
    let _ = flash_sinks_ok_metal;

    #[cfg(feature = "cuda")]
    if device.is_cuda() && flash_sinks_ok_cuda {
        return mistralrs_paged_attn::flash_attn_sinks_varlen(
            q,
            &k_packed,
            &v_packed,
            Some(sinks),
            cu_seqlens_q,
            cu_seqlens_k,
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    #[cfg(feature = "metal")]
    if device.is_metal() && flash_sinks_ok_metal {
        return mistralrs_quant::flash_attn_sinks_varlen_metal(
            q,
            &k_packed,
            &v_packed,
            Some(sinks),
            cu_seqlens_q,
            cu_seqlens_k,
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    // CPU fallback: per-sequence loop (to_vec1 is fine on CPU path)
    let cu_q_vec: Vec<u32> = cu_seqlens_q.to_vec1()?;
    let cu_k_vec: Vec<u32> = cu_seqlens_k.to_vec1()?;
    sinks_attn_cpu_varlen(
        q,
        &k_packed,
        &v_packed,
        sinks,
        sdpa_params,
        &cu_q_vec,
        &cu_k_vec,
        window_size,
    )
}

/// Additive `[1, 1, q_len, kv_len]` mask reproducing the varlen kernels' own
/// masking rule, so the CPU fallback computes the same thing the CUDA/Metal
/// varlen kernels do instead of an unmasked (non-causal) attention.
///
/// Rule, transcribed from `mistralrs-paged-attn/src/cuda/flash_attn_sinks.cu`
/// (`flash_attn_sinks_varlen_kernel`): with `kv_offset = kv_len - q_len`
/// (bottom-right alignment, so the last query row sees the whole cache),
///   * causal — `kv_pos < my_kv_end   = q_row + kv_offset + 1`
///   * window — `kv_pos >= my_kv_start = max(0, q_row + kv_offset - window + 1)`
///     when `window_size > 0`, else `0`.
fn varlen_causal_mask(
    q_len: usize,
    kv_len: usize,
    window_size: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let kv_offset = kv_len as i64 - q_len as i64;
    let mut data = vec![0f32; q_len * kv_len];
    for (r, row) in data.chunks_mut(kv_len).enumerate() {
        let end = r as i64 + kv_offset + 1;
        let start = if window_size > 0 {
            (r as i64 + kv_offset - window_size as i64 + 1).max(0)
        } else {
            0
        };
        for (j, cell) in row.iter_mut().enumerate() {
            let j = j as i64;
            if j >= end || j < start {
                *cell = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)
}

/// CPU fallback: unfused matmul + softmax_with_sinks
fn sinks_attn_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
    let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;

    let att = MatMul.matmul_affine_mul(q, &k.t()?, sdpa_params.softmax_scale.into())?;
    // R1: `softmax_with_sinks` expects sinks shaped [num_heads] AND matching the
    // logits dtype. V4 pre-shapes sinks as [1, n_heads, 1, 1] in F32 (for the
    // flash kernel), so flatten + cast to the logits dtype here. Then cast the
    // softmax result back to V's dtype before the value matmul.
    let sinks = sinks.flatten_all()?.to_dtype(att.dtype())?;
    let att = mistralrs_quant::softmax_with_sinks(&att, &sinks, mask)?;
    let att = att.to_dtype(v.dtype())?;
    MatMul.matmul(&att, &v)
}

/// CPU fallback for varlen: per-sequence unfused loop.
///
/// Each sequence is masked with `varlen_causal_mask`, matching the fused
/// varlen kernels. Passing `None` here (the pre-fix behavior) made every
/// `b_sz > 1` prefill on this path fully bidirectional — tokens attended to
/// their own future — so batched prefill was silently non-causal.
#[allow(clippy::too_many_arguments)]
fn sinks_attn_cpu_varlen(
    q: &Tensor,
    k_packed: &Tensor,
    v_packed: &Tensor,
    sinks: &Tensor,
    sdpa_params: &SdpaParams,
    cu_q: &[u32],
    cu_k: &[u32],
    window_size: usize,
) -> Result<Tensor> {
    let (b_sz, num_heads, max_q, head_dim) = q.dims4()?;
    let device = q.device();
    let mut outputs = Vec::with_capacity(b_sz);

    for i in 0..b_sz {
        let q_len = (cu_q[i + 1] - cu_q[i]) as usize;
        let kv_start = cu_k[i] as usize;
        let kv_len = (cu_k[i + 1] - cu_k[i]) as usize;

        // Extract this sequence's Q [1, H, q_len, D]
        let qi = q.narrow(0, i, 1)?;
        let qi = qi.narrow(2, 0, q_len)?;

        // Extract this sequence's K/V from packed [total_kv, kv_H, D]
        let ki = k_packed
            .narrow(0, kv_start, kv_len)?
            .transpose(0, 1)?
            .unsqueeze(0)?;
        let vi = v_packed
            .narrow(0, kv_start, kv_len)?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        let mask = varlen_causal_mask(q_len, kv_len, window_size, qi.dtype(), device)?;
        let oi = sinks_attn_cpu(&qi, &ki, &vi, sinks, Some(&mask), sdpa_params)?;

        // Pad back to max_q
        if q_len < max_q {
            let pad = Tensor::zeros((1, num_heads, max_q - q_len, head_dim), qi.dtype(), device)?;
            outputs.push(Tensor::cat(&[&oi, &pad], 2)?);
        } else {
            outputs.push(oi);
        }
    }

    Tensor::cat(&outputs, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn mk(dims: &[usize], seed: f32, dev: &Device) -> Result<Tensor> {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * seed).sin()).collect();
        Tensor::from_vec(data, dims, dev)
    }

    fn params(head_dim: usize, n_kv_groups: usize, window: Option<usize>) -> SdpaParams {
        SdpaParams {
            n_kv_groups,
            softcap: None,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: window,
            sinks: None,
        }
    }

    fn varlen_params(seqlens: &[u32], dev: &Device) -> FlashParams {
        let mut acc = 0u32;
        let mut cu = vec![0u32];
        for l in seqlens {
            acc += l;
            cu.push(acc);
        }
        let cu = Tensor::from_vec(cu, seqlens.len() + 1, dev).unwrap();
        let mut q_map = HashMap::new();
        let mut k_map = HashMap::new();
        q_map.insert(dev.location(), cu.clone());
        k_map.insert(dev.location(), cu);
        FlashParams {
            max_q: seqlens.iter().copied().max().unwrap_or(0),
            max_k: seqlens.iter().copied().max().unwrap_or(0),
            cumulative_seqlens_q: q_map,
            cumulative_seqlens_k: k_map,
            causal: true,
        }
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
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// The CPU varlen fallback must apply the same causal (+ sliding-window)
    /// masking the fused varlen kernels apply internally. It used to pass
    /// `mask = None`, i.e. every token in a `b_sz > 1` prefill attended to its
    /// own future.
    #[test]
    fn cpu_varlen_fallback_is_causal() -> Result<()> {
        let dev = Device::Cpu;
        let (h, d, seq) = (2usize, 16usize, 6usize);
        let b = 2usize;
        let sinks = Tensor::from_vec(vec![0.3f32, 0.7], (1, h, 1, 1), &dev)?;

        let q = mk(&[b, h, seq, d], 0.07, &dev)?;
        // Packed K/V, the layout the varlen path documents: [total_kv, kv_H, D].
        let k_packed = mk(&[b * seq, h, d], 0.11, &dev)?;
        let v_packed = mk(&[b * seq, h, d], 0.17, &dev)?;

        for window in [0usize, 3] {
            let sdpa = SdpaParams {
                sinks: Some(sinks.clone()),
                ..params(d, 1, if window == 0 { None } else { Some(window) })
            };
            let flash = varlen_params(&[seq as u32; 2], &dev);
            let got = sinks_attn(&q, &k_packed, &v_packed, &sinks, None, Some(&flash), &sdpa)?;

            // Reference: per sequence, an explicit causal (+ window) mask.
            for i in 0..b {
                let qi = q.narrow(0, i, 1)?;
                let ki = k_packed
                    .narrow(0, i * seq, seq)?
                    .transpose(0, 1)?
                    .unsqueeze(0)?;
                let vi = v_packed
                    .narrow(0, i * seq, seq)?
                    .transpose(0, 1)?
                    .unsqueeze(0)?;
                let mask = varlen_causal_mask(seq, seq, window, qi.dtype(), &dev)?;
                let want = sinks_attn_cpu(&qi, &ki, &vi, &sinks, Some(&mask), &sdpa)?;
                let diff = max_abs_diff(&flat(&got.narrow(0, i, 1)?), &flat(&want));
                assert_eq!(
                    diff, 0.0,
                    "window={window} seq {i}: varlen fallback disagrees with the masked \
                     per-sequence reference (max abs diff {diff})"
                );

                // Teeth: the unmasked (pre-fix) computation is a different
                // answer, by a wide margin.
                let unmasked = sinks_attn_cpu(&qi, &ki, &vi, &sinks, None, &sdpa)?;
                let signal = max_abs_diff(&flat(&want), &flat(&unmasked));
                assert!(
                    signal > 1e-2,
                    "window={window} seq {i}: masked and unmasked attention are \
                     indistinguishable (diff {signal}); this test has no teeth"
                );
            }
        }
        Ok(())
    }

    /// A caller-supplied mask must pin the call to the regular path. The
    /// varlen backend has no mask parameter, so routing there would drop it
    /// (and, for the `[B, kv_H, T, D]` K/V every in-tree caller actually
    /// passes, `squeeze(0)` is a no-op at `B > 1` and the narrow then fails
    /// outright).
    #[test]
    fn explicit_mask_keeps_the_call_off_the_varlen_path() -> Result<()> {
        let dev = Device::Cpu;
        let (b, h, seq, d) = (2usize, 2usize, 5usize, 16usize);
        let sinks = Tensor::from_vec(vec![0.25f32, 0.5], (1, h, 1, 1), &dev)?;
        let sdpa = SdpaParams {
            sinks: Some(sinks.clone()),
            ..params(d, h, None)
        };

        let q = mk(&[b, h, seq, d], 0.05, &dev)?;
        let k = mk(&[b, 1, seq, d], 0.13, &dev)?;
        let v = mk(&[b, 1, seq, d], 0.23, &dev)?;
        let mask = varlen_causal_mask(seq, seq, 0, q.dtype(), &dev)?;
        let flash = varlen_params(&[seq as u32; 2], &dev);

        let got = sinks_attn(&q, &k, &v, &sinks, Some(&mask), Some(&flash), &sdpa)?;
        let want = sinks_attn_cpu(&q, &k, &v, &sinks, Some(&mask), &sdpa)?;
        assert_eq!(
            max_abs_diff(&flat(&got), &flat(&want)),
            0.0,
            "a masked call did not take the regular (mask-honoring) path"
        );
        Ok(())
    }
}
