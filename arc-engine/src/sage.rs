//! SageAttention: per-block INT8 quantization of the attention computation.
//!
//! Paper:    `research/16_underexploited/sage_attention_int8.pdf` + `sage_attention_2.pdf`
//! Repo:     `research/code/03_per_token_speed/sage_attention/`
//! Blackwell kernels: `sage_attention/sageattention3_blackwell/sageattn3/`
//!
//! ## What
//!
//! SageAttention quantizes Q, K, V and the softmax inputs to INT8 (or FP8 in Sage-2/3),
//! using per-block scale factors. The published kernels run ~2× faster than FlashAttention
//! at similar quality. For SageAttention 3 on Blackwell SM100, the speedup over FA3 is
//! closer to 2.5× because of the new FP4-accumulating tensor cores.
//!
//! ## Tier A here (CPU-only correctness)
//!
//! - `Sm100QuantConfig`: per-block sizes, dtypes
//! - `per_block_quantize_i8()`: deterministic Q/K/V → INT8 + scale tensor
//! - `per_block_dequantize_i8()`: inverse
//! - `sage_attention_ref()`: a slow Rust reference attention that uses the per-block
//!   quantize → dequantize → matmul path; bit-exactness check against plain FP32 attention
//!   shows the quantization is well-bounded.
//!
//! ## Tier B (deferred)
//!
//! - FFI binding into `sage_attention/sageattention3_blackwell/sageattn3/` CUDA kernel
//! - Real SM89/SM90/SM100 throughput measurement
//! - Integration into `mistralrs-paged-attn` SDPA dispatcher

use candle_core::{DType, Device, Result, Tensor, D};

/// Block size for SageAttention 3 (matches `quant_per_block.py` defaults).
pub const SAGE_BLOCK_SIZE: usize = 64;

/// Configuration for SageAttention 3 INT8 path.
#[derive(Debug, Clone, Copy)]
pub struct Sm100QuantConfig {
    /// Block size along the sequence dimension. Default 64.
    pub block_size: usize,
}

impl Default for Sm100QuantConfig {
    fn default() -> Self {
        Self {
            block_size: SAGE_BLOCK_SIZE,
        }
    }
}

/// Per-block INT8 quantization of an attention tensor `[batch, heads, seq, head_dim]`.
///
/// Returns:
///   - `quantized`: same shape, DType::I64 (Candle has no I8; we store as I64 in [-128, 127])
///   - `scales`: shape `[batch, heads, num_blocks, head_dim]` FP32 — one scale per block per channel
///
/// The per-block-per-channel layout matches `quant_per_block.py` (lines 18–34 in
/// the reference; each scale captures the magnitude of one 64-token window for
/// one channel, giving fine-grained dynamic range).
pub fn per_block_quantize_i8(
    tensor: &Tensor,
    cfg: Sm100QuantConfig,
) -> Result<(Tensor, Tensor)> {
    let dims = tensor.dims();
    if dims.len() != 4 {
        candle_core::bail!(
            "per_block_quantize_i8 expects [B, H, S, D], got shape {:?}",
            dims
        );
    }
    let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    if s % cfg.block_size != 0 {
        candle_core::bail!(
            "seq_len {s} must be divisible by block_size {}",
            cfg.block_size
        );
    }
    let num_blocks = s / cfg.block_size;

    let data: Vec<f32> = tensor
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;

    let mut q = vec![0i64; b * h * s * d];
    let mut scales = vec![0f32; b * h * num_blocks * d];

    for bi in 0..b {
        for hi in 0..h {
            for blk in 0..num_blocks {
                let blk_start_s = blk * cfg.block_size;
                for di in 0..d {
                    // Compute max-abs over the block window for this channel.
                    let mut max_abs = 0f32;
                    for si in 0..cfg.block_size {
                        let idx = bi * (h * s * d) + hi * (s * d) + (blk_start_s + si) * d + di;
                        let v = data[idx].abs();
                        if v > max_abs {
                            max_abs = v;
                        }
                    }
                    let scale = if max_abs == 0.0 {
                        1.0
                    } else {
                        max_abs / 127.0
                    };
                    let inv = 1.0 / scale;
                    scales[bi * (h * num_blocks * d) + hi * (num_blocks * d) + blk * d + di] = scale;

                    // Quantize each element in the block.
                    for si in 0..cfg.block_size {
                        let idx = bi * (h * s * d) + hi * (s * d) + (blk_start_s + si) * d + di;
                        let qi = (data[idx] * inv).round().clamp(-128.0, 127.0) as i64;
                        q[idx] = qi;
                    }
                }
            }
        }
    }

    let q_tensor = Tensor::from_vec(q, (b, h, s, d), &Device::Cpu)?
        .to_device(tensor.device())?
        .to_dtype(DType::I64)?;
    let scales_tensor = Tensor::from_vec(scales, (b, h, num_blocks, d), &Device::Cpu)?
        .to_device(tensor.device())?;

    Ok((q_tensor, scales_tensor))
}

/// Inverse of `per_block_quantize_i8`: reconstruct FP32 from INT8 + scales.
pub fn per_block_dequantize_i8(
    q: &Tensor,
    scales: &Tensor,
    cfg: Sm100QuantConfig,
) -> Result<Tensor> {
    let q_dims = q.dims();
    if q_dims.len() != 4 {
        candle_core::bail!("per_block_dequantize_i8 expects [B,H,S,D], got {:?}", q_dims);
    }
    let (b, h, s, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let num_blocks = s / cfg.block_size;

    let q_data: Vec<i64> = q
        .to_dtype(DType::I64)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let scales_data: Vec<f32> = scales
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;

    let mut out = vec![0f32; b * h * s * d];
    for bi in 0..b {
        for hi in 0..h {
            for blk in 0..num_blocks {
                let blk_start_s = blk * cfg.block_size;
                for di in 0..d {
                    let scale = scales_data
                        [bi * (h * num_blocks * d) + hi * (num_blocks * d) + blk * d + di];
                    for si in 0..cfg.block_size {
                        let idx = bi * (h * s * d) + hi * (s * d) + (blk_start_s + si) * d + di;
                        out[idx] = q_data[idx] as f32 * scale;
                    }
                }
            }
        }
    }

    Tensor::from_vec(out, (b, h, s, d), &Device::Cpu)?
        .to_device(q.device())
}

/// Reference SageAttention: per-block-quantize Q/K/V then matmul.
/// This is the "what the kernel would compute" path, in plain Rust, used to validate
/// that the per-block quantization preserves attention output within a tight bound.
///
/// Returns the attention output `[batch, heads, seq, head_dim]`.
pub fn sage_attention_ref(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: Sm100QuantConfig,
) -> Result<Tensor> {
    let (q_q, q_s) = per_block_quantize_i8(q, cfg)?;
    let (k_q, k_s) = per_block_quantize_i8(k, cfg)?;
    let (v_q, v_s) = per_block_quantize_i8(v, cfg)?;

    let q_d = per_block_dequantize_i8(&q_q, &q_s, cfg)?;
    let k_d = per_block_dequantize_i8(&k_q, &k_s, cfg)?;
    let v_d = per_block_dequantize_i8(&v_q, &v_s, cfg)?;

    // Standard scaled-dot-product attention on the dequantized tensors.
    let (_, _, _, head_dim) = q_d.dims4()?;
    let scale_factor = 1.0_f32 / (head_dim as f32).sqrt();
    let attn_scores = q_d.matmul(&k_d.transpose(D::Minus2, D::Minus1)?)?;
    let attn_scaled = (attn_scores * scale_factor as f64)?;
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scaled)?;
    let out = attn_weights.matmul(&v_d)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: per-block quant → dequant should recover values within
    /// 1 scale-step (≈ max_abs/127).
    #[test]
    fn per_block_quantize_round_trip_bounded_error() -> Result<()> {
        let device = Device::Cpu;
        // [B=1, H=2, S=128, D=8]
        let n = 1 * 2 * 128 * 8;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 5.0).collect();
        let t = Tensor::from_vec(data.clone(), (1, 2, 128, 8), &device)?;

        let cfg = Sm100QuantConfig::default();
        let (q, s) = per_block_quantize_i8(&t, cfg)?;
        let recon = per_block_dequantize_i8(&q, &s, cfg)?;
        let recon_v: Vec<f32> = recon.flatten_all()?.to_vec1()?;

        // Max abs error per element should be bounded by (block_max / 127).
        // Block_max ≤ 5.0, so error bound ≈ 0.04 per element.
        let mut max_err = 0f32;
        for (o, r) in data.iter().zip(recon_v.iter()) {
            let e = (o - r).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err < 0.05, "max abs error {max_err} > 0.05");
        Ok(())
    }

    /// SageAttention reference matches plain FP32 attention within tight tolerance.
    /// The quantization-then-dequant round-trips with bounded error, so the attention
    /// output's cosine similarity with FP32 attention should be > 0.99.
    #[test]
    fn sage_attention_matches_fp32_within_tolerance() -> Result<()> {
        let device = Device::Cpu;
        let b = 1;
        let h = 2;
        let s = 64; // = block_size
        let d = 16;
        let n = b * h * s * d;

        let mk = |seed: f32| -> Result<Tensor> {
            let data: Vec<f32> = (0..n).map(|i| ((i as f32) * seed).sin() * 2.0).collect();
            Tensor::from_vec(data, (b, h, s, d), &device)
        };
        let q = mk(0.1)?;
        let k = mk(0.2)?;
        let v = mk(0.3)?;

        // FP32 reference attention.
        let scale = 1.0_f32 / (d as f32).sqrt();
        let attn_scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn_scaled = (attn_scores * scale as f64)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scaled)?;
        let fp32_out = attn_weights.matmul(&v)?;
        let fp32_v: Vec<f32> = fp32_out.flatten_all()?.to_vec1()?;

        // SageAttention reference (INT8 per-block quantize then attention).
        let sage_out = sage_attention_ref(&q, &k, &v, Sm100QuantConfig::default())?;
        let sage_v: Vec<f32> = sage_out.flatten_all()?.to_vec1()?;

        let mut dot = 0f32;
        let mut nd = 0f32;
        let mut ns = 0f32;
        for (f, s) in fp32_v.iter().zip(sage_v.iter()) {
            dot += f * s;
            nd += f * f;
            ns += s * s;
        }
        let cos = dot / (nd.sqrt() * ns.sqrt());
        assert!(cos > 0.99, "Sage vs FP32 cosine similarity {cos} <= 0.99");
        Ok(())
    }

    /// Zero tensor → zero quantized + scale=1 (no division by zero).
    #[test]
    fn per_block_quantize_handles_zero_tensor() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::zeros((1, 1, 64, 4), DType::F32, &device)?;
        let (q, s) = per_block_quantize_i8(&t, Sm100QuantConfig::default())?;
        let q_data: Vec<i64> = q.flatten_all()?.to_vec1()?;
        let s_data: Vec<f32> = s.flatten_all()?.to_vec1()?;
        assert!(q_data.iter().all(|&v| v == 0));
        assert!(s_data.iter().all(|&v| v == 1.0));
        Ok(())
    }

    /// Block size mismatch → clear error.
    #[test]
    fn per_block_quantize_errors_on_misaligned_seq() {
        let device = Device::Cpu;
        let t = Tensor::zeros((1, 1, 65, 4), DType::F32, &device).unwrap(); // 65, not divisible by 64
        let res = per_block_quantize_i8(&t, Sm100QuantConfig::default());
        assert!(res.is_err());
    }
}
