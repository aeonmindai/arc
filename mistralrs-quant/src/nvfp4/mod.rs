//! NVFP4: NVIDIA Blackwell-native FP4 quantization format.
//!
//! Format overview (per NVIDIA cuBLAS docs and Open Compute MX spec):
//! - **Values**: FP4 E2M1, 16 distinct values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
//! - **Block size**: 16 elements per scale (smaller than MXFP4's 32 — finer per-block granularity)
//! - **Per-block scale**: FP8 E4M3 (1 byte, range ~6e-3 .. 448), bias = 7
//! - **Per-tensor scale**: single FP32, multiplied into every dequant
//!
//! Effective bits per weight ≈ 4 + 8/16 + tiny = 4.5 bits/value (vs MXFP4 at 4 + 8/32 = 4.25).
//! In return, the FP8 per-block scale gives ~2× the dynamic range of MXFP4's power-of-2 scale,
//! which translates to lower perplexity drift at the same effective bit width.
//!
//! ## Tier A (this module)
//! - Pure-Rust software implementation of quantize / dequantize / matmul
//! - Round-trip tests vs FP32 reference
//! - Composes via the `QuantMethod` trait
//!
//! ## Tier B (deferred to RUN-136 / RUN-137)
//! - cublasLtMatmul FP4 path on SM100 (Blackwell)
//! - Real B200 throughput vs FP8 baseline
//!
//! References:
//! - `research/05_nvfp4_blackwell/mx_microscaling_data_formats.pdf`
//! - `research/code/06_foundation/tensorrt_llm/cpp/tensorrt_llm/kernels/quantization/`

use std::{
    borrow::Cow,
    io::Cursor,
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{quantized::GgmlDType, DType, Device, Result, Tensor};
use candle_nn::Linear;

use crate::{
    generate_isq, generate_isq_imatrix,
    hqq::{ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    utils::{deserialize_tensor, serialize_tensor, version_is_compatible, UQFF_VERSION},
    AfqBits, AfqGroupSize, AfqLayer, FP8Linear, GgufMatMul, HqqAxis, HqqBits, HqqConfig, HqqLayer,
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig, QuantizedSerde,
    QuantizedSerdeType, ShardedVarBuilder, UnquantLinear,
};

/// NVFP4 block size: 16 FP4 values per FP8 scale.
pub const NVFP4_BLOCK_SIZE: usize = 16;

pub(crate) const N_BITS: usize = 4;

/// FP4 E2M1 decode table (positive half).
/// Negative values share these magnitudes with sign bit set (indices 8..15).
const FP4_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// FP8 E4M3 → f32 decode (256-entry table).
///
/// E4M3 spec (per OCP FP8):
///   - 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits
///   - Special: 0x7F = NaN, 0xFF = NaN; no infinities (saturating)
///   - Subnormals when exp==0: value = sign * 2^(1-7) * (mantissa / 8)
///   - Normals: value = sign * 2^(exp-7) * (1 + mantissa/8)
const FP8_E4M3_LUT: [f32; 256] = build_fp8_e4m3_lut();

const fn build_fp8_e4m3_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    let mut byte: u32 = 0;
    while byte < 256 {
        let sign = (byte >> 7) & 1;
        let exp = (byte >> 3) & 0xF;
        let mant = byte & 0x7;
        let sign_f = if sign == 1 { -1.0 } else { 1.0 };

        let value: f32 = if exp == 0 && mant == 0 {
            0.0
        } else if byte == 0x7F || byte == 0xFF {
            // E4M3 reserves these for NaN. We map to 0 to keep arithmetic well-defined;
            // a real implementation should never produce these from clean weights.
            0.0
        } else if exp == 0 {
            // Subnormal: 2^(1-7) * (mant/8) = mant / 1024
            sign_f * (mant as f32) * (1.0 / 1024.0)
        } else {
            // Normal: 2^(exp-7) * (1 + mant/8)
            let exp_factor = const_powi2(exp as i32 - 7);
            let mant_factor = 1.0 + (mant as f32) * 0.125;
            sign_f * exp_factor * mant_factor
        };
        lut[byte as usize] = value;
        byte += 1;
    }
    lut
}

/// 2^n for small integer n, usable in const context.
const fn const_powi2(n: i32) -> f32 {
    if n >= 0 {
        let mut acc: f32 = 1.0;
        let mut i = 0;
        while i < n {
            acc *= 2.0;
            i += 1;
        }
        acc
    } else {
        let mut acc: f32 = 1.0;
        let mut i = 0;
        while i < -n {
            acc *= 0.5;
            i += 1;
        }
        acc
    }
}

/// Quantize an f32 value to nearest FP8 E4M3 byte.
/// Returns the FP8 byte such that `FP8_E4M3_LUT[byte]` is closest to `val`.
fn quantize_fp8_e4m3(val: f32) -> u8 {
    if val == 0.0 {
        return 0;
    }
    let sign_bit: u8 = if val < 0.0 { 0x80 } else { 0x00 };
    let abs = val.abs();

    // E4M3 max representable = 1.75 * 2^8 = 448.0
    let abs_clamped = abs.min(448.0);

    // Linear search through the positive table is fine for occasional quantization
    // (we only quantize per-block at quant time, not per-token at inference).
    let mut best_byte: u8 = 1; // skip 0
    let mut best_diff = f32::INFINITY;
    for byte in 1u8..=126u8 {
        let candidate = FP8_E4M3_LUT[byte as usize];
        if candidate <= 0.0 {
            continue;
        }
        let diff = (candidate - abs_clamped).abs();
        if diff < best_diff {
            best_diff = diff;
            best_byte = byte;
        }
    }
    best_byte | sign_bit
}

/// Quantize an f32 to the nearest FP4 E2M1 nibble (0..15).
fn quantize_to_fp4(val: f32) -> u8 {
    let sign = val < 0.0;
    let abs_val = val.abs();

    let nibble = if abs_val < 0.25 {
        0 // 0.0
    } else if abs_val < 0.75 {
        1 // 0.5
    } else if abs_val < 1.25 {
        2 // 1.0
    } else if abs_val < 1.75 {
        3 // 1.5
    } else if abs_val < 2.5 {
        4 // 2.0
    } else if abs_val < 3.5 {
        5 // 3.0
    } else if abs_val < 5.0 {
        6 // 4.0
    } else {
        7 // 6.0
    };

    if sign && nibble != 0 {
        nibble | 0x08
    } else {
        nibble
    }
}

/// Decode an FP4 nibble (0..15) back to f32.
#[inline]
fn dequant_fp4(nibble: u8) -> f32 {
    let mag = FP4_VALUES[(nibble & 0x07) as usize];
    if (nibble & 0x08) != 0 {
        -mag
    } else {
        mag
    }
}

#[derive(Debug)]
pub struct NVFP4Layer {
    /// Packed FP4 weights: [N, K/2] u8 — two FP4 nibbles per byte (low = k, high = k+1).
    blocks: Tensor,
    /// FP8 E4M3 per-block scales: [N, K/16] u8.
    scales: Tensor,
    /// Per-tensor FP32 scale (single scalar). Multiplied into every dequant.
    /// Captured as a Tensor of shape [] for serde simplicity.
    tensor_scale: Tensor,
    /// Optional bias: [N].
    bias: Option<Tensor>,
}

impl NVFP4Layer {
    /// Quantize an unquantized weight tensor [N, K] to NVFP4 format.
    pub fn quantize(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight_f32 = weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let dims = weight_f32.dims2()?;
        let (n, k) = (dims.0, dims.1);

        if k % NVFP4_BLOCK_SIZE != 0 {
            candle_core::bail!(
                "NVFP4 quantization requires K ({k}) divisible by block size ({NVFP4_BLOCK_SIZE})"
            );
        }

        let weight_data: Vec<f32> = weight_f32.flatten_all()?.to_vec1()?;

        // Pick per-tensor scale: largest absolute value in the tensor divided by the
        // largest representable FP8 scale * FP4 max. This keeps every block scale
        // within FP8 range (~448).
        let global_max = weight_data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        // FP4 max = 6.0; FP8 max ≈ 448. Target: max_block_scale * 6 ≈ global_max.
        // So tensor_scale chosen so that block_scale = (block_max / 6) / tensor_scale
        // stays ≤ 448.
        let tensor_scale_f32 = if global_max == 0.0 {
            1.0
        } else {
            // Allow a small margin (0.95 of FP8 max) to avoid clipping the very largest blocks.
            (global_max / (6.0 * 448.0 * 0.95)).max(1.0e-12)
        };

        let num_blocks_per_row = k / NVFP4_BLOCK_SIZE;
        let k_half = k / 2;

        use rayon::prelude::*;
        let row_results: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
            .into_par_iter()
            .map(|row| {
                let row_offset = row * k;
                let mut row_packed = vec![0u8; k_half];
                let mut row_scales = vec![0u8; num_blocks_per_row];

                for (blk, scale_byte) in row_scales.iter_mut().enumerate() {
                    let blk_start = row_offset + blk * NVFP4_BLOCK_SIZE;
                    let block = &weight_data[blk_start..blk_start + NVFP4_BLOCK_SIZE];

                    let max_abs = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

                    // block_scale * tensor_scale * 6 = max_abs (with margin handled above)
                    let target_block_scale = if max_abs == 0.0 {
                        0.0
                    } else {
                        max_abs / (6.0 * tensor_scale_f32)
                    };

                    *scale_byte = quantize_fp8_e4m3(target_block_scale);
                    let actual_block_scale = FP8_E4M3_LUT[*scale_byte as usize];
                    let inv = if actual_block_scale > 0.0 {
                        1.0 / (actual_block_scale * tensor_scale_f32)
                    } else {
                        0.0
                    };

                    for (elem, &val) in block.iter().enumerate() {
                        let nibble = quantize_to_fp4(val * inv);
                        let k_idx = blk * NVFP4_BLOCK_SIZE + elem;
                        let byte_idx = k_idx / 2;
                        if k_idx.is_multiple_of(2) {
                            row_packed[byte_idx] |= nibble;
                        } else {
                            row_packed[byte_idx] |= nibble << 4;
                        }
                    }
                }
                (row_packed, row_scales)
            })
            .collect();

        let mut packed = Vec::with_capacity(n * k_half);
        let mut scales = Vec::with_capacity(n * num_blocks_per_row);
        for (row_packed, row_scales) in row_results {
            packed.extend_from_slice(&row_packed);
            scales.extend_from_slice(&row_scales);
        }

        let blocks = Tensor::from_vec(packed, (n, k_half), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let scales = Tensor::from_vec(scales, (n, num_blocks_per_row), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let tensor_scale = Tensor::new(tensor_scale_f32, device)?;
        let bias = bias.map(|b| b.to_device(device)).transpose()?;

        Ok(Arc::new(Self {
            blocks,
            scales,
            tensor_scale,
            bias,
        }))
    }

    fn tensor_scale_value(&self) -> Result<f32> {
        let val = self.tensor_scale.to_dtype(DType::F32)?.to_vec0::<f32>()?;
        Ok(val)
    }

    /// CPU dequant: reconstruct the full FP32 weight matrix from blocks+scales.
    ///
    /// Returns:
    /// - `[N, K]` BF16 for 2-D blocks `[N, K/2]`.
    /// - `[E, N, K]` BF16 for 3-D blocks `[E, N, K/2]` (V4 Flash MoE expert
    ///   stacks). The expert dim is broadcast across the per-expert dequant
    ///   loop so the per-tensor scale and FP8 block scales apply identically
    ///   to every expert.
    pub fn dequantize_weights(&self) -> Result<Tensor> {
        let blocks_dims = self.blocks.dims();
        let (num_experts, n, k_half) = match blocks_dims.len() {
            2 => (1usize, blocks_dims[0], blocks_dims[1]),
            3 => (blocks_dims[0], blocks_dims[1], blocks_dims[2]),
            _ => candle_core::bail!(
                "NVFP4 dequantize_weights expects [N, K/2] or [E, N, K/2] blocks, got {:?}",
                blocks_dims
            ),
        };
        let k = k_half * 2;
        let num_blocks_per_row = k / NVFP4_BLOCK_SIZE;

        let blocks_cpu = self.blocks.to_device(&Device::Cpu)?;
        let scales_cpu = self.scales.to_device(&Device::Cpu)?;
        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<u8> = scales_cpu.flatten_all()?.to_vec1()?;
        let tensor_scale = self.tensor_scale_value()?;

        let mut weights = vec![0f32; num_experts * n * k];
        let half_block = NVFP4_BLOCK_SIZE / 2;

        for expert in 0..num_experts {
            let blocks_expert_base = expert * n * k_half;
            let scales_expert_base = expert * n * num_blocks_per_row;
            let weights_expert_base = expert * n * k;

            for row in 0..n {
                let blocks_row = blocks_expert_base + row * k_half;
                let scales_row = scales_expert_base + row * num_blocks_per_row;
                let weights_row = weights_expert_base + row * k;

                for blk in 0..num_blocks_per_row {
                    let scale_byte = scales_data[scales_row + blk];
                    let block_scale = FP8_E4M3_LUT[scale_byte as usize] * tensor_scale;
                    let blk_bytes_start = blocks_row + blk * half_block;
                    let out_start = weights_row + blk * NVFP4_BLOCK_SIZE;

                    for byte_i in 0..half_block {
                        let packed = blocks_data[blk_bytes_start + byte_i];
                        let lo = dequant_fp4(packed & 0x0F);
                        let hi = dequant_fp4((packed >> 4) & 0x0F);
                        weights[out_start + byte_i * 2] = lo * block_scale;
                        weights[out_start + byte_i * 2 + 1] = hi * block_scale;
                    }
                }
            }
        }

        let shape: Vec<usize> = if blocks_dims.len() == 3 {
            vec![num_experts, n, k]
        } else {
            vec![n, k]
        };
        Tensor::from_vec(weights, shape.as_slice(), &Device::Cpu)?
            .to_device(self.blocks.device())?
            .to_dtype(DType::BF16)
    }

    /// CPU forward: blocked dequant + matmul, avoiding full-weight allocation.
    fn forward_dequantize(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims().to_vec();

        let x_2d = if orig_dims.len() > 2 {
            let features = orig_dims[orig_dims.len() - 1];
            let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
            x.reshape((batch_size, features))?
        } else {
            x.clone()
        };

        let x_f32 = x_2d.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let (m, k) = x_f32.dims2()?;

        let blocks_dims = self.blocks.dims();
        let n = blocks_dims[0];
        let k_half = k / 2;
        let num_blocks_per_row = k / NVFP4_BLOCK_SIZE;
        let half_block = NVFP4_BLOCK_SIZE / 2;

        let blocks_cpu = self.blocks.to_device(&Device::Cpu)?;
        let scales_cpu = self.scales.to_device(&Device::Cpu)?;
        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<u8> = scales_cpu.flatten_all()?.to_vec1()?;
        let x_data: Vec<f32> = x_f32.flatten_all()?.to_vec1()?;
        let tensor_scale = self.tensor_scale_value()?;

        let mut output = vec![0f32; m * n];

        // Block-major loop: dequantize one (row, block) at a time, accumulate dot products.
        for blk in 0..num_blocks_per_row {
            let col_start = blk * NVFP4_BLOCK_SIZE;

            for row in 0..n {
                let scale_byte = scales_data[row * num_blocks_per_row + blk];
                let block_scale = FP8_E4M3_LUT[scale_byte as usize] * tensor_scale;
                let blk_bytes_start = row * k_half + blk * half_block;

                let mut w_block = [0f32; NVFP4_BLOCK_SIZE];
                for byte_i in 0..half_block {
                    let packed = blocks_data[blk_bytes_start + byte_i];
                    w_block[byte_i * 2] = dequant_fp4(packed & 0x0F) * block_scale;
                    w_block[byte_i * 2 + 1] = dequant_fp4((packed >> 4) & 0x0F) * block_scale;
                }

                for token in 0..m {
                    let x_row = &x_data[token * k + col_start..];
                    let mut acc = 0f32;
                    for i in 0..NVFP4_BLOCK_SIZE {
                        acc += x_row[i] * w_block[i];
                    }
                    output[token * n + row] += acc;
                }
            }
        }

        let mut result = Tensor::from_vec(output, (m, n), &Device::Cpu)?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;

        if let Some(bias) = &self.bias {
            result = result.broadcast_add(bias)?;
        }

        if orig_dims.len() > 2 {
            let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
            new_dims.push(result.dim(1)?);
            result = result.reshape(new_dims)?;
        }

        Ok(result)
    }
}

impl QuantMethod for NVFP4Layer {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. }
            | QuantMethodConfig::Qtip { .. }
            | QuantMethodConfig::TuckerFactored { .. } => {
                candle_core::bail!("NVFP4Layer requires QuantMethodConfig::NVFP4")
            }
            QuantMethodConfig::NVFP4 {
                blocks,
                scales,
                tensor_scale,
                bias,
            } => Ok(Self {
                blocks,
                scales,
                tensor_scale,
                bias,
            }),
        }
    }

    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        self.dequantize_weights()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Tier A: CPU dequant + matmul. Tier B (RUN-136) will add the SM100 cublasLt
        // FP4 path once B200 hardware is rented.
        self.forward_dequantize(x)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("NVFP4Layer does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (DType::BF16, self.scales.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        // Dequantize FP4 -> BF16 once; all ISQ targets re-quantize from this.
        // The result is 2-D `[N, K]` for a standard linear weight, or 3-D
        // `[E, N, K]` for a V4 Flash-style MoE expert stack. Only ISQ targets
        // that know about 3-D (currently: QTIP via `quantize_with_options_3d`,
        // and the trivial NVFP4 pass-through) accept the 3-D shape; the rest
        // bail with a clear error so callers know to stick with QTIP or load
        // the stack as per-expert 2-D weights instead.
        let weight = self.dequantize_weights()?;
        let weight_is_3d = weight.dims().len() == 3;
        if weight_is_3d
            && !matches!(
                dtype,
                Some(IsqType::QtipBitshift2) | Some(IsqType::NVFP4) | None
            )
        {
            candle_core::bail!(
                "NVFP4Layer::apply_isq: re-quantizing a 3-D stacked-expert weight \
                 (shape {:?}) is only supported for QTIP, NVFP4 (no-op), or \
                 unquantized passthrough. Got {:?}.",
                weight.dims(),
                dtype
            );
        }
        match dtype {
            Some(IsqType::HQQ4 | IsqType::HQQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("HQQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::HQQ8 => HqqBits::Eight,
                    IsqType::HQQ4 => HqqBits::Four,
                    _ => unreachable!(),
                };
                let cfg = HqqConfig {
                    bits,
                    group_size: ISQ_HQQ_GROUP_SIZE.try_into()?,
                    axis: HqqAxis::Zero,
                    optimization_steps: ISQ_HQQ_DEFAULT_OPT_STEPS,
                    round_zeros: false,
                    channel_wise: true,
                };
                let res = HqqLayer::quantize(&weight.to_device(&device)?, &device, cfg)?;
                if let Some(bias) = &self.bias {
                    let bias = bias
                        .to_device(&device)?
                        .to_dtype(res.dtype_and_device().0)?;
                    Ok(Arc::new(res.with_bias(bias)))
                } else {
                    Ok(Arc::new(res))
                }
            }
            Some(IsqType::AFQ2 | IsqType::AFQ3 | IsqType::AFQ4 | IsqType::AFQ6 | IsqType::AFQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("AFQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::AFQ8 => AfqBits::Eight,
                    IsqType::AFQ6 => AfqBits::Six,
                    IsqType::AFQ4 => AfqBits::Four,
                    IsqType::AFQ3 => AfqBits::Three,
                    IsqType::AFQ2 => AfqBits::Two,
                    _ => unreachable!(),
                };

                Ok(Arc::new(AfqLayer::new(QuantMethodConfig::Afq {
                    weight: weight.to_device(&device)?,
                    bias: self.bias.as_ref().map(|b| b.to_device(&device).unwrap()),
                    bits,
                    group_size: AfqGroupSize::default(),
                })?))
            }
            Some(
                IsqType::Q2K
                | IsqType::Q3K
                | IsqType::Q4K
                | IsqType::Q4_0
                | IsqType::Q4_1
                | IsqType::Q5K
                | IsqType::Q5_0
                | IsqType::Q5_1
                | IsqType::Q6K
                | IsqType::Q8K
                | IsqType::Q8_0
                | IsqType::Q8_1,
            ) => {
                let dtype: GgmlDType = dtype.unwrap().try_into()?;
                let res = if let Some(imatrix_weight) = imatrix_weight {
                    generate_isq_imatrix!(weight, imatrix_weight, device, dtype, n_quantized, guard)
                } else {
                    generate_isq!(weight, device, dtype, n_quantized, guard)
                };
                Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: res,
                    b: self
                        .bias
                        .as_ref()
                        .map(|b| b.to_dtype(DType::F32).unwrap().to_device(&device).unwrap()),
                })?))
            }
            Some(IsqType::F8E4M3) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("F8E4M3 does not support imatrix.");
                }

                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(FP8Linear::new(QuantMethodConfig::FP8 {
                    lin: Linear::new(w, b),
                    dtype: DType::F8E4M3,
                })?))
            }
            Some(IsqType::F8Q8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("F8Q8 does not support imatrix.");
                }

                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(crate::F8Q8Linear::from_weight(&w, b)?))
            }
            Some(IsqType::MXFP4) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("MXFP4 does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let w = weight.to_device(&device)?;
                let b = self
                    .bias
                    .as_ref()
                    .map(|b| b.to_device(&device))
                    .transpose()?;
                crate::MXFP4Layer::quantize(&w, b, &device)
            }
            Some(IsqType::NVFP4) => {
                // Re-quantizing FP4 -> FP4 is a no-op for residency; just pass through.
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("NVFP4 does not support imatrix.");
                }
                Ok(self)
            }
            Some(IsqType::QtipBitshift2) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("QTIP does not support imatrix.");
                }
                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // 3-D stacked-expert NVFP4 layers don't carry bias for V4
                // Flash MoE experts; the 3-D QTIP quantize path also bails
                // if a bias is passed. The branch below preserves the 2-D
                // bias for normal linears and drops it explicitly for the
                // 3-D dispatch.
                let bias = if weight_is_3d {
                    if self.bias.is_some() {
                        candle_core::bail!(
                            "NVFP4 -> QTIP for a 3-D stacked-expert weight: bias \
                             not supported (V4 Flash MoE experts are bias-free; \
                             a present bias here would be silently dropped)"
                        );
                    }
                    None
                } else {
                    self.bias
                        .as_ref()
                        .map(|b| b.to_device(&device))
                        .transpose()?
                };
                crate::QtipLayer::quantize_with_mode(
                    &weight.to_device(&device)?,
                    bias,
                    &device,
                    crate::QtipMode::Viterbi,
                )
            }
            None => {
                let _acquired_quantize_guard = guard.acquire(&device);
                // Ignore imatrix altogether
                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(UnquantLinear::new(
                    QuantMethodConfig::Unquantized(Linear::new(w, b)),
                )?))
            }
        }
    }
}

impl NVFP4Layer {
    /// Test-only constructor: build a 3-D stacked NVFP4 layer from a 3-D
    /// `[E, N, K]` weight by quantizing each expert's `[N, K]` slice via
    /// `NVFP4Layer::quantize` and stacking the resulting blocks/scales. Used
    /// by the NVFP4 -> QTIP 3-D round-trip test to simulate what a stacked
    /// checkpoint loader would produce.
    ///
    /// Available `pub(crate)` because the QTIP tests live in
    /// `crate::qtip::tests` and need to construct a 3-D NVFP4 fixture
    /// without poking at private fields.
    #[cfg(test)]
    pub(crate) fn quantize_3d_for_test(weight: &Tensor, device: &Device) -> Result<Self> {
        let dims = weight.dims3()?;
        let (e, _n, _k) = (dims.0, dims.1, dims.2);
        let mut blocks_e: Vec<Tensor> = Vec::with_capacity(e);
        let mut scales_e: Vec<Tensor> = Vec::with_capacity(e);
        let mut shared_tensor_scale: Option<Tensor> = None;
        for expert_idx in 0..e {
            let w_e = weight.narrow(0, expert_idx, 1)?.squeeze(0)?.contiguous()?;
            let (blk, scl, ts) = Self::quantize_2d_typed(&w_e, device)?;
            blocks_e.push(blk);
            scales_e.push(scl);
            if expert_idx == 0 {
                shared_tensor_scale = Some(ts);
            }
        }
        Ok(Self {
            blocks: Tensor::stack(&blocks_e, 0)?,
            scales: Tensor::stack(&scales_e, 0)?,
            tensor_scale: shared_tensor_scale.unwrap(),
            bias: None,
        })
    }

    /// Typed 2-D quantize: returns `(blocks, scales, tensor_scale)` directly.
    /// Mirrors the body of `NVFP4Layer::quantize` but returns the storage
    /// tensors instead of wrapping in `Arc<dyn QuantMethod>`. Test-only
    /// because production callers always want the wrapped trait object.
    #[cfg(test)]
    fn quantize_2d_typed(weight: &Tensor, device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let weight_f32 = weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let dims = weight_f32.dims2()?;
        let (n, k) = (dims.0, dims.1);
        if k % NVFP4_BLOCK_SIZE != 0 {
            candle_core::bail!(
                "NVFP4 quantization requires K ({k}) divisible by block size ({NVFP4_BLOCK_SIZE})"
            );
        }
        let weight_data: Vec<f32> = weight_f32.flatten_all()?.to_vec1()?;
        let global_max = weight_data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let tensor_scale_f32 = if global_max == 0.0 {
            1.0
        } else {
            (global_max / (6.0 * 448.0 * 0.95)).max(1.0e-12)
        };
        let num_blocks_per_row = k / NVFP4_BLOCK_SIZE;
        let k_half = k / 2;

        use rayon::prelude::*;
        let row_results: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
            .into_par_iter()
            .map(|row| {
                let row_offset = row * k;
                let mut row_packed = vec![0u8; k_half];
                let mut row_scales = vec![0u8; num_blocks_per_row];
                for (blk, scale_byte) in row_scales.iter_mut().enumerate() {
                    let blk_start = row_offset + blk * NVFP4_BLOCK_SIZE;
                    let block = &weight_data[blk_start..blk_start + NVFP4_BLOCK_SIZE];
                    let max_abs = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let target_block_scale = if max_abs == 0.0 {
                        0.0
                    } else {
                        max_abs / (6.0 * tensor_scale_f32)
                    };
                    *scale_byte = quantize_fp8_e4m3(target_block_scale);
                    let actual_block_scale = FP8_E4M3_LUT[*scale_byte as usize];
                    let inv = if actual_block_scale > 0.0 {
                        1.0 / (actual_block_scale * tensor_scale_f32)
                    } else {
                        0.0
                    };
                    for (elem, &val) in block.iter().enumerate() {
                        let nibble = quantize_to_fp4(val * inv);
                        let k_idx = blk * NVFP4_BLOCK_SIZE + elem;
                        let byte_idx = k_idx / 2;
                        if k_idx.is_multiple_of(2) {
                            row_packed[byte_idx] |= nibble;
                        } else {
                            row_packed[byte_idx] |= nibble << 4;
                        }
                    }
                }
                (row_packed, row_scales)
            })
            .collect();

        let mut packed = Vec::with_capacity(n * k_half);
        let mut scales = Vec::with_capacity(n * num_blocks_per_row);
        for (row_packed, row_scales) in row_results {
            packed.extend_from_slice(&row_packed);
            scales.extend_from_slice(&row_scales);
        }
        let blocks = Tensor::from_vec(packed, (n, k_half), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let scales = Tensor::from_vec(scales, (n, num_blocks_per_row), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let tensor_scale = Tensor::new(tensor_scale_f32, device)?;
        Ok((blocks, scales, tensor_scale))
    }

    pub fn linear_b(
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let QuantizedConfig::NVFP4 {} = config else {
            candle_core::bail!("Unexpected quantization config for NVFP4 layer.")
        };

        let blocks = vb.get_with_hints_dtype(
            (out_dim, in_dim / 2),
            "blocks",
            Default::default(),
            DType::U8,
        )?;
        let scales = vb.get_with_hints_dtype(
            (out_dim, in_dim / NVFP4_BLOCK_SIZE),
            "scales",
            Default::default(),
            DType::U8,
        )?;
        // Tensor scale: scalar fp32. If missing in checkpoint, default to 1.0.
        let tensor_scale = vb
            .get_with_hints_dtype((), "tensor_scale", Default::default(), DType::F32)
            .unwrap_or_else(|_| Tensor::new(1.0f32, vb.device()).unwrap());

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            tensor_scale,
            bias,
        }))
    }
}

// UQFF layout for NVFP4Layer:
//   [u32 LE] UQFF version
//   [u8]     QuantizedSerdeType::Nvfp4 (7)
//   [u8]     has_bias
//   tensor: blocks
//   tensor: scales
//   tensor: tensor_scale (scalar fp32)
//   [OPTIONAL] tensor: bias
impl QuantizedSerde for NVFP4Layer {
    fn name(&self) -> &'static str {
        "nvfp4-layer"
    }
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.bias.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        let mut buffer = Vec::new();

        buffer.extend(&UQFF_VERSION.to_le_bytes());
        buffer.push(QuantizedSerdeType::Nvfp4 as u8);
        buffer.push(bias.is_some() as u8);

        serialize_tensor(&mut buffer, &self.blocks)?;
        serialize_tensor(&mut buffer, &self.scales)?;
        serialize_tensor(&mut buffer, &self.tensor_scale)?;

        if let Some(bias) = &bias {
            serialize_tensor(&mut buffer, bias)?;
        }

        Ok(Cow::from(buffer))
    }

    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        let (layer, _bias) = Self::deserialize_ext_bias(data, device, guard)?;
        Ok(layer)
    }

    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data.to_vec());

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Nvfp4 as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Nvfp4 as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let _acquired_load_guard = guard.acquire(device);
        let blocks = deserialize_tensor(&mut buffer, device)?;
        let scales = deserialize_tensor(&mut buffer, device)?;
        let tensor_scale = deserialize_tensor(&mut buffer, device)?;

        let bias = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        let ext_bias = bias.clone();

        Ok((
            Arc::new(Self {
                blocks,
                scales,
                tensor_scale,
                bias,
            }),
            ext_bias,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FP4 round-trip: every representable FP4 value should quantize back to itself.
    #[test]
    fn fp4_quantize_roundtrip_exact_values() {
        let exact_values: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -0.0,
        ];
        for &v in &exact_values {
            let nibble = quantize_to_fp4(v);
            let decoded = dequant_fp4(nibble);
            assert!(
                (decoded - v).abs() < 1e-6 || (v == -0.0 && decoded == 0.0),
                "FP4 round-trip failed for {v}: nibble={nibble:#x}, decoded={decoded}"
            );
        }
    }

    /// FP4 nearest-rounding: values between representable points round correctly.
    #[test]
    fn fp4_quantize_rounding() {
        // Mid-range cases
        assert_eq!(dequant_fp4(quantize_to_fp4(0.24)), 0.0);
        assert_eq!(dequant_fp4(quantize_to_fp4(0.26)), 0.5);
        assert_eq!(dequant_fp4(quantize_to_fp4(0.74)), 0.5);
        assert_eq!(dequant_fp4(quantize_to_fp4(0.76)), 1.0);
        assert_eq!(dequant_fp4(quantize_to_fp4(7.0)), 6.0);
        assert_eq!(dequant_fp4(quantize_to_fp4(-2.5)), -3.0);
    }

    /// FP8 E4M3 LUT: spot-check known values from the OCP spec.
    #[test]
    fn fp8_e4m3_lut_known_values() {
        // 0x00 = +0.0
        assert_eq!(FP8_E4M3_LUT[0x00], 0.0);
        // 0x80 = -0.0 (or 0.0 in our table)
        assert!(FP8_E4M3_LUT[0x80].abs() < 1e-30);
        // 0x38 = +1.0 (exp=7, mant=0, bias=7 → 2^0 * 1 = 1.0)
        assert!((FP8_E4M3_LUT[0x38] - 1.0).abs() < 1e-6);
        // 0x40 = +2.0 (exp=8, bias=7 → 2^1 = 2.0)
        assert!((FP8_E4M3_LUT[0x40] - 2.0).abs() < 1e-6);
        // 0xB8 = -1.0
        assert!((FP8_E4M3_LUT[0xB8] - (-1.0)).abs() < 1e-6);
        // Maximum: 0x7E = 1.75 * 2^8 = 448.0
        assert!((FP8_E4M3_LUT[0x7E] - 448.0).abs() < 1e-3);
    }

    /// FP8 E4M3 quantize round-trip: representable values map to themselves.
    #[test]
    fn fp8_e4m3_quantize_roundtrip() {
        for &target in &[1.0f32, 2.0, 0.5, 16.0, -1.0, -8.0, 0.25] {
            let byte = quantize_fp8_e4m3(target);
            let decoded = FP8_E4M3_LUT[byte as usize];
            assert!(
                (decoded - target).abs() < target.abs() * 0.1 + 1e-6,
                "FP8 E4M3 round-trip failed: target={target}, byte={byte:#x}, decoded={decoded}"
            );
        }
    }

    /// End-to-end: quantize a random [4, 32] tensor to NVFP4 and dequantize.
    /// Reconstruction error per element should be bounded.
    #[test]
    fn nvfp4_quantize_dequantize_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        // Deterministic test tensor
        let mut data = vec![0.0f32; 4 * 32];
        for (i, v) in data.iter_mut().enumerate() {
            // Mix of magnitudes to stress per-block scaling
            *v = ((i as f32) * 0.137).sin() * (1.0 + (i / 32) as f32 * 0.5);
        }
        let weight = Tensor::from_vec(data.clone(), (4, 32), &device)?;

        let layer = NVFP4Layer::quantize(&weight, None, &device)?;
        let dequant = layer.dequantize_w()?;
        let dequant_f32 = dequant
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Check that reconstruction is within reasonable bounds.
        // FP4 has 8 magnitudes per sign × per-block FP8 scale = ~5-7% per-element error
        // is acceptable. We measure relative L2 error.
        let mut sum_err = 0.0f32;
        let mut sum_orig = 0.0f32;
        for (orig, recon) in data.iter().zip(dequant_f32.iter()) {
            sum_err += (orig - recon).powi(2);
            sum_orig += orig.powi(2);
        }
        let rel_err = (sum_err / sum_orig).sqrt();
        assert!(
            rel_err < 0.15,
            "NVFP4 reconstruction relative L2 error {rel_err} exceeds 15%"
        );
        Ok(())
    }

    /// End-to-end matmul: NVFP4 forward must approximate the dense FP32 matmul.
    #[test]
    fn nvfp4_matmul_matches_dense_within_tolerance() -> Result<()> {
        let device = Device::Cpu;
        // weight: [N=8, K=32]
        let mut wdata = vec![0.0f32; 8 * 32];
        for (i, v) in wdata.iter_mut().enumerate() {
            *v = ((i as f32) * 0.31).cos() * 2.0;
        }
        let weight = Tensor::from_vec(wdata.clone(), (8, 32), &device)?;

        // input: [M=2, K=32]
        let xdata: Vec<f32> = (0..(2 * 32)).map(|i| ((i as f32) * 0.07).sin()).collect();
        let x = Tensor::from_vec(xdata.clone(), (2, 32), &device)?.to_dtype(DType::F32)?;

        // Reference: dense matmul
        let dense = x.matmul(&weight.t()?)?;
        let dense_v: Vec<f32> = dense.flatten_all()?.to_vec1()?;

        // NVFP4 forward
        let layer = NVFP4Layer::quantize(&weight, None, &device)?;
        let nvfp4_out = layer.forward(&x)?;
        let nvfp4_v: Vec<f32> = nvfp4_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        // Cosine similarity check — should exceed 0.99
        let mut dot = 0.0f32;
        let mut nd = 0.0f32;
        let mut nn = 0.0f32;
        for (d, n) in dense_v.iter().zip(nvfp4_v.iter()) {
            dot += d * n;
            nd += d * d;
            nn += n * n;
        }
        let cos = dot / (nd.sqrt() * nn.sqrt());
        assert!(
            cos > 0.99,
            "NVFP4 matmul cosine similarity {cos} below 0.99"
        );
        Ok(())
    }
}
