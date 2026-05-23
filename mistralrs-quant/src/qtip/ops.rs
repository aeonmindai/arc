//! QTIP dequantize as a `CustomOp3` — three input tensors (packed
//! blocks, per-row scales, shared LUT) → one BF16 weight matrix.
//!
//! The CPU path mirrors the trellis decode in [`super::QtipLayer::dequantize_weights_rotated_f32`]
//! but reads directly from `CpuStorage` slices (no extra allocation /
//! to_device round-trip). The CUDA path dispatches to the
//! `qtip_dequantize_v2_k4_l16_bf16` kernel.
//!
//! Rotation is intentionally NOT applied here — the caller is
//! responsible for only routing rotation-disabled checkpoints through
//! this op. Rotation-enabled (Viterbi) checkpoints continue to use the
//! existing CPU `dequantize_weights` path until the GPU rotation
//! kernel lands.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Result, Shape, Tensor};

use super::{K, V, STATE_MASK};

pub(crate) struct QtipDequantize {
    pub(crate) n_rows: usize,
    pub(crate) k_in: usize,
}

impl QtipDequantize {
    fn decode_cpu_bf16(
        &self,
        blocks: &[u8],
        row_scales: &[f32],
        lut: &[f32],
    ) -> Vec<half::bf16> {
        let num_symbols = self.k_in / V as usize;
        let packed_per_row = num_symbols / 2;
        let mut out = vec![half::bf16::ZERO; self.n_rows * self.k_in];

        for row in 0..self.n_rows {
            let scale = row_scales[row];
            let row_packed_base = row * packed_per_row;
            let out_row_base = row * self.k_in;
            let mut state: u32 = 0;

            for sym_idx in 0..num_symbols {
                let byte = blocks[row_packed_base + sym_idx / 2];
                let sym = if sym_idx.is_multiple_of(2) {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                state = ((state << K) | sym as u32) & STATE_MASK;
                let lut_off = (state as usize) * V as usize;
                out[out_row_base + sym_idx * 2] =
                    half::bf16::from_f32(lut[lut_off] * scale);
                out[out_row_base + sym_idx * 2 + 1] =
                    half::bf16::from_f32(lut[lut_off + 1] * scale);
            }
        }
        out
    }
}

impl CustomOp3 for QtipDequantize {
    fn name(&self) -> &'static str {
        "qtip-dequantize-v2-k4-l16"
    }

    fn cpu_fwd(
        &self,
        blocks_s: &CpuStorage,
        blocks_l: &Layout,
        scales_s: &CpuStorage,
        scales_l: &Layout,
        lut_s: &CpuStorage,
        lut_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if !(blocks_l.is_contiguous() && scales_l.is_contiguous() && lut_l.is_contiguous()) {
            candle_core::bail!("qtip dequantize: all inputs must be contiguous");
        }
        let CpuStorage::U8(blocks) = blocks_s else {
            candle_core::bail!("qtip dequantize: blocks must be U8");
        };
        let CpuStorage::F32(scales) = scales_s else {
            candle_core::bail!("qtip dequantize: row_scales must be F32");
        };
        let CpuStorage::F32(lut) = lut_s else {
            candle_core::bail!("qtip dequantize: lut must be F32");
        };

        let out = self.decode_cpu_bf16(blocks, scales, lut);
        Ok((CpuStorage::BF16(out), Shape::from_dims(&[self.n_rows, self.k_in])))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        blocks_s: &candle_core::CudaStorage,
        blocks_l: &Layout,
        scales_s: &candle_core::CudaStorage,
        scales_l: &Layout,
        lut_s: &candle_core::CudaStorage,
        lut_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use crate::{qtip::ffi, utils::slice_ptr};
        use candle_core::CudaStorage;

        if !ffi::HAVE_QTIP_KERNELS {
            candle_core::bail!("QTIP CUDA kernel not compiled");
        }
        if !(blocks_l.is_contiguous() && scales_l.is_contiguous() && lut_l.is_contiguous()) {
            candle_core::bail!("qtip dequantize: all inputs must be contiguous");
        }

        let dev = blocks_s.device();
        let num_symbols = self.k_in / V as usize;
        let packed_per_row = (num_symbols / 2) as i32;
        let num_symbols_i32 = num_symbols as i32;
        let n_rows_i32 = self.n_rows as i32;

        let (blocks_ptr, _blocks_guard) =
            slice_ptr(blocks_s.as_cuda_slice::<u8>()?, blocks_l.start_offset());
        let (scales_ptr, _scales_guard) =
            slice_ptr(scales_s.as_cuda_slice::<f32>()?, scales_l.start_offset());
        let (lut_ptr, _lut_guard) =
            slice_ptr(lut_s.as_cuda_slice::<f32>()?, lut_l.start_offset());

        let output = dev.alloc_zeros::<half::bf16>(self.n_rows * self.k_in)?;
        let (output_ptr, output_guard) = slice_ptr(&output, 0);

        unsafe {
            ffi::launch_qtip_dequantize_v2_k4_l16_bf16(
                blocks_ptr as *const u8,
                scales_ptr as *const f32,
                lut_ptr as *const f32,
                output_ptr as *mut half::bf16,
                n_rows_i32,
                packed_per_row,
                num_symbols_i32,
                dev.cuda_stream().cu_stream(),
            );
        }
        drop(output_guard);

        let storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
        Ok((storage, Shape::from_dims(&[self.n_rows, self.k_in])))
    }
}

/// Dequantize QTIP-packed weights to BF16 via `CustomOp3`. Three input
/// tensors: blocks (U8 packed nibbles), row_scales (F32), shared LUT
/// (F32). Output: BF16 weights of shape `[n_rows, k_in]`.
///
/// Caller must ensure rotation is disabled — this op does NOT apply
/// the D·H·D unrotation that rotated-frame checkpoints require.
pub fn qtip_dequantize_bf16(
    blocks: &Tensor,
    row_scales: &Tensor,
    lut: &Tensor,
    n_rows: usize,
    k_in: usize,
) -> Result<Tensor> {
    if blocks.dtype() != DType::U8 {
        candle_core::bail!("qtip_dequantize: blocks must be U8, got {:?}", blocks.dtype());
    }
    let scales = if row_scales.dtype() != DType::F32 {
        row_scales.to_dtype(DType::F32)?
    } else {
        row_scales.clone()
    };
    let lut_f32 = if lut.dtype() != DType::F32 {
        lut.to_dtype(DType::F32)?
    } else {
        lut.clone()
    };
    blocks.apply_op3_no_bwd(&scales, &lut_f32, &QtipDequantize { n_rows, k_in })
}
