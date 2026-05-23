//! GPU dispatch for QTIP forward.
//!
//! Provides two CUDA helpers:
//! * `dequantize_rotated_cuda` — decode the packed trellis into a BF16/F16/F32
//!   weight matrix on the GPU (in the *rotated* frame, since that's what the
//!   stored symbols represent when the layer was quantized with rotation).
//! * `rotate_x_cuda` — apply the block-diagonal D·H·D rotation to a batch of
//!   activation rows on the GPU, returning a new tensor.
//!
//! Both are gated on `has_qtip_kernels`; the caller falls back to CPU when
//! the compile-time gate is off or any precondition fails.
//!
//! Architecture: the rotation has the invariance
//!   (x · R) @ (W · R)^T = x @ R · R^T · W^T = x @ I · W^T = x @ W^T
//! (R orthogonal AND involutory), so the forward path:
//!   1. dequantize W in the rotated frame (no inverse-rotation pass)
//!   2. rotate x in-place by R = block_diag(D · H · D)
//!   3. y = x_rot @ W_rotated^T
//! produces the algebraically-correct y, with each per-step cost paid on
//! the GPU exactly once.

#![cfg(feature = "cuda")]

use candle_core::{CudaStorage, DType, Result, Storage, Tensor};
use half::{bf16, f16};

use crate::utils::slice_ptr;

use super::ffi;

/// Returns true when the layer parameters live on CUDA *and* the kernels
/// were compiled in.
pub(crate) fn can_use_qtip_cuda(blocks: &Tensor) -> bool {
    if !ffi::HAVE_QTIP_KERNELS {
        return false;
    }
    matches!(blocks.device(), candle_core::Device::Cuda(_))
}

/// Dequantize the QTIP-packed weights into a `[n_rows, in_features]`
/// matrix on the GPU. Output lives in the *rotated* frame.
pub(crate) fn dequantize_rotated_cuda(
    blocks: &Tensor,
    row_scales: &Tensor,
    lut: &Tensor,
    in_features: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let n_rows = row_scales.dim(0)?;
    let packed_per_row = blocks.dim(1)?;
    let num_symbols = in_features / super::V as usize;
    if blocks.dtype() != DType::U8 {
        candle_core::bail!("QTIP dequantize CUDA: blocks dtype must be U8");
    }
    if row_scales.dtype() != DType::F32 {
        candle_core::bail!("QTIP dequantize CUDA: row_scales dtype must be F32");
    }
    if lut.dtype() != DType::F32 {
        candle_core::bail!("QTIP dequantize CUDA: lut dtype must be F32");
    }
    if !blocks.layout().is_contiguous()
        || !row_scales.layout().is_contiguous()
        || !lut.layout().is_contiguous()
    {
        candle_core::bail!("QTIP dequantize CUDA: all inputs must be contiguous");
    }

    let dev = match blocks.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("QTIP dequantize CUDA: blocks must live on CUDA"),
    };

    let (blocks_storage, blocks_layout) = blocks.storage_and_layout();
    let blocks_storage = match &*blocks_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP dequantize CUDA: blocks must be CUDA storage"),
    };
    let (scales_storage, scales_layout) = row_scales.storage_and_layout();
    let scales_storage = match &*scales_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP dequantize CUDA: scales must be CUDA storage"),
    };
    let (lut_storage, lut_layout) = lut.storage_and_layout();
    let lut_storage = match &*lut_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP dequantize CUDA: lut must be CUDA storage"),
    };

    let (blocks_ptr, _blocks_guard) =
        slice_ptr(blocks_storage.as_cuda_slice::<u8>()?, blocks_layout.start_offset());
    let (scales_ptr, _scales_guard) =
        slice_ptr(scales_storage.as_cuda_slice::<f32>()?, scales_layout.start_offset());
    let (lut_ptr, _lut_guard) =
        slice_ptr(lut_storage.as_cuda_slice::<f32>()?, lut_layout.start_offset());

    let num_weights = n_rows * in_features;
    let out_shape = candle_core::Shape::from_dims(&[n_rows, in_features]);

    let res = match out_dtype {
        DType::BF16 => {
            let out_buf = dev.alloc_zeros::<bf16>(num_weights)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                ffi::launch_qtip_dequantize_v2_k4_l16_bf16(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    num_symbols as i32,
                    dev.cuda_stream().cu_stream(),
                );
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }
        DType::F16 => {
            let out_buf = dev.alloc_zeros::<f16>(num_weights)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                ffi::launch_qtip_dequantize_v2_k4_l16_f16(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    num_symbols as i32,
                    dev.cuda_stream().cu_stream(),
                );
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }
        DType::F32 => {
            let out_buf = dev.alloc_zeros::<f32>(num_weights)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                ffi::launch_qtip_dequantize_v2_k4_l16_f32(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    num_symbols as i32,
                    dev.cuda_stream().cu_stream(),
                );
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }
        other => candle_core::bail!("QTIP dequantize CUDA: unsupported out dtype {other:?}"),
    };

    Ok(Tensor::from((Storage::Cuda(res), out_shape)))
}

/// Apply the block-diagonal D·H·D rotation to each row of `x` on the GPU,
/// returning a fresh tensor with the rotated values. `block_size` must be
/// one of `{2, 4, 8, 16, 32, 64, 128}`.
pub(crate) fn rotate_x_cuda(x: &Tensor, signs: &Tensor, block_size: usize) -> Result<Tensor> {
    if !ffi::HAVE_QTIP_KERNELS {
        candle_core::bail!("QTIP rotate-x CUDA: kernels not compiled in");
    }
    if !x.layout().is_contiguous() || !signs.layout().is_contiguous() {
        candle_core::bail!("QTIP rotate-x CUDA: x and signs must be contiguous");
    }
    let (batch, feat) = x.dims2()?;
    if signs.dim(0)? != feat {
        candle_core::bail!(
            "QTIP rotate-x CUDA: signs len {} != in_features {}",
            signs.dim(0)?,
            feat
        );
    }
    if signs.dtype() != DType::F32 {
        candle_core::bail!("QTIP rotate-x CUDA: signs dtype must be F32");
    }
    if !matches!(block_size, 2 | 4 | 8 | 16 | 32 | 64 | 128) {
        candle_core::bail!("QTIP rotate-x CUDA: unsupported block size {block_size}");
    }
    if feat % block_size != 0 {
        candle_core::bail!(
            "QTIP rotate-x CUDA: in_features {feat} not divisible by block {block_size}"
        );
    }

    let dev = match x.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("QTIP rotate-x CUDA: x must live on CUDA"),
    };

    // We need a writable destination buffer. Strategy: contiguous-clone `x`
    // (which produces a fresh CUDA buffer via device-to-device memcpy
    // inside candle), then launch the in-place kernel against that
    // buffer's raw pointer. The clone takes us off the original storage so
    // we are free to mutate.
    let x_contig = x.contiguous()?;
    let num_elements = batch * feat;
    let out_shape = candle_core::Shape::from_dims(&[batch, feat]);

    let (signs_storage, signs_layout) = signs.storage_and_layout();
    let signs_storage = match &*signs_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP rotate-x CUDA: signs must be CUDA storage"),
    };
    let (signs_ptr, _signs_guard) =
        slice_ptr(signs_storage.as_cuda_slice::<f32>()?, signs_layout.start_offset());

    macro_rules! rotate_dtype {
        ($T:ty, $launch:expr) => {{
            // Allocate a destination buffer and memcpy_dtod from the
            // contiguous source. Then launch the in-place rotation
            // kernel against the destination.
            let src_storage = {
                let (s, _l) = x_contig.storage_and_layout();
                match &*s {
                    Storage::Cuda(c) => c.as_cuda_slice::<$T>()?.clone(),
                    _ => candle_core::bail!("QTIP rotate-x CUDA: source must be CUDA"),
                }
            };
            // `clone()` on a CudaSlice in cudarc is a shallow alias, not a
            // memcpy. We need a fresh allocation so the rotation doesn't
            // mutate the original tensor's storage in-place.
            let mut dst = unsafe { dev.alloc::<$T>(num_elements) }?;
            dev.memcpy_dtod(&src_storage, &mut dst)?;
            let (dst_ptr, dst_guard) = slice_ptr(&dst, 0);
            unsafe {
                $launch(
                    dst_ptr as *mut _,
                    signs_ptr as *const _,
                    batch as i32,
                    feat as i32,
                    block_size as i32,
                    dev.cuda_stream().cu_stream(),
                );
            }
            drop(dst_guard);
            CudaStorage::wrap_cuda_slice(dst, dev.clone())
        }};
    }

    let res = match x.dtype() {
        DType::BF16 => rotate_dtype!(bf16, ffi::launch_qtip_rotate_x_bf16),
        DType::F16 => rotate_dtype!(f16, ffi::launch_qtip_rotate_x_f16),
        DType::F32 => rotate_dtype!(f32, ffi::launch_qtip_rotate_x_f32),
        other => candle_core::bail!("QTIP rotate-x CUDA: unsupported dtype {other:?}"),
    };
    Ok(Tensor::from((Storage::Cuda(res), out_shape)))
}
