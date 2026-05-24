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

    let (blocks_ptr, _blocks_guard) = slice_ptr(
        blocks_storage.as_cuda_slice::<u8>()?,
        blocks_layout.start_offset(),
    );
    let (scales_ptr, _scales_guard) = slice_ptr(
        scales_storage.as_cuda_slice::<f32>()?,
        scales_layout.start_offset(),
    );
    let (lut_ptr, _lut_guard) = slice_ptr(
        lut_storage.as_cuda_slice::<f32>()?,
        lut_layout.start_offset(),
    );

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

/// Single-token fused decode + gemv. Replaces the dequantize+matmul forward
/// path for the n_tokens=1 case. Reads the QTIP-packed bytes once, decodes
/// them in registers, accumulates the matvec `y = W @ x_rotated^T`.
///
/// Bandwidth comparison vs `dequantize_rotated_cuda` + `matmul`:
///   * dequant: reads ~0.25 byte/weight (packed) + writes 2 byte/weight (BF16)
///   * matmul:  reads 2 byte/weight (BF16) again + ~0 for x (broadcast)
///   * total:   ~4.25 byte/weight of HBM traffic on the weights
///   * fused:   ~0.25 byte/weight only → roughly 17× less HBM weight traffic.
///
/// Caller MUST pass `x_rotated` (already through `rotate_x_cuda`). The kernel
/// does no rotation. `x_rotated` is a `[1, k_in]` 1-token batch — we slice the
/// row, so any contiguous `[..., k_in]` shape is fine as long as the layout
/// gives a contiguous row.
///
/// Returns a `[1, n_rows]` tensor (one logit per output row, single token).
pub(crate) fn fused_gemv_cuda(
    blocks: &Tensor,
    row_scales: &Tensor,
    lut: &Tensor,
    x_rotated: &Tensor,
    in_features: usize,
) -> Result<Tensor> {
    let n_rows = row_scales.dim(0)?;
    let packed_per_row = blocks.dim(1)?;
    let num_symbols = in_features / super::V as usize;

    if blocks.dtype() != DType::U8 {
        candle_core::bail!("QTIP fused gemv CUDA: blocks dtype must be U8");
    }
    if row_scales.dtype() != DType::F32 {
        candle_core::bail!("QTIP fused gemv CUDA: row_scales dtype must be F32");
    }
    if lut.dtype() != DType::F32 {
        candle_core::bail!("QTIP fused gemv CUDA: lut dtype must be F32");
    }
    if !blocks.layout().is_contiguous()
        || !row_scales.layout().is_contiguous()
        || !lut.layout().is_contiguous()
    {
        candle_core::bail!("QTIP fused gemv CUDA: blocks/scales/lut must be contiguous");
    }

    // x_rotated must be a single-row contiguous tensor with `in_features` cols.
    let x_2d = match x_rotated.dims() {
        [k] if *k == in_features => x_rotated.unsqueeze(0)?,
        [b, k] if *b == 1 && *k == in_features => x_rotated.clone(),
        other => candle_core::bail!(
            "QTIP fused gemv CUDA: x_rotated must be [k_in] or [1, k_in]; got {other:?} (k_in={in_features})"
        ),
    };
    let x_2d = x_2d.contiguous()?;

    let dev = match blocks.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("QTIP fused gemv CUDA: blocks must live on CUDA"),
    };
    if !matches!(x_2d.device(), candle_core::Device::Cuda(_)) {
        candle_core::bail!("QTIP fused gemv CUDA: x_rotated must live on CUDA");
    }

    let out_shape = candle_core::Shape::from_dims(&[1, n_rows]);

    let (blocks_storage, blocks_layout) = blocks.storage_and_layout();
    let blocks_storage = match &*blocks_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP fused gemv CUDA: blocks storage must be CUDA"),
    };
    let (scales_storage, scales_layout) = row_scales.storage_and_layout();
    let scales_storage = match &*scales_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP fused gemv CUDA: scales storage must be CUDA"),
    };
    let (lut_storage, lut_layout) = lut.storage_and_layout();
    let lut_storage = match &*lut_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP fused gemv CUDA: lut storage must be CUDA"),
    };
    let (x_storage, x_layout) = x_2d.storage_and_layout();
    let x_storage = match &*x_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP fused gemv CUDA: x storage must be CUDA"),
    };

    let (blocks_ptr, _blocks_guard) =
        slice_ptr(blocks_storage.as_cuda_slice::<u8>()?, blocks_layout.start_offset());
    let (scales_ptr, _scales_guard) =
        slice_ptr(scales_storage.as_cuda_slice::<f32>()?, scales_layout.start_offset());
    let (lut_ptr, _lut_guard) =
        slice_ptr(lut_storage.as_cuda_slice::<f32>()?, lut_layout.start_offset());

    let res = match x_2d.dtype() {
        DType::BF16 => {
            let (x_ptr, _x_guard) =
                slice_ptr(x_storage.as_cuda_slice::<bf16>()?, x_layout.start_offset());
            let out_buf = dev.alloc_zeros::<bf16>(n_rows)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                ffi::launch_qtip_fused_gemv_v2_k4_l16_bf16(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    x_ptr as *const _,
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
            let (x_ptr, _x_guard) =
                slice_ptr(x_storage.as_cuda_slice::<f16>()?, x_layout.start_offset());
            let out_buf = dev.alloc_zeros::<f16>(n_rows)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                ffi::launch_qtip_fused_gemv_v2_k4_l16_f16(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    x_ptr as *const _,
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
            let (x_ptr, _x_guard) =
                slice_ptr(x_storage.as_cuda_slice::<f32>()?, x_layout.start_offset());
            let out_buf = dev.alloc_zeros::<f32>(n_rows)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                ffi::launch_qtip_fused_gemv_v2_k4_l16_f32(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    x_ptr as *const _,
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
        other => candle_core::bail!("QTIP fused gemv CUDA: unsupported x dtype {other:?}"),
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

    // We need a writable destination buffer. Strategy: contiguous-clone the
    // tensor, then CudaSlice::clone (which is a device-to-device memcpy in
    // cudarc 0.19) the storage so we own a fresh buffer. The rotation
    // kernel runs in-place on that owned buffer.
    let x_contig = x.contiguous()?;
    let out_shape = candle_core::Shape::from_dims(&[batch, feat]);

    let (signs_storage, signs_layout) = signs.storage_and_layout();
    let signs_storage = match &*signs_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP rotate-x CUDA: signs must be CUDA storage"),
    };
    let (signs_ptr, _signs_guard) = slice_ptr(
        signs_storage.as_cuda_slice::<f32>()?,
        signs_layout.start_offset(),
    );

    macro_rules! rotate_dtype {
        ($T:ty, $launch:expr) => {{
            // `CudaSlice::clone()` in cudarc 0.19 performs a device-to-device
            // copy (see `try_clone → clone_dtod` impl). So cloning here gives
            // us a fresh GPU buffer with the same contents as `x`, ready to
            // be mutated in place by the rotation kernel.
            let dst: candle_core::cuda::cudarc::driver::CudaSlice<$T> = {
                let (s, _l) = x_contig.storage_and_layout();
                match &*s {
                    Storage::Cuda(c) => c.as_cuda_slice::<$T>()?.clone(),
                    _ => candle_core::bail!("QTIP rotate-x CUDA: source must be CUDA"),
                }
            };
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

// ---------------------------------------------------------------------------
// Quantize-side ops (RUN-quant-on-gpu).
// ---------------------------------------------------------------------------
//
// Goal: eliminate the CPU detour in `QtipLayer::quantize_with_options`. For a
// 284B-parameter model on a 1xH100 box, the CPU-only Viterbi path took 30-90
// minutes; the GPU path is expected to fit in <60 seconds end-to-end.
//
// API: one entry point — `quantize_rows_cuda` — that:
//   1) (Optionally) applies block-diagonal D·H·D rotation in-place on weights.
//   2) Computes per-row scale = max(|w|)/3.0.
//   3) Runs greedy or Viterbi quantize per row, producing packed bytes.
//
// All work happens on CUDA; nothing crosses to the host until the caller
// pulls the resulting tensors. Inputs MUST already be F32 on CUDA; the
// caller is responsible for the dtype convert + device transfer (a single
// kernel launch, no CPU detour). The function returns `(packed_blocks,
// row_scales)`.

use super::QtipMode;

/// Apply block-diagonal D·H·D rotation in-place to each row of `weight`,
/// on GPU. `signs` is `[in_features]` F32 ±1 already on the device.
/// `block_size` must be a power of 2 in [2, 128].
///
/// The kernel mutates `weight` in place; we own the buffer here (we copy
/// the caller's input to a fresh buffer first, then run rotation on the
/// copy) so this is a non-destructive op from the caller's point of view.
pub(crate) fn rotate_weight_rows_cuda(
    weight: &Tensor,
    signs: &Tensor,
    block_size: usize,
) -> Result<Tensor> {
    if !ffi::HAVE_QTIP_KERNELS {
        candle_core::bail!("QTIP rotate-weight CUDA: kernels not compiled in");
    }
    if !matches!(block_size, 2 | 4 | 8 | 16 | 32 | 64 | 128) {
        candle_core::bail!("QTIP rotate-weight CUDA: unsupported block size {block_size}");
    }
    let (n_rows, k_in) = weight.dims2()?;
    if k_in % block_size != 0 {
        candle_core::bail!(
            "QTIP rotate-weight CUDA: in_features {k_in} not divisible by block {block_size}"
        );
    }
    if weight.dtype() != DType::F32 {
        candle_core::bail!(
            "QTIP rotate-weight CUDA: weight dtype must be F32, got {:?}",
            weight.dtype()
        );
    }
    if signs.dtype() != DType::F32 {
        candle_core::bail!("QTIP rotate-weight CUDA: signs dtype must be F32");
    }
    if signs.dim(0)? != k_in {
        candle_core::bail!(
            "QTIP rotate-weight CUDA: signs len {} != in_features {}",
            signs.dim(0)?,
            k_in
        );
    }

    let dev = match weight.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("QTIP rotate-weight CUDA: weight must live on CUDA"),
    };

    // Same fresh-buffer strategy as `rotate_x_cuda`: clone the storage so we
    // can mutate in place without aliasing the caller's tensor.
    let weight_contig = weight.contiguous()?;
    let dst: candle_core::cuda::cudarc::driver::CudaSlice<f32> = {
        let (s, _l) = weight_contig.storage_and_layout();
        match &*s {
            Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.clone(),
            _ => candle_core::bail!("QTIP rotate-weight CUDA: source must be CUDA"),
        }
    };

    let (signs_storage, signs_layout) = signs.storage_and_layout();
    let signs_storage = match &*signs_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP rotate-weight CUDA: signs must be CUDA storage"),
    };
    let (signs_ptr, _signs_guard) = slice_ptr(
        signs_storage.as_cuda_slice::<f32>()?,
        signs_layout.start_offset(),
    );
    let (dst_ptr, dst_guard) = slice_ptr(&dst, 0);
    unsafe {
        ffi::launch_qtip_rotate_weight_rows_f32(
            dst_ptr as *mut _,
            signs_ptr as *const _,
            n_rows as i32,
            k_in as i32,
            block_size as i32,
            dev.cuda_stream().cu_stream(),
        );
    }
    drop(dst_guard);

    let res = CudaStorage::wrap_cuda_slice(dst, dev.clone());
    Ok(Tensor::from((
        Storage::Cuda(res),
        candle_core::Shape::from_dims(&[n_rows, k_in]),
    )))
}

/// Compute per-row scale = max(|row|) / 3.0 (or 1.0 if max=0), returning a
/// fresh F32 tensor of shape `[n_rows]` on the same device as `weight`.
fn compute_row_scales_cuda(weight: &Tensor) -> Result<Tensor> {
    if !ffi::HAVE_QTIP_KERNELS {
        candle_core::bail!("QTIP row-scale CUDA: kernels not compiled in");
    }
    let (n_rows, k_in) = weight.dims2()?;
    if weight.dtype() != DType::F32 {
        candle_core::bail!(
            "QTIP row-scale CUDA: weight dtype must be F32, got {:?}",
            weight.dtype()
        );
    }
    let dev = match weight.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("QTIP row-scale CUDA: weight must live on CUDA"),
    };
    let weight_contig = weight.contiguous()?;

    let (w_storage, w_layout) = weight_contig.storage_and_layout();
    let w_storage = match &*w_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP row-scale CUDA: weight must be CUDA storage"),
    };
    let (w_ptr, _w_guard) = slice_ptr(w_storage.as_cuda_slice::<f32>()?, w_layout.start_offset());

    let scales_buf = dev.alloc_zeros::<f32>(n_rows)?;
    let (scales_ptr, scales_guard) = slice_ptr(&scales_buf, 0);
    unsafe {
        ffi::launch_qtip_compute_row_scales_f32(
            w_ptr as *const _,
            scales_ptr as *mut _,
            n_rows as i32,
            k_in as i32,
            dev.cuda_stream().cu_stream(),
        );
    }
    drop(scales_guard);

    let res = CudaStorage::wrap_cuda_slice(scales_buf, dev.clone());
    Ok(Tensor::from((
        Storage::Cuda(res),
        candle_core::Shape::from_dims(&[n_rows]),
    )))
}

/// Maximum scratch budget (bytes) for the Viterbi backtrace buffer.
/// Backtrace (prefix-grouped) = (rows_in_flight * num_symbols * 2^(L-K))
/// bytes; we cap at this to avoid blowing past available HBM on multi-tenant
/// boxes. ~6 GB leaves plenty of room for the model itself on H100 (80 GB)
/// and even on the smaller A100 80GB / L40 48GB cards. With prefix-grouping,
/// per-row backtrace is 16× smaller than naive per-state, so we fit far more
/// rows in flight per launch.
const VITERBI_MAX_SCRATCH_BYTES: usize = 6 * 1024 * 1024 * 1024;

/// 2^L = 65536, the trellis state count. Matches `LUT_SIZE` in `mod.rs`.
const QTIP_LUT_SIZE: usize = 1 << super::L;

/// 2^(L-K) = 4096, the trellis prefix count. The prefix-grouped Viterbi
/// kernel stores one backtrace entry per (timestep, prefix) — 16× smaller
/// than the naive per-state version because all 16 successors of a given
/// prefix share the same predecessor reduction.
const QTIP_PREFIX_COUNT: usize = 1 << (super::L - super::K);

/// One-shot quantize entry point. Returns `(packed_blocks, row_scales)`:
/// * `packed_blocks` — `[n_rows, num_symbols / 2]` U8, two K=4 symbols per byte.
/// * `row_scales`    — `[n_rows]` F32, per-row scale.
///
/// `weight_rotated_f32` should already be in the rotated frame (caller is
/// responsible for applying `rotate_weight_rows_cuda` first when rotation
/// is enabled). This split lets the test path verify each step on its own.
pub(crate) fn quantize_rows_cuda(
    weight_rotated_f32: &Tensor,
    lut: &Tensor,
    mode: QtipMode,
) -> Result<(Tensor, Tensor)> {
    if !ffi::HAVE_QTIP_KERNELS {
        candle_core::bail!("QTIP quantize CUDA: kernels not compiled in");
    }
    let (n_rows, k_in) = weight_rotated_f32.dims2()?;
    let num_symbols = k_in / super::V as usize;
    if !num_symbols.is_multiple_of(2) {
        candle_core::bail!(
            "QTIP quantize CUDA: num_symbols ({num_symbols}) must be even for K=4 packing"
        );
    }
    if weight_rotated_f32.dtype() != DType::F32 {
        candle_core::bail!(
            "QTIP quantize CUDA: weight dtype must be F32, got {:?}",
            weight_rotated_f32.dtype()
        );
    }
    if lut.dtype() != DType::F32 {
        candle_core::bail!("QTIP quantize CUDA: lut dtype must be F32");
    }
    let dev = match weight_rotated_f32.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("QTIP quantize CUDA: weight must live on CUDA"),
    };

    let weight_contig = weight_rotated_f32.contiguous()?;
    let lut_contig = lut.contiguous()?;

    // Step 1: row scales.
    let row_scales = compute_row_scales_cuda(&weight_contig)?;

    // Step 2: allocate packed output.
    let packed_per_row = num_symbols / 2;
    let packed_buf = dev.alloc_zeros::<u8>(n_rows * packed_per_row)?;
    let packed_shape = candle_core::Shape::from_dims(&[n_rows, packed_per_row]);

    // Borrow + launch in an inner scope so the storage guards drop before
    // we move `row_scales` into the return tuple.
    {
        let (w_storage, w_layout) = weight_contig.storage_and_layout();
        let w_storage = match &*w_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("QTIP quantize CUDA: weight must be CUDA storage"),
        };
        let (lut_storage, lut_layout) = lut_contig.storage_and_layout();
        let lut_storage = match &*lut_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("QTIP quantize CUDA: lut must be CUDA storage"),
        };
        let (rs_storage, rs_layout) = row_scales.storage_and_layout();
        let rs_storage = match &*rs_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("QTIP quantize CUDA: row_scales must be CUDA storage"),
        };
        let (w_ptr, _w_guard) =
            slice_ptr(w_storage.as_cuda_slice::<f32>()?, w_layout.start_offset());
        let (lut_ptr, _lut_guard) = slice_ptr(
            lut_storage.as_cuda_slice::<f32>()?,
            lut_layout.start_offset(),
        );
        let (rs_ptr, _rs_guard) =
            slice_ptr(rs_storage.as_cuda_slice::<f32>()?, rs_layout.start_offset());
        let (pkd_ptr, pkd_guard) = slice_ptr(&packed_buf, 0);

        match mode {
            QtipMode::Greedy => unsafe {
                ffi::launch_qtip_quantize_rows_greedy_f32(
                    w_ptr as *const _,
                    lut_ptr as *const _,
                    rs_ptr as *const _,
                    pkd_ptr as *mut _,
                    n_rows as i32,
                    k_in as i32,
                    num_symbols as i32,
                    dev.cuda_stream().cu_stream(),
                );
            },
            QtipMode::Viterbi => {
                // Allocate per-batch scratch. With prefix-grouped backtrace,
                //   bt_bytes_per_row = num_symbols * 2^(L-K) = num_symbols * 4096
                // Cap rows_in_flight so total <= VITERBI_MAX_SCRATCH_BYTES.
                // Example: Qwen 7B mlp.down_proj (in=18944 → num_symbols=9472)
                // gives bt_bytes_per_row ≈ 38 MB → ~158 rows in flight in 6 GB,
                // vs ~10 rows under the old per-state backtrace.
                let bt_bytes_per_row = num_symbols * QTIP_PREFIX_COUNT;
                let mut rows_in_flight = (VITERBI_MAX_SCRATCH_BYTES / bt_bytes_per_row).max(1);
                if rows_in_flight > n_rows {
                    rows_in_flight = n_rows;
                }

                let cost_a = dev.alloc_zeros::<f32>(rows_in_flight * QTIP_LUT_SIZE)?;
                let cost_b = dev.alloc_zeros::<f32>(rows_in_flight * QTIP_LUT_SIZE)?;
                let backtrace = dev.alloc_zeros::<u8>(rows_in_flight * bt_bytes_per_row)?;

                let (ca_ptr, ca_guard) = slice_ptr(&cost_a, 0);
                let (cb_ptr, cb_guard) = slice_ptr(&cost_b, 0);
                let (bt_ptr, bt_guard) = slice_ptr(&backtrace, 0);

                let mut row_offset = 0usize;
                while row_offset < n_rows {
                    let this_batch = rows_in_flight.min(n_rows - row_offset);
                    unsafe {
                        ffi::launch_qtip_quantize_rows_viterbi_f32(
                            w_ptr as *const _,
                            lut_ptr as *const _,
                            rs_ptr as *const _,
                            pkd_ptr as *mut _,
                            ca_ptr as *mut _,
                            cb_ptr as *mut _,
                            bt_ptr as *mut _,
                            this_batch as i32,
                            k_in as i32,
                            num_symbols as i32,
                            row_offset as i32,
                            dev.cuda_stream().cu_stream(),
                        );
                    }
                    row_offset += this_batch;
                }

                drop(ca_guard);
                drop(cb_guard);
                drop(bt_guard);
                // Scratch buffers freed when their CudaSlice drops at scope exit.
                let _ = (cost_a, cost_b, backtrace);
            }
        }

        drop(pkd_guard);
    }

    let packed_storage = CudaStorage::wrap_cuda_slice(packed_buf, dev.clone());
    let packed = Tensor::from((Storage::Cuda(packed_storage), packed_shape));

    Ok((packed, row_scales))
}
