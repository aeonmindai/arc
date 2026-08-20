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

use super::device_guard::ensure_same_cuda_device;
use super::ffi;
use super::trellis_v4l12::{RowScaleHoist, Rung};
use super::QtipCodebook;

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
    codebook: QtipCodebook,
) -> Result<Tensor> {
    let cb_mult = codebook.cuda_mult();
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
    // The launch below uses `blocks`' stream, so every other operand must be on
    // the same ORDINAL, not merely on some CUDA device.
    const OP: &str = "QTIP dequantize CUDA";
    ensure_same_cuda_device(
        OP,
        "row_scales",
        row_scales.device(),
        "blocks",
        blocks.device(),
    )?;
    ensure_same_cuda_device(OP, "lut", lut.device(), "blocks", blocks.device())?;

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
                    cb_mult,
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
                    cb_mult,
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
                    cb_mult,
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
    codebook: QtipCodebook,
) -> Result<Tensor> {
    let cb_mult = codebook.cuda_mult();
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
    // Kind alone is not enough: a `cuda:0` activation launched on `blocks`'
    // `cuda:1` stream reads an unmapped pointer. Compare ordinals.
    const OP: &str = "QTIP fused gemv CUDA";
    ensure_same_cuda_device(
        OP,
        "row_scales",
        row_scales.device(),
        "blocks",
        blocks.device(),
    )?;
    ensure_same_cuda_device(OP, "lut", lut.device(), "blocks", blocks.device())?;
    ensure_same_cuda_device(OP, "x_rotated", x_2d.device(), "blocks", blocks.device())?;

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
                    cb_mult,
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
                    cb_mult,
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
                    cb_mult,
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

/// Fused decode + gemv for the **V=4 / L=12** family — the shared-memory-table
/// rung, with K supplied by the caller's [`Rung`].
///
/// The K=4/V=2/L=16 sibling of this function is [`fused_gemv_cuda`]. Same
/// `x_rotated` contract, same `[1, n_rows]` output. Four things differ, and all
/// four are checked here rather than assumed:
///
/// * `lut` is **BF16** and must hold exactly `2^12 × 4 = 16,384` values
///   (32,768 B). That size is the reason the family exists — it fits static
///   shared memory — so a wrong-sized table is a wrong-geometry artifact and is
///   refused, not indexed.
/// * `blocks` rows must be `ceil(num_symbols · K / 8)` bytes. At K=9 that is
///   not a whole number of symbols per byte, so the K=4 path's
///   `num_symbols / 2` and the K=8 path's `num_symbols` are both wrong.
/// * `rung` selects the symbol width. An unsupported K never reaches the
///   kernel, whose `default:` arm returns without launching and would leave `y`
///   holding whatever it held.
/// * `hoist` selects the row-scale policy. [`RowScaleHoist::Off`] is what the
///   CPU parity reference pins; [`RowScaleHoist::On`] reassociates the sum and
///   is therefore not bit-comparable.
///
/// There is deliberately no codebook selector: the computed `sum2` codebook is
/// V=2-specific, so this family is table-only.
pub(crate) fn fused_gemv_v4l12_cuda(
    blocks: &Tensor,
    row_scales: &Tensor,
    lut: &Tensor,
    x_rotated: &Tensor,
    in_features: usize,
    rung: Rung,
    hoist: RowScaleHoist,
) -> Result<Tensor> {
    use super::trellis_v4l12 as v4l12;

    const OP: &str = "QTIP v4l12 fused gemv CUDA";

    let n_rows = row_scales.dim(0)?;
    let packed_per_row = blocks.dim(1)?;
    let num_symbols = rung.num_symbols(in_features);

    if blocks.dtype() != DType::U8 {
        candle_core::bail!("{OP}: blocks dtype must be U8, got {:?}", blocks.dtype());
    }
    if row_scales.dtype() != DType::F32 {
        candle_core::bail!(
            "{OP}: row_scales dtype must be F32, got {:?}",
            row_scales.dtype()
        );
    }
    // The table's dtype IS the family. An F32 table here means a K=4/V=2
    // artifact reached a V=4 launcher, which would decode 65,536 states' worth
    // of f32 as 16,384 bf16 and produce plausible garbage.
    if lut.dtype() != DType::BF16 {
        candle_core::bail!(
            "{OP}: table dtype must be BF16 at this geometry, got {:?}. An F32 table is the \
             K=4/V=2/L=16 rung's; these artifacts are not interchangeable.",
            lut.dtype()
        );
    }
    if lut.elem_count() != v4l12::LUT_ENTRIES {
        candle_core::bail!(
            "{OP}: table has {} values, expected {} (2^{} × {} = {} B)",
            lut.elem_count(),
            v4l12::LUT_ENTRIES,
            v4l12::L,
            v4l12::V,
            v4l12::LUT_BYTES
        );
    }
    if !in_features.is_multiple_of(v4l12::V as usize) {
        candle_core::bail!(
            "{OP}: in_features {in_features} must be a multiple of V={} (each symbol decodes \
             to V weights)",
            v4l12::V
        );
    }
    // `ceil(num_symbols * K / 8)`, the general form. If these disagree the
    // caller is holding a row packed at a different K — and since the table,
    // the state width and the activation indexing are all K-independent, the
    // row length is the ONLY thing that can catch it.
    let want_bytes = rung.row_stride(num_symbols);
    if packed_per_row != want_bytes {
        candle_core::bail!(
            "{OP}: blocks row is {packed_per_row} B but K={} with in_features={in_features} \
             ({num_symbols} symbols) needs a {want_bytes}-byte stride ({} data + {} padding). \
             The kernel reads a compile-time byte count with NO clamp — the padding is what \
             makes the last symbol's read in-bounds, so a short row is an out-of-bounds read, \
             not a slow path.",
            rung.k(),
            rung.data_bytes(num_symbols),
            rung.pad_bytes(num_symbols)
        );
    }
    if !blocks.layout().is_contiguous()
        || !row_scales.layout().is_contiguous()
        || !lut.layout().is_contiguous()
    {
        candle_core::bail!("{OP}: blocks/scales/table must be contiguous");
    }
    let x_2d = match x_rotated.dims() {
        [k] if *k == in_features => x_rotated.unsqueeze(0)?,
        [b, k] if *b == 1 && *k == in_features => x_rotated.clone(),
        other => candle_core::bail!(
            "{OP}: x_rotated must be [k_in] or [1, k_in]; got {other:?} (k_in={in_features})"
        ),
    };
    let x_2d = x_2d.contiguous()?;

    let dev = match blocks.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("{OP}: blocks must live on CUDA"),
    };
    ensure_same_cuda_device(
        OP,
        "row_scales",
        row_scales.device(),
        "blocks",
        blocks.device(),
    )?;
    ensure_same_cuda_device(OP, "lut", lut.device(), "blocks", blocks.device())?;
    ensure_same_cuda_device(OP, "x_rotated", x_2d.device(), "blocks", blocks.device())?;

    let out_shape = candle_core::Shape::from_dims(&[1, n_rows]);

    let (blocks_storage, blocks_layout) = blocks.storage_and_layout();
    let blocks_storage = match &*blocks_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("{OP}: blocks storage must be CUDA"),
    };
    let (scales_storage, scales_layout) = row_scales.storage_and_layout();
    let scales_storage = match &*scales_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("{OP}: scales storage must be CUDA"),
    };
    let (lut_storage, lut_layout) = lut.storage_and_layout();
    let lut_storage = match &*lut_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("{OP}: table storage must be CUDA"),
    };
    let (x_storage, x_layout) = x_2d.storage_and_layout();
    let x_storage = match &*x_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("{OP}: x storage must be CUDA"),
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
        lut_storage.as_cuda_slice::<bf16>()?,
        lut_layout.start_offset(),
    );

    let hoist_abi = hoist.as_abi();
    let res = match x_2d.dtype() {
        DType::BF16 => {
            let (x_ptr, _x_guard) =
                slice_ptr(x_storage.as_cuda_slice::<bf16>()?, x_layout.start_offset());
            let out_buf = dev.alloc_zeros::<bf16>(n_rows)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                ffi::launch_qtip_fused_gemv_v4_l12_bf16(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    x_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    num_symbols as i32,
                    rung.k() as i32,
                    hoist_abi,
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
                ffi::launch_qtip_fused_gemv_v4_l12_f16(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    x_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    num_symbols as i32,
                    rung.k() as i32,
                    hoist_abi,
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
                ffi::launch_qtip_fused_gemv_v4_l12_f32(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    x_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    num_symbols as i32,
                    rung.k() as i32,
                    hoist_abi,
                    dev.cuda_stream().cu_stream(),
                );
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }
        other => candle_core::bail!("{OP}: unsupported x dtype {other:?}"),
    };

    Ok(Tensor::from((Storage::Cuda(res), out_shape)))
}

/// On-device MoE gather + fused decode + gemv (CUDA-graph-capturable decode).
///
/// The capturable sibling of `fused_gemv_cuda` for stacked experts. Reads each
/// `(token, slot)` pair's expert id from `indices` **on-device** and runs the
/// trellis gemv against that expert's packed rows — so the MoE dispatch stays
/// on the stream with NO device->host sync (unlike `gather_forward_cuda`, which
/// pulls indices to the host). Intended for the small-`n_pairs` decode regime
/// (`n_pairs = n_tokens * top_k`); prefill keeps the grouped host path.
///
/// Shapes:
///   * `blocks`     : `[E, n_rows, packed_per_row]` U8 (3-D stacked experts)
///   * `row_scales` : `[E, n_rows]` F32
///   * `lut`        : `[2^L * V]` F32
///   * `x_rotated`  : `[n_pairs, in_features]` (ALREADY rotated)
///   * `indices`    : `[n_pairs]` U32 (expert id per pair)
/// Returns `[n_pairs, n_rows]` (same dtype as `x_rotated`).
pub(crate) fn gather_gemv_cuda(
    blocks: &Tensor,
    row_scales: &Tensor,
    lut: &Tensor,
    x_rotated: &Tensor,
    indices: &Tensor,
    in_features: usize,
    codebook: QtipCodebook,
) -> Result<Tensor> {
    let cb_mult = codebook.cuda_mult();
    // 3-D stacked layout: [E, n_rows, packed_per_row] / [E, n_rows].
    let num_experts = blocks.dim(0)?;
    let n_rows = row_scales.dim(1)?;
    let packed_per_row = blocks.dim(2)?;
    let num_symbols = in_features / super::V as usize;

    if blocks.dtype() != DType::U8 {
        candle_core::bail!("QTIP gather gemv CUDA: blocks dtype must be U8");
    }
    if row_scales.dtype() != DType::F32 {
        candle_core::bail!("QTIP gather gemv CUDA: row_scales dtype must be F32");
    }
    if lut.dtype() != DType::F32 {
        candle_core::bail!("QTIP gather gemv CUDA: lut dtype must be F32");
    }
    if indices.dtype() != DType::U32 {
        candle_core::bail!("QTIP gather gemv CUDA: indices dtype must be U32");
    }
    if !blocks.layout().is_contiguous()
        || !row_scales.layout().is_contiguous()
        || !lut.layout().is_contiguous()
    {
        candle_core::bail!("QTIP gather gemv CUDA: blocks/scales/lut must be contiguous");
    }

    // x_rotated: [n_pairs, in_features]; indices: [n_pairs].
    let (n_pairs, x_cols) = x_rotated.dims2()?;
    if x_cols != in_features {
        candle_core::bail!(
            "QTIP gather gemv CUDA: x_rotated cols {x_cols} != in_features {in_features}"
        );
    }
    let idx_pairs = indices.elem_count();
    if idx_pairs != n_pairs {
        candle_core::bail!("QTIP gather gemv CUDA: indices len {idx_pairs} != n_pairs {n_pairs}");
    }
    // The kernel maps one pair to one `grid.y` index. Past `maxGridSize[1]` the
    // launch fails, the `extern "C"` launcher discards the status, and the
    // caller silently receives the zero-initialized output buffer below.
    super::gather_policy::check_gather_gemv_pairs(n_pairs, "QTIP gather gemv CUDA")?;
    let x_2d = x_rotated.contiguous()?;
    let indices = indices.flatten_all()?.contiguous()?;

    let dev = match blocks.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("QTIP gather gemv CUDA: blocks must live on CUDA"),
    };
    // MoE path: under a multi-GPU expert map the activations and the expert
    // weights are the pair most likely to disagree. Compare ordinals.
    const OP: &str = "QTIP gather gemv CUDA";
    ensure_same_cuda_device(
        OP,
        "row_scales",
        row_scales.device(),
        "blocks",
        blocks.device(),
    )?;
    ensure_same_cuda_device(OP, "lut", lut.device(), "blocks", blocks.device())?;
    ensure_same_cuda_device(OP, "x_rotated", x_2d.device(), "blocks", blocks.device())?;
    ensure_same_cuda_device(OP, "indices", indices.device(), "blocks", blocks.device())?;

    let out_shape = candle_core::Shape::from_dims(&[n_pairs, n_rows]);

    let (blocks_storage, blocks_layout) = blocks.storage_and_layout();
    let blocks_storage = match &*blocks_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP gather gemv CUDA: blocks storage must be CUDA"),
    };
    let (scales_storage, scales_layout) = row_scales.storage_and_layout();
    let scales_storage = match &*scales_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP gather gemv CUDA: scales storage must be CUDA"),
    };
    let (lut_storage, lut_layout) = lut.storage_and_layout();
    let lut_storage = match &*lut_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP gather gemv CUDA: lut storage must be CUDA"),
    };
    let (x_storage, x_layout) = x_2d.storage_and_layout();
    let x_storage = match &*x_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP gather gemv CUDA: x storage must be CUDA"),
    };
    let (idx_storage, idx_layout) = indices.storage_and_layout();
    let idx_storage = match &*idx_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP gather gemv CUDA: indices storage must be CUDA"),
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
    let (idx_ptr, _idx_guard) = slice_ptr(
        idx_storage.as_cuda_slice::<u32>()?,
        idx_layout.start_offset(),
    );

    let n_out = n_pairs * n_rows;

    let res = match x_2d.dtype() {
        DType::BF16 => {
            let (x_ptr, _x_guard) =
                slice_ptr(x_storage.as_cuda_slice::<bf16>()?, x_layout.start_offset());
            let out_buf = crate::arc_outbuf::alloc_out_fully_written::<bf16>(&dev, n_out)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                ffi::launch_qtip_gather_gemv_v2_k4_l16_bf16(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    x_ptr as *const _,
                    idx_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    num_symbols as i32,
                    n_pairs as i32,
                    num_experts as i32,
                    cb_mult,
                    dev.cuda_stream().cu_stream(),
                );
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }
        DType::F16 => {
            let (x_ptr, _x_guard) =
                slice_ptr(x_storage.as_cuda_slice::<f16>()?, x_layout.start_offset());
            let out_buf = crate::arc_outbuf::alloc_out_fully_written::<f16>(&dev, n_out)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                ffi::launch_qtip_gather_gemv_v2_k4_l16_f16(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    x_ptr as *const _,
                    idx_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    num_symbols as i32,
                    n_pairs as i32,
                    num_experts as i32,
                    cb_mult,
                    dev.cuda_stream().cu_stream(),
                );
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }
        DType::F32 => {
            let (x_ptr, _x_guard) =
                slice_ptr(x_storage.as_cuda_slice::<f32>()?, x_layout.start_offset());
            let out_buf = crate::arc_outbuf::alloc_out_fully_written::<f32>(&dev, n_out)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                ffi::launch_qtip_gather_gemv_v2_k4_l16_f32(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    x_ptr as *const _,
                    idx_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    num_symbols as i32,
                    n_pairs as i32,
                    num_experts as i32,
                    cb_mult,
                    dev.cuda_stream().cu_stream(),
                );
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }
        other => candle_core::bail!("QTIP gather gemv CUDA: unsupported x dtype {other:?}"),
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
    // The rotation kernel runs on `x`'s stream and dereferences `signs`.
    ensure_same_cuda_device(
        "QTIP rotate-x CUDA",
        "signs",
        signs.device(),
        "x",
        x.device(),
    )?;

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

use super::{QtipMode, TrellisSearch};

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
    ensure_same_cuda_device(
        "QTIP rotate-weight CUDA",
        "signs",
        signs.device(),
        "weight",
        weight.device(),
    )?;

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

/// Compute per-row scale = max(|row|) / `divisor` (or 1.0 if max=0), returning
/// a fresh F32 tensor of shape `[n_rows]` on the same device as `weight`.
///
/// `divisor` is the codebook's [`QtipCodebook::scale_divisor`]: 3.0 for the
/// unit-sigma Gaussian LUT, 3*sigma for the computed sum2 codebook.
pub(crate) fn compute_row_scales_cuda(weight: &Tensor, divisor: f32) -> Result<Tensor> {
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
            divisor,
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

/// Scratch budget (bytes) for one Viterbi launch's rows-in-flight. The 6 GB
/// default was tuned for an 80 GB H100; on a larger card (H200 143 GB) it
/// starves the kernel to ~438 rows in flight, leaving the SMs idle through
/// every launch tail. Override with `ARC_VITERBI_SCRATCH_GB` to raise the
/// rows-in-flight (bit-identical: same per-row Viterbi, just more rows per
/// launch and fewer serial launches). RUN-161.
fn viterbi_scratch_bytes() -> usize {
    std::env::var("ARC_VITERBI_SCRATCH_GB")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|g| *g > 0.0)
        .map(|g| (g * (1u64 << 30) as f64) as usize)
        .unwrap_or(VITERBI_MAX_SCRATCH_BYTES)
}

/// 2^L = 65536, the trellis state count. Matches `LUT_SIZE` in `mod.rs`.
const QTIP_LUT_SIZE: usize = 1 << super::L;

/// 2^(L-K) = 4096, the trellis prefix count. The prefix-grouped Viterbi
/// kernel stores one backtrace entry per (timestep, prefix) — 16× smaller
/// than the naive per-state version because all 16 successors of a given
/// prefix share the same predecessor reduction.
const QTIP_PREFIX_COUNT: usize = 1 << (super::L - super::K);

/// Largest beam width `kernels/qtip/qtip_beam.cu` can run, read from the
/// kernel itself so the Rust limit can never drift from the CUDA one.
///
/// Returns 0 when the kernels were not compiled in.
pub(crate) fn beam_max_width() -> usize {
    if !ffi::HAVE_QTIP_KERNELS {
        return 0;
    }
    let w = unsafe { ffi::qtip_beam_max_width() };
    if w < 0 {
        0
    } else {
        w as usize
    }
}

/// The Rust trellis geometry as the kernels pack it: `(K << 16) | (V << 8) | L`.
fn rust_geometry_word() -> i32 {
    ((super::K as i32) << 16) | ((super::V as i32) << 8) | (super::L as i32)
}

/// Refuse a geometry the non-beam CUDA kernels were not compiled for.
///
/// `qtip_beam.cu` takes `(K, V, L)` as launch arguments and dispatches, so the
/// beam follows `qtip::{K, V, L}` automatically. The exhaustive DP, the greedy
/// walk and the scale refinement in `qtip_quantize.cu` do NOT: their prefix
/// blocking and their two-symbols-per-byte packing are K=4/V=2/L=16 by
/// construction. Without this check, moving the Rust consts would run those
/// kernels over buffers a different rung sized — a silent corruption with no
/// kernel error and no wrong-looking output shape.
fn ensure_non_beam_geometry(what: &str) -> Result<()> {
    let compiled = unsafe { ffi::qtip_exhaustive_geometry() };
    let wanted = rust_geometry_word();
    if compiled != wanted {
        candle_core::bail!(
            "QTIP quantize CUDA: {what} is compiled for K={}/V={}/L={}, but the Rust \
             trellis geometry is K={}/V={}/L={}. Refusing to run: only the beam search \
             (`kernels/qtip/qtip_beam.cu`) is geometry-parametric.",
            (compiled >> 16) & 0xFF,
            (compiled >> 8) & 0xFF,
            compiled & 0xFF,
            super::K,
            super::V,
            super::L,
        );
    }
    Ok(())
}

/// One-shot quantize entry point. Returns `(packed_blocks, row_scales)`:
/// * `packed_blocks` — `[n_rows, num_symbols / (8/K)]` U8, `8/K` symbols per byte.
/// * `row_scales`    — `[n_rows]` F32, per-row scale.
///
/// `weight_rotated_f32` should already be in the rotated frame (caller is
/// responsible for applying `rotate_weight_rows_cuda` first when rotation
/// is enabled). This split lets the test path verify each step on its own.
///
/// `search` selects the trellis search when `mode == Viterbi`:
/// [`TrellisSearch::Exhaustive`] runs the prefix-grouped DP over all `2^L`
/// states; [`TrellisSearch::Beam`] runs the pruned kernel. `Greedy` ignores it
/// (a greedy walk is not a trellis search at all — see `bake_header_line`).
pub(crate) fn quantize_rows_cuda(
    weight_rotated_f32: &Tensor,
    lut: &Tensor,
    mode: QtipMode,
    search: TrellisSearch,
    codebook: QtipCodebook,
) -> Result<(Tensor, Tensor)> {
    let cb_mult = codebook.cuda_mult();
    if !ffi::HAVE_QTIP_KERNELS {
        candle_core::bail!("QTIP quantize CUDA: kernels not compiled in");
    }
    let (n_rows, k_in) = weight_rotated_f32.dims2()?;
    let num_symbols = k_in / super::V as usize;
    // Symbols per packed byte, from the rung's K: 2 at K=4, 1 at K=8. Derived
    // rather than written as `2` so the packing follows `qtip::K` the day it
    // moves — `qtip_geom.cuh::SYMS_PER_BYTE` is the CUDA side of this number.
    let syms_per_byte = 8 / super::K as usize;
    if !num_symbols.is_multiple_of(syms_per_byte) {
        candle_core::bail!(
            "QTIP quantize CUDA: num_symbols ({num_symbols}) must be a multiple of \
             {syms_per_byte} for K={} packing",
            super::K
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
    ensure_same_cuda_device(
        "QTIP quantize CUDA",
        "lut",
        lut.device(),
        "weight",
        weight_rotated_f32.device(),
    )?;

    let weight_contig = weight_rotated_f32.contiguous()?;
    let lut_contig = lut.contiguous()?;

    // Step 1: row scales.
    let row_scales = compute_row_scales_cuda(&weight_contig, codebook.scale_divisor())?;

    // Step 2: allocate packed output.
    let packed_per_row = num_symbols / syms_per_byte;
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
            QtipMode::Greedy => {
                ensure_non_beam_geometry("the greedy quantize kernel")?;
                unsafe {
                    ffi::launch_qtip_quantize_rows_greedy_f32(
                        w_ptr as *const _,
                        lut_ptr as *const _,
                        rs_ptr as *const _,
                        pkd_ptr as *mut _,
                        n_rows as i32,
                        k_in as i32,
                        num_symbols as i32,
                        cb_mult,
                        dev.cuda_stream().cu_stream(),
                    );
                }
            }
            QtipMode::Viterbi if !matches!(search, TrellisSearch::Exhaustive) => {
                // wave13-AF beam kernel. The live state set is `width` entries
                // in shared memory, so there is NO cost ping-pong scratch at
                // all — only the compacted backtrace, `width * 4` bytes per
                // timestep instead of the exhaustive kernel's 4096.
                let width = match search {
                    TrellisSearch::Beam { width } => width,
                    TrellisSearch::Exhaustive => unreachable!("guarded by the match arm"),
                };
                let max_w = beam_max_width();
                if width == 0 || width > max_w {
                    candle_core::bail!(
                        "QTIP quantize CUDA: beam width {width} is outside the kernel's \
                         supported range 1..={max_w}. Refusing to substitute a different \
                         width — a bake must never silently change its search."
                    );
                }
                let trace_bytes_per_row = num_symbols * width * 4;
                let mut rows_in_flight = (viterbi_scratch_bytes() / trace_bytes_per_row).max(1);
                if rows_in_flight > n_rows {
                    rows_in_flight = n_rows;
                }

                // Uninit alloc: the kernel writes every trace slot it later
                // reads (the backtrace walk only visits slots the forward pass
                // populated), so zeroing GBs of scratch is wasted bandwidth.
                let trace = unsafe { dev.alloc::<u32>(rows_in_flight * num_symbols * width)? };
                let (tr_ptr, tr_guard) = slice_ptr(&trace, 0);

                let mut row_offset = 0usize;
                while row_offset < n_rows {
                    let this_batch = rows_in_flight.min(n_rows - row_offset);
                    // The geometry is passed, not assumed: the kernel is
                    // templated on (K, V, L) and dispatches, so this call
                    // follows `qtip::{K, V, L}` with no change here the day
                    // they move to K=8/V=4/L=12.
                    let rc = unsafe {
                        ffi::launch_qtip_quantize_rows_beam_geom_f32(
                            w_ptr as *const _,
                            lut_ptr as *const _,
                            rs_ptr as *const _,
                            pkd_ptr as *mut _,
                            tr_ptr as *mut _,
                            this_batch as i32,
                            k_in as i32,
                            num_symbols as i32,
                            row_offset as i32,
                            width as i32,
                            super::K as i32,
                            super::V as i32,
                            super::L as i32,
                            cb_mult,
                            dev.cuda_stream().cu_stream(),
                        )
                    };
                    if rc == -2 {
                        candle_core::bail!(
                            "QTIP quantize CUDA: no beam kernel is compiled for the trellis \
                             geometry K={}/V={}/L={}. Add the geometry to the dispatch table \
                             in `kernels/qtip/qtip_beam.cu` — never fall back to another rung.",
                            super::K,
                            super::V,
                            super::L,
                        );
                    }
                    if rc != 0 {
                        candle_core::bail!(
                            "QTIP quantize CUDA: beam kernel refused width {width} (rc={rc})"
                        );
                    }
                    row_offset += this_batch;
                }

                drop(tr_guard);
                let _ = trace;
            }
            QtipMode::Viterbi => {
                ensure_non_beam_geometry("the exhaustive Viterbi kernel")?;
                // Allocate per-batch scratch. With prefix-grouped backtrace,
                //   bt_bytes_per_row = num_symbols * 2^(L-K) = num_symbols * 4096
                // Cap rows_in_flight so total <= VITERBI_MAX_SCRATCH_BYTES.
                // Example: Qwen 7B mlp.down_proj (in=18944 → num_symbols=9472)
                // gives bt_bytes_per_row ≈ 38 MB → ~158 rows in flight in 6 GB,
                // vs ~10 rows under the old per-state backtrace.
                // Per-row scratch = backtrace + cost_a + cost_b (the two cost
                // ping-pong tables are each LUT_SIZE f32). Budget covers all
                // three so a raised cap can't silently OOM on the cost buffers.
                let bt_bytes_per_row = num_symbols * QTIP_PREFIX_COUNT;
                let per_row_bytes = bt_bytes_per_row + 2 * QTIP_LUT_SIZE * 4;
                let mut rows_in_flight = (viterbi_scratch_bytes() / per_row_bytes).max(1);
                if rows_in_flight > n_rows {
                    rows_in_flight = n_rows;
                }

                // RUN-161: uninit alloc (not alloc_zeros) — the kernel fully
                // writes cost_a at t=0, cost_b in Phase B, and backtrace in
                // Phase C before any read, so zeroing GBs of scratch every
                // matrix call is pure wasted bandwidth.
                let cost_a = unsafe { dev.alloc::<f32>(rows_in_flight * QTIP_LUT_SIZE)? };
                let cost_b = unsafe { dev.alloc::<f32>(rows_in_flight * QTIP_LUT_SIZE)? };
                let backtrace = unsafe { dev.alloc::<u8>(rows_in_flight * bt_bytes_per_row)? };

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
                            cb_mult,
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

    // Least-squares scale refinement: s* = dot(w, lut_values) / dot(lut_values, lut_values).
    // BUG: produces cos=0.628 at full-size N=2048 (kernel replay issue under
    // investigation). Gated behind ARC_QTIP_REFINE_SCALES=1 until fixed.
    if std::env::var("ARC_QTIP_REFINE_SCALES").as_deref() == Ok("1") {
        ensure_non_beam_geometry("the scale-refinement kernel")?;
        let (w_storage, w_layout) = weight_contig.storage_and_layout();
        let w_storage = match &*w_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("QTIP refine scales: weight must be CUDA storage"),
        };
        let (lut_storage, lut_layout) = lut_contig.storage_and_layout();
        let lut_storage = match &*lut_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("QTIP refine scales: lut must be CUDA storage"),
        };
        let (rs_storage, rs_layout) = row_scales.storage_and_layout();
        let rs_storage = match &*rs_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("QTIP refine scales: row_scales must be CUDA storage"),
        };
        let (w_ptr, _wg) = slice_ptr(w_storage.as_cuda_slice::<f32>()?, w_layout.start_offset());
        let (lut_ptr, _lg) = slice_ptr(
            lut_storage.as_cuda_slice::<f32>()?,
            lut_layout.start_offset(),
        );
        let (rs_ptr, _rg) = slice_ptr(rs_storage.as_cuda_slice::<f32>()?, rs_layout.start_offset());
        let (pkd_ptr, _pg) = slice_ptr(&packed_buf, 0);

        unsafe {
            ffi::launch_qtip_refine_scales_f32(
                w_ptr as *const _,
                pkd_ptr as *const _,
                lut_ptr as *const _,
                rs_ptr as *mut _,
                n_rows as i32,
                k_in as i32,
                num_symbols as i32,
                cb_mult,
                dev.cuda_stream().cu_stream(),
            );
        }
    }

    let packed_storage = CudaStorage::wrap_cuda_slice(packed_buf, dev.clone());
    let packed = Tensor::from((Storage::Cuda(packed_storage), packed_shape));

    Ok((packed, row_scales))
}

// ===========================================================================
// qtip2b — bitshift-trellis computed-codebook ops (no LUT anywhere).
// ===========================================================================
//
// These are the qtip2b (K=2/V=1) siblings of the LUT-rung helpers above.
// The codebook is computed in-register from the MCG multiplier, so none of
// these take (or upload) a LUT tensor. Rotation helpers (`rotate_x_cuda`,
// `rotate_weight_rows_cuda`) are format-agnostic and shared with the LUT
// rung.

use super::bitshift::QTIP2B_SCALE_DIVISOR;

/// Symbols packed per byte for qtip2b (2-bit symbols).
const Q2B_SYMS_PER_BYTE: usize = 4;
/// 2^L trellis states (cost-table width for the Viterbi scratch).
const Q2B_NSTATES: usize = 1 << 16;
/// 2^(L-K) prefixes (per-prefix scratch width).
const Q2B_PREFIX_COUNT: usize = 1 << 14;

/// Dequantize qtip2b-packed weights into `[n_rows, in_features]` on GPU
/// (rotated frame).
pub(crate) fn dequantize_2b_cuda(
    blocks: &Tensor,
    row_scales: &Tensor,
    mcg_mult: u32,
    in_features: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let n_rows = row_scales.dim(0)?;
    let packed_per_row = blocks.dim(1)?;
    let num_symbols = in_features; // V = 1
    if blocks.dtype() != DType::U8 {
        candle_core::bail!("qtip2b dequantize CUDA: blocks dtype must be U8");
    }
    if row_scales.dtype() != DType::F32 {
        candle_core::bail!("qtip2b dequantize CUDA: row_scales dtype must be F32");
    }
    if !blocks.layout().is_contiguous() || !row_scales.layout().is_contiguous() {
        candle_core::bail!("qtip2b dequantize CUDA: all inputs must be contiguous");
    }

    let dev = match blocks.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("qtip2b dequantize CUDA: blocks must live on CUDA"),
    };
    ensure_same_cuda_device(
        "qtip2b dequantize CUDA",
        "row_scales",
        row_scales.device(),
        "blocks",
        blocks.device(),
    )?;

    let (blocks_storage, blocks_layout) = blocks.storage_and_layout();
    let blocks_storage = match &*blocks_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b dequantize CUDA: blocks must be CUDA storage"),
    };
    let (scales_storage, scales_layout) = row_scales.storage_and_layout();
    let scales_storage = match &*scales_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b dequantize CUDA: scales must be CUDA storage"),
    };

    let (blocks_ptr, _blocks_guard) = slice_ptr(
        blocks_storage.as_cuda_slice::<u8>()?,
        blocks_layout.start_offset(),
    );
    let (scales_ptr, _scales_guard) = slice_ptr(
        scales_storage.as_cuda_slice::<f32>()?,
        scales_layout.start_offset(),
    );

    let num_weights = n_rows * in_features;
    let out_shape = candle_core::Shape::from_dims(&[n_rows, in_features]);

    macro_rules! dequant_dtype {
        ($T:ty, $launch:expr) => {{
            let out_buf = dev.alloc_zeros::<$T>(num_weights)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                $launch(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    num_symbols as i32,
                    mcg_mult,
                    dev.cuda_stream().cu_stream(),
                );
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }};
    }

    let res = match out_dtype {
        DType::BF16 => dequant_dtype!(bf16, ffi::launch_qtip2b_dequantize_bf16),
        DType::F16 => dequant_dtype!(f16, ffi::launch_qtip2b_dequantize_f16),
        DType::F32 => dequant_dtype!(f32, ffi::launch_qtip2b_dequantize_f32),
        other => candle_core::bail!("qtip2b dequantize CUDA: unsupported out dtype {other:?}"),
    };

    Ok(Tensor::from((Storage::Cuda(res), out_shape)))
}

/// Single-token fused decode + GEMV for a 2-D qtip2b layer. `x_rotated` must
/// already be rotated. Returns `[1, n_rows]`.
pub(crate) fn fused_gemv_2b_cuda(
    blocks: &Tensor,
    row_scales: &Tensor,
    mcg_mult: u32,
    x_rotated: &Tensor,
    in_features: usize,
) -> Result<Tensor> {
    let n_rows = row_scales.dim(0)?;
    let packed_per_row = blocks.dim(1)?;
    let num_symbols = in_features; // V = 1

    if blocks.dtype() != DType::U8 {
        candle_core::bail!("qtip2b fused gemv CUDA: blocks dtype must be U8");
    }
    if row_scales.dtype() != DType::F32 {
        candle_core::bail!("qtip2b fused gemv CUDA: row_scales dtype must be F32");
    }
    if !blocks.layout().is_contiguous() || !row_scales.layout().is_contiguous() {
        candle_core::bail!("qtip2b fused gemv CUDA: blocks/scales must be contiguous");
    }

    let x_2d = match x_rotated.dims() {
        [k] if *k == in_features => x_rotated.unsqueeze(0)?,
        [b, k] if *b == 1 && *k == in_features => x_rotated.clone(),
        other => candle_core::bail!(
            "qtip2b fused gemv CUDA: x_rotated must be [k_in] or [1, k_in]; got {other:?} (k_in={in_features})"
        ),
    };
    let x_2d = x_2d.contiguous()?;

    let dev = match blocks.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("qtip2b fused gemv CUDA: blocks must live on CUDA"),
    };
    const OP: &str = "qtip2b fused gemv CUDA";
    ensure_same_cuda_device(
        OP,
        "row_scales",
        row_scales.device(),
        "blocks",
        blocks.device(),
    )?;
    ensure_same_cuda_device(OP, "x_rotated", x_2d.device(), "blocks", blocks.device())?;

    let out_shape = candle_core::Shape::from_dims(&[1, n_rows]);

    let (blocks_storage, blocks_layout) = blocks.storage_and_layout();
    let blocks_storage = match &*blocks_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b fused gemv CUDA: blocks storage must be CUDA"),
    };
    let (scales_storage, scales_layout) = row_scales.storage_and_layout();
    let scales_storage = match &*scales_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b fused gemv CUDA: scales storage must be CUDA"),
    };
    let (x_storage, x_layout) = x_2d.storage_and_layout();
    let x_storage = match &*x_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b fused gemv CUDA: x storage must be CUDA"),
    };

    let (blocks_ptr, _blocks_guard) = slice_ptr(
        blocks_storage.as_cuda_slice::<u8>()?,
        blocks_layout.start_offset(),
    );
    let (scales_ptr, _scales_guard) = slice_ptr(
        scales_storage.as_cuda_slice::<f32>()?,
        scales_layout.start_offset(),
    );

    // Autotune dispatch: try the per-shape winning variant first (baked
    // table / ARC_QTIP_TUNE_TABLE / forced); a non-zero return means the
    // variant is inapplicable to this shape and we fall back to the legacy
    // fixed-config kernel, which shares the correctness contract.
    let tune_variant = super::tune::gemv_variant_for_shape(n_rows, in_features)
        .filter(|&v| v != super::tune::QTIP2B_GEMV_VARIANT_LEGACY);

    macro_rules! gemv_dtype {
        ($T:ty, $launch:expr, $launch_tuned:expr) => {{
            let (x_ptr, _x_guard) =
                slice_ptr(x_storage.as_cuda_slice::<$T>()?, x_layout.start_offset());
            let out_buf = dev.alloc_zeros::<$T>(n_rows)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            let mut launched = false;
            if let Some(v) = tune_variant {
                let rc = unsafe {
                    $launch_tuned(
                        v as i32,
                        blocks_ptr as *const _,
                        scales_ptr as *const _,
                        x_ptr as *const _,
                        std::ptr::null::<u32>(), // 2-D path: no gather indices
                        out_ptr as *mut _,
                        n_rows as i32,
                        packed_per_row as i32,
                        num_symbols as i32,
                        1i32, // n_pairs
                        1i32, // num_experts
                        mcg_mult,
                        dev.cuda_stream().cu_stream(),
                    )
                };
                launched = rc == 0;
            }
            if !launched {
                unsafe {
                    $launch(
                        blocks_ptr as *const _,
                        scales_ptr as *const _,
                        x_ptr as *const _,
                        std::ptr::null::<u32>(), // 2-D path: no gather indices
                        out_ptr as *mut _,
                        n_rows as i32,
                        packed_per_row as i32,
                        num_symbols as i32,
                        1i32, // n_pairs
                        1i32, // num_experts
                        mcg_mult,
                        dev.cuda_stream().cu_stream(),
                    );
                }
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }};
    }

    let res = match x_2d.dtype() {
        DType::BF16 => gemv_dtype!(
            bf16,
            ffi::launch_qtip2b_gemv_bf16,
            ffi::launch_qtip2b_gemv_tuned_bf16
        ),
        DType::F16 => gemv_dtype!(
            f16,
            ffi::launch_qtip2b_gemv_f16,
            ffi::launch_qtip2b_gemv_tuned_f16
        ),
        DType::F32 => gemv_dtype!(
            f32,
            ffi::launch_qtip2b_gemv_f32,
            ffi::launch_qtip2b_gemv_tuned_f32
        ),
        other => candle_core::bail!("qtip2b fused gemv CUDA: unsupported x dtype {other:?}"),
    };

    Ok(Tensor::from((Storage::Cuda(res), out_shape)))
}

/// On-device MoE gather + fused decode GEMV for a 3-D stacked qtip2b layer
/// (CUDA-graph-capturable decode dispatch). Returns `[n_pairs, n_rows]`.
pub(crate) fn gather_gemv_2b_cuda(
    blocks: &Tensor,
    row_scales: &Tensor,
    mcg_mult: u32,
    x_rotated: &Tensor,
    indices: &Tensor,
    in_features: usize,
) -> Result<Tensor> {
    let num_experts = blocks.dim(0)?;
    let n_rows = row_scales.dim(1)?;
    let packed_per_row = blocks.dim(2)?;
    let num_symbols = in_features; // V = 1

    if blocks.dtype() != DType::U8 {
        candle_core::bail!("qtip2b gather gemv CUDA: blocks dtype must be U8");
    }
    if row_scales.dtype() != DType::F32 {
        candle_core::bail!("qtip2b gather gemv CUDA: row_scales dtype must be F32");
    }
    if indices.dtype() != DType::U32 {
        candle_core::bail!("qtip2b gather gemv CUDA: indices dtype must be U32");
    }
    if !blocks.layout().is_contiguous() || !row_scales.layout().is_contiguous() {
        candle_core::bail!("qtip2b gather gemv CUDA: blocks/scales must be contiguous");
    }

    let (n_pairs, x_cols) = x_rotated.dims2()?;
    if x_cols != in_features {
        candle_core::bail!(
            "qtip2b gather gemv CUDA: x_rotated cols {x_cols} != in_features {in_features}"
        );
    }
    let idx_pairs = indices.elem_count();
    if idx_pairs != n_pairs {
        candle_core::bail!("qtip2b gather gemv CUDA: indices len {idx_pairs} != n_pairs {n_pairs}");
    }
    // One pair per `grid.y` index; past `maxGridSize[1]` the launch fails
    // silently and the zero-filled output buffer is returned as if valid.
    super::gather_policy::check_gather_gemv_pairs(n_pairs, "qtip2b gather gemv CUDA")?;
    let x_2d = x_rotated.contiguous()?;
    let indices = indices.flatten_all()?.contiguous()?;

    let dev = match blocks.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("qtip2b gather gemv CUDA: blocks must live on CUDA"),
    };
    const OP: &str = "qtip2b gather gemv CUDA";
    ensure_same_cuda_device(
        OP,
        "row_scales",
        row_scales.device(),
        "blocks",
        blocks.device(),
    )?;
    ensure_same_cuda_device(OP, "x_rotated", x_2d.device(), "blocks", blocks.device())?;
    ensure_same_cuda_device(OP, "indices", indices.device(), "blocks", blocks.device())?;

    let out_shape = candle_core::Shape::from_dims(&[n_pairs, n_rows]);

    let (blocks_storage, blocks_layout) = blocks.storage_and_layout();
    let blocks_storage = match &*blocks_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b gather gemv CUDA: blocks storage must be CUDA"),
    };
    let (scales_storage, scales_layout) = row_scales.storage_and_layout();
    let scales_storage = match &*scales_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b gather gemv CUDA: scales storage must be CUDA"),
    };
    let (x_storage, x_layout) = x_2d.storage_and_layout();
    let x_storage = match &*x_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b gather gemv CUDA: x storage must be CUDA"),
    };
    let (idx_storage, idx_layout) = indices.storage_and_layout();
    let idx_storage = match &*idx_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b gather gemv CUDA: indices storage must be CUDA"),
    };

    let (blocks_ptr, _blocks_guard) = slice_ptr(
        blocks_storage.as_cuda_slice::<u8>()?,
        blocks_layout.start_offset(),
    );
    let (scales_ptr, _scales_guard) = slice_ptr(
        scales_storage.as_cuda_slice::<f32>()?,
        scales_layout.start_offset(),
    );
    let (idx_ptr, _idx_guard) = slice_ptr(
        idx_storage.as_cuda_slice::<u32>()?,
        idx_layout.start_offset(),
    );

    let n_out = n_pairs * n_rows;

    // Autotune dispatch — same policy as `fused_gemv_2b_cuda` above.
    let tune_variant = super::tune::gemv_variant_for_shape(n_rows, in_features)
        .filter(|&v| v != super::tune::QTIP2B_GEMV_VARIANT_LEGACY);

    macro_rules! gather_dtype {
        ($T:ty, $launch:expr, $launch_tuned:expr) => {{
            let (x_ptr, _x_guard) =
                slice_ptr(x_storage.as_cuda_slice::<$T>()?, x_layout.start_offset());
            let out_buf = dev.alloc_zeros::<$T>(n_out)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            let mut launched = false;
            if let Some(v) = tune_variant {
                let rc = unsafe {
                    $launch_tuned(
                        v as i32,
                        blocks_ptr as *const _,
                        scales_ptr as *const _,
                        x_ptr as *const _,
                        idx_ptr as *const _,
                        out_ptr as *mut _,
                        n_rows as i32,
                        packed_per_row as i32,
                        num_symbols as i32,
                        n_pairs as i32,
                        num_experts as i32,
                        mcg_mult,
                        dev.cuda_stream().cu_stream(),
                    )
                };
                launched = rc == 0;
            }
            if !launched {
                unsafe {
                    $launch(
                        blocks_ptr as *const _,
                        scales_ptr as *const _,
                        x_ptr as *const _,
                        idx_ptr as *const _,
                        out_ptr as *mut _,
                        n_rows as i32,
                        packed_per_row as i32,
                        num_symbols as i32,
                        n_pairs as i32,
                        num_experts as i32,
                        mcg_mult,
                        dev.cuda_stream().cu_stream(),
                    );
                }
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }};
    }

    let res = match x_2d.dtype() {
        DType::BF16 => gather_dtype!(
            bf16,
            ffi::launch_qtip2b_gemv_bf16,
            ffi::launch_qtip2b_gemv_tuned_bf16
        ),
        DType::F16 => gather_dtype!(
            f16,
            ffi::launch_qtip2b_gemv_f16,
            ffi::launch_qtip2b_gemv_tuned_f16
        ),
        DType::F32 => gather_dtype!(
            f32,
            ffi::launch_qtip2b_gemv_f32,
            ffi::launch_qtip2b_gemv_tuned_f32
        ),
        other => candle_core::bail!("qtip2b gather gemv CUDA: unsupported x dtype {other:?}"),
    };

    Ok(Tensor::from((Storage::Cuda(res), out_shape)))
}

/// Largest beam width `kernels/qtip/qtip2b_beam.cu` can run, read from the
/// kernel itself so the Rust limit can never drift from the CUDA one.
///
/// Returns 0 when the kernels were not compiled in.
pub(crate) fn beam_2b_max_width() -> usize {
    if !ffi::HAVE_QTIP_KERNELS {
        return 0;
    }
    // SAFETY: a pure constant getter with no arguments and no side effects.
    let w = unsafe { ffi::qtip2b_beam_max_width() };
    if w < 0 {
        0
    } else {
        w as usize
    }
}

/// Per-row search scale for the K=2 rung: `max|row| / QTIP2B_SCALE_DIVISOR`,
/// as computed by the CUDA kernel.
///
/// Split out of [`quantize_rows_2b_cuda`] so the parity tests can hand the CPU
/// reference the SAME scale the kernel searched against — the LS refinement at
/// the end of `quantize_rows_2b_cuda` returns a *different* (post-hoc) scale,
/// so the returned scales of that function cannot be used to reproduce its own
/// symbol stream.
pub(crate) fn compute_row_scales_2b_cuda(weight_f32: &Tensor) -> Result<Tensor> {
    if !ffi::HAVE_QTIP_KERNELS {
        candle_core::bail!("qtip2b row-scale CUDA: kernels not compiled in");
    }
    let (n_rows, k_in) = weight_f32.dims2()?;
    if weight_f32.dtype() != DType::F32 {
        candle_core::bail!(
            "qtip2b row-scale CUDA: weight dtype must be F32, got {:?}",
            weight_f32.dtype()
        );
    }
    let dev = match weight_f32.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("qtip2b row-scale CUDA: weight must live on CUDA"),
    };
    let weight_contig = weight_f32.contiguous()?;
    let (w_storage, w_layout) = weight_contig.storage_and_layout();
    let w_storage = match &*w_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b row-scale CUDA: weight must be CUDA storage"),
    };
    let (w_ptr, _w_guard) = slice_ptr(w_storage.as_cuda_slice::<f32>()?, w_layout.start_offset());
    let scales_buf = dev.alloc_zeros::<f32>(n_rows)?;
    let (scales_ptr, scales_guard) = slice_ptr(&scales_buf, 0);
    unsafe {
        ffi::launch_qtip2b_compute_row_scales_f32(
            w_ptr as *const _,
            scales_ptr as *mut _,
            n_rows as i32,
            k_in as i32,
            QTIP2B_SCALE_DIVISOR,
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

/// One-shot qtip2b quantize on GPU. `weight_rotated_f32` must already be F32
/// on CUDA and in the rotated frame. Returns `(packed_blocks, row_scales)`
/// with `packed_blocks: [n_rows, num_symbols / 4]` U8.
///
/// `search` selects the trellis search when `mode == Viterbi`:
/// [`TrellisSearch::Exhaustive`] runs the prefix-grouped DP over all `2^L`
/// states (best quality, ~984 s/layer measured on H200 by wave41-BS);
/// [`TrellisSearch::Beam`] runs `qtip2b_beam.cu` at that width. `Greedy`
/// ignores it — a greedy walk is not a trellis search at all.
///
/// The caller is expected to have put `search` through
/// [`super::cuda_search_plan`] already; the explicit range check below is the
/// backstop that keeps a mis-plumbed width from becoming a silently different
/// bake rather than an error.
pub(crate) fn quantize_rows_2b_cuda(
    weight_rotated_f32: &Tensor,
    mcg_mult: u32,
    mode: QtipMode,
    search: TrellisSearch,
) -> Result<(Tensor, Tensor)> {
    if !ffi::HAVE_QTIP_KERNELS {
        candle_core::bail!("qtip2b quantize CUDA: kernels not compiled in");
    }
    let (n_rows, k_in) = weight_rotated_f32.dims2()?;
    let num_symbols = k_in; // V = 1
    if !num_symbols.is_multiple_of(Q2B_SYMS_PER_BYTE) {
        candle_core::bail!(
            "qtip2b quantize CUDA: num_symbols ({num_symbols}) must be a multiple of 4"
        );
    }
    if weight_rotated_f32.dtype() != DType::F32 {
        candle_core::bail!(
            "qtip2b quantize CUDA: weight dtype must be F32, got {:?}",
            weight_rotated_f32.dtype()
        );
    }
    let dev = match weight_rotated_f32.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("qtip2b quantize CUDA: weight must live on CUDA"),
    };

    let weight_contig = weight_rotated_f32.contiguous()?;

    // Step 1: row scales (max|row| / QTIP2B_SCALE_DIVISOR).
    let row_scales = compute_row_scales_2b_cuda(&weight_contig)?;

    // Step 2: packed output + quantize kernel.
    let packed_per_row = num_symbols / Q2B_SYMS_PER_BYTE;
    let packed_buf = dev.alloc_zeros::<u8>(n_rows * packed_per_row)?;
    let packed_shape = candle_core::Shape::from_dims(&[n_rows, packed_per_row]);

    {
        let (w_storage, w_layout) = weight_contig.storage_and_layout();
        let w_storage = match &*w_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("qtip2b quantize CUDA: weight must be CUDA storage"),
        };
        let (rs_storage, rs_layout) = row_scales.storage_and_layout();
        let rs_storage = match &*rs_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("qtip2b quantize CUDA: row_scales must be CUDA storage"),
        };
        let (w_ptr, _w_guard) =
            slice_ptr(w_storage.as_cuda_slice::<f32>()?, w_layout.start_offset());
        let (rs_ptr, _rs_guard) =
            slice_ptr(rs_storage.as_cuda_slice::<f32>()?, rs_layout.start_offset());
        let (pkd_ptr, pkd_guard) = slice_ptr(&packed_buf, 0);

        match mode {
            QtipMode::Greedy => unsafe {
                ffi::launch_qtip2b_quantize_rows_greedy_f32(
                    w_ptr as *const _,
                    rs_ptr as *const _,
                    pkd_ptr as *mut _,
                    n_rows as i32,
                    k_in as i32,
                    num_symbols as i32,
                    mcg_mult,
                    dev.cuda_stream().cu_stream(),
                );
            },
            QtipMode::Viterbi if !matches!(search, TrellisSearch::Exhaustive) => {
                // wave46-BX beam kernel. The live state set is `width` entries
                // in shared memory, so there is NO cost ping-pong scratch and
                // no per-prefix table at all — only the compacted backtrace,
                // `width * 4` bytes per timestep instead of the exhaustive
                // kernel's 16384.
                let width = match search {
                    TrellisSearch::Beam { width } => width,
                    TrellisSearch::Exhaustive => unreachable!("guarded by the match arm"),
                };
                let max_w = beam_2b_max_width();
                if width == 0 || width > max_w {
                    candle_core::bail!(
                        "qtip2b quantize CUDA: beam width {width} is outside the kernel's \
                         supported range 1..={max_w}. Refusing to substitute a different \
                         width — a bake must never silently change its search."
                    );
                }
                let trace_bytes_per_row = num_symbols * width * 4;
                let mut rows_in_flight = (viterbi_scratch_bytes() / trace_bytes_per_row).max(1);
                if rows_in_flight > n_rows {
                    rows_in_flight = n_rows;
                }

                // Uninit alloc: the kernel writes every trace slot it later
                // reads (the backtrace walk only visits slots the forward pass
                // populated), so zeroing GBs of scratch is wasted bandwidth.
                let trace = unsafe { dev.alloc::<u32>(rows_in_flight * num_symbols * width)? };
                let (tr_ptr, tr_guard) = slice_ptr(&trace, 0);

                let mut row_offset = 0usize;
                while row_offset < n_rows {
                    let this_batch = rows_in_flight.min(n_rows - row_offset);
                    let rc = unsafe {
                        ffi::launch_qtip2b_quantize_rows_beam_f32(
                            w_ptr as *const _,
                            rs_ptr as *const _,
                            pkd_ptr as *mut _,
                            tr_ptr as *mut _,
                            this_batch as i32,
                            k_in as i32,
                            num_symbols as i32,
                            row_offset as i32,
                            width as i32,
                            mcg_mult,
                            dev.cuda_stream().cu_stream(),
                        )
                    };
                    if rc != 0 {
                        candle_core::bail!(
                            "qtip2b quantize CUDA: beam kernel refused width {width} (rc={rc})"
                        );
                    }
                    row_offset += this_batch;
                }

                drop(tr_guard);
                let _ = trace;
            }
            QtipMode::Viterbi => {
                // Scratch budget mirrors the LUT rung: per-row backtrace is
                // num_symbols * 2^(L-K) bytes (prefix-grouped), plus two
                // 2^L-f32 cost ping-pong tables and one 2^(L-K)-f32 prefix
                // table (the K=2 prefix table no longer fits shared memory —
                // see the kernel comment).
                let bt_bytes_per_row = num_symbols * Q2B_PREFIX_COUNT;
                let per_row_bytes = bt_bytes_per_row + 2 * Q2B_NSTATES * 4 + Q2B_PREFIX_COUNT * 4;
                let mut rows_in_flight = (viterbi_scratch_bytes() / per_row_bytes).max(1);
                if rows_in_flight > n_rows {
                    rows_in_flight = n_rows;
                }

                // Uninit alloc (not alloc_zeros): the kernel fully writes
                // cost_a at t=0, prefix_cost in Phase A, cost_b in Phase B,
                // and the backtrace before any read.
                let cost_a = unsafe { dev.alloc::<f32>(rows_in_flight * Q2B_NSTATES)? };
                let cost_b = unsafe { dev.alloc::<f32>(rows_in_flight * Q2B_NSTATES)? };
                let prefix_cost = unsafe { dev.alloc::<f32>(rows_in_flight * Q2B_PREFIX_COUNT)? };
                let backtrace = unsafe { dev.alloc::<u8>(rows_in_flight * bt_bytes_per_row)? };

                let (ca_ptr, ca_guard) = slice_ptr(&cost_a, 0);
                let (cb_ptr, cb_guard) = slice_ptr(&cost_b, 0);
                let (pc_ptr, pc_guard) = slice_ptr(&prefix_cost, 0);
                let (bt_ptr, bt_guard) = slice_ptr(&backtrace, 0);

                let mut row_offset = 0usize;
                while row_offset < n_rows {
                    let this_batch = rows_in_flight.min(n_rows - row_offset);
                    unsafe {
                        ffi::launch_qtip2b_quantize_rows_viterbi_f32(
                            w_ptr as *const _,
                            rs_ptr as *const _,
                            pkd_ptr as *mut _,
                            ca_ptr as *mut _,
                            cb_ptr as *mut _,
                            pc_ptr as *mut _,
                            bt_ptr as *mut _,
                            this_batch as i32,
                            k_in as i32,
                            num_symbols as i32,
                            row_offset as i32,
                            mcg_mult,
                            dev.cuda_stream().cu_stream(),
                        );
                    }
                    row_offset += this_batch;
                }

                drop(ca_guard);
                drop(cb_guard);
                drop(pc_guard);
                drop(bt_guard);
                let _ = (cost_a, cost_b, prefix_cost, backtrace);
            }
        }

        drop(pkd_guard);
    }

    let packed_storage = CudaStorage::wrap_cuda_slice(packed_buf, dev.clone());
    let packed = Tensor::from((Storage::Cuda(packed_storage), packed_shape));

    // Least-squares scale refinement — the GPU mirror of the CPU path's
    // per-row `scale* = ⟨w, c⟩ / ⟨c, c⟩`. Implemented with candle tensor ops
    // (one extra dequant pass at bake time) instead of a bespoke kernel:
    //   ŵ = s₀·c  ⇒  s* = s₀·⟨w, ŵ⟩ / ⟨ŵ, ŵ⟩.
    // Rows with a non-positive numerator (pathological: refined scale would
    // flip every sign) or zero denominator keep the s₀ heuristic, matching
    // the CPU guard exactly.
    let row_scales = {
        let w_hat = dequantize_2b_cuda(&packed, &row_scales, mcg_mult, k_in, DType::F32)?;
        let num = (&weight_contig * &w_hat)?.sum(1)?; // [n_rows]
        let den = (&w_hat * &w_hat)?.sum(1)?; // [n_rows]
        let refined = ((&num / &den)? * &row_scales)?;
        let mask = (num.gt(0f64)? * den.gt(0f64)?)?;
        mask.where_cond(&refined, &row_scales)?
    };

    Ok((packed, row_scales))
}

/// Trellis grouped GEMM over expert-sorted (token, slot) pairs — the
/// batched-MoE prefill path (Arc Stage 4). Routes ON-DEVICE (histogram +
/// scans/tile-map + grouped scatter, zero host syncs), then runs the
/// persistent tensor-core kernel: cp.async-staged packed trellis tiles,
/// in-register 3INST decode into BF16/F16 fragments, mma.sync m16n8k16.
///
/// Shapes:
///   * `blocks`     : `[E, n_rows, packed_per_row]` U8 (3-D stacked experts)
///   * `row_scales` : `[E, n_rows]` F32
///   * `x_rotated`  : `[n_pairs, in_features]` BF16/F16 (ALREADY rotated)
///   * `indices`    : `[n_pairs]` U32 (expert id per pair)
/// Returns `[n_pairs, n_rows]` (same dtype as `x_rotated`). Pairs whose
/// router id is out of range are dropped by routing; their output rows are
/// zero (the kernel contract requires — and this wrapper provides — a
/// zero-initialized output buffer).
///
/// Requires `in_features % GROUPED_TILE_K == 0` (the k-chunked cp.async
/// pipeline; every production MoE hidden/intermediate size qualifies).
/// Callers with other shapes use the per-pair gather GEMV or CPU paths.
pub(crate) fn grouped_gemm_2b_cuda(
    blocks: &Tensor,
    row_scales: &Tensor,
    mcg_mult: u32,
    x_rotated: &Tensor,
    indices: &Tensor,
    in_features: usize,
) -> Result<Tensor> {
    use super::grouped::{grouped_max_m_tiles, GROUPED_TILE_K, GROUPED_TILE_M};

    if !ffi::HAVE_QTIP_KERNELS {
        candle_core::bail!("qtip2b grouped gemm CUDA: kernels not compiled in");
    }
    let num_experts = blocks.dim(0)?;
    let n_rows = row_scales.dim(1)?;
    let packed_per_row = blocks.dim(2)?;
    let num_symbols = in_features; // V = 1

    if blocks.dtype() != DType::U8 {
        candle_core::bail!("qtip2b grouped gemm CUDA: blocks dtype must be U8");
    }
    if row_scales.dtype() != DType::F32 {
        candle_core::bail!("qtip2b grouped gemm CUDA: row_scales dtype must be F32");
    }
    if indices.dtype() != DType::U32 {
        candle_core::bail!("qtip2b grouped gemm CUDA: indices dtype must be U32");
    }
    if !blocks.layout().is_contiguous() || !row_scales.layout().is_contiguous() {
        candle_core::bail!("qtip2b grouped gemm CUDA: blocks/scales must be contiguous");
    }
    if !num_symbols.is_multiple_of(GROUPED_TILE_K) {
        candle_core::bail!(
            "qtip2b grouped gemm CUDA: in_features {num_symbols} must be a multiple of {GROUPED_TILE_K}"
        );
    }

    let (n_pairs, x_cols) = x_rotated.dims2()?;
    if x_cols != in_features {
        candle_core::bail!(
            "qtip2b grouped gemm CUDA: x_rotated cols {x_cols} != in_features {in_features}"
        );
    }
    let idx_pairs = indices.elem_count();
    if idx_pairs != n_pairs {
        candle_core::bail!(
            "qtip2b grouped gemm CUDA: indices len {idx_pairs} != n_pairs {n_pairs}"
        );
    }
    let out_dtype = x_rotated.dtype();
    if !matches!(out_dtype, DType::BF16 | DType::F16) {
        candle_core::bail!(
            "qtip2b grouped gemm CUDA: 16-bit activations required (BF16/F16), got {out_dtype:?}"
        );
    }
    if n_pairs == 0 {
        return Tensor::zeros((0usize, n_rows), out_dtype, blocks.device());
    }
    // The kernel gathers activation rows and packed rows with 16-byte
    // cp.async transfers; the freshly-allocated contiguous buffers below
    // plus the `% GROUPED_TILE_K` gate satisfy its alignment contract.
    let x_2d = x_rotated.contiguous()?;
    let indices = indices.flatten_all()?.contiguous()?;

    let dev = match blocks.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("qtip2b grouped gemm CUDA: blocks must live on CUDA"),
    };
    // The grouped GEMM is the TD-MoE hot path; a cross-ordinal operand here is
    // exactly the silent-corruption case the ordinal guard exists for.
    const OP: &str = "qtip2b grouped gemm CUDA";
    ensure_same_cuda_device(
        OP,
        "row_scales",
        row_scales.device(),
        "blocks",
        blocks.device(),
    )?;
    ensure_same_cuda_device(OP, "x_rotated", x_2d.device(), "blocks", blocks.device())?;
    ensure_same_cuda_device(OP, "indices", indices.device(), "blocks", blocks.device())?;

    let max_m_tiles = grouped_max_m_tiles(n_pairs, num_experts);

    // Routing scratch. `counts` and `cursors` MUST be zeroed (histogram /
    // scatter accumulate into them); the rest is written before being read.
    // TUNING: fold these into one arena reused across layers and steps —
    // eight small allocs per MoE matmul is launch-latency noise but not
    // free.
    let counts = dev.alloc_zeros::<u32>(num_experts)?;
    let offsets = dev.alloc_zeros::<u32>(num_experts + 1)?;
    let cursors = dev.alloc_zeros::<u32>(num_experts)?;
    let tile_prefix = dev.alloc_zeros::<u32>(num_experts)?;
    let tile_expert = dev.alloc_zeros::<u32>(max_m_tiles)?;
    let tile_row_start = dev.alloc_zeros::<u32>(max_m_tiles)?;
    let num_tiles = dev.alloc_zeros::<u32>(1)?;
    let sorted_pairs = dev.alloc_zeros::<u32>(n_pairs)?;

    let out_shape = candle_core::Shape::from_dims(&[n_pairs, n_rows]);

    let (blocks_storage, blocks_layout) = blocks.storage_and_layout();
    let blocks_storage = match &*blocks_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b grouped gemm CUDA: blocks storage must be CUDA"),
    };
    let (scales_storage, scales_layout) = row_scales.storage_and_layout();
    let scales_storage = match &*scales_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b grouped gemm CUDA: scales storage must be CUDA"),
    };
    let (x_storage, x_layout) = x_2d.storage_and_layout();
    let x_storage = match &*x_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b grouped gemm CUDA: x storage must be CUDA"),
    };
    let (idx_storage, idx_layout) = indices.storage_and_layout();
    let idx_storage = match &*idx_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("qtip2b grouped gemm CUDA: indices storage must be CUDA"),
    };

    let (blocks_ptr, _blocks_guard) = slice_ptr(
        blocks_storage.as_cuda_slice::<u8>()?,
        blocks_layout.start_offset(),
    );
    let (scales_ptr, _scales_guard) = slice_ptr(
        scales_storage.as_cuda_slice::<f32>()?,
        scales_layout.start_offset(),
    );
    let (idx_ptr, _idx_guard) = slice_ptr(
        idx_storage.as_cuda_slice::<u32>()?,
        idx_layout.start_offset(),
    );

    let (counts_ptr, _counts_guard) = slice_ptr(&counts, 0);
    let (offsets_ptr, _offsets_guard) = slice_ptr(&offsets, 0);
    let (cursors_ptr, _cursors_guard) = slice_ptr(&cursors, 0);
    let (tp_ptr, _tp_guard) = slice_ptr(&tile_prefix, 0);
    let (te_ptr, _te_guard) = slice_ptr(&tile_expert, 0);
    let (trs_ptr, _trs_guard) = slice_ptr(&tile_row_start, 0);
    let (nt_ptr, _nt_guard) = slice_ptr(&num_tiles, 0);
    let (sp_ptr, _sp_guard) = slice_ptr(&sorted_pairs, 0);

    // Route: histogram -> scans + ragged tile map -> grouped scatter. All
    // on the stream; nothing comes back to the host.
    unsafe {
        ffi::launch_qtip2b_moe_route(
            idx_ptr as *const _,
            counts_ptr as *mut _,
            offsets_ptr as *mut _,
            cursors_ptr as *mut _,
            tp_ptr as *mut _,
            te_ptr as *mut _,
            trs_ptr as *mut _,
            nt_ptr as *mut _,
            sp_ptr as *mut _,
            n_pairs as i32,
            num_experts as i32,
            GROUPED_TILE_M as i32,
            dev.cuda_stream().cu_stream(),
        );
    }

    // Kernel variant is resolved PER LAUNCH (never memoized): a mode switch
    // that silently fails to take is what voided the last grouped microbench.
    let variant = super::grouped::grouped_variant();

    macro_rules! grouped_dtype {
        ($T:ty, $launch:expr) => {{
            let (x_ptr, _x_guard) =
                slice_ptr(x_storage.as_cuda_slice::<$T>()?, x_layout.start_offset());
            // alloc_zeros is part of the kernel contract: rows dropped by
            // routing (out-of-range router ids) are never written.
            let out_buf = dev.alloc_zeros::<$T>(n_pairs * n_rows)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            super::grouped::note_grouped_launch(variant);
            unsafe {
                $launch(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    x_ptr as *const _,
                    sp_ptr as *const _,
                    te_ptr as *const _,
                    trs_ptr as *const _,
                    offsets_ptr as *const _,
                    nt_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    num_symbols as i32,
                    max_m_tiles as i32,
                    mcg_mult,
                    variant,
                    dev.cuda_stream().cu_stream(),
                );
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }};
    }

    let res = match out_dtype {
        DType::BF16 => grouped_dtype!(bf16, ffi::launch_qtip2b_grouped_gemm_bf16),
        DType::F16 => grouped_dtype!(f16, ffi::launch_qtip2b_grouped_gemm_f16),
        other => candle_core::bail!("qtip2b grouped gemm CUDA: unsupported x dtype {other:?}"),
    };

    Ok(Tensor::from((Storage::Cuda(res), out_shape)))
}

/// LUT-rung (K=4 / V=2) trellis grouped GEMM over expert-sorted (token, slot)
/// pairs — the batched-prefill path for `QtipLayer`, i.e. for what
/// `qtip2-*.uqff` actually ships.
///
/// This is the direct twin of [`grouped_gemm_2b_cuda`]: same routing kernels
/// (they are codebook-agnostic), same tile geometry, same zeroed-output
/// contract. It exists because the LUT rung's trellis state is *also* a
/// 16-bit sliding window over the packed stream — `state = ((state << 4) |
/// sym) & 0xFFFF` means `state(t)` is just the last four nibbles — so its
/// weights are random-access decodable and a grouped GEMM is reachable. The
/// rung previously had only a per-(token,slot) fused GEMV with no dedup and,
/// above a token threshold, a host-synced per-expert dequantize-materialize
/// loop; neither amortizes an expert's weight bytes across the tokens routed
/// to it.
///
/// `codebook` picks the reproduction values: `Gaussian` gathers from the
/// stored `[2^16, 2]` table, `Mcg` decodes in registers. Both are compiled as
/// separate kernel specializations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grouped_gemm_lut_cuda(
    blocks: &Tensor,
    row_scales: &Tensor,
    lut: &Tensor,
    x_rotated: &Tensor,
    indices: &Tensor,
    in_features: usize,
    codebook: QtipCodebook,
) -> Result<Tensor> {
    use super::grouped::{grouped_max_m_tiles, GROUPED_TILE_K, GROUPED_TILE_M};

    if !ffi::HAVE_QTIP_KERNELS {
        candle_core::bail!("QTIP lut grouped gemm CUDA: kernels not compiled in");
    }
    let cb_mult = codebook.cuda_mult();
    let num_experts = blocks.dim(0)?;
    let n_rows = row_scales.dim(1)?;
    let packed_per_row = blocks.dim(2)?;

    if blocks.dtype() != DType::U8 {
        candle_core::bail!("QTIP lut grouped gemm CUDA: blocks dtype must be U8");
    }
    if row_scales.dtype() != DType::F32 {
        candle_core::bail!("QTIP lut grouped gemm CUDA: row_scales dtype must be F32");
    }
    if lut.dtype() != DType::F32 {
        candle_core::bail!("QTIP lut grouped gemm CUDA: lut dtype must be F32");
    }
    if indices.dtype() != DType::U32 {
        candle_core::bail!("QTIP lut grouped gemm CUDA: indices dtype must be U32");
    }
    if !blocks.layout().is_contiguous()
        || !row_scales.layout().is_contiguous()
        || !lut.layout().is_contiguous()
    {
        candle_core::bail!("QTIP lut grouped gemm CUDA: blocks/scales/lut must be contiguous");
    }
    // The kernel's k-chunk is GROUPED_TILE_K *weights* (16 packed bytes/row).
    if !in_features.is_multiple_of(GROUPED_TILE_K) {
        candle_core::bail!(
            "QTIP lut grouped gemm CUDA: in_features {in_features} must be a multiple of {GROUPED_TILE_K}"
        );
    }

    let (n_pairs, x_cols) = x_rotated.dims2()?;
    if x_cols != in_features {
        candle_core::bail!(
            "QTIP lut grouped gemm CUDA: x_rotated cols {x_cols} != in_features {in_features}"
        );
    }
    let idx_pairs = indices.elem_count();
    if idx_pairs != n_pairs {
        candle_core::bail!(
            "QTIP lut grouped gemm CUDA: indices len {idx_pairs} != n_pairs {n_pairs}"
        );
    }
    let out_dtype = x_rotated.dtype();
    if !matches!(out_dtype, DType::BF16 | DType::F16) {
        candle_core::bail!(
            "QTIP lut grouped gemm CUDA: 16-bit activations required (BF16/F16), got {out_dtype:?}"
        );
    }
    if n_pairs == 0 {
        return Tensor::zeros((0usize, n_rows), out_dtype, blocks.device());
    }

    let x_2d = x_rotated.contiguous()?;
    let indices = indices.flatten_all()?.contiguous()?;

    let dev = match blocks.device() {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => candle_core::bail!("QTIP lut grouped gemm CUDA: blocks must live on CUDA"),
    };
    if !matches!(x_2d.device(), candle_core::Device::Cuda(_))
        || !matches!(indices.device(), candle_core::Device::Cuda(_))
    {
        candle_core::bail!("QTIP lut grouped gemm CUDA: x_rotated and indices must live on CUDA");
    }

    let max_m_tiles = grouped_max_m_tiles(n_pairs, num_experts);

    // Routing scratch. `counts`/`cursors` MUST be zeroed.
    let counts = dev.alloc_zeros::<u32>(num_experts)?;
    let offsets = dev.alloc_zeros::<u32>(num_experts + 1)?;
    let cursors = dev.alloc_zeros::<u32>(num_experts)?;
    let tile_prefix = dev.alloc_zeros::<u32>(num_experts)?;
    let tile_expert = dev.alloc_zeros::<u32>(max_m_tiles)?;
    let tile_row_start = dev.alloc_zeros::<u32>(max_m_tiles)?;
    let num_tiles = dev.alloc_zeros::<u32>(1)?;
    let sorted_pairs = dev.alloc_zeros::<u32>(n_pairs)?;

    let out_shape = candle_core::Shape::from_dims(&[n_pairs, n_rows]);

    let (blocks_storage, blocks_layout) = blocks.storage_and_layout();
    let blocks_storage = match &*blocks_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP lut grouped gemm CUDA: blocks storage must be CUDA"),
    };
    let (scales_storage, scales_layout) = row_scales.storage_and_layout();
    let scales_storage = match &*scales_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP lut grouped gemm CUDA: scales storage must be CUDA"),
    };
    let (lut_storage, lut_layout) = lut.storage_and_layout();
    let lut_storage = match &*lut_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP lut grouped gemm CUDA: lut storage must be CUDA"),
    };
    let (x_storage, x_layout) = x_2d.storage_and_layout();
    let x_storage = match &*x_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP lut grouped gemm CUDA: x storage must be CUDA"),
    };
    let (idx_storage, idx_layout) = indices.storage_and_layout();
    let idx_storage = match &*idx_storage {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("QTIP lut grouped gemm CUDA: indices storage must be CUDA"),
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
    let (idx_ptr, _idx_guard) = slice_ptr(
        idx_storage.as_cuda_slice::<u32>()?,
        idx_layout.start_offset(),
    );

    let (counts_ptr, _counts_guard) = slice_ptr(&counts, 0);
    let (offsets_ptr, _offsets_guard) = slice_ptr(&offsets, 0);
    let (cursors_ptr, _cursors_guard) = slice_ptr(&cursors, 0);
    let (tp_ptr, _tp_guard) = slice_ptr(&tile_prefix, 0);
    let (te_ptr, _te_guard) = slice_ptr(&tile_expert, 0);
    let (trs_ptr, _trs_guard) = slice_ptr(&tile_row_start, 0);
    let (nt_ptr, _nt_guard) = slice_ptr(&num_tiles, 0);
    let (sp_ptr, _sp_guard) = slice_ptr(&sorted_pairs, 0);

    // Routing is shared with the qtip2b rung: it depends only on indices.
    unsafe {
        ffi::launch_qtip2b_moe_route(
            idx_ptr as *const _,
            counts_ptr as *mut _,
            offsets_ptr as *mut _,
            cursors_ptr as *mut _,
            tp_ptr as *mut _,
            te_ptr as *mut _,
            trs_ptr as *mut _,
            nt_ptr as *mut _,
            sp_ptr as *mut _,
            n_pairs as i32,
            num_experts as i32,
            GROUPED_TILE_M as i32,
            dev.cuda_stream().cu_stream(),
        );
    }

    macro_rules! grouped_lut_dtype {
        ($T:ty, $launch:expr) => {{
            let (x_ptr, _x_guard) =
                slice_ptr(x_storage.as_cuda_slice::<$T>()?, x_layout.start_offset());
            let out_buf = dev.alloc_zeros::<$T>(n_pairs * n_rows)?;
            let (out_ptr, out_guard) = slice_ptr(&out_buf, 0);
            unsafe {
                $launch(
                    blocks_ptr as *const _,
                    scales_ptr as *const _,
                    lut_ptr as *const _,
                    x_ptr as *const _,
                    sp_ptr as *const _,
                    te_ptr as *const _,
                    trs_ptr as *const _,
                    offsets_ptr as *const _,
                    nt_ptr as *const _,
                    out_ptr as *mut _,
                    n_rows as i32,
                    packed_per_row as i32,
                    in_features as i32,
                    max_m_tiles as i32,
                    cb_mult,
                    dev.cuda_stream().cu_stream(),
                );
            }
            drop(out_guard);
            CudaStorage::wrap_cuda_slice(out_buf, dev.clone())
        }};
    }

    let res = match out_dtype {
        DType::BF16 => grouped_lut_dtype!(bf16, ffi::launch_qtip_lut_grouped_gemm_bf16),
        DType::F16 => grouped_lut_dtype!(f16, ffi::launch_qtip_lut_grouped_gemm_f16),
        other => candle_core::bail!("QTIP lut grouped gemm CUDA: unsupported x dtype {other:?}"),
    };

    Ok(Tensor::from((Storage::Cuda(res), out_shape)))
}
