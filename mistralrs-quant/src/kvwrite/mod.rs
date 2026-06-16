#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::{DType, Result, Storage, Tensor};
    use half::{bf16, f16};
    use std::ffi::c_int;

    use crate::utils::slice_ptr;

    fn write_kv_<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        all_data: &Tensor,
        src: &Tensor,
        positions: &Tensor,
        dtype_id: u32,
    ) -> Result<()> {
        let dev = match all_data.device() {
            candle_core::Device::Cuda(d) => d.clone(),
            _ => candle_core::bail!("kv-write: all_data must live on CUDA"),
        };

        // all_data: [B, H, capacity, D]; src: [B, H, 1, D].
        let (b, h, capacity, d) = all_data.dims4()?;
        let (sb, sh, sseq, sd) = src.dims4()?;
        if (sb, sh, sd) != (b, h, d) || sseq != 1 {
            candle_core::bail!(
                "kv-write: src {:?} incompatible with cache {:?} (need [B,H,1,D])",
                src.dims(),
                all_data.dims()
            );
        }
        if positions.dims1()? != b {
            candle_core::bail!(
                "kv-write: positions len {} != batch {b}",
                positions.dims1()?
            );
        }
        if positions.dtype() != DType::U32 {
            candle_core::bail!("kv-write: positions must be U32 (got {:?})", positions.dtype());
        }
        if !all_data.is_contiguous() || !src.is_contiguous() {
            candle_core::bail!("kv-write: all_data and src must be contiguous");
        }

        let (ad_s, ad_l) = all_data.storage_and_layout();
        let ad_s = match &*ad_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("kv-write: all_data must be CUDA storage"),
        };
        let (src_s, src_l) = src.storage_and_layout();
        let src_s = match &*src_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("kv-write: src must be CUDA storage"),
        };
        let (pos_s, pos_l) = positions.storage_and_layout();
        let pos_s = match &*pos_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("kv-write: positions must be CUDA storage"),
        };

        let (ad_ptr, _ad_guard) =
            slice_ptr(ad_s.as_cuda_slice::<T>()?, ad_l.start_offset());
        let (src_ptr, _src_guard) =
            slice_ptr(src_s.as_cuda_slice::<T>()?, src_l.start_offset());
        let (pos_ptr, _pos_guard) =
            slice_ptr(pos_s.as_cuda_slice::<u32>()?, pos_l.start_offset());

        unsafe {
            super::ffi::kvcache_write(
                ad_ptr as *mut core::ffi::c_void,
                src_ptr as *const core::ffi::c_void,
                pos_ptr as *const c_int,
                b as c_int,
                h as c_int,
                capacity as c_int,
                d as c_int,
                dev.cuda_stream().cu_stream(),
                dtype_id,
            );
        }
        Ok(())
    }

    /// Write `src` ([B,H,1,D]) into the pre-grown KV cache `all_data`
    /// ([B,H,capacity,D]) at the per-row sequence slot held in the device
    /// tensor `positions` ([B] U32).
    ///
    /// The position is read on-device so a captured CUDA graph can be replayed
    /// for successive decode tokens: update `positions` (a tiny H2D copy)
    /// before each replay and the recorded kernel writes the new slot. The
    /// launch uses candle's stream, so it is recorded into the graph captured
    /// on that stream.
    pub fn write_kv_inplace(all_data: &Tensor, src: &Tensor, positions: &Tensor) -> Result<()> {
        let dtype_id = match all_data.dtype() {
            DType::F16 => 0u32,
            DType::BF16 => 1u32,
            DType::F32 => 2u32,
            other => candle_core::bail!("kv-write: unsupported cache dtype {other:?}"),
        };
        if src.dtype() != all_data.dtype() {
            candle_core::bail!(
                "kv-write: src dtype {:?} != cache dtype {:?}",
                src.dtype(),
                all_data.dtype()
            );
        }
        match all_data.dtype() {
            DType::F16 => write_kv_::<f16>(all_data, src, positions, dtype_id),
            DType::BF16 => write_kv_::<bf16>(all_data, src, positions, dtype_id),
            DType::F32 => write_kv_::<f32>(all_data, src, positions, dtype_id),
            other => candle_core::bail!("kv-write: unsupported cache dtype {other:?}"),
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::*;

/// Write `src` into the pre-grown KV cache `all_data` at the device-held
/// `positions` slot. CUDA-only (graph-capturable decode path).
#[cfg(not(feature = "cuda"))]
pub fn write_kv_inplace(
    _all_data: &candle_core::Tensor,
    _src: &candle_core::Tensor,
    _positions: &candle_core::Tensor,
) -> candle_core::Result<()> {
    candle_core::bail!("kv-write (write_kv_inplace) is only supported for cuda");
}
