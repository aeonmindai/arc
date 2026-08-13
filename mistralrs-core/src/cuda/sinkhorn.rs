/// Fused Sinkhorn normalization for V4 mHC.
///
/// Replaces the ~123-launch candle op chain in `sinkhorn_normalize`
/// (models/dsv4_mhc.rs) with a single CUDA kernel. Input is `[N, hc, hc]` F32;
/// output is the same shape. Matches the candle reference math exactly
/// (row softmax + eps, then `iters` row/col normalization passes).
#[cfg(feature = "cuda")]
pub fn sinkhorn_normalize_cuda(
    comb: &candle_core::Tensor,
    iters: usize,
    eps: f64,
) -> candle_core::Result<candle_core::Tensor> {
    use candle_core as candle;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let comb = comb.contiguous()?;
    let (n, hc, hc2) = comb.dims3()?;
    if hc != hc2 {
        candle::bail!("sinkhorn_normalize_cuda: last two dims must be square, got [{n}, {hc}, {hc2}]");
    }
    if hc > 16 {
        candle::bail!("sinkhorn_normalize_cuda: hc={hc} exceeds SINKHORN_MAX_HC=16");
    }
    if comb.dtype() != candle_core::DType::F32 {
        candle::bail!("sinkhorn_normalize_cuda: input must be F32, got {:?}", comb.dtype());
    }

    let dev = comb.device().as_cuda_device()?;

    let (in_ptr, in_s) = {
        let (s, l) = comb.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("sinkhorn_normalize_cuda: input must be on CUDA"),
        };
        let ptr = s.slice(l.start_offset()..).device_ptr(s.stream()).0 as *const std::ffi::c_void;
        (ptr, ())
    };
    let _ = in_s;

    let out_buf = unsafe { dev.alloc::<f32>(n * hc * hc) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    #[allow(clippy::cast_possible_truncation)]
    unsafe {
        crate::cuda::ffi::sinkhorn_normalize_f32(
            in_ptr,
            {
                let p = out_buf.device_ptr(out_buf.stream()).0 as *mut std::ffi::c_void;
                p
            },
            n as i32,
            hc as i32,
            iters as i32,
            eps as f32,
            stream,
        );
    }

    let out_storage = candle::CudaStorage::wrap_cuda_slice(out_buf, dev.clone());
    Ok(candle_core::Tensor::from((
        candle::Storage::Cuda(out_storage),
        (n, hc, hc),
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn sinkhorn_normalize_cuda(
    _comb: &candle_core::Tensor,
    _iters: usize,
    _eps: f64,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("sinkhorn_normalize_cuda requires the cuda feature")
}
