//! Parent system: ArcInfer / ArcAttention
//!
//! Rust side of the fused per-head Q RMSNorm (`cuda/qnorm.cu`).
//!
//! Structural port of vLLM `csrc/layernorm_kernels.cu:108-181` `rms_norm_kernel`
//! and SGLang `fused_add_rmsnorm.cuh:57` — one launch, in-kernel reduction, no
//! intermediate tensors. Apache-2.0; see `qnorm.cu` for the notice.
//!
//! Replaces SEVEN candle launches per attention layer (43x per decode token):
//! `sqr` -> `fast_sum` -> `affine(1/n, 0)` -> `affine(1, eps)` -> `recip` ->
//! `sqrt` -> `broadcast_mul`, each with its own device allocation.
//!
//! Bit-identical to that chain, including its BF16 accumulator — see the
//! contract in `qnorm.cu` and the on-GPU A/B with negative control below.

use candle_core::{DType, Result, Tensor};

/// `ARC_NO_FUSED_QNORM=1` forces the old candle chain (the A/B switch).
pub(crate) fn fused_qnorm_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("ARC_NO_FUSED_QNORM").is_some())
}

/// `ARC_QNORM_F32_ACC=1` switches the sum of squares to a float accumulator.
///
/// This is NOT bit-identical and is OFF by default. It exists because the
/// chain being replaced accumulates 512 squared values in BF16 (8-bit
/// mantissa), which carries roughly `sqrt(512) * 2^-8` ~ 9% relative error in
/// the norm. That is a pre-existing quality defect, not one this port
/// introduces; the flag makes it measurable in a future perplexity A/B without
/// changing the default.
fn f32_accumulator() -> bool {
    static F32ACC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F32ACC.get_or_init(|| std::env::var_os("ARC_QNORM_F32_ACC").is_some())
}

pub(crate) static QNORM_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[allow(dead_code)]
fn qnorm_ineligible(reason: &'static str) -> Option<Tensor> {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::warn!(
            "fused Q RMSNorm ineligible ({reason}); falling back to the candle op chain. \
             Any perf claim for this port is void while this is the steady state."
        );
    });
    None
}

/// Per-row RMS normalisation over the last dimension: `x * sqrt(1/(mean(x^2) + eps))`.
///
/// Returns `Ok(None)` when the call is not eligible, so callers fall back.
#[cfg(feature = "cuda")]
pub fn qnorm_rms_cuda(x: &Tensor, eps: f64) -> Result<Option<Tensor>> {
    use candle_core as candle;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    if !x.device().is_cuda() {
        return Ok(qnorm_ineligible("not a CUDA tensor"));
    }
    if x.dtype() != DType::BF16 {
        return Ok(qnorm_ineligible("dtype is not BF16"));
    }
    if !x.is_contiguous() {
        return Ok(qnorm_ineligible("non-contiguous input"));
    }
    let dims = x.dims();
    let Some(&cols) = dims.last() else {
        return Ok(qnorm_ineligible("scalar input"));
    };
    // The kernel reduces a row inside ONE block, exactly as candle's
    // `fast_sum` does for these shapes. Above 1024 candle switches to a
    // multi-pass order this kernel does not replicate, so refuse rather than
    // silently diverge.
    if cols == 0 || cols > 1024 {
        return Ok(qnorm_ineligible("head_dim outside the single-block regime"));
    }
    let n = x.elem_count();
    let rows = n / cols;
    if rows == 0 || rows > u32::MAX as usize {
        return Ok(qnorm_ineligible("row count out of range"));
    }

    // candle's `affine` down-converts BOTH constants to the tensor dtype on the
    // HOST (`TYPENAME mul, TYPENAME add`), so do the same conversion here and
    // pass raw bit patterns. Converting in-kernel from f32 would not
    // necessarily land on the same bf16.
    let inv_n_bits = half::bf16::from_f64(1f64 / cols as f64).to_bits();
    let eps_bits = half::bf16::from_f64(eps).to_bits();

    let dev = x.device().as_cuda_device()?;
    let shape = x.shape().clone();
    let stream = dev.cuda_stream().cu_stream() as i64;

    let x_ptr = {
        let (s, l) = x.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<half::bf16>()?,
            _ => return Ok(qnorm_ineligible("storage is not CUDA")),
        };
        let p = s.slice(l.start_offset()..).device_ptr(s.stream()).0 as *const std::ffi::c_void;
        p
    };
    let out_buf = unsafe { dev.alloc::<half::bf16>(n) }?;
    let o_ptr = {
        let p = out_buf.device_ptr(out_buf.stream()).0 as *mut std::ffi::c_void;
        p
    };
    let rc = unsafe {
        crate::cuda::ffi::qnorm_rms_bf16(
            x_ptr,
            o_ptr,
            rows as u32,
            cols as u32,
            inv_n_bits,
            eps_bits,
            i32::from(f32_accumulator()),
            stream,
        )
    };
    if rc != 0 {
        return Ok(qnorm_ineligible("kernel rejected the shape"));
    }

    let prior = QNORM_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if prior == 0 {
        tracing::info!(
            "fused Q RMSNorm ENGAGED: rows={rows} cols={cols} (vLLM rms_norm_kernel port)"
        );
    }

    let st = candle::CudaStorage::wrap_cuda_slice(out_buf, dev.clone());
    Ok(Some(Tensor::from((candle::Storage::Cuda(st), shape))))
}

#[cfg(not(feature = "cuda"))]
pub fn qnorm_rms_cuda(_x: &Tensor, _eps: f64) -> Result<Option<Tensor>> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On-GPU A/B: the fused kernel must be BIT-IDENTICAL to the seven-kernel
    /// candle chain, including its BF16 accumulator and `fast_sum`'s pairwise
    /// reduction order.
    ///
    /// Negative control at the bottom: one input element perturbed by a single
    /// BF16 ULP must make the same comparison report a difference.
    #[cfg(feature = "cuda")]
    #[test]
    fn fused_qnorm_is_bit_identical_to_candle_chain() -> Result<()> {
        use candle_core::Device;

        let dev = Device::new_cuda(0).expect(
            "ENVIRONMENT: --features cuda is on but CUDA device 0 is unavailable. This test \
             proves nothing without a GPU; it must not pass silently.",
        );
        let eps = 1e-6f64;
        let mut total = 0usize;
        // V4's real shape is [1, 64 heads, 1, 512]; the 96-col case forces the
        // non-power-of-two `cols` path where threads past `cols` pad with the
        // additive identity.
        for (rows, cols) in [(64usize, 512usize), (7usize, 96usize)] {
            let n = rows * cols;
            let v: Vec<f32> = (0..n)
                .map(|i| {
                    let t = (i as f32) * 0.011 + 0.3;
                    // wide dynamic range so BF16 accumulation order actually matters
                    (t.sin() * 3.0).tanh() * (1.0 + 20.0 * ((t * 0.7).cos()).abs())
                })
                .collect();
            let x = Tensor::from_vec(v, (rows, cols), &dev)?.to_dtype(DType::BF16)?;

            let reference = candle_chain(&x, eps)?;
            let fused = qnorm_rms_cuda(&x, eps)?
                .expect("fused path must be eligible for contiguous BF16 CUDA tensors");
            let (n_diff, first) = bitwise_diff(&reference, &fused)?;
            assert_eq!(
                n_diff, 0,
                "rows={rows} cols={cols}: fused Q RMSNorm is not bit-identical to the candle \
                 chain: {n_diff} mismatching elements, first at {first:?}"
            );

            // ---- NEGATIVE CONTROL ----
            let mut vals: Vec<half::bf16> = x.flatten_all()?.to_vec1::<half::bf16>()?;
            let before = vals[3];
            vals[3] = half::bf16::from_bits(before.to_bits().wrapping_add(1));
            assert_ne!(
                before.to_bits(),
                vals[3].to_bits(),
                "negative control setup failed: the 1-ULP perturbation was a no-op"
            );
            let x_pert = Tensor::from_vec(vals, (rows, cols), &dev)?;
            let fused_pert = qnorm_rms_cuda(&x_pert, eps)?.unwrap();
            let (n_ctrl, _) = bitwise_diff(&reference, &fused_pert)?;
            assert!(
                n_ctrl > 0,
                "rows={rows} cols={cols}: NEGATIVE CONTROL FAILED — a 1-ULP perturbation \
                 produced no reported difference, so the check above proves nothing"
            );
            println!(
                "qnorm A/B rows={rows} cols={cols}: {n} elements bit-identical; \
                 negative control reported {n_ctrl} differing elements"
            );
            total += n;
        }
        assert_eq!(
            total,
            64 * 512 + 7 * 96,
            "the A/B loop did not run over both shapes; nothing was proved"
        );
        Ok(())
    }

    /// Exactly the chain `deepseek4.rs` ran before the port.
    #[cfg(feature = "cuda")]
    fn candle_chain(x: &Tensor, eps: f64) -> Result<Tensor> {
        let inv_rms = x
            .sqr()?
            .mean_keepdim(candle_core::D::Minus1)?
            .affine(1.0, eps)?
            .recip()?
            .sqrt()?;
        x.broadcast_mul(&inv_rms)
    }

    #[cfg(feature = "cuda")]
    fn bitwise_diff(a: &Tensor, b: &Tensor) -> Result<(usize, Option<usize>)> {
        assert_eq!(a.dtype(), DType::BF16);
        assert_eq!(b.dtype(), DType::BF16);
        let ab: Vec<u16> = a
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let bb: Vec<u16> = b
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let mut n = 0usize;
        let mut first = None;
        for (i, (p, q)) in ab.iter().zip(bb.iter()).enumerate() {
            if p != q {
                n += 1;
                if first.is_none() {
                    first = Some(i);
                }
            }
        }
        Ok((n, first))
    }
}
