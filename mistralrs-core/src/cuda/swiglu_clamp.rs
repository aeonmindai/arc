//! Parent system: ArcInfer / ArcMoE
//!
//! Rust side of the fused clamped-SwiGLU port (`cuda/swiglu_clamp.cu`).
//!
//! Ported from vLLM `csrc/activation_kernels.cu` `silu_and_mul_clamp` and
//! SGLang `deepseek_v4/silu_and_mul_masked_post_quant.cuh`
//! (`silu_and_mul<kApplySwigluLimit = true>`) — both Apache-2.0; see the file
//! header of `swiglu_clamp.cu` for the license notice and exact line cites.
//!
//! Replaces this candle op chain, which V4 ran once per expert-MLP:
//!
//! ```text
//! gate.to_dtype(F32)         cast kernel  + 1-elem H2D for the scalar operand
//! .minimum(limit)            binary kernel
//! up.to_dtype(F32)           cast kernel  + 1-elem H2D
//! .clamp(-limit, limit)      TWO binary kernels (maximum then minimum) + 2 H2D
//! silu(gate)                 unary kernel
//! up * act                   binary kernel
//! .to_dtype(bf16)            cast kernel
//! ```
//!
//! Eight launches (plus five one-element host-to-device copies for the scalar
//! clamp operands — `Tensor::minimum`/`maximum` with a scalar builds the
//! operand on the CPU and `to_device`s it, see candle `binary_op_scalar!`)
//! collapse to ONE launch and zero H2D.

use candle_core::{DType, Result, Tensor};

/// `ARC_NO_FUSED_SWIGLU=1` forces the old candle chain. Read ONCE — this sits
/// on the per-expert-MLP path (86x per decode token), and `env::var_os` takes
/// the process-wide env lock, which would show up in the very A/B measurement
/// this switch exists to enable.
pub(crate) fn fused_swiglu_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("ARC_NO_FUSED_SWIGLU").is_some())
}

/// Engagement counters. A perf claim for this port is worthless unless the
/// fused kernel actually ran: a dtype or contiguity mismatch would silently
/// route every call back to the candle chain and the measurement would be of
/// nothing. `swiglu_engagement()` is asserted non-zero by the measurement
/// harness (see `ARC_SWIGLU_STATS`).
pub(crate) static FUSED_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static FALLBACK_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `(fused, fallback)` call counts since process start.
#[allow(dead_code)]
pub fn swiglu_engagement() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (FUSED_HITS.load(Relaxed), FALLBACK_HITS.load(Relaxed))
}

/// Record an ineligible call and log the reason ONCE, so a silent permanent
/// fallback shows up in the log instead of masquerading as a working port.
#[allow(dead_code)]
fn ineligible(reason: &'static str) -> Option<Tensor> {
    FALLBACK_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::warn!(
            "fused SwiGLU clamp ineligible ({reason}); falling back to the candle op chain. \
             Any perf claim for this port is void while this is the steady state."
        );
    });
    None
}

/// `silu(min(gate, limit)) * clamp(up, -limit, limit)`, fused.
///
/// Returns `Ok(None)` when this call is not eligible for the fused kernel
/// (wrong device, dtype, or shape mismatch) so callers can fall back to the
/// candle chain without losing the error.
///
/// Bit-identical to the candle chain by construction — see the contract in
/// `swiglu_clamp.cu` and the on-GPU A/B with negative control in `mod tests`.
#[cfg(feature = "cuda")]
pub fn swiglu_clamp_cuda(gate: &Tensor, up: &Tensor, limit: f32) -> Result<Option<Tensor>> {
    use candle_core as candle;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    if !gate.device().is_cuda() || !up.device().is_cuda() {
        return Ok(ineligible("not a CUDA tensor"));
    }
    if gate.dtype() != up.dtype() || gate.shape() != up.shape() {
        return Ok(ineligible("gate/up dtype or shape mismatch"));
    }
    if !matches!(gate.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return Ok(ineligible("unsupported dtype"));
    }
    if !gate.is_contiguous() || !up.is_contiguous() {
        return Ok(ineligible("non-contiguous input"));
    }
    let n = gate.elem_count();
    if n == 0 || n > u32::MAX as usize {
        return Ok(ineligible("element count out of range"));
    }
    // Engagement proof, in the server log, at the first real call.
    let prior = FUSED_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if prior == 0 {
        tracing::info!(
            "fused SwiGLU clamp ENGAGED: n={} dtype={:?} (vLLM silu_and_mul_clamp port)",
            gate.elem_count(),
            gate.dtype()
        );
    }

    let dev = gate.device().as_cuda_device()?;
    let dtype = gate.dtype();
    let shape = gate.shape().clone();
    let stream = dev.cuda_stream().cu_stream() as i64;

    macro_rules! run {
        ($ty:ty, $ffi:ident) => {{
            // Same block-scoped borrow idiom as cuda/sinkhorn.rs: the read
            // guard is released at the end of each block, the raw device
            // pointer outlives it.
            let g_ptr = {
                let (s, l) = gate.storage_and_layout();
                let s = match &*s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?,
                    _ => return Ok(None),
                };
                // Bind before yielding: `device_ptr` returns a
                // `(u64, SyncOnDrop<'_>)` whose guard must be dropped inside
                // the block, not at its tail (E0597).
                let p =
                    s.slice(l.start_offset()..).device_ptr(s.stream()).0 as *const std::ffi::c_void;
                p
            };
            let u_ptr = {
                let (s, l) = up.storage_and_layout();
                let s = match &*s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?,
                    _ => return Ok(None),
                };
                let p =
                    s.slice(l.start_offset()..).device_ptr(s.stream()).0 as *const std::ffi::c_void;
                p
            };
            let out_buf = unsafe { dev.alloc::<$ty>(n) }?;
            let o_ptr = {
                let p = out_buf.device_ptr(out_buf.stream()).0 as *mut std::ffi::c_void;
                p
            };
            unsafe {
                crate::cuda::ffi::$ffi(g_ptr, u_ptr, o_ptr, n as u32, limit, stream);
            }
            let st = candle::CudaStorage::wrap_cuda_slice(out_buf, dev.clone());
            Tensor::from((candle::Storage::Cuda(st), shape))
        }};
    }

    let out = match dtype {
        DType::BF16 => run!(half::bf16, swiglu_clamp_bf16),
        DType::F16 => run!(half::f16, swiglu_clamp_f16),
        DType::F32 => run!(f32, swiglu_clamp_f32),
        _ => unreachable!("dtype filtered above"),
    };
    Ok(Some(out))
}

#[cfg(not(feature = "cuda"))]
pub fn swiglu_clamp_cuda(_gate: &Tensor, _up: &Tensor, _limit: f32) -> Result<Option<Tensor>> {
    Ok(None)
}

/// Scalar replica of the candle op chain that `swiglu_clamp_cuda` replaces.
///
/// Kept so the *ordering* of the chain (one-sided gate clamp, symmetric up
/// clamp, both PRE-activation) is pinned by a CPU test even where no GPU is
/// available. The remaining scalar-vs-GPU gap is the same one sinkhorn.rs
/// documents: Rust/libm `f32::exp` vs CUDA libdevice `__nv_expf` may differ in
/// the last ulp. That gap CANCELS in the on-GPU A/B below, because both sides
/// of that comparison call the same `__nv_expf`.
#[allow(dead_code)] // consumed by `mod tests`; kept non-test for debug harnesses
pub(crate) fn swiglu_clamp_reference_f32(gate: f32, up: f32, limit: f32) -> f32 {
    let g = gate.min(limit);
    let u = up.max(-limit).min(limit);
    let act = g / (1.0f32 + (-g).exp());
    act * u
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clamp is one-sided on `gate` and symmetric on `up`, and both happen
    /// BEFORE the activation — the product itself is never clamped.
    #[test]
    fn reference_clamp_is_one_sided_on_gate_and_symmetric_on_up() {
        let l = 10.0f32;
        // gate below -limit is NOT clamped.
        let big_neg = swiglu_clamp_reference_f32(-50.0, 1.0, l);
        let expect = {
            let g = -50.0f32;
            g / (1.0 + (-g).exp())
        };
        assert_eq!(big_neg.to_bits(), expect.to_bits());
        // gate above +limit IS clamped.
        let big_pos = swiglu_clamp_reference_f32(50.0, 1.0, l);
        let expect = {
            let g = 10.0f32;
            g / (1.0 + (-g).exp())
        };
        assert_eq!(big_pos.to_bits(), expect.to_bits());
        // up is clamped on BOTH sides.
        assert_eq!(
            swiglu_clamp_reference_f32(1.0, -50.0, l).to_bits(),
            swiglu_clamp_reference_f32(1.0, -10.0, l).to_bits()
        );
        assert_eq!(
            swiglu_clamp_reference_f32(1.0, 50.0, l).to_bits(),
            swiglu_clamp_reference_f32(1.0, 10.0, l).to_bits()
        );
        // The PRODUCT is not clamped: silu(10) * 10 is far above the limit.
        assert!(swiglu_clamp_reference_f32(50.0, 50.0, l) > l);
    }

    /// On-GPU A/B: the fused kernel must be BIT-IDENTICAL to the candle chain
    /// it replaces, over inputs that straddle the clamp boundary on both
    /// tensors and both signs.
    ///
    /// The negative control is at the bottom: one input element is perturbed
    /// by a single bf16 ULP and the same comparison must REPORT a difference.
    /// Without it, a comparison that silently passes on everything (e.g. one
    /// that compares a tensor with itself) would look like success.
    #[cfg(feature = "cuda")]
    #[test]
    fn fused_kernel_is_bit_identical_to_candle_chain() -> Result<()> {
        use candle_core::Device;

        // NO silent skip. This test is compiled only under `--features cuda`;
        // if there is no usable GPU that is an ENVIRONMENT failure, and a
        // green "0 failures" here would be exactly the silent-success pattern
        // that has bitten this repo repeatedly. Fail loudly instead.
        let dev = Device::new_cuda(0).expect(
            "ENVIRONMENT: --features cuda is on but CUDA device 0 is unavailable. This test \
             proves nothing without a GPU; it must not pass silently.",
        );
        let limit = 10.0f32;
        let mut total_compared = 0usize;
        // 4096 * 7 = 28672 elements, divisible by 4 so the vec4 path is taken;
        // a second, prime-length case forces the scalar tail path.
        for n in [28672usize, 4099usize] {
            // Spread across [-40, 40] so a large fraction of elements exceed
            // the clamp on either side, and include exact +/-limit.
            let mk = |seed: f32| -> Result<Tensor> {
                let v: Vec<f32> = (0..n)
                    .map(|i| {
                        let t = (i as f32) * 0.017 + seed;
                        // deterministic, wide, and hits the boundary exactly
                        match i % 97 {
                            0 => limit,
                            1 => -limit,
                            2 => 0.0,
                            _ => 40.0 * (t.sin() * t.cos() * 2.3).tanh() * (t * 0.37).sin(),
                        }
                    })
                    .collect();
                Tensor::from_vec(v, n, &dev)?.to_dtype(DType::BF16)
            };
            let gate = mk(0.0)?;
            let up = mk(1.7)?;

            let reference = candle_chain(&gate, &up, limit)?;
            let fused = swiglu_clamp_cuda(&gate, &up, limit)?
                .expect("fused path must be eligible for contiguous BF16 CUDA tensors");

            let (n_diff, first) = bitwise_diff(&reference, &fused)?;
            assert_eq!(
                n_diff, 0,
                "n={n}: fused kernel is not bit-identical to the candle chain: \
                 {n_diff} mismatching elements, first at index {first:?}"
            );

            // ---- NEGATIVE CONTROL -------------------------------------------
            // Perturb ONE gate element by exactly one bf16 ULP and re-run the
            // fused kernel. The comparison above must now report a difference;
            // if it does not, the comparison itself is worthless.
            let mut vals: Vec<half::bf16> = gate.to_vec1::<half::bf16>()?;
            // index 5 is a generic (non-boundary, non-zero) element
            let before = vals[5];
            vals[5] = half::bf16::from_bits(before.to_bits().wrapping_add(1));
            assert_ne!(
                before.to_bits(),
                vals[5].to_bits(),
                "negative control setup failed: the 1-ULP perturbation was a no-op"
            );
            let gate_perturbed = Tensor::from_vec(vals, n, &dev)?;
            let fused_perturbed = swiglu_clamp_cuda(&gate_perturbed, &up, limit)?.unwrap();
            let (n_diff_ctrl, _) = bitwise_diff(&reference, &fused_perturbed)?;
            assert!(
                n_diff_ctrl > 0,
                "n={n}: NEGATIVE CONTROL FAILED — a 1-ULP perturbation of one \
                 input produced no reported difference, so the bit-identity \
                 check above proves nothing"
            );
            println!(
                "swiglu A/B n={n}: {n} elements bit-identical; negative control \
                 (1 bf16 ULP on gate[5]) reported {n_diff_ctrl} differing elements"
            );
            total_compared += n;
        }
        // Engagement assertion: a green run must prove it actually compared
        // something. "No failures" is not "results".
        assert_eq!(
            total_compared, 32771,
            "the A/B loop did not run over both shapes; nothing was proved"
        );
        Ok(())
    }

    /// Exactly the op chain `MoEExperts::swiglu` / `Mlp::gated_act` ran before
    /// the port. Deliberately written out here rather than calling the
    /// production helper, so a future edit to the production path cannot
    /// silently redefine what "bit-identical" is being asserted against.
    #[cfg(feature = "cuda")]
    fn candle_chain(gate: &Tensor, up: &Tensor, limit: f32) -> Result<Tensor> {
        let out_dtype = gate.dtype();
        let l = limit as f64;
        let g = gate.to_dtype(DType::F32)?.minimum(l)?;
        let u = up.to_dtype(DType::F32)?.clamp(-l, l)?;
        let act = g.apply(&crate::layers::Activation::Silu)?;
        u.mul(&act)?.to_dtype(out_dtype)
    }

    /// Count of elements whose raw BF16 bit patterns differ, plus the first
    /// index. Reads the bf16 storage directly — no f32 round-trip that could
    /// launder a difference away.
    #[cfg(feature = "cuda")]
    fn bitwise_diff(a: &Tensor, b: &Tensor) -> Result<(usize, Option<usize>)> {
        assert_eq!(a.dtype(), DType::BF16);
        assert_eq!(b.dtype(), DType::BF16);
        let ab: Vec<u16> = a
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let bb: Vec<u16> = b
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let mut n = 0usize;
        let mut first = None;
        for (i, (x, y)) in ab.iter().zip(bb.iter()).enumerate() {
            if x != y {
                n += 1;
                if first.is_none() {
                    first = Some(i);
                }
            }
        }
        Ok((n, first))
    }
}
