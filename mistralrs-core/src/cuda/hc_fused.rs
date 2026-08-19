//! Parent system: ArcInfer / ArcMoE
//!
//! Rust side of `cuda/hc_fused.cu` — the fused DeepSeek-V4 router-region
//! kernels.
//!
//! Two entry points, each replacing a launch-bound candle op chain with one
//! kernel:
//!
//! - [`hc_pre_fused_cuda`] — the 18-launch RMS-scale + pre/post/comb scoring
//!   block of [`crate::models::dsv4_mhc::V4MHCLayerParams::hc_pre`].
//! - [`sqrt_softplus_cuda`] — the 9-launch `sqrt(softplus(x))` gate scoring
//!   function of `MoeGate::forward`.
//!
//! Both are **bit-identical** to the chains they replace, not approximations.
//! The contract, and the three ways candle's arithmetic can be got wrong, are
//! documented at the top of `hc_fused.cu`. [`reference`] below carries scalar
//! replicas of both sides that `mod tests` asserts bit-identical without a GPU.
//!
//! Both are opt-out via `ARC_HC_FUSED=0`, which restores the eager chain. That
//! switch exists so the on-GPU A/B (fused vs. eager, greedy decode, compare
//! tokens) can be run from one binary.

/// `ARC_HC_FUSED=0` disables the fused kernels and restores the eager candle
/// chains. Any other value (or unset) keeps them on.
///
/// Read once and cached: this is consulted on every layer of every decode step,
/// and `std::env::var` takes a global lock.
pub fn fused_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !matches!(std::env::var("ARC_HC_FUSED").as_deref(), Ok("0")))
}

/// candle's `FastReduce` block size for a reduction of `len` elements
/// (`cuda_backend/mod.rs`: `usize::min(1024, el_to_sum_per_block)
/// .next_power_of_two()`). The fused kernel must use exactly this, because the
/// reduction tree's shape — and therefore the f32 rounding — depends on it.
pub(crate) fn candle_reduce_block_dim(len: usize) -> usize {
    usize::min(1024, len).next_power_of_two()
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core as candle;
    use candle_core::{DType, Result, Tensor};

    use super::candle_reduce_block_dim;

    /// Pull a contiguous F32 CUDA tensor's device pointer.
    fn f32_ptr(t: &Tensor, what: &str) -> Result<*const std::ffi::c_void> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        if t.dtype() != DType::F32 {
            candle::bail!("hc_fused: {what} must be F32, got {:?}", t.dtype());
        }
        if !t.is_contiguous() {
            candle::bail!("hc_fused: {what} must be contiguous");
        }
        let (s, l) = t.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("hc_fused: {what} must be on CUDA"),
        };
        // Bind before returning: `device_ptr` hands back a (ptr, guard) pair
        // whose guard borrows `s`, so the pointer must be extracted in its own
        // statement rather than in tail position. Same shape as
        // `sinkhorn::sinkhorn_normalize_cuda`.
        let ptr = s.slice(l.start_offset()..).device_ptr(s.stream()).0 as *const std::ffi::c_void;
        Ok(ptr)
    }

    /// Fused `hc_pre` middle section: the RMS statistic, its broadcast into
    /// `mixes`, and the three scoring blocks.
    ///
    /// Inputs (all F32, contiguous, same CUDA device):
    /// - `x_flat`    `[n, d]`  — the promoted residual stack, `d = hc * hidden`
    /// - `mixes_raw` `[n, m]`  — the gate GEMM output, `m = (2 + hc) * hc`
    /// - `hc_scale`  `[3]`
    /// - `hc_base`   `[m]`
    ///
    /// Returns `(pre [n, hc], post [n, hc], comb_pre [n, hc, hc])`.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_fused_cuda(
        x_flat: &Tensor,
        mixes_raw: &Tensor,
        hc_scale: &Tensor,
        hc_base: &Tensor,
        hc: usize,
        rms_eps: f64,
        hc_eps: f64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let (n, d) = x_flat.dims2()?;
        let (n2, m) = mixes_raw.dims2()?;
        if n != n2 {
            candle::bail!("hc_fused: x_flat has {n} rows but mixes_raw has {n2}");
        }
        if m != (2 + hc) * hc {
            candle::bail!(
                "hc_fused: mixes_raw has {m} columns, expected (2 + hc) * hc = {} for hc={hc}",
                (2 + hc) * hc
            );
        }
        if hc_scale.dims1()? != 3 {
            candle::bail!("hc_fused: hc_scale must be [3]");
        }
        if hc_base.dims1()? != m {
            candle::bail!("hc_fused: hc_base must be [{m}]");
        }
        let block_dim = candle_reduce_block_dim(d);
        if m > block_dim {
            // The scoring tail is done by threads 0..m, so it must fit the
            // block candle's reduction shape dictates. Unreachable for V4
            // (m = 24, block_dim = 1024) but a wrong answer if it ever isn't.
            candle::bail!(
                "hc_fused: m={m} exceeds the reduction block_dim={block_dim} implied by d={d}"
            );
        }
        if n == 0 {
            candle::bail!("hc_fused: empty batch");
        }

        let x_ptr = f32_ptr(x_flat, "x_flat")?;
        let mixes_ptr = f32_ptr(mixes_raw, "mixes_raw")?;
        let scale_ptr = f32_ptr(hc_scale, "hc_scale")?;
        let base_ptr = f32_ptr(hc_base, "hc_base")?;

        let dev = x_flat.device().as_cuda_device()?;
        let pre_buf = unsafe { dev.alloc::<f32>(n * hc) }?;
        let post_buf = unsafe { dev.alloc::<f32>(n * hc) }?;
        let comb_buf = unsafe { dev.alloc::<f32>(n * hc * hc) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        #[allow(clippy::cast_possible_truncation)]
        unsafe {
            crate::cuda::ffi::hc_pre_fused_f32(
                x_ptr,
                mixes_ptr,
                scale_ptr,
                base_ptr,
                pre_buf.device_ptr(pre_buf.stream()).0 as *mut std::ffi::c_void,
                post_buf.device_ptr(post_buf.stream()).0 as *mut std::ffi::c_void,
                comb_buf.device_ptr(comb_buf.stream()).0 as *mut std::ffi::c_void,
                n as i32,
                d as i32,
                m as i32,
                hc as i32,
                block_dim as i32,
                // candle's `mean_keepdim` scale is `T::from_f64(1f64 / len)`,
                // i.e. the f64 reciprocal narrowed to f32 — not `1.0f32 / d`.
                (1f64 / d as f64) as f32,
                rms_eps as f32,
                hc_eps as f32,
                stream,
            );
        }

        let pre_st = candle::CudaStorage::wrap_cuda_slice(pre_buf, dev.clone());
        let post_st = candle::CudaStorage::wrap_cuda_slice(post_buf, dev.clone());
        let comb_st = candle::CudaStorage::wrap_cuda_slice(comb_buf, dev.clone());
        Ok((
            Tensor::from((candle::Storage::Cuda(pre_st), (n, hc))),
            Tensor::from((candle::Storage::Cuda(post_st), (n, hc))),
            Tensor::from((candle::Storage::Cuda(comb_st), (n, hc, hc))),
        ))
    }

    /// Fused `sqrt(softplus(x))` — the V4 gate scoring function.
    pub fn sqrt_softplus_cuda(logits: &Tensor) -> Result<Tensor> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let logits = logits.contiguous()?;
        let numel = logits.elem_count();
        if numel == 0 {
            candle::bail!("hc_fused: sqrt_softplus on an empty tensor");
        }
        let in_ptr = f32_ptr(&logits, "logits")?;
        let dev = logits.device().as_cuda_device()?;
        let out_buf = unsafe { dev.alloc::<f32>(numel) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        #[allow(clippy::cast_possible_truncation)]
        unsafe {
            crate::cuda::ffi::sqrt_softplus_f32(
                in_ptr,
                out_buf.device_ptr(out_buf.stream()).0 as *mut std::ffi::c_void,
                numel as i64,
                stream,
            );
        }

        let st = candle::CudaStorage::wrap_cuda_slice(out_buf, dev.clone());
        Ok(Tensor::from((
            candle::Storage::Cuda(st),
            logits.shape().clone(),
        )))
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{hc_pre_fused_cuda, sqrt_softplus_cuda};

// Non-CUDA stubs so call sites need no `cfg`. They are never reached: every
// caller gates on [`usable`], which is false without a CUDA device.
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_fused_cuda(
    _x_flat: &candle_core::Tensor,
    _mixes_raw: &candle_core::Tensor,
    _hc_scale: &candle_core::Tensor,
    _hc_base: &candle_core::Tensor,
    _hc: usize,
    _rms_eps: f64,
    _hc_eps: f64,
) -> candle_core::Result<(
    candle_core::Tensor,
    candle_core::Tensor,
    candle_core::Tensor,
)> {
    candle_core::bail!("hc_pre_fused_cuda requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
pub fn sqrt_softplus_cuda(_logits: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("sqrt_softplus_cuda requires the cuda feature")
}

/// Whether the fused path should be taken for a tensor: a CUDA device, F32,
/// and not disabled by `ARC_HC_FUSED=0`.
///
/// Checked BEFORE the call rather than by catching an error from it, so that a
/// genuine CUDA failure inside the kernel surfaces as an error instead of being
/// silently swallowed into the eager path (the "silent success" failure mode
/// this codebase has been bitten by before).
pub fn usable(t: &candle_core::Tensor) -> bool {
    cfg!(feature = "cuda")
        && t.device().is_cuda()
        && t.dtype() == candle_core::DType::F32
        && fused_enabled()
}

/// Scalar f32 replicas that pin the fused kernels' op ORDER and ROUNDING to
/// what the candle CUDA backend actually executes, bit for bit.
///
/// Same discipline, and same known scalar-vs-GPU gap, as
/// [`crate::cuda::sinkhorn::reference`]: Rust/libm `f32::exp`/`ln` may differ
/// from CUDA libdevice `__nv_expf`/`__nv_logf` in the last ulp, but that
/// difference CANCELS in the on-GPU A/B because the candle chain and the fused
/// kernel call the same libdevice routine. Everything else here — add, mul,
/// div, fma, sqrt, single rounding per op — is exact IEEE f32 on both sides, so
/// the bitwise assertions in `mod tests` are meaningful.
#[allow(dead_code)] // consumed by `mod tests`
pub(crate) mod reference {
    /// candle-kernels `cuda_utils.cuh`: `recipg(float a) { return 1.0 / a; }`.
    /// The literal is a double, so this is `(float)(1.0 / (double)a)`.
    pub(crate) fn candle_recip(a: f32) -> f32 {
        (1.0f64 / a as f64) as f32
    }

    /// candle-kernels `unary.cu`: `sigmoid_fwd(x) = recipg(1 + expg(-x))`.
    pub(crate) fn candle_sigmoid(x: f32) -> f32 {
        candle_recip(1.0f32 + (-x).exp())
    }

    /// candle-kernels `fast_sum` (reduce.cu) over a contiguous row, with
    /// candle's `FastReduce` block size. Each virtual thread accumulates its
    /// strided slice sequentially into a zero-initialised accumulator, then the
    /// pairwise tree runs over `block_dim` slots.
    pub(crate) fn candle_fast_sum(vals: &[f32], block_dim: usize) -> f32 {
        let mut shr = vec![0.0f32; block_dim];
        for (tid, slot) in shr.iter_mut().enumerate() {
            let mut idx = tid;
            while idx < vals.len() {
                *slot += vals[idx];
                idx += block_dim;
            }
        }
        let mut s = block_dim / 2;
        while s > 0 {
            for t in 0..s {
                shr[t] += shr[t + s];
            }
            s /= 2;
        }
        shr[0]
    }

    /// The **candle op chain** of `dsv4_mhc::hc_pre`, one step per kernel
    /// launch, in the order the eager path issues them.
    ///
    /// Returns `(pre, post, comb_pre)`.
    #[allow(clippy::type_complexity)]
    pub(crate) fn hc_pre_candle_replay(
        x_flat: &[f32],
        mixes_raw: &[f32],
        hc_scale: &[f32],
        hc_base: &[f32],
        hc: usize,
        rms_eps: f64,
        hc_eps: f64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = x_flat.len();
        let m = mixes_raw.len();
        let block_dim = super::candle_reduce_block_dim(d);

        // sqr() -> a whole separate kernel, so x*x is rounded to f32 first.
        let sq: Vec<f32> = x_flat.iter().map(|v| v * v).collect();
        // mean_keepdim = fast_sum, then a SEPARATE affine(1/d, 0) == fmaf.
        let sum = candle_fast_sum(&sq, block_dim);
        let inv_d = (1f64 / d as f64) as f32;
        let mean = sum.mul_add(inv_d, 0.0f32);
        // + rms_norm_eps  ->  affine(1.0, eps) == fmaf(x, 1, eps)
        let meps = mean.mul_add(1.0f32, rms_eps as f32);
        // recip() then sqrt()
        let rsqrt = candle_recip(meps).sqrt();

        // broadcast_mul -> `mixes`
        let mixes: Vec<f32> = mixes_raw.iter().map(|v| v * rsqrt).collect();

        let mut pre = Vec::with_capacity(hc);
        for j in 0..hc {
            // broadcast_mul, broadcast_add, sigmoid, affine(1, hc_eps)
            let t = mixes[j] * hc_scale[0] + hc_base[j];
            pre.push(candle_sigmoid(t).mul_add(1.0f32, hc_eps as f32));
        }
        let mut post = Vec::with_capacity(hc);
        for j in 0..hc {
            let t = mixes[hc + j] * hc_scale[1] + hc_base[hc + j];
            post.push(candle_sigmoid(t).mul_add(2.0f32, 0.0f32));
        }
        let mut comb = Vec::with_capacity(hc * hc);
        for j in 0..(m - 2 * hc) {
            comb.push(mixes[2 * hc + j] * hc_scale[2] + hc_base[2 * hc + j]);
        }
        (pre, post, comb)
    }

    /// The **fused kernel** of `hc_fused.cu`, transcribed: one block, `acc` in
    /// a register, the same tree, then the scoring tail.
    #[allow(clippy::type_complexity)]
    pub(crate) fn hc_pre_fused_replay(
        x_flat: &[f32],
        mixes_raw: &[f32],
        hc_scale: &[f32],
        hc_base: &[f32],
        hc: usize,
        rms_eps: f64,
        hc_eps: f64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = x_flat.len();
        let m = mixes_raw.len();
        let block_dim = super::candle_reduce_block_dim(d);

        // Per-thread: acc = fadd(acc, fmul(v, v)) over the strided slice.
        let mut shr = vec![0.0f32; block_dim];
        for (tid, slot) in shr.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            let mut idx = tid;
            while idx < d {
                let v = x_flat[idx];
                acc += v * v;
                idx += block_dim;
            }
            *slot = acc;
        }
        let mut s = block_dim / 2;
        while s > 0 {
            for t in 0..s {
                shr[t] += shr[t + s];
            }
            s /= 2;
        }
        let sum = shr[0];
        let inv_d = (1f64 / d as f64) as f32;
        let mean = sum.mul_add(inv_d, 0.0f32);
        let meps = mean.mul_add(1.0f32, rms_eps as f32);
        let rsqrt = candle_recip(meps).sqrt();

        let mut pre = vec![0.0f32; hc];
        let mut post = vec![0.0f32; hc];
        let mut comb = vec![0.0f32; m - 2 * hc];
        for tid in 0..m {
            let mx = mixes_raw[tid] * rsqrt;
            if tid < hc {
                let t = mx * hc_scale[0] + hc_base[tid];
                pre[tid] = candle_sigmoid(t).mul_add(1.0f32, hc_eps as f32);
            } else if tid < 2 * hc {
                let j = tid - hc;
                let t = mx * hc_scale[1] + hc_base[hc + j];
                post[j] = candle_sigmoid(t).mul_add(2.0f32, 0.0f32);
            } else {
                let j = tid - 2 * hc;
                let t = mx * hc_scale[2] + hc_base[2 * hc + j];
                comb[j] = t;
            }
        }
        (pre, post, comb)
    }

    /// The **candle op chain** of `MoeGate::forward`'s `SqrtSoftplus` arm.
    pub(crate) fn sqrt_softplus_candle_replay(logits: &[f32]) -> Vec<f32> {
        // max0 = maximum(logits, zeros_like)
        let max0: Vec<f32> = logits.iter().map(|&l| l.max(0.0f32)).collect();
        // abs -> neg -> exp -> affine(1, 1) -> log
        let a: Vec<f32> = logits.iter().map(|&l| l.abs()).collect();
        let n: Vec<f32> = a.iter().map(|&v| -v).collect();
        let e: Vec<f32> = n.iter().map(|&v| v.exp()).collect();
        let p1: Vec<f32> = e.iter().map(|&v| v.mul_add(1.0f32, 1.0f32)).collect();
        let lg: Vec<f32> = p1.iter().map(|&v| v.ln()).collect();
        // badd then usqrt
        max0.iter()
            .zip(lg.iter())
            .map(|(&x, &y)| (x + y).sqrt())
            .collect()
    }

    /// The **fused kernel** of `hc_fused.cu`, transcribed.
    pub(crate) fn sqrt_softplus_fused_replay(logits: &[f32]) -> Vec<f32> {
        logits
            .iter()
            .map(|&l| {
                let mx = l.max(0.0f32);
                let a = l.abs();
                let e = (-a).exp();
                let p1 = e.mul_add(1.0f32, 1.0f32);
                let lg = p1.ln();
                (mx + lg).sqrt()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::reference::*;

    /// Deterministic LCG so a failure is reproducible without a fixture file.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((self.0 >> 40) as f32) / ((1u32 << 24) as f32);
            lo + u * (hi - lo)
        }
    }

    /// Source-level tripwires, mirroring `sinkhorn::tests`: the IEEE
    /// intrinsics and the fast-math `#error` guard must stay in hc_fused.cu,
    /// fast-math approximations must stay out, and build.rs must keep the file
    /// out of the `--use_fast_math` glob and in the `--fmad=false` builder.
    /// Getting this wiring wrong is silent — the kernel still runs, it just
    /// stops being bit-identical — so the `#error` guard is the hard stop and
    /// these string checks catch it on CPU CI too.
    #[test]
    fn kernel_source_and_build_wiring_guards() {
        let cu = include_str!("hc_fused.cu");
        assert!(
            cu.contains("#if defined(__USE_FAST_MATH__)") && cu.contains("#error"),
            "hc_fused.cu lost its fast-math #error guard"
        );
        for required in ["__fadd_rn", "__fmul_rn", "candle_recip", "candle_sigmoid"] {
            assert!(cu.contains(required), "hc_fused.cu lost required token {required}");
        }
        for forbidden in ["__expf(", "__logf(", "__fdividef(", "rsqrtf(", "__frsqrt_rn("] {
            assert!(
                !cu.contains(forbidden),
                "hc_fused.cu contains {forbidden}, which is not what candle-kernels computes"
            );
        }

        let build = include_str!("../../build.rs");
        let exclude = build
            .split(".exclude(&[")
            .nth(1)
            .and_then(|s| s.split("])").next())
            .expect("build.rs no longer calls .exclude(&[..]) on the fast-math builder");
        assert!(
            exclude.contains("\"hc_fused.cu\""),
            "build.rs no longer excludes hc_fused.cu from the fast-math builder \
             (exclude list is: {exclude})"
        );
        assert!(
            build.contains(r#""src/cuda/hc_fused.cu""#),
            "build.rs no longer feeds hc_fused.cu to the IEEE (no-fast-math) builder"
        );
        assert!(
            build.contains("--fmad=false"),
            "build.rs lost --fmad=false, so nvcc may contract mul+add into an FMA and break \
             bit-identity with candle's unfused op chain"
        );
    }

    #[test]
    fn candle_reduce_block_dim_matches_fast_reduce() {
        // cuda_backend/mod.rs: usize::min(1024, el_to_sum_per_block).next_power_of_two()
        assert_eq!(super::candle_reduce_block_dim(16384), 1024); // V4: hc_mult * hidden
        assert_eq!(super::candle_reduce_block_dim(1024), 1024);
        assert_eq!(super::candle_reduce_block_dim(1000), 1024);
        assert_eq!(super::candle_reduce_block_dim(24), 32);
        assert_eq!(super::candle_reduce_block_dim(4), 4);
        assert_eq!(super::candle_reduce_block_dim(3), 4);
    }

    /// The whole point of the file: the fused kernel and the candle op chain
    /// must agree BITWISE, at V4's real shapes.
    #[test]
    fn hc_pre_fused_is_bit_identical_to_candle_chain() {
        let hc = 4usize;
        let hidden = 4096usize;
        let d = hc * hidden;
        let m = (2 + hc) * hc;
        let mut rng = Lcg(0x5eed_1234_abcd_0001);

        for trial in 0..8 {
            let x: Vec<f32> = (0..d).map(|_| rng.next_f32(-3.0, 3.0)).collect();
            let mixes: Vec<f32> = (0..m).map(|_| rng.next_f32(-6.0, 6.0)).collect();
            let scale: Vec<f32> = (0..3).map(|_| rng.next_f32(-2.0, 2.0)).collect();
            let base: Vec<f32> = (0..m).map(|_| rng.next_f32(-2.0, 2.0)).collect();

            let (pa, qa, ca) = hc_pre_candle_replay(&x, &mixes, &scale, &base, hc, 1e-6, 1e-6);
            let (pb, qb, cb) = hc_pre_fused_replay(&x, &mixes, &scale, &base, hc, 1e-6, 1e-6);

            for (i, (a, b)) in pa.iter().zip(pb.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "trial {trial} pre[{i}]: {a} vs {b}");
            }
            for (i, (a, b)) in qa.iter().zip(qb.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "trial {trial} post[{i}]: {a} vs {b}");
            }
            for (i, (a, b)) in ca.iter().zip(cb.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "trial {trial} comb[{i}]: {a} vs {b}");
            }
        }
    }

    /// A guard that cannot go red is not a guard. Reassociating the reduction —
    /// the single most likely way to break this fusion — must be detected.
    #[test]
    fn hc_pre_guard_detects_a_reassociated_reduction() {
        let hc = 4usize;
        let d = hc * 4096;
        let m = (2 + hc) * hc;
        let mut rng = Lcg(0xdead_beef_0000_0007);
        let x: Vec<f32> = (0..d).map(|_| rng.next_f32(-3.0, 3.0)).collect();
        let mixes: Vec<f32> = (0..m).map(|_| rng.next_f32(-6.0, 6.0)).collect();
        let scale: Vec<f32> = (0..3).map(|_| rng.next_f32(-2.0, 2.0)).collect();
        let base: Vec<f32> = (0..m).map(|_| rng.next_f32(-2.0, 2.0)).collect();

        let (pre_ref, ..) = hc_pre_candle_replay(&x, &mixes, &scale, &base, hc, 1e-6, 1e-6);

        // Naive left-to-right sum instead of candle's strided-then-tree order.
        let mut seq = 0.0f32;
        for v in &x {
            seq += v * v;
        }
        let inv_d = (1f64 / d as f64) as f32;
        let rsqrt = candle_recip(seq.mul_add(inv_d, 0.0f32).mul_add(1.0f32, 1e-6f32)).sqrt();
        let pre_wrong: Vec<f32> = (0..hc)
            .map(|j| {
                let t = (mixes[j] * rsqrt) * scale[0] + base[j];
                candle_sigmoid(t).mul_add(1.0f32, 1e-6f32)
            })
            .collect();

        assert!(
            pre_ref
                .iter()
                .zip(pre_wrong.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "the reduction-order guard is vacuous: a sequential sum of {d} squares produced \
             bit-identical output to candle's strided+tree order, so this test would pass on a \
             kernel that reassociates the reduction"
        );
    }

    #[test]
    fn sqrt_softplus_fused_is_bit_identical_to_candle_chain() {
        let mut rng = Lcg(0x0bad_c0de_1111_2222);
        // n_routed_experts = 256 for V4-Flash.
        let logits: Vec<f32> = (0..256).map(|_| rng.next_f32(-40.0, 40.0)).collect();
        let a = sqrt_softplus_candle_replay(&logits);
        let b = sqrt_softplus_fused_replay(&logits);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "logit[{i}]={}: {x} vs {y}", logits[i]);
        }
        // Including the exact-zero / sign boundary that `maximum(x, 0)` and
        // `abs(x)` disagree on.
        let edge = [0.0f32, -0.0f32, 1e-30, -1e-30, 88.0, -88.0, 1.0, -1.0];
        let a = sqrt_softplus_candle_replay(&edge);
        let b = sqrt_softplus_fused_replay(&edge);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "edge[{i}]={}: {x} vs {y}", edge[i]);
        }
    }

    /// Same discipline for the softplus guard: prove it can go red.
    ///
    /// Mutation-tests the three ways a careless reimplementation of
    /// `sqrt(softplus(x))` actually goes wrong, and asserts the bitwise
    /// comparison catches each. The first one is the reason this test exists at
    /// all: an earlier version of it sampled only `[-30, -12.5, -1, 0.5, 3.25,
    /// 17]` and PASSED against the naive form, because over that range the two
    /// formulations happen to agree bit for bit — the stable split only earns
    /// its keep once `exp(x)` overflows f32 near x = 88.7. A guard that green-
    /// lights the unstable form is not a guard.
    #[test]
    fn sqrt_softplus_guard_detects_plausible_mistranscriptions() {
        // Spans the overflow boundary (exp overflows f32 above ~88.7) and the
        // sign boundary that `max(x, 0)` and `-|x|` turn on.
        let logits: Vec<f32> = vec![-120.0, -30.0, -1.0, -0.25, 0.0, 0.25, 3.25, 17.0, 95.0, 200.0];
        let reference = sqrt_softplus_candle_replay(&logits);

        let mutants: [(&str, fn(f32) -> f32); 3] = [
            // 1. the unstable form: log(1 + exp(x)), no max/abs split.
            ("unstable log(1+exp(x))", |l| (1.0f32 + l.exp()).ln().sqrt()),
            // 2. dropped the max(x, 0) term.
            ("dropped max(x,0)", |l| {
                (1.0f32 + (-l.abs()).exp()).ln().sqrt()
            }),
            // 3. lost the negation on the exponent.
            ("exp(+|x|) instead of exp(-|x|)", |l| {
                (l.max(0.0f32) + (1.0f32 + l.abs().exp()).ln()).sqrt()
            }),
        ];

        for (name, mutate) in mutants {
            let got: Vec<f32> = logits.iter().map(|&l| mutate(l)).collect();
            assert!(
                reference
                    .iter()
                    .zip(got.iter())
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "the softplus guard is vacuous for mutation '{name}': it produced bit-identical \
                 output to the reference on every sample, so this test would pass on a kernel \
                 carrying that bug"
            );
        }
    }
}
