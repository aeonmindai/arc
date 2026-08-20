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

/// `ARC_HC_FUSE2=0` disables ONLY the second-wave fusions — the `hc_pre`
/// weighted-average tail and the whole of `hc_post` — while leaving the
/// router-region kernels (`hc_pre_fused`, `sqrt_softplus`) on.
///
/// Separate from `ARC_HC_FUSED` on purpose: with one switch per wave, the
/// contribution of THIS change is measurable from a single binary, so a
/// before/after cannot be contaminated by a rebuild.
pub fn fuse2_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !matches!(std::env::var("ARC_HC_FUSE2").as_deref(), Ok("0")))
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

    /// Tag matching `HC_DTYPE_*` in hc_fused.cu. Only the dtypes whose cast
    /// semantics have been transcribed are accepted; anything else returns an
    /// error so the caller falls back to the eager chain rather than silently
    /// running a kernel that rounds differently.
    fn dtype_tag(d: DType) -> Result<i32> {
        match d {
            DType::F32 => Ok(0),
            DType::BF16 => Ok(1),
            other => candle::bail!("hc_fused: unsupported dtype {other:?}"),
        }
    }

    /// Device pointer for a contiguous CUDA tensor of a supported dtype.
    fn any_ptr(t: &Tensor, what: &str) -> Result<*const std::ffi::c_void> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        if !t.is_contiguous() {
            candle::bail!("hc_fused: {what} must be contiguous");
        }
        let (s, l) = t.storage_and_layout();
        let cuda = match &*s {
            candle::Storage::Cuda(c) => c,
            _ => candle::bail!("hc_fused: {what} must be on CUDA"),
        };
        // The (ptr, guard) pair borrows the slice, so bind before returning.
        let ptr = match t.dtype() {
            DType::F32 => {
                let sl = cuda.as_cuda_slice::<f32>()?;
                sl.slice(l.start_offset()..).device_ptr(sl.stream()).0
            }
            DType::BF16 => {
                let sl = cuda.as_cuda_slice::<half::bf16>()?;
                sl.slice(l.start_offset()..).device_ptr(sl.stream()).0
            }
            other => candle::bail!("hc_fused: {what} has unsupported dtype {other:?}"),
        };
        Ok(ptr as *const std::ffi::c_void)
    }

    /// Allocate an output buffer of `dtype` and wrap it as a `Tensor`.
    fn alloc_out(
        dev: &candle_core::CudaDevice,
        dtype: DType,
        elems: usize,
        shape: impl Into<candle_core::Shape>,
        run: impl FnOnce(*mut std::ffi::c_void) -> i32,
    ) -> Result<Tensor> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        match dtype {
            DType::F32 => {
                let buf = unsafe { dev.alloc::<f32>(elems) }?;
                let rc = run(buf.device_ptr(buf.stream()).0 as *mut std::ffi::c_void);
                if rc != 0 {
                    candle::bail!("hc_fused: kernel dispatch refused (rc={rc})");
                }
                let st = candle::CudaStorage::wrap_cuda_slice(buf, dev.clone());
                Ok(Tensor::from((candle::Storage::Cuda(st), shape.into())))
            }
            DType::BF16 => {
                let buf = unsafe { dev.alloc::<half::bf16>(elems) }?;
                let rc = run(buf.device_ptr(buf.stream()).0 as *mut std::ffi::c_void);
                if rc != 0 {
                    candle::bail!("hc_fused: kernel dispatch refused (rc={rc})");
                }
                let st = candle::CudaStorage::wrap_cuda_slice(buf, dev.clone());
                Ok(Tensor::from((candle::Storage::Cuda(st), shape.into())))
            }
            other => candle::bail!("hc_fused: unsupported output dtype {other:?}"),
        }
    }

    /// Fused V4 clamped SwiGLU: `narrow(silu(min(gate, limit)) * clamp(up, -limit, limit))`.
    ///
    /// Replaces the 8-launch (routed) / 7-launch (shared) chain in
    /// `moe::experts::swiglu_clamp` + the activation stage that follows it,
    /// plus the three 4-byte host-to-device copies candle makes for the clamp
    /// bounds.
    ///
    /// - `gate`, `up`: same shape, same dtype (F32 or BF16), contiguous, CUDA
    /// - `out_dtype`: normally `gate.dtype()`; F32 is used by the A/B harness
    ///   to obtain the product *before* the narrowing, where a bitwise
    ///   comparison is not swallowed by BF16's 8 mantissa bits.
    pub fn swiglu_clamp_fused_cuda(
        gate: &Tensor,
        up: &Tensor,
        limit: f32,
        out_dtype: DType,
    ) -> Result<Tensor> {
        if gate.dims() != up.dims() {
            candle::bail!(
                "hc_fused: swiglu gate {:?} vs up {:?}",
                gate.dims(),
                up.dims()
            );
        }
        if gate.dtype() != up.dtype() {
            candle::bail!(
                "hc_fused: swiglu gate is {:?} but up is {:?}",
                gate.dtype(),
                up.dtype()
            );
        }
        let total = gate.elem_count();
        if total == 0 {
            candle::bail!("hc_fused: empty swiglu");
        }
        let in_tag = dtype_tag(gate.dtype())?;
        let out_tag = dtype_tag(out_dtype)?;
        let gate_ptr = any_ptr(gate, "swiglu gate")?;
        let up_ptr = any_ptr(up, "swiglu up")?;
        let dev = gate.device().as_cuda_device()?;
        let stream = dev.cuda_stream().cu_stream() as i64;
        let shape = gate.shape().clone();

        #[allow(clippy::cast_possible_truncation)]
        alloc_out(&dev, out_dtype, total, shape, |out| unsafe {
            crate::cuda::ffi::arc_seam_swiglu_clamp(
                gate_ptr, up_ptr, out, limit, total as i64, in_tag, out_tag, stream,
            )
        })
    }

    /// The clamp half of [`swiglu_clamp_fused_cuda`], returning
    /// `(min(gate, limit), clamp(up, -limit, limit))` in F32.
    ///
    /// Used at the shared-expert site, whose activation is `fused_glu` compiled
    /// with `--use_fast_math` (mistralrs-quant/build.rs:281) and therefore not
    /// reproducible from this no-fast-math translation unit. See the kernel's
    /// comment in `hc_fused.cu` for why the weaker fusion is the honest one
    /// there.
    pub fn swiglu_clamp_split_cuda(
        gate: &Tensor,
        up: &Tensor,
        limit: f32,
    ) -> Result<(Tensor, Tensor)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        if gate.dims() != up.dims() {
            candle::bail!(
                "hc_fused: swiglu-split gate {:?} vs up {:?}",
                gate.dims(),
                up.dims()
            );
        }
        if gate.dtype() != up.dtype() {
            candle::bail!("hc_fused: swiglu-split gate/up dtypes differ");
        }
        let total = gate.elem_count();
        if total == 0 {
            candle::bail!("hc_fused: empty swiglu-split");
        }
        let in_tag = dtype_tag(gate.dtype())?;
        let gate_ptr = any_ptr(gate, "swiglu gate")?;
        let up_ptr = any_ptr(up, "swiglu up")?;
        let dev = gate.device().as_cuda_device()?;
        let stream = dev.cuda_stream().cu_stream() as i64;
        let shape = gate.shape().clone();

        let g_buf = unsafe { dev.alloc::<f32>(total) }?;
        let u_buf = unsafe { dev.alloc::<f32>(total) }?;
        #[allow(clippy::cast_possible_truncation)]
        let rc = unsafe {
            crate::cuda::ffi::arc_seam_swiglu_clamp_split(
                gate_ptr,
                up_ptr,
                g_buf.device_ptr(g_buf.stream()).0 as *mut std::ffi::c_void,
                u_buf.device_ptr(u_buf.stream()).0 as *mut std::ffi::c_void,
                limit,
                total as i64,
                in_tag,
                stream,
            )
        };
        if rc != 0 {
            candle::bail!("hc_fused: swiglu-split dispatch refused (rc={rc})");
        }
        let g_st = candle::CudaStorage::wrap_cuda_slice(g_buf, dev.clone());
        let u_st = candle::CudaStorage::wrap_cuda_slice(u_buf, dev.clone());
        Ok((
            Tensor::from((candle::Storage::Cuda(g_st), shape.clone())),
            Tensor::from((candle::Storage::Cuda(u_st), shape)),
        ))
    }

    /// Fused MoE expert combine: `out[n, h] = narrow(sum_j ys[n, j, h] * w[n, j])`,
    /// replacing `to_dtype + broadcast_mul + sum(-2) + to_dtype` (4 launches).
    ///
    /// - `ys` `[n, k, h]` model dtype, contiguous
    /// - `w`  `[n, k]`    F32, contiguous
    pub fn moe_weighted_sum_cuda(ys: &Tensor, w: &Tensor, out_dtype: DType) -> Result<Tensor> {
        let (n, k, h) = ys.dims3()?;
        let (wn, wk) = w.dims2()?;
        if wn != n || wk != k {
            candle::bail!("hc_fused: weights are [{wn}, {wk}] but ys is [{n}, {k}, {h}]");
        }
        if n == 0 || h == 0 {
            candle::bail!("hc_fused: empty weighted sum");
        }
        let in_tag = dtype_tag(ys.dtype())?;
        let out_tag = dtype_tag(out_dtype)?;
        let ys_ptr = any_ptr(ys, "ys")?;
        let w_ptr = f32_ptr(w, "topk_weights")?;
        let dev = ys.device().as_cuda_device()?;
        let stream = dev.cuda_stream().cu_stream() as i64;
        let total = (n * h) as i64;

        #[allow(clippy::cast_possible_truncation)]
        alloc_out(&dev, out_dtype, n * h, (n, h), |out| unsafe {
            crate::cuda::ffi::arc_seam_moe_weighted_sum(
                ys_ptr, w_ptr, out, k as i32, h as i32, total, in_tag, out_tag, stream,
            )
        })
    }

    /// Fused MoE gate renormalise + scale, replacing
    /// `sum_keepdim + affine(+eps) + broadcast_div + affine(*scale)`
    /// (4 launches).
    ///
    /// `w` is `[n, k]` F32 contiguous. `do_renorm == false` applies only the
    /// scale, which is what the config-gated eager path does when
    /// `norm_topk_prob` is off.
    pub fn gate_renorm_cuda(
        w: &Tensor,
        eps: f32,
        scale: f32,
        do_renorm: bool,
    ) -> Result<Tensor> {
        let (n, k) = w.dims2()?;
        if n == 0 || k == 0 {
            candle::bail!("hc_fused: empty gate renorm");
        }
        let w_ptr = f32_ptr(w, "topk_weight")?;
        let dev = w.device().as_cuda_device()?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        #[allow(clippy::cast_possible_truncation)]
        alloc_out(&dev, DType::F32, n * k, (n, k), |out| unsafe {
            crate::cuda::ffi::arc_seam_gate_renorm(
                w_ptr,
                out,
                eps,
                scale,
                i32::from(do_renorm),
                k as i32,
                n as i64,
                stream,
            )
        })
    }

    /// Fused `hc_pre` tail: `y = sum_i pre[i] * x_f32[i, :]`, narrowed to
    /// `out_dtype`. Replaces `broadcast_mul + sum(1) + to_dtype` (3 launches).
    ///
    /// - `x_f32` `[n, hc, h]` F32 contiguous (the promoted residual stack)
    /// - `pre`   `[n, hc]`    F32 contiguous
    pub fn hc_y_combine_cuda(x_f32: &Tensor, pre: &Tensor, out_dtype: DType) -> Result<Tensor> {
        let (n, hc, h) = x_f32.dims3()?;
        let (pn, phc) = pre.dims2()?;
        if pn != n || phc != hc {
            candle::bail!("hc_fused: pre is [{pn}, {phc}] but x_f32 is [{n}, {hc}, {h}]");
        }
        if n == 0 || h == 0 {
            candle::bail!("hc_fused: empty y-combine");
        }
        let tag = dtype_tag(out_dtype)?;
        let x_ptr = f32_ptr(x_f32, "x_f32")?;
        let pre_ptr = f32_ptr(pre, "pre")?;
        let dev = x_f32.device().as_cuda_device()?;
        let stream = dev.cuda_stream().cu_stream() as i64;
        let total = (n * h) as i64;

        #[allow(clippy::cast_possible_truncation)]
        alloc_out(&dev, out_dtype, n * h, (n, h), |out| unsafe {
            crate::cuda::ffi::hc_y_combine(
                x_ptr, pre_ptr, out, hc as i32, h as i32, total, tag, stream,
            )
        })
    }

    /// Fused `hc_post`: the whole 6-launch re-expansion in one launch.
    ///
    /// - `x`        `[n, h]`      model dtype
    /// - `residual` `[n, hc, h]`  model dtype
    /// - `post`     `[n, hc]`     F32
    /// - `comb`     `[n, hc, hc]` F32
    pub fn hc_post_fused_cuda(
        x: &Tensor,
        residual: &Tensor,
        post: &Tensor,
        comb: &Tensor,
    ) -> Result<Tensor> {
        let (n, h) = x.dims2()?;
        let (rn, hc, rh) = residual.dims3()?;
        if rn != n || rh != h {
            candle::bail!("hc_fused: residual [{rn}, {hc}, {rh}] vs x [{n}, {h}]");
        }
        if post.dims2()? != (n, hc) {
            candle::bail!("hc_fused: post must be [{n}, {hc}], got {:?}", post.dims());
        }
        if comb.dims3()? != (n, hc, hc) {
            candle::bail!("hc_fused: comb must be [{n}, {hc}, {hc}], got {:?}", comb.dims());
        }
        if x.dtype() != residual.dtype() {
            candle::bail!(
                "hc_fused: x is {:?} but residual is {:?}",
                x.dtype(),
                residual.dtype()
            );
        }
        if n == 0 || h == 0 {
            candle::bail!("hc_fused: empty hc_post");
        }
        let tag = dtype_tag(x.dtype())?;
        let x_ptr = any_ptr(x, "x")?;
        let res_ptr = any_ptr(residual, "residual")?;
        let post_ptr = f32_ptr(post, "post")?;
        let comb_ptr = f32_ptr(comb, "comb")?;
        let dev = x.device().as_cuda_device()?;
        let stream = dev.cuda_stream().cu_stream() as i64;
        let total = (n * hc * h) as i64;

        #[allow(clippy::cast_possible_truncation)]
        alloc_out(&dev, x.dtype(), n * hc * h, (n, hc, h), |out| unsafe {
            crate::cuda::ffi::hc_post_fused(
                x_ptr, res_ptr, post_ptr, comb_ptr, out, hc as i32, h as i32, total, tag, stream,
            )
        })
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{
    gate_renorm_cuda, hc_post_fused_cuda, hc_pre_fused_cuda, hc_y_combine_cuda,
    moe_weighted_sum_cuda, sqrt_softplus_cuda, swiglu_clamp_fused_cuda, swiglu_clamp_split_cuda,
};

#[cfg(not(feature = "cuda"))]
pub fn swiglu_clamp_split_cuda(
    _gate: &candle_core::Tensor,
    _up: &candle_core::Tensor,
    _limit: f32,
) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
    candle_core::bail!("swiglu_clamp_split_cuda requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
pub fn swiglu_clamp_fused_cuda(
    _gate: &candle_core::Tensor,
    _up: &candle_core::Tensor,
    _limit: f32,
    _out_dtype: candle_core::DType,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("swiglu_clamp_fused_cuda requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
pub fn moe_weighted_sum_cuda(
    _ys: &candle_core::Tensor,
    _w: &candle_core::Tensor,
    _out_dtype: candle_core::DType,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("moe_weighted_sum_cuda requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
pub fn gate_renorm_cuda(
    _w: &candle_core::Tensor,
    _eps: f32,
    _scale: f32,
    _do_renorm: bool,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("gate_renorm_cuda requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
pub fn hc_y_combine_cuda(
    _x_f32: &candle_core::Tensor,
    _pre: &candle_core::Tensor,
    _out_dtype: candle_core::DType,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("hc_y_combine_cuda requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
pub fn hc_post_fused_cuda(
    _x: &candle_core::Tensor,
    _residual: &candle_core::Tensor,
    _post: &candle_core::Tensor,
    _comb: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("hc_post_fused_cuda requires the cuda feature")
}

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

/// Same gate as [`usable`] but for the model-dtype tensors (`hc_post`, the
/// `hc_pre` tail), which are BF16 rather than F32.
pub fn usable_model_dtype(t: &candle_core::Tensor) -> bool {
    cfg!(feature = "cuda")
        && t.device().is_cuda()
        && matches!(t.dtype(), candle_core::DType::F32 | candle_core::DType::BF16)
        && fused_enabled()
}

// ---------------------------------------------------------------------------
// The F32 seam — gates and engagement accounting
// ---------------------------------------------------------------------------
//
// One master switch so the whole change can be A/B'd from a single binary
// (`ARC_F32SEAM=0` restores every eager chain), plus one switch per site so a
// site that turns out to cost bit-identity can be dropped without a rebuild.
//
// ENGAGEMENT IS REPORTED, NOT ASSUMED. A timing win from a fast path that never
// ran is this codebase's most-repeated failure. Every site logs the first time
// it engages AND the first time it declines, with the reason, and prints running
// totals; "no DECLINED line" is then a fact in the log rather than an inference.

/// `ARC_F32SEAM=0` restores every eager chain in the seam. Anything else keeps
/// them fused.
pub fn seam_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("ARC_F32SEAM").as_deref(), Ok("0")))
}

/// Per-site opt-out, e.g. `ARC_SEAM_SWIGLU=0`. Only consulted when the master
/// switch is on.
fn seam_site_enabled(var: &'static str) -> bool {
    // Sites are few and fixed; a tiny linear scan of cached values beats a
    // per-call `std::env::var` (which takes a global lock).
    static CACHE: std::sync::OnceLock<Vec<(&'static str, bool)>> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        ["ARC_SEAM_SWIGLU", "ARC_SEAM_WSUM", "ARC_SEAM_GATE"]
            .into_iter()
            .map(|v| (v, !matches!(std::env::var(v).as_deref(), Ok("0"))))
            .collect()
    });
    cache
        .iter()
        .find(|(name, _)| *name == var)
        .map(|(_, on)| *on)
        .unwrap_or(true)
}

/// Whether a seam site should fuse: master on, this site on, CUDA build.
pub fn seam_on(var: &'static str) -> bool {
    cfg!(feature = "cuda") && seam_enabled() && seam_site_enabled(var)
}

/// `ARC_SEAM_AB=1` recomputes the eager chain beside every seam kernel and
/// compares raw bits — at **F32**, before any narrowing, because a BF16
/// comparison has 8 mantissa bits and would swallow the reassociation errors
/// this is meant to catch. Shares `ARC_HC_AB_POISON=1` as its negative control.
pub fn seam_ab_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("ARC_SEAM_AB").as_deref(), Ok("1")))
}

static SEAM_ENGAGED: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static SEAM_DECLINED: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Site indices into the counters above.
pub const SEAM_SWIGLU: usize = 0;
pub const SEAM_WSUM: usize = 1;
pub const SEAM_GATE: usize = 2;
pub const SEAM_SWIGLU_SPLIT: usize = 3;

const SEAM_NAMES: [&str; 4] = [
    "swiglu_clamp",
    "moe_weighted_sum",
    "gate_renorm",
    "swiglu_clamp_split",
];

/// Record that a seam site fused. Logs the first one and then periodic totals.
pub fn seam_engaged(site: usize) {
    let prev = SEAM_ENGAGED[site].fetch_add(1, Ordering::Relaxed);
    if prev == 0 {
        tracing::warn!("arc-f32-seam ENGAGED site={}", SEAM_NAMES[site]);
    } else if (prev + 1) % 10_000 == 0 {
        tracing::warn!(
            "arc-f32-seam site={} engaged={} declined={}",
            SEAM_NAMES[site],
            prev + 1,
            SEAM_DECLINED[site].load(Ordering::Relaxed)
        );
    }
}

/// Record that a seam site fell back, and why. Logs the first one per site.
pub fn seam_declined(site: usize, reason: &str) {
    let prev = SEAM_DECLINED[site].fetch_add(1, Ordering::Relaxed);
    if prev == 0 {
        tracing::warn!(
            "arc-f32-seam DECLINED site={} reason={reason}",
            SEAM_NAMES[site]
        );
    }
}

/// `(engaged, declined)` totals for a site — used by tests and by the report.
pub fn seam_counts(site: usize) -> (u64, u64) {
    (
        SEAM_ENGAGED[site].load(Ordering::Relaxed),
        SEAM_DECLINED[site].load(Ordering::Relaxed),
    )
}

// ---------------------------------------------------------------------------
// The on-GPU bitwise A/B — `ARC_HC_AB=1`
// ---------------------------------------------------------------------------
//
// hc_fused.cu's header has cited "the final proof is the on-GPU A/B:
// `ARC_HC_AB=1`" since the file was written. That A/B DID NOT EXIST — the
// string appeared nowhere in the tree except that sentence. This is it.
//
// It recomputes the eager candle chain beside every fused kernel and compares
// the two BIT FOR BIT (not by tolerance: -0.0 vs 0.0 and NaN payloads are
// differences, and a f32 subtraction would hide both).
//
// `ARC_HC_AB_POISON=1` perturbs the fused tensor by exactly one ULP before the
// comparison, so the check MUST go red. A guard that has never been observed
// failing is not evidence of anything — this codebase has shipped fifteen that
// passed on broken code, including the `#if defined(__USE_FAST_MATH__)` in
// hc_fused.cu, which nvcc can never define (measured; see that file's header).
use std::sync::atomic::{AtomicU64, Ordering};

static AB_CMPS: AtomicU64 = AtomicU64::new(0);
static AB_BAD_TENSORS: AtomicU64 = AtomicU64::new(0);
static AB_BAD_ELEMS: AtomicU64 = AtomicU64::new(0);

pub fn ab_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("ARC_HC_AB").as_deref(), Ok("1")))
}

fn ab_poison() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("ARC_HC_AB_POISON").as_deref(), Ok("1")))
}

/// Raw bit patterns of a tensor, widened to u32 so BF16 and F32 share a path.
fn bits_of(t: &candle_core::Tensor) -> candle_core::Result<Vec<u32>> {
    let t = t.flatten_all()?.contiguous()?;
    Ok(match t.dtype() {
        candle_core::DType::F32 => t.to_vec1::<f32>()?.iter().map(|v| v.to_bits()).collect(),
        candle_core::DType::BF16 => t
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|v| u32::from(v.to_bits()))
            .collect(),
        other => candle_core::bail!("hc_fused ab_check: unsupported dtype {other:?}"),
    })
}

/// Compare a fused result against the eager one, bit for bit.
///
/// Returns the number of mismatching elements. Logs the first mismatch loudly
/// and emits a periodic running total — the periodic line is deliberate: a
/// silent A/B is indistinguishable from an A/B that never ran, which is the
/// single most common way a green result in this repo has been wrong.
pub fn ab_check(
    site: &str,
    fused: &candle_core::Tensor,
    eager: &candle_core::Tensor,
) -> candle_core::Result<u64> {
    if fused.dims() != eager.dims() {
        tracing::error!(
            "HC_AB[{site}] SHAPE MISMATCH fused={:?} eager={:?}",
            fused.dims(),
            eager.dims()
        );
        AB_BAD_TENSORS.fetch_add(1, Ordering::Relaxed);
        return Ok(u64::MAX);
    }
    let mut fb = bits_of(fused)?;
    let eb = bits_of(eager)?;

    if ab_poison() && !fb.is_empty() {
        // Exactly one ULP up on the first element. If the comparison below
        // still reports zero mismatches, the comparison is broken, not the
        // kernel.
        fb[0] = fb[0].wrapping_add(1);
    }

    let mut bad = 0u64;
    let mut first: Option<(usize, u32, u32)> = None;
    for (i, (a, b)) in fb.iter().zip(eb.iter()).enumerate() {
        if a != b {
            bad += 1;
            if first.is_none() {
                first = Some((i, *a, *b));
            }
        }
    }

    let n = AB_CMPS.fetch_add(1, Ordering::Relaxed) + 1;
    if bad > 0 {
        AB_BAD_TENSORS.fetch_add(1, Ordering::Relaxed);
        AB_BAD_ELEMS.fetch_add(bad, Ordering::Relaxed);
        if let Some((i, a, b)) = first {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::error!(
                    "HC_AB[{site}] BITWISE MISMATCH: {bad}/{} elements differ; \
                     first at [{i}] fused=0x{a:08x} eager=0x{b:08x}",
                    fb.len()
                );
            });
        }
    }
    if n % 1000 == 0 {
        tracing::info!(
            "HC_AB: {n} comparisons, {} tensors with mismatches, {} bad elements \
             (poison={})",
            AB_BAD_TENSORS.load(Ordering::Relaxed),
            AB_BAD_ELEMS.load(Ordering::Relaxed),
            ab_poison()
        );
    }
    Ok(bad)
}

/// Running totals, for an end-of-run assertion.
pub fn ab_totals() -> (u64, u64, u64) {
    (
        AB_CMPS.load(Ordering::Relaxed),
        AB_BAD_TENSORS.load(Ordering::Relaxed),
        AB_BAD_ELEMS.load(Ordering::Relaxed),
    )
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

    /// The **candle op chain** of the `hc_pre` tail:
    /// `pre.unsqueeze(-1).broadcast_mul(x_f32).sum(1).to_dtype(BF16)`.
    ///
    /// `x_f32` is `[hc][h]` row-major for one row `n`. Returns the BF16 bit
    /// patterns, because the narrowing is part of what has to match.
    /// F32 result, BEFORE the narrowing cast. The order assertions have to be
    /// made here: BF16 keeps 8 mantissa bits, so narrowing discards almost
    /// every reassociation difference and a bf16-only comparison would pass on
    /// a kernel summing in the wrong order. (It did — the vacuity test below
    /// caught exactly that and this split is the fix.)
    pub(crate) fn hc_y_combine_candle_replay_f32(
        x_f32: &[f32],
        pre: &[f32],
        hc: usize,
        h: usize,
    ) -> Vec<f32> {
        (0..h)
            .map(|col| {
                // broadcast_mul materialises [hc, h] first...
                let prods: Vec<f32> = (0..hc).map(|t| pre[t] * x_f32[t * h + col]).collect();
                // ...then `sum(1)` reduces the hc axis with block_dim =
                // next_power_of_two(hc), which is candle_fast_sum's tree.
                candle_fast_sum(&prods, super::candle_reduce_block_dim(hc))
            })
            .collect()
    }

    /// The **fused kernel** `hc_y_combine_kernel`, transcribed, before narrowing.
    pub(crate) fn hc_y_combine_fused_replay_f32(
        x_f32: &[f32],
        pre: &[f32],
        hc: usize,
        h: usize,
    ) -> Vec<f32> {
        (0..h)
            .map(|col| {
                let mut acc = vec![0.0f32; hc];
                for (t, a) in acc.iter_mut().enumerate() {
                    // the `shr[tid] = 0` start is not a no-op for -0.0f
                    *a = 0.0f32 + pre[t] * x_f32[t * h + col];
                }
                let mut s = hc / 2;
                while s > 0 {
                    for t in 0..s {
                        acc[t] += acc[t + s];
                    }
                    s /= 2;
                }
                acc[0]
            })
            .collect()
    }

    /// Sequential `a0+a1+a2+a3` — what the kernel must NOT do. Exists only so
    /// the vacuity test can show the comparison can tell the orders apart.
    pub(crate) fn hc_y_combine_sequential_replay_f32(
        x_f32: &[f32],
        pre: &[f32],
        hc: usize,
        h: usize,
    ) -> Vec<f32> {
        (0..h)
            .map(|col| {
                let mut s = 0.0f32;
                for t in 0..hc {
                    s += pre[t] * x_f32[t * h + col];
                }
                s
            })
            .collect()
    }

    pub(crate) fn hc_y_combine_candle_replay(
        x_f32: &[f32],
        pre: &[f32],
        hc: usize,
        h: usize,
    ) -> Vec<u16> {
        hc_y_combine_candle_replay_f32(x_f32, pre, hc, h)
            .into_iter()
            .map(|s| half::bf16::from_f32(s).to_bits())
            .collect()
    }

    /// The **fused kernel** `hc_y_combine_kernel`, transcribed.
    pub(crate) fn hc_y_combine_fused_replay(
        x_f32: &[f32],
        pre: &[f32],
        hc: usize,
        h: usize,
    ) -> Vec<u16> {
        hc_y_combine_fused_replay_f32(x_f32, pre, hc, h)
            .into_iter()
            .map(|s| half::bf16::from_f32(s).to_bits())
            .collect()
    }

    /// The **candle op chain** of `hc_post`, for one row `n`.
    /// `x` is `[h]`, `residual` is `[hc][h]`, both already widened to f32.
    pub(crate) fn hc_post_candle_replay(
        x: &[f32],
        residual: &[f32],
        post: &[f32],
        comb: &[f32],
        hc: usize,
        h: usize,
    ) -> Vec<u16> {
        let mut out = Vec::with_capacity(hc * h);
        for k in 0..hc {
            for (col, &xv) in x.iter().enumerate().take(h) {
                let term1 = post[k] * xv;
                // term2 = sum_j comb[j][k] * residual[j][col] — the K=hc GEMM.
                let mut term2 = 0.0f32;
                for j in 0..hc {
                    term2 = comb[j * hc + k].mul_add(residual[j * h + col], term2);
                }
                out.push(half::bf16::from_f32(term1 + term2).to_bits());
            }
        }
        out
    }

    /// The **fused kernel** `hc_post_fused_kernel`, transcribed.
    pub(crate) fn hc_post_fused_replay(
        x: &[f32],
        residual: &[f32],
        post: &[f32],
        comb: &[f32],
        hc: usize,
        h: usize,
    ) -> Vec<u16> {
        let mut out = vec![0u16; hc * h];
        for k in 0..hc {
            for col in 0..h {
                let t1 = post[k] * x[col];
                let mut acc = 0.0f32;
                for j in 0..hc {
                    acc = comb[j * hc + k].mul_add(residual[j * h + col], acc);
                }
                out[k * h + col] = half::bf16::from_f32(t1 + acc).to_bits();
            }
        }
        out
    }
}

#[cfg(test)]
mod seam_tests {
    use super::*;

    /// Scalar replica of `arc_seam_moe_weighted_sum_kernel` /
    /// `arc_seam_gate_renorm_kernel`'s reduction: the identity-padded pairwise
    /// tree over `P = next_pow2(K)` slots.
    fn seam_padded_tree(vals: &[f32]) -> f32 {
        let p = vals.len().next_power_of_two();
        let mut acc = vec![0.0f32; p];
        for (t, v) in vals.iter().enumerate() {
            acc[t] = 0.0f32 + v; // the `shr[tid] = 0; shr[tid] += v` of fast_sum
        }
        let mut s = p / 2;
        while s > 0 {
            for t in 0..s {
                acc[t] += acc[t + s];
            }
            s >>= 1;
        }
        acc[0]
    }

    /// The seam kernels must reduce in candle's `fast_sum` order, which for
    /// V4's `top_k = 6` is a tree over EIGHT slots, not six. This is the whole
    /// bit-identity risk of the two reduction kernels, and it is checkable
    /// without a GPU: `sinkhorn::reference::candle_tree_sum` is an independent
    /// transcription of the same candle kernel, written for a different fusion.
    #[test]
    fn seam_reduction_replays_candle_fast_sum_including_the_padding_lanes() {
        let mut seed = 0x5eedu64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((seed >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
        };
        for _ in 0..2000 {
            for k in [1usize, 2, 4, 6, 8] {
                let vals: Vec<f32> = (0..k).map(|_| next()).collect();
                assert_eq!(
                    seam_padded_tree(&vals).to_bits(),
                    crate::cuda::sinkhorn::reference::candle_tree_sum(&vals).to_bits(),
                    "k={k} vals={vals:?}"
                );
            }
        }
    }

    /// For `k = 6` the padded tree is NOT the sequential sum, and this pins the
    /// exact association the kernel must reproduce. A kernel that summed
    /// `a0..a5` left to right would pass a tolerance test and fail this one.
    #[test]
    fn k6_tree_is_not_the_sequential_sum() {
        // The tree pairs a0 with a4, so the two large opposite-sign terms
        // cancel exactly there while a left-to-right sum absorbs the small
        // terms into 1e8 and loses them. 1.0 vs 4.0 — not a last-ulp
        // difference, a catastrophic one.
        let v = [1e8f32, 1.0, 1.0, 1.0, -1e8, 1.0];
        let seq = v.iter().fold(0.0f32, |a, b| a + b);
        let tree = seam_padded_tree(&v);
        assert_ne!(
            seq.to_bits(),
            tree.to_bits(),
            "if these ever agree this test has stopped discriminating"
        );
        // ((0+a0)+(0+a4)) + ((0+a2)+0)  +  ((0+a1)+(0+a5)) + ((0+a3)+0)
        let lhs = (v[0] + v[4]) + v[2];
        let rhs = (v[1] + v[5]) + v[3];
        assert_eq!(tree.to_bits(), (lhs + rhs).to_bits());
    }

    /// `fmaf(x, scale, 0.0)` is what candle's `affine(mul, 0.)` compiles to and
    /// is deliberately not `x * scale`: they disagree on the sign of zero, and
    /// the gate renorm kernel writes the result of exactly this expression.
    #[test]
    fn affine_is_an_fma_not_a_multiply() {
        let x = -0.0f32;
        assert_ne!(
            (x * 2.0f32).to_bits(),
            x.mul_add(2.0, 0.0).to_bits(),
            "the two spellings must still differ, or this guard proves nothing"
        );
        assert_eq!(x.mul_add(2.0, 0.0).to_bits(), 0.0f32.to_bits());
    }

    /// Engagement accounting must actually count, in both directions —
    /// otherwise "no DECLINED line in the log" is not evidence of anything.
    #[test]
    fn seam_engagement_counters_move() {
        let before = seam_counts(SEAM_GATE);
        seam_engaged(SEAM_GATE);
        seam_declined(SEAM_GATE, "unit test");
        let after = seam_counts(SEAM_GATE);
        assert_eq!(after.0, before.0 + 1);
        assert_eq!(after.1, before.1 + 1);
    }

    /// The seam kernels must stay in the no-fast-math translation unit, and the
    /// shared-expert site must stay on the weaker fusion for as long as
    /// `fused_glu` is compiled with `--use_fast_math`. If that flag ever leaves
    /// mistralrs-quant/build.rs, the full fusion becomes legal there and this
    /// test is the reminder.
    #[test]
    fn seam_kernels_are_where_the_bit_identity_argument_says_they_are() {
        let cu = include_str!("hc_fused.cu");
        for sym in [
            "arc_seam_swiglu_clamp_kernel",
            "arc_seam_swiglu_clamp_split_kernel",
            "arc_seam_moe_weighted_sum_kernel",
            "arc_seam_gate_renorm_kernel",
        ] {
            assert!(cu.contains(sym), "{sym} left hc_fused.cu");
        }
        let build = include_str!("../../build.rs");
        assert!(
            build.contains("src/cuda/hc_fused.cu"),
            "hc_fused.cu must stay in the dedicated no-fast-math builder"
        );
        let quant_build = include_str!("../../../mistralrs-quant/build.rs");
        assert!(
            quant_build.contains("--use_fast_math"),
            "if mistralrs-quant stopped using fast math, fused_glu became \
             reproducible and the shared expert can take the full fusion"
        );
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

    /// Source-level tripwires, mirroring `sinkhorn::tests`: the IEEE intrinsics
    /// must stay in hc_fused.cu, fast-math approximations must stay out, and
    /// build.rs must keep the file out of the `--use_fast_math` glob and in the
    /// `--fmad=false` builder. Getting this wiring wrong is silent — the kernel
    /// still runs, it just stops being bit-identical.
    ///
    /// NOTE ON WHAT ACTUALLY PROTECTS THIS FILE. The `#if
    /// defined(__USE_FAST_MATH__)` `#error` in hc_fused.cu (and in sinkhorn.cu)
    /// is DECORATIVE: nvcc 12.4 defines neither `__USE_FAST_MATH__` nor
    /// `__FAST_MATH__` in either the host or the device pass, so it cannot fire.
    /// Measured, not assumed —
    ///     nvcc --use_fast_math -E -dM x.cu | grep -i fast   -> no output
    ///     nvcc --use_fast_math -arch=sm_90 -c guard.cu      -> compiles clean
    /// while `--use_fast_math` does reach the device pass (a float divide drops
    /// from 3 MUFU/RCP instructions to 1). The real protection is the build.rs
    /// wiring asserted below, plus the runtime `ARC_HC_AB=1` bitwise A/B, which
    /// catches a mis-compiled build on the first token. The presence assertion
    /// is kept only so the comment above the guard travels with it.
    #[test]
    fn kernel_source_and_build_wiring_guards() {
        let cu = include_str!("hc_fused.cu");
        assert!(
            cu.contains("#if defined(__USE_FAST_MATH__)") && cu.contains("#error"),
            "hc_fused.cu lost its (decorative) fast-math #error guard"
        );
        assert!(
            cu.contains("THE COMPILE-TIME FAST-MATH GUARD BELOW IS DEAD"),
            "hc_fused.cu lost the note recording that the #error guard cannot fire; \
             without it the next reader will trust a guard that has never been able to fail"
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

    /// Inputs shaped like V4 decode: hc_mult = 4, one row, a slice of hidden.
    fn y_combine_inputs(h: usize) -> (Vec<f32>, Vec<f32>) {
        let hc = 4;
        let mut r = Lcg(0x5eed_1234);
        let x: Vec<f32> = (0..hc * h).map(|_| r.next_f32(-4.0, 4.0)).collect();
        let pre: Vec<f32> = (0..hc).map(|_| r.next_f32(0.0, 1.0)).collect();
        (x, pre)
    }

    #[test]
    fn hc_y_combine_fused_is_bit_identical_to_candle_chain() {
        let (hc, h) = (4usize, 257usize);
        let (x, pre) = y_combine_inputs(h);
        // Assert at F32, before the narrowing: BF16 keeps 8 mantissa bits and
        // would hide an order difference (see the vacuity test below).
        assert_eq!(
            hc_y_combine_candle_replay_f32(&x, &pre, hc, h),
            hc_y_combine_fused_replay_f32(&x, &pre, hc, h),
            "fused y-combine diverged from the candle sum(1) chain in F32"
        );
        // ...and again after it, which is what the kernel actually stores.
        assert_eq!(
            hc_y_combine_candle_replay(&x, &pre, hc, h),
            hc_y_combine_fused_replay(&x, &pre, hc, h),
            "fused y-combine diverged after the BF16 narrowing"
        );
    }

    /// The pairwise tree is the whole point: candle's `sum(1)` over hc = 4 gives
    /// `(a0+a2)+(a1+a3)`, not `((a0+a1)+a2)+a3`.
    ///
    /// This test previously compared the two orders AFTER narrowing to BF16 and
    /// FAILED — not because the kernel was wrong, but because 8 mantissa bits
    /// swallow essentially every reassociation difference, which meant the
    /// bit-identity test above would have passed on a kernel summing in the
    /// wrong order. That is the failure this vacuity check exists to expose, and
    /// the fix is to make the claim where it is observable: at F32.
    #[test]
    fn hc_y_combine_tree_order_is_not_vacuous() {
        let (hc, h) = (4usize, 257usize);
        let (x, pre) = y_combine_inputs(h);
        let tree = hc_y_combine_candle_replay_f32(&x, &pre, hc, h);
        let sequential = hc_y_combine_sequential_replay_f32(&x, &pre, hc, h);
        let differing = tree
            .iter()
            .zip(sequential.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(
            differing > 0,
            "sequential summation produced bit-identical F32 output on all {h} elements, so \
             the tree-order assertion proves nothing on this data"
        );
    }

    fn post_inputs(h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let hc = 4;
        let mut r = Lcg(0xabcd_ef01);
        let x: Vec<f32> = (0..h).map(|_| r.next_f32(-3.0, 3.0)).collect();
        let residual: Vec<f32> = (0..hc * h).map(|_| r.next_f32(-3.0, 3.0)).collect();
        let post: Vec<f32> = (0..hc).map(|_| r.next_f32(0.0, 2.0)).collect();
        let comb: Vec<f32> = (0..hc * hc).map(|_| r.next_f32(0.0, 1.0)).collect();
        (x, residual, post, comb)
    }

    #[test]
    fn hc_post_fused_is_bit_identical_to_candle_chain() {
        let (hc, h) = (4usize, 193usize);
        let (x, residual, post, comb) = post_inputs(h);
        let a = hc_post_candle_replay(&x, &residual, &post, &comb, hc, h);
        let b = hc_post_fused_replay(&x, &residual, &post, &comb, hc, h);
        assert_eq!(a, b, "fused hc_post diverged from the candle term1+term2 chain");
    }

    /// `term2` transposes `comb` before the matmul. Reading `comb[k][j]` instead
    /// of `comb[j][k]` is the single easiest way to get this kernel wrong while
    /// keeping every shape valid, so prove the comparison would catch it.
    #[test]
    fn hc_post_guard_catches_a_transposed_comb() {
        let (hc, h) = (4usize, 193usize);
        let (x, residual, post, comb) = post_inputs(h);
        let reference = hc_post_candle_replay(&x, &residual, &post, &comb, hc, h);
        let mut transposed = vec![0.0f32; hc * hc];
        for j in 0..hc {
            for k in 0..hc {
                transposed[j * hc + k] = comb[k * hc + j];
            }
        }
        let mutated = hc_post_fused_replay(&x, &residual, &post, &transposed, hc, h);
        assert!(
            reference.iter().zip(mutated.iter()).any(|(a, b)| a != b),
            "a transposed comb produced bit-identical output, so the hc_post assertion is vacuous"
        );
    }
}
