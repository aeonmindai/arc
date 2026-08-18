//! Parent system: ArcQuant / TurboQuant — fused block-wise E4M3 activation
//! quantizer for the DeepSeek-V4 fused MQA key cache (ArcInfer / ArcKV / Fp8).
//!
//! One launch each for quantize and dequantize, replacing the ~11 and ~13
//! candle ops the caller used to issue — and, critically, replacing the
//! device→host→device round trip that `dsv4_kv_fp8::e4m3_codes_cpu` performs
//! because candle has no CUDA `F8E4M3` cast. That round trip is 43 blocking
//! `cuMemcpyDtoHAsync_v2` per V4 decode token and it makes CUDA graph capture
//! impossible: a graph cannot record a blocking D2H.
//!
//! The byte format is unchanged — same `codes`/`side` layout, same bits. See
//! `kernels/arc_kvquant/arc_kvquant.cu` for how bit-parity with the CPU path is
//! obtained, and for the deliberate mutant used to prove the parity test can
//! fail.

#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::{
        cuda_backend::{cudarc::driver::DeviceRepr, CudaDType},
        CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
    };
    use core::ffi::c_void;
    use half::{bf16, f16};

    use crate::utils::slice_ptr;

    /// `dtype` discriminant shared with the C ABI.
    fn dtype_id(dt: DType) -> Result<u32> {
        match dt {
            DType::F16 => Ok(0),
            DType::BF16 => Ok(1),
            DType::F32 => Ok(2),
            other => candle_core::bail!("arc-kvquant: unsupported activation dtype {other:?}"),
        }
    }

    /// Geometry shared by both kernels, validated once.
    struct Geom {
        b: usize,
        h: usize,
        t: usize,
        head_dim: usize,
        nope: usize,
        rope_dim: usize,
        n_blocks: usize,
        block_w: usize,
    }

    impl Geom {
        fn ntok(&self) -> usize {
            self.b * self.h * self.t
        }
        fn side_w(&self) -> usize {
            self.rope_dim + self.n_blocks
        }
    }

    fn cuda_device(t: &Tensor, what: &str) -> Result<CudaDevice> {
        match t.device() {
            Device::Cuda(d) => Ok(d.clone()),
            _ => candle_core::bail!("arc-kvquant: {what} must live on CUDA"),
        }
    }

    /// Whether the fused kernels can serve `dev`. Callers fall back to the
    /// candle op chain when this is false rather than failing.
    pub fn kv_fp8_fused_available(dev: &Device) -> bool {
        matches!(dev, Device::Cuda(_))
    }

    fn geometry(k: &Tensor, rope_dim: usize, block_w: usize) -> Result<Geom> {
        let (b, h, t, head_dim) = k.dims4()?;
        if rope_dim > head_dim {
            candle_core::bail!("arc-kvquant: rope_dim {rope_dim} > head_dim {head_dim}");
        }
        let nope = head_dim - rope_dim;
        if block_w == 0 || nope == 0 || nope % block_w != 0 {
            candle_core::bail!(
                "arc-kvquant: nope {nope} is not a whole number of {block_w}-wide blocks"
            );
        }
        Ok(Geom {
            b,
            h,
            t,
            head_dim,
            nope,
            rope_dim,
            n_blocks: nope / block_w,
            block_w,
        })
    }

    /// Fused block-wise E4M3 quantize of `k`'s non-RoPE dims.
    ///
    /// `k` is `[B, H, T, head_dim]` in the activation dtype. Returns
    /// `(codes, side)`:
    ///
    /// * `codes` — `[B, H, T, head_dim - rope_dim]` U8, one E4M3 code per dim;
    /// * `side`  — `[B, H, T, rope_dim + n_blocks]` in `k`'s dtype: the RoPE'd
    ///   tail verbatim followed by each 64-wide block's `amax`.
    ///
    /// Bit-identical to `dsv4_kv_fp8::quantize_k` under `KvQuantMode::CpuExact`.
    /// That is the contract; `kv_fp8_fused_is_bit_identical_to_cpu_exact` pins
    /// it on hardware, and the mutant below proves that test can fail.
    pub fn kv_fp8_quantize(
        k: &Tensor,
        rope_dim: usize,
        block_w: usize,
    ) -> Result<(Tensor, Tensor)> {
        quantize_inner(k, rope_dim, block_w, false)
    }

    /// D33 negative control: identical to [`kv_fp8_quantize`] except the E4M3
    /// rounding truncates instead of rounding to nearest even. Exists purely so
    /// the parity test can be shown to fail. Nothing in the serving path calls
    /// it.
    pub fn kv_fp8_quantize_mutant_for_test(
        k: &Tensor,
        rope_dim: usize,
        block_w: usize,
    ) -> Result<(Tensor, Tensor)> {
        quantize_inner(k, rope_dim, block_w, true)
    }

    fn quantize_inner(
        k: &Tensor,
        rope_dim: usize,
        block_w: usize,
        mutant: bool,
    ) -> Result<(Tensor, Tensor)> {
        let g = geometry(k, rope_dim, block_w)?;
        let dev = cuda_device(k, "k")?;
        let did = dtype_id(k.dtype())?;
        let k = k.contiguous()?;
        match k.dtype() {
            DType::F16 => quantize_t::<f16>(&k, &g, &dev, did, mutant),
            DType::BF16 => quantize_t::<bf16>(&k, &g, &dev, did, mutant),
            DType::F32 => quantize_t::<f32>(&k, &g, &dev, did, mutant),
            other => candle_core::bail!("arc-kvquant: unsupported activation dtype {other:?}"),
        }
    }

    fn quantize_t<T: CudaDType + DeviceRepr>(
        k: &Tensor,
        g: &Geom,
        dev: &CudaDevice,
        did: u32,
        mutant: bool,
    ) -> Result<(Tensor, Tensor)> {
        let ntok = g.ntok();
        // `alloc` rather than `alloc_zeros`: both outputs are fully written, and
        // the memset would be one more device op in the exact place where op
        // count is the disease.
        let codes_buf = unsafe { dev.alloc::<u8>(ntok * g.nope)? };
        let side_buf = unsafe { dev.alloc::<T>(ntok * g.side_w())? };

        let (k_storage, k_layout) = k.storage_and_layout();
        let k_s = match &*k_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("arc-kvquant: k must be CUDA storage"),
        };
        let (k_ptr, _k_guard) = slice_ptr(k_s.as_cuda_slice::<T>()?, k_layout.start_offset());
        let (codes_ptr, _codes_guard) = slice_ptr(&codes_buf, 0);
        let (side_ptr, _side_guard) = slice_ptr(&side_buf, 0);

        let launch = if mutant {
            super::ffi::arc_kv_fp8_quantize_mutant
        } else {
            super::ffi::arc_kv_fp8_quantize
        };
        unsafe {
            launch(
                k_ptr as *const c_void,
                codes_ptr as *mut u8,
                side_ptr as *mut c_void,
                g.head_dim as i32,
                g.nope as i32,
                g.rope_dim as i32,
                g.n_blocks as i32,
                g.block_w as i32,
                ntok as i64,
                dev.cuda_stream().cu_stream() as *mut c_void,
                did,
            )
        };

        drop(_k_guard);
        drop(_codes_guard);
        drop(_side_guard);
        drop(k_storage);

        let codes = Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(codes_buf, dev.clone())),
            Shape::from_dims(&[g.b, g.h, g.t, g.nope]),
        ));
        let side = Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(side_buf, dev.clone())),
            Shape::from_dims(&[g.b, g.h, g.t, g.side_w()]),
        ));
        Ok((codes, side))
    }

    /// Fused dequantize: `codes` + `side` back to `[B, H, T, head_dim]` in
    /// `side`'s dtype.
    ///
    /// `lut` is the 256-entry F32 table built from
    /// `F8E4M3::from_bits(i).to_f32()` — the *same* tensor the candle path fed
    /// to `index_select`, which is what makes the code→value half of the round
    /// trip bit-exact by construction rather than by argument.
    pub fn kv_fp8_dequantize(
        codes: &Tensor,
        side: &Tensor,
        lut: &Tensor,
        rope_dim: usize,
        block_w: usize,
    ) -> Result<Tensor> {
        let (b, h, t, nope) = codes.dims4()?;
        let (sb, sh, st, side_w) = side.dims4()?;
        if (sb, sh, st) != (b, h, t) {
            candle_core::bail!(
                "arc-kvquant: side dims {:?} do not match codes {:?}",
                side.dims(),
                codes.dims()
            );
        }
        if block_w == 0 || nope == 0 || nope % block_w != 0 {
            candle_core::bail!("arc-kvquant: nope {nope} not a multiple of block_w {block_w}");
        }
        let n_blocks = nope / block_w;
        if side_w != rope_dim + n_blocks {
            candle_core::bail!(
                "arc-kvquant: side width {side_w} != rope_dim {rope_dim} + n_blocks {n_blocks}"
            );
        }
        if codes.dtype() != DType::U8 {
            candle_core::bail!("arc-kvquant: codes must be U8, got {:?}", codes.dtype());
        }
        if lut.dtype() != DType::F32 || lut.elem_count() != 256 {
            candle_core::bail!("arc-kvquant: lut must be 256 F32 entries");
        }
        let g = Geom {
            b,
            h,
            t,
            head_dim: nope + rope_dim,
            nope,
            rope_dim,
            n_blocks,
            block_w,
        };
        let dev = cuda_device(side, "side")?;
        let did = dtype_id(side.dtype())?;
        // A `narrow` on the sequence dim leaves a contiguous view (with a start
        // offset) whenever B*H == 1 — the decode case this exists for — so
        // `contiguous()` is free there. For B*H > 1 it materialises the window,
        // which is what threading 4-D strides into the kernel would avoid; that
        // is a batch-path follow-up, not a correctness gap.
        let codes = codes.contiguous()?;
        let side = side.contiguous()?;
        let lut = lut.contiguous()?;
        match side.dtype() {
            DType::F16 => dequantize_t::<f16>(&codes, &side, &lut, &g, &dev, did),
            DType::BF16 => dequantize_t::<bf16>(&codes, &side, &lut, &g, &dev, did),
            DType::F32 => dequantize_t::<f32>(&codes, &side, &lut, &g, &dev, did),
            other => candle_core::bail!("arc-kvquant: unsupported activation dtype {other:?}"),
        }
    }

    fn dequantize_t<T: CudaDType + DeviceRepr>(
        codes: &Tensor,
        side: &Tensor,
        lut: &Tensor,
        g: &Geom,
        dev: &CudaDevice,
        did: u32,
    ) -> Result<Tensor> {
        let ntok = g.ntok();
        let out_buf = unsafe { dev.alloc::<T>(ntok * g.head_dim)? };

        let (codes_storage, codes_layout) = codes.storage_and_layout();
        let codes_s = match &*codes_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("arc-kvquant: codes must be CUDA storage"),
        };
        let (side_storage, side_layout) = side.storage_and_layout();
        let side_s = match &*side_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("arc-kvquant: side must be CUDA storage"),
        };
        let (lut_storage, lut_layout) = lut.storage_and_layout();
        let lut_s = match &*lut_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("arc-kvquant: lut must be CUDA storage"),
        };

        let (codes_ptr, _codes_guard) =
            slice_ptr(codes_s.as_cuda_slice::<u8>()?, codes_layout.start_offset());
        let (side_ptr, _side_guard) =
            slice_ptr(side_s.as_cuda_slice::<T>()?, side_layout.start_offset());
        let (lut_ptr, _lut_guard) =
            slice_ptr(lut_s.as_cuda_slice::<f32>()?, lut_layout.start_offset());
        let (out_ptr, _out_guard) = slice_ptr(&out_buf, 0);

        unsafe {
            super::ffi::arc_kv_fp8_dequantize(
                codes_ptr as *const u8,
                side_ptr as *const c_void,
                lut_ptr as *const f32,
                out_ptr as *mut c_void,
                g.head_dim as i32,
                g.nope as i32,
                g.rope_dim as i32,
                g.n_blocks as i32,
                g.block_w as i32,
                ntok as i64,
                dev.cuda_stream().cu_stream() as *mut c_void,
                did,
            )
        };

        drop(_codes_guard);
        drop(_side_guard);
        drop(_lut_guard);
        drop(_out_guard);
        drop(codes_storage);
        drop(side_storage);
        drop(lut_storage);

        Ok(Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out_buf, dev.clone())),
            Shape::from_dims(&[g.b, g.h, g.t, g.head_dim]),
        )))
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

#[cfg(not(feature = "cuda"))]
mod stub {
    use candle_core::{Device, Result, Tensor};

    pub fn kv_fp8_quantize(
        _k: &Tensor,
        _rope_dim: usize,
        _block_w: usize,
    ) -> Result<(Tensor, Tensor)> {
        candle_core::bail!("arc-kvquant: fused FP8 KV quantize requires the `cuda` feature")
    }

    pub fn kv_fp8_quantize_mutant_for_test(
        _k: &Tensor,
        _rope_dim: usize,
        _block_w: usize,
    ) -> Result<(Tensor, Tensor)> {
        candle_core::bail!("arc-kvquant: fused FP8 KV quantize requires the `cuda` feature")
    }

    pub fn kv_fp8_dequantize(
        _codes: &Tensor,
        _side: &Tensor,
        _lut: &Tensor,
        _rope_dim: usize,
        _block_w: usize,
    ) -> Result<Tensor> {
        candle_core::bail!("arc-kvquant: fused FP8 KV dequantize requires the `cuda` feature")
    }

    /// Never available without the `cuda` feature.
    pub fn kv_fp8_fused_available(_dev: &Device) -> bool {
        false
    }
}

#[cfg(not(feature = "cuda"))]
pub use stub::*;
