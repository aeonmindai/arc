/// Fused Sinkhorn normalization for V4 mHC.
///
/// Replaces the ~123-launch candle op chain in `sinkhorn_normalize`
/// (models/dsv4_mhc.rs) with a single CUDA kernel. Input is `[N, hc, hc]` F32;
/// output is the same shape.
///
/// Bit-identity contract: sinkhorn.cu mirrors the candle CUDA backend
/// op-for-op — candle's `fast_sum`/`fast_max` pairwise-tree reduction order,
/// unfused IEEE round-to-nearest arithmetic (the file is compiled WITHOUT
/// `--use_fast_math` by a dedicated builder in build.rs), and the reference's
/// exact eps placement. See the contract comment at the top of sinkhorn.cu and
/// the [`reference`] module below, whose scalar replicas of both op chains are
/// asserted bit-identical in `mod tests`.
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
        candle::bail!(
            "sinkhorn_normalize_cuda: last two dims must be square, got [{n}, {hc}, {hc2}]"
        );
    }
    if hc > 16 {
        candle::bail!("sinkhorn_normalize_cuda: hc={hc} exceeds SINKHORN_MAX_HC=16");
    }
    if comb.dtype() != candle_core::DType::F32 {
        candle::bail!(
            "sinkhorn_normalize_cuda: input must be F32, got {:?}",
            comb.dtype()
        );
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

/// Scalar f32 replicas that pin the fused kernel's op ORDER and ROUNDING to
/// what the candle CUDA backend actually executes, bit for bit.
///
/// Why this exists: the first fused kernel failed the H200 A/B (ppl drift +
/// 4/6 greedy divergences) because it summed rows/columns *sequentially* and
/// was compiled under `--use_fast_math`. candle's `sum_keepdim`/`max_keepdim`
/// on CUDA instead run candle-kernels `fast_sum`/`fast_max`: block_dim =
/// `next_pow2(reduced_len)` threads, one element per thread (identity-padded),
/// then a pairwise tree `shr[t] op= shr[t + s]` for `s = block/2, ..., 1` —
/// for `hc = 4` the sum is `(a0+a2)+(a1+a3)`, which rounds differently from
/// `((a0+a1)+a2)+a3` in f32.
///
/// Two independent replicas are kept deliberately:
/// - [`reference::sinkhorn_candle_gpu_replay`] transcribes the *candle op
///   chain* of `models::dsv4_mhc::sinkhorn_normalize` (whole-tensor passes,
///   one per candle kernel launch);
/// - [`reference::sinkhorn_fused_kernel_replay`] transcribes *cuda/sinkhorn.cu*
///   (per-row register walk, shared-tile column pass).
///
/// `mod tests` asserts them bit-identical over randomized inputs, so any
/// future edit that changes op order in one of the three places (candle chain,
/// kernel, replicas) trips CPU CI without a GPU.
///
/// Known scalar-vs-GPU gap (documented, gated by the on-GPU A/B): these
/// replicas use Rust/libm `f32::exp`, while both GPU paths use CUDA libdevice
/// `__nv_expf`; the two may differ in the last ulp. That difference CANCELS in
/// the GPU A/B because the candle chain and the fused kernel call the same
/// `__nv_expf`. Similarly `f32::max` here is IEEE maxNum like CUDA `fmaxf`
/// (both ignore NaN); `-0.0`/`+0.0` max ordering is unspecified in both and
/// unreachable after `exp`. Everything else (add/sub/div, single rounding per
/// op, no FMA) is exact IEEE f32 on both sides — Rust scalar ops and the
/// kernel's `__f{add,sub,div}_rn` — so the bitwise assertions are meaningful.
#[allow(dead_code)] // consumed by `mod tests`; kept non-test for GPU-side debug harnesses
pub(crate) mod reference {
    fn next_pow2(v: usize) -> usize {
        let mut p = 1usize;
        while p < v {
            p <<= 1;
        }
        p
    }

    /// candle-kernels `fast_sum` order (reduce.cu): zero-initialized
    /// accumulators (`shr[tid] = 0; shr[tid] += v` — note this canonicalizes
    /// `-0.0` to `+0.0`), identity padding to the next power of two, then the
    /// pairwise tree.
    pub(crate) fn candle_tree_sum(vals: &[f32]) -> f32 {
        let p = next_pow2(vals.len());
        let mut buf = vec![0.0f32; p];
        for (t, v) in vals.iter().enumerate() {
            buf[t] = 0.0f32 + v;
        }
        let mut s = p / 2;
        while s > 0 {
            for t in 0..s {
                buf[t] += buf[t + s];
            }
            s /= 2;
        }
        buf[0]
    }

    /// candle-kernels `fast_max` order (reduce.cu): `-INF` init, `maxg` ==
    /// `fmaxf` (NaN-ignoring IEEE maxNum, same as Rust `f32::max`), pairwise
    /// tree.
    pub(crate) fn candle_tree_max(vals: &[f32]) -> f32 {
        let p = next_pow2(vals.len());
        let mut buf = vec![f32::NEG_INFINITY; p];
        for (t, v) in vals.iter().enumerate() {
            buf[t] = f32::NEG_INFINITY.max(*v);
        }
        let mut s = p / 2;
        while s > 0 {
            for t in 0..s {
                buf[t] = buf[t].max(buf[t + s]);
            }
            s /= 2;
        }
        buf[0]
    }

    /// Replays the exact candle CUDA op chain of
    /// `models::dsv4_mhc::sinkhorn_normalize` in scalar f32: one whole-tensor
    /// pass per candle kernel launch, tree-ordered reductions, one rounding
    /// per op. `x` is `[n, hc, hc]` row-major; `eps` is converted to f32 once,
    /// exactly like candle's `affine` (`T::from_f64`).
    #[allow(clippy::cast_possible_truncation)] // eps -> f32 mirrors candle T::from_f64
    pub(crate) fn sinkhorn_candle_gpu_replay(
        x: &[f32],
        n: usize,
        hc: usize,
        iters: usize,
        eps: f64,
    ) -> Vec<f32> {
        assert_eq!(x.len(), n * hc * hc);
        let eps = eps as f32;
        let mut x = x.to_vec();
        for b in 0..n {
            let base = b * hc * hc;
            let mat = &mut x[base..base + hc * hc];

            // max_keepdim(-1): fast_max per row.
            let row_max: Vec<f32> = (0..hc)
                .map(|i| candle_tree_max(&mat[i * hc..(i + 1) * hc]))
                .collect();
            // broadcast_sub: one rounded sub per element.
            for i in 0..hc {
                for j in 0..hc {
                    mat[i * hc + j] -= row_max[i];
                }
            }
            // exp: unary kernel.
            for v in mat.iter_mut() {
                *v = v.exp();
            }
            // sum_keepdim(-1): fast_sum per row.
            let row_sum: Vec<f32> = (0..hc)
                .map(|i| candle_tree_sum(&mat[i * hc..(i + 1) * hc]))
                .collect();
            // broadcast_div: one rounded div per element.
            for i in 0..hc {
                for j in 0..hc {
                    mat[i * hc + j] /= row_sum[i];
                }
            }
            // `x + eps` = affine(1.0, eps): fmaf(x, 1.0, eps) == x + eps
            // rounded once.
            for v in mat.iter_mut() {
                *v += eps;
            }

            // Initial column normalize: sum_keepdim(1) (fast_sum down each
            // column), then affine(+eps), then broadcast_div.
            let col_div = |mat: &mut [f32]| {
                let col_sum_eps: Vec<f32> = (0..hc)
                    .map(|j| {
                        let col: Vec<f32> = (0..hc).map(|k| mat[k * hc + j]).collect();
                        candle_tree_sum(&col) + eps
                    })
                    .collect();
                for i in 0..hc {
                    for j in 0..hc {
                        mat[i * hc + j] /= col_sum_eps[j];
                    }
                }
            };
            col_div(mat);

            // (iters - 1) more row->col passes.
            for _ in 0..iters.saturating_sub(1) {
                let row_sum_eps: Vec<f32> = (0..hc)
                    .map(|i| candle_tree_sum(&mat[i * hc..(i + 1) * hc]) + eps)
                    .collect();
                for i in 0..hc {
                    for j in 0..hc {
                        mat[i * hc + j] /= row_sum_eps[i];
                    }
                }
                col_div(mat);
            }
        }
        x
    }

    /// Line-by-line transcription of `cuda/sinkhorn.cu` (the fused kernel):
    /// one virtual thread per row holding its row in "registers", the shared
    /// `[hc, hc]` tile, `csum` recomputed per column, same barrier structure
    /// collapsed to sequential phases.
    #[allow(clippy::cast_possible_truncation)] // eps -> f32 mirrors the CUDA wrapper's `eps as f32`
    #[allow(clippy::needless_range_loop)] // indexed loops deliberately mirror the .cu source
    pub(crate) fn sinkhorn_fused_kernel_replay(
        x: &[f32],
        n: usize,
        hc: usize,
        iters: usize,
        eps: f64,
    ) -> Vec<f32> {
        assert_eq!(x.len(), n * hc * hc);
        let eps = eps as f32;
        let mut out = vec![0.0f32; n * hc * hc];
        for batch in 0..n {
            let base = batch * hc * hc;
            // r[row][j]: per-thread registers.
            let mut r: Vec<Vec<f32>> = (0..hc)
                .map(|row| x[base + row * hc..base + (row + 1) * hc].to_vec())
                .collect();
            let mut mat = vec![0.0f32; hc * hc];
            let mut csum = vec![0.0f32; hc];

            // ---- 1. stable row softmax, then + eps ----
            for row in 0..hc {
                let m = candle_tree_max(&r[row]);
                for j in 0..hc {
                    r[row][j] = (r[row][j] - m).exp();
                }
                let rs = candle_tree_sum(&r[row]);
                for j in 0..hc {
                    r[row][j] = r[row][j] / rs + eps;
                }
                for j in 0..hc {
                    mat[row * hc + j] = r[row][j];
                }
            }

            // ---- 2. initial column normalize ----
            for row in 0..hc {
                let col: Vec<f32> = (0..hc).map(|k| mat[k * hc + row]).collect();
                csum[row] = candle_tree_sum(&col) + eps;
            }
            for row in 0..hc {
                for j in 0..hc {
                    r[row][j] = mat[row * hc + j] / csum[j];
                }
            }

            // ---- 3. (iters - 1) more row->col passes ----
            for _ in 0..iters.saturating_sub(1) {
                for row in 0..hc {
                    let rsum = candle_tree_sum(&r[row]) + eps;
                    for j in 0..hc {
                        r[row][j] /= rsum;
                    }
                    for j in 0..hc {
                        mat[row * hc + j] = r[row][j];
                    }
                }
                for row in 0..hc {
                    let col: Vec<f32> = (0..hc).map(|k| mat[k * hc + row]).collect();
                    csum[row] = candle_tree_sum(&col) + eps;
                }
                for row in 0..hc {
                    for j in 0..hc {
                        r[row][j] = mat[row * hc + j] / csum[j];
                    }
                }
            }

            for row in 0..hc {
                for j in 0..hc {
                    out[base + row * hc + j] = r[row][j];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::reference::*;

    /// Deterministic xorshift32 so the randomized bitwise sweep needs no
    /// external crates and reproduces exactly.
    struct XorShift32(u32);
    impl XorShift32 {
        #[allow(clippy::cast_precision_loss)] // uniform-ish [lo, hi] is all we need
        fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            lo + (hi - lo) * (x as f32 / u32::MAX as f32)
        }
    }

    /// The candle CUDA tree order really is different from sequential
    /// summation in f32 — the exact rounding gap that made the old sequential
    /// kernel drift. `[2^24, 1, 1, 1]`: sequential gives 2^24 (each +1 is
    /// absorbed), the tree gives (2^24+1)+(1+1) = 2^24+2 (representable).
    #[test]
    fn tree_sum_order_differs_from_sequential() {
        let vals = [16_777_216.0f32, 1.0, 1.0, 1.0];
        let sequential = vals.iter().copied().fold(0.0f32, |a, v| a + v);
        assert_eq!(sequential, 16_777_216.0);
        assert_eq!(candle_tree_sum(&vals), 16_777_218.0);
    }

    /// Non-power-of-two lengths use candle's zero/`-INF` identity padding.
    #[test]
    fn tree_reductions_handle_non_pow2_lengths() {
        let vals = [3.0f32, -1.0, 7.5];
        assert_eq!(candle_tree_sum(&vals), (3.0 + 7.5) + (-1.0 + 0.0));
        assert_eq!(candle_tree_max(&vals), 7.5);
        assert_eq!(candle_tree_sum(&[42.0]), 42.0);
        assert_eq!(candle_tree_max(&[42.0]), 42.0);
    }

    /// The kernel transcription and the candle-op-chain transcription must be
    /// BIT-identical: same reductions, same per-op rounding, same eps
    /// placement. Sweeps square sizes (incl. non-pow2 hc), batch sizes,
    /// iteration counts, and magnitudes from tame to softmax-saturating.
    #[test]
    fn fused_kernel_replay_is_bitwise_equal_to_candle_gpu_replay() {
        let eps = 1e-6f64;
        let mut rng = XorShift32(0xA5A5_1234);
        for &hc in &[3usize, 4, 5, 8, 16] {
            for &n in &[1usize, 2, 7] {
                for &iters in &[1usize, 2, 20] {
                    for &(lo, hi) in &[(-1.0f32, 1.0), (-8.0, 8.0), (-100.0, 100.0)] {
                        let x: Vec<f32> = (0..n * hc * hc).map(|_| rng.next_f32(lo, hi)).collect();
                        let a = sinkhorn_candle_gpu_replay(&x, n, hc, iters, eps);
                        let b = sinkhorn_fused_kernel_replay(&x, n, hc, iters, eps);
                        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                            assert_eq!(
                                av.to_bits(),
                                bv.to_bits(),
                                "bit mismatch at [{i}] (hc={hc} n={n} iters={iters} \
                                 range=({lo},{hi})): candle-replay {av} vs kernel-replay {bv}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The replay tracks the real candle reference (`sinkhorn_normalize`,
    /// models/dsv4_mhc.rs) on the CPU backend. Not bit-exact by design — the
    /// CPU backend reduces sequentially, the CUDA backend (which the replay
    /// mirrors) reduces pairwise — so this is a tight-tolerance check plus the
    /// doubly-stochastic invariant.
    #[test]
    fn candle_gpu_replay_tracks_cpu_reference() -> candle_core::Result<()> {
        use candle_core::{Device, Tensor};
        let (n, hc, iters, eps) = (2usize, 4usize, 20usize, 1e-6f64);
        let mut rng = XorShift32(0xDEAD_BEEF);
        let x: Vec<f32> = (0..n * hc * hc).map(|_| rng.next_f32(-4.0, 4.0)).collect();

        let t = Tensor::from_vec(x.clone(), (n, hc, hc), &Device::Cpu)?;
        let cpu_ref: Vec<f32> = crate::models::dsv4_mhc::sinkhorn_normalize(&t, iters, eps)?
            .flatten_all()?
            .to_vec1()?;
        let replay = sinkhorn_candle_gpu_replay(&x, n, hc, iters, eps);

        for (i, (a, b)) in cpu_ref.iter().zip(replay.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "replay drifted from CPU candle reference at [{i}]: {a} vs {b}"
            );
        }
        // Doubly-stochastic invariant on the replay itself.
        for b in 0..n {
            for i in 0..hc {
                let row: f32 = replay[b * hc * hc + i * hc..b * hc * hc + (i + 1) * hc]
                    .iter()
                    .sum();
                assert!((row - 1.0).abs() < 1e-3, "row sum {row} not ≈ 1.0");
                let col: f32 = (0..hc).map(|k| replay[b * hc * hc + k * hc + i]).sum();
                assert!((col - 1.0).abs() < 1e-3, "col sum {col} not ≈ 1.0");
            }
        }
        Ok(())
    }

    /// Source-level tripwires on the kernel and its build wiring: the IEEE
    /// intrinsics and the fast-math `#error` guard must stay in sinkhorn.cu,
    /// fast-math approximations must stay out, and build.rs must keep
    /// sinkhorn.cu out of the `--use_fast_math` glob (compiling it with
    /// `--fmad=false` instead). The `#error` guard makes any wiring regression
    /// a hard CUDA build failure; these string checks catch it on CPU CI too.
    #[test]
    fn kernel_source_and_build_wiring_guards() {
        let cu = include_str!("sinkhorn.cu");
        assert!(
            cu.contains("#if defined(__USE_FAST_MATH__)"),
            "fast-math #error guard missing"
        );
        assert!(cu.contains("#error"), "fast-math #error guard missing");
        for required in [
            "__fadd_rn",
            "__fsub_rn",
            "__fdiv_rn",
            "candle_tree_sum",
            "candle_tree_max",
        ] {
            assert!(
                cu.contains(required),
                "sinkhorn.cu lost required token {required}"
            );
        }
        for forbidden in ["__expf(", "__fdividef("] {
            assert!(
                !cu.contains(forbidden),
                "sinkhorn.cu contains forbidden fast-math call {forbidden}"
            );
        }

        let build = include_str!("../../build.rs");
        // The exclude list grows as more bit-identity-critical kernels are
        // added (hc_fused.cu joined it), so match on the list's CONTENTS rather
        // than on its exact spelling — otherwise this guard fails for the one
        // reason that is not a regression, and gets "fixed" by weakening it.
        let exclude = build
            .split(".exclude(&[")
            .nth(1)
            .and_then(|s| s.split("])").next())
            .expect("build.rs no longer calls .exclude(&[..]) on the fast-math builder");
        assert!(
            exclude.contains("\"sinkhorn.cu\""),
            "build.rs no longer excludes sinkhorn.cu from the fast-math builder \
             (exclude list is: {exclude})"
        );
        assert!(
            build.contains(r#""src/cuda/sinkhorn.cu""#),
            "build.rs no longer feeds sinkhorn.cu to the IEEE (no-fast-math) builder"
        );
        assert!(
            build.contains("--fmad=false"),
            "build.rs lost the --fmad=false flag for the sinkhorn IEEE builder"
        );
    }
}
