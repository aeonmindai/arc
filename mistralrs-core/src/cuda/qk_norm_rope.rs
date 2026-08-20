//! Parent system: ArcInfer / ArcAttention
//!
//! Rust side of `cuda/qk_norm_rope.cu` — the fused DeepSeek-V4 Q/K
//! pre-attention block.
//!
//! One kernel replaces the sixteen candle launches that
//! [`crate::models::deepseek4::Attention::forward`] currently spends between
//! the Q/KV projections and attention: the head transpose, the per-head Q
//! RMS-normalisation, the RoPE split/rotate/`cat`, and the two
//! `.contiguous()` calls that materialise the intermediates. Ten of the
//! sixteen are pure data movement (`ucopy_bf16` / `copy2d_bf16`), which is the
//! block this file exists to remove.
//!
//! The kernel is **bit-identical** to the chain it replaces, not an
//! approximation. The contract, and the three reasons it is achievable at all,
//! are documented at the top of `qk_norm_rope.cu`. The short version: candle's
//! `fast_sum` reduction tree shape is copied rather than chosen, bf16
//! arithmetic in CUDA is inline PTX and therefore immune to `--fmad` and
//! `--use_fast_math`, and the `affine` scalars are narrowed to bf16 host-side
//! by the same `half::bf16` code candle uses.
//!
//! Two switches, both read once and cached (`std::env::var` takes a global
//! lock and this is consulted on every layer of every step):
//!
//! - `ARC_QK_FUSED=0` restores the eager chain, so the A/B runs from ONE
//!   binary and cannot be contaminated by a rebuild.
//! - `ARC_QK_VERIFY=1` runs BOTH paths at every layer and asserts the outputs
//!   are bit-identical, with a negative control that proves the comparator is
//!   live. See [`verify_enabled`].

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Number of times the fused kernel has actually run.
///
/// This exists because "the benchmark got faster" and "the kernel ran" are
/// different claims, and a fused path that silently declines every shape would
/// produce a perfectly green no-op. The A/B harness reads
/// [`engaged_count`] and treats zero as an environment failure, not a result.
static ENGAGED: AtomicU64 = AtomicU64::new(0);
/// Number of times the shape/dtype gate refused and the eager chain ran.
static DECLINED: AtomicU64 = AtomicU64::new(0);
static LOGGED_FIRST: AtomicBool = AtomicBool::new(false);

/// `ARC_QK_FUSED=0` disables the fused Q/K kernel and restores the eager
/// candle chain. Any other value (or unset) keeps it on.
pub fn fused_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !matches!(std::env::var("ARC_QK_FUSED").as_deref(), Ok("0")))
}

/// `ARC_QK_VERIFY=1` runs the eager chain alongside the fused kernel at every
/// layer and compares the raw output bytes.
///
/// The comparison is on bf16 BIT PATTERNS, not a tolerance — an earlier
/// bf16-level assertion elsewhere in this codebase was vacuous because eight
/// mantissa bits swallowed the difference it claimed to detect. Bit equality
/// cannot swallow anything, and [`verify_pair`] additionally runs a negative
/// control (one element poisoned by a single ULP) that MUST be flagged,
/// so a comparator that had silently degenerated into `true` is caught.
pub fn verify_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("ARC_QK_VERIFY").as_deref(), Ok("1")))
}

/// `(engaged, declined)` — fused launches and shape-gate refusals so far.
pub fn engaged_count() -> (u64, u64) {
    (ENGAGED.load(Ordering::Relaxed), DECLINED.load(Ordering::Relaxed))
}

pub(crate) fn note_engaged(head_dim: usize, rope_dim: usize, n_heads: usize, dtype: &str) {
    let n = ENGAGED.fetch_add(1, Ordering::Relaxed);
    if !LOGGED_FIRST.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "ArcAttention: fused qk_norm_rope ENGAGED (head_dim={head_dim}, rope_dim={rope_dim}, \
             n_heads={n_heads}, dtype={dtype}) — replacing 16 candle launches per layer with 1"
        );
    }
    // Periodic proof of life for long runs; cheap (one relaxed load).
    if n > 0 && n % 100_000 == 0 {
        tracing::info!("ArcAttention: fused qk_norm_rope engaged {n} times");
    }
}

pub(crate) fn note_declined(reason: &str) {
    let n = DECLINED.fetch_add(1, Ordering::Relaxed);
    if n == 0 {
        tracing::warn!(
            "ArcAttention: fused qk_norm_rope DECLINED ({reason}) — falling back to the eager \
             chain. This is correct but slow; the launch saving is not being taken."
        );
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core as candle;
    use candle_core::{DType, Result, Tensor};

    /// Pull a contiguous BF16 CUDA tensor's device pointer.
    fn bf16_ptr(t: &Tensor, what: &str) -> Result<*const std::ffi::c_void> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        if t.dtype() != DType::BF16 {
            candle::bail!("qk_norm_rope: {what} must be BF16, got {:?}", t.dtype());
        }
        if !t.is_contiguous() {
            candle::bail!("qk_norm_rope: {what} must be contiguous");
        }
        let (s, l) = t.storage_and_layout();
        let cuda = match &*s {
            candle::Storage::Cuda(c) => c,
            _ => candle::bail!("qk_norm_rope: {what} must be on CUDA"),
        };
        // The (ptr, guard) pair borrows the slice, so bind before returning —
        // same shape as `hc_fused::any_ptr`.
        let sl = cuda.as_cuda_slice::<half::bf16>()?;
        let ptr = sl.slice(l.start_offset()..).device_ptr(sl.stream()).0;
        Ok(ptr as *const std::ffi::c_void)
    }

    /// Fused head-transpose + Q RMS-norm + RoPE + NoPE/PE recombination.
    ///
    /// - `q_in`  `[B, T, H * D]` or `[B, T, H, D]` BF16 contiguous — the raw
    ///   q_proj output, **not** transposed. The transpose is the kernel's
    ///   output addressing.
    /// - `k_in`  `[B, T, D]` or `[B, T, 1, D]` BF16 contiguous — post-`kv_norm`.
    /// - `cos` / `sin` `[max_pos, D_rope / 2]` BF16 contiguous — the FULL
    ///   tables; `pos_offset` replaces candle's `narrow`.
    ///
    /// Returns `(q_out [B, H, T, D], k_out [B, 1, T, D])`.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_cuda(
        q_in: &Tensor,
        k_in: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        n_heads: usize,
        batch: usize,
        seq_len: usize,
        head_dim: usize,
        rope_dim: usize,
        eps: f64,
        pos_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let q_ptr = bf16_ptr(q_in, "q_in")?;
        let k_ptr = bf16_ptr(k_in, "k_in")?;
        let cos_ptr = bf16_ptr(cos, "cos")?;
        let sin_ptr = bf16_ptr(sin, "sin")?;

        let dev = q_in.device().as_cuda_device()?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        let q_elems = batch * n_heads * seq_len * head_dim;
        let k_elems = batch * seq_len * head_dim;

        // candle's `affine(mul, add)` narrows its f64 arguments to T before the
        // kernel sees them. Do the same narrowing, with the same code, so no
        // rounding decision is taken twice.
        let inv_n = u32::from(half::bf16::from_f64(1.0f64 / head_dim as f64).to_bits());
        let zero = u32::from(half::bf16::from_f64(0.0).to_bits());
        let one = u32::from(half::bf16::from_f64(1.0).to_bits());
        let eps_b = u32::from(half::bf16::from_f64(eps).to_bits());

        let q_buf = unsafe { dev.alloc::<half::bf16>(q_elems) }?;
        let k_buf = unsafe { dev.alloc::<half::bf16>(k_elems) }?;

        let rc = unsafe {
            crate::cuda::ffi::arc_qk_norm_rope_bf16_v2(
                q_ptr,
                k_ptr,
                cos_ptr,
                sin_ptr,
                q_buf.device_ptr(q_buf.stream()).0 as *mut std::ffi::c_void,
                k_buf.device_ptr(k_buf.stream()).0 as *mut std::ffi::c_void,
                n_heads as i32,
                batch as i32,
                seq_len as i32,
                head_dim as i32,
                rope_dim as i32,
                pos_offset as i32,
                1, // QK_DTYPE_BF16
                inv_n,
                zero,
                one,
                eps_b,
                stream,
            )
        };
        if rc != 0 {
            // The host-side gate in `deepseek4.rs` should have caught this, so
            // reaching here is a bug rather than an unsupported config.
            candle::bail!(
                "qk_norm_rope: kernel dispatch refused (rc={rc}) for head_dim={head_dim} \
                 rope_dim={rope_dim} — host gate and device gate disagree"
            );
        }

        let q_st = candle::CudaStorage::wrap_cuda_slice(q_buf, dev.clone());
        let k_st = candle::CudaStorage::wrap_cuda_slice(k_buf, dev.clone());
        let q_out = Tensor::from((
            candle::Storage::Cuda(q_st),
            candle_core::Shape::from((batch, n_heads, seq_len, head_dim)),
        ));
        let k_out = Tensor::from((
            candle::Storage::Cuda(k_st),
            candle_core::Shape::from((batch, 1usize, seq_len, head_dim)),
        ));
        Ok((q_out, k_out))
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::qk_norm_rope_cuda;

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_cuda(
    _q_in: &candle_core::Tensor,
    _k_in: &candle_core::Tensor,
    _cos: &candle_core::Tensor,
    _sin: &candle_core::Tensor,
    _n_heads: usize,
    _batch: usize,
    _seq_len: usize,
    _head_dim: usize,
    _rope_dim: usize,
    _eps: f64,
    _pos_offset: usize,
) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
    candle_core::bail!("qk_norm_rope_cuda requires the cuda feature")
}

/// Count bit-pattern mismatches between two bf16 slices.
///
/// The ONE comparison routine used by both the real check and its negative
/// control, so the control provably exercises the same code.
fn bit_mismatches(a: &[half::bf16], b: &[half::bf16]) -> (usize, Option<(usize, u16, u16)>) {
    let mut n = 0usize;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            n += 1;
            if first.is_none() {
                first = Some((i, x.to_bits(), y.to_bits()));
            }
        }
    }
    (n, first)
}

/// Bit-identity check with a built-in negative control.
///
/// `fused` and `eager` must be byte-for-byte equal. The control runs
/// [`bit_mismatches`] — the very same routine — against a copy of `eager` with
/// one element perturbed by a single bf16 ULP. If that run does NOT report
/// exactly one difference, the comparator is not doing its job and the check
/// is declared VOID rather than passed.
///
/// Comparison is on bf16 BIT PATTERNS, never a tolerance. An earlier
/// bf16-level assertion in this codebase was vacuous precisely because eight
/// mantissa bits swallowed the difference it claimed to detect; bit equality
/// cannot swallow anything.
///
/// Returns `Ok(())` only on a proven match — a verify leg must be able to exit
/// non-zero.
pub fn verify_pair(
    name: &str,
    fused: &candle_core::Tensor,
    eager: &candle_core::Tensor,
) -> candle_core::Result<()> {
    if fused.shape() != eager.shape() {
        candle_core::bail!(
            "qk_norm_rope VERIFY {name}: shape {:?} != {:?}",
            fused.shape(),
            eager.shape()
        );
    }
    if fused.dtype() != candle_core::DType::BF16 || eager.dtype() != candle_core::DType::BF16 {
        candle_core::bail!(
            "qk_norm_rope VERIFY {name}: expected BF16/BF16, got {:?}/{:?}",
            fused.dtype(),
            eager.dtype()
        );
    }
    let f: Vec<half::bf16> = fused.flatten_all()?.to_vec1()?;
    let e: Vec<half::bf16> = eager.flatten_all()?.to_vec1()?;
    if f.len() != e.len() {
        candle_core::bail!("qk_norm_rope VERIFY {name}: len {} != {}", f.len(), e.len());
    }
    if f.is_empty() {
        candle_core::bail!("qk_norm_rope VERIFY {name}: empty tensor — nothing was compared");
    }

    // -- negative control, run FIRST so a degenerate comparator cannot pass --
    let poison_idx = f.len() / 2;
    let mut poisoned = e.clone();
    poisoned[poison_idx] = half::bf16::from_bits(e[poison_idx].to_bits() ^ 1);
    let (ctrl_n, _) = bit_mismatches(&poisoned, &e);
    if ctrl_n != 1 {
        candle_core::bail!(
            "qk_norm_rope VERIFY {name}: NEGATIVE CONTROL FAILED — a 1-ULP poison at index \
             {poison_idx} produced {ctrl_n} detected mismatches, expected exactly 1. The \
             comparator is not measuring what it claims; declaring VOID rather than passed."
        );
    }

    // -- the real comparison, same routine --
    let (n, first) = bit_mismatches(&f, &e);
    if n != 0 {
        let (i, fb, eb) = first.expect("mismatch count > 0 implies a first mismatch");
        // WHERE a mismatch falls inside the head vector separates the two
        // failure modes that look alike from a bare count: a wrong RMS
        // statistic scatters over all dims, while a wrong RoPE table row hits
        // only the low-j (high-frequency) pairs, because high-j pairs have
        // cos ~ 1 / sin ~ 0 at any nearby position and round to the same bf16.
        let hd = *fused.dims().last().unwrap_or(&1);
        let mut nope_bad = 0usize;
        let mut pairs: Vec<usize> = Vec::new();
        let rope = hd.min(64);
        let nope = hd - rope;
        for (idx, (x, y)) in f.iter().zip(e.iter()).enumerate() {
            if x.to_bits() == y.to_bits() {
                continue;
            }
            let d = idx % hd;
            if d < nope {
                nope_bad += 1;
            } else {
                let j = (d - nope) / 2;
                if !pairs.contains(&j) {
                    pairs.push(j);
                }
            }
        }
        pairs.sort_unstable();
        candle_core::bail!(
            "qk_norm_rope VERIFY {name}: {n}/{} elements differ; first at {i}: \
             fused=0x{fb:04x} eager=0x{eb:04x}; head_dim={hd} nope_mismatches={nope_bad} \
             rope_pairs_hit={pairs:?}",
            f.len()
        );
    }
    VERIFIED.fetch_add(f.len() as u64, Ordering::Relaxed);
    Ok(())
}

/// `ARC_QK_DIAG=1` runs [`diagnose_row`] on the first fused call.
pub fn diag_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("ARC_QK_DIAG").as_deref(), Ok("1")))
}

static DIAGGED: AtomicBool = AtomicBool::new(false);

/// Recover, from candle's own output, WHICH row of the RoPE table it rotated
/// by — and report whether that is the row this kernel reads.
///
/// `q_normed` is the eager normalised Q `[B, H, T, D]`; `q_eager` is the eager
/// post-RoPE Q of the same shape. Only head 0, t 0, pair 0 is needed: it is the
/// highest-frequency pair, so it pins the position most sharply.
pub fn diagnose_row(
    q_normed: &candle_core::Tensor,
    q_eager: &candle_core::Tensor,
    q_fused: &candle_core::Tensor,
    cos: &candle_core::Tensor,
    sin: &candle_core::Tensor,
    head_dim: usize,
    rope_dim: usize,
    pos_offset: usize,
    seq_len: usize,
) -> candle_core::Result<()> {
    if DIAGGED.swap(true, Ordering::Relaxed) {
        return Ok(());
    }
    tracing::error!("qk_norm_rope DIAG: pos_offset={pos_offset} seq_len={seq_len} head_dim={head_dim} rope_dim={rope_dim}");
    let nope = head_dim - rope_dim;
    let n: Vec<f32> = q_normed.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    let o: Vec<f32> = q_eager.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    let fu: Vec<f32> = q_fused.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    let a = n[nope] as f64;
    let b = n[nope + 1] as f64;
    let o0 = o[nope] as f64;
    let o1 = o[nope + 1] as f64;
    let den = a * a + b * b;
    if den == 0.0 {
        tracing::warn!("qk_norm_rope DIAG: pair 0 is zero; cannot solve for the row");
        return Ok(());
    }
    let c_used = (a * o0 + b * o1) / den;
    let s_used = (a * o1 - b * o0) / den;

    let (rows, cols) = cos.dims2()?;
    let half = rope_dim / 2;
    let scan = rows.min(pos_offset + 4096);
    let cv: Vec<f32> = cos.narrow(0, 0, scan)?.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    let sv: Vec<f32> = sin.narrow(0, 0, scan)?.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    let mut best = (f64::INFINITY, usize::MAX);
    for r in 0..scan {
        let dc = cv[r * cols] as f64 - c_used;
        let ds = sv[r * cols] as f64 - s_used;
        let d = dc * dc + ds * ds;
        if d < best.0 {
            best = (d, r);
        }
    }
    let mine = pos_offset;
    let cb: Vec<half::bf16> = cos.narrow(0, mine, 1)?.flatten_all()?.to_vec1()?;
    let sb: Vec<half::bf16> = sin.narrow(0, mine, 1)?.flatten_all()?.to_vec1()?;
    tracing::error!(
        "qk_norm_rope DIAG bits: kernel row {mine} pair0 cos=0x{:04x} sin=0x{:04x}; \
         pairs 0..4 cos=[{:04x},{:04x},{:04x},{:04x}] sin=[{:04x},{:04x},{:04x},{:04x}]",
        cb[0].to_bits(), sb[0].to_bits(),
        cb[0].to_bits(), cb[1].to_bits(), cb[2].to_bits(), cb[3].to_bits(),
        sb[0].to_bits(), sb[1].to_bits(), sb[2].to_bits(), sb[3].to_bits()
    );
    tracing::error!(
        "qk_norm_rope DIAG: table is [{rows}, {cols}], half={half}, seq_len={seq_len}. \
         candle rotated pair 0 by (cos={c_used:.6}, sin={s_used:.6}); the NEAREST table row is \
         {} (err={:.3e}); the kernel reads row {mine} whose pair 0 is (cos={:.6}, sin={:.6}). \
         ROW_USED_BY_CANDLE={} ROW_READ_BY_KERNEL={mine}",
        best.1,
        best.0.sqrt(),
        cv[mine * cols],
        sv[mine * cols],
        best.1
    );
    tracing::error!(
        "qk_norm_rope DIAG pair0: normed a={a:.6} b={b:.6} | eager out=({o0:.6}, {o1:.6}) |          fused out=({:.6}, {:.6}) | nope[0] eager={:.6} fused={:.6}",
        fu[nope], fu[nope + 1], o[0], fu[0]
    );
    Ok(())
}

/// Total bf16 elements proven bit-identical so far. Reported by the verify leg
/// so "0 mismatches" can never be confused with "nothing was compared".
static VERIFIED: AtomicU64 = AtomicU64::new(0);

/// Elements proven bit-identical so far.
pub fn verified_elements() -> u64 {
    VERIFIED.load(Ordering::Relaxed)
}
