//! FlashMLASparse high-level Rust API for the V4 Lightning Indexer.
//!
//! Wraps the BF16 indexer-score + radix-topk CUDA kernels into a single
//! `score_and_topk` call that takes Candle tensors, dispatches into the FFI,
//! and returns the top-k indices as a Candle tensor — matching exactly what
//! `mistralrs_core::models::dsv4_indexer::V4Indexer::forward` produces in
//! pure Rust.
//!
//! ## CPU reference
//!
//! The `cpu_reference` function below is the *spec* for the GPU kernels:
//! for the same FP32 inputs, the GPU's top-k indices must match the CPU's
//! (up to tied-score ordering, which we sidestep in tests by using inputs
//! with distinct scores). The agreement test in this module exercises the
//! CPU path with several shapes; the equivalent GPU path is run only when
//! `feature="cuda"` is enabled (and a device is present at runtime, which
//! the test guards against).
//!
//! ## Why a CPU reference and not a unit test against SGLang's Triton kernel?
//!
//! The Triton kernels in SGLang require a CUDA host to JIT and run, so we
//! can't import them into our CI. Instead, the CPU reference is a
//! straightforward `Vec<f32>` implementation of the same algorithm (dot
//! products + relu + weighted head-sum + per-row top-k) — auditable
//! line-by-line against the CUDA kernel.
//!
//! ## The scoring formula
//!
//! ```text
//! logit[b, q, c] = Σ_h relu(q[b,h,q,:] · k[b,h,c,:]) * w[b,h,q]
//! topk_idx[b, q] = argtopk_c(logit[b, q, :])   // ONE selection, all heads
//! ```
//!
//! Ported from SGLang `python/sglang/srt/layers/attention/dsv4/indexer.py`,
//! `fp8_paged_mqa_logits_torch` lines 84-89:
//!
//! ```python
//! score = F.linear(kvcache_value, q)   # per-head q.k
//! score = F.relu(score)                # rectify BEFORE mixing heads
//! score *= q_scale[None, :]            # per-head weight
//! score = score.sum(dim=1)             # ONE logit per key
//! score *= kvcache_scale               # per-key FP8 dequant scale (Arc: 1)
//! ```
//!
//! `w[b,h,q] = weights_proj(x)[b,q,h] * weight_scale` with
//! `weight_scale = softmax_scale * n_heads^-0.5` (SGLang `C4Indexer.__init__`
//! line 519) — the caller folds `weight_scale` in before calling here, exactly
//! as SGLang folds it in `compute_weights` / `fused_scale`.
//!
//! The relu and the head-sum are both load-bearing: without the relu a
//! negative head dot cancels a positive one and a different key wins; without
//! the head-sum the top-k runs per head and emits `n_heads` (= 64) distinct
//! key sets, multiplying sparse-gather traffic by 64.
//!
//! ## License attribution
//!
//! The CUDA kernels backing this module are vendored from MIT-licensed
//! upstream sources; see `src/cuda/flashmlasparse/LICENSE-MIT`.

#[cfg(feature = "cuda")]
use candle_core::{DType, Device, Result, Tensor};

/// Supported top-k values. The radix kernel dispatches on these only.
pub const SUPPORTED_TOPK: &[usize] = &[64, 128, 256, 512, 1024];

/// CPU reference: logits + top-k. Produces the same `[B, T_q, topk]`
/// I32 tensor that the GPU kernel produces (for inputs with distinct logits).
///
/// Inputs:
/// - `q`: `[B, n_heads, T_q, head_dim]` FP32, contiguous.
/// - `k`: `[B, n_heads, T_c, head_dim]` FP32, contiguous.
/// - `weights`: `[B, n_heads, T_q]` FP32, contiguous — `weights_proj(x)` with
///   `weight_scale` already folded in by the caller.
///
/// Output:
/// - `[B, T_q, topk]` u32 (Candle has no I32 host dtype; we use U32
///   matching the existing `V4Indexer::forward` output dtype). **No head
///   axis**: the selection is shared by every head.
///
/// Algorithm (matches `V4Indexer::forward` / `indexer_logits` line-by-line):
///   logits[b, q, c] = Σ_h relu(Σ_d q[b,h,q,d] * k[b,h,c,d]) * weights[b,h,q]
///   top_k(logits, dim=-1) → indices
pub fn cpu_reference(
    q: &[f32],
    q_dims: (usize, usize, usize, usize),
    k: &[f32],
    k_dims: (usize, usize, usize, usize),
    weights: &[f32],
    weight_dims: (usize, usize, usize),
    topk: usize,
) -> Vec<u32> {
    let (b, h_q, t_q, d_q) = q_dims;
    let (b_k, h_k, t_c, d_k) = k_dims;
    let (b_s, h_s, t_s) = weight_dims;
    assert_eq!(b, b_k);
    assert_eq!(b, b_s);
    assert_eq!(h_q, h_k);
    assert_eq!(h_q, h_s);
    assert_eq!(t_q, t_s);
    assert_eq!(d_q, d_k);
    assert_eq!(q.len(), b * h_q * t_q * d_q);
    assert_eq!(k.len(), b * h_k * t_c * d_k);
    assert_eq!(weights.len(), b * h_q * t_q);
    let head_dim = d_q;
    let n_heads = h_q;

    let k_top = topk.min(t_c);
    let mut out = vec![0u32; b * t_q * topk];

    for bi in 0..b {
        for qi in 0..t_q {
            // One logit per compressed key, summed over heads.
            let mut logits = vec![0.0f32; t_c];
            for hi in 0..n_heads {
                let w = weights[(bi * n_heads + hi) * t_q + qi];
                // q row: q[bi, hi, qi, :]
                let q_row_base = ((bi * n_heads + hi) * t_q + qi) * head_dim;
                for (ci, logit) in logits.iter_mut().enumerate() {
                    let k_row_base = ((bi * n_heads + hi) * t_c + ci) * head_dim;
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc += q[q_row_base + d] * k[k_row_base + d];
                    }
                    // relu BEFORE mixing heads, then weighted head-sum.
                    *logit += acc.max(0.0) * w;
                }
            }
            // Top-k (unsorted) by descending logit. We sort indices by
            // logit desc with lowest-index tiebreak to mirror the GPU
            // kernel's behaviour.
            let mut idx: Vec<usize> = (0..t_c).collect();
            idx.sort_by(|&a, &b_i| {
                logits[b_i]
                    .partial_cmp(&logits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b_i))
            });
            let out_base = (bi * t_q + qi) * topk;
            for i in 0..k_top {
                out[out_base + i] = idx[i] as u32;
            }
            // Pad remainder with sentinel -1 (cast to u32: 0xFFFFFFFF).
            for i in k_top..topk {
                out[out_base + i] = u32::MAX;
            }
        }
    }

    out
}

/// Validate kernel preconditions. Returns `Err` if the configuration isn't
/// supported by the current kernel dispatch table.
pub fn validate_dispatch(topk: usize, head_dim: usize) -> std::result::Result<(), String> {
    if !SUPPORTED_TOPK.contains(&topk) {
        return Err(format!(
            "FlashMLASparse: topk={} unsupported. Supported: {:?}",
            topk, SUPPORTED_TOPK
        ));
    }
    if head_dim != 64 && head_dim != 128 && (head_dim == 0 || !head_dim.is_multiple_of(8)) {
        return Err(format!(
            "FlashMLASparse: head_dim={} must be a positive multiple of 8 (typical: 128)",
            head_dim
        ));
    }
    Ok(())
}

// ===========================================================================
// CUDA dispatch — only compiled in with `feature="cuda"`.
// ===========================================================================

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::cuda::cudarc::driver::sys::CUstream;
    use candle_core::cuda::cudarc::driver::DevicePtr;
    use std::ffi::c_void;

    /// Convert a CUDA Candle tensor into a raw device pointer (byte-offset
    /// aware). Matches the helper used elsewhere in this crate
    /// (`sampling_cuda::cuda_tensor_ptr`). Returns the pointer as `usize` so
    /// callers can cast freely.
    fn cuda_tensor_ptr(t: &Tensor) -> Result<usize> {
        let t = t.contiguous()?;
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            candle_core::Storage::Cuda(cuda_storage) => {
                let slice = cuda_storage.as_cuda_slice::<u8>()?;
                let (ptr, _guard) = slice.device_ptr(slice.stream());
                let offset = layout.start_offset() * t.dtype().size_in_bytes();
                Ok(ptr as usize + offset)
            }
            _ => candle_core::bail!("cuda_tensor_ptr requires CUDA tensor"),
        }
    }

    fn cuda_stream(device: &Device) -> Result<CUstream> {
        let Device::Cuda(cuda_dev) = device else {
            candle_core::bail!("FlashMLASparse requires CUDA device");
        };
        Ok(cuda_dev.cuda_stream().cu_stream())
    }

    /// Logits-and-topk fused dispatch.
    ///
    /// Inputs (all on the same CUDA device):
    /// - `q`: `[B, n_heads, T_q, head_dim]` BF16, contiguous.
    /// - `k`: `[B, n_heads, T_c, head_dim]` BF16, contiguous.
    /// - `per_head_scale`: `[B, n_heads, T_q]` BF16, contiguous —
    ///   `weights_proj(x) * weight_scale`.
    /// - `topk`: top-k value (must be in `SUPPORTED_TOPK`).
    ///
    /// Output:
    /// - `[B, T_q, topk]` U32, where each entry is an index into
    ///   `[0, T_c)`. Rows where T_c < topk are padded with `u32::MAX`
    ///   (= -1 reinterpreted) — caller should treat those as invalid.
    ///   **One key set per query, shared by all heads.**
    ///
    /// Internally allocates a scratch FP32 logits buffer + an I32 seq_lens
    /// buffer (filled with `t_c` on every call). The output is allocated
    /// inside the function and returned; future revisions may accept a
    /// caller-provided scratch to avoid the alloc.
    pub fn score_and_topk_bf16(
        q: &Tensor,
        k: &Tensor,
        per_head_scale: &Tensor,
        topk: usize,
    ) -> Result<Tensor> {
        // ---- Shape + dtype validation ----
        let q_dims = q.dims();
        if q_dims.len() != 4 || q.dtype() != DType::BF16 {
            candle_core::bail!(
                "FlashMLASparse: q must be BF16 [B, H, T_q, D]; got dtype={:?} dims={:?}",
                q.dtype(),
                q_dims
            );
        }
        let k_dims = k.dims();
        if k_dims.len() != 4 || k.dtype() != DType::BF16 {
            candle_core::bail!(
                "FlashMLASparse: k must be BF16 [B, H, T_c, D]; got dtype={:?} dims={:?}",
                k.dtype(),
                k_dims
            );
        }
        let s_dims = per_head_scale.dims();
        if s_dims.len() != 3 || per_head_scale.dtype() != DType::BF16 {
            candle_core::bail!(
                "FlashMLASparse: per_head_scale must be BF16 [B, H, T_q]; got dtype={:?} dims={:?}",
                per_head_scale.dtype(),
                s_dims
            );
        }
        let (b, h, t_q, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let (b_k, h_k, t_c, d_k) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
        if b_k != b || h_k != h || d_k != d {
            candle_core::bail!(
                "FlashMLASparse: q/k batch/head/dim mismatch: q={:?} k={:?}",
                q_dims,
                k_dims
            );
        }
        if s_dims[0] != b || s_dims[1] != h || s_dims[2] != t_q {
            candle_core::bail!(
                "FlashMLASparse: per_head_scale dims {:?} != [{},{},{}]",
                s_dims,
                b,
                h,
                t_q
            );
        }

        validate_dispatch(topk, d).map_err(candle_core::Error::msg)?;

        let device = q.device().clone();
        let stream = cuda_stream(&device)?;

        // ---- Scratch + output allocs ----
        // Logits carry no head axis: the kernel relu's and sums the heads.
        let scores = Tensor::zeros((b, t_q, t_c), DType::F32, &device)?;
        let n_rows = (b * t_q) as i32;
        // seq_lens: fill with `t_c` — every row has the same valid length
        // (we don't have per-row T_c in the simple dense case).
        let seq_lens_vec: Vec<i32> = vec![t_c as i32; n_rows as usize];
        let seq_lens = Tensor::from_vec(seq_lens_vec, (n_rows as usize,), &device)?;
        let out = Tensor::zeros((b, t_q, topk), DType::U32, &device)?;

        // Materialise contiguous copies so the FFI sees flat memory.
        let q_c = q.contiguous()?;
        let k_c = k.contiguous()?;
        let s_c = per_head_scale.contiguous()?;

        let q_ptr = cuda_tensor_ptr(&q_c)? as *const c_void;
        let k_ptr = cuda_tensor_ptr(&k_c)? as *const c_void;
        let s_ptr = cuda_tensor_ptr(&s_c)? as *const c_void;
        let sc_ptr = cuda_tensor_ptr(&scores)? as *mut c_void;
        let sl_ptr = cuda_tensor_ptr(&seq_lens)? as *const c_void;
        let out_ptr = cuda_tensor_ptr(&out)? as *mut c_void;

        // ---- Launch logits kernel ----
        unsafe {
            crate::flashmlasparse_ffi::arc_flashmlasparse_logits_bf16(
                q_ptr,
                k_ptr,
                s_ptr,
                sc_ptr,
                b as i32,
                h as i32,
                t_q as i32,
                t_c as i32,
                d as i32,
                stream as *mut c_void,
            );
        }

        // ---- Launch top-k kernel ----
        unsafe {
            crate::flashmlasparse_ffi::arc_flashmlasparse_topk(
                sc_ptr as *const c_void,
                sl_ptr,
                out_ptr,
                n_rows,
                topk as i32,
                t_c as i64,
                stream as *mut c_void,
            );
        }

        Ok(out)
    }

    /// FP32 input variant — used only by the agreement test that runs the
    /// CUDA kernels against the CPU reference. Production callers use the
    /// BF16 path.
    pub fn score_and_topk_f32(
        q: &Tensor,
        k: &Tensor,
        per_head_scale: &Tensor,
        topk: usize,
    ) -> Result<Tensor> {
        let q_dims = q.dims();
        let k_dims = k.dims();
        let s_dims = per_head_scale.dims();
        if q_dims.len() != 4 || q.dtype() != DType::F32 {
            candle_core::bail!("score_and_topk_f32: q must be F32 [B,H,T_q,D]");
        }
        if k_dims.len() != 4 || k.dtype() != DType::F32 {
            candle_core::bail!("score_and_topk_f32: k must be F32 [B,H,T_c,D]");
        }
        if s_dims.len() != 3 || per_head_scale.dtype() != DType::F32 {
            candle_core::bail!("score_and_topk_f32: per_head_scale must be F32 [B,H,T_q]");
        }
        let (b, h, t_q, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let (b_k, h_k, t_c, d_k) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
        if b_k != b || h_k != h || d_k != d || s_dims[0] != b || s_dims[1] != h || s_dims[2] != t_q
        {
            candle_core::bail!("score_and_topk_f32: shape mismatch");
        }
        validate_dispatch(topk, d).map_err(candle_core::Error::msg)?;

        let device = q.device().clone();
        let stream = cuda_stream(&device)?;
        let scores = Tensor::zeros((b, t_q, t_c), DType::F32, &device)?;
        let n_rows = (b * t_q) as i32;
        let seq_lens_vec: Vec<i32> = vec![t_c as i32; n_rows as usize];
        let seq_lens = Tensor::from_vec(seq_lens_vec, (n_rows as usize,), &device)?;
        let out = Tensor::zeros((b, t_q, topk), DType::U32, &device)?;
        let q_c = q.contiguous()?;
        let k_c = k.contiguous()?;
        let s_c = per_head_scale.contiguous()?;
        let q_ptr = cuda_tensor_ptr(&q_c)? as *const c_void;
        let k_ptr = cuda_tensor_ptr(&k_c)? as *const c_void;
        let s_ptr = cuda_tensor_ptr(&s_c)? as *const c_void;
        let sc_ptr = cuda_tensor_ptr(&scores)? as *mut c_void;
        let sl_ptr = cuda_tensor_ptr(&seq_lens)? as *const c_void;
        let out_ptr = cuda_tensor_ptr(&out)? as *mut c_void;

        unsafe {
            crate::flashmlasparse_ffi::arc_flashmlasparse_logits_f32(
                q_ptr,
                k_ptr,
                s_ptr,
                sc_ptr,
                b as i32,
                h as i32,
                t_q as i32,
                t_c as i32,
                d as i32,
                stream as *mut c_void,
            );
            crate::flashmlasparse_ffi::arc_flashmlasparse_topk(
                sc_ptr as *const c_void,
                sl_ptr,
                out_ptr,
                n_rows,
                topk as i32,
                t_c as i64,
                stream as *mut c_void,
            );
        }

        Ok(out)
    }

    /// Plain radix top-k over the rows of an F32 matrix — no scoring step.
    ///
    /// Selects, per row, the indices of the `topk` largest values of
    /// `scores[row, 0..len]` using the multi-pass radix-256 select kernel
    /// (global-memory streaming: no shared-memory length limit, unlike
    /// candle's arg_sort). Used by the big-vocab sampler path in
    /// `mistralrs-core` (top-k/top-p over ~129K-vocab logits).
    ///
    /// Inputs:
    /// - `scores`: `[n_rows, len]` F32 on a CUDA device.
    /// - `topk`: must be in [`SUPPORTED_TOPK`].
    ///
    /// Output: `[n_rows, topk]` U32 indices into `[0, len)`, **unsorted**
    /// within each row. Rows with `len <= topk` are identity-filled and
    /// padded with `u32::MAX` (the kernel's -1 sentinel).
    pub fn radix_topk_rows_f32(scores: &Tensor, topk: usize) -> Result<Tensor> {
        let dims = scores.dims();
        if dims.len() != 2 || scores.dtype() != DType::F32 {
            candle_core::bail!(
                "radix_topk_rows_f32: scores must be F32 [n_rows, len]; got dtype={:?} dims={:?}",
                scores.dtype(),
                dims
            );
        }
        if !SUPPORTED_TOPK.contains(&topk) {
            candle_core::bail!(
                "radix_topk_rows_f32: topk={topk} unsupported. Supported: {SUPPORTED_TOPK:?}"
            );
        }
        let (n_rows, len) = (dims[0], dims[1]);
        let device = scores.device().clone();
        let stream = cuda_stream(&device)?;

        let scores_c = scores.contiguous()?;
        // Every row has the same valid length in the dense case. (The
        // Tensor::from_vec H2D is one tiny transfer per call; callers invoke
        // this once per sampled token, not inside a model forward.)
        let seq_lens_vec: Vec<i32> = vec![len as i32; n_rows];
        let seq_lens = Tensor::from_vec(seq_lens_vec, (n_rows,), &device)?;
        let out = Tensor::zeros((n_rows, topk), DType::U32, &device)?;

        let sc_ptr = cuda_tensor_ptr(&scores_c)? as *const c_void;
        let sl_ptr = cuda_tensor_ptr(&seq_lens)? as *const c_void;
        let out_ptr = cuda_tensor_ptr(&out)? as *mut c_void;

        unsafe {
            crate::flashmlasparse_ffi::arc_flashmlasparse_topk(
                sc_ptr,
                sl_ptr,
                out_ptr,
                n_rows as i32,
                topk as i32,
                len as i64,
                stream as *mut c_void,
            );
        }

        Ok(out)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU reference produces top-k indices that match a hand-rolled
    /// relu + weighted-head-sum + sort baseline.
    #[test]
    fn cpu_reference_matches_baseline() {
        let b = 2;
        let h = 3;
        let t_q = 2;
        let t_c = 8;
        let d = 4;
        let topk = 4;

        // Deterministic logits via a small LCG so we never hit a tie.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        let mut rand = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32) * 4.0 - 2.0
        };
        let q: Vec<f32> = (0..b * h * t_q * d).map(|_| rand()).collect();
        let k: Vec<f32> = (0..b * h * t_c * d).map(|_| rand()).collect();
        // Distinct per-head weights (no tie risk in scaling).
        let w: Vec<f32> = (0..b * h * t_q).map(|i| 0.5 + (i as f32) * 0.01).collect();

        let indices = cpu_reference(
            &q,
            (b, h, t_q, d),
            &k,
            (b, h, t_c, d),
            &w,
            (b, h, t_q),
            topk,
        );
        assert_eq!(indices.len(), b * t_q * topk);

        // Hand-rolled baseline: logit[b,q,c] = Σ_h relu(q_h·k_c) * w[b,h,q],
        // then ONE sort per (bi, qi).
        for bi in 0..b {
            for qi in 0..t_q {
                let mut scored: Vec<(usize, f32)> = (0..t_c)
                    .map(|ci| {
                        let mut logit = 0.0f32;
                        for hi in 0..h {
                            let w_v = w[(bi * h + hi) * t_q + qi];
                            let q_base = ((bi * h + hi) * t_q + qi) * d;
                            let k_base = ((bi * h + hi) * t_c + ci) * d;
                            let mut acc = 0.0f32;
                            for dd in 0..d {
                                acc += q[q_base + dd] * k[k_base + dd];
                            }
                            logit += acc.max(0.0) * w_v;
                        }
                        (ci, logit)
                    })
                    .collect();
                scored.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                let out_base = (bi * t_q + qi) * topk;
                for i in 0..topk {
                    assert_eq!(
                        indices[out_base + i],
                        scored[i].0 as u32,
                        "mismatch at bi={bi} qi={qi} i={i}"
                    );
                }
            }
        }
    }

    /// CPU reference pads with u32::MAX when topk > T_c.
    #[test]
    fn cpu_reference_pads_when_topk_exceeds_tc() {
        let b = 1;
        let h = 1;
        let t_q = 1;
        let t_c = 3;
        let d = 2;
        let topk = 5;
        let q = vec![1.0f32, 2.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w = vec![1.0f32];
        let out = cpu_reference(
            &q,
            (b, h, t_q, d),
            &k,
            (b, h, t_c, d),
            &w,
            (b, h, t_q),
            topk,
        );
        // Logits (single head, all dots positive so relu is a no-op):
        //   c0 = 1*1 + 2*0 = 1.0, c1 = 1*0 + 2*1 = 2.0, c2 = 1*1 + 2*1 = 3.0
        // Sorted desc: c2(3), c1(2), c0(1)
        assert_eq!(out.len(), b * t_q * topk);
        assert_eq!(out[0], 2);
        assert_eq!(out[1], 1);
        assert_eq!(out[2], 0);
        // Remaining topk-T_c = 2 slots are padded with u32::MAX.
        assert_eq!(out[3], u32::MAX);
        assert_eq!(out[4], u32::MAX);
    }

    /// **Force-full-selection dense-equivalence.** With `topk >= T_c` the
    /// selector has no freedom: it must return every key exactly once, so
    /// attending over the gathered subset must reproduce dense attention
    /// **bit-exactly**. Any selection bug that drops, duplicates, or reorders
    /// a key breaks this immediately — the guard the old per-head formula
    /// never had.
    #[test]
    fn cpu_reference_full_selection_is_dense_equivalent() {
        let b = 2;
        let h = 3;
        let t_q = 2;
        let t_c = 48;
        let d = 8;
        let topk = 64; // >= t_c → forced full selection (+ 16 pad slots)

        let mut state: u64 = 0x5EED_1234_ABCD_0001;
        let mut rand = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32) * 4.0 - 2.0
        };
        let q: Vec<f32> = (0..b * h * t_q * d).map(|_| rand()).collect();
        let k: Vec<f32> = (0..b * h * t_c * d).map(|_| rand()).collect();
        let w: Vec<f32> = (0..b * h * t_q).map(|_| rand()).collect();

        let out = cpu_reference(
            &q,
            (b, h, t_q, d),
            &k,
            (b, h, t_c, d),
            &w,
            (b, h, t_q),
            topk,
        );
        assert_eq!(out.len(), b * t_q * topk);

        // (a) Every row selects the complete key set, then pads.
        for row in 0..(b * t_q) {
            let slice = &out[row * topk..(row + 1) * topk];
            let mut valid: Vec<u32> = slice[..t_c].to_vec();
            valid.sort_unstable();
            let full: Vec<u32> = (0..t_c as u32).collect();
            assert_eq!(
                valid, full,
                "row {row}: forced-full selection must return every key exactly once"
            );
            for (i, &pad) in slice[t_c..].iter().enumerate() {
                assert_eq!(pad, u32::MAX, "row {row}: slot {} must be padded", t_c + i);
            }
        }

        // (b) Softmax attention over the gathered subset == dense attention.
        //     The sorted selection is the identity permutation, so the
        //     accumulation order is untouched and equality is bit-exact.
        let attend = |keys: &[usize], av: &[f32], ak: &[f32], head: usize| -> Vec<f32> {
            // av: [t_c, d] values; ak: [t_c, d] keys; query = av-independent.
            let q_row = &av[..d]; // reuse the first row as the query
            let mut logits: Vec<f32> = keys
                .iter()
                .map(|&c| {
                    let base = c * d;
                    let mut acc = 0.0f32;
                    for dd in 0..d {
                        acc += q_row[dd] * ak[base + dd];
                    }
                    acc
                })
                .collect();
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for l in logits.iter_mut() {
                *l = (*l - max).exp();
                denom += *l;
            }
            let mut outv = vec![0.0f32; d];
            for (pos, &c) in keys.iter().enumerate() {
                let base = c * d;
                let p = logits[pos] / denom;
                for dd in 0..d {
                    outv[dd] += p * av[base + dd];
                }
            }
            let _ = head;
            outv
        };

        // Use the (bi=0, hi=0) K block as both keys and values.
        let kv = &k[0..t_c * d];
        let dense_keys: Vec<usize> = (0..t_c).collect();
        let dense = attend(&dense_keys, kv, kv, 0);

        let mut sel: Vec<u32> = out[..t_c].to_vec();
        sel.sort_unstable();
        let sel_keys: Vec<usize> = sel.iter().map(|&i| i as usize).collect();
        let sparse = attend(&sel_keys, kv, kv, 0);

        assert_eq!(
            dense, sparse,
            "forced-full-selection attention must equal dense attention bit-exactly"
        );
    }

    /// Pin the scoring semantics on a hand-computed fixture where the OLD
    /// formula (`(q_h · k_c) * w_h`, no relu, top-k **per head**) and the NEW
    /// one (`Σ_h relu(q_h · k_c) * w_h`, one shared top-k) provably select
    /// **different** keys, so a regression cannot pass silently.
    ///
    /// B=1, T_q=1, n_heads=2, head_dim=1, T_c=2, w = [1, 1], q = 1 on both
    /// heads, k = [[3, -1], [-4, 2]] (head-major):
    ///
    /// | key | h0 dot | h1 dot | new: Σ relu·w | no-relu sum | old per-head |
    /// |-----|--------|--------|---------------|-------------|--------------|
    /// | c0  |  +3    |  -4    | 3 + 0 = **3** | 3 - 4 = -1  | h0 picks c0  |
    /// | c1  |  -1    |  +2    | 0 + 2 =   2   | -1 + 2 = +1 | h1 picks c1  |
    #[test]
    fn cpu_reference_relu_and_head_sum_pick_different_keys() {
        let (b, h, t_q, t_c, d) = (1usize, 2usize, 1usize, 2usize, 1usize);
        let q = vec![1.0f32, 1.0];
        let k = vec![3.0f32, -1.0, -4.0, 2.0];
        let w = vec![1.0f32, 1.0];

        let out = cpu_reference(&q, (b, h, t_q, d), &k, (b, h, t_c, d), &w, (b, h, t_q), 1);
        assert_eq!(out, vec![0u32], "new formula must select key c0");

        // The pre-fix formula, computed inline: no relu, summed over heads.
        let no_relu: Vec<f32> = (0..t_c)
            .map(|ci| (0..h).map(|hi| q[hi] * k[hi * t_c + ci] * w[hi]).sum())
            .collect();
        assert_eq!(no_relu, vec![-1.0f32, 1.0]);
        let no_relu_pick = if no_relu[0] >= no_relu[1] { 0u32 } else { 1u32 };
        assert_eq!(
            no_relu_pick, 1,
            "dropping the relu must select a DIFFERENT key"
        );
        assert_ne!(no_relu_pick, out[0]);

        // And per-head top-k (no relu) yields two DIFFERENT key sets — 2x the
        // gather traffic here, 64x at the V4 head count.
        let per_head: Vec<u32> = (0..h)
            .map(|hi| {
                let s: Vec<f32> = (0..t_c)
                    .map(|ci| q[hi] * k[hi * t_c + ci] * w[hi])
                    .collect();
                if s[0] >= s[1] {
                    0u32
                } else {
                    1u32
                }
            })
            .collect();
        assert_eq!(per_head, vec![0u32, 1u32]);
        assert_ne!(
            per_head[0], per_head[1],
            "fixture must make the heads disagree, or it proves nothing"
        );
    }

    /// CPU reference is deterministic.
    #[test]
    fn cpu_reference_is_deterministic() {
        let b = 1;
        let h = 2;
        let t_q = 2;
        let t_c = 6;
        let d = 4;
        let topk = 3;
        let q: Vec<f32> = (0..b * h * t_q * d).map(|i| (i as f32) * 0.123).collect();
        let k: Vec<f32> = (0..b * h * t_c * d)
            .map(|i| (i as f32) * 0.077 - 1.0)
            .collect();
        let scale: Vec<f32> = (0..b * h * t_q).map(|i| 1.0 + (i as f32) * 0.1).collect();

        let a = cpu_reference(
            &q,
            (b, h, t_q, d),
            &k,
            (b, h, t_c, d),
            &scale,
            (b, h, t_q),
            topk,
        );
        let b_ = cpu_reference(
            &q,
            (b, h, t_q, d),
            &k,
            (b, h, t_c, d),
            &scale,
            (b, h, t_q),
            topk,
        );
        assert_eq!(a, b_);
    }

    /// validate_dispatch rejects unsupported topk values.
    #[test]
    fn validate_dispatch_rejects_unsupported_topk() {
        assert!(validate_dispatch(512, 128).is_ok());
        assert!(validate_dispatch(64, 128).is_ok());
        // Not in SUPPORTED_TOPK.
        assert!(validate_dispatch(100, 128).is_err());
        assert!(validate_dispatch(0, 128).is_err());
    }

    /// validate_dispatch rejects unaligned head_dim.
    #[test]
    fn validate_dispatch_rejects_bad_head_dim() {
        assert!(validate_dispatch(512, 7).is_err()); // not multiple of 8
        assert!(validate_dispatch(512, 0).is_err()); // zero
                                                     // 128 and 64 are explicitly supported template instantiations.
        assert!(validate_dispatch(512, 128).is_ok());
        assert!(validate_dispatch(512, 64).is_ok());
        // Other multiples of 8 are accepted via the generic fallback path.
        assert!(validate_dispatch(512, 96).is_ok());
    }

    /// Top-k indices in [0, T_c) for non-pad slots.
    #[test]
    fn cpu_reference_indices_in_range() {
        let b = 1;
        let h = 1;
        let t_q = 1;
        let t_c = 1024;
        let d = 16;
        let topk = 128;
        let q: Vec<f32> = (0..d).map(|i| (i as f32) * 0.01).collect();
        let mut k = Vec::with_capacity(t_c * d);
        for ci in 0..t_c {
            for dd in 0..d {
                k.push(((ci * 7 + dd) as f32) * 0.003 - 0.5);
            }
        }
        let scale = vec![1.0f32];
        let out = cpu_reference(
            &q,
            (b, h, t_q, d),
            &k,
            (b, h, t_c, d),
            &scale,
            (b, h, t_q),
            topk,
        );
        assert_eq!(out.len(), topk);
        for &idx in &out {
            assert!((idx as usize) < t_c, "index {idx} out of range [0, {t_c})");
        }
        // No duplicate indices.
        let mut sorted = out.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), topk, "top-k must produce distinct indices");
    }
}
