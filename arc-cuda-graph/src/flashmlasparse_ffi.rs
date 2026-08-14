//! FFI bindings for the FlashMLASparse CUDA kernels.
//!
//! These symbols are defined in
//! `arc-cuda-graph/src/cuda/flashmlasparse/{indexer_score,topk_radix}.cu` and
//! get linked into `libarccudagraph.a` by the build script.
//!
//! The kernels are vendored from MIT-licensed upstream sources
//! (sgl-project/FlashMLA and sgl-project/sglang). See
//! `src/cuda/flashmlasparse/LICENSE-MIT` for the full attribution.
//!
//! # Surface
//!
//! Two C entry points:
//!
//! - `arc_flashmlasparse_logits_bf16` — Per (B, T_q) indexer logits:
//!   `logits[b,q,c] = Σ_h relu(dot(q[b,h,q,:], k[b,h,c,:])) * w[b,h,q]`.
//!   FP32 output for the top-k consumer. Note the **absent head axis**: the
//!   heads are relu'd and summed inside the kernel so a single key set is
//!   selected and shared by all heads (SGLang `dsv4/indexer.py` lines 84-89).
//! - `arc_flashmlasparse_topk` — Per (B*T_q) radix top-k selecting `topk`
//!   largest-score indices into [0, T_c). Output is I32 with -1 padding for
//!   rows where `seq_lens[row] < topk`.
//!
//! Both are non-blocking on the supplied CUDA stream. The high-level Rust
//! wrapper in `flashmlasparse.rs` owns the device buffers, validates shapes,
//! and exposes a single `forward(q, k, weights) -> top_k_indices` call that
//! the V4Indexer dispatches into when CUDA is available.

#![cfg(feature = "cuda")]

use std::ffi::c_void;

extern "C" {
    /// BF16 indexer logits kernel.
    ///
    /// Computes `logits[b,q,c] = Σ_h relu(q[b,h,q,:] · k[b,h,c,:]) * w[b,h,q]`.
    ///
    /// # Arguments
    /// - `q_ptr` — device ptr to `[B, n_heads, T_q, head_dim]` BF16 (contiguous).
    /// - `k_ptr` — device ptr to `[B, n_heads, T_c, head_dim]` BF16 (contiguous).
    /// - `weights_ptr` — device ptr to `[B, n_heads, T_q]` BF16 (contiguous).
    ///   `weight_scale = softmax_scale * n_heads^-0.5` must already be folded
    ///   in by the caller, as SGLang's `compute_weights` / `fused_scale` does.
    /// - `logits_ptr` — device ptr to `[B, T_q, T_c]` F32 (contiguous, written).
    /// - `batch`, `n_heads`, `t_q`, `t_c`, `head_dim` — int dimensions.
    /// - `stream` — CUstream (may be the default 0).
    pub fn arc_flashmlasparse_logits_bf16(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        weights_ptr: *const c_void,
        logits_ptr: *mut c_void,
        batch: i32,
        n_heads: i32,
        t_q: i32,
        t_c: i32,
        head_dim: i32,
        stream: *mut c_void,
    );

    /// FP32 indexer logits kernel (parity-test path).
    ///
    /// Same signature semantics as the BF16 variant; inputs and `weights`
    /// are FP32 instead of BF16. Used by the CPU-reference agreement test.
    pub fn arc_flashmlasparse_logits_f32(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        weights_ptr: *const c_void,
        logits_ptr: *mut c_void,
        batch: i32,
        n_heads: i32,
        t_q: i32,
        t_c: i32,
        head_dim: i32,
        stream: *mut c_void,
    );

    /// Radix top-k kernel.
    ///
    /// # Arguments
    /// - `scores_ptr` — F32 device ptr to `[n_rows, score_stride]` scores.
    /// - `seq_lens_ptr` — I32 device ptr to `[n_rows]` valid lengths
    ///   (effective T_c per row). Rows with `seq_lens[r] < topk` are padded
    ///   with -1.
    /// - `out_indices_ptr` — I32 device ptr to `[n_rows, topk]` output.
    /// - `n_rows` = B * T_q for the indexer (one row per query, heads already
    ///   reduced); the sampler path passes its own row count.
    /// - `topk` — must be one of {64, 128, 256, 512, 1024}.
    /// - `score_stride` — leading-dim stride into `scores` (= T_c for our use).
    /// - `stream` — CUstream.
    pub fn arc_flashmlasparse_topk(
        scores_ptr: *const c_void,
        seq_lens_ptr: *const c_void,
        out_indices_ptr: *mut c_void,
        n_rows: i32,
        topk: i32,
        score_stride: i64,
        stream: *mut c_void,
    );
}
