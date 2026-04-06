//! Dedicated decode forward pass — bypasses Candle entirely.
//!
//! Reads model weights via raw device pointers.
//! Runs on its own non-blocking stream (capturable).
//! Uses cuBLASLt for GEMMs, custom CUDA kernels for everything else.
//! Zero allocations during inference (all buffers pre-allocated).
//!
//! This is the hot path. Every microsecond matters.

#[cfg(feature = "cuda")]
use crate::ffi::*;
#[cfg(feature = "cuda")]
use crate::weights::ModelWeights;
#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::sys::CUstream;

/// Per-layer KV cache pointers for paged attention.
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct LayerKvCache {
    pub key_cache: u64,
    pub value_cache: u64,
    pub k_norms: u64,  // TurboQuant: [num_blocks, num_kv_heads, block_size] F16, 0 if standard
    pub v_norms: u64,  // TurboQuant: same, 0 if standard
}

/// All paged attention state needed for one decode step.
#[cfg(feature = "cuda")]
pub struct PagedAttentionState {
    pub layer_caches: Vec<LayerKvCache>,
    pub block_tables: u64,
    pub context_lens: u64,
    pub slot_mappings: u64,
    pub block_size: i32,
    pub max_context_len: i32,
    pub max_num_blocks_per_seq: i32,
    pub kv_block_stride: i32,
    pub kv_head_stride: i32,
    pub norm_block_stride: i32,
    pub norm_head_stride: i32,
    pub x: i32,
    pub is_turbo: bool,
}

/// Pre-allocated intermediate buffers for one decode step.
/// All at fixed GPU addresses — stable across graph replays.
#[cfg(feature = "cuda")]
pub struct DecodeBuffers {
    pub hidden_a: u64,
    pub hidden_b: u64,
    pub normed: u64,
    pub residual: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub attn_out: u64,
    pub q_f16: u64,
    pub k_f16: u64,
    pub v_f16: u64,
    pub attn_out_f32: u64,
    pub o_proj_out: u64,
    pub gate: u64,
    pub up: u64,
    pub mlp_act: u64,
    pub down_out: u64,
    pub logits: u64,
    pub token_ids: u64,
    pub positions: u64,
    pub cos_table: u64,
    pub sin_table: u64,
    pub is_neox: bool,
    pub batch_size: usize,
}

/// cuBLAS handle and workspace for the decode path.
#[cfg(feature = "cuda")]
pub struct CublasState {
    pub handle: *mut std::ffi::c_void, // cublasLtHandle_t
    pub workspace: u64,
    pub workspace_size: usize,
}

// cuBLASLt FFI
#[cfg(feature = "cuda")]
extern "C" {
    fn cublasLtCreate(handle: *mut *mut std::ffi::c_void) -> u32;
    fn cublasLtDestroy(handle: *mut std::ffi::c_void) -> u32;
    fn cublasLtMatmul(
        handle: *const std::ffi::c_void,
        desc: *const std::ffi::c_void,
        alpha: *const std::ffi::c_void,
        a: *const std::ffi::c_void, a_layout: *const std::ffi::c_void,
        b: *const std::ffi::c_void, b_layout: *const std::ffi::c_void,
        beta: *const std::ffi::c_void,
        c: *const std::ffi::c_void, c_layout: *const std::ffi::c_void,
        d: *mut std::ffi::c_void, d_layout: *const std::ffi::c_void,
        algo: *const std::ffi::c_void,
        workspace: *mut std::ffi::c_void, workspace_size: usize,
        stream: CUstream,
    ) -> u32;
    fn cublasLtMatmulDescCreate(desc: *mut *mut std::ffi::c_void, compute: u32, scale: u32) -> u32;
    fn cublasLtMatmulDescDestroy(desc: *mut std::ffi::c_void) -> u32;
    fn cublasLtMatmulDescSetAttribute(
        desc: *const std::ffi::c_void, attr: u32,
        buf: *const std::ffi::c_void, size: usize,
    ) -> u32;
    fn cublasLtMatrixLayoutCreate(
        layout: *mut *mut std::ffi::c_void,
        dtype: u32, rows: u64, cols: u64, ld: i64,
    ) -> u32;
    fn cublasLtMatrixLayoutDestroy(layout: *mut std::ffi::c_void) -> u32;
    fn cublasLtMatmulPreferenceCreate(pref: *mut *mut std::ffi::c_void) -> u32;
    fn cublasLtMatmulPreferenceDestroy(pref: *mut std::ffi::c_void) -> u32;
    fn cublasLtMatmulPreferenceSetAttribute(
        pref: *const std::ffi::c_void, attr: u32,
        buf: *const std::ffi::c_void, size: usize,
    ) -> u32;
    fn cublasLtMatmulAlgoGetHeuristic(
        handle: *const std::ffi::c_void,
        desc: *const std::ffi::c_void,
        a: *const std::ffi::c_void, b: *const std::ffi::c_void,
        c: *const std::ffi::c_void, d: *const std::ffi::c_void,
        pref: *const std::ffi::c_void,
        count: i32, results: *mut [u8; 80], found: *mut i32,
    ) -> u32;
}

// cuBLASLt constants
#[cfg(feature = "cuda")]
const CUBLAS_COMPUTE_32F: u32 = 68;
#[cfg(feature = "cuda")]
const CUDA_R_32F: u32 = 0;
#[cfg(feature = "cuda")]
const CUDA_R_16BF: u32 = 14;

/// Minimum N for cuBLASLt (padded from actual batch size).
/// cuBLASLt returns NOT_SUPPORTED for N=1 with BF16 on Blackwell.
/// N=8 is the smallest that reliably works and is still bandwidth-bound.
#[cfg(feature = "cuda")]
pub const GEMM_PAD_N: u64 = 8;

/// Execute a BF16 GEMM: D = A^T * B using cuBLASLt (graph-capture compatible).
///
/// Pads N to GEMM_PAD_N if needed. Input buffer must have room for padded N
/// (pre-allocated by DedicatedDecodePath). Only the first actual_n rows of
/// output are meaningful.
#[cfg(feature = "cuda")]
unsafe fn gemm_bf16(
    cublas: &CublasState,
    stream: CUstream,
    weight: u64,  // [out_dim, in_dim] row-major
    input: u64,   // [in_dim, padded_n] col-major
    output: u64,  // [out_dim, padded_n] col-major
    m: u64,       // out_dim
    n: u64,       // padded batch (>= GEMM_PAD_N)
    k: u64,       // in_dim
) {
    let mut desc: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    let transa: i32 = 1; // CUBLAS_OP_T
    cublasLtMatmulDescSetAttribute(desc, 0, &transa as *const _ as *const _, 4);
    let transb: i32 = 0; // CUBLAS_OP_N
    cublasLtMatmulDescSetAttribute(desc, 1, &transb as *const _ as *const _, 4);

    let mut a_layout: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatrixLayoutCreate(&mut a_layout, CUDA_R_16BF, k, m, k as i64);

    let mut b_layout: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatrixLayoutCreate(&mut b_layout, CUDA_R_16BF, k, n, k as i64);

    let mut c_layout: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatrixLayoutCreate(&mut c_layout, CUDA_R_16BF, m, n, m as i64);
    let mut d_layout: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatrixLayoutCreate(&mut d_layout, CUDA_R_16BF, m, n, m as i64);

    let mut pref: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatmulPreferenceCreate(&mut pref);
    let ws = cublas.workspace_size;
    cublasLtMatmulPreferenceSetAttribute(pref, 1, &ws as *const _ as *const _, std::mem::size_of::<usize>());

    let mut heuristic = [0u8; 80];
    let mut algo_count: i32 = 0;
    let s = cublasLtMatmulAlgoGetHeuristic(
        cublas.handle, desc, a_layout, b_layout, c_layout, d_layout,
        pref, 1, &mut heuristic, &mut algo_count,
    );
    if s != 0 || algo_count == 0 {
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::error!("cuBLASLt heuristic: status={s} count={algo_count} m={m} n={n} k={k}");
        }
    }

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    cublasLtMatmul(
        cublas.handle, desc,
        &alpha as *const _ as *const _,
        weight as *const _, a_layout,
        input as *const _, b_layout,
        &beta as *const _ as *const _,
        output as *const _, c_layout,
        output as *mut _, d_layout,
        heuristic.as_ptr() as *const _,
        cublas.workspace as *mut _, cublas.workspace_size,
        stream,
    );

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(d_layout);
    cublasLtMatrixLayoutDestroy(c_layout);
    cublasLtMatrixLayoutDestroy(b_layout);
    cublasLtMatrixLayoutDestroy(a_layout);
    cublasLtMatmulDescDestroy(desc);
}

// Paged attention FFI from mistralrs-paged-attn
#[cfg(feature = "cuda")]
extern "C" {
    fn reshape_and_cache(
        key: *const std::ffi::c_void, value: *const std::ffi::c_void,
        key_cache: *const std::ffi::c_void, value_cache: *const std::ffi::c_void,
        slot_mapping: *const i64,
        num_tokens: i32, num_heads: i32, head_size: i32,
        block_size: i32, x: i32, key_stride: i32, value_stride: i32,
        stream: CUstream, dtype: u32, cache_dtype: u32,
        k_scale: *const f32, v_scale: *const f32,
    );
    fn paged_attention_v1_bf16(
        out: *const std::ffi::c_void, query: *const std::ffi::c_void,
        key_cache: *const std::ffi::c_void, value_cache: *const std::ffi::c_void,
        alibi_slopes: *const std::ffi::c_void,
        num_kv_heads: i32, scale: f32, softcapping: f32,
        block_tables: *const i32, context_lens: *const i32,
        block_size: i32, max_context_len: i32,
        num_seqs: i32, num_heads: i32, head_size: i32,
        max_num_blocks_per_seq: i32, q_stride: i32,
        kv_block_stride: i32, kv_head_stride: i32,
        stream: CUstream, cache_dtype: u32,
        k_scale: *const f32, v_scale: *const f32, sinks: *const f32,
    );
    fn turbo_reshape_and_cache(
        key: *const std::ffi::c_void, value: *const std::ffi::c_void,
        key_cache: *const std::ffi::c_void, value_cache: *const std::ffi::c_void,
        k_norms: *const std::ffi::c_void, v_norms: *const std::ffi::c_void,
        slot_mapping: *const i64,
        num_tokens: i32, num_heads: i32, head_size: i32,
        block_size: i32, key_stride: i32, value_stride: i32,
        kv_block_stride: i32, kv_head_stride: i32,
        norm_block_stride: i32, norm_head_stride: i32,
        stream: CUstream, dtype: u32,
    );
    fn turbo_paged_attention_v1_f16(
        out: *const std::ffi::c_void, query: *const std::ffi::c_void,
        k_cache: *const std::ffi::c_void, v_cache: *const std::ffi::c_void,
        k_norms: *const std::ffi::c_void, v_norms: *const std::ffi::c_void,
        num_kv_heads: i32, scale: f32, softcapping: f32,
        block_tables: *const u32, context_lens: *const u32,
        block_size: i32, max_context_len: i32,
        num_seqs: i32, num_heads: i32, head_size: i32,
        max_num_blocks_per_seq: i32, q_stride: i32,
        kv_block_stride: i32, kv_head_stride: i32,
        norm_block_stride: i32, norm_head_stride: i32,
        stream: CUstream,
    );
}

// Custom CUDA kernel launchers
#[cfg(feature = "cuda")]
extern "C" {
    fn launch_gather_embedding_bf16(
        embed_table: *const std::ffi::c_void, token_ids: *const i32,
        output: *mut std::ffi::c_void,
        hidden_size: i32, batch_size: i32, stream: CUstream,
    );
    fn launch_fused_rmsnorm_residual_bf16(
        input: *const std::ffi::c_void, residual: *const std::ffi::c_void,
        weight: *const std::ffi::c_void,
        output: *mut std::ffi::c_void, residual_out: *mut std::ffi::c_void,
        hidden_size: i32, batch_size: i32, eps: f32, stream: CUstream,
    );
    fn launch_rmsnorm_head_bf16(
        input: *const std::ffi::c_void, weight: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        head_dim: i32, total_heads: i32, eps: f32, stream: CUstream,
    );
    fn launch_fused_silu_mul_bf16(
        gate: *const std::ffi::c_void, up: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        size: i32, stream: CUstream,
    );
    fn launch_residual_add_bf16(
        a: *const std::ffi::c_void, b: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        size: i32, stream: CUstream,
    );
    fn launch_cast_bf16_to_f16(
        input: *const std::ffi::c_void, output: *mut std::ffi::c_void,
        size: i32, stream: CUstream,
    );
    fn launch_cast_f32_to_bf16(
        input: *const std::ffi::c_void, output: *mut std::ffi::c_void,
        size: i32, stream: CUstream,
    );
}

/// Run the full decode forward pass for one step.
///
/// All GEMMs use padded N (GEMM_PAD_N) for cuBLASLt compatibility.
/// Custom kernels use the real batch_size (1).
#[cfg(feature = "cuda")]
pub unsafe fn decode_forward(
    weights: &ModelWeights,
    buffers: &DecodeBuffers,
    cublas: &CublasState,
    paged_attn: &PagedAttentionState,
    stream: CUstream,
) {
    let cfg = &weights.config;
    let bs = buffers.batch_size as u64;
    let pn = GEMM_PAD_N; // Padded N for GEMMs
    let hs = cfg.hidden_size as u64;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let inter = cfg.intermediate_size as u64;
    let eps = cfg.rms_norm_eps;

    // Step 0: Embedding lookup (uses real batch_size)
    launch_gather_embedding_bf16(
        weights.embed_tokens as *const _,
        buffers.token_ids as *const i32,
        buffers.hidden_a as *mut _,
        hs as i32, bs as i32, stream,
    );

    let mut h_in = buffers.hidden_a;
    let mut h_out = buffers.hidden_b;

    for layer_idx in 0..cfg.num_layers {
        let lw = &weights.layers[layer_idx];

        // RMSNorm (uses real batch_size)
        launch_fused_rmsnorm_residual_bf16(
            h_in as *const _, std::ptr::null(), lw.input_layernorm as *const _,
            buffers.normed as *mut _, std::ptr::null_mut(),
            hs as i32, bs as i32, eps, stream,
        );

        // QKV GEMMs (padded N for cuBLASLt)
        gemm_bf16(cublas, stream, lw.q_proj.ptr, buffers.normed, buffers.q,
            (nh * hd) as u64, pn, hs);
        gemm_bf16(cublas, stream, lw.k_proj.ptr, buffers.normed, buffers.k,
            (nkv * hd) as u64, pn, hs);
        gemm_bf16(cublas, stream, lw.v_proj.ptr, buffers.normed, buffers.v,
            (nkv * hd) as u64, pn, hs);

        // QK norm (if model uses it) — uses real batch_size
        if let (Some(qn), Some(kn)) = (lw.q_norm, lw.k_norm) {
            launch_rmsnorm_head_bf16(
                buffers.q as *const _, qn as *const _, buffers.q as *mut _,
                hd as i32, (bs as usize * nh) as i32, eps, stream,
            );
            launch_rmsnorm_head_bf16(
                buffers.k as *const _, kn as *const _, buffers.k as *mut _,
                hd as i32, (bs as usize * nkv) as i32, eps, stream,
            );
        }

        // RoPE (uses real batch_size)
        launch_gather_rope_decode_bf16(
            buffers.q as *mut _, buffers.k as *mut _,
            buffers.cos_table as *const _, buffers.sin_table as *const _,
            buffers.positions as *const i32,
            nh as i32, nkv as i32, hd as i32, hd as i32, (hd / 2) as i32,
            bs as i32, buffers.is_neox as i32, stream,
        );

        // Attention (uses real batch_size)
        let kv_stride = (nkv * hd) as i32;
        let q_stride = (nh * hd) as i32;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let lc = &paged_attn.layer_caches[layer_idx];

        if paged_attn.is_turbo {
            let kv_size = (bs as usize * nkv * hd) as i32;
            launch_cast_bf16_to_f16(buffers.k as *const _, buffers.k_f16 as *mut _, kv_size, stream);
            launch_cast_bf16_to_f16(buffers.v as *const _, buffers.v_f16 as *mut _, kv_size, stream);
            turbo_reshape_and_cache(
                buffers.k_f16 as *const _, buffers.v_f16 as *const _,
                lc.key_cache as *const _, lc.value_cache as *const _,
                lc.k_norms as *const _, lc.v_norms as *const _,
                paged_attn.slot_mappings as *const i64,
                bs as i32, nkv as i32, hd as i32, paged_attn.block_size,
                kv_stride, kv_stride,
                paged_attn.kv_block_stride, paged_attn.kv_head_stride,
                paged_attn.norm_block_stride, paged_attn.norm_head_stride,
                stream, 0,
            );
            let q_size = (bs as usize * nh * hd) as i32;
            launch_cast_bf16_to_f16(buffers.q as *const _, buffers.q_f16 as *mut _, q_size, stream);
            turbo_paged_attention_v1_f16(
                buffers.attn_out_f32 as *const _, buffers.q_f16 as *const _,
                lc.key_cache as *const _, lc.value_cache as *const _,
                lc.k_norms as *const _, lc.v_norms as *const _,
                nkv as i32, scale, 1.0,
                paged_attn.block_tables as *const u32, paged_attn.context_lens as *const u32,
                paged_attn.block_size, paged_attn.max_context_len,
                bs as i32, nh as i32, hd as i32, paged_attn.max_num_blocks_per_seq,
                q_stride,
                paged_attn.kv_block_stride, paged_attn.kv_head_stride,
                paged_attn.norm_block_stride, paged_attn.norm_head_stride,
                stream,
            );
            launch_cast_f32_to_bf16(buffers.attn_out_f32 as *const _, buffers.attn_out as *mut _, q_size, stream);
        } else {
            reshape_and_cache(
                buffers.k as *const _, buffers.v as *const _,
                lc.key_cache as *const _, lc.value_cache as *const _,
                paged_attn.slot_mappings as *const i64,
                bs as i32, nkv as i32, hd as i32, paged_attn.block_size, paged_attn.x,
                kv_stride, kv_stride,
                stream, 1, 1,
                std::ptr::null(), std::ptr::null(),
            );
            paged_attention_v1_bf16(
                buffers.attn_out as *const _, buffers.q as *const _,
                lc.key_cache as *const _, lc.value_cache as *const _,
                std::ptr::null(),
                nkv as i32, scale, 1.0,
                paged_attn.block_tables as *const i32, paged_attn.context_lens as *const i32,
                paged_attn.block_size, paged_attn.max_context_len,
                bs as i32, nh as i32, hd as i32, paged_attn.max_num_blocks_per_seq,
                q_stride,
                paged_attn.kv_block_stride, paged_attn.kv_head_stride,
                stream, 1,
                std::ptr::null(), std::ptr::null(), std::ptr::null(),
            );
        }

        // O projection (padded N)
        gemm_bf16(cublas, stream, lw.o_proj.ptr, buffers.attn_out, buffers.o_proj_out,
            hs, pn, (nh * hd) as u64);

        // Residual add (real batch_size)
        launch_residual_add_bf16(
            h_in as *const _, buffers.o_proj_out as *const _,
            buffers.residual as *mut _, (bs * hs) as i32, stream,
        );

        // Post-attention RMSNorm
        launch_fused_rmsnorm_residual_bf16(
            buffers.residual as *const _, std::ptr::null(),
            lw.post_attn_layernorm as *const _,
            buffers.normed as *mut _, std::ptr::null_mut(),
            hs as i32, bs as i32, eps, stream,
        );

        // MLP GEMMs (padded N)
        gemm_bf16(cublas, stream, lw.gate_proj.ptr, buffers.normed, buffers.gate,
            inter, pn, hs);
        gemm_bf16(cublas, stream, lw.up_proj.ptr, buffers.normed, buffers.up,
            inter, pn, hs);

        launch_fused_silu_mul_bf16(
            buffers.gate as *const _, buffers.up as *const _,
            buffers.mlp_act as *mut _, (bs * inter) as i32, stream,
        );

        gemm_bf16(cublas, stream, lw.down_proj.ptr, buffers.mlp_act, buffers.down_out,
            hs, pn, inter);

        // Residual add (real batch_size)
        launch_residual_add_bf16(
            buffers.residual as *const _, buffers.down_out as *const _,
            h_out as *mut _, (bs * hs) as i32, stream,
        );

        std::mem::swap(&mut h_in, &mut h_out);
    }

    // Final RMSNorm
    launch_fused_rmsnorm_residual_bf16(
        h_in as *const _, std::ptr::null(),
        weights.final_norm as *const _,
        buffers.normed as *mut _, std::ptr::null_mut(),
        hs as i32, bs as i32, eps, stream,
    );

    // LM head (padded N)
    gemm_bf16(cublas, stream, weights.lm_head.ptr, buffers.normed, buffers.logits,
        cfg.vocab_size as u64, pn, hs);
}
