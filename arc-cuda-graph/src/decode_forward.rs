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
use crate::weights::{DecodeConfig, ModelWeights};
#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::sys::CUstream;

/// Per-layer KV cache pointers for paged attention.
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct LayerKvCache {
    pub key_cache: u64,   // Standard: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
                          // TurboQuant: [num_blocks, num_kv_heads, packed_bytes, block_size] U8
    pub value_cache: u64, // Standard: [num_blocks, num_kv_heads, head_dim, block_size]
                          // TurboQuant: [num_blocks, num_kv_heads, packed_bytes, block_size] U8
    pub k_norms: u64,     // TurboQuant only: [num_blocks, num_kv_heads, block_size] F16, 0 if standard
    pub v_norms: u64,     // TurboQuant only: same shape, 0 if standard
}

/// All paged attention state needed for one decode step.
#[cfg(feature = "cuda")]
pub struct PagedAttentionState {
    pub layer_caches: Vec<LayerKvCache>,
    pub block_tables: u64,     // [num_seqs, max_num_blocks_per_seq] i32 (or u32 for turbo)
    pub context_lens: u64,     // [num_seqs] i32 (or u32 for turbo)
    pub slot_mappings: u64,    // [num_tokens] i64
    pub block_size: i32,
    pub max_context_len: i32,
    pub max_num_blocks_per_seq: i32,
    pub kv_block_stride: i32,  // stride[0] of key_cache
    pub kv_head_stride: i32,   // stride[1] of key_cache
    pub norm_block_stride: i32, // stride[0] of norms (TurboQuant, 0 if standard)
    pub norm_head_stride: i32,  // stride[1] of norms (TurboQuant, 0 if standard)
    pub x: i32,                // 16 / sizeof(element) — interleave factor for key cache
    pub is_turbo: bool,        // TurboQuant mode
}

/// Pre-allocated intermediate buffers for one decode step.
/// All at fixed GPU addresses — stable across graph replays.
#[cfg(feature = "cuda")]
pub struct DecodeBuffers {
    // Activation buffers (BF16, 2 bytes per element)
    pub hidden_a: u64,      // [batch, hidden_size]
    pub hidden_b: u64,      // [batch, hidden_size] (ping-pong)
    pub normed: u64,        // [batch, hidden_size]
    pub residual: u64,      // [batch, hidden_size]
    pub q: u64,             // [batch, num_heads * head_dim]
    pub k: u64,             // [batch, num_kv_heads * head_dim]
    pub v: u64,             // [batch, num_kv_heads * head_dim]
    pub attn_out: u64,      // [batch, num_heads * head_dim]
    pub q_f16: u64,         // [batch, num_heads * head_dim] F16 (TurboQuant needs F16 Q)
    pub attn_out_f32: u64,  // [batch, num_heads * head_dim] F32 (TurboQuant outputs F32)
    pub o_proj_out: u64,    // [batch, hidden_size]
    pub gate: u64,          // [batch, intermediate_size]
    pub up: u64,            // [batch, intermediate_size]
    pub mlp_act: u64,       // [batch, intermediate_size]
    pub down_out: u64,      // [batch, hidden_size]
    pub logits: u64,        // [batch, vocab_size] (BF16)

    // Input buffers (updated before each step)
    pub token_ids: u64,     // [batch] i32
    pub positions: u64,     // [batch] i32

    // RoPE tables (from the model, stable pointers)
    pub cos_table: u64,     // [max_seq_len, head_dim/2]
    pub sin_table: u64,     // [max_seq_len, head_dim/2]
    pub is_neox: bool,

    pub batch_size: usize,
}

/// cuBLASLt handle and workspace for the decode path.
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
        a: *const std::ffi::c_void,
        a_layout: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        b_layout: *const std::ffi::c_void,
        beta: *const std::ffi::c_void,
        c: *const std::ffi::c_void,
        c_layout: *const std::ffi::c_void,
        d: *mut std::ffi::c_void,
        d_layout: *const std::ffi::c_void,
        algo: *const std::ffi::c_void,
        workspace: *mut std::ffi::c_void,
        workspace_size: usize,
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
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        c: *const std::ffi::c_void,
        d: *const std::ffi::c_void,
        pref: *const std::ffi::c_void,
        count: i32,
        results: *mut [u8; 80], // cublasLtMatmulHeuristicResult_t is ~80 bytes
        found: *mut i32,
    ) -> u32;
}

// cuBLASLt constants
#[cfg(feature = "cuda")]
const CUBLAS_COMPUTE_32F: u32 = 68;
#[cfg(feature = "cuda")]
const CUDA_R_32F: u32 = 0;
#[cfg(feature = "cuda")]
const CUDA_R_16BF: u32 = 14;
#[cfg(feature = "cuda")]
const CUBLASLT_MATMUL_DESC_TRANSA: u32 = 0;
#[cfg(feature = "cuda")]
const CUBLAS_OP_T: i32 = 1;
#[cfg(feature = "cuda")]
const CUBLAS_OP_N: i32 = 0;
#[cfg(feature = "cuda")]
const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE: u32 = 1;

/// Execute a BF16 GEMM: D = A^T * B
/// A (weight): [M, K] row-major = [K, M] col-major → transA=T gives [M, K]
/// B (input): [K, N] col-major (batch=N on last dim)
/// D (output): [M, N] col-major
#[cfg(feature = "cuda")]
unsafe fn gemm_bf16(
    cublas: &CublasState,
    stream: CUstream,
    weight: u64,  // [out_dim, in_dim] row-major
    input: u64,   // [in_dim, batch] col-major = [batch, in_dim] row-major
    output: u64,  // [out_dim, batch] col-major = [batch, out_dim] row-major
    m: u64,       // out_dim
    n: u64,       // batch
    k: u64,       // in_dim
) {
    let mut desc: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    let transa = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
        &transa as *const _ as *const _, 4);
    let transb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, 1, // TRANSB
        &transb as *const _ as *const _, 4);

    // A: weight [M, K] row-major = [K, M] col-major, ld=K
    let mut a_layout: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatrixLayoutCreate(&mut a_layout, CUDA_R_16BF, k, m, k as i64);

    // B: input [K, N] col-major, ld=K
    let mut b_layout: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatrixLayoutCreate(&mut b_layout, CUDA_R_16BF, k, n, k as i64);

    // C/D: output [M, N] col-major, ld=M
    let mut c_layout: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatrixLayoutCreate(&mut c_layout, CUDA_R_16BF, m, n, m as i64);
    let mut d_layout: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatrixLayoutCreate(&mut d_layout, CUDA_R_16BF, m, n, m as i64);

    // Algorithm selection
    let mut pref: *mut std::ffi::c_void = std::ptr::null_mut();
    cublasLtMatmulPreferenceCreate(&mut pref);
    let ws = cublas.workspace_size;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE,
        &ws as *const _ as *const _, std::mem::size_of::<usize>());

    let mut heuristic = [0u8; 80];
    let mut algo_count: i32 = 0;
    cublasLtMatmulAlgoGetHeuristic(
        cublas.handle, desc, a_layout, b_layout, c_layout, d_layout,
        pref, 1, &mut heuristic, &mut algo_count,
    );

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
        heuristic.as_ptr() as *const _, // algo is first field
        cublas.workspace as *mut _,
        cublas.workspace_size,
        stream,
    );

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(d_layout);
    cublasLtMatrixLayoutDestroy(c_layout);
    cublasLtMatrixLayoutDestroy(b_layout);
    cublasLtMatrixLayoutDestroy(a_layout);
    cublasLtMatmulDescDestroy(desc);
}

/// Run the full decode forward pass for one step.
///
/// Input: token_ids and positions already in buffers.
/// Output: logits in buffers.logits (BF16).
///
/// This runs entirely on `stream` using raw device pointers.
/// No Candle tensors. No allocations. Pure kernel launches.
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
    let hs = cfg.hidden_size as u64;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let inter = cfg.intermediate_size as u64;
    let eps = cfg.rms_norm_eps;
    let _theta = cfg.rope_theta;

    // Step 0: Embedding lookup
    launch_gather_embedding_bf16(
        weights.embed_tokens as *const _,
        buffers.token_ids as *const i32,
        buffers.hidden_a as *mut _,
        hs as i32,
        bs as i32,
        stream,
    );

    let mut h_in = buffers.hidden_a;
    let mut h_out = buffers.hidden_b;

    for layer_idx in 0..cfg.num_layers {
        let lw = &weights.layers[layer_idx];

        // RMSNorm (input_layernorm)
        launch_fused_rmsnorm_residual_bf16(
            h_in as *const _, std::ptr::null(), lw.input_layernorm as *const _,
            buffers.normed as *mut _, std::ptr::null_mut(),
            hs as i32, bs as i32, eps, stream,
        );

        // QKV projections
        gemm_bf16(cublas, stream, lw.q_proj.ptr, buffers.normed, buffers.q,
            (nh * hd) as u64, bs, hs);
        gemm_bf16(cublas, stream, lw.k_proj.ptr, buffers.normed, buffers.k,
            (nkv * hd) as u64, bs, hs);
        gemm_bf16(cublas, stream, lw.v_proj.ptr, buffers.normed, buffers.v,
            (nkv * hd) as u64, bs, hs);

        // QK norm (if model uses it)
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

        // RoPE
        launch_gather_rope_decode_bf16(
            buffers.q as *mut _, buffers.k as *mut _,
            buffers.cos_table as *const _, buffers.sin_table as *const _,
            buffers.positions as *const i32,
            nh as i32, nkv as i32, hd as i32, hd as i32, (hd / 2) as i32,
            bs as i32, buffers.is_neox as i32, stream,
        );

        // Store K/V into the paged KV cache + run attention
        let kv_stride = (nkv * hd) as i32;
        let q_stride = (nh * hd) as i32;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let lc = &paged_attn.layer_caches[layer_idx];

        if paged_attn.is_turbo {
            // TurboQuant: quantize K/V into U8 cache with per-token norms
            turbo_reshape_and_cache(
                buffers.k as *const _, buffers.v as *const _,
                lc.key_cache as *const _, lc.value_cache as *const _,
                lc.k_norms as *const _, lc.v_norms as *const _,
                paged_attn.slot_mappings as *const i64,
                bs as i32, nkv as i32, hd as i32, paged_attn.block_size,
                kv_stride, kv_stride,
                paged_attn.kv_block_stride, paged_attn.kv_head_stride,
                paged_attn.norm_block_stride, paged_attn.norm_head_stride,
                stream,
                1, // dtype = BF16 input
            );
            // Cast Q from BF16 → F16 (TurboQuant kernel expects F16)
            let q_size = (bs as usize * nh * hd) as i32;
            launch_cast_bf16_to_f16(
                buffers.q as *const _, buffers.q_f16 as *mut _, q_size, stream,
            );
            // TurboQuant attention: F16 Q + U8 cache → F32 output
            turbo_paged_attention_v1_f16(
                buffers.attn_out_f32 as *const _, buffers.q_f16 as *const _,
                lc.key_cache as *const _, lc.value_cache as *const _,
                lc.k_norms as *const _, lc.v_norms as *const _,
                nkv as i32, scale, 1.0,
                paged_attn.block_tables as *const u32,
                paged_attn.context_lens as *const u32,
                paged_attn.block_size, paged_attn.max_context_len,
                bs as i32, nh as i32, hd as i32, paged_attn.max_num_blocks_per_seq,
                q_stride,
                paged_attn.kv_block_stride, paged_attn.kv_head_stride,
                paged_attn.norm_block_stride, paged_attn.norm_head_stride,
                stream,
            );
            // Cast attention output F32 → BF16 for O projection
            launch_cast_f32_to_bf16(
                buffers.attn_out_f32 as *const _, buffers.attn_out as *mut _, q_size, stream,
            );
        } else {
            // Standard BF16 cache
            reshape_and_cache(
                buffers.k as *const _, buffers.v as *const _,
                lc.key_cache as *const _, lc.value_cache as *const _,
                paged_attn.slot_mappings as *const i64,
                bs as i32, nkv as i32, hd as i32, paged_attn.block_size, paged_attn.x,
                kv_stride, kv_stride,
                stream, 1, 1, // dtype=BF16, cache_dtype=BF16
                std::ptr::null(), std::ptr::null(),
            );
            paged_attention_v1_bf16(
                buffers.attn_out as *const _, buffers.q as *const _,
                lc.key_cache as *const _, lc.value_cache as *const _,
                std::ptr::null(), // alibi_slopes
                nkv as i32, scale, 1.0,
                paged_attn.block_tables as *const i32,
                paged_attn.context_lens as *const i32,
                paged_attn.block_size, paged_attn.max_context_len,
                bs as i32, nh as i32, hd as i32, paged_attn.max_num_blocks_per_seq,
                q_stride,
                paged_attn.kv_block_stride, paged_attn.kv_head_stride,
                stream, 1, // cache_dtype=BF16
                std::ptr::null(), std::ptr::null(), std::ptr::null(),
            );
        }

        // O projection
        gemm_bf16(cublas, stream, lw.o_proj.ptr, buffers.attn_out, buffers.o_proj_out,
            hs, bs, (nh * hd) as u64);

        // Residual add
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

        // MLP: gate + up + silu_mul + down
        gemm_bf16(cublas, stream, lw.gate_proj.ptr, buffers.normed, buffers.gate,
            inter, bs, hs);
        gemm_bf16(cublas, stream, lw.up_proj.ptr, buffers.normed, buffers.up,
            inter, bs, hs);

        launch_fused_silu_mul_bf16(
            buffers.gate as *const _, buffers.up as *const _,
            buffers.mlp_act as *mut _, (bs * inter) as i32, stream,
        );

        gemm_bf16(cublas, stream, lw.down_proj.ptr, buffers.mlp_act, buffers.down_out,
            hs, bs, inter);

        // Residual add
        launch_residual_add_bf16(
            buffers.residual as *const _, buffers.down_out as *const _,
            h_out as *mut _, (bs * hs) as i32, stream,
        );

        // Swap ping-pong
        std::mem::swap(&mut h_in, &mut h_out);
    }

    // Final RMSNorm
    launch_fused_rmsnorm_residual_bf16(
        h_in as *const _, std::ptr::null(),
        weights.final_norm as *const _,
        buffers.normed as *mut _, std::ptr::null_mut(),
        hs as i32, bs as i32, eps, stream,
    );

    // LM head
    gemm_bf16(cublas, stream, weights.lm_head.ptr, buffers.normed, buffers.logits,
        cfg.vocab_size as u64, bs, hs);
}

// Paged attention FFI from mistralrs-paged-attn
#[cfg(feature = "cuda")]
extern "C" {
    fn reshape_and_cache(
        key: *const std::ffi::c_void,
        value: *const std::ffi::c_void,
        key_cache: *const std::ffi::c_void,
        value_cache: *const std::ffi::c_void,
        slot_mapping: *const i64,
        num_tokens: i32,
        num_heads: i32,
        head_size: i32,
        block_size: i32,
        x: i32,
        key_stride: i32,
        value_stride: i32,
        stream: CUstream,
        dtype: u32,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
    );
    fn paged_attention_v1_bf16(
        out: *const std::ffi::c_void,
        query: *const std::ffi::c_void,
        key_cache: *const std::ffi::c_void,
        value_cache: *const std::ffi::c_void,
        alibi_slopes: *const std::ffi::c_void,
        num_kv_heads: i32,
        scale: f32,
        softcapping: f32,
        block_tables: *const i32,
        context_lens: *const i32,
        block_size: i32,
        max_context_len: i32,
        num_seqs: i32,
        num_heads: i32,
        head_size: i32,
        max_num_blocks_per_seq: i32,
        q_stride: i32,
        kv_block_stride: i32,
        kv_head_stride: i32,
        stream: CUstream,
        cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        sinks: *const f32,
    );
    fn turbo_reshape_and_cache(
        key: *const std::ffi::c_void,
        value: *const std::ffi::c_void,
        key_cache: *const std::ffi::c_void,
        value_cache: *const std::ffi::c_void,
        k_norms: *const std::ffi::c_void,
        v_norms: *const std::ffi::c_void,
        slot_mapping: *const i64,
        num_tokens: i32,
        num_heads: i32,
        head_size: i32,
        block_size: i32,
        key_stride: i32,
        value_stride: i32,
        kv_block_stride: i32,
        kv_head_stride: i32,
        norm_block_stride: i32,
        norm_head_stride: i32,
        stream: CUstream,
        dtype: u32,
    );
    fn turbo_paged_attention_v1_f16(
        out: *const std::ffi::c_void,
        query: *const std::ffi::c_void,
        k_cache: *const std::ffi::c_void,
        v_cache: *const std::ffi::c_void,
        k_norms: *const std::ffi::c_void,
        v_norms: *const std::ffi::c_void,
        num_kv_heads: i32,
        scale: f32,
        softcapping: f32,
        block_tables: *const u32,
        context_lens: *const u32,
        block_size: i32,
        max_context_len: i32,
        num_seqs: i32,
        num_heads: i32,
        head_size: i32,
        max_num_blocks_per_seq: i32,
        q_stride: i32,
        kv_block_stride: i32,
        kv_head_stride: i32,
        norm_block_stride: i32,
        norm_head_stride: i32,
        stream: CUstream,
    );
}

// Additional kernel FFI from decode_kernels.cu (reuse existing)
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
