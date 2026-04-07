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
    pub qkv: u64,           // [batch, q_dim + k_dim + v_dim] fused output
    pub q: u64,             // alias into qkv (set at runtime)
    pub k: u64,             // alias into qkv
    pub v: u64,             // alias into qkv
    pub attn_out: u64,      // [batch, num_heads * head_dim]
    pub attn_out_f32: u64,  // [batch, num_heads * head_dim] F32 (TurboQuant outputs F32)
    pub o_proj_out: u64,    // [batch, hidden_size]
    pub gate: u64,          // [batch, intermediate_size]
    pub up: u64,            // [batch, intermediate_size]
    pub mlp_act: u64,       // [batch, intermediate_size]
    pub down_out: u64,      // [batch, hidden_size]
    pub logits: u64,        // [batch, vocab_size] (BF16)
    pub logits_f32: u64,    // [batch, vocab_size] (F32) — pre-allocated for output

    // Input buffers (updated before each step)
    pub token_ids: u64,     // [batch] i32
    pub positions: u64,     // [batch] i32

    // RoPE tables (from the model, stable pointers)
    pub cos_table: u64,     // [max_seq_len, head_dim/2]
    pub sin_table: u64,     // [max_seq_len, head_dim/2]
    pub is_neox: bool,

    pub batch_size: usize,
}

// Custom GEMV FFI — graph-capture compatible, no cuBLAS
#[cfg(feature = "cuda")]
extern "C" {
    fn arc_launch_gemv_bf16(
        weight: *const std::ffi::c_void,
        input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        m: i32, k: i32, stream: CUstream,
    );
    fn arc_launch_gemv_bf16_f32out(
        weight: *const std::ffi::c_void,
        input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        m: i32, k: i32, stream: CUstream,
    );
    // Fused RMSNorm + GEMV (eliminates separate norm kernel + L2 round-trip)
    fn arc_launch_rmsnorm_gemv_bf16(
        input: *const std::ffi::c_void,
        norm_weight: *const std::ffi::c_void,
        weight: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        m: i32, k: i32, eps: f32, stream: CUstream,
    );
    // Fused Q norm + K norm + RoPE (q_norm_w/k_norm_w nullable for non-qknorm models)
    fn arc_launch_qknorm_rope_bf16(
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        q_norm_w: *const std::ffi::c_void,
        k_norm_w: *const std::ffi::c_void,
        cos_table: *const std::ffi::c_void,
        sin_table: *const std::ffi::c_void,
        positions: *const i32,
        num_heads: i32, num_kv_heads: i32, head_dim: i32,
        batch_size: i32, is_neox: i32, eps: f32,
        stream: CUstream,
    );
}

/// BF16 GEMV: output[0..m] = weight[m,k] * input[0..k]. Graph-capturable.
#[cfg(feature = "cuda")]
#[inline(always)]
unsafe fn gemv(stream: CUstream, weight: u64, input: u64, output: u64, m: usize, k: usize) {
    arc_launch_gemv_bf16(weight as *const _, input as *const _, output as *mut _, m as i32, k as i32, stream);
}

/// Profile one forward pass with CUDA events. Runs once, then never again.
#[cfg(feature = "cuda")]
pub unsafe fn profile_forward(
    weights: &ModelWeights,
    buffers: &DecodeBuffers,
    paged_attn: &PagedAttentionState,
    stream: CUstream,
) {
    extern "C" {
        fn cudaEventCreate(event: *mut *mut std::ffi::c_void) -> u32;
        fn cudaEventRecord(event: *mut std::ffi::c_void, stream: CUstream) -> u32;
        fn cudaEventSynchronize(event: *mut std::ffi::c_void) -> u32;
        fn cudaEventElapsedTime(ms: *mut f32, start: *mut std::ffi::c_void, end: *mut std::ffi::c_void) -> u32;
        fn cudaEventDestroy(event: *mut std::ffi::c_void) -> u32;
    }

    let cfg = &weights.config;
    let bs = buffers.batch_size as u64;
    let hs = cfg.hidden_size as u64;
    let hs_z = cfg.hidden_size;
    let inter_z = cfg.intermediate_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let inter = cfg.intermediate_size as u64;
    let eps = cfg.rms_norm_eps;

    macro_rules! ev { () => {{ let mut e: *mut std::ffi::c_void = std::ptr::null_mut(); cudaEventCreate(&mut e); e }} }
    macro_rules! rec { ($e:expr) => { cudaEventRecord($e, stream); } }
    macro_rules! ms { ($a:expr, $b:expr) => {{ let mut t: f32 = 0.0; cudaEventSynchronize($b); cudaEventElapsedTime(&mut t, $a, $b); t }} }

    let e_start = ev!(); let e_embed = ev!(); let e_layer0 = ev!();
    let e_qkv = ev!(); let e_attn = ev!(); let e_oproj = ev!(); let e_mlp = ev!(); let e_layer_end = ev!();
    let e_final = ev!(); let e_lmhead = ev!(); let e_end = ev!();

    rec!(e_start);
    launch_gather_embedding_bf16(weights.embed_tokens as *const _, buffers.token_ids as *const i32, buffers.hidden_a as *mut _, hs as i32, bs as i32, stream);
    rec!(e_embed);

    let mut h_in = buffers.hidden_a; let mut h_out = buffers.hidden_b;
    for layer_idx in 0..cfg.num_layers {
        let lw = &weights.layers[layer_idx];
        if layer_idx == 0 { rec!(e_layer0); }

        // Norm + QKV
        launch_fused_rmsnorm_residual_bf16(h_in as *const _, std::ptr::null(), lw.input_layernorm as *const _, buffers.normed as *mut _, std::ptr::null_mut(), hs as i32, bs as i32, eps, stream);
        gemv(stream, lw.qkv_fused, buffers.normed, buffers.qkv, lw.qkv_rows, hs_z);
        let q_ptr = buffers.qkv;
        let k_ptr = buffers.qkv + (nh * hd * 2) as u64;
        let v_ptr = k_ptr + (nkv * hd * 2) as u64;
        if let (Some(qn), Some(kn)) = (lw.q_norm, lw.k_norm) {
            launch_rmsnorm_head_bf16(q_ptr as *const _, qn as *const _, q_ptr as *mut _, hd as i32, (bs as usize * nh) as i32, eps, stream);
            launch_rmsnorm_head_bf16(k_ptr as *const _, kn as *const _, k_ptr as *mut _, hd as i32, (bs as usize * nkv) as i32, eps, stream);
        }
        launch_gather_rope_decode_bf16(q_ptr as *mut _, k_ptr as *mut _, buffers.cos_table as *const _, buffers.sin_table as *const _, buffers.positions as *const i32, nh as i32, nkv as i32, hd as i32, hd as i32, (hd/2) as i32, bs as i32, buffers.is_neox as i32, stream);
        if layer_idx == 0 { rec!(e_qkv); }

        // Attention
        let kv_stride = (nkv * hd) as i32; let q_stride = (nh * hd) as i32; let scale = 1.0f32 / (hd as f32).sqrt();
        let lc = &paged_attn.layer_caches[layer_idx];
        if paged_attn.is_turbo {
            turbo_reshape_and_cache(k_ptr as *const _, v_ptr as *const _, lc.key_cache as *const _, lc.value_cache as *const _, lc.k_norms as *const _, lc.v_norms as *const _, paged_attn.slot_mappings as *const i64, bs as i32, nkv as i32, hd as i32, paged_attn.block_size, kv_stride, kv_stride, paged_attn.kv_block_stride, paged_attn.kv_head_stride, paged_attn.norm_block_stride, paged_attn.norm_head_stride, stream, 1);
            turbo_paged_attention_v1(buffers.attn_out_f32 as *const _, q_ptr as *const _, lc.key_cache as *const _, lc.value_cache as *const _, lc.k_norms as *const _, lc.v_norms as *const _, nkv as i32, scale, 1.0, paged_attn.block_tables as *const u32, paged_attn.context_lens as *const u32, paged_attn.block_size, paged_attn.max_context_len, bs as i32, nh as i32, hd as i32, paged_attn.max_num_blocks_per_seq, q_stride, paged_attn.kv_block_stride, paged_attn.kv_head_stride, paged_attn.norm_block_stride, paged_attn.norm_head_stride, stream, 1);
            launch_cast_f32_to_bf16(buffers.attn_out_f32 as *const _, buffers.attn_out as *mut _, (bs as usize * nh * hd) as i32, stream);
        } else {
            reshape_and_cache(k_ptr as *const _, v_ptr as *const _, lc.key_cache as *const _, lc.value_cache as *const _, paged_attn.slot_mappings as *const i64, bs as i32, nkv as i32, hd as i32, paged_attn.block_size, paged_attn.x, kv_stride, kv_stride, stream, 1, 1, std::ptr::null(), std::ptr::null());
            paged_attention_v1_bf16(buffers.attn_out as *const _, q_ptr as *const _, lc.key_cache as *const _, lc.value_cache as *const _, std::ptr::null(), nkv as i32, scale, 1.0, paged_attn.block_tables as *const i32, paged_attn.context_lens as *const i32, paged_attn.block_size, paged_attn.max_context_len, bs as i32, nh as i32, hd as i32, paged_attn.max_num_blocks_per_seq, q_stride, paged_attn.kv_block_stride, paged_attn.kv_head_stride, stream, 1, std::ptr::null(), std::ptr::null(), std::ptr::null());
        }
        if layer_idx == 0 { rec!(e_attn); }

        // O proj + residual
        gemv(stream, lw.o_proj.ptr, buffers.attn_out, buffers.o_proj_out, hs_z, nh*hd);
        launch_residual_add_bf16(h_in as *const _, buffers.o_proj_out as *const _, buffers.residual as *mut _, (bs * hs) as i32, stream);
        if layer_idx == 0 { rec!(e_oproj); }

        // MLP
        launch_fused_rmsnorm_residual_bf16(buffers.residual as *const _, std::ptr::null(), lw.post_attn_layernorm as *const _, buffers.normed as *mut _, std::ptr::null_mut(), hs as i32, bs as i32, eps, stream);
        gemv(stream, lw.gate_proj.ptr, buffers.normed, buffers.gate, inter_z, hs_z);
        gemv(stream, lw.up_proj.ptr, buffers.normed, buffers.up, inter_z, hs_z);
        launch_fused_silu_mul_bf16(buffers.gate as *const _, buffers.up as *const _, buffers.mlp_act as *mut _, (bs * inter) as i32, stream);
        gemv(stream, lw.down_proj.ptr, buffers.mlp_act, buffers.down_out, hs_z, inter_z);
        launch_residual_add_bf16(buffers.residual as *const _, buffers.down_out as *const _, h_out as *mut _, (bs * hs) as i32, stream);
        if layer_idx == 0 { rec!(e_mlp); rec!(e_layer_end); }

        std::mem::swap(&mut h_in, &mut h_out);
    }

    rec!(e_final);
    launch_fused_rmsnorm_residual_bf16(h_in as *const _, std::ptr::null(), weights.final_norm as *const _, buffers.normed as *mut _, std::ptr::null_mut(), hs as i32, bs as i32, eps, stream);
    arc_launch_gemv_bf16_f32out(weights.lm_head.ptr as *const _, buffers.normed as *const _, buffers.logits_f32 as *mut _, cfg.vocab_size as i32, hs_z as i32, stream);
    rec!(e_lmhead); rec!(e_end);

    let total = ms!(e_start, e_end);
    let embed = ms!(e_start, e_embed);
    let qkv_l0 = ms!(e_layer0, e_qkv);
    let attn_l0 = ms!(e_qkv, e_attn);
    let oproj_l0 = ms!(e_attn, e_oproj);
    let mlp_l0 = ms!(e_oproj, e_mlp);
    let layer0 = ms!(e_layer0, e_layer_end);
    let final_norm_lm = ms!(e_final, e_lmhead);

    tracing::info!(
        "PROFILE: total={total:.2}ms | embed={embed:.3}ms | layer0={layer0:.2}ms (qkv={qkv_l0:.2} attn={attn_l0:.2} oproj={oproj_l0:.2} mlp={mlp_l0:.2}) | est_all_layers={:.1}ms | final+lm_head={final_norm_lm:.2}ms",
        layer0 * cfg.num_layers as f32,
    );

    for e in [e_start, e_embed, e_layer0, e_qkv, e_attn, e_oproj, e_mlp, e_layer_end, e_final, e_lmhead, e_end] {
        cudaEventDestroy(e);
    }
}

/// Run the full decode forward pass for one step.
/// Pure kernel launches — no cuBLAS, no allocations, graph-capturable.
#[cfg(feature = "cuda")]
pub unsafe fn decode_forward(
    weights: &ModelWeights,
    buffers: &DecodeBuffers,
    paged_attn: &PagedAttentionState,
    stream: CUstream,
) {
    let cfg = &weights.config;
    let bs = buffers.batch_size as u64;
    let hs = cfg.hidden_size as u64;
    let hs_z = cfg.hidden_size;
    let inter_z = cfg.intermediate_size;
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

        // Fused: RMSNorm + QKV GEMV in one kernel (eliminates norm launch + L2 round-trip)
        arc_launch_rmsnorm_gemv_bf16(
            h_in as *const _,
            lw.input_layernorm as *const _,
            lw.qkv_fused as *const _,
            buffers.qkv as *mut _,
            lw.qkv_rows as i32, hs_z as i32, eps, stream,
        );
        // Split output via pointer offsets (zero-copy)
        let q_ptr = buffers.qkv;
        let k_ptr = buffers.qkv + (nh * hd * 2) as u64;      // BF16 = 2 bytes
        let v_ptr = k_ptr + (nkv * hd * 2) as u64;

        // Q/K norm (separate) then RoPE — bisecting fused kernel
        if let (Some(qn), Some(kn)) = (lw.q_norm, lw.k_norm) {
            launch_rmsnorm_head_bf16(q_ptr as *const _, qn as *const _, q_ptr as *mut _, hd as i32, (bs as usize * nh) as i32, eps, stream);
            launch_rmsnorm_head_bf16(k_ptr as *const _, kn as *const _, k_ptr as *mut _, hd as i32, (bs as usize * nkv) as i32, eps, stream);
        }
        launch_gather_rope_decode_bf16(q_ptr as *mut _, k_ptr as *mut _, buffers.cos_table as *const _, buffers.sin_table as *const _, buffers.positions as *const i32, nh as i32, nkv as i32, hd as i32, hd as i32, (hd/2) as i32, bs as i32, buffers.is_neox as i32, stream);

        // Store K/V into the paged KV cache + run attention
        let kv_stride = (nkv * hd) as i32;
        let q_stride = (nh * hd) as i32;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let lc = &paged_attn.layer_caches[layer_idx];

        if paged_attn.is_turbo {
            // Native BF16 — turbo kernels handle conversion internally (dtype=1)
            turbo_reshape_and_cache(
                k_ptr as *const _, v_ptr as *const _,
                lc.key_cache as *const _, lc.value_cache as *const _,
                lc.k_norms as *const _, lc.v_norms as *const _,
                paged_attn.slot_mappings as *const i64,
                bs as i32, nkv as i32, hd as i32, paged_attn.block_size,
                kv_stride, kv_stride,
                paged_attn.kv_block_stride, paged_attn.kv_head_stride,
                paged_attn.norm_block_stride, paged_attn.norm_head_stride,
                stream, 1, // dtype=BF16
            );
            let q_size = (bs as usize * nh * hd) as i32;
            turbo_paged_attention_v1(
                buffers.attn_out_f32 as *const _, q_ptr as *const _,
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
                stream, 1, // dtype=BF16
            );
            // F32→BF16 cast still needed (turbo attn outputs F32)
            launch_cast_f32_to_bf16(
                buffers.attn_out_f32 as *const _, buffers.attn_out as *mut _, q_size, stream,
            );
        } else {
            // Standard BF16 cache
            reshape_and_cache(
                k_ptr as *const _, v_ptr as *const _,
                lc.key_cache as *const _, lc.value_cache as *const _,
                paged_attn.slot_mappings as *const i64,
                bs as i32, nkv as i32, hd as i32, paged_attn.block_size, paged_attn.x,
                kv_stride, kv_stride,
                stream, 1, 1, // dtype=BF16, cache_dtype=BF16
                std::ptr::null(), std::ptr::null(),
            );
            paged_attention_v1_bf16(
                buffers.attn_out as *const _, q_ptr as *const _,
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
        gemv(stream, lw.o_proj.ptr, buffers.attn_out, buffers.o_proj_out,
            hs_z, nh * hd);

        // Residual add
        launch_residual_add_bf16(
            h_in as *const _, buffers.o_proj_out as *const _,
            buffers.residual as *mut _, (bs * hs) as i32, stream,
        );

        // Post-attention RMSNorm (separate — Up needs the normed buffer, fusion would
        // require recomputing or a dual-gemv kernel)
        launch_fused_rmsnorm_residual_bf16(
            buffers.residual as *const _, std::ptr::null(),
            lw.post_attn_layernorm as *const _,
            buffers.normed as *mut _, std::ptr::null_mut(),
            hs as i32, bs as i32, eps, stream,
        );
        gemv(stream, lw.gate_proj.ptr, buffers.normed, buffers.gate,
            inter_z, hs_z);
        gemv(stream, lw.up_proj.ptr, buffers.normed, buffers.up,
            inter_z, hs_z);

        launch_fused_silu_mul_bf16(
            buffers.gate as *const _, buffers.up as *const _,
            buffers.mlp_act as *mut _, (bs * inter) as i32, stream,
        );

        gemv(stream, lw.down_proj.ptr, buffers.mlp_act, buffers.down_out,
            hs_z, inter_z);

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

    // LM head → outputs F32 directly (sampling needs F32, no cast needed)
    arc_launch_gemv_bf16_f32out(
        weights.lm_head.ptr as *const _, buffers.normed as *const _,
        buffers.logits_f32 as *mut _,
        cfg.vocab_size as i32, hs_z as i32, stream,
    );
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
    fn turbo_paged_attention_v1(
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
        dtype: u32,
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
    fn launch_cast_f32_to_bf16(
        input: *const std::ffi::c_void, output: *mut std::ffi::c_void,
        size: i32, stream: CUstream,
    );
}
