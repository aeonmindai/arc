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
    /// Streaming-multiprocessor count for the active CUDA device. Queried at
    /// init via cudaDeviceGetAttribute(MultiProcessorCount). Used by GEMV
    /// dispatch to pick the wide vs original kernel without hardcoding GPU shape.
    pub sm_count: i32,
}

// Custom GEMV FFI — graph-capture compatible, no cuBLAS
#[cfg(feature = "cuda")]
extern "C" {
    fn arc_launch_gemv_bf16(
        weight: *const std::ffi::c_void,
        input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        m: i32, k: i32, sm_count: i32, stream: CUstream,
    );
    fn arc_launch_gemv_bf16_f32out(
        weight: *const std::ffi::c_void,
        input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        m: i32, k: i32, stream: CUstream,
    );
    fn arc_launch_gemv_bf16_silu_mul_down(
        weight: *const std::ffi::c_void,
        gate: *const std::ffi::c_void,
        up: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        m: i32, k: i32, stream: CUstream,
    );
    fn arc_launch_gemv_bf16_dual(
        weight_a: *const std::ffi::c_void, weight_b: *const std::ffi::c_void,
        input: *const std::ffi::c_void,
        out_a: *mut std::ffi::c_void, out_b: *mut std::ffi::c_void,
        m_a: i32, m_b: i32, k: i32, stream: CUstream,
    );
}

/// BF16 GEMV: output[0..m] = weight[m,k] * input[0..k]. Graph-capturable.
#[cfg(feature = "cuda")]
#[inline(always)]
unsafe fn gemv(stream: CUstream, sm_count: i32, weight: u64, input: u64, output: u64, m: usize, k: usize) {
    arc_launch_gemv_bf16(weight as *const _, input as *const _, output as *mut _, m as i32, k as i32, sm_count, stream);
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

    // Granular per-kernel events for layer 0. 17 events bracketing every kernel.
    let e_start = ev!(); let e_embed = ev!();
    let e_l0 = ev!();
    let e_pre_norm = ev!(); let e_qkv_gemv = ev!(); let e_qknorm = ev!(); let e_rope = ev!();
    let e_kv_cache = ev!(); let e_attn = ev!(); let e_attn_cast = ev!();
    let e_oproj = ev!(); let e_res1 = ev!();
    let e_post_norm = ev!(); let e_gate = ev!(); let e_up = ev!(); let e_silu = ev!(); let e_down = ev!(); let e_res2 = ev!();
    let e_l0_end = ev!();
    let e_layers_end = ev!();
    let e_final_norm = ev!(); let e_lm_head = ev!(); let e_end = ev!();

    rec!(e_start);
    launch_gather_embedding_bf16(weights.embed_tokens as *const _, buffers.token_ids as *const i32, buffers.hidden_a as *mut _, hs as i32, bs as i32, stream);
    rec!(e_embed);

    let mut h_in = buffers.hidden_a; let mut h_out = buffers.hidden_b;
    for layer_idx in 0..cfg.num_layers {
        let lw = &weights.layers[layer_idx];
        let prof = layer_idx == 0;
        if prof { rec!(e_l0); }

        // === pre-attn norm ===
        launch_fused_rmsnorm_residual_bf16(h_in as *const _, std::ptr::null(), lw.input_layernorm as *const _, buffers.normed as *mut _, std::ptr::null_mut(), hs as i32, bs as i32, eps, stream);
        if prof { rec!(e_pre_norm); }

        // === qkv gemv ===
        gemv(stream, buffers.sm_count, lw.qkv_fused, buffers.normed, buffers.qkv, lw.qkv_rows, hs_z);
        if prof { rec!(e_qkv_gemv); }

        let q_ptr = buffers.qkv;
        let k_ptr = buffers.qkv + (nh * hd * 2) as u64;
        let v_ptr = k_ptr + (nkv * hd * 2) as u64;

        // === q/k norm ===
        if let (Some(qn), Some(kn)) = (lw.q_norm, lw.k_norm) {
            launch_rmsnorm_head_bf16(q_ptr as *const _, qn as *const _, q_ptr as *mut _, hd as i32, (bs as usize * nh) as i32, eps, stream);
            launch_rmsnorm_head_bf16(k_ptr as *const _, kn as *const _, k_ptr as *mut _, hd as i32, (bs as usize * nkv) as i32, eps, stream);
        }
        if prof { rec!(e_qknorm); }

        // === rope ===
        launch_gather_rope_decode_bf16(q_ptr as *mut _, k_ptr as *mut _, buffers.cos_table as *const _, buffers.sin_table as *const _, buffers.positions as *const i32, nh as i32, nkv as i32, hd as i32, hd as i32, (hd/2) as i32, bs as i32, buffers.is_neox as i32, stream);
        if prof { rec!(e_rope); }

        // === kv cache write + attention + cast ===
        let kv_stride = (nkv * hd) as i32; let q_stride = (nh * hd) as i32; let scale = 1.0f32 / (hd as f32).sqrt();
        let lc = &paged_attn.layer_caches[layer_idx];
        if paged_attn.is_turbo {
            turbo_reshape_and_cache(k_ptr as *const _, v_ptr as *const _, lc.key_cache as *const _, lc.value_cache as *const _, lc.k_norms as *const _, lc.v_norms as *const _, paged_attn.slot_mappings as *const i64, bs as i32, nkv as i32, hd as i32, paged_attn.block_size, kv_stride, kv_stride, paged_attn.kv_block_stride, paged_attn.kv_head_stride, paged_attn.norm_block_stride, paged_attn.norm_head_stride, stream, 1);
            if prof { rec!(e_kv_cache); }
            turbo_paged_attention_v1(buffers.attn_out_f32 as *const _, q_ptr as *const _, lc.key_cache as *const _, lc.value_cache as *const _, lc.k_norms as *const _, lc.v_norms as *const _, nkv as i32, scale, 1.0, paged_attn.block_tables as *const u32, paged_attn.context_lens as *const u32, paged_attn.block_size, paged_attn.max_context_len, bs as i32, nh as i32, hd as i32, paged_attn.max_num_blocks_per_seq, q_stride, paged_attn.kv_block_stride, paged_attn.kv_head_stride, paged_attn.norm_block_stride, paged_attn.norm_head_stride, stream, 1);
            if prof { rec!(e_attn); }
            launch_cast_f32_to_bf16(buffers.attn_out_f32 as *const _, buffers.attn_out as *mut _, (bs as usize * nh * hd) as i32, stream);
            if prof { rec!(e_attn_cast); }
        } else {
            reshape_and_cache(k_ptr as *const _, v_ptr as *const _, lc.key_cache as *const _, lc.value_cache as *const _, paged_attn.slot_mappings as *const i64, bs as i32, nkv as i32, hd as i32, paged_attn.block_size, paged_attn.x, kv_stride, kv_stride, stream, 1, 1, std::ptr::null(), std::ptr::null());
            if prof { rec!(e_kv_cache); }
            paged_attention_v1_bf16(buffers.attn_out as *const _, q_ptr as *const _, lc.key_cache as *const _, lc.value_cache as *const _, std::ptr::null(), nkv as i32, scale, 1.0, paged_attn.block_tables as *const i32, paged_attn.context_lens as *const i32, paged_attn.block_size, paged_attn.max_context_len, bs as i32, nh as i32, hd as i32, paged_attn.max_num_blocks_per_seq, q_stride, paged_attn.kv_block_stride, paged_attn.kv_head_stride, stream, 1, std::ptr::null(), std::ptr::null(), std::ptr::null());
            if prof { rec!(e_attn); rec!(e_attn_cast); }
        }

        // === o proj ===
        gemv(stream, buffers.sm_count, lw.o_proj.ptr, buffers.attn_out, buffers.o_proj_out, hs_z, nh*hd);
        if prof { rec!(e_oproj); }

        // === residual1 ===
        launch_residual_add_bf16(h_in as *const _, buffers.o_proj_out as *const _, buffers.residual as *mut _, (bs * hs) as i32, stream);
        if prof { rec!(e_res1); }

        // === post-attn norm ===
        launch_fused_rmsnorm_residual_bf16(buffers.residual as *const _, std::ptr::null(), lw.post_attn_layernorm as *const _, buffers.normed as *mut _, std::ptr::null_mut(), hs as i32, bs as i32, eps, stream);
        if prof { rec!(e_post_norm); }

        // === gate / up / silu_mul / down ===
        gemv(stream, buffers.sm_count, lw.gate_proj.ptr, buffers.normed, buffers.gate, inter_z, hs_z);
        if prof { rec!(e_gate); }
        gemv(stream, buffers.sm_count, lw.up_proj.ptr, buffers.normed, buffers.up, inter_z, hs_z);
        if prof { rec!(e_up); }
        launch_fused_silu_mul_bf16(buffers.gate as *const _, buffers.up as *const _, buffers.mlp_act as *mut _, (bs * inter) as i32, stream);
        if prof { rec!(e_silu); }
        gemv(stream, buffers.sm_count, lw.down_proj.ptr, buffers.mlp_act, buffers.down_out, hs_z, inter_z);
        if prof { rec!(e_down); }

        // === residual2 ===
        launch_residual_add_bf16(buffers.residual as *const _, buffers.down_out as *const _, h_out as *mut _, (bs * hs) as i32, stream);
        if prof { rec!(e_res2); rec!(e_l0_end); }

        std::mem::swap(&mut h_in, &mut h_out);
    }
    rec!(e_layers_end);

    launch_fused_rmsnorm_residual_bf16(h_in as *const _, std::ptr::null(), weights.final_norm as *const _, buffers.normed as *mut _, std::ptr::null_mut(), hs as i32, bs as i32, eps, stream);
    rec!(e_final_norm);
    arc_launch_gemv_bf16_f32out(weights.lm_head.ptr as *const _, buffers.normed as *const _, buffers.logits_f32 as *mut _, cfg.vocab_size as i32, hs_z as i32, stream);
    rec!(e_lm_head);
    rec!(e_end);

    let total = ms!(e_start, e_end);
    let embed = ms!(e_start, e_embed);
    let layer0 = ms!(e_l0, e_l0_end);
    let pre_norm = ms!(e_l0, e_pre_norm);
    let qkv_gemv = ms!(e_pre_norm, e_qkv_gemv);
    let qknorm = ms!(e_qkv_gemv, e_qknorm);
    let rope = ms!(e_qknorm, e_rope);
    let kv_cache = ms!(e_rope, e_kv_cache);
    let attn = ms!(e_kv_cache, e_attn);
    let attn_cast = ms!(e_attn, e_attn_cast);
    let oproj = ms!(e_attn_cast, e_oproj);
    let res1 = ms!(e_oproj, e_res1);
    let post_norm = ms!(e_res1, e_post_norm);
    let gate = ms!(e_post_norm, e_gate);
    let up = ms!(e_gate, e_up);
    let silu = ms!(e_up, e_silu);
    let down = ms!(e_silu, e_down);
    let res2 = ms!(e_down, e_res2);
    let layers_total = ms!(e_embed, e_layers_end);
    let final_norm = ms!(e_layers_end, e_final_norm);
    let lm_head = ms!(e_final_norm, e_lm_head);

    let nl = cfg.num_layers as f32;
    let sum_l0 = pre_norm + qkv_gemv + qknorm + rope + kv_cache + attn + attn_cast + oproj + res1 + post_norm + gate + up + silu + down + res2;
    let layers_avg = layers_total / nl;
    let gap = layer0 - sum_l0;

    tracing::info!(
        "PROFILE total={total:.2}ms embed={embed:.3} | layer0={layer0:.3}ms (sum={sum_l0:.3} gap={gap:.3}) | layers_total={layers_total:.2} layers_avg={layers_avg:.3} | final_norm={final_norm:.3} lm_head={lm_head:.3}"
    );
    tracing::info!(
        "PROFILE_L0_us pre_norm={:.0} qkv_gemv={:.0} qknorm={:.0} rope={:.0} kv_cache={:.0} attn={:.0} attn_cast={:.0} oproj={:.0} res1={:.0} post_norm={:.0} gate={:.0} up={:.0} silu={:.0} down={:.0} res2={:.0}",
        pre_norm*1000.0, qkv_gemv*1000.0, qknorm*1000.0, rope*1000.0, kv_cache*1000.0, attn*1000.0, attn_cast*1000.0, oproj*1000.0, res1*1000.0, post_norm*1000.0, gate*1000.0, up*1000.0, silu*1000.0, down*1000.0, res2*1000.0
    );
    tracing::info!(
        "PROFILE_EXTRAP est_decode_per_token={:.2}ms tok_per_s={:.1}",
        embed + layers_avg * nl + final_norm + lm_head,
        1000.0 / (embed + layers_avg * nl + final_norm + lm_head)
    );

    for e in [e_start, e_embed, e_l0, e_pre_norm, e_qkv_gemv, e_qknorm, e_rope, e_kv_cache, e_attn, e_attn_cast, e_oproj, e_res1, e_post_norm, e_gate, e_up, e_silu, e_down, e_res2, e_l0_end, e_layers_end, e_final_norm, e_lm_head, e_end] {
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
    let mut deferred_residual: u64 = 0; // 0 = no deferred add, else = pointer to add

    for layer_idx in 0..cfg.num_layers {
        let lw = &weights.layers[layer_idx];

        // RMSNorm (input_layernorm) — fuses deferred residual from previous layer if any
        let residual_ptr = if deferred_residual != 0 { deferred_residual as *const _ } else { std::ptr::null() };
        let residual_out_ptr = if deferred_residual != 0 { buffers.residual as *mut _ } else { std::ptr::null_mut() };
        launch_fused_rmsnorm_residual_bf16(
            h_in as *const _, residual_ptr, lw.input_layernorm as *const _,
            buffers.normed as *mut _, residual_out_ptr,
            hs as i32, bs as i32, eps, stream,
        );
        // After fusion: residual buffer holds (h_in + deferred_residual). Use it as h_in for residual chain.
        if deferred_residual != 0 {
            h_in = buffers.residual;
        }

        // Fused QKV GEMV: 1 launch instead of 3, better bandwidth for small K/V
        gemv(stream, buffers.sm_count, lw.qkv_fused, buffers.normed, buffers.qkv,
            lw.qkv_rows, hs_z);
        // Split output via pointer offsets (zero-copy)
        let q_ptr = buffers.qkv;
        let k_ptr = buffers.qkv + (nh * hd * 2) as u64;      // BF16 = 2 bytes
        let v_ptr = k_ptr + (nkv * hd * 2) as u64;

        // QK norm (if model uses it) — fused into a single launch
        if let (Some(qn), Some(kn)) = (lw.q_norm, lw.k_norm) {
            launch_rmsnorm_qk_pair_bf16(
                q_ptr as *const _, qn as *const _, q_ptr as *mut _,
                k_ptr as *const _, kn as *const _, k_ptr as *mut _,
                hd as i32,
                (bs as usize * nh) as i32,
                (bs as usize * nkv) as i32,
                eps, stream,
            );
        }

        // RoPE
        launch_gather_rope_decode_bf16(
            q_ptr as *mut _, k_ptr as *mut _,
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
            // BF16-output variant: writes BF16 directly into buffers.attn_out,
            // eliminating the separate F32→BF16 cast kernel.
            turbo_paged_attention_v1_bf16out(
                buffers.attn_out as *const _, q_ptr as *const _,
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
                stream, 1, // qdtype=BF16
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
        gemv(stream, buffers.sm_count, lw.o_proj.ptr, buffers.attn_out, buffers.o_proj_out,
            hs_z, nh * hd);

        // Fused: residual = h_in + o_proj_out, normed = rmsnorm(residual)
        // Eliminates separate residual_add kernel (saves 1 launch per layer)
        launch_fused_rmsnorm_residual_bf16(
            h_in as *const _, buffers.o_proj_out as *const _,
            lw.post_attn_layernorm as *const _,
            buffers.normed as *mut _, buffers.residual as *mut _,
            hs as i32, bs as i32, eps, stream,
        );

        // MLP: dual-weight GEMV computes gate and up in one launch (no weight
        // duplication — reads from both gate_proj and up_proj buffers directly),
        // then silu_mul + down.
        arc_launch_gemv_bf16_dual(
            lw.gate_proj.ptr as *const _, lw.up_proj.ptr as *const _,
            buffers.normed as *const _,
            buffers.gate as *mut _, buffers.up as *mut _,
            inter_z as i32, inter_z as i32, hs_z as i32, stream,
        );
        launch_fused_silu_mul_bf16(
            buffers.gate as *const _, buffers.up as *const _,
            buffers.mlp_act as *mut _, (bs * inter) as i32, stream,
        );
        gemv(stream, buffers.sm_count, lw.down_proj.ptr, buffers.mlp_act, buffers.down_out,
            hs_z, inter_z);

        // Defer residual add: next layer's input RMSNorm fuses (residual + down_out)
        h_in = buffers.residual; // current residual (post-attention sum)
        deferred_residual = buffers.down_out; // to be added by next norm
    }

    // Final fused: normed = rmsnorm(h_in + down_out)
    launch_fused_rmsnorm_residual_bf16(
        h_in as *const _, deferred_residual as *const _,
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
    fn turbo_paged_attention_v1_bf16out(
        out_bf16: *const std::ffi::c_void,
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
        qdtype: u32,
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
    fn launch_rmsnorm_qk_pair_bf16(
        q_in: *const std::ffi::c_void, q_w: *const std::ffi::c_void, q_out: *mut std::ffi::c_void,
        k_in: *const std::ffi::c_void, k_w: *const std::ffi::c_void, k_out: *mut std::ffi::c_void,
        head_dim: i32, n_q_heads: i32, n_k_heads: i32, eps: f32,
        stream: CUstream,
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
