//! Dedicated decode path manager.
//!
//! Creates a non-blocking stream, allocates buffers, initializes cuBLASLt,
//! and runs decode_forward() with CUDA graph capture + replay.
//!
//! Flow:
//!   Step 0-1 (warmup): Candle handles decode, dedicated path warms up
//!   Step 2 (capture):  Copy inputs → begin capture → decode_forward → end capture → instantiate → launch
//!   Step 3+ (replay):  Copy inputs → cuGraphLaunch (all compute replayed from fixed addresses)

#[cfg(feature = "cuda")]
use crate::decode_forward::{CublasState, DecodeBuffers, LayerKvCache, PagedAttentionState, decode_forward};
#[cfg(feature = "cuda")]
use crate::ffi::*;
#[cfg(feature = "cuda")]
use crate::weights::ModelWeights;
#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::sys::CUstream;

#[cfg(feature = "cuda")]
extern "C" {
    fn cublasCreate_v2(handle: *mut *mut std::ffi::c_void) -> u32;
    fn cublasDestroy_v2(handle: *mut std::ffi::c_void) -> u32;
    fn cudaMalloc(ptr: *mut u64, size: usize) -> u32;
    fn cudaFree(ptr: u64) -> u32;
    fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
        count: usize, kind: u32, stream: CUstream,
    ) -> u32;
}

/// The dedicated decode path: owns its own non-blocking stream, buffers, and cuBLAS.
/// After warmup, captures the forward pass into a CUDA graph for replay.
#[cfg(feature = "cuda")]
pub struct DedicatedDecodePath {
    pub weights: ModelWeights,
    stream: CUstream,
    cublas: CublasState,
    buffers: Option<DecodeBuffers>,
    cos_table: u64,
    sin_table: u64,

    // CUDA graph state
    graph_exec: Option<CUgraphExec>,
    captured_batch_size: usize,

    // Staging buffers for paged attention metadata (fixed GPU addresses for graph replay)
    staging_block_tables: u64,      // [max_batch, max_blocks_per_seq] i32
    staging_context_lens: u64,      // [max_batch] i32
    staging_slot_mappings: u64,     // [max_batch] i64
    staging_max_blocks_per_seq: usize,
    staging_max_batch: usize,

    // Cached KV cache pointers (stable across steps — CacheEngine allocates once)
    cached_layer_caches: Option<Vec<LayerKvCache>>,
    // Cached scalar paged attention params (stable for a given batch size)
    cached_block_size: i32,
    cached_kv_block_stride: i32,
    cached_kv_head_stride: i32,
    cached_norm_block_stride: i32,
    cached_norm_head_stride: i32,
    cached_x: i32,
    cached_is_turbo: bool,

    enabled: bool,
    warmup_remaining: u32,
    eager_steps: u32,
    capture_failed: bool, // Stop retrying capture after failure
}

#[cfg(feature = "cuda")]
unsafe impl Send for DedicatedDecodePath {}
#[cfg(feature = "cuda")]
unsafe impl Sync for DedicatedDecodePath {}

#[cfg(feature = "cuda")]
impl DedicatedDecodePath {
    pub fn new(weights: ModelWeights) -> candle_core::Result<Self> {
        let mut stream: CUstream = std::ptr::null_mut();
        let s = unsafe { cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING) };
        if s != CUDA_SUCCESS {
            candle_core::bail!("Failed to create decode stream: {s}");
        }

        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let s = unsafe { cublasCreate_v2(&mut handle) };
        if s != 0 {
            unsafe { cuStreamDestroy_v2(stream); }
            candle_core::bail!("cublasCreate failed: {s}");
        }

        // cublasGemmEx manages its own workspace internally
        let workspace_size = 0usize;
        let workspace_ptr: u64 = 0;

        let (cos_table, sin_table) = Self::compute_rope_tables(&weights.config)?;

        tracing::info!(
            "Dedicated decode path initialized (stream + cuBLAS + RoPE[{}x{}])",
            weights.config.max_position_embeddings,
            weights.config.head_dim / 2,
        );

        Ok(Self {
            weights,
            stream,
            cublas: CublasState { handle, workspace: workspace_ptr, workspace_size },
            buffers: None,
            cos_table,
            sin_table,
            graph_exec: None,
            captured_batch_size: 0,
            staging_block_tables: 0,
            staging_context_lens: 0,
            staging_slot_mappings: 0,
            staging_max_blocks_per_seq: 0,
            staging_max_batch: 0,
            cached_layer_caches: None,
            cached_block_size: 0,
            cached_kv_block_stride: 0,
            cached_kv_head_stride: 0,
            cached_norm_block_stride: 0,
            cached_norm_head_stride: 0,
            cached_x: 0,
            cached_is_turbo: false,
            enabled: true,
            warmup_remaining: 2,
            eager_steps: 0,
            capture_failed: false,
        })
    }

    fn compute_rope_tables(cfg: &crate::weights::DecodeConfig) -> candle_core::Result<(u64, u64)> {
        let head_dim = cfg.head_dim;
        let max_pos = cfg.max_position_embeddings;
        let theta = cfg.rope_theta;
        let rot_dim = head_dim / 2;

        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0f32 / theta.powf(i as f32 / head_dim as f32))
            .collect();

        let table_len = max_pos * rot_dim;
        let mut cos_bf16 = vec![0u16; table_len];
        let mut sin_bf16 = vec![0u16; table_len];

        for pos in 0..max_pos {
            for (j, &freq) in inv_freq.iter().enumerate() {
                let angle = pos as f32 * freq;
                cos_bf16[pos * rot_dim + j] = (angle.cos().to_bits() >> 16) as u16;
                sin_bf16[pos * rot_dim + j] = (angle.sin().to_bits() >> 16) as u16;
            }
        }

        let table_bytes = table_len * 2;
        let mut cos_ptr: u64 = 0;
        let mut sin_ptr: u64 = 0;

        unsafe {
            let s = cudaMalloc(&mut cos_ptr, table_bytes);
            if s != 0 { candle_core::bail!("cudaMalloc cos_table failed: {s}"); }
            let s = cudaMalloc(&mut sin_ptr, table_bytes);
            if s != 0 { candle_core::bail!("cudaMalloc sin_table failed: {s}"); }

            extern "C" {
                fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: u32) -> u32;
            }
            let s = cudaMemcpy(cos_ptr as *mut _, cos_bf16.as_ptr() as *const _, table_bytes, 1);
            if s != 0 { candle_core::bail!("cudaMemcpy cos_table failed: {s}"); }
            let s = cudaMemcpy(sin_ptr as *mut _, sin_bf16.as_ptr() as *const _, table_bytes, 1);
            if s != 0 { candle_core::bail!("cudaMemcpy sin_table failed: {s}"); }
        }

        Ok((cos_ptr, sin_ptr))
    }

    fn ensure_buffers(&mut self, batch_size: usize) -> candle_core::Result<()> {
        if self.buffers.is_some() {
            return Ok(());
        }

        let cfg = &self.weights.config;
        let bf16 = 2usize;
        let bs = batch_size;

        macro_rules! alloc {
            ($size:expr) => {{
                let mut ptr: u64 = 0;
                let s = unsafe { cudaMalloc(&mut ptr, $size) };
                if s != 0 { candle_core::bail!("cudaMalloc failed for decode buffer: {s}"); }
                ptr
            }};
        }

        let buffers = DecodeBuffers {
            hidden_a: alloc!(bs * cfg.hidden_size * bf16),
            hidden_b: alloc!(bs * cfg.hidden_size * bf16),
            normed: alloc!(bs * cfg.hidden_size * bf16),
            residual: alloc!(bs * cfg.hidden_size * bf16),
            q: alloc!(bs * cfg.num_heads * cfg.head_dim * bf16),
            k: alloc!(bs * cfg.num_kv_heads * cfg.head_dim * bf16),
            v: alloc!(bs * cfg.num_kv_heads * cfg.head_dim * bf16),
            attn_out: alloc!(bs * cfg.num_heads * cfg.head_dim * bf16),
            q_f16: alloc!(bs * cfg.num_heads * cfg.head_dim * bf16),
            k_f16: alloc!(bs * cfg.num_kv_heads * cfg.head_dim * bf16),
            v_f16: alloc!(bs * cfg.num_kv_heads * cfg.head_dim * bf16),
            attn_out_f32: alloc!(bs * cfg.num_heads * cfg.head_dim * 4),
            o_proj_out: alloc!(bs * cfg.hidden_size * bf16),
            gate: alloc!(bs * cfg.intermediate_size * bf16),
            up: alloc!(bs * cfg.intermediate_size * bf16),
            mlp_act: alloc!(bs * cfg.intermediate_size * bf16),
            down_out: alloc!(bs * cfg.hidden_size * bf16),
            logits: alloc!(bs * cfg.vocab_size * bf16),
            token_ids: alloc!(bs * 4),
            positions: alloc!(bs * 4),
            cos_table: self.cos_table,
            sin_table: self.sin_table,
            is_neox: self.weights.config.is_gpt_neox,
            batch_size: bs,
        };

        let total_mb = (bs * cfg.hidden_size * bf16 * 4
            + bs * (cfg.num_heads + cfg.num_kv_heads * 2) * cfg.head_dim * bf16
            + bs * cfg.num_heads * cfg.head_dim * bf16
            + bs * cfg.hidden_size * bf16
            + bs * cfg.intermediate_size * bf16 * 3
            + bs * cfg.hidden_size * bf16
            + bs * cfg.vocab_size * bf16
        ) / 1_048_576;

        tracing::info!("Decode buffers allocated: ~{total_mb} MB for batch_size={bs}");
        self.buffers = Some(buffers);
        Ok(())
    }

    /// Allocate staging buffers for paged attention metadata.
    /// These are at fixed GPU addresses so the captured graph can replay.
    fn ensure_staging(&mut self, batch_size: usize, max_blocks_per_seq: usize) -> candle_core::Result<()> {
        if self.staging_max_batch >= batch_size && self.staging_max_blocks_per_seq >= max_blocks_per_seq {
            return Ok(());
        }
        // Free old staging if resizing — also invalidates captured graph
        if let Some(exec) = self.graph_exec.take() {
            unsafe { cuGraphExecDestroy(exec); }
            tracing::info!("Staging resize → graph invalidated");
        }
        unsafe {
            if self.staging_block_tables != 0 { cudaFree(self.staging_block_tables); }
            if self.staging_context_lens != 0 { cudaFree(self.staging_context_lens); }
            if self.staging_slot_mappings != 0 { cudaFree(self.staging_slot_mappings); }
        }
        let mut bt: u64 = 0;
        let mut cl: u64 = 0;
        let mut sm: u64 = 0;
        unsafe {
            let s = cudaMalloc(&mut bt, batch_size * max_blocks_per_seq * 4); // i32
            if s != 0 { candle_core::bail!("cudaMalloc staging_block_tables failed: {s}"); }
            let s = cudaMalloc(&mut cl, batch_size * 4); // i32
            if s != 0 { candle_core::bail!("cudaMalloc staging_context_lens failed: {s}"); }
            let s = cudaMalloc(&mut sm, batch_size * 8); // i64
            if s != 0 { candle_core::bail!("cudaMalloc staging_slot_mappings failed: {s}"); }
        }
        self.staging_block_tables = bt;
        self.staging_context_lens = cl;
        self.staging_slot_mappings = sm;
        self.staging_max_batch = batch_size;
        self.staging_max_blocks_per_seq = max_blocks_per_seq;
        tracing::info!("Staging buffers allocated: batch={batch_size}, max_blocks={max_blocks_per_seq}");
        Ok(())
    }

    /// Copy per-step paged attention data from caller's pointers to fixed staging buffers.
    /// D2D async on the dedicated stream — ordered before graph launch.
    /// Uses the ACTUAL per-step sizes, not the staging capacity.
    unsafe fn stage_paged_attn(&self, paged_attn: &PagedAttentionState, batch_size: usize) {
        let actual_blocks = paged_attn.max_num_blocks_per_seq as usize;
        // block_tables: [batch, actual_blocks] i32/u32
        cudaMemcpyAsync(
            self.staging_block_tables as *mut _, paged_attn.block_tables as *const _,
            batch_size * actual_blocks * 4, 3, // D2D
            self.stream,
        );
        // context_lens: [batch] i32/u32
        cudaMemcpyAsync(
            self.staging_context_lens as *mut _, paged_attn.context_lens as *const _,
            batch_size * 4, 3,
            self.stream,
        );
        // slot_mappings: [batch] i64
        cudaMemcpyAsync(
            self.staging_slot_mappings as *mut _, paged_attn.slot_mappings as *const _,
            batch_size * 8, 3,
            self.stream,
        );
    }

    /// Build a PagedAttentionState pointing to fixed staging buffers + cached KV caches.
    /// Uses per-step values for max_context_len and max_num_blocks_per_seq.
    fn staged_paged_attn(&self, paged_attn: &PagedAttentionState) -> PagedAttentionState {
        PagedAttentionState {
            layer_caches: self.cached_layer_caches.clone().unwrap_or_default(),
            block_tables: self.staging_block_tables,
            context_lens: self.staging_context_lens,
            slot_mappings: self.staging_slot_mappings,
            block_size: self.cached_block_size,
            max_context_len: paged_attn.max_context_len, // REAL per-step value
            max_num_blocks_per_seq: paged_attn.max_num_blocks_per_seq, // REAL per-step value
            kv_block_stride: self.cached_kv_block_stride,
            kv_head_stride: self.cached_kv_head_stride,
            norm_block_stride: self.cached_norm_block_stride,
            norm_head_stride: self.cached_norm_head_stride,
            x: self.cached_x,
            is_turbo: self.cached_is_turbo,
        }
    }

    /// Cache KV cache layer pointers and shape info from the first PagedAttentionState.
    /// These are stable across steps (CacheEngine allocates once).
    pub fn cache_kv_info(&mut self, paged_attn: &PagedAttentionState) {
        if self.cached_layer_caches.is_none() {
            self.cached_layer_caches = Some(paged_attn.layer_caches.clone());
            self.cached_block_size = paged_attn.block_size;
            self.cached_kv_block_stride = paged_attn.kv_block_stride;
            self.cached_kv_head_stride = paged_attn.kv_head_stride;
            self.cached_norm_block_stride = paged_attn.norm_block_stride;
            self.cached_norm_head_stride = paged_attn.norm_head_stride;
            self.cached_x = paged_attn.x;
            self.cached_is_turbo = paged_attn.is_turbo;
        }
    }

    /// Run one decode step with graph capture/replay.
    ///
    /// First few calls: run eagerly to verify correctness.
    /// Then: capture the forward pass into a CUDA graph.
    /// Subsequent calls: replay the graph (~10μs total).
    pub fn run_step(
        &mut self,
        token_ids: &[i32],
        positions: &[i32],
        paged_attn: &PagedAttentionState,
    ) -> candle_core::Result<u64> {
        let batch_size = token_ids.len();
        self.ensure_buffers(batch_size)?;
        // Pre-allocate staging for max possible blocks to avoid reallocation
        // (which would invalidate captured graph pointers)
        let max_possible_blocks = (self.weights.config.max_position_embeddings
            / paged_attn.block_size.max(1) as usize).max(paged_attn.max_num_blocks_per_seq as usize);
        self.ensure_staging(batch_size, max_possible_blocks)?;
        self.cache_kv_info(paged_attn);

        let buffers = self.buffers.as_ref().unwrap();

        unsafe {
            // Stage all changing inputs (NOT part of the graph — happens before capture/launch)
            cudaMemcpyAsync(
                buffers.token_ids as *mut _, token_ids.as_ptr() as *const _,
                batch_size * 4, 1, self.stream,
            );
            cudaMemcpyAsync(
                buffers.positions as *mut _, positions.as_ptr() as *const _,
                batch_size * 4, 1, self.stream,
            );
            self.stage_paged_attn(paged_attn, batch_size);

            // Use REAL per-step values for max_context_len and max_num_blocks_per_seq.
            // These determine shared memory allocation in attention kernels — using
            // max_position_embeddings would request 160KB+ smem and silently fail.
            let staged = self.staged_paged_attn(paged_attn);

            if let Some(exec) = self.graph_exec {
                // REPLAY: all inputs staged, just launch
                if batch_size != self.captured_batch_size {
                    tracing::warn!("Batch size changed ({} → {}), running eager", self.captured_batch_size, batch_size);
                    decode_forward(&self.weights, buffers, &self.cublas, &staged, self.stream);
                } else {
                    let s = cuGraphLaunch(exec, self.stream);
                    if s != CUDA_SUCCESS {
                        tracing::warn!("cuGraphLaunch failed ({s}), falling back to eager");
                        decode_forward(&self.weights, buffers, &self.cublas, &staged, self.stream);
                    }
                }
            } else if self.capture_failed {
                // Eager mode — still fast (no Candle overhead, just kernel launch costs)
                decode_forward(&self.weights, buffers, &self.cublas, &staged, self.stream);
            } else if self.eager_steps < 1000000 {
                // TODO: Graph capture disabled — max_context_len changes per step which
                // causes the TurboQuant attention kernel's shared memory allocation to be
                // wrong on replay. Need to either pad max_context_len to a fixed upper
                // bound during capture, or use kernel node parameter updates.
                // For now, eager mode gives us the full speed benefit of bypassing Candle.
                self.eager_steps += 1;
                decode_forward(&self.weights, buffers, &self.cublas, &staged, self.stream);
                if self.eager_steps == 2 {
                    tracing::info!("Dedicated decode: eager warmup done, will attempt capture next step");
                }
            } else {
                // CAPTURE: record the forward pass into a CUDA graph
                tracing::info!("Capturing CUDA graph for batch_size={batch_size}...");

                let s = cuStreamBeginCapture_v2(self.stream, CUstreamCaptureMode::THREAD_LOCAL);
                if s != CUDA_SUCCESS {
                    tracing::warn!("cuStreamBeginCapture failed ({s}), disabling capture");
                    self.capture_failed = true;
                    decode_forward(&self.weights, buffers, &self.cublas, &staged, self.stream);
                } else {
                    decode_forward(&self.weights, buffers, &self.cublas, &staged, self.stream);

                    let mut graph: CUgraph = std::ptr::null_mut();
                    let s = cuStreamEndCapture(self.stream, &mut graph);
                    if s != CUDA_SUCCESS || graph.is_null() {
                        tracing::warn!("cuStreamEndCapture failed ({s}), disabling capture");
                        if !graph.is_null() { cuGraphDestroy(graph); }
                        self.capture_failed = true;
                        decode_forward(&self.weights, buffers, &self.cublas, &staged, self.stream);
                    } else {
                        let mut exec: CUgraphExec = std::ptr::null_mut();
                        let s = cuGraphInstantiate_v2(
                            &mut exec, graph,
                            std::ptr::null_mut(), std::ptr::null_mut(), 0,
                        );
                        cuGraphDestroy(graph);

                        if s != CUDA_SUCCESS || exec.is_null() {
                            tracing::warn!("cuGraphInstantiate failed ({s}), disabling capture");
                            self.capture_failed = true;
                            decode_forward(&self.weights, buffers, &self.cublas, &staged, self.stream);
                        } else {
                            let s = cuGraphLaunch(exec, self.stream);
                            if s != CUDA_SUCCESS {
                                tracing::warn!("First cuGraphLaunch failed ({s}), disabling capture");
                                cuGraphExecDestroy(exec);
                                self.capture_failed = true;
                                decode_forward(&self.weights, buffers, &self.cublas, &staged, self.stream);
                            } else {
                                self.graph_exec = Some(exec);
                                self.captured_batch_size = batch_size;
                                tracing::info!("CUDA graph captured and launched for batch_size={batch_size}");
                            }
                        }
                    }
                }
            }

            cudaStreamSynchronize(self.stream);
        }

        Ok(buffers.logits)
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled && self.warmup_remaining == 0
    }

    pub fn tick_warmup(&mut self) -> bool {
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            if self.warmup_remaining == 0 {
                tracing::info!("Dedicated decode path: warmup done");
            }
            true
        } else {
            false
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for DedicatedDecodePath {
    fn drop(&mut self) {
        unsafe {
            if let Some(exec) = self.graph_exec {
                cuGraphExecDestroy(exec);
            }
            cublasDestroy_v2(self.cublas.handle);
            if self.cublas.workspace != 0 { cudaFree(self.cublas.workspace); }
            if self.cos_table != 0 { cudaFree(self.cos_table); }
            if self.sin_table != 0 { cudaFree(self.sin_table); }
            if self.staging_block_tables != 0 { cudaFree(self.staging_block_tables); }
            if self.staging_context_lens != 0 { cudaFree(self.staging_context_lens); }
            if self.staging_slot_mappings != 0 { cudaFree(self.staging_slot_mappings); }
            if let Some(ref b) = self.buffers {
                for ptr in [b.hidden_a, b.hidden_b, b.normed, b.residual,
                    b.q, b.k, b.v, b.attn_out,
                    b.q_f16, b.k_f16, b.v_f16, b.attn_out_f32,
                    b.o_proj_out,
                    b.gate, b.up, b.mlp_act, b.down_out, b.logits,
                    b.token_ids, b.positions] {
                    if ptr != 0 { cudaFree(ptr); }
                }
            }
            cuStreamDestroy_v2(self.stream);
        }
    }
}
