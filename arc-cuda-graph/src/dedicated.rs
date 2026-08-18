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
use crate::decode_forward::{decode_forward, DecodeBuffers, LayerKvCache, PagedAttentionState};
#[cfg(feature = "cuda")]
use crate::ffi::*;
#[cfg(feature = "cuda")]
use crate::weights::ModelWeights;
#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::sys::CUstream;

#[cfg(feature = "cuda")]
extern "C" {
    fn cudaMalloc(ptr: *mut u64, size: usize) -> u32;
    fn cudaFree(ptr: u64) -> u32;
    fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: u32,
        stream: CUstream,
    ) -> u32;
    /// Pitched 2-D copy. Required wherever the destination row stride differs
    /// from the source row stride — see `stage_paged_attn`.
    /// `cudaError_t cudaMemcpy2DAsync(void*, size_t dpitch, const void*,
    ///  size_t spitch, size_t width, size_t height, cudaMemcpyKind, cudaStream_t)`
    fn cudaMemcpy2DAsync(
        dst: *mut std::ffi::c_void,
        dpitch: usize,
        src: *const std::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: u32,
        stream: CUstream,
    ) -> u32;
}

/// The dedicated decode path: owns its own non-blocking stream, buffers, and cuBLAS.
/// After warmup, captures the forward pass into a CUDA graph for replay.
#[cfg(feature = "cuda")]
pub struct DedicatedDecodePath {
    pub weights: ModelWeights,
    stream: CUstream,
    buffers: Option<DecodeBuffers>,
    cos_table: u64,
    sin_table: u64,

    // CUDA graph state
    graph_exec: Option<CUgraphExec>,
    captured_batch_size: usize,

    // Staging buffers for paged attention metadata (fixed GPU addresses for graph replay)
    staging_block_tables: u64,  // [max_batch, max_blocks_per_seq] i32
    staging_context_lens: u64,  // [max_batch] i32
    staging_slot_mappings: u64, // [max_batch] i64
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
    step_count: u64,
    // Rolling sums for periodic granular timing report
    sum_setup_us: f64,
    sum_stage_us: f64,
    sum_launch_us: f64,
    sum_sync_us: f64,
    sum_total_us: f64,
}

#[cfg(feature = "cuda")]
unsafe impl Send for DedicatedDecodePath {}
#[cfg(feature = "cuda")]
unsafe impl Sync for DedicatedDecodePath {}

#[cfg(feature = "cuda")]
impl DedicatedDecodePath {
    pub fn new(mut weights: ModelWeights) -> candle_core::Result<Self> {
        // Make sure the CUDA primary context is bound to *this* thread before
        // we touch raw Driver/Runtime APIs (cuStreamCreate, cudaMalloc, ...).
        //
        // candle's cudarc wrapper binds the context per-thread inside every
        // safe call. The pipeline-construction thread that gets us here has
        // done plenty of CUDA work already, so the context is normally bound.
        // We still bind defensively here because:
        //
        //   1. Pre-bind paranoia: any reordering of construction (e.g. moving
        //      pipeline init off the main thread) would otherwise produce a
        //      SEGV in `libcuda` MOVAPS on the first cudaMalloc — no error
        //      code, just a fault, because the runtime lazy-init walks a
        //      thread-local that hasn't been seeded.
        //
        //   2. Validating that we have a real device. If the model is
        //      somehow on CPU, the anchors will be empty or non-CUDA — we
        //      bail with a clear message rather than corrupting state.
        Self::bind_context(&weights)?;

        let mut stream: CUstream = std::ptr::null_mut();
        let s = unsafe { cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING) };
        if s != CUDA_SUCCESS {
            candle_core::bail!(
                "cuStreamCreate(NON_BLOCKING) failed: code={s} \
                 (driver-level error — context may not be bound to this thread)"
            );
        }
        if stream.is_null() {
            candle_core::bail!(
                "cuStreamCreate returned CUDA_SUCCESS but stream is NULL — \
                 driver returned a malformed handle, refusing to proceed"
            );
        }

        let (cos_table, sin_table) = Self::compute_rope_tables(&weights.config)?;

        // Fuse QKV weights contiguously per layer (saves 2 kernel launches/layer).
        // Gate+Up are not physically fused — they would duplicate weights in memory
        // (often many GB extra). Instead the dual-weight GEMV kernel reads from
        // both gate_proj and up_proj buffers in a single launch.
        //
        // SAFETY: q_proj/k_proj/v_proj device pointers are anchored by
        // `weights.anchors` (see ModelWeights::anchors). Without those anchors
        // the cudaMemcpy below would read from dangling memory and segfault
        // inside libcuda's SSE memcpy fast path.
        Self::fuse_qkv(&mut weights)?;

        // After QKV fusion the source Q/K/V buffers are no longer read by
        // decode (only `qkv_fused` is). For ISQ-quantized models each of
        // those was a fresh dequantized BF16 copy, so dropping them frees a
        // meaningful chunk of VRAM (hidden * (heads + 2 * kv_heads) *
        // head_dim * 2 bytes per layer). The remaining anchors keep
        // O/gate/up/down/lm_head/residuals alive for the decode loop.
        //
        // SAFETY: we drop *after* `fuse_qkv` has completed and the
        // cudaMemcpy stream has buffered the D2D copies. The destination is
        // `qkv_fused` which owns its own cudaMalloc'd region, so the source
        // buffers can be freed even if the copy is still in flight (CUDA
        // guarantees the runtime tracks free ordering vs. queued work via
        // stream synchronization). We also call cudaDeviceSynchronize() to
        // be paranoid: any later kernel must observe the fused buffer
        // populated, AND no in-flight async copies must outlive the source.
        unsafe {
            extern "C" {
                fn cudaDeviceSynchronize() -> u32;
            }
            let s = cudaDeviceSynchronize();
            if s != 0 {
                candle_core::bail!(
                    "cudaDeviceSynchronize after fuse_qkv failed: {s} — \
                     refusing to drop Q/K/V anchors with uncertain copy ordering"
                );
            }
        }

        let qkv_dropped: usize = weights
            .anchors
            .layers
            .iter_mut()
            .map(|la| {
                let n = la.qkv.len();
                la.qkv.clear();
                n
            })
            .sum();

        tracing::info!(
            "Dedicated decode path initialized (stream + GEMV + RoPE[{}x{}], \
             {} live anchors after dropping {} Q/K/V anchors)",
            weights.config.max_position_embeddings,
            weights.config.head_dim / 2,
            weights.anchors.count(),
            qkv_dropped,
        );

        Ok(Self {
            weights,
            stream,
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
            step_count: 0,
            sum_setup_us: 0.0,
            sum_stage_us: 0.0,
            sum_launch_us: 0.0,
            sum_sync_us: 0.0,
            sum_total_us: 0.0,
        })
    }

    /// Bind the CUDA primary context that owns the model weights to the
    /// current thread. Required before *any* raw cu*/cuda* call we make
    /// from `DedicatedDecodePath::new`, because we do not go through
    /// cudarc's safe wrappers (which bind on every call).
    ///
    /// Returns `Err` if the model isn't on a CUDA device or has no anchors.
    fn bind_context(weights: &ModelWeights) -> candle_core::Result<()> {
        // Find any CUDA-backed Tensor in the anchors to recover the device.
        // Prefer the lm_head anchor (set first); fall back to the first
        // residual or per-layer anchor.
        let anchor = weights
            .anchors
            .lm_head
            .as_ref()
            .or_else(|| weights.anchors.residuals.first())
            .or_else(|| {
                weights.anchors.layers.iter().find_map(|la| {
                    la.qkv
                        .first()
                        .or(la.o.as_ref())
                        .or(la.gate.as_ref())
                        .or(la.up.as_ref())
                        .or(la.down.as_ref())
                })
            })
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "DedicatedDecodePath::new: ModelWeights has no anchors — \
                     weight extraction is broken (dangling-pointer protection \
                     missing)"
                        .to_string(),
                )
            })?;
        let device = anchor.device();
        match device {
            candle_core::Device::Cuda(cuda_dev) => {
                // cudarc's CudaStream::synchronize() calls bind_to_thread()
                // internally before doing any work. It's the cheapest safe
                // way to ensure the context is bound on this thread.
                cuda_dev.cuda_stream().synchronize().map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "Failed to bind CUDA context to dedicated-decode thread: {e}"
                    ))
                })?;
                Ok(())
            }
            other => candle_core::bail!("DedicatedDecodePath requires a CUDA model, got {other:?}"),
        }
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
            if s != 0 {
                candle_core::bail!("cudaMalloc cos_table failed: {s}");
            }
            let s = cudaMalloc(&mut sin_ptr, table_bytes);
            if s != 0 {
                candle_core::bail!("cudaMalloc sin_table failed: {s}");
            }

            extern "C" {
                fn cudaMemcpy(
                    dst: *mut std::ffi::c_void,
                    src: *const std::ffi::c_void,
                    count: usize,
                    kind: u32,
                ) -> u32;
            }
            let s = cudaMemcpy(
                cos_ptr as *mut _,
                cos_bf16.as_ptr() as *const _,
                table_bytes,
                1,
            );
            if s != 0 {
                candle_core::bail!("cudaMemcpy cos_table failed: {s}");
            }
            let s = cudaMemcpy(
                sin_ptr as *mut _,
                sin_bf16.as_ptr() as *const _,
                table_bytes,
                1,
            );
            if s != 0 {
                candle_core::bail!("cudaMemcpy sin_table failed: {s}");
            }
        }

        Ok((cos_ptr, sin_ptr))
    }

    /// Copy Q+K+V weights contiguously per layer for a single fused GEMV.
    ///
    /// PRECONDITION: every `lw.{q,k,v}_proj.ptr` must point to a CUDA buffer
    /// owned by an anchored Tensor in `weights.anchors`. If any pointer is
    /// dangling (e.g. extracted from a temporary `dequantize_w()` Tensor
    /// without holding a clone), the D2D cudaMemcpy below page-faults
    /// inside libcuda — exactly the SEGV observed in driver 570 / CUDA 12.8
    /// before this fix. The dangling-pointer path is *not* reported as a
    /// cudaError — the page-table walk for the source pointer faults inside
    /// the SSE memcpy fast path (`MOVAPS [rsi+0x10]`), killing the process.
    /// Anchors in `weights.anchors` prevent that fault by keeping the
    /// Arc<Storage> alive.
    fn fuse_qkv(weights: &mut ModelWeights) -> candle_core::Result<()> {
        extern "C" {
            fn cudaMemcpy(
                dst: *mut std::ffi::c_void,
                src: *const std::ffi::c_void,
                count: usize,
                kind: u32,
            ) -> u32;
        }
        let bf16 = 2usize;
        let k = weights.config.hidden_size;
        let mut total_bytes = 0usize;

        for (li, lw) in weights.layers.iter_mut().enumerate() {
            let qr = lw.q_proj.rows;
            let kr = lw.k_proj.rows;
            let vr = lw.v_proj.rows;
            let total_rows = qr + kr + vr;
            let total_size = total_rows * k * bf16;

            // Defensive: zero pointers would faulted in cudaMemcpy and
            // produced exactly the SEGV signature we're guarding against.
            if lw.q_proj.ptr == 0 || lw.k_proj.ptr == 0 || lw.v_proj.ptr == 0 {
                candle_core::bail!(
                    "fuse_qkv layer {li}: NULL Q/K/V pointer (q={:#x} k={:#x} v={:#x}) — \
                     weight extraction returned a dangling/unfilled pointer",
                    lw.q_proj.ptr,
                    lw.k_proj.ptr,
                    lw.v_proj.ptr,
                );
            }

            let mut ptr: u64 = 0;
            let s = unsafe { cudaMalloc(&mut ptr, total_size) };
            if s != 0 {
                candle_core::bail!(
                    "cudaMalloc QKV fused (layer {li}, {total_size} bytes) failed: {s}"
                );
            }

            unsafe {
                let mut off = 0usize;
                let s = cudaMemcpy(
                    (ptr + off as u64) as *mut _,
                    lw.q_proj.ptr as *const _,
                    qr * k * bf16,
                    3,
                );
                if s != 0 {
                    candle_core::bail!("cudaMemcpy Q→fused (layer {li}) failed: {s}");
                }
                off += qr * k * bf16;
                let s = cudaMemcpy(
                    (ptr + off as u64) as *mut _,
                    lw.k_proj.ptr as *const _,
                    kr * k * bf16,
                    3,
                );
                if s != 0 {
                    candle_core::bail!("cudaMemcpy K→fused (layer {li}) failed: {s}");
                }
                off += kr * k * bf16;
                let s = cudaMemcpy(
                    (ptr + off as u64) as *mut _,
                    lw.v_proj.ptr as *const _,
                    vr * k * bf16,
                    3,
                );
                if s != 0 {
                    candle_core::bail!("cudaMemcpy V→fused (layer {li}) failed: {s}");
                }
            }
            lw.qkv_fused = ptr;
            lw.qkv_rows = total_rows;
            total_bytes += total_size;
        }

        tracing::info!(
            "QKV fused: {} layers, {:.1} GB",
            weights.layers.len(),
            total_bytes as f64 / 1e9
        );
        Ok(())
    }

    /// Ensure the activation buffers can hold at least `batch_size` rows.
    ///
    /// Grow-only, and it MUST actually grow. The previous version returned early
    /// whenever any buffers existed, so every buffer was sized from the first
    /// batch size the process ever decoded and never resized. Since
    /// `decode_forward` then took its row count from the buffers, a server whose
    /// first decode was a single request computed one row forever: every later
    /// sequence in a batch got stale rows, and `wrap_f32_logits` copied
    /// `batch_size * vocab * 4` bytes out of a `1 * vocab * 4` allocation.
    ///
    /// `ensure_staging` immediately below already had the correct shape — free,
    /// invalidate the captured graph, reallocate. This mirrors it.
    fn ensure_buffers(&mut self, batch_size: usize) -> candle_core::Result<()> {
        if let Some(ref b) = self.buffers {
            if b.capacity >= batch_size {
                return Ok(());
            }
        }

        // Growing moves every activation pointer, and a captured graph baked
        // both the old addresses and the old row count. Drop it first, then
        // drain in-flight work before the pointers it references go away.
        if let Some(exec) = self.graph_exec.take() {
            unsafe {
                cuGraphExecDestroy(exec);
            }
            tracing::info!("Decode buffers grown → graph invalidated");
        }
        if let Some(old) = self.buffers.take() {
            unsafe {
                cudaStreamSynchronize(self.stream);
                Self::free_decode_buffers(&old);
            }
        }

        let cfg = &self.weights.config;
        let bf16 = 2usize;
        let bs = batch_size;

        macro_rules! alloc {
            ($size:expr) => {{
                let mut ptr: u64 = 0;
                let s = unsafe { cudaMalloc(&mut ptr, $size) };
                if s != 0 {
                    candle_core::bail!("cudaMalloc failed for decode buffer: {s}");
                }
                ptr
            }};
        }

        let buffers = DecodeBuffers {
            hidden_a: alloc!(bs * cfg.hidden_size * bf16),
            hidden_b: alloc!(bs * cfg.hidden_size * bf16),
            normed: alloc!(bs * cfg.hidden_size * bf16),
            residual: alloc!(bs * cfg.hidden_size * bf16),
            qkv: alloc!(
                bs * (cfg.num_heads * cfg.head_dim + 2 * cfg.num_kv_heads * cfg.head_dim) * bf16
            ),
            q: 0, // set per-layer as alias into qkv
            k: 0,
            v: 0,
            attn_out: alloc!(bs * cfg.num_heads * cfg.head_dim * bf16),
            attn_out_f32: alloc!(bs * cfg.num_heads * cfg.head_dim * 4),
            o_proj_out: alloc!(bs * cfg.hidden_size * bf16),
            gate: alloc!(bs * cfg.intermediate_size * bf16),
            up: alloc!(bs * cfg.intermediate_size * bf16),
            mlp_act: alloc!(bs * cfg.intermediate_size * bf16),
            down_out: alloc!(bs * cfg.hidden_size * bf16),
            logits: alloc!(bs * cfg.vocab_size * bf16),
            logits_f32: alloc!(bs * cfg.vocab_size * 4), // F32 output from LM head
            token_ids: alloc!(bs * 4),
            positions: alloc!(bs * 4),
            cos_table: self.cos_table,
            sin_table: self.sin_table,
            is_neox: self.weights.config.is_gpt_neox,
            capacity: bs,
            sm_count: {
                // Query SM count from the active CUDA device. Used by GEMV
                // dispatch to pick the wide vs original kernel without
                // hardcoding any GPU-specific shape.
                extern "C" {
                    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> u32;
                }
                let mut n: i32 = 0;
                // attr 16 = cudaDevAttrMultiProcessorCount.
                unsafe {
                    cudaDeviceGetAttribute(&mut n, 16, 0);
                }
                if n <= 0 {
                    1
                } else {
                    n
                }
            },
        };

        let total_mb = (bs * cfg.hidden_size * bf16 * 4
            + bs * (cfg.num_heads + cfg.num_kv_heads * 2) * cfg.head_dim * bf16
            + bs * cfg.num_heads * cfg.head_dim * bf16
            + bs * cfg.hidden_size * bf16
            + bs * cfg.intermediate_size * bf16 * 3
            + bs * cfg.hidden_size * bf16
            + bs * cfg.vocab_size * bf16)
            / 1_048_576;

        tracing::info!("Decode buffers allocated: ~{total_mb} MB for capacity={bs} rows");
        self.buffers = Some(buffers);
        Ok(())
    }

    /// Prove the invariant `decode_forward` depends on, rather than trusting it.
    ///
    /// `ensure_buffers` is called immediately before this on both step paths, so
    /// a failure here means the grow path silently did not grow. Returning an
    /// Err makes the engine surface it; the previous code had no check at all
    /// and simply launched kernels for more rows than were allocated.
    fn assert_capacity(buffers: &DecodeBuffers, batch_size: usize) -> candle_core::Result<()> {
        if buffers.capacity < batch_size {
            candle_core::bail!(
                "dedicated decode: buffer capacity {} < batch_size {batch_size} after \
                 ensure_buffers — refusing to launch out of bounds",
                buffers.capacity,
            );
        }
        Ok(())
    }

    /// Free the activation buffers. Deliberately does NOT touch `cos_table` /
    /// `sin_table`: those are owned by the path itself (they are only *copied*
    /// into `DecodeBuffers`) and are freed in `Drop`. Freeing them here would
    /// leave the next `ensure_buffers` handing dangling RoPE tables to every
    /// kernel.
    unsafe fn free_decode_buffers(b: &DecodeBuffers) {
        for ptr in [
            b.hidden_a,
            b.hidden_b,
            b.normed,
            b.residual,
            b.qkv,
            b.attn_out,
            b.attn_out_f32,
            b.o_proj_out,
            b.gate,
            b.up,
            b.mlp_act,
            b.down_out,
            b.logits,
            b.logits_f32,
            b.token_ids,
            b.positions,
        ] {
            if ptr != 0 {
                cudaFree(ptr);
            }
        }
    }

    /// Allocate staging buffers for paged attention metadata.
    /// These are at fixed GPU addresses so the captured graph can replay.
    fn ensure_staging(
        &mut self,
        batch_size: usize,
        max_blocks_per_seq: usize,
    ) -> candle_core::Result<()> {
        if self.staging_max_batch >= batch_size
            && self.staging_max_blocks_per_seq >= max_blocks_per_seq
        {
            return Ok(());
        }
        // Free old staging if resizing — also invalidates captured graph
        if let Some(exec) = self.graph_exec.take() {
            unsafe {
                cuGraphExecDestroy(exec);
            }
            tracing::info!("Staging resize → graph invalidated");
        }
        unsafe {
            if self.staging_block_tables != 0 {
                cudaFree(self.staging_block_tables);
            }
            if self.staging_context_lens != 0 {
                cudaFree(self.staging_context_lens);
            }
            if self.staging_slot_mappings != 0 {
                cudaFree(self.staging_slot_mappings);
            }
        }
        let mut bt: u64 = 0;
        let mut cl: u64 = 0;
        let mut sm: u64 = 0;
        unsafe {
            let s = cudaMalloc(&mut bt, batch_size * max_blocks_per_seq * 4); // i32
            if s != 0 {
                candle_core::bail!("cudaMalloc staging_block_tables failed: {s}");
            }
            let s = cudaMalloc(&mut cl, batch_size * 4); // i32
            if s != 0 {
                candle_core::bail!("cudaMalloc staging_context_lens failed: {s}");
            }
            let s = cudaMalloc(&mut sm, batch_size * 8); // i64
            if s != 0 {
                candle_core::bail!("cudaMalloc staging_slot_mappings failed: {s}");
            }
        }
        self.staging_block_tables = bt;
        self.staging_context_lens = cl;
        self.staging_slot_mappings = sm;
        self.staging_max_batch = batch_size;
        self.staging_max_blocks_per_seq = max_blocks_per_seq;
        tracing::info!(
            "Staging buffers allocated: batch={batch_size}, max_blocks={max_blocks_per_seq}"
        );
        Ok(())
    }

    /// Copy per-step paged attention data from caller's pointers to fixed staging buffers.
    /// D2D async on the dedicated stream — ordered before graph launch.
    /// Uses the ACTUAL per-step sizes, not the staging capacity.
    unsafe fn stage_paged_attn(&self, paged_attn: &PagedAttentionState, batch_size: usize) {
        let actual_blocks = paged_attn.max_num_blocks_per_seq as usize;
        // block_tables: source is [batch, actual_blocks]; the staging buffer is
        // read by the kernel as [batch, staging_max_blocks_per_seq] because
        // `staged_paged_attn` reports the CAPACITY as `max_num_blocks_per_seq`
        // (it must: the captured graph bakes that value, and the actual per-step
        // block count grows with context, which would invalidate the graph).
        //
        // The two strides are different integers, so this MUST be a pitched
        // copy. A single flat copy of `batch * actual_blocks * 4` bytes lays the
        // rows out at the SOURCE stride; the kernel then reads sequence i at
        // `i * capacity` (pagedattention.cuh:229) and, for every i >= 1, picks up
        // uninitialised memory and uses it as a physical KV block index —
        // `key_cache + garbage * kv_block_stride` — i.e. a wild device read.
        // Invisible at batch_size == 1 (both strides agree at offset 0), fatal
        // from batch_size == 2 up.
        let dst_pitch = self.staging_max_blocks_per_seq * 4;
        let src_pitch = actual_blocks * 4;
        cudaMemcpy2DAsync(
            self.staging_block_tables as *mut _,
            dst_pitch,
            paged_attn.block_tables as *const _,
            src_pitch,
            src_pitch, // width in bytes actually carried per row
            batch_size,
            3, // D2D
            self.stream,
        );
        // context_lens: [batch] i32/u32
        cudaMemcpyAsync(
            self.staging_context_lens as *mut _,
            paged_attn.context_lens as *const _,
            batch_size * 4,
            3,
            self.stream,
        );
        // slot_mappings: [batch] i64
        cudaMemcpyAsync(
            self.staging_slot_mappings as *mut _,
            paged_attn.slot_mappings as *const _,
            batch_size * 8,
            3,
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
            max_context_len: {
                if self.cached_is_turbo {
                    // Query DEFAULT per-block smem limit (not opt-in — we don't call cudaFuncSetAttribute)
                    let mut smem: i32 = 0;
                    unsafe {
                        extern "C" {
                            fn cudaDeviceGetAttribute(v: *mut i32, a: i32, d: i32) -> u32;
                        }
                        // attr 8 = cudaDevAttrMaxSharedMemoryPerBlock (default, not opt-in)
                        cudaDeviceGetAttribute(&mut smem, 8, 0);
                        if smem <= 0 {
                            smem = 49152;
                        }
                    }
                    let usable = (smem - 2048).max(16384); // leave 2KB headroom
                    let bs = self.cached_block_size.max(1);
                    ((usable / 4) / bs) * bs
                } else {
                    self.weights.config.max_position_embeddings as i32
                }
            },
            max_num_blocks_per_seq: self.staging_max_blocks_per_seq as i32,
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
    /// Same as run_step but takes a GPU pointer to u32 token IDs instead of a CPU slice.
    /// Avoids the D2H+H2D round-trip when the engine has the tokens already on GPU.
    pub fn run_step_gpu_tokens(
        &mut self,
        token_ids_gpu: u64,
        token_dtype_size: usize,
        batch_size: usize,
        positions: &[i32],
        paged_attn: &PagedAttentionState,
    ) -> candle_core::Result<u64> {
        let t0 = std::time::Instant::now();
        self.ensure_buffers(batch_size)?;
        let max_possible_blocks = (self.weights.config.max_position_embeddings
            / paged_attn.block_size.max(1) as usize)
            .max(paged_attn.max_num_blocks_per_seq as usize);
        self.ensure_staging(batch_size, max_possible_blocks)?;
        self.cache_kv_info(paged_attn);
        let buffers = self.buffers.as_ref().unwrap();
        Self::assert_capacity(buffers, batch_size)?;
        let t1 = std::time::Instant::now();

        unsafe {
            cudaMemcpyAsync(
                buffers.token_ids as *mut _,
                token_ids_gpu as *const _,
                batch_size * token_dtype_size,
                3,
                self.stream,
            );
            cudaMemcpyAsync(
                buffers.positions as *mut _,
                positions.as_ptr() as *const _,
                batch_size * 4,
                1,
                self.stream,
            );
            self.stage_paged_attn(paged_attn, batch_size);
            let staged = self.staged_paged_attn(paged_attn);
            let t2 = std::time::Instant::now();

            if let Some(exec) = self.graph_exec {
                if batch_size != self.captured_batch_size {
                    decode_forward(&self.weights, buffers, &staged, batch_size, self.stream);
                } else {
                    let s = cuGraphLaunch(exec, self.stream);
                    if s != CUDA_SUCCESS {
                        decode_forward(&self.weights, buffers, &staged, batch_size, self.stream);
                    }
                }
            } else {
                decode_forward(&self.weights, buffers, &staged, batch_size, self.stream);
            }
            let t3 = std::time::Instant::now();
            cudaStreamSynchronize(self.stream);
            let t4 = std::time::Instant::now();

            self.step_count += 1;
            if self.step_count > 4 {
                self.sum_setup_us += t1.duration_since(t0).as_secs_f64() * 1e6;
                self.sum_stage_us += t2.duration_since(t1).as_secs_f64() * 1e6;
                self.sum_launch_us += t3.duration_since(t2).as_secs_f64() * 1e6;
                self.sum_sync_us += t4.duration_since(t3).as_secs_f64() * 1e6;
                self.sum_total_us += t4.duration_since(t0).as_secs_f64() * 1e6;
                let n = (self.step_count - 4) as f64;
                if (self.step_count - 4) % 50 == 0 {
                    tracing::info!(
                        "RS_GPU_us avg n={}: setup={:.0} stage={:.0} launch={:.0} sync={:.0} TOTAL={:.0}",
                        self.step_count - 4,
                        self.sum_setup_us / n, self.sum_stage_us / n,
                        self.sum_launch_us / n, self.sum_sync_us / n,
                        self.sum_total_us / n,
                    );
                }
            }
        }
        Ok(self.buffers.as_ref().unwrap().logits_f32)
    }

    pub fn run_step(
        &mut self,
        token_ids: &[i32],
        positions: &[i32],
        paged_attn: &PagedAttentionState,
    ) -> candle_core::Result<u64> {
        let t_start = std::time::Instant::now();
        let batch_size = token_ids.len();
        self.ensure_buffers(batch_size)?;
        // Pre-allocate staging for max possible blocks to avoid reallocation
        // (which would invalidate captured graph pointers)
        let max_possible_blocks = (self.weights.config.max_position_embeddings
            / paged_attn.block_size.max(1) as usize)
            .max(paged_attn.max_num_blocks_per_seq as usize);
        self.ensure_staging(batch_size, max_possible_blocks)?;
        self.cache_kv_info(paged_attn);

        let buffers = self.buffers.as_ref().unwrap();
        Self::assert_capacity(buffers, batch_size)?;
        let t_after_setup = std::time::Instant::now();

        unsafe {
            // Stage all changing inputs (NOT part of the graph — happens before capture/launch)
            cudaMemcpyAsync(
                buffers.token_ids as *mut _,
                token_ids.as_ptr() as *const _,
                batch_size * 4,
                1,
                self.stream,
            );
            cudaMemcpyAsync(
                buffers.positions as *mut _,
                positions.as_ptr() as *const _,
                batch_size * 4,
                1,
                self.stream,
            );
            self.stage_paged_attn(paged_attn, batch_size);
            let t_after_stage = std::time::Instant::now();

            // Use REAL per-step values for max_context_len and max_num_blocks_per_seq.
            // These determine shared memory allocation in attention kernels — using
            // max_position_embeddings would request 160KB+ smem and silently fail.
            let staged = self.staged_paged_attn(paged_attn);

            if let Some(exec) = self.graph_exec {
                // REPLAY: all inputs staged, just launch
                if batch_size != self.captured_batch_size {
                    tracing::warn!(
                        "Batch size changed ({} → {}), running eager",
                        self.captured_batch_size,
                        batch_size
                    );
                    decode_forward(&self.weights, buffers, &staged, batch_size, self.stream);
                } else if self.step_count > 0 && self.step_count % 5000 == 0 {
                    // Periodic per-kernel profile so we can see steady-state hotspots.
                    // profile_forward is decode_forward with CUDA event timing — same compute,
                    // same KV-cache state advancement.
                    crate::decode_forward::profile_forward(
                        &self.weights,
                        buffers,
                        &staged,
                        batch_size,
                        self.stream,
                    );
                } else {
                    let s = cuGraphLaunch(exec, self.stream);
                    if s != CUDA_SUCCESS {
                        tracing::warn!("cuGraphLaunch failed ({s}), falling back to eager");
                        decode_forward(&self.weights, buffers, &staged, batch_size, self.stream);
                    }
                }
            } else if self.capture_failed {
                // Eager mode — still fast (no Candle overhead, just kernel launch costs)
                decode_forward(&self.weights, buffers, &staged, batch_size, self.stream);
            } else if self.eager_steps < 2 {
                // Profile on first eager step
                if self.eager_steps == 1 {
                    crate::decode_forward::profile_forward(
                        &self.weights,
                        buffers,
                        &staged,
                        batch_size,
                        self.stream,
                    );
                }
                self.eager_steps += 1;
                decode_forward(&self.weights, buffers, &staged, batch_size, self.stream);
                if self.eager_steps == 2 {
                    tracing::info!(
                        "Dedicated decode: eager warmup done, will attempt capture next step"
                    );
                }
            } else {
                // CAPTURE: record the forward pass into a CUDA graph
                tracing::info!("Capturing CUDA graph for batch_size={batch_size}...");

                let s = cuStreamBeginCapture_v2(self.stream, CUstreamCaptureMode::THREAD_LOCAL);
                if s != CUDA_SUCCESS {
                    tracing::warn!("cuStreamBeginCapture failed ({s}), disabling capture");
                    self.capture_failed = true;
                    decode_forward(&self.weights, buffers, &staged, batch_size, self.stream);
                } else {
                    decode_forward(&self.weights, buffers, &staged, batch_size, self.stream);

                    let mut graph: CUgraph = std::ptr::null_mut();
                    let s = cuStreamEndCapture(self.stream, &mut graph);
                    if s != CUDA_SUCCESS || graph.is_null() {
                        tracing::warn!("cuStreamEndCapture failed ({s}), disabling capture");
                        if !graph.is_null() {
                            cuGraphDestroy(graph);
                        }
                        self.capture_failed = true;
                        decode_forward(&self.weights, buffers, &staged, batch_size, self.stream);
                    } else {
                        let mut exec: CUgraphExec = std::ptr::null_mut();
                        let s = cuGraphInstantiate_v2(
                            &mut exec,
                            graph,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            0,
                        );
                        cuGraphDestroy(graph);

                        if s != CUDA_SUCCESS || exec.is_null() {
                            tracing::warn!("cuGraphInstantiate failed ({s}), disabling capture");
                            self.capture_failed = true;
                            decode_forward(
                                &self.weights,
                                buffers,
                                &staged,
                                batch_size,
                                self.stream,
                            );
                        } else {
                            let s = cuGraphLaunch(exec, self.stream);
                            if s != CUDA_SUCCESS {
                                tracing::warn!(
                                    "First cuGraphLaunch failed ({s}), disabling capture"
                                );
                                cuGraphExecDestroy(exec);
                                self.capture_failed = true;
                                decode_forward(
                                    &self.weights,
                                    buffers,
                                    &staged,
                                    batch_size,
                                    self.stream,
                                );
                            } else {
                                self.graph_exec = Some(exec);
                                self.captured_batch_size = batch_size;
                                tracing::info!(
                                    "CUDA graph captured and launched for batch_size={batch_size}"
                                );
                            }
                        }
                    }
                }
            }

            let t_after_launch = std::time::Instant::now();
            cudaStreamSynchronize(self.stream);
            let t_after_sync = std::time::Instant::now();

            // Accumulate phase timings (skip first 4 steps for warmup/capture noise)
            self.step_count += 1;
            if self.step_count > 4 {
                let setup_us = t_after_setup.duration_since(t_start).as_secs_f64() * 1e6;
                let stage_us = t_after_stage.duration_since(t_after_setup).as_secs_f64() * 1e6;
                let launch_us = t_after_launch.duration_since(t_after_stage).as_secs_f64() * 1e6;
                let sync_us = t_after_sync.duration_since(t_after_launch).as_secs_f64() * 1e6;
                let total_us = t_after_sync.duration_since(t_start).as_secs_f64() * 1e6;
                self.sum_setup_us += setup_us;
                self.sum_stage_us += stage_us;
                self.sum_launch_us += launch_us;
                self.sum_sync_us += sync_us;
                self.sum_total_us += total_us;

                let n = (self.step_count - 4) as f64;
                if (self.step_count - 4) % 25 == 0 {
                    tracing::info!(
                        "RUN_STEP_us avg over {} steps: setup={:.1} stage={:.1} launch={:.1} sync={:.1} TOTAL={:.1} (=> {:.1} tok/s)",
                        n as u64,
                        self.sum_setup_us / n,
                        self.sum_stage_us / n,
                        self.sum_launch_us / n,
                        self.sum_sync_us / n,
                        self.sum_total_us / n,
                        1e6 / (self.sum_total_us / n),
                    );
                }
            }
        }

        Ok(buffers.logits_f32) // F32 directly from LM head GEMV — no cast needed
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

    /// Shared accessor for the model weights. Used by AutonomousDecodeRunner
    /// wiring (it needs the same `ModelWeights` to call `decode_forward`).
    pub fn weights(&self) -> &crate::weights::ModelWeights {
        &self.weights
    }

    /// Ensure activation buffers are allocated for the given batch size, then
    /// return an immutable reference. The autonomous capture closure needs
    /// stable pointers, so the buffers must already exist before capture.
    pub fn ensure_and_get_buffers(
        &mut self,
        batch_size: usize,
    ) -> candle_core::Result<&crate::decode_forward::DecodeBuffers> {
        self.ensure_buffers(batch_size)?;
        Ok(self
            .buffers
            .as_ref()
            .expect("ensure_buffers populates buffers"))
    }

    /// Build a staged `PagedAttentionState` from the dedicated path's cached
    /// KV info + the current step's paged attention state. Pointers come from
    /// staging buffers — stable across captures provided staging hasn't been
    /// resized.
    pub fn build_paged_attn_state(
        &mut self,
        paged_attn: &crate::decode_forward::PagedAttentionState,
        batch_size: usize,
    ) -> candle_core::Result<crate::decode_forward::PagedAttentionState> {
        let max_possible_blocks = (self.weights.config.max_position_embeddings
            / paged_attn.block_size.max(1) as usize)
            .max(paged_attn.max_num_blocks_per_seq as usize);
        self.ensure_staging(batch_size, max_possible_blocks)?;
        self.cache_kv_info(paged_attn);
        // SAFETY: stage_paged_attn launches D2D memcpys on the dedicated path's
        // own stream — these are NOT captured into the autonomous graph because
        // we run them outside the capture region. Inside capture, the runner's
        // own input_buffers are used.
        unsafe {
            self.stage_paged_attn(paged_attn, batch_size);
        }
        Ok(self.staged_paged_attn(paged_attn))
    }
}

#[cfg(feature = "cuda")]
impl Drop for DedicatedDecodePath {
    fn drop(&mut self) {
        unsafe {
            if let Some(exec) = self.graph_exec {
                cuGraphExecDestroy(exec);
            }
            if self.cos_table != 0 {
                cudaFree(self.cos_table);
            }
            if self.sin_table != 0 {
                cudaFree(self.sin_table);
            }
            for lw in &self.weights.layers {
                if lw.qkv_fused != 0 {
                    cudaFree(lw.qkv_fused);
                }
            }
            if self.staging_block_tables != 0 {
                cudaFree(self.staging_block_tables);
            }
            if self.staging_context_lens != 0 {
                cudaFree(self.staging_context_lens);
            }
            if self.staging_slot_mappings != 0 {
                cudaFree(self.staging_slot_mappings);
            }
            if let Some(ref b) = self.buffers {
                // Same list `ensure_buffers` uses when it grows. Sharing the
                // helper keeps the two from drifting apart — a buffer added to
                // one list and not the other leaks or double-frees.
                Self::free_decode_buffers(b);
            }
            cuStreamDestroy_v2(self.stream);
        }
    }
}

/// Regression guard for the block-table staging stride.
///
/// `stage_paged_attn` writes the staging buffer and `paged_attention_v1` reads
/// it. The kernel indexes sequence `i` at `i * max_num_blocks_per_seq`
/// (`pagedattention.cuh:229`), and `staged_paged_attn` reports the staging
/// CAPACITY there — it must, because a captured graph bakes that value while the
/// real per-step block count grows with context.
///
/// So the copy MUST be pitched. These tests run without CUDA: they model the two
/// layouts as index arithmetic and pin the fact that a flat copy mislays every
/// row after the first. If someone "simplifies" `cudaMemcpy2DAsync` back to a
/// single flat `cudaMemcpyAsync`, the first test here states exactly what breaks.
#[cfg(test)]
mod block_table_stride_tests {
    /// Where a FLAT copy of `batch * actual * 4` bytes actually puts row `i`.
    fn flat_copy_row_offset(seq_idx: usize, actual_blocks: usize) -> usize {
        seq_idx * actual_blocks
    }

    /// Where the kernel LOOKS for row `i` (pagedattention.cuh:229), given the
    /// value `staged_paged_attn` reports as `max_num_blocks_per_seq`.
    fn kernel_row_offset(seq_idx: usize, reported_max_blocks: usize) -> usize {
        seq_idx * reported_max_blocks
    }

    /// Where a PITCHED copy with `dpitch = capacity` puts row `i`.
    fn pitched_copy_row_offset(seq_idx: usize, capacity: usize) -> usize {
        seq_idx * capacity
    }

    /// Realistic staging capacity: `(max_position_embeddings / block_size)
    /// .max(actual_blocks)` — see `run_step`.
    fn capacity(max_position_embeddings: usize, block_size: usize, actual: usize) -> usize {
        (max_position_embeddings / block_size.max(1)).max(actual)
    }

    #[test]
    fn flat_copy_mislays_every_row_after_the_first() {
        // Qwen2.5-0.5B shaped: 32k positions, 32-token blocks, short context.
        let cap = capacity(32768, 32, 2);
        assert_eq!(
            cap, 1024,
            "capacity is derived from max_position_embeddings"
        );

        // Sequence 0 always agrees — which is precisely why batch_size == 1
        // cannot observe this defect and the B=8 cell can.
        assert_eq!(flat_copy_row_offset(0, 2), kernel_row_offset(0, cap));

        // Every later sequence does not.
        for seq_idx in 1..8 {
            assert_ne!(
                flat_copy_row_offset(seq_idx, 2),
                kernel_row_offset(seq_idx, cap),
                "seq {seq_idx}: a flat copy writes at {} but the kernel reads at {} \
                 — the read lands in uninitialised memory and is used as a \
                 physical KV block index",
                flat_copy_row_offset(seq_idx, 2),
                kernel_row_offset(seq_idx, cap),
            );
        }
    }

    #[test]
    fn pitched_copy_agrees_with_the_kernel_for_every_sequence() {
        let cap = capacity(32768, 32, 2);
        for seq_idx in 0..8 {
            assert_eq!(
                pitched_copy_row_offset(seq_idx, cap),
                kernel_row_offset(seq_idx, cap),
                "the destination pitch must be the same number the kernel is told"
            );
        }
    }

    #[test]
    fn batch_one_cannot_discriminate() {
        // Stated as a test so no one reads a green B=1 run as evidence.
        let cap = capacity(32768, 32, 2);
        assert_eq!(
            flat_copy_row_offset(0, 2),
            kernel_row_offset(0, cap),
            "at batch_size == 1 the buggy and fixed layouts are byte-identical; \
             a passing B=1 measurement says nothing about this defect"
        );
    }

    /// The buffer-capacity policy, as arithmetic. `ensure_buffers` used to
    /// return early whenever buffers existed, so `alloc` was the FIRST batch
    /// size forever and every later row was out of bounds.
    fn grow_only(current: Option<usize>, requested: usize) -> usize {
        match current {
            Some(c) if c >= requested => c,
            _ => requested,
        }
    }

    /// What the OLD code did: allocate once, never revisit.
    fn pin_forever(current: Option<usize>, requested: usize) -> usize {
        current.unwrap_or(requested)
    }

    #[test]
    fn old_policy_pins_capacity_below_the_served_batch() {
        // A server whose first decode is a single request, then a batch of 8.
        let mut cap = None;
        for b in [1usize, 1, 8, 32] {
            cap = Some(pin_forever(cap, b));
        }
        assert_eq!(cap, Some(1), "old policy never grew past the first batch");

        // logits_f32 was sized from that capacity while wrap_f32_logits copied
        // the REAL batch. Qwen2.5-0.5B vocab.
        let vocab = 151_936usize;
        let pinned_capacity = cap.unwrap();
        let served_batch = 8usize;
        let allocated = pinned_capacity * vocab * 4;
        let copied = served_batch * vocab * 4;
        assert!(copied > allocated);
        assert_eq!(
            copied - allocated,
            4_254_208,
            "the over-read at B=8 with capacity pinned to 1"
        );
    }

    #[test]
    fn grow_only_policy_always_covers_the_served_batch() {
        let mut cap: Option<usize> = None;
        for b in [1usize, 1, 8, 32, 4, 32, 2] {
            cap = Some(grow_only(cap, b));
            assert!(
                cap.unwrap() >= b,
                "capacity {cap:?} must cover every served batch size, including {b}"
            );
        }
        // Grow-only: a later smaller batch must NOT shrink and re-alloc, which
        // would thrash the captured graph on every scheduler dip.
        assert_eq!(cap, Some(32));
    }

    #[test]
    fn growth_is_monotonic_so_the_graph_is_invalidated_at_most_once_per_high_water_mark() {
        let mut cap: Option<usize> = None;
        let mut reallocs = 0;
        for b in [1usize, 2, 2, 8, 4, 8, 8, 32, 1] {
            let before = cap;
            cap = Some(grow_only(cap, b));
            if before != cap {
                reallocs += 1;
            }
        }
        // 1 -> 2 -> 8 -> 32: four allocations for nine steps, not nine.
        assert_eq!(reallocs, 4);
    }

    #[test]
    fn pitched_copy_preconditions_hold() {
        // cudaMemcpy2DAsync requires dpitch >= width and spitch >= width.
        for &(max_pos, block_size, actual) in &[
            (32768usize, 32usize, 2usize),
            (4096, 16, 1),
            (1024, 16, 64),
            (128, 16, 64),
        ] {
            let cap = capacity(max_pos, block_size, actual);
            let width = actual * 4;
            let dpitch = cap * 4;
            let spitch = actual * 4;
            assert!(
                dpitch >= width,
                "capacity {cap} must cover actual {actual} (max() in run_step guarantees it)"
            );
            assert!(spitch >= width);
        }
    }
}
