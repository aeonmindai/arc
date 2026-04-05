//! Dedicated decode path manager.
//!
//! Creates a non-blocking stream, allocates buffers, initializes cuBLASLt,
//! and runs decode_forward() with CUDA graph capture.

#[cfg(feature = "cuda")]
use crate::decode_forward::{CublasState, DecodeBuffers, decode_forward};
#[cfg(feature = "cuda")]
use crate::ffi::*;
#[cfg(feature = "cuda")]
use crate::weights::ModelWeights;
#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::sys::CUstream;
#[cfg(feature = "cuda")]
use candle_core::Device;

#[cfg(feature = "cuda")]
extern "C" {
    fn cublasLtCreate(handle: *mut *mut std::ffi::c_void) -> u32;
    fn cublasLtDestroy(handle: *mut std::ffi::c_void) -> u32;
}

/// The dedicated decode path: owns its own non-blocking stream, buffers, and cuBLAS.
#[cfg(feature = "cuda")]
pub struct DedicatedDecodePath {
    pub weights: ModelWeights,
    stream: CUstream,
    cublas: CublasState,
    buffers: Option<DecodeBuffers>, // Allocated lazily on first decode
    graph_exec: Option<CUgraphExec>,
    graph_pool: Option<CUmemoryPool>,
    enabled: bool,
    warmup_remaining: u32,
}

#[cfg(feature = "cuda")]
unsafe impl Send for DedicatedDecodePath {}
#[cfg(feature = "cuda")]
unsafe impl Sync for DedicatedDecodePath {}

#[cfg(feature = "cuda")]
impl DedicatedDecodePath {
    pub fn new(weights: ModelWeights) -> candle_core::Result<Self> {
        // Create non-blocking stream for the decode path
        let mut stream: CUstream = std::ptr::null_mut();
        let s = unsafe { cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING) };
        if s != CUDA_SUCCESS {
            candle_core::bail!("Failed to create decode stream: {s}");
        }

        // Create cuBLASLt handle
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let s = unsafe { cublasLtCreate(&mut handle) };
        if s != 0 {
            unsafe { cuStreamDestroy_v2(stream); }
            candle_core::bail!("cublasLtCreate failed: {s}");
        }

        // Allocate cuBLASLt workspace (32MB for Hopper+)
        // We allocate on the non-blocking stream
        let workspace_size = 33_554_432usize;
        let mut workspace_ptr: u64 = 0;
        // Use cudaMalloc for the workspace (not stream-ordered, it's permanent)
        extern "C" {
            fn cudaMalloc(ptr: *mut u64, size: usize) -> u32;
        }
        let s = unsafe { cudaMalloc(&mut workspace_ptr, workspace_size) };
        if s != 0 {
            unsafe { cublasLtDestroy(handle); cuStreamDestroy_v2(stream); }
            candle_core::bail!("cudaMalloc workspace failed: {s}");
        }

        tracing::info!("Dedicated decode path initialized (stream + cuBLASLt + 32MB workspace)");

        Ok(Self {
            weights,
            stream,
            cublas: CublasState { handle, workspace: workspace_ptr, workspace_size },
            buffers: None,
            graph_exec: None,
            graph_pool: None,
            enabled: true,
            warmup_remaining: 2,
        })
    }

    /// Allocate decode buffers for a given batch size.
    fn ensure_buffers(&mut self, batch_size: usize) -> candle_core::Result<()> {
        if self.buffers.is_some() {
            return Ok(());
        }

        let cfg = &self.weights.config;
        let bf16 = 2usize;
        let bs = batch_size;

        // Allocate all buffers via cudaMalloc (permanent, not stream-ordered)
        extern "C" {
            fn cudaMalloc(ptr: *mut u64, size: usize) -> u32;
        }

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
            o_proj_out: alloc!(bs * cfg.hidden_size * bf16),
            gate: alloc!(bs * cfg.intermediate_size * bf16),
            up: alloc!(bs * cfg.intermediate_size * bf16),
            mlp_act: alloc!(bs * cfg.intermediate_size * bf16),
            down_out: alloc!(bs * cfg.hidden_size * bf16),
            logits: alloc!(bs * cfg.vocab_size * bf16),
            token_ids: alloc!(bs * 4), // i32
            positions: alloc!(bs * 4), // i32
            cos_table: 0, // TODO: extract from RoPE
            sin_table: 0,
            is_neox: true,
            batch_size: bs,
        };

        let total_mb = (bs * cfg.hidden_size * bf16 * 4  // hidden_a/b, normed, residual
            + bs * (cfg.num_heads + cfg.num_kv_heads * 2) * cfg.head_dim * bf16  // q, k, v
            + bs * cfg.num_heads * cfg.head_dim * bf16  // attn_out
            + bs * cfg.hidden_size * bf16  // o_proj_out
            + bs * cfg.intermediate_size * bf16 * 3  // gate, up, mlp_act
            + bs * cfg.hidden_size * bf16  // down_out
            + bs * cfg.vocab_size * bf16  // logits
        ) / 1_048_576;

        tracing::info!("Decode buffers allocated: ~{total_mb} MB for batch_size={bs}");
        self.buffers = Some(buffers);
        Ok(())
    }

    /// Run one decode step. Returns logits pointer (BF16).
    pub fn run_step(&mut self, token_ids: &[i32], positions: &[i32]) -> candle_core::Result<u64> {
        let batch_size = token_ids.len();
        self.ensure_buffers(batch_size)?;
        let buffers = self.buffers.as_ref().unwrap();

        // Copy inputs to decode buffers
        unsafe {
            extern "C" {
                fn cudaMemcpyAsync(
                    dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                    count: usize, kind: u32, stream: CUstream,
                ) -> u32;
            }
            cudaMemcpyAsync(
                buffers.token_ids as *mut _, token_ids.as_ptr() as *const _,
                batch_size * 4, 1, // H2D
                self.stream,
            );
            cudaMemcpyAsync(
                buffers.positions as *mut _, positions.as_ptr() as *const _,
                batch_size * 4, 1,
                self.stream,
            );
        }

        // Run the decode forward pass on our non-blocking stream
        unsafe {
            decode_forward(&self.weights, buffers, &self.cublas, self.stream);
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
            if let Some(pool) = self.graph_pool {
                cuMemPoolDestroy(pool);
            }
            cublasLtDestroy(self.cublas.handle);
            extern "C" { fn cudaFree(ptr: u64) -> u32; }
            cudaFree(self.cublas.workspace);
            if let Some(ref b) = self.buffers {
                for ptr in [b.hidden_a, b.hidden_b, b.normed, b.residual,
                    b.q, b.k, b.v, b.attn_out, b.o_proj_out,
                    b.gate, b.up, b.mlp_act, b.down_out, b.logits,
                    b.token_ids, b.positions] {
                    if ptr != 0 { cudaFree(ptr); }
                }
            }
            cuStreamDestroy_v2(self.stream);
        }
    }
}
