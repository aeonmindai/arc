//! Arc GPU-Autonomous Decode
//!
//! Dedicated decode path that bypasses Candle entirely.
//! Model loads normally through Candle (NULL stream).
//! Decode runs on a non-blocking stream with cuBLASLt + custom kernels.
//! Capturable by CUDA graphs.

pub mod ffi;
pub mod graph;
pub mod buffers;
pub mod autonomous;
pub mod weights;
pub mod decode_forward;
pub mod dedicated;
pub mod gemv_ffi;
pub mod nongemv_ffi;

/// FlashMLASparse — V4 Lightning Indexer scoring + radix top-k.
///
/// Vendored from MIT-licensed sgl-project/FlashMLA and sgl-project/sglang
/// (see `src/cuda/flashmlasparse/LICENSE-MIT`). Provides a CPU reference
/// (always available) and a CUDA dispatch (under `feature="cuda"`).
pub mod flashmlasparse;
#[cfg(feature = "cuda")]
pub mod flashmlasparse_ffi;
/// CPU reference sampler — bit-exact spec for the on-device GPU sampler.
/// Tier A: this module + tests, runs without CUDA.
/// Tier B: GPU sampler in autonomous.rs matches this output for the same RNG seed.
pub mod sampling_cpu;

/// CUDA-backed sampler. The kernel in `cuda/sampling_kernel.cu` matches
/// `sampling_cpu` bit-for-bit for any input with unique probabilities.
/// See module docs for the documented tied-probability divergence. The
/// module compiles host-only on CPU builds (kernel call sites and the
/// `CudaSampler` struct are CUDA-gated within the module) so we can
/// unit-test the algorithm without a GPU.
pub mod sampling_cuda;

pub use sampling_cuda::{gpu_algorithm_simulate, SamplingParams, GPU_MAX_KEEP};

#[cfg(feature = "cuda")]
pub use sampling_cuda::CudaSampler;

#[cfg(feature = "cuda")]
pub use graph::CudaGraphRunner;
#[cfg(feature = "cuda")]
pub use buffers::{DecodeInputBuffers, DecodeState};
#[cfg(feature = "cuda")]
pub use autonomous::{AutonomousDecodeConfig, AutonomousDecodeRunner};
#[cfg(feature = "cuda")]
pub use weights::{ModelWeights, DecodeConfig, LayerWeights, WeightPtr, WeightAnchors, LayerAnchors, tensor_device_ptr, extract_model_weights, quant_method_ptr};
#[cfg(feature = "cuda")]
pub use decode_forward::{DecodeBuffers, PagedAttentionState, LayerKvCache, decode_forward};
#[cfg(feature = "cuda")]
pub use dedicated::DedicatedDecodePath;

/// Try to create a CUDA graph runner for the given device.
#[cfg(feature = "cuda")]
pub fn try_init_graph_runner(device: &candle_core::Device) -> Option<CudaGraphRunner> {
    match CudaGraphRunner::new(device, 2) {
        Ok(runner) => {
            tracing::info!("CUDA graph runner initialized");
            Some(runner)
        }
        Err(e) => {
            tracing::warn!("CUDA graph runner unavailable: {e}");
            None
        }
    }
}
