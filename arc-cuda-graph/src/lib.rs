//! Arc GPU-Autonomous Decode
//!
//! **Parent system: ArcInfer/ArcGraph** (see `memory/mission/TAXONOMY.md`).
//! Its kernels are owned by ArcKernels/Decode.
//!
//! Dedicated decode path that bypasses Candle entirely.
//! Model loads normally through Candle (NULL stream).
//! Decode runs on a non-blocking stream with cuBLASLt + custom kernels.
//! Capturable by CUDA graphs.

pub mod autonomous;
pub mod buffers;
pub mod decode_forward;
pub mod dedicated;
pub mod ffi;
pub mod gemv_ffi;
pub mod graph;
pub mod nongemv_ffi;
pub mod weights;

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
pub use autonomous::{AutonomousDecodeConfig, AutonomousDecodeRunner};
#[cfg(feature = "cuda")]
pub use buffers::{DecodeInputBuffers, DecodeState};
#[cfg(feature = "cuda")]
pub use decode_forward::{decode_forward, DecodeBuffers, LayerKvCache, PagedAttentionState};
#[cfg(feature = "cuda")]
pub use dedicated::DedicatedDecodePath;
#[cfg(feature = "cuda")]
pub use graph::CudaGraphRunner;
#[cfg(feature = "cuda")]
pub use weights::{
    extract_model_weights, quant_method_ptr, tensor_device_ptr, DecodeConfig, LayerAnchors,
    LayerWeights, ModelWeights, WeightAnchors, WeightPtr,
};

/// Try to create a CUDA graph runner for the given device.
#[cfg(feature = "cuda")]
pub fn try_init_graph_runner(device: &candle_core::Device) -> Option<CudaGraphRunner> {
    // 4 eager warmup decode steps: with the candle caching allocator on, these
    // populate the cache so the captured forward is allocation-free (RUN-161).
    match CudaGraphRunner::new(device, 4) {
        Ok(runner) => {
            // D18. The old line here was `info!("CUDA graph runner initialized")`,
            // emitted unconditionally — including immediately after the runner
            // had logged that capture was disabled and it would do nothing. A
            // success message that a dead subsystem also prints is worse than
            // no message: it reads as confirmation. Report the state, not the
            // constructor's return.
            if runner.capture_possible() {
                tracing::info!(
                    "ArcGraph: runner initialized on a capturable stream — capture WILL be \
                     attempted after warmup. {}",
                    runner.status_line()
                );
            } else {
                tracing::warn!(
                    "ArcGraph: runner initialized but INERT — it will not capture or replay \
                     anything (see the preceding line for why). {}",
                    runner.status_line()
                );
            }
            Some(runner)
        }
        Err(e) => {
            tracing::warn!("ArcGraph: runner unavailable, decode runs eagerly: {e}");
            None
        }
    }
}

/// Try to create an autonomous decode runner (uncaptured) for the given device.
///
/// The returned runner has buffers + decode state allocated but its CUDA graph
/// has NOT been captured. Capture happens lazily on the first decode call via
/// `AutonomousDecodeRunner::capture(forward_fn)` once a real forward closure
/// is available. Until then, the runner is a stub: `run_decode_loop()` will
/// return an error and the caller MUST fall back to step-by-step decode.
///
/// Returns `None` (with a `warn` log) if the device is not CUDA or allocation
/// fails. Callers should treat this as "autonomous decode unavailable" and
/// continue with the standard step-by-step decode path — matching the
/// fall-back contract of `try_init_graph_runner`.
#[cfg(feature = "cuda")]
pub fn try_init_autonomous_runner(
    device: &candle_core::Device,
    config: AutonomousDecodeConfig,
) -> Option<AutonomousDecodeRunner> {
    if !matches!(device, candle_core::Device::Cuda(_)) {
        tracing::warn!("Autonomous decode runner unavailable: device is not CUDA");
        return None;
    }
    match AutonomousDecodeRunner::new(config, device) {
        Ok(runner) => {
            tracing::info!(
                "Autonomous decode runner allocated (graph capture deferred until first decode)"
            );
            Some(runner)
        }
        Err(e) => {
            tracing::warn!("Autonomous decode runner unavailable: {e}");
            None
        }
    }
}
