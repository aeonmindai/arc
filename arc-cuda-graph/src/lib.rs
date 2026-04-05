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

#[cfg(feature = "cuda")]
pub use graph::CudaGraphRunner;
#[cfg(feature = "cuda")]
pub use buffers::{DecodeInputBuffers, DecodeState};
#[cfg(feature = "cuda")]
pub use autonomous::{AutonomousDecodeConfig, AutonomousDecodeRunner};
#[cfg(feature = "cuda")]
pub use weights::{ModelWeights, DecodeConfig, LayerWeights, WeightPtr, tensor_device_ptr, extract_model_weights, quant_method_ptr};
#[cfg(feature = "cuda")]
pub use decode_forward::{DecodeBuffers, CublasState, decode_forward};

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
