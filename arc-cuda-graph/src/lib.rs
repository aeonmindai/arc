//! Arc GPU-Autonomous Decode
//!
//! Zero CPU overhead during decode. The GPU runs the entire generation loop —
//! forward pass → sampling → EOS check → position increment → loop —
//! via a CUDA 12.4 conditional WHILE graph node.
//!
//! One `cuGraphLaunch` generates an entire response.

pub mod ffi;
pub mod graph;
pub mod buffers;
pub mod autonomous;

#[cfg(feature = "cuda")]
pub use graph::CudaGraphRunner;

#[cfg(feature = "cuda")]
pub use buffers::{DecodeInputBuffers, DecodeState};

#[cfg(feature = "cuda")]
pub use autonomous::{AutonomousDecodeConfig, AutonomousDecodeRunner};

/// Try to create a CUDA graph runner for the given device.
/// Returns None if the device is not CUDA.
#[cfg(feature = "cuda")]
pub fn try_init_graph_runner(device: &candle_core::Device) -> Option<CudaGraphRunner> {
    match CudaGraphRunner::new(device, 2) {
        Ok(runner) => {
            tracing::info!("CUDA graph runner initialized for decode acceleration");
            Some(runner)
        }
        Err(e) => {
            tracing::warn!("CUDA graph runner unavailable: {e}");
            None
        }
    }
}
