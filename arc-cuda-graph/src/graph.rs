//! CUDA graph capture-once with dedicated capture stream.
//!
//! The device uses the legacy NULL stream (which cannot be captured).
//! We create a dedicated non-blocking stream and capture on it.
//!
//! For capture: we make the capture stream wait on the device stream,
//! then capture the forward pass. Candle's operations go to the NULL stream,
//! but with GLOBAL capture mode, operations on ANY stream in the process
//! are captured into the graph.
//!
//! For replay: we launch the graph on our capture stream, then make the
//! device stream wait on it.

#[cfg(feature = "cuda")]
use crate::ffi::*;
#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::sys::CUstream;
#[cfg(feature = "cuda")]
use candle_core::{Device, Tensor};
#[cfg(feature = "cuda")]
use std::collections::HashMap;

#[cfg(feature = "cuda")]
struct CapturedGraph {
    exec: CUgraphExec,
    output: Tensor,
}

#[cfg(feature = "cuda")]
impl Drop for CapturedGraph {
    fn drop(&mut self) {
        unsafe { cuGraphExecDestroy(self.exec); }
    }
}

#[cfg(feature = "cuda")]
pub struct CudaGraphRunner {
    /// Our dedicated non-blocking capture stream
    capture_stream: CUstream,
    graphs: HashMap<usize, CapturedGraph>,
    enabled: bool,
    warmup_remaining: u32,
    capturing: bool,
    capture_batch_size: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaGraphRunner {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaGraphRunner {}

#[cfg(feature = "cuda")]
impl CudaGraphRunner {
    pub fn new(device: &Device, warmup_steps: u32) -> candle_core::Result<Self> {
        let Device::Cuda(_cuda_dev) = device else {
            candle_core::bail!("CudaGraphRunner requires a CUDA device");
        };

        // Create a dedicated non-blocking stream for graph capture.
        // The device's NULL stream can't be captured, but with GLOBAL capture mode
        // we can capture operations that the NULL stream dispatches.
        let mut capture_stream: CUstream = std::ptr::null_mut();
        let status = unsafe { cuStreamCreate(&mut capture_stream, CU_STREAM_NON_BLOCKING) };
        if status != CUDA_SUCCESS {
            candle_core::bail!("cuStreamCreate for graph capture failed: CUDA error {status}");
        }

        tracing::info!("CUDA graph: created dedicated capture stream");

        Ok(Self {
            capture_stream,
            graphs: HashMap::new(),
            enabled: true,
            warmup_remaining: warmup_steps,
            capturing: false,
            capture_batch_size: 0,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled && self.warmup_remaining == 0
    }

    pub fn tick_warmup(&mut self) -> bool {
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            if self.warmup_remaining == 0 {
                tracing::info!("CUDA graph: warmup done, capture enabled");
            }
            true
        } else {
            false
        }
    }

    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.graphs.contains_key(&batch_size)
    }

    pub fn replay(&self, batch_size: usize) -> candle_core::Result<Tensor> {
        let captured = self.graphs.get(&batch_size).ok_or_else(|| {
            candle_core::Error::Msg(format!("No graph for batch_size={batch_size}"))
        })?;
        unsafe {
            // Launch on our capture stream
            let status = cuGraphLaunch(captured.exec, self.capture_stream);
            if status != CUDA_SUCCESS {
                candle_core::bail!("cuGraphLaunch failed: CUDA error {status}");
            }
            cudaStreamSynchronize(self.capture_stream);
        }
        Ok(captured.output.clone())
    }

    /// Begin graph capture with GLOBAL mode.
    /// GLOBAL mode captures operations on ALL streams in the process,
    /// including the legacy NULL stream that Candle uses.
    pub fn begin_capture(&mut self, batch_size: usize) -> candle_core::Result<()> {
        if self.capturing {
            candle_core::bail!("Already capturing");
        }

        unsafe {
            // Sync our stream first
            cudaStreamSynchronize(self.capture_stream);

            // Begin capture on our non-blocking stream with GLOBAL mode.
            // GLOBAL mode: any thread's operations on any stream (including NULL)
            // that are ordered after this point are captured.
            let status = cuStreamBeginCapture_v2(
                self.capture_stream,
                CUstreamCaptureMode::GLOBAL,
            );
            if status != CUDA_SUCCESS {
                self.enabled = false;
                candle_core::bail!("cuStreamBeginCapture failed: CUDA error {status}");
            }
        }

        self.capturing = true;
        self.capture_batch_size = batch_size;
        Ok(())
    }

    pub fn end_capture_and_cache(&mut self, output: Tensor) -> candle_core::Result<Tensor> {
        if !self.capturing {
            candle_core::bail!("Not capturing");
        }
        self.capturing = false;

        let mut graph: CUgraph = std::ptr::null_mut();
        unsafe {
            let status = cuStreamEndCapture(self.capture_stream, &mut graph);
            if status != CUDA_SUCCESS {
                self.enabled = false;
                candle_core::bail!("cuStreamEndCapture failed: CUDA error {status}");
            }
        }

        let mut exec: CUgraphExec = std::ptr::null_mut();
        unsafe {
            let status = cuGraphInstantiate_v2(
                &mut exec, graph,
                std::ptr::null_mut(), std::ptr::null_mut(), 0,
            );
            cuGraphDestroy(graph);
            if status != CUDA_SUCCESS {
                self.enabled = false;
                candle_core::bail!("cuGraphInstantiate failed: CUDA error {status}");
            }
        }

        // First launch
        unsafe {
            let status = cuGraphLaunch(exec, self.capture_stream);
            if status != CUDA_SUCCESS {
                cuGraphExecDestroy(exec);
                self.enabled = false;
                candle_core::bail!("cuGraphLaunch failed: CUDA error {status}");
            }
            cudaStreamSynchronize(self.capture_stream);
        }

        tracing::info!("CUDA graph: captured+launched for batch_size={}", self.capture_batch_size);

        let result = output.clone();
        self.graphs.insert(self.capture_batch_size, CapturedGraph { exec, output });
        Ok(result)
    }

    pub fn cancel_capture(&mut self) {
        if self.capturing {
            self.capturing = false;
            let mut graph: CUgraph = std::ptr::null_mut();
            unsafe {
                let _ = cuStreamEndCapture(self.capture_stream, &mut graph);
                if !graph.is_null() { cuGraphDestroy(graph); }
            }
        }
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaGraphRunner {
    fn drop(&mut self) {
        self.cancel_capture();
        if !self.capture_stream.is_null() {
            unsafe { cuStreamDestroy_v2(self.capture_stream); }
        }
    }
}
