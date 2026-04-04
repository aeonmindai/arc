//! Model-agnostic CUDA graph capture and replay for decode steps.
//!
//! Creates a dedicated non-blocking stream for graph capture.
//! The legacy default stream (NULL/stream 0) does NOT support capture —
//! that's what Candle uses by default via `CudaDevice::new()`.

#[cfg(feature = "cuda")]
use crate::ffi::*;
#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::sys::CUstream;
#[cfg(feature = "cuda")]
use candle_core::Device;

/// Model-agnostic CUDA graph runner for decode acceleration.
#[cfg(feature = "cuda")]
pub struct CudaGraphRunner {
    /// The device's actual stream (where Candle runs kernels)
    device_stream: CUstream,
    /// Our dedicated non-blocking stream for graph capture
    capture_stream: CUstream,
    /// Whether we own capture_stream and need to destroy it
    owns_capture_stream: bool,
    graph_exec: Option<CUgraphExec>,
    enabled: bool,
    warmup_remaining: u32,
    capturing: bool,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaGraphRunner {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaGraphRunner {}

#[cfg(feature = "cuda")]
impl CudaGraphRunner {
    pub fn new(device: &Device, warmup_steps: u32) -> candle_core::Result<Self> {
        let Device::Cuda(cuda_dev) = device else {
            candle_core::bail!("CudaGraphRunner requires a CUDA device");
        };
        let device_stream = cuda_dev.cuda_stream().cu_stream();

        // Check if the device stream is the legacy NULL stream
        let (capture_stream, owns) = if device_stream.is_null() {
            // Legacy default stream — cannot be captured.
            // Create a dedicated non-blocking stream for capture.
            let mut new_stream: CUstream = std::ptr::null_mut();
            let status = unsafe { cuStreamCreate(&mut new_stream, CU_STREAM_NON_BLOCKING) };
            if status != CUDA_SUCCESS {
                candle_core::bail!("cuStreamCreate for graph capture failed: CUDA error {status}");
            }
            tracing::info!(
                "CUDA graph: device uses legacy NULL stream, created dedicated capture stream"
            );
            (new_stream, true)
        } else {
            // Device already has a non-default stream — use it directly
            tracing::info!("CUDA graph: device has non-default stream, using it for capture");
            (device_stream, false)
        };

        Ok(Self {
            device_stream,
            capture_stream,
            owns_capture_stream: owns,
            graph_exec: None,
            enabled: true,
            warmup_remaining: warmup_steps,
            capturing: false,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled && self.warmup_remaining == 0
    }

    pub fn tick_warmup(&mut self) -> bool {
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            if self.warmup_remaining == 0 {
                tracing::info!("CUDA graph: warmup complete, enabling capture");
            }
            true
        } else {
            false
        }
    }

    /// Begin capturing GPU operations.
    ///
    /// IMPORTANT: This captures on our dedicated non-blocking stream.
    /// Candle operations run on the device's default stream. For the capture
    /// to work, we need the forward pass kernels to run on our capture stream.
    /// This is achieved by having the engine synchronize the device stream
    /// and then all subsequent operations go to our stream.
    ///
    /// However, since Candle internally uses the device stream for all ops,
    /// we capture on the DEVICE stream if it supports it. If the device stream
    /// is NULL (legacy), we need a different approach — see below.
    pub fn begin_capture(&mut self) -> candle_core::Result<()> {
        if self.capturing {
            candle_core::bail!("Already capturing — cannot nest captures");
        }

        // If we're using a dedicated capture stream (because device stream is NULL),
        // we can't capture Candle operations since they go to the NULL stream.
        // In this case, graph capture doesn't work with the default Candle setup.
        // We need the device to use new_cuda_with_stream() instead.
        if self.owns_capture_stream {
            // The device uses the legacy NULL stream. Candle operations go there.
            // Our capture stream is different — Candle won't launch kernels on it.
            // We CANNOT capture Candle's forward pass on our stream.
            //
            // The fix must happen at the device level: use Device::new_cuda_with_stream()
            // instead of Device::new_cuda(). This creates a real non-blocking stream
            // that supports capture.
            //
            // For now, disable graph capture and log a clear message.
            tracing::warn!(
                "CUDA graph: device uses legacy NULL stream which cannot be captured. \
                 To enable CUDA graphs, the device must be created with \
                 Device::new_cuda_with_stream() instead of Device::new_cuda(). \
                 Disabling graph capture."
            );
            self.enabled = false;
            candle_core::bail!(
                "CUDA graph capture requires a non-legacy stream. \
                 Device must use new_cuda_with_stream()."
            );
        }

        unsafe {
            cudaStreamSynchronize(self.capture_stream);

            let status = cuStreamBeginCapture_v2(
                self.capture_stream,
                CUstreamCaptureMode::THREAD_LOCAL,
            );
            if status != CUDA_SUCCESS {
                self.enabled = false;
                candle_core::bail!("cuStreamBeginCapture failed: CUDA error {status}");
            }
        }
        self.capturing = true;
        Ok(())
    }

    pub fn end_capture_and_launch(&mut self) -> candle_core::Result<()> {
        if !self.capturing {
            candle_core::bail!("Not capturing — call begin_capture() first");
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

        if let Some(exec) = self.graph_exec {
            let mut update_result = CUgraphExecUpdateResult::SUCCESS;
            let status = unsafe { cuGraphExecUpdate_v2(exec, graph, &mut update_result) };

            if status != CUDA_SUCCESS
                || update_result as u32 != CUgraphExecUpdateResult::SUCCESS as u32
            {
                tracing::debug!("CUDA graph: topology changed, re-instantiating");
                unsafe { cuGraphExecDestroy(exec); }
                self.graph_exec = None;
                self.instantiate_and_launch(graph)?;
            } else {
                unsafe { cuGraphDestroy(graph); }
                unsafe {
                    let status = cuGraphLaunch(exec, self.capture_stream);
                    if status != CUDA_SUCCESS {
                        self.enabled = false;
                        candle_core::bail!("cuGraphLaunch failed: CUDA error {status}");
                    }
                }
            }
        } else {
            self.instantiate_and_launch(graph)?;
        }

        Ok(())
    }

    pub fn cancel_capture(&mut self) {
        if self.capturing {
            self.capturing = false;
            let mut graph: CUgraph = std::ptr::null_mut();
            unsafe {
                let _ = cuStreamEndCapture(self.capture_stream, &mut graph);
                if !graph.is_null() {
                    cuGraphDestroy(graph);
                }
            }
        }
    }

    fn instantiate_and_launch(&mut self, graph: CUgraph) -> candle_core::Result<()> {
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let status = unsafe {
            cuGraphInstantiate_v2(
                &mut exec, graph,
                std::ptr::null_mut(), std::ptr::null_mut(), 0,
            )
        };
        unsafe { cuGraphDestroy(graph); }

        if status != CUDA_SUCCESS {
            self.enabled = false;
            candle_core::bail!("cuGraphInstantiate failed: CUDA error {status}");
        }

        tracing::info!("CUDA graph: instantiated successfully");
        self.graph_exec = Some(exec);

        unsafe {
            let status = cuGraphLaunch(exec, self.capture_stream);
            if status != CUDA_SUCCESS {
                self.enabled = false;
                candle_core::bail!("cuGraphLaunch failed: CUDA error {status}");
            }
        }

        Ok(())
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaGraphRunner {
    fn drop(&mut self) {
        self.cancel_capture();
        if let Some(exec) = self.graph_exec {
            unsafe { cuGraphExecDestroy(exec); }
        }
        if self.owns_capture_stream && !self.capture_stream.is_null() {
            unsafe { cuStreamDestroy_v2(self.capture_stream); }
        }
    }
}
