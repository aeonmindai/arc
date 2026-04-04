//! Model-agnostic CUDA graph capture and replay for decode steps.
//!
//! # How it works
//!
//! On every decode step:
//! 1. `begin_capture()` — starts recording GPU kernels (not executed)
//! 2. Run the normal forward pass — CPU code runs, GPU kernels are captured
//! 3. `end_capture_and_launch()` — builds/updates graph, launches all kernels as one batch
//!
//! Total overhead: ~50-100μs vs 3-7ms of individual kernel launches.
//! No model-specific code. Works with any Candle forward pass.

#[cfg(feature = "cuda")]
use crate::ffi::*;
#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::sys::CUstream;
#[cfg(feature = "cuda")]
use candle_core::Device;

/// Model-agnostic CUDA graph runner for decode acceleration.
///
/// The API is split into `begin_capture()` / `end_capture_and_launch()` to
/// avoid holding a mutable borrow during the forward pass.
#[cfg(feature = "cuda")]
pub struct CudaGraphRunner {
    stream: CUstream,
    /// Cached graph executable (reused across steps via cuGraphExecUpdate)
    graph_exec: Option<CUgraphExec>,
    enabled: bool,
    /// Warmup steps remaining before enabling capture
    warmup_remaining: u32,
    /// Whether we're currently inside a capture
    capturing: bool,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaGraphRunner {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaGraphRunner {}

#[cfg(feature = "cuda")]
impl CudaGraphRunner {
    /// Create a new graph runner.
    ///
    /// `warmup_steps`: eager decode steps before enabling graph capture (cuBLAS stabilization).
    pub fn new(device: &Device, warmup_steps: u32) -> candle_core::Result<Self> {
        let Device::Cuda(cuda_dev) = device else {
            candle_core::bail!("CudaGraphRunner requires a CUDA device");
        };
        let stream = cuda_dev.cuda_stream().cu_stream();

        Ok(Self {
            stream,
            graph_exec: None,
            enabled: true,
            warmup_remaining: warmup_steps,
            capturing: false,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled && self.warmup_remaining == 0
    }

    /// Check if still in warmup phase. Decrements warmup counter.
    /// Returns true if we should run eagerly (warmup not done).
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

    /// Begin capturing GPU operations on the stream.
    ///
    /// After this call, all GPU kernel launches on the device stream are
    /// recorded but NOT executed. Call `end_capture_and_launch()` after
    /// the forward pass to build the graph and execute it.
    pub fn begin_capture(&mut self) -> candle_core::Result<()> {
        if self.capturing {
            candle_core::bail!("Already capturing — cannot nest captures");
        }
        unsafe {
            // Synchronize stream before capture — pending operations prevent capture
            cudaStreamSynchronize(self.stream);

            let status = cuStreamBeginCapture_v2(
                self.stream,
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

    /// End capture, update/instantiate the graph executable, and launch it.
    ///
    /// This must be called after `begin_capture()` and the forward pass.
    /// All captured GPU kernels are executed as a single graph launch.
    pub fn end_capture_and_launch(&mut self) -> candle_core::Result<()> {
        if !self.capturing {
            candle_core::bail!("Not capturing — call begin_capture() first");
        }
        self.capturing = false;

        // End capture → get graph template
        let mut graph: CUgraph = std::ptr::null_mut();
        unsafe {
            let status = cuStreamEndCapture(self.stream, &mut graph);
            if status != CUDA_SUCCESS {
                self.enabled = false;
                candle_core::bail!("cuStreamEndCapture failed: CUDA error {status}");
            }
        }

        // Update existing graph exec or instantiate new one
        if let Some(exec) = self.graph_exec {
            let mut update_result = CUgraphExecUpdateResult::SUCCESS;
            let status = unsafe { cuGraphExecUpdate_v2(exec, graph, &mut update_result) };

            if status != CUDA_SUCCESS
                || update_result as u32 != CUgraphExecUpdateResult::SUCCESS as u32
            {
                // Topology changed — full re-instantiate
                tracing::debug!("CUDA graph: topology changed, re-instantiating");
                unsafe { cuGraphExecDestroy(exec); }
                self.graph_exec = None;
                self.instantiate_and_launch(graph)?;
            } else {
                unsafe { cuGraphDestroy(graph); }
                // Launch with updated exec
                unsafe {
                    let status = cuGraphLaunch(exec, self.stream);
                    if status != CUDA_SUCCESS {
                        self.enabled = false;
                        candle_core::bail!("cuGraphLaunch failed: CUDA error {status}");
                    }
                }
            }
        } else {
            // First capture — instantiate
            self.instantiate_and_launch(graph)?;
        }

        Ok(())
    }

    /// Cancel an in-progress capture (e.g., if forward_inputs failed).
    pub fn cancel_capture(&mut self) {
        if self.capturing {
            self.capturing = false;
            // End the capture to return stream to normal mode
            let mut graph: CUgraph = std::ptr::null_mut();
            unsafe {
                let _ = cuStreamEndCapture(self.stream, &mut graph);
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
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        unsafe { cuGraphDestroy(graph); }

        if status != CUDA_SUCCESS {
            self.enabled = false;
            candle_core::bail!("cuGraphInstantiate failed: CUDA error {status}");
        }

        tracing::info!("CUDA graph: instantiated");
        self.graph_exec = Some(exec);

        // Launch
        unsafe {
            let status = cuGraphLaunch(exec, self.stream);
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
    }
}
