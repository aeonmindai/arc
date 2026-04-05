//! Capture-once CUDA graph with split API to avoid borrow conflicts.
//!
//! First decode at a given batch size: capture → instantiate → launch.
//! Subsequent decodes at same batch size: replay the cached graph.

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
    /// Output tensor from capture run. Graph writes here on every replay.
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
    stream: CUstream,
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
        let Device::Cuda(cuda_dev) = device else {
            candle_core::bail!("CudaGraphRunner requires a CUDA device");
        };
        let stream = cuda_dev.cuda_stream().cu_stream();

        if stream.is_null() {
            return Ok(Self {
                stream, graphs: HashMap::new(),
                enabled: false, warmup_remaining: 0,
                capturing: false, capture_batch_size: 0,
            });
        }

        Ok(Self {
            stream, graphs: HashMap::new(),
            enabled: true, warmup_remaining: warmup_steps,
            capturing: false, capture_batch_size: 0,
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

    /// Replay the cached graph for a batch size. Returns the output logits.
    /// The graph writes new output values to the cached tensor's address.
    pub fn replay(&self, batch_size: usize) -> candle_core::Result<Tensor> {
        let captured = self.graphs.get(&batch_size).ok_or_else(|| {
            candle_core::Error::Msg(format!("No graph for batch_size={batch_size}"))
        })?;
        unsafe {
            let status = cuGraphLaunch(captured.exec, self.stream);
            if status != CUDA_SUCCESS {
                candle_core::bail!("cuGraphLaunch failed: CUDA error {status}");
            }
            cudaStreamSynchronize(self.stream);
        }
        Ok(captured.output.clone())
    }

    /// Begin capturing a graph for the given batch size.
    /// Call forward_inputs() after this, then end_capture().
    pub fn begin_capture(&mut self, batch_size: usize) -> candle_core::Result<()> {
        if self.capturing {
            candle_core::bail!("Already capturing");
        }
        unsafe {
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
        self.capture_batch_size = batch_size;
        Ok(())
    }

    /// End capture, instantiate, launch, and cache the graph.
    /// `output` is the logits tensor from the forward pass.
    pub fn end_capture_and_cache(&mut self, output: Tensor) -> candle_core::Result<Tensor> {
        if !self.capturing {
            candle_core::bail!("Not capturing");
        }
        self.capturing = false;

        let mut graph: CUgraph = std::ptr::null_mut();
        unsafe {
            let status = cuStreamEndCapture(self.stream, &mut graph);
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

        // Launch the graph (executes the captured forward for the first time)
        unsafe {
            let status = cuGraphLaunch(exec, self.stream);
            if status != CUDA_SUCCESS {
                cuGraphExecDestroy(exec);
                self.enabled = false;
                candle_core::bail!("cuGraphLaunch failed: CUDA error {status}");
            }
            cudaStreamSynchronize(self.stream);
        }

        tracing::info!("CUDA graph: captured, instantiated, launched for batch_size={}", self.capture_batch_size);

        let result = output.clone();
        self.graphs.insert(self.capture_batch_size, CapturedGraph { exec, output });
        Ok(result)
    }

    /// Cancel an in-progress capture.
    pub fn cancel_capture(&mut self) {
        if self.capturing {
            self.capturing = false;
            let mut graph: CUgraph = std::ptr::null_mut();
            unsafe {
                let _ = cuStreamEndCapture(self.stream, &mut graph);
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
    }
}
